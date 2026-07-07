// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_prepacked_rocket.c — isolate the pack-weights-once (v2a) path.
 *
 * Runs C[M,N] = A[M,K] * B[N,K]^T via the prepacked API (rocket_ctx_create +
 * rocket_weights_pack + rocket_matmul_fp16_prepacked) AND via the validated
 * rocket_matmul_fp16_mt path, and verifies BOTH against a CPU fp32 reference.
 *
 * NOTE: this standalone test PASSES every case (all Gemma shapes incl. K=15360,
 * M=4..16, W=400 resident weights, free+re-pack churn incl. across MAX_TILE,
 * DUMMY_GB memory pressure, DATA=1 realistic full-range data) -- and that is
 * CORRECT: the driver prepacked path is sound. A whole-model "gibberish" failure mode
 * belongs to the BACKEND, not this path: keying a weight cache on src0->data is unsafe
 * because ggml-backend-sched copies weights into a pooled, address-reused buffer, so one
 * address can back many logical weights and the cache would serve the wrong resident
 * weight (correct math on the wrong weights). This test cannot exhibit that -- it has no
 * scheduler pool, so each weight gets a distinct, stable address. A backend avoids it by
 * keying the cache on the weight NAME + supports_buft=is_host (so sched stops copying
 * weights). This gate covers the driver prepacked path itself.
 * Args: M K N [T] [W].  Env: DATA, DUMMY_GB, WARMUP_M, ROCKET_NO_SHARED_PACK.
 *   sudo ./matmul_prepacked_rocket 16 3840 15360
 *   sudo ./matmul_prepacked_rocket 4 3840 15360 4 1
 *
 * Build: added to the test foreach in CMakeLists.txt.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_matmul.h"

static void verify(const char *tag, const _Float16 *C, const float *ref,
                   int M, int N, int ref_rows)
{
    (void)M;   /* kept for call-site symmetry; the row count is ref_rows */
    float max_abs = 0, max_rel = 0;
    for (int m = 0; m < ref_rows; m++) for (int n = 0; n < N; n++) {
        float got = (float)C[(size_t)m*N+n], want = ref[(size_t)m*N+n];
        float ad = fabsf(got-want), rd = ad/(fabsf(want)+1e-6f);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
    }
    printf("  %-12s max_abs=%.3f max_rel=%.4f -> %s\n", tag, max_abs, max_rel,
           (max_abs < 0.5f || max_rel < 0.05f) ? "PASS" : "FAIL");
}

int main(int argc, char **argv)
{
    int M = argc>=4 ? atoi(argv[1]) : 16;
    int K = argc>=4 ? atoi(argv[2]) : 3840;
    int N = argc>=4 ? atoi(argv[3]) : 3840;
    int T = argc>=5 ? atoi(argv[4]) : 4;
    int W = argc>=6 ? atoi(argv[5]) : 1;   // # distinct resident weights on one ctx
    printf("matmul C[%d,%d] = A[%d,%d] x B[%d,%d]^T  (T=%d, W=%d resident weights)\n",
           M, N, M, K, N, K, T, W);

    /* DUMMY_GB=N: leak+touch N GiB to simulate the 24GB model's memory pressure,
     * which the prepacked path must coexist with (its resident weight BOs sit on
     * TOP of the model). This is the one thing the standalone test otherwise lacks. */
    const char *dg = getenv("DUMMY_GB");
    if (dg) {
        size_t g = (size_t)atoi(dg);
        char *d = malloc(g << 30);
        if (d) { memset(d, 0xAB, g << 30); printf("touched %zu GiB dummy (memory pressure)\n", g); }
        else   printf("DUMMY_GB=%zu malloc FAILED\n", g);
    }

    _Float16 *A = malloc((size_t)M*K*sizeof(_Float16));
    _Float16 *C = malloc((size_t)M*N*sizeof(_Float16));
    float *ref  = malloc((size_t)M*N*sizeof(float));
    /* Realistic scaled-activation-like data: full [-1,1] range (DATA=1, default)
     * vs the old tiny sawtooth (DATA=0). LCG pseudo-random spanning the fp16 range. */
    int realistic = getenv("DATA") ? atoi(getenv("DATA")) : 1;
    if (realistic) {
        uint32_t st=2463534242u;
        for (size_t i=0;i<(size_t)M*K;i++){ st^=st<<13; st^=st>>17; st^=st<<5; A[i]=(_Float16)((st/2147483648.0f)-1.0f); }
    } else {
        for (size_t i=0;i<(size_t)M*K;i++) A[i]=(_Float16)(((i*7)%13-6)*0.05f);
    }

    /* Each resident weight w gets DISTINCT content (seeded by w) so an aliased /
     * wrong-BO read shows up as a verify FAIL. Hash-based, full fp16 [-1,1] range
     * (realistic) instead of a tiny sawtooth — deterministic per (i,seed). */
    #define BVAL(i, seed) (_Float16)((float)(((uint32_t)((i)*2654435761u+(seed)*40503u))>>16 & 0xFFFF)/32768.0f - 1.0f)

    /* single-weight mt reference baseline (w=0 content) */
    _Float16 *B0 = malloc((size_t)N*K*sizeof(_Float16));
    for (size_t i=0;i<(size_t)N*K;i++) B0[i]=BVAL(i,0);
    int ref_rows = M <= 64 ? M : 64;
    for (int m=0;m<ref_rows;m++) for (int n=0;n<N;n++){
        float a=0; for (int k=0;k<K;k++) a+=(float)A[(size_t)m*K+k]*(float)B0[(size_t)n*K+k];
        ref[(size_t)m*N+n]=a;
    }
    for (size_t i=0;i<(size_t)M*N;i++) C[i]=(_Float16)-99.0f;
    if (rocket_matmul_fp16_mt(M,K,N,A,B0,C,T)) { fprintf(stderr,"mt failed\n"); return 1; }
    verify("mt(w=0)", C, ref, M, N, ref_rows);
    free(B0);

    /* Pack W distinct weights resident on ONE ctx (as the ggml backend does for a
     * whole model), THEN verify each — replicates resident-BO/IOVA accumulation. */
    rocket_ctx *ctx = rocket_ctx_create(T);
    if (!ctx) { fprintf(stderr,"ctx_create failed\n"); return 1; }

    rocket_weights **ws = calloc(W, sizeof(*ws));
    _Float16 *B = malloc((size_t)N*K*sizeof(_Float16));
    for (int w=0; w<W; w++) {
        for (size_t i=0;i<(size_t)N*K;i++) B[i]=BVAL(i,w);
        ws[w] = rocket_weights_pack(ctx, M, K, N, B);
        if (!ws[w]) { fprintf(stderr,"weights_pack FAILED at w=%d (of %d)\n", w, W); W = w; break; }
    }
    printf("packed %d resident weights\n", W);

    int fails = 0;
    for (int w=0; w<W; w++) {
        for (size_t i=0;i<(size_t)N*K;i++) B[i]=BVAL(i,w);   // reuse B buffer for ref
        for (int m=0;m<ref_rows;m++) for (int n=0;n<N;n++){
            float a=0; for (int k=0;k<K;k++) a+=(float)A[(size_t)m*K+k]*(float)B[(size_t)n*K+k];
            ref[(size_t)m*N+n]=a;
        }
        for (size_t i=0;i<(size_t)M*N;i++) C[i]=(_Float16)-99.0f;
        if (rocket_matmul_fp16_prepacked(ctx,M,K,N,A,C,ws[w])) { fprintf(stderr,"prepacked w=%d failed\n",w); return 1; }
        float max_abs=0,max_rel=0;
        for (int m=0;m<ref_rows;m++) for (int n=0;n<N;n++){
            float got=(float)C[(size_t)m*N+n], want=ref[(size_t)m*N+n];
            float ad=fabsf(got-want), rd=ad/(fabsf(want)+1e-6f);
            if (ad>max_abs) max_abs=ad;
            if (rd>max_rel) max_rel=rd;
        }
        int ok = (max_abs<0.5f||max_rel<0.05f);
        if (!ok) { fails++;
            if (fails<=8) printf("  w=%-3d max_abs=%.3f max_rel=%.4f -> FAIL\n", w, max_abs, max_rel);
        }
    }
    printf("prepacked verify: %d/%d FAILED\n", fails, W);

    for (int w=0; w<W; w++) rocket_weights_free(ctx, ws[w]);

    /* Free + re-pack CHURN — what the ggml backend does when a weight's
     * M changes (warmup M != prefill M). The reproducer never hit this; the real
     * run does. Pack at a different (warmup-like) M, free, re-pack at M, compute. */
    printf("=== free+re-pack churn (warmup M != prefill M) ===\n");
    for (size_t i=0;i<(size_t)N*K;i++) B[i]=BVAL(i,0);
    for (int m=0;m<ref_rows;m++) for (int n=0;n<N;n++){
        float a=0; for (int k=0;k<K;k++) a+=(float)A[(size_t)m*K+k]*(float)B[(size_t)n*K+k];
        ref[(size_t)m*N+n]=a;
    }
    /* warmup M: llama.cpp warms up at n_ubatch (512) tokens, crossing the MAX_TILE
     * (256) boundary -> a different tile plan than the M=4 prefill. WARMUP_M overrides. */
    int Mw = getenv("WARMUP_M") ? atoi(getenv("WARMUP_M")) : M + 16;
    for (int cyc=0; cyc<3; cyc++) {
        rocket_weights *warm = rocket_weights_pack(ctx, Mw, K, N, B);   /* pack at Mw */
        if (warm) rocket_weights_free(ctx, warm);                       /* then free  */
        rocket_weights *real = rocket_weights_pack(ctx, M, K, N, B);    /* re-pack at M */
        if (!real) { fprintf(stderr,"re-pack failed cyc=%d\n",cyc); break; }
        for (size_t i=0;i<(size_t)M*N;i++) C[i]=(_Float16)-99.0f;
        rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, real);
        char tag[24]; snprintf(tag,sizeof tag,"repack#%d",cyc);
        verify(tag, C, ref, M, N, ref_rows);
        rocket_weights_free(ctx, real);
    }

    rocket_ctx_free(ctx);
    free(ws); free(A); free(B); free(C); free(ref);
    return fails ? 2 : 0;
}
