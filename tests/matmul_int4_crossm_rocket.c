// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int4_crossm_rocket.c — cross-M reuse gate for the RESIDENT group-wise int4
 * path: a weight packed at ONE M is REUSED at other M (incl. a small short-prompt M)
 * with no re-pack, and stays bit-exact vs the one-shot oracle at each M.
 *
 * The resident int4 weight scatter depends only on the N-split + Nt/Kt tiling, which
 * the canonical-tileM plan (rk4_worker_alloc plans at MAX_TILE, not the actual M) makes
 * M-INDEPENDENT. So rocket_matmul_int4_prepacked_gw at a different M than pack must (a)
 * NOT return -2 (the re-pack signal) and (b) match rocket_matmul_int4_groupwise at that
 * M to fp32 rounding. This is the guard for the warmup-M-serves-prefill-M fix (the
 * short-prompt re-pack stall: warmup M=512 then a small-M prefill no longer re-packs the
 * whole int4 model). Pack at M=PACKM, compute at 512/256/768/64/8 against the SAME weight.
 *
 * CTest target matmul_int4_crossm_rocket; skip-code 2 off-device.
 *   sudo ./matmul_int4_crossm_rocket            # K=3840 N=256 group=128, pack M=512
 *   sudo ./matmul_int4_crossm_rocket 7680 512 128 512
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int8_t rand_i4(void)    { return (int8_t)(rand() % 15 - 7); }          /* [-7,7] */
static float  rand_scale(void) { return 0.5f + (rand() % 1000) / 1000.0f; }   /* [0.5,1.5) */

/* fp64 reference: per-group scaled accumulation. */
static void ref_gw(int M, int K, int N, int group, const int8_t *A, const int8_t *B,
                   const float *as, const float *bs, double *ref) {
    const int nG = K / group;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int g = 0; g < nG; g++) {
                long ip = 0;
                for (int k = 0; k < group; k++)
                    ip += (long)A[(size_t)m*K + (size_t)g*group + k] * (long)B[(size_t)n*K + (size_t)g*group + k];
                acc += (double)as[(size_t)m*nG + g] * (double)bs[(size_t)n*nG + g] * (double)ip;
            }
            ref[(size_t)m*N + n] = acc;
        }
}

static int cmp_ref(const char *tag, const float *got, const double *ref, size_t n) {
    float max_abs = 0, max_rel = 0; long nbad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) { nbad++; continue; }
        double r = ref[i];
        float ad = (float)fabs((double)got[i] - r);
        float rd = ad / (float)(fabs(r) + 1e-6);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (rd > 1e-4f && ad > 1e-2f) nbad++;
    }
    printf("  [%s] vs fp64: max_abs=%.4f max_rel=%.2e nbad=%ld -> %s\n",
           tag, max_abs, max_rel, nbad, nbad ? "FAIL" : "PASS");
    return nbad ? -1 : 0;
}

/* resident vs one-shot at the SAME M: bit-exact (identical NPU compute + accum order). */
static int cmp_oneshot(const char *tag, const float *got, const float *ref, size_t n) {
    float max_abs = 0; long nbad = 0;
    for (size_t i = 0; i < n; i++) {
        float ad = fabsf(got[i] - ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad != 0.0f && nbad < 6) printf("  [%s] [%zu] oneshot=%.6f resident=%.6f\n", tag, i, ref[i], got[i]);
        if (ad != 0.0f) nbad++;
    }
    printf("  [%s] vs one-shot: max_abs=%.6f nbad=%ld -> %s\n",
           tag, max_abs, nbad, nbad ? "FAIL" : "PASS (bit-exact)");
    return nbad ? -1 : 0;
}

static int oneshot(int M, int K, int N, const int8_t *A, const int8_t *B,
                   const float *as, const float *bs, float *Cf, int group) {
    int fd = rocket_open();
    if (fd < 0) return fd;
    int rc = rocket_matmul_int4_groupwise(fd, M, K, N, A, B, as, bs, Cf, group);
    rocket_close(fd);
    return rc;
}

int main(int argc, char **argv) {
    int K = 3840, N = 256, group = 128, PACKM = 512;
    if (argc >= 4) { K = atoi(argv[1]); N = atoi(argv[2]); group = atoi(argv[3]); }
    if (argc >= 5) PACKM = atoi(argv[4]);
    if (K % 32 || N % 64 || K % group || group % 32 || 49 * group >= 32767 || PACKM % 4) {
        fprintf(stderr, "bad shape: need K%%32, N%%64, K%%group, group%%32, 49*group<32767, PACKM%%4\n");
        return -1;
    }
    const int nG = K / group;
    int Ms[] = { 512, 256, 768, 64, 8 };   /* compute these against the weight packed at PACKM */
    const int nMs = (int)(sizeof(Ms) / sizeof(Ms[0]));
    int Mmax = PACKM;
    for (int i = 0; i < nMs; i++) if (Ms[i] > Mmax) Mmax = Ms[i];

    printf("cross-M resident GROUPWISE int4: pack M=%d, K=%d N=%d group=%d (nG=%d), reuse at",
           PACKM, K, N, group, nG);
    for (int i = 0; i < nMs; i++) printf(" %d", Ms[i]);
    printf("\n");

    const char *te = getenv("ROCKET_N_THREADS");
    int nthreads = te ? atoi(te) : 5;

    /* Allocate A/as for the LARGEST M; sub-M runs use the leading Mmax rows' prefix. */
    int8_t *A  = malloc((size_t)Mmax * K);
    float  *as = malloc((size_t)Mmax * nG * sizeof(float));
    int8_t *B  = malloc((size_t)N * K);
    float  *bs = malloc((size_t)N * nG * sizeof(float));
    float  *Co = malloc((size_t)Mmax * N * sizeof(float));
    float  *Cr = malloc((size_t)Mmax * N * sizeof(float));
    double *ref = malloc((size_t)Mmax * N * sizeof(double));
    if (!A || !as || !B || !bs || !Co || !Cr || !ref) { fprintf(stderr, "alloc\n"); return -1; }
    srand(20260627);
    for (size_t i = 0; i < (size_t)Mmax * K;  i++) A[i]  = rand_i4();
    for (size_t i = 0; i < (size_t)Mmax * nG; i++) as[i] = rand_scale();
    for (size_t i = 0; i < (size_t)N * K;  i++) B[i]  = rand_i4();
    for (size_t i = 0; i < (size_t)N * nG; i++) bs[i] = rand_scale();

    rocket_i4_ctx *ctx = rocket_i4_ctx_create(nthreads);
    if (!ctx) { fprintf(stderr, "ctx create failed — no NPU, skipping\n"); return 2; }

    /* Pack the resident weight ONCE at PACKM. */
    rocket_i4_weights *rw = rocket_i4_weights_pack_gw(ctx, PACKM, K, N, B, group);
    if (!rw) { fprintf(stderr, "pack_gw failed\n"); return -1; }

    int fails = 0;
    for (int i = 0; i < nMs; i++) {
        int M = Ms[i];
        char tag[24]; snprintf(tag, sizeof(tag), "M=%d", M);
        /* one-shot oracle at this M (uses the leading M rows of A/as). */
        if (oneshot(M, K, N, A, B, as, bs, Co, group)) { fprintf(stderr, "oneshot %s failed\n", tag); return -1; }
        ref_gw(M, K, N, group, A, B, as, bs, ref);
        memset(Cr, 0, (size_t)M * N * sizeof(float));
        int rc = rocket_matmul_int4_prepacked_gw(ctx, M, K, N, A, as, bs, Cr, rw);
        if (rc == -2) { printf("  [%s] RE-PACK SIGNALLED (-2) — weight not reused -> FAIL\n", tag); fails++; continue; }
        if (rc)       { fprintf(stderr, "  [%s] prepacked_gw failed (%d)\n", tag, rc); return -1; }
        if (cmp_oneshot(tag, Cr, Co, (size_t)M * N)) fails++;
        if (cmp_ref(tag, Cr, ref, (size_t)M * N))    fails++;
    }

    rocket_i4_weights_free(ctx, rw);
    rocket_i4_ctx_free(ctx);
    printf("\n==> %s (%d checks failed)\n", fails ? "FAIL" : "ALL PASS", fails);
    free(A); free(as); free(B); free(bs); free(Co); free(Cr); free(ref);
    return fails ? -1 : 0;
}
