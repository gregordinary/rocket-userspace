// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int4_prepacked_gw_rocket.c — the RESIDENT group-wise int4 path
 * (rocket_i4_weights_pack_gw / rocket_matmul_int4_prepacked_gw) must match the
 * validated one-shot rocket_matmul_int4_groupwise oracle, must track an fp64
 * reference, and must not alias across distinct resident weights sharing one ctx.
 *
 * The resident path fans N across worker fds and packs the weight into resident NPU
 * BOs once; the one-shot is single-fd and re-packs B every call. Both force the
 * K-tile to `group`, dequant each tile's int16 partial by a_scale[m,g]*b_scale[n,g],
 * and fp32-accumulate across groups — so the per-output-element K-accumulation order
 * is identical and the two agree to fp32 rounding (≈bit-exact). This is the in-model
 * W4A4 resident path (the backend bakes Hadamard into B + rotates A; here B/A are raw
 * int4 in [-7,7] and the scales random — it isolates the matmul, not quant quality).
 *
 * Usage: matmul_int4_prepacked_gw_rocket [M K N group [W]]  (default 256 3840 256 128, W=3)
 *   M%4, K%32, N%64, K%group, group%32, 49*group < 32767.
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

/* fp64 reference: per-group scaled accumulation (the same metric as the one-shot gate). */
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

/* Compare a result against an fp64 reference (fp32-accum tolerance). */
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

/* Compare resident vs one-shot (expect bit-exact: identical NPU compute + accum order). */
static int cmp_oneshot(const char *tag, const float *got, const float *ref, size_t n) {
    float max_abs = 0; long nbad = 0;
    for (size_t i = 0; i < n; i++) {
        float ad = fabsf(got[i] - ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad != 0.0f && nbad < 6) { printf("  [%s] [%zu] oneshot=%.6f resident=%.6f\n", tag, i, ref[i], got[i]); }
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
    int M = 256, K = 3840, N = 256, group = 128, W = 3;
    if (argc >= 5) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); group = atoi(argv[4]); }
    if (argc == 6) W = atoi(argv[5]);
    if (W < 1) W = 1;
    if (W > 8) W = 8;
    if (M % 4 || K % 32 || N % 64 || K % group || group % 32 || 49 * group >= 32767) {
        fprintf(stderr, "bad shape: need M%%4, K%%32, N%%64, K%%group, group%%32, 49*group<32767\n");
        return -1;
    }
    const int nG = K / group;
    printf("resident GROUPWISE int4 C_f[%d,%d]=A[%d,%d]xB[%d,%d]^T group=%d (nG=%d) W=%d\n",
           M, N, M, K, N, K, group, nG, W);

    const char *te = getenv("ROCKET_N_THREADS");
    int nthreads = te ? atoi(te) : 5;
    size_t Asz = (size_t)M*K, Bsz = (size_t)N*K, Csz = (size_t)M*N;
    int8_t *A = malloc(Asz);
    int8_t **B = malloc(W * sizeof(*B));
    float  *as = malloc((size_t)M*nG*sizeof(float));
    float **bs = malloc(W * sizeof(*bs));
    float  *Co = malloc(Csz*sizeof(float)), *Cr = malloc(Csz*sizeof(float));
    double *ref = malloc(Csz*sizeof(double));
    if (!A || !B || !as || !bs || !Co || !Cr || !ref) { fprintf(stderr, "alloc\n"); return -1; }
    srand(20260625);
    for (size_t i = 0; i < Asz; i++) A[i] = rand_i4();
    for (size_t i = 0; i < (size_t)M*nG; i++) as[i] = rand_scale();
    for (int w = 0; w < W; w++) {
        B[w]  = malloc(Bsz);
        bs[w] = malloc((size_t)N*nG*sizeof(float));
        for (size_t i = 0; i < Bsz; i++) B[w][i] = rand_i4();
        for (size_t i = 0; i < (size_t)N*nG; i++) bs[w][i] = rand_scale();
    }

    rocket_i4_ctx *ctx = rocket_i4_ctx_create(nthreads);
    if (!ctx) { fprintf(stderr, "ctx create failed — no NPU, skipping\n"); return 2; }
    rocket_i4_weights **rw = malloc(W * sizeof(*rw));
    for (int w = 0; w < W; w++) {
        rw[w] = rocket_i4_weights_pack_gw(ctx, M, K, N, B[w], group);
        if (!rw[w]) { fprintf(stderr, "pack_gw %d failed\n", w); return -1; }
    }

    int fails = 0;
    for (int w = 0; w < W; w++) {
        char tag[16]; snprintf(tag, sizeof(tag), "w%d", w);
        ref_gw(M, K, N, group, A, B[w], as, bs[w], ref);
        if (oneshot(M, K, N, A, B[w], as, bs[w], Co, group)) { fprintf(stderr, "oneshot %d failed\n", w); return -1; }
        memset(Cr, 0, Csz*sizeof(float));
        if (rocket_matmul_int4_prepacked_gw(ctx, M, K, N, A, as, bs[w], Cr, rw[w])) {
            fprintf(stderr, "prepacked_gw %d failed\n", w); return -1; }
        if (cmp_oneshot(tag, Cr, Co, Csz)) fails++;
        if (cmp_ref(tag, Cr, ref, Csz))    fails++;
    }
    /* reuse w0 — guards against scratch aliasing across calls */
    ref_gw(M, K, N, group, A, B[0], as, bs[0], ref);
    if (!oneshot(M, K, N, A, B[0], as, bs[0], Co, group)) {
        memset(Cr, 0, Csz*sizeof(float));
        if (!rocket_matmul_int4_prepacked_gw(ctx, M, K, N, A, as, bs[0], Cr, rw[0])) {
            if (cmp_oneshot("reuse-w0", Cr, Co, Csz)) fails++;
            if (cmp_ref("reuse-w0", Cr, ref, Csz))    fails++;
        }
    }

    for (int w = 0; w < W; w++) rocket_i4_weights_free(ctx, rw[w]);
    rocket_i4_ctx_free(ctx);
    printf("\n==> %s (%d checks failed)\n", fails ? "FAIL" : "ALL PASS", fails);
    free(A); free(as); free(Co); free(Cr); free(ref); free(rw);
    for (int w = 0; w < W; w++) { free(B[w]); free(bs[w]); }
    free(B); free(bs);
    return fails ? -1 : 0;
}
