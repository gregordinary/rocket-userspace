// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int8_prepacked_gw_rocket.c — the RESIDENT group-wise int8 path
 * (rocket_i8_weights_pack_gw / rocket_matmul_int8_prepacked_gw) must match the validated
 * one-shot rocket_matmul_int8_groupwise oracle, must track an fp64 reference, and must
 * not alias across distinct resident weights sharing one ctx.
 *
 * This is the primitive that carries a natively-quantized MoE expert: the int8 codes are
 * scattered into NPU BOs once and never leave, so the per-micro-batch host dequant and
 * weight scatter both disappear. The resident path fans N across worker fds; the one-shot
 * is single-fd and re-packs B every call.
 *
 * WHEN THE TWO ARE BIT-EXACT, AND WHY IT IS NOT AUTOMATIC. For one output element the
 * host accumulates over K-tiles in ascending order in BOTH paths (the N-split and the
 * M-tiling only decide which tile computes an element, not the order its K-partials
 * arrive), so the two agree bit-for-bit as soon as they agree on Kt. But they plan Kt at
 * different tile sizes — the one-shot at (Mt=min(M,MAX_TILE), Nt=min(N,MAX_TILE)), the
 * resident at the canonical (Mt=MAX_TILE, Nt=its own N-slice) — so a `group` too wide for
 * the CBUF can land on different Kt divisors, and then the two agree only to fp32
 * reassociation. A group that fits the CBUF at the WORST-CASE tile (Mt=Nt=MAX_TILE) forces
 * Kt == group everywhere, hence bit-exactness; kt_is_group() probes exactly that through
 * the public planner rather than reimplementing the bank math. group=576 — the MoE
 * operating point — is on the bit-exact side.
 *
 * Usage: matmul_int8_prepacked_gw_rocket [M K N group [W]]  (default 256 2880 512 576, W=3)
 *   M%4, K%32, N%32, K%group, group%32.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

static int8_t rand_i8(void)    { return (int8_t)(rand() % 255 - 127); }
static float  rand_scale(void) { return 0.5f + (rand() % 1000) / 1000.0f; }

/* True iff every path plans Kt == group, which is what makes resident and one-shot
 * bit-identical (see the header). Probing the worst-case tile (Mt = Nt = MAX_TILE)
 * suffices: CBUF bank use is monotone in Mt and Nt, so if the group fits there it fits
 * at every smaller tile either path can choose. */
static int kt_is_group(int K, int group)
{
    const int mt = rocket_hw_current()->max_tile;
    int Mt, Kt, Nt;
    if (rocket_matmul_plan_int8_gw(mt, K, mt, group, &Mt, &Kt, &Nt) < 0) return 0;
    return Kt == group;
}

static void ref_gw(int M, int K, int N, int group, const int8_t *A, const int8_t *B,
                   const float *as, const float *bs, double *ref)
{
    const int nG = K / group;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int g = 0; g < nG; g++) {
                long ip = 0;
                for (int k = 0; k < group; k++)
                    ip += (long)A[(size_t)m*K + (size_t)g*group + k] *
                          (long)B[(size_t)n*K + (size_t)g*group + k];
                acc += (double)as[(size_t)m*nG + g] * (double)bs[(size_t)n*nG + g] * (double)ip;
            }
            ref[(size_t)m*N + n] = acc;
        }
}

/* Scale-normalized fp64 check — see matmul_int8_groupwise_rocket.c for why a per-element
 * relative error is the wrong metric on random-signed data. */
static int cmp_ref(const char *tag, const float *got, const double *ref, size_t n)
{
    double sq = 0;
    for (size_t i = 0; i < n; i++) sq += ref[i] * ref[i];
    double rms = sqrt(sq / (double)n), tol = 1e-5 * rms;
    double max_abs = 0; long nbad = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) { nbad++; continue; }
        double ad = fabs((double)got[i] - ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad > tol) nbad++;
    }
    printf("  [%s] vs fp64: max_abs=%.4g norm_err=%.2e nbad=%ld -> %s\n",
           tag, max_abs, rms > 0 ? max_abs / rms : 0.0, nbad, nbad ? "FAIL" : "PASS");
    return nbad ? -1 : 0;
}

/* Resident vs one-shot. Bit-exact when Kt == group in both (`exact`); otherwise the two
 * differ only by fp32 reassociation of the same exact-integer partials, so hold them to
 * the same scale-normalized bound. */
static int cmp_oneshot(const char *tag, const float *got, const float *ref, size_t n, int exact)
{
    double sq = 0;
    for (size_t i = 0; i < n; i++) sq += (double)ref[i] * (double)ref[i];
    double rms = sqrt(sq / (double)n), tol = exact ? 0.0 : 1e-5 * rms;
    double max_abs = 0; long nbad = 0;
    for (size_t i = 0; i < n; i++) {
        double ad = fabs((double)got[i] - (double)ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad > tol && nbad < 6)
            printf("  [%s] [%zu] oneshot=%.6f resident=%.6f\n", tag, i, ref[i], got[i]);
        if (ad > tol) nbad++;
    }
    printf("  [%s] vs one-shot: max_abs=%.6g nbad=%ld -> %s\n", tag, max_abs, nbad,
           nbad ? "FAIL" : (exact ? "PASS (bit-exact)" : "PASS (fp32-reassoc)"));
    return nbad ? -1 : 0;
}

static int oneshot(int M, int K, int N, const int8_t *A, const int8_t *B,
                   const float *as, const float *bs, float *Cf, int group)
{
    int fd = rocket_open();
    if (fd < 0) return fd;
    int rc = rocket_matmul_int8_groupwise(fd, M, K, N, A, B, as, bs, Cf, group);
    rocket_close(fd);
    return rc;
}

int main(int argc, char **argv)
{
    int M = 256, K = 2880, N = 512, group = 576, W = 3;
    if (argc >= 5) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); group = atoi(argv[4]); }
    if (argc == 6) W = atoi(argv[5]);
    if (W < 1) W = 1;
    if (W > 8) W = 8;
    if (M % 4 || K % 32 || N % 32 || group < 32 || group % 32 || K % group) {
        fprintf(stderr, "bad shape: need M%%4, K%%32, N%%32, group>=32, group%%32, K%%group\n");
        return -1;
    }
    const int nG = K / group;
    const int exact = kt_is_group(K, group);
    printf("resident GROUPWISE int8 C_f[%d,%d]=A[%d,%d]xB[%d,%d]^T group=%d (nG=%d) W=%d "
           "[Kt==group: %s -> %s]\n",
           M, N, M, K, N, K, group, nG, W, exact ? "yes" : "no",
           exact ? "expect bit-exact" : "expect fp32-reassoc");

    const char *te = getenv("ROCKET_N_THREADS");
    int nthreads = te ? atoi(te) : 5;
    size_t Asz = (size_t)M*K, Bsz = (size_t)N*K, Csz = (size_t)M*N;
    int8_t  *A  = malloc(Asz);
    int8_t **B  = malloc(W * sizeof(*B));
    float   *as = malloc((size_t)M*nG*sizeof(float));
    float  **bs = malloc(W * sizeof(*bs));
    float   *Co = malloc(Csz*sizeof(float)), *Cr = malloc(Csz*sizeof(float));
    double  *ref = malloc(Csz*sizeof(double));
    if (!A || !B || !as || !bs || !Co || !Cr || !ref) { fprintf(stderr, "alloc\n"); return -1; }
    srand(20260714);
    for (size_t i = 0; i < Asz; i++) A[i] = rand_i8();
    for (size_t i = 0; i < (size_t)M*nG; i++) as[i] = rand_scale();
    for (int w = 0; w < W; w++) {
        B[w]  = malloc(Bsz);
        bs[w] = malloc((size_t)N*nG*sizeof(float));
        if (!B[w] || !bs[w]) { fprintf(stderr, "alloc\n"); return -1; }
        for (size_t i = 0; i < Bsz; i++) B[w][i] = rand_i8();
        for (size_t i = 0; i < (size_t)N*nG; i++) bs[w][i] = rand_scale();
    }

    rocket_i8_ctx *ctx = rocket_i8_ctx_create(nthreads);
    if (!ctx) { fprintf(stderr, "ctx create failed — no NPU, skipping\n"); return 2; }
    rocket_i8_weights **rw = malloc(W * sizeof(*rw));
    if (!rw) { fprintf(stderr, "alloc\n"); return -1; }
    for (int w = 0; w < W; w++) {
        rw[w] = rocket_i8_weights_pack_gw(ctx, M, K, N, B[w], group);
        if (!rw[w]) { fprintf(stderr, "pack_gw %d failed\n", w); return -1; }
    }
    printf("resident weight bytes (per weight): %zu MB\n", rocket_i8_weights_bytes(rw[0]) >> 20);

    int fails = 0;
    for (int w = 0; w < W; w++) {
        char tag[16]; snprintf(tag, sizeof(tag), "w%d", w);
        ref_gw(M, K, N, group, A, B[w], as, bs[w], ref);
        if (oneshot(M, K, N, A, B[w], as, bs[w], Co, group)) {
            fprintf(stderr, "oneshot %d failed\n", w); return -1; }
        memset(Cr, 0, Csz*sizeof(float));
        if (rocket_matmul_int8_prepacked_gw(ctx, M, K, N, A, as, bs[w], Cr, rw[w])) {
            fprintf(stderr, "prepacked_gw %d failed\n", w); return -1; }
        if (cmp_oneshot(tag, Cr, Co, Csz, exact)) fails++;
        if (cmp_ref(tag, Cr, ref, Csz))           fails++;
    }
    /* reuse w0 — guards against scratch aliasing across calls (the shared per-shape
     * scratch is reused by every weight; a stale accumulator would show up here) */
    ref_gw(M, K, N, group, A, B[0], as, bs[0], ref);
    if (!oneshot(M, K, N, A, B[0], as, bs[0], Co, group)) {
        memset(Cr, 0, Csz*sizeof(float));
        if (!rocket_matmul_int8_prepacked_gw(ctx, M, K, N, A, as, bs[0], Cr, rw[0])) {
            if (cmp_oneshot("reuse-w0", Cr, Co, Csz, exact)) fails++;
            if (cmp_ref("reuse-w0", Cr, ref, Csz))           fails++;
        }
    }
    /* the per-channel entry point must REJECT a group-wise weight rather than read the
     * wrong accumulator (they share the ctx and the thread body) */
    int32_t *C32 = malloc(Csz * sizeof(int32_t));
    if (C32) {
        int rc = rocket_matmul_int8_prepacked(ctx, M, K, N, A, C32, rw[0]);
        printf("  [guard] per-channel entry on a group-wise weight -> %d %s\n",
               rc, rc < 0 ? "PASS (rejected)" : "FAIL (accepted)");
        if (rc >= 0) fails++;
        free(C32);
    }

    for (int w = 0; w < W; w++) rocket_i8_weights_free(ctx, rw[w]);
    rocket_i8_ctx_free(ctx);
    printf("\n==> %s (%d checks failed)\n", fails ? "FAIL" : "ALL PASS", fails);
    free(A); free(as); free(Co); free(Cr); free(ref); free(rw);
    for (int w = 0; w < W; w++) { free(B[w]); free(bs[w]); }
    free(B); free(bs);
    return fails ? -1 : 0;
}
