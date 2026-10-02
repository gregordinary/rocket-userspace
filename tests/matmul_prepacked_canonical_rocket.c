// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_prepacked_canonical_rocket.c — gate for ROCKET_CTX_TILING_CANONICAL: one resident
 * weight serves every call M, small ones included.
 *
 * On a canonical ctx a plan at M < MAX_TILE keeps the M = MAX_TILE plan's K and N tiling, so
 * the weight layout never moves. For each shape the gate packs B ONCE, alternately at a small
 * M and at M >= MAX_TILE (so both directions of reuse are exercised), then computes at every
 * M in the ladder against that same handle. Any return other than 0 fails, -2 included: a
 * canonical ctx promises the weight fits.
 *
 * THE CHECK IS EXACT AND COVERS EVERY ELEMENT. The operands are hashed integers (A in [-3,3],
 * B in [-2,2], tests/test_fill.h), so every sum is an exact fp16 integer and the result must
 * equal the integer reference bit for bit; the gate asserts max|ref| < 2048. The output
 * starts as an fp16 NaN sentinel.
 *
 * The shapes are the Laya decision model's projections (ModernBERT-large and -base, and the
 * torch pre-norm head), plus one K past a single canonical K tile.
 *
 * CTest target matmul_prepacked_canonical_rocket; skip-code 2 off-device.
 *   sudo ./matmul_prepacked_canonical_rocket          # every shape, T=3
 *   sudo ./matmul_prepacked_canonical_rocket 1024 3072 3   # one K N T
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "test_fill.h"

static int verify_all(const char *tag, const _Float16 *C, const _Float16 *A, const _Float16 *B,
                      int M, int K, int N)
{
    _Float16 *ref = malloc((size_t)M * N * sizeof(_Float16));
    if (!ref) { fprintf(stderr, "oom\n"); return 1; }
    double peak = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double a = 0;
            for (int k = 0; k < K; k++) a += (double)A[(size_t)m*K+k] * (double)B[(size_t)n*K+k];
            if (fabs(a) > peak) peak = fabs(a);
            ref[(size_t)m*N+n] = (_Float16)a;
        }
    const int dims[2] = { M, N };
    long bad = tf_cmp_f16(tag, C, ref, dims, 2);
    int inexact = !(peak < 2048.0);
    if (bad || inexact)
        printf("    %s: %ld of %zu elements differ, max|ref| %.0f%s -> FAIL\n", tag, bad,
               (size_t)M * N, peak, inexact ? " (not below 2048: the fill range is wrong)" : "");
    free(ref);
    return (bad || inexact) ? 1 : 0;
}

static int run_shape(int K, int N, int T, int Mpack)
{
    static const int Mrun[] = { 4, 8, 48, 52, 60, 64, 100, 128, 196, 252, 256, 260, 512, 640 };
    const int nrun = (int)(sizeof Mrun / sizeof Mrun[0]);
    const int Mmax = 640;

    rocket_ctx *ctx = rocket_ctx_create_ex(T, ROCKET_CTX_TILING_CANONICAL);
    if (!ctx) { fprintf(stderr, "rocket_ctx_create_ex failed\n"); return 1; }
    _Float16 *A = malloc((size_t)Mmax*K*sizeof(_Float16));
    _Float16 *B = malloc((size_t)N*K*sizeof(_Float16));
    _Float16 *C = malloc((size_t)Mmax*N*sizeof(_Float16));
    if (!A || !B || !C) { fprintf(stderr, "oom\n"); return 1; }
    tf_fill_f16_int(A, (size_t)Mmax*K, 0xCA0A + K, -3, 3);
    tf_fill_f16_int(B, (size_t)N*K,    0xCA0B + N, -2, 2);

    int fail = 0;
    rocket_weights *w = rocket_weights_pack(ctx, Mpack, K, N, B);
    if (!w) { printf("  K=%d N=%d: pack@M=%d failed -> FAIL\n", K, N, Mpack); fail = 1; goto out; }
    for (int i = 0; i < nrun; i++) {
        int M = Mrun[i];
        char tag[64];
        snprintf(tag, sizeof tag, "K=%d N=%d M=%d", K, N, M);
        tf_sentinel_f16(C, (size_t)M*N);
        int r = rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, w);
        if (r) { printf("    %s: prepacked ret=%d -> FAIL\n", tag, r); fail = 1; continue; }
        fail |= verify_all(tag, C, A, B, M, K, N);
    }
    printf("  K=%-5d N=%-5d packed@M=%-3d run@M=4..640 (%d sizes): %s\n", K, N, Mpack, nrun,
           fail ? "FAIL" : "PASS");
    rocket_weights_free(ctx, w);
out:
    rocket_ctx_free(ctx);
    free(A); free(B); free(C);
    return fail;
}

static double now_ms(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* The price of the canonical tiling: warm median of `reps` calls at each small M, the
 * canonical ctx's one pack against a default ctx packed at that same M. */
static void bench_shape(int K, int N, int T, int reps)
{
    static const int Ms[] = { 48, 64, 128, 196, 256, 640 };
    _Float16 *A = calloc((size_t)640 * K, sizeof(_Float16));
    _Float16 *B = malloc((size_t)N * K * sizeof(_Float16));
    _Float16 *C = malloc((size_t)640 * N * sizeof(_Float16));
    double *t = malloc((size_t)reps * sizeof(double));
    tf_fill_f16_int(A, (size_t)640 * K, 1, -3, 3);
    tf_fill_f16_int(B, (size_t)N * K, 2, -2, 2);
    rocket_ctx *cc = rocket_ctx_create_ex(T, ROCKET_CTX_TILING_CANONICAL);
    rocket_weights *wc = rocket_weights_pack(cc, 256, K, N, B);
    for (int i = 0; i < (int)(sizeof Ms / sizeof Ms[0]); i++) {
        int M = Ms[i];
        double med[2];
        rocket_ctx *dc = rocket_ctx_create(T);
        rocket_weights *wd = rocket_weights_pack(dc, M, K, N, B);
        for (int arm = 0; arm < 2; arm++) {
            rocket_ctx *c = arm ? dc : cc; rocket_weights *w = arm ? wd : wc;
            for (int r = 0; r < 3; r++) rocket_matmul_fp16_prepacked(c, M, K, N, A, C, w);
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms();
                rocket_matmul_fp16_prepacked(c, M, K, N, A, C, w);
                t[r] = now_ms() - t0;
            }
            for (int a = 0; a < reps; a++)           /* insertion sort, then the median */
                for (int b = a; b > 0 && t[b] < t[b - 1]; b--) { double x = t[b]; t[b] = t[b-1]; t[b-1] = x; }
            med[arm] = t[reps / 2];
        }
        printf("  bench K=%-5d N=%-5d M=%-4d canonical %7.3f ms  per-M pack %7.3f ms  (%.2fx)\n",
               K, N, M, med[0], med[1], med[0] / med[1]);
        rocket_weights_free(dc, wd); rocket_ctx_free(dc);
    }
    rocket_weights_free(cc, wc); rocket_ctx_free(cc);
    free(A); free(B); free(C); free(t);
}

int main(int argc, char **argv)
{
    {   /* a missing device is the only skip: probe it before the context can fail */
        int fd = rocket_open();
        if (fd < 0) { fprintf(stderr, "no NPU (%d) -> SKIP\n", fd); return 2; }
        rocket_close(fd);
    }
    if (getenv("ROCKET_CANON_BENCH")) {   /* timing only, no verdict */
        int reps = atoi(getenv("ROCKET_CANON_BENCH")); if (reps < 5) reps = 21;
        static const int bs[][2] = { { 1024, 3072 }, { 1024, 5248 }, { 2624, 1024 }, { 768, 2304 } };
        for (int s = 0; s < 4; s++) bench_shape(bs[s][0], bs[s][1], 3, reps);
        return 0;
    }
    if (argc > 2) {
        int K = atoi(argv[1]), N = atoi(argv[2]), T = argc > 3 ? atoi(argv[3]) : 3;
        if (K % 32 || N % 16) { fprintf(stderr, "need K%%32==0, N%%16==0\n"); return 1; }
        int f = run_shape(K, N, T, 60) | run_shape(K, N, T, 512);
        printf("==== %s ====\n", f ? "FAIL" : "PASS");
        return f;
    }
    static const int shapes[][2] = {
        { 1024, 3072 }, { 1024, 1024 }, { 1024, 5248 }, { 2624, 1024 },   /* ModernBERT-large */
        { 1024, 4096 }, { 4096, 1024 },                                    /* the pre-norm head */
        { 768, 2304 }, { 768, 768 }, { 1152, 768 },                        /* ModernBERT-base */
    };
    int fail = 0;
    for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++)
        fail |= run_shape(shapes[s][0], shapes[s][1], 3, (s & 1) ? 512 : 60);
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
