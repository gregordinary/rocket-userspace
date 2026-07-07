// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_dtype_perf_rocket.c — raw resident-multicore throughput of fp16 vs
 * int8 vs int4 at a matched shape (no model). All three use their RESIDENT
 * path (weights packed ONCE into NPU BOs, N fanned across worker fds), so the
 * timing isolates NPU compute + readback — the apples-to-apples to the in-model
 * resident bars (fp16 ~15.1, int8 ~9.1 t/s). PURE TIMING (correctness gated
 * elsewhere).
 *
 * The headline negative result: all precisions TIE at ~460 GOP/s — the NPU
 * is DMA/dispatch-bound, not MAC-bound, so int8's 2x and int4's 4x MAC don't
 * express. (int16 has no native matmul output on this silicon and so is absent;
 * the full-precision int16 path is the int8 byte-decomposition, ~4x int8 cost.)
 * The plan/tiling env knobs (ROCKET_MM_MT/NT/KT, ROCKET_N_THREADS) apply to all.
 *
 * Usage: matmul_dtype_perf_rocket [M K N [reps]]   (default 512 3840 4096, reps=5)
 *   N must be %64 (so all three dtypes accept it). Discards the first (cold) rep.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

int main(int argc, char **argv) {
    int M = 512, K = 3840, N = 4096, reps = 5;
    if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    if (argc == 5) reps = atoi(argv[4]);
    if (argc != 1 && argc != 4 && argc != 5) { printf("usage: %s [M K N [reps]]\n", argv[0]); return -1; }
    if (N % 64) { fprintf(stderr, "N must be %%64 (int4 N-group)\n"); return -1; }
    if (reps < 2) reps = 2;

    const char *te = getenv("ROCKET_N_THREADS");
    int nthreads = te ? atoi(te) : 5;
    /* ROCKET_PERF_DT selects a subset of dtypes (substring match) so each can be
     * timed in ISOLATION, free of cross-dtype ordering/heap effects. Unset = all. */
    const char *only = getenv("ROCKET_PERF_DT");
#define WANT(nm) (!only || strstr(only, nm))
    const double gop = 2.0 * M * K * N / 1e9;
    printf("dtype perf C[%d,%d]=A[%d,%d]xB[%d,%d]^T  %.2f GOP, %d threads, reps=%d (drop cold)\n",
           M, N, M, K, N, K, gop, nthreads, reps);

    /* show the int4 tiling so we can see nKt (single-pass?) */
    int mt, kt, nt;
    if (rocket_matmul_plan_int4(M, K, N, &mt, &kt, &nt) >= 0)
        printf("int4 tiling: Mt=%d Kt=%d Nt=%d nKt=%d%s\n", mt, kt, nt, (K+kt-1)/kt,
               (K+kt-1)/kt == 1 ? "  (SINGLE-PASS K)" : "");

    size_t Asz = (size_t)M * K, Bsz = (size_t)N * K, Csz = (size_t)M * N;

    /* ---------------- fp16 ---------------- */
    if (WANT("fp16")) {
        _Float16 *A = malloc(Asz * sizeof(_Float16)), *B = malloc(Bsz * sizeof(_Float16));
        _Float16 *C = malloc(Csz * sizeof(_Float16));
        if (!A || !B || !C) { fprintf(stderr, "alloc\n"); return -1; }
        for (size_t i = 0; i < Asz; i++) A[i] = (_Float16)((i % 7) * 0.01f);
        for (size_t i = 0; i < Bsz; i++) B[i] = (_Float16)((i % 5) * 0.01f);
        rocket_ctx *ctx = rocket_ctx_create(nthreads);
        rocket_weights *w = ctx ? rocket_weights_pack(ctx, M, K, N, B) : NULL;
        if (!w) fprintf(stderr, "fp16 pack failed\n");
        else {
            double best = 1e30; long sum = 0; int ok = 1;
            for (int r = 0; r < reps && ok; r++) {
                int64_t t0 = now_us();
                if (rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, w) != 0) { ok = 0; break; }
                int64_t us = now_us() - t0;
                if (r > 0) { if (us < best*1e6) best = us/1e6; sum += us; }
            }
            if (ok) printf("  fp16 : best %.2f ms (%.1f GOP/s)  avg %.2f ms\n",
                           best*1e3, gop/best, (sum/(double)(reps-1))/1e3);
        }
        if (w) rocket_weights_free(ctx, w);
        if (ctx) rocket_ctx_free(ctx);
        free(A); free(B); free(C);
    }

    /* ---------------- int8 ---------------- */
    if (WANT("int8")) {
        int8_t *A = malloc(Asz), *B = malloc(Bsz);
        int32_t *C = malloc(Csz * sizeof(int32_t));
        if (!A || !B || !C) { fprintf(stderr, "alloc\n"); return -1; }
        for (size_t i = 0; i < Asz; i++) A[i] = (int8_t)(i % 15 - 7);
        for (size_t i = 0; i < Bsz; i++) B[i] = (int8_t)(i % 13 - 6);
        rocket_i8_ctx *ctx = rocket_i8_ctx_create(nthreads);
        rocket_i8_weights *w = ctx ? rocket_i8_weights_pack(ctx, M, K, N, B) : NULL;
        if (!w) fprintf(stderr, "int8 pack failed\n");
        else {
            double best = 1e30; long sum = 0; int ok = 1;
            for (int r = 0; r < reps && ok; r++) {
                int64_t t0 = now_us();
                if (rocket_matmul_int8_prepacked(ctx, M, K, N, A, C, w) != 0) { ok = 0; break; }
                int64_t us = now_us() - t0;
                if (r > 0) { if (us < best*1e6) best = us/1e6; sum += us; }
            }
            if (ok) printf("  int8 : best %.2f ms (%.1f GOP/s)  avg %.2f ms\n",
                           best*1e3, gop/best, (sum/(double)(reps-1))/1e3);
        }
        if (w) rocket_i8_weights_free(ctx, w);
        if (ctx) rocket_i8_ctx_free(ctx);
        free(A); free(B); free(C);
    }

    /* ---------------- int4 ---------------- */
    if (WANT("int4")) {
        int8_t *A = malloc(Asz), *B = malloc(Bsz);
        int32_t *C = malloc(Csz * sizeof(int32_t));
        if (!A || !B || !C) { fprintf(stderr, "alloc\n"); return -1; }
        for (size_t i = 0; i < Asz; i++) A[i] = (int8_t)(i % 7 - 3);
        for (size_t i = 0; i < Bsz; i++) B[i] = (int8_t)(i % 5 - 2);
        rocket_i4_ctx *ctx = rocket_i4_ctx_create(nthreads);
        rocket_i4_weights *w = ctx ? rocket_i4_weights_pack(ctx, M, K, N, B) : NULL;
        if (!w) fprintf(stderr, "int4 pack failed\n");
        else {
            double best = 1e30; long sum = 0; int ok = 1;
            for (int r = 0; r < reps && ok; r++) {
                int64_t t0 = now_us();
                if (rocket_matmul_int4_prepacked(ctx, M, K, N, A, C, w) != 0) { ok = 0; break; }
                int64_t us = now_us() - t0;
                if (r > 0) { if (us < best*1e6) best = us/1e6; sum += us; }
            }
            if (ok) printf("  int4 : best %.2f ms (%.1f GOP/s)  avg %.2f ms\n",
                           best*1e3, gop/best, (sum/(double)(reps-1))/1e3);
        }
        if (w) rocket_i4_weights_free(ctx, w);
        if (ctx) rocket_i4_ctx_free(ctx);
        free(A); free(B); free(C);
    }
    return 0;
}
