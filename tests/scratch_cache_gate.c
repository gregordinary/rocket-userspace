// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * scratch_cache_gate.c — GATE on the shared per-shape scratch cache's recycling.
 *
 * A rocket_ctx caches one compute scratch per (M,K,N,group), 32 slots. It used to
 * REFUSE past that: the 33rd distinct shape returned NULL, every caller turned that
 * into a bare -1 with no log, and it was indistinguishable from an unsupported shape.
 * Any caller that sweeps M reaches it in normal use — variable prompt lengths, a
 * micro-batch tail, per-expert token counts in a MoE router. It now recycles the least
 * recently used slot instead.
 *
 * Recycling is the part that needs a gate, because a resident weight is packed against
 * a scratch and outlives it. What this asserts, on real hardware:
 *
 *   1. Sweeping well past the slot count keeps computing, and every result stays
 *      correct — the shape that comes back after eviction is not a stale one.
 *   2. A resident weight still computes correctly after the scratch it was packed
 *      against has been evicted and rebuilt by an unrelated sweep. This is the case
 *      that would have dangled had a weight kept a pointer into the cache rather than
 *      a layout signature.
 *   3. Re-touching an evicted shape returns a WORKING scratch, not a freed one.
 *
 * CTest target scratch_cache_gate; skip-code 2 with no NPU.
 *   sudo ./scratch_cache_gate           # K=256 N=256, 40 shapes (cache holds 32)
 *   sudo ./scratch_cache_gate 512 256 48
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

/* Spot-check C[M,N] = A * B^T on a few rows. Same tolerance idiom as the sibling
 * prepacked gates: fp16 accumulation, so compare against an fp32 host sum. */
static int verify(const char *what, const _Float16 *C, const _Float16 *A,
                  const _Float16 *B, int M, int K, int N)
{
    int bad = 0;
    float maxrel = 0;
    const int rows[] = { 0, M / 2, M - 1 };
    for (size_t r = 0; r < sizeof rows / sizeof rows[0]; r++) {
        int m = rows[r];
        for (int n = 0; n < N; n += 7) {
            float a = 0;
            for (int k = 0; k < K; k++) a += (float)A[(size_t)m * K + k] * (float)B[(size_t)n * K + k];
            float got = (float)C[(size_t)m * N + n];
            float ad = fabsf(got - a), rd = ad / (fabsf(a) + 1e-6f);
            if (rd > maxrel) maxrel = rd;
            if (ad > 0.5f && rd > 0.05f) bad++;
        }
    }
    if (bad) printf("    FAIL %s: %d bad elements, max_rel=%.4f\n", what, bad, maxrel);
    return bad ? 1 : 0;
}

static void fill(_Float16 *p, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        p[i] = (_Float16)(((int)(seed >> 20) & 0xFF) - 128) * (_Float16)0.01f;
    }
}

int main(int argc, char **argv)
{
    const int K      = argc > 1 ? atoi(argv[1]) : 256;
    const int N      = argc > 2 ? atoi(argv[2]) : 256;
    /* Comfortably past ROCKET_FANOUT_MAX_SLOTS (32) so eviction is forced, not hoped for. */
    const int nshape = argc > 3 ? atoi(argv[3]) : 40;

    int probe = rocket_open();
    if (probe < 0) { printf("SKIP: no NPU (%d)\n", probe); return 2; }
    rocket_close(probe);

    /* Every shape is M = 256 + 4*i, all >= MAX_TILE, so they tile identically and one
     * resident weight is valid at all of them — which is what lets test 2 separate
     * "the weight survived eviction" from "the tiling changed". */
    const int Mmax = 256 + 4 * nshape;
    _Float16 *A = malloc((size_t)Mmax * K * sizeof(_Float16));
    _Float16 *B = malloc((size_t)N * K * sizeof(_Float16));
    _Float16 *C = malloc((size_t)Mmax * N * sizeof(_Float16));
    if (!A || !B || !C) { printf("FAIL: host alloc\n"); free(A); free(B); free(C); return 1; }
    fill(A, (size_t)Mmax * K, 1); fill(B, (size_t)N * K, 2);

    rocket_ctx *ctx = rocket_ctx_create(2);
    if (!ctx) { printf("FAIL: rocket_ctx_create\n"); free(A); free(B); free(C); return 1; }

    int fails = 0;
    rocket_weights *w = rocket_weights_pack(ctx, 256, K, N, B);
    if (!w) { printf("FAIL: rocket_weights_pack\n"); rocket_ctx_free(ctx); free(A); free(B); free(C); return 1; }

    /* 1 + 2: sweep M past the slot count against the SAME resident weight. Every call
     * after the 32nd runs on a scratch built in a slot something else was evicted from. */
    printf("sweeping %d distinct M against one resident weight (cache holds 32)\n", nshape);
    for (int i = 0; i < nshape; i++) {
        int M = 256 + 4 * i;
        memset(C, 0, (size_t)M * N * sizeof(_Float16));
        int rc = rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, w);
        if (rc != 0) { printf("    FAIL: shape %d (M=%d) returned %d\n", i, M, rc); fails++; continue; }
        char tag[64]; snprintf(tag, sizeof tag, "shape %d M=%d", i, M);
        fails += verify(tag, C, A, B, M, K, N);
    }
    printf("  -> %s\n", fails ? "FAIL" : "PASS");

    /* 3: the FIRST shape is now long evicted. Touching it must rebuild, not resurrect. */
    printf("re-touching the evicted first shape (M=256)\n");
    int before = fails;
    memset(C, 0, (size_t)256 * N * sizeof(_Float16));
    int rc = rocket_matmul_fp16_prepacked(ctx, 256, K, N, A, C, w);
    if (rc != 0) { printf("    FAIL: re-touch returned %d\n", rc); fails++; }
    else fails += verify("re-touched M=256", C, A, B, 256, K, N);
    printf("  -> %s\n", fails > before ? "FAIL" : "PASS");

    /* And the streaming path, which shares the same cache from the other side. */
    printf("streaming path over the same swept shapes\n");
    before = fails;
    rocket_stream *st = rocket_stream_create(2);
    if (!st) { printf("    FAIL: rocket_stream_create\n"); fails++; }
    else {
        for (int i = 0; i < nshape; i++) {
            int M = 256 + 4 * i;
            memset(C, 0, (size_t)M * N * sizeof(_Float16));
            if (rocket_matmul_fp16_stream(st, M, K, N, A, B, C) != 0) {
                printf("    FAIL: stream shape %d (M=%d)\n", i, M); fails++; continue;
            }
            if (i % 8 == 0) {
                char tag[64]; snprintf(tag, sizeof tag, "stream M=%d", M);
                fails += verify(tag, C, A, B, M, K, N);
            }
        }
        rocket_stream_free(st);
    }
    printf("  -> %s\n", fails > before ? "FAIL" : "PASS");

    rocket_weights_free(ctx, w);
    rocket_ctx_free(ctx);
    free(A); free(B); free(C);
    printf("scratch_cache_gate: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
