// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * membench.c — DRAM bandwidth + NPU readback de-tile microbench (no deps).
 *
 * Grounds the perf model from the walkthrough on THIS box, which exposes no DDR
 * PMU and has no tinymembench installed:
 *   - achievable LPDDR bandwidth (memcpy / read / write);
 *   - the cost of mm_compute's readback de-tile: a feat_idx gather + fp16->fp32
 *     accumulate, long claimed ~0.69 GB/s = index-math-bound, NOT bandwidth-bound.
 *     We compare a CONTIGUOUS fp16->fp32 accumulate against the feat_idx-GATHERED
 *     one to separate the gather penalty from the convert cost. If "contig" is
 *     near memcpy but "gather" is ~10x slower, the readback wall is the index
 *     math (a layout/vectorization problem), not DRAM bandwidth.
 *
 * Build (on RK3588):  gcc -O3 -march=native -o membench membench.c
 * Run:          ./membench
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef _Float16 f16;

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

/* fp16 output-cube de-tile index — identical to feat_idx() in rocket_matmul.c
 * (C2=8). This is the exact gather mm_compute does on readback. */
static inline size_t feat_idx(int H, int ch, int h) {
    return ((size_t)(ch - 1) / 8) * (size_t)H * 8 + 8 * (size_t)(h - 1) + (ch - 1) % 8;
}

int main(void) {
    const size_t MB = (size_t)1 << 20;
    const size_t bytes = (size_t)256 * MB;          /* >> 3MB L3 -> DRAM, not cache */
    const int iters = 5;

    char *a = malloc(bytes), *b = malloc(bytes);
    if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
    memset(a, 1, bytes); memset(b, 2, bytes);

    /* 1. memcpy bandwidth (reported as bytes moved on the read side). The asm
     * clobber per iter stops -O3 from dead-code-eliminating the (otherwise unread)
     * memcpy/memset stores -> previously printed "inf". */
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) { memcpy(b, a, bytes); __asm__ __volatile__("" :: "r"(b) : "memory"); }
    double dt = now_ms() - t0;
    printf("memcpy        : %6.1f GB/s   (%zu MB x %d)\n",
           (double)bytes * iters / (dt * 1e-3) / 1e9, bytes / MB, iters);

    /* 2. sequential read (sum 8B words) */
    uint64_t *p = (uint64_t *)a; size_t nq = bytes / 8; volatile uint64_t sink = 0;
    t0 = now_ms();
    for (int i = 0; i < iters; i++) { uint64_t s = 0; for (size_t j = 0; j < nq; j++) s += p[j]; sink += s; }
    dt = now_ms() - t0;
    printf("read (sum)    : %6.1f GB/s\n", (double)bytes * iters / (dt * 1e-3) / 1e9);

    /* 3. sequential write */
    t0 = now_ms();
    for (int i = 0; i < iters; i++) { memset(a, i, bytes); __asm__ __volatile__("" :: "r"(a) : "memory"); }
    dt = now_ms() - t0;
    printf("write (memset): %6.1f GB/s\n", (double)bytes * iters / (dt * 1e-3) / 1e9);

    /* 4. readback de-tile: contiguous vs feat_idx-gathered fp16->fp32 accumulate.
     * Representative output tile M x N, repeated to build comparable volume. */
    const int M = 256, N = 4096, reps = 100;
    const size_t tn = (size_t)M * N;
    f16   *src = malloc(tn * sizeof(f16));
    float *acc = malloc(tn * sizeof(float));
    if (!src || !acc) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (size_t i = 0; i < tn; i++) src[i] = (f16)1.0f;
    const double rb_gb = (double)tn * sizeof(f16) * reps / 1e9;   /* fp16 bytes read */

    memset(acc, 0, tn * sizeof(float));
    t0 = now_ms();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < tn; i++) acc[i] += (float)src[i];
    dt = now_ms() - t0;
    printf("de-tile contig: %6.2f GB/s   (fp16->f32 accumulate, %dx%d x%d reps)\n",
           rb_gb / (dt * 1e-3), M, N, reps);

    memset(acc, 0, tn * sizeof(float));
    t0 = now_ms();
    for (int r = 0; r < reps; r++)
        for (int h = 1; h <= M; h++)
            for (int nn = 1; nn <= N; nn++)
                acc[(size_t)(h - 1) * N + (nn - 1)] += (float)src[feat_idx(M, nn, h)];
    dt = now_ms() - t0;
    printf("de-tile gather: %6.2f GB/s   (feat_idx, the real mm_compute readback)  acc[0]=%.0f\n",
           rb_gb / (dt * 1e-3), acc[0]);

    free(a); free(b); free(src); free(acc);
    (void)sink;
    return 0;
}
