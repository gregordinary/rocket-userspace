// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * ddr_mm_bytes.c — the arm the `rockchip_ddr` PMU measures when the question is how
 * many DRAM bytes one tiled matmul actually moves.
 *
 * `bytes_moved_rocket` predicts that number analytically, from the shape, the real
 * tiling and the reuse mode, because RK3588 exposes no NPU DMA byte counter. The DDR
 * PMU can now check the prediction, but only differentially: a single process also
 * pays for its own load, its allocations, the A/B/C first touch and the idle floor,
 * and none of those is in the model. So this runs the SINGLE-fd streaming path --
 * the one bytes_moved_rocket models, not the resident or multicore one -- `reps`
 * times with NO CPU reference (a reference would move more DRAM than the matmul).
 * Run it at two rep counts under `perf stat -a` and difference: what remains is
 * (hi-lo) matmuls and nothing else.
 *
 * A second mode runs the PREPACKED path instead, which is the same arithmetic with the
 * weight scatter and its BO hoisted out of the loop. Differencing the two modes charges
 * packB -- the model's largest host term -- what the bus says it costs, rather than what
 * counting each weight byte once says it costs.
 *
 * Build:  gcc -O2 -Iinclude tests/ddr_mm_bytes.c -L build -lrocketnpu -o ddr_mm_bytes -lm
 * Usage:  ddr_mm_bytes [M K N [reps [mode]]]   (default 512 3840 4096 10 stream; mode = stream|prepacked)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <string.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}
static void fill(_Float16 *v, size_t n, uint32_t s) {
    uint32_t st = 0x9e3779b9u ^ s;
    for (size_t i = 0; i < n; i++) { st = st * 1664525u + 1013904223u;
        v[i] = (_Float16)(((float)((st >> 8) & 0xffff) / 65535.f) * 2.f - 1.f); }
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 512;
    int K = argc > 2 ? atoi(argv[2]) : 3840;
    int N = argc > 3 ? atoi(argv[3]) : 4096;
    int reps = argc > 4 ? atoi(argv[4]) : 10;
    const char *mode = argc > 5 ? argv[5] : "stream";
    int prepacked = !strcmp(mode, "prepacked");

    size_t an = (size_t)M * K, bn = (size_t)N * K, cn = (size_t)M * N;
    _Float16 *A = malloc(an * 2), *B = malloc(bn * 2), *C = malloc(cn * 2);
    if (!A || !B || !C) { printf("oom\n"); return 1; }
    fill(A, an, 1); fill(B, bn, 2);

    double t;
    if (prepacked) {
        rocket_ctx *ctx = rocket_ctx_create(1);
        if (!ctx) { printf("no NPU -> SKIP\n"); return 2; }
        rocket_weights *w = rocket_weights_pack(ctx, M, K, N, B);
        if (!w) { printf("pack failed\n"); rocket_ctx_free(ctx); return 1; }
        double t0 = now_ms();
        for (int r = 0; r < reps; r++) {
            int rc = rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, w);
            if (rc < 0) { printf("matmul rc=%d\n", rc); return 1; }
        }
        t = now_ms() - t0;
        rocket_weights_free(ctx, w);
        rocket_ctx_free(ctx);
    } else {
        int fd = rocket_open();
        if (fd < 0) { printf("no NPU -> SKIP\n"); return 2; }
        double t0 = now_ms();
        for (int r = 0; r < reps; r++) {
            int rc = rocket_matmul_fp16(fd, M, K, N, A, B, C);
            if (rc < 0) { printf("matmul rc=%d\n", rc); rocket_close(fd); return 1; }
        }
        t = now_ms() - t0;
        rocket_close(fd);
    }
    printf("ddr_mm_bytes: %dx%dx%d fp16 %s, %d reps, %.0f ms total, %.2f ms/rep\n",
           M, K, N, mode, reps, t, t / reps);
    printf("  REPS %d\n", reps);
    free(A); free(B); free(C);
    return 0;
}
