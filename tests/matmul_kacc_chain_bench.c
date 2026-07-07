// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_kacc_chain_bench.c — warm-loop perf probe for cross-ki KACC chaining.
 *
 * Times `iters` warm rocket_matmul_fp16 calls for one shape and reports per-iter ms.
 * The caller picks chained vs unchained via ROCKET_KACC_CHAIN (and ROCKET_MM_PROFILE
 * for the pack/gen/sync/submit/wait/read split). Not a CTest gate — a hand-run probe.
 *
 * Usage: matmul_kacc_chain_bench [M K N [iters]]   (default 512 15360 3840 20)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    int M = 512, K = 15360, N = 3840, iters = 20;
    if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    if (argc >= 5) iters = atoi(argv[4]);

    int Mt, Kt, Nt;
    int njobs = rocket_matmul_plan(M, K, N, &Mt, &Kt, &Nt);
    if (njobs < 0) { fprintf(stderr, "unsupported shape\n"); return 1; }
    int nKt = (K + Kt - 1) / Kt;

    _Float16 *A = malloc((size_t)M * K * sizeof(_Float16));
    _Float16 *B = malloc((size_t)N * K * sizeof(_Float16));
    _Float16 *C = malloc((size_t)M * N * sizeof(_Float16));
    if (!A || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }
    srand(7);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (_Float16)(rand() % 3);
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (_Float16)(rand() % 3);

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU -> SKIP\n"); return 2; }

    const char *chain = getenv("ROCKET_KACC_CHAIN");
    printf("shape %dx%dx%d  Mt=%d Kt=%d Nt=%d nKt=%d  KACC_CHAIN=%s  iters=%d\n",
           M, K, N, Mt, Kt, Nt, nKt, chain ? chain : "(unset)", iters);

    for (int w = 0; w < 3; w++) rocket_matmul_fp16(fd, M, K, N, A, B, C);  /* warm */

    double *t = malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = now_ms();
        rocket_matmul_fp16(fd, M, K, N, A, B, C);
        t[i] = now_ms() - t0;
    }
    rocket_close(fd);

    qsort(t, iters, sizeof(double), cmp_d);
    double sum = 0; for (int i = 0; i < iters; i++) sum += t[i];
    double gflop = 2.0 * M * K * N / 1e9;
    double med = t[iters / 2];
    printf("  per-iter ms: min=%.2f  median=%.2f  mean=%.2f  max=%.2f   (%.1f GFLOP/s @ median)\n",
           t[0], med, sum / iters, t[iters - 1], gflop / (med / 1e3));
    free(t); free(A); free(B); free(C);
    return 0;
}
