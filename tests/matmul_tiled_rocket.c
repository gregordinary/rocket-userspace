// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_tiled_rocket.c — exercise the tiled NPU matmul driver at transformer
 * scale. Computes C[M,N] = A[M,K] * B[N,K]^T on the NPU (tiled internally) and
 * checks it against a CPU fp32-accumulate reference.
 *
 * Usage: matmul_tiled_rocket [M K N]   (default 128 1024 1024)
 *
 * The default shape does NOT fit a single CBUF pass (the single-task generator
 * would silently overflow its weight banks), so it actually requires tiling.
 * Try also:  512 3840 15360  (Gemma FFN up, K fits, M/N tiled)
 *            512 15360 3840  (Gemma FFN down, K>9830 -> K-tiled + fp32 accum)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

int main(int argc, char **argv) {
    int M = 128, K = 1024, N = 1024;
    if (argc == 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    else if (argc != 1) { printf("usage: %s [M K N]\n", argv[0]); return -1; }

    int Mt, Kt, Nt;
    int njobs = rocket_matmul_plan(M, K, N, &Mt, &Kt, &Nt);
    if (njobs < 0) {
        fprintf(stderr, "unsupported shape (need K%%32==0, N%%16==0, M%%4==0||1)\n");
        return -1;
    }
    printf("matmul C[%d,%d] = A[%d,%d] x B[%d,%d]^T\n", M, N, M, K, N, K);
    printf("tiling: Mt=%d Kt=%d Nt=%d  -> %d NPU jobs (%dx%dx%d tiles)\n",
           Mt, Kt, Nt, njobs,
           (M + Mt - 1) / Mt, (N + Nt - 1) / Nt, (K + Kt - 1) / Kt);

    _Float16 *A = malloc((size_t)M * K * sizeof(_Float16));
    _Float16 *B = malloc((size_t)N * K * sizeof(_Float16));
    _Float16 *C = malloc((size_t)M * N * sizeof(_Float16));
    float    *R = malloc((size_t)M * N * sizeof(float));   /* reference */
    if (!A || !B || !C || !R) { fprintf(stderr, "host alloc failed\n"); return -1; }

    /* Small integer inputs keep results inside fp16 range even for large K. */
    srand(1234);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (_Float16)(rand() % 3);
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (_Float16)(rand() % 3);

    int verify_fail = 0;
    int fd = rocket_open();
    /* no NPU -> exit 2 (the CTest SKIP_RETURN_CODE), not a negative errno that looks
     * like a hard failure, so this gate skips cleanly off-device. */
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); free(A); free(B); free(C); free(R); return 2; }

    memset(C, 0, (size_t)M * N * sizeof(_Float16));
    int64_t t0 = now_us();
    int ret = rocket_matmul_fp16(fd, M, K, N, A, B, C);
    int64_t us = now_us() - t0;
    rocket_close(fd);
    if (ret) { fprintf(stderr, "rocket_matmul_fp16 = %d\n", ret); goto out; }

    double secs = us / 1e6;
    double gflop = 2.0 * M * K * N / 1e9;
    printf("NPU time: %.2f ms  (%.2f GFLOP -> %.2f GFLOP/s)\n",
           us / 1000.0, gflop, gflop / secs);

    /* CPU reference (fp32 accumulate, fp16 store) */
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += (float)A[(size_t)m*K+k] * (float)B[(size_t)n*K+k];
            R[(size_t)m*N+n] = s;
        }

    /* Compare. An element counts as bad only when it is wrong in BOTH absolute AND
     * relative terms — large abs alone is just big-magnitude fp16 rounding, large rel
     * alone is a near-zero reference. The integer inputs make a correct result
     * fp16-exact, so a genuinely-bad element (layout/scatter corruption) is off by a
     * lot in both. PASS requires zero bad elements; the OLD "max_rel<=0.02 OR
     * max_abs<=1.0" could pass garbage if it stayed small in one metric. */
    double max_abs = 0, max_rel = 0; long nbad = 0; int shown = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        float got = (float)C[i], exp = (float)(_Float16)R[i];
        double ad = fabs(got - exp);
        double rd = ad / (fabs(exp) + 1e-6);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (rd > 0.02 && ad > 1.0) {
            nbad++;
            if (shown < 8) { printf("  mismatch [%zu] exp=%.1f got=%.1f\n", i, exp, got); shown++; }
        }
    }
    verify_fail = (nbad != 0);
    printf("verify: max_abs=%.3f max_rel=%.4f nbad=%ld -> %s\n",
           max_abs, max_rel, nbad, verify_fail ? "FAIL" : "PASS");

out:
    free(A); free(B); free(C); free(R);
    /* exit nonzero on EITHER an NPU error (ret) OR a numeric verification failure */
    return ret ? ret : (verify_fail ? 1 : 0);
}
