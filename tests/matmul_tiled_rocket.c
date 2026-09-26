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
#include "test_fill.h"

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

int main(int argc, char **argv) {
    /* K != N: with K == N a field that swapped the two would compute a plausible surface */
    int M = 128, K = 1024, N = 768;
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

    /* Signed small integers (tests/test_fill.h): every partial and final sum stays an
     * exact fp16 integer well below 2048, so the check below is exact. The old rand()%3
     * was non-negative, which put every output near K (about 1024 here), and its bar let
     * an element pass unless it was off by more than 1.0 AND 2%, so a dropped k (at most
     * 4) always passed. */
    tf_fill_f16_int(A, (size_t)M * K, 0x7A11EDAull, -2, 2);
    tf_fill_f16_int(B, (size_t)N * K, 0x7A11EDBull, -2, 2);

    int verify_fail = 0;
    int fd = rocket_open();
    /* no NPU -> exit 2 (the CTest SKIP_RETURN_CODE), not a negative errno that looks
     * like a hard failure, so this gate skips cleanly off-device. */
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); free(A); free(B); free(C); free(R); return 2; }

    tf_sentinel_f16(C, (size_t)M * N);
    int64_t t0 = now_us();
    int ret = rocket_matmul_fp16(fd, M, K, N, A, B, C);
    int64_t us = now_us() - t0;
    rocket_close(fd);
    if (ret) { fprintf(stderr, "rocket_matmul_fp16 = %d\n", ret); verify_fail = 1; goto out; }

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

    /* Compare exactly: the integer inputs make every correct element an exact fp16
     * integer, so any difference is a defect. */
    {
        _Float16 *want = malloc((size_t)M * N * sizeof(_Float16));
        if (!want) { fprintf(stderr, "host alloc failed\n"); verify_fail = 1; goto out; }
        double peak = 0;
        for (size_t i = 0; i < (size_t)M * N; i++) {
            want[i] = (_Float16)R[i];
            if (fabs(R[i]) > peak) peak = fabs(R[i]);
        }
        const int dims[2] = { M, N };
        long nbad = tf_cmp_f16("verify", C, want, dims, 2);
        verify_fail = (nbad != 0) || !(peak < 2048.0);
        printf("verify: %ld of %zu elements differ, max|ref| %.0f -> %s\n", nbad,
               (size_t)M * N, peak, verify_fail ? "FAIL" : "PASS");
        free(want);
    }

out:
    free(A); free(B); free(C); free(R);
    /* exit 1 on EITHER an NPU error OR a numeric verification failure */
    return verify_fail ? 1 : 0;
}
