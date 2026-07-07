// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int8_tiled_rocket.c — exercise the TILED int8 matmul driver
 * (rocket_matmul_int8) at transformer scale: C[M,N] = A[M,K] * B[N,K]^T as
 * int8 x int8 -> int32, tiled internally (M/N output tiles + host int32
 * K-accumulation), checked BIT-EXACT against an int64 CPU reference.
 *
 * This is the step-3 test (after the single-task matmul_int8_rocket test): it
 * proves the int8 tiling — weight_int8 / C2=16 / C2=4 layouts, banks_for x1, and
 * host K-accum — composes correctly across tiles.
 *
 * Usage: matmul_int8_tiled_rocket [M K N]   (default 128 1024 1024)
 *   Needs K%32==0, N%32==0, (M%4==0 || M==1).
 * Try:  512 3840 15360   (Gemma FFN up: K fits-ish, M/N tiled)
 *       512 15360 3840    (Gemma FFN down: K-tiled -> host int32 accum)
 *
 * int8 exactness: HW computes int32 per K-tile, host sums in int64, reference
 * sums in int64 — both exact, so the compare is ==, not a tolerance. (The full
 * dot product fits int32: 127*127*15360 = 248M < 2^31.)
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
static int8_t rand_i8(void) { return (int8_t)(rand() % 256 - 128); }

int main(int argc, char **argv) {
    int M = 128, K = 1024, N = 1024;
    if (argc == 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    else if (argc != 1) { printf("usage: %s [M K N]\n", argv[0]); return -1; }

    int Mt, Kt, Nt;
    int njobs = rocket_matmul_plan_int8(M, K, N, &Mt, &Kt, &Nt);
    if (njobs < 0) {
        fprintf(stderr, "unsupported shape (need K%%32==0, N%%32==0, M%%4==0||1)\n");
        return -1;
    }
    printf("int8 matmul C[%d,%d] = A[%d,%d] x B[%d,%d]^T\n", M, N, M, K, N, K);
    printf("tiling: Mt=%d Kt=%d Nt=%d  -> %d NPU jobs (%dx%dx%d tiles, nKt=%d => %s)\n",
           Mt, Kt, Nt, njobs, (M+Mt-1)/Mt, (N+Nt-1)/Nt, (K+Kt-1)/Kt, (K+Kt-1)/Kt,
           (K+Kt-1)/Kt > 1 ? "host K-accum exercised" : "single K-pass");

    int8_t  *A = malloc((size_t)M * K);
    int8_t  *B = malloc((size_t)N * K);
    int32_t *C = malloc((size_t)M * N * sizeof(int32_t));
    int32_t *R = malloc((size_t)M * N * sizeof(int32_t));   /* reference */
    if (!A || !B || !C || !R) { fprintf(stderr, "host alloc failed\n"); return -1; }

    srand(1234);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rand_i8();
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = rand_i8();

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); free(A); free(B); free(C); free(R); return 2; }

    memset(C, 0, (size_t)M * N * sizeof(int32_t));
    int64_t t0 = now_us();
    int ret = rocket_matmul_int8(fd, M, K, N, A, B, C);
    int64_t us = now_us() - t0;
    rocket_close(fd);
    if (ret) { fprintf(stderr, "rocket_matmul_int8 = %d\n", ret); goto out; }

    double secs = us / 1e6, gop = 2.0 * M * K * N / 1e9;
    printf("NPU time: %.2f ms  (%.2f GOP -> %.2f GOP/s)\n", us / 1000.0, gop, gop / secs);

    /* CPU reference: int64 accumulate (exact), store int32 */
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)A[(size_t)m*K+k] * (int32_t)B[(size_t)n*K+k];
            R[(size_t)m*N+n] = (int32_t)s;
        }

    int bad = 0; long maxabs = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        long d = labs((long)C[i] - (long)R[i]);
        if (d > maxabs) maxabs = d;
        if (C[i] != R[i] && bad < 8) {
            printf("  mismatch [%zu] exp=%d got=%d\n", i, R[i], C[i]);
            bad++;
        }
    }
    printf("verify: max_abs_diff=%ld mismatches=%s -> %s\n",
           maxabs, bad ? "some" : "0", bad ? "FAIL" : "PASS (bit-exact)");
    if (bad) ret = -1;

out:
    free(A); free(B); free(C); free(R);
    return ret;
}
