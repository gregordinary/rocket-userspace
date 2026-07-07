// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int4_tiled_rocket.c — exercise the TILED int4 matmul driver
 * (rocket_matmul_int4) at transformer scale: C[M,N] = A[M,K] * B[N,K]^T as
 * int4 x int4 -> int16, tiled (M/N output tiles + host int16->int64 K-accum),
 * checked BIT-EXACT vs an int64 CPU reference.
 *
 * The int4 sibling of matmul_int8_tiled_rocket. KEY DIFFERENCE: int4's NPU output
 * is int16, so each K-tile partial SATURATES if its |sum| > 32767 and the host
 * can't recover it. This test therefore uses a REDUCED int4 value range [-3,3] so
 * that no single Kt-pass partial saturates (|partial| <= 9*Kt; at the CBUF Kt
 * limit ~1536 that is ~13.8k < 32767) — isolating the TILING geometry from the
 * (data-dependent) saturation, which is a backend scaling concern. The full int32
 * accumulation across all K is exact in int64 regardless.
 *
 * It also reports the tiling (Mt/Kt/Nt -> nKt): the headline int4 question is how
 * many Gemma shapes hit nKt=1 (single-pass K), where int4 escapes the readback
 * wall that capped int8.
 *
 * Usage: matmul_int4_tiled_rocket [M K N]   (default 512 3840 4096)
 *   Needs K%32, N%64 (int4 N-group is 64), (M%4||1). Gemma FFN: 512 15360 3840,
 *   512 3840 15360.
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
/* reduced range [-3,3] so no Kt-partial saturates int16 (see header) */
static int8_t rand_i4_small(void) { return (int8_t)(rand() % 7 - 3); }

int main(int argc, char **argv) {
    int M = 512, K = 3840, N = 4096;
    if (argc == 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    else if (argc != 1) { printf("usage: %s [M K N]\n", argv[0]); return -1; }

    int Mt, Kt, Nt;
    int njobs = rocket_matmul_plan_int4(M, K, N, &Mt, &Kt, &Nt);
    if (njobs < 0) {
        fprintf(stderr, "unsupported shape (need K%%32, N%%64, M%%4||1)\n");
        return -1;
    }
    int nKt = (K + Kt - 1) / Kt;
    printf("int4 matmul C[%d,%d] = A[%d,%d] x B[%d,%d]^T  (int4xint4->int16)\n", M, N, M, K, N, K);
    printf("tiling: Mt=%d Kt=%d Nt=%d -> %d jobs (%dx%dx%d tiles, nKt=%d => %s)\n",
           Mt, Kt, Nt, njobs, (M+Mt-1)/Mt, (N+Nt-1)/Nt, nKt, nKt,
           nKt == 1 ? "SINGLE-PASS K (no readback K-accum!)" : "host K-accum");

    int8_t  *A = malloc((size_t)M * K);
    int8_t  *B = malloc((size_t)N * K);
    int32_t *C = malloc((size_t)M * N * sizeof(int32_t));
    int32_t *R = malloc((size_t)M * N * sizeof(int32_t));
    if (!A || !B || !C || !R) { fprintf(stderr, "host alloc failed\n"); return -1; }

    srand(1234);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rand_i4_small();
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = rand_i4_small();

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); free(A); free(B); free(C); free(R); return 2; }

    memset(C, 0, (size_t)M * N * sizeof(int32_t));
    int64_t t0 = now_us();
    int ret = rocket_matmul_int4(fd, M, K, N, A, B, C);
    int64_t us = now_us() - t0;
    rocket_close(fd);
    if (ret) { fprintf(stderr, "rocket_matmul_int4 = %d\n", ret); goto out; }

    double secs = us / 1e6, gop = 2.0 * M * K * N / 1e9;
    printf("NPU time: %.2f ms  (%.2f GOP -> %.2f GOP/s)\n", us / 1000.0, gop, gop / secs);

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
        if (C[i] != R[i] && bad < 8) { printf("  mismatch [%zu] exp=%d got=%d\n", i, R[i], C[i]); bad++; }
    }
    printf("verify: max_abs_diff=%ld -> %s\n", maxabs, bad ? "FAIL" : "PASS (bit-exact)");
    if (bad) ret = -1;

out:
    free(A); free(B); free(C); free(R);
    return ret;
}
