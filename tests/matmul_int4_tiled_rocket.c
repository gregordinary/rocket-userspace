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
 * can't recover it. The inputs span the entry's whole contract, [-8,7], because a
 * mis-encoded -8 nibble is exactly what a narrower fill cannot see (this gate used
 * [-3,3] and no int4 gate reached -8). A random full-range partial sits far below
 * the int16 bound, and the gate checks that rather than assuming it: the reference
 * accumulates per K-tile at the planner's Kt and fails the INSTRUMENT if any partial
 * would saturate, which isolates the tiling geometry from the (data-dependent)
 * saturation, a backend scaling concern. The full accumulation across all K is exact
 * in int64 regardless. The output starts as a 0xAA sentinel.
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
#include "test_fill.h"

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

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

    tf_fill_i8(A, (size_t)M * K, 0x14A11ull, -8, 7);
    tf_fill_i8(B, (size_t)N * K, 0x14B11ull, -8, 7);

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); free(A); free(B); free(C); free(R); return 2; }

    tf_sentinel_bytes(C, (size_t)M * N * sizeof(int32_t));
    int64_t t0 = now_us();
    int ret = rocket_matmul_int4(fd, M, K, N, A, B, C);
    int64_t us = now_us() - t0;
    rocket_close(fd);
    if (ret) { fprintf(stderr, "rocket_matmul_int4 = %d\n", ret); ret = 1; goto out; }

    double secs = us / 1e6, gop = 2.0 * M * K * N / 1e9;
    printf("NPU time: %.2f ms  (%.2f GOP -> %.2f GOP/s)\n", us / 1000.0, gop, gop / secs);

    /* The reference, accumulated per K-tile so the int16 precondition is checked. */
    int64_t peak_partial = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64_t s = 0;
            for (int k0 = 0; k0 < K; k0 += Kt) {
                int64_t p = 0;
                int k1 = k0 + Kt < K ? k0 + Kt : K;
                for (int k = k0; k < k1; k++)
                    p += (int32_t)A[(size_t)m*K+k] * (int32_t)B[(size_t)n*K+k];
                if (llabs(p) > peak_partial) peak_partial = llabs(p);
                s += p;
            }
            R[(size_t)m*N+n] = (int32_t)s;
        }
    printf("largest |K-tile partial| %lld (int16 holds 32767)\n", (long long)peak_partial);
    if (peak_partial > 32767) {
        printf("the fill saturates an int16 partial at Kt=%d: the INSTRUMENT is wrong here -> FAIL\n",
               Kt);
        ret = 1;
        goto out;
    }

    const int dims[2] = { M, N };
    long bad = tf_cmp_i32("verify", C, R, dims, 2);
    printf("verify: %ld of %zu elements differ -> %s\n", bad, (size_t)M * N,
           bad ? "FAIL" : "PASS (bit-exact)");
    if (bad) ret = 1;

out:
    free(A); free(B); free(C); free(R);
    return ret;
}
