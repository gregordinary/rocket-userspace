// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_mt_rocket.c — multi-core tiled matmul: correctness + scaling.
 *
 * Runs C[M,N] = A[M,K] * B[N,K]^T via rocket_matmul_fp16_mt across T worker fds
 * (one NPU core each), verifies against a CPU fp32 reference, and times T=1..4.
 *
 * Build:
 *   gcc -O2 -Iinclude tests/matmul_mt_rocket.c src/rocket_npu.c \
 *       src/npu_regcmd.c src/rocket_matmul.c src/rocket_matmul_mt.c \
 *       -o matmul_mt_rocket -lm -lpthread
 * Run:
 *   sudo ./matmul_mt_rocket 512 3840 4096
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "test_fill.h"

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6; }

int main(int argc, char **argv)
{
    int M = argc>=4 ? atoi(argv[1]) : 512;
    int K = argc>=4 ? atoi(argv[2]) : 3840;
    int N = argc>=4 ? atoi(argv[3]) : 4096;
    printf("matmul C[%d,%d] = A[%d,%d] x B[%d,%d]^T\n", M, N, M, K, N, K);

    /* no-NPU probe so this gate SKIPs (exit 2) cleanly off-device instead of failing
     * inside the mt layer (which can't tell "no NPU" from a real compute error). */
    { int fd = rocket_open(); if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); return 2; } rocket_close(fd); }

    _Float16 *A = malloc((size_t)M*K*sizeof(_Float16));
    _Float16 *B = malloc((size_t)N*K*sizeof(_Float16));
    _Float16 *C = malloc((size_t)M*N*sizeof(_Float16));
    float *ref  = malloc((size_t)M*N*sizeof(float));
    /* NON-PERIODIC inputs, deliberately. This gate used to fill A with period 13 along
     * k and B with period 11 along the flat index, which makes B[n,:] repeat every 11
     * COLUMNS at these K — so a whole-tile or whole-slice mixup between workers copies
     * bytes that already matched and the gate saw nothing. It missed a live one: CBUF
     * operand reuse crossing a job boundary, which corrupts a 128-column tile with
     * plausible values. Vary the data along both axes and the same mixup is loud. */
    /* And INTEGER-valued (tests/test_fill.h): in [-1,1] every partial and final sum is
     * an exact fp16 integer at these K, so the check below is exact. With fractional
     * inputs the bar had to allow fp16 rounding, and at 0.5 absolute AND 5% relative it
     * let a dropped k, about 0.04 against sums near 2.5, through every time. */
    tf_fill_f16_int(A, (size_t)M*K, 0x3A7A11ull, -1, 1);
    tf_fill_f16_int(B, (size_t)N*K, 0x3A7A12ull, -1, 1);

    /* CPU reference — ALL M rows. Tile-boundary / tail-row corruption (the bug class
     * the multicore fan-out is most likely to hit) lives in the LAST rows, so a
     * first-64-row spot-check would silently miss it. Computed once (one O(M*N*K)
     * pass, comparable to a single NPU run). */
    double peak = 0;
    for (int m=0;m<M;m++) for (int n=0;n<N;n++){
        float a=0; for (int k=0;k<K;k++) a+=(float)A[(size_t)m*K+k]*(float)B[(size_t)n*K+k];
        ref[(size_t)m*N+n]=a;
        if (fabs(a) > peak) peak = fabs(a);
    }
    _Float16 *want = malloc((size_t)M*N*sizeof(_Float16));
    if (!want) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (size_t i=0;i<(size_t)M*N;i++) want[i]=(_Float16)ref[i];
    if (!(peak < 2048.0)) {
        printf("max|ref| = %.0f is not below 2048, so fp16 is not exact here (FAIL)\n", peak);
        return 1;
    }
    const int dims[2] = { M, N };

    double gflop = 2.0*M*N*K/1e9;
    double base = 0;
    int fails = 0;
    for (int T=1; T<=4; T++) {
        tf_sentinel_f16(C, (size_t)M*N);
        double t0 = now_ms();
        int r = rocket_matmul_fp16_mt(M,K,N,A,B,C,T);
        double dt = now_ms()-t0;
        if (r) { fprintf(stderr,"T=%d: matmul failed (%d)\n",T,r); return 1; }

        /* exact: the integer inputs make every correct element an exact fp16 integer */
        char tag[16]; snprintf(tag, sizeof tag, "  T=%d", T);
        long nbad = tf_cmp_f16(tag, C, want, dims, 2);
        int t_fail = (nbad!=0);
        if (t_fail) fails++;
        if (T==1) base=dt;
        printf("T=%d: %7.1f ms  %6.2f GFLOP/s  %.2fx  verify %ld of %zu differ (max|ref| %.0f) -> %s\n",
               T, dt, gflop/(dt/1000.0), base/dt, nbad, (size_t)M*N, peak,
               t_fail?"FAIL":"PASS");
    }
    /* DETERMINISM at the widest fan-out. The reference check above is a one-shot, and
     * the failure this gate exists to catch is intermittent — a job interleaving
     * between two of a batch's tiles, which happens on some runs and not others. Same
     * inputs, same tiling, same order must give the same bytes; a run-to-run move is a
     * race whatever the value looks like. Repeats are cheap next to the CPU reference. */
    {
        const int T = 3, REPS = 6;
        _Float16 *first = malloc((size_t)M*N*sizeof(_Float16));
        int moved = 0;
        for (int r = 0; r < REPS && first; r++) {
            tf_sentinel_f16(C, (size_t)M*N);
            if (rocket_matmul_fp16_mt(M,K,N,A,B,C,T)) { fprintf(stderr,"rep %d failed\n",r); fails++; break; }
            if (r == 0) memcpy(first, C, (size_t)M*N*sizeof(_Float16));
            else if (memcmp(first, C, (size_t)M*N*sizeof(_Float16)) != 0) moved++;
        }
        printf("T=%d determinism over %d runs: %d moved -> %s\n", T, REPS, moved,
               moved ? "FAIL" : "PASS");
        if (moved) fails++;
        free(first);
    }

    free(A); free(B); free(C); free(ref); free(want);
    return fails ? 1 : 0;
}
