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
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

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
    for (size_t i=0;i<(size_t)M*K;i++) A[i]=(_Float16)(((i*7)%13-6)*0.05f);
    for (size_t i=0;i<(size_t)N*K;i++) B[i]=(_Float16)(((i*5)%11-5)*0.05f);

    /* CPU reference — ALL M rows. Tile-boundary / tail-row corruption (the bug class
     * the multicore fan-out is most likely to hit) lives in the LAST rows, so a
     * first-64-row spot-check would silently miss it. Computed once (one O(M*N*K)
     * pass, comparable to a single NPU run). */
    for (int m=0;m<M;m++) for (int n=0;n<N;n++){
        float a=0; for (int k=0;k<K;k++) a+=(float)A[(size_t)m*K+k]*(float)B[(size_t)n*K+k];
        ref[(size_t)m*N+n]=a;
    }

    double gflop = 2.0*M*N*K/1e9;
    double base = 0;
    int fails = 0;
    for (int T=1; T<=4; T++) {
        for (size_t i=0;i<(size_t)M*N;i++) C[i]=(_Float16)-99.0f;
        double t0 = now_ms();
        int r = rocket_matmul_fp16_mt(M,K,N,A,B,C,T);
        double dt = now_ms()-t0;
        if (r) { fprintf(stderr,"T=%d: matmul failed (%d)\n",T,r); return 1; }

        /* An element is bad only if wrong in BOTH abs AND rel (large abs alone =
         * fp16 rounding on a big value; large rel alone = near-zero reference). */
        float max_abs=0, max_rel=0; long nbad=0;
        for (int m=0;m<M;m++) for (int n=0;n<N;n++){
            float got=(float)C[(size_t)m*N+n], want=ref[(size_t)m*N+n];
            float ad=fabsf(got-want), rd=ad/(fabsf(want)+1e-6f);
            if (ad>max_abs) max_abs=ad;
            if (rd>max_rel) max_rel=rd;
            if (ad>0.5f && rd>0.05f) nbad++;
        }
        int t_fail = (nbad!=0);
        if (t_fail) fails++;
        if (T==1) base=dt;
        printf("T=%d: %7.1f ms  %6.2f GFLOP/s  %.2fx  verify max_abs=%.3f max_rel=%.4f nbad=%ld -> %s\n",
               T, dt, gflop/(dt/1000.0), base/dt, max_abs, max_rel, nbad,
               t_fail?"FAIL":"PASS");
    }
    free(A); free(B); free(C); free(ref);
    return fails ? 1 : 0;
}
