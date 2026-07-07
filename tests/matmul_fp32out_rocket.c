// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_fp32out_rocket.c — gate for the fp32-OUTPUT fp16 matmul.
 *
 * Compares the two fp16-input matmul paths against the EXACT fp16-input dot product
 * (computed in fp64) over the same random inputs:
 *   - rocket_matmul_fp16        -> fp16 C  (narrows each K-partial to fp16 before the
 *                                            host fp32 sum, then narrows the result)
 *   - rocket_matmul_fp16_f32out -> fp32 C  (full fp32 accumulator out, fp64 host K-accum)
 *
 * The reference rounds A,B to fp16 (the inputs already ARE fp16) and accumulates in
 * double, so it is the genuine fp16-input result with no output rounding. The point
 * of the fp32-out path is that fp32-out removes the per-K-tile + final fp16 rounding, so on a
 * multi-K-tile shape (nKt>1) it must be SUBSTANTIALLY closer to that reference than
 * the fp16-out path.
 *
 * PASS =
 *   (1) fp32-out tracks the fp64 reference tightly (norm_err < TOL32, ~fp32 noise), AND
 *   (2) fp32-out is at least IMPROVE_X times more accurate than fp16-out (proves the
 *       feature actually does something — defaults to a conservative 5x).
 * Also reports wall time for each path so the 2x-output-readback cost is visible.
 *
 * Usage: matmul_fp32out_rocket <M> <K> <N>     (K%32, N%16, M%4||1)
 *   good multi-K-tile shapes: 64 4096 512 | 256 3840 512 | 128 8192 256
 * Env: ROCKET_F32O_TOL32 (def 2e-4), ROCKET_F32O_IMPROVE (def 5.0),
 *      ROCKET_F32O_MAG (def 4.0), ROCKET_F32O_SAMPLES (def 8192).
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}
static int env_int(const char *n, int d){ const char*e=getenv(n); return e?(int)strtol(e,NULL,0):d; }
static double env_dbl(const char *n, double d){ const char*e=getenv(n); return e?strtod(e,NULL):d; }

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s <M> <K> <N>  (try 64 4096 512)\n", argv[0]); return -1; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    const double TOL32   = env_dbl("ROCKET_F32O_TOL32", 2e-4);
    const double IMPROVE = env_dbl("ROCKET_F32O_IMPROVE", 5.0);
    const float  mag     = (float)env_dbl("ROCKET_F32O_MAG", 4.0);
    const int    samples = env_int("ROCKET_F32O_SAMPLES", 8192);

    int Mt=0, Kt=0, Nt=0;
    int njobs = rocket_matmul_plan(M, K, N, &Mt, &Kt, &Nt);
    if (njobs < 0) { fprintf(stderr, "unsupported shape (need K%%32, N%%16, M%%4||1)\n"); return 2; }
    int nKt = (K + Kt - 1) / Kt;
    printf("fp32-OUT gate: C[%d,%d] = A[%d,%d] x B[%d,%d]^T  Mt=%d Kt=%d Nt=%d nKt=%d (%d jobs)\n",
           M, N, M, K, N, K, Mt, Kt, Nt, nKt, njobs);
    if (nKt < 2)
        printf("  NOTE: nKt=%d (single K-tile) — fp32-out then differs from fp16-out only by the\n"
               "  final narrowing; use a larger K (e.g. %d) to exercise per-K-tile rounding.\n",
               nKt, Kt * 2);

    _Float16 *A  = malloc((size_t)M*K*sizeof(_Float16));
    _Float16 *B  = malloc((size_t)N*K*sizeof(_Float16));
    _Float16 *C16= malloc((size_t)M*N*sizeof(_Float16));
    float    *C32= malloc((size_t)M*N*sizeof(float));
    if (!A||!B||!C16||!C32) { fprintf(stderr,"host malloc failed\n"); return -1; }
    srand(1234);
    for (size_t i=0;i<(size_t)M*K;i++) A[i]=(_Float16)(((float)rand()/RAND_MAX*2.f-1.f)*mag);
    for (size_t i=0;i<(size_t)N*K;i++) B[i]=(_Float16)(((float)rand()/RAND_MAX*2.f-1.f)*mag);

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr,"rocket_open failed (%d)\n", fd); return 2; }

    /* Warm each path once (untimed) so the NPU spin-up + page-fault storm + caches
     * don't get charged to whichever path runs first — otherwise the cold/warm
     * ordering dominates and the 2x-output-readback delta is invisible. Then take the
     * best of 3 warm runs each. */
    int r16 = rocket_matmul_fp16(fd, M, K, N, A, B, C16);
    int r32 = rocket_matmul_fp16_f32out(fd, M, K, N, A, B, C32);
    double ms16 = 1e30, ms32 = 1e30;
    for (int it = 0; it < 3; it++) {
        int64_t t0 = now_us(); r16 |= rocket_matmul_fp16(fd, M, K, N, A, B, C16);
        double d = (now_us()-t0)/1000.0; if (d < ms16) ms16 = d;
        t0 = now_us(); r32 |= rocket_matmul_fp16_f32out(fd, M, K, N, A, B, C32);
        d = (now_us()-t0)/1000.0; if (d < ms32) ms32 = d;
    }
    rocket_close(fd);
    if (r16 || r32) { fprintf(stderr,"matmul failed (fp16=%d fp32=%d)\n", r16, r32);
                      free(A);free(B);free(C16);free(C32); return -1; }

    double gops = 2.0*(double)M*N*K/1e9;
    /* output readback bytes: fp32-out is 2x fp16-out. As a fraction of the total
     * device traffic this only matters when M*N (output) rivals N*K (weights). */
    double out_mb_16 = (double)M*N*2/1e6, out_mb_32 = (double)M*N*4/1e6, wt_mb = (double)N*K*2/1e6;
    printf("rocket_matmul_fp16        (best-of-3 warm) %.2f ms, %.1f GOP/s\n", ms16, gops/(ms16/1000.0));
    printf("rocket_matmul_fp16_f32out (best-of-3 warm) %.2f ms, %.1f GOP/s  [%+.1f%% wall]\n",
           ms32, gops/(ms32/1000.0), ms16>0 ? (ms32-ms16)/ms16*100.0 : 0.0);
    printf("  readback: fp16-out %.2f MB vs fp32-out %.2f MB (2x); weight pack %.2f MB "
           "(out/wt ratio %.2f)\n", out_mb_16, out_mb_32, wt_mb, wt_mb>0 ? (double)M*N*2/((double)N*K*2) : 0.0);

    /* sample-based exact fp64 reference over the fp16 inputs. */
    size_t MN = (size_t)M*N;
    size_t stride = MN > (size_t)samples ? MN/(size_t)samples : 1;
    int checked=0, nonfin=0, shown=0;
    double max_ref=0, abs16=0, abs32=0;     /* track per-path max abs error + max|ref| */
    for (size_t idx=0; idx<MN; idx+=stride) {
        int m=(int)(idx/N), n=(int)(idx%N);
        double ref=0;
        for (int k=0;k<K;k++) ref += (double)A[(size_t)m*K+k] * (double)B[(size_t)n*K+k];
        double c16=(double)C16[idx], c32=(double)C32[idx];
        if (!isfinite(c16)||!isfinite(c32)) { nonfin++; if(shown<8){printf("  nonfinite m=%d n=%d\n",m,n);shown++;} continue; }
        double a16=fabs(c16-ref), a32=fabs(c32-ref);
        if (a16>abs16) abs16=a16;
        if (a32>abs32) abs32=a32;
        if (fabs(ref)>max_ref) max_ref=fabs(ref);
        checked++;
    }
    double norm16 = max_ref>0 ? abs16/max_ref : abs16;
    double norm32 = max_ref>0 ? abs32/max_ref : abs32;
    double improve = norm32>0 ? norm16/norm32 : (norm16>0 ? 1e9 : 1.0);
    printf("checked %d samples (stride %zu) | max|ref|=%.4g\n", checked, stride, max_ref);
    printf("  fp16-out norm_err = %.4g\n", norm16);
    printf("  fp32-out norm_err = %.4g   (%.1fx more accurate than fp16-out)\n", norm32, improve);

    int pass = (nonfin==0) && (norm32 < TOL32) && (improve >= IMPROVE);
    if (!pass) {
        if (nonfin)            printf("  FAIL: %d nonfinite outputs\n", nonfin);
        if (norm32 >= TOL32)   printf("  FAIL: fp32-out norm_err %.4g >= TOL32 %.4g (tiling/readback bug?)\n", norm32, TOL32);
        if (improve < IMPROVE) printf("  FAIL: fp32-out only %.1fx better than fp16-out (< %.1fx) — is fp32tofp16=0 taking effect?\n", improve, IMPROVE);
    }
    printf("==> %s (fp32-out %s; norm32=%.4g, %.1fx vs fp16-out)\n",
           pass?"PASS":"FAIL",
           pass?"tracks the fp64 reference and beats fp16-out":"see failures above",
           norm32, improve);

    free(A);free(B);free(C16);free(C32);
    return pass ? 0 : -1;
}
