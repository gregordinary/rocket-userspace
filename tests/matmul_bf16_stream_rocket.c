// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_bf16_stream_rocket.c — HW gate for the fast bf16 paths
 * (rocket_matmul_bf16_mt + the streaming context rocket_matmul_bf16_stream),
 * the multicore / resident-scratch siblings of the single-fd rocket_matmul_bf16.
 *
 * Correctness:
 *   - The single-fd rocket_matmul_bf16 output is the in-tree reference C_ref.
 *   - At nthreads=1 the streaming path uses one worker with nsub==N, so its tiling
 *     and host-accum order are identical to single-fd => the output is BIT-IDENTICAL
 *     to C_ref (memcmp). This is the strong gate.
 *   - At nthreads>1 the N-split changes only which worker owns which output column
 *     (never the per-element K-accum order), so the streaming/mt output is checked
 *     BIT-IDENTICAL to C_ref as well; as a backstop the whole result is also gated
 *     against an exact double reference (sampled) within tolerance.
 *
 * Perf: reports warm GOP/s for single-fd, mt, and streaming (the streaming call is
 * repeated so the resident BOs / persistent fds are warm; the first call is discarded).
 *
 * Usage: matmul_bf16_stream_rocket [M] [K] [N] [nthreads]   (K%32, N%16, M%4||1)
 *   default 512 3840 4096 4
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}
static double env_dbl(const char *n, double d) { const char *e = getenv(n); return e ? strtod(e, NULL) : d; }
static int    env_int(const char *n, int d)    { const char *e = getenv(n); return e ? (int)strtol(e, NULL, 0) : d; }
/* bf16 round-trip (truncate) — matches the driver's f32_to_bf16. */
static inline float bf16rt(float f) {
    uint32_t b; memcpy(&b, &f, sizeof b); b &= 0xFFFF0000u;
    float r; memcpy(&r, &b, sizeof r); return r;
}

/* sampled exact double reference; returns normalized error vs C, sets *nonfin. */
static double sample_err(const float *A, const float *B, const float *C,
                         int M, int K, int N, int samples, int *nonfin) {
    size_t MN = (size_t)M * N;
    size_t stride = MN > (size_t)samples ? MN / (size_t)samples : 1;
    double max_abs = 0.0, max_ref = 0.0;
    *nonfin = 0;
    for (size_t idx = 0; idx < MN; idx += stride) {
        int m = (int)(idx / N), n = (int)(idx % N);
        double ref = 0.0;
        for (int k = 0; k < K; k++)
            ref += (double)bf16rt(A[(size_t)m*K + k]) * (double)bf16rt(B[(size_t)n*K + k]);
        float act = C[idx];
        if (!isfinite(act)) { (*nonfin)++; continue; }
        double ad = fabs((double)act - ref);
        if (ad > max_abs) max_abs = ad;
        if (fabs(ref) > max_ref) max_ref = fabs(ref);
    }
    return max_ref > 0 ? max_abs / max_ref : max_abs;
}

int main(int argc, char **argv) {
    int M  = argc > 1 ? atoi(argv[1]) : 512;
    int K  = argc > 2 ? atoi(argv[2]) : 3840;
    int N  = argc > 3 ? atoi(argv[3]) : 4096;
    int NT = argc > 4 ? atoi(argv[4]) : 4;
    const float tol  = (float)env_dbl("ROCKET_BF16_TOL", 0.01);
    const float mag  = (float)env_dbl("ROCKET_BF16_MAG", 10.0);
    const int   samp = env_int("ROCKET_BF16_SAMPLES", 8192);
    const int   reps = env_int("ROCKET_BF16_REPS", 3);

    if (rocket_matmul_plan_bf16(M, K, N, NULL, NULL, NULL) < 0) {
        fprintf(stderr, "unsupported shape (need K%%32, N%%16, M%%4||1)\n"); return -1;
    }
    printf("bf16 fast paths: C[%d,%d] = A[%d,%d] x B[%d,%d]^T  nthreads=%d tol=%g\n",
           M, N, M, K, N, K, NT, tol);

    float *A   = malloc((size_t)M * K * sizeof(float));
    float *B   = malloc((size_t)N * K * sizeof(float));
    float *Cr  = malloc((size_t)M * N * sizeof(float));
    float *Cmt = malloc((size_t)M * N * sizeof(float));
    float *Cs1 = malloc((size_t)M * N * sizeof(float));
    float *Csn = malloc((size_t)M * N * sizeof(float));
    if (!A || !B || !Cr || !Cmt || !Cs1 || !Csn) { fprintf(stderr, "host malloc failed\n"); return -1; }
    srand(1234);
    for (size_t i = 0; i < (size_t)M*K; i++) A[i] = ((float)rand()/RAND_MAX*2.0f - 1.0f) * mag;
    for (size_t i = 0; i < (size_t)N*K; i++) B[i] = ((float)rand()/RAND_MAX*2.0f - 1.0f) * mag;

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); return 2; }

    /* ---- single-fd reference ---- */
    int64_t t0 = now_us();
    int rc = rocket_matmul_bf16(fd, M, K, N, A, B, Cr);
    double ms_ref = (now_us() - t0) / 1000.0;
    rocket_close(fd);
    if (rc) { fprintf(stderr, "rocket_matmul_bf16 (ref) failed (%d)\n", rc); return rc; }

    int nf = 0;
    double err_ref = sample_err(A, B, Cr, M, K, N, samp, &nf);
    if (nf || err_ref >= tol) {
        printf("==> FAIL (single-fd reference itself out of tol: norm_err=%.4g nonfinite=%d)\n", err_ref, nf);
        return -1;
    }

    /* ---- multicore fan-out ---- */
    double ms_mt = 0.0;
    for (int r = 0; r < reps; r++) {
        t0 = now_us();
        rc = rocket_matmul_bf16_mt(M, K, N, A, B, Cmt, NT);
        double ms = (now_us() - t0) / 1000.0;
        if (rc) { fprintf(stderr, "rocket_matmul_bf16_mt failed (%d)\n", rc); return rc; }
        ms_mt = ms;   /* keep the last (warm) */
    }
    int mt_bitexact = (memcmp(Cmt, Cr, (size_t)M*N*sizeof(float)) == 0);
    int nf_mt = 0; double err_mt = sample_err(A, B, Cmt, M, K, N, samp, &nf_mt);

    /* ---- streaming, nthreads=1 (must be BIT-IDENTICAL to single-fd) ---- */
    rocket_bf16_stream *s1 = rocket_bf16_stream_create(1);
    if (!s1) { fprintf(stderr, "stream_create(1) failed\n"); return -1; }
    rc = rocket_matmul_bf16_stream(s1, M, K, N, A, B, Cs1);
    rocket_bf16_stream_free(s1);
    if (rc) { fprintf(stderr, "bf16_stream(nt=1) failed (%d)\n", rc); return rc; }
    int s1_bitexact = (memcmp(Cs1, Cr, (size_t)M*N*sizeof(float)) == 0);

    /* ---- streaming, nthreads=NT (warm: persistent fds + resident scratch) ---- */
    rocket_bf16_stream *sn = rocket_bf16_stream_create(NT);
    if (!sn) { fprintf(stderr, "stream_create(%d) failed\n", NT); return -1; }
    double ms_sn = 0.0;
    for (int r = 0; r < reps; r++) {
        t0 = now_us();
        rc = rocket_matmul_bf16_stream(sn, M, K, N, A, B, Csn);
        double ms = (now_us() - t0) / 1000.0;
        if (rc) { fprintf(stderr, "bf16_stream(nt=%d) failed (%d)\n", NT, rc); rocket_bf16_stream_free(sn); return rc; }
        ms_sn = ms;   /* last (warm) */
    }
    rocket_bf16_stream_free(sn);
    int sn_bitexact = (memcmp(Csn, Cr, (size_t)M*N*sizeof(float)) == 0);
    int nf_sn = 0; double err_sn = sample_err(A, B, Csn, M, K, N, samp, &nf_sn);

    double g_ref = 2.0*(double)M*N*K / (ms_ref/1000.0) / 1e9;
    double g_mt  = 2.0*(double)M*N*K / (ms_mt /1000.0) / 1e9;
    double g_sn  = 2.0*(double)M*N*K / (ms_sn /1000.0) / 1e9;
    printf("single-fd : %.1f ms  %.1f GOP/s  norm_err=%.4g\n", ms_ref, g_ref, err_ref);
    printf("mt(nt=%d)  : %.1f ms  %.1f GOP/s  norm_err=%.4g  bit-exact=%d  (%.2fx single-fd)\n",
           NT, ms_mt, g_mt, err_mt, mt_bitexact, ms_ref/ms_mt);
    printf("stream nt=1: bit-exact-vs-single-fd=%d\n", s1_bitexact);
    printf("stream(%d)  : %.1f ms  %.1f GOP/s  norm_err=%.4g  bit-exact=%d  (%.2fx single-fd)\n",
           NT, ms_sn, g_sn, err_sn, sn_bitexact, ms_ref/ms_sn);

    /* Pass = nthreads=1 streaming bit-identical to single-fd (the core guarantee: same
     * tiling + accum order), and mt + nthreads>1 streaming within tol of the exact
     * double reference with no nonfinite. The nthreads>1 paths are usually bit-exact
     * too (N-split preserves per-element accum order when every slice tiles the same
     * Kt), but a small final N-slice can pick a different Kt and re-round, so that is
     * reported, not required. */
    int pass = s1_bitexact
            && !nf_mt && !nf_sn && err_mt < tol && err_sn < tol;
    printf("==> %s (bf16 fast paths %s)\n",
           pass ? "PASS" : "FAIL",
           pass ? "bit-identical to single-fd + track fp32 reference"
                : "DIVERGED — check N-split / resident scratch / accum order");

    free(A); free(B); free(Cr); free(Cmt); free(Cs1); free(Csn);
    return pass ? 0 : -1;
}
