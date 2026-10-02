// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * softmax_exp_bench.c — what share of the flash-attention handler's host softmax is expf,
 * and what a vectorised exp would take out of it.
 *
 * A CPU-only bench, no NPU. The handler's softmax bucket is capped at 6.00% of the pinned
 * prefill wall on gemma4-12b F16 at pp2048, but that bucket is the whole row loop: the row
 * maximum, the exp pass with its fp16 conversions, and the normalize pass. This splits the loop
 * on one core, on the handler's own code, at row lengths around the KV length:
 *
 *   ref      host_softmax_rows exactly as src/rocket_attn.c has it (scalar expf)
 *   noexp    the same loop with expf replaced by its argument: the floor of everything but exp
 *   nexp     the same loop with the exp pass in NEON, a degree-5 polynomial exp, 4 lanes
 *   neon     the whole row in NEON: max, exp pass and normalize
 *
 * Each arm's time is the median over reps of one pass over M rows; the process pins itself to
 * one CPU (argument, default 4, an A76 on the RK3588). Accuracy is scored against `ref` in
 * fp16 units in the last place over every output, and a masked score (-30000, as the handler
 * clamps them) must still give exactly zero.
 *
 * What it does NOT show: the loop under the handler's four to five concurrent workers and the
 * device beside them, the other buckets' response, or the model's output with a different exp.
 * The share it measures is of the loop's single-thread time; the bucket also counts descheduled
 * time, so a fraction of the 6.00% it gives is an upper bound.
 *
 * Usage: softmax_exp_bench [cpu] [reps]
 * Exit: 0.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <arm_neon.h>

#include "test_fill.h"

/* ---- ref: src/rocket_attn.c host_softmax_rows, verbatim ---- */
static void host_softmax_rows(int M, int N, const _Float16 *in, _Float16 *out){
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = in + (size_t)m*N; _Float16 *op = out + (size_t)m*N;
        float mx = -INFINITY; for (int n=0;n<N;n++){ float v=(float)xp[n]; if (v>mx) mx=v; }
        float s = 0.f; for (int n=0;n<N;n++){ float e=expf((float)xp[n]-mx); op[n]=(_Float16)e; s+=e; }
        float inv = s>0.f ? 1.f/s : 0.f; for (int n=0;n<N;n++) op[n]=(_Float16)((float)op[n]*inv);
    }
}

/* ---- noexp: the same loop, the exp replaced by its argument ---- */
static void noexp_rows(int M, int N, const _Float16 *in, _Float16 *out){
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = in + (size_t)m*N; _Float16 *op = out + (size_t)m*N;
        float mx = -INFINITY; for (int n=0;n<N;n++){ float v=(float)xp[n]; if (v>mx) mx=v; }
        float s = 0.f; for (int n=0;n<N;n++){ float e=((float)xp[n]-mx); op[n]=(_Float16)e; s+=e; }
        float inv = s!=0.f ? 1.f/s : 0.f; for (int n=0;n<N;n++) op[n]=(_Float16)((float)op[n]*inv);
    }
}

/* exp(x) for 4 lanes: n = round(x log2 e), r = x - n ln 2 in two parts, a degree-5 Taylor
 * polynomial on r (|r| <= ln2/2), scaled by 2^n through the exponent field. Relative error
 * at most about 2.4e-6 (the r^6/720 term), against an fp16 ulp of 4.9e-4, so it moves an
 * fp16 output only where the exact value sits that close to a rounding boundary. Clamped to
 * [-87, 88]. */
static inline float32x4_t exp_f32x4(float32x4_t x)
{
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.0f)), vdupq_n_f32(-87.0f));
    const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.4426950408889634f));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693145751953125f));
    r = vfmsq_f32(r, n, vdupq_n_f32(1.428606765330187e-06f));
    float32x4_t p = vdupq_n_f32(1.0f / 120.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    const int32x4_t e = vshlq_n_s32(vcvtq_s32_f32(n), 23);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), e));
}

static inline float32x4_t ld4(const _Float16 *p) { return vcvt_f32_f16(vld1_f16((const float16_t *)p)); }
static inline void st4(_Float16 *p, float32x4_t v) { vst1_f16((float16_t *)p, vcvt_f16_f32(v)); }

/* ---- nexp: the exp pass in NEON, the max and normalize passes as in ref ---- */
static void nexp_rows(int M, int N, const _Float16 *in, _Float16 *out){
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = in + (size_t)m*N; _Float16 *op = out + (size_t)m*N;
        float mx = -INFINITY; for (int n=0;n<N;n++){ float v=(float)xp[n]; if (v>mx) mx=v; }
        float32x4_t acc = vdupq_n_f32(0), vm = vdupq_n_f32(mx);
        int n = 0;
        for (; n + 4 <= N; n += 4) {
            float32x4_t e = exp_f32x4(vsubq_f32(ld4(xp + n), vm));
            st4(op + n, e);
            acc = vaddq_f32(acc, e);
        }
        float s = vaddvq_f32(acc);
        for (; n < N; n++) { float e = expf((float)xp[n] - mx); op[n] = (_Float16)e; s += e; }
        float inv = s>0.f ? 1.f/s : 0.f; for (int k=0;k<N;k++) op[k]=(_Float16)((float)op[k]*inv);
    }
}

/* ---- neon: max, exp pass and normalize all in NEON ---- */
static void neon_rows(int M, int N, const _Float16 *in, _Float16 *out){
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = in + (size_t)m*N; _Float16 *op = out + (size_t)m*N;
        float32x4_t vmx = vdupq_n_f32(-INFINITY);
        int n = 0;
        for (; n + 4 <= N; n += 4) vmx = vmaxq_f32(vmx, ld4(xp + n));
        float mx = vmaxvq_f32(vmx);
        for (; n < N; n++) { float v = (float)xp[n]; if (v > mx) mx = v; }
        float32x4_t acc = vdupq_n_f32(0), vm = vdupq_n_f32(mx);
        for (n = 0; n + 4 <= N; n += 4) {
            float32x4_t e = exp_f32x4(vsubq_f32(ld4(xp + n), vm));
            st4(op + n, e);
            acc = vaddq_f32(acc, e);
        }
        float s = vaddvq_f32(acc);
        for (; n < N; n++) { float e = expf((float)xp[n] - mx); op[n] = (_Float16)e; s += e; }
        const float inv = s>0.f ? 1.f/s : 0.f;
        for (n = 0; n + 4 <= N; n += 4) st4(op + n, vmulq_n_f32(ld4(op + n), inv));
        for (; n < N; n++) op[n] = (_Float16)((float)op[n] * inv);
    }
}

typedef void (*rows_fn)(int, int, const _Float16 *, _Float16 *);

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static int ulp_diff(_Float16 a, _Float16 b)
{
    int16_t x, y;
    memcpy(&x, &a, 2); memcpy(&y, &b, 2);
    if (x < 0) x = (int16_t)(0x8000 - x);
    if (y < 0) y = (int16_t)(0x8000 - y);
    return abs(x - y);
}

int main(int argc, char **argv)
{
    const int cpu = argc > 1 ? atoi(argv[1]) : 4, reps = argc > 2 ? atoi(argv[2]) : 31;
    cpu_set_t set;
    CPU_ZERO(&set); CPU_SET(cpu, &set);
    const int pinned = sched_setaffinity(0, sizeof set, &set) == 0;
    static const int Ns[] = { 512, 1024, 2048, 4096 };
    const rows_fn fns[] = { host_softmax_rows, noexp_rows, nexp_rows, neon_rows };
    const char *names[] = { "ref", "noexp", "nexp", "neon" };
    printf("== host softmax split, one thread %s cpu %d, %d reps, median ns per element ==\n",
           pinned ? "pinned to" : "NOT pinned to", cpu, reps);
    for (unsigned t = 0; t < sizeof Ns / sizeof Ns[0]; t++) {
        const int N = Ns[t], M = (1 << 21) / N;              /* 2 Mi elements a pass, 4 MiB */
        const size_t ne = (size_t)M * N;
        _Float16 *in = malloc(ne * 2), *ref = malloc(ne * 2), *out = malloc(ne * 2);
        double *tm = malloc((size_t)reps * sizeof(double)), med[4];
        /* scores as the handler sees them after scale and mask: most in [-12, 12], and one
         * column in eight masked to -30000 */
        tf_fill_f16(in, ne, 0x7100 + (uint64_t)t, -12.0, 12.0);
        for (size_t i = 0; i < ne; i += 8) in[i] = (_Float16)-30000.0f;
        host_softmax_rows(M, N, in, ref);
        for (int f = 0; f < 4; f++) {
            fns[f](M, N, in, out);                           /* warm */
            for (int r = 0; r < reps; r++) {
                const double t0 = now_ns();
                fns[f](M, N, in, out);
                tm[r] = now_ns() - t0;
            }
            qsort(tm, (size_t)reps, sizeof(double), cmp_d);
            med[f] = tm[reps / 2] / (double)ne;
            long diff = 0, big = 0, masked_nonzero = 0;
            int worst = 0;
            if (f != 1) {
                for (size_t i = 0; i < ne; i++) {
                    const int u = ulp_diff(out[i], ref[i]);
                    if (u) diff++;
                    if (u > 1) big++;
                    if (u > worst) worst = u;
                    if (i % 8 == 0 && (float)out[i] != 0.0f) masked_nonzero++;
                }
            }
            printf("  N %4d  %-5s %7.3f ns/elem  (%.3f of ref)", N, names[f], med[f], med[f] / med[0]);
            if (f != 1)
                printf("  vs ref: %ld of %zu differ, %ld by >1 ulp, worst %d ulp, %ld masked not zero",
                       diff, ne, big, worst, masked_nonzero);
            printf("\n");
        }
        printf("  N %4d  exp's share of the loop (1 - noexp/ref) %.3f; nexp takes %.3f of it, neon %.3f\n",
               N, 1.0 - med[1] / med[0], 1.0 - med[2] / med[0], 1.0 - med[3] / med[0]);
        free(in); free(ref); free(out); free(tm);
    }
    return 0;
}
