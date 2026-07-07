// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_softmax.c — row-wise softmax over the last axis, composed from the shipped on-NPU
 * primitives: host row-max + subtract, EXP on the DPU LUT, the feature-axis ones-matmul sum,
 * a host reciprocal, and the per-row broadcast scale. See rocket_softmax.h for the work split
 * and why the row-max is on the host (no max-reduce datapath; the PPU-max is the resident path).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_softmax.h"
#include "rocket_activation.h"   /* ROCKET_ACTIVATION_EXP (the new LUT) */
#include "rocket_reduce.h"       /* rocket_reduce_feature_fp16 (the row-sum) */
#include "rocket_norm.h"         /* rocket_scale_rows_fp16 (the per-row scale) */

/* ============================================================================
 * SECTION — Softmax
 * ==========================================================================*/

void rocket_softmax_ref_fp16(int M, int N, const _Float16 *in, _Float16 *out)
{
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = in + (size_t)m * N;
        _Float16 *op = out + (size_t)m * N;
        double mx = -INFINITY;
        for (int n = 0; n < N; n++) { double v = (double)xp[n]; if (v > mx) mx = v; }
        double s = 0.0;
        for (int n = 0; n < N; n++) s += exp((double)xp[n] - mx);
        double inv = 1.0 / s;
        for (int n = 0; n < N; n++) op[n] = (_Float16)(exp((double)xp[n] - mx) * inv);
    }
}

int rocket_softmax_fp16(int fd, int M, int N, const _Float16 *in, _Float16 *out)
{
    if (M < 1 || N < 1) return -1;
    if (fd < 0) { rocket_softmax_ref_fp16(M, N, in, out); return 0; }

    const size_t MN = (size_t)M * N;
    int rc = -2;
    _Float16 *xs = malloc(MN * sizeof(_Float16));   /* x - rowmax (<=0) */
    _Float16 *e  = malloc(MN * sizeof(_Float16));   /* exp(xs)          */
    float    *s  = malloc((size_t)M * sizeof(float));
    float    *inv= malloc((size_t)M * sizeof(float));
    if (!xs || !e || !s || !inv) goto out;

    /* 1. host row-max + subtract: xs[m][n] = x[m][n] - max_n x[m][n] (<=0, the EXP domain). */
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = in + (size_t)m * N;
        _Float16 *sp = xs + (size_t)m * N;
        float mx = -INFINITY;
        for (int n = 0; n < N; n++) { float v = (float)xp[n]; if (v > mx) mx = v; }
        for (int n = 0; n < N; n++) sp[n] = (_Float16)((float)xp[n] - mx);
    }

    /* 2. e = exp(xs) on the NPU (DPU LUT, default domain [-16,0]; deep tail clamps to ~0). */
    if ((rc = rocket_activation_fp16(fd, ROCKET_ACTIVATION_EXP, xs, e, (int)MN)) != 0) goto out;

    /* 3. s[m] = sum_n e[m][n] on the NPU (feature-axis ones-matmul reduce, fp32 accumulate). */
    if ((rc = rocket_reduce_feature_fp16(fd, M, N, e, s, 0 /*sum*/)) != 0) goto out;

    /* 4. inv[m] = 1/s[m] on the host (exact, O(M)). s>=1 (the max term contributes exp(0)=1). */
    for (int m = 0; m < M; m++) inv[m] = (s[m] > 0.f) ? (1.f / s[m]) : 0.f;

    /* 5. out[m][n] = e[m][n] * inv[m] on the NPU (per-row broadcast scale = ew_mul). */
    rc = rocket_scale_rows_fp16(fd, M, N, e, inv, out);

out:
    free(inv); free(s); free(e); free(xs);
    return rc;
}

/* ============================================================================
 * SECTION — LogSoftmax
 * ==========================================================================*/

void rocket_logsoftmax_ref_fp16(int M, int N, const _Float16 *in, _Float16 *out)
{
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = in + (size_t)m * N;
        _Float16 *op = out + (size_t)m * N;
        double mx = -INFINITY;
        for (int n = 0; n < N; n++) { double v = (double)xp[n]; if (v > mx) mx = v; }
        double s = 0.0;
        for (int n = 0; n < N; n++) s += exp((double)xp[n] - mx);
        double lse = mx + log(s);                 /* logsumexp = mx + log(sum exp(x-mx)) */
        for (int n = 0; n < N; n++) op[n] = (_Float16)((double)xp[n] - lse);
    }
}

int rocket_logsoftmax_fp16(int fd, int M, int N, const _Float16 *in, _Float16 *out)
{
    if (M < 1 || N < 1) return -1;
    if (fd < 0) { rocket_logsoftmax_ref_fp16(M, N, in, out); return 0; }

    const size_t MN = (size_t)M * N;
    int rc = -2;
    _Float16 *xs  = malloc(MN * sizeof(_Float16));   /* x - rowmax (<=0)            */
    _Float16 *e   = malloc(MN * sizeof(_Float16));   /* exp(xs)                     */
    _Float16 *lsb = malloc(MN * sizeof(_Float16));   /* log(s[m]) broadcast over n  */
    float    *s   = malloc((size_t)M * sizeof(float));
    if (!xs || !e || !lsb || !s) goto out;

    /* 1. host row-max + subtract: xs = x - rowmax (<=0, the EXP domain). */
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = in + (size_t)m * N;
        _Float16 *sp = xs + (size_t)m * N;
        float mx = -INFINITY;
        for (int n = 0; n < N; n++) { float v = (float)xp[n]; if (v > mx) mx = v; }
        for (int n = 0; n < N; n++) sp[n] = (_Float16)((float)xp[n] - mx);
    }

    /* 2. e = exp(xs) on the NPU (DPU LUT). 3. s[m] = sum_n e (NPU fp32 reduce). */
    if ((rc = rocket_activation_fp16(fd, ROCKET_ACTIVATION_EXP, xs, e, (int)MN)) != 0) goto out;
    if ((rc = rocket_reduce_feature_fp16(fd, M, N, e, s, 0 /*sum*/)) != 0) goto out;

    /* 4. ls[m] = log(s[m]) on the host (exact, O(M); s>=1 => ls>=0), broadcast across n. */
    for (int m = 0; m < M; m++) {
        float ls = (s[m] > 0.f) ? logf(s[m]) : 0.f;
        _Float16 lv = (_Float16)ls;
        _Float16 *row = lsb + (size_t)m * N;
        for (int n = 0; n < N; n++) row[n] = lv;
    }

    /* 5. out[m][n] = xs[m][n] - ls[m] on the NPU (per-row broadcast subtract = ew_sub).
     *    out = (x - rowmax) - log(s) = x - (rowmax + log(s)) = x - logsumexp. */
    rc = rocket_ew_sub_fp16(fd, xs, lsb, out, (int)MN);

out:
    free(s); free(lsb); free(e); free(xs);
    return rc;
}

/* ============================================================================
 * SECTION — Cross-entropy (logsumexp reduction + host gather)
 * ==========================================================================*/

void rocket_cross_entropy_ref_fp16(int M, int N,
                                   const _Float16 *logits, const int *target, float *loss)
{
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = logits + (size_t)m * N;
        double mx = -INFINITY;
        for (int n = 0; n < N; n++) { double v = (double)xp[n]; if (v > mx) mx = v; }
        double s = 0.0;
        for (int n = 0; n < N; n++) s += exp((double)xp[n] - mx);
        double lse = mx + log(s);                         /* logsumexp = mx + log(sum exp(x-mx)) */
        int t = target[m];
        double g = (t >= 0 && t < N) ? (double)xp[t] : 0.0;
        loss[m] = (float)(lse - g);                       /* CE = logsumexp - logit[target]      */
    }
}

int rocket_cross_entropy_fp16(int fd, int M, int N,
                              const _Float16 *logits, const int *target, float *loss)
{
    if (M < 1 || N < 1) return -1;
    if (fd < 0) { rocket_cross_entropy_ref_fp16(M, N, logits, target, loss); return 0; }

    const size_t MN = (size_t)M * N;
    int rc = -2;
    _Float16 *xs  = malloc(MN * sizeof(_Float16));        /* logits - rowmax (<=0)  */
    _Float16 *e   = malloc(MN * sizeof(_Float16));        /* exp(xs)                */
    float    *s   = malloc((size_t)M * sizeof(float));    /* row sums of e (NPU)    */
    float    *rmx = malloc((size_t)M * sizeof(float));    /* per-row max (kept for lse) */
    if (!xs || !e || !s || !rmx) goto out;

    /* 1. host row-max + subtract: xs = logits - rowmax (<=0, the EXP domain). Keep rowmax for lse. */
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = logits + (size_t)m * N;
        _Float16 *sp = xs + (size_t)m * N;
        float mx = -INFINITY;
        for (int n = 0; n < N; n++) { float v = (float)xp[n]; if (v > mx) mx = v; }
        rmx[m] = mx;
        for (int n = 0; n < N; n++) sp[n] = (_Float16)((float)xp[n] - mx);
    }

    /* 2. e = exp(xs) on the NPU (DPU LUT). 3. s[m] = sum_n e (NPU fp32 reduce). */
    if ((rc = rocket_activation_fp16(fd, ROCKET_ACTIVATION_EXP, xs, e, (int)MN)) != 0) goto out;
    if ((rc = rocket_reduce_feature_fp16(fd, M, N, e, s, 0 /*sum*/)) != 0) goto out;

    /* 4. lse[m] = rowmax[m] + log(s[m])   (host, exact, O(M); s>=1 => log(s)>=0).
     * 5. CE[m] = lse[m] - logits[m][target[m]]   (HOST GATHER + subtract; no HW gather exists). */
    for (int m = 0; m < M; m++) {
        float lse = rmx[m] + ((s[m] > 0.f) ? logf(s[m]) : 0.f);
        int t = target[m];
        float g = (t >= 0 && t < N) ? (float)logits[(size_t)m * N + t] : 0.f;
        loss[m] = lse - g;
    }
    rc = 0;

out:
    free(rmx); free(s); free(e); free(xs);
    return rc;
}
