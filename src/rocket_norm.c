// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_norm.c — on-NPU transformer normalization (RMSNorm) + the per-row broadcast
 * scale primitive, composed from the feature-axis reduce + the DPU elementwise multiply.
 * See rocket_norm.h for the cost model and the NPU/host work split.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_norm.h"
#include "rocket_reduce.h"      /* rocket_reduce_feature_fp16 (the H-contraction)        */
#include "rocket_activation.h"  /* rocket_ew_mul_fp16 (the DPU elementwise multiply)     */

/* ============================================================================
 * SECTION — per-row broadcast multiply
 * ==========================================================================*/

void rocket_scale_rows_ref_fp16(int M, int N,
                                const _Float16 *in, const float *r, _Float16 *out)
{
    for (int m = 0; m < M; m++) {
        float rm = r[m];
        const _Float16 *ip = in + (size_t)m * N;
        _Float16 *op = out + (size_t)m * N;
        for (int n = 0; n < N; n++) op[n] = (_Float16)((float)ip[n] * rm);
    }
}

int rocket_scale_rows_fp16(int fd, int M, int N,
                           const _Float16 *in, const float *r, _Float16 *out)
{
    if (M < 1 || N < 1) return -1;
    if (fd < 0) { rocket_scale_rows_ref_fp16(M, N, in, r, out); return 0; }

    /* materialize the per-row scalar across the columns: rb[m][n] = (fp16)r[m].
     * The broadcast is a pure host fill (no arithmetic) — the DPU does the multiply. */
    size_t MN = (size_t)M * N;
    _Float16 *rb = malloc(MN * sizeof(_Float16));
    if (!rb) return -2;
    for (int m = 0; m < M; m++) {
        _Float16 rv = (_Float16)r[m];
        _Float16 *row = rb + (size_t)m * N;
        for (int n = 0; n < N; n++) row[n] = rv;
    }
    int rc = rocket_ew_mul_fp16(fd, in, rb, out, (int)MN);
    free(rb);
    return rc;
}

/* ============================================================================
 * SECTION — RMSNorm
 * ==========================================================================*/

void rocket_rmsnorm_ref_fp16(int M, int H, const _Float16 *x,
                             const _Float16 *weight, float eps, _Float16 *out)
{
    const double invH = 1.0 / (double)H;
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = x + (size_t)m * H;
        double ss = 0.0;
        for (int h = 0; h < H; h++) { double v = (double)xp[h]; ss += v * v; }
        double r = 1.0 / sqrt(ss * invH + (double)eps);
        _Float16 *op = out + (size_t)m * H;
        for (int h = 0; h < H; h++)
            op[h] = (_Float16)((double)xp[h] * r * (double)weight[h]);
    }
}

int rocket_rmsnorm_fp16(int fd, int M, int H, const _Float16 *x,
                        const _Float16 *weight, float eps, _Float16 *out)
{
    if (M < 1 || H < 1) return -1;
    if (fd < 0) { rocket_rmsnorm_ref_fp16(M, H, x, weight, eps, out); return 0; }

    const size_t MH = (size_t)M * H;
    int rc = -2;
    _Float16 *xs_buf = NULL, *sq = NULL, *s = NULL;
    float *ms = NULL, *rrow = NULL;

    /* 1. fp16-square overflow guard: scale x by p = 2^-k so (|x|max * p)^2 stays well under
     *    fp16 max (~65504); k=0 for the common |x|<=223 case (no copy, x used directly). */
    float amax = 0.f;
    for (size_t i = 0; i < MH; i++) { float a = fabsf((float)x[i]); if (a > amax) amax = a; }
    int k = (amax > 223.f) ? (int)ceilf(log2f(amax / 223.f)) : 0;
    const float p = ldexpf(1.f, -k);            /* 2^-k exact */
    const _Float16 *xs = x;
    if (k > 0) {
        xs_buf = malloc(MH * sizeof(_Float16));
        if (!xs_buf) goto out;
        for (size_t i = 0; i < MH; i++) xs_buf[i] = (_Float16)((float)x[i] * p);
        xs = xs_buf;
    }

    sq   = malloc(MH * sizeof(_Float16));
    ms   = malloc((size_t)M * sizeof(float));
    rrow = malloc((size_t)M * sizeof(float));
    s    = malloc(MH * sizeof(_Float16));
    if (!sq || !ms || !rrow || !s) goto out;

    /* 2. sq = xs (x) xs  (on NPU) */
    if ((rc = rocket_ew_mul_fp16(fd, xs, xs, sq, (int)MH)) != 0) goto out;

    /* 3. ms_scaled[m] = mean_h sq  (on NPU, fp32 accumulate); true ms = ms_scaled * 4^k */
    if ((rc = rocket_reduce_feature_fp16(fd, M, H, sq, ms, 1)) != 0) goto out;
    const float four_k = ldexpf(1.f, 2 * k);    /* 4^k exact */

    /* 4. per-row r[m] = 1/sqrt(ms+eps)  (host, exact); combined scale s[m][h] = r[m]*weight[h] */
    for (int m = 0; m < M; m++) rrow[m] = 1.f / sqrtf(ms[m] * four_k + eps);
    for (int m = 0; m < M; m++) {
        float rm = rrow[m];
        _Float16 *srow = s + (size_t)m * H;
        for (int h = 0; h < H; h++) srow[h] = (_Float16)(rm * (float)weight[h]);
    }

    /* 5. out = x (x) s  (on NPU). The per-row r and per-column weight are pre-combined into
     *    s here for one ew_mul; the standalone per-row primitive is rocket_scale_rows_fp16,
     *    and in-model both fold into the adjacent matmul (weight->W', r->post-scale). */
    rc = rocket_ew_mul_fp16(fd, x, s, out, (int)MH);

out:
    free(s); free(rrow); free(ms); free(sq); free(xs_buf);
    return rc;
}

/* ============================================================================
 * SECTION — LayerNorm
 * ==========================================================================*/

void rocket_layernorm_ref_fp16(int M, int H, const _Float16 *x, const _Float16 *gamma,
                               const _Float16 *beta, float eps, _Float16 *out)
{
    const double invH = 1.0 / (double)H;
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = x + (size_t)m * H;
        double sx = 0.0, sx2 = 0.0;
        for (int h = 0; h < H; h++) { double v = (double)xp[h]; sx += v; sx2 += v * v; }
        double mean = sx * invH;
        double var  = sx2 * invH - mean * mean;
        if (var < 0.0) var = 0.0;
        double r = 1.0 / sqrt(var + (double)eps);
        _Float16 *op = out + (size_t)m * H;
        for (int h = 0; h < H; h++) {
            double g = (double)gamma[h], b = beta ? (double)beta[h] : 0.0;
            op[h] = (_Float16)(((double)xp[h] - mean) * r * g + b);
        }
    }
}

int rocket_layernorm_fp16(int fd, int M, int H, const _Float16 *x, const _Float16 *gamma,
                          const _Float16 *beta, float eps, _Float16 *out)
{
    if (M < 1 || H < 1) return -1;
    if (fd < 0) { rocket_layernorm_ref_fp16(M, H, x, gamma, beta, eps, out); return 0; }

    const size_t MH = (size_t)M * H;
    int rc = -2;
    _Float16 *xs_buf = NULL, *sq = NULL, *stack = NULL, *A = NULL, *B = NULL, *tmp = NULL;
    float *csum = NULL, *mean = NULL, *rrow = NULL;

    /* fp16-square overflow guard (matches RMSNorm): prescale x by p=2^-k for the x^2 branch
     * only — the mean(x) branch uses x directly (no square, no overflow). */
    float amax = 0.f;
    for (size_t i = 0; i < MH; i++) { float a = fabsf((float)x[i]); if (a > amax) amax = a; }
    int k = (amax > 223.f) ? (int)ceilf(log2f(amax / 223.f)) : 0;
    const float p = ldexpf(1.f, -k);
    const _Float16 *xs = x;
    if (k > 0) {
        xs_buf = malloc(MH * sizeof(_Float16));
        if (!xs_buf) goto out;
        for (size_t i = 0; i < MH; i++) xs_buf[i] = (_Float16)((float)x[i] * p);
        xs = xs_buf;
    }

    sq    = malloc(MH * sizeof(_Float16));
    stack = malloc(2 * MH * sizeof(_Float16));
    csum  = malloc((size_t)2 * M * sizeof(float));
    mean  = malloc((size_t)M * sizeof(float));
    rrow  = malloc((size_t)M * sizeof(float));
    A     = malloc(MH * sizeof(_Float16));
    B     = malloc(MH * sizeof(_Float16));
    tmp   = malloc(MH * sizeof(_Float16));
    if (!sq || !stack || !csum || !mean || !rrow || !A || !B || !tmp) goto out;

    /* 1. sq = xs (x) xs  (on NPU). */
    if ((rc = rocket_ew_mul_fp16(fd, xs, xs, sq, (int)MH)) != 0) goto out;

    /* 2. ONE stacked reduce: A_stack = [x ; sq] (2M rows). Row m -> sum_h x; row M+m -> sum_h sq.
     *    (Two reductions, one job — the documented LayerNorm optimization.) */
    memcpy(stack,            x,  MH * sizeof(_Float16));
    memcpy(stack + MH,       sq, MH * sizeof(_Float16));
    if ((rc = rocket_reduce_feature_fp16(fd, 2 * M, H, stack, csum, 0 /*sum*/)) != 0) goto out;

    /* 3. per-row mean/var/rsqrt (host, exact). sum(x^2) = csum[M+m] * 4^k (undo the prescale). */
    const float invH   = 1.f / (float)H;
    const float four_k = ldexpf(1.f, 2 * k);
    for (int m = 0; m < M; m++) {
        float mn  = csum[m] * invH;
        float msq = csum[M + m] * four_k * invH;
        float var = msq - mn * mn;
        if (var < 0.f) var = 0.f;
        mean[m] = mn;
        rrow[m] = 1.f / sqrtf(var + eps);
    }

    /* 4. fold the affine: out = x (x) A + B, A[m][h]=r[m]*gamma[h],
     *    B[m][h]=beta[h]-mean[m]*r[m]*gamma[h]  (so (x-mean)*r*gamma+beta == x*A+B). */
    for (int m = 0; m < M; m++) {
        float rm = rrow[m], mm = mean[m];
        _Float16 *Ar = A + (size_t)m * H, *Br = B + (size_t)m * H;
        for (int h = 0; h < H; h++) {
            float g = (float)gamma[h], a = rm * g;
            Ar[h] = (_Float16)a;
            Br[h] = (_Float16)((beta ? (float)beta[h] : 0.f) - mm * a);
        }
    }

    /* 5. tmp = x (x) A ; out = tmp + B   (two NPU ew passes). */
    if ((rc = rocket_ew_mul_fp16(fd, x, A, tmp, (int)MH)) != 0) goto out;
    rc = rocket_ew_add_fp16(fd, tmp, B, out, (int)MH);

out:
    free(tmp); free(B); free(A); free(rrow); free(mean); free(csum); free(stack); free(sq); free(xs_buf);
    return rc;
}
