// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_normvision.c — the on-NPU VISION normalization family (BatchNorm / GroupNorm /
 * InstanceNorm / L2-Normalize), composed from the feature-axis reduce + the DPU elementwise
 * path, exactly like the transformer norms in rocket_norm.c. See rocket_normvision.h for the
 * cost model, layout, and the NPU/host work split.
 *
 * The four ops differ only in WHICH axis is reduced and HOW the affine broadcasts:
 *   - BatchNorm    : no reduce; per-channel affine from precomputed running stats.
 *   - GroupNorm    : reduce [C/G,H,W] per (n,g); per-CHANNEL affine (full broadcast A/B).
 *   - InstanceNorm : GroupNorm with G=C (per (n,c) reduce; affine is then per-row).
 *   - L2-Normalize : reduce x^2 per row; per-row scale.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_normvision.h"
#include "rocket_norm.h"        /* rocket_scale_rows_fp16 (per-row broadcast multiply)   */
#include "rocket_reduce.h"      /* rocket_reduce_feature_fp16 (the group/row contraction) */
#include "rocket_activation.h"  /* rocket_ew_mul_fp16 / rocket_ew_add_fp16               */

/* fp16-square overflow guard, shared with RMSNorm/LayerNorm: pick k so (|x|max * 2^-k)^2
 * stays well under fp16 max (~65504). 223^2 ~ 49729 < 65504, so the threshold is 223; the
 * recovered mean-square is (reduced sum of the prescaled squares) * 4^k. k==0 (the common
 * case) means x is used directly with no copy. */
static int square_prescale_k(const _Float16 *x, size_t n)
{
    float amax = 0.f;
    for (size_t i = 0; i < n; i++) { float a = fabsf((float)x[i]); if (a > amax) amax = a; }
    return (amax > 223.f) ? (int)ceilf(log2f(amax / 223.f)) : 0;
}

/* ============================================================================
 * SECTION — BatchNorm (inference)
 * ==========================================================================*/

void rocket_batchnorm_ref_fp16(int N, int C, int P, const _Float16 *x,
                               const _Float16 *gamma, const _Float16 *beta,
                               const _Float16 *mean, const _Float16 *var,
                               float eps, _Float16 *out)
{
    for (int c = 0; c < C; c++) {
        double g = gamma ? (double)gamma[c] : 1.0;
        double b = beta  ? (double)beta[c]  : 0.0;
        double s = g / sqrt((double)var[c] + (double)eps);
        double bc = b - (double)mean[c] * s;
        for (int n = 0; n < N; n++) {
            const _Float16 *xp = x   + ((size_t)n * C + c) * P;
            _Float16       *op = out + ((size_t)n * C + c) * P;
            for (int p = 0; p < P; p++) op[p] = (_Float16)((double)xp[p] * s + bc);
        }
    }
}

int rocket_batchnorm_fp16(int fd, int N, int C, int P, const _Float16 *x,
                          const _Float16 *gamma, const _Float16 *beta,
                          const _Float16 *mean, const _Float16 *var,
                          float eps, _Float16 *out)
{
    if (N < 1 || C < 1 || P < 1 || !x || !mean || !var || !out) return -1;
    if (fd < 0) { rocket_batchnorm_ref_fp16(N, C, P, x, gamma, beta, mean, var, eps, out);
                  return 0; }

    const size_t total = (size_t)N * C * P;
    int rc = -2;
    _Float16 *A = malloc(total * sizeof(_Float16));   /* per-channel scale, broadcast */
    _Float16 *B = malloc(total * sizeof(_Float16));   /* per-channel bias,  broadcast */
    _Float16 *tmp = malloc(total * sizeof(_Float16));
    if (!A || !B || !tmp) goto out;

    /* fold (x-mean)/sqrt(var+eps)*gamma+beta == x*s[c] + b[c], materialized as broadcasts */
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++) {
            float g  = gamma ? (float)gamma[c] : 1.f;
            float bt = beta  ? (float)beta[c]  : 0.f;
            float s  = g / sqrtf((float)var[c] + eps);
            float bc = bt - (float)mean[c] * s;
            _Float16 sv = (_Float16)s, bv = (_Float16)bc;
            _Float16 *Ar = A + ((size_t)n * C + c) * P;
            _Float16 *Br = B + ((size_t)n * C + c) * P;
            for (int p = 0; p < P; p++) { Ar[p] = sv; Br[p] = bv; }
        }

    if ((rc = rocket_ew_mul_fp16(fd, x, A, tmp, (int)total)) != 0) goto out;
    rc = rocket_ew_add_fp16(fd, tmp, B, out, (int)total);

out:
    free(tmp); free(B); free(A);
    return rc;
}

/* ============================================================================
 * SECTION — GroupNorm
 * ==========================================================================*/

void rocket_groupnorm_ref_fp16(int N, int C, int G, int P, const _Float16 *x,
                               const _Float16 *gamma, const _Float16 *beta,
                               float eps, _Float16 *out)
{
    const int Cg = C / G;
    const double invPg = 1.0 / ((double)Cg * (double)P);
    for (int n = 0; n < N; n++)
        for (int g = 0; g < G; g++) {
            /* mean/var over the group's Cg channels * P spatial positions */
            double sx = 0.0, sx2 = 0.0;
            for (int cc = 0; cc < Cg; cc++) {
                int c = g * Cg + cc;
                const _Float16 *xp = x + ((size_t)n * C + c) * P;
                for (int p = 0; p < P; p++) { double v = (double)xp[p]; sx += v; sx2 += v * v; }
            }
            double mean = sx * invPg;
            double var  = sx2 * invPg - mean * mean;
            if (var < 0.0) var = 0.0;
            double r = 1.0 / sqrt(var + (double)eps);
            for (int cc = 0; cc < Cg; cc++) {
                int c = g * Cg + cc;
                double gm = gamma ? (double)gamma[c] : 1.0;
                double bt = beta  ? (double)beta[c]  : 0.0;
                const _Float16 *xp = x   + ((size_t)n * C + c) * P;
                _Float16       *op = out + ((size_t)n * C + c) * P;
                for (int p = 0; p < P; p++)
                    op[p] = (_Float16)(((double)xp[p] - mean) * r * gm + bt);
            }
        }
}

int rocket_groupnorm_fp16(int fd, int N, int C, int G, int P, const _Float16 *x,
                          const _Float16 *gamma, const _Float16 *beta,
                          float eps, _Float16 *out)
{
    if (N < 1 || C < 1 || P < 1 || G < 1 || (C % G) != 0 || !x || !out) return -1;
    if (fd < 0) { rocket_groupnorm_ref_fp16(N, C, G, P, x, gamma, beta, eps, out); return 0; }

    const int Cg = C / G;
    const int M  = N * G;               /* one row per (batch, group)         */
    const int Pg = Cg * P;              /* elements reduced per group row      */
    const size_t total = (size_t)N * C * P;

    int rc = -2;
    _Float16 *xs_buf = NULL, *sq = NULL, *stack = NULL, *A = NULL, *B = NULL, *tmp = NULL;
    float *csum = NULL, *meanv = NULL, *rinv = NULL;

    /* fp16-square overflow prescale for the x^2 branch (the mean(x) branch uses x directly) */
    int k = square_prescale_k(x, total);
    const float p = ldexpf(1.f, -k);
    const _Float16 *xs = x;
    if (k > 0) {
        xs_buf = malloc(total * sizeof(_Float16));
        if (!xs_buf) goto out;
        for (size_t i = 0; i < total; i++) xs_buf[i] = (_Float16)((float)x[i] * p);
        xs = xs_buf;
    }

    sq    = malloc(total * sizeof(_Float16));
    stack = malloc(2 * total * sizeof(_Float16));
    csum  = malloc((size_t)2 * M * sizeof(float));
    meanv = malloc((size_t)M * sizeof(float));
    rinv  = malloc((size_t)M * sizeof(float));
    A     = malloc(total * sizeof(_Float16));
    B     = malloc(total * sizeof(_Float16));
    tmp   = malloc(total * sizeof(_Float16));
    if (!sq || !stack || !csum || !meanv || !rinv || !A || !B || !tmp) goto out;

    /* 1. sq = xs (x) xs (NPU). */
    if ((rc = rocket_ew_mul_fp16(fd, xs, xs, sq, (int)total)) != 0) goto out;

    /* 2. ONE stacked feature-reduce over Pg: [x ; sq] as 2*M rows. The group's Pg elements
     *    are contiguous in [N,C,P], so x viewed as [M][Pg] needs no reorder. */
    memcpy(stack,         x,  total * sizeof(_Float16));
    memcpy(stack + total, sq, total * sizeof(_Float16));
    if ((rc = rocket_reduce_feature_fp16(fd, 2 * M, Pg, stack, csum, 0 /*sum*/)) != 0) goto out;

    /* 3. per-(n,g) mean/var/rsqrt (host, exact). sum(x^2)=csum[M+r]*4^k. */
    const float invPg  = 1.f / (float)Pg;
    const float four_k = ldexpf(1.f, 2 * k);
    for (int r = 0; r < M; r++) {
        float mn  = csum[r] * invPg;
        float msq = csum[M + r] * four_k * invPg;
        float var = msq - mn * mn;
        if (var < 0.f) var = 0.f;
        meanv[r] = mn;
        rinv[r]  = 1.f / sqrtf(var + eps);
    }

    /* 4. per-CHANNEL affine fold: out = x (x) A + B, A[n,c,*]=rinv[n,g]*gamma[c],
     *    B[n,c,*]=beta[c]-mean[n,g]*A (broadcast across the channel's P positions). */
    for (int n = 0; n < N; n++)
        for (int g = 0; g < G; g++) {
            int r = n * G + g;
            float rm = rinv[r], mm = meanv[r];
            for (int cc = 0; cc < Cg; cc++) {
                int c = g * Cg + cc;
                float gm = gamma ? (float)gamma[c] : 1.f;
                float a  = rm * gm;
                float b  = (beta ? (float)beta[c] : 0.f) - mm * a;
                _Float16 av = (_Float16)a, bv = (_Float16)b;
                _Float16 *Ar = A + ((size_t)n * C + c) * P;
                _Float16 *Br = B + ((size_t)n * C + c) * P;
                for (int pp = 0; pp < P; pp++) { Ar[pp] = av; Br[pp] = bv; }
            }
        }

    /* 5. tmp = x (x) A ; out = tmp + B (two NPU ew passes). */
    if ((rc = rocket_ew_mul_fp16(fd, x, A, tmp, (int)total)) != 0) goto out;
    rc = rocket_ew_add_fp16(fd, tmp, B, out, (int)total);

out:
    free(tmp); free(B); free(A); free(rinv); free(meanv); free(csum);
    free(stack); free(sq); free(xs_buf);
    return rc;
}

/* ============================================================================
 * SECTION — InstanceNorm (GroupNorm with G=C)
 * ==========================================================================*/

void rocket_instancenorm_ref_fp16(int N, int C, int P, const _Float16 *x,
                                  const _Float16 *gamma, const _Float16 *beta,
                                  float eps, _Float16 *out)
{
    rocket_groupnorm_ref_fp16(N, C, C, P, x, gamma, beta, eps, out);
}

int rocket_instancenorm_fp16(int fd, int N, int C, int P, const _Float16 *x,
                             const _Float16 *gamma, const _Float16 *beta,
                             float eps, _Float16 *out)
{
    return rocket_groupnorm_fp16(fd, N, C, C, P, x, gamma, beta, eps, out);
}

/* ============================================================================
 * SECTION — L2-Normalize
 * ==========================================================================*/

void rocket_l2norm_ref_fp16(int M, int H, const _Float16 *x, float eps, _Float16 *out)
{
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = x + (size_t)m * H;
        double ss = 0.0;
        for (int h = 0; h < H; h++) { double v = (double)xp[h]; ss += v * v; }
        double r = 1.0 / sqrt(ss + (double)eps);
        _Float16 *op = out + (size_t)m * H;
        for (int h = 0; h < H; h++) op[h] = (_Float16)((double)xp[h] * r);
    }
}

int rocket_l2norm_fp16(int fd, int M, int H, const _Float16 *x, float eps, _Float16 *out)
{
    if (M < 1 || H < 1 || !x || !out) return -1;
    if (fd < 0) { rocket_l2norm_ref_fp16(M, H, x, eps, out); return 0; }

    const size_t MH = (size_t)M * H;
    int rc = -2;
    _Float16 *xs_buf = NULL, *sq = NULL;
    float *ss = NULL, *r = NULL;

    int k = square_prescale_k(x, MH);
    const float p = ldexpf(1.f, -k);
    const _Float16 *xs = x;
    if (k > 0) {
        xs_buf = malloc(MH * sizeof(_Float16));
        if (!xs_buf) goto out;
        for (size_t i = 0; i < MH; i++) xs_buf[i] = (_Float16)((float)x[i] * p);
        xs = xs_buf;
    }

    sq = malloc(MH * sizeof(_Float16));
    ss = malloc((size_t)M * sizeof(float));
    r  = malloc((size_t)M * sizeof(float));
    if (!sq || !ss || !r) goto out;

    /* sq = xs (x) xs (NPU) -> ss[m]=sum_h sq (NPU fp32 reduce); true ss = ss*4^k. */
    if ((rc = rocket_ew_mul_fp16(fd, xs, xs, sq, (int)MH)) != 0) goto out;
    if ((rc = rocket_reduce_feature_fp16(fd, M, H, sq, ss, 0 /*sum*/)) != 0) goto out;
    const float four_k = ldexpf(1.f, 2 * k);
    for (int m = 0; m < M; m++) r[m] = 1.f / sqrtf(ss[m] * four_k + eps);

    /* out = x scaled per-row by r (NPU per-row broadcast multiply). */
    rc = rocket_scale_rows_fp16(fd, M, H, x, r, out);

out:
    free(r); free(ss); free(sq); free(xs_buf);
    return rc;
}
