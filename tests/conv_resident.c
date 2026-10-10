// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_resident.c — HW gate for the resident-weight forward conv
 * (rocket_conv2d_weights_pack + rocket_conv2d_fp16_prepacked): a host im2col into channel
 * planes and one resident fp16 matmul, against rocket_conv2d_ref_fp16, which shares no code
 * with it.
 *
 * Four arms:
 *
 *  1. EXACT. Hashed integer fills with no period (tests/test_fill.h), so every product,
 *     every partial and every sum is an integer. The gate computes, on the host, the
 *     largest magnitude any matmul partial reaches at every 32-plane boundary of the
 *     unfolded K (a superset of the K tiles the NPU rounds at) and the largest output, and
 *     asserts both are below 2048, which is what makes every fp16 rounding on the way
 *     exact. Under that condition the output must equal the reference bit for bit, into a
 *     NaN-sentinel buffer. The sweep varies one unfold path at a time: stride 1, 2 and 3
 *     per axis, pads 0-3 with pad_top != pad_left, dilation 2, an explicit output extent
 *     with a trailing pad, a 1x1 kernel read in place and one at stride 2, a kernel equal to
 *     its stride (a ViT patch embedding at IC 3), 1xK and Kx1 kernels, odd input widths on
 *     both sides of the deinterleave's 16-element step, OC not a multiple of 16 and an
 *     OH*OW not a multiple of 4 (the staging copy), then pix2pix's eight encoder layers
 *     and Depth Anything's head shapes at reduced extent. An integer bias rides on every
 *     other shape. Each shape runs once with one unfold thread and once with three.
 *  2. BOTH SPLITS. Four shapes run once with the workers splitting M and once N
 *     (ROCKET_CONV_SPLIT at pack time), each exact.
 *  3. BOUNDED. pix2pix's eight layers and three Depth Anything shapes with real-valued
 *     fills and a bias, against an fp64 reference: |out - ref| <= 2^-9 * mag per element,
 *     mag being the element's own sum over taps of |x*w| plus |bias|. fp16 rounds at 2^-11
 *     of a value, the GEMM's partial rounds at most once per K tile and the bias sum once,
 *     each under mag; the same unfold and GEMM priced at pix2pix's layers reached 2^-11.6
 *     (tests/conv_model_bench), so 2^-9 sits a factor 6 above it, while a dropped or
 *     doubled tap moves an element by about mag / (its tap count), far past 2^-9 at every
 *     shape here. The worst ratio is printed.
 *  4. CONTRACT. A depthwise descriptor is refused by the plan and by the pack; a handle
 *     keeps computing its packed weight after the caller's buffer changes, and a re-pack
 *     of the changed buffer computes the new one; a handle refuses a context it was not
 *     packed on.
 *
 * Run under ROCKET_KACC=0 too (a second ctest entry): that takes the matmul's CPU-
 * accumulate path and its planes narrow, where the default takes the NPU K-accumulation
 * and the cube de-interleave.
 *
 * Exit 0 pass, 1 fail, 2 no device or not an RK3588.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "test_fill.h"

static int g_checked = 0;

static uint64_t shape_seed(const rocket_conv2d_desc *d, int salt)
{
    const int f[] = { d->ic, d->ih, d->iw, d->oc, d->kh, d->kw, d->stride_y, d->stride_x,
                      d->pad_top, d->pad_left, d->dil_y, d->dil_x, d->oh, d->ow, salt };
    uint64_t h = 0xA4093822299F31D0ull;
    for (size_t i = 0; i < sizeof f / sizeof f[0]; i++) h = tf_hash(h, (uint64_t)(int64_t)f[i]);
    return h;
}

static void print_desc(const char *tag, const rocket_conv2d_desc *d)
{
    printf("%s conv IC=%d %dx%d -> OC=%d  K=%dx%d s=%dx%d p=%d,%d d=%dx%d  (OH=%d OW=%d%s)\n",
           tag, d->ic, d->ih, d->iw, d->oc, d->kh, d->kw, d->stride_y, d->stride_x,
           d->pad_top, d->pad_left, d->dil_y, d->dil_x, rocket_conv2d_oh(d), rocket_conv2d_ow(d),
           d->oh || d->ow ? ", explicit" : "");
}

/* The unfolded input as int32, K planes of M (K = IC*KH*KW in the entry's plane order). */
static int32_t *unfold_int(const rocket_conv2d_desc *d, const _Float16 *in)
{
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d), M = OH * OW;
    const int K = d->ic * d->kh * d->kw;
    int32_t *a = calloc((size_t)K * M, sizeof *a);
    if (!a) return NULL;
    for (int ic = 0; ic < d->ic; ic++)
        for (int ky = 0; ky < d->kh; ky++)
            for (int kx = 0; kx < d->kw; kx++) {
                int32_t *pl = a + ((size_t)(ic * d->kh + ky) * d->kw + kx) * M;
                for (int oy = 0; oy < OH; oy++) {
                    const int iy = oy * d->stride_y + ky * d->dil_y - d->pad_top;
                    if (iy < 0 || iy >= d->ih) continue;
                    for (int ox = 0; ox < OW; ox++) {
                        const int ix = ox * d->stride_x + kx * d->dil_x - d->pad_left;
                        if (ix < 0 || ix >= d->iw) continue;
                        pl[(size_t)oy * OW + ox] =
                            (int32_t)(float)in[((size_t)ic * d->ih + iy) * d->iw + ix];
                    }
                }
            }
    return a;
}

/* The largest |partial| of W . A at every 32-plane prefix of the unfolded K, integer data. */
static long max_prefix_partial(const rocket_conv2d_desc *d, const _Float16 *in, const _Float16 *W)
{
    const int M = rocket_conv2d_oh(d) * rocket_conv2d_ow(d), K = d->ic * d->kh * d->kw;
    int32_t *a = unfold_int(d, in);
    int32_t *wi = malloc((size_t)d->oc * K * sizeof *wi);
    int32_t *acc = malloc((size_t)M * sizeof *acc);
    long peak = 0;
    if (!a || !wi || !acc) { free(a); free(wi); free(acc); return 1L << 30; }
    for (size_t i = 0; i < (size_t)d->oc * K; i++) wi[i] = (int32_t)(float)W[i];
    for (int oc = 0; oc < d->oc; oc++) {
        memset(acc, 0, (size_t)M * sizeof *acc);
        const int32_t *w = wi + (size_t)oc * K;
        for (int k = 0; k < K; k++) {
            const int32_t wv = w[k];
            if (wv) {
                const int32_t *pl = a + (size_t)k * M;
                for (int m = 0; m < M; m++) acc[m] += wv * pl[m];
            }
            if (k % 32 == 31 || k == K - 1)
                for (int m = 0; m < M; m++) {
                    const long v = labs((long)acc[m]);
                    if (v > peak) peak = v;
                }
        }
    }
    free(a); free(wi); free(acc);
    return peak;
}

static const char *g_threads = NULL;   /* ROCKET_IM2COL_THREADS for the next pack */

/* Integer arm: exact against the reference (+ an integer bias). */
static int run_exact(rocket_ctx *ctx, const rocket_conv2d_desc *d, int with_bias,
                     const char *split, int xr, int wr)
{
    char tag[32];
    snprintf(tag, sizeof tag, "%s%st%s", split ? split : "", split ? " " : "",
             g_threads ? g_threads : "ctx");
    print_desc(tag, d);
    int plan = rocket_conv2d_prepacked_plan(d);
    if (plan) { printf("  plan refused (%d) (FAIL: every shape here is in the entry's set)\n", plan); return 1; }
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const size_t ni = (size_t)d->ic * d->ih * d->iw, nw = (size_t)d->oc * d->ic * d->kh * d->kw;
    const size_t no = (size_t)d->oc * OH * OW;
    _Float16 *in = malloc(ni * 2), *W = malloc(nw * 2), *out = malloc(no * 2), *ref = malloc(no * 2);
    _Float16 *bias = with_bias ? malloc((size_t)d->oc * 2) : NULL;
    if (!in || !W || !out || !ref || (with_bias && !bias)) { printf("  oom\n"); return 1; }
    const uint64_t seed = shape_seed(d, with_bias);
    tf_fill_f16_int(in, ni, tf_hash(seed, 1), -xr, xr);
    tf_fill_f16_int(W, nw, tf_hash(seed, 2), -wr, wr);
    if (bias) tf_fill_f16_int(bias, (size_t)d->oc, tf_hash(seed, 3), -9, 9);

    int fail = 0;
    rocket_conv2d_ref_fp16(d, in, W, ref);
    double peak = 0;
    for (size_t i = 0; i < no; i++) {
        if (bias) ref[i] = (_Float16)((float)ref[i] + (float)bias[i / ((size_t)OH * OW)]);
        if (fabs((float)ref[i]) > peak) peak = fabs((float)ref[i]);
    }
    const long pp = max_prefix_partial(d, in, W);
    if (!(peak < 2048.0) || pp >= 2048) {
        printf("  max|out| %.0f, max|partial prefix| %ld: not both below 2048, so fp16 is not "
               "exact and the fill is wrong for this shape (FAIL)\n", peak, pp);
        fail = 1;
    }
    if (split) setenv("ROCKET_CONV_SPLIT", split, 1);
    if (g_threads) setenv("ROCKET_IM2COL_THREADS", g_threads, 1);
    rocket_conv2d_weights *h = rocket_conv2d_weights_pack(ctx, d, W, bias);
    unsetenv("ROCKET_CONV_SPLIT");
    unsetenv("ROCKET_IM2COL_THREADS");
    if (!h) { printf("  pack failed (FAIL)\n"); fail = 1; goto done; }
    tf_sentinel_f16(out, no);
    int r = rocket_conv2d_fp16_prepacked(ctx, h, in, out);
    if (r) { printf("  run = %d (FAIL)\n", r); fail = 1; }
    else {
        const int dims[3] = { d->oc, OH, OW };
        long bad = tf_cmp_f16("  exact", out, ref, dims, 3);
        g_checked++;
        printf("  exact%s: max|out| %.0f, max|partial| %ld: %ld of %zu differ -> %s\n",
               bias ? " (+bias)" : "", peak, pp, bad, no, bad ? "FAIL" : "PASS");
        if (bad) fail = 1;
    }
    rocket_conv2d_weights_free(ctx, h);
done:
    free(in); free(W); free(out); free(ref); free(bias);
    return fail;
}

/* fp64 reference with bias, and each element's magnitude sum. */
static void ref64(const rocket_conv2d_desc *d, const _Float16 *in, const _Float16 *W,
                  const _Float16 *bias, double *ref, double *mag)
{
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    for (int oc = 0; oc < d->oc; oc++)
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++) {
                double s = bias ? (double)bias[oc] : 0.0, m = bias ? fabs((double)bias[oc]) : 0.0;
                for (int ic = 0; ic < d->ic; ic++)
                    for (int ky = 0; ky < d->kh; ky++) {
                        const int iy = oy * d->stride_y + ky * d->dil_y - d->pad_top;
                        if (iy < 0 || iy >= d->ih) continue;
                        for (int kx = 0; kx < d->kw; kx++) {
                            const int ix = ox * d->stride_x + kx * d->dil_x - d->pad_left;
                            if (ix < 0 || ix >= d->iw) continue;
                            const double p = (double)in[((size_t)ic * d->ih + iy) * d->iw + ix] *
                                (double)W[(((size_t)oc * d->ic + ic) * d->kh + ky) * d->kw + kx];
                            s += p; m += fabs(p);
                        }
                    }
                const size_t o = ((size_t)oc * OH + oy) * OW + ox;
                ref[o] = s; mag[o] = m;
            }
}

static int run_bounded(rocket_ctx *ctx, const rocket_conv2d_desc *d)
{
    print_desc("real", d);
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const size_t ni = (size_t)d->ic * d->ih * d->iw, nw = (size_t)d->oc * d->ic * d->kh * d->kw;
    const size_t no = (size_t)d->oc * OH * OW;
    _Float16 *in = malloc(ni * 2), *W = malloc(nw * 2), *out = malloc(no * 2), *bias = malloc((size_t)d->oc * 2);
    double *ref = malloc(no * sizeof *ref), *mag = malloc(no * sizeof *mag);
    if (!in || !W || !out || !bias || !ref || !mag) { printf("  oom\n"); return 1; }
    const uint64_t seed = shape_seed(d, 7);
    tf_fill_f16(in, ni, tf_hash(seed, 1), -1.0, 1.0);
    tf_fill_f16(W, nw, tf_hash(seed, 2), -0.05, 0.05);
    tf_fill_f16(bias, (size_t)d->oc, tf_hash(seed, 3), -0.5, 0.5);
    ref64(d, in, W, bias, ref, mag);

    int fail = 0;
    rocket_conv2d_weights *h = rocket_conv2d_weights_pack(ctx, d, W, bias);
    if (!h) { printf("  pack failed (FAIL)\n"); fail = 1; goto done; }
    tf_sentinel_f16(out, no);
    int r = rocket_conv2d_fp16_prepacked(ctx, h, in, out);
    if (r) { printf("  run = %d (FAIL)\n", r); fail = 1; }
    else {
        long bad = 0;
        double worst = 0;
        size_t wi = 0;
        for (size_t i = 0; i < no; i++) {
            const double e = fabs((double)out[i] - ref[i]);
            const double m = mag[i] > 0 ? mag[i] : 1e-30;
            if (isnan((double)out[i]) || e > ldexp(mag[i], -9) + 1e-30) bad++;
            if (e / m > worst) { worst = e / m; wi = i; }
        }
        g_checked++;
        printf("  bounded: worst |err|/mag %.3g (2^%.1f) at %zu, %ld of %zu past 2^-9 -> %s\n",
               worst, worst > 0 ? log2(worst) : -99.0, wi, bad, no, bad ? "FAIL" : "PASS");
        if (bad) fail = 1;
    }
    rocket_conv2d_weights_free(ctx, h);
done:
    free(in); free(W); free(out); free(bias); free(ref); free(mag);
    return fail;
}

/* A handle computes the weight it packed, not whatever the caller's buffer holds now. */
static int run_contract(rocket_ctx *ctx, rocket_ctx *other)
{
    rocket_conv2d_desc d = { .ic = 64, .ih = 24, .iw = 20, .oc = 32, .kh = 4, .kw = 4,
        .stride_y = 2, .stride_x = 2, .pad_top = 1, .pad_left = 1, .dil_y = 1, .dil_x = 1 };
    print_desc("contract", &d);
    const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d);
    const size_t ni = (size_t)d.ic * d.ih * d.iw, nw = (size_t)d.ic * d.oc * d.kh * d.kw;
    const size_t no = (size_t)d.oc * OH * OW;
    const int dims[3] = { d.oc, OH, OW };
    _Float16 *in = malloc(ni * 2), *W = malloc(nw * 2), *out = malloc(no * 2), *ref = malloc(no * 2);
    int fail = 0;
    tf_fill_f16_int(in, ni, 11, -4, 4);
    tf_fill_f16_int(W, nw, 12, -3, 3);
    rocket_conv2d_ref_fp16(&d, in, W, ref);
    rocket_conv2d_weights *h = rocket_conv2d_weights_pack(ctx, &d, W, NULL);
    if (!h) { printf("  pack failed (FAIL)\n"); fail = 1; goto done; }
    tf_fill_f16_int(W, nw, 13, -3, 3);                 /* the caller's buffer changes */
    tf_sentinel_f16(out, no);
    if (rocket_conv2d_fp16_prepacked(ctx, h, in, out) ||
        tf_cmp_f16("  packed weight after the buffer changed", out, ref, dims, 3)) {
        printf("  the handle did not keep its packed weight (FAIL)\n"); fail = 1;
    } else printf("  the handle computes its packed weight after the buffer changed: PASS\n");
    if (other) {
        int r = rocket_conv2d_fp16_prepacked(other, h, in, out);
        printf("  a handle on another context: %d -> %s\n", r, r < 0 ? "refused, PASS" : "ran (FAIL)");
        if (r >= 0) fail = 1;
    }
    rocket_conv2d_weights_free(ctx, h);
    rocket_conv2d_ref_fp16(&d, in, W, ref);            /* the new weight's result */
    h = rocket_conv2d_weights_pack(ctx, &d, W, NULL);
    tf_sentinel_f16(out, no);
    if (!h || rocket_conv2d_fp16_prepacked(ctx, h, in, out) ||
        tf_cmp_f16("  re-packed weight", out, ref, dims, 3)) {
        printf("  a re-pack did not compute the new weight (FAIL)\n"); fail = 1;
    } else printf("  a re-pack computes the new weight: PASS\n");
    rocket_conv2d_weights_free(ctx, h);

    /* depthwise: refused by the plan and the pack */
    rocket_conv2d_desc dw = { .ic = 32, .ih = 8, .iw = 8, .oc = 32, .kh = 3, .kw = 3,
        .stride_y = 1, .stride_x = 1, .pad_top = 1, .pad_left = 1, .dil_y = 1, .dil_x = 1,
        .depthwise = 1 };
    int p = rocket_conv2d_prepacked_plan(&dw);
    rocket_conv2d_weights *hd = rocket_conv2d_weights_pack(ctx, &dw, W, NULL);
    printf("  depthwise: plan %d, pack %s -> %s\n", p, hd ? "a handle" : "NULL",
           (p == ROCKET_E_UNSUPPORTED && !hd) ? "PASS" : "FAIL");
    if (p != ROCKET_E_UNSUPPORTED || hd) fail = 1;
    rocket_conv2d_weights_free(ctx, hd);
done:
    free(in); free(W); free(out); free(ref);
    return fail;
}

#define CD(IC, IH, IW, OC, KH, KW, SY, SX, PT, PL, DY, DX, OH, OW)                          \
    { .ic = IC, .ih = IH, .iw = IW, .oc = OC, .kh = KH, .kw = KW, .stride_y = SY,             \
      .stride_x = SX, .pad_top = PT, .pad_left = PL, .dil_y = DY, .dil_x = DX,               \
      .oh = OH, .ow = OW }

int main(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (!hw || strcmp(hw->name, "rk3588")) { printf("not an RK3588; skipping\n"); return 2; }
    int fd = rocket_open();
    if (fd < 0) { printf("SKIP: no /dev/accel/accel0\n"); return 2; }
    rocket_close(fd);
    rocket_ctx *ctx = rocket_ctx_create(3);
    rocket_ctx *other = rocket_ctx_create(1);
    if (!ctx) { printf("rocket_ctx_create failed (FAIL)\n"); return 1; }
    int fail = 0;

    /* 1. exact: one unfold path at a time, then pix2pix and Depth Anything */
    static const rocket_conv2d_desc sweep[] = {
        CD(32, 16, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0),     /* stride 1, pad 1 */
        CD(32, 16, 40, 32, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),     /* stride 2, pix2pix's kernel */
        CD(32, 17, 37, 16, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0),     /* stride 2, odd input */
        CD(32, 15, 33, 16, 3, 3, 3, 3, 1, 1, 1, 1, 0, 0),     /* stride 3, the gather */
        CD(32, 12, 31, 32, 3, 5, 1, 2, 2, 1, 1, 1, 0, 0),     /* stride per axis, pad per axis */
        CD(32, 16, 16, 32, 3, 3, 1, 1, 2, 2, 2, 2, 0, 0),     /* dilation 2 */
        CD(32, 16, 16, 16, 3, 3, 2, 2, 0, 0, 1, 1, 8, 8),     /* explicit extent: SAME's trailing pad */
        CD(64, 12, 20, 48, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0),     /* 1x1 in place, OC%16 = 0 */
        CD(64, 9, 11, 40, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0),      /* 1x1 in place, OC 40, odd pixels */
        CD(64, 16, 18, 32, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0),     /* 1x1 at stride 2: unfolded */
        CD(3, 64, 64, 48, 16, 16, 16, 16, 0, 0, 1, 1, 0, 0),  /* a ViT patch embedding */
        CD(3, 70, 70, 32, 14, 14, 14, 14, 0, 0, 1, 1, 0, 0),  /* DINOv2's, IC 3 */
        CD(32, 20, 24, 32, 1, 7, 1, 1, 0, 3, 1, 1, 0, 0),     /* 1xK */
        CD(32, 20, 24, 32, 7, 1, 1, 1, 3, 0, 1, 1, 0, 0),     /* Kx1 */
        CD(3, 32, 31, 20, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),      /* IC 3, OC 20, odd input */
        CD(32, 7, 9, 16, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0),       /* OH*OW 4x5: staged output */
        CD(32, 5, 5, 16, 3, 3, 1, 1, 3, 3, 1, 1, 0, 0),       /* pad 3 > k-1 */
        CD(48, 8, 8, 32, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),       /* deinterleave below 16 */
    };
    const size_t ns = sizeof sweep / sizeof sweep[0];
    static const char *thr[] = { "1", "3" };
    for (size_t t = 0; t < 2; t++) {
        g_threads = thr[t];
        for (size_t i = 0; i < ns; i++) {
            fail |= run_exact(ctx, &sweep[i], (int)(i & 1), NULL, 4, 3);
            printf("\n");
        }
    }
    g_threads = NULL;
    static const rocket_conv2d_desc models[] = {
        CD(3, 256, 256, 64, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),       /* pix2pix c1 */
        CD(64, 128, 128, 128, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),     /* c2 */
        CD(128, 64, 64, 256, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),      /* c3 */
        CD(256, 32, 32, 512, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),      /* c4 */
        CD(512, 16, 16, 512, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),      /* c5 */
        CD(512, 8, 8, 512, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),        /* c6 */
        CD(512, 4, 4, 512, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),        /* c7 */
        CD(512, 2, 2, 512, 4, 4, 2, 2, 1, 1, 1, 1, 0, 0),        /* c8: one output pixel */
        CD(64, 74, 74, 64, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0),        /* Depth Anything fusion 3x3 */
        CD(384, 37, 37, 384, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0),      /* its reassemble resize */
        CD(64, 96, 96, 32, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0),        /* its head conv1, reduced */
    };
    const size_t nm = sizeof models / sizeof models[0];
    for (size_t i = 0; i < nm; i++) {
        fail |= run_exact(ctx, &models[i], (int)(i & 1), NULL, 2, 1);
        printf("\n");
    }

    /* 2. both splits */
    static const int both[] = { 1, 3, 7, 8 };
    for (size_t i = 0; i < sizeof both / sizeof both[0]; i++) {
        fail |= run_exact(ctx, &models[both[i]], 1, "m", 2, 1);
        fail |= run_exact(ctx, &models[both[i]], 1, "n", 2, 1);
        printf("\n");
    }

    /* 3. bounded, real-valued */
    for (size_t i = 0; i < nm; i++) {
        fail |= run_bounded(ctx, &models[i]);
        printf("\n");
    }

    /* 4. the contract */
    fail |= run_contract(ctx, other);

    if (other) rocket_ctx_free(other);
    rocket_ctx_free(ctx);
    if (g_checked == 0) { printf("no shape reached a numeric check (FAIL)\n"); fail = 1; }
    printf("%d numeric checks\n==== %s ====\n", g_checked, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
