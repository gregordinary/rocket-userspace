// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_transpose_resident.c — HW gate for the resident-weight transposed conv
 * (rocket_conv_transpose2d_weights_pack + rocket_conv_transpose2d_fp16_prepacked): one
 * resident fp16 matmul in channel-planes form and a host scatter-add, against the scatter
 * reference rocket_conv_transpose2d_ref_fp16, which shares no code with it.
 *
 * Four arms:
 *
 *  1. EXACT. Hashed integer fills with no period (tests/test_fill.h), so every product,
 *     every partial and every sum is an integer. The gate computes, on the host, the
 *     largest magnitude any matmul partial reaches at every 32-channel boundary (a superset
 *     of the K tiles the NPU rounds at) and the largest output, and asserts both are below
 *     2048, which is what makes every fp16 rounding on the way exact. Under that condition
 *     the output must equal the reference bit for bit, into a NaN-sentinel buffer. The
 *     shapes are the old gate's direct sweep (stride 1/2/3, output_padding, dilation 2, a
 *     1x3 kernel with stride 1x2, a 3x5 kernel with pads (1,2) and output_padding on x only,
 *     IC 3, OC not a multiple of 16, an odd pixel count), a cropping pad the one-shot entry
 *     refuses, and pix2pix's eight and SAM's two layers, with an integer bias on some.
 *  2. BOTH SPLITS. Four shapes run once with the workers splitting M and once N
 *     (ROCKET_CT_SPLIT at pack time), each exact.
 *  3. BOUNDED. pix2pix's and SAM's layers with real-valued fills and a bias, against an
 *     fp64 scatter reference: |out - ref| <= 2^-9 * mag per element, mag being the
 *     element's own sum over taps of |x*w| plus |bias|. The bound is the arithmetic's: fp16
 *     rounds at 2^-11 of a value, the partial rounds once per K tile (at most two at IC
 *     1024 here) and the sum once, all under mag, so 3 * 2^-11 * mag bounds it; 2^-9 leaves
 *     a factor 4/3 and still sits far under a dropped or doubled tap, which moves an element
 *     by about mag / (taps it has). The worst ratio is printed.
 *  4. CONTRACT. A depthwise descriptor is refused by the plan and by the pack; a handle
 *     keeps computing its packed weight after the caller's buffer changes, and a re-pack of
 *     the changed buffer computes the new one; a handle refuses a context it was not
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

static uint64_t shape_seed(const rocket_conv_transpose2d_desc *d, int salt)
{
    const int f[] = { d->ic, d->ih, d->iw, d->oc, d->kh, d->kw, d->stride_y, d->stride_x,
                      d->pad_top, d->pad_left, d->opad_y, d->opad_x, d->dil_y, d->dil_x,
                      d->depthwise, salt };
    uint64_t h = 0x13198A2E03707344ull;
    for (size_t i = 0; i < sizeof f / sizeof f[0]; i++) h = tf_hash(h, (uint64_t)(int64_t)f[i]);
    return h;
}

static void print_desc(const char *tag, const rocket_conv_transpose2d_desc *d)
{
    printf("%s convT IC=%d %dx%d -> OC=%d  K=%dx%d s=%dx%d p=%d,%d op=%d,%d d=%dx%d  (OH=%d OW=%d)\n",
           tag, d->ic, d->ih, d->iw, d->oc, d->kh, d->kw, d->stride_y, d->stride_x,
           d->pad_top, d->pad_left, d->opad_y, d->opad_x, d->dil_y, d->dil_x,
           rocket_conv_transpose2d_oh(d), rocket_conv_transpose2d_ow(d));
}

/* The largest |partial| of in^T . W at every 32-channel prefix, integer data. */
static long max_prefix_partial(const rocket_conv_transpose2d_desc *d, const _Float16 *in,
                               const _Float16 *W)
{
    const int IC = d->ic, M = d->ih * d->iw, NN = d->oc * d->kh * d->kw;
    int32_t *xi = malloc((size_t)IC * M * sizeof *xi), *wi = malloc((size_t)IC * NN * sizeof *wi);
    int32_t *row = malloc((size_t)NN * sizeof *row);
    long peak = 0;
    for (size_t i = 0; i < (size_t)IC * M; i++) xi[i] = (int32_t)(float)in[i];
    for (size_t i = 0; i < (size_t)IC * NN; i++) wi[i] = (int32_t)(float)W[i];
    for (int m = 0; m < M; m++) {
        memset(row, 0, (size_t)NN * sizeof *row);
        for (int ic = 0; ic < IC; ic++) {
            const int32_t x = xi[(size_t)ic * M + m];
            const int32_t *w = wi + (size_t)ic * NN;
            if (x) for (int n = 0; n < NN; n++) row[n] += x * w[n];
            if (ic % 32 == 31 || ic == IC - 1)
                for (int n = 0; n < NN; n++) {
                    long a = labs((long)row[n]);
                    if (a > peak) peak = a;
                }
        }
    }
    free(xi); free(wi); free(row);
    return peak;
}

/* Integer arm: exact against the scatter reference (+ an integer bias). */
static int run_exact(rocket_ctx *ctx, const rocket_conv_transpose2d_desc *d, int with_bias,
                     const char *split, int xr, int wr)
{
    print_desc(split ? split : "", d);
    int plan = rocket_conv_transpose2d_prepacked_plan(d);
    if (plan) { printf("  plan refused (%d) (FAIL: every shape here is in the entry's set)\n", plan); return 1; }
    const int OH = rocket_conv_transpose2d_oh(d), OW = rocket_conv_transpose2d_ow(d);
    const size_t ni = (size_t)d->ic * d->ih * d->iw, nw = (size_t)d->ic * d->oc * d->kh * d->kw;
    const size_t no = (size_t)d->oc * OH * OW;
    _Float16 *in = malloc(ni * 2), *W = malloc(nw * 2), *out = malloc(no * 2), *ref = malloc(no * 2);
    _Float16 *bias = with_bias ? malloc((size_t)d->oc * 2) : NULL;
    if (!in || !W || !out || !ref || (with_bias && !bias)) { printf("  oom\n"); return 1; }
    const uint64_t seed = shape_seed(d, with_bias);
    tf_fill_f16_int(in, ni, tf_hash(seed, 1), -xr, xr);
    tf_fill_f16_int(W, nw, tf_hash(seed, 2), -wr, wr);
    if (bias) tf_fill_f16_int(bias, (size_t)d->oc, tf_hash(seed, 3), -9, 9);

    int fail = 0;
    rocket_conv_transpose2d_ref_fp16(d, in, W, ref);
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
    if (split) setenv("ROCKET_CT_SPLIT", split, 1);
    rocket_conv_transpose2d_weights *h = rocket_conv_transpose2d_weights_pack(ctx, d, W, bias);
    if (split) unsetenv("ROCKET_CT_SPLIT");
    if (!h) { printf("  pack failed (FAIL)\n"); fail = 1; goto done; }
    tf_sentinel_f16(out, no);
    int r = rocket_conv_transpose2d_fp16_prepacked(ctx, h, in, out);
    if (r) { printf("  run = %d (FAIL)\n", r); fail = 1; }
    else {
        const int dims[3] = { d->oc, OH, OW };
        long bad = tf_cmp_f16("  exact", out, ref, dims, 3);
        g_checked++;
        printf("  exact%s: max|out| %.0f, max|partial| %ld: %ld of %zu differ -> %s\n",
               bias ? " (+bias)" : "", peak, pp, bad, no, bad ? "FAIL" : "PASS");
        if (bad) fail = 1;
    }
    rocket_conv_transpose2d_weights_free(ctx, h);
done:
    free(in); free(W); free(out); free(ref); free(bias);
    return fail;
}

/* fp64 scatter reference with bias, and each element's magnitude sum. */
static void ref64(const rocket_conv_transpose2d_desc *d, const _Float16 *in, const _Float16 *W,
                  const _Float16 *bias, double *ref, double *mag)
{
    const int OH = rocket_conv_transpose2d_oh(d), OW = rocket_conv_transpose2d_ow(d);
    const size_t plane = (size_t)OH * OW;
    for (int oc = 0; oc < d->oc; oc++)
        for (size_t j = 0; j < plane; j++) {
            ref[oc * plane + j] = bias ? (double)bias[oc] : 0.0;
            mag[oc * plane + j] = bias ? fabs((double)bias[oc]) : 0.0;
        }
    for (int ic = 0; ic < d->ic; ic++)
        for (int ih = 0; ih < d->ih; ih++)
            for (int iw = 0; iw < d->iw; iw++) {
                const double x = (double)in[((size_t)ic * d->ih + ih) * d->iw + iw];
                if (x == 0) continue;
                for (int oc = 0; oc < d->oc; oc++)
                    for (int kh = 0; kh < d->kh; kh++) {
                        const int ph = ih * d->stride_y - d->pad_top + kh * d->dil_y;
                        if (ph < 0 || ph >= OH) continue;
                        for (int kw = 0; kw < d->kw; kw++) {
                            const int pw = iw * d->stride_x - d->pad_left + kw * d->dil_x;
                            if (pw < 0 || pw >= OW) continue;
                            const double p = x * (double)W[(((size_t)ic * d->oc + oc) * d->kh + kh) * d->kw + kw];
                            ref[oc * plane + (size_t)ph * OW + pw] += p;
                            mag[oc * plane + (size_t)ph * OW + pw] += fabs(p);
                        }
                    }
            }
}

static int run_bounded(rocket_ctx *ctx, const rocket_conv_transpose2d_desc *d)
{
    print_desc("real", d);
    const int OH = rocket_conv_transpose2d_oh(d), OW = rocket_conv_transpose2d_ow(d);
    const size_t ni = (size_t)d->ic * d->ih * d->iw, nw = (size_t)d->ic * d->oc * d->kh * d->kw;
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
    rocket_conv_transpose2d_weights *h = rocket_conv_transpose2d_weights_pack(ctx, d, W, bias);
    if (!h) { printf("  pack failed (FAIL)\n"); fail = 1; goto done; }
    tf_sentinel_f16(out, no);
    int r = rocket_conv_transpose2d_fp16_prepacked(ctx, h, in, out);
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
    rocket_conv_transpose2d_weights_free(ctx, h);
done:
    free(in); free(W); free(out); free(bias); free(ref); free(mag);
    return fail;
}

/* A handle computes the weight it packed, not whatever the caller's buffer holds now. */
static int run_contract(rocket_ctx *ctx, rocket_ctx *other)
{
    rocket_conv_transpose2d_desc d = { .ic = 64, .ih = 12, .iw = 10, .oc = 32, .kh = 4, .kw = 4,
        .stride_y = 2, .stride_x = 2, .pad_top = 1, .pad_left = 1, .dil_y = 1, .dil_x = 1 };
    print_desc("contract", &d);
    const int OH = rocket_conv_transpose2d_oh(&d), OW = rocket_conv_transpose2d_ow(&d);
    const size_t ni = (size_t)d.ic * d.ih * d.iw, nw = (size_t)d.ic * d.oc * d.kh * d.kw;
    const size_t no = (size_t)d.oc * OH * OW;
    const int dims[3] = { d.oc, OH, OW };
    _Float16 *in = malloc(ni * 2), *W = malloc(nw * 2), *out = malloc(no * 2), *ref = malloc(no * 2);
    int fail = 0;
    tf_fill_f16_int(in, ni, 11, -4, 4);
    tf_fill_f16_int(W, nw, 12, -3, 3);
    rocket_conv_transpose2d_ref_fp16(&d, in, W, ref);
    rocket_conv_transpose2d_weights *h = rocket_conv_transpose2d_weights_pack(ctx, &d, W, NULL);
    if (!h) { printf("  pack failed (FAIL)\n"); fail = 1; goto done; }
    tf_fill_f16_int(W, nw, 13, -3, 3);                 /* the caller's buffer changes */
    tf_sentinel_f16(out, no);
    if (rocket_conv_transpose2d_fp16_prepacked(ctx, h, in, out) ||
        tf_cmp_f16("  packed weight after the buffer changed", out, ref, dims, 3)) {
        printf("  the handle did not keep its packed weight (FAIL)\n"); fail = 1;
    } else printf("  the handle computes its packed weight after the buffer changed: PASS\n");
    if (other) {
        int r = rocket_conv_transpose2d_fp16_prepacked(other, h, in, out);
        printf("  a handle on another context: %d -> %s\n", r, r < 0 ? "refused, PASS" : "ran (FAIL)");
        if (r >= 0) fail = 1;
    }
    rocket_conv_transpose2d_weights_free(ctx, h);
    rocket_conv_transpose2d_ref_fp16(&d, in, W, ref);  /* the new weight's result */
    h = rocket_conv_transpose2d_weights_pack(ctx, &d, W, NULL);
    tf_sentinel_f16(out, no);
    if (!h || rocket_conv_transpose2d_fp16_prepacked(ctx, h, in, out) ||
        tf_cmp_f16("  re-packed weight", out, ref, dims, 3)) {
        printf("  a re-pack did not compute the new weight (FAIL)\n"); fail = 1;
    } else printf("  a re-pack computes the new weight: PASS\n");
    rocket_conv_transpose2d_weights_free(ctx, h);

    /* depthwise: refused by the plan and the pack */
    rocket_conv_transpose2d_desc dw = { .ic = 32, .ih = 8, .iw = 8, .oc = 32, .kh = 2, .kw = 2,
        .stride_y = 2, .stride_x = 2, .dil_y = 1, .dil_x = 1, .depthwise = 1 };
    int p = rocket_conv_transpose2d_prepacked_plan(&dw);
    rocket_conv_transpose2d_weights *hd = rocket_conv_transpose2d_weights_pack(ctx, &dw, W, NULL);
    printf("  depthwise: plan %d, pack %s -> %s\n", p, hd ? "a handle" : "NULL",
           (p == ROCKET_E_UNSUPPORTED && !hd) ? "PASS" : "FAIL");
    if (p != ROCKET_E_UNSUPPORTED || hd) fail = 1;
    rocket_conv_transpose2d_weights_free(ctx, hd);
done:
    free(in); free(W); free(out); free(ref);
    return fail;
}

#define CTD(IC, IH, IW, OC, KH, KW, SY, SX, PT, PL, OPY, OPX, DY, DX)                         \
    { .ic = IC, .ih = IH, .iw = IW, .oc = OC, .kh = KH, .kw = KW, .stride_y = SY,             \
      .stride_x = SX, .pad_top = PT, .pad_left = PL, .opad_y = OPY, .opad_x = OPX,           \
      .dil_y = DY, .dil_x = DX }

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

    /* 1. exact: the old gate's direct sweep, a cropping pad, then pix2pix and SAM */
    static const rocket_conv_transpose2d_desc sweep[] = {
        CTD(32, 8, 8, 16, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1),
        CTD(32, 8, 8, 16, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),
        CTD(32, 8, 8, 16, 3, 3, 1, 1, 0, 0, 0, 0, 1, 1),
        CTD(64, 7, 9, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1),
        CTD(32, 6, 6, 16, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1),
        CTD(32, 8, 8, 16, 3, 3, 1, 1, 0, 0, 0, 0, 2, 2),
        CTD(64, 6, 6, 48, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1),
        CTD(32, 8, 7, 16, 1, 3, 1, 2, 0, 1, 0, 0, 1, 1),
        CTD(3, 8, 8, 16, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),
        CTD(64, 32, 32, 64, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1),
        CTD(64, 32, 32, 64, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),
        CTD(32, 16, 16, 32, 4, 4, 4, 4, 0, 0, 0, 0, 1, 1),
        CTD(32, 24, 20, 24, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1),
        CTD(32, 12, 12, 32, 3, 3, 2, 1, 1, 1, 0, 0, 1, 1),
        CTD(128, 16, 16, 64, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),
        CTD(32, 10, 14, 48, 4, 2, 2, 2, 1, 0, 0, 0, 1, 1),
        CTD(64, 12, 8, 32, 2, 4, 2, 4, 0, 1, 1, 0, 1, 1),
        CTD(48, 9, 13, 40, 3, 5, 2, 2, 1, 2, 0, 1, 1, 1),
        CTD(32, 8, 8, 16, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1),     /* pad > k-1: a cropping transpose */
        CTD(40, 5, 7, 20, 3, 3, 2, 3, 1, 0, 1, 2, 2, 1),     /* dilation and stride per axis */
    };
    for (size_t i = 0; i < sizeof sweep / sizeof sweep[0]; i++) {
        fail |= run_exact(ctx, &sweep[i], (int)(i & 1), NULL, 4, 3);
        printf("\n");
    }
    static const rocket_conv_transpose2d_desc models[] = {
        CTD(512, 1, 1, 512, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),        /* pix2pix up1 */
        CTD(1024, 2, 2, 512, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),       /* up2 */
        CTD(1024, 4, 4, 512, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),       /* up3 */
        CTD(1024, 8, 8, 512, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),       /* up4 */
        CTD(1024, 16, 16, 256, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),     /* up5 */
        CTD(512, 32, 32, 128, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),      /* up6 */
        CTD(256, 64, 64, 64, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),       /* up7 */
        CTD(128, 128, 128, 3, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),      /* up8 */
        CTD(256, 64, 64, 64, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1),       /* SAM upscale1 */
        CTD(64, 128, 128, 32, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1),      /* SAM upscale2 */
    };
    const size_t nm = sizeof models / sizeof models[0];
    for (size_t i = 0; i < nm; i++) {
        fail |= run_exact(ctx, &models[i], (int)(i & 1), NULL, 2, 1);
        printf("\n");
    }

    /* 2. both splits */
    static const int both[] = { 5, 6, 8, 9 };
    for (size_t i = 0; i < sizeof both / sizeof both[0]; i++) {
        fail |= run_exact(ctx, &models[both[i]], 1, "m", 2, 1);
        fail |= run_exact(ctx, &models[both[i]], 1, "n", 2, 1);
        printf("\n");
    }
    fail |= run_exact(ctx, &sweep[3], 1, "m", 4, 3);    /* odd pixel count, M split */
    printf("\n");

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
