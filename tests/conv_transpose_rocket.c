// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_transpose_rocket.c — HW gate for transposed convolution (ConvTranspose2d /
 * "deconvolution") on the rocket NPU (rocket_conv_transpose2d_fp16).
 *
 * Two independent layers (mirroring conv2d_fp16_rocket):
 *
 *  1. LOWERING SELF-CHECK (runs anywhere, incl. x86, no NPU). The library lowers a
 *     transposed conv to dilate-input + rot180/transpose-weight + a STRIDE-1 forward
 *     conv. This test re-derives that lowering INDEPENDENTLY (builds the dilated input
 *     and flipped weights here, runs the forward-conv CPU oracle rocket_conv2d_ref_fp16)
 *     and compares it to the DIRECT scatter-add definition rocket_conv_transpose2d_ref_fp16.
 *     A PASS proves the dilation geometry + the 180-deg flip + channel transpose are
 *     correct — the novel math, verifiable off-hardware.
 *
 *  2. ON-HARDWARE END-TO-END (only if /dev/accel/accel0 opens). Runs
 *     rocket_conv_transpose2d_fp16 on the NPU and compares to the direct scatter-add
 *     reference, through whichever route rocket_conv_transpose2d_route() names. A shape
 *     on the hardware deconvolution route runs a second time with
 *     ROCKET_CONV_TRANSPOSE_HW=0, and the two routes' outputs are compared too.
 *
 * On an RK3588 the gate refuses to pass unless some shape actually took the hardware
 * route, so a planner that silently declines everything cannot turn it green.
 *
 * THE INSTRUMENT (tests/test_fill.h). Every tensor gets a hashed fill with no period:
 * integers in [-4,4] for the input and [-3,3] for the weights, a different seed per
 * shape and per tensor. The results are exact integers, so every comparison is EXACT,
 * element by element. The gate asserts max|ref| < 2048 for each shape, because that is
 * what makes fp16 exact, rather than assuming it. The output starts as an fp16 NaN
 * sentinel, so an element nothing wrote can never agree with its reference.
 *
 * Each of those replaced a blind spot. The old fill, W[i]=(i%3)-1, is a function of the
 * kernel column alone on every 3x3 kernel, so a missing channel transpose or a vertical
 * flip passed. At IC=OC=64 with a 4x4 kernel it gave [OC][IC] the same values as
 * [IC][OC], and the old input (i%5)-2 was identical in every channel and row of a
 * 24x20 plane. The bar was max_abs <= 1.0 over exact sums, which passes a dropped tap
 * whose product is +-1, and the output started at zero.
 *
 * The sweep exercises stride 1/2/3, pad 0..K-1, output_padding, dilation>1, asymmetric
 * kernels, IC/OC crossing the cube group boundaries, an RGB-width input, and a large
 * shape that forces the forward conv to tile. Two shapes on the hardware route have a
 * non-square kernel, unequal pads and unequal strides on the two axes, because a field
 * swapped between them computes a plausible surface wherever the axes agree.
 *
 * Usage: conv_transpose_rocket            (built-in shape sweep)
 *        conv_transpose_rocket IC IH IW OC KH KW SY SX PT PL OPY OPX DY DX  (one shape)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "test_fill.h"

/* Independent host lowering: dilate+pad the input, rot180+transpose the weights, run
 * the forward-conv CPU oracle. Returns the forward conv output (== the transpose). */
static void lowering_oracle(const rocket_conv_transpose2d_desc *d,
                            const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    const int IC = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    const int ly = d->dil_y * (KH - 1) - d->pad_top;
    const int lx = d->dil_x * (KW - 1) - d->pad_left;
    const int ty = ly + d->opad_y, tx = lx + d->opad_x;
    const int IHd = (IH - 1) * d->stride_y + 1 + ly + ty;
    const int IWd = (IW - 1) * d->stride_x + 1 + lx + tx;

    const int DW = d->depthwise;
    _Float16 *xd = calloc((size_t)IC * IHd * IWd, sizeof(_Float16));
    _Float16 *wf = malloc((DW ? (size_t)IC * KH * KW : (size_t)OC * IC * KH * KW) * sizeof(_Float16));
    if (!xd || !wf) { free(xd); free(wf); return; }

    for (int ic = 0; ic < IC; ic++)
        for (int ih = 0; ih < IH; ih++)
            for (int iw = 0; iw < IW; iw++)
                xd[((size_t)ic * IHd + (ly + ih * d->stride_y)) * IWd + (lx + iw * d->stride_x)] =
                    in[((size_t)ic * IH + ih) * IW + iw];

    if (DW)
        for (int c = 0; c < IC; c++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    wf[((size_t)c * KH + kh) * KW + kw] =
                        W[((size_t)c * KH + (KH - 1 - kh)) * KW + (KW - 1 - kw)];
    else
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < IC; ic++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        wf[(((size_t)oc * IC + ic) * KH + kh) * KW + kw] =
                            W[(((size_t)ic * OC + oc) * KH + (KH - 1 - kh)) * KW + (KW - 1 - kw)];

    rocket_conv2d_desc fwd = { .ic = IC, .ih = IHd, .iw = IWd, .oc = OC, .kh = KH, .kw = KW,
        .stride_y = 1, .stride_x = 1, .pad_top = 0, .pad_left = 0,
        .dil_y = d->dil_y, .dil_x = d->dil_x, .depthwise = DW };
    rocket_conv2d_ref_fp16(&fwd, xd, wf, out);
    free(xd); free(wf);
}

/* Cases that actually reached the driver and were compared against the reference.
 * A per-case planner refusal returns 0, so without this a run in which EVERY shape
 * was refused would print N "skipping" lines and exit PASS over zero evidence. */
static int g_checked = 0;
static int g_deconv = 0;       /* of those, the ones on the hardware deconvolution route */

/* One seed per shape: every field of the descriptor goes into it, so two shapes never
 * share a fill by accident. */
static uint64_t shape_seed(const rocket_conv_transpose2d_desc *d)
{
    const int f[] = { d->ic, d->ih, d->iw, d->oc, d->kh, d->kw, d->stride_y, d->stride_x,
                      d->pad_top, d->pad_left, d->opad_y, d->opad_x, d->dil_y, d->dil_x,
                      d->depthwise };
    uint64_t h = 0x243F6A8885A308D3ull;
    for (size_t i = 0; i < sizeof f / sizeof f[0]; i++) h = tf_hash(h, (uint64_t)(int64_t)f[i]);
    return h;
}

/* Run the entry with ROCKET_CONV_TRANSPOSE_HW forced to `value`, then put back whatever
 * the caller had: unsetting it would change the route of every later shape. */
static int run_with_route_env(int fd, const rocket_conv_transpose2d_desc *d, const char *value,
                              const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    const char *prev = getenv("ROCKET_CONV_TRANSPOSE_HW");
    char *saved = prev ? strdup(prev) : NULL;
    setenv("ROCKET_CONV_TRANSPOSE_HW", value, 1);
    int r = rocket_conv_transpose2d_fp16(fd, d, in, W, out);
    if (saved) { setenv("ROCKET_CONV_TRANSPOSE_HW", saved, 1); free(saved); }
    else unsetenv("ROCKET_CONV_TRANSPOSE_HW");
    return r;
}

static int run_shape(int fd, const rocket_conv_transpose2d_desc *d)
{
    int OH = rocket_conv_transpose2d_oh(d), OW = rocket_conv_transpose2d_ow(d);
    printf("%sconvT IC=%d IH=%d IW=%d -> OC=%d  K=%dx%d s=%dx%d p=%d,%d op=%d,%d d=%dx%d  (OH=%d OW=%d)\n",
           d->depthwise ? "DEPTHWISE " : "",
           d->ic, d->ih, d->iw, d->oc, d->kh, d->kw, d->stride_y, d->stride_x,
           d->pad_top, d->pad_left, d->opad_y, d->opad_x, d->dil_y, d->dil_x, OH, OW);

    int plan = rocket_conv_transpose2d_plan(d);
    if (plan) { printf("  plan: unsupported (%d) — skipping\n", plan); return 0; }
    const int route = rocket_conv_transpose2d_route(d);
    printf("  route: %s\n", route == ROCKET_CONV_TRANSPOSE_DECONV
                              ? "hardware deconvolution mode" : "lowering");

    size_t in_n  = (size_t)d->ic * d->ih * d->iw;
    /* W is [IC][OC][KH][KW] direct, [C][1][KH][KW] depthwise */
    size_t wt_n  = d->depthwise ? (size_t)d->ic * d->kh * d->kw : (size_t)d->ic * d->oc * d->kh * d->kw;
    size_t out_n = (size_t)d->oc * OH * OW;
    _Float16 *in  = malloc(in_n * sizeof(_Float16));
    _Float16 *W   = malloc(wt_n * sizeof(_Float16));
    _Float16 *out = malloc(out_n * sizeof(_Float16));
    _Float16 *ref = malloc(out_n * sizeof(_Float16));
    _Float16 *low = malloc(out_n * sizeof(_Float16));
    if (!in || !W || !out || !ref || !low) { fprintf(stderr, "oom\n"); return -1; }

    const uint64_t seed = shape_seed(d);
    tf_fill_f16_int(in, in_n, tf_hash(seed, 1), -4, 4);
    tf_fill_f16_int(W,  wt_n, tf_hash(seed, 2), -3, 3);
    const int dims[3] = { d->oc, OH, OW };

    int fail = 0;

    /* (1) lowering self-check: independent host lowering vs direct scatter definition */
    tf_sentinel_f16(ref, out_n);
    tf_sentinel_f16(low, out_n);
    rocket_conv_transpose2d_ref_fp16(d, in, W, ref);
    lowering_oracle(d, in, W, low);
    {
        /* The exact comparisons below hold only while every sum is an fp16 integer. */
        double peak = 0;
        for (size_t i = 0; i < out_n; i++)
            if (fabs((float)ref[i]) > peak) peak = fabs((float)ref[i]);
        if (!(peak < 2048.0)) {
            printf("  max|ref| = %.0f is not below 2048, so fp16 is not exact here and the "
                   "fill range is wrong for this shape (FAIL)\n", peak);
            fail = 1;
        }
        long bad = tf_cmp_f16("  lowering self-check", low, ref, dims, 3);
        printf("  lowering self-check: max|ref|=%.0f -> %s\n", peak, bad ? "FAIL" : "PASS");
        if (bad) fail = 1;
    }

    /* (2) on hardware (or CPU-tiled forward conv when fd<0): NPU vs scatter reference,
     * exactly, into a sentinel-filled buffer */
    {
        const char *tag = (fd >= 0) ? "HW end-to-end" : "CPU-lowered decomp";
        tf_sentinel_f16(out, out_n);
        int r = rocket_conv_transpose2d_fp16(fd, d, in, W, out);
        if (r) { printf("  %s: rocket_conv_transpose2d_fp16 = %d (FAIL)\n", tag, r); fail = 1; }
        else {
            g_checked++;
            long bad = tf_cmp_f16(fd >= 0 ? "  HW end-to-end" : "  CPU-lowered decomp",
                                  out, ref, dims, 3);
            printf("  %s: %ld of %zu elements differ -> %s\n", tag, bad, out_n,
                   bad ? "FAIL" : "PASS");
            if (bad) fail = 1;
        }
    }

    /* (3) the hardware route against the lowering, on the same shape, exactly */
    if (fd >= 0 && route == ROCKET_CONV_TRANSPOSE_DECONV && !fail) {
        g_deconv++;
        tf_sentinel_f16(low, out_n);
        int r = run_with_route_env(fd, d, "0", in, W, low);
        if (r) { printf("  lowering A/B: rocket_conv_transpose2d_fp16 = %d (FAIL)\n", r); fail = 1; }
        else {
            long bad = tf_cmp_f16("  lowering A/B", low, out, dims, 3);
            printf("  lowering A/B: %s -> %s\n", bad ? "differs" : "identical",
                   bad ? "FAIL" : "PASS");
            if (bad) fail = 1;
        }
    }

    free(in); free(W); free(out); free(ref); free(low);
    return fail;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0)
        printf("note: no /dev/accel/accel0 (%d) — running lowering self-check + CPU decomp only (no HW)\n\n", fd);

    int fail = 0;

    if (argc == 15 || argc == 16) {
        rocket_conv_transpose2d_desc d = { .ic=atoi(argv[1]),.ih=atoi(argv[2]),.iw=atoi(argv[3]),
            .oc=atoi(argv[4]),.kh=atoi(argv[5]),.kw=atoi(argv[6]),
            .stride_y=atoi(argv[7]),.stride_x=atoi(argv[8]),
            .pad_top=atoi(argv[9]),.pad_left=atoi(argv[10]),
            .opad_y=atoi(argv[11]),.opad_x=atoi(argv[12]),
            .dil_y=atoi(argv[13]),.dil_x=atoi(argv[14]),
            .depthwise=(argc==16 ? atoi(argv[15]) : 0) };
        fail = run_shape(fd, &d);
    } else {
        rocket_conv_transpose2d_desc shapes[] = {
            /* the classic learned 2x upsample: stride 2, k=2, no pad (decoder workhorse) */
            { .ic=32,.ih=8,.iw=8,.oc=16,.kh=2,.kw=2,.stride_y=2,.stride_x=2,.dil_y=1,.dil_x=1 },
            /* stride 2, k=4, pad 1: the common "clean 2x" decoder block */
            { .ic=32,.ih=8,.iw=8,.oc=16,.kh=4,.kw=4,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },
            /* stride 1, k=3, no pad: the "full" transpose (grows by K-1) */
            { .ic=32,.ih=8,.iw=8,.oc=16,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.dil_y=1,.dil_x=1 },
            /* stride 2, k=3, pad 1, output_padding 1: odd output-size disambiguation */
            { .ic=64,.ih=7,.iw=9,.oc=32,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.opad_y=1,.opad_x=1,.dil_y=1,.dil_x=1 },
            /* stride 3, k=3, no pad: 3x upsample */
            { .ic=32,.ih=6,.iw=6,.oc=16,.kh=3,.kw=3,.stride_y=3,.stride_x=3,.dil_y=1,.dil_x=1 },
            /* dilation 2, k=3, stride 1 */
            { .ic=32,.ih=8,.iw=8,.oc=16,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.dil_y=2,.dil_x=2 },
            /* multi-group IC/OC (IC=64 -> 2 K-groups, OC=48 -> 3 oc-groups), stride 2 k=2 */
            { .ic=64,.ih=6,.iw=6,.oc=48,.kh=2,.kw=2,.stride_y=2,.stride_x=2,.dil_y=1,.dil_x=1 },
            /* asymmetric kernel + stride: 1x3, stride 1x2 */
            { .ic=32,.ih=8,.iw=7,.oc=16,.kh=1,.kw=3,.stride_y=1,.stride_x=2,.pad_left=1,.dil_y=1,.dil_x=1 },
            /* narrow input channel count (IC=3), zero-padded to 32 by the driver */
            { .ic=3,.ih=8,.iw=8,.oc=16,.kh=4,.kw=4,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },
            /* large: stride 2 k=2 over 32x32 -> 64x64 output (dilated input ~63x63 forces tiling) */
            { .ic=64,.ih=32,.iw=32,.oc=64,.kh=2,.kw=2,.stride_y=2,.stride_x=2,.dil_y=1,.dil_x=1 },
            /* DEPTHWISE transpose (per-channel; the resize substrate). C%32==0. */
            /* nearest 2x: box kernel s=2 k=2 (each pixel -> 2x2 block) */
            { .ic=32,.ih=8,.iw=8,.oc=32,.kh=2,.kw=2,.stride_y=2,.stride_x=2,.dil_y=1,.dil_x=1,.depthwise=1 },
            /* bilinear-shaped 2x: k=4 s=2 pad 1 per channel */
            { .ic=64,.ih=8,.iw=8,.oc=64,.kh=4,.kw=4,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1,.depthwise=1 },
            /* DW 3x upsample, asymmetric input */
            { .ic=32,.ih=6,.iw=9,.oc=32,.kh=3,.kw=3,.stride_y=3,.stride_x=3,.dil_y=1,.dil_x=1,.depthwise=1 },
            /* The hardware deconvolution route at decoder sizes, where the lowering's dilated
             * input overflows one CBUF pass and tiles: 2x k4 p1 at 64 channels, 4x, an OC that
             * is not a multiple of 16, a per-axis stride and a 2x with output_padding. */
            { .ic=64,.ih=32,.iw=32,.oc=64,.kh=4,.kw=4,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },
            { .ic=32,.ih=16,.iw=16,.oc=32,.kh=4,.kw=4,.stride_y=4,.stride_x=4,.dil_y=1,.dil_x=1 },
            { .ic=32,.ih=24,.iw=20,.oc=24,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.opad_y=1,.opad_x=1,.dil_y=1,.dil_x=1 },
            { .ic=32,.ih=12,.iw=12,.oc=32,.kh=3,.kw=3,.stride_y=2,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },
            { .ic=128,.ih=16,.iw=16,.oc=64,.kh=4,.kw=4,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },
            /* The hardware route with the two axes different in every field it programs:
             * a 4x2 kernel with pads (1,0), and a 2x4 kernel with pads (0,1), strides (2,4)
             * and output_padding on one axis only. */
            { .ic=32,.ih=10,.iw=14,.oc=48,.kh=4,.kw=2,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=0,.dil_y=1,.dil_x=1 },
            { .ic=64,.ih=12,.iw=8,.oc=32,.kh=2,.kw=4,.stride_y=2,.stride_x=4,.pad_top=0,.pad_left=1,.opad_y=1,.dil_y=1,.dil_x=1 },
            /* The same asymmetry on the lowering: a 3x5 kernel, pads (1,2), output_padding
             * on x only. */
            { .ic=48,.ih=9,.iw=13,.oc=40,.kh=3,.kw=5,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=2,.opad_x=1,.dil_y=1,.dil_x=1 },
        };
        for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++) {
            fail |= run_shape(fd, &shapes[i]);
            printf("\n");
        }
    }

    if (fd >= 0) rocket_close(fd);
    if (g_checked == 0) {
        printf("no shape reached a numeric check — every case was refused or "
               "skipped; this gate proved nothing\n");
        fail = 1;
    }
    printf("%d shapes checked, %d of them on the hardware deconvolution route\n",
           g_checked, g_deconv);
    {
        const char *e = getenv("ROCKET_CONV_TRANSPOSE_HW");
        if (fd >= 0 && argc != 15 && argc != 16 && g_deconv == 0 && !(e && *e == '0') &&
            strcmp(rocket_hw_current()->name, "rk3588") == 0) {
            printf("the sweep's power-of-two shapes all LOWERED on an RK3588: the hardware "
                   "route was never exercised\n");
            fail = 1;
        }
    }
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : (fd < 0 ? 2 : 0);
}
