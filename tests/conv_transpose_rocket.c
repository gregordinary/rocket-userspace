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
 *     reference. This confirms the lowered forward conv runs HW-exact (the on-device task).
 *
 * Small integer inputs keep every result exact in fp16 (|sum| < 2048), so the bar is
 * max_abs == 0 for the lowering self-check and <= 1.0 for the HW path (fp16 narrowing).
 *
 * The sweep exercises stride 1/2/3, pad 0..K-1, output_padding, dilation>1, asymmetric
 * kernels, IC/OC crossing the cube group boundaries, an RGB-width input, and a large
 * shape that forces the forward conv to tile.
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

static int run_shape(int fd, const rocket_conv_transpose2d_desc *d)
{
    int OH = rocket_conv_transpose2d_oh(d), OW = rocket_conv_transpose2d_ow(d);
    printf("%sconvT IC=%d IH=%d IW=%d -> OC=%d  K=%dx%d s=%dx%d p=%d,%d op=%d,%d d=%dx%d  (OH=%d OW=%d)\n",
           d->depthwise ? "DEPTHWISE " : "",
           d->ic, d->ih, d->iw, d->oc, d->kh, d->kw, d->stride_y, d->stride_x,
           d->pad_top, d->pad_left, d->opad_y, d->opad_x, d->dil_y, d->dil_x, OH, OW);

    int plan = rocket_conv_transpose2d_plan(d);
    if (plan) { printf("  plan: unsupported (%d) — skipping\n", plan); return 0; }

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

    for (size_t i = 0; i < in_n; i++) in[i] = (_Float16)((int)(i % 5) - 2);
    for (size_t i = 0; i < wt_n; i++) W[i]  = (_Float16)((int)(i % 3) - 1);

    int fail = 0;

    /* (1) lowering self-check: independent host lowering vs direct scatter definition */
    rocket_conv_transpose2d_ref_fp16(d, in, W, ref);
    lowering_oracle(d, in, W, low);
    {
        double max_abs = 0; int bad = 0;
        for (size_t i = 0; i < out_n; i++) {
            double ad = fabs((float)low[i] - (float)ref[i]);
            if (ad > max_abs) max_abs = ad;
            if (ad != 0.0 && bad < 6) {
                printf("    low[%zu] scatter=%.2f lowered=%.2f\n", i, (float)ref[i], (float)low[i]); bad++;
            }
        }
        printf("  lowering self-check: max_abs=%.4f -> %s\n", max_abs, max_abs == 0.0 ? "PASS" : "FAIL");
        if (max_abs != 0.0) fail = 1;
    }

    /* (2) on hardware (or CPU-tiled forward conv when fd<0): NPU vs scatter reference */
    {
        const char *tag = (fd >= 0) ? "HW end-to-end" : "CPU-lowered decomp";
        memset(out, 0, out_n * sizeof(_Float16));
        int r = rocket_conv_transpose2d_fp16(fd, d, in, W, out);
        if (r) { printf("  %s: rocket_conv_transpose2d_fp16 = %d (FAIL)\n", tag, r); fail = 1; }
        else {
            double max_abs = 0; int bad = 0;
            for (size_t i = 0; i < out_n; i++) {
                double ad = fabs((float)out[i] - (float)ref[i]);
                if (ad > max_abs) max_abs = ad;
                if (ad > 1.0 && bad < 6) {
                    printf("    [%zu] ref=%.2f got=%.2f\n", i, (float)ref[i], (float)out[i]); bad++;
                }
            }
            printf("  %s: max_abs=%.3f -> %s\n", tag, max_abs, max_abs <= 1.0 ? "PASS" : "FAIL");
            if (max_abs > 1.0) fail = 1;
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
        };
        for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++) {
            fail |= run_shape(fd, &shapes[i]);
            printf("\n");
        }
    }

    if (fd >= 0) rocket_close(fd);
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : (fd < 0 ? 2 : 0);
}
