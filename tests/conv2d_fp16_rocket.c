// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv2d_fp16_rocket.c — standalone test for the general fp16 CONV_2D path.
 *
 * Two layers, mirroring how the dtype matmul generators were brought up:
 *
 *  1. CUBE-LAYOUT SELF-CHECK (runs anywhere, incl. x86, no NPU). Scatters the
 *     input feature + weights into the native NPU cube layouts (feature_data,
 *     weight_conv_fp16), then computes the convolution by reading THROUGH those
 *     same index functions exactly as the CNA would (gather the KH*KW*IC window
 *     per output position from the feature cube + the weight from the weight cube),
 *     writes the output cube, de-scatters it, and compares bit-for-bit (fp32
 *     accumulate) to a naive NCHW oracle. A PASS proves the scatter / index /
 *     de-scatter math is self-consistent and correctly expresses the conv — the
 *     part that is verifiable off-hardware. It also smoke-checks that
 *     gen_conv2d_fp16 emits a non-empty regcmd for the shape.
 *
 *  2. ON-HARDWARE END-TO-END (only if /dev/accel/accel0 opens). Runs
 *     rocket_conv2d_fp16 on the NPU and compares to the same oracle. This is the
 *     gate that confirms the register fields (feature_grains / data_entries / the
 *     dilation+stride+pad geometry) are HW-correct — the on-device task.
 *
 * The shape set deliberately exercises multiple KH/KW, multiple ic-groups (IC>32),
 * multiple oc-groups (OC>16), stride>1, pad>0, and dilation>1 — so no degenerate
 * single-group case can hide a wrong layout (the tf32/int4 lesson).
 *
 * Usage: conv2d_fp16_rocket            (run the built-in shape sweep)
 *        conv2d_fp16_rocket IC IH IW OC KH KW SY SX PT PL DY DX  (one custom shape)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "npu_matmul.h"   /* feature_data, weight_conv_fp16, gen_conv2d_fp16 */

static int dwg(void) { const char *e = getenv("ROCKET_CONV_DW_GROUP"); int g = e?atoi(e):32; return g>0?g:32; }

/* Compute the conv ENTIRELY in the NPU cube domain: scatter -> gather-through-
 * index-math -> descatter. If this equals the naive oracle, the layout is right.
 * Handles both the direct and depthwise weight cubes (DW reads one ic per oc). */
static int cube_self_check(const rocket_conv2d_desc *d,
                           const _Float16 *in, const _Float16 *W)
{
    const int IC = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int DW = d->depthwise, G = dwg();
    const int Cpad = DW ? ((IC + G - 1) / G) * G : IC;
    /* the cubes reserve full groups: direct weight K-group=32, oc-group=16, feature
     * C2=8 (so IC<32 like the RGB stem still indexes a padded 32-channel slot). */
    const int ICc = DW ? Cpad : ((IC + 31) / 32) * 32;
    const int OCc = ((OC + 15) / 16) * 16, OCo = ((OC + 7) / 8) * 8;

    _Float16 *in_cube  = calloc((size_t)ICc * IH * IW, sizeof(_Float16));
    _Float16 *wt_cube  = calloc(DW ? (size_t)Cpad*KH*KW : (size_t)OCc*ICc*KH*KW, sizeof(_Float16));
    _Float16 *out_cube = calloc((size_t)OCo * OH * OW, sizeof(_Float16));
    _Float16 *got      = calloc((size_t)OC * OH * OW, sizeof(_Float16));
    _Float16 *ref      = calloc((size_t)OC * OH * OW, sizeof(_Float16));
    if (!in_cube || !wt_cube || !out_cube || !got || !ref) { fprintf(stderr,"oom\n"); return -1; }

    /* scatter into native cubes (the exact host packing rocket_conv2d_fp16 does) */
    for (int ic = 0; ic < IC; ic++)
        for (int ih = 0; ih < IH; ih++)
            for (int iw = 0; iw < IW; iw++)
                in_cube[feature_data(IC, IH, IW, 8, ic+1, ih+1, iw+1)] =
                    in[((size_t)ic*IH+ih)*IW+iw];
    if (DW) {
        for (int c = 0; c < IC; c++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    wt_cube[weight_conv_dw_fp16(IC, KH, KW, G, c+1, kh+1, kw+1)] =
                        W[((size_t)c*KH+kh)*KW+kw];
    } else {
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < IC; ic++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        wt_cube[weight_conv_fp16(OC, IC, KH, KW, oc+1, ic+1, kh+1, kw+1)] =
                            W[(((size_t)oc*IC+ic)*KH+kh)*KW+kw];
    }

    /* the convolution, read through the cube indices (CNA's view) */
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                float s = 0.f;
                int ic_lo = DW ? oc : 0, ic_hi = DW ? oc+1 : IC;
                for (int ic = ic_lo; ic < ic_hi; ic++)
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh*d->stride_y + kh*d->dil_y - d->pad_top;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow*d->stride_x + kw*d->dil_x - d->pad_left;
                            if (iw < 0 || iw >= IW) continue;
                            float a = (float)in_cube[feature_data(IC, IH, IW, 8, ic+1, ih+1, iw+1)];
                            float w = DW
                                ? (float)wt_cube[weight_conv_dw_fp16(IC, KH, KW, G, oc+1, kh+1, kw+1)]
                                : (float)wt_cube[weight_conv_fp16(OC, IC, KH, KW, oc+1, ic+1, kh+1, kw+1)];
                            s += a * w;
                        }
                    }
                out_cube[feature_data(OC, OH, OW, 8, oc+1, oh+1, ow+1)] = (_Float16)s;
            }

    /* de-scatter + oracle */
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                got[((size_t)oc*OH+oh)*OW+ow] =
                    out_cube[feature_data(OC, OH, OW, 8, oc+1, oh+1, ow+1)];
    rocket_conv2d_ref_fp16(d, in, W, ref);

    double max_abs = 0; int bad = 0;
    for (size_t i = 0; i < (size_t)OC*OH*OW; i++) {
        double ad = fabs((float)got[i] - (float)ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad != 0.0 && bad < 6) { printf("    cube[%zu] ref=%.3f got=%.3f\n",
                                           i, (float)ref[i], (float)got[i]); bad++; }
    }
    free(in_cube); free(wt_cube); free(out_cube); free(got); free(ref);
    printf("  cube-self-check: OH=%d OW=%d max_abs=%.4f -> %s\n",
           OH, OW, max_abs, max_abs == 0.0 ? "PASS" : "FAIL");
    return max_abs == 0.0 ? 0 : 1;
}

static int run_shape(int fd, const rocket_conv2d_desc *d)
{
    int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    printf("%sconv IC=%d IH=%d IW=%d -> OC=%d  K=%dx%d s=%dx%d p=%d,%d d=%dx%d  (OH=%d OW=%d)\n",
           d->depthwise ? "DEPTHWISE " : "", d->ic, d->ih, d->iw, d->oc, d->kh, d->kw,
           d->stride_y, d->stride_x, d->pad_top, d->pad_left, d->dil_y, d->dil_x, OH, OW);

    int plan = rocket_conv2d_plan(d);
    if (plan) { printf("  plan: unsupported (%d) — skipping\n", plan); return 0; }

    size_t in_n = (size_t)d->ic*d->ih*d->iw;
    /* depthwise weight is [OC][1][KH][KW] (one filter per channel) */
    size_t wt_n = d->depthwise ? (size_t)d->oc*d->kh*d->kw : (size_t)d->oc*d->ic*d->kh*d->kw;
    size_t out_n = (size_t)d->oc*OH*OW;
    _Float16 *in = malloc(in_n*sizeof(_Float16)), *W = malloc(wt_n*sizeof(_Float16));
    _Float16 *out = malloc(out_n*sizeof(_Float16)), *ref = malloc(out_n*sizeof(_Float16));
    if (!in || !W || !out || !ref) { fprintf(stderr,"oom\n"); return -1; }

    /* small integer values keep results exact in fp16 */
    for (size_t i = 0; i < in_n; i++) in[i] = (_Float16)((int)(i % 5) - 2);
    for (size_t i = 0; i < wt_n; i++) W[i]  = (_Float16)((int)(i % 3) - 1);

    int fail = cube_self_check(d, in, W);

    /* regcmd smoke: the generator must emit a non-empty program */
    {
        uint64_t regs[256] = {0};
        conv_params_t p = { .ic=d->ic,.ih=d->ih,.iw=d->iw,.oc=d->oc,.oh=OH,.ow=OW,
            .kh=d->kh,.kw=d->kw,.stride_y=d->stride_y,.stride_x=d->stride_x,
            .dil_y=d->dil_y,.dil_x=d->dil_x,.pad_top=d->pad_top,.pad_left=d->pad_left,
            .input_dma=0x1000,.weights_dma=0x2000,.output_dma=0x3000,.tasks=regs,.fp32tofp16=1,
            .dw_group=(uint8_t)(d->depthwise ? dwg() : 0) };
        int g = d->depthwise ? gen_conv2d_dw_fp16(&p) : gen_conv2d_fp16(&p);
        /* g==-1/-2 = the WHOLE shape overflows one CBUF pass — expected for a
         * tiling shape (the driver tiles it); only other failures are real. */
        const char *gs = (g==0 && p.task_count>0) ? "OK"
                       : (g==-1 || g==-2) ? "needs tiling (OK)" : "FAIL";
        printf("  %s: ret=%d task_count=%u -> %s\n",
               d->depthwise ? "gen_conv2d_dw_fp16" : "gen_conv2d_fp16", g, p.task_count, gs);
        if (gs[0] == 'F') fail = 1;
    }

    /* Run the driver. With a device this is the NPU end-to-end; without one
     * (fd<0) it is the pure-CPU TILED path (conv2d_one_job falls back to the
     * oracle per tile) — so this validates the tiling decomposition on x86 too.
     * The label says which. -5 = depthwise gated (skip, not fail). */
    {
        const char *tag = (fd >= 0) ? "HW end-to-end" : "CPU-tiled decomp";
        memset(out, 0, out_n*sizeof(_Float16));
        int r = rocket_conv2d_fp16(fd, d, in, W, out);
        if (r == -5) { printf("  %s: depthwise gated (ROCKET_CONV_DW_NATIVE) — SKIP\n", tag); }
        else if (r) { printf("  %s: rocket_conv2d_fp16 = %d (FAIL)\n", tag, r); fail = 1; }
        else {
            rocket_conv2d_ref_fp16(d, in, W, ref);
            double max_abs = 0; int bad = 0;
            for (size_t i = 0; i < out_n; i++) {
                double ad = fabs((float)out[i] - (float)ref[i]);
                if (ad > max_abs) max_abs = ad;
                if (ad > 1.0 && bad < 6) { printf("    [%zu] ref=%.2f got=%.2f\n",
                                                  i,(float)ref[i],(float)out[i]); bad++; }
            }
            printf("  %s: max_abs=%.3f -> %s\n",
                   tag, max_abs, max_abs <= 1.0 ? "PASS" : "FAIL");
            if (max_abs > 1.0) fail = 1;
        }
    }

    free(in); free(W); free(out); free(ref);
    return fail;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0)
        printf("note: no /dev/accel/accel0 (%d) — running cube-layout self-check + "
               "regcmd smoke only (no HW)\n\n", fd);

    int fail = 0;

    if (argc == 13 || argc == 14) {
        /* IC IH IW OC KH KW SY SX PT PL DY DX [depthwise] */
        rocket_conv2d_desc d = { .ic=atoi(argv[1]),.ih=atoi(argv[2]),.iw=atoi(argv[3]),
            .oc=atoi(argv[4]),.kh=atoi(argv[5]),.kw=atoi(argv[6]),
            .stride_y=atoi(argv[7]),.stride_x=atoi(argv[8]),
            .pad_top=atoi(argv[9]),.pad_left=atoi(argv[10]),
            .dil_y=atoi(argv[11]),.dil_x=atoi(argv[12]),
            .depthwise=(argc==14 ? atoi(argv[13]) : 0) };
        fail = run_shape(fd, &d);
    } else {
        /* multi-Kh/Kw, multi ic/oc-group, stride>1, pad>0, dilation>1 sweep */
        rocket_conv2d_desc shapes[] = {
            { .ic=32,.ih=8, .iw=8, .oc=16,.kh=1,.kw=1,.stride_y=1,.stride_x=1,.pad_top=0,.pad_left=0,.dil_y=1,.dil_x=1 }, /* 1x1 == matmul */
            { .ic=32,.ih=8, .iw=8, .oc=16,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* same-pad 3x3 */
            { .ic=64,.ih=10,.iw=12,.oc=32,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* stride2, 2 ic + 2 oc groups */
            { .ic=32,.ih=9, .iw=9, .oc=16,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=2,.pad_left=2,.dil_y=2,.dil_x=2 }, /* dilation 2 */
            { .ic=64,.ih=7, .iw=11,.oc=48,.kh=1,.kw=5,.stride_y=1,.stride_x=1,.pad_top=0,.pad_left=2,.dil_y=1,.dil_x=1 }, /* asymmetric 1x5 */
            { .ic=96,.ih=6, .iw=6, .oc=16,.kh=5,.kw=3,.stride_y=2,.stride_x=1,.pad_top=2,.pad_left=1,.dil_y=1,.dil_x=1 }, /* 5x3, mixed stride, 3 ic groups */
            /* large: forces OH-band + OC tiling (feature 64*64*64*2=512KB > budget) */
            { .ic=64,.ih=64,.iw=64,.oc=128,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* tiled same-pad */
            { .ic=64,.ih=48,.iw=40,.oc=96,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* tiled stride2 */
            { .ic=256,.ih=8,.iw=256,.oc=32,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* very wide: forces OW (column) tiling */
            /* first layer: IC=3 (RGB), zero-padded to 32 by the driver */
            { .ic=3,.ih=16,.iw=16,.oc=16,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* RGB stem, stride2 */
            /* depthwise (OC==IC, default group G=64; the MobileNet workhorse) */
            { .ic=64,.ih=8, .iw=8, .oc=64,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1,.depthwise=1 }, /* 3x3 DW same-pad */
            { .ic=64,.ih=8, .iw=12,.oc=64,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1,.depthwise=1 }, /* 3x3 DW stride2 */
            { .ic=128,.ih=7,.iw=7, .oc=128,.kh=5,.kw=5,.stride_y=1,.stride_x=1,.pad_top=2,.pad_left=2,.dil_y=1,.dil_x=1,.depthwise=1 }, /* 5x5 DW, 2 groups */
        };
        for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++) {
            fail |= run_shape(fd, &shapes[i]);
            printf("\n");
        }
    }

    if (fd >= 0) rocket_close(fd);
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
