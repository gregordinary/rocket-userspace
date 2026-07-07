// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv2d_int8_rocket.c — standalone gate for the NATIVE int8 CONV_2D path
 * (gen_conv2d_int8 / gen_conv2d_dw_int8 + the int8 conv cube layouts). This is to
 * the int8 conv what conv2d_fp16_rocket.c is to the fp16 conv and what
 * matmul_int8_rocket.c is to the int8 matmul: nothing downstream (the rocket_conv.c
 * int8 runtime, the delegate per-axis requant, the A/B vs the fp16 approx) proceeds
 * until this PASSES on the NPU. int8 x int8 -> int32 is EXACT integer arithmetic, so
 * every comparison below is bit-exact (max_abs == 0), validated against an int64
 * accumulate reference (NOT float — a 7x7x512 conv sums to ~406M, past float's 2^24).
 *
 * Two layers, same structure as conv2d_fp16_rocket.c:
 *
 *  1. CUBE-LAYOUT SELF-CHECK (runs anywhere, incl. x86, no NPU). Scatters int8
 *     feature + weights into the native cubes (feature_data C2=16, weight_conv_int8
 *     oc-group 32 / ic-group 32, or weight_conv_dw_int8), computes the conv reading
 *     THROUGH those index functions exactly as the CNA would, writes the int32
 *     output cube (C2=4), de-scatters it, and compares bit-for-bit to a naive NCHW
 *     int64-accumulate oracle. A PASS proves the scatter/index/de-scatter math is
 *     self-consistent and correctly expresses the int8 conv. Covers ALL shapes,
 *     including OC%32 pad and IC<32 (the cube math is pure host).
 *
 *  2. ON-HARDWARE SINGLE JOB (only if /dev/accel/accel0 opens AND the shape fits one
 *     CBUF pass). Allocates the int8/int32 BOs, scatters, submits gen_conv2d_int8's
 *     regcmd, reads back the int32 cube, compares to the oracle. This is the gate
 *     that confirms the register fields (precision/size_e=7/surf*8/data_entries) are
 *     HW-correct — the bottom-up step before the tiling runtime exists. A shape that
 *     overflows the CBUF (gen returns <0) is HW-skipped (the runtime will tile it).
 *
 * Usage: conv2d_int8_rocket            (run the built-in shape sweep)
 *        conv2d_int8_rocket IC IH IW OC KH KW SY SX PT PL DY DX [depthwise]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "npu_matmul.h"   /* feature_data, weight_conv_int8, gen_conv2d_int8 */

#define CBUF_BANK 32768

/* int8 DW channel group: Mesa's value is 64; ROCKET_CONV_DW_GROUP sweeps it on HW
 * (the int8 DW geometry is unconfirmed, like the fp16 G=32 crack was). */
static int dwg(void) { const char *e = getenv("ROCKET_CONV_DW_GROUP"); int g = e?atoi(e):64; return g>0?g:64; }

static int rup(int x, int a) { return ((x + a - 1) / a) * a; }

/* int8 x int8 -> int32 NCHW reference (int64 accumulate, int32 store + range guard). */
static void ref_conv_int8(const rocket_conv2d_desc *d,
                          const int8_t *in, const int8_t *W, int32_t *out)
{
    int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    int IC = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t s = 0;
                int ic_lo = d->depthwise ? oc : 0, ic_hi = d->depthwise ? oc + 1 : IC;
                for (int ic = ic_lo; ic < ic_hi; ic++) {
                    int wic = d->depthwise ? 0 : ic, wic_span = d->depthwise ? 1 : IC;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * d->stride_y + kh * d->dil_y - d->pad_top;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * d->stride_x + kw * d->dil_x - d->pad_left;
                            if (iw < 0 || iw >= IW) continue;
                            int32_t a = in[((size_t)ic * IH + ih) * IW + iw];
                            int32_t w = W[(((size_t)oc * wic_span + wic) * KH + kh) * KW + kw];
                            s += (int64_t)a * w;
                        }
                    }
                }
                if (s > INT32_MAX || s < INT32_MIN)
                    fprintf(stderr, "  WARN: ref accum %lld exceeds int32 at oc=%d oh=%d ow=%d\n",
                            (long long)s, oc, oh, ow);
                out[((size_t)oc * OH + oh) * OW + ow] = (int32_t)s;
            }
}

/* Pure cube self-check: scatter -> gather-through-index-math (int64 accum) ->
 * descatter, compare EXACT to the NCHW oracle. Validates the layout for ANY shape. */
static int cube_self_check(const rocket_conv2d_desc *d, const int8_t *in, const int8_t *W)
{
    const int IC = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int DW = d->depthwise, G = dwg();
    const int Cpad = DW ? rup(IC, G) : IC;
    /* cubes reserve full groups: feature C2=16 (-> ceil(C/16)*16), direct weight
     * oc-group 32 / ic-group 32, int32 output cube C2=4. */
    const int ICc = DW ? Cpad : rup(IC, 32);     /* feature channels (also %16) */
    const int OCw = rup(OC, 32);                 /* weight oc-group 32 */
    const int OCo = rup(OC, 4);                  /* int32 output cube C2=4 */

    int8_t  *in_cube  = calloc((size_t)rup(ICc, 16) * IH * IW, 1);
    int8_t  *wt_cube  = calloc(DW ? (size_t)Cpad * KH * KW : (size_t)OCw * ICc * KH * KW, 1);
    int32_t *out_cube = calloc((size_t)OCo * OH * OW, sizeof(int32_t));
    int32_t *got      = calloc((size_t)OC * OH * OW, sizeof(int32_t));
    int32_t *ref      = calloc((size_t)OC * OH * OW, sizeof(int32_t));
    if (!in_cube || !wt_cube || !out_cube || !got || !ref) { fprintf(stderr, "oom\n"); return -1; }

    for (int ic = 0; ic < IC; ic++)
        for (int ih = 0; ih < IH; ih++)
            for (int iw = 0; iw < IW; iw++)
                in_cube[feature_data(IC, IH, IW, 16, ic + 1, ih + 1, iw + 1)] =
                    in[((size_t)ic * IH + ih) * IW + iw];
    if (DW) {
        for (int c = 0; c < IC; c++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    wt_cube[weight_conv_dw_int8(IC, KH, KW, G, c + 1, kh + 1, kw + 1)] =
                        W[((size_t)c * KH + kh) * KW + kw];
    } else {
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < IC; ic++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        wt_cube[weight_conv_int8(OC, IC, KH, KW, oc + 1, ic + 1, kh + 1, kw + 1)] =
                            W[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
    }

    /* the convolution, read through the cube indices (CNA's view), int64 accum */
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t s = 0;
                int ic_lo = DW ? oc : 0, ic_hi = DW ? oc + 1 : IC;
                for (int ic = ic_lo; ic < ic_hi; ic++)
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * d->stride_y + kh * d->dil_y - d->pad_top;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * d->stride_x + kw * d->dil_x - d->pad_left;
                            if (iw < 0 || iw >= IW) continue;
                            int32_t a = in_cube[feature_data(IC, IH, IW, 16, ic + 1, ih + 1, iw + 1)];
                            int32_t w = DW
                                ? wt_cube[weight_conv_dw_int8(IC, KH, KW, G, oc + 1, kh + 1, kw + 1)]
                                : wt_cube[weight_conv_int8(OC, IC, KH, KW, oc + 1, ic + 1, kh + 1, kw + 1)];
                            s += (int64_t)a * w;
                        }
                    }
                out_cube[feature_data(OC, OH, OW, 4, oc + 1, oh + 1, ow + 1)] = (int32_t)s;
            }

    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                got[((size_t)oc * OH + oh) * OW + ow] =
                    out_cube[feature_data(OC, OH, OW, 4, oc + 1, oh + 1, ow + 1)];
    ref_conv_int8(d, in, W, ref);

    int64_t max_abs = 0; int bad = 0;
    for (size_t i = 0; i < (size_t)OC * OH * OW; i++) {
        int64_t ad = (int64_t)got[i] - ref[i]; if (ad < 0) ad = -ad;
        if (ad > max_abs) max_abs = ad;
        if (ad != 0 && bad < 6) { printf("    cube[%zu] ref=%d got=%d\n", i, ref[i], got[i]); bad++; }
    }
    free(in_cube); free(wt_cube); free(out_cube); free(got); free(ref);
    printf("  cube-self-check: OH=%d OW=%d max_abs=%lld -> %s\n",
           OH, OW, (long long)max_abs, max_abs == 0 ? "PASS" : "FAIL");
    return max_abs == 0 ? 0 : 1;
}

/* Run ONE int8 conv as a single NPU job (must fit one CBUF pass). Direct: OC is
 * zero-padded up to 32 (the weight oc-group); the first OC output channels are read
 * back. Depthwise: OC==IC, channel group G. got[OC*OH*OW] receives the int32 output.
 * Returns 0 (ran), 1 (gen needs tiling -> HW-skip), <0 (error). */
static int run_hw_single_job(int fd, const rocket_conv2d_desc *d,
                             const int8_t *in, const int8_t *W, int32_t *got)
{
    const int IC = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int DW = d->depthwise, G = dwg();
    if (!DW && (IC % 32)) { printf("  HW: direct IC%%32!=0 (IC=%d) — runtime-only (cube-check covers); SKIP\n", IC); return 1; }
    if (DW && (IC % G))   { printf("  HW: depthwise IC%%G!=0 (IC=%d G=%d) — SKIP\n", IC, G); return 1; }

    const int OCp = DW ? OC : rup(OC, 32);       /* direct: pad OC kernels to 32 */
    const int OCcube = DW ? IC : OCp;            /* output cube channel count    */

    /* padded direct weight [OCp][IC][KH][KW] (extra kernels = 0); DW uses W as-is */
    int8_t *Wp = NULL;
    const int8_t *Wuse = W;
    if (!DW && OCp != OC) {
        Wp = calloc((size_t)OCp * IC * KH * KW, 1);
        if (!Wp) return -1;
        memcpy(Wp, W, (size_t)OC * IC * KH * KW);
        Wuse = Wp;
    }

    size_t in_elems  = (size_t)IC * IH * IW;                       /* C2=16, IC%16==0 */
    size_t wt_elems  = DW ? (size_t)rup(IC, G) * KH * KW : (size_t)OCp * IC * KH * KW;
    size_t out_elems = (size_t)OCcube * OH * OW;                   /* C2=4, OCcube%4==0 */

    rocket_bo guard = {0}, regcmd = {0}, input = {0}, weights = {0}, output = {0};
    uint64_t regs[256] = {0};
    int ret = 0, rc = -1;

    rocket_bo_alloc(fd, 4096, &guard);            /* reserve IOVA 0 */
    ret |= rocket_bo_alloc(fd, 256 * sizeof(uint64_t),               &regcmd);
    ret |= rocket_bo_alloc(fd, in_elems  + CBUF_BANK,                &input);
    ret |= rocket_bo_alloc(fd, wt_elems  + CBUF_BANK,                &weights);
    ret |= rocket_bo_alloc(fd, out_elems * sizeof(int32_t) + CBUF_BANK, &output);
    if (ret) { fprintf(stderr, "  HW: BO alloc failed\n"); goto out; }
    if (((input.dma_address + input.size) | (weights.dma_address + weights.size) |
         (output.dma_address + output.size) | (regcmd.dma_address + regcmd.size)) >> 32) {
        fprintf(stderr, "  HW: a BO dma_address exceeds 32 bits\n"); goto out;
    }

    /* scatter input feature -> NC1HWC2 cube, C2=16 (int8) */
    rocket_bo_prep(fd, &input, 1, 0);
    memset(input.ptr, 0, input.size);
    {
        int8_t *dst = input.ptr;
        for (int ic = 0; ic < IC; ic++)
            for (int ih = 0; ih < IH; ih++)
                for (int iw = 0; iw < IW; iw++)
                    dst[feature_data(IC, IH, IW, 16, ic + 1, ih + 1, iw + 1)] =
                        in[((size_t)ic * IH + ih) * IW + iw];
    }
    rocket_bo_fini(fd, &input);

    /* scatter weights -> int8 conv weight cube (direct oc-group 32; DW group G) */
    rocket_bo_prep(fd, &weights, 1, 0);
    memset(weights.ptr, 0, weights.size);
    {
        int8_t *dst = weights.ptr;
        if (DW) {
            for (int c = 0; c < IC; c++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        dst[weight_conv_dw_int8(IC, KH, KW, G, c + 1, kh + 1, kw + 1)] =
                            W[((size_t)c * KH + kh) * KW + kw];
        } else {
            for (int oc = 0; oc < OCp; oc++)
                for (int ic = 0; ic < IC; ic++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                            dst[weight_conv_int8(OCp, IC, KH, KW, oc + 1, ic + 1, kh + 1, kw + 1)] =
                                Wuse[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
        }
    }
    rocket_bo_fini(fd, &weights);

    /* regcmd */
    {
        conv_params_t p = {
            .ic = IC, .ih = IH, .iw = IW, .oc = OCp, .oh = OH, .ow = OW,
            .kh = KH, .kw = KW, .stride_y = d->stride_y, .stride_x = d->stride_x,
            .dil_y = d->dil_y, .dil_x = d->dil_x, .pad_top = d->pad_top, .pad_left = d->pad_left,
            .input_dma = (uint32_t)input.dma_address, .weights_dma = (uint32_t)weights.dma_address,
            .output_dma = (uint32_t)output.dma_address, .tasks = regs,
            .dw_group = (uint8_t)(DW ? G : 0),
        };
        int g = DW ? gen_conv2d_dw_int8(&p) : gen_conv2d_int8(&p);
        if (g == -1 || g == -2) { printf("  HW: gen needs tiling (%d) — runtime-only; SKIP\n", g); rc = 1; goto out; }
        if (g != 0) { fprintf(stderr, "  HW: gen failed (%d)\n", g); goto out; }
        rocket_bo_prep(fd, &regcmd, 1, 0);
        memcpy(regcmd.ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &regcmd);

        /* sentinel-fill output: 0xAA intact -> NPU never wrote */
        rocket_bo_prep(fd, &output, 1, 0);
        memset(output.ptr, 0xAA, output.size);
        rocket_bo_fini(fd, &output);

        rocket_task_desc task = { .regcmd = (uint32_t)regcmd.dma_address, .regcmd_count = p.task_count };
        uint32_t in_h[]  = { input.handle, weights.handle, regcmd.handle };
        uint32_t out_h[] = { output.handle };
        if ((ret = rocket_submit_tasks(fd, &task, 1, in_h, 3, out_h, 1)) != 0) {
            fprintf(stderr, "  HW: submit failed (%d)\n", ret); goto out;
        }
    }

    if ((ret = rocket_bo_prep(fd, &output, 0, 2000000000ULL)) != 0) {
        fprintf(stderr, "  HW: wait timeout (%d) — DPU completion never fired\n", ret); goto out;
    }
    {
        int touched = 0;
        for (size_t i = 0; i < output.size; i++)
            if (((uint8_t *)output.ptr)[i] != 0xAA) { touched = 1; break; }
        if (!touched) printf("  HW DIAGNOSTIC: output still 0xAA — NPU did not write\n");
        /* int32 output cube C2=4 (direct, HW-proven). ROCKET_CONV_OUT_C2 overrides
         * the de-scatter atom for the DW geometry sweep (the 2x-stride signature). */
        int out_c2 = 4;
        { const char *e2 = getenv("ROCKET_CONV_OUT_C2"); if (e2 && atoi(e2) > 0) out_c2 = atoi(e2); }
        int32_t *src = output.ptr;
        for (int oc = 0; oc < OC; oc++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                    got[((size_t)oc * OH + oh) * OW + ow] =
                        src[feature_data(OCcube, OH, OW, out_c2, oc + 1, oh + 1, ow + 1)];
    }
    rocket_bo_fini(fd, &output);
    rc = 0;

out:
    rocket_bo_free(fd, &output); rocket_bo_free(fd, &weights); rocket_bo_free(fd, &input);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &guard);
    free(Wp);
    return rc;
}
/* DIRECT int8 via the TILED RUNTIME (rocket_conv2d_int8): OC%32 / IC<32 pad + OC/OH/OW
 * spatial tiling, each tile an independent single job. This catches per-tile HW bugs
 * the fd<0 assembly oracle cannot (the fp16 datain_height<4 lesson) — so the big/wide/
 * IC<32 shapes below MUST run on /dev/accel/accel0. On x86 (fd<0) the runtime computes
 * each tile on the int64 oracle, validating the tiling DECOMPOSITION (band/halo/placement)
 * bit-exact vs the whole-image oracle. Bit-exact either way (int8->int32 is exact). */
static int run_tiled_runtime(int fd, const rocket_conv2d_desc *d)
{
    int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    printf("RUNTIME conv IC=%d IH=%d IW=%d -> OC=%d  K=%dx%d s=%dx%d p=%d,%d d=%dx%d (OH=%d OW=%d)\n",
           d->ic, d->ih, d->iw, d->oc, d->kh, d->kw, d->stride_y, d->stride_x,
           d->pad_top, d->pad_left, d->dil_y, d->dil_x, OH, OW);
    if (OH <= 0 || OW <= 0) { printf("  degenerate output — skip\n"); return 0; }

    size_t in_n  = (size_t)d->ic * d->ih * d->iw;
    size_t wt_n  = (size_t)d->oc * d->ic * d->kh * d->kw;
    size_t out_n = (size_t)d->oc * OH * OW;
    int8_t  *in = malloc(in_n), *W = malloc(wt_n);
    int32_t *got = malloc(out_n * sizeof(int32_t)), *ref = malloc(out_n * sizeof(int32_t));
    if (!in || !W || !got || !ref) { fprintf(stderr, "oom\n"); free(in); free(W); free(got); free(ref); return -1; }
    for (size_t i = 0; i < in_n; i++) in[i] = (int8_t)(rand() % 256 - 128);
    for (size_t i = 0; i < wt_n; i++) W[i]  = (int8_t)(rand() % 256 - 128);

    memset(got, 0, out_n * sizeof(int32_t));
    int r = rocket_conv2d_int8(fd, d, in, W, got);
    int fail = 0;
    if (r != 0) { printf("  rocket_conv2d_int8 = %d -> FAIL\n", r); fail = 1; }
    else {
        ref_conv_int8(d, in, W, ref);
        int64_t max_abs = 0; int bad = 0;
        for (size_t i = 0; i < out_n; i++) {
            int64_t ad = (int64_t)got[i] - ref[i]; if (ad < 0) ad = -ad;
            if (ad > max_abs) max_abs = ad;
            if (ad != 0 && bad < 6) { printf("    [%zu] ref=%d got=%d\n", i, ref[i], got[i]); bad++; }
        }
        printf("  %s: max_abs=%lld -> %s\n", fd >= 0 ? "HW tiled runtime" : "CPU-oracle tiling",
               (long long)max_abs, max_abs == 0 ? "PASS" : "FAIL");
        if (max_abs != 0) fail = 1;
    }
    free(in); free(W); free(got); free(ref);
    return fail;
}

static int run_shape(int fd, const rocket_conv2d_desc *d)
{
    int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    printf("%sconv IC=%d IH=%d IW=%d -> OC=%d  K=%dx%d s=%dx%d p=%d,%d d=%dx%d  (OH=%d OW=%d)\n",
           d->depthwise ? "DEPTHWISE " : "", d->ic, d->ih, d->iw, d->oc, d->kh, d->kw,
           d->stride_y, d->stride_x, d->pad_top, d->pad_left, d->dil_y, d->dil_x, OH, OW);
    if (OH <= 0 || OW <= 0) { printf("  degenerate output — skip\n"); return 0; }

    size_t in_n  = (size_t)d->ic * d->ih * d->iw;
    size_t wt_n  = d->depthwise ? (size_t)d->oc * d->kh * d->kw : (size_t)d->oc * d->ic * d->kh * d->kw;
    size_t out_n = (size_t)d->oc * OH * OW;
    int8_t  *in = malloc(in_n), *W = malloc(wt_n);
    int32_t *got = malloc(out_n * sizeof(int32_t)), *ref = malloc(out_n * sizeof(int32_t));
    if (!in || !W || !got || !ref) { fprintf(stderr, "oom\n"); free(in); free(W); free(got); free(ref); return -1; }

    /* full signed int8 range exercises data_sign=1 + the int32 accumulator width */
    for (size_t i = 0; i < in_n; i++) in[i] = (int8_t)(rand() % 256 - 128);
    for (size_t i = 0; i < wt_n; i++) W[i]  = (int8_t)(rand() % 256 - 128);

    int fail = cube_self_check(d, in, W);

    /* regcmd smoke */
    {
        uint64_t regs[256] = {0};
        conv_params_t p = { .ic=d->ic,.ih=d->ih,.iw=d->iw,.oc=rup(d->oc, d->depthwise?1:32),.oh=OH,.ow=OW,
            .kh=d->kh,.kw=d->kw,.stride_y=d->stride_y,.stride_x=d->stride_x,
            .dil_y=d->dil_y,.dil_x=d->dil_x,.pad_top=d->pad_top,.pad_left=d->pad_left,
            .input_dma=0x1000,.weights_dma=0x2000,.output_dma=0x3000,.tasks=regs,
            .dw_group=(uint8_t)(d->depthwise ? dwg() : 0) };
        int g = d->depthwise ? gen_conv2d_dw_int8(&p) : gen_conv2d_int8(&p);
        const char *gs = (g==0 && p.task_count>0) ? "OK" : (g==-1||g==-2) ? "needs tiling (OK)" : "FAIL";
        printf("  %s: ret=%d task_count=%u -> %s\n",
               d->depthwise ? "gen_conv2d_dw_int8" : "gen_conv2d_int8", g, p.task_count, gs);
        if (gs[0] == 'F') fail = 1;
    }

    /* HW single job (if device). DEPTHWISE int8 is UNPROVEN (its int32 output
     * geometry needs a Teflon ground-truth capture, like the fp16 DW crack — blind
     * register sweeping mis-led the fp16 DW G-search once), so a DW mismatch is a
     * DIAGNOSTIC, NOT a gate failure: the gate's verdict tracks the validated DIRECT
     * scope. The DW cube-self-check above DOES gate (pure layout math). */
    if (fd >= 0 && d->depthwise) {
        /* Depthwise int8 = the int8-OUT on-chip-requant path, CRACKED + HW-validated
         * bit-exact against Mesa/Teflon ground truth by the `replay_dw_mesa` test
         * (regcmd EMITs also match the Teflon capture). The correct oracle for this
         * op is Teflon itself (its NPU int8 DW diverges from CPU TFLite by up to ~143
         * on full-range random int8), so the bit-exact gate is capture-replay, not a
         * from-scratch software reference. Here the gate covers the cube-layout
         * self-check (above) + regcmd smoke (below). */
        printf("  HW: DW int8-out on-chip-requant — correctness gated by replay_dw_mesa (bit-exact vs Teflon)\n");
    } else if (fd >= 0) {
        memset(got, 0, out_n * sizeof(int32_t));
        int r = run_hw_single_job(fd, d, in, W, got);
        if (r == 0) {
            ref_conv_int8(d, in, W, ref);
            int64_t max_abs = 0; int bad = 0;
            for (size_t i = 0; i < out_n; i++) {
                int64_t ad = (int64_t)got[i] - ref[i]; if (ad < 0) ad = -ad;
                if (ad > max_abs) max_abs = ad;
                if (ad != 0 && bad < 6) { printf("    [%zu] ref=%d got=%d\n", i, ref[i], got[i]); bad++; }
            }
            if (max_abs == 0)
                printf("  HW end-to-end: max_abs=0 -> PASS\n");
            else { printf("  HW end-to-end: max_abs=%lld -> FAIL\n", (long long)max_abs); fail = 1; }
        } else if (r < 0) fail = 1;   /* r==1 is an intentional HW-skip, not a fail */
    }

    free(in); free(W); free(got); free(ref);
    return fail;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0)
        printf("note: no /dev/accel/accel0 (%d) — cube-layout self-check + regcmd smoke only (no HW)\n\n", fd);
    srand(1234);   /* deterministic */

    int fail = 0;
    if (argc == 13 || argc == 14) {
        rocket_conv2d_desc d = { .ic=atoi(argv[1]),.ih=atoi(argv[2]),.iw=atoi(argv[3]),
            .oc=atoi(argv[4]),.kh=atoi(argv[5]),.kw=atoi(argv[6]),
            .stride_y=atoi(argv[7]),.stride_x=atoi(argv[8]),
            .pad_top=atoi(argv[9]),.pad_left=atoi(argv[10]),
            .dil_y=atoi(argv[11]),.dil_x=atoi(argv[12]),
            .depthwise=(argc==14 ? atoi(argv[13]) : 0) };
        fail = run_shape(fd, &d);
    } else {
        rocket_conv2d_desc shapes[] = {
            /* DIRECT — int8 N-group(oc) is 32, so use OC%32==0 for clean single-job HW */
            { .ic=32,.ih=8, .iw=8, .oc=32,.kh=1,.kw=1,.stride_y=1,.stride_x=1,.pad_top=0,.pad_left=0,.dil_y=1,.dil_x=1 }, /* 1x1 == int8 matmul */
            { .ic=32,.ih=8, .iw=8, .oc=32,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* same-pad 3x3 */
            { .ic=64,.ih=10,.iw=12,.oc=64,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* stride2, 2 ic + 2 oc groups */
            { .ic=32,.ih=9, .iw=9, .oc=32,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=2,.pad_left=2,.dil_y=2,.dil_x=2 }, /* dilation 2 */
            { .ic=64,.ih=7, .iw=11,.oc=64,.kh=1,.kw=5,.stride_y=1,.stride_x=1,.pad_top=0,.pad_left=2,.dil_y=1,.dil_x=1 }, /* asymmetric 1x5 */
            { .ic=96,.ih=6, .iw=6, .oc=32,.kh=5,.kw=3,.stride_y=2,.stride_x=1,.pad_top=2,.pad_left=1,.dil_y=1,.dil_x=1 }, /* 5x3, mixed stride, 3 ic groups */
            /* OC%32 pad (cube-check covers; HW pads to 32) */
            { .ic=32,.ih=8, .iw=8, .oc=48,.kh=1,.kw=1,.stride_y=1,.stride_x=1,.pad_top=0,.pad_left=0,.dil_y=1,.dil_x=1 }, /* OC=48 -> pad 64 */
            /* IC<32 first layer (RGB) — cube-check only on HW (runtime pads IC) */
            { .ic=3,.ih=16,.iw=16,.oc=32,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* RGB stem */
            /* DEPTHWISE (OC==IC, group G; UNPROVEN int8 geometry — expect to sweep) */
            { .ic=64,.ih=8, .iw=8, .oc=64,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1,.depthwise=1 }, /* 3x3 DW same-pad */
            { .ic=64,.ih=8, .iw=12,.oc=64,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1,.depthwise=1 }, /* 3x3 DW stride2 */
            { .ic=128,.ih=7,.iw=7, .oc=128,.kh=5,.kw=5,.stride_y=1,.stride_x=1,.pad_top=2,.pad_left=2,.dil_y=1,.dil_x=1,.depthwise=1 }, /* 5x5 DW, 2 groups */
        };
        for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++) {
            fail |= run_shape(fd, &shapes[i]);
            printf("\n");
        }

        /* DIRECT TILED RUNTIME — big/wide/IC<32/OC-pad shapes that exceed one CBUF pass,
         * so they exercise the OC/OH/OW spatial tiler (NEVER run on HW before). Mirrors
         * the fp16 tiled gate's shapes; bit-exact vs the int64 oracle. */
        printf("---- DIRECT int8 TILED RUNTIME (rocket_conv2d_int8) ----\n\n");
        rocket_conv2d_desc big[] = {
            { .ic=64,.ih=64,.iw=64,.oc=128,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* OC+OH tiled */
            { .ic=256,.ih=8,.iw=256,.oc=32,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 }, /* OW-column tile (datain_height<4 risk) */
            { .ic=64,.ih=40,.iw=40,.oc=64,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },   /* stride-2 big */
            { .ic=3,.ih=64,.iw=64,.oc=32,.kh=3,.kw=3,.stride_y=2,.stride_x=2,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },    /* RGB stem, tiled */
            { .ic=32,.ih=40,.iw=40,.oc=48,.kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1 },   /* OC=48 pad + tile */
            { .ic=128,.ih=28,.iw=28,.oc=64,.kh=1,.kw=1,.stride_y=1,.stride_x=1,.pad_top=0,.pad_left=0,.dil_y=1,.dil_x=1 },  /* 1x1 big (matmul) */
        };
        for (size_t i = 0; i < sizeof(big)/sizeof(big[0]); i++) {
            fail |= run_tiled_runtime(fd, &big[i]);
            printf("\n");
        }
    }

    if (fd >= 0) rocket_close(fd);
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
