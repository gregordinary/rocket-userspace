// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * pool_fp16_rocket.c — standalone test for on-NPU fp16 MaxPool / AveragePool (the PPU).
 *
 * Three layers, mirroring the conv/matmul bring-ups:
 *
 *  1. CUBE-LAYOUT SELF-CHECK (runs anywhere, incl. x86, no NPU). Scatters the input into
 *     the NC1HWC2 cube (feature_data, C2=8), pools by reading the kernel window THROUGH
 *     those same cube indices (the PPU's view), writes + de-scatters the output cube, and
 *     compares to a naive NCHW oracle. A PASS proves the scatter / window-index /
 *     de-scatter math expresses the pool — the part verifiable off-hardware.
 *
 *  2. REGCMD SMOKE: gen_pool_fp16 emits a non-empty program for the shape.
 *
 *  3. ON-HARDWARE END-TO-END (only if /dev/accel/accel0 opens). rocket_pool_fp16 on the
 *     NPU vs the oracle: MAX is bit-exact; AVG is within tolerance (the PPU divides via a
 *     per-axis fp16(65536/k) reciprocal).
 *
 * The sweep exercises max & avg, kernels 2/3, stride 1/2, single- and multi-C-plane
 * (C=8/16/24), global pooling, and (max) padding.
 *
 * Usage: pool_fp16_rocket
 *        pool_fp16_rocket METHOD C IH IW KH KW SY SX PT PL PB PR  (one shape; METHOD max|avg)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_pool.h"
#include "npu_matmul.h"   /* feature_data */

static const char *mname(int m) { return m == POOL_METHOD_MAX ? "max" : "avg"; }

/* Pool entirely in the cube domain: scatter -> window-gather-through-index -> descatter.
 * Equals the naive oracle iff the layout is right. (True average; the HW recip is a
 * separate HW-only concern checked in layer 3.) */
static int cube_self_check(const rocket_pool_desc *d, const _Float16 *in)
{
    const int C = d->c, IH = d->ih, IW = d->iw;
    const int OH = rocket_pool_oh(d), OW = rocket_pool_ow(d);
    const int C1 = (C + 7) / 8;

    _Float16 *in_cube  = calloc((size_t)C1 * IH * IW * 8, sizeof(_Float16));
    _Float16 *out_cube = calloc((size_t)C1 * OH * OW * 8, sizeof(_Float16));
    _Float16 *got      = calloc((size_t)C * OH * OW, sizeof(_Float16));
    _Float16 *ref      = calloc((size_t)C * OH * OW, sizeof(_Float16));
    if (!in_cube || !out_cube || !got || !ref) { fprintf(stderr, "oom\n"); return -1; }

    for (int c = 0; c < C; c++)
        for (int h = 0; h < IH; h++)
            for (int w = 0; w < IW; w++)
                in_cube[feature_data(C, IH, IW, 8, c+1, h+1, w+1)] = in[((size_t)c*IH+h)*IW+w];

    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                float acc = (d->method == POOL_METHOD_MAX) ? -INFINITY : 0.f;
                for (int kh = 0; kh < d->kh; kh++) {
                    int ih = oh*d->stride_y + kh - d->pad_top;
                    if (ih < 0 || ih >= IH) continue;
                    for (int kw = 0; kw < d->kw; kw++) {
                        int iw = ow*d->stride_x + kw - d->pad_left;
                        if (iw < 0 || iw >= IW) continue;
                        float v = (float)in_cube[feature_data(C, IH, IW, 8, c+1, ih+1, iw+1)];
                        if (d->method == POOL_METHOD_MAX) { if (v > acc) acc = v; }
                        else acc += v;
                    }
                }
                float o = (d->method == POOL_METHOD_AVG) ? acc/(float)(d->kh*d->kw) : acc;
                out_cube[feature_data(C, OH, OW, 8, c+1, oh+1, ow+1)] = (_Float16)o;
            }

    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                got[((size_t)c*OH+oh)*OW+ow] = out_cube[feature_data(C, OH, OW, 8, c+1, oh+1, ow+1)];
    rocket_pool_ref_fp16(d, in, ref);

    double max_abs = 0;
    for (size_t i = 0; i < (size_t)C*OH*OW; i++) {
        double ad = fabs((float)got[i] - (float)ref[i]);
        if (ad > max_abs) max_abs = ad;
    }
    free(in_cube); free(out_cube); free(got); free(ref);
    printf("  cube-self-check: OH=%d OW=%d max_abs=%.5f -> %s\n",
           OH, OW, max_abs, max_abs == 0.0 ? "PASS" : "FAIL");
    return max_abs == 0.0 ? 0 : 1;
}

static int run_shape(int fd, const rocket_pool_desc *d)
{
    int OH = rocket_pool_oh(d), OW = rocket_pool_ow(d);
    printf("%s C=%d %dx%d  k=%dx%d s=%dx%d p=%d,%d,%d,%d -> %dx%d\n",
           mname(d->method), d->c, d->ih, d->iw, d->kh, d->kw,
           d->stride_y, d->stride_x, d->pad_top, d->pad_left, d->pad_bottom, d->pad_right, OH, OW);

    int plan = rocket_pool_fp16_plan(d);
    if (plan) { printf("  plan: unsupported (%d) — skipping\n", plan); return 0; }

    size_t in_n = (size_t)d->c*d->ih*d->iw, out_n = (size_t)d->c*OH*OW;
    _Float16 *in = malloc(in_n*sizeof(_Float16)), *out = malloc(out_n*sizeof(_Float16));
    _Float16 *ref = malloc(out_n*sizeof(_Float16));
    if (!in || !out || !ref) { fprintf(stderr, "oom\n"); return -1; }
    /* distinct small values; integers keep max bit-exact and average near-exact in fp16 */
    for (size_t i = 0; i < in_n; i++) in[i] = (_Float16)((int)(i % 13) - 6);

    int fail = cube_self_check(d, in);

    {   /* regcmd smoke */
        uint64_t regs[64] = {0};
        pool_params_t p = { .c=d->c,.ih=d->ih,.iw=d->iw,.oh=OH,.ow=OW,.kh=d->kh,.kw=d->kw,
            .stride_y=d->stride_y,.stride_x=d->stride_x,.pad_top=d->pad_top,.pad_left=d->pad_left,
            .pad_bottom=d->pad_bottom,.pad_right=d->pad_right,.method=(uint8_t)d->method,
            .recip_w=ppu_recip_kernel_fp16(d->kw),.recip_h=ppu_recip_kernel_fp16(d->kh),
            .input_dma=0x1000,.output_dma=0x3000,.tasks=regs };
        int g = gen_pool_fp16(&p);
        const char *gs = (g==0 && p.task_count>0) ? "OK" : "FAIL";
        printf("  gen_pool_fp16: ret=%d task_count=%u -> %s\n", g, p.task_count, gs);
        if (gs[0]=='F') fail = 1;
    }

    {   /* run the driver: HW end-to-end (fd>=0) or CPU oracle (fd<0) */
        const char *tag = (fd >= 0) ? "HW end-to-end" : "CPU fallback";
        memset(out, 0, out_n*sizeof(_Float16));
        int r = rocket_pool_fp16(fd, d, in, out);
        if (r) { printf("  %s: rocket_pool_fp16 = %d (FAIL)\n", tag, r); fail = 1; }
        else {
            rocket_pool_ref_fp16(d, in, ref);
            /* MAX is exact; AVG within tolerance (fp16(65536/k) recip + fp16 rounding). */
            double tol = (d->method == POOL_METHOD_MAX) ? 0.0 : 0.06;
            double max_abs = 0; int bad = 0;
            for (size_t i = 0; i < out_n; i++) {
                double ad = fabs((float)out[i] - (float)ref[i]);
                if (ad > max_abs) max_abs = ad;
                if (ad > tol && bad < 6) {
                    printf("    [%zu] ref=%.4f got=%.4f d=%.4f\n",
                           i, (float)ref[i], (float)out[i], ad); bad++;
                }
            }
            printf("  %s: max_abs=%.5f (tol=%.3f) -> %s\n",
                   tag, max_abs, tol, max_abs <= tol ? "PASS" : "FAIL");
            if (max_abs > tol) fail = 1;
        }
    }
    free(in); free(out); free(ref);
    return fail;
}

static int parse_method(const char *s)
{ return (s[0]=='m' && s[1]=='a' && s[2]=='x') ? POOL_METHOD_MAX : POOL_METHOD_AVG; }

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0)
        printf("note: no /dev/accel/accel0 (%d) — cube self-check + regcmd smoke only\n\n", fd);

    int fail = 0;
    if (argc == 13) {
        rocket_pool_desc d = { .method=parse_method(argv[1]),
            .c=atoi(argv[2]),.ih=atoi(argv[3]),.iw=atoi(argv[4]),
            .kh=atoi(argv[5]),.kw=atoi(argv[6]),.stride_y=atoi(argv[7]),.stride_x=atoi(argv[8]),
            .pad_top=atoi(argv[9]),.pad_left=atoi(argv[10]),.pad_bottom=atoi(argv[11]),.pad_right=atoi(argv[12]) };
        fail = run_shape(fd, &d);
    } else {
        rocket_pool_desc shapes[] = {
            /* MAX: HW-validated shapes (2x2 s2, C=8) */
            { .method=POOL_METHOD_MAX,.c=8, .ih=4, .iw=4, .kh=2,.kw=2,.stride_y=2,.stride_x=2 },
            { .method=POOL_METHOD_MAX,.c=8, .ih=6, .iw=6, .kh=3,.kw=3,.stride_y=1,.stride_x=1 },         /* k=3 valid */
            { .method=POOL_METHOD_MAX,.c=16,.ih=8, .iw=8, .kh=2,.kw=2,.stride_y=2,.stride_x=2 },         /* 2-plane */
            { .method=POOL_METHOD_MAX,.c=24,.ih=10,.iw=12,.kh=3,.kw=3,.stride_y=2,.stride_x=2 },         /* 3-plane, stride2 */
            { .method=POOL_METHOD_MAX,.c=8, .ih=7, .iw=7, .kh=7,.kw=7,.stride_y=1,.stride_x=1 },         /* global max -> 1x1 */
            { .method=POOL_METHOD_MAX,.c=8, .ih=6, .iw=6, .kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.pad_bottom=1,.pad_right=1 }, /* same-pad max */
            /* AVG: HW-validated shapes (2x2 s2 C=8; 3x3 s1 C=8 = k=3) */
            { .method=POOL_METHOD_AVG,.c=8, .ih=4, .iw=4, .kh=2,.kw=2,.stride_y=2,.stride_x=2 },
            { .method=POOL_METHOD_AVG,.c=8, .ih=6, .iw=6, .kh=3,.kw=3,.stride_y=1,.stride_x=1 },
            { .method=POOL_METHOD_AVG,.c=16,.ih=8, .iw=8, .kh=2,.kw=2,.stride_y=2,.stride_x=2 },         /* 2-plane avg */
            { .method=POOL_METHOD_AVG,.c=24,.ih=8, .iw=8, .kh=4,.kw=4,.stride_y=4,.stride_x=4 },         /* k=4, 3-plane */
            { .method=POOL_METHOD_AVG,.c=8, .ih=7, .iw=7, .kh=7,.kw=7,.stride_y=1,.stride_x=1 },         /* global avg -> 1x1 */
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
