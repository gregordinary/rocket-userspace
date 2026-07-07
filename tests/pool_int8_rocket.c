// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * pool_int8_rocket.c — standalone test for on-NPU int8 / uint8 MaxPool / AveragePool.
 *
 * NPU FACT (RE'd here): the RK3588 PPU has no native int8 pooling precision — a job with
 * PROC_PRECISION=int8 over a packed int8 cube reads the bytes as fp16 (garbage); the
 * allbilly reference emits PROC_PRECISION=fp16 for every pool. So int8/uint8 pooling
 * ROUTES THROUGH the fp16 PPU path (rocket_pool_int8 / rocket_pool_uint8): int8/uint8 are
 * exact in fp16, so MAX is bit-exact and AVG matches the fp16(65536/k) recip.
 *
 * Two layers:
 *   1. INDEPENDENT INTEGER GOLDEN: plain-arithmetic max / round(exact-average), computed
 *      here with no library code, as the oracle.
 *   2. DRIVER vs GOLDEN — HW end-to-end (fd>=0) or the fp16 CPU oracle (fd<0):
 *        int8/uint8 MAX  : bit-exact (tol 0)
 *        int8     AVG    : within ±1 ULP (fp16 recip + round)
 *
 * Usage: pool_int8_rocket
 *        pool_int8_rocket METHOD DTYPE C IH IW KH KW SY SX PT PL PB PR
 *          METHOD = max|avg, DTYPE = i8|u8
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_pool.h"

static const char *mname(int m) { return m == POOL_METHOD_MAX ? "max" : "avg"; }

/* Independent integer golden. For uint8 (is_uint8) values are read unsigned and pooled in
 * the unsigned domain; for int8 they are signed. MAX uses the dtype minimum for out-of-
 * range; AVG = round-to-nearest of the count-include-pad average (matches the fp16 path). */
static void golden(const rocket_pool_desc *d, const int8_t *in, int8_t *out, int is_uint8)
{
    const int C = d->c, IH = d->ih, IW = d->iw;
    const int OH = rocket_pool_oh(d), OW = rocket_pool_ow(d);
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int acc = (d->method == POOL_METHOD_MAX) ? (is_uint8 ? 0 : -128) : 0;
                for (int kh = 0; kh < d->kh; kh++) {
                    int ih = oh*d->stride_y + kh - d->pad_top;
                    for (int kw = 0; kw < d->kw; kw++) {
                        int iw = ow*d->stride_x + kw - d->pad_left;
                        int oor = (ih < 0 || ih >= IH || iw < 0 || iw >= IW);
                        /* MAX: padding does NOT contribute (PPU pad-fill -inf). AVG:
                         * count-include-pad, OOR contributes 0 to the sum. */
                        if (oor) { if (d->method == POOL_METHOD_AVG) acc += 0; continue; }
                        int v = is_uint8 ? (int)(uint8_t)in[((size_t)c*IH+ih)*IW+iw]
                                         : (int)in[((size_t)c*IH+ih)*IW+iw];
                        if (d->method == POOL_METHOD_MAX) { if (v > acc) acc = v; }
                        else acc += v;
                    }
                }
                int o = acc;
                if (d->method == POOL_METHOD_AVG) {
                    float avg = (float)acc / (float)(d->kh*d->kw);
                    o = (int)lrintf(avg);
                }
                int lo = is_uint8 ? 0 : -128, hi = is_uint8 ? 255 : 127;
                if (o > hi) o = hi; else if (o < lo) o = lo;
                ((uint8_t*)out)[((size_t)c*OH+oh)*OW+ow] = (uint8_t)o;
            }
}

static int run_shape(int fd, const rocket_pool_desc *d, int is_uint8)
{
    int OH = rocket_pool_oh(d), OW = rocket_pool_ow(d);
    printf("%s %s C=%d %dx%d  k=%dx%d s=%dx%d p=%d,%d,%d,%d -> %dx%d\n",
           mname(d->method), is_uint8 ? "u8" : "i8", d->c, d->ih, d->iw, d->kh, d->kw,
           d->stride_y, d->stride_x, d->pad_top, d->pad_left, d->pad_bottom, d->pad_right, OH, OW);

    int plan = rocket_pool_fp16_plan(d);
    if (plan) { printf("  plan: unsupported (%d) — skipping\n", plan); return 0; }

    size_t in_n = (size_t)d->c*d->ih*d->iw, out_n = (size_t)d->c*OH*OW;
    int8_t *in = malloc(in_n), *out = malloc(out_n), *gold = malloc(out_n);
    if (!in || !out || !gold) { fprintf(stderr, "oom\n"); return -1; }
    /* spread across the range incl. the sign boundary (exercises the recenter + clamp). */
    for (size_t i = 0; i < in_n; i++) {
        int v = (int)((i*37 + 11) % 251);                 /* 0..250 */
        in[i] = is_uint8 ? (int8_t)(uint8_t)v : (int8_t)(v - 125);
    }

    int fail = 0;
    {   /* regcmd smoke (fp16 path is what int8 routes through) */
        uint64_t regs[64] = {0};
        pool_params_t p = { .c=d->c,.ih=d->ih,.iw=d->iw,.oh=OH,.ow=OW,.kh=d->kh,.kw=d->kw,
            .stride_y=d->stride_y,.stride_x=d->stride_x,.pad_top=d->pad_top,.pad_left=d->pad_left,
            .pad_bottom=d->pad_bottom,.pad_right=d->pad_right,.method=(uint8_t)d->method,
            .recip_w=ppu_recip_kernel_fp16(d->kw),.recip_h=ppu_recip_kernel_fp16(d->kh),
            .input_dma=0x1000,.output_dma=0x3000,.tasks=regs };
        int g = gen_pool_fp16(&p);
        if (!(g==0 && p.task_count>0)) { printf("  gen_pool_fp16: ret=%d -> FAIL\n", g); fail = 1; }
    }

    {
        const char *tag = (fd >= 0) ? "HW end-to-end" : "CPU fallback";
        memset(out, 0, out_n);
        int r = is_uint8 ? rocket_pool_uint8(fd, d, (uint8_t*)in, (uint8_t*)out)
                         : rocket_pool_int8(fd, d, in, out);
        if (r) { printf("  %s: rocket_pool_%s = %d (FAIL)\n", tag, is_uint8?"uint8":"int8", r); fail = 1; }
        else {
            golden(d, in, gold, is_uint8);
            int tol = (d->method == POOL_METHOD_MAX) ? 0 : 1;
            int maxd = 0, bad = 0;
            for (size_t i = 0; i < out_n; i++) {
                int a = is_uint8 ? (uint8_t)out[i]  : out[i];
                int b = is_uint8 ? (uint8_t)gold[i] : (int8_t)gold[i];
                int ad = abs(a - b);
                if (ad > maxd) maxd = ad;
                if (ad > tol && bad < 6) { printf("    [%zu] gold=%d got=%d d=%d\n", i, b, a, ad); bad++; }
            }
            printf("  %s: maxd=%d (tol=%d) -> %s\n", tag, maxd, tol, maxd <= tol ? "PASS" : "FAIL");
            if (maxd > tol) fail = 1;
        }
    }
    free(in); free(out); free(gold);
    return fail;
}

static int parse_method(const char *s)
{ return (s[0]=='m') ? POOL_METHOD_MAX : POOL_METHOD_AVG; }

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0)
        printf("note: no /dev/accel/accel0 (%d) — CPU oracle (fd<0) only\n\n", fd);

    int fail = 0;
    if (argc == 14) {
        int is_u8 = (argv[2][0]=='u');
        rocket_pool_desc d = { .method=parse_method(argv[1]),
            .c=atoi(argv[3]),.ih=atoi(argv[4]),.iw=atoi(argv[5]),
            .kh=atoi(argv[6]),.kw=atoi(argv[7]),.stride_y=atoi(argv[8]),.stride_x=atoi(argv[9]),
            .pad_top=atoi(argv[10]),.pad_left=atoi(argv[11]),.pad_bottom=atoi(argv[12]),.pad_right=atoi(argv[13]) };
        fail = run_shape(fd, &d, is_u8);
    } else {
        struct { rocket_pool_desc d; int u8; } shapes[] = {
            /* int8 MAX: single & multi C-plane, stride, global, same-pad, C not %16 */
            {{ .method=POOL_METHOD_MAX,.c=16,.ih=4, .iw=4, .kh=2,.kw=2,.stride_y=2,.stride_x=2 }, 0},
            {{ .method=POOL_METHOD_MAX,.c=16,.ih=6, .iw=6, .kh=3,.kw=3,.stride_y=1,.stride_x=1 }, 0},
            {{ .method=POOL_METHOD_MAX,.c=32,.ih=8, .iw=8, .kh=2,.kw=2,.stride_y=2,.stride_x=2 }, 0},
            {{ .method=POOL_METHOD_MAX,.c=48,.ih=10,.iw=12,.kh=3,.kw=3,.stride_y=2,.stride_x=2 }, 0},
            {{ .method=POOL_METHOD_MAX,.c=16,.ih=7, .iw=7, .kh=7,.kw=7,.stride_y=1,.stride_x=1 }, 0},
            {{ .method=POOL_METHOD_MAX,.c=16,.ih=6, .iw=6, .kh=3,.kw=3,.stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.pad_bottom=1,.pad_right=1 }, 0},
            {{ .method=POOL_METHOD_MAX,.c=24,.ih=5, .iw=5, .kh=2,.kw=2,.stride_y=2,.stride_x=2 }, 0},
            /* uint8 MAX: the detection dtype, values straddling 128 */
            {{ .method=POOL_METHOD_MAX,.c=16,.ih=8, .iw=8, .kh=2,.kw=2,.stride_y=2,.stride_x=2 }, 1},
            {{ .method=POOL_METHOD_MAX,.c=32,.ih=6, .iw=6, .kh=3,.kw=3,.stride_y=1,.stride_x=1 }, 1},
            {{ .method=POOL_METHOD_MAX,.c=24,.ih=7, .iw=7, .kh=7,.kw=7,.stride_y=1,.stride_x=1 }, 1},
            /* int8 AVG: pad-free (exact count) — the fp16 recip path, rounded to int8 */
            {{ .method=POOL_METHOD_AVG,.c=16,.ih=4, .iw=4, .kh=2,.kw=2,.stride_y=2,.stride_x=2 }, 0},
            {{ .method=POOL_METHOD_AVG,.c=16,.ih=8, .iw=8, .kh=4,.kw=4,.stride_y=4,.stride_x=4 }, 0},
            {{ .method=POOL_METHOD_AVG,.c=32,.ih=7, .iw=7, .kh=7,.kw=7,.stride_y=1,.stride_x=1 }, 0},
        };
        for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++) {
            fail |= run_shape(fd, &shapes[i].d, shapes[i].u8);
            printf("\n");
        }
    }
    if (fd >= 0) rocket_close(fd);
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
