// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv2d_int8_q_rocket.c — RK3588 gate: rocket_conv2d_int8_q, the direct int8 conv with the
 * int8-OUT writer (on-chip requant), bit-exact against a host model of what it computes.
 *
 * The model is independent of the library: TFLite's accumulator bias + sum (x - in_zp)(w -
 * w_zp) over the taps inside the image (a padded tap contributes nothing), then the OUT_CVT
 * (tests/requant_model.h: the vendor's 15-bit multiplier and shift from in*w/out, ties to
 * even, then the output zero point, saturated to int8). Every element must equal it.
 *
 * The shapes cover the entry's paths: the one-job fast path under a CNA pad, the tile path
 * with a materialized halo (a padded input-channel count forces it), an OC that is not a
 * multiple of 32 (padded to the group), a plane that tiles by rows, an OC split across
 * several jobs, the RGB stem (IC 3, stride 2), a 1x1 head at a tiny plane, a 5x5 stride-2
 * conv, symmetric weights at pad 0, and 1x1s at the planes and channel counts of the
 * pointwise layers a frontend moves off the int8 matmul (160x160 to 20x20, OC 8 to 768);
 * weight zero points of both signs and output zero points
 * at and off the -128 rail. Each shape runs through the single-fd entry, the ctx entry and
 * the three-fd pool entry, and through the ctx entry on fd -1 (the library's own host model
 * of the program, through the same tiler). ctest also runs it under ROCKET_CONV_BATCH=1.
 *
 * What it cannot see: a defect the host model shares (it is the probe's model, which
 * tests/conv_i8out_probe scores against the device element by element), per-channel weight
 * scales (the entry is per-tensor), and TFLite's own double rounding (the requant here is
 * the OUT_CVT's, which can differ from TFLite's by one at a rounding boundary).
 *
 * Exit 0 when every element of every run is exact, 1 otherwise, 2 without a device.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "test_fill.h"
#include "requant_model.h"

typedef struct {
    const char *name;
    int IC, IH, IW, OC, KH, KW, s, pad;
    int zx, zw, zo;
} gshape_t;

static void host_model(const gshape_t *g, int OH, int OW, const int8_t *x, const int8_t *w,
                       const int32_t *bias, float in_s, float w_s, float out_s, int8_t *ref)
{
    unsigned mul, shift;
    requant_params((in_s * w_s) / out_s, &mul, &shift);
    for (int oc = 0; oc < g->OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t acc = bias[oc];
                for (int ic = 0; ic < g->IC; ic++)
                    for (int kh = 0; kh < g->KH; kh++) {
                        const int ih = oh * g->s + kh - g->pad;
                        if (ih < 0 || ih >= g->IH) continue;
                        for (int kw = 0; kw < g->KW; kw++) {
                            const int iw = ow * g->s + kw - g->pad;
                            if (iw < 0 || iw >= g->IW) continue;
                            acc += (int64_t)(x[((size_t)ic * g->IH + ih) * g->IW + iw] - g->zx) *
                                   (w[(((size_t)oc * g->IC + ic) * g->KH + kh) * g->KW + kw] - g->zw);
                        }
                    }
                ref[((size_t)oc * OH + oh) * OW + ow] = (int8_t)requant_apply_zp(acc, mul, shift, g->zo);
            }
}

static int check(const char *shape, const char *how, int rc, const int8_t *got, const int8_t *ref,
                 size_t n)
{
    if (rc) {
        printf("  %-18s %-10s FAIL rc=%d\n", shape, how, rc);
        return 1;
    }
    long bad = 0, maxd = 0;
    size_t first = n;
    for (size_t i = 0; i < n; i++) {
        long d = labs((long)got[i] - ref[i]);
        if (d) { if (!bad) first = i; bad++; if (d > maxd) maxd = d; }
    }
    if (bad) {
        printf("  %-18s %-10s FAIL %ld/%zu wrong, max %ld, first at %zu (got %d want %d)\n",
               shape, how, bad, n, maxd, first, got[first], ref[first]);
        return 1;
    }
    printf("  %-18s %-10s exact %zu/%zu\n", shape, how, n, n);
    return 0;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no /dev/accel/accel0 (%d)\n", fd); return 2; }

    static const gshape_t shapes[] = {
        /* name                 IC   IH   IW   OC KH KW s pad   zx   zw    zo */
        { "fast k3p1",          64,  10,  12,  64, 3, 3, 1, 1,   23, -37,    5 },
        { "IC40 OC48 halo",     40,  20,  20,  48, 3, 3, 1, 1,    7,  12,   -9 },
        { "tall plane",         32, 120, 100,  64, 3, 3, 1, 1,   -5,   9, -128 },
        { "OC split IC256",    256,  20,  20, 288, 3, 3, 1, 1,    4,  -6,    3 },
        { "stem IC3 s2",         3,  64,  64,  32, 3, 3, 2, 1,    0,   6, -128 },
        { "head 1x1 OC546",    512,   5,   5, 546, 1, 1, 1, 0,    0,   9,   92 },
        { "k5 s2 IC16",         16,  40,  40, 128, 5, 5, 2, 2,    1,  12, -128 },
        { "sym pad0",           96,   9,  11,  32, 3, 3, 1, 0, -100,   0,    0 },
        /* the 1x1s the delegate moves off the int8 matmul (MobileDet's and SSD's planes) */
        { "1x1 160x160 OC8",    32, 160, 160,   8, 1, 1, 1, 0,   -5,   9,    3 },
        { "1x1 150x150 OC96",   16, 150, 150,  96, 1, 1, 1, 0,   12, -30,   -7 },
        { "1x1 80x80 IC128",   128,  80,  80,  16, 1, 1, 1, 0, -128,  21, -100 },
        { "1x1 40x40 OC320",    40,  40,  40, 320, 1, 1, 1, 0,    0,  -5,   30 },
        { "1x1 20x20 OC768",    96,  20,  20, 768, 1, 1, 1, 0,    7,   3,   -2 },
    };
    const int NS = (int)(sizeof shapes / sizeof shapes[0]);
    rocket_conv_ctx *ctx = rocket_conv_ctx_create(fd);
    rocket_conv_ctx *hctx = rocket_conv_ctx_create(-1);
    rocket_conv_pool *pool = rocket_conv_pool_create(3);
    int bad = 0;

    printf("conv2d_int8_q_rocket: direct int8 conv, int8-out writer, vs the host model\n");
    for (int si = 0; si < NS; si++) {
        const gshape_t *g = &shapes[si];
        rocket_conv2d_desc d = { .ic = g->IC, .ih = g->IH, .iw = g->IW, .oc = g->OC,
            .kh = g->KH, .kw = g->KW, .stride_y = g->s, .stride_x = g->s,
            .pad_top = g->pad, .pad_left = g->pad, .dil_y = 1, .dil_x = 1, .depthwise = 0 };
        const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d);
        const size_t ni = (size_t)g->IC * g->IH * g->IW, nw = (size_t)g->OC * g->IC * g->KH * g->KW;
        const size_t no = (size_t)g->OC * OH * OW;
        int8_t *x = malloc(ni), *w = malloc(nw), *ref = malloc(no), *got = malloc(no);
        int32_t *bias = malloc((size_t)g->OC * 4);
        const uint64_t seed = 0x19a0000ull + 97 * si;
        tf_fill_i8(x, ni, seed + 1, g->zx - 100 < -128 ? -128 : g->zx - 100,
                   g->zx + 100 > 127 ? 127 : g->zx + 100);
        tf_fill_i8(w, nw, seed + 2, g->zw - 100 < -128 ? -128 : g->zw - 100,
                   g->zw + 100 > 127 ? 127 : g->zw + 100);
        tf_fill_i32(bias, g->OC, seed + 3, -20000, 20000);
        /* a requant that spreads the accumulator (std about sqrt(K) * 58^2) over the range */
        const double K = (double)g->IC * g->KH * g->KW;
        const float cs = (float)(60.0 / (sqrt(K) * 3364.0));
        const float in_s = 0.05f, out_s = 0.1f, w_s = cs * out_s / in_s;
        host_model(g, OH, OW, x, w, bias, in_s, w_s, out_s, ref);

        if (rocket_conv2d_int8_q_plan(&d) != ROCKET_OK) {
            printf("  %-18s plan refused\n", g->name);
            bad = 1;
            goto next;
        }
        memset(got, 0xAA, no);
        bad |= check(g->name, "fd", rocket_conv2d_int8_q(fd, &d, x, w, bias, in_s, w_s, out_s,
                     g->zx, g->zw, g->zo, got), got, ref, no);
        memset(got, 0xAA, no);
        bad |= check(g->name, "ctx", rocket_conv2d_int8_q_ctx(ctx, &d, x, w, bias, in_s, w_s,
                     out_s, g->zx, g->zw, g->zo, got), got, ref, no);
        memset(got, 0xAA, no);
        bad |= check(g->name, "pool3", pool ? rocket_conv2d_int8_q_mt(pool, &d, x, w, bias, in_s,
                     w_s, out_s, g->zx, g->zw, g->zo, got) : -99, got, ref, no);
        memset(got, 0xAA, no);
        bad |= check(g->name, "host(fd-1)", rocket_conv2d_int8_q_ctx(hctx, &d, x, w, bias, in_s,
                     w_s, out_s, g->zx, g->zw, g->zo, got), got, ref, no);
next:
        free(x); free(w); free(ref); free(got); free(bias);
    }
    rocket_conv_pool_free(pool);
    rocket_conv_ctx_free(hctx);
    rocket_conv_ctx_free(ctx);
    rocket_close(fd);
    printf("%s\n", bad ? "RESULT: FAIL" : "RESULT: every element exact");
    return bad ? 1 : 0;
}
