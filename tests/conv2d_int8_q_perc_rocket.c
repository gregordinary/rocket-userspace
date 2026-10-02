// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv2d_int8_q_perc_rocket.c — RK3588 gate: rocket_conv2d_int8_q_perc, the direct int8 conv
 * with a per-output-channel weight scale on the int8-OUT writer, against two oracles.
 *
 *   1. A host model of the device arithmetic, bit-exact, from the plan the entry used
 *      (rocket_conv2d_int8_q_perc_plan_channels gives each channel's C, BS shift and job
 *      gain): TFLite's int32 accumulator bias + sum (x - in_zp) * w over the taps inside the
 *      image, then sat32(rne(acc * C >> s)) at the BS stage and the OUT_CVT model of
 *      tests/requant_model.h at the job's gain; a channel the plan gives to the host,
 *      TFLite's float requant. The accumulator and the arithmetic are this file's, not the
 *      library's. Every element must match.
 *   2. TFLite's per-channel arithmetic itself (the float requant on every channel). The
 *      device differs by one where a value sits near a rounding boundary; the gate reports
 *      the rate and fails on any difference past one, which is what a wrong plan (a gain, a
 *      channel order, a job's shift) would produce.
 *
 * The shapes carry what the entry has to handle: several scale-class jobs in one layer (a
 * spread past 16x), all-zero filters (the host's), a 1x1 over a plane that tiles by rows, an
 * OC split across tiles inside a job, a padded input-channel count under a CNA pad at a
 * nonzero input zero point (the materialized halo), an OC that ends part way through a
 * 32-kernel group, the RGB stem at stride 2, a 5x5 stride-2 conv, and a small plane. Output
 * zero points sit inside the range so a wrong shift for negative products shows (the BS
 * shift is per sign). The output is prefilled with a sentinel. Each shape runs through the
 * fd, ctx and three-fd pool entries and the ctx entry on fd -1 (the library's own host model
 * of the program). ctest also runs it under ROCKET_CONV_BATCH=1.
 *
 * What it cannot see: a defect in npu_out_cvt_pair() (tests/requant_edge_probe scores it),
 * and a plan that is valid but coarser than it could be (oracle 2 bounds it at one count).
 *
 * Exit 0 on pass, 1 on a mismatch, 2 without a device.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_conv_internal.h"
#include "test_fill.h"
#include "requant_model.h"

typedef struct {
    const char *name;
    int IC, IH, IW, OC, KH, KW, s, pad;
    double spread;      /* max/min live weight scale */
    int nzero;          /* all-zero filters */
    int zx, zo;
} pshape_t;

static int64_t sat32(int64_t v) { return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : v; }

static int tfl_chan(int64_t acc, float in_s, float ws, float out_s, int zo)
{
    float v = (in_s * ws) * (float)acc;
    long q = (long)lrintf(v / out_s) + zo;
    return q < -128 ? -128 : q > 127 ? 127 : (int)q;
}

static int run_shape(int fd, rocket_conv_ctx *ctx, rocket_conv_ctx *hctx, rocket_conv_pool *pool,
                     const pshape_t *g, uint64_t seed)
{
    rocket_conv2d_desc d = { .ic = g->IC, .ih = g->IH, .iw = g->IW, .oc = g->OC,
        .kh = g->KH, .kw = g->KW, .stride_y = g->s, .stride_x = g->s,
        .pad_top = g->pad, .pad_left = g->pad, .dil_y = 1, .dil_x = 1, .depthwise = 0 };
    const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d), OC = g->OC;
    const size_t K = (size_t)g->IC * g->KH * g->KW;
    const size_t ni = (size_t)g->IC * g->IH * g->IW, nw = (size_t)OC * K, no = (size_t)OC * OH * OW;
    int8_t *x = malloc(ni), *w = malloc(nw), *got = malloc(no), *ref = malloc(no), *tfl = malloc(no);
    int32_t *bias = malloc((size_t)OC * 4);
    float *ws = malloc((size_t)OC * 4), *cvt = malloc((size_t)OC * 4);
    int16_t *cmul = malloc((size_t)OC * 2);
    uint8_t *shift = malloc((size_t)OC), *host = malloc((size_t)OC);
    int64_t *acc = malloc(no * sizeof(int64_t));
    int bad = 1;

    tf_fill_i8(x, ni, seed + 1, -128, 127);
    tf_fill_i8(w, nw, seed + 2, -127, 127);
    tf_fill_i32(bias, OC, seed + 3, -20000, 20000);
    for (int z = 0; z < g->nzero; z++) {                 /* pruned channels, spread over OC */
        const int c = (int)(((long)z * 7 + 3) % OC);
        memset(w + (size_t)c * K, 0, K);
    }
    const float in_s = 0.02f, out_s = 0.05f;
    /* the base weight scale puts the largest channel's accumulator spread (about sqrt(K) * 64
     * * 73 raw) at a few tens of output counts */
    const float wbase = (float)(out_s * 40.0 / (in_s * sqrt((double)K) * 64.0 * 73.0));
    {
        int32_t r[4096];
        tf_fill_i32(r, OC, seed + 4, 0, 1 << 20);
        for (int c = 0; c < OC; c++)          /* log-uniform over the spread */
            ws[c] = (float)(wbase * pow(g->spread, -(double)r[c] / (1 << 20)));
    }
    if (rocket_conv2d_int8_q_perc_plan(&d) != ROCKET_OK) {
        printf("  %-22s plan refused\n", g->name);
        goto out;
    }
    if (rocket_conv2d_int8_q_perc_plan_channels(&d, w, bias, in_s, ws, out_s, g->zx, cmul, shift,
                                                cvt, host)) {
        printf("  %-22s plan_channels failed\n", g->name);
        goto out;
    }
    /* the accumulator, this file's own */
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t a = bias[oc];
                for (int ic = 0; ic < g->IC; ic++)
                    for (int kh = 0; kh < g->KH; kh++) {
                        const int ih = oh * g->s + kh - g->pad;
                        if (ih < 0 || ih >= g->IH) continue;
                        for (int kw = 0; kw < g->KW; kw++) {
                            const int iw = ow * g->s + kw - g->pad;
                            if (iw < 0 || iw >= g->IW) continue;
                            a += (int64_t)(x[((size_t)ic * g->IH + ih) * g->IW + iw] - g->zx) *
                                 w[(((size_t)oc * g->IC + ic) * g->KH + kh) * g->KW + kw];
                        }
                    }
                acc[((size_t)oc * OH + oh) * OW + ow] = a;
            }
    int nhost = 0, njobs = 0, pos = 0, neg = 0;
    for (int c = 0; c < OC; c++) {              /* jobs: the distinct job gains */
        if (host[c]) { nhost++; continue; }
        int seen = 0;
        for (int e = 0; e < c && !seen; e++) seen = !host[e] && cvt[e] == cvt[c];
        if (!seen) njobs++;
    }
    for (int oc = 0; oc < OC; oc++) {
        unsigned mul = 0, sh = 0;
        if (!host[oc]) requant_params(cvt[oc], &mul, &sh);
        for (size_t p = 0; p < (size_t)OH * OW; p++) {
            const size_t i = (size_t)oc * OH * OW + p;
            tfl[i] = (int8_t)tfl_chan(acc[i], in_s, ws[oc], out_s, g->zo);
            if (host[oc]) { ref[i] = tfl[i]; continue; }
            const int64_t pr = acc[i] * (int64_t)cmul[oc];
            if (pr >= 0) pos++; else neg++;
            const int64_t b = sat32(requant_round_shift(pr, shift[oc]));
            ref[i] = (int8_t)requant_apply_zp(b, mul, sh, g->zo);
        }
    }
    printf("  %-22s OC %d, %d host channels, %d jobs, products %d >= 0 / %d < 0\n",
           g->name, OC, nhost, njobs, pos, neg);

    bad = 0;
    for (int how = 0; how < 4; how++) {
        static const char *hn[] = { "fd", "ctx", "pool3", "host(fd-1)" };
        memset(got, 0x5A, no);
        int rc = how == 0 ? rocket_conv2d_int8_q_perc(fd, &d, x, w, bias, in_s, ws, out_s, g->zx,
                                                      g->zo, got)
               : how == 1 ? rocket_conv2d_int8_q_perc_ctx(ctx, &d, x, w, bias, in_s, ws, out_s,
                                                          g->zx, g->zo, got)
               : how == 2 ? (pool ? rocket_conv2d_int8_q_perc_mt(pool, &d, x, w, bias, in_s, ws,
                                                                 out_s, g->zx, g->zo, got) : -99)
               : rocket_conv2d_int8_q_perc_ctx(hctx, &d, x, w, bias, in_s, ws, out_s, g->zx,
                                               g->zo, got);
        if (rc) { printf("    %-10s FAIL rc=%d\n", hn[how], rc); bad = 1; continue; }
        long wrong = 0, off1 = 0, offm = 0;
        size_t first = no;
        for (size_t i = 0; i < no; i++) {
            if (got[i] != ref[i]) { if (!wrong) first = i; wrong++; }
            const long dt = labs((long)got[i] - tfl[i]);
            if (dt == 1) off1++; else if (dt > 1) offm++;
        }
        if (wrong || offm) {
            printf("    %-10s FAIL %ld/%zu differ from the device model", hn[how], wrong, no);
            if (wrong) printf(" (first at %zu: got %d want %d, channel %zu)", first, got[first],
                              ref[first], first / ((size_t)OH * OW));
            printf("; vs TFLite %ld off by one, %ld by more\n", off1, offm);
            bad = 1;
        } else {
            printf("    %-10s exact %zu/%zu; vs TFLite %ld off by one (%.3f%%), none by more\n",
                   hn[how], no, no, off1, 100.0 * off1 / no);
        }
    }
out:
    free(x); free(w); free(got); free(ref); free(tfl); free(bias); free(ws); free(cvt);
    free(cmul); free(shift); free(host); free(acc);
    return bad;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no /dev/accel/accel0 (%d)\n", fd); return 2; }

    static const pshape_t shapes[] = {
        /* name                    IC   IH   IW   OC KH KW s pad  spread nzero   zx   zo */
        { "1x1 80x80 3 jobs",      24,  80,  80, 144, 1, 1, 1, 0, 1000.0,  2,  -9,  11 },
        { "1x1 20x20 OC672",      112,  20,  20, 672, 1, 1, 1, 0,    9.0,  7,   2, -20 },
        { "1x1 160x160 IC32",      32, 160, 160,  16, 1, 1, 1, 0,    3.6,  0, -128,  0 },
        { "1x1 5x5 2 jobs",        64,   5,   5,  64, 1, 1, 1, 0,   54.0,  0,   3,  40 },
        { "k3p1 IC40 OC48 halo",   40,  20,  20,  48, 3, 3, 1, 1,   30.0,  1,   7,  -9 },
        { "stem IC3 s2",            3,  64,  64,  32, 3, 3, 2, 1,   70.0,  4,  -1,  17 },
        { "k5 s2 IC16",            16,  40,  40, 128, 5, 5, 2, 2,  200.0,  0,   1,  -3 },
        { "1x1 10x10 OC1152",     192,  10,  10, 1152, 1, 1, 1, 0, 1000.0, 33,  -1,  60 },
    };
    const int NS = (int)(sizeof shapes / sizeof shapes[0]);
    rocket_conv_ctx *ctx = rocket_conv_ctx_create(fd);
    rocket_conv_ctx *hctx = rocket_conv_ctx_create(-1);
    rocket_conv_pool *pool = rocket_conv_pool_create(3);
    int bad = 0;

    printf("conv2d_int8_q_perc_rocket: per-channel direct int8 conv, int8-out, vs the device "
           "model and TFLite\n");
    for (int si = 0; si < NS; si++)
        bad |= run_shape(fd, ctx, hctx, pool, &shapes[si], 0x2a0000ull + 131 * si);
    rocket_conv_pool_free(pool);
    rocket_conv_ctx_free(hctx);
    rocket_conv_ctx_free(ctx);
    rocket_close(fd);
    printf("%s\n", bad ? "RESULT: FAIL" : "RESULT: every element exact, within one of TFLite");
    return bad ? 1 : 0;
}
