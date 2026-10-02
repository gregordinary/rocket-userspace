// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_dw_int8_perc_runtime.c — gate for the RK3588 PER-CHANNEL int8 depthwise entry
 * (rocket_conv2d_dw_int8_perc), against two oracles.
 *
 *   1. A host model of the device arithmetic, bit-exact, from the plan the entry itself
 *      used (rocket_conv2d_dw_int8_perc_plan_channels): per channel, TFLite's int32
 *      accumulator, then sat32(rne(acc*C >> s)) at the BS stage and the OUT_CVT model of
 *      tests/requant_model.h at the job's base gain; a channel the plan gives to the host,
 *      TFLite's float requant (the library's dw_int8_ref). Every element must match.
 *   2. TFLite's per-channel arithmetic itself (the same float requant on every channel).
 *      The device differs by one where a value sits near a rounding boundary; the gate
 *      reports the rate and fails on any difference past one.
 *
 * The shapes carry what the planner has to handle: a wide per-channel scale spread, all-zero
 * filters (a pruned channel), channels scaled far below the rest (the host's), a layer that
 * runs as several jobs (so the scale sort matters), a plane past one CBUF pass (bands), a
 * partial 64-channel group, and pads past CNA_PAD_CON0's 4-bit field on planes that fit one
 * pass (pads 16 and 17 read 9694 and 14132 elements wrong on the library before 96d8e32). The output is prefilled with a sentinel, so a channel no path
 * writes cannot pass. Every output zero point sits inside the range: at -128 every negative
 * value saturates to the same byte, and a program that shifts negative products wrongly
 * (the BS shift is per sign) passes there.
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
    int C, IH, IW, K, S, P;
    double spread;      /* max/min live weight scale */
    int nzero;          /* all-zero filters */
    int ntiny;          /* live channels at 1e-5 of the largest scale */
    int in_zp, out_zp;
} shape_t;

static int64_t sat32(int64_t v) { return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : v; }

static int tfl_chan(int64_t acc, float in_s, float ws, float out_s, int zo)
{
    float v = (in_s * ws) * (float)acc;
    long q = (long)lrintf(v / out_s) + zo;
    return q < -128 ? -128 : q > 127 ? 127 : (int)q;
}

static int run_shape(int fd, rocket_conv_ctx *ctx, const shape_t *sh, uint64_t seed)
{
    const int C = sh->C, IH = sh->IH, IW = sh->IW, K = sh->K;
    rocket_conv2d_desc d = { .ic = C, .ih = IH, .iw = IW, .oc = C, .kh = K, .kw = K,
                             .stride_y = sh->S, .stride_x = sh->S, .pad_top = sh->P,
                             .pad_left = sh->P, .dil_y = 1, .dil_x = 1, .depthwise = 1 };
    const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d);
    const size_t ni = (size_t)C * IH * IW, nw = (size_t)C * K * K, no = (size_t)C * OH * OW;
    int8_t *x = malloc(ni), *w = malloc(nw), *got = malloc(no);
    int32_t *bias = malloc((size_t)C * 4);
    float *ws = malloc((size_t)C * 4), *cvt = malloc((size_t)C * 4);
    int16_t *cmul = malloc((size_t)C * 2);
    uint8_t *shift = malloc((size_t)C), *host = malloc((size_t)C);
    int bad = 1;

    tf_fill_i8(x, ni, seed + 1, -128, 127);
    tf_fill_i8(w, nw, seed + 2, -127, 127);
    tf_fill_i32(bias, C, seed + 3, -20000, 20000);
    const float in_s = 0.02f, out_s = 0.05f, wbase = 0.004f;
    {
        int32_t r[4096];
        tf_fill_i32(r, C, seed + 4, 0, 1 << 20);
        for (int c = 0; c < C; c++)          /* log-uniform over the spread */
            ws[c] = (float)(wbase * pow(sh->spread, -(double)r[c] / (1 << 20)));
    }
    for (int k = 0; k < sh->nzero; k++) {    /* pruned: an all-zero filter, a floor scale */
        const int c = (k * 37 + 5) % C;
        memset(w + (size_t)c * K * K, 0, (size_t)K * K);
        ws[c] = 1e-9f;
    }
    for (int k = 0; k < sh->ntiny; k++) {    /* live but far below the rest: the host's */
        const int c = (k * 53 + 11) % C;
        ws[c] = wbase * 1e-5f;
    }

    if (rocket_conv2d_dw_int8_perc_plan(&d) != ROCKET_OK) {
        printf("  %-12s plan refused\n", sh->name);
        goto done;
    }
    if (rocket_conv2d_dw_int8_perc_plan_channels(&d, w, bias, in_s, ws, out_s, sh->in_zp, cmul,
                                                 shift, cvt, host)) {
        printf("  %-12s plan_channels failed\n", sh->name);
        goto done;
    }
    memset(got, TF_SENTINEL_BYTE, no);
    int rc = ctx ? rocket_conv2d_dw_int8_perc_ctx(ctx, &d, x, w, bias, in_s, ws, out_s,
                                                  sh->in_zp, sh->out_zp, got)
                 : rocket_conv2d_dw_int8_perc(fd, &d, x, w, bias, in_s, ws, out_s,
                                              sh->in_zp, sh->out_zp, got);
    if (rc) { printf("  %-12s entry rc=%d\n", sh->name, rc); goto done; }

    long mexact = 0, texact = 0, toff1 = 0, tfar = 0, nhost = 0, sent = 0, mbad_host = 0;
    int first_bad = -1, nshow = 0;
    for (int c = 0; c < C; c++) nhost += host[c];
    for (int c = 0; c < C; c++) {
        unsigned mul = 0, sft = 0;
        if (!host[c]) requant_params(cvt[c], &mul, &sft);
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t acc = bias[c];
                for (int kh = 0; kh < K; kh++)
                    for (int kw = 0; kw < K; kw++) {
                        int ih = oh * sh->S + kh - sh->P, iw = ow * sh->S + kw - sh->P;
                        int q = (ih >= 0 && ih < IH && iw >= 0 && iw < IW)
                              ? x[((size_t)c * IH + ih) * IW + iw] : sh->in_zp;
                        acc += (int64_t)(q - sh->in_zp) * w[((size_t)c * K + kh) * K + kw];
                    }
                const size_t o = ((size_t)c * OH + oh) * OW + ow;
                const int t = tfl_chan(acc, in_s, ws[c], out_s, sh->out_zp);
                int m;
                if (host[c]) {
                    m = t;
                } else {
                    int64_t p = acc * (int64_t)cmul[c];
                    int64_t v = sat32(requant_round_shift(p, shift[c]));
                    m = requant_apply_zp(v, mul, sft, sh->out_zp);
                }
                if ((uint8_t)got[o] == TF_SENTINEL_BYTE && m != (int8_t)TF_SENTINEL_BYTE) sent++;
                if (got[o] == m) mexact++;
                else {
                    if (first_bad < 0) first_bad = (int)o;
                    if (host[c]) mbad_host++;
                    if (nshow < 8) {
                        nshow++;
                        printf("    mismatch c %d (%d,%d): acc %lld  C %d s %u  model %d  TFLite %d  "
                               "got %d\n", c, oh, ow, (long long)acc, cmul[c], shift[c], m, t, got[o]);
                    }
                }
                const int dt = abs(got[o] - t);
                if (!dt) texact++; else if (dt == 1) toff1++; else tfar++;
            }
    }
    printf("  %-12s C %4d %3dx%-3d k%d s%d  %zu el  host ch %3ld | model %s %ld/%zu "
           "(host-ch bad %ld, sentinel %ld) | TFLite exact %ld off1 %ld (%.3f%%) far %ld\n",
           sh->name, C, IH, IW, K, sh->S, no, nhost, mexact == (long)no ? "EXACT" : "WRONG",
           mexact, no, mbad_host, sent, texact, toff1, 100.0 * toff1 / (double)no, tfar);
    if (first_bad >= 0) {
        const int c = first_bad / (OH * OW);
        printf("    first mismatch at %d (channel %d, host %d, C %d, s %u, cvt %g): got %d\n",
               first_bad, c, host[c], cmul[c], shift[c], (double)cvt[c], got[first_bad]);
    }
    bad = !(mexact == (long)no && tfar == 0);
done:
    free(x); free(w); free(got); free(bias); free(ws); free(cvt); free(cmul); free(shift);
    free(host);
    return bad;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { printf("SKIP: no /dev/accel/accel0\n"); return 2; }
    rocket_conv_ctx *ctx = rocket_conv_ctx_create(fd);
    const shape_t shapes[] = {
        /* name        C    IH   IW  K S P  spread  zero tiny  in_zp out_zp */
        { "one job",    64,  12,  12, 3, 1, 1,   10.0,  0,  0,  -128,     0 },
        { "deep",     1152,  10,  10, 5, 1, 2, 3.0e4,  33,  6,  -128,   -20 },
        { "wide 480",  480,  20,  20, 5, 1, 2,   65.0,  0,  0,  -128,     7 },
        { "banded",     96, 160, 160, 3, 2, 1,   45.0,  1,  0,  -128,     5 },
        { "partial",    80,  15,  17, 3, 2, 1,   20.0,  2,  1,    13,   -40 },
        { "three jobs", 192, 40,  40, 5, 1, 2,   30.0,  1,  0,   -60,    12 },
        /* pads past CNA_PAD_CON0's 4-bit field: each plane fits one pass, so only the pad
         * routes it to the bands. Pad 15 is the control, the field's largest value. */
        { "pad 15",     64,   8,   8, 3, 1, 15,  10.0,  0,  0,   -60,    12 },
        { "pad 16",     64,   8,   8, 3, 1, 16,  10.0,  0,  0,   -60,    12 },
        { "pad 17",     96,   6,  10, 3, 1, 17,  20.0,  1,  0,    13,   -40 },
    };
    int fails = 0;
    printf("conv_dw_int8_perc_runtime: per-channel int8 depthwise vs its device model and TFLite\n");
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        fails += run_shape(fd, ctx, &shapes[i], 7000 + 31 * i);
        if (i == 0) fails += run_shape(fd, NULL, &shapes[i], 7000);   /* the fd entry too */
    }
    rocket_conv_ctx_free(ctx);
    rocket_close(fd);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
