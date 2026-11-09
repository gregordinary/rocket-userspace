// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * requant_zp_order_probe.c -- on the RK3588's int8-OUTPUT writer, is the output zero point
 * added before the OUT_CVT's rounding or after it?
 *
 * Measured on the fp32-output convert (notes: out-cvt-converter.md, cvt_type 0): the offset
 * is added at the output scale BEFORE the rounding, out = rne(acc*MUL>>SHIFT + OFFSET). The
 * int8 entries' documented model, and tests/requant_model.h, round first and add the zero
 * point after: out = sat8(rne(acc*MUL>>SHIFT) + out_zp). The two differ only at a tie with
 * an odd zero point, which no gate reaches: the derivation makes MUL odd for every round
 * scale, so a tie never reaches the rounder.
 *
 * The probe builds ties on purpose. It drives rocket_conv2d_int8_q (Mesa's direct int8-out
 * program, a validated entry) with every input at its zero point, so each output channel's
 * accumulator is exactly its bias. The scale is one whose top 14 mantissa bits are all ones,
 * where the derivation's +1 carries out and MUL is exactly 2^14, so the requant is a plain
 * acc >> e. Half the channels sit on ties (odd multiples of 2^(e-1)), half off them. Each
 * output zero point runs as its own call; each element is scored against:
 *   pre    sat8(rne((acc + zp*2^e) / 2^e))     the zero point added before the rounding
 *   post   sat8(rne(acc / 2^e) + zp)           the zero point added after it
 * They agree on every non-tie and at every even zero point, so those elements check the
 * accumulator; the ties at odd zero points decide the order.
 *
 * ROCKET_OUT_CVT_ROUND=1 (the library's switch for OUT_CVT_SHIFT bit 30) changes the tie
 * rule to half away from zero; the models then use that rule.
 * Exit 0 when one model explains every element of every call, 1 otherwise, 2 without a device.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npu_requant.h"
#include "rocket_conv.h"
#include "rocket_npu.h"

#define IC 32
#define HW 4
#define OC 64

static int g_away;   /* tie rule: 0 half to even, 1 half away from zero */

static int64_t rnd_shift(int64_t n, int e)
{
    if (e == 0) return n;
    const int64_t d = (int64_t)1 << e;
    int64_t q = n >= 0 ? n / d : -((-n + d - 1) / d);   /* floor */
    int64_t r = n - q * d;                             /* 0 <= r < d */
    if (2 * r > d) return q + 1;
    if (2 * r < d) return q;
    if (g_away) return n >= 0 ? q + 1 : q;             /* tie: away from zero */
    return q + (q & 1);                                /* tie: to even */
}
static int sat8(int64_t v) { return (int)(v < -128 ? -128 : v > 127 ? 127 : v); }

typedef struct { const char *name; float scale; int e; } tscale;
/* significand 1.99993896484375: MUL carries out to 0x4000 at one less shift */
static const tscale SC[] = {
    { "acc>>1", 0.4999847412109375f, 1 },
    { "acc>>3", 0.12499618530273438f, 3 },
};

int main(void)
{
    const char *rb = getenv("ROCKET_OUT_CVT_ROUND");
    g_away = rb && atoi(rb);
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no device (%d)\n", fd); return 2; }

    rocket_conv2d_desc d;
    memset(&d, 0, sizeof d);
    d.ic = IC; d.ih = HW; d.iw = HW; d.oc = OC; d.kh = 1; d.kw = 1;
    d.stride_y = d.stride_x = 1; d.dil_y = d.dil_x = 1;
    if (rocket_conv2d_int8_q_plan(&d) != 0) { fprintf(stderr, "plan refused\n"); return 2; }

    static int8_t in[IC * HW * HW], W[OC * IC], out[OC * HW * HW];
    static int32_t bias[OC];
    uint32_t s = 7;
    for (int i = 0; i < OC * IC; i++) { s = s * 1103515245u + 12345u; W[i] = (int8_t)(s >> 16); }
    const int in_zp = 5;
    memset(in, in_zp, sizeof in);           /* x - in_zp == 0 everywhere: acc == bias */

    const int ZP[] = { -5, -1, 0, 1, 2, 3, 7, 100 };
    int ok_all = 1;
    printf("int8-out requant zero-point order, tie rule %s\n", g_away ? "half away" : "half to even");
    for (unsigned si = 0; si < sizeof SC / sizeof SC[0]; si++) {
        unsigned mul, shift;
        npu_out_cvt_pair(SC[si].scale, &mul, &shift);
        const int e = SC[si].e;
        printf("scale %s: MUL 0x%x SHIFT %u\n", SC[si].name, mul, shift);
        if (mul != 0x4000) { printf("  MUL is not 2^14: no tie can be built, skipped\n"); continue; }
        /* channels 0..31: ties (2k+1)*2^(e-1), k = -16..15; channels 32..63: off ties */
        for (int c = 0; c < 32; c++) bias[c] = (2 * (c - 16) + 1) * (1 << (e - 1));
        for (int c = 32; c < OC; c++) bias[c] = (c - 48) * (1 << e) + (e > 1 ? 1 : 0) * ((c & 1) ? 1 : -1);
        for (unsigned zi = 0; zi < sizeof ZP / sizeof ZP[0]; zi++) {
            const int zp = ZP[zi];
            memset(out, 0x55, sizeof out);
            int r = rocket_conv2d_int8_q(fd, &d, in, W, bias, SC[si].scale, 1.0f, 1.0f, in_zp, 0,
                                         zp, out);
            if (r) { printf("  zp %4d: call failed (%d)\n", zp, r); ok_all = 0; continue; }
            int n_pre = 0, n_post = 0, n_tie_odd = 0, n_tie_pre = 0, n_tie_post = 0, n_none = 0;
            for (int c = 0; c < OC; c++)
                for (int p = 0; p < HW * HW; p++) {
                    int g = out[c * HW * HW + p];
                    int64_t acc = bias[c];
                    int pre = sat8(rnd_shift(acc + (int64_t)zp * (1 << e), e));
                    int post = sat8(rnd_shift(acc, e) + zp);
                    n_pre += g == pre; n_post += g == post;
                    n_none += g != pre && g != post;
                    if (c < 32 && (zp & 1)) {
                        n_tie_odd++; n_tie_pre += g == pre && g != post; n_tie_post += g == post && g != pre;
                    }
                }
            const int tot = OC * HW * HW;
            printf("  zp %4d: pre %4d post %4d of %d, neither %d; ties at odd zp: %d, pre-only %d, "
                   "post-only %d\n", zp, n_pre, n_post, tot, n_none, n_tie_odd, n_tie_pre, n_tie_post);
            if (n_pre != tot && n_post != tot) ok_all = 0;
        }
    }
    rocket_close(fd);
    return ok_all ? 0 : 1;
}
