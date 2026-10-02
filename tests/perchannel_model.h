// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * perchannel_model.h — the RK3576 per-channel requant ramp, planned on the CPU.
 *
 * The per-channel entries (the convolution's and the matmul's per-column form) carry each
 * channel's scale as an int16 C ramp over one (MUL, SHIFT) per tile, which the library's
 * rocket_rk3576_plan_perchannel() plans. This is that planner spelled again from the
 * documented rule, so a gate can predict the registers the part runs WITHOUT calling the
 * planner it is checking: a model that shares the planner cannot catch the planner. One
 * copy, shared by rk3576_perchannel_gate and rk3576_mm_requant.
 *
 * The rule: every channel must reach its own scale with a C that neither exceeds the
 * int16 field nor saturates the int32 product, where |acc| <= 128*sum|w| and the bias
 * term rides with it. The tightest such base gain is quantized into the (MUL, SHIFT) pair
 * the emitter programs, and each channel's C is its scale over the gain that pair gives,
 * rounded half up and held to [1, 32767].
 *
 * `perm` maps a tile slot to a channel (NULL is the identity), `A` is indexed by slot
 * from 0 at the layer's first channel, and C is written at `C[oc0 + j]`.
 */
#ifndef ROCKET_TESTS_PERCHANNEL_MODEL_H
#define ROCKET_TESTS_PERCHANNEL_MODEL_H

#include <math.h>
#include <stdint.h>
#include <limits.h>
#include "requant_model.h"

static inline void plan_c(unsigned oc0, unsigned tile_oc, const unsigned *perm,
                          const int64_t *sum_abs_w, const int32_t *A,
                          float in_scale, const float *w_scale, float out_scale,
                          int16_t *C, unsigned *mul, unsigned *shift)
{
    double best = 0.0, base;
    unsigned j;
    for (j = 0; j < tile_oc; j++) {
        unsigned c = perm ? perm[oc0 + j] : oc0 + j;
        double cs = (double)in_scale * (double)w_scale[c] / (double)out_scale;
        double bound = 128.0 * (double)sum_abs_w[c] + fabs((double)A[oc0 + j]) + 1.0;
        double cmax = (double)INT32_MAX / bound;
        double need;
        if (cmax > 32767.0) cmax = 32767.0;
        if (cmax < 1.0)     cmax = 1.0;
        need = cs / cmax;
        if (need > best) best = need;
    }
    requant_params((float)best, mul, shift);
    base = (double)*mul / (double)((uint64_t)1 << *shift);
    for (j = 0; j < tile_oc; j++) {
        unsigned c = perm ? perm[oc0 + j] : oc0 + j;
        double cs = (double)in_scale * (double)w_scale[c] / (double)out_scale;
        long long v = (long long)(cs / base + 0.5);
        if (v < 1)     v = 1;
        if (v > 32767) v = 32767;
        C[oc0 + j] = (int16_t)v;
    }
}

/* ---- the shift ramp -------------------------------------------------------------
 *
 * The convolution's per-axis path by default: rocket_rk3576_plan_perchannel_bs(), spelled
 * again from its documented rule.
 *
 * A channel whose exact output is one byte over its whole reachable accumulator (raw int8
 * input, so [A - 128*pos - 127*neg, A + 127*pos + 128*neg] over its positive and negative
 * weight sums), with 0.01 of a count to spare, is a CONSTANT: `cst[j]` carries that byte
 * and the part is held to it exactly. Every other channel is live: the largest live scale
 * sits at C = 32767, the BS shift `bs` is the smallest at which every live product fits
 * int32 at its bound (raised while a constant's byte is out of reach at C = 32767), the
 * OUT_CVT carries G = best*2^bs, and C = round(cs/(G/2^bs)) held to [1, 32767]. A scale
 * whose derived OUT_CVT pair would come back at half its value is replaced by the next
 * power of two, as the library does.
 *
 * `cst[j]` is -1000 for a live channel. Returns the number of constant channels. The
 * library also returns a constant it cannot reach to the ramp; this model does not, so a
 * cell that needs that fallback fails here rather than passing on a shared shortcut.
 *
 * `sum_w` NULL plans every channel live, which is the matmul's per-column entry: its
 * resident-weight caller hands over only the sums of |B|, so no column's signed reach is
 * known and none can be proved constant. */
#define PC_MODEL_LIVE (-1000)

static inline int pc_const_byte(double A, int64_t sw, int64_t sa, double cs, int out_zp)
{
    const double M = 0.01;
    double pos = (double)(sa + sw) / 2.0, neg = (double)(sa - sw) / 2.0;
    double ylo = (A - 128.0 * pos - 127.0 * neg) * cs + (double)out_zp;
    double yhi = (A + 127.0 * pos + 128.0 * neg) * cs + (double)out_zp;
    double tt;
    if (sa == 0) {
        long q = lround(A * cs + (double)out_zp);
        return q < -128 ? -128 : (q > 127 ? 127 : (int)q);
    }
    if (yhi < -127.5 - M) return -128;
    if (ylo >  126.5 + M) return 127;
    tt = floor(ylo + 0.5);
    if (tt >= -128.0 && tt <= 127.0 && ylo > tt - 0.5 + M && yhi < tt + 0.5 - M)
        return (int)tt;
    return PC_MODEL_LIVE;
}

static inline unsigned plan_c_bs(unsigned oc0, unsigned tile_oc, const unsigned *perm,
                                 const int64_t *sum_w, const int64_t *sum_abs_w,
                                 const int32_t *A, float in_scale, const float *w_scale,
                                 float out_scale, int out_zp, int16_t *C, int *cst,
                                 unsigned *mul, unsigned *shift, unsigned *bs)
{
    double mhi = 0.0, reach = 0.0, best, realized, base;
    unsigned j, s, nconst = 0;
    float G;
#define PCM_CS(j) ((double)in_scale * \
                   (double)w_scale[perm ? perm[oc0 + (j)] : oc0 + (j)] / (double)out_scale)
#define PCM_CH(j) (perm ? perm[oc0 + (j)] : oc0 + (j))
    for (j = 0; j < tile_oc; j++) {
        unsigned c = PCM_CH(j);
        cst[oc0 + j] = sum_w ? pc_const_byte((double)A[oc0 + j], sum_w[c], sum_abs_w[c],
                                             PCM_CS(j), out_zp)
                             : PC_MODEL_LIVE;
        if (cst[oc0 + j] == PC_MODEL_LIVE) { if (PCM_CS(j) > mhi) mhi = PCM_CS(j); }
        else {
            double r = fabs((double)(cst[oc0 + j] - out_zp)) + 2.0;
            if (r > reach) reach = r;
            nconst++;
        }
    }
    if (!(mhi > 0.0))
        for (j = 0; j < tile_oc; j++) if (PCM_CS(j) > mhi) mhi = PCM_CS(j);
    for (s = 0; s < 48u; s++) {
        double room = 2147483647.0 * ldexp(1.0, (int)s);
        int fits = 1;
        for (j = 0; j < tile_oc; j++) {
            unsigned c = PCM_CH(j);
            double bound = 128.0 * (double)sum_abs_w[c] + fabs((double)A[oc0 + j]) + 1.0;
            if (cst[oc0 + j] != PC_MODEL_LIVE) continue;
            if (bound * (PCM_CS(j) * 32767.0 / mhi + 1.0) > room) { fits = 0; break; }
        }
        if (fits) break;
    }
    for (;;) {
        double vmax;
        best = 0.0;
        for (j = 0; j < tile_oc; j++) {
            unsigned c = PCM_CH(j);
            double bound = 128.0 * (double)sum_abs_w[c] + fabs((double)A[oc0 + j]) + 1.0;
            double cmax = 2147483647.0 * ldexp(1.0, (int)s) / bound;
            if (cst[oc0 + j] != PC_MODEL_LIVE) continue;
            if (cmax > 32767.0) cmax = 32767.0;
            if (cmax < 1.0) cmax = 1.0;
            if (PCM_CS(j) / cmax > best) best = PCM_CS(j) / cmax;
        }
        if (!(best > 0.0)) best = mhi / 32767.0;
        G = (float)(best * ldexp(1.0, (int)s));
        requant_params(G, mul, shift);
        realized = (double)*mul / ldexp(1.0, (int)*shift);
        if (realized < (double)G * (1.0 - 1e-6)) {
            G = (float)ldexp(1.0, (int)ceil(log2((double)G)));
            requant_params(G, mul, shift);
            realized = (double)*mul / ldexp(1.0, (int)*shift);
        }
        vmax = 2147483647.0 * 32767.0 / ldexp(1.0, (int)s);
        if (vmax > 2147483647.0) vmax = 2147483647.0;
        if (realized * vmax >= reach || s >= 48u) break;
        s++;
    }
    base = realized / ldexp(1.0, (int)s);
    for (j = 0; j < tile_oc; j++) {
        long long v = (long long)(PCM_CS(j) / base + 0.5);
        if (v < 1)     v = 1;
        if (v > 32767) v = 32767;
        C[oc0 + j] = (int16_t)v;
    }
    *bs = s;
#undef PCM_CS
#undef PCM_CH
    return nconst;
}

/* The epilogue under a BS shift: (acc + A)*C held wide, rounded half to even at the shift
 * and saturated to int32, then the OUT_CVT. At bs == 0 this is the shift-0 model. */
static inline int epilogue_bs(int64_t acc_plus_a, int64_t c, unsigned bs, unsigned mul,
                              unsigned shift, int out_zp)
{
    int64_t v = requant_round_shift(acc_plus_a * c, bs);
    if (v >  (int64_t)INT32_MAX) v = INT32_MAX;
    if (v <  (int64_t)INT32_MIN) v = INT32_MIN;
    return requant_sat8(requant_round_shift(v * (int64_t)mul, shift) + out_zp);
}

#endif /* ROCKET_TESTS_PERCHANNEL_MODEL_H */
