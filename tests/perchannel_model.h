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

#endif /* ROCKET_TESTS_PERCHANNEL_MODEL_H */
