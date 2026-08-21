// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_lut_rk3576.c — the RK3576 DPU LUT's table builder.
 *
 * The hardware side of the LUT is two 513-entry tables and a window. This file is the
 * arithmetic that fills them: given the meaning of a datapath value and the units the
 * entries are read in, it evaluates the function at each of the 1026 grid points.
 *
 * WHAT THE GRID IS. `index = (value - le_start) / 2^sel`, with the 0x00020000 table
 * covering [le_start, 0] and the 0x00030000 one [0, lo_end), 512 intervals each, and
 * the hardware interpolating linearly between neighbours at the full resolution of the
 * step. So entry i of the low table is the function at `le_start + i*2^sel` and entry i
 * of the high table is the function at `i*2^sel` — the grid is the same either side of
 * zero and the two tables abut at it, which is why a table built this way is continuous
 * across the join without either table having to know about the other.
 *
 * THE CLAMPS ARE THE TABLES' OWN ENDPOINTS. That is the vendor's convention and it is
 * what makes the saturating tails exact: sigmoid at -inf is sigmoid at le_start to
 * within an entry, and reading the endpoint below the domain extends the curve flat
 * rather than dropping it to zero. A function that does NOT saturate (swish, elu at the
 * top) is then linear-extrapolated wrong outside its window, so a caller picks a window
 * that covers the input range — which for an int8 activation it can do exactly.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "npu_regcmd_rk3576.h"
#include "rocket_log.h"

static double r76_sigmoid(double x)
{
    /* Split at zero so neither branch overflows: exp(-x) for x >= 0 and exp(x) for
     * x < 0 both stay in [0, 1]. */
    if (x >= 0.0) return 1.0 / (1.0 + exp(-x));
    return exp(x) / (1.0 + exp(x));
}

static double r76_relu6(double x)
{
    return x < 0.0 ? 0.0 : (x > 6.0 ? 6.0 : x);
}

static double r76_act(int kind, double x)
{
    switch (kind) {
    case ROCKET_RK3576_ACT_SIGMOID:     return r76_sigmoid(x);
    case ROCKET_RK3576_ACT_TANH:        return tanh(x);
    case ROCKET_RK3576_ACT_SWISH:       return x * r76_sigmoid(x);
    case ROCKET_RK3576_ACT_HARDSWISH:   return x * r76_relu6(x + 3.0) / 6.0;
    case ROCKET_RK3576_ACT_HARDSIGMOID: return r76_relu6(x + 3.0) / 6.0;
    case ROCKET_RK3576_ACT_ELU:         return x >= 0.0 ? x : expm1(x);
    default:                            return 0.0;
    }
}

const char *rocket_rk3576_act_name(int kind)
{
    switch (kind) {
    case ROCKET_RK3576_ACT_SIGMOID:     return "sigmoid";
    case ROCKET_RK3576_ACT_TANH:        return "tanh";
    case ROCKET_RK3576_ACT_SWISH:       return "swish";
    case ROCKET_RK3576_ACT_HARDSWISH:   return "hardswish";
    case ROCKET_RK3576_ACT_HARDSIGMOID: return "hardsigmoid";
    case ROCKET_RK3576_ACT_ELU:         return "elu";
    default:                            return NULL;
    }
}

/* Round half away from zero and saturate. The DPU's own requant rounds ties to EVEN,
 * but this is the HOST building a table and the two never have to agree: what the gate
 * checks is that the part reads back the entry this wrote. */
static int16_t r76_q16(double v)
{
    double r = v < 0.0 ? ceil(v - 0.5) : floor(v + 0.5);
    if (r >  32767.0) return  32767;
    if (r < -32768.0) return -32768;
    return (int16_t)r;
}

int rocket_rk3576_lut_build(int kind, double value_scale, double entry_scale,
                            unsigned sel, int16_t *le, int16_t *lo, lut_rk3576_t *w)
{
    double step;
    long start;
    unsigned i;

    if (!le || !lo || !w || !rocket_rk3576_act_name(kind)) return -1;
    if (sel > 15u || !(value_scale > 0.0) || !(entry_scale > 0.0)) return -1;

    step  = (double)(1u << sel);
    /* The symmetric window the step implies: 512 intervals each side of zero. The
     * hardware takes any le_start and lo_end, but a caller that wants an asymmetric
     * one is placing its value with the BS stage's C and can say so there. */
    start = -512L * (long)(1u << sel);

    for (i = 0; i < RK3576_LUT_ENTRIES; i++) {
        double v_le = (double)start + (double)i * step;
        double v_lo = (double)i * step;
        le[i] = r76_q16(r76_act(kind, v_le * value_scale) / entry_scale);
        lo[i] = r76_q16(r76_act(kind, v_lo * value_scale) / entry_scale);
    }

    w->le_start = (int32_t)start;
    w->lo_end   = (int32_t)(-start);
    w->sel      = (uint8_t)sel;
    w->clamp_lo = le[0];
    w->clamp_hi = lo[RK3576_LUT_ENTRIES - 1];
    return 0;
}
