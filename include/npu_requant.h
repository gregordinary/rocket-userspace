// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * npu_requant.h — a float requant scale -> the DPU OUT_CVT (MUL, SHIFT) pair.
 *
 * The one derivation. Every emitter that programs a per-tensor requant, every planner
 * that sizes a per-channel ramp against the gain the emitter will program, and the host
 * model the gates score against (tests/requant_model.h) call this, so none of them can
 * drift from the others. The price is that a model built on it cannot see a defect in
 * it: tests/requant_edge_probe.c scores it against the FLOAT scale instead.
 *
 *   gain = MUL / 2^SHIFT, with scale <= gain <= scale * (1 + 2^-14) at every scale
 *
 * MUL carries the scale's implicit bit and its top 14 mantissa bits, rounded up by one
 * unit, so it is 15 bits: 0x4001-0x7FFF, or 0x4000 where the rounding carried out.
 * SHIFT is the REGISTER value, 141 minus the biased exponent, already pre-decremented,
 * so a model shifts by exactly what the DPU has. A caller refuses a scale whose SHIFT
 * leaves the register's field (the RK3588 int8-out entry refuses outside 1-63).
 *
 * BIT 15 OF THE MULTIPLIER IS A SIGN. Both parts read OUT_CVT_SCALE's [15:0] as a signed
 * 16-bit value: MUL 0x8000 multiplies by -32768, and every output of the layer comes back
 * with its sign flipped [HW sweep, RK1 0x4084 and H96 MAX M9 0x40B0, 64 of 64 elements
 * each at two scales, tests/requant_edge_probe, 2026-09-27]. So a carry out of the 15 bits
 * is renormalized to 0x4000 at one less shift, which is the same gain.
 *
 * Mesa's adaptation of QNNPACK's derivation (rkt_regcmd.c), which these emitters shipped
 * with, masks ((bits >> 9) & 0x7FFF) and forces bit 14 afterwards. The mask takes the
 * exponent's low bit as bit 14, so the two agree except where mantissa bits [22:9] are
 * all set: there the +1 carries out, and MUL comes back 0x4000 at an even exponent (half
 * the scale) and 0x8000 at an odd one (the scale negated). One scale in 16384 [host
 * sweep of every mantissa at ten exponents; both measured on both parts, same probe].
 */
#ifndef NPU_REQUANT_H
#define NPU_REQUANT_H

#include <stdint.h>

static inline void npu_out_cvt_pair(float scale, unsigned *mul, unsigned *shift)
{
    union { float f; uint32_t u; } cv = { .f = scale };
    unsigned m = ((1u << 14) | ((cv.u >> 9) & 0x3FFFu)) + 1u;
    unsigned s = 127u + 31u - 32u - (cv.u >> 23) + 16u - 1u;

    if (m == (1u << 15)) {
        /* A shift of zero is a scale near 2^14, which no requant reaches; the largest
         * positive multiplier is the nearest gain the field holds there. */
        if (s) { m = 1u << 14; s--; }
        else   m = 0x7FFFu;
    }
    *mul = m;
    *shift = s;
}

#endif /* NPU_REQUANT_H */
