// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * Declares `npu_out_cvt_pair`, which converts a float requant scale into the DPU `OUT_CVT`
 * (MUL, SHIFT) pair.
 *
 * It is the library's one derivation of the pair, and three kinds of code call it:
 *
 * - Every emitter that programs a per-tensor requant.
 * - Every planner that sizes a per-channel ramp against the gain the emitter will program.
 * - The host model the tests score against (`tests/requant_model.h`).
 *
 * None of them can therefore drift from the others. A model built on this function cannot see
 * a defect in it, so `tests/requant_edge_probe.c` scores it against the float scale instead.
 *
 *   gain = MUL / 2^SHIFT, with scale <= gain <= scale * (1 + 2^-14) at every scale
 *
 * `MUL` contains the scale's implicit bit and its top 14 mantissa bits, rounded up by one
 * unit. It is therefore 15 bits: 0x4001-0x7FFF, or 0x4000 where the rounding produced a
 * carry-out. `SHIFT` is the register value: 141 minus the biased exponent, already
 * pre-decremented. A host model therefore shifts by exactly the amount the DPU applies. The
 * caller must refuse a scale whose `SHIFT` falls outside the register's field. The RK3588
 * int8-out entry refuses one outside 1-63.
 *
 * Caution: bit 15 of the multiplier is a sign. The RK3588 and the RK3576 both read
 * `OUT_CVT_SCALE` [15:0] as a signed 16-bit value. `MUL` 0x8000 multiplies by -32768, and
 * every output of the layer comes back with its sign flipped. This function therefore
 * renormalizes a carry-out of the 15 bits to 0x4000 at one less shift, which is the same gain.
 *
 * Evidence: [HW sweep, RK1 0x4084 and H96 MAX M9 0x40B0, 2026-09-27], 64 of 64 elements each
 * at two scales (`tests/requant_edge_probe`).
 *
 * Caution: Mesa's adaptation of QNNPACK's derivation (`rkt_regcmd.c`) halves or negates one
 * scale in 16384. It masks `((bits >> 9) & 0x7FFF)` and forces bit 14 afterwards. The mask
 * takes the exponent's low bit as bit 14. Mesa's pair and this function's therefore agree
 * except where mantissa bits [22:9] are all set. There the +1 produces a carry-out, and `MUL`
 * comes back 0x4000 at an even exponent (half the scale) and 0x8000 at an odd one (the scale
 * negated).
 *
 * Evidence: one scale in 16384 [host-computed, every mantissa at ten exponents]. Both outcomes
 * were measured on both SoCs with `tests/requant_edge_probe`.
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
        /* A shift of zero is a scale near 2^14, which no requant reaches. The largest
         * positive multiplier is the nearest gain the field holds there. */
        if (s) { m = 1u << 14; s--; }
        else   m = 0x7FFFu;
    }
    *mul = m;
    *shift = s;
}

#endif /* NPU_REQUANT_H */
