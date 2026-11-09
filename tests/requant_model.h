// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * requant_model.h — the DPU output convertor, on the CPU.
 *
 * One model, shared by every gate that predicts an int8 surface, because a requant is
 * the last thing every int8 program does and a gate that models it differently from its
 * neighbour is a gate that disagrees with the part for a reason that is not the part.
 *
 *     out = sat8( round_half_to_even(acc * MUL / 2^SHIFT + offset) )
 *
 * MUL and SHIFT come from the caller's fp32 scale through npu_out_cvt_pair()
 * (include/npu_requant.h), the one derivation the emitters program. So this model cannot
 * see a defect in that derivation: tests/requant_edge_probe scores it against the float.
 *
 * THE TIE ROUNDS TO EVEN. Measured, both signs, at two shifts (tests/requant_round_probe
 * on the H96 MAX M9): the part rounds `acc*MUL >> SHIFT` half to even — 0.5 -> 0, 1.5 ->
 * 2, -0.5 -> 0, -1.5 -> -2. Not the arithmetic-shift `(x + half) >> shift` these models
 * used to spell, which takes ties toward +inf, and not the round-half-away-from-zero the
 * ancestor IP's documentation specifies [nvdla.org/hw/v1/ias/precision.html]. Inherited
 * datapath semantics are worth a prediction and not a conclusion.
 *
 * THE OFFSET JOINS BEFORE THE ROUNDING. At CVT_ROUND 0, the setting the emitters program,
 * the converter adds the offset to the truncated quotient and then decides a tie by the
 * parity of that sum, so a tie lands on the even neighbor of quotient + offset
 * (tests/requant_zp_order_probe on the RK1: 1024 of 1024 outputs at eight zero points, both
 * tie scales, 2026-11-07). A model that rounds first and adds after is one count off at a
 * tie with an odd zero point. Under CVT_ROUND 1 the two orders differ
 * elsewhere, and this model does not cover that setting. The RK3576's order is unmeasured;
 * its tie rule is the RK3588's, so the same order is expected there.
 *
 * A TIE IS RARE AND IS NOT REACHABLE FROM A ROUND SCALE. The derivation ends in `+1`, so
 * MUL is odd for every power-of-two scale, and an odd multiplier moves an exact half off
 * the tie in the outward direction — which is why no gate has ever exercised the case and
 * why the rule had to be measured against a scale chosen to make MUL exactly 2^14. With
 * MUL odd the ties are one accumulator residue in 2^SHIFT, and half of those round to an
 * odd value where the two rules differ: about 2^-(SHIFT+1) of a surface, tens of elements
 * in a large prefill and none at all in a small gate case.
 */
#ifndef ROCKET_TESTS_REQUANT_MODEL_H
#define ROCKET_TESTS_REQUANT_MODEL_H

#include <stdint.h>

#include "npu_requant.h"

/* fp32 conv scale -> the register pair the emitters write. `shift` is the REGISTER
 * value, already pre-decremented, so the model shifts by exactly what the DPU has. */
static inline void requant_params(float conv_scale, unsigned *mul, unsigned *shift)
{
    npu_out_cvt_pair(conv_scale, mul, shift);
}

/* The rounder itself: `p >> shift` to nearest, ties to even, NOT saturated. This is the
 * one place the rule lives; everything below is packaging. */
static inline int64_t requant_round_shift(int64_t p, unsigned shift)
{
    int64_t half, rem, v;
    if (!shift) return p;
    half = (int64_t)1 << (shift - 1);
    v    = (p + half) >> shift;             /* nearest, ties toward +inf */
    rem  = p & (((int64_t)1 << shift) - 1); /* two's complement: p mod 2^shift, >= 0 */
    if (rem == half && (v & 1)) v -= 1;     /* an exact tie lands on the even side */
    return v;
}

static inline int requant_sat8(int64_t v)
{
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int)v;
}

/* `acc * mul >> shift`, saturated to int8. */
static inline int requant_apply(int64_t acc, unsigned mul, unsigned shift)
{
    return requant_sat8(requant_round_shift(acc * (int64_t)mul, shift));
}

/* The rounder with the OUT_CVT offset: `p / 2^shift + offset` to nearest, ties to even. The
 * offset is an integer at the output scale, so adding `offset * 2^shift` to `p` puts it in
 * the sum the tie is decided on, which is where the converter adds it. */
static inline int64_t requant_round_shift_zp(int64_t p, unsigned shift, int offset)
{
    return requant_round_shift(p + (int64_t)offset * ((int64_t)1 << shift), shift);
}

/* The same with the output zero point the OUT_CVT offset carries. Saturation is after the
 * offset, which is where the DPU applies it. */
static inline int requant_apply_zp(int64_t acc, unsigned mul, unsigned shift, int out_zp)
{
    return requant_sat8(requant_round_shift_zp(acc * (int64_t)mul, shift, out_zp));
}

/* Scale to result in one call, for the gates that carry the float rather than the pair. */
static inline int requant_scale(int64_t acc, float conv_scale)
{
    unsigned mul, shift;
    requant_params(conv_scale, &mul, &shift);
    return requant_apply(acc, mul, shift);
}

#endif /* ROCKET_TESTS_REQUANT_MODEL_H */
