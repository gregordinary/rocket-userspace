// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * requant_model.h — the DPU output convertor, on the CPU.
 *
 * One model, shared by every gate that predicts an int8 surface, because a requant is
 * the last thing every int8 program does and a gate that models it differently from its
 * neighbour is a gate that disagrees with the part for a reason that is not the part.
 *
 *     out = sat8( round_half_to_even(acc * MUL >> SHIFT) + offset )
 *
 * MUL and SHIFT come from the caller's fp32 scale through the vendor's (QNNPACK)
 * derivation, which is what the emitters program.
 *
 * THE TIE ROUNDS TO EVEN. Measured, both signs, at two shifts (tests/requant_round_probe
 * on the H96 MAX M9): the part rounds `acc*MUL >> SHIFT` half to even — 0.5 -> 0, 1.5 ->
 * 2, -0.5 -> 0, -1.5 -> -2. Not the arithmetic-shift `(x + half) >> shift` these models
 * used to spell, which takes ties toward +inf, and not the round-half-away-from-zero the
 * ancestor IP's documentation specifies [nvdla.org/hw/v1/ias/precision.html]. Inherited
 * datapath semantics are worth a prediction and not a conclusion.
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

/* fp32 conv scale -> the register pair the emitters write. `shift` is the REGISTER
 * value, already pre-decremented, so the model shifts by exactly what the DPU has. */
static inline void requant_params(float conv_scale, unsigned *mul, unsigned *shift)
{
    union { float f; uint32_t u; } cv;
    uint32_t bits;
    unsigned m;
    cv.f = conv_scale;
    bits = cv.u;
    *shift = 127u + 31u - 32u - (bits >> 23) + 16u - 1u;
    m = ((bits >> 9) & 0x7FFFu) + 1u;
    if (m < (1u << 14)) m |= (1u << 14);
    *mul = m;
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

/* The same, then the output zero point the OUT_CVT offset carries. Saturation is after
 * the offset, which is where the DPU applies it. */
static inline int requant_apply_zp(int64_t acc, unsigned mul, unsigned shift, int out_zp)
{
    return requant_sat8(requant_round_shift(acc * (int64_t)mul, shift) + out_zp);
}

/* Scale to result in one call, for the gates that carry the float rather than the pair. */
static inline int requant_scale(int64_t acc, float conv_scale)
{
    unsigned mul, shift;
    requant_params(conv_scale, &mul, &shift);
    return requant_apply(acc, mul, shift);
}

#endif /* ROCKET_TESTS_REQUANT_MODEL_H */
