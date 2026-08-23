// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_norm_internal.h — the fp16-square overflow guard the normalization family shares.
 *
 * Every op that reduces x^2 on the NPU hits the same wall: the square is computed in fp16,
 * so |x| above ~223 overflows (223^2 = 49729, against fp16's 65504) and the reduction
 * returns inf for a row the fp64 reference handles fine. The guard is a POWER-OF-TWO
 * prescale p = 2^-k applied before the square, which is exact (no rounding), with the
 * true mean-square recovered on the host as the reduced sum times 4^k.
 *
 * It was written out three times -- once factored in rocket_normvision.c and open-coded
 * twice in rocket_norm.c, threshold, ceilf(log2f()) and all. One copy is enough: the
 * 223 is a hardware fact, and a copy that drifted from it would return inf on exactly
 * the inputs the guard exists for.
 */
#ifndef ROCKET_NORM_INTERNAL_H
#define ROCKET_NORM_INTERNAL_H

#include <math.h>
#include <stddef.h>

/* Pick k so (|x|max * 2^-k)^2 stays under fp16 max. k == 0 -- the common case -- means x
 * is used directly with no prescale and no copy. */
static inline int rocket_square_prescale_k(const _Float16 *x, size_t n)
{
    float amax = 0.f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf((float)x[i]);
        if (a > amax) amax = a;
    }
    return (amax > 223.f) ? (int)ceilf(log2f(amax / 223.f)) : 0;
}

#endif /* ROCKET_NORM_INTERNAL_H */
