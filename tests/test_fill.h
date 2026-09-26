// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * test_fill.h — the instrument's own parameters, shared by every gate.
 *
 * A gate's fill, sample stride, tolerance and output pre-state decide what it can see,
 * and each has let a wrong surface score as a right one here:
 *
 *   - A fill whose period divides a row, a plane, a channel group or a tile gives a
 *     transposed or shifted surface the same values as the right one. `W[i]=(i%3)-1` is a
 *     function of the kernel column alone on every 3x3 kernel, so a missing channel
 *     transpose passes.
 *   - A sample stride that shares a factor with N checks the same lanes over and over:
 *     stride 256 at N=4096 reads 16 of 4096 columns.
 *   - A tolerance wider than the arithmetic's exactness passes a dropped term.
 *   - An output that starts at zero passes every unwritten element whose reference is
 *     near zero.
 *
 * So this header holds one tool for each: a hashed fill with no period, a sentinel
 * prefill, an exact comparator that names the first mismatch by coordinate and counts
 * the elements still holding the sentinel, and a sample stride coprime to N.
 *
 * Everything is static inline and host-only apart from the two BO prefills.
 */
#ifndef ROCKET_TESTS_TEST_FILL_H
#define ROCKET_TESTS_TEST_FILL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"

/* ---- a fill with no period ------------------------------------------------------
 *
 * Element i of a tensor seeded `seed` is splitmix64's i-th output from a state derived
 * from the seed. Nothing in it repeats at any stride a layout can have, so two elements
 * a transpose, a flip or a wrong channel-group stride would exchange carry different
 * values unless the range forces a coincidence. Give each tensor of one gate its own
 * seed: the same seed on A and B makes them equal.
 */
static inline uint64_t tf_mix64(uint64_t z)
{
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint64_t tf_hash(uint64_t seed, uint64_t i)
{
    return tf_mix64(tf_mix64(seed + 0x632BE59BD9B4E019ULL) + i * 0x9E3779B97F4A7C15ULL);
}

/* An integer in [lo, hi], inclusive. The modulo bias is below 2^-50 for any range a
 * gate uses. */
static inline int tf_int(uint64_t seed, uint64_t i, int lo, int hi)
{
    return lo + (int)(tf_hash(seed, i) % (uint64_t)((int64_t)hi - lo + 1));
}

/* A double in [lo, hi). */
static inline double tf_real(uint64_t seed, uint64_t i, double lo, double hi)
{
    return lo + (hi - lo) * (double)(tf_hash(seed, i) >> 11) * 0x1.0p-53;
}

static inline void tf_fill_i8(int8_t *p, size_t n, uint64_t seed, int lo, int hi)
{
    for (size_t i = 0; i < n; i++) p[i] = (int8_t)tf_int(seed, i, lo, hi);
}

static inline void tf_fill_u8(uint8_t *p, size_t n, uint64_t seed, int lo, int hi)
{
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)tf_int(seed, i, lo, hi);
}

static inline void tf_fill_i16(int16_t *p, size_t n, uint64_t seed, int lo, int hi)
{
    for (size_t i = 0; i < n; i++) p[i] = (int16_t)tf_int(seed, i, lo, hi);
}

static inline void tf_fill_i32(int32_t *p, size_t n, uint64_t seed, int lo, int hi)
{
    for (size_t i = 0; i < n; i++) p[i] = (int32_t)tf_int(seed, i, lo, hi);
}

/* Integers in [lo, hi] stored as fp16. An fp16 path whose every partial sum stays within
 * +-2048 then computes exactly, whatever order it accumulates in, so the gate can compare
 * bit for bit rather than under a tolerance. */
static inline void tf_fill_f16_int(_Float16 *p, size_t n, uint64_t seed, int lo, int hi)
{
    for (size_t i = 0; i < n; i++) p[i] = (_Float16)tf_int(seed, i, lo, hi);
}

/* Reals in [lo, hi) rounded to fp16, for a gate that scores a float path. */
static inline void tf_fill_f16(_Float16 *p, size_t n, uint64_t seed, double lo, double hi)
{
    for (size_t i = 0; i < n; i++) p[i] = (_Float16)tf_real(seed, i, lo, hi);
}

static inline void tf_fill_f32(float *p, size_t n, uint64_t seed, double lo, double hi)
{
    for (size_t i = 0; i < n; i++) p[i] = (float)tf_real(seed, i, lo, hi);
}

/* ---- the sentinel prefill -------------------------------------------------------
 *
 * An output that starts at zero agrees with every reference element that is zero, and a
 * datapath that writes nothing then passes wherever the answer is small. Prefill instead.
 * An fp16 buffer gets a quiet NaN with a recognisable payload, which equals no reference
 * value. Any other buffer gets 0xAA bytes: -86 as int8 (a dropped element passes only
 * where the reference is -86, 1 in 256 at a uniform fill) and -1431655766 as int32, which
 * no gate's sum reaches.
 */
#define TF_SENTINEL_BYTE     0xAA
#define TF_SENTINEL_F16_BITS 0x7EAAu

static inline void tf_sentinel_f16(_Float16 *p, size_t n)
{
    uint16_t b = TF_SENTINEL_F16_BITS;
    for (size_t i = 0; i < n; i++) memcpy(&p[i], &b, sizeof b);
}

static inline int tf_is_sentinel_f16(_Float16 v)
{
    uint16_t b;
    memcpy(&b, &v, sizeof b);
    return b == TF_SENTINEL_F16_BITS;
}

static inline void tf_sentinel_bytes(void *p, size_t n)
{
    memset(p, TF_SENTINEL_BYTE, n);
}

/* Stamp a whole BO from the CPU. The write sits inside PREP_BO/FINI_BO because rocket BOs
 * are cached: a memset outside the bracket leaves dirty lines that can land after the
 * NPU's own write and overwrite it. Returns 0 or a negative errno. */
static inline int tf_sentinel_bo(int fd, rocket_bo *bo)
{
    int rc = rocket_bo_prep(fd, bo, 1, 0);
    if (rc) return rc;
    memset(bo->ptr, TF_SENTINEL_BYTE, bo->size);
    return rocket_bo_fini(fd, bo);
}

/* As tf_sentinel_bo, with the fp16 NaN for a BO the NPU writes in fp16. */
static inline int tf_sentinel_bo_f16(int fd, rocket_bo *bo)
{
    int rc = rocket_bo_prep(fd, bo, 1, 0);
    if (rc) return rc;
    tf_sentinel_f16((_Float16 *)bo->ptr, bo->size / sizeof(_Float16));
    return rocket_bo_fini(fd, bo);
}

/* ---- comparison that names the first mismatch -----------------------------------
 *
 * `dims` lists the tensor's extents outermost first, as the flat index walks them, and a
 * mismatch prints as a coordinate in that order. Each comparator returns the number of
 * elements that differ and prints one line when that is nonzero: the count, how many of
 * them still hold the sentinel (unwritten, as against written wrong), and the first one.
 */
static inline void tf_print_coord(size_t flat, const int *dims, int nd)
{
    int c[8];
    if (nd < 1 || nd > 8) { printf("[%zu]", flat); return; }
    for (int d = nd - 1; d >= 0; d--) {
        c[d] = (int)(flat % (size_t)dims[d]);
        flat /= (size_t)dims[d];
    }
    printf("(");
    for (int d = 0; d < nd; d++) printf(d ? ",%d" : "%d", c[d]);
    printf(")");
}

static inline size_t tf_dims_count(const int *dims, int nd)
{
    size_t n = 1;
    for (int d = 0; d < nd; d++) n *= (size_t)dims[d];
    return n;
}

#define TF_DEFINE_EXACT_CMP(NAME, T, IS_SENTINEL, EQ, FMT, CAST)                          \
static inline long NAME(const char *label, const T *got, const T *want,                 \
                        const int *dims, int nd)                                         \
{                                                                                        \
    size_t n = tf_dims_count(dims, nd), first = 0;                                       \
    long bad = 0, unwritten = 0;                                                         \
    for (size_t i = 0; i < n; i++) {                                                     \
        if (EQ(got[i], want[i])) continue;                                               \
        if (!bad) first = i;                                                             \
        bad++;                                                                           \
        if (IS_SENTINEL(got[i])) unwritten++;                                            \
    }                                                                                    \
    if (bad) {                                                                           \
        printf("%s: %ld of %zu elements differ (%ld still hold the sentinel), first at ", \
               label, bad, n, unwritten);                                                \
        tf_print_coord(first, dims, nd);                                                 \
        printf(": got " FMT ", want " FMT "\n", CAST(got[first]), CAST(want[first]));    \
    }                                                                                    \
    return bad;                                                                          \
}

#define TF_EQ_INT(a, b)       ((a) == (b))
#define TF_EQ_F16(a, b)       ((float)(a) == (float)(b))   /* +0 == -0, NaN never equal */
#define TF_SENT_I8(v)         ((uint8_t)(v) == TF_SENTINEL_BYTE)
#define TF_SENT_I16(v)        ((uint16_t)(v) == 0xAAAAu)
#define TF_SENT_I32(v)        ((uint32_t)(v) == 0xAAAAAAAAu)
#define TF_SENT_I64(v)        ((uint64_t)(v) == 0xAAAAAAAAAAAAAAAAull)
#define TF_CAST_LL(v)         ((long long)(v))
#define TF_CAST_D(v)          ((double)(v))

TF_DEFINE_EXACT_CMP(tf_cmp_i8,  int8_t,   TF_SENT_I8,         TF_EQ_INT, "%lld", TF_CAST_LL)
TF_DEFINE_EXACT_CMP(tf_cmp_u8,  uint8_t,  TF_SENT_I8,         TF_EQ_INT, "%lld", TF_CAST_LL)
TF_DEFINE_EXACT_CMP(tf_cmp_i16, int16_t,  TF_SENT_I16,        TF_EQ_INT, "%lld", TF_CAST_LL)
TF_DEFINE_EXACT_CMP(tf_cmp_i32, int32_t,  TF_SENT_I32,        TF_EQ_INT, "%lld", TF_CAST_LL)
TF_DEFINE_EXACT_CMP(tf_cmp_i64, int64_t,  TF_SENT_I64,        TF_EQ_INT, "%lld", TF_CAST_LL)
TF_DEFINE_EXACT_CMP(tf_cmp_f16, _Float16, tf_is_sentinel_f16, TF_EQ_F16, "%.9g", TF_CAST_D)

/* A float path compared under a bound: element i fails when |got - want| exceeds
 * abs_tol + rel_tol*|want|, or when got is not a number. Size the bound from the
 * arithmetic, not from what passed: it has to sit below the smallest error the gate
 * exists to catch. */
static inline long tf_cmp_tol(const char *label, const double *got, const double *want,
                              const int *dims, int nd, double abs_tol, double rel_tol)
{
    size_t n = tf_dims_count(dims, nd), first = 0;
    long bad = 0, nan = 0;
    double worst = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(got[i] - want[i]);
        int fail = isnan(got[i]) || e > abs_tol + rel_tol * fabs(want[i]);
        if (!fail) continue;
        if (!bad) first = i;
        bad++;
        if (isnan(got[i])) nan++;
        else if (e > worst) worst = e;
    }
    if (bad) {
        printf("%s: %ld of %zu elements outside |e| <= %g + %g*|want| (%ld NaN, worst %g), "
               "first at ", label, bad, n, abs_tol, rel_tol, nan, worst);
        tf_print_coord(first, dims, nd);
        printf(": got %.9g, want %.9g\n", got[first], want[first]);
    }
    return bad;
}

/* ---- a sample stride coprime to N ------------------------------------------------
 *
 * A gate that checks a sample of N columns at stride s visits only the residues that
 * gcd(s, N) allows, so a stride sharing a factor with a tile or lane width checks the
 * same lane of every tile. A stride coprime to N walks N distinct indices before it
 * repeats, and its first few land on different residues of every factor of N.
 */
static inline int tf_gcd(int a, int b)
{
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { int t = a % b; a = b; b = t; }
    return a;
}

/* The smallest stride >= want that is coprime to n. */
static inline int tf_coprime_stride(int n, int want)
{
    int s = want < 1 ? 1 : want;
    if (n <= 1) return 1;
    while (tf_gcd(s, n) != 1) s++;
    return s;
}

/* The k-th sample index of n at a coprime stride. Distinct for k < n. */
static inline int tf_sample_index(int k, int n, int stride)
{
    return (int)(((long long)k * stride) % n);
}

#endif /* ROCKET_TESTS_TEST_FILL_H */
