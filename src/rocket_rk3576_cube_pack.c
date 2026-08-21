// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_rk3576_cube_pack.c — the CHW <-> NC1HWC2 transpose, shared by every RK3576 entry
 * that owns a feature cube.
 *
 * It lives in a file of its own because it is not a convolution's: a POOLING layer packs
 * and unpacks the same cube, and on a concat-topology graph it is the larger consumer of
 * the two. Written an element at a time it ran at about 145 MB/s against this block's
 * multiple GB/s, which on Inception V1 was 8.6 ms of an 11.2 ms pooling wall
 * [HW sweep, H96 MAX M9].
 */
#include <stddef.h>
#include <stdint.h>

#include "rocket_rk3576_internal.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define R76_HAVE_NEON 1
#endif

#define C2 16u      /* int8 feature/output channel atom */

/* ============================================================================
 * SECTION — the CHW <-> NC1HWC2 transpose
 *
 * A convolution on this part spends most of its host time HERE and not in the submit:
 * on a MobileNet graph the feature scatter and the output de-scatter are together about
 * seventy per cent of the library's wall, against the submit's thirty.
 *
 * The cube interleaves sixteen channels into every sixteen-byte atom, so a row-major
 * tensor and a cube are a TRANSPOSE rather than a copy. Written an element at a time
 * that is one useful byte per destination cache line, and the sixteen channels of a
 * group each walk the same lines again — four line touches per byte moved.
 *
 * A 16x16 BLOCK moves the same bytes in whole vectors: sixteen 16-byte loads, a 16x16
 * byte transpose (four levels of TRN1/TRN2, the standard recursive block transpose), and
 * four full cache lines stored. Each line is touched ONCE.
 *
 * CHANNEL-OUTER IS STILL THE READ ORDER, and the block does not change that. Sixteen
 * concurrent read streams is what made the scalar pixel-outer rewrite a measured
 * negative — but a block reads a whole VECTOR from each stream instead of one byte, so
 * the streams advance sixteen times more slowly for the same work and the line traffic
 * falls instead of rising.
 *
 * The scalar loop in each is the pixel tail AND the non-NEON build, and it is the loop
 * that was there before the block — so a shape whose pixel count is not a multiple of
 * sixteen runs both against each other on every gate, and most of them are.
 * ==========================================================================*/
#ifdef R76_HAVE_NEON
#define R76_U8(x)  vreinterpretq_u8_u64(x)
#define R76_U64(x) vreinterpretq_u64_u8(x)

/* Transpose eight 16-byte rows as two independent 8x8 byte blocks — the low halves and
 * the high halves. EIGHT LIVE VECTORS IS THE POINT: a 16x16 done as one array of sixteen
 * needs thirty-two registers with the scratch and the compiler puts the array on the
 * STACK, which measured 3x slower than this on the part. Two of these plus a swap of the
 * off-diagonal halves is the 16x16. */
static inline void r76_trans8(uint8x16_t *r0, uint8x16_t *r1, uint8x16_t *r2,
                              uint8x16_t *r3, uint8x16_t *r4, uint8x16_t *r5,
                              uint8x16_t *r6, uint8x16_t *r7)
{
    uint8x16_t a0, a1, a2, a3, a4, a5, a6, a7;
    uint8x16_t b0, b1, b2, b3, b4, b5, b6, b7;

#define R76_T1(o1, o2, x, y) do { uint8x16_t _x = (x), _y = (y);                       \
        o1 = vtrn1q_u8(_x, _y); o2 = vtrn2q_u8(_x, _y); } while (0)
#define R76_T2(o1, o2, x, y) do { uint16x8_t _x = vreinterpretq_u16_u8(x),             \
                                             _y = vreinterpretq_u16_u8(y);             \
        o1 = vreinterpretq_u8_u16(vtrn1q_u16(_x, _y));                                 \
        o2 = vreinterpretq_u8_u16(vtrn2q_u16(_x, _y)); } while (0)
#define R76_T4(o1, o2, x, y) do { uint32x4_t _x = vreinterpretq_u32_u8(x),             \
                                             _y = vreinterpretq_u32_u8(y);             \
        o1 = vreinterpretq_u8_u32(vtrn1q_u32(_x, _y));                                 \
        o2 = vreinterpretq_u8_u32(vtrn2q_u32(_x, _y)); } while (0)
    R76_T1(a0, a1, *r0, *r1); R76_T1(a2, a3, *r2, *r3);
    R76_T1(a4, a5, *r4, *r5); R76_T1(a6, a7, *r6, *r7);
    R76_T2(b0, b2, a0, a2); R76_T2(b1, b3, a1, a3);
    R76_T2(b4, b6, a4, a6); R76_T2(b5, b7, a5, a7);
    R76_T4(*r0, *r4, b0, b4); R76_T4(*r1, *r5, b1, b5);
    R76_T4(*r2, *r6, b2, b6); R76_T4(*r3, *r7, b3, b7);
#undef R76_T1
#undef R76_T2
#undef R76_T4
}
#endif

/*
 * ONE cube group, both directions. `sp`/`dp` are up to C2 channel planes — pointers
 * rather than a base and a stride, because a per-axis convolution sorts its output
 * channels by scale, so channel c of a group can land anywhere in the caller's tensor.
 *
 * `full` is a CONSTANT at every call site, so the compiler emits the common
 * sixteen-live-channel loop with no per-lane predicate and the partial one with it —
 * one source, two specializations, rather than two hand-written kernels that have to be
 * kept in agreement. The dead lanes of a partial group are still WRITTEN on the way in,
 * because a whole atom has to be stored either way — so they carry `pad`, which is the
 * cube's channel padding and is NOT zero: it has to be the same value the CNA substitutes
 * at a border tap, or the B term's sum stops being position-independent. See h->taps in
 * r76_w_prepare().
 */
#ifdef R76_HAVE_NEON
__attribute__((always_inline))
static inline size_t r76_c2_pack_vec(int8_t *cube, const int8_t *const *sp,
                                     unsigned live, size_t px, unsigned char pad,
                                     const int full)
{
    const uint8x16_t padv = vdupq_n_u8(pad);
    size_t p;
#define R76_LD(i) ((full || (i) < live) ? vld1q_u8((const uint8_t *)sp[i] + p) : padv)
    for (p = 0; p + C2 <= px; p += C2) {
        uint8x16_t v0, v1, v2, v3, v4, v5, v6, v7;
        uint8x16_t w0, w1, w2, w3, w4, w5, w6, w7;
        uint8_t *d = (uint8_t *)cube + p * C2;
        uint64x2_t z;
        v0 = R76_LD(0); v1 = R76_LD(1); v2 = R76_LD(2); v3 = R76_LD(3);
        v4 = R76_LD(4); v5 = R76_LD(5); v6 = R76_LD(6); v7 = R76_LD(7);
        r76_trans8(&v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7);
        w0 = R76_LD(8);  w1 = R76_LD(9);  w2 = R76_LD(10); w3 = R76_LD(11);
        w4 = R76_LD(12); w5 = R76_LD(13); w6 = R76_LD(14); w7 = R76_LD(15);
        r76_trans8(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);
        /* Row i of the low block is output row i's first eight channels and row i of the
         * high block its last eight; atom i is the two halves joined. */
#define R76_EMIT(i, lo, hi)                                                            \
        z = vcombine_u64(vget_low_u64(R76_U64(lo)), vget_low_u64(R76_U64(hi)));        \
        vst1q_u8(d + (i) * C2, R76_U8(z));                                             \
        z = vcombine_u64(vget_high_u64(R76_U64(lo)), vget_high_u64(R76_U64(hi)));      \
        vst1q_u8(d + ((i) + 8) * C2, R76_U8(z))
        R76_EMIT(0, v0, w0); R76_EMIT(1, v1, w1);
        R76_EMIT(2, v2, w2); R76_EMIT(3, v3, w3);
        R76_EMIT(4, v4, w4); R76_EMIT(5, v5, w5);
        R76_EMIT(6, v6, w6); R76_EMIT(7, v7, w7);
#undef R76_EMIT
    }
#undef R76_LD
    return p;
}

__attribute__((always_inline))
static inline size_t r76_c2_unpack_vec(int8_t *const *dp, unsigned live,
                                       const int8_t *cube, size_t px, const int full)
{
    size_t p;
    for (p = 0; p + C2 <= px; p += C2) {
        const uint8_t *s = (const uint8_t *)cube + p * C2;
        uint8x16_t v0, v1, v2, v3, v4, v5, v6, v7;
        uint8x16_t w0, w1, w2, w3, w4, w5, w6, w7;
        uint64x2_t z;
#define R76_LOAD(i, a, b) do {                                                         \
        uint8x16_t _l = vld1q_u8(s + (i) * C2), _h = vld1q_u8(s + ((i) + 8) * C2);     \
        z = vcombine_u64(vget_low_u64(R76_U64(_l)), vget_low_u64(R76_U64(_h)));        \
        a = R76_U8(z);                                                                 \
        z = vcombine_u64(vget_high_u64(R76_U64(_l)), vget_high_u64(R76_U64(_h)));      \
        b = R76_U8(z); } while (0)
        R76_LOAD(0, v0, w0); R76_LOAD(1, v1, w1);
        R76_LOAD(2, v2, w2); R76_LOAD(3, v3, w3);
        R76_LOAD(4, v4, w4); R76_LOAD(5, v5, w5);
        R76_LOAD(6, v6, w6); R76_LOAD(7, v7, w7);
#undef R76_LOAD
        r76_trans8(&v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7);
        r76_trans8(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);
#define R76_ST(i, r) if (full || (i) < live) vst1q_u8((uint8_t *)dp[i] + p, r)
        R76_ST(0, v0); R76_ST(1, v1); R76_ST(2, v2);  R76_ST(3, v3);
        R76_ST(4, v4); R76_ST(5, v5); R76_ST(6, v6);  R76_ST(7, v7);
        R76_ST(8, w0); R76_ST(9, w1); R76_ST(10, w2); R76_ST(11, w3);
        R76_ST(12, w4); R76_ST(13, w5); R76_ST(14, w6); R76_ST(15, w7);
#undef R76_ST
    }
    return p;
}
#endif  /* R76_HAVE_NEON */

/* cube[p*C2 + c] = sp[c][p] for c < live, `pad` above it. The scalar tail leaves the dead
 * lanes alone rather than filling them, because those bytes were never live and still hold
 * the allocation's own fill — which is the same `pad`. */
void rocket_rk3576_c2_pack(int8_t *cube, const int8_t *const *sp, unsigned live, size_t px,
                        unsigned char pad)
{
    size_t p = 0, q;
    unsigned c;

#ifdef R76_HAVE_NEON
    p = (live == C2) ? r76_c2_pack_vec(cube, sp, C2, px, pad, 1)
                     : r76_c2_pack_vec(cube, sp, live, px, pad, 0);
#else
    (void)pad;
#endif
    for (c = 0; c < live; c++)
        for (q = p; q < px; q++) cube[q * C2 + c] = sp[c][q];
}

/* dp[c][p] = cube[p*C2 + c] for c < live. */
void rocket_rk3576_c2_unpack(int8_t *const *dp, unsigned live, const int8_t *cube, size_t px)
{
    size_t p = 0, q;
    unsigned c;

#ifdef R76_HAVE_NEON
    p = (live == C2) ? r76_c2_unpack_vec(dp, C2, cube, px, 1)
                     : r76_c2_unpack_vec(dp, live, cube, px, 0);
#endif
    for (c = 0; c < live; c++)
        for (q = p; q < px; q++) dp[c][q] = cube[q * C2 + c];
}

/* ============================================================================
 * SECTION — the CHW -> packed-image interleave
 *
 * The packed-image (ARGB) first conv reads an image of `ic` INTERLEAVED bytes per pixel,
 * not an NC1HWC2 cube, so it needs a different transform: a 2-, 3- or 4-way interleave of
 * whole planes rather than a 16x16 block transpose. NEON stores exactly that shape with
 * one instruction — vst2/vst3/vst4 write N registers de-interleaved across N lanes — so a
 * plane pair, triple or quad moves in whole vectors from `ic` sequential read streams.
 *
 * It matters because this pack is what the packed encoding SPENDS. That encoding's
 * saving is a MAC count 8.0x smaller than the direct lowering's at a 224x224 k7 s2 stem
 * — a measured 1.70 ms of submit — and written an element at a time the interleave gave
 * most of it back: one byte per store from `ic` streams, with a multiply in the index.
 *
 * The scalar loop is the pixel tail and the non-NEON build. `ic` here is the PROGRAMMED
 * count, so a widened one-channel image passes 2 and the second plane is the caller's own
 * zeros; there is no dead-lane question, since every programmed lane is a real plane.
 * ==========================================================================*/
void rocket_rk3576_argb_pack(int8_t *img, const int8_t *const *sp, unsigned ic, size_t px)
{
    size_t p = 0, q;
    unsigned c;

#ifdef R76_HAVE_NEON
    if (ic >= 2u && ic <= 4u) {
        size_t n = px & ~(size_t)15;
        for (; p < n; p += 16) {
            if (ic == 2u) {
                int8x16x2_t v;
                v.val[0] = vld1q_s8(sp[0] + p); v.val[1] = vld1q_s8(sp[1] + p);
                vst2q_s8(img + p * 2u, v);
            } else if (ic == 3u) {
                int8x16x3_t v;
                v.val[0] = vld1q_s8(sp[0] + p); v.val[1] = vld1q_s8(sp[1] + p);
                v.val[2] = vld1q_s8(sp[2] + p);
                vst3q_s8(img + p * 3u, v);
            } else {
                int8x16x4_t v;
                v.val[0] = vld1q_s8(sp[0] + p); v.val[1] = vld1q_s8(sp[1] + p);
                v.val[2] = vld1q_s8(sp[2] + p); v.val[3] = vld1q_s8(sp[3] + p);
                vst4q_s8(img + p * 4u, v);
            }
        }
    }
#endif
    for (q = p; q < px; q++)
        for (c = 0; c < ic; c++) img[q * ic + c] = sp[c][q];
}
