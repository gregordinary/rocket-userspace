// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_cube.h — the NPU tile-layout index math, and the blocked moves that invert it.
 *
 * Where an element lands in an NPU cube is a property of the SILICON, not of the entry
 * point that packs it, so it is written once. It used to live in five places: exported
 * out-of-line in npu_regcmd.c, re-implemented as static inlines in rocket_matmul.c "to
 * kill the per-element out-of-line call+divide", and copied again into
 * rocket_prepacked_int8.c, rocket_prepacked_int4.c and rocket_bf16_stream.c, each
 * carrying a comment saying it was a copy. Every copy was pinned by a bit-exactness
 * test, which is why they never diverged in VALUE — but it meant that blocking the
 * scatters, the single largest host cost of a matmul, was a five-times job.
 *
 * All of them are 1-BASED in the channel/row index, matching weight_fp16() and the
 * hardware's own numbering.
 */
#ifndef ROCKET_CUBE_H
#define ROCKET_CUBE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ── The cube moves are BLOCKED, not per element ─────────────────────────────────
 *
 * Every NPU cube in this library is the SAME shape at a different granule, and in each of
 * them a run of consecutive channel indices is a run of consecutive elements at BOTH
 * ends. Written per element — `slot[idx_fn(...)] = src[...]` — that pays an out-of-line
 * call, an integer divide and a modulo to move ONE element that was already adjacent to
 * the last, and it is the dominant host cost of a matmul: the fp16 path measured its own
 * packA at "the bulk of packA wall time" before it was blocked, the RK3576 sibling
 * measured a K*N weight scatter at "most of that shape's wall clock", and the resident
 * int8 route measured 195 ms -> 65 ms on the RK1 from exactly this change.
 *
 * The index functions belong in the loop BOUNDS. Written once here rather than per
 * dtype, because the algebra is one family:
 *
 *   FEATURE / OUTPUT   idx(H, ch, h) = (ch-1)/G * H*G + G*(h-1) + (ch-1)%G
 *                      -> for fixed h, G consecutive ch are G consecutive elements;
 *                         groups stride by H*G.
 *   WEIGHT             idx(C, k, c)  = (c-1)/KG * KG*NG + (k-1)/NG * NG*C
 *                                    + (c-1)%KG + (k-1)%NG * KG
 *                      -> for fixed k, KG consecutive c are KG consecutive elements;
 *                         groups stride by KG*NG.
 *
 * G / KG / NG per dtype: fp16 (8, 32, 16), int8 (16, 32, 32), int4 (32 nibbles, 32, 64),
 * int16+bf16 (8, 32, 16), tf32 (4, 16, 16); the 4-byte output cube is G=4 everywhere.
 * The contiguous run is 16 BYTES on the feature side for every one of them.
 *
 * The scalar tail is the same expression the whole loop used to be, so a tile whose
 * channel count is not a whole number of groups runs the two forms against each other
 * on every gate. Bit-identical by construction — same elements, same positions, and for
 * the converting dtypes the same per-element conversion, just without the index math. */

/* Scatter a row-major [r0.., c0..] tile into a FEATURE cube of granule G.
 * CVT converts one source element (MM_ID where the types already match). */
#define MM_ID(x) (x)
#define MM_CUBE_SCATTER(DT, ST, G, IDX, slot, src, sstride, r0, c0, ROWS, COLS, CVT)     \
    do {                                                                                 \
        const int mmH_ = (ROWS), mmng_ = (COLS) / (G);                                   \
        for (int mmh_ = 1; mmh_ <= (ROWS); mmh_++) {                                     \
            const ST *mms_ = (src) + (size_t)((r0) + mmh_ - 1) * (size_t)(sstride) + (c0); \
            DT *mmd_ = (DT *)(slot) + (size_t)(G) * (size_t)(mmh_ - 1);                  \
            for (int mmg_ = 0; mmg_ < mmng_; mmg_++) {                                   \
                DT *restrict mmdg_ = mmd_ + (size_t)mmg_ * (size_t)mmH_ * (G);           \
                const ST *restrict mmsg_ = mms_ + (size_t)mmg_ * (G);                    \
                for (int mmj_ = 0; mmj_ < (G); mmj_++) mmdg_[mmj_] = CVT(mmsg_[mmj_]);   \
            }                                                                            \
            for (int mmc_ = mmng_ * (G) + 1; mmc_ <= (COLS); mmc_++)   /* COLS%G tail */ \
                ((DT *)(slot))[IDX(mmH_, mmc_, mmh_)] = CVT(mms_[mmc_ - 1]);             \
        }                                                                                \
    } while (0)

/* Scatter a row-major [n0.., k0..] tile into a WEIGHT cube of granule (KG, NG). */
#define MM_WT_SCATTER(DT, ST, KG, NG, IDX, slot, src, sstride, n0, k0, NT, KT, CVT)      \
    do {                                                                                 \
        const int mmC_ = (KT), mmng_ = (KT) / (KG);                                      \
        for (int mmk_ = 1; mmk_ <= (NT); mmk_++) {                                       \
            const ST *mms_ = (src) + (size_t)((n0) + mmk_ - 1) * (size_t)(sstride) + (k0); \
            DT *mmd_ = (DT *)(slot) + (size_t)((mmk_ - 1) / (NG)) * (NG) * (size_t)mmC_  \
                                    + (size_t)((mmk_ - 1) % (NG)) * (KG);                \
            for (int mmg_ = 0; mmg_ < mmng_; mmg_++) {                                   \
                DT *restrict mmdg_ = mmd_ + (size_t)mmg_ * (KG) * (NG);                  \
                const ST *restrict mmsg_ = mms_ + (size_t)mmg_ * (KG);                   \
                for (int mmj_ = 0; mmj_ < (KG); mmj_++) mmdg_[mmj_] = CVT(mmsg_[mmj_]);  \
            }                                                                            \
            for (int mmc_ = mmng_ * (KG) + 1; mmc_ <= (KT); mmc_++)    /* KT%KG tail */  \
                ((DT *)(slot))[IDX(mmC_, mmk_, mmc_)] = CVT(mms_[mmc_ - 1]);             \
        }                                                                                \
    } while (0)

/* Scatter a row-major conv weight W[OC][IC][KH][KW] into the conv weight cube
 * (OC/OG, IC/32, KH, KW, OG, 32), whose contiguous axis is the INPUT channel:
 *
 *   idx = ((((oc1*nIC1 + ic1)*KH + kh)*KW + kw)*OG + oc2)*32 + ic2
 *
 * OG is the output-channel group -- 16 for fp16, 32 for int8 -- and is the only thing
 * that separates the two.
 *
 * IC INNERMOST, not kw. The loop that wrote this nested oc, ic, kh, kw with kw
 * innermost, which walks the SOURCE contiguously and steps the DESTINATION by OG*32
 * elements -- a fresh cache line touched for every single element written. Reordering
 * makes the destination a 32-element run and the source a stride-KH*KW gather: about ten
 * lines read against thirty-two written.
 *
 * SIZE THE CLAIM: measured on the RK1 (RK3588, 600 MHz), best of 5, the weight scatter
 * is 28.2% of a 128-channel 28x28 k3s1 fp16 conv call, 1.4% of a 256-channel 56x56 one
 * and 1.2% of a 64-channel 112x112 one. The term INVERTS with shape -- the big-MAC
 * layers are device-bound and the small-channel ones host-bound -- so this is a lever on
 * a MobileNet-class graph's middle layers and a rounding error on a ResNet's wide ones.
 * (The RK3576's "most of that shape's wall clock" for the analogous scatter does NOT
 * transfer as a number; only the mechanism does.)
 *
 * The dead lanes of a partial ic group are not written, exactly as before, so the
 * caller's prezero still supplies them. */
#define ROCKET_CONV_WT_SCATTER(DT, OG, dst, W, OC, IC, KH, KW)                          \
    do {                                                                                \
        const int rcw_nIC1 = ((IC) + 31) / 32;                                          \
        const size_t rcw_kstep = (size_t)(KH) * (size_t)(KW);   /* source ic stride */  \
        for (int rcw_oc = 0; rcw_oc < (OC); rcw_oc++) {                                 \
            const int rcw_oc1 = rcw_oc / (OG), rcw_oc2 = rcw_oc % (OG);                 \
            for (int rcw_kh = 0; rcw_kh < (KH); rcw_kh++)                                \
                for (int rcw_kw = 0; rcw_kw < (KW); rcw_kw++)                            \
                    for (int rcw_ic1 = 0; rcw_ic1 < rcw_nIC1; rcw_ic1++) {               \
                        DT *restrict rcw_d = (DT *)(dst)                                 \
                            + (((((size_t)rcw_oc1 * rcw_nIC1 + rcw_ic1) * (KH) + rcw_kh) \
                                * (KW) + rcw_kw) * (OG) + rcw_oc2) * 32;                 \
                        const DT *restrict rcw_s = (const DT *)(W)                       \
                            + ((size_t)rcw_oc * (IC) + (size_t)rcw_ic1 * 32) * rcw_kstep \
                            + (size_t)rcw_kh * (KW) + rcw_kw;                            \
                        int rcw_n = (IC) - rcw_ic1 * 32;                                 \
                        if (rcw_n > 32) rcw_n = 32;                                      \
                        for (int rcw_j = 0; rcw_j < rcw_n; rcw_j++)                      \
                            rcw_d[rcw_j] = rcw_s[(size_t)rcw_j * rcw_kstep];             \
                    }                                                                    \
        }                                                                                \
    } while (0)

/* Gather an OUTPUT cube of granule G. ACC(h, nn, v) is the caller's accumulate, with
 * 1-based row h and column nn and the cube element v — kept as a parameter so each
 * readback's scaling stays visible at its own site. */
#define MM_CUBE_GATHER(CT, G, IDX, cube, ROWS, COLS, ACC)                                \
    do {                                                                                 \
        const int mmH_ = (ROWS), mmng_ = (COLS) / (G);                                   \
        for (int mmh_ = 1; mmh_ <= (ROWS); mmh_++) {                                     \
            const CT *mmc_ = (const CT *)(cube) + (size_t)(G) * (size_t)(mmh_ - 1);      \
            for (int mmg_ = 0; mmg_ < mmng_; mmg_++) {                                   \
                const CT *restrict mmcg_ = mmc_ + (size_t)mmg_ * (size_t)mmH_ * (G);     \
                for (int mmj_ = 0; mmj_ < (G); mmj_++)                                   \
                    ACC(mmh_, mmg_ * (G) + mmj_ + 1, mmcg_[mmj_]);                       \
            }                                                                            \
            for (int mmn_ = mmng_ * (G) + 1; mmn_ <= (COLS); mmn_++)   /* COLS%G tail */ \
                ACC(mmh_, mmn_, ((const CT *)(cube))[IDX(mmH_, mmn_, mmh_)]);            \
        }                                                                                \
    } while (0)


/* int8 NPU layout index math (cf. feat_idx/wt_idx). Input feature cube C2=16,
 * weight k-group 32 (== weight_int8()), int32-output cube C2=4. */
static inline size_t feat_idx_i8(int H, int ch, int h) {   /* input, C2=16 */
    return ((size_t)(ch - 1) / 16) * (size_t)H * 16 + 16 * (size_t)(h - 1) + (ch - 1) % 16;
}
static inline size_t wt_idx_i8(int C, int k, int c) {      /* weight, k-group 32 */
    return (size_t)((c - 1) / 32) * 32 * 32 + (size_t)((k - 1) / 32) * 32 * C
         + (size_t)((c - 1) % 32) + (size_t)((k - 1) % 32) * 32;
}
static inline size_t out_idx_i8(int H, int ch, int h) {    /* output, C2=4 */
    return ((size_t)(ch - 1) / 4) * (size_t)H * 4 + 4 * (size_t)(h - 1) + (ch - 1) % 4;
}

/* int4 NPU layout index math (NIBBLE indices for in/wt; int16 elems for out). */
static inline size_t feat_idx_i4(int H, int ch, int h) {   /* input nibble, C2=32 */
    return ((size_t)(ch - 1) / 32) * (size_t)H * 32 + 32 * (size_t)(h - 1) + (ch - 1) % 32;
}
static inline size_t wt_idx_i4(int C, int k, int c) {      /* weight nibble, (N/64,K/32,64,32) */
    size_t nKgrp   = (size_t)((C + 31) / 32);
    size_t Ngrp    = (size_t)(k - 1) / 64, Nwithin = (size_t)(k - 1) % 64;
    size_t Kgrp    = (size_t)(c - 1) / 32, Kwithin = (size_t)(c - 1) % 32;
    return Ngrp * nKgrp * 64 * 32 + Kgrp * 64 * 32 + Nwithin * 32 + Kwithin;
}
static inline size_t out_idx_i4(int H, int ch, int h) {    /* output int16 elem, C2=8 */
    return ((size_t)(ch - 1) / 8) * (size_t)H * 8 + 8 * (size_t)(h - 1) + (ch - 1) % 8;
}
/* set nibble `idx` (byte idx/2; even=low, odd=high) in a packed buffer. */
static inline void put_nib(uint8_t *buf, size_t idx, int8_t v) {
    uint8_t nib = (uint8_t)(v & 0xF);
    if (idx & 1) buf[idx >> 1] = (uint8_t)((buf[idx >> 1] & 0x0F) | (nib << 4));
    else         buf[idx >> 1] = (uint8_t)((buf[idx >> 1] & 0xF0) | nib);
}

/* The int4 cubes are NIBBLE-indexed, so the blocked form is its own: G = 32 nibbles is
 * 16 WHOLE BYTES, because a group's base nibble index always carries a factor of 32 and
 * is therefore even. Packing two source elements per byte also removes put_nib's READ
 * side entirely — it was a load, a mask, an or and a store per FOUR BITS, into a slot
 * the caller had memset to zero immediately before. */
static inline void i4_pack16(uint8_t *restrict d, const int8_t *restrict s)
{
    for (int j = 0; j < 16; j++)
        d[j] = (uint8_t)((s[2 * j] & 0xF) | ((s[2 * j + 1] & 0xF) << 4));
}

/* packA: row-major A[m0.., k0..] -> the C2=32-nibble int4 feature cube. */
static inline void i4_feat_scatter(uint8_t *restrict slot, const int8_t *restrict A,
                                   size_t Astride, int m0, int k0, int Mtile, int Ktile)
{
    const int H = Mtile, ng = Ktile / 32;
    for (int h = 1; h <= Mtile; h++) {
        const int8_t *restrict arow = A + (size_t)(m0 + h - 1) * Astride + k0;
        for (int g = 0; g < ng; g++)   /* feat_idx_i4(H, 32g+1, h) = g*H*32 + 32*(h-1) */
            i4_pack16(slot + (((size_t)g * H * 32 + 32 * (size_t)(h - 1)) >> 1),
                      arow + (size_t)g * 32);
        for (int c = ng * 32 + 1; c <= Ktile; c++)          /* Ktile%32 tail */
            put_nib(slot, feat_idx_i4(H, c, h), arow[c - 1]);
    }
}

/* packB: row-major B[n0.., k0..] -> the (N/64,K/32,64,32)-nibble int4 weight cube. */
static inline void i4_wt_scatter(uint8_t *restrict slot, const int8_t *restrict B,
                                 size_t Bstride, int n0, int k0, int Ntile, int Ktile)
{
    const size_t nKgrp = (size_t)((Ktile + 31) / 32);
    const int ng = Ktile / 32;
    for (int k = 1; k <= Ntile; k++) {
        const int8_t *restrict brow = B + (size_t)(n0 + k - 1) * Bstride + k0;
        const size_t nbase = (size_t)((k - 1) / 64) * nKgrp * 64 * 32
                           + (size_t)((k - 1) % 64) * 32;
        for (int g = 0; g < ng; g++)   /* the Kgrp stride is 64*32 nibbles */
            i4_pack16(slot + ((nbase + (size_t)g * 64 * 32) >> 1), brow + (size_t)g * 32);
        for (int c = ng * 32 + 1; c <= Ktile; c++)          /* Ktile%32 tail */
            put_nib(slot, wt_idx_i4(Ktile, k, c), brow[c - 1]);
    }
}


/* fp32 -> bf16 by TRUNCATION of the low 16 bits, which is what the hardware's bf16 MAC
 * consumes -- not a round-to-nearest-even narrowing. Here rather than per path because
 * it is half of the bf16 cube's contract: the cube says where, this says what. */
static inline uint16_t f32_to_bf16(float f) {
    uint32_t b; memcpy(&b, &f, sizeof b); return (uint16_t)(b >> 16);
}

/* int16 NPU layout index math. Feature cube C2=8 (== fp16); weight (N/16,K/32,
 * 16,32) (== weight_fp16); int32-output cube C2=4 (== int8). */
static inline size_t feat_idx_i16(int H, int ch, int h) {   /* input, C2=8 */
    return ((size_t)(ch - 1) / 8) * (size_t)H * 8 + 8 * (size_t)(h - 1) + (ch - 1) % 8;
}
static inline size_t wt_idx_i16(int C, int k, int c) {      /* weight, (N/16,K/32,16,32) */
    return (size_t)((c - 1) / 32) * 32 * 16 + (size_t)((k - 1) / 16) * 16 * C
         + (size_t)((c - 1) % 32) + (size_t)((k - 1) % 16) * 32;
}
static inline size_t out_idx_i16(int H, int ch, int h) {    /* output int32 elem, C2=4 */
    return ((size_t)(ch - 1) / 4) * (size_t)H * 4 + 4 * (size_t)(h - 1) + (ch - 1) % 4;
}

/* tf32 NPU layout index math. Feature cube C2=4 (4-byte atom); weight
 * (N/16,K/16,16,16) (== weight_tf32). The fp32 output reuses out_idx_i16 (cube
 * C2=4) — identical to the int16/bf16 fp32-out writer. */
static inline size_t feat_idx_tf32(int H, int ch, int h) {   /* input, C2=4 */
    return ((size_t)(ch - 1) / 4) * (size_t)H * 4 + 4 * (size_t)(h - 1) + (ch - 1) % 4;
}
static inline size_t wt_idx_tf32(int C, int k, int c) {      /* weight, (N/16,K/16,16,16) */
    return (size_t)((c - 1) / 16) * 16 * 16 + (size_t)((k - 1) / 16) * 16 * C
         + (size_t)((c - 1) % 16) + (size_t)((k - 1) % 16) * 16;
}

/* ── Planes <-> one channel group of a cube ───────────────────────────────────────
 *
 * A conv entry's tensors are PLANES, [C][H*W], and a cube group holds G channels of every
 * pixel side by side, so here the two ends are a transpose rather than two runs of the
 * same order. Moved one channel at a time, each pass writes one element of every G-wide
 * line, and a line is touched G times. These move G channels of a block of pixels at
 * once: a 16x16 byte or 8x8 halfword transpose (a perfect shuffle, zip rows j and j+n/2,
 * applied log2(n) times) and a four-way de-interleave for the int32 output cube, so each
 * line of the cube is written or read once.
 *
 * `planes[j]` is channel j's plane for j < nc; lanes nc..G-1 of the cube are written 0
 * on the way in and not read on the way out. `npix` pixels are moved, starting at pixel
 * 0 of each plane and of the group. Same bytes as the per-channel loop by construction. */
#if defined(__aarch64__)
#include <arm_neon.h>
static inline void rc_tr16x16_u8(uint8x16_t v[16])
{
#pragma GCC unroll 4
    for (int st = 0; st < 4; st++) {
        uint8x16_t n[16];
#pragma GCC unroll 8
        for (int j = 0; j < 8; j++) {
            n[2 * j]     = vzip1q_u8(v[j], v[j + 8]);
            n[2 * j + 1] = vzip2q_u8(v[j], v[j + 8]);
        }
#pragma GCC unroll 16
        for (int j = 0; j < 16; j++) v[j] = n[j];
    }
}
static inline void rc_tr8x8_u16(uint16x8_t v[8])
{
#pragma GCC unroll 3
    for (int st = 0; st < 3; st++) {
        uint16x8_t n[8];
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) {
            n[2 * j]     = vzip1q_u16(v[j], v[j + 4]);
            n[2 * j + 1] = vzip2q_u16(v[j], v[j + 4]);
        }
#pragma GCC unroll 8
        for (int j = 0; j < 8; j++) v[j] = n[j];
    }
}
#endif

/* planes (bytes) -> a G=16 byte cube group: dst[p*16 + j] = planes[j][p]. */
static inline void rc_planes_to_group16_u8(uint8_t *restrict dst, const uint8_t *const *planes,
                                           int nc, size_t npix)
{
    size_t p = 0;
#if defined(__aarch64__)
    for (; p + 16 <= npix; p += 16) {
        uint8x16_t v[16];
        for (int j = 0; j < 16; j++) v[j] = j < nc ? vld1q_u8(planes[j] + p) : vdupq_n_u8(0);
        rc_tr16x16_u8(v);
        for (int k = 0; k < 16; k++) vst1q_u8(dst + (p + k) * 16, v[k]);
    }
#endif
    for (; p < npix; p++)
        for (int j = 0; j < 16; j++) dst[p * 16 + j] = j < nc ? planes[j][p] : 0;
}

/* a G=16 byte cube group -> planes: planes[j][p] = src[p*16 + j], j < nc. */
static inline void rc_group16_to_planes_u8(uint8_t *const *planes, int nc,
                                           const uint8_t *restrict src, size_t npix)
{
    size_t p = 0;
#if defined(__aarch64__)
    for (; p + 16 <= npix; p += 16) {
        uint8x16_t v[16];
        for (int k = 0; k < 16; k++) v[k] = vld1q_u8(src + (p + k) * 16);
        rc_tr16x16_u8(v);
        for (int j = 0; j < nc; j++) vst1q_u8(planes[j] + p, v[j]);
    }
#endif
    for (; p < npix; p++)
        for (int j = 0; j < nc; j++) planes[j][p] = src[p * 16 + j];
}

/* planes (halfwords) -> a G=8 halfword cube group: dst[p*8 + j] = planes[j][p]. */
static inline void rc_planes_to_group8_u16(uint16_t *restrict dst, const uint16_t *const *planes,
                                           int nc, size_t npix)
{
    size_t p = 0;
#if defined(__aarch64__)
    for (; p + 8 <= npix; p += 8) {
        uint16x8_t v[8];
        for (int j = 0; j < 8; j++) v[j] = j < nc ? vld1q_u16(planes[j] + p) : vdupq_n_u16(0);
        rc_tr8x8_u16(v);
        for (int k = 0; k < 8; k++) vst1q_u16(dst + (p + k) * 8, v[k]);
    }
#endif
    for (; p < npix; p++)
        for (int j = 0; j < 8; j++) dst[p * 8 + j] = j < nc ? planes[j][p] : 0;
}

/* a G=8 halfword cube group -> planes: planes[j][p] = src[p*8 + j], j < nc. */
static inline void rc_group8_to_planes_u16(uint16_t *const *planes, int nc,
                                           const uint16_t *restrict src, size_t npix)
{
    size_t p = 0;
#if defined(__aarch64__)
    for (; p + 8 <= npix; p += 8) {
        uint16x8_t v[8];
        for (int k = 0; k < 8; k++) v[k] = vld1q_u16(src + (p + k) * 8);
        rc_tr8x8_u16(v);
        for (int j = 0; j < nc; j++) vst1q_u16(planes[j] + p, v[j]);
    }
#endif
    for (; p < npix; p++)
        for (int j = 0; j < nc; j++) planes[j][p] = src[p * 8 + j];
}

/* a G=4 int32 cube group -> planes: planes[j][p] = src[p*4 + j], j < nc. */
static inline void rc_group4_to_planes_i32(int32_t *const *planes, int nc,
                                           const int32_t *restrict src, size_t npix)
{
    size_t p = 0;
#if defined(__aarch64__)
    for (; p + 4 <= npix; p += 4) {
        int32x4x4_t v = vld4q_s32(src + p * 4);
        if (nc > 0) vst1q_s32(planes[0] + p, v.val[0]);
        if (nc > 1) vst1q_s32(planes[1] + p, v.val[1]);
        if (nc > 2) vst1q_s32(planes[2] + p, v.val[2]);
        if (nc > 3) vst1q_s32(planes[3] + p, v.val[3]);
    }
#endif
    for (; p < npix; p++)
        for (int j = 0; j < nc; j++) planes[j][p] = src[p * 4 + j];
}

#endif /* ROCKET_CUBE_H */
