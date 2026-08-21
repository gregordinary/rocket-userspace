// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_conv2d_rk3576.c — CONV_2D for the RK3576, behind the library's own entries.
 *
 * The register encoders in npu_regcmd_rk3576.c compute bit-exactly on this part, and
 * for a long time they were reachable only from a test harness: everything a caller
 * needs between a row-major tensor and a submit — the BO management, the operand
 * scatter, the tiling, the output de-scatter, and the submit-loop discipline the part
 * demands — lived in tests/rk3576_conv_gate.c. That made the reverse engineering done
 * and the chip unusable at the same time. This file is that layer.
 *
 * WHAT THE PART COMPUTES, and so what these entries expose:
 *
 *   DIRECT int8 -> int8, requantized ON CHIP by the DPU. That is the native shape of
 *   this datapath, not a convenience: the DPU's OUT_CVT applies the conv scale and
 *   writes a byte. rocket_conv2d_int8_rk3576() therefore takes the quant parameters
 *   and writes int8, where the RK3588's rocket_conv2d_int8() writes a raw int32
 *   accumulator. Same reasoning, and the same answer, as the matmul: a per-chip entry
 *   states the semantics the chip has rather than emulating another chip's.
 *
 *   DEPTHWISE int8 -> int8, likewise, and rocket_conv2d_dw_int8()'s public signature
 *   already IS that shape, so that entry dispatches here rather than refusing.
 *
 *   fp16 -> fp16, through the input-channel split. One fp16 task contracts exactly
 *   sixteen input channels, so an arbitrary channel count is ic/16 submits summed on
 *   the host.
 *
 * WHAT TILES, AND WHY IT MOVES THE ENVELOPE. Rows split through
 * rocket_rk3576_plan_rows() exactly as the harness drives them. Output channels split
 * here, and that is new: the resident weight slice is 32*ic*kh*kw bytes — one
 * output-channel GROUP, independent of oc — but the slice the part tolerates is a
 * function of how many groups the conv drives, 144 KiB at four or more and rising as
 * the count falls. So a conv the single-program emitter refuses for its slice computes
 * when its output channels are split into fewer groups per submit, at a cost of one
 * submit per tile. The planner below picks the largest tile whose group count still
 * admits the slice. tests/rk3576_conv_lib_gate.c is what says whether that holds: it
 * runs the emitter gate's own shape table through these entries, and the shapes the
 * emitter refuses for their weight slice are exactly the ones expected to compute here.
 *
 * WHAT IS REFUSED, and it is refused rather than approximated:
 *
 *   ic <= 4 takes the CNA's ARGB first-conv sub-encoding, whose register program is
 *   transcribed and gated but whose WEIGHT CUBE is not decoded. A capture carries a
 *   register program, not the memory it addresses, and this is the memory.
 *
 *   Dilation. conv_params_t carries the fields and no RK3576 shape has been run
 *   through them, so they are not claimed.
 *
 *   A weight zero point together with per-channel weight scales. A per-axis
 *   quantization is symmetric by construction, so that pair is not a shape any real
 *   model carries.
 *
 *   A shape whose weight slice does not fit even one output-channel group. The
 *   recourse there is an input-channel split, which the on-chip requant forecloses:
 *   int8 partials cannot be summed without quantizing each one. The int32-output
 *   writer is where that shape belongs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"
#include "rocket_chain.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define R76_HAVE_NEON 1
#endif

#define C2     16u      /* int8 feature/output channel atom */
#define C2F     8u      /* fp16 feature/output channel atom */

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
static void r76_c2_pack(int8_t *cube, const int8_t *const *sp, unsigned live, size_t px,
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
static void r76_c2_unpack(int8_t *const *dp, unsigned live, const int8_t *cube, size_t px)
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

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);
int weight_conv_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

/* ============================================================================
 * SECTION — shared plumbing
 * ==========================================================================*/
struct r76_conv_bos {
    rocket_bo in, w, coeff, out, rc;
    /* The DPU LUT's table-load program, when this conv's EW stage applies a table. It
     * is its own BO because it must be a SEPARATE TASK of the same job — the tables
     * live in the LUT RAM and a second submit can take a runtime-PM cycle in between —
     * and because it does not change from call to call. The dummy cube the load
     * program writes 16 bytes into sits past the program in the same BO. */
    rocket_bo lut;
    uint32_t  lut_ops;      /* words of program; 0 = no LUT on this conv */
    uint32_t  lut_scratch;  /* byte offset of the dummy cube inside lut  */
};

/* A LUT as an entry point hands it to the packing path: the window the emitter writes
 * into the conv program, and the two tables the load program bursts. */
struct r76_lut {
    lut_rk3576_t   w;
    const int16_t *le, *lo;
};

static void r76_conv_free(int fd, struct r76_conv_bos *b)
{
    if (b->lut.ptr)   rocket_bo_free(fd, &b->lut);
    if (b->rc.ptr)    rocket_bo_free(fd, &b->rc);
    if (b->out.ptr)   rocket_bo_free(fd, &b->out);
    if (b->coeff.ptr) rocket_bo_free(fd, &b->coeff);
    if (b->w.ptr)     rocket_bo_free(fd, &b->w);
    if (b->in.ptr)    rocket_bo_free(fd, &b->in);
    memset(b, 0, sizeof *b);
}

static int r76_is_this_chip(const char *entry)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (hw && hw->name && !strcmp(hw->name, "rk3576")) return 1;
    ROCKET_LOGE("%s emits the RK3576 geometry-register encoding and the active profile "
                "is %s\n", entry, hw && hw->name ? hw->name : "?");
    return 0;
}

/* The shape checks that are common to every precision here. `dw` selects the
 * depthwise contract (oc == ic). Returns 0, or a negative rocket_status. */
/* 1 when the descriptor asks for an output extent the symmetric formula does NOT give,
 * which on this part is how an asymmetric pad is expressed: the CNA takes the pad its last
 * window CONSUMES, derived from the extent and the leading pad, so TFLite's SAME at an even
 * input and stride two is a zero leading pad against a larger extent. A symmetric SAME
 * descriptor consumes a trailing pad too, so the trailing pad alone does not answer this —
 * what does is whether the caller set the extent itself. */
static int r76_desc_asym(const rocket_conv2d_desc *d)
{
    if (!d) return 0;
    return (d->oh && d->oh != rocket_conv_out_dim(d->ih, d->kh, d->stride_y, d->pad_top,
                                                  d->dil_y ? d->dil_y : 1)) ||
           (d->ow && d->ow != rocket_conv_out_dim(d->iw, d->kw, d->stride_x, d->pad_left,
                                                  d->dil_x ? d->dil_x : 1));
}

static int r76_conv_check(const char *entry, int fd, const rocket_conv2d_desc *d,
                          int dw, int argb_ok, unsigned *ow_out, unsigned *oh_out)
{
    int ow, oh;

    if (fd < 0 || !d) return ROCKET_E_SHAPE;
    if (!r76_is_this_chip(entry)) return ROCKET_E_UNSUPPORTED;
    if (d->ic <= 0 || d->oc <= 0 || d->ih <= 0 || d->iw <= 0 ||
        d->kh <= 0 || d->kw <= 0 || d->stride_y <= 0 || d->stride_x <= 0 ||
        d->pad_top < 0 || d->pad_left < 0)
        return ROCKET_E_SHAPE;
    if ((d->dil_y && d->dil_y != 1) || (d->dil_x && d->dil_x != 1)) {
        ROCKET_LOGE("%s: dilation is not claimed on this part — no RK3576 shape has "
                    "been run through the CONV_CON3 rate fields\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    /* The packed-image first conv is a different program with a different feature
     * buffer and a different weight cube, so the entries that own those say so by
     * passing argb_ok; anything else reaching here with four or fewer channels would
     * be driving the normal path's buffers at a channel count it cannot contract.
     *
     * UNLESS THE CALLER ASKED FOR THE DIRECT DATAPATH. The int8 direct cube is a
     * 32-channel MAC group at every count — rocket_rk3576_pad_ic(3) and _pad_ic(32) are
     * both 32 — so ic 3 needs no separate encoding, only the cube's own zero padding and
     * a zero-point fold over the LIVE taps. `ic` 5..31 already reach it this way; the
     * flag is what stops four or fewer being routed away from it. */
    if (d->ic <= 4 && !argb_ok && !d->direct_datapath) {
        ROCKET_LOGE("%s: %d input channels takes the CNA's ARGB first-conv "
                    "sub-encoding, which this entry does not own. Both precisions of it "
                    "run: rocket_conv2d_int8_rk3576() and rocket_conv2d_fp16_rk3576(). "
                    "rocket_conv2d_desc.direct_datapath runs it on the direct path "
                    "instead, which this entry does own\n", entry, d->ic);
        return ROCKET_E_UNSUPPORTED;
    }
    if (d->direct_datapath && dw) {
        ROCKET_LOGE("%s: direct_datapath asks for the direct encoding and this is a "
                    "depthwise conv, which has no packed-image form to be steered away "
                    "from — the flag would say nothing here, so it is refused rather "
                    "than dropped\n", entry);
        return ROCKET_E_SHAPE;
    }
    if (d->ic <= 4 && dw) {
        ROCKET_LOGE("%s: the first conv has no depthwise form — the channel fold leaves "
                    "nothing to be depthwise over\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    if (dw && d->oc != d->ic) {
        ROCKET_LOGE("%s: a depthwise conv has oc == ic (got oc=%d ic=%d)\n",
                    entry, d->oc, d->ic);
        return ROCKET_E_SHAPE;
    }
    if (d->pad_top > 255 || d->pad_left > 255) return ROCKET_E_SHAPE;

    ow = rocket_conv2d_ow(d);
    oh = rocket_conv2d_oh(d);
    if (ow <= 0 || oh <= 0) return ROCKET_E_SHAPE;
    /* AN ASYMMETRIC PAD IS AN OUTPUT EXTENT HERE. The CNA takes the pad its last window
     * consumes, derived from the extent and the leading pad, so a descriptor that asks for
     * more output than the symmetric formula gives is asking for a trailing pad — which is
     * exactly TFLite's SAME at an even input and stride two, and needs no border in the
     * caller's buffer.
     *
     * What bounds it is the KERNEL: a trailing pad of kh or more means an output row whose
     * whole window is pad, which is not a convolution anyone means, and the CNA's pad field
     * is eight bits. Refused rather than clamped. */
    if (d->oh || d->ow) {
        int ty = rocket_conv2d_trail_y(d), tx = rocket_conv2d_trail_x(d);
        if (ty >= d->kh || tx >= d->kw || ty > 255 || tx > 255) {
            ROCKET_LOGE("%s: an output extent of %dx%d over a %dx%d plane with a leading "
                        "pad of %d/%d needs a trailing pad of %d/%d, and a whole window of "
                        "pad is not a convolution (kernel %dx%d)\n",
                        entry, oh, ow, d->ih, d->iw, d->pad_top, d->pad_left,
                        ty, tx, d->kh, d->kw);
            return ROCKET_E_SHAPE;
        }
    }
    *ow_out = (unsigned)ow;
    *oh_out = (unsigned)oh;
    return ROCKET_OK;
}

/* Where one row task's own output lives, in bytes, so "did it write" can be asked of
 * exactly that task. The output cube is NC1HWC2, so a row run is one span per channel
 * group at that group's own surface offset — scanning only the first span would call a
 * task written on channel group 0 alone. */
struct r76_task_extent {
    size_t   base;        /* the surface's byte offset inside its BO    */
    unsigned groups;      /* output channel groups the surface carries  */
    size_t   group_bytes; /* one channel group's whole surface          */
    size_t   row_off;     /* this task's first row within a group       */
    size_t   span;        /* this task's rows within a group            */
};

/* A span still holding the stamp everywhere was never written.
 *
 * PER TASK, not per tile: one poisoned submit among several leaves its own rows stale
 * while its siblings are full, so a whole-tile check reads "something was written" and
 * passes the hole straight through to the caller. */
static int r76_task_wrote(const unsigned char *o, const struct r76_task_extent *e,
                          unsigned char stamp)
{
    unsigned g;
    for (g = 0; g < e->groups; g++) {
        const unsigned char *p = o + e->base + (size_t)g * e->group_bytes + e->row_off;
        size_t i;
        for (i = 0; i < e->span; i++)
            if (p[i] != stamp) return 1;
    }
    return 0;
}

/* Env-gated phase timing for the fp16 input-channel split. On-chip accumulation across
 * the slices would remove the READBACK and nothing else — the slice count, and so the
 * wide-output submits and the poisoning retries they carry, is set by the sixteen-channel
 * contraction either way. So the lever is only worth building if the readback is a real
 * share of the wall, and this is what says. ROCKET_RK3576_FP16_PROF=1 logs one line per
 * call, at ROCKET_LOG_INFO. */
struct r76_fp16_prof {
    int      on;
    unsigned slices;
    double   pack_us, stamp_us, submit_us, read_us;
};

static double r76_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static int r76_fp16_prof_on(void)
{
    const char *e = getenv("ROCKET_RK3576_FP16_PROF");
    return (e && *e && *e != '0');
}

/* The same question on the INT8 path, where it has a different answer.
 *
 * A layer's submit COUNT is the first thing to ask about its time, because the part's
 * per-submit floor is ~439 us — but a graph run says the count is already 1 on the
 * layers that cost the most, so the count is not what the time is. This splits the
 * rest: the operand scatters (feature cube, weight cube, coefficient group), the
 * submits, and the output de-scatter. ROCKET_RK3576_INT8_PROF=1 logs one line per
 * call, at ROCKET_LOG_INFO. */
struct r76_int8_prof {
    int      on;
    unsigned tiles, tasks;
    /* drain_us is INSIDE read_us: the PREP_BO that waits for the surface to land. */
    double   sums_us, in_us, w_us, coeff_us, gen_us, submit_us, read_us, free_us,
             drain_us;
};

static int r76_int8_prof_on(void)
{
    const char *e = getenv("ROCKET_RK3576_INT8_PROF");
    return (e && *e && *e != '0');
}

/* us since an arbitrary epoch when profiling is on, 0 when it is not — so the
 * instrumentation costs one predictable branch on the hot path. */
#define R76_PT(p) ((p).on ? r76_now_us() : 0.0)
#define R76_ACC(p, field, t0) do { if ((p).on) (p).field += r76_now_us() - (t0); } while (0)

/* How long to let a surface drain before calling it unwritten, in microseconds.
 * ROCKET_RK3576_DRAIN_US sets it; DEFAULT 0, because it is a MEASURED NEGATIVE — see
 * r76_task_wrote_late() below. */
static int r76_drain_us(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_DRAIN_US");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) : 0;
        if (cached < 0) cached = 0;
    }
    return cached;
}

/* How many times to redo a row task that wrote nothing. ROCKET_RK3576_TASK_ATTEMPTS.
 *
 * THE POWER CYCLE THE REDO WAITS FOR CLEARS THE POISONING ABOUT 87% OF THE TIME, so the
 * count is what makes the guard reliable, not a better cycle. Over twelve fp16 gate runs
 * the redo fired on 358 row tasks and the attempt that failed next was attempt 2 for 45
 * of them, 3 for 10, 4 for 2, 5 for 1 and 6 for none — every one of them recovering, and
 * each cycle CONFIRMED to have taken the domain to `suspended` first. At four attempts
 * that is about 0.6% of retried tasks returned as a device error, which is the rate the
 * conv gate saw. Eight matches the matmul path's R76_I32_TASK_ATTEMPTS, and costs
 * nothing on a task that succeeds. [HW sweep, H96 MAX M9] */
static unsigned r76_task_attempts(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_TASK_ATTEMPTS");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) : 8;
        if (cached < 1) cached = 1;
    }
    return (unsigned)cached;
}

/* Ask the surface again after a settle. A task whose DPU output element is wider than
 * one byte raises no DPU completion on this part, so the driver retires it on PC_DONE
 * plus a blind grace — and PC_DONE means the program counter finished ISSUING, not that
 * the writes have landed. The fence can therefore signal while the surface is still
 * draining, and a surface that arrives late is a completion-visibility fact rather than
 * the poisoning. It is asked BEFORE the power cycle because the cycle cannot fix it and
 * costs four orders of magnitude more.
 *
 * IT RESCUES NOTHING, and that is the result: at a 2 ms settle, 0 of 67 row tasks that
 * read unwritten had arrived by the time it looked again, while the power cycle behind
 * it recovered all 67. So a task that reads unwritten on this path really is unwritten,
 * the fence is not signalling ahead of the writes, and the 14% wall this costs when it
 * is on buys nothing. Off by default; the knob is kept because it is the instrument
 * that settled it. [HW sweep, H96 MAX M9] */
/* 1 if every task's own rows landed. `first_missing` names the earliest that did not,
 * which is what separates "the stream was poisoned" (none of them) from "the program
 * counter ran one task and stopped" (all but the first). */
static int r76_all_wrote(const unsigned char *o, const struct r76_task_extent *e,
                         unsigned ne, unsigned char stamp, unsigned *first_missing)
{
    unsigned i, missing = 0;
    int ok = 1;
    for (i = 0; i < ne; i++)
        if (!r76_task_wrote(o, &e[i], stamp)) {
            if (ok) missing = i;
            ok = 0;
        }
    if (first_missing) *first_missing = missing;
    return ok;
}

static int r76_task_wrote_late(int fd, struct r76_conv_bos *b,
                               const struct r76_task_extent *e, unsigned ne,
                               unsigned char stamp)
{
    struct timespec ts;
    int us = r76_drain_us(), wrote;

    if (!us) return 0;
    ts.tv_sec = us / 1000000;
    ts.tv_nsec = (long)(us % 1000000) * 1000L;
    nanosleep(&ts, NULL);

    rocket_bo_prep(fd, &b->out, 0, 0);
    wrote = r76_all_wrote((const unsigned char *)b->out.ptr, e, ne, stamp, NULL);
    rocket_bo_fini(fd, &b->out);
    return wrote;
}

/* Whether the row tasks of one output-channel tile go out as ONE submit.
 *
 * The per-submit floor on this part is ~439 us and a row-windowed convolution is one
 * submit per window, so a plane that plans into n windows pays it n times for work the
 * program counter could issue back to back. The row tasks of one tile are independent by
 * construction — each writes its own rows of the same surface, reads its own window of
 * the same feature cube, and the weight and coefficient buffers do not change across
 * them — so concatenating their programs into one regcmd stream is arithmetically sound.
 *
 * A CONCATENATED STREAM IN ONE DRM TASK DOES NOT RUN, and the mechanism is decoded. A
 * drm task descriptor carries ONE PC program however many the stream holds: the driver
 * programs PC_TASK_CON with TASK_NUMBER = 1 per descriptor, so the program counter
 * executes the first task and stops with the rest of the stream unexecuted. Measured
 * with a per-task write check that names the earliest task missing: at 3 tasks and at 6,
 * task 0 lands and task 1 is the first missing, on every one of eight attempts with the
 * power domain confirmed cycled between them — deterministic, and not the poisoning.
 *
 * WHAT COLLECTS IT IS THE JOINT LAYOUT CONTRACT the RK3588 already uses: userspace lays
 * the n programs out contiguously at a fixed stride and rewrites each trailer to point
 * the PC at the next (rocket_chain.c), submits them as n drm task descriptors with
 * DRM_ROCKET_JOB_BATCHED, and the kernel programs TASK_NUMBER = n so the PC streams them
 * from ONE kick and raises ONE completion. That is what makes it a lever rather than an
 * ioctl saving — the completion poll IS the ~439 us floor, and n row tasks then pay it
 * once. The kernel half is patches/rk3576/npu/0015 (extensible submit descriptors) and
 * 0016 (the flag); rocket_batched_submit_supported() refuses to self-chain without it,
 * because a chained layout run down the per-task path stalls.
 *
 * IT IS ON BY DEFAULT, and the reason it was not is a measurement taken on the wrong
 * workload. Over the per-op conv gate it is worth about 3%, because most shapes there plan
 * into one or two row tasks and there is no per-call floor left to remove. On a GRAPH whose
 * layers are already cheap it is worth 9-11% — MobileNetV1-224 resident with the cube chain
 * goes 11.8-12.2 ms to 10.5-11.1 on the same 40 tasks, 40 submits to 29 — and the saving is
 * exactly `(n-1) * 439 us` per call, so it can only grow as the host side gets cheaper.
 * Every gate passes with it on and the graph's logits are byte-identical.
 *
 * ROCKET_RK3576_BATCH_TASKS=0 turns it off. A kernel without 0015-0016, or with
 * `rocket_batch_submit=0`, takes the per-task path instead: that is the DEFAULT falling
 * back and is logged at debug, where asking for it explicitly is a warning.
 * [HW sweep, H96 MAX M9] */
static int r76_batch_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_BATCH_TASKS");
        int asked = (e && *e);
        int want = asked ? (*e != '0') : 1;
        if (want && !rocket_batched_submit_supported()) {
            if (asked)
                ROCKET_LOGW("ROCKET_RK3576_BATCH_TASKS=1 but this kernel does not honor "
                            "DRM_ROCKET_JOB_BATCHED (needs patches/rk3576/npu/0015-0016; "
                            "the driver must report >= 1.1, and rocket_batch_submit must "
                            "not be 0). One submit per row task.\n");
            else
                ROCKET_LOGD("chained row tasks are off: this kernel does not honor "
                            "DRM_ROCKET_JOB_BATCHED. One submit per row task, which costs "
                            "~439 us per extra row window.\n");
            want = 0;
        }
        cached = want;
    }
    return cached;
}

/* The row tasks one chained job may carry. Every plan on this part is bounded by the
 * output height, so this is far above anything the row planner emits; a shape that
 * somehow exceeded it runs one submit per task instead. */
#define R76_MAX_CHAIN_TASKS 512u

/* Lay `ne` programs of `task_ops` words each out in the regcmd BO as ONE chained stream
 * and describe them as ne drm tasks. The host buffer holds each program at the fixed
 * RK3576_CONV_TASK_OPS stride (the generator's upper bound); the BO gets them at the
 * chain's own even-word stride with each trailer rewritten to point the PC at the next,
 * and the last task's forward link cleared. The kernel then programs TASK_NUMBER = ne so
 * the PC streams all of them from one kick and raises one completion. */
static void r76_chain_stream(struct r76_conv_bos *b, rocket_task_desc *td,
                             const uint64_t *ops, uint32_t task_ops, unsigned ne)
{
    unsigned t;
    for (t = 0; t < ne; t++)
        rkt_chain_pack(1, &b->rc, td, (int)t,
                       ops + (size_t)t * RK3576_CONV_TASK_OPS, task_ops, 0);
    rkt_chain_seal(1, &b->rc, (int)ne, task_ops);
}

/* Submit one tile's row tasks and satisfy ourselves that each wrote. `ops` holds `ne`
 * programs of `task_ops` words; they go out as one chained job when batching is on and
 * as one submit each when it is not (ne is then 1). The retry is what covers the
 * poisoning an int32-output job leaves behind — it crosses calls and processes, so a
 * conv inherits it from whatever ran before — and the idle in front of the redo is the
 * NPU power domain cycling rather than a settling time. */
/* One word past the longest program, so the unchained multi-task layout puts each row
 * task at its own offset in the regcmd BO. Matches the chain's own even-word stride. */
#define R76_TASK_SLOT_WORDS  (RK3576_CONV_TASK_OPS + 1u)

static int r76_submit_ops(int fd, struct r76_conv_bos *b, const uint64_t *ops,
                          uint32_t task_ops, const uint32_t *in_h, unsigned n_in,
                          const uint32_t *out_h, const struct r76_task_extent *e,
                          unsigned ne, int chained, unsigned char stamp,
                          uint32_t job_flags, const char *entry)
{
    unsigned attempt, attempts = r76_task_attempts(), missing = 0;
    int cycled = 0, cycles_confirmed = 0;
    /* The LUT's table load is task 0 of the job, ahead of every row task. */
    unsigned lead = b->lut_ops ? 1u : 0u;
    rocket_task_desc td[R76_MAX_CHAIN_TASKS + 1u];

    if (ne > R76_MAX_CHAIN_TASKS) {
        ROCKET_LOGE("%s: %u tasks exceeds the %u this path lays out\n",
                    entry, ne, (unsigned)R76_MAX_CHAIN_TASKS);
        return ROCKET_E_SHAPE;
    }

    for (attempt = 0; attempt < attempts; attempt++) {
        int srv;

        rocket_bo_prep(fd, &b->rc, 1, 0);
        if (chained) {
            r76_chain_stream(b, td, ops, task_ops, ne);
        } else if (ne > 1u || lead) {
            /* Every program at its own slot, described as its own drm task. One PC
             * program per descriptor is the part's rule either way; what chaining adds
             * on top is a single hardware kick, and a foreign leading BO is not a shape
             * the chain rewrite lays out. */
            uint64_t *dst = (uint64_t *)b->rc.ptr;
            unsigned t;
            for (t = 0; t < ne; t++) {
                memcpy(dst + (size_t)t * R76_TASK_SLOT_WORDS,
                       ops + (size_t)t * RK3576_CONV_TASK_OPS,
                       (size_t)task_ops * sizeof(uint64_t));
                td[lead + t].regcmd = b->rc.dma_address +
                    (uint32_t)((size_t)t * R76_TASK_SLOT_WORDS * sizeof(uint64_t));
                td[lead + t].regcmd_count = task_ops;
            }
        } else {
            memcpy(b->rc.ptr, ops, (size_t)task_ops * sizeof(uint64_t));
        }
        rocket_bo_fini(fd, &b->rc);
        if (lead) {
            td[0].regcmd = b->lut.dma_address;
            td[0].regcmd_count = b->lut_ops;
        }

        srv = chained
            ? rocket_submit_tasks_flags(fd, td, ne, in_h, n_in, out_h, 1,
                                        job_flags | ROCKET_JOB_BATCHED)
            : (ne > 1u || lead)
            ? rocket_submit_tasks_flags(fd, td, lead + ne, in_h, n_in, out_h, 1,
                                        job_flags)
            : rocket_submit_matmul_flags(fd, &b->rc, task_ops, in_h, n_in, out_h, 1,
                                         job_flags);
        if (srv != 0) {
            ROCKET_LOGE("%s: submit failed\n", entry);
            return ROCKET_E_DEVICE;
        }
        if (rocket_bo_prep(fd, &b->out, 0, 2000000000ull) < 0) {
            ROCKET_LOGE("%s: PREP_BO on the output timed out\n", entry);
            return ROCKET_E_DEVICE;
        }
        if (!stamp) { rocket_bo_fini(fd, &b->out); return ROCKET_OK; }
        {
            /* EVERY task in the stream, not the stream as a whole: one dead task among
             * several leaves its own rows stale while its siblings land, and a check
             * that asks "did anything change" passes that hole straight to the caller. */
            int wrote = r76_all_wrote((const unsigned char *)b->out.ptr, e, ne, stamp,
                                      &missing);
            rocket_bo_fini(fd, &b->out);
            if (wrote) return ROCKET_OK;
        }
        if (r76_task_wrote_late(fd, b, e, ne, stamp)) {
            ROCKET_LOGD("%s: the surface arrived after the fence, not with it — a "
                        "drain, not the poisoning (attempt %u)\n", entry, attempt + 1u);
            return ROCKET_OK;
        }
        ROCKET_LOGD("%s: of %u row task(s) in this submit the first that wrote nothing "
                    "is %u, on attempt %u; cycling the power domain and redoing it\n",
                    entry, ne, missing, attempt + 1u);
        cycled++;
        cycles_confirmed += rocket_rk3576_power_idle();
    }
    /* Which of the two failures this was, rather than only that it failed: a redo after
     * a CONFIRMED domain collapse that still wrote nothing is not the poisoning, and a
     * redo after an unconfirmed one never had the guard the retry assumes. */
    ROCKET_LOGE("%s: a row task wrote nothing over %u attempts (%d power cycles, "
                "%d of them confirmed to reach suspended)\n",
                entry, attempts, cycled, cycles_confirmed);
    return ROCKET_E_DEVICE;
}

static int r76_submit_task(int fd, struct r76_conv_bos *b, const conv_params_t *q,
                           const uint64_t *ops, const uint32_t *in_h, unsigned n_in,
                           const uint32_t *out_h, const struct r76_task_extent *e,
                           unsigned char stamp, uint32_t job_flags, const char *entry)
{
    return r76_submit_ops(fd, b, ops, q->task_count, in_h, n_in, out_h, e, 1, 0, stamp,
                          job_flags, entry);
}

/* ============================================================================
 * SECTION — the output-channel tile
 *
 * r76_weight_slice_cap() in the emitter is the measured table: a resident weight slice
 * of 144 KiB leaves every output-channel group exact at four groups, 148 KiB at three,
 * 156 KiB at two, and a single group is governed by the CBUF pool alone. The loss is
 * GRADED — the leading groups stay bit-exact and the trailing ones come back wrong — so
 * driving fewer groups per submit is what buys the slice back.
 *
 * This is the inverse of that table: given the slice a shape needs, the most output
 * channels one submit may drive. Splitting there is exact by construction — each tile
 * is an independent convolution over its own channels — and costs one submit per tile.
 * ==========================================================================*/
static unsigned r76_conv_oc_tile(unsigned icreg, unsigned kh, unsigned kw, unsigned oc)
{
    size_t slice = (size_t)32u * icreg * kh * kw;
    unsigned groups;

    if (slice <= 144u * 1024u)      return oc;      /* no group-count constraint */
    else if (slice <= 148u * 1024u) groups = 3u;
    else if (slice <= 156u * 1024u) groups = 2u;
    else                            groups = 1u;    /* the pool check governs */

    return groups * 32u < oc ? groups * 32u : oc;
}

/* ============================================================================
 * SECTION — the int8 convolution, direct and depthwise
 * ==========================================================================*/

/* The zero-point algebra, once, so both paths agree on it.
 *
 * The hardware computes  acc = sum(x_s*w_s) + A + B*sum(x_s)  over ALL taps, with an
 * out-of-bounds tap substituting the border constant, which the emitter programs to the
 * input zero point — so a pad tap's TRUE value is zero and needs no special case.
 *
 * The caller's convolution is  y = sum((x_s - in_zp)*(w_s - w_zp)) + bias, so
 *
 *     B[oc] = -w_zp                       (the DPU ADDS the B term)
 *     A[oc] = bias[oc] - in_zp*sum_w[oc] + in_zp*w_zp*N
 *
 * with sum_w the sum of that output channel's whole filter and N its tap count. Both
 * corrections are pixel-independent, which is what makes them foldable at all. */
/* The output channel a tile's slot j actually carries. Identity unless a per-channel
 * requant has reordered them; see r76_sort_by_scale. */
static unsigned r76_oc_of(const unsigned *perm, unsigned i)
{
    return perm ? perm[i] : i;
}

/*
 * Sort the output channels by their weight scale, ascending.
 *
 * Every output-channel TILE is its own task and so carries its own OUT_CVT shift, and
 * the C ramp inside a tile only has to span THAT tile's range of scales. Left in model
 * order a tile sees the whole layer's spread; sorted, each tile sees roughly the
 * spread's n-th root. That is the difference between a usable per-axis requant and a
 * useless one on a layer with both a wide scale spread and a large fan-in, where the
 * int32 clamp already caps the largest C at a couple of hundred.
 *
 * The permutation is a relabelling of the output axis and nothing else: the weight
 * cube, the bias fold, the C ramp and the de-scatter all read the same slot, so the
 * caller's `out` comes back in the caller's channel order.
 *
 * An insertion sort, because oc is at most a few thousand and this runs once.
 */
static void r76_sort_by_scale(unsigned *perm, unsigned oc, const float *w_scale)
{
    unsigned i, j;
    for (i = 0; i < oc; i++) perm[i] = i;
    for (i = 1; i < oc; i++) {
        unsigned v = perm[i];
        float    s = w_scale[v];
        j = i;
        while (j > 0 && w_scale[perm[j - 1]] > s) { perm[j] = perm[j - 1]; j--; }
        perm[j] = v;
    }
}

static void r76_fold_coeff(int32_t *A, const int32_t *bias, unsigned oc0,
                           unsigned tile_oc, const int64_t *sum_w, int in_zp, int w_zp,
                           unsigned taps, const unsigned *perm)
{
    unsigned j;
    for (j = 0; j < tile_oc; j++) {
        unsigned c = r76_oc_of(perm, oc0 + j);
        int64_t a = bias ? (int64_t)bias[c] : 0;
        a -= (int64_t)in_zp * sum_w[c];
        a += (int64_t)in_zp * w_zp * taps;
        A[j] = (int32_t)a;
    }
}

/*
 * The per-output-channel requant plan.
 *
 * The DPU's epilogue is `(acc + A[oc]) * C[oc]` in saturating int32, then ONE
 * `(v*MUL)>>SHIFT` for the whole task. So a per-axis weight quantization rides on
 * C, and the job here is to pick the one (MUL, SHIFT) and the C ramp that together
 * approximate every channel's own `in_scale*w_scale[oc]/out_scale` as closely as the
 * two hardware bounds allow:
 *
 *   - C is an integer, so channel oc's gain resolution is 0.5/C[oc]. Bigger is better.
 *   - `(acc + A) * C` saturates at INT32_MAX, so C[oc] is capped by that channel's own
 *     worst-case accumulator. Bigger is not free.
 *
 * The cap is computed from the ACTUAL weights rather than from the int8 envelope,
 * because 127*sum|w| over a real filter is one to two orders of magnitude below
 * ic*kh*kw*127*127 and the difference is most of the available precision.
 *
 * Returns the worst-case relative gain error over the channels in `*max_rel_err`.
 */
static int r76_plan_perchannel(const char *entry, unsigned oc0, unsigned tile_oc,
                               unsigned ocreg, const int32_t *A,
                               const int64_t *sum_abs_w, float in_scale,
                               const float *w_scale, float out_scale,
                               const unsigned *perm,
                               int16_t *C, float *base_scale, double *max_rel_err)
{
    double best_base = 0.0;
    unsigned j;

    /* The tightest base gain: every channel must reach its own scale with a C that
     * neither exceeds the int16 field nor saturates the int32 product. */
    for (j = 0; j < tile_oc; j++) {
        unsigned oc_j = r76_oc_of(perm, oc0 + j);
        double cs = (double)in_scale * (double)w_scale[oc_j] / (double)out_scale;
        /* |acc| <= 128*sum|w| (the input is signed int8), and A rides with it. */
        double bound = 128.0 * (double)sum_abs_w[oc_j] + fabs((double)A[j]) + 1.0;
        double cmax = (double)INT32_MAX / bound;
        double need;
        if (!(cs > 0.0)) {
            ROCKET_LOGE("%s: w_scale[%u] is %g — every per-channel scale must be "
                        "positive\n", entry, oc_j, (double)w_scale[oc_j]);
            return -1;
        }
        if (cmax > 32767.0) cmax = 32767.0;
        if (cmax < 1.0)     cmax = 1.0;
        need = cs / cmax;
        if (need > best_base) best_base = need;
    }
    if (!(best_base > 0.0)) return -1;

    /* Quantize the base into the pair the emitter will actually program, then read it
     * back: the C ramp has to be built against the gain the hardware gets, not the one
     * the planner asked for. */
    {
        unsigned mul, shift;
        double base_actual;
        rocket_rk3576_requant_params((float)best_base, &mul, &shift);
        base_actual = (double)mul / (double)((uint64_t)1 << shift);
        *base_scale = (float)best_base;
        *max_rel_err = 0.0;
        for (j = 0; j < ocreg; j++) {
            double cs, want, err;
            long long c;
            if (j >= tile_oc) { C[j] = 1; continue; }   /* a padded channel computes nothing */
            cs   = (double)in_scale * (double)w_scale[r76_oc_of(perm, oc0 + j)] /
                   (double)out_scale;
            want = cs / base_actual;
            c = (long long)(want + 0.5);
            if (c < 1)     c = 1;
            if (c > 32767) c = 32767;
            C[j] = (int16_t)c;
            err = fabs((double)c * base_actual - cs) / cs;
            if (err > *max_rel_err) *max_rel_err = err;
        }
    }
    return 0;
}

/*
 * Pick the output-channel tile on the per-channel path.
 *
 * Everywhere else the tile is a CBUF-fit decision. Here it is also an ACCURACY one,
 * because a tile is one task and a task carries one OUT_CVT shift: the C ramp inside a
 * tile only spans that tile's range of scales, so halving the tile roughly halves the
 * spread the ramp has to cover and the worst channel's gain resolution improves with
 * it. Measured on the part at ic=128 oc=128 with a 100x spread: 26.6 counts of
 * deviation from an exact per-axis requant in one tile, 2.7 at 64 channels, 1.0 at 32.
 *
 * The cost is submits — one row-task set per tile at the part's ~439 us floor — so
 * this takes the LARGEST tile that meets the error target rather than the smallest
 * tile that would fit. The error is predicted from the plan alone, with no submits.
 *
 * ROCKET_RK3576_PC_MAX_ERR sets the target (default 1%); ROCKET_RK3576_PC_OC_TILE
 * forces a tile and skips the search.
 */
static double r76_pc_err_at(unsigned OC, unsigned tile, const unsigned *perm,
                            const int64_t *sum_abs_w, const int32_t *bias,
                            const int64_t *sum_w, int in_zp, int w_zp, unsigned taps,
                            float in_scale, const float *w_scale, float out_scale)
{
    double worst = 0.0;
    unsigned oc0;
    for (oc0 = 0; oc0 < OC; oc0 += tile) {
        unsigned n = OC - oc0 < tile ? OC - oc0 : tile, j;
        double best_base = 0.0, base_actual;
        unsigned mul, shift;
        for (j = 0; j < n; j++) {
            unsigned c = r76_oc_of(perm, oc0 + j);
            double cs = (double)in_scale * (double)w_scale[c] / (double)out_scale;
            double A  = (double)(bias ? bias[c] : 0)
                      - (double)in_zp * (double)sum_w[c]
                      + (double)in_zp * w_zp * taps;
            double bound = 128.0 * (double)sum_abs_w[c] + fabs(A) + 1.0;
            double cmax = (double)INT32_MAX / bound;
            double need;
            if (!(cs > 0.0)) return 1e30;
            if (cmax > 32767.0) cmax = 32767.0;
            if (cmax < 1.0)     cmax = 1.0;
            need = cs / cmax;
            if (need > best_base) best_base = need;
        }
        if (!(best_base > 0.0)) return 1e30;
        rocket_rk3576_requant_params((float)best_base, &mul, &shift);
        base_actual = (double)mul / (double)((uint64_t)1 << shift);
        for (j = 0; j < n; j++) {
            unsigned c = r76_oc_of(perm, oc0 + j);
            double cs = (double)in_scale * (double)w_scale[c] / (double)out_scale;
            long long v = (long long)(cs / base_actual + 0.5);
            double err;
            if (v < 1)     v = 1;
            if (v > 32767) v = 32767;
            err = fabs((double)v * base_actual - cs) / cs;
            if (err > worst) worst = err;
        }
    }
    return worst;
}

static unsigned r76_pc_oc_tile(unsigned OC, unsigned cbuf_tile, const unsigned *perm,
                               const int64_t *sum_abs_w, const int32_t *bias,
                               const int64_t *sum_w, int in_zp, int w_zp, unsigned taps,
                               float in_scale, const float *w_scale, float out_scale)
{
    const char *e = getenv("ROCKET_RK3576_PC_OC_TILE");
    double target = 0.01;
    unsigned tile;

    if (e && *e) {
        long v = strtol(e, NULL, 0);
        if (v > 0) return (unsigned)v < cbuf_tile ? (unsigned)v : cbuf_tile;
    }
    e = getenv("ROCKET_RK3576_PC_MAX_ERR");
    if (e && *e) {
        double v = strtod(e, NULL);
        if (v > 0.0) target = v;
    }
    /* Largest first, so the search stops at the fewest submits that will do. The floor
     * is the 32-channel MAC group; below it a tile is not a shape the emitter takes. */
    for (tile = cbuf_tile; ; tile = tile / 2u > 32u ? tile / 2u : 32u) {
        if (r76_pc_err_at(OC, tile, perm, sum_abs_w, bias, sum_w, in_zp, w_zp, taps,
                          in_scale, w_scale, out_scale) <= target)
            return tile;
        if (tile <= 32u) return 32u;   /* nothing narrower to try */
    }
}

/* ============================================================================
 * SECTION — the packed weight handle
 *
 * Everything a convolution on this part re-derives from the WEIGHTS ALONE: the
 * per-output-channel filter sums, the coefficient group, the weight cube, and on the
 * per-axis path the scale sort and the output-channel tile the accuracy target picks.
 * None of it depends on the input, and on a real graph it is a quarter of the wall — two
 * loops over a million weight elements are what a MobileNet spends its host time in, not
 * the submit. So a caller that runs the same layer repeatedly packs once.
 *
 * ONE CODE PATH SERVES BOTH. A transient call builds a handle, packs each output-channel
 * tile as the tile loop reaches it and drops it again; a resident caller packs every tile
 * up front and keeps them. The arithmetic is identical by construction rather than by a
 * second implementation agreeing with the first, which is the only way to add an entry
 * point without putting "bit-exact" back in question.
 *
 * TWO THINGS THE HANDLE MUST CARRY, and both are properties of this part rather than
 * bookkeeping:
 *
 *   THE TILE PLAN IT WAS PACKED FOR. The weight cube's group count follows the TILE, not
 *   the layer, so a cube packed for one tiling is not a cube for another. On the per-axis
 *   path the tile is an ACCURACY choice made from the weights and the scales — a tile is
 *   one task and a task carries one OUT_CVT shift — so it is decided at pack time and
 *   frozen with the cubes it produced.
 *
 *   THE fd. A BO belongs to the file that created it and its IOVA is per-fd, so a handle
 *   packed on one fd addresses nothing on another. It owns one and refuses a foreign one.
 *
 * The QUANT CONTRACT is frozen for the same reason: both zero points and all three scales
 * are folded into the coefficient group at pack time. A layer whose quantization changes
 * needs a new handle, which is what a graph does anyway.
 * ==========================================================================*/

/* One output-channel tile's operands. `out` is scratch rather than a weight product, and
 * it is held here because its size is a function of the frozen geometry — reusing it costs
 * a resident caller one allocation instead of one per inference. Reuse is safe whichever
 * way the write check is configured: the sentinel is re-stamped before every submit, and
 * with the sentinel off there is no check to mislead. */
struct r76_int8_wtile {
    rocket_bo w, coeff, out;
    unsigned  oc0, tile_oc, ocreg;
    float     base_scale;   /* the gain this tile's OUT_CVT programs */
    /* This surface already carries the sentinel and needs no bracket of its own.
     * Set when the PREVIOUS call re-stamped it inside the de-scatter's PREP/FINI pair;
     * see "THE STAMP RIDES THE DE-SCATTER'S BRACKET" in r76_int8_exec(). */
    int       stamped;
    /* THIS SURFACE CARRIES THE GROUPS A DIRECT CONSUMER WALKS, NOT ONLY THE ONES THE
     * PROGRAM WRITES. Set on a depthwise tile whose round-32 count is deeper than its
     * register count: the extra group is filled with the output zero point once, at pack
     * time, and nothing ever overwrites it — the program is told the raw count and does not
     * address it, and the sentinel covers only the written groups. See r76_wtile_pack(). */
    int       tail_zp;
};

struct rocket_conv2d_int8_weights_rk3576 {
    int      fd;            /* the file every BO below belongs to */
    int      resident;      /* 1 = a caller packed these and holds them */
    rocket_conv2d_desc d;
    int      dw;
    float    in_scale, w_scale, out_scale;
    int      in_zp, w_zp, out_zp;
    /* Geometry, so a call derives none of it. */
    unsigned IC, OC, IH, IW, KH, KW, SY, SX, PT, PL;
    unsigned ow, oh, icreg, icpad, surf_elems, taps;
    unsigned oc_tile, ntile, max_tasks;
    size_t   in_bytes;
    int      batch;
    /* A DPU LUT applied in the EW stage. `multi` is the layout that follows from it: a
     * job's row tasks all have to be generated before it is submitted, because the table
     * load is task 0 and one load has to serve them all. Chaining wants the same layout
     * and takes it further, so `multi` is the union of the two. */
    int      multi;
    int      has_lut;
    lut_rk3576_t lut;
    int16_t *lut_le, *lut_lo;   /* RK3576_LUT_ENTRIES each, owned */
    rocket_bo lut_bo;
    uint32_t  lut_ops, lut_scratch;
    /* What a tile is packed FROM. Kept because a transient call packs inside the tile
     * loop, and COPIED rather than borrowed so a handle never outlives a caller's array. */
    int64_t  *sum_w, *sum_abs_w;
    int32_t  *bias;         /* NULL when the caller passed none */
    float    *w_scale_oc;   /* NULL unless the requant is per-axis */
    unsigned *perm;         /* NULL unless a per-axis DIRECT conv sorted the channels */
    struct r76_int8_wtile *tile;
    double   worst_rel_err;
    /* CUBE I/O. `cube_in` borrows a producer's output surface as this handle's feature
     * cube — `src` is a COPY of that BO's descriptor and this handle never frees it — and
     * `cube_out` leaves the output where the DPU wrote it. Both are layout decisions and
     * neither changes the arithmetic: the surfaces are the same bytes the row-major path
     * transposes at each end. */
    int      cube_in, cube_out;
    rocket_bo src;
    size_t   src_off;       /* the input cube's byte offset inside `src`             */
    /* The borrowed cube's channel-group stride, when it is NOT this plane. A depthwise
     * producer writes round4(ow*oh) per group where a direct one writes ow*oh exactly,
     * and the CNA's DDR group jump takes the padded quantity verbatim [HW sweep, H96
     * MAX M9, tests/rk3576_surf_stride.c]. Zero on every other path, which is what
     * keeps their programs byte-identical. */
    unsigned src_surf_elems;
    /* An EXTERNAL output surface: a caller's buffer this handle writes its own slice of.
     * Borrowed like `src` and never freed here; `out_ext.ptr` is what says it is in use,
     * and while it is, this handle's own tile[0].out is released. */
    rocket_bo out_ext;
    size_t   out_off;
    /* THIS SURFACE IS READ BY SOMETHING OUTSIDE ANY CHAIN THE HANDLE IS IN, later in the
     * same inference — a producer whose output two consumers read, where only one of them
     * is the layer after it. A chain re-stamps every interior layer's surface in its verify
     * bracket, because it will be rewritten before anyone reads it; that is exactly what is
     * not true here, and stamping it would replace this layer's output with the sentinel
     * invisibly. Set, the stamp goes on at the start of this handle's NEXT call instead —
     * the same place cube_out already puts it. */
    int      shared_out;
    /* Per-call scratch, held across calls when resident. */
    rocket_bo in, rc;
    uint64_t *ops;
    struct r76_task_extent *ext;
    rocket_rk3576_row_task *plan;
};

typedef struct rocket_conv2d_int8_weights_rk3576 r76_w;

/*
 * WHERE A TILE'S SURFACE LIVES, in one place — because it is two things now. A handle
 * normally owns one BO per output-channel tile; told to write into a caller's buffer it
 * owns none, and every base address, cache bracket, sentinel fill and write check has to
 * follow the same pair. An external destination is single-tile by construction, so only
 * tile 0 can ever be the borrowed one.
 *
 * The offset is a plain address on this part: a program whose output base moves by G
 * channel groups computes the same surface, writes only its own slice, and composes with
 * the row-task offset already added to the same register.
 * [HW sweep, H96 MAX M9, tests/rk3576_offset_cube.c]
 */
static rocket_bo *r76_surf(r76_w *h, unsigned ti, size_t *off)
{
    if (h->out_ext.ptr && !ti) { *off = h->out_off; return &h->out_ext; }
    *off = 0;
    return &h->tile[ti].out;
}

/* The same, for a const handle that only needs the address the program is emitted with. */
static uint64_t r76_surf_dma(const r76_w *h, unsigned ti)
{
    if (h->out_ext.ptr && !ti) return h->out_ext.dma_address + h->out_off;
    return h->tile[ti].out.dma_address;
}

/* The feature cube's base, borrowed or the handle's own. */
static uint64_t r76_feat_dma(const r76_w *h)
{
    return h->cube_in ? h->src.dma_address + h->src_off : h->in.dma_address;
}

/* Release one tile's operands. Called per tile by a transient run and once per tile by the
 * handle teardown, so it must tolerate a tile that was never packed. */
static void r76_wtile_drop(r76_w *h, unsigned t)
{
    struct r76_int8_wtile *s = &h->tile[t];
    if (s->w.ptr)     rocket_bo_free(h->fd, &s->w);
    if (s->coeff.ptr) rocket_bo_free(h->fd, &s->coeff);
    if (s->out.ptr)   rocket_bo_free(h->fd, &s->out);
    memset(&s->w, 0, sizeof s->w);
    memset(&s->coeff, 0, sizeof s->coeff);
    memset(&s->out, 0, sizeof s->out);
    s->stamped = 0;
}

static void r76_w_free(r76_w *h)
{
    if (!h) return;
    if (h->tile) {
        unsigned t;
        for (t = 0; t < h->ntile; t++) r76_wtile_drop(h, t);
        free(h->tile);
    }
    if (h->lut_bo.ptr) rocket_bo_free(h->fd, &h->lut_bo);
    if (h->rc.ptr) rocket_bo_free(h->fd, &h->rc);
    if (h->in.ptr) rocket_bo_free(h->fd, &h->in);
    free(h->sum_w); free(h->sum_abs_w); free(h->bias); free(h->w_scale_oc);
    free(h->perm); free(h->lut_le); free(h->lut_lo);
    free(h->ops); free(h->ext); free(h->plan);
    free(h);
}

/*
 * Validate the shape and the quant contract, then derive the geometry.
 *
 * This is every check both entry points owe a caller, in one place — a resident pack that
 * validated less than the transient call would be a way to reach the hardware with a shape
 * the library refuses.
 */
static int r76_w_prepare(const char *entry, int fd, const rocket_conv2d_desc *d, int dw,
                         const int32_t *bias, float in_scale, float w_scale,
                         const float *w_scale_oc, float out_scale,
                         int in_zp, int w_zp, int out_zp,
                         const struct r76_lut *lut, r76_w **out_h)
{
    r76_w *h;
    unsigned ow, oh;
    int rc;

    *out_h = NULL;

    rc = r76_conv_check(entry, fd, d, dw, 0, &ow, &oh);
    if (rc != ROCKET_OK) return rc;
    if (w_scale_oc && w_zp) {
        ROCKET_LOGE("%s: a per-axis weight quantization is symmetric by construction; "
                    "a non-zero weight zero point with per-channel scales is not a "
                    "shape this path models\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    if (!(in_scale > 0.0f) || !(out_scale > 0.0f) ||
        (!w_scale_oc && !(w_scale > 0.0f))) {
        ROCKET_LOGE("%s: the quant scales must be positive — the DPU's OUT_CVT gates "
                    "the whole BS stage off at zero and writes an empty surface\n", entry);
        return ROCKET_E_SHAPE;
    }
    if (in_zp < -128 || in_zp > 127 || w_zp < -128 || w_zp > 127 ||
        out_zp < -128 || out_zp > 127)
        return ROCKET_E_SHAPE;
    /* A DEPTHWISE WEIGHT ZERO POINT RIDES IN THE CUBE, not in the coefficient group.
     * That group really has no B field and the correction really is not a per-channel
     * constant, so folding it into the bias is impossible — but it does not have to be
     * folded at all. This part's int8 depthwise cube gives every (channel, tap) TWO live
     * bytes and the datapath ADDS them, so the effective weight is a nine-bit value and
     * `w - w_zp` is carried exactly by splitting it across the pair. The coefficient A
     * already computes `-in_zp*(sum_w - w_zp*taps)`, which is the fold this pre-centred
     * cube wants, so nothing downstream changes. r76_wtile_pack() below is that split.
     *
     * What bounds it is the pair's own range: two signed bytes reach [-256, 254], and a
     * `w - w_zp` outside that is refused rather than clamped. It cannot arise from an
     * int8 weight and an int8 zero point without one of them being out of domain.
     * [HW sweep, H96 MAX M9, tests/rk3576_conv_lib_gate.c dwzp] */
    if (dw && w_zp) {
        int lo = -128 - w_zp, hi = 127 - w_zp;
        if (lo < -256 || hi > 254) {
            ROCKET_LOGE("%s: a depthwise weight zero point of %d puts w-w_zp outside "
                        "the [-256, 254] the cube's two live bytes reach\n", entry, w_zp);
            return ROCKET_E_UNSUPPORTED;
        }
    }

    h = calloc(1, sizeof *h);
    if (!h) return ROCKET_E_NOMEM;
    h->fd = fd;
    h->d = *d;
    h->dw = dw;
    h->in_scale = in_scale; h->w_scale = w_scale; h->out_scale = out_scale;
    h->in_zp = in_zp; h->w_zp = w_zp; h->out_zp = out_zp;
    h->ow = ow; h->oh = oh;

    h->IC = (unsigned)d->ic; h->OC = (unsigned)d->oc;
    h->IH = (unsigned)d->ih; h->IW = (unsigned)d->iw;
    h->KH = (unsigned)d->kh; h->KW = (unsigned)d->kw;
    h->SY = (unsigned)d->stride_y; h->SX = (unsigned)d->stride_x;
    h->PT = (unsigned)d->pad_top;  h->PL = (unsigned)d->pad_left;

    /* Channel counts as told to the REGISTERS. The direct path needs both rounded to the
     * 32-channel MAC group. The depthwise path takes the RAW count — its own two granules
     * (the weight cube rounds to 16, the CBUF allocation sometimes one 16-group further)
     * are the emitter's business, and rounding here would hide every count where the two
     * differ. */
    h->icreg = dw ? h->IC : rocket_rk3576_pad_ic(h->IC);
    h->icpad = (h->icreg + 31u) / 32u * 32u;
    h->surf_elems = rocket_rk3576_out_surf_elems(ow, oh, dw);
    /* THE TAP COUNT IS THE PROGRAMMED ONE, NOT THE CALLER'S, and it has to be — the CNA's
     * border substitution covers the whole programmed channel group, so at a pad tap the
     * cube's padding channels read the border constant while at an interior tap they read
     * whatever was memset into them. The B term sums over every tap, so any disagreement
     * between those two values lands in the accumulator scaled by the number of pad taps —
     * which varies per output position and so cannot be folded into A at all.
     *
     * r76_feature_fill() below closes that by memsetting the padding channels to the
     * border constant itself, the input zero point. Every tap then supplies in_zp on those
     * channels, the sum is position-independent, and the count the fold needs is the
     * REGISTER count: A = bias - in_zp*sum_w + in_zp*w_zp*icreg*kh*kw. Both halves are
     * required — the fill without the count, or the count without the fill, is wrong by
     * in_zp*w_zp*(icreg-ic)*kh*kw everywhere instead of only on the border.
     *
     * It is inert wherever it was already right: at ic a multiple of 32 the two counts are
     * equal, and at w_zp == 0 the term is zero. What it fixes is a NON-ZERO weight zero
     * point at a padded channel count, which is every ic from 1 to 31 and not only the
     * narrow ones. [HW sweep, H96 MAX M9, tests/rk3576_conv_lib_gate.c nic] */
    h->taps = dw ? h->KH * h->KW : h->icreg * h->KH * h->KW;
    h->in_bytes = (size_t)((h->icpad + C2 - 1u) / C2) * h->IH * h->IW * C2;
    h->max_tasks = oh + 2u;

    h->oc_tile = dw ? h->OC
                    : r76_conv_oc_tile(h->icreg, h->KH, h->KW,
                                       rocket_rk3576_pad_oc(h->OC));
    if (!h->oc_tile) { r76_w_free(h); return ROCKET_E_SHAPE; }

    /* Room for every row task's program at once when the job carries more than one, and
     * one task's when it does not — the plan is bounded by max_tasks either way. */
    h->batch = r76_batch_on() && h->max_tasks <= R76_MAX_CHAIN_TASKS;
    if (lut) {
        if (h->max_tasks > R76_MAX_CHAIN_TASKS) {
            ROCKET_LOGE("%s: %u row tasks exceeds the %u one job lays out\n",
                        entry, h->max_tasks, (unsigned)R76_MAX_CHAIN_TASKS);
            free(h);
            return ROCKET_E_SHAPE;
        }
        /* The chain rewrite links programs inside ONE BO, and the table load is a
         * separate one — so a LUT job takes the multi-descriptor layout and not the
         * single-kick one. It costs the kick saving, not correctness. */
        if (h->batch)
            ROCKET_LOGD("%s: task chaining is off for this call — the LUT's table load "
                        "is a task in its own BO, which the chain does not lay out\n",
                        entry);
        h->batch = 0;
        h->has_lut = 1;
        h->lut = lut->w;
        h->lut_le = calloc(RK3576_LUT_ENTRIES, sizeof *h->lut_le);
        h->lut_lo = calloc(RK3576_LUT_ENTRIES, sizeof *h->lut_lo);
        if (!h->lut_le || !h->lut_lo) { r76_w_free(h); return ROCKET_E_NOMEM; }
        memcpy(h->lut_le, lut->le, RK3576_LUT_ENTRIES * sizeof *h->lut_le);
        memcpy(h->lut_lo, lut->lo, RK3576_LUT_ENTRIES * sizeof *h->lut_lo);
    }
    h->multi = h->batch || h->has_lut;
    h->ops  = calloc((size_t)(h->multi ? h->max_tasks : 1u) * RK3576_CONV_TASK_OPS,
                     sizeof *h->ops);
    h->ext  = calloc(h->max_tasks, sizeof *h->ext);
    h->plan = calloc(h->max_tasks, sizeof *h->plan);
    h->sum_w = calloc(h->OC, sizeof *h->sum_w);
    if (bias) {
        h->bias = calloc(h->OC, sizeof *h->bias);
        if (h->bias) memcpy(h->bias, bias, (size_t)h->OC * sizeof *h->bias);
    }
    if (w_scale_oc) {
        h->sum_abs_w  = calloc(h->OC, sizeof *h->sum_abs_w);
        h->w_scale_oc = calloc(h->OC, sizeof *h->w_scale_oc);
        if (h->w_scale_oc)
            memcpy(h->w_scale_oc, w_scale_oc, (size_t)h->OC * sizeof *h->w_scale_oc);
        /* The sort and the output-channel tiling are the DIRECT path's levers only. A
         * depthwise conv pairs output channel c with INPUT channel c, and the feature cube
         * is packed once for every channel — so neither reordering the output channels nor
         * programming a window of them is a shape the emitter can express here. A
         * depthwise per-axis task is therefore one task carrying the whole layer, which it
         * can afford: its accumulator is bounded by kh*kw taps rather than ic*kh*kw, so
         * the C ramp reaches the int16 field's own ceiling and the gain resolution a
         * single OUT_CVT shift leaves is far better than the direct path's at the same
         * channel count. */
        if (!dw) h->perm = calloc(h->OC, sizeof *h->perm);
    }
    if (!h->ops || !h->ext || !h->plan || !h->sum_w || (bias && !h->bias) ||
        (w_scale_oc && (!h->sum_abs_w || !h->w_scale_oc || (!dw && !h->perm)))) {
        r76_w_free(h);
        return ROCKET_E_NOMEM;
    }
    *out_h = h;
    return ROCKET_OK;
}

/*
 * The weight-side arithmetic that is not per tile, over the caller's W: the filter sums,
 * the scale sort, and the output-channel tile the accuracy target picks.
 *
 * Split from r76_w_prepare() only because it needs the weights and the validation does
 * not; an entry that has both calls them back to back.
 */
static void r76_w_sums(r76_w *h, const int8_t *W)
{
    /* Each output channel's whole filter, for the input zero point's fold — and its
     * absolute sum, which is what bounds that channel's accumulator and so how large a
     * per-channel C multiplier it can carry.
     *
     * One output channel's filter is CONTIGUOUS in the caller's tensor on both paths, so
     * this is a linear scan and not a strided one. Walking it through the index expression
     * instead cost three 64-bit multiplies per weight, which on a 1024x1024 1x1 layer is a
     * million of them — 45% of that layer's wall, against a submit that is 2% of it. An
     * int32 accumulator is enough: the widest filter the envelope admits is ~5600 taps, so
     * |sum| <= 5600*128 and cannot overflow. */
    const size_t per_oc = h->dw ? (size_t)h->KH * h->KW
                                : (size_t)h->IC * h->KH * h->KW;
    unsigned c;
    size_t k;

    for (c = 0; c < h->OC; c++) {
        const int8_t *w = W + (size_t)c * per_oc;
        int32_t s = 0;
        if (h->sum_abs_w) {
            int32_t sa = 0;
            for (k = 0; k < per_oc; k++) {
                int v = w[k];
                s += v; sa += v < 0 ? -v : v;
            }
            h->sum_abs_w[c] = sa;
        } else {
            for (k = 0; k < per_oc; k++) s += w[k];
        }
        h->sum_w[c] = s;
    }

    /* Sorted by scale, so each output-channel tile — its own task, its own OUT_CVT shift —
     * spans as little of the layer's scale range as the tiling allows. */
    if (h->perm) r76_sort_by_scale(h->perm, h->OC, h->w_scale_oc);

    /* AFTER the filter sums, which is what the accumulator bound is built from: on this
     * path the output-channel tile is an ACCURACY parameter as well as a CBUF one, because
     * a tile is one task and a task carries one OUT_CVT shift. */
    if (h->w_scale_oc && !h->dw)
        h->oc_tile = r76_pc_oc_tile(h->OC, h->oc_tile, h->perm, h->sum_abs_w, h->bias,
                                    h->sum_w, h->in_zp, h->w_zp, h->taps, h->in_scale,
                                    h->w_scale_oc, h->out_scale);
}

/* The tile table, once the tile is decided. Separate from the packing so a transient run
 * can walk the tiles and pack them one at a time. */
static int r76_w_tiles(r76_w *h)
{
    unsigned oc0, t = 0;

    h->ntile = (h->OC + h->oc_tile - 1u) / h->oc_tile;
    h->tile = calloc(h->ntile, sizeof *h->tile);
    if (!h->tile) return ROCKET_E_NOMEM;
    for (oc0 = 0; oc0 < h->OC; oc0 += h->oc_tile, t++) {
        struct r76_int8_wtile *s = &h->tile[t];
        s->oc0 = oc0;
        s->tile_oc = h->OC - oc0 < h->oc_tile ? h->OC - oc0 : h->oc_tile;
        s->ocreg = h->dw ? s->tile_oc : rocket_rk3576_pad_oc(s->tile_oc);
        s->base_scale = h->w_scale;
    }
    return ROCKET_OK;
}

/* How big a tile's surface BO is, against the `ceil(ocreg/C2)` groups the program writes.
 *
 * A DIRECT program is told the round-32 count, so it writes every group its consumer walks
 * and the two are the same number. A DEPTHWISE one is told the RAW count and writes nothing
 * past it, so at a channel count that is not a multiple of 32 its surface is a group SHORT
 * of what a direct consumer's feature DMA reads — 9 groups at oc=144 against 10 — and the
 * join has to be refused however sound the arithmetic is. [HW sweep, H96 MAX M9,
 * tests/rk3576_pad_channels.c]
 *
 * So a depthwise tile allocates the deeper surface and fills the group past its own with
 * the OUTPUT ZERO POINT, which is exactly the border constant the consumer's own cube would
 * have used and so exactly what its B term's fold over the programmed group assumes. That
 * fill is written ONCE, at pack time: the program does not address those bytes, and the
 * sentinel covers the written groups alone.
 *
 * Only when the raw count fills its last written group. Below that the tail lies INSIDE a
 * group the program does write, and whether the DPU leaves those channels alone has not
 * been measured — so that case keeps the short surface and its join keeps refusing. */
static size_t r76_surf_bytes(const r76_w *h, unsigned ocreg)
{
    unsigned groups = (ocreg + C2 - 1u) / C2;

    if (h->dw && ocreg % C2 == 0u && rocket_rk3576_pad_oc(ocreg) > ocreg)
        groups = rocket_rk3576_pad_oc(ocreg) / C2;
    return (size_t)groups * h->surf_elems * C2;
}

/*
 * Pack one output-channel tile: its weight cube, its coefficient group, and the output
 * surface it writes. Everything here is a function of the WEIGHTS, the quant contract and
 * the frozen geometry — nothing of the input — which is what makes it resident-able.
 */
static int r76_wtile_pack(const char *entry, r76_w *h, unsigned t, const int8_t *W,
                          struct r76_int8_prof *prof)
{
    struct r76_int8_wtile *s = &h->tile[t];
    unsigned ocreg = s->ocreg, tile_oc = s->tile_oc, oc0 = s->oc0;
    unsigned KH = h->KH, KW = h->KW, IC = h->IC, icreg = h->icreg;
    size_t w_bytes = h->dw ? rocket_rk3576_weight_dw_bytes(ocreg, KH, KW)
                           : (size_t)((ocreg + 31u) / 32u) * ((icreg + 31u) / 32u) *
                             32u * 32u * KH * KW;
    size_t coeff_bytes = h->dw ? rocket_rk3576_coeff_bytes_dw(ocreg)
                               : rocket_rk3576_coeff_bytes(ocreg);
    size_t obytes = (size_t)((ocreg + C2 - 1u) / C2) * h->surf_elems * C2;
    size_t surf_bytes = r76_surf_bytes(h, ocreg);
    int32_t *A = NULL;
    int16_t *B = NULL, *Cmul = NULL;
    int rc = ROCKET_OK;
    double pt0 = R76_PT(*prof);

    if (rocket_bo_alloc(h->fd, w_bytes, &s->w) < 0 ||
        rocket_bo_alloc(h->fd, coeff_bytes, &s->coeff) < 0)
        return ROCKET_E_NOMEM;
    /* The surface, unless this handle writes into a caller's buffer instead — then it owns
     * none and allocating one here would be a megabyte per layer of nothing. */
    if (!h->out_ext.ptr && rocket_bo_alloc(h->fd, surf_bytes, &s->out) < 0)
        return ROCKET_E_NOMEM;
    /* THE TAIL A DIRECT CONSUMER WALKS, filled once. See r76_surf_bytes(). */
    s->tail_zp = 0;
    if (!h->out_ext.ptr && surf_bytes > obytes) {
        rocket_bo_prep(h->fd, &s->out, 1, 0);
        memset((char *)s->out.ptr + obytes, (unsigned char)(int8_t)h->out_zp,
               surf_bytes - obytes);
        rocket_bo_fini(h->fd, &s->out);
        s->tail_zp = 1;
    }

    /* The WEIGHT cube is per-tile and each tile is its own convolution, so its group count
     * follows the tile rather than the whole output-channel count. */
    rocket_bo_prep(h->fd, &s->w, 1, 0);
    memset(s->w.ptr, 0, w_bytes);
    {
        int8_t *cube = (int8_t *)s->w.ptr;
        unsigned c, i, y, x;
        if (h->dw) {
            /* This part's own depthwise cube: channels grouped by 64, tap-major inside a
             * group, two byte slots per weight — and at int8 the byte a channel owns
             * inside a tap block is 4*(c/2) + (c%2), with its SECOND two further on.
             * Neither the direct path's cube nor the RK3588's single-byte one. Both bytes
             * contribute, so the pair carries the zero-point-centred weight when one byte
             * cannot. */
            for (c = 0; c < tile_oc; c++) {
                const int8_t *wrow = W + (size_t)r76_oc_of(h->perm, oc0 + c) * KH * KW;
                for (y = 0; y < KH; y++)
                    for (x = 0; x < KW; x++) {
                        int v = wrow[(size_t)y * KW + x] - h->w_zp;
                        int lo = v > 127 ? 127 : (v < -128 ? -128 : v);
                        int at = rocket_rk3576_weight_dw_int8(ocreg, KH, KW, c, y, x);
                        cube[at]      = (int8_t)lo;
                        cube[at + 2]  = (int8_t)(v - lo);
                    }
            }
        } else {
            /* The same hoist. weight_conv_int8() at these groups is
             * (c/32)*nIC1*KH*KW*1024 + (i/32)*KH*KW*1024 + (y*KW + x)*1024 +
             * (c%32)*32 + (i%32), so the taps of one (channel, input channel) pair walk a
             * fixed 1024-byte stride from a contiguous source run. */
            unsigned nIC1 = (icreg + 31u) / 32u, tap, ktaps = KH * KW;
            for (c = 0; c < tile_oc; c++) {
                size_t cbase = (size_t)(c / 32u) * nIC1 * ktaps * 1024u
                             + (size_t)(c % 32u) * 32u;
                /* The source channel is a property of c, not of (c, i) — resolving it
                 * inside the input-channel loop is one call per weight, and on a
                 * 1024x1024 1x1 layer that is a million of them. */
                const int8_t *wrow = W + (size_t)r76_oc_of(h->perm, oc0 + c) * IC * ktaps;
                /* AT ONE TAP THE CUBE IS A RUN OF MEMCPYS. The 32 input channels of one
                 * group land on 32 CONSECUTIVE bytes of the cube, and at kh=kw=1 they come
                 * from 32 consecutive bytes of the filter — so a 1x1 conv, which is what a
                 * MobileNet or a transformer projection is mostly made of, packs its cube
                 * in IC/32 block moves instead of IC byte stores. A taller kernel reads at
                 * a tap stride and cannot. */
                if (ktaps == 1u) {
                    unsigned k, nblk = (IC + 31u) / 32u;
                    for (k = 0; k < nblk; k++) {
                        unsigned n = IC - k * 32u < 32u ? IC - k * 32u : 32u;
                        memcpy(cube + cbase + (size_t)k * 1024u,
                               wrow + (size_t)k * 32u, n);
                    }
                    continue;
                }
                for (i = 0; i < IC; i++) {
                    int8_t *dst = cube + cbase
                                + (size_t)(i / 32u) * ktaps * 1024u + (i % 32u);
                    const int8_t *src = wrow + (size_t)i * ktaps;
                    for (tap = 0; tap < ktaps; tap++) dst[(size_t)tap * 1024u] = src[tap];
                }
            }
        }
    }
    rocket_bo_fini(h->fd, &s->w);
    R76_ACC(*prof, w_us, pt0);
    pt0 = R76_PT(*prof);

    /* The COEFFICIENT buffer is NOT a flat int32 bias array on this part, and a zeroed one
     * makes the DPU write a full but entirely empty surface whatever the MAC did — the C
     * term gates the BS stage. The tail channels of a partial group get a zero A term so
     * they carry a C term too. */
    A = calloc(ocreg, sizeof *A);
    /* B is the DIRECT path's weight-zero-point term. The depthwise group has no such field
     * and does not need one: its cube is pre-centred above. */
    B = (h->w_zp && !h->dw) ? calloc(ocreg, sizeof *B) : NULL;
    Cmul = h->w_scale_oc ? calloc(ocreg, sizeof *Cmul) : NULL;
    if (!A || (h->w_zp && !h->dw && !B) || (h->w_scale_oc && !Cmul)) {
        rc = ROCKET_E_NOMEM; goto out;
    }
    r76_fold_coeff(A, h->bias, oc0, tile_oc, h->sum_w, h->in_zp, h->w_zp, h->taps,
                   h->perm);
    if (B) {
        unsigned j;
        /* An asymmetric weight is w_true = w_stored - w_zp, whose correction is
         * -w_zp*sum(x), and the DPU ADDS the B term. */
        for (j = 0; j < tile_oc; j++) B[j] = (int16_t)(-h->w_zp);
        /* AND THE TAIL OF A PARTIAL GROUP GETS NONE, which is what makes this surface
         * usable as the next layer's cube. Those channels carry zero weights and a zero A,
         * so with no B they requant a zero accumulator and land on the output zero point —
         * the constant a consumer whose own channel count is not a multiple of 32 needs to
         * find there. Give them the live channels' B instead and they carry
         * `requant(B*sum(x))`, which is data-dependent: measured 5776 of 12544 padding
         * elements away from the output zero point at oc=16 w_zp=12
         * [HW sweep, H96 MAX M9, tests/rk3576_pad_channels.c]. Nothing reads these channels
         * as an output, so the correction they lose is one nobody applies. */
        for (j = tile_oc; j < ocreg; j++) B[j] = 0;
    }
    /* The per-channel gain is planned per TILE, because the one (MUL, SHIFT) the OUT_CVT
     * carries is per task and each output-channel tile is its own task. */
    if (h->w_scale_oc) {
        double err = 0.0;
        if (r76_plan_perchannel(entry, oc0, tile_oc, ocreg, A, h->sum_abs_w,
                                h->in_scale, h->w_scale_oc, h->out_scale, h->perm,
                                Cmul, &s->base_scale, &err) != 0) {
            rc = ROCKET_E_SHAPE; goto out;
        }
        if (err > h->worst_rel_err) h->worst_rel_err = err;
    }
    rocket_bo_prep(h->fd, &s->coeff, 1, 0);
    if (h->dw)       rocket_rk3576_pack_coeff_dw_perc(s->coeff.ptr, coeff_bytes, A,
                                                      ocreg, Cmul, 1);
    else if (Cmul)   rocket_rk3576_pack_coeff_perc(s->coeff.ptr, coeff_bytes, A, ocreg,
                                                   B, Cmul, 1);
    else if (B)      rocket_rk3576_pack_coeff_asym(s->coeff.ptr, coeff_bytes, A, ocreg,
                                                   B, 1);
    else             rocket_rk3576_pack_coeff(s->coeff.ptr, coeff_bytes, A, ocreg);
    rocket_bo_fini(h->fd, &s->coeff);
    R76_ACC(*prof, coeff_us, pt0);

out:
    free(A); free(B); free(Cmul);
    return rc;
}

/* Scatter the caller's row-major CHW tensor into this handle's feature cube.
 *
 * CUBE IN is the early return: the producer's surface already IS this cube, so there is no
 * scatter and no cache maintenance either — nothing on the CPU touches these lines, and the
 * two PREP_BO/FINI_BO ioctls the scatter needs go with it.
 *
 * Split out of r76_int8_exec() because a cross-layer chain packs the FIRST handle's cube and
 * nothing else: there is one scatter per chain, not one per layer.
 */
static int r76_feature_pack(r76_w *h, const int8_t *in)
{
    if (h->cube_in) return ROCKET_OK;
    if (!h->in.ptr) {
        if (rocket_bo_alloc(h->fd, h->in_bytes, &h->in) < 0) return ROCKET_E_NOMEM;
        rocket_bo_prep(h->fd, &h->in, 1, 0);
        /* Only on the allocation. The channels past `ic` are the cube's padding and nothing
         * writes them again, while every live channel is fully overwritten below — so a
         * reused cube needs the scatter and not the fill.
         *
         * THE FILL VALUE IS THE BORDER CONSTANT, not zero: the CNA substitutes the input
         * zero point across the whole programmed channel group at a pad tap, so a padding
         * channel holding anything else makes the B term's sum depend on how many taps of
         * that output position fell outside the plane. See h->taps in r76_w_prepare(). */
        memset(h->in.ptr, (unsigned char)(int8_t)h->in_zp, h->in_bytes);
    } else {
        rocket_bo_prep(h->fd, &h->in, 1, 0);
    }
    /* The channel-group arithmetic is hoisted out of the pixel loop. A CHW tensor and an
     * NC1HWC2 cube are a transpose, not a copy, so this stays a strided store — but
     * calling the index function per element pays a divide and a modulo on every one of
     * IC*IH*IW, and that was most of what a conv on this part spent outside the NPU.
     *
     * CHANNEL OUTER, PIXEL INNER — one sequential read stream and a strided store. The
     * other order (pixel outer, channel inner) fills each C2-byte atom in one go and so
     * touches every destination line once instead of once per channel, which looks like
     * the better trade and is not: it needs C2 concurrent read streams, and measured on
     * the graph it cost the depthwise layers more than it bought the 1x1 ones (layer 3's
     * feature pack 3.2 -> 5.3 ms). */
    {
        int8_t *cube = (int8_t *)h->in.ptr;
        size_t px = (size_t)h->IH * h->IW;
        const int8_t *sp[C2];
        unsigned c, k;
        for (c = 0; c < h->IC; c += C2) {
            unsigned live = h->IC - c < C2 ? h->IC - c : C2;
            for (k = 0; k < live; k++) sp[k] = in + (size_t)(c + k) * px;
            r76_c2_pack(cube + (size_t)(c / C2) * px * C2, sp, live, px,
                        (unsigned char)(int8_t)h->in_zp);
        }
    }
    rocket_bo_fini(h->fd, &h->in);
    return ROCKET_OK;
}

/*
 * Run the convolution. `W` is NULL for a resident handle, whose tiles are already packed,
 * and the caller's weights for a transient one, whose tiles are packed as the loop reaches
 * them and dropped behind it — so a one-shot call still holds one tile's cube at a time,
 * which is the memory profile it had before residency existed.
 *
 * The caller owns `prof` and logs it, because the transient path's BO TEARDOWN happens
 * after this function returns and it is not small — freeing a tile's cube unmaps a
 * megabyte-sized dirty mapping, and a profile that stopped at the last bucket left that
 * time attributed to nothing at all.
 */
static int r76_int8_exec(const char *entry, r76_w *h, const int8_t *W,
                         const int8_t *in, int8_t *out, struct r76_int8_prof *profp)
{
    struct r76_int8_prof prof = *profp;
    unsigned t, ti;
    unsigned char stamp;
    int rc = ROCKET_OK;
    double pt0;

    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;

    /* The FEATURE cube is packed once and shared by every output-channel tile: the tiling
     * is on the output axis, which the feature side does not see. */
    pt0 = R76_PT(prof);
    rc = r76_feature_pack(h, in);
    R76_ACC(prof, in_us, pt0);
    if (rc != ROCKET_OK) return rc;

    /* One extra word per task: the chain lays each program out at an EVEN word stride
     * (rkt_chain_words), which rounds an odd program length up by one, and the
     * multi-descriptor layout uses the same slot. */
    if (!h->rc.ptr &&
        rocket_bo_alloc(h->fd, (size_t)(h->multi ? h->max_tasks : 1u) *
                        R76_TASK_SLOT_WORDS * sizeof(uint64_t), &h->rc) < 0)
        return ROCKET_E_NOMEM;

    /* The table load, once per handle: it is a function of the tables alone, so a
     * resident caller pays for it at the first inference and never again. The dummy
     * cube it writes 16 bytes into sits past the program in the same BO. */
    if (h->has_lut && !h->lut_bo.ptr) {
        lut_load_params_rk3576_t lp = {0};
        size_t prog = (size_t)RK3576_LUT_TASK_OPS * sizeof(uint64_t);
        uint64_t *lops = calloc(RK3576_LUT_TASK_OPS, sizeof(uint64_t));

        if (!lops) return ROCKET_E_NOMEM;
        h->lut_scratch = (uint32_t)((prog + 63u) & ~(size_t)63u);
        if (rocket_bo_alloc(h->fd, h->lut_scratch + 64u, &h->lut_bo) < 0) {
            free(lops);
            return ROCKET_E_NOMEM;
        }
        lp.lo = h->lut_le; lp.hi = h->lut_lo;
        lp.scratch_dma = h->lut_bo.dma_address + h->lut_scratch;
        lp.tasks = lops;
        if (gen_lut_load_rk3576(&lp) != 0) {
            free(lops);
            ROCKET_LOGE("%s: the LUT table-load generator refused\n", entry);
            return ROCKET_E_UNSUPPORTED;
        }
        h->lut_ops = lp.task_count;
        rocket_bo_prep(h->fd, &h->lut_bo, 1, 0);
        memcpy(h->lut_bo.ptr, lops, (size_t)lp.task_count * sizeof(uint64_t));
        rocket_bo_fini(h->fd, &h->lut_bo);
        free(lops);
    }

    for (ti = 0; ti < h->ntile; ti++) {
        struct r76_int8_wtile *s = &h->tile[ti];
        /* This call's feature cube: the handle's own, or the producer's it borrowed. */
        const rocket_bo *fin = h->cube_in ? &h->src : &h->in;
        unsigned ocreg = s->ocreg, tile_oc = s->tile_oc;
        size_t obytes = (size_t)((ocreg + C2 - 1u) / C2) * h->surf_elems * C2;
        /* This tile's surface and where inside its BO it starts — the handle's own, or the
         * slice of a caller's buffer it was told to write. */
        size_t ooff = 0;
        rocket_bo *surf;
        struct r76_conv_bos b;
        conv_params_t p = {0};
        uint32_t in_h[5], out_h[1];
        unsigned ntask = 1u, n_in;
        uint32_t task_ops;

        prof.tiles++;
        if (!s->w.ptr) {
            rc = r76_wtile_pack(entry, h, ti, W, &prof);
            if (rc != ROCKET_OK) goto done;
        }
        surf = r76_surf(h, ti, &ooff);
        pt0 = R76_PT(prof);

        /* The submit helpers take the five BOs as one group, so hand them this tile's — a
         * borrowed view, released through the handle rather than through r76_conv_free. */
        memset(&b, 0, sizeof b);
        b.in = *fin; b.rc = h->rc;
        b.w = s->w; b.coeff = s->coeff; b.out = *surf;
        b.lut = h->lut_bo; b.lut_ops = h->lut_ops; b.lut_scratch = h->lut_scratch;

        p.ic = (uint16_t)h->icreg; p.ih = (uint16_t)h->IH; p.iw = (uint16_t)h->IW;
        p.oc = (uint16_t)ocreg; p.oh = (uint16_t)h->oh; p.ow = (uint16_t)h->ow;
        p.kh = (uint16_t)h->KH; p.kw = (uint16_t)h->KW;
        p.stride_y = (uint8_t)h->SY; p.stride_x = (uint8_t)h->SX;
        p.pad_top  = (uint8_t)h->PT; p.pad_left = (uint8_t)h->PL;
        p.ih_full = (uint16_t)h->IH; p.oh_full = (uint16_t)h->oh;
        /* Zero unless a borrowed cube's groups sit further apart than this plane. */
        p.in_surf_elems = h->src_surf_elems;
        p.int8_out = 1;
        /* Per-channel: the C multipliers carry every channel's gain RELATIVE to one base,
         * and the base is what the OUT_CVT programs — so it is handed over whole rather
         * than as the in/w/out triple the per-tensor path decomposes into. */
        if (h->w_scale_oc) {
            p.in_scale = s->base_scale; p.w_scale = 1.0f; p.out_scale = 1.0f;
        } else {
            p.in_scale = h->in_scale; p.w_scale = h->w_scale; p.out_scale = h->out_scale;
        }
        /* Both zero points reach the registers uint8-centered: the emitter programs the
         * border constant as (input_zero_point & 0xff) - 0x80 and the output offset as
         * output_zero_point - 0x80, so a model-domain signed zero point is that value plus
         * 0x80. The weight zero point rides in the coefficient group's B term instead and
         * this field is inert on the RK3576 path. */
        p.input_zero_point  = h->in_zp  + 0x80;
        p.output_zero_point = h->out_zp + 0x80;
        p.weight_zero_point = 0x80;
        p.lut         = h->has_lut ? &h->lut : NULL;
        p.tasks       = h->ops;
        p.input_dma   = (uint32_t)r76_feat_dma(h);
        p.weights_dma = s->w.dma_address;
        p.bias_dma    = s->coeff.dma_address;
        p.output_dma  = (uint32_t)(surf->dma_address + ooff);

        {
            conv_params_t q = p;
            if (rocket_rk3576_plan_rows(&q, h->dw, h->plan, h->max_tasks, &ntask) < 0) {
                ROCKET_LOGE("%s: no row plan for ic=%u oc tile %u (%ux%u k%ux%u s%u) — "
                            "the recourse is an input-channel split, which this path's "
                            "on-chip requant forecloses\n",
                            entry, h->IC, tile_oc, h->IW, h->IH, h->KW, h->KH, h->SX);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }
        }

        n_in = 0u;
        /* A BO IS NAMED ONCE PER JOB. The driver locks each through drm_exec and locking
         * one twice is EALREADY, which rejects the submit — and a layer reading one slice
         * of a concatenation buffer while writing another is exactly one BO in both roles.
         * Named as the OUTPUT, which is where the chain layout names an intermediate too. */
        if (fin->handle != surf->handle) in_h[n_in++] = fin->handle;
        in_h[n_in++] = s->w.handle;
        in_h[n_in++] = s->coeff.handle;
        in_h[n_in++] = h->rc.handle;
        if (h->has_lut) in_h[n_in++] = h->lut_bo.handle;
        out_h[0] = surf->handle;

        /* THE STAMP RIDES THE DE-SCATTER'S BRACKET, so this pair only runs for a
         * surface that does not already carry it — a fresh allocation, or a transient
         * call whose tile is dropped after it. See the re-stamp below.
         *
         * ONLY THIS TILE'S SLICE. With an external destination the rest of that buffer
         * belongs to another producer, and stamping it would erase a surface that has
         * already been written this inference. */
        if (stamp && !s->stamped) {
            rocket_bo_prep(h->fd, surf, 1, 0);
            memset((char *)surf->ptr + ooff, stamp, obytes);
            rocket_bo_fini(h->fd, surf);
        }
        s->stamped = 0;

        /* The row tasks of this tile, as one stream when batching is on and as one submit
         * each when it is not. Independent by construction: task t reads its own window of
         * the shared feature cube and writes its own rows of the shared surface, and the
         * weight and coefficient buffers do not change across them. */
        task_ops = 0;
        for (t = 0; t < ntask; t++) {
            conv_params_t q = p;
            q.ih = h->plan[t].ih; q.oh = h->plan[t].oh;
            q.pad_top = h->plan[t].pad_top;
            q.input_dma  = p.input_dma  + h->plan[t].feature_off;
            q.output_dma = p.output_dma + h->plan[t].output_off;
            /* ALWAYS, not only when the plan split. A single-task plan's window is still
             * shorter than the plane whenever the last output row does not reach the
             * bottom input row — ordinary stride-2 VALID geometry — and leaving these at
             * the window makes the emitter derive the DDR channel-group stride from the
             * WINDOW, so every group past the first reads at the wrong offset and the
             * surface comes back unrelated to the input, with nothing to fault on. */
            q.ih_full = (uint16_t)h->IH; q.oh_full = (uint16_t)h->oh;
            q.tasks = h->ops + (size_t)(h->multi ? t : 0u) * RK3576_CONV_TASK_OPS;
            if ((h->dw ? gen_conv2d_dw_int8_rk3576(&q)
                       : gen_conv2d_int8_rk3576(&q)) != 0) {
                ROCKET_LOGE("%s: the generator refused task %u of %u\n", entry, t, ntask);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }
            /* The chain lays every program out at ONE stride, so a tile whose tasks
             * somehow differ in length cannot be chained. They do not differ — the row
             * tasks of a tile write the same registers with different values — but
             * ROCKET_RK3576_ADD can lengthen a program, so check rather than assume. */
            if (!t) task_ops = q.task_count;
            else if (h->multi && q.task_count != task_ops) {
                ROCKET_LOGE("%s: task %u is %u ops against task 0's %u, so this tile "
                            "cannot go out as one job\n", entry, t, q.task_count,
                            task_ops);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }
            h->ext[t].base        = ooff;
            h->ext[t].groups      = (ocreg + C2 - 1u) / C2;
            h->ext[t].group_bytes = (size_t)h->surf_elems * C2;
            h->ext[t].row_off     = (size_t)h->plan[t].oy0 * h->ow * C2;
            h->ext[t].span        = (size_t)h->plan[t].oh * h->ow * C2;
            if (!h->multi) {
                R76_ACC(prof, gen_us, pt0);
                pt0 = R76_PT(prof);
                rc = r76_submit_ops(h->fd, &b, h->ops, q.task_count, in_h, n_in, out_h,
                                    &h->ext[t], 1u, 0, stamp, 0u, entry);
                R76_ACC(prof, submit_us, pt0);
                pt0 = R76_PT(prof);
                if (rc != ROCKET_OK) goto done;
            }
        }
        prof.tasks += ntask;
        if (h->multi) {
            R76_ACC(prof, gen_us, pt0);
            pt0 = R76_PT(prof);
            rc = r76_submit_ops(h->fd, &b, h->ops, task_ops, in_h, n_in, out_h, h->ext,
                                ntask, h->batch, stamp, 0u, entry);
            R76_ACC(prof, submit_us, pt0);
            pt0 = R76_PT(prof);
            if (rc != ROCKET_OK) goto done;
        } else {
            R76_ACC(prof, gen_us, pt0);
            pt0 = R76_PT(prof);
        }

        /* CUBE OUT. The surface stays where the DPU wrote it and the next layer's DPU
         * reads it, so there is nothing to de-scatter and nothing to make CPU-visible —
         * the fence was already waited on inside r76_submit_ops. The sentinel is NOT
         * re-stamped here either: this surface is about to be read as an input, so its
         * stamp goes on at the START of the next call, in its own bracket. */
        if (h->cube_out) continue;

        /* De-scatter this tile's channels straight into the caller's row-major out.
         * The PREP_BO here is the DPU's write DRAIN and not host work at all, so it is
         * measured apart: with the transpose vectorized it is the larger half of this
         * bucket, and a lever aimed at the de-scatter would be aimed at the wrong thing. */
        {
            double dt0 = R76_PT(prof);
            rocket_bo_prep(h->fd, surf, 0, 2000000000ull);
            R76_ACC(prof, drain_us, dt0);
        }
        /* The same hoist on the way back: y*ow + x is the pixel index, contiguous in the
         * CHW output and at a fixed C2 stride in the cube. */
        {
            const int8_t *o = (const int8_t *)surf->ptr + ooff;
            size_t px = (size_t)h->oh * h->ow;
            unsigned c, k;
            for (c = 0; c < tile_oc; c += C2) {
                unsigned m = tile_oc - c < C2 ? tile_oc - c : C2;
                int8_t *dp[C2];
                for (k = 0; k < m; k++)
                    dp[k] = out + (size_t)r76_oc_of(h->perm, s->oc0 + c + k) * px;
                r76_c2_unpack(dp, m, o + (size_t)(c / C2) * h->surf_elems * C2, px);
            }   /* unchanged shape; the helper took the pointer table */
        }
        /* THE STAMP RIDES THE DE-SCATTER'S BRACKET. The sentinel STAYS — it is what
         * makes "this task never wrote" a property of the surface rather than a guess,
         * and a conv inherits the wide-output poisoning from whatever ran before it —
         * so what moves is WHERE it is written, not whether. Writing it here costs the
         * fill and nothing else, because the buffer is already inside a PREP_BO/FINI_BO
         * pair; writing it before the next submit costs a second pair, and the two
         * cache-maintenance ioctls were 99% of that bucket, not the memset.
         *
         * Only for a surface that will still be here next call. A transient tile is
         * dropped a few lines below, so stamping it would be pure cost.
         *
         * Still bracketed, never bare: PREP_BO has already synced these lines for the
         * CPU and the FINI_BO under it writes them back, so nothing is left dirty to
         * race the DPU's DMA. */
        if (stamp && h->resident) {
            memset((char *)surf->ptr + ooff, stamp, obytes);
            s->stamped = 1;
        }
        rocket_bo_fini(h->fd, surf);
        R76_ACC(prof, read_us, pt0);

        /* A transient call holds one tile's cube at a time, as it did before residency. */
        if (!h->resident) {
            pt0 = R76_PT(prof);
            r76_wtile_drop(h, ti);
            R76_ACC(prof, free_us, pt0);
        }
    }

done:
    *profp = prof;
    return rc;
}

/* What the profile line names a call by. Taken off the handle before it is freed, because
 * the teardown is one of the buckets. */
struct r76_prof_shape {
    unsigned ic, oc, iw, ih, kw, kh, sx;
    int      resident;
};

static void r76_prof_shape_of(struct r76_prof_shape *s, const r76_w *h)
{
    s->ic = h->IC; s->oc = h->OC; s->iw = h->IW; s->ih = h->IH;
    s->kw = h->KW; s->kh = h->KH; s->sx = h->SX;
    s->resident = h->resident;
}

/* One line per call, with every bucket. Called by the entry rather than by the exec above,
 * so the transient path's teardown is inside it. */
static void r76_int8_prof_log(const char *entry, const struct r76_prof_shape *s,
                              const struct r76_int8_prof *prof)
{
    double host = prof->sums_us + prof->in_us + prof->w_us + prof->coeff_us +
                  prof->gen_us + prof->read_us + prof->free_us;
    double tot = host + prof->submit_us;

    if (!prof->on) return;
    if (tot <= 0.0) tot = 1.0;
    ROCKET_LOGI("%s: ic=%u oc=%u %ux%u k%ux%u s%u — %u tile(s), %u task(s), "
                "%.2f ms: sums %.2f (%.0f%%) feature %.2f (%.0f%%) weights %.2f "
                "(%.0f%%) coeff %.2f (%.0f%%) plan+gen %.2f (%.0f%%) submit %.2f "
                "(%.0f%%) readback %.2f (%.0f%%, drain %.2f) teardown %.2f (%.0f%%) "
                "— host %.0f%%%s\n",
                entry, s->ic, s->oc, s->iw, s->ih, s->kw, s->kh, s->sx,
                prof->tiles, prof->tasks, tot / 1e3,
                prof->sums_us / 1e3,   100.0 * prof->sums_us / tot,
                prof->in_us / 1e3,     100.0 * prof->in_us / tot,
                prof->w_us / 1e3,      100.0 * prof->w_us / tot,
                prof->coeff_us / 1e3,  100.0 * prof->coeff_us / tot,
                prof->gen_us / 1e3,    100.0 * prof->gen_us / tot,
                prof->submit_us / 1e3, 100.0 * prof->submit_us / tot,
                prof->read_us / 1e3,   100.0 * prof->read_us / tot,
                prof->drain_us / 1e3,
                prof->free_us / 1e3,   100.0 * prof->free_us / tot,
                100.0 * host / tot,
                s->resident ? " — resident weights" : "");
}

static int r76_conv_int8_run(const char *entry, int fd, const rocket_conv2d_desc *d,
                             int dw, const int8_t *in, const int8_t *W,
                             const int32_t *bias, float in_scale, float w_scale,
                             const float *w_scale_oc,
                             float out_scale, int in_zp, int w_zp, int out_zp,
                             const struct r76_lut *lut, int8_t *out)
{
    struct r76_int8_prof prof = {0};
    struct r76_prof_shape shape;
    r76_w *h = NULL;
    double pt0;
    int rc;

    prof.on = r76_int8_prof_on();
    rc = r76_w_prepare(entry, fd, d, dw, bias, in_scale, w_scale, w_scale_oc,
                       out_scale, in_zp, w_zp, out_zp, lut, &h);
    if (rc != ROCKET_OK) return rc;
    if (!in || !W || !out) { r76_w_free(h); return ROCKET_E_SHAPE; }

    pt0 = R76_PT(prof);
    r76_w_sums(h, W);
    R76_ACC(prof, sums_us, pt0);

    rc = r76_w_tiles(h);
    if (rc == ROCKET_OK) rc = r76_int8_exec(entry, h, W, in, out, &prof);

    if (h->w_scale_oc && rc == ROCKET_OK)
        ROCKET_LOGI("%s: per-channel requant, worst-case gain error %.3g%%\n",
                    entry, h->worst_rel_err * 100.0);
    /* The handle's own teardown — the feature cube, the regcmd buffer, whatever tile the
     * loop was holding — which is the unmapping of every BO this layer touched, and was
     * the last unattributed time on the path. */
    r76_prof_shape_of(&shape, h);
    pt0 = R76_PT(prof);
    r76_w_free(h);
    R76_ACC(prof, free_us, pt0);
    r76_int8_prof_log(entry, &shape, &prof);
    return rc;
}

/* ============================================================================
 * SECTION — the INT8 first conv, on the packed-image datapath
 *
 * The quantized stem. Same CNA sub-encoding as the fp16 first conv and a different
 * object in three places, each of which had to be read off the part:
 *
 *   - THE WEIGHT CUBE is single bytes in output-channel groups of 32 with the tap row
 *     outside the group and the tap column folded in beside the four lanes, where the
 *     float cube is 16-bit slots in groups of sixteen with the tap axis outermost.
 *     rocket_rk3576_argb_int8_pack_weights() owns it.
 *   - THE LEFT PAD MUST BE NON-ZERO and THE OUTPUT WIDTH MUST BE iw/stride. Neither
 *     holds on the float path, and neither appears in any capture, because every
 *     captured first conv is a SAME convolution and satisfies both by construction. A
 *     zero left pad writes NOTHING; a wrong output width writes a full surface sheared
 *     by one column per row. Both are refused here rather than computed wrong.
 *   - ONE PROGRAM DELIVERS 64 OUTPUT CHANNELS, not the float path's 32.
 *
 * Everything else is the direct int8 path's: the zero-point fold, the coefficient
 * group, the row window, the NC1HWC2 output and its de-scatter.
 * ==========================================================================*/
/* Output channels one program drives. 32 and 64 are measured — every live weight byte
 * of a 64-channel cube lands on exactly one output position of one channel — and above
 * that the split is the caller's, exactly as on the float path. [HW sweep] */
#define R76_ARGB_INT8_OC_MAX 64u

static int r76_conv_int8_argb(const char *entry, int fd, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *W, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out,
                              unsigned ow, unsigned oh)
{
    struct r76_conv_bos b = {0};
    uint64_t *ops = NULL;
    rocket_rk3576_row_task *rows = NULL;
    int32_t *A = NULL;
    int16_t *B = NULL;
    int64_t *sum_w = NULL;
    int8_t *wtile = NULL;
    unsigned IC = (unsigned)d->ic, OC = (unsigned)d->oc;
    /* ONE image channel is programmed as TWO. The int8 first conv writes nothing at
     * ic=1 — an untouched surface, not a wrong one — and what gates it is the feature
     * DMA's row width: the emitted `line_stride - 1` is correct from ic=2 up, and at
     * ic=1 the program only writes when that field is raised to `line_stride`, at
     * which point the DMA reads past the packed row and nothing is exact. So the row
     * is widened rather than the register: a second interleaved channel of zero
     * samples against zero weights. The arithmetic is untouched by it — the MAC term
     * is zero because the weight is, `sum_w` is unchanged so the coefficient A is,
     * and the asymmetric B multiplies a sum of RAW samples that gains only zeros — so
     * `taps` below stays the caller's count. The cost is one byte per pixel of host
     * packing and a doubled feature read. */
    unsigned ICP = ((unsigned)d->ic == 1u) ? 2u : (unsigned)d->ic;
    unsigned IH = (unsigned)d->ih, IW = (unsigned)d->iw;
    unsigned KH = (unsigned)d->kh, KW = (unsigned)d->kw;
    unsigned SY = (unsigned)d->stride_y, SX = (unsigned)d->stride_x;
    unsigned PT = (unsigned)d->pad_top,  PL = (unsigned)d->pad_left;
    unsigned surf_elems = rocket_rk3576_out_surf_elems(ow, oh, 0);
    unsigned tile = OC < R76_ARGB_INT8_OC_MAX ? OC : R76_ARGB_INT8_OC_MAX;
    unsigned taps = IC * KH * KW;
    unsigned oc0, nrow = 0, r, max_tasks = oh + 2u;
    size_t in_bytes;
    conv_params_t plan = {0};
    unsigned char stamp;
    struct r76_int8_prof prof = {0};
    struct r76_prof_shape shape;
    double t0;
    int rc;

    prof.on = r76_int8_prof_on();

    /* This path's four geometry bounds are jointly the ONNX symmetric-SAME convention, and
     * an explicit output extent is not one of them: an asymmetric pad is refused here rather
     * than programmed onto the first conv's own encoding. */
    if (r76_desc_asym(d)) {
        ROCKET_LOGE("%s: an output extent this descriptor did not derive is an asymmetric "
                    "pad, which the packed-image first conv does not claim\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }

    if (IW % 16u) {
        ROCKET_LOGE("%s: the first conv needs iw a multiple of 16 (iw=%u); its DDR row "
                    "stride and CBUF row are both counted in 16-byte granules\n",
                    entry, IW);
        return ROCKET_E_UNSUPPORTED;
    }
    /* The two bounds this path adds, both measured and both silent if violated. */
    if (PL == 0) {
        ROCKET_LOGE("%s: the int8 first conv needs a NON-ZERO left pad (pad_left=0). At "
                    "pad_left 0 the DPU writes nothing at all — an untouched surface, not "
                    "a wrong one — at every plane, stride and kernel. The fp16 form of "
                    "the same conv has no such bound: rocket_conv2d_fp16_rk3576()\n",
                    entry);
        return ROCKET_E_UNSUPPORTED;
    }
    /* ONE image channel writes nothing, where two, three and four are bit-exact. The
     * mode word's ARGB_IN nibble is 8 | (ic-1), so ic=1 is the one value that leaves
     * its low bits clear, and the fp16 form of the same conv computes at ic=1 — so
     * this is an int8-side gap rather than a property of the packed datapath. Refused
     * rather than left to write an untouched surface. */
    /* The OUTPUT width has its own granule, and it is not implied by the input's: at
     * ow 24 and 56 — both from an iw that is a multiple of 16 — output row 0 is exact
     * and every row after it is wrong, while ow 16, 32, 48, 64, 80, 112 are exact.
     * The direct path carries no such rule; this one comes with the channel fold. */
    if (ow % 16u || oh == 0u) {
        ROCKET_LOGE("%s: the int8 first conv needs ow a multiple of 16 (ow=%u from "
                    "iw=%u stride %u). At any other output width the first output row "
                    "is exact and every row after it is wrong\n", entry, ow, IW, SX);
        return ROCKET_E_UNSUPPORTED;
    }
    if (ow * SX != IW || oh * SY != IH) {
        ROCKET_LOGE("%s: the int8 first conv needs the SAME-padding output extent — "
                    "ow*stride_x == iw and oh*stride_y == ih (got %u*%u vs iw=%u, %u*%u "
                    "vs ih=%u). Any other output width writes a full surface SHEARED by "
                    "one column per row, with nothing to fault on\n",
                    entry, ow, SX, IW, oh, SY, IH);
        return ROCKET_E_UNSUPPORTED;
    }
    /* A NON-ZERO INPUT ZERO POINT COMPUTES A WRONG BORDER HERE, and only the border:
     * the interior is bit-exact at every zero point tried, the DIRECT path is exact
     * everywhere including its border at the same zero points and against the same
     * reference, and this path's border falls to about half exact the moment the zero
     * point leaves zero. So it is the padded taps and nothing else. CNA_PAD_CON1 is
     * live — driving it moves the count — but no single centring of it explains the
     * readings, so the constant's domain on this sub-encoding is not decoded and the
     * shape is refused rather than computed wrong.
     *
     * It was invisible until a graph asked for it: every first-conv shape in every gate
     * carries a zero input zero point, and a TFLite MobileNet's does too after the
     * uint8 rebase, so nothing was wrong in anything that had been run.
     * [HW sweep, H96 MAX M9, tests/rk3576_argb_pad.c] */
    if (in_zp != 0) {
        ROCKET_LOGE("%s: the int8 first conv computes a wrong BORDER at a non-zero "
                    "input zero point (in_zp=%d); its interior is exact and its padded "
                    "taps are not. Widen the image to eight channels and take the "
                    "direct path, which is exact at any zero point\n", entry, in_zp);
        return ROCKET_E_UNSUPPORTED;
    }
    if (OC != rocket_rk3576_pad_oc(OC)) {
        ROCKET_LOGE("%s: oc=%u is a partial 32-channel group and writes nothing; size "
                    "the output and coefficient buffers for %u channels and pass that "
                    "count (rocket_rk3576_pad_oc)\n",
                    entry, OC, rocket_rk3576_pad_oc(OC));
        return ROCKET_E_UNSUPPORTED;
    }

    in_bytes = (size_t)IH * IW * ICP;
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;

    ops   = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    rows  = calloc(max_tasks, sizeof *rows);
    sum_w = calloc(OC, sizeof *sum_w);
    wtile = calloc((size_t)tile * ICP * KH * KW, 1);
    if (!ops || !rows || !sum_w || !wtile) { rc = ROCKET_E_NOMEM; goto done; }

    /* Each output channel's whole filter, for the input zero point's fold. */
    t0 = R76_PT(prof);
    {
        unsigned c, i, y, x;
        for (c = 0; c < OC; c++) {
            int64_t s = 0;
            for (i = 0; i < IC; i++)
                for (y = 0; y < KH; y++)
                    for (x = 0; x < KW; x++)
                        s += W[(((size_t)c * IC + i) * KH + y) * KW + x];
            sum_w[c] = s;
        }
    }
    R76_ACC(prof, sums_us, t0);

    /* The row window, on the same planner as every other path. Told the precision
     * because the offsets come back in PACKED-IMAGE row units here — an int8 packed
     * image is `ic` interleaved bytes per pixel where a float one is halfwords. */
    plan.ic = (uint16_t)ICP; plan.ih = (uint16_t)IH; plan.iw = (uint16_t)IW;
    plan.oc = (uint16_t)tile; plan.oh = (uint16_t)oh; plan.ow = (uint16_t)ow;
    plan.kh = (uint16_t)KH; plan.kw = (uint16_t)KW;
    plan.stride_y = (uint8_t)SY; plan.stride_x = (uint8_t)SX;
    plan.pad_top = (uint8_t)PT; plan.pad_left = (uint8_t)PL;
    plan.ih_full = (uint16_t)IH; plan.oh_full = (uint16_t)oh;
    t0 = R76_PT(prof);
    if (rocket_rk3576_plan_rows_prec(&plan, 0, precision_int8, rows, max_tasks,
                                     &nrow) < 0 || !nrow) {
        ROCKET_LOGE("%s: no row plan for the first conv (ic=%u %ux%u k%ux%u)\n",
                    entry, IC, IW, IH, KW, KH);
        rc = ROCKET_E_UNSUPPORTED; goto done;
    }
    R76_ACC(prof, gen_us, t0);

    t0 = R76_PT(prof);
    if (rocket_bo_alloc(fd, in_bytes, &b.in) < 0 ||
        rocket_bo_alloc(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }

    /* CHW in, interleaved out — the packed image the CNA reads, shared by every tile.
     * The sample goes in as PLAIN TWO'S COMPLEMENT and the MAC reads it that way.
     *
     * THE CONVERTER'S OFFSET IS INERT, which is the opposite of what the datapath's
     * description says and is why the zero point is folded exactly as the direct path
     * folds it. The CVT registers are transcribed from captures that are all zero
     * point 0, so nothing ever exercised them; driven on the part, an image written at
     * raw = s + (zp + 0x80) comes back as the raw byte read as a signed int8 with no
     * subtraction at all — 0x80 reads -128, 0xC0 reads -64, 0xFF reads -1. So the
     * packed image is signed, and A carries the -in_zp*sum_w correction. */
    rocket_bo_prep(fd, &b.in, 1, 0);
    {
        int8_t *img = (int8_t *)b.in.ptr;
        unsigned c, y, x;
        for (y = 0; y < IH; y++)
            for (x = 0; x < IW; x++)
                for (c = 0; c < ICP; c++)
                    img[((size_t)y * IW + x) * ICP + c] =
                        c < IC ? in[((size_t)c * IH + y) * IW + x] : 0;
    }
    rocket_bo_fini(fd, &b.in);
    R76_ACC(prof, in_us, t0);

    for (oc0 = 0; oc0 < OC; oc0 += tile) {
        prof.tiles++;
        unsigned n = OC - oc0 < tile ? OC - oc0 : tile;
        size_t w_bytes = rocket_rk3576_weight_argb_int8_bytes(n, KH, KW);
        size_t coeff_bytes = rocket_rk3576_coeff_bytes(n);
        size_t obytes = (size_t)((n + C2 - 1u) / C2) * surf_elems * C2;
        conv_params_t p = {0};
        struct r76_task_extent e = {0};
        uint32_t in_h[4], out_h[1];

        t0 = R76_PT(prof);
        if (b.w.ptr)     rocket_bo_free(fd, &b.w);
        if (b.coeff.ptr) rocket_bo_free(fd, &b.coeff);
        if (b.out.ptr)   rocket_bo_free(fd, &b.out);
        memset(&b.w, 0, sizeof b.w);
        memset(&b.coeff, 0, sizeof b.coeff);
        memset(&b.out, 0, sizeof b.out);
        if (rocket_bo_alloc(fd, w_bytes, &b.w) < 0 ||
            rocket_bo_alloc(fd, coeff_bytes, &b.coeff) < 0 ||
            rocket_bo_alloc(fd, obytes, &b.out) < 0) { rc = ROCKET_E_NOMEM; goto done; }
        R76_ACC(prof, free_us, t0);

        /* This tile's channels renumbered from zero — its own whole convolution. */
        t0 = R76_PT(prof);
        if (ICP == IC) {
            memcpy(wtile, W + (size_t)oc0 * IC * KH * KW, (size_t)n * IC * KH * KW);
        } else {
            unsigned j, t;
            memset(wtile, 0, (size_t)tile * ICP * KH * KW);
            for (j = 0; j < n; j++)
                for (t = 0; t < IC * KH * KW; t++)
                    wtile[(size_t)j * ICP * KH * KW + t] =
                        W[(size_t)(oc0 + j) * IC * KH * KW + t];
        }
        rocket_bo_prep(fd, &b.w, 1, 0);
        rc = rocket_rk3576_argb_int8_pack_weights(b.w.ptr, w_bytes, wtile, n, ICP, KH, KW);
        rocket_bo_fini(fd, &b.w);
        if (rc < 0) { rc = ROCKET_E_SHAPE; goto done; }
        R76_ACC(prof, w_us, t0);

        t0 = R76_PT(prof);
        free(A); free(B);
        A = calloc(n, sizeof *A);
        B = w_zp ? calloc(n, sizeof *B) : NULL;
        if (!A || (w_zp && !B)) { rc = ROCKET_E_NOMEM; goto done; }
        r76_fold_coeff(A, bias, oc0, n, sum_w, in_zp, w_zp, taps, NULL);
        if (B) { unsigned j; for (j = 0; j < n; j++) B[j] = (int16_t)(-w_zp); }
        rocket_bo_prep(fd, &b.coeff, 1, 0);
        if (B) rocket_rk3576_pack_coeff_asym(b.coeff.ptr, coeff_bytes, A, n, B, 1);
        else   rocket_rk3576_pack_coeff(b.coeff.ptr, coeff_bytes, A, n);
        rocket_bo_fini(fd, &b.coeff);
        R76_ACC(prof, coeff_us, t0);

        p.ic = (uint16_t)ICP; p.iw = (uint16_t)IW; p.ih = (uint16_t)IH;
        p.oc = (uint16_t)n;  p.ow = (uint16_t)ow; p.oh = (uint16_t)oh;
        p.kh = (uint16_t)KH; p.kw = (uint16_t)KW;
        p.stride_y = (uint8_t)SY; p.stride_x = (uint8_t)SX;
        p.pad_top = (uint8_t)PT; p.pad_left = (uint8_t)PL;
        p.ih_full = (uint16_t)IH; p.oh_full = (uint16_t)oh;
        p.int8_out = 1;
        p.in_scale = in_scale; p.w_scale = w_scale; p.out_scale = out_scale;
        /* uint8-centered, as on the direct path: the emitter programs the border
         * constant as (input_zero_point + 0x80) & 0xFF, and that byte has to BE the
         * stored zero point, so that a pad tap's true value is zero. The converter
         * offset the same field feeds is inert (above), so it costs nothing. */
        p.input_zero_point  = in_zp + 0x80;
        p.output_zero_point = out_zp + 0x80;
        p.weight_zero_point = 0x80;
        p.tasks       = ops;
        p.input_dma   = b.in.dma_address;
        p.weights_dma = b.w.dma_address;
        p.bias_dma    = b.coeff.dma_address;
        p.output_dma  = b.out.dma_address;

        in_h[0] = b.in.handle; in_h[1] = b.w.handle;
        in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
        out_h[0] = b.out.handle;

        t0 = R76_PT(prof);
        if (stamp) {
            rocket_bo_prep(fd, &b.out, 1, 0);
            memset(b.out.ptr, stamp, obytes);
            rocket_bo_fini(fd, &b.out);
        }
        R76_ACC(prof, gen_us, t0);

        for (r = 0; r < nrow; r++) {
            conv_params_t q = p;
            prof.tasks++;
            t0 = R76_PT(prof);
            q.ih = rows[r].ih; q.oh = rows[r].oh;
            q.pad_top = rows[r].pad_top;
            q.input_dma  = p.input_dma  + rows[r].feature_off;
            q.output_dma = p.output_dma + rows[r].output_off;
            q.ih_full = (uint16_t)IH; q.oh_full = (uint16_t)oh;
            if (gen_conv2d_int8_rk3576(&q) != 0) {
                ROCKET_LOGE("%s: the generator refused the first-conv program (ic=%u "
                            "%ux%u k%ux%u oc %u..%u rows %u..%u)\n", entry, IC, IW, IH,
                            KW, KH, oc0, oc0 + n, rows[r].oy0, rows[r].oy0 + rows[r].oh);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }
            /* Per TASK, not per tile: one poisoned submit among several leaves its own
             * rows stale while its siblings are full. */
            e.groups      = (n + C2 - 1u) / C2;
            e.group_bytes = (size_t)surf_elems * C2;
            e.row_off     = (size_t)rows[r].oy0 * ow * C2;
            e.span        = (size_t)rows[r].oh * ow * C2;
            R76_ACC(prof, gen_us, t0);
            t0 = R76_PT(prof);
            rc = r76_submit_task(fd, &b, &q, ops, in_h, 4u, out_h, &e, stamp, 0u, entry);
            R76_ACC(prof, submit_us, t0);
            if (rc != ROCKET_OK) goto done;
        }

        t0 = R76_PT(prof);
        rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
        R76_ACC(prof, drain_us, t0);
        t0 = R76_PT(prof);
        {
            const int8_t *o = (const int8_t *)b.out.ptr;
            size_t px = (size_t)oh * ow;
            unsigned c, k;
            for (c = 0; c < n; c += C2) {
                unsigned m = n - c < C2 ? n - c : C2;
                int8_t *dp[C2];
                for (k = 0; k < m; k++) dp[k] = out + (size_t)(oc0 + c + k) * px;
                r76_c2_unpack(dp, m, o + (size_t)(c / C2) * surf_elems * C2, px);
            }
        }
        rocket_bo_fini(fd, &b.out);
        R76_ACC(prof, read_us, t0);
    }
    rc = ROCKET_OK;

done:
    t0 = R76_PT(prof);
    free(ops); free(rows); free(A); free(B); free(sum_w); free(wtile);
    r76_conv_free(fd, &b);
    R76_ACC(prof, free_us, t0);
    shape.ic = IC; shape.oc = OC; shape.iw = IW; shape.ih = IH;
    shape.kw = KW; shape.kh = KH; shape.sx = SX; shape.resident = 0;
    r76_int8_prof_log(entry, &shape, &prof);
    return rc;
}

int rocket_conv2d_int8_rk3576(int fd, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *W, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out)
{
    const char *entry = "rocket_conv2d_int8_rk3576";

    if (d && d->depthwise)
        return rocket_conv2d_dw_int8_rk3576(fd, d, in, W, bias, in_scale, w_scale,
                                            out_scale, in_zp, w_zp, out_zp, out);
    /* Four or fewer channels is the packed-image first conv: a different CNA program,
     * a different feature buffer and a different weight cube. `direct_datapath` is how a
     * caller asks for the ordinary one instead — cheaper resident, and with none of the
     * packed-image path's three geometry bounds. */
    if (d && d->ic <= 4 && !d->direct_datapath) {
        unsigned ow, oh;
        int rc = r76_conv_check(entry, fd, d, 0, 1, &ow, &oh);
        if (rc != ROCKET_OK) return rc;
        if (!in || !W || !out) return ROCKET_E_SHAPE;
        if (!(in_scale > 0.0f) || !(w_scale > 0.0f) || !(out_scale > 0.0f))
            return ROCKET_E_SHAPE;
        if (in_zp < -128 || in_zp > 127 || w_zp < -128 || w_zp > 127 ||
            out_zp < -128 || out_zp > 127)
            return ROCKET_E_SHAPE;
        return r76_conv_int8_argb(entry, fd, d, in, W, bias, in_scale, w_scale,
                                  out_scale, in_zp, w_zp, out_zp, out, ow, oh);
    }
    return r76_conv_int8_run(entry, fd, d, 0, in, W, bias,
                             in_scale, w_scale, NULL, out_scale,
                             in_zp, w_zp, out_zp, NULL, out);
}

unsigned rocket_conv2d_int8_perchannel_oc_tile_rk3576(const rocket_conv2d_desc *d,
                                                      const int8_t *W,
                                                      const int32_t *bias,
                                                      float in_scale,
                                                      const float *w_scale,
                                                      float out_scale, int in_zp)
{
    unsigned IC, OC, KH, KW, icreg, taps, cbuf_tile, tile = 0;
    int64_t *sum_w = NULL, *sum_abs_w = NULL;
    unsigned *perm = NULL;
    unsigned c, i, y, x;

    if (!d || !W || !w_scale) return 0;
    /* A depthwise layer is one task by construction — output channel c is bound to
     * input channel c, so there is no output-channel window to program. */
    if (d->depthwise) return (unsigned)d->oc;
    if (d->ic <= 4) return 0;
    IC = (unsigned)d->ic; OC = (unsigned)d->oc;
    KH = (unsigned)d->kh; KW = (unsigned)d->kw;
    icreg = rocket_rk3576_pad_ic(IC);
    /* The PROGRAMMED count, matching r76_w_prepare()'s fold — see the tap-count note
     * there. It only enters the A term, and a per-axis quantization has w_zp == 0, so the
     * two counts predict the same error; kept in agreement rather than left to differ. */
    taps  = icreg * KH * KW;
    cbuf_tile = r76_conv_oc_tile(icreg, KH, KW, rocket_rk3576_pad_oc(OC));
    if (!cbuf_tile) return 0;

    sum_w = calloc(OC, sizeof *sum_w);
    sum_abs_w = calloc(OC, sizeof *sum_abs_w);
    perm = calloc(OC, sizeof *perm);
    if (!sum_w || !sum_abs_w || !perm) goto done;
    for (c = 0; c < OC; c++) {
        int64_t s = 0, sa = 0;
        for (i = 0; i < IC; i++)
            for (y = 0; y < KH; y++)
                for (x = 0; x < KW; x++) {
                    int64_t v = W[(((size_t)c * IC + i) * KH + y) * KW + x];
                    s += v; sa += v < 0 ? -v : v;
                }
        sum_w[c] = s; sum_abs_w[c] = sa;
    }
    r76_sort_by_scale(perm, OC, w_scale);
    tile = r76_pc_oc_tile(OC, cbuf_tile, perm, sum_abs_w, bias, sum_w,
                          in_zp, 0, taps, in_scale, w_scale, out_scale);
done:
    free(sum_w); free(sum_abs_w); free(perm);
    return tile;
}

int rocket_conv2d_int8_perchannel_rk3576(int fd, const rocket_conv2d_desc *d,
                                         const int8_t *in, const int8_t *W,
                                         const int32_t *bias, float in_scale,
                                         const float *w_scale, float out_scale,
                                         int in_zp, int out_zp, int8_t *out)
{
    const char *entry = "rocket_conv2d_int8_perchannel_rk3576";

    if (!w_scale) return ROCKET_E_SHAPE;
    if (d && !d->depthwise && d->ic <= 4 && !d->direct_datapath) {
        ROCKET_LOGE("%s: four or fewer input channels is the packed-image first conv, "
                    "a different program whose coefficient path this planner does not "
                    "drive. rocket_conv2d_desc.direct_datapath runs the layer on the "
                    "direct one, which it does\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    return r76_conv_int8_run(entry, fd, d, d && d->depthwise, in, W, bias,
                             in_scale, 1.0f, w_scale, out_scale,
                             in_zp, 0, out_zp, NULL, out);
}

int rocket_conv2d_dw_int8_rk3576(int fd, const rocket_conv2d_desc *d,
                                 const int8_t *in, const int8_t *w, const int32_t *bias,
                                 float in_scale, float w_scale, float out_scale,
                                 int in_zp, int w_zp, int out_zp, int8_t *out)
{
    return r76_conv_int8_run("rocket_conv2d_dw_int8_rk3576", fd, d, 1, in, w, bias,
                             in_scale, w_scale, NULL, out_scale,
                             in_zp, w_zp, out_zp, NULL, out);
}

/* ============================================================================
 * SECTION — the residual add, as a convolution's weights
 *
 * The DPU's elementwise stage takes exactly ONE operand and no register in its
 * interface carries a second, so a skip connection is lowered onto the datapath this
 * part does have: concatenate the operands along channels and convolve with a 1x1
 * kernel of two diagonal blocks. What is chip knowledge in that lowering is the
 * WEIGHTS, and only the weights — the descriptor, the buffers and the submit are an
 * ordinary convolution the entries above already run — so that is what lives here.
 *
 *     out[o] = requant( w1*(a[o] - a_zp) + w2*(b[o] - b_zp) )
 *
 * THE RATIO IS THE WHOLE PROBLEM. The datapath applies one requant gain, so the two
 * operands' different scales have to ride in the weights as `w2/w1` — and w1, w2 are
 * int8. Anchoring the larger of the pair at 127 resolves the ratio to about one part in
 * 127; searching every w1 instead gives the best rational approximation with a
 * denominator that fits, which is one part in ~127^2. On MobileNetV2's ten skips that is
 * the difference between 1e-3 and 1e-5 relative, so the search is not a refinement, it
 * is most of the accuracy. Ties go to the LARGER w1, which is the finer accumulator.
 *
 * BOTH ZERO POINTS RIDE EXACTLY. The datapath subtracts one input zero point from every
 * channel, so the skip half is short by `w2*(a_zp - b_zp)` — a per-output-channel
 * constant, which is what the bias is. Nothing here is approximated except the ratio.
 * ==========================================================================*/
int rocket_residual_add_weights_rk3576(unsigned c, float a_scale, float b_scale,
                                       int a_zp, int b_zp, int8_t *W, int32_t *bias,
                                       float *w_scale, int *w_pair)
{
    const char *entry = "rocket_residual_add_weights_rk3576";
    double r, best_err = -1.0;
    int w1 = 1, w2 = 1, cand;
    unsigned o;

    if (!W || !bias || !w_scale || !c) return ROCKET_E_SHAPE;
    if (!(a_scale > 0.0f) || !(b_scale > 0.0f)) return ROCKET_E_SHAPE;

    /* The weight slice rule: this is a 1x1 convolution over 2c input channels, and one
     * program contracts at most 4608 of them. Past that the add needs splitting, which
     * is the caller's decision rather than a silently wrong answer. */
    if (2u * c > 4608u) {
        ROCKET_LOGE("%s: c=%u makes a %u-channel contraction and one program takes "
                    "4608, so this add does not fit in one convolution\n",
                    entry, c, 2u * c);
        return ROCKET_E_UNSUPPORTED;
    }

    r = (double)b_scale / (double)a_scale;
    if (r > 127.0 || r < 1.0 / 127.0) {
        ROCKET_LOGE("%s: the operand scales differ by %.3gx and a pair of int8 weights "
                    "spans 127x, so this ratio is not expressible\n", entry, r);
        return ROCKET_E_UNSUPPORTED;
    }

    /* Best rational approximation to `r` with both terms in [1,127]. */
    for (cand = 1; cand <= 127; cand++) {
        int other = (int)(r * cand + 0.5);
        double err;
        if (other < 1 || other > 127) continue;
        err = fabs((double)other / (double)cand - r) / r;
        if (best_err < 0.0 || err < best_err - 1e-15) {
            best_err = err; w1 = cand; w2 = other;
        } else if (err < best_err + 1e-15 && cand > w1) {
            best_err = err; w1 = cand; w2 = other;   /* the finer accumulator */
        }
    }

    memset(W, 0, (size_t)c * 2u * c);
    for (o = 0; o < c; o++) {
        W[(size_t)o * 2u * c + o]      = (int8_t)w1;
        W[(size_t)o * 2u * c + c + o]  = (int8_t)w2;
        bias[o] = w2 * (a_zp - b_zp);
    }
    /* The first operand's own gain must come out as one: the caller passes `a_scale` as
     * the convolution's input scale, so `a_scale * w_scale * w1 == a_scale`. */
    *w_scale = 1.0f / (float)w1;
    if (w_pair) { w_pair[0] = w1; w_pair[1] = w2; }
    return ROCKET_OK;
}

/*
 * THE FUSED FORM, AND WHY A REAL QUANTIZATION REFUSES IT.
 *
 * The standalone add above CHOOSES its weight scale — its weights are two diagonal blocks
 * rather than a quantized tensor, so `w_scale = 1/w1` is a free parameter and the operand
 * ratio only has to be a ratio of two int8s. A convolution that absorbs the skip into its
 * own kernel has no such freedom: its weight scale is the one its filter was quantized
 * with, so the skip's contribution must reach the output through the accumulator unit
 * `in_scale*w_scale` that the main term already uses. That fixes the identity weight, and
 * nothing else in the program can move it — the coefficient group's C multiplies the whole
 * accumulator, main term and skip together, so a per-channel gain cannot separate them.
 *
 * Writing it as the convolution's own requant gain makes the bound legible:
 *
 *     w_skip = skip_scale/(in_scale*w_scale) = (skip_scale/out_scale) / gain
 *
 * and a residual's two operands have comparable ranges, so `skip_scale/out_scale` is order
 * one and the fusion needs `gain > ~1/127`. A convolution contracting over hundreds of
 * channels has a gain one to two orders below that: MobileNetV2's ten project convolutions
 * run at 1/242 to 1/893 and want identity weights of 173 to 865.
 */
int rocket_residual_fuse_weight_rk3576(float in_scale, float w_scale, float skip_scale,
                                       int in_zp, int skip_zp,
                                       int *w_skip, int32_t *bias_delta, float *rel_err)
{
    const char *entry = "rocket_residual_fuse_weight_rk3576";
    double unit, want;
    int q;

    if (!w_skip || !bias_delta) return ROCKET_E_SHAPE;
    if (!(in_scale > 0.0f) || !(w_scale > 0.0f) || !(skip_scale > 0.0f))
        return ROCKET_E_SHAPE;

    unit = (double)in_scale * (double)w_scale;
    want = (double)skip_scale / unit;
    q = (int)(want < 0.0 ? want - 0.5 : want + 0.5);
    if (q < -127 || q > 127 || q == 0) {
        ROCKET_LOGE("%s: the skip's scale is %.4g and this convolution's accumulator unit "
                    "is %.4g, so the identity weight would be %.1f — the field is int8, so "
                    "the skip cannot ride in this convolution's kernel\n",
                    entry, (double)skip_scale, unit, want);
        return ROCKET_E_UNSUPPORTED;
    }
    *w_skip = q;
    /* The datapath subtracts ONE input zero point from every channel, the convolution's
     * own. The skip channels want their own, and the difference is a per-output-channel
     * constant — which is what the bias is, so it rides exactly. */
    *bias_delta = (int32_t)q * (int32_t)(in_zp - skip_zp);
    if (rel_err) *rel_err = (float)(fabs((double)q - want) / want);
    return ROCKET_OK;
}

/* ============================================================================
 * SECTION — the nonlinear activation, on the DPU LUT
 *
 * A LUT activation on this part is TWO programs in one job: a DPU-only one that bursts
 * the tables into the LUT RAM, and an ordinary convolution with the EW stage switched
 * from bypass onto the table. So the op the library exposes is a convolution whose
 * arithmetic is chosen to make the datapath value land on an exact table index.
 *
 * THE IDENTITY IS DEPTHWISE, and it has to be. A 1x1 direct conv would need an oc x ic
 * identity weight matrix — quadratic in the channel count, and against the emitter's
 * `ic*kh*kw <= 4608` weight-slice rule at any real width. Output channel c of a
 * depthwise task is bound to INPUT channel c, so the identity is one weight per channel
 * and the whole layer is ONE task whatever the channel count.
 *
 * WHY THE WEIGHT IS 2^sel AND NOT 1. The index step is 2^sel datapath units, so an
 * identity of 1 would put all 256 possible inputs inside eight index intervals and read
 * the activation off an interpolation. Scaling the accumulator by the step instead puts
 * input value q at index q - in_zp + 512 EXACTLY: every distinct int8 input gets its own
 * table entry, and the interpolation never runs. The same could be done with the BS
 * stage's per-channel C, but the weight is free — the coefficient A fold already carries
 * `-in_zp * sum_w`, so the input zero point comes out right with no special case.
 *
 * THE OUT_CVT IS THE ENTRY UNIT. The emitter programs `(v*MUL)>>SHIFT` from
 * in_scale*w_scale/out_scale, so the table is built in units of that gain rather than in
 * the op's own output scale: entry = f(x) / (out_scale * gain). Held at 1/128, which is
 * the finest gain whose int16 entry still spans the whole int8 output range the zero
 * point can ask for (128 * 255 = 32640).
 * ==========================================================================*/

/* The index step, as an exponent. 5 is the vendor's own span and the one the input map
 * is gated at; the weight below is 2^this, so it has to stay inside int8. */
#define R76_ACT_SEL   5u
/* in_scale*w_scale/out_scale for the identity conv. See "THE OUT_CVT IS THE ENTRY UNIT". */
#define R76_ACT_CONV_SCALE  (1.0f / 128.0f)

/* The gain the emitter will actually program for that conv scale — read back from the
 * one derivation rather than re-derived, because a table built against a slightly
 * different gain is a curve that is quietly off everywhere. */
static double r76_act_entry_gain(void)
{
    unsigned mul, shift;
    rocket_rk3576_requant_params(R76_ACT_CONV_SCALE, &mul, &shift);
    return (double)mul / ldexp(1.0, (int)shift);
}

int rocket_act_int8_rk3576(int fd, int kind, unsigned c, unsigned h, unsigned w,
                           const int8_t *in, float in_scale, int in_zp,
                           int8_t *out, float out_scale, int out_zp)
{
    const char *entry = "rocket_act_int8_rk3576";
    rocket_conv2d_desc d = {0};
    struct r76_lut lut;
    int16_t le[RK3576_LUT_ENTRIES], lo[RK3576_LUT_ENTRIES];
    int8_t *wt;
    unsigned i;
    int rc;

    if (!r76_is_this_chip(entry)) return ROCKET_E_UNSUPPORTED;
    if (!in || !out || !c || !h || !w) return ROCKET_E_SHAPE;
    if (!(in_scale > 0.0f) || !(out_scale > 0.0f)) return ROCKET_E_SHAPE;
    if (!rocket_rk3576_act_name(kind)) {
        ROCKET_LOGE("%s: activation kind %d has no table. exp, gelu and sqrt are absent "
                    "deliberately — no vendor capture emits a 513-entry table for any of "
                    "them, so they lower some other way\n", entry, kind);
        return ROCKET_E_UNSUPPORTED;
    }

    /* The table's own two scales. A datapath value v means the real number
     * `in_scale * v / 2^sel`, and an entry is read in units of `out_scale * gain`. */
    if (rocket_rk3576_lut_build(kind,
                                (double)in_scale / (double)(1u << R76_ACT_SEL),
                                (double)out_scale * r76_act_entry_gain(),
                                R76_ACT_SEL, le, lo, &lut.w) != 0)
        return ROCKET_E_SHAPE;
    lut.le = le; lut.lo = lo;

    wt = malloc(c);
    if (!wt) return ROCKET_E_NOMEM;
    for (i = 0; i < c; i++) wt[i] = (int8_t)(1u << R76_ACT_SEL);

    d.ic = (int)c; d.oc = (int)c;
    d.ih = (int)h; d.iw = (int)w;
    d.kh = 1; d.kw = 1;
    d.stride_y = 1; d.stride_x = 1;
    d.depthwise = 1;
    rc = r76_conv_int8_run(entry, fd, &d, 1, in, wt, NULL,
                           1.0f, 1.0f, NULL, 1.0f / R76_ACT_CONV_SCALE,
                           in_zp, 0, out_zp, &lut, out);
    free(wt);
    return rc;
}

/* ============================================================================
 * SECTION — a convolution with the activation FUSED into its epilogue
 *
 * The standalone op above invents its own convolution to reach the LUT. A real one
 * already has one, and the LUT sits in the EW stage of the same program, so fusing costs
 * nothing but the table-load task — the activation becomes free.
 *
 * IT IS NOT THE SAME ARITHMETIC AS conv-THEN-act, AND IT IS THE BETTER ONE. The standalone
 * op reads the convolution's REQUANTIZED int8 output, so its input alphabet is 256 values;
 * the fused form reads the ACCUMULATOR, before any rounding. What it gives up in exchange
 * is exactness: the standalone form guarantees each of its 256 inputs its own table entry,
 * and a real accumulator lands between entries, so the hardware's linear interpolation
 * runs and the result is a reported quantity rather than an asserted one.
 *
 * THREE THINGS THE CALLER DOES NOT HAVE TO SUPPLY, because they follow from the scales it
 * already has:
 *
 *   THE VALUE UNIT. With a per-tensor requant the BS multiplier C is unity, so the value
 *   the LUT sees is the accumulator itself — `A` included, which is where the bias and the
 *   zero-point fold already are — and one count of it means `in_scale*w_scale`.
 *
 *   THE ENTRY UNIT. OUT_CVT scales the table's output by the gain the emitter derives from
 *   `in_scale*w_scale/out_scale`, so an entry is read in units of `out_scale*gain` and the
 *   table is built in those. The standalone op has to PIN its conv scale at 1/128 to make
 *   this well defined; a real conv has the model's own three and needs no such contract
 *   change.
 *
 *   THE STEP. The window is the symmetric `[-512*2^sel, 512*2^sel)`, so `sel` is the
 *   smallest one whose real half-span covers the domain the activation still varies over.
 *   Past that the LUT returns its endpoint clamp, which is the right answer for a
 *   SATURATING kind and the wrong one for a kind that keeps growing — for those the bound
 *   is where the int8 OUTPUT saturates instead, since f(x) is x by then.
 *
 * Two bounds are refused rather than computed wrong: a value unit too fine for the table
 * to hold the peak in an int16, and a domain no `sel` reaches.
 * ==========================================================================*/

/* The real half-span the window has to cover. */
static double r76_act_domain(int kind, float out_scale, int out_zp)
{
    double hi = 127.0 - (double)out_zp, lo = (double)out_zp + 128.0;
    double sat = (hi > lo ? hi : lo) * (double)out_scale;
    switch (kind) {
    case ROCKET_RK3576_ACT_HARDSIGMOID: return 3.0;
    case ROCKET_RK3576_ACT_HARDSWISH:   return sat > 3.0 ? sat : 3.0;
    case ROCKET_RK3576_ACT_SWISH:
    case ROCKET_RK3576_ACT_ELU:         return sat > 8.0 ? sat : 8.0;
    default:                            return 8.0;    /* sigmoid, tanh: bounded by 1 */
    }
}

/* The largest |f| the table has to hold, for the int16 entry check. The unbounded kinds
 * reach the edge of their own domain, since they are x there. */
static double r76_act_peak(int kind, double domain)
{
    switch (kind) {
    case ROCKET_RK3576_ACT_SIGMOID:
    case ROCKET_RK3576_ACT_TANH:
    case ROCKET_RK3576_ACT_HARDSIGMOID: return 1.0;
    default:                            return domain;
    }
}

/*
 * A PER-AXIS FUSED ACTIVATION IS NOT OFFERED, AND THE ORDER IT WOULD TAKE IS DECIDED.
 *
 * The two features want the same register. `C` divides the value unit per output channel,
 * which is both how a per-axis requant buys its gain resolution (one OUT_CVT shift serves a
 * whole task, so a per-channel SCALE is expressible and a per-channel SHIFT is not) and how
 * each channel's accumulator range is placed inside the LUT's window [LE_START, LO_END). The
 * two can disagree, and nothing here arbitrates: the entry takes no per-channel scales.
 *
 * There is no measured consumer to arbitrate for. The per-tensor fused form is already
 * within ONE count of an exact f(), and MobileNet's per-axis graph is within one count per
 * layer with no fused activation anywhere — so the combination would be built for a model
 * that has not been run. What is owed is the decision, so that the next person does not
 * have to re-derive it under pressure:
 *
 *   THE LUT WINDOW WINS, AND A CONFLICT REFUSES RATHER THAN TRADES.
 *
 * The two failure modes are not comparable. A value outside the window takes a CLAMP, which
 * is a wrong answer with no bound on its size. A suboptimal `C` costs counts of requant
 * resolution, which is a graded loss the planner already predicts and reports. So the
 * ordering is: place the window first (pick `C` per channel from the accumulator range the
 * activation still varies over), then evaluate the gain error that `C` implies against
 * ROCKET_RK3576_PC_MAX_ERR, and REFUSE if it does not meet the target — leaving the caller
 * to shrink the output-channel tile, which is the lever that buys resolution back, or to
 * drop the fusion and run rocket_act_int8_rk3576() after the convolution. The planner must
 * not silently pick the better-looking of the two.
 */
int rocket_conv2d_int8_act_rk3576(int fd, const rocket_conv2d_desc *d,
                                  const int8_t *in, const int8_t *W, const int32_t *bias,
                                  float in_scale, float w_scale, float out_scale,
                                  int in_zp, int w_zp, int out_zp,
                                  int kind, int8_t *out)
{
    const char *entry = "rocket_conv2d_int8_act_rk3576";
    struct r76_lut lut;
    int16_t le[RK3576_LUT_ENTRIES], lo[RK3576_LUT_ENTRIES];
    double value_scale, entry_scale, domain, peak;
    unsigned mul, shift, sel;

    if (!r76_is_this_chip(entry)) return ROCKET_E_UNSUPPORTED;
    if (!d || !in || !W || !out) return ROCKET_E_SHAPE;
    if (!(in_scale > 0.0f) || !(w_scale > 0.0f) || !(out_scale > 0.0f))
        return ROCKET_E_SHAPE;
    if (!rocket_rk3576_act_name(kind)) {
        ROCKET_LOGE("%s: activation kind %d has no table\n", entry, kind);
        return ROCKET_E_UNSUPPORTED;
    }
    /* The packed-image first conv is a different program and does not carry the EW
     * stage's LUT window; it has its own entry and no LUT descriptor reaches it. The
     * direct datapath does carry it, at any channel count. */
    if (!d->depthwise && d->ic <= 4 && !d->direct_datapath) {
        ROCKET_LOGE("%s: four or fewer input channels is the packed-image first conv, "
                    "which this path does not drive — set "
                    "rocket_conv2d_desc.direct_datapath, or run it plain and follow it "
                    "with rocket_act_int8_rk3576()\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }

    value_scale = (double)in_scale * (double)w_scale;
    rocket_rk3576_requant_params((float)(value_scale / (double)out_scale), &mul, &shift);
    entry_scale = (double)out_scale * ((double)mul / ldexp(1.0, (int)shift));
    if (!(entry_scale > 0.0)) {
        ROCKET_LOGE("%s: in_scale*w_scale/out_scale (%g) requantizes to a zero gain, so "
                    "the table has no unit to be read in\n",
                    entry, value_scale / (double)out_scale);
        return ROCKET_E_SHAPE;
    }

    domain = r76_act_domain(kind, out_scale, out_zp);
    peak   = r76_act_peak(kind, domain);
    for (sel = 0; sel <= 15u; sel++)
        if (512.0 * (double)(1u << sel) * value_scale >= domain) break;
    if (sel > 15u) {
        ROCKET_LOGE("%s: one accumulator count is %g of an output unit, so even the "
                    "widest window (sel 15) spans %g and %s still varies over %g. The "
                    "value unit is in_scale*w_scale; a coarser weight scale is what "
                    "widens it\n", entry, value_scale,
                    512.0 * 32768.0 * value_scale, rocket_rk3576_act_name(kind), domain);
        return ROCKET_E_UNSUPPORTED;
    }
    if (peak / entry_scale > 32767.0) {
        ROCKET_LOGE("%s: %s peaks at %g and an entry is read in units of %g, so the "
                    "table would need %.0f and saturates at 32767 — the curve would be "
                    "flat-topped rather than folded, which is why this is refused. The "
                    "entry unit tracks in_scale*w_scale (%g); a coarser one fixes it\n",
                    entry, rocket_rk3576_act_name(kind), peak, entry_scale,
                    peak / entry_scale, value_scale);
        return ROCKET_E_UNSUPPORTED;
    }

    if (rocket_rk3576_lut_build(kind, value_scale, entry_scale, sel, le, lo, &lut.w) != 0)
        return ROCKET_E_SHAPE;
    lut.le = le; lut.lo = lo;

    return r76_conv_int8_run(entry, fd, d, d->depthwise ? 1 : 0, in, W, bias,
                             in_scale, w_scale, NULL, out_scale,
                             in_zp, w_zp, out_zp, &lut, out);
}

rocket_conv2d_int8_weights_rk3576 *
rocket_conv2d_int8_pack_rk3576(int fd, const rocket_conv2d_desc *d,
                               const int8_t *W, const int32_t *bias,
                               float in_scale, float w_scale, const float *w_scale_oc,
                               float out_scale, int in_zp, int w_zp, int out_zp)
{
    const char *entry = "rocket_conv2d_int8_pack_rk3576";
    r76_w *h = NULL;
    unsigned t;
    int dw = d && d->depthwise;

    if (!W) return NULL;
    /* The packed-image first conv is a different program with a different weight cube, and
     * this handle drives neither. It is refused here rather than at the first inference so
     * a caller learns it while it can still fall back.
     *
     * THE DIRECT DATAPATH IS THE ONE TO HOLD, and a narrow-ic layer reaches it with the
     * flag. Residency is where it wins: the packed-image form's advantage is its smaller
     * feature read, and a resident handle has already collected the BO churn and the
     * weight repack that dominated the direct one. */
    if (d && !dw && d->ic <= 4 && !d->direct_datapath) {
        ROCKET_LOGE("%s: four or fewer input channels takes the packed-image first conv, "
                    "whose cube this handle does not pack — set "
                    "rocket_conv2d_desc.direct_datapath to hold this layer on the direct "
                    "path, or call rocket_conv2d_int8_rk3576() per inference for it\n",
                    entry);
        return NULL;
    }
    if (r76_w_prepare(entry, fd, d, dw, bias, in_scale, w_scale, w_scale_oc,
                      out_scale, in_zp, w_zp, out_zp, NULL, &h) != ROCKET_OK)
        return NULL;

    h->resident = 1;
    r76_w_sums(h, W);
    if (r76_w_tiles(h) != ROCKET_OK) { r76_w_free(h); return NULL; }

    for (t = 0; t < h->ntile; t++) {
        struct r76_int8_prof quiet = {0};
        if (r76_wtile_pack(entry, h, t, W, &quiet) != ROCKET_OK) {
            r76_w_free(h);
            return NULL;
        }
    }
    if (h->w_scale_oc)
        ROCKET_LOGI("%s: per-channel requant, worst-case gain error %.3g%% over %u "
                    "output-channel tile(s) of %u\n",
                    entry, h->worst_rel_err * 100.0, h->ntile, h->oc_tile);
    return h;
}

int rocket_conv2d_int8_prepacked_rk3576(int fd, rocket_conv2d_int8_weights_rk3576 *h,
                                        const int8_t *in, int8_t *out)
{
    const char *entry = "rocket_conv2d_int8_prepacked_rk3576";

    if (!h) return ROCKET_E_SHAPE;
    /* Each end is required only when it is the end this handle uses: cube I/O replaces the
     * caller's tensor, it does not sit alongside it. */
    if ((!in && !h->cube_in) || (!out && !h->cube_out)) return ROCKET_E_SHAPE;
    /* A BO belongs to the file that created it and its IOVA is per-fd, so a handle run on
     * a foreign fd would hand the NPU addresses that mean nothing in this context — and
     * the submit would succeed. */
    if (fd != h->fd) {
        ROCKET_LOGE("%s: this handle was packed on fd %d and its BOs live there; an IOVA "
                    "is per-fd, so it cannot be run on fd %d\n", entry, h->fd, fd);
        return ROCKET_E_SHAPE;
    }
    {
        struct r76_int8_prof prof = {0};
        struct r76_prof_shape shape;
        int rc;
        prof.on = r76_int8_prof_on();
        rc = r76_int8_exec(entry, h, NULL, in, out, &prof);
        r76_prof_shape_of(&shape, h);
        r76_int8_prof_log(entry, &shape, &prof);
        return rc;
    }
}

/* ---- cube I/O between consecutive layers ------------------------------------
 * A convolution's output surface and a feature cube are the same object laid out the same
 * way, so layer n's surface is layer n+1's cube byte for byte whenever the plane, the
 * channel rounding and the channel-group STRIDE agree. What that removes is the pair of
 * host transposes at the join, which on MobileNetV1-224 is the largest host cost left.
 *
 * The stride is the one of those three that is not a geometric bound: a direct convolution
 * writes `ow*oh` elements per group and a depthwise one `round4(ow*oh)`, and the consumer's
 * DDR group jump is a REGISTER the emitter fills (`0x1094`, in 16-byte atoms) which the
 * part honours at any value at or above the plane [HW sweep, H96 MAX M9,
 * tests/rk3576_surf_stride.c]. So the cube carries its stride and the consumer is told it.
 *
 * The bounds below are refusals rather than fix-ups, because each is a shape where the two
 * buffers are NOT the same object:
 *
 *   MORE THAN ONE OUTPUT-CHANNEL TILE. Each tile owns its own surface BO and several BOs
 *   are not one cube. (Nothing in a MobileNet reaches this: the weight slice a 1x1 layer
 *   needs is 32 KiB against the 144 KiB that leaves the tile at the whole layer.)
 *
 *   AN INPUT CHANNEL COUNT THAT IS NOT A MULTIPLE OF 32. The handle's own cube would round
 *   the count up and rely on the ZERO it memset there; a producer's surface has whatever
 *   the DPU wrote into the padded output channels, which is not zero. Refusing keeps the
 *   arithmetic identical to the row-major path rather than approximately so.
 * ==========================================================================*/
static int r76_cube_shape_ok(const char *entry, const r76_w *h)
{
    if (h->ntile != 1u) {
        ROCKET_LOGE("%s: this handle splits its %u output channels across %u tiles, and "
                    "each tile owns its own surface — several buffers are not one cube\n",
                    entry, h->OC, h->ntile);
        return 0;
    }
    return 1;
}

int rocket_conv2d_int8_cube_of_rk3576(const rocket_conv2d_int8_weights_rk3576 *h,
                                     rocket_rk3576_cube *out)
{
    const char *entry = "rocket_conv2d_int8_cube_of_rk3576";

    if (!h || !out) return ROCKET_E_SHAPE;
    if (!r76_cube_shape_ok(entry, h)) return ROCKET_E_UNSUPPORTED;
    if (!h->out_ext.ptr && !h->tile[0].out.ptr) {
        ROCKET_LOGE("%s: this handle's output surface is not allocated yet\n", entry);
        return ROCKET_E_SHAPE;
    }
    memset(out, 0, sizeof *out);
    out->fd = h->fd;
    out->c = h->OC;
    out->h = h->oh; out->w = h->ow;
    out->groups = h->tile[0].tail_zp
                    ? rocket_rk3576_pad_oc(h->tile[0].ocreg) / C2
                    : (h->tile[0].ocreg + C2 - 1u) / C2;
    out->surf_elems = h->surf_elems;
    out->bo = h->out_ext.ptr ? h->out_ext : h->tile[0].out;
    out->off = h->out_ext.ptr ? h->out_off : 0u;
    /* THE CONSTANT TAIL, and it is a different fact on the two datapaths. The direct path
     * is programmed with round32(OC) and the DPU writes every group it is told; a partial
     * group's channels carry zero weights and a zero A term, so their accumulator is zero
     * and the OUT_CVT sends them to the output zero point. The depthwise path is programmed
     * with the RAW count and never writes past it, so the tail it declares is one the HOST
     * put there: r76_surf_bytes() gives such a tile the deeper surface and fills the group
     * past its own with the output zero point at pack time. Where that fill did not happen
     * — a tail lying inside a written group, or a caller's buffer this handle does not own
     * — the only tail it can declare is an EMPTY one, its output channels filling the last
     * group exactly. [HW sweep, H96 MAX M9, tests/rk3576_pad_channels.c: 12544 of 12544
     * elements at the output zero point on the direct path at oc 16/24/144 and k 1/3, and
     * 12544 of 12544 still holding the stamp on the depthwise one — which is what says the
     * fill survives.] */
    if (!h->dw || h->tile[0].tail_zp || out->groups * C2 == out->c) {
        out->pad_from = h->OC;
        out->pad_value = h->out_zp;
    }
    return ROCKET_OK;
}

int rocket_conv2d_int8_cube_in_rk3576(rocket_conv2d_int8_weights_rk3576 *h,
                                     const rocket_rk3576_cube *src)
{
    const char *entry = "rocket_conv2d_int8_cube_in_rk3576";

    if (!h) return ROCKET_E_SHAPE;
    if (!src) {
        h->cube_in = 0;
        h->src_off = 0;
        h->src_surf_elems = 0;
        memset(&h->src, 0, sizeof h->src);
        return ROCKET_OK;
    }
    if (src->fd != h->fd) {
        ROCKET_LOGE("%s: the cube belongs to fd %d and this handle to fd %d; an IOVA is "
                    "per-fd\n", entry, src->fd, h->fd);
        return ROCKET_E_SHAPE;
    }
    /* AN UNALIGNED INPUT CHANNEL COUNT IS A CONSTRAINT ON THE WEIGHT ZERO POINT, NOT ON
     * THE CHANNEL COUNT — and it is met whenever the cube says what its tail holds. This
     * handle's own cube fills the channels past `ic` with the border constant, its input
     * zero point, and folds their contribution over the REGISTER tap count. A producer's
     * surface carries whatever the DPU wrote into its own padded OUTPUT channels instead,
     * so the question is only whether that content reaches the accumulator. Their weights
     * are the zeros the cube was memset with, so their MAC contribution is zero whatever
     * they hold. What is NOT zero at a non-zero weight zero point is the B term: the DPU
     * adds B*sum(x) over every tap of the whole programmed group, padding channels
     * included.
     *
     * So the refusal is B's, and there are two ways past it. A symmetric weight
     * quantization — every TFLite int8 filter and every per-axis one by construction — has
     * no B term at all. And a cube that DECLARES a constant tail equal to this handle's
     * input zero point holds exactly what the fold already assumes, which is what a direct
     * producer's surface does at a graph join: its partial output group lands on its own
     * output zero point, and that is the consumer's input zero point. */
    if (h->IC % 32u && h->w_zp &&
        (!src->pad_from || src->pad_from > h->IC || src->pad_value != h->in_zp)) {
        ROCKET_LOGE("%s: %u input channels is not a multiple of 32 and the weight zero "
                    "point is %d, so the coefficient group's B term sums the padding "
                    "channels this handle fills with its border constant %d — and this "
                    "cube %s\n", entry, h->IC, h->w_zp, h->in_zp,
                    !src->pad_from ? "does not say what its tail holds"
                                   : (src->pad_from > h->IC
                                        ? "declares a tail that starts past them"
                                        : "declares a tail of a different constant"));
        return ROCKET_E_UNSUPPORTED;
    }
    if (src->c != h->IC || src->h != h->IH || src->w != h->IW) {
        ROCKET_LOGE("%s: the cube is %ux%ux%u and this handle reads %ux%ux%u\n",
                    entry, src->c, src->h, src->w, h->IC, h->IH, h->IW);
        return ROCKET_E_SHAPE;
    }
    /* A PADDED GROUP STRIDE IS A REGISTER, NOT A GEOMETRIC BOUND. The CNA's DDR group
     * jump (0x1094) is emitted rather than fixed, and the part honours a value larger
     * than the plane: bit-exact at +3, +16 and +64 elements over five geometries, at one
     * row task and at ten, with the task-window stride (0x1098) left derived
     * [HW sweep, H96 MAX M9, tests/rk3576_surf_stride.c]. That is what lets a DEPTHWISE
     * producer's round4(ow*oh) surface be a direct consumer's input cube — MobileNetV2's
     * four 7x7 joins, 52 elements against a plane of 49. A stride SHORTER than the plane
     * would make the groups overlap and is not a layout anything here produces. */
    if (src->surf_elems < (size_t)h->IH * h->IW) {
        ROCKET_LOGE("%s: the cube's channel-group stride is %zu elements, shorter than "
                    "the %u this handle's feature DMA walks\n",
                    entry, src->surf_elems, h->IH * h->IW);
        return ROCKET_E_SHAPE;
    }
    if (src->surf_elems > 0xFFFFFFFFu / C2) return ROCKET_E_SHAPE;
    /* The handle's own cube is icpad channels deep — round32 of the register count — and
     * the feature DMA reads every group of it. A shorter buffer would be read past its
     * end, and a SLICE has to leave room for those groups past its own offset. */
    if ((size_t)src->groups * C2 < h->icpad ||
        src->bo.size < src->off + (size_t)src->groups * src->surf_elems * C2) {
        ROCKET_LOGE("%s: the cube carries %u channel group(s) of %zu bytes at byte %zu of "
                    "a %zu-byte buffer, and this handle's feature DMA walks %u channels\n",
                    entry, src->groups, src->surf_elems * C2, src->off, src->bo.size,
                    h->icpad);
        return ROCKET_E_SHAPE;
    }
    /* A group offset is a plain address on this part [HW sweep, tests/rk3576_offset_cube.c],
     * so a cube that is a slice of a bigger buffer needs nothing but the addition. */
    if (src->off % ((size_t)src->surf_elems * C2)) {
        ROCKET_LOGE("%s: the cube starts at byte %zu, which is not a whole channel group "
                    "of %zu bytes into its buffer\n",
                    entry, src->off, (size_t)src->surf_elems * C2);
        return ROCKET_E_SHAPE;
    }
    h->src = src->bo;
    h->src_off = src->off;
    h->cube_in = 1;
    /* Told to the emitter only when it is NOT the plane, so every join that was already
     * accepted emits the program it emitted before. */
    h->src_surf_elems = src->surf_elems == (size_t)h->IH * h->IW
                            ? 0u : (unsigned)src->surf_elems;
    /* The handle's own feature cube is dead weight now. Dropped rather than kept, because
     * a graph holding one per layer is megabytes of it. */
    if (h->in.ptr) rocket_bo_free(h->fd, &h->in);
    memset(&h->in, 0, sizeof h->in);
    return ROCKET_OK;
}

/* ---- a surface that is a SLICE of a caller's buffer -------------------------
 * The two entries below are what a residual topology wants. A cube chain breaks wherever
 * the host has to touch a tensor between two programs, and on MobileNetV2 two thirds of
 * those places are a channel CONCATENATION the host builds: the two operands of an add,
 * and a skip a later layer reads beside a convolution's own output. Neither needs a copy
 * if the producers write into slices of one allocation — the surfaces ARE the operand.
 *
 * What that costs is OWNERSHIP, not encoding: the offset is a plain address the hardware
 * honours on either base [HW sweep, tests/rk3576_offset_cube.c], and all that changes here
 * is that a handle can be told to write somewhere it did not allocate.
 * ==========================================================================*/
int rocket_rk3576_cube_alloc(int fd, unsigned c, unsigned h, unsigned w,
                             rocket_rk3576_cube *out)
{
    const char *entry = "rocket_rk3576_cube_alloc";
    unsigned groups = (c + C2 - 1u) / C2;

    if (!out || !c || !h || !w) return ROCKET_E_SHAPE;
    memset(out, 0, sizeof *out);
    if (rocket_bo_alloc(fd, (size_t)groups * h * w * C2, &out->bo) < 0) {
        ROCKET_LOGE("%s: %u channel(s) of a %ux%u plane could not be allocated\n",
                    entry, c, w, h);
        return ROCKET_E_NOMEM;
    }
    /* A cube a producer has not written yet is not zero — it is whatever the pages carry —
     * and a consumer reading a slice nobody filled would compute on that silently. The
     * bracket is the rule, never a bare memset: dirty lines race the DPU's own DMA. */
    rocket_bo_prep(fd, &out->bo, 1, 0);
    memset(out->bo.ptr, 0, (size_t)groups * h * w * C2);
    rocket_bo_fini(fd, &out->bo);
    out->fd = fd;
    out->c = c; out->h = h; out->w = w;
    out->groups = groups;
    out->surf_elems = (size_t)h * w;
    out->off = 0;
    return ROCKET_OK;
}

void rocket_rk3576_cube_free(int fd, rocket_rk3576_cube *buf)
{
    if (!buf || !buf->bo.ptr) return;
    if (fd != buf->fd) {
        ROCKET_LOGE("rocket_rk3576_cube_free: this buffer belongs to fd %d, not %d\n",
                    buf->fd, fd);
        return;
    }
    rocket_bo_free(fd, &buf->bo);
    memset(buf, 0, sizeof *buf);
}

int rocket_rk3576_cube_slice(const rocket_rk3576_cube *buf, unsigned c0, unsigned c,
                             rocket_rk3576_cube *out)
{
    const char *entry = "rocket_rk3576_cube_slice";

    if (!buf || !out || !c) return ROCKET_E_SHAPE;
    /* A CHANNEL GROUP is the unit an address can name: sixteen channels share every
     * sixteen-byte atom, so a slice starting inside one is not addressable at all. */
    if (c0 % C2) {
        ROCKET_LOGE("%s: channel %u is not a group boundary; a slice starts every %u\n",
                    entry, c0, C2);
        return ROCKET_E_SHAPE;
    }
    if ((size_t)c0 + c > (size_t)buf->groups * C2) {
        ROCKET_LOGE("%s: channels [%u, %u) run past the %u this buffer carries\n",
                    entry, c0, c0 + c, buf->groups * C2);
        return ROCKET_E_SHAPE;
    }
    *out = *buf;
    out->c = c;
    out->groups = (c + C2 - 1u) / C2;
    out->off = buf->off + (size_t)(c0 / C2) * buf->surf_elems * C2;
    /* A SLICE'S TAIL IS NOT ITS PARENT'S. What lies past a slice's live channels is
     * whichever producer owns the next slice, so the parent's declaration says nothing
     * about it and carrying it forward would be a claim nobody made. */
    out->pad_from = 0;
    out->pad_value = 0;
    return ROCKET_OK;
}

int rocket_conv2d_int8_cube_out_at_rk3576(rocket_conv2d_int8_weights_rk3576 *h,
                                          const rocket_rk3576_cube *dst)
{
    const char *entry = "rocket_conv2d_int8_cube_out_at_rk3576";
    unsigned need;
    size_t bytes;

    if (!h) return ROCKET_E_SHAPE;
    if (!dst) {
        /* BACK TO OWNING A SURFACE, AND IT HAS TO BE ALLOCATED HERE. A packed tile is
         * never packed again, so leaving this to the next call would leave the handle
         * with no surface at all — and the first thing that reaches for one is a
         * cache-maintenance ioctl on handle 0. */
        memset(&h->out_ext, 0, sizeof h->out_ext);
        h->out_off = 0;
        h->cube_out = 0;
        h->tile[0].stamped = 0;
        if (h->tile && h->tile[0].w.ptr && !h->tile[0].out.ptr) {
            size_t obytes = (size_t)((h->tile[0].ocreg + C2 - 1u) / C2) *
                            h->surf_elems * C2;
            size_t surf_bytes = r76_surf_bytes(h, h->tile[0].ocreg);
            h->tile[0].tail_zp = 0;
            if (rocket_bo_alloc(h->fd, surf_bytes, &h->tile[0].out) < 0) {
                ROCKET_LOGE("%s: this handle's own output surface could not be allocated "
                            "again\n", entry);
                return ROCKET_E_NOMEM;
            }
            if (surf_bytes > obytes) {
                rocket_bo_prep(h->fd, &h->tile[0].out, 1, 0);
                memset((char *)h->tile[0].out.ptr + obytes,
                       (unsigned char)(int8_t)h->out_zp, surf_bytes - obytes);
                rocket_bo_fini(h->fd, &h->tile[0].out);
                h->tile[0].tail_zp = 1;
            }
        }
        return ROCKET_OK;
    }
    if (!r76_cube_shape_ok(entry, h)) return ROCKET_E_UNSUPPORTED;
    if (dst->fd != h->fd) {
        ROCKET_LOGE("%s: the buffer belongs to fd %d and this handle to fd %d; an IOVA is "
                    "per-fd\n", entry, dst->fd, h->fd);
        return ROCKET_E_SHAPE;
    }
    if (dst->h != h->oh || dst->w != h->ow ||
        dst->surf_elems != (size_t)h->surf_elems) {
        ROCKET_LOGE("%s: the slice is %ux%u at a %zu-element stride and this handle writes "
                    "%ux%u at %u\n", entry, dst->w, dst->h, dst->surf_elems,
                    h->ow, h->oh, h->surf_elems);
        return ROCKET_E_SHAPE;
    }
    /* The REGISTER channel count, not the caller's: the DPU writes every group it is
     * programmed with, so a slice sized to the live channels alone would have its tail
     * written over the next producer's. */
    need = (h->tile[0].ocreg + C2 - 1u) / C2;
    bytes = (size_t)need * h->surf_elems * C2;
    if (dst->groups < need || dst->bo.size < dst->off + bytes) {
        ROCKET_LOGE("%s: this handle writes %u channel group(s) (%zu bytes) and the slice "
                    "carries %u at byte %zu of a %zu-byte buffer\n",
                    entry, need, bytes, dst->groups, dst->off, dst->bo.size);
        return ROCKET_E_SHAPE;
    }
    if (dst->off % ((size_t)h->surf_elems * C2)) {
        ROCKET_LOGE("%s: the slice starts at byte %zu, which is not a whole channel group "
                    "of %zu bytes into its buffer\n",
                    entry, dst->off, (size_t)h->surf_elems * C2);
        return ROCKET_E_SHAPE;
    }
    h->out_ext = dst->bo;
    h->out_off = dst->off;
    h->cube_out = 1;
    /* The handle's own surface is dead weight now — the point of the shared buffer is to
     * pay for one allocation and not two. */
    if (h->tile[0].out.ptr) rocket_bo_free(h->fd, &h->tile[0].out);
    memset(&h->tile[0].out, 0, sizeof h->tile[0].out);
    /* Its sentinel goes on at the START of the next call, like every other cube-out
     * surface: this one is read as an input before this handle runs again. */
    h->tile[0].stamped = 0;
    return ROCKET_OK;
}

int rocket_conv2d_int8_cube_shared_rk3576(rocket_conv2d_int8_weights_rk3576 *h, int on)
{
    if (!h) return ROCKET_E_SHAPE;
    h->shared_out = on ? 1 : 0;
    return ROCKET_OK;
}

int rocket_conv2d_int8_cube_out_rk3576(rocket_conv2d_int8_weights_rk3576 *h, int on)
{
    const char *entry = "rocket_conv2d_int8_cube_out_rk3576";

    if (!h) return ROCKET_E_SHAPE;
    if (!on) { h->cube_out = 0; return ROCKET_OK; }
    if (h->out_ext.ptr) { h->cube_out = 1; return ROCKET_OK; }
    if (!r76_cube_shape_ok(entry, h)) return ROCKET_E_UNSUPPORTED;
    h->cube_out = 1;
    /* The stamp rides the de-scatter's bracket, and there is no de-scatter now: this
     * surface is read as an input before this handle runs again, so its sentinel has to be
     * written at the start of the next call and not at the end of this one. Clearing the
     * flag is what puts it back there. */
    h->tile[0].stamped = 0;
    return ROCKET_OK;
}

void rocket_conv2d_int8_weights_free_rk3576(int fd,
                                            rocket_conv2d_int8_weights_rk3576 *h)
{
    if (!h) return;
    if (fd != h->fd)
        ROCKET_LOGW("rocket_conv2d_int8_weights_free_rk3576: freeing a handle packed on "
                    "fd %d through fd %d; its BOs are released on the fd that owns "
                    "them\n", h->fd, fd);
    r76_w_free(h);
}


/* ============================================================================
 * SECTION — a run of cube-linked layers as ONE hardware kick
 *
 * Once the host transposes are gone the SUBMIT is the largest single bucket: on a resident
 * cube-chained MobileNetV1-224 it is 29 jobs for 29 layers, ~190 us each against 130-250 us
 * of actual compute. Chaining a TILE's row tasks already collects the same floor n-1 times
 * per call; this collects it across LAYERS.
 *
 * WHAT MAKES IT SOUND is two measured facts and one structural one:
 *
 *   A chained stream HONOURS READ-AFTER-WRITE between its programs. The PC advances on a
 *   program's OP_ENABLE and the next program's feature read sees the previous one's output
 *   surface — over five planes from 2 KB to 387 KB, with the second program's cube being the
 *   first's surface exactly as here (tests/rk3576_chain_raw.c). That was the question the
 *   row-task chain never had to ask, because row tasks write disjoint rows.
 *
 *   A CUBE LINK means there is no host work between two layers at all — no de-scatter, no
 *   scatter, no cache maintenance — so there is nothing that has to happen between the two
 *   programs. The link is therefore the eligibility test, and it is one the caller has
 *   already made: this constructor asserts it rather than re-deriving it.
 *
 *   The regcmd stream is a function of the frozen geometry and the BO addresses, both fixed
 *   at pack time. So it is generated ONCE here instead of per call. (As an optimization on
 *   its own that was measured at 0.13 ms of 33.6 and reverted; as the mechanism for this it
 *   is not optional.)
 *
 * THE POISON GUARD STILL ASKS EVERY TASK. A wide-output job anywhere on the system poisons
 * the next submit, and a chain is one submit — but a dead program inside it leaves its own
 * surface stale, and the layer after it then computes on the sentinel and writes a perfectly
 * plausible surface. So "did the last layer write" proves nothing about the middle of the
 * stream and every surface is checked. The check costs no extra cache maintenance: the
 * sentinel already needs a PREP_BO/FINI_BO pair per surface per call, and the check rides
 * it — read, verify, re-stamp, done. A failure redoes the WHOLE chain after a power cycle,
 * because a chain is one kick and there is no way to restart it in the middle.
 *
 * That pair per surface is the largest cost left on a whole graph, and asking only the last
 * layer is what a caller trades correctness coverage for it: r76_guard_per_kick().
 * ==========================================================================*/

/* HOW LONG A CROSS-LAYER STREAM MAY BE, and it is WHERE THE KERNEL STARTS ITS COMPLETION
 * WAIT that decides — not a hardware bound on the stream.
 *
 * The RK3576 driver retires a job when the writing block raises its own completion and,
 * failing that, on a deadline (`dpu_grace_us`, 500 us), and it starts that wait at PC_DONE.
 * PC_DONE is raised per TASK: on a chained stream the two PC_DONE bits alternate as the
 * program counter retires each program, so the wait starts a few tens of microseconds into
 * a stream that runs for milliseconds and the deadline is covering the rest of the stream
 * rather than a drain. Past it the job retires with programs still to run — the caller
 * reads a surface that still holds the sentinel, and the next job is programmed into a core
 * that has not finished the last one. On MobileNetV1-224 that is exactly where a 24-layer /
 * 35-program run stopped working while a 23-layer / 34-program one did not, and the
 * deadline each stream needed was not a function of its program count [HW sweep, H96 MAX
 * M9]:
 *
 *   34 programs (23 layers)   under 200 us      35 programs (24 layers)   650 - 700 us
 *   30 programs (23 layers)   200 - 500 us      220 programs (44 layers)  under 150 us
 *
 * From interface 1.4 the driver reads PC_TASK_STATUS — a live count of the tasks a kick has
 * started and completed — and holds the wait off until the whole kick has retired, so there
 * is nothing to cap: the ceiling is then this file's own array bound, and a chain of 220
 * programs is bit-exact. Below it the cap stays where MobileNet's own mix stopped working,
 * because that kernel gives no way to ask for more. (Scaling the deadline by the task count
 * instead is a fitted number that a stream of slower programs outruns — one 224x224 k3
 * convolution at 256 channels row-splits into 75 tasks and retires with 8 of them left.)
 *
 * ROCKET_RK3576_CHAIN_MAX_TASKS overrides either. [HW sweep, H96 MAX M9] */
#define R76_CHAIN_TASKS_LEGACY 34u

static unsigned r76_chain_task_cap(void)
{
    static unsigned cached = 0;
    if (!cached) {
        const char *e = getenv("ROCKET_RK3576_CHAIN_MAX_TASKS");
        long v = (e && *e) ? strtol(e, NULL, 0) : 0;
        unsigned def = rocket_batch_completion_tracked() ? R76_MAX_CHAIN_TASKS
                                                         : R76_CHAIN_TASKS_LEGACY;
        cached = (v >= 2 && v <= (long)R76_MAX_CHAIN_TASKS) ? (unsigned)v : def;
    }
    return cached;
}

/* One layer's place in the stream: where its programs start, and what to check afterwards.
 * `h` and `p` are the two node kinds and exactly one is set — a pool is its own register
 * program here rather than a convolution epilogue, and it runs inside the stream. */
struct r76_chain_layer {
    r76_w   *h;
    rocket_pool_int8_rk3576_handle *p;
    struct rocket_rk3576_pool_link  pl;   /* the pool's frozen view; unused for a conv */
    unsigned task0;      /* index of this layer's first task in the job */
    unsigned ntask;
};

struct rocket_conv2d_int8_chain_rk3576 {
    int       fd;
    unsigned  n;                       /* layers */
    unsigned  ntask;                   /* tasks across all of them */
    struct r76_chain_layer *layer;
    rocket_bo rc;                      /* every program, contiguous and linked */
    rocket_task_desc *td;              /* one descriptor per task, frozen */
    /* Where each task's own output rows live, so "did THIS task write" can be asked of
     * exactly them. Owned rather than read off the handles: the per-layer path fills the
     * same array on the handle from the same plan, and relying on the two agreeing is a
     * coupling with nothing asserting it. */
    struct r76_task_extent *ext;
    uint32_t *in_h, *out_h;
    unsigned  n_in, n_out;
    unsigned  kicks;                   /* hardware kicks the last run took */
};

typedef struct rocket_conv2d_int8_chain_rk3576 r76_chain;

/* Fill `p` with the conv_params a handle's single tile runs at — the same values
 * r76_int8_exec() builds, so a chained program is the program the per-layer path would have
 * emitted. Kept beside the chain rather than shared with the exec loop because the loop
 * builds it inside its own tile iteration and has the profiling hooks interleaved. */
static void r76_chain_params(const r76_w *h, conv_params_t *p)
{
    const struct r76_int8_wtile *s = &h->tile[0];

    memset(p, 0, sizeof *p);
    p->ic = (uint16_t)h->icreg; p->ih = (uint16_t)h->IH; p->iw = (uint16_t)h->IW;
    p->oc = (uint16_t)s->ocreg; p->oh = (uint16_t)h->oh; p->ow = (uint16_t)h->ow;
    p->kh = (uint16_t)h->KH; p->kw = (uint16_t)h->KW;
    p->stride_y = (uint8_t)h->SY; p->stride_x = (uint8_t)h->SX;
    p->pad_top  = (uint8_t)h->PT; p->pad_left = (uint8_t)h->PL;
    p->ih_full = (uint16_t)h->IH; p->oh_full = (uint16_t)h->oh;
    p->in_surf_elems = h->src_surf_elems;
    p->int8_out = 1;
    if (h->w_scale_oc) {
        p->in_scale = s->base_scale; p->w_scale = 1.0f; p->out_scale = 1.0f;
    } else {
        p->in_scale = h->in_scale; p->w_scale = h->w_scale; p->out_scale = h->out_scale;
    }
    p->input_zero_point  = h->in_zp  + 0x80;
    p->output_zero_point = h->out_zp + 0x80;
    p->weight_zero_point = 0x80;
    p->input_dma   = (uint32_t)r76_feat_dma(h);
    p->weights_dma = s->w.dma_address;
    p->bias_dma    = s->coeff.dma_address;
    p->output_dma  = (uint32_t)r76_surf_dma(h, 0);
}

/* ---- THE TWO NODE KINDS ------------------------------------------------------------
 * A run is described by nodes rather than by convolution handles, because a pool is its
 * own register program on this part and one runs inside a convolution stream [HW sweep,
 * tests/rk3576_chain_pool.c]. Everything the layout and the eligibility test ask of a
 * layer goes through these six accessors, so there is one set of rules rather than one
 * per kind. A pool's fields come from rocket_rk3576_pool_link(), which is where the
 * pooling entry states them; nothing here reaches into that handle.
 *
 * A node with neither pointer set is the caller saying "this layer cannot be in a run",
 * which is what a NULL entry means in the convolution-only form. */
static int r76_node_is_pool(const rocket_chain_node_rk3576 *nd)
{
    return nd && !nd->conv && nd->pool;
}

static int r76_node_set(const rocket_chain_node_rk3576 *nd)
{
    return nd && ((nd->conv && !nd->pool) || (!nd->conv && nd->pool));
}

/* A pool handle's link view, rebuilt on demand. Cheap — it regenerates 31 register writes
 * from frozen geometry — and pure, so the finder stays free of hardware and of state. */
static int r76_node_pool_link(const rocket_chain_node_rk3576 *nd,
                              struct rocket_rk3576_pool_link *pl)
{
    uint64_t ops[RK3576_POOL_TASK_OPS];
    return rocket_rk3576_pool_link(nd->pool, pl, ops) == ROCKET_OK;
}

static int r76_node_cube_in(const rocket_chain_node_rk3576 *nd)
{
    struct rocket_rk3576_pool_link pl;
    if (!r76_node_set(nd)) return 0;
    if (!r76_node_is_pool(nd)) return nd->conv->cube_in;
    return r76_node_pool_link(nd, &pl) && pl.cube_in;
}

static int r76_node_cube_out(const rocket_chain_node_rk3576 *nd)
{
    struct rocket_rk3576_pool_link pl;
    if (!r76_node_set(nd)) return 0;
    if (!r76_node_is_pool(nd)) return nd->conv->cube_out;
    return r76_node_pool_link(nd, &pl) && pl.cube_out;
}

static uint64_t r76_node_feat_dma(const rocket_chain_node_rk3576 *nd)
{
    struct rocket_rk3576_pool_link pl;
    if (!r76_node_is_pool(nd)) return r76_feat_dma(nd->conv);
    return r76_node_pool_link(nd, &pl) ? pl.feat_dma : 0;
}

static uint64_t r76_node_surf_dma(const rocket_chain_node_rk3576 *nd)
{
    struct rocket_rk3576_pool_link pl;
    if (!r76_node_is_pool(nd)) return r76_surf_dma(nd->conv, 0);
    return r76_node_pool_link(nd, &pl) ? pl.surf_dma : 0;
}

/* Why a handle cannot be part of a chain, or NULL if it can. */
static const char *r76_chain_why_not(const r76_w *h)
{
    if (!h->resident)
        return "it is not a resident handle, so its operand BOs are not held across calls";
    if (h->ntile != 1u)
        return "its output channels are split across more than one tile, and each tile "
               "owns its own surface — the layout would need a fan-out the PC has no "
               "encoding for";
    if (h->has_lut)
        return "it applies a DPU LUT, whose table load is a task in a BO of its own and "
               "not something this contiguous layout carries";
    if (!h->tile || !h->tile[0].w.ptr || !h->tile[0].coeff.ptr ||
        (!h->tile[0].out.ptr && !h->out_ext.ptr))
        return "its tile was never packed";
    return NULL;
}

/* THE LINK, AGAINST THE ADDRESSES. Two handles can both carry the flags and still be
 * linked to someone else, and with slices the address is a base plus an offset — a
 * consumer reading the WHOLE of a concatenation buffer is not linked to the producer that
 * writes its second half, even though both name the same BO.
 *
 * THE RUN'S RULE IS WEAKER THAN THE PAIR'S, AND IT IS THE PART THAT SETS IT. A chained
 * stream honours read-after-write between its programs [HW sweep, 40/40,
 * tests/rk3576_chain_raw.c], so what a stream needs is not that layer j reads layer j-1's
 * surface but that it reads a surface some EARLIER program of the same stream wrote. A
 * ResNet block that changes width is exactly the difference: its 1x1 downsample reads the
 * block input three layers back and writes a slice of the add's buffer, so a pairwise test
 * strands it between two kicks even though no host work happens on either side of it.
 *
 * Sound because the chain emits its programs in ARRAY ORDER, so "an earlier layer" is "an
 * earlier program": the surface is written before it is read, in the same order the
 * per-layer path would have written it. The chain removes host work between layers and
 * nothing else, so the whole question is whether any host work is left — the consumer must
 * take a cube (no scatter) and the producer must leave one (no de-scatter). Requiring an
 * IN-RUN producer rather than merely a stable buffer is the conservative half: an operand
 * no layer of this stream writes is a BO the job never names, and the layout's dedup is
 * built on every intermediate surface being some member's output.
 *
 * A superset of the pairwise test — i == j-1 is the pair — so nothing that chains today
 * stops chaining. */
static int r76_run_fed(const rocket_chain_node_rk3576 *nd, unsigned first, unsigned j)
{
    unsigned i;

    if (!r76_node_cube_in(&nd[j])) return 0;
    for (i = first; i < j; i++)
        if (r76_node_cube_out(&nd[i]) && r76_node_surf_dma(&nd[i]) ==
                                         r76_node_feat_dma(&nd[j])) return 1;
    return 0;
}

/* How many programs a handle contributes to a stream: its row plan's task count. Pure —
 * the plan is a function of the frozen geometry and the planner writes into a scratch
 * copy, so asking does not disturb the handle. 0 if the handle has no plan at all, which
 * a caller must treat as unchainable rather than as one program. */
unsigned rocket_conv2d_int8_programs_rk3576(const rocket_conv2d_int8_weights_rk3576 *h)
{
    conv_params_t p, q;
    unsigned nt = 1u;

    if (!h || r76_chain_why_not(h)) return 0;
    r76_chain_params(h, &p);
    q = p;
    if (rocket_rk3576_plan_rows(&q, h->dw, h->plan, h->max_tasks, &nt) < 0) return 0;
    return nt;
}

unsigned rocket_conv2d_int8_chain_max_programs_rk3576(void)
{
    return r76_chain_task_cap();
}

/* Why a node cannot be part of a run, or NULL if it can. `interior` says whether it would
 * sit strictly inside the run — which is the only place a POOL may go, because one that
 * began a run would need its cube scattered into and one that ended a run would need its
 * output de-scattered, and both are the host work the stream exists to remove. */
static const char *r76_node_why_not(const rocket_chain_node_rk3576 *nd, int interior)
{
    struct rocket_rk3576_pool_link pl;

    if (!r76_node_set(nd))
        return "it is neither a resident convolution nor a resident pool";
    if (!r76_node_is_pool(nd)) return r76_chain_why_not(nd->conv);
    if (!r76_node_pool_link(nd, &pl))
        return "its register program could not be built, so it was never packed or its "
               "geometry is not one this part pools";
    if (!interior)
        return "it is a pooling layer at the START or the END of the run, where the chain "
               "would have to scatter into its cube or de-scatter its output — host work "
               "between programs is exactly what a stream removes, so a pool may only be "
               "an interior node";
    if (!pl.cube_in || !pl.cube_out)
        return "it is an interior pooling layer that does not both read and leave a cube";
    return NULL;
}

unsigned rocket_chain_node_programs_rk3576(const rocket_chain_node_rk3576 *node)
{
    if (!r76_node_set(node)) return 0;
    /* A pool is ONE program whatever its plane: the PPU has no row window. Asked as an
     * interior node, which is the only place one is legal — a caller checking a node at
     * the end of a run gets the refusal from the constructor. */
    if (r76_node_is_pool(node)) return r76_node_why_not(node, 1) ? 0u : 1u;
    return rocket_conv2d_int8_programs_rk3576(node->conv);
}

/*
 * THE RUN FINDER. The chain primitive needed hardware knowledge; grouping already-linked
 * handles into maximal runs does not, and every frontend that owns a graph would otherwise
 * write the same thirty lines. So the library keeps the primitive and gains this, and does
 * NOT gain a graph-execution object: the frontends already own scheduling, and the one
 * thing they cannot own is which streams the part will run.
 *
 * A run is a maximal sequence of nodes in which every layer but the last leaves a cube and
 * every layer but the first reads a cube some EARLIER member of the same run wrote — not
 * necessarily the one before it. Asserted against the BO ADDRESSES exactly as the
 * constructor does, since two handles can both carry the flags and still be linked to
 * someone else, and see r76_run_fed() for why the weaker rule is the part's. That is what
 * lets a block which changes width stay in one stream: its 1x1 shortcut reads the block
 * input several layers back, so a pairwise rule strands it. An UNSET node breaks a run,
 * which is how a caller says "this layer is not a resident convolution or pool" (a host op)
 * or "a run may not START here" (a layer whose input the caller prepares on the host: a
 * chain scatters the tensor it is handed and cannot know a widened or bordered copy was
 * meant).
 *
 * A POOL node is carried but never starts or ends a run, so the finder trims one off
 * either end rather than refusing the run — which is what keeps a classifier's trailing
 * `pool -> fc` inside the stream while its trailing pool alone is not a run at all.
 *
 * The PROGRAM-COUNT bound is applied here rather than left to a build-and-retry loop,
 * because a layer is one or several programs and the longest legal run is therefore not a
 * function of the layer count. A run that would exceed the cap is SPLIT at a layer
 * boundary, so the caller gets several legal runs instead of one refusal.
 *
 * Pure: no hardware, no allocation, nothing on the handles is written. Returns the number
 * of runs of two or more layers found, and fills up to `max_runs` of them.
 */
unsigned rocket_chain_plan_rk3576(const rocket_chain_node_rk3576 *nd, unsigned n,
                                  rocket_conv2d_int8_run_rk3576 *runs, unsigned max_runs)
{
    unsigned i, found = 0, cap = r76_chain_task_cap();

    if (!nd || n < 2u) return 0;
    for (i = 0; i + 1u < n; ) {
        unsigned first = i, count = 0, tasks = 0;
        /* A pool cannot START a run, so skip past one rather than opening a run on it —
         * the run that matters begins at the layer after. */
        if (r76_node_is_pool(&nd[i])) { i++; continue; }
        while (i < n && !r76_node_why_not(&nd[i], i > first)) {
            unsigned nt = rocket_chain_node_programs_rk3576(&nd[i]);
            if (!nt) break;
            /* One layer past the cap ends the run rather than shortening it below two:
             * a single layer of more programs than the cap is simply not chainable, and
             * the caller runs it through the per-layer path. */
            if (count && tasks + nt > cap) break;
            tasks += nt;
            count++;
            if (i + 1u >= n) { i++; break; }
            /* Growing the run makes nd[i] a non-last layer, so it must leave a cube; and
             * nd[i+1] must read one that a member already wrote. See r76_run_fed(). */
            if (!r76_node_cube_out(&nd[i]) || !r76_run_fed(nd, first, i + 1u)) {
                i++;
                break;
            }
            i++;
        }
        /* A run that ENDS on a pool gives that layer back: the chain would have to
         * de-scatter it, and the layer before is a legal last. */
        while (count && r76_node_is_pool(&nd[first + count - 1u])) {
            count--;
            tasks -= rocket_chain_node_programs_rk3576(&nd[first + count]);
        }
        if (count >= 2u) {
            if (runs && found < max_runs) {
                runs[found].first = first;
                runs[found].count = count;
                runs[found].programs = tasks;
            }
            found++;
        }
        if (i == first) i++;      /* a layer that could not start a run */
    }
    return found;
}

/* The convolution-only form: the same finder with every node a `conv`. One implementation,
 * because two would be a second set of rules to keep in agreement. */
unsigned rocket_conv2d_int8_chain_plan_rk3576(rocket_conv2d_int8_weights_rk3576 *const *h,
                                              unsigned n,
                                              rocket_conv2d_int8_run_rk3576 *runs,
                                              unsigned max_runs)
{
    rocket_chain_node_rk3576 *nd;
    unsigned i, found;

    if (!h || n < 2u) return 0;
    nd = calloc(n, sizeof *nd);
    if (!nd) return 0;
    for (i = 0; i < n; i++) nd[i].conv = h[i];
    found = rocket_chain_plan_rk3576(nd, n, runs, max_runs);
    free(nd);
    return found;
}

static int r76_h_has(const uint32_t *list, unsigned n, uint32_t handle)
{
    unsigned i;
    for (i = 0; i < n; i++) if (list[i] == handle) return 1;
    return 0;
}

static void r76_h_add(uint32_t *list, unsigned *n, uint32_t handle)
{
    if (!r76_h_has(list, *n, handle)) list[(*n)++] = handle;
}

/* The surface bytes one layer's tile owns. */
static size_t r76_chain_obytes(const r76_w *h)
{
    return (size_t)((h->tile[0].ocreg + C2 - 1u) / C2) * h->surf_elems * C2;
}

void rocket_conv2d_int8_chain_free_rk3576(int fd, rocket_conv2d_int8_chain_rk3576 *c)
{
    if (!c) return;
    if (fd != c->fd)
        ROCKET_LOGW("rocket_conv2d_int8_chain_free_rk3576: freeing a chain built on fd %d "
                    "through fd %d; its regcmd BO is released on the fd that owns it\n",
                    c->fd, fd);
    if (c->rc.ptr) rocket_bo_free(c->fd, &c->rc);
    free(c->layer); free(c->td); free(c->ext); free(c->in_h); free(c->out_h);
    free(c);
}

rocket_conv2d_int8_chain_rk3576 *
rocket_chain_new_rk3576(int fd, const rocket_chain_node_rk3576 *nd, unsigned n)
{
    const char *entry = "rocket_chain_new_rk3576";
    r76_chain *c;
    unsigned i, words = 0;
    size_t *off = NULL;
    uint32_t *cnt = NULL;
    uint64_t *all = NULL;

    if (!r76_is_this_chip(entry)) return NULL;
    if (!nd || n < 2u) {
        ROCKET_LOGE("%s: a chain is two or more layers; one layer is "
                    "rocket_conv2d_int8_prepacked_rk3576()\n", entry);
        return NULL;
    }
    if (!rocket_batched_submit_supported()) {
        ROCKET_LOGE("%s: this kernel does not honor DRM_ROCKET_JOB_BATCHED (needs "
                    "patches/rk3576/npu/0015-0016, and rocket_batch_submit must not be 0), "
                    "so a chained layout would run down the per-task path and stall\n",
                    entry);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        struct rocket_rk3576_pool_link pl;
        const char *why;
        int node_fd;
        if (!r76_node_set(&nd[i])) {
            ROCKET_LOGE("%s: layer %u names neither a convolution nor a pool, or both\n",
                        entry, i);
            return NULL;
        }
        node_fd = r76_node_is_pool(&nd[i])
                      ? (r76_node_pool_link(&nd[i], &pl) ? pl.fd : -1)
                      : nd[i].conv->fd;
        if (node_fd != fd) {
            ROCKET_LOGE("%s: layer %u was packed on fd %d and this chain is on fd %d; an "
                        "IOVA is per-fd\n", entry, i, node_fd, fd);
            return NULL;
        }
        if ((why = r76_node_why_not(&nd[i], i > 0u && i + 1u < n))) {
            ROCKET_LOGE("%s: layer %u cannot be chained: %s\n", entry, i, why);
            return NULL;
        }
    }
    /* THE LINK IS THE ELIGIBILITY TEST, AND IT IS OVER THE RUN RATHER THAN OVER THE PAIRS.
     * Between two chained layers nothing may happen on the host, which is two statements:
     * every layer but the last leaves its surface in place (no de-scatter), and every layer
     * but the first reads a cube some EARLIER member already wrote (no scatter). Asserted
     * against the ADDRESSES, not against the flags — two handles can both be flagged and
     * still be linked to someone else. See r76_run_fed() for why an earlier member, and not
     * the immediately preceding one, is what the part requires. */
    for (i = 0; i + 1u < n; i++) {
        if (!r76_node_cube_out(&nd[i])) {
            ROCKET_LOGE("%s: layer %u is not the last and does not leave a cube, so its "
                        "output would need a host de-scatter that a chain cannot insert "
                        "between two programs. Set it with "
                        "rocket_conv2d_int8_cube_out_rk3576() / "
                        "rocket_pool_int8_cube_out_rk3576()\n", entry, i);
            return NULL;
        }
    }
    for (i = 1u; i < n; i++) {
        if (!r76_run_fed(nd, 0, i)) {
            ROCKET_LOGE("%s: layer %u reads 0x%llx, which no earlier layer of this run "
                        "writes (cube_in=%d), so its input would need a host scatter. Link "
                        "it to its producer with rocket_conv2d_int8_cube_of_rk3576 / "
                        "_cube_in_ first\n",
                        entry, i, (unsigned long long)r76_node_feat_dma(&nd[i]),
                        r76_node_cube_in(&nd[i]));
            return NULL;
        }
    }

    /* THE FIRST LAYER'S FEATURE CUBE HAS TO EXIST NOW. The per-layer path allocates it
     * lazily on the first inference, and this constructor FREEZES its IOVA into every
     * program that reads it — so a chain built before that layer ever ran would address
     * IOVA 0 and compute on whatever lives there, writing a full and entirely wrong
     * surface. Nothing downstream could detect that: the task wrote. The first layer is
     * always a convolution: a pool may only be interior. */
    if (!nd[0].conv->cube_in && !nd[0].conv->in.ptr) {
        r76_w *h0 = nd[0].conv;
        if (rocket_bo_alloc(fd, h0->in_bytes, &h0->in) < 0) {
            ROCKET_LOGE("%s: the first layer's feature cube could not be allocated\n",
                        entry);
            return NULL;
        }
        rocket_bo_prep(fd, &h0->in, 1, 0);
        memset(h0->in.ptr, (unsigned char)(int8_t)h0->in_zp, h0->in_bytes);
        rocket_bo_fini(fd, &h0->in);
    }

    c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->fd = fd;
    c->n = n;
    c->layer = calloc(n, sizeof *c->layer);
    if (!c->layer) { rocket_conv2d_int8_chain_free_rk3576(fd, c); return NULL; }

    /* Plan every layer's rows first, so the total task count and the stream length are known
     * before anything is allocated. The plan lives on the handle, which is where the
     * per-layer path keeps it too. */
    for (i = 0; i < n; i++) {
        conv_params_t p, q;
        unsigned nt = 1u;
        r76_w *hi = nd[i].conv;
        if (!hi) {
            /* A pool is ONE program whatever its plane: the PPU has no row window. Its
             * link view is taken below, where the program it describes is generated, so
             * the layout, the extents and the guard all read one set of addresses. */
            c->layer[i].p = nd[i].pool;
            c->layer[i].task0 = c->ntask;
            c->layer[i].ntask = 1u;
            c->ntask += 1u;
            continue;
        }
        r76_chain_params(hi, &p);
        q = p;
        if (rocket_rk3576_plan_rows(&q, hi->dw, hi->plan, hi->max_tasks, &nt) < 0) {
            ROCKET_LOGE("%s: no row plan for layer %u (ic=%u %ux%u k%ux%u s%u)\n",
                        entry, i, hi->IC, hi->IW, hi->IH, hi->KW, hi->KH, hi->SX);
            rocket_conv2d_int8_chain_free_rk3576(fd, c);
            return NULL;
        }
        c->layer[i].h = hi;
        c->layer[i].task0 = c->ntask;
        c->layer[i].ntask = nt;
        c->ntask += nt;
    }
    if (c->ntask > r76_chain_task_cap()) {
        ROCKET_LOGE("%s: %u program(s) over %u layer(s) is past the %u a MIXED-geometry "
                    "stream is measured good to on this part — past it a kick intermittently "
                    "writes nothing at all, and which layer reads unwritten varies. Shorten "
                    "the run, or raise ROCKET_RK3576_CHAIN_MAX_TASKS to measure a different "
                    "mix (a uniform 38-program stream is exact)\n",
                    entry, c->ntask, n, r76_chain_task_cap());
        rocket_conv2d_int8_chain_free_rk3576(fd, c);
        return NULL;
    }

    c->td   = calloc(c->ntask, sizeof *c->td);
    off     = calloc(c->ntask, sizeof *off);
    cnt     = calloc(c->ntask, sizeof *cnt);
    /* Every program, host-side, so the stream length is known before the BO is sized. Its
     * own buffer rather than the handles' `ops` scratch: that one is sized from the handle's
     * `multi` flag and its regcmd BO is sized to match, so growing it here would leave a
     * later per-layer call writing max_tasks programs into a one-program BO. A chain must
     * not change how the handles behave on their own — they stay callable through
     * rocket_conv2d_int8_prepacked_rk3576(), which is the fallback. */
    all     = calloc((size_t)c->ntask * RK3576_CONV_TASK_OPS, sizeof *all);
    c->ext  = calloc(c->ntask, sizeof *c->ext);
    /* One handle per weight cube, coefficient group and surface, plus the stream and the
     * first layer's feature cube. An intermediate surface is named ONCE — it is the
     * producer's output and the consumer's input, and a handle in both lists is rejected
     * with EALREADY, since the driver locks each BO once per job. */
    c->in_h  = calloc((size_t)2u * n + 2u, sizeof *c->in_h);
    c->out_h = calloc(n, sizeof *c->out_h);
    if (!c->td || !off || !cnt || !all || !c->ext || !c->in_h || !c->out_h) goto fail;

    /* Generate every program and record its length, so the contiguous offsets can be laid
     * out once the total is known. */
    for (i = 0; i < n; i++) {
        r76_w *hi = c->layer[i].h;
        conv_params_t p;
        unsigned t;
        if (!hi) {
            /* The pool's one program, from the same builder the per-layer pooling entry
             * uses. Its surface stride is round4(ow*oh) and it fills only ow*oh of each
             * group, so the write check asks about that prefix — the padding is never
             * written and would read as a task that did nothing. */
            unsigned k = c->layer[i].task0;
            struct rocket_rk3576_pool_link *pl = &c->layer[i].pl;
            if (rocket_rk3576_pool_link(c->layer[i].p, pl,
                                        all + (size_t)k * RK3576_CONV_TASK_OPS)
                != ROCKET_OK) {
                ROCKET_LOGE("%s: layer %u's pooling program could not be built\n", entry, i);
                goto fail;
            }
            cnt[k] = pl->nops;
            off[k] = words;
            words += (unsigned)rkt_chain_words(pl->nops);
            c->ext[k].base        = pl->surf_off;
            c->ext[k].groups      = pl->groups;
            c->ext[k].group_bytes = (size_t)pl->surf_elems * C2;
            c->ext[k].row_off     = 0;
            c->ext[k].span        = (size_t)pl->live_elems * C2;
            continue;
        }
        r76_chain_params(hi, &p);
        for (t = 0; t < c->layer[i].ntask; t++) {
            conv_params_t q = p;
            unsigned k = c->layer[i].task0 + t;
            q.ih = hi->plan[t].ih; q.oh = hi->plan[t].oh;
            q.pad_top = hi->plan[t].pad_top;
            q.input_dma  = p.input_dma  + hi->plan[t].feature_off;
            q.output_dma = p.output_dma + hi->plan[t].output_off;
            q.ih_full = (uint16_t)hi->IH; q.oh_full = (uint16_t)hi->oh;
            q.tasks = all + (size_t)k * RK3576_CONV_TASK_OPS;
            if ((hi->dw ? gen_conv2d_dw_int8_rk3576(&q)
                        : gen_conv2d_int8_rk3576(&q)) != 0) {
                ROCKET_LOGE("%s: the generator refused task %u of layer %u\n", entry, t, i);
                goto fail;
            }
            cnt[k] = q.task_count;
            off[k] = words;
            /* An even word stride, which is what the PC's segment length rounds to. The
             * lengths DIFFER across layers — a depthwise program and a direct one are not
             * the same size — which is why the offsets are explicit rather than derived from
             * a single stride. */
            words += (unsigned)rkt_chain_words(q.task_count);
            c->ext[k].base        = hi->out_ext.ptr ? hi->out_off : 0u;
            c->ext[k].groups      = (hi->tile[0].ocreg + C2 - 1u) / C2;
            c->ext[k].group_bytes = (size_t)hi->surf_elems * C2;
            c->ext[k].row_off     = (size_t)hi->plan[t].oy0 * hi->ow * C2;
            c->ext[k].span        = (size_t)hi->plan[t].oh * hi->ow * C2;
        }
    }

    if (rocket_bo_alloc(fd, (size_t)words * sizeof(uint64_t), &c->rc) < 0) goto fail;

    /* Lay the whole stream out and link it, across layer boundaries as well as inside them.
     * The last program's forward link is cleared: the hardware halts there on TASK_NUMBER,
     * but a live address would have the PC prefetch past the chain. */
    rocket_bo_prep(fd, &c->rc, 1, 0);
    for (i = 0; i < n; i++) {
        unsigned t;
        for (t = 0; t < c->layer[i].ntask; t++) {
            unsigned k = c->layer[i].task0 + t;
            int more = (k + 1u < c->ntask);
            size_t next = more ? off[k + 1u] : off[k];
            if (rkt_chain_pack_at(&c->rc, c->td, (int)k, off[k], next,
                                  all + (size_t)k * RK3576_CONV_TASK_OPS,
                                  cnt[k], more ? cnt[k + 1u] : cnt[k]) != 0) {
                ROCKET_LOGE("%s: the chain link could not be written for task %u\n",
                            entry, k);
                rocket_bo_fini(fd, &c->rc);
                goto fail;
            }
        }
    }
    rocket_bo_fini(fd, &c->rc);

    /* EACH BO ONCE, ACROSS BOTH LISTS. The driver locks every BO of a job through
     * drm_exec and locking one twice is EALREADY, which rejects the submit — so an
     * intermediate surface is named once as an output and not again as the consumer's
     * input. With SLICES that is no longer automatic: two producers writing halves of one
     * concatenation buffer name the same BO, and the first layer's feature cube can be a
     * slice of a buffer a later layer writes. Deduplicating here is what keeps the layout
     * a property of the addresses rather than of who happens to share an allocation. */
    for (i = 0; i < n; i++) {
        r76_w *hi = c->layer[i].h;
        if (!hi) {
            /* A pool has no weight cube and no coefficient group — its whole program is
             * PPU and PPU_RDMA register writes — so it names only its surface. */
            r76_h_add(c->out_h, &c->n_out, c->layer[i].pl.surf->handle);
            continue;
        }
        r76_h_add(c->in_h, &c->n_in, hi->tile[0].w.handle);
        r76_h_add(c->in_h, &c->n_in, hi->tile[0].coeff.handle);
        r76_h_add(c->out_h, &c->n_out,
                  hi->out_ext.ptr ? hi->out_ext.handle : hi->tile[0].out.handle);
    }
    r76_h_add(c->in_h, &c->n_in, c->rc.handle);
    /* The first layer's feature cube. When it reads a cube from OUTSIDE the chain that
     * buffer belongs to someone else and still has to be named; when it scatters into its
     * own, that one does. Either is skipped if a layer in this chain already writes it.
     * Always a convolution — a pool may only be interior. */
    {
        r76_w *h0 = c->layer[0].h;
        uint32_t fh = h0->cube_in ? h0->src.handle
                                  : (h0->in.ptr ? h0->in.handle : 0u);
        if (fh && !r76_h_has(c->out_h, c->n_out, fh))
            r76_h_add(c->in_h, &c->n_in, fh);
    }

    free(off); free(cnt); free(all);
    ROCKET_LOGI("%s: %u layer(s), %u task(s), %u regcmd word(s) — ONE hardware kick where "
                "the per-layer path takes %u\n", entry, n, c->ntask, words, n);
    return c;

fail:
    free(off); free(cnt); free(all);
    rocket_conv2d_int8_chain_free_rk3576(fd, c);
    return NULL;
}

/* The convolution-only form: the same constructor with every node a `conv`. One
 * implementation, for the same reason the finder has one. */
rocket_conv2d_int8_chain_rk3576 *
rocket_conv2d_int8_chain_new_rk3576(int fd,
                                    rocket_conv2d_int8_weights_rk3576 *const *h,
                                    unsigned n)
{
    rocket_chain_node_rk3576 *nd;
    rocket_conv2d_int8_chain_rk3576 *c;
    unsigned i;

    if (!h || n < 2u) {
        ROCKET_LOGE("rocket_conv2d_int8_chain_new_rk3576: a chain is two or more layers\n");
        return NULL;
    }
    nd = calloc(n, sizeof *nd);
    if (!nd) return NULL;
    for (i = 0; i < n; i++) nd[i].conv = h[i];
    c = rocket_chain_new_rk3576(fd, nd, n);
    free(nd);
    return c;
}

unsigned rocket_conv2d_int8_chain_kicks_rk3576(const rocket_conv2d_int8_chain_rk3576 *c)
{
    return c ? c->kicks : 0u;
}

/* THE FIRST LAYER THE POISON GUARD ASKS ABOUT. Zero — every surface — unless
 * ROCKET_RK3576_GUARD_PER_KICK is set, which reduces the guard to the LAST layer alone.
 *
 * The guard's cost is a PREP_BO/FINI_BO pair per surface per call and both halves sync the
 * whole BO, so on a graph that goes out as one kick it is a bracket per LAYER: 3.1 ms of
 * MobileNetV2's 6.8 and 2.0 of MobileNetV1's 5.2 [measured, the ROCKET_RK3576_I32_SENTINEL=0
 * A/B at `rk3576_net_gate bench 100`]. Asking one surface instead of n is the only way to
 * spend fewer brackets without a ranged cache-maintenance ioctl, which the uAPI does not
 * have — `drm_rocket_prep_bo` is `{handle, reserved, timeout_ns}`.
 *
 * WHAT IT STILL COVERS. The hazard the guard exists for is the wide-output poisoning: a job
 * whose DPU output element is wider than one byte leaves the next submit of any kind, across
 * processes, completing normally and writing NOTHING. That kills a whole submit and a kick is
 * one submit, so any surface in the kick witnesses it — including the last.
 *
 * WHAT IT STOPS COVERING, and this is the trade rather than a free saving: a single program
 * dying INSIDE the stream. That leaves its own surface stale and the layer after it computes
 * on the sentinel and writes a perfectly plausible surface, so the last layer's answer proves
 * nothing about it. Nothing on this part has been observed to do that with the completion
 * fence anchored on PC_TASK_STATUS (interface 1.4) — the suite runs 0 backstop hits — but
 * "not observed" is not "cannot happen", which is why the default asks every layer.
 *
 * The intermittent dropped atom is a third thing and is not what changes here: it is per
 * ATOM rather than per submit, and it was measured NOT to reach the int8 convolution entries
 * (156 row tasks, 7, 0 quiet under a host streaming memcpy to DDR, against 51.7% for the
 * int32 matmul writer) [HW sweep, H96 MAX M9]. */
static int r76_guard_per_kick(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_GUARD_PER_KICK");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 0;
    }
    return cached;
}

/* WHAT A KICK IS MADE OF. A cross-layer kick charges a whole run of layers to one call, so
 * a graph's per-layer table reports it as a single number and cannot say what is inside it —
 * and on a graph that is mostly kicks, that is most of the wall unattributed. The five terms
 * below are the whole of this function: the first layer's feature scatter, the sentinel the
 * steady state almost never pays, the submit and its fence, the write guard's bracket per
 * surface, and the last layer's de-scatter.
 *
 * Shares the ROCKET_RK3576_INT8_PROF knob with the per-call path, so one run reads a graph
 * end to end. Unlike that one it prints per CALL as well, so read a bench's later iterations
 * rather than the first — the first pays the sentinel loop for every surface. */
struct r76_chain_prof {
    int      on;
    unsigned kicks;
    double   in_us, stamp_us, submit_us, verify_us, read_us;
};

static void r76_chain_prof_log(const char *entry, const rocket_conv2d_int8_chain_rk3576 *c,
                               const struct r76_chain_prof *prof)
{
    double tot = prof->in_us + prof->stamp_us + prof->submit_us + prof->verify_us +
                 prof->read_us;

    if (!prof->on) return;
    if (tot <= 0.0) tot = 1.0;
    ROCKET_LOGI("%s: %u layer(s), %u task(s), %u kick(s) — %.2f ms: feature %.2f (%.0f%%) "
                "stamp %.2f (%.0f%%) submit %.2f (%.0f%%) verify %.2f (%.0f%%) "
                "de-scatter %.2f (%.0f%%)\n",
                entry, c->n, c->ntask, prof->kicks, tot / 1e3,
                prof->in_us / 1e3,     100.0 * prof->in_us / tot,
                prof->stamp_us / 1e3,  100.0 * prof->stamp_us / tot,
                prof->submit_us / 1e3, 100.0 * prof->submit_us / tot,
                prof->verify_us / 1e3, 100.0 * prof->verify_us / tot,
                prof->read_us / 1e3,   100.0 * prof->read_us / tot);
}

/* The surface a layer of the stream writes, whichever kind of node it is, and how many
 * bytes of it that layer owns. A pool's surface is the one its link view names. */
static rocket_bo *r76_layer_surf(struct r76_chain_layer *L, size_t *off)
{
    if (L->h) return r76_surf(L->h, 0, off);
    *off = L->pl.surf_off;
    return L->pl.surf;
}

static size_t r76_layer_obytes(const struct r76_chain_layer *L)
{
    if (L->h) return r76_chain_obytes(L->h);
    return (size_t)L->pl.groups * L->pl.surf_elems * C2;
}

/* Whether a layer's surface may be re-stamped inside the verify bracket. A convolution's
 * may unless it is a slice of a caller's buffer or a declared shared cube, both of which
 * something outside this stream reads later in the same inference. A POOL's never is: the
 * pooling handle keeps no "already stamped" flag for the chain to clear, so its stamp goes
 * on at the start of the next call — one extra bracket, and there is at most one pooling
 * layer in a run. */
static int r76_layer_restamp_ok(const struct r76_chain_layer *L)
{
    return L->h && !L->h->out_ext.ptr && !L->h->shared_out;
}

int rocket_conv2d_int8_chain_run_rk3576(int fd, rocket_conv2d_int8_chain_rk3576 *c,
                                        const int8_t *in, int8_t *out)
{
    const char *entry = "rocket_conv2d_int8_chain_run_rk3576";
    unsigned char stamp;
    unsigned attempt, attempts, i, i0;
    r76_w *first, *last;
    struct r76_chain_prof prof = {0};
    double pt;
    int rc;

    if (!c) return ROCKET_E_SHAPE;
    if (fd != c->fd) {
        ROCKET_LOGE("%s: this chain was built on fd %d and its BOs live there\n",
                    entry, c->fd);
        return ROCKET_E_SHAPE;
    }
    first = c->layer[0].h;
    last  = c->layer[c->n - 1u].h;
    if ((!first->cube_in && !in) || (!last->cube_out && !out)) {
        ROCKET_LOGE("%s: the first layer %s a row-major input and the last %s a row-major "
                    "output\n", entry,
                    first->cube_in ? "does not take" : "needs",
                    last->cube_out ? "does not write" : "needs");
        return ROCKET_E_SHAPE;
    }

    prof.on = r76_int8_prof_on();
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;
    attempts = r76_task_attempts();
    c->kicks = 0;
    /* Every layer, or only the last. See r76_guard_per_kick(): the same index bounds the
     * stamp, the check and the redo's re-stamp, so an unchecked surface never carries a
     * sentinel and nothing spends a bracket on it. */
    i0 = r76_guard_per_kick() ? c->n - 1u : 0u;

    pt = R76_PT(prof);
    rc = r76_feature_pack(first, in);
    R76_ACC(prof, in_us, pt);
    if (rc != ROCKET_OK) return rc;

    /* The sentinel, on any surface that is not already carrying it. In the steady state that
     * is none of them: the verify pass below re-stamps inside its own bracket, so a surface
     * costs one PREP_BO/FINI_BO pair per call rather than two. */
    pt = R76_PT(prof);
    if (stamp)
        for (i = i0; i < c->n; i++) {
            struct r76_chain_layer *L = &c->layer[i];
            size_t off = 0;
            rocket_bo *surf = r76_layer_surf(L, &off);
            if (L->h && L->h->tile[0].stamped) continue;
            rocket_bo_prep(fd, surf, 1, 0);
            memset((char *)surf->ptr + off, stamp, r76_layer_obytes(L));
            rocket_bo_fini(fd, surf);
        }
    R76_ACC(prof, stamp_us, pt);

    for (attempt = 0; attempt < attempts; attempt++) {
        unsigned bad = c->n, bad_task = 0;
        size_t loff = 0;
        rocket_bo *lsurf = r76_surf(last, 0, &loff);

        c->kicks++;
        prof.kicks++;
        pt = R76_PT(prof);
        if (rocket_submit_tasks_flags(fd, c->td, c->ntask, c->in_h, c->n_in,
                                      c->out_h, c->n_out, ROCKET_JOB_BATCHED) != 0) {
            ROCKET_LOGE("%s: the chained submit failed\n", entry);
            return ROCKET_E_DEVICE;
        }
        /* One fence for the whole kick, waited on through the LAST layer's surface. Every
         * BO in the job carries it, so any of them would do; the last one is the one whose
         * writes have to have drained before the de-scatter reads it. */
        if (rocket_bo_prep(fd, lsurf, 0, 2000000000ull) < 0) {
            ROCKET_LOGE("%s: PREP_BO on the last layer's surface timed out\n", entry);
            return ROCKET_E_DEVICE;
        }
        rocket_bo_fini(fd, lsurf);
        R76_ACC(prof, submit_us, pt);
        if (!stamp) break;
        pt = R76_PT(prof);

        /* EVERY LAYER, not the last one. A dead program leaves its own surface holding the
         * sentinel, and the layer after it then computes on that and writes a surface that
         * looks entirely normal — so asking only the output asks the wrong buffer.
         *
         * THE CHECK AND THE NEXT CALL'S SENTINEL SHARE ONE BRACKET. PREP_BO syncs the whole
         * BO for the CPU and FINI_BO syncs the whole BO back for the device, so a bracket
         * costs two passes over the surface however few bytes are touched inside it — the
         * cost of the guard is the BRACKET, not the fill. Reading a surface and then
         * re-stamping it in a second bracket pays that twice for one CPU visit.
         *
         * The order is safe: the redo path below re-stamps every surface anyway, so a layer
         * stamped here before a LATER one is found bad loses nothing. */
        for (i = i0; i < c->n && bad == c->n; i++) {
            struct r76_chain_layer *L = &c->layer[i];
            r76_w *hi = L->h;
            unsigned missing = 0;
            size_t off = 0;
            rocket_bo *surf = r76_layer_surf(L, &off);
            rocket_bo_prep(fd, surf, 0, 2000000000ull);
            /* The extents carry the slice's own base, so this is the BO's pointer. */
            if (!r76_all_wrote((const unsigned char *)surf->ptr,
                               c->ext + L->task0, L->ntask, stamp, &missing)) {
                bad = i; bad_task = missing;
            } else if (hi != last) {
                /* RE-STAMPING NOW IS ONLY SAFE FOR A SURFACE THIS CHAIN REWRITES BEFORE
                 * ANYONE READS IT. Every layer but the last is read by the layer after it
                 * INSIDE the kick, so the producer overwrites the sentinel in the next kick.
                 *
                 * A SURFACE THIS CHAIN SHARES WITH SOMETHING OUTSIDE IT is not that case: a
                 * slice of a caller's buffer is read beside another producer's slice later
                 * in this same inference, so stamping it now would replace this layer's
                 * output with the sentinel, invisibly. Its stamp goes on at the start of the
                 * next call. So is a surface the caller has declared SHARED: two consumers
                 * read it and only one of them is inside this stream. The LAST layer is the
                 * same case again and is handled under the de-scatter, which has to read it
                 * before it may be overwritten. So is a POOL, which keeps no flag for this
                 * to clear — see r76_layer_restamp_ok(). */
                if (!r76_layer_restamp_ok(L)) {
                    if (hi) hi->tile[0].stamped = 0;
                } else {
                    memset((char *)surf->ptr + off, stamp, r76_layer_obytes(L));
                    hi->tile[0].stamped = 1;
                }
            }
            rocket_bo_fini(fd, surf);
        }
        R76_ACC(prof, verify_us, pt);
        if (bad == c->n) break;
        ROCKET_LOGD("%s: layer %u's row task %u wrote nothing on attempt %u; cycling the "
                    "power domain and redoing the WHOLE chain, which is one kick and "
                    "cannot be restarted in the middle\n",
                    entry, bad, bad_task, attempt + 1u);
        rocket_rk3576_power_idle();
        pt = R76_PT(prof);
        /* Every CHECKED surface goes back to the sentinel: a partial chain left some of them
         * written and a redo has to be able to tell again. */
        for (i = i0; i < c->n; i++) {
            struct r76_chain_layer *L = &c->layer[i];
            size_t off = 0;
            rocket_bo *surf = r76_layer_surf(L, &off);
            rocket_bo_prep(fd, surf, 1, 0);
            memset((char *)surf->ptr + off, stamp, r76_layer_obytes(L));
            rocket_bo_fini(fd, surf);
        }
        R76_ACC(prof, stamp_us, pt);
        if (attempt + 1u == attempts) {
            ROCKET_LOGE("%s: layer %u wrote nothing over %u attempts\n",
                        entry, bad, attempts);
            return ROCKET_E_DEVICE;
        }
    }

    /* The de-scatter, and the next call's sentinel, in ONE bracket — the trick the per-layer
     * path uses, for the same reason: PREP_BO has already synced these lines for the CPU and
     * the FINI_BO under it writes them back, so nothing is left dirty to race the DPU. */
    pt = R76_PT(prof);
    {
    size_t loff = 0;
    rocket_bo *lsurf = r76_surf(last, 0, &loff);
    rocket_bo_prep(fd, lsurf, 0, 2000000000ull);
    if (!last->cube_out) {
        const int8_t *o = (const int8_t *)lsurf->ptr + loff;
        size_t px = (size_t)last->oh * last->ow;
        unsigned cc, k, tile_oc = last->tile[0].tile_oc;
        for (cc = 0; cc < tile_oc; cc += C2) {
            unsigned m = tile_oc - cc < C2 ? tile_oc - cc : C2;
            int8_t *dp[C2];
            for (k = 0; k < m; k++)
                dp[k] = out + (size_t)r76_oc_of(last->perm, last->tile[0].oc0 + cc + k) * px;
            r76_c2_unpack(dp, m, o + (size_t)(cc / C2) * last->surf_elems * C2, px);
        }
    }
    /* The LAST layer's own sentinel. Every other layer took its stamp in the verify bracket
     * above; this one could not, because the de-scatter here has to read the surface first.
     *
     * When it leaves a CUBE it is the shared-surface case: its consumer is OUTSIDE the chain
     * and reads that surface LATER IN THIS SAME INFERENCE. Stamping it here overwrites the
     * layer's output with the sentinel and the consumer computes on 0xA5 — a full, plausible,
     * entirely wrong surface, and nothing between here and the next materialised layer can
     * see it. So that one is left alone with `stamped` clear, which is what puts its sentinel
     * back at the START of the next call, where the per-layer cube-out path also puts it. */
    if (stamp) {
        if (!last->cube_out) {
            memset((char *)lsurf->ptr + loff, stamp, r76_chain_obytes(last));
            last->tile[0].stamped = 1;
        } else {
            last->tile[0].stamped = 0;
        }
    }
    rocket_bo_fini(fd, lsurf);
    }
    R76_ACC(prof, read_us, pt);
    r76_chain_prof_log(entry, c, &prof);
    return ROCKET_OK;
}

/* ============================================================================
 * SECTION — the fp16 FIRST CONV, on the packed-image datapath
 *
 * A convolution whose input carries four or fewer channels is not the one below at a
 * small channel count: the CNA reads a PACKED IMAGE straight out of DDR and expands
 * every pixel to four lanes, so one task contracts the whole image and there is no
 * input-channel split to make. That is how a vision stem runs on this part at all —
 * the normal float path contracts sixteen channels at a time and an RGB image is three.
 *
 * Two things differ from every other entry here and both are the caller's:
 *
 *   - The FEATURE BUFFER is a packed image, `ic` interleaved fp16 per pixel, not the
 *     NC1HWC2 cube. This entry still takes the library's row-major CHW tensor and
 *     interleaves it itself, so a caller sees one convolution API either way.
 *   - The WEIGHT CUBE is neither the int8 cube nor the float one: four lanes per
 *     (output channel, tap), output channels interleaved by sixteen inside a tap.
 *     rocket_rk3576_argb_fp16_pack_weights() owns it.
 *
 * The output is the ordinary fp16 surface, so nothing downstream changes.
 * ==========================================================================*/
/* THE OUTPUT-CHANNEL BOUND ON THIS PATH, measured rather than transcribed: one
 * first-conv program delivers THIRTY-TWO output channels and no more. At oc 48, 64 and
 * 96 exactly 32 whole channels come back bit-exact and the rest of the surface is never
 * written — a contiguous prefix, not the interleave the int32 writer's byte budget
 * gives — while oc 24 and 32 are complete. It is not a register the emitter is getting
 * wrong: our program matches the vendor's own oc=48 and oc=64 captures register for
 * register, and the weight cube reproduces those captures too, so the vendor's compiler
 * emits single programs the part does not fully execute.
 *
 * So the split is the caller's, exactly as the direct path splits for its weight slice:
 * each tile is an independent convolution over its own output channels, at one submit
 * each. [HW sweep, H96 MAX M9] */
#define R76_ARGB_OC_MAX 32u

static int r76_conv_fp16_argb(const char *entry, int fd, const rocket_conv2d_desc *d,
                              const _Float16 *in, const _Float16 *W, _Float16 *out,
                              unsigned ow, unsigned oh)
{
    struct r76_conv_bos b = {0};
    uint64_t *ops = NULL;
    float *acc = NULL;
    _Float16 *wtile = NULL;
    unsigned IC = (unsigned)d->ic, OC = (unsigned)d->oc;
    unsigned IH = (unsigned)d->ih, IW = (unsigned)d->iw;
    unsigned KH = (unsigned)d->kh, KW = (unsigned)d->kw;
    unsigned tile = OC < R76_ARGB_OC_MAX ? OC : R76_ARGB_OC_MAX;
    unsigned tilepad = rocket_rk3576_fp16_pad_oc(tile);
    unsigned oc0, nrow = 0, r;
    rocket_rk3576_row_task *rows = NULL;
    size_t in_bytes, w_bytes, coeff_bytes, surf, tile_elems;
    conv_params_t plan = {0};
    struct r76_task_extent e = {0};
    uint32_t in_h[4], out_h[1];
    unsigned char stamp;
    int rc;

    /* This path's four geometry bounds are jointly the ONNX symmetric-SAME convention, and
     * an explicit output extent is not one of them: an asymmetric pad is refused here rather
     * than programmed onto the first conv's own encoding. */
    if (r76_desc_asym(d)) {
        ROCKET_LOGE("%s: an output extent this descriptor did not derive is an asymmetric "
                    "pad, which the packed-image first conv does not claim\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }

    in_bytes    = (size_t)IH * IW * IC * sizeof(_Float16);
    w_bytes     = rocket_rk3576_weight_argb_fp16_bytes(tilepad, KH, KW);
    coeff_bytes = rocket_rk3576_coeff_bytes(tilepad);
    surf        = rocket_rk3576_fp16_out_bytes(OC, oh, ow);
    tile_elems  = (size_t)tilepad * IC * KH * KW;
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;

    ops   = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    acc   = calloc((size_t)OC * oh * ow, sizeof *acc);
    wtile = calloc(tile_elems, sizeof *wtile);
    rows  = calloc(oh ? oh : 1u, sizeof *rows);
    if (!ops || !acc || !wtile || !rows) { rc = ROCKET_E_NOMEM; goto done; }

    /* The row window, on the same axis and the same planner as every other path. A
     * stem-sized plane does not fit the CBUF in one task — 224x224 is 6272 granules
     * against a 6144 ceiling — so the plane is cut into windows, each reading the input
     * rows its output rows need. The offsets come back in PACKED-IMAGE row units here,
     * which is why the planner has to be told the precision: a float packed image is
     * `ic` interleaved halfwords per pixel where an int8 one is bytes. */
    plan.ic = (uint16_t)IC; plan.ih = (uint16_t)IH; plan.iw = (uint16_t)IW;
    plan.oc = (uint16_t)tilepad; plan.oh = (uint16_t)oh; plan.ow = (uint16_t)ow;
    plan.kh = (uint16_t)KH; plan.kw = (uint16_t)KW;
    plan.stride_y = (uint8_t)d->stride_y; plan.stride_x = (uint8_t)d->stride_x;
    plan.pad_top = (uint8_t)d->pad_top; plan.pad_left = (uint8_t)d->pad_left;
    plan.ih_full = (uint16_t)IH; plan.oh_full = (uint16_t)oh;
    if (rocket_rk3576_plan_rows_prec(&plan, 0, precision_float16, rows,
                                     oh ? oh : 1u, &nrow) < 0 || !nrow) {
        ROCKET_LOGE("%s: no row plan for the first conv (ic=%u %ux%u k%ux%u)\n",
                    entry, IC, IW, IH, KW, KH);
        rc = ROCKET_E_UNSUPPORTED; goto done;
    }

    if (rocket_bo_alloc(fd, in_bytes, &b.in) < 0 ||
        rocket_bo_alloc(fd, w_bytes, &b.w) < 0 ||
        rocket_bo_alloc(fd, coeff_bytes, &b.coeff) < 0 ||
        rocket_bo_alloc(fd, surf, &b.out) < 0 ||
        rocket_bo_alloc(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }

    /* CHW in, interleaved out — the packed image the CNA reads. Shared by every tile. */
    rocket_bo_prep(fd, &b.in, 1, 0);
    {
        _Float16 *img = (_Float16 *)b.in.ptr;
        unsigned c, y, x;
        for (y = 0; y < IH; y++)
            for (x = 0; x < IW; x++)
                for (c = 0; c < IC; c++)
                    img[((size_t)y * IW + x) * IC + c] =
                        in[((size_t)c * IH + y) * IW + x];
    }
    rocket_bo_fini(fd, &b.in);

    /* C gates the BS stage and a float program reads it as fp16, where the integer 1
     * is a denormal that empties the surface. One tile's worth, reused by every tile. */
    rocket_bo_prep(fd, &b.coeff, 1, 0);
    rc = rocket_rk3576_pack_coeff_prec(b.coeff.ptr, coeff_bytes, NULL, tilepad,
                                       precision_float16);
    rocket_bo_fini(fd, &b.coeff);
    if (rc < 0) { rc = ROCKET_E_SHAPE; goto done; }

    in_h[0] = b.in.handle; in_h[1] = b.w.handle;
    in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
    out_h[0] = b.out.handle;

    if (stamp) {
        rocket_bo_prep(fd, &b.out, 1, 0);
        memset(b.out.ptr, stamp, surf);
        rocket_bo_fini(fd, &b.out);
    }

    for (oc0 = 0; oc0 < OC; oc0 += tile) {
        unsigned n = OC - oc0 < tile ? OC - oc0 : tile;
        unsigned npad = rocket_rk3576_fp16_pad_oc(n);

        /* This tile's channels, renumbered from zero — its own whole convolution. The
         * cube is the same for every row window, so it is packed once per tile. */
        memset(wtile, 0, tile_elems * sizeof *wtile);
        memcpy(wtile, W + (size_t)oc0 * IC * KH * KW,
               (size_t)n * IC * KH * KW * sizeof *wtile);
        rocket_bo_prep(fd, &b.w, 1, 0);
        rc = rocket_rk3576_argb_fp16_pack_weights(b.w.ptr, w_bytes, wtile, npad, IC,
                                                  KH, KW);
        rocket_bo_fini(fd, &b.w);
        if (rc < 0) { rc = ROCKET_E_SHAPE; goto done; }

        for (r = 0; r < nrow; r++) {
            conv_params_t p = {0};

            p.ic = (uint16_t)IC; p.iw = (uint16_t)IW;
            p.ih = rows[r].ih;   p.oh = rows[r].oh;
            p.oc = (uint16_t)npad; p.ow = (uint16_t)ow;
            p.kh = (uint16_t)KH; p.kw = (uint16_t)KW;
            p.stride_y = (uint8_t)d->stride_y; p.stride_x = (uint8_t)d->stride_x;
            p.pad_top = rows[r].pad_top; p.pad_left = (uint8_t)d->pad_left;
            p.ih_full = (uint16_t)IH; p.oh_full = (uint16_t)oh;
            p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;
            p.input_zero_point = 0; p.output_zero_point = 0; p.weight_zero_point = 0;
            p.tasks       = ops;
            p.input_dma   = b.in.dma_address + rows[r].feature_off;
            p.weights_dma = b.w.dma_address;
            p.bias_dma    = b.coeff.dma_address;
            /* The output cube's channel groups are contiguous planes, so a tile
             * starting on a group boundary is the same BO at a plane offset, and a row
             * window is that plus the planner's own row offset. */
            p.output_dma  = b.out.dma_address +
                            (uint32_t)((size_t)oc0 * oh * ow * sizeof(_Float16)) +
                            rows[r].output_off;
            if (gen_conv2d_fp16_rk3576(&p) != 0) {
                ROCKET_LOGE("%s: the generator refused the first-conv program (ic=%u "
                            "%ux%u k%ux%u oc %u..%u rows %u..%u)\n", entry, IC, IW, IH,
                            KW, KH, oc0, oc0 + n, rows[r].oy0,
                            rows[r].oy0 + rows[r].oh);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }

            /* Per TASK, not per tile: one poisoned submit among several leaves its own
             * rows stale while its siblings are full. */
            e.groups      = (npad + C2F - 1u) / C2F;
            e.group_bytes = (size_t)ow * oh * C2F * sizeof(_Float16);
            e.row_off     = (size_t)oc0 * oh * ow * sizeof(_Float16)
                            + (size_t)rows[r].output_off;
            e.span        = (size_t)rows[r].oh * ow * C2F * sizeof(_Float16);
            rc = r76_submit_task(fd, &b, &p, ops, in_h, 4u, out_h, &e, stamp,
                             ROCKET_JOB_NO_DPU_DONE, entry);
            if (rc != ROCKET_OK) goto done;
        }
    }

    rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
    rc = rocket_rk3576_fp16_accumulate(acc, b.out.ptr, surf, OC, oh, ow);
    rocket_bo_fini(fd, &b.out);
    if (rc < 0) { rc = ROCKET_E_SHAPE; goto done; }

    {
        size_t i, n = (size_t)OC * oh * ow;
        for (i = 0; i < n; i++) out[i] = (_Float16)acc[i];
    }
    rc = ROCKET_OK;

done:
    free(ops); free(acc); free(wtile); free(rows);
    r76_conv_free(fd, &b);
    return rc;
}

/* ============================================================================
 * SECTION — the fp16 convolution, through the input-channel split
 *
 * One fp16 task contracts exactly SIXTEEN input channels — the DPU's output element
 * stride is 16/ic words, so that is the only count at which an element occupies its own
 * two bytes — and gen_conv2d_fp16_rk3576() refuses any other. rocket_rk3576_plan_ic()
 * turns an arbitrary count into that sequence; the feature cube is shared and addressed
 * by each slice's offset, the weight cube is per-slice, and the partial surfaces are
 * summed on the host.
 *
 * WHAT THIS DOES NOT DO: compose the split with the row window. A plane whose 16-channel
 * slice still overflows the CBUF is refused by the planner, and running the ic slices of
 * each row task would carry it further. On-chip accumulation across the slices — the
 * RK3588's ROCKET_KACC analog — would delete ic/16 readbacks and is no longer blocked by
 * anything in the writer. Both are open levers rather than defects.
 * ==========================================================================*/
int rocket_conv2d_fp16_rk3576(int fd, const rocket_conv2d_desc *d,
                              const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    const char *entry = "rocket_conv2d_fp16_rk3576";
    struct r76_conv_bos b = {0};
    uint64_t *ops = NULL;
    float *acc = NULL;
    rocket_rk3576_ic_task *slices = NULL;
    unsigned IC, OC, IH, IW, KH, KW;
    unsigned ow, oh, icpad, ocpad, nslice = 0, s, max_slices;
    size_t in_bytes, w_bytes, coeff_bytes, surf;
    conv_params_t base = {0};
    struct r76_task_extent e = {0};
    struct r76_fp16_prof prof = {0, 0, 0, 0, 0, 0};
    uint32_t in_h[4], out_h[1];
    unsigned char stamp;
    int rc;

    rc = r76_conv_check(entry, fd, d, d && d->depthwise, 1, &ow, &oh);
    if (rc != ROCKET_OK) return rc;
    if (!in || !W || !out) return ROCKET_E_SHAPE;
    if (r76_desc_asym(d)) {
        ROCKET_LOGE("%s: an ASYMMETRIC pad — an output extent this descriptor did not "
                    "derive — is claimed on the int8 path, where the correctness envelope "
                    "carries a zero leading pad against a consumed trailing one. Nothing "
                    "on the float path has been run through that geometry\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    if (d->depthwise) {
        ROCKET_LOGE("%s: the fp16 depthwise cube is not decoded on this part; the int8 "
                    "depthwise path is (rocket_conv2d_dw_int8_rk3576)\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    /* THE FLOAT CUBE DOES NOT SHARE THE 32-CHANNEL RULE that makes a narrow ic free on the
     * int8 direct path: its groups are eight output channels and sixteen input ones, and
     * one task contracts exactly sixteen input channels, so ic 3 is a padded slice of a
     * different object and no shape below five has been run through it. Refused rather
     * than dropped — a caller that set the flag asked for a datapath, and quietly getting
     * the other one is how a wrong number reads as a working encoding. */
    if (d->direct_datapath) {
        ROCKET_LOGE("%s: direct_datapath is the int8 path's lever — the float weight cube "
                    "groups sixteen input channels, not thirty-two, and no ic below five "
                    "has been gated on it. rocket_conv2d_int8_rk3576() is where a narrow "
                    "channel count runs direct\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }

    IC = (unsigned)d->ic; OC = (unsigned)d->oc;
    IH = (unsigned)d->ih; IW = (unsigned)d->iw;
    KH = (unsigned)d->kh; KW = (unsigned)d->kw;

    /* Four or fewer channels is the packed-image first conv, a different CNA program
     * with a different feature buffer and a different weight cube. */
    if (IC <= 4u)
        return r76_conv_fp16_argb(entry, fd, d, in, W, out, ow, oh);

    icpad = rocket_rk3576_fp16_pad_ic(IC);
    ocpad = rocket_rk3576_fp16_pad_oc(OC);
    in_bytes    = (size_t)(icpad / C2F) * IH * IW * C2F * sizeof(_Float16);
    w_bytes     = rocket_rk3576_fp16_slice_weight_bytes(OC, IC, KH, KW);
    coeff_bytes = rocket_rk3576_coeff_bytes(ocpad);
    surf        = rocket_rk3576_fp16_out_bytes(OC, oh, ow);
    max_slices  = icpad / ROCKET_RK3576_FP16_IC_SLICE + 2u;
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;

    base.ic = (uint16_t)IC; base.ih = (uint16_t)IH; base.iw = (uint16_t)IW;
    base.oc = (uint16_t)OC; base.oh = (uint16_t)oh; base.ow = (uint16_t)ow;
    base.kh = (uint16_t)KH; base.kw = (uint16_t)KW;
    base.stride_y = (uint8_t)d->stride_y; base.stride_x = (uint8_t)d->stride_x;
    base.pad_top  = (uint8_t)d->pad_top;  base.pad_left = (uint8_t)d->pad_left;
    base.ih_full = (uint16_t)IH; base.oh_full = (uint16_t)oh;
    base.in_scale = 1.0f; base.w_scale = 1.0f; base.out_scale = 1.0f;
    base.input_zero_point = 0x80; base.output_zero_point = 0x80;
    base.weight_zero_point = 0x80;

    ops    = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    slices = calloc(max_slices, sizeof *slices);
    acc    = calloc((size_t)OC * oh * ow, sizeof *acc);
    if (!ops || !slices || !acc) { rc = ROCKET_E_NOMEM; goto done; }

    if (rocket_rk3576_plan_ic(&base, slices, max_slices, &nslice) < 0) {
        ROCKET_LOGE("%s: no input-channel plan for ic=%u (%ux%u k%ux%u) — a 16-channel "
                    "slice of this plane still overflows the CBUF, and composing the "
                    "split with the row window is not wired\n", entry, IC, IW, IH, KW, KH);
        rc = ROCKET_E_UNSUPPORTED; goto done;
    }

    if (rocket_bo_alloc(fd, in_bytes, &b.in) < 0 ||
        rocket_bo_alloc(fd, w_bytes, &b.w) < 0 ||
        rocket_bo_alloc(fd, coeff_bytes, &b.coeff) < 0 ||
        rocket_bo_alloc(fd, surf, &b.out) < 0 ||
        rocket_bo_alloc(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }

    /* The feature cube is shared: at an 8-channel atom its channel groups are contiguous
     * planes, so slice s is the same BO at input_dma + feature_off. */
    rocket_bo_prep(fd, &b.in, 1, 0);
    memset(b.in.ptr, 0, in_bytes);
    {
        _Float16 *cube = (_Float16 *)b.in.ptr;
        size_t px = (size_t)IH * IW, p;
        unsigned c;
        for (c = 0; c < IC; c++) {
            _Float16 *dst = cube + (size_t)(c / C2F) * px * C2F + (c % C2F);
            const _Float16 *src = in + (size_t)c * px;
            for (p = 0; p < px; p++) dst[p * C2F] = src[p];
        }
    }
    rocket_bo_fini(fd, &b.in);

    /* The coefficient buffer is shared and carries no bias, but its C multiplier still
     * gates the BS stage — and a FLOAT program reads C as fp16, where the integer 1 is
     * the denormal 6e-8 and underflows the whole surface to empty. */
    rocket_bo_prep(fd, &b.coeff, 1, 0);
    if (rocket_rk3576_pack_coeff_prec(b.coeff.ptr, coeff_bytes, NULL, ocpad,
                                      precision_float16) < 0) {
        rocket_bo_fini(fd, &b.coeff);
        rc = ROCKET_E_SHAPE; goto done;
    }
    rocket_bo_fini(fd, &b.coeff);

    in_h[0] = b.in.handle; in_h[1] = b.w.handle;
    in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
    out_h[0] = b.out.handle;

    prof.on = r76_fp16_prof_on();
    prof.slices = nslice;

    for (s = 0; s < nslice; s++) {
        conv_params_t p = base;
        double t0;

        t0 = prof.on ? r76_now_us() : 0;
        rocket_bo_prep(fd, &b.w, 1, 0);
        if (rocket_rk3576_fp16_pack_slice_weights(b.w.ptr, w_bytes, W, OC, IC, KH, KW,
                                                  &slices[s]) < 0) {
            rocket_bo_fini(fd, &b.w);
            rc = ROCKET_E_SHAPE; goto done;
        }
        rocket_bo_fini(fd, &b.w);
        if (prof.on) prof.pack_us += r76_now_us() - t0;

        p.ic          = slices[s].ic;
        p.tasks       = ops;
        p.input_dma   = b.in.dma_address + slices[s].feature_off;
        p.weights_dma = b.w.dma_address;
        p.bias_dma    = b.coeff.dma_address;
        p.output_dma  = b.out.dma_address;
        if (gen_conv2d_fp16_rk3576(&p) != 0) {
            ROCKET_LOGE("%s: the generator refused slice %u of %u\n", entry, s, nslice);
            rc = ROCKET_E_UNSUPPORTED; goto done;
        }

        /* Every slice rewrites the whole surface, so it is stamped per slice and read
         * back between submits. */
        t0 = prof.on ? r76_now_us() : 0;
        if (stamp) {
            rocket_bo_prep(fd, &b.out, 1, 0);
            memset(b.out.ptr, stamp, surf);
            rocket_bo_fini(fd, &b.out);
        }
        if (prof.on) prof.stamp_us += r76_now_us() - t0;

        e.groups      = (ocpad + C2F - 1u) / C2F;
        e.group_bytes = (size_t)ow * oh * C2F * sizeof(_Float16);
        e.row_off     = 0;
        e.span        = e.group_bytes;
        t0 = prof.on ? r76_now_us() : 0;
        rc = r76_submit_task(fd, &b, &p, ops, in_h, 4u, out_h, &e, stamp,
                             ROCKET_JOB_NO_DPU_DONE, entry);
        if (prof.on) prof.submit_us += r76_now_us() - t0;
        if (rc != ROCKET_OK) goto done;

        t0 = prof.on ? r76_now_us() : 0;
        rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
        if (rocket_rk3576_fp16_accumulate(acc, b.out.ptr, surf, OC, oh, ow) < 0) {
            rocket_bo_fini(fd, &b.out);
            rc = ROCKET_E_SHAPE; goto done;
        }
        rocket_bo_fini(fd, &b.out);
        if (prof.on) prof.read_us += r76_now_us() - t0;
    }

    if (prof.on) {
        double tot = prof.pack_us + prof.stamp_us + prof.submit_us + prof.read_us;
        ROCKET_LOGI("rk3576 fp16 ic-split ic=%u oc=%u %ux%u k%ux%u: %u slices, %.2f ms"
                    " -- weights %.2f (%.0f%%)  stamp %.2f (%.0f%%)  submit %.2f (%.0f%%)"
                    "  readback %.2f (%.0f%%)\n",
                    IC, OC, IW, IH, KW, KH, nslice, tot / 1e3,
                    prof.pack_us / 1e3, 100.0 * prof.pack_us / (tot > 0 ? tot : 1),
                    prof.stamp_us / 1e3, 100.0 * prof.stamp_us / (tot > 0 ? tot : 1),
                    prof.submit_us / 1e3, 100.0 * prof.submit_us / (tot > 0 ? tot : 1),
                    prof.read_us / 1e3, 100.0 * prof.read_us / (tot > 0 ? tot : 1));
    }

    {
        size_t i, n = (size_t)OC * oh * ow;
        for (i = 0; i < n; i++) out[i] = (_Float16)acc[i];
    }
    rc = ROCKET_OK;

done:
    free(ops); free(slices); free(acc);
    r76_conv_free(fd, &b);
    return rc;
}
