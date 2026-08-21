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
 *   A weight zero point on the depthwise path. The depthwise coefficient group has no
 *   B field at all, and the correction is not a per-channel constant, so it cannot be
 *   folded into the bias either.
 *
 *   A shape whose weight slice does not fit even one output-channel group. The
 *   recourse there is an input-channel split, which the on-chip requant forecloses:
 *   int8 partials cannot be summed without quantizing each one. The int32-output
 *   writer is where that shape belongs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"

#define C2     16u      /* int8 feature/output channel atom */
#define C2F     8u      /* fp16 feature/output channel atom */

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);
int weight_conv_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

/* ============================================================================
 * SECTION — shared plumbing
 * ==========================================================================*/
struct r76_conv_bos {
    rocket_bo in, w, coeff, out, rc;
};

static void r76_conv_free(int fd, struct r76_conv_bos *b)
{
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
     * be driving the normal path's buffers at a channel count it cannot contract. */
    if (d->ic <= 4 && !argb_ok) {
        ROCKET_LOGE("%s: %d input channels takes the CNA's ARGB first-conv "
                    "sub-encoding, which this entry does not own. Both precisions of it "
                    "run: rocket_conv2d_int8_rk3576() and rocket_conv2d_fp16_rk3576()\n",
                    entry, d->ic);
        return ROCKET_E_UNSUPPORTED;
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
    *ow_out = (unsigned)ow;
    *oh_out = (unsigned)oh;
    return ROCKET_OK;
}

/* Where one row task's own output lives, in bytes, so "did it write" can be asked of
 * exactly that task. The output cube is NC1HWC2, so a row run is one span per channel
 * group at that group's own surface offset — scanning only the first span would call a
 * task written on channel group 0 alone. */
struct r76_task_extent {
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
        const unsigned char *p = o + (size_t)g * e->group_bytes + e->row_off;
        size_t i;
        for (i = 0; i < e->span; i++)
            if (p[i] != stamp) return 1;
    }
    return 0;
}

/* Submit one row task and satisfy ourselves that it wrote. The retry is what covers the
 * poisoning an int32-output job leaves behind — it crosses calls and processes, so a
 * conv inherits it from whatever ran before — and the idle in front of the redo is the
 * NPU power domain cycling rather than a settling time. */
static int r76_submit_task(int fd, struct r76_conv_bos *b, const conv_params_t *q,
                           const uint64_t *ops, const uint32_t *in_h, unsigned n_in,
                           const uint32_t *out_h, const struct r76_task_extent *e,
                           unsigned char stamp, const char *entry)
{
    unsigned attempt;

    for (attempt = 0; attempt < 4u; attempt++) {
        rocket_bo_prep(fd, &b->rc, 1, 0);
        memcpy(b->rc.ptr, ops, q->task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &b->rc);

        if (rocket_submit_matmul(fd, &b->rc, q->task_count, in_h, n_in, out_h, 1,
                                 4000) != 0) {
            ROCKET_LOGE("%s: submit failed\n", entry);
            return ROCKET_E_DEVICE;
        }
        if (rocket_bo_prep(fd, &b->out, 0, 2000000000ull) < 0) {
            ROCKET_LOGE("%s: PREP_BO on the output timed out\n", entry);
            return ROCKET_E_DEVICE;
        }
        if (!stamp) { rocket_bo_fini(fd, &b->out); return ROCKET_OK; }
        {
            int wrote = r76_task_wrote((const unsigned char *)b->out.ptr, e, stamp);
            rocket_bo_fini(fd, &b->out);
            if (wrote) return ROCKET_OK;
        }
        ROCKET_LOGD("%s: a row task wrote nothing, cycling the power domain and "
                    "redoing it\n", entry);
        rocket_rk3576_power_idle();
    }
    ROCKET_LOGE("%s: a row task wrote nothing over four attempts\n", entry);
    return ROCKET_E_DEVICE;
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
static void r76_fold_coeff(int32_t *A, const int32_t *bias, unsigned oc0,
                           unsigned tile_oc, const int64_t *sum_w, int in_zp, int w_zp,
                           unsigned taps)
{
    unsigned j;
    for (j = 0; j < tile_oc; j++) {
        int64_t a = bias ? (int64_t)bias[oc0 + j] : 0;
        a -= (int64_t)in_zp * sum_w[oc0 + j];
        a += (int64_t)in_zp * w_zp * taps;
        A[j] = (int32_t)a;
    }
}

static int r76_conv_int8_run(const char *entry, int fd, const rocket_conv2d_desc *d,
                             int dw, const int8_t *in, const int8_t *W,
                             const int32_t *bias, float in_scale, float w_scale,
                             float out_scale, int in_zp, int w_zp, int out_zp,
                             int8_t *out)
{
    struct r76_conv_bos b = {0};
    uint64_t *ops = NULL;
    rocket_rk3576_row_task *plan = NULL;
    int32_t *A = NULL;
    int64_t *sum_w = NULL;
    int16_t *B = NULL;
    unsigned IC, OC, IH, IW, KH, KW, SY, SX, PT, PL;
    unsigned ow, oh, icreg, icpad, surf_elems, oc_tile, oc0, max_tasks, taps;
    size_t in_bytes;
    unsigned char stamp;
    int rc;

    rc = r76_conv_check(entry, fd, d, dw, 0, &ow, &oh);
    if (rc != ROCKET_OK) return rc;
    if (!in || !W || !out) return ROCKET_E_SHAPE;
    if (!(in_scale > 0.0f) || !(w_scale > 0.0f) || !(out_scale > 0.0f)) {
        ROCKET_LOGE("%s: the quant scales must be positive — the DPU's OUT_CVT gates "
                    "the whole BS stage off at zero and writes an empty surface\n", entry);
        return ROCKET_E_SHAPE;
    }
    if (in_zp < -128 || in_zp > 127 || w_zp < -128 || w_zp > 127 ||
        out_zp < -128 || out_zp > 127)
        return ROCKET_E_SHAPE;
    if (dw && w_zp) {
        ROCKET_LOGE("%s: the depthwise coefficient group has no B field, and a weight "
                    "zero point's correction is not a per-channel constant, so it "
                    "cannot be folded into the bias either\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }

    IC = (unsigned)d->ic; OC = (unsigned)d->oc;
    IH = (unsigned)d->ih; IW = (unsigned)d->iw;
    KH = (unsigned)d->kh; KW = (unsigned)d->kw;
    SY = (unsigned)d->stride_y; SX = (unsigned)d->stride_x;
    PT = (unsigned)d->pad_top;  PL = (unsigned)d->pad_left;

    /* Channel counts as told to the REGISTERS. The direct path needs both rounded to the
     * 32-channel MAC group. The depthwise path takes the RAW count — its own two
     * granules (the weight cube rounds to 16, the CBUF allocation sometimes one 16-group
     * further) are the emitter's business, and rounding here would hide every count
     * where the two differ. */
    icreg = dw ? IC : rocket_rk3576_pad_ic(IC);
    icpad = (icreg + 31u) / 32u * 32u;
    surf_elems = rocket_rk3576_out_surf_elems(ow, oh, dw);
    taps = dw ? KH * KW : IC * KH * KW;
    in_bytes = (size_t)((icpad + C2 - 1u) / C2) * IH * IW * C2;
    max_tasks = oh + 2u;
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;

    oc_tile = dw ? OC : r76_conv_oc_tile(icreg, KH, KW, rocket_rk3576_pad_oc(OC));
    if (!oc_tile) return ROCKET_E_SHAPE;

    ops   = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    plan  = calloc(max_tasks, sizeof *plan);
    sum_w = calloc(OC, sizeof *sum_w);
    if (!ops || !plan || !sum_w) { rc = ROCKET_E_NOMEM; goto done; }

    /* Each output channel's whole filter, for the input zero point's fold. */
    {
        unsigned c, i, y, x;
        for (c = 0; c < OC; c++) {
            int64_t s = 0;
            if (dw) {
                for (y = 0; y < KH; y++)
                    for (x = 0; x < KW; x++)
                        s += W[((size_t)c * KH + y) * KW + x];
            } else {
                for (i = 0; i < IC; i++)
                    for (y = 0; y < KH; y++)
                        for (x = 0; x < KW; x++)
                            s += W[(((size_t)c * IC + i) * KH + y) * KW + x];
            }
            sum_w[c] = s;
        }
    }

    /* The FEATURE cube is packed once and shared by every output-channel tile: the
     * tiling is on the output axis, which the feature side does not see. */
    if (rocket_bo_alloc(fd, in_bytes, &b.in) < 0) { rc = ROCKET_E_NOMEM; goto done; }
    rocket_bo_prep(fd, &b.in, 1, 0);
    memset(b.in.ptr, 0, in_bytes);
    /* The channel-group arithmetic is hoisted out of the pixel loop. A CHW tensor and an
     * NC1HWC2 cube are a transpose, not a copy, so this stays a strided store — but
     * calling the index function per element pays a divide and a modulo on every one of
     * IC*IH*IW, and that was most of what a conv on this part spent outside the NPU. */
    {
        int8_t *cube = (int8_t *)b.in.ptr;
        size_t px = (size_t)IH * IW, p;
        unsigned c;
        for (c = 0; c < IC; c++) {
            int8_t *dst = cube + (size_t)(c / C2) * px * C2 + (c % C2);
            const int8_t *src = in + (size_t)c * px;
            for (p = 0; p < px; p++) dst[p * C2] = src[p];
        }
    }
    rocket_bo_fini(fd, &b.in);

    if (rocket_bo_alloc(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }

    for (oc0 = 0; oc0 < OC; oc0 += oc_tile) {
        unsigned tile_oc = OC - oc0 < oc_tile ? OC - oc0 : oc_tile;
        unsigned ocreg = dw ? tile_oc : rocket_rk3576_pad_oc(tile_oc);
        size_t w_bytes = dw ? rocket_rk3576_weight_dw_bytes(ocreg, KH, KW)
                            : (size_t)((ocreg + 31u) / 32u) * ((icreg + 31u) / 32u) *
                              32u * 32u * KH * KW;
        size_t coeff_bytes = dw ? rocket_rk3576_coeff_bytes_dw(ocreg)
                                : rocket_rk3576_coeff_bytes(ocreg);
        size_t obytes = (size_t)((ocreg + C2 - 1u) / C2) * surf_elems * C2;
        conv_params_t p = {0};
        struct r76_task_extent e;
        uint32_t in_h[4], out_h[1];
        unsigned ntask = 1u, t;

        if (b.w.ptr)     rocket_bo_free(fd, &b.w);
        if (b.coeff.ptr) rocket_bo_free(fd, &b.coeff);
        if (b.out.ptr)   rocket_bo_free(fd, &b.out);
        memset(&b.w, 0, sizeof b.w);
        memset(&b.coeff, 0, sizeof b.coeff);
        memset(&b.out, 0, sizeof b.out);

        if (rocket_bo_alloc(fd, w_bytes, &b.w) < 0 ||
            rocket_bo_alloc(fd, coeff_bytes, &b.coeff) < 0 ||
            rocket_bo_alloc(fd, obytes, &b.out) < 0) { rc = ROCKET_E_NOMEM; goto done; }

        /* The WEIGHT cube is per-tile and each tile is its own convolution, so its
         * group count follows the tile rather than the whole output-channel count. */
        rocket_bo_prep(fd, &b.w, 1, 0);
        memset(b.w.ptr, 0, w_bytes);
        {
            int8_t *cube = (int8_t *)b.w.ptr;
            unsigned c, i, y, x;
            if (dw) {
                /* This part's own depthwise cube: channels grouped by 32, tap-major
                 * inside a group, two byte slots per weight — and at int8 the byte a
                 * channel owns inside a tap block is 4*(c/2) + (c%2). Neither the direct
                 * path's cube nor the RK3588's 64-channel single-byte one. */
                for (c = 0; c < tile_oc; c++)
                    for (y = 0; y < KH; y++)
                        for (x = 0; x < KW; x++)
                            cube[rocket_rk3576_weight_dw_int8(ocreg, KH, KW, c, y, x)] =
                                W[(((size_t)(oc0 + c) * KH) + y) * KW + x];
            } else {
                /* The same hoist. weight_conv_int8() at these groups is
                 * (c/32)*nIC1*KH*KW*1024 + (i/32)*KH*KW*1024 + (y*KW + x)*1024 +
                 * (c%32)*32 + (i%32), so the taps of one (channel, input channel) pair
                 * walk a fixed 1024-byte stride from a contiguous source run. */
                unsigned nIC1 = (icreg + 31u) / 32u, t, taps = KH * KW;
                for (c = 0; c < tile_oc; c++) {
                    size_t cbase = (size_t)(c / 32u) * nIC1 * taps * 1024u
                                 + (size_t)(c % 32u) * 32u;
                    for (i = 0; i < IC; i++) {
                        int8_t *dst = cube + cbase
                                    + (size_t)(i / 32u) * taps * 1024u + (i % 32u);
                        const int8_t *src = W + ((size_t)(oc0 + c) * IC + i) * taps;
                        for (t = 0; t < taps; t++) dst[(size_t)t * 1024u] = src[t];
                    }
                }
            }
        }
        rocket_bo_fini(fd, &b.w);

        /* The COEFFICIENT buffer is NOT a flat int32 bias array on this part, and a
         * zeroed one makes the DPU write a full but entirely empty surface whatever the
         * MAC did — the C term gates the BS stage. The tail channels of a partial group
         * get a zero A term so they carry a C term too. */
        free(A); free(B);
        A = calloc(ocreg, sizeof *A);
        B = w_zp ? calloc(ocreg, sizeof *B) : NULL;
        if (!A || (w_zp && !B)) { rc = ROCKET_E_NOMEM; goto done; }
        r76_fold_coeff(A, bias, oc0, tile_oc, sum_w, in_zp, w_zp, taps);
        if (B) {
            unsigned j;
            /* An asymmetric weight is w_true = w_stored - w_zp, whose correction is
             * -w_zp*sum(x), and the DPU ADDS the B term. */
            for (j = 0; j < ocreg; j++) B[j] = (int16_t)(-w_zp);
        }
        rocket_bo_prep(fd, &b.coeff, 1, 0);
        if (dw)      rocket_rk3576_pack_coeff_dw(b.coeff.ptr, coeff_bytes, A, ocreg);
        else if (B)  rocket_rk3576_pack_coeff_asym(b.coeff.ptr, coeff_bytes, A, ocreg,
                                                   B, 1);
        else         rocket_rk3576_pack_coeff(b.coeff.ptr, coeff_bytes, A, ocreg);
        rocket_bo_fini(fd, &b.coeff);

        p.ic = (uint16_t)icreg; p.ih = (uint16_t)IH; p.iw = (uint16_t)IW;
        p.oc = (uint16_t)ocreg; p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
        p.kh = (uint16_t)KH;    p.kw = (uint16_t)KW;
        p.stride_y = (uint8_t)SY; p.stride_x = (uint8_t)SX;
        p.pad_top  = (uint8_t)PT; p.pad_left = (uint8_t)PL;
        p.ih_full = (uint16_t)IH; p.oh_full = (uint16_t)oh;
        p.int8_out = 1;
        p.in_scale = in_scale; p.w_scale = w_scale; p.out_scale = out_scale;
        /* Both zero points reach the registers uint8-centered: the emitter programs the
         * border constant as (input_zero_point & 0xff) - 0x80 and the output offset as
         * output_zero_point - 0x80, so a model-domain signed zero point is that value
         * plus 0x80. The weight zero point rides in the coefficient group's B term
         * instead and this field is inert on the RK3576 path. */
        p.input_zero_point  = in_zp  + 0x80;
        p.output_zero_point = out_zp + 0x80;
        p.weight_zero_point = 0x80;
        p.tasks       = ops;
        p.input_dma   = b.in.dma_address;
        p.weights_dma = b.w.dma_address;
        p.bias_dma    = b.coeff.dma_address;
        p.output_dma  = b.out.dma_address;

        {
            conv_params_t q = p;
            if (rocket_rk3576_plan_rows(&q, dw, plan, max_tasks, &ntask) < 0) {
                ROCKET_LOGE("%s: no row plan for ic=%u oc tile %u (%ux%u k%ux%u s%u) — "
                            "the recourse is an input-channel split, which this path's "
                            "on-chip requant forecloses\n",
                            entry, IC, tile_oc, IW, IH, KW, KH, SX);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }
        }

        in_h[0] = b.in.handle; in_h[1] = b.w.handle;
        in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
        out_h[0] = b.out.handle;

        if (stamp) {
            rocket_bo_prep(fd, &b.out, 1, 0);
            memset(b.out.ptr, stamp, obytes);
            rocket_bo_fini(fd, &b.out);
        }

        for (t = 0; t < ntask; t++) {
            conv_params_t q = p;
            q.ih = plan[t].ih; q.oh = plan[t].oh;
            q.pad_top = plan[t].pad_top;
            q.input_dma  = p.input_dma  + plan[t].feature_off;
            q.output_dma = p.output_dma + plan[t].output_off;
            /* ALWAYS, not only when the plan split. A single-task plan's window is still
             * shorter than the plane whenever the last output row does not reach the
             * bottom input row — ordinary stride-2 VALID geometry — and leaving these at
             * the window makes the emitter derive the DDR channel-group stride from the
             * WINDOW, so every group past the first reads at the wrong offset and the
             * surface comes back unrelated to the input, with nothing to fault on. */
            q.ih_full = (uint16_t)IH; q.oh_full = (uint16_t)oh;
            if ((dw ? gen_conv2d_dw_int8_rk3576(&q) : gen_conv2d_int8_rk3576(&q)) != 0) {
                ROCKET_LOGE("%s: the generator refused task %u of %u\n", entry, t, ntask);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }
            e.groups      = (ocreg + C2 - 1u) / C2;
            e.group_bytes = (size_t)surf_elems * C2;
            e.row_off     = (size_t)plan[t].oy0 * ow * C2;
            e.span        = (size_t)plan[t].oh * ow * C2;
            rc = r76_submit_task(fd, &b, &q, ops, in_h, 4u, out_h, &e, stamp, entry);
            if (rc != ROCKET_OK) goto done;
        }

        /* De-scatter this tile's channels straight into the caller's row-major out. */
        rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
        /* The same hoist on the way back: y*ow + x is the pixel index, contiguous in the
         * CHW output and at a fixed C2 stride in the cube. */
        {
            const int8_t *o = (const int8_t *)b.out.ptr;
            size_t px = (size_t)oh * ow, p;
            unsigned c;
            for (c = 0; c < tile_oc; c++) {
                const int8_t *src = o + (size_t)(c / C2) * surf_elems * C2 + (c % C2);
                int8_t *dst = out + (size_t)(oc0 + c) * px;
                for (p = 0; p < px; p++) dst[p] = src[p * C2];
            }
        }
        rocket_bo_fini(fd, &b.out);
    }
    rc = ROCKET_OK;

done:
    free(ops); free(plan); free(A); free(B); free(sum_w);
    r76_conv_free(fd, &b);
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
    int rc;

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
    if (IC < 2u) {
        ROCKET_LOGE("%s: the int8 first conv writes nothing at one image channel "
                    "(ic=%u); two, three and four compute. The fp16 form has no such "
                    "bound: rocket_conv2d_fp16_rk3576()\n", entry, IC);
        return ROCKET_E_UNSUPPORTED;
    }
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
    if (OC != rocket_rk3576_pad_oc(OC)) {
        ROCKET_LOGE("%s: oc=%u is a partial 32-channel group and writes nothing; size "
                    "the output and coefficient buffers for %u channels and pass that "
                    "count (rocket_rk3576_pad_oc)\n",
                    entry, OC, rocket_rk3576_pad_oc(OC));
        return ROCKET_E_UNSUPPORTED;
    }

    in_bytes = (size_t)IH * IW * IC;
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;

    ops   = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    rows  = calloc(max_tasks, sizeof *rows);
    sum_w = calloc(OC, sizeof *sum_w);
    wtile = calloc((size_t)tile * IC * KH * KW, 1);
    if (!ops || !rows || !sum_w || !wtile) { rc = ROCKET_E_NOMEM; goto done; }

    /* Each output channel's whole filter, for the input zero point's fold. */
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

    /* The row window, on the same planner as every other path. Told the precision
     * because the offsets come back in PACKED-IMAGE row units here — an int8 packed
     * image is `ic` interleaved bytes per pixel where a float one is halfwords. */
    plan.ic = (uint16_t)IC; plan.ih = (uint16_t)IH; plan.iw = (uint16_t)IW;
    plan.oc = (uint16_t)tile; plan.oh = (uint16_t)oh; plan.ow = (uint16_t)ow;
    plan.kh = (uint16_t)KH; plan.kw = (uint16_t)KW;
    plan.stride_y = (uint8_t)SY; plan.stride_x = (uint8_t)SX;
    plan.pad_top = (uint8_t)PT; plan.pad_left = (uint8_t)PL;
    plan.ih_full = (uint16_t)IH; plan.oh_full = (uint16_t)oh;
    if (rocket_rk3576_plan_rows_prec(&plan, 0, precision_int8, rows, max_tasks,
                                     &nrow) < 0 || !nrow) {
        ROCKET_LOGE("%s: no row plan for the first conv (ic=%u %ux%u k%ux%u)\n",
                    entry, IC, IW, IH, KW, KH);
        rc = ROCKET_E_UNSUPPORTED; goto done;
    }

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
                for (c = 0; c < IC; c++)
                    img[((size_t)y * IW + x) * IC + c] =
                        in[((size_t)c * IH + y) * IW + x];
    }
    rocket_bo_fini(fd, &b.in);

    for (oc0 = 0; oc0 < OC; oc0 += tile) {
        unsigned n = OC - oc0 < tile ? OC - oc0 : tile;
        size_t w_bytes = rocket_rk3576_weight_argb_int8_bytes(n, KH, KW);
        size_t coeff_bytes = rocket_rk3576_coeff_bytes(n);
        size_t obytes = (size_t)((n + C2 - 1u) / C2) * surf_elems * C2;
        conv_params_t p = {0};
        struct r76_task_extent e;
        uint32_t in_h[4], out_h[1];

        if (b.w.ptr)     rocket_bo_free(fd, &b.w);
        if (b.coeff.ptr) rocket_bo_free(fd, &b.coeff);
        if (b.out.ptr)   rocket_bo_free(fd, &b.out);
        memset(&b.w, 0, sizeof b.w);
        memset(&b.coeff, 0, sizeof b.coeff);
        memset(&b.out, 0, sizeof b.out);
        if (rocket_bo_alloc(fd, w_bytes, &b.w) < 0 ||
            rocket_bo_alloc(fd, coeff_bytes, &b.coeff) < 0 ||
            rocket_bo_alloc(fd, obytes, &b.out) < 0) { rc = ROCKET_E_NOMEM; goto done; }

        /* This tile's channels renumbered from zero — its own whole convolution. */
        memcpy(wtile, W + (size_t)oc0 * IC * KH * KW, (size_t)n * IC * KH * KW);
        rocket_bo_prep(fd, &b.w, 1, 0);
        rc = rocket_rk3576_argb_int8_pack_weights(b.w.ptr, w_bytes, wtile, n, IC, KH, KW);
        rocket_bo_fini(fd, &b.w);
        if (rc < 0) { rc = ROCKET_E_SHAPE; goto done; }

        free(A); free(B);
        A = calloc(n, sizeof *A);
        B = w_zp ? calloc(n, sizeof *B) : NULL;
        if (!A || (w_zp && !B)) { rc = ROCKET_E_NOMEM; goto done; }
        r76_fold_coeff(A, bias, oc0, n, sum_w, in_zp, w_zp, taps);
        if (B) { unsigned j; for (j = 0; j < n; j++) B[j] = (int16_t)(-w_zp); }
        rocket_bo_prep(fd, &b.coeff, 1, 0);
        if (B) rocket_rk3576_pack_coeff_asym(b.coeff.ptr, coeff_bytes, A, n, B, 1);
        else   rocket_rk3576_pack_coeff(b.coeff.ptr, coeff_bytes, A, n);
        rocket_bo_fini(fd, &b.coeff);

        p.ic = (uint16_t)IC; p.iw = (uint16_t)IW; p.ih = (uint16_t)IH;
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

        if (stamp) {
            rocket_bo_prep(fd, &b.out, 1, 0);
            memset(b.out.ptr, stamp, obytes);
            rocket_bo_fini(fd, &b.out);
        }

        for (r = 0; r < nrow; r++) {
            conv_params_t q = p;
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
            rc = r76_submit_task(fd, &b, &q, ops, in_h, 4u, out_h, &e, stamp, entry);
            if (rc != ROCKET_OK) goto done;
        }

        rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
        {
            const int8_t *o = (const int8_t *)b.out.ptr;
            size_t px = (size_t)oh * ow, p;
            unsigned c;
            for (c = 0; c < n; c++) {
                const int8_t *src = o + (size_t)(c / C2) * surf_elems * C2 + (c % C2);
                int8_t *dst = out + (size_t)(oc0 + c) * px;
                for (p = 0; p < px; p++) dst[p] = src[p * C2];
            }
        }
        rocket_bo_fini(fd, &b.out);
    }
    rc = ROCKET_OK;

done:
    free(ops); free(rows); free(A); free(B); free(sum_w); free(wtile);
    r76_conv_free(fd, &b);
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
     * a different feature buffer and a different weight cube. */
    if (d && d->ic <= 4) {
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
                             in_scale, w_scale, out_scale, in_zp, w_zp, out_zp, out);
}

int rocket_conv2d_dw_int8_rk3576(int fd, const rocket_conv2d_desc *d,
                                 const int8_t *in, const int8_t *w, const int32_t *bias,
                                 float in_scale, float w_scale, float out_scale,
                                 int in_zp, int w_zp, int out_zp, int8_t *out)
{
    return r76_conv_int8_run("rocket_conv2d_dw_int8_rk3576", fd, d, 1, in, w, bias,
                             in_scale, w_scale, out_scale, in_zp, w_zp, out_zp, out);
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
    struct r76_task_extent e;
    uint32_t in_h[4], out_h[1];
    unsigned char stamp;
    int rc;

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
            rc = r76_submit_task(fd, &b, &p, ops, in_h, 4u, out_h, &e, stamp, entry);
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
    struct r76_task_extent e;
    uint32_t in_h[4], out_h[1];
    unsigned char stamp;
    int rc;

    rc = r76_conv_check(entry, fd, d, d && d->depthwise, 1, &ow, &oh);
    if (rc != ROCKET_OK) return rc;
    if (!in || !W || !out) return ROCKET_E_SHAPE;
    if (d->depthwise) {
        ROCKET_LOGE("%s: the fp16 depthwise cube is not decoded on this part; the int8 "
                    "depthwise path is (rocket_conv2d_dw_int8_rk3576)\n", entry);
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

    for (s = 0; s < nslice; s++) {
        conv_params_t p = base;

        rocket_bo_prep(fd, &b.w, 1, 0);
        if (rocket_rk3576_fp16_pack_slice_weights(b.w.ptr, w_bytes, W, OC, IC, KH, KW,
                                                  &slices[s]) < 0) {
            rocket_bo_fini(fd, &b.w);
            rc = ROCKET_E_SHAPE; goto done;
        }
        rocket_bo_fini(fd, &b.w);

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
        if (stamp) {
            rocket_bo_prep(fd, &b.out, 1, 0);
            memset(b.out.ptr, stamp, surf);
            rocket_bo_fini(fd, &b.out);
        }
        e.groups      = (ocpad + C2F - 1u) / C2F;
        e.group_bytes = (size_t)ow * oh * C2F * sizeof(_Float16);
        e.row_off     = 0;
        e.span        = e.group_bytes;
        rc = r76_submit_task(fd, &b, &p, ops, in_h, 4u, out_h, &e, stamp, entry);
        if (rc != ROCKET_OK) goto done;

        rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
        if (rocket_rk3576_fp16_accumulate(acc, b.out.ptr, surf, OC, oh, ow) < 0) {
            rocket_bo_fini(fd, &b.out);
            rc = ROCKET_E_SHAPE; goto done;
        }
        rocket_bo_fini(fd, &b.out);
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
