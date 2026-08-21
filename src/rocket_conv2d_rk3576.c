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
                          int dw, unsigned *ow_out, unsigned *oh_out)
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
    if (d->ic <= 4) {
        ROCKET_LOGE("%s: %d input channels takes the CNA's ARGB first-conv "
                    "sub-encoding, whose weight cube is not decoded on this part — the "
                    "register program is gated, the memory it addresses is not\n",
                    entry, d->ic);
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

    rc = r76_conv_check(entry, fd, d, dw, &ow, &oh);
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
    {
        int8_t *cube = (int8_t *)b.in.ptr;
        unsigned c, y, x;
        for (c = 0; c < IC; c++)
            for (y = 0; y < IH; y++)
                for (x = 0; x < IW; x++)
                    cube[feature_data((int)icreg, (int)IH, (int)IW, (int)C2,
                                      (int)c + 1, (int)y + 1, (int)x + 1)] =
                        in[((size_t)c * IH + y) * IW + x];
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
                for (c = 0; c < tile_oc; c++)
                    for (i = 0; i < IC; i++)
                        for (y = 0; y < KH; y++)
                            for (x = 0; x < KW; x++)
                                cube[weight_conv_int8((int)tile_oc, (int)icreg,
                                                      (int)KH, (int)KW, (int)c + 1,
                                                      (int)i + 1, (int)y + 1,
                                                      (int)x + 1)] =
                                    W[((((size_t)(oc0 + c) * IC + i) * KH) + y) * KW + x];
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
        {
            const int8_t *o = (const int8_t *)b.out.ptr;
            unsigned c, y, x;
            for (c = 0; c < tile_oc; c++)
                for (y = 0; y < oh; y++)
                    for (x = 0; x < ow; x++)
                        out[(((size_t)(oc0 + c) * oh) + y) * ow + x] =
                            o[(size_t)(c / C2) * surf_elems * C2 +
                              (size_t)C2 * (y * ow + x) + (c % C2)];
        }
        rocket_bo_fini(fd, &b.out);
    }
    rc = ROCKET_OK;

done:
    free(ops); free(plan); free(A); free(B); free(sum_w);
    r76_conv_free(fd, &b);
    return rc;
}

int rocket_conv2d_int8_rk3576(int fd, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *W, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out)
{
    if (d && d->depthwise)
        return rocket_conv2d_dw_int8_rk3576(fd, d, in, W, bias, in_scale, w_scale,
                                            out_scale, in_zp, w_zp, out_zp, out);
    return r76_conv_int8_run("rocket_conv2d_int8_rk3576", fd, d, 0, in, W, bias,
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

    rc = r76_conv_check(entry, fd, d, d && d->depthwise, &ow, &oh);
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
        unsigned c, y, x;
        for (c = 0; c < IC; c++)
            for (y = 0; y < IH; y++)
                for (x = 0; x < IW; x++)
                    cube[feature_data((int)icpad, (int)IH, (int)IW, (int)C2F,
                                      (int)c + 1, (int)y + 1, (int)x + 1)] =
                        in[((size_t)c * IH + y) * IW + x];
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
