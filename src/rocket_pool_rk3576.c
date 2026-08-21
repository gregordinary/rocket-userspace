// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_pool_rk3576.c — POOLING for the RK3576, behind a library entry.
 *
 * gen_pool_rk3576() emits the PPU's own 31-write program and computes bit-exactly, and
 * for as long as it was reachable only from tests/rk3576_pool_probe.c a caller with a
 * row-major tensor could not use it: the NC1HWC2 cube, the sentinel, the submit and the
 * de-scatter all lived in the probe. This file is that layer, and it is the same shape
 * as rocket_conv2d_rk3576.c's — one entry, row-major in and out, the part's submit
 * discipline owned here rather than by the caller.
 *
 * WHAT THE PART COMPUTES, and so what this entry states:
 *
 *   int8 -> int8, a window reduction with NO requant in the path. The PPU is not a
 *   convolution epilogue on this part: it is 23 PPU writes and 8 PPU_RDMA, no CNA, no
 *   CORE, no DPU, reading and writing the same 16-byte-atom cube the convolution path
 *   already packs.
 *
 *   THE AVERAGE ROUNDS HALF TO EVEN, which is where this diverges from the RK3588's
 *   rocket_pool_int8(): that one truncates toward zero. Two different roundings are two
 *   different functions, so the RK3588 entry refuses here and names this one rather than
 *   dispatching — the same answer rocket_conv2d_int8() and rocket_matmul_int8() give.
 *
 *   THE DIVISOR IS THE WINDOW, NOT THE TAP COUNT. The PPU has no divider; it multiplies
 *   the window sum by a per-axis Q16 reciprocal, 0x10000/kw and 0x10000/kh. So
 *   POOL_METHOD_AVG divides by kh*kw whatever the padding excluded — TFLite's
 *   count-include-pad = TRUE — and ROCKET_RK3576_POOL_AVG_NOPAD is the mode bit that
 *   drops the pad taps instead. THAT RECIPROCAL IS TRUNCATED, and the error it leaves
 *   grows with the window sum: at k2/k3/k5 it is far under half a count over any int8
 *   window and the result is the exact rounded average, and at larger windows it is not
 *   guaranteed to be. rocket_pool_int8_rk3576_exact() answers that question for a
 *   descriptor without running it, from the reciprocal the emitter will program and the
 *   worst-case int8 window sum.
 *
 * THE PAD VALUE IS THE INPUT ZERO POINT on the average path, which is why this entry
 * takes one where the RK3588's does not: a quantized average pool whose input zero point
 * is not zero pads with that value, and the emitter carries the field.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_pool.h"
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"

#define C2 16u      /* the int8 feature/output channel atom */

static unsigned r76p_round4(unsigned v) { return (v + 3u) & ~3u; }

/* How many times a task that wrote nothing is redone. The guard a wide-output job leaves
 * behind is probabilistic — a confirmed power-domain collapse clears it about 87% of the
 * time — so the redo count is the lever. Eight matches the conv and matmul paths. */
static unsigned r76p_attempts(void)
{
    const char *e = getenv("ROCKET_RK3576_TASK_ATTEMPTS");
    long v = e && *e ? strtol(e, NULL, 0) : 8;
    if (v < 1) v = 1;
    if (v > 64) v = 64;
    return (unsigned)v;
}

static int r76p_is_this_chip(const char *entry)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (hw == &rocket_hw_rk3576) return 1;
    ROCKET_LOGE("%s emits the RK3576 PPU program and the active profile is %s\n",
                entry, hw && hw->name ? hw->name : "unknown");
    return 0;
}

int rocket_pool_int8_rk3576_plan(const rocket_pool_desc *d)
{
    if (!d) return ROCKET_E_SHAPE;
    if (d->c <= 0 || d->ih <= 0 || d->iw <= 0 || d->kh <= 0 || d->kw <= 0 ||
        d->stride_y <= 0 || d->stride_x <= 0)
        return ROCKET_E_SHAPE;
    if (d->pad_top < 0 || d->pad_left < 0 || d->pad_bottom < 0 || d->pad_right < 0)
        return ROCKET_E_SHAPE;
    if (d->method != POOL_METHOD_MAX && d->method != POOL_METHOD_AVG)
        return ROCKET_E_SHAPE;
    /* The kernel and stride fields are four bits each; a larger window cascades. */
    if (d->kh > 16 || d->kw > 16 || d->stride_y > 16 || d->stride_x > 16)
        return ROCKET_E_UNSUPPORTED;
    if (d->pad_top > 255 || d->pad_left > 255 ||
        d->pad_bottom > 255 || d->pad_right > 255)
        return ROCKET_E_UNSUPPORTED;
    if (rocket_pool_oh(d) <= 0 || rocket_pool_ow(d) <= 0)
        return ROCKET_E_SHAPE;
    return ROCKET_OK;
}

/*
 * Whether the programmed reciprocal can reproduce the exactly-rounded average for this
 * window, over EVERY int8 input. The PPU multiplies the window sum by 0x10000/kw and
 * 0x10000/kh, both truncated, so the computed quotient is low by
 *
 *     |sum| * (1/(kh*kw) - (floor(2^16/kw)/2^16)*(floor(2^16/kh)/2^16))
 *
 * and the largest |sum| an int8 window can reach is 128*kh*kw. The rounded average moves
 * when that error can cross a half-count boundary; the closest a quotient of an integer
 * by n can come to a half without being one is 1/(2n), so the test is against that.
 * Exact for max, which divides nothing.
 */
int rocket_pool_int8_rk3576_exact(const rocket_pool_desc *d)
{
    unsigned n;
    double rw, rh, err;
    if (rocket_pool_int8_rk3576_plan(d) != ROCKET_OK) return 0;
    if (d->method == POOL_METHOD_MAX) return 1;
    n = (unsigned)d->kh * (unsigned)d->kw;
    rw = (double)(0x10000u / (unsigned)d->kw) / 65536.0;
    rh = (double)(0x10000u / (unsigned)d->kh) / 65536.0;
    err = (1.0 / (double)n - rw * rh) * (128.0 * (double)n);
    if (err < 0.0) err = -err;
    return err < 1.0 / (2.0 * (double)n);
}

int rocket_pool_int8_rk3576(int fd, const rocket_pool_desc *d, int in_zp,
                            const int8_t *in, int8_t *out)
{
    static const char entry[] = "rocket_pool_int8_rk3576";
    unsigned c = (unsigned)d->c, ih = (unsigned)d->ih, iw = (unsigned)d->iw;
    unsigned oh, ow, creg, in_surf, out_surf;
    size_t in_bytes, out_bytes;
    rocket_bo bo_in = {0}, bo_out = {0}, bo_rc = {0};
    uint64_t ops[RK3576_POOL_TASK_OPS];
    uint32_t in_h[2], out_h[1];
    pool_params_rk3576_t p;
    unsigned attempt, attempts, ci, y, x;
    unsigned char stamp;
    int rc = ROCKET_E_DEVICE, cycled = 0, confirmed = 0;

    if (!d || !in || !out) return ROCKET_E_SHAPE;
    if (!r76p_is_this_chip(entry)) return ROCKET_E_UNSUPPORTED;
    rc = rocket_pool_int8_rk3576_plan(d);
    if (rc != ROCKET_OK) return rc;
    if (in_zp < -128 || in_zp > 127) return ROCKET_E_SHAPE;
    if (fd < 0) return ROCKET_E_SHAPE;

    oh = (unsigned)rocket_pool_oh(d);
    ow = (unsigned)rocket_pool_ow(d);
    creg = ((c + C2 - 1u) / C2) * C2;
    in_surf  = r76p_round4(iw * ih);
    out_surf = r76p_round4(ow * oh);
    in_bytes  = (size_t)(creg / C2) * in_surf * C2;
    out_bytes = (size_t)(creg / C2) * out_surf * C2;

    if (rocket_bo_alloc(fd, in_bytes, &bo_in) < 0 ||
        rocket_bo_alloc(fd, out_bytes, &bo_out) < 0 ||
        rocket_bo_alloc(fd, sizeof ops, &bo_rc) < 0) {
        ROCKET_LOGE("%s: BO allocation failed\n", entry);
        rc = ROCKET_E_NOMEM; goto done;
    }

    /* The feature cube. The channel atom is the innermost axis, so a row-major CHW
     * source lands strided however this is written; the loop is ordered to keep the
     * source sequential and the destination's 16-byte stride predictable. */
    rocket_bo_prep(fd, &bo_in, 1, 0);
    memset(bo_in.ptr, 0, in_bytes);
    {
        int8_t *cube = (int8_t *)bo_in.ptr;
        for (ci = 0; ci < c; ci++) {
            const int8_t *src = in + (size_t)ci * ih * iw;
            int8_t *dst = cube + (size_t)(ci / C2) * in_surf * C2 + (ci % C2);
            for (y = 0; y < ih; y++)
                for (x = 0; x < iw; x++)
                    dst[(size_t)C2 * (y * iw + x)] = *src++;
        }
    }
    rocket_bo_fini(fd, &bo_in);

    memset(&p, 0, sizeof p);
    memset(ops, 0, sizeof ops);
    p.iw = (uint16_t)iw; p.ih = (uint16_t)ih; p.c = (uint16_t)c;
    p.ow = (uint16_t)ow; p.oh = (uint16_t)oh;
    p.kw = (uint8_t)d->kw; p.kh = (uint8_t)d->kh;
    p.stride_x = (uint8_t)d->stride_x; p.stride_y = (uint8_t)d->stride_y;
    p.pad_left   = (uint8_t)d->pad_left;   p.pad_right  = (uint8_t)d->pad_right;
    p.pad_top    = (uint8_t)d->pad_top;    p.pad_bottom = (uint8_t)d->pad_bottom;
    p.mode = (uint8_t)(d->method == POOL_METHOD_MAX ? ROCKET_RK3576_POOL_MAX
                                                    : ROCKET_RK3576_POOL_AVG);
    p.input_zero_point = in_zp;
    p.input_dma  = bo_in.dma_address;
    p.output_dma = bo_out.dma_address;
    p.tasks = ops;

    if (gen_pool_rk3576(&p) != 0) {
        ROCKET_LOGE("%s: the generator refused (%ux%u c%u k%ux%u s%ux%u -> %ux%u)\n",
                    entry, iw, ih, c, d->kw, d->kh, d->stride_x, d->stride_y, ow, oh);
        rc = ROCKET_E_UNSUPPORTED; goto done;
    }

    rocket_bo_prep(fd, &bo_rc, 1, 0);
    memcpy(bo_rc.ptr, ops, (size_t)p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &bo_rc);

    in_h[0] = bo_in.handle; in_h[1] = bo_rc.handle;
    out_h[0] = bo_out.handle;

    /* One PPU program, so the write check is over the whole surface. The stamp is what
     * makes "never written" a fact rather than a guess — a fresh BO's zeros cannot tell
     * an unwritten atom from a legitimately zero one — and it is BRACKETED, because a
     * bare memset leaves dirty lines that race the PPU's write DMA. */
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;
    attempts = r76p_attempts();
    rc = ROCKET_E_DEVICE;
    for (attempt = 0; attempt < attempts; attempt++) {
        size_t i;
        int wrote;

        if (stamp) {
            rocket_bo_prep(fd, &bo_out, 1, 0);
            memset(bo_out.ptr, stamp, out_bytes);
            rocket_bo_fini(fd, &bo_out);
        }
        if (rocket_submit_matmul(fd, &bo_rc, p.task_count, in_h, 2, out_h, 1, 2000) != 0) {
            ROCKET_LOGE("%s: submit failed\n", entry);
            rc = ROCKET_E_DEVICE; goto done;
        }
        if (rocket_bo_prep(fd, &bo_out, 0, 2000000000ull) < 0) {
            ROCKET_LOGE("%s: PREP_BO on the output timed out\n", entry);
            rc = ROCKET_E_DEVICE; goto done;
        }
        if (!stamp) { rc = ROCKET_OK; break; }
        wrote = 0;
        for (i = 0; i < out_bytes; i++)
            if (((const unsigned char *)bo_out.ptr)[i] != stamp) { wrote = 1; break; }
        rocket_bo_fini(fd, &bo_out);
        if (wrote) { rc = ROCKET_OK; break; }
        ROCKET_LOGD("%s: the program wrote nothing on attempt %u; cycling the power "
                    "domain and redoing it\n", entry, attempt + 1u);
        cycled++;
        confirmed += rocket_rk3576_power_idle();
    }
    if (rc != ROCKET_OK) {
        ROCKET_LOGE("%s: the program wrote nothing over %u attempts (%d power cycles, "
                    "%d of them confirmed to reach suspended)\n",
                    entry, attempts, cycled, confirmed);
        goto done;
    }

    {
        const int8_t *o = (const int8_t *)bo_out.ptr;
        for (ci = 0; ci < c; ci++) {
            const int8_t *src = o + (size_t)(ci / C2) * out_surf * C2 + (ci % C2);
            int8_t *dst = out + (size_t)ci * oh * ow;
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++)
                    *dst++ = src[(size_t)C2 * (y * ow + x)];
        }
    }
    rc = ROCKET_OK;

done:
    if (bo_rc.ptr)  rocket_bo_free(fd, &bo_rc);
    if (bo_out.ptr) rocket_bo_free(fd, &bo_out);
    if (bo_in.ptr)  rocket_bo_free(fd, &bo_in);
    return rc;
}

/*
 * The CPU model of the above: signed integer max, or an average over the WINDOW that
 * rounds half to even. Against round-half-away-from-zero a k2 average disagrees on one
 * output in eight, so this is the rule and not a detail. An odd window has no exact
 * half and so no tie to break.
 */
void rocket_pool_ref_int8_rk3576(const rocket_pool_desc *d, int in_zp,
                                 const int8_t *in, int8_t *out)
{
    int oh = rocket_pool_oh(d), ow = rocket_pool_ow(d);
    int c, y, x, kh, kw;

    for (c = 0; c < d->c; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                long best = -1000, sum = 0;
                long n = (long)d->kh * d->kw;
                for (kh = 0; kh < d->kh; kh++)
                    for (kw = 0; kw < d->kw; kw++) {
                        int iy = y * d->stride_y + kh - d->pad_top;
                        int ix = x * d->stride_x + kw - d->pad_left;
                        int v;
                        if (iy < 0 || ix < 0 || iy >= d->ih || ix >= d->iw)
                            v = (d->method == POOL_METHOD_MAX) ? -128 : in_zp;
                        else
                            v = in[((size_t)c * d->ih + iy) * d->iw + ix];
                        if (v > best) best = v;
                        sum += v;
                    }
                if (d->method == POOL_METHOD_MAX) {
                    out[((size_t)c * oh + y) * ow + x] = (int8_t)best;
                } else {
                    long half = n / 2, q, r;
                    q = sum >= 0 ? (sum + half) / n : -(((-sum) + half) / n);
                    if ((n & 1) == 0) {
                        r = sum - q * n;
                        if ((r == half || r == -half) && (q & 1))
                            q += (sum >= 0) ? -1 : 1;
                    }
                    out[((size_t)c * oh + y) * ow + x] = (int8_t)q;
                }
            }
}
