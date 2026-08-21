// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_matmul_rk3576.c — the int8 matmul for the RK3576.
 *
 * A matmul is a 1x1 convolution over this part's CNA->CORE->DPU blocks, exactly as it
 * is on the RK3588; what differs is the geometry-register encoding, the operand cubes,
 * and the envelope. The register program is not the new part here — the RK3576 already
 * computes 1x1 convolutions bit-exactly — so this file is the userspace layer around
 * one: the operand scatter, the output de-scatter, the tiling planner, and the
 * per-chip dispatch point.
 *
 * WHY int8 AND NOT fp16, which is the opposite of the RK3588's answer. The two
 * precisions contract at wildly different rates on this part. One int8 task takes
 * ic*kh*kw up to 4608, so a K of 4608 lands in a SINGLE submit; one fp16 task
 * contracts exactly sixteen input channels, so the same K costs 288 submits at about
 * 1.4 ms each. Measured on one shape at both precisions, fp16 runs 30x slower at
 * K=512 and 300x slower at K=4608. int8 is the matmul precision here.
 * [HW sweep, H96 MAX M9]
 *
 * WHAT ONE SUBMIT COSTS, and what that makes the planner's job. A submit is about
 * 1.4 ms whatever it carries, so throughput is MACs per submit and nothing else. Two
 * of the three axes are capped: M*K by the CBUF feature budget at 262144 int8
 * elements, and K by the resident weight slice at 4608. That leaves N, and N is what
 * the planner should spend — measured at the feature-budget cap, throughput rises
 * almost linearly with it, from 12 GOP/s at N=32 to about 1.0 TOP/s at N=2560.
 * [HW sweep, H96 MAX M9]
 *
 * THE M AXIS CARRIES NO CONSTRAINT, which is also the opposite of the RK3588. There
 * rows are the conv's spatial height, a height under 4 mis-computes, and software pads
 * M==1 to 4. Here M=1, 2, 3, 5, 7, 17 and 31 are each bit-exact, and so is every
 * factorization of M into a plane — a single row of pixels, a single column, and
 * everything between. The plane is a TILING choice, not a correctness one.
 * [HW sweep, H96 MAX M9]
 *
 * THE OUTPUT IS int8, THROUGH THE DPU's REQUANT, at every K one task can contract, and
 * that is the fast path. The RK3588's int8 matmul writes int32 and sums K partials on
 * the host; this part's DPU will do that too, but its 32-bit writer delivers only some
 * of the output channels — the byte budget is a function of the DPU's operand width and
 * not of the output element's. The way round it is the weight cube rather than a
 * register — see the int32 section below — and it costs the output-channel axis twice
 * over. So K past one task's contraction is SPLIT rather than refused, through that
 * path, and the requant moves to the host.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"

#define C2 16

/* The idle a poisoned submit needs and whether the output surface is stamped before it
 * runs are properties of the CHIP, not of the matmul: the convolution entry points hit
 * the same hazard through the same driver. Both live in rocket_rk3576_internal.h, and
 * the int32 section below is where they are documented and defined. */
#define R76_SENTINEL_BYTE ROCKET_RK3576_SENTINEL_BYTE

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);
int weight_conv_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

/* Both machine parameters come from the profile, not from literals here: the feature
 * budget in int8 elements is the CBUF data allocation, and the output-channel tile is
 * the profile's max_tile. ROCKET_RK3576_MM_NT overrides the tile per run. */
static unsigned r76_mm_feature_elems(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    return (unsigned)(hw->cbuf_banks * hw->cbuf_bank_size);
}

/* ============================================================================
 * SECTION — the plane
 *
 * M rows become a plane of M pixels, and any (iw, ih) with iw*ih == M carries them.
 * What the choice costs is granules: a feature row is ceil(iw*K/64) of them, against
 * the profile's CBUF data allocation, so the pixels one task can carry come out the
 * same however the plane is cut — PROVIDED iw*K is a whole number of granules. When it
 * is not, every row of the plane rounds up and the waste is real, so the widest
 * divisor of M that divides evenly is the one to take.
 * ==========================================================================*/
static void r76_mm_plane(unsigned M, unsigned K, unsigned *iw_out, unsigned *ih_out)
{
    unsigned budget_px = r76_mm_feature_elems() / K;
    unsigned iw, best = 1;

    if (!budget_px) budget_px = 1;
    for (iw = 1; iw <= M; iw++) {
        if (M % iw) continue;
        if (iw > budget_px) break;
        if ((iw * K) % 64u) continue;
        best = iw;
    }
    *iw_out = best;
    *ih_out = M / best;
}

static unsigned r76_mm_nt(void)
{
    const char *e = getenv("ROCKET_RK3576_MM_NT");
    long v = (e && *e) ? strtol(e, NULL, 0) : 0;
    if (v > 0) return (unsigned)v;
    return (unsigned)rocket_hw_current()->max_tile;
}

/* ============================================================================
 * SECTION — the plan
 * ==========================================================================*/
/* The output-channel tile this shape can actually run, or 0 if none can.
 *
 * Shrinking rather than refusing is the whole of the planner's job on this axis. The
 * emitter's guards — the resident weight slice, the 2944-channel bound, the 6 MiB
 * weight cube — are all functions of the TILE's oc, so a tile that is too wide is not
 * an unsupported shape, it is a tile to halve. What no tile can fix is K: it is the
 * contraction, the output is int8 through the requant, and a partial cannot be summed
 * without quantizing it. */
static unsigned r76_mm_fit_nt_mult(unsigned iw, unsigned K, unsigned N,
                                   unsigned oc_mult, unsigned *rows_out)
{
    unsigned nt = r76_mm_nt();

    if (nt > N) nt = N;
    nt = (nt / 32u) * 32u;
    if (!nt) nt = 32u;

    for (;;) {
        unsigned rows = rocket_rk3576_max_task_rows(iw, K, nt * oc_mult, 1u, 1u, 0);
        if (rows) { if (rows_out) *rows_out = rows; return nt; }
        if (nt <= 32u) return 0;
        nt = ((nt / 2u) / 32u) * 32u;
        if (!nt) nt = 32u;
    }
}

static unsigned r76_mm_fit_nt(unsigned iw, unsigned K, unsigned N, unsigned *rows_out)
{
    return r76_mm_fit_nt_mult(iw, K, N, 1u, rows_out);
}

int rocket_matmul_plan_int8_rk3576(int M, int K, int N, int *Mt, int *Kt, int *Nt)
{
    unsigned iw, ih, nt, rows = 0;

    if (M <= 0 || K <= 0 || N <= 0) return ROCKET_E_SHAPE;
    if (K % 32 || N % 32) {
        ROCKET_LOGE("rk3576 matmul: K=%d N=%d — both must be multiples of 32 (the "
                    "int8 weight cube groups each channel axis by 32)\n", K, N);
        return ROCKET_E_SHAPE;
    }

    r76_mm_plane((unsigned)M, (unsigned)K, &iw, &ih);
    nt = r76_mm_fit_nt(iw, (unsigned)K, (unsigned)N, &rows);
    if (!nt) {
        /* Not an error: this entry reports the SINGLE-TASK plan, and a K past one
         * task's contraction has one — it is split through the int32 writer instead.
         * rocket_matmul_int8_rk3576() takes that route when this returns E_SHAPE, so
         * saying so at error level would shout on a supported path. */
        ROCKET_LOGI("rk3576 matmul: K=%d does not fit one task even at a 32-channel "
                    "output tile, so there is no single-task plan to report. "
                    "rocket_matmul_int8_rk3576() still runs it, splitting K through "
                    "the int32 writer and requantizing on the host\n", K);
        return ROCKET_E_SHAPE;
    }

    if (Mt) *Mt = (int)(rows * iw);
    if (Kt) *Kt = K;
    if (Nt) *Nt = (int)nt;
    return (int)(((unsigned)N + nt - 1u) / nt * ((ih + rows - 1u) / rows));
}

/* ============================================================================
 * SECTION — one N tile
 * ==========================================================================*/
struct r76_mm_bos {
    rocket_bo in, w, coeff, out, rc;
};

static void r76_mm_free(int fd, struct r76_mm_bos *b)
{
    if (b->rc.ptr)    rocket_bo_free(fd, &b->rc);
    if (b->out.ptr)   rocket_bo_free(fd, &b->out);
    if (b->coeff.ptr) rocket_bo_free(fd, &b->coeff);
    if (b->w.ptr)     rocket_bo_free(fd, &b->w);
    if (b->in.ptr)    rocket_bo_free(fd, &b->in);
    memset(b, 0, sizeof *b);
}

int rocket_matmul_int8_rk3576(int fd, int M, int K, int N,
                              const int8_t *A, const int8_t *B,
                              const int32_t *bias, float scale, int8_t *C)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    struct r76_mm_bos b = {0};
    uint64_t *ops = NULL;
    int8_t *stage = NULL;
    int32_t *tile_bias = NULL;
    rocket_rk3576_row_task *plan = NULL;
    unsigned iw, ih, nt, n0, surf_elems, max_tasks;
    size_t in_bytes;
    unsigned char blank = rocket_rk3576_sentinel_on() ? (unsigned char)R76_SENTINEL_BYTE : 0;
    int rc = ROCKET_E_SHAPE;

    if (fd < 0 || !A || !B || !C) return ROCKET_E_SHAPE;
    if (strcmp(hw->name, "rk3576") != 0) {
        ROCKET_LOGE("rocket_matmul_int8_rk3576: this is the RK3576 encoding and the "
                    "active profile is %s\n", hw->name);
        return ROCKET_E_UNSUPPORTED;
    }
    if (scale <= 0.0f) {
        ROCKET_LOGE("rk3576 matmul: scale must be positive (the DPU's OUT_CVT gates "
                    "the whole BS stage off at zero and writes an empty surface)\n");
        return ROCKET_E_SHAPE;
    }
    {
        int ntile = 0;
        if (rocket_matmul_plan_int8_rk3576(M, K, N, NULL, NULL, &ntile) < 0) {
            /* A K past one task's contraction is not an unsupported shape any more: run
             * it through the int32 writer, which splits K, and requant on the host. The
             * requant is then the host's float rounding rather than the DPU's OUT_CVT
             * approximation of it — a different arithmetic on this side of the boundary,
             * and the more accurate one. */
            int32_t *acc;
            int m, rc32;
            if (M <= 0 || N <= 0) return ROCKET_E_SHAPE;
            acc = calloc((size_t)M * N, sizeof *acc);
            if (!acc) return ROCKET_E_NOMEM;
            rc32 = rocket_matmul_int8_rk3576_i32(fd, M, K, N, A, B, bias, acc);
            if (rc32 == ROCKET_OK) {
                for (m = 0; m < M * N; m++) {
                    float v = (float)acc[m] * scale;
                    long r = (long)(v < 0.0f ? v - 0.5f : v + 0.5f);
                    C[m] = (int8_t)(r < -128 ? -128 : (r > 127 ? 127 : r));
                }
            }
            free(acc);
            return rc32;
        }
        nt = (unsigned)ntile;
    }

    r76_mm_plane((unsigned)M, (unsigned)K, &iw, &ih);
    surf_elems = rocket_rk3576_out_surf_elems(iw, ih, 0);
    in_bytes   = (size_t)(((unsigned)K + C2 - 1) / C2) * ih * iw * C2;
    max_tasks  = ih + 1u;

    ops   = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    plan  = calloc(max_tasks, sizeof *plan);
    stage = calloc(in_bytes, 1);
    if (!ops || !plan || !stage) { rc = ROCKET_E_NOMEM; goto done; }

    /* The FEATURE cube is packed once and shared by every N tile: the tiling is on the
     * output-channel axis, which the feature side does not see. */
    {
        int m, k;
        for (m = 0; m < M; m++)
            for (k = 0; k < K; k++)
                stage[feature_data(K, (int)ih, (int)iw, C2,
                                   k + 1, (int)(m / (int)iw) + 1, (int)(m % (int)iw) + 1)] =
                    A[(size_t)m * K + k];
    }
    if (rocket_bo_alloc(fd, in_bytes, &b.in) < 0) { rc = ROCKET_E_NOMEM; goto done; }
    rocket_bo_prep(fd, &b.in, 1, 0);
    memcpy(b.in.ptr, stage, in_bytes);
    rocket_bo_fini(fd, &b.in);
    free(stage);
    stage = NULL;

    if (rocket_bo_alloc(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }

    for (n0 = 0; n0 < (unsigned)N; n0 += nt) {
        unsigned tile_n = (unsigned)N - n0 < nt ? (unsigned)N - n0 : nt;
        unsigned nreg = rocket_rk3576_pad_oc(tile_n);
        size_t w_bytes = (size_t)((nreg + 31) / 32) * (((unsigned)K + 31) / 32) * 32 * 32;
        size_t coeff_bytes = rocket_rk3576_coeff_bytes(nreg);
        size_t obytes = (size_t)((nreg + C2 - 1) / C2) * surf_elems * C2;
        conv_params_t p = {0};
        uint32_t in_h[4], out_h[1];
        unsigned ntask = 1, t, n, k;

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
         * group count follows the tile rather than the whole N. */
        rocket_bo_prep(fd, &b.w, 1, 0);
        memset(b.w.ptr, 0, w_bytes);
        for (n = 0; n < tile_n; n++)
            for (k = 0; k < (unsigned)K; k++)
                ((int8_t *)b.w.ptr)[weight_conv_int8((int)tile_n, K, 1, 1,
                                                     (int)n + 1, (int)k + 1, 1, 1)] =
                    B[(size_t)(n0 + n) * K + k];
        rocket_bo_fini(fd, &b.w);

        /* The COEFFICIENT buffer is NOT a flat int32 bias array on this part, and a
         * zeroed one makes the DPU write a full but entirely empty surface whatever
         * the MAC did — the C term gates the BS stage. Pad the tail channels of a
         * partial group with a zero bias so they carry a C term too. */
        free(tile_bias);
        tile_bias = calloc(nreg, sizeof *tile_bias);
        if (!tile_bias) { rc = ROCKET_E_NOMEM; goto done; }
        if (bias)
            for (n = 0; n < tile_n; n++) tile_bias[n] = bias[n0 + n];
        rocket_bo_prep(fd, &b.coeff, 1, 0);
        rocket_rk3576_pack_coeff(b.coeff.ptr, coeff_bytes, tile_bias, nreg);
        rocket_bo_fini(fd, &b.coeff);

        p.ic = (uint16_t)K;    p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
        p.oc = (uint16_t)nreg; p.oh = (uint16_t)ih; p.ow = (uint16_t)iw;
        p.kh = 1; p.kw = 1;
        p.stride_y = 1; p.stride_x = 1;
        p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)ih;
        p.int8_out = 1;
        /* The requant the caller asked for. conv_scale = in*w/out is what the emitter
         * derives OUT_CVT from, so a unit in/w and out = 1/scale puts the caller's
         * factor there directly. Per-tensor: the DPU's OUT_CVT has no per-channel
         * multiplier on this path. */
        p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f / scale;
        p.input_zero_point = 0x80;   /* symmetric int8: every zero-point term cancels */
        p.output_zero_point = 0x80;
        p.weight_zero_point = 0x80;
        p.tasks       = ops;
        p.input_dma   = b.in.dma_address;
        p.weights_dma = b.w.dma_address;
        p.bias_dma    = b.coeff.dma_address;
        p.output_dma  = b.out.dma_address;

        {
            conv_params_t q = p;
            if (rocket_rk3576_plan_rows(&q, 0, plan, max_tasks, &ntask) < 0) {
                ROCKET_LOGE("rk3576 matmul: no row plan for M=%d K=%d N tile %u "
                            "(plane %ux%u)\n", M, K, tile_n, iw, ih);
                rc = ROCKET_E_SHAPE; goto done;
            }
        }

        in_h[0] = b.in.handle; in_h[1] = b.w.handle;
        in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
        out_h[0] = b.out.handle;

        /* THE POISONING REACHES THIS PATH TOO, and it used to pass straight through.
         * An int32-output job leaves the next submit of ANY kind writing nothing, across
         * calls and across processes, so the plain int8 matmul inherits the hazard from
         * whatever ran before it — its first submit comes back untouched and the caller
         * reads a correctly sized, entirely stale tile with no fault and normal timing.
         * Measured in a soak: one whole gate run in twenty failed its first shape this
         * way, at 1 element of 256 correct.
         *
         * Stamping the surface is what makes that detectable without false positives. A
         * region that is still the sentinel throughout is a submit that did not write;
         * an all-zero region is a legitimate int8 result after the requant and must not
         * be read as failure. The stamp is bracketed by PREP_BO and FINI_BO so no dirty
         * line is left to race the DPU's DMA. */
        if (blank) {
            rocket_bo_prep(fd, &b.out, 1, 0);
            memset(b.out.ptr, blank, obytes);
            rocket_bo_fini(fd, &b.out);
        }

        for (t = 0; t < ntask; t++) {
            unsigned tattempt;
            conv_params_t q = p;
            q.ih = plan[t].ih; q.oh = plan[t].oh;
            q.pad_top = plan[t].pad_top;
            q.input_dma  = p.input_dma  + plan[t].feature_off;
            q.output_dma = p.output_dma + plan[t].output_off;
            /* ALWAYS, not only when the plan split: the emitter derives the DDR
             * channel-group stride from the FULL plane, and leaving these at the
             * window makes every group past the first read at the wrong offset. */
            q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
            if (gen_conv2d_int8_rk3576(&q) != 0) { rc = ROCKET_E_SHAPE; goto done; }
            /* PER TASK, not per tile. One poisoned submit among several leaves its own
             * rows stale while its siblings are full, so a check over the whole tile
             * reads "something was written" and passes the hole through. */
            for (tattempt = 0; tattempt < 4; tattempt++) {
                rocket_bo_prep(fd, &b.rc, 1, 0);
                memcpy(b.rc.ptr, ops, q.task_count * sizeof(uint64_t));
                rocket_bo_fini(fd, &b.rc);
                if (rocket_submit_matmul(fd, &b.rc, q.task_count, in_h, 4, out_h, 1, 4000) != 0) {
                    rc = ROCKET_E_DEVICE; goto done;
                }
                if (rocket_bo_prep(fd, &b.out, 0, 2000000000ull) < 0) {
                    rc = ROCKET_E_DEVICE; goto done;
                }
                if (blank) {
                    const unsigned char *sp = (const unsigned char *)b.out.ptr +
                                              plan[t].output_off;
                    size_t tb = (size_t)plan[t].oh * iw * C2, si;
                    int twrote = 0;
                    for (si = 0; si < tb; si++) if (sp[si] != blank) { twrote = 1; break; }
                    rocket_bo_fini(fd, &b.out);
                    if (twrote) break;
                    ROCKET_LOGD("rk3576 matmul: n0=%u row task %u wrote nothing, idling "
                                "and redoing it\n", n0, t);
                    rocket_rk3576_power_idle();
                    continue;
                }
                rocket_bo_fini(fd, &b.out);
                break;
            }
        }

        /* De-scatter this tile's channels straight into the caller's row-major C. */
        rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
        {
            const int8_t *o = (const int8_t *)b.out.ptr;
            int m;
            for (m = 0; m < M; m++) {
                unsigned y = (unsigned)m / iw, x = (unsigned)m % iw;
                for (n = 0; n < tile_n; n++)
                    C[(size_t)m * N + n0 + n] =
                        o[(size_t)(n / C2) * surf_elems * C2 +
                          (size_t)C2 * (y * iw + x) + (n % C2)];
            }
        }
        rocket_bo_fini(fd, &b.out);
    }
    rc = ROCKET_OK;

done:
    free(ops); free(plan); free(stage); free(tile_bias);
    r76_mm_free(fd, &b);
    return rc;
}

/* ============================================================================
 * SECTION — the int32 output, and the K split it buys
 *
 * The DPU will emit its raw 32-bit accumulator on an integer program. What it will not
 * do is write all of them. Its write budget is one 16-byte atom per (16-channel block,
 * pixel) — the INT8 surface — whatever the OUTPUT element width is, so at four bytes an
 * element most channels never reach DDR. What that budget IS a function of is the DPU's
 * own operand width, PROC_PRECISION, and widening that to int32 doubles it. Two writers
 * come out of it, and the arithmetic is bit-identical in both:
 *
 *   gen_conv2d_int8_rk3576_i32out()       one atom:  the first 8 channels of every 32
 *   gen_conv2d_int8_rk3576_i32out_wide()  two atoms: the first 8 channels of every 16
 *
 * Two atoms is the ceiling — int16, fp16 and bfloat16 reach it too, the float ones
 * destroying the arithmetic on the way, and int4 and int8 do not. [HW sweep, H96 MAX M9]
 *
 * THE WAY AROUND IT IS THE WEIGHT CUBE, not a register. Program a multiple of the output
 * channels and put real channel n in a slot the writer delivers, leaving the rest zero.
 * Every real channel then lands in a delivered slot and the surface reads back as a
 * plain cube. The narrow writer wants 4x and slot 32*(n/8)+n%8, which collapses to the
 * RK3588's int32 cube `(n/4)*ow*oh_full*4 + 4*p + n%4`; the wide one wants 2x and slot
 * 16*(n/8)+n%8, addressed with rocket_rk3576_i32_wide_word().
 *
 * WHAT IT COSTS is the output-channel axis, spent over. The bytes are not wasted — the
 * budget comes out exactly the int32 surface — but the resident weight slice, the N tile
 * and the 2944-channel bound are all functions of the PROGRAMMED oc, so the multiplier
 * is paid in MACs per submit. Halving it from 4 to 2 doubles the N tile and so halves
 * the submits, which is what this path is billed in: measured 4 submits to 2 at
 * M=64 K=1024 N=2048, and 8 to 4 at M=32 K=1024 N=4096. Below the tile cap the two cost
 * the same. [HW sweep, H96 MAX M9]
 *
 * THE WIDE WRITER'S SURFACE HEIGHT IS BOUNDED, at oh_full * oc < 4096, and past it the
 * map silently stops being the map. The bound is measured; the mechanism is not decoded.
 * A row task here is a standalone 1x1 convolution with its own surface, so the planner
 * below honours the bound by splitting the row plan further.
 * ==========================================================================*/

/* The idle an i32out job needs before the NEXT submit will write.
 *
 * IT IS NOT A SETTLING TIME. What clears the hazard is the driver's runtime-PM
 * autosuspend cycling the NPU power domain, so the idle that works is the driver's
 * autosuspend delay plus the suspend/resume round trip — and with runtime suspend
 * turned off entirely (`power/control` = `on`) no amount of idle clears it at all. See
 * the submit loop for the measurement.
 *
 * So the default is read from the driver rather than fixed: twice the autosuspend delay
 * plus 20 ms covers it at every delay swept (5, 10, 20 and 50 ms), and a system that
 * lowers the delay gets a proportionally cheaper path for free. ROCKET_RK3576_MM_GAP_MS
 * overrides it outright. */
static int r76_autosuspend_ms(void)
{
    static const char *const PATHS[] = {
        "/sys/bus/platform/drivers/rocket/27700000.npu/power/autosuspend_delay_ms",
        "/sys/bus/platform/drivers/rocket/27f00000.npu/power/autosuspend_delay_ms",
    };
    unsigned i;
    for (i = 0; i < sizeof PATHS / sizeof PATHS[0]; i++) {
        FILE *fp = fopen(PATHS[i], "r");
        long v;
        if (!fp) continue;
        if (fscanf(fp, "%ld", &v) == 1 && v >= 0 && v < 10000) { fclose(fp); return (int)v; }
        fclose(fp);
    }
    return -1;
}

void rocket_rk3576_power_idle(void)
{
    static int cached = -2;
    const char *g = getenv("ROCKET_RK3576_MM_GAP_MS");
    int gms;
    struct timespec ts;

    if (g && *g) {
        gms = (int)strtol(g, NULL, 0);
    } else {
        if (cached == -2) {
            int d = r76_autosuspend_ms();
            cached = d >= 0 ? 2 * d + 20 : 150;
            if (cached < 20) cached = 20;
            if (cached > 300) cached = 300;
            ROCKET_LOGI("rk3576 matmul i32: the 32-bit writer's hazard is cleared by the "
                        "NPU power domain cycling, so the inter-submit idle is sized from "
                        "the driver's autosuspend delay (%d ms) at %d ms\n", d, cached);
        }
        gms = cached;
    }
    if (gms <= 0) return;
    ts.tv_sec = gms / 1000;
    ts.tv_nsec = (long)(gms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

/* How many programmed output channels one real channel costs.
 *
 * FOUR is the default: the narrow writer delivers the first eight output channels of
 * every thirty-two. TWO is the wide writer (DPU PROC_PRECISION widened to int32), which
 * delivers the first eight of every sixteen and so halves the programmed oc — worth
 * double the N tile and half the submits. `ROCKET_RK3576_I32_OC_MULT=2` selects it.
 *
 * IT IS OPT-IN BECAUSE IT INTERMITTENTLY EMITS ZEROS. On about one run in ten of the
 * shape that shows it most, the wide writer emits a contiguous block of one 32-channel
 * super-group's stream with zero data while the rest of the surface is exact.
 *
 * IT DOES NOT DROP WRITES, which is what this looked like before the surface was stamped.
 * Against a sentinel written into the output BO beforehand — and verified to reach DDR —
 * the bad atoms come back ZERO rather than holding the sentinel, so the writer reached
 * them and the data was wrong. It is not a readback race either: a second fence and a
 * second read return byte-identical contents. And it is not the poisoning above, whose
 * signature is an empty region, which the power domain cycling clears, and which this is
 * indifferent to — 3 failures in 30 at no idle against 6 in 30 at 800 ms.
 * [HW sweep, H96 MAX M9]
 *
 * IT IS REPAIRED BY REDOING THE TASK, and r76_i32_wide_suspect() is what sees it. The
 * corruption is contiguous in the writer's EMISSION order and in no other, lies inside a
 * single super-group, and ends two emissions short of that group's last; a redo lands the
 * task exactly, with no idle in front of it. Measured over 40 runs of the shape that
 * fails most: 0 failures, 3 runs redoing one task. [HW sweep, H96 MAX M9]
 *
 * The narrow writer is also the negative control for the wide one: a result that differs
 * between the two says one of the two output maps is wrong and takes the arithmetic out
 * of the question. */
static unsigned r76_i32_oc_mult(void)
{
    const char *e = getenv("ROCKET_RK3576_I32_OC_MULT");
    return (e && *e && strtol(e, NULL, 0) == 2) ? 2u : 4u;
}

/* Programmed output channel carrying real output channel n. Both writers deliver the
 * low eight of a group; the group is 32 channels wide on the narrow one and 16 on the
 * wide one. */
static unsigned r76_i32_prog_oc(unsigned n, unsigned mult)
{
    return (mult == 4u ? 32u : 16u) * (n / 8u) + (n % 8u);
}
/* One task's contraction is bounded by the resident weight slice, which the 4x oc
 * multiplier eats into directly. Slices descend by halves from this. */
#define R76_I32_KS_MAX    4608u
/* The wide writer's surface-height bound, as oh_full * oc. See where it is applied. */
#define R76_I32_WIDE_OC_ROWS  4095u

/* ROCKET_RK3576_I32_DIAG locates the wide writer's dropped atoms in the coordinates of
 * its OWN map rather than in (m, n).
 *
 * The detector needs no expected values. Every atom of a wide surface is a delivered
 * atom — the writer emits `oc/16 * 2` atoms per pixel and the map covers exactly those —
 * and the scatter puts a real output channel in every delivered slot, so with operands
 * that are not degenerate each 16-byte atom holds four accumulators that are all zero
 * only by coincidence. An all-zero atom is therefore a write that did not happen, and
 * the scan below names it as (super-group, run, offset-in-run, lane group, pixel).
 *
 * At level 2 each task's surface is also given a guard band of its own, so an atom that
 * was written PAST its task's surface shows up as a nonzero guard rather than as
 * nothing at all — that separates a dropped write from an address-generation slip. */
static int r76_i32_diag(void)
{
    const char *e = getenv("ROCKET_RK3576_I32_DIAG");
    return (e && *e) ? (int)strtol(e, NULL, 0) : 0;
}
#define R76_DIAG_TASK_GUARD 256u

/* The surface is stamped with a sentinel before the tasks run, so "this atom was never
 * written" is a property of the surface rather than an inference from its value.
 *
 * That is what makes the wide writer's dropped atoms repairable. A fresh BO arrives
 * zeroed, and zero is also a legitimate accumulator, so a zeroed surface can only
 * answer "did this task write anything at all" — enough to catch a poisoned submit,
 * blind to a task that wrote all but thirteen of its atoms. Against a sentinel the
 * question becomes per atom and the answer is exact, so the retry that already covers
 * the poisoning covers the drop as well.
 *
 * Stamping it is safe here for the same reason the operand buffers are: the fill is
 * bracketed by PREP_BO and FINI_BO, so the lines are written back before the submit and
 * none are left dirty to race the DPU's DMA. Zeroing an output BO with a bare memset —
 * no FINI_BO — is the trap, and it is a different thing.
 *
 * The value is chosen to be an implausible accumulator rather than an impossible one. A
 * real accumulator that happens to equal it costs one wasted submit and stays correct. */
int rocket_rk3576_sentinel_on(void)
{
    const char *e = getenv("ROCKET_RK3576_I32_SENTINEL");
    /* On by default on the wide path: it is what makes that path's dropped atoms
     * visible, and so what lets the retry heal them. =0 turns it off. */
    return (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 1;
}

/* One task's surface, scanned for atoms the writer never wrote. Returns how many. */
static unsigned r76_i32_diag_task(const unsigned char *surf, unsigned A,
                                  unsigned oc_prog, unsigned atoms_per_px,
                                  unsigned k0, unsigned n0, unsigned t,
                                  unsigned char want, const char *label, int quiet)
{
    unsigned a, natoms = atoms_per_px * A, nzero = 0;
    long prev = -2;

    for (a = 0; a < natoms; a++) {
        const unsigned char *ap = surf + (size_t)a * C2;
        unsigned i, sg, rem, r, off, s, p, j, L, c0;
        for (i = 0; i < C2; i++) if (ap[i] != want) break;
        if (i != C2) continue;
        if (quiet) { nzero++; continue; }
        sg  = A ? a / (4u * A) : 0u;
        rem = a - sg * 4u * A;
        r   = rem / A;
        off = rem % A;
        L   = r % 2u;
        s   = (r / 2u) * A + off;
        p   = s / 2u;
        j   = s % 2u;
        c0  = 32u * sg + 16u * j + 4u * L;
        ROCKET_LOGI("rk3576 i32 diag: k0=%u n0=%u task=%u %s atom %u "
                    "byte %zu (mod64 %zu) A=%u oc=%u | super-group %u run %u "
                    "off-in-run %u lane-group %u | s=%u pixel %u block %u "
                    "channels %u..%u%s\n",
                    k0, n0, t, label, a, (size_t)a * C2, ((size_t)a * C2) % 64u, A, oc_prog,
                    sg, r, off, L, s, p, j, c0, c0 + 3u,
                    prev == (long)a - 1 ? " (contiguous with the last)" : "");
        prev = (long)a;
        nzero++;
    }
    return nzero;
}

/* The wide writer's corruption signature, in the coordinates of its own emission order.
 *
 * WHAT THE DEFECT IS. On a minority of runs the wide writer emits a contiguous block of
 * one 32-channel super-group's stream with ZERO data. It is not a dropped write: on a
 * surface stamped with a sentinel beforehand — and the stamp is verified to reach DDR —
 * those atoms come back zero rather than holding the sentinel, so the writer reached
 * them and the data was wrong. Nor is it a readback race, since a second fence and a
 * second read return byte-identical contents; nor the poisoning, whose signature is an
 * empty region and which is cleared by the power domain cycling, where this is
 * indifferent to the idle ahead of it (3 failures in 30 at no idle, 6 in 30 at 800 ms).
 * [HW sweep, H96 MAX M9]
 *
 * WHAT IT LOOKS LIKE. The writer emits two atoms per stream position s = 2*pixel +
 * block, one per lane group, so its emission order is (s, L) ascending. The corrupt
 * block is contiguous in THAT order and in no other — in address order it appears as two
 * separate holes — it lies inside a single super-group, and it always ends at s = 2A-2,
 * two emissions short of that super-group's last. Its start varies run to run, from two
 * emissions to most of the group. Measured over 40 runs of one shape, every failure had
 * that shape. [HW sweep, H96 MAX M9]
 *
 * WHAT THIS FUNCTION DOES with it. Two adjacent emissions coming back zero is the
 * smallest corruption seen and is implausible as arithmetic — eight output channels at
 * one pixel, all exactly zero — so it is the signal, and the task is redone. A tile
 * whose data really does hold such a pair costs a wasted submit and stays correct.
 * Returns the number of atoms in runs of two or more consecutive zero emissions. */
static unsigned r76_i32_wide_suspect(const unsigned char *surf, unsigned A,
                                     unsigned oc_prog)
{
    unsigned sg, nsg = (oc_prog + 31u) / 32u, total = 0;

    for (sg = 0; sg < nsg; sg++) {
        unsigned e, run = 0;
        for (e = 0; e < 4u * A; e++) {
            unsigned s = e / 2u, L = e % 2u;
            unsigned atom = 4u * A * sg + A * (2u * (s / A) + L) + s % A;
            const unsigned char *ap = surf + (size_t)atom * C2;
            unsigned i;
            for (i = 0; i < C2; i++) if (ap[i]) break;
            if (i == C2) { run++; continue; }
            if (run >= 2u) total += run;
            run = 0;
        }
        if (run >= 2u) total += run;
    }
    return total;
}

/* The largest K slice this shape can run, or 0 if none can. The plane is chosen for
 * the LARGEST slice and then held fixed across every slice, because the output surface
 * the partials accumulate into has to be the same one. */
static unsigned r76_i32_plan(unsigned M, unsigned K, unsigned N, unsigned mult,
                             unsigned *iw_out, unsigned *ih_out, unsigned *nt_out)
{
    unsigned ks;
    unsigned cap = R76_I32_KS_MAX;
    const char *e = getenv("ROCKET_RK3576_MM_KS");

    if (e && *e) {
        long v = strtol(e, NULL, 0);
        if (v >= 32) cap = (unsigned)(v / 32 * 32);
    }
    for (ks = K < cap ? K : cap; ks >= 32u; ks = (ks / 2u / 32u) * 32u) {
        unsigned iw, ih, nt;
        r76_mm_plane(M, ks, &iw, &ih);
        nt = r76_mm_fit_nt_mult(iw, ks, N, mult, NULL);
        if (nt) {
            *iw_out = iw; *ih_out = ih; *nt_out = nt;
            return ks;
        }
        if (ks <= 32u) break;
    }
    return 0;
}

int rocket_matmul_int8_rk3576_i32(int fd, int M, int K, int N,
                                  const int8_t *A, const int8_t *B,
                                  const int32_t *bias, int32_t *C)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    struct r76_mm_bos b = {0};
    uint64_t *ops = NULL;
    int8_t *stage = NULL;
    rocket_rk3576_row_task *plan = NULL, *wplan = NULL;
    size_t *task_off = NULL;
    unsigned iw = 0, ih = 0, nt = 0, ks, k0, n0, surf_elems, max_tasks;
    unsigned mult = r76_i32_oc_mult();
    int diag = r76_i32_diag();
    unsigned diag_dropped = 0, heals = 0;
    unsigned char blank = 0;
    size_t o_surf_max, guard_off, in_slot = 0;
    unsigned nslices = 1, submits = 0;
    int rc = ROCKET_E_SHAPE;

    if (fd < 0 || !A || !B || !C || M <= 0 || K <= 0 || N <= 0) return ROCKET_E_SHAPE;
    if (strcmp(hw->name, "rk3576") != 0) {
        ROCKET_LOGE("rocket_matmul_int8_rk3576_i32: this is the RK3576 encoding and the "
                    "active profile is %s\n", hw->name);
        return ROCKET_E_UNSUPPORTED;
    }
    if (K % 32 || N % 32) {
        ROCKET_LOGE("rk3576 matmul i32: K=%d N=%d — both must be multiples of 32 (the "
                    "int8 weight cube groups each channel axis by 32)\n", K, N);
        return ROCKET_E_SHAPE;
    }

    ks = r76_i32_plan((unsigned)M, (unsigned)K, (unsigned)N, mult, &iw, &ih, &nt);
    ROCKET_LOGI("rk3576 matmul i32: M=%d K=%d N=%d -> plane %ux%u, K slice %u "
                "(%u slices, last %u), N tile %u (programmed oc %u, %ux writer)\n",
                M, K, N, iw, ih, ks,
                ks ? ((unsigned)K + ks - 1u) / ks : 0u,
                ks ? ((unsigned)K % ks ? (unsigned)K % ks : ks) : 0u,
                nt, mult * nt, mult);
    if (!ks) {
        ROCKET_LOGE("rk3576 matmul i32: M=%d K=%d N=%d does not fit one task even at a "
                    "32-channel output tile and a 32-deep K slice\n", M, K, N);
        return ROCKET_E_SHAPE;
    }

    /* The sentinel is the wide path's; the narrow one is single-surface per task and has
     * never dropped an atom, so it keeps the plain zeroed-BO signal. */
    if (mult != 4u && rocket_rk3576_sentinel_on()) blank = (unsigned char)R76_SENTINEL_BYTE;

    surf_elems = rocket_rk3576_out_surf_elems(iw, ih, 0);
    /* One task per output ROW is the worst case the wide writer's height bound can
     * force, so the plan array is sized for it rather than for the row planner alone. */
    max_tasks  = ih + 1u;
    ops   = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    plan  = calloc(max_tasks, sizeof *plan);
    wplan = calloc(max_tasks, sizeof *wplan);
    task_off = calloc(max_tasks, sizeof *task_off);
    stage = calloc((size_t)(ks + C2 - 1) / C2 * ih * iw * C2, 1);
    if (!ops || !plan || !wplan || !task_off || !stage) { rc = ROCKET_E_NOMEM; goto done; }

    memset(C, 0, (size_t)M * N * sizeof *C);

    /* TWO buffer rules here, and both were paid for on the part.
     *
     * The FEATURE cube for every K slice is packed into ONE buffer before any job runs,
     * and each slice's task is pointed at its own offset. Writing one slice at a time
     * between submits — the obvious way — leaves the next job reading what the last one
     * saw, so it recomputes the previous slice byte for byte with no error anywhere.
     *
     * Everything the DPU WRITES, and everything a task reads that changes with the
     * task, is a FRESH buffer per submit. Rewriting one in place between submits does
     * not take effect either, and the failure is the same silent one: the job runs, the
     * timing is normal, nothing faults, and the surface still holds the previous
     * result. The int8 path above allocates per tile for the same reason. Pacing the
     * submits about 200 ms apart also hides it, which is what makes it read as a timing
     * problem rather than a buffer-lifetime one. */
    {
        size_t in_max = (size_t)((ks + C2 - 1) / C2) * ih * iw * C2;
        in_slot = (in_max + 63u) & ~(size_t)63u;
        nslices = ((unsigned)K + ks - 1u) / ks;
        if (rocket_bo_alloc(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0 ||
            rocket_bo_alloc(fd, in_slot * nslices, &b.in) < 0) {
            rc = ROCKET_E_NOMEM; goto done;
        }
    }

    rocket_bo_prep(fd, &b.in, 1, 0);
    memset(b.in.ptr, 0, in_slot * nslices);
    for (k0 = 0; k0 < (unsigned)K; k0 += ks) {
        unsigned kslice = (unsigned)K - k0 < ks ? (unsigned)K - k0 : ks;
        int8_t *slot = (int8_t *)b.in.ptr + (size_t)(k0 / ks) * in_slot;
        int m;
        for (m = 0; m < M; m++) {
            unsigned k;
            for (k = 0; k < kslice; k++)
                slot[feature_data((int)kslice, (int)ih, (int)iw, C2,
                                  (int)k + 1, m / (int)iw + 1, m % (int)iw + 1)] =
                    A[(size_t)m * K + k0 + k];
        }
    }
    rocket_bo_fini(fd, &b.in);

    for (k0 = 0; k0 < (unsigned)K; k0 += ks) {
        unsigned kslice = (unsigned)K - k0 < ks ? (unsigned)K - k0 : ks;

        for (n0 = 0; n0 < (unsigned)N; n0 += nt) {
            unsigned tile_n = (unsigned)N - n0 < nt ? (unsigned)N - n0 : nt;
            unsigned oc_prog = rocket_rk3576_pad_oc(mult * tile_n);
            size_t w_bytes = (size_t)((oc_prog + 31) / 32) * ((kslice + 31) / 32) * 32 * 32;
            size_t coeff_bytes = rocket_rk3576_coeff_bytes(oc_prog);
            /* The writer's byte budget is one 16-byte atom per (16-channel block,
             * pixel) on the narrow writer and TWO on the wide one, whatever the output
             * element width is. Either way it comes out at exactly the int32 cube for
             * `tile_n` real channels, because the programmed count absorbs the rest. */
            size_t atoms_per_px = (mult == 4u ? 1u : 2u) * ((oc_prog + C2 - 1) / C2);
            size_t surf_bytes;
            conv_params_t p = {0};
            uint32_t in_h[4], out_h[1];
            unsigned ntask = 1, t, n, k, attempt;
            int wrote = 0;

            /* The row plan comes FIRST here, because on the wide writer it SIZES the
             * output BO: each row task carries its own surface rather than a window
             * into a shared one. */
            {
                conv_params_t q = {0};
                q.ic = (uint16_t)kslice; q.ih = (uint16_t)ih; q.iw = (uint16_t)iw;
                q.oc = (uint16_t)oc_prog; q.oh = (uint16_t)ih; q.ow = (uint16_t)iw;
                q.kh = 1; q.kw = 1; q.stride_y = 1; q.stride_x = 1;
                q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
                if (rocket_rk3576_plan_rows(&q, 0, plan, max_tasks, &ntask) < 0) {
                    ROCKET_LOGE("rk3576 matmul i32: no row plan for M=%d K slice %u "
                                "N tile %u (plane %ux%u, programmed oc %u)\n",
                                M, kslice, tile_n, iw, ih, oc_prog);
                    rc = ROCKET_E_SHAPE; goto done;
                }
            }
            if (mult == 4u) {
                surf_bytes  = atoms_per_px * surf_elems * C2;
                task_off[0] = 0;
            } else {
                /* THE WIDE WRITER'S SURFACE HEIGHT IS BOUNDED, and past the bound its
                 * map is not the one above — the delivered set starts drifting with the
                 * pixel and a whole tile comes back part right, silently. Measured on
                 * the part: it holds while oh_full * oc < 4096 and breaks at it, at both
                 * ends tested (oc 192 breaks between oh 21 and 22, oc 64 between 48 and
                 * 64). The mechanism is not decoded; only the boundary is.
                 * [HW sweep, H96 MAX M9]
                 *
                 * A row task here is a standalone 1x1 convolution with its own surface,
                 * so honouring the bound costs nothing but submits: split the plan's
                 * tasks further until each one's height fits. */
                unsigned cap = R76_I32_WIDE_OC_ROWS / oc_prog;
                unsigned src;
                if (!cap) cap = 1u;
                surf_bytes = 0;
                for (src = 0, t = 0; src < ntask; src++) {
                    unsigned done_rows = 0;
                    while (done_rows < plan[src].oh) {
                        unsigned take = plan[src].oh - done_rows;
                        if (take > cap) take = cap;
                        if (t == max_tasks) {
                            ROCKET_LOGE("rk3576 matmul i32: the wide writer's row bound "
                                        "needs more than %u tasks at programmed oc %u\n",
                                        max_tasks, oc_prog);
                            rc = ROCKET_E_SHAPE; goto done;
                        }
                        wplan[t].oy0 = (uint16_t)(plan[src].oy0 + done_rows);
                        wplan[t].oh  = (uint16_t)take;
                        wplan[t].iy0 = wplan[t].oy0;
                        wplan[t].ih  = (uint16_t)take;
                        task_off[t]  = surf_bytes;
                        surf_bytes  += atoms_per_px *
                                       rocket_rk3576_out_surf_elems(iw, take, 0) * C2;
                        /* A band between one task's surface and the next, so an atom
                         * written past a task's extent lands somewhere observable
                         * instead of in its neighbour's first run. */
                        if (diag >= 2) surf_bytes += R76_DIAG_TASK_GUARD;
                        done_rows += take;
                        t++;
                    }
                }
                ntask = t;
                for (t = 0; t < ntask; t++) plan[t] = wplan[t];
            }

            if (b.w.ptr)     rocket_bo_free(fd, &b.w);
            if (b.coeff.ptr) rocket_bo_free(fd, &b.coeff);
            if (b.out.ptr)   rocket_bo_free(fd, &b.out);
            memset(&b.w, 0, sizeof b.w);
            memset(&b.coeff, 0, sizeof b.coeff);
            memset(&b.out, 0, sizeof b.out);

            /* NOTHING writes the output buffer from the CPU — not the surface, and not
             * the band past it. A fresh BO arrives zeroed, which is already the
             * sentinel: the band staying zero is the overrun check, and the DPU writes
             * the whole surface when it writes at all.
             *
             * Zeroing it by hand is the trap. Those dirty cache lines race the DPU's
             * DMA and the writeback lands on top of the result, so the surface comes
             * back all zeros — intermittently, on the first submit as readily as a
             * later one, with no fault anywhere and normal timing. It reads exactly
             * like an arithmetic failure. The band is 64-byte aligned so nothing in it
             * can share a line with a surface byte either. */
            guard_off = (surf_bytes + 63u) & ~(size_t)63u;
            o_surf_max = surf_bytes;
            if (rocket_bo_alloc(fd, w_bytes, &b.w) < 0 ||
                rocket_bo_alloc(fd, coeff_bytes, &b.coeff) < 0 ||
                rocket_bo_alloc(fd, guard_off + (size_t)surf_elems * C2, &b.out) < 0) {
                rc = ROCKET_E_NOMEM; goto done;
            }

            /* The scatter: real channel n lives at programmed channel 32*(n/8)+n%8,
             * which is a slot the 32-bit writer delivers. Everything else stays zero. */
            rocket_bo_prep(fd, &b.w, 1, 0);
            memset(b.w.ptr, 0, w_bytes);
            for (n = 0; n < tile_n; n++)
                for (k = 0; k < kslice; k++)
                    ((int8_t *)b.w.ptr)[weight_conv_int8((int)oc_prog, (int)kslice, 1, 1,
                                                         (int)r76_i32_prog_oc(n, mult) + 1,
                                                         (int)k + 1, 1, 1)] =
                        B[(size_t)(n0 + n) * K + k0 + k];
            rocket_bo_fini(fd, &b.w);

            /* No bias on the device here: a K split would add it once per slice. The C
             * term still has to be there — it gates the BS stage, and a zeroed
             * coefficient buffer returns a full but empty surface — which is what
             * pack_coeff writes for a NULL bias array. */
            rocket_bo_prep(fd, &b.coeff, 1, 0);
            rocket_rk3576_pack_coeff(b.coeff.ptr, coeff_bytes, NULL, oc_prog);
            rocket_bo_fini(fd, &b.coeff);

            p.ic = (uint16_t)kslice; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
            p.oc = (uint16_t)oc_prog; p.oh = (uint16_t)ih; p.ow = (uint16_t)iw;
            p.kh = 1; p.kw = 1;
            p.stride_y = 1; p.stride_x = 1;
            p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)ih;
            p.int8_out = 1;
            /* The i32out emitter pins OUT_CVT to exact unity itself; these only keep the
             * shared derivation away from a divide by zero. */
            p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;
            p.input_zero_point = 0x80;
            p.output_zero_point = 0x80;
            p.weight_zero_point = 0x80;
            p.tasks       = ops;
            p.input_dma   = b.in.dma_address + (uint32_t)((size_t)(k0 / ks) * in_slot);
            p.weights_dma = b.w.dma_address;
            p.bias_dma    = b.coeff.dma_address;
            p.output_dma  = b.out.dma_address;

            in_h[0] = b.in.handle; in_h[1] = b.w.handle;
            in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
            out_h[0] = b.out.handle;

          for (attempt = 0; attempt < 2; attempt++) {
            /* Stamp the surface, but not the band past it — the band staying zero is the
             * overrun check, and it has to keep meaning that. */
            if (blank) {
                rocket_bo_prep(fd, &b.out, 1, 0);
                memset(b.out.ptr, blank, surf_bytes);
                rocket_bo_fini(fd, &b.out);
                /* The stamp is only evidence if it reaches DDR. Reading an atom back as
                 * ZERO rather than as the sentinel says the writer emitted it with the
                 * wrong data — but only once the sentinel is known to have landed, since
                 * a fresh BO arrives zeroed and a stamp that never left the cache would
                 * read the same way. */
                if (diag >= 3) {
                    size_t si, bad = 0;
                    const unsigned char *sp;
                    rocket_bo_prep(fd, &b.out, 0, 0);
                    sp = (const unsigned char *)b.out.ptr;
                    for (si = 0; si < surf_bytes; si++) if (sp[si] != blank) bad++;
                    ROCKET_LOGI("rk3576 i32 diag: stamp check k0=%u n0=%u: %zu of %zu "
                                "bytes are not the sentinel before any submit\n",
                                k0, n0, bad, surf_bytes);
                    rocket_bo_fini(fd, &b.out);
                }
            }
            for (t = 0; t < ntask; t++) {
              /* PER-TASK retry, and it has to be per task on the wide path. Each row
               * task there owns its own sub-surface, so one poisoned submit leaves that
               * region empty while its siblings are full — and a check over the whole
               * tile then reads "something was written" and passes the hole through.
               * At the stock autosuspend delay this never fires; at a short one it does,
               * which is exactly the configuration that made it visible. */
              unsigned tattempt;
              for (tattempt = 0; tattempt < 4; tattempt++) {
                conv_params_t q = p;
                q.ih = plan[t].ih; q.oh = plan[t].oh;
                q.pad_top = plan[t].pad_top;
                q.input_dma  = p.input_dma  + plan[t].feature_off;
                if (mult == 4u) {
                    /* The narrow writer's surface is one shared cube and the row plan's
                     * output offset counts int8 atoms; a 32-bit element moves the same
                     * 16-byte atom, so the offset carries over unchanged. */
                    q.output_dma = p.output_dma + plan[t].output_off;
                    q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
                } else {
                    /* The WIDE writer's surface cannot be entered part-way. Its map is
                     * one stream, s = 2*pixel + block, cut into runs of ow*oh_full — so
                     * a task starting at row oy0 is at stream position 2*oy0*ow, which
                     * crosses a run boundary at the surface's midpoint and stops being
                     * expressible as a base offset. A shared cube with a shifted base
                     * therefore computes correctly and lands in the wrong words, which
                     * is exactly what a row-split tile did before this.
                     *
                     * The kernel here is 1x1 at stride 1, so a row task is not a window
                     * into a larger convolution at all — it is a smaller convolution.
                     * Each one gets its OWN full surface, laid end to end in the same
                     * BO, and the de-scatter walks them. */
                    q.input_dma  = p.input_dma +
                                   (uint32_t)plan[t].iy0 * iw * C2;
                    q.output_dma = p.output_dma + (uint32_t)task_off[t];
                    /* ih_full stays the FULL plane — it is the feature cube's
                     * channel-group stride, and the cube in the BO is the whole one.
                     * Only oh_full moves, which is the output surface's. */
                    q.ih_full = (uint16_t)ih;
                    q.oh_full = plan[t].oh;
                    q.pad_top = 0;
                }
                if ((mult == 4u ? gen_conv2d_int8_rk3576_i32out(&q)
                                : gen_conv2d_int8_rk3576_i32out_wide(&q)) != 0) {
                    rc = ROCKET_E_SHAPE; goto done;
                }
                ROCKET_LOGD("rk3576 matmul i32: submit k0=%u n0=%u ic=%u oc=%u "
                            "in@%08x[%d %d %d %d] w@%08x[%d %d %d %d] out@%08x\n",
                            k0, n0, kslice, oc_prog, q.input_dma,
                            ((const int8_t *)b.in.ptr)[(size_t)(k0 / ks) * in_slot + 0],
                            ((const int8_t *)b.in.ptr)[(size_t)(k0 / ks) * in_slot + 1],
                            ((const int8_t *)b.in.ptr)[(size_t)(k0 / ks) * in_slot + 2],
                            ((const int8_t *)b.in.ptr)[(size_t)(k0 / ks) * in_slot + 3],
                            q.weights_dma,
                            ((const int8_t *)b.w.ptr)[0], ((const int8_t *)b.w.ptr)[1],
                            ((const int8_t *)b.w.ptr)[2], ((const int8_t *)b.w.ptr)[3],
                            q.output_dma);
                /* AN i32out JOB POISONS THE NEXT SUBMIT, and what clears it is a
                 * POWER CYCLE rather than elapsed time.
                 *
                 * The job that does not write completes normally — no fault, no IOMMU
                 * error, no timeout, the usual ~1.4 ms — and leaves the output buffer
                 * untouched, so it reads as the previous result rather than as a
                 * failure. Fresh buffers per submit, a different feature address and a
                 * different output address were each tried and none of them helps.
                 *
                 * What does: the driver's runtime-PM autosuspend cycling the NPU power
                 * domain. Write `on` to the device's `power/control` and no gap clears
                 * it — 600 ms still writes nothing — and the working gap tracks
                 * `power/autosuspend_delay_ms` one for one across 5, 10, 20 and 50 ms.
                 * The "50-100 ms of idle" this was first read as is that file's stock
                 * value plus the suspend round trip, and nothing about the silicon.
                 *
                 * What poisons is DPU 0x4010, the DATA_FORMAT word: driving its output
                 * WIDTH field to int32 does it, and driving its PROC_PRECISION field to
                 * int32 does it separately. Pinning OUT_CVT to unity — the other thing
                 * the i32out program changes — does not, and neither does any plain
                 * int8 job. The next job programs 0x4010 back to all-int8 and still
                 * comes back empty, so this is latched datapath state and not a stale
                 * register. [HW sweep, H96 MAX M9]
                 *
                 * So the cost is a submit per power cycle, and it is SETTABLE: lowering
                 * the driver's autosuspend delay shortens it proportionally, and a
                 * kernel-side reset of the DPU at job start would remove it. */
                /* The FIRST submit of a call is not idled — a caller that enters with the
                 * part already poisoned is covered by the per-task check below, which
                 * costs an idle only when it fires, where a leading idle would cost one
                 * every call. Sweeping the idle ahead of that first submit changes
                 * nothing about the wide writer's zero-data defect either: 3 failures in
                 * 30 at no idle against 6 in 30 at 800 ms. [HW sweep, H96 MAX M9] */
                if (submits++) rocket_rk3576_power_idle();
                rocket_bo_prep(fd, &b.rc, 1, 0);
                memcpy(b.rc.ptr, ops, q.task_count * sizeof(uint64_t));
                rocket_bo_fini(fd, &b.rc);
                if (rocket_submit_matmul(fd, &b.rc, q.task_count, in_h, 4, out_h, 1, 4000) != 0) {
                    rc = ROCKET_E_DEVICE; goto done;
                }
                if (rocket_bo_prep(fd, &b.out, 0, 2000000000ull) < 0) {
                    rc = ROCKET_E_DEVICE; goto done;
                }
                {
                    /* Did THIS task's region get written? A fresh BO arrives zeroed and
                     * the DPU fills the whole region when it writes at all, so a region
                     * still entirely zero is a poisoned submit. A legitimately zero
                     * region costs one wasted redo and stays correct.
                     *
                     * It has to be per TASK. One poisoned submit among several leaves
                     * its own rows empty while its siblings are full, so a check over
                     * the whole tile reads "something was written" and passes the hole
                     * straight through to the caller — a tile that is exactly one row
                     * task short, with no fault and normal timing.
                     *
                     * The region differs by writer. The wide one gives each task its own
                     * surface, so it is that surface. The narrow one shares a cube, and
                     * the task's rows of CHANNEL BLOCK 0 are the contiguous part of what
                     * it wrote — enough to tell a dead submit from a live one. */
                    size_t off = mult == 4u ? (size_t)plan[t].output_off : task_off[t];
                    size_t tb  = mult == 4u
                               ? (size_t)plan[t].oh * iw * C2
                               : atoms_per_px *
                                 rocket_rk3576_out_surf_elems(iw, plan[t].oh, 0) * C2;
                    const unsigned char *sp = (const unsigned char *)b.out.ptr + off;
                    size_t si;
                    int twrote = 0;
                    unsigned holes = 0, unwritten = 0, suspect = 0;
                    if (blank && mult != 4u) {
                        /* AGAINST A SENTINEL THE QUESTION IS PER ATOM, and that is what
                         * covers the wide writer's dropped atoms as well as a dead
                         * submit. The drop is a contiguous block in the writer's
                         * EMISSION order that simply never reaches DDR — a second read
                         * after an idle returns byte-identical contents — so redoing the
                         * task is the repair, and it lands every atom on the retry.
                         * [HW sweep, H96 MAX M9] */
                        /* Two failures to catch here, and they want different recovery.
                         * An atom still holding the sentinel was never emitted, which is
                         * the poisoning, and only the power domain cycling clears that —
                         * so that redo has to be idled. An emission stream that goes zero
                         * for a stretch was emitted with the wrong data, which is the
                         * wide writer's own defect and is indifferent to idle, so that
                         * redo is immediate and costs one submit. */
                        unsigned A = iw * plan[t].oh;
                        unwritten = r76_i32_diag_task(sp, A, oc_prog,
                                                      (unsigned)atoms_per_px, k0, n0, t,
                                                      blank, "unwritten", 1);
                        suspect = r76_i32_wide_suspect(sp, A, oc_prog);
                        holes   = unwritten + suspect;
                        twrote  = holes == 0;
                    } else {
                        for (si = 0; si < tb; si++)
                            if (sp[si] != blank) { twrote = 1; break; }
                    }
                    rocket_bo_fini(fd, &b.out);
                    if (twrote) break;
                    ROCKET_LOGD("rk3576 matmul i32: k0=%u n0=%u row task %u — %u atoms "
                                "never emitted, %u emitted zero; redoing it\n",
                                k0, n0, t, unwritten, suspect);
                    if (suspect) heals++;
                    /* Only the poisoning needs the power cycle. Idling on a zero-data
                     * redo would put 120 ms on a failure that does not care about it. */
                    if (unwritten) rocket_rk3576_power_idle();
                }
              }
            }
            /* A poisoned job leaves the surface exactly as it found it, and a fresh BO
             * arrives zeroed — so "still all zero" is the signal, and redoing the job
             * after an idle is the recovery. This covers the poisoning that crosses
             * CALLS, which the inter-submit idle above cannot see: the last i32out job
             * of a previous matmul poisons the first submit of this one. A surface that
             * is legitimately all zero costs one wasted retry and stays correct. */
            rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
            wrote = 0;
            {
                const unsigned char *sp = (const unsigned char *)b.out.ptr;
                size_t si;
                for (si = 0; si < surf_bytes; si++)
                    if (sp[si] != blank) { wrote = 1; break; }
            }
            rocket_bo_fini(fd, &b.out);
            if (wrote) break;
            if (attempt == 0) {
                ROCKET_LOGD("rk3576 matmul i32: slice k0=%u tile n0=%u wrote nothing, "
                            "idling and redoing it\n", k0, n0);
                rocket_rk3576_power_idle();
            }
          }

            rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
            {
                const unsigned char *guard = (const unsigned char *)b.out.ptr + guard_off;
                size_t g;
                for (g = 0; g < (size_t)surf_elems * C2; g++)
                    if (guard[g] != 0) {
                        ROCKET_LOGE("rk3576 matmul i32: the 32-bit writer reached past "
                                    "the largest surface any tile of this run claims "
                                    "(%zu bytes) at programmed oc=%u — the extent model "
                                    "is wrong for this shape\n", o_surf_max, oc_prog);
                        rocket_bo_fini(fd, &b.out);
                        rc = ROCKET_E_DEVICE; goto done;
                    }
            }
            if (diag && mult != 4u) {
                const unsigned char *sp = (const unsigned char *)b.out.ptr;
                unsigned dropped = 0, zeros = 0;
                for (t = 0; t < ntask; t++) {
                    unsigned A = iw * plan[t].oh;
                    dropped += r76_i32_diag_task(sp + task_off[t], A, oc_prog,
                                                 (unsigned)atoms_per_px, k0, n0, t,
                                                 blank, "UNWRITTEN (still the sentinel)",
                                                 0);
                    if (blank)
                        zeros += r76_i32_diag_task(sp + task_off[t], A, oc_prog,
                                                   (unsigned)atoms_per_px, k0, n0, t,
                                                   0, "ZERO (emitted, wrong data)", 0);
                    if (diag >= 2) {
                        size_t gb = task_off[t] + (size_t)atoms_per_px * A * C2, g;
                        for (g = 0; g < R76_DIAG_TASK_GUARD; g++)
                            if (sp[gb + g] != blank) {
                                ROCKET_LOGI("rk3576 i32 diag: k0=%u n0=%u task=%u wrote "
                                            "%u bytes PAST its surface (first at +%zu) — "
                                            "this is address generation, not a dropped "
                                            "write\n", k0, n0, t,
                                            (unsigned)(R76_DIAG_TASK_GUARD - g), g);
                                break;
                            }
                    }
                }
                diag_dropped += dropped;
                if (zeros)
                    ROCKET_LOGI("rk3576 i32 diag: k0=%u n0=%u %u atoms came back ZERO on a "
                                "stamped surface — the writer emitted them and the data is "
                                "wrong, rather than skipping them\n", k0, n0, zeros);
                diag_dropped += zeros;
                if (dropped)
                    ROCKET_LOGI("rk3576 i32 diag: k0=%u n0=%u dropped %u atoms over %u "
                                "row task%s (plane %ux%u, programmed oc %u)\n",
                                k0, n0, dropped, ntask, ntask == 1 ? "" : "s",
                                iw, ih, oc_prog);
            }
            /* ROCKET_RK3576_I32_RECHECK re-reads the surface after a second fence and
             * says whether anything changed. It separates the two ways a hole in the
             * output can arise — a word the DPU never wrote, and a word this side read
             * stale — which look identical from the accumulate below. */
            if (getenv("ROCKET_RK3576_I32_RECHECK")) {
                unsigned char *copy = malloc(surf_bytes);
                if (copy) {
                    size_t di, ndiff = 0, nzero = 0;
                    const unsigned char *sp;
                    memcpy(copy, b.out.ptr, surf_bytes);
                    rocket_bo_fini(fd, &b.out);
                    rocket_rk3576_power_idle();
                    rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
                    sp = (const unsigned char *)b.out.ptr;
                    for (di = 0; di < surf_bytes; di++) {
                        if (sp[di] != copy[di]) ndiff++;
                        if (!sp[di]) nzero++;
                    }
                    ROCKET_LOGI("rk3576 matmul i32: recheck k0=%u n0=%u: %zu of %zu "
                                "bytes changed on a second read, %zu still zero\n",
                                k0, n0, ndiff, surf_bytes, nzero);
                    free(copy);
                }
            }
            /* Accumulate this slice's partials into the caller's row-major C. */
            {
                const int32_t *o32 = (const int32_t *)b.out.ptr;
                int m;
                ROCKET_LOGD("rk3576 matmul i32: slice k0=%u(%u) tile n0=%u: "
                            "first word %d\n", k0, kslice, n0, o32[0]);
                t = 0;
                for (m = 0; m < M; m++) {
                    unsigned p_idx = (unsigned)m, y = p_idx / iw;
                    size_t base = 0;
                    if (mult != 4u) {
                        /* Which row task's surface this row landed in, and where it
                         * sits inside it. */
                        while (t + 1u < ntask && y >= (unsigned)plan[t + 1].oy0) t++;
                        while (t && y < (unsigned)plan[t].oy0) t--;
                        base = task_off[t] / 4u;
                        p_idx -= (unsigned)plan[t].oy0 * iw;
                    }
                    for (n = 0; n < tile_n; n++) {
                        int w;
                        if (mult == 4u) {
                            /* The narrow writer's scatter collapses to the plain
                             * RK3588 int32 cube. */
                            w = (int)((size_t)(n / 4u) * surf_elems * 4u +
                                      4u * p_idx + (n % 4u));
                        } else {
                            w = rocket_rk3576_i32_wide_word(iw, plan[t].oh,
                                                            r76_i32_prog_oc(n, mult),
                                                            p_idx);
                            if (w < 0) { rc = ROCKET_E_SHAPE; goto done; }
                        }
                        C[(size_t)m * N + n0 + n] += o32[base + (size_t)w];
                    }
                }
            }
            rocket_bo_fini(fd, &b.out);
        }
    }

    if (bias) {
        int m;
        unsigned n;
        for (m = 0; m < M; m++)
            for (n = 0; n < (unsigned)N; n++) C[(size_t)m * N + n] += bias[n];
    }
    /* Leave the part unpoisoned. The hazard an i32out job creates outlives the job and
     * outlives the PROCESS, because it lives in the NPU rather than in this fd: the next
     * submit of ANY kind writes nothing, and the plain int8 matmul in another program is
     * as exposed as this one — its output BO comes back untouched and the caller reads a
     * correctly sized, entirely stale surface. The entry that creates the hazard is the
     * one that should absorb it, so this idles once on the way out rather than leaving
     * the next caller to discover it. */
    rocket_rk3576_power_idle();
    /* Submits is the cost that matters on this path: the poisoning workaround pays an
     * idle per submit, so what a change buys is measured in submits and not in MACs. */
    ROCKET_LOGI("rk3576 matmul i32: M=%d K=%d N=%d done in %u submits (%ux writer)\n",
                M, K, N, submits, mult);
    if (diag && mult != 4u)
        ROCKET_LOGI("rk3576 i32 diag: M=%d K=%d N=%d dropped %u atoms in total, %u task%s "
                    "redone for a partial write\n",
                    M, K, N, diag_dropped, heals, heals == 1 ? "" : "s");
    else if (heals)
        ROCKET_LOGI("rk3576 matmul i32: %u row task%s redone — the wide writer left a "
                    "partial surface and the retry completed it\n",
                    heals, heals == 1 ? " was" : "s were");
    rc = ROCKET_OK;

done:
    free(ops); free(plan); free(wplan); free(task_off); free(stage);
    r76_mm_free(fd, &b);
    return rc;
}
