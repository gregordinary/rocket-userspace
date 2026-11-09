// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_pool.c — on-NPU fp16 MaxPool / AveragePool runtime (the PPU pooling engine).
 *
 * A single self-contained PPU + PPU_RDMA job: scatter the input feature into the
 * NC1HWC2 cube (C2=8, the same cube the conv path uses), generate the PPU regcmd
 * (gen_pool_fp16), submit, then de-scatter the output cube. No weights, no CNA/CORE/DPU.
 * The register program + the average RECIP_KERNEL=fp16(65536/k) format are HW-validated
 * (MAX bit-exact; AVG within fp16-reciprocal tolerance). See rocket_pool.h.
 */
#include <stdio.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_pool.h"
#include "npu_matmul.h"   /* feature_index (NC1HWC2 cube index) */
#include "rocket_op.h"      /* the shared one-shot scaffolding */
#include "rocket_log.h"     // centralized log channel

/* slack past the scattered cube, so a DMA burst over-read can't run off the BO */
#define POOL_BO_SLACK 32768
/* One band's regcmd slot in words: gen_pool_fp16 writes 30, and an even slot keeps every
 * task's address 16-byte aligned, which the submit check requires. */
#define POOL_TASK_SLOT 64

/* ============================================================================
 * SECTION — Whether the running kernel can retire a pool
 * ==========================================================================*/

/* A pooling program enables PPU and PPU_RDMA only (0x60), so it raises no DPU completion.
 * A rocket driver without the PPU-completion flag arms only the DPU pair, nothing it waits
 * for ever fires, and drm_sched retires the job at JOB_TIMEOUT_MS through a core reset:
 * ~507 ms a submit on the RK1, output correct (rockchip-npu-notes perf/pool-completion.md). The
 * stock driver is in that class at interface 1.0, and so is the RK3576 upstream series as
 * Armbian's edge kernel carries it. The host computes a pool in microseconds, so refusing
 * is the whole fix there.
 *
 * Keyed on rocket_ppu_done_supported() rather than on the interface version directly: that
 * probe is part of the submit seam, so the vendor provider answers for its own driver (1,
 * since it waits on whichever block the task's interrupt mask names). The plan functions
 * stay pure and never call this; the entries and the frontends' claims do. */
int rocket_pool_npu_supported(void)
{
    static _Atomic int warned = 0;
    int ok = rocket_ppu_done_supported();

    if (!ok && !atomic_exchange(&warned, 1))
        ROCKET_LOGW("rocket: this driver has no PPU completion, so it would retire each "
                    "pooling job at its 500 ms job timeout with a core reset. The pool "
                    "entries refuse and the global pools compute on the host.\n");
    return ok;
}

/* ============================================================================
 * SECTION — The PPU's per-task output-width allowance and the column bands
 * ==========================================================================*/

/*
 * HOW MANY OUTPUT COLUMNS ONE RK3588 POOLING TASK MAY PRODUCE.
 *
 * Past this a task writes a complete, correctly sized surface that is wrong across the
 * whole output, and most windows also raise no completion, so the job retires at the
 * driver's timeout (rockchip-npu-notes encodings/ppu-pooling.md, "Output-width allowance").
 * The allowance depends on the kernel height and the vertical stride, not on the plane's
 * height or channel count. MEASURED, last exact output width [HW sweep, RK1, 928 MHz,
 * 2026-11-02, fp16, no pad]:
 *
 *     2x2 s2 128      2x2 s1, 2x1 s1, 2x2 sy1 sx2  64      3x3 s1, 3x1 s1  32
 *     3x3 s2, s3 64   4x4 s1 32    4x4 s2, s4 64    5x5 s1, 8x8 s1 16
 *     5x5..8x8 at a stride equal to the window 64   1x2 s1, 1x3 s1  256 and more
 *     1x1 MAX to an input width of 8191
 *
 * and every window past 8 on either axis fails at or near a width of one (12x12, 16x16,
 * 16x1 and 1x16 at a stride equal to the window). The cells the bound below admits that
 * the table does not carry were each exact at the bound, one task, MAX and four also AVG:
 * 6x6 and 7x7 at stride 1 (16); 5x5 to 8x8 at stride 2, 6x6 and 8x8 at stride 3 (32);
 * 4x4 to 6x6 at stride 3, 7x7 and 8x8 at stride 4 (64); 8x2 (16), 4x2 (32), 2x8 (64),
 * 1x5, 1x8 and 1x8 at stride 8 (64); 3x3 at stride 1x2 (32); padded 3x3 stride 2 (56),
 * 5x5 stride 1 (16) and 3x3 stride 1 (32) [HW sweep, RK1, 928 MHz, 2026-11-06].
 *
 * What fits every measured cell is a budget keyed on the output rows in flight over one
 * input row, o = ceil(kh / stride_y), the same axis the RK3576's allowance is keyed on
 * (rocket_pool_int8_rk3576_max_ow): 64 at o 1-2, 32 at o 3-4, 16 at o 5-8, with the
 * measured 2x2-stride-2 and one-row cells above it [hypothesis for the cells between
 * the measured ones]. Being under costs a band; being over costs a silent wrong surface,
 * so a window past 8 on either axis gets 0 and is refused.
 */
unsigned rocket_pool_fp16_max_ow(const rocket_pool_desc *d)
{
    unsigned kh, kw, sy, sx, o;

    if (!d || d->kh < 1 || d->kw < 1 || d->stride_y < 1 || d->stride_x < 1) return 0u;
    kh = (unsigned)d->kh; kw = (unsigned)d->kw;
    sy = (unsigned)d->stride_y; sx = (unsigned)d->stride_x;
    if (kh > 8u || kw > 8u) return 0u;
    /* A STRIDE PAST THE WINDOW, on either axis, is its own failure and not a narrower
     * allowance: a 2x2 window at stride 3x3, 2x3 or 3x2 is wrong at an output width of 8
     * and retires at the job timeout, and exact only at a 1x1 output [HW sweep, RK1,
     * 928 MHz, 2026-11-06]. A 1x1 window at stride 2 was exact at width 8, one run, so it
     * is refused with the rest rather than admitted on that. */
    if (sy > kh || sx > kw) return 0u;
    if (kh == 1u) {
        if (kw == 1u && sx == 1u) return 8192u;
        if (kw <= 3u && sx == 1u) return 256u;
        return 64u;
    }
    if (kh == 2u && sy == 2u && sx == 2u) return 128u;
    o = (kh + sy - 1u) / sy;
    if (o <= 2u) return 64u;
    if (o <= 4u) return 32u;
    return 16u;
}

/*
 * One band's geometry. Band `s` of `n` produces output columns [x0, x0+ow_s). The input
 * columns its windows reach are [x0*stride_x - pad_left, (x0+ow_s-1)*stride_x + kw -
 * pad_left), clipped to the plane, and what the clip removed becomes that band's own
 * pad, so only a band at an edge carries the edge's pad. The bands are evenly sized with
 * the remainder spread over the leading ones, so every band is inside the allowance.
 * Returns n, or 0 if the window has no allowance.
 */
static unsigned pool_band(const rocket_pool_desc *d, unsigned s,
                          unsigned *x0, unsigned *ow_s, unsigned *in_col,
                          unsigned *iw_s, int *pad_l, int *pad_r)
{
    unsigned ow = (unsigned)rocket_pool_ow(d);
    unsigned allow = rocket_pool_fp16_max_ow(d);
    unsigned n, i, x = 0, w;

    if (!allow) return 0u;
    n = (ow + allow - 1u) / allow;
    if (n == 0u) n = 1u;
    if (s >= n) return n;
    for (i = 0; i < s; i++) x += ow / n + (i < ow % n ? 1u : 0u);
    w = ow / n + (s < ow % n ? 1u : 0u);
    {
        long lo = (long)x * d->stride_x - d->pad_left;
        long hi = ((long)x + (long)w - 1) * d->stride_x + d->kw - d->pad_left;
        int pl = 0, pr = 0;
        if (lo < 0) { pl = (int)(-lo); lo = 0; }
        if (hi > d->iw) { pr = (int)(hi - d->iw); hi = d->iw; }
        *x0 = x; *ow_s = w;
        *in_col = (unsigned)lo;
        *iw_s = (unsigned)(hi - lo);
        *pad_l = pl; *pad_r = pr;
    }
    return n;
}

/* The band's own descriptor: the caller's, narrowed to the band's input columns. */
static rocket_pool_desc pool_band_desc(const rocket_pool_desc *d, unsigned iw_s,
                                       int pad_l, int pad_r)
{
    rocket_pool_desc b = *d;
    b.iw = (int)iw_s; b.pad_left = pad_l; b.pad_right = pad_r;
    return b;
}

unsigned rocket_pool_fp16_bands(const rocket_pool_desc *d)
{
    unsigned x0, ows, col, iws, n, s;
    int pl, pr;

    if (!d || rocket_pool_ow(d) < 1) return 0u;
    n = pool_band(d, 0, &x0, &ows, &col, &iws, &pl, &pr);
    if (!n || n > ROCKET_POOL_MAX_BANDS) return 0u;
    /* Every band's own descriptor must derive back to the width it was cut to; one that
     * does not is refused rather than run on a geometry nobody checked. */
    for (s = 0; s < n; s++) {
        rocket_pool_desc b;
        pool_band(d, s, &x0, &ows, &col, &iws, &pl, &pr);
        b = pool_band_desc(d, iws, pl, pr);
        if ((unsigned)rocket_pool_ow(&b) != ows || ows > rocket_pool_fp16_max_ow(&b))
            return 0u;
    }
    return n;
}

int rocket_pool_fp16_band(const rocket_pool_desc *d, unsigned s,
                          unsigned *x0, unsigned *in_col, rocket_pool_desc *band)
{
    unsigned n = rocket_pool_fp16_bands(d), bx, ows, col, iws;
    int pl, pr;

    if (!n || s >= n || !x0 || !in_col || !band) return ROCKET_E_SHAPE;
    pool_band(d, s, &bx, &ows, &col, &iws, &pl, &pr);
    *x0 = bx; *in_col = col;
    *band = pool_band_desc(d, iws, pl, pr);
    return 0;
}

/* ============================================================================
 * SECTION — Geometry validation and CPU references
 * ==========================================================================*/

int rocket_pool_fp16_plan(const rocket_pool_desc *d)
{
    /* A malformed descriptor is ROCKET_E_SHAPE and one past the PPU's fields or this
     * path's features is ROCKET_E_UNSUPPORTED, as in rocket_pool_int8_rk3576_plan(). */
    if (!d) return ROCKET_E_SHAPE;
    if (d->c < 1 || d->ih < 1 || d->iw < 1) return ROCKET_E_SHAPE;
    if (d->kh < 1 || d->kw < 1 || d->stride_y < 1 || d->stride_x < 1) return ROCKET_E_SHAPE;
    if (d->pad_top < 0 || d->pad_left < 0 || d->pad_bottom < 0 || d->pad_right < 0)
        return ROCKET_E_SHAPE;
    if (d->method != POOL_METHOD_MAX && d->method != POOL_METHOD_AVG) return ROCKET_E_SHAPE;
    /* count-include-pad = FALSE is claimed on the RK3576 int8 path alone
     * (rocket_pool_int8_rk3576()). These entries always divide by kh*kw, so a 1 here is
     * refused rather than silently dropped — the two are the same function only at pad 0
     * and a caller that asked has a padded window in mind. */
    if (d->avg_exclude_pad) return ROCKET_E_UNSUPPORTED;
    int oh = rocket_pool_oh(d), ow = rocket_pool_ow(d);
    if (oh < 1 || ow < 1) return ROCKET_E_SHAPE;
    /* CUBE_* dims are 13-bit (value-1); kernel/stride 4-bit (value-1); pad 3-bit. */
    if ((unsigned)(d->iw-1) >> 13 || (unsigned)(d->ih-1) >> 13 || (unsigned)(ow-1) >> 13 ||
        (unsigned)(oh-1) >> 13 || (unsigned)(d->c-1) >> 13) return ROCKET_E_UNSUPPORTED;
    if ((unsigned)(d->kw-1) >> 4 || (unsigned)(d->kh-1) >> 4 ||
        (unsigned)(d->stride_x-1) >> 4 || (unsigned)(d->stride_y-1) >> 4)
        return ROCKET_E_UNSUPPORTED;
    if (d->pad_left >> 3 || d->pad_top >> 3 || d->pad_right >> 3 || d->pad_bottom >> 3)
        return ROCKET_E_UNSUPPORTED;
    /* AVG reciprocal fp16(65536/k) overflows fp16 at k=1, so an average needs k>=2 per
     * axis (a 1-wide average kernel is degenerate anyway). MAX is fine with k=1. */
    if (d->method == POOL_METHOD_AVG && (d->kw < 2 || d->kh < 2)) return ROCKET_E_UNSUPPORTED;
    /* A window past 8 on either axis has no measured allowance (it fails at or near one
     * output column), and a plane past the allowance needs more bands than one job
     * carries: both refused here so a frontend declines at claim time. */
    if (rocket_pool_fp16_bands(d) == 0u) {
        ROCKET_LOGD("rocket_pool_fp16: a %dx%d window at stride %dx%d to an output width "
                    "of %d has no column banding inside the PPU's allowance (%u a task, "
                    "%d bands at most)\n", d->kh, d->kw, d->stride_y, d->stride_x,
                    ow, rocket_pool_fp16_max_ow(d), ROCKET_POOL_MAX_BANDS);
        return ROCKET_E_UNSUPPORTED;
    }
    return 0;
}

void rocket_pool_ref_fp16(const rocket_pool_desc *d, const _Float16 *in, _Float16 *out)
{
    const int C = d->c, IH = d->ih, IW = d->iw;
    const int OH = rocket_pool_oh(d), OW = rocket_pool_ow(d);
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                float acc = (d->method == POOL_METHOD_MAX) ? -INFINITY : 0.f;
                for (int kh = 0; kh < d->kh; kh++) {
                    int ih = oh * d->stride_y + kh - d->pad_top;
                    if (ih < 0 || ih >= IH) continue;
                    for (int kw = 0; kw < d->kw; kw++) {
                        int iw = ow * d->stride_x + kw - d->pad_left;
                        if (iw < 0 || iw >= IW) continue;
                        float v = (float)in[((size_t)c * IH + ih) * IW + iw];
                        if (d->method == POOL_METHOD_MAX) { if (v > acc) acc = v; }
                        else acc += v;
                    }
                }
                /* AVG divides by kh*kw (count-include-pad = TRUE), matching the PPU recip. */
                float o = (d->method == POOL_METHOD_AVG) ? acc / (float)(d->kh * d->kw) : acc;
                out[((size_t)c * OH + oh) * OW + ow] = (_Float16)o;
            }
}

/* CPU int8 reference (golden oracle / fd<0 fallback). int8 values are exactly
 * representable in fp16, so the int8 pool IS the fp16 pool of the same values, rounded
 * back to int8: MAX is exact integer max; AVG matches the fp16(65536/k) recip path. The
 * reference is therefore the fp16 oracle composed with round-to-nearest + clamp. */
void rocket_pool_ref_int8(const rocket_pool_desc *d, const int8_t *in, int8_t *out)
{
    size_t ni = (size_t)d->c*d->ih*d->iw, no = (size_t)d->c*rocket_pool_oh(d)*rocket_pool_ow(d);
    _Float16 *fi = malloc(ni*sizeof(_Float16)), *fo = malloc(no*sizeof(_Float16));
    if (!fi || !fo) { free(fi); free(fo); return; }
    for (size_t i = 0; i < ni; i++) fi[i] = (_Float16)(float)in[i];
    rocket_pool_ref_fp16(d, fi, fo);
    for (size_t i = 0; i < no; i++) {
        long o = lrintf((float)fo[i]);
        if (o > 127) o = 127; else if (o < -128) o = -128;
        out[i] = (int8_t)o;
    }
    free(fi); free(fo);
}

/* ============================================================================
 * SECTION — int8/uint8 pooling (routed through fp16)
 * ==========================================================================*/

/* int8/uint8 pooling on the PPU, ROUTED THROUGH THE fp16 PATH.
 *
 * The routing is a CHOICE, not a hardware bound. The PPU does have a native int8 pooling
 * precision — `PPU_DATA_FORMAT[2:0]` = 0 with `PPU_RDMA_DATA_FORMAT[1:0]` = 1 over a
 * packed int8 C2=16 cube pools MAX, MIN and AVG bit-exactly, given an INTEGER Q16
 * `0x10000/k` reciprocal and the integer pad fill (-128 sign-extended, not fp16 -inf)
 * [HW sweep, Turing RK1, 2026-09-20, tests/pool_int8_native_probe.c]. An earlier probe
 * concluded otherwise because it programmed IN_PRECISION 0, which is the 4-bit storage
 * width; that negative is withdrawn.
 *
 * This path stays fp16-routed because it is correct and because the native one is not a
 * win on its own — what it would buy is the conversion either side, and a cube-resident
 * int8 primitive for the fused-partition path. Every int8/uint8 value is exact in fp16,
 * so lifting the feature, running the proven fp16 PPU job and narrowing back gives:
 *   - MAX is BIT-EXACT (fp16 max of exact values == int max; round-trip lossless).
 *   - AVG matches the fp16(65536/k) recip (within fp16/recip tolerance), then rounds.
 * For uint8 the feature is recentered by -128 (MAX is shift-invariant; an AVG of the
 * centered values + 128 == the AVG) so the fp16 domain stays small. `recenter` selects
 * the uint8 byte interpretation + output clamp [0,255] vs signed int8 [-128,127]. */
static int rocket_pool_int8_core(int fd, const rocket_pool_desc *d,
                                 const int8_t *in, int8_t *out, int recenter)
{
    int pr = rocket_pool_fp16_plan(d);   /* geometry rules are dtype-independent */
    if (pr) { ROCKET_LOGE("rocket_pool_int8: unsupported (%d)\n", pr); return pr; }

    const size_t ni = (size_t)d->c*d->ih*d->iw;
    const size_t no = (size_t)d->c*rocket_pool_oh(d)*rocket_pool_ow(d);
    const int bias = recenter ? 128 : 0;

    _Float16 *fi = malloc(ni*sizeof(_Float16)), *fo = malloc(no*sizeof(_Float16));
    if (!fi || !fo) { free(fi); free(fo); return -20; }
    for (size_t i = 0; i < ni; i++) {
        int v = recenter ? (int)(uint8_t)in[i] : (int)in[i];
        fi[i] = (_Float16)(float)(v - bias);
    }
    int r = rocket_pool_fp16(fd, d, fi, fo);    /* fd<0 -> the fp16 CPU oracle */
    if (r) { free(fi); free(fo); return r; }
    for (size_t i = 0; i < no; i++) {
        long o = lrintf((float)fo[i]) + bias;
        if (recenter) { if (o < 0) o = 0; else if (o > 255) o = 255; ((uint8_t*)out)[i] = (uint8_t)o; }
        else          { if (o < -128) o = -128; else if (o > 127) o = 127; out[i] = (int8_t)o; }
    }
    free(fi); free(fo);
    return 0;
}

int rocket_pool_int8(int fd, const rocket_pool_desc *d, const int8_t *in, int8_t *out)
{ return rocket_pool_int8_core(fd, d, in, out, 0); }

int rocket_pool_uint8(int fd, const rocket_pool_desc *d, const uint8_t *in, uint8_t *out)
{ return rocket_pool_int8_core(fd, d, (const int8_t*)in, (int8_t*)out, 1); }

/* ============================================================================
 * SECTION — fp16 PPU pooling device path
 * ==========================================================================*/

int rocket_pool_fp16(int fd, const rocket_pool_desc *d, const _Float16 *in, _Float16 *out)
{
    int pr = rocket_pool_fp16_plan(d);
    if (pr) { ROCKET_LOGE("rocket_pool_fp16: unsupported (%d)\n", pr); return pr; }

    const int C = d->c, IH = d->ih, IW = d->iw;
    const int OH = rocket_pool_oh(d), OW = rocket_pool_ow(d);

    /* No device: the CPU oracle (host fallback / off-device self-check). */
    if (fd < 0) { rocket_pool_ref_fp16(d, in, out); return 0; }
    /* A driver that cannot retire the job: refuse before any BO is allocated, so the
     * caller's own host kernel runs instead of a 500 ms timeout. */
    if (!rocket_pool_npu_supported()) return ROCKET_E_UNSUPPORTED;

    const int C2 = 8;
    const int C1 = (C + C2 - 1) / C2;
    /* A plane wider than the PPU's per-task allowance runs as column bands, one task each,
     * all in one job. Each band is scattered into its own dense cube, so a band is an
     * ordinary pool and the generator's program for it is the one a lone pool of that
     * width gets; a plane inside the allowance is one band and runs the program it always
     * did. The input columns two bands' windows share are scattered into both. */
    const unsigned nb = rocket_pool_fp16_bands(d);    /* > 0: the plan above checked it */
    typedef struct { unsigned x0, ow, col, iw; int pl, pr; size_t in_off, out_off; } band_t;
    band_t *bd = calloc(nb, sizeof(*bd));
    rocket_task_desc *tasks = calloc(nb, sizeof(*tasks));
    size_t in_elems = 0, out_elems = 0;

    rocket_bo guard = {0}, in_bo = {0}, rc_bo = {0}, out_bo = {0};
    int rc = -1, ret = -1;

    if (!bd || !tasks) { ROCKET_LOGE("rocket_pool_fp16: out of memory\n"); goto out; }
    for (unsigned s = 0; s < nb; s++) {
        band_t *b = &bd[s];
        pool_band(d, s, &b->x0, &b->ow, &b->col, &b->iw, &b->pl, &b->pr);
        b->in_off = in_elems;   b->out_off = out_elems;
        in_elems  += (size_t)C1 * IH * b->iw * C2;
        out_elems += (size_t)C1 * OH * b->ow * C2;
    }

    if (rocket_bo_alloc(fd, 4096, &guard) ||                                  /* off IOVA 0 */
        rocket_bo_alloc(fd, in_elems * sizeof(_Float16) + POOL_BO_SLACK, &in_bo) ||
        rocket_bo_alloc(fd, (size_t)nb * POOL_TASK_SLOT * sizeof(uint64_t), &rc_bo) ||
        rocket_bo_alloc(fd, out_elems * sizeof(_Float16) + POOL_BO_SLACK, &out_bo)) {
        ROCKET_LOGE("rocket_pool_fp16: BO alloc failed\n"); goto out;
    }
    {
        rocket_bo *const set[] = { &in_bo, &rc_bo, &out_bo };
        if (rocket_op_iova_overflow("rocket_pool_fp16", set, 3)) goto out;
    }

    /* scatter each band's input columns -> its own NC1HWC2 cube (C2=8) */
    rocket_bo_prep(fd, &in_bo, 1, 0);
    memset(in_bo.ptr, 0, in_bo.size);
    for (unsigned s = 0; s < nb; s++) {
        const band_t *b = &bd[s];
        _Float16 *dst = (_Float16 *)in_bo.ptr + b->in_off;
        for (int c = 0; c < C; c++)
            for (int h = 0; h < IH; h++)
                for (int w = 0; w < (int)b->iw; w++)
                    dst[feature_index(C, IH, (int)b->iw, 8, c + 1, h + 1, w + 1)] =
                        in[((size_t)c * IH + h) * IW + b->col + (unsigned)w];
    }
    rocket_bo_fini(fd, &in_bo);

    /* one program per band, each in its own 16-byte-aligned slot of the regcmd BO */
    rocket_bo_prep(fd, &rc_bo, 1, 0);
    memset(rc_bo.ptr, 0, rc_bo.size);
    for (unsigned s = 0; s < nb; s++) {
        const band_t *b = &bd[s];
        uint64_t regs[POOL_TASK_SLOT];
        pool_params_t p = {
            .c = C, .ih = IH, .iw = (uint16_t)b->iw, .oh = OH, .ow = (uint16_t)b->ow,
            .kh = d->kh, .kw = d->kw,
            .stride_y = d->stride_y, .stride_x = d->stride_x,
            .pad_top = d->pad_top, .pad_left = (uint8_t)b->pl,
            .pad_bottom = d->pad_bottom, .pad_right = (uint8_t)b->pr,
            .method = (uint8_t)d->method,
            .recip_w = ppu_recip_kernel_fp16(d->kw),
            .recip_h = ppu_recip_kernel_fp16(d->kh),
            .input_iova  = (uint32_t)(in_bo.dma_address + b->in_off * sizeof(_Float16)),
            .output_iova = (uint32_t)(out_bo.dma_address + b->out_off * sizeof(_Float16)),
            .tasks = regs,
        };
        if ((ret = gen_pool_fp16(&p)) != 0) {
            ROCKET_LOGE("rocket_pool_fp16: gen failed (%d)\n", ret);
            rocket_bo_fini(fd, &rc_bo);
            goto out;
        }
        if (p.task_count == 0 || p.task_count > POOL_TASK_SLOT) {   /* bound regs[] */
            ROCKET_LOGE("rocket_pool_fp16: regcmd overflow (task_count %u)\n", p.task_count);
            rocket_bo_fini(fd, &rc_bo);
            ret = -1; goto out;
        }
        memcpy((uint64_t *)rc_bo.ptr + (size_t)s * POOL_TASK_SLOT, regs,
               (size_t)p.task_count * sizeof(uint64_t));
        tasks[s].regcmd = (uint32_t)(rc_bo.dma_address +
                                     (size_t)s * POOL_TASK_SLOT * sizeof(uint64_t));
        tasks[s].regcmd_count = p.task_count;
    }
    rocket_bo_fini(fd, &rc_bo);

    /* Prezero the output: a partial channel group's dead lanes are read back with the
     * live ones, and the de-scatter below indexes past them. */
    rocket_bo_prep(fd, &out_bo, 1, 0);
    memset(out_bo.ptr, 0, out_bo.size);
    rocket_bo_fini(fd, &out_bo);

    {
        uint32_t in_h[]  = { in_bo.handle, rc_bo.handle };
        uint32_t out_h[] = { out_bo.handle };
        /* This program is a POOL: it finishes on the PPU, not the DPU. Naming the
         * block costs the mainline driver's grace period nothing to skip and is what
         * lets a driver that waits for the named block retire the job at all. */
        ret = rocket_submit_tasks_flags(fd, tasks, nb, in_h, 2, out_h, 1,
                                        rocket_ppu_done_supported() ? ROCKET_JOB_PPU_DONE : 0u);
        if (ret) { ROCKET_LOGE("rocket_pool_fp16: submit failed (%d)\n", ret); goto out; }
        if ((ret = rocket_bo_prep(fd, &out_bo, 0, ROCKET_OP_WAIT_NS)) != 0) {
            ROCKET_LOGE("rocket_pool_fp16: the output fence did not signal within %llu ms "
                        "(%d)\n", (unsigned long long)(ROCKET_OP_WAIT_NS / 1000000ULL), ret);
            goto out;
        }
    }
    for (unsigned s = 0; s < nb; s++) {
        const band_t *b = &bd[s];
        const _Float16 *src = (const _Float16 *)out_bo.ptr + b->out_off;
        for (int c = 0; c < C; c++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < (int)b->ow; ow++)
                    out[((size_t)c * OH + oh) * OW + b->x0 + (unsigned)ow] =
                        src[feature_index(C, OH, (int)b->ow, 8, c + 1, oh + 1, ow + 1)];
    }
    rocket_bo_fini(fd, &out_bo);
    rc = 0;

out:
    rocket_bo_free(fd, &out_bo);
    rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &in_bo);
    rocket_bo_free(fd, &guard);
    free(tasks);
    free(bd);
    return rc ? (ret ? ret : -1) : 0;
}
