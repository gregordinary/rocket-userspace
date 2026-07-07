// SPDX-License-Identifier: GPL-3.0-or-later
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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_pool.h"
#include "npu_matmul.h"   /* feature_data (NC1HWC2 cube index) */
#include "rocket_log.h"     // centralized log channel

/* slack past the scattered cube, so a DMA burst over-read can't run off the BO */
#define POOL_BO_SLACK 32768

/* ============================================================================
 * SECTION — Geometry validation and CPU references
 * ==========================================================================*/

int rocket_pool_fp16_plan(const rocket_pool_desc *d)
{
    if (!d) return -1;
    if (d->c < 1 || d->ih < 1 || d->iw < 1) return -2;
    if (d->kh < 1 || d->kw < 1 || d->stride_y < 1 || d->stride_x < 1) return -3;
    if (d->method != POOL_METHOD_MAX && d->method != POOL_METHOD_AVG) return -4;
    int oh = rocket_pool_oh(d), ow = rocket_pool_ow(d);
    if (oh < 1 || ow < 1) return -5;
    /* CUBE_* dims are 13-bit (value-1); kernel/stride 4-bit (value-1); pad 3-bit. */
    if ((unsigned)(d->iw-1) >> 13 || (unsigned)(d->ih-1) >> 13 || (unsigned)(ow-1) >> 13 ||
        (unsigned)(oh-1) >> 13 || (unsigned)(d->c-1) >> 13) return -6;
    if ((unsigned)(d->kw-1) >> 4 || (unsigned)(d->kh-1) >> 4 ||
        (unsigned)(d->stride_x-1) >> 4 || (unsigned)(d->stride_y-1) >> 4) return -7;
    if ((unsigned)d->pad_left >> 3 || (unsigned)d->pad_top >> 3 ||
        (unsigned)d->pad_right >> 3 || (unsigned)d->pad_bottom >> 3) return -8;
    /* AVG reciprocal fp16(65536/k) overflows fp16 at k=1, so an average needs k>=2 per
     * axis (a 1-wide average kernel is degenerate anyway). MAX is fine with k=1. */
    if (d->method == POOL_METHOD_AVG && (d->kw < 2 || d->kh < 2)) return -9;
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
 * NPU FACT (RE'd 2026-06-22): the RK3588 PPU has NO native int8 pooling precision. A
 * PPU job with PROC_PRECISION/IN_PRECISION=int8 (0) over a packed int8 C2=16 cube does
 * NOT pool in int8 — the HW reads the bytes as fp16 (garbage). HW-verified: PROC_PRECISION(2)=fp16
 * is required for EVERY pool (no int8 pool precision exists). So int8/uint8 pooling lifts the
 * feature into fp16 (every
 * int8/uint8 value is exact in fp16), runs the proven fp16 PPU job, and narrows back:
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

    const int C2 = 8;
    const int C1 = (C + C2 - 1) / C2;
    size_t in_cube_elems  = (size_t)C1 * IH * IW * C2;
    size_t out_cube_elems = (size_t)C1 * OH * OW * C2;

    rocket_bo guard = {0}, in_bo = {0}, rc_bo = {0}, out_bo = {0};
    uint64_t regs[64] = {0};
    int rc = -1, ret = -1;

    if (rocket_bo_alloc(fd, 4096, &guard) ||                                  /* off IOVA 0 */
        rocket_bo_alloc(fd, in_cube_elems * sizeof(_Float16) + POOL_BO_SLACK, &in_bo) ||
        rocket_bo_alloc(fd, sizeof(regs), &rc_bo) ||
        rocket_bo_alloc(fd, out_cube_elems * sizeof(_Float16) + POOL_BO_SLACK, &out_bo)) {
        ROCKET_LOGE("rocket_pool_fp16: BO alloc failed\n"); goto out;
    }
    if (((in_bo.dma_address + in_bo.size) | (out_bo.dma_address + out_bo.size) |
         (rc_bo.dma_address + rc_bo.size)) >> 32) {
        ROCKET_LOGE("rocket_pool_fp16: a BO dma_address exceeds 32 bits\n"); goto out;
    }

    /* scatter input feature -> NC1HWC2 cube (C2=8) */
    rocket_bo_prep(fd, &in_bo, 1, 0);
    memset(in_bo.ptr, 0, in_bo.size);
    {
        _Float16 *dst = in_bo.ptr;
        for (int c = 0; c < C; c++)
            for (int h = 0; h < IH; h++)
                for (int w = 0; w < IW; w++)
                    dst[feature_data(C, IH, IW, 8, c + 1, h + 1, w + 1)] =
                        in[((size_t)c * IH + h) * IW + w];
    }
    rocket_bo_fini(fd, &in_bo);

    pool_params_t p = {
        .c = C, .ih = IH, .iw = IW, .oh = OH, .ow = OW,
        .kh = d->kh, .kw = d->kw,
        .stride_y = d->stride_y, .stride_x = d->stride_x,
        .pad_top = d->pad_top, .pad_left = d->pad_left,
        .pad_bottom = d->pad_bottom, .pad_right = d->pad_right,
        .method = (uint8_t)d->method,
        .recip_w = ppu_recip_kernel_fp16(d->kw),
        .recip_h = ppu_recip_kernel_fp16(d->kh),
        .input_dma  = (uint32_t)in_bo.dma_address,
        .output_dma = (uint32_t)out_bo.dma_address,
        .tasks = regs,
    };
    if ((ret = gen_pool_fp16(&p)) != 0) {
        ROCKET_LOGE("rocket_pool_fp16: gen failed (%d)\n", ret); goto out;
    }
    if (p.task_count > sizeof(regs)/sizeof(regs[0])) {   /* bound the stack regs[] */
        ROCKET_LOGE("rocket_pool_fp16: regcmd overflow (task_count %u)\n", p.task_count);
        ret = -1; goto out;
    }
    rocket_bo_prep(fd, &rc_bo, 1, 0);
    memcpy(rc_bo.ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &rc_bo);

    rocket_bo_prep(fd, &out_bo, 1, 0);
    memset(out_bo.ptr, 0, out_bo.size);
    rocket_bo_fini(fd, &out_bo);

    {
        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo.dma_address,
                                  .regcmd_count = p.task_count };
        uint32_t in_h[]  = { in_bo.handle, rc_bo.handle };
        uint32_t out_h[] = { out_bo.handle };
        ret = rocket_submit_tasks(fd, &task, 1, in_h, 2, out_h, 1);
        if (ret) { ROCKET_LOGE("rocket_pool_fp16: submit failed (%d)\n", ret); goto out; }
    }

    ret = rocket_bo_prep(fd, &out_bo, 0, 2000000000ULL);   /* 2s wait */
    if (ret) { ROCKET_LOGE("rocket_pool_fp16: wait timeout (%d)\n", ret); goto out; }
    {
        _Float16 *src = out_bo.ptr;
        for (int c = 0; c < C; c++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                    out[((size_t)c * OH + oh) * OW + ow] =
                        src[feature_data(C, OH, OW, 8, c + 1, oh + 1, ow + 1)];
    }
    rocket_bo_fini(fd, &out_bo);
    rc = 0;

out:
    rocket_bo_free(fd, &out_bo);
    rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &in_bo);
    rocket_bo_free(fd, &guard);
    return rc ? (ret ? ret : -1) : 0;
}
