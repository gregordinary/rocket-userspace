// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_reduce.c — on-NPU spatial reductions (GlobalAveragePool / Mean / ReduceMean
 * over [H,W]) built on the PPU pooling engine.
 *
 * The PPU pool window is capped at 16x16 (4-bit kernel field), so a global/large mean
 * cannot run as one pool. This decomposes the global mean into a CHAIN of small
 * non-overlapping average pools and telescopes them — the average of equal-sized
 * group-averages is the grand average — keeping the intermediates resident in the
 * NC1HWC2 cube between passes (input scattered once, 1x1xC result de-scattered once).
 * See rocket_reduce.h.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_reduce.h"
#include "rocket_matmul.h" /* rocket_matmul_fp16_f32out (the feature-axis ones-matmul reduce) */
#include "npu_pool.h"     /* gen_pool_fp16, pool_params_t, ppu_recip_kernel_fp16, POOL_METHOD_AVG */
#include "npu_matmul.h"   /* feature_data (NC1HWC2 cube index) */
#include "rocket_log.h"     // centralized log channel

#define REDUCE_BO_SLACK 32768      /* DMA burst over-read slack past the cube */
#define REDUCE_MAX_PASS 24         /* factors-of-H + factors-of-W upper bound  */

int rocket_reduce_factor_axis(int n, int *f, int cap)
{
    if (n < 1) return -1;
    int c = 0;
    while (n > 16) {
        int d = 0;
        for (int x = 16; x >= 2; x--) if (n % x == 0) { d = x; break; }
        if (d == 0) return -1;                 /* prime factor > 16: not 16-smooth */
        if (c >= cap) return -1;
        f[c++] = d; n /= d;
    }
    if (n >= 2) { if (c >= cap) return -1; f[c++] = n; }
    /* Sort ASCENDING (smallest kernel first). This keeps the running quotient large for
     * as long as possible: after pass i it equals the product of the remaining (larger)
     * factors, hence >= the largest factor. For any axis > 16 the largest factor is >= 4,
     * so NO intermediate spatial dim is < 4 — and a PPU-WRITTEN sub-4 cube is not read
     * back correctly by the next chained pass (NPU->NPU only; standalone sub-4 pools are
     * fine; observed on-device, gated by reduce_mean_rocket's telescoping multi-pass). */
    for (int i = 1; i < c; i++)
        for (int j = i; j > 0 && f[j-1] > f[j]; j--) { int t = f[j]; f[j] = f[j-1]; f[j-1] = t; }
    return c;                                  /* n==1 -> 0 (no pooling on this axis) */
}

int rocket_global_avgpool_plan(int C, int H, int W)
{
    if (C < 1 || H < 1 || W < 1) return -1;
    if (H > ROCKET_REDUCE_MAX_DIM || W > ROCKET_REDUCE_MAX_DIM) return -2;
    if ((unsigned)(C - 1) >> 13) return -3;    /* PPU CUBE_*_CHANNEL is 13-bit (value-1) */
    int fh[REDUCE_MAX_PASS], fw[REDUCE_MAX_PASS];
    int nh = rocket_reduce_factor_axis(H, fh, REDUCE_MAX_PASS);
    int nw = rocket_reduce_factor_axis(W, fw, REDUCE_MAX_PASS);
    if (nh < 0) return -4;                       /* H not 16-smooth */
    if (nw < 0) return -5;                       /* W not 16-smooth */
    /* Equal factor count per axis => every pass pools BOTH axes (kernels >= 2) and no
     * axis is reduced to 1 ahead of the other. Combined with ascending order this keeps
     * all intermediates >= 4 (no PPU-written sub-4 chained cube). Covers all square maps
     * (H==W -> identical factor lists) and equal-count rectangles; the rare unequal-count
     * case (e.g. 56x8) host-falls-back. nh==nw==0 (1x1) is the trivial identity. */
    if (nh != nw) return -6;
    return 0;
}

void rocket_global_avgpool_ref_fp16(int C, int H, int W,
                                    const _Float16 *in, _Float16 *out)
{
    const double inv = 1.0 / ((double)H * (double)W);
    for (int c = 0; c < C; c++) {
        double acc = 0.0;
        const _Float16 *p = in + (size_t)c * H * W;
        for (int i = 0; i < H * W; i++) acc += (double)p[i];
        out[c] = (_Float16)(acc * inv);
    }
}

/* Per-axis average reciprocals for one pass. Symmetric kernel uses the exact validated
 * per-axis fp16(65536/k); an asymmetric kernel splits the divisor geometrically so the
 * product recip_h*recip_w*2^-32 still equals 1/(kh*kw) with both reciprocals in fp16
 * range (the tail passes of a non-square map). */
static void recip_pair(int kh, int kw, uint32_t *rh, uint32_t *rw)
{
    if (kh == kw) { *rh = *rw = ppu_recip_kernel_fp16(kh); return; }
    _Float16 h = (_Float16)(65536.0 / sqrt((double)kh * (double)kw));
    uint16_t b; memcpy(&b, &h, sizeof b);
    *rh = *rw = b;
}

int rocket_global_avgpool_fp16(int fd, int C, int H, int W,
                               const _Float16 *in, _Float16 *out)
{
    /* Non-decomposable shape, or no device: exact host reduction (always correct). */
    if (fd < 0 || rocket_global_avgpool_plan(C, H, W) != 0) {
        rocket_global_avgpool_ref_fp16(C, H, W, in, out);
        return 0;
    }

    int fh[REDUCE_MAX_PASS], fw[REDUCE_MAX_PASS];
    int nh = rocket_reduce_factor_axis(H, fh, REDUCE_MAX_PASS);
    int nw = rocket_reduce_factor_axis(W, fw, REDUCE_MAX_PASS);
    int npass = nh > nw ? nh : nw;

    const int C2 = 8;
    const int C1 = (C + C2 - 1) / C2;
    size_t cube_elems = (size_t)C1 * H * W * C2;          /* input cube == the largest */
    size_t cube_bytes = cube_elems * sizeof(_Float16) + REDUCE_BO_SLACK;

    rocket_bo guard = {0}, A = {0}, B = {0}, rc_bo = {0};
    uint64_t regs[64] = {0};
    int rc = -1, ret = -1;

    if (rocket_bo_alloc(fd, 4096, &guard) ||                 /* keep IOVA 0 unused */
        rocket_bo_alloc(fd, cube_bytes, &A) ||
        rocket_bo_alloc(fd, cube_bytes, &B) ||
        rocket_bo_alloc(fd, sizeof(regs), &rc_bo)) {
        ROCKET_LOGE("rocket_global_avgpool: BO alloc failed\n"); goto out;
    }
    if (((A.dma_address + A.size) | (B.dma_address + B.size) |
         (rc_bo.dma_address + rc_bo.size)) >> 32) {
        ROCKET_LOGE("rocket_global_avgpool: a BO dma_address exceeds 32 bits\n"); goto out;
    }

    /* scatter input feature [C][H][W] -> NC1HWC2 cube in A */
    rocket_bo_prep(fd, &A, 1, 0);
    memset(A.ptr, 0, A.size);
    {
        _Float16 *dst = A.ptr;
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    dst[feature_data(C, H, W, 8, c + 1, h + 1, w + 1)] =
                        in[((size_t)c * H + h) * W + w];
    }
    rocket_bo_fini(fd, &A);

    /* telescoping average-pool passes, ping-ponging A <-> B (device-resident) */
    int ch = H, cw = W;
    rocket_bo *final_bo = &A;
    int ran_pass = 0;
    for (int i = 0; i < npass; i++) {
        int kh = (i < nh) ? fh[i] : 1;
        int kw = (i < nw) ? fw[i] : 1;
        if (kh == 1 && kw == 1) continue;            /* nothing to do this pass */
        ran_pass = 1;
        int oh = ch / kh, ow = cw / kw;              /* exact tiling (kernel | size) */

        rocket_bo *src = (i & 1) ? &B : &A;
        rocket_bo *dst = (i & 1) ? &A : &B;

        uint32_t rh, rw; recip_pair(kh, kw, &rh, &rw);
        pool_params_t p = {
            .c = C, .ih = ch, .iw = cw, .oh = oh, .ow = ow,
            .kh = (uint8_t)kh, .kw = (uint8_t)kw,
            .stride_y = (uint8_t)kh, .stride_x = (uint8_t)kw,   /* non-overlapping tile */
            .pad_top = 0, .pad_left = 0, .pad_bottom = 0, .pad_right = 0,
            .method = POOL_METHOD_AVG, .recip_w = rw, .recip_h = rh,
            .input_dma = (uint32_t)src->dma_address,
            .output_dma = (uint32_t)dst->dma_address,
            .tasks = regs,
        };
        if ((ret = gen_pool_fp16(&p)) != 0) {
            ROCKET_LOGE("rocket_global_avgpool: gen pass %d failed (%d)\n", i, ret);
            goto out;
        }
        if (p.task_count > sizeof(regs)/sizeof(regs[0])) {   /* bound the stack regs[] */
            ROCKET_LOGE("rocket_global_avgpool: regcmd overflow (task_count %u)\n", p.task_count);
            ret = -1; goto out;
        }
        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);

        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo.dma_address,
                                  .regcmd_count = p.task_count };
        uint32_t in_h[]  = { src->handle, rc_bo.handle };
        uint32_t out_h[] = { dst->handle };
        if ((ret = rocket_submit_tasks(fd, &task, 1, in_h, 2, out_h, 1)) != 0) {
            ROCKET_LOGE("rocket_global_avgpool: submit pass %d failed (%d)\n", i, ret);
            goto out;
        }
        /* wait for this pass to land before it feeds the next (RAW chain) */
        if ((ret = rocket_bo_prep(fd, dst, 0, 2000000000ULL)) != 0) {
            ROCKET_LOGE("rocket_global_avgpool: wait pass %d (%d)\n", i, ret);
            goto out;
        }
        ch = oh; cw = ow; final_bo = dst;
    }

    /* If no pass ran (degenerate 1x1 reduce), the result BO is still A in its
     * post-write (device-owned) state -- it never got a read-prep in the loop, so
     * invalidate it for CPU read here before de-scattering. */
    if (!ran_pass && (ret = rocket_bo_prep(fd, final_bo, 0, 2000000000ULL)) != 0) {
        ROCKET_LOGE("rocket_global_avgpool: read-prep degenerate result (%d)\n", ret);
        goto out;
    }

    /* de-scatter the 1x1xC result (final cube is ch x cw == 1 x 1) */
    {
        _Float16 *src = final_bo->ptr;     /* already invalidated by the last prep(read) */
        for (int c = 0; c < C; c++)
            out[c] = src[feature_data(C, ch, cw, 8, c + 1, 1, 1)];
    }
    rc = 0;

out:
    rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &B);
    rocket_bo_free(fd, &A);
    rocket_bo_free(fd, &guard);
    return rc ? (ret ? ret : -1) : 0;
}

/* ===================================================================== *
 * GlobalMaxPool / GlobalMinPool — spatial ReduceMax/ReduceMin over [H,W] *
 * Same telescoping engine as the average pool, but MAX/MIN are exact and *
 * need no reciprocal. See rocket_reduce.h.                               *
 * ===================================================================== */

void rocket_global_maxpool_ref_fp16(int C, int H, int W,
                                    const _Float16 *in, _Float16 *out)
{
    for (int c = 0; c < C; c++) {
        const _Float16 *p = in + (size_t)c * H * W;
        _Float16 m = p[0];
        for (int i = 1; i < H * W; i++) if (p[i] > m) m = p[i];
        out[c] = m;
    }
}

void rocket_global_minpool_ref_fp16(int C, int H, int W,
                                    const _Float16 *in, _Float16 *out)
{
    for (int c = 0; c < C; c++) {
        const _Float16 *p = in + (size_t)c * H * W;
        _Float16 m = p[0];
        for (int i = 1; i < H * W; i++) if (p[i] < m) m = p[i];
        out[c] = m;
    }
}

/* Shared core for the two extrema reductions (method = POOL_METHOD_MAX / _MIN). Mirrors the
 * average-pool telescoping but drops the per-pass reciprocal (idempotent => exact). Kept
 * separate from rocket_global_avgpool_fp16 so that HW-validated path stays byte-for-byte. */
static int run_global_extrema_pool(int fd, int C, int H, int W, int method,
                                   const _Float16 *in, _Float16 *out)
{
    int fh[REDUCE_MAX_PASS], fw[REDUCE_MAX_PASS];
    int nh = rocket_reduce_factor_axis(H, fh, REDUCE_MAX_PASS);
    int nw = rocket_reduce_factor_axis(W, fw, REDUCE_MAX_PASS);
    int npass = nh > nw ? nh : nw;

    const int C2 = 8;
    const int C1 = (C + C2 - 1) / C2;
    size_t cube_elems = (size_t)C1 * H * W * C2;
    size_t cube_bytes = cube_elems * sizeof(_Float16) + REDUCE_BO_SLACK;

    rocket_bo guard = {0}, A = {0}, B = {0}, rc_bo = {0};
    uint64_t regs[64] = {0};
    int rc = -1, ret = -1;

    if (rocket_bo_alloc(fd, 4096, &guard) ||
        rocket_bo_alloc(fd, cube_bytes, &A) ||
        rocket_bo_alloc(fd, cube_bytes, &B) ||
        rocket_bo_alloc(fd, sizeof(regs), &rc_bo)) {
        ROCKET_LOGE("rocket_global_extrema_pool: BO alloc failed\n"); goto out;
    }
    if (((A.dma_address + A.size) | (B.dma_address + B.size) |
         (rc_bo.dma_address + rc_bo.size)) >> 32) {
        ROCKET_LOGE("rocket_global_extrema_pool: a BO dma_address exceeds 32 bits\n"); goto out;
    }

    /* scatter input feature [C][H][W] -> NC1HWC2 cube in A */
    rocket_bo_prep(fd, &A, 1, 0);
    memset(A.ptr, 0, A.size);
    {
        _Float16 *dst = A.ptr;
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    dst[feature_data(C, H, W, 8, c + 1, h + 1, w + 1)] =
                        in[((size_t)c * H + h) * W + w];
    }
    rocket_bo_fini(fd, &A);

    int ch = H, cw = W;
    rocket_bo *final_bo = &A;
    for (int i = 0; i < npass; i++) {
        int kh = (i < nh) ? fh[i] : 1;
        int kw = (i < nw) ? fw[i] : 1;
        if (kh == 1 && kw == 1) continue;
        int oh = ch / kh, ow = cw / kw;

        rocket_bo *src = (i & 1) ? &B : &A;
        rocket_bo *dst = (i & 1) ? &A : &B;

        pool_params_t p = {
            .c = C, .ih = ch, .iw = cw, .oh = oh, .ow = ow,
            .kh = (uint8_t)kh, .kw = (uint8_t)kw,
            .stride_y = (uint8_t)kh, .stride_x = (uint8_t)kw,
            .pad_top = 0, .pad_left = 0, .pad_bottom = 0, .pad_right = 0,
            .method = (uint8_t)method, .recip_w = 0, .recip_h = 0,   /* recip ignored for MAX/MIN */
            .input_dma = (uint32_t)src->dma_address,
            .output_dma = (uint32_t)dst->dma_address,
            .tasks = regs,
        };
        if ((ret = gen_pool_fp16(&p)) != 0) {
            ROCKET_LOGE("rocket_global_extrema_pool: gen pass %d failed (%d)\n", i, ret);
            goto out;
        }
        if (p.task_count > sizeof(regs)/sizeof(regs[0])) {   /* bound the stack regs[] */
            ROCKET_LOGE("rocket_global_extrema_pool: regcmd overflow (task_count %u)\n", p.task_count);
            ret = -1; goto out;
        }
        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);

        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo.dma_address,
                                  .regcmd_count = p.task_count };
        uint32_t in_h[]  = { src->handle, rc_bo.handle };
        uint32_t out_h[] = { dst->handle };
        if ((ret = rocket_submit_tasks(fd, &task, 1, in_h, 2, out_h, 1)) != 0) {
            ROCKET_LOGE("rocket_global_extrema_pool: submit pass %d failed (%d)\n", i, ret);
            goto out;
        }
        if ((ret = rocket_bo_prep(fd, dst, 0, 2000000000ULL)) != 0) {
            ROCKET_LOGE("rocket_global_extrema_pool: wait pass %d (%d)\n", i, ret);
            goto out;
        }
        ch = oh; cw = ow; final_bo = dst;
    }

    {
        _Float16 *src = final_bo->ptr;
        for (int c = 0; c < C; c++)
            out[c] = src[feature_data(C, ch, cw, 8, c + 1, 1, 1)];
    }
    rc = 0;

out:
    rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &B);
    rocket_bo_free(fd, &A);
    rocket_bo_free(fd, &guard);
    return rc ? (ret ? ret : -1) : 0;
}

int rocket_global_maxpool_fp16(int fd, int C, int H, int W,
                               const _Float16 *in, _Float16 *out)
{
    if (fd < 0 || rocket_global_avgpool_plan(C, H, W) != 0) {
        rocket_global_maxpool_ref_fp16(C, H, W, in, out);
        return 0;
    }
    return run_global_extrema_pool(fd, C, H, W, POOL_METHOD_MAX, in, out);
}

int rocket_global_minpool_fp16(int fd, int C, int H, int W,
                               const _Float16 *in, _Float16 *out)
{
    if (fd < 0 || rocket_global_avgpool_plan(C, H, W) != 0) {
        rocket_global_minpool_ref_fp16(C, H, W, in, out);
        return 0;
    }
    return run_global_extrema_pool(fd, C, H, W, POOL_METHOD_MIN, in, out);
}

/* ===================================================================== *
 * Feature-axis reduce (sum/mean over the hidden axis, per row)          *
 * via a ones-vector matmul. See rocket_reduce.h for the rationale.      *
 * ===================================================================== */

void rocket_reduce_feature_ref_fp16(int M, int H,
                                    const _Float16 *in, float *out, int mean)
{
    const double inv = mean ? (1.0 / (double)H) : 1.0;
    for (int m = 0; m < M; m++) {
        double acc = 0.0;
        const _Float16 *p = in + (size_t)m * H;
        for (int h = 0; h < H; h++) acc += (double)p[h];
        out[m] = (float)(acc * inv);
    }
}

int rocket_reduce_feature_fp16(int fd, int M, int H,
                               const _Float16 *in, float *out, int mean)
{
    if (M < 1 || H < 1) return -1;

    /* No device: exact host reduction (always correct). */
    if (fd < 0) { rocket_reduce_feature_ref_fp16(M, H, in, out, mean); return 0; }

    /* out[m] = sum_h in[m,h] = (in[M,H] · ones[N,H]^T)[m, *]. The matmul needs
     * K%32, N%16, M%4; pad H->Kpad, M->Mpad with zeros (zero rows/cols add 0 to the
     * dot product) and use N=16 ones-columns, reading column 0 of the fp32 result. */
    const int N    = 16;
    const int Kpad = (H + 31) & ~31;
    const int Mpad = (M +  3) & ~3;

    /* ones weight [N, Kpad]: 1 over the real H columns, 0 over the K-pad. */
    _Float16 *ones = malloc((size_t)N * Kpad * sizeof(_Float16));
    float    *C    = malloc((size_t)Mpad * N * sizeof(float));
    if (!ones || !C) { free(ones); free(C); return -2; }
    for (int r = 0; r < N; r++) {
        _Float16 *row = ones + (size_t)r * Kpad;
        for (int k = 0; k < H;    k++) row[k] = (_Float16)1.0f;
        for (int k = H; k < Kpad; k++) row[k] = (_Float16)0.0f;
    }

    /* A is `in` directly when it already meets the alignment (the common LLM case:
     * H%32==0, M%4==0 — no host copy); else a zero-padded [Mpad,Kpad] staging copy. */
    const _Float16 *A = in;
    _Float16 *Apad = NULL;
    if (Kpad != H || Mpad != M) {
        Apad = calloc((size_t)Mpad * Kpad, sizeof(_Float16));
        if (!Apad) { free(ones); free(C); return -2; }
        for (int m = 0; m < M; m++)
            memcpy(Apad + (size_t)m * Kpad, in + (size_t)m * H, (size_t)H * sizeof(_Float16));
        A = Apad;
    }

    int r = rocket_matmul_fp16_f32out(fd, Mpad, Kpad, N, A, ones, C);
    if (r == 0) {
        const float inv = mean ? (1.0f / (float)H) : 1.0f;
        for (int m = 0; m < M; m++) out[m] = C[(size_t)m * N] * inv;   /* column 0 */
    }

    free(Apad); free(ones); free(C);
    return r;
}

/* ===================================================================== *
 * Cumsum (prefix sum along the last axis) — the feature-reduce widened   *
 * from a single ones-COLUMN to a full triangular ones MATRIX.            *
 * See rocket_reduce.h for the lowering rationale.                        *
 * ===================================================================== */

void rocket_cumsum_ref_fp16(int M, int N, const _Float16 *in, _Float16 *out,
                            int exclusive, int reverse)
{
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = in  + (size_t)m * N;
        _Float16       *op = out + (size_t)m * N;
        double acc = 0.0;
        if (!reverse) {
            for (int j = 0; j < N; j++) {
                if (exclusive) { op[j] = (_Float16)acc; acc += (double)xp[j]; }
                else           { acc += (double)xp[j]; op[j] = (_Float16)acc; }
            }
        } else {
            for (int j = N - 1; j >= 0; j--) {
                if (exclusive) { op[j] = (_Float16)acc; acc += (double)xp[j]; }
                else           { acc += (double)xp[j]; op[j] = (_Float16)acc; }
            }
        }
    }
}

int rocket_cumsum_fp16(int fd, int M, int N, const _Float16 *in, _Float16 *out,
                       int exclusive, int reverse)
{
    if (M < 1 || N < 1) return -1;

    /* No device: exact host scan (always correct). */
    if (fd < 0) { rocket_cumsum_ref_fp16(M, N, in, out, exclusive, reverse); return 0; }

    /* out[M,N] = in[M,N] · L^T. The contraction K == the input width N (pad %32); the output
     * column count also == N (pad %16, the matmul N-group); M pads %4. The triangular weight
     * L[Npad,Kpad] selects the prefix columns; pad rows/cols are 0 (contribute nothing). */
    const int Kpad = (N + 31) & ~31;
    const int Npad = (N + 15) & ~15;
    const int Mpad = (M +  3) & ~3;

    _Float16 *L = calloc((size_t)Npad * Kpad, sizeof(_Float16));   /* zero pad rows + cols */
    float    *C = malloc((size_t)Mpad * Npad * sizeof(float));
    if (!L || !C) { free(L); free(C); return -2; }

    /* Triangular ones: L[n][k] = 1 iff input column k is in prefix n (the requested variant).
     * Only the real [N,N] block is set; k>=N (K-pad) and n>=N (unused output cols) stay 0. */
    for (int n = 0; n < N; n++) {
        _Float16 *row = L + (size_t)n * Kpad;
        for (int k = 0; k < N; k++) {
            int one = reverse ? (exclusive ? (k > n) : (k >= n))
                              : (exclusive ? (k < n) : (k <= n));
            row[k] = one ? (_Float16)1.0f : (_Float16)0.0f;
        }
    }

    /* A is `in` directly when it already meets K%32 / M%4 (no host copy); else a zero-padded
     * [Mpad,Kpad] staging copy (zero columns contribute 0; zero rows produce ignored output). */
    const _Float16 *A = in;
    _Float16 *Apad = NULL;
    if (Kpad != N || Mpad != M) {
        Apad = calloc((size_t)Mpad * Kpad, sizeof(_Float16));
        if (!Apad) { free(L); free(C); return -2; }
        for (int m = 0; m < M; m++)
            memcpy(Apad + (size_t)m * Kpad, in + (size_t)m * N, (size_t)N * sizeof(_Float16));
        A = Apad;
    }

    int r = rocket_matmul_fp16_f32out(fd, Mpad, Kpad, Npad, A, L, C);
    if (r == 0) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
                out[(size_t)m * N + n] = (_Float16)C[(size_t)m * Npad + n];   /* narrow fp32->fp16 */
    }

    free(Apad); free(L); free(C);
    return r;
}
