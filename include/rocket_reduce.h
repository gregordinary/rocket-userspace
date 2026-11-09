// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_REDUCE_H
#define ROCKET_REDUCE_H

#include <stdint.h>

/*
 * Declares the spatial reductions on the NPU, built on the PPU pooling engine, and the
 * feature-axis reductions, built on the matmul.
 *
 * The spatial entries compute GlobalAveragePool, Mean and ReduceMean over the spatial axes
 * [H,W] of each channel:
 *
 *     out[c] = (1 / (H*W)) * sum_{h,w} in[c][h][w]
 *
 * They cover TFLite MEAN over axes [1,2] with keepdims, ONNX GlobalAveragePool and
 * ReduceMean, the squeeze of every SE block, and most classifier and detection heads.
 *
 * The PPU's `POOLING_KERNEL_CFG` kernel and stride fields are 4-bit (value-1), so one pool
 * window is at most 16x16. On the RK3588 a window past 8x8 computes incorrect output
 * (`rocket_pool_fp16_max_ow()`, `rocket_pool.h`). A global pool over 56x56, for example,
 * cannot run as one pool. The entry decomposes the global mean into a chain of small
 * non-overlapping average pools and telescopes them:
 *
 *     mean over k1*k2*...*kn equal groups  ==  (...((mean_k1)mean_k2)...)mean_kn
 *
 * The identity holds because the average of equal-sized group averages is the grand
 * average. Each pass tiles the current extent exactly (kernel | size, stride = kernel,
 * pad = 0). The product of the per-pass divisors is therefore exactly H*W. Intermediates
 * stay resident in the NC1HWC2 cube between passes with no host round trip. The input is
 * scattered once and the 1x1xC result is de-tiled once.
 *
 * A shape runs on the NPU under three conditions:
 *
 * - Each axis is 8-smooth. Every prime factor is <= 7, so the axis factors into kernels in
 *   [2,8]. This covers every power of two and most feature-map sizes
 *   (7,14,28,56,49,98,112,224,...).
 * - `H` and `W` have the same number of factors, so every pass pools both axes and neither
 *   collapses to 1 ahead of the other. Every square map (H==W, the usual GlobalAvgPool
 *   case) and every equal-count rectangle meets it.
 * - Every pass's output width is within the PPU's per-task allowance for its window
 *   (`rocket_pool_fp16_max_ow()`): 128 for a 2x2 pass and 64 for a larger one. The passes
 *   run smallest window first. Where that order has a pass past its allowance, they run
 *   largest first (216: 8x8 first, to 27). A plane neither order fits, such as one 1024
 *   wide, computes on the host.
 *
 * Caution: a pass past the allowance writes a complete, incorrect cube. The passes after it
 * reduce that cube into a plausible incorrect mean [HW sweep, RK1, 2026-11-02]. The planes
 * measured were 216, 256, 288 and 512 square, with windows to 16.
 *
 * A shape that fails a condition falls back to an exact host reduction, which still
 * returns the correct answer. A shape fails with a prime factor of 11 or more, with unequal
 * factor counts, or with a pass past its allowance in both orders.
 * `rocket_global_avgpool_plan()` reports which path a shape takes before the call.
 *
 * Each pass divides by its kernel through the PPU's per-axis fp16 reciprocal fp16(65536/k),
 * because the PPU has no hardware divider (`npu_pool.h`). A symmetric kernel uses the
 * validated per-axis reciprocal exactly. An asymmetric kernel, which occurs only in the tail
 * passes of a non-square map, splits the divisor geometrically across the two axes. Both
 * keep the product == 1/(kh*kw). The net error is fp16 rounding plus ~1e-3 of reciprocal
 * quantization per pass, so a test must compare with a relative tolerance, not equality.
 */

/* ============================================================================
 * Spatial reduce: GlobalAveragePool and Mean over [H,W]
 * ==========================================================================*/

/* The largest input `H` or `W` this op accepts. The cube dimension field is 13-bit, which is
 * well above any real feature map, and this limit bounds the factor buffers. */
#define ROCKET_REDUCE_MAX_DIM 8192

/* Returns 0 for a `(C,H,W)` that runs entirely on the NPU, and a negative `rocket_status`
 * otherwise. To run on the NPU, a shape needs both axes 8-smooth with an equal factor
 * count. It also needs every pass inside its allowance and its dimensions in range. The
 * statuses:
 *
 * - `ROCKET_E_SHAPE`: a dimension below 1
 * - `ROCKET_E_UNSUPPORTED`: `H` or `W` above `ROCKET_REDUCE_MAX_DIM`, or `C` above 8192
 * - `ROCKET_E_TILING`: an axis that is not 8-smooth, unequal factor counts, or a pass past
 *   the PPU's output-width allowance
 *
 * A negative result is not an error for `rocket_global_avgpool_fp16`, which then computes
 * on the host. It lets a caller, such as the TFLite delegate, choose the path before the
 * call. The function is pure and touches no hardware, so a caller choosing the path also
 * asks `rocket_pool_npu_supported()` (`rocket_pool.h`). */

#ifdef __cplusplus
extern "C" {
#endif
int rocket_global_avgpool_plan(int C, int H, int W);

/* Writes the pass windows the three global pools run a `(C,H,W)` as, in run order. Pass
 * `i` pools a `kh[i]` x `kw[i]` window at a stride equal to the window. Returns the pass
 * count (0 for a 1x1 plane), or `ROCKET_E_SHAPE` where it exceeds `cap`. For a shape the
 * plan refuses, it returns the plan's negative status. It is pure. */
int rocket_global_pool_passes(int C, int H, int W, int *kh, int *kw, int cap);

/* Computes GlobalAveragePool, the spatial mean over [H,W] of each channel.
 *
 * `in` is row-major fp16 [C][H][W], channels-major, the conv and pool feature layout. `out`
 * is fp16 [C], the per-channel spatial mean. `fd` is an open rocket device from
 * `rocket_open()`. With `fd` < 0, for a shape that does not decompose, or where
 * `rocket_pool_npu_supported()` returns 0, it computes the exact host reduction. Returns 0
 * on success and a negative value on error. */
int rocket_global_avgpool_fp16(int fd, int C, int H, int W,
                               const _Float16 *in, _Float16 *out);

/* Computes the CPU reference with fp64 accumulation. The tests compare against it, and it
 * is the `fd` < 0 and fallback path. It runs on the host only and touches no hardware. */
void rocket_global_avgpool_ref_fp16(int C, int H, int W,
                                    const _Float16 *in, _Float16 *out);

/* ============================================================================
 * Spatial reduce: GlobalMaxPool, GlobalMinPool and the factor helper
 * ==========================================================================*/

/*
 * Computes GlobalMaxPool or GlobalMinPool, the spatial ReduceMax or ReduceMin over [H,W] of
 * each channel:
 *
 *     out[c] = max (or min) over { in[c][h][w] }
 *
 * They cover ONNX GlobalMaxPool and ReduceMax or ReduceMin over axes [2,3], TFLite
 * REDUCE_MAX and REDUCE_MIN over the spatial axes, and global-max-pool classifier and
 * attention heads.
 *
 * They use the same telescoping decomposition and the same shape conditions as
 * GlobalAveragePool above, because one pool window is at most 8x8 on this path.
 * `rocket_global_avgpool_plan()` tests a shape for them too. MAX and MIN are idempotent, so
 * no pass needs a reciprocal (max-of-block-maxes == global max). The result is exact, with
 * no fp16 reciprocal rounding, and a decomposable shape is bit-exact against the host
 * reduction.
 *
 * With `fd` < 0, for a shape that does not decompose, or where `rocket_pool_npu_supported()`
 * returns 0, they compute the exact host reduction. The layout and the return values are
 * those of `rocket_global_avgpool_fp16`. */
int rocket_global_maxpool_fp16(int fd, int C, int H, int W,
                               const _Float16 *in, _Float16 *out);
int rocket_global_minpool_fp16(int fd, int C, int H, int W,
                               const _Float16 *in, _Float16 *out);

/* Computes the CPU references for max and min. The tests compare against them, and they are
 * the `fd` < 0 and fallback path. They run on the host only. */
void rocket_global_maxpool_ref_fp16(int C, int H, int W,
                                    const _Float16 *in, _Float16 *out);
void rocket_global_minpool_ref_fp16(int C, int H, int W,
                                    const _Float16 *in, _Float16 *out);

/* Decomposes an axis of length `n` into pooling factors, each in [2,8], whose product is
 * `n`, in ascending order. Writes up to `cap` factors into `f[]` and returns the count.
 * Returns -1 for an `n` that is not 8-smooth, one with a prime factor > 8. For `n`==1 it
 * returns 0, because
 * that axis needs no pooling. The tests and introspection use it. The function is pure. */
int rocket_reduce_factor_axis(int n, int *f, int cap);

/* ============================================================================
 * Feature-axis reduce: a contraction by a matmul with ones
 * ==========================================================================*/

/*
 * Computes the sum, or the mean, over the hidden or feature axis of each row:
 *
 *     out[m] = sum_h in[m][h]         (mean==0)
 *     out[m] = (1/H) * sum_h in[m][h] (mean==1)
 *
 * PPU pooling reduces the spatial axes [H,W] within one channel and never crosses channels,
 * so the spatial reduce above cannot contract the feature axis. A transformer needs that
 * contraction for the RMSNorm and LayerNorm reduce over the hidden axis, and for the softmax
 * denominator over the sequence axis. The entry computes it as a reduce over the matmul's
 * `K` (contraction) axis:
 *
 *     out[M,1] = in[M,H] · ones[1,H]^T      (each output = sum_h in[m,h]·1)
 *
 * The entry adds no regcmd of its own and calls `rocket_matmul_fp16_f32out`. The K-partials
 * therefore sum in fp32 and fp64, and no tile narrows them to fp16. The main consumer is the
 * RMSNorm and LayerNorm sum of squares, where an fp16-narrowed partial sum would visibly
 * bias the variance. `N` is padded to 16, the matmul's N-group, so 16 identical columns are
 * computed and column 0 is read back.
 *
 * `in` is row-major fp16 [M][H]. `out` is fp32 [M], so that a large sum, such as a sum of
 * squares, does not clip at the fp16 range. The caller must still pass finite fp16 input
 * elements, squaring or scaling upstream. Any `H` > 0 is accepted. The entry zero-pads `H`
 * and `M` internally to the matmul's K%32 and M%4 alignment.
 *
 * With `fd` < 0 it computes the exact host reduction with fp64 accumulation. Returns 0 on
 * success and a negative value on error.
 */
int rocket_reduce_feature_fp16(int fd, int M, int H,
                               const _Float16 *in, float *out, int mean);

/* Computes the CPU reference with fp64 accumulation. The tests compare against it, and it
 * is the `fd` < 0 path. It runs on the host only. */
void rocket_reduce_feature_ref_fp16(int M, int H,
                                    const _Float16 *in, float *out, int mean);

/* ============================================================================
 * Cumsum: a prefix scan by a matmul with triangular ones
 * ==========================================================================*/

/*
 * Computes the prefix sum along the last axis of each row, the TFLite and ONNX CumSum. It is
 * the running-total scan of beam search, CTC, autoregressive masking and RoPE-free
 * positional schemes:
 *
 *     inclusive forward : out[m][j] = sum_{i<=j} in[m][i]
 *     exclusive forward : out[m][j] = sum_{i<j}  in[m][i]
 *     reverse (incl)    : out[m][j] = sum_{i>=j} in[m][i]
 *     reverse (excl)    : out[m][j] = sum_{i>j}  in[m][i]
 *
 * A cumulative sum is a matmul by a triangular ones matrix. The feature reduce above is the
 * special case `N`=1, one all-ones column that sums over all of `K`. Widening that column to
 * the full triangle gives each prefix its own output column:
 *
 *     out[M,N] = in[M,N] · L^T,   L[n][k] = 1 iff k is in prefix n
 *
 * In the matmul's C[m,n] = sum_k A[m,k]·B[n,k] convention, B (=L) is the [N,N] weight.
 * L[n][k] = 1 for k<=n (inclusive), k<n (exclusive), k>=n (reverse incl) and k>n (reverse
 * excl). C[m][n] is then the sum over exactly the input columns of prefix n.
 *
 * The entry adds no regcmd of its own and calls `rocket_matmul_fp16_f32out`, which
 * accumulates `K` in fp32. A long prefix accumulates many terms, and the fp32 accumulator
 * avoids the per-tile fp16 narrowing of the plain fp16 path. The entry zero-pads `N` to the
 * matmul's K%32 and N%16 groups and `M` to %4. The [N,N] triangular weight is N^2 fp16
 * (N=1500 -> 4.5 MB), and the matmul tiles it. The fp32 result is narrowed to fp16 on
 * read-back, so the result is accurate to fp16 rounding (relative tolerance) and not
 * bit-exact.
 *
 * `in` and `out` are row-major fp16 [M][N], and `out` holds the prefix sums. With
 * `exclusive` = 0 position `j` includes `in[j]`, and with 1 it excludes it. With `reverse` = 0
 * the prefix grows left to right, and with 1 the suffix grows right to left. Any `M` and `N`
 * > 0 are accepted, and the entry pads them internally.
 *
 * With `fd` < 0 it computes the exact host scan with fp64 accumulation. Returns 0 on success
 * and a negative value on error.
 */
int rocket_cumsum_fp16(int fd, int M, int N, const _Float16 *in, _Float16 *out,
                       int exclusive, int reverse);

/* Computes the CPU reference with fp64 accumulation. The tests compare against it, and it
 * is the `fd` < 0 path. It runs on the host only. */
void rocket_cumsum_ref_fp16(int M, int N, const _Float16 *in, _Float16 *out,
                            int exclusive, int reverse);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_REDUCE_H */
