// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_REDUCE_H
#define ROCKET_REDUCE_H

#include <stdint.h>

/*
 * rocket_reduce — on-NPU spatial reductions built on the PPU pooling engine.
 *
 * GlobalAveragePool / Mean / ReduceMean over the spatial axes [H,W] (per channel):
 *
 *     out[c] = (1 / (H*W)) * sum_{h,w} in[c][h][w]
 *
 * (TFLite MEAN over axes [1,2] with keepdims, ONNX GlobalAveragePool / ReduceMean,
 * the squeeze of every SE block and most classifier / detection heads.)
 *
 * WHY THIS IS NOT JUST A BIG AVERAGE POOL: the PPU's POOLING_KERNEL_CFG kernel and
 * stride fields are 4-bit (value-1) => a single pool window is capped at 16x16. A
 * global pool over, say, 56x56 cannot run as one pool. So this op DECOMPOSES the global
 * mean into a chain of small non-overlapping average pools and telescopes them:
 *
 *     mean over k1*k2*...*kn equal groups  ==  (...((mean_k1)mean_k2)...)mean_kn
 *
 * because the average of equal-sized group-averages is the grand average. Each pass
 * tiles the current extent EXACTLY (kernel | size, stride = kernel, pad = 0), so the
 * product of the per-pass divisors is exactly H*W. Intermediates stay resident in the
 * NC1HWC2 cube between passes (no host round-trip), input is scattered once and the
 * 1x1xC result de-scattered once.
 *
 * DECOMPOSABILITY: the NPU path needs (a) each axis "16-smooth" — every prime factor
 * is <= 16, so it factors into kernels in [2,16] (covers every power of two and all real
 * feature-map sizes 7,14,28,56,49,98,...) — and (b) H and W to have the SAME number of
 * factors, so every pass pools both axes and neither collapses to 1 ahead of the other.
 * (b) holds for every SQUARE map (H==W, the usual GlobalAvgPool case) and equal-count
 * rectangles. The factors are applied SMALLEST-FIRST so the running quotient stays >= 4
 * at every step — a PPU-WRITTEN intermediate cube with a spatial dim < 4 is NOT read
 * back correctly by the next chained pass (an NPU->NPU chaining quirk; standalone sub-4
 * pools work). A non-16-smooth axis
 * (prime factor 17,19,23,...) or an unequal factor count falls back to an exact host
 * reduction (still returns the correct answer). rocket_global_avgpool_plan() reports
 * which path a shape takes up front.
 *
 * PRECISION: each pass divides by its kernel via the PPU's per-axis fp16 reciprocal
 * fp16(65536/k) (no hardware divider — see npu_pool.h). A symmetric kernel uses the
 * exact validated per-axis reciprocal; an asymmetric kernel (only in the tail passes
 * of a non-square map) splits the divisor geometrically across the two axes. Both keep
 * the product == 1/(kh*kw). Net error is fp16 rounding + ~1e-3 reciprocal quant per
 * pass; gate with a relative tolerance, not equality.
 */

/* ============================================================================
 * SECTION — Spatial reduce: GlobalAveragePool / Mean over [H,W]
 * ==========================================================================*/

/* Largest input H or W this op accepts (cube dim field is 13-bit; well above any real
 * feature map, but bound the factor buffers). */
#define ROCKET_REDUCE_MAX_DIM 8192

/* Returns 0 if (C,H,W) runs entirely on the NPU (both axes 16-smooth, dims in range),
 * else a negative reason. A negative result is NOT an error for the runtime below — it
 * means "host fallback" — but lets a caller (e.g. the TFLite delegate) decide up front.
 * Pure, no hardware. */

#ifdef __cplusplus
extern "C" {
#endif
int rocket_global_avgpool_plan(int C, int H, int W);

/* GlobalAveragePool / spatial Mean over [H,W], per channel.
 *   in  : row-major fp16 [C][H][W]  (channels-major, the conv/pool feature layout)
 *   out : fp16 [C]                  (the per-channel spatial mean)
 * fd is an open rocket device (rocket_open()); fd<0 (or a non-decomposable shape)
 * computes the exact host reduction. Returns 0, negative on error. */
int rocket_global_avgpool_fp16(int fd, int C, int H, int W,
                               const _Float16 *in, _Float16 *out);

/* CPU fp32-accumulate reference (the golden oracle; also the fd<0 / fallback path).
 * Pure host, no hardware. */
void rocket_global_avgpool_ref_fp16(int C, int H, int W,
                                    const _Float16 *in, _Float16 *out);

/* ============================================================================
 * SECTION — Spatial reduce: GlobalMaxPool / GlobalMinPool & factor helper
 * ==========================================================================*/

/*
 * GlobalMaxPool / GlobalMinPool — spatial ReduceMax / ReduceMin over [H,W], per channel:
 *
 *     out[c] = max (or min) over { in[c][h][w] }
 *
 * (ONNX GlobalMaxPool / ReduceMax/ReduceMin over axes [2,3], TFLite REDUCE_MAX/REDUCE_MIN
 * over the spatial axes, global-max-pool classifier/attention heads.)
 *
 * Same telescoping decomposition and decidability as GlobalAveragePool above (the PPU pool
 * window is 16x16-capped) — reuse rocket_global_avgpool_plan() to test a shape — but MAX/MIN
 * are IDEMPOTENT so no per-pass reciprocal is needed (max-of-block-maxes == global max), and
 * the result is EXACT (no fp16 reciprocal rounding): a decomposable shape is bit-exact vs the
 * host reduction. The sub-4 chained-intermediate quirk is method-independent, so the same
 * ascending-factor / equal-count plan keeps every intermediate >= 4. Non-decomposable shape
 * or fd<0 => exact host reduction. Layout/returns as rocket_global_avgpool_fp16. */
int rocket_global_maxpool_fp16(int fd, int C, int H, int W,
                               const _Float16 *in, _Float16 *out);
int rocket_global_minpool_fp16(int fd, int C, int H, int W,
                               const _Float16 *in, _Float16 *out);

/* CPU references (golden oracle; also the fd<0 / fallback path). Pure host. */
void rocket_global_maxpool_ref_fp16(int C, int H, int W,
                                    const _Float16 *in, _Float16 *out);
void rocket_global_minpool_ref_fp16(int C, int H, int W,
                                    const _Float16 *in, _Float16 *out);

/* Decompose an axis of length n into pooling factors, each in [2,16], product == n.
 * Writes up to `cap` factors into f[], returns the count, or -1 if n is not 16-smooth
 * (has a prime factor > 16). n==1 returns 0 (no pooling needed on that axis). Exposed
 * for the gate / introspection. Pure. */
int rocket_reduce_factor_axis(int n, int *f, int cap);

/* ============================================================================
 * SECTION — Feature-axis reduce (matmul-by-ones contraction)
 * ==========================================================================*/

/*
 * FEATURE-AXIS reduce: sum (or mean) over the HIDDEN/feature axis, PER ROW.
 *
 *     out[m] = sum_h in[m][h]         (mean==0)
 *     out[m] = (1/H) * sum_h in[m][h] (mean==1)
 *
 * This is the sibling the PPU spatial reduce above CANNOT provide: PPU pooling reduces
 * the spatial axes [H,W] *within* a channel and never crosses channels, so it cannot
 * contract the feature axis. The contraction the transformer needs — RMSNorm/LayerNorm
 * reduce over hidden, softmax denominator over the sequence axis — is a reduce over the
 * matmul's K (contraction) axis, so we get it for free from the validated matmul:
 *
 *     out[M,1] = in[M,H] · ones[1,H]^T      (each output = sum_h in[m,h]·1)
 *
 * No new regcmd — it reuses rocket_matmul_fp16_f32out, so it inherits that path's
 * GENUINE fp32 accumulation (the K-partials sum in fp32/fp64, not fp16-narrowed per
 * tile). That precision matters: this primitive's main consumer is the RMSNorm/LayerNorm
 * sum-of-squares, where a fp16-narrowed partial sum would visibly bias the variance.
 * (N is padded to 16 — the matmul's N-group — so 16 identical columns are computed and
 * column 0 read back; the redundant columns are a tiny fixed cost on the K-dominated
 * shape, no extra DRAM weight traffic beyond the 16xKpad ones vector.)
 *
 *   in  : row-major fp16 [M][H]
 *   out : fp32 [M]   (fp32 so a large sum, e.g. sum-of-squares, doesn't clip fp16 range;
 *                     the fp16 INPUT elements must still be finite — square/scale upstream)
 * fd<0 computes the exact host (fp64-accumulate) reduction. Any H>0 is accepted (H and M
 * are zero-padded to the matmul's K%32 / M%4 alignment internally). Returns 0, <0 on error.
 */
int rocket_reduce_feature_fp16(int fd, int M, int H,
                               const _Float16 *in, float *out, int mean);

/* CPU fp64-accumulate reference (the golden oracle; also the fd<0 path). Pure host. */
void rocket_reduce_feature_ref_fp16(int M, int H,
                                    const _Float16 *in, float *out, int mean);

/* ============================================================================
 * SECTION — Cumsum (matmul-by-triangular-ones prefix scan)
 * ==========================================================================*/

/*
 * CUMSUM — prefix sum along the last axis, PER ROW (TFLite/ONNX CumSum, the running-total
 * scan in beam search / CTC / autoregressive masking / RoPE-free positional schemes):
 *
 *     inclusive forward : out[m][j] = sum_{i<=j} in[m][i]
 *     exclusive forward : out[m][j] = sum_{i<j}  in[m][i]
 *     reverse (incl)    : out[m][j] = sum_{i>=j} in[m][i]
 *     reverse (excl)    : out[m][j] = sum_{i>j}  in[m][i]
 *
 * KEY LOWERING — a cumulative sum IS a matmul by a TRIANGULAR ones matrix. The feature
 * reduce above is the N=1 all-ones-column special case (sum over ALL of K); widen that
 * single column to the full triangle and every prefix appears as its own output column:
 *
 *     out[M,N] = in[M,N] · L^T,   L[n][k] = 1 iff k is in prefix n
 *
 * In the matmul's C[m,n] = sum_k A[m,k]·B[n,k] convention, B (=L) is the [N,N] weight with
 * L[n][k] = 1 for k<=n (inclusive) / k<n (exclusive) / k>=n (reverse incl) / k>n (reverse
 * excl), so C[m][n] = sum over exactly the prefix-n input columns. NO new regcmd — it reuses
 * rocket_matmul_fp16_f32out (genuine fp32 K-accumulation: a long prefix accumulates many
 * terms, so the fp32 accumulator avoids the per-tile fp16 narrowing the plain fp16 path would
 * apply). N is zero-padded to the matmul's K%32 / N%16 group, M to %4; the [N,N] triangular
 * weight is N^2 fp16 (N=1500 -> 4.5 MB, the matmul tiles it). The fp32 result is narrowed to
 * fp16 on read-back, so this is fp16-rounding-accurate (relative tol), NOT bit-exact.
 *
 *   in  : row-major fp16 [M][N]
 *   out : row-major fp16 [M][N]   (the prefix sums; fp16 narrows the partial sums on read-back)
 *   exclusive : 0 = include in[j] at position j, 1 = exclude it
 *   reverse   : 0 = prefix grows left->right, 1 = suffix grows right->left
 * fd<0 (or any error path) computes the exact fp64-accumulate host scan. Any M,N>0 accepted
 * (padded internally). Returns 0, <0 on error.
 */
int rocket_cumsum_fp16(int fd, int M, int N, const _Float16 *in, _Float16 *out,
                       int exclusive, int reverse);

/* CPU fp64-accumulate reference (the golden oracle; also the fd<0 path). Pure host. */
void rocket_cumsum_ref_fp16(int M, int N, const _Float16 *in, _Float16 *out,
                            int exclusive, int reverse);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_REDUCE_H */
