// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_NORM_H
#define ROCKET_NORM_H

#include <stdint.h>

/*
 * rocket_norm — on-NPU transformer normalization, built on the feature-axis reduce
 * (rocket_reduce_feature_fp16) + the DPU elementwise path (rocket_ew_mul_fp16).
 *
 * THE COST MODEL (read this before "optimizing"). RMSNorm/LayerNorm are memory-bound
 * elementwise+reduce ops. Standalone on the NPU they are submit-bound (the flat ew_mul
 * tiles ~1020 rows/submit, so a [512,3840] square is ~60 submits) — for an ISOLATED norm
 * the host A76 single memory pass is faster. The win is COMPOSITIONAL: when the norm sits
 * between two NPU matmuls (the FFN/attention case), running it on-device lets the activation
 * stay in the NC1HWC2 cube, avoiding the de-tile -> host -> re-pack layout round-trip on
 * every op (the dominant not-mac-bound cost). So these primitives exist to be FUSED into a
 * resident block, not to beat the host standalone.
 *
 * THE WORK SPLIT. The O(M*H) work (square, the H-contraction reduce, the final scale) runs
 * on the NPU; the O(M) per-row tail (mean, +eps, rsqrt) is exact fp32 host scalar work.
 * Sending M tiny per-row scalars back to the NPU for an APPROXIMATE LUT rsqrt would add a
 * round-trip AND hit the rsqrt-LUT domain problem (ms can span many decades across rows /
 * layers; a uniform-grid LUT can't cover that) for no benefit — the DPU rsqrt LUT stays
 * reserved for large-TENSOR rsqrt, not the M-vector here.
 */

/*
 * Per-row broadcast multiply (the reusable primitive RMSNorm/attention need):
 *
 *     out[m][n] = in[m][n] * r[m]      (r is a per-row fp32 scalar, broadcast over n)
 *
 * This is the FFN/attention post-scale: after a matmul, scale each output row m by its own
 * 1/rms (RMSNorm folded into the surrounding linear — the per-column weight folds into the
 * matmul weight, the per-row 1/rms folds here). Implemented by materializing the per-row
 * scalar across the columns and reusing the DPU elementwise multiply, so it inherits that
 * path bit-for-bit. (In-model this folds further into the matmul's activation pack — the
 * scatter already touches every element — which is the optimization target; here it is a
 * standalone gate-grade op.) `r` is fp32 (rounded to fp16 for the multiply). fd<0 = host.
 * Returns 0, <0 on error.
 */

#ifdef __cplusplus
extern "C" {
#endif
int  rocket_scale_rows_fp16(int fd, int M, int N,
                            const _Float16 *in, const float *r, _Float16 *out);
void rocket_scale_rows_ref_fp16(int M, int N,
                                const _Float16 *in, const float *r, _Float16 *out);

/*
 * RMSNorm over the hidden axis, per row:
 *
 *     out[m][h] = x[m][h] / sqrt( mean_h( x[m][h]^2 ) + eps ) * weight[h]
 *
 * `weight` is the EFFECTIVE per-hidden scale [H] (Gemma stores w and uses (1+w); pass that
 * pre-incremented — this op does not add 1). `eps` is the fp32 epsilon (Gemma: 1e-6).
 *
 * Chain: sq = x (x) x  (NPU ew_mul) -> ms[m] = mean_h sq  (NPU feature reduce, fp32) ->
 * r[m] = 1/sqrt(ms[m]+eps)  (host) -> out = x (x) (r (x) weight)  (NPU ew_mul).
 * fp16-square OVERFLOW (|x|>~223 => x^2 > fp16 max) is handled by a power-of-2 prescale
 * p=2^-k applied to x before squaring (exact, no rounding); the true mean-square is recovered
 * on the host as ms*4^k. fd<0 computes the exact host reference. Returns 0, <0 on error.
 */
int  rocket_rmsnorm_fp16(int fd, int M, int H, const _Float16 *x,
                         const _Float16 *weight, float eps, _Float16 *out);
void rocket_rmsnorm_ref_fp16(int M, int H, const _Float16 *x,
                             const _Float16 *weight, float eps, _Float16 *out);

/*
 * LayerNorm over the hidden axis, per row (the Whisper/transformer-encoder norm):
 *
 *     mean[m] = mean_h x[m][h]
 *     var[m]  = mean_h x[m][h]^2 - mean[m]^2
 *     out[m][h] = (x[m][h] - mean[m]) / sqrt(var[m] + eps) * gamma[h] + beta[h]
 *
 * `gamma`/`beta` are the per-hidden affine [H] (beta may be NULL for no bias). `eps` is the
 * fp32 epsilon (Whisper: 1e-5). Chain: BOTH reductions share ONE feature-reduce job by
 * STACKING the rows — A = [x ; x(x)x] (2M rows) under the ones weight, so the first M outputs
 * are sum(x) and the next M are sum(x^2). The
 * O(M) per-row tail (mean, var, rsqrt) is exact fp32 host scalar work; the affine is folded to
 * out = x (x) A + B with A[m][h]=r[m]*gamma[h], B[m][h]=beta[h]-mean[m]*r[m]*gamma[h] (one NPU
 * ew_mul + one ew_add). The fp16-square overflow prescale matches RMSNorm. fd<0 = exact host
 * reference. Returns 0, <0 on error.
 */
int  rocket_layernorm_fp16(int fd, int M, int H, const _Float16 *x, const _Float16 *gamma,
                           const _Float16 *beta, float eps, _Float16 *out);
void rocket_layernorm_ref_fp16(int M, int H, const _Float16 *x, const _Float16 *gamma,
                               const _Float16 *beta, float eps, _Float16 *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_NORM_H */
