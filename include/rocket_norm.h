// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_NORM_H
#define ROCKET_NORM_H

#include <stdint.h>

/*
 * Declares transformer normalization on the NPU, RMSNorm and LayerNorm, and the per-row
 * broadcast multiply they use. The entries are built on the feature-axis reduce
 * (`rocket_reduce_feature_fp16`) and the DPU elementwise path (`rocket_ew_mul_fp16`).
 *
 * RMSNorm and LayerNorm are memory-bound elementwise and reduce ops. Standalone on the NPU
 * they are submit-bound: the flat ew_mul tiles ~1020 rows/submit, so a [512,3840] square is
 * ~60 submits. For an isolated norm, the host A76's single memory pass is faster.
 *
 * Between two NPU matmuls, the FFN and attention case, the norm gains from running on the
 * NPU. There the activation can stay in the NC1HWC2 cube. Keeping it there avoids the
 * de-tile -> host -> re-pack layout round trip on every op, which is the dominant cost. These
 * primitives exist to be fused into a resident block, not to run faster than the host
 * standalone.
 *
 * The O(M*H) work (the square, the H-contraction reduce and the final scale) runs on the NPU.
 * The O(M) per-row tail (mean, +eps, rsqrt) is exact fp32 scalar work on the host.
 */

/*
 * Computes the per-row broadcast multiply that RMSNorm and attention need:
 *
 *     out[m][n] = in[m][n] * r[m]      (r is a per-row fp32 scalar, broadcast over n)
 *
 * It is the FFN and attention post-scale. After a matmul, it scales each output row m by its
 * own 1/rms, which folds RMSNorm into the surrounding linear layer. The per-column weight
 * folds into the matmul weight, and the per-row 1/rms folds here. The entry materializes the
 * per-row scalar across the columns and runs the DPU elementwise multiply, so its output
 * matches that path bit-for-bit. `r` is fp32, rounded to fp16 for the multiply.
 *
 * With `fd` < 0 it computes the host result. Returns 0 on success and a negative value on
 * error.
 */

#ifdef __cplusplus
extern "C" {
#endif
int  rocket_scale_rows_fp16(int fd, int M, int N,
                            const _Float16 *in, const float *r, _Float16 *out);
void rocket_scale_rows_ref_fp16(int M, int N,
                                const _Float16 *in, const float *r, _Float16 *out);

/*
 * Computes RMSNorm over the hidden axis of each row:
 *
 *     out[m][h] = x[m][h] / sqrt( mean_h( x[m][h]^2 ) + eps ) * weight[h]
 *
 * `weight` is the effective per-hidden scale [H]. Gemma stores w and uses (1+w), and the
 * caller must pass that pre-incremented, because this op does not add 1. `eps` is the fp32
 * epsilon (Gemma: 1e-6).
 *
 * The entry runs this chain:
 *
 *     sq = x (x) x  (NPU ew_mul) -> ms[m] = mean_h sq  (NPU feature reduce, fp32) ->
 *     r[m] = 1/sqrt(ms[m]+eps)  (host) -> out = x (x) (r (x) weight)  (NPU ew_mul).
 *
 * The fp16 square overflows for |x|>~223 (x^2 > fp16 max). A power-of-2 prescale p=2^-k,
 * applied to x before squaring, handles it exactly, with no rounding. The host recovers the
 * true mean square as ms*4^k.
 *
 * With `fd` < 0 it computes the exact host reference. Returns 0 on success and a negative
 * value on error.
 *
 * Caution: the reciprocal side has no such guard, and cannot have one. The entry combines
 * r[m] = 1/sqrt(ms[m]+eps) with `weight` into a single fp16 scale s[m][h] = r[m]*weight[h] for
 * one ew_mul. A row whose mean square is near zero against a small `eps` makes r large. A
 * large r can push the product past fp16's 65504 -> inf, where the fp64 reference stays
 * finite. The entry assumes the input is away from that edge. A normalization input at
 * |x| ~ 0 for a whole row is a degenerate activation, and this op does not refuse it.
 *
 * Raise `eps` for an input that can reach that edge.
 */
int  rocket_rmsnorm_fp16(int fd, int M, int H, const _Float16 *x,
                         const _Float16 *weight, float eps, _Float16 *out);
void rocket_rmsnorm_ref_fp16(int M, int H, const _Float16 *x,
                             const _Float16 *weight, float eps, _Float16 *out);

/*
 * Computes LayerNorm over the hidden axis of each row, the Whisper and transformer-encoder
 * norm:
 *
 *     mean[m] = mean_h x[m][h]
 *     var[m]  = mean_h x[m][h]^2 - mean[m]^2
 *     out[m][h] = (x[m][h] - mean[m]) / sqrt(var[m] + eps) * gamma[h] + beta[h]
 *
 * `gamma` and `beta` are the per-hidden affine [H], and `beta` can be NULL for no bias. `eps`
 * is the fp32 epsilon (Whisper: 1e-5).
 *
 * Both reductions share one feature-reduce job by stacking the rows. `A = [x ; x(x)x]` (2M
 * rows) runs under the ones weight, so the first M outputs are sum(x) and the next M are
 * sum(x^2). The O(M) per-row tail (mean, var, rsqrt) is exact fp32 scalar work on the host.
 * The affine folds to out = x (x) A + B, with A[m][h]=r[m]*gamma[h] and
 * B[m][h]=beta[h]-mean[m]*r[m]*gamma[h], for one NPU ew_mul and one ew_add. The fp16-square
 * overflow prescale matches RMSNorm.
 *
 * With `fd` < 0 it computes the exact host reference. Returns 0 on success and a negative
 * value on error.
 */
int  rocket_layernorm_fp16(int fd, int M, int H, const _Float16 *x, const _Float16 *gamma,
                           const _Float16 *beta, float eps, _Float16 *out);
void rocket_layernorm_ref_fp16(int M, int H, const _Float16 *x, const _Float16 *gamma,
                               const _Float16 *beta, float eps, _Float16 *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_NORM_H */
