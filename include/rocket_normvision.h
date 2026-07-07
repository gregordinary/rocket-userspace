// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_NORMVISION_H
#define ROCKET_NORMVISION_H

#include <stdint.h>

/*
 * rocket_normvision — the on-NPU VISION normalization family (BatchNorm / GroupNorm /
 * InstanceNorm / L2-Normalize), built on the SAME primitives as the transformer norms in
 * rocket_norm.h: the feature-axis reduce (rocket_reduce_feature_fp16) for the per-group
 * mean/variance and the DPU elementwise path (rocket_ew_mul/_add) for the affine. See also the vision-norm family. /
 * op-coverage.
 *
 * Cost model (read rocket_norm.h first). Like RMSNorm/LayerNorm these are memory-bound
 * reduce+elementwise ops: standalone on the NPU they are submit-bound and a host single
 * memory pass is faster. They exist to be FUSED into a resident NCHW-cube block (so the
 * activation never leaves device layout between two NPU convs/matmuls — the dominant
 * not-mac-bound cost), and as op-coverage so a delegate need not spill the node to CPU.
 *
 * Layout. All take channels-major NCHW-style buffers, `P = H*W` = the spatial count per
 * channel (P=1 for a pure [N,C] tensor). The reduce axis differs per op (that is the only
 * thing that distinguishes them):
 *   - BatchNorm     : NO reduction — per-channel affine from PRECOMPUTED running stats.
 *   - InstanceNorm  : reduce over [H,W] per (n,c)        == GroupNorm with G=C.
 *   - GroupNorm     : reduce over [C/G,H,W] per (n,g).
 *   - L2-Normalize  : reduce x^2 over the last axis per row.
 *
 * Precision. The square branch uses the exact power-of-2 prescale from RMSNorm/LayerNorm
 * (|x|>~223 => x^2 overflows fp16); the reduce accumulates in fp32 (genuine, matters for the
 * variance); the per-group mean/var/rsqrt O(rows) tail is exact fp32 host scalar work; the
 * affine narrows in fp16. Gate with a RELATIVE tolerance, not equality (fp16 rounding +
 * reciprocal). fd<0 (or any error-free degenerate shape) computes the exact fp64 host
 * reference — the same value the `_ref_fp16` oracle returns.
 */

/*
 * BatchNorm (INFERENCE) over an [N,C,P] tensor, per channel, from running statistics:
 *
 *     out[n][c][p] = (x[n][c][p] - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]
 *
 * No reduction happens here (inference BN uses the stored running mean/var) — it is a
 * per-channel affine folded to out = x * s[c] + b[c] with s[c]=gamma[c]/sqrt(var[c]+eps),
 * b[c]=beta[c]-mean[c]*s[c], then one NPU ew_mul + one ew_add over the broadcast tensors.
 * `mean`/`var` are the per-channel running stats [C] (required). `gamma` [C] may be NULL
 * (=> 1), `beta` [C] may be NULL (=> 0). Returns 0, <0 on error.
 */

#ifdef __cplusplus
extern "C" {
#endif
int  rocket_batchnorm_fp16(int fd, int N, int C, int P, const _Float16 *x,
                           const _Float16 *gamma, const _Float16 *beta,
                           const _Float16 *mean, const _Float16 *var,
                           float eps, _Float16 *out);
void rocket_batchnorm_ref_fp16(int N, int C, int P, const _Float16 *x,
                               const _Float16 *gamma, const _Float16 *beta,
                               const _Float16 *mean, const _Float16 *var,
                               float eps, _Float16 *out);

/*
 * GroupNorm over an [N,C,P] tensor with G groups (C % G == 0), normalizing each
 * (n, group) over its C/G channels and P spatial positions:
 *
 *     mean[n][g], var[n][g]  over { x[n][c][p] : c in group g, all p }
 *     out[n][c][p] = (x[n][c][p] - mean[n][g]) / sqrt(var[n][g] + eps) * gamma[c] + beta[c]
 *
 * A group's elements are contiguous in [N,C,P] (channels-major), so the per-(n,g) reduce is
 * a [N*G, (C/G)*P] feature-axis reduce (stacked [x ; x^2] in ONE job, like LayerNorm). The
 * affine is per-CHANNEL so it varies within a group's row => folded to out = x*A + B with
 * full broadcast A/B (one ew_mul + one ew_add). `gamma`/`beta` are [C] (NULL => 1 / 0).
 * Returns 0, <0 on error (G<1 or C%G!=0 => -1).
 */
int  rocket_groupnorm_fp16(int fd, int N, int C, int G, int P, const _Float16 *x,
                           const _Float16 *gamma, const _Float16 *beta,
                           float eps, _Float16 *out);
void rocket_groupnorm_ref_fp16(int N, int C, int G, int P, const _Float16 *x,
                               const _Float16 *gamma, const _Float16 *beta,
                               float eps, _Float16 *out);

/*
 * InstanceNorm over an [N,C,P] tensor: normalize each (n,c) over its P spatial positions.
 * Exactly GroupNorm with G=C (every channel its own group); the affine is then per-row, so
 * this is the cheaper per-row-scalar path. `gamma`/`beta` are [C] (NULL => 1 / 0). P>=1.
 * Returns 0, <0 on error.
 */
int  rocket_instancenorm_fp16(int fd, int N, int C, int P, const _Float16 *x,
                              const _Float16 *gamma, const _Float16 *beta,
                              float eps, _Float16 *out);
void rocket_instancenorm_ref_fp16(int N, int C, int P, const _Float16 *x,
                                  const _Float16 *gamma, const _Float16 *beta,
                                  float eps, _Float16 *out);

/*
 * L2-Normalize each row of an [M,H] tensor over its H elements (TFLite L2_NORMALIZATION,
 * ONNX LpNormalization p=2):
 *
 *     out[m][h] = x[m][h] / sqrt( sum_h x[m][h]^2 + eps )
 *
 * Chain: sq = x (x) x (NPU, with the fp16 overflow prescale) -> ss[m]=sum_h sq (NPU fp32
 * reduce) -> r[m]=1/sqrt(ss+eps) (host) -> out = scale_rows(x, r) (NPU). `eps` is an additive
 * guard inside the sqrt (pass a small value, e.g. 1e-12, to avoid div-by-zero on a zero row).
 * Returns 0, <0 on error.
 */
int  rocket_l2norm_fp16(int fd, int M, int H, const _Float16 *x, float eps, _Float16 *out);
void rocket_l2norm_ref_fp16(int M, int H, const _Float16 *x, float eps, _Float16 *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_NORMVISION_H */
