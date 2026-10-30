// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_NORMVISION_H
#define ROCKET_NORMVISION_H

#include <stdint.h>

/*
 * Declares the vision normalization family on the NPU: BatchNorm, GroupNorm, InstanceNorm and
 * L2-Normalize. They are built on the same primitives as the transformer norms in
 * `rocket_norm.h`. The feature-axis reduce (`rocket_reduce_feature_fp16`) computes the
 * per-group mean and variance, and the DPU elementwise path (`rocket_ew_mul`/`_add`) applies
 * the affine.
 *
 * The cost model is that of `rocket_norm.h`. Like RMSNorm and LayerNorm, these are
 * memory-bound reduce and elementwise ops. Standalone on the NPU they are submit-bound, and a
 * single memory pass on the host is faster. They exist to be fused into a resident NCHW-cube
 * block, so that the activation stays in device layout between two NPU convs or matmuls. The
 * layout round trip they avoid is the dominant cost. They also give op coverage, so a delegate
 * need not move the node to the CPU.
 *
 * All the entries take channels-major NCHW-style buffers, where `P = H*W` is the spatial count
 * per channel (P=1 for a pure [N,C] tensor). The reduce axis is the only thing that
 * distinguishes them:
 *
 * - BatchNorm: no reduction. A per-channel affine from precomputed running stats.
 * - InstanceNorm: a reduce over [H,W] per (n,c), which is GroupNorm with G=C.
 * - GroupNorm: a reduce over [C/G,H,W] per (n,g).
 * - L2-Normalize: a reduce of x^2 over the last axis of each row.
 *
 * The square branch uses the exact power-of-2 prescale of RMSNorm and LayerNorm, because
 * x^2 overflows fp16 for |x|>~223. The reduce accumulates in fp32, which matters for the
 * variance. The per-group mean, var and rsqrt tail is O(rows) exact fp32 scalar work on the
 * host, and the affine narrows in fp16. A test must compare with a relative tolerance, not for
 * equality, because of the fp16 rounding and the reciprocal.
 *
 * With `fd` < 0 each entry computes the exact fp64 host reference, the same value its
 * `_ref_fp16` function returns.
 */

/*
 * Computes inference BatchNorm over an [N,C,P] tensor, per channel, from running statistics:
 *
 *     out[n][c][p] = (x[n][c][p] - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]
 *
 * No reduction runs, because inference BatchNorm uses the stored running mean and var. The
 * entry folds the per-channel affine to out = x * s[c] + b[c], with
 * s[c]=gamma[c]/sqrt(var[c]+eps) and b[c]=beta[c]-mean[c]*s[c]. It then runs one NPU ew_mul
 * and one ew_add over the broadcast tensors.
 *
 * `mean` and `var` are the per-channel running stats [C], and the caller must pass both.
 * `gamma` [C] can be NULL (=> 1), and `beta` [C] can be NULL (=> 0). Returns 0 on success and
 * a negative value on error.
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
 * Computes GroupNorm over an [N,C,P] tensor with G groups, normalizing each (n, group) over
 * its C/G channels and P spatial positions:
 *
 *     mean[n][g], var[n][g]  over { x[n][c][p] : c in group g, all p }
 *     out[n][c][p] = (x[n][c][p] - mean[n][g]) / sqrt(var[n][g] + eps) * gamma[c] + beta[c]
 *
 * `C` must be a multiple of `G` (C % G == 0). A group's elements are contiguous in [N,C,P]
 * (channels-major), so the per-(n,g) reduce is a [N*G, (C/G)*P] feature-axis reduce. It runs
 * the stacked `[x ; x^2]` in one job, like LayerNorm. The affine is per channel, so it varies
 * within a group's row. The entry folds it to out = x*A + B with full broadcast A and B, for
 * one ew_mul and one ew_add. `gamma` and `beta` are [C], and NULL means 1 and 0 respectively.
 *
 * Returns 0 on success and a negative value on error, -1 for G<1 or C%G!=0.
 */
int  rocket_groupnorm_fp16(int fd, int N, int C, int G, int P, const _Float16 *x,
                           const _Float16 *gamma, const _Float16 *beta,
                           float eps, _Float16 *out);
void rocket_groupnorm_ref_fp16(int N, int C, int G, int P, const _Float16 *x,
                               const _Float16 *gamma, const _Float16 *beta,
                               float eps, _Float16 *out);

/*
 * Computes InstanceNorm over an [N,C,P] tensor, normalizing each (n,c) over its P spatial
 * positions. It is exactly GroupNorm with G=C, every channel its own group, and the entry runs
 * `rocket_groupnorm_fp16`. The affine is then per row. `gamma` and `beta` are [C], and NULL
 * means 1 and 0 respectively. `P` must be >= 1. Returns 0 on success and a negative value on
 * error.
 */
int  rocket_instancenorm_fp16(int fd, int N, int C, int P, const _Float16 *x,
                              const _Float16 *gamma, const _Float16 *beta,
                              float eps, _Float16 *out);
void rocket_instancenorm_ref_fp16(int N, int C, int P, const _Float16 *x,
                                  const _Float16 *gamma, const _Float16 *beta,
                                  float eps, _Float16 *out);

/*
 * Computes the L2 normalization of each row of an [M,H] tensor over its H elements, as TFLite
 * L2_NORMALIZATION and ONNX LpNormalization p=2 define it:
 *
 *     out[m][h] = x[m][h] / sqrt( sum_h x[m][h]^2 + eps )
 *
 * The entry runs this chain:
 *
 *     sq = x (x) x (NPU, with the fp16 overflow prescale) -> ss[m]=sum_h sq (NPU fp32 reduce)
 *     -> r[m]=1/sqrt(ss+eps) (host) -> out = scale_rows(x, r) (NPU)
 *
 * `eps` is an additive guard inside the sqrt. Pass a small value, such as 1e-12, to avoid a
 * division by zero on a zero row. Returns 0 on success and a negative value on error.
 */
int  rocket_l2norm_fp16(int fd, int M, int H, const _Float16 *x, float eps, _Float16 *out);
void rocket_l2norm_ref_fp16(int M, int H, const _Float16 *x, float eps, _Float16 *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_NORMVISION_H */
