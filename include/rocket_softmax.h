// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_SOFTMAX_H
#define ROCKET_SOFTMAX_H

#include <stdint.h>

/*
 * Declares row-wise softmax, LogSoftmax and cross-entropy over the last axis, on the NPU. The
 * three share one composition: a host row max, the exponential on the DPU LUT, and the row
 * sum on the feature-axis reduce. They differ in the tail that follows the row sum.
 */

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Computes the row-wise softmax over the last axis on the NPU, the attention-score
 * nonlinearity of a Whisper or transformer encoder:
 *
 *     out[m][n] = exp(x[m][n] - rowmax[m]) / sum_n exp(x[m][n] - rowmax[m])
 *
 * The entry composes existing primitives:
 *
 *     1. rowmax[m] = max_n x[m][n], and xs = x - rowmax      (host, see below)
 *     2. e = exp(xs)                                          (NPU  DPU LUT, ROCKET_ACTIVATION_EXP)
 *     3. s[m] = sum_n e[m][n]                                 (NPU  feature-axis ones-matmul reduce, fp32)
 *     4. inv[m] = 1 / s[m]                                    (host, O(M), exact)
 *     5. out = e * inv (broadcast over n)                     (NPU  per-row scale = ew_mul)
 *
 * The O(M*N) work (exp, sum, scale) runs on the NPU. The matmul and the feature reduce can
 * only sum, never take a maximum (no max-reduce datapath), so the row max runs on the host.
 * The scores are host-resident for this standalone entry. The host row max and subtract is
 * then the same O(M*N) memory pass the packing already makes, with no extra round trip.
 *
 * The row-max subtraction is required, because exp overflows fast. It maps xs into (-∞,0],
 * the EXP LUT's default domain [-16,0], where the deep tail clamps to ~0, which is correct.
 * The host computes inv exactly, in O(M).
 *
 * With `fd` < 0 it computes the exact host reference. Returns 0 on success and a negative
 * value on error.
 */
int  rocket_softmax_fp16(int fd, int M, int N, const _Float16 *in, _Float16 *out);
void rocket_softmax_ref_fp16(int M, int N, const _Float16 *in, _Float16 *out);

/*
 * Computes the row-wise LogSoftmax over the last axis on the NPU, the numerically stable log
 * of the softmax. It serves a classification or NLL-loss head and the log-prob output of an
 * LM:
 *
 *     out[m][n] = x[m][n] - rowmax[m] - log( sum_n exp(x[m][n] - rowmax[m]) )
 *               = x[m][n] - logsumexp_n(x[m][n])
 *
 * It shares steps 1-3 of the softmax composition (host row max and subtract -> xs, NPU exp,
 * NPU row sum). The tail subtracts a per-row scalar instead of dividing:
 *
 *     4. ls[m] = log(s[m])                                     (host, O(M), exact)
 *     5. out[m][n] = xs[m][n] - ls[m]   (broadcast over n)     (NPU  per-row ew_sub)
 *
 * The log runs on the host, not on the `ROCKET_ACTIVATION_LOG` DPU LUT, because it covers only
 * the M per-row sums. Those are already on the host in fp32 after the reduce read-back, so the
 * host log is exact and adds no round trip.
 *
 * `s[m]` >= 1, because the row-max term contributes exp(0)=1, so log(s) >= 0 and out <= 0
 * everywhere. LogSoftmax is better conditioned than softmax for a test, because it is
 * all-additive with no tiny-probability blow-up. A test must therefore check absolute error.
 *
 * With `fd` < 0 it computes the exact host reference. Returns 0 on success and a negative
 * value on error.
 */
int  rocket_logsoftmax_fp16(int fd, int M, int N, const _Float16 *in, _Float16 *out);
void rocket_logsoftmax_ref_fp16(int M, int N, const _Float16 *in, _Float16 *out);

/*
 * Computes the numerically stable per-row cross-entropy against an integer class target. It is
 * the NLL loss of a softmax classifier or a language model, one scalar loss per row:
 *
 *     CE[m] = -logsoftmax(logits[m])[ target[m] ]
 *           = logsumexp_n(logits[m][n]) - logits[m][target[m]]
 *
 * The entry never materializes the softmax, so it has no divide and no tiny-probability
 * blow-up. It reuses the LogSoftmax front half, the logsumexp reduction, exactly:
 *
 *     1. rowmax[m] = max_n logits[m][n], xs = logits - rowmax        (host)
 *     2. e = exp(xs)                                                 (NPU  DPU LUT, EXP)
 *     3. s[m] = sum_n e[m][n]                                        (NPU  feature-axis reduce, fp32)
 *     4. lse[m] = rowmax[m] + log(s[m])                              (host, O(M), exact)
 *     5. CE[m] = lse[m] - logits[m][target[m]]                       (host gather + subtract)
 *
 * Cross-entropy is thus the on-NPU logsumexp reduction plus a host gather. The gather runs on
 * the host because the RK3588 NPU has no hardware gather (no indexed or scatter read path). It
 * is M scalar index lookups, correct and free, like softmax's host 1/s and LogSoftmax's host
 * log(s). The entry needs only the M target log-probs, so it skips the per-row ew_sub tail of
 * LogSoftmax. CE therefore costs strictly less than LogSoftmax.
 *
 * `s[m]` >= 1, because the row-max term contributes exp(0)=1, so log(s) >= 0 and CE >= 0
 * always. CE is all-additive, so a test must check absolute error. The dominant error term is
 * the EXP-LUT bias in log(s), a per-row offset.
 *
 * `logits` is row-major fp16 [M][N], the class scores. `target` is int [M], the true class
 * index of each row (0..N-1). `loss` is fp32 [M], the per-row cross-entropy. The loss is fp32
 * like the feature reduce's fp32 output, because a sum can exceed the fp16 range and a loss is
 * conventionally fp32.
 *
 * With `fd` < 0 it computes the exact fp64 host reference. Returns 0 on success and a negative
 * value on error.
 */
int  rocket_cross_entropy_fp16(int fd, int M, int N,
                               const _Float16 *logits, const int *target, float *loss);
void rocket_cross_entropy_ref_fp16(int M, int N,
                                   const _Float16 *logits, const int *target, float *loss);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_SOFTMAX_H */
