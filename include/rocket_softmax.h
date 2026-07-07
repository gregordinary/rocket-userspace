// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_SOFTMAX_H
#define ROCKET_SOFTMAX_H

#include <stdint.h>

/*
 * rocket_softmax — row-wise softmax over the last axis, on the NPU. The attention-score
 * nonlinearity of the Whisper/transformer encoder.
 *
 *     out[m][n] = exp(x[m][n] - rowmax[m]) / sum_n exp(x[m][n] - rowmax[m])
 *
 * COMPOSITION (most of softmax is already shipped):
 *   1. rowmax[m] = max_n x[m][n], and xs = x - rowmax        (HOST — see below)
 *   2. e = exp(xs)                                            (NPU  DPU LUT, ROCKET_ACTIVATION_EXP)
 *   3. s[m] = sum_n e[m][n]                                   (NPU  feature-axis ones-matmul reduce, fp32)
 *   4. inv[m] = 1 / s[m]                                      (HOST — O(M), exact)
 *   5. out = e * inv (broadcast over n)                       (NPU  per-row scale = ew_mul)
 *
 * The O(M*N) work (exp, sum, scale) runs on the NPU; the genuinely-new piece vs the shipped
 * primitives is the ROW-MAX. The row-max is on the HOST here:
 *   - matmul/the feature reduce can only SUM, never max (no max-reduce datapath);
 *   - the only on-NPU max is the PPU max-pool, which would need the seq axis laid on the PPU
 *     spatial axis + telescoping for N>16 (the reduce_mean machinery with POOL_METHOD_MAX) —
 *     that is the RESIDENT-fusion path (keeps scores cube-resident, avoids the readback), a
 *     perf follow-on. For the standalone primitive the scores are host-resident, so the host
 *     row-max+subtract is the same O(M*N) memory pass the packing already does — no round-trip.
 * The row-max subtraction is MANDATORY (exp overflows fast; it maps xs into (-∞,0] = EXP's
 * default LUT domain [-16,0], where the deep tail clamps to ~0, correct). inv is computed on
 * the host (exact, O(M)); the reciprocal LUT is the resident-fusion alternative (avoids reading
 * s back) but is approximate and unnecessary when s is already host-side from the reduce.
 *
 * fd<0 computes the exact host reference. Returns 0, <0 on error.
 */

#ifdef __cplusplus
extern "C" {
#endif
int  rocket_softmax_fp16(int fd, int M, int N, const _Float16 *in, _Float16 *out);
void rocket_softmax_ref_fp16(int M, int N, const _Float16 *in, _Float16 *out);

/*
 * Row-wise LogSoftmax over the last axis, on the NPU (the numerically-stable log of softmax,
 * the classification / NLL-loss head and the log-prob output of an LM):
 *
 *     out[m][n] = x[m][n] - rowmax[m] - log( sum_n exp(x[m][n] - rowmax[m]) )
 *               = x[m][n] - logsumexp_n(x[m][n])
 *
 * Same composition as softmax, sharing steps 1-3 (host row-max + subtract -> xs, NPU exp,
 * NPU row-sum), but the tail is a SUBTRACT of a per-row scalar instead of a divide:
 *   4. ls[m] = log(s[m])                                     (HOST — O(M), exact)
 *   5. out[m][n] = xs[m][n] - ls[m]   (broadcast over n)     (NPU  per-row ew_sub)
 *
 * Why the log is on the HOST (not the new ROCKET_ACTIVATION_LOG DPU LUT): ls is over the M
 * per-row sums only — already host-side as fp32 after the reduce read-back — so the host log is
 * EXACT and adds no round-trip, exactly as softmax keeps 1/s on the host (and RMSNorm keeps
 * rsqrt there). The DPU LOG LUT is for a LARGE-tensor element-wise log, not M scalars. s[m] >= 1
 * (the row-max term contributes exp(0)=1) so log(s) >= 0 and out <= 0 everywhere. LogSoftmax is
 * BETTER-conditioned than softmax for a gate — it is all-additive, no tiny-probability blow-up —
 * so check ABSOLUTE error. fd<0 computes the exact host reference. Returns 0, <0 on error.
 */
int  rocket_logsoftmax_fp16(int fd, int M, int N, const _Float16 *in, _Float16 *out);
void rocket_logsoftmax_ref_fp16(int M, int N, const _Float16 *in, _Float16 *out);

/*
 * Stable per-row CROSS-ENTROPY against an integer class target (the softmax-classifier /
 * language-model NLL loss; one scalar loss per row):
 *
 *     CE[m] = -logsoftmax(logits[m])[ target[m] ]
 *           = logsumexp_n(logits[m][n]) - logits[m][target[m]]
 *
 * The numerically-stable form — it NEVER materializes softmax (no divide, no tiny-prob
 * blow-up). Reuses the LogSoftmax front half (the logsumexp reduction) exactly:
 *   1. rowmax[m] = max_n logits[m][n], xs = logits - rowmax        (HOST)
 *   2. e = exp(xs)                                                 (NPU  DPU LUT, EXP)
 *   3. s[m] = sum_n e[m][n]                                        (NPU  feature-axis reduce, fp32)
 *   4. lse[m] = rowmax[m] + log(s[m])                             (HOST — O(M), exact)
 *   5. CE[m] = lse[m] - logits[m][target[m]]                      (HOST gather + subtract)
 *
 * So cross-entropy is the on-NPU logsumexp reduction + a HOST GATHER. The gather is on the
 * host because THERE IS NO HARDWARE GATHER on the RK3588 NPU (no indexed/scatter read path —
 * the same fact the reduce/attention notes record): it is M scalar index-lookups, correct and
 * free, exactly like softmax's host 1/s and LogSoftmax's host log(s). NOT the tail per-row
 * ew_sub of LogSoftmax — only the M target log-probs are needed, so the broadcast subtract is
 * skipped (CE is strictly CHEAPER than LogSoftmax). s[m] >= 1 (the row-max term contributes
 * exp(0)=1) so log(s) >= 0; CE >= 0 always. loss is fp32 (a sum can exceed fp16 range, and a
 * loss is conventionally fp32) — like the feature reduce's fp32 output. CE is all-additive ->
 * gate with ABSOLUTE error (the dominant term is the EXP-LUT bias in log(s), a per-row offset).
 *
 *   logits : row-major fp16 [M][N]   (the class scores)
 *   target : int [M]                 (the true class index per row, 0..N-1)
 *   loss   : fp32 [M]                 (the per-row cross-entropy)
 * fd<0 computes the exact fp64 host reference. Returns 0, <0 on error.
 */
int  rocket_cross_entropy_fp16(int fd, int M, int N,
                               const _Float16 *logits, const int *target, float *loss);
void rocket_cross_entropy_ref_fp16(int M, int N,
                                   const _Float16 *logits, const int *target, float *loss);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_SOFTMAX_H */
