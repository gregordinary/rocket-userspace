// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_ENCODER_H
#define ROCKET_ENCODER_H

#include <stdint.h>

/*
 * Declares one Whisper or transformer encoder block on the NPU, assembled from the validated
 * primitives. It is Whisper's pre-norm residual block:
 *
 *     x  = x + MHA( LayerNorm1(x) )                 (attention sublayer)
 *     x  = x + MLP( LayerNorm2(x) )                 (feed-forward sublayer)
 *     MLP(h) = GELU(h·Wf1^T + bf1)·Wf2^T + bf2      (d -> d_ff -> d)
 *
 * `rocket_encoder_block_fp16` runs these on the NPU:
 *
 * - Both LayerNorms.
 * - All attention matmuls and every softmax (`rocket_mha_self_fp16`).
 * - Both residual adds (`rocket_ew_add_fp16`).
 * - The two FFN projection matmuls.
 * - The Φ-gate DPU LUT of the FFN's GELU, the 2-pass GELU(x)=x·Φ(x).
 *
 * By default the FFN runs through `rocket_mlp_fp16_fused`, which keeps the first projection's
 * output in cube layout through the second. That path adds `bf1` on the NPU, as an
 * elementwise add of a bias cube the host scatters. Its GELU multiplies x·Φ(x) on the host
 * unless `ROCKET_ACT_NPU_MUL=1` moves the multiply to the DPU EW unit. A shape it cannot
 * tile-match falls back to a host handoff inside it, which adds `bf1` on the host.
 *
 * `ROCKET_ENCODER_NOFUSE`, set to any value, runs the FFN as separate calls instead, each
 * de-tiling its output to the host. That path adds `bf1` on the host and multiplies x·Φ(x)
 * on the DPU EW unit (`rocket_ew_mul_fp16`). It is unset by default.
 *
 * The single-pass GELU has an error spike in the flat tail, so the block does not use it. The
 * other host work is O(M·d) glue on both paths: head slicing, the attention scale, the
 * attention biases, `bf2` and the residual-buffer trims.
 *
 * All weights are row-major [out,in] (PyTorch nn.Linear). Biases are optional, and NULL means
 * none (Whisper attention has bq, bv and bo, and no bk). With `fd` < 0 it computes the exact
 * host reference. Returns 0 on success and a negative value on error.
 */

#ifdef __cplusplus
extern "C" {
#endif
int  rocket_encoder_block_fp16(int fd, int T, int d, int n_head, int d_ff,
                               const _Float16 *x,
                               const _Float16 *ln1_g, const _Float16 *ln1_b,
                               const _Float16 *Wq, const _Float16 *bq,
                               const _Float16 *Wk, const _Float16 *bk,
                               const _Float16 *Wv, const _Float16 *bv,
                               const _Float16 *Wo, const _Float16 *bo,
                               const _Float16 *ln2_g, const _Float16 *ln2_b,
                               const _Float16 *Wf1, const _Float16 *bf1,
                               const _Float16 *Wf2, const _Float16 *bf2,
                               float eps, _Float16 *out);
void rocket_encoder_block_ref_fp16(int T, int d, int n_head, int d_ff,
                               const _Float16 *x,
                               const _Float16 *ln1_g, const _Float16 *ln1_b,
                               const _Float16 *Wq, const _Float16 *bq,
                               const _Float16 *Wk, const _Float16 *bk,
                               const _Float16 *Wv, const _Float16 *bv,
                               const _Float16 *Wo, const _Float16 *bo,
                               const _Float16 *ln2_g, const _Float16 *ln2_b,
                               const _Float16 *Wf1, const _Float16 *bf1,
                               const _Float16 *Wf2, const _Float16 *bf2,
                               float eps, _Float16 *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_ENCODER_H */
