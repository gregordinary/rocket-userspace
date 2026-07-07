// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_ENCODER_H
#define ROCKET_ENCODER_H

#include <stdint.h>

/*
 * rocket_encoder — one Whisper/transformer encoder block on the NPU (the encoder capstone),
 * assembled from the validated primitives. Whisper's PRE-norm residual block:
 *
 *     x  = x + MHA( LayerNorm1(x) )                 (attention sublayer)
 *     x  = x + MLP( LayerNorm2(x) )                 (feed-forward sublayer)
 *     MLP(h) = GELU(h·Wf1^T + bf1)·Wf2^T + bf2      (d -> d_ff -> d)
 *
 * FULLY on the NPU: both LayerNorms, all attention matmuls + every softmax (rocket_mha_self_fp16),
 * both residual adds (rocket_ew_add_fp16), the two MLP projection matmuls, AND the MLP's GELU —
 * computed as the 2-pass GELU(x)=x·Φ(x) (the Φ-gate DPU LUT + the DPU EW-mul; the single-pass
 * GELU spikes in the flat tail, so it is not used). The only host work is O(M·d) glue (head
 * slicing, scale, bias adds, residual-buffer trims). All weights are row-major [out,in] (PyTorch
 * nn.Linear). Biases optional (NULL = none; Whisper attn has bq,bv,bo and no bk). fd<0 = exact
 * host reference. Returns 0, <0 on error.
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
