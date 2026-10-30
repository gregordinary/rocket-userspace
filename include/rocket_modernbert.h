// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_MODERNBERT_H
#define ROCKET_MODERNBERT_H

/*
 * Declares a ModernBERT text encoder, resident on the NPU, over a batch of variable-length
 * sequences. An optional stack of torch pre-norm transformer layers (nn.TransformerEncoderLayer
 * with norm_first) runs after it. A decision or classification head on the encoder's hidden
 * states is usually written that way.
 *
 *     x  = LayerNorm(emb)                                  (no bias)
 *     per layer l:
 *       h  = LayerNorm_attn(x), or x at an identity norm  (ModernBERT's layer 0)
 *       q,k,v = h . Wqkv^T ; RoPE(q), RoPE(k) at the layer's inverse frequencies
 *       x += Wo . attention(q, k, v)     global, or sliding: key j visible from i iff |i-j| <= window
 *       h  = LayerNorm_mlp(x)
 *       a,g = h . Wi^T  (a = the first d_ff columns, g = the rest)
 *       x += Wo_mlp . (GELU(a) * g)      exact erf GELU
 *     x  = LayerNorm_final(x) + post_bias
 *     per post layer:  x += Wo.MHA(LN1(x)) + bo ; x += W2.act(W1.LN2(x) + b1) + b2
 *
 * Every projection and both attention matmuls run on the NPU in fp16 with fp32 accumulation.
 * The residual stream, every LayerNorm, RoPE, the softmax, GeGLU and the bias adds run on the
 * host in fp32. The residual stays fp32 because ModernBERT's grows past 29000 in the large
 * checkpoints measured, where an fp16 residual has an ulp of 16.
 *
 * The entry scales an FFN down-projection's input by an exact power of two into fp16's range
 * before the NPU sees it. It scales the output back in fp32. An activation near fp16's limit
 * (35500 measured) therefore neither overflows nor rounds differently. The entry retries a
 * non-finite projection output at a larger scale, and reports one that persists (-5).
 *
 * Sequences are right-padded: row t of sequence b is live iff t < len[b]. Each sequence
 * attends only to its own live rows, so no padding mask is built. Every per-token projection
 * runs once over all live rows of the batch, stacked into one M. Output rows past len[b] are
 * zero.
 *
 * Weights are torch [out, in] fp16 (C = A . W^T). `rocket_modernbert_ctx_create` packs them
 * into resident NPU buffers on an M-independent ctx (ROCKET_CTX_TILING_CANONICAL), so one pack
 * serves every batch and length. The caller must keep the model's arrays alive for the life of
 * the ctx. Every call reads the norms, biases and RoPE frequencies from them, and the host
 * mode reads the weights too.
 *
 * Attention runs through the resident flash-attention fan-out. A sliding layer with
 * L >= 2T + 2 window and L <= 65T runs banded. Its query tiles of T rows (ROCKET_MB_BAND_TILE,
 * default 128, 0 off) each see their T + 2 window keys. The interior tiles of all heads run in
 * one call, and the two edge tiles in one call each. Banding takes 13-14% off a 639- and a
 * 726-token encode.
 *
 * `ROCKET_MB_ATTN=host` runs every attention in fp32 on the host instead. Against the
 * 2026-09-27 library, host attention tied the NPU at 58 tokens and was slower at 639 [HW sweep,
 * RK1, ~816 MHz]. The NPU path's flash-attention host passes are faster in later builds. Not
 * characterized: the crossover in those builds. `ROCKET_MB_PROF=1` prints each encode's phase
 * split.
 *
 * Against ONNX Runtime's fp32 CPU provider on the Laya decision model, the worst layer cosine
 * is 0.9999936 (ModernBERT-large, 639 tokens) and 0.99995 (mmBERT-base, 726 tokens). The
 * post-stack output's cosine is 0.99999995.
 *
 * Evidence: [HW sweep 2026-09-27, RK1, ~816 MHz]. `tests/modernbert_rocket.c` checks it, with
 * taps from every layer.
 */

#include <stddef.h>

#define ROCKET_MB_MAX_LAYERS 64
#define ROCKET_MB_MAX_POST   8

typedef struct {
    const _Float16 *attn_norm_g;  /* [d], or NULL: the attention norm is the identity         */
    const _Float16 *Wqkv;         /* [3d][d], rows q | k | v                                   */
    const _Float16 *Wo;           /* [d][d]                                                     */
    const _Float16 *mlp_norm_g;   /* [d]                                                        */
    const _Float16 *Wi;           /* [2 d_ff][d], rows [0, d_ff) the GELU input, then the gate */
    const _Float16 *Wo_mlp;       /* [d][d_ff]                                                  */
    const float    *inv_freq;     /* [head_dim / 2] RoPE inverse frequencies                    */
    int             sliding;      /* 1: local attention within `window`, 0: global              */
} rocket_mb_layer;

typedef struct {
    const _Float16 *ln1_g, *ln1_b;  /* [d]                        */
    const _Float16 *Wqkv, *bqkv;    /* [3d][d], [3d], rows q|k|v  */
    const _Float16 *Wo, *bo;        /* [d][d], [d]                */
    const _Float16 *ln2_g, *ln2_b;  /* [d]                        */
    const _Float16 *W1, *b1;        /* [post_d_ff][d], [post_d_ff] */
    const _Float16 *W2, *b2;        /* [d][post_d_ff], [d]        */
} rocket_mb_post_layer;

typedef struct {
    int   d, n_layers, n_head, d_ff;    /* d_ff: the GeGLU width (Wi has 2 d_ff rows)         */
    int   window;                       /* sliding half-width (ModernBERT: local_attention/2) */
    float eps;                          /* the encoder's LayerNorm epsilon                    */
    const _Float16 *emb_norm_g;         /* [d]                                                */
    const _Float16 *final_norm_g;       /* [d]                                                */
    rocket_mb_layer layers[ROCKET_MB_MAX_LAYERS];

    int   n_post, post_n_head, post_d_ff;
    int   post_act;                     /* 0: ReLU, 1: exact erf GELU                         */
    float post_eps;
    rocket_mb_post_layer post[ROCKET_MB_MAX_POST];
} rocket_modernbert_model;

typedef struct rocket_mb_ctx rocket_mb_ctx;

#ifdef __cplusplus
extern "C" {
#endif

/* Validates the model and packs its weights. With `nthreads` >= 1, each GEMM and the
 * attention heads spread across that many NPU worker fds (3 matches the RK3588's cores).
 * `nthreads` == 0 is the host mode, which opens no device. In it, every GEMM and attention
 * runs the exact host reference in the same fp16 operand plan, so the glue can be checked
 * off-device. Returns NULL on a pack failure, or on an invalid model (head_dim % 32, d % 32,
 * d_ff % 32, an out-of-range count). */
rocket_mb_ctx *rocket_modernbert_ctx_create(const rocket_modernbert_model *m, int nthreads);
void           rocket_modernbert_ctx_free(rocket_mb_ctx *c);

/*
 * Encodes `B` right-padded sequences of up to `T` tokens. The operands:
 *
 * - `len`: [B] live rows per sequence, 1..T.
 * - `emb`: [B][T][d] fp32 token embeddings, before the embedding LayerNorm.
 * - `post_bias`: [B][d] fp32, added to every row after the final norm, or NULL.
 * - `out`: [B][T][d] fp32, the last post layer's output, zero past len[b]. With `n_post` 0 it
 *   is the final norm's output.
 * - `hidden`: NULL, or [n_layers + 2 + n_post][B][T][d] fp32. Index 0 is the embedding norm,
 *   1..n_layers each encoder layer's output, n_layers + 1 the final norm (before post_bias),
 *   and then each post layer's output.
 *
 * Returns 0 on success. On error it returns -1 for bad arguments, -2 for out of memory and -3
 * for a failed NPU GEMM. It returns -4 for a failed attention and -5 for a non-finite
 * projection output.
 *
 * Caution: a ctx is not thread-safe. Each concurrent caller must use its own ctx.
 */
int rocket_modernbert_encode(rocket_mb_ctx *c, int B, int T, const int *len,
                             const float *emb, const float *post_bias,
                             float *out, float *hidden);

#ifdef __cplusplus
}
#endif
#endif /* ROCKET_MODERNBERT_H */
