// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_MODERNBERT_H
#define ROCKET_MODERNBERT_H

/*
 * rocket_modernbert: a ModernBERT text encoder, resident on the NPU, over a batch of
 * variable-length sequences. An optional stack of torch pre-norm transformer layers
 * (nn.TransformerEncoderLayer with norm_first) runs after it, which is how a decision or
 * classification head built on the encoder's hidden states is usually written.
 *
 *   x  = LayerNorm(emb)                                  (no bias)
 *   per layer l:
 *     h  = LayerNorm_attn(x), or x at an identity norm  (ModernBERT's layer 0)
 *     q,k,v = h . Wqkv^T ; RoPE(q), RoPE(k) at the layer's inverse frequencies
 *     x += Wo . attention(q, k, v)     global, or sliding: key j visible from i iff |i-j| <= window
 *     h  = LayerNorm_mlp(x)
 *     a,g = h . Wi^T  (a = the first d_ff columns, g = the rest)
 *     x += Wo_mlp . (GELU(a) * g)      exact erf GELU
 *   x  = LayerNorm_final(x) + post_bias
 *   per post layer:  x += Wo.MHA(LN1(x)) + bo ; x += W2.act(W1.LN2(x) + b1) + b2
 *
 * Numerics. Every projection and both attention matmuls run on the NPU in fp16 with fp32
 * accumulation. The residual stream, every LayerNorm, RoPE, the softmax, GeGLU and the bias
 * adds run on the host in fp32. The residual stays fp32 because ModernBERT's grows past
 * 29000 in the large checkpoints measured, where an fp16 residual carries an ulp of 16. An
 * MLP down-projection's input is scaled by an exact power of two into fp16's range before
 * the NPU sees it, and the output is scaled back in fp32, so an activation near fp16's
 * limit (35500 measured) neither overflows nor rounds differently. A non-finite projection
 * output is retried at a larger scale, and reported (-5) if it persists.
 *
 * Sequences are right-padded: row t of sequence b is live iff t < len[b]. Each sequence
 * attends only to its own live rows, so no padding mask is built, and every per-token
 * projection runs once over all live rows of the batch stacked into one M. Output rows past
 * len[b] are zero.
 *
 * Weights are torch [out, in] fp16 (C = A . W^T). They are packed into resident NPU buffers
 * at create on an M-independent ctx (ROCKET_CTX_TILING_CANONICAL), so one pack serves every
 * batch and length. The model's arrays must outlive the ctx: the norms, biases and RoPE
 * frequencies are read from them on every call, and the host mode reads the weights too.
 *
 * Attention runs through the resident flash-attention fan-out. A sliding layer at
 * L >= 2T + 2 window is BANDED: query tiles of T rows (ROCKET_MB_BAND_TILE, default 128, 0 off)
 * each see their T + 2 window keys, the interior tiles of all heads in one call and the two
 * edge tiles in one each. That is 13-14% off a 639- and a 726-token encode.
 * ROCKET_MB_ATTN=host runs every attention in fp32 on the host instead: it ties the NPU at 58
 * tokens and loses at 639. ROCKET_MB_PROF=1 prints each encode's phase split.
 *
 * Accuracy, against ONNX Runtime's fp32 CPU provider on the Laya decision model (`tests/
 * modernbert_rocket.c`, taps from every layer): worst layer cosine 0.9999936 (ModernBERT-large,
 * 639 tokens) and 0.99995 (mmBERT-base, 726 tokens), post-stack output 0.99999995
 * [HW 2026-09-27, RK1, 600 MHz].
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
    int             sliding;      /* 1: local attention within `window`; 0: global              */
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
    int   d, n_layers, n_head, d_ff;    /* d_ff: the GeGLU width (Wi carries 2 d_ff rows)     */
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

/* Validate the model and pack its weights. nthreads >= 1 fans each GEMM and the attention
 * heads across that many NPU worker fds (3 matches the RK3588's cores). nthreads == 0 is the
 * host mode: no device is opened, and every GEMM and attention runs the exact host
 * reference in the same fp16 operand plan, so the glue can be checked off-device. Returns
 * NULL on an invalid model (head_dim % 32, d % 32, d_ff % 32, an out-of-range count) or a
 * pack failure. */
rocket_mb_ctx *rocket_modernbert_ctx_create(const rocket_modernbert_model *m, int nthreads);
void           rocket_modernbert_ctx_free(rocket_mb_ctx *c);

/*
 * Encode B right-padded sequences of up to T tokens.
 *   len       [B] live rows per sequence, 1..T
 *   emb       [B][T][d] fp32 token embeddings, before the embedding LayerNorm
 *   post_bias [B][d] fp32 added to every row after the final norm, or NULL
 *   out       [B][T][d] fp32: the last post layer's output (the final norm's when n_post
 *             is 0), zero past len[b]
 *   hidden    NULL, or [n_layers + 2 + n_post][B][T][d] fp32: index 0 the embedding norm,
 *             1..n_layers each encoder layer's output, n_layers + 1 the final norm (before
 *             post_bias), then each post layer's output
 * Not thread-safe: one ctx per concurrent caller. Returns 0; -1 bad arguments, -2 out of
 * memory, -3 an NPU GEMM failed, -4 the attention failed, -5 a non-finite projection output.
 */
int rocket_modernbert_encode(rocket_mb_ctx *c, int B, int T, const int *len,
                             const float *emb, const float *post_bias,
                             float *out, float *hidden);

#ifdef __cplusplus
}
#endif
#endif /* ROCKET_MODERNBERT_H */
