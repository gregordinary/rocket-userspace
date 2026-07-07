// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_ATTN_H
#define ROCKET_ATTN_H

#include <stdint.h>

/*
 * rocket_attn — multi-head self-attention on the NPU (the Whisper/transformer encoder
 * attention sublayer), the composition that exercises the new on-NPU softmax.
 *
 *   q = x·Wq^T + bq ,  k = x·Wk^T + bk ,  v = x·Wv^T + bv          (NPU matmuls)
 *   per head h (d_head = d / n_head, scale = d_head^-0.5):
 *     scores = (q_h · k_h^T) * scale                               (NPU matmul)
 *     P      = softmax(scores)  row-wise over the key axis          (NPU softmax)
 *     ctx_h  = P · v_h                                              (NPU matmul)
 *   out = concat_h(ctx_h)·Wo^T + bo                                 (NPU matmul)
 *
 * This is the PURE attention sublayer (no LayerNorm, no residual — the encoder block wraps it
 * with those). All weights are row-major [out,in] (the rocket_matmul_fp16 C=A·B^T convention,
 * matching a PyTorch nn.Linear weight). Biases are optional (NULL = none; Whisper has bq,bv,bo
 * and no bk). The matmuls + softmax run on the NPU; the head slicing, the d_head^-0.5 scale, the
 * bias adds, and the per-head V transpose (matmul needs B as [N,K]) are O(T*d) host glue.
 *
 * scale is the standard 1/sqrt(d_head) applied to the scores (equivalently d_head^-0.25 on q and
 * k each, the OpenAI-Whisper convention — algebraically identical). T (rows) is padded to a
 * multiple of 4 internally (the matmul M%4 requirement). fd<0 = exact host reference.
 * Returns 0, <0 on error.
 */

#ifdef __cplusplus
extern "C" {
#endif
int  rocket_mha_self_fp16(int fd, int T, int d, int n_head, const _Float16 *x,
                          const _Float16 *Wq, const _Float16 *bq,
                          const _Float16 *Wk, const _Float16 *bk,
                          const _Float16 *Wv, const _Float16 *bv,
                          const _Float16 *Wo, const _Float16 *bo, _Float16 *out);
void rocket_mha_self_ref_fp16(int T, int d, int n_head, const _Float16 *x,
                              const _Float16 *Wq, const _Float16 *bq,
                              const _Float16 *Wk, const _Float16 *bk,
                              const _Float16 *Wv, const _Float16 *bv,
                              const _Float16 *Wo, const _Float16 *bo, _Float16 *out);

/*
 * rocket_flash_attn_fp16 — masked grouped-query attention on the NPU: the decoder
 * (LLM prefill) attention sublayer, matching a ggml GGML_OP_FLASH_ATTN_EXT. The Q/K/V
 * projections happen UPSTREAM (the caller); this computes only the attention, given an
 * additive mask the caller supplies (causal + sliding-window — this code does not
 * generate it):
 *
 *   per query head h (kv head hk = h / (n_head/n_kv_heads), so n_head>n_kv_heads is GQA):
 *     S = scale·(q_h · k_hk^T)            [if softcap: S = softcap·tanh(S/softcap)]
 *     S += mask                            (additive, per (token,key); -inf -> ~0 weight)
 *     P = softmax(S) over the key axis     (NPU softmax)
 *     ctx_h = P · v_hk                      (NPU matmul)
 *
 * head_dim is the KEY/query head dim (the QK contraction). dv is the VALUE/output head
 * dim. They are equal for standard GQA/MHA (pass dv == head_dim); they DIFFER for MLA
 * (Multi-head Latent Attention, e.g. DeepSeek: head_dim = 192 = 128 nope + 64 rope,
 * dv = 128), where the query/key carry a rope-extended dim the value does not.
 *
 * Dense fp16 layouts (head-major, so each head's slice is a contiguous matmul operand):
 *   Q   : [n_head]    [n_tokens][head_dim]    (q_h = Q + h*n_tokens*head_dim)
 *   K   : [n_kv_heads][n_kv]    [head_dim]    (key rows; the QK B-operand [n_kv,head_dim])
 *   V   : [n_kv_heads][dv]      [n_kv]        (v_hk as [dv,n_kv], the AV B-operand)
 *   mask: [n_tokens]  [n_kv]                  (additive fp16; NULL = unmasked)
 *   out : [n_head]    [n_tokens][dv]
 *
 * scale/softcap follow ggml's FLASH_ATTN_EXT op_params (softcap==0 disables the tanh
 * soft-cap; ALiBi/max_bias is not supported — rope models pass 0). Requires
 * n_head % n_kv_heads == 0, head_dim % 32 == 0 (the QK contraction alignment), and
 * dv % 16 == 0 (the AV output alignment). When dv != head_dim (MLA) the materialized
 * per-head path runs; the chained-batch and online-tiled optimizations (which assume a
 * single head dim) engage only for dv == head_dim.
 * n_tokens is padded to %4 and n_kv to %32 internally; pad keys score -inf so they
 * weigh ~0. The scores are brought host-side for the mandatory additive mask, so the
 * softmax defaults to the host (the on-NPU softmax would be a pure round-trip here);
 * ROCKET_ATTN_HOST_SOFTMAX=0 forces the on-NPU softmax for comparison. fd<0 = exact host
 * reference (the test oracle). Returns 0, <0 on error.
 *
 * rocket_flash_attn_fp16_mt is the multi-core fan-out: the heads are independent (each
 * writes its own output slice), so they split into contiguous ranges across `nthreads`
 * worker fds — one drm scheduling entity per fd, so the kernel dispatches the ranges
 * across the NPU cores in parallel (the rocket_matmul_fp16_mt strategy, by head rather
 * than by output column), and each worker runs its own host gather + softmax. Per-head
 * attention is dispatch-bound on this NPU, so single-fd serialization is the v1 bottleneck.
 * `fd` is used only for the host-reference (fd<0) and nthreads==1 paths; the workers open
 * their own fds. nthreads is clamped to [1, min(8, n_head)]. Numerically identical to the
 * single-fd entry. Returns 0, <0 on error.
 *
 * ROCKET_FA_CHAIN (default ON) batches each worker's per-head QK matmuls into a SINGLE
 * NPU job (and, after the softmax, its per-head AV matmuls into a second), instead of one
 * submit + fence per head — one submit + one wait for the whole head range (and one IRQ
 * under ROCKET_BATCH_SUBMIT=1 + the kernel half). On the persistent _ctx path each worker
 * holds resident rocket_mm_batch contexts (one per shape; BOs + score scratch kept
 * resident, prezeroed once), so the batch pays neither a per-call BO alloc nor a full-BO
 * zero — nearly doubling the FA-op throughput vs a per-call batch. This collapses the
 * small-GEMM dispatch floor that made short/mid-context attention lose to the CPU, moving
 * the FA-NPU-vs-CPU prefill crossover from ~6K down to ~2K (parity <=1K); hence default-on.
 * The mask + softmax sit between the two batched jobs, so it is compatible with the
 * host-softmax default. Numerically identical to the per-head path (ROCKET_FA_CHAIN_ELEMS
 * bounds the head group by score-matrix size, default 4M elems; ROCKET_FA_CHAIN=0 forces
 * the per-head path). Applies to the _mt and _ctx paths (per worker) and the nthreads==1 /
 * single-fd path (the whole range).
 *
 * ROCKET_FA_TILE_KV (long context) selects the online/tiled path: instead of one QK matmul
 * over the full key axis (materializing the [n_tokens,n_kv] score matrix host-side for the
 * mask + softmax — 32 MB/head at 32K, traffic that streams to DRAM and grows with n_kv), the
 * key axis is walked in tiles of ROCKET_FA_TILE_KV keys (default 2048, padded to %32) carrying
 * the FlashAttention-2 running softmax (max/denom/unnormalized-output, fp32). The working score
 * tile is [n_tokens,tile] and stays cache-resident, and the resident scratch is bounded — same
 * total MACs and host exp work, traded for more (smaller) per-tile NPU submits, so it only
 * engages above ROCKET_FA_TILE_MIN_KV (default 8192; short/mid context keeps the materialized +
 * chained path, where the small-GEMM dispatch floor dominates). The softmax is host-side on this
 * path (the running rescale is sequential across tiles), compatible with the host-softmax
 * default; a fully-masked tile is skipped per query row (causal / sliding-window correctness).
 * Numerically at least as accurate as the materialized path (fp32 accumulation, same fp16 NPU
 * matmuls). ROCKET_FA_TILE_KV=0 (default) disables it -> the materialized path. Applies to every
 * entry (single-fd, _mt, _ctx) since it lives in the shared per-head range.
 *
 * rocket_fa_ctx is the persistent-context form for repeated calls (an LLM prefill calls the
 * handler once per layer per forward). It holds the worker fds open and keeps each worker's
 * per-head scratch resident (grown to the largest shape seen), so a call pays neither the
 * fd open/close nor the per-call malloc of the 8-16 MB score matrices that, at long context,
 * cross glibc's mmap threshold and mmap+fault+munmap every call. Create once with the worker
 * count (clamped to [1,8]); _ctx is numerically identical to _mt and uses up to
 * min(nthreads, n_head) workers per call. Create returns NULL if a worker fd can't open (the
 * caller falls back to the _mt path). Free after the last call. Returns 0, <0 on error.
 */
typedef struct rocket_fa_ctx rocket_fa_ctx;
rocket_fa_ctx *rocket_fa_ctx_create(int nthreads);
void           rocket_fa_ctx_free(rocket_fa_ctx *c);

int  rocket_flash_attn_fp16(int fd, int n_tokens, int n_kv, int head_dim, int dv,
                            int n_head, int n_kv_heads, float scale, float softcap,
                            const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                            const _Float16 *mask, _Float16 *out);
int  rocket_flash_attn_fp16_mt(int fd, int n_tokens, int n_kv, int head_dim, int dv,
                               int n_head, int n_kv_heads, float scale, float softcap,
                               const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                               const _Float16 *mask, _Float16 *out, int nthreads);
int  rocket_flash_attn_fp16_ctx(rocket_fa_ctx *c, int n_tokens, int n_kv, int head_dim, int dv,
                                int n_head, int n_kv_heads, float scale, float softcap,
                                const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                                const _Float16 *mask, _Float16 *out);
void rocket_flash_attn_ref_fp16(int n_tokens, int n_kv, int head_dim, int dv,
                                int n_head, int n_kv_heads, float scale, float softcap,
                                const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                                const _Float16 *mask, _Float16 *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_ATTN_H */
