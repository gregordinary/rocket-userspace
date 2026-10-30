// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_ATTN_H
#define ROCKET_ATTN_H

#include <stdint.h>

/*
 * Declares attention on the NPU in two forms. `rocket_mha_self_fp16` computes the whole
 * multi-head self-attention sublayer of a transformer encoder, projections included. The
 * flash-attention entries compute masked grouped-query attention for an LLM prefill from
 * projections the caller supplies. They come in three forms: one fd, a multi-core fan-out,
 * and a persistent context for repeated calls.
 */

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Computes multi-head self-attention on the NPU, the attention sublayer of a Whisper or
 * transformer encoder. It composes the NPU matmul and the on-NPU softmax:
 *
 *     q = x·Wq^T + bq ,  k = x·Wk^T + bk ,  v = x·Wv^T + bv          (NPU matmuls)
 *     per head h (d_head = d / n_head, scale = d_head^-0.5):
 *       scores = (q_h · k_h^T) * scale                               (NPU matmul)
 *       P      = softmax(scores)  row-wise over the key axis          (NPU softmax)
 *       ctx_h  = P · v_h                                              (NPU matmul)
 *     out = concat_h(ctx_h)·Wo^T + bo                                 (NPU matmul)
 *
 * The entry computes the attention sublayer alone, with no LayerNorm and no residual. The
 * encoder block adds those around it. All weights are row-major [out,in], the
 * `rocket_matmul_fp16` C=A·B^T convention, which matches a PyTorch nn.Linear weight. Biases
 * are optional, and NULL means none (Whisper has bq, bv and bo, and no bk).
 *
 * The matmuls and the softmax run on the NPU by default. `ROCKET_ATTN_HOST_SOFTMAX=1` runs the
 * softmax on the host, which skips the softmax's score transfers to and from the NPU. The
 * variable reads the same on every attention path: a nonzero value selects the host and 0 the
 * NPU.
 *
 * The head slicing, the d_head^-0.5 scale, the bias adds and the per-head V transpose (the
 * matmul needs B as [N,K]) run on the host, at O(T*d) cost. The scale is the standard 1/sqrt(d_head) applied to the scores. Applying
 * d_head^-0.25 to q and to k, the OpenAI-Whisper convention, is algebraically identical. The
 * entry pads `T` (rows) to a multiple of 4 internally, the matmul's M%4 requirement.
 *
 * With `fd` < 0 it computes the exact host reference. Returns 0 on success and a negative
 * value on error.
 */
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
 * Creates and frees `rocket_fa_ctx`, the persistent context for
 * `rocket_flash_attn_fp16_ctx`, the form for repeated calls. An LLM prefill calls the
 * attention once per layer per forward pass.
 *
 * The context holds the worker fds open and keeps each worker's per-head scratch resident,
 * grown to the largest shape seen. A call then needs neither the fd open and close nor the
 * per-call allocation of the 8-16 MB score matrices. The chained path keeps its resident
 * batched-matmul contexts per op and per 4x class of Tp x Kn. A context syncs its whole grown
 * BO, and clears it on a layout change. One context per class keeps a short call after a long
 * one from syncing the long call's buffers.
 *
 * Create the context once with the worker count, which is clamped to [1,8], and free it after
 * the last call. When a worker fd cannot open, `rocket_fa_ctx_create` returns NULL, and the
 * caller falls back to the `_mt` path.
 */
typedef struct rocket_fa_ctx rocket_fa_ctx;
rocket_fa_ctx *rocket_fa_ctx_create(int nthreads);
void           rocket_fa_ctx_free(rocket_fa_ctx *c);

/*
 * Computes masked grouped-query attention on the NPU, the decoder (LLM prefill) attention
 * sublayer, matching a ggml `GGML_OP_FLASH_ATTN_EXT`. The caller computes the Q, K and V
 * projections upstream and supplies the additive mask (causal and sliding-window). The entry
 * computes only the attention and does not generate the mask:
 *
 *     per query head h (kv head hk = h / (n_head/n_kv_heads), so n_head>n_kv_heads is GQA):
 *       S = scale·(q_h · k_hk^T)            [if softcap: S = softcap·tanh(S/softcap)]
 *       S += mask                            (additive, per (token,key); -inf -> ~0 weight)
 *       P = softmax(S) over the key axis     (NPU softmax)
 *       ctx_h = P · v_hk                      (NPU matmul)
 *
 * `head_dim` is the key and query head dimension, the QK contraction. `dv` is the value and
 * output head dimension. They are equal for standard GQA and MHA, where the caller passes
 * `dv` == `head_dim`. They differ for MLA (Multi-head Latent Attention, e.g. DeepSeek:
 * head_dim = 192 = 128 nope + 64 rope, dv = 128). There the query and key have a
 * rope-extended dimension the value does not.
 *
 * The operands are dense fp16 and head-major, so each head's slice is a contiguous matmul
 * operand:
 *
 *     Q   : [n_head]    [n_tokens][head_dim]    (q_h = Q + h*n_tokens*head_dim)
 *     K   : [n_kv_heads][n_kv]    [head_dim]    (key rows; the QK B-operand [n_kv,head_dim])
 *     V   : [n_kv_heads][dv]      [n_kv]        (v_hk as [dv,n_kv], the AV B-operand)
 *     mask: [n_tokens]  [n_kv]                  (additive fp16; NULL = unmasked)
 *     out : [n_head]    [n_tokens][dv]
 *
 * `scale` and `softcap` follow ggml's FLASH_ATTN_EXT op_params, and `softcap`==0 disables the
 * tanh soft-cap. ALiBi (`max_bias`) is not supported, and rope models pass 0. `n_head` must be
 * a multiple of `n_kv_heads`. `head_dim` must be a multiple of 32, the QK contraction
 * alignment, and `dv` a multiple of 16, the AV output alignment. The entry pads `n_tokens` to
 * %4 and `n_kv` to %32 internally, and pad keys score -inf so they weigh ~0.
 *
 * With `fd` < 0 it computes the exact host reference, which the tests compare against.
 * Returns 0 on success, `ROCKET_E_RANGE` for a real key's score that reached the fp16 limit
 * (see the Caution below), and a negative value on another error.
 *
 * With `dv` != `head_dim` (MLA), the materialized per-head path runs. The chained-batch and
 * online-tiled paths assume a single head dimension and run only for `dv` == `head_dim`. The
 * scores come to the host for the mandatory additive mask, so the softmax runs on the host by
 * default. The on-NPU softmax would be a pure round trip here. `ROCKET_ATTN_HOST_SOFTMAX=0`
 * runs it on the NPU, and a nonzero value keeps the host default.
 *
 * On the materialized paths, `ROCKET_FA_FUSED_SOFTMAX` (default on) runs the host scale,
 * soft-cap, mask and softmax as one fp32 pass per row. The pass uses a vectorized exp and a
 * single fp16 rounding, and its error is equal or lower on every case of
 * `tests/flash_attn_rocket.c`. It takes 19% off a 639-token ModernBERT-large encoder on its
 * generic vector form [HW sweep 2026-09-27, RK1]. `ROCKET_FA_FUSED_SOFTMAX=0` restores the
 * separate mask pass and the two-rounding softmax.
 *
 * On aarch64 the fused pass runs its score and exp passes in NEON. Its output is
 * bit-identical to the generic form, at 0.635-0.652x the generic form's time (one A76,
 * 2026-10-10). Not characterized: the ModernBERT-large figure on the NEON form.
 *
 * `ROCKET_FA_CHAIN` (default on) batches each worker's per-head QK matmuls into one NPU job,
 * and after the softmax its per-head AV matmuls into a second. The whole head range then
 * takes one submit and one wait, instead of one submit and fence per head. Under
 * `ROCKET_BATCH_SUBMIT=1` with the kernel's batched-submit support it also takes one IRQ. On
 * the persistent `_ctx` path each worker holds resident `rocket_mm_batch` contexts. A batch
 * then needs neither a per-call BO allocation nor a full-BO zero, which nearly doubles the
 * FA-op throughput against a per-call batch.
 *
 * Chaining collapses the fixed small-GEMM dispatch overhead that made short and mid-context
 * attention slower on the NPU than on the CPU. It moves the FA-NPU-vs-CPU prefill crossover
 * from ~6K down to ~2K (parity <=1K), and so it is the default. The mask and softmax run
 * between the two batched jobs, so chaining is compatible with the host-softmax default. The
 * result is numerically identical to the per-head path. `ROCKET_FA_CHAIN_ELEMS` bounds the
 * head group by score-matrix size (default 32M elems), and `ROCKET_FA_CHAIN=0` forces the
 * per-head path.
 *
 * Chaining applies to the `_mt` and `_ctx` paths per worker, and to the `nthreads`==1 and
 * single-fd path over the whole range. The group size is an upper bound. When the driver
 * refuses the group's single score buffer, the entry halves the group and retries, down to
 * the per-head path. The result is unchanged, and only the submit count changes.
 *
 * `ROCKET_FA_ROWMAJOR` (default on) hands the scores and P between the chained QK and AV jobs
 * as rows, on the `_ctx` path with the fused host softmax. QK writes the scores into a
 * resident BO through the row-major matmul form, and the softmax writes P as plain rows. AV
 * reads its K slices of P as rows. Neither the de-tile of the scores out of the NPU's cube
 * layout nor the pack of P back into it runs.
 *
 * While a row of scores is at most `ROCKET_FA_RM_MAX_ROW` bytes (default 4096), the softmax
 * rewrites it as P in place. Past that, QK writes each of its `N` tiles as its own block of
 * rows at the tile's width. A wider row-major write costs the device more. P then goes to a
 * second BO. `ROCKET_FA_ROWMAJOR=0` keeps the cube handoff.
 *
 * The row handoff is bit-identical to the cube handoff, with one exception. A lone head in a
 * chained group keeps the batch's fp32 K-accumulation for AV. The cube handoff runs that head
 * as a one-shot matmul, which accumulates in fp16 on the device by default (a few fp16 ulps).
 * On the RK1 at ~816 MHz, the call measured 0.64-0.69x the cube handoff's time at a 1370-token
 * ViT layer [HW sweep 2026-10-11]. In the same sweep it measured 0.73-0.82x at a 512-query LLM
 * prefill shape from 2048 to 32768 keys.
 *
 * `ROCKET_FA_TILE_KV` selects the online, tiled path for long context. The materialized path
 * runs one QK matmul over the full key axis. It holds the [n_tokens,n_kv] score matrix on the
 * host for the mask and softmax. That matrix is 32 MB/head at 32K, and its traffic streams to
 * DRAM and grows with `n_kv`. The tiled path walks the key axis in tiles of `ROCKET_FA_TILE_KV`
 * keys (padded to %32). It keeps the FlashAttention-2 running softmax (max, denominator and
 * unnormalized output, in fp32).
 *
 * The working score tile is [n_tokens,tile] and stays cache-resident, and the resident scratch
 * is bounded. The total MACs and host exp work are the same, traded for more and smaller
 * per-tile NPU submits. The tiled path therefore runs only above `ROCKET_FA_TILE_MIN_KV`
 * (default 8192). Short and mid context keep the materialized and chained path, where the
 * fixed small-GEMM dispatch overhead dominates. `ROCKET_FA_TILE_KV=0` (default) disables it and
 * keeps the materialized path.
 *
 * The softmax runs on the host on the tiled path, because the running rescale is sequential
 * across tiles, which is compatible with the host-softmax default. A fully masked tile is
 * skipped per query row, for causal and sliding-window correctness. The tiled path is at least
 * as accurate as the materialized path, with fp32 accumulation and the same fp16 NPU matmuls.
 * It applies to every entry (single-fd, `_mt`, `_ctx`), because it acts in the shared
 * per-head range.
 *
 * Caution: the scores are fp16, and they saturate. A score past 65504 (scale·q·k, before the
 * soft-cap) is written as ±65504, not as inf. The keys past it tie. The softmax then returns a
 * complete, finite output of the right size with wrong values [HW sweep 2026-10-09, RK3588
 * at ~816 MHz]. Every path (fused, separate mask pass, on-NPU softmax, online-tiled) checks the raw
 * scores in its host mask pass. When a real key reached the limit, the entry returns
 * `ROCKET_E_RANGE` and `out` is not valid.
 *
 * A caller that holds the operands at higher precision can then recompute from them. Folding
 * the scale into Q before the call moves the limit from a raw q·k of 65504 to a scaled one.
 * The same contract holds for `rocket_flash_attn_fp16_mt` and `rocket_flash_attn_fp16_ctx`.
 */
int  rocket_flash_attn_fp16(int fd, int n_tokens, int n_kv, int head_dim, int dv,
                            int n_head, int n_kv_heads, float scale, float softcap,
                            const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                            const _Float16 *mask, _Float16 *out);
/*
 * Computes the same attention as `rocket_flash_attn_fp16`, with the heads spread across the
 * NPU cores. The heads are independent, and each writes its own output slice. The entry splits
 * them into contiguous ranges across `nthreads` worker fds, and each worker runs its own host
 * gather and softmax. The `rocket` driver makes one DRM scheduling entity per fd, so the
 * kernel dispatches the ranges across the NPU cores in parallel. The strategy is that of
 * `rocket_matmul_fp16_mt`, by head rather than by output column.
 *
 * Per-head attention is dispatch-bound on this NPU, so on one fd its serialization is the
 * bottleneck. `fd` is used only on the host-reference (`fd` < 0) and `nthreads`==1 paths, and
 * the workers open their own fds. `nthreads` is clamped to [1, min(8, n_head)]. The result is
 * numerically identical to the single-fd entry. Returns 0 on success and a negative value on
 * error.
 */
int  rocket_flash_attn_fp16_mt(int fd, int n_tokens, int n_kv, int head_dim, int dv,
                               int n_head, int n_kv_heads, float scale, float softcap,
                               const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                               const _Float16 *mask, _Float16 *out, int nthreads);
/*
 * Computes the same attention as `rocket_flash_attn_fp16_mt` on the persistent context `c`
 * from `rocket_fa_ctx_create`. The result is numerically identical to `_mt`, and a call uses
 * up to min(nthreads, n_head) workers. Returns 0 on success and a negative value on error.
 */
int  rocket_flash_attn_fp16_ctx(rocket_fa_ctx *c, int n_tokens, int n_kv, int head_dim, int dv,
                                int n_head, int n_kv_heads, float scale, float softcap,
                                const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                                const _Float16 *mask, _Float16 *out);
/* Computes the exact host reference, the `fd` < 0 path of the entries above. */
void rocket_flash_attn_ref_fp16(int n_tokens, int n_kv, int head_dim, int dv,
                                int n_head, int n_kv_heads, float scale, float softcap,
                                const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                                const _Float16 *mask, _Float16 *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_ATTN_H */
