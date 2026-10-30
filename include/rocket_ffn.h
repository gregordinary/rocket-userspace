// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_FFN_H
#define ROCKET_FFN_H

#include <stdint.h>

/*
 * The transformer FFN block on the NPU: the gated FFN of an LLM layer, and the non-gated FFN
 * of an encoder layer.
 *
 *     gate = x · Wg^T        [M,I]      (gate projection)
 *     up   = x · Wu^T        [M,I]      (up projection;  Wg/Wu share x -> fuse-able)
 *     prod = act(gate) ⊙ up  [M,I]      (the gated activation — GeGLU/SwiGLU core)
 *     out  = prod · Wd^T     [M,H]      (down projection)
 *
 * The one operation beyond a plain matmul is the gated activation `act(gate) ⊙ up`, which
 * `rocket_geglu_fp16` exposes on its own. `rocket_ffn_fp16` runs it between the three matmuls
 * and hands each intermediate back to the host. Each matmul output is de-tiled to row-major and
 * re-packed for the next op, and that host round trip is most of the block's cost. The
 * `_fused` entries keep the intermediates in cube layout on the NPU and avoid it.
 */

/*
 * Computes the gated activation `prod[i] = act(gate[i]) * up[i]` elementwise, the core of
 * GeGLU and SwiGLU. `n` is the flat element count, `M*I`.
 *
 * `kind` is a `rocket_activation_kind`, either `ROCKET_ACTIVATION_SILU` or
 * `ROCKET_ACTIVATION_GELU`. SiLU runs in two passes, a sigmoid LUT and then a multiply, which
 * avoids the error spike a one-pass LUT has near x = 0. The multiply runs on the host by
 * default. Set `ROCKET_ACT_NPU_MUL=1` to run it on the NPU.
 *
 * Caution: the activation LUT covers a bounded input range, [-12, 12] for SiLU, and a gate
 * logit outside it saturates.
 *
 * With `fd` < 0 it computes the exact host result. Returns 0 on success and a negative value
 * on error.
 */

#ifdef __cplusplus
extern "C" {
#endif
int  rocket_geglu_fp16(int fd, const _Float16 *gate, const _Float16 *up,
                       int kind, _Float16 *prod, int n);
void rocket_geglu_ref_fp16(const _Float16 *gate, const _Float16 *up,
                           int kind, _Float16 *prod, int n);

/*
 * Computes the gated FFN block, handing each intermediate back to the host between ops.
 *
 * `x` is [M,H], `Wg` and `Wu` are [I,H], `Wd` is [H,I] and `out` is [M,H], all row-major fp16
 * in the [N,K] matmul weight convention. `kind` selects the gated activation, as for
 * `rocket_geglu_fp16`. `H` and `I` must be multiples of 32, and `M` a multiple of 4. These are
 * the matmul requirements, where `I` is both an `N` and the down matmul's `K`. The block uses
 * the multicore matmul, the on-NPU activation and the elementwise multiply (`ew_mul`), and
 * needs no other setup.
 *
 * With `fd` < 0 it computes the exact host reference. Returns 0 on success and a negative
 * value on error.
 */
int  rocket_ffn_fp16(int fd, int M, int H, int I,
                     const _Float16 *x, const _Float16 *Wg, const _Float16 *Wu,
                     const _Float16 *Wd, int kind, _Float16 *out);
void rocket_ffn_ref_fp16(int M, int H, int I,
                         const _Float16 *x, const _Float16 *Wg, const _Float16 *Wu,
                         const _Float16 *Wd, int kind, _Float16 *out);

/*
 * Computes the same gated FFN block as `rocket_ffn_fp16`, with the same signature, and keeps
 * the [M,I] intermediates in cube layout on the NPU. The host packs only `x` and reads back
 * only `out`.
 *
 * The two projections leave their output cubes in resident BOs. The gated activation runs
 * elementwise over the cube bytes. The activation commutes with the cube reindex and keeps the
 * pad lanes at zero (act(0)·0 = 0), so the result is unchanged. The down matmul reads the
 * product cube as its input, at the same IOVA.
 *
 * The entry accepts fp16 only, because the aliasing depends on fp16's narrowed output having
 * the layout of an input feature cube. The down matmul's K tiling is pinned so that its `Kt`
 * equals the projections' `Nt`. This grouping of the fp16 K-accumulation is the only
 * numerical difference from `rocket_ffn_fp16`. The result has cosine similarity ~1 against it
 * and against the fp64 reference. The alignment is the same: `H` and `I` multiples of 32, and
 * `M` a multiple of 4.
 *
 * A shape whose projection output exceeds one NPU batch (`nMt*nNt > 64`), or whose tiling
 * cannot be matched, falls back to `rocket_ffn_fp16`. The fallback is correct and not
 * cube-resident. With `fd` < 0 it computes the host reference. Returns 0 on success and a negative value on
 * error.
 */
int  rocket_ffn_fp16_fused(int fd, int M, int H, int I,
                           const _Float16 *x, const _Float16 *Wg, const _Float16 *Wu,
                           const _Float16 *Wd, int kind, _Float16 *out);

/*
 * Computes the non-gated encoder FFN, `out = act(x·W1^T + b1) · W2^T`, and keeps the [M,Dff]
 * intermediate in cube layout on the NPU. The host packs only `x` and reads back only `out`.
 * Encoder blocks gain the most from this, because SigLIP and Whisper encoders spend ~80% of
 * their time in layout transforms.
 *
 * The bias `b1` is added on the cube: a per-channel bias scattered into the cube layout, then
 * a flat elementwise add. The activation runs elementwise over the cube bytes, and the fc2
 * matmul reads the activated cube directly.
 *
 * `x` is [M,Din], `W1` is [Dff,Din], `b1` is [Dff] or NULL for no bias, `W2` is [Dout,Dff] and
 * `out` is [M,Dout], all row-major fp16. `kind` is the activation, `ROCKET_ACTIVATION_GELU`
 * for a standard encoder, which runs the accurate two-pass path. The entry accepts fp16 only,
 * for the same aliasing reason as `rocket_ffn_fp16_fused`. `Din` and `Dff` must be multiples
 * of 32, `Dout` a multiple of 16 and `M` a multiple of 4.
 *
 * A shape whose fc1 output exceeds one NPU batch (`nMt*nNt > 64`), or whose tiling cannot be
 * matched, falls back to a host-handoff FFN. The fallback is correct and not cube-resident.
 * With `fd` < 0 it computes the host reference. Returns 0 on success and a negative value on
 * error.
 */
int  rocket_mlp_fp16_fused(int fd, int M, int Din, int Dff, int Dout,
                           const _Float16 *x, const _Float16 *W1, const _Float16 *b1,
                           const _Float16 *W2, int kind, _Float16 *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_FFN_H */
