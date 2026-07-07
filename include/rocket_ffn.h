// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_FFN_H
#define ROCKET_FFN_H

#include <stdint.h>

/*
 * rocket_ffn — the gated-MLP transformer FFN block on the NPU.
 *
 *     gate = x · Wg^T        [M,I]      (gate projection)
 *     up   = x · Wu^T        [M,I]      (up projection;  Wg/Wu share x -> fuse-able)
 *     prod = act(gate) ⊙ up  [M,I]      (the gated activation — GeGLU/SwiGLU core)
 *     out  = prod · Wd^T     [M,H]      (down projection)
 *
 * The only computation here beyond plain matmul is the GATED ACTIVATION `act(gate) ⊙ up`,
 * so that is the reusable primitive (rocket_geglu_fp16); the block (rocket_ffn_fp16) just
 * wraps it between the three matmuls.
 *
 * RESIDENT-FUSION NOTE: this composition uses HOST handoff between ops (each matmul reads
 * back to row-major; the activation/ew re-scatter). That is correct but pays the
 * de-tile->host->re-pack round-trip the not-mac-bound budget is dominated by. The perf win
 * (keeping the [M,I] intermediates cube-resident between matmul->act->mul->matmul) is the
 * follow-on; this block is the correctness substrate + the round-trip measurement
 * baseline it builds on.
 */

/*
 * Gated activation (GeGLU / SwiGLU core):  prod[i] = act(gate[i]) * up[i], elementwise.
 *   `kind` is a rocket_activation_kind (ROCKET_ACTIVATION_SILU or _GELU — the gated-MLP
 *   nonlinearities). SiLU runs the robust 2-pass activation (sigmoid LUT then multiply, no
 *   x≈0 glitch); set ROCKET_ACT_NPU_MUL=1 for the fully-on-NPU SiLU multiply. The activation
 *   LUT domain is bounded (SiLU [-12,12]) — gate logits outside it saturate (see findings).
 *   n is the flat element count (M*I). fd<0 = exact host. Returns 0, <0 on error.
 */

#ifdef __cplusplus
extern "C" {
#endif
int  rocket_geglu_fp16(int fd, const _Float16 *gate, const _Float16 *up,
                       int kind, _Float16 *prod, int n);
void rocket_geglu_ref_fp16(const _Float16 *gate, const _Float16 *up,
                           int kind, _Float16 *prod, int n);

/*
 * Full FFN block. x[M,H], Wg[I,H], Wu[I,H], Wd[H,I] (all row-major fp16, the [N,K] matmul
 * weight convention), out[M,H]. `kind` selects the gated nonlinearity. Alignment: H%32,
 * I%32, M%4 (the matmul requirements; I is both an N and the down-matmul's K). Self-contained
 * (uses the multicore matmul + on-NPU activation/ew_mul). fd<0 = exact host reference.
 * Returns 0, <0 on error.
 */
int  rocket_ffn_fp16(int fd, int M, int H, int I,
                     const _Float16 *x, const _Float16 *Wg, const _Float16 *Wu,
                     const _Float16 *Wd, int kind, _Float16 *out);
void rocket_ffn_ref_fp16(int M, int H, int I,
                         const _Float16 *x, const _Float16 *Wg, const _Float16 *Wu,
                         const _Float16 *Wd, int kind, _Float16 *out);

/*
 * CROSS-OP CUBE-RESIDENT FFN — same math and signature as rocket_ffn_fp16, but the [M,I]
 * intermediates stay in NPU cube layout across gate/up -> act⊙up -> down, so the host never
 * de-tiles/re-tiles them. The two projections leave their full output cubes in resident BOs
 * (no de-tile); the gated activation runs element-wise over the cube bytes (it commutes with
 * the cube reindex — pad lanes stay 0 since act(0)·0 = 0); the down matmul reads the product
 * cube directly as its input (same IOVA, zero host touch of the intermediate). Only x is
 * packed in and only `out` is read back.
 *
 * fp16-only (the cube aliasing is fp16's narrowed-output == input-feature-cube coincidence).
 * The numerics differ from rocket_ffn_fp16 only in the down matmul's K-tiling (pinned so its
 * Kt == the projections' Nt), i.e. a different fp16 K-accumulation grouping — same op, cosine
 * ~1 vs the host-handoff path and vs the fp64 oracle. Same alignment (H%32, I%32, M%4). Shapes
 * whose projection output exceeds one NPU batch (nMt*nNt > 64) or whose tiling can't be matched
 * transparently FALL BACK to rocket_ffn_fp16 (correct, just not cube-resident). fd<0 = host
 * reference. Returns 0, <0 on error.
 */
int  rocket_ffn_fp16_fused(int fd, int M, int H, int I,
                           const _Float16 *x, const _Float16 *Wg, const _Float16 *Wu,
                           const _Float16 *Wd, int kind, _Float16 *out);

/*
 * CROSS-OP CUBE-RESIDENT non-gated MLP — the transformer ENCODER feed-forward
 * (out = act(x·W1^T + b1) · W2^T), the regime where cross-op chaining pays (SigLIP/Whisper
 * are ~80% transform-bound). The fc1 output stays in NPU cube layout: the bias b1 is added
 * on the cube (a per-channel bias scattered into the cube layout, then a flat ew-add), the
 * activation runs element-wise over the cube bytes, and the fc2 matmul reads the activated
 * cube directly — so the [M,Dff] intermediate never de-tiles/re-tiles. Only x is packed in
 * and only `out` is read back.
 *
 * x[M,Din], W1[Dff,Din], b1[Dff] (row-major fp16; NULL = no bias), W2[Dout,Dff], out[M,Dout].
 * `kind` is the activation (ROCKET_ACTIVATION_GELU for the standard encoder; the 2-pass
 * accurate route). fp16-only (the cube aliasing). Alignment: Din%32, Dff%32, Dout%16, M%4.
 * Shapes whose fc1 output exceeds one NPU batch (nMt*nNt > 64) or can't be tiling-matched
 * FALL BACK to a host-handoff MLP (correct, not cube-resident). fd<0 = host reference.
 * Returns 0, <0 on error.
 */
int  rocket_mlp_fp16_fused(int fd, int M, int Din, int Dff, int Dout,
                           const _Float16 *x, const _Float16 *W1, const _Float16 *b1,
                           const _Float16 *W2, int kind, _Float16 *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_FFN_H */
