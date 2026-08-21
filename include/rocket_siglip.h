// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_SIGLIP_H
#define ROCKET_SIGLIP_H

#include <stdint.h>

/*
 * rocket_siglip — the SigLIP-B/16 vision encoder (the SmolVLM-256M front-end) end to end
 * on the rocket NPU, assembled from the validated primitives. The graph:
 *
 *   patch-embed   Conv2d(ic->d, k=patch, s=patch)  ==  im2col + matmul (host gather, NPU GEMM)
 *   + pos-embed   add the learned [L,d] position table (host glue)
 *   12 x block    rocket_encoder_block_fp16 — pre-norm (LN->MHA->res, LN->GELU-MLP->res), on NPU
 *   post-LN       rocket_layernorm_fp16 over the d axis (on NPU)
 *
 * The big GEMMs (q/k/v/o, fc1, fc2, the patch projection), every softmax, both LayerNorms
 * per block, the residual adds, and the MLP GELU all run on the NPU; the only host work is
 * O(L*d) glue (the im2col gather, the patch-bias + position adds, head slicing inside the MHA).
 *
 * GELU note: the encoder block uses the exact-erf 2-pass GELU (x*Phi(x)); SigLIP trains with
 * gelu_pytorch_tanh. The two differ by ~1e-3, well under the fidelity target — no tanh LUT is
 * needed unless a future cosine gate misses.
 *
 * Weights come from a flat fp16 blob built by tools/siglip_extract.py (header + the weights
 * in declaration order). rocket_siglip_load() mmaps it and points the struct at each tensor.
 * fd < 0 runs the exact host reference (a structural self-check vs the fp32 oracle).
 */

#define ROCKET_SIGLIP_MAX_TAPS 8   /* DPT heads tap 4; headroom for wider variants */

typedef struct {
    /* mmap handle (private) */
    void  *map;
    size_t map_size;

    /* geometry (from the blob header) */
    int   d, n_layers, n_head, d_ff;
    int   L;            /* number of patches = side*side                */
    int   patch_dim;    /* ic*kh*kw (the im2col / patch_W contraction)  */
    int   ic, kh, kw, stride, image_size;
    float eps;

    /* The patch projection's matmul K. The NPU matmul requires K % 32, and ic*kh*kw is only
     * a multiple of 32 for some patch sizes (768 for patch 16, but 588 for patch 14). Set this
     * to patch_dim rounded up to a multiple of 32 and lay patch_W out at THAT row stride with a
     * zeroed [patch_dim, patch_dim_pad) tail; the encoder then zero-pads the im2col columns to
     * match, which leaves the product unchanged. 0 means "no padding needed" (patch_dim is
     * already %32) and is what the blob loader sets, so a v1 blob is unaffected. */
    int   patch_dim_pad;

    /* base weight pointers into the mmap (fp16) */
    const _Float16 *patch_W;     /* [d][patch_dim]  row-major [oc][ic*kh*kw]  */
    const _Float16 *patch_b;     /* [d], or NULL for a bias-less patch Conv    */
    const _Float16 *pos;         /* [ntok][d] position table (ntok = L+n_prefix) */
    const _Float16 *layers;      /* base of the per-layer weight region       */
    const _Float16 *post_g;      /* [d] post_layernorm weight, or NULL         */
    const _Float16 *post_b;      /* [d] post_layernorm bias, or NULL           */
    size_t          layer_stride;/* fp16 elements between consecutive layers  */

    /* Plain-ViT variant knobs (the EP marshaling path sets these; the blob loader
     * leaves them at the SigLIP defaults 0/NULL, so a v1 blob loads unchanged).
     * ntok = L + n_prefix is the token-sequence length the encoder runs over. */
    int             n_prefix;    /* prepended tokens (CLIP cls = 1); 0 for SigLIP */
    const _Float16 *prefix;      /* [n_prefix][d] prefix (cls) embeddings, or NULL */
    const _Float16 *pre_g;       /* [d] pre-encoder LayerNorm weight (CLIP), or NULL */
    const _Float16 *pre_b;       /* [d] pre-encoder LayerNorm bias, or NULL         */
    int             act_kind;    /* MLP activation: 0 = gelu_pytorch_tanh (SigLIP),
                                  *                 1 = quick-GELU x*sigmoid(1.702x) (CLIP),
                                  *                 2 = exact erf GELU x*Phi(x) (DINOv2) */

    /* Multi-tap exit. A DPT-style dense head (Depth Anything v2) does not consume the final
     * encoder output — it consumes SEVERAL intermediate layer outputs, each through the same
     * post_g/post_b LayerNorm. With n_taps > 0 the encoder writes n_taps token blocks to `out`
     * ([n_taps][ntok][d], tap t = the output of layer tap_layer[t], normalized by post_g/post_b
     * when present) instead of the single final block. 0 (the default) keeps the single-exit
     * behaviour SigLIP and CLIP use. tap_layer must be ascending and < n_layers. */
    int             n_taps;
    int             tap_layer[ROCKET_SIGLIP_MAX_TAPS];
} rocket_siglip_model;

#ifdef __cplusplus
extern "C" {
#endif

/* mmap + validate a weight blob from siglip_extract.py. Returns 0, <0 on error
 * (-1 open, -2 mmap/size, -3 bad magic/version, -4 inconsistent dims). */
int  rocket_siglip_load(const char *path, rocket_siglip_model *m);
void rocket_siglip_free(rocket_siglip_model *m);

/*
 * Encode one image. `pixels_chw` is the preprocessed input [ic][image_size][image_size]
 * fp16 (the same tensor the oracle saved). `out` receives the encoder output
 * [ntok][d] fp16 (ntok = L + n_prefix; post_layernorm applied when post_g != NULL, else the
 * raw last residual == CLIP's last_hidden_state), or, when n_taps > 0, the n_taps normalized
 * tap blocks [n_taps][ntok][d]. If `hidden_opt` is non-NULL it receives the
 * (n_layers+1) intermediate hidden states [(n_layers+1)][ntok][d] fp16 (index 0 = embeddings,
 * index k = the output of encoder layer k-1) — used by the gate to score per-layer cosine.
 * fd >= 0 runs on the NPU; fd < 0 runs the exact host reference. Returns 0, <0 on error.
 *
 * This per-call path handles the SigLIP configuration (n_prefix==0, no pre-LN, gelu-tanh,
 * post-LN, single exit); a CLIP-shaped model (cls token / pre-LN / quick-GELU) or a multi-tap
 * DPT-style model must use the resident ctx path below, and this returns -3 if handed one.
 */
int  rocket_siglip_encode(int fd, const rocket_siglip_model *m,
                          const _Float16 *pixels_chw, _Float16 *out,
                          _Float16 *hidden_opt);

/* ---- resident (prepacked, multicore) encoder — the latency path ----------------
 * The encode above re-packs every static weight and re-allocates BOs on every call
 * (fine for a one-shot, but pack+readback dominates). The resident path packs the 7
 * static GEMMs per layer (patch, q/k/v/o, fc1, fc2) into resident NPU BOs ONCE at
 * ctx-create and fans them across the 3 NPU cores; per image only the activations are
 * packed. The computed attention matmuls (scores, P·V) use a resident streaming
 * context; softmax runs on the NPU; LayerNorm / GELU / residual / bias adds run on
 * the host (memory-bound, faster once the data is already de-tiled). The host GELU is
 * the gelu_pytorch_tanh formula (an exact match to SigLIP's training activation).
 *
 *   c = rocket_siglip_ctx_create(&m, nthreads);   // packs all weights once (slow)
 *   ... per image:
 *       rocket_siglip_encode_ctx(c, pixels, out, hidden_opt);
 *   rocket_siglip_ctx_free(c);
 *
 * Bit-faithful to rocket_siglip_encode within fp16 (the gate cross-checks cosine).
 * Returns NULL on create failure (e.g. a weight pack hit the IOVA limit).
 */
typedef struct rocket_siglip_ctx rocket_siglip_ctx;

rocket_siglip_ctx *rocket_siglip_ctx_create(const rocket_siglip_model *m, int nthreads);
void               rocket_siglip_ctx_free(rocket_siglip_ctx *c);
int  rocket_siglip_encode_ctx(rocket_siglip_ctx *c, const _Float16 *pixels_chw,
                              _Float16 *out, _Float16 *hidden_opt);

#ifdef __cplusplus
}
#endif
#endif /* ROCKET_SIGLIP_H */
