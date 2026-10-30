// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_SIGLIP_H
#define ROCKET_SIGLIP_H

#include <stddef.h>
#include <stdint.h>

/*
 * Declares the SigLIP-B/16 vision encoder (the SmolVLM-256M front end), end to end on the
 * rocket NPU, assembled from the validated primitives. The graph:
 *
 *     patch-embed   Conv2d(ic->d, k=patch, s=patch)  ==  im2col + matmul (host gather, NPU GEMM)
 *     + pos-embed   add the learned [L,d] position table (host glue)
 *     12 x block    rocket_encoder_block_fp16: pre-norm (LN->MHA->res, LN->GELU-FFN->res), on NPU
 *     post-LN       rocket_layernorm_fp16 over the d axis (on NPU)
 *
 * On the per-call path, the large GEMMs run on the NPU: q/k/v/o, fc1, fc2 and the patch
 * projection. Every softmax, both LayerNorms per block, the residual adds and the GELU's Φ
 * lookup run there too. The host work is O(L*d) glue:
 *
 * - The im2col gather, and the patch-bias and position adds
 * - The head slicing, and the attention scale and biases
 * - The `bf2` add
 * - The GELU's multiply, unless `ROCKET_ACT_NPU_MUL=1` (`rocket_encoder.h` has the work split)
 *
 * Note: the encoder block uses the exact-erf 2-pass GELU (x*Phi(x)), and SigLIP trains with
 * gelu_pytorch_tanh. The two functions differ by at most 4.7e-4 [host-computed], well under
 * the fidelity target.
 *
 * Weights come from a flat fp16 blob that `tools/siglip_extract.py` builds: a header, then the
 * weights in declaration order. `rocket_siglip_load()` maps it and points the struct at each
 * tensor. With `fd` < 0 the encoder runs the exact host reference, a structural self-check
 * against the fp32 oracle.
 */

#define ROCKET_SIGLIP_MAX_TAPS 8   /* DPT heads tap 4, plus headroom for wider variants */

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

    /* The patch projection's matmul `K`. The NPU matmul requires `K` % 32, and ic*kh*kw is a
     * multiple of 32 only for some patch sizes (768 for patch 16, but 588 for patch 14). The
     * caller sets this to `patch_dim` rounded up to a multiple of 32. It lays `patch_W` out at
     * that row stride, with a zeroed [patch_dim, patch_dim_pad) tail. The encoder then zero-pads
     * the im2col columns to match, which leaves the product unchanged. 0 means no padding is
     * needed (`patch_dim` is already %32). The blob loader sets 0, so a v1 blob is unaffected. */
    int   patch_dim_pad;

    /* base weight pointers into the mmap (fp16) */
    const _Float16 *patch_W;     /* [d][patch_dim]  row-major [oc][ic*kh*kw]  */
    const _Float16 *patch_b;     /* [d], or NULL for a bias-less patch Conv    */
    const _Float16 *pos;         /* [ntok][d] position table (ntok = L+n_prefix) */
    const _Float16 *layers;      /* base of the per-layer weight region       */
    const _Float16 *post_g;      /* [d] post_layernorm weight, or NULL         */
    const _Float16 *post_b;      /* [d] post_layernorm bias, or NULL           */
    size_t          layer_stride;/* fp16 elements between consecutive layers  */

    /* Plain-ViT variant fields. The ort-rocket EP's marshaling path sets them. The blob loader
     * leaves them at the SigLIP defaults (0 or NULL), so a v1 blob loads unchanged.
     * `ntok` = L + n_prefix is the token-sequence length the encoder runs over. */
    int             n_prefix;    /* prepended tokens (CLIP cls = 1), 0 for SigLIP */
    const _Float16 *prefix;      /* [n_prefix][d] prefix (cls) embeddings, or NULL */
    const _Float16 *pre_g;       /* [d] pre-encoder LayerNorm weight (CLIP), or NULL */
    const _Float16 *pre_b;       /* [d] pre-encoder LayerNorm bias, or NULL         */
    int             act_kind;    /* FFN activation: 0 = gelu_pytorch_tanh (SigLIP),
                                  *                 1 = quick-GELU x*sigmoid(1.702x) (CLIP),
                                  *                 2 = exact erf GELU x*Phi(x) (DINOv2) */

    /* Multi-tap exit. A DPT-style dense head (Depth Anything v2) consumes several intermediate
     * layer outputs instead of the final encoder output, each through the same
     * `post_g`/`post_b` LayerNorm. With `n_taps` > 0 the encoder writes `n_taps` token blocks
     * to `out` instead of the single final block. The blocks are [n_taps][ntok][d], where tap t
     * is the output of layer tap_layer[t], normalized by post_g/post_b where present. The
     * default, 0, keeps the single-exit behavior SigLIP and CLIP use. `tap_layer` must be
     * ascending and < `n_layers`. */
    int             n_taps;
    int             tap_layer[ROCKET_SIGLIP_MAX_TAPS];
} rocket_siglip_model;

#ifdef __cplusplus
extern "C" {
#endif

/* Maps and validates a weight blob from siglip_extract.py. Returns 0 on success and a
 * negative value on error. The error is -1 for open, -2 for mmap or size, -3 for a bad magic
 * or version, and -4 for inconsistent dimensions. */
int  rocket_siglip_load(const char *path, rocket_siglip_model *m);
void rocket_siglip_free(rocket_siglip_model *m);

/*
 * Encodes one image. `pixels_chw` is the preprocessed input [ic][image_size][image_size]
 * fp16, the same tensor the oracle saved. `out` receives the encoder output [ntok][d] fp16,
 * where ntok = L + n_prefix. With `post_g` != NULL the output has post_layernorm applied, and
 * otherwise it is the raw last residual (== CLIP's last_hidden_state). With `n_taps` > 0, `out`
 * receives the `n_taps` normalized tap blocks [n_taps][ntok][d] instead.
 *
 * A non-NULL `hidden_opt` receives the (n_layers+1) intermediate hidden states
 * [(n_layers+1)][ntok][d] fp16. Index 0 is the embeddings, and index k is the output of
 * encoder layer k-1. The test uses them to score the cosine similarity of each layer.
 *
 * With `fd` >= 0 the encoder runs on the NPU, and with `fd` < 0 it runs the exact host
 * reference. Returns 0 on success and a negative value on error.
 *
 * This per-call path handles only the SigLIP configuration (n_prefix==0, no pre-LN,
 * gelu-tanh, post-LN, single exit). A CLIP-shaped model (cls token, pre-LN, quick-GELU) or a
 * multi-tap DPT-style model must use the resident ctx path below. For one, this entry returns
 * -3.
 */
int  rocket_siglip_encode(int fd, const rocket_siglip_model *m,
                          const _Float16 *pixels_chw, _Float16 *out,
                          _Float16 *hidden_opt);

/* ---- Resident encoder: prepacked weights on several cores, for low latency ---- */

/*
 * Creates, frees and runs the resident encoder. The encode above re-packs every static
 * weight and re-allocates BOs on every call. That suits a one-shot call, but the pack and
 * readback dominate its time. The resident path packs the 7 static GEMMs per layer (patch,
 * q/k/v/o, fc1, fc2) into resident NPU BOs once, at context creation. It spreads them across
 * the 3 NPU cores. Per image, only the activations are packed.
 *
 * The computed attention matmuls (scores, P·V) run on the NPU through the flash-attention
 * fan-out across the worker fds. `ROCKET_SIGLIP_FA=0`, or `nthreads` <= 1, keeps one resident
 * streaming context instead. By default the softmax runs on the host on both paths, as do
 * LayerNorm, GELU, the residual and the bias adds. Those are memory-bound, and faster on the
 * host once the data is already de-tiled. The host GELU follows `act_kind`, and for SigLIP it
 * is the gelu_pytorch_tanh formula, an exact match to SigLIP's training activation.
 *
 *     c = rocket_siglip_ctx_create(&m, nthreads);   // packs all weights once (slow)
 *     ... per image:
 *         rocket_siglip_encode_ctx(c, pixels, out, hidden_opt);
 *     rocket_siglip_ctx_free(c);
 *
 * The result is not bit-identical to `rocket_siglip_encode`'s. For SigLIP the resident host
 * GELU is the gelu_pytorch_tanh formula, while the per-call encoder block uses the exact-erf
 * GELU. The two functions differ by at most 4.7e-4 [host-computed]. `tests/siglip_rocket.c`
 * holds each path to the fp32 reference on its own. Each must reach a mean per-layer cosine similarity of at least 0.99
 * and a post-LayerNorm cosine similarity of at least 0.99. The test runs the resident path
 * only with `ROCKET_SIGLIP_BENCH` set, and does not compare the two paths with each other.
 *
 * `rocket_siglip_ctx_create` returns NULL on failure, for example on a weight pack that hit
 * the IOVA limit, or on host scratch that did not allocate.
 *
 * The context allocates the per-image host scratch once, at create: the token buffers, the
 * im2col patches and the attention operands (~17 MiB at Depth Anything's ViT-S at 518 px). An
 * encode therefore does no heap allocation.
 *
 * Caution: a context is not thread-safe. Each concurrent caller must use its own context.
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
