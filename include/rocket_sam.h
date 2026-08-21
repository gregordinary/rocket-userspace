// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_SAM_H
#define ROCKET_SAM_H

#include <stdint.h>
#include <stddef.h>

/*
 * rocket_sam — the SAM ViT-Det image encoder (facebook/sam-vit-base) end to end on the
 * rocket NPU, assembled from the validated primitives. It is the SigLIP/CLIP plain-ViT
 * encoder's windowed, relative-position-biased sibling; a separate module so the
 * bit-exact rocket_siglip path stays untouched, but it reuses the same primitive layer
 * (resident matmul, stream matmul, conv2d, the host LN/GELU/softmax discipline).
 *
 * The graph (per HF modeling_sam.py, verified against the ONNX export):
 *
 *   patch-embed   Conv2d(ic->d, k=patch, s=patch) WITH bias   ==  im2col + matmul
 *   + pos-embed   add the learned 2D [grid,grid,d] table (host glue)
 *   12 x block    pre-norm (LN1 -> attn -> res, LN2 -> erf-GELU MLP -> res):
 *                   - a FUSED qkv Linear (d -> 3d), split into q/k/v
 *                   - WINDOWED attention on all layers except the periodic global ones:
 *                     the 2D token grid is partitioned into `win`x`win` windows (the grid
 *                     is zero-padded up to a multiple of `win`, and those pad tokens DO
 *                     participate as keys/queries in boundary windows -- faithful repro);
 *                     the global layers run full self-attention over the whole grid.
 *                   - scale is folded into q before QK^T (DINOv2-style, NOT SigLIP)
 *                   - a decomposed RELATIVE-POSITION bias is added to the QK^T scores
 *                     BEFORE softmax (computed from the UNSCALED q), so attention runs as
 *                     matmul -> +bias -> softmax -> AV (NOT a fused flash-attn).
 *   neck          Transpose NHWC->NCHW -> conv1x1 d->neck_out -> channels-first LayerNorm
 *                 -> conv3x3 pad1 -> channels-first LayerNorm -> [1,neck_out,grid,grid]
 *
 * The big GEMMs (qkv/o, fc1/fc2, the patch projection, the neck convs), the per-window /
 * per-head QK^T and P*V matmuls run on the NPU; the O(tokens*d) glue (im2col, pos add,
 * window partition/unpartition, the LayerNorms, the erf-GELU, the softmax, and the
 * decomposed rel-pos matmul+add) runs on the host. The rel-pos add is faithful in fp16
 * (validated cos >= 0.9999991/layer), so there is no host-fp32 fallback for it.
 *
 * DECOMPOSED REL-POS. Per head, with q reshaped to [B, side, side, dhead] (B = windows*heads):
 *   rel_h[b, i, j, i'] = sum_c q[b,i,j,c] * Rh[i, i', c]     (Rh is [side, side, dhead])
 *   rel_w[b, i, j, j'] = sum_c q[b,i,j,c] * Rw[j, j', c]     (Rw is [side, side, dhead])
 *   bias[b, (i,j), (i',j')] = rel_h[b,i,j,i'] + rel_w[b,i,j,j']
 *   scores += bias                                          (added to scale*QK^T)
 * The [side,side,dhead] tables are the resolved get_rel_pos output (q_size==k_size, so the
 * F.interpolate is identity): Rh[i,j,c] = rel_pos_param[(i-j)+(side-1), c]. `side` is `win`
 * for a windowed layer and `grid` for a global one.
 *
 * Weights come from a flat fp16 blob (tools/sam_extract.py; header + weights in declaration
 * order) OR are filled directly by a frontend (e.g. the ort-rocket EP marshaling). The blob
 * loader mmaps and points the struct at each tensor; a frontend sets the pointers itself and
 * leaves `map` a non-NULL sentinel it owns (NEVER rocket_sam_free). fd < 0 runs the exact
 * host reference (the off-device datapath self-check).
 */

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKET_SAM_MAX_LAYERS 32   /* SAM-B 12, ViT-L 24, ViT-H 32 */

typedef struct {
    /* mmap handle (blob loader only; a frontend leaves map a sentinel it owns) */
    void  *map;
    size_t map_size;

    /* geometry */
    int   d;            /* hidden size (768)                         */
    int   n_layers;     /* 12                                        */
    int   n_head;       /* 12                                        */
    int   dhead;        /* d / n_head (64)                           */
    int   d_ff;         /* mlp intermediate (3072)                   */
    int   grid;         /* token-grid side = image_size / patch (64) */
    int   win;          /* window size (14)                          */
    int   patch;        /* patch conv kernel==stride (16)            */
    int   image_size;   /* 1024                                      */
    int   ic;           /* input channels (3)                        */
    int   patch_dim;    /* ic*patch*patch (768)                      */
    int   neck_out;     /* neck output channels (256)                */
    float eps;          /* layer_norm_eps (1e-6)                     */

    /* stem (fp16, into the blob / owned buffer) */
    const _Float16 *patch_W;    /* [d][patch_dim]  row-major [oc][ic*kh*kw] */
    const _Float16 *patch_b;    /* [d]                                      */
    const _Float16 *pos;        /* [grid*grid][d]  (the [1,grid,grid,d] table flattened) */

    /* per-layer weight pointers */
    const _Float16 *ln1_g[ROCKET_SAM_MAX_LAYERS], *ln1_b[ROCKET_SAM_MAX_LAYERS];
    const _Float16 *ln2_g[ROCKET_SAM_MAX_LAYERS], *ln2_b[ROCKET_SAM_MAX_LAYERS];
    const _Float16 *Wqkv[ROCKET_SAM_MAX_LAYERS], *bqkv[ROCKET_SAM_MAX_LAYERS];  /* [3d][d], [3d] */
    const _Float16 *Wo[ROCKET_SAM_MAX_LAYERS],   *bo[ROCKET_SAM_MAX_LAYERS];    /* [d][d], [d]   */
    const _Float16 *Wf1[ROCKET_SAM_MAX_LAYERS],  *bf1[ROCKET_SAM_MAX_LAYERS];   /* [d_ff][d], [d_ff] */
    const _Float16 *Wf2[ROCKET_SAM_MAX_LAYERS],  *bf2[ROCKET_SAM_MAX_LAYERS];   /* [d][d_ff], [d]    */
    const _Float16 *Rh[ROCKET_SAM_MAX_LAYERS],   *Rw[ROCKET_SAM_MAX_LAYERS];    /* [side][side][dhead] */
    int             Sl[ROCKET_SAM_MAX_LAYERS];       /* side per layer (win or grid)  */
    int             windowed[ROCKET_SAM_MAX_LAYERS]; /* 1 windowed, 0 global          */

    /* neck */
    const _Float16 *neck_c1;         /* [neck_out][d]            (1x1 conv weight [oc][ic]) */
    const _Float16 *neck_ln1_g, *neck_ln1_b;   /* [neck_out]                              */
    const _Float16 *neck_c2;         /* [neck_out][neck_out][3][3]                         */
    const _Float16 *neck_ln2_g, *neck_ln2_b;   /* [neck_out]                              */
} rocket_sam_model;

/* mmap + validate a weight blob from sam_extract.py. Returns 0, <0 on error
 * (-1 open, -2 mmap/size, -3 bad magic/version, -4 inconsistent dims). */
int  rocket_sam_load(const char *path, rocket_sam_model *m);
void rocket_sam_free(rocket_sam_model *m);

/*
 * Encode one image. `pixels_chw` is the preprocessed input [ic][image_size][image_size]
 * fp16. `out` receives the neck output [neck_out][grid][grid] fp16 (== the ONNX
 * last_hidden_state [1,neck_out,grid,grid]). If `hidden_opt` is non-NULL it receives the
 * (n_layers+1) intermediate token grids [(n_layers+1)][grid*grid][d] fp16 (index 0 =
 * embeddings, index k = the output of encoder layer k-1) -- used by the gate for per-layer
 * cosine. fd >= 0 runs on the NPU (one-shot: re-packs weights every call); fd < 0 runs the
 * exact host reference. Returns 0, <0 on error.
 */
int  rocket_sam_encode(int fd, const rocket_sam_model *m,
                       const _Float16 *pixels_chw, _Float16 *out, _Float16 *hidden_opt);

/* ---- resident (prepacked, multicore) encoder -- the latency path ----------------
 * Packs the static GEMM weights (patch, per-layer qkv/o/fc1/fc2, neck conv1) into resident
 * NPU BOs ONCE at ctx-create and reuses them across images; per image only the activations
 * are packed. Bit-faithful to rocket_sam_encode within fp16. Returns NULL on create failure.
 *
 *   c = rocket_sam_ctx_create(&m, nthreads);
 *   ... per image: rocket_sam_encode_ctx(c, pixels, out, hidden_opt);
 *   rocket_sam_ctx_free(c);
 */
typedef struct rocket_sam_ctx rocket_sam_ctx;

rocket_sam_ctx *rocket_sam_ctx_create(const rocket_sam_model *m, int nthreads);
void            rocket_sam_ctx_free(rocket_sam_ctx *c);
int  rocket_sam_encode_ctx(rocket_sam_ctx *c, const _Float16 *pixels_chw,
                           _Float16 *out, _Float16 *hidden_opt);

#ifdef __cplusplus
}
#endif
#endif /* ROCKET_SAM_H */
