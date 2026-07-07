// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_CONV_H
#define ROCKET_CONV_H

#include <stdint.h>

/*
 * rocket_conv — general fp16 CONV_2D on the rocket NPU, built on the validated
 * single-task conv generator (gen_conv2d_fp16, npu_regcmd.c) + the rocket shim.
 *
 * The NPU CNA is a native convolution engine, so this is NOT im2col-on-a-matmul:
 * the host scatters the input feature into the NC1HWC2 cube and the weights into
 * the conv weight cube, and the CNA performs the KxK / stride / pad / dilation
 * sliding-window MAC in hardware. (The matmul path is the degenerate 1x1 case.)
 *
 * Tensor layouts (row-major, batch 1, NCHW-style):
 *   input    in [IC][IH][IW]
 *   weights  W  [OC][IC][KH][KW]   (direct);  [OC][1][KH][KW]  (depthwise, OC==IC)
 *   output   out[OC][OH][OW]
 *
 *   out[oc][oh][ow] = sum_{ic,kh,kw} W[oc][ic][kh][kw] *
 *       in[ic][oh*stride_y + kh*dil_y - pad_top][ow*stride_x + kw*dil_x - pad_left]
 *   (out-of-range input contributes 0 — zero padding).
 *
 * Alignment (the direct path): OC%16==0 (the conv weight oc group). IC need NOT be
 * a multiple of 32 — IC<32 (e.g. the RGB first layer, IC=3) is zero-padded up to 32
 * by the driver. Depthwise: OC==IC and IC%G==0, where G is the depthwise channel
 * group (ROCKET_CONV_DW_GROUP, default 32).
 *
 * Both the direct and depthwise paths are HW-validated bit-exact. Depthwise (weight
 * cube (C/G,KH,KW,G), CONV_MODE=3 + DW_EN) needs G=32 for fp16 plus the DPU
 * output-geometry regs (size_e=3, surf_add*2, feature_grains=52, bs_ow_op=128); it
 * tiles over CHANNELS (each channel independent) when a layer won't fit one CBUF pass.
 * ROCKET_CONV_DW_GROUP overrides G.
 *
 * TILING: the direct path tiles automatically over output channels (OC) and output
 * SPACE (OH rows then OW columns) when the problem won't fit one CBUF pass — each
 * tile is an independent HW-validated single job. Spatial tiles MATERIALIZE their
 * edge padding into the sub-input (the CNA has only symmetric pad_top/pad_left), so
 * a tile runs with pad_top=pad_left=0 over real-halo + explicit-zero rows/cols.
 * Depthwise tiles over CHANNELS instead (channels are independent — no halo): a wide
 * DW layer is split into chunks of Cc channels (a multiple of G) that each fit one
 * pass. A single channel whose own feature is too large for one pass would need
 * SPATIAL tiling (not yet implemented) — rocket_conv2d_plan returns <0 for that case.
 */

/* Output spatial dim for one axis. dil is the dilation rate (1 = none). */

#ifdef __cplusplus
extern "C" {
#endif
static inline int rocket_conv_out_dim(int in, int k, int stride, int pad, int dil)
{
    return (in + 2 * pad - (dil * (k - 1) + 1)) / stride + 1;
}

/* Convolution descriptor (shapes + hyperparameters; no buffers). */
typedef struct {
    int ic, ih, iw;     /* input  channels / height / width  */
    int oc;             /* output channels                   */
    int kh, kw;         /* kernel height / width             */
    int stride_y, stride_x;
    int pad_top, pad_left;
    int dil_y, dil_x;   /* dilation rate (1 = none)          */
    int depthwise;      /* 0 = direct conv, 1 = depthwise (OC==IC) */
} rocket_conv2d_desc;

/* Output dims implied by a descriptor (pure helper). */
static inline int rocket_conv2d_oh(const rocket_conv2d_desc *d)
{ return rocket_conv_out_dim(d->ih, d->kh, d->stride_y, d->pad_top, d->dil_y); }
static inline int rocket_conv2d_ow(const rocket_conv2d_desc *d)
{ return rocket_conv_out_dim(d->iw, d->kw, d->stride_x, d->pad_left, d->dil_x); }

/* Validate a descriptor against the supported set (alignment + single-tile CBUF
 * fit). Returns 0 if runnable, <0 (negated reason) otherwise. Pure, no hardware. */
int rocket_conv2d_plan(const rocket_conv2d_desc *d);

/* Run the conv on the NPU. `fd` is an open rocket device (rocket_open()). in / W /
 * out are row-major fp16 in the layouts above. Returns 0, negative on error. */
int rocket_conv2d_fp16(int fd, const rocket_conv2d_desc *d,
                       const _Float16 *in, const _Float16 *W, _Float16 *out);

/* ---- conv1d (the Whisper encoder front-end: width-only 1D conv over time) -------
 * 1D convolution over the time axis, lowered onto the HW-validated rocket_conv2d_fp16
 * as a HEIGHT-1 conv (IH=KH=OH=1, the time axis on the width/IW axis):
 *
 *   in  [IC][IT]            (IC channels, IT time steps)
 *   W   [OC][IC][KW]        (OC filters, kernel width KW)
 *   out [OC][OT]            OT = rocket_conv_out_dim(IT, KW, stride, pad, 1)
 *
 *   out[oc][t] = sum_{ic,kw} W[oc][ic][kw] * in[ic][t*stride + kw - pad]   (zero-padded)
 *
 * Whisper's two front-end convs are this op (KW=3, pad=1; conv1 stride 1, conv2 stride 2),
 * each followed by GELU (use the conv->act epilogue for the GELU, not the standalone op).
 * Pure descriptor wrapper — inherits conv2d's bit-exactness, alignment, and tiling. The
 * matching CPU oracle is rocket_conv2d_ref_fp16 on the same height-1 descriptor. Returns 0,
 * negative on error. */
int rocket_conv1d_fp16(int fd, int ic, int it, int oc, int kw, int stride, int pad,
                       const _Float16 *in, const _Float16 *W, _Float16 *out);

/* ---- resident-BO conv context (the per-call BO-churn lever) ----------------
 * rocket_conv2d_fp16 allocs + frees its NPU BOs (input / weight / regcmd / output
 * + an IOVA guard) on EVERY call — and per TILE for a tiled conv. For a delegate
 * running many convs (each possibly tiled) per inference, that per-call alloc/free
 * dominates the small-conv cost. A rocket_conv_ctx caches those BOs on a BORROWED
 * fd and grows each to the largest tile it has seen, so repeat calls reuse them
 * instead of re-allocating. Every job still memsets + refills its BOs, so reuse is
 * bit-identical to a fresh allocation (the resident matmul path uses the same trick).
 *
 *   ctx = rocket_conv_ctx_create(fd);            // borrows fd (does NOT close it)
 *   ... per inference, any supported conv shape:
 *       rocket_conv2d_fp16_ctx(ctx, &desc, in, W, out);
 *   rocket_conv_ctx_free(ctx);                   // frees the cached BOs (not the fd)
 *
 * The fd is BORROWED: the ctx never opens or closes it (the caller owns it). fd < 0
 * is accepted and inert — conv2d_one_job takes the CPU oracle before touching any BO,
 * so a ctx wrapping fd<0 just threads through to the oracle (one code path on/off
 * device). One ctx may serve convs of different shapes; the pool grows to the max. */
typedef struct rocket_conv_ctx rocket_conv_ctx;

rocket_conv_ctx *rocket_conv_ctx_create(int fd);
void             rocket_conv_ctx_free(rocket_conv_ctx *ctx);

int rocket_conv2d_fp16_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                           const _Float16 *in, const _Float16 *W, _Float16 *out);

/* ---- conv -> activation fusion (DIRECT fp16 conv; SiLU / tanh / GELU) ----------
 * Same as rocket_conv2d_fp16, but applies a SMOOTH activation f(x) in the SAME NPU
 * job via the DPU LUT epilogue (BN-mul index scale -> EW LUT -> affine OUT_CVT), so
 * out = f(conv(...)) with no second NPU round-trip / host activation pass. `kind` is
 * ROCKET_ACTIVATION_SILU / _TANH / _GELU (rocket_activation.h). HardSwish is rejected
 * (its exactly-flat x<=-3 tail trips the NVDLA LE/LO mux — host/2-pass only), as is
 * depthwise (direct-conv scope). Returns 0, <0 on error (-10 depthwise, -11 bad kind).
 * The fused f is an fp16 LUT approximation — gate with a tolerance, not equality. */
int rocket_conv2d_act_fp16(int fd, const rocket_conv2d_desc *d, int kind,
                           const _Float16 *in, const _Float16 *W, _Float16 *out);
int rocket_conv2d_act_fp16_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d, int kind,
                               const _Float16 *in, const _Float16 *W, _Float16 *out);

/* CPU fp32-accumulate reference (the golden oracle; also a host fallback). Writes
 * out[OC*OH*OW] as fp16 (fp32 accumulate, fp16 store), matching the NPU's
 * fp16-narrowed output. Pure host, no hardware. */
void rocket_conv2d_ref_fp16(const rocket_conv2d_desc *d,
                            const _Float16 *in, const _Float16 *W, _Float16 *out);

/* ---- native int8 CONV_2D (DIRECT, int32-raw) -------------------------------
 * The exact-W8A8 sibling of rocket_conv2d_fp16: int8 features x int8 weights reduced
 * NATIVELY by the CNA into an int32 accumulator written raw to DRAM (the caller
 * requants C[oc] by (per-output-channel weight scale x per-tensor activation scale),
 * folding any zero-point correction into the bias). No host dequant/requant round
 * trip and EXACT int8 semantics (bit-identical to TFLite's int8 CPU accumulate, modulo
 * the final requant rounding). The regcmd (gen_conv2d_int8) + cube layouts are
 * HW-validated bit-exact (tests/conv2d_int8_rocket.c).
 *
 * Layouts: in [IC][IH][IW] int8, W [OC][IC][KH][KW] int8, out [OC][OH][OW] int32.
 * Alignment: OC is zero-padded to the int8 weight oc-group (32) and IC to 32 (the RGB
 * first layer, IC<32, is zero-padded too) by the driver — any OC/IC is accepted.
 * Tiles over OC (mult-32) + OH-rows + OW-cols exactly like the fp16 path (each tile an
 * independent HW-validated single job; spatial tiles materialize their edge padding).
 * desc.depthwise must be 0 (int8 depthwise is the separate on-chip-requant entry
 * rocket_conv2d_dw_int8). Returns 0, negative on error. */
int rocket_conv2d_int8(int fd, const rocket_conv2d_desc *d,
                       const int8_t *in, const int8_t *W, int32_t *out);
int rocket_conv2d_int8_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                           const int8_t *in, const int8_t *W, int32_t *out);

/* ---- multicore worker pool for the native int8/uint8 DIRECT conv ---------------
 * The single-fd conv2d serializes its independent OC/OH/OW tiles onto one of the 3
 * NPU cores (one fd == one drm_sched entity == one core while it has queued work).
 * A pool of N worker fds, each with its own resident rocket_conv_ctx, lets the tiles
 * fan out across all 3 cores while keeping the resident-BO (no per-call alloc/free)
 * win. Create one per delegate partition (reused across ops/inferences); the pool
 * OWNS its fds. rocket_conv2d_int8_mt is bit-identical to rocket_conv2d_int8 (same
 * tiles, same single jobs) and falls back to serial for single-tile convs.
 * nthreads is clamped to [1,8] and degrades to however many fds actually opened. */
typedef struct rocket_conv_pool rocket_conv_pool;
rocket_conv_pool *rocket_conv_pool_create(int nthreads);
void              rocket_conv_pool_free(rocket_conv_pool *pool);
int rocket_conv2d_int8_mt(rocket_conv_pool *pool, const rocket_conv2d_desc *d,
                          const int8_t *in, const int8_t *W, int32_t *out);

/* CPU int64-accumulate -> int32 reference for the native int8 DIRECT/depthwise conv
 * (the golden oracle; also the fd<0 host fallback). The accumulate is int64 (a
 * 7x7x512 conv sums past int32) with an int32 store. Pure host, no hardware. */
void rocket_conv2d_ref_int8(const rocket_conv2d_desc *d,
                            const int8_t *in, const int8_t *W, int32_t *out);

/* ---- native int8 DEPTHWISE CONV_2D (int8-OUT, on-chip requant) -------------
 * The Teflon-cracked depthwise path: int8 in x int8 weight reduced per channel, then
 * REQUANTIZED ON-CHIP to int8 (no int32 readback) — bit-exact to Mesa/Teflon ground
 * truth (tests/replay_dw_mesa.c). PER-TENSOR quant only (Teflon's constraint); a
 * per-channel depthwise filter must stay on the dequant->fp16-DW->requant path until
 * the BS_MUL per-OC requant lands. The driver folds Mesa's zero-point correction into
 * the bias and centers the in/weight cubes in the uint8 domain, so the caller passes
 * the raw model tensors: in/w/out are int8 (model domain, [C][IH][IW] / [C][KH][KW] /
 * [C][OH][OW]); bias is the TFLite int32 bias [C] (may be NULL = no bias); the six
 * quant params are per-tensor. Single CBUF pass (no DW spatial tiling on this
 * path); channel-tiles like the fp16 DW path. Returns 0, negative on error. */
int rocket_conv2d_dw_int8(int fd, const rocket_conv2d_desc *d,
                          const int8_t *in, const int8_t *w, const int32_t *bias,
                          float in_scale, float w_scale, float out_scale,
                          int in_zp, int w_zp, int out_zp, int8_t *out);
int rocket_conv2d_dw_int8_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *w, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out);

/* ---- transposed convolution (ConvTranspose2d / "deconvolution") -------------
 * The transpose of a strided conv: each input pixel scatter-adds a kernel-weighted
 * copy into a LARGER output — learned upsampling for segmentation heads, decoder /
 * super-resolution / GAN-generator blocks, and the FPN learned-upsample. It is NOT a
 * new HW primitive: it is lowered onto the validated forward CONV_2D engine, so it
 * inherits the HW-exact direct-conv tiling/CBUF path bit-for-bit. The lowering: the
 * input is interior-dilated (stride-1 zeros inserted between pixels) + border-padded,
 * the kernel is spatially rotated 180 deg with its in/out channels transposed, and a
 * STRIDE-1 forward conv produces the result.
 *
 * Tensor layouts (row-major, batch 1):
 *   input    in [IC][IH][IW]
 *   weights  W  [IC][OC][KH][KW]   (direct;   PyTorch/ONNX order: IN channels first)
 *            W  [C][1][KH][KW]     (depthwise, OC==IC==C: one kernel per channel)
 *   output   out[OC][OH][OW]
 *
 *   OH = (IH-1)*stride_y - 2*pad_top  + dil_y*(KH-1) + opad_y + 1
 *   OW = (IW-1)*stride_x - 2*pad_left + dil_x*(KW-1) + opad_x + 1
 *
 *   out[oc][ph][pw] = sum_{ic,kh,kw : ih,iw integral and in-range} in[ic][ih][iw]*W[ic][oc][kh][kw]
 *   where ph = ih*stride_y - pad_top + kh*dil_y (and the analogous pw). Depthwise sums
 *   only ic==oc (each channel upsampled by its own kernel) — the per-channel form used by
 *   nearest / bilinear resize (a depthwise transpose with a box / triangle kernel).
 *
 * `opad_*` is the ONNX/PyTorch output_padding (an extra trailing-only border that
 * disambiguates the output size when stride>1; 0 = none, must be < stride). Constraint
 * (this bring-up): pad_top <= dil_y*(KH-1) and pad_left <= dil_x*(KW-1) so the lowered
 * border pad is non-negative (the usual case; a larger pad would CROP — _plan returns
 * -2). Direct: any OC zero-padded to 16, any IC to 32. Depthwise: OC==IC and C%G==0
 * (G=32, the depthwise channel group). The lowering materialises the dilated input, so
 * cost scales with the UPSAMPLED size (multiply-by-zero on the inserted gaps) —
 * correctness-first; a sub-pixel/stride^2 decomposition is the perf follow-on.
 * HW-validated bit-faithful to the direct scatter reference (tests/conv_transpose_rocket.c). */
typedef struct {
    int ic, ih, iw;     /* input  channels / height / width  */
    int oc;             /* output channels                   */
    int kh, kw;         /* kernel height / width             */
    int stride_y, stride_x;
    int pad_top, pad_left;
    int opad_y, opad_x; /* output_padding (extra trailing border; 0 = none, < stride) */
    int dil_y, dil_x;   /* kernel dilation rate (1 = none)   */
    int depthwise;      /* 0 = direct (W [IC][OC][KH][KW]); 1 = depthwise (OC==IC, W [C][1][KH][KW]) */
} rocket_conv_transpose2d_desc;

/* Output dims implied by a descriptor (pure helpers). */
static inline int rocket_conv_transpose2d_oh(const rocket_conv_transpose2d_desc *d)
{ return (d->ih - 1) * d->stride_y - 2 * d->pad_top + d->dil_y * (d->kh - 1) + d->opad_y + 1; }
static inline int rocket_conv_transpose2d_ow(const rocket_conv_transpose2d_desc *d)
{ return (d->iw - 1) * d->stride_x - 2 * d->pad_left + d->dil_x * (d->kw - 1) + d->opad_x + 1; }

/* Validate against the supported set (lowering feasibility + forward-conv CBUF fit).
 * Returns 0 if runnable, <0 (negated reason) otherwise. Pure, no hardware. */
int rocket_conv_transpose2d_plan(const rocket_conv_transpose2d_desc *d);

/* Run the transposed conv on the NPU. in / W / out are row-major fp16 in the layouts
 * above. Returns 0, negative on error. The _ctx form reuses a resident BO pool. */
int rocket_conv_transpose2d_fp16(int fd, const rocket_conv_transpose2d_desc *d,
                                 const _Float16 *in, const _Float16 *W, _Float16 *out);
int rocket_conv_transpose2d_fp16_ctx(rocket_conv_ctx *ctx,
                                     const rocket_conv_transpose2d_desc *d,
                                     const _Float16 *in, const _Float16 *W, _Float16 *out);

/* CPU fp32-accumulate reference (the golden oracle; also the fd<0 host fallback). Uses
 * the DIRECT scatter-add definition — independent of the lowering, so the gate validates
 * the lowering, not just the forward conv. Pure host, no hardware. */
void rocket_conv_transpose2d_ref_fp16(const rocket_conv_transpose2d_desc *d,
                                      const _Float16 *in, const _Float16 *W, _Float16 *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_CONV_H */
