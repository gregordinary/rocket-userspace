// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_CONV_H
#define ROCKET_CONV_H

#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_pool.h"   /* a pooling layer may sit inside a cross-layer chain */

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
    /* AN EXPLICIT OUTPUT EXTENT, WHICH IS HOW AN ASYMMETRIC PAD IS EXPRESSED. The RK3576
     * CNA does not take a trailing pad: it DERIVES the pad its last window consumes from
     * the output extent and the leading pad. So TFLite's SAME at an even input and stride
     * two — pad_before 0, pad_after 1 — is pad_top/pad_left ZERO with an output one larger
     * than the symmetric formula gives, and it needs no border materialised in the caller's
     * buffer at all.
     *
     * Zero means "whatever rocket_conv2d_oh/ow derive", which is every caller that memsets
     * this struct. The RK3576 int8 entries honour it inside a decoded bound (the derived
     * trailing pad must be less than the kernel); the fp16 and RK3588 paths refuse an
     * extent they did not derive themselves. */
    int oh, ow;
    /* RUN A NARROW INPUT-CHANNEL COUNT ON THE DIRECT DATAPATH, not on a packed-image
     * first-conv sub-encoding.
     *
     * The RK3576 CNA has two ways to read an image of four or fewer channels: its own
     * packed-image sub-encoding, which interleaves the channels per pixel and carries
     * three silent geometry bounds (a non-zero left pad, an output width of exactly
     * iw/stride, and that width also a multiple of 16), and the ordinary direct one,
     * whose int8 cube is a 32-channel MAC group whatever the count — so ic 3 and ic 32
     * are the SAME register program, the same weight-cube size and the same feature-cube
     * size, and the channels past ic are the cube's own zero padding. The zero-point fold
     * is over the LIVE tap count, so the arithmetic is exact at any zero point.
     *
     * That makes the direct path the general one: no geometry bound, and nothing to
     * widen on the host. It costs the packed-image path's smaller feature read, which is
     * why the choice is the caller's — measured on this part, the packed-image form wins
     * a transient call and loses a resident one, where the host scatter is what remains.
     *
     * Set it and rocket_conv2d_int8_rk3576(), _pack_rk3576(), _perchannel_rk3576() and
     * _act_rk3576() take the direct path at any ic. Clear (every caller that memsets this
     * struct) keeps the packed-image routing at ic <= 4. The RK3576 fp16 entry refuses
     * it — its direct path has a different channel granularity and no gate at ic <= 4 —
     * and the RK3588 paths never take a packed-image encoding at all, so it is already
     * true there and accepted. */
    int direct_datapath;
} rocket_conv2d_desc;

/* Output dims implied by a descriptor (pure helper). An explicit d->oh / d->ow wins. */
static inline int rocket_conv2d_oh(const rocket_conv2d_desc *d)
{ return d->oh ? d->oh
               : rocket_conv_out_dim(d->ih, d->kh, d->stride_y, d->pad_top, d->dil_y); }
static inline int rocket_conv2d_ow(const rocket_conv2d_desc *d)
{ return d->ow ? d->ow
               : rocket_conv_out_dim(d->iw, d->kw, d->stride_x, d->pad_left, d->dil_x); }

/* The pad the last window CONSUMES past the plane, which is what the CNA programs. Zero
 * for a symmetric descriptor, and for TFLite's asymmetric SAME the trailing half. */
static inline int rocket_conv2d_trail_y(const rocket_conv2d_desc *d)
{
    int reach = (rocket_conv2d_oh(d) - 1) * d->stride_y + (d->dil_y ? d->dil_y : 1) *
                (d->kh - 1) + 1;
    int have  = d->pad_top + d->ih;
    return reach > have ? reach - have : 0;
}
static inline int rocket_conv2d_trail_x(const rocket_conv2d_desc *d)
{
    int reach = (rocket_conv2d_ow(d) - 1) * d->stride_x + (d->dil_x ? d->dil_x : 1) *
                (d->kw - 1) + 1;
    int have  = d->pad_left + d->iw;
    return reach > have ? reach - have : 0;
}

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

/* ---- RK3576: the per-chip convolution entries ------------------------------
 * The RK3576 NPU is the same IP family and runs through the same uAPI, but its
 * CNA/CORE/DPU blocks use a different geometry-register encoding at the same block
 * bases, so a program built for the RK3588 there submits, completes, and writes
 * nothing. These are that part's own encoders, driven.
 *
 * WHERE THE PUBLIC ENTRIES DISPATCH HERE, and where they cannot:
 *
 *   rocket_conv2d_dw_int8() and rocket_conv2d_fp16() have the semantics this part
 *   computes, so on an RK3576 they route to the two entries below and a caller writes
 *   nothing chip-specific.
 *
 *   rocket_conv2d_int8() does not. It writes a RAW int32 accumulator and requants on
 *   the host; the RK3576's direct int8 datapath requantizes ON CHIP and writes a byte.
 *   Emulating one with the other is not a wrapper — it is a different arithmetic — so
 *   that entry refuses on this part and names rocket_conv2d_int8_rk3576(), which takes
 *   the quant parameters and writes int8. Same reasoning, and the same answer, as
 *   rocket_matmul_int8() and rocket_matmul_int8_rk3576().
 *
 * Layouts and descriptor semantics are the ones above: in [IC][IH][IW], W
 * [OC][IC][KH][KW] (direct) or [C][KH][KW] (depthwise), out [OC][OH][OW], all
 * row-major. Any IC and OC are accepted — the driver pads to the part's own granules
 * — and rows and output channels tile automatically.
 *
 * The quant contract is per-tensor and the zero points are MODEL DOMAIN signed int8.
 * bias may be NULL. The input zero point and the weight zero point are folded into the
 * coefficient buffer exactly; the output zero point becomes the DPU's OUT_CVT offset.
 *
 * A DEPTHWISE WEIGHT ZERO POINT is carried in the WEIGHT CUBE, not in the coefficient
 * group. That group has no B field, and a depthwise zero point's correction is not a
 * per-channel constant so it cannot ride the bias either — but the cube gives every
 * (channel, tap) two live bytes and the datapath ADDS them, so `w - w_zp` is expressed
 * exactly across the pair. It is the form every per-tensor TFLite depthwise filter
 * carries. The bound is the pair's own range, [-256, 254].
 *
 * Refused rather than approximated: dilation (no RK3576 shape has been run through the
 * rate fields); IC <= 4, which takes the CNA's ARGB first-conv sub-encoding whose
 * weight cube is not decoded; and a shape whose resident weight slice does not fit even
 * one output-channel group, which needs an input-channel split that the on-chip requant
 * forecloses.
 *
 * Bit-exact against a CPU model over tests/rk3576_conv_lib_gate.c. [HW sweep, H96 MAX M9] */
int rocket_conv2d_int8_rk3576(int fd, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *W, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out);
int rocket_conv2d_dw_int8_rk3576(int fd, const rocket_conv2d_desc *d,
                                 const int8_t *in, const int8_t *w, const int32_t *bias,
                                 float in_scale, float w_scale, float out_scale,
                                 int in_zp, int w_zp, int out_zp, int8_t *out);

/* PER-OUTPUT-CHANNEL weight scales — `w_scale[oc]`, the form every TFLite int8 and ONNX
 * QDQ model carries. Otherwise identical to rocket_conv2d_int8_rk3576(): same layouts,
 * same tiling, same row window, int8 in and int8 out with the requant on chip.
 *
 * HOW IT IS EXPRESSED, AND WHAT THAT COSTS. The DPU's epilogue is
 * `(acc + A[oc]) * C[oc]` in saturating int32 followed by ONE `(v*MUL)>>SHIFT` for the
 * whole task, so the per-channel gain rides on the coefficient group's int16 C and the
 * OUT_CVT carries a single base. That reproduces a per-axis model CLOSELY, not
 * bit-exactly: TFLite's per-axis requant has a per-channel shift as well as a
 * per-channel multiplier, and this datapath has only the multiplier. Two things bound
 * the fidelity — C is an integer, so channel oc resolves its gain to 0.5/C[oc], and the
 * int32 product saturates, which caps C[oc] at `INT32_MAX / max|acc + A|`. The planner
 * takes the accumulator bound from the ACTUAL weights (128*sum|w[oc]|), not from the
 * int8 envelope, because that is one to two orders of magnitude tighter and the
 * difference is most of the available precision. It logs the worst-case relative gain
 * error it settled for; a layer with a wide scale spread and a large fan-in is where
 * that number gets big.
 *
 * So this is an ACCURACY decision, not a bit-exact one. What is gated bit-exactly is
 * the chip arithmetic above against a CPU model of it; how far that sits from an exact
 * per-axis reference is reported per shape rather than asserted.
 *
 * DEPTHWISE runs here too, and it is the shape a real per-axis model is mostly made of.
 * The 48-byte coefficient group carries the same per-channel C, and a depthwise task's
 * accumulator is bounded by kh*kw taps rather than ic*kh*kw — one to two orders of
 * magnitude smaller — so the C ramp reaches the int16 field's own 32767 ceiling and the
 * gain resolution a single OUT_CVT shift leaves is far finer than the direct path's at
 * the same channel count. It is ONE task whatever the channel count: output channel c
 * is bound to input channel c, so there is no output-channel window to program and no
 * scale sort to make, and the tile query below returns `oc` for a depthwise descriptor.
 *
 * Refused rather than approximated: four or fewer input channels on the DIRECT path
 * (the packed-image first conv is a different program), and a weight zero point (a
 * per-axis quantization is symmetric by construction).
 * [HW sweep, H96 MAX M9, tests/rk3576_coeff_c.c + tests/rk3576_perchannel_gate.c] */
int rocket_conv2d_int8_perchannel_rk3576(int fd, const rocket_conv2d_desc *d,
                                         const int8_t *in, const int8_t *W,
                                         const int32_t *bias, float in_scale,
                                         const float *w_scale, float out_scale,
                                         int in_zp, int out_zp, int8_t *out);

/*
 * The same convolution with a nonlinear ACTIVATION fused into its epilogue: one job, and
 * the activation costs only the table-load task. `kind` is a ROCKET_RK3576_ACT_* from
 * npu_regcmd_rk3576.h. Per-tensor quantization, direct or depthwise, RK3576 only.
 *
 * NOT THE SAME ARITHMETIC AS conv-THEN-rocket_act_int8_rk3576(), and the better one: the
 * standalone op reads the convolution's requantized int8 output and so sees 256 possible
 * values, while this reads the ACCUMULATOR before any rounding. The trade is exactness —
 * the standalone form puts each of its 256 inputs on its own table entry, and a real
 * accumulator lands between entries, so the hardware's linear interpolation runs. The
 * distance to an exact f() is a REPORTED quantity here, not an asserted one.
 *
 * The table's units and its step follow from the three scales the caller already passes,
 * so there is no extra contract: with a per-tensor requant the BS multiplier is unity, so
 * one accumulator count means `in_scale*w_scale`, an entry is read in units of
 * `out_scale` times the gain OUT_CVT programs, and the window step is the smallest that
 * covers the domain the activation still varies over. Two bounds are refused rather than
 * computed wrong — a value unit too fine for the table's peak to fit an int16, and a
 * domain no step reaches — and both name `in_scale*w_scale` as the quantity to change.
 *
 * PER-TENSOR ONLY. Per-output-channel weight scales and the LUT want the same register —
 * the coefficient group's `C` divides the value unit per channel, which is both how a
 * per-axis requant buys gain resolution and how a channel's range is placed inside the
 * table's window — and there is no measured model that needs both (the per-tensor fused form
 * is within one count of an exact f(), and a per-axis graph is within one count per layer
 * with no fusion). The order it would take if there ever is one is written down beside the
 * implementation: the window wins, and a conflict refuses rather than trades.
 *
 * Returns 0 or a negative rocket error. [HW sweep, H96 MAX M9, tests/rk3576_act_gate.c]
 */
int rocket_conv2d_int8_act_rk3576(int fd, const rocket_conv2d_desc *d,
                                  const int8_t *in, const int8_t *W, const int32_t *bias,
                                  float in_scale, float w_scale, float out_scale,
                                  int in_zp, int w_zp, int out_zp,
                                  int kind, int8_t *out);

/* The output-channel tile rocket_conv2d_int8_perchannel_rk3576() will use, without
 * running it — which is also
 * its submit count, `ceil(oc/tile)` row-task sets at the part's per-submit floor.
 *
 * On this path the tile is an ACCURACY parameter as well as a CBUF one: a tile is one
 * task and a task carries one OUT_CVT shift, so the C ramp inside a tile spans only
 * that tile's range of scales. The channels are sorted by scale first, so halving the
 * tile roughly halves the spread the ramp must cover. Measured at ic=128 oc=128 with a
 * 100x spread: 26.6 counts of deviation from an exact per-axis requant in one tile,
 * 2.7 at 64 channels, 1.0 at 32. The planner takes the LARGEST tile whose PREDICTED
 * worst-case gain error meets ROCKET_RK3576_PC_MAX_ERR (default 1%), so a layer that
 * does not need the split does not pay for it. ROCKET_RK3576_PC_OC_TILE forces one.
 *
 * Returns 0 for a descriptor this path refuses. [HW sweep, H96 MAX M9] */
unsigned rocket_conv2d_int8_perchannel_oc_tile_rk3576(const rocket_conv2d_desc *d,
                                                      const int8_t *W,
                                                      const int32_t *bias,
                                                      float in_scale,
                                                      const float *w_scale,
                                                      float out_scale, int in_zp);

/* ---- the residual add, as a convolution's weights --------------------------
 * The DPU's elementwise stage takes exactly ONE operand — every register in its
 * interface has been swept and none carries a second — so a skip connection is not an
 * elementwise op on this part. It is a CONVOLUTION: concatenate the two operands along
 * the channel axis and convolve with a 1x1 kernel of two diagonal blocks.
 *
 *     out[o] = requant( w1*(a[o] - a_zp) + w2*(b[o] - b_zp) )
 *
 * Everything in that lowering except the WEIGHTS is an ordinary convolution the entries
 * above already run, so this fills a caller's weight matrix, bias and weight scale and
 * stops. It is pure — no hardware, no allocation — and the caller then runs
 *
 *     d.ic = 2*c;  d.oc = c;  d.kh = d.kw = 1;  d.direct_datapath = 1;
 *     rocket_conv2d_int8_rk3576(fd, &d, concat, W, bias, a_scale, w_scale, out_scale,
 *                               a_zp, 0, out_zp, out);
 *
 * with `concat` the two operands' channels one after the other. In CUBE layout that
 * concatenation is free: a feature cube's channel-group stride is the plane and a direct
 * conv's output surface stride is `ow*oh`, so two producers writing adjacent halves of
 * one buffer already ARE the concatenated operand.
 *
 * TWO PROPERTIES BEAT THE VENDOR'S OWN ELEMENTWISE PROGRAM. The operands may carry
 * DIFFERENT scales — the ratio rides in the weights as `w2/w1`, resolved to about one
 * part in 127^2 by searching every denominator rather than anchoring one term at 127 —
 * where a single operand converter forces both operands onto a common scale. And BOTH
 * zero points ride exactly, because `w2*(a_zp - b_zp)` is a per-output-channel constant
 * and that is what the bias is.
 *
 * IT ALSO FUSES, in the sense that the datapath computes it — but on a real quantization
 * the weight the fusion needs does not fit the field. See
 * rocket_residual_fuse_weight_rk3576() below, which is where that bound lives.
 *
 * `W` is `c * 2c` int8 in the [oc][ic] order the direct entries take, `bias` is `c`
 * int32, and `w_pair` (optional) receives the chosen `{w1, w2}`. Refused rather than
 * approximated: `2c > 4608`, and operand scales more than 127x apart.
 * [HW sweep, H96 MAX M9, tests/rk3576_residual_add.c] */
int rocket_residual_add_weights_rk3576(unsigned c, float a_scale, float b_scale,
                                       int a_zp, int b_zp, int8_t *W, int32_t *bias,
                                       float *w_scale, int *w_pair);

/* ---- the FUSED residual add, and the quantization that refuses it ----------
 * A block's last convolution can absorb its own skip instead of paying a program for it:
 * take `c` more input channels, put an identity block at the CENTRE TAP of the kernel, and
 * the skip is part of a convolution the network was already running. The datapath computes
 * it — bit-exact at MobileNetV2's project convolution for ic 168-1120 and at ResNet-18's
 * second convolution for ic 128-512 [HW sweep, H96 MAX M9, tests/rk3576_residual_add.c
 * `fused`] — and it is the hardware's own idiom, since the vendor compiler folds
 * `Add(Conv(x), x)` into exactly this shape.
 *
 * WHAT REFUSES IT IS THE IDENTITY WEIGHT, NOT THE SLICE RULE. A standalone add gets to
 * CHOOSE its weight scale, because its "weights" are not a quantized tensor — which is how
 * the operand ratio fits in a pair of int8s. A fused one INHERITS the convolution's, and
 * the convolution's is fixed by its own filter. So the skip's contribution has to reach the
 * output through the accumulator unit the main term already uses:
 *
 *     w_skip = skip_scale / (in_scale * w_scale)          and it must be an int8
 *
 * Since the skip and the sum have comparable ranges, that is about `1/gain` with `gain` the
 * convolution's own requant gain — so the fusion needs a gain above about 1/127, which is a
 * SHALLOW contraction. A project convolution over 576-960 channels is nowhere near it:
 * MobileNetV2's ten residual blocks want 173 to 865 and NOT ONE of the ten fits.
 *
 * A per-axis weight quantization makes it strictly worse, not better — the per-tensor
 * `w_scale` is the largest of the per-channel ones, so every channel's weight grows.
 *
 * Pure: no hardware, no allocation. `w_skip` receives the identity weight and `bias_delta`
 * the per-output-channel constant to ADD to every bias — the datapath subtracts the
 * convolution's own input zero point from the skip channels and `w_skip*(in_zp - skip_zp)`
 * corrects that exactly. `rel_err` (optional) receives the identity weight's rounding error
 * relative to the exact ratio. ROCKET_E_UNSUPPORTED when the weight does not fit int8,
 * which is the answer for every real residual block measured so far. */
int rocket_residual_fuse_weight_rk3576(float in_scale, float w_scale, float skip_scale,
                                       int in_zp, int skip_zp,
                                       int *w_skip, int32_t *bias_delta, float *rel_err);

/* ---- pack-weights-once path ------------------------------------------------
 * For a layer run repeatedly with the same weights — a graph's forward pass, a video
 * stream — everything the three entries above rebuild from the WEIGHTS ALONE is packed
 * once instead of once per inference: the per-output-channel filter sums, the coefficient
 * group, the weight cube, and on the per-axis path the scale sort and the output-channel
 * tile. None of it depends on the input. On MobileNetV1-224 that work is roughly a
 * quarter of the wall, where the submits are 1-16% of a layer.
 *
 *   h = rocket_conv2d_int8_pack_rk3576(fd, &d, W, bias, in_scale, w_scale, NULL,
 *                                     out_scale, in_zp, w_zp, out_zp);
 *   ... per inference:
 *       rocket_conv2d_int8_prepacked_rk3576(fd, h, in, out);
 *   rocket_conv2d_int8_weights_free_rk3576(fd, h);
 *
 * `w_scale_oc` selects which requant the handle carries: NULL for the per-tensor contract
 * of rocket_conv2d_int8_rk3576() / rocket_conv2d_dw_int8_rk3576() (with `w_scale` the
 * per-tensor scale), non-NULL for the per-axis one of
 * rocket_conv2d_int8_perchannel_rk3576() (with `w_scale` ignored). desc.depthwise selects
 * direct or depthwise. The arithmetic is the SAME code as the per-call entries, so a
 * prepacked call is bit-identical to the transient one by construction.
 *
 * WHAT A HANDLE FREEZES, and why each is not bookkeeping:
 *
 *   THE fd. Its BOs belong to the file that created them and an IOVA is per-fd, so
 *   running a handle on another fd would hand the NPU addresses that mean nothing there
 *   and the submit would still succeed. A foreign fd is refused.
 *
 *   THE TILE PLAN. The weight cube's group count follows the output-channel TILE rather
 *   than the layer, so a cube packed for one tiling is not a cube for another. On the
 *   per-axis path that tile is also an accuracy choice made from the weights and the
 *   scales, and it is made once here.
 *
 *   THE QUANT CONTRACT. Both zero points and all three scales are folded into the
 *   coefficient group at pack time. A layer whose quantization changes needs a new handle.
 *
 * The handle also holds the scratch whose size follows the frozen geometry — the feature
 * cube, the regcmd buffer and each tile's output surface — so an inference allocates no
 * BOs at all. Its cost is memory: every tile's cube is resident at once, about the size of
 * the caller's own W, where a transient call holds one tile's.
 *
 * NOT thread-safe: one handle carries the scratch a call writes through, so concurrent
 * prepacked calls on ONE handle race. Pack per thread, or serialize. Refuses four or
 * fewer input channels on the direct path (the packed-image first conv is a different
 * program whose cube this does not pack) and returns NULL for anything the per-call
 * entries would refuse. */
typedef struct rocket_conv2d_int8_weights_rk3576 rocket_conv2d_int8_weights_rk3576;

rocket_conv2d_int8_weights_rk3576 *
rocket_conv2d_int8_pack_rk3576(int fd, const rocket_conv2d_desc *d,
                               const int8_t *W, const int32_t *bias,
                               float in_scale, float w_scale, const float *w_scale_oc,
                               float out_scale, int in_zp, int w_zp, int out_zp);

int rocket_conv2d_int8_prepacked_rk3576(int fd, rocket_conv2d_int8_weights_rk3576 *h,
                                        const int8_t *in, int8_t *out);

void rocket_conv2d_int8_weights_free_rk3576(int fd,
                                            rocket_conv2d_int8_weights_rk3576 *h);

/* ---- cube I/O between consecutive layers -----------------------------------
 * The entries above take and return ROW-MAJOR tensors, so a graph pays a host transpose
 * at both ends of every layer: the cube interleaves sixteen channels into every
 * sixteen-byte atom, and a row-major tensor and a cube are a transpose rather than a
 * copy. On MobileNetV1-224 those two buckets are the largest host cost left.
 *
 * They do not have to be paid BETWEEN layers, because a direct convolution's output
 * surface stride is `ow*oh` exactly — so layer n's output surface IS layer n+1's input
 * cube, byte for byte, whenever the channel counts round the same way. A handle can be
 * told to read its input from a producer's surface and to leave its own output in place:
 *
 *   struct rocket_rk3576_cube c;
 *   rocket_conv2d_int8_cube_out_rk3576(prod, 1);          // no de-scatter
 *   rocket_conv2d_int8_cube_of_rk3576(prod, &c);
 *   rocket_conv2d_int8_cube_in_rk3576(cons, &c);          // no feature scatter
 *   ...
 *   rocket_conv2d_int8_prepacked_rk3576(fd, prod, in, NULL);   // out may be NULL
 *   rocket_conv2d_int8_prepacked_rk3576(fd, cons, NULL, out);  // in  may be NULL
 *
 * A cube is a BORROWED view of the producer handle's output surface and is valid for
 * that handle's lifetime; freeing the producer invalidates it. Nothing here changes
 * what is computed — the surfaces are the same bytes the row-major entries transpose. */
typedef struct rocket_rk3576_cube {
    int       fd;          /* the file the buffer belongs to                        */
    unsigned  c;           /* live channels                                          */
    unsigned  h, w;        /* the plane                                              */
    unsigned  groups;      /* channel groups of 16 this cube carries                 */
    size_t    surf_elems;  /* elements per channel group; == h*w for a direct conv    */
    rocket_bo bo;          /* borrowed; owned by the producer handle or the caller   */
    size_t    off;         /* byte offset of this cube's first channel group in `bo` */
    /* THE CONSTANT TAIL. A consumer whose channel count is not a multiple of 32 walks
     * groups past the live ones, and at a non-zero weight zero point their CONTENT is
     * load-bearing — the coefficient group's B term sums every channel of the programmed
     * group. `pad_from` non-zero says channels [pad_from, groups*16) all hold `pad_value`,
     * which is what lets such a join be accepted instead of refused. Zero means the tail
     * is whatever wrote there, and the join is refused at a non-zero weight zero point. */
    unsigned  pad_from;    /* first channel of the constant tail; 0 = not declared    */
    int       pad_value;   /* what channels [pad_from, groups*16) hold                */
} rocket_rk3576_cube;

/* ---- a cube that is a SLICE of a larger buffer ------------------------------
 * A cube's base is a plain ADDRESS on both sides of a convolution: a program whose feature
 * or output base is moved by G channel groups inside a bigger buffer computes exactly what
 * it computes at offset 0, writes only its own slice, and composes with the row-task offset
 * [HW sweep, H96 MAX M9, tests/rk3576_offset_cube.c — 34 placements over three geometries].
 *
 * That is what lets several producers write into ONE allocation, which is what a residual
 * topology wants. A chain of layers fragments where the host has to touch a tensor between
 * two programs, and on MobileNetV2 two thirds of those places are a channel CONCATENATION
 * the host builds: an add's two operands, and a skip a later layer reads beside a
 * convolution's own output. Written into slices of one buffer they need no host copy —
 * the two surfaces ARE the concatenated operand.
 *
 *   rocket_rk3576_cube buf, lo, hi;
 *   rocket_rk3576_cube_alloc(fd, 2*C, H, W, &buf);        // the caller owns this one
 *   rocket_rk3576_cube_slice(&buf, 0, C, &lo);
 *   rocket_rk3576_cube_slice(&buf, C, C, &hi);
 *   rocket_conv2d_int8_cube_out_at_rk3576(a, &lo);        // two producers, one buffer
 *   rocket_conv2d_int8_cube_out_at_rk3576(b, &hi);
 *   rocket_conv2d_int8_cube_in_rk3576(consumer, &buf);    // reads both halves
 *
 * The channel offset must be a multiple of 16 — a group is the unit an address can name. */
int rocket_rk3576_cube_alloc(int fd, unsigned c, unsigned h, unsigned w,
                             rocket_rk3576_cube *out);

void rocket_rk3576_cube_free(int fd, rocket_rk3576_cube *buf);

/* A view of `c` channels starting at channel `c0` of `buf`. Borrowed: the view is valid
 * while `buf` is. Refuses a `c0` that is not a multiple of 16 and a range past the end. */
int rocket_rk3576_cube_slice(const rocket_rk3576_cube *buf, unsigned c0, unsigned c,
                             rocket_rk3576_cube *out);

/* Describe this handle's output surface as a cube. Refuses a handle whose output
 * channels are split across more than one tile — each tile owns its own buffer and
 * several buffers are not one cube. Valid as soon as the handle is packed.
 *
 * The cube carries the handle's own channel-group STRIDE, which is `ow*oh` on the direct
 * path and `round4(ow*oh)` on the depthwise one; a consumer is told it rather than
 * deriving it, so a depthwise producer at a plane whose element count is not a multiple
 * of four joins like any other.
 *
 * A DIRECT handle declares its constant tail. It is programmed with the round-32 output
 * channel count, the DPU writes every group it is told, and a partial group's channels are
 * packed with zero weights and a zero A term — so they land on the OUTPUT ZERO POINT, which
 * at a graph join is the consumer's input zero point and so the border constant its own
 * cube would have used [HW sweep, H96 MAX M9, tests/rk3576_pad_channels.c]. A DEPTHWISE
 * handle is programmed with the RAW count and leaves everything past it untouched, so it
 * declares nothing unless its output channels fill the last group exactly. */
int rocket_conv2d_int8_cube_of_rk3576(const rocket_conv2d_int8_weights_rk3576 *h,
                                      rocket_rk3576_cube *out);

/* Read this handle's feature input from `src` instead of a row-major tensor, and skip the
 * scatter. `src` NULL restores the row-major input. Refuses a cube whose plane, channel
 * count or fd does not match what this handle would have packed for itself, and a
 * channel-group stride SHORTER than the plane — a longer one is programmed into the CNA's
 * DDR group jump, which the part honours verbatim [HW sweep, H96 MAX M9,
 * tests/rk3576_surf_stride.c]. The cube is borrowed, so it must outlive the calls that
 * read it.
 *
 * An input channel count that is not a multiple of 32 is refused only at a NON-ZERO weight
 * zero point, and only when the cube does not declare a constant tail that matches this
 * handle's input zero point: the padding channels' weights are zero so they never reach the
 * MAC, and the one thing that reads them is the coefficient group's B term. */
int rocket_conv2d_int8_cube_in_rk3576(rocket_conv2d_int8_weights_rk3576 *h,
                                      const rocket_rk3576_cube *src);

/* Leave the output in the cube instead of de-scattering it into the caller's tensor.
 * Requires the same single-tile shape rocket_conv2d_int8_cube_of_rk3576 does. */
int rocket_conv2d_int8_cube_out_rk3576(rocket_conv2d_int8_weights_rk3576 *h, int on);

/* Declare that this handle's output surface is read by a consumer OUTSIDE any chain the
 * handle is part of, later in the same inference — a producer two layers read, where only
 * one of them is the layer immediately after it.
 *
 * A chain re-stamps every interior layer's surface in its verify bracket, which is sound
 * only because the next kick rewrites that surface before anything reads it. A shared one
 * is read first, so stamping it replaces the layer's output with the sentinel and the
 * outside consumer computes a full and plausible surface from 0xA5 — invisible until the
 * next materialised layer. Declared, the stamp goes on at the start of this handle's next
 * call instead, which is where cube-out already puts it. Costs one PREP/FINI pair per call
 * and nothing else; a handle that writes into a caller's buffer is already this case and
 * needs no declaration. */
int rocket_conv2d_int8_cube_shared_rk3576(rocket_conv2d_int8_weights_rk3576 *h, int on);

/* Write this handle's output surface into `dst` — a caller's buffer, at `dst`'s channel-group
 * offset — instead of into the handle's own. Implies cube-out: the surface is left where the
 * DPU wrote it, and rocket_conv2d_int8_cube_of_rk3576() then describes the slice.
 *
 * The handle's own surface BO is released, so a graph pays for the shared buffer instead of
 * for both. `dst` NULL restores it (and clears cube-out), which allocates again on the next
 * call. The buffer is BORROWED and must outlive the calls that write it.
 *
 * Refuses a foreign fd, a handle with more than one output-channel tile, a plane or a
 * channel-group stride that is not this handle's, and a slice too short for its output
 * channels — every one a case where the surface and the slice are not the same object. */
int rocket_conv2d_int8_cube_out_at_rk3576(rocket_conv2d_int8_weights_rk3576 *h,
                                          const rocket_rk3576_cube *dst);

/* ---- a run of cube-linked layers as ONE hardware kick -----------------------
 * The per-submit floor on this part is ~439 us and a resident MobileNetV1-224 layer costs
 * 130-250 us of compute, so once the host transposes are gone the SUBMIT is the largest
 * single bucket: 29 of them for 29 layers. They do not all have to be separate jobs.
 *
 * A chained stream HONOURS READ-AFTER-WRITE between its programs — measured over five
 * planes from 2 KB to 387 KB, where the second program's feature cube is the first's output
 * surface (tests/rk3576_chain_raw.c) — so a run of handles can be laid out as one contiguous
 * regcmd stream and submitted as ONE batched job. What makes a run eligible is that no host
 * work is left between its programs, which is two statements and NOT a pairwise one: every
 * handle but the last leaves a cube, and every handle but the first reads a cube that some
 * EARLIER handle of the run wrote — not necessarily the one immediately before it. Read
 * after write across the stream is the part's own guarantee, so a shortcut convolution that
 * takes its input several layers back stays in the stream.
 *
 *   rocket_conv2d_int8_chain_rk3576 *c = rocket_conv2d_int8_chain_new_rk3576(fd, h, n);
 *   for (...) rocket_conv2d_int8_chain_run_rk3576(fd, c, in, out);
 *   rocket_conv2d_int8_chain_free_rk3576(fd, c);
 *
 * `in` is the FIRST handle's row-major input (NULL if it reads a cube from outside the
 * chain) and `out` the LAST handle's row-major output (NULL if it leaves a cube). The
 * constructor asserts the two run conditions above rather than assuming them.
 *
 * The handles are BORROWED and must outlive the chain; they stay usable one at a time
 * through rocket_conv2d_int8_prepacked_rk3576(), which is what the chain falls back to if a
 * program has to be redone. Nothing here changes what is computed.
 *
 * REFUSED, each because the chain would otherwise be silently wrong or unbuildable: fewer
 * than two handles; a handle with more than one output-channel tile; a handle with a DPU
 * LUT (its table load is a task in a BO of its own, which this layout does not carry); a
 * non-last handle that does not leave a cube; a non-first handle whose input no earlier
 * handle of the run writes; a foreign fd; and a kernel that does not honour
 * DRM_ROCKET_JOB_BATCHED. */
typedef struct rocket_conv2d_int8_chain_rk3576 rocket_conv2d_int8_chain_rk3576;

rocket_conv2d_int8_chain_rk3576 *
rocket_conv2d_int8_chain_new_rk3576(int fd,
                                    rocket_conv2d_int8_weights_rk3576 *const *h,
                                    unsigned n);

int rocket_conv2d_int8_chain_run_rk3576(int fd, rocket_conv2d_int8_chain_rk3576 *c,
                                        const int8_t *in, int8_t *out);

/* How many hardware kicks the last run took. One unless a program had to be redone, in
 * which case the whole chain is retried; a caller measuring the lever reads this rather
 * than assuming. */
unsigned rocket_conv2d_int8_chain_kicks_rk3576(const rocket_conv2d_int8_chain_rk3576 *c);

void rocket_conv2d_int8_chain_free_rk3576(int fd, rocket_conv2d_int8_chain_rk3576 *c);

/* ---- finding the runs -------------------------------------------------------
 * The constructor above takes ONE run and asserts it. A caller with a graph has to find
 * the runs first, and that part needs no hardware knowledge — so it lives here rather
 * than being written again in every frontend. The library keeps the primitive and gains
 * a finder; it does NOT gain a graph-execution object, because scheduling is the
 * frontend's and the only thing it cannot own is which streams the part will run.
 *
 *   rocket_conv2d_int8_run_rk3576 runs[16];
 *   unsigned n = rocket_conv2d_int8_chain_plan_rk3576(handles, n_layers, runs, 16);
 *   for (i = 0; i < n && i < 16; i++)
 *       chain[i] = rocket_conv2d_int8_chain_new_rk3576(fd, handles + runs[i].first,
 *                                                      runs[i].count);
 *
 * A run is a maximal sequence of handles in which every layer but the last leaves a cube
 * and every layer but the first reads a cube an EARLIER member wrote, asserted against the
 * BO addresses. A NULL entry breaks a run: that is how a caller says "this
 * layer is not a resident convolution" or "a run may not START here" — a chain scatters
 * the tensor it is handed, so a layer whose input the caller prepares on the host (a
 * widened or bordered copy) must not begin one.
 *
 * The program-count bound is applied by the finder, so a run too long is SPLIT at a layer
 * boundary rather than refused. Pure: no hardware, no allocation, and nothing on the
 * handles is written. Returns the number of runs of two or more layers, which may exceed
 * `max_runs`; only the first `max_runs` are written. */
typedef struct {
    unsigned first;      /* index into the caller's handle array */
    unsigned count;      /* layers in the run                    */
    unsigned programs;   /* register programs those layers emit  */
} rocket_conv2d_int8_run_rk3576;

unsigned rocket_conv2d_int8_chain_plan_rk3576(rocket_conv2d_int8_weights_rk3576 *const *h,
                                              unsigned n,
                                              rocket_conv2d_int8_run_rk3576 *runs,
                                              unsigned max_runs);

/* How many register programs this handle emits — its row plan's task count, which is what
 * the chain's length bound counts. 0 for a handle that cannot be chained at all. */
unsigned rocket_conv2d_int8_programs_rk3576(const rocket_conv2d_int8_weights_rk3576 *h);

/* ---- a POOLING layer inside the stream --------------------------------------
 * A classifier puts a pooling layer in the middle of a run, and a chain of convolution
 * handles has to break at one. It does not have to: a pool is its own register program
 * (PPU and PPU_RDMA, PC_OPERATION_ENABLE 0x60 against a convolution's 0x1D) and the two
 * bitmaps being disjoint turns out not to partition a stream — 20 of 20 iterations over
 * five geometries to a 110x110 plane gave the same pool output AND the same output from
 * the convolution after it as separate submits, both intermediates read so a failure would
 * have named its boundary [HW sweep, tests/rk3576_chain_pool.c].
 *
 * So a run is described by NODES rather than by convolution handles. Exactly one of the
 * two pointers is set; both NULL is the "this layer cannot be in a run" break that a NULL
 * entry means in the convolution-only form.
 *
 *   rocket_chain_node_rk3576 nodes[N] = {0};
 *   nodes[i].conv = conv_handle;      // or
 *   nodes[j].pool = pool_handle;
 *   unsigned n = rocket_chain_plan_rk3576(nodes, N, runs, 16);
 *   chain[k] = rocket_chain_new_rk3576(fd, nodes + runs[k].first, runs[k].count);
 *
 * A pool may only be an INTERIOR node. One that began a run would need the chain to
 * scatter a row-major tensor into its cube and one that ended a run would need its output
 * de-scattered, and both are host work the stream exists to remove; the run finder
 * therefore never starts or ends a run on a pool, and the constructor refuses one.
 *
 * The convolution-only entries above are these with every node a `conv`, and are kept
 * because most callers have no pooling layer to place. There is one implementation. */
typedef struct {
    rocket_conv2d_int8_weights_rk3576 *conv;
    rocket_pool_int8_rk3576_handle    *pool;
} rocket_chain_node_rk3576;

rocket_conv2d_int8_chain_rk3576 *
rocket_chain_new_rk3576(int fd, const rocket_chain_node_rk3576 *nodes, unsigned n);

unsigned rocket_chain_plan_rk3576(const rocket_chain_node_rk3576 *nodes, unsigned n,
                                  rocket_conv2d_int8_run_rk3576 *runs, unsigned max_runs);

/* How many register programs this node emits. 0 for a node that cannot be chained. */
unsigned rocket_chain_node_programs_rk3576(const rocket_chain_node_rk3576 *node);

/* The longest program stream a chain will accept. What bounds it is the KERNEL'S
 * completion deadline, not the part: a batched job is one PC_DONE for the whole kick, and
 * a driver before interface 1.4 gives it one task's deadline — past which the job retires
 * with its last program's writes still in flight. On 1.4 and later that deadline scales and
 * this is simply the largest stream the library will lay out.
 * ROCKET_RK3576_CHAIN_MAX_TASKS moves it. */
unsigned rocket_conv2d_int8_chain_max_programs_rk3576(void);

/* fp16 -> fp16 through the input-channel split: one fp16 task on this part contracts
 * exactly sixteen input channels, so an arbitrary count is ic/16 submits summed on the
 * host. A plane whose 16-channel slice still overflows the CBUF is refused — composing
 * the split with the row window is not wired — and the depthwise fp16 cube is not
 * decoded, so desc.depthwise is refused too. */
int rocket_conv2d_fp16_rk3576(int fd, const rocket_conv2d_desc *d,
                              const _Float16 *in, const _Float16 *W, _Float16 *out);

#ifdef __cplusplus
}
#endif
#endif /* ROCKET_CONV_H */
