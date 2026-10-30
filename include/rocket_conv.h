// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_CONV_H
#define ROCKET_CONV_H

#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_pool.h"   /* a pooling layer can sit inside a cross-layer chain */

/*
 * Declares the 2D convolution (CONV_2D) entries, direct and depthwise, in fp16 and int8.
 * The RK3588 entries also include a 1D convolution, the transposed convolution and
 * resident-weight forms. The RK3576 entries follow them, with packed weights, cube I/O
 * between layers and chained runs of layers.
 *
 * The CNA is a native convolution engine. It computes the KxK, stride, pad and dilation
 * sliding-window MAC in hardware, so the direct entries run without im2col on a matmul. The
 * host scatters the input feature into the NC1HWC2 cube and the weights into the conv weight
 * cube. The matmul path is the degenerate 1x1 case.
 *
 * Tensor layouts (row-major, batch 1, NCHW-style):
 *
 *     input    in [IC][IH][IW]
 *     weights  W  [OC][IC][KH][KW]   (direct);  [OC][1][KH][KW]  (depthwise, OC==IC)
 *     output   out[OC][OH][OW]
 *
 *     out[oc][oh][ow] = sum_{ic,kh,kw} W[oc][ic][kh][kw] *
 *         in[ic][oh*stride_y + kh*dil_y - pad_top][ow*stride_x + kw*dil_x - pad_left]
 *     (out-of-range input contributes 0 — zero padding).
 *
 * The direct path accepts any `IC` and `OC`. The library zero-pads `OC` up to a multiple of
 * 16 (OC%16==0, the conv weight oc group) and `IC` up to a multiple of 32. An `IC` < 32,
 * such as the RGB first layer's `IC`=3, is zero-padded up to 32. A depthwise descriptor must
 * have `OC`==`IC`, and `IC` must be a multiple of `G` (IC%G==0), the depthwise channel
 * group. `ROCKET_CONV_DW_GROUP` sets `G`, default 32.
 *
 * The direct and the depthwise paths are bit-exact on hardware
 * (`tests/conv2d_fp16_rocket.c`). When a problem does not fit one CBUF pass, the direct path
 * tiles automatically. It tiles over output channels and output space, OH rows and then OW
 * columns, and each tile is an independent task.
 *
 * Depthwise tiles over channels, in chunks of Cc channels, a multiple of `G`, that each fit
 * one pass. A `G`-channel chunk whose feature is still too large for one pass is tiled over
 * space as well. For a depthwise descriptor, `rocket_conv2d_plan` returns a negative value
 * in two cases only. Either one `G`-channel weight cube does not fit a bank, or the minimal
 * spatial tile (`G` channels, one output row by one column) does not fit the feature budget.
 */

/* Returns the output extent along one axis. `dil` is the dilation rate (1 = none). */

#ifdef __cplusplus
extern "C" {
#endif
static inline int rocket_conv_out_dim(int in, int k, int stride, int pad, int dil)
{
    return (in + 2 * pad - (dil * (k - 1) + 1)) / stride + 1;
}

/* A convolution descriptor: the shapes and hyperparameters, and no buffers. */
typedef struct {
    int ic, ih, iw;     /* input  channels / height / width  */
    int oc;             /* output channels                   */
    int kh, kw;         /* kernel height / width             */
    int stride_y, stride_x;
    int pad_top, pad_left;
    int dil_y, dil_x;   /* dilation rate (1 = none)          */
    int depthwise;      /* 0 = direct conv, 1 = depthwise (OC==IC) */
    /* An explicit output extent, which is how a caller expresses an asymmetric pad. The
     * RK3576 CNA takes no trailing pad. It derives the pad its last window consumes from the
     * output extent and the leading pad. TFLite's SAME at an even input and stride two
     * (pad_before 0, pad_after 1) is therefore zero `pad_top` and `pad_left`, with an output
     * one larger than the symmetric formula's. The caller's buffer needs no materialized
     * border.
     *
     * Zero means the extent `rocket_conv2d_oh` and `rocket_conv2d_ow` derive, which is what
     * a caller that memsets this struct gets. The RK3576 int8 entries accept an explicit
     * extent whose derived trailing pad is less than the kernel. The fp16 entries and the
     * RK3588 entries refuse an extent they did not derive. */
    int oh, ow;
    /* Runs a narrow input-channel count on the direct datapath instead of the RK3576's
     * packed-image first-conv sub-encoding.
     *
     * The RK3576 CNA can read an image of four or fewer channels in two ways. Its
     * packed-image sub-encoding interleaves the channels per pixel. That sub-encoding has
     * three silent geometry bounds: a non-zero left pad, an output width of exactly
     * iw/stride, and that width a multiple of 16. The ordinary direct datapath's int8 cube
     * is a 32-channel MAC group whatever the count. `ic` 3 and `ic` 32 are therefore the
     * same register program, with the same weight-cube and feature-cube sizes. The channels
     * past `ic` are the cube's own zero padding.
     *
     * The zero-point fold counts only the live taps, so the direct datapath is exact at any
     * zero point. The direct lowering of a three-channel stem programs a 32-channel group,
     * so the packed form programs fewer MACs. At a 224x224 k7 s2 stem it programs 8.0x fewer
     * MACs and takes 7.2x less execution and drain time, a measured 1.566 ms. Its cost is
     * the host image pack, which moves ~10x fewer bytes at ~10x worse throughput.
     *
     * A non-zero input zero point is not one of the packed form's bounds. The int8 entries
     * materialize its pad columns. The image is programmed one 16-output granule wider, with
     * the extension at the input zero point. The de-tile crops the surplus output columns.
     * The materialized form is bit-exact against the direct lowering over kernels 3/5/7,
     * both strides, model pads 1/2/3 and both signs of zero point. The descriptor and both
     * buffers keep the caller's own geometry, and the cost is the surplus columns' MACs and
     * a row-wise de-tile.
     *
     * Note: with materialized pad columns, the output is wider than the caller's plane. A
     * plane's row stride is implied by `iw` with no register to move it. Such a layer
     * therefore cannot leave a cube for the next layer.
     *
     * Use the packed form where its bounds are met. The direct path is the general one: it
     * has no geometry bound, widens nothing on the host, and its output can be a consumer's
     * cube. With the flag set, `rocket_conv2d_int8_rk3576()`, `_pack_rk3576()`,
     * `_perchannel_rk3576()` and `_act_rk3576()` take the direct path at any `ic`. With the
     * flag clear, which is what a caller that memsets this struct gets, `ic` <= 4 takes the
     * packed-image path.
     *
     * At `ic` <= 4 with the flag clear, the per-axis and fused-activation entries refuse the
     * descriptor, because neither is supported on the packed-image encoding. The RK3576 fp16
     * entry refuses the flag: its direct path has a different channel granularity and is not
     * tested at `ic` <= 4. The RK3588 entries never take a packed-image encoding, so the
     * flag already holds there and they accept it. */
    int direct_datapath;
} rocket_conv2d_desc;

/* Returns the output height and width a descriptor implies. Pure. An explicit `d->oh` or
 * `d->ow` takes precedence over the derived extent. */
static inline int rocket_conv2d_oh(const rocket_conv2d_desc *d)
{ return d->oh ? d->oh
               : rocket_conv_out_dim(d->ih, d->kh, d->stride_y, d->pad_top, d->dil_y); }
static inline int rocket_conv2d_ow(const rocket_conv2d_desc *d)
{ return d->ow ? d->ow
               : rocket_conv_out_dim(d->iw, d->kw, d->stride_x, d->pad_left, d->dil_x); }

/* Returns the pad the last window consumes past the plane, which is the pad the CNA
 * programs. It is zero for a symmetric descriptor, and the trailing half for TFLite's
 * asymmetric SAME. */
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

/* Checks a descriptor against the supported set: the alignment, and the CBUF fit of a single
 * tile. Pure, with no hardware. Returns 0 if the descriptor runs, and a negated reason
 * otherwise.
 *
 * Every RK3588 conv task is at most 1022 input rows by 2047 columns. The CNA holds a task's
 * input width in 11 bits and its row count plus one in 10. A wrapped field produces a
 * complete output of the right size with wrong values. The tilers split any larger plane
 * into bands within both limits. The plan refuses (-4) only a kernel whose own dilated
 * extent passes one of them.
 *
 * Evidence: `tests/conv_width_gate` tests planes past each limit. */
int rocket_conv2d_plan(const rocket_conv2d_desc *d);

/* Runs the convolution on the NPU. `fd` is an open `rocket` device (`rocket_open()`). `in`,
 * `W` and `out` are row-major fp16 in the layouts above. Returns 0 on success and a negative
 * value on error. */
int rocket_conv2d_fp16(int fd, const rocket_conv2d_desc *d,
                       const _Float16 *in, const _Float16 *W, _Float16 *out);

/* ---- conv1d: the Whisper encoder front end --------------------------------------
 *
 * Computes a 1D convolution over the time axis. The entry lowers it onto
 * `rocket_conv2d_fp16` as a width-1 conv (IW=KW=OW=1), with the time axis on the height (IH)
 * axis. The tiler splits that axis into row bands of at most 1022 input rows.
 *
 *     in  [IC][IT]            (IC channels, IT time steps)
 *     W   [OC][IC][KW]        (OC filters, kernel width KW)
 *     out [OC][OT]            OT = rocket_conv_out_dim(IT, KW, stride, pad, 1)
 *
 *     out[oc][t] = sum_{ic,kw} W[oc][ic][kw] * in[ic][t*stride + kw - pad]   (zero-padded)
 *
 * Whisper's two front-end convs are this op (KW=3, pad=1, conv1 at stride 1 and conv2 at
 * stride 2), each followed by GELU. For the GELU, use the conv->act epilogue rather than the
 * standalone op. The entry is a pure descriptor wrapper, so it inherits conv2d's
 * bit-exactness, alignment and tiling. The matching CPU reference is
 * `rocket_conv2d_ref_fp16` on the same width-1 descriptor. Returns 0 on success and a
 * negative value on error. */
int rocket_conv1d_fp16(int fd, int ic, int it, int oc, int kw, int stride, int pad,
                       const _Float16 *in, const _Float16 *W, _Float16 *out);

/* ---- the resident-BO conv context -------------------------------------------------
 *
 * Caches a convolution's NPU BOs across calls. `rocket_conv2d_fp16` allocates and frees its
 * NPU BOs (input, weight, regcmd, output and an IOVA guard) on every call, and per tile for
 * a tiled conv. For a delegate that runs many convs per inference, each possibly tiled, that
 * allocation dominates the cost of a small conv.
 *
 * A `rocket_conv_ctx` keeps those BOs on a borrowed `fd`, one per power-of-two size class
 * from 64 KiB for each role, and repeat calls reuse them. A job syncs its BOs whole, and
 * never more than twice the bytes it uses. Every job still clears and refills its BOs, so
 * reuse is bit-identical to a fresh allocation.
 *
 *     ctx = rocket_conv_ctx_create(fd);            // borrows fd (does NOT close it)
 *     ... per inference, any supported conv shape:
 *         rocket_conv2d_fp16_ctx(ctx, &desc, in, W, out);
 *     rocket_conv_ctx_free(ctx);                   // frees the cached BOs (not the fd)
 *
 * The context never opens or closes `fd`, and the caller is responsible for it. A context
 * with `fd` < 0 is accepted and computes every call with the CPU reference, before it
 * touches any BO. One context can serve convs of different shapes. Each role holds at most
 * about twice its largest job's bytes over its size classes. */
typedef struct rocket_conv_ctx rocket_conv_ctx;

rocket_conv_ctx *rocket_conv_ctx_create(int fd);
void             rocket_conv_ctx_free(rocket_conv_ctx *ctx);

int rocket_conv2d_fp16_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                           const _Float16 *in, const _Float16 *W, _Float16 *out);

/* ---- conv -> activation fusion ----------------------------------------------------
 *
 * Computes `out = f(conv(...))`: the convolution of `rocket_conv2d_fp16` with a smooth
 * activation `f` applied in the same NPU job. The DPU LUT epilogue applies it as a BN-mul
 * index scale, then the EW LUT, then an affine OUT_CVT. There is no second NPU round trip
 * and no host activation pass. The entry takes a direct fp16 conv only.
 *
 * `kind` is `ROCKET_ACTIVATION_SILU`, `ROCKET_ACTIVATION_TANH` or `ROCKET_ACTIVATION_GELU`
 * (`rocket_activation.h`). The entry refuses HardSwish, which runs on the host or in two
 * passes only, and a depthwise descriptor. Returns 0 on success, -10 for a depthwise
 * descriptor, -11 for an unsupported `kind`, and another negative value on any other error.
 *
 * The fused `f` is an fp16 LUT approximation, so a test must compare its output with a
 * tolerance, not for equality. */
int rocket_conv2d_act_fp16(int fd, const rocket_conv2d_desc *d, int kind,
                           const _Float16 *in, const _Float16 *W, _Float16 *out);
int rocket_conv2d_act_fp16_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d, int kind,
                               const _Float16 *in, const _Float16 *W, _Float16 *out);

/* Computes the CPU reference, accumulating in fp32, which also serves as a host fallback.
 * Writes `out[OC*OH*OW]` as fp16 (fp32 accumulate, fp16 store), matching the NPU's
 * fp16-narrowed output. Pure host, with no hardware. */
void rocket_conv2d_ref_fp16(const rocket_conv2d_desc *d,
                            const _Float16 *in, const _Float16 *W, _Float16 *out);

/* ---- native int8 CONV_2D, int32 output ----------------------------------------------
 *
 * Computes the exact W8A8 direct convolution, the int8 sibling of `rocket_conv2d_fp16`. The
 * CNA reduces int8 features by int8 weights natively into an int32 accumulator, written raw
 * to DRAM. The caller requantizes `C[oc]` by the per-output-channel weight scale times the
 * per-tensor activation scale, folding any zero-point correction into the bias. There is no
 * host dequant and requant round trip, and the int8 semantics are exact. The result is
 * bit-identical to TFLite's int8 CPU accumulate, apart from the final requant rounding.
 *
 * `in` is [IC][IH][IW] int8, `W` is [OC][IC][KH][KW] int8 and `out` is [OC][OH][OW] int32.
 * Any `OC` and `IC` are accepted. The library zero-pads `OC` to the int8 weight oc-group
 * (32) and `IC` to 32, including an RGB first layer with `IC` < 32. Like the fp16 path, the
 * entry tiles over `OC` in multiples of 32, OH rows and OW columns. Each tile is an
 * independent task, and `desc.depthwise` must be 0, because the int8 depthwise conv is the
 * separate on-chip-requant entry `rocket_conv2d_dw_int8`. Returns 0 on success and a
 * negative value on error.
 *
 * Evidence: the register program (`gen_conv2d_int8`) and the cube layouts are bit-exact on
 * hardware (`tests/conv2d_int8_rocket.c`).
 */
int rocket_conv2d_int8(int fd, const rocket_conv2d_desc *d,
                       const int8_t *in, const int8_t *W, int32_t *out);
int rocket_conv2d_int8_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                           const int8_t *in, const int8_t *W, int32_t *out);

/* ---- multicore worker pool for the native int8/uint8 direct conv -------------------
 *
 * Runs the int8 direct conv's tiles across the three NPU cores. The single-fd conv2d places
 * its independent OC, OH and OW tiles on one of the 3 NPU cores. One fd is one drm_sched
 * entity, and so one core, while it has queued work. A pool of N worker fds, each with its
 * own resident `rocket_conv_ctx`, spreads the tiles across all 3 cores. It keeps the
 * resident BOs, with no per-call allocation.
 *
 * Create one pool per delegate partition and reuse it across ops and inferences. The pool
 * opens and closes its own fds. `rocket_conv2d_int8_mt` is bit-identical to
 * `rocket_conv2d_int8`, with the same tiles and the same tasks, and runs a single-tile conv
 * serially. `nthreads` is clamped to [1,8], and the pool degrades to however many fds
 * opened.
 */
typedef struct rocket_conv_pool rocket_conv_pool;
rocket_conv_pool *rocket_conv_pool_create(int nthreads);
void              rocket_conv_pool_free(rocket_conv_pool *pool);
int rocket_conv2d_int8_mt(rocket_conv_pool *pool, const rocket_conv2d_desc *d,
                          const int8_t *in, const int8_t *W, int32_t *out);

/* ---- native int8 direct CONV_2D, int8 output, RK3588 ---------------------------------
 *
 * Computes TFLite's quantized direct conv with per-tensor quantization, requantized on chip
 * to int8. No int32 accumulator is read back and no host requant runs. The register program
 * is Mesa's direct int8-output writer, which runs these stages in order:
 *
 * - The CNA and CORE program of `rocket_conv2d_int8`
 * - The folded per-OC bias in the DPU's BS stage
 * - CPEND for the weight zero point
 * - The OUT_CVT multiplier and shift
 *
 * The entry computes `acc = bias + sum (x - in_zp)(w - w_zp)`, where a padded tap
 * contributes nothing, and then `out = sat8(rne(acc * MUL >> SHIFT) + out_zp)`. `MUL` and
 * `SHIFT` are the OUT_CVT pair derived from `in_scale*w_scale/out_scale`, a 15-bit
 * multiplier with ties to even. The accumulator is exactly TFLite's. The requant is the
 * OUT_CVT's, so an output can differ from TFLite's by one where it sits at a rounding
 * boundary.
 *
 * Any weight zero point in the int8 range is accepted. The DPU's CPEND operand adds `-w_zp`
 * times each window's input sum on chip, border pad included.
 *
 * Layouts and alignment are those of `rocket_conv2d_int8`: `in` [IC][IH][IW], `W`
 * [OC][IC][KH][KW] and `out` [OC][OH][OW], all int8 in the model's int8 domain. `bias` is
 * [OC] int32 or NULL. A uint8 tensor maps onto the entry by subtracting 128 from its values
 * and its zero point, and adding it back to the output. `OC` is zero-padded to 32 and `IC`
 * to 32 inside, so every job runs whole 32-kernel groups. The entry takes the same shapes as
 * `rocket_conv2d_int8` and the same tiles, from one shared planner. Spatial tiles
 * materialize their halo as `in_zp` on the real channels.
 *
 * `rocket_conv2d_int8_q_plan()` is the pure shape check. It returns `ROCKET_OK`,
 * `ROCKET_E_SHAPE`, or `ROCKET_E_UNSUPPORTED` on any SoC but the RK3588. The entries return
 * 0, or a negative status. `ROCKET_E_UNSUPPORTED` is for a zero point outside int8, a
 * requant scale the OUT_CVT cannot program, a folded bias outside int32, or another SoC.
 *
 * Evidence: [HW sweep, RK1, tests/conv_i8out_probe.c, and tests/conv2d_int8_q_rocket.c
 * against a host model of that arithmetic].
 */
int rocket_conv2d_int8_q_plan(const rocket_conv2d_desc *d);
int rocket_conv2d_int8_q(int fd, const rocket_conv2d_desc *d,
                         const int8_t *in, const int8_t *W, const int32_t *bias,
                         float in_scale, float w_scale, float out_scale,
                         int in_zp, int w_zp, int out_zp, int8_t *out);
int rocket_conv2d_int8_q_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                             const int8_t *in, const int8_t *W, const int32_t *bias,
                             float in_scale, float w_scale, float out_scale,
                             int in_zp, int w_zp, int out_zp, int8_t *out);
int rocket_conv2d_int8_q_mt(rocket_conv_pool *pool, const rocket_conv2d_desc *d,
                            const int8_t *in, const int8_t *W, const int32_t *bias,
                            float in_scale, float w_scale, float out_scale,
                            int in_zp, int w_zp, int out_zp, int8_t *out);

/* ---- per-channel int8 direct CONV_2D, int8 output, RK3588 ----------------------------
 *
 * Computes TFLite's int8 direct conv with a per-output-channel weight scale, the
 * quantization TFLite gives a conv filter by default. The program, layouts, tiles and
 * zero-point handling are those of `rocket_conv2d_int8_q`. `w_scale` is [OC], and the
 * filter is symmetric, since a per-axis quantization has no weight zero point.
 *
 * The requant is split across two stages of the DPU, as in `rocket_conv2d_dw_int8_perc`.
 * The BS stage multiplies channel c's biased accumulator by an int16 `C[c]` and shifts. The
 * OUT_CVT then applies one gain per job:
 *
 *     sat8(rne(sat32(rne(acc*C[c] >> s)) * MUL >> SHIFT) + out_zp)
 *
 * The channels are sorted by scale and cut into jobs where a channel's `C` would fall under
 * `ROCKET_DW_PERC_MIN_C` (default 2048, a gain within 2^-12). Every job runs whole 32-kernel
 * groups, and the jobs' tiles fan out across a pool together. An output can therefore
 * differ from TFLite's by one where it sits near a rounding boundary. A layer whose scales
 * spread past 16x runs as more jobs. The host writes a channel with an all-zero filter with
 * TFLite's arithmetic.
 *
 * `rocket_conv2d_int8_q_perc_plan()` is the pure shape check, with the same answer as
 * `rocket_conv2d_int8_q_plan`. The entries return 0, or a negative `rocket_status`.
 * `ROCKET_E_UNSUPPORTED` is for a zero point outside int8, a job gain the OUT_CVT cannot
 * program, a folded bias outside int32, or another SoC.
 */
int rocket_conv2d_int8_q_perc_plan(const rocket_conv2d_desc *d);
int rocket_conv2d_int8_q_perc(int fd, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *W, const int32_t *bias,
                              float in_scale, const float *w_scale, float out_scale,
                              int in_zp, int out_zp, int8_t *out);
int rocket_conv2d_int8_q_perc_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                                  const int8_t *in, const int8_t *W, const int32_t *bias,
                                  float in_scale, const float *w_scale, float out_scale,
                                  int in_zp, int out_zp, int8_t *out);
int rocket_conv2d_int8_q_perc_mt(rocket_conv_pool *pool, const rocket_conv2d_desc *d,
                                 const int8_t *in, const int8_t *W, const int32_t *bias,
                                 float in_scale, const float *w_scale, float out_scale,
                                 int in_zp, int out_zp, int8_t *out);

/* Computes the CPU int64-accumulate -> int32 reference for the native int8 direct and
 * depthwise conv, which is also the host fallback at `fd`<0. The accumulate is int64 because
 * a 7x7x512 conv sums past int32, and the store is int32. Pure host, with no hardware. */
void rocket_conv2d_ref_int8(const rocket_conv2d_desc *d,
                            const int8_t *in, const int8_t *W, int32_t *out);

/* ---- native int8 depthwise CONV_2D, int8 output ---------------------------------------
 *
 * Computes TFLite's int8 depthwise conv: int8 input by int8 weight, reduced per channel and
 * requantized on chip to int8, with no int32 readback. The register program is Mesa's
 * int8-output depthwise program. The accumulator is exactly TFLite's, where a padded tap
 * contributes nothing. The requant is the DPU's OUT_CVT multiplier and shift, so an output
 * can differ from TFLite's by one where it sits at a rounding boundary.
 *
 * The quantization is per-tensor, with any weight zero point in the int8 range. The input
 * zero point folds into the bias. A weight zero point cannot fold, because it adds `-w_zp`
 * times each window's input sum. The DPU's CPEND operand (DPU_BS_OW_OP) adds exactly that
 * term on chip, border pad included. An asymmetric per-tensor filter therefore computes the
 * same function [HW sweep, RK1, tests/cpend_wzp_probe.c and tests/conv_dw_int8_runtime.c]. A
 * per-channel depthwise filter takes `rocket_conv2d_dw_int8_perc`.
 *
 * The caller passes the raw model tensors. `in`, `w` and `out` are int8 in the model domain,
 * [C][IH][IW], [C][KH][KW] and [C][OH][OW]. `bias` is the TFLite int32 bias [C], or NULL for
 * no bias, and the scales and zero points are per-tensor. A uint8 tensor maps onto this
 * entry by subtracting 128 from its values and its zero point. Returns 0 on success and a
 * negative value on error.
 *
 * The entry tiles over channels like the fp16 depthwise path. A plane whose one 64-channel
 * group is past a CBUF pass runs in row bands, and in column bands if needed. Each band's
 * halo is materialized as the input zero point, which is what the hardware pad writes, so
 * the bands compute the same bytes.
 *
 * `rocket_conv2d_dw_int8_plan()` is the pure shape check behind the entry. It returns
 * `ROCKET_OK` when the entry's NPU path takes `d`, and `ROCKET_E_SHAPE` when it refuses the
 * geometry. It returns `ROCKET_E_UNSUPPORTED` on an SoC with no program for it. It opens no
 * device, so a frontend can claim a node with it rather than copy the limits. On the RK3576
 * it answers through `rocket_conv2d_int8_plan_rk3576()`.
 *
 * Evidence: [HW sweep, RK1, tests/conv_dw_int8_runtime.c]. The entry is bit-exact against a
 * host model of that requant at five shapes with zero points at both ends of the int8 range.
 * It is within one of TFLite's reference kernels on 11 of 4096 outputs of a real model.
 */
int rocket_conv2d_dw_int8_plan(const rocket_conv2d_desc *d);
int rocket_conv2d_dw_int8(int fd, const rocket_conv2d_desc *d,
                          const int8_t *in, const int8_t *w, const int32_t *bias,
                          float in_scale, float w_scale, float out_scale,
                          int in_zp, int w_zp, int out_zp, int8_t *out);
int rocket_conv2d_dw_int8_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *w, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out);

/* ---- per-channel int8 depthwise, RK3588 -----------------------------------------------
 *
 * Computes TFLite's int8 depthwise conv with a per-output-channel weight scale, the
 * quantization TFLite gives a depthwise filter by default. The layouts, zero points,
 * fused-pad handling and shape limits are those of `rocket_conv2d_dw_int8`.
 * `rocket_conv2d_dw_int8_perc_plan()` is the pure check, RK3588 only. `w_scale` is [C], and
 * the filter is symmetric, since a per-axis quantization has no weight zero point.
 *
 * The requant is split across two stages of the DPU. The BS stage multiplies channel c's
 * accumulator by an int16 `C[c]` read from the bias cube and shifts. The OUT_CVT then
 * applies one base gain per job, with the product held wide and both shifts rounding half to
 * even [HW sweep, RK1, tests/dw_perc_probe.c]:
 *
 *     sat8(rne(sat32(rne(acc*C[c] >> s)) * MUL >> SHIFT) + out_zp)
 *
 * The job's largest channel sits at `C` = 32767, so channel c resolves its gain to
 * 0.5/C[c]. An output is within one count of TFLite where a value sits near a rounding
 * boundary, like the per-tensor entry, and is not bit-exact.
 *
 * Channels are sorted by scale and cut into jobs by count and by scale class. A job ends
 * where a channel's `C` would fall under `ROCKET_DW_PERC_MIN_C` (default 2048, a gain within
 * 2^-12). A layer whose scales spread wide therefore runs as more jobs rather than losing
 * resolution. A channel whose filter is all zero, a pruned channel, has the bias for its
 * accumulator at every output. The host writes its plane as that one value.
 *
 * Returns 0, or a negative `rocket_status`, and `ROCKET_E_UNSUPPORTED` on any SoC but the
 * RK3588.
 */
int rocket_conv2d_dw_int8_perc_plan(const rocket_conv2d_desc *d);
int rocket_conv2d_dw_int8_perc(int fd, const rocket_conv2d_desc *d,
                               const int8_t *in, const int8_t *w, const int32_t *bias,
                               float in_scale, const float *w_scale, float out_scale,
                               int in_zp, int out_zp, int8_t *out);
int rocket_conv2d_dw_int8_perc_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                                   const int8_t *in, const int8_t *w, const int32_t *bias,
                                   float in_scale, const float *w_scale, float out_scale,
                                   int in_zp, int out_zp, int8_t *out);

/* ---- transposed convolution (ConvTranspose2d, "deconvolution") ----------------------
 *
 * Computes the transpose of a strided conv: each input pixel scatter-adds a kernel-weighted
 * copy into a larger output. It is the learned upsampling of segmentation heads, of decoder,
 * super-resolution and GAN-generator blocks, and of the FPN.
 * `rocket_conv_transpose2d_route()` chooses one of two paths per descriptor:
 *
 * - The CNA's hardware deconvolution mode, for a direct conv at power-of-two strides whose
 *   compact input fits one CBUF pass. It runs one task per output-channel tile, the hardware
 *   dilates the input, and the host rotates the kernel and transposes its channels.
 * - The lowering, for everything else. The host interior-dilates the input (stride-1 zeros
 *   inserted between pixels) and pads its border, and rotates the kernel 180 deg with its in
 *   and out channels transposed. A stride-1 forward conv produces the result and inherits
 *   the direct conv's exact tiling bit for bit.
 *
 * Both paths are bit-exact against the scatter reference.
 *
 * Tensor layouts (row-major, batch 1):
 *
 *     input    in [IC][IH][IW]
 *     weights  W  [IC][OC][KH][KW]   (direct;   PyTorch/ONNX order: IN channels first)
 *              W  [C][1][KH][KW]     (depthwise, OC==IC==C: one kernel per channel)
 *     output   out[OC][OH][OW]
 *
 *     OH = (IH-1)*stride_y - 2*pad_top  + dil_y*(KH-1) + opad_y + 1
 *     OW = (IW-1)*stride_x - 2*pad_left + dil_x*(KW-1) + opad_x + 1
 *
 *     out[oc][ph][pw] = sum_{ic,kh,kw : ih,iw integral and in-range} in[ic][ih][iw]*W[ic][oc][kh][kw]
 *
 * Here ph = ih*stride_y - pad_top + kh*dil_y, and pw is analogous. A depthwise transpose
 * sums only ic==oc, so each channel is upsampled by its own kernel. Nearest and bilinear
 * resize use this per-channel form, as a depthwise transpose with a box or triangle kernel.
 *
 * `opad_*` is the ONNX and PyTorch output_padding: an extra trailing-only border that
 * disambiguates the output size when stride>1. It is 0 for none, and must be < stride.
 * `pad_top` must be <= dil_y*(KH-1) and `pad_left` <= dil_x*(KW-1), so that the lowered
 * border pad is non-negative, which is the usual case. A larger pad would crop the output,
 * and `rocket_conv_transpose2d_plan` returns -2 for it.
 *
 * The direct form accepts any `OC`, zero-padded to 16, and any `IC`, zero-padded to 32. A
 * depthwise descriptor must have `OC`==`IC` and C%G==0, with G=32, the depthwise channel
 * group. The lowering materializes the dilated input, so its cost scales with the upsampled
 * size, with a multiply by zero on each inserted gap.
 *
 * Evidence: bit-faithful on hardware to the direct scatter reference
 * (`tests/conv_transpose_rocket.c`).
 */
typedef struct {
    int ic, ih, iw;     /* input  channels / height / width  */
    int oc;             /* output channels                   */
    int kh, kw;         /* kernel height / width             */
    int stride_y, stride_x;
    int pad_top, pad_left;
    int opad_y, opad_x; /* output_padding: extra trailing border, 0 = none, < stride */
    int dil_y, dil_x;   /* kernel dilation rate (1 = none)   */
    int depthwise;      /* 0 = direct (W [IC][OC][KH][KW]), 1 = depthwise (OC==IC, W [C][1][KH][KW]) */
} rocket_conv_transpose2d_desc;

/* Returns the output height and width a descriptor implies. Pure. */
static inline int rocket_conv_transpose2d_oh(const rocket_conv_transpose2d_desc *d)
{ return (d->ih - 1) * d->stride_y - 2 * d->pad_top + d->dil_y * (d->kh - 1) + d->opad_y + 1; }
static inline int rocket_conv_transpose2d_ow(const rocket_conv_transpose2d_desc *d)
{ return (d->iw - 1) * d->stride_x - 2 * d->pad_left + d->dil_x * (d->kw - 1) + d->opad_x + 1; }

/* Checks a descriptor against the supported set: the lowering's feasibility and the forward
 * conv's CBUF fit. Pure, with no hardware. Returns 0 if the descriptor runs, and a negated
 * reason otherwise. */
int rocket_conv_transpose2d_plan(const rocket_conv_transpose2d_desc *d);

/* Returns the path `rocket_conv_transpose2d_fp16` takes for a descriptor. Pure, with no
 * hardware. The return is one of:
 *
 * - `ROCKET_CONV_TRANSPOSE_DECONV` (1): the CNA's hardware deconvolution mode, one task per
 *   output-channel tile over the compact input, with the output extent programmed as the
 *   transposed one
 * - `ROCKET_CONV_TRANSPOSE_LOWERED` (0): the dilated-input lowering described above
 * - A negative value: refused, with the reasons of `rocket_conv_transpose2d_plan`
 *
 * The deconvolution mode is taken on the RK3588 for a direct transposed conv with all of:
 *
 * - Dilation 1
 * - Power-of-two strides: 1, 2, 4 or 8 per axis, with 2, 4 or 8 on at least one axis
 * - `output_padding` below the stride
 * - A `k-1-pad` of at most 11
 * - `IC` a multiple of 32
 * - A compact input that fits one CBUF pass beside a 16-channel weight tile
 *
 * The mode is bit-exact against the scatter reference, and takes 0.35-0.54x the lowering's
 * wall-clock time at decoder shapes [HW sweep, Turing RK1 at ~816 MHz, 2026-09-23].
 * `ROCKET_CONV_TRANSPOSE_HW=0` forces the lowering. */
#define ROCKET_CONV_TRANSPOSE_LOWERED 0
#define ROCKET_CONV_TRANSPOSE_DECONV  1
int rocket_conv_transpose2d_route(const rocket_conv_transpose2d_desc *d);

/* Runs the transposed conv on the NPU. `in`, `W` and `out` are row-major fp16 in the layouts
 * above. The `_ctx` form reuses a resident BO pool. Returns 0 on success and a negative
 * value on error. */
int rocket_conv_transpose2d_fp16(int fd, const rocket_conv_transpose2d_desc *d,
                                 const _Float16 *in, const _Float16 *W, _Float16 *out);
int rocket_conv_transpose2d_fp16_ctx(rocket_conv_ctx *ctx,
                                     const rocket_conv_transpose2d_desc *d,
                                     const _Float16 *in, const _Float16 *W, _Float16 *out);

/* Computes the CPU fp32-accumulate reference, which is also the host fallback at `fd`<0. It
 * uses the direct scatter-add definition, independent of the lowering, so a test against it
 * checks the lowering and not only the forward conv. Pure host, with no hardware. */
void rocket_conv_transpose2d_ref_fp16(const rocket_conv_transpose2d_desc *d,
                                      const _Float16 *in, const _Float16 *W, _Float16 *out);

/* ---- the resident-weight transposed conv (col2im over the resident fp16 matmul) -------
 *
 * Computes the transposed conv with resident weights, for a model that runs the same
 * ConvTranspose repeatedly. It is one GEMM and a host scatter-add. The input, read as IC
 * planes of IH*IW pixels, times the weight as it is stored, [IC][OC*KH*KW], gives every tap
 * of every input pixel:
 *
 *     P[(oc,kh,kw)][ih*IW + iw] = sum_ic in[ic][ih][iw] * W[ic][oc][kh][kw],
 *
 * There are no inserted zeros and no im2col. Each tap's plane then adds into the output at
 * the offset the tap puts it.
 *
 * The GEMM is the resident fp16 matmul (`rocket_matmul.h`'s `rocket_ctx`). Its weight is
 * transposed and packed once into the handle and fanned across the context's worker fds. It
 * splits along M, with each worker holding the whole weight, where the input outweighs the
 * weight, and along N otherwise. `ROCKET_CT_SPLIT=m|n` forces one. The input and P are in
 * channel-planes form, so neither side takes a host transpose.
 *
 *     ctx = rocket_ctx_create(3);
 *     w   = rocket_conv_transpose2d_weights_pack(ctx, &desc, W, bias);   // bias may be NULL
 *     ... per call:
 *         rocket_conv_transpose2d_fp16_prepacked(ctx, w, in, out);
 *     rocket_conv_transpose2d_weights_free(ctx, w);                     // before the ctx
 *     rocket_ctx_free(ctx);
 *
 * Layouts are those of `rocket_conv_transpose2d_fp16`: `in` [IC][IH][IW], `W`
 * [IC][OC][KH][KW] and `out` [OC][OH][OW]. `bias` is [OC], added before the one fp16
 * rounding. The handle copies what it needs at pack time and never reads `W` or `bias`
 * again. A changed weight is a new pack, and nothing is cached by pointer, so a handle
 * cannot serve a stale weight.
 *
 * The handle holds the descriptor, the resident weight and a host buffer of P,
 * OC*KH*KW*IH*IW fp16 values. The resident weight is the context worker count x the
 * weight's bytes under the M split, and 1x under the N split. The handle is tied to the
 * context it was packed on, and is not thread-safe, as the context is not.
 *
 * The scatter-add splits the output channels across the context's worker count of threads,
 * and uses one thread below 2^16 output elements. `ROCKET_CT_SCATTER_THREADS` overrides the
 * thread count.
 *
 * Each P element is an fp32 sum of exact fp16 products, rounded to fp16 once per K tile of
 * the matmul. The matmul has one or two K tiles at IC <= 1024. The scatter-add sums, in
 * fp32, the ceil(KH/sy)*ceil(KW/sx) planes that reach an output pixel and the bias, and
 * rounds once. An output is therefore within a few fp16 roundings of its own magnitude sum
 * (mag) of the exact result. Here mag = sum over taps of |x*w|, plus |bias|.
 *
 * The error bound was measured at pix2pix's eight and SAM's two layers, with real-valued
 * data and a bias. Against an fp64 scatter reference, the worst |err| / mag is 2^-13.9 to
 * 2^-11.1 per layer, under a tested limit of 2^-9.
 *
 * A call takes 0.11-0.58x ONNX Runtime 1.27.0's 2-thread per-node time at pix2pix's and
 * SAM's layers. It takes 0.02-0.29x the time of `rocket_conv_transpose2d_fp16_ctx`, which
 * refuses up2-up5. These times are [HW sweep, mainline RK1 at ~816 MHz, rocket 1.3.0,
 * three-worker context, governor performance, taskset -c 4-7 against ORT on 6-7, 2026-09-27,
 * tests/ct_model_bench.c].
 *
 * `rocket_conv_transpose2d_prepacked_plan()` is pure, and refuses with:
 *
 * - `ROCKET_E_SHAPE` for an invalid descriptor
 * - -3 for an empty output
 * - `ROCKET_E_UNSUPPORTED` for a depthwise transpose, which has no GEMM and which the
 *   one-shot `rocket_conv_transpose2d_fp16` runs, and on any SoC but the RK3588
 * - -4 when IH*IW or OC*KH*KW passes 2^24, or P would pass 64 MiB
 *
 * A pad past the kernel's reach (a cropping transpose) and any output_padding are accepted,
 * because only the scatter-add bounds them. A plan of 0 runs, and what can still fail at run
 * time is an allocation. `_pack` returns NULL on a refused descriptor or a failed
 * allocation. `_prepacked` returns 0, or a negative value with the matmul's or the
 * arguments' reason.
 *
 * Evidence: [HW sweep, RK1 at ~816 MHz, rocket 1.3.0, 2026-09-27,
 * tests/conv_transpose_resident.c]. Integer data whose partials and sums stay below 2048 is
 * bit-exact to `rocket_conv_transpose2d_ref_fp16` at 20 sweep shapes (stride 1-4 per axis,
 * output_padding, dilation 2, 1x3 to 4x4 kernels, IC 3, a cropping pad) and all ten layers.
 * It holds under both worker splits and `ROCKET_KACC=0`.
 */
typedef struct rocket_conv_transpose2d_weights rocket_conv_transpose2d_weights;
struct rocket_ctx;

int rocket_conv_transpose2d_prepacked_plan(const rocket_conv_transpose2d_desc *d);
rocket_conv_transpose2d_weights *
rocket_conv_transpose2d_weights_pack(struct rocket_ctx *ctx, const rocket_conv_transpose2d_desc *d,
                                     const _Float16 *W, const _Float16 *bias);
int rocket_conv_transpose2d_fp16_prepacked(struct rocket_ctx *ctx,
                                           rocket_conv_transpose2d_weights *w,
                                           const _Float16 *in, _Float16 *out);
void rocket_conv_transpose2d_weights_free(struct rocket_ctx *ctx,
                                          rocket_conv_transpose2d_weights *w);

/* ---- the resident-weight forward conv (im2col over the resident fp16 matmul) ----------
 *
 * Computes the forward conv with resident weights, for a model that runs the same
 * convolution repeatedly. It is a host unfold and one GEMM. The input is unfolded into
 * IC*KH*KW planes of the OH*OW output pixels:
 *
 *     A[(ic,ky,kx)][oy*OW + ox] = in[ic][oy*sy + ky*dy - pad_top][ox*sx + kx*dx - pad_left]
 *
 * A is zero outside the input. A times the weight as it is stored, [OC][IC*KH*KW], gives the
 * output as OC planes of OH*OW pixels, which is its own layout.
 *
 * The GEMM is the resident fp16 matmul (`rocket_matmul.h`'s `rocket_ctx`). Its weight is
 * packed once into the handle and fanned across the context's worker fds. It splits along M
 * where the unfolded input outweighs the weight, and along N otherwise.
 * `ROCKET_CONV_SPLIT=m|n` forces one. The planes are interleaved straight into the GEMM's
 * input cube, and its output cube is de-interleaved straight into the output. Neither side
 * takes a host transpose.
 *
 *     ctx = rocket_ctx_create(3);
 *     w   = rocket_conv2d_weights_pack(ctx, &desc, W, bias);   // bias may be NULL
 *     ... per call:
 *         rocket_conv2d_fp16_prepacked(ctx, w, in, out);
 *     rocket_conv2d_weights_free(ctx, w);                     // before the ctx
 *     rocket_ctx_free(ctx);
 *
 * Layouts are those of `rocket_conv2d_fp16`: `in` [IC][IH][IW], `W` [OC][IC][KH][KW] and
 * `out` [OC][OH][OW]. `bias` is [OC]. An explicit `desc->oh` or `desc->ow` is honored, with
 * its trailing pad read as zeros. The handle copies what it needs at pack time and never
 * reads `W` or `bias` again, so a changed weight is a new pack.
 *
 * The unfold runs on the context's worker count of threads, and on one thread below 2^17
 * unfolded elements. `ROCKET_IM2COL_THREADS` overrides the thread count. A 1x1 kernel at
 * stride 1 with no pad reads the input in place and unfolds nothing.
 *
 * The handle holds the descriptor, the resident weight and the unfold buffer (IC*KH*KW x
 * OH*OW fp16). The resident weight is the context worker count x the weight's bytes under
 * the M split, and 1x under the N split. When OC is not a multiple of 16 or OH*OW of 4, the
 * handle also holds a staging buffer for the GEMM's padded output. It is tied to the context
 * it was packed on, and is not thread-safe, as the context is not.
 *
 * Each output is an fp32 sum of exact fp16 products, rounded to fp16 once per K tile of the
 * matmul. A bias is added to that fp16 value in fp32, and the sum rounded once more. An
 * output is therefore within a few fp16 roundings of its own magnitude sum, where mag = sum
 * over taps of |x*w|, plus |bias|.
 *
 * The error bound was measured at pix2pix's eight encoder layers and three Depth Anything v2
 * shapes, with real-valued data and a bias. Against an fp64 reference, the worst |err| / mag
 * is 2^-14.8 to 2^-11.0 per layer, under a tested limit of 2^-9.
 *
 * A call takes 0.21-0.62x ONNX Runtime 1.27.0's 4-thread per-node time at pix2pix's layers
 * c2-c8 and 1.16x at c1, which has three input channels (50 MMAC over 16384 pixels). At
 * Depth Anything v2-small's neck at 518x518 it takes 0.12-1.43x. It is slower at the 19x19
 * and 37x37 3x3 convs, and 8.2x faster at the 3x3 stride-2 one. These times are [HW sweep,
 * mainline RK1 at ~816 MHz, three-worker context, governor ondemand, ORT at allow_spinning 0
 * on the A76s, 2026-10-08, tests/conv_model_bench.c].
 *
 * The unfold is the host's share of the time, and a frontend adds its own conversions.
 * `rocket_conv2d_prepacked_estimate_ms()` below models these times.
 *
 * `rocket_conv2d_prepacked_plan()` is pure, and refuses with:
 *
 * - `ROCKET_E_SHAPE` for an invalid descriptor
 * - -3 for an empty output
 * - `ROCKET_E_UNSUPPORTED` for a depthwise conv, which has no GEMM and which
 *   `rocket_conv2d_fp16` runs, and on any SoC but the RK3588
 * - -4 when OH*OW, IC*KH*KW or OC passes 2^24
 * - -4 when the unfold buffer or the staging buffer would pass 64 MiB
 * - -4 when the matmul's plan refuses the GEMM
 *
 * A plan of 0 runs, and what can still fail at run time is an allocation. `_pack` returns
 * NULL on a refused descriptor or a failed allocation. `_prepacked` returns 0, or a negative
 * value with the matmul's or the arguments' reason.
 *
 * Evidence: [HW sweep, RK1 at ~816 MHz, rocket 1.3.0, 2026-10-08, tests/conv_resident.c].
 * Integer data whose partials and sums stay below 2048 is bit-exact to
 * `rocket_conv2d_ref_fp16` at 18 sweep shapes (stride 1-3 per axis, pads 0-3 per axis,
 * dilation 2, an explicit extent, 1x1 in place and at stride 2, a patch embedding, 1xK and
 * Kx1, IC 3, OC and OH*OW off their alignment), under one and three unfold threads. It is
 * also bit-exact at all eleven layers, under both worker splits and `ROCKET_KACC=0`.
 */
typedef struct rocket_conv2d_weights rocket_conv2d_weights;

int rocket_conv2d_prepacked_plan(const rocket_conv2d_desc *d);
/* Returns an estimate of one `rocket_conv2d_fp16_prepacked` call in ms, or a negative value
 * where the plan refuses. Pure. A frontend uses it to decide whether to claim a node.
 *
 * The estimate is a linear model in the GEMM's aligned `M`, `K` and `N`. Its terms are a
 * fixed term, the unfold's writes, the input, output and weight cubes, and the MACs. It is
 * fitted to the warm median of 22 shapes on a three-worker context. The shapes are pix2pix's
 * eight encoder layers, Depth Anything v2-small's neck at 518x518 (1x1 and 3x3, stride 1 and
 * 2, 19x19 to 296x296), and the DINOv2 and ViT-B/16 patch embeddings. Over those shapes it is
 * within 0.85-1.16x of the measured time [HW sweep, mainline RK1 at 928 MHz (the 900 MHz
 * request), rocket 1.3.0, governor ondemand, 2026-10-27, tests/conv_model_bench.c].
 *
 * Note: it is a model, not a measurement. Another clock, worker count or governor moves
 * every term, and a shape far outside the fit set can fall outside that spread. */
double rocket_conv2d_prepacked_estimate_ms(const rocket_conv2d_desc *d);
rocket_conv2d_weights *rocket_conv2d_weights_pack(struct rocket_ctx *ctx, const rocket_conv2d_desc *d,
                                                  const _Float16 *W, const _Float16 *bias);
int rocket_conv2d_fp16_prepacked(struct rocket_ctx *ctx, rocket_conv2d_weights *w,
                                 const _Float16 *in, _Float16 *out);
void rocket_conv2d_weights_free(struct rocket_ctx *ctx, rocket_conv2d_weights *w);

/* ---- RK3576: the per-chip convolution entries ---------------------------------------
 *
 * Computes int8 direct and depthwise convolutions on the RK3576, with the RK3576's own
 * encoders. The RK3576 NPU is the same IP family as the RK3588 and runs through the same
 * uAPI. Its CNA, CORE and DPU blocks use a different geometry-register encoding at the same
 * block bases. A program built for the RK3588 therefore submits and completes on the RK3576,
 * and writes nothing.
 *
 * `rocket_conv2d_dw_int8()` and `rocket_conv2d_fp16()` have semantics the RK3576 computes.
 * On an RK3576 they dispatch to the two entries below, and a caller writes nothing
 * chip-specific.
 *
 * `rocket_conv2d_int8()` does not dispatch. It writes a raw int32 accumulator and
 * requantizes on the host, while the RK3576's direct int8 datapath requantizes on chip and
 * writes a byte. Emulating one with the other is a different arithmetic, not a wrapper. That
 * entry therefore refuses on the RK3576 and names `rocket_conv2d_int8_rk3576()`, which takes
 * the quant parameters and writes int8. `rocket_matmul_int8()` and
 * `rocket_matmul_int8_rk3576()` follow the same rule for the same reason.
 *
 * Layouts and descriptor semantics are the ones above, all row-major: `in` [IC][IH][IW], `W`
 * [OC][IC][KH][KW] (direct) or [C][KH][KW] (depthwise), and `out` [OC][OH][OW]. Any IC and
 * OC are accepted, because the library pads to the RK3576's own granules. Rows and output
 * channels tile automatically.
 *
 * The quant contract is per-tensor, and the zero points are signed int8 in the model domain.
 * `bias` can be NULL. The input and weight zero points are folded exactly into the
 * coefficient buffer, and the output zero point becomes the DPU's OUT_CVT offset.
 *
 * A depthwise weight zero point is programmed in the weight cube, not in the coefficient
 * group. That group has no B field, and a depthwise zero point's correction is not a
 * per-channel constant, so the bias cannot hold it either. The cube gives every (channel,
 * tap) two live bytes and the datapath adds them, so `w - w_zp` is expressed exactly across
 * the pair. Every per-tensor TFLite depthwise filter has this form. The bound is the pair's
 * own range, [-256, 254].
 *
 * On the direct entry, IC <= 4 takes the CNA's packed-image first-conv sub-encoding. The
 * depthwise entry refuses that count, because there is no depthwise form to send it to: the
 * channel fold leaves nothing to be depthwise over. `rocket_conv2d_desc.direct_datapath`
 * runs a narrow count on the ordinary datapath instead, and documents the trade.
 *
 * The entries refuse, rather than approximate, each of these:
 *
 * - Dilation, because no RK3576 shape has been run through the rate fields
 * - A shape whose resident weight slice does not fit even one output-channel group, which
 *   needs an input-channel split that the on-chip requant forecloses
 * - An input or output width past 8192
 * - A depthwise channel count past 8192
 * - A kernel past 32 on either axis
 * - A stride past 7
 *
 * The last four are refused at claim time as well. They are the widths of the fields the
 * RK3576 computes with. Past each one, a task returns a complete output of the right size
 * with wrong values [HW sweep, H96, tests/rk3576_conv_width_probe]. Nothing here tiles the
 * width.
 *
 * Evidence: bit-exact against a CPU model over `tests/rk3576_conv_lib_gate.c` [HW sweep, H96
 * MAX M9].
 */
int rocket_conv2d_int8_rk3576(int fd, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *W, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out);
int rocket_conv2d_dw_int8_rk3576(int fd, const rocket_conv2d_desc *d,
                                 const int8_t *in, const int8_t *w, const int32_t *bias,
                                 float in_scale, float w_scale, float out_scale,
                                 int in_zp, int w_zp, int out_zp, int8_t *out);

/* Computes the convolution of `rocket_conv2d_int8_rk3576()` with per-output-channel weight
 * scales, `w_scale[oc]`, the form every TFLite int8 and ONNX QDQ model has. It is otherwise
 * identical: the same layouts, the same tiling and the same row window, int8 in and int8
 * out, with the requant on chip.
 *
 * The per-channel gain is programmed in the coefficient group's int16 `C`, and the OUT_CVT
 * applies a single base gain per task. The result reproduces a per-axis model closely, and
 * is not bit-exact. TFLite's per-axis requant has a per-channel shift as well as a
 * per-channel multiplier, and this datapath has only the multiplier. `C` is an integer, so
 * channel oc resolves its gain to 0.5/C[oc]. `rocket_conv2d_int8_perchannel_plan_rk3576()`
 * below describes the ramp of `C` values and its bounds.
 *
 * Under `ROCKET_RK3576_PC_SHIFT=0`, the DPU's epilogue is `(acc + A[oc]) * C[oc]` in
 * saturating int32, followed by one `(v*MUL)>>SHIFT` for the whole task. The int32 product
 * then saturates, which caps C[oc] at `INT32_MAX / max|acc + A|`. The planner takes the
 * accumulator bound from the actual weights (128*sum|w[oc]|) rather than from the int8
 * range. The weight bound is one to two orders of magnitude tighter, and the difference is
 * most of the available precision.
 *
 * The planner logs the worst-case relative gain error it chose. On the shift-0 ramp, that
 * error is largest for a layer with a wide scale spread and a large fan-in. The chip
 * arithmetic is tested bit-exactly against a CPU model of it. Its distance from an exact
 * per-axis reference is reported per shape rather than asserted.
 *
 * The entry runs depthwise too, which is the shape a real per-axis model is mostly made of.
 * The 48-byte coefficient group holds the same per-channel `C`. A depthwise task's
 * accumulator is bounded by kh*kw taps rather than ic*kh*kw, one to two orders of magnitude
 * smaller. On the shift-0 ramp, its `C` ramp therefore reaches the int16 field's own 32767
 * maximum. The gain resolution a single OUT_CVT shift leaves is then far finer than the
 * direct path's at the same channel count. A depthwise layer is one task whatever the
 * channel count, because output channel c is bound to input channel c.
 *
 * A depthwise layer has no output-channel window to program and no scale sort to make. The
 * tile query below returns `oc` for a depthwise descriptor.
 *
 * The entry refuses two inputs rather than approximating them. One is four or fewer input
 * channels on the direct path unless `direct_datapath` is set, because the packed-image
 * first conv is a different program. The other is a weight zero point, because a per-axis
 * quantization is symmetric by construction.
 *
 * Evidence: [HW sweep, H96 MAX M9, tests/rk3576_coeff_c.c + tests/rk3576_perchannel_gate.c].
 */
int rocket_conv2d_int8_perchannel_rk3576(int fd, const rocket_conv2d_desc *d,
                                         const int8_t *in, const int8_t *W,
                                         const int32_t *bias, float in_scale,
                                         const float *w_scale, float out_scale,
                                         int in_zp, int out_zp, int8_t *out);

/*
 * Computes the convolution of `rocket_conv2d_int8_rk3576()` with a nonlinear activation
 * fused into its epilogue. It runs as one job, and the activation costs only the table-load
 * task. `kind` is a `ROCKET_RK3576_ACT_*` from `npu_regcmd_rk3576.h`. The quantization is
 * per-tensor, the conv direct or depthwise, and the entry is RK3576 only.
 *
 * The arithmetic differs from the conv followed by `rocket_act_int8_rk3576()`, and is the
 * more accurate of the two. The standalone op reads the convolution's requantized int8
 * output, so it sees 256 possible values. This entry reads the accumulator before any
 * rounding.
 *
 * The cost is exactness. The standalone form puts each of its 256 inputs on its own table
 * entry. A real accumulator falls between entries, and the hardware's linear interpolation
 * runs. The distance to an exact f() is reported here, not asserted.
 *
 * The table's units and step follow from the three scales the caller already passes, so
 * there is no extra contract. With a per-tensor requant the BS multiplier is unity, so one
 * accumulator count means `in_scale*w_scale`. An entry is read in units of `out_scale` times
 * the gain OUT_CVT programs. The window step is the smallest that covers the domain over
 * which the activation still varies.
 *
 * The entry refuses two bounds rather than computing them wrong. One is a value unit too
 * fine for the table's peak to fit an int16, and the other a domain no step reaches. Both
 * refusals name `in_scale*w_scale` as the quantity to change.
 *
 * The entry takes per-tensor scales only. Per-output-channel weight scales and the LUT need
 * the same register, the coefficient group's `C`, which divides the value unit per channel.
 * A per-axis requant uses it for gain resolution, and the LUT uses it to place a channel's
 * range inside the table's window. No measured model needs both. The per-tensor fused form
 * is within one count of an exact f(), and a per-axis graph is within one count per layer
 * with no fusion.
 *
 * Returns 0 or a negative rocket error.
 *
 * Evidence: [HW sweep, H96 MAX M9, tests/rk3576_act_gate.c].
 */
int rocket_conv2d_int8_act_rk3576(int fd, const rocket_conv2d_desc *d,
                                  const int8_t *in, const int8_t *W, const int32_t *bias,
                                  float in_scale, float w_scale, float out_scale,
                                  int in_zp, int w_zp, int out_zp,
                                  int kind, int8_t *out);

/* Returns whether the int8 convolution path takes a geometry at all, from the descriptor
 * alone, with no weights, no fd and no submit. Pure, so a frontend can ask it while deciding
 * what to claim. A refusal here costs one node the framework runs itself, while the same
 * refusal at pack or run time fails the caller's whole model.
 *
 * It answers the bounds that are a function of shape rather than of values, which no
 * per-node inspection can predict:
 *
 * - The descriptor checks: dilation, a depthwise oc != ic, a trailing pad of a whole kernel,
 *   and the 8-bit pad fields
 * - The resident weight slice, refused past `ic*kh*kw` 4608, where the RK3576 drops trailing
 *   output-channel groups while still writing a complete output
 * - The row window, which bounds a large plane at a deep channel count
 *
 * A plane past the CBUF allowance is cut into row tasks. A shape for which no window fits at
 * all is refused rather than run wrong. The recourse for both the weight slice and the row
 * window is an input-channel split, which is the caller's to choose.
 *
 * Four or fewer input channels are planned on the direct datapath, whatever the descriptor
 * says. A `ROCKET_OK` for such a descriptor therefore answers for a pack with
 * `direct_datapath` set. With the flag clear, the pack takes the packed-image first conv
 * instead, and returns NULL when that encoding's geometry bounds refuse the layer. The
 * library does not fall back to the direct encoding. The caller must then pack again with
 * `direct_datapath` set.
 *
 * The plan does not ask the questions about values, which have their own entries: the
 * per-axis gain ramp (`rocket_conv2d_int8_perchannel_plan_rk3576`, below) and the quant
 * scales. Returns `ROCKET_OK`, `ROCKET_E_SHAPE` or `ROCKET_E_UNSUPPORTED`.
 *
 * Evidence: [HW sweep, H96 MAX M9, tests/rk3576_conv_lib_gate.c claimplan].
 */
int rocket_conv2d_int8_plan_rk3576(const rocket_conv2d_desc *d);

/* Returns whether every channel's gain is a value the `C` ramp holds, and the output-channel
 * tile this path would use. Both come from the weights and the quant contract alone, with no
 * device and no submit. Pure, so a frontend can ask it while deciding what to claim. A
 * refusal here costs one node the framework runs itself rather than the whole model.
 *
 * A tile is one task, and a task has one OUT_CVT gain and one DPU shift word. A channel
 * therefore reaches its own scale through the int16 `C` in its coefficient group. The BS
 * stage computes sat32(rne(((acc + A)*C) >> s)) with the product held wide [HW sweep, H96
 * MAX M9]. The planner puts the task's largest live scale at `C` = 32767 and every other
 * live channel in proportion. It takes the smallest `s` at which every live product fits
 * int32 at its accumulator bound. A live channel resolves its gain to 0.5/C relative, and
 * the ramp spans 32767:1 in every task whatever the fan-in.
 *
 * A channel whose output cannot vary is programmed as its byte. For any int8 input, its
 * accumulator stays in [A - 128*pos - 127*neg, A + 127*pos + 128*neg] over its positive and
 * negative weight sums. The exact per-axis output can be one byte over that whole range,
 * 0.01 of a count clear of a rounding boundary or past the rail. The planner then chooses
 * the channel's A and C against the RK3576's own arithmetic so that it writes that byte. The
 * channel takes no part in the ramp. An all-zero filter is one such channel, and so is a
 * dead channel whose bias drives its whole range past the rail.
 *
 * The plan refuses a live channel below 1/65534 of its task's largest live scale. Its `C`
 * would round to zero, and the field's minimum is one. `ROCKET_RK3576_PC_ALLOW_CLAMP=1`
 * computes anyway. The pack and run entries refuse on the same prediction, so a claim made
 * here cannot fail there.
 *
 * Caution: with `ROCKET_RK3576_PC_ALLOW_CLAMP=1`, such a channel computes at the wrong gain,
 * on a complete output of the right size with wrong values.
 *
 * Through tflite-rocket on EfficientDet-Lite0's 182 per-axis layers, the plan refuses none
 * and keeps every direct layer in one output-channel tile. The worst predicted live gain
 * error is 0.61% (median 0.008%), and 345 channels over 28 layers are programmed as their
 * byte, with BS shifts 0-7. The model's warm wall-clock time is 0.661x the shift-0 ramp's.
 * Its mAP@[.5:.95] over 500 val2017 images is 0.2818, against the CPU's 0.2823 [HW sweep,
 * H96 MAX M9, walls with both clusters on `performance`, 2026-09-27].
 *
 * `ROCKET_RK3576_PC_SHIFT=0` restores the shift-0 ramp exactly. The shift word is zero, `C`
 * is capped at INT32_MAX / max|acc + A| where the product saturates, and there are no
 * constant channels. A clamped channel is refused unless `ROCKET_RK3576_PC_INERT` finds it
 * an all-zero filter whose fold sits past the same rail under both gains.
 * `ROCKET_RK3576_PC_INERT` is on by default, `0` refuses every clamp, and `dw` accepts only
 * on the depthwise path. It is read in the shift-0 mode only.
 *
 * Returns `ROCKET_OK`, `ROCKET_E_UNSUPPORTED` where a live channel's gain is off the ramp,
 * or `ROCKET_E_SHAPE` for a descriptor this path does not take. `oc_tile` and `max_rel_err`
 * can be NULL, and both are filled even when the answer is a refusal. The error is the worst
 * live channel's relative gain error. On the shift-0 ramp it also counts clamped all-zero
 * filters, so an accepted layer can report one above 100%. It is a reported quantity at
 * every accepted shape, not an asserted one.
 *
 * Evidence: [HW sweep, H96 MAX M9, tests/rk3576_perchannel_gate.c].
 */
int rocket_conv2d_int8_perchannel_plan_rk3576(const rocket_conv2d_desc *d,
                                              const int8_t *W, const int32_t *bias,
                                              float in_scale, const float *w_scale,
                                              float out_scale, int in_zp, int out_zp,
                                              unsigned *oc_tile, double *max_rel_err);

/* Returns the output-channel tile `rocket_conv2d_int8_perchannel_rk3576()` will use, without
 * running it. The tile also sets the entry's submit count: `ceil(oc/tile)` row-task sets,
 * each at the RK3576's fixed per-submit cost.
 *
 * On this path the tile is an accuracy parameter as well as a CBUF one. A tile is one task,
 * and a task has one OUT_CVT shift. The `C` ramp inside a tile therefore spans only that
 * tile's range of scales. The channels are sorted by scale first, so halving the tile
 * roughly halves the spread the ramp must cover. Under the BS shift, a shape at ic=128
 * oc=128 with a 100x spread is 0.6 counts from an exact per-axis requant in one tile.
 *
 * The planner takes the largest tile whose predicted worst-case gain error meets
 * `ROCKET_RK3576_PC_MAX_ERR` (default 1%). A layer that does not need the split therefore
 * does not take it. `ROCKET_RK3576_PC_OC_TILE` forces a tile.
 *
 * Returns 0 for a descriptor this path refuses. [HW sweep, H96 MAX M9] */
unsigned rocket_conv2d_int8_perchannel_oc_tile_rk3576(const rocket_conv2d_desc *d,
                                                      const int8_t *W,
                                                      const int32_t *bias,
                                                      float in_scale,
                                                      const float *w_scale,
                                                      float out_scale, int in_zp,
                                                      int out_zp);

/* ---- the residual add, as a convolution's weights ---------------------------------
 *
 * Fills the weights, bias and weight scale that express a residual add as a convolution on
 * the RK3576. The DPU's elementwise stage there takes exactly one operand, so a skip
 * connection cannot be an elementwise op. It is a convolution instead: concatenate the two
 * operands along the channel axis and convolve with a 1x1 kernel of two diagonal blocks.
 *
 *     out[o] = requant( w1*(a[o] - a_zp) + w2*(b[o] - b_zp) )
 *
 * Everything in that lowering except the weights is an ordinary convolution the entries
 * above already run. This entry is pure, with no hardware and no allocation, and the caller
 * then runs:
 *
 *     d.ic = 2*c;  d.oc = c;  d.kh = d.kw = 1;  d.direct_datapath = 1;
 *     rocket_conv2d_int8_rk3576(fd, &d, concat, W, bias, a_scale, w_scale, out_scale,
 *                               a_zp, 0, out_zp, out);
 *
 * Here `concat` is the two operands' channels one after the other. In cube layout that
 * concatenation is free. A feature cube's channel-group stride is the plane, and a direct
 * conv's output stride is `ow*oh`. Two producers that write adjacent halves of one buffer
 * therefore already form the concatenated operand.
 *
 * The lowering has two properties the vendor's own elementwise program lacks. The operands
 * can have different scales. The ratio is programmed in the weights as `w2/w1`. The entry
 * resolves it to about one part in 127^2 by searching every denominator rather than fixing
 * one term at 127. A single operand converter instead forces both operands onto a common
 * scale. Both zero points are also exact, because `w2*(a_zp - b_zp)` is a per-output-channel
 * constant, which is what the bias is.
 *
 * The datapath can also compute the add fused into the producing convolution. On a real
 * quantization, however, the weight the fusion needs does not fit the field.
 * `rocket_residual_fuse_weight_rk3576()` below states that bound.
 *
 * `W` is `c * 2c` int8 in the [oc][ic] order the direct entries take, `bias` is `c` int32,
 * and `w_pair`, which is optional, receives the chosen `{w1, w2}`. The entry refuses, rather
 * than approximates, `2c > 4608` and operand scales more than 127x apart.
 *
 * Evidence: [HW sweep, H96 MAX M9, tests/rk3576_residual_add.c]. Every register in the
 * elementwise stage's interface was swept, and none takes a second operand.
 */
int rocket_residual_add_weights_rk3576(unsigned c, float a_scale, float b_scale,
                                       int a_zp, int b_zp, int8_t *W, int32_t *bias,
                                       float *w_scale, int *w_pair);

/* ---- the fused residual add, and the quantization that refuses it -------------------
 *
 * Computes the identity weight that fuses a residual add into a block's last convolution, or
 * refuses it. The convolution takes `c` more input channels with an identity block at the
 * center tap of the kernel. The skip is then part of a convolution the network already runs,
 * and needs no program of its own. The fused form is bit-exact at MobileNetV2's project
 * convolution for ic 168-1120. It is also bit-exact at ResNet-18's second convolution for ic
 * 128-512 [HW sweep, H96 MAX M9, tests/rk3576_residual_add.c `fused`]. It is the hardware's
 * own idiom, since the vendor compiler folds `Add(Conv(x), x)` into exactly this shape.
 *
 * The identity weight refuses the fusion, not the slice rule. A standalone add chooses its
 * weight scale, because its weights are not a quantized tensor. Choosing the scale is how
 * the operand ratio fits in a pair of int8s. A fused add inherits the convolution's weight
 * scale, which the convolution's own filter fixes. The skip's contribution must therefore
 * reach the output through the accumulator unit the main term already uses:
 *
 *     w_skip = skip_scale / (in_scale * w_scale)          and it must be an int8
 *
 * The skip and the sum have comparable ranges, so `w_skip` is about `1/gain`, where `gain`
 * is the convolution's own requant gain. The fusion therefore needs a gain above about
 * 1/127, which is a shallow contraction. A project convolution over 576-960 channels is far
 * from it: MobileNetV2's ten residual blocks need 173 to 865, and none of the ten fits.
 *
 * A per-axis weight quantization makes it strictly worse. The per-tensor `w_scale` is the
 * largest of the per-channel ones, so every channel's weight grows.
 *
 * Pure, with no hardware and no allocation. `w_skip` receives the identity weight, and
 * `bias_delta` the per-output-channel constant to add to every bias. The datapath subtracts
 * the convolution's own input zero point from the skip channels, and
 * `w_skip*(in_zp - skip_zp)` corrects that exactly. `rel_err`, which is optional, receives
 * the identity weight's rounding error relative to the exact ratio. Returns
 * `ROCKET_E_UNSUPPORTED` when the weight does not fit int8, which is the answer for every
 * real residual block measured.
 */
int rocket_residual_fuse_weight_rk3576(float in_scale, float w_scale, float skip_scale,
                                       int in_zp, int skip_zp,
                                       int *w_skip, int32_t *bias_delta, float *rel_err);

/* ---- the pack-weights-once path ---------------------------------------------------
 *
 * Packs once what the per-call RK3576 int8 entries rebuild from the weights alone. It is for
 * a layer run repeatedly with the same weights, such as a graph's forward pass or a video
 * stream. The pack covers the per-output-channel filter sums, the coefficient group and the
 * weight cube. On the per-axis path it also covers the scale sort and the output-channel
 * tile. None of it depends on the input. On MobileNetV1-224 that work is roughly a quarter
 * of the wall-clock time, where the submits are 1-16% of a layer.
 *
 *     h = rocket_conv2d_int8_pack_rk3576(fd, &d, W, bias, in_scale, w_scale, NULL,
 *                                       out_scale, in_zp, w_zp, out_zp);
 *     ... per inference:
 *         rocket_conv2d_int8_prepacked_rk3576(fd, h, in, out);
 *     rocket_conv2d_int8_weights_free_rk3576(fd, h);
 *
 * `w_scale_oc` selects which requant the handle applies. NULL selects the per-tensor
 * contract of `rocket_conv2d_int8_rk3576()` and `rocket_conv2d_dw_int8_rk3576()`, with
 * `w_scale` the per-tensor scale. Non-NULL selects the per-axis contract of
 * `rocket_conv2d_int8_perchannel_rk3576()`, and `w_scale` is ignored. `desc.depthwise`
 * selects direct or depthwise. The arithmetic is the same code as the per-call entries, so a
 * prepacked call is bit-identical to the transient one by construction.
 *
 * A handle freezes three things:
 *
 * - The fd. Its BOs belong to the file that created them, and an IOVA is per fd. On another
 *   fd the NPU would get addresses that mean nothing there and the submit would still
 *   succeed, so a foreign fd is refused.
 * - The tile plan. The weight cube's group count follows the output-channel tile rather than
 *   the layer. A cube packed for one tiling is not a cube for another. On the per-axis path
 *   that tile is also an accuracy choice made from the weights and the scales, and the pack
 *   makes it once.
 * - The quant contract. Both zero points and all three scales are folded into the
 *   coefficient group at pack time. A layer whose quantization changes needs a new handle.
 *
 * The handle also holds the scratch whose size follows the frozen geometry: the feature
 * cube, the regcmd buffer and each tile's output. An inference therefore allocates no BOs at
 * all. The cost is memory. Every tile's cube is resident at once, about the size of the
 * caller's own `W`, where a transient call holds one tile's.
 *
 * The handle packs the packed-image first conv's cube as well as the direct one. Four or
 * fewer input channels take that encoding unless the descriptor sets `direct_datapath`. The
 * pack returns NULL for anything the per-call entries would refuse.
 *
 * A packed-image handle cannot take a cube as input, because its feature buffer is an
 * interleaved image, so it can only begin a run. Its output side is an ordinary cube and
 * joins normally. The exception is a handle that materializes its pad columns for a non-zero
 * input zero point. Its output is wider than the caller's plane, and every cube entry
 * refuses it.
 *
 * Caution: a handle is not thread-safe. It holds the scratch a call writes through, so
 * concurrent prepacked calls on one handle race. Pack one handle per thread, or serialize
 * the calls.
 */
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

/* ---- cube I/O between consecutive layers -----------------------------------------
 *
 * The entries above take and return row-major tensors, so a graph needs a host transpose at
 * both ends of every layer. The cube interleaves sixteen channels into every sixteen-byte
 * atom. A row-major tensor and a cube are therefore a transpose of each other rather than a
 * copy. On MobileNetV1-224 those two transposes are the largest host cost after the weight
 * pack.
 *
 * Between layers the transposes are unnecessary, because a direct convolution's output
 * stride is exactly `ow*oh`. Layer n's output is layer n+1's input cube, byte for byte,
 * whenever the channel counts round the same way. A handle can read its input from a
 * producer's output and leave its own output in place:
 *
 *     struct rocket_rk3576_cube c;
 *     rocket_conv2d_int8_cube_out_rk3576(prod, 1);          // no de-scatter
 *     rocket_conv2d_int8_cube_of_rk3576(prod, &c);
 *     rocket_conv2d_int8_cube_in_rk3576(cons, &c);          // no feature scatter
 *     ...
 *     rocket_conv2d_int8_prepacked_rk3576(fd, prod, in, NULL);   // out may be NULL
 *     rocket_conv2d_int8_prepacked_rk3576(fd, cons, NULL, out);  // in  may be NULL
 *
 * A cube is a borrowed view of the producer handle's output and is valid for that handle's
 * lifetime. Freeing the producer invalidates it. Nothing here changes what is computed: the
 * outputs are the same bytes the row-major entries transpose.
 */
typedef struct rocket_rk3576_cube {
    int       fd;          /* the file the buffer belongs to                        */
    unsigned  c;           /* live channels                                          */
    unsigned  h, w;        /* the plane                                              */
    unsigned  groups;      /* channel groups of 16 in this cube                      */
    size_t    surf_elems;  /* elements per channel group, == h*w for a direct conv   */
    rocket_bo bo;          /* borrowed, allocated by the producer handle or the caller */
    size_t    off;         /* byte offset of this cube's first channel group in `bo` */
    /* The constant tail. A consumer whose channel count is not a multiple of 32 reads groups
     * past the live ones. At a non-zero weight zero point their content matters, because the
     * coefficient group's B term sums every channel of the programmed group. A non-zero
     * `pad_from` states that channels [pad_from, groups*16) all hold `pad_value`, which lets
     * such a join be accepted instead of refused. Zero means the tail holds whatever was
     * written there, and the join is refused at a non-zero weight zero point. */
    unsigned  pad_from;    /* first channel of the constant tail, 0 = not declared   */
    int       pad_value;   /* what channels [pad_from, groups*16) hold                */
    /* The row pitch, when the buffer that holds this plane is wider than the plane. The
     * packed-image first conv materializes its pad columns, and so programs a wider output
     * extent than the caller asked for. At ResNet-18's stem that is a 128-wide output for a
     * 112-wide tensor. The surplus columns sit between the rows the consumer wants. A
     * `pitch_w` of 0 means the rows are contiguous, `pitch_w == w`, which is true of every
     * other producer on the RK3576.
     *
     * Only a pool can read a cube with a row pitch. The PPU has the consumed extent
     * (`0x600C`) and the DDR line stride (`0x7024`) in different registers, and honors a
     * pitch above the extent. The CNA's feature DMA has the width and the line stride as one
     * quantity in two units. A convolution consumer therefore has no such register, and
     * refuses this cube.
     *
     * Evidence: [HW sweep, tests/rk3576_row_pitch.c], 16 test points over four geometries at
     * gaps of 1/4/16/64, each against a control that leaves the pitch derived and differs.
     */
    unsigned  pitch_w;     /* elements per row in `bo`, 0 = w (rows contiguous)      */
    /* The first column, when the plane does not start at the left edge of those rows. A stem
     * that materializes its pad columns discards its leading output column, the one the
     * hardware still synthesizes. The caller's tensor is therefore a sub-rectangle of the
     * output and not a prefix of it. The PPU's source base is a plain address at atom
     * granularity [HW sweep, tests/rk3576_row_pitch.c: offsets 1/3/8 over four geometries].
     */
    unsigned  col_off;     /* elements into each row where the plane starts, 0 = 0   */
} rocket_rk3576_cube;

/* ---- a cube that is a slice of a larger buffer -----------------------------------
 *
 * Allocates a cube buffer that several producers can write into. A cube's base is a plain
 * address on both sides of a convolution. A program whose feature or output base is moved by
 * G channel groups inside a bigger buffer computes the same result as at offset 0. It writes
 * only its own slice, and composes with the row-task offset [HW sweep, H96 MAX M9,
 * tests/rk3576_offset_cube.c, 34 placements over three geometries].
 *
 * Several producers can therefore write into one allocation, which is what a residual
 * topology needs. A chain of layers breaks wherever the host has to touch a tensor between
 * two programs. On MobileNetV2, two thirds of those places are a channel concatenation the
 * host builds. The concatenation is an add's two operands, or a skip a later layer reads
 * beside a convolution's own output. Written into slices of one buffer, they need no host
 * copy, because the two outputs already form the concatenated operand.
 *
 *     rocket_rk3576_cube buf, lo, hi;
 *     rocket_rk3576_cube_alloc(fd, 2*C, H, W, &buf);        // the caller owns this one
 *     rocket_rk3576_cube_slice(&buf, 0, C, &lo);
 *     rocket_rk3576_cube_slice(&buf, C, C, &hi);
 *     rocket_conv2d_int8_cube_out_at_rk3576(a, &lo);        // two producers, one buffer
 *     rocket_conv2d_int8_cube_out_at_rk3576(b, &hi);
 *     rocket_conv2d_int8_cube_in_rk3576(consumer, &buf);    // reads both halves
 *
 * The channel offset must be a multiple of 16, because a group is the unit an address can
 * name.
 */
int rocket_rk3576_cube_alloc(int fd, unsigned c, unsigned h, unsigned w,
                             rocket_rk3576_cube *out);

void rocket_rk3576_cube_free(int fd, rocket_rk3576_cube *buf);

/* Fills `out` with a view of `c` channels starting at channel `c0` of `buf`. The view is
 * borrowed, and is valid while `buf` is. Refuses a `c0` that is not a multiple of 16, and a
 * range past the end. */
int rocket_rk3576_cube_slice(const rocket_rk3576_cube *buf, unsigned c0, unsigned c,
                             rocket_rk3576_cube *out);

/* Narrows a cube to `live` channels, and states what the groups past them hold.
 *
 * A slice is sized from its live channels, which is what a producer needs, since it writes
 * exactly its own groups. A consumer whose input count is not a multiple of 32 needs one
 * group more. Its feature DMA reads round32(ic) channels, and at a non-zero weight zero
 * point the coefficient group's B term sums those channels. A concatenation buffer allocated
 * to the round-32 count has them, and this call tells the reader that it does, and with what
 * constant. The caller asserts a fact about the bytes: nothing writes anything else there. A
 * producer handle asserts the same when it declares its own partial output group.
 *
 * Refuses a `live` count past the channels the cube already has. */
int rocket_rk3576_cube_declare_tail(rocket_rk3576_cube *c, unsigned live, int value);

/* Describes this handle's output as a cube. Refuses a handle whose output channels are split
 * across more than one tile. Each tile has its own buffer, and several buffers are not one
 * cube. The description is valid as soon as the handle is packed.
 *
 * The cube has the handle's own channel-group stride, which is `ow*oh` on the direct path
 * and `round4(ow*oh)` on the depthwise one. A consumer is told the stride rather than
 * deriving it. A depthwise producer at a plane whose element count is not a multiple of four
 * therefore joins like any other.
 *
 * A direct handle declares its constant tail. It is programmed with the round-32 output
 * channel count, and the DPU writes every group it is told to. A partial group's channels
 * are packed with zero weights and a zero A term, so they are written as the output zero
 * point. At a graph join that is the consumer's input zero point, the border constant its
 * own cube would have used. A depthwise handle is programmed with the raw count and leaves
 * everything past it untouched. It declares nothing unless its output channels fill the last
 * group exactly.
 *
 * Evidence: [HW sweep, H96 MAX M9, tests/rk3576_pad_channels.c] for the constant tail. */
int rocket_conv2d_int8_cube_of_rk3576(const rocket_conv2d_int8_weights_rk3576 *h,
                                      rocket_rk3576_cube *out);

/* Reads this handle's feature input from `src` instead of a row-major tensor, and skips the
 * scatter. A NULL `src` restores the row-major input. The cube is borrowed, so it must
 * outlive the calls that read it.
 *
 * Refuses a cube whose plane, channel count or fd does not match what this handle would have
 * packed for itself. It also refuses a channel-group stride shorter than the plane. A longer
 * stride is programmed into the CNA's DDR group jump, which the RK3576 honors verbatim [HW
 * sweep, H96 MAX M9, tests/rk3576_surf_stride.c].
 *
 * An input channel count that is not a multiple of 32 is refused only at a non-zero weight
 * zero point. Even then it is refused only when the cube does not declare a constant tail
 * that matches this handle's input zero point. The padding channels' weights are zero, so
 * they never reach the MAC, and the one thing that reads them is the coefficient group's B
 * term. */
int rocket_conv2d_int8_cube_in_rk3576(rocket_conv2d_int8_weights_rk3576 *h,
                                      const rocket_rk3576_cube *src);

/* Leaves the output in the cube instead of de-tiling it into the caller's tensor. The handle
 * must have the single-tile shape `rocket_conv2d_int8_cube_of_rk3576` requires. */
int rocket_conv2d_int8_cube_out_rk3576(rocket_conv2d_int8_weights_rk3576 *h, int on);

/* Declares that this handle's output is read by a consumer outside any chain the handle is
 * part of, later in the same inference. An example is a producer that two layers read, only
 * one of which is the layer immediately after it.
 *
 * A chain re-stamps every interior layer's output in its verify bracket. The stamp is sound
 * only because the next kick rewrites that output before anything reads it. When the output
 * is declared shared, the stamp goes on at the start of this handle's next call instead.
 * Cube-out already puts it there. The declaration costs one PREP/FINI pair per call and
 * nothing else. A handle that writes into a caller's buffer is already this case and needs
 * no declaration.
 *
 * Caution: without the declaration, the chain stamps a shared output before the outside
 * consumer reads it. The stamp replaces the layer's output with the sentinel, and the
 * outside consumer computes a complete, plausible output from 0xA5. Nothing shows the error
 * until the next materialized layer. */
int rocket_conv2d_int8_cube_shared_rk3576(rocket_conv2d_int8_weights_rk3576 *h, int on);

/* Writes this handle's output into `dst`, a caller's buffer, at `dst`'s channel-group
 * offset, instead of into the handle's own. This implies cube-out: the output stays where
 * the DPU wrote it, and `rocket_conv2d_int8_cube_of_rk3576()` then describes the slice.
 *
 * The handle's own output BO is released, so a graph allocates the shared buffer instead of
 * both. A NULL `dst` restores it and clears cube-out, and the next call allocates again. The
 * buffer is borrowed, and must outlive the calls that write it.
 *
 * Refuses each case where the output and the slice are not the same object:
 *
 * - A foreign fd
 * - A handle with more than one output-channel tile
 * - A plane or a channel-group stride that is not this handle's
 * - A slice too short for its output channels
 */
int rocket_conv2d_int8_cube_out_at_rk3576(rocket_conv2d_int8_weights_rk3576 *h,
                                          const rocket_rk3576_cube *dst);

/* ---- a run of cube-joined layers as one hardware kick --------------------------------
 *
 * Submits a run of resident handles as one batched job and one hardware kick. The fixed
 * per-submit cost on the RK3576 is ~439 us, and a resident MobileNetV1-224 layer costs
 * 130-250 us of compute. Once the host transposes are gone, the submits are the largest
 * single cost: 29 of them for 29 layers.
 *
 * A chained stream honors read-after-write between its programs. A run of handles can
 * therefore be laid out as one contiguous regcmd stream and submitted as one batched job. A
 * run is eligible when no host work is left between its programs. That takes two conditions,
 * and neither is pairwise:
 *
 * - Every handle but the last leaves a cube.
 * - Every handle but the first reads a cube that some earlier handle of the run wrote, not
 *   necessarily the one immediately before it.
 *
 * Read-after-write across the stream is the RK3576's own guarantee, so a shortcut
 * convolution that takes its input several layers back stays in the stream.
 *
 *     rocket_conv2d_int8_chain_rk3576 *c = rocket_conv2d_int8_chain_new_rk3576(fd, h, n);
 *     for (...) rocket_conv2d_int8_chain_run_rk3576(fd, c, in, out);
 *     rocket_conv2d_int8_chain_free_rk3576(fd, c);
 *
 * `in` is the first handle's row-major input, or NULL if it reads a cube from outside the
 * chain. `out` is the last handle's row-major output, or NULL if it leaves a cube. The
 * constructor checks the two run conditions above rather than assuming them.
 *
 * The handles are borrowed and must outlive the chain. They stay usable one at a time
 * through `rocket_conv2d_int8_prepacked_rk3576()`, which is what the chain falls back to if
 * a program has to be redone. Nothing here changes what is computed.
 *
 * The constructor refuses each case where the chain would otherwise be silently wrong or
 * impossible to build:
 *
 * - Fewer than two handles
 * - A handle with more than one output-channel tile
 * - A handle with a DPU LUT, whose table load is a task in a BO of its own, which this
 *   layout does not hold
 * - A non-last handle that does not leave a cube
 * - A non-first handle whose input no earlier handle of the run writes
 * - A foreign fd
 * - A kernel that does not honor DRM_ROCKET_JOB_BATCHED
 *
 * Evidence: read-after-write between chained programs is measured over five planes from 2 KB
 * to 387 KB, with the second program's feature cube the first's output
 * (`tests/rk3576_chain_raw.c`).
 */
typedef struct rocket_conv2d_int8_chain_rk3576 rocket_conv2d_int8_chain_rk3576;

rocket_conv2d_int8_chain_rk3576 *
rocket_conv2d_int8_chain_new_rk3576(int fd,
                                    rocket_conv2d_int8_weights_rk3576 *const *h,
                                    unsigned n);

int rocket_conv2d_int8_chain_run_rk3576(int fd, rocket_conv2d_int8_chain_rk3576 *c,
                                        const int8_t *in, int8_t *out);

/* Returns how many hardware kicks the most recent `rocket_conv2d_int8_chain_run_rk3576()`
 * call took. It is one unless a program had to be redone, in which case the whole chain is
 * retried. A caller measuring the chain's gain should read this rather than assume it. */
unsigned rocket_conv2d_int8_chain_kicks_rk3576(const rocket_conv2d_int8_chain_rk3576 *c);

void rocket_conv2d_int8_chain_free_rk3576(int fd, rocket_conv2d_int8_chain_rk3576 *c);

/* ---- finding the runs -------------------------------------------------------------
 *
 * Finds the maximal runs in an array of handles, for the chain constructor above, which
 * takes one run and checks it. Finding runs needs no hardware knowledge, so the library
 * provides the finder rather than each frontend writing its own. Scheduling stays with the
 * caller.
 *
 *     rocket_conv2d_int8_run_rk3576 runs[16];
 *     unsigned n = rocket_conv2d_int8_chain_plan_rk3576(handles, n_layers, runs, 16);
 *     for (i = 0; i < n && i < 16; i++)
 *         chain[i] = rocket_conv2d_int8_chain_new_rk3576(fd, handles + runs[i].first,
 *                                                        runs[i].count);
 *
 * A run is a maximal sequence of handles. Every layer in it but the last leaves a cube, and
 * every layer but the first reads a cube an earlier member wrote. The finder checks this
 * against the BO addresses. A NULL entry breaks a run, which is how a caller says "this
 * layer is not a resident convolution" or "a run cannot start here". A chain scatters the
 * tensor it is handed. A layer whose input the caller prepares on the host, such as a
 * widened or bordered copy, must therefore not begin a run.
 *
 * The finder applies the program-count bound, so it splits a run that is too long at a layer
 * boundary rather than refusing it. Pure, with no hardware and no allocation, and nothing on
 * the handles is written. Returns the number of runs of two or more layers, which can exceed
 * `max_runs`. Only the first `max_runs` are written.
 */
typedef struct {
    unsigned first;      /* index into the caller's handle array */
    unsigned count;      /* layers in the run                    */
    unsigned programs;   /* register programs those layers emit  */
} rocket_conv2d_int8_run_rk3576;

unsigned rocket_conv2d_int8_chain_plan_rk3576(rocket_conv2d_int8_weights_rk3576 *const *h,
                                              unsigned n,
                                              rocket_conv2d_int8_run_rk3576 *runs,
                                              unsigned max_runs);

/* Returns how many register programs this handle emits: its row plan's task count, which is
 * what the chain's length bound counts. Returns 0 for a handle that cannot be chained at
 * all. */
unsigned rocket_conv2d_int8_programs_rk3576(const rocket_conv2d_int8_weights_rk3576 *h);

/* ---- a pooling layer inside the stream ---------------------------------------------
 *
 * Describes a run by nodes, so a pooling layer can sit inside a chained run. A classifier
 * puts a pooling layer in the middle of a run, where a chain of convolution handles would
 * break. A pool is its own register program (PPU and PPU_RDMA, PC_OPERATION_ENABLE 0x60
 * against a convolution's 0x1D). The two bitmaps are disjoint, and still do not partition a
 * stream. A pool and the convolution after it give the same outputs chained as in separate
 * submits.
 *
 * A node sets exactly one of `conv`, `pool` and `placement`, the third kind described below.
 * A node with none set is the "this layer cannot be in a run" break that a NULL entry means
 * in the convolution-only form. The finder treats a node with more than one set the same
 * way, and the constructor refuses both.
 *
 *     rocket_chain_node_rk3576 nodes[N] = {0};
 *     nodes[i].conv = conv_handle;      // or
 *     nodes[j].pool = pool_handle;
 *     unsigned n = rocket_chain_plan_rk3576(nodes, N, runs, 16);
 *     chain[k] = rocket_chain_new_rk3576(fd, nodes + runs[k].first, runs[k].count);
 *
 * A pool can only be an interior node. A pool that began a run would need the chain to
 * scatter a row-major tensor into its cube. A pool that ended a run would need its output
 * de-tiled. Both are host work the stream exists to remove. The run finder therefore never
 * starts or ends a run on a pool, and the constructor refuses one.
 *
 * A padded average pool whose divisor can lag, as `rocket_pool_int8_rk3576_lag_can_show()`
 * reports, is not a run node, even as an interior one. A kick is one submit, so a redo for
 * the lag reruns every program of the run. The finder splits a run at such a pool, and the
 * constructor refuses one. `ROCKET_RK3576_CHAIN_POOL_LAG=1` admits it, and is unset by
 * default. A max pool, an unpadded average pool and a count-include-pad average pool stay
 * run nodes.
 *
 * A third kind of node has no program at all. A channel concatenation whose operands are
 * placed slices of one buffer is a name for an address. Its producers wrote those bytes
 * themselves, and its consumers read the buffer as a cube. There is nothing to run and
 * nothing for the host to do. Left unset, such a layer would break every run it sits in,
 * which on a graph built of concatenated branches is every run. The caller declares it
 * instead, and the finder spans it:
 *
 *     nodes[k].placement = 1;           // neither conv nor pool
 *
 * A placement node contributes zero programs and is exempt from both run conditions, since
 * it neither reads nor leaves a cube. Like a pool, it can only be interior, because one at
 * either end would extend a run by a layer that does nothing. It is the caller's statement
 * that no host work happens at this layer. The library cannot see that, as it cannot see
 * that a caller prepares a layer's input on the host.
 *
 * The convolution-only entries above are these with every node a `conv`. They exist because
 * most callers have no pooling layer to place, and they share one implementation.
 *
 * Evidence: [HW sweep, tests/rk3576_chain_pool.c], over five geometries to a 110x110 plane,
 * with both intermediates read.
 */
typedef struct {
    rocket_conv2d_int8_weights_rk3576 *conv;
    rocket_pool_int8_rk3576_handle    *pool;
    int                                placement;  /* a layer with no program at all */
} rocket_chain_node_rk3576;

rocket_conv2d_int8_chain_rk3576 *
rocket_chain_new_rk3576(int fd, const rocket_chain_node_rk3576 *nodes, unsigned n);

unsigned rocket_chain_plan_rk3576(const rocket_chain_node_rk3576 *nodes, unsigned n,
                                  rocket_conv2d_int8_run_rk3576 *runs, unsigned max_runs);

/* Returns how many register programs this node emits, or 0 for a node that cannot be
 * chained. */
unsigned rocket_chain_node_programs_rk3576(const rocket_chain_node_rk3576 *node);

/* Returns the longest program stream a chain accepts. The kernel's completion deadline
 * bounds it, not the RK3576. A batched job raises one PC_DONE for the whole kick. A driver
 * before interface 1.4 gives that job one task's deadline, past which the job retires with
 * its last program's writes still in flight. On 1.4 and later the deadline scales, and the
 * bound is the largest stream the library lays out. `ROCKET_RK3576_CHAIN_MAX_TASKS` sets it.
 */
unsigned rocket_conv2d_int8_chain_max_programs_rk3576(void);

/* Computes an fp16 -> fp16 convolution on the RK3576 through the input-channel split. One
 * fp16 task on the RK3576 contracts exactly sixteen input channels, so an arbitrary count is
 * ic/16 submits summed on the host. Each submit runs its output channels in tiles of at most
 * 32, since one float program writes no more. It uses tiles of 16 where the weight-slice
 * rule binds. The rule binds when a 16-channel group's slice is past 32 KiB on a plane that
 * needs the F=2048 allowance, such as k9 and k11.
 *
 * The entry refuses a plane whose 16-channel slice still overflows the CBUF, because the
 * split is not composed with the row window. It refuses `desc.depthwise`, because the
 * depthwise fp16 cube is not decoded. Four or fewer input channels take the packed-image
 * first conv instead, with one program per row window and 32 output channels.
 *
 * The slice programs write fp32 through a whole output stage
 * (`gen_conv2d_fp16_rk3576_f32out`), which corrupts nothing. The programs of a call
 * therefore go back to back, and the call leaves the RK3576 clean. The slices sum as fp32
 * partials and round once, to the fp16 result, and every element is exact. At k9 and k11
 * over 32 channels on 120x64 a call takes 9.5-9.9 ms, and at k9 over two slices 15.1 ms. At
 * 28x28 k3, ic 128 into oc 256 is 64 programs in 47 ms.
 *
 * `ROCKET_RK3576_FP16_F32OUT=0` restores the fp16 writer, whose own fp16 output prevents the
 * next submit from writing. The fp16 writer takes a power cycle per program, about 105 ms at
 * the stock 50 ms autosuspend, and one more on exit. The cycles made the same calls 115-227
 * ms. The packed-image first conv (ic <= 4) has not run under the fp32 stage, and still
 * cycles per program.
 *
 * The entry returns with every element written. Each program raises the DPU's own completion
 * and goes out without `ROCKET_JOB_NO_DPU_DONE`, so the driver waits for it.
 *
 * Caution: on a kernel whose wait for that completion is a deadline (interface below 1.6, or
 * `rocket.dpu_grace_us` set), a program that outruns the deadline still returns short,
 * silently.
 *
 * Evidence: [HW sweep, H96 MAX M9, rocket 1.6.0, `tests/rk3576_drain_probe gate`,
 * 2026-09-27] for the completion. The timings are [HW sweep, H96 MAX M9, rocket 1.6.0,
 * `tests/rk3576_drain_probe gate` and `rk3576_fp16_split_cost`, 2026-09-30].
 */
int rocket_conv2d_fp16_rk3576(int fd, const rocket_conv2d_desc *d,
                              const _Float16 *in, const _Float16 *W, _Float16 *out);

#ifdef __cplusplus
}
#endif
#endif /* ROCKET_CONV_H */
