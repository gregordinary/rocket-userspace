// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef ROCKET_ACTIVATION_H
#define ROCKET_ACTIVATION_H

#include <stdint.h>
#include "npu_dpu.h"   /* lut_epilogue_t (the conv->activation fusion epilogue) */

/*
 * Copyright (C) 2026  The rocket-userspace authors
 *
 * prose-lint: off
 *
 * This file is part of rocket-userspace.
 *
 * rocket-userspace is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rocket-userspace is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rocket-userspace.  If not, see <https://www.gnu.org/licenses/>.
 *
 * prose-lint: on
 */

/*
 * Declares the elementwise fp16 activations on the DPU LUT block, the elementwise binary ops,
 * the parametric activations, the LUT-epilogue builder and the RK3576 int8 activation.
 *
 * `rocket_activation_fp16` computes `out[i] = f(in[i])` for a flat fp16 vector, entirely on
 * the NPU through the NVDLA-style SDP LUT, with no convolution or matmul. The result is an
 * fp16 LUT approximation and is not bit-exact, so a test must compare it with a tolerance
 * rather than for equality. The entry pads the element count up to a multiple of 8
 * internally (the C2 cube atom).
 *
 * The [0,1]-output kinds (sigmoid, hardsigmoid and the GELU and Mish gates) share the
 * validated sigmoid LUT geometry, and only the table content differs. The smooth and
 * asymmetric-output kinds (tanh, GELU, SiLU, HardSwish) need a different output-converter
 * path, which each kind's comment below describes.
 */

#ifdef __cplusplus
extern "C" {
#endif
/* ============================================================================
 * Activation kind enumeration
 * ==========================================================================*/

enum rocket_activation_kind {
    ROCKET_ACTIVATION_SIGMOID     = 0,
    ROCKET_ACTIVATION_HARDSIGMOID = 1,
    /* gate(x) * x: a LUT pass for the gate, then an elementwise multiply, on the host by
     * default and on the NPU with `ROCKET_ACT_NPU_MUL=1` */
    ROCKET_ACTIVATION_HARDSWISH   = 2,   /* x * hardsigmoid(x) */
    ROCKET_ACTIVATION_SILU        = 3,   /* x * sigmoid(x)  (a.k.a. swish) */
    /* Signed-output kinds run a single LUT pass through the bias trick. The table stores
     * (f(x)-lo)/(hi-lo) in [0,1], the proven unsigned Q0.15 geometry. The DPU OUT_CVT then
     * affine-decodes f = q*(hi-lo)*2^-shift + offset back to the signed range. Unlike
     * HARDSWISH and SILU above, they need no EW multiply or second pass. */
    ROCKET_ACTIVATION_TANH        = 4,   /* [-1,1], odd, the keystone signed test */
    /* The standalone GELU runs the accurate 2-pass path GELU(x)=x·Φ(x). Φ is the Gaussian CDF
     * gate on the clean unit-LUT geometry (`ROCKET_ACTIVATION_GELU_GATE`). The single-pass
     * GELU spikes ~128 in the flat negative tail for wide FFN inputs. `ROCKET_ACT_WIDE_LUT`
     * forces the single pass, for reverse engineering only. The conv->activation fusion uses
     * the single-pass epilogue (`rocket_lut_epilogue_build`), which is accurate only for
     * inputs in the curved region. Cosine similarity is 1.000000 against the true erf GELU
     * over [-12,12] (`tests/gelu_rocket.c`). */
    ROCKET_ACTIVATION_GELU        = 5,   /* exact erf GELU, 2-pass x·Φ(x) (accurate, wide range) */
    /* Parametric kind: LeakyReLU(x) = x>=0 ? x : alpha*x, with a per-tensor slope `alpha`
     * rather than a fixed table. It therefore has a dedicated entry point
     * (`rocket_leaky_relu_fp16`) instead of the parameterless `rocket_activation_fp16`. One
     * DPU LUT pass runs on the natural LE/LO split (LE the alpha*x branch, LO the x branch).
     * The runtime repairs the LE/LO mux's sign-based x~0 spike on the host. It serves YOLO,
     * and ONNX LeakyRelu and PRelu with a scalar slope. */
    ROCKET_ACTIVATION_LEAKY_RELU  = 6,
    /* Positive-domain kinds: f is defined for x>0, so x=0 is outside the range and there is
     * no LE/LO mux glitch. They run on the shifted single-table LUT, where the whole domain
     * [x_lo,x_hi] maps onto the positive index half, interior to one table. The LUT grid is
     * uniform, so accuracy depends on the domain. Tune the domain to the data. RSQRT is the
     * RMSNorm/LayerNorm core, and RECIPROCAL underlies the softmax denominator and Div. */
    ROCKET_ACTIVATION_SQRT        = 7,   /* sqrt(x),  x>=0   */
    ROCKET_ACTIVATION_RSQRT       = 8,   /* 1/sqrt(x), x>0   */
    ROCKET_ACTIVATION_RECIPROCAL  = 9,   /* 1/x,       x>0   */
    /* Shifted-table kind, like the positive-domain family above, except that the domain can
     * include x<=0. The BN-ALU bias maps exp over [x_lo,x_hi] onto the LO (positive index)
     * half, so the index never crosses 0. The LE/LO mux selects on sign(x), not on the index,
     * so that mapping is not what avoids the x~0 glitch. EXP decodes an unsigned output
     * (out_lo=0), and in the kinds measured the glitch appears only with a signed decode. It
     * has no LE/LO sign-mux glitch and works standalone, unlike the single-pass affine GELU.
     * The test sweep includes x=0, the softmax row maximum [HW sweep]
     * (`tests/exp_lut_rocket.c`).
     *
     * The default domain [-16,0] is the softmax case. After the row-max subtraction, the input
     * to EXP is in (-∞,0] and the output in (0,1] (out_lo=0, S=1, Q0.15 absolute resolution
     * ~3e-5). The tiny tail terms quantize to ~0, which is harmless in the softmax sum. The
     * grid is uniform, so exp's relative interpolation error ~Δ²/8 is constant (~1e-4 at 512
     * intervals). Q0.15 sets the minimum of the low tail's relative error. Tune the domain with
     * `ROCKET_LUT_XLO/XHI`, and with `ROCKET_LUT_SEXP` for x_hi>0. */
    ROCKET_ACTIVATION_EXP         = 10,  /* exp(x), default domain [-16,0] (softmax) */
    /* Internal gate kind: the Gaussian CDF Φ(x)=0.5(1+erf(x/√2)). It is the [0,1]-output gate
     * of the 2-pass GELU, GELU(x)=x·Φ(x). Φ is monotone [0,1] like sigmoid, so it uses the
     * clean unit-LUT geometry. It does not use the flat-region-prone affine geometry that
     * makes the single-pass GELU spike. It is not meant to be called directly.
     * `ROCKET_ACTIVATION_GELU` runs it as its gate. */
    ROCKET_ACTIVATION_GELU_GATE   = 11,
    /* Shifted-table kind, like EXP: Softplus(x)=log(1+e^x), monotone (0,∞), with output ~0
     * for x≪0 and ~x for x≫0. The whole domain [x_lo,x_hi] maps onto the positive index
     * half. The default domain is [-16,16] (out [0,~16], S=32). Where |x|>x_hi, the output
     * saturates at the edge, and the high tail is linear. Softplus is the smooth gate inside
     * Mish. (TFLite/ONNX Softplus.)
     *
     * The domain straddles 0, and the kind still has no LE/LO sign-mux glitch at x~0, so it
     * works standalone. In the kinds measured, the glitch appears only with a signed output
     * decode (a negative OUT_CVT offset), and Softplus decodes an unsigned output
     * (out_lo=0). A dense sweep around x=0 shows no spike [HW sweep]
     * (`tests/x0_glitch_probe.c`). */
    ROCKET_ACTIVATION_SOFTPLUS    = 12,
    /* Internal gate kind: tanh(softplus(x)), the [0,1]-output gate of the 2-pass Mish
     * (Mish(x)=x·tanh(softplus(x))). It is monotone 0->1 like sigmoid, so it uses the clean
     * unit-LUT geometry, and the full transition fits the ±6.3 sigmoid grid. It is not called
     * directly. `ROCKET_ACTIVATION_MISH` runs it as its gate. */
    ROCKET_ACTIVATION_MISH_GATE   = 13,
    /* Gated kind, on the 2-pass path like SILU and GELU: Mish(x)=x·tanh(softplus(x)). The
     * gate is MISH_GATE, a [0,1]-output LUT, followed by an elementwise multiply by x, on the
     * host or on the NPU with `ROCKET_ACT_NPU_MUL`. Mish is the YOLOv4/v7 backbone activation.
     * (TFLite/ONNX Mish.) */
    ROCKET_ACTIVATION_MISH        = 14,
    /* Shifted-table kind with a symmetric domain: Abs(x)=|x|. The symmetric domain [-R,R]
     * puts the kink at x=0 on the exact middle table sample (j=256). It also maps everything
     * onto the positive index half. The result |x| is therefore piecewise-linear-exact, and
     * the interpolation is exact at the kink. The output is [0,R], S covers R, and |x|>R
     * saturates. (TFLite/ONNX Abs.)
     *
     * The domain straddles 0, and there is still no LE/LO mux glitch at x~0. Like Softplus,
     * Abs decodes an unsigned output (out_lo=0), and a dense sweep around x=0 shows no spike
     * [HW sweep] (`tests/x0_glitch_probe.c`). */
    ROCKET_ACTIVATION_ABS         = 15,
    /* Parametric kinds with dedicated entry points, like LEAKY_RELU: ELU(x)=x>=0?x:alpha*(e^x-1)
     * and SELU(x)=lambda*ELU_alpha(x), with fixed alpha=1.67326 and lambda=1.05070. They run on
     * the symmetric shifted single table (the Abs trick). The x=0 kink falls on the middle
     * sample, and the whole domain maps onto the positive index half.
     *
     * The output decode is signed (out_lo=-lambda*alpha), and the x~0 band meets the LE/LO
     * mux glitch, as in the other signed-output kinds measured [HW sweep]
     * (`tests/x0_glitch_probe.c` with `ROCKET_ELU_NOREPAIR=1`). The runtime repairs that band
     * on the host, as for LeakyReLU, and `ROCKET_ELU_NOREPAIR` disables the repair. The negative branch is curved (uniform-grid interpolation error ~alpha*dx^2/8).
     * Where |x|>R, the output saturates, and the positive tail is linear. The enum values are
     * for naming and reverse engineering. Call `rocket_elu_fp16` or `rocket_selu_fp16`. */
    ROCKET_ACTIVATION_ELU         = 16,
    ROCKET_ACTIVATION_SELU        = 17,
    /* Positive-domain kind, like sqrt, rsqrt and reciprocal, and the first with a signed
     * output: Log(x)=ln(x) over x>0, where log(x)<0 for x<1. It runs on the shifted single
     * table. The whole positive domain [x_lo,x_hi] maps onto the LO (positive) index half.
     * The OUT_CVT offset decodes a negative out_lo (= log(x_lo)), the same signed-output
     * machinery tanh and ELU use, here on the positive-domain path. The default domain is
     * [0.25,32] (out [log .25, log 32] ~ [-1.39, 3.47], S=8).
     *
     * There is no LE/LO mux glitch, although the decode is signed. The mux selects on
     * sign(x), and the domain x>0 never reaches 0. A log over a domain that included x<=0
     * would spike at x~0 [expected], as the other signed-output kinds do.
     *
     * The grid is uniform, so accuracy depends on the domain and is worst at the steep small-x
     * end. Absolute error is the right metric, because log is used additively and relative
     * error is ill-defined at the x=1 zero crossing. Log is the natural inverse of EXP, and
     * the per-element log for log-probabilities, NLL and cross-entropy. `ROCKET_LUT_XLO/XHI`
     * tune the domain. For a custom domain, also set `ROCKET_LUT_OFFSET/MINEXP/SEXP`.
     * (TFLite/ONNX Log.) */
    ROCKET_ACTIVATION_LOG         = 18,
};

/* ============================================================================
 * Standalone activation entry point
 * ==========================================================================*/

/* Runs the activation `kind` on the NPU. `fd` is an open rocket device (`rocket_open()`).
 * Returns 0 on success and a negative value on error.
 *
 * For the two-pass gated kinds (HardSwish, SiLU, GELU and Mish), `ROCKET_ACT_NPU_MUL=1` runs
 * the final x·gate(x) multiply on the NPU EW unit instead of on the host. The NPU multiply
 * changes the result slightly, because the NPU fp16 EW multiply rounds differently from the
 * host multiply. With it set, the output is not bit-identical to the host-multiply path. */
int rocket_activation_fp16(int fd, int kind,
                           const _Float16 *in, _Float16 *out, int n);

/* ============================================================================
 * Elementwise binary ops, clip and divide
 * ==========================================================================*/

/* Computes the elementwise fp16 binary ops on the NPU: `out[i] = a[i] * b[i]` (mul),
 * `a[i] + b[i]` (add) or `a[i] - b[i]` (sub). All use the conv-main EW path, an identity
 * matmul main feed with an ERDMA operand. The entry reshapes `n` to [M,32] and tiles `M`
 * internally. ADD is the residual-connection primitive, and MUL drives the two-pass gated
 * activations above. SUB reuses the ADD datapath with the operand negated (a-b == a+(-b), an
 * exact fp16 sign flip).
 *
 * Returns 0 on success and a negative value on error. */
int rocket_ew_mul_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);
int rocket_ew_add_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);
int rocket_ew_sub_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);

/* Computes the elementwise two-tensor MAX or MIN on the NPU: `out[i] = max/min(a[i], b[i])`.
 * It uses the same conv-main EW path as add and mul, with the DPU EW ALU algorithm set to MAX
 * or MIN (`EW_ALU_ALGO`, `DPU_EW_CFG` bits [17:16]). It covers TFLite/ONNX Maximum and
 * Minimum, and ReLU as max(x,0). Returns 0 on success and a negative value on error. */
int rocket_ew_max_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);
int rocket_ew_min_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);

/* Computes Clip(x, lo, hi) = min(max(x, lo), hi) on the NPU, as a constant-operand MAX then
 * MIN in two EW passes. It covers TFLite/ONNX Clip and the bounded-ReLU family
 * (ReLU6 = Clip(0,6)). With `fd` < 0 it computes the host reference. Returns 0 on success and
 * a negative value on error. */
int rocket_clip_fp16(int fd, float lo, float hi, const _Float16 *in, _Float16 *out, int n);

/* Computes the elementwise fp16 divide `out[i] = a[i] / b[i]` fully on the NPU, as
 * reciprocal(b) on the DPU LUT, then `a[i] * recip(b[i])` on the conv-main EW multiply. `b`
 * must be positive and within the reciprocal LUT domain, by default [2^-2, 2^5].
 * `ROCKET_LUT_XLO/XHI` override the domain. The result is an fp16 LUT approximation, so a
 * test must compare it with a tolerance. It covers TFLite/ONNX DIV. Returns 0 on success and a
 * negative value on error.
 *
 * If `b` straddles 0 or has a wide range, the caller should divide on the host instead. */
int rocket_ew_div_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);

/* ============================================================================
 * Parametric activations
 * ==========================================================================*/

/* Computes LeakyReLU(x) = x>=0 ? x : alpha*x elementwise on the DPU LUT, in one flying pass.
 * `alpha` is the per-tensor negative slope, with 0<alpha<=1. With alpha=0 the all-zero
 * negative branch trips a flat-region LUT quirk, so the caller should use plain ReLU. The
 * LUT is the natural LE/LO split (LE the alpha*x branch, LO the x branch) over [-R,R], with R
 * from `ROCKET_LEAKY_R` (default 16). The result is exact within [-R,R]. Where |x|>R, it
 * saturates at the table edge, which is rare after a normalization.
 *
 * The DPU LUT's sign-based LE/LO mux spikes at exactly x=0. The runtime repairs that
 * razor-thin band on the host, at negligible cost because the readback already streams every
 * element. `ROCKET_LEAKY_NOREPAIR` disables the repair. `n` is padded to a multiple of 8.
 * Returns 0 on success and a negative value on error. */
int rocket_leaky_relu_fp16(int fd, float alpha, const _Float16 *in, _Float16 *out, int n);

/* Computes the CPU reference for LeakyReLU(alpha), fp16-rounded, for tests. */
void rocket_leaky_relu_ref_fp16(float alpha, const _Float16 *in, _Float16 *out, int n);

/* Computes ELU(x) = x>=0 ? x : alpha*(e^x - 1) elementwise on the DPU LUT, in one flying pass
 * on the symmetric shifted single table. The x=0 kink falls on the middle sample, and the
 * runtime repairs the x~0 band on the host (`ROCKET_ELU_NOREPAIR` disables it). `alpha` must
 * be > 0, and the negative saturation level is -alpha. The result is exact to the LUT over
 * [-R,R], with R from `ROCKET_ELU_R` (default 8). R=8 captures the negative branch fully,
 * since e^-8~0.
 *
 * Where |x|>R, the output saturates at the table edge. `n` is padded to a multiple of 8.
 * Returns 0 on success and a negative value on error. (TFLite/ONNX Elu.) */
int rocket_elu_fp16(int fd, float alpha, const _Float16 *in, _Float16 *out, int n);
/* Computes SELU(x) = lambda*(x>=0 ? x : alpha*(e^x-1)) with the fixed self-normalizing
 * constants alpha=1.6732632423543772 and lambda=1.0507009873554805, on the same shifted-table
 * path as ELU. */
int rocket_selu_fp16(int fd, const _Float16 *in, _Float16 *out, int n);

/* Compute the CPU references, fp16-rounded, for tests. */
void rocket_elu_ref_fp16(float alpha, const _Float16 *in, _Float16 *out, int n);
void rocket_selu_ref_fp16(const _Float16 *in, _Float16 *out, int n);

/* Computes PReLU, `out[c][s] = x[c][s] >= 0 ? x[c][s] : alpha[c]*x[c][s]`, with a per-channel
 * negative slope (YOLO, segmentation, ONNX PRelu). The input is laid out [C][S], channel-major
 * with S spatial elements per channel, and `alpha` is fp32 [C]. It runs fully on the NPU with
 * no LUT, so there is no x~0 mux glitch.
 *
 * For the universal case of alpha in [0,1], it computes max(x, alpha_c*x): a per-channel scale
 * (`scale_rows`), then `ew_max`, in 2 passes, bit-exact. For any alpha outside [0,1], it falls
 * back to the general relu(x)+alpha_c*min(x,0) in 4 passes, also exact. With `fd` < 0 it
 * computes the host reference. Returns 0 on success and a negative value on error. */
int rocket_prelu_fp16(int fd, int C, int S, const _Float16 *x, const float *alpha, _Float16 *out);
void rocket_prelu_ref_fp16(int C, int S, const _Float16 *x, const float *alpha, _Float16 *out);

/* Computes the CPU reference for the same mathematical function as `rocket_activation_fp16`,
 * fp16-rounded, for tests. */
void rocket_activation_ref_fp16(int kind, const _Float16 *in, _Float16 *out, int n);

/* ============================================================================
 * Name lookup and the LUT-epilogue builder
 * ==========================================================================*/

const char *rocket_activation_name(int kind);

/* Builds the LE/LO LUT tables and the DPU epilogue constants for a smooth single-pass
 * activation. The kinds are SiLU, TANH and GELU, and HardSwish only over its curved knee. It
 * fills `lut` (1026 entries) and `ep`, setting `ep->lut = lut`. The caller allocates `lut`
 * and must keep it alive while `ep` is in use. The same parameters drive the standalone DPU op
 * (`rocket_activation_fp16`) and the conv->activation fusion (`conv_params_t.act`,
 * `gen_conv2d_task`).
 *
 * Returns 0 on success. If `kind` is not single-pass, it returns a negative value.
 *
 * The environment variables of the single-pass reverse-engineering sweep apply here too:
 *
 *     ROCKET_LUT_IDXSCALE  ROCKET_LUT_LO      ROCKET_LUT_SEXP     ROCKET_LUT_OFFSET
 *     ROCKET_LUT_MINEXP    ROCKET_LUT_SCALE   ROCKET_LUT_BNMUL    ROCKET_LUT_LESLOPE
 *     ROCKET_LUT_LOSLOPE   ROCKET_LUT_LOSHIFT
 */
int rocket_lut_epilogue_build(int kind, uint16_t *lut, lut_epilogue_t *ep);

/* ============================================================================
 * RK3576: the int8 nonlinear activation on the DPU LUT
 * ==========================================================================*/

/* Computes `out[c][y][x] = f(in[c][y][x])`, int8 in and int8 out, entirely on the NPU. Both
 * tensors are row-major [C][H][W], the layout the RK3576 convolution entries take. The
 * quantization is per-tensor, with model-domain signed zero points.
 *
 * `kind` is a `ROCKET_RK3576_ACT_*` from `npu_regcmd_rk3576.h`. The kinds do not include exp,
 * gelu or sqrt. No vendor capture emits a 513-entry table for any of the three, so they lower
 * some other way. A table for them here would be an invention.
 *
 * It is exact for every int8 input, rather than an interpolation of one. The op lowers onto a
 * depthwise 1x1 identity convolution whose weight is the LUT's own index step. The weight puts
 * each of the 256 possible inputs on its own table entry. The only error against an exact f()
 * is therefore the entry's own quantization, and the hardware's linear interpolation between
 * entries never runs. The test checks the chip's arithmetic against a CPU model of it, and
 * reports the distance to an exact f().
 *
 * The op runs two programs in one submit. The tables live in the LUT RAM, and a DPU-only
 * program bursts them there as an earlier task of the same submit. A second submit can take a
 * runtime-PM cycle in between and lose them.
 *
 * It is available on the RK3576 only and refuses elsewhere. Returns 0 or a negative rocket
 * error.
 *
 * Evidence: [HW sweep, H96 MAX M9] (`tests/rk3576_act_gate.c`).
 */
int rocket_act_int8_rk3576(int fd, int kind, unsigned c, unsigned h, unsigned w,
                           const int8_t *in, float in_scale, int in_zp,
                           int8_t *out, float out_scale, int out_zp);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_ACTIVATION_H */
