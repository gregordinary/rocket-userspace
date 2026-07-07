// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef ROCKET_ACTIVATION_H
#define ROCKET_ACTIVATION_H

#include <stdint.h>
#include "npu_dpu.h"   /* lut_epilogue_t (the conv->activation fusion epilogue) */

/*
 * Copyright (C) 2026  The rocket-userspace authors
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
 */

/*
 * Elementwise fp16 activation on the DPU LUT block.
 *
 * out[i] = f(in[i]) for a flat fp16 vector, computed entirely on the NPU via the
 * NVDLA-style SDP LUT (no conv/matmul). The result is an fp16 LUT approximation,
 * not bit-exact — gate it with a tolerance, not equality. Element count is padded
 * up to a multiple of 8 internally (the C2 cube atom).
 *
 * Today's kinds output in [0,1] and share the validated sigmoid LUT geometry
 * (only the table content differs). Smooth/asymmetric-output kinds (tanh, GELU,
 * SiLU, HardSwish) need a different output-converter path and land separately.
 */

#ifdef __cplusplus
extern "C" {
#endif
/* ============================================================================
 * SECTION — Activation kind enumeration
 * ==========================================================================*/

enum rocket_activation_kind {
    ROCKET_ACTIVATION_SIGMOID     = 0,
    ROCKET_ACTIVATION_HARDSIGMOID = 1,
    /* gate(x) * x, computed as a LUT pass then an on-NPU elementwise multiply */
    ROCKET_ACTIVATION_HARDSWISH   = 2,   /* x * hardsigmoid(x) */
    ROCKET_ACTIVATION_SILU        = 3,   /* x * sigmoid(x)  (a.k.a. swish) */
    /* SIGNED-OUTPUT kinds: a single LUT pass via the bias trick — the table stores
     * (f(x)-lo)/(hi-lo) in [0,1] (the proven unsigned Q0.15 geometry) and the DPU
     * OUT_CVT affine-decodes f = q*(hi-lo)*2^-shift + offset back to the signed range.
     * No EW multiply / second pass (unlike HARDSWISH/SILU above). */
    ROCKET_ACTIVATION_TANH        = 4,   /* [-1,1], odd; the keystone signed test */
    /* GELU: the standalone op runs the ACCURATE 2-pass route GELU(x)=x·Φ(x) (Φ = the Gaussian
     * CDF gate on the clean unit-LUT geometry; see ROCKET_ACTIVATION_GELU_GATE) — the single-
     * pass GELU (build_lut_affine) spikes ~128 in the flat negative tail (QUIRK 1) for wide FFN
     * inputs. ROCKET_ACT_WIDE_LUT forces the single-pass (RE only). The conv→act FUSION still
     * uses the single-pass epilogue (rocket_lut_epilogue_build) — fine only for curved-region
     * inputs. cos=1.000000 vs true erf-GELU over [-12,12] (tests/gelu_rocket.c). */
    ROCKET_ACTIVATION_GELU        = 5,   /* exact erf GELU; 2-pass x·Φ(x) (accurate, wide range) */
    /* PARAMETRIC: LeakyReLU(x) = x>=0 ? x : alpha*x. Per-tensor slope `alpha` (not a fixed
     * table), so it has a dedicated entry point (rocket_leaky_relu_fp16) rather than the
     * parameterless rocket_activation_fp16. Single DPU LUT pass on the natural LE/LO split
     * (LE=alpha*x branch, LO=x branch); the LE/LO mux's sign-based x≈0 spike is repaired on
     * the host. (YOLO; ONNX LeakyRelu/PRelu with a scalar slope.) */
    ROCKET_ACTIVATION_LEAKY_RELU  = 6,
    /* POSITIVE-DOMAIN kinds: f defined for x>0 (no x=0 in range ⇒ no LE/LO mux glitch),
     * realised with the SHIFTED single-table LUT (the whole domain [x_lo,x_hi] maps onto
     * the positive index half, interior to one table). Uniform-grid LUT ⇒ accuracy is
     * domain-bounded (tune the domain to the data)
     * RSQRT is the RMSNorm/LayerNorm core; RECIPROCAL underlies softmax-denominator / Div. */
    ROCKET_ACTIVATION_SQRT        = 7,   /* sqrt(x),  x>=0   */
    ROCKET_ACTIVATION_RSQRT       = 8,   /* 1/sqrt(x), x>0   */
    ROCKET_ACTIVATION_RECIPROCAL  = 9,   /* 1/x,       x>0   */
    /* SHIFTED-TABLE kind, like the positive-domain family above but the domain may include
     * x<=0: exp over [x_lo,x_hi] maps onto the LO (positive index) half via the BN-ALU bias,
     * so x never crosses 0 in the INDEX domain ⇒ no LE/LO sign-mux glitch ⇒ works standalone
     * (unlike the build_lut_affine GELU). Default domain [-16,0] is the softmax case: after
     * the row-max subtraction EXP's input is in (-∞,0], output (0,1] (out_lo=0, S=1, Q0.15
     * abs res ~3e-5 — the tiny tail terms quantize to ~0, harmless in the softmax sum).
     * Uniform grid ⇒ exp's relative INTERP error ~Δ²/8 is constant (~1e-4 at 512 cells); the
     * low tail's relative error is Q0.15-floored ().
     * Tune the domain with ROCKET_LUT_XLO/XHI (+ ROCKET_LUT_SEXP for x_hi>0). */
    ROCKET_ACTIVATION_EXP         = 10,  /* exp(x); default domain [-16,0] (softmax) */
    /* INTERNAL gate kind: the Gaussian CDF Φ(x)=0.5(1+erf(x/√2)), the [0,1]-output gate of
     * the 2-pass GELU (GELU(x)=x·Φ(x)). Φ is monotone [0,1] like sigmoid, so it uses the
     * CLEAN unit-LUT geometry (build_lut_unit) — NOT the flat-region-prone build_lut_affine
     * that makes the SINGLE-pass GELU spike. Not meant to be called directly; GELU routes here. */
    ROCKET_ACTIVATION_GELU_GATE   = 11,
    /* SHIFTED-TABLE kind (like EXP): Softplus(x)=log(1+e^x), monotone (0,∞), output ≈0 for
     * x≪0 and ≈x for x≫0. The whole domain [x_lo,x_hi] maps onto the positive index half, so
     * no LE/LO sign-mux glitch (works standalone). Default domain [-16,16] (out [0,~16], S=32);
     * |x|>x_hi saturates at the edge (the high tail is linear — slope-extrapolation is a
     * follow-on). The smooth gate inside Mish. (TFLite/ONNX Softplus.) */
    ROCKET_ACTIVATION_SOFTPLUS    = 12,
    /* INTERNAL gate kind: tanh(softplus(x)), the [0,1]-output gate of the 2-pass Mish
     * (Mish(x)=x·tanh(softplus(x))). Monotone 0→1 like sigmoid, so it uses the CLEAN unit-LUT
     * geometry (build_lut_unit) — the full transition fits the ±6.3 sigmoid grid. Not called
     * directly; MISH routes here. */
    ROCKET_ACTIVATION_MISH_GATE   = 13,
    /* GATED kind (the 2-pass route, like SILU/GELU): Mish(x)=x·tanh(softplus(x)). gate =
     * MISH_GATE (a [0,1]-output LUT) then an elementwise multiply by x (host, or on the NPU
     * with ROCKET_ACT_NPU_MUL). The YOLOv4/v7 backbone activation. (TFLite/ONNX Mish.) */
    ROCKET_ACTIVATION_MISH        = 14,
    /* SHIFTED-TABLE kind, SYMMETRIC domain: Abs(x)=|x|. The symmetric domain [-R,R] puts the
     * kink at x=0 on the EXACT middle table sample (j=256) and maps everything onto the positive
     * index half — so |x| is piecewise-linear-exact (interp exact at the kink) with NO LE/LO mux
     * glitch. out [0,R], S covers R; |x|>R saturates. (TFLite/ONNX Abs.) */
    ROCKET_ACTIVATION_ABS         = 15,
    /* PARAMETRIC, dedicated entry points (like LEAKY_RELU): ELU(x)=x>=0?x:alpha*(e^x-1) and
     * SELU(x)=lambda*ELU_alpha(x) (fixed alpha=1.67326, lambda=1.05070). Realised on the
     * SYMMETRIC shifted single-table (the Abs trick) — the x=0 kink lands on the middle sample
     * and the whole domain maps onto the positive index half, so there is NO LE/LO mux glitch
     * (no host repair, unlike LeakyReLU). The negative branch is curved (uniform-grid interp
     * error ~alpha*dx^2/8); |x|>R saturates (the positive tail is linear — slope-extrapolation
     * is a follow-on). The enum values are for naming/RE; call rocket_elu_fp16/rocket_selu_fp16. */
    ROCKET_ACTIVATION_ELU         = 16,
    ROCKET_ACTIVATION_SELU        = 17,
    /* POSITIVE-DOMAIN kind (like sqrt/rsqrt/reciprocal) but the FIRST with a SIGNED output:
     * Log(x)=ln(x) over x>0, log(x)<0 for x<1. Realised on the shifted single-table (the whole
     * positive domain [x_lo,x_hi] maps onto the LO/positive index half ⇒ no LE/LO mux glitch),
     * with a NEGATIVE out_lo (= log(x_lo)) decoded by the OUT_CVT offset — the same signed-output
     * machinery tanh/ELU use, now exercised on the positive-domain path. Default domain [0.25,32]
     * (out [log .25, log 32] ≈ [-1.39, 3.47], S=8); uniform-grid ⇒ accuracy is domain-bounded and
     * worst at the steep small-x end (ABSOLUTE error is the right metric — log is used additively,
     * and relative error is ill-defined at the x=1 zero crossing). The natural inverse of EXP and
     * the per-element log for log-probabilities / NLL / cross-entropy. ROCKET_LUT_XLO/XHI tune the
     * domain (also set ROCKET_LUT_OFFSET/MINEXP/SEXP for a custom domain). (TFLite/ONNX Log.) */
    ROCKET_ACTIVATION_LOG         = 18,
};

/* ============================================================================
 * SECTION — Standalone activation entry point
 * ==========================================================================*/

/* Run the activation on the NPU. fd is an open rocket device (rocket_open()).
 * Returns 0 on success, negative on error.
 *
 * NUMERICS KNOB: for the two-pass gated kinds (HardSwish / GELU / Mish), ROCKET_ACT_NPU_MUL=1
 * does the final x·gate(x) multiply on the NPU EW unit instead of on the host. This CHANGES
 * THE RESULT slightly (the NPU fp16 EW-mul rounds differently from the host multiply) — set
 * it knowing the output is no longer bit-identical to the host-multiply path. */
int rocket_activation_fp16(int fd, int kind,
                           const _Float16 *in, _Float16 *out, int n);

/* ============================================================================
 * SECTION — Elementwise binary ops (mul/add/sub/max/min/clip/div)
 * ==========================================================================*/

/* Elementwise fp16 binary ops on the NPU: out[i] = a[i] * b[i] (mul) / a[i] + b[i] (add) /
 * a[i] - b[i] (sub). All use the conv-main EW path (identity matmul main feed + ERDMA
 * operand); n is reshaped to [M,32] and M-tiled internally. ADD is the residual-connection
 * primitive; MUL drives the two-pass gated activations above; SUB reuses the ADD datapath
 * with the operand negated (a-b == a+(-b); exact fp16 sign flip). 0 on success, <0 on error. */
int rocket_ew_mul_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);
int rocket_ew_add_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);
int rocket_ew_sub_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);

/* Elementwise two-tensor MAX/MIN on the NPU: out[i] = max/min(a[i], b[i]). Same conv-main EW
 * path as add/mul, with the DPU EW ALU algo set to MAX/MIN (EW_ALU_ALGO, DPU_EW_CFG bits
 * [17:16]). Covers TFLite/ONNX Maximum/Minimum (and ReLU = max(x,0)). 0 ok, <0 on error. */
int rocket_ew_max_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);
int rocket_ew_min_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);

/* Clip(x, lo, hi) = min(max(x, lo), hi) on the NPU (constant-operand MAX then MIN, two EW
 * passes). Covers TFLite/ONNX Clip and the bounded-ReLU family (ReLU6 = Clip(0,6)). fd<0 =
 * host reference. 0 on success, <0 on error. */
int rocket_clip_fp16(int fd, float lo, float hi, const _Float16 *in, _Float16 *out, int n);

/* Elementwise fp16 DIVIDE: out[i] = a[i] / b[i], fully on the NPU = reciprocal(b) (DPU LUT)
 * then a[i] * recip(b[i]) (the conv-main EW multiply). `b` must be POSITIVE and within the
 * reciprocal LUT domain (default [2^-4, 2^5]; ROCKET_LUT_XLO/XHI override) — it is an fp16
 * LUT approximation, so gate with a tolerance. Covers TFLite/ONNX DIV. 0 on success, <0 on
 * error. (For a b that straddles 0 or has wide range, host divide stays the right call.) */
int rocket_ew_div_fp16(int fd, const _Float16 *a, const _Float16 *b,
                       _Float16 *out, int n);

/* ============================================================================
 * SECTION — Parametric activations (LeakyReLU / ELU / SELU / PReLU)
 * ==========================================================================*/

/* LeakyReLU(x) = x>=0 ? x : alpha*x, elementwise on the DPU LUT (one flying pass).
 * `alpha` is the per-tensor negative slope (0<alpha<=1; for alpha=0 use plain ReLU — its
 * all-zero negative branch trips a flat-region LUT quirk). Natural LE/LO LUT (LE=alpha*x
 * branch, LO=x branch) over [-R,R] (R from ROCKET_LEAKY_R, default 16); exact within
 * [-R,R], |x|>R saturates at the table edge (rare post-norm; slope-extrapolation is a
 * follow-on). The DPU LUT's sign-based LE/LO mux spikes at exactly x=0; the runtime repairs
 * that razor-thin band on the host (the readback already streams every element, so the cost
 * is negligible; ROCKET_LEAKY_NOREPAIR disables it). n is padded to a multiple of 8.
 * Returns 0, negative on error. */
int rocket_leaky_relu_fp16(int fd, float alpha, const _Float16 *in, _Float16 *out, int n);

/* CPU reference for LeakyReLU(alpha) (fp16-rounded), for gates. */
void rocket_leaky_relu_ref_fp16(float alpha, const _Float16 *in, _Float16 *out, int n);

/* ELU(x) = x>=0 ? x : alpha*(e^x - 1), elementwise on the DPU LUT (one flying pass, the
 * SYMMETRIC shifted single-table — the x=0 kink on the middle sample, no LE/LO mux glitch).
 * `alpha` > 0 (the negative saturation level is -alpha). Exact-to-LUT over [-R,R] (R from
 * ROCKET_ELU_R, default 8; the negative branch is fully captured by R=8 since e^-8~0); |x|>R
 * saturates at the table edge. n padded to a multiple of 8. Returns 0, negative on error.
 * (TFLite/ONNX Elu.) */
int rocket_elu_fp16(int fd, float alpha, const _Float16 *in, _Float16 *out, int n);
/* SELU(x) = lambda*(x>=0 ? x : alpha*(e^x-1)) with the fixed self-normalizing constants
 * alpha=1.6732632423543772, lambda=1.0507009873554805. Same shifted-table path as ELU. */
int rocket_selu_fp16(int fd, const _Float16 *in, _Float16 *out, int n);

/* CPU references (fp16-rounded), for gates. */
void rocket_elu_ref_fp16(float alpha, const _Float16 *in, _Float16 *out, int n);
void rocket_selu_ref_fp16(const _Float16 *in, _Float16 *out, int n);

/* PReLU: out[c][s] = x[c][s] >= 0 ? x[c][s] : alpha[c]*x[c][s], a PER-CHANNEL negative slope
 * (YOLO, segmentation; ONNX PRelu). Input laid out [C][S] (channel-major, S spatial elements
 * per channel), alpha fp32 [C]. Fully on the NPU with NO LUT (so no x≈0 mux glitch): for the
 * universal alpha in [0,1] case it is max(x, alpha_c*x) (a per-channel scale = scale_rows, then
 * ew_max) — 2 passes, bit-exact; for any alpha outside [0,1] it falls back to the general
 * relu(x)+alpha_c*min(x,0) (4 passes), also exact. fd<0 = host reference. 0 ok, <0 on error. */
int rocket_prelu_fp16(int fd, int C, int S, const _Float16 *x, const float *alpha, _Float16 *out);
void rocket_prelu_ref_fp16(int C, int S, const _Float16 *x, const float *alpha, _Float16 *out);

/* CPU reference for the same mathematical function (fp16-rounded), for gates. */
void rocket_activation_ref_fp16(int kind, const _Float16 *in, _Float16 *out, int n);

/* ============================================================================
 * SECTION — Name lookup & LUT-epilogue builder
 * ==========================================================================*/

const char *rocket_activation_name(int kind);

/* Build the LE/LO LUT tables + DPU epilogue constants for a SMOOTH single-pass
 * activation (SiLU / TANH / GELU; HardSwish only over its curved knee). Fills `lut`
 * (1026 entries, caller-owned and kept alive while `ep` is used) and `ep` (with
 * ep->lut = lut). The SAME params drive the standalone DPU op (rocket_activation_fp16)
 * and the conv->activation fusion (conv_params_t.act, gen_conv2d_task). The single-pass
 * RE sweep env knobs (ROCKET_LUT_IDXSCALE/LO/SEXP/OFFSET/MINEXP/SCALE/BNMUL/LESLOPE/
 * LOSLOPE/LOSHIFT) apply here too. Returns 0, or <0 if kind is not single-pass. */
int rocket_lut_epilogue_build(int kind, uint16_t *lut, lut_epilogue_t *ep);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_ACTIVATION_H */
