// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_activation.c — elementwise fp16 activation on the DPU LUT block.
 *
 * Wraps gen_lut_activation_fp16 (npu_regcmd.c) with the host side: it builds the
 * per-activation LUT tables + converter constants, lays the input out as a flat
 * fp16 cube, submits the single DPU flying-mode job, and reads the result back.
 *
 * The LUT mechanism is the NVDLA SDP design (two 513-entry tables, LE for the
 * negative branch and LO for the positive branch, sampled on a uniform grid of
 * step = 32/index_scale in the input domain; the hardware linearly interpolates
 * between adjacent entries and extrapolates with a slope outside the table). The
 * sigmoid geometry constants (index_scale 2596, BN multiplier 0x6912, the OUT_CVT that
 * maps the Q0.15 result back to fp16) are HW-verified: the activation is bit-exact on
 * device, index_scale 2596 places the table over x in +/-6.31 (sigmoid saturated), and
 * the BN/OUT_CVT operand format was pinned by a BNALU sweep. The table CONTENTS are
 * computed from the activation itself (build_lut_unit). HardSigmoid reuses the exact
 * same geometry — only the table content changes — because its output is also in [0,1].
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_activation.h"
#include "npu_activation.h"
#include "npu_matmul.h"   /* gen_matmul_fp16 + feature_data/weight_fp16 (on-NPU EW mul) */
#include "rocket_log.h"     // centralized log channel

#define CBUF_BANK 4096   /* pad BOs so a feature DMA never reads past the alloc */

/* ============================================================================
 * SECTION — Host math references and CPU oracles
 * ==========================================================================*/

/* fp16 bit pattern of a value, for the BN_MUL index-scale operand. */
static uint16_t fp16_bits(double v)
{
    _Float16 h = (_Float16)v;
    uint16_t b;
    memcpy(&b, &h, sizeof b);
    return b;
}

static double hardsigmoid_d(double x)
{
    double y = x / 6.0 + 0.5;            /* TFLite HARD_SIGMOID: clip(x/6+0.5,0,1) */
    return y < 0.0 ? 0.0 : (y > 1.0 ? 1.0 : y);
}

static double act_eval(int kind, double x)
{
    switch (kind) {
    case ROCKET_ACTIVATION_SIGMOID:     return 1.0 / (1.0 + exp(-x));
    case ROCKET_ACTIVATION_HARDSIGMOID: return hardsigmoid_d(x);
    case ROCKET_ACTIVATION_HARDSWISH:   return x * hardsigmoid_d(x);
    case ROCKET_ACTIVATION_SILU:        return x * (1.0 / (1.0 + exp(-x)));
    case ROCKET_ACTIVATION_TANH:        return tanh(x);
    case ROCKET_ACTIVATION_GELU:        return 0.5 * x * (1.0 + erf(x / 1.4142135623730951));
    case ROCKET_ACTIVATION_SQRT:        return x > 0.0 ? sqrt(x) : 0.0;
    case ROCKET_ACTIVATION_RSQRT:       return x > 0.0 ? 1.0 / sqrt(x) : 0.0;
    case ROCKET_ACTIVATION_RECIPROCAL:  return x != 0.0 ? 1.0 / x : 0.0;
    case ROCKET_ACTIVATION_LOG:         return x > 0.0 ? log(x) : 0.0;   /* domain x>0 (gate feeds x>0) */
    case ROCKET_ACTIVATION_EXP:         return exp(x);
    case ROCKET_ACTIVATION_GELU_GATE:   return 0.5 * (1.0 + erf(x / 1.4142135623730951)); /* Φ(x) */
    /* Softplus = log(1+e^x), stable form (avoid overflow for x>0: = x + log1p(e^-x)). */
    case ROCKET_ACTIVATION_SOFTPLUS:    return x > 0.0 ? x + log1p(exp(-x)) : log1p(exp(x));
    /* Mish gate = tanh(softplus(x)) (the [0,1] gate); Mish = x·tanh(softplus(x)). */
    case ROCKET_ACTIVATION_MISH_GATE:   return tanh(x > 0.0 ? x + log1p(exp(-x)) : log1p(exp(x)));
    case ROCKET_ACTIVATION_MISH:        return x * tanh(x > 0.0 ? x + log1p(exp(-x)) : log1p(exp(x)));
    case ROCKET_ACTIVATION_ABS:         return fabs(x);
    default: return 0.0;
    }
}

/* The SHIFTED-table kinds (SQRT/RSQRT/RECIPROCAL + EXP): f realised with the SHIFTED
 * single-table LUT over [x_lo,x_hi] — the whole domain maps onto the positive LUT index
 * half via the BN-ALU bias, so x never crosses 0 IN THE INDEX domain ⇒ no LE/LO sign-mux
 * glitch (works standalone). SQRT/RSQRT/RECIPROCAL are defined for x>0 (positive domain);
 * EXP is defined for all x and its default domain INCLUDES x<=0 — the shifting is what makes
 * that safe. The output range [out_lo, out_lo+2^Sexp] covers f over the domain. Domain
 * defaults are tuned to typical magnitudes; ROCKET_LUT_XLO/XHI override. Returns 1 (and fills
 * params) if `kind` is a shifted-table kind, else 0. */
static int act_shifted_domain(int kind, double *x_lo, double *x_hi, double *out_lo, int *Sexp)
{
    /* Domains tuned so the UNIFORM 512-cell grid resolves the steep low end: the low-end
     * relative interpolation error ~ (Δ/x_lo)²·c/8 stays ~1-2% for a domain ratio ~128×
     * placed away from 0. Inputs below x_lo clamp to f(x_lo); above x_hi clamp to f(x_hi).
     * Tune to your data with ROCKET_LUT_XLO/XHI (e.g. a wider RMSNorm-operand range). */
    switch (kind) {
    case ROCKET_ACTIVATION_SQRT:       *x_lo = 0.25; *x_hi = 64.0; *out_lo = 0.0; *Sexp = 3; break; /* out [0.5,8]   */
    case ROCKET_ACTIVATION_RSQRT:      *x_lo = 0.50; *x_hi = 64.0; *out_lo = 0.0; *Sexp = 1; break; /* out [0.125,1.41] */
    case ROCKET_ACTIVATION_RECIPROCAL: *x_lo = 0.25; *x_hi = 32.0; *out_lo = 0.0; *Sexp = 2; break; /* out [0.03,4]  */
    /* LOG: SIGNED output (log<0 for x<1) — out_lo=log(x_lo) (negative), S=8 covers
     * [log .25, log 32]≈[-1.39,3.47]. First act_shifted_domain kind with out_lo!=0; the
     * OUT_CVT offset (line ~353) decodes the signed range, same as tanh/ELU. */
    case ROCKET_ACTIVATION_LOG:        *x_lo = 0.25; *x_hi = 32.0; *out_lo = log(0.25); *Sexp = 3; break; /* out [-1.39,3.47] */
    /* EXP for softmax: input in (-∞,0] after the row-max subtraction, output (0,1]. S=1 (out
     * in [0,1]); exp's constant-relative interp error (~1e-4 over 512 cells) dominates near
     * x=0 where it matters, the Q0.15 floor only hits the negligible tail. x_hi=0 clamps tiny
     * positive (fp16-rounding) inputs to exp(0)=1, correct. General exp: widen via env. */
    case ROCKET_ACTIVATION_EXP:        *x_lo = -16.0; *x_hi = 0.0; *out_lo = 0.0; *Sexp = 0; break; /* out [~0,1] */
    /* Softplus: monotone (0,∞). [-16,16] -> out [~0,16] (softplus(16)≈16), S=32. The low tail
     * (softplus≈0) is the same flat-tail-to-zero shape EXP handles via the q>=1 floor. */
    case ROCKET_ACTIVATION_SOFTPLUS:   *x_lo = -16.0; *x_hi = 16.0; *out_lo = 0.0; *Sexp = 5; break; /* out [0,16] */
    /* Abs: SYMMETRIC domain so the x=0 kink lands on the middle sample (exact). out [0,16], S=32. */
    case ROCKET_ACTIVATION_ABS:        *x_lo = -16.0; *x_hi = 16.0; *out_lo = 0.0; *Sexp = 5; break; /* out [0,16] */
    default: return 0;
    }
    return 1;
}

const char *rocket_activation_name(int kind)
{
    switch (kind) {
    case ROCKET_ACTIVATION_SIGMOID:     return "sigmoid";
    case ROCKET_ACTIVATION_HARDSIGMOID: return "hardsigmoid";
    case ROCKET_ACTIVATION_HARDSWISH:   return "hardswish";
    case ROCKET_ACTIVATION_SILU:        return "silu";
    case ROCKET_ACTIVATION_TANH:        return "tanh";
    case ROCKET_ACTIVATION_GELU:        return "gelu";
    case ROCKET_ACTIVATION_LEAKY_RELU:  return "leaky_relu";
    case ROCKET_ACTIVATION_SQRT:        return "sqrt";
    case ROCKET_ACTIVATION_RSQRT:       return "rsqrt";
    case ROCKET_ACTIVATION_RECIPROCAL:  return "reciprocal";
    case ROCKET_ACTIVATION_LOG:         return "log";
    case ROCKET_ACTIVATION_EXP:         return "exp";
    case ROCKET_ACTIVATION_GELU_GATE:   return "gelu_gate";
    case ROCKET_ACTIVATION_SOFTPLUS:    return "softplus";
    case ROCKET_ACTIVATION_MISH_GATE:   return "mish_gate";
    case ROCKET_ACTIVATION_MISH:        return "mish";
    case ROCKET_ACTIVATION_ABS:         return "abs";
    case ROCKET_ACTIVATION_ELU:         return "elu";
    case ROCKET_ACTIVATION_SELU:        return "selu";
    default: return "?";
    }
}

void rocket_activation_ref_fp16(int kind, const _Float16 *in, _Float16 *out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = (_Float16)act_eval(kind, (double)(float)in[i]);
}

void rocket_leaky_relu_ref_fp16(float alpha, const _Float16 *in, _Float16 *out, int n)
{
    for (int i = 0; i < n; i++) {
        float x = (float)in[i];
        out[i] = (_Float16)(x >= 0.f ? x : alpha * x);
    }
}

/* SELU self-normalizing constants (Klambauer et al. 2017). */
#define SELU_ALPHA  1.6732632423543772
#define SELU_LAMBDA 1.0507009873554805

static double elu_d(double alpha, double lambda, double x)
{
    return lambda * (x >= 0.0 ? x : alpha * (exp(x) - 1.0));
}

void rocket_elu_ref_fp16(float alpha, const _Float16 *in, _Float16 *out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = (_Float16)elu_d((double)alpha, 1.0, (double)(float)in[i]);
}

void rocket_selu_ref_fp16(const _Float16 *in, _Float16 *out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = (_Float16)elu_d(SELU_ALPHA, SELU_LAMBDA, (double)(float)in[i]);
}

void rocket_prelu_ref_fp16(int C, int S, const _Float16 *x, const float *alpha, _Float16 *out)
{
    for (int c = 0; c < C; c++) {
        float a = (float)(_Float16)alpha[c];   /* fp16-rounded slope (matches the NPU broadcast) */
        for (int j = 0; j < S; j++) {
            float v = (float)x[(size_t)c * S + j];
            out[(size_t)c * S + j] = (_Float16)(v >= 0.f ? v : a * v);
        }
    }
}

/* ============================================================================
 * SECTION — DPU LUT table builders
 * ==========================================================================*/

/* Build the LE+LO Q0.15 tables for a [0,1]-output activation on the sigmoid grid.
 *   LE[i] = f( -((512-i)*step) )   inputs -range .. 0   (table id 0)
 *   LO[i] = f(    i*step       )   inputs 0 .. +range    (table id 1)
 * with step = 32/index_scale. Computed directly from f; for f = sigmoid this is the
 * standard sigmoid LUT, and any other f with codomain [0,1] (e.g. hardsigmoid) drops in. */
static void build_lut_unit(int kind, double index_scale, uint16_t lut[1026])
{
    const double step = 32.0 / index_scale;
    for (int i = 0; i <= 512; i++) {
        double y = act_eval(kind, -((double)(512 - i) * step));
        long q = lround(y * 32768.0);
        if (q < 0) q = 0;
        if (q > 32767) q = 32767;
        lut[i] = (uint16_t)q;
    }
    for (int i = 0; i <= 512; i++) {
        double y = act_eval(kind, (double)i * step);
        long q = lround(y * 32768.0);
        if (q < 0) q = 0;
        if (q > 32767) q = 32767;
        lut[513 + i] = (uint16_t)q;
    }
}

/* Build the LE+LO tables for a SIGNED [lo,hi]-output activation via the bias trick:
 * store the normalized g(x) = (f(x) - lo) / (hi - lo) in [0,1] (the proven unsigned
 * Q0.15 grid), and let the DPU OUT_CVT affine-decode f back. For an odd function with
 * codomain [-1,1] (tanh) this is the standard (tanh+1)/2 bias encoding. The OUT_CVT then
 * recovers f = q*2^-shift*(hi-lo) + lo (the runtime programs shift/scale/offset). */
static void build_lut_affine(int kind, double index_scale, double lo, double hi,
                             uint16_t lut[1026])
{
    const double step = 32.0 / index_scale;
    const double span = (hi - lo) != 0.0 ? (hi - lo) : 1.0;
    for (int i = 0; i <= 512; i++) {
        double g = (act_eval(kind, -((double)(512 - i) * step)) - lo) / span;
        long q = lround(g * 32768.0);
        if (q < 0) q = 0;
        if (q > 32767) q = 32767;
        lut[i] = (uint16_t)q;
    }
    for (int i = 0; i <= 512; i++) {
        double g = (act_eval(kind, (double)i * step) - lo) / span;
        long q = lround(g * 32768.0);
        if (q < 0) q = 0;
        if (q > 32767) q = 32767;
        lut[513 + i] = (uint16_t)q;
    }
}

/* fp32 bit pattern (for the BN_ALU pre-scale bias on the shifted single-table path). */
static uint32_t fp32_bits(double v) { float f=(float)v; uint32_t b; memcpy(&b,&f,4); return b; }

/* SHIFTED single-table build (the x≈0 LE/LO boundary-glitch mitigation). Instead of
 * splitting LE=[neg,0] / LO=[0,pos] at the data's zero (where the hybrid mux glitches),
 * map the whole input domain [x_lo,x_hi] onto the POSITIVE index range via a BN-ALU bias:
 *   index = (x + B)*scale,  B = -x_lo,  scale = 16384/(x_hi-x_lo)
 * so x in [x_lo,x_hi] -> index in [0,16384], and a data range strictly inside [x_lo,x_hi]
 * is INTERIOR (index never 0 or 16384) -> no LE/LO boundary, no overflow edge. The LO table
 * holds f over the full domain; LE is never hit (filled flat = LO[0]). OUT_CVT (out_lo,S) is
 * unchanged (it decodes f's OUTPUT range). Table sample j -> x = x_lo + j*(x_hi-x_lo)/512.
 *
 * HW FACT (found 2026-06-22 bringing up EXP): a LUT table entry of EXACTLY q=0 mis-decodes
 * to a garbage constant (~4.0 in the fp16 output) — NOT 0. EXP's deep tail (exp(x)<~1.5e-5,
 * x<~-11 for the [-16,0] domain) quantizes to q=0 and so read ~4 instead of ~0, blowing up
 * the softmax sum. Flooring every entry to q>=1 fixes it: the floored value decodes to ~3e-5
 * (rounds to ~0 on readback), correct for the tail. sqrt/rsqrt/reciprocal never produce a
 * q=0 entry over their domains, so the floor is a no-op for them. ROCKET_LUT_QFLOOR overrides
 * (0 reproduces the raw glitch for RE). */
static void build_lut_shifted(int kind, double x_lo, double x_hi, double out_lo, double S,
                              uint16_t lut[1026])
{
    const double dx = (x_hi - x_lo) / 512.0;
    long qfloor = 1;   /* a q=0 entry mis-decodes to ~4 (HW); floor to 1 (== ~3e-5 out) */
    { const char *e = getenv("ROCKET_LUT_QFLOOR"); if (e) qfloor = strtol(e, NULL, 0); }
    for (int j = 0; j <= 512; j++) {
        double g = (act_eval(kind, x_lo + j * dx) - out_lo) / S;
        long q = lround(g * 32768.0);
        if (q < qfloor) q = qfloor;
        if (q > 32767) q = 32767;
        lut[513 + j] = (uint16_t)q;          /* LO = the full-range table */
    }
    for (int j = 0; j <= 512; j++) lut[j] = lut[513];   /* LE never hit (index>=0): flat */
}

/* SHIFTED single-table build for LeakyReLU(alpha) over a SYMMETRIC domain [-R,R]. The
 * symmetric domain is the point: the x=0 kink lands on an EXACT table sample (j=256, the
 * table middle) AND the BN-ALU bias puts it at the positive-index interior — so there is
 * neither an interpolation error at the kink NOR an x≈0 LE/LO mux boundary. LeakyReLU is
 * linear on each side, so linear interpolation between samples is exact: the result is
 * exact (Q0.15 + fp16 quant) for x in [-R,R]; |x|>R saturates at the edge value. The output
 * span is [-R*alpha, R]; S (a power of two) covers it. Mirrors build_lut_shifted but with
 * the parametric f. */
/* Build LeakyReLU into the natural LE/LO split: LE (table id 0) is the NEGATIVE branch
 * (x in [-range,0], slope alpha), LO (table id 1) the POSITIVE branch (x in [0,range],
 * slope 1) — a perfect structural fit for the kink at 0. Both encode g=(f-out_lo)/S in
 * Q0.15; the affine OUT_CVT decodes f back. step = 32/index_scale in the x domain; |x|>range
 * saturates at the table edge. The LE/LO mux glitches at exactly x=0 (sign-based, HW quirk);
 * the runtime repairs that razor-thin band on the host. */
static void build_lut_leaky(double alpha, double index_scale, double out_lo, double S,
                            uint16_t lut[1026])
{
    const double step = 32.0 / index_scale;
    for (int i = 0; i <= 512; i++) {                     /* LE: x in [-range, 0] */
        double x = -((double)(512 - i)) * step;
        double f = (x >= 0.0) ? x : alpha * x;
        long q = lround((f - out_lo) / S * 32768.0);
        if (q < 0) q = 0;
        if (q > 32767) q = 32767;
        lut[i] = (uint16_t)q;
    }
    for (int i = 0; i <= 512; i++) {                     /* LO: x in [0, +range] */
        double x = (double)i * step;
        double f = (x >= 0.0) ? x : alpha * x;
        long q = lround((f - out_lo) / S * 32768.0);
        if (q < 0) q = 0;
        if (q > 32767) q = 32767;
        lut[513 + i] = (uint16_t)q;
    }
}

/* SHIFTED single-table build for ELU/SELU over a SYMMETRIC domain [-R,R] (the Abs trick):
 * the x=0 kink lands on the middle sample (j=256) and the BN-ALU bias maps the whole domain
 * onto the positive LUT index half, so there is NO LE/LO sign-mux glitch (no host repair).
 * f(x) = lambda*(x>=0 ? x : alpha*(e^x-1)); the curved negative branch interpolates on the
 * uniform grid (error ~ lambda*alpha*dx^2/8). out_lo is the negative saturation (-lambda*alpha);
 * S (a power of two) covers [out_lo, lambda*R]. Like build_lut_shifted, every entry is floored
 * to q>=1 (a q=0 entry mis-decodes to ~4 on this HW — the EXP finding). */
static void build_lut_shifted_elu(double alpha, double lambda, double x_lo, double x_hi,
                                  double out_lo, double S, uint16_t lut[1026])
{
    const double dx = (x_hi - x_lo) / 512.0;
    long qfloor = 1;
    { const char *e = getenv("ROCKET_LUT_QFLOOR"); if (e) qfloor = strtol(e, NULL, 0); }
    for (int j = 0; j <= 512; j++) {
        double f = elu_d(alpha, lambda, x_lo + j * dx);
        long q = lround((f - out_lo) / S * 32768.0);
        if (q < qfloor) q = qfloor;
        if (q > 32767) q = 32767;
        lut[513 + j] = (uint16_t)q;            /* LO = the full-range table */
    }
    for (int j = 0; j <= 512; j++) lut[j] = lut[513];   /* LE never hit (index>=0): flat */
}

/* ============================================================================
 * SECTION — LUT epilogue parameter builder
 * ==========================================================================*/

/* Single source of truth for the SMOOTH single-pass LUT params (the bias-trick +
 * affine OUT_CVT). Drives BOTH the standalone DPU op and the conv->activation fusion,
 * so the two can never drift (the signed/wide-LUT split is handled here). */
int rocket_lut_epilogue_build(int kind, uint16_t *lut, lut_epilogue_t *ep)
{
    if (!lut || !ep) return -1;
    memset(ep, 0, sizeof *ep);
    ep->kind            = kind;     /* host reference only (oracle); generator ignores */
    ep->lut             = lut;
    ep->bn_alu_cfg      = 0x80000000;   /* fp32 -0.0: a no-op BN ALU bias add        */
    ep->out_cvt_scale   = 1;
    ep->out_cvt_cvt_type= 1;
    ep->lut_le_start    = 0xffffc000;   /* -16384 (Q index)                          */
    ep->lut_lo_end      = 0x00004000;   /*  16384                                    */
    ep->le_index_select = 5;            /* step = 2^5 = 32 in the Q index domain     */
    ep->lo_index_select = 5;

    /* POSITIVE-DOMAIN kinds (sqrt / rsqrt / reciprocal): ALWAYS the shifted single-table
     * mode — the whole positive domain [x_lo,x_hi] maps onto the LO (positive index) half,
     * so x never crosses 0 and the LE/LO sign mux never glitches. Uniform-grid LUT, so
     * accuracy is bounded by the domain width (tune to the data; ROCKET_LUT_XLO/XHI). */
    {
        double x_lo, x_hi, out_lo; int Sexp;
        if (act_shifted_domain(kind, &x_lo, &x_hi, &out_lo, &Sexp)) {
            const char *e;
            if ((e = getenv("ROCKET_LUT_XLO")))  x_lo = atof(e);
            if ((e = getenv("ROCKET_LUT_XHI")))  x_hi = atof(e);
            if ((e = getenv("ROCKET_LUT_SEXP"))) Sexp = atoi(e);
            const double S = (double)(1u << Sexp);
            const double scale = 16384.0 / (x_hi - x_lo);
            build_lut_shifted(kind, x_lo, x_hi, out_lo, S, lut);
            ep->bn_mul_operand    = fp16_bits(scale);
            ep->bn_alu_cfg        = fp32_bits(-x_lo * scale);     /* index = (x-x_lo)*scale */
            ep->out_cvt_minus_exp = (uint8_t)lround(15.0 - log2(S));
            ep->out_cvt_offset    = (uint32_t)(int32_t)lround(out_lo * 32768.0 / S);
            if ((e = getenv("ROCKET_LUT_OFFSET"))) ep->out_cvt_offset    = (uint32_t)strtol(e, NULL, 0);
            if ((e = getenv("ROCKET_LUT_MINEXP"))) ep->out_cvt_minus_exp = (uint8_t)strtoul(e, NULL, 0);
            return 0;
        }
    }

    /* Per-kind geometry. tanh: [-1,1], S=2, saturating tail (tuned LE slope). silu/
     * gelu: asymptotic (no flat run), wide +-16 table, S=32 — for x beyond the table
     * the LO tail saturates (slope 0), an approximation for |x|>16 (rare post-norm).
     * hardswish: KNEE-ONLY +-3 (S=4) — the NVDLA flat-tail quirk forbids its x<=-3
     * constant run, so the fused/standalone single-pass covers only the curved knee. */
    /* SHIFTED single-table mode (ROCKET_LUT_SHIFT): map the whole domain onto the
     * positive index half (data interior to one table) so there is NO x≈0 LE/LO
     * boundary to glitch. BN-ALU adds B=-x_lo (fp32, pre-scale), BN-MUL scales by
     * 16384/(x_hi-x_lo); OUT_CVT (out_lo,S) decodes f's output range as usual. */
    if (getenv("ROCKET_LUT_SHIFT")) {
        double x_lo, x_hi, out_lo, S;
        switch (kind) {
        case ROCKET_ACTIVATION_TANH: x_lo=-8;  x_hi=8;  out_lo=-1.0; S=2.0;  break;
        case ROCKET_ACTIVATION_SILU: x_lo=-12; x_hi=12; out_lo=-0.5; S=16.0; break;
        case ROCKET_ACTIVATION_GELU: x_lo=-12; x_hi=12; out_lo=-0.5; S=16.0; break;
        default: return -3;
        }
        { const char *e;
          if ((e=getenv("ROCKET_LUT_XLO"))) x_lo=atof(e);
          if ((e=getenv("ROCKET_LUT_XHI"))) x_hi=atof(e); }
        double scale = 16384.0 / (x_hi - x_lo);
        build_lut_shifted(kind, x_lo, x_hi, out_lo, S, lut);
        ep->bn_mul_operand    = fp16_bits(scale);
        /* index = x*BN_MUL + BN_ALU, the BN ALU bias added POST-scale (index domain) as
         * fp32 — HW-confirmed by the BNALU sweep (0x46000000=fp32(8192) gave tanh
         * ACC 0.0007, x0-spikes 0). So B = -x_lo*scale (= 8192, the index centre, for a
         * symmetric domain: x=0 maps to the table middle, far from either edge). */
        ep->bn_alu_cfg        = fp32_bits(-x_lo * scale);
        ep->out_cvt_minus_exp = (uint8_t)lround(15.0 - log2(S));
        ep->out_cvt_offset    = (uint32_t)(int32_t)lround(out_lo * 32768.0 / S);
        { const char *e;
          if ((e=getenv("ROCKET_LUT_BNALU")))  ep->bn_alu_cfg     = (uint32_t)strtoul(e, NULL, 0);
          if ((e=getenv("ROCKET_LUT_BNMUL")))  ep->bn_mul_operand = (uint16_t)strtoul(e, NULL, 0);
          if ((e=getenv("ROCKET_LUT_OFFSET"))) ep->out_cvt_offset = (uint32_t)strtol(e, NULL, 0);
          if ((e=getenv("ROCKET_LUT_MINEXP"))) ep->out_cvt_minus_exp = (uint8_t)strtoul(e, NULL, 0); }
        return 0;
    }

    double index_scale, lo; int Sexp;
    uint16_t le_slope_sc = 0, lo_slope_sc = 0; uint8_t le_slope_sh = 0, lo_slope_sh = 0;
    switch (kind) {
    case ROCKET_ACTIVATION_TANH:
        index_scale = 5216.0; lo = -1.0; Sexp = 1;
        le_slope_sc = 23107; le_slope_sh = 22; break;
    case ROCKET_ACTIVATION_SILU:
        index_scale = 1024.0; lo = -0.5; Sexp = 5; break;
    case ROCKET_ACTIVATION_GELU:        /* SiLU-shaped (smooth, x*CDF); same grid     */
        index_scale = 1024.0; lo = -0.5; Sexp = 5; break;
    case ROCKET_ACTIVATION_HARDSWISH:
        index_scale = 5461.0; lo = -0.5; Sexp = 2; break;
    default: return -3;
    }
    { const char *e;
      if ((e = getenv("ROCKET_LUT_IDXSCALE"))) index_scale = atof(e);
      if ((e = getenv("ROCKET_LUT_LO")))       lo          = atof(e);
      if ((e = getenv("ROCKET_LUT_SEXP")))     Sexp        = atoi(e); }
    const double S = (double)(1u << Sexp);
    build_lut_affine(kind, index_scale, lo, lo + S, lut);
    ep->bn_mul_operand    = fp16_bits(index_scale);
    ep->out_cvt_minus_exp = (uint8_t)(15 - Sexp);
    ep->out_cvt_offset    = (uint32_t)(int32_t)lround(lo * 32768.0 / S);
    ep->le_slope_scale = le_slope_sc; ep->le_slope_shift = le_slope_sh;
    ep->lo_slope_scale = lo_slope_sc; ep->lo_slope_shift = lo_slope_sh;
    { const char *e;
      if ((e = getenv("ROCKET_LUT_OFFSET")))  ep->out_cvt_offset    = (uint32_t)strtol(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_MINEXP")))  ep->out_cvt_minus_exp = (uint8_t)strtoul(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_SCALE")))   ep->out_cvt_scale     = (uint16_t)strtoul(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_BNMUL")))   ep->bn_mul_operand    = (uint16_t)strtoul(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_LESLOPE"))) ep->le_slope_scale    = (uint16_t)strtoul(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_LOSLOPE"))) ep->lo_slope_scale    = (uint16_t)strtoul(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_LOSHIFT"))) ep->lo_slope_shift    = (uint8_t)strtoul(e, NULL, 0); }
    return 0;
}

/* ============================================================================
 * SECTION — DPU LUT job runner
 * ==========================================================================*/

/* Run a populated DPU LUT job: alloc BOs, lay the flat fp16 input cube, gen the regcmd,
 * submit, fence, read back. `p` must already carry .n (padded to 8), .lut and every
 * BN/OUT_CVT/LUT/slope constant; this fills .input_dma/.output_dma/.tasks. Shared by the
 * enum activations and the parametric LeakyReLU. n = real element count (<= p->n). */
/* Per-op element cap for the LUT tiling. The DPU_DATA_CUBE_WIDTH (cols=n/8) is 13 bits, so the
 * hard ceiling is n/8 <= 0x1FFF => n <= 65528. We tile WELL UNDER that (32768 => cols 4096):
 * riding the exact 8191 max width mis-reads a thin tail of cube positions (HW-confirmed: a
 * 65528 chunk corrupts ~54 elements; 32768 chunks are clean). Bigger vectors (a transformer's
 * [M,I] activation cube is millions of elements) are tiled over this cap, so every activation
 * caller — SiLU/sigmoid/tanh/GELU/leaky/the positive-domain family — works at any n. */
#define DPU_LUT_MAXN 32768

static int run_dpu_lut(int fd, lut_act_params_t *p, const _Float16 *in, _Float16 *out, int n)
{
    rocket_bo guard = {0}, in_bo = {0}, out_bo = {0}, rc_bo = {0};
    uint64_t  regs[2048] = {0};
    int rc = -1;

    /* alloc the BOs once at the tile cap (or the whole vector if smaller) and reuse them
     * across chunks — the LUT params in `p` are identical per chunk, only width (n) changes. */
    int cap = n < DPU_LUT_MAXN ? n : DPU_LUT_MAXN;
    int cap_np = (cap + 7) & ~7;
    if (rocket_bo_alloc(fd, 4096, &guard) ||                                   /* off IOVA 0 */
        rocket_bo_alloc(fd, (size_t)cap_np * sizeof(_Float16) + CBUF_BANK, &in_bo) ||
        rocket_bo_alloc(fd, (size_t)cap_np * sizeof(_Float16) + CBUF_BANK, &out_bo) ||
        rocket_bo_alloc(fd, sizeof(regs), &rc_bo)) {
        ROCKET_LOGE("run_dpu_lut: BO alloc failed\n");
        goto out;
    }
    if (((in_bo.dma_address + in_bo.size) | (out_bo.dma_address + out_bo.size) |
         (rc_bo.dma_address + rc_bo.size)) >> 32) {
        ROCKET_LOGE("run_dpu_lut: a BO dma_address exceeds 32 bits\n");
        goto out;
    }

    for (int base = 0; base < n; base += DPU_LUT_MAXN) {
        int cn = n - base; if (cn > DPU_LUT_MAXN) cn = DPU_LUT_MAXN;
        p->n          = (cn + 7) & ~7;          /* pad chunk to the C2=8 atom */
        p->input_dma  = in_bo.dma_address;
        p->output_dma = out_bo.dma_address;
        p->tasks      = regs;

        rocket_bo_prep(fd, &in_bo, 1, 0);
        memset(in_bo.ptr, 0, in_bo.size);
        memcpy(in_bo.ptr, in + base, (size_t)cn * sizeof(_Float16));
        rocket_bo_fini(fd, &in_bo);

        int g = gen_lut_activation_fp16(p);
        if (g != 0 || p->task_count <= 0 ||
            (size_t)p->task_count > sizeof(regs)/sizeof(regs[0])) {   /* bound the stack regs[] */
            ROCKET_LOGE("run_dpu_lut: gen failed (%d, count=%d)\n", g, p->task_count);
            goto out;
        }
        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, regs, (size_t)p->task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);

        rocket_bo_prep(fd, &out_bo, 1, 0);
        memset(out_bo.ptr, 0, out_bo.size);
        rocket_bo_fini(fd, &out_bo);

        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo.dma_address,
                                  .regcmd_count = (uint32_t)p->task_count };
        uint32_t in_h[]  = { in_bo.handle, rc_bo.handle };
        uint32_t out_h[] = { out_bo.handle };
        if (rocket_submit_tasks(fd, &task, 1, in_h, 2, out_h, 1)) {
            ROCKET_LOGE("run_dpu_lut: submit failed\n"); goto out;
        }
        if (rocket_bo_prep(fd, &out_bo, 0, 2000000000ULL)) {   /* 2s wait */
            ROCKET_LOGE("run_dpu_lut: wait timeout\n");
            goto out;
        }
        memcpy(out + base, out_bo.ptr, (size_t)cn * sizeof(_Float16));
        rocket_bo_fini(fd, &out_bo);
    }
    rc = 0;

out:
    rocket_bo_free(fd, &out_bo);
    rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &in_bo);
    rocket_bo_free(fd, &guard);
    return rc;
}

/* ============================================================================
 * SECTION — Public single-pass activations (enum + LeakyReLU)
 * ==========================================================================*/

int rocket_activation_fp16(int fd, int kind, const _Float16 *in, _Float16 *out, int n)
{
    if (fd < 0 || n <= 0) return -1;

    /* Gated activations: f(x) = x * gate(x). The nonlinear gate (a [0,1]-output
     * LUT activation) runs ON THE NPU. The elementwise multiply by x can run
     * either on the host (default) or fully on the NPU (ROCKET_ACT_NPU_MUL=1).
     *
     * The fully-on-NPU multiply needs a conv MAIN feed (the rocket DPU EW reads
     * its second operand via ERDMA EW_BASE + MRDMA SRC_BASE + COMB_USE(5)
     * *combined with the conv/CACC result*; a pure flying-MRDMA-main + ERDMA-
     * operand pair reads the operand as 0 — confirmed by the Mesa Teflon
     * add_tensor RE). `rocket_ew_mul_fp16` now drives exactly that path (an
     * identity-conv main + EW_OP_TYPE=1 multiply) and is HW-validated bit-exact
     * (tests/ew_mul_rocket.c) — so ROCKET_ACT_NPU_MUL=1 gives a fully-on-NPU
     * HardSwish/SiLU. The host multiply stays the DEFAULT because a separate EW
     * job costs an extra NPU round-trip while the host mul is ~n cheap multiplies
     * (the real perf win is FUSING the mul into the producing conv — future
     * work).
     *
     * ROCKET_ACT_WIDE_LUT selects instead the SINGLE-PASS wide-output LUT (below): one
     * DPU LUT job that emits the FULL x*gate(x) directly via the affine OUT_CVT — no separate
     * multiply / second round-trip. SiLU's wide-LUT is HW-validated; HardSwish's needs the
     * knee-only-table NVDLA-LUT-quirk mitigation (its exactly-0 x<=-3 tail is a flat run that
     * trips the LE/LO mux) — see the single-pass block below. */
    /* GELU joins HardSwish/SiLU on the robust 2-pass path: GELU(x) = x·Φ(x), gate = the
     * Gaussian CDF Φ (GELU_GATE), a monotone [0,1] function on the CLEAN unit-LUT geometry.
     * This is the accurate on-NPU GELU — the SINGLE-pass GELU (build_lut_affine, below) spikes
     * in the flat negative tail (QUIRK 1), useless for wide FFN inputs. (ROCKET_ACT_WIDE_LUT
     * still selects the single-pass for RE.) */
    if ((kind == ROCKET_ACTIVATION_HARDSWISH || kind == ROCKET_ACTIVATION_SILU ||
         kind == ROCKET_ACTIVATION_GELU || kind == ROCKET_ACTIVATION_MISH) &&
        !getenv("ROCKET_ACT_WIDE_LUT")) {
        int gate = (kind == ROCKET_ACTIVATION_HARDSWISH) ? ROCKET_ACTIVATION_HARDSIGMOID
                 : (kind == ROCKET_ACTIVATION_GELU)      ? ROCKET_ACTIVATION_GELU_GATE
                 : (kind == ROCKET_ACTIVATION_MISH)      ? ROCKET_ACTIVATION_MISH_GATE
                 : ROCKET_ACTIVATION_SIGMOID;
        _Float16 *g = malloc((size_t)n * sizeof(_Float16));
        if (!g) return ROCKET_E_NOMEM;   /* was -2; disambiguated from the tiling/-2 sentinel */
        int r = rocket_activation_fp16(fd, gate, in, g, n);
        if (!r) {
            if (getenv("ROCKET_ACT_NPU_MUL")) {
                r = rocket_ew_mul_fp16(fd, in, g, out, n);   /* fully on NPU */
            } else {
                for (int i = 0; i < n; i++)
                    out[i] = (_Float16)((float)in[i] * (float)g[i]);
            }
        }
        free(g);
        return r;
    }

    if (kind != ROCKET_ACTIVATION_SIGMOID && kind != ROCKET_ACTIVATION_HARDSIGMOID &&
        kind != ROCKET_ACTIVATION_TANH && kind != ROCKET_ACTIVATION_HARDSWISH &&
        kind != ROCKET_ACTIVATION_SILU && kind != ROCKET_ACTIVATION_GELU &&
        kind != ROCKET_ACTIVATION_SQRT && kind != ROCKET_ACTIVATION_RSQRT &&
        kind != ROCKET_ACTIVATION_RECIPROCAL && kind != ROCKET_ACTIVATION_EXP &&
        kind != ROCKET_ACTIVATION_LOG &&
        kind != ROCKET_ACTIVATION_GELU_GATE && kind != ROCKET_ACTIVATION_SOFTPLUS &&
        kind != ROCKET_ACTIVATION_MISH_GATE && kind != ROCKET_ACTIVATION_ABS)
        return -3;   /* single-pass LUT kinds (HARDSWISH/SILU here only via WIDE_LUT) */

    const int np = (n + 7) & ~7;          /* pad element count to the C2=8 atom */

    uint16_t lut[1026];                   /* stack-local: the LUT is rebuilt per call, so a
                                             function-static buffer would race across threads */
    lut_act_params_t p = {0};
    p.n               = np;
    p.lut             = lut;
    p.bn_alu_cfg      = 0x80000000;
    p.out_cvt_scale   = 1;
    p.out_cvt_cvt_type  = 1;
    p.lut_le_start    = 0xffffc000;
    p.lut_lo_end      = 0x00004000;
    p.le_slope_scale  = 23107;
    p.le_slope_shift  = 22;
    p.le_index_select = 5;        /* step = 2^5 = 32 in the Q index domain       */
    p.lo_index_select = 5;

    if (kind == ROCKET_ACTIVATION_SIGMOID || kind == ROCKET_ACTIVATION_HARDSIGMOID ||
        kind == ROCKET_ACTIVATION_GELU_GATE || kind == ROCKET_ACTIVATION_MISH_GATE) {
        /* The [0,1]-output kinds share the validated sigmoid geometry (sigmoid / hardsigmoid /
         * the Gaussian CDF Φ = the GELU 2-pass gate / tanh(softplus) = the Mish gate) — only the
         * table content differs. Each spans its full 0→1 transition within the ±6.3 grid (the Mish
         * gate saturates even faster than sigmoid), same flat tails as sigmoid (which the unit-LUT
         * le_slope extrapolation handles cleanly — no QUIRK-1 spike). */
        build_lut_unit(kind, 2596.0, lut);
        p.bn_mul_operand    = 0x6912;   /* fp16(2596) — input -> LUT index scale */
        p.out_cvt_offset    = 1;        /* +1 LSB rounding bias (Q domain)        */
        p.out_cvt_minus_exp = 15;       /* Q0.15 -> fp16: multiply by 2^-15       */
    } else {
        /* SMOOTH single-pass kinds (TANH/SILU/GELU; HardSwish only over its curved knee)
         * via the bias-trick + affine OUT_CVT. The shared rocket_lut_epilogue_build is
         * the SINGLE SOURCE of these params — it also drives the conv->activation fusion
         * (conv_params_t.act), so the standalone op and the fused conv can never diverge.
         * Geometry rationale + the NVDLA flat-tail quirk live in the builder. */
        lut_epilogue_t ep;
        int b = rocket_lut_epilogue_build(kind, lut, &ep);
        if (b) return b;
        p.bn_mul_operand    = ep.bn_mul_operand;
        p.bn_alu_cfg        = ep.bn_alu_cfg;
        p.out_cvt_offset    = ep.out_cvt_offset;
        p.out_cvt_scale     = ep.out_cvt_scale;
        p.out_cvt_minus_exp = ep.out_cvt_minus_exp;
        p.out_cvt_cvt_type  = ep.out_cvt_cvt_type;
        p.lut_le_start      = ep.lut_le_start;
        p.lut_lo_end        = ep.lut_lo_end;
        p.le_slope_scale    = ep.le_slope_scale;
        p.le_slope_shift    = ep.le_slope_shift;
        p.lo_slope_scale    = ep.lo_slope_scale;
        p.lo_slope_shift    = ep.lo_slope_shift;
        p.le_index_select   = ep.le_index_select;
        p.lo_index_select   = ep.lo_index_select;
    }

    /* HW-RE sweep knobs (env): override the OUT_CVT affine + index scale without a
     * recompile, to pin the DPU output-converter semantics for the signed kinds. */
    { const char *e;
      if ((e = getenv("ROCKET_LUT_OFFSET")))  p.out_cvt_offset    = (uint32_t)strtol(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_MINEXP")))  p.out_cvt_minus_exp = (uint8_t)strtoul(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_SCALE")))   p.out_cvt_scale     = (uint16_t)strtoul(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_BNMUL")))   p.bn_mul_operand    = (uint16_t)strtoul(e, NULL, 0); }

    return run_dpu_lut(fd, &p, in, out, n);   /* alloc + lay input + submit + read back */
}

int rocket_leaky_relu_fp16(int fd, float alpha, const _Float16 *in, _Float16 *out, int n)
{
    if (fd < 0 || n <= 0) return -1;
    if (alpha <= 0.f) return -3;   /* alpha=0 is plain ReLU (all-zero/flat negative branch
                                    * trips the flat-region LUT quirk); use the native ReLU. */

    /* R = the table half-width (ROCKET_LEAKY_R, default 16; covers post-norm activations).
     * Natural LE/LO split: index = x*index_scale, index_scale = 16384/R, so x in [-R,R]
     * maps to index [-16384,16384] = LE table (x<0) + LO table (x>=0). |x|>R saturates at
     * the table edge (a slope-extrapolation refinement is a follow-on). The LE/LO mux
     * glitches at exactly x=0 (sign-based — see below); repaired on the host after readback. */
    double R = 16.0;
    { const char *e = getenv("ROCKET_LEAKY_R"); if (e) { double r = atof(e); if (r >= 1.0) R = r; } }
    const double index_scale = 16384.0 / R;

    /* Output span [-R*alpha, R]; S = smallest power of two covering it (exact OUT_CVT). */
    const double out_lo = -(double)alpha * R;
    const double need   = R - out_lo;                  /* = R*(1+alpha) */
    int Sexp = 0; while ((double)(1u << Sexp) < need) Sexp++;
    const double S = (double)(1u << Sexp);

    const int np = (n + 7) & ~7;
    static uint16_t lut[1026];
    build_lut_leaky((double)alpha, index_scale, out_lo, S, lut);

    lut_act_params_t p = {0};
    p.n                 = np;
    p.lut               = lut;
    p.bn_mul_operand    = fp16_bits(index_scale);
    p.bn_alu_cfg        = 0x80000000;                  /* fp32 -0.0: a no-op BN ALU bias */
    p.out_cvt_scale     = 1;
    p.out_cvt_cvt_type  = 1;
    p.out_cvt_minus_exp = (uint8_t)(15 - Sexp);
    p.out_cvt_offset    = (uint32_t)(int32_t)lround(out_lo * 32768.0 / S);
    p.lut_le_start      = 0xffffc000;
    p.lut_lo_end        = 0x00004000;
    p.le_index_select   = 5;
    p.lo_index_select   = 5;
    /* slopes left 0: |x|>R saturates at the table edge (rare post-norm). */

    { const char *e;   /* RE/tuning knobs (shared names with the smooth single-pass path) */
      if ((e = getenv("ROCKET_LUT_OFFSET")))  p.out_cvt_offset    = (uint32_t)strtol(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_MINEXP")))  p.out_cvt_minus_exp = (uint8_t)strtoul(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_BNMUL")))   p.bn_mul_operand    = (uint16_t)strtoul(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_BNALU")))   p.bn_alu_cfg        = (uint32_t)strtoul(e, NULL, 0); }

    int r = run_dpu_lut(fd, &p, in, out, n);
    if (r) return r;

    /* x≈0 REPAIR. The DPU LUT's LE/LO hybrid mux is selected by the SIGN of the
     * BN-scaled input (i.e. the sign of x), so at x≈0 it mis-toggles and emits a spurious
     * spike — independent of any index bias (confirmed: the spike follows x=0 wherever it
     * maps). It is a razor-thin band (~±R/8192). The readback already streams every
     * element, so we patch that band with the exact host value at negligible cost. Unless
     * disabled (ROCKET_LEAKY_NOREPAIR), this makes LeakyReLU correct for ALL inputs.
     */
    if (!getenv("ROCKET_LEAKY_NOREPAIR")) {
        const float guard = (float)(R / 1024.0);   /* >> the ~R/8192 glitch band, still tiny */
        for (int i = 0; i < n; i++) {
            float x = (float)in[i];
            if (x <= guard && x >= -guard)
                out[i] = (_Float16)(x >= 0.f ? x : alpha * x);
        }
    }
    return 0;
}

/* ============================================================================
 * SECTION — Elementwise binary ops (conv-main EW datapath)
 * ==========================================================================*/

/* Fully-on-NPU elementwise fp16 BINARY op: out[i] = a[i] (op) b[i],
 *   op = EW_OP_MUL (multiply) / EW_OP_ADD (add) / EW_OP_SUB (subtract).
 * Shared body for rocket_ew_mul_fp16 / rocket_ew_add_fp16 / rocket_ew_sub_fp16.
 *
 * Mechanism (the conv-main EW op the Teflon RE prescribed): an IDENTITY matmul
 * C[M,NN] = a[M,NN] · I[NN,NN]^T reproduces `a` into the DPU conv accumulator (CACC)
 * as the EW MAIN feed, then the EW unit combines it with the operand `b`, delivered via
 * ERDMA EW_BASE + MRDMA SRC_BASE + COMB_USE(5). gen_matmul_fp16 accumulate=1 selects the
 * op: ew_mul=1 => EW_OP_TYPE=1 (the fp16 eltwise-multiply word); ew_mul=0 => the proven
 * fp16 K-accum eltwise-ADD. SUBTRACT reuses the ADD datapath with the operand negated
 * during packing (a-b == a+(-b); fp16 negate is an exact sign-bit flip), so it inherits
 * the add path bit-for-bit. The EW runs through ERDMA + MRDMA with COMB_USE(5) so it has a
 * main feed; a flying-MRDMA gen_ew_mul with no main feed does not fire. HW-validated
 * bit-exact (tests/ew_mul_rocket.c). The flat
 * vector is reshaped to [M,NN] with NN=32 (K%32, N%16 satisfied) and M-tiled to fit CBUF. */
enum { EW_OP_ADD = 0, EW_OP_MUL = 1, EW_OP_SUB = 2, EW_OP_MAX = 3, EW_OP_MIN = 4 };
static int ew_binary_fp16(int fd, const _Float16 *a, const _Float16 *b,
                          _Float16 *out, int n, int op)
{
    if (fd < 0 || n <= 0) return -1;

    const int NN = 32;                 /* N == K; identity I_32                    */
    /* rows/tile cap: gen_matmul_task asserts feature_grains (= M+1) <= 0x3FF, so
     * M <= 1022; 1020 is the largest %4 value under that (CBUF is not the limit
     * here — M=1020,K=32 is ~2 banks). */
    const int M_TILE = 1020;

    /* The K-accum EW operand geometry floors the surface stride at MAX(M,12), so
     * M<12 mis-strides the upper channel surfaces (see gen_matmul_fp16). Keep
     * M>=12 (and %4) for every tile — the extra rows are zero-padded and unread. */
    const int M_MIN = 12;
    int total_rows = (n + NN - 1) / NN;
    int alloc_M = total_rows < M_TILE ? total_rows : M_TILE;
    if (alloc_M < M_MIN) alloc_M = M_MIN;
    alloc_M = (alloc_M + 3) & ~3;      /* M%4 (gen_matmul_fp16 requirement)        */

    rocket_bo wt = {0}, in_bo = {0}, op_bo = {0}, out_bo = {0}, rc_bo = {0};
    uint64_t  regs[256] = {0};
    int rc = -1;

    size_t cube_sz = (size_t)alloc_M * NN * sizeof(_Float16) + CBUF_BANK;
    size_t wt_sz   = (size_t)NN * NN * sizeof(_Float16) + CBUF_BANK;
    if (rocket_bo_alloc(fd, wt_sz,   &wt)     ||
        rocket_bo_alloc(fd, cube_sz, &in_bo)  ||
        rocket_bo_alloc(fd, cube_sz, &op_bo)  ||
        rocket_bo_alloc(fd, cube_sz, &out_bo) ||
        rocket_bo_alloc(fd, sizeof(regs), &rc_bo)) {
        ROCKET_LOGE("rocket_ew_mul_fp16: BO alloc failed\n");
        goto out;
    }
    if (((in_bo.dma_address + in_bo.size) | (op_bo.dma_address + op_bo.size) |
         (out_bo.dma_address + out_bo.size) | (rc_bo.dma_address + rc_bo.size) |
         (wt.dma_address + wt.size)) >> 32) {
        ROCKET_LOGE("rocket_ew_mul_fp16: a BO dma_address exceeds 32 bits\n");
        goto out;
    }

    /* identity weights I_NN (built once, reused for every tile) */
    rocket_bo_prep(fd, &wt, 1, 0);
    memset(wt.ptr, 0, wt.size);
    for (int kk = 1; kk <= NN; kk++)
        for (int c = 1; c <= NN; c++)
            ((_Float16 *)wt.ptr)[weight_fp16(NN, kk, c)] = (_Float16)(kk == c ? 1.0f : 0.0f);
    rocket_bo_fini(fd, &wt);

    for (int base = 0; base < n; base += M_TILE * NN) {
        int telems = n - base; if (telems > M_TILE * NN) telems = M_TILE * NN;
        int M = (telems + NN - 1) / NN;
        if (M < M_MIN) M = M_MIN;
        M = (M + 3) & ~3;

        /* pack a -> feature cube, b -> operand cube (output layout); zero the pad */
        rocket_bo_prep(fd, &in_bo, 1, 0); memset(in_bo.ptr, 0, in_bo.size);
        rocket_bo_prep(fd, &op_bo, 1, 0); memset(op_bo.ptr, 0, op_bo.size);
        {
            _Float16 *ip = in_bo.ptr, *bp = op_bo.ptr;
            for (int t = 0; t < telems; t++) {
                int pos = feature_data(NN, M, 1, 8, (t % NN) + 1, (t / NN) + 1, 1);
                ip[pos] = a[base + t];
                bp[pos] = (op == EW_OP_SUB) ? (_Float16)(-(float)b[base + t]) : b[base + t];
            }
        }
        rocket_bo_fini(fd, &in_bo);
        rocket_bo_fini(fd, &op_bo);

        rocket_bo_prep(fd, &out_bo, 1, 0); memset(out_bo.ptr, 0, out_bo.size); rocket_bo_fini(fd, &out_bo);

        matmul_params_t p = { .m = (uint16_t)M, .k = (uint16_t)NN, .n = (uint16_t)NN,
            .input_dma   = (uint32_t)in_bo.dma_address,
            .weights_dma = (uint32_t)wt.dma_address,
            .output_dma  = (uint32_t)out_bo.dma_address,
            .tasks = regs, .fp32tofp16 = 1,
            .accumulate = 1, .ew_mul = (op == EW_OP_MUL),
            .ew_op = (op == EW_OP_MAX) ? 2 : (op == EW_OP_MIN) ? 3 : 0,
            .add_dma = (uint32_t)op_bo.dma_address };
        int g = gen_matmul_fp16(&p);
        if (g != 0 || p.task_count <= 0 ||
            (size_t)p.task_count > sizeof(regs)/sizeof(regs[0])) {   /* bound the stack regs[] */
            ROCKET_LOGE("rocket_ew_mul_fp16: gen failed (%d, count=%d)\n", g, p.task_count);
            goto out;
        }

        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);

        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo.dma_address,
                                  .regcmd_count = (uint32_t)p.task_count };
        uint32_t in_h[]  = { in_bo.handle, wt.handle, rc_bo.handle, op_bo.handle };
        uint32_t out_h[] = { out_bo.handle };
        if (rocket_submit_tasks(fd, &task, 1, in_h, 4, out_h, 1)) {
            ROCKET_LOGE("rocket_ew_mul_fp16: submit failed\n"); goto out;
        }
        if (rocket_bo_prep(fd, &out_bo, 0, 2000000000ULL)) {
            ROCKET_LOGE("rocket_ew_mul_fp16: wait timeout\n"); goto out;
        }
        {
            _Float16 *ob = out_bo.ptr;
            for (int t = 0; t < telems; t++)
                out[base + t] = ob[feature_data(NN, M, 1, 8, (t % NN) + 1, (t / NN) + 1, 1)];
        }
        rocket_bo_fini(fd, &out_bo);
    }
    rc = 0;

out:
    rocket_bo_free(fd, &out_bo);
    rocket_bo_free(fd, &op_bo);
    rocket_bo_free(fd, &in_bo);
    rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &wt);
    return rc;
}

/* ============================================================================
 * SECTION — ELU/SELU and composite ops (clip, PReLU, div)
 * ==========================================================================*/

/* ELU/SELU: f(x)=lambda*(x>=0?x:alpha*(e^x-1)) on the SYMMETRIC shifted single-table (the Abs
 * trick — x=0 kink on the middle sample, whole domain on the positive index half, so NO LE/LO
 * sign-mux glitch and NO host repair). Shared body; the public wrappers fix (alpha,lambda). */
static int elu_run(int fd, double alpha, double lambda, const _Float16 *in, _Float16 *out, int n)
{
    if (fd < 0 || n <= 0) return -1;
    if (alpha <= 0.0) return -3;

    /* SYMMETRIC domain [-R,R] (R from ROCKET_ELU_R, default 8 — the negative branch is fully
     * captured since e^-8~3e-4). scale = 16384/(2R); x=0 -> index 8192 (the middle sample, the
     * kink), interior. |x|>R saturates at the table edge (the positive tail is linear). */
    double R = 8.0;
    { const char *e = getenv("ROCKET_ELU_R"); if (e) { double r = atof(e); if (r >= 1.0) R = r; } }
    const double x_lo = -R, x_hi = R;
    const double scale = 16384.0 / (x_hi - x_lo);

    /* Output span [out_lo, lambda*R]; out_lo = the negative saturation -lambda*alpha. S (pow2)
     * covers it. (out_lo<0 lifts g(0)=lambda*alpha/S>0, so the x=0 sample is never the q=0 entry —
     * only the extreme x=-R tail floors to q>=1, decoding to ~out_lo, correct.) */
    const double out_lo = -lambda * alpha;
    const double need = lambda * R - out_lo;            /* = lambda*(R+alpha) */
    int Sexp = 0; while ((double)(1u << Sexp) < need) Sexp++;
    const double S = (double)(1u << Sexp);

    static uint16_t lut[1026];
    build_lut_shifted_elu(alpha, lambda, x_lo, x_hi, out_lo, S, lut);

    lut_act_params_t p = {0};
    p.n                 = (n + 7) & ~7;
    p.lut               = lut;
    p.bn_mul_operand    = fp16_bits(scale);
    p.bn_alu_cfg        = fp32_bits(-x_lo * scale);     /* index = (x - x_lo)*scale */
    p.out_cvt_scale     = 1;
    p.out_cvt_cvt_type  = 1;
    p.out_cvt_minus_exp = (uint8_t)lround(15.0 - log2(S));
    p.out_cvt_offset    = (uint32_t)(int32_t)lround(out_lo * 32768.0 / S);
    p.lut_le_start      = 0xffffc000;
    p.lut_lo_end        = 0x00004000;
    p.le_index_select   = 5;
    p.lo_index_select   = 5;

    { const char *e;   /* OUT_CVT RE knobs (shared names with the other LUT paths) */
      if ((e = getenv("ROCKET_LUT_OFFSET"))) p.out_cvt_offset    = (uint32_t)strtol(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_MINEXP"))) p.out_cvt_minus_exp = (uint8_t)strtoul(e, NULL, 0);
      if ((e = getenv("ROCKET_LUT_BNMUL")))  p.bn_mul_operand    = (uint16_t)strtoul(e, NULL, 0); }

    int r = run_dpu_lut(fd, &p, in, out, n);
    if (r) return r;

    /* x≈0 REPAIR. A shifted single-table over a domain that STRADDLES 0 does NOT escape the
     * DPU LUT's x≈0 LE/LO sign-mux glitch (the mux decides on sign(x), independent of the
     * BN-ALU index shift — the LE/LO-mux QUIRK 2). The shifted positive-domain /
     * exp kinds avoid it only because their domain is ONE-SIDED. ELU/SELU straddle 0, so the
     * razor-thin x≈0 band spikes; patch it on the host with the exact value (the readback
     * streams every element anyway). ELU(x≈0)≈lambda*x, exact via elu_d. */
    if (!getenv("ROCKET_ELU_NOREPAIR")) {
        const float guard = (float)(R / 1024.0);
        for (int i = 0; i < n; i++) {
            float x = (float)in[i];
            if (x <= guard && x >= -guard) out[i] = (_Float16)elu_d(alpha, lambda, x);
        }
    }
    return 0;
}

int rocket_elu_fp16(int fd, float alpha, const _Float16 *in, _Float16 *out, int n)
{ return elu_run(fd, (double)alpha, 1.0, in, out, n); }

int rocket_selu_fp16(int fd, const _Float16 *in, _Float16 *out, int n)
{ return elu_run(fd, SELU_ALPHA, SELU_LAMBDA, in, out, n); }

int rocket_ew_mul_fp16(int fd, const _Float16 *a, const _Float16 *b, _Float16 *out, int n)
{ return ew_binary_fp16(fd, a, b, out, n, EW_OP_MUL); }

int rocket_ew_add_fp16(int fd, const _Float16 *a, const _Float16 *b, _Float16 *out, int n)
{ return ew_binary_fp16(fd, a, b, out, n, EW_OP_ADD); }

int rocket_ew_sub_fp16(int fd, const _Float16 *a, const _Float16 *b, _Float16 *out, int n)
{ return ew_binary_fp16(fd, a, b, out, n, EW_OP_SUB); }

/* Elementwise two-tensor MAX/MIN on the NPU: out[i] = max/min(a[i], b[i]). Same conv-main EW
 * datapath as add/mul, with the EW ALU algo set to MAX(0)/MIN(1) (DPU_EW_CFG bits[17:16]).
 * Covers TFLite/ONNX Maximum/Minimum (and ReLU as max(x,0)). 0 on success, <0 on error. */
int rocket_ew_max_fp16(int fd, const _Float16 *a, const _Float16 *b, _Float16 *out, int n)
{ return ew_binary_fp16(fd, a, b, out, n, EW_OP_MAX); }

int rocket_ew_min_fp16(int fd, const _Float16 *a, const _Float16 *b, _Float16 *out, int n)
{ return ew_binary_fp16(fd, a, b, out, n, EW_OP_MIN); }

/* Clip(x, lo, hi) = min(max(x, lo), hi), fully on the NPU via a constant-operand MAX then MIN
 * (two EW passes). Covers TFLite/ONNX Clip and the bounded ReLU family (ReLU6 = Clip(0,6)).
 * fd<0 = host reference. 0 on success, <0 on error. */
int rocket_clip_fp16(int fd, float lo, float hi, const _Float16 *in, _Float16 *out, int n)
{
    if (n <= 0) return -1;
    if (fd < 0) {
        for (int i = 0; i < n; i++) {
            float x = (float)in[i];
            out[i] = (_Float16)(x < lo ? lo : (x > hi ? hi : x));
        }
        return 0;
    }
    _Float16 *cst = malloc((size_t)n * sizeof(_Float16));
    if (!cst) return -2;
    for (int i = 0; i < n; i++) cst[i] = (_Float16)lo;
    int r = rocket_ew_max_fp16(fd, in, cst, out, n);          /* out = max(x, lo)     */
    if (!r) {
        for (int i = 0; i < n; i++) cst[i] = (_Float16)hi;
        r = rocket_ew_min_fp16(fd, out, cst, out, n);         /* out = min(out, hi)   */
    }
    free(cst);
    return r;
}

/* PReLU with a PER-CHANNEL negative slope alpha[C], input laid out [C][S]. Fully on the NPU
 * with NO LUT (so no x≈0 mux glitch). For the universal alpha in [0,1] case PReLU == max(x,
 * alpha_c*x) — a per-channel scale (ew_mul against the broadcast alpha) then ew_max — 2 NPU
 * passes, bit-exact. For any alpha outside [0,1] the identity max(x,alpha*x) breaks, so it
 * falls back to the general relu(x)+alpha_c*min(x,0) (4 passes), also exact (each step only
 * scales or adds/subtracts a zero). */
int rocket_prelu_fp16(int fd, int C, int S, const _Float16 *x, const float *alpha, _Float16 *out)
{
    if (C < 1 || S < 1) return -1;
    const size_t N = (size_t)C * S;
    if (fd < 0) { rocket_prelu_ref_fp16(C, S, x, alpha, out); return 0; }

    int all01 = 1;
    for (int c = 0; c < C; c++) if (alpha[c] < 0.f || alpha[c] > 1.f) { all01 = 0; break; }

    /* broadcast the per-channel slope across the spatial axis: ab[c][s] = (fp16)alpha[c] */
    _Float16 *ab = malloc(N * sizeof(_Float16));
    _Float16 *s  = malloc(N * sizeof(_Float16));
    if (!ab || !s) { free(ab); free(s); return -2; }
    for (int c = 0; c < C; c++) {
        _Float16 a = (_Float16)alpha[c];
        for (int j = 0; j < S; j++) ab[(size_t)c * S + j] = a;
    }

    int rc;
    if (all01) {
        rc = rocket_ew_mul_fp16(fd, x, ab, s, (int)N);             /* s = alpha_c * x      */
        if (!rc) rc = rocket_ew_max_fp16(fd, x, s, out, (int)N);   /* out = max(x, s)      */
    } else {
        _Float16 *z = calloc(N, sizeof(_Float16));                 /* zeros (for relu)     */
        if (!z) { free(ab); free(s); return -2; }
        rc = rocket_ew_max_fp16(fd, x, z, out, (int)N);            /* out = relu(x) = p    */
        if (!rc) rc = rocket_ew_sub_fp16(fd, x, out, s, (int)N);   /* s = x - p = min(x,0) */
        if (!rc) rc = rocket_ew_mul_fp16(fd, s, ab, s, (int)N);    /* s = alpha_c*min(x,0) */
        if (!rc) rc = rocket_ew_add_fp16(fd, out, s, out, (int)N); /* out = p + alpha*neg  */
        free(z);
    }
    free(ab); free(s);
    return rc;
}

/* out = a / b  fully on the NPU: reciprocal(b) on the DPU LUT, then a*recip on the EW unit.
 * `b` must be positive and within the reciprocal LUT domain (see header). fd<0 = host. */
int rocket_ew_div_fp16(int fd, const _Float16 *a, const _Float16 *b, _Float16 *out, int n)
{
    if (n <= 0) return -1;
    if (fd < 0) {                                  /* host reference path */
        for (int i = 0; i < n; i++)
            out[i] = (_Float16)((float)a[i] / (float)b[i]);
        return 0;
    }
    _Float16 *recip = malloc((size_t)n * sizeof(_Float16));
    if (!recip) return -2;
    int r = rocket_activation_fp16(fd, ROCKET_ACTIVATION_RECIPROCAL, b, recip, n);
    if (!r) r = rocket_ew_mul_fp16(fd, a, recip, out, n);
    free(recip);
    return r;
}
