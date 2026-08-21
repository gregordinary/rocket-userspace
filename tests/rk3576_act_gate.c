// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_act_gate.c — rocket_act_int8_rk3576(), against a CPU model of the part.
 *
 * The op is a nonlinear activation run entirely on the NPU: a DPU-only program bursts
 * two 513-entry tables into the LUT RAM and an ordinary depthwise convolution reads
 * them out of its EW stage, both as tasks of ONE job.
 *
 * WHAT IS ASSERTED IS THE CHIP'S OWN ARITHMETIC, bit for bit. The model here is not
 * f(x): it is the table the library built, indexed the way the hardware indexes it, and
 * requantized by the OUT_CVT pair the emitter programs. Asserting against f() instead
 * would fold three separate questions — did the table load, is the index right, is the
 * requant right — into one number that a table-building bug can satisfy.
 *
 * THE DISTANCE TO f() IS REPORTED SEPARATELY, per case, because that is the accuracy
 * question and it has a different answer: the op lowers onto an identity whose weight
 * is the index step, so every one of the 256 possible int8 inputs lands on its OWN
 * table entry and the hardware's interpolation never runs. What is left is the entry's
 * quantization and the requant's double rounding.
 *
 * A DENSE INPUT is what makes both meaningful: every case walks all 256 byte values, so
 * "the table is indexed by the value" is checked at every value it can take rather than
 * at whatever a random tensor happened to contain.
 *
 * Modes:
 *   gate    every kind x every quantization x every shape, then the fused pass (default)
 *   one     one case, printed as a curve — the instrument
 *   fused   only the conv->activation epilogue
 *
 * Exit: 0, 1 on a failure, 2 no NPU or not this chip (skip).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_activation.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_hw_profile.h"
#include "requant_model.h"

/* Held in step with rocket_conv2d_rk3576.c's R76_ACT_SEL / R76_ACT_CONV_SCALE. A gate
 * that re-derived them from the surface would be fitting the answer; these are the two
 * constants the entry point chose, restated so a change to either breaks here. */
#define ACT_SEL         5u
#define ACT_CONV_SCALE  (1.0f / 128.0f)

struct act_quant { float in_scale; int in_zp; float out_scale; int out_zp; };
struct act_shape { unsigned c, h, w; };

/* Quantizations chosen so each exercises something different: a symmetric pair, an
 * asymmetric input, an asymmetric output, a wide input range that pushes a saturating
 * activation into both tails, and a narrow one that puts every sample in the knee. */
static const struct act_quant QUANTS[] = {
    { 0.0625f,    0, 0.0078125f,    0 },   /* +-8 in, [0,2) out, both symmetric  */
    { 0.0625f,  -37, 0.0078125f,  -128 },  /* an asymmetric input and a uint8 out */
    { 0.03125f,  64, 0.015625f,     19 },  /* both zero points off zero           */
    { 0.25f,      0, 0.03125f,     -64 },  /* +-32 in: deep into both tails       */
    { 0.0078125f, 5, 0.00390625f,   11 },  /* +-1 in: every sample in the knee    */
};

/* Shapes: one channel group and several, a plane that is not a multiple of four (the
 * depthwise surface stride rounds), a tall plane that the row planner splits, and a
 * single row. */
static const struct act_shape SHAPES[] = {
    {  16, 16, 16 },
    {  32,  8, 32 },
    {   8,  5,  7 },
    {  64, 32, 32 },
    { 128,  1, 64 },
    {  16, 64, 64 },
};

static const int KINDS[] = {
    ROCKET_RK3576_ACT_SIGMOID, ROCKET_RK3576_ACT_TANH,
    ROCKET_RK3576_ACT_SWISH,   ROCKET_RK3576_ACT_HARDSWISH,
    ROCKET_RK3576_ACT_HARDSIGMOID, ROCKET_RK3576_ACT_ELU,
};

/* The part's own arithmetic for one input byte.
 *
 * The datapath value is `2^sel * (q - in_zp)`, so the index is `q - in_zp + 512` in the
 * low table and `q - in_zp` in the high one — an EXACT grid point either way, which is
 * the whole reason the entry point scales the identity. Outside the tables the two
 * clamps apply, and the domain is half-open at the top.
 */
static int act_model(const int16_t *le, const int16_t *lo, const lut_rk3576_t *w,
                     int q, int in_zp, unsigned mul, unsigned shift, int out_zp)
{
    long v = (long)(1 << ACT_SEL) * (long)(q - in_zp);
    long e;

    if (v < w->le_start)      e = w->clamp_lo;
    else if (v >= w->lo_end)  e = w->clamp_hi;
    else if (v < 0)           e = le[(v - w->le_start) >> ACT_SEL];
    else                      e = lo[v >> ACT_SEL];
    return requant_apply_zp(e, mul, shift, out_zp);
}

static double act_ref(int kind, double x)
{
    double s;
    switch (kind) {
    case ROCKET_RK3576_ACT_SIGMOID:
        return x >= 0.0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
    case ROCKET_RK3576_ACT_TANH:        return tanh(x);
    case ROCKET_RK3576_ACT_SWISH:
        s = x >= 0.0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
        return x * s;
    case ROCKET_RK3576_ACT_HARDSWISH:
        s = x + 3.0; s = s < 0.0 ? 0.0 : (s > 6.0 ? 6.0 : s);
        return x * s / 6.0;
    case ROCKET_RK3576_ACT_HARDSIGMOID:
        s = x + 3.0; s = s < 0.0 ? 0.0 : (s > 6.0 ? 6.0 : s);
        return s / 6.0;
    case ROCKET_RK3576_ACT_ELU:         return x >= 0.0 ? x : expm1(x);
    default:                            return 0.0;
    }
}

static int sat8(long v) { return v > 127 ? 127 : (v < -128 ? -128 : (int)v); }

/* One case. Returns 0 on a bit-exact surface, 1 otherwise; `*ref_worst` receives the
 * largest distance to an exact f(), which is reported and not asserted. */
static int run_case(int fd, int kind, const struct act_quant *q,
                    const struct act_shape *sh, int verbose, int *ref_worst)
{
    size_t n = (size_t)sh->c * sh->h * sh->w;
    int8_t *in = malloc(n), *out = malloc(n);
    int16_t le[RK3576_LUT_ENTRIES], lo[RK3576_LUT_ENTRIES];
    lut_rk3576_t w;
    unsigned mul, shift;
    size_t i;
    int bad = 0, worst = 0, rworst = 0, shown = 0, rc;

    if (!in || !out) { free(in); free(out); return 1; }

    /* Dense: every byte value appears, and appears in every channel, so a per-channel
     * fault and a per-value one are both visible. */
    for (i = 0; i < n; i++) in[i] = (int8_t)(int)((i % 256u) - 128u);
    memset(out, 0x5A, n);

    /* The same two scales the entry point derives, restated rather than borrowed: this
     * is the model, and a model that called the library's own builder with the library's
     * own arguments would agree with it by construction even if both were wrong. */
    requant_params(ACT_CONV_SCALE, &mul, &shift);
    if (rocket_rk3576_lut_build(kind,
                                (double)q->in_scale / (double)(1u << ACT_SEL),
                                (double)q->out_scale * ((double)mul / ldexp(1.0, (int)shift)),
                                ACT_SEL, le, lo, &w) != 0) {
        printf("      the table builder refused\n");
        free(in); free(out); return 1;
    }

    rc = rocket_act_int8_rk3576(fd, kind, sh->c, sh->h, sh->w,
                                in, q->in_scale, q->in_zp,
                                out, q->out_scale, q->out_zp);
    if (rc != ROCKET_OK) {
        printf("      the entry returned %d\n", rc);
        free(in); free(out); return 1;
    }

    for (i = 0; i < n; i++) {
        int want = act_model(le, lo, &w, in[i], q->in_zp, mul, shift, q->out_zp);
        int have = out[i];
        double x = (double)q->in_scale * (double)(in[i] - q->in_zp);
        double y = act_ref(kind, x) / (double)q->out_scale + (double)q->out_zp;
        int exact = sat8((long)(y < 0.0 ? ceil(y - 0.5) : floor(y + 0.5)));
        int d = have - want, dr = have - exact;

        if (d < 0) d = -d;
        if (dr < 0) dr = -dr;
        if (d > worst) worst = d;
        if (dr > rworst) rworst = dr;
        if (d && ++bad && shown < 5) {
            printf("      q=%-5d v=%-7ld want %4d got %4d\n",
                   in[i], (long)(1 << ACT_SEL) * (long)(in[i] - q->in_zp), want, have);
            shown++;
        }
    }
    if (verbose) {
        int k;
        printf("      curve:");
        for (k = -128; k < 128; k += 24) printf(" %d->%d", k, out[(size_t)(k + 128)]);
        printf("\n");
    }
    *ref_worst = rworst;
    if (bad)
        printf("      %d of %zu elements disagree with the chip model, worst %d\n",
               bad, n, worst);
    free(in); free(out);
    return bad ? 1 : 0;
}

/* ============================================================================
 * SECTION — the activation FUSED into a real convolution's epilogue
 *
 * The standalone op above is asserted BIT-EXACT against a model of the chip, because it
 * arranges for each of its 256 possible inputs to land on its own table entry. A fused
 * activation cannot make that arrangement: the value the LUT sees is the convolution's
 * ACCUMULATOR, which lands between entries, so the hardware's linear interpolation runs
 * and its exact rounding is not decoded. What this pass asserts is therefore the two
 * things that ARE decidable, and reports the third:
 *
 *   IT RUNS AND IT IS THE ACTIVATION. The same convolution without the LUT must produce
 *   a DIFFERENT surface — otherwise a bypassed EW stage would pass every accuracy check
 *   that a gentle activation happens to sit close to.
 *   IT IS WITHIN A FEW COUNTS OF AN EXACT f(). Interpolation over a 512-interval half
 *   costs a count or two; anything past FUSED_DEFECT is a wrong table or a wrong window,
 *   not the interpolation.
 *   THE DISTANCE ITSELF IS REPORTED, per case, because that is the quantity a caller
 *   choosing between the fused and the two-op form needs.
 * ==========================================================================*/
#define FUSED_DEFECT 4

struct fused_case { int ic, oc, ih, iw, k, dw; float in_s, w_s, out_s; int izp, ozp; };

static const struct fused_case FUSED[] = {
    /* in_scale*w_scale is the value unit: it has to be coarse enough for the table's
     * peak to fit an int16 and fine enough that the accumulator stays in the window. */
    {  32,  32, 8,  8, 1, 0, 1.0f/64, 1.0f/64, 1.0f/256, 0,  -128 },
    {  32,  32, 8,  8, 1, 0, 1.0f/64, 1.0f/64, 1.0f/128, 7,   -64 },
    {  64,  64, 6,  6, 3, 0, 1.0f/64, 1.0f/96, 1.0f/256, -5, -128 },
    {  32,  32, 8,  8, 3, 1, 1.0f/32, 1.0f/64, 1.0f/256, 0,  -128 },
    {  64,  64, 4,  4, 1, 1, 1.0f/32, 1.0f/64, 1.0f/128, 11,  -64 },
};

static int fused_case(int fd, int kind, const struct fused_case *c, int *worst_out)
{
    unsigned ow, oh;
    rocket_conv2d_desc d;
    size_t nin = (size_t)c->ic * c->ih * c->iw, nout, i;
    int8_t *in = NULL, *W = NULL, *o_act = NULL, *o_plain = NULL;
    int32_t *bias = NULL;
    size_t wn = c->dw ? (size_t)c->oc * c->k * c->k
                      : (size_t)c->oc * c->ic * c->k * c->k;
    int rc, worst = 0, differ = 0, ref_differ = 0, ok = 0;
    double lo_x = 1e30, hi_x = -1e30;
    unsigned seed = 7u;

    memset(&d, 0, sizeof d);
    d.ic = c->ic; d.oc = c->oc; d.ih = c->ih; d.iw = c->iw;
    d.kh = c->k; d.kw = c->k; d.stride_y = 1; d.stride_x = 1;
    d.pad_top = c->k / 2; d.pad_left = c->k / 2;
    d.dil_y = d.dil_x = 1; d.depthwise = c->dw;
    ow = (unsigned)rocket_conv2d_ow(&d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    nout = (size_t)c->oc * ow * oh;

    in = malloc(nin); W = malloc(wn); bias = calloc(c->oc, sizeof *bias);
    o_act = malloc(nout); o_plain = malloc(nout);
    if (!in || !W || !bias || !o_act || !o_plain) goto done;

    /* Small magnitudes deliberately: the point is to land the accumulator INSIDE the
     * activation's curved region, where interpolation is doing something. A saturating
     * tail would pass on the clamp alone. */
    for (i = 0; i < nin; i++) {
        seed = seed * 1103515245u + 12345u;
        in[i] = (int8_t)((int)((seed >> 16) % 97u) - 48);
    }
    for (i = 0; i < wn; i++) {
        seed = seed * 1103515245u + 12345u;
        W[i] = (int8_t)((int)((seed >> 16) % 9u) - 4);
    }

    rc = rocket_conv2d_int8_act_rk3576(fd, &d, in, W, bias, c->in_s, c->w_s, c->out_s,
                                       c->izp, 0, c->ozp, kind, o_act);
    if (rc == ROCKET_E_UNSUPPORTED) {
        /* The entry's own bounds, and it names which quantity is out of range. Both are
         * real: an unbounded activation over a fine value unit needs more than an int16
         * entry, and a domain wider than sel 15 reaches has no window. A refusal is not a
         * failure — what WOULD be one is a kind that refuses everywhere, which the
         * per-kind tally below catches. */
        printf("  %-12s refused ic=%-4d oc=%-4d %dx%d k%d %-9s (see the entry's reason)\n",
               rocket_rk3576_act_name(kind), c->ic, c->oc, c->ih, c->iw, c->k,
               c->dw ? "depthwise" : "direct");
        free(in); free(W); free(bias); free(o_act); free(o_plain);
        return 2;
    }
    if (rc != ROCKET_OK) { printf("      the fused entry returned %d\n", rc); goto done; }
    rc = c->dw
       ? rocket_conv2d_dw_int8_rk3576(fd, &d, in, W, bias, c->in_s, c->w_s, c->out_s,
                                      c->izp, 0, c->ozp, o_plain)
       : rocket_conv2d_int8_rk3576(fd, &d, in, W, bias, c->in_s, c->w_s, c->out_s,
                                   c->izp, 0, c->ozp, o_plain);
    if (rc != ROCKET_OK) { printf("      the plain control returned %d\n", rc); goto done; }

    /* MORE THAN A ROUNDING, on both sides. The two surfaces disagreeing by one count is
     * the requant's tie-to-even against a reference that rounds half away from zero, and
     * counting that as "the activation did something" makes the liveness check fire on a
     * kind that IS the identity here. */
    for (i = 0; i < nout; i++)
        if (o_act[i] - o_plain[i] > 1 || o_plain[i] - o_act[i] > 1) differ++;

    /* The reference: an exact f() over the accumulator, in the caller's own output
     * quantization. The accumulator is recomputed here rather than read from the plain
     * control, whose own output has already been requantized to int8. */
    {
        int oc_, y, x, ky, kx, ic_;
        for (oc_ = 0; oc_ < c->oc; oc_++)
            for (y = 0; y < (int)oh; y++)
                for (x = 0; x < (int)ow; x++) {
                    long acc = 0;
                    double xr, yr;
                    int have, want, dd;
                    for (ky = 0; ky < c->k; ky++)
                        for (kx = 0; kx < c->k; kx++) {
                            int sy = y + ky - d.pad_top, sx = x + kx - d.pad_left;
                            if (sy < 0 || sx < 0 || sy >= c->ih || sx >= c->iw) continue;
                            if (c->dw) {
                                acc += (long)(in[((size_t)oc_ * c->ih + sy) * c->iw + sx]
                                              - c->izp)
                                     * W[((size_t)oc_ * c->k + ky) * c->k + kx];
                            } else {
                                for (ic_ = 0; ic_ < c->ic; ic_++)
                                    acc += (long)(in[((size_t)ic_ * c->ih + sy) * c->iw
                                                     + sx] - c->izp)
                                         * W[(((size_t)oc_ * c->ic + ic_) * c->k + ky)
                                             * c->k + kx];
                            }
                        }
                    xr = (double)acc * (double)c->in_s * (double)c->w_s;
                    if (xr < lo_x) lo_x = xr;
                    if (xr > hi_x) hi_x = xr;
                    yr = act_ref(kind, xr) / (double)c->out_s + (double)c->ozp;
                    want = sat8((long)(yr < 0.0 ? ceil(yr - 0.5) : floor(yr + 0.5)));
                    have = o_act[((size_t)oc_ * oh + y) * ow + x];
                    dd = have - want;
                    if (dd < 0) dd = -dd;
                    if (dd > worst) worst = dd;
                    /* Whether the ACTIVATION is distinguishable from the identity at
                     * this quantization at all. ELU and swish are both x to within a
                     * rounding over a small pre-activation range, so "the fused surface
                     * differs from the plain one" cannot detect a bypassed LUT there —
                     * and asserting it anyway reads as a hardware fault. ELU under an
                     * output quantization with no negative range IS the identity clamped
                     * at zero, which the DPU's int8 saturation already computes. */
                    {
                        int dp = want - o_plain[((size_t)oc_ * oh + y) * ow + x];
                        if (dp > 1 || dp < -1) ref_differ++;
                    }
                }
    }
    ok = worst <= FUSED_DEFECT && (differ > 0 || ref_differ == 0);
    printf("  %-12s %s ic=%-4d oc=%-4d %dx%d k%d %-9s pre-act [%+.2f,%+.2f]  "
           "worst %d count%s vs an exact f()%s\n",
           rocket_rk3576_act_name(kind), ok ? "ok  " : "FAIL",
           c->ic, c->oc, c->ih, c->iw, c->k, c->dw ? "depthwise" : "direct",
           lo_x, hi_x, worst, worst == 1 ? "" : "s",
           ref_differ == 0 ? "  [indistinguishable from the identity here, so the "
                             "differs-from-plain check does not apply]"
                           : (differ ? "" : "  — AND IT MATCHES THE PLAIN CONV, so the "
                                            "LUT is bypassed"));
    *worst_out = worst;
done:
    free(in); free(W); free(bias); free(o_act); free(o_plain);
    return ok ? 0 : 1;
}

static int pass_fused(int fd)
{
    unsigned ci, ki;
    int failed = 0, passed = 0, refused = 0, worst_all = 0;

    printf("\n== FUSED: the activation in a real convolution's epilogue ==\n");
    printf("   ASSERTED: it runs, it differs from the same conv without the LUT, and it "
           "is within %d counts of an exact f(). The distance is REPORTED — a fused LUT "
           "reads an accumulator, which lands between table entries, so the hardware "
           "interpolates and bit-exactness is not on offer.\n", FUSED_DEFECT);
    for (ki = 0; ki < sizeof KINDS / sizeof KINDS[0]; ki++) {
        int ran = 0;
        for (ci = 0; ci < sizeof FUSED / sizeof FUSED[0]; ci++) {
            int w = 0, r = fused_case(fd, KINDS[ki], &FUSED[ci], &w);
            if (r == 2)      refused++;
            else if (r == 0) { passed++; ran++; }
            else             failed++;
            if (w > worst_all) worst_all = w;
        }
        if (!ran) {
            printf("     %s ran on NO case — a kind that refuses everywhere is a "
                   "defect, not a bound\n", rocket_rk3576_act_name(KINDS[ki]));
            failed++;
        }
    }
    printf("== %d passed, %d refused as the entry's bounds require, %d failed; the "
           "largest distance to an exact f() anywhere is %d count%s ==\n",
           passed, refused, failed, worst_all, worst_all == 1 ? "" : "s");
    return failed;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *mode = argc > 1 ? argv[1] : "gate";
    unsigned k, qi, si;
    int fd, passed = 0, failed = 0, ref_worst_all = 0;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_act_gate: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_act_gate: no NPU device — skipping\n"); return 2; }

    if (!strcmp(mode, "one")) {
        int rw = 0;
        int rc = run_case(fd, ROCKET_RK3576_ACT_SIGMOID, &QUANTS[0], &SHAPES[0], 1, &rw);
        printf("one: %s, worst distance to an exact sigmoid %d count%s\n",
               rc ? "FAIL" : "bit-exact against the chip model", rw, rw == 1 ? "" : "s");
        rocket_close(fd);
        return rc;
    }
    if (!strcmp(mode, "fused")) {
        int rc = pass_fused(fd);
        rocket_close(fd);
        return rc ? 1 : 0;
    }

    printf("rk3576_act_gate: the DPU LUT activation, %u kinds x %u quantizations x "
           "%u shapes\n", (unsigned)(sizeof KINDS / sizeof *KINDS),
           (unsigned)(sizeof QUANTS / sizeof *QUANTS),
           (unsigned)(sizeof SHAPES / sizeof *SHAPES));
    printf("  asserted: the chip's own arithmetic, bit for bit. reported: the distance "
           "to an exact f().\n");

    for (k = 0; k < sizeof KINDS / sizeof *KINDS; k++) {
        int worst_kind = 0, fails = 0;
        for (qi = 0; qi < sizeof QUANTS / sizeof *QUANTS; qi++)
            for (si = 0; si < sizeof SHAPES / sizeof *SHAPES; si++) {
                int rw = 0, rc;
                rc = run_case(fd, KINDS[k], &QUANTS[qi], &SHAPES[si], 0, &rw);
                if (rc) {
                    printf("    FAIL %s  in(%g,%d) out(%g,%d)  %ux%ux%u\n",
                           rocket_rk3576_act_name(KINDS[k]),
                           (double)QUANTS[qi].in_scale, QUANTS[qi].in_zp,
                           (double)QUANTS[qi].out_scale, QUANTS[qi].out_zp,
                           SHAPES[si].c, SHAPES[si].h, SHAPES[si].w);
                    failed++; fails++;
                } else {
                    passed++;
                }
                if (rw > worst_kind) worst_kind = rw;
            }
        if (worst_kind > ref_worst_all) ref_worst_all = worst_kind;
        printf("  %-12s %s   worst distance to an exact f(): %d count%s\n",
               rocket_rk3576_act_name(KINDS[k]), fails ? "FAIL" : "ok",
               worst_kind, worst_kind == 1 ? "" : "s");
    }

    printf("== %d passed, %d failed; the largest distance to an exact f() anywhere is "
           "%d count%s ==\n", passed, failed, ref_worst_all,
           ref_worst_all == 1 ? "" : "s");
    failed += pass_fused(fd);
    rocket_close(fd);
    return failed ? 1 : 0;
}
