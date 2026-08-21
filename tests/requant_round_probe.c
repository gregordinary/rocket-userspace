// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * requant_round_probe.c — which way does the DPU's OUT_CVT round a TIE?
 *
 * The output convertor is `out = sat8( round(acc * MUL >> SHIFT) + offset )`, and every
 * CPU reference model in this tree spells that round as `(acc*MUL + half) >> SHIFT` —
 * an arithmetic shift, so ties go toward +inf (round half UP). The IP this datapath
 * descends from documents the other rule: "the rounding method used after the shift is
 * to round half AWAY FROM ZERO" [nvdla.org/hw/v1/ias/precision.html]. The two differ by
 * one count, and only on NEGATIVE ties — which is exactly the size and the sparsity of
 * the model disagreement that has been standing noise under every int8 measurement here.
 *
 * NO GATE CAN SEE IT, which is why it survived. The scale a caller passes is turned into
 * MUL by the vendor's (QNNPACK) derivation, `MUL = ((bits>>9) & 0x7fff) + 1` with bit 14
 * forced — and that trailing +1 makes MUL ODD for every round scale, including every
 * power of two. An odd multiplier moves an exact half off the tie by MUL/2^SHIFT, always
 * outward, so a tie never reaches the rounder and every rule agrees. The gates pick
 * powers of two, so they have never once exercised the case.
 *
 * The probe therefore chooses the scale so that MUL lands on exactly 2^14: it needs
 * `(bits>>9) & 0x7fff == 0x3fff`, i.e. a float whose top 14 mantissa bits are all ones
 * under an even exponent field. Then the requant is a plain arithmetic `acc >> e` and
 * every odd multiple of 2^(e-1) is a true tie.
 *
 * The operands are zero and the whole accumulator comes from the per-channel BIAS, which
 * the DPU adds in the BS ALU ahead of the convertor. That makes each output element a
 * chosen integer rather than a sum, so a disagreement names its own input.
 *
 * Run with `sudo -E`. Exits 2 (skip) where there is no int8 requant entry to drive.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"

#define M_ROWS 4
#define K_DEP  32
#define N_CH   64

/* The emitter's scale -> (MUL, SHIFT), copied from the encoders so the probe reports the
 * multiplier it is actually going to get rather than the one it assumes. */
static void derive(float conv_scale, unsigned *mul, unsigned *shift)
{
    union { float f; uint32_t u; } cv;
    uint32_t bits;
    unsigned s, m;
    cv.f = conv_scale;
    bits = cv.u;
    s = 127u + 31u - 32u - (bits >> 23) + 16u;
    m = ((bits >> 9) & 0x7FFFu) + 1u;
    if (m < (1u << 14)) m |= (1u << 14);
    *mul = m;
    *shift = s - 1u;
}

static int sat8(int64_t v)
{
    return (int)(v < -128 ? -128 : (v > 127 ? 127 : v));
}

/* The four candidate rules, all on `acc * mul >> shift`. */
static int r_half_up(int64_t acc, unsigned mul, unsigned shift)
{
    int64_t p = acc * (int64_t)mul;
    int64_t half = shift ? ((int64_t)1 << (shift - 1)) : 0;
    return sat8((p + half) >> shift);
}
static int r_half_away(int64_t acc, unsigned mul, unsigned shift)
{
    int64_t p = acc * (int64_t)mul;
    int64_t half = shift ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t mag = p < 0 ? -p : p;
    int64_t q = (mag + half) >> shift;
    return sat8(p < 0 ? -q : q);
}
static int r_half_even(int64_t acc, unsigned mul, unsigned shift)
{
    int64_t p = acc * (int64_t)mul;
    int64_t half = shift ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q = (p + half) >> shift;
    if (shift && (p & (((int64_t)1 << shift) - 1)) == half && (q & 1)) q -= 1;
    return sat8(q);
}
static int r_trunc(int64_t acc, unsigned mul, unsigned shift)
{
    int64_t p = acc * (int64_t)mul;
    int64_t mag = p < 0 ? -p : p;
    int64_t q = mag >> shift;
    return sat8(p < 0 ? -q : q);
}

typedef struct { const char *name; int (*f)(int64_t, unsigned, unsigned); } rule;
static const rule RULES[] = {
    { "half-up (our model)", r_half_up },
    { "half-away-from-0",    r_half_away },
    { "half-to-even",        r_half_even },
    { "truncate-toward-0",   r_trunc },
};
#define N_RULES ((int)(sizeof RULES / sizeof RULES[0]))

/* Each scale is chosen for MUL == 0x4000 exactly: significand 1.99993896484375 under an
 * even exponent field, so the top 14 mantissa bits are all ones and the +1 carries into
 * bit 14 rather than making the multiplier odd. */
typedef struct { const char *name; float scale; } probe_scale;
static const probe_scale SCALES[] = {
    { "acc>>1", 0.999969482421875f },      /* 0x3f7ffe00 */
    { "acc>>3", 0.24999237060546875f },    /* 0x3e7ffe00 */
};
#define N_SCALES ((int)(sizeof SCALES / sizeof SCALES[0]))

static int run_scale(int fd, const probe_scale *ps, int verbose)
{
    int8_t *A, *B, *C;
    int32_t *bias;
    unsigned mul, shift;
    int n, m, rc, live[N_RULES], ties = 0, r, survivors = 0;

    derive(ps->scale, &mul, &shift);
    printf("  scale %-22.18g MUL=0x%04x SHIFT=%u", (double)ps->scale, mul, shift);
    if (mul != (1u << 14)) {
        printf("  -- MUL is not 2^14, this scale cannot make a tie\n");
        return 1;
    }
    printf("\n");

    A    = calloc((size_t)M_ROWS * K_DEP, 1);
    B    = calloc((size_t)N_CH * K_DEP, 1);
    C    = calloc((size_t)M_ROWS * N_CH, 1);
    bias = calloc(N_CH, sizeof *bias);
    if (!A || !B || !C || !bias) { free(A); free(B); free(C); free(bias); return 1; }

    /* Operands zero: the accumulator IS the bias. Span both signs, and both parities of
     * the tie so a rule that only differs on ties has non-tie elements to agree on. */
    for (n = 0; n < N_CH; n++) bias[n] = n - N_CH / 2;

    rc = rocket_matmul_int8_rk3576(fd, M_ROWS, K_DEP, N_CH, A, B, bias, ps->scale, C);
    if (rc != 0) {
        printf("    rocket_matmul_int8_rk3576 returned %d\n", rc);
        free(A); free(B); free(C); free(bias);
        return 1;
    }

    for (r = 0; r < N_RULES; r++) live[r] = 1;
    for (n = 0; n < N_CH; n++) {
        int64_t acc = bias[n];
        int64_t p = acc * (int64_t)mul;
        int tie = shift && (p & (((int64_t)1 << shift) - 1)) == ((int64_t)1 << (shift - 1));
        if (tie) ties++;
        for (m = 0; m < M_ROWS; m++) {
            int got = C[(size_t)m * N_CH + n];
            for (r = 0; r < N_RULES; r++)
                if (live[r] && RULES[r].f(acc, mul, shift) != got) live[r] = 0;
        }
        if (verbose && tie)
            printf("    acc %+5d %s -> got %+4d   up %+4d away %+4d even %+4d trunc %+4d\n",
                   (int)acc, "TIE", C[n],
                   r_half_up(acc, mul, shift), r_half_away(acc, mul, shift),
                   r_half_even(acc, mul, shift), r_trunc(acc, mul, shift));
    }

    printf("    %d of %d accumulators are exact ties.  consistent rules:", ties, N_CH);
    for (r = 0; r < N_RULES; r++)
        if (live[r]) { printf("  %s", RULES[r].name); survivors++; }
    if (!survivors) printf("  NONE");
    printf("\n");

    free(A); free(B); free(C); free(bias);
    /* A probe, not a gate: it fails only when the part disagrees with every rule, which
     * would mean the convertor is not doing `acc*MUL>>SHIFT` at all. */
    return survivors ? 0 : 1;
}

/* ---- the RK3588 arm: NOT YET DRIVING THE ACCUMULATOR -------------------------
 *
 * UNFINISHED, and it says so at runtime rather than reporting a rule it has not earned.
 * The tie rule on the RK3588 is therefore UNMEASURED; do not read this arm's output as
 * a negative about the rounder.
 *
 * The RK3588's public int8 matmul writes a RAW int32 accumulator and requants on the
 * host, so it cannot be asked at all. The only entry on that part with the on-chip
 * requant is the DEPTHWISE int8 conv — the Teflon-cracked int8-OUT path, per-tensor
 * scales and an int32 bias — and every parameter set tried here (accumulator carried by
 * the bias or by the input, 1x1 and 3x3 kernels, in_zp at 0x80 to null the driver's
 * correction term) comes back an ALL-ZERO surface. That is a quant-convention problem in
 * this probe, not a statement about the part: the entry is HW-validated bit-exact against
 * Mesa/Teflon elsewhere in this tree, so what is wrong is what this probe hands it.
 *
 * Whoever finishes it: start from tests/replay_dw_mesa.c, which drives the same path from
 * a capture that is known to compute, and change one parameter at a time from there.
 *
 * The self-check is the load-bearing part and stays whatever the plumbing does: the
 * NON-TIE accumulators must match the model — every rounding rule agrees on those — and
 * the ties are only read out if they do.
 */
#define DW_C  64
#define DW_HW 4

static int run_scale_dw(int fd, const probe_scale *ps, int verbose)
{
    rocket_conv2d_desc d;
    int8_t *in, *w, *out;
    int32_t *bias;
    unsigned mul, shift;
    int c, i, rc, live[N_RULES], ties = 0, r, survivors = 0, setup_ok = 1, bad = 0, neg = 0;

    derive(ps->scale, &mul, &shift);
    printf("  scale %-22.18g MUL=0x%04x SHIFT=%u", (double)ps->scale, mul, shift);
    if (mul != (1u << 14)) { printf("  -- MUL is not 2^14\n"); return 1; }
    printf("\n");

    memset(&d, 0, sizeof d);
    d.ic = DW_C; d.oc = DW_C; d.ih = DW_HW; d.iw = DW_HW;
    d.kh = 3; d.kw = 3; d.stride_y = 1; d.stride_x = 1; d.pad_top = 1; d.pad_left = 1;
    d.dil_y = 1; d.dil_x = 1; d.depthwise = 1;

    in   = calloc((size_t)DW_C * DW_HW * DW_HW, 1);
    w    = calloc((size_t)DW_C * 9, 1);
    out  = calloc((size_t)DW_C * DW_HW * DW_HW, 1);
    bias = calloc(DW_C, sizeof *bias);
    if (!in || !w || !out || !bias) { free(in); free(w); free(out); free(bias); return 1; }

    /* The accumulator is carried by the INPUT, not the bias: a 1x1 depthwise with a unit
     * weight makes each output element its own input element, and that keeps the whole
     * coefficient-buffer path (bias layout, zero-point folding into it) out of a question
     * that is only about the rounder. in_zp = 0x80 is what makes the driver's correction
     * term vanish — it carries a factor (in_zp - 0x80). */
    /* A 3x3 depthwise with ONE live tap at the centre: the path is cracked from a 3x3
     * Teflon capture and a 1x1 kernel writes nothing through it. */
    for (c = 0; c < DW_C; c++) w[(size_t)c * 9 + 4] = 1;
    for (i = 0; i < DW_C * DW_HW * DW_HW; i++)
        in[i] = (int8_t)((i % 64) - 32);

    /* The captured int8-out depthwise writer CLAMPS NEGATIVES TO ZERO — a ReLU in the
     * BS/EW stage of the Teflon program this path replays. So only the non-negative half
     * of the sweep carries information, which is still enough: a positive tie separates
     * half-up and half-away-from-zero (1 -> 1) from half-to-even and truncate (1 -> 0). */
    for (r = 0; r < N_RULES; r++) live[r] = 1;
    for (i = 0; i < DW_C * DW_HW * DW_HW; i++) {
        int64_t acc = in[i];
        if (acc < 0) { neg += (out[i] != 0); continue; }
        int64_t p = acc * (int64_t)mul;
        int tie = shift && (p & (((int64_t)1 << shift) - 1)) == ((int64_t)1 << (shift - 1));
        int got = out[i];
        if (tie) ties++;
        else if (got != r_half_up(acc, mul, shift)) {
            if (bad < 6)
                printf("    setup: acc %+4d -> want %+4d got %+4d\n",
                       (int)acc, r_half_up(acc, mul, shift), got);
            bad++;
            setup_ok = 0;
        }
        for (r = 0; r < N_RULES; r++)
            if (live[r] && RULES[r].f(acc, mul, shift) != got) live[r] = 0;
        if (verbose && tie && i < 64)
            printf("    acc %+5d TIE -> got %+4d   up %+4d away %+4d even %+4d trunc %+4d\n",
                   (int)acc, got, r_half_up(acc, mul, shift), r_half_away(acc, mul, shift),
                   r_half_even(acc, mul, shift), r_trunc(acc, mul, shift));
    }
    if (!setup_ok) {
        printf("    the NON-TIE channels do not match the model — the accumulator is not "
               "the bias here, so the ties say nothing about rounding\n");
        free(in); free(w); free(out); free(bias);
        return 1;
    }

    printf("    negatives clamped to zero: %s.  %d non-negative ties.  consistent rules:",
           neg ? "NO (some survived)" : "yes", ties);
    for (r = 0; r < N_RULES; r++)
        if (live[r]) { printf("  %s", RULES[r].name); survivors++; }
    if (!survivors) printf("  NONE");
    printf("\n");

    free(in); free(w); free(out); free(bias);
    return survivors ? 0 : 1;
}

int main(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, i, fails = 0, is76;
    int verbose = getenv("ROCKET_RR_VERBOSE") != NULL;

    is76 = strcmp(hw->name, "rk3576") == 0;
    if (!is76 && strcmp(hw->name, "rk3588") != 0) {
        printf("requant_round_probe: profile is %s — no int8-requant entry to drive "
               "here; skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("requant_round_probe: no NPU device — skipping\n"); return 2; }

    printf("== DPU OUT_CVT tie rounding (%s) ==\n", hw->name);
    for (i = 0; i < N_SCALES; i++)
        fails += is76 ? run_scale(fd, &SCALES[i], verbose)
                      : run_scale_dw(fd, &SCALES[i], verbose);

    rocket_close(fd);
    return fails ? 1 : 0;
}
