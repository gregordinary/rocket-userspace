// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * requant_edge_probe.c — does the OUT_CVT program the scale the caller asked for, at the
 * derivation's carry edge?
 *
 * The emitters shipped with Mesa's adaptation of QNNPACK's derivation of the OUT_CVT
 * (MUL, SHIFT) pair: MUL = ((bits >> 9) & 0x7FFF) + 1, bit 14 forced when clear. The mask
 * takes the exponent's low bit as MUL's bit 14 and the forcing patches it for an even
 * exponent, so it agrees with QNNPACK everywhere but one pattern, mantissa bits [22:9] all
 * set, where the +1 carries out: at an EVEN biased exponent MUL comes back 0x4000 at the
 * exponent's shift, half the scale, and at an ODD one 0x8000, which both parts read as
 * -32768, so the layer's outputs come back negated [HW sweep, RK1 and H96 MAX M9, this
 * probe, 2026-09-27: every element exact to the vendor pair read signed, none to it read
 * unsigned]. One scale in 16384 sits on an edge. The library's derivation is now
 * npu_out_cvt_pair() (include/npu_requant.h), which renormalizes the carry.
 *
 * NO MODEL-BASED GATE CAN SEE THIS. tests/requant_model.h predicts the device from the
 * pair the emitter programs, so a model and a device that share the derivation agree at
 * the wrong scale. This probe compares against the FLOAT scale instead.
 *
 * THE HOST HALF sweeps every mantissa at exponents 118-127 and counts the scales whose
 * programmed gain MUL / 2^SHIFT leaves [scale, scale * (1 + 2^-14)]: the +1 rounds a
 * scale up by at most one unit of MUL's 15 bits, so nothing else is correct. It scores
 * the vendor form, read unsigned (which must fail on exactly the even-exponent edge
 * patterns, or this probe is not measuring what it says), and the library's form, which
 * must also keep MUL below 0x8000 everywhere.
 *
 * THE DEVICE HALF runs a per-tensor int8-out entry with zero operands, so the accumulator
 * is the bias, at edge scales of both parities and two controls, and reads every element
 * against (a) the float reference sat8(rint(acc * scale)), which must hold within one, and
 * (b) the exact models at the vendor pair (MUL 0x8000 read unsigned, signed, and masked to
 * 15 bits) and at the library's pair, reporting which the part matches. The RK3588 arm
 * drives rocket_conv2d_int8_q (1x1, 32 in, 64 out, 8x8); the RK3576 arm
 * rocket_matmul_int8_rk3576 (M 4, K 32, N 64).
 *
 * What it cannot see: a scale that is not per-tensor (the per-channel planners derive
 * through the same function and are gated by their own probes), the tie rule (see
 * tests/requant_round_probe.c), and any entry other than the two it drives.
 *
 * Exits 1 when the vendor form does not fail exactly on the edge patterns, the library's
 * form fails anywhere, or a device element is more than one from the float; 2 with no
 * device (after the host half has passed); 0 otherwise.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "npu_requant.h"
#include "requant_model.h"

/* The form the emitters shipped with (Mesa's rkt_regcmd.c), kept here as the thing this
 * probe detects. `shift` is the register value. */
static void vendor_pair(float scale, unsigned *mul, unsigned *shift)
{
    union { float f; uint32_t u; } cv = { .f = scale };
    unsigned m = ((cv.u >> 9) & 0x7FFFu) + 1u;
    if (m < (1u << 14)) m |= (1u << 14);
    *mul = m;
    *shift = 127u + 31u - 32u - (cv.u >> 23) + 16u - 1u;
}

typedef void (*pair_fn)(float, unsigned *, unsigned *);

static int is_edge(uint32_t bits) { return ((bits >> 9) & 0x3FFFu) == 0x3FFFu; }

/* Counts the scales at exponents lo..hi whose gain, read unsigned, leaves [1, 1 + 2^-14]
 * of the scale, how many of those are edge patterns, and how many set MUL's sign bit. */
static void sweep(pair_fn f, unsigned lo, unsigned hi, long *bad, long *bad_edge,
                  long *edges, long *wide, double *worst)
{
    *bad = *bad_edge = *edges = *wide = 0;
    *worst = 1.0;
    for (unsigned e = lo; e <= hi; e++)
        for (uint32_t mant = 0; mant < (1u << 23); mant++) {
            union { float f; uint32_t u; } cv = { .u = (e << 23) | mant };
            unsigned mul, shift;
            f(cv.f, &mul, &shift);
            *wide += mul >= (1u << 15);
            const double gain = ldexp((double)mul, -(int)shift);
            const double r = gain / (double)cv.f;
            const int edge = is_edge(cv.u);
            *edges += edge;
            if (!(r >= 1.0 && r <= 1.0 + ldexp(1.0, -14))) {
                (*bad)++;
                *bad_edge += edge;
                if (fabs(r - 1.0) > fabs(*worst - 1.0)) *worst = r;
            }
        }
}

/* ---- the device arms ---------------------------------------------------------------- */

typedef struct { const char *name; uint32_t bits; } edge_scale;
static const edge_scale SCALES[] = {
    { "even edge 0x3F7FFE00", 0x3F7FFE00u },   /* 0.99997 */
    { "even edge 0x3F7FFFFF", 0x3F7FFFFFu },   /* 0.99999994 */
    { "even edge 0x3D7FFE00", 0x3D7FFE00u },   /* 0.0625 - 2^-19 */
    { "odd edge  0x3EFFFE00", 0x3EFFFE00u },   /* 0.49998 */
    { "odd edge  0x3CFFFE00", 0x3CFFFE00u },   /* 0.03125 - 2^-20 */
    { "control   0x3F400000", 0x3F400000u },   /* 0.75 */
    { "control   0x3D4CCCCD", 0x3D4CCCCDu },   /* 0.05 */
};
#define N_SCALES ((int)(sizeof SCALES / sizeof SCALES[0]))
#define N_CH 64

static float as_float(uint32_t b) { union { uint32_t u; float f; } c = { .u = b }; return c.f; }

/* One output per channel: exact output spans about +-96 at every scale. */
static void make_bias(float scale, int32_t *bias)
{
    for (int n = 0; n < N_CH; n++)
        bias[n] = (int32_t)lrint((double)(n - N_CH / 2) * 3.0 / (double)scale) + (n & 1);
}

/* Models at a pair; `mulv` is the multiplier's VALUE, so 0x8000 can be read as +32768,
 * -32768 or 0. */
static int model(int64_t acc, int64_t mulv, unsigned shift)
{
    return requant_sat8(requant_round_shift(acc * mulv, shift));
}

/* Runs one scale; got[n] is channel n's output. 0, or <0 when the entry failed. */
static int run_rk3588(int fd, float scale, const int32_t *bias, int *got)
{
    enum { IC = 32, H = 8, W = 8 };
    rocket_conv2d_desc d;
    int8_t *in = calloc((size_t)IC * H * W, 1), *wt = calloc((size_t)N_CH * IC, 1);
    int8_t *out = malloc((size_t)N_CH * H * W);
    int rc = -1;
    memset(&d, 0, sizeof d);
    d.ic = IC; d.ih = H; d.iw = W; d.oc = N_CH; d.kh = d.kw = 1;
    d.stride_y = d.stride_x = 1; d.dil_y = d.dil_x = 1;
    if (in && wt && out) {
        rc = rocket_conv2d_int8_q(fd, &d, in, wt, bias, 1.0f, scale, 1.0f, 0, 0, 0, out);
        if (rc == 0)
            for (int n = 0; n < N_CH; n++) {
                got[n] = out[(size_t)n * H * W];
                for (int p = 1; p < H * W; p++)
                    if (out[(size_t)n * H * W + p] != out[(size_t)n * H * W]) {
                        printf("      channel %d is not constant over its plane\n", n);
                        rc = -1; break;
                    }
            }
        else printf("      rocket_conv2d_int8_q returned %d\n", rc);
    }
    free(in); free(wt); free(out);
    return rc;
}

static int run_rk3576(int fd, float scale, const int32_t *bias, int *got)
{
    enum { M = 4, K = 32 };
    int8_t *A = calloc((size_t)M * K, 1), *B = calloc((size_t)N_CH * K, 1);
    int8_t *C = malloc((size_t)M * N_CH);
    int rc = -1;
    if (A && B && C) {
        rc = rocket_matmul_int8_rk3576(fd, M, K, N_CH, A, B, bias, scale, C);
        if (rc == 0)
            for (int n = 0; n < N_CH; n++) {
                got[n] = C[n];
                for (int m = 1; m < M; m++)
                    if (C[(size_t)m * N_CH + n] != C[n]) {
                        printf("      column %d is not constant over its rows\n", n);
                        rc = -1; break;
                    }
            }
        else printf("      rocket_matmul_int8_rk3576 returned %d\n", rc);
    }
    free(A); free(B); free(C);
    return rc;
}

int main(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    long bad, bad_edge, edges, wide;
    double worst;
    int fails = 0, fd, is76;

    printf("== OUT_CVT derivation against the float scale, exponents 118-127 ==\n");
    sweep(vendor_pair, 118, 127, &bad, &bad_edge, &edges, &wide, &worst);
    printf("  vendor form:  %ld of %ld scales outside [1, 1+2^-14] read unsigned, %ld of them "
           "edge patterns (%ld edges, half at an even exponent), worst ratio %.6f; %ld set "
           "MUL's sign bit\n", bad, 10L << 23, bad_edge, edges, worst, wide);
    if (bad != edges / 2 || bad_edge != bad || wide != edges / 2) {
        printf("  FAIL: the vendor form should fail on exactly the edges\n");
        fails++;
    }
    sweep(npu_out_cvt_pair, 118, 127, &bad, &bad_edge, &edges, &wide, &worst);
    printf("  library form: %ld outside, %ld set MUL's sign bit; worst ratio %.9f\n",
           bad, wide, worst);
    if (bad || wide) { printf("  FAIL: the library's form is outside the bound\n"); fails++; }

    is76 = strcmp(hw->name, "rk3576") == 0;
    if (!is76 && strcmp(hw->name, "rk3588") != 0) {
        printf("requant_edge_probe: profile %s has no device arm\n", hw->name);
        return fails ? 1 : 2;
    }
    fd = rocket_open();
    if (fd < 0) {
        printf("requant_edge_probe: no NPU device, device half skipped\n");
        return fails ? 1 : 2;
    }

    printf("== the device (%s, %s) ==\n", hw->name,
           is76 ? "rocket_matmul_int8_rk3576" : "rocket_conv2d_int8_q");
    for (int i = 0; i < N_SCALES; i++) {
        const float scale = as_float(SCALES[i].bits);
        int32_t bias[N_CH];
        int got[N_CH];
        unsigned vm, vs, fm, fs;
        int far = 0, worst_d = 0, m_vu = 0, m_vs = 0, m_v15 = 0, m_fx = 0;

        vendor_pair(scale, &vm, &vs);
        npu_out_cvt_pair(scale, &fm, &fs);
        make_bias(scale, bias);
        printf("  %s  scale %.9g  vendor MUL 0x%04x SHIFT %u  library MUL 0x%04x SHIFT %u\n",
               SCALES[i].name, (double)scale, vm, vs, fm, fs);
        if ((is76 ? run_rk3576 : run_rk3588)(fd, scale, bias, got) != 0) { fails++; continue; }
        for (int n = 0; n < N_CH; n++) {
            const int ref = requant_sat8(llrint((double)bias[n] * (double)scale));
            const int d = abs(got[n] - ref);
            if (d > 1) far++;
            if (d > worst_d) worst_d = d;
            m_vu  += got[n] == model(bias[n], (int64_t)vm, vs);
            m_vs  += got[n] == model(bias[n], (int64_t)(int16_t)vm, vs);
            m_v15 += got[n] == model(bias[n], (int64_t)(vm & 0x7FFFu), vs);
            m_fx  += got[n] == model(bias[n], (int64_t)fm, fs);
        }
        printf("      %d of %d more than one from the float (worst %d); exact to the vendor "
               "pair unsigned %d, signed %d, 15-bit %d; to the library's pair %d\n",
               far, N_CH, worst_d, m_vu, m_vs, m_v15, m_fx);
        if (far) fails++;
    }
    rocket_close(fd);
    return fails ? 1 : 0;
}
