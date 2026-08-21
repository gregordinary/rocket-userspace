// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_conv_sym.c — THE SYMMETRY AUDIT, on the CNA.
 *
 * A field or a formula that is fitted where `kh == kw` or `sy == sx` cannot be
 * distinguished from its transpose by any cell that holds both pairs square, and both
 * failure modes look identical from outside: the part computes a full, correctly sized,
 * entirely plausible surface that is wrong. The RK3576's pooling block has produced one of
 * each — a transposed pad-nibble map, and an output-width allowance keyed on the kernel
 * width where the axis is the height — and every cell of `rk3576_conv_lib_gate` sets
 * `d.kh = d.kw` and `d.stride_y = d.stride_x`, so the convolution block's two remaining
 * axes have never been driven.
 *
 * THE METHOD IS A PAIR, and which half fails is the answer. A transpose maps each cell of
 * a pair onto the other, so a transposed FIELD fails both halves; a formula keyed on the
 * wrong axis fails one. Run the pair at one rectangular plane and read which half failed.
 *
 * Two arms:
 *
 *   stride   NON-SQUARE STRIDES. `s2x1` against `s1x2` at a rectangular plane, over
 *            square and non-square kernels and the depthwise path. Nothing in the corpus
 *            has one — every graph here is isotropic in stride — so this is unverified
 *            rather than suspected.
 *
 *   rung     THE CBUF `F` RUNG LIVENESS, which is a FORMULA fitted at kh == kw. Rungs 256
 *            and 512 deliver only 4096 granules — the F=0 budget — at a kernel with
 *            vertical extent, so the planner offers them only where `kh == 1`. That was
 *            measured at k=1 against k=3 and k=5, all SQUARE, so "kh" and "any kernel with
 *            more than one tap" fit it equally, and the two differ exactly at `k1xN`: the
 *            planner offers the rung there, and if the axis is the tap count the task
 *            overruns its allowance and writes a wrong TAIL. Forced with
 *            ROCKET_RK3576_CBUF_RUNGS so a shape lands on the rung under test, and the F
 *            the planner chose is REPORTED per cell rather than assumed.
 *
 * A PROBE that asserts one thing: a cell the library ACCEPTED must be bit-exact. A cell it
 * refuses is reported and is not a failure — the envelope is allowed to be narrow, it is
 * not allowed to be wrong.
 *
 * Usage: rk3576_conv_sym [stride|rung|all]
 * Exit:  0, 1 on a wrong surface, 2 to skip (no NPU or wrong chip).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "npu_regcmd_rk3576.h"
#include "requant_model.h"

static int NFAIL, NRUN, NREFUSED;

/* One cell: build the descriptor, run the entry, score every output element against a CPU
 * model of the part's own arithmetic. `f_note` is printed beside the result so a cell that
 * did not reach the rung it was written for says so. */
static void cell(int fd, const char *name, unsigned ic, unsigned oc, unsigned iw,
                 unsigned ih, unsigned kh, unsigned kw, unsigned sy, unsigned sx,
                 int pad_y, int pad_x, int dw, int in_zp, int w_zp, int out_zp)
{
    rocket_conv2d_desc d;
    int8_t *in = NULL, *W = NULL, *out = NULL;
    int32_t *bias = NULL;
    unsigned ow, oh, terms, divisor, scale, shift_reg, seed, f = 0;
    unsigned c, y, x, ky, kx, i;
    size_t wrong = 0, total;
    int rc, first_c = -1, first_y = -1, first_x = -1, got0 = 0, want0 = 0;

    memset(&d, 0, sizeof d);
    d.ic = (int)ic; d.oc = (int)oc; d.ih = (int)ih; d.iw = (int)iw;
    d.kh = (int)kh; d.kw = (int)kw;
    d.stride_y = (int)sy; d.stride_x = (int)sx;
    d.pad_top = pad_y; d.pad_left = pad_x;
    d.dil_y = 1; d.dil_x = 1;
    d.depthwise = dw;
    ow = (unsigned)rocket_conv2d_ow(&d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    if (!ow || !oh) { printf("  %-26s geometry is empty\n", name); return; }

    /* What the row planner chose, so "this cell reached the rung" is an output. */
    if (rocket_rk3576_cbuf_f(iw, ic, ih, oc, kh, kw, dw, &f) != 0) f = (unsigned)-1;

    terms   = dw ? (kh * kw) : (ic * kh * kw);
    divisor = 1;
    while ((double)divisor < 2.0 * sqrt((double)terms)) divisor *= 2;
    requant_params(1.0f / (float)divisor, &scale, &shift_reg);

    in   = calloc((size_t)ic * ih * iw, 1);
    W    = calloc(dw ? (size_t)ic * kh * kw : (size_t)oc * ic * kh * kw, 1);
    out  = calloc((size_t)oc * oh * ow, 1);
    bias = calloc(oc, sizeof *bias);
    if (!in || !W || !out || !bias) goto done;

#define INP(c_, y_, x_)      in[(((size_t)(c_) * ih) + (y_)) * iw + (x_)]
#define WD(oc_, ic_, h_, w_) W[((((size_t)(oc_) * ic + (ic_)) * kh + (h_)) * kw + (w_))]
#define WW(c_, h_, w_)       W[(((size_t)(c_) * kh + (h_)) * kw + (w_))]

    /* Varying on EVERY axis. A feature flat along an axis proves nothing about that
     * axis's stride, which is the whole question here. */
    seed = 0x9E3779B9u ^ (unsigned)(ic * 31 + oc * 17 + iw * 7 + ih * 3 + kh * 5 + kw);
    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++) {
                int v = ((int)((c * 7 + y * 13 + x * 3) % 61)) - 30 + in_zp;
                INP(c, y, x) = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
            }
    for (c = 0; c < (dw ? ic : oc); c++)
        for (i = 0; i < (unsigned)(dw ? 1u : ic); i++)
            for (ky = 0; ky < kh; ky++)
                for (kx = 0; kx < kw; kx++) {
                    int v;
                    seed = seed * 1103515245u + 12345u;
                    v = (int)((seed >> 16) % 17u) - 8 + w_zp;
                    if (dw) WW(c, ky, kx) = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
                    else    WD(c, i, ky, kx) = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
                }
    for (c = 0; c < oc; c++) bias[c] = (int32_t)((int)c - (int)oc / 2) * 8;

    rc = dw ? rocket_conv2d_dw_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f,
                                           (float)divisor, in_zp, w_zp, out_zp, out)
            : rocket_conv2d_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f,
                                        (float)divisor, in_zp, w_zp, out_zp, out);
    if (rc == ROCKET_E_UNSUPPORTED) {
        printf("  %-26s REFUSED by the library (F=%d) — the envelope, not a failure\n",
               name, (int)f);
        NREFUSED++;
        goto done;
    }
    if (rc != ROCKET_OK) {
        printf("  %-26s the entry returned %d\n", name, rc);
        NFAIL++; NRUN++;
        goto done;
    }

    NRUN++;
    total = (size_t)oc * oh * ow;
    for (c = 0; c < oc; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                int64_t acc = bias[c];
                int got, want;
                for (ky = 0; ky < kh; ky++)
                    for (kx = 0; kx < kw; kx++) {
                        int iy = (int)(y * sy + ky) - pad_y;
                        int ix = (int)(x * sx + kx) - pad_x;
                        if (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw) continue;
                        if (dw)
                            acc += (int64_t)(INP(c, iy, ix) - in_zp) *
                                   (WW(c, ky, kx) - w_zp);
                        else
                            for (i = 0; i < ic; i++)
                                acc += (int64_t)(INP(i, iy, ix) - in_zp) *
                                       (WD(c, i, ky, kx) - w_zp);
                    }
                want = requant_apply_zp(acc, scale, shift_reg, out_zp);
                got  = out[((size_t)c * oh + y) * ow + x];
                if (got != want) {
                    if (!wrong) { first_c = (int)c; first_y = (int)y; first_x = (int)x;
                                  got0 = got; want0 = want; }
                    wrong++;
                }
            }
    if (wrong) {
        printf("  %-26s WRONG: %zu of %zu, first at c%d y%d x%d (%d vs %d), F=%d\n",
               name, wrong, total, first_c, first_y, first_x, got0, want0, (int)f);
        NFAIL++;
    } else {
        printf("  %-26s exact over %zu element(s), F=%d\n", name, total, (int)f);
    }
done:
#undef INP
#undef WD
#undef WW
    free(in); free(W); free(out); free(bias);
}

/* ---- the STRIDE arm ------------------------------------------------------------
 * Each row is a PAIR at one rectangular plane: a transpose maps one cell onto the other,
 * so a transposed field fails BOTH and a formula keyed on the wrong axis fails one. */
static void arm_stride(int fd)
{
    printf("\n== NON-SQUARE STRIDES, as pairs at a rectangular plane ==\n");
    printf("   every cell of rk3576_conv_lib_gate sets stride_y == stride_x, so this axis "
           "is unverified\n");
    /* Square kernel, both strides asymmetric each way. */
    cell(fd, "k3x3 s2x1 48x32",     32, 32, 48, 32, 3, 3, 2, 1, 1, 1, 0, -5, 3, -2);
    cell(fd, "k3x3 s1x2 48x32",     32, 32, 48, 32, 3, 3, 1, 2, 1, 1, 0, -5, 3, -2);
    cell(fd, "k5x5 s2x1 48x32",     32, 32, 48, 32, 5, 5, 2, 1, 2, 2, 0, -5, 3, -2);
    cell(fd, "k5x5 s1x2 48x32",     32, 32, 48, 32, 5, 5, 1, 2, 2, 2, 0, -5, 3, -2);
    /* Both pairs asymmetric at once, which a square-kernel cell cannot reach. */
    cell(fd, "k3x1 s2x1 48x32",     32, 32, 48, 32, 3, 1, 2, 1, 1, 0, 0, -5, 3, -2);
    cell(fd, "k1x3 s1x2 48x32",     32, 32, 48, 32, 1, 3, 1, 2, 0, 1, 0, -5, 3, -2);
    cell(fd, "k3x1 s1x2 48x32",     32, 32, 48, 32, 3, 1, 1, 2, 1, 0, 0, -5, 3, -2);
    cell(fd, "k1x3 s2x1 48x32",     32, 32, 48, 32, 1, 3, 2, 1, 0, 1, 0, -5, 3, -2);
    /* The DEPTHWISE datapath is a different weight cube and a different program. */
    cell(fd, "dw k3x3 s2x1 48x32",  64, 64, 48, 32, 3, 3, 2, 1, 1, 1, 1, -5, 3, -2);
    cell(fd, "dw k3x3 s1x2 48x32",  64, 64, 48, 32, 3, 3, 1, 2, 1, 1, 1, -5, 3, -2);
    /* A stride larger than the kernel on one axis only, where the window grid skips. */
    cell(fd, "k3x3 s3x1 48x33",     32, 32, 48, 33, 3, 3, 3, 1, 0, 0, 0, 0, 0, -8);
    cell(fd, "k3x3 s1x3 48x33",     32, 32, 48, 33, 3, 3, 1, 3, 0, 0, 0, 0, 0, -8);
}

/* ---- the RUNG arm --------------------------------------------------------------
 *
 * The planner offers the F=256 and F=512 rungs only where `kh == 1`, from a measurement
 * taken at SQUARE k1 / k3 / k5 — so `the kernel height` and `any kernel with more than one
 * tap` both fit it, and `k1xN` is where the two readings differ.
 *
 * THE ARM THAT MUST SUCCEED IS `k1x1`, because that is where the rung was measured to
 * deliver its face value. A probe whose control fails has found something about the
 * control, not about the cells it was written for — so this arm is a MAP rather than a
 * verdict: the F=0 boundary is crossed row by row, and the same granule TOTAL is reached
 * through different (row size, row count) factorizations.
 *
 * `entries` is ceil(iw*ic/64) granules a row and the budget is 4096+F, so F=0 carries
 * 4096/entries rows and F=256 carries 4352/entries. The original measurement's own cell is
 * here as `iw16 ic32` — five plane widths at ic 32, 4352 granules, all reported exact —
 * beside the same total reached at a wider row.
 *
 * ROCKET_RK3576_MAX_ROWS caps the window, so the SAME plane can be run as one task on the
 * rung and as two under it: if the tall cell is wrong and the capped one is exact, the
 * rung is the axis and nothing else in the shape is.
 */
static void rung_capped(int fd, const char *name, unsigned ic, unsigned oc, unsigned iw,
                        unsigned ih, unsigned rows)
{
    char buf[16];
    snprintf(buf, sizeof buf, "%u", rows);
    setenv("ROCKET_RK3576_MAX_ROWS", buf, 1);
    cell(fd, name, ic, oc, iw, ih, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    unsetenv("ROCKET_RK3576_MAX_ROWS");
}

static void arm_rung(int fd)
{
    printf("\n== THE CBUF F RUNGS, AS A MAP ==\n");
    printf("   k1x1 is the arm that must SUCCEED — it is where 256 and 512 were measured "
           "to deliver — and k1xN is where `kh` and `any kernel with more than one tap` "
           "differ\n");

    printf("\n   (a) one row size, the task height walked across the F=0 boundary "
           "(iw32 ic128: 64 granules a row, so F=0 carries 64 rows and F=256 carries 68)\n");
    cell(fd, "k1x1 60 rows  F=0",   128, 32, 32,  60, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "k1x1 64 rows  F=0",   128, 32, 32,  64, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "k1x1 65 rows  F=256", 128, 32, 32,  65, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "k1x1 68 rows  F=256", 128, 32, 32,  68, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "k1x1 72 rows  F=512", 128, 32, 32,  72, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "k1x1 80 rows  F=1024",128, 32, 32,  80, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    printf("   the SAME planes forced under the F=0 boundary, which must all be exact:\n");
    rung_capped(fd, "k1x1 68 rows cap 64", 128, 32, 32, 68, 64);
    rung_capped(fd, "k1x1 72 rows cap 64", 128, 32, 32, 72, 64);
    rung_capped(fd, "k1x1 80 rows cap 64", 128, 32, 32, 80, 64);

    printf("\n   (b) ONE granule total (4352 = the F=256 rung exactly), reached through "
           "different (row size, row count) factorizations\n");
    cell(fd, "4352gr iw16  ic32",    32, 32,  16, 544, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "4352gr iw32  ic32",    32, 32,  32, 272, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "4352gr iw64  ic32",    32, 32,  64, 136, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "4352gr iw128 ic32",    32, 32, 128,  68, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "4352gr iw64  ic64",    64, 32,  64,  68, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "4352gr iw32  ic128",  128, 32,  32,  68, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);

    /* `ic`, the weight SLICE (32*ic*kh*kw bytes, per output-channel group) and the whole
     * CUBE (that times the group count) all rise together in (b), so it separates none of
     * them. `oc` moves the cube alone and `ic` at a fixed kernel moves the slice alone. */
    printf("\n   (b2) which of ic, the weight SLICE and the whole CUBE the wall follows — "
           "(b) moves all three together\n");
    cell(fd, "ic32 oc1024 (cube x32)",  32,1024,  32,  68, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "ic48  (slice 24 gr)",     48,  32,  32,  68, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "ic40  (slice 20 gr)",     40,  32,  32,  68, 1, 1, 1, 1, 0, 0, 0, -4, 2, -6);
    cell(fd, "ic32 k2x2 (slice 64 gr)", 32,  32,  32,  68, 2, 2, 1, 1, 0, 0, 0, -4, 2, -6);

    printf("\n   (c) the cells the audit was written for: k1xN at a rung, where the "
           "planner offers one and the kernel has horizontal extent\n");
    cell(fd, "k1x3 68 rows  F=256", 128, 32, 32, 68, 1, 3, 1, 1, 0, 1, 0, -4, 2, -6);
    cell(fd, "k1x5 68 rows  F=256", 128, 32, 32, 68, 1, 5, 1, 1, 0, 2, 0, -4, 2, -6);
    cell(fd, "k1x7 68 rows  F=256", 128, 32, 32, 68, 1, 7, 1, 1, 0, 3, 0, -4, 2, -6);
    cell(fd, "k3x1 68 rows  (ctl)", 128, 32, 32, 68, 3, 1, 1, 1, 1, 0, 0, -4, 2, -6);
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "all";
    int fd;

    if (rocket_hw_current() != &rocket_hw_rk3576) {
        printf("SKIP: this probe is the RK3576's CNA\n");
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("SKIP: no /dev/accel/accel0\n"); return 2; }

    if (!strcmp(mode, "stride") || !strcmp(mode, "all")) arm_stride(fd);
    if (!strcmp(mode, "rung")   || !strcmp(mode, "all")) arm_rung(fd);

    printf("\n%s: %d cell(s) run, %d wrong, %d refused by the library\n",
           NFAIL ? "FAIL" : "PASS", NRUN, NFAIL, NREFUSED);
    rocket_close(fd);
    return NFAIL ? 1 : 0;
}
