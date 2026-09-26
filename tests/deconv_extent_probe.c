// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * deconv_extent_probe.c — does the CNA deconvolution mode compute the WHOLE transposed
 * convolution when the output extent is programmed rather than derived?
 *
 * WHAT IS ALREADY KNOWN (deconv_geometry/arith/pad_probe, encodings/conv-transpose.md).
 * With CONV_CON1 bit 16 set and CONV_CON3's stride field at s-1, the CNA interior-dilates
 * its input by s, the kernel arrives flipped, and the pad lands on the dilated surface at
 * offset (k-1)-P. Those probes all ran the mode through the forward conv entry, which
 * derives the output geometry from the UNDILATED input: ih + 2P - k + 1 rows. The part
 * wrote exactly that many, a complete result needed P >= toh - ih, and the 4-bit pad field
 * then capped one pass at ih <= 14 for s = 2.
 *
 * WHAT THIS ASKS. open-rknpu's RV1106 compiler, on a part whose CNA geometry words sit at
 * the RK3588's offsets, programs the transposed extent directly: DATA_SIZE0 at the compact
 * input, DATA_SIZE2/3, the CORE output size and the DPU cube at toh x tow, and the pad at
 * k-1-p. If the RK3588 honours it too, one task computes the whole ConvTranspose at any
 * input the CBUF holds, and the single-pass bound belonged to the probes' geometry.
 *
 * HOW IT SCORES. The whole surface against rocket_conv_transpose2d_ref_fp16, the direct
 * scatter-add oracle, which is independent of every lowering. A wrong element is counted
 * by where it falls: inside the forward extent ih + 2P - k + 1, or past it. Wrong only
 * past it is the extent bound, i.e. the part walks forward rows whatever it is told. Wrong
 * inside it as well is a different defect, such as the kernel layout or the grain count.
 * Every cell runs twice and reports only when the two runs agree.
 *
 * The CONTROL cell programs the forward extent instead and compares only the rows it
 * covers. It reproduces the recorded truncation, so the harness is known to read the mode
 * the way the earlier probes did.
 *
 * Integer operands small enough that every sum is exact in fp32, so the oracle and the
 * part round the same exact value to fp16 and a mismatch is the hardware. The fills are
 * not periodic along any axis the job tiles.
 *
 * `deconv_extent_probe bench [reps]` times one hardware job against the shipping
 * rocket_conv_transpose2d_fp16 lowering at decoder shapes instead.
 *
 * Needs /dev/accel (the render group suffices). Exits 2 (skip) on a part that is not
 * an RK3588 or with no device, 1 if any cell is wrong.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "rocket_conv_internal.h"

typedef struct {
    const char *name;
    int ic, oc, ih, iw, kh, kw, sy, sx, py, px, opy, opx;
    int forward_extent;   /* 1 = the control: program the forward extent instead */
} cell_t;

/* Every cell but the last holds kh == kw, py == px and opy == opx, so a field swapped
 * between the axes reads the same value either way. The last one differs on the two axes
 * in the kernel, the pad, the stride and the output padding, so a swap there is visible. */
static const cell_t CELLS[] = {
    /* name                    ic  oc  ih  iw kh kw sy sx py px opy opx fwd */
    { "control, fwd extent",   32, 32, 16, 16, 3, 3, 2, 2, 0, 0, 0, 0,  1 },
    { "inside the old bound",  32, 32,  4,  4, 3, 3, 2, 2, 0, 0, 0, 0,  0 },
    { "past it, s=2",          32, 32, 32, 32, 3, 3, 2, 2, 0, 0, 0, 0,  0 },
    { "past it, s=4",          32, 32, 16, 16, 3, 3, 4, 4, 0, 0, 0, 0,  0 },
    { "2x up, k4 p1",          32, 32, 32, 32, 4, 4, 2, 2, 1, 1, 0, 0,  0 },
    { "2x up, k2 p0",          32, 32, 32, 32, 2, 2, 2, 2, 0, 0, 0, 0,  0 },
    { "non-square, p1",        32, 32,  8, 16, 3, 3, 2, 2, 1, 1, 0, 0,  0 },
    { "per-axis, sy2 sx1",     32, 32,  8,  8, 3, 3, 2, 1, 0, 0, 0, 0,  0 },
    { "IC 64 -> OC 32, p1",    64, 32, 16, 16, 3, 3, 2, 2, 1, 1, 0, 0,  0 },
    { "output_padding 1",      32, 32, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1,  0 },
    { "64x64 -> 128x128",      32, 32, 64, 64, 4, 4, 2, 2, 1, 1, 0, 0,  0 },
    { "k4x2 p1,0 s2x4 op1,0",  32, 48, 10, 14, 4, 2, 2, 4, 1, 0, 1, 0,  0 },
};
#define N_CELLS ((int)(sizeof CELLS / sizeof CELLS[0]))

static void set_knob(const char *name, int v)
{
    char b[16];
    snprintf(b, sizeof b, "%d", v);
    setenv(name, b, 1);
}

/* Returns 0 exact, 1 wrong, 2 not comparable (a failed or unreproducible run). */
static int run_cell(int fd, const cell_t *c)
{
    rocket_conv_transpose2d_desc t;
    _Float16 *rin = NULL, *Wt = NULL, *ref = NULL, *Wf = NULL, *out = NULL, *again = NULL;
    int toh, tow, oh, ow, fwd_oh, fwd_ow, P_y, P_x, i, ic, oc, ky, kx, y, x, rc = 2;
    size_t n_rin, n_wt, n_ref, n_out;
    long cmp = 0, bad_in = 0, bad_past = 0;
    double worst = 0.0;
    int fy = -1, fx = -1, foc = -1;
    double fgot = 0, fwant = 0;

    memset(&t, 0, sizeof t);
    t.ic = c->ic; t.oc = c->oc; t.ih = c->ih; t.iw = c->iw;
    t.kh = c->kh; t.kw = c->kw; t.stride_y = c->sy; t.stride_x = c->sx;
    t.pad_top = c->py; t.pad_left = c->px; t.opad_y = c->opy; t.opad_x = c->opx;
    t.dil_y = 1; t.dil_x = 1;
    toh = rocket_conv_transpose2d_oh(&t);
    tow = rocket_conv_transpose2d_ow(&t);

    /* The pad on the dilated surface that makes row 0 of the result row 0 of the
     * transposed conv: ConvTranspose(x, W, s, p) == Conv(dilate_s(x), flip(W), 1, k-1-p). */
    P_y = c->kh - 1 - c->py;
    P_x = c->kw - 1 - c->px;
    fwd_oh = c->ih + 2 * P_y - c->kh + 1;
    fwd_ow = c->iw + 2 * P_x - c->kw + 1;
    oh = c->forward_extent ? fwd_oh : toh;
    ow = c->forward_extent ? fwd_ow : tow;

    printf("  %-22s ic %2d oc %2d  %2dx%-2d k%dx%d s%dx%d p%d,%d op%d,%d -> %3dx%-3d  pad %d,%d, "
           "fwd extent %dx%d\n",
           c->name, c->ic, c->oc, c->ih, c->iw, c->kh, c->kw, c->sy, c->sx, c->py, c->px,
           c->opy, c->opx, toh, tow, P_y, P_x, fwd_oh, fwd_ow);
    if (c->forward_extent)
        printf("  %-22s programs the FORWARD extent %dx%d and compares only those rows\n",
               "", oh, ow);

    n_rin = (size_t)c->ic * c->ih * c->iw;
    n_wt  = (size_t)c->ic * c->oc * c->kh * c->kw;
    n_ref = (size_t)c->oc * toh * tow;
    n_out = (size_t)c->oc * oh * ow;
    rin   = calloc(n_rin, sizeof *rin);
    Wt    = calloc(n_wt,  sizeof *Wt);
    Wf    = calloc(n_wt,  sizeof *Wf);
    ref   = calloc(n_ref, sizeof *ref);
    out   = calloc(n_out, sizeof *out);
    again = calloc(n_out, sizeof *again);
    if (!rin || !Wt || !Wf || !ref || !out || !again) goto done;

    for (i = 0; i < (int)n_rin; i++) rin[i] = (_Float16)((i * 7 + 3) % 11 - 5);
    for (i = 0; i < (int)n_wt;  i++) Wt[i]  = (_Float16)((i * 5 + 1) % 7  - 3);
    rocket_conv_transpose2d_ref_fp16(&t, rin, Wt, ref);

    /* The two transforms the mode leaves to the caller: the 180-degree spatial flip and
     * the channel transpose Wt[IC][OC][KH][KW] -> Wf[OC][IC][KH][KW]. */
    for (ic = 0; ic < c->ic; ic++)
        for (oc = 0; oc < c->oc; oc++)
            for (ky = 0; ky < c->kh; ky++)
                for (kx = 0; kx < c->kw; kx++)
                    Wf[(((size_t)oc * c->ic + ic) * c->kh + ky) * c->kw + kx] =
                        Wt[(((size_t)ic * c->oc + oc) * c->kh + (c->kh - 1 - ky)) * c->kw
                           + (c->kw - 1 - kx)];

    set_knob("ROCKET_CNA_DECONV", 1);
    set_knob("ROCKET_CNA_DECONV_Y", c->sy - 1);
    set_knob("ROCKET_CNA_DECONV_X", c->sx - 1);
    i = rocket_conv2d_fp16_job_extent(fd, c->ic, c->ih, c->iw, c->oc, oh, ow,
                                      c->kh, c->kw, P_y, P_x, rin, Wf, out);
    if (i == 0)
        i = rocket_conv2d_fp16_job_extent(fd, c->ic, c->ih, c->iw, c->oc, oh, ow,
                                          c->kh, c->kw, P_y, P_x, rin, Wf, again);
    unsetenv("ROCKET_CNA_DECONV");
    unsetenv("ROCKET_CNA_DECONV_Y");
    unsetenv("ROCKET_CNA_DECONV_X");
    if (i != 0) {
        printf("  %-22s the job FAILED (%d): no surface to score\n", "", i);
        goto done;
    }
    if (memcmp(out, again, n_out * sizeof *out) != 0) {
        printf("  %-22s NOT REPRODUCIBLE across two runs: not a result\n", "");
        goto done;
    }

    for (oc = 0; oc < c->oc; oc++)
        for (y = 0; y < oh && y < toh; y++)
            for (x = 0; x < ow && x < tow; x++) {
                double g = (double)out[((size_t)oc * oh + y) * ow + x];
                double w = (double)ref[((size_t)oc * toh + y) * tow + x];
                double e = fabs(g - w);
                cmp++;
                if (e > 0.0) {
                    if (foc < 0) { foc = oc; fy = y; fx = x; fgot = g; fwant = w; }
                    if (y < fwd_oh && x < fwd_ow) bad_in++; else bad_past++;
                    if (e > worst) worst = e;
                }
            }

    if (!bad_in && !bad_past) {
        printf("  %-22s BIT-EXACT over %ld elements%s\n", "", cmp,
               c->forward_extent ? " (the rows the forward extent covers)" : ", the whole surface");
        rc = 0;
    } else {
        printf("  %-22s WRONG: %ld inside the forward extent, %ld past it, of %ld "
               "(worst %g); first oc %d (%d,%d) got %g want %g\n", "",
               bad_in, bad_past, cmp, worst, foc, fy, fx, fgot, fwant);
        rc = 1;
    }
done:
    free(rin); free(Wt); free(Wf); free(ref); free(out); free(again);
    return rc;
}

/* ---- bench: the hardware job against the shipping lowering ---------------------------
 *
 * Per call, warm, no ctx on either side, the two arms interleaved rep by rep so board drift
 * lands on both. The hardware arm's time includes the kernel flip and channel transpose,
 * which the shipping entry does internally, so both arms start from the same W[IC][OC][K][K].
 * Both outputs are scored against the oracle, and against each other. Not a gate. */
typedef struct { const char *name; int ic, oc, ih, iw, k, s, p; } bench_t;
static const bench_t BENCH[] = {
    { "64->64   32x32  -> 64x64   k4 s2 p1",  64, 64, 32, 32, 4, 2, 1 },
    { "32->32   64x64  -> 128x128 k4 s2 p1",  32, 32, 64, 64, 4, 2, 1 },
    { "128->64  16x16  -> 32x32   k4 s2 p1", 128, 64, 16, 16, 4, 2, 1 },
    { "64->64   32x32  -> 64x64   k2 s2 p0",  64, 64, 32, 32, 2, 2, 0 },
};
#define N_BENCH ((int)(sizeof BENCH / sizeof BENCH[0]))

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double max_err(const _Float16 *a, const _Float16 *b, size_t n)
{
    double w = 0.0;
    size_t i;
    for (i = 0; i < n; i++) {
        double e = fabs((double)a[i] - (double)b[i]);
        if (e > w) w = e;
    }
    return w;
}

static int hw_job(int fd, const bench_t *b, const rocket_conv_transpose2d_desc *t,
                  const _Float16 *rin, const _Float16 *Wt, _Float16 *Wf, _Float16 *out)
{
    int toh = rocket_conv_transpose2d_oh(t), tow = rocket_conv_transpose2d_ow(t);
    int P = b->k - 1 - b->p, ic, oc, ky, kx, r;
    for (ic = 0; ic < b->ic; ic++)
        for (oc = 0; oc < b->oc; oc++)
            for (ky = 0; ky < b->k; ky++)
                for (kx = 0; kx < b->k; kx++)
                    Wf[(((size_t)oc * b->ic + ic) * b->k + ky) * b->k + kx] =
                        Wt[(((size_t)ic * b->oc + oc) * b->k + (b->k - 1 - ky)) * b->k
                           + (b->k - 1 - kx)];
    set_knob("ROCKET_CNA_DECONV", 1);
    set_knob("ROCKET_CNA_DECONV_Y", b->s - 1);
    set_knob("ROCKET_CNA_DECONV_X", b->s - 1);
    r = rocket_conv2d_fp16_job_extent(fd, b->ic, b->ih, b->iw, b->oc, toh, tow,
                                      b->k, b->k, P, P, rin, Wf, out);
    unsetenv("ROCKET_CNA_DECONV");
    unsetenv("ROCKET_CNA_DECONV_Y");
    unsetenv("ROCKET_CNA_DECONV_X");
    return r;
}

static int run_bench(int fd, int reps)
{
    int i, j, bad = 0;
    printf("== hardware deconvolution job vs the shipping lowering, %d interleaved reps ==\n",
           reps);
    printf("  %-38s %10s %10s %10s %10s %8s  %s\n", "shape", "ship med", "ship min",
           "hw med", "hw min", "hw/ship", "errors vs oracle (ship, hw)");
    for (i = 0; i < N_BENCH; i++) {
        const bench_t *b = &BENCH[i];
        rocket_conv_transpose2d_desc t;
        _Float16 *rin, *Wt, *Wf, *ref, *o_ship, *o_hw;
        double *ts, *th, e_ship, e_hw;
        size_t n_out;
        int toh, tow, ok = 1;

        memset(&t, 0, sizeof t);
        t.ic = b->ic; t.oc = b->oc; t.ih = b->ih; t.iw = b->iw;
        t.kh = b->k; t.kw = b->k; t.stride_y = b->s; t.stride_x = b->s;
        t.pad_top = b->p; t.pad_left = b->p; t.dil_y = 1; t.dil_x = 1;
        toh = rocket_conv_transpose2d_oh(&t);
        tow = rocket_conv_transpose2d_ow(&t);
        n_out = (size_t)b->oc * toh * tow;

        rin    = calloc((size_t)b->ic * b->ih * b->iw, sizeof *rin);
        Wt     = calloc((size_t)b->ic * b->oc * b->k * b->k, sizeof *Wt);
        Wf     = calloc((size_t)b->ic * b->oc * b->k * b->k, sizeof *Wf);
        ref    = calloc(n_out, sizeof *ref);
        o_ship = calloc(n_out, sizeof *o_ship);
        o_hw   = calloc(n_out, sizeof *o_hw);
        ts     = calloc((size_t)reps, sizeof *ts);
        th     = calloc((size_t)reps, sizeof *th);
        if (!rin || !Wt || !Wf || !ref || !o_ship || !o_hw || !ts || !th) return 1;
        for (j = 0; j < b->ic * b->ih * b->iw; j++) rin[j] = (_Float16)((j * 7 + 3) % 11 - 5);
        for (j = 0; j < b->ic * b->oc * b->k * b->k; j++) Wt[j] = (_Float16)((j * 5 + 1) % 7 - 3);
        rocket_conv_transpose2d_ref_fp16(&t, rin, Wt, ref);

        /* Warm-up, discarded: the clock parks at idle and the first call pays for it. */
        if (rocket_conv_transpose2d_fp16(fd, &t, rin, Wt, o_ship) != 0 ||
            hw_job(fd, b, &t, rin, Wt, Wf, o_hw) != 0) {
            printf("  %-38s a warm-up call FAILED\n", b->name);
            ok = 0;
        }
        for (j = 0; ok && j < reps; j++) {
            double t0 = now_ms();
            if (rocket_conv_transpose2d_fp16(fd, &t, rin, Wt, o_ship) != 0) { ok = 0; break; }
            ts[j] = now_ms() - t0;
            t0 = now_ms();
            if (hw_job(fd, b, &t, rin, Wt, Wf, o_hw) != 0) { ok = 0; break; }
            th[j] = now_ms() - t0;
        }
        if (ok) {
            qsort(ts, (size_t)reps, sizeof *ts, cmp_d);
            qsort(th, (size_t)reps, sizeof *th, cmp_d);
            e_ship = max_err(o_ship, ref, n_out);
            e_hw = max_err(o_hw, ref, n_out);
            printf("  %-38s %8.3fms %8.3fms %8.3fms %8.3fms %8.3f  %g, %g%s\n", b->name,
                   ts[reps / 2], ts[0], th[reps / 2], th[0], th[reps / 2] / ts[reps / 2],
                   e_ship, e_hw,
                   memcmp(o_ship, o_hw, n_out * sizeof *o_hw) ? "" : " (outputs identical)");
            if (e_hw > 0.0) bad++;
        } else {
            printf("  %-38s a timed call FAILED\n", b->name);
            bad++;
        }
        free(rin); free(Wt); free(Wf); free(ref); free(o_ship); free(o_hw); free(ts); free(th);
    }
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, i, wrong = 0, skipped = 0;

    if (strcmp(hw->name, "rk3588") != 0) {
        printf("deconv_extent_probe: profile is %s; this drives the RK3588 conv "
               "generator. Skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("deconv_extent_probe: no NPU device. Skipping\n"); return 2; }
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        int reps = argc > 2 ? atoi(argv[2]) : 15;
        int r = run_bench(fd, reps < 3 ? 3 : reps);
        rocket_close(fd);
        return r;
    }

    printf("== CNA DECONV with the transposed extent PROGRAMMED ==\n");
    for (i = 0; i < N_CELLS; i++) {
        int r = run_cell(fd, &CELLS[i]);
        if (r == 1) wrong++;
        if (r == 2) skipped++;
    }
    printf("deconv_extent_probe: %d cells, %d wrong, %d not scored\n",
           N_CELLS, wrong, skipped);
    rocket_close(fd);
    return (wrong || skipped) ? 1 : 0;
}
