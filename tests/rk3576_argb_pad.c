// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_argb_pad.c — can the int8 first conv run a TFLite stem after all?
 *
 * The packed-image (ARGB) sub-encoding computes a 224x224 k3 s2 stem in 3.6 ms where
 * the direct path with the image widened to eight channels takes about four times that,
 * so it is worth roughly 4 ms of a MobileNet inference. Two of its geometry bounds are
 * silent and jointly the ONNX symmetric-SAME convention:
 *
 *   THE LEFT PAD MUST BE NON-ZERO — with CNA_PAD_CON0's pad_left field at zero the DPU
 *   writes nothing at all, at every plane, stride, kernel and channel count tried.
 *   THE OUTPUT WIDTH MUST BE iw/stride.
 *
 * A TFLite SAME stem has pad_left = 0: its padding is ASYMMETRIC and the odd column
 * goes at the END. Materialising that border the way the direct path does gives a
 * zero-pad convolution, which this program will not run — and shifting the sample grid
 * by changing the pad instead moves it by ONE, which at stride 2 is half a step and
 * cannot be undone by re-indexing the output. That is the argument this file tests, and
 * it is incomplete: the grid can also be moved by shifting the INPUT, and an input
 * shifted one column against a pad of one is a shift of TWO — a whole stride.
 *
 * So the candidate is: extend the image by TFLite's trailing row and column, then feed
 * the program that buffer starting at ROW 1, COLUMN 1 with both register pads left at
 * 1. Output (y,x) then taps image rows 2y..2y+2 and columns 2x..2x+2, which is exactly
 * TFLite's grid, for every position except the output's own top ROW and left COLUMN —
 * whose outermost taps read the hardware's pad constant instead of image row 0 and
 * column 0. That border is a host fix-up of 1.8% of a 224x224 stem's output.
 *
 * TWO MORE BOUNDS DECIDE THE SHAPE OF THE ANSWER, and both are the entry's own: the
 * packed row's width must be a multiple of 16, and the output extent must satisfy
 * ow*stride == iw AND oh*stride == ih. The second is what makes a zero pad
 * unreachable BEFORE any register is written — at k3 s2 a zero lead gives
 * ow = iw/2 - 1, never iw/2 — so "can the program express a zero left pad" is settled
 * at the geometry and not at CNA_PAD_CON0. It is also why the shift has to be applied
 * to BOTH axes: an input one row taller cannot satisfy oh*stride == ih.
 *
 * Four cases, and the first is the control that makes the rest mean anything:
 *
 *   symmetric   the ONNX stem the path is known to compute. If this is not exact the
 *               run says nothing about the others.
 *
 * A NON-ZERO INPUT ZERO POINT IS A SEPARATE FINDING and this program's third argument
 * drives it. The packed-image path computes a WRONG BORDER there — its interior stays
 * bit-exact and the direct path is exact everywhere against the same reference, so it
 * is the padded taps alone — and the entry now refuses that shape. Passing a non-zero
 * zero point here asserts the refusal rather than the arithmetic. Nothing that had
 * been run was wrong: every first-conv shape in every gate carries a zero input zero
 * point, and a TFLite MobileNet's does too after the uint8 rebase.
 *   zero-pad    pad_left = 0 over the materialised border — what a TFLite stem lowered
 *               the direct path's way asks for. Refused, and the refusal is the point.
 *   zero-top    pad_top = 0 with pad_left = 1, likewise.
 *   tflite      the candidate, against a reference that is TFLite's own SAME grid over
 *               the ORIGINAL image. The output's top row and left column are scored
 *               separately from the interior.
 *
 * THREE OUTCOMES, KEPT APART: NOTHING (the surface is still the stamp), WROTE (it
 * changed and disagrees), EXACT.
 *
 * Run with `sudo -E` on the RK3576.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "requant_model.h"

#define STAMP 0x5A
#define IC     3
#define K      3
#define STRIDE 2

static const float SCALE = 1.0f / 256.0f;

static void fill_i8(int8_t *p, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        p[i] = (int8_t)((int)((seed >> 16) % 61u) - 30);
    }
}

/* A convolution over a CHW tensor with an explicit lead pad per axis. The reference for
 * every case; they differ only in the image, the lead pads and the extents.
 *
 * THE PAD IS THE INPUT ZERO POINT, so in the MODEL domain an out-of-range sample
 * contributes `(in_zp - in_zp) * w` = nothing. Writing it as the stored value `in_zp`
 * times the weight instead is right only at a zero zero point, and at any other it
 * disagrees with the part everywhere — which reads as a hardware result and is a
 * reference bug. */
static int ref_at(const int8_t *img, int ic, int ih, int iw, const int8_t *W, int oc,
                  int o, int y, int x, int lead_y, int lead_x, int in_zp)
{
    int64_t acc = 0;
    int c, ky, kx;
    (void)oc;
    for (c = 0; c < ic; c++)
        for (ky = 0; ky < K; ky++)
            for (kx = 0; kx < K; kx++) {
                int sy = y * STRIDE + ky - lead_y;
                int sx = x * STRIDE + kx - lead_x;
                int v = (sy < 0 || sy >= ih || sx < 0 || sx >= iw)
                        ? 0 : img[((size_t)c * ih + sy) * iw + sx] - in_zp;
                acc += (int64_t)v * W[(((size_t)o * ic + c) * K + ky) * K + kx];
            }
    return requant_scale(acc, SCALE);
}

struct outcome { int touched, exact, total, exact_b, total_b; };

/* Run the entry over `in` at (ih, iw) with the given register pads, then score every
 * output against `ref_*` — which describes a convolution over `rimg` at (rih, riw) with
 * its own lead pads, so a case can be scored against a DIFFERENT geometry than the one
 * it submitted. That is the whole point of the tflite case. */
static int run_case(int fd, const char *name, int oc,
                    const int8_t *in, int ih, int iw, int pad_top, int pad_left,
                    const int8_t *rimg, int rih, int riw, int rlead_y, int rlead_x,
                    int in_zp, const int8_t *W, struct outcome *res)
{
    rocket_conv2d_desc d = {0};
    int32_t *bias = NULL;
    int8_t *out = NULL;
    unsigned ow, oh, n_out, i;
    int rc, o, y, x;

    memset(res, 0, sizeof *res);
    d.ic = IC; d.oc = oc; d.ih = ih; d.iw = iw;
    d.kh = K; d.kw = K;
    d.stride_y = STRIDE; d.stride_x = STRIDE;
    d.pad_top = pad_top; d.pad_left = pad_left;
    d.dil_y = 1; d.dil_x = 1;
    ow = (unsigned)rocket_conv2d_ow(&d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    n_out = (unsigned)oc * ow * oh;

    out  = malloc(n_out);
    bias = calloc(oc, sizeof *bias);
    if (!out || !bias) { free(out); free(bias); return 2; }
    memset(out, STAMP, n_out);

    rc = rocket_conv2d_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f, 1.0f / SCALE,
                                   in_zp, 0, 0, out);
    if (rc != ROCKET_OK) {
        printf("  %-10s REFUSED (%d)\n", name, rc);
        free(out); free(bias);
        return 1;
    }
    for (i = 0; i < n_out; i++)
        if ((unsigned char)out[i] != STAMP) { res->touched = 1; break; }

    for (o = 0; o < oc; o++)
        for (y = 0; y < (int)oh; y++)
            for (x = 0; x < (int)ow; x++) {
                int want = ref_at(rimg, IC, rih, riw, W, oc, o, y, x,
                                  rlead_y, rlead_x, in_zp);
                int got = out[((size_t)o * oh + y) * ow + x];
                if (x == 0 || y == 0) {
                    res->total_b++; if (want == got) res->exact_b++;
                } else {
                    res->total++;   if (want == got) res->exact++;
                }
            }

    printf("  %-10s %s  %ux%u out, %d/%d exact in the interior, %d/%d on the top row "
           "and left column\n",
           name, res->touched ? (res->exact == res->total ? "EXACT  " : "WROTE  ")
                              : "NOTHING",
           ow, oh, res->exact, res->total, res->exact_b, res->total_b);
    free(out); free(bias);
    return 0;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int N = argc > 1 ? atoi(argv[1]) : 224;
    int oc = argc > 2 ? atoi(argv[2]) : 32;
    int in_zp = argc > 3 ? atoi(argv[3]) : 0;
    int8_t *img = NULL, *ext = NULL, *shifted = NULL, *W = NULL;
    struct outcome r;
    int fd, y, x, c, fails = 0, tflite_ok = 0;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_argb_pad: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_argb_pad: no NPU device — skipping\n"); return 2; }

    img     = malloc((size_t)IC * N * N);
    ext     = malloc((size_t)IC * (N + 1) * (N + 1));
    shifted = malloc((size_t)IC * N * N);
    W       = malloc((size_t)oc * IC * K * K);
    if (!img || !ext || !shifted || !W) { fails = 1; goto done; }
    fill_i8(img, (size_t)IC * N * N, 3u);
    fill_i8(W, (size_t)oc * IC * K * K, 11u);

    /* The image extended by TFLite's trailing pad: one row and one column of the input
     * zero point at the END, which is where SAME puts the odd one at stride 2. */
    for (c = 0; c < IC; c++)
        for (y = 0; y <= N; y++)
            for (x = 0; x <= N; x++)
                ext[((size_t)c * (N + 1) + y) * (N + 1) + x] =
                    (y < N && x < N) ? img[((size_t)c * N + y) * N + x] : (int8_t)in_zp;

    /* The candidate's input: the extended image starting at ROW 1, COLUMN 1. Its
     * extent stays NxN, which is what keeps both output-extent bounds satisfied, and
     * against register pads of 1 the sample grid moves by a whole stride on each axis. */
    for (c = 0; c < IC; c++)
        for (y = 0; y < N; y++)
            for (x = 0; x < N; x++)
                shifted[((size_t)c * N + y) * N + x] =
                    ext[((size_t)c * (N + 1) + y + 1) * (N + 1) + x + 1];

    printf("== RK3576 int8 first conv: is a TFLite stem reachable? "
           "%dx%d k%d s%d oc=%d in_zp=%d ==\n", N, N, K, STRIDE, oc, in_zp);

    /* CONTROL ZERO: the SAME convolution on the DIRECT path, with the image widened to
     * eight channels — the lowering the net gate's stem already uses, and one whose
     * zero-point fold is gated. It scores against the same reference, so a border that
     * is exact here and wrong on the packed-image path below is a property of that path
     * and not of the reference. */
    {
        int8_t *wide = calloc((size_t)8 * N * N, 1);
        int8_t *wideW = calloc((size_t)oc * 8 * K * K, 1);
        rocket_conv2d_desc d = {0};
        int32_t *bias = calloc(oc, sizeof *bias);
        int8_t *o8 = NULL;
        unsigned ow, oh, n_out;
        int i, ex = 0, exb = 0, tb = 0, ti = 0;

        if (!wide || !wideW || !bias) { free(wide); free(wideW); free(bias);
                                        fails++; goto done; }
        /* A register override is process-wide and reaches this control too, so a sweep
         * of the packed-image pad constant would abort here on a value the direct path
         * does not like. Under an override the control is skipped and said to be. */
        if (getenv("ROCKET_RK3576_SET")) {
            printf("  %-10s SKIPPED (a register override is in force)\n", "direct");
            free(wide); free(wideW); free(bias);
            goto after_direct;
        }
        for (c = 0; c < IC; c++)
            memcpy(wide + (size_t)c * N * N, img + (size_t)c * N * N, (size_t)N * N);
        for (i = 8 * N * N - 1; i >= IC * N * N; i--) wide[i] = (int8_t)in_zp;
        for (i = 0; i < oc; i++)
            memcpy(wideW + (size_t)i * 8 * K * K, W + (size_t)i * IC * K * K,
                   (size_t)IC * K * K);
        d.ic = 8; d.oc = oc; d.ih = N; d.iw = N;
        d.kh = K; d.kw = K; d.stride_y = STRIDE; d.stride_x = STRIDE;
        d.pad_top = 1; d.pad_left = 1; d.dil_y = 1; d.dil_x = 1;
        ow = (unsigned)rocket_conv2d_ow(&d);
        oh = (unsigned)rocket_conv2d_oh(&d);
        n_out = (unsigned)oc * ow * oh;
        o8 = malloc(n_out);
        if (!o8 || rocket_conv2d_int8_rk3576(fd, &d, wide, wideW, bias, 1.0f, 1.0f,
                                             1.0f / SCALE, in_zp, 0, 0, o8) != ROCKET_OK) {
            printf("  %-10s the direct control did not run\n", "direct");
            fails++;
        } else {
            int o, yy, xx;
            for (o = 0; o < oc; o++)
                for (yy = 0; yy < (int)oh; yy++)
                    for (xx = 0; xx < (int)ow; xx++) {
                        int want = ref_at(img, IC, N, N, W, oc, o, yy, xx, 1, 1, in_zp);
                        int got = o8[((size_t)o * oh + yy) * ow + xx];
                        if (xx == 0 || yy == 0) { tb++; if (want == got) exb++; }
                        else                    { ti++; if (want == got) ex++; }
                    }
            printf("  %-10s %s  %d/%d exact in the interior, %d/%d on the top row and "
                   "left column\n", "direct",
                   ex == ti && exb == tb ? "EXACT  " : "WROTE  ", ex, ti, exb, tb);
            if (ex != ti || exb != tb) {
                printf("     the DIRECT control disagrees too, so the reference is what "
                       "is wrong and nothing below is a statement about the part\n");
                fails++;
            }
        }
        free(wide); free(wideW); free(bias); free(o8);
        if (fails) goto done;
    }
after_direct:

    /* CONTROL: the ONNX symmetric stem, which this path is known to compute. */
    {
        int rc = run_case(fd, "symmetric", oc, img, N, N, 1, 1, img, N, N, 1, 1,
                          in_zp, W, &r);
        if (in_zp != 0) {
            /* The border defect: the shape is refused, and the direct control above
             * having been exact at this same zero point is what makes the refusal a
             * statement about the packed path. */
            if (rc == 1)
                printf("     the packed-image path REFUSES a non-zero input zero point, "
                       "as it should — its border is wrong there and its interior is "
                       "not, and the direct control above is exact at the same zero "
                       "point\n");
            else {
                printf("     the packed-image path ACCEPTED a non-zero input zero "
                       "point; the border refusal has been lost\n");
                fails++;
            }
            goto done;
        }
        if (rc || r.exact != r.total || r.exact_b != r.total_b) {
            printf("     the packed-image control is not exact — nothing below means "
                   "anything\n");
            fails++;
            goto done;
        }
    }

    /* THE BOUND, re-measured: a zero left pad over the materialised border, which is
     * what a TFLite stem lowered the direct path's way would ask for. */
    run_case(fd, "zero-pad", oc, ext, N + 1, N + 1, 0, 0, ext, N + 1, N + 1, 0, 0,
             in_zp, W, &r);
    if (r.touched)
        printf("     IT WROTE. The left-pad bound does not hold at this geometry — "
               "re-open it before building anything on the shift below\n");

    /* The vertical axis on its own terms: pad_top 0 against pad_left 1. */
    run_case(fd, "zero-top", oc, ext, N + 1, N + 1, 0, 1, ext, N + 1, N + 1, 0, 1,
             in_zp, W, &r);

    /* THE CANDIDATE. Scored against TFLite's own SAME grid over the ORIGINAL image:
     * lead 0 on both axes, the trailing row and column supplied by `ext`. */
    if (run_case(fd, "tflite", oc, shifted, N, N, 1, 1,
                 ext, N + 1, N + 1, 0, 0, in_zp, W, &r) == 0) {
        tflite_ok = r.touched && r.exact == r.total;
        if (tflite_ok)
            printf("     A TFLITE STEM RUNS HERE. Every interior output is exact "
                   "against TFLite's own grid; the top row and left column are %d/%d, "
                   "and they are a host fix-up of %.1f%% of the surface (their "
                   "outermost taps read the pad constant instead of image row 0 and "
                   "column 0)\n", r.exact_b, r.total_b,
                   100.0 * (double)r.total_b / (double)(r.total_b + r.total));
        else if (r.touched)
            printf("     it wrote and disagrees — the shift does not reproduce the "
                   "grid, so the parity argument stands with the input shift in it\n");
        else
            printf("     nothing written\n");
    }
    if (!tflite_ok) fails++;

    printf("== %s ==\n", fails ? "the int8 first conv still cannot run a TFLite stem"
                               : "reachable");
done:
    free(img); free(ext); free(shifted); free(W);
    rocket_close(fd);
    return fails ? 1 : 0;
}
