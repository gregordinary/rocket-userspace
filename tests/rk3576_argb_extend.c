// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_argb_extend.c — can the packed-image first conv run at a NON-ZERO input zero
 * point by materialising the pad columns, instead of by repairing them afterwards?
 *
 * THE BOUND. The packed sub-encoding's pad COLUMNS feed the MAC a constant 0 at both ends
 * and at every row, while its pad ROWS and corners honour `CNA_PAD_CON1`
 * [HW sweep, tests/rk3576_argb_padmap.c]. At `in_zp` 0 that is the right answer; at any
 * other zero point every output whose window reaches a pad column is wrong. No value of
 * the register reaches those columns and no single register of the seventeen that differ
 * between the two sub-encodings turns them on. That is what keeps ResNet-18's
 * `ic=3 224x224 k7 s2 pad 3` stem, `in_zp = -128`, on the direct lowering — where the
 * packed program is worth about 2 ms of a 9.2 ms graph.
 *
 * WHY REPAIR IS THE WRONG ROUTE. The error is data-INDEPENDENT — `-in_zp` times the sum of
 * the weights over the taps in a pad column — but it lives in the ACCUMULATOR and the part
 * returns a requantized int8, so repairing an affected output means recomputing it. On that
 * stem it is 3 output columns x 112 rows x 64 channels at 147 taps: 9.8 ms of host work
 * against a ~2 ms saving [tests/rk3576_argb_border_cost.c]. A repair that costs five times
 * what the encoding saves is not a lever.
 *
 * THE ROUTE THIS FILE TESTS. Do not let the hardware synthesize the columns at all.
 * Materialise them as real image columns at the input zero point, and shift the window so
 * the one column the hardware still synthesizes is read only by an output that is thrown
 * away. Output column `x + d` reads the window output column `x` wants when the leading
 * extension is
 *
 *     L = pad - p + stride*d
 *
 * with `p` the register pad the program is given. Output `x' = 0` is then the only one that
 * reads the synthesized column, and it is discarded. The encoding's own geometry bounds —
 * `iw % 16`, `ow % 16`, `ow*stride == iw` — force the programmed input and output WIDER than
 * the model asks for, so the trailing extension is sized to reach the last wanted window and
 * the surplus output columns are cropped.
 *
 * THE LIBRARY OWNS THAT NOW, so a caller passes its OWN geometry and the entry programs the
 * wider one: rocket_conv2d_int8_rk3576() and _pack_rk3576() plan the extension whenever the
 * packed encoding meets a non-zero input zero point, materialise the columns inside the
 * image they already interleave, and crop the surplus in the de-scatter they already run —
 * so the crop is a narrower read rather than a pass of its own. What it forecloses is a cube
 * OUT: the surface is wider than the caller's plane, and a plane's row stride is implied by
 * `iw` with no register to move it.
 *
 * ROCKET_RK3576_ARGB_INZP suppresses the extension as well as the refusal, which is what
 * keeps the caller-built arm below a test of the FORMULA rather than of the entry.
 *
 * THE CELLS VARY WHAT THE FORMULA IS A FUNCTION OF — kernel 3/5/7, stride 1 and 2, model pad
 * 1/2/3 and so extension 1/2/3/4, plane 64 and 224, and both a negative and a positive zero
 * point. A law fitted to one geometry cannot be told from a coincidence at that geometry.
 *
 * FOUR ARMS PER CELL, and the first two are the controls that make the last two mean
 * anything:
 *   1. the DIRECT lowering at the model's own geometry — the ORACLE. It is measured bit-exact
 *      at every zero point including its border, and using it rather than a written reference
 *      keeps a second requant out of the loop;
 *   2. the packed path taken NAIVELY past its zero-point bound — must be wrong, and only on
 *      the outputs whose window reaches a pad column, which is what says the extension
 *      repairs the thing this file claims it does;
 *   3. the packed path on a CALLER-BUILT extended image, cropped by this file — the FORMULA;
 *   4. the entry at the MODEL's own geometry, with no escape set and no buffer built here —
 *      the LIBRARY. It must agree with arm 1 element for element, and its handle must refuse
 *      to hand its surface to a consumer as a cube.
 *
 * Usage: rk3576_argb_extend [reps]
 * Exit:  0 every cell's controls held and arm 3 was exact, 1 otherwise, 2 no NPU/wrong chip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"

struct cell {
    const char *name;
    unsigned ic, oc, ih, iw, k, s, pad;
    int in_zp;
    unsigned p, d;              /* the register pad, and the output shift  */
    unsigned iwx;               /* the programmed width; owx = iwx/s       */
};

static const struct cell CELLS[] = {
    /* ResNet-18's own stem, the shape the lever is for. */
    { "R18 stem k7 s2 pad3", 3, 64, 224, 224, 7, 2, 3, -128, 1, 1, 256 },
    /* A TFLite-shaped kernel at the same plane: a different extension (2, not 4). */
    { "k3 s2 pad1",          3, 32, 224, 224, 3, 2, 1, -128, 1, 1, 256 },
    /* Between the two, and an odd model pad. */
    { "k5 s2 pad2",          3, 64, 224, 224, 5, 2, 2, -128, 1, 1, 256 },
    /* STRIDE 1, where the formula's stride term vanishes and L is 1. */
    { "k3 s1 pad1 (64)",     3, 32,  64,  64, 3, 1, 1,   20, 1, 1,  80 },
};
#define NCELL (sizeof CELLS / sizeof *CELLS)

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int run_cell(int fd, const struct cell *c, int reps)
{
    unsigned oh = c->ih / c->s, ow = c->iw / c->s;
    unsigned L  = c->pad - c->p + c->s * c->d;
    unsigned owx = c->iwx / c->s;
    rocket_conv2d_desc d;
    int8_t *in, *inx, *W, *out, *outx, *ref, *crop;
    uint8_t *touch;
    int rc = ROCKET_OK, failed = 0, r;
    unsigned i, y, x, o;
    double t0, ms_direct = 0.0, ms_ext = 0.0, ms_naive = 0.0, ms_lib = 0.0;
    size_t n_b = 0, n_i = 0;

    /* The geometry the encoding's own bounds demand, checked here rather than discovered
     * from a refusal — a cell that cannot be expressed is a bug in this table. */
    if (c->iwx % 16u || owx % 16u || owx * c->s != c->iwx ||
        L + c->iw > c->iwx || c->d + ow > owx) {
        printf("   %-22s TABLE ERROR: L=%u iwx=%u owx=%u does not satisfy the bounds\n",
               c->name, L, c->iwx, owx);
        return 1;
    }

    in    = malloc((size_t)c->ic * c->ih * c->iw);
    inx   = malloc((size_t)c->ic * c->ih * c->iwx);
    W     = malloc((size_t)c->oc * c->ic * c->k * c->k);
    out   = malloc((size_t)c->oc * oh * ow);
    outx  = malloc((size_t)c->oc * oh * owx);
    ref   = malloc((size_t)c->oc * oh * ow);
    crop  = malloc((size_t)c->oc * oh * ow);
    touch = malloc((size_t)oh * ow);
    if (!in || !inx || !W || !out || !outx || !ref || !crop || !touch) {
        free(in); free(inx); free(W); free(out); free(outx); free(ref); free(crop);
        free(touch);
        return 1;
    }

    /* A full-range image, since a zero point of -128 makes every byte a value of 0..255 and
     * the border error is proportional to it. Weights small and zero-mean, so the interior
     * does not saturate and a wrong border shows rather than being clipped away. */
    for (i = 0; i < c->ic * c->ih * c->iw; i++)
        in[i] = (int8_t)((int)((i * 37u + 11u) & 0xFFu) - 128);
    for (i = 0; i < c->oc * c->ic * c->k * c->k; i++) W[i] = (int8_t)((int)(i % 5u) - 2);

    for (i = 0; i < c->ic; i++)
        for (y = 0; y < c->ih; y++) {
            int8_t *row = inx + ((size_t)i * c->ih + y) * c->iwx;
            for (x = 0; x < c->iwx; x++) row[x] = (int8_t)c->in_zp;
            memcpy(row + L, in + ((size_t)i * c->ih + y) * c->iw, c->iw);
        }

    /* Which outputs reach outside the plane on the X axis. Pure geometry. */
    for (y = 0; y < oh; y++)
        for (x = 0; x < ow; x++) {
            long lo = (long)x * c->s - (long)c->pad;
            touch[y * ow + x] = (lo < 0 || lo + (long)c->k - 1 >= (long)c->iw) ? 1u : 0u;
        }
    for (y = 0; y < oh * ow; y++) { if (touch[y]) n_b++; else n_i++; }
    n_b *= c->oc; n_i *= c->oc;

    memset(&d, 0, sizeof d);
    d.ic = (int)c->ic; d.oc = (int)c->oc; d.ih = (int)c->ih; d.iw = (int)c->iw;
    d.kh = (int)c->k; d.kw = (int)c->k;
    d.stride_y = (int)c->s; d.stride_x = (int)c->s;
    d.dil_y = d.dil_x = 1;
    d.pad_top = (int)c->pad; d.pad_left = (int)c->pad;

    /* --- ARM 1: the direct lowering, exact at any zero point. It IS the oracle. ------ */
    d.direct_datapath = 1;
    for (r = 0; r < reps + 1; r++) {
        if (r == 1) ms_direct = 0.0;
        t0 = now_ms();
        rc = rocket_conv2d_int8_rk3576(fd, &d, in, W, NULL, 1.0f, 1.0f, 512.0f,
                                       c->in_zp, 0, 0, out);
        ms_direct += now_ms() - t0;
        if (rc != ROCKET_OK) break;
    }
    if (rc != ROCKET_OK) {
        printf("   %-22s direct lowering REFUSED (%d)\n", c->name, rc);
        failed = 1;
        goto done;
    }
    memcpy(ref, out, (size_t)c->oc * oh * ow);

    /* --- ARM 2: naive past the bound. The negative control. ------------------------- */
    setenv("ROCKET_RK3576_ARGB_INZP", "1", 1);
    d.direct_datapath = 0;
    for (r = 0; r < reps + 1; r++) {
        if (r == 1) ms_naive = 0.0;
        t0 = now_ms();
        rc = rocket_conv2d_int8_rk3576(fd, &d, in, W, NULL, 1.0f, 1.0f, 512.0f,
                                       c->in_zp, 0, 0, out);
        ms_naive += now_ms() - t0;
        if (rc != ROCKET_OK) break;
    }
    {
        size_t bad_b = 0, bad_i = 0;
        if (rc != ROCKET_OK) {
            printf("   %-22s packed (naive) REFUSED (%d)\n", c->name, rc);
            failed = 1;
        } else {
            for (o = 0; o < c->oc; o++)
                for (y = 0; y < oh; y++)
                    for (x = 0; x < ow; x++) {
                        size_t pi = ((size_t)o * oh + y) * ow + x;
                        if (out[pi] == ref[pi]) continue;
                        if (touch[y * ow + x]) bad_b++; else bad_i++;
                    }
            if (bad_i) {
                printf("   %-22s naive arm moved the INTERIOR (%zu) — the bound is not "
                       "the columns alone here\n", c->name, bad_i);
                failed = 1;
            }
            if (!bad_b) {
                printf("   %-22s naive arm is already exact — this cell does not "
                       "exercise the bound\n", c->name);
                failed = 1;
            }
            printf("   %-22s naive: %6zu/%6zu pad-column wrong, interior clean       "
                   "%7.3f ms\n", c->name, bad_b, n_b, ms_naive / reps);
        }
    }

    /* --- ARM 3: the extended image, cropped ---------------------------------------- */
    memset(&d, 0, sizeof d);
    d.direct_datapath = 0;
    d.ic = (int)c->ic; d.oc = (int)c->oc; d.ih = (int)c->ih; d.iw = (int)c->iwx;
    d.kh = (int)c->k; d.kw = (int)c->k;
    d.stride_y = (int)c->s; d.stride_x = (int)c->s;
    d.dil_y = d.dil_x = 1;
    d.pad_top = (int)c->pad; d.pad_left = (int)c->p;
    d.oh = (int)oh; d.ow = (int)owx;
    for (r = 0; r < reps + 1; r++) {
        if (r == 1) ms_ext = 0.0;
        t0 = now_ms();
        rc = rocket_conv2d_int8_rk3576(fd, &d, inx, W, NULL, 1.0f, 1.0f, 512.0f,
                                       c->in_zp, 0, 0, outx);
        if (rc == ROCKET_OK)
            for (o = 0; o < c->oc; o++)
                for (y = 0; y < oh; y++)
                    memcpy(crop + ((size_t)o * oh + y) * ow,
                           outx + ((size_t)o * oh + y) * owx + c->d, ow);
        ms_ext += now_ms() - t0;
        if (rc != ROCKET_OK) break;
    }
    unsetenv("ROCKET_RK3576_ARGB_INZP");
    if (rc != ROCKET_OK) {
        printf("   %-22s packed EXTENDED REFUSED (%d)\n", c->name, rc);
        failed = 1;
    } else {
        size_t bad = 0;
        for (i = 0; i < c->oc * oh * ow; i++) if (crop[i] != ref[i]) bad++;
        printf("   %-22s L=%u iw %u->%u ow %u->%u  %s (%zu/%zu wrong)      %7.3f ms "
               "vs direct %7.3f\n",
               c->name, L, c->iw, c->iwx, ow, owx,
               bad ? "WRONG" : "bit-exact", bad, (size_t)c->oc * oh * ow,
               ms_ext / reps, ms_direct / reps);
        if (bad) failed = 1;
    }

    /* --- ARM 4: the ENTRY, at the model's own geometry ------------------------------ */
    /* No escape, no extended buffer and no crop here: the descriptor is the same one arm 1
     * ran on the direct datapath, and the only difference is which encoding it asks for. */
    memset(&d, 0, sizeof d);
    d.ic = (int)c->ic; d.oc = (int)c->oc; d.ih = (int)c->ih; d.iw = (int)c->iw;
    d.kh = (int)c->k; d.kw = (int)c->k;
    d.stride_y = (int)c->s; d.stride_x = (int)c->s;
    d.dil_y = d.dil_x = 1;
    d.pad_top = (int)c->pad; d.pad_left = (int)c->pad;
    for (r = 0; r < reps + 1; r++) {
        if (r == 1) ms_lib = 0.0;
        t0 = now_ms();
        rc = rocket_conv2d_int8_rk3576(fd, &d, in, W, NULL, 1.0f, 1.0f, 512.0f,
                                       c->in_zp, 0, 0, out);
        ms_lib += now_ms() - t0;
        if (rc != ROCKET_OK) break;
    }
    if (rc != ROCKET_OK) {
        printf("   %-22s the ENTRY refused the model geometry (%d) — it materialises "
               "nothing here\n", c->name, rc);
        failed = 1;
    } else {
        size_t bad = 0;
        for (i = 0; i < c->oc * oh * ow; i++) if (out[i] != ref[i]) bad++;
        printf("   %-22s entry at the model's own geometry: %s (%zu/%zu wrong)      "
               "%7.3f ms\n", c->name, bad ? "WRONG" : "bit-exact", bad,
               (size_t)c->oc * oh * ow, ms_lib / reps);
        if (bad) failed = 1;
    }

    /* AND ITS SURFACE IS A PLANE INSIDE WIDER ROWS, WHICH THE CUBE SAYS. A materialising
     * handle writes `owx` columns where the caller asked for `ow` and discards the leading
     * `d`, so its output is a SUB-RECTANGLE of its surface. The cube has to carry both
     * numbers: a consumer that reads it as a contiguous `ow`-wide plane takes the surplus
     * columns as data, silently, since the plane is a plausible one. A POOL can be told
     * (the PPU carries the consumed extent and the DDR line stride in different registers);
     * a CONVOLUTION cannot, and must refuse. */
    {
        rocket_conv2d_int8_weights_rk3576 *h;
        h = rocket_conv2d_int8_pack_rk3576(fd, &d, W, NULL, 1.0f, 1.0f, NULL, 512.0f,
                                           c->in_zp, 0, 0);
        if (!h) {
            printf("   %-22s the entry ran but _pack_rk3576() refused the same "
                   "descriptor\n", c->name);
            failed = 1;
        } else {
            rocket_rk3576_cube cube;
            int crc = rocket_conv2d_int8_cube_of_rk3576(h, &cube);
            if (crc != ROCKET_OK) {
                printf("   %-22s cube_of returned %d rather than describing the "
                       "surface\n", c->name, crc);
                failed = 1;
            } else if (cube.w != ow || cube.pitch_w != owx || cube.col_off != c->d) {
                printf("   %-22s cube_of says a %u-wide plane at column %u of %u-element "
                       "rows; the caller's tensor is %u wide at column 1 of %u\n",
                       c->name, cube.w, cube.col_off, cube.pitch_w, ow, owx);
                failed = 1;
            } else {
                /* THE CONSUMER SIDE. A convolution has one register for the width and the
                 * line stride, so this cube has to be refused there — and as UNSUPPORTED,
                 * the geometry refusal, rather than as a shape mismatch. */
                rocket_conv2d_desc dc = d;
                rocket_conv2d_int8_weights_rk3576 *cons;
                int8_t *cw = calloc((size_t)16 * c->oc, 1);
                dc.ic = (int)c->oc; dc.oc = 16; dc.ih = (int)oh; dc.iw = (int)ow;
                dc.kh = dc.kw = 1; dc.stride_x = dc.stride_y = 1;
                dc.pad_top = dc.pad_left = 0; dc.oh = dc.ow = 0;
                dc.direct_datapath = 1;
                cons = cw ? rocket_conv2d_int8_pack_rk3576(fd, &dc, cw, NULL, 1.0f, 1.0f,
                                                           NULL, 512.0f, 0, 0, 0) : NULL;
                if (cons) {
                    int irc = rocket_conv2d_int8_cube_in_rk3576(cons, &cube);
                    if (irc != ROCKET_E_UNSUPPORTED) {
                        printf("   %-22s a convolution consumer returned %d for a cube at "
                               "a row pitch, not the geometry refusal (%d)\n",
                               c->name, irc, ROCKET_E_UNSUPPORTED);
                        failed = 1;
                    }
                    rocket_conv2d_int8_weights_free_rk3576(fd, cons);
                }
                free(cw);
            }
            rocket_conv2d_int8_weights_free_rk3576(fd, h);
        }
    }

done:
    free(in); free(inx); free(W); free(out); free(outx); free(ref); free(crop); free(touch);
    return failed;
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 3;
    int fd, failed = 0;
    unsigned i;

    if (strcmp(rocket_hw_current()->name, "rk3576") != 0) {
        printf("rk3576_argb_extend: profile is %s, not rk3576 — skipping\n",
               rocket_hw_current()->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_argb_extend: no NPU device — skipping\n"); return 2; }

    printf("== packed first conv at a non-zero zero point, by MATERIALISING the columns ==\n");
    printf("   L = pad - p + stride*d, register pad p=1, output shift d=1; the model's\n"
           "   output x is the program's x+d, and the surplus columns are cropped.\n"
           "   The oracle is the DIRECT lowering at the model's own geometry.\n\n");

    for (i = 0; i < NCELL; i++) failed += run_cell(fd, &CELLS[i], reps);

    printf("\n   %s\n", failed
           ? "   AT LEAST ONE CELL FAILED — read the per-cell line."
           : "THE COLUMNS ARE MATERIALISABLE, over kernel 3/5/7, both strides, model pad\n"
             "   1/2/3 and both signs of zero point — by the formula and by the ENTRY, which\n"
             "   owns it at the caller's own geometry. The bound was on the naive\n"
             "   descriptor, not on the encoding.");
    printf("\n   The `ms` columns are whole transient calls — the same residency on every\n"
           "   arm, but not a graph's, where the weights are resident and the crop is a\n"
           "   strided pass over a cube. Read them for the DIRECTION; the graph terms are in\n"
           "   rk3576_argb_border_cost and the execution model.\n");

    rocket_close(fd);
    return failed ? 1 : 0;
}
