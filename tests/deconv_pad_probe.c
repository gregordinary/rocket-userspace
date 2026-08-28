// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * deconv_pad_probe.c — can the CNA DECONV mode take a pad, and where does its result land?
 *
 * WHY THIS MATTERS FOR AN ENCODER. `deconv_arith_probe` proved the mode computes a real
 * transposed convolution — bit-exact against the scatter-add oracle — but it did so with
 * pad 0 over a large zero-padded canvas, because the mode does NOT enlarge the output
 * extent: the CORE/DPU geometry registers are still driven from the UNDILATED forward
 * arithmetic, so the part writes the forward extent and truncates anything past it.
 *
 * That workaround is not an encoder. Padding a 4x4 input out to 24x24 to buy a 22-row
 * output extent moves 36x the input bytes, which would give back the DMA saving the mode
 * exists to provide. A usable encoder needs the COMPACT input plus a pad that buys the
 * extent — so the question is what the mode does with CNA's pad, and whether the pad is
 * applied before or after the hardware dilation. Those two differ by a factor of s in
 * where the result lands, and both produce a full, plausible, correctly-sized surface.
 *
 * WHAT THIS READS, AND WHY IT IS A MAP AND NOT A SWEEP. For each pad P it searches every
 * offset at which the mode's output could match the oracle, and reports THE OFFSET THAT
 * MATCHES rather than pass/fail. A sweep that only asks "does some P work" returns
 * nothing when none does and teaches nothing; an offset map says which rule the hardware
 * follows even when no P gives the alignment an encoder would want:
 *
 *   offset == (k-1) - P        -> the pad is applied to the DILATED surface (the useful
 *                                 case: pad P buys P rows of extent at the true scale)
 *   offset == (k-1) - s*P      -> the pad is applied BEFORE dilation, so each padded row
 *                                 is itself dilated and the pad is worth s rows
 *   no offset matches          -> the pad interacts with the mode in some third way, and
 *                                 an encoder cannot use it until that is decoded
 *
 * Run with `sudo -E` on an RK3588. Exits 2 (skip) elsewhere.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"

#define IC   32
#define OC   32
#define KS   3
#define RH   4
#define RW   4
#define MAXPAD_DEFAULT 4

/* Does the mode's surface equal the oracle's, shifted by (dy,dx)? Compares only where
 * both are in range, and requires a useful number of overlapping elements so a one-element
 * corner cannot read as a match. */
static int matches_at(const _Float16 *got, int oh, int ow,
                      const _Float16 *ref, int toh, int tow,
                      int dy, int dx, int *n_cmp)
{
    int oc, y, x, n = 0;
    for (oc = 0; oc < OC; oc++)
        for (y = 0; y < oh; y++) {
            int ry = y + dy;
            if (ry < 0 || ry >= toh) continue;
            for (x = 0; x < ow; x++) {
                int rx = x + dx;
                if (rx < 0 || rx >= tow) continue;
                if (got[((size_t)oc * oh + y) * ow + x] !=
                    ref[((size_t)oc * toh + ry) * tow + rx]) { *n_cmp = n; return 0; }
                n++;
            }
        }
    *n_cmp = n;
    return n > 0;
}

int main(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, rc = 0, stride, P, maxpad;

    if (strcmp(hw->name, "rk3588") != 0) {
        printf("deconv_pad_probe: profile is %s — RK3588 generator; skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("deconv_pad_probe: no NPU device — skipping\n"); return 2; }
    {   /* The useful pad is bounded by CNA_PAD_CON0's field width, which Mesa's map says
         * is 4 bits (max 15). That map has been wrong on this part before, so the range is
         * swept past the claimed ceiling rather than stopping at it. */
        const char *e = getenv("ROCKET_DECONV_MAXPAD");
        maxpad = e ? (int)strtol(e, NULL, 0) : MAXPAD_DEFAULT;
    }

    printf("== CNA DECONV: what does a pad do, and where does the result land ==\n");
    printf("   compact %dx%d input, k%d. offset (k-1)-P means the pad reaches the DILATED\n"
           "   surface (usable); (k-1)-s*P means it is applied before dilation.\n", RH, RW, KS);

    for (stride = 2; stride <= 4; stride *= 2) {
        int field = stride - 1;
        rocket_conv_transpose2d_desc t;
        _Float16 *rin, *Wt, *ref, *Wf;
        size_t n_rin, n_wt, n_ref, n_wf;
        int toh, tow, i, oc, ic, ky, kx;

        memset(&t, 0, sizeof t);
        t.ic = IC; t.ih = RH; t.iw = RW; t.oc = OC;
        t.kh = KS; t.kw = KS; t.stride_y = stride; t.stride_x = stride;
        t.pad_top = 0; t.pad_left = 0; t.dil_y = 1; t.dil_x = 1;
        toh = rocket_conv_transpose2d_oh(&t);
        tow = rocket_conv_transpose2d_ow(&t);

        n_rin = (size_t)IC * RH * RW;
        n_wt  = (size_t)IC * OC * KS * KS;
        n_ref = (size_t)OC * toh * tow;
        n_wf  = (size_t)OC * IC * KS * KS;
        rin = calloc(n_rin, sizeof *rin);
        Wt  = calloc(n_wt,  sizeof *Wt);
        ref = calloc(n_ref, sizeof *ref);
        Wf  = calloc(n_wf,  sizeof *Wf);
        if (!rin || !Wt || !ref || !Wf) { rc = 1; free(rin); free(Wt); free(ref); free(Wf); break; }

        for (i = 0; i < (int)n_rin; i++) rin[i] = (_Float16)((i * 7 + 3) % 11 - 5);
        for (i = 0; i < (int)n_wt;  i++) Wt[i]  = (_Float16)((i * 5 + 1) % 7  - 3);
        rocket_conv_transpose2d_ref_fp16(&t, rin, Wt, ref);

        for (ic = 0; ic < IC; ic++)
            for (oc = 0; oc < OC; oc++)
                for (ky = 0; ky < KS; ky++)
                    for (kx = 0; kx < KS; kx++)
                        Wf[(((size_t)oc * IC + ic) * KS + ky) * KS + kx] =
                            Wt[(((size_t)ic * OC + oc) * KS + (KS-1-ky)) * KS + (KS-1-kx)];

        printf("\n-- stride %d (field %d), transposed output %dx%d --\n",
               stride, field, toh, tow);

        for (P = 0; P <= maxpad; P++) {
            rocket_conv2d_desc d;
            _Float16 *in, *out, *again;
            size_t n_in, n_out;
            int oh, ow, dy, found = 0, y2, x2;
            char b[8];

            memset(&d, 0, sizeof d);
            d.ic = IC; d.ih = RH; d.iw = RW; d.oc = OC;
            d.kh = KS; d.kw = KS; d.stride_y = 1; d.stride_x = 1;
            d.pad_top = P; d.pad_left = P; d.dil_y = 1; d.dil_x = 1;
            if (rocket_conv2d_plan(&d) < 0) {
                printf("   pad %d: the shape does not PLAN (refused before hardware)\n", P);
                continue;
            }
            oh = rocket_conv2d_oh(&d); ow = rocket_conv2d_ow(&d);
            n_in  = (size_t)IC * RH * RW;
            n_out = (size_t)OC * oh * ow;
            in    = calloc(n_in,  sizeof *in);
            out   = calloc(n_out, sizeof *out);
            again = calloc(n_out, sizeof *again);
            if (!in || !out || !again) { free(in); free(out); free(again); rc = 1; break; }
            memcpy(in, rin, n_in * sizeof *in);

            snprintf(b, sizeof b, "%d", 1);     setenv("ROCKET_CNA_DECONV",   b, 1);
            snprintf(b, sizeof b, "%d", field); setenv("ROCKET_CNA_DECONV_X", b, 1);
            snprintf(b, sizeof b, "%d", field); setenv("ROCKET_CNA_DECONV_Y", b, 1);

            if (rocket_conv2d_fp16(fd, &d, in, Wf, out) != 0) {
                printf("   pad %d: the submit FAILED\n", P); goto cell_done;
            }
            if (rocket_conv2d_fp16(fd, &d, in, Wf, again) != 0) {
                printf("   pad %d: the repeat FAILED\n", P); goto cell_done;
            }
            if (memcmp(out, again, n_out * sizeof *out) != 0) {
                printf("   pad %d: NOT REPRODUCIBLE — not a result\n", P); goto cell_done;
            }

            /* Search the whole plausible offset range on both axes rather than testing
             * the two predicted rules only: a third rule should be readable, not invisible. */
            for (dy = -(int)toh; dy <= (int)toh && !found; dy++)
                for (x2 = -(int)tow; x2 <= (int)tow && !found; x2++) {
                    int n = 0;
                    if (matches_at(out, oh, ow, ref, toh, tow, dy, x2, &n) && n >= 16) {
                        const char *rule = "a third rule";
                        if (dy == (KS-1) - P)          rule = "pad reaches the DILATED surface — USABLE";
                        else if (dy == (KS-1) - stride*P) rule = "pad applied BEFORE dilation";
                        printf("   pad %d: extent %dx%d, MATCHES oracle at offset (%d,%d) "
                               "over %d elements — %s\n", P, oh, ow, dy, x2, n, rule);
                        found = 1;
                    }
                }
            if (!found) {
                int n = 0;
                matches_at(out, oh, ow, ref, toh, tow, 0, 0, &n);
                printf("   pad %d: extent %dx%d, NO offset matches the oracle\n", P, oh, ow);
            }
            y2 = 0; (void)y2;
cell_done:
            unsetenv("ROCKET_CNA_DECONV");
            unsetenv("ROCKET_CNA_DECONV_X");
            unsetenv("ROCKET_CNA_DECONV_Y");
            free(in); free(out); free(again);
        }
        free(rin); free(Wt); free(ref); free(Wf);
    }

    rocket_close(fd);
    return rc;
}
