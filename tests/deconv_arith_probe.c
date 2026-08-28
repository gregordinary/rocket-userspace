// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * deconv_arith_probe.c — does the CNA DECONV mode compute a real transposed convolution?
 *
 * WHAT THE IMPULSE PROBE ESTABLISHED. `deconv_geometry_probe` read the mode's geometry
 * off an impulse response: with CONV_CON1 bit 16 set and CONV_CON3's stride field F, a
 * single input element at (r,c) produces the kernel at output box
 * [s*r-(k-1) .. s*r] x [s*c-(k-1) .. s*c] with s = F+1, and the kernel appears FLIPPED —
 * the same orientation the plain forward correlation gives.
 *
 * That combination is a fingerprint. Placement at s*r with a flipped kernel and a
 * -(k-1) offset is exactly a forward correlation over an input INTERIOR-DILATED by s.
 * Which is the textbook transposed-convolution identity:
 *
 *     ConvTranspose(x, W, stride s, pad p)  ==  Conv(dilate_s(x), flip(W), stride 1,
 *                                                    pad k-1-p)
 *
 * So the hypothesis this probe tests is precise: THE HARDWARE PERFORMS THE INPUT
 * DILATION, and everything else — the spatial 180-degree flip and the in/out channel
 * transpose — remains the caller's job, exactly as the shipping software lowering already
 * does them. If that holds, the mode removes the s^2-larger dilated input the host
 * currently materialises, which is the whole point of decoding it.
 *
 * WHY AN IMPULSE CANNOT SETTLE THIS. One live input element produces one kernel copy and
 * never exercises the accumulation where two scattered copies overlap. A dilation bug in
 * the accumulate path, a wrong channel walk, or a stride applied on only one axis all
 * survive an impulse and all break a real transposed conv. This probe therefore drives a
 * DENSE input and compares against rocket_conv_transpose2d_ref_fp16, the independent
 * scatter-add oracle the shipping ConvTranspose gate already validates against.
 *
 * THE TWO TRANSFORMS THE CALLER STILL OWES, and why getting either wrong is legible here:
 *   - SPATIAL FLIP.  conv2d correlates; ConvTranspose convolves. W is reversed on both
 *     spatial axes. A missed flip gives a result that is elementwise plausible and
 *     transposed-looking, so it is checked by value, not by eye.
 *   - CHANNEL TRANSPOSE. conv2d wants W[OC][IC][KH][KW]; the transposed-conv reference
 *     wants W[IC][OC][KH][KW]. A missed transpose computes a full, correctly-shaped,
 *     entirely wrong surface — this datapath's signature failure.
 *
 * THE OUTPUT EXTENT IS NOT ENLARGED BY THE MODE, and that bounds what this can check.
 * The CORE/DPU output geometry registers are still driven from the UNDILATED forward
 * arithmetic, so the part writes the forward extent and the dilated result is truncated
 * to it. The impulse probe saw this directly: at s=8 an impulse at row 3 scattered to row
 * 24 of a 22-row surface and channel 0 came back empty. This probe therefore pads the
 * input with zeros to a size whose FORWARD extent covers the transposed result, and
 * compares over the window that survives. With pad 0 the correlation also shifts the
 * result by (k-1), so the comparison is out[y][x] against ref[y+(k-1)][x+(k-1)].
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
#define RH   4          /* the REAL input extent: the transposed conv's input */
#define RW   4
#define CANV 24         /* zero-padded canvas; forward extent CANV-KS+1 = 22 holds it */

int main(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    rocket_conv2d_desc d;
    rocket_conv_transpose2d_desc t;
    _Float16 *in, *Wf, *out, *again, *rin, *Wt, *ref;
    size_t n_in, n_wf, n_out, n_rin, n_wt, n_ref;
    int fd, oh, ow, toh, tow, i, oc, ic, ky, kx, y, x, rc = 0;
    int stride, field;
    char b[8];

    if (strcmp(hw->name, "rk3588") != 0) {
        printf("deconv_arith_probe: profile is %s — this drives the RK3588 conv "
               "generator; skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("deconv_arith_probe: no NPU device — skipping\n"); return 2; }

    printf("== CNA DECONV vs the transposed-conv oracle ==\n");
    printf("   hypothesis: the mode dilates the INPUT by s in hardware; the caller still\n"
           "   owes the 180-degree spatial flip and the [OC][IC] <- [IC][OC] transpose.\n");

    /* Two strides the impulse probe found live. s=8 is skipped: a 4x4 input scatters to
     * row 24 and the 22-row forward extent cannot hold it, which is a property of the
     * extent rule above and not a new fact. */
    for (stride = 2; stride <= 4; stride *= 2) {
        field = stride - 1;

        /* --- the transposed conv we are asking for --- */
        memset(&t, 0, sizeof t);
        t.ic = IC; t.ih = RH; t.iw = RW; t.oc = OC;
        t.kh = KS; t.kw = KS; t.stride_y = stride; t.stride_x = stride;
        t.pad_top = 0; t.pad_left = 0; t.dil_y = 1; t.dil_x = 1;
        toh = rocket_conv_transpose2d_oh(&t);
        tow = rocket_conv_transpose2d_ow(&t);

        /* --- the forward conv we drive, over the zero-padded canvas --- */
        memset(&d, 0, sizeof d);
        d.ic = IC; d.ih = CANV; d.iw = CANV; d.oc = OC;
        d.kh = KS; d.kw = KS; d.stride_y = 1; d.stride_x = 1;
        d.pad_top = 0; d.pad_left = 0; d.dil_y = 1; d.dil_x = 1;
        if (rocket_conv2d_plan(&d) < 0) {
            printf("  s=%d: the forward shape does not plan — skipping\n", stride);
            continue;
        }
        oh = rocket_conv2d_oh(&d); ow = rocket_conv2d_ow(&d);

        n_rin = (size_t)IC * RH * RW;
        n_wt  = (size_t)IC * OC * KS * KS;
        n_ref = (size_t)OC * toh * tow;
        n_in  = (size_t)IC * CANV * CANV;
        n_wf  = (size_t)OC * IC * KS * KS;
        n_out = (size_t)OC * oh * ow;

        rin   = calloc(n_rin, sizeof *rin);
        Wt    = calloc(n_wt,  sizeof *Wt);
        ref   = calloc(n_ref, sizeof *ref);
        in    = calloc(n_in,  sizeof *in);
        Wf    = calloc(n_wf,  sizeof *Wf);
        out   = calloc(n_out, sizeof *out);
        again = calloc(n_out, sizeof *again);
        if (!rin || !Wt || !ref || !in || !Wf || !out || !again) { rc = 1; goto done; }

        /* Small exact-in-fp16 integers, varying on every axis, so a mismatch is the
         * hardware and not rounding. Deliberately NOT periodic along any axis the
         * implementation tiles — a periodic fill has hidden a live tile mixup here
         * before, by making a wrong copy byte-identical to a right one. */
        for (i = 0; i < (int)n_rin; i++) rin[i] = (_Float16)((i * 7 + 3) % 11 - 5);
        for (i = 0; i < (int)n_wt;  i++) Wt[i]  = (_Float16)((i * 5 + 1) % 7  - 3);

        /* The oracle: the direct scatter-add definition, independent of any lowering. */
        rocket_conv_transpose2d_ref_fp16(&t, rin, Wt, ref);

        /* Scatter the real input into the zero canvas at (0,0), UNDILATED — the mode
         * does the dilation. */
        for (ic = 0; ic < IC; ic++)
            for (y = 0; y < RH; y++)
                for (x = 0; x < RW; x++)
                    in[((size_t)ic * CANV + y) * CANV + x] =
                        rin[((size_t)ic * RH + y) * RW + x];

        /* The two transforms the caller owes: spatial 180-degree flip AND the channel
         * transpose Wt[IC][OC][KH][KW] -> Wf[OC][IC][KH][KW]. */
        for (ic = 0; ic < IC; ic++)
            for (oc = 0; oc < OC; oc++)
                for (ky = 0; ky < KS; ky++)
                    for (kx = 0; kx < KS; kx++)
                        Wf[(((size_t)oc * IC + ic) * KS + ky) * KS + kx] =
                            Wt[(((size_t)ic * OC + oc) * KS + (KS-1-ky)) * KS + (KS-1-kx)];

        snprintf(b, sizeof b, "%d", 1);     setenv("ROCKET_CNA_DECONV",   b, 1);
        snprintf(b, sizeof b, "%d", field); setenv("ROCKET_CNA_DECONV_X", b, 1);
        snprintf(b, sizeof b, "%d", field); setenv("ROCKET_CNA_DECONV_Y", b, 1);

        memset(out, 0, n_out * sizeof *out);
        if (rocket_conv2d_fp16(fd, &d, in, Wf, out) != 0) {
            printf("  s=%d: the submit FAILED\n", stride); goto next;
        }
        memset(again, 0, n_out * sizeof *again);
        if (rocket_conv2d_fp16(fd, &d, in, Wf, again) != 0) {
            printf("  s=%d: the repeat FAILED\n", stride); goto next;
        }
        if (memcmp(out, again, n_out * sizeof *out) != 0) {
            printf("  s=%d: NOT REPRODUCIBLE across two runs — not a result\n", stride);
            goto next;
        }

        /* Compare over the window the forward extent can hold, shifted by (k-1) because
         * the pad-0 correlation drops the first (k-1) rows/cols of the true result. */
        {
            int cmp = 0, bad = 0, firsty = -1, firstx = -1;
            double worst = 0.0;
            double gotv = 0, wantv = 0;
            for (oc = 0; oc < OC; oc++)
                for (y = 0; y + KS - 1 < toh && y < oh; y++)
                    for (x = 0; x + KS - 1 < tow && x < ow; x++) {
                        double g = (double)out[((size_t)oc * oh + y) * ow + x];
                        double w = (double)ref[((size_t)oc * toh + (y + KS-1)) * tow
                                               + (x + KS-1)];
                        double e = fabs(g - w);
                        cmp++;
                        if (e > 0.0) {
                            if (!bad) { firsty = y; firstx = x; gotv = g; wantv = w; }
                            bad++;
                            if (e > worst) worst = e;
                        }
                    }
            printf("  s=%d (field %d): transposed out %dx%d, forward extent %dx%d, "
                   "%d elements compared\n", stride, field, toh, tow, oh, ow, cmp);
            if (!cmp) {
                printf("     NOTHING COMPARABLE — the forward extent does not overlap "
                       "the transposed result\n");
            } else if (!bad) {
                printf("     BIT-EXACT against rocket_conv_transpose2d_ref_fp16 over the "
                       "whole window\n");
            } else {
                printf("     %d of %d differ (worst |err| %g); first at oc0 (%d,%d): "
                       "got %g want %g\n", bad, cmp, worst, firsty, firstx, gotv, wantv);
            }
        }
next:
        unsetenv("ROCKET_CNA_DECONV");
        unsetenv("ROCKET_CNA_DECONV_X");
        unsetenv("ROCKET_CNA_DECONV_Y");
        free(rin); free(Wt); free(ref); free(in); free(Wf); free(out); free(again);
        rin = Wt = ref = in = Wf = out = again = NULL;
    }

done:
    rocket_close(fd);
    return rc;
}
