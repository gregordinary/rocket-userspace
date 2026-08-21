// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * deconv_mode_probe.c — is CNA_CONV_CON1 bit 16 (DECONV) a live hardware mode?
 *
 * THE QUESTION. The shipping transposed convolution lowers onto a stride-1 forward conv
 * over an interior-dilated input, which is correct and costs s^2 zero-MACs per output.
 * The CNA register map carries what looks like the hardware mode that would avoid that:
 * CONV_CON1 bit 16 `DECONV`, plus CONV_CON3 [13:11] `DECONV_Y_STRIDE` and [10:8]
 * `DECONV_X_STRIDE` [Mesa registers.xml; allbilly rkt_registers.h — both name all
 * three]. Nothing in this tree has ever written them, and the notes say flatly that the
 * part has no deconvolution mode, which the register map contradicts.
 *
 * The ancestor IP cannot settle it: NVDLA has no deconvolution engine at any revision —
 * its CDMA fetches direct, Winograd or image, and nothing else — so this is Rockchip's
 * own addition and there is no documentation to read. Only the part can answer.
 *
 * WHAT THIS ASKS, AND WHAT IT DOES NOT. One bit: does setting DECONV change the output
 * of a conv the library otherwise computes bit-exactly? A forward conv is run twice on
 * identical operands, once plain and once with the mode bit and a stride field set, and
 * the two surfaces are compared.
 *
 *   identical   -> the bit is inert on this program. Either it is not implemented, or it
 *                  needs a companion field this probe does not write. A negative that
 *                  costs one run and is worth having in writing.
 *   different   -> the mode is live, and the next session has a lead worth an encoder:
 *                  what layout it wants the kernel in, what output geometry it produces,
 *                  and whether the stride fields are `s` or `s-1`.
 *
 * It deliberately does NOT try to make a correct transposed convolution. The output
 * geometry registers are driven from the forward arithmetic, so a real deconvolution
 * would want a larger surface than this program describes; a wrong answer here is not
 * evidence against the mode. The comparison is "did anything change", nothing more.
 *
 * Run with `sudo -E` on an RK3588. Exits 2 (skip) elsewhere — the RK3576 has its own
 * encoder and this generator refuses there by construction.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"

#define IC 32
#define OC 32
#define IWH 8
#define KS 3

static int run_once(int fd, const rocket_conv2d_desc *d, const _Float16 *in,
                    const _Float16 *W, _Float16 *out, size_t n_out,
                    int bit, int fx, int fy)
{
    char b[8];
    memset(out, 0, n_out * sizeof *out);
    snprintf(b, sizeof b, "%d", bit); setenv("ROCKET_CNA_DECONV",   b, 1);
    snprintf(b, sizeof b, "%d", fx);  setenv("ROCKET_CNA_DECONV_X", b, 1);
    snprintf(b, sizeof b, "%d", fy);  setenv("ROCKET_CNA_DECONV_Y", b, 1);
    return rocket_conv2d_fp16(fd, d, in, W, out);
}

static int differs(const _Float16 *a, const _Float16 *b, size_t n, size_t *first)
{
    size_t i, d = 0;
    for (i = 0; i < n; i++)
        if (memcmp(&a[i], &b[i], sizeof a[i]) != 0) { if (!d) *first = i; d++; }
    return (int)d;
}

int main(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    rocket_conv2d_desc d;
    _Float16 *in, *W, *base, *got;
    size_t n_in, n_w, n_out, first = 0;
    int fd, oh, ow, i, rc = 0, nz = 0, bit, f;
    int verbose = getenv("ROCKET_DECONV_VERBOSE") != NULL;
    _Float16 *again;

    if (strcmp(hw->name, "rk3588") != 0) {
        printf("deconv_mode_probe: profile is %s — this drives the RK3588 conv "
               "generator; skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("deconv_mode_probe: no NPU device — skipping\n"); return 2; }

    memset(&d, 0, sizeof d);
    d.ic = IC; d.ih = IWH; d.iw = IWH; d.oc = OC;
    d.kh = KS; d.kw = KS; d.stride_y = 1; d.stride_x = 1;
    d.pad_top = 0; d.pad_left = 0; d.dil_y = 1; d.dil_x = 1;
    if (rocket_conv2d_plan(&d) < 0) {
        printf("deconv_mode_probe: the probe shape does not plan — skipping\n");
        rocket_close(fd);
        return 2;
    }
    oh = rocket_conv2d_oh(&d); ow = rocket_conv2d_ow(&d);
    n_in = (size_t)IC * IWH * IWH;
    n_w  = (size_t)OC * IC * KS * KS;
    n_out = (size_t)OC * oh * ow;

    in   = calloc(n_in,  sizeof *in);
    W    = calloc(n_w,   sizeof *W);
    base = calloc(n_out, sizeof *base);
    got  = calloc(n_out, sizeof *got);
    if (!in || !W || !base || !got) { rc = 1; goto done; }

    /* Vary on every axis, and keep the magnitudes exact in fp16 so a difference is the
     * hardware and not a rounding artefact. */
    for (i = 0; i < (int)n_in; i++) in[i] = (_Float16)((i % 7) - 3);
    for (i = 0; i < (int)n_w;  i++) W[i]  = (_Float16)((i % 5) - 2);

    if (run_once(fd, &d, in, W, base, n_out, 0, 0, 0) != 0) {
        printf("  the plain forward conv failed — nothing to compare against\n");
        rc = 1; goto done;
    }
    for (i = 0; i < (int)n_out; i++) if (base[i] != (_Float16)0) nz++;
    printf("== CNA DECONV (CONV_CON1 bit 16) on a %dx%dx%d k%d s1 conv, out %dx%d ==\n",
           IC, IWH, IWH, KS, oh, ow);
    printf("  baseline (bit clear): %d of %d output elements non-zero\n", nz, (int)n_out);

    again = calloc(n_out, sizeof *again);
    if (!again) { rc = 1; goto done; }

    /* Sweep both 3-bit fields with the mode bit SET, and the same fields with it CLEAR.
     * The pair is what separates "the bit does something" from "those CONV_CON3 bits do
     * something whatever bit 16 says" — they sit in a range this library has always left
     * at zero, so either could be the live one. Every cell is run TWICE and reported only
     * if the two agree, because one reading of a surface on this part is not a result. */
    for (bit = 1; bit >= 0; bit--) {
        printf("  -- DECONV bit %d --\n", bit);
        for (f = 0; f < 8; f++) {
            int nd, nd2;
            if (run_once(fd, &d, in, W, got, n_out, bit, f, f) != 0) {
                printf("     field %d: the submit FAILED\n", f);
                continue;
            }
            if (run_once(fd, &d, in, W, again, n_out, bit, f, f) != 0) {
                printf("     field %d: the repeat FAILED\n", f);
                continue;
            }
            nd  = differs(base, got,  n_out, &first);
            nd2 = differs(base, again, n_out, &first);
            if (nd != nd2 || differs(got, again, n_out, &first)) {
                printf("     field %d: NOT REPRODUCIBLE (%d then %d elements differ) — "
                       "read this as noise, not as a mode\n", f, nd, nd2);
                continue;
            }
            if (!nd)
                printf("     field %d: byte-identical to the forward conv\n", f);
            else {
                int z = 0, y, x;
                for (i = 0; i < (int)n_out; i++) if (got[i] == (_Float16)0) z++;
                differs(base, got, n_out, &first);
                printf("     field %d: %4d of %d elements differ (first at %zu: %g -> %g), "
                       "%d zero\n", f, nd, (int)n_out, first,
                       (double)base[first], (double)got[first], z);
                /* Channel 0's plane, so the next session can see whether the surface has
                 * the sparse structure a scatter into a dilated grid would leave. */
                if (verbose)
                    for (y = 0; y < oh; y++) {
                        printf("        ");
                        for (x = 0; x < ow; x++)
                            printf("%7g", (double)got[(size_t)y * ow + x]);
                        printf("\n");
                    }
            }
        }
    }
    free(again);
    unsetenv("ROCKET_CNA_DECONV");
    unsetenv("ROCKET_CNA_DECONV_X");
    unsetenv("ROCKET_CNA_DECONV_Y");

done:
    free(in); free(W); free(base); free(got);
    rocket_close(fd);
    return rc;
}
