// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_argb_ic1.c — the int8 first conv at ONE image channel, against a CPU model.
 *
 * A single-channel packed image is what a depth or grayscale stem opens with, and it
 * is the one channel count the emitted program will not fetch: the feature DMA's row
 * width (`0x1078` bits[31:16]) carries `line_stride - 1`, which is right from ic=2 up
 * and at ic=1 leaves the DPU writing nothing at all — an untouched surface, not a
 * wrong one. Raising that field alone revives the write and nothing is exact, because
 * the DMA then reads past the packed row. So the entry widens the ROW instead, to two
 * interleaved channels with a zero second lane against zero weights, and this gates
 * that: ic=1 must be bit-exact and must match its ic=2 control shape for shape.
 *
 * The three outcomes are kept apart deliberately, because two of them look alike:
 *
 *   NOTHING — the surface is still the caller's stamp. A dead program.
 *   WROTE   — the surface changed and disagrees with the model.
 *   EXACT   — every element.
 *
 * A register candidate is swept with ROCKET_RK3576_SET (`0x1078=0x0004003f` and so
 * on), which patches the emitted program after emission, so a sweep never touches the
 * transcribed one; ROCKET_RK3576_DUMP prints what was submitted, and two runs diffed
 * is how the ic=1 gap was narrowed to one register. Pair any candidate with the ic=2
 * control in the same run — an override that revives ic=1 by breaking ic=2 has found
 * nothing.
 *
 * Run with `sudo -E` on the RK3576.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "requant_model.h"

#define STAMP 0x5A

static void fill_i8(int8_t *p, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        p[i] = (int8_t)((int)((seed >> 16) % 31u) - 15);
    }
}

/* The model the on-chip requant approximates, in the caller's terms.
 * tests/requant_model.h carries the rule, ties to even included. */
static int model_requant(int64_t acc, float scale)
{
    return requant_scale(acc, scale);
}

static int run(int fd, int ic, int oc, int ih, int iw, int k, int stride, int pad)
{
    rocket_conv2d_desc d = {0};
    int8_t *in = NULL, *W = NULL, *out = NULL;
    int32_t *bias = NULL;
    unsigned ow, oh, n_out, i;
    size_t nin, nw;
    int rc, exact = 0, touched = 0, o, y, x;
    const float scale = 1.0f / 256.0f;

    d.ic = ic; d.oc = oc; d.ih = ih; d.iw = iw;
    d.kh = k;  d.kw = k;
    d.stride_y = stride; d.stride_x = stride;
    d.pad_top = pad; d.pad_left = pad;
    d.dil_y = 1; d.dil_x = 1;
    ow = (unsigned)rocket_conv2d_ow(&d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    n_out = (unsigned)oc * ow * oh;
    nin = (size_t)ic * ih * iw;
    nw  = (size_t)oc * ic * k * k;

    in   = calloc(nin, 1);
    W    = calloc(nw, 1);
    out  = malloc(n_out);
    bias = calloc(oc, sizeof *bias);
    if (!in || !W || !out || !bias) { rc = 2; goto done; }
    fill_i8(in, nin, 1u);
    fill_i8(W, nw, 7u);
    /* Stamp, so "never written" is a property of the surface rather than a guess. */
    memset(out, STAMP, n_out);

    rc = rocket_conv2d_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f, 1.0f / scale,
                                   0, 0, 0, out);
    if (rc != ROCKET_OK) {
        printf("  ic=%d  REFUSED (%d)\n", ic, rc);
        rc = 1; goto done;
    }
    for (i = 0; i < n_out; i++) if ((unsigned char)out[i] != STAMP) { touched = 1; break; }

    for (o = 0; o < oc; o++)
        for (y = 0; y < (int)oh; y++)
            for (x = 0; x < (int)ow; x++) {
                int64_t acc = 0;
                int c, ky, kx, want, got;
                for (c = 0; c < ic; c++)
                    for (ky = 0; ky < k; ky++)
                        for (kx = 0; kx < k; kx++) {
                            int sy = y * stride + ky - pad;
                            int sx = x * stride + kx - pad;
                            if (sy < 0 || sy >= ih || sx < 0 || sx >= iw) continue;
                            acc += (int64_t)in[((size_t)c * ih + sy) * iw + sx] *
                                   W[(((size_t)o * ic + c) * k + ky) * k + kx];
                        }
                want = model_requant(acc, scale);
                got  = out[((size_t)o * oh + y) * ow + x];
                if (want == got) exact++;
            }

    printf("  ic=%d  %s  %d/%u exact  (oc=%d %dx%d k%d s%d pad%d -> %ux%u)\n",
           ic, touched ? (exact == (int)n_out ? "EXACT " : "WROTE ") : "NOTHING",
           exact, n_out, oc, ih, iw, k, stride, pad, ow, oh);
    rc = (exact == (int)n_out) ? 0 : 1;

done:
    free(in); free(W); free(out); free(bias);
    return rc;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, oc = 32, ih = 64, iw = 64, k = 3, stride = 2, pad = 1, fails = 0;
    int only = argc > 1 ? atoi(argv[1]) : 0;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_argb_ic1: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    if (getenv("ROCKET_A1_OC")) oc = atoi(getenv("ROCKET_A1_OC"));
    if (getenv("ROCKET_A1_K"))  k  = atoi(getenv("ROCKET_A1_K"));
    if (getenv("ROCKET_A1_N"))  ih = iw = atoi(getenv("ROCKET_A1_N"));

    fd = rocket_open();
    if (fd < 0) { printf("rk3576_argb_ic1: no NPU device — skipping\n"); return 2; }

    printf("== RK3576 int8 first conv, one image channel ==\n");
    if (only == 0 || only == 1) fails += run(fd, 1, oc, ih, iw, k, stride, pad);
    if (only == 0 || only == 2) fails += run(fd, 2, oc, ih, iw, k, stride, pad);
    /* The plane the vision stem actually opens with, at both channel counts. */
    if (only == 0) {
        fails += run(fd, 1, 32, 224, 224, 3, 2, 1);
        fails += run(fd, 2, 32, 224, 224, 3, 2, 1);
    }
    printf("== %s ==\n", fails ? "FAILED" : "one image channel is bit-exact");

    rocket_close(fd);
    return fails ? 1 : 0;
}
