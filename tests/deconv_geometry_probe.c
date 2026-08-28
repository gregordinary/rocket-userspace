// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * deconv_geometry_probe.c — what does the CNA's DECONV mode actually COMPUTE?
 *
 * WHAT IS ALREADY KNOWN. `deconv_mode_probe` established that CONV_CON1 bit 16 is live
 * on the RK3588: with it set, CONV_CON3's DECONV_X/Y_STRIDE fields 1, 3 and 7 each change
 * the surface reproducibly, every other value is inert, and the surfaces sparsify as the
 * field grows. What it could NOT establish is the part that matters — it compared two
 * surfaces and counted differing elements, and a count says nothing about WHERE the mode
 * puts its result, what the stride field encodes, or which way round the kernel is read.
 *
 * WHY A COUNT CANNOT ANSWER IT. Both candidate readings of the field predict "many
 * elements changed" and "more zeros as the field grows". A diff cannot separate them.
 * The geometry only becomes readable if the input is chosen so the answer is written in
 * the output's ADDRESSES rather than in its values.
 *
 * THE INSTRUMENT: AN IMPULSE. Drive one non-zero input element (channel 0, position
 * (r,c), value 1) through one non-zero filter (oc 0, ic 0, the 3x3 values 1..9 in raster
 * order). Everything else is zero. Then output channel 0 holds exactly one copy of the
 * kernel, and its POSITION and ORIENTATION are the measurement:
 *
 *   forward conv (the control): out[y][x] = W[r-y][c-x], so the kernel appears
 *     180-degree FLIPPED with its box ending at (r,c) — the correlation-vs-convolution
 *     flip, and a check that the instrument reads what it should.
 *   transposed conv at stride s: out[r*s+ky][c*s+kx] = W[ky][kx], so the kernel appears
 *     UNFLIPPED with its box STARTING at (r*s, c*s).
 *
 * Those two are distinguishable by position and by orientation independently, which is
 * what makes this a discriminator rather than a fit. And sweeping r over two values
 * measures the stride directly: the response's box moves by exactly s per unit of r. That
 * is the number the field encodes, read off rather than inferred — if field 1 moves the
 * box by 2, the field is s-1 and this is a power-of-two-only stride.
 *
 * WHY THE CANVAS IS BIG AND UNPADDED. The output geometry registers are driven from the
 * forward arithmetic, so a transposed result needs a surface larger than the forward
 * shape describes. The previous probe's k3/s1/pad0 8x8 left only 6x6 to land in. Rather
 * than pad (the RK3588 fp16 path refuses an output extent it did not derive, and a pad
 * wider than the kernel is its own question), this uses a LARGE input with a small
 * impulse: only one element is non-zero, so the response is local, and the 22x22 canvas a
 * 24x24 input gives is room for a stride-8 scatter to land in whole.
 *
 * WHAT THIS STILL DOES NOT ASK. Whether the mode is FASTER, and whether a full-density
 * input composes correctly. An impulse cannot see an accumulation bug across overlapping
 * scatters. This decodes the geometry and the kernel orientation, and those are the two
 * things an encoder cannot be written without.
 *
 * Run with `sudo -E` on an RK3588. Exits 2 (skip) elsewhere.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"

#define IC   32
#define OC   32
#define IWH  24        /* input is large so the response has room to land */
#define KS   3

/* Impulse rows/cols to sweep. Both are >= KS-1 so the FORWARD response is complete
 * rather than clipped at the corner, which keeps the control readable. */
static const int IMPULSE[] = { 2, 3 };
#define N_IMPULSE ((int)(sizeof IMPULSE / sizeof IMPULSE[0]))

struct bbox { int ymin, ymax, xmin, xmax, n; };

/* The bounding box of the non-zero elements of output channel 0, and how many there
 * are. This IS the measurement: a position and an extent, not a difference. */
static struct bbox plane_bbox(const _Float16 *out, int oh, int ow)
{
    struct bbox b = { 1 << 30, -1, 1 << 30, -1, 0 };
    int y, x;
    for (y = 0; y < oh; y++)
        for (x = 0; x < ow; x++)
            if (out[(size_t)y * ow + x] != (_Float16)0) {
                b.n++;
                if (y < b.ymin) b.ymin = y;
                if (y > b.ymax) b.ymax = y;
                if (x < b.xmin) b.xmin = x;
                if (x > b.xmax) b.xmax = x;
            }
    if (!b.n) { b.ymin = b.xmin = -1; }
    return b;
}

static void print_bbox_values(const _Float16 *out, int ow, const struct bbox *b)
{
    int y, x;
    if (b->n <= 0) { printf("        (empty)\n"); return; }
    if ((b->ymax - b->ymin) > 7 || (b->xmax - b->xmin) > 7) {
        printf("        (box %dx%d is too large to print)\n",
               b->ymax - b->ymin + 1, b->xmax - b->xmin + 1);
        return;
    }
    for (y = b->ymin; y <= b->ymax; y++) {
        printf("        ");
        for (x = b->xmin; x <= b->xmax; x++)
            printf("%6g", (double)out[(size_t)y * ow + x]);
        printf("\n");
    }
}

/* Is the box an UNFLIPPED copy of the 1..9 kernel (a scatter), a FLIPPED one (a
 * correlation), or neither? Returns "scatter", "flipped" or "other". */
static const char *orientation(const _Float16 *out, int ow, const struct bbox *b)
{
    int y, x, fwd = 1, rev = 1;
    if (b->n != KS * KS) return "partial-or-other";
    if (b->ymax - b->ymin != KS - 1 || b->xmax - b->xmin != KS - 1) return "other";
    for (y = 0; y < KS; y++)
        for (x = 0; x < KS; x++) {
            double v = (double)out[(size_t)(b->ymin + y) * ow + (b->xmin + x)];
            if (v != (double)(1 + y * KS + x))                 fwd = 0;
            if (v != (double)(1 + (KS-1-y) * KS + (KS-1-x)))   rev = 0;
        }
    if (fwd) return "UNFLIPPED (a scatter — transposed-conv orientation)";
    if (rev) return "flipped (correlation orientation — same as the forward conv)";
    return "other";
}

static int run_once(int fd, const rocket_conv2d_desc *d, const _Float16 *in,
                    const _Float16 *W, _Float16 *out, size_t n_out,
                    int bit, int f)
{
    char b[8];
    memset(out, 0, n_out * sizeof *out);
    snprintf(b, sizeof b, "%d", bit); setenv("ROCKET_CNA_DECONV",   b, 1);
    snprintf(b, sizeof b, "%d", f);   setenv("ROCKET_CNA_DECONV_X", b, 1);
    snprintf(b, sizeof b, "%d", f);   setenv("ROCKET_CNA_DECONV_Y", b, 1);
    return rocket_conv2d_fp16(fd, d, in, W, out);
}

int main(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    rocket_conv2d_desc d;
    _Float16 *in, *W, *out, *again;
    size_t n_in, n_w, n_out;
    int fd, oh, ow, i, rc = 0;
    /* The fields deconv_mode_probe found live, plus 0 as the in-mode control. */
    const int FIELDS[] = { 0, 1, 3, 7 };
    const int NF = (int)(sizeof FIELDS / sizeof FIELDS[0]);

    if (strcmp(hw->name, "rk3588") != 0) {
        printf("deconv_geometry_probe: profile is %s — this drives the RK3588 conv "
               "generator; skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("deconv_geometry_probe: no NPU device — skipping\n"); return 2; }

    memset(&d, 0, sizeof d);
    d.ic = IC; d.ih = IWH; d.iw = IWH; d.oc = OC;
    d.kh = KS; d.kw = KS; d.stride_y = 1; d.stride_x = 1;
    d.pad_top = 0; d.pad_left = 0; d.dil_y = 1; d.dil_x = 1;
    if (rocket_conv2d_plan(&d) < 0) {
        printf("deconv_geometry_probe: the probe shape does not plan — skipping\n");
        rocket_close(fd); return 2;
    }
    oh = rocket_conv2d_oh(&d); ow = rocket_conv2d_ow(&d);
    n_in  = (size_t)IC * IWH * IWH;
    n_w   = (size_t)OC * IC * KS * KS;
    n_out = (size_t)OC * oh * ow;

    in    = calloc(n_in,  sizeof *in);
    W     = calloc(n_w,   sizeof *W);
    out   = calloc(n_out, sizeof *out);
    again = calloc(n_out, sizeof *again);
    if (!in || !W || !out || !again) { rc = 1; goto done; }

    /* ONE non-zero filter: oc 0, ic 0, values 1..9 in raster order. Distinct and
     * asymmetric, so the orientation is legible from the values alone. W is
     * [OC][IC][KH][KW] on the forward path — whether the mode wants [IC][OC] instead is
     * one of the open questions, and a swap would show up as an EMPTY channel 0. */
    for (i = 0; i < KS * KS; i++) W[i] = (_Float16)(i + 1);

    printf("== CNA DECONV geometry, impulse response ==\n");
    printf("   in %dx%dx%d k%d s1 pad0 -> out %dx%dx%d; one filter (oc0,ic0) = 1..9\n",
           IC, IWH, IWH, KS, OC, oh, ow);
    printf("   a forward conv puts a FLIPPED kernel with its box ENDING at the impulse;\n"
           "   a transposed conv at stride s puts an UNFLIPPED one STARTING at (r*s,c*s).\n");

    for (i = 0; i < N_IMPULSE; i++) {
        int r = IMPULSE[i], c = IMPULSE[i], j;
        memset(in, 0, n_in * sizeof *in);
        in[(size_t)r * IWH + c] = (_Float16)1;      /* channel 0 */
        printf("\n-- impulse at (%d,%d) --\n", r, c);

        for (j = -1; j < NF; j++) {
            int bit = (j < 0) ? 0 : 1;
            int f   = (j < 0) ? 0 : FIELDS[j];
            struct bbox b, b2;
            const char *tag = (j < 0) ? "bit CLEAR (forward control)" : NULL;
            char label[64];
            if (!tag) { snprintf(label, sizeof label, "bit SET, field %d", f); tag = label; }

            if (run_once(fd, &d, in, W, out, n_out, bit, f) != 0) {
                printf("   %-28s : the submit FAILED\n", tag); continue;
            }
            if (run_once(fd, &d, in, W, again, n_out, bit, f) != 0) {
                printf("   %-28s : the repeat FAILED\n", tag); continue;
            }
            b  = plane_bbox(out,   oh, ow);
            b2 = plane_bbox(again, oh, ow);
            if (memcmp(out, again, n_out * sizeof *out) != 0 || b.n != b2.n) {
                printf("   %-28s : NOT REPRODUCIBLE across two runs — not a result\n", tag);
                continue;
            }
            if (b.n == 0) {
                printf("   %-28s : channel 0 is EMPTY (nothing scattered here)\n", tag);
                continue;
            }
            printf("   %-28s : %d nz, box y[%d..%d] x[%d..%d], %s\n",
                   tag, b.n, b.ymin, b.ymax, b.xmin, b.xmax,
                   orientation(out, ow, &b));
            print_bbox_values(out, ow, &b);
        }
    }

    printf("\n   READ IT AS: box origin against the impulse gives the stride. If the box\n"
           "   for field F starts at (r*s, c*s) with s = F+1, the field is s-1 and only\n"
           "   the values 1/3/7 (s = 2/4/8) are implemented. If channel 0 is empty under\n"
           "   the mode, the weights are wanted in [IC][OC] order instead.\n");

    unsetenv("ROCKET_CNA_DECONV");
    unsetenv("ROCKET_CNA_DECONV_X");
    unsetenv("ROCKET_CNA_DECONV_Y");
done:
    free(in); free(W); free(out); free(again);
    rocket_close(fd);
    return rc;
}
