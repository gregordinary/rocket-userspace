// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_argb_border_cost.c — what the two HOST routes past the packed stem's column
 * bound cost, on the shape that would use them.
 *
 * WHY THIS EXISTS. The packed-image first conv's pad COLUMNS feed the MAC a constant 0
 * at both ends and at every row, while its pad ROWS honour `CNA_PAD_CON1`
 * [HW sweep, tests/rk3576_argb_padmap.c]. At an input zero point of 0 that is the right
 * answer and the encoding is exact; at any other zero point every output whose window
 * reaches a pad column is wrong, which is why the entry refuses one — and which is what
 * keeps ResNet-18's `ic=3 224x224 k7 s2 pad 3` stem, `in_zp = -128`, on the direct
 * lowering where the packed program is worth about 2 ms.
 *
 * The error is `-in_zp * (sum of the weights over the taps in a pad column)` — DATA
 * INDEPENDENT, a constant per output channel and output column. That does not make it
 * free to repair, because the correction lives in the ACCUMULATOR and the part returns a
 * requantized int8: fixing an affected output means recomputing it. So there are two host
 * routes and this file prices both on the real shape, before any cap is quoted.
 *
 *   A. RECOMPUTE the affected columns on the host, as the TFLite-shift stem already does
 *      for its own border. The affected set is every output column whose window reaches
 *      outside the plane on the X axis.
 *
 *   B. MATERIALISE the pad columns as real image columns and discard the outputs that
 *      read the hardware's synthesized one. With a leading extension of `L = P - pad + 2d`
 *      columns at the input zero point and the program given `pad_left = 1`, output column
 *      `x + d` reads exactly the window output column `x` wants; only `x' = 0` reads the
 *      synthesized column, and it is discarded. The geometry bounds (`ow == iw/stride`,
 *      `ow` a multiple of 16) then force a WIDER output than the model asks for, so the
 *      surface must be CROPPED back — a strided pass over the whole output in cube layout,
 *      which is what this arm times.
 *
 * Neither arm needs the NPU: both are host work, and host work is what decides whether
 * the encoding's 8x MAC saving survives. The governor matters — pin `performance` before
 * reading these and restore `ondemand` after.
 *
 * Usage: rk3576_argb_border_cost [reps]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int8_t sat8(int32_t v) { return (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v)); }

/* One output element of an int8 convolution with a zero-point border, requantized the way
 * the part's OUT_CVT does (multiply, arithmetic shift, saturate). This is the same shape
 * of loop the gate's own host_conv_rect() runs. */
static int8_t one_out(const int8_t *in, const int8_t *W, unsigned ic, unsigned ih,
                      unsigned iw, unsigned kh, unsigned kw, unsigned s,
                      unsigned pad_y, unsigned pad_x, int in_zp, int w_zp,
                      unsigned oc_i, unsigned y, unsigned x, int32_t mul, int shift)
{
    int32_t acc = 0;
    unsigned c, ky, kx;
    for (c = 0; c < ic; c++)
        for (ky = 0; ky < kh; ky++) {
            long iy = (long)y * s + ky - pad_y;
            for (kx = 0; kx < kw; kx++) {
                long ix = (long)x * s + kx - pad_x;
                int v = (iy < 0 || iy >= (long)ih || ix < 0 || ix >= (long)iw)
                        ? in_zp
                        : in[((size_t)c * ih + iy) * iw + ix];
                int w = W[((size_t)oc_i * ic + c) * kh * kw + ky * kw + kx];
                acc += (v - in_zp) * (w - w_zp);
            }
        }
    return sat8((int32_t)(((int64_t)acc * mul) >> shift));
}

/* Which output columns reach outside the plane on the X axis — the set route A must
 * recompute, and the set route B's extension removes. */
static unsigned affected_cols(unsigned ow, unsigned s, unsigned kw, unsigned pad_x,
                              unsigned iw, unsigned *list, unsigned max)
{
    unsigned n = 0, x;
    for (x = 0; x < ow; x++) {
        long lo = (long)x * s - (long)pad_x, hi = lo + (long)kw - 1;
        if ((lo < 0 || hi >= (long)iw) && n < max) list[n++] = x;
    }
    return n;
}

struct shape {
    const char *name;
    unsigned ic, oc, ih, iw, k, s, pad;
};

static const struct shape SHAPES[] = {
    { "ResNet-18 stem  k7 s2", 3, 64, 224, 224, 7, 2, 3 },
    { "MobileNetV1 stem k3 s2", 3, 32, 224, 224, 3, 2, 1 },
    { "Inception V1 stem k7 s2", 3, 64, 224, 224, 7, 2, 2 },
};
#define NSHAPE (sizeof SHAPES / sizeof *SHAPES)

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 20;
    unsigned si, i;

    printf("RK3576 packed stem: what the two HOST routes past the column bound cost\n");
    printf("%d timed repeats after a discarded first. Pin the governor before reading\n"
           "these — both arms are pure host work.\n\n", reps);

    printf("  %-24s %8s %7s %9s %10s %10s\n",
           "shape", "aff.cols", "outputs", "taps", "recompute", "ms/1k out");
    printf("  %-24s %8s %7s %9s %10s %10s\n",
           "------------------------", "--------", "-------", "---------",
           "----------", "----------");

    for (si = 0; si < NSHAPE; si++) {
        const struct shape *sh = &SHAPES[si];
        unsigned ow = (sh->iw + 2 * sh->pad - sh->k) / sh->s + 1;
        unsigned oh = ow;
        unsigned cols[64], ncol;
        int8_t *in, *W, *out;
        double t0, ms = 0.0;
        size_t nout;
        int r;

        ncol = affected_cols(ow, sh->s, sh->k, sh->pad, sh->iw, cols, 64);
        nout = (size_t)ncol * oh * sh->oc;

        in  = malloc((size_t)sh->ic * sh->ih * sh->iw);
        W   = malloc((size_t)sh->oc * sh->ic * sh->k * sh->k);
        out = malloc(nout ? nout : 1);
        if (!in || !W || !out) { free(in); free(W); free(out); continue; }
        for (i = 0; i < sh->ic * sh->ih * sh->iw; i++) in[i] = (int8_t)((i * 7u) & 0x7F);
        for (i = 0; i < sh->oc * sh->ic * sh->k * sh->k; i++) W[i] = (int8_t)((i % 13) - 6);

        for (r = 0; r < reps + 1; r++) {
            unsigned o, y, ci;
            size_t p = 0;
            if (r == 1) ms = 0.0;
            t0 = now_ms();
            for (o = 0; o < sh->oc; o++)
                for (ci = 0; ci < ncol; ci++)
                    for (y = 0; y < oh; y++)
                        out[p++] = one_out(in, W, sh->ic, sh->ih, sh->iw, sh->k, sh->k,
                                           sh->s, sh->pad, sh->pad, -128, 0, o, y,
                                           cols[ci], 1 << 14, 20);
            ms += now_ms() - t0;
        }
        ms /= reps;

        printf("  %-24s %8u %7zu %9u %9.3f %10.4f\n",
               sh->name, ncol, nout, sh->ic * sh->k * sh->k, ms,
               nout ? ms * 1000.0 / (double)nout : 0.0);
        fflush(stdout);
        free(in); free(W); free(out);
    }

    /* --- ROUTE B's host term: cropping a wider output surface back to the model's own
     * width, in the cube layout the next layer reads. The surface is 16-byte atoms
     * indexed by (channel group, pixel), so a column crop is a strided copy of
     * `oh * ow` atoms per group. Timed against a flat memcpy of the same bytes, which
     * is the floor a perfectly contiguous crop could reach. */
    {
        unsigned oc = 64, oh = 112, ow_w = 128, ow = 112, g, y;
        size_t atoms_src = (size_t)(oc / 16) * oh * ow_w, atoms_dst = (size_t)(oc / 16) * oh * ow;
        int8_t *src = malloc(atoms_src * 16), *dst = malloc(atoms_dst * 16);
        double t0, ms = 0.0, msc = 0.0;
        int r;

        if (src && dst) {
            memset(src, 1, atoms_src * 16);
            for (r = 0; r < reps + 1; r++) {
                if (r == 1) ms = 0.0;
                t0 = now_ms();
                for (g = 0; g < oc / 16; g++)
                    for (y = 0; y < oh; y++)
                        memcpy(dst + ((size_t)g * oh + y) * ow * 16,
                               src + ((size_t)g * oh + y) * ow_w * 16,
                               (size_t)ow * 16);
                ms += now_ms() - t0;
            }
            ms /= reps;
            for (r = 0; r < reps + 1; r++) {
                if (r == 1) msc = 0.0;
                t0 = now_ms();
                memcpy(dst, src, atoms_dst * 16);
                msc += now_ms() - t0;
            }
            msc /= reps;
            printf("\n  ROUTE B host term — crop a %u-wide surface to %u, oc=%u oh=%u,\n"
                   "  cube layout (%zu KiB out): %.3f ms strided, %.3f ms flat memcpy floor\n",
                   ow_w, ow, oc, oh, atoms_dst * 16 / 1024, ms, msc);
        }
        free(src); free(dst);
    }

    printf("\n  `recompute` is route A on that shape. Compare it against the packed\n"
           "  encoding's own saving on the same stem before quoting any cap: a repair\n"
           "  that costs more than the encoding saves is not a lever.\n");
    return 0;
}
