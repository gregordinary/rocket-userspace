// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_perchannel_gate.c — rocket_conv2d_int8_perchannel_rk3576(), two ways.
 *
 * Per-axis weight scales are the form every real TFLite int8 and ONNX QDQ model
 * carries, and this part expresses them through the coefficient group's int16 C
 * multiplier against ONE OUT_CVT shift. That is close to a per-axis requant and not
 * equal to it, so the gate has two halves and they answer different questions.
 *
 *   BIT-EXACT — against a CPU model of what the CHIP computes:
 *       out = sat8( rhe( sat32((acc + A[oc]) * C[oc]) * MUL >> SHIFT ) + out_zp )
 *   with the same (MUL, SHIFT) and the same C ramp the library planned. A failure here
 *   is a defect in the emitter, the packing or the planner. This is the gate.
 *
 *   ACCURACY — against an exact per-axis float reference, the thing a model actually
 *   wants. This is REPORTED, not asserted: how far the two sit apart is a property of
 *   the datapath (one shift for every channel, an integer C, and an int32 product that
 *   saturates), not of the implementation, and the number is what a caller needs to
 *   decide whether the layer belongs on the NPU at all.
 *
 * The scale SPREAD is the axis that matters, because it is what forces the C ramp
 * wide, so the table sweeps it explicitly rather than taking whatever a random weight
 * draw produces.
 *
 * DEPTHWISE shapes are in the table too, and they are the interesting half: a real
 * per-axis model is mostly depthwise, and this path is both cheaper and closer there.
 * A depthwise task's accumulator is bounded by kh*kw taps rather than ic*kh*kw, so the
 * C ramp reaches the int16 field's own ceiling and one OUT_CVT shift covers the whole
 * layer -- one task, no output-channel split, and a spread of 1000x still lands inside
 * a count of an exact per-axis requant.
 *
 * Usage:  rk3576_perchannel_gate [name-substring]
 * Exit:   0 every shape bit-exact, 1 a shape failed, 2 no NPU (skip).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_hw_profile.h"
#include "requant_model.h"

typedef struct {
    const char *name;
    unsigned ic, oc, iw, ih, k, stride, pad;
    float    spread;      /* w_scale[oc] spans [base, base*spread] */
    int      dw;          /* depthwise: oc == ic, one filter per channel */
} pc_shape;

static const pc_shape SHAPES[] = {
    /* name          ic   oc   iw  ih   k  s  pad  spread   dw */
    { "flat",        32,  32,  16, 16,  1, 1, 0,   1.0f,    0 },
    { "narrow",      32,  32,  16, 16,  3, 1, 1,   2.0f,    0 },
    { "decade",      32,  64,  16, 16,  3, 1, 1,   10.0f,   0 },
    { "wide",        32,  64,  16, 16,  3, 1, 1,   100.0f,  0 },
    { "k1-64",       64,  64,   8,  8,  1, 1, 0,   10.0f,   0 },
    { "s2",          64,  32,  16, 16,  3, 2, 1,   10.0f,   0 },
    { "deep",       128,  64,   8,  8,  3, 1, 1,   10.0f,   0 },
    { "deep-wide",  128, 128,   8,  8,  3, 1, 1,   100.0f,  0 },
    { "big-plane",   32,  32,  28, 28,  3, 1, 1,   10.0f,   0 },
    { "k5",          32,  32,  16, 16,  5, 1, 2,   10.0f,   0 },
    { "oc-tile",     64, 256,   8,  8,  3, 1, 1,   10.0f,   0 },
    { "huge-fanin", 256, 128,   8,  8,  3, 1, 1,   10.0f,   0 },
    /* DEPTHWISE. The 48-byte coefficient group carries the same per-channel C, and a
     * depthwise task's accumulator is bounded by kh*kw taps rather than ic*kh*kw — so
     * the C ramp reaches the int16 field's own ceiling and a single OUT_CVT shift is
     * enough for the whole layer. A depthwise conv is one task by construction here:
     * output channel c is bound to input channel c, so there is no output-channel
     * window to program and no scale sort to make. A MobileNet is mostly depthwise,
     * which is what a per-axis model needs from this path. */
    { "dw-flat",     32,  32,  16, 16,  3, 1, 1,   1.0f,    1 },
    { "dw-decade",   32,  32,  16, 16,  3, 1, 1,   10.0f,   1 },
    { "dw-wide",     64,  64,  16, 16,  3, 1, 1,   100.0f,  1 },
    { "dw-k1",       32,  32,  16, 16,  1, 1, 0,   10.0f,   1 },
    { "dw-s2",       64,  64,  16, 16,  3, 2, 1,   10.0f,   1 },
    { "dw-k5",       32,  32,  16, 16,  5, 1, 2,   10.0f,   1 },
    { "dw-deep",    256, 256,   8,  8,  3, 1, 1,   100.0f,  1 },
    { "dw-1000x",    64,  64,  14, 14,  3, 1, 1,   1000.0f, 1 },
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof *SHAPES))

static void sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

/*
 * The library's own C-ramp planner, spelled again here so the model predicts the same
 * registers. It is deliberately a SECOND implementation from the documented rule rather
 * than a call into the first: a model that shares the planner cannot catch the planner.
 */
static void plan_c(unsigned oc0, unsigned tile_oc, const unsigned *perm,
                   const int64_t *sum_abs_w, const int32_t *A,
                   float in_scale, const float *w_scale, float out_scale,
                   int16_t *C, unsigned *mul, unsigned *shift)
{
    double best = 0.0, base;
    unsigned j;
    for (j = 0; j < tile_oc; j++) {
        unsigned c = perm[oc0 + j];
        double cs = (double)in_scale * (double)w_scale[c] / (double)out_scale;
        double bound = 128.0 * (double)sum_abs_w[c] + fabs((double)A[oc0 + j]) + 1.0;
        double cmax = (double)INT32_MAX / bound;
        double need;
        if (cmax > 32767.0) cmax = 32767.0;
        if (cmax < 1.0)     cmax = 1.0;
        need = cs / cmax;
        if (need > best) best = need;
    }
    rocket_rk3576_requant_params((float)best, mul, shift);
    base = (double)*mul / (double)((uint64_t)1 << *shift);
    for (j = 0; j < tile_oc; j++) {
        unsigned c = perm[oc0 + j];
        double cs = (double)in_scale * (double)w_scale[c] / (double)out_scale;
        long long v = (long long)(cs / base + 0.5);
        if (v < 1)     v = 1;
        if (v > 32767) v = 32767;
        C[oc0 + j] = (int16_t)v;
    }
}

/* The library sorts the output channels by scale so each tile spans as little of the
 * layer's scale range as the tiling allows. Bookkeeping, not the arithmetic under
 * test — mirrored here so the model reads the same slot the part did. */
static void sort_by_scale(unsigned *perm, unsigned oc, const float *w_scale)
{
    unsigned i, j;
    for (i = 0; i < oc; i++) perm[i] = i;
    for (i = 1; i < oc; i++) {
        unsigned v = perm[i];
        float s = w_scale[v];
        j = i;
        while (j > 0 && w_scale[perm[j - 1]] > s) { perm[j] = perm[j - 1]; j--; }
        perm[j] = v;
    }
}

static int32_t sat32(int64_t v)
{
    if (v >  (int64_t)INT32_MAX) return INT32_MAX;
    if (v <  (int64_t)INT32_MIN) return INT32_MIN;
    return (int32_t)v;
}

static int run_shape(int fd, const pc_shape *s, int verbose)
{
    unsigned ic = s->ic, oc = s->oc, iw = s->iw, ih = s->ih, k = s->k;
    unsigned st = s->stride, pad = s->pad;
    unsigned ow = (iw + 2 * pad - k) / st + 1;
    unsigned oh = (ih + 2 * pad - k) / st + 1;
    int dw = s->dw;
    size_t in_n = (size_t)ic * ih * iw;
    size_t w_n = dw ? (size_t)oc * k * k : (size_t)oc * ic * k * k;
    size_t out_n = (size_t)oc * oh * ow;
    int8_t  *in = malloc(in_n), *W = malloc(w_n), *got = malloc(out_n);
    int8_t  *want = malloc(out_n);
    int32_t *bias = malloc(oc * sizeof *bias), *A = malloc(oc * sizeof *A);
    int64_t *sum_w = malloc(oc * sizeof *sum_w);
    int64_t *sum_abs_w = malloc(oc * sizeof *sum_abs_w);
    float   *w_scale = malloc(oc * sizeof *w_scale);
    int16_t *C = malloc(oc * sizeof *C);
    int32_t *Aslot = malloc(oc * sizeof *Aslot);
    unsigned *perm = malloc(oc * sizeof *perm);
    unsigned *slot_of = malloc(oc * sizeof *slot_of);
    unsigned *tile_mul = malloc(oc * sizeof *tile_mul);
    unsigned *tile_shift = malloc(oc * sizeof *tile_shift);
    rocket_conv2d_desc d = {0};
    const float in_scale = 0.021f, out_scale = 0.037f;
    const int in_zp = -7, out_zp = 11;
    unsigned mul, shift, c, i, y, x, ky, kx;
    unsigned oc_tile = 0;
    unsigned seed = 0x2545F491u ^ (ic * 31 + oc * 17 + iw * 7 + k);
    int exact = 0, worst = 0, rc, fail = 0;
    double ref_worst = 0.0, ref_sum = 0.0;

    if (!in || !W || !got || !want || !bias || !A || !sum_w || !sum_abs_w ||
        !w_scale || !C || !Aslot || !perm || !slot_of || !tile_mul || !tile_shift) {
        fail = 1; goto done;
    }

    for (i = 0; i < in_n; i++) {
        seed = seed * 1103515245u + 12345u;
        in[i] = (int8_t)((int)((seed >> 16) % 255u) - 127);
    }
    for (i = 0; i < w_n; i++) {
        seed = seed * 1103515245u + 12345u;
        W[i] = (int8_t)((int)((seed >> 16) % 255u) - 127);
    }
    for (c = 0; c < oc; c++) {
        seed = seed * 1103515245u + 12345u;
        bias[c] = (int32_t)((int)((seed >> 16) % 2001u) - 1000);
        /* A geometric ramp across the spread: a per-axis model's scales are the
         * per-channel weight maxima, which do not vary linearly. */
        w_scale[c] = (float)(0.0009 *
            pow((double)s->spread, oc > 1 ? (double)c / (double)(oc - 1) : 0.0));
    }

    d.ic = (int)ic; d.oc = (int)oc;
    d.ih = (int)ih; d.iw = (int)iw;
    d.kh = (int)k;  d.kw = (int)k;
    d.stride_y = (int)st; d.stride_x = (int)st;
    d.pad_top = (int)pad; d.pad_left = (int)pad;
    d.dil_y = 1; d.dil_x = 1;
    d.depthwise = dw;

    /* The tile is bookkeeping — one task, one OUT_CVT shift — not the arithmetic
     * under test, so the model asks the library where it split rather than
     * re-deriving a CBUF rule. Everything downstream of it is modelled independently. */
    oc_tile = rocket_conv2d_int8_perchannel_oc_tile_rk3576(&d, W, bias, in_scale,
                                                           w_scale, out_scale, in_zp);
    if (!oc_tile) {
        printf("  %-11s ic=%-4u oc=%-4u  the entry refuses this descriptor\n",
               s->name, ic, oc);
        fail = 1; goto done;
    }
    rc = rocket_conv2d_int8_perchannel_rk3576(fd, &d, in, W, bias, in_scale,
                                              w_scale, out_scale, in_zp, out_zp, got);
    if (rc != ROCKET_OK) {
        printf("  %-11s ic=%-4u oc=%-4u %2ux%-2u k%u s%u  returned %d\n",
               s->name, ic, oc, iw, ih, k, st, rc);
        fail = 1; goto done;
    }

    /* The model. Same folds the library does, then the same planner, then the DPU's
     * epilogue in the order the part was measured to apply it. */
    for (c = 0; c < oc; c++) {
        int64_t sw = 0, sa = 0;
        if (dw) {
            for (ky = 0; ky < k; ky++)
                for (kx = 0; kx < k; kx++) {
                    int64_t v = W[((size_t)c * k + ky) * k + kx];
                    sw += v; sa += v < 0 ? -v : v;
                }
        } else {
            for (i = 0; i < ic; i++)
                for (ky = 0; ky < k; ky++)
                    for (kx = 0; kx < k; kx++) {
                        int64_t v = W[(((size_t)c * ic + i) * k + ky) * k + kx];
                        sw += v; sa += v < 0 ? -v : v;
                    }
        }
        sum_w[c] = sw; sum_abs_w[c] = sa;
        A[c] = (int32_t)((int64_t)bias[c] - (int64_t)in_zp * sw);
    }
    /* Depthwise binds output channel c to INPUT channel c, so the library cannot
     * reorder the output channels and does not try: the model reads the coefficient
     * slots in natural order there. */
    if (dw) { for (i = 0; i < oc; i++) perm[i] = i; }
    else    sort_by_scale(perm, oc, w_scale);
    /* A and C are per SLOT, in the permuted order the library packs the coefficient
     * group in, so the model builds both that way and maps back through slot_of. */
    for (i = 0; i < oc; i++) {
        unsigned cc = perm[i];
        Aslot[i] = (int32_t)((int64_t)bias[cc] - (int64_t)in_zp * sum_w[cc]);
        slot_of[cc] = i;
    }
    for (i = 0; i < oc; i += oc_tile) {
        unsigned n = oc - i < oc_tile ? oc - i : oc_tile, j;
        plan_c(i, n, perm, sum_abs_w, Aslot, in_scale, w_scale, out_scale,
               C, &mul, &shift);
        /* one OUT_CVT pair per tile, recorded against the channels it covers */
        for (j = 0; j < n; j++) {
            tile_mul[perm[i + j]]   = mul;
            tile_shift[perm[i + j]] = shift;
        }
    }

    for (c = 0; c < oc; c++) {
        for (y = 0; y < oh; y++) {
            for (x = 0; x < ow; x++) {
                int64_t acc = 0;
                double  ref;
                int32_t v;
                int m;
                for (i = dw ? c : 0; i < (dw ? c + 1 : ic); i++)
                    for (ky = 0; ky < k; ky++)
                        for (kx = 0; kx < k; kx++) {
                            long iy = (long)(y * st + ky) - (long)pad;
                            long ix = (long)(x * st + kx) - (long)pad;
                            int8_t fv;
                            if (iy < 0 || ix < 0 || iy >= (long)ih || ix >= (long)iw)
                                fv = (int8_t)in_zp;      /* the emitter's border constant */
                            else
                                fv = in[((size_t)i * ih + (size_t)iy) * iw + (size_t)ix];
                            acc += (int64_t)fv *
                                   (dw ? W[((size_t)c * k + ky) * k + kx]
                                       : W[(((size_t)c * ic + i) * k + ky) * k + kx]);
                        }
                acc -= (int64_t)in_zp * sum_w[c];
                acc += bias[c];
                v = sat32(acc * (int64_t)C[slot_of[c]]);
                m = requant_sat8(requant_round_shift((int64_t)v * (int64_t)tile_mul[c],
                                                     tile_shift[c]) + out_zp);
                want[((size_t)c * oh + y) * ow + x] = (int8_t)m;

                /* and what an exact per-axis requant would have produced */
                ref = (double)acc * (double)in_scale * (double)w_scale[c] /
                      (double)out_scale + (double)out_zp;
                if (ref >  127.0) ref =  127.0;
                if (ref < -128.0) ref = -128.0;
                {
                    double e = fabs(ref - (double)got[((size_t)c * oh + y) * ow + x]);
                    if (e > ref_worst) ref_worst = e;
                    ref_sum += e;
                }
            }
        }
    }

    for (i = 0; i < out_n; i++) {
        int diff = got[i] - want[i];
        if (diff < 0) diff = -diff;
        if (!diff) exact++;
        else if (diff > worst) worst = diff;
    }
    if (exact != (int)out_n) fail = 1;

    printf("  %-4s %-11s ic=%-4u oc=%-4u %2ux%-2u k%u s%u  %7d/%-7zu  "
           "C[%d..%d] tile=%-4u vs exact per-axis: max %.1f mean %.2f\n",
           fail ? "FAIL" : "PASS", s->name, ic, oc, iw, ih, k, st,
           exact, out_n, C[0], C[oc - 1], oc_tile,
           ref_worst, ref_sum / (double)out_n);
    if (fail && verbose) {
        int shown = 0;
        for (i = 0; i < out_n && shown < 8; i++)
            if (got[i] != want[i]) {
                printf("      [%zu] chip model %d, part %d\n",
                       (size_t)i, want[i], got[i]);
                shown++;
            }
    }
done:
    free(in); free(W); free(got); free(want); free(bias); free(A);
    free(sum_w); free(sum_abs_w); free(w_scale); free(C);
    free(Aslot); free(perm); free(slot_of); free(tile_mul); free(tile_shift);
    return fail;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, i, fails = 0, ran = 0;
    int verbose = getenv("ROCKET_PC_VERBOSE") != NULL;
    int gap = getenv("ROCKET_PC_GAP_MS") ? atoi(getenv("ROCKET_PC_GAP_MS")) : 0;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_perchannel_gate: profile is %s, not rk3576 — skipping\n",
               hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_perchannel_gate: no NPU device — skipping\n"); return 2; }

    printf("== RK3576 per-output-channel int8 conv ==\n");
    printf("   bit-exact against the CHIP's arithmetic; the distance to an exact "
           "per-axis requant is reported, not gated\n");
    for (i = 0; i < N_SHAPES; i++) {
        if (argc > 1 && !strstr(SHAPES[i].name, argv[1])) continue;
        sleep_ms(gap);
        fails += run_shape(fd, &SHAPES[i], verbose);
        ran++;
    }
    rocket_close(fd);
    printf("== %d passed, %d failed ==\n", ran - fails, fails);
    return fails ? 1 : 0;
}
