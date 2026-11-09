// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv1d_gelu_epilogue_probe.c -- is the single-pass conv->GELU epilogue usable on a real
 * Whisper front end?
 *
 * Whisper's encoder opens with two convs over time (KW 3, pad 1; conv1 at stride 1 from the
 * mel bins, conv2 at stride 2), each followed by a bias and GELU. The fused entry,
 * rocket_conv2d_act_fp16(GELU), applies GELU as one DPU LUT in the conv's own job. That LUT
 * spikes near x = 0 and in a flat tail, and a dense random surface hits the band near 0 a
 * few times. This probe runs the real thing: a model's front-end weights on a real log-mel,
 * from tools/whisper_frontend_dump.py.
 *
 *     conv1d_gelu_epilogue_probe DIR      (DIR holds mel.f16, w1.f16, b1.f32, w2.f16,
 *                                          b2.f32 and meta.txt)
 *
 * The conv descriptor has no bias, so each bias rides in as one more input channel held at
 * 1.0, whose weight is the bias at the center tap (a tap that never reads the zero pad).
 * The channel count is padded to a multiple of 32 with zero channels.
 *
 * Each conv runs three ways on the NPU:
 *   x      rocket_conv1d_fp16, the conv alone
 *   two    rocket_activation_fp16(GELU) on x, the two-pass x * Phi(x)
 *   fused  rocket_conv2d_act_fp16(GELU) on the same input, the single-pass epilogue
 * and both GELU outputs are scored against erf-GELU of x in double: cosine, max |error|,
 * counts past 0.05 and 1, and the same split by where x falls (within 0.0015 of 0, below
 * -3, elsewhere). conv2 reads the exact GELU of conv1's x rounded to fp16, so each conv's
 * epilogue is scored on its own.
 *
 * Exit 0 when the two-pass output is within 0.01 on both convs (the harness is sound; the
 * fused verdict is in the report), 1 otherwise, 2 without a device or data.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_activation.h"
#include "rocket_conv.h"
#include "rocket_npu.h"

static void *load(const char *dir, const char *name, size_t bytes)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    void *p = malloc(bytes);
    size_t got = p ? fread(p, 1, bytes, f) : 0;
    fclose(f);
    if (got != bytes) { fprintf(stderr, "%s: read %zu of %zu bytes\n", path, got, bytes); free(p); return NULL; }
    return p;
}

static double gelu(double x) { return 0.5 * x * (1.0 + erf(x / sqrt(2.0))); }

typedef struct {
    double dot, nr, ny, max_abs, max_band, max_tail, max_rest;
    long n05, n1, band, tail, n05_band, n05_tail, n05_rest;
} score_t;

static score_t score(const _Float16 *x, const _Float16 *y, size_t n)
{
    score_t s;
    memset(&s, 0, sizeof s);
    for (size_t i = 0; i < n; i++) {
        double xv = (double)x[i], r = gelu(xv), v = (double)y[i], e = fabs(v - r);
        s.dot += r * v; s.nr += r * r; s.ny += v * v;
        if (e > s.max_abs) s.max_abs = e;
        s.n05 += e > 0.05; s.n1 += e > 1.0;
        if (fabs(xv) < 0.0015) {
            s.band++; s.n05_band += e > 0.05; if (e > s.max_band) s.max_band = e;
        } else if (xv < -3.0) {
            s.tail++; s.n05_tail += e > 0.05; if (e > s.max_tail) s.max_tail = e;
        } else {
            s.n05_rest += e > 0.05; if (e > s.max_rest) s.max_rest = e;
        }
    }
    return s;
}

static void report(const char *conv, const char *path, score_t s, size_t n)
{
    printf("  %s %-5s cos %.6f  max_abs %.4f  >0.05 %ld  >1 %ld  of %zu\n", conv, path,
           s.dot / sqrt(s.nr * s.ny + 1e-300), s.max_abs, s.n05, s.n1, n);
    printf("            |x|<0.0015: %ld elems, %ld >0.05, max %.4f   x<-3: %ld elems, %ld >0.05, "
           "max %.4f   rest: %ld >0.05, max %.4f\n", s.band, s.n05_band, s.max_band, s.tail,
           s.n05_tail, s.max_tail, s.n05_rest, s.max_rest);
}

/* One conv three ways. `in` is [icp][it] with the bias channel at `ic`; returns the
 * two-pass max |error|, or -1 on a device error. `gelu_out`, if non-NULL, receives the
 * exact GELU of x rounded to fp16 (the next conv's input). */
static double run_conv(int fd, const char *name, int icp, int ic, int it, int oc, int stride,
                       const _Float16 *in, const _Float16 *W, _Float16 *gelu_out)
{
    const int ot = rocket_conv_out_dim(it, 3, stride, 1, 1);
    const size_t n = (size_t)oc * ot;
    _Float16 *x = malloc(n * 2), *two = malloc(n * 2), *fused = malloc(n * 2);
    double ret = -1;
    if (!x || !two || !fused) goto out;
    (void)ic;

    int rc = rocket_conv1d_fp16(fd, icp, it, oc, 3, stride, 1, in, W, x);
    if (rc) { fprintf(stderr, "%s: rocket_conv1d_fp16 %d\n", name, rc); goto out; }
    rc = rocket_activation_fp16(fd, ROCKET_ACTIVATION_GELU, x, two, (int)n);
    if (rc) { fprintf(stderr, "%s: rocket_activation_fp16 %d\n", name, rc); goto out; }
    rocket_conv2d_desc d = {
        .ic = icp, .ih = it, .iw = 1, .oc = oc, .kh = 3, .kw = 1,
        .stride_y = stride, .stride_x = 1, .pad_top = 1, .pad_left = 0,
        .dil_y = 1, .dil_x = 1, .depthwise = 0,
    };
    rc = rocket_conv2d_act_fp16(fd, &d, ROCKET_ACTIVATION_GELU, in, W, fused);
    if (rc) { fprintf(stderr, "%s: rocket_conv2d_act_fp16 %d\n", name, rc); goto out; }

    double xmin = 1e30, xmax = -1e30;
    long below3 = 0, band = 0;
    for (size_t i = 0; i < n; i++) {
        double v = (double)x[i];
        if (v < xmin) xmin = v;
        if (v > xmax) xmax = v;
        below3 += v < -3.0; band += fabs(v) < 0.0015;
    }
    printf("%s: oc %d ot %d, x in [%.3f, %.3f], %ld below -3 (%.2f%%), %ld within 0.0015 of 0\n",
           name, oc, ot, xmin, xmax, below3, 100.0 * below3 / n, band);
    score_t st = score(x, two, n), sf = score(x, fused, n);
    report(name, "two", st, n);
    report(name, "fused", sf, n);
    if (gelu_out)
        for (size_t i = 0; i < n; i++) gelu_out[i] = (_Float16)gelu((double)x[i]);
    ret = st.max_abs;
out:
    free(x); free(two); free(fused);
    return ret;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: %s DIR\n", argv[0]); return 2; }
    const char *dir = argv[1];
    int n_mels, n_state, it;
    {
        char path[1024];
        snprintf(path, sizeof path, "%s/meta.txt", dir);
        FILE *f = fopen(path, "r");
        if (!f || fscanf(f, "%d %d %d", &n_mels, &n_state, &it) != 3) {
            fprintf(stderr, "cannot read %s\n", path);
            if (f) fclose(f);
            return 2;
        }
        fclose(f);
    }
    _Float16 *mel = load(dir, "mel.f16", (size_t)n_mels * it * 2);
    _Float16 *w1 = load(dir, "w1.f16", (size_t)n_state * n_mels * 3 * 2);
    float *b1 = load(dir, "b1.f32", (size_t)n_state * 4);
    _Float16 *w2 = load(dir, "w2.f16", (size_t)n_state * n_state * 3 * 2);
    float *b2 = load(dir, "b2.f32", (size_t)n_state * 4);
    if (!mel || !w1 || !b1 || !w2 || !b2) return 2;

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no device (%d)\n", fd); return 2; }

    /* conv1: mel bins plus the bias channel, padded to 32. */
    const int ic1 = n_mels, icp1 = (ic1 + 1 + 31) / 32 * 32;
    _Float16 *in1 = calloc((size_t)icp1 * it, 2), *W1 = calloc((size_t)n_state * icp1 * 3, 2);
    for (int c = 0; c < ic1; c++) memcpy(in1 + (size_t)c * it, mel + (size_t)c * it, (size_t)it * 2);
    for (int t = 0; t < it; t++) in1[(size_t)ic1 * it + t] = (_Float16)1.0f;
    for (int o = 0; o < n_state; o++) {
        memcpy(W1 + (size_t)o * icp1 * 3, w1 + (size_t)o * ic1 * 3, (size_t)ic1 * 3 * 2);
        W1[((size_t)o * icp1 + ic1) * 3 + 1] = (_Float16)b1[o];
    }
    _Float16 *g1 = malloc((size_t)n_state * it * 2);
    double e1 = run_conv(fd, "conv1", icp1, ic1, it, n_state, 1, in1, W1, g1);

    /* conv2: conv1's exact GELU plus the bias channel, at stride 2. */
    const int ic2 = n_state, icp2 = (ic2 + 1 + 31) / 32 * 32;
    _Float16 *in2 = calloc((size_t)icp2 * it, 2), *W2 = calloc((size_t)n_state * icp2 * 3, 2);
    memcpy(in2, g1, (size_t)ic2 * it * 2);
    for (int t = 0; t < it; t++) in2[(size_t)ic2 * it + t] = (_Float16)1.0f;
    for (int o = 0; o < n_state; o++) {
        memcpy(W2 + (size_t)o * icp2 * 3, w2 + (size_t)o * ic2 * 3, (size_t)ic2 * 3 * 2);
        W2[((size_t)o * icp2 + ic2) * 3 + 1] = (_Float16)b2[o];
    }
    double e2 = e1 < 0 ? -1 : run_conv(fd, "conv2", icp2, ic2, it, n_state, 2, in2, W2, NULL);

    rocket_close(fd);
    printf("two-pass max_abs: conv1 %.4f, conv2 %.4f\n", e1, e2);
    return (e1 >= 0 && e2 >= 0 && e1 <= 0.01 && e2 <= 0.01) ? 0 : 1;
}
