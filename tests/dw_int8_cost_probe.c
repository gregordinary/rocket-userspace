// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * dw_int8_cost_probe.c — where the RK3588 int8 depthwise entry's per-call time goes.
 *
 * A probe, not a gate: it measures and asserts nothing about the numbers. At each shape it
 * times, as medians over warm repeats:
 *
 *   call     rocket_conv2d_dw_int8_ctx(), the whole entry on a resident context
 *   scat     the entry's input scatter as written: a memset of the input BO's size, then
 *            one feature_data() call per element from the [C][IH][IW] source
 *   gath     the entry's output gather as written: one feature_data() call per element
 *   bscat    a blocked scatter to the same cube, the index hoisted out of the inner loop
 *   bgath    a blocked gather from the same cube
 *
 * scat and gath run on host buffers of the entry's sizes, not on the BOs, so they price the
 * loops rather than the mapping. The blocked forms are checked byte for byte against the
 * scalar ones before either is timed. The shapes are MobileDet's largest depthwise layers at
 * the extents the delegate programs (its SAME pad materialized, so pad 0).
 *
 *   dw_int8_cost_probe [reps]      default 20
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "test_fill.h"

#define C2 16

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int cmpd(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n)
{
    qsort(v, n, sizeof v[0], cmpd);
    return v[n / 2];
}

static void scat_scalar(int8_t *dst, size_t dst_bytes, const int8_t *in, int C, int Cj,
                        int IH, int IHj, int IW)
{
    memset(dst, 0, dst_bytes);
    for (int c = 0; c < C; c++)
        for (int ih = 0; ih < IH; ih++)
            for (int iw = 0; iw < IW; iw++)
                dst[feature_data(Cj, IHj, IW, C2, c + 1, ih + 1, iw + 1)] =
                    in[((size_t)c * IH + ih) * IW + iw];
}

static void scat_blocked(int8_t *dst, size_t dst_bytes, const int8_t *in, int C, int IH,
                         int IHj, int IW)
{
    memset(dst, 0, dst_bytes);
    const size_t plane = (size_t)IHj * IW * C2;
    for (int c = 0; c < C; c++) {
        int8_t *d = dst + (size_t)(c / C2) * plane + (c % C2);
        const int8_t *s = in + (size_t)c * IH * IW;
        for (size_t p = 0; p < (size_t)IH * IW; p++) d[p * C2] = s[p];
    }
}

static void gath_scalar(int8_t *out, const int8_t *src, int C, int Cj, int OH, int OW)
{
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                out[((size_t)c * OH + oh) * OW + ow] =
                    src[feature_data(Cj, OH, OW, C2, c + 1, oh + 1, ow + 1)];
}

static void gath_blocked(int8_t *out, const int8_t *src, int C, int OH, int OW)
{
    const size_t plane = (size_t)OH * OW * C2;
    for (int c = 0; c < C; c++) {
        const int8_t *s = src + (size_t)(c / C2) * plane + (c % C2);
        int8_t *d = out + (size_t)c * OH * OW;
        for (size_t p = 0; p < (size_t)OH * OW; p++) d[p] = s[p * C2];
    }
}

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    /* C, IH, IW (pad materialized), K, stride */
    static const int S[][5] = {
        { 320, 41, 41, 3, 2 }, { 576, 22, 22, 3, 1 }, { 576, 24, 24, 5, 1 },
        { 768, 24, 24, 5, 1 }, { 768, 22, 22, 3, 1 }, { 768, 23, 23, 5, 2 },
        { 960, 12, 12, 3, 1 }, { 960, 14, 14, 5, 1 }, {  96, 22, 22, 3, 1 },
    };
    int fd = rocket_open();
    if (fd < 0) { printf("dw_int8_cost_probe: no NPU device\n"); return 77; }
    rocket_conv_ctx *ctx = rocket_conv_ctx_create(fd);
    double *t = malloc(sizeof(double) * (reps > 0 ? reps : 1));
    if (!ctx || !t || reps < 1) { fprintf(stderr, "setup failed\n"); return 1; }
    printf("dw_int8_cost_probe: ms per call, median of %d warm repeats\n", reps);
    printf("  C    IHxIW  K s |   call   scat   gath |  bscat  bgath\n");
    double sum[5] = {0};
    for (size_t i = 0; i < sizeof S / sizeof S[0]; i++) {
        const int C = S[i][0], IH = S[i][1], IW = S[i][2], K = S[i][3], s = S[i][4];
        rocket_conv2d_desc d = { .ic = C, .ih = IH, .iw = IW, .oc = C, .kh = K, .kw = K,
            .stride_y = s, .stride_x = s, .dil_y = 1, .dil_x = 1, .depthwise = 1 };
        const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d);
        const int Cj = ((C + 63) / 64) * 64;
        const int IHj = IH < 4 ? 4 : IH;
        const size_t in_n = (size_t)C * IH * IW, out_n = (size_t)C * OH * OW;
        const size_t cube_in = (size_t)Cj * IHj * IW + NPU_CBUF_BANK_SIZE;
        const size_t cube_out = (size_t)Cj * OH * OW;
        int8_t *in = malloc(in_n), *w = malloc((size_t)C * K * K), *out = malloc(out_n);
        int8_t *o2 = malloc(out_n), *ci = malloc(cube_in), *ci2 = malloc(cube_in);
        int8_t *co = malloc(cube_out);
        int32_t *bias = calloc((size_t)C, sizeof(int32_t));
        if (!in || !w || !out || !o2 || !ci || !ci2 || !co || !bias) { fprintf(stderr, "oom\n"); return 1; }
        tf_fill_i8(in, in_n, tf_hash(0xC057, i * 2 + 1), -128, 127);
        tf_fill_i8(w, (size_t)C * K * K, tf_hash(0xC057, i * 2 + 2), -128, 127);
        tf_fill_i8(co, cube_out, tf_hash(0xC057, i * 2 + 3), -128, 127);

        /* the blocked forms must write exactly what the scalar ones write */
        scat_scalar(ci, cube_in, in, C, Cj, IH, IHj, IW);
        scat_blocked(ci2, cube_in, in, C, IH, IHj, IW);
        gath_scalar(out, co, C, Cj, OH, OW);
        gath_blocked(o2, co, C, OH, OW);
        if (memcmp(ci, ci2, cube_in) || memcmp(out, o2, out_n)) {
            printf("  C=%d %dx%d: blocked form differs from the scalar one -> FAIL\n", C, IH, IW);
            return 1;
        }

        double m[5];
        for (int r = 0; r < 3; r++)                       /* warm the context and BOs */
            if (rocket_conv2d_dw_int8_ctx(ctx, &d, in, w, bias, 0.02f, 0.003f, 0.05f,
                                          -3, 5, 2, out)) { printf("  call failed\n"); return 1; }
        for (int r = 0; r < reps; r++) {
            double a = now_ms();
            rocket_conv2d_dw_int8_ctx(ctx, &d, in, w, bias, 0.02f, 0.003f, 0.05f, -3, 5, 2, out);
            t[r] = now_ms() - a;
        }
        m[0] = median(t, reps);
        for (int r = 0; r < reps; r++) { double a = now_ms(); scat_scalar(ci, cube_in, in, C, Cj, IH, IHj, IW); t[r] = now_ms() - a; }
        m[1] = median(t, reps);
        for (int r = 0; r < reps; r++) { double a = now_ms(); gath_scalar(out, co, C, Cj, OH, OW); t[r] = now_ms() - a; }
        m[2] = median(t, reps);
        for (int r = 0; r < reps; r++) { double a = now_ms(); scat_blocked(ci2, cube_in, in, C, IH, IHj, IW); t[r] = now_ms() - a; }
        m[3] = median(t, reps);
        for (int r = 0; r < reps; r++) { double a = now_ms(); gath_blocked(o2, co, C, OH, OW); t[r] = now_ms() - a; }
        m[4] = median(t, reps);
        printf("  %4d %3dx%-3d %d %d | %6.3f %6.3f %6.3f | %6.3f %6.3f\n", C, IH, IW, K, s,
               m[0], m[1], m[2], m[3], m[4]);
        for (int k = 0; k < 5; k++) sum[k] += m[k];
        free(in); free(w); free(out); free(o2); free(ci); free(ci2); free(co); free(bias);
    }
    printf("  sum             | %6.3f %6.3f %6.3f | %6.3f %6.3f\n",
           sum[0], sum[1], sum[2], sum[3], sum[4]);
    rocket_conv_ctx_free(ctx);
    rocket_close(fd);
    free(t);
    return 0;
}
