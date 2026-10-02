// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_pack_cost_probe.c — how much of the RK3588 fp16 conv's and int8 direct conv's per-call
 * time is their per-element host pack.
 *
 * A probe, not a gate: it measures and asserts nothing about the numbers. Two sections:
 *
 *   fp16   rocket_conv2d_fp16_ctx() at EfficientDet-Lite0's depthwise shapes (its SAME pad
 *          materialized, so pad 0). The pack replicated is the entry's: the input scatter to
 *          the C2=8 feature cube, the depthwise weight scatter (group 32) and the output
 *          gather, each one feature_data() / weight_conv_dw_fp16() call per element.
 *   int8   rocket_conv2d_int8_ctx() at MobileDet's direct-conv shapes. The pack replicated is
 *          the single-tile one: the input scatter to the C2=16 cube and the int32 output
 *          gather from the C2=4 cube. A shape the entry tiles packs per tile, one element at
 *          a time as well, so this prices the loop and not the tile count.
 *
 * Each row is the whole warm call, the scalar pack, and a blocked pack with the index hoisted,
 * all medians. The blocked forms are checked byte for byte against the scalar ones first. The
 * replicas write host buffers of the entry's sizes, not the BOs.
 *
 *   conv_pack_cost_probe [reps]      default 20
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

#define TIME(dst, reps, t, expr) do { for (int r_ = 0; r_ < (reps); r_++) { \
        double a_ = now_ms(); expr; (t)[r_] = now_ms() - a_; } (dst) = median((t), (reps)); } while (0)

/* ---- fp16 ---- */
static void f16_scat_s(_Float16 *d, size_t bytes, const _Float16 *in, int C, int IH, int IHj, int IW)
{
    memset(d, 0, bytes);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < IH; h++)
            for (int w = 0; w < IW; w++)
                d[feature_data(C, IHj, IW, 8, c + 1, h + 1, w + 1)] = in[((size_t)c * IH + h) * IW + w];
}
static void f16_scat_b(_Float16 *d, size_t bytes, const _Float16 *in, int C, int IH, int IHj, int IW)
{
    memset(d, 0, bytes);
    const size_t plane = (size_t)IHj * IW * 8, hw = (size_t)IH * IW;
    for (int c = 0; c < C; c++) {
        _Float16 *o = d + (size_t)(c / 8) * plane + (c % 8);
        const _Float16 *s = in + (size_t)c * hw;
        for (size_t p = 0; p < hw; p++) o[p * 8] = s[p];
    }
}
static void f16_wdw_s(_Float16 *d, size_t bytes, const _Float16 *W, int C, int K, int G)
{
    memset(d, 0, bytes);
    for (int c = 0; c < C; c++)
        for (int kh = 0; kh < K; kh++)
            for (int kw = 0; kw < K; kw++)
                d[weight_conv_dw_fp16(C, K, K, G, c + 1, kh + 1, kw + 1)] = W[((size_t)c * K + kh) * K + kw];
}
static void f16_wdw_b(_Float16 *d, size_t bytes, const _Float16 *W, int C, int K, int G)
{
    memset(d, 0, bytes);
    const size_t taps = (size_t)K * K;
    for (int c = 0; c < C; c++) {
        _Float16 *o = d + (size_t)(c / G) * taps * G + (c % G);
        const _Float16 *s = W + (size_t)c * taps;
        for (size_t t = 0; t < taps; t++) o[t * G] = s[t];
    }
}
static void f16_gath_s(_Float16 *out, const _Float16 *src, int C, int OH, int OW)
{
    for (int c = 0; c < C; c++)
        for (int h = 0; h < OH; h++)
            for (int w = 0; w < OW; w++)
                out[((size_t)c * OH + h) * OW + w] = src[feature_data(C, OH, OW, 8, c + 1, h + 1, w + 1)];
}
static void f16_gath_b(_Float16 *out, const _Float16 *src, int C, int OH, int OW)
{
    const size_t plane = (size_t)OH * OW * 8, hw = (size_t)OH * OW;
    for (int c = 0; c < C; c++) {
        const _Float16 *s = src + (size_t)(c / 8) * plane + (c % 8);
        _Float16 *o = out + (size_t)c * hw;
        for (size_t p = 0; p < hw; p++) o[p] = s[p * 8];
    }
}

/* ---- int8 direct ---- */
static void i8_scat_s(int8_t *d, size_t bytes, const int8_t *in, int C, int IH, int IHj, int IW)
{
    memset(d, 0, bytes);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < IH; h++)
            for (int w = 0; w < IW; w++)
                d[feature_data(C, IHj, IW, 16, c + 1, h + 1, w + 1)] = in[((size_t)c * IH + h) * IW + w];
}
static void i8_scat_b(int8_t *d, size_t bytes, const int8_t *in, int C, int IH, int IHj, int IW)
{
    memset(d, 0, bytes);
    const size_t plane = (size_t)IHj * IW * 16, hw = (size_t)IH * IW;
    for (int c = 0; c < C; c++) {
        int8_t *o = d + (size_t)(c / 16) * plane + (c % 16);
        const int8_t *s = in + (size_t)c * hw;
        for (size_t p = 0; p < hw; p++) o[p * 16] = s[p];
    }
}
static void i32_gath_s(int32_t *out, const int32_t *src, int C, int OH, int OW)
{
    for (int c = 0; c < C; c++)
        for (int h = 0; h < OH; h++)
            for (int w = 0; w < OW; w++)
                out[((size_t)c * OH + h) * OW + w] = src[feature_data(C, OH, OW, 4, c + 1, h + 1, w + 1)];
}
static void i32_gath_b(int32_t *out, const int32_t *src, int C, int OH, int OW)
{
    const size_t plane = (size_t)OH * OW * 4, hw = (size_t)OH * OW;
    for (int c = 0; c < C; c++) {
        const int32_t *s = src + (size_t)(c / 4) * plane + (c % 4);
        int32_t *o = out + (size_t)c * hw;
        for (size_t p = 0; p < hw; p++) o[p] = s[p * 4];
    }
}

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    int fd = rocket_open();
    if (fd < 0) { printf("conv_pack_cost_probe: no NPU device\n"); return 77; }
    rocket_conv_ctx *ctx = rocket_conv_ctx_create(fd);
    double *t = malloc(sizeof(double) * (reps > 0 ? reps : 1));
    if (!ctx || !t || reps < 1) { fprintf(stderr, "setup failed\n"); return 1; }

    /* fp16 depthwise: C, IH, IW (pad materialized), K, stride */
    static const int F[][5] = {
        {  144, 82, 82, 3, 1 }, {  144, 83, 83, 5, 2 }, {  240, 44, 44, 5, 1 },
        {  240, 41, 41, 3, 2 }, {  480, 22, 22, 3, 1 }, {  672, 24, 24, 5, 1 },
        { 1152, 14, 14, 5, 1 }, {   64, 42, 42, 3, 1 }, {   64, 22, 22, 3, 1 },
    };
    const int G = 32;
    double fs[7] = {0};
    printf("conv_pack_cost_probe: ms per call, median of %d warm repeats\n", reps);
    printf("fp16 depthwise   C   IHxIW K s |   call  scat  wdw   gath |  bscat  bwdw  bgath\n");
    for (size_t i = 0; i < sizeof F / sizeof F[0]; i++) {
        const int C = F[i][0], IH = F[i][1], IW = F[i][2], K = F[i][3], s = F[i][4];
        rocket_conv2d_desc d = { .ic = C, .ih = IH, .iw = IW, .oc = C, .kh = K, .kw = K,
            .stride_y = s, .stride_x = s, .dil_y = 1, .dil_x = 1, .depthwise = 1 };
        if (rocket_conv2d_plan(&d)) { printf("  C=%d %dx%d: plan refuses, skipped\n", C, IH, IW); continue; }
        const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d), IHj = IH < 4 ? 4 : IH;
        const int Cpad = ((C + G - 1) / G) * G;
        const size_t in_n = (size_t)C * IH * IW, out_n = (size_t)C * OH * OW;
        const size_t cin = ((size_t)C * IHj * IW + NPU_CBUF_BANK_SIZE) * 2;
        const size_t cw = ((size_t)Cpad * K * K + NPU_CBUF_BANK_SIZE) * 2, cout = (size_t)Cpad * OH * OW * 2;
        _Float16 *in = malloc(in_n * 2), *w = malloc((size_t)C * K * K * 2), *out = malloc(out_n * 2);
        _Float16 *o2 = malloc(out_n * 2), *a = malloc(cin), *b = malloc(cin);
        _Float16 *wa = malloc(cw), *wb = malloc(cw), *co = malloc(cout);
        if (!in || !w || !out || !o2 || !a || !b || !wa || !wb || !co) { fprintf(stderr, "oom\n"); return 1; }
        for (size_t k = 0; k < in_n; k++) in[k] = (_Float16)((int)((k * 37 + 11) % 200) - 100) * (_Float16)0.01f;
        for (size_t k = 0; k < (size_t)C * K * K; k++) w[k] = (_Float16)((int)((k * 13 + 5) % 50) - 25) * (_Float16)0.01f;
        for (size_t k = 0; k < cout / 2; k++) co[k] = (_Float16)(int)(k % 97);
        f16_scat_s(a, cin, in, C, IH, IHj, IW); f16_scat_b(b, cin, in, C, IH, IHj, IW);
        int bad = memcmp(a, b, cin) != 0;
        f16_wdw_s(wa, cw, w, C, K, G); f16_wdw_b(wb, cw, w, C, K, G);
        bad |= memcmp(wa, wb, cw) != 0;
        f16_gath_s(out, co, C, OH, OW); f16_gath_b(o2, co, C, OH, OW);
        bad |= memcmp(out, o2, out_n * 2) != 0;
        if (bad) { printf("  C=%d %dx%d: a blocked form differs -> FAIL\n", C, IH, IW); return 1; }
        double m[7];
        for (int r = 0; r < 3; r++)
            if (rocket_conv2d_fp16_ctx(ctx, &d, in, w, out)) { printf("  call failed\n"); return 1; }
        TIME(m[0], reps, t, rocket_conv2d_fp16_ctx(ctx, &d, in, w, out));
        TIME(m[1], reps, t, f16_scat_s(a, cin, in, C, IH, IHj, IW));
        TIME(m[2], reps, t, f16_wdw_s(wa, cw, w, C, K, G));
        TIME(m[3], reps, t, f16_gath_s(out, co, C, OH, OW));
        TIME(m[4], reps, t, f16_scat_b(b, cin, in, C, IH, IHj, IW));
        TIME(m[5], reps, t, f16_wdw_b(wb, cw, w, C, K, G));
        TIME(m[6], reps, t, f16_gath_b(o2, co, C, OH, OW));
        printf("  %4d %3dx%-3d %d %d | %6.3f %5.3f %5.3f %6.3f | %6.3f %5.3f %6.3f\n", C, IH, IW, K, s,
               m[0], m[1], m[2], m[3], m[4], m[5], m[6]);
        for (int k = 0; k < 7; k++) fs[k] += m[k];
        free(in); free(w); free(out); free(o2); free(a); free(b); free(wa); free(wb); free(co);
    }
    printf("  sum                  | %6.3f %5.3f %5.3f %6.3f | %6.3f %5.3f %6.3f\n",
           fs[0], fs[1], fs[2], fs[3], fs[4], fs[5], fs[6]);

    /* int8 direct: IC, IH, IW (pad materialized), K, stride, OC */
    static const int D[][6] = {
        {  3, 322, 322, 3, 2,  32 }, {  8, 162, 162, 3, 1,  16 }, { 16, 161, 161, 3, 2, 128 },
        { 16,  82,  82, 3, 1, 128 }, { 16,  82,  82, 3, 1,  64 }, { 40,  42,  42, 3, 1, 160 },
        { 72,  22,  22, 3, 1, 288 },
    };
    double is[5] = {0};
    printf("int8 direct    IC   IHxIW  K s  OC |   call   scat   gath |  bscat  bgath\n");
    for (size_t i = 0; i < sizeof D / sizeof D[0]; i++) {
        const int IC = D[i][0], IH = D[i][1], IW = D[i][2], K = D[i][3], s = D[i][4], OC = D[i][5];
        rocket_conv2d_desc d = { .ic = IC, .ih = IH, .iw = IW, .oc = OC, .kh = K, .kw = K,
            .stride_y = s, .stride_x = s, .dil_y = 1, .dil_x = 1 };
        const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d), IHj = IH < 4 ? 4 : IH;
        const size_t in_n = (size_t)IC * IH * IW, out_n = (size_t)OC * OH * OW;
        const size_t cin = (size_t)((IC + 15) / 16 * 16) * IHj * IW + NPU_CBUF_BANK_SIZE;
        const size_t cout = (size_t)((OC + 3) / 4 * 4) * OH * OW;
        int8_t *in = malloc(in_n), *w = malloc((size_t)OC * IC * K * K), *a = malloc(cin), *b = malloc(cin);
        int32_t *out = malloc(out_n * 4), *o2 = malloc(out_n * 4), *co = malloc(cout * 4);
        if (!in || !w || !a || !b || !out || !o2 || !co) { fprintf(stderr, "oom\n"); return 1; }
        tf_fill_i8(in, in_n, tf_hash(0xB10C, i * 2 + 1), -128, 127);
        tf_fill_i8(w, (size_t)OC * IC * K * K, tf_hash(0xB10C, i * 2 + 2), -128, 127);
        for (size_t k = 0; k < cout; k++) co[k] = (int32_t)(k * 2654435761u);
        i8_scat_s(a, cin, in, IC, IH, IHj, IW); i8_scat_b(b, cin, in, IC, IH, IHj, IW);
        int bad = memcmp(a, b, cin) != 0;
        i32_gath_s(out, co, OC, OH, OW); i32_gath_b(o2, co, OC, OH, OW);
        bad |= memcmp(out, o2, out_n * 4) != 0;
        if (bad) { printf("  IC=%d %dx%d: a blocked form differs -> FAIL\n", IC, IH, IW); return 1; }
        double m[5];
        for (int r = 0; r < 3; r++)
            if (rocket_conv2d_int8_ctx(ctx, &d, in, w, out)) { printf("  call failed\n"); return 1; }
        TIME(m[0], reps, t, rocket_conv2d_int8_ctx(ctx, &d, in, w, out));
        TIME(m[1], reps, t, i8_scat_s(a, cin, in, IC, IH, IHj, IW));
        TIME(m[2], reps, t, i32_gath_s(out, co, OC, OH, OW));
        TIME(m[3], reps, t, i8_scat_b(b, cin, in, IC, IH, IHj, IW));
        TIME(m[4], reps, t, i32_gath_b(o2, co, OC, OH, OW));
        printf("  %4d %3dx%-3d %d %d %4d | %6.3f %6.3f %6.3f | %6.3f %6.3f\n", IC, IH, IW, K, s, OC,
               m[0], m[1], m[2], m[3], m[4]);
        for (int k = 0; k < 5; k++) is[k] += m[k];
        free(in); free(w); free(a); free(b); free(out); free(o2); free(co);
    }
    printf("  sum                      | %6.3f %6.3f %6.3f | %6.3f %6.3f\n",
           is[0], is[1], is[2], is[3], is[4]);
    rocket_conv_ctx_free(ctx);
    rocket_close(fd);
    free(t);
    return 0;
}
