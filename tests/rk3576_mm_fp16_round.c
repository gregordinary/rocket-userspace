// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_mm_fp16_round.c — how the RK3576 fp16 matmul rounds an fp32 sum that is not exact.
 *
 * A probe, not a gate. rk3576_mm_fp16_gate's fills make every sum exact
 * in fp32 in any order, so it cannot see how the part rounds. Here the fills carry full fp16
 * mantissas, so each product is exact in fp32 (11 x 11 bits) but the running sum is not, and
 * the device's fp32 output is scored against CPU models of the accumulation:
 *
 *   exact       the whole dot product summed in double, rounded once to fp32
 *   seq         fp32 accumulation one product at a time, in K order
 *   blkB        each block of B consecutive products summed in double and rounded to fp32,
 *               then the block sums accumulated in fp32 in K order (B = 8, 16, 32, 64)
 *   blkB-rz     the same with the fp32 accumulation rounding toward zero
 *
 * Each model is computed with round to nearest even unless named. For each shape and model the
 * probe prints the fraction of outputs that match bit for bit and the largest distance in
 * units of the last place. A model that matches everywhere at every K is the part's rounding;
 * one that matches at small K only is a different order that coincides there.
 *
 * Two fills: "unit", values k/1024 in [-1, 1] (one exponent range), and "wide", mantissas
 * with values spread over 2^-14 to 2^-3, so products differ in scale and alignment matters.
 * The device runs each shape twice and says whether the two outputs are identical.
 *
 *   rk3576_mm_fp16_round [M N]        default M 4, N 64; K runs 32 to 2048
 *
 * Exit 0 when every call returned 0, 1 otherwise, 2 with no device.
 */
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_matmul.h"
#include "rocket_npu.h"
#include "test_fill.h"

static const unsigned KS[] = { 32, 64, 128, 256, 512, 1024, 2048 };
static const unsigned BLK[] = { 8, 16, 32, 64 };
#define NMODEL (2 + 2 * (int)(sizeof(BLK) / sizeof(BLK[0])))

static void fill(_Float16 *p, size_t n, uint64_t seed, int wide)
{
    for (size_t i = 0; i < n; i++) {
        double v;
        if (!wide) {
            v = tf_int(seed, i, -1024, 1024) / 1024.0;
        } else {
            const int m = tf_int(seed, 2 * i, 1024, 2047);
            const int e = tf_int(seed, 2 * i + 1, -24, -14);   /* m * 2^e: 2^-14 .. 2^-3 */
            v = ldexp((double)m, e) * (tf_int(seed ^ 0x55, i, 0, 1) ? 1.0 : -1.0);
        }
        p[i] = (_Float16)v;
    }
}

/* Distance in units of the last place between two finite floats. */
static uint32_t ulps(float a, float b)
{
    int32_t x, y;
    memcpy(&x, &a, 4);
    memcpy(&y, &b, 4);
    if (x < 0) x = (int32_t)0x80000000 - x;
    if (y < 0) y = (int32_t)0x80000000 - y;
    return x > y ? (uint32_t)(x - y) : (uint32_t)(y - x);
}

/* One output of each model for row a, column b over K. */
static void models(const _Float16 *a, const _Float16 *b, unsigned K, float *out)
{
    volatile float acc;
    double s = 0.0;
    int j = 0;

    for (unsigned k = 0; k < K; k++) s += (double)a[k] * (double)b[k];
    out[j++] = (float)s;

    acc = 0.0f;
    for (unsigned k = 0; k < K; k++) acc = acc + (float)((double)a[k] * (double)b[k]);
    out[j++] = acc;

    for (size_t bi = 0; bi < sizeof(BLK) / sizeof(BLK[0]); bi++) {
        for (int rz = 0; rz < 2; rz++) {
            acc = 0.0f;
            for (unsigned k0 = 0; k0 < K; k0 += BLK[bi]) {
                double bs = 0.0;
                for (unsigned k = k0; k < k0 + BLK[bi] && k < K; k++)
                    bs += (double)a[k] * (double)b[k];
                const float bf = (float)bs;
                if (rz) fesetround(FE_TOWARDZERO);
                acc = acc + bf;
                if (rz) fesetround(FE_TONEAREST);
            }
            out[j++] = acc;
        }
    }
}

static void model_name(int j, char *buf, size_t n)
{
    if (j == 0) snprintf(buf, n, "exact");
    else if (j == 1) snprintf(buf, n, "seq");
    else snprintf(buf, n, "blk%u%s", BLK[(j - 2) / 2], (j - 2) % 2 ? "-rz" : "");
}

static int run_shape(int fd, unsigned M, unsigned K, unsigned N, int wide)
{
    _Float16 *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    float *C = malloc((size_t)M * N * 4), *C2 = malloc((size_t)M * N * 4);
    float mv[NMODEL];
    size_t match[NMODEL] = { 0 };
    uint32_t worst[NMODEL] = { 0 };
    int rc, rc2, same;

    if (!A || !B || !C || !C2) return -1;
    fill(A, (size_t)M * K, 0xF1 + K + 7u * (unsigned)wide, wide);
    fill(B, (size_t)N * K, 0xF2 + K + 7u * (unsigned)wide, wide);
    rc = rocket_matmul_fp16_rk3576(fd, (int)M, (int)K, (int)N, A, B, C);
    rc2 = rocket_matmul_fp16_rk3576(fd, (int)M, (int)K, (int)N, A, B, C2);
    if (rc || rc2) {
        printf("  %s %ux%ux%u: the entry returned %d and %d\n", wide ? "wide" : "unit", M, K, N,
               rc, rc2);
        free(A); free(B); free(C); free(C2);
        return -1;
    }
    same = memcmp(C, C2, (size_t)M * N * 4) == 0;
    for (unsigned m = 0; m < M; m++)
        for (unsigned n = 0; n < N; n++) {
            const float dev = C[(size_t)m * N + n];
            models(A + (size_t)m * K, B + (size_t)n * K, K, mv);
            for (int j = 0; j < NMODEL; j++) {
                const uint32_t u = ulps(dev, mv[j]);
                match[j] += u == 0;
                if (u > worst[j]) worst[j] = u;
            }
        }
    printf("  %s %ux%4ux%u, two runs %s:", wide ? "wide" : "unit", M, K, N,
           same ? "identical" : "DIFFER");
    for (int j = 0; j < NMODEL; j++) {
        char nm[16];
        model_name(j, nm, sizeof nm);
        printf(" %s %.1f%%/%u", nm, 100.0 * (double)match[j] / (double)(M * N), worst[j]);
    }
    printf("\n");
    free(A); free(B); free(C); free(C2);
    return 0;
}

int main(int argc, char **argv)
{
    unsigned M = 4, N = 64;
    int fd, bad = 0;

    if (argc > 2) { M = (unsigned)atoi(argv[1]); N = (unsigned)atoi(argv[2]); }
    fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }
    printf("== fp16 matmul rounding, M %u N %u: per model, %% of outputs bit-exact / worst ulps ==\n",
           M, N);
    for (int wide = 0; wide < 2; wide++)
        for (size_t i = 0; i < sizeof(KS) / sizeof(KS[0]); i++)
            bad += run_shape(fd, M, KS[i], N, wide) != 0;
    rocket_close(fd);
    return bad ? 1 : 0;
}
