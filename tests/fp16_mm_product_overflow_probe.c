// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * fp16_mm_product_overflow_probe.c -- where does the RK3588 fp16 matmul overflow: in the
 * output narrowing only, or already in a product?
 *
 * The output converter narrows an fp32 sum to fp16 with round-to-nearest-even and writes
 * +-inf from 65520 [HW sweep 2026-11-02, a sum of small products]. That measurement never put
 * a single product past 65504, which is what a flash-attention score with one-hot Q and K
 * does. This probe does, through rocket_matmul_fp16 (C = A.B^T), the entry the flash
 * attention's scores come from, at M 4, K 128, N 64 (one K pass).
 *
 * A[m][k] = s_m * alpha_k: a few large activations (k 0-11) and ones (k 16-127), with the
 * rows' signs + - + -. Each column of B selects products: one product past 65504, several
 * in-range products summing past it, many small ones, and CANCELLATIONS -- a product past
 * 65504 brought back into range by negative products. Every element is scored against:
 *   wide      each product exact, the sum exact, narrowed once at the output (inf from 65520)
 *   outsat    the same, the output clamped to +-65504 instead of inf
 *   prodsat   each product clamped to +-65504 before the sum
 *   prodinf   each product past 65520 is +-inf before the sum (inf - inf is NaN)
 *   prodf16   each product rounded to fp16 before the sum
 * The cancellations separate the product-level models from the output-level ones: 70000 -
 * 10000 reads 60000 wide, 55504 under prodsat and inf under prodinf. Column 25 (33300 -
 * 33000) separates prodf16, which reads 320, from wide's 300. Columns 27-63 are in-range
 * controls. rocket_matmul_fp16_batch then runs the same operands as two items, and its
 * bits are compared with the single call's.
 *
 * Exit 0 when one model explains every element and the batch matches, 1 otherwise, 2
 * without a device.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocket_matmul.h"
#include "rocket_npu.h"

#define M 4
#define K 128
#define N 64
#define K_ONES 16          /* first of the ones slots, alpha = 1 */

enum { WIDE, OUTSAT, PRODSAT, PRODINF, PRODF16, NMODEL };
static const char *model_name[NMODEL] = { "wide", "outsat", "prodsat", "prodinf", "prodf16" };

static const double alpha[16] = { 256, 45, 625, 375, 1000, 100, 100, 2047,
                                  200, 200, 200, 999, 0, 0, 0, 0 };
static const int sign[M] = { 1, -1, 1, -1 };

static double A_d[M][K];
static double B_d[N][K];
static const char *col_name[N];

static uint16_t bits(_Float16 h) { uint16_t u; memcpy(&u, &h, 2); return u; }
static _Float16 f16(double x) { return (_Float16)(float)x; }   /* x exact in fp32 here */

/* Spread `t` over the ones slots [k0, k1) as integers of at most 2048 in magnitude. */
static void spread(int n, int k0, int k1, double t)
{
    double sgn = t < 0 ? -1.0 : 1.0, r = fabs(t);
    for (int k = k0; k < k1 && r > 0; k++) {
        double v = r > 2048 ? 2048 : r;
        B_d[n][k] += sgn * v;
        r -= v;
    }
    if (r > 0) { fprintf(stderr, "spread: %g does not fit [%d,%d)\n", t, k0, k1); }
}

static void build(void)
{
    static const double many_t[] = { 65472, 65488, 65504, 65519, 65520, 65536, 70000, 135000,
                                     200000 };
    static const char *many_n[] = { "many 65472", "many 65488 tie", "many 65504",
                                    "many 65519", "many 65520 tie", "many 65536",
                                    "many 70000", "many 135000", "many 200000" };
    unsigned rng = 0x2545F491u;

    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            A_d[m][k] = sign[m] * (k < 16 ? alpha[k] : 1.0);
    memset(B_d, 0, sizeof B_d);

    B_d[0][0] = 255.75;  col_name[0] = "one 65472";
    B_d[1][7] = 32;      col_name[1] = "one 65504";
    B_d[2][1] = 1456;    col_name[2] = "one 65520 tie";
    B_d[3][0] = 256;     col_name[3] = "one 65536";
    B_d[4][2] = 112;     col_name[4] = "one 70000";
    B_d[5][3] = 360;     col_name[5] = "one 135000";
    B_d[6][4] = 1000;    col_name[6] = "one 1e6";
    B_d[7][5] = 600;     col_name[7] = "one 60000 (in range)";
    B_d[8][5] = 350;  B_d[8][6] = 350;                      col_name[8] = "two 35000+35000";
    B_d[9][8] = 225;  B_d[9][9] = 225;  B_d[9][10] = 225;  col_name[9] = "three 45000x3";
    for (int i = 0; i < 9; i++) { spread(10 + i, K_ONES, K, many_t[i]); col_name[10 + i] = many_n[i]; }
    B_d[19][2] = 112;  B_d[19][5] = -100;                   col_name[19] = "70000-10000";
    B_d[20][3] = 360;  B_d[20][8] = B_d[20][9] = B_d[20][10] = -200;
    col_name[20] = "135000-3x40000";
    B_d[21][4] = 1000; B_d[21][11] = -1000;                 col_name[21] = "1e6-999000";
    B_d[22][0] = 256;  spread(22, K_ONES, K, -64);          col_name[22] = "65536-64";
    B_d[23][1] = 1456; spread(23, K_ONES, K, -16);          col_name[23] = "65520-16";
    spread(24, K_ONES, 80, 70000); spread(24, 80, K, -10000);
    col_name[24] = "many 70000-10000";
    B_d[25][5] = 333;  B_d[25][6] = -330;                   col_name[25] = "33300-33000";
    B_d[26][7] = 31;   B_d[26][0] = 1;                      col_name[26] = "63457+256";
    for (int n = 27; n < N; n++) {
        for (int k = K_ONES; k < K; k++) {
            rng = rng * 1664525u + 1013904223u;
            B_d[n][k] = (double)((int)((rng >> 24) % 9) - 4);
        }
        rng = rng * 1664525u + 1013904223u;
        B_d[n][5] = (double)((int)((rng >> 24) % 101) - 50);
        col_name[n] = "control";
    }
}

static double clampv(double x, double c) { return x > c ? c : x < -c ? -c : x; }

static _Float16 predict(int model, int m, int n)
{
    double s = 0;
    for (int k = 0; k < K; k++) {
        double p = A_d[m][k] * B_d[n][k];
        if (p == 0) continue;
        switch (model) {
        case PRODSAT: p = clampv(p, 65504.0); break;
        case PRODINF: if (fabs(p) >= 65520.0) p = p > 0 ? INFINITY : -INFINITY; break;
        case PRODF16: p = (double)f16(p); break;
        default: break;
        }
        s += p;
    }
    if (model == OUTSAT && fabs(s) >= 65520.0) s = s > 0 ? 65504.0 : -65504.0;
    return f16(s);
}

static int same(_Float16 a, _Float16 b)
{
    if (isnan((float)a) && isnan((float)b)) return 1;
    return bits(a) == bits(b);
}

int main(void)
{
    static _Float16 A[M * K], B[N * K], C[M * N], C2a[M * N], C2b[M * N];
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no device (%d)\n", fd); return 2; }

    build();
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) A[m * K + k] = f16(A_d[m][k]);
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) B[n * K + k] = f16(B_d[n][k]);
    for (int n = 0; n < N; n++) for (int k = 0; k < K; k++)
        if ((double)B[n * K + k] != B_d[n][k]) { fprintf(stderr, "B[%d][%d] not fp16\n", n, k); return 1; }

    memset(C, 0xAA, sizeof C);
    int rc = rocket_matmul_fp16(fd, M, K, N, A, B, C);
    if (rc) { fprintf(stderr, "rocket_matmul_fp16: %d\n", rc); return 1; }

    int hit[NMODEL] = { 0 };
    printf("fp16 matmul product overflow, M %d K %d N %d\n", M, K, N);
    printf("  %-22s %3s %12s  got     wide    outsat  prodsat prodinf prodf16\n",
           "column", "row", "exact");
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            _Float16 got = C[m * N + n];
            double ex = 0;
            for (int k = 0; k < K; k++) ex += A_d[m][k] * B_d[n][k];
            _Float16 pr[NMODEL];
            for (int md = 0; md < NMODEL; md++) {
                pr[md] = predict(md, m, n);
                hit[md] += same(got, pr[md]);
            }
            if (n < 27 && m < 2)
                printf("  %-22s %3d %12.2f  0x%04x  0x%04x  0x%04x  0x%04x  0x%04x  0x%04x%s\n",
                       col_name[n], m, ex, bits(got), bits(pr[WIDE]), bits(pr[OUTSAT]),
                       bits(pr[PRODSAT]), bits(pr[PRODINF]), bits(pr[PRODF16]),
                       same(got, pr[WIDE]) ? "" : "   <- not wide");
        }
    }
    int best = 0;
    for (int md = 0; md < NMODEL; md++) {
        printf("  model %-8s explains %3d of %d\n", model_name[md], hit[md], M * N);
        if (hit[md] > hit[best]) best = md;
    }

    /* The flash attention runs its scores through the batched entry: same operands, two items. */
    const _Float16 *pA[2] = { A, A }, *pB[2] = { B, B };
    _Float16 *pC[2] = { C2a, C2b };
    memset(C2a, 0xAA, sizeof C2a);
    memset(C2b, 0xAA, sizeof C2b);
    rc = rocket_matmul_fp16_batch(fd, M, K, N, 2, pA, pB, pC);
    int bdiff = 0;
    if (rc) { fprintf(stderr, "rocket_matmul_fp16_batch: %d\n", rc); bdiff = -1; }
    else
        for (int i = 0; i < M * N; i++)
            bdiff += !same(C[i], C2a[i]) + !same(C[i], C2b[i]);
    printf("  batch (2 items) against the single call: %d of %d elements differ\n",
           bdiff, 2 * M * N);

    rocket_close(fd);
    return (hit[best] == M * N && bdiff == 0) ? 0 : 1;
}
