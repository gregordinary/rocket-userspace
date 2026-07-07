// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int4_groupwise_rocket.c — bit-exact gate for rocket_matmul_int4_groupwise.
 *
 * The group-wise int4 matmul computes, in fp32:
 *   C_f[m,n] = sum_g a_scale[m,g] * b_scale[n,g] * (sum_{k in group g} A[m,k]*B[n,k])
 * with one K-tile == one quant group. This validates that per-K-slice dequant against a
 * host reference using IDENTICAL int4 inputs and per-group scales — it isolates the
 * matmul (packing / per-group scaling / fp32 K-accum), independent of quant quality.
 *
 * Inputs are int4 in [-7,7] and group is small enough (49*group < 32767) that no int16
 * K-tile partial saturates, so the NPU partial is an exact integer and the only residual
 * is the fp32 accumulation of the per-group scaled terms (ref in fp64) -> max_rel ~1e-6.
 *
 * Usage: matmul_int4_groupwise_rocket [M K N group]   (default 128 3840 256 128)
 *   Needs M%4, K%32, N%64, K%group, group%32, 49*group < 32767.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int8_t rand_i4(void) { return (int8_t)(rand() % 15 - 7); }   /* [-7,7] */
static float  rand_scale(void) { return 0.5f + (rand() % 1000) / 1000.0f; }  /* [0.5,1.5) */

int main(int argc, char **argv) {
    int M = 128, K = 3840, N = 256, group = 128;
    if (argc == 5) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); group = atoi(argv[4]); }
    else if (argc != 1) { printf("usage: %s [M K N group]\n", argv[0]); return -1; }
    if (M % 4 || K % 32 || N % 64 || K % group || group % 32 || 49 * group >= 32767) {
        fprintf(stderr, "bad shape: need M%%4, K%%32, N%%64, K%%group, group%%32, 49*group<32767\n");
        return -1;
    }
    const int nG = K / group;
    printf("int4 GROUPWISE C_f[%d,%d] = A[%d,%d] x B[%d,%d]^T, group=%d (nG=%d)\n",
           M, N, M, K, N, K, group, nG);

    int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K);
    float  *as = malloc((size_t)M * nG * sizeof(float)), *bs = malloc((size_t)N * nG * sizeof(float));
    float  *Cf = malloc((size_t)M * N * sizeof(float));
    double *ref = malloc((size_t)M * N * sizeof(double));
    if (!A || !B || !as || !bs || !Cf || !ref) { fprintf(stderr, "host alloc failed\n"); return -1; }

    srand(20260624);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rand_i4();
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = rand_i4();
    for (size_t i = 0; i < (size_t)M * nG; i++) as[i] = rand_scale();
    for (size_t i = 0; i < (size_t)N * nG; i++) bs[i] = rand_scale();

    /* host fp64 reference: per-group scaled accumulation */
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int g = 0; g < nG; g++) {
                long ip = 0;
                for (int k = 0; k < group; k++)
                    ip += (long)A[(size_t)m*K + (size_t)g*group + k] * (long)B[(size_t)n*K + (size_t)g*group + k];
                acc += (double)as[(size_t)m*nG + g] * (double)bs[(size_t)n*nG + g] * (double)ip;
            }
            ref[(size_t)m*N + n] = acc;
        }

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); return 2; }
    int rc = rocket_matmul_int4_groupwise(fd, M, K, N, A, B, as, bs, Cf, group);
    rocket_close(fd);
    if (rc) { fprintf(stderr, "rocket_matmul_int4_groupwise = %d\n", rc); return -1; }

    float max_abs = 0, max_rel = 0; long nbad = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        if (!isfinite(Cf[i])) { nbad++; continue; }
        double r = ref[i];
        float ad = (float)fabs((double)Cf[i] - r);
        float rd = ad / (float)(fabs(r) + 1e-6);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (rd > 1e-4f && ad > 1e-2f) nbad++;   /* fp32-accum tolerance vs fp64 ref */
    }
    printf("max_abs=%.4f max_rel=%.2e nbad=%ld -> %s\n",
           max_abs, max_rel, nbad, nbad == 0 ? "PASS" : "FAIL");

    free(A); free(B); free(as); free(bs); free(Cf); free(ref);
    return nbad == 0 ? 0 : 1;
}
