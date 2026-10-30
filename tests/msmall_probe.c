// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * msmall_probe.c — does a one-task fp16 or int8 matmul compute at every M, not only at M%4 == 0?
 *
 * The CNA reads a feature cube in lines of four rows and reaches the next channel surface
 * by the line stride plus the surface stride, so a surface of H packed rows needs a surface
 * stride of width*(H - 4), negative below four rows (cna_feature_surf_stride()). The
 * truncating line*(H/4 - 1) the generators once wrote, clamped at 0, is that value only at
 * H%4 == 0, and every other M came back wrong; that was the library's M%4 rule. An external
 * userspace (rknpu2_matmul_open, vendor driver) runs M 1..8 with M along the CNA width at
 * height 1 and a surface stride of -3M, the same formula at width M and height 1.
 * gen_matmul_fp16's RE knobs select the arm, one process each, since it reads them once:
 *
 *   (none)                    width 1, height M, the packed surface stride (shipped)
 *   ROCKET_MM_SURF_TRUNC=1    width 1, height M, the old truncating stride clamped at 0
 *   ROCKET_MM_MWIDTH=1        width M, height 1, the packed surface stride (-3M)
 *   ROCKET_MM_GRAINS=n        feature grains n instead of M+1, with any of the above
 *   ROCKET_MM_LINE_STRIDE=n   the line stride raw, and ROCKET_MM_SURF_PER_M=s the surface
 *                             stride s*M: the external program is MWIDTH, 0x200, -3, grains 2
 *   MSP_ROWMAJOR=1            A read as plain rows at height M (matmul_params_t.a_row_atoms,
 *                             GROUP_LINE_OFF), the external's standard program
 *   MSP_DT=int8               gen_matmul_int8 instead (C2=16 input, int32 output; N%32)
 *   MSP_KACC=1                fp16 K-accumulation, the library's ki > 0 task: a first job
 *                             contracts K/2 into an fp16 partial, a second contracts the
 *                             rest with accumulate=1 and EW-adds the partial (ERDMA read
 *                             at height M), fp16 out. The ERDMA surface stride and notch
 *                             are M; MSP_EW_FLOOR=1 sets them to MAX(M, 12) instead
 *                             (ROCKET_EW_SURF/_NOTCH), the floor that misread every M < 12
 *                             at N > 16
 *
 * The input and output cubes are the same bytes in every arm. Operands are integers in
 * [-3, 3], so the fp32 output is exact and compared for equality; the output is stamped
 * 0xFF (a NaN) so an unwritten element cannot pass. Each M runs twice with different
 * operands, so a surface left over from an earlier task cannot pass either.
 *
 * Usage: msmall_probe [K N]     (default K 128, N 64)
 * Exit: 0 every M exact, 1 otherwise, 2 no NPU.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "test_fill.h"

static const int Ms[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 33, 63, 64 };
#define MAX_M 64

static int g_rowmajor, g_int8, g_kacc, g_ew_floor;

/* Scatter an M x K fp16 A and an N x K B into a fresh input and weight cube. */
static void load_ab(int fd, rocket_bo *in, rocket_bo *w, const _Float16 *A, const _Float16 *B,
                    int M, int K, int lda, int N)
{
    rocket_bo_prep(fd, in, 1, 0);
    rocket_bo_prep(fd, w, 1, 0);
    memset(in->ptr, 0, in->size);
    memset(w->ptr, 0, w->size);
    for (int m = 1; m <= M; m++)
        for (int k = 1; k <= K; k++)
            ((_Float16 *)in->ptr)[feature_data(K, M, 1, 8, k, m, 1)] = A[(size_t)(m - 1) * lda + k - 1];
    for (int n = 1; n <= N; n++)
        for (int k = 1; k <= K; k++)
            ((_Float16 *)w->ptr)[weight_fp16(K, n, k)] = B[(size_t)(n - 1) * lda + k - 1];
    rocket_bo_fini(fd, in);
    rocket_bo_fini(fd, w);
}

/* MSP_KACC: two jobs, the second accumulating onto the first's fp16 partial. */
static int run_kacc(int fd, int M, int K, int N, uint64_t seed)
{
    const int K1 = K / 2, K2 = K - K1;
    rocket_bo rc = { 0 }, in1 = { 0 }, w1 = { 0 }, in2 = { 0 }, w2 = { 0 }, part = { 0 }, out = { 0 };
    _Float16 *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    int bad = -1;
    if (!A || !B || rocket_bo_alloc32(fd, 4096, &rc) ||
        rocket_bo_alloc32(fd, (size_t)MAX_M * K * 2, &in1) || rocket_bo_alloc32(fd, (size_t)N * K * 2, &w1) ||
        rocket_bo_alloc32(fd, (size_t)MAX_M * K * 2, &in2) || rocket_bo_alloc32(fd, (size_t)N * K * 2, &w2) ||
        rocket_bo_alloc32(fd, (size_t)MAX_M * N * 2 + 4096, &part) ||
        rocket_bo_alloc32(fd, (size_t)MAX_M * N * 2 + 4096, &out))
        goto done;
    tf_fill_f16_int(A, (size_t)M * K, seed, -3, 3);
    tf_fill_f16_int(B, (size_t)N * K, seed + 1, -3, 3);
    load_ab(fd, &in1, &w1, A, B, M, K1, K, N);
    load_ab(fd, &in2, &w2, A + K1, B + K1, M, K2, K, N);
    rocket_bo_prep(fd, &part, 1, 0); memset(part.ptr, 0xFF, part.size); rocket_bo_fini(fd, &part);
    rocket_bo_prep(fd, &out, 1, 0);  memset(out.ptr, 0xFF, out.size);  rocket_bo_fini(fd, &out);

    char ev[32];
    snprintf(ev, sizeof ev, "%d", M > 12 ? M : 12);
    if (g_ew_floor) { setenv("ROCKET_EW_SURF", ev, 1); setenv("ROCKET_EW_NOTCH", ev, 1); }
    uint64_t slow0 = rocket_fence_wait_slow_count();
    for (int pass = 0; pass < 2; pass++) {
        uint64_t ops[256] = { 0 };
        matmul_params_t p = {
            .m = (uint16_t)M, .k = (uint16_t)(pass ? K2 : K1), .n = (uint16_t)N,
            .input_dma = (uint32_t)(pass ? in2 : in1).dma_address,
            .weights_dma = (uint32_t)(pass ? w2 : w1).dma_address,
            .output_dma = (uint32_t)(pass ? out : part).dma_address, .tasks = ops, .fp32tofp16 = 1,
            .accumulate = (uint8_t)pass, .add_dma = pass ? (uint32_t)part.dma_address : 0,
        };
        if (gen_matmul_fp16(&p) != 0) { printf("  M %2d: generator refused\n", M); goto done; }
        rocket_bo_prep(fd, &rc, 1, 0);
        memcpy(rc.ptr, ops, (size_t)p.task_count * 8);
        rocket_bo_fini(fd, &rc);
        uint32_t ih[] = { (pass ? in2 : in1).handle, (pass ? w2 : w1).handle, rc.handle, part.handle };
        uint32_t oh[] = { (pass ? out : part).handle };
        int r = rocket_submit_matmul(fd, &rc, p.task_count, ih, pass ? 4 : 3, oh, 1, 6000);
        if (r == 0) r = rocket_bo_prep(fd, pass ? &out : &part, 0, 2000000000ull);
        if (r == 0) rocket_bo_fini(fd, pass ? &out : &part);
        if (r != 0) { printf("  M %2d: pass %d submit or wait failed (%d)\n", M, pass, r); goto done; }
    }
    int slow = rocket_fence_wait_slow_count() != slow0;
    rocket_bo_prep(fd, &part, 0, 2000000000ull);
    rocket_bo_prep(fd, &out, 0, 2000000000ull);
    long wrong = 0, unwritten = 0, rows_bad = 0, pwrong = 0;
    for (int m = 0; m < M; m++) {
        long rw = 0;
        for (int n = 0; n < N; n++) {
            double want = 0, pwant = 0;
            for (int k = 0; k < K; k++) {
                double t = (double)A[(size_t)m * K + k] * (double)B[(size_t)n * K + k];
                want += t;
                if (k < K1) pwant += t;
            }
            int oi = feature_data(N, M, 1, 8, n + 1, m + 1, 1);
            float got = (float)((_Float16 *)out.ptr)[oi], pg = (float)((_Float16 *)part.ptr)[oi];
            if (isnan(got)) unwritten++;
            if (got != want) rw++;
            if (pg != pwant) pwrong++;
        }
        wrong += rw;
        rows_bad += rw != 0;
    }
    rocket_bo_fini(fd, &part);
    rocket_bo_fini(fd, &out);
    printf("  M %2d: %5ld of %5d wrong (%ld of %d rows), %ld unwritten, partial %ld wrong%s\n", M,
           wrong, M * N, rows_bad, M, unwritten, pwrong, slow ? " [SLOW FENCE]" : "");
    bad = (wrong || pwrong || slow) ? 1 : 0;
done:
    rocket_bo_free(fd, &rc); rocket_bo_free(fd, &in1); rocket_bo_free(fd, &w1);
    rocket_bo_free(fd, &in2); rocket_bo_free(fd, &w2); rocket_bo_free(fd, &part);
    rocket_bo_free(fd, &out);
    free(A); free(B);
    return bad;
}

static int run(int fd, int M, int K, int N, uint64_t seed)
{
    rocket_bo rc = { 0 }, in = { 0 }, w = { 0 }, out = { 0 };
    _Float16 *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    uint64_t ops[256] = { 0 };
    int bad = -1;
    if (!A || !B || rocket_bo_alloc32(fd, 4096, &rc) || rocket_bo_alloc32(fd, (size_t)MAX_M * K * 2, &in) ||
        rocket_bo_alloc32(fd, (size_t)N * K * 2, &w) || rocket_bo_alloc32(fd, (size_t)MAX_M * N * 4, &out))
        goto done;
    tf_fill_f16_int(A, (size_t)M * K, seed, -3, 3);
    tf_fill_f16_int(B, (size_t)N * K, seed + 1, -3, 3);

    matmul_params_t p = {
        .m = (uint16_t)M, .k = (uint16_t)K, .n = (uint16_t)N,
        .input_dma = (uint32_t)in.dma_address, .weights_dma = (uint32_t)w.dma_address,
        .output_dma = (uint32_t)out.dma_address, .tasks = ops, .fp32tofp16 = 0,
        .a_row_atoms = g_rowmajor ? (uint32_t)(K * 2 / 16) : 0,
    };
    if ((g_int8 ? gen_matmul_int8(&p) : gen_matmul_fp16(&p)) != 0) {
        printf("  M %2d: generator refused\n", M);
        goto done;
    }

    rocket_bo_prep(fd, &rc, 1, 0);
    memcpy(rc.ptr, ops, (size_t)p.task_count * 8);
    rocket_bo_fini(fd, &rc);
    rocket_bo_prep(fd, &in, 1, 0);
    rocket_bo_prep(fd, &w, 1, 0);
    memset(in.ptr, 0, in.size);
    memset(w.ptr, 0, w.size);
    for (int m = 1; m <= M; m++)
        for (int k = 1; k <= K; k++)
            if (g_int8)
                ((int8_t *)in.ptr)[feature_data(K, M, 1, 16, k, m, 1)] =
                    (int8_t)A[(size_t)(m - 1) * K + k - 1];
            else
                ((_Float16 *)in.ptr)[g_rowmajor ? (size_t)(m - 1) * K + k - 1
                                                : (size_t)feature_data(K, M, 1, 8, k, m, 1)] =
                    A[(size_t)(m - 1) * K + k - 1];
    for (int n = 1; n <= N; n++)
        for (int k = 1; k <= K; k++)
            if (g_int8)
                ((int8_t *)w.ptr)[weight_int8(K, n, k)] = (int8_t)B[(size_t)(n - 1) * K + k - 1];
            else
                ((_Float16 *)w.ptr)[weight_fp16(K, n, k)] = B[(size_t)(n - 1) * K + k - 1];
    rocket_bo_fini(fd, &in);
    rocket_bo_fini(fd, &w);
    rocket_bo_prep(fd, &out, 1, 0);
    memset(out.ptr, 0xFF, out.size);
    rocket_bo_fini(fd, &out);

    uint32_t ih[] = { in.handle, w.handle, rc.handle }, oh[] = { out.handle };
    uint64_t slow0 = rocket_fence_wait_slow_count();
    int r = rocket_submit_matmul(fd, &rc, p.task_count, ih, 3, oh, 1, 6000);
    if (r == 0) r = rocket_bo_prep(fd, &out, 0, 2000000000ull);
    if (r != 0) { printf("  M %2d: submit or wait failed (%d)\n", M, r); goto done; }
    int slow = rocket_fence_wait_slow_count() != slow0;

    long wrong = 0, unwritten = 0, rows_bad = 0;
    for (int m = 0; m < M; m++) {
        long rw = 0;
        for (int n = 0; n < N; n++) {
            double want = 0;
            for (int k = 0; k < K; k++) want += (double)A[(size_t)m * K + k] * (double)B[(size_t)n * K + k];
            int oi = feature_data(N, M, 1, 4, n + 1, m + 1, 1);
            float got = g_int8 ? (float)((int32_t *)out.ptr)[oi] : ((float *)out.ptr)[oi];
            if (isnan(got)) unwritten++;
            if (got != want) rw++;
        }
        wrong += rw;
        rows_bad += rw != 0;
    }
    rocket_bo_fini(fd, &out);
    printf("  M %2d: %5ld of %5d wrong (%ld of %d rows), %ld unwritten%s\n", M, wrong, M * N,
           rows_bad, M, unwritten, slow ? " [SLOW FENCE]" : "");
    bad = (wrong || slow) ? 1 : 0;
done:
    rocket_bo_free(fd, &rc); rocket_bo_free(fd, &in); rocket_bo_free(fd, &w); rocket_bo_free(fd, &out);
    free(A); free(B);
    return bad;
}

int main(int argc, char **argv)
{
    int K = 128, N = 64, bad = 0;
    if (argc >= 3) { K = atoi(argv[1]); N = atoi(argv[2]); }
    if (K < 32 || K % 32 || N < 16 || N % 16) { fprintf(stderr, "usage: %s [K N]\n", argv[0]); return 1; }
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }
    const char *mw = getenv("ROCKET_MM_MWIDTH"), *ss = getenv("ROCKET_MM_SURF_TRUNC"),
               *gr = getenv("ROCKET_MM_GRAINS"), *ls = getenv("ROCKET_MM_LINE_STRIDE"),
               *sp = getenv("ROCKET_MM_SURF_PER_M"), *rm = getenv("MSP_ROWMAJOR");
    g_rowmajor = rm && atoi(rm);
    g_int8 = getenv("MSP_DT") && strcmp(getenv("MSP_DT"), "int8") == 0;
    g_kacc = getenv("MSP_KACC") && atoi(getenv("MSP_KACC"));
    g_ew_floor = getenv("MSP_EW_FLOOR") && atoi(getenv("MSP_EW_FLOOR"));
    if (g_kacc && (g_int8 || g_rowmajor || K % 64 || K > 224)) {
        /* |sum| <= 9K stays an exact fp16 integer below 2048 */
        fprintf(stderr, "kacc: fp16 cube A only, K%%64 and K <= 224 (an exact fp16 sum)\n");
        return 1;
    }
    if (g_int8 && (g_rowmajor || N % 32)) { fprintf(stderr, "int8: cube A only, N%%32\n"); return 1; }
    printf("== M-geometry probe: %s, K %d, N %d, %s, %s, surface stride %s, grains %s, line %s ==\n",
           g_kacc ? (g_ew_floor ? "fp16 K-accumulation, ERDMA stride MAX(M, 12)" : "fp16 K-accumulation")
                  : g_int8 ? "int8" : "fp16", K, N, mw && atoi(mw) ? "M along the width" : "M along the height",
           g_rowmajor ? "A as rows" : "A as a cube",
           sp ? sp : ss && atoi(ss) ? "truncated" : "packed", gr ? gr : "M+1",
           ls ? ls : "4*width");
    for (unsigned i = 0; i < sizeof Ms / sizeof Ms[0]; i++)
        for (int rep = 0; rep < 2; rep++)
            bad |= (g_kacc ? run_kacc : run)(fd, Ms[i], K, N,
                                              0x700 + (uint64_t)Ms[i] * 16 + (uint64_t)rep) != 0;
    printf("== %s ==\n", bad ? "NOT exact at every M" : "exact at every M");
    rocket_close(fd);
    return bad ? 1 : 0;
}
