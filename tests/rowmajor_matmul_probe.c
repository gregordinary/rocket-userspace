// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rowmajor_matmul_probe.c — can one RK3588 matmul task read A row-major and write C
 * row-major, with no host scatter or de-scatter of either?
 *
 * A probe, not a gate. The library scatters A into
 * the C2=8 feature cube and de-scatters C from the C2=4 output cube on the host. allbilly's
 * generators (experimental/gemm_int16.py, examples/gemm.py) drive the same silicon with A
 * as plain [M][K] rows and C as plain [M][N] rows, by strides alone, and are exact under the
 * vendor driver up to K 4096 once their own caps are lifted. This transcribes that form onto
 * gen_matmul_fp16's program by patching six registers:
 *
 *   CNA  CONV_CON1        bit 29 set (GROUP_LINE_OFF)
 *   CNA  DMA_CON1         line stride K/8, one padded row of A
 *   CNA  DMA_CON2         surface stride 0
 *   DPU  DST_SURF_STRIDE  one 16-byte atom
 *   DPU  DATA_CUBE_NOTCH  N/4 - 1 in both halves
 *   DPU  SURFACE_ADD      four atoms
 *
 * Each shape runs two arms, one single-task job each, fp16 in and fp32 out:
 *
 *   cube      the program as it ships, A scattered and C de-scattered on the host
 *   rowmajor  the patched program, A and C in plain row-major buffers
 *
 * Operands are integers in [-3, 3], so every fp32 partial sum is exact and each element is
 * compared for equality with a double reference. The output is prefilled with 0xFF (a NaN),
 * so an unwritten element cannot pass, and a job the kernel retired at its watchdog is
 * reported. The shapes cover K up to 4096 and N up to 2048, all M, K, N different.
 *
 * After the correctness arms, a timing pass runs each arm interleaved and prints the median
 * submit-to-fence time. It is device time plus a fixed submit cost, not a price for the
 * host work the form removes.
 *
 * What a green result does NOT show: int8 or the other dtypes, a K split across tasks, a
 * tiled or multi-core job, resident weights, or what the form is worth to a model.
 *
 * Usage: rowmajor_matmul_probe [M K N]   (K%32, N%32, M%4)
 * Exit: 0 every rowmajor arm exact, 1 otherwise, 2 no NPU.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "test_fill.h"

struct shape { int M, K, N; };

static const struct shape ladder[] = {
    { 64, 128, 96 }, { 100, 256, 64 }, { 48, 384, 256 }, { 36, 96, 384 },
    { 256, 320, 160 }, { 64, 512, 128 }, { 32, 1024, 64 }, { 16, 2048, 64 },
    { 8, 4096, 32 }, { 32, 64, 1024 }, { 20, 64, 2048 }, { 64, 256, 512 },
};

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* Rewrite one register's value in a generated program. OR-in when `orbits` is set.
 * Returns 0, or -1 when the program never writes that register. */
static int patch(uint64_t *ops, int n, uint16_t target, uint16_t reg, uint32_t value, int orbits)
{
    for (int i = 0; i < n; i++) {
        if ((uint16_t)(ops[i] >> 48) != target || (uint16_t)(ops[i] & 0xFFFF) != reg) continue;
        uint32_t v = orbits ? (uint32_t)(ops[i] >> 16) | value : value;
        ops[i] = NPUOP(target, v, reg);
        return 0;
    }
    return -1;
}

struct bufs { rocket_bo rc, in, w, out; };

static int run_arm(int fd, struct bufs *b, const struct shape *s, int rowmajor,
                   const _Float16 *A, const _Float16 *B, const double *want, int quiet,
                   double *ms, int *timed_out)
{
    const int M = s->M, K = s->K, N = s->N;
    uint64_t ops[256] = { 0 };
    matmul_params_t p = {
        .m = (uint16_t)M, .k = (uint16_t)K, .n = (uint16_t)N,
        .input_dma = (uint32_t)b->in.dma_address, .weights_dma = (uint32_t)b->w.dma_address,
        .output_dma = (uint32_t)b->out.dma_address, .tasks = ops, .fp32tofp16 = 0,
    };
    if (gen_matmul_fp16(&p) != 0) { printf("  %dx%dx%d: generator refused\n", M, K, N); return -1; }
    const int n = (int)p.task_count;
    if (rowmajor) {
        const uint32_t notch = (uint32_t)(N / 4 - 1);
        if (patch(ops, n, OP_REG_CNA, CNA_CONV_CON1, 1u << 29, 1) ||
            patch(ops, n, OP_REG_CNA, CNA_DMA_CON1, (uint32_t)(K / 8), 0) ||
            patch(ops, n, OP_REG_CNA, CNA_DMA_CON2, 0, 0) ||
            patch(ops, n, OP_REG_DPU, DPU_DST_SURF_STRIDE, 1u << 4, 0) ||
            patch(ops, n, OP_REG_DPU, DPU_DATA_CUBE_NOTCH_ADDR, (notch << 16) | notch, 0) ||
            patch(ops, n, OP_REG_DPU, DPU_SURFACE_ADD, 4u << 4, 0)) {
            printf("  %dx%dx%d: a register to patch is missing from the program\n", M, K, N);
            return -1;
        }
    }

    rocket_bo_prep(fd, &b->rc, 1, 0);
    memcpy(b->rc.ptr, ops, (size_t)n * sizeof(uint64_t));
    rocket_bo_fini(fd, &b->rc);
    rocket_bo_prep(fd, &b->in, 1, 0);
    rocket_bo_prep(fd, &b->w, 1, 0);
    memset(b->in.ptr, 0, b->in.size);
    memset(b->w.ptr, 0, b->w.size);
    {
        _Float16 *id = b->in.ptr, *wd = b->w.ptr;
        if (rowmajor) memcpy(id, A, (size_t)M * K * sizeof(_Float16));
        else
            for (int m = 1; m <= M; m++)
                for (int k = 1; k <= K; k++)
                    id[feature_data(K, M, 1, 8, k, m, 1)] = A[(size_t)(m - 1) * K + (k - 1)];
        for (int nn = 1; nn <= N; nn++)
            for (int k = 1; k <= K; k++)
                wd[weight_fp16(K, nn, k)] = B[(size_t)(nn - 1) * K + (k - 1)];
    }
    rocket_bo_fini(fd, &b->in);
    rocket_bo_fini(fd, &b->w);
    rocket_bo_prep(fd, &b->out, 1, 0);
    memset(b->out.ptr, 0xFF, b->out.size);
    rocket_bo_fini(fd, &b->out);

    uint32_t in_h[] = { b->in.handle, b->w.handle, b->rc.handle }, out_h[] = { b->out.handle };
    uint64_t slow0 = rocket_fence_wait_slow_count();
    double t0 = now_ms();
    int rc = rocket_submit_matmul(fd, &b->rc, (uint32_t)n, in_h, 3, out_h, 1, 6000);
    if (rc == 0) rc = rocket_bo_prep(fd, &b->out, 0, 2000000000ull);
    *ms = now_ms() - t0;
    *timed_out = rocket_fence_wait_slow_count() != slow0;
    if (rc != 0) { printf("  %dx%dx%d: submit or wait failed (%d)\n", M, K, N, rc); return -1; }

    const float *o = b->out.ptr;
    long wrong = 0, unwritten = 0, first = -1;
    for (int m = 0; m < M; m++)
        for (int nn = 0; nn < N; nn++) {
            size_t at = rowmajor ? (size_t)m * N + nn
                                 : (size_t)feature_data(N, M, 1, 4, nn + 1, m + 1, 1);
            double got = o[at];
            if (isnan(got)) unwritten++;
            if (got != want[(size_t)m * N + nn]) { if (first < 0) first = (long)m * N + nn; wrong++; }
        }
    rocket_bo_fini(fd, &b->out);
    if (!quiet) {
        printf("  %-8s %4dx%4dx%4d: %6ld of %6d wrong, %6ld unwritten, %.3f ms%s", rowmajor ?
               "rowmajor" : "cube", M, K, N, wrong, M * N, unwritten, *ms,
               *timed_out ? " [WATCHDOG]" : "");
        if (wrong) printf(", first (m %ld, n %ld)", first / N, first % N);
        printf("\n");
    }
    return (wrong || *timed_out) ? 1 : 0;
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    struct shape one;
    const struct shape *shapes = ladder;
    int nshapes = (int)(sizeof(ladder) / sizeof(ladder[0])), bad = 0, timing = 20;

    if (argc >= 4) {
        one.M = atoi(argv[1]); one.K = atoi(argv[2]); one.N = atoi(argv[3]);
        if (one.M < 4 || one.M % 4 || one.K % 32 || one.K < 32 || one.N % 32 || one.N < 32) {
            fprintf(stderr, "usage: %s [M K N], M%%4, K%%32, N%%32\n", argv[0]);
            return 1;
        }
        shapes = &one; nshapes = 1;
    }
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }
    printf("== row-major matmul probe: %d shape(s), fp16 in, fp32 out, one task each ==\n", nshapes);

    for (int si = 0; si < nshapes; si++) {
        const struct shape *s = &shapes[si];
        const int M = s->M, K = s->K, N = s->N;
        struct bufs b = { 0 };
        _Float16 *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
        double *want = malloc((size_t)M * N * sizeof(double));
        if (!A || !B || !want ||
            rocket_bo_alloc32(fd, 4096, &b.rc) || rocket_bo_alloc32(fd, (size_t)M * K * 2, &b.in) ||
            rocket_bo_alloc32(fd, (size_t)N * K * 2, &b.w) ||
            rocket_bo_alloc32(fd, (size_t)M * N * 4, &b.out)) {
            printf("  %dx%dx%d: allocation failed\n", M, K, N); bad++; goto next;
        }
        tf_fill_f16_int(A, (size_t)M * K, 0x3100 + (uint64_t)si, -3, 3);
        tf_fill_f16_int(B, (size_t)N * K, 0x3200 + (uint64_t)si, -3, 3);
        for (int m = 0; m < M; m++)
            for (int nn = 0; nn < N; nn++) {
                double acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (double)A[(size_t)m * K + k] * (double)B[(size_t)nn * K + k];
                want[(size_t)m * N + nn] = acc;
            }
        {
            double ms; int to, rc_c, rc_r;
            rc_c = run_arm(fd, &b, s, 0, A, B, want, 0, &ms, &to);
            rc_r = run_arm(fd, &b, s, 1, A, B, want, 0, &ms, &to);
            if (rc_r != 0) bad++;
            if (timing > 0 && rc_c == 0 && rc_r == 0) {
                double *tc = calloc((size_t)timing, sizeof(double)), *tr = calloc((size_t)timing, sizeof(double));
                for (int r = 0; r < timing; r++) {
                    int first = r & 1;
                    run_arm(fd, &b, s, first, A, B, want, 1, first ? &tr[r] : &tc[r], &to);
                    run_arm(fd, &b, s, !first, A, B, want, 1, first ? &tc[r] : &tr[r], &to);
                }
                qsort(tc, (size_t)timing, sizeof(double), cmp_d);
                qsort(tr, (size_t)timing, sizeof(double), cmp_d);
                printf("  timing %4dx%4dx%4d, %d interleaved reps, submit to fence: cube %.4f ms, "
                       "rowmajor %.4f ms (medians)\n", M, K, N, timing, tc[timing / 2], tr[timing / 2]);
                free(tc); free(tr);
            }
        }
    next:
        rocket_bo_free(fd, &b.rc); rocket_bo_free(fd, &b.in);
        rocket_bo_free(fd, &b.w); rocket_bo_free(fd, &b.out);
        free(A); free(B); free(want);
    }
    printf("== rowmajor %s ==\n", bad ? "NOT exact on every shape" : "exact on every shape");
    rocket_close(fd);
    return bad ? 1 : 0;
}
