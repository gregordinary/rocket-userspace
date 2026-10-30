// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * gemv_bw_probe.c — how fast does the RK3588 NPU stream a resident weight at decode M?
 *
 * A probe, not a gate. Decode is a GEMV: every token reads every weight once, so its
 * device rate is the weight-streaming bandwidth. The library keeps decode on the CPU on
 * evidence that is all fp16 and all whole-tile (a Whisper decoder step, M 4). An external
 * userspace (rknpu2_matmul_open, vendor driver) runs decode on the NPU with integer
 * weights resident in the tile layout and implies ~24 GB/s over three cores. This
 * measures our side: one task per job whose weight cube streams through the CBUF
 * (gen_matmul_fp16 / gen_matmul_int8), N split across `jobs` jobs submitted in one ioctl,
 * so the kernel places them on separate cores.
 *
 * The first run of each configuration is scored exactly (integer operands: fp16 in [-3, 3]
 * against fp32 out, int8 in [-127, 127] against int32 out); then `reps` timed runs, and
 * the median submit-to-fence time gives the weight bytes per second. That rate includes
 * one submit and fence a run, as a decode step would pay per matmul.
 *
 * Usage: gemv_bw_probe fp16|int8 K N [jobs [reps [M]]]     (jobs 1-3, reps 20, M 4)
 * Exit: 0 exact, 1 not, 2 no NPU.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "test_fill.h"

#define MAXJ 3

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s fp16|int8 K N [jobs [reps [M]]]\n", argv[0]); return 1; }
    const int i8 = !strcmp(argv[1], "int8");
    const int K = atoi(argv[2]), N = atoi(argv[3]);
    const int J = argc > 4 ? atoi(argv[4]) : 1, reps = argc > 5 ? atoi(argv[5]) : 20;
    const int M = argc > 6 ? atoi(argv[6]) : 4;
    const int es = i8 ? 1 : 2, c2 = i8 ? 16 : 8;
    if (J < 1 || J > MAXJ || K % 32 || N % (32 * J) || M < 1 || M > 16) {
        fprintf(stderr, "K%%32, N%%(32*jobs), jobs 1-3, M 1-16\n");
        return 1;
    }
    const int Nj = N / J;
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }

    rocket_bo in = { 0 }, rc[MAXJ] = { { 0 } }, w[MAXJ] = { { 0 } }, out[MAXJ] = { { 0 } };
    rocket_task_desc td[MAXJ];
    uint32_t ih[MAXJ][3];
    rocket_job_desc jobs[MAXJ];
    int bad = 0;

    void *A = malloc((size_t)M * K * es), *B = malloc((size_t)N * K * es);
    if (!A || !B || rocket_bo_alloc32(fd, (size_t)16 * K * es + 4096, &in)) { printf("alloc failed\n"); return 1; }
    if (i8) {
        tf_fill_i8(A, (size_t)M * K, 0x6100, -127, 127);
        tf_fill_i8(B, (size_t)N * K, 0x6200, -127, 127);
    } else {
        tf_fill_f16_int(A, (size_t)M * K, 0x6100, -3, 3);
        tf_fill_f16_int(B, (size_t)N * K, 0x6200, -3, 3);
    }
    rocket_bo_prep(fd, &in, 1, 0);
    memset(in.ptr, 0, in.size);
    for (int m = 1; m <= M; m++)
        for (int k = 1; k <= K; k++) {
            size_t at = feature_data(K, M, 1, c2, k, m, 1), src = (size_t)(m - 1) * K + k - 1;
            if (i8) ((int8_t *)in.ptr)[at] = ((int8_t *)A)[src];
            else    ((_Float16 *)in.ptr)[at] = ((_Float16 *)A)[src];
        }
    rocket_bo_fini(fd, &in);

    for (int j = 0; j < J; j++) {
        uint64_t ops[256] = { 0 };
        if (rocket_bo_alloc32(fd, 4096, &rc[j]) || rocket_bo_alloc32(fd, (size_t)Nj * K * es, &w[j]) ||
            rocket_bo_alloc32(fd, (size_t)16 * Nj * 4, &out[j])) { printf("alloc failed\n"); return 1; }
        rocket_bo_prep(fd, &w[j], 1, 0);
        memset(w[j].ptr, 0, w[j].size);
        for (int n = 1; n <= Nj; n++)
            for (int k = 1; k <= K; k++) {
                size_t src = (size_t)(j * Nj + n - 1) * K + k - 1;
                if (i8) ((int8_t *)w[j].ptr)[weight_int8(K, n, k)] = ((int8_t *)B)[src];
                else    ((_Float16 *)w[j].ptr)[weight_fp16(K, n, k)] = ((_Float16 *)B)[src];
            }
        rocket_bo_fini(fd, &w[j]);
        matmul_params_t p = {
            .m = (uint16_t)M, .k = (uint16_t)K, .n = (uint16_t)Nj,
            .input_dma = (uint32_t)in.dma_address, .weights_dma = (uint32_t)w[j].dma_address,
            .output_dma = (uint32_t)out[j].dma_address, .tasks = ops, .fp32tofp16 = 0,
        };
        int g = i8 ? gen_matmul_int8(&p) : gen_matmul_fp16(&p);
        if (g != 0) { printf("generator refused %dx%dx%d (%d)\n", M, K, Nj, g); return 1; }
        rocket_bo_prep(fd, &rc[j], 1, 0);
        memcpy(rc[j].ptr, ops, (size_t)p.task_count * 8);
        rocket_bo_fini(fd, &rc[j]);
        td[j] = (rocket_task_desc){ (uint32_t)rc[j].dma_address, p.task_count };
        ih[j][0] = in.handle; ih[j][1] = w[j].handle; ih[j][2] = rc[j].handle;
        jobs[j] = (rocket_job_desc){ &td[j], 1, ih[j], 3, &out[j].handle, 1 };
    }

    double *t = calloc((size_t)reps + 1, sizeof(double));
    for (int r = 0; r <= reps; r++) {
        if (r == 0)
            for (int j = 0; j < J; j++) {
                rocket_bo_prep(fd, &out[j], 1, 0);
                memset(out[j].ptr, 0xFF, out[j].size);
                rocket_bo_fini(fd, &out[j]);
            }
        double t0 = now_ms();
        int rcode = rocket_submit_jobs(fd, jobs, (uint32_t)J);
        for (int j = 0; j < J && rcode == 0; j++) rcode = rocket_bo_prep(fd, &out[j], 0, 4000000000ull);
        t[r] = now_ms() - t0;
        if (rcode != 0) { printf("submit or wait failed (%d)\n", rcode); return 1; }
        if (r == 0) {
            long wrong = 0, unw = 0;
            for (int j = 0; j < J; j++)
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < Nj; n++) {
                        size_t at = feature_data(Nj, M, 1, 4, n + 1, m + 1, 1);
                        double want = 0, got;
                        for (int k = 0; k < K; k++) {
                            size_t ai = (size_t)m * K + k, bi = (size_t)(j * Nj + n) * K + k;
                            want += i8 ? (double)((int8_t *)A)[ai] * ((int8_t *)B)[bi]
                                       : (double)((_Float16 *)A)[ai] * (double)((_Float16 *)B)[bi];
                        }
                        if (i8) { int32_t v = ((int32_t *)out[j].ptr)[at]; got = v; unw += (uint32_t)v == 0xFFFFFFFFu; }
                        else    { float v = ((float *)out[j].ptr)[at]; got = v; unw += isnan(v); }
                        wrong += got != want;
                    }
            printf("%s M %d K %d N %d over %d job(s): %ld of %d wrong, %ld unwritten, first run %.3f ms\n",
                   argv[1], M, K, N, J, wrong, M * N, unw, t[0]);
            bad = wrong != 0;
        }
        for (int j = 0; j < J; j++) rocket_bo_fini(fd, &out[j]);
    }
    qsort(t + 1, (size_t)reps, sizeof(double), cmp_d);
    double med = t[1 + reps / 2], bytes = (double)N * K * es;
    printf("  %d reps: median %.3f ms (min %.3f, max %.3f), weight %.1f MB -> %.2f GB/s\n", reps, med,
           t[1], t[reps], bytes / 1e6, bytes / med / 1e6);
    return bad;
}
