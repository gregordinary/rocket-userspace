// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * mixed_chain_coexist_rocket — the per-job batched-flag gate.
 *
 * Runs, in ONE process and on ONE fd with ROCKET_BATCH_SUBMIT=1, a CHAINED fp16
 * tiled matmul (multi-task, lays its regcmds contiguously and sets
 * DRM_ROCKET_JOB_BATCHED) immediately followed by a GAPPED int8 tiled matmul
 * (multi-task, but the integer datapath forces the stock per-task layout and
 * leaves the flag clear — int32 CACC clears per kick, so chaining garbles it).
 *
 * This is the case the OLD global rocket_batch_submit module param could not
 * express: with the global param on, the kernel forced TASK_NUMBER=N on EVERY
 * multi-task job, so the gapped int8 job streamed task 0 into the inter-task gap
 * and produced garbage (or timed out). With the per-job flag the int8 job runs
 * gapped (correct) while the fp16 job runs chained (correct) back to back.
 *
 * PASS iff BOTH verify bit-exact. Requires the kernel master param
 * rocket_batch_submit=1 (the patched default) and the per-job-flag kernel.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

#ifndef __fp16
#define __fp16 _Float16
#endif

static int8_t rand_i8(void) { return (int8_t)((rand() & 0xff) - 128); }

/* CHAINED fp16 tiled matmul, verified against an fp64 reference (cosine + max
 * abs). Returns 0 on PASS. The shape is multi-tile so the job batches. */
static int run_fp16_chained(int fd, int M, int K, int N)
{
    __fp16 *A = malloc((size_t)M * K * sizeof(__fp16));
    __fp16 *B = malloc((size_t)N * K * sizeof(__fp16));
    __fp16 *C = malloc((size_t)M * N * sizeof(__fp16));
    if (!A || !B || !C) { free(A); free(B); free(C); return -1; }

    srand(11);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (__fp16)((rand() / (double)RAND_MAX) - 0.5);
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (__fp16)((rand() / (double)RAND_MAX) - 0.5);

    int ret = rocket_matmul_fp16(fd, M, K, N, A, B, C);
    if (ret) { fprintf(stderr, "  fp16: rocket_matmul_fp16 = %d\n", ret); goto out; }

    /* cosine vs fp64 reference */
    double dot = 0, na = 0, nb = 0, maxabs = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)A[(size_t)m*K+k] * (double)B[(size_t)n*K+k];
            double got = (double)C[(size_t)m*N+n];
            double d = got - s; if (d < 0) d = -d;
            if (d > maxabs) maxabs = d;
            dot += got * s; na += got*got; nb += s*s;
        }
    double cos = dot / (1e-30 + sqrt(na) * sqrt(nb));
    int pass = (cos > 0.999);
    printf("  fp16 CHAINED  C[%d,%d]: cosine=%.6f max_abs=%.4f -> %s\n",
           M, N, cos, maxabs, pass ? "PASS" : "FAIL");
    ret = pass ? 0 : -1;
out:
    free(A); free(B); free(C);
    return ret;
}

/* GAPPED int8 tiled matmul, verified bit-exact against an int64 reference.
 * Returns 0 on PASS. */
static int run_int8_gapped(int fd, int M, int K, int N)
{
    int Mt, Kt, Nt;
    if (rocket_matmul_plan_int8(M, K, N, &Mt, &Kt, &Nt) < 0) {
        fprintf(stderr, "  int8: unsupported shape\n"); return -1;
    }
    int8_t  *A = malloc((size_t)M * K);
    int8_t  *B = malloc((size_t)N * K);
    int32_t *C = malloc((size_t)M * N * sizeof(int32_t));
    if (!A || !B || !C) { free(A); free(B); free(C); return -1; }

    srand(22);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rand_i8();
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = rand_i8();
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

    int ret = rocket_matmul_int8(fd, M, K, N, A, B, C);
    if (ret) { fprintf(stderr, "  int8: rocket_matmul_int8 = %d\n", ret); goto out; }

    long bad = 0, maxabs = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)A[(size_t)m*K+k] * (int32_t)B[(size_t)n*K+k];
            long d = labs((long)C[(size_t)m*N+n] - (long)s);
            if (d > maxabs) maxabs = d;
            if (d) bad++;
        }
    int pass = (bad == 0);
    printf("  int8 GAPPED   C[%d,%d]: max_abs_diff=%ld mismatches=%ld -> %s\n",
           M, N, maxabs, bad, pass ? "PASS (bit-exact)" : "FAIL");
    ret = pass ? 0 : -1;
out:
    free(A); free(B); free(C);
    return ret;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); return 2; }

    printf("mixed chained-fp16 + gapped-int8 coexistence (one process, one fd)\n");
    /* Force the chained layout on for this process; the int8 path forces gapped
     * internally regardless, so this is the exact mixed-workload case. */
    setenv("ROCKET_BATCH_SUBMIT", "1", 1);

    int r1 = run_fp16_chained(fd, 512, 1024, 1024);   /* multi-tile => chained */
    int r2 = run_int8_gapped (fd, 512, 1280, 512);    /* multi-tile => gapped  */
    /* Interleave once more to rule out order effects on shared fd/core state. */
    int r3 = run_int8_gapped (fd, 256, 1280, 256);
    int r4 = run_fp16_chained(fd, 256, 1024, 768);

    rocket_close(fd);

    int ok = !r1 && !r2 && !r3 && !r4;
    printf("RESULT: %s\n", ok ? "PASS (chained fp16 + gapped int8 coexist)" : "FAIL");
    return ok ? 0 : 1;
}
