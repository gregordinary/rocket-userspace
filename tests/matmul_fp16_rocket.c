// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_fp16_rocket.c — one gen_gemm_fp16() task on the NPU, against the CPU.
 *
 *     matmul_fp16_rocket <M> <K> <N>
 *
 * Builds C[M][N] = A[M][K] * B[N][K]^T as a single register program, with no tiling, so
 * the shape must fit one task: M a multiple of 4 (or 1), K a multiple of 32, N a multiple
 * of 16, and the operands within the CBUF (the generator refuses otherwise). The operands
 * are small integers, 0 to 9, so every product and every partial sum the shapes reach is
 * an exact fp32 integer and the fp16 result is one rounding of it: the device must match
 * the host bit for bit.
 *
 * ROCKET_TEST_SEED fixes the operands; the seed in use is printed either way.
 *
 * Exit status: 0 when every element matches, 1 on a mismatch or a failed step, 2 with no
 * device.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

/* The operands' generator: xorshift32, reduced to 0..9. */
static uint32_t rng_state;

static int next_digit(void)
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return (int)(x % 10u);
}

static double ms_since(const struct timespec *t0)
{
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0->tv_sec) * 1e3 + (double)(t1.tv_nsec - t0->tv_nsec) * 1e-6;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <M> <K> <N>\n", argv[0]);
        return 1;
    }
    const int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    if (M <= 0 || K <= 0 || N <= 0) {
        fprintf(stderr, "M, K and N must be positive\n");
        return 1;
    }

    uint64_t program[256];
    gemm_params_t p;
    memset(&p, 0, sizeof p);
    p.m = (uint16_t)M; p.k = (uint16_t)K; p.n = (uint16_t)N;
    p.tasks = program;
    p.fp16_out = 1;

    const int fd = rocket_open();
    if (fd < 0) {
        printf("no NPU device (%d): skipped\n", fd);
        return 2;
    }

    /* Buffers. The first one holds IOVA 0, which no program may point at: a register
     * program at address 0 runs as an empty one and the output is never written. */
    struct { rocket_bo zero, prog, a, b, c; } bo;
    memset(&bo, 0, sizeof bo);
    _Float16 *A = NULL, *B = NULL;
    float *ref = NULL;
    long wrong = 0;
    int status = 1;
    if (rocket_bo_alloc(fd, 4096, &bo.zero) ||
        rocket_bo_alloc(fd, sizeof program, &bo.prog) ||
        rocket_bo_alloc(fd, (size_t)M * K * sizeof(_Float16), &bo.a) ||
        rocket_bo_alloc(fd, (size_t)N * K * sizeof(_Float16), &bo.b) ||
        rocket_bo_alloc(fd, (size_t)M * N * sizeof(_Float16), &bo.c)) {
        fprintf(stderr, "buffer allocation failed\n");
        goto close;
    }
    /* Every address register is 32 bits wide. */
    if ((bo.prog.dma_address | bo.a.dma_address | bo.b.dma_address | bo.c.dma_address) >> 32) {
        fprintf(stderr, "a buffer's IOVA is above 4 GiB; the program cannot address it\n");
        goto close;
    }

    p.input_iova = (uint32_t)bo.a.dma_address;
    p.weights_iova = (uint32_t)bo.b.dma_address;
    p.output_iova = (uint32_t)bo.c.dma_address;
    int g = gen_gemm_fp16(&p);
    if (g != 0) {
        fprintf(stderr, "gen_gemm_fp16 refused %dx%dx%d (%d): the shape needs tiling\n",
                M, K, N, g);
        goto close;
    }
    /* task_count includes the four-word PC trailer whose last word starts the blocks; a
     * submit that stops short of it configures the task and never runs it. */
    printf("program: %u words\n", p.task_count);

    /* Operands, row-major, and the host result. */
    const char *se = getenv("ROCKET_TEST_SEED");
    const uint32_t seed = se ? (uint32_t)strtoul(se, NULL, 0) : (uint32_t)time(NULL);
    rng_state = seed ? seed : 1u;
    printf("seed: %u\n", seed);
    A = malloc((size_t)M * K * sizeof *A);
    B = malloc((size_t)N * K * sizeof *B);
    ref = calloc((size_t)M * N, sizeof *ref);
    if (!A || !B || !ref) {
        fprintf(stderr, "host allocation failed\n");
        goto free_host;
    }
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (_Float16)next_digit();
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (_Float16)next_digit();
    for (int kk = 0; kk < K; kk++)
        for (int r = 0; r < M; r++) {
            const float a = (float)A[(size_t)r * K + kk];
            for (int col = 0; col < N; col++)
                ref[(size_t)r * N + col] += a * (float)B[(size_t)col * K + kk];
        }

    /* Into the device layouts: A as a feature cube one pixel wide, B as N 1x1 kernels. The
     * index helpers count from 1. */
    rocket_bo_prep(fd, &bo.prog, 1, 0);
    rocket_bo_prep(fd, &bo.a, 1, 0);
    rocket_bo_prep(fd, &bo.b, 1, 0);
    memcpy(bo.prog.ptr, program, (size_t)p.task_count * sizeof *program);
    memset(bo.a.ptr, 0, bo.a.size);
    memset(bo.b.ptr, 0, bo.b.size);
    _Float16 *a_cube = bo.a.ptr, *b_cube = bo.b.ptr;
    for (int r = 0; r < M; r++)
        for (int kk = 0; kk < K; kk++)
            a_cube[feature_index(K, M, 1, 8, kk + 1, r + 1, 1)] = A[(size_t)r * K + kk];
    for (int col = 0; col < N; col++)
        for (int kk = 0; kk < K; kk++)
            b_cube[gemm_weight_index_fp16(K, col + 1, kk + 1)] = B[(size_t)col * K + kk];
    rocket_bo_fini(fd, &bo.prog);
    rocket_bo_fini(fd, &bo.a);
    rocket_bo_fini(fd, &bo.b);

    /* A sentinel in C tells "never written" (0xAA bytes survive) from "written wrong". */
    rocket_bo_prep(fd, &bo.c, 1, 0);
    memset(bo.c.ptr, 0xAA, bo.c.size);
    rocket_bo_fini(fd, &bo.c);

    uint32_t reads[] = { bo.prog.handle, bo.a.handle, bo.b.handle };
    uint32_t writes[] = { bo.c.handle };
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = rocket_submit_matmul(fd, &bo.prog, p.task_count, reads, 3, writes, 1, 6000);
    if (rc) {
        fprintf(stderr, "submit failed: %d (%s)\n", rc, strerror(-rc));
        goto free_host;
    }
    /* The submit returns once the job is queued; waiting on C's fence, with a real
     * deadline, is what waits for the NPU. */
    rc = rocket_bo_prep(fd, &bo.c, 0, 2000000000LL);
    printf("submit + wait: %.3f ms\n", ms_since(&t0));
    if (rc) {
        fprintf(stderr, "the job did not complete: %d (%s)\n", rc, strerror(-rc));
        goto free_host;
    }

    const uint8_t *raw = bo.c.ptr;
    size_t untouched = 0;
    for (size_t i = 0; i < bo.c.size; i++) untouched += raw[i] == 0xAA;
    if (untouched == bo.c.size) printf("C still holds the sentinel: the NPU wrote nothing\n");

    const _Float16 *c_cube = bo.c.ptr;
    for (int r = 0; r < M; r++)
        for (int col = 0; col < N; col++) {
            const _Float16 got = c_cube[feature_index(N, M, 1, 8, col + 1, r + 1, 1)];
            const _Float16 want = (_Float16)ref[(size_t)r * N + col];
            if (got == want) continue;
            if (wrong < 8)
                printf("C[%d][%d] = %g, want %g\n", r, col, (double)got, (double)want);
            wrong++;
        }
    rocket_bo_fini(fd, &bo.c);
    if (wrong)
        printf("FAIL: %ld of %d elements differ\n", wrong, M * N);
    else
        printf("OK: %dx%d times %dx%d, every element exact\n", M, K, K, N);
    status = wrong ? 1 : 0;

free_host:
    free(A);
    free(B);
    free(ref);
close:
    rocket_bo_free(fd, &bo.zero);
    rocket_bo_free(fd, &bo.prog);
    rocket_bo_free(fd, &bo.a);
    rocket_bo_free(fd, &bo.b);
    rocket_bo_free(fd, &bo.c);
    rocket_close(fd);
    return status;
}
