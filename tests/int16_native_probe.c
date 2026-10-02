// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * int16_native_probe.c — does an int16 x int16 matmul write a full int32 surface on the
 * RK3588?
 *
 * A probe, not a gate. The record says no: with `gen_matmul_int16`
 * the int32 writer emitted one 1x16 tile, row 0 and the first sixteen channels, and nothing
 * else, invariant to every field swept. Two things about that sweep decide what it could see:
 *
 *   - It moved one knob at a time at M=4 K=32 N=64, from int8's integer-output geometry
 *     (`size_e`=7, `surf_add` = stride x 8). The fp16 -> fp32 path, whose operands are also
 *     two bytes and whose output is also four, writes with `size_e`=3 and `surf_add` =
 *     stride x 4, and so does allbilly's `experimental/gemm_int16.py`, which reports
 *     int16 x int16 -> int32 exact. A pair of fields moved together was never an arm.
 *   - allbilly's evidence is weak in its own way: operands in [-16, 16), whose high byte is
 *     only a sign extension, and every shape square, which cannot tell M from N in a stride.
 *
 * So each arm here is one single-task job of `gen_matmul_int16`, scored against a CPU model,
 * under one of two output geometries:
 *
 *   int8g  size_e 7, surf_add x8: the generator as it ships, the record's arm
 *   fp16g  size_e 3, surf_add x4: the two-byte-in, four-byte-out geometry
 *
 * and one of four operand fills, hashed with no period (test_fill.h):
 *
 *   small  A and B in [-16, 15], allbilly's range
 *   afull  A over all of int16, B in [-b, b] with b = (2^31-1) / (K x 32768): exact in int32
 *   bfull  the same with A and B exchanged, so a high byte the part drops shows in B too
 *   big    both over all of int16: sums past int32, scored against the int64 dot product
 *          saturated to int32, the model a 48-bit accumulator with a saturating writer gives
 *
 * at shapes with M, K and N all different, K >= 64 but for the record's own 4x32x64.
 *
 * Every element is scored, the output is prefilled with 0xAA so an unwritten element shows,
 * and a job the kernel retired at its 500 ms watchdog is counted apart from one that ran.
 *
 * What a green result does NOT show: a K split across tasks, a tiled or chained job, speed
 * against the int8 byte decomposition (`rocket_matmul_int16_exact`), the transposed int16
 * writer, or the RK3576.
 *
 * Usage: int16_native_probe [M K N [reps]]   (no shape: the built-in ladder)
 * Exit: 0 when every fp16g arm is exact, 1 otherwise, 2 with no NPU.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "test_fill.h"

enum { G_INT8, G_FP16, G_N };
static const char *const geom_name[G_N] = { "int8g", "fp16g" };
static const char *const geom_size_e[G_N] = { "7", "3" };
static const char *const geom_surf[G_N] = { "8", "4" };

enum { F_SMALL, F_AFULL, F_BFULL, F_BIG, F_N };
static const char *const fill_name[F_N] = { "small", "afull", "bfull", "big" };

struct shape { int M, K, N; };

static const struct shape ladder[] = {
    { 4, 32, 64 },      /* the record's shape */
    { 8, 64, 48 },      /* N not a multiple of 32 */
    { 12, 256, 80 },
    { 36, 512, 112 },
    { 64, 128, 256 },   /* N > M: sixty-four output surfaces */
    { 100, 1024, 64 },  /* M > N, seven data banks */
};

struct score {
    size_t wrong, unwritten, exact64, saturated;
    size_t row0_first16;     /* right elements in row 0, columns 0-15: the record's tile */
    long first_m, first_n;
    int64_t first_want;
    int32_t first_got;
};

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int32_t sat_i32(int64_t v)
{
    return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t)v;
}

static void fill_operands(int f, int M, int K, int N, uint64_t seed, int16_t *A, int16_t *B)
{
    const int b = (int)((int64_t)INT32_MAX / ((int64_t)K * 32768));
    int alo = -16, ahi = 15, blo = -16, bhi = 15;

    if (f == F_AFULL) { alo = -32768; ahi = 32767; blo = -b; bhi = b; }
    if (f == F_BFULL) { alo = -b; ahi = b; blo = -32768; bhi = 32767; }
    if (f == F_BIG)   { alo = blo = -32768; ahi = bhi = 32767; }
    tf_fill_i16(A, (size_t)M * K, seed, alo, ahi);
    tf_fill_i16(B, (size_t)N * K, seed ^ 0x5A5A, blo, bhi);
}

/* A is M x K row-major, B is N x K row-major (the kernels), C is M x N. */
static void cpu_ref(const int16_t *A, const int16_t *B, int M, int K, int N, int64_t *C)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++) s += (int64_t)A[(size_t)m * K + k] * B[(size_t)n * K + k];
            C[(size_t)m * N + n] = s;
        }
}

/* One arm: returns 0 when every element equals the saturating model, 1 when not, -1 when
 * the job was refused, failed or retired by the watchdog. */
static int run_arm(int fd, int g, int f, struct shape s, int rep, int *timed_out)
{
    const int M = s.M, K = s.K, N = s.N;
    const size_t in_b = (size_t)M * K * 2, w_b = (size_t)N * K * 2, out_b = (size_t)M * N * 4;
    rocket_bo rc = { 0 }, in = { 0 }, w = { 0 }, out = { 0 };
    uint64_t ops[256] = { 0 };
    int16_t *A = malloc(in_b), *B = malloc(w_b);
    int64_t *C = malloc((size_t)M * N * sizeof(int64_t));
    struct score sc = { .first_m = -1, .first_n = -1 };
    int ret = -1;
    double ms = 0;

    *timed_out = 0;
    if (!A || !B || !C) goto out;
    if (rocket_bo_alloc32(fd, 4096, &rc) || rocket_bo_alloc32(fd, in_b, &in) ||
        rocket_bo_alloc32(fd, w_b, &w) || rocket_bo_alloc32(fd, out_b, &out)) {
        printf("  %s %-5s %dx%dx%d: BO allocation failed\n", geom_name[g], fill_name[f], M, K, N);
        goto out;
    }

    setenv("ROCKET_INT16_SIZE_E", geom_size_e[g], 1);
    setenv("ROCKET_INT16_SURF_MULT", geom_surf[g], 1);
    matmul_params_t p = {
        .m = (uint16_t)M, .k = (uint16_t)K, .n = (uint16_t)N,
        .input_dma = (uint32_t)in.dma_address,
        .weights_dma = (uint32_t)w.dma_address,
        .output_dma = (uint32_t)out.dma_address,
        .tasks = ops,
    };
    int grc = gen_matmul_int16(&p);
    unsetenv("ROCKET_INT16_SIZE_E");
    unsetenv("ROCKET_INT16_SURF_MULT");
    if (grc != 0) {
        printf("  %s %-5s %dx%dx%d: generator refused (%d)\n", geom_name[g], fill_name[f], M, K,
               N, grc);
        goto out;
    }

    fill_operands(f, M, K, N, 0x1600 + (uint64_t)(f * 131 + rep * 7919 + M * 31 + N), A, B);
    cpu_ref(A, B, M, K, N, C);

    rocket_bo_prep(fd, &rc, 1, 0);
    memcpy(rc.ptr, ops, (size_t)p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &rc);
    rocket_bo_prep(fd, &in, 1, 0);
    rocket_bo_prep(fd, &w, 1, 0);
    memset(in.ptr, 0, in.size);
    memset(w.ptr, 0, w.size);
    {
        int16_t *id = in.ptr, *wd = w.ptr;
        for (int m = 1; m <= M; m++)
            for (int k = 1; k <= K; k++)
                id[feature_data(K, M, 1, 8, k, m, 1)] = A[(size_t)(m - 1) * K + (k - 1)];
        for (int n = 1; n <= N; n++)
            for (int k = 1; k <= K; k++)
                wd[weight_int16(K, n, k)] = B[(size_t)(n - 1) * K + (k - 1)];
    }
    rocket_bo_fini(fd, &in);
    rocket_bo_fini(fd, &w);
    if (tf_sentinel_bo(fd, &out)) goto out;

    {
        uint32_t in_h[] = { in.handle, w.handle, rc.handle }, out_h[] = { out.handle };
        uint64_t slow0 = rocket_fence_wait_slow_count();
        double t0 = now_ms();
        int src = rocket_submit_matmul(fd, &rc, p.task_count, in_h, 3, out_h, 1, 6000);
        if (src == 0) src = rocket_bo_prep(fd, &out, 0, 2000000000ull);
        ms = now_ms() - t0;
        if (src != 0) {
            printf("  %s %-5s %dx%dx%d rep %d: submit or wait failed (%d)\n", geom_name[g],
                   fill_name[f], M, K, N, rep, src);
            goto out;
        }
        if (rocket_fence_wait_slow_count() != slow0 || ms > 400.0) *timed_out = 1;
    }

    {
        const int32_t *o = out.ptr;
        for (int m = 1; m <= M; m++)
            for (int n = 1; n <= N; n++) {
                const int64_t want64 = C[(size_t)(m - 1) * N + (n - 1)];
                const int32_t want = sat_i32(want64);
                const int32_t got = o[feature_data(N, M, 1, 4, n, m, 1)];
                if ((uint32_t)got == 0xAAAAAAAAu && want != (int32_t)0xAAAAAAAAu) sc.unwritten++;
                if (got == want) {
                    if ((int64_t)got == want64) sc.exact64++; else sc.saturated++;
                    if (m == 1 && n <= 16) sc.row0_first16++;
                    continue;
                }
                if (!sc.wrong) {
                    sc.first_m = m - 1; sc.first_n = n - 1;
                    sc.first_want = want64; sc.first_got = got;
                }
                sc.wrong++;
            }
    }
    rocket_bo_fini(fd, &out);

    printf("  %s %-5s %4dx%4dx%4d rep %d: %6zu of %6zu wrong, %6zu unwritten, %6zu exact, "
           "%6zu saturated-right, row0[0:16] %2zu/16, %.3f ms%s",
           geom_name[g], fill_name[f], M, K, N, rep, sc.wrong, (size_t)M * N, sc.unwritten,
           sc.exact64, sc.saturated, sc.row0_first16 > 16 ? 16 : sc.row0_first16, ms,
           *timed_out ? " [WATCHDOG]" : "");
    if (sc.wrong)
        printf(", first (m %ld, n %ld) want %lld got %d", sc.first_m, sc.first_n,
               (long long)sc.first_want, sc.first_got);
    printf("\n");
    ret = (sc.wrong || *timed_out) ? 1 : 0;

out:
    rocket_bo_free(fd, &rc); rocket_bo_free(fd, &in);
    rocket_bo_free(fd, &w); rocket_bo_free(fd, &out);
    free(A); free(B); free(C);
    return ret;
}

int main(int argc, char **argv)
{
    struct shape one;
    const struct shape *shapes = ladder;
    int nshapes = (int)(sizeof(ladder) / sizeof(ladder[0]));
    int reps = 2, fd, fp16_bad = 0, watchdog = 0;
    int tally[G_N][F_N][2] = { { { 0 } } };   /* [geom][fill][exact, not] */

    if (argc >= 4) {
        one.M = atoi(argv[1]); one.K = atoi(argv[2]); one.N = atoi(argv[3]);
        if (one.M < 4 || one.M % 4 || one.K < 32 || one.K % 32 || one.N < 16 || one.N % 16) {
            fprintf(stderr, "usage: %s [M K N [reps]], M%%4 >= 4, K%%32, N%%16\n", argv[0]);
            return 1;
        }
        shapes = &one; nshapes = 1;
    }
    if (argc >= 5) reps = atoi(argv[4]);
    if (reps < 1) reps = 1;

    fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }
    printf("== int16 native-output probe: %d shape(s), %d rep(s), geometries int8g (size_e 7, "
           "surf x8) and fp16g (size_e 3, surf x4) ==\n", nshapes, reps);

    for (int r = 0; r < reps; r++)
        for (int si = 0; si < nshapes; si++)
            for (int g = 0; g < G_N; g++)
                for (int f = 0; f < F_N; f++) {
                    int to = 0, rc = run_arm(fd, g, f, shapes[si], r, &to);
                    watchdog += to;
                    tally[g][f][rc == 0 ? 0 : 1]++;
                    if (g == G_FP16 && rc != 0) fp16_bad++;
                }

    printf("== summary (exact arms / all arms) ==\n");
    for (int g = 0; g < G_N; g++) {
        printf("  %s:", geom_name[g]);
        for (int f = 0; f < F_N; f++)
            printf("  %s %d/%d", fill_name[f], tally[g][f][0], tally[g][f][0] + tally[g][f][1]);
        printf("\n");
    }
    printf("== %d job(s) reached the watchdog mark; fp16g %s ==\n", watchdog,
           fp16_bad ? "NOT exact on every arm" : "exact on every arm");
    rocket_close(fd);
    return fp16_bad ? 1 : 0;
}
