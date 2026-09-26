// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_mm_fp16_cost.c — what the RK3576 fp16 matmul costs, against the int8 entry.
 *
 * A probe, not a gate: it measures and asserts nothing about the numbers. Both entries
 * take one submit a call at every shape in the fp16 envelope, so the per-call wall
 * compares like with like. Five arms per shape:
 *
 *   fp16   rocket_matmul_fp16_rk3576(), the whole call: its buffers, the weight
 *          permutation, the coefficient buffer, the stamp, the submit, the fence, the copy.
 *   int8   rocket_matmul_int8_rk3576() at the same M, K and N, the whole call, per-tensor
 *          scale and no bias. Skipped where N%32 != 0, which that entry refuses.
 *   floor  the fp16 program's own device floor: the same register program with its
 *          buffers packed once, timed from the submit to the end of the fence wait. It is
 *          what a resident-weight fp16 entry could approach, and what the call pays above
 *          it is host work.
 *   f16r   rocket_matmul_fp16_rk3576_wbo(), the whole call, with the weight and the
 *          coefficient buffer in a handle created once per shape outside the timing.
 *   i8r    rocket_matmul_int8_rk3576_perc_wbo(), the entry the RK3576 LLM route calls,
 *          with its weight in a handle, one scale for every column and the column sums
 *          computed once, all outside the timing. Skipped where N%32 != 0.
 *
 * Two ratios are paired within each round: fp16/int8 for the per-call entries, and
 * f16r/i8r for the resident ones.
 *
 * THE ARMS ARE INTERLEAVED, and the order rotates every round, so a drift in the board's
 * state lands on all five rather than on one. Each ratio is paired within its round. The
 * first ROCKET_MFC_WARM rounds (default 5) are discarded. Pin the governor before running
 * it: a load-sampling governor parks the cores while the process waits on the fence, and
 * that biases every arm by how long its host half runs.
 *
 * The first fp16 and f16r results per shape are checked exactly against the CPU, with
 * the gate's fills, so a price is never quoted for a wrong answer. The int8 arm's correctness is
 * rk3576_matmul_gate's, and here only its return code is checked.
 *
 *   rk3576_mm_fp16_cost [rounds] [shape-filter]      all five arms (default 50 rounds)
 *   rk3576_mm_fp16_cost int8only [rounds]            the int8 arm alone, for
 *   rk3576_mm_fp16_cost fp16only [rounds]            ROCKET_MM_PROFILE runs of one entry
 *   rk3576_mm_fp16_cost f16ronly [rounds]            one resident arm with the floor, for
 *   rk3576_mm_fp16_cost i8ronly [rounds]             ROCKET_RK3576_BO_POOL=1 runs
 *   rk3576_mm_fp16_cost resident [rounds] MxKxN...   f16r, i8r and the floor alone, at
 *                                                    any shape the entries take
 *   rk3576_mm_fp16_cost resident:f16r [rounds] ...   f16r and the floor, or i8r alone:
 *   rk3576_mm_fp16_cost resident:i8r [rounds] ...    one entry per process, for the pool
 *                                                    and for ROCKET_MM_PROFILE
 *
 * THE FLOOR IS TILED LIKE THE ENTRY. Past one task's rows the floor job cuts M the way
 * rocket_matmul_fp16_rk3576() does and submits the tasks as one job, so it stays the
 * device's share of that entry's call.
 *
 * THE CHECK IS SAMPLED AT LARGE SHAPES. Past 2^26 multiply-adds the first f16r result is
 * checked on every row and on 61 columns at a stride coprime to N, which touches every
 * task and every 16-column group once N is past 976; a full reference would take longer
 * than the arms.
 *
 * WITH THE POOL ON, RUN ONE ENTRY PER PROCESS. The pool is per process and hands a request
 * the smallest free BO that covers it, so arms sharing a process trade BOs: one arm can
 * take another's larger buffer, and PREP_BO and FINI_BO sync the whole BO, so its cache
 * maintenance is priced at the other arm's size. The floor arm keeps its own BOs and
 * never touches the pool.
 *
 * Exit 0, 1 if an fp16 result was wrong or an entry failed, 2 with no RK3576 device.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"
#include "test_fill.h"

static const unsigned SHAPES[][3] = {
#include "rk3576_mm_fp16_shapes.h"
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof SHAPES[0]))

enum { ARM_FP16, ARM_INT8, ARM_FLOOR, ARM_F16R, ARM_I8R, N_ARMS };
#define ONLY_RESIDENT 100
#define ONLY_RES_F16R 101
#define ONLY_RES_I8R  102
static const char *const ARM_NAME[N_ARMS] = { "fp16", "int8", "floor", "f16r", "i8r" };

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double pct(double *v, int n, double p)
{
    int i = (int)(p * (n - 1) + 0.5);
    qsort(v, (size_t)n, sizeof *v, cmp_d);
    return v[i < 0 ? 0 : i >= n ? n - 1 : i];
}

/* The fp16 program with its buffers packed once: submit to fence, nothing else. Cut
 * along M the way the entry cuts it, so past one task it is still the entry's device
 * share. */
#define FLOOR_MAX_TASKS 128
struct floor_job {
    rocket_bo in, w, coef, out, rc;
    uint32_t in_h[4], out_h[1];
    rocket_task_desc td[FLOOR_MAX_TASKS];
    unsigned ntask;
};

static int floor_init(int fd, struct floor_job *j, unsigned m, unsigned k, unsigned n,
                      const _Float16 *A, const _Float16 *B)
{
    unsigned cap = rocket_rk3576_mm_fp16_task_rows(k), mt, t;
    memset(j, 0, sizeof *j);
    if (!cap) return -1;
    j->ntask = (m + cap - 1) / cap;
    mt = (m + j->ntask - 1) / j->ntask;
    j->ntask = (m + mt - 1) / mt;
    if (j->ntask > FLOOR_MAX_TASKS) return -1;
    if (rocket_bo_alloc32(fd, rocket_rk3576_mm_fp16_in_bytes(m, k), &j->in) ||
        rocket_bo_alloc32(fd, rocket_rk3576_mm_fp16_weight_bytes(k, n), &j->w) ||
        rocket_bo_alloc32(fd, rocket_rk3576_mm_fp16_coef_bytes(n), &j->coef) ||
        rocket_bo_alloc32(fd, rocket_rk3576_mm_fp16_out_bytes(m, n), &j->out) ||
        rocket_bo_alloc32(fd, (size_t)j->ntask * RK3576_MM_FP16_OPS * 8, &j->rc))
        return -1;
    rocket_bo_prep(fd, &j->in, 1, 0);
    memset(j->in.ptr, 0, j->in.size);
    memcpy(j->in.ptr, A, (size_t)m * k * 2);
    rocket_bo_fini(fd, &j->in);
    rocket_bo_prep(fd, &j->w, 1, 0);
    rocket_rk3576_mm_fp16_pack_weights(j->w.ptr, j->w.size, (const uint16_t *)B, k, n);
    rocket_bo_fini(fd, &j->w);
    rocket_bo_prep(fd, &j->coef, 1, 0);
    rocket_rk3576_mm_fp16_pack_coef(j->coef.ptr, j->coef.size, n);
    rocket_bo_fini(fd, &j->coef);
    rocket_bo_prep(fd, &j->rc, 1, 0);
    for (t = 0; t < j->ntask; t++) {
        unsigned r0 = t * mt, rows = m - r0 < mt ? m - r0 : mt;
        if (gen_matmul_fp16_rk3576((uint64_t *)j->rc.ptr + (size_t)t * RK3576_MM_FP16_OPS,
                                   rows, k, n,
                                   (uint32_t)(j->in.dma_address + (uint64_t)r0 * k * 2),
                                   (uint32_t)j->w.dma_address,
                                   (uint32_t)(j->out.dma_address + (uint64_t)r0 * n * 4),
                                   (uint32_t)j->coef.dma_address) != RK3576_MM_FP16_OPS) {
            rocket_bo_fini(fd, &j->rc);
            return -1;
        }
        j->td[t].regcmd = (uint32_t)(j->rc.dma_address +
                                     (uint64_t)t * RK3576_MM_FP16_OPS * 8);
        j->td[t].regcmd_count = RK3576_MM_FP16_OPS;
    }
    rocket_bo_fini(fd, &j->rc);
    j->in_h[0] = j->in.handle; j->in_h[1] = j->w.handle;
    j->in_h[2] = j->coef.handle; j->in_h[3] = j->rc.handle;
    j->out_h[0] = j->out.handle;
    return 0;
}

static double floor_run(int fd, struct floor_job *j)
{
    double t0 = now_ms();
    if (j->ntask == 1
        ? rocket_submit_matmul(fd, &j->rc, RK3576_MM_FP16_OPS, j->in_h, 4, j->out_h, 1, 4000)
        : rocket_submit_tasks(fd, j->td, j->ntask, j->in_h, 4, j->out_h, 1))
        return -1;
    if (rocket_bo_prep(fd, &j->out, 0, 2000000000ull) < 0) return -1;
    t0 = now_ms() - t0;
    rocket_bo_fini(fd, &j->out);
    return t0;
}

static void floor_free(int fd, struct floor_job *j)
{
    rocket_bo *b[5] = { &j->rc, &j->out, &j->coef, &j->w, &j->in };
    for (int i = 0; i < 5; i++) if (b[i]->ptr) rocket_bo_free(fd, b[i]);
}

static int run_shape(int fd, unsigned m, unsigned k, unsigned n, int rounds, int warm,
                     int only)
{
    _Float16 *A = malloc((size_t)m * k * 2), *B = malloc((size_t)n * k * 2);
    int8_t *A8 = malloc((size_t)m * k), *B8 = malloc((size_t)n * k), *C8 = malloc((size_t)m * n);
    float *C = malloc((size_t)m * n * 4), *scale_n = malloc(sizeof(float) * n);
    int64_t *sum_abs = calloc(n, sizeof(int64_t));
    double *t[N_ARMS], *ratio = malloc(sizeof(double) * (size_t)rounds);
    double *ratio_r = malloc(sizeof(double) * (size_t)rounds);
    int have[N_ARMS];
    struct floor_job fj;
    struct rocket_rk3576_wbo_fp16 *h16 = NULL;
    struct rocket_rk3576_wbo *h8 = NULL;
    int r, a, bad = 0, nr = 0;

    for (a = 0; a < N_ARMS; a++) {
        t[a] = malloc(sizeof(double) * (size_t)rounds);
        have[a] = only == ONLY_RESIDENT ? (a == ARM_F16R || a == ARM_I8R || a == ARM_FLOOR)
                : only == ONLY_RES_F16R ? (a == ARM_F16R || a == ARM_FLOOR)
                : only == ONLY_RES_I8R  ? a == ARM_I8R
                : only < 0 || a == only || (a == ARM_FLOOR && only >= ARM_F16R);
    }
    have[ARM_INT8] &= n % 32 == 0;
    have[ARM_I8R] &= n % 32 == 0;
    for (size_t i = 0; i < (size_t)m * k; i++) {
        A[i] = (_Float16)(tf_int(0xA1, i, -6, 6) * 0.25);
        A8[i] = (int8_t)tf_int(0xA8, i, -100, 100);
    }
    for (size_t i = 0; i < (size_t)n * k; i++) {
        B[i] = (_Float16)(tf_int(0xB2, i, -3, 3) * 0.5);
        B8[i] = (int8_t)tf_int(0xB8, i, -100, 100);
    }
    for (unsigned c = 0; c < n; c++) {
        scale_n[c] = 1.0f / 4096.0f;
        for (unsigned i = 0; i < k; i++) {
            int v = B8[(size_t)c * k + i];
            sum_abs[c] += v < 0 ? -v : v;
        }
    }
    if (have[ARM_FLOOR] && floor_init(fd, &fj, m, k, n, A, B) != 0) {
        printf("  %2ux%4ux%-3u the floor job's buffers would not set up\n", m, k, n);
        return 1;
    }
    if (have[ARM_F16R] &&
        rocket_rk3576_wbo_fp16_create(fd, (int)k, (int)n, B, &h16) != ROCKET_OK) {
        printf("  %2ux%4ux%-3u the fp16 handle would not create\n", m, k, n);
        return 1;
    }
    if (have[ARM_I8R] && rocket_rk3576_wbo_create(fd, (int)k, (int)n, B8, &h8) != ROCKET_OK) {
        printf("  %2ux%4ux%-3u the int8 handle would not create\n", m, k, n);
        return 1;
    }

    /* The first fp16 and f16r results, exactly. */
    for (int ci = 0; ci < 2; ci++) {
        int erc;
        a = ci ? ARM_F16R : ARM_FP16;
        if (!have[a]) continue;
        erc = a == ARM_FP16
            ? rocket_matmul_fp16_rk3576(fd, (int)m, (int)k, (int)n, A, B, C)
            : rocket_matmul_fp16_rk3576_wbo(fd, (int)m, (int)k, (int)n, A, h16, C);
        if (erc != ROCKET_OK) {
            printf("  %2ux%4ux%-3u the %s arm failed: %d\n", m, k, n, ARM_NAME[a], erc);
            return 1;
        }
        {
            int sampled = (double)m * k * n > (double)(1 << 26);
            int ncol = sampled ? (n < 61 ? (int)n : 61) : (int)n;
            int stride = tf_coprime_stride((int)n, sampled ? (int)n / 61 + 1 : 1);
            for (unsigned rr = 0; rr < m && !bad; rr++)
                for (int ci = 0; ci < ncol; ci++) {
                    unsigned c = sampled ? (unsigned)tf_sample_index(ci, (int)n, stride)
                                         : (unsigned)ci;
                    double acc = 0;
                    for (unsigned i = 0; i < k; i++)
                        acc += (double)A[(size_t)rr * k + i] * (double)B[(size_t)c * k + i];
                    if ((double)C[(size_t)rr * n + c] != acc) { bad = 1; break; }
                }
        }
        if (bad) {
            printf("  %2ux%4ux%-3u the %s result is WRONG: no price is quoted\n", m, k, n,
                   ARM_NAME[a]);
            return 1;
        }
    }

    for (r = 0; r < warm + rounds; r++) {
        double ts[N_ARMS] = { 0 };
        for (int s = 0; s < N_ARMS; s++) {
            a = (s + r) % N_ARMS;              /* the order rotates every round */
            if (!have[a]) continue;
            if (a == ARM_FP16) {
                double t0 = now_ms();
                if (rocket_matmul_fp16_rk3576(fd, (int)m, (int)k, (int)n, A, B, C) != ROCKET_OK)
                    return 1;
                ts[a] = now_ms() - t0;
            } else if (a == ARM_INT8) {
                double t0 = now_ms();
                if (rocket_matmul_int8_rk3576(fd, (int)m, (int)k, (int)n, A8, B8, NULL,
                                              1.0f / 4096.0f, C8) != ROCKET_OK) {
                    printf("  %2ux%4ux%-3u rocket_matmul_int8_rk3576 failed\n", m, k, n);
                    return 1;
                }
                ts[a] = now_ms() - t0;
            } else if (a == ARM_F16R) {
                double t0 = now_ms();
                if (rocket_matmul_fp16_rk3576_wbo(fd, (int)m, (int)k, (int)n, A, h16, C) !=
                    ROCKET_OK)
                    return 1;
                ts[a] = now_ms() - t0;
            } else if (a == ARM_I8R) {
                double t0 = now_ms();
                if (rocket_matmul_int8_rk3576_perc_wbo(fd, (int)m, (int)k, (int)n, A8, h8,
                                                       NULL, scale_n, sum_abs, C8,
                                                       NULL) != ROCKET_OK) {
                    printf("  %2ux%4ux%-3u rocket_matmul_int8_rk3576_perc_wbo failed\n",
                           m, k, n);
                    return 1;
                }
                ts[a] = now_ms() - t0;
            } else {
                ts[a] = floor_run(fd, &fj);
                if (ts[a] < 0) return 1;
            }
        }
        if (r < warm) continue;
        for (a = 0; a < N_ARMS; a++) t[a][nr] = ts[a];
        ratio[nr] = (have[ARM_FP16] && have[ARM_INT8]) ? ts[ARM_FP16] / ts[ARM_INT8] : 0;
        ratio_r[nr] = (have[ARM_F16R] && have[ARM_I8R]) ? ts[ARM_F16R] / ts[ARM_I8R] : 0;
        nr++;
    }

    printf("  %2ux%4ux%-3u", m, k, n);
    for (a = 0; a < N_ARMS; a++) {
        if (!have[a]) { printf("  %5s %22s", ARM_NAME[a], "-"); continue; }
        double p10 = pct(t[a], nr, 0.10), p50 = pct(t[a], nr, 0.50), p90 = pct(t[a], nr, 0.90);
        printf("  %5s %6.3f [%6.3f-%6.3f]", ARM_NAME[a], p50, p10, p90);
    }
    if (have[ARM_FP16] && have[ARM_INT8])
        printf("  fp16/int8 %.2fx", pct(ratio, nr, 0.50));
    if (have[ARM_F16R] && have[ARM_I8R])
        printf("  f16r/i8r %.2fx", pct(ratio_r, nr, 0.50));
    printf("\n");

    if (have[ARM_FLOOR]) floor_free(fd, &fj);
    rocket_rk3576_wbo_fp16_free(fd, h16);
    rocket_rk3576_wbo_free(fd, h8);
    for (a = 0; a < N_ARMS; a++) free(t[a]);
    free(ratio); free(ratio_r); free(scale_n); free(sum_abs);
    free(A); free(B); free(A8); free(B8); free(C8); free(C);
    return 0;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, i, fails = 0, only = -1, ai = 1, rounds = 50;
    int warm = getenv("ROCKET_MFC_WARM") ? atoi(getenv("ROCKET_MFC_WARM")) : 5;
    const char *filter = NULL;

    if (argc > 1 && !strcmp(argv[1], "int8only")) { only = ARM_INT8; ai = 2; }
    else if (argc > 1 && !strcmp(argv[1], "fp16only")) { only = ARM_FP16; ai = 2; }
    else if (argc > 1 && !strcmp(argv[1], "f16ronly")) { only = ARM_F16R; ai = 2; }
    else if (argc > 1 && !strcmp(argv[1], "i8ronly")) { only = ARM_I8R; ai = 2; }
    else if (argc > 1 && !strcmp(argv[1], "resident")) { only = ONLY_RESIDENT; ai = 2; }
    else if (argc > 1 && !strcmp(argv[1], "resident:f16r")) { only = ONLY_RES_F16R; ai = 2; }
    else if (argc > 1 && !strcmp(argv[1], "resident:i8r")) { only = ONLY_RES_I8R; ai = 2; }
    if (argc > ai) rounds = atoi(argv[ai]);
    if (argc > ai + 1 && only < ONLY_RESIDENT) filter = argv[ai + 1];
    if (rounds < 1) rounds = 1;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_mm_fp16_cost: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_mm_fp16_cost: no NPU device — skipping\n"); return 2; }

    printf("== RK3576 fp16 matmul cost, ms per call: median [p10-p90] over %d rounds, "
           "%d warm-up rounds discarded ==\n", rounds, warm);
    if (only >= ONLY_RESIDENT) {
        for (i = ai + 1; i < argc; i++) {
            unsigned m, k, n;
            if (sscanf(argv[i], "%ux%ux%u", &m, &k, &n) != 3) {
                printf("bad shape %s\n", argv[i]);
                fails++;
                continue;
            }
            fails += run_shape(fd, m, k, n, rounds, warm, only);
            fflush(stdout);
        }
    } else
    for (i = 0; i < N_SHAPES; i++) {
        char name[32];
        snprintf(name, sizeof name, "%ux%ux%u", SHAPES[i][0], SHAPES[i][1], SHAPES[i][2]);
        if (filter && !strstr(name, filter)) continue;
        fails += run_shape(fd, SHAPES[i][0], SHAPES[i][1], SHAPES[i][2], rounds, warm, only);
    }
    rocket_close(fd);
    return fails ? 1 : 0;
}
