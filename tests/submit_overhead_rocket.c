// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * submit_overhead_rocket.c — isolate the per-job dispatch floor (µs/submit).
 *
 * A/B probe for the IOMMU keep-attached kernel lever (rocket patch 5). The cost
 * being measured lives in the kernel's drm_sched run_job path: stock `rocket`
 * does iommu_attach_group() on every job and iommu_detach_group() on every
 * completion; the patch keeps the per-context domain attached across same-fd
 * jobs. To expose ONLY that term we:
 *   - allocate every BO (input, weights, output, regcmd) ONCE, outside the loop
 *   - generate the regcmd ONCE (tiny shape, so HW compute is negligible)
 *   - submit the SAME 1-task job N times on the SAME fd (same IOMMU domain),
 *     waiting for completion each time, and report wall µs/submit.
 *
 * Single fd => every job after the first shares one domain, so under the patch
 * the attach/detach handshake happens exactly once; under stock it happens on
 * every job. Δ(µs/submit) baseline−patched ≈ the saved attach+detach cost.
 *
 * It also A/Bs the ROCKET_BUSY_POLL latency lever: the same submit+wait loop is
 * timed twice in one process — once with the stock blocking wait (the waiter
 * sleeps and is woken by the completion IRQ) and once spinning on a non-blocking
 * completion probe before sleeping (rocket_busy_poll_set_us). Busy-poll removes
 * the waiter's sleep->wake scheduler round-trip at the cost of a busy core, so
 * the win shows up here, on the tiny submit-bound shape with a small output BO.
 *
 * This is a PROBE, not a gate: it exits 0 on a successful run (so CI doesn't flap
 * on absolute timing). It does not score the values. It checks that the job WRITES:
 * the output BO is stamped with an fp16 NaN before the first warm-up submit and must
 * hold no NaN after it, because a submit path that completes without writing would
 * otherwise time as the fastest dispatch of all.
 *
 * Build (added to CMake as submit_overhead_rocket):
 *   gcc -O2 -Iinclude tests/submit_overhead_rocket.c src/rocket_npu.c \
 *       src/npu_regcmd.c -o submit_overhead_rocket -lm
 * Run:
 *   sudo ./submit_overhead_rocket               # M=8 K=64 N=16, 2000 submits
 *   sudo ./submit_overhead_rocket 8 64 16 2000  # M K N iters
 *   sudo ./submit_overhead_rocket 8 64 16 2000 300  # ...300µs busy-poll budget
 *   sudo ./submit_overhead_rocket 256 512 256 500 5000 16  # ...16 tasks a job
 *
 * The sixth argument repeats the one task that many times in one job. A job then lasts
 * as long as a tiled prefill matmul's does, a train of ~0.1-0.2 ms tasks with a
 * completion IRQ after each, while the waiter sleeps through the whole train. That is
 * the case a deep CPU idle state can reach, which a one-task job shorter than the
 * state's target residency cannot.
 *
 * Pin to an idle A76 for a clean read: sudo taskset -c 4 ./submit_overhead_rocket
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "test_fill.h"

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Time `iters` submit+wait cycles of the prebuilt job of n_tasks; fills samp[] with
 * per-iteration µs and writes the summary stats. busy_poll_us selects the wait
 * mode (0 = blocking IRQ wait, >0 = spin budget). Returns 0, or 1 on a HW error. */
static int time_loop(int fd, rocket_task_desc *t, unsigned n_tasks, uint32_t *in_h,
                     uint32_t *out_h, rocket_bo *out, int iters, long busy_poll_us,
                     double *samp, double *mean, double *median, double *minv,
                     double *p10, double *p90)
{
    rocket_busy_poll_set_us(busy_poll_us);
    double t0_all = now_us();
    for (int i = 0; i < iters; i++) {
        double a = now_us();
        if (rocket_submit_tasks(fd, t, n_tasks, in_h, 3, out_h, 1)) {
            fprintf(stderr, "submit %d failed\n", i); return 1;
        }
        if (rocket_bo_prep(fd, out, 0, 2000000000LL)) {
            fprintf(stderr, "wait %d timed out\n", i); return 1;
        }
        samp[i] = now_us() - a;
    }
    double total = now_us() - t0_all;
    qsort(samp, iters, sizeof(double), cmp_double);
    *mean   = total / iters;
    *median = samp[iters / 2];
    *p10    = samp[iters / 10];
    *p90    = samp[(iters * 9) / 10];
    *minv   = samp[0];
    return 0;
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 8;
    int K = argc > 2 ? atoi(argv[2]) : 64;
    int N = argc > 3 ? atoi(argv[3]) : 16;
    int iters = argc > 4 ? atoi(argv[4]) : 2000;
    long busy_us = argc > 5 ? atol(argv[5]) : 300;   /* busy-poll budget for the A/B */
    int n_tasks = argc > 6 ? atoi(argv[6]) : 1;      /* the one task, repeated in a job */
    int warmup = 64;

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open failed\n"); return 1; }

    /* tiny tile: input [M,K] fp16, weights packed [N,K] fp16, output [M,N] fp32 */
    size_t in_sz  = (size_t)M * K * sizeof(_Float16);
    size_t wt_sz  = (size_t)N * K * sizeof(_Float16);
    size_t out_sz = (size_t)M * N * sizeof(float) + 4096;
    rocket_bo in = {0}, wt = {0}, out = {0}, rc = {0};
    int e = 0;
    e |= rocket_bo_alloc(fd, in_sz,  &in);
    e |= rocket_bo_alloc(fd, wt_sz,  &wt);
    e |= rocket_bo_alloc(fd, out_sz, &out);
    e |= rocket_bo_alloc(fd, 256 * sizeof(uint64_t), &rc);
    if (e) { fprintf(stderr, "bo_alloc failed\n"); return 1; }

    /* fill input/weights once (values irrelevant — we time dispatch, not result,
     * but we do one correctness check so a broken path is caught) */
    rocket_bo_prep(fd, &in, 1, 0);
    _Float16 *A = (_Float16 *)in.ptr;
    for (int i = 0; i < M * K; i++) A[i] = (_Float16)((i % 7) * 0.5f);
    rocket_bo_fini(fd, &in);

    /* weights packed as the NPU expects: [N,K] row-major fp16 (B^T tile) */
    rocket_bo_prep(fd, &wt, 1, 0);
    _Float16 *B = (_Float16 *)wt.ptr;
    for (int i = 0; i < N * K; i++) B[i] = (_Float16)((i % 5) * 0.25f);
    rocket_bo_fini(fd, &wt);

    /* generate the regcmd ONCE */
    uint64_t ops[256] = {0};
    matmul_params_t p = {
        .m = M, .k = K, .n = N,
        .input_dma   = (uint32_t)in.dma_address,
        .weights_dma = (uint32_t)wt.dma_address,
        .output_dma  = (uint32_t)out.dma_address,
        .tasks = ops, .fp32tofp16 = 1,
        .accumulate = 0, .add_dma = 0,
    };
    if (gen_matmul_fp16(&p)) { fprintf(stderr, "gen_matmul_fp16 failed\n"); return 1; }
    rocket_bo_prep(fd, &rc, 1, 0);
    memcpy(rc.ptr, ops, (size_t)p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &rc);

    if (n_tasks < 1 || n_tasks > 256) { fprintf(stderr, "tasks a job: 1-256\n"); return 1; }
    rocket_task_desc t = { (uint32_t)rc.dma_address, p.task_count };
    rocket_task_desc tj[256];
    for (int i = 0; i < n_tasks; i++) tj[i] = t;
    uint32_t in_h[] = { in.handle, wt.handle, rc.handle };
    uint32_t out_h[] = { out.handle };

    /* the written-check: stamp the output, run one job, and find no stamp left */
    tf_sentinel_bo_f16(fd, &out);
    if (rocket_submit_tasks(fd, &t, 1, in_h, 3, out_h, 1) ||
        rocket_bo_prep(fd, &out, 0, 2000000000LL)) {
        fprintf(stderr, "first submit failed\n"); return 1;
    }
    {
        const _Float16 *o = (const _Float16 *)out.ptr;
        size_t n = out.size / sizeof(_Float16), unwritten = 0, want = (size_t)M * N;
        for (size_t i = 0; i < n && i < want; i++) unwritten += tf_is_sentinel_f16(o[i]);
        if (unwritten) {
            fprintf(stderr, "the job left %zu of the first %zu output elements unwritten\n",
                    unwritten, want);
            return 1;
        }
    }

    /* warmup: spins NPU up (pm_runtime resume + clk/volt vote) so it isn't in
     * the timed window; under the patch this also does the one-time attach */
    for (int i = 0; i < warmup; i++) {
        if (rocket_submit_tasks(fd, &t, 1, in_h, 3, out_h, 1)) {
            fprintf(stderr, "warmup submit failed\n"); return 1;
        }
        if (rocket_bo_prep(fd, &out, 0, 2000000000LL)) {
            fprintf(stderr, "warmup wait timed out\n"); return 1;
        }
    }

    /* timed loops: one submit + wait per iteration, all BOs reused. Run the
     * blocking baseline first, then the busy-poll variant, in the same warm
     * process so the only difference is the wait mode. */
    double *samp = malloc(sizeof(double) * iters);
    double b_mean, b_median, b_min, b_p10, b_p90;
    double s_mean, s_median, s_min, s_p10, s_p90;

    if (time_loop(fd, tj, (unsigned)n_tasks, in_h, out_h, &out, iters, 0,
                  samp, &b_mean, &b_median, &b_min, &b_p10, &b_p90)) return 1;
    if (time_loop(fd, tj, (unsigned)n_tasks, in_h, out_h, &out, iters, busy_us,
                  samp, &s_mean, &s_median, &s_min, &s_p10, &s_p90)) return 1;

    printf("shape M=%d K=%d N=%d  regcmd words=%u  tasks/job=%d  iters=%d (warmup=%d)\n",
           M, K, N, p.task_count, n_tasks, iters, warmup);
    printf("blocking   us/submit: mean=%.2f  median=%.2f  min=%.2f  p10=%.2f  p90=%.2f\n",
           b_mean, b_median, b_min, b_p10, b_p90);
    printf("busy-poll  us/submit: mean=%.2f  median=%.2f  min=%.2f  p10=%.2f  p90=%.2f  (budget=%ldus)\n",
           s_mean, s_median, s_min, s_p10, s_p90, busy_us);
    printf("delta      median %+.2f us (%+.1f%%)   mean %+.2f us (%+.1f%%)\n",
           s_median - b_median, b_median ? 100.0 * (s_median - b_median) / b_median : 0.0,
           s_mean - b_mean, b_mean ? 100.0 * (s_mean - b_mean) / b_mean : 0.0);
    printf("RESULT submit_us_blocking_median=%.2f submit_us_buspoll_median=%.2f "
           "submit_us_blocking_mean=%.2f submit_us_buspoll_mean=%.2f "
           "submit_us_blocking_min=%.2f\n",
           b_median, s_median, b_mean, s_mean, b_min);

    free(samp);
    rocket_bo_free(fd, &in);
    rocket_bo_free(fd, &wt);
    rocket_bo_free(fd, &out);
    rocket_bo_free(fd, &rc);
    rocket_close(fd);
    return 0;
}
