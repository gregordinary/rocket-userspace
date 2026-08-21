// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_multicore.c — what a second NPU core buys on the RK3576.
 *
 * The part has two NPU cores. The kernel gives every open fd one drm scheduling
 * entity over the list of every core's scheduler, so work fans out across cores by
 * opening several fds and driving each from its own thread — there is no per-core
 * ioctl. This probe prices that: the same total number of matmuls, run through W
 * worker fds, for W = 1..workers.
 *
 * With the per-submit floor at a few hundred microseconds and most RK3576 cost-table
 * rows sitting near it, submit CONCURRENCY is the throughput lever a second core
 * offers. The shape is deliberately small, so each call is one submit and the wall
 * time is dispatch, not arithmetic: what is being measured is how many submits the
 * part retires per second, not how fast one is.
 *
 * Every worker checks its own first result against a CPU model before the timed
 * region, so a core that computes garbage cannot read as a fast one. That check is
 * the point as much as the timing is: on a kernel where the second core's probe was
 * forced through (its shared CBUF clocks taken optionally), this is what says whether
 * the core actually computes.
 *
 * The plain int8 matmul is used because its output element is one byte, so it carries
 * none of the wide-output poisoning — a hazard whose interaction with two cores in
 * flight is not characterised. Do not switch this to an int32 or fp16 output without
 * dealing with that first.
 *
 * WHAT THIS MEASURES IS NOT THE SECOND CORE unless you run the one-core control beside
 * it. W>1 opens W fds, and two fds fill one core's dispatch gaps whether or not a second
 * core exists: at a 439 us submit floor a single fd leaves the core idle between submits.
 * Unbind core 1 (`echo 27708000.npu > /sys/bus/platform/drivers/rocket/unbind`, rebind
 * with `bind`) and the same W=2 numbers come back — 1.51-1.69x at 8x64x32, 1.62-1.66x at
 * 32x1024x512, 1.49-1.51x at 128x1024x1024 — matching or beating the two-core run. So the
 * gain is pipelining, and the second core adds nothing on top of it. [HW sweep, H96 MAX
 * M9, 2026-07-28]
 *
 * AND TWO CORES IN FLIGHT COMPUTE WRONG ANSWERS AT SOME SHAPES. The corruption is sparse
 * and small — a few positions off by exactly +/-2 where the reference is 0 — and it is
 * intermittent, so one passing run is not evidence: 256x512x2048 failed 3 of 3 two-core
 * runs here and was exact on 2 of 2 one-core runs with the same two fds. That is why the
 * per-worker check below runs before the timed region rather than after.
 *
 * This is a PROBE, not a gate: it exits 0 on any successful run, non-zero only if a
 * worker computed the wrong answer or the device could not be opened.
 *
 * Run with `sudo -E`. Workers are pinned one per big core starting at CPU 4, because
 * an unpinned fan-out scatters onto the little cores and the straggler sets the wall.
 *
 *   sudo -E ./build/rk3576_multicore                 # 8x64x32, 200 calls, W = 1,2
 *   sudo -E ./build/rk3576_multicore 32 1024 512 60  # a shape with real work in it
 *   ROCKET_MC_WORKERS=4 sudo -E ./build/rk3576_multicore
 *
 * Discard the first run: the clock parks at idle.
 */
#define _GNU_SOURCE     /* CPU_ZERO / CPU_SET / pthread_setaffinity_np */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

#define MC_MAX_WORKERS 8
#define MC_FIRST_BIG_CPU 4

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

struct mc_worker {
    pthread_t   thread;
    int         index;
    int         fd;
    int         M, K, N;
    int         calls;          /* how many matmuls this worker runs */
    int         cpu;            /* host cpu to pin to, or -1 */
    const int8_t *A, *B;
    float       scale;
    int8_t     *C;              /* private output */
    int         rc;             /* first non-zero return, or 0 */
    int         wrong;          /* set if the CPU model disagreed */
    double      us;             /* this worker's own wall time */
};

/* The reference the workers are checked against: the same integer dot product the
 * part computes, requantised by the output scale. */
static void mc_reference(const struct mc_worker *w, int8_t *ref)
{
    for (int m = 0; m < w->M; m++) {
        for (int n = 0; n < w->N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < w->K; k++)
                acc += (int32_t)w->A[(size_t)m * w->K + k] *
                       (int32_t)w->B[(size_t)n * w->K + k];
            float f = acc * w->scale;
            int q = (int)(f < 0 ? f - 0.5f : f + 0.5f);
            if (q > 127)  q = 127;
            if (q < -128) q = -128;
            ref[(size_t)m * w->N + n] = (int8_t)q;
        }
    }
}

static void *mc_run(void *arg)
{
    struct mc_worker *w = arg;

    if (w->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(w->cpu, &set);
        /* Advisory: a failure here costs accuracy of the reading, not correctness. */
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }

    /* Correctness first, outside the timed region. A worker that computes garbage
     * must not be reported as a worker that computed fast. */
    w->rc = rocket_matmul_int8_rk3576(w->fd, w->M, w->K, w->N, w->A, w->B, NULL,
                                      w->scale, w->C);
    if (w->rc != 0)
        return NULL;

    int8_t *ref = malloc((size_t)w->M * w->N);
    if (!ref) {
        w->rc = -1;
        return NULL;
    }
    mc_reference(w, ref);
    for (int i = 0; i < w->M * w->N; i++) {
        int got = w->C[i], want = ref[i];
        if (got < want - 1 || got > want + 1) {
            fprintf(stderr, "  worker %d: WRONG at %d, got %d want %d\n",
                    w->index, i, got, want);
            w->wrong = 1;
            break;
        }
    }
    free(ref);
    if (w->wrong)
        return NULL;

    /* Warm this worker's own fd: the first submit on a fresh entity pays setup. */
    for (int i = 0; i < 5; i++)
        rocket_matmul_int8_rk3576(w->fd, w->M, w->K, w->N, w->A, w->B, NULL,
                                  w->scale, w->C);

    double t0 = now_us();
    for (int i = 0; i < w->calls; i++) {
        int rc = rocket_matmul_int8_rk3576(w->fd, w->M, w->K, w->N, w->A, w->B, NULL,
                                           w->scale, w->C);
        if (rc != 0 && w->rc == 0)
            w->rc = rc;
    }
    w->us = now_us() - t0;
    return NULL;
}

/* One pass at W workers. Returns wall us for the whole pass, or a negative value if
 * any worker failed. */
static double mc_pass(int workers, int M, int K, int N, int total_calls,
                      const int8_t *A, const int8_t *B, float scale, int pin,
                      double *worker_us, int *out_failed)
{
    struct mc_worker w[MC_MAX_WORKERS];
    memset(w, 0, sizeof(w));

    for (int i = 0; i < workers; i++) {
        w[i].index = i;
        w[i].M = M; w[i].K = K; w[i].N = N;
        w[i].A = A; w[i].B = B;
        w[i].scale = scale;
        w[i].cpu = pin ? (MC_FIRST_BIG_CPU + i) : -1;
        /* Split the SAME total work, so the passes are comparable. */
        w[i].calls = total_calls / workers + (i < total_calls % workers ? 1 : 0);
        w[i].C = malloc((size_t)M * N);
        w[i].fd = rocket_open();
        if (w[i].fd < 0 || !w[i].C) {
            fprintf(stderr, "  worker %d: could not open the device (%d)\n", i, w[i].fd);
            for (int j = 0; j <= i; j++) {
                if (w[j].fd >= 0) rocket_close(w[j].fd);
                free(w[j].C);
            }
            *out_failed = 1;
            return -1;
        }
    }

    double t0 = now_us();
    for (int i = 0; i < workers; i++)
        pthread_create(&w[i].thread, NULL, mc_run, &w[i]);
    for (int i = 0; i < workers; i++)
        pthread_join(w[i].thread, NULL);
    double wall = now_us() - t0;

    int failed = 0;
    for (int i = 0; i < workers; i++) {
        if (w[i].rc != 0) {
            fprintf(stderr, "  worker %d: matmul returned %d\n", i, w[i].rc);
            failed = 1;
        }
        if (w[i].wrong)
            failed = 1;
        if (worker_us)
            worker_us[i] = w[i].us;
        rocket_close(w[i].fd);
        free(w[i].C);
    }
    *out_failed = failed;
    return failed ? -1 : wall;
}

int main(int argc, char **argv)
{
    /* This probe exists to put two jobs in flight at once, which is what the
     * library refuses to let a caller build by accident. Lift its own guard. */
    setenv("ROCKET_RK3576_ALLOW_MULTI_FD", "1", 0);
    int M = argc > 1 ? atoi(argv[1]) : 8;
    int K = argc > 2 ? atoi(argv[2]) : 64;
    int N = argc > 3 ? atoi(argv[3]) : 32;
    int total = argc > 4 ? atoi(argv[4]) : 200;

    const char *wenv = getenv("ROCKET_MC_WORKERS");
    int max_workers = (wenv && *wenv) ? (int)strtol(wenv, NULL, 0) : 2;
    if (max_workers < 1) max_workers = 1;
    if (max_workers > MC_MAX_WORKERS) max_workers = MC_MAX_WORKERS;

    const char *penv = getenv("ROCKET_MC_PIN");
    int pin = (penv && *penv) ? (int)strtol(penv, NULL, 0) : 1;

    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_multicore: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }

    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)N * K);
    if (!A || !B)
        return 1;
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i * 7 + 3) % 17 - 8);
    for (int i = 0; i < N * K; i++)
        B[i] = (int8_t)((i * 5 + 1) % 13 - 6);

    const float scale = 1.0f / 512.0f;

    printf("rk3576 multicore  %dx%dx%d  %d calls total  pin=%d\n", M, K, N, total, pin);
    printf("  the same total work, split across W worker fds\n\n");

    double base = 0;
    int failed_any = 0;

    for (int workers = 1; workers <= max_workers; workers++) {
        double worker_us[MC_MAX_WORKERS];
        int failed = 0;
        double wall = mc_pass(workers, M, K, N, total, A, B, scale, pin,
                              worker_us, &failed);
        if (failed || wall < 0) {
            printf("  W=%d  FAILED\n", workers);
            failed_any = 1;
            continue;
        }

        if (workers == 1)
            base = wall;

        printf("  W=%d  wall %8.1f ms   %7.1f us/call   %6.0f calls/s",
               workers, wall / 1000.0, wall / total, total / (wall / 1e6));
        if (base > 0)
            printf("   %.2fx", base / wall);
        printf("\n");

        if (workers > 1) {
            printf("        per worker:");
            for (int i = 0; i < workers; i++)
                printf(" %.1f ms", worker_us[i] / 1000.0);
            printf("\n");
        }
    }

    free(A);
    free(B);
    return failed_any ? 1 : 0;
}
