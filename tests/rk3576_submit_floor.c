// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_submit_floor.c — the RK3576 per-submit dispatch floor (us/submit).
 *
 * On this part PC_DONE is not routable to the GIC, so the kernel polls for it on an
 * hrtimer and a submit cannot complete faster than one poll period. That period is
 * the floor, and every RK3576 cost table is a submit-count table because of it.
 * This probe reads the floor directly so a kernel-side change to the period, the
 * IOMMU attach policy or the post-completion settle can be priced.
 *
 * Method: the smallest shape the part will run, submitted N times through the public
 * matmul entry, reporting wall us/submit. The shape is chosen so the compute is
 * negligible against the dispatch cost — what is left is the ioctl round trips, the
 * job setup and the completion poll. It is deliberately NOT a gate's total wall time:
 * the conv gate is dominated by host packing and reads flat at every poll period.
 *
 * The host-side cost (the operand scatter and the output de-scatter) is inside the
 * timed region because the entry owns it, so the number is a per-submit cost for a
 * caller, not a kernel-only figure. Keep the shape fixed when comparing.
 *
 * This is a PROBE, not a gate: it exits 0 on any successful run. It does check the
 * first result against a CPU model, so a broken submit path cannot read as a fast one.
 *
 * Run with `sudo -E`, and pin it for a clean read:
 *   sudo -E taskset -c 4 ./build/rk3576_submit_floor          # 8x64x32, 300 submits
 *   sudo -E taskset -c 4 ./build/rk3576_submit_floor 8 64 32 300
 *
 * ROCKET_SF_GAP_MS puts an idle BETWEEN submits, outside the timed region, which is
 * what prices the runtime-PM autosuspend delay. A back-to-back chain never lets the
 * autosuspend timer fire, so it is blind to the delay; a workload whose inter-job gap
 * falls between the delay and the previous delay pays a suspend and a resume per job
 * that it did not pay before. A 60 Hz frame is 16.7 ms, so that is the interval to
 * read when the question is whether lowering the delay costs a media pipeline
 * anything.
 *
 * Discard the first run: the clock parks at idle.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 8;
    int K = argc > 2 ? atoi(argv[2]) : 64;
    int N = argc > 3 ? atoi(argv[3]) : 32;
    int iters = argc > 4 ? atoi(argv[4]) : 300;
    const char *gapenv = getenv("ROCKET_SF_GAP_MS");
    int gap_ms = (gapenv && *gapenv) ? (int)strtol(gapenv, NULL, 0) : 0;

    const struct rocket_hw_profile *hw = rocket_hw_current();

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_submit_floor: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }

    int fd = rocket_open();
    if (fd < 0) {
        fprintf(stderr, "rk3576_submit_floor: rocket_open failed (%d)\n", fd);
        return 1;
    }

    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)N * K);
    int8_t *C = malloc((size_t)M * N);
    if (!A || !B || !C)
        return 1;

    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i * 7 + 3) % 17 - 8);
    for (int i = 0; i < N * K; i++)
        B[i] = (int8_t)((i * 5 + 1) % 13 - 6);

    const float scale = 1.0f / 512.0f;

    /* One correctness check, so a broken submit path cannot read as a fast one. */
    int rc = rocket_matmul_int8_rk3576(fd, M, K, N, A, B, NULL, scale, C);
    if (rc != 0) {
        fprintf(stderr, "rk3576_submit_floor: matmul refused (%d) for %dx%dx%d\n",
                rc, M, K, N);
        return 1;
    }
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[(size_t)m * K + k] * (int32_t)B[(size_t)n * K + k];
            float f = acc * scale;
            int want = (int)(f < 0 ? f - 0.5f : f + 0.5f);
            if (want > 127) want = 127;
            if (want < -128) want = -128;
            int got = C[(size_t)m * N + n];
            if (got < want - 1 || got > want + 1) {
                fprintf(stderr, "rk3576_submit_floor: WRONG at [%d][%d] got %d want %d"
                        " -- not timing a broken path\n", m, n, got, want);
                return 1;
            }
        }
    }

    double *us = malloc(sizeof(double) * iters);
    if (!us)
        return 1;

    /* Warm: the clock parks at idle and a cold run reads low. */
    for (int i = 0; i < 20; i++)
        rocket_matmul_int8_rk3576(fd, M, K, N, A, B, NULL, scale, C);

    for (int i = 0; i < iters; i++) {
        /* Outside the timed region on purpose: what is being priced is what the gap
         * does to the SUBMIT, not the gap itself. */
        if (gap_ms > 0) {
            struct timespec ts;
            ts.tv_sec = gap_ms / 1000;
            ts.tv_nsec = (long)(gap_ms % 1000) * 1000000L;
            nanosleep(&ts, NULL);
        }
        double t0 = now_us();
        rocket_matmul_int8_rk3576(fd, M, K, N, A, B, NULL, scale, C);
        us[i] = now_us() - t0;
    }

    double sum = 0;
    for (int i = 0; i < iters; i++)
        sum += us[i];
    qsort(us, iters, sizeof(double), cmp_double);

    printf("rk3576 submit floor  %dx%dx%d  n=%d  inter-submit gap %d ms\n",
           M, K, N, iters, gap_ms);
    printf("  mean %.1f us   median %.1f us   min %.1f us   p90 %.1f us   max %.1f us\n",
           sum / iters, us[iters / 2], us[0], us[(iters * 9) / 10], us[iters - 1]);

    free(us);
    free(A);
    free(B);
    free(C);
    rocket_close(fd);
    return 0;
}
