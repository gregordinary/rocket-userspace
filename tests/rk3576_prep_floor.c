// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_prep_floor.c — what a cache-maintenance bracket costs, and how much of it
 * is the SYSCALL rather than the BYTES.
 *
 * The write guard is a PREP_BO/FINI_BO pair per surface per call, and both halves sync
 * the WHOLE BO (`dma_sync_sgtable_for_cpu` / `_for_device` over the object's whole
 * scatterlist, DMA_BIDIRECTIONAL). So the guard's cost has two terms — a per-ioctl
 * floor and a bytes-proportional cache walk — and a RANGED cache-maintenance ioctl can
 * only remove the second. This probe measures both before any kernel side is built,
 * because a change that removes the bytes and keeps the syscalls lands at whatever
 * fraction of the guard the bytes actually are.
 *
 * What it reports, per BO size:
 *   reject   an ioctl the kernel refuses at handle lookup (no sync, no fence wait) —
 *            the syscall + drm dispatch + GEM lookup cost with the cache walk removed.
 *   prep     PREP_BO alone: fence poll + dma_sync_sgtable_for_cpu.
 *   fini     FINI_BO alone: dma_sync_sgtable_for_device.
 *   pair     one bracket, which is what the guard spends per surface per call.
 *   +fill    the same bracket with the sentinel memset inside it, whole surface and
 *            64 bytes — the fill against the bracket, at each size.
 *
 * A linear fit over the ladder gives the floor (the intercept, what a ranged ioctl
 * would still pay) and the slope in GB/s (what it could remove).
 *
 * This is a PROBE, not a gate: it exits 0 on any successful run. No job is ever in
 * flight, so PREP_BO's fence wait returns immediately and what is timed is the sync.
 *
 *   sudo -E taskset -c 4 ./build/rk3576_prep_floor            # the ladder, n=400
 *   sudo -E taskset -c 4 ./build/rk3576_prep_floor 1000       # more iterations
 *
 * Pin it. The cache walk runs on whichever core the ioctl lands on and an A53 reads
 * differently from an A72.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>

#include <libdrm/drm.h>
#include <drm/rocket_accel.h>

#include "rocket_npu.h"
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

/* Median of n timings of one operation, in us. */
static double median(double *v, int n)
{
    qsort(v, n, sizeof(double), cmp_double);
    return v[n / 2];
}

/* An ioctl the kernel rejects at drm_gem_object_lookup: no fence wait and no cache
 * walk, so what is left is the syscall, the drm ioctl dispatch and the lookup. This is
 * the hard floor a ranged ioctl cannot go below. */
static double reject_us(int fd, int iters, double *v)
{
    struct drm_rocket_prep_bo prep = { .handle = 0x7fffffffu, .timeout_ns = 0 };
    int i;

    for (i = 0; i < iters; i++) {
        double t0 = now_us();
        (void)ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, &prep);
        v[i] = now_us() - t0;
    }
    return median(v, iters);
}

struct row {
    size_t bytes;
    double reject, prep, fini, pair, fill_all, fill_64;
};

static void measure(int fd, size_t bytes, int iters, double *v, struct row *r)
{
    rocket_bo bo;
    int i;

    memset(r, 0, sizeof *r);
    r->bytes = bytes;
    if (rocket_bo_alloc(fd, bytes, &bo) != 0) {
        fprintf(stderr, "rk3576_prep_floor: alloc of %zu bytes failed\n", bytes);
        return;
    }
    /* Fault the mapping in and leave the lines resident, which is the state the guard
     * finds a surface in — the previous call read it and re-stamped it. */
    memset(bo.ptr, 0xA5, bytes);

    for (i = 0; i < iters; i++) {
        double t0 = now_us();
        rocket_bo_prep(fd, &bo, 0, 0);
        v[i] = now_us() - t0;
        rocket_bo_fini(fd, &bo);
    }
    r->prep = median(v, iters);

    for (i = 0; i < iters; i++) {
        rocket_bo_prep(fd, &bo, 0, 0);
        double t0 = now_us();
        rocket_bo_fini(fd, &bo);
        v[i] = now_us() - t0;
    }
    r->fini = median(v, iters);

    for (i = 0; i < iters; i++) {
        double t0 = now_us();
        rocket_bo_prep(fd, &bo, 0, 0);
        rocket_bo_fini(fd, &bo);
        v[i] = now_us() - t0;
    }
    r->pair = median(v, iters);

    /* The guard's own shape: read the surface (the scan short-circuits on the first
     * byte that differs from the sentinel, so it is the memset that costs), re-stamp
     * it, and close the bracket. */
    for (i = 0; i < iters; i++) {
        double t0 = now_us();
        rocket_bo_prep(fd, &bo, 0, 0);
        memset(bo.ptr, 0xA5, bytes);
        rocket_bo_fini(fd, &bo);
        v[i] = now_us() - t0;
    }
    r->fill_all = median(v, iters);

    for (i = 0; i < iters; i++) {
        double t0 = now_us();
        rocket_bo_prep(fd, &bo, 0, 0);
        memset(bo.ptr, 0xA5, bytes < 64 ? bytes : 64);
        rocket_bo_fini(fd, &bo);
        v[i] = now_us() - t0;
    }
    r->fill_64 = median(v, iters);

    rocket_bo_free(fd, &bo);
}

/* Least-squares fit of us = a + b*KiB over the rows, reported as a floor and a GB/s. */
static void fit(const struct row *rows, int n, const char *what,
                double (*pick)(const struct row *))
{
    double sx = 0, sy = 0, sxx = 0, sxy = 0, b, a;
    int i;

    for (i = 0; i < n; i++) {
        double x = rows[i].bytes / 1024.0, y = pick(&rows[i]);
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    a = (sy - b * sx) / n;
    /* b is us per KiB, so the rate is 1 KiB / b us = 1.024/b GB/s. */
    printf("  %-6s floor %6.2f us   slope %6.4f us/KiB  (%.2f GB/s)\n",
           what, a, b, b > 0 ? 1.024 / b : 0.0);
}

static double pick_pair(const struct row *r) { return r->pair; }
static double pick_prep(const struct row *r) { return r->prep; }
static double pick_fini(const struct row *r) { return r->fini; }

int main(int argc, char **argv)
{
    static const size_t ladder[] = {
        4u << 10, 16u << 10, 64u << 10, 128u << 10, 256u << 10,
        512u << 10, 1u << 20, 2u << 20, 4u << 20,
    };
    const int nrows = (int)(sizeof ladder / sizeof ladder[0]);
    int iters = argc > 1 ? atoi(argv[1]) : 400;
    struct row rows[sizeof ladder / sizeof ladder[0]];
    double *v, rej;
    int fd, i;

    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_prep_floor: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    if (iters < 20) iters = 20;

    fd = rocket_open();
    if (fd < 0) {
        fprintf(stderr, "rk3576_prep_floor: rocket_open failed (%d)\n", fd);
        return 1;
    }
    v = malloc(sizeof(double) * (size_t)iters);
    if (!v) return 1;

    /* A guard BO on IOVA 0, which is a real address on this part. */
    rocket_bo guard;
    if (rocket_bo_alloc(fd, 4096, &guard) != 0) return 1;

    rej = reject_us(fd, iters, v);

    for (i = 0; i < nrows; i++)
        measure(fd, ladder[i], iters, v, &rows[i]);

    printf("rk3576 cache-maintenance bracket  n=%d  (median us per operation)\n", iters);
    printf("  a rejected ioctl (syscall + dispatch + lookup, no sync): %.2f us\n\n", rej);
    printf("  %8s  %8s %8s %8s   %10s %10s\n",
           "bytes", "prep", "fini", "pair", "pair+fill", "pair+64B");
    for (i = 0; i < nrows; i++) {
        const struct row *r = &rows[i];
        printf("  %6zuK  %8.2f %8.2f %8.2f   %10.2f %10.2f\n",
               r->bytes >> 10, r->prep, r->fini, r->pair, r->fill_all, r->fill_64);
    }
    printf("\n");
    fit(rows, nrows, "prep", pick_prep);
    fit(rows, nrows, "fini", pick_fini);
    fit(rows, nrows, "pair", pick_pair);

    /* What the arithmetic says for a graph: a ranged ioctl removes the slope term and
     * keeps the floor, so the guard cannot fall below (brackets x pair-floor). */
    printf("\n  A ranged ioctl keeps the floor and removes the slope. For a graph of B\n"
           "  brackets the guard cannot fall below B x the pair floor above.\n");

    rocket_bo_free(fd, &guard);
    free(v);
    rocket_close(fd);
    return 0;
}
