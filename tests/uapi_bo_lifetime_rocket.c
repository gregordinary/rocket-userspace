// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * uapi_bo_lifetime_rocket.c — a BO must be allowed to outlive the file that made it.
 *
 * The per-context IOVA allocator (drm_mm + mm_lock) lives in struct rocket_file_priv
 * and is torn down and freed in rocket_postclose(). But a BO's IOVA node is removed
 * only in its GEM free path, rocket_gem_bo_free(), and a job's BO references are
 * dropped ASYNCHRONOUSLY by the drm_sched free worker — which can run after the
 * owning file has closed. A client that submits and closes without waiting therefore
 * leaves rocket_gem_bo_free() taking bo->driver_priv->mm_lock and calling
 * drm_mm_remove_node() on memory rocket_postclose() already freed.
 *
 * That is a use-after-free reachable by any member of group `render`. It first shows
 * as `drm_mm_takedown: allocator still has nodes` (a WARNING out of drm_mm.c), then
 * as whatever the freed allocator's bytes happen to say — a NULL dereference inside
 * drm_gem_shmem_free()'s DMA unmap being one observed shape.
 *
 * WHAT THIS PROBE DOES, and why it is shaped this way: the crash is rare when it is
 * chased through ordinary gate runs, because a normal caller waits for its job. This
 * one does not. Each iteration opens its own fd, submits a real program, and closes
 * IMMEDIATELY — no PREP_BO, no wait — so the free worker is racing postclose by
 * construction rather than by luck. It then reads the kernel log for the signatures
 * and reports what appeared.
 *
 * The verdict is the KERNEL LOG, not the exit status of the client: this defect does
 * not fail the ioctl that triggers it. A run can therefore pass every syscall and
 * still be the failing side of the A/B.
 *
 * GATE (registered in CTest): exit 0 = no BO-lifetime signature in the log,
 * 1 = one appeared, 2 = no NPU device, no program for this part, or the log is
 * unreadable (needs root to be a gate).
 *
 * Usage: sudo -E ./uapi_bo_lifetime_rocket [iterations]     (default 200)
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "rocket_hw_profile.h"
#include "npu_regcmd_rk3576.h"

/* The strings a live BO-lifetime defect puts in the log.
 *
 * `drm_mm_takedown` and the `rocket_postclose` frame under it are the direct
 * signature: the allocator is being torn down with nodes still in it. Match the
 * SYMBOL rather than drm_mm's message text — a 7.1 kernel prints
 * `WARNING: drivers/gpu/drm/drm_mm.c:965 at drm_mm_takedown+0x28/0x38` and the
 * "allocator still has nodes" wording does not appear at all, so a message-only
 * detector reads clean on a kernel where the defect is firing every iteration.
 * The rest catch the consequences once the allocator has been freed under it. */
static const char *const SIGNATURES[] = {
    "drm_mm_takedown",
    "rocket_postclose",
    "rocket_gem_bo_free",
    "drm_gem_shmem_release",
};
#define N_SIG ((int)(sizeof SIGNATURES / sizeof SIGNATURES[0]))

static long dmesg_lines(void)
{
    FILE *f = popen("dmesg 2>/dev/null | wc -l", "r");
    long n = -1;
    if (!f) return -1;
    if (fscanf(f, "%ld", &n) != 1) n = -1;
    pclose(f);
    return n;
}

/* Read the log delta ONCE and count every signature in it.
 *
 * A live defect puts a full WARN backtrace in the ring buffer per iteration, so by the
 * time the probe reports, dmesg is large — and re-reading it once per signature turned
 * a two-second probe into a two-minute one. Counts land in `hits`, and up to
 * `keep` matching lines are copied out for the report. */
static int dmesg_scan(long from, const char *const *sigs, int nsig, long *hits,
                      char (*sample)[200], int keep, int *nsample)
{
    char cmd[128], line[1024];
    FILE *f;
    int i;

    snprintf(cmd, sizeof cmd, "dmesg 2>/dev/null | tail -n +%ld", from + 1);
    f = popen(cmd, "r");
    if (!f) return -1;
    for (i = 0; i < nsig; i++) hits[i] = 0;
    *nsample = 0;
    while (fgets(line, sizeof line, f)) {
        int matched = 0;
        for (i = 0; i < nsig; i++)
            if (strstr(line, sigs[i])) { hits[i]++; matched = 1; }
        if (matched && *nsample < keep) {
            size_t n = strcspn(line, "\n");
            if (n >= sizeof sample[0]) n = sizeof sample[0] - 1;
            memcpy(sample[*nsample], line, n);
            sample[*nsample][n] = 0;
            (*nsample)++;
        }
    }
    pclose(f);
    return 0;
}

/* One fd's worth of the race: allocate, submit a program this part runs, and close
 * without waiting. Returns 1 if a job was submitted, 0 if the part gave us nothing to
 * submit (reported once by the caller). */
static int submit_and_close(int *built)
{
    int fd = rocket_open();
    rocket_bo in = {0}, wt = {0}, coeff = {0}, rc_bo = {0}, out = {0};
    rocket_bo *all[] = { &in, &wt, &coeff, &rc_bo, &out };
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    unsigned i;
    int submitted = 0;
    const unsigned IC = 32, OC = 32, IW = 32, IH = 32;

    if (fd < 0) return 0;

    if (rocket_bo_alloc(fd, (size_t)IC * IH * IW, &in) ||
        rocket_bo_alloc(fd, (size_t)OC * IC, &wt) ||
        rocket_bo_alloc(fd, rocket_rk3576_coeff_bytes(OC), &coeff) ||
        rocket_bo_alloc(fd, sizeof ops, &rc_bo) ||
        rocket_bo_alloc(fd, (size_t)OC * rocket_rk3576_out_surf_elems(IW, IH, 0), &out))
        goto out;

    {
        conv_params_t p = {
            .ic = IC, .ih = IH, .iw = IW, .oc = OC, .oh = IH, .ow = IW,
            .kh = 1, .kw = 1, .stride_y = 1, .stride_x = 1, .dil_y = 1, .dil_x = 1,
            .ih_full = IH, .oh_full = IH, .int8_out = 1, .tasks = ops,
            .in_scale = 1.0f, .w_scale = 1.0f, .out_scale = 1.0f,
            .input_dma   = (uint32_t)in.dma_address,
            .weights_dma = (uint32_t)wt.dma_address,
            .bias_dma    = (uint32_t)coeff.dma_address,
            .output_dma  = (uint32_t)out.dma_address,
        };
        if (gen_conv2d_int8_rk3576(&p) != 0 || p.task_count == 0) goto out;
        *built = 1;

        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, ops, p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);

        {
            rocket_task_desc t = { (uint32_t)rc_bo.dma_address, p.task_count };
            uint32_t inh[] = { in.handle, wt.handle, coeff.handle, rc_bo.handle };
            uint32_t outh[] = { out.handle };
            if (rocket_submit_tasks(fd, &t, 1, inh, 4, outh, 1) == 0) submitted = 1;
        }
    }

out:
    /* THE POINT OF THE PROBE: free the handles and close while the job may still be
     * running. Freeing the handles here does NOT free the BOs — the in-flight job
     * holds its own references, which is exactly the lifetime under test. */
    for (i = 0; i < sizeof all / sizeof all[0]; i++)
        if (all[i]->handle) rocket_bo_free(fd, all[i]);
    rocket_close(fd);
    return submitted;
}

int main(int argc, char **argv)
{
    /* 64 is plenty: a live defect fires on essentially every iteration, and each one
     * costs a full WARN backtrace in the kernel log. */
    int iters = (argc > 1) ? atoi(argv[1]) : 64;
    int built = 0, submitted = 0, i, bad = 0, nsample = 0;
    long hits[N_SIG];
    char sample[8][200];
    long mark;

    if (iters <= 0) iters = 64;

    {
        int fd = rocket_open();
        if (fd < 0) { fprintf(stderr, "no rocket device (skip)\n"); return 2; }
        rocket_close(fd);
    }

    printf("== rocket BO lifetime past close ==\n");
    printf("  info : chip %s, %d iterations, each its own fd\n",
           rocket_hw_current()->name, iters);
    printf("  info : a BO referenced by an in-flight job must survive its file's\n"
           "         postclose, and its IOVA node must be removable afterwards\n");

    mark = dmesg_lines();
    if (mark < 0) {
        printf("  info : dmesg unreadable — this probe's verdict IS the kernel log, so\n"
               "         run it as root (skip)\n");
        return 2;
    }

    for (i = 0; i < iters; i++)
        submitted += submit_and_close(&built);

    if (!built) {
        printf("  info : no generator for this part built a program — nothing was\n"
               "         submitted, so the race was never set up (skip)\n");
        return 2;
    }

    /* The free worker is asynchronous; give it room to run before reading the log. */
    sleep(2);

    printf("  info : %d of %d iterations submitted a job\n", submitted, iters);
    if (dmesg_scan(mark, SIGNATURES, N_SIG, hits, sample, 8, &nsample) < 0) {
        printf("  info : could not read the kernel log back (skip)\n");
        return 2;
    }
    for (i = 0; i < N_SIG; i++) {
        printf("  %s : %ld x \"%s\"\n", hits[i] > 0 ? "FAIL" : "ok  ", hits[i],
               SIGNATURES[i]);
        if (hits[i] > 0) bad++;
    }

    if (bad) {
        printf("  ---- : first matching log lines\n");
        for (i = 0; i < nsample; i++) printf("         %s\n", sample[i]);
        printf("\n== the BO-lifetime defect is LIVE on this kernel ==\n");
        return 1;
    }
    printf("\n== %d submits, no BO-lifetime signature in the log ==\n", submitted);
    return 0;
}
