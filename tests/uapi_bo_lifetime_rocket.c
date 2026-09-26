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
 * THE PROGRAM IS THE PART'S OWN. Each part gets a job it runs to completion: a 1x1
 * int8 conv on the RK3576 and a repeated fp16 matmul on the RK3588. This probe once
 * built the RK3576 program on every part, and on the RK3588 each of its 64 submits
 * hung a core until the kernel's watchdog retired it, so every ctest run there left
 * 64 "NPU job timed out" lines and 64 core resets that were read as by design. Any
 * such line in the window is now a failure, as is any rk_iommu line.
 *
 * GATE (registered in CTest): exit 0 = no signature in the log, 1 = one appeared,
 * 2 = no NPU device, no program for this part, or the kernel journal is unreadable
 * (test_klog.h: the caller's own access, then sudo -n).
 *
 * Usage: ./uapi_bo_lifetime_rocket [iterations]     (default 64)
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
#include "test_klog.h"

/* The strings a live BO-lifetime defect puts in the log.
 *
 * `drm_mm_takedown` and the `rocket_postclose` frame under it are the direct
 * signature: the allocator is being torn down with nodes still in it. Match the
 * SYMBOL rather than drm_mm's message text — a 7.1 kernel prints
 * `WARNING: drivers/gpu/drm/drm_mm.c:965 at drm_mm_takedown+0x28/0x38` and the
 * "allocator still has nodes" wording does not appear at all, so a message-only
 * detector reads clean on a kernel where the defect is firing every iteration.
 * The rest catch the consequences once the allocator has been freed under it, and the
 * last two catch a job this probe submitted that the part did not run: a hang the
 * watchdog retired, or a fault in the NPU's IOMMU. */
static const char *const SIGNATURES[] = {
    "drm_mm_takedown",
    "rocket_postclose",
    "rocket_gem_bo_free",
    "drm_gem_shmem_release",
    "timed out",
    "rk_iommu",
};
#define N_SIG ((int)(sizeof SIGNATURES / sizeof SIGNATURES[0]))

/* The RK3588's job: a 64x64x64 fp16 matmul repeated over 64 tasks of one job, so it is
 * still running when the file closes. The operands are left zeroed: what the race needs
 * is a well-formed program this part runs, not an answer. Returns 1 if it submitted. */
static int submit_rk3588(int fd, int *built)
{
    enum { M = 64, K = 64, N = 64, NTASK = 64 };
    rocket_bo in = {0}, wt = {0}, rc_bo = {0}, out = {0};
    rocket_bo *all[] = { &in, &wt, &rc_bo, &out };
    uint64_t ops[512] = {0};
    rocket_task_desc tasks[NTASK];
    unsigned i;
    int submitted = 0;

    if (rocket_bo_alloc(fd, (size_t)M * K * sizeof(_Float16) + 4096, &in) ||
        rocket_bo_alloc(fd, (size_t)N * K * sizeof(_Float16) + 4096, &wt) ||
        rocket_bo_alloc(fd, sizeof ops, &rc_bo) ||
        rocket_bo_alloc(fd, (size_t)M * N * sizeof(_Float16) + 4096, &out))
        goto out;

    {
        matmul_params_t p = {
            .m = M, .k = K, .n = N, .tasks = ops, .fp32tofp16 = 1,
            .input_dma   = (uint32_t)in.dma_address,
            .weights_dma = (uint32_t)wt.dma_address,
            .output_dma  = (uint32_t)out.dma_address,
        };
        if (gen_matmul_fp16(&p) != 0 || p.task_count == 0) goto out;
        *built = 1;

        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, ops, p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);

        for (i = 0; i < NTASK; i++)
            tasks[i] = (rocket_task_desc){ (uint32_t)rc_bo.dma_address, p.task_count };
        {
            uint32_t inh[] = { in.handle, wt.handle, rc_bo.handle };
            uint32_t outh[] = { out.handle };
            if (rocket_submit_tasks(fd, tasks, NTASK, inh, 3, outh, 1) == 0) submitted = 1;
        }
    }

out:
    /* As on the RK3576 below: free the handles and close while the job may still run. */
    for (i = 0; i < sizeof all / sizeof all[0]; i++)
        if (all[i]->handle) rocket_bo_free(fd, all[i]);
    rocket_close(fd);
    return submitted;
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
    if (strcmp(rocket_hw_current()->name, "rk3576") != 0)
        return submit_rk3588(fd, built);

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
    tk_mark mark;

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

    if (tk_mark_take(&mark) < 0) {
        printf("  info : the kernel journal is unreadable, and this probe's verdict IS the\n"
               "         kernel log: add the user to `adm`, or allow `sudo -n` (skip)\n");
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
    if (tk_scan(&mark, SIGNATURES, N_SIG, hits, sample, 8, &nsample) < 0) {
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
        printf("\n== the kernel logged a BO-lifetime signature, a job timeout or an IOMMU "
               "fault ==\n");
        return 1;
    }
    printf("\n== %d submits, no BO-lifetime signature in the log ==\n", submitted);
    return 0;
}
