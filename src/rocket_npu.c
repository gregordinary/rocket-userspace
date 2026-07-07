// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_npu.c — see rocket_npu.h.
 *
 * Targets the mainline DRM-accel "rocket" driver over /dev/accel/accel0, using the uAPI
 * from <drm/rocket_accel.h> (the DRM_IOCTL_ROCKET_* ioctls and struct drm_rocket_*). The
 * descriptor field names below follow that mainline header; build against the
 * <drm/rocket_accel.h> installed on the target kernel.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdlib.h>

#include <time.h>

#include <stddef.h>         /* offsetof */

#include <libdrm/drm.h>
#include <drm/rocket_accel.h>   /* DRM_IOCTL_ROCKET_*, struct drm_rocket_*    */

#include "rocket_npu.h"
#include "rocket_log.h"     // centralized log channel

/*
 * Per-job batched-submit flag (DRM_ROCKET_JOB_BATCHED): runs a job's tasks as
 * one chained HW kick (see rocket_chain.c / the patches/rocket kernel half).
 * It is an OPTIONAL trailing field of drm_rocket_job. When the installed
 * <drm/rocket_accel.h> already carries it, set job.flags directly. When it
 * predates the field (e.g. an off-device syntax check against an unpatched
 * header), append the field via a wrapper at exactly the offset the kernel
 * expects (pinned by the static_assert) and pass the larger job_struct_size; a
 * patched kernel reads min(job_struct_size, its sizeof) and honors it, a stock
 * one ignores the tail. Either way the flag is a no-op against a stock kernel.
 */
#ifndef DRM_ROCKET_JOB_BATCHED
#define DRM_ROCKET_JOB_BATCHED (1u << 0)
struct rocket_job_flagged {
    struct drm_rocket_job job;
    __u32 flags;
    __u32 reserved;
};
_Static_assert(offsetof(struct rocket_job_flagged, flags) ==
               sizeof(struct drm_rocket_job),
               "appended flags must sit exactly at the stock drm_rocket_job end");
#define ROCKET_JOB_FLAGS_VENDORED 1
#endif

/* Submit one job (n_tasks tasks) on `fd`, optionally with the per-job batched
 * flag set. Centralizes the vendored-vs-native job_struct_size handling so the
 * callers below stay uniform. Returns 0 / negative errno. */
static int rkt_submit_one_job(int fd, struct drm_rocket_task *dt, uint32_t n_tasks,
                              const uint32_t *in_handles,  uint32_t n_in,
                              const uint32_t *out_handles, uint32_t n_out,
                              int batched)
{
    __u32 flags = batched ? DRM_ROCKET_JOB_BATCHED : 0u;
    struct drm_rocket_job job = {
        .tasks               = (uint64_t)(uintptr_t)dt,
        .task_count          = n_tasks,
        .task_struct_size    = sizeof(struct drm_rocket_task),
        .in_bo_handles       = (uint64_t)(uintptr_t)in_handles,
        .in_bo_handle_count  = n_in,
        .out_bo_handles      = (uint64_t)(uintptr_t)out_handles,
        .out_bo_handle_count = n_out,
    };
#ifdef ROCKET_JOB_FLAGS_VENDORED
    struct rocket_job_flagged jx = { .job = job, .flags = flags, .reserved = 0 };
    struct drm_rocket_submit submit = {
        .jobs            = (uint64_t)(uintptr_t)&jx,
        .job_count       = 1,
        .job_struct_size = sizeof(jx),
    };
#else
    job.flags = flags;
    struct drm_rocket_submit submit = {
        .jobs            = (uint64_t)(uintptr_t)&job,
        .job_count       = 1,
        .job_struct_size = sizeof(job),
    };
#endif
    if (ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, &submit) < 0) {
        ROCKET_LOGE("ROCKET_SUBMIT(%u tasks, batched=%d): %s\n",
                    n_tasks, batched, strerror(errno));
        return -errno;
    }
    return 0;
}

/* ============================================================================
 * SECTION — Device lifecycle (open / close)
 * ==========================================================================*/

int rocket_open(void)
{
    /* Default device, overridable via ROCKET_DEV for multi-NPU boxes / test rigs. */
    const char *dev = getenv("ROCKET_DEV");
    if (!dev || !*dev) dev = "/dev/accel/accel0";
    int fd = open(dev, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        ROCKET_LOGE("open(%s): %s\n", dev, strerror(errno));
        return -errno;
    }

    char name[32] = {0};
    struct drm_version dv = { .name = name, .name_len = sizeof(name) - 1 };
    if (ioctl(fd, DRM_IOCTL_VERSION, &dv) < 0) {
        int e = errno;                       /* capture before close()/fprintf clobber */
        ROCKET_LOGE("DRM_IOCTL_VERSION: %s\n", strerror(e));
        close(fd);
        return -e;
    }
    if (strcmp(name, "rocket") != 0) {   /* full NUL-terminated name, not a 6-char prefix */
        ROCKET_LOGE("accel0 driver is '%s', expected 'rocket'\n", name);
        close(fd);
        return -ENODEV;
    }
    if (getenv("ROCKET_DEBUG"))
        ROCKET_LOGD("opened rocket: %s %d.%d.%d\n", name,
                dv.version_major, dv.version_minor, dv.version_patchlevel);
    return fd;
}

void rocket_close(int fd) { if (fd >= 0) close(fd); }

/* ============================================================================
 * SECTION — Buffer-object allocation and teardown
 * ==========================================================================*/

int rocket_bo_alloc(int fd, size_t size, rocket_bo *bo)
{
    memset(bo, 0, sizeof(*bo));

    struct drm_rocket_create_bo create = { .size = size };
    if (ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, &create) < 0) {
        int e = errno;                       /* capture before fprintf clobber */
        ROCKET_LOGE("ROCKET_CREATE_BO(%zu): %s\n", size, strerror(e));
        return -e;
    }

    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, create.offset);
    if (ptr == MAP_FAILED) {
        int e = errno;                       /* capture before fprintf/ioctl clobber */
        ROCKET_LOGE("mmap bo: %s\n", strerror(e));
        /* best-effort: close handle via GEM_CLOSE */
        struct drm_gem_close gc = { .handle = create.handle };
        ioctl(fd, DRM_IOCTL_GEM_CLOSE, &gc);
        return -e;
    }

    bo->handle      = create.handle;
    bo->dma_address = create.dma_address;
    bo->mmap_offset = create.offset;
    bo->size        = size;
    bo->ptr         = ptr;
    return 0;
}

void rocket_bo_free(int fd, rocket_bo *bo)
{
    /* munmap/GEM_CLOSE failures are otherwise silent; a failed GEM_CLOSE leaks the GEM
     * handle + its IOVA, and across thousands of resident prepacked weights that could
     * exhaust the per-fd 4 GB IOVA window invisibly. Neither call fails on a healthy
     * teardown, so warn UNCONDITIONALLY (WARN prints by default) — a leak then leaves a
     * forensic trail instead of surfacing as a silent late-run OOM (cf. the loud FINI_BO
     * log). The message is one line per failed free; a systemic failure that floods it is
     * itself the signal. */
    if (bo->ptr) {
        if (munmap(bo->ptr, bo->size) < 0)
            ROCKET_LOGW("rocket_bo_free munmap(handle=%u, size=%zu): %s\n",
                    bo->handle, bo->size, strerror(errno));
        bo->ptr = NULL;
    }
    if (bo->handle) {
        struct drm_gem_close gc = { .handle = bo->handle };
        if (ioctl(fd, DRM_IOCTL_GEM_CLOSE, &gc) < 0)
            ROCKET_LOGW("rocket_bo_free GEM_CLOSE(handle=%u): %s — leaking GEM handle/IOVA\n",
                    bo->handle, strerror(errno));
        bo->handle = 0;
    }
}

/* ============================================================================
 * SECTION — BO cache sync and completion-fence wait
 * ==========================================================================*/

/* ROCKET_BUSY_POLL — spin-poll the completion fence for up to this many
 * microseconds before falling back to the blocking IRQ wait. 0/unset = off.
 *
 * A single-stream LATENCY lever for tiny submit-bound jobs: a blocking wait puts
 * the thread to sleep, so completion costs an IRQ-thread wakeup plus a scheduler
 * round-trip to re-run the waiter. Spinning on a non-blocking completion probe
 * keeps the waiter runnable, so it returns within one probe of the fence
 * signalling. Trade-offs (why it is a knob, not a default): it burns a CPU core
 * while spinning (latency, not throughput — a pooled/oversubscribed run wants
 * that core for other work), and the only completion probe the mainline uAPI
 * exposes (PREP_BO with a zero deadline) re-invalidates the output BO's cache
 * every poll, so it pays only when that output is SMALL. It does NOT skip the
 * kernel IRQ — the mainline `rocket` fence is still IRQ-signalled — so it
 * removes the waiter-side wakeup, not the interrupt itself (that needs a
 * kernel-side poll, a patches/rocket item). Resolved once from the env, with a
 * programmatic override for in-process A/B (see rocket_busy_poll_set_us). */
static _Atomic long g_busy_poll_us = -1;   /* -1 = unresolved; resolve lazily from env (also set by rocket_busy_poll_set_us) */

void rocket_busy_poll_set_us(long us)
{
    g_busy_poll_us = us < 0 ? 0 : us;
}

static long rkt_busy_poll_us(void)
{
    if (g_busy_poll_us < 0) {
        const char *e = getenv("ROCKET_BUSY_POLL");
        long v = (e && *e) ? strtol(e, NULL, 10) : 0;
        g_busy_poll_us = v < 0 ? 0 : v;
    }
    return g_busy_poll_us;
}

/* Non-blocking completion probe: PREP_BO with a zero (=poll) deadline. Returns
 * 0 if the job's writes to this BO have landed (the BO is now CPU-synced),
 * -EBUSY if still in flight, or another negative errno on a real failure. */
static int rkt_prep_poll(int fd, uint32_t handle)
{
    struct drm_rocket_prep_bo prep = { .handle = handle, .timeout_ns = 0 };
    if (ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, &prep) < 0)
        return -errno;
    return 0;
}

int rocket_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t timeout_ns)
{
    /* CONFIRMED from rocket_gem.c: drm_rocket_prep_bo is {handle, reserved,
     * timeout_ns}, no direction flag. The kernel waits on the BO's WRITE-usage
     * fence, then always does dma_sync_sgtable_for_cpu(); FINI_BO syncs back for
     * the NPU. `dir` documents caller intent only.
     *
     * CRITICAL: the kernel runs timeout_ns through drm_timeout_abs_to_jiffies()
     * — it is an ABSOLUTE CLOCK_MONOTONIC deadline, not a duration. A small raw
     * value (e.g. "2e9") lands in the past => the wait degrades to a poll and,
     * with the job still running, returns -EBUSY immediately. So we take a
     * RELATIVE timeout here (0 = non-blocking poll) and convert to an absolute
     * deadline. */
    (void)dir;

    /* Optional spin-before-sleep (ROCKET_BUSY_POLL, µs). Only for a real wait
     * (timeout_ns>0); a timeout_ns==0 caller is a plain cache sync, not a wait.
     * Poll the fence for up to the budget, then fall through to the blocking
     * wait below for the remainder of the original timeout. A successful probe
     * has already CPU-synced the BO, so we return straight away. */
    if (timeout_ns) {
        long spin_us = rkt_busy_poll_us();
        if (spin_us > 0) {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            int64_t spin_deadline = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec
                                    + (int64_t)spin_us * 1000;
            for (;;) {
                int r = rkt_prep_poll(fd, bo->handle);
                if (r == 0) return 0;            /* completed; BO already synced */
                if (r != -EBUSY) return r;       /* a genuine ioctl failure */
                clock_gettime(CLOCK_MONOTONIC, &ts);
                if ((int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec >= spin_deadline)
                    break;                       /* budget spent -> blocking wait */
#if defined(__aarch64__)
                __asm__ __volatile__("yield");   /* SEV/WFE hint, no scheduler give-up */
#endif
            }
        }
    }
    int64_t deadline = 0; /* 0 -> kernel poll (drm_timeout_abs_to_jiffies(0)==0) */
    if (timeout_ns) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        deadline = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec + (int64_t)timeout_ns;
    }
    struct drm_rocket_prep_bo prep = {
        .handle     = bo->handle,
        .timeout_ns = deadline,
    };
    if (ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, &prep) < 0) {
        int e = errno;                       /* capture before any clobber */
        /* A timed wait (timeout_ns>0) returning -EBUSY/-ETIMEDOUT is the caller's to
         * report (it owns the retry/abort policy, e.g. the "WAIT TIMEOUT" messages).
         * A timeout_ns==0 sync (prep-for-device before a pack, or a post-wait readback
         * sync) should not fail; many callers ignore its return, so surface it loudly
         * here — otherwise an ioctl failure silently lets the NPU run on, or the host
         * read back, an un-synced buffer (stale-data / silent-corruption). */
        if (timeout_ns == 0)
            ROCKET_LOGE("rocket_bo_prep(sync) PREP_BO handle=%u: %s\n",
                    bo->handle, strerror(e));
        return -e;
    }
    return 0;
}

int rocket_bo_fini(int fd, rocket_bo *bo)
{
    struct drm_rocket_fini_bo fini = { .handle = bo->handle };
    if (ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini) < 0) {
        int e = errno;                       /* capture before any clobber */
        /* FINI_BO syncs the BO back for the NPU and should always succeed; callers
         * universally ignore its return, so a failure here must not be silent (the
         * NPU would then read un-synced host writes). */
        ROCKET_LOGE("rocket_bo_fini FINI_BO handle=%u: %s\n", bo->handle, strerror(e));
        return -e;
    }
    return 0;
}

/* ============================================================================
 * SECTION — Job submission
 * ==========================================================================*/

int rocket_submit_matmul(int fd,
                         const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                         const uint32_t *in_handles,  uint32_t n_in,
                         const uint32_t *out_handles, uint32_t n_out,
                         uint32_t timeout_ms)
{
    (void)timeout_ms; /* CONFIRMED: drm_rocket_submit/job carry no timeout field;
                         the driver applies its own internal job timeout. */

    /* drm_rocket_task.regcmd is __u32 — the NPU PC's BASE_ADDRESS register is
     * 32-bit, so the regcmd BO must live in the low 4GB of NPU IOVA space. Fail
     * loudly if it does not, rather than silently truncating a high address (the
     * int8/int4 paths enforce the same window via i8_iova_overflow). */
    if ((regcmd_bo->dma_address >> 32) != 0) {
        ROCKET_LOGE("rocket_submit_matmul: regcmd BO dma_address 0x%llx exceeds "
                "32-bit NPU IOVA window\n", (unsigned long long)regcmd_bo->dma_address);
        return -EINVAL;
    }
    struct drm_rocket_task task = {
        .regcmd       = (uint32_t)regcmd_bo->dma_address,
        .regcmd_count = regcmd_count,
    };

    struct drm_rocket_job job = {
        .tasks              = (uint64_t)(uintptr_t)&task,
        .task_count         = 1,
        .task_struct_size   = sizeof(task),
        .in_bo_handles      = (uint64_t)(uintptr_t)in_handles,
        .in_bo_handle_count = n_in,
        .out_bo_handles     = (uint64_t)(uintptr_t)out_handles,
        .out_bo_handle_count= n_out,
    };

    struct drm_rocket_submit submit = {
        .jobs            = (uint64_t)(uintptr_t)&job,
        .job_count       = 1,
        .job_struct_size = sizeof(job),
    };

    if (ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, &submit) < 0) {
        ROCKET_LOGE("ROCKET_SUBMIT: %s\n", strerror(errno));
        return -errno;
    }
    return 0;
}

/* Scratch (bytes) a caller must provide to rocket_submit_tasks_pre for up to
 * max_tasks tasks. Lets the hot path keep the drm_rocket_task[] array resident
 * instead of calloc/free-ing it on every submit. */
size_t rocket_submit_scratch_size(uint32_t max_tasks)
{
    return (size_t)max_tasks * sizeof(struct drm_rocket_task);
}

/* No-alloc core of rocket_submit_tasks: `scratch` is a caller-owned buffer of at
 * least rocket_submit_scratch_size(n_tasks) bytes, reused across submits. Each
 * drm_rocket_task is fully (re)written via a zeroing compound literal, so any
 * reserved uapi fields stay 0 (the compound literal zero-initializes them); elements
 * past n_tasks are never read (task_count = n_tasks). Same semantics/return as the
 * public shim below. */
int rocket_submit_tasks_pre(int fd, void *scratch,
                            const rocket_task_desc *tasks, uint32_t n_tasks,
                            const uint32_t *in_handles,  uint32_t n_in,
                            const uint32_t *out_handles, uint32_t n_out,
                            int batched)
{
    struct drm_rocket_task *dt = (struct drm_rocket_task *)scratch;
    for (uint32_t i = 0; i < n_tasks; i++) {
        dt[i] = (struct drm_rocket_task){
            .regcmd       = tasks[i].regcmd,
            .regcmd_count = tasks[i].regcmd_count,
        };
    }
    return rkt_submit_one_job(fd, dt, n_tasks, in_handles, n_in,
                              out_handles, n_out, batched);
}

/* Public shim: allocates the per-submit task array and calls the no-alloc core.
 * Kept for external callers/tests; the matmul hot path uses _pre with resident
 * scratch instead. */
int rocket_submit_tasks(int fd,
                        const rocket_task_desc *tasks, uint32_t n_tasks,
                        const uint32_t *in_handles,  uint32_t n_in,
                        const uint32_t *out_handles, uint32_t n_out)
{
    if (n_tasks == 0) return 0;   // nothing to submit; calloc(0) may return NULL
    struct drm_rocket_task *dt = calloc(n_tasks, sizeof(*dt));
    if (!dt) return -ENOMEM;
    /* Public shim keeps the stock gapped per-task path (batched=0); the chained
     * path is opt-in via rocket_submit_tasks_pre's batched arg on the hot path. */
    int r = rocket_submit_tasks_pre(fd, dt, tasks, n_tasks,
                                    in_handles, n_in, out_handles, n_out, 0);
    free(dt);
    return r;
}

int rocket_submit_jobs(int fd, const rocket_job_desc *jobs, uint32_t n_jobs)
{
    if (n_jobs == 0) return 0;   // nothing to submit; calloc(0) may return NULL (cf. rocket_submit_tasks)
    struct drm_rocket_job *dj = calloc(n_jobs, sizeof(*dj));
    struct drm_rocket_task **dts = calloc(n_jobs, sizeof(*dts));
    if (!dj || !dts) { free(dj); free(dts); return -ENOMEM; }

    int r = 0;
    for (uint32_t j = 0; j < n_jobs; j++) {
        dts[j] = calloc(jobs[j].n_tasks, sizeof(struct drm_rocket_task));
        if (!dts[j]) { r = -ENOMEM; goto cleanup; }
        for (uint32_t i = 0; i < jobs[j].n_tasks; i++) {
            dts[j][i].regcmd       = jobs[j].tasks[i].regcmd;
            dts[j][i].regcmd_count = jobs[j].tasks[i].regcmd_count;
        }
        dj[j] = (struct drm_rocket_job){
            .tasks               = (uint64_t)(uintptr_t)dts[j],
            .task_count          = jobs[j].n_tasks,
            .task_struct_size    = sizeof(struct drm_rocket_task),
            .in_bo_handles       = (uint64_t)(uintptr_t)jobs[j].in_handles,
            .in_bo_handle_count  = jobs[j].n_in,
            .out_bo_handles      = (uint64_t)(uintptr_t)jobs[j].out_handles,
            .out_bo_handle_count = jobs[j].n_out,
        };
    }

    struct drm_rocket_submit submit = {
        .jobs            = (uint64_t)(uintptr_t)dj,
        .job_count       = n_jobs,
        .job_struct_size = sizeof(struct drm_rocket_job),
    };
    if (ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, &submit) < 0) {
        ROCKET_LOGE("ROCKET_SUBMIT(%u jobs): %s\n", n_jobs, strerror(errno));
        r = -errno;
    }

cleanup:
    for (uint32_t j = 0; j < n_jobs; j++) free(dts[j]);
    free(dts); free(dj);
    return r;
}
