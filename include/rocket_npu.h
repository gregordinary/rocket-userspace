// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_npu.h — thin userspace shim over the mainline DRM-accel "rocket"
 * driver for the RK3588 NPU. It wraps the kernel uAPI (CREATE_BO / SUBMIT /
 * PREP_BO / FINI_BO) so the matmul layer can allocate NPU buffers, submit
 * register-command programs, and fence on their output.
 *
 * Build against the kernel uAPI header:
 *     #include <drm/rocket_accel.h>
 * (ships as include/uapi/drm/rocket_accel.h; usually installed to
 *  /usr/include/drm/rocket_accel.h). It provides the DRM_IOCTL_ROCKET_*
 *  macros and the drm_rocket_* structs used below.
 */
#ifndef ROCKET_NPU_H
#define ROCKET_NPU_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * SECTION — Status codes & buffer-object type
 * ==========================================================================*/

/* Shared status vocabulary for the library's int-returning entry points. The
 * values match the conventions already in use, so they read as named constants
 * for the historical magic numbers (e.g. ROCKET_E_TILING == -2).
 *
 * UNIFORM CONTRACT: the canonical error test is `ret < 0` everywhere. A function
 * returns ROCKET_OK (0) on success and a negative rocket_status on failure. The
 * one documented elaboration is the matmul *_plan() previews, which additionally
 * return a POSITIVE tile/job count on success (so `>= 0` is success, `< 0` is the
 * error) — never test a plan result with `!= 0`. A handful of routing sentinels
 * deliberately reuse a negative code as "not a hard error, take another path"
 * (e.g. mm_compute_kacc returns ROCKET_E_TILING for a tiny last M-tile so the
 * caller falls back to the CPU-accum oracle); those are documented at the site. */

#ifdef __cplusplus
extern "C" {
#endif
enum rocket_status {
    ROCKET_OK            =  0,  /* success */
    ROCKET_E_SHAPE       = -1,  /* unsupported shape / bad parameter */
    ROCKET_E_TILING      = -2,  /* shape valid but not tileable here (caller may fall back) */
    ROCKET_E_NOMEM       = -3,  /* host allocation failed */
    ROCKET_E_DEVICE      = -4,  /* device / driver / ioctl failure */
    ROCKET_E_UNSUPPORTED = -5,  /* feature gated or not implemented on this path */
};

/* A buffer object: NPU-visible memory we can also mmap on the CPU. */
typedef struct {
    uint32_t handle;       /* GEM handle (for residency lists + destroy)      */
    uint64_t dma_address;  /* NPU-side device VA — patch THIS into registers  */
    uint64_t mmap_offset;  /* offset arg for mmap() on the accel fd           */
    size_t   size;
    void    *ptr;          /* CPU mapping (NULL until rocket_bo_map)          */
} rocket_bo;

/* ============================================================================
 * SECTION — Device open / close
 * ==========================================================================*/

/* Open /dev/accel/accel0 and sanity-check the driver name == "rocket".
 * Returns fd >= 0, or negative errno. */
int  rocket_open(void);
void rocket_close(int fd);

/* ---- CPU-affinity base (in-process multi-pool spreading) -------------------
 * The library pins its fan-out workers to the big (A76) cores, round-robin from
 * a per-thread BASE. One context pool leaves the base 0 (workers -> big cores
 * 0,1,2,...). To run SEVERAL context pools CONCURRENTLY in one process (e.g. one
 * detector instance per camera, each on its own thread) without their workers
 * colliding on the same cores, each pool thread sets a distinct base ONCE before
 * creating/using its context(s) — conventionally base = pool_index * nthreads:
 *
 *     // on pool thread p, before rocket_*_ctx_create / matmul / conv calls:
 *     rocket_affinity_set_base(p * nthreads_per_pool);
 *
 * The base is thread-local and is inherited by the workers that thread spawns. It
 * is a SCHEDULING hint only and never changes numerics. This is the in-process
 * analogue of the per-process ROCKET_CPU_AFFINITY env (which sets the big-core
 * SET); the base selects WHERE in that set a thread's workers start. Calling it
 * with the default 0 (or never calling it) preserves the historical behaviour. */
void rocket_affinity_set_base(int base);
int  rocket_affinity_get_base(void);

/* ============================================================================
 * SECTION — Buffer allocation, cache management & fence polling
 * ==========================================================================*/

/* CREATE_BO + mmap. `size` is rounded to page size by the kernel.
 * On success bo->ptr is a valid CPU pointer and bo->dma_address is the VA to
 * program into CNA_FEATURE_DATA_ADDR / DPU_DST_BASE_ADD / decompress_addr0. */
int  rocket_bo_alloc(int fd, size_t size, rocket_bo *bo);
void rocket_bo_free(int fd, rocket_bo *bo);

/* Cache management around CPU access. rocket BOs are cached, so you MUST
 * bracket CPU reads/writes:
 *   prep -> memcpy/inspect on CPU -> fini  (fini flushes for the NPU;
 *   prep before reading results invalidates so you see NPU writes).
 * dir: 0 = read (invalidate), 1 = write (will flush on fini).
 * timeout_ns is a RELATIVE duration (0 = non-blocking poll); the shim converts
 * it to the kernel's absolute CLOCK_MONOTONIC deadline. Use a real value (e.g.
 * 2e9) when waiting on a submitted job's output, else you get -EBUSY. */
int  rocket_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t timeout_ns);
int  rocket_bo_fini(int fd, rocket_bo *bo);

/* 1 if the running kernel honors DRM_ROCKET_JOB_BATCHED (per-job chained submit)
 * AND its master switch is on; 0 otherwise. Probed once and cached. Chaining is a
 * joint layout contract with the kernel, so a kernel that would ignore the flag
 * must not be self-chained into -- callers must gate on this, not on the
 * ROCKET_BATCH_SUBMIT env var alone. */
int  rocket_batched_submit_supported(void);

/* Spin-poll the completion fence for up to `us` microseconds before a blocking
 * wait falls asleep (overrides the ROCKET_BUSY_POLL env, which sets the default;
 * us<=0 disables). A single-stream latency lever for tiny submit-bound jobs with
 * a small output BO — see the note at rocket_bo_prep's definition. Mainly for
 * in-process A/B harnesses (the env knob is the production path). */
void rocket_busy_poll_set_us(long us);

/* ============================================================================
 * SECTION — Job / task submission
 * ==========================================================================*/

/* Submit one job containing one task = one register-command program.
 *   regcmd_bo     : BO holding the packed NPUOP uint64 words (the ops[] array)
 *   regcmd_count  : TOTAL NPUOP words the PC executes, incl. the control trailer
 *                   whose final OP_ENABLE triggers the compute (104 + 4 trailer
 *                   = 108 for gen_matmul_*). Driver encodes to HW as
 *                   (count+1)/2 - 1. NOTE: too-small a count silently hangs the
 *                   NPU (job timed out), not an error return.
 *   in_handles[]  : BOs the NPU reads  (input, weights, regcmd_bo)
 *   out_handles[] : BOs the NPU writes (output)
 * Blocks until the job completes (or timeout). Returns 0 / negative errno. */
int  rocket_submit_matmul(int fd,
                          const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                          const uint32_t *in_handles,  uint32_t n_in,
                          const uint32_t *out_handles, uint32_t n_out,
                          uint32_t timeout_ms);

/* One task = one register-command program. regcmd is the 32-bit NPU IOVA of that
 * program (e.g. a slot inside a shared regcmd BO), regcmd_count its word count. */
typedef struct { uint32_t regcmd; uint32_t regcmd_count; } rocket_task_desc;

/* Submit ONE job containing many tasks (the NPU runs them back-to-back, a single
 * completion fence for the whole batch). The throughput path: amortise the submit
 * syscall + fence/IRQ + cache-sync over many tiles. ASYNC like SUBMIT — wait by
 * PREP_BO'ing the output BO(s) the tasks wrote.
 *   tasks[]       : per-task {regcmd IOVA, count}
 *   in_handles[]  : all BOs the job reads  (packed input, weights, regcmd BO)
 *   out_handles[] : all BOs the job writes (the shared output BO)
 * Returns 0 / negative errno. */
int  rocket_submit_tasks(int fd,
                         const rocket_task_desc *tasks, uint32_t n_tasks,
                         const uint32_t *in_handles,  uint32_t n_in,
                         const uint32_t *out_handles, uint32_t n_out);

/* No-alloc variant of rocket_submit_tasks for the hot path: the caller
 * keeps a resident `scratch` buffer of >= rocket_submit_scratch_size(n_tasks)
 * bytes and reuses it across submits, so no drm_rocket_task[] is calloc/free'd
 * per submit. Identical semantics/return to rocket_submit_tasks otherwise.
 *
 * `batched`: set the per-job DRM_ROCKET_JOB_BATCHED flag — run the job's tasks
 * as one chained HW kick instead of one submit/IRQ per task. ONLY valid when the
 * caller has laid the tasks' regcmds out contiguously and self-chained (the
 * rocket_chain.c helpers under ROCKET_BATCH_SUBMIT); a 0 here is the stock gapped
 * per-task path. Per-job, so a chained job and a gapped job can share one fd. */
size_t rocket_submit_scratch_size(uint32_t max_tasks);
int    rocket_submit_tasks_pre(int fd, void *scratch,
                               const rocket_task_desc *tasks, uint32_t n_tasks,
                               const uint32_t *in_handles,  uint32_t n_in,
                               const uint32_t *out_handles, uint32_t n_out,
                               int batched);

/* One job = tasks that run sequentially on ONE core (uapi: same job -> same core,
 * for SRAM residency). To use all 3 NPU cores, submit MULTIPLE jobs in a single
 * submit — the kernel schedules them across cores in dependency order. For real
 * concurrency each job must write a SEPARATE output BO (shared out BO => the
 * kernel serialises on the write-after-write dependency). */
typedef struct {
    const rocket_task_desc *tasks;  uint32_t n_tasks;
    const uint32_t *in_handles;     uint32_t n_in;
    const uint32_t *out_handles;    uint32_t n_out;
} rocket_job_desc;

/* Submit N jobs in ONE ioctl. ASYNC: wait by PREP_BO'ing each job's output BO. */
int  rocket_submit_jobs(int fd, const rocket_job_desc *jobs, uint32_t n_jobs);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_NPU_H */
