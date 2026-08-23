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

/* Pin the CALLING thread to a big (A76) core; worker_idx selects which by round-robin over
 * the detected big cluster (+ this thread's affinity base). The NPU fan-out workers use this;
 * host-side worker pools (dequant, encoder glue) should too, so they are not scattered onto
 * the A55 little cores where a join barrier stalls on the slow straggler. No-op if pinning is
 * disabled or no distinct big cluster is found. Never changes numerics. */
void rocket_pin_worker(int worker_idx);

/* Number of big (A76) cores, or 0 if pinning is disabled / no distinct big cluster. Size a
 * host worker pool to this instead of hardware_concurrency: the A55 little cores are a
 * ~7x-slower straggler on fp16 dequant, so an equal-chunk fan-out that includes them runs
 * several times slower than one confined to the big cluster. */
int  rocket_num_big_cores(void);

/* ============================================================================
 * SECTION — Buffer allocation, cache management & fence polling
 * ==========================================================================*/

/* CREATE_BO + mmap. `size` is rounded to page size by the kernel.
 * On success bo->ptr is a valid CPU pointer and bo->dma_address is the VA to
 * program into CNA_FEATURE_DATA_ADDR / DPU_DST_BASE_ADD / decompress_addr0. */
int  rocket_bo_alloc(int fd, size_t size, rocket_bo *bo);

/* rocket_bo_alloc, plus the check every regcmd-programmed BO owes the hardware: the
 * address fields are 32 BITS WIDE, so a BO whose last byte leaves the low 4 GB encodes
 * an address the datapath reads wrapped — a full, correctly sized, entirely wrong
 * surface. A per-fd IOVA window is 4 GB and a resident-weight workload can walk right
 * up to it, so this is reachable rather than theoretical.
 *
 * Folding the check into the allocation is the point: it was previously written out at
 * each call site as `((a + sza) | (b + szb) | ...) >> 32`, present fourteen times in the
 * RK3588 matmul and absent from the RK3576 entries entirely — which allocate twenty-eight
 * BOs between them and program every one of those addresses into a 32-bit field.
 *
 * Frees the BO and returns ROCKET_E_DEVICE if it lands high (an address constraint, not
 * a shortage of memory) and ROCKET_E_NOMEM if the allocation itself failed. Either way
 * the BO is left freed, so a caller that can retry gets a clean slate rather than a BO
 * it must not use. */
int  rocket_bo_alloc32(int fd, size_t size, rocket_bo *bo);

/* Grow `bo` to at least `need` bytes, keeping it inside the 32-bit IOVA window.
 *
 * Returns 1 if it REALLOCATED, 0 if the existing BO already fit, <0 on failure (the BO
 * is left freed). The distinction is the whole contract: a caller that relies on zeroed
 * padding has to re-zero after a realloc and must not after a reuse. There were two of
 * these with OPPOSITE conventions -- one returned 0 for both cases -- so neither could
 * be substituted for the other without reading both. */
int  rocket_bo_ensure32(int fd, rocket_bo *bo, size_t need);
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

/* A byte range within a BO, for the ranged forms below. */
typedef struct { uint64_t offset, size; } rocket_bo_range;

/* As rocket_bo_prep / rocket_bo_fini, but the cache maintenance covers only the
 * named ranges instead of the whole object — the cost of a bracket is what it
 * WALKS (about 10.6 GB/s per direction on an RK3576, against a 1.1 us per-ioctl
 * floor), so a caller that touches a few bytes of a large surface otherwise pays
 * for the surface.
 *
 * Ranges must be ASCENDING and non-overlapping and lie inside the BO; the kernel
 * checks all three and refuses with -EINVAL. `n == 0` means the whole object, so
 * these are supersets of the plain forms.
 *
 * A kernel that does not carry the ranged ioctls (interface < 1.5) falls back to
 * the whole-BO form, which is always correct and never faster — so a caller may
 * use these unconditionally, and rocket_bo_ranges_supported() only says whether
 * the saving is available. FINI's ranges must cover everything the CPU dirtied:
 * a range left out is not written back and the NPU then reads stale memory. */
int  rocket_bo_prep_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r,
                           unsigned n, uint64_t timeout_ns);
int  rocket_bo_fini_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r,
                           unsigned n);

/* 1 if the running kernel carries DRM_ROCKET_PREP_BO_RANGES / _FINI_BO_RANGES
 * (interface >= 1.5). Probed once and cached. */
int  rocket_bo_ranges_supported(void);

/* 1 if the running kernel honors DRM_ROCKET_JOB_BATCHED (per-job chained submit)
 * AND its master switch is on; 0 otherwise. Probed once and cached. Chaining is a
 * joint layout contract with the kernel, so a kernel that would ignore the flag
 * must not be self-chained into -- callers must gate on this, not on the
 * ROCKET_BATCH_SUBMIT env var alone. */
int  rocket_batched_submit_supported(void);

/* 1 if the running kernel honors DRM_ROCKET_JOB_PPU_DONE (retire a pooling job on
 * the PPU's own completion); 0 otherwise. Probed once and cached. Callers MUST gate
 * on this: the submit ioctl REJECTS a flag word it does not recognise, so an older
 * kernel fails the submit rather than ignoring the bit. */
int  rocket_ppu_done_supported(void);

/* 1 if the running kernel arms a BATCHED job's completion wait on the task counter,
 * so the wait starts once the whole kick has retired (RK3576, interface 1.4); 0
 * otherwise. Probed once and cached. Gates a REFUSAL, not a flag: on an older kernel
 * PC_DONE is per task, so the wait starts at the FIRST program and a long chained
 * stream can retire with programs still to run — the chain length a caller may build
 * is capped there and is the part's own here. */
int  rocket_batch_completion_tracked(void);

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

/* rocket_submit_matmul with a ROCKET_JOB_* bitmask (see below). */
int  rocket_submit_matmul_flags(int fd,
                                const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                                const uint32_t *in_handles,  uint32_t n_in,
                                const uint32_t *out_handles, uint32_t n_out,
                                uint32_t job_flags);

/* Per-job submit flags, matching the kernel's DRM_ROCKET_JOB_* bits.
 *
 * ROCKET_JOB_BATCHED: this job's tasks are laid out contiguously and
 *   self-chained (rocket_chain.c), so the driver runs them from ONE HW kick
 *   with a single completion instead of one submit and one completion each.
 *   Requires rocket_batched_submit_supported() -- a kernel that ignores the
 *   flag runs the chained layout down the per-task path, which stalls.
 *
 * ROCKET_JOB_NO_DPU_DONE: every task in this job has a DPU output element
 *   wider than one byte, so on the RK3576 it raises no DPU completion at all
 *   and the driver's wait past PC_DONE is a blind settle rather than a
 *   deadline on a completion that is coming. The driver cannot tell the two
 *   classes apart; the caller can, because it emitted the program. ADVISORY:
 *   a completion that does arrive retires the job immediately whatever this
 *   says, so a wrong hint costs time and not correctness. Ignored by a kernel
 *   older than interface version 1.2.
 *
 * ROCKET_JOB_PPU_DONE: this job's last program is a PPU program (pooling),
 *   whose completion is the PPU's own bit and not the DPU's. A pool enables no
 *   DPU stage at all -- PC_OPERATION_ENABLE is a per-block bitmap and a pool
 *   sets 0x60 against a convolution's 0x1d -- so the bits the driver waits on
 *   by default can never set and every pooling job pays the whole grace period.
 *   Setting it on a job whose last program is a convolution costs that job the
 *   grace period and nothing else. Do NOT set it on a job that mixes DPU and
 *   PPU programs: an interior program's PPU bit would retire the job while a
 *   later DPU write was still draining. Ignored by a kernel older than
 *   interface version 1.3.
 */
#define ROCKET_JOB_BATCHED      (1u << 0)
#define ROCKET_JOB_NO_DPU_DONE  (1u << 1)
#define ROCKET_JOB_PPU_DONE     (1u << 2)

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
 * `job_flags`: ROCKET_JOB_* bitmask. Bit 0 is the batched flag — run the job's tasks
 * as one chained HW kick instead of one submit/IRQ per task. ONLY valid when the
 * caller has laid the tasks' regcmds out contiguously and self-chained (the
 * rocket_chain.c helpers under ROCKET_BATCH_SUBMIT); a 0 here is the stock gapped
 * per-task path. Per-job, so a chained job and a gapped job can share one fd. */
/* As rocket_submit_tasks, but sets the per-job batched flag: the job's tasks run
 * as ONE chained HW kick with a single completion instead of one submit and one
 * completion each. The caller must have laid the regcmds out contiguously and
 * self-chained (rocket_chain.c) and must have checked
 * rocket_batched_submit_supported() — an older kernel ignores the flag and runs
 * the chained layout down the per-task path, which stalls. */
int  rocket_submit_tasks_flags(int fd,
                               const rocket_task_desc *tasks, uint32_t n_tasks,
                               const uint32_t *in_handles,  uint32_t n_in,
                               const uint32_t *out_handles, uint32_t n_out,
                               uint32_t job_flags);

size_t rocket_submit_scratch_size(uint32_t max_tasks);
int    rocket_submit_tasks_pre(int fd, void *scratch,
                               const rocket_task_desc *tasks, uint32_t n_tasks,
                               const uint32_t *in_handles,  uint32_t n_in,
                               const uint32_t *out_handles, uint32_t n_out,
                               uint32_t job_flags);

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

/* How many ROCKET_SUBMIT ioctls this process has issued, and the tasks they carried.
 * The RK3576's wall is dominated by a per-submit floor (~439 us), so "how many submits
 * did this call cost" is the first question about any layer's time — a chained job
 * (ROCKET_JOB_BATCHED) is ONE submit carrying n tasks, an unchained one is n submits
 * carrying one task each, and the two are told apart by reading both counters.
 * Process-wide and NOT thread-safe (plain non-atomic counters, as with the rest of the
 * context APIs); read them around a single-threaded call. */
uint64_t rocket_submit_ioctl_count(void);
uint64_t rocket_submit_task_count(void);
void     rocket_submit_counters_reset(void);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_NPU_H */
