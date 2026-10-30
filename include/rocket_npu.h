// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * Declares the submit seam: device open and close, buffer objects and their cache
 * maintenance, job submission, and the capability and counter queries. It is a thin
 * userspace layer over the mainline DRM-accel `rocket` driver for the RK3588 and RK3576
 * NPUs. It wraps the kernel uAPI (`CREATE_BO`, `SUBMIT`, `PREP_BO`, `FINI_BO`) so that
 * the matmul layer can allocate NPU buffers, submit register programs, and wait on their
 * output.
 *
 * Build against the kernel uAPI header:
 *
 *     #include <drm/rocket_accel.h>
 *
 * The kernel provides it as include/uapi/drm/rocket_accel.h, and it is usually installed
 * to /usr/include/drm/rocket_accel.h. It provides the `DRM_IOCTL_ROCKET_*` macros and the
 * `drm_rocket_*` structs used below.
 */
#ifndef ROCKET_NPU_H
#define ROCKET_NPU_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * Status codes and the buffer-object type
 * ==========================================================================*/

/* The status codes the library's int-returning entry points share. Each is a named
 * constant for a value the library already returns, such as `ROCKET_E_TILING` == -2.
 *
 * The error test is `ret < 0` for every entry point. The submit seam in this header
 * returns a negative errno on failure: `rocket_open`, `rocket_bo_alloc`, `rocket_bo_prep`,
 * `rocket_bo_fini`, their ranged forms, and each submitting `rocket_submit_*` function.
 * Two seam functions also return a `rocket_status`. `rocket_bo_alloc32` returns
 * `ROCKET_E_DEVICE` for a BO above the 32-bit window, and `rocket_submit_jobs` returns
 * `ROCKET_E_SHAPE` for a job with no tasks.
 *
 * The other entry points return `ROCKET_OK` (0) on success and a negative `rocket_status`
 * on failure. When a seam call fails inside one, it can return that call's negative errno
 * unchanged. The two sets share values, so -5 is both -EIO and `ROCKET_E_UNSUPPORTED`, and
 * a value alone does not say which set it came from.
 *
 * The matmul `*_plan()` previews return a positive tile or job count on success. For them,
 * `>= 0` is success and `< 0` is an error, so a caller must not test a plan result with
 * `!= 0`. A few internal routing paths return a negative code to mean "take another path"
 * rather than a hard error. Each one documents this where it is defined. */

#ifdef __cplusplus
extern "C" {
#endif
enum rocket_status {
    ROCKET_OK            =  0,  /* success */
    ROCKET_E_SHAPE       = -1,  /* unsupported shape or bad parameter */
    ROCKET_E_TILING      = -2,  /* shape valid but not tileable here (caller can fall back) */
    ROCKET_E_NOMEM       = -3,  /* host allocation failed */
    ROCKET_E_DEVICE      = -4,  /* device, driver or ioctl failure */
    ROCKET_E_UNSUPPORTED = -5,  /* feature disabled or not implemented on this path */
    ROCKET_E_RANGE       = -6,  /* a value left the range of an fp16 output. The output is
                                   not valid, and a wider-precision path must compute it */
};

/* A buffer object (BO): NPU-visible memory that the CPU can also map. */
typedef struct {
    uint32_t handle;       /* GEM handle, for residency lists and destroy     */
    uint64_t dma_address;  /* NPU-side device VA, the address to program      */
    uint64_t mmap_offset;  /* offset arg for mmap() on the accel fd           */
    size_t   size;
    void    *ptr;          /* CPU mapping, set by `rocket_bo_alloc`           */
    /* Memory-object address. The mainline `rocket` driver keys every BO ioctl on the GEM
     * handle and leaves this 0. The vendor `rknpu` driver keys `MEM_SYNC` on the object
     * address instead (its DRM build keys `MEM_DESTROY` on the handle), and a provider for
     * it stores the value here. Where a provider does not need it, the field is unused: no
     * other library code reads it or programs it into a register. */
    uint64_t obj_addr;
} rocket_bo;

/* ============================================================================
 * Device open and close
 * ==========================================================================*/

/* Opens /dev/accel/accel0, or the node `ROCKET_DEV` names, and checks that its driver name
 * is "rocket". Returns the fd (>= 0) on success and a negative errno on error.
 *
 * On an RK3576 with more than one core bound, a second open while an fd is held returns
 * -EBUSY. Two fds put two jobs in flight at once, and on that part two jobs in flight
 * compute wrong answers. `ROCKET_RK3576_ALLOW_MULTI_FD=1` lifts the refusal for a caller
 * that serializes its own submits. */
int  rocket_open(void);
void rocket_close(int fd);

/* Sets and reads the calling thread's CPU-affinity base, which spreads several context
 * pools across the big cores in one process.
 *
 * The library pins its fan-out workers to the big (A76) cores, round-robin from a
 * per-thread base. With one context pool the base stays 0, and the workers take big cores
 * 0, 1, 2 and onward. Several context pools can run concurrently in one process, such as
 * one detector instance per camera on its own thread. Each pool thread must then set a
 * distinct base once, before it creates or uses its contexts. Otherwise the pools' workers
 * collide on the same cores. The usual choice is `base = pool_index * nthreads`:
 *
 *     // on pool thread p, before rocket_*_ctx_create / matmul / conv calls:
 *     rocket_affinity_set_base(p * nthreads_per_pool);
 *
 * The base is thread-local, and the workers that thread spawns inherit it. It is a
 * scheduling hint only and never changes numerics. `ROCKET_CPU_AFFINITY` sets the big-core
 * set for the whole process, and the base selects where in that set a thread's workers
 * start. A base of 0, the default, gives the same placement as never calling it. */
void rocket_affinity_set_base(int base);
int  rocket_affinity_get_base(void);

/* Pins the calling thread to a big (A76) core. `worker_idx` selects the core round-robin
 * over the detected big cluster, offset by this thread's affinity base. The NPU fan-out
 * workers use it, and host-side worker pools, such as dequantization or encoder glue,
 * should use it too. Otherwise their threads land on the A55 little cores, and a join
 * barrier waits on the slowest of them. If pinning is disabled or no distinct big cluster
 * is found, it does nothing. It never changes numerics. */
void rocket_pin_worker(int worker_idx);

/* Returns the number of big (A76) cores. If pinning is disabled or there is no distinct
 * big cluster, it returns 0. Size a host worker pool to this count rather than to
 * `hardware_concurrency`. The A55 little cores are ~7x slower on fp16 dequantization. An
 * equal-chunk fan-out that includes them runs several times slower than one confined to
 * the big cluster. */
int  rocket_num_big_cores(void);

/* ============================================================================
 * Buffer allocation, cache management and fence polling
 * ==========================================================================*/

/* Allocates a BO with `CREATE_BO` and maps it with `mmap`. The kernel rounds `size` to the
 * page size. On success, `bo->ptr` is a valid CPU pointer, and `bo->dma_address` is the VA
 * to program into `CNA_FEATURE_DATA_ADDR`, `DPU_DST_BASE_ADD` or `decompress_addr0`. */
int  rocket_bo_alloc(int fd, size_t size, rocket_bo *bo);

/* Allocates a BO as `rocket_bo_alloc` does, and checks that its last byte lies inside the
 * low 4 GB. Every BO whose address a regcmd programs needs this check, because the regcmd
 * address fields are 32 bits wide.
 *
 * Caution: the datapath reads the address of a BO whose last byte leaves the low 4 GB as
 * wrapped. The program then writes a complete output of the right size with wrong values.
 * The mainline `rocket` driver gives each fd a 4 GB IOVA window, and a resident-weight
 * workload can reach the end of it.
 *
 * The window is set by the driver rather than by the silicon. The vendor `rknpu` BSP
 * driver has one domain shared process-wide, and it fragments over a board's uptime.
 * Freshly booted, it serves ~3.9 GB at every size, as mainline does. After a day of use,
 * the same board served 35 BOs of 16 MiB, one of 128 MiB and none of 192 MiB.
 *
 * On that driver a second fd adds no address space, and a second process competes for the
 * same budget. A large contiguous request can also be refused while smaller ones still
 * fit. Software should therefore treat a NOMEM on a large BO as transient, and retry
 * smaller first.
 *
 * Returns `ROCKET_E_DEVICE` if the BO lies above the window, which is an address
 * constraint rather than a shortage of memory. Returns the negative errno of
 * `rocket_bo_alloc` if the allocation itself failed. In both cases the BO is left freed,
 * so a caller that retries starts clean rather than with a BO it must not use. */
int  rocket_bo_alloc32(int fd, size_t size, rocket_bo *bo);

/* Grows `bo` to at least `need` bytes, keeping it inside the 32-bit IOVA window.
 *
 * Returns 1 if it reallocated the BO, 0 if the existing BO already fit, and a negative
 * value on failure, with the BO left freed. A caller that relies on zeroed padding must
 * re-zero it after a reallocation and must not after a reuse. */
int  rocket_bo_ensure32(int fd, rocket_bo *bo, size_t need);
void rocket_bo_free(int fd, rocket_bo *bo);

/* Perform cache maintenance around CPU access to a BO. `rocket` BOs are cached, so the
 * caller must bracket every CPU read or write: `rocket_bo_prep`, then the CPU access
 * (`memcpy` or inspection), then `rocket_bo_fini`. `rocket_bo_fini` flushes CPU writes for
 * the NPU. `rocket_bo_prep` before reading results invalidates the CPU cache, so the CPU
 * sees the NPU's writes.
 *
 * `dir` states the caller's intent, 0 for a read and 1 for a write. The builtin provider
 * ignores it: `rocket_bo_prep` always invalidates and `rocket_bo_fini` always flushes.
 *
 * `timeout_ns` is a relative duration, and 0 makes the call a non-blocking poll. The
 * library converts it to the kernel's absolute `CLOCK_MONOTONIC` deadline. When waiting on
 * a submitted job's output, pass a real value such as 2e9, or the call returns -EBUSY. */
int  rocket_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t timeout_ns);
int  rocket_bo_fini(int fd, rocket_bo *bo);

/* A byte range within a BO, for the ranged forms below. */
typedef struct { uint64_t offset, size; } rocket_bo_range;

/* Perform the cache maintenance of `rocket_bo_prep` and `rocket_bo_fini` over only the
 * named ranges of the BO. A bracket costs what it walks: about 10.6 GB/s per direction on
 * an RK3576, plus a fixed 1.1 us per ioctl. Without ranges, a caller that touches a few
 * bytes of a large output takes the time of the whole output.
 *
 * The ranges must be ascending, must not overlap, and must lie inside the BO. The kernel
 * checks all three and refuses a violation with -EINVAL. With `n` == 0 the call covers the
 * whole object, so each ranged form is a superset of its plain form.
 *
 * On a kernel without the ranged ioctls (interface < 1.5), the calls fall back to the
 * whole-BO form, which is always correct and never faster. A caller can therefore use them
 * unconditionally, and `rocket_bo_ranges_supported()` says only whether the saving is
 * available.
 *
 * Caution: the ranges passed to `rocket_bo_fini_ranges` must cover everything the CPU
 * dirtied. A range left out is not written back, and the NPU then reads stale memory. */
int  rocket_bo_prep_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r,
                           unsigned n, uint64_t timeout_ns);
int  rocket_bo_fini_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r,
                           unsigned n);

/* Returns 1 if the running kernel has `DRM_ROCKET_PREP_BO_RANGES` and
 * `DRM_ROCKET_FINI_BO_RANGES` (interface >= 1.5). The result is probed once and cached. */
int  rocket_bo_ranges_supported(void);

/* Returns 1 if the running kernel honors `DRM_ROCKET_JOB_BATCHED` (per-job chained
 * submit) and its master switch is on, and 0 otherwise. The result is probed once and
 * cached. Chaining is a joint layout contract with the kernel, so software must not send
 * a self-chained job to a kernel that would ignore the flag. Callers must condition on
 * this function, not on the `ROCKET_BATCH_SUBMIT` environment variable alone. */
int  rocket_batched_submit_supported(void);

/* Returns 1 if the chained layout is this uAPI's native multi-program mode, and 0 if
 * chaining is an extension the caller has to ask for. Where it returns 1, a multi-task
 * submit should use the chained layout unless the caller says otherwise. The result sets
 * the default of `ROCKET_BATCH_SUBMIT`, and the environment variable still forces either
 * way. It is a separate question from `rocket_batched_submit_supported()`, which says
 * whether chaining works at all.
 *
 * Mainline returns 0. The stock kernel runs a job's tasks one per IRQ. The chained layout
 * needs both a patch and a module parameter, so chaining stays opt-in there. The
 * vendor `rknpu` path returns 1. Its job programs only the first task's address, so a
 * multi-program job can only be a chain. The chain is also the only shape that keeps a
 * batch to one job, which CBUF reuse needs (see `rocket_submit_batch_atomic()`).
 *
 * Measured on the vendor provider at 512×3840×4096 fp16 over three warm rounds, a chained
 * submit takes 36.8-39.3 ms against 68-77 ms unchained. With the driver's own core
 * scheduler as well it takes 26.1-27.7 ms, against 23.5-24.4 ms for the same shape on
 * mainline.
 *
 * The result is probed once and cached. */
int  rocket_batched_submit_native(void);

/* Returns 1 if the running kernel honors `DRM_ROCKET_JOB_PPU_DONE` (retire a pooling job
 * on the PPU's own completion), and 0 otherwise. The result is probed once and cached.
 * Callers must condition on this function. A kernel whose job struct has a flag word
 * rejects a bit it does not recognize. Such a kernel older than interface version 1.3
 * therefore fails the submit rather than ignoring the bit. */
int  rocket_ppu_done_supported(void);

/* Returns whether `ROCKET_JOB_NO_DPU_DONE` is safe to set on the running kernel. The
 * version number alone cannot answer this. The two SoCs' drivers number their interfaces
 * independently, and the flag sets behind a given minor version diverge. The RK3576 gains
 * this bit at 1.2, and the RK3588 series does not have it at any version. A kernel whose
 * job struct has a flag word rejects any bit it does not recognize.
 *
 * Caution: without this check, the bit turns one hint into -EINVAL on every submit. */
int  rocket_no_dpu_done_supported(void);

/* Returns 1 if the running kernel bases a batched job's completion wait on the task
 * counter (RK3576, interface 1.4), and 0 otherwise. The wait then starts once the whole
 * kick has retired. The result is probed once and cached.
 *
 * The result decides a refusal rather than a flag. On an older kernel `PC_DONE` is raised
 * per task, so the wait starts at the first program. A long chained stream can then retire
 * with programs still to run. The chain length a caller can build is capped on such a
 * kernel, and is the SoC's own limit on a tracking one. */
int  rocket_batch_completion_tracked(void);

/* Returns 1 if the tasks of one submit call run as one uninterrupted job, with no other
 * job (from this fd or any other context) scheduled between them. Returns 0 if the provider
 * decomposes a multi-task submit into independent jobs. The result is probed once and
 * cached.
 *
 * CBUF operand reuse depends on this contract. A tile with the CNA `WEIGHT_REUSE` or
 * `DATA_REUSE` bit set reads its operand from CBUF instead of fetching it again. The read
 * is sound only while the previous tile's operand is still there, which holds while
 * nothing else has run on that core. A caller that sets those bits must have either a
 * chained (single-kick) layout or this function returning 1.
 *
 * Caution: when the contract does not hold, the failure is silent. The output is complete
 * and the right size, with plausible values that change between runs, because the CBUF
 * held whatever the interleaving job left. Periodic inputs hide the failure, so a test
 * must vary its data.
 *
 * Evidence: measured on the vendor `rknpu` provider, where an unchained multi-task submit
 * is n ioctls and therefore n jobs. One 128×1024×1024 fp16 matmul corrupted 29 of 120 runs
 * with reuse on, 0 of 120 with it off, and 0 of 80 chained. */
int  rocket_submit_batch_atomic(void);

/* Returns the kernel driver this build reaches, as a short name for a device listing or a
 * log line. The builtin provider returns "mainline rocket driver". An external provider
 * returns the name its build gave itself, such as "vendor rknpu driver" for
 * `rknpu-submit`, or "external submit provider" when nothing named it.
 *
 * The name is fixed at build time rather than probed, because the provider is chosen at
 * build time. The call makes no ioctl and is valid before any device is open. It is not
 * part of the submit seam, so a provider defines nothing for it, and
 * `-DROCKETNPU_DRIVER_NAME=...` sets it.
 *
 * Print this rather than a literal. A frontend that hardcodes one driver's name tells
 * every user of the other driver that the wrong kernel path is live. It does so on the
 * platform where knowing which path is live matters most. */
const char *rocket_driver_name(void);

/* Sets how long a blocking wait spin-polls the completion fence, in microseconds, before
 * it sleeps. It overrides `ROCKET_BUSY_POLL`, which sets the default, and `us` <= 0
 * disables the spin. It is a single-stream latency option for tiny submit-bound jobs with
 * a small output BO. Spinning occupies a CPU core, and each poll re-invalidates the output
 * BO's cache, so it helps only when that output is small. The function is mainly for
 * comparisons within one process, and `ROCKET_BUSY_POLL` is the setting for production. */
void rocket_busy_poll_set_us(long us);

/* ============================================================================
 * Job and task submission
 * ==========================================================================*/

/* Submits one job containing one task, which is one register program:
 *
 * - `regcmd_bo`: the BO holding the packed NPUOP uint64 words (the `ops[]` array).
 * - `regcmd_count`: the total NPUOP words the PC executes, including the control trailer
 *   whose final `OP_ENABLE` triggers the compute. A generator returns it in
 *   `task_count`: 122 + 4 trailer = 126 for `gen_matmul_fp16` and `gen_matmul_int8`. The
 *   driver encodes it to hardware as (count+1)/2 - 1.
 * - `in_handles`: the BOs the NPU reads (input, weights, `regcmd_bo`).
 * - `out_handles`: the BOs the NPU writes (output).
 *
 * The builtin provider returns once the job is queued and ignores `timeout_ms`, so the
 * caller must wait on an output BO with `rocket_bo_prep`. The `rknpu-submit` provider
 * blocks until the job completes or `timeout_ms` expires. Returns 0 on success and a
 * negative errno on error. The builtin provider returns -EINVAL for a `regcmd_bo` whose
 * address is above 4 GB, because the PC's base-address register is 32 bits wide.
 *
 * Caution: a `regcmd_count` that is too small hangs the NPU silently until the job times
 * out, and the call returns no error. */
int  rocket_submit_matmul(int fd,
                          const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                          const uint32_t *in_handles,  uint32_t n_in,
                          const uint32_t *out_handles, uint32_t n_out,
                          uint32_t timeout_ms);

/* Submits as `rocket_submit_matmul` does, with a `ROCKET_JOB_*` bitmask (below). */
int  rocket_submit_matmul_flags(int fd,
                                const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                                const uint32_t *in_handles,  uint32_t n_in,
                                const uint32_t *out_handles, uint32_t n_out,
                                uint32_t job_flags);

/* Per-job submit flags, matching the kernel's `DRM_ROCKET_JOB_*` bits. The library passes
 * them to the kernel unchanged.
 *
 * A kernel whose job struct has no flag word does not run a flagged job as asked. Without
 * extensible submit structs it never reads the word and ignores every flag. With them
 * (interface 1.0), it refuses a nonzero flag word with -E2BIG. A kernel that has the flag
 * word rejects a bit it does not recognize with -EINVAL. Before setting a bit, software
 * must check its probe: `rocket_batched_submit_supported()`,
 * `rocket_no_dpu_done_supported()` or `rocket_ppu_done_supported()`. The library's own
 * entry points do so.
 *
 * `ROCKET_JOB_BATCHED`: the job's tasks are laid out contiguously and self-chained. Each
 * task's trailer must point the PC at the next task with two writes. A `PC_BASE_ADDRESS`
 * write gives the next task's regcmd address. `PC_REGISTER_AMOUNTS` gives its word count,
 * encoded as the driver encodes `regcmd_count`. The `gen_*` trailer is [`OP_NONE`,
 * `PC_REGISTER_AMOUNTS`, `OP_40`, `OP_ENABLE`], and the address write takes its `OP_NONE`
 * slot. The last task's forward link must be cleared back to `OP_NONE`.
 *
 * The driver runs the tasks in one kick with a single completion, instead of one kick and
 * one completion per task. The caller must check `rocket_batched_submit_supported()`
 * first. A kernel that ignores the flag runs the chained layout down the per-task path,
 * which stalls.
 *
 * `ROCKET_JOB_NO_DPU_DONE`: the job's program raises no DPU completion. The driver's wait
 * past `PC_DONE` is then a blind settle (`rocket.dpu_blind_us`, 250 us), rather than a wait
 * for a completion that is coming. The driver cannot tell the two classes apart, and the
 * caller can, because it emitted the program. The RK3576 driver has the flag from
 * interface version 1.2, and the RK3588 driver has it at no version.
 *
 * On the RK3576 the narrow and wide int32 writers raise none, and neither does a job the
 * wide-output hazard reaches, which writes nothing. Without the hint, on `rocket` 1.6.0,
 * each such job waits out the driver's 125 ms backstop, and the kernel logs its
 * retirement. The dense int32 writer, the int32 entry's default, raises its own and goes
 * out without the hint. The fp16 convolution programs (the input-channel slice and the
 * packed-image first conv) and the one-byte int8 program raise one too.
 *
 * Caution: the completion class is a correctness property as well as a cost. `PC_DONE` on
 * a one-task job marks the end of the kick, not of the program. Under the hint the fence
 * comes at the settle whatever the program runs, so a program still writing returns with
 * its tail unwritten. A completion that arrives first still retires the job at once. The
 * hint is therefore harmless only on a program that is done before the settle ends. Set it
 * on a program that raises no completion, and read its whole output against a stamp: the
 * fence says nothing about the writes.
 *
 * Evidence: [HW sweep, H96 MAX M9, rocket 1.6.0, 2026-09-27], `tests/rk3576_drain_probe
 * raw`. Under the hint, an fp16 conv (k9 over 32 channels on a 120×64 plane) fenced at
 * 0.74 ms with the last rows of every 8-channel plane unwritten until 1.18 ms later (10
 * of 10 runs). Without it, the conv fenced at 2.05 ms with every element written (10 of 10
 * runs).
 *
 * `ROCKET_JOB_PPU_DONE`: the job's last program is a PPU (pooling) program, whose
 * completion is the PPU's own bit rather than the DPU's. A pool enables no DPU stage:
 * `PC_OPERATION_ENABLE` is a per-block bitmap, and a pool sets 0x60 where a convolution
 * sets 0x1d. Without the flag, the bits the driver waits on by default never set, and
 * every pooling job waits out the whole grace period. Setting it on a job whose last
 * program is a convolution costs that job the grace period and nothing else. Both SoCs'
 * drivers have the flag from interface version 1.3.
 *
 * Caution: do not set `ROCKET_JOB_PPU_DONE` on a job that mixes DPU and PPU programs. An
 * interior program's PPU bit would retire the job while a later DPU write was still
 * draining.
 */
#define ROCKET_JOB_BATCHED      (1u << 0)
#define ROCKET_JOB_NO_DPU_DONE  (1u << 1)
#define ROCKET_JOB_PPU_DONE     (1u << 2)

/* Describes one task, which is one register program. `regcmd` is the program's 32-bit NPU
 * IOVA, such as a slot inside a shared regcmd BO, and `regcmd_count` is its word count. */
typedef struct { uint32_t regcmd; uint32_t regcmd_count; } rocket_task_desc;

/* Submits one job containing many tasks, which the NPU runs back to back with a single
 * completion fence for the whole batch. For throughput, it spreads the submit syscall,
 * the fence and IRQ, and the cache sync over many tiles. The call is asynchronous, like
 * `SUBMIT`, and the caller waits by calling `rocket_bo_prep` on the output BOs the tasks
 * wrote. The arrays are:
 *
 * - `tasks`: the per-task regcmd IOVA and word count.
 * - `in_handles`: every BO the job reads (packed input, weights, regcmd BO).
 * - `out_handles`: every BO the job writes (the shared output BO).
 *
 * Returns 0 on success and a negative errno on error. */
int  rocket_submit_tasks(int fd,
                         const rocket_task_desc *tasks, uint32_t n_tasks,
                         const uint32_t *in_handles,  uint32_t n_in,
                         const uint32_t *out_handles, uint32_t n_out);

/* Submits as `rocket_submit_tasks` does, and passes `job_flags` to the kernel unchanged as
 * the job's flag word. With `ROCKET_JOB_BATCHED`, the job's tasks run as one chained kick
 * with a single completion, instead of one kick and one completion per task. The caller
 * must have laid the regcmds out contiguously and self-chained, as `ROCKET_JOB_BATCHED`
 * describes, and must have checked `rocket_batched_submit_supported()`. A kernel without the
 * flag ignores it and runs the chained layout down the per-task path, which stalls, or
 * refuses the submit (see the `ROCKET_JOB_*` flags). */
int  rocket_submit_tasks_flags(int fd,
                               const rocket_task_desc *tasks, uint32_t n_tasks,
                               const uint32_t *in_handles,  uint32_t n_in,
                               const uint32_t *out_handles, uint32_t n_out,
                               uint32_t job_flags);

/* A variant of `rocket_submit_tasks` that allocates nothing, for the hot path. The caller
 * keeps a resident `scratch` buffer of at least `rocket_submit_scratch_size(n_tasks)`
 * bytes and reuses it across submits. No `drm_rocket_task[]` array is then allocated and
 * freed per submit. Its semantics and return are otherwise those of `rocket_submit_tasks`.
 *
 * `job_flags` is a `ROCKET_JOB_*` bitmask. Bit 0 is the batched flag, which runs the job's
 * tasks as one chained kick instead of one kick and one IRQ per task. It is valid only
 * when the caller has laid the tasks' regcmds out contiguously and self-chained, as
 * `ROCKET_JOB_BATCHED` describes. A 0 here selects the stock gapped per-task path. The
 * flag is per job, so a chained job and a gapped job can share one fd. */
size_t rocket_submit_scratch_size(uint32_t max_tasks);
int    rocket_submit_tasks_pre(int fd, void *scratch,
                               const rocket_task_desc *tasks, uint32_t n_tasks,
                               const uint32_t *in_handles,  uint32_t n_in,
                               const uint32_t *out_handles, uint32_t n_out,
                               uint32_t job_flags);

/* Describes one job: tasks that run sequentially on one core. The uAPI runs every task of
 * a job on the same core, for SRAM residency. To use all 3 NPU cores, the caller submits
 * several jobs in a single submit, and the kernel schedules them across cores in
 * dependency order. For concurrency, each job must write a separate output BO. With a
 * shared output BO, the kernel serializes the jobs on the write-after-write dependency. */
typedef struct {
    const rocket_task_desc *tasks;  uint32_t n_tasks;
    const uint32_t *in_handles;     uint32_t n_in;
    const uint32_t *out_handles;    uint32_t n_out;
} rocket_job_desc;

/* Submits `n_jobs` jobs in one ioctl. The call is asynchronous: the caller waits by
 * calling `rocket_bo_prep` on each job's output BO. */
int  rocket_submit_jobs(int fd, const rocket_job_desc *jobs, uint32_t n_jobs);

/* Return how many `ROCKET_SUBMIT` ioctls this process has issued, and how many tasks they
 * contained. On the RK3576, a fixed cost of ~439 us per submit dominates wall-clock time.
 * The number of submits a call cost is therefore the first question about any layer's
 * time.
 *
 * A job of n tasks is one submit containing n tasks, whether it is chained
 * (`ROCKET_JOB_BATCHED`) or gapped. A path that submits each task alone is n submits
 * containing one task each. Reading both counters tells those two apart. The counters do
 * not count kicks, so they cannot tell a chained job from a gapped one.
 *
 * The counters are process-wide and atomic. They count every submit, including the ones
 * the fan-out paths issue from several worker threads at once. No caller needs to
 * serialize submits for the counts to be right. They cannot attribute a submit to a
 * caller, because a reset or a read sees every thread's submits. A per-call figure is
 * therefore that call's alone only when nothing else in the process submits between the
 * reset and the read. */
uint64_t rocket_submit_ioctl_count(void);
uint64_t rocket_submit_task_count(void);
void     rocket_submit_counters_reset(void);

/* Return how many fenced waits this process made (`rocket_bo_prep` and
 * `rocket_bo_prep_ranges` with `timeout_ns` > 0), how many reached the slow-wait mark, and
 * the longest, in microseconds.
 *
 * Caution: the `rocket` driver retires a job that never completes and signals its fence.
 * It does so at 500 ms on the RK3588 and at a 125 ms backstop in the RK3576 series.
 * `PREP_BO` then returns 0 exactly as it does for a job that completed. A hung job
 * therefore looks like a slow wait over whatever the output BO held before. A test that
 * compares that BO against a reference can pass on it.
 *
 * A wait at or past the mark also logs a warning naming the BO. The mark is
 * `ROCKET_SLOW_WAIT_MS`, whose default is the SoC's hardware profile value: 450 ms on the
 * RK3588 and 110 on the RK3576.
 *
 * The counts are a heuristic in one direction only. A wait past the mark can also be one
 * that covered other jobs queued ahead of it. A caller that waited late can see a retired
 * job end under the mark, so zero is not proof that nothing timed out. The kernel log's
 * "NPU job timed out" line is that proof. The builtin provider counts its waits, and an
 * external provider that does not report its waits leaves all three at zero. The counters
 * are process-wide and atomic. */
uint64_t rocket_fence_wait_count(void);
uint64_t rocket_fence_wait_slow_count(void);
uint64_t rocket_fence_wait_max_us(void);
void     rocket_fence_wait_counters_reset(void);

/* Returns how many RK3576 jobs this process's write guards scored as retired rather than
 * completed. The counts are split by whether the guard's own write check saw the output
 * written.
 *
 * Every RK3576 entry that stamps its output (convolution, depthwise, packed image, fp16
 * convolution, the chained kick, the int8, int32 and fp16 matmuls, pooling) times each job
 * from before its submit to its fence. A job at or past the driver's backstop was
 * retired. The backstop is the hardware profile's `backstop_ms`, 125 on the RK3576, and
 * `ROCKET_RK3576_BACKSTOP_US` overrides it.
 *
 * The entry redoes a retired job after a power cycle, exactly as it redoes a job that
 * wrote nothing. It refuses once its attempts run out. `written` counts the retirements
 * that a write check alone would have passed, a partially written output, which is the
 * class this scoring exists for. Each one also logs a warning naming the entry.
 *
 * The count is exact in one direction. The kick follows the submit's start, so every
 * retirement at the backstop is counted. A job queued behind another process's job can be
 * counted too, and costs one redo. The count cannot see a job the kernel retires early,
 * when the DPU grace is shorter than the drain, because that job finishes fast. The
 * counters are process-wide and atomic, and either pointer can be NULL. */
void rocket_rk3576_retired_counts(uint64_t *unwritten, uint64_t *written);
void rocket_rk3576_retired_counts_reset(void);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_NPU_H */
