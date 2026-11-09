// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * pool_stock_kernel_gate.c — host-only GATE: on a driver without the PPU completion, no
 * pooling entry allocates a BO or submits, and the global pools return the host reduction.
 *
 * A pooling program raises no DPU completion. A rocket driver that enables only the DPU
 * interrupts (the stock driver at interface 1.0, and the upstream RK3576 series) retires a
 * pooling job at its 500 ms job timeout with a core reset. rocket_pool_npu_supported() is
 * the library's answer to "can this driver retire one", and every pool entry asks it. The
 * boards here run patched drivers that answer 1, so no device test ever takes the 0 branch.
 * This one does.
 *
 * THE MOCK IS THE WHOLE SUBMIT SEAM, defined in this file, as pool_rk3576_bracket_gate does:
 * with every seam symbol defined here the builtin provider never links. The mock's
 * rocket_ppu_done_supported() returns g_ppu_done. Its rocket_bo_alloc() counts and fails,
 * and its submits count and fail, so any attempt to reach the device is visible.
 *
 *   refused   g_ppu_done = 0: the fp16, int8 and uint8 pools return ROCKET_E_UNSUPPORTED,
 *             the RK3576 pack returns NULL and its transient entry ROCKET_E_UNSUPPORTED,
 *             the three global pools return 0 with the host reference's bytes, and the
 *             mock saw no allocation and no submit.
 *   control   g_ppu_done = 1: the same pool calls reach rocket_bo_alloc(), which proves it
 *             is the predicate, and not something earlier, that stopped them above.
 *
 * No NPU and no device file; forces ROCKET_CHIP=rk3576 itself. Exit 0 or 1, never skips.
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_pool.h"
#include "rocket_reduce.h"

#define MOCK_FD 77

static int g_ppu_done;
static unsigned g_allocs, g_submits;

/* ============================================================================
 * SECTION — the mock submit seam
 * ==========================================================================*/

int rocket_open(void) { return MOCK_FD; }
void rocket_close(int fd) { (void)fd; }

int rocket_bo_alloc(int fd, size_t size, rocket_bo *bo)
{
    (void)fd; (void)size;
    memset(bo, 0, sizeof *bo);
    g_allocs++;
    return -ENOMEM;
}
int rocket_bo_alloc32(int fd, size_t size, rocket_bo *bo) { return rocket_bo_alloc(fd, size, bo); }
int rocket_bo_ensure32(int fd, rocket_bo *bo, size_t need) { return rocket_bo_alloc(fd, need, bo); }
void rocket_bo_free(int fd, rocket_bo *bo) { (void)fd; (void)bo; }
int rocket_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t timeout_ns)
{ (void)fd; (void)bo; (void)dir; (void)timeout_ns; return 0; }
int rocket_bo_fini(int fd, rocket_bo *bo) { (void)fd; (void)bo; return 0; }
int rocket_bo_prep_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r, unsigned n,
                          uint64_t timeout_ns)
{ (void)fd; (void)bo; (void)r; (void)n; (void)timeout_ns; return 0; }
int rocket_bo_fini_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r, unsigned n)
{ (void)fd; (void)bo; (void)r; (void)n; return 0; }

int rocket_bo_ranges_supported(void)       { return 0; }
int rocket_batched_submit_supported(void)  { return 0; }
int rocket_batched_submit_native(void)     { return 0; }
int rocket_ppu_done_supported(void)        { return g_ppu_done; }
int rocket_no_dpu_done_supported(void)     { return 0; }
int rocket_batch_completion_tracked(void)  { return 0; }
int rocket_submit_batch_atomic(void)       { return 0; }

int rocket_submit_matmul_flags(int fd, const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                               const uint32_t *in_handles, uint32_t n_in,
                               const uint32_t *out_handles, uint32_t n_out,
                               uint32_t job_flags)
{
    (void)fd; (void)regcmd_bo; (void)regcmd_count; (void)in_handles; (void)n_in;
    (void)out_handles; (void)n_out; (void)job_flags;
    g_submits++;
    return -EIO;
}
int rocket_submit_matmul(int fd, const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                         const uint32_t *in_handles, uint32_t n_in,
                         const uint32_t *out_handles, uint32_t n_out, uint32_t timeout_ms)
{
    (void)timeout_ms;
    return rocket_submit_matmul_flags(fd, regcmd_bo, regcmd_count, in_handles, n_in,
                                      out_handles, n_out, 0u);
}
int rocket_submit_tasks_flags(int fd, const rocket_task_desc *tasks, uint32_t n_tasks,
                              const uint32_t *in_handles, uint32_t n_in,
                              const uint32_t *out_handles, uint32_t n_out,
                              uint32_t job_flags)
{
    (void)fd; (void)tasks; (void)n_tasks; (void)in_handles; (void)n_in;
    (void)out_handles; (void)n_out; (void)job_flags;
    g_submits++;
    return -EIO;
}
int rocket_submit_tasks(int fd, const rocket_task_desc *tasks, uint32_t n_tasks,
                        const uint32_t *in_handles, uint32_t n_in,
                        const uint32_t *out_handles, uint32_t n_out)
{
    return rocket_submit_tasks_flags(fd, tasks, n_tasks, in_handles, n_in, out_handles,
                                     n_out, 0u);
}
size_t rocket_submit_scratch_size(uint32_t max_tasks) { return (size_t)max_tasks * 16u + 64u; }
int rocket_submit_tasks_pre(int fd, void *scratch, const rocket_task_desc *tasks,
                            uint32_t n_tasks, const uint32_t *in_handles, uint32_t n_in,
                            const uint32_t *out_handles, uint32_t n_out, uint32_t job_flags)
{
    (void)scratch;
    return rocket_submit_tasks_flags(fd, tasks, n_tasks, in_handles, n_in, out_handles,
                                     n_out, job_flags);
}
int rocket_submit_jobs(int fd, const rocket_job_desc *jobs, uint32_t n_jobs)
{
    (void)fd; (void)jobs; (void)n_jobs;
    g_submits++;
    return -EIO;
}
uint64_t rocket_submit_ioctl_count(void) { return g_submits; }
uint64_t rocket_submit_task_count(void)  { return g_submits; }
void     rocket_submit_counters_reset(void) {}

/* ============================================================================
 * SECTION — the checks
 * ==========================================================================*/

static int fails;

#define CHECK(cond, ...) do {                                              \
        if (cond) { printf("  ok     "); printf(__VA_ARGS__); printf("\n"); } \
        else { fails++; printf("  FAIL   "); printf(__VA_ARGS__); printf("\n"); } \
    } while (0)

enum { C = 16, IH = 8, IW = 8, OH = 4, OW = 4 };

int main(void)
{
    static _Float16 in[C * IH * IW], out[C * OH * OW], g[C], gref[C];
    static int8_t i8[C * IH * IW], o8[C * OH * OW];
    rocket_pool_desc d;
    int r;

    setenv("ROCKET_CHIP", "rk3576", 1);    /* the RK3576 pack checks the part first */

    for (int i = 0; i < C * IH * IW; i++) {
        in[i] = (_Float16)((i * 37) % 101 - 50);
        i8[i] = (int8_t)((i * 53) % 255 - 127);
    }
    memset(&d, 0, sizeof d);
    d.c = C; d.ih = IH; d.iw = IW; d.kh = 2; d.kw = 2; d.stride_y = 2; d.stride_x = 2;
    d.method = POOL_METHOD_MAX;

    printf("refused: the driver has no PPU completion\n");
    g_ppu_done = 0;
    g_allocs = g_submits = 0;
    CHECK(rocket_pool_npu_supported() == 0, "rocket_pool_npu_supported() is 0");
    CHECK(rocket_pool_fp16_plan(&d) == 0, "the shape itself is legal (the plan accepts it)");
    r = rocket_pool_fp16(MOCK_FD, &d, in, out);
    CHECK(r == ROCKET_E_UNSUPPORTED, "rocket_pool_fp16 refuses (%d)", r);
    r = rocket_pool_int8(MOCK_FD, &d, i8, o8);
    CHECK(r == ROCKET_E_UNSUPPORTED, "rocket_pool_int8 refuses (%d)", r);
    r = rocket_pool_uint8(MOCK_FD, &d, (const uint8_t *)i8, (uint8_t *)o8);
    CHECK(r == ROCKET_E_UNSUPPORTED, "rocket_pool_uint8 refuses (%d)", r);
    CHECK(rocket_pool_int8_rk3576_plan(&d) == ROCKET_OK, "the RK3576 plan accepts the shape");
    CHECK(rocket_pool_int8_pack_rk3576(MOCK_FD, &d, 0) == NULL, "the RK3576 pack returns NULL");
    r = rocket_pool_int8_rk3576(MOCK_FD, &d, 0, i8, o8);
    CHECK(r == ROCKET_E_UNSUPPORTED, "rocket_pool_int8_rk3576 refuses (%d)", r);

    CHECK(rocket_global_avgpool_plan(C, IH, IW) == 0, "the global-pool plan accepts %dx%d", IH, IW);
    r = rocket_global_avgpool_fp16(MOCK_FD, C, IH, IW, in, g);
    rocket_global_avgpool_ref_fp16(C, IH, IW, in, gref);
    CHECK(r == 0 && !memcmp(g, gref, sizeof g), "rocket_global_avgpool_fp16 is the host reference");
    r = rocket_global_maxpool_fp16(MOCK_FD, C, IH, IW, in, g);
    rocket_global_maxpool_ref_fp16(C, IH, IW, in, gref);
    CHECK(r == 0 && !memcmp(g, gref, sizeof g), "rocket_global_maxpool_fp16 is the host reference");
    r = rocket_global_minpool_fp16(MOCK_FD, C, IH, IW, in, g);
    rocket_global_minpool_ref_fp16(C, IH, IW, in, gref);
    CHECK(r == 0 && !memcmp(g, gref, sizeof g), "rocket_global_minpool_fp16 is the host reference");
    CHECK(g_allocs == 0 && g_submits == 0,
          "nothing reached the device: %u allocations, %u submits", g_allocs, g_submits);

    printf("control: the driver has the PPU completion\n");
    g_ppu_done = 1;
    g_allocs = 0;
    CHECK(rocket_pool_npu_supported() == 1, "rocket_pool_npu_supported() is 1");
    (void)rocket_pool_fp16(MOCK_FD, &d, in, out);
    CHECK(g_allocs > 0, "rocket_pool_fp16 reached rocket_bo_alloc (%u)", g_allocs);
    g_allocs = 0;
    (void)rocket_pool_int8_pack_rk3576(MOCK_FD, &d, 0);
    CHECK(g_allocs > 0, "the RK3576 pack reached rocket_bo_alloc (%u)", g_allocs);

    printf(fails ? "\n%d CHECK(s) FAILED\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
