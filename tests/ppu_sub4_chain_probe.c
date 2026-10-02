// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * ppu_sub4_chain_probe.c — does a pooling pass read a sub-4 cube the PPU itself just wrote?
 *
 * A probe, not a gate. The record says a two-pass PPU chain whose
 * intermediate is 2x2 comes back wrong, about half the channels zero or garbage, while a 4x4
 * intermediate is right, the PPU writes a 2x2 cube correctly, and the PPU_RDMA reads a
 * CPU-scattered 2x2 cube correctly. It was measured while every pool job retired by the
 * kernel's watchdog and a core reset, before the kernel gained a PPU completion class
 * (DRM_ROCKET_JOB_PPU_DONE, rocket 1.3). The library's global pools order their factors
 * smallest-first so no intermediate is below 4; this probe forces the other order.
 *
 * Each case pools a C x H x W fp16 map to 1 x 1 in two passes, kernel k1 then k2, stride
 * equal to the kernel. Three things are scored per case, all per channel or per element:
 *   pass 1   the device-written intermediate, read by the CPU, against a CPU model of pass 1
 *            (bit-exact for MAX, within 2e-3 relative for AVG)
 *   chain    pass 2 reading that intermediate where pass 1 left it, as the library does
 *            (PREP_BO waits and syncs it, nothing writes it, no FINI_BO)
 *   ref      pass 2 over a CPU-scattered copy of the SAME device-written intermediate, in a
 *            fresh BO: the standalone read the record calls correct
 * so `chain` against `ref` is bit-exact exactly when the chained read is right, whatever the
 * pool's arithmetic does. A channel of `chain` that is zero is counted apart, and `chain` is
 * also scored against a CPU model of both passes (bit-exact for MAX, 4e-3 relative for AVG).
 *
 * Every case runs with the PPU completion flag when the kernel has it. The reproduction arm
 * (--no-ppu-done) runs two cases without it, so each pool job retires by the watchdog as the
 * record's did; it costs a kernel "job timed out" line and a core reset per pass, on purpose.
 * Each pass's wait time and the process's slow-fence count are printed, so a retired job is
 * told from a finished one apart from its values.
 *
 * Fills are random fp16 in [-4, 4) with no period. Outputs are prefilled with 0xAA bytes.
 *
 * What a green result does NOT show: chains longer than two passes, int8 pooling, padded or
 * overlapping windows, rectangles with unequal factor counts, a chained multi-task job, or
 * the RK3576.
 *
 * Usage: ppu_sub4_chain_probe [--no-ppu-done]
 * Exit: 0 every chain equals its ref and every control is right, 1 otherwise, 2 no NPU.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_pool.h"
#include "npu_matmul.h"
#include "test_fill.h"

#define SLACK 32768

struct pcase { const char *name; int C, H, k1, k2, method; };

static const struct pcase cases[] = {
    { "max_8_k2k4_ctl",  64,  8, 2, 4, POOL_METHOD_MAX },   /* 4x4 intermediate: control */
    { "max_8_k4k2",      64,  8, 4, 2, POOL_METHOD_MAX },   /* 2x2: the record's case     */
    { "max_6_k3k2",      64,  6, 3, 2, POOL_METHOD_MAX },   /* 2x2                        */
    { "max_12_k4k3",     64, 12, 4, 3, POOL_METHOD_MAX },   /* 3x3                        */
    { "max_16_k8k2",     64, 16, 8, 2, POOL_METHOD_MAX },   /* 2x2 after a large kernel   */
    { "max_8_k4k2_c130", 130, 8, 4, 2, POOL_METHOD_MAX },   /* C % 8 != 0                 */
    { "avg_8_k2k4_ctl",  64,  8, 2, 4, POOL_METHOD_AVG },
    { "avg_8_k4k2",      64,  8, 4, 2, POOL_METHOD_AVG },
    { "avg_12_k4k3",     64, 12, 4, 3, POOL_METHOD_AVG },
};

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static size_t cube_bytes(int C, int H, int W)
{
    return (size_t)((C + 7) / 8) * H * W * 8 * sizeof(_Float16) + SLACK;
}

/* One pooling job src -> dst over a C x ih x ih cube, kernel and stride k. Waits with PREP_BO
 * on dst (read), which leaves dst synced for the CPU. Returns 0, or -1. */
static int pool_pass(int fd, rocket_bo *rc, rocket_bo *src, rocket_bo *dst, int C, int ih, int k,
                     int method, uint32_t flags, double *ms)
{
    uint64_t regs[64] = { 0 };
    pool_params_t p = {
        .c = (uint16_t)C, .ih = (uint16_t)ih, .iw = (uint16_t)ih,
        .oh = (uint16_t)(ih / k), .ow = (uint16_t)(ih / k),
        .kh = (uint8_t)k, .kw = (uint8_t)k, .stride_y = (uint8_t)k, .stride_x = (uint8_t)k,
        .method = (uint8_t)method,
        .recip_w = ppu_recip_kernel_fp16(k), .recip_h = ppu_recip_kernel_fp16(k),
        .input_dma = (uint32_t)src->dma_address, .output_dma = (uint32_t)dst->dma_address,
        .tasks = regs,
    };
    if (gen_pool_fp16(&p) != 0 || p.task_count > 64) { printf("    generator refused\n"); return -1; }
    rocket_bo_prep(fd, rc, 1, 0);
    memcpy(rc->ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, rc);
    rocket_task_desc task = { .regcmd = (uint32_t)rc->dma_address, .regcmd_count = p.task_count };
    uint32_t in_h[] = { src->handle, rc->handle }, out_h[] = { dst->handle };
    double t0 = now_ms();
    int r = rocket_submit_tasks_flags(fd, &task, 1, in_h, 2, out_h, 1, flags);
    if (r == 0) r = rocket_bo_prep(fd, dst, 0, 2000000000ull);
    *ms = now_ms() - t0;
    if (r != 0) printf("    submit or wait failed (%d)\n", r);
    return r ? -1 : 0;
}

/* CPU model of one non-overlapping pass over a [C][h][h] tensor. AVG sums in double and
 * divides by k*k; the device's reciprocal multiply is within the tolerance used on it. */
static void model_pass(const double *in, double *out, int C, int h, int k, int method)
{
    const int o = h / k;
    for (int c = 0; c < C; c++)
        for (int y = 0; y < o; y++)
            for (int x = 0; x < o; x++) {
                double acc = method == POOL_METHOD_MAX ? -INFINITY : 0;
                for (int dy = 0; dy < k; dy++)
                    for (int dx = 0; dx < k; dx++) {
                        const double v = in[((size_t)c * h + y * k + dy) * h + x * k + dx];
                        if (method == POOL_METHOD_MAX) { if (v > acc) acc = v; }
                        else acc += v;
                    }
                out[((size_t)c * o + y) * o + x] = method == POOL_METHOD_MAX ? acc : acc / (k * k);
            }
}

static int f16_bits(_Float16 v) { uint16_t b; memcpy(&b, &v, 2); return b; }

static int run_case(int fd, const struct pcase *pc, uint32_t flags, uint64_t seed)
{
    const int C = pc->C, H = pc->H, m1 = H / pc->k1;
    const size_t nin = (size_t)C * H * H, nmid = (size_t)C * m1 * m1;
    _Float16 *x = malloc(nin * 2), *mid = malloc(nmid * 2);
    double *xd = malloc(nin * sizeof(double)), *want1 = malloc(nmid * sizeof(double));
    rocket_bo guard = { 0 }, rc = { 0 }, A = { 0 }, B = { 0 }, D = { 0 }, S = { 0 }, T = { 0 };
    int bad = 1;
    if (!x || !mid || !xd || !want1 ||
        rocket_bo_alloc(fd, 4096, &guard) || rocket_bo_alloc(fd, 4096, &rc) ||
        rocket_bo_alloc(fd, cube_bytes(C, H, H), &A) || rocket_bo_alloc(fd, cube_bytes(C, m1, m1), &B) ||
        rocket_bo_alloc(fd, cube_bytes(C, 1, 1), &D) || rocket_bo_alloc(fd, cube_bytes(C, m1, m1), &S) ||
        rocket_bo_alloc(fd, cube_bytes(C, 1, 1), &T)) {
        printf("  %s: allocation failed\n", pc->name); goto out;
    }
    tf_fill_f16(x, nin, seed, -4.0, 4.0);
    for (size_t i = 0; i < nin; i++) xd[i] = x[i];
    model_pass(xd, want1, C, H, pc->k1, pc->method);

    /* stage the input cube, stamp every other BO */
    rocket_bo_prep(fd, &A, 1, 0);
    memset(A.ptr, 0, A.size);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < H; w++)
                ((_Float16 *)A.ptr)[feature_data(C, H, H, 8, c + 1, h + 1, w + 1)] = x[((size_t)c * H + h) * H + w];
    rocket_bo_fini(fd, &A);
    if (tf_sentinel_bo(fd, &B) || tf_sentinel_bo(fd, &D) || tf_sentinel_bo(fd, &T)) goto out;

    uint64_t slow0 = rocket_fence_wait_slow_count();
    double ms1, ms2, ms3;
    if (pool_pass(fd, &rc, &A, &B, C, H, pc->k1, pc->method, flags, &ms1)) goto out;

    /* pass 1 as the CPU reads it (B is synced for the CPU by the wait; read only) */
    long wrong1 = 0;
    for (int c = 0; c < C; c++)
        for (int h = 0; h < m1; h++)
            for (int w = 0; w < m1; w++) {
                const _Float16 v = ((_Float16 *)B.ptr)[feature_data(C, m1, m1, 8, c + 1, h + 1, w + 1)];
                const double wv = want1[((size_t)c * m1 + h) * m1 + w];
                mid[((size_t)c * m1 + h) * m1 + w] = v;
                const int ok = pc->method == POOL_METHOD_MAX ? (double)v == wv
                             : fabs((double)v - wv) <= 2e-3 * fmax(1.0, fabs(wv));
                if (!ok) wrong1++;
            }

    /* chain: pass 2 reads B where pass 1 left it */
    if (pool_pass(fd, &rc, &B, &D, C, m1, pc->k2, pc->method, flags, &ms2)) goto out;

    /* ref: pass 2 over a CPU-scattered copy of the same intermediate */
    rocket_bo_prep(fd, &S, 1, 0);
    memset(S.ptr, 0, S.size);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < m1; h++)
            for (int w = 0; w < m1; w++)
                ((_Float16 *)S.ptr)[feature_data(C, m1, m1, 8, c + 1, h + 1, w + 1)] = mid[((size_t)c * m1 + h) * m1 + w];
    rocket_bo_fini(fd, &S);
    if (pool_pass(fd, &rc, &S, &T, C, m1, pc->k2, pc->method, flags, &ms3)) goto out;
    const int slow = rocket_fence_wait_slow_count() != slow0;

    /* the whole reduction on the CPU, pass 2 applied to the model of pass 1 */
    double want2[512];
    model_pass(want1, want2, C, m1, pc->k2, pc->method);
    long diff = 0, zero = 0, sentinel = 0, first = -1, off_model = 0;
    double worst = 0;
    for (int c = 0; c < C; c++) {
        const _Float16 d = ((_Float16 *)D.ptr)[feature_data(C, 1, 1, 8, c + 1, 1, 1)];
        const _Float16 t = ((_Float16 *)T.ptr)[feature_data(C, 1, 1, 8, c + 1, 1, 1)];
        if (f16_bits(d) != f16_bits(t)) { diff++; if (first < 0) first = c; }
        if (f16_bits(d) == 0 && f16_bits(t) != 0) zero++;
        if (tf_is_sentinel_f16(d)) sentinel++;
        if (fabs((double)d - (double)t) > worst) worst = fabs((double)d - (double)t);
        const int ok = pc->method == POOL_METHOD_MAX ? (double)d == want2[c]
                     : fabs((double)d - want2[c]) <= 4e-3 * fmax(1.0, fabs(want2[c]));
        if (!ok) off_model++;
    }
    rocket_bo_fini(fd, &D);
    rocket_bo_fini(fd, &T);
    rocket_bo_fini(fd, &B);
    printf("  %-16s %s C %3d %2dx%-2d k%d->%dx%d k%d: pass1 %4ld of %5zu wrong | chain vs ref %3ld of %3d "
           "channels differ (%ld zero, %ld unwritten, worst %.4g), %ld off the model | waits %.2f %.2f "
           "%.2f ms%s\n",
           pc->name, flags ? "ppu_done" : "no_flag ", C, H, H, pc->k1, m1, m1, pc->k2, wrong1, nmid,
           diff, C, zero, sentinel, worst, off_model, ms1, ms2, ms3, slow ? " [SLOW FENCE]" : "");
    if (diff && first >= 0) printf("    first differing channel %ld\n", first);
    bad = (wrong1 || diff || off_model) ? 1 : 0;
out:
    rocket_bo_free(fd, &T); rocket_bo_free(fd, &S); rocket_bo_free(fd, &D);
    rocket_bo_free(fd, &B); rocket_bo_free(fd, &A); rocket_bo_free(fd, &rc); rocket_bo_free(fd, &guard);
    free(x); free(mid); free(xd); free(want1);
    return bad;
}

int main(int argc, char **argv)
{
    int repro = 0, bad = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-ppu-done")) repro = 1;
        else { fprintf(stderr, "usage: %s [--no-ppu-done]\n", argv[0]); return 1; }
    }
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }
    const uint32_t flags = rocket_ppu_done_supported() ? ROCKET_JOB_PPU_DONE : 0u;
    const int n = (int)(sizeof(cases) / sizeof(cases[0]));
    printf("== PPU sub-4 chain probe: %d cases, two pooling passes each; PPU completion flag %s ==\n",
           n, flags ? "supported, set" : "NOT supported by this kernel");
    for (int i = 0; i < n; i++) bad += run_case(fd, &cases[i], flags, 0x6100 + (uint64_t)i);
    if (repro) {
        printf("== reproduction: the control and the record's case with no completion flag "
               "(each job retires by the watchdog) ==\n");
        bad += run_case(fd, &cases[0], 0u, 0x6200);
        bad += run_case(fd, &cases[1], 0u, 0x6201);
    }
    printf("== %d case(s) not clean; slow-fence count %llu ==\n", bad,
           (unsigned long long)rocket_fence_wait_slow_count());
    rocket_close(fd);
    return bad ? 1 : 0;
}
