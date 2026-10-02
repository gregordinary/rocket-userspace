// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rowmajor_kacc_chain_probe.c — the row-major matmul form in the job the library actually
 * issues: one chained fp16 K-accumulation kick with DATA_REUSE, priced against the cube program.
 *
 * A probe, not a gate. rowmajor_tile_probe timed single
 * tasks: the form at a tile's own width runs 0.96-1.12x the cube program, the A read at a wide
 * pitch costs up to 1.21x at one tile shape, and writing into a wide C grows with its pitch. The
 * library's fp16 path does not issue single tasks. It chains every task of a tile group into one
 * kick, K slices outer and N tiles inner, each K slice after the first adding the previous
 * slice's fp16 partial through the DPU_RDMA, and with DATA_REUSE the CNA keeps a K slice of A in
 * the CBUF across the N tiles that follow. So a row-major A read is paid once per K slice, not
 * once per task, and only the last slice's write is the output. This measures that job.
 *
 * One M tile (M rows), K in nKt slices of Kt, N in nNt tiles of Nt: nKt x nNt tasks, one kick.
 * Partials ping-pong between two BOs of nNt fp16 cube slots, as rocket_matmul's KACC chain does.
 * Arms, each scored per element over a sentinel-filled output against a double reference:
 *   cube      A scattered into per-slice cube slots, the final slice writes an fp16 cube slot
 *             the host would de-tile: the shipped program
 *   rm_own    A read in place (row-major [M][K], line stride one row of K), partials still
 *             cubes, the final slice writing each tile row-major at its own width into a
 *             compact slot the host would place with a row copy
 *   rm_wide   the same, the final slice writing straight into a row-major C of N columns
 * each with DATA_REUSE on (`_r`) and off. Operands are integers in [-1, 1] and K <= 2048, so
 * every fp16 partial is exact and equality is the test.
 *
 * Timing: all arms interleaved with the order rotated each rep, the wait on a 4 KiB BO the job
 * lists as an output (PREP_BO syncs the whole BO it waits on), every arm's BOs the same size.
 * Medians of submit-to-fence and of the per-rep ratio to `cube_r`.
 *
 * What a green result does NOT show: several M tiles or tile groups, a multi-core fan-out,
 * resident weights (the weights here are packed once, outside the timed region, which is what
 * residency makes them), the host side of the price, or a model.
 *
 * Usage: rowmajor_kacc_chain_probe [M K N Kt Nt [reps]]   default 256 1536 1024 384 256, 30 reps
 * Exit: 0 every arm exact, 1 otherwise, 2 no NPU, 3 no chained submit on this kernel.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_chain.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "test_fill.h"

#define RC_STRIDE 256            /* gapped slot stride in u64 words, unused when chained */
#define MAX_TASKS 128

enum { CUBE, RM_OWN, RM_WIDE };
struct arm { const char *name; int form, reuse; };
static const struct arm arms[] = {
    { "cube_r", CUBE, 1 }, { "rm_own_r", RM_OWN, 1 }, { "rm_wide_r", RM_WIDE, 1 },
    { "cube", CUBE, 0 }, { "rm_own", RM_OWN, 0 }, { "rm_wide", RM_WIDE, 0 },
};
#define NARMS ((int)(sizeof(arms) / sizeof(arms[0])))

static int M, K, N, Kt, Nt, nKt, nNt;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int patch(uint64_t *ops, int n, uint16_t target, uint16_t reg, uint32_t value, int orbits)
{
    for (int i = 0; i < n; i++) {
        if ((uint16_t)(ops[i] >> 48) != target || (uint16_t)(ops[i] & 0xFFFF) != reg) continue;
        uint32_t v = orbits ? (uint32_t)(ops[i] >> 16) | value : value;
        ops[i] = NPUOP(target, v, reg);
        return 0;
    }
    return -1;
}

/* Every arm's buffers: A in both layouts, the packed weight slots, the two partial ping-pong
 * BOs, the compact final slots, the wide C, the regcmd and the fence. */
struct bufs { rocket_bo a_cube, a_rm, w, p[2], fin, c, rc, fence; };

static int bufs_alloc(int fd, struct bufs *b)
{
    const size_t slot = (size_t)M * Nt * 2;
    return rocket_bo_alloc32(fd, (size_t)nKt * M * Kt * 2, &b->a_cube) ||
           rocket_bo_alloc32(fd, (size_t)M * K * 2, &b->a_rm) ||
           rocket_bo_alloc32(fd, (size_t)nNt * nKt * Nt * Kt * 2, &b->w) ||
           rocket_bo_alloc32(fd, nNt * slot + 64, &b->p[0]) ||
           rocket_bo_alloc32(fd, nNt * slot + 64, &b->p[1]) ||
           rocket_bo_alloc32(fd, nNt * slot + 64, &b->fin) ||
           rocket_bo_alloc32(fd, (size_t)M * N * 2 + 64, &b->c) ||
           rocket_bo_alloc32(fd, (size_t)MAX_TASKS * RC_STRIDE * 8, &b->rc) ||
           rocket_bo_alloc32(fd, 4096, &b->fence);
}

static void bufs_free(int fd, struct bufs *b)
{
    rocket_bo *all[] = { &b->a_cube, &b->a_rm, &b->w, &b->p[0], &b->p[1], &b->fin, &b->c, &b->rc, &b->fence };
    for (unsigned i = 0; i < sizeof all / sizeof all[0]; i++) rocket_bo_free(fd, all[i]);
}

/* Stage A both ways and the weight slots (ni, ki), once. */
static void stage(int fd, struct bufs *b, const _Float16 *A, const _Float16 *B)
{
    rocket_bo_prep(fd, &b->a_cube, 1, 0);
    rocket_bo_prep(fd, &b->a_rm, 1, 0);
    rocket_bo_prep(fd, &b->w, 1, 0);
    memset(b->a_cube.ptr, 0, b->a_cube.size);
    memset(b->w.ptr, 0, b->w.size);
    memcpy(b->a_rm.ptr, A, (size_t)M * K * 2);
    for (int ki = 0; ki < nKt; ki++) {
        _Float16 *s = (_Float16 *)b->a_cube.ptr + (size_t)ki * M * Kt;
        for (int m = 1; m <= M; m++)
            for (int k = 1; k <= Kt; k++)
                s[feature_data(Kt, M, 1, 8, k, m, 1)] = A[(size_t)(m - 1) * K + ki * Kt + k - 1];
    }
    for (int ni = 0; ni < nNt; ni++)
        for (int ki = 0; ki < nKt; ki++) {
            _Float16 *s = (_Float16 *)b->w.ptr + (size_t)(ni * nKt + ki) * Nt * Kt;
            for (int n = 1; n <= Nt; n++)
                for (int k = 1; k <= Kt; k++)
                    s[weight_fp16(Kt, n, k)] = B[(size_t)(ni * Nt + n - 1) * K + ki * Kt + k - 1];
        }
    rocket_bo_fini(fd, &b->a_cube);
    rocket_bo_fini(fd, &b->a_rm);
    rocket_bo_fini(fd, &b->w);
}

/* Emit and pack one arm's whole chained job. Returns the task count, or -1. */
static int build(const struct arm *a, struct bufs *b, rocket_task_desc *td)
{
    const size_t slot = (size_t)M * Nt * 2;
    uint64_t ops[256];
    uint32_t count = 0;
    int nt = 0;
    for (int ki = 0; ki < nKt; ki++) {
        rocket_bo *dst = &b->p[ki & 1], *src = &b->p[(ki + 1) & 1];
        const int last = (ki == nKt - 1);
        for (int ni = 0; ni < nNt; ni++) {
            memset(ops, 0, sizeof ops);
            uint32_t in_dma = a->form == CUBE
                ? (uint32_t)b->a_cube.dma_address + (uint32_t)((size_t)ki * M * Kt * 2)
                : (uint32_t)b->a_rm.dma_address + (uint32_t)((size_t)ki * Kt * 2);
            uint32_t out_dma = (uint32_t)dst->dma_address + (uint32_t)(ni * slot);
            if (last && a->form == RM_OWN) out_dma = (uint32_t)b->fin.dma_address + (uint32_t)(ni * slot);
            if (last && a->form == RM_WIDE) out_dma = (uint32_t)b->c.dma_address + (uint32_t)((size_t)ni * Nt * 2);
            matmul_params_t p = {
                .m = (uint16_t)M, .k = (uint16_t)Kt, .n = (uint16_t)Nt,
                .input_dma = in_dma,
                .weights_dma = (uint32_t)b->w.dma_address + (uint32_t)((size_t)(ni * nKt + ki) * Nt * Kt * 2),
                .output_dma = out_dma, .tasks = ops, .fp32tofp16 = 1,
                .accumulate = (uint8_t)(ki > 0),
                .add_dma = ki > 0 ? (uint32_t)src->dma_address + (uint32_t)(ni * slot) : 0,
                .data_reuse = (uint8_t)(a->reuse && ni > 0),
            };
            if (gen_matmul_fp16(&p) != 0) { printf("  %s: generator refused\n", a->name); return -1; }
            const int n = (int)p.task_count;
            if (a->form != CUBE &&
                (patch(ops, n, OP_REG_CNA, CNA_CONV_CON1, 1u << 29, 1) ||
                 patch(ops, n, OP_REG_CNA, CNA_DMA_CON1, (uint32_t)(K * 2 / 16), 0) ||
                 patch(ops, n, OP_REG_CNA, CNA_DMA_CON2, 0, 0))) return -1;
            if (last && a->form != CUBE) {
                const uint32_t notch = (uint32_t)((a->form == RM_OWN ? Nt : N) * 2 / 16 - 1);
                if (patch(ops, n, OP_REG_DPU, DPU_DST_SURF_STRIDE, 1u << 4, 0) ||
                    patch(ops, n, OP_REG_DPU, DPU_DATA_CUBE_NOTCH_ADDR, (notch << 16) | notch, 0) ||
                    patch(ops, n, OP_REG_DPU, DPU_SURFACE_ADD, 2u << 4, 0)) return -1;
            }
            if (count && (uint32_t)n != count) { printf("  %s: program lengths differ\n", a->name); return -1; }
            count = (uint32_t)n;
            if (rkt_chain_pack(1, &b->rc, td, nt, ops, count, RC_STRIDE) != 0) {
                printf("  %s: chain pack refused task %d\n", a->name, nt); return -1;
            }
            nt++;
        }
    }
    rkt_chain_seal(1, &b->rc, nt, count);
    if (rkt_chain_verify(&b->rc, td, nt) != 0) { printf("  %s: chain does not verify\n", a->name); return -1; }
    return nt;
}

static double arm_run(int fd, const struct arm *a, struct bufs *b, const double *want, int score, int *bad)
{
    rocket_task_desc td[MAX_TASKS];
    rocket_bo_prep(fd, &b->rc, 1, 0);
    memset(b->rc.ptr, 0, b->rc.size);
    const int nt = build(a, b, td);
    rocket_bo_fini(fd, &b->rc);
    if (nt < 0) { *bad = 1; return -1; }
    tf_sentinel_bo(fd, &b->p[0]); tf_sentinel_bo(fd, &b->p[1]);
    tf_sentinel_bo(fd, &b->fin); tf_sentinel_bo(fd, &b->c);
    uint32_t in_h[] = { b->a_cube.handle, b->a_rm.handle, b->w.handle, b->rc.handle };
    uint32_t out_h[] = { b->p[0].handle, b->p[1].handle, b->fin.handle, b->c.handle, b->fence.handle };
    uint64_t slow0 = rocket_fence_wait_slow_count();
    double t0 = now_ms();
    int rc = rocket_submit_tasks_flags(fd, td, (uint32_t)nt, in_h, 4, out_h, 5, ROCKET_JOB_BATCHED);
    if (rc == 0) rc = rocket_bo_prep(fd, &b->fence, 0, 2000000000ull);
    const double ms = now_ms() - t0;
    if (rc == 0) rocket_bo_fini(fd, &b->fence);
    if (rc != 0) { printf("  %s: submit or wait failed (%d)\n", a->name, rc); *bad = 1; return -1; }
    const int slow = rocket_fence_wait_slow_count() != slow0;
    if (!score) { if (slow) *bad = 1; return ms; }

    rocket_bo *res = a->form == CUBE ? &b->p[(nKt - 1) & 1] : a->form == RM_OWN ? &b->fin : &b->c;
    rocket_bo_prep(fd, res, 0, 2000000000ull);
    const _Float16 *o = res->ptr;
    long wrong = 0, unwritten = 0;
    for (int ni = 0; ni < nNt; ni++)
        for (int m = 0; m < M; m++)
            for (int n = 0; n < Nt; n++) {
                size_t at;
                if (a->form == CUBE) at = (size_t)ni * M * Nt + (size_t)feature_data(Nt, M, 1, 8, n + 1, m + 1, 1);
                else if (a->form == RM_OWN) at = (size_t)ni * M * Nt + (size_t)m * Nt + n;
                else at = (size_t)m * N + (size_t)ni * Nt + n;
                if (tf_is_sentinel_f16(o[at])) unwritten++;
                if ((double)o[at] != want[(size_t)m * N + (size_t)ni * Nt + n]) wrong++;
            }
    rocket_bo_fini(fd, res);
    printf("  %-10s %2d tasks, one kick, DATA_REUSE %s: %7ld of %7d wrong, %6ld unwritten, %.3f ms%s\n",
           a->name, nt, a->reuse ? "on " : "off", wrong, M * N, unwritten, ms, slow ? " [SLOW FENCE]" : "");
    if (wrong || slow) *bad = 1;
    return ms;
}

static int cmp_d(const void *x, const void *y)
{
    const double p = *(const double *)x, q = *(const double *)y;
    return p < q ? -1 : p > q;
}

int main(int argc, char **argv)
{
    int reps = 30, bad = 0;
    M = 256; K = 1536; N = 1024; Kt = 384; Nt = 256;
    if (argc >= 6) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); Kt = atoi(argv[4]); Nt = atoi(argv[5]); }
    if (argc >= 7) reps = atoi(argv[6]);
    if (M % 4 || M > 256 || K % Kt || N % Nt || Kt % 32 || Nt % 32 || K > 2048 || reps < 3) {
        fprintf(stderr, "usage: %s [M K N Kt Nt [reps]]: M%%4, M<=256, Kt|K, Nt|N, K<=2048\n", argv[0]);
        return 1;
    }
    nKt = K / Kt; nNt = N / Nt;
    if (nKt * nNt > MAX_TASKS) { fprintf(stderr, "too many tasks\n"); return 1; }
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }
    if (!rocket_batched_submit_supported()) { printf("chained submit not supported\n"); rocket_close(fd); return 3; }

    _Float16 *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    double *want = malloc((size_t)M * N * sizeof(double));
    tf_fill_f16_int(A, (size_t)M * K, 0x8100, -1, 1);
    tf_fill_f16_int(B, (size_t)N * K, 0x8200, -1, 1);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)A[(size_t)m * K + k] * (double)B[(size_t)n * K + k];
            want[(size_t)m * N + n] = s;
        }
    struct bufs b;
    memset(&b, 0, sizeof b);
    if (bufs_alloc(fd, &b)) { printf("allocation failed\n"); rocket_close(fd); return 1; }
    stage(fd, &b, A, B);
    printf("== row-major KACC chain probe: M %d K %d N %d, tiles %dx%dx%d, %d K slices x %d N tiles "
           "= %d tasks a kick; C row %d B ==\n", M, K, N, M, Kt, Nt, nKt, nNt, nKt * nNt, N * 2);
    for (int i = 0; i < NARMS; i++) arm_run(fd, &arms[i], &b, want, 1, &bad);

    if (!bad) {
        double *t[NARMS], *r[NARMS];
        for (int i = 0; i < NARMS; i++) { t[i] = calloc((size_t)reps, sizeof(double)); r[i] = calloc((size_t)reps, sizeof(double)); }
        for (int rep = 0; rep < reps && !bad; rep++)
            for (int j = 0; j < NARMS; j++) {
                const int i = (j + rep) % NARMS;
                t[i][rep] = arm_run(fd, &arms[i], &b, want, 0, &bad);
            }
        if (!bad) {
            for (int rep = 0; rep < reps; rep++)
                for (int i = 0; i < NARMS; i++) r[i][rep] = t[i][rep] / t[0][rep];
            printf("  timing, %d rotated reps, submit to fence of the whole kick, median ms (ratio to cube_r, "
                   "median of per-rep ratios, quartiles):\n", reps);
            for (int i = 0; i < NARMS; i++) {
                qsort(t[i], (size_t)reps, sizeof(double), cmp_d);
                qsort(r[i], (size_t)reps, sizeof(double), cmp_d);
                printf("    %-10s %.4f ms (%.4f-%.4f)  x%.3f (%.3f-%.3f)\n", arms[i].name, t[i][reps / 2],
                       t[i][0], t[i][reps - 1], r[i][reps / 2], r[i][reps / 4], r[i][(3 * reps) / 4]);
            }
        }
        for (int i = 0; i < NARMS; i++) { free(t[i]); free(r[i]); }
    }
    printf("== %s; slow-fence count %llu ==\n", bad ? "NOT every arm exact" : "every arm exact",
           (unsigned long long)rocket_fence_wait_slow_count());
    bufs_free(fd, &b);
    free(A); free(B); free(want);
    rocket_close(fd);
    return bad ? 1 : 0;
}
