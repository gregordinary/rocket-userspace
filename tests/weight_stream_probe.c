// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * weight_stream_probe.c — does one RK3588 fp16 matmul task compute when its weight cube is
 * larger than the CBUF weight banks, and what does a wide task save over narrow ones?
 *
 * A probe, not a gate. The library's planner (`mm_fits`, rocket_matmul.c) requires the whole
 * Nt x Kt weight tile to fit in the CBUF beside the input tile, and caps Nt at 256. The
 * generator does not: `gen_matmul_fp16` gives every bank the input does not take to the
 * weights and refuses only a kernel wider than one bank. With the input tile resident, the
 * CNA can then stream the kernel groups through the weight banks, so one task covers all
 * of N. An external userspace (rknpu2_matmul_open, vendor driver) runs fp16 tasks this way
 * to N 4096, sizing the weight banks for one 32-kernel atom; this checks the same program
 * shape under `rocket`, from the register facts alone.
 *
 * Each shape runs three arms, each as ONE job on one core, chained when the kernel supports
 * ROCKET_JOB_BATCHED:
 *
 *   tile        today's rule: (Mt_tile, Nt_tile) chosen so input and weight tiles both fit,
 *               every task fetching its operands from DRAM;
 *   tile+reuse  the same tasks with DATA_REUSE set on every task after the first of an
 *               M-tile, the library's default, so the input is fetched once per M-tile and
 *               the DRAM bytes equal the streaming arm's;
 *   stream      one task per M-tile (Mt_stream), N whole, weight cube far past the banks.
 *
 * Operands are integers in [-3, 3], so every fp32 sum is exact and each element is compared
 * for equality with a double reference. The output is stamped 0xFF (a NaN) before every run,
 * so an unwritten element cannot pass. After every streaming job a small canary matmul
 * (64x256x64) runs as its own job and is scored too: a task past a field's width on this
 * silicon can write a plausible surface AND corrupt the next legal job.
 *
 * Timing: the arms run interleaved with the order rotated, and the median submit-to-fence
 * time of each is printed with its GFLOP/s. That is device time plus one submit's fixed cost
 * a job, not a price for the host work a tiling implies.
 *
 * What a green result does NOT show: int8 or int4, a K split across tasks, multicore, the
 * library's planner, or any model-level effect.
 *
 * Usage: weight_stream_probe [reps]          the built-in ladder, `reps` timed runs (12)
 *        weight_stream_probe M K N Mt_stream Mt_tile Nt_tile [reps]
 * Env:   WSP_GAPPED=1 forces the per-task (unchained) path.
 * Exit:  0 every arm and canary exact, 1 otherwise, 2 no NPU.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_chain.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "test_fill.h"

#define SLOT_WORDS 128   /* gapped slot per task; a matmul program is ~110 words */

struct shape { int M, K, N, mt_s, mt_t, nt_t; };

/* Each row: the streaming M-tile keeps input + one 32-kernel weight atom within 12 banks;
 * the tile arm's (Mt, Nt) keeps input + whole weight tile within 12 banks. */
static const struct shape ladder[] = {
    { 128,  512, 4096, 128, 128, 256 },
    { 512,  512, 2048, 128, 128, 256 },
    { 256, 1024, 4096, 128,  64, 128 },
    { 160, 1024, 4096, 160,  64, 128 },
    {  64, 1024, 1040,  64,  64, 128 },
    { 128, 2560, 4096,  32,  32,  32 },
};

enum { ARM_TILE, ARM_REUSE, ARM_STREAM, NARMS };
static const char *arm_name[NARMS] = { "tile", "tile+reuse", "stream" };

struct arm {
    int mt, nt, nmt, nnt, ntasks, count;
    rocket_bo in, w, out, rc;
    rocket_task_desc *td;
    size_t *out_off;         /* byte offset of task t's output cube */
};

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static int banks(long bytes) { return (int)((bytes + NPU_CBUF_BANK_SIZE - 1) / NPU_CBUF_BANK_SIZE); }

static int min_i(int a, int b) { return a < b ? a : b; }

/* Rewrite one register's value in a generated program. Returns 0, or -1 when the program
 * never writes that register. */
static int patch(uint64_t *ops, int n, uint16_t target, uint16_t reg, uint32_t value)
{
    for (int i = 0; i < n; i++) {
        if ((uint16_t)(ops[i] >> 48) != target || (uint16_t)(ops[i] & 0xFFFF) != reg) continue;
        ops[i] = NPUOP(target, value, reg);
        return 0;
    }
    return -1;
}

/* Streaming-arm variants, applied to the generated program:
 *   WSP_WBANKS=atom  weight banks = one 32-kernel atom, the input gets every other bank
 *   WSP_WBANKS=n     weight banks = n, the input the rest
 *   WSP_GRAINS=n     CONV_CON2 feature grains = n (the generator writes M+1) */
static int stream_patch(uint64_t *ops, int n, int m, int K)
{
    const char *wb = getenv("WSP_WBANKS"), *gr = getenv("WSP_GRAINS");
    if (wb) {
        int w = !strcmp(wb, "atom") ? banks((long)K * 2 * 32) : atoi(wb);
        int d = NPU_CBUF_BANKS - w;
        if (w < 1 || d < banks((long)m * K * 2)) return -1;
        if (patch(ops, n, OP_REG_CNA, CNA_CBUF_CON0, ((uint32_t)w << 4) | (uint32_t)d)) return -1;
    }
    if (gr && patch(ops, n, OP_REG_CNA, CNA_CONV_CON2, ((uint32_t)atoi(gr) & 0x3FF) << 4)) return -1;
    return 0;
}

/* Lay out one arm: pack its inputs per M-tile and weights per N-tile, generate and pack its
 * programs. Returns 0, or -1 with a message. */
static int arm_build(int fd, struct arm *a, int which, const struct shape *s,
                     const _Float16 *A, const _Float16 *B, int chained)
{
    const int M = s->M, K = s->K, N = s->N;
    a->mt = which == ARM_STREAM ? s->mt_s : s->mt_t;
    a->nt = which == ARM_STREAM ? N : s->nt_t;
    a->nmt = (M + a->mt - 1) / a->mt;
    a->nnt = (N + a->nt - 1) / a->nt;
    a->ntasks = a->nmt * a->nnt;
    a->td = calloc((size_t)a->ntasks, sizeof(*a->td));
    a->out_off = calloc((size_t)a->ntasks, sizeof(size_t));
    if (!a->td || !a->out_off) return -1;

    const size_t in_slot = (size_t)a->mt * K * 2, w_slot = (size_t)a->nt * K * 2;
    size_t out_bytes = 0;
    for (int mi = 0; mi < a->nmt; mi++)
        for (int ni = 0; ni < a->nnt; ni++) {
            int mt = min_i(a->mt, M - mi * a->mt), nt = min_i(a->nt, N - ni * a->nt);
            a->out_off[mi * a->nnt + ni] = out_bytes;
            out_bytes += ((size_t)mt * nt * 4 + 63) & ~(size_t)63;
        }
    if (rocket_bo_alloc32(fd, in_slot * a->nmt, &a->in) ||
        rocket_bo_alloc32(fd, w_slot * a->nnt, &a->w) ||
        rocket_bo_alloc32(fd, out_bytes, &a->out) ||
        rocket_bo_alloc32(fd, (size_t)a->ntasks * SLOT_WORDS * 8, &a->rc)) {
        printf("  %s: allocation failed\n", arm_name[which]);
        return -1;
    }

    rocket_bo_prep(fd, &a->in, 1, 0);
    rocket_bo_prep(fd, &a->w, 1, 0);
    memset(a->in.ptr, 0, a->in.size);
    memset(a->w.ptr, 0, a->w.size);
    for (int mi = 0; mi < a->nmt; mi++) {
        _Float16 *id = (_Float16 *)((uint8_t *)a->in.ptr + mi * in_slot);
        int mt = min_i(a->mt, M - mi * a->mt);
        for (int m = 1; m <= mt; m++)
            for (int k = 1; k <= K; k++)
                id[feature_data(K, mt, 1, 8, k, m, 1)] = A[(size_t)(mi * a->mt + m - 1) * K + (k - 1)];
    }
    for (int ni = 0; ni < a->nnt; ni++) {
        _Float16 *wd = (_Float16 *)((uint8_t *)a->w.ptr + ni * w_slot);
        int nt = min_i(a->nt, N - ni * a->nt);
        for (int n = 1; n <= nt; n++)
            for (int k = 1; k <= K; k++)
                wd[weight_fp16(K, n, k)] = B[(size_t)(ni * a->nt + n - 1) * K + (k - 1)];
    }
    rocket_bo_fini(fd, &a->in);
    rocket_bo_fini(fd, &a->w);

    rocket_bo_prep(fd, &a->rc, 1, 0);
    memset(a->rc.ptr, 0, a->rc.size);
    for (int mi = 0; mi < a->nmt; mi++)
        for (int ni = 0; ni < a->nnt; ni++) {
            int t = mi * a->nnt + ni;
            uint64_t ops[SLOT_WORDS] = { 0 };
            matmul_params_t p = {
                .m = (uint16_t)min_i(a->mt, M - mi * a->mt), .k = (uint16_t)K,
                .n = (uint16_t)min_i(a->nt, N - ni * a->nt),
                .input_dma = (uint32_t)(a->in.dma_address + mi * in_slot),
                .weights_dma = (uint32_t)(a->w.dma_address + ni * w_slot),
                .output_dma = (uint32_t)(a->out.dma_address + a->out_off[t]),
                .tasks = ops, .fp32tofp16 = 0,
                .data_reuse = (which == ARM_REUSE && ni > 0) ? 1 : 0,
            };
            int rc = gen_matmul_fp16(&p);
            if (rc != 0) {
                printf("  %s: the generator refused task %d (%dx%dx%d, rc %d)\n", arm_name[which],
                       t, p.m, K, p.n, rc);
                rocket_bo_fini(fd, &a->rc);
                return -1;
            }
            if (which == ARM_STREAM && stream_patch(ops, (int)p.task_count, p.m, K) != 0) {
                printf("  %s: a register to patch is missing from the program\n", arm_name[which]);
                rocket_bo_fini(fd, &a->rc);
                return -1;
            }
            if (t == 0) a->count = (int)p.task_count;
            if ((int)p.task_count != a->count ||
                rkt_chain_pack(chained, &a->rc, a->td, t, ops, p.task_count, SLOT_WORDS) != 0) {
                printf("  %s: the chain pack refused task %d\n", arm_name[which], t);
                rocket_bo_fini(fd, &a->rc);
                return -1;
            }
        }
    rkt_chain_seal(chained, &a->rc, a->ntasks, (uint32_t)a->count);
    if (chained && rkt_chain_verify(&a->rc, a->td, a->ntasks) != 0) {
        printf("  %s: the chained layout does not verify\n", arm_name[which]);
        rocket_bo_fini(fd, &a->rc);
        return -1;
    }
    rocket_bo_fini(fd, &a->rc);
    return 0;
}

static void arm_free(int fd, struct arm *a)
{
    rocket_bo_free(fd, &a->in); rocket_bo_free(fd, &a->w);
    rocket_bo_free(fd, &a->out); rocket_bo_free(fd, &a->rc);
    free(a->td); free(a->out_off);
}

/* Run one arm once; score it when `want` is given. Returns 0 exact, 1 wrong, -1 failed. */
static int arm_run(int fd, struct arm *a, const struct shape *s, int chained, const double *want,
                   double *ms, const char *label)
{
    const int M = s->M, N = s->N;
    rocket_bo_prep(fd, &a->out, 1, 0);
    memset(a->out.ptr, 0xFF, a->out.size);
    rocket_bo_fini(fd, &a->out);

    uint32_t in_h[3] = { a->in.handle, a->w.handle, a->rc.handle }, out_h[1] = { a->out.handle };
    uint64_t slow0 = rocket_fence_wait_slow_count();
    double t0 = now_ms();
    int rc = rocket_submit_tasks_flags(fd, a->td, (uint32_t)a->ntasks, in_h, 3, out_h, 1,
                                       chained ? ROCKET_JOB_BATCHED : 0);
    if (rc == 0) rc = rocket_bo_prep(fd, &a->out, 0, 4000000000ull);
    *ms = now_ms() - t0;
    int slow = rocket_fence_wait_slow_count() != slow0;
    if (rc != 0) { printf("  %s: submit or wait failed (%d)\n", label, rc); return -1; }
    if (!want) { rocket_bo_fini(fd, &a->out); return slow ? 1 : 0; }

    long wrong = 0, unwritten = 0, first = -1;
    for (int mi = 0; mi < a->nmt; mi++)
        for (int ni = 0; ni < a->nnt; ni++) {
            const float *o = (const float *)((uint8_t *)a->out.ptr + a->out_off[mi * a->nnt + ni]);
            int mt = min_i(a->mt, M - mi * a->mt), nt = min_i(a->nt, N - ni * a->nt);
            for (int m = 0; m < mt; m++)
                for (int n = 0; n < nt; n++) {
                    double got = o[feature_data(nt, mt, 1, 4, n + 1, m + 1, 1)];
                    size_t at = (size_t)(mi * a->mt + m) * N + (ni * a->nt + n);
                    if (isnan(got)) unwritten++;
                    if (got != want[at]) { if (first < 0) first = (long)at; wrong++; }
                }
        }
    rocket_bo_fini(fd, &a->out);
    printf("  %-10s %4d task(s) of %4dx%4d: %7ld of %8d wrong, %7ld unwritten, %.3f ms%s",
           label, a->ntasks, a->mt, a->nt, wrong, M * N, unwritten, *ms, slow ? " [SLOW FENCE]" : "");
    if (wrong) printf(", first (m %ld, n %ld)", first / N, first % N);
    printf("\n");
    return (wrong || slow) ? 1 : 0;
}

/* The canary: one fresh 64x256x64 task in its own job, scored. 0 exact, 1 not. */
static int canary(int fd)
{
    static const struct shape c = { 64, 256, 64, 64, 64, 64 };
    struct arm a = { 0 };
    _Float16 *A = malloc(64 * 256 * 2), *B = malloc(64 * 256 * 2);
    double *want = malloc(64 * 64 * sizeof(double)), ms;
    int bad = 1;
    if (!A || !B || !want) goto out;
    tf_fill_f16_int(A, 64 * 256, 0xCA11, -3, 3);
    tf_fill_f16_int(B, 64 * 256, 0xCA12, -3, 3);
    for (int m = 0; m < 64; m++)
        for (int n = 0; n < 64; n++) {
            double acc = 0;
            for (int k = 0; k < 256; k++) acc += (double)A[m * 256 + k] * (double)B[n * 256 + k];
            want[m * 64 + n] = acc;
        }
    if (arm_build(fd, &a, ARM_TILE, &c, A, B, 0) == 0)
        bad = arm_run(fd, &a, &c, 0, want, &ms, "canary") != 0;
    arm_free(fd, &a);
out:
    free(A); free(B); free(want);
    return bad;
}

static int run_shape(int fd, const struct shape *s, int chained, int reps)
{
    const int M = s->M, K = s->K, N = s->N;
    int bad = 0;
    printf("-- %dx%dx%d: stream Mt %d (input %d + atom %d of %d banks, weight cube %ld KiB); "
           "tile %dx%d (input %d + weights %d banks)\n", M, K, N, s->mt_s,
           banks((long)s->mt_s * K * 2), banks((long)K * 2 * 32), NPU_CBUF_BANKS,
           (long)N * K * 2 / 1024, s->mt_t, s->nt_t, banks((long)s->mt_t * K * 2),
           banks((long)s->nt_t * K * 2));
    if (banks((long)s->mt_s * K * 2) + banks((long)K * 2 * 32) > NPU_CBUF_BANKS ||
        banks((long)s->mt_t * K * 2) + banks((long)s->nt_t * K * 2) > NPU_CBUF_BANKS) {
        printf("  shape outside the bank budget, skipped\n");
        return 1;
    }

    _Float16 *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    double *want = malloc((size_t)M * N * sizeof(double));
    struct arm arms[NARMS] = { { 0 } };
    if (!A || !B || !want) { bad = 1; goto out; }
    tf_fill_f16_int(A, (size_t)M * K, 0x5100 + (uint64_t)M * 7 + K, -3, 3);
    tf_fill_f16_int(B, (size_t)N * K, 0x5200 + (uint64_t)N * 7 + K, -3, 3);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0;
            const _Float16 *a = A + (size_t)m * K, *b = B + (size_t)n * K;
            for (int k = 0; k < K; k++) acc += (double)a[k] * (double)b[k];
            want[(size_t)m * N + n] = acc;
        }
    for (int i = 0; i < NARMS; i++)
        if (arm_build(fd, &arms[i], i, s, A, B, chained) != 0) { bad = 1; goto out; }

    /* Correctness: every arm once, scored; a canary after the streaming arm. */
    for (int i = 0; i < NARMS; i++) {
        double ms;
        int r = arm_run(fd, &arms[i], s, chained, want, &ms, arm_name[i]);
        if (r != 0) bad = 1;
        if (i == ARM_STREAM && canary(fd)) { printf("  canary after stream: NOT exact\n"); bad = 1; }
    }
    if (bad || reps <= 0) goto out;

    /* Timing: interleaved, order rotated, every run scored silently for a slow fence; the
     * stream arm's runs re-scored in full at the end of the pass. */
    {
        double *t[NARMS];
        for (int i = 0; i < NARMS; i++) t[i] = calloc((size_t)reps, sizeof(double));
        for (int r = 0; r < reps; r++)
            for (int j = 0; j < NARMS; j++) {
                int i = (j + r) % NARMS;
                if (arm_run(fd, &arms[i], s, chained, NULL, &t[i][r], arm_name[i]) != 0) {
                    printf("  %s: timed run %d failed or fenced slow\n", arm_name[i], r);
                    bad = 1;
                }
            }
        double med[NARMS], flop = 2.0 * M * K * N;
        for (int i = 0; i < NARMS; i++) {
            qsort(t[i], (size_t)reps, sizeof(double), cmp_d);
            med[i] = t[i][reps / 2];
        }
        printf("  timing, %d interleaved runs, median submit to fence:", reps);
        for (int i = 0; i < NARMS; i++)
            printf(" %s %.3f ms (%.0f GFLOP/s, %.3f-%.3f)", arm_name[i], med[i], flop / med[i] / 1e6,
                   t[i][0], t[i][reps - 1]);
        printf("\n  ratios: tile/stream %.3fx, tile+reuse/stream %.3fx, tile/tile+reuse %.3fx\n",
               med[ARM_TILE] / med[ARM_STREAM], med[ARM_REUSE] / med[ARM_STREAM],
               med[ARM_TILE] / med[ARM_REUSE]);
        for (int i = 0; i < NARMS; i++) free(t[i]);
        double ms;
        if (arm_run(fd, &arms[ARM_STREAM], s, chained, want, &ms, "stream/end") != 0) bad = 1;
        if (canary(fd)) { printf("  canary at end: NOT exact\n"); bad = 1; }
    }
out:
    for (int i = 0; i < NARMS; i++) arm_free(fd, &arms[i]);
    free(A); free(B); free(want);
    return bad;
}

int main(int argc, char **argv)
{
    struct shape one;
    const struct shape *shapes = ladder;
    int nshapes = (int)(sizeof(ladder) / sizeof(ladder[0])), reps = 12, bad = 0;

    if (argc >= 7) {
        one = (struct shape){ atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
                              atoi(argv[4]), atoi(argv[5]), atoi(argv[6]) };
        if (argc >= 8) reps = atoi(argv[7]);
        if (one.M < 4 || one.M % 4 || one.K % 32 || one.K < 64 || one.N % 16 || one.N < 16 ||
            one.mt_s % 4 || one.mt_t % 4 || one.nt_t % 16) {
            fprintf(stderr, "usage: %s M K N Mt_stream Mt_tile Nt_tile [reps]\n", argv[0]);
            return 1;
        }
        shapes = &one; nshapes = 1;
    } else if (argc == 2) {
        reps = atoi(argv[1]);
    }
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }
    const char *g = getenv("WSP_GAPPED");
    int chained = !(g && atoi(g)) && rocket_batched_submit_supported() && rkt_chain_enabled();
    printf("== weight-stream probe: %d shape(s), fp16 in, fp32 out, one job an arm, %s ==\n",
           nshapes, chained ? "chained (one kick)" : "gapped (one kick a task)");
    for (int i = 0; i < nshapes; i++) bad |= run_shape(fd, &shapes[i], chained, reps);
    printf("== %s ==\n", bad ? "NOT exact everywhere" : "every arm and canary exact");
    rocket_close(fd);
    return bad ? 1 : 0;
}
