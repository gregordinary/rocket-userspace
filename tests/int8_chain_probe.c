// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * int8_chain_probe.c — does a chained batch of int8 matmul tasks compute, on the RK3588?
 *
 * A probe, not a gate (GATE_AUDIT_PLAN GA-5.3). The record says a chained integer batch
 * computes its first task exactly and every later one wrong, because the int32
 * accumulator (CACC) clears per hardware kick rather than per task, so task t lands on
 * task t-1's residual. Two things about that record are unexamined. Its oracle was
 * rocket_matmul_int8(), a device path that itself submits multi-task jobs, on a kernel
 * where a global batching parameter was known to garble such jobs. And the mechanism
 * was named, never tested: no run compared a wrong task against its neighbour's answer.
 *
 * So this runs NT independent int8 x int8 -> int32 matmuls, each with its own input,
 * weights and output region, as ONE job three ways, and scores every task against a CPU
 * model:
 *
 *   gapped    one drm task descriptor each, the stock per-task path (the control);
 *   chained   the same programs self-chained (rocket_chain.c) and submitted with
 *             ROCKET_JOB_BATCHED, so the PC runs all of them from one kick;
 *   chained0  the same, with task 0's input all zero, so a residual from task 0 is zero
 *             and the CACC reading predicts task 1 comes back exact.
 *
 * For a task that is not exact it reports what fraction of its elements equal each
 * candidate the CACC reading allows: its own answer plus the previous task's
 * (ref[t] + ref[t-1]), plus the previous task's OUTPUT (ref[t] + out[t-1]), the running
 * sum of every earlier answer, and the previous answer alone (a stale surface).
 *
 * The output is stamped with 0xAA before every run, so an unwritten element is counted
 * as unwritten rather than scored against a leftover.
 *
 * WHETHER THE CHAIN ENGAGED is read, not assumed: each run prints the NPU interrupts it
 * raised (the /proc/interrupts rows naming an `.npu` device, which share their line with
 * that core's IOMMU) and its submit-to-fence time. A gapped job of NT tasks raises one
 * completion a task; a chained one raises one for the kick.
 *
 *   int8_chain_probe [NT [M K N [reps]]]      default 4 tasks of 64x256x64, 3 reps
 *
 * Needs a kernel that honours ROCKET_JOB_BATCHED (rocket_batched_submit_supported());
 * without it the chained arms are skipped. Exit 0 if the gapped control was exact on
 * every rep, 1 otherwise, 2 with no device.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_chain.h"
#include "npu_matmul.h"
#include "test_fill.h"

#define MAX_NT 16
#define GAPPED_STRIDE 256   /* words a gapped slot takes; a matmul program is ~110 */

struct probe {
    int NT, M, K, N;
    size_t in_slot, w_slot, out_slot;
    rocket_bo guard, in, w, out, rc;
    int8_t *A[MAX_NT], *B[MAX_NT];
    int32_t *ref[MAX_NT];          /* row-major [M][N] */
    uint64_t ops[MAX_NT][GAPPED_STRIDE];
    uint32_t count;
};

/* Every interrupt the NPU cores have raised, summed over CPUs and cores. */
static unsigned long long npu_irqs(void)
{
    FILE *f = fopen("/proc/interrupts", "r");
    char line[1024];
    unsigned long long total = 0;
    if (!f) return 0;
    while (fgets(line, sizeof line, f)) {
        char *p, *end;
        if (!strstr(line, ".npu")) continue;
        p = strchr(line, ':');
        if (!p) continue;
        p++;
        for (;;) {
            unsigned long long v = strtoull(p, &end, 10);
            if (end == p) break;
            total += v;
            p = end;
        }
    }
    fclose(f);
    return total;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void cpu_ref(const int8_t *A, const int8_t *B, int M, int K, int N, int32_t *C)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)A[(size_t)m * K + k] * B[(size_t)n * K + k];
            C[(size_t)m * N + n] = (int32_t)s;
        }
}

/* Pack task t's input (optionally zeroed), weights and program. */
static int pack_task(int fd, struct probe *p, int t, int zero_input)
{
    int8_t *in = (int8_t *)p->in.ptr + (size_t)t * p->in_slot;
    int8_t *w = (int8_t *)p->w.ptr + (size_t)t * p->w_slot;
    int M = p->M, K = p->K, N = p->N;

    (void)fd;
    memset(in, 0, p->in_slot);
    memset(w, 0, p->w_slot);
    for (int n = 1; n <= N; n++)
        for (int k = 1; k <= K; k++) w[weight_int8(K, n, k)] = p->B[t][(size_t)(n - 1) * K + (k - 1)];
    if (!zero_input)
        for (int m = 1; m <= M; m++)
            for (int k = 1; k <= K; k++)
                in[feature_data(K, M, 1, 16, k, m, 1)] = p->A[t][(size_t)(m - 1) * K + (k - 1)];
    {
        matmul_params_t mp = {
            .m = (uint16_t)M, .k = (uint16_t)K, .n = (uint16_t)N,
            .input_dma = (uint32_t)(p->in.dma_address + (size_t)t * p->in_slot),
            .weights_dma = (uint32_t)(p->w.dma_address + (size_t)t * p->w_slot),
            .output_dma = (uint32_t)(p->out.dma_address + (size_t)t * p->out_slot),
            .tasks = p->ops[t],
        };
        if (gen_matmul_int8(&mp) != 0 || mp.task_count > GAPPED_STRIDE) return -1;
        p->count = mp.task_count;
    }
    return 0;
}

/* Read task t's output into row-major int32. */
static void read_task(const struct probe *p, int t, int32_t *o, size_t *unwritten)
{
    const int32_t *od = (const int32_t *)((const uint8_t *)p->out.ptr + (size_t)t * p->out_slot);
    *unwritten = 0;
    for (int m = 1; m <= p->M; m++)
        for (int n = 1; n <= p->N; n++) {
            int32_t v = od[feature_data(p->N, p->M, 1, 4, n, m, 1)];
            if ((uint32_t)v == 0xAAAAAAAAu) (*unwritten)++;
            o[(size_t)(m - 1) * p->N + (n - 1)] = v;
        }
}

static int run_arm(int fd, struct probe *p, const char *arm, int chained, int zero0, int rep)
{
    rocket_task_desc td[MAX_NT];
    uint32_t in_h[3], out_h[1];
    size_t ne = (size_t)p->M * p->N;
    int32_t *o[MAX_NT], *want0 = NULL;
    int t, exact_all = 1, rc;

    /* The reference for task 0 under chained0 is all zero. */
    if (zero0) {
        want0 = calloc(ne, sizeof(int32_t));
        if (!want0) return -1;
    }
    rocket_bo_prep(fd, &p->in, 1, 0);
    rocket_bo_prep(fd, &p->w, 1, 0);
    for (t = 0; t < p->NT; t++)
        if (pack_task(fd, p, t, zero0 && t == 0) != 0) {
            printf("  %s: the generator refused %dx%dx%d\n", arm, p->M, p->K, p->N);
            rocket_bo_fini(fd, &p->in); rocket_bo_fini(fd, &p->w);
            free(want0);
            return -1;
        }
    rocket_bo_fini(fd, &p->in);
    rocket_bo_fini(fd, &p->w);

    rocket_bo_prep(fd, &p->rc, 1, 0);
    memset(p->rc.ptr, 0, p->rc.size);
    for (t = 0; t < p->NT; t++)
        if (rkt_chain_pack(chained, &p->rc, td, t, p->ops[t], p->count, GAPPED_STRIDE) != 0) {
            printf("  %s: the chain pack refused task %d\n", arm, t);
            rocket_bo_fini(fd, &p->rc);
            free(want0);
            return -1;
        }
    rkt_chain_seal(chained, &p->rc, p->NT, p->count);
    if (chained && rkt_chain_verify(&p->rc, td, p->NT) != 0) {
        printf("  %s: the chained layout does not verify\n", arm);
        rocket_bo_fini(fd, &p->rc);
        free(want0);
        return -1;
    }
    rocket_bo_fini(fd, &p->rc);

    rocket_bo_prep(fd, &p->out, 1, 0);
    memset(p->out.ptr, 0xAA, p->out.size);
    rocket_bo_fini(fd, &p->out);

    in_h[0] = p->in.handle; in_h[1] = p->w.handle; in_h[2] = p->rc.handle;
    out_h[0] = p->out.handle;
    {
        unsigned long long irq0 = npu_irqs();
        double t0 = now_ms();
        rc = rocket_submit_tasks_flags(fd, td, (uint32_t)p->NT, in_h, 3, out_h, 1,
                                       chained ? ROCKET_JOB_BATCHED : 0);
        if (rc == 0) rc = rocket_bo_prep(fd, &p->out, 0, 2000000000ull);
        t0 = now_ms() - t0;
        if (rc != 0) {
            printf("  %s rep %d: submit or wait failed (%d)\n", arm, rep, rc);
            free(want0);
            return -1;
        }
        printf("  %-8s rep %d: %llu NPU interrupts, %.3f ms submit to fence\n", arm, rep,
               npu_irqs() - irq0, t0);
    }
    for (t = 0; t < p->NT; t++) o[t] = malloc(ne * sizeof(int32_t));
    for (t = 0; t < p->NT; t++) {
        const int32_t *want = (zero0 && t == 0) ? want0 : p->ref[t];
        size_t unw, wrong = 0, e;
        read_task(p, t, o[t], &unw);
        for (e = 0; e < ne; e++) wrong += o[t][e] != want[e];
        printf("  %-8s rep %d task %d: %5zu of %zu wrong, %zu unwritten", arm, rep, t, wrong,
               ne, unw);
        if (wrong && t > 0) {
            const int32_t *prev = (zero0 && t == 1) ? want0 : p->ref[t - 1];
            size_t h_sum = 0, h_sumout = 0, h_cum = 0, h_prev = 0;
            for (e = 0; e < ne; e++) {
                int64_t cum = 0;
                for (int j = 0; j <= t; j++)
                    cum += (zero0 && j == 0) ? 0 : p->ref[j][e];
                h_sum += (int64_t)o[t][e] == (int64_t)want[e] + prev[e];
                h_sumout += (int64_t)o[t][e] == (int64_t)want[e] + o[t - 1][e];
                h_cum += (int64_t)o[t][e] == cum;
                h_prev += o[t][e] == prev[e];
            }
            printf("  = ref+prev %.1f%%, ref+out[t-1] %.1f%%, running sum %.1f%%, "
                   "prev alone %.1f%%", 100.0 * h_sum / ne, 100.0 * h_sumout / ne,
                   100.0 * h_cum / ne, 100.0 * h_prev / ne);
        }
        if (wrong) {
            for (e = 0; e < ne && o[t][e] == want[e]; e++) ;
            printf("  first m=%zu n=%zu got %d want %d", e / p->N, e % p->N, o[t][e], want[e]);
            exact_all = 0;
        }
        printf("\n");
    }
    rocket_bo_fini(fd, &p->out);
    for (t = 0; t < p->NT; t++) free(o[t]);
    free(want0);
    return exact_all ? 0 : 1;
}

int main(int argc, char **argv)
{
    struct probe p = { .NT = 4, .M = 64, .K = 256, .N = 64 };
    int reps = 3, fd, t, r, gapped_bad = 0, chain_ok;

    if (argc > 1) p.NT = atoi(argv[1]);
    if (argc > 4) { p.M = atoi(argv[2]); p.K = atoi(argv[3]); p.N = atoi(argv[4]); }
    if (argc > 5) reps = atoi(argv[5]);
    if (p.NT < 2 || p.NT > MAX_NT || p.M < 4 || p.M % 4 || p.K % 32 || p.N % 32) {
        fprintf(stderr, "usage: %s [NT (2-%d) [M K N [reps]]], M%%4, K%%32, N%%32\n",
                argv[0], MAX_NT);
        return 1;
    }
    fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }
    chain_ok = rocket_batched_submit_supported();
    printf("== int8 chain probe: %d tasks of %dx%dx%d, %d reps, batched submit %s ==\n",
           p.NT, p.M, p.K, p.N, reps, chain_ok ? "supported" : "NOT supported");

    p.in_slot = ((size_t)p.M * p.K + 4095) & ~(size_t)4095;
    p.w_slot = ((size_t)p.N * p.K + 4095) & ~(size_t)4095;
    p.out_slot = ((size_t)p.M * p.N * 4 + 4095) & ~(size_t)4095;
    if (rocket_bo_alloc32(fd, 4096, &p.guard) ||
        rocket_bo_alloc32(fd, p.in_slot * p.NT, &p.in) ||
        rocket_bo_alloc32(fd, p.w_slot * p.NT, &p.w) ||
        rocket_bo_alloc32(fd, p.out_slot * p.NT, &p.out) ||
        rocket_bo_alloc32(fd, (size_t)GAPPED_STRIDE * 8 * p.NT, &p.rc)) {
        printf("BO allocation failed\n");
        return 1;
    }
    for (t = 0; t < p.NT; t++) {
        size_t na = (size_t)p.M * p.K, nb = (size_t)p.N * p.K;
        p.A[t] = malloc(na); p.B[t] = malloc(nb);
        p.ref[t] = malloc((size_t)p.M * p.N * sizeof(int32_t));
        tf_fill_i8(p.A[t], na, 0xA0 + (uint64_t)t, -128, 127);
        tf_fill_i8(p.B[t], nb, 0xB0 + (uint64_t)t, -128, 127);
        cpu_ref(p.A[t], p.B[t], p.M, p.K, p.N, p.ref[t]);
    }
    for (r = 0; r < reps; r++) {
        gapped_bad += run_arm(fd, &p, "gapped", 0, 0, r) != 0;
        if (chain_ok) {
            run_arm(fd, &p, "chained", 1, 0, r);
            run_arm(fd, &p, "chained0", 1, 1, r);
        }
    }
    printf("== gapped control %s ==\n", gapped_bad ? "NOT exact: nothing above is scoreable"
                                                    : "exact on every rep");
    rocket_close(fd);
    return gapped_bad ? 1 : 0;
}
