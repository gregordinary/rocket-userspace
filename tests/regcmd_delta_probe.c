// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * regcmd_delta_probe.c — can a task in a job carry only the registers that changed, on the
 * RK3588?
 *
 * A probe, not a gate (GATE_AUDIT_PLAN GA-5.5). regcmd_persist_rocket recorded "delta regcmd
 * is not usable": a 2-task job whose second task wrote only its addresses and the enable left
 * that output untouched, and passed only right after a full job. Two things about that probe
 * decide what it could see:
 *
 *   - Its delta wrote DPU_S_POINTER alone. The full program writes DPU_S_POINTER and
 *     DPU_RDMA_S_POINTER as 0xE (ping-pong on, toggling by pointer) and never writes the CNA
 *     or CORE pointers, which reset to 0: one register group. So under ping-pong the delta's
 *     DPU writes land in the OTHER group from the full task before it, which that job never
 *     programmed. A previous 2-task full job does program it, which is the "passes only after
 *     a full job" the record describes.
 *   - Its S_POINTER=0 arm patched the DPU pointer alone and left the DPU_RDMA one at 0xE, and
 *     cleared neither pointer, so no arm put the delta in a group its own job had written.
 *
 * The S_POINTER fields [TRM, RK3588 Part 1, RKNN_cna_s_pointer and its siblings]: bit 0 is
 * the group the registers are written into, bit 1 enables ping-pong of that pointer, bit 2
 * of the executer's, bit 3 toggles the pointer by pointer (1) or by executer (0), and bits 4
 * and 5 are write-1-to-clear for the write and execute pointers. Every field resets to 0.
 * The KERNEL writes the CNA and CORE pointers before every task, as 0xE plus
 * 0x10000000 * core index, a bit the TRM marks reserved [source-confirmed, rocket_job.c
 * rocket_job_hw_submit()]. So as shipped all four blocks ping-pong.
 *
 * So each arm here is ONE job of gapped tasks (one kick a task), every task with its own
 * input, weights and output, scored against a CPU model, under one of three pointer modes:
 *
 *   ship   the generator's program as it ships: DPU and DPU_RDMA 0xE from the regcmd, CNA
 *          and CORE 0xE from the kernel
 *   zero   all four pointers written 0x30 in every task: both pointers cleared to group 0,
 *          ping-pong off, so every task writes and executes group 0. The CNA and CORE
 *          writes overwrite the kernel's, per-core bit included.
 *   zeroB  zero with 0x10000000 in the CNA and CORE writes: core 1's bit
 *
 * and a task is one of:
 *
 *   F  the full program, M x K x N
 *   S  a SCRUB: the full program at M x K x N/2, so a task that inherits its configuration
 *      writes only the first N/2 columns and leaves the rest unwritten. That signature is
 *      what says which task a delta took its configuration from.
 *   D  a DELTA: its pointer writes, the registers whose values differ between two full
 *      programs that differ only in their buffers (found by diffing them, and printed), and
 *      the full program's own 4-word trailer. Nothing else.
 *
 * The arms, each with what it asks:
 *
 *   ship  FFFF    control
 *   ship  FSD     THE QUESTION: the delta lands two tasks back, in F's group
 *   ship  SFD     must carry the scrub's signature, or the arm above cannot be read
 *   ship  FFDDDD  two full tasks, then deltas: each lands in the group its task-2 wrote
 *   ship  SSFD    the record, with the groups scrubbed: the delta lands in the scrub's group
 *   zero  FF, FD and zeroB FF, FD   whether a single group lets a delta follow the task
 *                 before it, and whether the kernel's per-core bit decides if the mode runs
 *
 * A zero or zeroB job that lands on the core whose bit it does not carry never completes:
 * no interrupt, nothing written, retired by the kernel's 500 ms timeout, so a `zero` hang
 * scores the bit and says nothing about the delta [HW sweep, RK1, 2026-09-25]. Read those
 * arms only on the jobs that raised interrupts.
 *
 * The arm count is odd so that, with jobs alternating between cores, an arm lands on a
 * different core from one rep to the next. Each job prints the NPU interrupts it raised on
 * each core, one a task when the tasks ran as separate kicks, so a job that completes says
 * where it ran; a job that raises none is attributed by the kernel log's timeout line.
 * Every output is stamped 0xAA first, so an unwritten element is counted as unwritten.
 *
 * Then, if the ship FFDDDD arm was exact on every rep, `timing` reps of ship FFFF against
 * ship FFDD, interleaved, report the submit-to-fence median of each: what two deltas save.
 *
 *   regcmd_delta_probe [M K N [reps [timing]]]      default 64 256 64, 4 reps, 40 timing
 *
 * Exit 0 if every control was exact on every rep, 1 otherwise, 2 with no device.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_hw.h"
#include "npu_matmul.h"
#include "test_fill.h"

#define NT_MAX 6
#define STRIDE 256               /* words a program slot holds; a full program is ~110 */
#define OP_REG_DPU_RDMA_ (BLOCK_DPU_RDMA | PC_OP_01)
#define DPU_RDMA_S_POINTER_ 0x5004

enum { P_SHIP, P_ZERO, P_ZEROB };
static const char *pmode_name[] = { "ship", "zero", "zeroB" };

struct probe {
    int M, K, N;
    size_t in_slot, w_slot, out_slot;
    rocket_bo guard, in, w, out, rc;
    int8_t *A[NT_MAX], *B[NT_MAX];
    int32_t *ref[NT_MAX];        /* row-major [M][N], slot t's A x B */
    uint16_t dreg[32];           /* the registers a delta rewrites, and their op words */
    uint16_t dop[32];
    int ndreg;
};

/* The interrupts each NPU core has raised, in /proc/interrupts order (core 0 first on the
 * RK3588 device tree), summed over CPUs. Returns the number of cores found. */
static int npu_irqs(unsigned long long *per, int max)
{
    FILE *f = fopen("/proc/interrupts", "r");
    char line[1024];
    int n = 0;
    if (!f) return 0;
    while (n < max && fgets(line, sizeof line, f)) {
        char *p, *end;
        if (!strstr(line, ".npu")) continue;
        p = strchr(line, ':');
        if (!p) continue;
        p++;
        per[n] = 0;
        for (;;) {
            unsigned long long v = strtoull(p, &end, 10);
            if (end == p) break;
            per[n] += v;
            p = end;
        }
        n++;
    }
    fclose(f);
    return n;
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

/* The full program for slot t at n output columns, into ops. Returns its length or -1. */
static int gen_full(const struct probe *p, int t, int n, uint64_t *ops)
{
    matmul_params_t mp = {
        .m = (uint16_t)p->M, .k = (uint16_t)p->K, .n = (uint16_t)n,
        .input_dma = (uint32_t)(p->in.dma_address + (size_t)t * p->in_slot),
        .weights_dma = (uint32_t)(p->w.dma_address + (size_t)t * p->w_slot),
        .output_dma = (uint32_t)(p->out.dma_address + (size_t)t * p->out_slot),
        .tasks = ops,
    };
    if (gen_matmul_int8(&mp) != 0 || mp.task_count + 2 > STRIDE) return -1;
    return (int)mp.task_count;
}

/* The pointer writes a task opens with under mode `pm`. */
static int pointer_prologue(int pm, uint64_t *ops)
{
    int i = 0;
    const uint32_t v = pm == P_SHIP ? 0xE : 0x30;
    if (pm != P_SHIP) {
        const uint32_t cb = pm == P_ZEROB ? 0x10000000u : 0;
        ops[i++] = NPUOP(OP_REG_CNA, v | cb, CNA_S_POINTER);
        ops[i++] = NPUOP(OP_REG_CORE, v | cb, CORE_S_POINTER);
    }
    ops[i++] = NPUOP(OP_REG_DPU, v, DPU_S_POINTER);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_, v, DPU_RDMA_S_POINTER_);
    return i;
}

/* Rewrite a generated program's own pointer writes for mode `pm`: drop them, and open with
 * the mode's prologue. Returns the new length. */
static int apply_pointers(int pm, uint64_t *ops, int n)
{
    uint64_t tmp[STRIDE];
    int i, j = pointer_prologue(pm, tmp);
    for (i = 0; i < n; i++) {
        const uint16_t reg = (uint16_t)(ops[i] & 0xffff);
        if (reg == DPU_S_POINTER || reg == DPU_RDMA_S_POINTER_ ||
            reg == CNA_S_POINTER || reg == CORE_S_POINTER)
            continue;
        tmp[j++] = ops[i];
    }
    memcpy(ops, tmp, (size_t)j * 8);
    return j;
}

/* Which registers differ between the full programs for two slots: the ones a delta must
 * rewrite. The two programs must have the same length and the same register order. */
static int find_delta_regs(struct probe *p)
{
    uint64_t a[STRIDE], b[STRIDE];
    int na = gen_full(p, 0, p->N, a), nb = gen_full(p, 1, p->N, b);
    if (na < 0 || na != nb) return -1;
    p->ndreg = 0;
    for (int i = 0; i < na; i++) {
        if ((a[i] & 0xffff) != (b[i] & 0xffff) || (a[i] >> 48) != (b[i] >> 48)) return -1;
        if (a[i] != b[i]) {
            if (p->ndreg == 32) return -1;
            p->dreg[p->ndreg] = (uint16_t)(a[i] & 0xffff);
            p->dop[p->ndreg] = (uint16_t)(a[i] >> 48);
            p->ndreg++;
        }
    }
    return 0;
}

/* The delta for slot t under mode pm: pointers, the changed registers at slot t's values,
 * and the full program's trailer (its last 4 words). */
static int gen_delta(const struct probe *p, int t, int pm, uint64_t *ops)
{
    uint64_t full[STRIDE];
    int n = gen_full(p, t, p->N, full), i = pointer_prologue(pm, ops);
    if (n < 4) return -1;
    for (int d = 0; d < p->ndreg; d++)
        for (int k = 0; k < n; k++)
            if ((full[k] & 0xffff) == p->dreg[d] && (uint16_t)(full[k] >> 48) == p->dop[d]) {
                ops[i++] = full[k];
                break;
            }
    if ((full[n - 1] & 0xffff) != PC_OPERATION_ENABLE) return -1;
    for (int k = n - 4; k < n; k++) ops[i++] = full[k];
    return i;
}

static void pack_slot(struct probe *p, int t)
{
    int8_t *in = (int8_t *)p->in.ptr + (size_t)t * p->in_slot;
    int8_t *w = (int8_t *)p->w.ptr + (size_t)t * p->w_slot;
    memset(in, 0, p->in_slot);
    memset(w, 0, p->w_slot);
    for (int n = 1; n <= p->N; n++)
        for (int k = 1; k <= p->K; k++)
            w[weight_int8(p->K, n, k)] = p->B[t][(size_t)(n - 1) * p->K + (k - 1)];
    for (int m = 1; m <= p->M; m++)
        for (int k = 1; k <= p->K; k++)
            in[feature_data(p->K, p->M, 1, 16, k, m, 1)] = p->A[t][(size_t)(m - 1) * p->K + (k - 1)];
}

struct score {
    size_t wrong, unwritten;     /* against the task's own reference, over its columns */
    size_t lo_ok, hi_unw;        /* delta only: first N/2 columns right, the rest unwritten */
};

/* Score slot t over `ncols` columns (N, or N/2 for a scrub). */
static void score_slot(const struct probe *p, int t, int ncols, struct score *s)
{
    const int32_t *od = (const int32_t *)((const uint8_t *)p->out.ptr + (size_t)t * p->out_slot);
    memset(s, 0, sizeof(*s));
    for (int m = 1; m <= p->M; m++)
        for (int n = 1; n <= p->N; n++) {
            const int32_t v = od[feature_data(p->N, p->M, 1, 4, n, m, 1)];
            const int32_t want = p->ref[t][(size_t)(m - 1) * p->N + (n - 1)];
            const int unw = (uint32_t)v == 0xAAAAAAAAu;
            if (n <= ncols) {
                s->unwritten += unw;
                s->wrong += v != want;
            }
            if (n <= p->N / 2) s->lo_ok += v == want;
            else s->hi_unw += unw;
        }
}

/* Run one arm: `pm` and a task string of F/S/D. Returns 0 if every task was exact, 1 if
 * not, -1 if the arm could not be run. `ms` gets the submit-to-fence time. */
static int run_arm(int fd, struct probe *p, int pm, const char *tasks, int rep, int quiet,
                   double *ms)
{
    const int nt = (int)strlen(tasks);
    rocket_task_desc td[NT_MAX];
    uint32_t in_h[3], out_h[1];
    int exact = 1;

    rocket_bo_prep(fd, &p->rc, 1, 0);
    memset(p->rc.ptr, 0, p->rc.size);
    for (int t = 0; t < nt; t++) {
        uint64_t *ops = (uint64_t *)p->rc.ptr + (size_t)t * STRIDE;
        int n;
        if (tasks[t] == 'D') n = gen_delta(p, t, pm, ops);
        else {
            n = gen_full(p, t, tasks[t] == 'S' ? p->N / 2 : p->N, ops);
            if (n > 0) n = apply_pointers(pm, ops, n);
        }
        if (n <= 0) {
            rocket_bo_fini(fd, &p->rc);
            printf("  %s %s: task %d refused\n", pmode_name[pm], tasks, t);
            return -1;
        }
        td[t].regcmd = (uint32_t)(p->rc.dma_address + (size_t)t * STRIDE * 8);
        td[t].regcmd_count = (uint32_t)n;
    }
    rocket_bo_fini(fd, &p->rc);

    rocket_bo_prep(fd, &p->out, 1, 0);
    memset(p->out.ptr, 0xAA, p->out.size);
    rocket_bo_fini(fd, &p->out);

    in_h[0] = p->in.handle; in_h[1] = p->w.handle; in_h[2] = p->rc.handle;
    out_h[0] = p->out.handle;
    {
        unsigned long long i0[4] = { 0 }, i1[4] = { 0 };
        const int nc = npu_irqs(i0, 4);
        double t0 = now_ms();
        int rc = rocket_submit_tasks_flags(fd, td, (uint32_t)nt, in_h, 3, out_h, 1, 0);
        if (rc == 0) rc = rocket_bo_prep(fd, &p->out, 0, 2000000000ull);
        t0 = now_ms() - t0;
        if (ms) *ms = t0;
        if (rc != 0) {
            printf("  %-5s %-6s rep %d: submit or wait failed (%d)\n", pmode_name[pm], tasks,
                   rep, rc);
            return -1;
        }
        npu_irqs(i1, 4);
        if (!quiet) {
            printf("  %-5s %-6s rep %d: NPU interrupts by core", pmode_name[pm], tasks, rep);
            for (int c = 0; c < nc; c++) printf(" %llu", i1[c] - i0[c]);
            printf(", %.3f ms submit to fence\n", t0);
        }
    }
    for (int t = 0; t < nt; t++) {
        struct score s;
        const int ncols = tasks[t] == 'S' ? p->N / 2 : p->N;
        const size_t ne = (size_t)p->M * ncols;
        score_slot(p, t, ncols, &s);
        if (s.wrong) exact = 0;
        if (quiet) continue;
        printf("  %-5s %-6s rep %d task %d (%c): %5zu of %zu wrong, %zu unwritten", pmode_name[pm],
               tasks, rep, t, tasks[t], s.wrong, ne, s.unwritten);
        if (tasks[t] == 'D' && s.wrong)
            printf("  [scrub signature: %zu of %zu low columns right, %zu of %zu high unwritten]",
                   s.lo_ok, (size_t)p->M * (p->N / 2), s.hi_unw, (size_t)p->M * (p->N - p->N / 2));
        printf("\n");
    }
    rocket_bo_fini(fd, &p->out);
    return exact ? 0 : 1;
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    struct probe p = { .M = 64, .K = 256, .N = 64 };
    int reps = 4, timing = 40, fd, bad_ctl = 0, fd_exact = 1;
    static const struct { int pm; const char *tasks; int control; } arms[] = {
        { P_SHIP, "FFFF", 1 }, { P_SHIP, "FSD", 0 }, { P_SHIP, "SFD", 0 },
        { P_SHIP, "FFDDDD", 0 }, { P_SHIP, "SSFD", 0 },
        { P_ZERO, "FF", 0 }, { P_ZERO, "FD", 0 }, { P_ZEROB, "FF", 0 }, { P_ZEROB, "FD", 0 },
    };

    if (argc > 3) { p.M = atoi(argv[1]); p.K = atoi(argv[2]); p.N = atoi(argv[3]); }
    if (argc > 4) reps = atoi(argv[4]);
    if (argc > 5) timing = atoi(argv[5]);
    if (p.M < 4 || p.M % 4 || p.K % 32 || p.K < 64 || p.N % 64 || p.N < 64) {
        fprintf(stderr, "usage: %s [M K N [reps [timing]]], M%%4, K%%32 >= 64, N%%64 >= 64\n",
                argv[0]);
        return 1;
    }
    fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }

    p.in_slot = ((size_t)p.M * p.K + 4095) & ~(size_t)4095;
    p.w_slot = ((size_t)p.N * p.K + 4095) & ~(size_t)4095;
    p.out_slot = ((size_t)p.M * p.N * 4 + 4095) & ~(size_t)4095;
    if (rocket_bo_alloc32(fd, 4096, &p.guard) ||
        rocket_bo_alloc32(fd, p.in_slot * NT_MAX, &p.in) ||
        rocket_bo_alloc32(fd, p.w_slot * NT_MAX, &p.w) ||
        rocket_bo_alloc32(fd, p.out_slot * NT_MAX, &p.out) ||
        rocket_bo_alloc32(fd, (size_t)STRIDE * 8 * NT_MAX, &p.rc)) {
        printf("BO allocation failed\n");
        return 1;
    }
    for (int t = 0; t < NT_MAX; t++) {
        size_t na = (size_t)p.M * p.K, nb = (size_t)p.N * p.K;
        p.A[t] = malloc(na); p.B[t] = malloc(nb);
        p.ref[t] = malloc((size_t)p.M * p.N * sizeof(int32_t));
        tf_fill_i8(p.A[t], na, 0xD0 + (uint64_t)t, -128, 127);
        tf_fill_i8(p.B[t], nb, 0xE0 + (uint64_t)t, -128, 127);
        cpu_ref(p.A[t], p.B[t], p.M, p.K, p.N, p.ref[t]);
    }
    rocket_bo_prep(fd, &p.in, 1, 0);
    rocket_bo_prep(fd, &p.w, 1, 0);
    for (int t = 0; t < NT_MAX; t++) pack_slot(&p, t);
    rocket_bo_fini(fd, &p.in);
    rocket_bo_fini(fd, &p.w);

    if (find_delta_regs(&p) != 0) {
        printf("the two full programs do not line up register for register\n");
        return 1;
    }
    printf("== regcmd delta probe: %dx%dx%d (scrub N %d), %d reps ==\n", p.M, p.K, p.N,
           p.N / 2, reps);
    printf("a delta rewrites %d registers:", p.ndreg);
    for (int d = 0; d < p.ndreg; d++) printf(" 0x%04x", p.dreg[d]);
    printf(" (plus its pointers and the 4-word trailer)\n");

    for (int r = 0; r < reps; r++)
        for (size_t a = 0; a < sizeof(arms) / sizeof(arms[0]); a++) {
            int rc = run_arm(fd, &p, arms[a].pm, arms[a].tasks, r, 0, NULL);
            if (arms[a].control && rc != 0) bad_ctl++;
            if (arms[a].pm == P_SHIP && !strcmp(arms[a].tasks, "FFDDDD") && rc != 0) fd_exact = 0;
        }

    if (timing > 0 && fd_exact && !bad_ctl) {
        double *tff = calloc((size_t)timing, sizeof(double));
        double *tfd = calloc((size_t)timing, sizeof(double));
        int ok = 1;
        for (int r = 0; r < timing && ok; r++) {
            if ((r & 1) == 0)
                ok = run_arm(fd, &p, P_SHIP, "FFFF", r, 1, &tff[r]) == 0 &&
                     run_arm(fd, &p, P_SHIP, "FFDD", r, 1, &tfd[r]) == 0;
            else
                ok = run_arm(fd, &p, P_SHIP, "FFDD", r, 1, &tfd[r]) == 0 &&
                     run_arm(fd, &p, P_SHIP, "FFFF", r, 1, &tff[r]) == 0;
        }
        if (ok) {
            qsort(tff, (size_t)timing, sizeof(double), cmp_d);
            qsort(tfd, (size_t)timing, sizeof(double), cmp_d);
            printf("== timing, %d interleaved reps, submit to fence: ship FFFF median %.4f ms "
                   "[p10 %.4f p90 %.4f], ship FFDD median %.4f ms [p10 %.4f p90 %.4f] ==\n",
                   timing, tff[timing / 2], tff[timing / 10], tff[timing * 9 / 10],
                   tfd[timing / 2], tfd[timing / 10], tfd[timing * 9 / 10]);
        } else {
            printf("== timing: a rep was not exact, no numbers ==\n");
        }
        free(tff); free(tfd);
    }
    printf("== controls %s ==\n", bad_ctl ? "NOT exact: nothing above is scoreable"
                                         : "exact on every rep");
    rocket_close(fd);
    return bad_ctl ? 1 : 0;
}
