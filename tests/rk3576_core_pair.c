// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_core_pair.c — what makes two RK3576 NPU cores in flight compute wrong answers.
 *
 * With both cores bound, a multi-fd caller intermittently gets a matmul back with a few
 * positions off by a small amount while the rest of the surface is exact. It is not a
 * throughput question — the second core buys nothing over a one-core control — so the
 * only reason to open it is to say what the mechanism is, and whether anything about it
 * reaches the supported single-core configuration.
 *
 * THE TWO HYPOTHESES DIFFER IN WHETHER SIMULTANEITY IS NEEDED, and that is what the
 * `pair` matrix asks. Every cell runs the same total work through the same two fds; only
 * the concurrency and the bound-core set change.
 *
 *   solo        one fd, one job at a time. The supported configuration.
 *   serial      TWO fds, but a mutex so at most one job is in flight. The kernel still
 *               spreads those jobs over both cores, so a program still lands on a core
 *               the previous program did not run on.
 *   concurrent  two fds, both driving at once. Two jobs execute at the same time.
 *
 * IT IS SIMULTANEOUS EXECUTION, AND FOUR OTHER MECHANISMS ARE REFUTED. `serial` is clean
 * — 96 calls per cell, zero wrong, at 256x512x2048, 32x1024x512 and 128x1024x1024. The
 * same two fds driving at once are wrong on 96-100% of calls, over as much as 94% of the
 * surface, with deltas spanning the whole int8 range. Refuted: a program inheriting the
 * other core's register file (it would be wrong in `serial` too); the two register banks
 * being one aliased twice (they read back distinct values); the IOMMU domain (separate
 * MMUs in separate groups, and a driver forced to give every context ONE domain still
 * corrupts with zero domain swaps); and the vendor's RK3576-only state_init together
 * with its pc_dma_ctrl PC serialization (applied as a whole delta, no change).
 * [HW sweep, H96 MAX M9]
 *
 * WHAT MAKES A CELL A MEASUREMENT IS THE PLACEMENT COUNTERS, NOT THE ACTIVE TIME. Both
 * cores report runtime-active time for the whole wall in every cell because both stay
 * resumed — that says POWERED, not that the cell's jobs were spread over them, and a
 * cell whose jobs all landed on one core is the one-core control wearing a different
 * label. That is not a hypothetical: `ctx`'s dup-fd cell is clean precisely because one
 * drm_sched entity never has two jobs in flight, which the counters show and the wall
 * time does not. Run this against an instrumented rocket (stat_jobs / stat_attach /
 * stat_overlap module parameters) whenever a cell's result is going to be believed.
 *
 * The corruption is GROSS, not sparse. An earlier reading of it as "a few positions off
 * by +/-2" came from checking against a CPU model of the DPU's integer requant, which
 * disagrees by one count on a fraction of a percent of elements in EVERY configuration —
 * noise larger than the signal it was being used to find.
 *
 * THE REFERENCE IS A SOLO RUN, not a CPU model — see cp_golden().
 *
 * `serial` is the load-bearing cell and it is easy to get wrong. A mutex around the
 * library call is not enough on its own to prove jobs alternated cores — read the
 * per-core `runtime_active_time` the probe reports and confirm both moved.
 *
 * What the two jobs are doing to each other is only partly decoded. `cross` gives each
 * worker its own weights and asks whether a wrong element holds the OTHER worker's
 * correct value: 1.7-3.3% of them do, against 0.4% by chance on int8 — real cross-talk, but
 * not a clean swap, so the dominant effect is neither job's arithmetic. That is
 * consistent with state that should be per-core being shared (an operand path, a CBUF, a
 * mapping) and not with anything a job could inherit between submits.
 *
 * THE ERROR IS READ IN MAC TERMS, not in output counts. `chars` runs the same
 * concurrent cell with operands in {0,1} and an output scale of 1, so the requant is the
 * identity and the int8 output IS the accumulator: a wrong element then says directly
 * how many multiply-accumulate terms went the wrong way. A delta of a few terms is a
 * datapath that dropped or repeated part of a contraction; a delta of hundreds is a
 * stale operand. With the dense operands of `pair` neither is distinguishable, because
 * the requant compresses a wide accumulator delta into a couple of output counts.
 *
 *   sudo -E ./build/rk3576_core_pair pair            # the concurrency matrix
 *   sudo -E ./build/rk3576_core_pair pair 256 512 2048
 *   sudo -E ./build/rk3576_core_pair chars           # decode the delta in MAC terms
 *   sudo -E ./build/rk3576_core_pair cross           # do the two jobs read each other?
 *   sudo -E ./build/rk3576_core_pair ctx             # one kernel context or two
 *   sudo -E ./build/rk3576_core_pair xproc           # one fd per PROCESS
 *   sudo -E ./build/rk3576_core_pair stagger         # how long the window is
 *
 * THE WINDOW IS THE JOB'S OWN DURATION. `stagger` delays one worker past a common
 * barrier, one call each per rep: 100% of calls wrong at +0 and +100 us, 91.7% at +250,
 * 4.2% at +500, and zero from +750 us out to +5 ms, with the driver's overlap count
 * reaching zero at the same rung. The submit floor is 439 us, so whatever is contended is
 * held for the whole submit rather than for a phase of it. [HW sweep, H96 MAX M9]
 *
 * IT CROSSES PROCESSES. `xproc` gives each of two processes one fd, which no library
 * mutex can reach across, and they corrupt exactly as two fds in one process do. There is
 * no userspace mitigation; the envelope is one job in flight at a time, part-wide.
 *
 * Env: ROCKET_CP_REPS reps per cell (12), ROCKET_CP_CALLS matmuls per rep per worker (4),
 * ROCKET_CP_SHOW wrong elements to name (10).
 *
 * BIND STATE IS NOT SET BY THIS PROBE — it reports it and leaves it alone. Run the whole
 * matrix twice, once with core 1 bound and once with it unbound, because the one-core
 * run of `concurrent` is the control that says whether two fds alone do this:
 *   echo 27708000.npu | sudo tee /sys/bus/platform/drivers/rocket/unbind
 *   echo 27708000.npu | sudo tee /sys/bus/platform/drivers/rocket/bind
 *
 * The plain int8 entry is used throughout: its output element is one byte, so it carries
 * none of the wide-output poisoning, whose interaction with two cores in flight is not
 * characterised. Do not switch this to an int32 or fp16 output.
 *
 * This is a PROBE. It exits 0 on a completed run and 1 only if it could not run at all —
 * a wrong answer is the measurement, not a failure.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

#define CP_MAX_WORKERS 2
#define CP_FIRST_BIG_CPU 4

static int env_int(const char *k, int dflt)
{
    const char *e = getenv(k);
    return (e && *e) ? (int)strtol(e, NULL, 0) : dflt;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ---------------------------------------------------------------------------
 * Which cores are bound, and how much each of them ran.
 *
 * There is no per-core submit ioctl, so the only evidence that a cell's jobs really
 * did land on both cores is the driver's own runtime accounting. A `serial` cell whose
 * second core did not move is not a measurement of anything.
 * ------------------------------------------------------------------------- */
#define CP_MAX_DEV 4
static char cp_dev[CP_MAX_DEV][96];
static int  cp_ndev;

static void cp_scan_devs(void)
{
    DIR *d = opendir("/sys/bus/platform/drivers/rocket");
    struct dirent *e;
    cp_ndev = 0;
    if (!d) return;
    while ((e = readdir(d)) && cp_ndev < CP_MAX_DEV)
        if (strstr(e->d_name, ".npu"))
            snprintf(cp_dev[cp_ndev++], sizeof cp_dev[0], "%s", e->d_name);
    closedir(d);
}

static unsigned long long cp_active_ms(int i)
{
    char path[320];
    unsigned long long v = 0;
    FILE *f;
    snprintf(path, sizeof path,
             "/sys/bus/platform/drivers/rocket/%s/power/runtime_active_time", cp_dev[i]);
    f = fopen(path, "r");
    if (!f) return 0;
    if (fscanf(f, "%llu", &v) != 1) v = 0;
    fclose(f);
    return v;   /* milliseconds */
}

/* ---------------------------------------------------------------------------
 * Where the jobs went, from the driver rather than from inference.
 *
 * Both cores read `runtime_active_time` for the whole wall in every cell, because both
 * stay runtime-resumed throughout -- that says the cores are POWERED, not that a cell's
 * jobs were spread over them. A cell whose jobs all landed on one core is the one-core
 * control wearing a different label, so the placement has to be read out.
 *
 * These come from an instrumented rocket (patches/rk3576/npu, conc counters): jobs per
 * core, how many of those had to swap that core's IOMMU domain, and how many started
 * while another core already had a job in flight. Absent on a stock module, in which
 * case the cell prints nothing and the reader is told so.
 * ------------------------------------------------------------------------- */
#define CP_STAT_JOBS    "/sys/module/rocket/parameters/stat_jobs"
#define CP_STAT_ATTACH  "/sys/module/rocket/parameters/stat_attach"
#define CP_STAT_OVERLAP "/sys/module/rocket/parameters/stat_overlap"

struct cp_stat {
    unsigned jobs[CP_MAX_DEV], attach[CP_MAX_DEV], overlap;
    int present;
};

static void cp_stat_array(const char *path, unsigned *out, int *present)
{
    FILE *f = fopen(path, "r");
    int i = 0;
    for (i = 0; i < CP_MAX_DEV; i++) out[i] = 0;
    if (!f) return;
    *present = 1;
    for (i = 0; i < CP_MAX_DEV; i++) {
        if (fscanf(f, "%u", &out[i]) != 1) break;
        if (fgetc(f) != ',') break;
    }
    fclose(f);
}

static void cp_stat_read(struct cp_stat *s)
{
    FILE *f;
    memset(s, 0, sizeof *s);
    cp_stat_array(CP_STAT_JOBS, s->jobs, &s->present);
    cp_stat_array(CP_STAT_ATTACH, s->attach, &s->present);
    f = fopen(CP_STAT_OVERLAP, "r");
    if (f) {
        if (fscanf(f, "%u", &s->overlap) != 1) s->overlap = 0;
        fclose(f);
        s->present = 1;
    }
}

/* ---------------------------------------------------------------------------
 * The workers.
 * ------------------------------------------------------------------------- */
/* How a worker gets its fd.
 *
 * CP_FD_OWN     rocket_open() per worker: two kernel contexts, so two IOMMU domains,
 *               and the driver detaches and re-attaches a core's domain whenever the
 *               core's next job comes from the other context.
 * CP_FD_DUP     worker 1 dup()s worker 0's: two fd NUMBERS over ONE open file, so one
 *               rocket_file_priv, one IOMMU domain and one IOVA allocator. Two jobs are
 *               still in flight on two cores; what is removed is the domain switching.
 *               That is the whole difference between the two cells.
 */
#define CP_FD_OWN 0
#define CP_FD_DUP 1
static int cp_fd_mode = CP_FD_OWN;
static int cp_delay_us;         /* worker 1's head start, for the stagger sweep */

struct cp_shared {
    int M, K, N, calls;
    const int8_t *A, *B;
    const int8_t *ref;
    /* `cross` gives each worker its OWN weights and its own golden, so a wrong element
     * can be asked whether it holds the OTHER worker's correct value. */
    const int8_t *B2, *ref2;
    float scale;
    int serialize;              /* hold the mutex across the library call */
    int delay_us;               /* worker 1 waits this long past the barrier */
    pthread_mutex_t lock;
    pthread_barrier_t start;
};

struct cp_worker {
    pthread_t thread;
    int index, fd, cpu;
    struct cp_shared *sh;
    int8_t *C;
    int rc;
    unsigned calls_wrong;       /* library calls whose result differed from the model */
    unsigned elems_wrong;       /* wrong elements, summed over calls                  */
    unsigned elems_other;       /* wrong elements holding the OTHER worker's value     */
    int maxdiff;
    int8_t *first_bad;          /* a copy of the first wrong result                   */
    int has_bad;
    double us;
};

static void *cp_run(void *arg)
{
    struct cp_worker *w = arg;
    struct cp_shared *sh = w->sh;
    size_t elems = (size_t)sh->M * sh->N;
    double t0;
    int c;

    if (w->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(w->cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
    pthread_barrier_wait(&sh->start);
    if (w->index && sh->delay_us) {
        struct timespec ts = { 0, (long)sh->delay_us * 1000L };
        nanosleep(&ts, NULL);
    }
    t0 = now_ms();
    for (c = 0; c < sh->calls; c++) {
        int rc;
        memset(w->C, 0, elems);
        if (sh->serialize) pthread_mutex_lock(&sh->lock);
        rc = rocket_matmul_int8_rk3576(w->fd, sh->M, sh->K, sh->N, sh->A,
                                       (w->index && sh->B2) ? sh->B2 : sh->B,
                                       NULL, sh->scale, w->C);
        if (sh->serialize) pthread_mutex_unlock(&sh->lock);
        if (rc != 0) { w->rc = rc; break; }
        {
            const int8_t *mine  = (w->index && sh->ref2) ? sh->ref2 : sh->ref;
            const int8_t *other = (w->index && sh->ref2) ? sh->ref  : sh->ref2;
            unsigned bad = 0;
            for (size_t i = 0; i < elems; i++)
                if (w->C[i] != mine[i]) {
                    if (other && w->C[i] == other[i]) w->elems_other++;
                    int d = (int)w->C[i] - (int)mine[i];
                    if (d < 0) d = -d;
                    if (d > w->maxdiff) w->maxdiff = d;
                    bad++;
                }
            if (bad) {
                w->calls_wrong++;
                w->elems_wrong += bad;
                if (!w->has_bad) { memcpy(w->first_bad, w->C, elems); w->has_bad = 1; }
            }
        }
    }
    w->us = (now_ms() - t0) * 1000.0;
    return NULL;
}

/* THE REFERENCE IS THE PART ITSELF, RUN ALONE. A CPU model of this entry has to
 * reproduce the DPU's integer requant exactly — `(acc*SCALE)>>SHIFT` against a float
 * round — and where it does not it disagrees on a fraction of a percent of elements by
 * one count, densely, in every configuration. That noise is far larger than the sparse
 * corruption being looked for and would drown it. A solo run has no such gap: whatever
 * the requant does, it does the same thing to both sides, so every difference left is
 * the concurrency.
 *
 * It is taken twice and required to agree, because the golden run is exposed to the
 * same dropped atoms as everything else — an unstable golden would report the whole
 * matrix as corrupt. Returns 0 if it never settled. */
static int cp_golden(int M, int K, int N, const int8_t *A, const int8_t *B,
                     float scale, int8_t *ref)
{
    size_t elems = (size_t)M * N;
    int8_t *tmp = malloc(elems);
    int fd, tries, ok = 0;
    if (!tmp) return 0;
    fd = rocket_open();
    if (fd < 0) { free(tmp); return 0; }
    for (tries = 0; tries < 6 && !ok; tries++) {
        if (rocket_matmul_int8_rk3576(fd, M, K, N, A, B, NULL, scale, ref) != 0) break;
        if (rocket_matmul_int8_rk3576(fd, M, K, N, A, B, NULL, scale, tmp) != 0) break;
        ok = memcmp(ref, tmp, elems) == 0;
    }
    rocket_close(fd);
    free(tmp);
    return ok;
}

struct cp_result {
    unsigned reps, reps_wrong, calls, calls_wrong, elems_wrong, elems_other;
    int maxdiff;
    double ms;
    unsigned long long active[CP_MAX_DEV];
};

/* One cell of the matrix: `workers` fds, `serialize` on or off, repeated. */
static void cp_cell(const char *label, int workers, int serialize,
                    int M, int K, int N, const int8_t *A, const int8_t *B,
                    const int8_t *ref, const int8_t *B2, const int8_t *ref2,
                    float scale, unsigned reps, int calls,
                    struct cp_result *out, int8_t **keep_bad)
{
    size_t elems = (size_t)M * N;
    unsigned long long a0[CP_MAX_DEV], a1[CP_MAX_DEV];
    struct cp_stat s0, s1;
    double t0 = now_ms();

    memset(out, 0, sizeof *out);
    cp_stat_read(&s0);
    for (int i = 0; i < cp_ndev; i++) a0[i] = cp_active_ms(i);

    for (unsigned r = 0; r < reps; r++) {
        struct cp_shared sh;
        struct cp_worker w[CP_MAX_WORKERS];
        int ok = 1;

        memset(&sh, 0, sizeof sh);
        sh.M = M; sh.K = K; sh.N = N; sh.calls = calls;
        sh.A = A; sh.B = B; sh.ref = ref; sh.scale = scale;
        sh.B2 = B2; sh.ref2 = ref2;
        sh.serialize = serialize;
        sh.delay_us = cp_delay_us;
        pthread_mutex_init(&sh.lock, NULL);
        pthread_barrier_init(&sh.start, NULL, (unsigned)workers);

        memset(w, 0, sizeof w);
        for (int i = 0; i < workers; i++) {
            w[i].index = i;
            w[i].sh = &sh;
            w[i].cpu = CP_FIRST_BIG_CPU + i;
            w[i].C = malloc(elems);
            w[i].first_bad = malloc(elems);
            w[i].fd = (cp_fd_mode == CP_FD_DUP && i > 0 && w[0].fd >= 0)
                          ? dup(w[0].fd) : rocket_open();
            if (w[i].fd < 0 || !w[i].C || !w[i].first_bad) ok = 0;
        }
        if (ok) {
            for (int i = 0; i < workers; i++)
                pthread_create(&w[i].thread, NULL, cp_run, &w[i]);
            for (int i = 0; i < workers; i++)
                pthread_join(w[i].thread, NULL);
        }

        {
            unsigned rep_bad = 0;
            for (int i = 0; i < workers; i++) {
                out->calls       += (unsigned)calls;
                out->calls_wrong += w[i].calls_wrong;
                out->elems_wrong += w[i].elems_wrong;
                out->elems_other += w[i].elems_other;
                if (w[i].maxdiff > out->maxdiff) out->maxdiff = w[i].maxdiff;
                rep_bad += w[i].calls_wrong;
                /* Keep the first wrong surface the matrix ever produces, so `chars` and
                 * the position report have something to decode without re-running. */
                if (w[i].has_bad && keep_bad && !*keep_bad) {
                    *keep_bad = malloc(elems);
                    if (*keep_bad) memcpy(*keep_bad, w[i].first_bad, elems);
                }
            }
            out->reps++;
            if (rep_bad) out->reps_wrong++;
        }
        for (int i = 0; i < workers; i++) {
            if (w[i].fd >= 0) rocket_close(w[i].fd);
            free(w[i].C); free(w[i].first_bad);
        }
        pthread_barrier_destroy(&sh.start);
        pthread_mutex_destroy(&sh.lock);
    }

    for (int i = 0; i < cp_ndev; i++) a1[i] = cp_active_ms(i);
    for (int i = 0; i < cp_ndev; i++) out->active[i] = a1[i] - a0[i];
    out->ms = now_ms() - t0;

    printf("  %-12s %5u reps %6u calls | %5u calls wrong (%5.1f%%) %8u elems maxdiff %-4d | "
           "%7.0f ms | active",
           label, out->reps, out->calls, out->calls_wrong,
           out->calls ? 100.0 * out->calls_wrong / out->calls : 0.0,
           out->elems_wrong, out->maxdiff, out->ms);
    for (int i = 0; i < cp_ndev; i++)
        printf(" %s=%llums", cp_dev[i], out->active[i]);
    if (out->elems_wrong)
        printf(" | %u (%.1f%%) hold the OTHER worker's value",
               out->elems_other, 100.0 * out->elems_other / out->elems_wrong);
    printf("\n");

    cp_stat_read(&s1);
    if (s1.present) {
        printf("      %-12s jobs", "");
        for (int i = 0; i < cp_ndev; i++)
            printf(" core%d=%u", i, s1.jobs[i] - s0.jobs[i]);
        printf(" | domain swaps");
        for (int i = 0; i < cp_ndev; i++)
            printf(" core%d=%u", i, s1.attach[i] - s0.attach[i]);
        printf(" | started with another core in flight: %u\n",
               s1.overlap - s0.overlap);
    }
    fflush(stdout);
}

/* Where the wrong elements are, and what the delta is. With unit operands and scale 1
 * the delta is the number of MAC terms that went the wrong way. */
static void cp_report(int M, int N, const int8_t *got, const int8_t *ref, int unit)
{
    unsigned shown = 0, cap = (unsigned)env_int("ROCKET_CP_SHOW", 10);
    unsigned n = 0, rowmask_lo = 0;
    int mmin = -1, mmax = -1, nmin = -1, nmax = -1;
    int hist[9];    /* deltas -4..+4, anything else counted as `other`  */
    unsigned other = 0;
    memset(hist, 0, sizeof hist);

    for (int m = 0; m < M; m++)
        for (int j = 0; j < N; j++) {
            size_t i = (size_t)m * N + j;
            int d;
            if (got[i] == ref[i]) continue;
            d = (int)got[i] - (int)ref[i];
            n++;
            if (mmin < 0) mmin = m;
            mmax = m;
            if (nmin < 0 || j < nmin) nmin = j;
            if (j > nmax) nmax = j;
            if (m < 32) rowmask_lo |= 1u << m;
            if (d >= -4 && d <= 4) hist[d + 4]++; else other++;
            if (shown < cap) {
                printf("        m=%-5d n=%-5d got=%-5d want=%-5d delta=%+d%s\n",
                       m, j, got[i], ref[i], d,
                       unit ? " MAC terms" : "");
                shown++;
            }
        }
    printf("      -> %u wrong of %d | rows %d..%d, cols %d..%d | low-row mask 0x%x\n",
           n, M * N, mmin, mmax, nmin, nmax, rowmask_lo);
    printf("      -> delta histogram");
    for (int d = -4; d <= 4; d++) if (hist[d + 4]) printf("  %+d:%d", d, hist[d + 4]);
    if (other) printf("  beyond +/-4: %u", other);
    printf("\n");
}

/* ---------------------------------------------------------------------------
 * The cross-process control.
 *
 * Every other cell runs two fds in ONE process, which a library mutex can serialize.
 * Two processes each holding one fd cannot be serialized that way, so if that corrupts
 * too there is no userspace mitigation and the fix has to be in the kernel or the DTS.
 * fork() after the operands and the reference are in place: the children inherit them
 * copy-on-write and report through a shared page.
 * ------------------------------------------------------------------------- */
struct cp_xproc_slot {
    unsigned calls, calls_wrong, elems_wrong;
    int maxdiff, rc;
};

struct cp_xproc {
    volatile int go;
    struct cp_xproc_slot slot[CP_MAX_WORKERS];
};

static void cp_xproc_child(struct cp_xproc *sh, int index, int M, int K, int N,
                           const int8_t *A, const int8_t *B, const int8_t *ref,
                           float scale, int calls)
{
    struct cp_xproc_slot *s = &sh->slot[index];
    size_t elems = (size_t)M * N;
    int8_t *C = malloc(elems);
    cpu_set_t set;
    int fd;

    CPU_ZERO(&set);
    CPU_SET(CP_FIRST_BIG_CPU + index, &set);
    sched_setaffinity(0, sizeof set, &set);

    fd = rocket_open();
    if (fd < 0 || !C) { s->rc = -1; _exit(1); }

    while (!sh->go) sched_yield();

    for (int c = 0; c < calls; c++) {
        unsigned bad = 0;
        memset(C, 0, elems);
        if (rocket_matmul_int8_rk3576(fd, M, K, N, A, B, NULL, scale, C) != 0) {
            s->rc = -2;
            break;
        }
        s->calls++;
        for (size_t i = 0; i < elems; i++)
            if (C[i] != ref[i]) {
                int d = (int)C[i] - (int)ref[i];
                if (d < 0) d = -d;
                if (d > s->maxdiff) s->maxdiff = d;
                bad++;
            }
        if (bad) { s->calls_wrong++; s->elems_wrong += bad; }
    }
    rocket_close(fd);
    _exit(0);
}

static void cp_xproc_cell(const char *label, int procs, int M, int K, int N,
                          const int8_t *A, const int8_t *B, const int8_t *ref,
                          float scale, unsigned reps, int calls)
{
    unsigned tot_calls = 0, tot_wrong = 0, tot_elems = 0;
    int maxdiff = 0;
    double t0 = now_ms();

    for (unsigned r = 0; r < reps; r++) {
        struct cp_xproc *sh = mmap(NULL, sizeof *sh, PROT_READ | PROT_WRITE,
                                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        pid_t pid[CP_MAX_WORKERS];
        if (sh == MAP_FAILED) { printf("  %-12s mmap failed\n", label); return; }
        memset(sh, 0, sizeof *sh);

        for (int i = 0; i < procs; i++) {
            pid[i] = fork();
            if (pid[i] == 0)
                cp_xproc_child(sh, i, M, K, N, A, B, ref, scale, calls);
        }
        /* Both children are open and spinning before either submits. */
        usleep(120000);
        sh->go = 1;
        for (int i = 0; i < procs; i++) {
            int st;
            waitpid(pid[i], &st, 0);
        }
        for (int i = 0; i < procs; i++) {
            tot_calls += sh->slot[i].calls;
            tot_wrong += sh->slot[i].calls_wrong;
            tot_elems += sh->slot[i].elems_wrong;
            if (sh->slot[i].maxdiff > maxdiff) maxdiff = sh->slot[i].maxdiff;
        }
        munmap(sh, sizeof *sh);
    }

    printf("  %-12s %5u reps %6u calls | %5u calls wrong (%5.1f%%) %8u elems maxdiff %-4d | "
           "%7.0f ms\n",
           label, reps, tot_calls, tot_wrong,
           tot_calls ? 100.0 * tot_wrong / tot_calls : 0.0,
           tot_elems, maxdiff, now_ms() - t0);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    /* This probe exists to put two jobs in flight at once, which is what the
     * library refuses to let a caller build by accident. Lift its own guard. */
    setenv("ROCKET_RK3576_ALLOW_MULTI_FD", "1", 0);
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *mode = argc > 1 ? argv[1] : "pair";
    int M = argc > 2 ? atoi(argv[2]) : 256;
    int K = argc > 3 ? atoi(argv[3]) : 512;
    int N = argc > 4 ? atoi(argv[4]) : 2048;
    unsigned reps = (unsigned)env_int("ROCKET_CP_REPS", 12);
    int calls = env_int("ROCKET_CP_CALLS", 4);
    int unit = !strcmp(mode, "chars");
    int8_t *A, *B, *ref, *bad = NULL;
    float scale;
    struct cp_result res;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_core_pair: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    cp_scan_devs();
    if (!cp_ndev) { printf("rk3576_core_pair: no rocket devices bound\n"); return 1; }

    A   = malloc((size_t)M * K);
    B   = malloc((size_t)N * K);
    ref = malloc((size_t)M * N);
    if (!A || !B || !ref) return 1;

    if (unit) {
        /* Operands in {0,1} and an identity requant, so the output IS the accumulator
         * and a wrong element reads as a count of MAC terms. The density is chosen to
         * keep the accumulator inside int8 at this K — a saturated output would hide
         * exactly the small deltas being measured. */
        int density = K / 100 + 1;
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)((i % 3u) == 0);
        for (size_t i = 0; i < (size_t)N * K; i++)
            B[i] = (int8_t)(((i * 7u + 1u) % (unsigned)density) == 0);
        scale = 1.0f;
    } else {
        unsigned seed = 0x9E3779B9u ^ (unsigned)(M * 31 + K * 17 + N * 7);
        for (size_t i = 0; i < (size_t)M * K; i++) {
            seed = seed * 1103515245u + 12345u;
            A[i] = (int8_t)((int)((seed >> 16) % 255u) - 127);
        }
        for (size_t i = 0; i < (size_t)N * K; i++) {
            seed = seed * 1103515245u + 12345u;
            B[i] = (int8_t)((int)((seed >> 16) % 255u) - 127);
        }
        /* A scale that puts a typical accumulator well inside int8, so a saturated
         * output is not what a "wrong" element is measuring. */
        scale = 1.0f / (float)(K * 4);
    }
    if (!cp_golden(M, K, N, A, B, scale, ref)) {
        printf("rk3576_core_pair: the solo reference did not settle — two runs of the "
               "same matmul on one fd disagreed six times over. Nothing can be measured "
               "against that; fix the single-core path first.\n");
        free(A); free(B); free(ref);
        return 1;
    }

    printf("== RK3576 two cores in flight: %s, %dx%dx%d ==\n", mode, M, K, N);
    printf("   bound:");
    for (int i = 0; i < cp_ndev; i++) printf(" %s", cp_dev[i]);
    printf("   (%d core%s) | reps %u, %d calls per worker per rep%s\n",
           cp_ndev, cp_ndev == 1 ? "" : "s", reps, calls,
           unit ? ", unit operands and an identity requant" : "");
    if (cp_ndev < 2)
        printf("   NOTE: this is the ONE-CORE CONTROL. `concurrent` here is two fds on a "
               "single core, which is the null the two-core result has to beat.\n");

    if (unit) {
        cp_cell("concurrent", 2, 0, M, K, N, A, B, ref, NULL, NULL, scale, reps, calls,
                &res, &bad);
        if (bad) {
            printf("   the first wrong surface, in MAC terms:\n");
            cp_report(M, N, bad, ref, 1);
        } else {
            printf("   no wrong surface in this run — nothing to decode\n");
        }
    } else if (!strcmp(mode, "ctx")) {
        /* ONE kernel context or two, with everything else held fixed.
         *
         * Two fds are two rocket_file_privs, so two IOMMU domains, and a core whose
         * next job comes from the other fd detaches its domain and attaches the other
         * -- while the OTHER core is mid-DMA. dup() gives two fds over one open file,
         * so the domain never changes and the only thing left is two jobs computing at
         * the same time. If `dup` is clean the mechanism is the domain switch, which is
         * the driver's; if it corrupts too, the domain is exonerated. */
        cp_fd_mode = CP_FD_OWN;
        cp_cell("own-fd",  2, 0, M, K, N, A, B, ref, NULL, NULL, scale, reps, calls,
                &res, &bad);
        cp_fd_mode = CP_FD_DUP;
        cp_cell("dup-fd",  2, 0, M, K, N, A, B, ref, NULL, NULL, scale, reps, calls,
                &res, &bad);
        cp_fd_mode = CP_FD_OWN;
    } else if (!strcmp(mode, "xproc")) {
        /* One fd per PROCESS. No library mutex can reach across that. */
        cp_xproc_cell("1 process",  1, M, K, N, A, B, ref, scale, reps, calls * 2);
        cp_xproc_cell("2 processes", 2, M, K, N, A, B, ref, scale, reps, calls);
    } else if (!strcmp(mode, "stagger")) {
        /* How long the vulnerable window is. Both workers do ONE call per rep from a
         * common barrier, worker 1 delayed by d. At a shape that is a single submit,
         * the delay at which the corruption stops is the length of the overlap that
         * does the damage. */
        static const int delays[] = { 0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000 };
        printf("   one call per worker per rep, worker 1 delayed past the barrier\n");
        for (unsigned d = 0; d < sizeof delays / sizeof delays[0]; d++) {
            char label[32];
            cp_delay_us = delays[d];
            snprintf(label, sizeof label, "+%dus", delays[d]);
            cp_cell(label, 2, 0, M, K, N, A, B, ref, NULL, NULL, scale, reps, 1,
                    &res, &bad);
        }
        cp_delay_us = 0;
    } else if (!strcmp(mode, "cross")) {
        /* Each worker gets its own weights, so a wrong element can be asked whether it
         * holds the OTHER worker's correct value. If it does, the two jobs are reading
         * each other's memory and the mechanism is the mapping, not the arithmetic. */
        int8_t *B2 = malloc((size_t)N * K), *ref2 = malloc((size_t)M * N);
        unsigned seed = 0x12345677u;
        if (!B2 || !ref2) return 1;
        for (size_t i = 0; i < (size_t)N * K; i++) {
            seed = seed * 1103515245u + 12345u;
            B2[i] = (int8_t)((int)((seed >> 16) % 255u) - 127);
        }
        if (!cp_golden(M, K, N, A, B2, scale, ref2)) {
            printf("rk3576_core_pair: the second worker's solo reference did not "
                   "settle\n");
            return 1;
        }
        printf("   each worker has its own weights; `other` counts wrong elements that "
               "hold the other worker's correct value\n");
        cp_cell("solo",       1, 0, M, K, N, A, B, ref, B2, ref2, scale, reps,
                calls * 2, &res, &bad);
        /* `serial` in the plain matrix gives both workers the SAME operands, so a
         * mechanism that hands one worker the other's inputs is invisible there. Here
         * the weights differ, so a serialized cell can see it. */
        cp_cell("serial",     2, 1, M, K, N, A, B, ref, B2, ref2, scale, reps, calls,
                &res, &bad);
        cp_cell("concurrent", 2, 0, M, K, N, A, B, ref, B2, ref2, scale, reps, calls,
                &res, &bad);
        free(B2); free(ref2);
    } else {
        cp_cell("solo",       1, 0, M, K, N, A, B, ref, NULL, NULL, scale, reps,
                calls * 2, &res, &bad);
        cp_cell("serial",     2, 1, M, K, N, A, B, ref, NULL, NULL, scale, reps, calls,
                &res, &bad);
        cp_cell("concurrent", 2, 0, M, K, N, A, B, ref, NULL, NULL, scale, reps, calls,
                &res, &bad);
        if (bad) {
            printf("   the first wrong surface this matrix produced:\n");
            cp_report(M, N, bad, ref, 0);
        }
    }

    free(A); free(B); free(ref); free(bad);
    return 0;
}
