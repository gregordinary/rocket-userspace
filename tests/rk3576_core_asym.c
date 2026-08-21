// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_core_asym.c — is the resource the two RK3576 cores share a CAPACITY or a
 * CONTROL PATH?
 *
 * Two jobs executing at once on this part's two cores compute wrong answers on 96-100%
 * of calls. Four mechanisms are refuted (residual per-core state, the register banks
 * being one aliased twice, the IOMMU domain, the vendor's state_init plus its PC
 * serialization), leaving one named candidate — the CBUF, of which the TRM shows exactly
 * one, 1 MB at 0x3fe80000 with 16 banks gated from a single shared NPU_GRF field.
 *
 * EVERY EARLIER CELL GAVE BOTH WORKERS THE SAME SHAPE, so none of them could tell a
 * capacity from a control path. That is what this asks. One worker is the VICTIM and
 * never changes; the other is the AGGRESSOR and its footprint is swept from a handful of
 * CBUF granules to the victim's own. Both are scored separately.
 *
 * THE ANSWER IS BOTH, AND THE TWO SEPARATE CLEANLY. [HW sweep, H96 MAX M9, 3 runs]
 *
 *   The TRIGGER is not capacity. An aggressor whose whole feature plane is FOUR
 *   granules -- 256 bytes against a 448 KiB pool -- corrupts 62-76% of the victim's
 *   calls. Nothing is being exhausted; the collision is unconditional.
 *
 *   The EXTENT is the aggressor's FEATURE PLANE, and nothing else. Damage per corrupted
 *   call is monotone in its granules: 0.6-0.7% of the victim's surface at 4, 2.7-3.0% at
 *   32, 12.9-16.9% at 256, 27.9-38.9% at 1024, 48.5-55.3% at 2048.
 *
 * The last pair is what makes that a measurement rather than a correlation, because
 * footprint and submit duration co-vary across the first five. `wideN` is 4 granules
 * behind a 128 KiB N-tiled weight cube -- small plane, long submit -- and does 5.6-6.5%.
 * `flatN` is a 2048-granule plane against 32 KiB of weights -- large plane, short submit
 * -- and does 30.4-34.9%. Five to six times the damage from FOUR TIMES LESS weight and a
 * shorter occupancy, but 512x the feature plane. So it is neither the weight path nor
 * how long the intruder held anything.
 *
 * That names the resource: the two cores stage their FEATURE planes into the same CBUF,
 * unconditionally, and what an intruder destroys is however much plane it staged. What
 * is NOT established here is the geometry of the overlap -- whether they share a base,
 * and at what granularity. The badRow/badCol columns were meant to settle it and do not:
 * one corrupted call per cell is too small a sample and the shapes come back
 * heterogeneous.
 *
 * THE REFERENCE IS A SOLO RUN OF THE PART, NOT A CPU MODEL. A model of the DPU's integer
 * requant disagrees by one count on a fraction of a percent of elements in every
 * configuration, which is noise larger than some of the signals here. Each shape's
 * golden is taken twice, alone, and required to agree.
 *
 * A CELL IS NOT A MEASUREMENT UNTIL IT OVERLAPPED. The instrumented module exports
 * stat_overlap; this prints it per cell and prints "-" on a stock module. A tiny-
 * aggressor cell that reads clean is only interesting if the driver saw the two jobs
 * in flight together, and at a small shape the aggressor's own submit is short enough
 * that it may simply have missed the victim. Read the overlap column before the
 * wrongness column.
 *
 * Requires BOTH cores bound. With one core bound the kernel has nowhere to put the
 * second job and every cell is a one-core control wearing a different label.
 *
 *   ls /sys/bus/platform/drivers/rocket/ | grep npu
 *   echo 27708000.npu | sudo tee /sys/bus/platform/drivers/rocket/bind
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

#include "rocket_npu.h"
#include "rocket_matmul.h"

#define CA_FIRST_BIG_CPU 4
#define CA_MAX_DEV 4

#define CA_STAT_JOBS    "/sys/module/rocket/parameters/stat_jobs"
#define CA_STAT_OVERLAP "/sys/module/rocket/parameters/stat_overlap"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* The driver's own count of jobs that started while another core had one in flight.
 * Absent on a stock module, which is why `present` is carried separately: zero from an
 * absent counter and zero from a real one mean opposite things. */
static int ca_stat_overlap(unsigned *out)
{
    FILE *f = fopen(CA_STAT_OVERLAP, "r");
    if (!f) return 0;
    if (fscanf(f, "%u", out) != 1) *out = 0;
    fclose(f);
    return 1;
}

static int ca_ndev(void)
{
    DIR *d = opendir("/sys/bus/platform/drivers/rocket");
    struct dirent *e;
    int n = 0;
    if (!d) return 0;
    while ((e = readdir(d)))
        if (strstr(e->d_name, ".npu")) n++;
    closedir(d);
    return n;
}

struct ca_shape {
    const char *name;
    int M, K, N;
};

/* Per-worker operands and result. Each worker owns a whole problem, which is the point:
 * the two workers no longer share a shape. */
struct ca_prob {
    struct ca_shape s;
    int8_t *A, *B, *ref, *C;
    size_t elems;
};

static void ca_fill(int8_t *p, size_t n, unsigned seed)
{
    unsigned x = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        p[i] = (int8_t)((x >> 16) & 0xff);
    }
}

/* Take the reference alone, twice, and require it to agree. A golden run is exposed to
 * the same dropped atoms as everything else, and an unstable one would report the whole
 * matrix as corrupt. */
static int ca_golden(struct ca_prob *p, float scale)
{
    int8_t *tmp = malloc(p->elems);
    int fd, tries, ok = 0;
    if (!tmp) return 0;
    fd = rocket_open();
    if (fd < 0) { free(tmp); return 0; }
    for (tries = 0; tries < 6 && !ok; tries++) {
        if (rocket_matmul_int8_rk3576(fd, p->s.M, p->s.K, p->s.N,
                                      p->A, p->B, NULL, scale, p->ref) != 0) break;
        if (rocket_matmul_int8_rk3576(fd, p->s.M, p->s.K, p->s.N,
                                      p->A, p->B, NULL, scale, tmp) != 0) break;
        ok = memcmp(p->ref, tmp, p->elems) == 0;
    }
    rocket_close(fd);
    free(tmp);
    return ok;
}

static int ca_prob_init(struct ca_prob *p, struct ca_shape s, unsigned seed, float scale)
{
    memset(p, 0, sizeof *p);
    p->s = s;
    p->elems = (size_t)s.M * s.N;
    p->A   = malloc((size_t)s.M * s.K);
    p->B   = malloc((size_t)s.N * s.K);
    p->ref = malloc(p->elems);
    p->C   = malloc(p->elems);
    if (!p->A || !p->B || !p->ref || !p->C) return 0;
    ca_fill(p->A, (size_t)s.M * s.K, seed);
    ca_fill(p->B, (size_t)s.N * s.K, seed + 7u);
    return ca_golden(p, scale);
}

static void ca_prob_free(struct ca_prob *p)
{
    free(p->A); free(p->B); free(p->ref); free(p->C);
}

/* THE AGGRESSOR HAS TO DRIVE FOR THE VICTIM'S WHOLE RUN, not for a matching call count.
 * A small shape finishes a fixed number of calls inside the victim's HOST-SIDE packing —
 * which is most of a large matmul's wall on this part — and the driver then records zero
 * overlap for the cell, making it a one-core control wearing a different label. The
 * aggressor therefore loops until the victim says it is done. */
struct ca_worker {
    pthread_t thread;
    int index, fd, cpu, calls, rc, continuous;
    float scale;
    struct ca_prob *p;
    pthread_barrier_t *start;
    volatile int *done;
    unsigned calls_run, calls_wrong, elems_wrong;
    int maxdiff;
    /* THE SHAPE OF THE DAMAGE, which says which axis of the victim's work the intruder
     * lands on. The victim's output is tiled along N, so a corrupted SUBMIT shows up as
     * a column band spanning every row, and min/max row cannot tell that from anything
     * else. Count the DISTINCT rows and columns carrying a wrong element instead:
     * few columns and all rows is one N tile lost; all columns and few rows is an M
     * split; many of both is neither. Taken on the first corrupted call of the cell. */
    unsigned shape_rows, shape_cols, shape_taken;
    unsigned long long row_lo_sum, row_hi_sum;
    double us;
};

static void *ca_run(void *arg)
{
    struct ca_worker *w = arg;
    struct ca_prob *p = w->p;
    double t0;

    if (w->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(w->cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
    pthread_barrier_wait(w->start);
    t0 = now_ms();
    for (int c = 0; w->continuous ? !*w->done : c < w->calls; c++) {
        unsigned bad = 0;
        long row_lo = -1, row_hi = -1;
        memset(p->C, 0, p->elems);
        w->rc = rocket_matmul_int8_rk3576(w->fd, p->s.M, p->s.K, p->s.N,
                                          p->A, p->B, NULL, w->scale, p->C);
        if (w->rc != 0) break;
        w->calls_run++;
        for (size_t i = 0; i < p->elems; i++)
            if (p->C[i] != p->ref[i]) {
                int d = (int)p->C[i] - (int)p->ref[i];
                long row = (long)(i / (size_t)p->s.N);
                if (d < 0) d = -d;
                if (d > w->maxdiff) w->maxdiff = d;
                if (row_lo < 0) row_lo = row;
                row_hi = row;
                bad++;
            }
        if (bad) {
            w->calls_wrong++;
            w->elems_wrong += bad;
            w->row_lo_sum += (unsigned long long)row_lo;
            w->row_hi_sum += (unsigned long long)row_hi;
            if (!w->shape_taken) {
                unsigned char *rows = calloc((size_t)p->s.M, 1);
                unsigned char *cols = calloc((size_t)p->s.N, 1);
                if (rows && cols) {
                    for (size_t i = 0; i < p->elems; i++)
                        if (p->C[i] != p->ref[i]) {
                            rows[i / (size_t)p->s.N] = 1;
                            cols[i % (size_t)p->s.N] = 1;
                        }
                    for (int r = 0; r < p->s.M; r++) w->shape_rows += rows[r];
                    for (int cc = 0; cc < p->s.N; cc++) w->shape_cols += cols[cc];
                    w->shape_taken = 1;
                }
                free(rows); free(cols);
            }
        }
    }
    if (!w->continuous && w->done) *w->done = 1;
    w->us = (now_ms() - t0) * 1000.0;
    return NULL;
}

/* granules of feature plane a shape stages: the plane is M, the channels K, and a
 * 64-byte granule holds 64 of those bytes. */
static unsigned ca_granules(struct ca_shape s)
{
    return (unsigned)(((long long)s.M * s.K + 63) / 64);
}

struct ca_cell {
    unsigned calls[2], wrong[2], elems[2];
    unsigned long long row_lo[2], row_hi[2];
    unsigned shape_rows, shape_cols;
    int maxdiff[2];
    unsigned overlap;
    int overlap_present;
    double ms;
};

/* One cell: `nw` workers running their own problems at once, repeated. */
static int ca_cell(struct ca_prob *pv, struct ca_prob *pa, int nw, int calls,
                   unsigned reps, float scale, struct ca_cell *out)
{
    unsigned o0 = 0, o1 = 0;
    double t0 = now_ms();

    memset(out, 0, sizeof *out);
    out->overlap_present = ca_stat_overlap(&o0);

    for (unsigned r = 0; r < reps; r++) {
        struct ca_worker w[2];
        pthread_barrier_t start;
        volatile int done = 0;
        int ok = 1;

        pthread_barrier_init(&start, NULL, (unsigned)nw);
        memset(w, 0, sizeof w);
        for (int i = 0; i < nw; i++) {
            w[i].index      = i;
            w[i].cpu        = CA_FIRST_BIG_CPU + i;
            w[i].calls      = calls;
            w[i].scale      = scale;
            w[i].p          = i ? pa : pv;
            w[i].start      = &start;
            w[i].done       = &done;
            w[i].continuous = (i != 0);   /* the victim sets the run length */
            w[i].fd         = rocket_open();
            if (w[i].fd < 0) ok = 0;
        }
        if (ok) {
            for (int i = 0; i < nw; i++)
                pthread_create(&w[i].thread, NULL, ca_run, &w[i]);
            for (int i = 0; i < nw; i++)
                pthread_join(w[i].thread, NULL);
        } else {
            done = 1;
        }
        for (int i = 0; i < nw; i++) {
            out->calls[i] += w[i].calls_run;
            out->wrong[i] += w[i].calls_wrong;
            out->elems[i] += w[i].elems_wrong;
            out->row_lo[i] += w[i].row_lo_sum;
            out->row_hi[i] += w[i].row_hi_sum;
            if (i == 0 && w[i].shape_taken && !out->shape_rows) {
                out->shape_rows = w[i].shape_rows;
                out->shape_cols = w[i].shape_cols;
            }
            if (w[i].maxdiff > out->maxdiff[i]) out->maxdiff[i] = w[i].maxdiff;
            if (w[i].fd >= 0) rocket_close(w[i].fd);
        }
        pthread_barrier_destroy(&start);
        if (!ok) return 0;
    }

    if (out->overlap_present && ca_stat_overlap(&o1)) out->overlap = o1 - o0;
    out->ms = now_ms() - t0;
    return 1;
}

/* The victim never changes. It is the shape the settled result was measured at. */
static const struct ca_shape ca_victim = { "victim", 256, 512, 2048 };

/* The aggressor, from a handful of granules to the victim's own footprint.
 *
 * The last two pull the two candidate causes apart. Footprint (M*K) and submit duration
 * co-vary in the first five, so a damage figure that rises with footprint could equally
 * be rising with how long the aggressor held something. `wideN` is a 4-granule feature
 * plane behind a large, N-tiled weight cube — small footprint, long occupancy. `flatN`
 * is the victim's own 2048-granule plane against 64 output channels — large footprint,
 * short occupancy. Whichever of the two does more damage names the axis. */
static const struct ca_shape ca_aggressors[] = {
    { "tiny",   4,   64,   64 },
    { "small",  16,  128,  128 },
    { "mid",    64,  256,  256 },
    { "large",  128, 512,  512 },
    { "equal",  256, 512,  2048 },
    { "wideN",  4,   64,   2048 },   /* 4 granules, long submit  */
    { "flatN",  256, 512,  64 },     /* 2048 granules, short submit */
};

int main(int argc, char **argv)
{
    /* This probe exists to put two jobs in flight at once, which is what the
     * library refuses to let a caller build by accident. Lift its own guard. */
    setenv("ROCKET_RK3576_ALLOW_MULTI_FD", "1", 0);
    const float scale = 0.005f;
    unsigned reps = 3;
    int calls = 8;
    int ndev = ca_ndev();
    struct ca_prob pv;
    struct ca_cell c;
    unsigned probe = 0;
    int instrumented;

    if (argc > 1) reps  = (unsigned)atoi(argv[1]);
    if (argc > 2) calls = atoi(argv[2]);

    instrumented = ca_stat_overlap(&probe);

    printf("rk3576 core asymmetry — is the shared resource capacity or a control path?\n");
    printf("bound rocket devices: %d%s\n", ndev,
           ndev < 2 ? "  *** NEED TWO — every cell below is a one-core control ***" : "");
    printf("driver overlap counter: %s\n", instrumented ? "present" : "ABSENT (stock module)");
    printf("victim %dx%dx%d  (%u feature granules, %u KiB weights), reps %u, %d calls each\n\n",
           ca_victim.M, ca_victim.K, ca_victim.N, ca_granules(ca_victim),
           (unsigned)((long long)ca_victim.K * ca_victim.N / 1024), reps, calls);

    if (!ca_prob_init(&pv, ca_victim, 12345u, scale)) {
        fprintf(stderr, "victim golden never settled — cannot measure\n");
        return 1;
    }

    /* The control. One worker, no aggressor: whatever this reports wrong is the floor
     * every other row has to be read against. */
    if (!ca_cell(&pv, NULL, 1, calls, reps, scale, &c)) {
        fprintf(stderr, "control cell failed\n");
        ca_prob_free(&pv);
        return 1;
    }
    printf("%-8s %6s %6s %6s %7s   %8s %7s %7s  %6s %6s\n",
           "aggr", "gran", "wKiB", "ovlp", "aCalls",
           "victim%", "elem%", "dmg/hit", "badRow", "badCol");
    printf("%-8s %6s %6s %6s %7s   %7.1f%% %6.2f%% %7s  %6s %6s\n",
           "none", "-", "-",
           c.overlap_present ? "0" : "-", "-",
           100.0 * c.wrong[0] / (c.calls[0] ? c.calls[0] : 1),
           100.0 * c.elems[0] / (double)(pv.elems * (c.calls[0] ? c.calls[0] : 1)),
           "-", "-", "-");

    for (size_t i = 0; i < sizeof ca_aggressors / sizeof ca_aggressors[0]; i++) {
        struct ca_prob pa;
        struct ca_shape s = ca_aggressors[i];

        if (!ca_prob_init(&pa, s, 999u + (unsigned)i, scale)) {
            printf("%-8s %8u %8u   golden never settled — skipped\n",
                   s.name, ca_granules(s), (unsigned)((long long)s.K * s.N / 1024));
            ca_prob_free(&pa);
            continue;
        }
        if (!ca_cell(&pv, &pa, 2, calls, reps, scale, &c)) {
            printf("%-8s cell failed\n", s.name);
            ca_prob_free(&pa);
            continue;
        }
        {
            double vpct = 100.0 * c.wrong[0] / (c.calls[0] ? c.calls[0] : 1);
            double epct = 100.0 * c.elems[0] /
                          (double)(pv.elems * (c.calls[0] ? c.calls[0] : 1));
            printf("%-8s %6u %6u %6u %7u   %7.1f%% %6.2f%% %6.1f%%  %6.0f %6.0f\n",
                   s.name, ca_granules(s), (unsigned)((long long)s.K * s.N / 1024),
                   c.overlap, c.calls[1], vpct, epct,
                   vpct > 0 ? epct / vpct * 100.0 : 0.0,
                   (double)c.shape_rows, (double)c.shape_cols);
        }
        fflush(stdout);
        ca_prob_free(&pa);
    }

    printf("\ndmg/hit is elem%%/victim%%: the share of the victim's surface destroyed per\n"
           "CORRUPTED call, which is the extent the trigger rate is divided out of.\n"
           "badRow/badCol are the DISTINCT wrong rows and columns of the victim's\n"
           "%d x %d, on the first corrupted call. Few columns over every row is one lost\n"
           "N tile; every column over few rows is an M split; many of both is neither.\n",
           ca_victim.M, ca_victim.N);
    if (!instrumented)
        printf("NOTE: stock module — the overlap column is absent, so a clean row is NOT\n"
               "evidence until re-run against ~/rocketoot_conc.\n");

    ca_prob_free(&pv);
    return 0;
}
