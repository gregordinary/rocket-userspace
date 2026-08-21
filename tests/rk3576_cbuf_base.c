// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_cbuf_base.c — is the CBUF base a real address into a SHARED pool, and does
 * biasing it separate two cores that otherwise corrupt each other?
 *
 * Two jobs executing at once on this part's two cores compute wrong answers on 96-100%
 * of calls. The resource is named — they stage their FEATURE planes into the same CBUF,
 * unconditionally, and what an intruder destroys is however much plane it staged. What
 * is open is the GEOMETRY of the overlap, which is what a fix needs: whether the two
 * cores share a base, and whether anything a userspace encoder emits can move one of
 * them out of the other's way.
 *
 * THE FIELD TO ASK ABOUT IS ALREADY DECODED AND ALREADY EMITTED. The low half of
 * CNA_CBUF_ENTRIES (0x103c) and of CNA_CBUF_CON0 (0x1040) is a GRANULE OFFSET into the
 * CBUF — where a task's window starts and where its fetch resumes. The row-reuse path
 * drives it across a plane's tasks; every plain entry point leaves it at zero, so every
 * job this library submits stages from granule 0. Two cores both starting at granule 0
 * of one physical CBUF collide unconditionally at any size, which is exactly the
 * measured signature: the trigger is not capacity and the extent is the plane.
 *
 * That is also the NVDLA reading. This IP's ancestor allocates CBUF in whole banks by
 * COUNT — CDMA/CSC `D_BANK` is "number of banks allocated for data/weight" — and has no
 * CBUF base register at all; a layer stages implicitly from the bottom and reuse is
 * expressed by releasing slices, not by moving a pointer. The RK3588's `CNA_CBUF_CON0`
 * is that register (DATA_BANK / WEIGHT_BANK / FC_DATA_BANK), and this part's is an
 * allowance in bits[16:27]. Counts, both. The low half is the one thing here that
 * looks like an address. [nvdla.org/hw/v1/ias/programming_guide.html]
 *
 * Nor is there anywhere else to put a partition. The RK3576 TRM's NPU GRF has exactly
 * one CBUF field — `NPU_GRF_MEMGATE_CON0` bits[15:0], "clock gate enable for cbuf bank
 * 0-15", shared, not per core — and no base, mask or window register. So if the base
 * does not separate the cores, no userspace-reachable field does. [TRM,
 * Rockchip_RK3576_TRM_Part1_V1.2]
 *
 * TWO MODES, and the first one gates the second.
 *
 *   extent — SOLO. Bias one job's base and ask whether it still computes. A bias the
 *            hardware honours keeps the task self-consistent (window base and fetch
 *            base move together), so it computes at every bias that stays inside the
 *            pool and FAILS past the end of it. A bias the hardware IGNORES computes
 *            at every value including absurd ones. So the reading is the BREAK POINT,
 *            not the passes: where it breaks is how far the base reaches, and never
 *            breaking means the field is inert on a first task and the lever is dead.
 *
 *   pair   — TWO CORES. Victim at base 0, aggressor biased past the victim's plane.
 *            If the pool is shared and the base addresses it, the collision is now a
 *            choice and the corruption goes to zero. If it does not move, the cores
 *            are not separated by anything an encoder writes.
 *
 * A CELL IS NOT A MEASUREMENT UNTIL IT OVERLAPPED — the driver's stat_overlap counter,
 * printed per cell, and "-" on a stock module. A clean pair cell with zero overlap is a
 * one-core control wearing a different label, and that is the trap this whole line of
 * work has fallen into once already.
 *
 * THE REFERENCE IS A SOLO RUN OF THE PART, NOT A CPU MODEL: a model of the DPU's
 * integer requant disagrees by one count on a fraction of a percent of elements in
 * every configuration. Each golden is taken twice, alone, and required to agree.
 *
 *   ls /sys/bus/platform/drivers/rocket/ | grep npu     # extent needs one core
 *   echo 27708000.npu | sudo tee /sys/bus/platform/drivers/rocket/bind   # pair needs two
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
#include "rocket_rk3576_internal.h"

#define CB_FIRST_BIG_CPU 4
#define CB_STAT_OVERLAP "/sys/module/rocket/parameters/stat_overlap"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Absent on a stock module, which is why presence is carried separately: zero from an
 * absent counter and zero from a real one mean opposite things. */
static int cb_stat_overlap(unsigned *out)
{
    FILE *f = fopen(CB_STAT_OVERLAP, "r");
    if (!f) return 0;
    if (fscanf(f, "%u", out) != 1) *out = 0;
    fclose(f);
    return 1;
}

static int cb_ndev(void)
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

struct cb_shape { const char *name; int M, K, N; };

struct cb_prob {
    struct cb_shape s;
    int8_t *A, *B, *ref, *C;
    size_t elems;
};

static void cb_fill(int8_t *p, size_t n, unsigned seed)
{
    unsigned x = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        p[i] = (int8_t)((x >> 16) & 0xff);
    }
}

/* The granules a shape's feature plane stages: the plane is M, the channels K, and a
 * 64-byte granule holds 64 of those bytes. */
static unsigned cb_granules(struct cb_shape s)
{
    return (unsigned)(((long long)s.M * s.K + 63) / 64);
}

/* Take the reference alone, twice, at bias 0, and require it to agree. A golden run is
 * exposed to the same dropped atoms as everything else. */
static int cb_golden(struct cb_prob *p, float scale)
{
    int8_t *tmp = malloc(p->elems);
    int fd, tries, ok = 0;
    if (!tmp) return 0;
    rocket_rk3576_set_cbuf_bias(0);
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

static int cb_prob_init(struct cb_prob *p, struct cb_shape s, unsigned seed, float scale)
{
    memset(p, 0, sizeof *p);
    p->s = s;
    p->elems = (size_t)s.M * s.N;
    p->A   = malloc((size_t)s.M * s.K);
    p->B   = malloc((size_t)s.N * s.K);
    p->ref = malloc(p->elems);
    p->C   = malloc(p->elems);
    if (!p->A || !p->B || !p->ref || !p->C) return 0;
    cb_fill(p->A, (size_t)s.M * s.K, seed);
    cb_fill(p->B, (size_t)s.N * s.K, seed + 7u);
    return cb_golden(p, scale);
}

static void cb_prob_free(struct cb_prob *p)
{
    free(p->A); free(p->B); free(p->ref); free(p->C);
}

/* Wrong elements in one call, and whether the surface was written at all. */
static unsigned cb_diff(const struct cb_prob *p)
{
    unsigned bad = 0;
    for (size_t i = 0; i < p->elems; i++)
        if (p->C[i] != p->ref[i]) bad++;
    return bad;
}

/* ---------------------------------------------------------------- extent (solo) --
 *
 * The break point is the reading. Biases are granule offsets; the field is 16 bits, so
 * 65535 is the top of it. The landmarks: 4096 is what F=0 buys the data side, 6144 the
 * data-side cap, 7168 this library's measured pool (448 KiB), 8192 the vendor's core-1
 * primary block, 16384 the whole 1 MB the TRM shows. Where it stops computing says
 * which of those the base reaches. */
static const unsigned cb_biases[] = {
    0, 1, 4, 16, 64, 256, 1024, 2048, 3072, 4032, 4096, 4160, 5120, 6144,
    7168, 8192, 10240, 12288, 14336, 16256, 16384, 20480, 32768, 49152, 65024,
};

static int cb_extent(int reps)
{
    struct cb_shape s = { "small", 8, 512, 256 };
    struct cb_prob p;
    unsigned gran = cb_granules(s);
    int fd, rc = 0;

    printf("== extent: does a biased CBUF base still compute, and where does it break?\n");
    printf("   shape %s M=%d K=%d N=%d, feature plane %u granules (%u B)\n\n",
           s.name, s.M, s.K, s.N, gran, gran * 64u);

    if (!cb_prob_init(&p, s, 11u, 0.02f)) {
        printf("   FAIL: no stable golden at bias 0\n");
        return 1;
    }
    fd = rocket_open();
    if (fd < 0) { cb_prob_free(&p); return 1; }

    printf("   %8s  %10s  %7s  %s\n", "bias", "byte off", "wrong", "verdict");
    for (size_t i = 0; i < sizeof cb_biases / sizeof cb_biases[0]; i++) {
        unsigned bias = cb_biases[i];
        unsigned worst = 0;
        int failed_call = 0;
        rocket_rk3576_set_cbuf_bias(bias);
        for (int r = 0; r < reps; r++) {
            unsigned bad;
            memset(p.C, 0, p.elems);
            if (rocket_matmul_int8_rk3576(fd, s.M, s.K, s.N, p.A, p.B, NULL,
                                          0.02f, p.C) != 0) { failed_call = 1; break; }
            bad = cb_diff(&p);
            if (bad > worst) worst = bad;
        }
        printf("   %8u  %10u  %6.2f%%  %s\n", bias, bias * 64u,
               100.0 * worst / (double)p.elems,
               failed_call ? "CALL FAILED" : worst ? "WRONG" : "exact");
    }
    rocket_rk3576_set_cbuf_bias(0);
    rocket_close(fd);
    cb_prob_free(&p);
    printf("\n   Reading: the BREAK POINT, not the passes. Exact everywhere including\n"
           "   absurd biases means the field is inert on a first task and no bias can\n"
           "   separate two cores; a break says how far the base reaches.\n");
    return rc;
}

/* ------------------------------------------------------------------------ bound --
 *
 * Where the base stops working, as a function of the PLANE. The landmark sweep breaks
 * between 4032 and 4096 on a 64-granule plane, and 4032 + 64 is exactly the 4096
 * granules F=0 buys the data side. If that is the rule rather than a coincidence, the
 * last working base is `4096 + F - plane` at every plane — the base addresses a task's
 * own ALLOWANCE WINDOW, not the physical megabyte, and a job cannot be staged outside
 * the budget it programmed. Test the prediction at four planes instead of fitting the
 * tail of one sweep, which is intermittent enough to overfit. */
static int cb_bound(int reps)
{
    static const struct cb_shape shapes[] = {
        { "64gr",     8,  512,  256 },   /*   64 granules */
        { "256gr",   32,  512,  256 },   /*  256 granules */
        { "1024gr", 128,  512,  256 },   /* 1024 granules */
        { "2048gr", 256,  512,  256 },   /* 2048 granules */
    };
    int rc = 0;

    printf("== bound: is the last working base 4096+F - plane, at every plane?\n");
    printf("   (the base addresses the task's own allowance window, not the CBUF)\n\n");

    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        struct cb_prob p;
        unsigned gran = cb_granules(shapes[s]);
        unsigned pred = 4096u > gran ? 4096u - gran : 0u;
        /* Straddle the prediction, and carry a 0 control so a shape that is simply
         * broken is not read as a boundary. */
        const int off[] = { -1000000, -256, -64, 0, 64, 256 };
        int fd;

        if (!cb_prob_init(&p, shapes[s], 11u, 0.02f)) {
            printf("   %-8s FAIL: no stable golden at bias 0\n", shapes[s].name);
            rc = 1;
            continue;
        }
        fd = rocket_open();
        if (fd < 0) { cb_prob_free(&p); return 1; }
        printf("   %-8s plane %4u granules, predicted last working base %u\n",
               shapes[s].name, gran, pred);
        for (size_t i = 0; i < sizeof off / sizeof off[0]; i++) {
            unsigned bias = off[i] == -1000000 ? 0u
                          : (int)pred + off[i] < 0 ? 0u : (unsigned)((int)pred + off[i]);
            unsigned worst = 0;
            int failed_call = 0;
            rocket_rk3576_set_cbuf_bias(bias);
            for (int r = 0; r < reps; r++) {
                unsigned bad;
                memset(p.C, 0, p.elems);
                if (rocket_matmul_int8_rk3576(fd, shapes[s].M, shapes[s].K, shapes[s].N,
                                              p.A, p.B, NULL, 0.02f, p.C) != 0) {
                    failed_call = 1; break;
                }
                bad = cb_diff(&p);
                if (bad > worst) worst = bad;
            }
            printf("       base %6u  %-10s %6.2f%%  %s\n", bias,
                   off[i] == -1000000 ? "(control)"
                                      : (off[i] == 0 ? "= pred" : "vs pred"),
                   100.0 * worst / (double)p.elems,
                   failed_call ? "CALL FAILED" : worst ? "WRONG" : "exact");
        }
        rocket_rk3576_set_cbuf_bias(0);
        rocket_close(fd);
        cb_prob_free(&p);
        printf("\n");
    }
    printf("   Reading: exact at the prediction and WRONG just past it, at every plane,\n"
           "   means the base is an offset inside the task's own allowance — so it can\n"
           "   only separate jobs whose planes together fit one allowance window.\n");
    return rc;
}

/* ------------------------------------------------------------------------- bits --
 *
 * What is left of CNA_CBUF_CON0 after the allowance and the base. The emitted word is
 * `bit28 | F<<16 | base`, so on a shape whose plane fits F=0 at base 0 it is exactly
 * 0x10000000 and bits[31:29] are the only ones this library never varies. The RK3588's
 * same offset carries bank SELECTS, so if a bank base survives anywhere on this part it
 * is here.
 *
 * Solo, and pass/fail only. A bit that CHANGES the answer is live and worth a pair
 * cell; a bit that changes nothing is either reserved or a select whose other value is
 * equivalent, and neither is a partition. Overriding the whole word is safe here
 * precisely because the shape's natural word is known. */
static int cb_bits(int reps)
{
    struct cb_shape s = { "small", 8, 512, 256 };
    struct cb_prob p;
    int fd, rc = 0;

    printf("== bits: is anything left in CNA_CBUF_CON0 above the allowance?\n");
    printf("   shape M=%d K=%d N=%d, natural word 0x10000000 (F=0, base 0)\n\n",
           s.M, s.K, s.N);

    if (!cb_prob_init(&p, s, 11u, 0.02f)) {
        printf("   FAIL: no stable golden\n");
        return 1;
    }
    fd = rocket_open();
    if (fd < 0) { cb_prob_free(&p); return 1; }

    printf("   %-12s %8s  %s\n", "0x1040", "wrong", "verdict");
    for (unsigned combo = 0; combo < 8u; combo++) {
        uint32_t word = 0x10000000u | (combo << 29);
        char spec[64];
        unsigned worst = 0;
        int failed_call = 0;
        snprintf(spec, sizeof spec, "0x1040=0x%08x", word);
        setenv("ROCKET_RK3576_SET", spec, 1);
        for (int r = 0; r < reps; r++) {
            unsigned bad;
            memset(p.C, 0, p.elems);
            if (rocket_matmul_int8_rk3576(fd, s.M, s.K, s.N, p.A, p.B, NULL,
                                          0.02f, p.C) != 0) { failed_call = 1; break; }
            bad = cb_diff(&p);
            if (bad > worst) worst = bad;
        }
        printf("   0x%08x  %7.2f%%  %s%s\n", word,
               100.0 * worst / (double)p.elems,
               failed_call ? "CALL FAILED" : worst ? "CHANGED — live" : "no change",
               combo == 0 ? "  (the natural word)" : "");
    }
    unsetenv("ROCKET_RK3576_SET");
    rocket_close(fd);
    cb_prob_free(&p);
    printf("\n   Reading: a bit that changes nothing is not a partition. Even a live one\n"
           "   would have to survive the pair cell, which placement already failed.\n");
    return rc;
}

/* ------------------------------------------------------------------ pair (2 cores) --
 *
 * The aggressor drives CONTINUOUSLY until the victim is done. A fixed call count lets a
 * small shape finish inside the victim's host-side packing, which is most of a large
 * matmul's wall on this part, and the cell then records zero overlap. */
struct cb_worker {
    pthread_t thread;
    int fd, cpu, calls, rc, continuous;
    unsigned bias;
    float scale;
    struct cb_prob *p;
    pthread_barrier_t *start;
    volatile int *done;
    unsigned calls_run, calls_wrong, elems_wrong;
};

static void *cb_run(void *arg)
{
    struct cb_worker *w = arg;
    struct cb_prob *p = w->p;

    if (w->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(w->cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
    /* Per THREAD, which is the whole reason the knob is not just an environment
     * variable: the two workers must stage at different bases in one process. */
    rocket_rk3576_set_cbuf_bias(w->bias);
    pthread_barrier_wait(w->start);
    for (int c = 0; w->continuous ? !*w->done : c < w->calls; c++) {
        unsigned bad;
        memset(p->C, 0, p->elems);
        w->rc = rocket_matmul_int8_rk3576(w->fd, p->s.M, p->s.K, p->s.N,
                                          p->A, p->B, NULL, w->scale, p->C);
        if (w->rc != 0) break;
        w->calls_run++;
        bad = cb_diff(p);
        if (bad) { w->calls_wrong++; w->elems_wrong += bad; }
    }
    if (!w->continuous && w->done) *w->done = 1;
    return NULL;
}

struct cb_cell {
    unsigned calls[2], wrong[2], elems[2];
    unsigned overlap;
    int overlap_present;
    double ms;
};

static int cb_cell_run(struct cb_prob *pv, struct cb_prob *pa,
                       unsigned bias_v, unsigned bias_a,
                       int calls, unsigned reps, float scale, struct cb_cell *out)
{
    unsigned o0 = 0, o1 = 0;
    double t0 = now_ms();

    memset(out, 0, sizeof *out);
    out->overlap_present = cb_stat_overlap(&o0);

    for (unsigned r = 0; r < reps; r++) {
        struct cb_worker w[2];
        pthread_barrier_t start;
        volatile int done = 0;

        pthread_barrier_init(&start, NULL, 2);
        memset(w, 0, sizeof w);
        for (int i = 0; i < 2; i++) {
            w[i].cpu        = CB_FIRST_BIG_CPU + i;
            w[i].calls      = calls;
            w[i].scale      = scale;
            w[i].start      = &start;
            w[i].done       = &done;
            w[i].p          = i ? pa : pv;
            w[i].bias       = i ? bias_a : bias_v;
            w[i].continuous = i;              /* the aggressor drives until told to stop */
            w[i].fd         = rocket_open();
            if (w[i].fd < 0) {
                for (int j = 0; j <= i; j++) if (w[j].fd >= 0) rocket_close(w[j].fd);
                pthread_barrier_destroy(&start);
                return -1;
            }
        }
        for (int i = 0; i < 2; i++) pthread_create(&w[i].thread, NULL, cb_run, &w[i]);
        for (int i = 0; i < 2; i++) pthread_join(w[i].thread, NULL);
        for (int i = 0; i < 2; i++) {
            out->calls[i] += w[i].calls_run;
            out->wrong[i] += w[i].calls_wrong;
            out->elems[i] += w[i].elems_wrong;
            rocket_close(w[i].fd);
        }
        pthread_barrier_destroy(&start);
    }
    if (out->overlap_present && cb_stat_overlap(&o1)) out->overlap = o1 - o0;
    out->ms = now_ms() - t0;
    return 0;
}

static int cb_pair(int calls, unsigned reps)
{
    /* The victim never changes: the shape whose 2048-granule plane every earlier cell
     * scored. The aggressor is the TINY one, whose four granules still corrupt three
     * quarters of the victim's calls — the cell where a capacity story is already dead,
     * so anything that fixes it fixed a collision and not a shortage. */
    struct cb_shape sv = { "victim", 256, 512, 2048 };
    struct cb_shape sa = { "tiny",     8,  32,   64 };
    struct cb_prob pv, pa;
    unsigned gv = cb_granules(sv), ga = cb_granules(sa);
    /* Where to put the aggressor. 0 is the control — today's behaviour, both staging
     * from the bottom. 1024 still lands inside the victim's 2048-granule plane and is
     * the negative control: if a bias that does NOT clear the victim also stops the
     * corruption, the mechanism is not placement. 2048 is the first base that clears
     * it, and 3072/4032 are clear of it with room to spare.
     *
     * The base only reaches inside the task's OWN allowance: the last working base is
     * 4096 + F - plane at every plane measured, and past it a task corrupts itself.
     * See cb_bound(). Every scored row therefore stays under the aggressor's bound of
     * 4092, so a broken intruder never confounds the cell.
     *
     * The last row is a POSITIVE CONTROL and is meant to fail: 4156 is past the
     * aggressor's own bound of 4092, so its own surface must go wrong. Without it a
     * clean negative is unreadable — "biasing changed nothing" and "the bias never
     * reached the emitter on this thread" produce the same table. */
    static const unsigned biases[] = { 0, 1024, 2048, 3072, 4032, 4156 };
    int rc = 0;

    if (cb_ndev() < 2) {
        printf("SKIP: %d rocket device(s) bound; the pair cell needs BOTH cores or "
               "every row is a one-core control wearing a different label.\n"
               "  echo 27708000.npu | sudo tee /sys/bus/platform/drivers/rocket/bind\n",
               cb_ndev());
        return 2;
    }

    printf("== pair: does biasing the aggressor's CBUF base stop it corrupting the victim?\n");
    printf("   victim  M=%d K=%d N=%d, %u granules, base 0 throughout\n",
           sv.M, sv.K, sv.N, gv);
    printf("   aggress M=%d K=%d N=%d, %u granules, base swept\n\n",
           sa.M, sa.K, sa.N, ga);

    if (!cb_prob_init(&pv, sv, 11u, 0.02f)) { printf("   FAIL: victim golden\n"); return 1; }
    if (!cb_prob_init(&pa, sa, 29u, 0.02f)) { printf("   FAIL: aggressor golden\n");
                                              cb_prob_free(&pv); return 1; }

    printf("   %10s  %8s  %14s  %14s  %8s\n",
           "aggr base", "byte off", "victim wrong", "aggr wrong", "overlap");
    for (size_t i = 0; i < sizeof biases / sizeof biases[0]; i++) {
        struct cb_cell c;
        if (cb_cell_run(&pv, &pa, 0u, biases[i], calls, reps, 0.02f, &c) != 0) {
            rc = 1; break;
        }
        {
            char ov[16];
            if (c.overlap_present) snprintf(ov, sizeof ov, "%u", c.overlap);
            else                   snprintf(ov, sizeof ov, "-");
            printf("   %10u  %8u  %5u/%-5u %4.0f%%  %5u/%-5u %4.0f%%  %8s\n",
                   biases[i], biases[i] * 64u,
                   c.wrong[0], c.calls[0],
                   c.calls[0] ? 100.0 * c.wrong[0] / c.calls[0] : 0.0,
                   c.wrong[1], c.calls[1],
                   c.calls[1] ? 100.0 * c.wrong[1] / c.calls[1] : 0.0,
                   ov);
        }
        if (c.overlap_present && !c.overlap)
            printf("     ^ ZERO OVERLAP — not a measurement. The two jobs never ran "
                   "together.\n");
    }
    rocket_rk3576_set_cbuf_bias(0);
    cb_prob_free(&pv);
    cb_prob_free(&pa);
    printf("\n   Reading: the base-0 row is the control and should reproduce the known\n"
           "   71.9-75.6%%. A biased row at zero wrong, with non-zero overlap, is a\n"
           "   partition — and one a userspace encoder can express.\n");
    return rc;
}

int main(int argc, char **argv)
{
    /* This probe exists to put two jobs in flight at once, which is what the
     * library refuses to let a caller build by accident. Lift its own guard. */
    setenv("ROCKET_RK3576_ALLOW_MULTI_FD", "1", 0);
    const char *mode = argc > 1 ? argv[1] : "extent";
    int reps = argc > 2 ? atoi(argv[2]) : 3;

    if (cb_ndev() < 1) {
        printf("SKIP: no rocket device bound\n");
        return 2;
    }
    if (!strcmp(mode, "extent")) return cb_extent(reps);
    if (!strcmp(mode, "bound"))  return cb_bound(reps);
    if (!strcmp(mode, "bits"))   return cb_bits(reps);
    if (!strcmp(mode, "pair"))   return cb_pair(20, (unsigned)reps);
    if (!strcmp(mode, "all")) {
        int a = cb_extent(reps);
        int b = cb_bound(reps);
        int c = cb_bits(reps);
        int d = cb_pair(20, (unsigned)reps);
        return a ? a : b ? b : c ? c : (d == 2 ? 0 : d);
    }
    fprintf(stderr, "usage: %s [extent|bound|bits|pair|all] [reps]\n", argv[0]);
    return 1;
}
