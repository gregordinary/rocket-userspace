// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_i32_height_bound.c — the wide int32 writer's surface bound.
 *
 * The DPU's 32-bit writer at PROC_PRECISION int32 delivers the first eight output
 * channels of every sixteen, and `rocket_rk3576_i32_wide_word()` is the map that says
 * where each one lands. That map holds only while a row task's output surface stays
 * under **8 KiB**, which in the emitter's terms is
 *
 *     ow * oh_full * oc_prog  <  4096
 *
 * — the surface is two bytes per unit of that product. Past it a tile comes back PART
 * RIGHT and silently, which is why the planner splits row tasks to stay inside it.
 *
 * The bound is measured, not fitted. Swept at K=32 with the cap lifted, the first wrong
 * height is the smallest `oh_full` with `oh_full * oc_prog >= 4096` at every one of the
 * eight programmed counts 64/128/192/256/320/384/448/512 — heights 64, 32, 22, 16, 13,
 * 11, 10 and 8. Three things make it read as a rule that mispredicts, and this probe
 * exists to keep them apart:
 *
 *   - STATE IT IN THE PROGRAMMED CHANNEL COUNT. `oc_prog` is twice the real `oc` on this
 *     path, so the same bound written in `oc` is out by a factor of two and reports
 *     failures "inside the rule". The two cannot be separated by sweeping N, because
 *     every N the entry accepts is a multiple of 32 and `oc_prog` is then always 2*N.
 *   - SWEEP MORE THAN ONE K. K only ever relaxes the bound: oc 128 is wrong from oh 16
 *     at K <= 128, wrong at only 31, 32 and 40 at K = 512, and exact to oh 40 at
 *     K >= 1024. 8 KiB is the worst-case floor, and a mid-range K sweep finds exactness
 *     far past it.
 *   - `ow` IS ALREADY IN `A`. Doubling ow at a fixed A is the same surface, so ow 2 and
 *     ow 4 are exact across the range in which ow 1 fails.
 *
 * Every wrong element comes back ZERO — never another element's value, never anything
 * else — so the failure is an empty atom rather than a misplaced one, and the probe
 * reports each in the writer's own emission coordinates to keep that visible.
 *
 * It drives the ordinary library entry with two probe knobs: ROCKET_RK3576_MM_IW forces
 * the plane, and ROCKET_RK3576_I32_WIDE_SURF raises the surface cap out of the way so the
 * bound can be reached at all. Both are set here rather than by the caller.
 *
 *   sudo -E ./build/rk3576_i32_height_bound              # the standard channel sweep
 *   sudo -E ROCKET_HB_OC=192 ./build/rk3576_i32_height_bound
 *   sudo -E ROCKET_HB_LIFT=0 ... ./build/rk3576_i32_height_bound   # as a CALLER gets it
 *
 * Env knobs: ROCKET_HB_K contraction depth (512), ROCKET_HB_NS a comma-separated N list,
 * ROCKET_HB_OC / ROCKET_HB_OW pick one row, ROCKET_HB_OHMIN / ROCKET_HB_OHMAX the height
 * walk, ROCKET_HB_SHOW how many wrong elements to name (12), ROCKET_HB_MAP=0 to drop the
 * coordinate report, ROCKET_HB_LIFT=0 to leave the shipped cap in force.
 *
 * The coordinate report assumes the whole surface is ONE row task, which is true with
 * the cap lifted and false without it — at ROCKET_HB_LIFT=0 read the counts and the
 * values, not the atom numbers.
 *
 * This is a PROBE, not a gate: it reports the boundary it finds and exits 0 unless the
 * device could not be opened. A refused shape is reported as refused, not as exact.
 *
 * Run with `sudo -E`, and on a quiet NPU: the wide writer poisons the next submit, so
 * the entry pays a power cycle per call and a run of this takes minutes.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int env_int(const char *k, int dflt)
{
    const char *e = getenv(k);
    return (e && *e) ? (int)strtol(e, NULL, 0) : dflt;
}

static int hb_k(void) { return env_int("ROCKET_HB_K", 512); }

/* ---------------------------------------------------------------------------
 * The writer's own coordinates.
 *
 * A wrong element read in (m, n) says nothing about what selects it — the map from
 * (m, n) to the surface is not order-preserving in either axis, so a rule that is
 * simple in the writer's emission order looks scattered in the caller's. These
 * mirror the library's scatter and rocket_rk3576_i32_wide_word() so a failing shape
 * can be reported where the rule would live.
 *
 * With the row cap lifted the whole surface is ONE task, so the task's pixel index is
 * the caller's row and A is the whole plane.
 * ------------------------------------------------------------------------- */
struct atom_coord {
    unsigned c;     /* the PROGRAMMED output channel the real one was scattered into */
    unsigned sg;    /* 32-channel super-group                                        */
    unsigned s;     /* stream position 2*pixel + block                               */
    unsigned L;     /* lane group, the slower axis of the run                        */
    unsigned e;     /* emission index 2*s + L — the order the writer actually emits   */
    unsigned atom;  /* atom index into the surface                                   */
    unsigned run;   /* which run of A atoms inside the super-group                   */
    unsigned off;   /* offset inside that run                                        */
};

static unsigned hb_prog_oc(unsigned n) { return 16u * (n / 8u) + (n % 8u); }
static unsigned hb_pad_oc(unsigned oc) { return ((oc + 31u) / 32u) * 32u; }

static void hb_coord(unsigned A, unsigned m, unsigned n, struct atom_coord *o)
{
    unsigned c = hb_prog_oc(n);
    unsigned s = 2u * m + ((c % 32u) / 16u);
    o->c    = c;
    o->sg   = c / 32u;
    o->s    = s;
    o->L    = (c % 16u) / 4u;
    o->e    = 2u * s + o->L;
    o->run  = 2u * (s / A) + o->L;
    o->off  = s % A;
    o->atom = 4u * A * o->sg + A * o->run + o->off;
}

/* The CPU model. int8 operands, int32 accumulators, no requant on this path. */
static void hb_reference(int M, int K, int N, const int8_t *A, const int8_t *B,
                         int32_t *C)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[(size_t)m * K + k] * B[(size_t)n * K + k];
            C[(size_t)m * N + n] = acc;
        }
}

/* 1 if the entry reproduced the model exactly, 0 if it did not, -1 if it refused.
 * `bad` receives the number of wrong elements. */
static int hb_run(int fd, int M, int K, int N, unsigned iw, unsigned *bad)
{
    int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K);
    int32_t *C = malloc((size_t)M * N * sizeof *C);
    int32_t *R = malloc((size_t)M * N * sizeof *R);
    char buf[32];
    int rc, ok = 0;

    *bad = 0;
    if (!A || !B || !C || !R) { free(A); free(B); free(C); free(R); return -1; }
    /* A fixed pattern rather than random: the question is a map, and a reproducible
     * input makes a partial surface comparable between runs. */
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)((i * 7u + 3u) % 61u) - 30;
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (int8_t)((i * 11u + 5u) % 59u) - 29;
    memset(C, 0, (size_t)M * N * sizeof *C);

    snprintf(buf, sizeof buf, "%u", iw);
    setenv("ROCKET_RK3576_MM_IW", buf, 1);
    rc = rocket_matmul_int8_rk3576_i32(fd, M, K, N, A, B, NULL, C);
    unsetenv("ROCKET_RK3576_MM_IW");

    if (rc == 0) {
        hb_reference(M, K, N, A, B, R);
        for (size_t i = 0; i < (size_t)M * N; i++)
            if (C[i] != R[i]) (*bad)++;
        ok = (*bad == 0);
        /* A wrong shape is worth more than its count. Report every wrong element in
         * the writer's own coordinates, and say whether the value that arrived is a
         * ZERO (a write that did not land) or another element's correct value (an
         * address-generation slip, which names a rule). */
        if (!ok && env_int("ROCKET_HB_MAP", 1)) {
            unsigned surfA = (unsigned)(M / (int)iw) * iw;
            unsigned shown = 0, nzero = 0, nalias = 0, nother = 0;
            unsigned emin = ~0u, emax = 0, sgmask = 0;
            unsigned cap = (unsigned)env_int("ROCKET_HB_SHOW", 12);
            printf("      wrong in the writer's coordinates (A=%u, 4A=%u emissions "
                   "per super-group, oc_prog=%u):\n", surfA, 4u * surfA,
                   hb_pad_oc(2u * (unsigned)N));
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++) {
                    size_t i = (size_t)m * N + n;
                    struct atom_coord co;
                    long alias_m = -1, alias_n = -1;
                    if (C[i] == R[i]) continue;
                    hb_coord(surfA, (unsigned)m, (unsigned)n, &co);
                    if (co.e < emin) emin = co.e;
                    if (co.e > emax) emax = co.e;
                    sgmask |= 1u << (co.sg & 31u);
                    if (C[i] == 0) nzero++;
                    else {
                        for (size_t j = 0; j < (size_t)M * N && alias_m < 0; j++)
                            if (R[j] == C[i]) { alias_m = (long)(j / N); alias_n = (long)(j % N); }
                        if (alias_m >= 0) nalias++; else nother++;
                    }
                    if (shown < cap) {
                        printf("        m=%-3d n=%-4d c=%-4u sg=%u run=%-3u off=%-3u "
                               "L=%u s=%-4u e=%-5u atom=%-6u  got=%-11d want=%-11d",
                               m, n, co.c, co.sg, co.run, co.off, co.L, co.s, co.e,
                               co.atom, C[i], R[i]);
                        if (C[i] == 0) printf("  ZERO\n");
                        else if (alias_m >= 0) {
                            struct atom_coord ac;
                            hb_coord(surfA, (unsigned)alias_m, (unsigned)alias_n, &ac);
                            printf("  = (m=%ld,n=%ld) de=%+d datom=%+d\n", alias_m, alias_n,
                                   (int)ac.e - (int)co.e, (int)ac.atom - (int)co.atom);
                        } else printf("  unexplained\n");
                        shown++;
                    }
                }
            printf("      -> %u zero, %u alias another element, %u neither | "
                   "emissions %u..%u of %u | super-groups 0x%x\n",
                   nzero, nalias, nother, emin, emax, 4u * surfA, sgmask);
        }
    } else {
        ok = -1;
    }
    free(A); free(B); free(C); free(R);
    return ok;
}

/* Walk oh up at a fixed (oc, ow) and report the first height that is not exact. */
static void hb_sweep(int fd, unsigned oc, unsigned ow)
{
    /* The bound under test predicts a break near 4096/oc rows; go half again past it so
     * a plane-width term (which would break at a quarter of that at ow=4) is inside the
     * walk too. */
    unsigned predicted = 4096u / oc;
    unsigned oh_max = (unsigned)env_int("ROCKET_HB_OHMAX", 0);
    unsigned oh_min = (unsigned)env_int("ROCKET_HB_OHMIN", 1);
    if (!oh_max) oh_max = predicted + predicted / 2u + 2u;
    unsigned oh, first_bad = 0, worst = 0;

    printf("  oc=%-4u (oc_prog=%-4u) ow=%-2u K=%-5d  rows-only bound predicts a break "
           "at oh=%u\n", oc, hb_pad_oc(2u * oc), ow, hb_k(), predicted);

    unsigned nrun = 0, nbadh = 0, nref = 0;

    for (oh = oh_min; oh <= oh_max; oh++) {
        unsigned bad = 0;
        int r = hb_run(fd, (int)(ow * oh), hb_k(), (int)oc, ow, &bad);
        if (r < 0) { nref++; continue; }
        nrun++;
        if (!r) {
            if (!first_bad) first_bad = oh;
            if (bad > worst) worst = bad;
            nbadh++;
            printf("    oh=%-3u M=%-5u WRONG  %u of %u elements\n",
                   oh, ow * oh, bad, ow * oh * oc);
        }
    }
    /* A refused shape is not an exact one. Reporting the two together is how a sweep
     * over an N the entry will not take reads as a clean result. */
    if (nref)
        printf("    -> %u of %u heights REFUSED (N must be a multiple of 32, so the "
               "programmed oc is always 2*oc here and the two cannot be separated)\n",
               nref, nref + nrun);
    if (!nrun)
        printf("    -> nothing ran\n");
    else if (first_bad)
        printf("    -> %u of %u heights wrong, first oh=%u (oh*oc=%u), worst %u elements\n",
               nbadh, nrun, first_bad, first_bad * oc, worst);
    else
        printf("    -> exact at all %u heights run, to oh=%u (oh*oc=%u)\n",
               nrun, oh_max, oh_max * oc);
}

int main(void)
{
    int fd = rocket_open();
    unsigned oc_only = (unsigned)env_int("ROCKET_HB_OC", 0);
    unsigned ow_only = (unsigned)env_int("ROCKET_HB_OW", 0);
    /* Every N the entry accepts is a multiple of 32, so the programmed channel count is
     * always exactly 2*N and no shape here can tell a rule over one from a rule over the
     * other. Separating them needs the raw emitter, where the programmed count is free.
     * What this list is for is the MAP: one common height range across every N, so the
     * failing heights are comparable between rows. */
    static const unsigned OCS[] = { 32u, 64u, 96u, 128u, 160u, 192u, 224u, 256u };
    static const unsigned OWS[] = { 1u, 2u, 4u };
    unsigned ocs_env[16];
    unsigned nocs = sizeof OCS / sizeof OCS[0];
    const unsigned *ocs = OCS;
    const char *nlist = getenv("ROCKET_HB_NS");

    if (nlist && *nlist) {
        char *copy = strdup(nlist), *tok, *save = NULL;
        nocs = 0;
        for (tok = strtok_r(copy, ",", &save); tok && nocs < 16;
             tok = strtok_r(NULL, ",", &save))
            ocs_env[nocs++] = (unsigned)strtoul(tok, NULL, 0);
        free(copy);
        ocs = ocs_env;
    }

    if (fd < 0) {
        fprintf(stderr, "could not open the NPU (%d)\n", fd);
        return 1;
    }
    /* Raise the planner's row cap out of the way: the point is to reach the boundary,
     * which the library exists to stay inside of. ROCKET_HB_LIFT=0 leaves the shipped
     * cap in force instead, which is how a shape is checked as a CALLER would get it
     * rather than as the boundary sweep needs it. */
    if (env_int("ROCKET_HB_LIFT", 1))
        setenv("ROCKET_RK3576_I32_WIDE_SURF", "1000000", 1);
    /* And force the WIDE writer, so the narrow one's map is not what is being read. */
    setenv("ROCKET_RK3576_I32_OC_MULT", "2", 1);

    printf("== rk3576 wide int32 writer: what is in the height bound ==\n");
    printf("   K=%d, the row cap lifted, oc_mult forced to 2 (the wide writer)\n\n",
           hb_k());

    for (unsigned i = 0; i < nocs; i++) {
        if (oc_only && ocs[i] != oc_only) continue;
        for (unsigned j = 0; j < sizeof OWS / sizeof OWS[0]; j++) {
            if (ow_only && OWS[j] != ow_only) continue;
            hb_sweep(fd, ocs[i], OWS[j]);
        }
        printf("\n");
    }

    printf("== the bound is 8 KiB of surface per row task: ow*oh_full*oc_prog < 4096.\n");
    printf("   The first wrong height should be the smallest oh with oh*oc_prog >= 4096,\n");
    printf("   and only at a K short enough to reach it — K >= 1024 hides it entirely ==\n");
    rocket_close(fd);
    return 0;
}
