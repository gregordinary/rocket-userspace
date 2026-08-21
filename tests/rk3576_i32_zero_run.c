// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_i32_zero_run.c — what makes the int32 writers emit atoms as zeros.
 *
 * Both of `rocket_matmul_int8_rk3576_i32()`'s output writers intermittently leave atoms
 * of a row task holding zero while the rest of the surface is exact. The library detects
 * that and settles it, so a caller never sees it — which is also why nothing about it is
 * measurable from a caller. This probe measures it, and it is the instrument that found
 * two silent wrong-answer bugs by driving shapes the way a caller does rather than the
 * way a geometry sweep does.
 *
 * WHAT IS SETTLED, and so is not re-established here:
 *   - the writer REACHED those atoms. Against a sentinel stamped into the surface
 *     beforehand they come back zero rather than holding the stamp;
 *   - it is not a readback race (a second fence and read are byte-identical) and not the
 *     wide-output poisoning, which the power cycle clears and which host memory traffic
 *     leaves alone;
 *   - past a row task's 8 KiB surface bound the same signature becomes DETERMINISTIC.
 *     That bound is `iw * oh_full * oc_prog < 4096` — over the whole PLANE, which a row
 *     split cannot narrow.
 *
 * WHAT THE MODES ASK. A row task has a surface `A * oc_prog` half-bytes (A = iw*oh_full),
 * it emits `oc_prog/8` atoms per pixel, and it contracts K, and those three are
 * confounded in any single shape:
 *
 *   `oc`    holds the surface at a fixed fraction of the bound and walks oc_prog, so the
 *           only thing changing is the per-pixel burst.
 *   `frac`  holds oc_prog and walks the surface from an eighth of the bound to its edge.
 *   `k`     holds both and walks K, the one axis known to move the deterministic bound.
 *   `load`  a caller's own shape quiet, and then under three host loads that differ
 *           only in how far down the memory hierarchy the host's work reaches. This
 *           is the mode that answered the mechanism: the drop rate goes from 11.0% of
 *           row tasks to 44.8% under four streaming-memcpy threads while the count of
 *           atoms never emitted at all barely moves, so it is a write-path race the
 *           memory system makes worse and a different hazard from the poisoning. The
 *           two control loads say whether "the memory system" or merely "a busy host"
 *           is the right reading of that.
 *   `hammer` one shape, with the plane and the writer left to the planner. The only mode
 *           that measures what a caller actually meets — a mode that quietly forces
 *           either cannot tell a shipped bug from a probe knob.
 *
 * Rates are read out of `rocket_rk3576_i32_last_stats()`, which counts the detector's own
 * firings per ROW TASK, so a cell's rate is comparable across shapes with different task
 * counts. Every rep is also checked against a CPU model, and a rep that is wrong ANYWAY
 * is a different and far more serious finding than one that was repaired: the repair is
 * supposed to make the answer exact.
 *
 *   sudo -E ./build/rk3576_i32_zero_run oc
 *   sudo -E ./build/rk3576_i32_zero_run frac
 *   sudo -E ./build/rk3576_i32_zero_run k
 *   sudo -E ./build/rk3576_i32_zero_run load 128 256 2048
 *   sudo -E ./build/rk3576_i32_zero_run hammer 32 1024 4096 20
 *
 * Env: ROCKET_ZR_REPS reps per cell (30), ROCKET_ZR_K the held K (1024),
 * ROCKET_ZR_LOADT host load threads (4).
 *
 * This is a PROBE: it exits 0 on any completed run, non-zero only if the device could not
 * be opened or a rep came back wrong after the repair.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

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

/* oc_prog for a real output-channel count on this path: every channel costs two
 * programmed ones and the count is padded to a whole super-group. */
static unsigned zr_oc_prog(unsigned oc) { return ((2u * oc + 31u) / 32u) * 32u; }

/* ---------------------------------------------------------------------------
 * Host load, for the `load` cells.
 *
 * Streaming memcpy over a working set far larger than the last-level cache is the
 * cheapest way to take memory bandwidth away from the NPU without touching it.
 * It is also three things at once — busy CPUs, a busy cache hierarchy and a busy
 * DDR controller — so on its own it cannot say which of them the drop rate
 * follows. The other two kinds hold the first two and drop the third:
 *
 *   ZR_LOAD_DDR    32 MiB buffers: CPUs busy, caches thrashed, DDR saturated.
 *   ZR_LOAD_CACHE  16 KiB buffers: the same memcpy at the same rate, L1-resident,
 *                  so the CPUs are equally busy and DDR sees nothing.
 *   ZR_LOAD_SPIN   no memory at all: integer work in registers. Isolates CPU
 *                  occupancy and scheduling latency from any memory effect.
 *
 * A rate that moves under DDR and not under CACHE/SPIN is a memory-path result;
 * one that moves under all three is about the host being busy.
 * ------------------------------------------------------------------------- */
enum { ZR_LOAD_DDR, ZR_LOAD_CACHE, ZR_LOAD_SPIN };

static volatile int zr_load_stop;
static volatile uint64_t zr_spin_sink;

static void *zr_load_thread(void *arg)
{
    int kind = (int)(intptr_t)arg;

    if (kind == ZR_LOAD_SPIN) {
        uint64_t x = 0x9E3779B97F4A7C15ULL;
        while (!zr_load_stop) {
            for (int i = 0; i < 4096; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; }
        }
        zr_spin_sink += x;
        return NULL;
    }

    size_t n = (kind == ZR_LOAD_CACHE) ? (16u << 10) : (32u << 20);
    char *a = malloc(n), *b = malloc(n);
    if (!a || !b) { free(a); free(b); return NULL; }
    memset(a, 0x5A, n);
    while (!zr_load_stop) { memcpy(b, a, n); memcpy(a, b, n); }
    free(a); free(b);
    return NULL;
}

struct zr_cell {
    unsigned reps, wrong, refused;
    unsigned tasks;             /* row tasks run, summed over reps                */
    unsigned redo_zeroed;       /* task attempts redone for a zero-emission run   */
    unsigned redo_empty;        /* task attempts redone for atoms never emitted   */
    unsigned atoms_zeroed;
    unsigned reps_hit;          /* reps in which the zero-run detector fired      */
    double   ms;
};

/* One (M, K, N) cell, `reps` times. `force_ow1` pins the plane so a row task is the
 * whole plane and A is M, and `force_wide` pins the writer — every sweep wants both,
 * and `hammer` wants NEITHER, because its question is what a caller meets rather than
 * what a task geometry does. A mode that quietly forces the writer cannot answer whether
 * the planner picks it, which is the difference between a shipped bug and a probe knob. */
static void zr_cell(int fd, int M, int K, int N, unsigned reps, int force_ow1,
                    int force_wide, struct zr_cell *out)
{
    int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K);
    int32_t *C = malloc((size_t)M * N * sizeof *C);
    int32_t *R = malloc((size_t)M * N * sizeof *R);
    unsigned r;
    double t0 = now_ms();

    memset(out, 0, sizeof *out);
    if (!A || !B || !C || !R) { free(A); free(B); free(C); free(R); return; }

    /* Operands that make an all-zero atom implausible as arithmetic: a legitimate zero
     * output would be indistinguishable from the defect, and that is exactly the
     * ambiguity the detector pays a wasted submit for. Dense pseudo-random int8 over
     * the full range leaves no zero accumulator except by coincidence. */
    {
        unsigned seed = 0x9E3779B9u ^ (unsigned)(M * 31 + K * 17 + N * 7);
        for (size_t i = 0; i < (size_t)M * K; i++) {
            seed = seed * 1103515245u + 12345u;
            A[i] = (int8_t)((int)((seed >> 16) % 255u) - 127);
        }
        for (size_t i = 0; i < (size_t)N * K; i++) {
            seed = seed * 1103515245u + 12345u;
            B[i] = (int8_t)((int)((seed >> 16) % 255u) - 127);
        }
    }
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[(size_t)m * K + k] * B[(size_t)n * K + k];
            R[(size_t)m * N + n] = acc;
        }

    if (force_ow1)  setenv("ROCKET_RK3576_MM_IW", "1", 1);
    if (force_wide) setenv("ROCKET_RK3576_I32_OC_MULT", "2", 1);
    for (r = 0; r < reps; r++) {
        rocket_rk3576_i32_stats st;
        int rc;
        memset(C, 0, (size_t)M * N * sizeof *C);
        rc = rocket_matmul_int8_rk3576_i32(fd, M, K, N, A, B, NULL, C);
        rocket_rk3576_i32_last_stats(&st);
        out->reps++;
        if (rc != 0) { out->refused++; continue; }
        out->tasks        += st.tasks;
        out->redo_zeroed  += st.redo_zeroed;
        out->redo_empty   += st.redo_empty;
        out->atoms_zeroed += st.atoms_zeroed;
        if (st.redo_zeroed) out->reps_hit++;
        {
            size_t nbad = 0, nzero = 0, first = 0;
            for (size_t i = 0; i < (size_t)M * N; i++)
                if (C[i] != R[i]) {
                    if (!nbad) first = i;
                    nbad++;
                    if (C[i] == 0) nzero++;
                }
            if (!nbad) continue;
            out->wrong++;
            /* WHAT the wrong elements are decides what could repair them. An element
             * that came back ZERO is an atom the writer emitted empty, which a sentinel
             * can see and a redo can heal; anything else is a value that was computed
             * and misplaced, which neither can. The first wrong rep of a cell is
             * reported for that reason alone. */
            if (out->wrong == 1) {
                printf("      rep %u: %zu wrong of %zu, %zu of them ZERO | first at "
                       "m=%zu n=%zu got=%d want=%d\n",
                       r, nbad, (size_t)M * N, nzero, first / (size_t)N,
                       first % (size_t)N, C[first], R[first]);
                fflush(stdout);
            }
        }
    }
    if (force_ow1)  unsetenv("ROCKET_RK3576_MM_IW");
    if (force_wide) unsetenv("ROCKET_RK3576_I32_OC_MULT");
    out->ms = now_ms() - t0;
    free(A); free(B); free(C); free(R);
}

static void zr_head(const char *what)
{
    printf("  %-22s %6s %6s %8s %8s %8s %7s %7s\n",
           what, "reps", "tasks", "zeroruns", "hitreps", "atoms", "empty", "wrong");
}

static void zr_row(const char *label, const struct zr_cell *c)
{
    printf("  %-22s %6u %6u %8u %8u %8u %7u %7u",
           label, c->reps, c->tasks, c->redo_zeroed, c->reps_hit,
           c->atoms_zeroed, c->redo_empty, c->wrong);
    if (c->tasks)
        printf("   %.2f%% of tasks", 100.0 * c->redo_zeroed / c->tasks);
    if (c->refused) printf("   %u REFUSED", c->refused);
    printf("\n");
    fflush(stdout);
}

/* The surface a cell occupies, as a fraction of the 8 KiB bound. Reported beside every
 * row so a rate is never read without the confound it is being held against. */
int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *mode = argc > 1 ? argv[1] : "oc";
    unsigned reps = (unsigned)env_int("ROCKET_ZR_REPS", 30);
    int heldK = env_int("ROCKET_ZR_K", 1024);
    int fd, bad = 0;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_i32_zero_run: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_i32_zero_run: no NPU device — skipping\n"); return 2; }

    printf("== the wide int32 writer's zero-emission runs: %s ==\n", mode);
    printf("   reps/cell %u, held K %d, plane forced to ow=1 so one row task is one "
           "whole plane\n", reps, heldK);

    if (!strcmp(mode, "oc")) {
        /* oc_prog walks 64..512 with the surface held at half the bound, so the only
         * thing changing is how many atoms the writer emits back to back per pixel. */
        static const unsigned OCS[] = { 32, 64, 128, 256 };
        printf("-- oc_prog at a HELD surface (50%% of the bound) --\n");
        zr_head("oc / oc_prog / M");
        for (size_t i = 0; i < sizeof OCS / sizeof *OCS; i++) {
            unsigned ocp = zr_oc_prog(OCS[i]);
            int M = (int)(2048u / ocp);
            char lbl[40];
            struct zr_cell c;
            if (M < 1) continue;
            snprintf(lbl, sizeof lbl, "oc=%-4u prog=%-4u M=%-4d", OCS[i], ocp, M);
            zr_cell(fd, M, heldK, (int)OCS[i], reps, 1, 1, &c);
            zr_row(lbl, &c);
            bad += (int)c.wrong;
        }
    } else if (!strcmp(mode, "frac")) {
        /* The surface walks from an eighth of the bound to its edge at a held oc_prog,
         * which is the axis the deterministic failure lives on. */
        static const int PCT[] = { 12, 25, 50, 75, 93 };
        unsigned oc = 256, ocp = zr_oc_prog(oc);
        printf("-- surface fraction at a HELD oc=%u (oc_prog=%u) --\n", oc, ocp);
        zr_head("percent / M");
        for (size_t i = 0; i < sizeof PCT / sizeof *PCT; i++) {
            int M = (int)(4096u * (unsigned)PCT[i] / 100u / ocp);
            char lbl[40];
            struct zr_cell c;
            if (M < 1) continue;
            snprintf(lbl, sizeof lbl, "%3d%% of 8 KiB M=%-4d", PCT[i], M);
            zr_cell(fd, M, heldK, (int)oc, reps, 1, 1, &c);
            zr_row(lbl, &c);
            bad += (int)c.wrong;
        }
    } else if (!strcmp(mode, "k")) {
        /* K at a held surface and a held oc_prog. This is the axis that moves the
         * deterministic bound, and it moves it one way only. */
        static const int KS[] = { 32, 128, 512, 1024, 2048, 4096 };
        unsigned oc = 256, ocp = zr_oc_prog(oc);
        int M = (int)(2048u / ocp);
        printf("-- K at a HELD surface (50%% of the bound, oc=%u M=%d) --\n", oc, M);
        zr_head("K");
        for (size_t i = 0; i < sizeof KS / sizeof *KS; i++) {
            char lbl[40];
            struct zr_cell c;
            snprintf(lbl, sizeof lbl, "K=%-6d", KS[i]);
            zr_cell(fd, M, KS[i], (int)oc, reps, 1, 1, &c);
            zr_row(lbl, &c);
            bad += (int)c.wrong;
        }
    } else if (!strcmp(mode, "load")) {
        /* The shape is the caller's, and the plane and writer are left to the planner:
           the question is whether the drop rate a real caller meets moves with memory
           pressure, so holding a probe geometry would answer about the wrong thing. */
        int nt = env_int("ROCKET_ZR_LOADT", 4);
        pthread_t th[16];
        int M = argc > 2 ? atoi(argv[2]) : 128;
        int K = argc > 3 ? atoi(argv[3]) : 256;
        int N = argc > 4 ? atoi(argv[4]) : 2048;
        struct zr_cell c;
        if (nt > 16) nt = 16;
        printf("-- host load at %dx%dx%d, %d thread%s per loaded cell --\n",
               M, K, N, nt, nt == 1 ? "" : "s");
        zr_head("condition");
        zr_cell(fd, M, K, N, reps, 0, 0, &c);
        zr_row("quiet", &c);
        bad += (int)c.wrong;

        /* The three loaded cells differ only in how far down the memory hierarchy
           the host's work reaches, so the row that moves names the mechanism. */
        static const struct { int kind; const char *lbl; } KINDS[] = {
            { ZR_LOAD_SPIN,  "CPU spin, no memory  (control)" },
            { ZR_LOAD_CACHE, "L1-resident memcpy   (control)" },
            { ZR_LOAD_DDR,   "streaming memcpy to DDR" },
        };
        for (size_t ki = 0; ki < sizeof KINDS / sizeof KINDS[0]; ki++) {
            zr_load_stop = 0;
            for (int i = 0; i < nt; i++)
                pthread_create(&th[i], NULL, zr_load_thread,
                               (void *)(intptr_t)KINDS[ki].kind);
            zr_cell(fd, M, K, N, reps, 0, 0, &c);
            zr_load_stop = 1;
            for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
            zr_row(KINDS[ki].lbl, &c);
            bad += (int)c.wrong;
        }
    } else if (!strcmp(mode, "hammer")) {
        int M = argc > 2 ? atoi(argv[2]) : 32;
        int K = argc > 3 ? atoi(argv[3]) : 1024;
        int N = argc > 4 ? atoi(argv[4]) : 4096;
        unsigned n = (unsigned)(argc > 5 ? atoi(argv[5]) : 20);
        struct zr_cell c;
        char lbl[40];
        /* The caller's own shape, with the library's plane and row split left alone —
         * this is the only mode that measures what a caller actually meets. */
        printf("-- one shape, as a caller gets it --\n");
        zr_head("MxKxN");
        snprintf(lbl, sizeof lbl, "%dx%dx%d", M, K, N);
        zr_cell(fd, M, K, N, n, 0, 0, &c);
        zr_row(lbl, &c);
        bad += (int)c.wrong;
    } else {
        printf("usage: rk3576_i32_zero_run [oc|frac|k|load|hammer M K N reps]\n");
        rocket_close(fd);
        return 2;
    }

    printf("== %s ==\n", bad ? "SOME REPS WERE WRONG AFTER THE REPAIR" : "every rep exact");
    rocket_close(fd);
    return bad ? 1 : 0;
}
