// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_affinity.c — pin matmul fan-out workers to the big (A76) cores. See
 * rocket_affinity.h. Keeps the pack/readback-heavy worker threads off the A55
 * little cores so a parked worker doesn't stall the join.
 */
#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "rocket_affinity.h"
#include "rocket_npu.h"     // public rocket_affinity_set_base / _get_base
#include "rocket_log.h"     // centralized log channel

/* Resolved big-core set, computed once. g_n_big: -1 = uninitialised, 0 = pinning
 * disabled (explicit off, no heterogeneous cluster, or no cpufreq), >0 = count. */
static int            g_big[CPU_SETSIZE];
static int            g_n_big = -1;
static pthread_once_t g_once  = PTHREAD_ONCE_INIT;

/* Per-thread big-core rotation base. A context pool inherits this into its
 * workers so several pools in one process spread across the cluster (default 0
 * => unchanged single-pool behaviour). Public knob: rocket_affinity_set_base. */
static __thread int   t_core_base = 0;

void rocket_affinity_set_base(int base) { t_core_base = base < 0 ? 0 : base; }
int  rocket_affinity_get_base(void)     { return t_core_base; }

/* ============================================================================
 * SECTION — Big-core detection
 * ==========================================================================*/

static long read_long_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

/* Parse a CPU list ("4-7", "4,5,6,7", "0,4-7") into out[]; returns the count. */
static int parse_cpu_list(const char *s, int *out, int max)
{
    int n = 0;
    while (*s && n < max) {
        while (*s == ',' || *s == ' ') s++;
        if (!*s) break;
        char *end;
        long a = strtol(s, &end, 10);
        if (end == s) break;                 /* not a number -> stop */
        s = end;
        long b = a;
        if (*s == '-') {                      /* a-b range */
            s++;
            b = strtol(s, &end, 10);
            if (end == s) break;
            s = end;
        }
        if (a < 0) a = 0;
        for (long c = a; c <= b && n < max; c++) out[n++] = (int)c;
    }
    return n;
}

static void detect_once(void)
{
    const char *e = getenv("ROCKET_CPU_AFFINITY");
    if (e) {
        if (!*e || !strcmp(e, "off") || !strcmp(e, "none") || !strcmp(e, "0")) {
            g_n_big = 0;
            if (getenv("ROCKET_DEBUG"))
                ROCKET_LOGD("rocket affinity: disabled (ROCKET_CPU_AFFINITY=%s)\n", e);
            return;
        }
        int n = parse_cpu_list(e, g_big, CPU_SETSIZE);
        g_n_big = (n > 0) ? n : 0;
        if (getenv("ROCKET_DEBUG"))
            ROCKET_LOGD("rocket affinity: ROCKET_CPU_AFFINITY=%s -> %d cpus\n", e, g_n_big);
        return;
    }

    /* Auto: "big" = CPUs whose cpuinfo_max_freq equals the global max. On RK3588
     * that's the 4 A76s (~2.4GHz) vs the 4 A55s (~1.8GHz). No hardcoded ids. */
    long nconf = sysconf(_SC_NPROCESSORS_CONF);
    if (nconf < 1) nconf = sysconf(_SC_NPROCESSORS_ONLN);
    if (nconf < 1) { g_n_big = 0; return; }
    if (nconf > CPU_SETSIZE) nconf = CPU_SETSIZE;

    static long freq[CPU_SETSIZE];           /* static: keep it off the stack */
    long maxf = -1;
    int  have = 0;
    for (long c = 0; c < nconf; c++) {
        char p[160];
        snprintf(p, sizeof p,
                 "/sys/devices/system/cpu/cpu%ld/cpufreq/cpuinfo_max_freq", c);
        freq[c] = read_long_file(p);
        if (freq[c] > 0) { have++; if (freq[c] > maxf) maxf = freq[c]; }
    }
    if (!have || maxf <= 0) {                 /* no cpufreq -> can't tier cores */
        g_n_big = 0;
        if (getenv("ROCKET_DEBUG"))
            ROCKET_LOGD("rocket affinity: no cpufreq info; pinning disabled\n");
        return;
    }

    int n = 0, known = 0;
    for (long c = 0; c < nconf; c++) {
        if (freq[c] <= 0) continue;           /* unknown tier -> exclude */
        known++;
        if (freq[c] == maxf) g_big[n++] = (int)c;
    }
    /* All known CPUs at the same max freq => no distinct big cluster; pinning to
     * "all of them" constrains nothing, so disable and let the scheduler decide. */
    if (n >= known) { g_n_big = 0; n = 0; }
    else            { g_n_big = n; }

    if (getenv("ROCKET_DEBUG")) {
        if (g_n_big > 0) {
            ROCKET_LOGD("rocket affinity: detected %d big cpus (max_freq=%ld kHz):",
                    g_n_big, maxf);
            for (int i = 0; i < g_n_big; i++) ROCKET_LOGD(" %d", g_big[i]);
            ROCKET_LOGD("\n");
        } else {
            ROCKET_LOGD("rocket affinity: homogeneous CPU set; pinning disabled\n");
        }
    }
}

/* ============================================================================
 * SECTION — Worker pinning
 * ==========================================================================*/

void rocket_pin_worker_based(int worker_idx, int core_base)
{
    pthread_once(&g_once, detect_once);
    if (g_n_big <= 0) return;                 /* disabled / no big cluster */
    if (worker_idx < 0) worker_idx = 0;
    if (core_base < 0) core_base = 0;

    int cpu = g_big[(core_base + worker_idx) % g_n_big];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    if (getenv("ROCKET_DEBUG"))
        ROCKET_LOGD("rocket affinity: worker %d (base %d) -> cpu %d (%s)\n",
                worker_idx, core_base, cpu, rc == 0 ? "ok" : strerror(rc));
}

/* Convenience: pin relative to the CALLING thread's affinity base (0 unless the
 * thread set one). Worker threads default to base 0, so an unconverted call site
 * keeps its historical behaviour; converted sites pass the inherited base. */
void rocket_pin_worker(int worker_idx)
{
    rocket_pin_worker_based(worker_idx, t_core_base);
}
