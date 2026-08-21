// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * hostload.c — put the host memory system under load while something else runs.
 *
 * The RK3576 int32 matmul writers drop output atoms at a rate that follows host
 * DDR traffic (11% of row tasks quiet, 45% under four streaming-memcpy threads),
 * so "is this path affected too?" is a question about running an existing gate
 * with the memory system busy. This is the load half of that, as a separate
 * process, so any gate can be put under it without being rebuilt.
 *
 * Three kinds, so a result has its controls:
 *   ddr    (default) 32 MiB buffers -- CPUs busy, caches thrashed, DDR saturated
 *   cache  16 KiB buffers -- the same memcpy, L1-resident, DDR sees nothing
 *   spin   integer work in registers -- no memory traffic at all
 *
 * A gate that fails under `ddr` and survives `cache`/`spin` is a memory-path
 * result; one that fails under all three is about the host being busy.
 *
 * Usage: ./hostload [ddr|cache|spin] [threads] [seconds]
 *   defaults: ddr 4 60. Runs for `seconds` then exits, so it cannot outlive the
 *   run it is loading if that run is started after it and finishes first.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

enum { HL_DDR, HL_CACHE, HL_SPIN };

static volatile int hl_stop;
static volatile uint64_t hl_sink;

static void *hl_thread(void *arg)
{
    int kind = (int)(intptr_t)arg;

    if (kind == HL_SPIN) {
        uint64_t x = 0x9E3779B97F4A7C15ULL;
        while (!hl_stop)
            for (int i = 0; i < 4096; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; }
        hl_sink += x;
        return NULL;
    }

    size_t n = (kind == HL_CACHE) ? (16u << 10) : (32u << 20);
    char *a = malloc(n), *b = malloc(n);
    if (!a || !b) { free(a); free(b); return NULL; }
    memset(a, 0x5A, n);
    while (!hl_stop) { memcpy(b, a, n); memcpy(a, b, n); }
    free(a); free(b);
    return NULL;
}

int main(int argc, char **argv)
{
    const char *k = (argc > 1) ? argv[1] : "ddr";
    int nt        = (argc > 2) ? atoi(argv[2]) : 4;
    int secs      = (argc > 3) ? atoi(argv[3]) : 60;
    int kind = !strcmp(k, "spin") ? HL_SPIN : !strcmp(k, "cache") ? HL_CACHE : HL_DDR;
    pthread_t th[64];

    if (nt < 1) nt = 1;
    if (nt > 64) nt = 64;

    printf("hostload: %s, %d threads, %d s\n", k, nt, secs);
    fflush(stdout);

    for (int i = 0; i < nt; i++)
        pthread_create(&th[i], NULL, hl_thread, (void *)(intptr_t)kind);

    struct timespec ts = { .tv_sec = secs, .tv_nsec = 0 };
    nanosleep(&ts, NULL);

    hl_stop = 1;
    for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);

    printf("hostload: done\n");
    return 0;
}
