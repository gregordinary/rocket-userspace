// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * ddr_pmu_cal.c — the positive control for the `rockchip_ddr` DDR PMU.
 *
 * The NPU's own DMA byte counters are dead on RK3588 (reading the 0x2xxx page
 * hard-locks the SoC), so bytes-moved ground truth has to come from a system-level
 * PMU. `rockchip_ddr` exposes `read-bytes` / `write-bytes` / `bytes` with a sysfs
 * `.scale` claiming MB. This tool checks that claim: it moves a KNOWN number of
 * bytes past every cache and reports them, so the counter can be quoted through a
 * measured ratio instead of through its own sysfs metadata.
 *
 * An absolute single run cannot do it — allocation, the kernel's page zeroing, the
 * page faults and the idle floor are all counted too, and none of them is known. So
 * run it TWICE at different pass counts and DIFFERENCE the counters: every fixed term
 * is identical in the two arms and cancels, leaving (passes_hi - passes_lo) * size of
 * known traffic. `tools/ddr-pmu-cal.sh` drives that and prints the ratio.
 *
 * The buffer must be far larger than the 3 MB L3 or the passes never reach DRAM;
 * anything from 1 GiB up is safe on this board. A read pass touches one byte per
 * 64-byte line, which is one full line fill each; a write pass memsets, which on an
 * A76 in write-streaming mode is a line write with no read-for-ownership — so the
 * read-bytes column of a write-only differential is itself a readout of whether that
 * mode engaged, not noise.
 *
 * Build (no NPU, no library needed — plain gcc works too):
 *   gcc -O2 -o ddr_pmu_cal tests/ddr_pmu_cal.c
 * Usage: ddr_pmu_cal <MiB> <write_passes> <read_passes>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>

#define LINE 64

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

int main(int argc, char **argv) {
    size_t mib   = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 2048;
    int    wpass = argc > 2 ? atoi(argv[2]) : 1;
    int    rpass = argc > 3 ? atoi(argv[3]) : 1;
    size_t n     = mib << 20;

    unsigned char *buf = mmap(NULL, n, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (buf == MAP_FAILED) { perror("mmap"); return 1; }

    /* First touch: faults every page in. Counted, fixed across arms, cancels. */
    double t0 = now_ms();
    memset(buf, 1, n);
    double t_touch = now_ms() - t0;

    t0 = now_ms();
    for (int p = 0; p < wpass; p++) memset(buf, (int)(p & 0xff), n);
    double t_w = now_ms() - t0;

    volatile uint64_t sink = 0;
    t0 = now_ms();
    for (int p = 0; p < rpass; p++) {
        uint64_t s = 0;
        for (size_t i = 0; i < n; i += LINE) s += buf[i];
        sink += s;
    }
    double t_r = now_ms() - t0;

    double wb = (double)n * wpass, rb = (double)n * rpass;
    printf("ddr_pmu_cal: buffer %zu MiB, first-touch %.0f ms\n", mib, t_touch);
    printf("  write phase: %d pass  KNOWN %.4f GiB  %.0f ms  %.2f GB/s\n",
           wpass, wb / (1024.0*1024*1024), t_w, wb / (t_w * 1e6));
    printf("  read  phase: %d pass  KNOWN %.4f GiB  %.0f ms  %.2f GB/s\n",
           rpass, rb / (1024.0*1024*1024), t_r, rb / (t_r * 1e6));
    printf("  KNOWN-WRITE-BYTES %.0f\n  KNOWN-READ-BYTES %.0f\n  sink %llu\n",
           wb, rb, (unsigned long long)sink);
    munmap(buf, n);
    return 0;
}
