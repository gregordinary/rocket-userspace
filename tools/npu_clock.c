// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * npu_clock.c — read the RK3588 NPU compute clock's path and its physical frequency.
 *
 * Above 200 MHz the NPU clock is clk_npu_pvtpll, a ring oscillator in the NPU GRF whose
 * frequency follows the rail voltage. BL31 programs one ring length per requested rate, and
 * the SCMI read-back (clk_get_rate, clk_summary, a driver's "reads back" log) returns the
 * request, not the frequency. At a 0.80 V rail the 600 MHz request runs at ~816 MHz. This
 * tool reads what runs:
 *
 *   path   CRU_CLKSEL_CON74 bit 0 (clk_rknn_dsu0_sel): 1 selects the PVTPLL, 0 the
 *          normal-PLL path whose source and divider CRU_CLKSEL_CON73 [9:7] and [6:2] give.
 *          The CRU is always on, so this is safe at any time. Prints "pvtpll", or the
 *          normal-PLL rate in MHz (198 for the GPLL / 6 boot and park rate).
 *   count  The PVTPLL monitor count, NPU GRF 0x4c and 0x50, which the TRM does not list.
 *          BL31 sets the window to 24 cycles of the 24 MHz reference (PVTPLL_CON1 cal_cnt),
 *          1 us, so the count is the frequency in MHz. Also prints the ring select, the ring
 *          length and the SRAM read margin (MEM_CON0 [4:2]). The NPU GRF is inside the NPU
 *          power domain, so this refuses unless all three cores are held: power/control "on"
 *          and runtime_status "active". It re-checks the hold between samples.
 *
 *   cc -O2 -o npu_clock npu_clock.c
 *   sudo ./npu_clock path
 *   for d in fdab0000 fdac0000 fdad0000; do
 *       echo on | sudo tee /sys/bus/platform/devices/$d.npu/power/control; done
 *   sudo ./npu_clock count [samples]
 *   for d in fdab0000 fdac0000 fdad0000; do
 *       echo auto | sudo tee /sys/bus/platform/devices/$d.npu/power/control; done
 *
 * Restore power/control to "auto" before any rmmod: unloading the module with the cores held
 * on and the clock raised hard-locks the board. The requested-rate table, the voltage
 * dependence and the counter's unit check are in rockchip-npu-notes, perf/clock.md.
 * RK3588 only: it refuses on any other SoC. Exit 2 is a refusal, 1 an error.
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CRU_BASE      0xfd7c0000UL
#define CRU_CLKSEL73  0x424
#define CRU_CLKSEL74  0x428
#define NPUGRF_BASE   0xfd5a2000UL
#define GRF_MEM_CON0  0x00
#define GRF_PVT_CON0L 0x0c
#define GRF_PVT_CON0H 0x10
#define GRF_PVT_CNT0  0x4c
#define GRF_PVT_CNT1  0x50

static const char *const CORES[] = { "fdab0000.npu", "fdac0000.npu", "fdad0000.npu" };

/* CRU_CLKSEL_CON73 [9:7] sources, at the rates TF-A v2.12's rk3588_clk.c programs. */
static const double SRC_MHZ[8] = { 1188.0, 1500.0, 786.431952, 850.0, 702.0, 0, 0, 0 };

static int file_has(const char *path, const char *want)
{
    char b[256] = { 0 };
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return 0;
    ssize_t n = read(fd, b, sizeof b - 1);
    close(fd);
    if (n <= 0)
        return 0;
    /* compatible is NUL-separated: search every string in it. */
    for (ssize_t i = 0; i < n; i += (ssize_t)strlen(b + i) + 1)
        if (strstr(b + i, want))
            return 1;
    return 0;
}

static int core_is(const char *core, const char *attr, const char *want)
{
    char p[128];
    snprintf(p, sizeof p, "/sys/bus/platform/devices/%s/power/%s", core, attr);
    return file_has(p, want);
}

static int held(void)
{
    for (int i = 0; i < 3; i++)
        if (!core_is(CORES[i], "control", "on") || !core_is(CORES[i], "runtime_status", "active"))
            return 0;
    return 1;
}

static volatile uint32_t *map(unsigned long base)
{
    int fd = open("/dev/mem", O_RDONLY | O_SYNC);
    if (fd < 0) {
        perror("npu_clock: open /dev/mem");
        return NULL;
    }
    void *p = mmap(NULL, 0x1000, PROT_READ, MAP_SHARED, fd, (off_t)base);
    close(fd);
    if (p == MAP_FAILED) {
        perror("npu_clock: mmap");
        return NULL;
    }
    return (volatile uint32_t *)p;
}

static int do_path(void)
{
    volatile uint32_t *cru = map(CRU_BASE);
    if (!cru)
        return 1;
    uint32_t c73 = cru[CRU_CLKSEL73 / 4], c74 = cru[CRU_CLKSEL74 / 4];
    if (c74 & 1) {
        printf("pvtpll\n");
        return 0;
    }
    unsigned src = (c73 >> 7) & 7, div = ((c73 >> 2) & 0x1f) + 1;
    if (SRC_MHZ[src] == 0) {
        fprintf(stderr, "npu_clock: unknown normal-PLL source %u\n", src);
        return 1;
    }
    printf("%.0f\n", SRC_MHZ[src] / div);
    return 0;
}

static int do_count(int samples)
{
    if (!held()) {
        fprintf(stderr, "npu_clock: refusing: every NPU core must be held at power/control "
                        "\"on\" and read runtime_status \"active\"\n");
        return 2;
    }
    volatile uint32_t *cru = map(CRU_BASE), *grf = map(NPUGRF_BASE);
    if (!cru || !grf)
        return 1;
    if (!(cru[CRU_CLKSEL74 / 4] & 1)) {
        fprintf(stderr, "npu_clock: the clock is on the normal-PLL path; use \"path\"\n");
        return 2;
    }
    double sum = 0;
    unsigned lo = ~0u, hi = 0;
    for (int s = 0; s < samples; s++) {
        if (!held()) {
            fprintf(stderr, "npu_clock: a core left the hold mid-sample; stopped\n");
            return 2;
        }
        unsigned v = grf[GRF_PVT_CNT1 / 4] & 0xffff;
        sum += v;
        if (v < lo)
            lo = v;
        if (v > hi)
            hi = v;
        usleep(2000);
    }
    printf("mhz=%.1f min=%u max=%u ring=%u length=%u rm=%u cnt0=%u\n", sum / samples, lo, hi,
           (grf[GRF_PVT_CON0L / 4] >> 8) & 7, grf[GRF_PVT_CON0H / 4] & 0x3f,
           (grf[GRF_MEM_CON0 / 4] >> 2) & 7, grf[GRF_PVT_CNT0 / 4] & 0xffff);
    return 0;
}

int main(int argc, char **argv)
{
    if (!file_has("/proc/device-tree/compatible", "rockchip,rk3588")) {
        fprintf(stderr, "npu_clock: refusing: not an RK3588\n");
        return 2;
    }
    if (argc >= 2 && !strcmp(argv[1], "path"))
        return do_path();
    if (argc >= 2 && !strcmp(argv[1], "count")) {
        int n = argc >= 3 ? atoi(argv[2]) : 100;
        return do_count(n > 0 ? n : 100);
    }
    fprintf(stderr, "usage: npu_clock path | count [samples]\n");
    return 1;
}
