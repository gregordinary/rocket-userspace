/* rk3576-npu-grf: read the RK3576 NPU GRF (0x26018000) through /dev/mem, and sample
 * NPU_GRF_RKNNST (+0x1C), whose bits 0 and 1 the TRM names core0_work_on and core1_work_on.
 *
 * The GRF's bus clock hangs under pclk_nputop_root, which rocket takes as its "pclk" and
 * gates whenever 27700000.npu is runtime-suspended. So this refuses unless the device is
 * held at power/control "on" AND reads "active", and it re-checks both every 20 ms while
 * sampling. A read with that clock gated has not been taken.
 *
 *   cc -O2 -o rk3576-npu-grf rk3576-npu-grf.c
 *   echo on | sudo tee /sys/bus/platform/devices/27700000.npu/power/control
 *   sudo ./rk3576-npu-grf dump          every documented word, against its TRM reset value
 *   sudo ./rk3576-npu-grf once          one read of RKNNST
 *   sudo taskset -c 7 ./rk3576-npu-grf sample 20000 &   then run an NPU workload
 *   echo auto | sudo tee /sys/bus/platform/devices/27700000.npu/power/control
 *
 * `dump` is the control for `sample`: a GRF that did not decode a non-secure read would
 * return 0 for every word, which is indistinguishable from a dead status bit. The words
 * with non-zero reset values (MEM_CON0-2, the MCU cache window, URGENT_CON0-3) say which.
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define GRF_BASE 0x26018000UL
#define RKNNST   0x1C
#define DEV      "/sys/bus/platform/devices/27700000.npu/power/"
#define MAXT     65536

static int sysfs_is(const char *f, const char *want)
{
    char p[128], b[32] = {0};
    snprintf(p, sizeof p, DEV "%s", f);
    int fd = open(p, O_RDONLY);
    if (fd < 0) return 0;
    ssize_t n = read(fd, b, sizeof b - 1);
    close(fd);
    return n > 0 && !strncmp(b, want, strlen(want));
}

static int held(void) { return sysfs_is("control", "on") && sysfs_is("runtime_status", "active"); }

static double now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static int cmpd(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc < 2 || (strcmp(argv[1], "once") && strcmp(argv[1], "dump") &&
                     (strcmp(argv[1], "sample") || argc < 3))) {
        fprintf(stderr, "usage: %s once | dump | sample <ms>\n", argv[0]);
        return 2;
    }
    if (!held()) {
        fprintf(stderr, "refusing: 27700000.npu is not held at power/control=on and active\n");
        return 3;
    }
    int fd = open("/dev/mem", O_RDONLY | O_SYNC);
    if (fd < 0) { perror("/dev/mem"); return 1; }
    volatile uint32_t *g = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, GRF_BASE);
    if (g == MAP_FAILED) { perror("mmap"); return 1; }

    if (!strcmp(argv[1], "dump")) {
        // Documented offsets only, with the TRM reset value beside each.
        static const struct { unsigned off; uint32_t rst; const char *name; } R[] = {
            {0x08, 0x00001410, "MEM_CON0"},     {0x0C, 0x00110010, "MEM_CON1"},
            {0x10, 0x00008010, "MEM_CON2"},     {0x14, 0x00000000, "MEMGATE_CON0"},
            {0x18, 0x00000000, "MEMGATE_CON1"}, {0x1C, 0x00000000, "RKNNST"},
            {0x20, 0x00000000, "NSP_SLV_ADDR"}, {0x28, 0x00000000, "NPUTOP_CON"},
            {0x2C, 0x00000000, "STCALIB"},      {0x30, 0x20000000, "START_ADDR"},
            {0x34, 0x2FFFF000, "END_ADDR"},     {0x38, 0x00000000, "NPUTOP_ST"},
            {0x58, 0x00000003, "CACHE_MAINTAIN"}, {0x5C, 0x00000000, "RV_BASE_ADDR"},
            {0x6C, 0x000000FF, "URGENT_CON0"},  {0x70, 0x000000FF, "URGENT_CON1"},
            {0x74, 0x000000FF, "URGENT_CON2"},  {0x78, 0x000000FF, "URGENT_CON3"},
        };
        for (unsigned i = 0; i < sizeof R / sizeof R[0]; i++) {
            uint32_t v = g[R[i].off / 4];
            printf("  0x%02x %-15s 0x%08x  reset 0x%08x%s\n", R[i].off, R[i].name, v, R[i].rst,
                   v == R[i].rst ? "" : "  DIFFERS");
        }
        return 0;
    }
    if (!strcmp(argv[1], "once")) {
        uint32_t v = g[RKNNST / 4];
        printf("NPU_GRF_RKNNST = 0x%08x (core0_work_on %u, core1_work_on %u, dap_swactive %u)\n",
               v, v & 1, (v >> 1) & 1, (v >> 2) & 1);
        return 0;
    }

    double ms = atof(argv[2]);
    static double tt[MAXT];
    static uint32_t tv[MAXT];
    static double runs[MAXT];
    unsigned long long cnt[8] = {0}, other = 0, n = 0;
    int nt = 0, nr = 0;
    double t0 = now_us(), tend = t0 + ms * 1e3, tchk = t0 + 20e3, rise = 0, hi = 0;
    uint32_t prev = g[RKNNST / 4];
    if (prev & 1) rise = t0;
    for (;;) {
        double t = now_us();
        if (t >= tend) break;
        if (t >= tchk) {
            if (!held()) { fprintf(stderr, "device left the held state; stopping\n"); break; }
            tchk = t + 20e3;
        }
        uint32_t v = g[RKNNST / 4];
        n++;
        if (v & ~7u) other++; else cnt[v]++;
        if (v != prev) {
            if (nt < MAXT) { tt[nt] = t - t0; tv[nt] = v; nt++; }
            if ((v & 1) && !(prev & 1)) rise = t;
            if (!(v & 1) && (prev & 1)) {
                hi += t - rise;
                if (nr < MAXT) runs[nr++] = t - rise;
            }
            prev = v;
        }
    }
    double el = now_us() - t0;
    if (prev & 1) hi += t0 + el - rise;
    printf("samples %llu over %.1f ms (%.2f Msample/s, %.0f ns/read)\n",
           n, el / 1e3, n / el, el * 1e3 / (n ? n : 1));
    for (int i = 0; i < 8; i++)
        if (cnt[i]) printf("  value 0x%x: %llu (%.4f)\n", i, cnt[i], (double)cnt[i] / n);
    if (other) printf("  value outside bits 0-2: %llu\n", other);
    printf("transitions %d%s; bit-0 high %.3f ms of %.1f (%.4f)\n",
           nt, nt == MAXT ? " (log full)" : "", hi / 1e3, el / 1e3, hi / el);
    if (nr) {
        qsort(runs, nr, sizeof runs[0], cmpd);
        printf("bit-0 high runs %d: min %.1f us, p10 %.1f, median %.1f, p90 %.1f, max %.1f\n",
               nr, runs[0], runs[nr / 10], runs[nr / 2], runs[nr * 9 / 10], runs[nr - 1]);
    }
    for (int i = 0; i < nt && i < 12; i++) printf("  t=%.1f us -> 0x%x\n", tt[i], tv[i]);
    return 0;
}
