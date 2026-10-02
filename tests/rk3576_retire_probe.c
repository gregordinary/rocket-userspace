// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_retire_probe.c — does every RK3576 write guard score a job's COMPLETION, and not
 * only whether it wrote?
 *
 * A probe, not a gate. The `rocket` driver retires a job that runs past its backstop (125 ms
 * from the kick on this part) and signals its fence as if it had completed, and a retired
 * job can leave a partial surface that a "did each task write anything" check passes. Each
 * guard site now times its job from before the submit to the fence and redoes one at or
 * past the backstop. This asks whether each site does, in two modes:
 *
 *   forced   sets ROCKET_RK3576_BACKSTOP_US=1 before the first library call, so every job
 *            reads as retired. Every entry must then REFUSE (rc < 0) and the retirement
 *            counters must move. An entry that returns 0 here does not consult the check.
 *   real     the part's own backstop. Every small entry must return 0, and count exactly
 *            as many retirements as the KERNEL logged ("retiring it", read from the journal
 *            around each arm): more is a false positive, fewer a miss. Zero is not the bar,
 *            because the part retires real jobs in ordinary use: a job after a wide-output
 *            writer (the fp16 convolution here) is poisoned, writes nothing and retires at
 *            the backstop. Then the control, the int8 matmul at M 512 K 2240 N 1536, whose
 *            jobs raise no completion on their first submit (chips/rk3576.md §"An int8
 *            matmul job can raise no completion"), must be counted retired, match the
 *            kernel, and then either refuse or return the exact surface against the DPU's
 *            requant model. The one outcome it must never have is rc 0 over a wrong surface,
 *            which is what the write-check-only guard returned at every stall K of that map
 *            (2208, 2240, 2272, 2432, 4480: 75-90% of the elements, every row from a column
 *            onward) [HW sweep, H96, 2026-09-27]. A control that is not counted means the
 *            scoring cannot see a real retirement.
 *
 * The journal is read with journalctl, which the `adm` group allows. Without it the kernel
 * column reads -1 and real mode falls back to asserting rc alone.
 *
 * The seven small calls reach six guard sites between them: the conv submit helper (the
 * direct, depthwise, packed-image and fp16 convolutions), the int8, int32 and fp16 matmul
 * writers, and the pool. The seventh site, the chained kick, is the graph's: run
 * rk3576_net_gate under ROCKET_RK3576_BACKSTOP_US=1 for it, which must fail.
 *
 * What it does NOT show: a retirement the kernel signals EARLY (a DPU grace shorter than
 * the drain), which reads fast and which no timing can see; a partial write by a job that
 * completed; the depthwise 8224-channel case that found this, which the geometry guard now
 * refuses before it reaches the device; or the small calls' arithmetic, which the gate list
 * scores.
 *
 * Usage: rk3576_retire_probe forced|real [K [N [M]]]
 *   K    the control's contraction depth in real mode, default 2240. A K whose jobs do not
 *        retire (2304) is the neighbour that must pass untouched.
 *   N    the control's output width, default 1536. A narrower N gives each job fewer columns
 *        to write before the backstop.
 *   M    the control's rows, default 512. The plane, and so the rows a task carries and the
 *        feature allowance it programs, follow M.
 * The entry plans around the stall a priori, so real mode sets ROCKET_RK3576_MM_PHASE=0
 * unless the caller set it, which keeps the control a job that retires; with the plan on
 * (=1) the control is a measurement of the plan and is expected not to retire.
 * Exit: 0 all arms as expected, 1 an arm not as expected, 2 no NPU or not an RK3576,
 *       3 (real) the control was not retired, so it cannot say anything about the scoring.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"
#include "rocket_pool.h"
#include "rocket_hw_profile.h"
#include "test_fill.h"
#include "requant_model.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static uint64_t g_u0, g_w0;
static long g_k0;

/* How many jobs the kernel has retired at its backstop this boot, or -1 if the journal
 * cannot be read. The line lands in the journal a few milliseconds after the fence, so
 * the read waits for it first. */
static long klog_retired(void)
{
    struct timespec ts = { 0, 200000000L };
    FILE *f;
    long n = -1;
    nanosleep(&ts, NULL);
    f = popen("journalctl -b -k -q --no-pager 2>/dev/null | grep -c 'retiring it'", "r");
    if (!f) return -1;
    if (fscanf(f, "%ld", &n) != 1) n = -1;
    if (pclose(f) == -1) n = -1;
    return n;
}

static void mark(void)
{
    g_k0 = klog_retired();
    rocket_rk3576_retired_counts(&g_u0, &g_w0);
}

/* Print one arm and return 1 if it went as the mode says it must. `*kdelta` is the
 * kernel's own count over the arm, or -1. */
static int arm(const char *name, int forced, int rc, double ms, long *kdelta)
{
    uint64_t u, w, du, dw;
    long k1 = klog_retired(), kd = (g_k0 >= 0 && k1 >= 0) ? k1 - g_k0 : -1;
    int ok;
    rocket_rk3576_retired_counts(&u, &w);
    du = u - g_u0; dw = w - g_w0;
    ok = forced ? (rc < 0 && du + dw >= 1)
                : (rc == 0 && (kd < 0 || (long)(du + dw) == kd));
    printf("  %-5s %-20s rc %3d  retired: %2llu unwritten, %2llu written, kernel %2ld"
           "  %8.1f ms\n", ok ? "OK" : "WRONG", name, rc, (unsigned long long)du,
           (unsigned long long)dw, kd, ms);
    if (kdelta) *kdelta = kd;
    return ok;
}

static void desc_conv(rocket_conv2d_desc *d, int ic, int hw, int oc, int k, int dw)
{
    memset(d, 0, sizeof *d);
    d->ic = ic; d->ih = hw; d->iw = hw; d->oc = oc;
    d->kh = k; d->kw = k; d->stride_y = 1; d->stride_x = 1;
    d->pad_top = k / 2; d->pad_left = k / 2;
    d->dil_y = 1; d->dil_x = 1;
    d->depthwise = dw;
}

/* The small calls. Buffers are sized for the largest of them. */
static int small_arms(int fd, int forced)
{
    enum { CAP = 64 * 1024 };
    int8_t *in = malloc(CAP), *W = malloc(CAP), *out = malloc(CAP);
    int8_t *C8 = malloc(CAP);
    int32_t *bias = calloc(256, sizeof(int32_t)), *C32 = malloc(CAP * 4);
    _Float16 *fin = malloc(CAP * 2), *fw = malloc(CAP * 2), *fout = malloc(CAP * 2);
    _Float16 *fa = malloc(CAP * 2), *fb = malloc(CAP * 2);
    float *fc = malloc(CAP * 4);
    rocket_conv2d_desc d;
    rocket_pool_desc p;
    int good = 0, n = 0, rc;
    double t0;

    if (!in || !W || !out || !C8 || !bias || !C32 || !fin || !fw || !fout || !fa || !fb ||
        !fc) {
        fprintf(stderr, "out of memory\n");
        return -1;
    }
    tf_fill_i8(in, CAP, 11, -20, 20);
    tf_fill_i8(W, CAP, 12, -8, 8);
    tf_fill_f16(fin, CAP, 13, -1.0, 1.0);
    tf_fill_f16(fw, CAP, 14, -0.25, 0.25);
    tf_fill_f16(fa, CAP, 15, -1.0, 1.0);
    tf_fill_f16(fb, CAP, 16, -1.0, 1.0);

    desc_conv(&d, 32, 16, 32, 3, 0);
    mark(); t0 = now_ms();
    rc = rocket_conv2d_int8_rk3576(fd, &d, in, W, bias, 0.05f, 0.02f, 0.1f, 0, 0, 0, out);
    good += arm("conv direct", forced, rc, now_ms() - t0, NULL); n++;

    desc_conv(&d, 32, 16, 32, 3, 1);
    mark(); t0 = now_ms();
    rc = rocket_conv2d_dw_int8_rk3576(fd, &d, in, W, bias, 0.05f, 0.02f, 0.1f, 0, 0, 0,
                                      out);
    good += arm("conv depthwise", forced, rc, now_ms() - t0, NULL); n++;

    /* ic 3, stride 1, 32 wide: inside the packed-image sub-encoding's bounds, and a
     * whole 32-channel output group, which the int8 entries require. */
    desc_conv(&d, 3, 32, 32, 3, 0);
    mark(); t0 = now_ms();
    rc = rocket_conv2d_int8_rk3576(fd, &d, in, W, bias, 0.05f, 0.02f, 0.1f, 0, 0, 0, out);
    good += arm("conv packed image", forced, rc, now_ms() - t0, NULL); n++;

    desc_conv(&d, 16, 8, 16, 3, 0);
    mark(); t0 = now_ms();
    rc = rocket_conv2d_fp16_rk3576(fd, &d, fin, fw, fout);
    good += arm("conv fp16", forced, rc, now_ms() - t0, NULL); n++;

    mark(); t0 = now_ms();
    rc = rocket_matmul_int8_rk3576(fd, 64, 128, 64, in, W, bias, 1.0f / 256.0f, C8);
    good += arm("matmul int8", forced, rc, now_ms() - t0, NULL); n++;

    mark(); t0 = now_ms();
    rc = rocket_matmul_int8_rk3576_i32(fd, 8, 64, 32, in, W, bias, C32);
    good += arm("matmul int32", forced, rc, now_ms() - t0, NULL); n++;

    mark(); t0 = now_ms();
    rc = rocket_matmul_fp16_rk3576(fd, 64, 128, 64, fa, fb, fc);
    good += arm("matmul fp16", forced, rc, now_ms() - t0, NULL); n++;

    memset(&p, 0, sizeof p);
    p.c = 32; p.ih = 16; p.iw = 16; p.kh = 2; p.kw = 2; p.stride_y = 2; p.stride_x = 2;
    p.method = POOL_METHOD_MAX;
    mark(); t0 = now_ms();
    rc = rocket_pool_int8_rk3576(fd, &p, 0, in, out);
    good += arm("pool max", forced, rc, now_ms() - t0, NULL); n++;

    free(in); free(W); free(out); free(C8); free(bias); free(C32);
    free(fin); free(fw); free(fout); free(fa); free(fb); free(fc);
    printf("  %d of %d small arms as expected\n", good, n);
    return good == n ? 0 : 1;
}

/* The control: a shape whose first submit per job retires at the backstop. */
static int control_arm(int fd, int K, int N, int M)
{
    int Mt = 0, Kt = 0, Nt = 0, jobs, col_lo = N, col_hi = -1, cols = 0, rows = 0;
    unsigned char *colbad = calloc((size_t)N, 1);
    const float scale = 1.0f / 2048.0f;
    int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K), *C = malloc((size_t)M * N);
    int32_t *bias = malloc((size_t)N * sizeof(int32_t));
    uint64_t u, w;
    size_t wrong = 0, first = (size_t)-1;
    int rc, m, n, k;
    double t0, ms;

    if (!A || !B || !C || !bias || !colbad) { fprintf(stderr, "out of memory\n"); return 1; }
    jobs = rocket_matmul_plan_int8_rk3576(M, K, N, &Mt, &Kt, &Nt);
    tf_fill_i8(A, (size_t)M * K, 21, -16, 16);
    tf_fill_i8(B, (size_t)N * K, 22, -16, 16);
    tf_fill_i32(bias, (size_t)N, 23, -4096, 4096);
    long kd;
    int agree;
    mark(); t0 = now_ms();
    rc = rocket_matmul_int8_rk3576(fd, M, K, N, A, B, bias, scale, C);
    ms = now_ms() - t0;
    rocket_rk3576_retired_counts(&u, &w);
    u -= g_u0; w -= g_w0;
    {
        long k1 = klog_retired();
        kd = (g_k0 >= 0 && k1 >= 0) ? k1 - g_k0 : -1;
    }
    agree = kd < 0 || (long)(u + w) == kd;
    if (rc == 0)
        for (m = 0; m < M; m++) {
            int rowbad = 0;
            for (n = 0; n < N; n++) {
                int64_t acc = bias[n];
                const int8_t *a = A + (size_t)m * K, *b = B + (size_t)n * K;
                for (k = 0; k < K; k++) acc += (int64_t)a[k] * b[k];
                if (C[(size_t)m * N + n] != requant_scale(acc, scale)) {
                    if (first == (size_t)-1) first = (size_t)m * N + n;
                    wrong++;
                    rowbad = 1;
                    colbad[n] = 1;
                }
            }
            rows += rowbad;
        }
    for (n = 0; n < N; n++)
        if (colbad[n]) {
            cols++;
            if (n < col_lo) col_lo = n;
            if (n > col_hi) col_hi = n;
        }
    printf("  control: int8 matmul %dx%dx%d rc %d  retired: %llu unwritten, %llu written, "
           "kernel %ld  %.1f ms  %zu of %d wrong", M, K, N, rc, (unsigned long long)u,
           (unsigned long long)w, kd, ms, wrong, M * N);
    if (wrong) printf(" (first at m %zu n %zu)", first / N, first % N);
    printf("\n           plan: %d job(s), Mt %d Kt %d Nt %d", jobs, Mt, Kt, Nt);
    if (wrong)
        printf("; wrong in %d of %d rows and %d of %d columns, columns %d..%d", rows, M,
               cols, N, col_lo, col_hi);
    printf("\n");
    if (u + w == 0 && K == 2240 && N == 1536 && M == 512 &&
        !strcmp(getenv("ROCKET_RK3576_MM_PHASE") ? getenv("ROCKET_RK3576_MM_PHASE") : "1",
                "0")) {
        printf("  the control was not retired: this run says nothing about the scoring\n");
        free(A); free(B); free(C); free(bias); free(colbad);
        return 3;
    }
    printf("           %s\n", rc < 0 ? "refused" : wrong ? "RETURNED A WRONG SURFACE"
                                                     : "exact");
    /* The same call again in this process: what a caller repeating the shape pays, and
     * whether it returns the same surface. */
    if (rc == 0) {
        int8_t *C2 = malloc((size_t)M * N);
        if (C2) {
            int rc2;
            uint64_t u2, w2;
            rocket_rk3576_retired_counts(&u, &w);
            t0 = now_ms();
            rc2 = rocket_matmul_int8_rk3576(fd, M, K, N, A, B, bias, scale, C2);
            ms = now_ms() - t0;
            rocket_rk3576_retired_counts(&u2, &w2);
            printf("           again: rc %d, %.1f ms, %llu retired, %s the first surface\n",
                   rc2, ms, (unsigned long long)(u2 + w2 - u - w),
                   rc2 == 0 && !memcmp(C, C2, (size_t)M * N) ? "equal to" : "NOT equal to");
            if (rc2 != 0 || memcmp(C, C2, (size_t)M * N)) wrong++;
            free(C2);
        }
    }
    free(A); free(B); free(C); free(bias); free(colbad);
    return ((rc < 0 || wrong == 0) && agree) ? 0 : 1;
}

int main(int argc, char **argv)
{
    int forced, fd, rc;

    if (argc < 2 || argc > 5 || (strcmp(argv[1], "forced") && strcmp(argv[1], "real"))) {
        fprintf(stderr, "usage: %s forced|real [control K, default 2240 [N, default 1536 [M, default 512]]]\n",
                argv[0]);
        return 1;
    }
    forced = !strcmp(argv[1], "forced");
    /* Before the first library call: the backstop is resolved once per process. Fewer
     * attempts keep the forced arms short; each one still costs a power cycle. */
    /* The control is a stall K, and the entry now plans around the stall
     * (rocket_rk3576_weight_phase_groups()), so it would never retire. Off by default here
     * so the control stays the retiring job it exists to be; ROCKET_RK3576_MM_PHASE=1 set
     * by the caller measures the plan instead. */
    if (!forced) setenv("ROCKET_RK3576_MM_PHASE", "0", 0);
    if (forced) {
        setenv("ROCKET_RK3576_BACKSTOP_US", "1", 1);
        if (!getenv("ROCKET_RK3576_TASK_ATTEMPTS"))
            setenv("ROCKET_RK3576_TASK_ATTEMPTS", "2", 1);
    }
    if (strcmp(rocket_hw_current()->name, "rk3576") != 0) {
        fprintf(stderr, "not an RK3576 (profile %s)\n", rocket_hw_current()->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no NPU\n"); return 2; }
    printf("rk3576_retire_probe %s (backstop %s)\n", argv[1],
           forced ? "forced to 1 us" : "the part's own");
    rc = small_arms(fd, forced);
    if (!forced) {
        int crc = control_arm(fd, argc > 2 ? atoi(argv[2]) : 2240,
                              argc > 3 ? atoi(argv[3]) : 1536,
                              argc > 4 ? atoi(argv[4]) : 512);
        if (rc == 0 || crc == 3) rc = crc;
    }
    rocket_close(fd);
    return rc < 0 ? 1 : rc;
}
