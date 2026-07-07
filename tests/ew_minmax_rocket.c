// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * ew_minmax_rocket.c — HW gate for on-NPU elementwise two-tensor MAX/MIN and Clip.
 *
 *   rocket_ew_max_fp16(a,b) = max(a,b)   rocket_ew_min_fp16(a,b) = min(a,b)
 *   rocket_clip_fp16(lo,hi,x) = min(max(x,lo),hi)
 *
 * MAX/MIN ride the same conv-main EW datapath as add/mul, with the DPU EW ALU algo set to
 * MAX(0)/MIN(1) (EW_ALU_ALGO, DPU_EW_CFG bits[17:16]) — the RE finding of this gate. Because
 * they merely SELECT one of the two fp16 operands (no arithmetic), the result is BIT-EXACT vs
 * the host (max_abs must be 0). Clip is two passes (constant-operand max then min), also exact.
 * Random inputs across the M-tile boundary (M_TILE=1020 rows of 32 => 32640 elems).
 *
 * Off-device: SKIP (exit 2). Usage: ew_minmax_rocket
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

static int g_fail = 0;
static uint32_t rng = 0x1234567u;
static float frand(float lo, float hi)
{ rng = rng * 1664525u + 1013904223u; return lo + (hi - lo) * ((float)((rng >> 8) & 0xffff) / 65535.f); }

/* MAX or MIN, bit-exact check. */
static void minmax(int fd, int is_min, int n)
{
    _Float16 *a = calloc(n, sizeof(_Float16)), *b = calloc(n, sizeof(_Float16));
    _Float16 *o = calloc(n, sizeof(_Float16));
    if (!a || !b || !o) { fprintf(stderr, "oom\n"); g_fail = 1; free(a); free(b); free(o); return; }
    for (int i = 0; i < n; i++) { a[i] = (_Float16)frand(-8.f, 8.f); b[i] = (_Float16)frand(-8.f, 8.f); }

    int r = is_min ? rocket_ew_min_fp16(fd, a, b, o, n) : rocket_ew_max_fp16(fd, a, b, o, n);
    const char *nm = is_min ? "min" : "max";
    if (r) { printf("  %s n=%d: call=%d FAIL\n", nm, n, r); g_fail = 1; free(a); free(b); free(o); return; }

    int bad = 0; double max_abs = 0;
    for (int i = 0; i < n; i++) {
        float wa = (float)a[i], wb = (float)b[i];
        float want = is_min ? (wa < wb ? wa : wb) : (wa > wb ? wa : wb);
        double ad = fabs((double)(float)o[i] - (double)want);
        if (ad > max_abs) max_abs = ad;
        if (ad != 0.0) bad++;
    }
    int ok = (bad == 0);
    printf("  %s n=%d: max_abs=%.4g bad=%d -> %s%s\n", nm, n, max_abs, bad,
           ok ? "PASS" : "FAIL", ok ? " (bit-exact)" : "");
    if (!ok) g_fail = 1;
    free(a); free(b); free(o);
}

/* Clip(lo,hi), exact vs host. */
static void clip(int fd, float lo, float hi, int n)
{
    _Float16 *x = calloc(n, sizeof(_Float16)), *o = calloc(n, sizeof(_Float16));
    if (!x || !o) { fprintf(stderr, "oom\n"); g_fail = 1; free(x); free(o); return; }
    for (int i = 0; i < n; i++) x[i] = (_Float16)frand(-10.f, 10.f);

    int r = rocket_clip_fp16(fd, lo, hi, x, o, n);
    if (r) { printf("  clip[%.1f,%.1f] n=%d: call=%d FAIL\n", lo, hi, n, r); g_fail = 1; free(x); free(o); return; }

    int bad = 0; double max_abs = 0;
    for (int i = 0; i < n; i++) {
        float v = (float)x[i]; float want = v < lo ? lo : (v > hi ? hi : v);
        double ad = fabs((double)(float)o[i] - (double)(_Float16)want);
        if (ad > max_abs) max_abs = ad;
        if (ad != 0.0) bad++;
    }
    int ok = (bad == 0);
    printf("  clip[%.1f,%.1f] n=%d: max_abs=%.4g bad=%d -> %s\n", lo, hi, n, max_abs, bad, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
    free(x); free(o);
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd); printf("==== PASS (skipped) ====\n"); return 2; }

    printf("-- elementwise two-tensor MAX/MIN (DPU EW ALU algo MAX/MIN) bit-exact --\n");
    minmax(fd, 0, 256);
    minmax(fd, 1, 256);
    minmax(fd, 0, 4096);
    minmax(fd, 1, 4096);
    minmax(fd, 0, 40000);   /* crosses the M-tile boundary */
    minmax(fd, 1, 40000);

    printf("\n-- Clip(lo,hi) = min(max(x,lo),hi) (two EW passes) --\n");
    clip(fd, 0.f, 6.f, 4096);     /* ReLU6 */
    clip(fd, -1.f, 1.f, 4096);
    clip(fd, -3.f, 3.f, 40000);

    rocket_close(fd);
    printf("\n==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
