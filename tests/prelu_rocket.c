// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * prelu_rocket.c — HW gate for on-NPU PReLU (per-channel negative slope), the YOLO/segmentation
 * activation (ONNX PRelu). out[c][s] = x[c][s]>=0 ? x[c][s] : alpha[c]*x[c][s], input [C][S].
 *
 * No LUT is used (so no x≈0 mux glitch): for alpha in [0,1] PReLU == max(x, alpha_c*x) (a
 * per-channel scale then ew_max — 2 NPU passes); for alpha outside [0,1] it falls back to the
 * general relu(x)+alpha_c*min(x,0) (4 passes). Both only scale/select/add-zero existing fp16
 * values, so the result is BIT-EXACT vs the fp16-faithful reference (max_abs must be 0).
 *
 * Off-device: SKIP (exit 2). Usage: prelu_rocket
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

static int g_fail = 0;
static uint32_t rng = 0x9e3779b1u;
static float frand(float lo, float hi)
{ rng = rng * 1664525u + 1013904223u; return lo + (hi - lo) * ((float)((rng >> 8) & 0xffff) / 65535.f); }

/* Run PReLU(C,S) with per-channel alpha drawn from [alo,ahi]; bit-exact check vs the ref. */
static void prelu(int fd, int C, int S, float alo, float ahi, const char *tag)
{
    const size_t N = (size_t)C * S;
    _Float16 *x   = calloc(N, sizeof(_Float16));
    _Float16 *o   = calloc(N, sizeof(_Float16));
    _Float16 *ref = calloc(N, sizeof(_Float16));
    float    *a   = calloc(C, sizeof(float));
    if (!x || !o || !ref || !a) { fprintf(stderr, "oom\n"); g_fail = 1; free(x); free(o); free(ref); free(a); return; }

    for (size_t i = 0; i < N; i++) x[i] = (_Float16)frand(-8.f, 8.f);
    for (int c = 0; c < C; c++)    a[c] = frand(alo, ahi);

    rocket_prelu_ref_fp16(C, S, x, a, ref);
    int r = rocket_prelu_fp16(fd, C, S, x, a, o);
    if (r) { printf("  %-18s C=%d S=%d: call=%d FAIL\n", tag, C, S, r); g_fail = 1; goto done; }

    int bad = 0; double max_abs = 0;
    for (size_t i = 0; i < N; i++) {
        double ad = fabs((double)(float)o[i] - (double)(float)ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad != 0.0) bad++;
    }
    int ok = (bad == 0);
    printf("  %-18s C=%d S=%d alpha~[%.2f,%.2f]: max_abs=%.4g bad=%d/%zu -> %s%s\n",
           tag, C, S, alo, ahi, max_abs, bad, N, ok ? "PASS" : "FAIL", ok ? " (bit-exact)" : "");
    if (!ok) g_fail = 1;
done:
    free(x); free(o); free(ref); free(a);
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd); printf("==== PASS (skipped) ====\n"); return 2; }

    printf("-- PReLU alpha in [0,1] (max(x, alpha_c*x), 2 passes) --\n");
    prelu(fd, 8,   64,  0.0f, 1.0f, "prelu[0,1]");
    prelu(fd, 32,  256, 0.1f, 0.3f, "prelu (yolo~.25)");   /* the typical learned range */
    prelu(fd, 64,  784, 0.0f, 1.0f, "prelu[0,1] big");     /* crosses M-tile boundary   */

    printf("\n-- PReLU alpha outside [0,1] (general relu+alpha*min, 4 passes) --\n");
    prelu(fd, 16,  256, 1.0f, 2.5f, "prelu alpha>1");
    prelu(fd, 16,  256, -0.5f, 0.5f, "prelu alpha mixed");  /* some negative slopes      */

    rocket_close(fd);
    printf("\n==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
