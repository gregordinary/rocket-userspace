// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * softplus_mish_rocket.c — HW gate for three new on-NPU activations:
 *   Softplus(x) = log(1+e^x)            shifted single-table LUT (the EXP path)
 *   Mish(x)     = x·tanh(softplus(x))   2-pass: a [0,1] gate LUT then an EW multiply
 *   Abs(x)      = |x|                    shifted single-table over a SYMMETRIC domain
 *
 * Softplus and Abs ride the shifted single-table mode (the whole domain maps onto the
 * positive LUT index half ⇒ no LE/LO sign-mux glitch — works standalone). Mish is the
 * SiLU/GELU 2-pass shape with a different gate (tanh∘softplus, a monotone 0→1 function on
 * the clean sigmoid grid). The YOLOv4/v7 backbone activation.
 *
 * The LUT is a uniform-grid fp16 approximation, so we gate with tolerances, not equality:
 *   - max ABSOLUTE error over the swept domain;
 *   - max RELATIVE error where the function is non-tiny (the deep tail's relative error is
 *     Q0.15-floored BY DESIGN — irrelevant where the value ~0).
 * A large-n sweep crosses the DPU_LUT_MAXN tile boundary (and, for Mish, the EW M-tile).
 *
 * Off-device: SKIP (exit 2). Usage: softplus_mish_rocket
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

static int g_fail = 0;

static double softplus_d(double x) { return x > 0.0 ? x + log1p(exp(-x)) : log1p(exp(x)); }
static double mish_d(double x)     { return x * tanh(softplus_d(x)); }

/* Sweep f over a uniform grid on [lo,hi]; max abs error (full domain) + max rel error
 * restricted to |f|>=rel_floor (where the value actually carries information). */
static void sweep(int fd, int kind, double (*ref)(double), const char *nm,
                  double lo, double hi, double rel_floor, double abs_bar, double rel_bar)
{
    const int N = 2048, n = (N + 7) & ~7;
    _Float16 *in  = calloc(n, sizeof(_Float16));
    _Float16 *out = calloc(n, sizeof(_Float16));
    if (!in || !out) { fprintf(stderr, "oom\n"); g_fail = 1; free(in); free(out); return; }

    for (int i = 0; i < N; i++) in[i] = (_Float16)(lo + (hi - lo) * ((double)i / (N - 1)));
    for (int i = N; i < n; i++) in[i] = (_Float16)lo;

    int r = rocket_activation_fp16(fd, kind, in, out, n);
    if (r) { printf("  %-9s [%.1f,%.1f]: call=%d FAIL\n", nm, lo, hi, r); g_fail = 1; free(in); free(out); return; }

    double max_abs = 0, max_rel = 0, wx_abs = 0, wx_rel = 0;
    for (int i = 0; i < N; i++) {
        double x = (double)(float)in[i], want = ref(x), got = (double)(float)out[i];
        double ad = fabs(got - want);
        if (ad > max_abs) { max_abs = ad; wx_abs = x; }
        if (fabs(want) >= rel_floor) {
            double rd = ad / (fabs(want) + 1e-30);
            if (rd > max_rel) { max_rel = rd; wx_rel = x; }
        }
    }
    int ok = (max_abs <= abs_bar) && (max_rel <= rel_bar);
    printf("  %-9s [%.1f,%.1f]: max_abs=%.4g (@x=%.3g)  max_rel[|f|>=%.2g]=%.3f%% (@x=%.3g) -> %s\n",
           nm, lo, hi, max_abs, wx_abs, rel_floor, 100 * max_rel, wx_rel, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
    free(in); free(out);
}

/* Large-n sweep to cross the internal tile boundaries (DPU_LUT_MAXN=32768; and the EW
 * M-tile for the 2-pass Mish). Random inputs in [lo,hi]; check max abs error only. */
static void big(int fd, int kind, double (*ref)(double), const char *nm,
                int n_real, double lo, double hi, double abs_bar)
{
    const int n = (n_real + 7) & ~7;
    _Float16 *in  = calloc(n, sizeof(_Float16));
    _Float16 *out = calloc(n, sizeof(_Float16));
    if (!in || !out) { fprintf(stderr, "oom\n"); g_fail = 1; free(in); free(out); return; }

    uint32_t st = 0xC0FFEE ^ (uint32_t)(n_real * 2654435761u);
    for (int i = 0; i < n_real; i++) {
        st = st * 1664525u + 1013904223u;
        double u = (double)((st >> 8) & 0xffff) / 65535.0;
        in[i] = (_Float16)(lo + (hi - lo) * u);
    }
    int r = rocket_activation_fp16(fd, kind, in, out, n);
    if (r) { printf("  %-9s n=%d: call=%d FAIL\n", nm, n_real, r); g_fail = 1; free(in); free(out); return; }

    double max_abs = 0; int bad = 0;
    for (int i = 0; i < n_real; i++) {
        double ad = fabs((double)(float)out[i] - ref((double)(float)in[i]));
        if (ad > max_abs) max_abs = ad;
        if (ad > abs_bar) bad++;
    }
    int ok = (bad == 0);
    printf("  %-9s n=%d (tile-boundary): max_abs=%.4g bad=%d -> %s\n", nm, n_real, max_abs, bad, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
    free(in); free(out);
}

int main(void)
{
    /* off-device self-check: the references are well-defined and finite */
    if (!(softplus_d(0.0) > 0.69 && softplus_d(0.0) < 0.70) || fabs(mish_d(0.0)) > 1e-12) {
        printf("ref self-check FAIL\n"); return 1;
    }

    int fd = rocket_open();
    if (fd < 0) { printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd); printf("==== PASS (skipped) ====\n"); return 2; }

    printf("-- Softplus = log(1+e^x) (shifted single-table) --\n");
    sweep(fd, ROCKET_ACTIVATION_SOFTPLUS, softplus_d, "softplus", -16.0, 16.0, 0.5, 3e-2, 0.03);
    sweep(fd, ROCKET_ACTIVATION_SOFTPLUS, softplus_d, "softplus",  -8.0,  8.0, 0.5, 2e-2, 0.03);

    printf("\n-- Mish = x·tanh(softplus(x)) (2-pass gate+mul) --\n");
    sweep(fd, ROCKET_ACTIVATION_MISH, mish_d, "mish", -10.0, 10.0, 0.5, 5e-2, 0.05);
    sweep(fd, ROCKET_ACTIVATION_MISH, mish_d, "mish",  -4.0,  4.0, 0.3, 2e-2, 0.05);

    printf("\n-- Abs = |x| (symmetric shifted single-table; kink on the middle sample) --\n");
    sweep(fd, ROCKET_ACTIVATION_ABS, fabs, "abs", -16.0, 16.0, 0.5, 1.5e-2, 0.02);

    printf("\n-- large-n (crosses the DPU_LUT_MAXN / EW M-tile boundaries) --\n");
    big(fd, ROCKET_ACTIVATION_SOFTPLUS, softplus_d, "softplus", 40000, -8.0, 8.0, 3e-2);
    big(fd, ROCKET_ACTIVATION_MISH,     mish_d,     "mish",     40000, -8.0, 8.0, 5e-2);
    big(fd, ROCKET_ACTIVATION_ABS,      fabs,       "abs",      40000, -8.0, 8.0, 1.5e-2);

    rocket_close(fd);
    printf("\n==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
