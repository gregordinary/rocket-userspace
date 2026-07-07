// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * x0_glitch_probe.c — RE probe: does the DPU LUT x≈0 LE/LO sign-mux glitch fire for a
 * SHIFTED single-table whose domain STRADDLES 0?
 *
 * Background (the LE/LO-mux QUIRK 2): the LE/LO hybrid mux decides LE-vs-LO on
 * sign(x) (pre-BN-ALU), so a glitch sits at the input value x=0. The shifted single-table
 * (positive-domain / exp) was thought to escape it — but only because those domains are
 * ONE-SIDED (x never changes sign). This probe densely samples [-0.02, 0.02] (step 1e-5,
 * so x=0 and the ~±4e-4 spike band are hit) for the straddling kinds, and reports the worst
 * |got - want| and whether any element exceeds a tight bar.
 *
 * Diagnostic (stays unregistered): prints findings, returns 0 unless a call errors. Run with
 * ROCKET_ELU_NOREPAIR=1 to see the raw ELU glitch (default: repaired).
 *
 * Usage: x0_glitch_probe
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

static double softplus_d(double x){ return x>0?x+log1p(exp(-x)):log1p(exp(x)); }
static double mish_d(double x){ return x*tanh(softplus_d(x)); }
static double absd(double x){ return fabs(x); }
static double elu1(double x){ return x>=0?x:1.0*(exp(x)-1.0); }

/* Probe one kind on a dense grid around 0; report worst error + its x, count over bar. */
static int probe(int fd, const char *nm, int is_elu, int kind, double (*ref)(double), double bar)
{
    const int N = 4001;                 /* x in [-0.02, 0.02], step 1e-5 */
    const int n = (N + 7) & ~7;
    _Float16 *in = calloc(n, sizeof(_Float16)), *out = calloc(n, sizeof(_Float16));
    if (!in || !out) { free(in); free(out); return -1; }
    for (int i = 0; i < N; i++) in[i] = (_Float16)(-0.02 + 4e-2 * ((double)i / (N - 1)));

    int r = is_elu ? rocket_elu_fp16(fd, 1.0f, in, out, n)
                   : rocket_activation_fp16(fd, kind, in, out, n);
    if (r) { printf("  %-9s call=%d FAIL\n", nm, r); free(in); free(out); return r; }

    double worst = 0, wx = 0, wg = 0, ww = 0; int over = 0;
    for (int i = 0; i < N; i++) {
        double x = (double)(float)in[i], want = ref(x), got = (double)(float)out[i];
        double ad = fabs(got - want);
        if (ad > worst) { worst = ad; wx = x; wg = got; ww = want; }
        if (ad > bar) over++;
    }
    printf("  %-9s worst|Δ|=%.4g @x=%.6g (got=%.5g want=%.5g)  over(%.2g)=%d/%d  %s\n",
           nm, worst, wx, wg, ww, bar, over, N, over ? "<-- GLITCH" : "clean");
    free(in); free(out);
    return 0;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU — SKIP\n"); return 2; }

    printf("Dense x≈0 probe over [-0.02,0.02] (step 1e-5).%s\n",
           getenv("ROCKET_ELU_NOREPAIR") ? "  [ELU repair OFF]" : "");
    probe(fd, "softplus", 0, ROCKET_ACTIVATION_SOFTPLUS, softplus_d, 0.02);
    probe(fd, "abs",      0, ROCKET_ACTIVATION_ABS,      absd,       0.02);
    probe(fd, "mish",     0, ROCKET_ACTIVATION_MISH,     mish_d,     0.02);
    probe(fd, "elu(a=1)", 1, 0,                          elu1,       0.02);

    rocket_close(fd);
    return 0;
}
