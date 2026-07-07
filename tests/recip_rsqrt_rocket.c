// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * recip_rsqrt_rocket.c — HW gate for the positive-domain DPU-LUT activations
 * (sqrt / rsqrt / reciprocal) and the fully-on-NPU elementwise Div.
 *
 * These are the first NON-monotone-bounded LUT kinds: f defined only for x>0, realised with
 * the SHIFTED single-table mode (the whole domain maps onto the positive LUT index half, so
 * x never crosses 0 — no LE/LO mux glitch). The LUT grid is UNIFORM, so the headline result
 * is the achieved accuracy over the domain (worst at the steep low end). rsqrt is the
 * RMSNorm/LayerNorm core; reciprocal underlies softmax-denominator and Div.
 *
 * The gate sweeps each kind over its domain and reports max/mean RELATIVE error (vs the exact
 * double-precision function), and validates ew_div = a/b. Off-device (no NPU): SKIP.
 *
 * Usage: recip_rsqrt_rocket
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

static int g_fail = 0;

static double exact(int kind, double x)
{
    switch (kind) {
    case ROCKET_ACTIVATION_SQRT:       return sqrt(x);
    case ROCKET_ACTIVATION_RSQRT:      return 1.0 / sqrt(x);
    case ROCKET_ACTIVATION_RECIPROCAL: return 1.0 / x;
    }
    return 0;
}

/* sweep a kind over [lo,hi] (log-spaced, the natural scale for these functions) and report
 * max/mean relative error. `bar` is the pass threshold on max relative error. */
static void sweep(int fd, int kind, double lo, double hi, double bar)
{
    const char *nm = rocket_activation_name(kind);
    const int N = 1024, n = (N + 7) & ~7;
    _Float16 *in  = calloc(n, sizeof(_Float16));
    _Float16 *out = calloc(n, sizeof(_Float16));
    if (!in || !out) { fprintf(stderr, "oom\n"); g_fail = 1; free(in); free(out); return; }

    for (int i = 0; i < N; i++) {
        double t = (double)i / (N - 1);
        in[i] = (_Float16)(lo * pow(hi / lo, t));     /* log-spaced over [lo,hi] */
    }
    for (int i = N; i < n; i++) in[i] = (_Float16)lo;

    int r = rocket_activation_fp16(fd, kind, in, out, n);
    if (r) { printf("  %-10s: call=%d FAIL\n", nm, r); g_fail = 1; free(in); free(out); return; }

    double max_rel = 0, sum_rel = 0; double worst_x = 0;
    for (int i = 0; i < N; i++) {
        double x = (double)(float)in[i], want = exact(kind, x), got = (double)(float)out[i];
        double rel = fabs(got - want) / (fabs(want) + 1e-9);
        if (rel > max_rel) { max_rel = rel; worst_x = x; }
        sum_rel += rel;
    }
    int ok = max_rel <= bar;
    printf("  %-10s [%.4g,%.4g]: max_rel=%.3f%% (@x=%.4g)  mean_rel=%.3f%%  -> %s\n",
           nm, lo, hi, 100*max_rel, worst_x, 100*sum_rel/N, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
    free(in); free(out);
}

/* LOG sweep — ABSOLUTE error (relative is ill-defined at the x=1 zero crossing; log is used
 * additively so absolute error is what matters). Default LUT domain [0.25,32]. */
static void test_log(int fd, double lo, double hi, double bar_abs)
{
    const int N = 1024, n = (N + 7) & ~7;
    _Float16 *in  = calloc(n, sizeof(_Float16));
    _Float16 *out = calloc(n, sizeof(_Float16));
    if (!in || !out) { fprintf(stderr, "oom\n"); g_fail = 1; free(in); free(out); return; }
    for (int i = 0; i < N; i++) {
        double t = (double)i / (N - 1);
        in[i] = (_Float16)(lo * pow(hi / lo, t));     /* log-spaced over [lo,hi] */
    }
    for (int i = N; i < n; i++) in[i] = (_Float16)1.0;

    int r = rocket_activation_fp16(fd, ROCKET_ACTIVATION_LOG, in, out, n);
    if (r) { printf("  log: call=%d FAIL\n", r); g_fail = 1; free(in); free(out); return; }

    double max_abs = 0, sum_abs = 0, worst_x = 0;
    for (int i = 0; i < N; i++) {
        double x = (double)(float)in[i], want = log(x), got = (double)(float)out[i];
        double ad = fabs(got - want);
        if (ad > max_abs) { max_abs = ad; worst_x = x; }
        sum_abs += ad;
    }
    int ok = max_abs <= bar_abs;
    printf("  log        [%.4g,%.4g]: max_abs=%.4f (@x=%.4g)  mean_abs=%.4f  -> %s\n",
           lo, hi, max_abs, worst_x, sum_abs/N, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
    free(in); free(out);
}

static void test_div(int fd)
{
    const int N = 600, n = (N + 7) & ~7;
    _Float16 *a = calloc(n,sizeof(_Float16)), *b = calloc(n,sizeof(_Float16));
    _Float16 *out = calloc(n,sizeof(_Float16));
    if (!a||!b||!out){fprintf(stderr,"oom\n");g_fail=1;free(a);free(b);free(out);return;}
    /* a in [-4,4], b positive in [0.5, 8] (inside the reciprocal domain) */
    for (int i = 0; i < N; i++) {
        a[i] = (_Float16)(-4.0 + 8.0 * ((i * 37 % 100) / 99.0));
        b[i] = (_Float16)(0.5 * pow(16.0, (i * 53 % 100) / 99.0));   /* 0.5 .. 8 */
    }
    for (int i=N;i<n;i++) b[i]=(_Float16)1.0;
    int r = rocket_ew_div_fp16(fd, a, b, out, n);
    if (r){printf("  ew_div: call=%d FAIL\n",r);g_fail=1;free(a);free(b);free(out);return;}
    double max_rel=0,sum_rel=0;
    for (int i=0;i<N;i++){
        double want=(double)(float)a[i]/(double)(float)b[i], got=(double)(float)out[i];
        double rel=fabs(got-want)/(fabs(want)+1e-3);
        if(rel>max_rel)max_rel=rel;
        sum_rel+=rel;
    }
    int ok = max_rel <= 0.05;
    printf("  ew_div a/b (b in[0.5,8]): max_rel=%.3f%% mean_rel=%.3f%% -> %s\n",
           100*max_rel, 100*sum_rel/N, ok?"PASS":"FAIL");
    if(!ok) g_fail=1;
    free(a);free(b);free(out);
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) {   /* no NPU: SKIP cleanly (CTest skip code), like the other activation gates */
        printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd);
        printf("==== SKIP ====\n");
        return 2;
    }

    /* sweep WITHIN each kind's default LUT domain (act_positive_domain); the bars are the
     * uniform-grid accuracy over a ~128x domain placed away from the steep near-0 region */
    sweep(fd, ROCKET_ACTIVATION_SQRT,       0.30, 60.0, 0.02);
    sweep(fd, ROCKET_ACTIVATION_RSQRT,      0.60, 60.0, 0.02);
    sweep(fd, ROCKET_ACTIVATION_RECIPROCAL, 0.30, 30.0, 0.03);
    /* LOG: signed output, absolute-error metric; worst at the steep small-x end. */
    test_log(fd, 0.30, 30.0, 0.03);
    if (fd >= 0) test_div(fd);

    if (fd >= 0) rocket_close(fd);
    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : (fd < 0 ? 2 : 0);
}
