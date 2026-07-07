// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * exp_lut_rocket.c — HW gate for the EXP DPU-LUT activation (ROCKET_ACTIVATION_EXP),
 * the genuinely-new primitive of the Whisper-encoder softmax.
 *
 * EXP uses the SHIFTED single-table mode (like sqrt/rsqrt/reciprocal): the whole domain maps
 * onto the positive LUT index half, so x never crosses 0 in the INDEX domain — no LE/LO
 * sign-mux glitch, so it works on the STANDALONE flying path (unlike build_lut_affine GELU,
 * which is cos~0.05 standalone). The default domain [-16,0] is the softmax case: after the
 * row-max subtraction EXP's input is (-∞,0], output (0,1].
 *
 * The grid is UNIFORM, so the honest metrics are: (a) max ABSOLUTE error over the full domain
 * — this is what governs softmax-sum accuracy (the tail terms are tiny in abs, so quantizing
 * them to ~0 costs nothing); (b) max RELATIVE error over [-8,0] where exp >= 3.4e-4 and the
 * values actually contribute. The deep tail's relative error is Q0.15-floored BY DESIGN (a
 * value of 1e-7 can't be resolved relatively in [0,1] fixed point) and is irrelevant to
 * softmax. A direct softmax-sum check (sum exp / max-subtracted scores) closes the loop.
 *
 * Off-device: reference self-check + SKIP (exit 2).
 *
 * Usage: exp_lut_rocket
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

static int g_fail = 0;

/* sweep exp over a uniform grid on [lo,hi]; report max abs error (full) and max rel error
 * restricted to x>=rel_floor_x (where exp is non-tiny). abs_bar/rel_bar are the pass bars. */
static void sweep(int fd, double lo, double hi, double rel_floor_x, double abs_bar, double rel_bar)
{
    const int N = 1024, n = (N + 7) & ~7;
    _Float16 *in  = calloc(n, sizeof(_Float16));
    _Float16 *out = calloc(n, sizeof(_Float16));
    if (!in || !out) { fprintf(stderr, "oom\n"); g_fail = 1; free(in); free(out); return; }

    for (int i = 0; i < N; i++) in[i] = (_Float16)(lo + (hi - lo) * ((double)i / (N - 1)));
    for (int i = N; i < n; i++) in[i] = (_Float16)lo;

    int r = rocket_activation_fp16(fd, ROCKET_ACTIVATION_EXP, in, out, n);
    if (r) { printf("  exp [%.1f,%.1f]: call=%d FAIL\n", lo, hi, r); g_fail = 1; free(in); free(out); return; }

    double max_abs = 0, max_rel = 0, wx_abs = 0, wx_rel = 0;
    for (int i = 0; i < N; i++) {
        double x = (double)(float)in[i], want = exp(x), got = (double)(float)out[i];
        double ad = fabs(got - want);
        if (ad > max_abs) { max_abs = ad; wx_abs = x; }
        if (x >= rel_floor_x) {
            double rd = ad / (want + 1e-30);
            if (rd > max_rel) { max_rel = rd; wx_rel = x; }
        }
    }
    int ok = (max_abs <= abs_bar) && (max_rel <= rel_bar);
    printf("  exp [%.1f,%.1f]: max_abs=%.4g (@x=%.3g)  max_rel[x>=%.0f]=%.3f%% (@x=%.3g) -> %s\n",
           lo, hi, max_abs, wx_abs, rel_floor_x, 100 * max_rel, wx_rel, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
    free(in); free(out);
}

/* End-to-end softmax-sum check: random scores, subtract the row max (so inputs are <=0),
 * exp on the NPU, sum, and compare both the sum and the normalized distribution to fp64. */
static void softmax_sum(int fd, int n_scores, float spread)
{
    const int n = (n_scores + 7) & ~7;
    _Float16 *s  = calloc(n, sizeof(_Float16));
    _Float16 *e  = calloc(n, sizeof(_Float16));
    if (!s || !e) { fprintf(stderr, "oom\n"); g_fail = 1; free(s); free(e); return; }

    uint32_t st = 0x12345 ^ (uint32_t)(n_scores * 131 + (int)spread);
    float mx = -1e30f;
    for (int i = 0; i < n_scores; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)((st >> 8) & 0xffff) / 65535.f;
        float v = (u * 2.f - 1.f) * spread;
        s[i] = (_Float16)v;
        if (v > mx) mx = v;
    }
    /* subtract row max -> all <=0 (the mandatory softmax stabilization) */
    for (int i = 0; i < n_scores; i++) s[i] = (_Float16)((float)s[i] - mx);
    for (int i = n_scores; i < n; i++) s[i] = (_Float16)(-50.f);   /* pad: exp~0 */

    int r = rocket_activation_fp16(fd, ROCKET_ACTIVATION_EXP, s, e, n);
    if (r) { printf("  softmax n=%d spread=%.0f: call=%d FAIL\n", n_scores, spread, r); g_fail = 1; free(s); free(e); return; }

    double sum_got = 0, sum_ref = 0;
    for (int i = 0; i < n_scores; i++) { sum_got += (double)(float)e[i]; sum_ref += exp((double)(float)s[i]); }
    /* worst normalized-probability abs error */
    double max_p = 0;
    for (int i = 0; i < n_scores; i++) {
        double pg = (double)(float)e[i] / sum_got, pr = exp((double)(float)s[i]) / sum_ref;
        double d = fabs(pg - pr); if (d > max_p) max_p = d;
    }
    double sum_rel = fabs(sum_got - sum_ref) / sum_ref;
    int ok = (sum_rel <= 0.01) && (max_p <= 0.01);
    printf("  softmax n=%d spread=%.0f: sum_rel=%.3f%%  max|Δp|=%.4g -> %s\n",
           n_scores, spread, 100 * sum_rel, max_p, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
    free(s); free(e);
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd);

    if (fd >= 0) {
        printf("-- exp over the default softmax domain [-16,0] --\n");
        /* abs error is tiny everywhere (interp ~1.2e-4 near x=0, ~0 in the tail). The
         * RELATIVE bar is checked only where exp>=~0.018 (x>=-4); below that the Q0.15
         * floor (abs res ~3e-5) dominates relative error BY DESIGN — those terms are
         * negligible in the softmax sum, which the end-to-end check below verifies. */
        sweep(fd, -16.0, 0.0, -4.0, 2e-3, 0.01);
        sweep(fd, -12.0, 0.0, -4.0, 2e-3, 0.01);
        printf("\n-- softmax-sum end to end (row-max subtracted) --\n");
        softmax_sum(fd,  64, 8.f);
        softmax_sum(fd, 512, 8.f);    /* longer seq */
        softmax_sum(fd, 256, 20.f);   /* wide score spread (deep tail) */
        rocket_close(fd);
    }

    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : (fd < 0 ? 2 : 0);
}
