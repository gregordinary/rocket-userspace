// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * elu_rocket.c — HW gate for ELU/SELU on the DPU LUT.
 *   ELU(x)  = x>=0 ? x : alpha*(e^x-1)          (parametric, rocket_elu_fp16)
 *   SELU(x) = lambda*(x>=0 ? x : alpha*(e^x-1)) (fixed self-normalizing constants)
 *
 * Both ride the SYMMETRIC shifted single-table mode (the Abs trick): the x=0 kink lands on
 * the middle table sample and the whole domain maps onto the positive LUT index half, so
 * there is NO LE/LO sign-mux glitch (no host repair, unlike LeakyReLU). The negative branch
 * is curved (uniform-grid interpolation), so we gate with tolerances: max abs error over the
 * domain + max rel error where |f| is non-tiny. A large-n sweep crosses the DPU_LUT_MAXN tile.
 *
 * Off-device: SKIP (exit 2). Usage: elu_rocket
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

#define SELU_ALPHA  1.6732632423543772
#define SELU_LAMBDA 1.0507009873554805

static int g_fail = 0;

static double elu_ref(double alpha, double lambda, double x)
{ return lambda * (x >= 0.0 ? x : alpha * (exp(x) - 1.0)); }

/* Sweep ELU(alpha) or SELU over [-R,R]; max abs error + max rel where |f|>=rel_floor. */
static void sweep(int fd, int is_selu, double alpha, double R, double abs_bar, double rel_bar)
{
    const int N = 2048, n = (N + 7) & ~7;
    _Float16 *in  = calloc(n, sizeof(_Float16));
    _Float16 *out = calloc(n, sizeof(_Float16));
    if (!in || !out) { fprintf(stderr, "oom\n"); g_fail = 1; free(in); free(out); return; }

    for (int i = 0; i < N; i++) in[i] = (_Float16)(-R + 2.0 * R * ((double)i / (N - 1)));

    int r = is_selu ? rocket_selu_fp16(fd, in, out, n)
                    : rocket_elu_fp16(fd, (float)alpha, in, out, n);
    const double lam = is_selu ? SELU_LAMBDA : 1.0;
    const double a   = is_selu ? SELU_ALPHA  : alpha;
    const char *nm   = is_selu ? "selu" : "elu";
    if (r) { printf("  %-4s a=%.3f: call=%d FAIL\n", nm, a, r); g_fail = 1; free(in); free(out); return; }

    const double rel_floor = 0.1;
    double max_abs = 0, max_rel = 0, wx = 0;
    for (int i = 0; i < N; i++) {
        double x = (double)(float)in[i], want = elu_ref(a, lam, x), got = (double)(float)out[i];
        double ad = fabs(got - want);
        if (ad > max_abs) { max_abs = ad; wx = x; }
        if (fabs(want) >= rel_floor) {
            double rd = ad / (fabs(want) + 1e-30);
            if (rd > max_rel) max_rel = rd;
        }
    }
    int ok = (max_abs <= abs_bar) && (max_rel <= rel_bar);
    printf("  %-4s a=%.3f lambda=%.3f [-%.0f,%.0f]: max_abs=%.4g (@x=%.3g) max_rel[|f|>=%.1f]=%.3f%% -> %s\n",
           nm, a, lam, R, R, max_abs, wx, rel_floor, 100 * max_rel, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
    free(in); free(out);
}

/* Large-n random sweep crossing the DPU_LUT_MAXN tile boundary. */
static void big(int fd, int is_selu, double alpha, double R, int n_real, double abs_bar)
{
    const int n = (n_real + 7) & ~7;
    _Float16 *in  = calloc(n, sizeof(_Float16));
    _Float16 *out = calloc(n, sizeof(_Float16));
    if (!in || !out) { fprintf(stderr, "oom\n"); g_fail = 1; free(in); free(out); return; }
    uint32_t st = 0xE1FU ^ (uint32_t)(n_real * 40503u);
    for (int i = 0; i < n_real; i++) {
        st = st * 1664525u + 1013904223u;
        double u = (double)((st >> 8) & 0xffff) / 65535.0;
        in[i] = (_Float16)(-R + 2.0 * R * u);
    }
    int r = is_selu ? rocket_selu_fp16(fd, in, out, n)
                    : rocket_elu_fp16(fd, (float)alpha, in, out, n);
    const double lam = is_selu ? SELU_LAMBDA : 1.0, a = is_selu ? SELU_ALPHA : alpha;
    if (r) { printf("  %-4s n=%d: call=%d FAIL\n", is_selu ? "selu" : "elu", n_real, r); g_fail = 1; free(in); free(out); return; }
    double max_abs = 0; int bad = 0;
    for (int i = 0; i < n_real; i++) {
        double x = (double)(float)in[i], want = elu_ref(a, lam, x), got = (double)(float)out[i];
        double ad = fabs(got - want);
        if (ad > max_abs) max_abs = ad;
        if (ad > abs_bar) { if (bad < 6) printf("      spike: x=%.6g got=%.6g want=%.6g\n", x, got, want); bad++; }
    }
    int ok = (bad == 0);
    printf("  %-4s n=%d (tile-boundary): max_abs=%.4g bad=%d -> %s\n",
           is_selu ? "selu" : "elu", n_real, max_abs, bad, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
    free(in); free(out);
}

int main(void)
{
    if (fabs(elu_ref(1.0, 1.0, -100.0) - (-1.0)) > 1e-6) { printf("ref self-check FAIL\n"); return 1; }

    int fd = rocket_open();
    if (fd < 0) { printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd); printf("==== PASS (skipped) ====\n"); return 2; }

    printf("-- ELU(alpha) (symmetric shifted single-table; kink on the middle sample) --\n");
    sweep(fd, 0, 1.0, 8.0, 2e-2, 0.03);
    sweep(fd, 0, 0.5, 8.0, 2e-2, 0.03);
    sweep(fd, 0, 2.0, 8.0, 3e-2, 0.03);

    printf("\n-- SELU (fixed alpha=1.673, lambda=1.051) --\n");
    sweep(fd, 1, 0.0, 8.0, 3e-2, 0.03);

    printf("\n-- large-n (crosses the DPU_LUT_MAXN tile boundary) --\n");
    big(fd, 0, 1.0, 8.0, 40000, 2e-2);
    big(fd, 1, 0.0, 8.0, 40000, 3e-2);

    rocket_close(fd);
    printf("\n==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
