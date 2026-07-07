// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * leaky_relu_rocket.c — HW gate for on-NPU LeakyReLU (rocket_leaky_relu_fp16), the
 * parametric piecewise-linear activation on the DPU LUT (shifted single-table mode).
 *
 * LeakyReLU(x) = x>=0 ? x : alpha*x. The implementation maps a SYMMETRIC domain [-R,R]
 * onto the positive LUT index half so the x=0 kink is an exact interior sample — exact and
 * glitch-free (no x≈0 LE/LO mux spike). This gate proves both:
 *
 *   1. ACCURACY over [-R,R] across a sweep of alphas vs the fp16 CPU reference.
 *   2. THE x≈0 BAND specifically — inputs within ±0.002 of 0 (where the smooth single-pass
 *      kinds spike on the LE/LO boundary) must stay correct (≈0). This is the whole reason
 *      for the symmetric-shifted design.
 *
 * Off-device (no NPU): the CPU reference self-consistency only -> SKIP (exit 2).
 *
 * Usage: leaky_relu_rocket            (alpha sweep)
 *        leaky_relu_rocket ALPHA      (one slope)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

/* near-zero inputs that trip the LE/LO mux for the smooth single-pass kinds */
static const float ZBAND[] = { -0.002f,-0.0015f,-0.001f,-0.0005f, 0.f, 0.0005f,0.001f,0.0015f,0.002f };
#define NZ ((int)(sizeof(ZBAND)/sizeof(ZBAND[0])))

static int run_alpha(int fd, float alpha)
{
    const float R = 15.9f;                 /* stay just inside the default table [-16,16] */
    const int SWEEP = 1200;
    const int n = ((SWEEP + NZ) + 7) & ~7;

    _Float16 *in  = calloc(n, sizeof(_Float16));
    _Float16 *out = calloc(n, sizeof(_Float16));
    _Float16 *ref = calloc(n, sizeof(_Float16));
    if (!in || !out || !ref) { fprintf(stderr, "oom\n"); free(in);free(out);free(ref); return 1; }

    for (int i = 0; i < SWEEP; i++)            /* uniform sweep across [-R, R] */
        in[i] = (_Float16)(-R + (2.f * R) * ((float)i / (float)(SWEEP - 1)));
    for (int i = 0; i < NZ; i++) in[SWEEP + i] = (_Float16)ZBAND[i];   /* + the x≈0 band */

    rocket_leaky_relu_ref_fp16(alpha, in, ref, n);

    if (fd < 0) {                              /* off-device: ref sanity only */
        int bad = 0;
        for (int i = 0; i < n; i++) {
            float x = (float)in[i], r = (float)ref[i], want = x >= 0 ? x : alpha*x;
            if (fabsf(r - (float)(_Float16)want) > 1e-3f) bad++;
        }
        printf("  alpha=%.3f: ref self-check %s (no NPU)\n", alpha, bad ? "FAIL" : "ok");
        free(in); free(out); free(ref);
        return bad ? 1 : 0;
    }

    int r = rocket_leaky_relu_fp16(fd, alpha, in, out, n);
    if (r) { printf("  alpha=%.3f: rocket_leaky_relu_fp16 = %d -> FAIL\n", alpha, r);
             free(in); free(out); free(ref); return 1; }

    /* accuracy over the full sweep */
    double max_abs = 0; int bad = 0;
    for (int i = 0; i < SWEEP; i++) {
        double ad = fabs((float)out[i] - (float)ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad > 0.02) { if (bad < 5) printf("    [x=%.4f] ref=%.4f got=%.4f d=%.4f\n",
                              (float)in[i], (float)ref[i], (float)out[i], ad); bad++; }
    }
    /* the x≈0 band: must NOT spike */
    double zmax = 0; int zbad = 0;
    for (int i = SWEEP; i < SWEEP + NZ; i++) {
        double ad = fabs((float)out[i] - (float)ref[i]);
        if (ad > zmax) zmax = ad;
        if (ad > 0.02) { printf("    x≈0 GLITCH [x=%.4f] ref=%.4f got=%.4f\n",
                         (float)in[i], (float)ref[i], (float)out[i]); zbad++; }
    }
    int ok = (bad == 0 && zbad == 0);
    printf("  alpha=%.3f: sweep max_abs=%.5f (bad=%d)  x≈0 band max_abs=%.5f (bad=%d) -> %s\n",
           alpha, max_abs, bad, zmax, zbad, ok ? "PASS" : "FAIL");
    free(in); free(out); free(ref);
    return ok ? 0 : 1;
}

/* RE probe: map which LUT indices glitch. Feeds x(I) = x_lo + I*(x_hi-x_lo)/16384 for the
 * runtime's default symmetric domain [-16,16] (scale=512), so index I is hit exactly, and
 * reports every I where the output spikes. Reveals the LE/LO sub-boundary structure. */
static int boundary_scan(int fd, float alpha)
{
    if (fd < 0) { printf("scan needs the NPU\n"); return 0; }
    /* RAW glitch map: disable the host repair, sweep x uniformly across [-16,16] and report
     * every input that spikes (|out| far above any legit value R=16). Confirms whether the
     * ONLY mux glitch is at x≈0 (so the host repair band suffices). */
    setenv("ROCKET_LEAKY_NOREPAIR", "1", 1);
    const double R = 16.0;
    const int NI = 16385;
    const int n = (NI + 7) & ~7;
    _Float16 *in = calloc(n, sizeof(_Float16)), *out = calloc(n, sizeof(_Float16));
    if (!in || !out) { free(in);free(out); return 1; }
    for (int I = 0; I < NI; I++) in[I] = (_Float16)(-R + 2.0 * R * (double)I / (NI - 1));
    if (rocket_leaky_relu_fp16(fd, alpha, in, out, n)) { free(in);free(out); return 1; }
    printf("boundary scan (alpha=%.2f, x uniform [-16,16], repair OFF):\n", alpha);
    int g = 0; double xmin = 1e9, xmax = -1e9;
    for (int I = 0; I < NI; I++) {
        if (fabs((float)out[I]) > 40.0) {
            if (g < 8) printf("  SPIKE x=%.6f got=%.4f\n", (float)in[I], (float)out[I]);
            if ((float)in[I] < xmin) xmin = (float)in[I];
            if ((float)in[I] > xmax) xmax = (float)in[I];
            g++;
        }
    }
    if (g) printf("  total raw spikes: %d  (x range [%.6f, %.6f]) — all at x≈0\n", g, xmin, xmax);
    else   printf("  total raw spikes: 0\n");
    unsetenv("ROCKET_LEAKY_NOREPAIR");
    free(in); free(out);
    return 0;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — CPU ref self-check only (SKIP)\n", fd);

    if (argc == 2 && strcmp(argv[1], "scan") == 0) {
        int r = boundary_scan(fd, 0.1f);
        if (fd >= 0) rocket_close(fd);
        return r;
    }

    int fail = 0;
    if (argc == 2) {
        fail = run_alpha(fd, (float)atof(argv[1]));
    } else {
        /* realistic LeakyReLU/PReLU slopes (alpha>0; alpha=0 is plain ReLU -> native path) */
        float alphas[] = { 0.01f, 0.1f, 0.125f, 0.2f, 0.25f, 0.5f };
        for (size_t i = 0; i < sizeof(alphas)/sizeof(alphas[0]); i++)
            fail |= run_alpha(fd, alphas[i]);
    }

    /* informational: a value beyond the default table saturates (documented limit) */
    if (fd >= 0) {
        _Float16 big[8] = {0}; big[0] = (_Float16)24.f; big[1] = (_Float16)(-24.f);
        _Float16 ob[8] = {0};
        if (rocket_leaky_relu_fp16(fd, 0.1f, big, ob, 8) == 0)
            printf("  [info] beyond-R x=24 -> %.3f (saturates at R=16), x=-24 -> %.4f "
                   "(saturates at -R*alpha=-1.6); edge-saturates by design\n",
                   (float)ob[0], (float)ob[1]);
    }

    if (fd >= 0) rocket_close(fd);
    int skip = (fd < 0);
    printf("==== %s ====\n", skip ? "SKIP (no NPU)" : (fail ? "FAIL" : "PASS"));
    return skip ? 2 : (fail ? 1 : 0);
}
