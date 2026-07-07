// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * lut_tanh_rocket.c — gate / HW-RE harness for the SIGNED-output DPU LUT path.
 *
 * tanh is the keystone signed test ([-1,1], odd): the table stores (tanh(x)+1)/2 in
 * the proven unsigned Q0.15 grid (the "bias trick") and the DPU OUT_CVT affine-decodes
 * tanh = (q - 16384) * 2^-14 — a SIGNED, pre-shift OUT_CVT_OFFSET. Confirming this on HW
 * pins the output-converter semantics that the full-range single-pass HardSwish/SiLU LUT
 * (the conv->activation fusion perf path) needs.
 *
 * The OUT_CVT affine is env-overridable in rocket_activation.c (ROCKET_LUT_OFFSET /
 * _MINEXP / _SCALE / _BNMUL), so this same binary doubles as the RE sweep:
 *   ROCKET_LUT_OFFSET=-16384 ROCKET_LUT_MINEXP=14 ./lut_tanh_rocket
 *
 * Exit: 0 PASS, 1 FAIL, 2 no NPU (skipped HW).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

/* fp16 LUT over a signed/wide range: tanh is smooth+bounded (tight); hardswish/silu
 * span a wider output (S=16, coarser Q step ~5e-4) and have a sharper knee, so a looser
 * gate. Both are far below the gated-path host-mul tol (0.03). */
#define TANH_TOL 0.01
#define WIDE_TOL 0.02

static int run_kind(int fd, int kind, double x0, double x1, double tol, int n)
{
    _Float16 *in  = malloc((size_t)n * sizeof(_Float16));
    _Float16 *out = malloc((size_t)n * sizeof(_Float16));
    _Float16 *ref = malloc((size_t)n * sizeof(_Float16));
    if (!in || !out || !ref) { free(in); free(out); free(ref); return 1; }

    for (int i = 0; i < n; i++)
        in[i] = (_Float16)(x0 + (x1 - x0) * ((double)i / (double)(n - 1)));

    memset(out, 0, (size_t)n * sizeof(_Float16));
    int r = rocket_activation_fp16(fd, kind, in, out, n);
    if (r) { printf("  %-9s n=%-5d rocket_activation_fp16=%d -> FAIL\n",
                    rocket_activation_name(kind), n, r);
             free(in); free(out); free(ref); return 1; }

    rocket_activation_ref_fp16(kind, in, ref, n);
    double max_abs = 0; int iworst = 0;
    for (int i = 0; i < n; i++) {
        double ad = fabs((double)(float)out[i] - (double)(float)ref[i]);
        if (ad > max_abs) { max_abs = ad; iworst = i; }
    }
    if (getenv("ACT_DUMP")) {           /* list every point that exceeds tol (cap 24) */
        int shown = 0;
        for (int i = 0; i < n && shown < 24; i++) {
            double ad = fabs((double)(float)out[i] - (double)(float)ref[i]);
            if (ad > tol) {
                printf("    [bad i=%d] x=%+8.4f got=%+12.4f want=%+12.4f\n",
                       i, (double)(float)in[i], (double)(float)out[i], (double)(float)ref[i]);
                shown++;
            }
        }
        if (!shown) printf("    (no points exceed tol)\n");
    }
    int pass = max_abs <= tol;
    printf("  %-9s n=%-5d max_abs=%.5f @i=%d x=%+.3f (got %+.4f want %+.4f, tol=%.3f) -> %s\n",
           rocket_activation_name(kind), n, max_abs, iworst, (double)(float)in[iworst],
           (double)(float)out[iworst], (double)(float)ref[iworst], tol, pass ? "PASS" : "FAIL");
    fflush(stdout);
    free(in); free(out); free(ref);
    return pass ? 0 : 1;
}

int main(void)
{
    const char *o = getenv("ROCKET_LUT_OFFSET"), *m = getenv("ROCKET_LUT_MINEXP");
    const char *s = getenv("ROCKET_LUT_SCALE"), *b = getenv("ROCKET_LUT_BNMUL");
    if (o || m || s || b)
        printf("OUT_CVT sweep override: offset=%s minexp=%s scale=%s bnmul=%s\n",
               o ? o : "-", m ? m : "-", s ? s : "-", b ? b : "-");

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP HW\n", fd); return 2; }

    int fail = 0;
    /* tanh: the keystone signed test (bias trick + signed pre-shift OUT_CVT_OFFSET). */
    fail |= run_kind(fd, ROCKET_ACTIVATION_TANH, -10.0, 10.0, TANH_TOL, 128);
    fail |= run_kind(fd, ROCKET_ACTIVATION_TANH, -10.0, 10.0, TANH_TOL, 1024);
    /* SiLU via the SINGLE-PASS wide-output LUT (one DPU job, no EW mul / 2nd round-trip).
     * Force that path (vs the default 2-pass gate+mul) and validate vs the CPU ref. NOTE:
     * HARDSWISH's wide-LUT is NOT gated here — its exactly-flat LE tail reads back garbage
     * on the DPU LUT (characterized on the DPU LUT); HardSwish keeps the
     * proven 2-pass path (gated in activation_lut_rocket.c). */
    setenv("ROCKET_ACT_WIDE_LUT", "1", 1);
    fail |= run_kind(fd, ROCKET_ACTIVATION_SILU, -8.0, 8.0, WIDE_TOL, 1024);
    /* hardswish single-pass via the KNEE-ONLY table (NVDLA flat-run mitigation): over the
     * curved knee [-3,3] the table has no constant run, so it is correct (GATED). Beyond
     * +-3 the deep-negative flat-0 region is LE extrapolation, which trips the NVDLA LE/LO
     * overflow mux (the asymptotic-saturation quirk) -> informational only; full-range
     * HardSwish stays on the proven 2-pass gate+mul / host path. */
    fail |= run_kind(fd, ROCKET_ACTIVATION_HARDSWISH, -3.0, 3.0, WIDE_TOL, 1024);
    printf("  [informational] full-range hardswish-wide (extrapolation tail trips the LE/LO mux):\n");
    run_kind(fd, ROCKET_ACTIVATION_HARDSWISH, -8.0, 8.0, WIDE_TOL, 1024);
    unsetenv("ROCKET_ACT_WIDE_LUT");

    rocket_close(fd);
    printf("%s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
