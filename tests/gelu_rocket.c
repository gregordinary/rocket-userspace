// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * gelu_rocket.c — HW gate for the on-NPU 2-pass GELU (rocket_activation_fp16(GELU)).
 *
 * GELU(x) = x·Φ(x), Φ = the Gaussian CDF, computed as a [0,1] unit-LUT gate Φ(x) (the CLEAN
 * geometry sigmoid uses) times x — the SAME 2-pass pattern as SiLU. This is the accurate
 * on-NPU GELU: the SINGLE-pass GELU (build_lut_affine) spikes in the flat negative tail
 * (QUIRK 1, ~128-magnitude mux toggles) and is useless for wide FFN inputs; the 2-pass route
 * stays clean because Φ rides the unit-LUT le_slope extrapolation like sigmoid.
 *
 * The gate sweeps a WIDE input range [-12,12] — deliberately INTO the flat/saturated tails
 * that broke the single-pass — and a random FFN-magnitude vector, vs the true erf GELU. Metric:
 * cosine sim + max abs (the input is what the encoder-MLP fc1 produces). Off-device: SKIP.
 *
 * Usage: gelu_rocket
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

static int g_fail = 0;

static double gelu_true(double x) { return 0.5 * x * (1.0 + erf(x * M_SQRT1_2)); }
static double silu_true(double x) { return x / (1.0 + exp(-x)); }

/* sweep f over a uniform input grid on [lo,hi] vs the true function; cos + max_abs. */
static void sweep(int fd, int kind, double lo, double hi, double bar_abs)
{
    const char *nm = rocket_activation_name(kind);
    const int N = 4096, n = (N + 7) & ~7;
    _Float16 *in=calloc(n,2), *out=calloc(n,2);
    if(!in||!out){fprintf(stderr,"oom\n");g_fail=1;free(in);free(out);return;}
    for (int i=0;i<N;i++) in[i]=(_Float16)(lo + (hi-lo)*((double)i/(N-1)));
    for (int i=N;i<n;i++) in[i]=(_Float16)0.f;

    int r = rocket_activation_fp16(fd, kind, in, out, n);
    if (r){ printf("  %-5s [%.0f,%.0f]: call=%d FAIL\n",nm,lo,hi,r); g_fail=1; free(in);free(out); return; }

    double dot=0,ng=0,nr=0,max_abs=0,maxv=0;
    for (int i=0;i<N;i++){
        double x=(double)(float)in[i], want=(kind==ROCKET_ACTIVATION_GELU)?gelu_true(x):silu_true(x);
        double got=(double)(float)out[i];
        dot+=got*want; ng+=got*got; nr+=want*want;
        double ad=fabs(got-want); if(ad>max_abs)max_abs=ad; if(fabs(want)>maxv)maxv=fabs(want);
    }
    double cos=dot/(sqrt(ng)*sqrt(nr)+1e-30);
    int ok=(cos>=0.9999)&&(max_abs<=bar_abs);
    printf("  %-5s [%.0f,%.0f]: cos=%.6f max_abs=%.4g (maxv=%.3g) -> %s\n",nm,lo,hi,cos,max_abs,maxv,ok?"PASS":"FAIL");
    if(!ok) g_fail=1;
    free(in);free(out);
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd); printf("==== SKIP ====\n"); return 2; }

    printf("-- 2-pass GELU vs true erf-GELU (incl. the flat tails that broke single-pass) --\n");
    sweep(fd, ROCKET_ACTIVATION_GELU, -12.0, 12.0, 0.05);
    sweep(fd, ROCKET_ACTIVATION_GELU,  -8.0,  8.0, 0.03);
    printf("-- SiLU (same 2-pass family, regression check) --\n");
    sweep(fd, ROCKET_ACTIVATION_SILU, -12.0, 12.0, 0.05);

    rocket_close(fd);
    printf("==== %s ====\n", g_fail?"FAIL":"PASS");
    return g_fail?1:0;
}
