// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * layernorm_rocket.c — HW gate for on-NPU LayerNorm (rocket_layernorm_fp16),
 * Whisper-encoder normalization. out = (x-mean)/sqrt(var+eps)*gamma + beta, per row over H.
 *
 * The chain runs the O(M*H) work on the NPU (square via ew_mul, BOTH the sum(x) and sum(x^2)
 * reductions in ONE stacked feature-reduce job, the affine as one ew_mul + one ew_add) and the
 * O(M) per-row tail (mean, var, rsqrt) exact on the host. The gate checks against an fp64
 * oracle across the M-tile boundary, realistic Whisper widths (d_model 512), the small-M /
 * H%32!=0 cases, AND a large-magnitude input that exercises the fp16-square overflow prescale.
 *
 * Usage: layernorm_rocket            (sweep)
 *        layernorm_rocket M H        (one shape)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_norm.h"

static int g_fail = 0;

static void fill(_Float16 *v, size_t n, float amp, uint32_t seed)
{
    uint32_t st = 0x9e3779b9u ^ seed;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)((st >> 8) & 0xffff) / 65535.f;
        v[i] = (_Float16)((u * 2.f - 1.f) * amp);
    }
}

static int compare(const char *tag, const _Float16 *got, const _Float16 *ref,
                   size_t n, double rel_tol)
{
    double maxv = 0;
    for (size_t i = 0; i < n; i++) { double a = fabs((double)ref[i]); if (a > maxv) maxv = a; }
    const double abs_tol = rel_tol * maxv + 1e-4;
    double max_abs = 0, max_rel = 0; int bad = 0;
    for (size_t i = 0; i < n; i++) {
        double ad = fabs((double)got[i] - (double)ref[i]);
        double rd = ad / (fabs((double)ref[i]) + 1e-9);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (rd > rel_tol && ad > abs_tol) {
            if (bad < 5) printf("    [%zu] ref=%.5g got=%.5g d=%.4g\n", i, (double)ref[i], (double)got[i], ad);
            bad++;
        }
    }
    int ok = (bad == 0);
    printf("  %s: maxv=%.4g max_abs=%.4g max_rel=%.2g bad=%d -> %s\n",
           tag, maxv, max_abs, max_rel, bad, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_layernorm(int fd, int M, int H, float amp, int use_beta)
{
    size_t MH = (size_t)M * H;
    _Float16 *x = malloc(MH * sizeof(_Float16));
    _Float16 *g = malloc((size_t)H * sizeof(_Float16));
    _Float16 *b = malloc((size_t)H * sizeof(_Float16));
    _Float16 *got = malloc(MH * sizeof(_Float16));
    _Float16 *ref = malloc(MH * sizeof(_Float16));
    if (!x||!g||!b||!got||!ref){ fprintf(stderr,"oom\n"); free(x);free(g);free(b);free(got);free(ref); return 1; }
    fill(x, MH, amp, (uint32_t)(M*131 + H*17 + (uint32_t)amp));
    for (int h = 0; h < H; h++) g[h] = (_Float16)(0.5f + ((h * 53 % 100) / 99.f));   /* [0.5,1.5] */
    for (int h = 0; h < H; h++) b[h] = (_Float16)(((h * 29 % 100) / 99.f) - 0.5f);   /* [-0.5,0.5] */
    const float eps = 1e-5f;
    const _Float16 *beta = use_beta ? b : NULL;

    rocket_layernorm_ref_fp16(M, H, x, g, beta, eps, ref);
    int rc = rocket_layernorm_fp16(fd, M, H, x, g, beta, eps, got);
    char tag[96]; snprintf(tag, sizeof tag, "layernorm M=%d H=%d amp=%.0f beta=%d", M, H, amp, use_beta);
    int fail = 0;
    if (rc) { printf("  %s: call=%d -> FAIL\n", tag, rc); fail = 1; }
    else    fail = compare(tag, got, ref, MH, 0.03);
    free(x); free(g); free(b); free(got); free(ref);
    return fail;
}

/* host reference self-check: with gamma==1, beta==0, each output row has mean~0 and var~1 */
static int ref_selfcheck(void)
{
    const int M = 4, H = 256;
    _Float16 *x = malloc((size_t)M*H*sizeof(_Float16)), *g = malloc(H*sizeof(_Float16));
    _Float16 *o = malloc((size_t)M*H*sizeof(_Float16));
    if (!x||!g||!o){ free(x);free(g);free(o); return 1; }
    fill(x, (size_t)M*H, 3.f, 5); for (int h=0;h<H;h++) g[h]=(_Float16)1.0f;
    rocket_layernorm_ref_fp16(M, H, x, g, NULL, 0.f, o);
    int bad = 0;
    for (int m = 0; m < M; m++) {
        double sm=0, ss=0; for (int h=0;h<H;h++){ double v=(double)o[(size_t)m*H+h]; sm+=v; ss+=v*v; }
        double mean=sm/H, var=ss/H-mean*mean;
        if (fabs(mean) > 0.02 || fabs(var - 1.0) > 0.05) bad++;
    }
    free(x);free(g);free(o);
    printf("ref self-check (out mean~0 var~1 when gamma==1): %s\n\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — ref self-check + SKIP\n\n", fd);

    g_fail |= ref_selfcheck();

    if (argc == 3) {
        if (fd >= 0) g_fail |= test_layernorm(fd, atoi(argv[1]), atoi(argv[2]), 4.f, 1);
        if (fd >= 0) rocket_close(fd);
        printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : (fd < 0 ? 2 : 0);
    }

    if (fd >= 0) {
        g_fail |= test_layernorm(fd,  16,   64, 4.f, 1);
        g_fail |= test_layernorm(fd, 256,  512, 4.f, 1);   /* M-tile boundary, Whisper d_model */
        g_fail |= test_layernorm(fd, 260,  512, 4.f, 1);
        g_fail |= test_layernorm(fd,  64,  512, 4.f, 0);   /* no beta */
        g_fail |= test_layernorm(fd,  12,  100, 4.f, 1);   /* small M, H%32!=0 */
        g_fail |= test_layernorm(fd,  32, 1024, 1000.f, 1);/* large |x| -> fp16-square prescale */
        rocket_close(fd);
    }

    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : (fd < 0 ? 2 : 0);
}
