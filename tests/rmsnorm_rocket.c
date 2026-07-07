// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rmsnorm_rocket.c — HW gate for on-NPU RMSNorm (rocket_rmsnorm_fp16) and the per-row
 * broadcast-scale primitive (rocket_scale_rows_fp16), the on-NPU normalization building blocks.
 *
 * RMSNorm = x / sqrt(mean_h(x^2)+eps) * weight[h]. The chain runs the O(M*H) work on the NPU
 * (square via DPU ew_mul, the H-contraction via the feature-axis ones-matmul reduce, the final
 * scale via ew_mul) and the O(M) per-row tail (mean,+eps,rsqrt) exact on the host. The gate
 * checks both ops against an fp64 oracle across the M-tile boundary, realistic Gemma hidden
 * widths, the small-M / H%32!=0 cases, AND a large-magnitude input that exercises the
 * fp16-square overflow prescale (|x|>>223). Pass = no element wrong in BOTH rel and abs
 * (false-green-audit two-metric criterion; the per-row square-rounding is a benign ~1e-3).
 *
 * Usage: rmsnorm_rocket            (sweep)
 *        rmsnorm_rocket M H        (one RMSNorm shape)
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

/* compare got vs ref with the two-metric AND criterion, scaled to the output magnitude */
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

static int test_scale_rows(int fd, int M, int N)
{
    size_t MN = (size_t)M * N;
    _Float16 *in = malloc(MN * sizeof(_Float16));
    _Float16 *got = malloc(MN * sizeof(_Float16));
    _Float16 *ref = malloc(MN * sizeof(_Float16));
    float *r = malloc((size_t)M * sizeof(float));
    if (!in || !got || !ref || !r) { fprintf(stderr,"oom\n"); free(in);free(got);free(ref);free(r); return 1; }
    fill(in, MN, 3.f, (uint32_t)(M*7 + N));
    for (int m = 0; m < M; m++) r[m] = 0.25f + 2.0f * ((m * 37 % 100) / 99.f);   /* [0.25, 2.25] */

    rocket_scale_rows_ref_fp16(M, N, in, r, ref);
    int rc = rocket_scale_rows_fp16(fd, M, N, in, r, got);
    char tag[64]; snprintf(tag, sizeof tag, "scale_rows M=%d N=%d", M, N);
    int fail = 0;
    if (rc) { printf("  %s: call=%d -> FAIL\n", tag, rc); fail = 1; }
    else    fail = compare(tag, got, ref, MN, 0.02);
    free(in); free(got); free(ref); free(r);
    return fail;
}

static int test_rmsnorm(int fd, int M, int H, float amp)
{
    size_t MH = (size_t)M * H;
    _Float16 *x = malloc(MH * sizeof(_Float16));
    _Float16 *w = malloc((size_t)H * sizeof(_Float16));
    _Float16 *got = malloc(MH * sizeof(_Float16));
    _Float16 *ref = malloc(MH * sizeof(_Float16));
    if (!x || !w || !got || !ref) { fprintf(stderr,"oom\n"); free(x);free(w);free(got);free(ref); return 1; }
    fill(x, MH, amp, (uint32_t)(M*131 + H*17 + (uint32_t)amp));
    for (int h = 0; h < H; h++) w[h] = (_Float16)(0.5f + ((h * 53 % 100) / 99.f));  /* [0.5,1.5] */
    const float eps = 1e-6f;

    rocket_rmsnorm_ref_fp16(M, H, x, w, eps, ref);
    int rc = rocket_rmsnorm_fp16(fd, M, H, x, w, eps, got);
    char tag[80]; snprintf(tag, sizeof tag, "rmsnorm M=%d H=%d amp=%.0f", M, H, amp);
    int fail = 0;
    if (rc) { printf("  %s: call=%d -> FAIL\n", tag, rc); fail = 1; }
    else    fail = compare(tag, got, ref, MH, 0.03);
    free(x); free(w); free(got); free(ref);
    return fail;
}

/* layer 1: host reference internal-consistency (no NPU) */
static int ref_selfcheck(void)
{
    const int M = 4, H = 128;
    _Float16 *x = malloc((size_t)M*H*sizeof(_Float16)), *w = malloc(H*sizeof(_Float16));
    _Float16 *o = malloc((size_t)M*H*sizeof(_Float16));
    if (!x||!w||!o){ free(x);free(w);free(o); return 1; }
    fill(x, (size_t)M*H, 2.f, 5); for (int h=0;h<H;h++) w[h]=(_Float16)1.0f;
    rocket_rmsnorm_ref_fp16(M, H, x, w, 0.f, o);
    /* with weight==1, the per-row RMS of the output must be ~1 */
    int bad = 0;
    for (int m = 0; m < M; m++) {
        double ss = 0; for (int h=0;h<H;h++){ double v=(double)o[(size_t)m*H+h]; ss+=v*v; }
        double rms = sqrt(ss/H);
        if (fabs(rms - 1.0) > 0.05) bad++;
    }
    free(x);free(w);free(o);
    printf("ref self-check (out RMS==1 when weight==1): %s\n\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — ref self-check + SKIP\n\n", fd);

    g_fail |= ref_selfcheck();

    if (argc == 3) {
        if (fd >= 0) g_fail |= test_rmsnorm(fd, atoi(argv[1]), atoi(argv[2]), 4.f);
        if (fd >= 0) rocket_close(fd);
        printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : (fd < 0 ? 2 : 0);
    }

    if (fd >= 0) {
        printf("-- scale_rows (per-row broadcast multiply) --\n");
        g_fail |= test_scale_rows(fd,  16,   64);
        g_fail |= test_scale_rows(fd, 256,  512);   /* M-tile boundary */
        g_fail |= test_scale_rows(fd, 260,  512);
        g_fail |= test_scale_rows(fd,   5,  100);   /* small M, N%32!=0 */
        printf("\n-- rmsnorm --\n");
        g_fail |= test_rmsnorm(fd,  16,   64, 4.f);
        g_fail |= test_rmsnorm(fd,  64, 2048, 4.f);
        g_fail |= test_rmsnorm(fd, 256, 3840, 4.f);   /* M-tile boundary, Gemma hidden */
        g_fail |= test_rmsnorm(fd, 512, 3840, 4.f);   /* realistic full (slower)        */
        g_fail |= test_rmsnorm(fd,  12,  100, 4.f);   /* small M, H%32!=0                */
        g_fail |= test_rmsnorm(fd,  32, 1024, 1000.f);/* large |x| -> fp16-square prescale */
    }

    if (fd >= 0) rocket_close(fd);
    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : (fd < 0 ? 2 : 0);
}
