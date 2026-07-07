// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * softmax_rocket.c — HW gate for on-NPU row-wise softmax (rocket_softmax_fp16), the attention
 * nonlinearity of the Whisper/transformer encoder.
 *
 * softmax(x)[m][n] = exp(x-rowmax) / sum exp(x-rowmax). The chain runs exp (DPU LUT) + the
 * row-sum (feature-axis ones-matmul reduce) + the per-row scale on the NPU; the row-max and the
 * O(M) reciprocal are on the host. The gate checks the output distribution against an fp64
 * oracle across the M-tile boundary, realistic Whisper seq lengths (T up to 1500), and a
 * wide-score-spread case (exercises the EXP deep-tail clamp). Two-metric pass (rel where the
 * probability is non-tiny, abs floor for the small probs) PLUS a row-sum==1 check.
 *
 * Usage: softmax_rocket            (sweep)
 *        softmax_rocket M N        (one shape)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_softmax.h"

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

static int test_softmax(int fd, int M, int N, float amp)
{
    size_t MN = (size_t)M * N;
    _Float16 *x  = calloc(MN, sizeof(_Float16));
    _Float16 *got= malloc(MN * sizeof(_Float16));
    _Float16 *ref= malloc(MN * sizeof(_Float16));
    if (!x || !got || !ref) { fprintf(stderr,"oom\n"); free(x);free(got);free(ref); return 1; }
    fill(x, MN, amp, (uint32_t)(M*131 + N*17 + (uint32_t)amp));

    rocket_softmax_ref_fp16(M, N, x, ref);
    int rc = rocket_softmax_fp16(fd, M, N, x, got);

    char tag[80]; snprintf(tag, sizeof tag, "softmax M=%d N=%d amp=%.0f", M, N, amp);
    if (rc) { printf("  %s: call=%d -> FAIL\n", tag, rc); free(x);free(got);free(ref); return 1; }

    /* element accuracy: rel where prob is non-tiny, abs floor otherwise */
    const double rel_tol = 0.03, abs_tol = 2e-3;
    double max_abs = 0, max_rel = 0; int bad = 0;
    for (size_t i = 0; i < MN; i++) {
        double rf = (double)ref[i], gt = (double)got[i];
        double ad = fabs(gt - rf), rd = ad / (rf + 1e-6);
        if (ad > max_abs) max_abs = ad;
        if (rf > 1e-3 && rd > max_rel) max_rel = rd;
        if (ad > abs_tol && (rf <= 1e-3 || rd > rel_tol)) {
            if (bad < 5) printf("    [%zu] ref=%.5g got=%.5g d=%.4g\n", i, rf, gt, ad);
            bad++;
        }
    }
    /* each row must sum to ~1 */
    double worst_dev = 0, worst_sum = 1.0;
    for (int m = 0; m < M; m++) {
        double sg = 0; for (int n = 0; n < N; n++) sg += (double)got[(size_t)m*N+n];
        if (fabs(sg - 1.0) > worst_dev) { worst_dev = fabs(sg - 1.0); worst_sum = sg; }
    }
    int ok = (bad == 0) && (fabs(worst_sum - 1.0) <= 0.01);
    printf("  %s: max_abs=%.4g max_rel[p>1e-3]=%.2g bad=%d worst_rowsum=%.5f -> %s\n",
           tag, max_abs, max_rel, bad, worst_sum, ok ? "PASS" : "FAIL");
    free(x); free(got); free(ref);
    return ok ? 0 : 1;
}

static int test_logsoftmax(int fd, int M, int N, float amp)
{
    size_t MN = (size_t)M * N;
    _Float16 *x  = calloc(MN, sizeof(_Float16));
    _Float16 *got= malloc(MN * sizeof(_Float16));
    _Float16 *ref= malloc(MN * sizeof(_Float16));
    if (!x || !got || !ref) { fprintf(stderr,"oom\n"); free(x);free(got);free(ref); return 1; }
    fill(x, MN, amp, (uint32_t)(M*149 + N*23 + (uint32_t)amp));

    rocket_logsoftmax_ref_fp16(M, N, x, ref);
    int rc = rocket_logsoftmax_fp16(fd, M, N, x, got);

    char tag[80]; snprintf(tag, sizeof tag, "logsoftmax M=%d N=%d amp=%.0f", M, N, amp);
    if (rc) { printf("  %s: call=%d -> FAIL\n", tag, rc); free(x);free(got);free(ref); return 1; }

    /* LogSoftmax is all-additive (out<=0) -> ABSOLUTE error is the right metric (no tiny-prob
     * relative blow-up). Error = EXP-LUT bias in log(s) (a per-row offset) + fp16 storage of the
     * large-magnitude low-prob entries. Also verify the identity sum_n exp(out) == 1. */
    double max_abs = 0; int bad = 0;
    for (size_t i = 0; i < MN; i++) {
        double ad = fabs((double)got[i] - (double)ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad > 0.05) { if (bad < 5) printf("    [%zu] ref=%.5g got=%.5g d=%.4g\n",
                                             i, (double)ref[i], (double)got[i], ad); bad++; }
    }
    double worst_dev = 0, worst_sum = 1.0;
    for (int m = 0; m < M; m++) {
        double se = 0; for (int n = 0; n < N; n++) se += exp((double)got[(size_t)m*N+n]);
        if (fabs(se - 1.0) > worst_dev) { worst_dev = fabs(se - 1.0); worst_sum = se; }
    }
    int ok = (bad == 0) && (fabs(worst_sum - 1.0) <= 0.03);
    printf("  %s: max_abs=%.4g bad=%d worst_sum_exp=%.5f -> %s\n",
           tag, max_abs, bad, worst_sum, ok ? "PASS" : "FAIL");
    free(x); free(got); free(ref);
    return ok ? 0 : 1;
}

/* host reference self-check: every row sums to exactly 1 (within fp16) */
static int ref_selfcheck(void)
{
    const int M = 4, N = 200;
    _Float16 *x = malloc((size_t)M*N*sizeof(_Float16)), *o = malloc((size_t)M*N*sizeof(_Float16));
    if (!x||!o){ free(x);free(o); return 1; }
    fill(x, (size_t)M*N, 5.f, 9);
    rocket_softmax_ref_fp16(M, N, x, o);
    int bad = 0;
    for (int m = 0; m < M; m++) {
        double s = 0; for (int n=0;n<N;n++) s += (double)o[(size_t)m*N+n];
        if (fabs(s - 1.0) > 0.02) bad++;
    }
    free(x);free(o);
    printf("ref self-check (rows sum to 1): %s\n\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — ref self-check + SKIP\n\n", fd);

    g_fail |= ref_selfcheck();

    if (argc == 3) {
        if (fd >= 0) g_fail |= test_softmax(fd, atoi(argv[1]), atoi(argv[2]), 4.f);
        if (fd >= 0) rocket_close(fd);
        printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : (fd < 0 ? 2 : 0);
    }

    if (fd >= 0) {
        g_fail |= test_softmax(fd,  16,   64, 4.f);
        g_fail |= test_softmax(fd, 256,  512, 4.f);   /* M-tile boundary */
        g_fail |= test_softmax(fd, 260,  512, 4.f);
        g_fail |= test_softmax(fd,   4, 1500, 4.f);   /* Whisper encoder seq (T=1500) */
        g_fail |= test_softmax(fd,  64, 1500, 4.f);
        g_fail |= test_softmax(fd,   5,  100, 4.f);   /* small M, N%32!=0 */
        g_fail |= test_softmax(fd,  32,  500, 20.f);  /* wide score spread -> EXP deep-tail clamp */

        g_fail |= test_logsoftmax(fd,  16,   64, 4.f);
        g_fail |= test_logsoftmax(fd, 256,  512, 4.f);  /* M-tile boundary */
        g_fail |= test_logsoftmax(fd,   4, 1500, 4.f);  /* Whisper/LM seq length */
        g_fail |= test_logsoftmax(fd,   5,  100, 4.f);  /* small M, N%32!=0 */
        g_fail |= test_logsoftmax(fd,  32,  500, 20.f); /* wide spread -> very negative log-probs */
        rocket_close(fd);
    }

    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : (fd < 0 ? 2 : 0);
}
