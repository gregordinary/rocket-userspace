// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * cross_entropy_rocket.c — HW gate for stable per-row CROSS-ENTROPY
 * (rocket_cross_entropy_fp16): CE[m] = logsumexp(logits[m]) - logits[m][target[m]], the
 * softmax-classifier / language-model NLL loss in its numerically-stable form (never
 * materializes softmax).
 *
 * The on-NPU work is the LogSoftmax front half (host row-max + subtract -> NPU exp ->
 * NPU fp32 row-sum); the tail is a HOST GATHER + subtract (there is NO hardware gather on
 * the RK3588 NPU — the gather is M scalar index-lookups, like softmax's host 1/s). So this
 * gate also exercises the EXP LUT + the feature-axis reduce end to end and checks them
 * against an fp64 CE oracle.
 *
 * Layers:
 *  1. REFERENCE self-check (anywhere, no NPU): the fp64 CE oracle equals
 *     -logsoftmax_ref(logits)[target] (an INDEPENDENT code path — the LogSoftmax reference),
 *     and CE >= 0 — confirms the gate's own yardstick.
 *  2. ON-HARDWARE end-to-end (only if /dev/accel/accel0 opens): the NPU CE vs the fp64 oracle
 *     across the M-tile boundary (Mt caps at 256), at N-not-%32 widths, a large-N (T=1500)
 *     LM vocab, random targets per row (plus one forced argmax target -> the smallest CE,
 *     exercising the non-negativity boundary), and a wide-score-spread case. CE is all-additive
 *     -> ABSOLUTE error (the dominant term is the EXP-LUT bias in log(s), a per-row offset),
 *     plus the CE >= 0 loss invariant.
 *
 * Usage: cross_entropy_rocket            (sweep)
 *        cross_entropy_rocket M N        (one shape)
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

/* deterministic random class targets in [0,N); row 0 forced to its own argmax (smallest CE) */
static void make_targets(const _Float16 *x, int M, int N, int *tgt, uint32_t seed)
{
    uint32_t st = 0x85ebca6bu ^ seed;
    for (int m = 0; m < M; m++) { st = st * 1664525u + 1013904223u; tgt[m] = (int)((st >> 8) % (uint32_t)N); }
    int amax = 0; for (int n = 1; n < N; n++) if (x[n] > x[amax]) amax = n;
    tgt[0] = amax;
}

static int test_ce(int fd, int M, int N, float amp)
{
    size_t MN = (size_t)M * N;
    _Float16 *x  = malloc(MN * sizeof(_Float16));
    int      *t  = malloc((size_t)M * sizeof(int));
    float    *got= malloc((size_t)M * sizeof(float));
    float    *ref= malloc((size_t)M * sizeof(float));
    if (!x || !t || !got || !ref) { fprintf(stderr,"oom\n"); free(x);free(t);free(got);free(ref); return 1; }
    fill(x, MN, amp, (uint32_t)(M*131 + N*17 + (uint32_t)amp));
    make_targets(x, M, N, t, (uint32_t)(M*7 + N*3));

    rocket_cross_entropy_ref_fp16(M, N, x, t, ref);
    int rc = rocket_cross_entropy_fp16(fd, M, N, x, t, got);

    char tag[80]; snprintf(tag, sizeof tag, "CE M=%d N=%d amp=%.0f", M, N, amp);
    if (rc) { printf("  %s: call=%d -> FAIL\n", tag, rc); free(x);free(t);free(got);free(ref); return 1; }

    /* CE is all-additive -> absolute error; plus the CE>=0 loss invariant (allow one abs_tol
     * of LUT slack at the argmax row where CE == log(s) ~ 0). */
    const double abs_tol = 0.06;
    double max_abs = 0, maxce = 0; int bad = 0, neg = 0;
    for (int m = 0; m < M; m++) {
        double ad = fabs((double)got[m] - (double)ref[m]);
        if (ad > max_abs) max_abs = ad;
        if (fabs((double)ref[m]) > maxce) maxce = fabs((double)ref[m]);
        if (got[m] < -abs_tol) { if (neg < 3) printf("    [m=%d] CE<0: got=%.5g\n", m, got[m]); neg++; }
        if (ad > abs_tol) { if (bad < 5) printf("    [m=%d] ref=%.5g got=%.5g d=%.4g (tgt=%d)\n",
                                                m, ref[m], got[m], ad, t[m]); bad++; }
    }
    int ok = (bad == 0) && (neg == 0);
    printf("  %s: maxce=%.4g max_abs=%.4g bad=%d neg=%d -> %s\n",
           tag, maxce, max_abs, bad, neg, ok ? "PASS" : "FAIL");
    free(x); free(t); free(got); free(ref);
    return ok ? 0 : 1;
}

/* host reference self-check: CE == -logsoftmax[target] (independent path) and CE >= 0 */
static int ref_selfcheck(void)
{
    const int M = 6, N = 200;
    _Float16 *x  = malloc((size_t)M*N*sizeof(_Float16));
    _Float16 *ls = malloc((size_t)M*N*sizeof(_Float16));
    float    *ce = malloc((size_t)M*sizeof(float));
    int      *t  = malloc((size_t)M*sizeof(int));
    if (!x||!ls||!ce||!t){ free(x);free(ls);free(ce);free(t); return 1; }
    fill(x, (size_t)M*N, 5.f, 9);
    for (int m = 0; m < M; m++) t[m] = (m*37 + 5) % N;

    rocket_logsoftmax_ref_fp16(M, N, x, ls);
    rocket_cross_entropy_ref_fp16(M, N, x, t, ce);
    int bad = 0;
    for (int m = 0; m < M; m++) {
        double expect = -(double)ls[(size_t)m*N + t[m]];   /* -logsoftmax[target] == CE */
        if (fabs((double)ce[m] - expect) > 1e-2) bad++;
        if (ce[m] < -1e-3) bad++;                          /* CE >= 0 */
    }
    free(x);free(ls);free(ce);free(t);
    printf("ref self-check (CE == -logsoftmax[target], CE>=0): %s\n\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — ref self-check + SKIP\n\n", fd);

    g_fail |= ref_selfcheck();

    if (argc == 3) {
        if (fd >= 0) g_fail |= test_ce(fd, atoi(argv[1]), atoi(argv[2]), 4.f);
        if (fd >= 0) rocket_close(fd);
        printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : (fd < 0 ? 2 : 0);
    }

    if (fd >= 0) {
        g_fail |= test_ce(fd,  16,   64, 4.f);
        g_fail |= test_ce(fd, 256,  512, 4.f);   /* M-tile boundary */
        g_fail |= test_ce(fd, 260,  512, 4.f);
        g_fail |= test_ce(fd,   4, 1500, 4.f);   /* large vocab / Whisper-LM seq (T=1500) */
        g_fail |= test_ce(fd,  64, 1500, 4.f);
        g_fail |= test_ce(fd,   5,  100, 4.f);   /* small M, N%32!=0 */
        g_fail |= test_ce(fd,  32,  500, 20.f);  /* wide score spread -> EXP deep-tail clamp */
        rocket_close(fd);
    }

    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : (fd < 0 ? 2 : 0);
}
