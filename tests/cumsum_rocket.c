// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * cumsum_rocket.c — HW gate for on-NPU CUMSUM (prefix sum along the last axis,
 * rocket_cumsum_fp16): out[m][j] = sum over the prefix-j input columns of row m.
 *
 * Cumsum is the feature-axis reduce widened from a single ones-COLUMN to a full
 * triangular ones MATRIX (out = in · L^T), so it reuses the validated fp32-output matmul
 * (rocket_matmul_fp16_f32out) and inherits its genuine fp32 K-accumulation — important
 * because a long prefix sums many terms. The fp32 result is narrowed to fp16 on read-back,
 * so this is fp16-rounding-accurate (relative tolerance), NOT bit-exact.
 *
 * Layers:
 *  1. REFERENCE self-check (anywhere, no NPU): the fp64-accumulate oracle agrees with an
 *     independent forward/reverse re-derivation and the inclusive/exclusive identity
 *     (inclusive[j] - exclusive[j] == in[j]) — confirms the gate's own yardstick.
 *  2. ON-HARDWARE end-to-end (only if /dev/accel/accel0 opens): the NPU triangular-matmul
 *     scan vs the fp64 host oracle, for all four variants (inclusive/exclusive x forward/
 *     reverse), across the M-tile boundary (Mt caps at 256), at N-not-%32 widths, and at a
 *     large-N (T=1500) Whisper/LM sequence length. Pass = no element wrong in BOTH relative
 *     and absolute terms (the false-green-audit two-metric criterion; a prefix passing
 *     through zero stays small in abs, real layout/readback corruption is large in both).
 *
 * Usage: cumsum_rocket            (sweep)
 *        cumsum_rocket M N        (one shape; all four variants)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_reduce.h"

static int g_fail = 0;

/* deterministic pseudo-random fp16 input in [-amp, amp], per shape */
static void fill(_Float16 *in, size_t n, float amp, uint32_t seed)
{
    uint32_t st = 0x9e3779b9u ^ seed;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)((st >> 8) & 0xffff) / 65535.f;   /* [0,1] */
        in[i] = (_Float16)((u * 2.f - 1.f) * amp);
    }
}

static const char *vname(int exclusive, int reverse)
{
    return reverse ? (exclusive ? "excl-rev" : "incl-rev")
                   : (exclusive ? "excl-fwd" : "incl-fwd");
}

/* one (M,N, exclusive, reverse) HW check vs the fp64 oracle */
static int check(int fd, int M, int N, const _Float16 *in, int exclusive, int reverse)
{
    size_t MN = (size_t)M * N;
    _Float16 *got = malloc(MN * sizeof(_Float16));
    _Float16 *ref = malloc(MN * sizeof(_Float16));
    if (!got || !ref) { fprintf(stderr, "oom\n"); free(got); free(ref); return 1; }

    rocket_cumsum_ref_fp16(M, N, in, ref, exclusive, reverse);
    int r = rocket_cumsum_fp16(fd, M, N, in, got, exclusive, reverse);
    if (r) { printf("  M=%d N=%d %-8s: call=%d -> FAIL\n", M, N, vname(exclusive, reverse), r);
             free(got); free(ref); return 1; }

    /* value scale of this shape; abs threshold is relative to it so a prefix passing through
     * zero (cancellation) doesn't inflate the relative error into a false FAIL. */
    double maxref = 0;
    for (size_t i = 0; i < MN; i++) if (fabs((double)ref[i]) > maxref) maxref = fabs((double)ref[i]);

    const double REL_TOL = 0.02;                 /* fp32-accum + single fp16 narrow ~5e-4; 2% is margin */
    const double ABS_TOL = REL_TOL * maxref + 1e-2;
    double max_abs = 0, max_rel = 0; int bad = 0;
    for (size_t i = 0; i < MN; i++) {
        double ad = fabs((double)got[i] - (double)ref[i]);
        double rd = ad / (fabs((double)ref[i]) + 1e-9);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (rd > REL_TOL && ad > ABS_TOL) {       /* wrong in BOTH metrics */
            if (bad < 5) printf("    [m=%zu n=%zu] ref=%.5g got=%.5g d=%.4g\n",
                                i / N, i % N, (double)ref[i], (double)got[i], ad);
            bad++;
        }
    }
    int ok = (bad == 0);
    printf("  M=%d N=%d %-8s: maxref=%.4g max_abs=%.4g max_rel=%.2g bad=%d -> %s\n",
           M, N, vname(exclusive, reverse), maxref, max_abs, max_rel, bad, ok ? "PASS" : "FAIL");
    free(got); free(ref);
    return ok ? 0 : 1;
}

static int run_shape(int fd, int M, int N, float amp)
{
    size_t n = (size_t)M * N;
    _Float16 *in = malloc(n * sizeof(_Float16));
    if (!in) { fprintf(stderr, "oom\n"); return 1; }
    fill(in, n, amp, (uint32_t)(M * 2654435761u + N * 40503u + (uint32_t)(amp * 7)));
    printf("[M=%d N=%d amp=%.0f]\n", M, N, amp);
    int fail = 0;
    fail |= check(fd, M, N, in, 0, 0);   /* inclusive forward  */
    fail |= check(fd, M, N, in, 1, 0);   /* exclusive forward  */
    fail |= check(fd, M, N, in, 0, 1);   /* inclusive reverse  */
    fail |= check(fd, M, N, in, 1, 1);   /* exclusive reverse  */
    free(in);
    return fail;
}

/* layer 1: the running-accumulator oracle agrees with an INDEPENDENT O(N^2) recompute-from-
 * scratch (every prefix re-summed in fp64 directly from the prefix-membership rule). This
 * catches an off-by-one prefix boundary (which would differ by a whole element ~amp), while
 * tolerating the fp16 narrowing of each output (<=0.5 ULP, ~0.016 at the ~40 magnitudes here). */
static int ref_selfcheck(void)
{
    const int M = 5, N = 130;
    _Float16 *in  = malloc((size_t)M * N * sizeof(_Float16));
    _Float16 *got = malloc((size_t)M * N * sizeof(_Float16));
    if (!in || !got) { free(in); free(got); return 1; }
    fill(in, (size_t)M * N, 3.f, 123);

    int bad = 0;
    for (int variant = 0; variant < 4; variant++) {
        int excl = variant & 1, rev = (variant >> 1) & 1;
        rocket_cumsum_ref_fp16(M, N, in, got, excl, rev);
        for (int m = 0; m < M; m++)
            for (int j = 0; j < N; j++) {
                double acc = 0.0;                              /* independent direct prefix sum */
                for (int i = 0; i < N; i++) {
                    int inpref = rev ? (excl ? (i > j) : (i >= j))
                                     : (excl ? (i < j) : (i <= j));
                    if (inpref) acc += (double)in[(size_t)m*N+i];
                }
                if (fabs((double)got[(size_t)m*N+j] - acc) > 0.05) bad++;
            }
    }
    free(in); free(got);
    printf("ref self-check (vs independent O(N^2) fp64 recompute, 4 variants): %s\n\n",
           bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — ref self-check + SKIP\n\n", fd);

    g_fail |= ref_selfcheck();

    if (argc == 3) {
        if (fd >= 0) g_fail |= run_shape(fd, atoi(argv[1]), atoi(argv[2]), 4.f);
        if (fd >= 0) rocket_close(fd);
        printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : (fd < 0 ? 2 : 0);
    }

    if (fd >= 0) {
        /* M, N, amplitude */
        g_fail |= run_shape(fd,   4,   64, 4.f);    /* tiny, single M-tile, N%32==0        */
        g_fail |= run_shape(fd, 256,  512, 4.f);    /* exactly the M-tile cap (256)         */
        g_fail |= run_shape(fd, 260,  512, 4.f);    /* just past the M-tile boundary        */
        g_fail |= run_shape(fd,   8,  100, 4.f);    /* N%32!=0 -> K-pad to 128, N-pad to 112*/
        g_fail |= run_shape(fd,   5,   48, 4.f);    /* M%4!=0 AND N%32!=0 -> both pads       */
        g_fail |= run_shape(fd,   4, 1500, 4.f);    /* Whisper/LM seq (T=1500, long prefix)  */
        g_fail |= run_shape(fd,  16,  768, 4.f);    /* realistic width                       */
        g_fail |= run_shape(fd,   8,  256, 1.f);    /* small magnitude (all-positive-ish tail)*/
    }

    if (fd >= 0) rocket_close(fd);
    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : (fd < 0 ? 2 : 0);
}
