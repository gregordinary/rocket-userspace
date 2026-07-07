// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * reduce_feature_rocket.c — HW gate for the FEATURE-AXIS reduce
 * (rocket_reduce_feature_fp16): out[m] = sum_h in[m][h] (or the per-row mean).
 *
 * This is the primitive the transformer normalization/softmax blocks are built on, and
 * the one the PPU spatial reduce cannot supply (PPU pools [H,W] within a channel, never
 * across channels). It is realised as a ones-vector matmul that reuses the validated
 * fp32-output matmul (rocket_matmul_fp16_f32out), so it inherits genuine fp32 K-accum.
 *
 * Layers:
 *  1. REFERENCE self-check (anywhere, no NPU): the fp64-accumulate oracle agrees with an
 *     independent naive float accumulation — confirms the gate's own yardstick.
 *  2. ON-HARDWARE end-to-end (only if /dev/accel/accel0 opens): the NPU ones-matmul reduce
 *     vs the fp64 host oracle, across the M-tile boundary (Mt caps at 256), realistic Gemma
 *     hidden widths, K-not-%32 / M-not-%4 padding cases, and both sum and mean — at realistic
 *     and large activation magnitudes. Pass = no element wrong in BOTH relative and absolute
 *     terms (the false-green-audit two-metric criterion; near-cancellation sums stay small in
 *     abs, real layout/readback corruption is large in both).
 *
 * Usage: reduce_feature_rocket            (sweep)
 *        reduce_feature_rocket M H        (one shape; both sum and mean)
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
static void fill(_Float16 *in, int M, int H, float amp, uint32_t seed)
{
    uint32_t st = 0x9e3779b9u ^ seed;
    for (size_t i = 0; i < (size_t)M * H; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)((st >> 8) & 0xffff) / 65535.f;   /* [0,1] */
        in[i] = (_Float16)((u * 2.f - 1.f) * amp);
    }
}

/* one (M,H, mean) HW check vs the fp64 oracle */
static int check(int fd, int M, int H, const _Float16 *in, int mean)
{
    float *got = malloc((size_t)M * sizeof(float));
    float *ref = malloc((size_t)M * sizeof(float));
    if (!got || !ref) { fprintf(stderr, "oom\n"); free(got); free(ref); return 1; }

    rocket_reduce_feature_ref_fp16(M, H, in, ref, mean);
    int r = rocket_reduce_feature_fp16(fd, M, H, in, got, mean);
    if (r) { printf("  M=%d H=%d %-4s: call=%d -> FAIL\n", M, H, mean?"mean":"sum", r);
             free(got); free(ref); return 1; }

    /* value scale of this shape; abs threshold is relative to it so near-zero rows
     * (random-sign cancellation) don't inflate the relative error into a false FAIL. */
    double maxref = 0;
    for (int m = 0; m < M; m++) if (fabs(ref[m]) > maxref) maxref = fabs(ref[m]);

    const double REL_TOL = 0.01;                 /* fp32-out path is ~1e-6; 1% is huge margin */
    const double ABS_TOL = REL_TOL * maxref + 1e-6;
    double max_abs = 0, max_rel = 0; int bad = 0;
    for (int m = 0; m < M; m++) {
        double ad = fabs((double)got[m] - (double)ref[m]);
        double rd = ad / (fabs((double)ref[m]) + 1e-9);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (rd > REL_TOL && ad > ABS_TOL) {       /* wrong in BOTH metrics */
            if (bad < 5) printf("    [m=%d] ref=%.5g got=%.5g d=%.4g\n", m, ref[m], got[m], ad);
            bad++;
        }
    }
    int ok = (bad == 0);
    printf("  M=%d H=%d %-4s: maxref=%.4g max_abs=%.4g max_rel=%.2g bad=%d -> %s\n",
           M, H, mean ? "mean" : "sum", maxref, max_abs, max_rel, bad, ok ? "PASS" : "FAIL");
    free(got); free(ref);
    return ok ? 0 : 1;
}

static int run_shape(int fd, int M, int H, float amp)
{
    size_t n = (size_t)M * H;
    _Float16 *in = malloc(n * sizeof(_Float16));
    if (!in) { fprintf(stderr, "oom\n"); return 1; }
    fill(in, M, H, amp, (uint32_t)(M * 2654435761u + H * 40503u + (uint32_t)(amp * 7)));
    printf("[M=%d H=%d amp=%.0f]\n", M, H, amp);
    int fail = 0;
    fail |= check(fd, M, H, in, 0);   /* sum  */
    fail |= check(fd, M, H, in, 1);   /* mean */
    free(in);
    return fail;
}

/* layer 1: the oracle agrees with an independent naive accumulation (no NPU) */
static int ref_selfcheck(void)
{
    const int M = 7, H = 130;
    _Float16 *in = malloc((size_t)M * H * sizeof(_Float16));
    float *ref = malloc(M * sizeof(float));
    if (!in || !ref) { free(in); free(ref); return 1; }
    fill(in, M, H, 3.f, 123);
    rocket_reduce_feature_ref_fp16(M, H, in, ref, 0);
    int bad = 0;
    for (int m = 0; m < M; m++) {
        double naive = 0; for (int h = 0; h < H; h++) naive += (double)in[(size_t)m*H+h];
        if (fabs(naive - ref[m]) > 1e-3) bad++;
    }
    free(in); free(ref);
    printf("ref self-check (M=%d H=%d): %s\n\n", M, H, bad ? "FAIL" : "PASS");
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
        /* M, H, amplitude */
        g_fail |= run_shape(fd,   4,   64, 4.f);    /* tiny, single M-tile               */
        g_fail |= run_shape(fd, 256, 1024, 4.f);    /* exactly the M-tile cap (256)       */
        g_fail |= run_shape(fd, 260, 1024, 4.f);    /* just past the M-tile boundary       */
        g_fail |= run_shape(fd, 512, 3840, 4.f);    /* realistic Gemma hidden (K-tiled)    */
        g_fail |= run_shape(fd, 512, 2048, 4.f);    /* another realistic width             */
        g_fail |= run_shape(fd,   8,   48, 4.f);    /* H%32!=0 -> K-pad to 64              */
        g_fail |= run_shape(fd,   5,  100, 4.f);    /* M%4!=0 AND H%32!=0 -> both pads      */
        g_fail |= run_shape(fd, 128, 3840, 40.f);   /* large magnitude (sum-of-squares-ish) */
    }

    if (fd >= 0) rocket_close(fd);
    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : (fd < 0 ? 2 : 0);
}
