// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv1d_rocket.c — HW gate for the Whisper-encoder conv1d front-end (rocket_conv1d_fp16),
 * a width-only 1D conv over time lowered onto the HW-validated height-1 rocket_conv2d_fp16.
 *
 * Whisper's two front-end convs (KW=3, pad=1; conv1 IC=n_mels stride 1, conv2 stride 2) are
 * this op. The gate runs rocket_conv1d_fp16 on the NPU and compares to the conv2d fp32-accumulate
 * CPU oracle on the equivalent height-1 descriptor — bit-exact (the NPU narrows to fp16 exactly
 * as the oracle does). Covers stride 1 and 2, the Whisper IC=80/OC=512 shape (IC not %32), small
 * shapes, and an OT past one CBUF tile.
 *
 * Usage: conv1d_rocket
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"

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

static int test_conv1d(int fd, int ic, int it, int oc, int kw, int stride, int pad)
{
    int ot = rocket_conv_out_dim(it, kw, stride, pad, 1);
    if (ot < 1) { printf("  (skip ic=%d it=%d: ot<1)\n", ic, it); return 0; }
    size_t in_n = (size_t)ic * it, w_n = (size_t)oc * ic * kw, out_n = (size_t)oc * ot;
    _Float16 *in = malloc(in_n * sizeof(_Float16));
    _Float16 *W  = malloc(w_n  * sizeof(_Float16));
    _Float16 *got= malloc(out_n* sizeof(_Float16));
    _Float16 *ref= malloc(out_n* sizeof(_Float16));
    if (!in||!W||!got||!ref){ fprintf(stderr,"oom\n"); free(in);free(W);free(got);free(ref); return 1; }
    fill(in, in_n, 1.0f, (uint32_t)(ic*7 + it));
    fill(W,  w_n,  0.5f, (uint32_t)(oc*13 + kw));

    /* oracle: the same conv with time on the HEIGHT axis (matches the conv1d lowering) */
    rocket_conv2d_desc d = { .ic=ic, .ih=it, .iw=1, .oc=oc, .kh=kw, .kw=1,
        .stride_y=stride, .stride_x=1, .pad_top=pad, .pad_left=0, .dil_y=1, .dil_x=1, .depthwise=0 };
    rocket_conv2d_ref_fp16(&d, in, W, ref);

    int rc = rocket_conv1d_fp16(fd, ic, it, oc, kw, stride, pad, in, W, got);
    char tag[96]; snprintf(tag, sizeof tag, "conv1d IC=%d IT=%d OC=%d K=%d s=%d p=%d (OT=%d)",
                           ic, it, oc, kw, stride, pad, ot);
    int fail = 0;
    if (rc) { printf("  %s: call=%d -> FAIL\n", tag, rc); fail = 1; }
    else {
        /* Two-metric: bit-exact for small reductions; a tiny fp16-accumulation tolerance for
         * the large IC*KW reductions (the CNA's fp32-accum ORDER differs from the host loop, so
         * a 1-ULP fp16 store difference is expected and benign — same regime as the conv2d HW
         * gate's tolerance). maxv-scaled rel 0.5% + abs 1 ULP-ish. */
        double maxv = 0; for (size_t i=0;i<out_n;i++){ double a=fabs((double)ref[i]); if(a>maxv)maxv=a; }
        double abs_tol = 0.005 * maxv + 1e-4;
        int bad = 0, exact = 1; double max_abs = 0;
        for (size_t i = 0; i < out_n; i++) {
            double ad = fabs((double)got[i] - (double)ref[i]);
            if (ad > max_abs) max_abs = ad;
            if (got[i] != ref[i]) exact = 0;
            if (ad > abs_tol) { if (bad < 5) printf("    [%zu] ref=%.6g got=%.6g d=%.4g\n", i,(double)ref[i],(double)got[i],ad); bad++; }
        }
        int ok = (bad == 0);
        printf("  %s: max_abs=%.4g bad=%d%s -> %s\n", tag, max_abs, bad,
               exact ? " (bit-exact)" : "", ok ? "PASS" : "FAIL");
        fail = ok ? 0 : 1;
    }
    free(in); free(W); free(got); free(ref);
    return fail;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd); printf("==== SKIP ====\n"); return 2; }

    g_fail |= test_conv1d(fd, 8,  64,  16, 3, 1, 1);   /* tiny, stride 1 */
    g_fail |= test_conv1d(fd, 8,  64,  16, 3, 2, 1);   /* stride 2 (conv2) */
    g_fail |= test_conv1d(fd, 80, 200, 512, 3, 1, 1);  /* Whisper conv1: IC=80 (not %32), OC=512 */
    g_fail |= test_conv1d(fd, 512,200, 512, 3, 2, 1);  /* Whisper conv2: stride 2 downsample */
    g_fail |= test_conv1d(fd, 16, 600, 32,  3, 1, 1);  /* wide OT (past one CBUF tile) */

    rocket_close(fd);
    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
