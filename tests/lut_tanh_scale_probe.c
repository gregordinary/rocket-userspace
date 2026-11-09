// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * lut_tanh_scale_probe.c — the tanh LUT's index scale against its error, on the device.
 *
 * A probe, not a gate. Every finite fp16 input in [-8, 8] (36866 of them) goes through
 * rocket_activation_fp16(TANH) once per scale, and each output is scored against fp64 tanh
 * rounded to fp16. The scale is the BN multiplier that maps x onto the LUT index, so the
 * table spans x in +/-16384/scale and holds its edge value past that. Per scale it prints:
 *
 *   max        the largest error anywhere, which the x~0 LE/LO spike sets at every scale
 *   max(|x|>0.01)  the largest error outside that band, the figure a scale is chosen by
 *   band       the largest |x| whose error exceeds 0.002; past a scale whose edge error
 *              itself exceeds 0.002 it reads as the end of the input range
 *   inexact    outputs that differ from fp16 tanh at all
 *
 *   lut_tanh_scale_probe                   the library's own scale
 *   lut_tanh_scale_probe SCALE [SCALE ...] each scale through ROCKET_LUT_IDXSCALE
 *
 * ROCKET_LUT_LESLOPE in the environment overrides the LE underflow slope for every run.
 * Exit 0 when every call returned 0, 1 otherwise, 2 with no NPU.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_activation.h"

static int run(int fd, const _Float16 *in, _Float16 *out, int n, const char *label)
{
    memset(out, 0, (size_t)n * sizeof *out);
    int rc = rocket_activation_fp16(fd, ROCKET_ACTIVATION_TANH, in, out, n);
    double mx = 0, mx_out = 0, band = 0;
    float x_worst = 0;
    long inexact = 0;
    for (int i = 0; i < n; i++) {
        double x = (double)(float)in[i];
        double want = (double)(float)(_Float16)tanh(x);
        double d = fabs((double)(float)out[i] - want);
        if (d > mx) { mx = d; x_worst = (float)x; }
        if (fabs(x) > 0.01 && d > mx_out) mx_out = d;
        if (d > 0.002 && fabs(x) > band) band = fabs(x);
        inexact += d > 0;
    }
    printf("scale %-8s rc %d  max %.5f @x=%+.4f  max(|x|>0.01) %.5f  band |x|<=%.5f  "
           "inexact %ld of %d\n", label, rc, mx, x_worst, mx_out, band, inexact, n);
    fflush(stdout);
    return rc;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); return 2; }

    _Float16 *in = malloc(65536 * sizeof *in), *out = malloc(65536 * sizeof *out);
    if (!in || !out) { free(in); free(out); rocket_close(fd); return 1; }
    int n = 0;
    for (uint32_t b = 0; b < 65536; b++) {
        uint16_t u = (uint16_t)b;
        _Float16 h;
        memcpy(&h, &u, sizeof h);
        float f = (float)h;
        if (isfinite(f) && f >= -8.0f && f <= 8.0f) in[n++] = h;
    }

    int fail = 0;
    if (argc < 2) {
        fail |= run(fd, in, out, n, "default") != 0;
    } else {
        for (int a = 1; a < argc; a++) {
            setenv("ROCKET_LUT_IDXSCALE", argv[a], 1);
            fail |= run(fd, in, out, n, argv[a]) != 0;
        }
        unsetenv("ROCKET_LUT_IDXSCALE");
    }
    free(in);
    free(out);
    rocket_close(fd);
    return fail;
}
