// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * activation_lut_rocket.c — gate for the DPU LUT elementwise activation path.
 *
 *  1. REGCMD SMOKE (runs anywhere, no NPU): gen_lut_activation_fp16 must emit a
 *     non-empty program for a valid shape and reject a non-multiple-of-8 count.
 *
 *  2. ON-HARDWARE (only if /dev/accel/accel0 opens): run rocket_activation_fp16
 *     on the NPU over a representative input range and compare to the fp16 CPU
 *     reference. The LUT is an fp16 approximation, so the gate is a tolerance on
 *     the max absolute error (NOT bit-exactness). Two element counts exercise the
 *     n=128 reference cube and a larger generalized cube; the input range spills
 *     past the table edges to exercise the slope extrapolation (saturation).
 *     Then the fully-on-NPU elementwise MULTIPLY (rocket_ew_mul_fp16, the conv-
 *     main EW op) is gated bit-exact across sizes incl. one past the M_TILE
 *     boundary, and the gated activations (HardSwish/SiLU) are re-run through it
 *     (ROCKET_ACT_NPU_MUL) so the whole x*gate(x) runs on the NPU.
 *
 * Exit: 0 PASS, 1 FAIL, 2 no NPU (skipped HW, smoke still ran).
 *
 * Usage: activation_lut_rocket            (sigmoid + hardsigmoid, n=128 and 1024)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_activation.h"
#include "npu_activation.h"

/* fp16 LUT (Q0.15 + linear interp + fp16 out): HW-measured max_abs is ~0.0015
 * (sigmoid) / ~0.0005 (hardsigmoid). The gated kinds (x*gate(x)) amplify the
 * gate error by |x| and add an fp16-product rounding (ulp ~0.008 near |x|=8),
 * so they get a looser absolute gate. */
#define ACT_TOL      0.005
#define ACT_TOL_GATE 0.03

static int run_kind(int fd, int kind, int n, double tol)
{
    _Float16 *in  = calloc((size_t)n, sizeof(_Float16));
    _Float16 *out = malloc((size_t)n * sizeof(_Float16));
    _Float16 *ref = malloc((size_t)n * sizeof(_Float16));
    if (!in || !out || !ref) { fprintf(stderr, "oom\n"); free(in); free(out); free(ref); return 1; }

    for (int i = 0; i < n; i++)
        in[i] = (_Float16)(-8.0 + 16.0 * ((double)i / (double)(n - 1)));   /* [-8, 8] */

    memset(out, 0, (size_t)n * sizeof(_Float16));
    int r = rocket_activation_fp16(fd, kind, in, out, n);
    if (r) { printf("  %-12s n=%-5d rocket_activation_fp16=%d  -> FAIL\n",
                    rocket_activation_name(kind), n, r); free(in); free(out); free(ref); return 1; }

    rocket_activation_ref_fp16(kind, in, ref, n);
    double max_abs = 0; int bad = 0;
    for (int i = 0; i < n; i++) {
        double ad = fabs((double)(float)out[i] - (double)(float)ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad > tol && bad < 6) {
            printf("    [%d] x=%.3f ref=%.5f got=%.5f |d|=%.5f\n",
                   i, (float)in[i], (float)ref[i], (float)out[i], ad); bad++;
        }
    }
    int pass = max_abs <= tol;
    printf("  %-12s n=%-5d max_abs=%.5f (tol=%.3f) -> %s\n",
           rocket_activation_name(kind), n, max_abs, tol, pass ? "PASS" : "FAIL");
    free(in); free(out); free(ref);
    return pass ? 0 : 1;
}

/* Direct gate for the NPU elementwise multiply (independent of the activations). */
static int run_ew_mul(int fd, int n)
{
    _Float16 *a = malloc((size_t)n*sizeof(_Float16)), *b = malloc((size_t)n*sizeof(_Float16));
    _Float16 *o = malloc((size_t)n*sizeof(_Float16)), *r = malloc((size_t)n*sizeof(_Float16));
    if (!a||!b||!o||!r){ free(a);free(b);free(o);free(r); return 1; }
    for (int i = 0; i < n; i++) {
        a[i] = (_Float16)(((i % 7) - 3));            /* small ints -> exact fp16 product */
        b[i] = (_Float16)(((i % 5) - 2));
        r[i] = (_Float16)((float)a[i] * (float)b[i]);
    }
    memset(o, 0, (size_t)n*sizeof(_Float16));
    int rc = rocket_ew_mul_fp16(fd, a, b, o, n);
    int fail = 0;
    if (rc) { printf("  ew_mul       n=%-5d rocket_ew_mul_fp16=%d -> FAIL\n", n, rc); fail = 1; }
    else {
        double max_abs = 0; int bad = 0;
        for (int i = 0; i < n; i++) {
            double ad = fabs((double)(float)o[i] - (double)(float)r[i]);
            if (ad > max_abs) max_abs = ad;
            if (ad > 0 && bad < 6) { printf("    [%d] a=%.0f b=%.0f ref=%.0f got=%.3f\n",
                                            i,(float)a[i],(float)b[i],(float)r[i],(float)o[i]); bad++; }
        }
        fail = max_abs != 0.0;   /* small-int products are bit-exact */
        printf("  ew_mul       n=%-5d max_abs=%.5f -> %s\n", n, max_abs, fail ? "FAIL" : "PASS");
    }
    free(a); free(b); free(o); free(r);
    return fail;
}

int main(void)
{
    int fail = 0;

    /* 1. regcmd smoke (no HW needed) */
    {
        static uint16_t lut[1026];
        for (int i = 0; i < 1026; i++) lut[i] = (uint16_t)(i & 0x7FFF);
        uint64_t regs[2048] = {0};
        lut_act_params_t p = { .n = 128, .lut = lut, .bn_mul_operand = 0x6912,
            .bn_alu_cfg = 0x80000000, .out_cvt_offset = 1, .out_cvt_scale = 1,
            .out_cvt_minus_exp = 15, .out_cvt_cvt_type = 1, .lut_le_start = 0xffffc000,
            .lut_lo_end = 0x00004000, .le_slope_scale = 23107, .le_slope_shift = 22,
            .le_index_select = 5, .lo_index_select = 5,
            .input_dma = 0x10000, .output_dma = 0x20000, .tasks = regs };
        int g = gen_lut_activation_fp16(&p);
        int ok = (g == 0 && p.task_count > 1030 && p.task_count < 1100);
        printf("regcmd smoke: ret=%d task_count=%d -> %s\n", g, p.task_count, ok ? "OK" : "FAIL");
        if (!ok) fail = 1;

        lut_act_params_t bad = p; bad.n = 130;        /* not a multiple of 8 */
        int gb = gen_lut_activation_fp16(&bad);
        printf("regcmd reject n=130: ret=%d -> %s\n", gb, gb == -1 ? "OK" : "FAIL");
        if (gb != -1) fail = 1;

        uint64_t mregs[128] = {0};
        ew_mul_params_t mp = { .input_a_dma = 0x10000, .input_b_dma = 0x18000,
            .output_dma = 0x20000, .n = 128, .tasks = mregs };
        int gm = gen_ew_mul_fp16(&mp);
        int mok = (gm == 0 && mp.task_count > 20 && mp.task_count < 64);
        printf("ew_mul smoke: ret=%d task_count=%d -> %s\n", gm, mp.task_count, mok ? "OK" : "FAIL");
        if (!mok) fail = 1;
    }

    int fd = rocket_open();
    if (fd < 0) {
        printf("note: no /dev/accel/accel0 (%d) — smoke only, skipping HW gate\n", fd);
        return fail ? 1 : 2;
    }

    printf("\nHW activation gate (NPU vs fp16 CPU ref):\n");
    int ns[] = { 128, 1024 };
    for (size_t j = 0; j < sizeof(ns)/sizeof(ns[0]); j++) {
        fail |= run_kind(fd, ROCKET_ACTIVATION_SIGMOID,     ns[j], ACT_TOL);
        fail |= run_kind(fd, ROCKET_ACTIVATION_HARDSIGMOID, ns[j], ACT_TOL);
        /* HardSwish/SiLU = NPU gate (LUT) + host multiply by x — the nonlinear
         * part is on-NPU; gated here for end-to-end correctness. */
        fail |= run_kind(fd, ROCKET_ACTIVATION_HARDSWISH,   ns[j], ACT_TOL_GATE);
        fail |= run_kind(fd, ROCKET_ACTIVATION_SILU,        ns[j], ACT_TOL_GATE);
    }

    /* FULLY-on-NPU elementwise multiply (conv-main EW op, EW_OP_TYPE=1) — now
     * HW-correct (the old flying-MRDMA path could not deliver the operand; the
     * working path uses an identity conv as the main feed, see rocket_activation.c
     * + tests/ew_mul_rocket.c). Bit-exact on small-int products. The largest size
     * crosses the M_TILE=4096 boundary to exercise the runtime's tile loop. */
    printf("\nfully-on-NPU elementwise multiply (rocket_ew_mul_fp16):\n");
    int mn[] = { 128, 1024, 8192, 140000 };
    for (size_t j = 0; j < sizeof(mn)/sizeof(mn[0]); j++)
        fail |= run_ew_mul(fd, mn[j]);

    /* And the gated activations routed through that on-NPU multiply (the fully-on-
     * NPU HardSwish/SiLU: LUT gate job + EW-mul job, no host arithmetic). */
    printf("\nfully-on-NPU gated activations (ROCKET_ACT_NPU_MUL):\n");
    setenv("ROCKET_ACT_NPU_MUL", "1", 1);
    fail |= run_kind(fd, ROCKET_ACTIVATION_HARDSWISH, 1024, ACT_TOL_GATE);
    fail |= run_kind(fd, ROCKET_ACTIVATION_SILU,      1024, ACT_TOL_GATE);
    unsetenv("ROCKET_ACT_NPU_MUL");

    rocket_close(fd);
    printf("\n%s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
