// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_refusal_gate.c — the op library refuses rather than writing nothing.
 *
 * The datapath semantics are shared across this IP family, but the CNA/CORE/DPU
 * GEOMETRY-register encoding is not: another revision re-packs the same block bases. A
 * program built for the RK3588 therefore reaches an RK3576 as a valid-looking job that
 * completes in the usual time, faults nothing, and writes nothing — and a caller reads a
 * correctly sized, entirely stale buffer. Silence is the hazard, so an op with no encoder
 * for the running part has to fail loudly.
 *
 * This gate holds every entry to that. It calls one op per family with a shape that is
 * runnable on the RK3588 and asserts the call FAILS on a part whose encoding the library
 * does not emit. It is the test for a negative, so it is deliberately blunt: it does not
 * check which error came back, only that the call did not claim success.
 *
 * IT ALSO HOLDS THE OTHER DIRECTION, for the entries that no longer refuse. Where the
 * RK3576 has an encoder AND the public entry's semantics are the ones the part computes,
 * that entry dispatches to it, and a refusal there would be the regression — the whole
 * point of writing the encoders was to stop refusing. So rocket_conv2d_fp16() and
 * rocket_conv2d_dw_int8() are asserted to RUN. rocket_conv2d_int8() still refuses,
 * because it writes a raw int32 accumulator and this part's direct int8 datapath
 * requantizes on chip: the semantics differ, so it names rocket_conv2d_int8_rk3576()
 * instead of pretending. rk3576_conv_lib_gate.c is what scores those entries' answers;
 * this one only asks whether they claim to have one.
 *
 * On the RK3588 every one of these is expected to run, so the gate skips there — the
 * ordinary op gates cover that direction.
 *
 * Run with `sudo -E` on the RK3576. Exits 2 (skip) on the RK3588 or with no device.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "rocket_matmul.h"
#include "rocket_conv.h"
#include "rocket_pool.h"
#include "rocket_activation.h"
#include "rocket_reduce.h"

static int fails;
static int checks;

static void expect_refused(const char *what, int rc)
{
    checks++;
    if (rc == 0) {
        printf("  FAIL   %-28s returned 0 — it emitted the RK3588 encoding and the job "
               "wrote nothing\n", what);
        fails++;
    } else {
        printf("  PASS   %-28s refused (%d)\n", what, rc);
    }
}

/* The inverse assertion, for an entry that now reaches this part's own encoder. */
static void expect_ran(const char *what, int rc)
{
    checks++;
    if (rc != 0) {
        printf("  FAIL   %-28s refused (%d) — this part has an encoder for it and the "
               "entry is supposed to dispatch there\n", what, rc);
        fails++;
    } else {
        printf("  PASS   %-28s ran\n", what);
    }
}

int main(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd;

    if (!strcmp(hw->name, "rk3588")) {
        printf("rk3576_refusal_gate: profile is rk3588, whose encoding this library "
               "emits — skipping\n");
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_refusal_gate: no NPU device — skipping\n"); return 2; }

    printf("== the op library on a %s: every datapath with no encoder refuses ==\n",
           hw->name);

    /* The matmuls. These two already dispatched per chip; they are here so the gate
     * covers the whole surface rather than only the parts it added. */
    {
        enum { M = 8, K = 64, N = 32 };
        static _Float16 A[M * K], B[N * K], C[M * N];
        static int8_t  qA[M * K], qB[N * K];
        static int32_t qC[M * N];
        expect_refused("rocket_matmul_fp16",
                       rocket_matmul_fp16(fd, M, K, N, A, B, C));
        expect_refused("rocket_matmul_int8",
                       rocket_matmul_int8(fd, M, K, N, qA, qB, qC));
    }

    /* Convolution, direct and depthwise, at both precisions. Two of the three now reach
     * this part's own encoder through the public entry and are asserted to RUN; the
     * third keeps refusing because its output semantics are not the ones the part
     * computes, and it names the entry that is. */
    {
        rocket_conv2d_desc d = {0};
        static _Float16 in[32 * 8 * 8], W[32 * 32], out[32 * 8 * 8];
        static int8_t  qin[32 * 16 * 16], qW[32 * 32 * 9], qout[32 * 16 * 16];
        static int32_t bias[32], qout32[32 * 8 * 8];
        d.ic = 32; d.ih = 8; d.iw = 8; d.oc = 32;
        d.kh = 1; d.kw = 1; d.stride_y = 1; d.stride_x = 1; d.dil_y = 1; d.dil_x = 1;
        expect_ran("rocket_conv2d_fp16",
                   rocket_conv2d_fp16(fd, &d, in, W, out));
        expect_refused("rocket_conv2d_int8",
                       rocket_conv2d_int8(fd, &d, qin, qW, qout32));
        d.depthwise = 1; d.ih = 16; d.iw = 16;
        d.kh = 3; d.kw = 3; d.pad_top = 1; d.pad_left = 1;
        expect_ran("rocket_conv2d_dw_int8",
                   rocket_conv2d_dw_int8(fd, &d, qin, qW, bias,
                                         1.0f, 1.0f, 64.0f, 0, 0, 0, qout));
    }

    /* The FIRST CONV. A packed image runs the CNA's own sub-encoding and BOTH
     * precisions of it compute, so what is asserted here is the shape of the int8
     * envelope rather than a blanket refusal — the bounds it adds are silent when
     * violated (a zero left pad writes an untouched surface; a wrong output width
     * writes a sheared one), which makes each refusal load-bearing in both directions.
     *
     * `rocket_conv2d_int8()` stays refused: it is the int32-output entry, and this
     * part's first conv writes int8 through the DPU's requant. */
    {
        rocket_conv2d_desc d = {0};
        static _Float16 fin[3 * 32 * 32], fW[16 * 3 * 9], fout[16 * 32 * 32];
        static int8_t  qin[3 * 32 * 32], qW[32 * 3 * 9], qout[32 * 32 * 32];
        static int32_t qout32[16 * 32 * 32];
        d.ic = 3; d.ih = 32; d.iw = 32; d.oc = 16;
        d.kh = 3; d.kw = 3; d.stride_y = 1; d.stride_x = 1;
        d.pad_top = 1; d.pad_left = 1; d.dil_y = 1; d.dil_x = 1;
        expect_ran("rocket_conv2d_fp16 first conv",
                   rocket_conv2d_fp16(fd, &d, fin, fW, fout));
        expect_refused("rocket_conv2d_int8 first conv (int32 out)",
                       rocket_conv2d_int8(fd, &d, qin, qW, qout32));
        /* oc=16 is a partial 32-channel group and writes nothing. */
        expect_refused("rocket_conv2d_int8_rk3576 first conv, oc 16",
                       rocket_conv2d_int8_rk3576(fd, &d, qin, qW, NULL,
                                                 1.0f, 1.0f, 64.0f, 0, 0, 0, qout));
        d.oc = 32;
        expect_ran("rocket_conv2d_int8_rk3576 first conv",
                   rocket_conv2d_int8_rk3576(fd, &d, qin, qW, NULL,
                                             1.0f, 1.0f, 64.0f, 0, 0, 0, qout));
        d.pad_top = 0; d.pad_left = 0;
        expect_refused("rocket_conv2d_int8_rk3576 first conv, pad_left 0",
                       rocket_conv2d_int8_rk3576(fd, &d, qin, qW, NULL,
                                                 1.0f, 1.0f, 64.0f, 0, 0, 0, qout));
        d.pad_top = 1; d.pad_left = 1; d.ic = 1;
        expect_refused("rocket_conv2d_int8_rk3576 first conv, one image channel",
                       rocket_conv2d_int8_rk3576(fd, &d, qin, qW, NULL,
                                                 1.0f, 1.0f, 64.0f, 0, 0, 0, qout));
    }

    /* Pooling, on the PPU. */
    {
        rocket_pool_desc d = {0};
        static _Float16 in[32 * 8 * 8], out[32 * 4 * 4];
        d.c = 32; d.ih = 8; d.iw = 8; d.kh = 2; d.kw = 2;
        d.stride_y = 2; d.stride_x = 2;
        expect_refused("rocket_pool_fp16", rocket_pool_fp16(fd, &d, in, out));
    }

    /* The DPU LUT activations, and the elementwise ALU under them. */
    {
        enum { N = 256 };
        static _Float16 in[N], b[N], out[N];
        expect_refused("rocket_activation_fp16",
                       rocket_activation_fp16(fd, 0, in, out, N));
        expect_refused("rocket_ew_mul_fp16", rocket_ew_mul_fp16(fd, in, b, out, N));
    }

    /* The feature-axis reduction, which the norms and softmax are built on. */
    {
        enum { M = 8, H = 64 };
        static _Float16 in[M * H];
        static float out[M];
        expect_refused("rocket_reduce_feature_fp16",
                       rocket_reduce_feature_fp16(fd, M, H, in, out, 0));
    }

    printf("== %d entries behaved as required, %d did not ==\n",
           checks - fails, fails);
    rocket_close(fd);
    return fails ? 1 : 0;
}
