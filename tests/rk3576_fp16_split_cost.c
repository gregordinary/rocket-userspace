// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_fp16_split_cost.c — where the fp16 input-channel split spends its wall.
 *
 * One fp16 task on this part contracts exactly sixteen input channels, so an arbitrary
 * channel count is ic/16 submits whose partial surfaces are summed. Those partials are
 * summed on the HOST today, and the standing proposal is to sum them ON CHIP through the
 * DPU's eltwise stage — the RK3588's ROCKET_KACC analog.
 *
 * That proposal removes the READBACK and nothing else. The slice count is set by the
 * sixteen-channel contraction, so the wide-output submits stay, and with them the
 * poisoning each one leaves for the next and the power cycle the retry pays to clear it.
 * This probe prices the readback against the rest before that work is done: it sweeps the
 * channel count so the per-slice cost is visible, with ROCKET_RK3576_FP16_PROF on so the
 * entry reports its own phase split.
 *
 * Every shape is checked against a CPU reference first, so a broken path cannot read as a
 * cheap one. Operands are small integers held in fp16 and the accumulation order is short,
 * so both sides are exact and the comparison is equality, not a tolerance.
 *
 * This is a PROBE, not a gate: it exits 0 on any successful run, non-zero only if a shape
 * computed the wrong answer.
 *
 *   sudo -E taskset -c 4-7 ./build/rk3576_fp16_split_cost
 *   sudo -E taskset -c 4-7 ./build/rk3576_fp16_split_cost 32 28 3   # oc, plane, k
 *
 * Discard the first run: the clock parks at idle.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_log.h"
#include "rocket_hw_profile.h"

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

/* The same convolution the part computes, in float. Operands are small integers, so this
 * is exact and so is the fp16 the entry returns. */
static void ref_conv(const rocket_conv2d_desc *d, const _Float16 *in, const _Float16 *W,
                     _Float16 *out, int ow, int oh)
{
    for (int oc = 0; oc < d->oc; oc++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                float acc = 0;
                for (int ic = 0; ic < d->ic; ic++) {
                    for (int ky = 0; ky < d->kh; ky++) {
                        for (int kx = 0; kx < d->kw; kx++) {
                            int iy = y * d->stride_y + ky - d->pad_top;
                            int ix = x * d->stride_x + kx - d->pad_left;
                            if (iy < 0 || iy >= d->ih || ix < 0 || ix >= d->iw)
                                continue;
                            size_t io = ((size_t)ic * d->ih + iy) * d->iw + ix;
                            size_t wo = (((size_t)oc * d->ic + ic) * d->kh + ky) * d->kw + kx;
                            acc += (float)in[io] * (float)W[wo];
                        }
                    }
                }
                out[((size_t)oc * oh + y) * ow + x] = (_Float16)acc;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int OC    = argc > 1 ? atoi(argv[1]) : 32;
    int plane = argc > 2 ? atoi(argv[2]) : 28;
    int K     = argc > 3 ? atoi(argv[3]) : 3;
    const int ICS[] = {16, 32, 64, 128};
    const int NICS = (int)(sizeof(ICS) / sizeof(ICS[0]));

    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_fp16_split_cost: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }

    /* The entry's own phase split is the point of the run. */
    setenv("ROCKET_RK3576_FP16_PROF", "1", 1);
    rocket_log_set_level(ROCKET_LOG_INFO);

    int fd = rocket_open();
    if (fd < 0) {
        fprintf(stderr, "rk3576_fp16_split_cost: rocket_open failed (%d)\n", fd);
        return 1;
    }

    printf("rk3576 fp16 ic split   oc=%d  %dx%d  k%d\n", OC, plane, plane, K);
    printf("  on-chip accumulation would remove the READBACK column and nothing else\n\n");

    int failed = 0;

    for (int i = 0; i < NICS; i++) {
        int IC = ICS[i];
        rocket_conv2d_desc d = {0};
        d.ic = IC; d.oc = OC; d.ih = plane; d.iw = plane;
        d.kh = K;  d.kw = K;
        d.stride_y = 1; d.stride_x = 1;
        d.pad_top = K / 2; d.pad_left = K / 2;
        d.dil_y = 1; d.dil_x = 1;

        int ow = rocket_conv2d_ow(&d);
        int oh = rocket_conv2d_oh(&d);

        size_t n_in  = (size_t)IC * plane * plane;
        size_t n_w   = (size_t)OC * IC * K * K;
        size_t n_out = (size_t)OC * oh * ow;

        _Float16 *in  = malloc(n_in  * sizeof *in);
        _Float16 *W   = malloc(n_w   * sizeof *W);
        _Float16 *out = malloc(n_out * sizeof *out);
        _Float16 *ref = malloc(n_out * sizeof *ref);
        if (!in || !W || !out || !ref) { fprintf(stderr, "oom\n"); return 1; }

        for (size_t j = 0; j < n_in; j++) in[j] = (_Float16)(int)((j * 7 + 3) % 5 - 2);
        for (size_t j = 0; j < n_w;  j++) W[j]  = (_Float16)(int)((j * 5 + 1) % 3 - 1);

        int rc = rocket_conv2d_fp16(fd, &d, in, W, out);
        if (rc != ROCKET_OK) {
            printf("  ic=%-4d  REFUSED (%d)\n", IC, rc);
            free(in); free(W); free(out); free(ref);
            continue;
        }

        ref_conv(&d, in, W, ref, ow, oh);
        size_t bad = 0;
        for (size_t j = 0; j < n_out; j++)
            if (out[j] != ref[j]) bad++;
        if (bad) {
            printf("  ic=%-4d  WRONG: %zu of %zu elements — not timing a broken path\n",
                   IC, bad, n_out);
            failed = 1;
            free(in); free(W); free(out); free(ref);
            continue;
        }

        /* Warm, then the timed call whose phase split the entry logs. */
        rocket_conv2d_fp16(fd, &d, in, W, out);
        double t0 = now_us();
        rocket_conv2d_fp16(fd, &d, in, W, out);
        double us = now_us() - t0;

        printf("  ic=%-4d  %d slices  %8.2f ms total  (exact)\n",
               IC, IC / 16, us / 1e3);

        free(in); free(W); free(out); free(ref);
    }

    rocket_close(fd);
    return failed;
}
