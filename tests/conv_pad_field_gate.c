// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_pad_field_gate.c — does a conv pad past 15 ever reach CNA_PAD_CON0?
 *
 * CNA_PAD_CON0 holds PAD_TOP and PAD_LEFT in 4 bits each, and a pad of 16 behaves exactly
 * as a pad of 0, 17 as 1 [HW sweep, tests/deconv_pad_probe.c]. The tilers materialize the
 * pad into each tile's input and program 0, but a conv that fits one CBUF pass skips the
 * tiler and programs the descriptor's pad directly. A pad of 16 or more on that path is a
 * wrapped field: a full, correctly sized, plausible surface computed at the wrong offset.
 *
 * Pads that large come from dilation. A "same" 1-D conv at kernel 7 and dilation 12 pads
 * 36, which is what a VITS/HiFi-GAN vocoder's residual blocks carry, and a stock kernel of
 * 33 or more pads 16. Every case here is small enough to take the one-pass path:
 *
 *   - fp16 direct, 8x8 at pads 15 (the control, the field's largest value), 16 and 17
 *   - fp16 depthwise at pads 15 and 16
 *   - fp16 1-D dilated conv, time on the height axis (rocket_conv1d_fp16's layout), k7 d12
 *     pad 36
 *   - int8 direct at pads 15 and 16
 *
 * through the public entries, so the routing is under test and not a single job. Every
 * element is scored exactly: int8 against an int64 accumulate over the full int8 range,
 * fp16 against the CPU oracle on integer fills whose sums fp16 holds exactly. A refusal is
 * a failure, because every shape here is computable.
 *
 * Usage: conv_pad_field_gate
 * Exit: 0 every case exact, 1 otherwise, 2 no NPU.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "test_fill.h"

struct pcase {
    const char *name;
    int int8, dw;
    int ic, ih, iw, oc, kh, kw, dy, dx, pt, pl;
};

static const struct pcase cases[] = {
    { "fp16 direct, pad 15",      0, 0, 32,   8, 8, 32, 3, 3,  1, 1, 15, 15 },
    { "fp16 direct, pad 16",      0, 0, 32,   8, 8, 32, 3, 3,  1, 1, 16, 16 },
    { "fp16 direct, pad 17/0",    0, 0, 32,   8, 8, 32, 3, 3,  1, 1, 17,  0 },
    { "fp16 depthwise, pad 15",   0, 1, 32,   8, 8, 32, 3, 3,  1, 1, 15, 15 },
    { "fp16 depthwise, pad 16",   0, 1, 32,   8, 8, 32, 3, 3,  1, 1, 16, 16 },
    { "fp16 1-D k7 d12, pad 36",  0, 0, 64, 200, 1, 64, 7, 1, 12, 1, 36,  0 },
    { "int8 direct, pad 15",      1, 0, 32,   8, 8, 32, 3, 3,  1, 1, 15, 15 },
    { "int8 direct, pad 16",      1, 0, 32,   8, 8, 32, 3, 3,  1, 1, 16, 16 },
};

static void ref_int8(const rocket_conv2d_desc *d, const int8_t *in, const int8_t *W,
                     int32_t *out)
{
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    for (int oc = 0; oc < d->oc; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t acc = 0;
                for (int ic = 0; ic < d->ic; ic++)
                    for (int kh = 0; kh < d->kh; kh++) {
                        int ih = oh * d->stride_y + kh * d->dil_y - d->pad_top;
                        if (ih < 0 || ih >= d->ih) continue;
                        for (int kw = 0; kw < d->kw; kw++) {
                            int iw = ow * d->stride_x + kw * d->dil_x - d->pad_left;
                            if (iw < 0 || iw >= d->iw) continue;
                            acc += (int64_t)in[((size_t)ic * d->ih + ih) * d->iw + iw] *
                                   W[(((size_t)oc * d->ic + ic) * d->kh + kh) * d->kw + kw];
                        }
                    }
                out[((size_t)oc * OH + oh) * OW + ow] = (int32_t)acc;
            }
}

/* Returns 0 exact, 1 wrong or refused. */
static int run_case(int fd, const struct pcase *c, uint64_t seed)
{
    rocket_conv2d_desc d = { .ic = c->ic, .ih = c->ih, .iw = c->iw, .oc = c->oc,
        .kh = c->kh, .kw = c->kw, .stride_y = 1, .stride_x = 1,
        .pad_top = c->pt, .pad_left = c->pl, .dil_y = c->dy, .dil_x = c->dx,
        .depthwise = c->dw };
    const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d);
    const size_t n_in = (size_t)c->ic * c->ih * c->iw;
    const size_t n_w = (size_t)c->oc * (c->dw ? 1 : c->ic) * c->kh * c->kw;
    const size_t n_out = (size_t)c->oc * OH * OW;
    const int dims[3] = { c->oc, OH, OW };
    long bad = -1;
    int rc;
    uint64_t slow0 = rocket_fence_wait_slow_count();

    if (c->int8) {
        int8_t *in = malloc(n_in), *W = malloc(n_w);
        int32_t *got = malloc(n_out * 4), *want = malloc(n_out * 4);
        if (!in || !W || !got || !want) { free(in); free(W); free(got); free(want); return 1; }
        tf_fill_i8(in, n_in, seed, -128, 127);
        tf_fill_i8(W, n_w, seed ^ 0x77, -128, 127);
        ref_int8(&d, in, W, want);
        tf_sentinel_bytes(got, n_out * 4);
        rc = rocket_conv2d_int8(fd, &d, in, W, got);
        if (rc == 0) {
            bad = 0;
            for (size_t i = 0; i < n_out; i++) bad += got[i] != want[i];
            if (bad) tf_cmp_i32(c->name, got, want, dims, 3);
        }
        free(in); free(W); free(got); free(want);
    } else {
        _Float16 *in = malloc(n_in * 2), *W = malloc(n_w * 2);
        _Float16 *got = malloc(n_out * 2), *want = malloc(n_out * 2);
        if (!in || !W || !got || !want) { free(in); free(W); free(got); free(want); return 1; }
        tf_fill_f16_int(in, n_in, seed, -2, 2);
        tf_fill_f16_int(W, n_w, seed ^ 0x77, -2, 2);
        rocket_conv2d_ref_fp16(&d, in, W, want);
        tf_sentinel_f16(got, n_out);
        rc = rocket_conv2d_fp16(fd, &d, in, W, got);
        if (rc == 0) {
            bad = 0;
            for (size_t i = 0; i < n_out; i++) bad += got[i] != want[i];
            if (bad) tf_cmp_f16(c->name, got, want, dims, 3);
        }
        free(in); free(W); free(got); free(want);
    }

    const int retired = rocket_fence_wait_slow_count() != slow0;
    printf("  %-26s %3dx%-3d -> %3dx%-3d  ", c->name, c->ih, c->iw, OH, OW);
    if (rc != 0) printf("REFUSED (%d)", rc);
    else printf("%ld of %zu wrong", bad, n_out);
    printf("%s  %s\n", retired ? "  [WATCHDOG]" : "", (rc == 0 && bad == 0 && !retired) ? "PASS" : "FAIL");
    return (rc == 0 && bad == 0 && !retired) ? 0 : 1;
}

int main(void)
{
    int fd = rocket_open(), fails = 0;
    const int n = (int)(sizeof(cases) / sizeof(cases[0]));

    if (fd < 0) { printf("no NPU device\n"); return 2; }
    printf("== conv pad field gate: %d cases through the public conv entries ==\n", n);
    for (int i = 0; i < n; i++) fails += run_case(fd, &cases[i], 0x9AD0 + (uint64_t)i * 31);
    printf("== %d of %d cases exact ==\n", n - fails, n);
    rocket_close(fd);
    return fails ? 1 : 0;
}
