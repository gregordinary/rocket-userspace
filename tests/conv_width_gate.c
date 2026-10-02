// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_width_gate.c — does a conv tile ever reach the RK3588 generator past the CNA's
 * 11-bit geometry fields?
 *
 * CNA DATA_SIZE0 holds the input width and height in 11 bits each, and DATA_SIZE2 the
 * output width in 11 bits [source-confirmed: Mesa registers.xml 0x1020, 0x1028], so a
 * tile past 2047 on any of them programs a wrapped extent and computes a full, plausible,
 * wrong surface. The tilers bound a tile's AREA against the CBUF budget, which leaves one
 * axis free to pass 2047 when the channels are thin and the other axis short. Every case
 * here sits in that corner or beside it:
 *
 *   - a wide, short input (two rows or one) at 32 or 16 channels, int8 and fp16
 *   - a tall, narrow input (3000 rows, one column), which is how rocket_conv1d_fp16 lays out
 *     a long sequence
 *   - controls under 2048 on the same datapaths, one of them wide enough that the budget
 *     alone narrows it
 *
 * through the public entries, so the tilers are under test and not a single job. Every
 * element is scored: int8 against an int64 accumulate over the full int8 range, fp16
 * against the CPU oracle on integer fills whose sums fp16 holds exactly. A call that
 * refuses is a failure too, because every shape here fits a tiling. A job the kernel
 * retired at its 500 ms watchdog is reported beside its case.
 *
 * Usage: conv_width_gate
 * Exit: 0 every case exact, 1 otherwise, 2 no NPU.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "test_fill.h"

struct wcase {
    const char *name;
    int int8;                 /* 1 = rocket_conv2d_int8, 0 = rocket_conv2d_fp16 */
    int ic, ih, iw, oc, kh, kw, pt, pl;
};

static const struct wcase cases[] = {
    { "int8 wide, 2 rows",        1, 32,    2, 2300, 32, 1, 1, 0, 0 },
    { "int8 wide, 1 row, 1x3",    1, 16,    1, 2500, 16, 1, 3, 0, 1 },
    { "int8 tall, 1 column, 3x1", 1, 32, 3000,    1, 32, 3, 1, 1, 0 },
    { "int8 control, 1500 wide",  1, 32,    2, 1500, 32, 1, 1, 0, 0 },
    { "int8 control, 4 rows",     1, 32,    4, 3000, 32, 1, 1, 0, 0 },
    { "fp16 wide, 2 rows",        0, 32,    2, 2300, 32, 1, 1, 0, 0 },
    { "fp16 wide, 1 row, 1x3",    0, 16,    1, 2500, 16, 1, 3, 0, 1 },
    { "fp16 tall, 1 column, 3x1", 0, 16, 3000,    1, 16, 3, 1, 1, 0 },
    { "fp16 control, 1200 wide",  0, 32,    2, 1200, 32, 1, 1, 0, 0 },
    { "fp16 control, 8 rows",     0, 32,    8, 1500, 32, 1, 1, 0, 0 },
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
static int run_case(int fd, const struct wcase *c, uint64_t seed)
{
    rocket_conv2d_desc d = { .ic = c->ic, .ih = c->ih, .iw = c->iw, .oc = c->oc,
        .kh = c->kh, .kw = c->kw, .stride_y = 1, .stride_x = 1,
        .pad_top = c->pt, .pad_left = c->pl, .dil_y = 1, .dil_x = 1, .depthwise = 0 };
    const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d);
    const size_t n_in = (size_t)c->ic * c->ih * c->iw;
    const size_t n_w = (size_t)c->oc * c->ic * c->kh * c->kw;
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
    printf("  %-26s IC %2d  %4dx%-4d -> %4dx%-4d  ", c->name, c->ic, c->ih, c->iw, OH, OW);
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
    printf("== conv width gate: %d cases through the public conv entries ==\n", n);
    for (int i = 0; i < n; i++) fails += run_case(fd, &cases[i], 0xC0DE + (uint64_t)i * 17);
    printf("== %d of %d cases exact ==\n", n - fails, n);
    rocket_close(fd);
    return fails ? 1 : 0;
}
