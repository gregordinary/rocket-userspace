// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * ct_model_bench.c — the ConvTranspose entry's per-call time at real models' shapes.
 *
 * Before a frontend wires ConvTranspose, the question is what one
 * call costs on the NPU at the shapes a model that spends real time in ConvTranspose uses.
 * The table holds pix2pix's U-Net-256 generator (eight k4 s2 p1 layers) and SAM ViT-B's mask
 * decoder upscaler (two k2 s2 layers). Two arms, each timed per shape: "old" is
 * rocket_conv_transpose2d_fp16_ctx with one resident conv context, whichever route it
 * chooses; "resident" is rocket_conv_transpose2d_fp16_prepacked with its weight packed
 * once on a three-worker rocket_ctx (the pack is outside the timing). Each arm: one
 * warm-up call discarded, then `reps` timed calls, median and minimum. The output is
 * scored against the scatter reference once per arm. Not a gate: the verdict is the
 * table, read against the CPU's time for the same layer.
 *
 * Usage: ct_model_bench [reps] [shape|-1] [arms: 1 old, 2 resident, 3 both] [order]
 *   (defaults 9, all, 1). `order` 1 runs the resident arm first within each shape, for a
 *   rotated interleaved pass. Exit 2 without a device or off the RK3588.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

/* `bias`: the layer has one in the model (the resident arm adds it; the old entry has no
 * bias input, so its arm runs without). pix2pix's batch-normed layers do not. */
typedef struct { const char *name; int ic, ih, iw, oc, k, s, p, bias; } shape_t;
static const shape_t SH[] = {
    { "pix2pix up1 512x1x1->512",    512,   1,   1, 512, 4, 2, 1, 0 },
    { "pix2pix up2 1024x2x2->512",  1024,   2,   2, 512, 4, 2, 1, 0 },
    { "pix2pix up3 1024x4x4->512",  1024,   4,   4, 512, 4, 2, 1, 0 },
    { "pix2pix up4 1024x8x8->512",  1024,   8,   8, 512, 4, 2, 1, 0 },
    { "pix2pix up5 1024x16x16->256", 1024, 16,  16, 256, 4, 2, 1, 0 },
    { "pix2pix up6 512x32x32->128",  512,  32,  32, 128, 4, 2, 1, 0 },
    { "pix2pix up7 256x64x64->64",   256,  64,  64,  64, 4, 2, 1, 0 },
    { "pix2pix up8 128x128x128->3",  128, 128, 128,   3, 4, 2, 1, 1 },
    { "SAM upscale1 256x64x64->64",  256,  64,  64,  64, 2, 2, 0, 1 },
    { "SAM upscale2 64x128x128->32",  64, 128, 128,  32, 2, 2, 0, 1 },
};
#define NSH ((int)(sizeof SH / sizeof SH[0]))

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const int reps = argc > 1 ? atoi(argv[1]) : 9;
    const int only = argc > 2 ? atoi(argv[2]) : -1;   /* one shape by index, for a profile */
    const int arms = argc > 3 ? atoi(argv[3]) : 1;
    const int order = argc > 4 ? atoi(argv[4]) : 0;
    if (!hw || strcmp(hw->name, "rk3588")) { printf("not an RK3588; skipping\n"); return 2; }
    int fd = rocket_open();
    if (fd < 0) { printf("SKIP: no /dev/accel/accel0\n"); return 2; }
    rocket_conv_ctx *ctx = rocket_conv_ctx_create(fd);
    rocket_ctx *mctx = (arms & 2) ? rocket_ctx_create(3) : NULL;
    if ((arms & 2) && !mctx) { printf("rocket_ctx_create failed\n"); return 1; }
    int bad = 0;
    printf("ct_model_bench: %d reps after one warm-up; old = rocket_conv_transpose2d_fp16_ctx, "
           "resident = rocket_conv_transpose2d_fp16_prepacked on 3 workers\n", reps);
    printf("  %-30s %-8s %9s %9s %10s %12s\n", "shape", "route", "median", "min", "GMAC",
           "max|err|");
    for (int i = 0; i < NSH; i++) {
        const shape_t *s = &SH[i];
        if (only >= 0 && i != only) continue;
        rocket_conv_transpose2d_desc d;
        memset(&d, 0, sizeof d);
        d.ic = s->ic; d.ih = s->ih; d.iw = s->iw; d.oc = s->oc;
        d.kh = d.kw = s->k; d.stride_y = d.stride_x = s->s; d.pad_top = d.pad_left = s->p;
        d.dil_y = d.dil_x = 1;
        const int oh = rocket_conv_transpose2d_oh(&d), ow = rocket_conv_transpose2d_ow(&d);
        const size_t ni = (size_t)s->ic * s->ih * s->iw, nw = (size_t)s->ic * s->oc * s->k * s->k;
        const size_t no = (size_t)s->oc * oh * ow;
        const int route = rocket_conv_transpose2d_route(&d);
        _Float16 *in = malloc(ni * 2), *W = malloc(nw * 2), *out = malloc(no * 2), *ref = malloc(no * 2);
        double *t = malloc((size_t)reps * sizeof *t);
        if (!in || !W || !out || !ref || !t) return 1;
        for (size_t j = 0; j < ni; j++) in[j] = (_Float16)((double)((j * 7 + 3) % 11) / 5.0 - 1.0);
        for (size_t j = 0; j < nw; j++) W[j] = (_Float16)((double)((j * 5 + 1) % 7) / 64.0 - 0.05);
        const double gmac = (double)ni * s->oc * s->k * s->k / 1e9;
        rocket_conv_transpose2d_ref_fp16(&d, in, W, ref);
        for (int step = 0; step < 2; step++) {
        const int arm = (step == 0) == (order == 0) ? 1 : 2;
        if (!(arms & arm)) continue;
        if (arm == 2) {
            _Float16 *bias = s->bias ? malloc((size_t)s->oc * 2) : NULL;
            for (int j = 0; bias && j < s->oc; j++) bias[j] = (_Float16)((double)(j % 5) / 8.0 - 0.25);
            rocket_conv_transpose2d_weights *h = rocket_conv_transpose2d_weights_pack(mctx, &d, W, bias);
            int rc = h ? rocket_conv_transpose2d_fp16_prepacked(mctx, h, in, out) : -1;
            if (rc) {
                printf("  %-30s resident refused or failed (plan %d, rc %d)\n", s->name,
                       rocket_conv_transpose2d_prepacked_plan(&d), rc);
                bad++;
            } else {
                int ok = 1;
                for (int r = 0; r < reps; r++) {
                    double t0 = now_ms();
                    if (rocket_conv_transpose2d_fp16_prepacked(mctx, h, in, out)) { ok = 0; break; }
                    t[r] = now_ms() - t0;
                }
                double e = 0.0;
                const size_t plane = (size_t)oh * ow;
                for (size_t j = 0; j < no; j++) {
                    const double b = bias ? (double)bias[j / plane] : 0.0;
                    double x = fabs((double)out[j] - ((double)ref[j] + b));
                    if (x > e) e = x;
                }
                if (ok) {
                    qsort(t, (size_t)reps, sizeof *t, cmp_d);
                    printf("  %-30s %-8s %7.2fms %7.2fms %10.3f %12g\n", s->name,
                           bias ? "res+bias" : "resident",
                           t[reps / 2], t[0], gmac, e);
                } else {
                    printf("  %-30s a timed resident call FAILED\n", s->name);
                    bad++;
                }
            }
            rocket_conv_transpose2d_weights_free(mctx, h);
            free(bias);
            continue;
        }
        int rc0 = route < 0 ? route : rocket_conv_transpose2d_fp16_ctx(ctx, &d, in, W, out);
        if (rc0 != 0) {
            printf("  %-30s refused or failed (route %d, plan %d, rc %d)\n", s->name, route,
                   rocket_conv_transpose2d_plan(&d), rc0);
            bad++;
        } else {
            int ok = 1;
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms();
                if (rocket_conv_transpose2d_fp16_ctx(ctx, &d, in, W, out) != 0) { ok = 0; break; }
                t[r] = now_ms() - t0;
            }
            double e = 0.0;
            for (size_t j = 0; j < no; j++) {
                double x = fabs((double)out[j] - (double)ref[j]);
                if (x > e) e = x;
            }
            if (ok) {
                qsort(t, (size_t)reps, sizeof *t, cmp_d);
                printf("  %-30s %-8s %7.2fms %7.2fms %10.3f %12g\n", s->name,
                       route == ROCKET_CONV_TRANSPOSE_DECONV ? "deconv" : "lowered", t[reps / 2],
                       t[0], gmac, e);
            } else {
                printf("  %-30s a timed call FAILED\n", s->name);
                bad++;
            }
        }
        }
        free(in); free(W); free(out); free(ref); free(t);
    }
    if (mctx) rocket_ctx_free(mctx);
    rocket_conv_ctx_free(ctx);
    rocket_close(fd);
    return bad ? 1 : 0;
}
