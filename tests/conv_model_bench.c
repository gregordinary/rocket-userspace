// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_model_bench.c — what one forward convolution costs on the NPU at a real model's
 * shapes, by route, before a frontend claims Conv nodes.
 *
 * The table holds pix2pix's U-Net-256 encoder (eight k4 s2 p1 layers, 3->64 at 256x256
 * down to 512->512 at 2x2). Two arms, each timed per shape:
 *
 *   direct   rocket_conv2d_fp16_ctx on one resident conv context, the weight passed (and
 *            scattered) on every call, which is the only fp16 conv entry the RK3588 has.
 *   im2col   the input unfolded on the host into K = IC*KH*KW planes of OH*OW pixels
 *            (fp16, single thread), then the resident fp16 matmul in its channel-planes
 *            form with the weight packed once on a three-worker rocket_ctx (outside the
 *            timing). Its output planes are the conv's [OC][OH][OW] directly. The unfold
 *            and the GEMM are timed apart.
 *
 * Each arm: one warm-up call discarded, then `reps` timed calls, median and minimum, and
 * the output scored once against rocket_conv2d_ref_fp16 as worst |err| over the element's
 * magnitude sum. Not a gate: the verdict is the table, read against the CPU's time for the
 * same layer.
 *
 * Usage: conv_model_bench [reps] [shape|-1] [arms: 1 direct, 2 im2col, 3 both] [order]
 *                         [split: -1 auto, 0 N, 1 M]
 *   `order` 1 runs the im2col arm first within each shape. Exit 2 without a device or off
 *   the RK3588.
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
#include "rocket_matmul_internal.h"   /* the channel-planes resident matmul */

typedef struct { const char *name; int ic, ih, iw, oc, k, s, p; } shape_t;
static const shape_t SH[] = {
    { "pix2pix c1 3x256x256->64",     3, 256, 256,  64, 4, 2, 1 },
    { "pix2pix c2 64x128x128->128",  64, 128, 128, 128, 4, 2, 1 },
    { "pix2pix c3 128x64x64->256",  128,  64,  64, 256, 4, 2, 1 },
    { "pix2pix c4 256x32x32->512",  256,  32,  32, 512, 4, 2, 1 },
    { "pix2pix c5 512x16x16->512",  512,  16,  16, 512, 4, 2, 1 },
    { "pix2pix c6 512x8x8->512",    512,   8,   8, 512, 4, 2, 1 },
    { "pix2pix c7 512x4x4->512",    512,   4,   4, 512, 4, 2, 1 },
    { "pix2pix c8 512x2x2->512",    512,   2,   2, 512, 4, 2, 1 },
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

static int rup(int x, int a) { return (x + a - 1) / a * a; }

/* The host unfold: plane k = (ic*KH + ky)*KW + kx holds in[ic][oy*s + ky - p][ox*s + kx - p]
 * over the OH*OW output pixels, zero outside the input. */
static void im2col_planes(const rocket_conv2d_desc *d, int OH, int OW, const _Float16 *in,
                          _Float16 *At, size_t lda)
{
    for (int ic = 0; ic < d->ic; ic++)
        for (int ky = 0; ky < d->kh; ky++)
            for (int kx = 0; kx < d->kw; kx++) {
                _Float16 *pl = At + ((size_t)(ic * d->kh + ky) * d->kw + kx) * lda;
                const _Float16 *src = in + (size_t)ic * d->ih * d->iw;
                for (int oy = 0; oy < OH; oy++) {
                    _Float16 *o = pl + (size_t)oy * OW;
                    const int iy = oy * d->stride_y + ky * d->dil_y - d->pad_top;
                    if (iy < 0 || iy >= d->ih) { memset(o, 0, (size_t)OW * sizeof *o); continue; }
                    const _Float16 *r = src + (size_t)iy * d->iw;
                    for (int ox = 0; ox < OW; ox++) {
                        const int ix = ox * d->stride_x + kx * d->dil_x - d->pad_left;
                        o[ox] = (ix >= 0 && ix < d->iw) ? r[ix] : (_Float16)0;
                    }
                }
            }
}

/* worst |out - ref| / mag, mag the element's sum of |x*w| (fp64). */
static double score(const rocket_conv2d_desc *d, int OH, int OW, const _Float16 *in,
                    const _Float16 *W, const _Float16 *out, size_t ostride, double *maxabs)
{
    double worst = 0.0, ma = 0.0;
    for (int oc = 0; oc < d->oc; oc++)
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++) {
                double acc = 0.0, mag = 0.0;
                for (int ic = 0; ic < d->ic; ic++)
                    for (int ky = 0; ky < d->kh; ky++) {
                        const int iy = oy * d->stride_y + ky - d->pad_top;
                        if (iy < 0 || iy >= d->ih) continue;
                        for (int kx = 0; kx < d->kw; kx++) {
                            const int ix = ox * d->stride_x + kx - d->pad_left;
                            if (ix < 0 || ix >= d->iw) continue;
                            const double p = (double)in[((size_t)ic * d->ih + iy) * d->iw + ix] *
                                (double)W[(((size_t)oc * d->ic + ic) * d->kh + ky) * d->kw + kx];
                            acc += p; mag += fabs(p);
                        }
                    }
                const double e = fabs((double)out[(size_t)oc * ostride + (size_t)oy * OW + ox] - acc);
                if (e > ma) ma = e;
                if (mag > 0 && e / mag > worst) worst = e / mag;
            }
    if (maxabs) *maxabs = ma;
    return worst;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const int reps = argc > 1 ? atoi(argv[1]) : 9;
    const int only = argc > 2 ? atoi(argv[2]) : -1;
    const int arms = argc > 3 ? atoi(argv[3]) : 3;
    const int order = argc > 4 ? atoi(argv[4]) : 0;
    const int force_split = argc > 5 ? atoi(argv[5]) : -1;
    const int do_score = getenv("CMB_NOSCORE") == NULL;
    if (!hw || strcmp(hw->name, "rk3588")) { printf("not an RK3588; skipping\n"); return 2; }
    int fd = rocket_open();
    if (fd < 0) { printf("SKIP: no /dev/accel/accel0\n"); return 2; }
    rocket_conv_ctx *cctx = rocket_conv_ctx_create(fd);
    rocket_ctx *mctx = (arms & 2) ? rocket_ctx_create(3) : NULL;
    if ((arms & 2) && !mctx) { printf("rocket_ctx_create failed\n"); return 1; }
    int bad = 0;
    printf("conv_model_bench: %d reps after one warm-up; direct = rocket_conv2d_fp16_ctx, "
           "im2col = host unfold + resident planes matmul on 3 workers\n", reps);
    printf("  %-28s %-9s %9s %9s %9s %9s %8s %10s\n", "shape", "arm", "median", "min",
           "unfold", "gemm", "GMAC", "err/mag");
    for (int i = 0; i < NSH; i++) {
        const shape_t *s = &SH[i];
        if (only >= 0 && i != only) continue;
        rocket_conv2d_desc d;
        memset(&d, 0, sizeof d);
        d.ic = s->ic; d.ih = s->ih; d.iw = s->iw; d.oc = s->oc;
        d.kh = d.kw = s->k; d.stride_y = d.stride_x = s->s; d.pad_top = d.pad_left = s->p;
        d.dil_y = d.dil_x = 1;
        const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d);
        const size_t ni = (size_t)s->ic * s->ih * s->iw, nw = (size_t)s->oc * s->ic * s->k * s->k;
        const size_t no = (size_t)s->oc * OH * OW;
        _Float16 *in = malloc(ni * 2), *W = malloc(nw * 2), *out = malloc(no * 2);
        double *t = malloc((size_t)reps * sizeof *t), *tu = malloc((size_t)reps * sizeof *tu),
               *tg = malloc((size_t)reps * sizeof *tg);
        if (!in || !W || !out || !t || !tu || !tg) return 1;
        srand(1234 + i);
        for (size_t j = 0; j < ni; j++) in[j] = (_Float16)((double)rand() / RAND_MAX * 2.0 - 1.0);
        const double wsc = 1.0 / sqrt((double)s->ic * s->k * s->k);
        for (size_t j = 0; j < nw; j++)
            W[j] = (_Float16)(((double)rand() / RAND_MAX * 2.0 - 1.0) * wsc);
        const double gmac = (double)no * s->ic * s->k * s->k / 1e9;
        for (int step = 0; step < 2; step++) {
            const int arm = (step == 0) == (order == 0) ? 1 : 2;
            if (!(arms & arm)) continue;
            if (arm == 1) {
                int rc = rocket_conv2d_plan(&d);
                if (!rc) rc = rocket_conv2d_fp16_ctx(cctx, &d, in, W, out);
                if (rc) { printf("  %-28s direct    refused or failed (%d)\n", s->name, rc); bad++; continue; }
                int ok = 1;
                for (int r = 0; r < reps; r++) {
                    double t0 = now_ms();
                    if (rocket_conv2d_fp16_ctx(cctx, &d, in, W, out)) { ok = 0; break; }
                    t[r] = now_ms() - t0;
                }
                if (!ok) { printf("  %-28s direct    a timed call FAILED\n", s->name); bad++; continue; }
                double e = do_score ? score(&d, OH, OW, in, W, out, (size_t)OH * OW, NULL) : -1;
                qsort(t, (size_t)reps, sizeof *t, cmp_d);
                printf("  %-28s %-9s %7.2fms %7.2fms %9s %9s %8.3f %10.3g\n", s->name, "direct",
                       t[reps / 2], t[0], "-", "-", gmac, e);
                continue;
            }
            /* im2col arm */
            const int kl = s->ic * s->k * s->k, ml = OH * OW;
            const int M = rup(ml, 4), K = rup(kl, 32), N = rup(s->oc, 16);
            const int nt = rocket_ctx_nthreads(mctx);
            int split = force_split >= 0 ? force_split
                      : (nt > 1 && M / nt >= hw->max_tile &&
                         (size_t)nt * K * N * sizeof(_Float16) <= ((size_t)64 << 20));
            _Float16 *B = calloc((size_t)N * K, sizeof *B);
            _Float16 *At = calloc((size_t)K * M, sizeof *At);
            _Float16 *Ct = calloc((size_t)N * M, sizeof *Ct);
            if (!B || !At || !Ct) return 1;
            for (int n = 0; n < s->oc; n++)
                memcpy(B + (size_t)n * K, W + (size_t)n * kl, (size_t)kl * sizeof *B);
            struct rocket_weights *w = rocket_weights_pack_split(mctx, M, K, N, B, split);
            if (!w) { printf("  %-28s im2col    pack failed\n", s->name); bad++; goto next; }
            im2col_planes(&d, OH, OW, in, At, (size_t)M);
            int rc = rocket_matmul_fp16_prepacked_planes(mctx, M, K, N, At, (size_t)M, ml, kl, Ct, w);
            if (rc) { printf("  %-28s im2col    gemm failed (%d)\n", s->name, rc); bad++; rocket_weights_free(mctx, w); goto next; }
            int ok = 1;
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms();
                im2col_planes(&d, OH, OW, in, At, (size_t)M);
                double t1 = now_ms();
                if (rocket_matmul_fp16_prepacked_planes(mctx, M, K, N, At, (size_t)M, ml, kl, Ct, w)) { ok = 0; break; }
                double t2 = now_ms();
                t[r] = t2 - t0; tu[r] = t1 - t0; tg[r] = t2 - t1;
            }
            rocket_weights_free(mctx, w);
            if (!ok) { printf("  %-28s im2col    a timed call FAILED\n", s->name); bad++; goto next; }
            {
                double e = do_score ? score(&d, OH, OW, in, W, Ct, (size_t)M, NULL) : -1;
                qsort(t, (size_t)reps, sizeof *t, cmp_d);
                qsort(tu, (size_t)reps, sizeof *tu, cmp_d);
                qsort(tg, (size_t)reps, sizeof *tg, cmp_d);
                printf("  %-28s %-9s %7.2fms %7.2fms %7.2fms %7.2fms %8.3f %10.3g  (M%d K%d N%d split %s)\n",
                       s->name, "im2col", t[reps / 2], t[0], tu[reps / 2], tg[reps / 2], gmac, e,
                       M, K, N, split ? "M" : "N");
            }
        next:
            free(B); free(At); free(Ct);
        }
        free(in); free(W); free(out); free(t); free(tu); free(tg);
    }
    if (mctx) rocket_ctx_free(mctx);
    rocket_conv_ctx_free(cctx);
    rocket_close(fd);
    return bad ? 1 : 0;
}
