// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * ct_route_probe.c — price the candidate ConvTranspose routes at pix2pix's and SAM's shapes
 * before one of them is built into the library.
 *
 * Three routes, each timed per call with its host phases separated, at the shapes
 * tests/ct_model_bench.c runs:
 *
 *   col2im  one resident fp16 matmul C[IH*IW, OC*KH*KW] = in^T . W, whose rows are the
 *           input pixels and whose columns are every (oc, kh, kw) tap, then a host
 *           scatter-add of each column into its output pixel. No zero-MACs, no im2col,
 *           and the weight is W itself read as [IC][OC*KH*KW]; the output is KH*KW/s^2
 *           times the result, and the scatter-add is the host's cost.
 *   subpix  the sub-pixel form as one resident fp16 matmul: an im2col of the COMPACT input
 *           over a ceil(k/s) window, against every phase's sub-kernel as N (s^2*OC
 *           columns), then a depth-to-space. No zero-MACs past the window's edge, and
 *           the output is the result's size; the im2col is the host's cost.
 *   subconv the same sub-pixel form as one stride-1 forward conv of kernel ceil(k/s) over
 *           the compact input, through rocket_conv2d_fp16_ctx (weights scattered per
 *           job, as that entry does today; ROCKET_CONV_PROFILE prices the scatter).
 *
 * Each route's output is scored against the scatter reference. Not a gate: the verdict
 * is the table.
 *
 * Usage: ct_route_probe [reps] [shape-index] [routes-mask: 1 col2im, 2 subpix, 4 subconv]
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

typedef struct { const char *name; int ic, ih, iw, oc, k, s, p; } shape_t;
static const shape_t SH[] = {
    { "pix2pix up1 512x1x1->512",    512,   1,   1, 512, 4, 2, 1 },
    { "pix2pix up2 1024x2x2->512",  1024,   2,   2, 512, 4, 2, 1 },
    { "pix2pix up3 1024x4x4->512",  1024,   4,   4, 512, 4, 2, 1 },
    { "pix2pix up4 1024x8x8->512",  1024,   8,   8, 512, 4, 2, 1 },
    { "pix2pix up5 1024x16x16->256", 1024, 16,  16, 256, 4, 2, 1 },
    { "pix2pix up6 512x32x32->128",  512,  32,  32, 128, 4, 2, 1 },
    { "pix2pix up7 256x64x64->64",   256,  64,  64,  64, 4, 2, 1 },
    { "pix2pix up8 128x128x128->3",  128, 128, 128,   3, 4, 2, 1 },
    { "SAM upscale1 256x64x64->64",  256,  64,  64,  64, 2, 2, 0 },
    { "SAM upscale2 64x128x128->32",  64, 128, 128,  32, 2, 2, 0 },
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
static double med(double *t, int n) { qsort(t, (size_t)n, sizeof *t, cmp_d); return t[n / 2]; }
static int rup(int x, int a) { return (x + a - 1) / a * a; }

/* max |out - ref| and the largest |out - ref| / (sum_{taps} |x w|), the error in units of
 * the element's own magnitude sum (what an fp16 rounding bound is written against). */
static void score(const rocket_conv_transpose2d_desc *d, const _Float16 *in, const _Float16 *W,
                  const _Float16 *out, const _Float16 *ref, double *emax, double *erel)
{
    const int OH = rocket_conv_transpose2d_oh(d), OW = rocket_conv_transpose2d_ow(d);
    const size_t no = (size_t)d->oc * OH * OW;
    float *mag = calloc(no, sizeof *mag);
    for (int ic = 0; ic < d->ic; ic++)
        for (int ih = 0; ih < d->ih; ih++)
            for (int iw = 0; iw < d->iw; iw++) {
                float x = fabsf((float)in[((size_t)ic * d->ih + ih) * d->iw + iw]);
                if (x == 0.f) continue;
                for (int oc = 0; oc < d->oc; oc++)
                    for (int kh = 0; kh < d->kh; kh++) {
                        int ph = ih * d->stride_y - d->pad_top + kh;
                        if (ph < 0 || ph >= OH) continue;
                        for (int kw = 0; kw < d->kw; kw++) {
                            int pw = iw * d->stride_x - d->pad_left + kw;
                            if (pw < 0 || pw >= OW) continue;
                            mag[((size_t)oc * OH + ph) * OW + pw] +=
                                x * fabsf((float)W[(((size_t)ic * d->oc + oc) * d->kh + kh) * d->kw + kw]);
                        }
                    }
            }
    double e = 0, r = 0;
    for (size_t j = 0; j < no; j++) {
        double x = fabs((double)out[j] - (double)ref[j]);
        if (x > e) e = x;
        if (mag[j] > 0 && x / mag[j] > r) r = x / mag[j];
    }
    free(mag);
    *emax = e; *erel = r;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    const int only = argc > 2 ? atoi(argv[2]) : -1;
    const int mask = argc > 3 ? atoi(argv[3]) : 7;
    if (!hw || strcmp(hw->name, "rk3588")) { printf("not an RK3588; skipping\n"); return 2; }
    int fd = rocket_open();
    if (fd < 0) { printf("SKIP: no /dev/accel/accel0\n"); return 2; }
    const char *nte = getenv("CT_NT");
    rocket_ctx *mctx = rocket_ctx_create(nte ? atoi(nte) : 3);
    rocket_conv_ctx *cctx = rocket_conv_ctx_create(fd);
    if (!mctx || !cctx) return 1;
    double *t = malloc((size_t)(reps + 1) * 8 * sizeof *t);

    printf("ct_route_probe: %d reps after one warm-up; medians in ms\n", reps);
    for (int si = 0; si < NSH; si++) {
        const shape_t *s = &SH[si];
        if (only >= 0 && si != only) continue;
        rocket_conv_transpose2d_desc d;
        memset(&d, 0, sizeof d);
        d.ic = s->ic; d.ih = s->ih; d.iw = s->iw; d.oc = s->oc;
        d.kh = d.kw = s->k; d.stride_y = d.stride_x = s->s; d.pad_top = d.pad_left = s->p;
        d.dil_y = d.dil_x = 1;
        const int IC = s->ic, IH = s->ih, IW = s->iw, OC = s->oc, K = s->k, S = s->s, P = s->p;
        const int OH = rocket_conv_transpose2d_oh(&d), OW = rocket_conv_transpose2d_ow(&d);
        const size_t ni = (size_t)IC * IH * IW, nw = (size_t)IC * OC * K * K, no = (size_t)OC * OH * OW;
        _Float16 *in = malloc(ni * 2), *W = malloc(nw * 2), *out = malloc(no * 2), *ref = malloc(no * 2);
        for (size_t j = 0; j < ni; j++) in[j] = (_Float16)((double)((j * 7 + 3) % 11) / 5.0 - 1.0);
        for (size_t j = 0; j < nw; j++) W[j] = (_Float16)((double)((j * 5 + 1) % 7) / 64.0 - 0.05);
        rocket_conv_transpose2d_ref_fp16(&d, in, W, ref);
        printf("%s  (OH %d, useful %.3f GMAC)\n", s->name, OH, (double)OH * OW * OC * IC * K * K / (S * S) / 1e9);

        if (mask & 1) {   /* ---- col2im ---- */
            const int M = rup(IH * IW, 4), Kd = rup(IC, 32), N = rup(OC * K * K, 16);
            _Float16 *B = calloc((size_t)N * Kd, 2), *A = calloc((size_t)M * Kd, 2);
            _Float16 *C = malloc((size_t)M * N * 2);
            float *acc = malloc(no * sizeof *acc);
            for (int ic = 0; ic < IC; ic++)             /* B[(oc,kh,kw)][ic] = W[ic][oc][kh][kw] */
                for (int n = 0; n < OC * K * K; n++)
                    B[(size_t)n * Kd + ic] = W[(size_t)ic * OC * K * K + n];
            double tp0 = now_ms();
            rocket_weights *w = rocket_weights_pack(mctx, M, Kd, N, B);
            double tpack = now_ms() - tp0;
            if (!w) { printf("  col2im: weights_pack failed (M %d K %d N %d)\n", M, Kd, N); goto c2; }
            int bad = 0;
            for (int r = 0; r <= reps && !bad; r++) {
                double t0 = now_ms();
                for (int ic = 0; ic < IC; ic++)        /* A[m][ic] = in[ic][m] */
                    for (int m = 0; m < IH * IW; m++)
                        A[(size_t)m * Kd + ic] = in[(size_t)ic * IH * IW + m];
                double t1 = now_ms();
                if (rocket_matmul_fp16_prepacked(mctx, M, Kd, N, A, C, w)) { bad = 1; break; }
                double t2 = now_ms();
                memset(acc, 0, no * sizeof *acc);
                for (int ih = 0; ih < IH; ih++)
                    for (int iw = 0; iw < IW; iw++) {
                        const _Float16 *c = C + ((size_t)ih * IW + iw) * N;
                        for (int oc = 0; oc < OC; oc++)
                            for (int kh = 0; kh < K; kh++) {
                                int ph = ih * S - P + kh;
                                if (ph < 0 || ph >= OH) continue;
                                float *arow = acc + ((size_t)oc * OH + ph) * OW;
                                const _Float16 *cr = c + ((size_t)oc * K + kh) * K;
                                for (int kw = 0; kw < K; kw++) {
                                    int pw = iw * S - P + kw;
                                    if (pw < 0 || pw >= OW) continue;
                                    arow[pw] += (float)cr[kw];
                                }
                            }
                    }
                for (size_t j = 0; j < no; j++) out[j] = (_Float16)acc[j];
                double t3 = now_ms();
                if (r > 0) {
                    t[(r - 1) * 4 + 0] = t1 - t0; t[(r - 1) * 4 + 1] = t2 - t1;
                    t[(r - 1) * 4 + 2] = t3 - t2; t[(r - 1) * 4 + 3] = t3 - t0;
                }
            }
            if (bad) printf("  col2im: matmul failed (M %d K %d N %d)\n", M, Kd, N);
            else {
                double ph[4];
                for (int q = 0; q < 4; q++) {
                    double v[64];
                    for (int r = 0; r < reps; r++) v[r] = t[r * 4 + q];
                    ph[q] = med(v, reps);
                }
                double e, er; score(&d, in, W, out, ref, &e, &er);
                printf("  col2im  M %5d K %4d N %5d  total %8.2f  [A^T %6.2f  mm %7.2f  col2im %6.2f]  pack-once %7.1f  max|err| %.3g  err/mag %.3g\n",
                       M, Kd, N, ph[3], ph[0], ph[1], ph[2], tpack, e, er);
            }
            rocket_weights_free(mctx, w);
        c2:
            free(B); free(A); free(C); free(acc);
        }

        if (mask & 2) {   /* ---- subpix (im2col over the compact input) ---- */
            const int T = (K + S - 1) / S;
            const int Q0 = P / S, Q1 = (OH - 1 + P) / S, nQ = Q1 - Q0 + 1;   /* square */
            const int Q0x = P / S, Q1x = (OW - 1 + P) / S, nQx = Q1x - Q0x + 1;
            const int M = rup(nQ * nQx, 4), Kd = rup(IC * T * T, 32), N = rup(S * S * OC, 16);
            _Float16 *B = calloc((size_t)N * Kd, 2), *A = calloc((size_t)M * Kd, 2);
            _Float16 *C = malloc((size_t)M * N * 2);
            for (int ry = 0; ry < S; ry++)
                for (int rx = 0; rx < S; rx++)
                    for (int oc = 0; oc < OC; oc++) {
                        const int n = (ry * S + rx) * OC + oc;
                        for (int ic = 0; ic < IC; ic++)
                            for (int th = 0; th < T; th++)
                                for (int tw = 0; tw < T; tw++) {
                                    int kh = ry + S * (T - 1 - th), kw = rx + S * (T - 1 - tw);
                                    if (kh >= K || kw >= K) continue;
                                    B[(size_t)n * Kd + ((size_t)ic * T + th) * T + tw] =
                                        W[(((size_t)ic * OC + oc) * K + kh) * K + kw];
                                }
                    }
            double tp0 = now_ms();
            rocket_weights *w = rocket_weights_pack(mctx, M, Kd, N, B);
            double tpack = now_ms() - tp0;
            if (!w) { printf("  subpix: weights_pack failed (M %d K %d N %d)\n", M, Kd, N); goto c3; }
            int bad = 0;
            for (int r = 0; r <= reps && !bad; r++) {
                double t0 = now_ms();
                for (int qy = 0; qy < nQ; qy++)
                    for (int qx = 0; qx < nQx; qx++) {
                        _Float16 *a = A + ((size_t)qy * nQx + qx) * Kd;
                        const int Qy = Q0 + qy, Qx = Q0x + qx;
                        for (int ic = 0; ic < IC; ic++)
                            for (int th = 0; th < T; th++) {
                                int ih = Qy - (T - 1) + th;
                                for (int tw = 0; tw < T; tw++) {
                                    int iw = Qx - (T - 1) + tw;
                                    a[((size_t)ic * T + th) * T + tw] =
                                        (ih < 0 || ih >= IH || iw < 0 || iw >= IW) ? (_Float16)0 :
                                        in[((size_t)ic * IH + ih) * IW + iw];
                                }
                            }
                    }
                double t1 = now_ms();
                if (rocket_matmul_fp16_prepacked(mctx, M, Kd, N, A, C, w)) { bad = 1; break; }
                double t2 = now_ms();
                for (int oc = 0; oc < OC; oc++)
                    for (int oh = 0; oh < OH; oh++) {
                        const int qy = (oh + P) / S - Q0, ry = (oh + P) % S;
                        for (int ow = 0; ow < OW; ow++) {
                            const int qx = (ow + P) / S - Q0x, rx = (ow + P) % S;
                            out[((size_t)oc * OH + oh) * OW + ow] =
                                C[((size_t)qy * nQx + qx) * N + (ry * S + rx) * OC + oc];
                        }
                    }
                double t3 = now_ms();
                if (r > 0) {
                    t[(r - 1) * 4 + 0] = t1 - t0; t[(r - 1) * 4 + 1] = t2 - t1;
                    t[(r - 1) * 4 + 2] = t3 - t2; t[(r - 1) * 4 + 3] = t3 - t0;
                }
            }
            if (bad) printf("  subpix: matmul failed (M %d K %d N %d)\n", M, Kd, N);
            else {
                double ph[4];
                for (int q = 0; q < 4; q++) {
                    double v[64];
                    for (int r = 0; r < reps; r++) v[r] = t[r * 4 + q];
                    ph[q] = med(v, reps);
                }
                double e, er; score(&d, in, W, out, ref, &e, &er);
                printf("  subpix  M %5d K %4d N %5d  total %8.2f  [im2col %6.2f  mm %7.2f  d2s %6.2f]  pack-once %7.1f  max|err| %.3g  err/mag %.3g\n",
                       M, Kd, N, ph[3], ph[0], ph[1], ph[2], tpack, e, er);
            }
            rocket_weights_free(mctx, w);
        c3:
            free(B); free(A); free(C);
        }

        if (mask & 4) {   /* ---- subconv: the sub-pixel form on the conv tiler ---- */
            const int T = (K + S - 1) / S;
            const int Q0 = P / S, Q1 = (OH - 1 + P) / S, nQ = Q1 - Q0 + 1;
            const int Q0x = P / S, Q1x = (OW - 1 + P) / S, nQx = Q1x - Q0x + 1;
            /* padded compact input: row r of xp is input row Q0 - (T-1) + r */
            const int PH = nQ + T - 1, PW = nQx + T - 1, OCs = S * S * OC;
            _Float16 *xp = calloc((size_t)IC * PH * PW, 2);
            _Float16 *wc = calloc((size_t)OCs * IC * T * T, 2), *C = malloc((size_t)OCs * nQ * nQx * 2);
            for (int ic = 0; ic < IC; ic++)
                for (int r = 0; r < PH; r++) {
                    int ih = Q0 - (T - 1) + r;
                    if (ih < 0 || ih >= IH) continue;
                    for (int c = 0; c < PW; c++) {
                        int iw = Q0x - (T - 1) + c;
                        if (iw < 0 || iw >= IW) continue;
                        xp[((size_t)ic * PH + r) * PW + c] = in[((size_t)ic * IH + ih) * IW + iw];
                    }
                }
            for (int ry = 0; ry < S; ry++)
                for (int rx = 0; rx < S; rx++)
                    for (int oc = 0; oc < OC; oc++)
                        for (int ic = 0; ic < IC; ic++)
                            for (int th = 0; th < T; th++)
                                for (int tw = 0; tw < T; tw++) {
                                    int kh = ry + S * (T - 1 - th), kw = rx + S * (T - 1 - tw);
                                    if (kh >= K || kw >= K) continue;
                                    wc[((((size_t)(ry * S + rx) * OC + oc) * IC + ic) * T + th) * T + tw] =
                                        W[(((size_t)ic * OC + oc) * K + kh) * K + kw];
                                }
            rocket_conv2d_desc cd;
            memset(&cd, 0, sizeof cd);
            cd.ic = IC; cd.ih = PH; cd.iw = PW; cd.oc = OCs; cd.kh = cd.kw = T;
            cd.stride_y = cd.stride_x = 1; cd.dil_y = cd.dil_x = 1;
            int bad = rocket_conv2d_plan(&cd);
            for (int r = 0; r <= reps && !bad; r++) {
                double t0 = now_ms();
                if (rocket_conv2d_fp16_ctx(cctx, &cd, xp, wc, C)) { bad = 1; break; }
                double t1 = now_ms();
                for (int oc = 0; oc < OC; oc++)
                    for (int oh = 0; oh < OH; oh++) {
                        const int qy = (oh + P) / S - Q0, ry = (oh + P) % S;
                        for (int ow = 0; ow < OW; ow++) {
                            const int qx = (ow + P) / S - Q0x, rx = (ow + P) % S;
                            out[((size_t)oc * OH + oh) * OW + ow] =
                                C[(((size_t)(ry * S + rx) * OC + oc) * nQ + qy) * nQx + qx];
                        }
                    }
                double t2 = now_ms();
                if (r > 0) { t[(r - 1) * 4 + 0] = t1 - t0; t[(r - 1) * 4 + 1] = t2 - t1;
                             t[(r - 1) * 4 + 3] = t2 - t0; }
            }
            if (bad) printf("  subconv: plan/run refused (%d)\n", bad);
            else {
                double ph[4] = {0};
                for (int q = 0; q < 4; q++) {
                    if (q == 2) continue;
                    double v[64];
                    for (int r = 0; r < reps; r++) v[r] = t[r * 4 + q];
                    ph[q] = med(v, reps);
                }
                double e, er; score(&d, in, W, out, ref, &e, &er);
                printf("  subconv IC %4d %dx%d k%d OC %5d  total %8.2f  [conv %7.2f  d2s %6.2f]  max|err| %.3g  err/mag %.3g\n",
                       IC, PH, PW, T, OCs, ph[3], ph[0], ph[1], e, er);
            }
            free(xp); free(wc); free(C);
        }
        free(in); free(W); free(out); free(ref);
        fflush(stdout);
    }
    free(t);
    rocket_conv_ctx_free(cctx);
    rocket_ctx_free(mctx);
    rocket_close(fd);
    return 0;
}
