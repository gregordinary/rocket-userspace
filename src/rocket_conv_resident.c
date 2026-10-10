// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_conv_resident.c — the resident-weight forward convolution: a host im2col into
 * channel planes feeding the resident fp16 matmul.
 *
 * A convolution is one GEMM over its unfolded input. With the input unfolded into
 * K = IC*KH*KW planes of the OH*OW output pixels,
 *
 *   A[(ic,ky,kx)][oy*OW + ox] = in[ic][oy*sy + ky*dy - pt][ox*sx + kx*dx - pl]   (0 outside)
 *
 * and the weight as it is stored, W[OC][IC*KH*KW] = B, the GEMM's output in channel-planes
 * form is the convolution's [OC][OH][OW] directly. The matmul is rocket_matmul.h's resident
 * fp16 matmul on a rocket_ctx, its weight packed once into the handle and fanned across the
 * context's worker fds; the input planes interleave straight into its input cube and its
 * output cube de-interleaves straight into the output planes, so no side takes a host
 * transpose.
 *
 * The unfold is the host's term. It runs on the context's worker count of threads, split by
 * input channel, each writing its channels' KH*KW planes as contiguous output-row runs: a
 * stride-1 row is a copy of an input row, a stride-2 row a copy out of the input row's
 * even or odd half (deinterleaved once per row), anything else a gather. A 1x1 kernel at
 * stride 1 with no pad and no explicit extent has nothing to unfold: its input planes ARE
 * the GEMM's A, read in place.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_conv_internal.h"
#include "rocket_hw_profile.h"
#include "rocket_matmul.h"
#include "rocket_matmul_internal.h"   /* the resident matmul's channel-planes form */
#include "rocket_affinity.h"
#include "rocket_fanout.h"
#include "rocket_log.h"

/* The unfold buffer (K planes of M) and an unaligned output's staging (N planes of M);
 * past this the plan refuses rather than allocate them (a banded GEMM would lift it). */
#define RC_SCRATCH_MAX ((size_t)64 << 20)
/* Below this many unfolded elements the unfold runs on the calling thread: a thread's
 * spawn and join cost more than its share of a copy this small. */
#define RC_UNFOLD_MT_MIN ((size_t)1 << 17)
#define RC_UNFOLD_MAX_THREADS 8

struct rocket_conv2d_weights {
    rocket_conv2d_desc d;
    int OH, OW;
    int ml, kl;                  /* live GEMM rows (OH*OW) and planes (IC*KH*KW) */
    int M, K, N;                 /* the GEMM, aligned */
    int in_place;                /* 1x1, stride 1, no pad: the input planes are A */
    int direct_out;              /* N == OC and M == OH*OW: the GEMM writes `out` itself */
    int nthr;                    /* unfold threads */
    struct rocket_ctx *ctx;      /* the context the weight lives on */
    struct rocket_weights *w;    /* resident B = W */
    int split_m;                 /* the workers split M (1) or N (0) */
    _Float16 *at;                /* the unfold, kl planes of M; NULL when in_place */
    _Float16 *ct;                /* N planes of M; NULL when direct_out */
    _Float16 *rows;              /* per-thread deinterleave rows, 2 * half_iw each */
    int half_iw;
    int *xcol;                   /* per kx: the output columns [lo, hi) whose input column is
                                  * inside the row, and lo's input column; 3 * KW entries */
    float *bias;                 /* OC entries, or NULL */
};

static int rc_rup(int x, int a) { return (x + a - 1) / a * a; }
static int rc_floordiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
static int rc_ceildiv(int a, int b) { return -rc_floordiv(-a, b); }
static int rc_dy(const rocket_conv2d_desc *d) { return d->dil_y ? d->dil_y : 1; }
static int rc_dx(const rocket_conv2d_desc *d) { return d->dil_x ? d->dil_x : 1; }

static int rc_in_place(const rocket_conv2d_desc *d)
{
    return d->kh == 1 && d->kw == 1 && d->stride_y == 1 && d->stride_x == 1 &&
           d->pad_top == 0 && d->pad_left == 0 &&
           rocket_conv2d_oh(d) == d->ih && rocket_conv2d_ow(d) == d->iw;
}

/* ROCKET_CONV_PROFILE=1: one line at exit for this entry, its calls' unfold, GEMM and
 * output-pass time, all on the calling thread, so they sum to the entry's wall. */
static struct { double unfold, gemm, out; long calls; } g_rcprof;
static pthread_mutex_t g_rcprof_mu = PTHREAD_MUTEX_INITIALIZER;
static int rc_prof_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) v = getenv("ROCKET_CONV_PROFILE") != NULL;
    return v;
}
static double rc_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static void rc_prof_dump(void)
{
    ROCKET_LOGI("ROCKET conv profile conv-resident total(ms): unfold=%.1f gemm=%.1f out=%.1f "
                "over %ld calls\n", g_rcprof.unfold, g_rcprof.gemm, g_rcprof.out, g_rcprof.calls);
}
static void rc_prof_add(double unfold, double gemm, double out)
{
    pthread_mutex_lock(&g_rcprof_mu);
    if (!g_rcprof.calls) atexit(rc_prof_dump);
    g_rcprof.unfold += unfold; g_rcprof.gemm += gemm; g_rcprof.out += out; g_rcprof.calls++;
    pthread_mutex_unlock(&g_rcprof_mu);
}

int rocket_conv2d_prepacked_plan(const rocket_conv2d_desc *d)
{
    if (!d) return ROCKET_E_SHAPE;
    if (d->ic <= 0 || d->ih <= 0 || d->iw <= 0 || d->oc <= 0 || d->kh <= 0 || d->kw <= 0 ||
        d->stride_x <= 0 || d->stride_y <= 0 || d->dil_x < 0 || d->dil_y < 0 ||
        d->pad_top < 0 || d->pad_left < 0 || d->oh < 0 || d->ow < 0)
        return ROCKET_E_SHAPE;
    /* A depthwise conv is no GEMM; rocket_conv2d_fp16 runs it. */
    if (d->depthwise) return ROCKET_E_UNSUPPORTED;
    if (strcmp(rocket_hw_current()->name, "rk3588") != 0) return ROCKET_E_UNSUPPORTED;
    const long long OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    if (OH <= 0 || OW <= 0) return -3;
    const long long pix = OH * OW, taps = (long long)d->ic * d->kh * d->kw;
    if (pix > (1 << 24) || taps > (1 << 24) || d->oc > (1 << 24)) return -4;
    const int M = rc_rup((int)pix, 4), K = rc_rup((int)taps, 32), N = rc_rup(d->oc, 16);
    if (!rc_in_place(d) && (size_t)K * M * sizeof(_Float16) > RC_SCRATCH_MAX) return -4;
    if ((N != d->oc || M != pix) && (size_t)N * M * sizeof(_Float16) > RC_SCRATCH_MAX) return -4;
    if (rocket_matmul_plan(M, K, N, NULL, NULL, NULL) < 0) return -4;
    return 0;
}

/* The per-call estimate: a linear model in the GEMM's aligned extents, least squares
 * weighted by relative error over 21 measured shapes (rocket_conv.h has the fit set and
 * its spread). Terms in ms: a fixed cost, the unfold's writes, the input cube (A), the
 * output (C), the weight (B), and the MACs. */
double rocket_conv2d_prepacked_estimate_ms(const rocket_conv2d_desc *d)
{
    if (rocket_conv2d_prepacked_plan(d)) return -1.0;
    const double M = rc_rup(rocket_conv2d_oh(d) * rocket_conv2d_ow(d), 4);
    const double K = rc_rup(d->ic * d->kh * d->kw, 32), N = rc_rup(d->oc, 16);
    const double unfold = rc_in_place(d) ? 0.0 : K * M;
    return 0.4125 + 0.2279e-6 * unfold + 0.5874e-6 * K * M + 0.7280e-6 * N * M +
           0.1776e-6 * K * N + 2.7796e-9 * M * K * N;
}

rocket_conv2d_weights *rocket_conv2d_weights_pack(struct rocket_ctx *ctx, const rocket_conv2d_desc *d,
                                                  const _Float16 *W, const _Float16 *bias)
{
    if (!ctx || !d || !W) return NULL;
    int r = rocket_conv2d_prepacked_plan(d);
    if (r) {
        ROCKET_LOGE("rocket_conv2d_weights_pack: plan refused (%d)\n", r);
        return NULL;
    }
    rocket_conv2d_weights *h = calloc(1, sizeof *h);
    if (!h) return NULL;
    h->d = *d;
    h->ctx = ctx;
    h->OH = rocket_conv2d_oh(d);
    h->OW = rocket_conv2d_ow(d);
    h->ml = h->OH * h->OW;
    h->kl = d->ic * d->kh * d->kw;
    h->M = rc_rup(h->ml, 4);
    h->K = rc_rup(h->kl, 32);
    h->N = rc_rup(d->oc, 16);
    h->in_place = rc_in_place(d);
    h->direct_out = h->N == d->oc && h->M == h->ml;

    int nthr = rocket_ctx_nthreads(ctx);
    const char *e = getenv("ROCKET_IM2COL_THREADS");
    if (e && atoi(e) >= 1) nthr = atoi(e);
    if (nthr > RC_UNFOLD_MAX_THREADS) nthr = RC_UNFOLD_MAX_THREADS;
    if (nthr > d->ic) nthr = d->ic;
    if ((size_t)h->kl * h->ml < RC_UNFOLD_MT_MIN || nthr < 1) nthr = 1;
    h->nthr = nthr;
    h->half_iw = (d->iw + 1) / 2 + 8;

    /* B[oc][k] = W[oc][ic][ky][kx]: the stored weight is B, padded to K. Pack-time only. */
    _Float16 *B = calloc((size_t)h->N * h->K, sizeof *B);
    if (!h->in_place) {
        h->at = malloc((size_t)h->kl * h->M * sizeof *h->at);
        h->rows = malloc((size_t)nthr * 2 * h->half_iw * sizeof *h->rows);
        h->xcol = malloc((size_t)3 * d->kw * sizeof *h->xcol);
    }
    if (!h->direct_out) h->ct = malloc((size_t)h->N * h->M * sizeof *h->ct);
    if (bias) h->bias = malloc((size_t)d->oc * sizeof *h->bias);
    if (!B || (!h->in_place && (!h->at || !h->rows || !h->xcol)) || (!h->direct_out && !h->ct) ||
        (bias && !h->bias))
        goto fail;
    if (!h->in_place) {
        const int sx = d->stride_x, dx = rc_dx(d), IW = d->iw, OW = h->OW, KW = d->kw;
        for (int kx = 0; kx < KW; kx++) {
            const int ix0 = kx * dx - d->pad_left;
            int lo = ix0 >= 0 ? 0 : rc_ceildiv(-ix0, sx);
            int hi = rc_floordiv(IW - 1 - ix0, sx) + 1;
            if (hi > OW) hi = OW;
            if (lo > OW) lo = OW;
            if (hi < lo) hi = lo;
            h->xcol[kx] = lo; h->xcol[KW + kx] = hi; h->xcol[2 * KW + kx] = lo * sx + ix0;
        }
    }
    for (int oc = 0; oc < d->oc; oc++)
        memcpy(B + (size_t)oc * h->K, W + (size_t)oc * h->kl, (size_t)h->kl * sizeof *B);
    if (bias)
        for (int oc = 0; oc < d->oc; oc++) h->bias[oc] = (float)bias[oc];
    h->split_m = rocket_conv_resident_split_m(ctx, h->M, h->K, h->N, "ROCKET_CONV_SPLIT");
    h->w = rocket_weights_pack_split(ctx, h->M, h->K, h->N, B, h->split_m);
    if (!h->w) goto fail;
    free(B);
    return h;
fail:
    free(B);
    free(h->at); free(h->ct); free(h->rows); free(h->xcol); free(h->bias);
    free(h);
    return NULL;
}

void rocket_conv2d_weights_free(struct rocket_ctx *ctx, rocket_conv2d_weights *h)
{
    if (!h) return;
    if (h->w) rocket_weights_free(ctx ? ctx : h->ctx, h->w);
    free(h->at); free(h->ct); free(h->rows); free(h->xcol); free(h->bias);
    free(h);
}

/* ev[j] = r[2j], od[j] = r[2j+1]: one input row split into its even and odd halves. */
static void rc_deinterleave(const _Float16 *restrict r, int n, _Float16 *restrict ev,
                            _Float16 *restrict od)
{
    int j = 0;
#if defined(__aarch64__)
    for (; 2 * j + 16 <= n; j += 8) {
        const float16x8x2_t v = vld2q_f16((const __fp16 *)(r + 2 * j));
        vst1q_f16((__fp16 *)(ev + j), v.val[0]);
        vst1q_f16((__fp16 *)(od + j), v.val[1]);
    }
#endif
    for (; 2 * j < n; j++) {
        ev[j] = r[2 * j];
        if (2 * j + 1 < n) od[j] = r[2 * j + 1];
    }
}

typedef struct {
    const rocket_conv2d_weights *h;
    const _Float16 *in;
    int ic0, ic1;
    int idx, core_base;
} rc_unfold_arg;

/* The unfold of input channels [ic0, ic1): each (ky, kx) plane row by row. */
static void rc_unfold_range(const rocket_conv2d_weights *h, const _Float16 *in, int ic0, int ic1,
                            _Float16 *ev, _Float16 *od)
{
    const rocket_conv2d_desc *d = &h->d;
    const int IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw, sy = d->stride_y, sx = d->stride_x;
    const int dy = rc_dy(d), OH = h->OH, OW = h->OW;
    const size_t M = (size_t)h->M;
    const int *xlo = h->xcol, *xhi = h->xcol + KW, *x0 = h->xcol + 2 * KW;
    for (int ic = ic0; ic < ic1; ic++) {
        const _Float16 *src = in + (size_t)ic * IH * IW;
        for (int ky = 0; ky < KH; ky++) {
            const int iy0 = ky * dy - d->pad_top;
            int ylo = iy0 >= 0 ? 0 : rc_ceildiv(-iy0, sy);
            int yhi = rc_floordiv(IH - 1 - iy0, sy) + 1;
            if (yhi > OH) yhi = OH;
            if (ylo > OH) ylo = OH;
            if (yhi < ylo) yhi = ylo;
            _Float16 *pk = h->at + ((size_t)(ic * KH + ky) * KW) * M;
            for (int kx = 0; kx < KW; kx++) {
                _Float16 *pl = pk + (size_t)kx * M;
                memset(pl, 0, (size_t)ylo * OW * sizeof *pl);
                memset(pl + (size_t)yhi * OW, 0, (size_t)(OH - yhi) * OW * sizeof *pl);
            }
            for (int oy = ylo; oy < yhi; oy++) {
                const _Float16 *r = src + (size_t)(oy * sy + iy0) * IW;
                if (sx == 2 && OW >= 16) rc_deinterleave(r, IW, ev, od);
                for (int kx = 0; kx < KW; kx++) {
                    _Float16 *o = pk + (size_t)kx * M + (size_t)oy * OW;
                    const int lo = xlo[kx], hi = xhi[kx], n = hi - lo, ix = x0[kx];
                    if (OW < 16) {   /* a short row: three library calls cost more than it */
                        int j = 0;
                        for (; j < lo; j++) o[j] = 0;
                        for (; j < hi; j++) o[j] = r[ix + (j - lo) * sx];
                        for (; j < OW; j++) o[j] = 0;
                        continue;
                    }
                    memset(o, 0, (size_t)lo * sizeof *o);
                    memset(o + hi, 0, (size_t)(OW - hi) * sizeof *o);
                    if (sx == 1)
                        memcpy(o + lo, r + ix, (size_t)n * sizeof *o);
                    else if (sx == 2)
                        memcpy(o + lo, ((ix & 1) ? od : ev) + ix / 2, (size_t)n * sizeof *o);
                    else
                        for (int j = 0; j < n; j++) o[lo + j] = r[ix + j * sx];
                }
            }
        }
    }
}

static void *rc_unfold_thread(void *a)
{
    rc_unfold_arg *t = a;
    rocket_pin_worker_based(t->idx, t->core_base);
    _Float16 *ev = t->h->rows + (size_t)t->idx * 2 * t->h->half_iw;
    rc_unfold_range(t->h, t->in, t->ic0, t->ic1, ev, ev + t->h->half_iw);
    return NULL;
}

static void rc_unfold(const rocket_conv2d_weights *h, const _Float16 *in)
{
    const int IC = h->d.ic, nt = h->nthr;
    if (nt <= 1) {
        rc_unfold_range(h, in, 0, IC, h->rows, h->rows + h->half_iw);
        return;
    }
    rc_unfold_arg args[RC_UNFOLD_MAX_THREADS];
    const int base = rocket_affinity_get_base();
    for (int t = 0; t < nt; t++)
        args[t] = (rc_unfold_arg){ h, in, (int)((long)IC * t / nt), (int)((long)IC * (t + 1) / nt),
                                   t, base };
    rocket_fanout_run(nt, args, sizeof args[0], rc_unfold_thread);
}

/* out[i] = fp16((float)src[i] + b) over n elements, one rounding; out may be src. */
static void rc_add_bias(_Float16 *out, const _Float16 *src, size_t n, float b)
{
    size_t i = 0;
#if defined(__aarch64__)
    const float32x4_t bb = vdupq_n_f32(b);
    for (; i + 8 <= n; i += 8) {
        const float16x8_t v = vld1q_f16((const __fp16 *)(src + i));
        vst1q_f16((__fp16 *)(out + i), vcombine_f16(
            vcvt_f16_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(v)), bb)),
            vcvt_f16_f32(vaddq_f32(vcvt_high_f32_f16(v), bb))));
    }
#endif
    for (; i < n; i++) out[i] = (_Float16)((float)src[i] + b);
}

int rocket_conv2d_fp16_prepacked(struct rocket_ctx *ctx, rocket_conv2d_weights *h,
                                 const _Float16 *in, _Float16 *out)
{
    if (!ctx || !h || !in || !out) return ROCKET_E_SHAPE;
    if (ctx != h->ctx) {
        ROCKET_LOGE("rocket_conv2d_fp16_prepacked: the weight was packed on another context\n");
        return ROCKET_E_SHAPE;
    }
    const int prof = rc_prof_on();
    const double t0 = prof ? rc_now_ms() : 0.0;
    const _Float16 *A = in;
    size_t lda = (size_t)h->d.ih * h->d.iw;
    if (!h->in_place) {
        rc_unfold(h, in);
        A = h->at;
        lda = (size_t)h->M;
    }
    const double t1 = prof ? rc_now_ms() : 0.0;
    _Float16 *C = h->direct_out ? out : h->ct;
    int r = rocket_matmul_fp16_prepacked_planes(ctx, h->M, h->K, h->N, A, lda, h->ml, h->kl, C,
                                                h->w);
    if (r) return r;
    const double t2 = prof ? rc_now_ms() : 0.0;
    const size_t plane = (size_t)h->ml;
    if (!h->direct_out || h->bias)
        for (int oc = 0; oc < h->d.oc; oc++) {
            const _Float16 *src = C + (size_t)oc * h->M;
            _Float16 *dst = out + (size_t)oc * plane;
            if (h->bias) rc_add_bias(dst, src, plane, h->bias[oc]);
            else         memcpy(dst, src, plane * sizeof *dst);
        }
    if (prof) rc_prof_add(t1 - t0, t2 - t1, rc_now_ms() - t2);
    return 0;
}
