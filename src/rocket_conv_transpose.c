// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_conv_transpose.c — transposed convolution (ConvTranspose2d / "deconvolution")
 * on the rocket NPU.
 *
 * Two routes, chosen per descriptor by rocket_conv_transpose2d_route(). A direct conv at
 * power-of-two strides whose compact input fits one CBUF pass runs on the CNA's hardware
 * deconvolution mode (rocket_conv2d_fp16_deconv, rocket_conv.c): the part dilates the
 * input itself, the output extent is programmed as the transposed one, and the pad is
 * k-1-p. Everything else is lowered onto the validated forward CONV_2D engine, as below.
 *
 * ConvTranspose is the transpose (gradient) of a strided conv: every input pixel
 * scatter-adds a kernel-weighted copy into a larger output. The standard identity is
 *
 *   ConvTranspose(X; W, stride s, pad p, dil d, opad)
 *     == Conv( dilate_and_pad(X), rot180(W^T); stride 1, pad 0, dil d )
 *
 * where dilate_and_pad inserts (s-1) zero rows/cols *between* input pixels (interior
 * dilation), then borders the result with  lead = d*(K-1) - p  on the leading edge and
 * lead + opad on the trailing edge; and rot180(W^T) flips the kernel spatially AND
 * transposes its in/out channels:  wf[oc][ic][kh][kw] = W[ic][oc][K-1-kh][K-1-kw].
 *
 * Derivation (one axis, the others identical):
 *   ConvTranspose puts in[ih] into output position ph = ih*s - p + kh*d.
 *   In the lowered input xd, in[ih] sits at row r = lead + ih*s (lead = d*(K-1) - p).
 *   The forward conv reads xd[ph + kf*d] for forward-kernel index kf; that equals
 *   lead + ih*s exactly when kf = K-1-kh. So the forward kernel index K-1-kh carries
 *   ConvTranspose weight kh -> the 180-deg flip. xd is zero off the stride lattice and
 *   in the border, so only integral, in-range ih contribute (zero padding handles the
 *   rest). Output size matches:  IHd - d*(K-1) = (IH-1)*s + d*(K-1) - 2p + opad + 1 = OH.
 *
 * The forward conv is the HW-validated rocket_conv2d_fp16 (auto-tiled over OC/OH/OW),
 * so the transpose inherits it bit-for-bit. The only NPU-specific work is host-side
 * (dilate the input, flip the weights); the cost scales with the *upsampled* size
 * because the inserted zeros are still MAC'd. The hardware route above is 0.35-0.54x
 * this lowering's wall per call at decoder shapes.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "rocket_conv.h"
#include "rocket_conv_internal.h"
#include "rocket_hw_profile.h"
#include "rocket_matmul.h"
#include "rocket_matmul_internal.h"   /* the resident matmul's channel-planes form */
#include "rocket_log.h"

/* ============================================================================
 * SECTION — Lowering + plan (forward-conv descriptor, validation)
 * ==========================================================================*/

/* Build the forward-conv descriptor that the transpose lowers to: stride-1, pad-0,
 * same dilation, over the dilated+bordered input. Also returns the lowered input dims
 * (IHd/IWd) and the leading border (lead_y/lead_x) via out-params. Pure. */
static void lower_desc(const rocket_conv_transpose2d_desc *d, rocket_conv2d_desc *fwd,
                       int *IHd, int *IWd, int *lead_y, int *lead_x)
{
    const int ly = d->dil_y * (d->kh - 1) - d->pad_top;
    const int lx = d->dil_x * (d->kw - 1) - d->pad_left;
    const int ty = ly + d->opad_y;                 /* trailing border = leading + opad */
    const int tx = lx + d->opad_x;
    const int core_h = (d->ih - 1) * d->stride_y + 1;   /* interior-dilated input height */
    const int core_w = (d->iw - 1) * d->stride_x + 1;

    *lead_y = ly;
    *lead_x = lx;
    *IHd = core_h + ly + ty;
    *IWd = core_w + lx + tx;

    /* Zero first: the fields this lowering does not set — `oh`/`ow` and the RK3576
     * encoding flags — carry the contract "zero means derive it", so a caller-supplied
     * struct must start zeroed. Both call sites declare `fwd` on the stack. */
    memset(fwd, 0, sizeof *fwd);
    fwd->ic = d->ic; fwd->ih = *IHd; fwd->iw = *IWd;
    fwd->oc = d->oc; fwd->kh = d->kh; fwd->kw = d->kw;
    fwd->stride_y = 1; fwd->stride_x = 1;
    fwd->pad_top = 0;  fwd->pad_left = 0;
    fwd->dil_y = d->dil_y; fwd->dil_x = d->dil_x;
    fwd->depthwise = d->depthwise;
}

/* The descriptor checks shared by both routes. 0 or a negative reason. Pure. */
static int desc_valid(const rocket_conv_transpose2d_desc *d)
{
    if (!d) return -1;
    if (d->ic <= 0 || d->ih <= 0 || d->iw <= 0 || d->oc <= 0 || d->kh <= 0 || d->kw <= 0)
        return -1;
    if (d->stride_x <= 0 || d->stride_y <= 0 || d->dil_x <= 0 || d->dil_y <= 0)
        return -1;
    if (d->pad_top < 0 || d->pad_left < 0 || d->opad_y < 0 || d->opad_x < 0)
        return -1;
    if (d->depthwise && d->oc != d->ic) return -1;   /* depthwise: one kernel per channel */
    /* A pad larger than the (dilated) kernel reach would CROP the output: the lowered
     * leading border goes negative, which this bring-up does not implement. */
    if (d->pad_top  > d->dil_y * (d->kh - 1)) return -2;
    if (d->pad_left > d->dil_x * (d->kw - 1)) return -2;
    if (rocket_conv_transpose2d_oh(d) <= 0 || rocket_conv_transpose2d_ow(d) <= 0)
        return -3;
    return 0;
}

static int pow2_stride(int s) { return s == 1 || s == 2 || s == 4 || s == 8; }

/* The hardware route's own conditions, beyond desc_valid(). The CNA deconvolution mode
 * dilates by a power-of-two stride only, has no depthwise or dilated form measured, and
 * its pad field (k-1-p) has an undecoded anomaly at 12-15 when s = 2. */
static int deconv_eligible(const rocket_conv_transpose2d_desc *d)
{
    const char *e = getenv("ROCKET_CONV_TRANSPOSE_HW");
    if (e && *e == '0') return 0;
    if (strcmp(rocket_hw_current()->name, "rk3588") != 0) return 0;
    if (d->depthwise || d->dil_y != 1 || d->dil_x != 1) return 0;
    if (!pow2_stride(d->stride_y) || !pow2_stride(d->stride_x)) return 0;
    if (d->stride_y == 1 && d->stride_x == 1) return 0;     /* nothing to dilate */
    if (d->opad_y >= d->stride_y || d->opad_x >= d->stride_x) return 0;
    if (d->kh - 1 - d->pad_top > 11 || d->kw - 1 - d->pad_left > 11) return 0;
    return rocket_conv2d_fp16_deconv_fits(d->ic, d->ih, d->iw, d->oc,
                                          rocket_conv_transpose2d_oh(d),
                                          rocket_conv_transpose2d_ow(d), d->kh, d->kw);
}

int rocket_conv_transpose2d_route(const rocket_conv_transpose2d_desc *d)
{
    int r = desc_valid(d);
    if (r) return r;
    if (deconv_eligible(d)) return ROCKET_CONV_TRANSPOSE_DECONV;

    rocket_conv2d_desc fwd; int IHd, IWd, ly, lx;
    lower_desc(d, &fwd, &IHd, &IWd, &ly, &lx);
    r = rocket_conv2d_plan(&fwd);                 /* propagate CBUF-fit / alignment */
    return r ? (r == -4 ? -4 : r) : ROCKET_CONV_TRANSPOSE_LOWERED;
}

int rocket_conv_transpose2d_plan(const rocket_conv_transpose2d_desc *d)
{
    int r = rocket_conv_transpose2d_route(d);
    return r < 0 ? r : 0;
}

/* ============================================================================
 * SECTION — Host reference (scatter-add oracle)
 * ==========================================================================*/

void rocket_conv_transpose2d_ref_fp16(const rocket_conv_transpose2d_desc *d,
                                      const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    const int IC = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv_transpose2d_oh(d), OW = rocket_conv_transpose2d_ow(d);

    /* fp32 accumulator per output element, then narrow to fp16 once (matches the NPU's
     * fp32-accumulate / fp16-store). Scatter-add straight from the definition. */
    float *acc = calloc((size_t)OC * OH * OW, sizeof(float));
    if (!acc) return;

    const int DW = d->depthwise;
    const int wic_span = DW ? 1 : OC;          /* weight is [IC][OC..] direct / [C][1..] DW */
    for (int ic = 0; ic < IC; ic++) {
        const int oc_lo = DW ? ic : 0, oc_hi = DW ? ic + 1 : OC;
        for (int ih = 0; ih < IH; ih++) {
            for (int iw = 0; iw < IW; iw++) {
                float x = (float)in[((size_t)ic * IH + ih) * IW + iw];
                if (x == 0.f) continue;
                for (int oc = oc_lo; oc < oc_hi; oc++) {
                    const int woc = DW ? 0 : oc;     /* weight oc index (DW: the singleton) */
                    for (int kh = 0; kh < KH; kh++) {
                        int ph = ih * d->stride_y - d->pad_top + kh * d->dil_y;
                        if (ph < 0 || ph >= OH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int pw = iw * d->stride_x - d->pad_left + kw * d->dil_x;
                            if (pw < 0 || pw >= OW) continue;
                            float w = (float)W[(((size_t)ic * wic_span + woc) * KH + kh) * KW + kw];
                            acc[((size_t)oc * OH + ph) * OW + pw] += x * w;
                        }
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < (size_t)OC * OH * OW; i++)
        out[i] = (_Float16)acc[i];
    free(acc);
}

/* ============================================================================
 * SECTION — NPU run (dilate input, flip weights, forward conv) + entry points
 * ==========================================================================*/

/* Shared body: build the lowered input + flipped weights, run the forward conv (on a
 * borrowed ctx if non-NULL, else a one-shot fd). Returns the forward conv's status. */
static int transpose_run(int fd, rocket_conv_ctx *ctx,
                         const rocket_conv_transpose2d_desc *d,
                         const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    const int route = rocket_conv_transpose2d_route(d);
    if (route < 0) return route;

    const int IC = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    rocket_conv2d_desc fwd; int IHd, IWd, ly, lx;
    lower_desc(d, &fwd, &IHd, &IWd, &ly, &lx);

    const int DW = d->depthwise;
    const int deconv = route == ROCKET_CONV_TRANSPOSE_DECONV;
    size_t wf_n = DW ? (size_t)IC * KH * KW : (size_t)OC * IC * KH * KW;
    /* The hardware route reads the compact input as it is, so only the lowering builds
     * the dilated+padded one. */
    _Float16 *xd = deconv ? NULL : calloc((size_t)IC * IHd * IWd, sizeof(_Float16));
    _Float16 *wf = malloc(wf_n * sizeof(_Float16));                    /* rot180 (+W^T direct) */
    if ((!deconv && !xd) || !wf) { free(xd); free(wf); return -5; }

    /* scatter the input onto the stride lattice inside the leading border */
    for (int ic = 0; ic < IC && !deconv; ic++)
        for (int ih = 0; ih < IH; ih++)
            for (int iw = 0; iw < IW; iw++) {
                int r = ly + ih * d->stride_y;
                int c = lx + iw * d->stride_x;
                xd[((size_t)ic * IHd + r) * IWd + c] = in[((size_t)ic * IH + ih) * IW + iw];
            }

    if (DW) {
        /* depthwise: wf[c][0][kh][kw] = W[c][0][KH-1-kh][KW-1-kw] (spatial flip only) */
        for (int c = 0; c < IC; c++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    wf[((size_t)c * KH + kh) * KW + kw] =
                        W[((size_t)c * KH + (KH - 1 - kh)) * KW + (KW - 1 - kw)];
    } else {
        /* direct: wf[oc][ic][kh][kw] = W[ic][oc][KH-1-kh][KW-1-kw] (180-flip + channel transpose) */
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < IC; ic++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        wf[(((size_t)oc * IC + ic) * KH + kh) * KW + kw] =
                            W[(((size_t)ic * OC + oc) * KH + (KH - 1 - kh)) * KW + (KW - 1 - kw)];
    }

    int rc;
    if (deconv)
        /* The pad on the dilated surface is k-1-p, which puts row 0 of the result at row
         * 0 of the transposed conv; the output extent is the transposed one. */
        rc = rocket_conv2d_fp16_deconv(fd, ctx, IC, IH, IW, OC,
                                       rocket_conv_transpose2d_oh(d),
                                       rocket_conv_transpose2d_ow(d), KH, KW,
                                       d->stride_y, d->stride_x,
                                       KH - 1 - d->pad_top, KW - 1 - d->pad_left, in, wf, out);
    else
        rc = ctx ? rocket_conv2d_fp16_ctx(ctx, &fwd, xd, wf, out)
                 : rocket_conv2d_fp16(fd, &fwd, xd, wf, out);
    free(xd);
    free(wf);
    return rc;
}

int rocket_conv_transpose2d_fp16(int fd, const rocket_conv_transpose2d_desc *d,
                                 const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    if (!d || !in || !W || !out) return -1;
    if (fd < 0) {                                  /* off-device: host oracle */
        rocket_conv_transpose2d_ref_fp16(d, in, W, out);
        return 0;
    }
    return transpose_run(fd, NULL, d, in, W, out);
}

int rocket_conv_transpose2d_fp16_ctx(rocket_conv_ctx *ctx,
                                     const rocket_conv_transpose2d_desc *d,
                                     const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    if (!ctx || !d || !in || !W || !out) return -1;
    return transpose_run(-1, ctx, d, in, W, out);
}

/* ============================================================================
 * SECTION — The resident entry: col2im over the resident fp16 matmul
 *
 * A transposed conv is one GEMM and a scatter-add. With the input read as a matmul's A^T
 * (IC planes of IH*IW pixels) and the weight as it is stored, W[IC][OC*KH*KW] = B^T,
 *
 *   P[(oc,kh,kw)][ih*IW + iw] = sum_ic in[ic][ih][iw] * W[ic][oc][kh][kw]
 *
 * is every tap of every input pixel, with no zero-MACs and no host im2col. Each plane then
 * adds into the output at the offset its tap puts it:
 *
 *   out[oc][ih*sy - pt + kh*dy][iw*sx - pl + kw*dx] += P[(oc,kh,kw)][ih][iw]
 *
 * which is the scatter reference's own loop order, so crop pads, dilation and
 * output_padding need nothing beyond the bounds. The GEMM is the resident fp16 matmul in
 * its channel-planes form: the input planes interleave straight into its input cube and
 * its output cube de-interleaves straight into P, so neither side takes a host transpose.
 * The weight is transposed and packed once, into the handle.
 *
 * The scatter-add runs per output channel over s_y*s_x PHASE planes: output pixel
 * (qy*sy + ry, qx*sx + rx) lives in phase (ry, rx) at (qy, qx), and every tap lands in
 * exactly one phase at a constant offset, so each tap is a run of contiguous row adds;
 * one final pass interleaves the phases into the output row and narrows it to fp16. A
 * kernel equal to its stride with no pad (SAM's upscalers) puts each output pixel under
 * one tap, and that case is a straight interleave of the planes.
 * ==========================================================================*/

/* The intermediate P is KH*KW/(sy*sx) times the output; past this the plan refuses rather
 * than allocate it (a banded GEMM would lift the bound; nothing measured needs it). */
#define CT_SCRATCH_MAX ((size_t)64 << 20)

struct rocket_conv_transpose2d_weights {
    rocket_conv_transpose2d_desc d;
    int M, K, N;                 /* the GEMM: M = IH*IW, K = IC, N = OC*KH*KW, aligned */
    int OH, OW, QH, QW;          /* output extent and one phase plane's (ceil(O/s)) */
    struct rocket_ctx *ctx;      /* the context the weight lives on */
    struct rocket_weights *w;    /* resident B = W^T */
    int split_m;                 /* the workers split M (1) or N (0) */
    _Float16 *ct;                /* P, N planes of M */
    float *acc;                  /* the phase planes of one output channel */
    float *bias;                 /* OC entries, or NULL */
};

static int ct_rup(int x, int a) { return (x + a - 1) / a * a; }

/* ROCKET_CONV_PROFILE=1: one line at exit for the resident entry, its calls' GEMM (the
 * matmul's own buckets are ROCKET_MM_PROFILE's) and scatter-add time, both on the
 * calling thread, so they sum to the entry's wall. */
static struct { double gemm, scatter; long calls; } g_ctprof;
static pthread_mutex_t g_ctprof_mu = PTHREAD_MUTEX_INITIALIZER;
static int ct_prof_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) v = getenv("ROCKET_CONV_PROFILE") != NULL;
    return v;
}
static double ct_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static void ct_prof_dump(void)
{
    ROCKET_LOGI("ROCKET conv profile convT-resident total(ms): gemm=%.1f scatter=%.1f over %ld "
                "calls\n", g_ctprof.gemm, g_ctprof.scatter, g_ctprof.calls);
}
static void ct_prof_add(double gemm, double scatter)
{
    pthread_mutex_lock(&g_ctprof_mu);
    if (!g_ctprof.calls) atexit(ct_prof_dump);
    g_ctprof.gemm += gemm; g_ctprof.scatter += scatter; g_ctprof.calls++;
    pthread_mutex_unlock(&g_ctprof_mu);
}
static int ct_floordiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

int rocket_conv_transpose2d_prepacked_plan(const rocket_conv_transpose2d_desc *d)
{
    if (!d) return ROCKET_E_SHAPE;
    if (d->ic <= 0 || d->ih <= 0 || d->iw <= 0 || d->oc <= 0 || d->kh <= 0 || d->kw <= 0 ||
        d->stride_x <= 0 || d->stride_y <= 0 || d->dil_x <= 0 || d->dil_y <= 0 ||
        d->pad_top < 0 || d->pad_left < 0 || d->opad_y < 0 || d->opad_x < 0)
        return ROCKET_E_SHAPE;
    /* A depthwise transpose is no GEMM; rocket_conv_transpose2d_fp16 runs it. */
    if (d->depthwise) return ROCKET_E_UNSUPPORTED;
    if (strcmp(rocket_hw_current()->name, "rk3588") != 0) return ROCKET_E_UNSUPPORTED;
    if (rocket_conv_transpose2d_oh(d) <= 0 || rocket_conv_transpose2d_ow(d) <= 0) return -3;
    const long long pix = (long long)d->ih * d->iw, taps = (long long)d->oc * d->kh * d->kw;
    if (pix > (1 << 24) || taps > (1 << 24)) return -4;
    const int M = ct_rup((int)pix, 4), K = ct_rup(d->ic, 32), N = ct_rup((int)taps, 16);
    if ((size_t)M * N * sizeof(_Float16) > CT_SCRATCH_MAX) return -4;
    if (rocket_matmul_plan(M, K, N, NULL, NULL, NULL) < 0) return -4;
    return 0;
}

/* Which way the context's workers split the GEMM. Under the N split the calling thread
 * packs all of A once, every worker copies it into its own BO and the NPU reads it once
 * per worker; under the M split each worker packs and reads only its rows, in parallel,
 * and holds all of B. The M split is taken once every worker has at least one whole
 * max_tile of rows and the replicated weight stays within CT_SPLIT_M_WT_MAX. Against the
 * N split, per call: 0.88x at M = 1024 (pix2pix up6, where B is twice A), 0.92x at 4096
 * (up7), 0.51x at 16384 (up8), 0.70x and 0.67x at SAM's 4096 and 16384 [HW sweep, RK1,
 * three rotated passes, 2026-09-27, tests/ct_model_bench.c]; 1.2-2.5x at M <= 256
 * (pix2pix up1-up5, one pass). ROCKET_CT_SPLIT=m|n forces one. */
#define CT_SPLIT_M_WT_MAX ((size_t)64 << 20)
static int ct_split_m(struct rocket_ctx *ctx, int M, int K, int N)
{
    const char *e = getenv("ROCKET_CT_SPLIT");
    if (e && (*e == 'm' || *e == 'M')) return 1;
    if (e && (*e == 'n' || *e == 'N')) return 0;
    const int nt = rocket_ctx_nthreads(ctx);
    return nt > 1 && M / nt >= rocket_hw_current()->max_tile &&
           (size_t)nt * K * N * sizeof(_Float16) <= CT_SPLIT_M_WT_MAX;
}

rocket_conv_transpose2d_weights *
rocket_conv_transpose2d_weights_pack(struct rocket_ctx *ctx, const rocket_conv_transpose2d_desc *d,
                                     const _Float16 *W, const _Float16 *bias)
{
    if (!ctx || !d || !W) return NULL;
    int r = rocket_conv_transpose2d_prepacked_plan(d);
    if (r) {
        ROCKET_LOGE("rocket_conv_transpose2d_weights_pack: plan refused (%d)\n", r);
        return NULL;
    }
    rocket_conv_transpose2d_weights *h = calloc(1, sizeof *h);
    if (!h) return NULL;
    h->d = *d;
    h->ctx = ctx;
    const int IC = d->ic, NN = d->oc * d->kh * d->kw;
    h->M = ct_rup(d->ih * d->iw, 4);
    h->K = ct_rup(IC, 32);
    h->N = ct_rup(NN, 16);
    h->OH = rocket_conv_transpose2d_oh(d);
    h->OW = rocket_conv_transpose2d_ow(d);
    h->QH = (h->OH + d->stride_y - 1) / d->stride_y;
    h->QW = (h->OW + d->stride_x - 1) / d->stride_x;

    /* B[n][ic] = W[ic][n]: the stored weight is B^T. Pack-time only. */
    _Float16 *B = calloc((size_t)h->N * h->K, sizeof *B);
    h->ct = malloc((size_t)h->N * h->M * sizeof *h->ct);
    h->acc = malloc((size_t)d->stride_y * d->stride_x * h->QH * h->QW * sizeof *h->acc);
    if (bias) h->bias = malloc((size_t)d->oc * sizeof *h->bias);
    if (!B || !h->ct || !h->acc || (bias && !h->bias)) goto fail;
    for (int ic = 0; ic < IC; ic++) {
        const _Float16 *wr = W + (size_t)ic * NN;
        for (int n = 0; n < NN; n++) B[(size_t)n * h->K + ic] = wr[n];
    }
    if (bias)
        for (int oc = 0; oc < d->oc; oc++) h->bias[oc] = (float)bias[oc];
    h->split_m = ct_split_m(ctx, h->M, h->K, h->N);
    h->w = rocket_weights_pack_split(ctx, h->M, h->K, h->N, B, h->split_m);
    if (!h->w) goto fail;
    free(B);
    return h;
fail:
    free(B);
    free(h->ct); free(h->acc); free(h->bias);
    free(h);
    return NULL;
}

void rocket_conv_transpose2d_weights_free(struct rocket_ctx *ctx, rocket_conv_transpose2d_weights *h)
{
    if (!h) return;
    if (h->w) rocket_weights_free(ctx ? ctx : h->ctx, h->w);
    free(h->ct); free(h->acc); free(h->bias);
    free(h);
}

/* d[0..n) += s[0..n), fp16 into fp32. */
static inline void ct_add_f16(float *restrict d, const _Float16 *restrict s, int n)
{
    int i = 0;
#if defined(__aarch64__)
    for (; i + 8 <= n; i += 8) {
        const float16x8_t v = vld1q_f16((const __fp16 *)(s + i));
        vst1q_f32(d + i,     vaddq_f32(vld1q_f32(d + i),     vcvt_f32_f16(vget_low_f16(v))));
        vst1q_f32(d + i + 4, vaddq_f32(vld1q_f32(d + i + 4), vcvt_high_f32_f16(v)));
    }
#endif
    for (; i < n; i++) d[i] += (float)s[i];
}

/* One output row from its sx phase rows (row rx at a + rx*pstride), narrowed to fp16. */
static inline void ct_emit_row(_Float16 *restrict o, const float *restrict a, size_t pstride,
                               int sx, int OW)
{
    int pw = 0;
#if defined(__aarch64__)
    if (sx == 2)
        for (; pw + 8 <= OW; pw += 8) {
            const float32x4_t x0 = vld1q_f32(a + pw / 2), x1 = vld1q_f32(a + pstride + pw / 2);
            const float32x4x2_t z = vzipq_f32(x0, x1);
            vst1q_f16((__fp16 *)(o + pw), vcombine_f16(vcvt_f16_f32(z.val[0]),
                                                       vcvt_f16_f32(z.val[1])));
        }
    else if (sx == 1)
        for (; pw + 8 <= OW; pw += 8)
            vst1q_f16((__fp16 *)(o + pw), vcombine_f16(vcvt_f16_f32(vld1q_f32(a + pw)),
                                                       vcvt_f16_f32(vld1q_f32(a + pw + 4))));
#endif
    for (; pw < OW; pw++) o[pw] = (_Float16)a[(size_t)(pw % sx) * pstride + pw / sx];
}

/* The general scatter-add, one output channel at a time over its phase planes. */
static void ct_col2im(const rocket_conv_transpose2d_weights *h, _Float16 *out)
{
    const rocket_conv_transpose2d_desc *d = &h->d;
    const int IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw, sy = d->stride_y, sx = d->stride_x;
    const int OH = h->OH, OW = h->OW, QH = h->QH, QW = h->QW;
    const size_t M = (size_t)h->M, plane = (size_t)QH * QW;
    float *acc = h->acc;

    for (int oc = 0; oc < d->oc; oc++) {
        const float b0 = h->bias ? h->bias[oc] : 0.0f;
        for (size_t i = 0; i < (size_t)sy * sx * plane; i++) acc[i] = b0;
        for (int kh = 0; kh < KH; kh++) {
            const int a = kh * d->dil_y - d->pad_top;
            const int ry = a - ct_floordiv(a, sy) * sy, oy = ct_floordiv(a, sy);
            /* phase rows: qy = ih + oy must satisfy 0 <= qy and qy*sy + ry < OH */
            const int qhr = (OH - ry + sy - 1) / sy;
            int ih0 = -oy > 0 ? -oy : 0, ih1 = qhr - oy;
            if (ih1 > IH) ih1 = IH;
            for (int kw = 0; kw < KW; kw++) {
                const int b = kw * d->dil_x - d->pad_left;
                const int rx = b - ct_floordiv(b, sx) * sx, ox = ct_floordiv(b, sx);
                const int qwr = (OW - rx + sx - 1) / sx;
                int iw0 = -ox > 0 ? -ox : 0, iw1 = qwr - ox;
                if (iw1 > IW) iw1 = IW;
                if (ih0 >= ih1 || iw0 >= iw1) continue;
                const _Float16 *P = h->ct + ((size_t)(oc * KH + kh) * KW + kw) * M;
                float *ph = acc + (size_t)(ry * sx + rx) * plane;
                for (int ih = ih0; ih < ih1; ih++)
                    ct_add_f16(ph + (size_t)(ih + oy) * QW + (iw0 + ox),
                               P + (size_t)ih * IW + iw0, iw1 - iw0);
            }
        }
        _Float16 *o = out + (size_t)oc * OH * OW;
        for (int y = 0; y < OH; y++)
            ct_emit_row(o + (size_t)y * OW, acc + (size_t)(y % sy) * sx * plane +
                        (size_t)(y / sy) * QW, plane, sx, OW);
    }
}

/* Kernel == stride, no pad, dilation 1, no output_padding: every output pixel is one tap,
 * out[oc][ih*sy + kh][iw*sx + kw] = P[(oc,kh,kw)][ih][iw] (+ bias). */
static int ct_is_shuffle(const rocket_conv_transpose2d_desc *d)
{
    return d->kh == d->stride_y && d->kw == d->stride_x && d->pad_top == 0 &&
           d->pad_left == 0 && d->dil_y == 1 && d->dil_x == 1 && d->opad_y == 0 &&
           d->opad_x == 0;
}

static void ct_shuffle(const rocket_conv_transpose2d_weights *h, _Float16 *out)
{
    const rocket_conv_transpose2d_desc *d = &h->d;
    const int IH = d->ih, IW = d->iw, sy = d->stride_y, sx = d->stride_x, OW = h->OW;
    const size_t M = (size_t)h->M;
    for (int oc = 0; oc < d->oc; oc++) {
        const float b0 = h->bias ? h->bias[oc] : 0.0f;
        for (int kh = 0; kh < sy; kh++) {
            const _Float16 *P = h->ct + (size_t)(oc * sy + kh) * sx * M;
            for (int ih = 0; ih < IH; ih++) {
                _Float16 *o = out + ((size_t)oc * h->OH + (size_t)ih * sy + kh) * OW;
                const _Float16 *p0 = P + (size_t)ih * IW;
                int iw = 0;
#if defined(__aarch64__)
                if (sx == 2 && !h->bias)
                    for (; iw + 8 <= IW; iw += 8) {
                        const float16x8_t a = vld1q_f16((const __fp16 *)(p0 + iw));
                        const float16x8_t c = vld1q_f16((const __fp16 *)(p0 + M + iw));
                        vst1q_f16((__fp16 *)(o + 2 * iw),     vzip1q_f16(a, c));
                        vst1q_f16((__fp16 *)(o + 2 * iw + 8), vzip2q_f16(a, c));
                    }
                else if (sx == 2) {
                    /* the bias is added in fp32 and the sum rounded once, as the tail does */
                    const float32x4_t bb = vdupq_n_f32(b0);
                    for (; iw + 8 <= IW; iw += 8) {
                        const float16x8_t a = vld1q_f16((const __fp16 *)(p0 + iw));
                        const float16x8_t c = vld1q_f16((const __fp16 *)(p0 + M + iw));
                        const float16x8_t z[2] = { vzip1q_f16(a, c), vzip2q_f16(a, c) };
                        for (int q = 0; q < 2; q++)
                            vst1q_f16((__fp16 *)(o + 2 * iw + 8 * q), vcombine_f16(
                                vcvt_f16_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(z[q])), bb)),
                                vcvt_f16_f32(vaddq_f32(vcvt_high_f32_f16(z[q]), bb))));
                    }
                }
#endif
                for (; iw < IW; iw++)
                    for (int kw = 0; kw < sx; kw++) {
                        const _Float16 v = p0[(size_t)kw * M + iw];
                        o[iw * sx + kw] = h->bias ? (_Float16)((float)v + b0) : v;
                    }
            }
        }
    }
}

int rocket_conv_transpose2d_fp16_prepacked(struct rocket_ctx *ctx, rocket_conv_transpose2d_weights *h,
                                           const _Float16 *in, _Float16 *out)
{
    if (!ctx || !h || !in || !out) return ROCKET_E_SHAPE;
    if (ctx != h->ctx) {
        ROCKET_LOGE("rocket_conv_transpose2d_fp16_prepacked: the weight was packed on another "
                    "context\n");
        return ROCKET_E_SHAPE;
    }
    const rocket_conv_transpose2d_desc *d = &h->d;
    const int pix = d->ih * d->iw, prof = ct_prof_on();
    const double t0 = prof ? ct_now_ms() : 0.0;
    int r = rocket_matmul_fp16_prepacked_planes(ctx, h->M, h->K, h->N, in, (size_t)pix, pix,
                                                d->ic, h->ct, h->w);
    if (r) return r;
    const double t1 = prof ? ct_now_ms() : 0.0;
    if (ct_is_shuffle(d)) ct_shuffle(h, out);
    else                  ct_col2im(h, out);
    if (prof) ct_prof_add(t1 - t0, ct_now_ms() - t1);
    return 0;
}
