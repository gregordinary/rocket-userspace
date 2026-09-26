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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_conv.h"
#include "rocket_conv_internal.h"
#include "rocket_hw_profile.h"

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
