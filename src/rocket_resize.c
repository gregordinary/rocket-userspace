// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_resize.c — integer-factor spatial upsample (nearest / bilinear) on the rocket
 * NPU, realised as a DEPTHWISE ConvTranspose2d with a fixed per-channel kernel.
 *
 * An upsample IS a transposed conv: scatter each input pixel onto a stride-`scale`
 * lattice and convolve with a small kernel.
 *   nearest  — kernel = ones(scale): each pixel replicated into a scale x scale block.
 *   bilinear — kernel = separable triangle of size k = 2*scale - scale%2,
 *              tri[i] = 1 - |i - (k-1)/2| / scale. The stride-`scale` subsample of this
 *              triangle is a partition of unity (each output phase's taps sum to 1), so
 *              the interior is exact 2-tap bilinear with the half-pixel map
 *              src = (o+0.5)/scale - 0.5 and a zero boundary.
 *
 * Both use pad = (k - scale)/2, opad = 0 -> output = IH*scale x IW*scale (the k cancels:
 * OH = (IH-1)*scale - 2*pad + (k-1) + 1 = IH*scale). The depthwise ConvTranspose lowers
 * onto the HW-validated forward conv, so these inherit it bit-for-bit.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_resize.h"
#include "rocket_conv.h"

/* bilinear 1D triangle kernel size for an integer factor (FCN bilinear-deconv form). */
static int bilinear_ksize(int scale) { return 2 * scale - (scale % 2); }

/* Fill the per-channel depthwise kernel W[C][1][KH][KW]. mode 0 = nearest (ones),
 * 1 = bilinear (separable triangle). All channels share the same kernel. */
static int build_kernel(int mode, int C, int KH, int KW, int sy, int sx, _Float16 *W)
{
    double *ty = malloc((size_t)KH * sizeof(double));
    double *tx = malloc((size_t)KW * sizeof(double));
    if (!ty || !tx) { free(ty); free(tx); return -1; }
    if (mode == 0) {
        for (int i = 0; i < KH; i++) ty[i] = 1.0;
        for (int j = 0; j < KW; j++) tx[j] = 1.0;
    } else {
        const double cy = (KH - 1) / 2.0, cx = (KW - 1) / 2.0;
        for (int i = 0; i < KH; i++) ty[i] = 1.0 - fabs(i - cy) / (double)sy;
        for (int j = 0; j < KW; j++) tx[j] = 1.0 - fabs(j - cx) / (double)sx;
    }
    for (int kh = 0; kh < KH; kh++)
        for (int kw = 0; kw < KW; kw++) {
            _Float16 v = (_Float16)(ty[kh] * tx[kw]);
            for (int c = 0; c < C; c++)
                W[((size_t)c * KH + kh) * KW + kw] = v;
        }
    free(ty); free(tx);
    return 0;
}

/* Build the depthwise-transpose descriptor for an upsample. mode 0 nearest / 1 bilinear. */
static rocket_conv_transpose2d_desc upsample_desc(int mode, int C, int IH, int IW,
                                                  int sy, int sx, int *KH, int *KW)
{
    *KH = (mode == 0) ? sy : bilinear_ksize(sy);
    *KW = (mode == 0) ? sx : bilinear_ksize(sx);
    rocket_conv_transpose2d_desc d = {
        .ic = C, .ih = IH, .iw = IW, .oc = C, .kh = *KH, .kw = *KW,
        .stride_y = sy, .stride_x = sx,
        .pad_top = (*KH - sy) / 2, .pad_left = (*KW - sx) / 2,
        .opad_y = 0, .opad_x = 0, .dil_y = 1, .dil_x = 1, .depthwise = 1 };
    return d;
}

static int upsample_plan(int mode, int C, int IH, int IW, int sy, int sx)
{
    if (C <= 0 || IH <= 0 || IW <= 0 || sy <= 0 || sx <= 0) return -1;
    int KH, KW;
    rocket_conv_transpose2d_desc d = upsample_desc(mode, C, IH, IW, sy, sx, &KH, &KW);
    return rocket_conv_transpose2d_plan(&d);
}

int rocket_upsample_nearest_plan(int C, int IH, int IW, int sy, int sx)
{ return upsample_plan(0, C, IH, IW, sy, sx); }
int rocket_upsample_bilinear_plan(int C, int IH, int IW, int sy, int sx)
{ return upsample_plan(1, C, IH, IW, sy, sx); }

/* shared NPU path: build kernel + run the depthwise transpose */
static int upsample_run(int fd, int mode, const _Float16 *in, _Float16 *out,
                        int C, int IH, int IW, int sy, int sx)
{
    int KH, KW;
    rocket_conv_transpose2d_desc d = upsample_desc(mode, C, IH, IW, sy, sx, &KH, &KW);
    int pr = rocket_conv_transpose2d_plan(&d);
    if (pr) return pr;
    _Float16 *W = malloc((size_t)C * KH * KW * sizeof(_Float16));
    if (!W) return -5;
    if (build_kernel(mode, C, KH, KW, sy, sx, W) != 0) { free(W); return -5; }
    int rc = rocket_conv_transpose2d_fp16(fd, &d, in, W, out);
    free(W);
    return rc;
}

int rocket_upsample_nearest_fp16(int fd, const _Float16 *in, _Float16 *out,
                                 int C, int IH, int IW, int sy, int sx)
{
    if (!in || !out) return -1;
    if (fd < 0) { rocket_upsample_nearest_ref_fp16(in, out, C, IH, IW, sy, sx); return 0; }
    return upsample_run(fd, 0, in, out, C, IH, IW, sy, sx);
}

int rocket_upsample_bilinear_fp16(int fd, const _Float16 *in, _Float16 *out,
                                  int C, int IH, int IW, int sy, int sx)
{
    if (!in || !out) return -1;
    if (fd < 0) { rocket_upsample_bilinear_ref_fp16(in, out, C, IH, IW, sy, sx); return 0; }
    return upsample_run(fd, 1, in, out, C, IH, IW, sy, sx);
}

void rocket_upsample_nearest_ref_fp16(const _Float16 *in, _Float16 *out,
                                      int C, int IH, int IW, int sy, int sx)
{
    const int OH = IH * sy, OW = IW * sx;
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                out[((size_t)c * OH + oh) * OW + ow] =
                    in[((size_t)c * IH + oh / sy) * IW + ow / sx];
}

void rocket_upsample_bilinear_ref_fp16(const _Float16 *in, _Float16 *out,
                                       int C, int IH, int IW, int sy, int sx)
{
    const int OH = IH * sy, OW = IW * sx;
    for (int c = 0; c < C; c++) {
        const _Float16 *ic = in + (size_t)c * IH * IW;
        for (int oh = 0; oh < OH; oh++) {
            double scy = (oh + 0.5) / sy - 0.5;        /* half-pixel coordinate map */
            int y0 = (int)floor(scy); double dy = scy - y0;
            for (int ow = 0; ow < OW; ow++) {
                double scx = (ow + 0.5) / sx - 0.5;
                int x0 = (int)floor(scx); double dx = scx - x0;
                /* zero boundary: out-of-range taps contribute 0 */
                double p00 = (y0   >= 0 && y0   < IH && x0   >= 0 && x0   < IW) ? (double)ic[(size_t)y0*IW + x0]     : 0.0;
                double p01 = (y0   >= 0 && y0   < IH && x0+1 >= 0 && x0+1 < IW) ? (double)ic[(size_t)y0*IW + x0+1]   : 0.0;
                double p10 = (y0+1 >= 0 && y0+1 < IH && x0   >= 0 && x0   < IW) ? (double)ic[(size_t)(y0+1)*IW + x0] : 0.0;
                double p11 = (y0+1 >= 0 && y0+1 < IH && x0+1 >= 0 && x0+1 < IW) ? (double)ic[(size_t)(y0+1)*IW + x0+1] : 0.0;
                double v = (1-dy)*((1-dx)*p00 + dx*p01) + dy*((1-dx)*p10 + dx*p11);
                out[((size_t)c * OH + oh) * OW + ow] = (_Float16)v;
            }
        }
    }
}
