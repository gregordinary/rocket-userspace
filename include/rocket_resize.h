// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_RESIZE_H
#define ROCKET_RESIZE_H

#include <stdint.h>

/*
 * rocket_resize — integer-factor spatial UPSAMPLE on the rocket NPU, the FPN /
 * decoder neck operator (TFLite RESIZE_NEAREST_NEIGHBOR / RESIZE_BILINEAR, ONNX Resize,
 * PyTorch F.interpolate). Both modes are realised as a DEPTHWISE ConvTranspose2d (one
 * fixed kernel per channel) over the validated forward-conv engine — so there is no new
 * NPU primitive: an upsample IS a transposed conv with a particular kernel.
 *
 *   nearest  : a BOX kernel of ones (size scale) — each input pixel is replicated into a
 *              scale x scale output block. Bit-exact block replication.
 *   bilinear : a separable TRIANGLE kernel (the FCN/segmentation "bilinear deconv" init,
 *              size 2*scale - scale%2). The stride-scale subsample of the triangle is a
 *              partition of unity, so the interior is true 2-tap bilinear interpolation
 *              with the HALF-PIXEL coordinate map src = (o+0.5)/scale - 0.5 and a ZERO
 *              boundary (matches F.interpolate(mode='bilinear', align_corners=False) in
 *              the interior). Framework-exact coordinate modes (align_corners / clamp
 *              boundary) are a delegate-wiring concern, not this primitive.
 *
 * Layouts (row-major, batch 1): in [C][IH][IW], out [C][IH*scale_y][IW*scale_x], fp16.
 * Constraint: C % G == 0 (G=32, the depthwise channel group) — inherited from the
 * depthwise forward conv. Cost scales with the upsampled size (materialised dilation;
 * the sub-pixel/stride^2 decomposition is the perf follow-on, shared with ConvTranspose).
 * HW-validated bit-exact (nearest) / within fp16 interpolation tolerance (bilinear) vs
 * independent CPU references (tests/resize_rocket.c).
 */

/* Output spatial dims (both modes): OH = IH*scale_y, OW = IW*scale_x. */

/* Validate (C%G, positive scales, forward-conv CBUF fit). 0 if runnable, <0 otherwise. */

#ifdef __cplusplus
extern "C" {
#endif
int rocket_upsample_nearest_plan(int C, int IH, int IW, int scale_y, int scale_x);
int rocket_upsample_bilinear_plan(int C, int IH, int IW, int scale_y, int scale_x);

/* Nearest-neighbour upsample (block replication). fd<0 ⇒ CPU reference. */
int  rocket_upsample_nearest_fp16(int fd, const _Float16 *in, _Float16 *out,
                                  int C, int IH, int IW, int scale_y, int scale_x);
/* Independent CPU oracle: out[c][oh][ow] = in[c][oh/scale_y][ow/scale_x]. */
void rocket_upsample_nearest_ref_fp16(const _Float16 *in, _Float16 *out,
                                      int C, int IH, int IW, int scale_y, int scale_x);

/* Bilinear upsample (half-pixel, zero boundary). fd<0 ⇒ CPU reference. */
int  rocket_upsample_bilinear_fp16(int fd, const _Float16 *in, _Float16 *out,
                                   int C, int IH, int IW, int scale_y, int scale_x);
/* Independent CPU oracle: 2-tap half-pixel bilinear, src=(o+0.5)/scale-0.5, zero boundary. */
void rocket_upsample_bilinear_ref_fp16(const _Float16 *in, _Float16 *out,
                                       int C, int IH, int IW, int scale_y, int scale_x);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_RESIZE_H */
