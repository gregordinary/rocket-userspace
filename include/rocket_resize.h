// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_RESIZE_H
#define ROCKET_RESIZE_H

#include <stdint.h>

/*
 * Declares the integer-factor spatial upsample on the NPU, the FPN and decoder neck operator
 * (TFLite `RESIZE_NEAREST_NEIGHBOR` and `RESIZE_BILINEAR`, ONNX Resize, PyTorch
 * `F.interpolate`). Both modes run as a depthwise ConvTranspose2d with one fixed kernel per
 * channel, lowered onto the forward convolution. An upsample is a transposed convolution with a
 * particular kernel. The two modes use these kernels:
 *
 * - Nearest: a box kernel of ones, of size `scale`. Each input pixel is replicated into a
 *   `scale` x `scale` output block, and the block replication is bit-exact.
 * - Bilinear: a separable triangle kernel of size `2*scale - scale%2`, the "bilinear deconv"
 *   initialization of FCN segmentation models. The interior is true 2-tap bilinear
 *   interpolation with the half-pixel coordinate map `src = (o+0.5)/scale - 0.5` and a zero
 *   boundary. The interior matches `F.interpolate(mode='bilinear', align_corners=False)`.
 *
 * A delegate that needs a framework's exact coordinate mode (`align_corners`, a clamped
 * boundary) must handle it itself.
 *
 * `in` is [C][IH][IW] and `out` is [C][IH*scale_y][IW*scale_x], both row-major fp16 at
 * batch 1. `C` must satisfy `C % G == 0`, where `G` = 32 is the depthwise channel group of
 * the forward convolution. The cost scales with the upsampled size, because the lowering
 * materializes the dilated input.
 *
 * Nearest is bit-exact, and bilinear is within fp16 interpolation tolerance, against
 * independent CPU references [HW sweep] (`tests/resize_rocket.c`).
 */

/* The output size in both modes is `OH = IH*scale_y` and `OW = IW*scale_x`. */

/*
 * Checks that an upsample can run: `C % G`, positive scales and the forward convolution's
 * CBUF fit. If the upsample can run, it returns 0, and otherwise a negative value.
 */

#ifdef __cplusplus
extern "C" {
#endif
int rocket_upsample_nearest_plan(int C, int IH, int IW, int scale_y, int scale_x);
int rocket_upsample_bilinear_plan(int C, int IH, int IW, int scale_y, int scale_x);

/* Computes the nearest-neighbor upsample by block replication. With `fd` < 0 it computes the
 * CPU reference. */
int  rocket_upsample_nearest_fp16(int fd, const _Float16 *in, _Float16 *out,
                                  int C, int IH, int IW, int scale_y, int scale_x);
/* Computes the nearest upsample on the CPU, independently of the NPU path:
 * `out[c][oh][ow] = in[c][oh/scale_y][ow/scale_x]`. */
void rocket_upsample_nearest_ref_fp16(const _Float16 *in, _Float16 *out,
                                      int C, int IH, int IW, int scale_y, int scale_x);

/* Computes the bilinear upsample with the half-pixel map and a zero boundary. With `fd` < 0 it
 * computes the CPU reference. */
int  rocket_upsample_bilinear_fp16(int fd, const _Float16 *in, _Float16 *out,
                                   int C, int IH, int IW, int scale_y, int scale_x);
/* Computes the bilinear upsample on the CPU, independently of the NPU path: 2-tap half-pixel
 * bilinear, `src=(o+0.5)/scale-0.5`, with a zero boundary. */
void rocket_upsample_bilinear_ref_fp16(const _Float16 *in, _Float16 *out,
                                       int C, int IH, int IW, int scale_y, int scale_x);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_RESIZE_H */
