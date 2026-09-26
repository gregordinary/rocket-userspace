// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_conv_internal.h — conv entry points that are not part of the public API.
 */
#ifndef ROCKET_CONV_INTERNAL_H
#define ROCKET_CONV_INTERNAL_H

#include "rocket_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One direct fp16 conv job, one task, with the output extent OH x OW programmed as given
 * instead of derived from IH/IW, the stride and the pad. Stride and dilation are 1.
 * Layouts are rocket_conv2d_fp16()'s: in [IC][IH][IW], W [OC][IC][KH][KW],
 * out [OC][OH][OW]. No tiling: the input and one weight kernel must each fit the CBUF,
 * or the generator refuses. Returns 0, or <0 on a refusal or a device error.
 *
 * It exists for the CNA deconvolution mode (ROCKET_CNA_DECONV with the _X/_Y stride
 * fields), whose output is larger than any forward extent of its input. */
int rocket_conv2d_fp16_job_extent(int fd, int IC, int IH, int IW, int OC, int OH, int OW,
                                  int KH, int KW, int pt, int pl,
                                  const _Float16 *in, const _Float16 *W, _Float16 *out);

/* rocket_conv_transpose2d_fp16's hardware route, the CNA deconvolution mode.
 *
 * _fits() says whether one job per output-channel tile holds the shape: IC in whole
 * 32-channel groups, the COMPACT input within one CBUF pass beside a 16-channel weight
 * tile, one kernel within one bank, and an output extent the geometry fields carry
 * (OW and OH <= 2047, OH*OW <= 262143). Pure.
 *
 * _deconv() runs it. Wf is the FORWARD kernel [OC][IC][KH][KW], i.e. the transposed
 * conv's weight already flipped 180 degrees and channel-transposed; in is the compact
 * [IC][IH][IW]; out is [OC][OH][OW] at the transposed extent. sy/sx are the transposed
 * conv's strides (2, 4 or 8, or 1 on an axis that is not upsampled), and pt/pl are
 * k-1-p. ctx may be NULL. Returns 0, -4 when the shape does not fit (callers check
 * _fits() first), or another negative on an error. RK3588 only. */
int rocket_conv2d_fp16_deconv_fits(int IC, int IH, int IW, int OC, int OH, int OW,
                                   int KH, int KW);
int rocket_conv2d_fp16_deconv(int fd, rocket_conv_ctx *ctx, int IC, int IH, int IW, int OC,
                              int OH, int OW, int KH, int KW, int sy, int sx, int pt, int pl,
                              const _Float16 *in, const _Float16 *Wf, _Float16 *out);

#ifdef __cplusplus
}
#endif
#endif /* ROCKET_CONV_INTERNAL_H */
