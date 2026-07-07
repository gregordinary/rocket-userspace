// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_POOL_H
#define ROCKET_POOL_H

#include <stdint.h>
#include "npu_pool.h"   /* POOL_METHOD_*, gen_pool_fp16, ppu_recip_kernel_fp16 */

/*
 * rocket_pool — on-NPU fp16 MaxPool / AveragePool on the rocket NPU, built on the
 * PPU pooling engine (gen_pool_fp16, npu_regcmd.c) + the rocket shim.
 *
 * The PPU is the NPU's pooling processor (NVDLA PDP analog). The host scatters the
 * input feature into the NC1HWC2 cube (C2=8 fp16, the SAME cube the conv path uses),
 * the PPU reduces each kernel window per channel and writes the output cube, the host
 * de-scatters it. NOT a conv/matmul — a self-contained PPU + PPU_RDMA job (no weights).
 *
 * Tensor layouts (row-major, batch 1, channels-major — same as rocket_conv2d_fp16):
 *   input   in [C][IH][IW]
 *   output  out[C][OH][OW]
 *   out[c][oh][ow] = reduce_{kh,kw} in[c][oh*stride_y + kh - pad_top,
 *                                          ow*stride_x + kw - pad_left]
 *   reduce = max (MaxPool) or mean (AveragePool). Out-of-range reads do not contribute
 *   to max (pad fill -inf); for average the PPU divides by kh*kw (see the caveat below).
 *
 * AVERAGE-POOL DIVISOR CAVEAT: the PPU has no divider — it multiplies the window sum by
 * a fixed per-axis reciprocal fp16(65536/k), so it always divides by kh*kw
 * (count-include-pad = TRUE semantics). With pad=0 (the common detection / global-avg
 * case) this is the exact average. With padding it diverges from TFLite's
 * AVERAGE_POOL_2D (which divides by the valid count) at the border windows — the
 * delegate keeps padded average on the host until/unless that is acceptable.
 */

/* Output spatial dim for one axis (floor; VALID / explicit-pad semantics). */

#ifdef __cplusplus
extern "C" {
#endif
static inline int rocket_pool_out_dim(int in, int k, int stride, int pad_lo, int pad_hi)
{
    return (in + pad_lo + pad_hi - k) / stride + 1;
}

typedef struct {
    int c, ih, iw;       /* input channels / height / width */
    int kh, kw;          /* pooling kernel height / width   */
    int stride_y, stride_x;
    int pad_top, pad_left, pad_bottom, pad_right;
    int method;          /* POOL_METHOD_MAX / POOL_METHOD_AVG */
} rocket_pool_desc;

static inline int rocket_pool_oh(const rocket_pool_desc *d)
{ return rocket_pool_out_dim(d->ih, d->kh, d->stride_y, d->pad_top, d->pad_bottom); }
static inline int rocket_pool_ow(const rocket_pool_desc *d)
{ return rocket_pool_out_dim(d->iw, d->kw, d->stride_x, d->pad_left, d->pad_right); }

/* Validate against the supported set (field widths, single-job fit). 0 if runnable,
 * <0 (negated reason) otherwise. Pure, no hardware. */
int rocket_pool_fp16_plan(const rocket_pool_desc *d);

/* Run the pool on the NPU. `fd` is an open rocket device (rocket_open()). in / out are
 * row-major fp16 in the layouts above. fd<0 computes the CPU reference (host fallback /
 * off-device self-check). Returns 0, negative on error. */
int rocket_pool_fp16(int fd, const rocket_pool_desc *d,
                     const _Float16 *in, _Float16 *out);

/* CPU fp32-accumulate reference (the golden oracle; also the fd<0 fallback). Average
 * divides by kh*kw (count-include-pad = TRUE) to match the PPU recip; max uses -inf for
 * out-of-range. Pure host, no hardware. */
void rocket_pool_ref_fp16(const rocket_pool_desc *d,
                          const _Float16 *in, _Float16 *out);

/* ---- int8 / uint8 pooling on the PPU (C2=16 cube) --------------------------------
 * The int8 sibling of rocket_pool_fp16. The PPU reduces in the SIGNED int8 domain.
 *
 *   rocket_pool_int8  — signed int8 in/out (in [C][IH][IW], out [C][OH][OW]).
 *   rocket_pool_uint8 — uint8 in/out; the host recenters (byte ^ 0x80, an order-
 *                       preserving uint8<->int8 map) around the signed PPU compare.
 *
 * MAX is bit-exact integer max (the detection-relevant case). AVERAGE is supported
 * (same fp16(65536/k) reciprocal) but the PPU truncates toward zero in the int8 domain,
 * so it is NOT round-to-nearest — d->method==POOL_METHOD_AVG returns the HW-truncated
 * average and the CPU reference below matches that truncation. fd<0 => CPU reference.
 * Returns 0, negative on error (the -9 avg-k<2 rule of the fp16 plan still applies). */
int rocket_pool_int8(int fd, const rocket_pool_desc *d, const int8_t *in, int8_t *out);
int rocket_pool_uint8(int fd, const rocket_pool_desc *d, const uint8_t *in, uint8_t *out);

/* CPU int8 reference: signed integer max, or truncating average (sum*recip semantics of
 * the PPU). Matches the HW the two entry points drive. Pure host, no hardware. */
void rocket_pool_ref_int8(const rocket_pool_desc *d, const int8_t *in, int8_t *out);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_POOL_H */
