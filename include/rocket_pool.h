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

/* ---- RK3576: the part's own pooling entry ----------------------------------------
 * Same descriptor, same row-major [C][IH][IW] -> [C][OH][OW] int8 layouts, and a
 * different function — so it is a separate entry rather than a dispatch, the same
 * answer rocket_conv2d_int8() and rocket_matmul_int8() give on this part:
 *
 *   THE AVERAGE ROUNDS HALF TO EVEN here, where rocket_pool_int8() above truncates
 *   toward zero. Two roundings are two functions.
 *
 *   THE PAD VALUE IS THE INPUT ZERO POINT on the average path, so this entry takes one.
 *   Max pads with -128 and ignores it.
 *
 * The divisor is the WINDOW (kh*kw), not the tap count — TFLite's count-include-pad =
 * TRUE — because the PPU has no divider and multiplies by a per-axis Q16 reciprocal.
 * That reciprocal is TRUNCATED, so whether the result is the exactly-rounded average is
 * a property of the window size: rocket_pool_int8_rk3576_exact() answers it for a
 * descriptor, from the reciprocal the emitter will program and the worst-case int8
 * window sum, without running anything.
 *
 * Bit-exact against rocket_pool_ref_int8_rk3576() over tests/rk3576_pool_probe.c.
 * [HW sweep, H96 MAX M9] */
int rocket_pool_int8_rk3576_plan(const rocket_pool_desc *d);
int rocket_pool_int8_rk3576_exact(const rocket_pool_desc *d);
int rocket_pool_int8_rk3576(int fd, const rocket_pool_desc *d, int in_zp,
                            const int8_t *in, int8_t *out);
void rocket_pool_ref_int8_rk3576(const rocket_pool_desc *d, int in_zp,
                                 const int8_t *in, int8_t *out);

/* ---- the resident form ------------------------------------------------------
 * A pool run repeatedly on the same shape — a graph's forward pass, a video stream —
 * allocates three BOs, generates one register program and frees the three again on
 * every call, and none of that depends on the input. A handle holds them:
 *
 *   h = rocket_pool_int8_pack_rk3576(fd, &d, in_zp);
 *   ... per inference: rocket_pool_int8_prepacked_rk3576(fd, h, in, out);
 *   rocket_pool_int8_free_rk3576(fd, h);
 *
 * rocket_pool_int8_rk3576() above IS this sequence, so a prepacked call is bit-identical
 * to a transient one by construction rather than by agreement.
 *
 * A handle freezes the fd (a BO belongs to its file and an IOVA is per-fd, so a foreign
 * fd is refused rather than submitted against addresses that mean nothing there), the
 * geometry, and the input zero point — the average path's pad value, which is folded into
 * the register program. NOT thread-safe: one handle carries the scratch a call writes
 * through.
 *
 * CUBE INPUT. The PPU reads the same NC1HWC2 cube the convolution path packs, and a
 * direct convolution's output surface stride is `ow*oh` exactly — so a producer's surface
 * IS this pool's input cube, byte for byte, and the scatter and its cache maintenance do
 * not have to be paid at all. The PPU's source surface stride is a register and is
 * programmed from the cube verbatim, so a plane whose element count is not a multiple of
 * four joins as readily as one that is. [HW sweep, H96 MAX M9]
 *
 *   rocket_conv2d_int8_cube_out_rk3576(prod, 1);
 *   rocket_conv2d_int8_cube_of_rk3576(prod, &c);
 *   rocket_pool_int8_cube_in_rk3576(h, &c);      // `in` may then be NULL
 *
 * The cube is BORROWED and must outlive the calls that read it; `src` NULL restores the
 * row-major input. Refuses a foreign fd, a mismatched plane or channel count, a buffer
 * too short for the channel groups the PPU walks, and a channel count that is not a
 * multiple of 16 (below that the handle's own cube carries padding channels a producer
 * does not write). */
typedef struct rocket_pool_int8_rk3576_handle rocket_pool_int8_rk3576_handle;
struct rocket_rk3576_cube;

rocket_pool_int8_rk3576_handle *
rocket_pool_int8_pack_rk3576(int fd, const rocket_pool_desc *d, int in_zp);

int rocket_pool_int8_prepacked_rk3576(int fd, rocket_pool_int8_rk3576_handle *h,
                                      const int8_t *in, int8_t *out);

int rocket_pool_int8_cube_in_rk3576(rocket_pool_int8_rk3576_handle *h,
                                    const struct rocket_rk3576_cube *src);

/* CUBE OUTPUT, the other side of the same join. A pool between two convolutions is a
 * producer like any other: leave its output surface where the PPU wrote it and the next
 * convolution reads it as its feature cube. The PPU's own surface stride is `round4(ow*oh)`
 * rather than the plane, which is not a bound on the consumer — the CNA's DDR channel-group
 * jump is a register the emitter fills and the part honours it at any value at or above the
 * plane [HW sweep, H96 MAX M9, tests/rk3576_surf_stride.c] — so the cube carries that stride
 * and the join is available at any plane.
 *
 *   rocket_pool_int8_cube_out_rk3576(h, 1);            // no de-scatter
 *   rocket_pool_int8_cube_of_rk3576(h, &c);
 *   rocket_conv2d_int8_cube_in_rk3576(cons, &c);
 *   rocket_pool_int8_prepacked_rk3576(fd, h, in, NULL);   // out may be NULL
 *
 * `_cube_of_` refuses a channel count that is not a multiple of 16: pooling reduces WITHIN
 * a channel, so a partial group's channels hold whatever the input cube's padding held and
 * the tail cannot be declared. */
int rocket_pool_int8_cube_out_rk3576(rocket_pool_int8_rk3576_handle *h, int on);

int rocket_pool_int8_cube_of_rk3576(const rocket_pool_int8_rk3576_handle *h,
                                    struct rocket_rk3576_cube *out);

/* Write this handle's output surface into `dst` — a caller's buffer, at `dst`'s channel-group
 * offset — instead of into its own, which is what puts a pool's output beside another
 * producer's as one concatenated operand. Implies cube-out. `dst` NULL restores the handle's
 * own surface. The buffer is BORROWED and must outlive the calls that write it.
 *
 * REFUSES A SLICE WHOSE CHANNEL-GROUP STRIDE IS NOT round4(ow*oh). The pooling program's
 * destination stride is derived from the output plane and is not a field a caller can move,
 * so a buffer at any other stride would be written at the wrong offsets; a plane whose
 * element count is already a multiple of four needs nothing. Also refuses a foreign fd, a
 * plane that is not this handle's, a slice too short for the channel groups the PPU writes,
 * and an offset that is not a whole group. */
int rocket_pool_int8_cube_out_at_rk3576(rocket_pool_int8_rk3576_handle *h,
                                        const struct rocket_rk3576_cube *dst);

void rocket_pool_int8_free_rk3576(int fd, rocket_pool_int8_rk3576_handle *h);

#ifdef __cplusplus
}
#endif
#endif /* ROCKET_POOL_H */
