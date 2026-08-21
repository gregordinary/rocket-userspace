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
    /* WHICH DIVISOR AN AVERAGE USES, and it is a property of the OP rather than of the
     * part: TFLite's AVERAGE_POOL_2D divides a border window by the number of taps that
     * fell inside the plane, ONNX's AveragePool carries the choice as an attribute, and
     * the PPU has a mode bit for each. 0 divides by kh*kw whatever the padding excluded
     * (count-include-pad = TRUE) and is what every caller before this asked for; 1 drops
     * the pad taps from the divisor as well as the sum.
     *
     * LAST FIELD DELIBERATELY. A positional initializer that predates it leaves it 0,
     * which is the old behaviour, and at pad 0 the two are the same function anyway.
     * RK3576 int8 only — every other pool entry refuses a 1 rather than ignoring it. */
    int avg_exclude_pad;
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

/* Output columns one pooling task may produce on this part. Past it the PPU writes a
 * full, correctly sized surface that is WRONG FROM COLUMN 0 at every height.
 *
 * THE ALLOWANCE IS A FUNCTION OF THE KERNEL HEIGHT AND THE VERTICAL STRIDE — it halves
 * per output ROW in flight over a given input row, from 128 — so it takes the whole
 * descriptor rather than a (kw, stride_x) pair, which is what it used to take and which a
 * square-kernel table could not distinguish.
 *
 * A WIDER PLANE IS SPLIT BY COLUMNS rather than refused: the entry runs one task per
 * slice, each reading a column window of the caller's tensor and writing a column window
 * of the output, so neither end costs a copy. The PPU's DESTINATION has a channel-group
 * stride (`0x607C`, honoured at any value) and no line stride — its rows are always the
 * programmed output width apart — so each slice writes its OWN surface and the slices are
 * separate submits. That is what a split costs: `ceil(ow / max_ow) - 1` extra submits,
 * plus a de-scatter that walks rows instead of whole planes.
 *
 * `rocket_pool_int8_rk3576_ow_slices()` is that count, pure and without a device, so a
 * caller can price the shape before packing it. A split handle refuses a cube on its
 * OUTPUT side and refuses to be a chain node — each slice owns a surface, so there is no
 * single one for a consumer to read — but its INPUT side takes one: what a slice reads is
 * a column window, which is the ordinary pitched cube. [HW sweep, H96 MAX M9] */
unsigned rocket_pool_int8_rk3576_max_ow(const rocket_pool_desc *d);
unsigned rocket_pool_int8_rk3576_ow_slices(const rocket_pool_desc *d);

/* The most column slices one handle will carry. A shape needing more is refused rather
 * than run as dozens of submits. */
#define ROCKET_RK3576_POOL_MAX_SLICES 8
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
 * does not write).
 *
 * A COLUMN-SPLIT PLANE TAKES A CUBE IN TOO. A plane wider than one pooling task's
 * output-width allowance is packed as several handles, each owning a column window at
 * both ends — and a window of a producer's surface is the ordinary pitched cube, since the
 * PPU carries the DDR line stride separately from what the windows consume and honours a
 * base part way into a row [HW sweep, H96 MAX M9]. So the split costs the INPUT side
 * nothing. Its OUTPUT stays refused: the destination surface stride is derived from the
 * plane with no register to move it, so the slices cannot write one plane between them,
 * and such a handle is neither a cube producer nor a chain node. */
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
 * THE SLICE'S CHANNEL-GROUP STRIDE IS PROGRAMMED, so a pool writes beside convolutions
 * whose surface stride is the plane exactly. `0x607C` carries it and the part honours any
 * value at or above the plane — including strides that are neither a multiple of four atoms
 * nor of a 64-byte line, and odd ones [HW sweep, H96 MAX M9, tests/rk3576_pool_probe.c dst].
 * BELOW the plane the channel groups overlap, and that is the one stride refused. There is
 * no destination LINE stride to go with it, so a slice may be a plane inside a deeper buffer
 * and never a plane inside wider rows. Also refuses a foreign fd, a plane that is not this
 * handle's, a slice too short for the channel groups the PPU writes, and an offset that is
 * not a whole group. */
int rocket_pool_int8_cube_out_at_rk3576(rocket_pool_int8_rk3576_handle *h,
                                        const struct rocket_rk3576_cube *dst);

/* THE DIVISOR LAG, for a caller that owns the submit.
 *
 * A padded average pool on this part intermittently divides an output window by the tap
 * count of the PREVIOUS window in raster order. The sum is right and the surface is fully
 * written, so a write check cannot see it; the per-call entry above checks the positions
 * where it can show and redoes its own submit. A CROSS-LAYER KICK owns the submit instead,
 * so it runs the same check from its verify bracket:
 *
 *   if (!rocket_pool_int8_rk3576_lag_check(fd, h, surf)) redo the kick;   // no power cycle
 *
 * `surf` is this handle's output surface, already synced for the CPU by the caller's own
 * bracket. The input side is a producer's cube and is bracketed inside. Returns 1 when the
 * surface is right OR the hazard cannot show on this handle, so a caller may call it
 * unconditionally; `_lag_observable()` says in advance whether it will do anything, and
 * `_lag_src_handle()` names the BO it reads — a caller holding that surface must not
 * overwrite it (with a sentinel, say) before the check runs.
 *
 * ROCKET_RK3576_POOL_LAGCHECK=0 turns all of it off. */
int rocket_pool_int8_rk3576_lag_check(int fd, rocket_pool_int8_rk3576_handle *h,
                                      const void *surf);
int rocket_pool_int8_rk3576_lag_observable(const rocket_pool_int8_rk3576_handle *h);
/* Whether this handle's GEOMETRY has any position where the lag could show, whatever the
 * check is set to. Separate from _lag_observable() on purpose: turning the check off is an
 * RE arm, and anything that decides PLACEMENT off this must not move when it is thrown, or
 * the arm prices two things at once. */
int rocket_pool_int8_rk3576_lag_can_show(const rocket_pool_int8_rk3576_handle *h);
uint32_t rocket_pool_int8_rk3576_lag_src_handle(const rocket_pool_int8_rk3576_handle *h);
/* And how many bytes that bracket walks, for a caller pricing the check. 0 when there is
 * no check on this handle. */
size_t rocket_pool_int8_rk3576_lag_src_bytes(const rocket_pool_int8_rk3576_handle *h);
/* How often this handle's divisor has lagged, and over how many checks. The rate is what
 * every cost about the hazard is a function of, and it is a property of the LAYER rather
 * than of the geometry — so a caller with a graph prints it per layer. A parent handle
 * sums its column slices. */
void rocket_pool_int8_rk3576_lag_counts(const rocket_pool_int8_rk3576_handle *h,
                                        unsigned *fires, unsigned *calls);
unsigned long rocket_pool_int8_rk3576_lag_discr(const rocket_pool_int8_rk3576_handle *h);
/* How many redos to give it — NOT the poisoning's budget, which is small because each of
 * those costs a power cycle. A lag redo costs a submit, and the lag is COMMON where it can
 * show: roughly a third of Inception V3's padded average pool calls. Default 32,
 * ROCKET_RK3576_POOL_LAG_ATTEMPTS. */
unsigned rocket_pool_int8_rk3576_lag_attempts(void);

void rocket_pool_int8_free_rk3576(int fd, rocket_pool_int8_rk3576_handle *h);

#ifdef __cplusplus
}
#endif
#endif /* ROCKET_POOL_H */
