// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_POOL_H
#define ROCKET_POOL_H

#include <stddef.h>
#include <stdint.h>
#include "npu_pool.h"   /* POOL_METHOD_*, gen_pool_fp16, ppu_recip_kernel_fp16 */

/*
 * Declares MaxPool and AveragePool on the NPU's PPU pooling engine. The fp16 entry is built
 * on `gen_pool_fp16` (`npu_regcmd.c`) and the submit seam. The header also declares the
 * int8 and uint8 entries, and the RK3576's own int8 entry with its resident form.
 *
 * The PPU is the NPU's pooling processor (the counterpart of NVDLA's PDP). The host
 * scatters the input into the NC1HWC2 cube layout (C2=8 for fp16, the same cube the
 * convolution path uses). The PPU reduces each kernel window per channel and writes the
 * output cube, and the host de-tiles it. A pool is a self-contained PPU and PPU_RDMA job
 * with no weights, rather than a convolution or a matmul.
 *
 * The tensors are row-major, batch 1 and channels-major, as for `rocket_conv2d_fp16`:
 *
 *     input   in [C][IH][IW]
 *     output  out[C][OH][OW]
 *     out[c][oh][ow] = reduce_{kh,kw} in[c][oh*stride_y + kh - pad_top,
 *                                            ow*stride_x + kw - pad_left]
 *
 * `reduce` is max for MaxPool and mean for AveragePool. Out-of-range reads do not
 * contribute to a max, because the pad fill is -inf. For an average, the PPU divides by
 * kh*kw (below).
 *
 * Caution: the PPU has no divider. It multiplies the window sum by a fixed per-axis
 * reciprocal fp16(65536/k), so it divides by kh*kw (count-include-pad = TRUE). The
 * exception is `avg_exclude_pad`, which only the RK3576 int8 entry accepts. With pad=0, the
 * common detection and global-average case, this is the exact average. With padding it
 * diverges at the border windows from TFLite's AVERAGE_POOL_2D, which divides by the valid
 * count. On the RK3588, the `tflite-rocket` delegate keeps a padded average on the host.
 */

/* Returns the output size along one axis, rounded down, with VALID or explicit-pad
 * semantics. */

#ifdef __cplusplus
extern "C" {
#endif
static inline int rocket_pool_out_dim(int in, int k, int stride, int pad_lo, int pad_hi)
{
    return (in + pad_lo + pad_hi - k) / stride + 1;
}

typedef struct {
    int c, ih, iw;       /* input channels, height, width */
    int kh, kw;          /* pooling kernel height, width  */
    int stride_y, stride_x;
    int pad_top, pad_left, pad_bottom, pad_right;
    int method;          /* POOL_METHOD_MAX or POOL_METHOD_AVG */
    /* Selects the divisor an average uses, which is a property of the operator rather than
     * of the SoC. TFLite's AVERAGE_POOL_2D divides a border window by the number of taps
     * that fell inside the plane. ONNX's AveragePool takes the choice as an attribute, and
     * the PPU has a mode bit for each. A value of 0 divides by kh*kw whatever the padding
     * excluded (count-include-pad = TRUE). A value of 1 drops the pad taps from the
     * divisor as well as from the sum.
     *
     * The field is last, so a positional initializer that omits it leaves it 0. At pad 0
     * the two settings compute the same function. Only the RK3576 int8 entry accepts 1,
     * and every other pool entry refuses a 1 rather than ignoring it. */
    int avg_exclude_pad;
} rocket_pool_desc;

static inline int rocket_pool_oh(const rocket_pool_desc *d)
{ return rocket_pool_out_dim(d->ih, d->kh, d->stride_y, d->pad_top, d->pad_bottom); }
static inline int rocket_pool_ow(const rocket_pool_desc *d)
{ return rocket_pool_out_dim(d->iw, d->kw, d->stride_x, d->pad_left, d->pad_right); }

/* Checks a descriptor against the supported set: the field widths and the fit in a single
 * job. Returns 0 if the pool can run, and a negative reason code otherwise. It is pure and
 * touches no hardware. */
int rocket_pool_fp16_plan(const rocket_pool_desc *d);

/* Runs the pool on the NPU. `fd` is an open `rocket` device from `rocket_open()`, and `in`
 * and `out` are row-major fp16 in the layouts above. With `fd` < 0 it computes the CPU
 * reference, as a host fallback or an off-device self-check. Returns 0 on success and a
 * negative value on error. */
int rocket_pool_fp16(int fd, const rocket_pool_desc *d,
                     const _Float16 *in, _Float16 *out);

/* Computes the CPU reference with fp32 accumulation. It is the reference the tests compare
 * against, and the `fd` < 0 fallback. An average divides by kh*kw (count-include-pad =
 * TRUE) to match the PPU's reciprocal, and a max uses -inf for out-of-range reads. It runs
 * on the host only and touches no hardware. */
void rocket_pool_ref_fp16(const rocket_pool_desc *d,
                          const _Float16 *in, _Float16 *out);

/* Pool int8 and uint8 tensors on the PPU, in the C2=16 cube. These are the int8
 * counterparts of `rocket_pool_fp16`, and the PPU reduces in the signed int8 domain:
 *
 * - `rocket_pool_int8`: signed int8 in and out (`in` [C][IH][IW], `out` [C][OH][OW]).
 * - `rocket_pool_uint8`: uint8 in and out. The host recenters each byte around the
 *   signed PPU compare with `byte ^ 0x80`, an order-preserving uint8-to-int8 map.
 *
 * A max is a bit-exact integer max, the case detection needs. An average is supported,
 * with the same fp16(65536/k) reciprocal, but the PPU truncates toward zero in the int8
 * domain. The result is therefore not rounded to nearest. With `POOL_METHOD_AVG`, the
 * entries return the hardware-truncated average, and the CPU reference below matches that
 * truncation.
 *
 * With `fd` < 0 they compute the CPU reference. Returns 0 on success and a negative value
 * on error. The fp16 plan's rule that an average needs a kernel of at least 2 (-9) still
 * applies. */
int rocket_pool_int8(int fd, const rocket_pool_desc *d, const int8_t *in, int8_t *out);
int rocket_pool_uint8(int fd, const rocket_pool_desc *d, const uint8_t *in, uint8_t *out);

/* Computes the CPU int8 reference: a signed integer max, or the PPU's truncating average
 * (the window sum times the reciprocal). It matches the hardware the two entries drive,
 * and runs on the host only. */
void rocket_pool_ref_int8(const rocket_pool_desc *d, const int8_t *in, int8_t *out);

/* Pools int8 tensors on the RK3576 with its own entry. It takes the same descriptor and
 * the same row-major [C][IH][IW] -> [C][OH][OW] int8 layouts as `rocket_pool_int8`, and
 * computes a different function. It is therefore a separate entry rather than a dispatch,
 * as `rocket_conv2d_int8()` and `rocket_matmul_int8()` are on the RK3576. It differs in two
 * ways:
 *
 * - The average rounds half to even, where `rocket_pool_int8()` truncates toward zero.
 * - On the average path the pad value is the input zero point, so this entry takes one.
 *   With `avg_exclude_pad` set, the pad value is 0 instead, so the pad taps add nothing to
 *   the sum. A max pads with -128 and ignores the zero point.
 *
 * By default the divisor is the window (kh*kw) rather than the tap count, which is
 * count-include-pad = TRUE. With `avg_exclude_pad` set, it is the tap count. The PPU has no
 * divider and multiplies by a per-axis Q16 reciprocal, which is truncated. Whether the
 * result is the exactly rounded average therefore depends on the window size.
 * `rocket_pool_int8_rk3576_exact()` answers that for a descriptor without running
 * anything, from the reciprocal the emitter will program and the worst-case int8 window
 * sum.
 *
 * The plane is split by columns, and rows and channels are not split. Past 8192 rows or
 * channels the RK3576's fields wrap, and a task returns wrong output. The plan function
 * refuses them [HW sweep, H96 MAX M9, tests/rk3576_conv_width_probe].
 *
 * The entry is bit-exact against `rocket_pool_ref_int8_rk3576()` over
 * `tests/rk3576_pool_probe.c` [HW sweep, H96 MAX M9]. */
int rocket_pool_int8_rk3576_plan(const rocket_pool_desc *d);
int rocket_pool_int8_rk3576_exact(const rocket_pool_desc *d);

/* `rocket_pool_int8_rk3576_max_ow()` returns the number of output columns one pooling
 * task can produce on the RK3576.
 *
 * Caution: past this width the PPU writes a complete output of the right size that is
 * wrong from column 0 at every height.
 *
 * The allowance depends on the kernel height and the vertical stride. It halves for each
 * output row in flight over a given input row, from 128, so the function takes the whole
 * descriptor.
 *
 * A wider plane is split by columns rather than refused. The entry runs one task per
 * slice. Each slice reads a column window of the caller's tensor and writes a column
 * window of the output, so neither end costs a copy. The PPU's destination has a
 * channel-group stride (`0x607C`, honored at any value) and no line stride, and its rows
 * are always the programmed output width apart. Each slice therefore writes its own
 * output, and the slices are separate submits. A split costs `ceil(ow / max_ow) - 1` extra
 * submits, plus a de-tile that walks rows instead of whole planes.
 *
 * `rocket_pool_int8_rk3576_ow_slices()` returns the number of slices, without a device,
 * so a caller can estimate the cost of a shape before packing it. A split handle refuses
 * a cube on its output side and refuses to be a chain node. Each slice writes its own
 * output, so no single one exists for a consumer to read. Its input side takes a cube:
 * what a slice reads is a column window, which is the ordinary pitched cube
 * [HW sweep, H96 MAX M9]. */
unsigned rocket_pool_int8_rk3576_max_ow(const rocket_pool_desc *d);
unsigned rocket_pool_int8_rk3576_ow_slices(const rocket_pool_desc *d);

/* The maximum number of column slices one handle holds. The plan function refuses a shape
 * that needs more, rather than running it as dozens of submits. */
#define ROCKET_RK3576_POOL_MAX_SLICES 8
int rocket_pool_int8_rk3576(int fd, const rocket_pool_desc *d, int in_zp,
                            const int8_t *in, int8_t *out);
void rocket_pool_ref_int8_rk3576(const rocket_pool_desc *d, int in_zp,
                                 const int8_t *in, int8_t *out);

/* Pack, run and free the resident form of the RK3576 int8 pool. A pool can run
 * repeatedly on the same shape, as in a graph's forward pass or a video stream. Without a
 * handle, each call allocates three BOs, generates one register program and frees the BOs
 * again. None of that depends on the input, and a handle holds it:
 *
 *     h = rocket_pool_int8_pack_rk3576(fd, &d, in_zp);
 *     ... per inference: rocket_pool_int8_prepacked_rk3576(fd, h, in, out);
 *     rocket_pool_int8_free_rk3576(fd, h);
 *
 * `rocket_pool_int8_rk3576()` runs this same sequence, so a prepacked call is
 * bit-identical to a transient one by construction rather than by agreement.
 *
 * A handle fixes the fd, the geometry and the input zero point. A BO belongs to its file
 * and an IOVA is per fd. The handle therefore refuses a foreign fd rather than submitting
 * against addresses that mean nothing there. The input zero point is the average path's pad
 * value, which is folded into the register program. With `avg_exclude_pad` set, the
 * program pads with 0 and does not use it. A handle is not thread-safe, because
 * it holds the scratch buffer a call writes through.
 *
 * A handle can also read its input from a producer's cube. The PPU reads the same NC1HWC2
 * cube the convolution path packs, and a direct convolution's output surface stride is
 * exactly `ow*oh`. A producer's output is therefore this pool's input cube byte for byte,
 * and the call skips the scatter and its cache maintenance. The PPU's source surface
 * stride is a register, programmed from the cube as given. A plane whose element count is
 * not a multiple of four therefore joins as readily as one that is [HW sweep, H96 MAX M9].
 * A join takes three calls:
 *
 *     rocket_conv2d_int8_cube_out_rk3576(prod, 1);
 *     rocket_conv2d_int8_cube_of_rk3576(prod, &c);
 *     rocket_pool_int8_cube_in_rk3576(h, &c);      // `in` may then be NULL
 *
 * The cube is borrowed and must outlive the calls that read it, and a NULL `src` restores
 * the row-major input. `rocket_pool_int8_cube_in_rk3576()` refuses:
 *
 * - A foreign fd
 * - A mismatched plane or channel count
 * - A buffer too short for the channel groups the PPU walks
 * - A channel count that is not a multiple of 16 (below that, the handle's own cube
 *   contains padding channels that a producer does not write)
 *
 * A column-split plane takes a cube input too. A plane wider than one pooling task's
 * output-width allowance is packed as several handles, each with a column window at both
 * ends. A window of a producer's output is the ordinary pitched cube. The PPU has the DDR
 * line stride in a register separate from what the windows consume. It also honors a base
 * part way into a row [HW sweep, H96 MAX M9]. The split therefore costs the input side
 * nothing.
 *
 * The output of a column split stays refused: the destination has no line stride, so the
 * slices cannot write one plane between them. Such a handle is neither a cube producer nor
 * a chain node. */
typedef struct rocket_pool_int8_rk3576_handle rocket_pool_int8_rk3576_handle;
struct rocket_rk3576_cube;

rocket_pool_int8_rk3576_handle *
rocket_pool_int8_pack_rk3576(int fd, const rocket_pool_desc *d, int in_zp);

int rocket_pool_int8_prepacked_rk3576(int fd, rocket_pool_int8_rk3576_handle *h,
                                      const int8_t *in, int8_t *out);

int rocket_pool_int8_cube_in_rk3576(rocket_pool_int8_rk3576_handle *h,
                                    const struct rocket_rk3576_cube *src);

/* Leaves this handle's output in cube layout for the next consumer, the other side of the
 * same join. A pool between two convolutions is a producer like any other. Its output
 * stays where the PPU wrote it, and the next convolution reads it as its feature cube.
 *
 * The PPU's own surface stride is `round4(ow*oh)` rather than the plane, and the consumer
 * accepts it. The CNA's DDR channel-group jump is a register the emitter fills. The RK3576
 * honors any value of it at or above the plane [HW sweep, H96 MAX M9,
 * tests/rk3576_surf_stride.c]. The cube therefore records that stride, and the join works
 * at any plane:
 *
 *     rocket_pool_int8_cube_out_rk3576(h, 1);            // no de-scatter
 *     rocket_pool_int8_cube_of_rk3576(h, &c);
 *     rocket_conv2d_int8_cube_in_rk3576(cons, &c);
 *     rocket_pool_int8_prepacked_rk3576(fd, h, in, NULL);   // out may be NULL
 *
 * `rocket_pool_int8_cube_of_rk3576()` refuses a channel count that is not a multiple
 * of 16. Pooling reduces within a channel, so a partial group's channels hold whatever
 * the input cube's padding held, and the tail cannot be declared. */
int rocket_pool_int8_cube_out_rk3576(rocket_pool_int8_rk3576_handle *h, int on);

int rocket_pool_int8_cube_of_rk3576(const rocket_pool_int8_rk3576_handle *h,
                                    struct rocket_rk3576_cube *out);

/* Writes this handle's output into `dst`, a caller's buffer at `dst`'s channel-group
 * offset, instead of into the handle's own buffer. The call places a pool's output beside
 * another producer's as one concatenated operand, and implies cube output. A NULL `dst`
 * restores the handle's own buffer. The buffer is borrowed and must outlive the calls that
 * write it.
 *
 * The slice's channel-group stride is programmed, so a pool writes beside convolutions
 * whose surface stride is exactly the plane. `0x607C` contains it, and the RK3576 honors
 * any value at or above the plane [HW sweep, H96 MAX M9, tests/rk3576_pool_probe.c dst].
 * The accepted values include strides that are neither a multiple of four atoms nor of a
 * 64-byte line, and odd ones. Below the plane the channel groups overlap, and that is the
 * one stride refused.
 *
 * There is no destination line stride, so a slice can be a plane inside a deeper buffer
 * and never a plane inside wider rows. The function also refuses:
 *
 * - A foreign fd
 * - A plane that is not this handle's
 * - A slice too short for the channel groups the PPU writes
 * - An offset that is not a whole group */
int rocket_pool_int8_cube_out_at_rk3576(rocket_pool_int8_rk3576_handle *h,
                                        const struct rocket_rk3576_cube *dst);

/* Checks a pool's output for the divisor lag, for a caller that issues its own submit.
 *
 * Caution: a padded average pool on the RK3576 intermittently divides an output window by
 * the tap count of the previous window in raster order. The sum is right and the output
 * is fully written, so a write check cannot see it. The per-call entry above checks the
 * positions where the lag can show, and redoes its own submit.
 *
 * A cross-layer kick issues the submit instead, so it runs the same check from its verify
 * bracket:
 *
 *     if (!rocket_pool_int8_rk3576_lag_check(fd, h, surf)) redo the kick;   // no power cycle
 *
 * `surf` is this handle's output, already synced for the CPU by the caller's own bracket.
 * The input side is a producer's cube, and the function brackets it itself. Returns 1
 * when the output is correct or the hazard cannot show on this handle, so a caller can
 * call it unconditionally. `_lag_observable()` says in advance whether it will do
 * anything. `_lag_src_handle()` names the BO it reads, and a caller holding that output
 * must not overwrite it, with a sentinel for example, before the check runs.
 *
 * `ROCKET_RK3576_POOL_LAGCHECK=0` turns all of it off. */
int rocket_pool_int8_rk3576_lag_check(int fd, rocket_pool_int8_rk3576_handle *h,
                                      const void *surf);
int rocket_pool_int8_rk3576_lag_observable(const rocket_pool_int8_rk3576_handle *h);
/* Returns whether this handle's geometry has any position where the lag could show,
 * whatever the check is set to. A placement decision must not change when the check is
 * turned off, so a caller that decides placement reads this rather than
 * `_lag_observable()`. */
int rocket_pool_int8_rk3576_lag_can_show(const rocket_pool_int8_rk3576_handle *h);
uint32_t rocket_pool_int8_rk3576_lag_src_handle(const rocket_pool_int8_rk3576_handle *h);
/* Returns how many bytes the lag check's bracket walks, for a caller estimating the
 * check's cost. Returns 0 when there is no check on this handle. */
size_t rocket_pool_int8_rk3576_lag_src_bytes(const rocket_pool_int8_rk3576_handle *h);
/* Returns how often this handle's divisor has lagged, and over how many checks. Every
 * cost of the hazard depends on this rate, and the rate is a property of the layer rather
 * than of the geometry. A caller with a graph therefore prints it per layer. A parent
 * handle sums its column slices. */
void rocket_pool_int8_rk3576_lag_counts(const rocket_pool_int8_rk3576_handle *h,
                                        unsigned *fires, unsigned *calls);
unsigned long rocket_pool_int8_rk3576_lag_discr(const rocket_pool_int8_rk3576_handle *h);
/* Returns how many redos the lag check gives a call. The default is 32, and
 * `ROCKET_RK3576_POOL_LAG_ATTEMPTS` sets it. The budget is separate from the wide-output
 * hazard's, which is small because each of those redos costs a power cycle. A lag redo
 * costs a submit, and the lag is common where it can show: roughly a third of Inception
 * V3's padded average pool calls. */
unsigned rocket_pool_int8_rk3576_lag_attempts(void);

void rocket_pool_int8_free_rk3576(int fd, rocket_pool_int8_rk3576_handle *h);

#ifdef __cplusplus
}
#endif
#endif /* ROCKET_POOL_H */
