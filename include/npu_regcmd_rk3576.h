// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * Declares the RK3576 CONV_2D regcmd generator and its variants. It also declares the
 * pooling, LUT, elementwise and matmul-form programs, and the buffer layouts each program
 * reads.
 *
 * The RK3576 NPU belongs to the RK3588's IP family and runs through the same mainline
 * `rocket` uAPI. Its CNA, CORE and DPU blocks use a different geometry-register encoding
 * at the same block bases. Registers move, the bit packing differs at shared offsets, and
 * the RK3576 programs offsets that the RK3588 leaves at reset. An address-translation
 * table cannot express these differences, so the geometry-register emission is per-chip.
 *
 * The encoder shares these with `npu_regcmd.c`:
 *
 * - The descriptor structs (`npu_cna_desc`, `npu_core_desc` and `npu_dpu_desc`)
 * - The `NPUOP` word format
 * - The block targets
 * - The `S_POINTER` value
 * - The `PC_OPERATION_ENABLE` trailer that starts execution
 *
 * The int8 and uint8 direct convolution runs as one task or as a row-window sequence. It
 * runs on both of the CNA's feature datapaths, which are separate sub-encodings that the
 * channel count selects:
 *
 * - The normal datapath (`in_ch` > 4) reads an NC1HWC2 cube.
 * - The first-conv ARGB datapath (`in_ch` <= 4) reads a packed image.
 *
 * The fp16 direct convolution is here too. Its precision fields are transcribed from a
 * vendor float capture, and its cube groups and output map were measured on the RK3576.
 * It is bit-faithful and delivers every output channel. One task contracts sixteen input
 * channels, and `rocket_rk3576_plan_ic()` splits a convolution at an arbitrary channel
 * count into a task sequence. The fp16 section below has the detail.
 *
 * `gen_conv2d_dw_int8_rk3576()` is the depthwise generator, and it is bit-exact at these
 * geometries [HW sweep, H96 MAX M9]:
 *
 * - Channel counts 8 to 256, including those that are not multiples of 32
 * - Kernels 1 through 7, both strides, SAME and VALID
 * - Planes to 112x112, including the odd ones
 * - A row window
 *
 * Its register program comes from manufactured vendor captures. Its two buffer layouts,
 * the coefficient group and the int8 weight cube, cannot come from a capture and were
 * measured on the RK3576. Each differs from the direct path's, and a register program that
 * matches a capture says nothing about either. `rocket_rk3576_pack_coeff_dw()` and
 * `rocket_rk3576_weight_dw_int8()` below produce them.
 */
#ifndef NPU_REGCMD_RK3576_H
#define NPU_REGCMD_RK3576_H

#include <stddef.h>
#include "npu_matmul.h"   /* conv_params_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Ops emitted per task: 139 register writes and the 4-word PC trailer, plus headroom for
 * the `ROCKET_RK3576_ADD` bring-up environment variable. It appends writes for registers
 * the vendor stream never touches. */
#define RK3576_CONV_TASK_OPS 175

/* Emits one int8 or uint8 CONV_2D task for the RK3576. `_dw` is the depthwise variant.
 *
 * The geometry comes from `conv_params_t`. The RK3576 stream distinguishes the task's row
 * window from the full feature plane, so two pairs of fields matter more here than on the
 * RK3588:
 *
 * - `params->ih` and `params->oh` must be the rows this task reads and writes.
 * - `params->ih_full` and `params->oh_full` must be the full plane.
 *
 * The caller can leave `ih_full` and `oh_full` at 0 only when `ih` is the whole plane. The
 * emitter then derives the DDR channel-group stride from `ih`. A caller driving
 * `rocket_rk3576_plan_rows()` must therefore set them from the plane on every task,
 * including a plan of one. The row-window section below explains why a single-task plan
 * can still have a short window, and what a wrong setting produces.
 *
 * Returns 0 on success. Returns a negative value if the geometry is outside the
 * transcribed limits: a field would overflow, or an ARGB check below fails.
 *
 * The caller must pass channel counts that satisfy the hardware's granularity
 * (`rocket_rk3576_pad_ic` and `_pad_oc` below). The emitter warns on a violation and does
 * not reject it, because the correction changes the caller's buffer sizes. The ARGB path is
 * exempt: 3 is its natural channel count, and the granularity rule does not apply there.
 */
int gen_conv2d_int8_rk3576(conv_params_t *params);
int gen_conv2d_dw_int8_rk3576(conv_params_t *params);

/* Emits the int8 direct convolution with the DPU's raw int32 accumulator as its output, in
 * place of the requantized int8 byte. With this output, a matmul on the RK3576 is exact
 * past one task's contraction. Partial sums in int32 add exactly on the host, and int8
 * partials cannot be summed without quantizing each one.
 *
 * Two fields change, and neither is geometry. DPU 0x4010's output width takes
 * `precision_int32`, and OUT_CVT is pinned to exact unity. The quant scale fields are
 * therefore ignored here.
 *
 * The output's byte layout is unchanged. The int8 writer puts 16 lanes in a 16-byte atom
 * and this one puts 4. A channel group therefore occupies the same bytes, and every stride
 * register already holds the right value. The caller changes two things: the output BO,
 * which is four times the size, and the de-tile, which reads 32-bit words:
 *
 *     word = (oc/4) * ow*oh_full * 4 + 4 * (y*ow + x) + (oc%4)
 *
 * Evidence: [HW sweep, H96 MAX M9], with operands whose every accumulator is distinct.
 * Each accumulator names exactly one word in the raw BO, and every position fits the
 * map. */
int gen_conv2d_int8_rk3576_i32out(conv_params_t *params);

/* Emits the same raw-int32 convolution with the writer's byte budget doubled. The writer
 * delivers the first eight output channels of every sixteen, where the narrow writer
 * delivers the first eight of every thirty-two. A matmul then programs two times its output
 * channels instead of four, which halves the weight-cube scatter's cost, and one submit
 * contains twice the MACs.
 *
 * One field moves, the DPU's own PROC_PRECISION: 0x4010[2:0] goes from int8 to int32. The
 * operands stay int8, the arithmetic is bit-identical, and the CNA and CORE programs are
 * byte-unchanged.
 *
 * The output layout differs from the narrow writer's. The caller must address the output
 * with `rocket_rk3576_i32_wide_word()` and size the BO for twice the narrow extent. The
 * mode adds two bounds, which that function applies: `oc` a multiple of 32, and only
 * channels with c%16 < 8 delivered.
 *
 * Evidence: [HW sweep, H96 MAX M9]. */
int gen_conv2d_int8_rk3576_i32out_wide(conv_params_t *params);

/* Returns the word index at which `gen_conv2d_int8_rk3576_i32out_wide()` writes output
 * channel `c` at pixel `p` (row-major, p = y*ow + x). Returns -1 if `c` is not a delivered
 * channel. `oc` must be a multiple of 32. The definition in the source describes what a
 * partial group does. */
int rocket_rk3576_i32_wide_word(unsigned ow, unsigned oh_full, unsigned c, unsigned p);

/* ---- The first-conv ARGB datapath (params->ic <= 4) ----
 *
 * `gen_conv2d_int8_rk3576()` takes this path automatically when the input has 4 or fewer
 * channels. It is a different CNA program, and not the same convolution at a small channel
 * count. A vision model's stem runs on the RK3576 through this path. The normal path needs
 * `ic` to be a multiple of 32, and an image has 3 channels.
 *
 * The caller changes the feature buffer, and the registers are unchanged:
 *
 * - The input is a packed image, iw*ic interleaved bytes per row, as a camera or a decoder
 *   produces it. It is not the NC1HWC2 cube every other layer reads. The row stride is
 *   iw*ic bytes, so `rocket_rk3576_plan_rows()` returns feature offsets in those units on
 *   this path. The output is a normal cube on both paths.
 * - Pixels are signed int8, two's complement. The CNA's converter runs here and is
 *   bypassed on every other layer, but its per-channel offset is inert. The MAC reads each
 *   packed byte as a signed int8 with nothing subtracted. The input zero point therefore
 *   folds into the coefficient group's A term, as on the direct path. The border pad is
 *   the stored zero point [HW sweep, H96 MAX M9].
 * - `iw` must be a multiple of 16, because the DDR row stride is counted in 16-byte atoms
 *   and the CBUF row in 64-byte granules. Both are whole only there. The emitter rejects
 *   any other `iw`.
 * - The path accepts int8 and uint8 only, and has no depthwise form. The emitter rejects
 *   both.
 *
 * The weight cube is not in any capture, because a capture contains register programs and
 * no weight BOs. It was measured on the RK3576 instead, and
 * `rocket_rk3576_weight_argb_int8()` below produces it.
 *
 * The emitter programs the channel count as 4*kw: 4 lanes per pixel column, kw columns. At
 * kw=3 the fold programs 12 against 3 in the DMA, which every ARGB capture contains. The
 * first conv is bit-exact at kw 3, 5 and 7 [HW sweep, H96 MAX M9].
 */

/*
 * ---- fp16 and the general precision entry point ----
 *
 * `gen_conv2d_fp16_rk3576()` emits one fp16 task, which contracts exactly 16 input
 * channels. It refuses any other `ic` above 4, because any other count produces wrong
 * output silently. The ic split below is how a caller reaches an arbitrary channel count.
 * An `ic` of 4 or less takes the packed-image first conv, a different CNA program. At
 * ic=16 the emitted task is bit-faithful against a CPU model at any kernel size, plane and
 * stride, and every programmed output channel reaches DDR.
 *
 * `gen_conv2d_rk3576_prec()` is the unchecked bring-up entry beside it, and applies no
 * limits.
 *
 * The precision fields are transcribed from an RKNN-Toolkit2 float build for the RK3576
 * (`do_quantization=False` on a float ONNX), which emits a genuine fp16 conv program. They
 * are constant across every geometry it emits, and they determine these registers:
 *
 * - CNA `0x100C` is `bit21 | proc<<7 | in<<4 | conv_mode`. `proc` and `in` are both fp16,
 *   and bit 21 is a float enable the integer programs never set. With any one of the three
 *   wrong, the contraction reads every feature surface twice and skips the odd ones. The
 *   `proc` field is the operand width here, not the datapath width.
 * - DPU `0x4038` has a low half of 0x0092, where the direct path has 0x0080. The high half
 *   is free but must be non-zero, or the DPU writes nothing.
 * - DPU `0x4050` has bit 31 clear. With it set and every other float field right, the
 *   accumulator sums the doubled reads and returns +inf.
 * - DPU `0x4010` is `out<<29 | in<<26 | proc`. Its `out` field selects a width: 5 for the
 *   whole 32-bit epilogue word, and 2 and 3 for its low and high halves.
 * - DPU `0x40B0` is `fp32tofp16_en<<16 | scale`, the float narrowing. With it, width 2
 *   writes fp16 and width 3 writes bfloat16. Without it, width 2 writes the fp32 word's low
 *   mantissa bits, which are zero for small values.
 *
 * CORE 0x3018 pins the operand type, and only 2 computes. CNA 0x100C's `in` field pins the
 * 2-byte width class. The two words differ in this way, so an fp16 and a bf16 program are
 * indistinguishable in the CNA word.
 *
 * The float weight cube groups both channel axes by 16 (`rocket_rk3576_weight_conv_fp16`),
 * where the int8 cube groups both by 32. The output group is observable only at k > 1,
 * because the kernel index sits between it and the (oc2, ic2) pair. At k = 1 it cancels
 * for every candidate value. The feature cube keeps C2 = 8 fp16 lanes per 16-byte atom.
 * The element size enters the feature side only through the CBUF entry count.
 *
 * The epilogue is float, and that decides the coefficient buffer. A (the bias) is read as
 * fp32 and C (the multiplier) as fp16. The integer C = 1 that means "no scaling" on the
 * int8 path is the denormal 6e-8 here.
 *
 * Caution: an integer C = 1 underflows the whole output to empty, with no fault. The
 * caller must pack the coefficients with `rocket_rk3576_pack_coeff_prec()`, which folds in
 * fp16 1.0 (0x3C00), and not with `rocket_rk3576_pack_coeff()`.
 *
 * The contraction is sixteen input channels wide. The DPU's output element stride is 16/ic
 * words, so ic = 16 is the count at which an element occupies exactly its own two bytes.
 * Below it the writer spreads each element over a wider slot, and half the channels never
 * reach DDR. Above it two channels collide in one slot, and the odd ones are lost.
 * `rocket_rk3576_plan_ic()` below is the split around the bound.
 *
 * At `precision_int8` every one of those fields is zero, so the int8 program is
 * byte-identical to the transcribed one. The capture test still checks it.
 *
 * Caution: a wrong precision field does not fault. It writes a complete output of the
 * right size with wrong values. Check any result against a CPU model.
 * `tests/rk3576_fp16_sweep.c` is the test, and its probes measure the lane, pixel and
 * channel maps on the RK3576.
 */
int gen_conv2d_fp16_rk3576(conv_params_t *params);
/* Emits the same input-channel slice program with a whole fp32 output stage of nine DPU
 * words. The stage keeps the float arithmetic's two bits set, which otherwise prevent the
 * next submit from writing. It still leaves the next submit clean. In a run of 80
 * back-to-back submits with no idle, every submit wrote [HW sweep, H96 MAX M9,
 * rocket 1.6.0].
 *
 * The output is fp32 in the plain int32 cube (`rocket_rk3576_f32_cube_accumulate`), exact
 * against a CPU model at k 1 and 3. The caller must size the output for
 * `rocket_rk3576_f32_cube_bytes()`. The entry refuses the packed-image first conv
 * (ic <= 4), which has not run under the stage. Returns 0 or -1. */
int gen_conv2d_fp16_rk3576_f32out(conv_params_t *params);
/* Emits the int8 direct program on the same whole output stage, a dense int32 writer. It
 * writes every accumulator of the task as a raw int32 with no bias, in oc*ow*oh words. It
 * leaves the next submit able to write, and raises its own DPU completion
 * [HW sweep, H96 MAX M9, rocket 1.6.0].
 *
 * `oc` must be a whole number of 32-channel groups. The task must cover the whole output
 * (oh == oh_full), because 0x40B8 at a row window is not measured.
 * `rocket_rk3576_i32_dense_word()` is the map. It returns the word index of channel `c` at
 * pixel `p` in an output of `A` pixels, or -1. */
int gen_conv2d_int8_rk3576_i32out_dense(conv_params_t *params);
int rocket_rk3576_i32_dense_word(unsigned A, unsigned c, unsigned p);
int gen_conv2d_rk3576_prec(conv_params_t *params, int dw, unsigned prec);

/* Returns the element index into the float weight cube, 0-based on every axis, or -1 if
 * the kernel position is out of range. The float cube's groups were measured on the RK3576
 * and differ from the int8 cube's. Output channels and input channels both group by 16,
 * where the int8 cube groups both by 32.
 *
 * Caution: with a cube packed at the int8 groups, the RK3576 contracts the first 16 input
 * channels twice. It spills the rest into the next output channel, and the result looks
 * like an arithmetic fault, not a packing one. */
int rocket_rk3576_weight_conv_fp16(unsigned oc_total, unsigned ic_total,
                                   unsigned kh_total, unsigned kw_total,
                                   unsigned oc, unsigned ic, unsigned kh, unsigned kw);

/* Returns the channel counts to program, given the logical ones. Both round up to a
 * multiple of 32, the group `weight_conv_int8` pads its two channel axes to. Otherwise a
 * convolution whose channel counts are not multiples of 32 produces wrong output at every
 * geometry [HW sweep, H96 MAX M9]. No vendor capture shows this, because every captured
 * geometry already satisfies it.
 *
 * The weight cube needs nothing extra. `weight_conv_int8` already pads both channel axes
 * to 32, and its index depends on either count only through ceil(n/32). A cube packed at
 * the logical count is therefore byte-identical to one packed at the padded count. The
 * caller must size the surrounding buffers to match:
 *
 * - `rocket_rk3576_pad_ic`: pad the feature cube out to the returned channel count with
 *   zeros. An `ic` of 8, 16, 17 or 48 produces wrong output, and 32 and 64 are exact. An
 *   `ic` of 17 fails despite spanning two whole C2=16 surfaces, so the unit is the
 *   32-channel group and not the C2 surface.
 * - `rocket_rk3576_pad_oc`: size the output BO for the returned count, and pass that count
 *   to `rocket_rk3576_coeff_bytes()` and `rocket_rk3576_pack_coeff()` as well. A padded
 *   channel then gets a C term, and its group is not disabled.
 *
 * Caution: a partial `oc` group fails in two different ways, and only one of them looks
 * like a channel problem. At k=1 the DPU writes only output row 0 of the trailing group
 * and leaves the rest of the output untouched. At k>1 the weights come out wrong instead.
 * WEIGHT_BYTES = ic*oc*kh*kw describes a cube tighter than the padded one, and the
 * truncation drops whole (kh,kw) planes rather than trimming `oc` inside each. An `oc`
 * that looks fine at k=1 (24, 56) is wrong at k=3.
 */
unsigned rocket_rk3576_pad_ic(unsigned ic);
unsigned rocket_rk3576_pad_oc(unsigned oc);

/* ---- The first conv's float weight cube -------------------------------------
 *
 * The packed-image datapath contracts four lanes per pixel, whatever the image contains.
 * The cube therefore has a lane axis of four in place of an input-channel axis. Its taps
 * stay on their own axes, with no 4*kw fold and no round-up to sixteen. The layout was
 * decoded from manufactured float captures at ic 3/4, k 1/3/5/7 and oc 16/32/48/64:
 *
 *     slot(oc, c, kh, kw) = (oc/16) * (KH*KW*64)
 *                         + kh * (KW*64) + kw * 64
 *                         + (oc%16) * 4 + c
 *
 * Slots are 16-bit, with four lanes per (output channel, tap), and lane c contains image
 * channel c. Output channels interleave in groups of sixteen inside one tap, and the tap
 * axis is kh-outer. The caller must size the buffer with `_bytes()`, which rounds `oc` up
 * to a whole group.
 *
 * The int8 cube below is a different layout from this float cube
 * [source-confirmed: RKNN-Toolkit2 capture, rk3576 float build]. */
int rocket_rk3576_weight_argb_fp16(unsigned oc_total, unsigned ic_total,
                                   unsigned kh_total, unsigned kw_total,
                                   unsigned oc, unsigned ic, unsigned kh, unsigned kw);
size_t rocket_rk3576_weight_argb_fp16_bytes(unsigned oc, unsigned kh, unsigned kw);
int rocket_rk3576_argb_fp16_pack_weights(void *dst, size_t dst_bytes,
                                         const void *src, unsigned oc, unsigned ic,
                                         unsigned kh, unsigned kw);

/* ---- The first conv's int8 weight cube ---------------------------------------
 *
 * The int8 cube differs from the float cube in three ways:
 *
 * - A weight is one byte.
 * - Output channels group by thirty-two.
 * - The tap row sits outside that group, and the tap column is folded into the same
 *   16-byte row as the four lanes.
 *
 * The layout was measured on the RK3576 with an impulse image and a one-byte cube
 * (`rk3576_conv_gate fcmap`):
 *
 *     byte(oc, c, kh, kw) = (oc/32) * (KH * R * 32)
 *                         + kh * (32 * R)
 *                         + (oc%32) * R
 *                         + kw * 4
 *                         + c            R = round16(4*KW)
 *
 * Lane c contains image channel c, and the lanes past `ic` are don't-care. Of each R-byte
 * row, 4*KW bytes are live. It is a bijection over every live byte at oc 32 and 64,
 * k 3/5/7 and ic 3 and 4 [HW sweep, H96 MAX M9]. The 32-channel group is observable only
 * above one group, because at oc=32 a flat oc*R fits equally.
 *
 * Caution: the path has two geometry bounds, and no capture shows either, because every
 * captured first conv is a 3x3 stride-2 SAME convolution:
 *
 * - The left pad must be non-zero. At `pad_left` = 0 the DPU writes nothing at every
 *   plane, stride, kernel and channel count, and leaves the output untouched.
 *   CNA_PAD_CON0's `pad_left` field alone decides it. Forcing 0x0100 into a zero-pad
 *   program makes that program write, forcing 0 into a working one stops it, and the
 *   `pad_top` field does neither.
 * - The output width must be iw/stride. Any other width writes a complete output that is
 *   sheared. The tap a weight is applied at drifts one output column per output row, the
 *   signature of a row-stride mismatch. It shears for a narrow `ow` and a wide one alike.
 *
 * Together the two bounds are what SAME padding means.
 *
 * The output-channel count follows the direct path's multiple-of-32 rule, and oc=16 writes
 * nothing. The float first conv's 32-channel per-program limit does not apply: one int8
 * program delivers 64 output channels [HW sweep, H96 MAX M9]. */
int rocket_rk3576_weight_argb_int8(unsigned oc_total, unsigned ic_total,
                                   unsigned kh_total, unsigned kw_total,
                                   unsigned oc, unsigned ic, unsigned kh, unsigned kw);
size_t rocket_rk3576_weight_argb_int8_bytes(unsigned oc, unsigned kh, unsigned kw);
int rocket_rk3576_argb_int8_pack_weights(void *dst, size_t dst_bytes,
                                         const void *src, unsigned oc, unsigned ic,
                                         unsigned kh, unsigned kw);

/* ---- The depthwise weight cube ----------------------------------------------
 *
 * The depthwise cube differs from the RK3588's and from the direct path's. It was decoded
 * from vendor captures with a unique weight value per (channel, tap). The captures cover
 * C = 24, 32, 48, 64 and 128, and k = 3 and 5:
 *
 * - A weight occupies a sixteen-bit slot whatever the precision, so one geometry's float
 *   and int8 cubes are the same size.
 * - Channels group by 32, each group a contiguous block.
 * - Inside a group the order is tap-major, kh outer: every channel of the group for one
 *   tap, then the next tap.
 * - A trailing partial group is dense. Its tap stride is the channels it holds, not 32.
 *
 * `rocket_rk3576_weight_dw()` returns the 16-bit slot index, and the caller writes a 16-bit
 * element there. It returns -1 on an out-of-range position.
 * `rocket_rk3576_weight_dw_bytes()` returns the buffer size. When C is not a multiple of
 * 16, the size is larger than the slots handed out. The tail is padding that the DMA still
 * fetches [source-confirmed: RKNN-Toolkit2 capture, rk3576 depthwise builds]. */
int rocket_rk3576_weight_dw(unsigned c_total, unsigned kh_total, unsigned kw_total,
                            unsigned c, unsigned kh, unsigned kw);

/*
 * Returns the byte offset of a weight in the int8 depthwise cube, or -1 if the position is
 * out of range. A weight is one byte here. The cube is tap-major inside a channel group,
 * and a trailing partial group is dense, as in the float cube. The group is sixty-four
 * channels, where the float cube's is 32. The order inside a tap block is neither the
 * float slot order nor its low byte.
 *
 * Inside a tap block the channel a byte contains is 2*(b/4) + (b%2). Channel c therefore
 * has two live bytes per tap, at 4*(c/2) + (c%2) and two bytes further on, and both
 * contribute. `rocket_rk3576_weight_dw_int8()` returns the first. The caller must leave the
 * second at zero, or the weight is added twice. A single group and a k=1 kernel both hide
 * the group size, and C=64 k=3 is what separates 64 from 32. The layout was measured on
 * the RK3576 (`rk3576_conv_gate dwmap`).
 *
 * Caution: writing an int8 weight as a 16-bit value into `rocket_rk3576_weight_dw()`'s
 * slot puts its sign extension into the byte that belongs to the next channel. The result
 * is a silent -1 weight on a neighbor, not padding.
 */
int rocket_rk3576_weight_dw_int8(unsigned c_total, unsigned kh_total, unsigned kw_total,
                                 unsigned c, unsigned kh, unsigned kw);

size_t rocket_rk3576_weight_dw_bytes(unsigned c_total, unsigned kh, unsigned kw);

/* ---- The ic split: an fp16 conv at an arbitrary input-channel count ----------
 *
 * `gen_conv2d_fp16_rk3576()` emits one task, and one task contracts exactly 16 input
 * channels, so it refuses any other `ic` above 4. `rocket_rk3576_plan_ic()` is how a caller
 * reaches any other channel count.
 *
 * The bound is the DPU's output element stride, 16/ic words. It is not the arithmetic, and
 * not the surface index, which the float precision fields close. At ic = 32 two channels
 * occupy one slot, and the odd ones are lost. The DPU writes a complete output of the
 * right size with nothing to fault on. The split makes this a bound on a task: each slice
 * takes exactly one contraction step, which is the shape the RK3576 computes
 * bit-faithfully.
 *
 * The split is bit-exact against a CPU model [HW sweep, H96 MAX M9]. The test covered
 * ic 32/64/128, oc 32/64, k 1/3/5, planes 8x8 through 32x32, and stride 1 and 2.
 *
 * The split costs ic/16 submits and a host accumulation. The sequence, and what is
 * per-slice and what is shared:
 *
 * - The feature cube is shared, and nothing is repacked. At C2 = 8 its channel groups are
 *   contiguous planes of iw*ih_full 16-byte atoms, so slice s is the same BO at
 *   `input_dma + t[s].feature_off`.
 * - The weight cube is per-slice, and it is not a sub-cube of the whole conv's. Each slice
 *   is its own convolution, so its group count follows the slice.
 *   `rocket_rk3576_fp16_pack_slice_weights()` builds it from row-major OIHW weights. The
 *   caller must size the BO with `rocket_rk3576_fp16_slice_weight_bytes()`, which sizes by
 *   the input-channel groups of 16, a partial group taking a whole one. A 16-channel slice,
 *   the only width the planner emits, fills exactly one group.
 * - The coefficient buffer is shared, packed once with
 *   `rocket_rk3576_pack_coeff_prec(..., precision_float16)` so that C contains fp16 1.0.
 *   If it contains a bias, the caller must give it to the first slice only and pass a NULL
 *   bias to the rest. Every slice adds the whole A term, so a bias on all of them is added
 *   ic/16 times.
 * - Every slice rewrites the output BO. The caller must read it back between submits and
 *   accumulate with `rocket_rk3576_fp16_accumulate()`.
 *
 * To emit slice s, copy the `conv_params_t`, set these fields, and call
 * `gen_conv2d_fp16_rk3576()`:
 *
 *     ic = t[s].ic,  input_dma += t[s].feature_off,  weights_dma = the slice cube
 *
 * Accumulation is on the host.
 *
 * `rocket_rk3576_plan_ic()` plans the ic axis only. The row window
 * (`rocket_rk3576_plan_rows`) is a separate split on a separate axis. The planner refuses a
 * plane whose 16-channel slice still overflows the CBUF. The caller must then compose the
 * two splits: plan the rows of a slice, and run the ic slices of each row task.
 *
 * Returns 0 with `*count` set. Returns a negative value in these cases:
 *
 * - `ic` or `oc` violates the granularity above.
 * - The sequence needs more than `max_tasks`.
 * - A slice does not fit one task.
 */
/* The input-channel count one fp16 task contracts. Sixteen is the only count at which the
 * RK3576's fp16 task is correct. The DPU's output element stride is 16/ic words, so at
 * ic=16 an element occupies exactly its own two bytes. Below it the writer spreads each
 * element over a wider slot, and half the channels never reach DDR. Above it two channels
 * collide in one slot, and the odd ones are lost [HW sweep, H96 MAX M9]. */
#define ROCKET_RK3576_FP16_IC_SLICE 16u

typedef struct {
    uint16_t ic0;          /* first input channel this task contracts          */
    uint16_t ic;           /* input channels this task contracts               */
    uint32_t feature_off;  /* byte offset into the feature cube for this task  */
} rocket_rk3576_ic_task;

int rocket_rk3576_plan_ic(const conv_params_t *p, rocket_rk3576_ic_task *out,
                          unsigned max_tasks, unsigned *count);

/* Returns the channel counts an fp16 conv must program, given the logical ones. Neither
 * follows the int8 rule. `rocket_rk3576_pad_ic/_pad_oc` round to 32 because the integer
 * cube groups both axes there. The float cube groups `oc` by 16, and its `ic` axis is
 * bounded by the slice. The caller must pad the feature cube with zeros to `_fp16_pad_ic`,
 * and size the output and coefficient buffers for `_fp16_pad_oc`.
 *
 * `oc` is confirmed at 16, 32 and 64. */
unsigned rocket_rk3576_fp16_pad_ic(unsigned ic);
unsigned rocket_rk3576_fp16_pad_oc(unsigned oc);

size_t rocket_rk3576_fp16_slice_weight_bytes(unsigned oc, unsigned ic,
                                             unsigned kh, unsigned kw);
int rocket_rk3576_fp16_pack_slice_weights(void *dst, size_t dst_bytes,
                                          const _Float16 *w_oihw,
                                          unsigned oc, unsigned ic_total,
                                          unsigned kh, unsigned kw,
                                          const rocket_rk3576_ic_task *t);

/* ---- Reading an fp16 output back ----
 *
 * At the 16-channel contraction the writer produces the plain native float cube. It has 8
 * channels to a 16-byte atom, one atom per pixel, and channel groups as contiguous planes.
 * Every programmed channel is present exactly once [HW sweep, H96 MAX M9].
 *
 * For channel c at pixel (y,x), the map in 16-bit words is
 * (ow*oh*8)*(c/8) + 8*(y*ow+x) + (c%8). The map is measured, not fitted: a probe whose
 * every output lane has a unique name decodes through it as a bijection.
 *
 * `rocket_rk3576_fp16_accumulate()` de-tiles one output through that map and adds it into
 * a row-major [out_channels][oh][ow] fp32 accumulator, which is what an ic split needs.
 * The caller must zero the accumulator before the first slice.
 */
unsigned rocket_rk3576_fp16_out_channels(unsigned oc);
size_t   rocket_rk3576_fp16_out_bytes(unsigned oc, unsigned oh, unsigned ow);
int rocket_rk3576_fp16_out_index(unsigned oh, unsigned ow, unsigned c,
                                 unsigned y, unsigned x);
int rocket_rk3576_fp16_accumulate(float *acc, const void *surface,
                                  size_t surface_bytes,
                                  unsigned oc, unsigned oh, unsigned ow);

/* Sizes and reads the fp32 output that `gen_conv2d_fp16_rk3576_f32out()` writes. The
 * output is the plain int32 cube. It has four fp32 lanes to a 16-byte atom and one atom
 * per pixel, and each quad of channels is a plane of ow*oh atoms. `_bytes` sizes it for
 * `oc` channels. `_accumulate` de-tiles it and adds it into a row-major [oc][oh][ow] fp32
 * accumulator, as `rocket_rk3576_fp16_accumulate()` does for the fp16 output
 * [HW sweep, H96 MAX M9, tests/rk3576_fp16_fp32out_probe]. */
size_t rocket_rk3576_f32_cube_bytes(unsigned oc, unsigned oh, unsigned ow);
int    rocket_rk3576_f32_cube_accumulate(float *acc, const void *surface,
                                         size_t surface_bytes,
                                         unsigned oc, unsigned oh, unsigned ow);

/* CBUF budget planning.
 *
 * The RK3576's feature budget is programmable. CNA_CBUF_CON0 (0x1040) contains a granule
 * allowance F in bits[16:27], and one task's feature plane must satisfy:
 *
 *     ceil(iw*ic/64) granules per row  x  the task's input rows  <=  4096 + F
 *
 * F=0 is the value the vendor's full-plane captures contain, so 4096 granules (256 KiB) is
 * the default and not the hardware's limit. The data side reaches 6144 granules (384 KiB).
 * Data and weights share one ~448 KiB pool, so raising F takes the same capacity from the
 * weight path. At F=0 a resident weight slice of 175 KiB is correct and 200 KiB is not. At
 * F=2048 the slice must stay under 75 KiB. An F large enough to leave the weight path
 * nothing hangs the CNA, which then writes no output [HW sweep, H96 MAX M9].
 *
 * The emitter plans F from the operand shape on every task, so a caller that only emits a
 * conv gets a correct allowance without asking. The two functions below are for a caller
 * that tiles, which must use `rocket_rk3576_max_task_rows()` to choose a row window.
 *
 * Caution: a window past the allowance produces wrong output. The DPU still writes a
 * complete output of the right size, and no fault catches it.
 *
 * `rocket_rk3576_cbuf_f()` returns 0 and writes the allowance to program. It returns a
 * negative value if one task cannot hold this shape. Either the plane needs more than the
 * data cap (window the rows), or the weight slice does not fit even at F=0 (split ic). It
 * needs `oc`, `kh`, `kw` and `dw` because the weight side of the trade depends on them.
 *
 * It uses only F values measured to deliver their face value: 0, 256, 512, 1024 and 2048.
 * A combination of allowance bits delivers less than their sum. An arbitrary F would
 * therefore program a budget the hardware does not honor, and corrupt the output silently.
 *
 * The two low values are path-conditional. On the direct path, 256 and 512 deliver only
 * where the resident slice is at or under 1 KiB. On the depthwise path they are not used.
 * No threshold on the depthwise footprint fits the measurements, which have 4 granules
 * dead and 32 live. A value the RK3576 does not honor delivers the F=0 budget instead
 * [HW sweep, H96 MAX M9].
 *
 * Caution: an F value the RK3576 does not honor makes the failure non-monotone in the
 * window height. The output is exact, then wrong across the band of windows that select
 * that value, then exact again above it. A probe that bisects the window cannot see it.
 * Declining the two low values costs nothing, because 1024 is strictly larger, always
 * live, and needs no extra submit.
 *
 * The pool arithmetic is characterized at one output-channel group. A conv driving several
 * groups loses the trailing ones well before the slice reaches what the pool leaves. It
 * loses one group at a time as the slice grows, with the leading groups bit-exact and a
 * complete output written. The failure therefore looks like an output-channel defect, not
 * a capacity one. The governing quantity is the resident slice 32*ic*kh*kw itself, and
 * the same slice behaves identically whichever (ic,kh,kw) produces it.
 *
 * At F=0 and oc=128, 144 KiB drives all four groups, 145-148 KiB three, 152-156 KiB two,
 * and 162 KiB one. The product ic*kh*kw <= 4608 is therefore correct at every `oc`.
 * `rocket_rk3576_cbuf_f()` refuses past a small measured table keyed on the group count,
 * and does not let the output be corrupted silently. The caller's recourse is an ic split
 * [HW sweep, H96 MAX M9].
 *
 * `rocket_rk3576_max_task_rows()` returns the tallest row window one task can take for
 * this operand shape, at the largest allowance the weight slice leaves room for. It
 * returns 0 if no window fits.
 */
/* The feature plane's granule cost is per byte, so a 2-byte element doubles it. The caller
 * must plan the allowance at the precision the conv will be emitted at. The two entry
 * points without a precision are the `_prec` forms at int8, which is what every caller of
 * the int8 conv path wants. An fp16 caller must use the `_prec` forms.
 *
 * Caution: an allowance sized for half the plane produces wrong output, with a complete
 * output written.
 *
 * The weight side of the trade is modeled at int8 in both. At the fp16 path's one
 * contraction width, a 16-channel slice, 32*ic*kh*kw equals the float cube's own
 * 16-output-channel group slice. The graded output-channel-loss table it feeds was
 * measured on int8 only. */
int rocket_rk3576_cbuf_f(unsigned iw, unsigned ic, unsigned ih, unsigned oc,
                         unsigned kh, unsigned kw, int dw, unsigned *f_out);
unsigned rocket_rk3576_max_task_rows(unsigned iw, unsigned ic, unsigned oc,
                                     unsigned kh, unsigned kw, int dw);
/* Returns how many leading output-channel groups one direct task can drive before its
 * weight fetch stalls. `slice_bytes` is one group's weight slice (32*ic*kh*kw), beside a
 * feature allowance of 4096+f granules. The result is the first g with
 * (g*slice mod 64 KiB) + slice past the weight area, which is
 * 64 KiB * floor((3072-f)/1024). Returns 0 if the slice does not fit, and UINT_MAX if no
 * group ever stalls.
 *
 * A stalled job writes its leading groups, raises no completion, and is retired at the
 * driver's backstop [HW sweep, H96 MAX M9]. The mechanism behind the stall is a
 * hypothesis. */
unsigned rocket_rk3576_weight_phase_groups(unsigned slice_bytes, unsigned f);
int rocket_rk3576_cbuf_f_prec(unsigned iw, unsigned ic, unsigned ih, unsigned oc,
                              unsigned kh, unsigned kw, int dw, unsigned prec,
                              unsigned *f_out);
unsigned rocket_rk3576_max_task_rows_prec(unsigned iw, unsigned ic, unsigned oc,
                                          unsigned kh, unsigned kw, int dw,
                                          unsigned prec);

/* ---- The row window: a conv too tall for one task, cut into a task sequence ----
 *
 * A feature plane over the CBUF allowance produces wrong output, with the DPU still
 * writing a complete output. A plane past `rocket_rk3576_max_task_rows()` must therefore
 * be split, and does not merely run slowly. The hardware's split is by input rows. Each
 * task reads a row window of the full plane and writes the output rows that window
 * supports. The full plane is still described alongside the window (`conv_params_t` ih/oh
 * against ih_full/oh_full). The vendor's own toolkit does the same, and its depthwise
 * capture is one such windowed task.
 *
 * The split is needed at ordinary vision-model geometry, not only at extremes. A 112x112
 * plane at ic=32 is 6272 granules against a 6144-granule data cap.
 *
 * `rocket_rk3576_plan_rows()` lays out the sequence. Each entry contains the window, the
 * leading pad that window consumes, and the byte offsets into the feature and output
 * cubes. The offsets are all the caller adds to its base addresses. The cubes are NC1HWC2
 * with a 16-byte channel atom, and the CNA takes the DDR group stride from the full plane
 * (0x1094 = iw*ih_full). One base plus a row offset therefore addresses every channel
 * group correctly.
 *
 * On the ARGB path the feature offset counts packed-image rows (iw*ic bytes). The output
 * offset is unchanged, because the output is a cube on both paths.
 *
 * To emit task t, copy the `conv_params_t`, set these fields, and call
 * `gen_conv2d_int8_rk3576()` or `_dw()` as usual:
 *
 *     ih = t.ih, oh = t.oh, ih_full/oh_full = the full plane, pad_top = t.pad_top,
 *     input_dma  += t.feature_off,
 *     output_dma += t.output_off,
 *
 * Caution: the caller must set `ih_full` and `oh_full` on every task, not only when
 * `*count` > 1. A plan of one task still has a window shorter than the plane whenever the
 * last output row does not reach the last input row. The case is ordinary geometry, not
 * an edge case. Any stride > 1 whose output does not consume the plane leaves trailing
 * input rows unread. A 32x32 k1 s2 VALID conv, for example, plans one task over 31 of its
 * 32 rows.
 *
 * A caller that sets these only for a split hands the emitter a 31-row plane. The DDR
 * channel-group stride then comes out iw*31 instead of iw*32. Every channel group past the
 * first reads at the wrong offset, and the output is unrelated to the input. The DPU
 * writes a complete output of the right size with nothing to fault on. The result looks
 * exactly like a broken geometry encoder [HW sweep, H96 MAX M9].
 *
 * The emitter derives each task's trailing pad from what its last window consumes. A
 * middle task therefore takes none, and the last one takes the plane's bottom pad.
 *
 * Returns 0 on success with `*count` set. Returns a negative value if the geometry cannot
 * be split, or if the sequence needs more than `max_tasks`. A geometry cannot be split
 * when a single output row's window already exceeds the allowance, and the caller's
 * recourse is then an ic split.
 *
 * `ROCKET_RK3576_MAX_ROWS` forces the per-task row cap below what the CBUF allows. A
 * shape that fits one task can then test the split against its own single-task result.
 *
 * These tasks are independent submits, one per window, and each costs about 1.2 ms on top
 * of the compute.
 *
 * Caution: on a kernel whose `rocket` driver programs PC_TASK_CON with the RK3588
 * task-number width, only the first submit of each NPU power session computes. Every
 * later one silently writes nothing. A caller there must check that each task's own rows
 * were written, and let the NPU power-cycle between them.
 */
typedef struct {
    uint16_t iy0;          /* first input row this task reads, in the full plane  */
    uint16_t ih;           /* input rows this task reads                          */
    uint16_t oy0;          /* first output row this task writes                   */
    uint16_t oh;           /* output rows this task writes                        */
    uint8_t  pad_top;      /* leading pad rows this task's first window consumes  */
    uint32_t feature_off;  /* byte offset into the feature cube for this task     */
    uint32_t output_off;   /* byte offset into the output cube for this task      */
    /* CBUF row reuse, used by `gen_conv2d_int8_rk3576_reuse()`. The plain entry points
     * ignore these fields and refetch the whole window. */
    uint16_t retained;     /* leading rows of this window the previous task left  */
                           /* resident in the CBUF (0 on the first task)          */
    uint32_t cbuf_resident;/* CBUF granules the sequence has filled up to here    */
} rocket_rk3576_row_task;

int rocket_rk3576_plan_rows(const conv_params_t *p, int dw,
                            rocket_rk3576_row_task *out, unsigned max_tasks,
                            unsigned *count);

/* Plans the same rows at another precision. The precision changes three things and
 * nothing else:
 *
 * - The CBUF entry count per row
 * - The row cap the allowance affords
 * - On the packed-image first conv, the feature offsets, because a float packed image is
 *   `ic` interleaved halfwords per pixel where an int8 one is bytes
 *
 * `rocket_rk3576_plan_rows()` is this function at `precision_int8`. */
int rocket_rk3576_plan_rows_prec(const conv_params_t *p, int dw, unsigned prec,
                                 rocket_rk3576_row_task *out, unsigned max_tasks,
                                 unsigned *count);

/* Plans int8 rows with a caller's row cap on top of the allowance's, where 0 means none.
 * The cap can only shorten a window, so it cannot make a plan the allowance refuses. The
 * int8 matmul uses it to keep each task on an allowance whose weight area holds every
 * group of its tile. */
int rocket_rk3576_plan_rows_cap(const conv_params_t *p, int dw, unsigned row_cap,
                                rocket_rk3576_row_task *out, unsigned max_tasks,
                                unsigned *count);

/* ---- Emitting a windowed task that reuses the previous task's overlap rows ----
 *
 * Emits a windowed task that keeps the previous task's overlap rows resident in the CBUF.
 * Consecutive row windows share kh - stride_y input rows. The plain entry points refetch
 * them, and each task fetches its whole window against a CBUF base of zero. The plain
 * entry points are self-consistent and bit-exact, and are the ones to use unless there is
 * a reason not to.
 *
 * The reuse entry follows the vendor's programs. It keeps the shared rows resident and
 * moves the CBUF write pointer:
 *
 * - The task fetches ih - retained rows, not ih.
 * - The CBUF window starts at `cbuf_resident - entries*retained` granules.
 * - Fetching resumes at `cbuf_resident`.
 * - The feature address points at the first new row, not at the window start. The caller
 *   must pass `input_dma` advanced by (iy0 + retained) rows, not by iy0.
 *
 * The arithmetic reproduces all six of the vendor's two-task splits exactly, at three
 * kernel sizes and both strides. The six include the k=1 case, where the overlap is zero
 * and the vendor resets the base and does not continue. A three-task sequence is the
 * natural extension of the same arithmetic, and no capture confirms it.
 *
 * The reuse saves `entries * retained` granules of feature DMA per continuation task,
 * under a tenth of the window's traffic at the captured geometries. A row window costs
 * about 1.2 ms on the RK3576, and that time is the driver's completion poll, not the
 * fetch. The reuse therefore does not shorten a windowed conv measurably. Its value is
 * that it matches the vendor's programs, which makes those programs a complete oracle.
 *
 * Caution: a wrong base reads resident rows that are not there, and corrupts the output
 * silently. The reuse is not the default for this reason.
 *
 * `index` is the task's position in the sequence. The register that marks a task as a
 * continuation is set for every task past the first, whether or not it retains anything.
 * The caller must pass the task's own `conv_params_t` (window rows in ih/oh, plane in
 * ih_full/oh_full), exactly as for the plain entry point.
 */
int gen_conv2d_int8_rk3576_reuse(conv_params_t *p, int dw,
                                 const rocket_rk3576_row_task *t, unsigned index);

/*
 * Returns the elements per output surface in DDR, the distance from one 16-channel output
 * group to the next. It is the plane on the direct path, and the plane rounded up to four
 * on the depthwise one.
 *
 * The caller must size the output BO with this value and de-tile with it.
 *
 * Caution: assuming ow*oh_full on a depthwise conv places every group past the first up
 * to four elements early. The error is invisible at any plane whose ow*oh_full is already
 * a multiple of four, so 16x16 passes and 15x18, 17x19 and 19x19 do not.
 */
unsigned rocket_rk3576_out_surf_elems(unsigned ow, unsigned oh_full, int dw);

/* Sizes and packs the coefficient buffer the RK3576 reads at BS_BASE_ADDR (`0x5020`). It
 * differs from the RK3588's flat per-OC int32 bias array, and is groups of 64 bytes, each
 * covering 8 output channels:
 *
 *     A[oc]  int32  at (oc%8)*4        per-channel bias term
 *     B[oc]  int16  at 32 + (oc%8)*2   weight-zero-point correction
 *     C[oc]  int16  at 48 + (oc%8)*2   per-channel multiplier
 *
 * Channel `oc` is in group `oc/8`. Handed a flat int32 array, the RK3576 reads that array
 * as this structure. The result looks like a bias applied to the first 16 channels only,
 * on alternating channels, at 1024x the intended magnitude.
 *
 * Caution: `C` is required, and its effect is not confined to the bias. It enables the
 * whole BS stage. A buffer of zeros makes the DPU write an empty output of the right size,
 * whatever the CNA and the MAC did. A caller who does not know about this layout hands
 * over exactly such a buffer. The failure looks exactly like a wrong geometry encoder
 * [HW sweep, H96 MAX M9]. The packing is in the library, and not in each caller, for that
 * reason.
 *
 * `rocket_rk3576_pack_coeff()` writes `B` = 0, which is what symmetric quantization wants.
 * A non-zero weight zero point goes through `rocket_rk3576_pack_coeff_asym()` below, whose
 * B term was measured at 1, 40, 127, -37 and -128.
 *
 * The groups of A, B and C terms are followed by one more 64-byte group, left at zero,
 * which the emitter points BS_BASE_ADDR1 (`0x5024`) at. BS_BASE_ADDR1 is a live operand
 * base. The DPU reads one 32-bit shift word through it. It right-shifts the accumulator by
 * bits[5:0] where the result is non-negative, and by bits[13:8] where it is negative. Zeros
 * mean no shift. A buffer sized without this tail points the DPU past the end of the BO
 * [HW sweep, H96 MAX M9].
 *
 * `rocket_rk3576_coeff_bytes()` sizes the buffer, and the caller must allocate exactly
 * this much: the groups are padded and the shift-word tail is included.
 * `rocket_rk3576_pack_coeff()` fills it. `bias` can be NULL for a bias-free conv, which
 * still needs the buffer for its `C` terms. Returns 0, or a negative value if `dst_bytes`
 * is too small.
 */
size_t rocket_rk3576_coeff_bytes(unsigned oc);
int rocket_rk3576_pack_coeff(void *dst, size_t dst_bytes, const int32_t *bias, unsigned oc);

/*
 * Packs the same buffer for a program of a given precision. How the RK3576 reads `C`
 * depends on the precision. The integer path takes 1, and a float program reads the field
 * as fp16, where the integer 1 is the denormal 6e-8. fp16 1.0 is 0x3C00.
 *
 * Caution: an integer 1 in a float program underflows the whole output to empty, the same
 * silent failure C=0 gives on the integer path.
 *
 * The caller must pass the precision the conv will be emitted at, and the unit multiplier
 * follows. `rocket_rk3576_pack_coeff()` is this function at `precision_int8`
 * [HW sweep, H96 MAX M9].
 */
int rocket_rk3576_pack_coeff_prec(void *dst, size_t dst_bytes, const int32_t *bias,
                                  unsigned oc, unsigned prec);

/*
 * Packs the asymmetric form: a per-output-channel B term, and the C multiplier under the
 * caller's control. `rocket_rk3576_pack_coeff()` is this function with `b_term = NULL`
 * and `multiplier = 1`.
 *
 * The DPU adds the B term. The accumulator correction is `+B*sum(x)`, where sum(x) is the
 * sum of the input elements the output contracted. The RK3576 was measured against a CPU
 * model at B = 1, 40, 127, -37 and -128. `+B*sum(x)` explains every output of every direct
 * shape in the test's `envelope` group exactly. `-B*sum(x)` and an inert B explain only the
 * few percent that coincide.
 *
 * A weight zero point is therefore programmed negated. An asymmetric weight is
 * `w_true = w_stored - wzp`, whose correction is `-wzp*sum(x)`, so the caller must pass
 * `B = -wzp`.
 *
 * Caution: a B of the wrong sign produces a plausible output with a bias-shaped error, and
 * nothing faults.
 *
 * The B term applies to the direct path only. The depthwise coefficient group has no B
 * field, so an asymmetric depthwise weight goes in the weight cube instead. The datapath
 * adds its two live bytes per (channel, tap), so `w - w_zp` splits across them.
 *
 * `multiplier` must not be 0, and the function rejects 0. A zero multiplier disables the
 * whole BS stage for the group, and the DPU writes an empty output of the right size.
 *
 * Evidence: [HW sweep, H96 MAX M9], `tests/rk3576_conv_gate.c` with `ROCKET_G_WZP` driving
 * the B term.
 */
int rocket_rk3576_pack_coeff_asym(void *dst, size_t dst_bytes, const int32_t *bias,
                                  unsigned oc, const int16_t *b_term,
                                  int16_t multiplier);

/* ---- The PPU: pooling ---------------------------------------------------------
 *
 * Pooling is its own program on the RK3576, not a convolution epilogue: 23 PPU writes and
 * 8 PPU_RDMA, and no CNA, CORE or DPU. It reads and writes the same NC1HWC2 cube (16-byte
 * channel atom) the convolution path already packs. A caller that has a feature cube
 * therefore has a pool input.
 *
 * `GlobalAveragePool` does not use this program. It lowers onto convolutions, the same way
 * the RK3588 lowers a reduce. `GlobalMaxPool` is pooling, as two cascaded passes.
 *
 * The input extent is what the windows consume, and the emitter derives it. The caller
 * passes the real plane in `iw`/`ih` and the real output in `ow`/`oh`. The strides are the
 * plane's, and are the only place the full plane appears.
 *
 * The destination base register is `0x6070`. No capture shows it. Every capture stores zero
 * for it, as for the feature, weight, output and bias bases, because the vendor runtime
 * patches those at load time. Five PPU registers read zero in every capture. Driven in turn
 * on the RK3576, only `0x6070` moves the output off its sentinel [HW sweep, H96 MAX M9]
 * (`tests/rk3576_pool_probe.c sweep`). `ppu_dst_reg` selects the register, and 0 takes
 * `0x6070`.
 *
 * Wherever the captures constrain it, the program is register-identical to the vendor's
 * [source-confirmed: RKNN-Toolkit2 capture, rk3576, 23 shapes]. The test
 * `tests/rk3576_pool_probe.c gate` checks its output against a CPU model. */
enum {
    ROCKET_RK3576_POOL_MAX = 0,
    ROCKET_RK3576_POOL_AVG,
    ROCKET_RK3576_POOL_AVG_NOPAD   /* average with the pad excluded from the divisor */
};

typedef struct {
    uint16_t iw, ih, c;          /* input plane and channels (the real plane)      */
    uint16_t ow, oh;             /* output plane                                   */
    uint8_t  kw, kh;             /* window, in 4-bit fields, so at most 16         */
    uint8_t  stride_x, stride_y; /* likewise                                       */
    uint8_t  pad_left, pad_right, pad_top, pad_bottom;
    uint8_t  mode;               /* ROCKET_RK3576_POOL_*                           */
    int32_t  input_zero_point;   /* the average path's pad value, unused for max   */
    uint32_t input_dma, output_dma;
    uint16_t ppu_dst_reg;        /* 0 = 0x6070, the destination base, above        */
    /* Elements per input channel group: the source surface stride, in the units
     * PPU_RDMA programs. A value of 0 takes round4(iw*ih), which the vendor's own pooling
     * programs contain and which a cube this emitter allocates for itself uses. A caller
     * whose input cube comes from elsewhere passes that cube's stride. A convolution's
     * output surface has a stride of exactly ow*oh, and the join then needs no copy. The
     * emitter applies the field verbatim. A caller can therefore also use it to test
     * whether the PPU takes a stride that is not a multiple of four. */
    uint32_t src_surf_elems;
    /* Elements per input row: the DDR line stride that `0x7024` contains, a different
     * quantity from the plane. `iw` is what the windows consume against, and this field
     * is how far apart the rows holding it sit. A value of 0 takes `iw`, so every program
     * that does not set it is byte-identical.
     *
     * A producer whose surface is wider than its tensor sets it. The packed-image first
     * conv materializes its pad columns and writes a 128-wide surface for a 112-wide
     * output. A consumer told the pitch reads that surface in place, and the producer
     * does not have to de-tile and re-scatter it. Only the PPU can do this. The CNA's
     * feature DMA specifies the width and the line stride as one quantity in two units,
     * so a convolution consumer has no such register. */
    uint32_t src_line_elems;
    /* Elements per output channel group: the destination surface stride that `0x607C`
     * contains. A value of 0 takes round4(ow*oh), the value the vendor's own pooling
     * programs contain, so every program that does not set it is byte-identical.
     *
     * A caller writing into a buffer another producer allocates sets it. A shared
     * concatenation buffer's group stride is exactly the plane, because a direct
     * convolution's output surface stride is `ow*oh` and the other operands are
     * convolutions. The RK3576 honors any value at or above the plane. The honored values
     * include odd ones, and ones that are neither a multiple of four atoms nor of a 64-byte
     * line. At a 5x3 plane, 240 bytes and 272 bytes both write every atom. A stride of 224
     * bytes (one atom under the plane) loses one, which shows the register is live
     * [HW sweep, H96 MAX M9, tests/rk3576_pool_probe.c dst].
     *
     * There is no destination line stride to go with it. `0x6084` is inert, and the rows
     * are always the programmed output width. A slice can therefore be a plane inside a
     * deeper buffer, but never a plane inside wider rows. */
    uint32_t dst_surf_elems;
    uint64_t *tasks;             /* at least RK3576_POOL_TASK_OPS words            */
    uint32_t task_count;         /* words written                                  */
} pool_params_rk3576_t;

#define RK3576_POOL_TASK_OPS 40

int gen_pool_rk3576(pool_params_rk3576_t *p);

/* ---- The DPU LUT: loading the two tables --------------------------------------
 *
 * A nonlinear activation on the RK3576 is two programs. The first uses DPU and DPU_RDMA
 * only, with no CNA or CORE, and loads the table without computing. It drives a 1x1x16
 * dummy cube and writes 1026 entries in a burst through a two-register window. The second
 * is an ordinary convolution with the LUT enabled, and it computes.
 * `gen_lut_load_rk3576()` emits the first.
 *
 * In the window, `0x4100` (LUT_ACCESS_CFG) selects a table and the start offset. Every
 * later write to `0x4104` (LUT_ACCESS_DATA) stores one entry and advances. `0x00020000`
 * selects the low table and `0x00030000` the high one, with 513 entries each. The entries
 * are 512 intervals with both endpoints, the shape a linear-interpolating LUT needs.
 *
 * The entries are the function's output. The vendor compiler writes Q15 of it, and pairs
 * that with an OUT_CVT that converts Q15 to the op's own quantization.
 *
 * `scratch_dma` is where the dummy cube's 16 bytes are written. The vendor stores zero,
 * which is a real IOVA in this stack. The caller must pass a BO of its own, and not
 * inherit whatever buffer is at IOVA 0 [source-confirmed: RKNN-Toolkit2 capture, rk3576]. */
#define RK3576_LUT_ENTRIES   513
#define RK3576_LUT_TASK_OPS  1128   /* the emitted program is 1121, as the capture is */

typedef struct {
    const int16_t *lo;           /* RK3576_LUT_ENTRIES entries, table 0x00020000   */
    const int16_t *hi;           /* RK3576_LUT_ENTRIES entries, table 0x00030000   */
    uint32_t scratch_dma;        /* the dummy cube's destination and source        */
    uint64_t *tasks;             /* at least RK3576_LUT_TASK_OPS words             */
    uint32_t task_count;         /* words written                                  */
} lut_load_params_rk3576_t;

int gen_lut_load_rk3576(lut_load_params_rk3576_t *p);

/* ---- The DPU LUT: applying a loaded table -------------------------------------
 *
 * Describes the window a convolution's LUT stage applies. The LUT is the EW stage, between
 * BN and the OUT_CVT, so a convolution with this descriptor computes:
 *
 *     out = OUT_CVT( LUT( (acc + A[oc] + B[oc]*sum(x)) * C[oc] ) )
 *
 * The BN stage is bypassed. The caller must attach the descriptor to `conv_params_t.lut`,
 * and run `gen_lut_load_rk3576()`'s program as an earlier task of the same `rocket` job.
 * The tables are in the LUT RAM, and a separate submit can take a runtime-PM cycle in
 * between.
 *
 * The input map is `index = (value - le_start) / 2^sel`. The `0x00020000` table covers
 * [le_start, 0] and the `0x00030000` one [0, lo_end), 512 intervals each. The hardware
 * interpolates linearly between entries at the full resolution of the step. The domain is
 * half-open at the top: `value == lo_end` takes `clamp_hi`, and `value == le_start` is in
 * the domain and reads the low table's entry 0. A value below `le_start` takes
 * `clamp_lo`.
 *
 * The vendor never moves this window. It scales the value into a fixed window with the BS
 * stage's per-channel `C`. Every activation it compiles therefore emits a
 * register-identical program with different table contents. All three registers are live,
 * and this descriptor programs them.
 *
 * Caution: every vendor activation also sets BN_CFG `0x4060` = 0x20, the BN stage active.
 * Transcribing it pins the LUT at the table join for every input across twenty-two
 * binades, while the readout still tracks the table. The result looks like a working LUT
 * with no input map in any register. This emitter leaves BN_CFG at 0x903, bypassed, and
 * the map appears.
 *
 * Evidence: [HW sweep, H96 MAX M9], `tests/rk3576_lut_probe.c` over four spans of 8192
 * samples each, every output bit-explained to at most one output count. */
typedef struct lut_rk3576 {
    int32_t le_start;   /* the datapath value at the low table's entry 0            */
    int32_t lo_end;     /* one past the high table's domain: >= it takes clamp_hi   */
    uint8_t sel;        /* the index step is 2^sel                                  */
    int16_t clamp_lo;   /* what a value below le_start reads                        */
    int16_t clamp_hi;   /* what a value at or above lo_end reads                    */
} lut_rk3576_t;

/* The activations the RK3576 table builder covers. Each is the function applied to the
 * real input value, and the builder handles the quantization on both sides.
 *
 * `exp`, `gelu` and `sqrt` are absent. No vendor capture emits a 513-entry table for any
 * of them, so they lower some other way. A table for them here would be an invention and
 * not a transcription. */
enum {
    ROCKET_RK3576_ACT_SIGMOID = 0,   /* 1/(1+e^-x)                                  */
    ROCKET_RK3576_ACT_TANH,
    ROCKET_RK3576_ACT_SWISH,         /* x*sigmoid(x)  (SiLU)                        */
    ROCKET_RK3576_ACT_HARDSWISH,     /* x*relu6(x+3)/6                              */
    ROCKET_RK3576_ACT_HARDSIGMOID,   /* relu6(x+3)/6                                */
    ROCKET_RK3576_ACT_ELU,           /* x >= 0 ? x : e^x - 1                        */
    ROCKET_RK3576_ACT_KINDS
};

/* Returns the name of `kind`, for a log line or a test's table, or NULL for an unknown
 * kind. */
const char *rocket_rk3576_act_name(int kind);

/* Builds the two tables and the window for `kind`, over a datapath whose value `v` means
 * the real number `v * value_scale`. The entries encode the op's output in units of
 * `entry_scale`, so entry = f(v*value_scale) / entry_scale.
 *
 * `le` and `lo` have `RK3576_LUT_ENTRIES` entries each, and `w` receives the window. `sel`
 * picks the step, and the window is then the symmetric [-512*2^sel, 512*2^sel). The clamps
 * are set to the tables' own endpoints. The endpoints are the vendor's convention, and
 * they make the saturating tails of sigmoid/tanh exact outside the table.
 *
 * Returns 0, or a negative value for an unknown kind or a null argument. Entries saturate
 * at int16 and do not wrap, so a caller that picked `entry_scale` too small gets a clamped
 * curve and not a folded one. */
int rocket_rk3576_lut_build(int kind, double value_scale, double entry_scale,
                            unsigned sel, int16_t *le, int16_t *lo, lut_rk3576_t *w);

/* Converts an fp32 conv scale to the OUT_CVT pair the emitter programs, through
 * `npu_out_cvt_pair()` in `npu_requant.h`. `mul` is the 15-bit multiplier, and bit 15 is
 * never set, because the RK3576 reads it as a sign. `shift` is the register value, already
 * pre-decremented.
 *
 * A per-channel requant must size its C multipliers against the gain the emitter
 * programs. The caller must take that gain from this function, because a gain re-derived
 * independently drifts from the emitter's. */
void rocket_rk3576_requant_params(float conv_scale, unsigned *mul, unsigned *shift);

/*
 * Packs the per-output-channel form. `c_term[oc]` is that channel's C multiplier, and
 * `multiplier` is the fallback for a NULL `c_term`. A per-axis weight quantization is
 * applied through this field.
 *
 * The BS stage adds `A` and then multiplies by `C`. The output is
 * `(acc + A[oc] + B[oc]*sum(x)) * C[oc]`, so a bias quantized in the accumulator domain
 * takes the per-channel gain with no extra step. The caller must not pre-divide the bias
 * by the gain. The two orders use the same buffer and different arithmetic.
 *
 * Caution: the wrong order produces a bias scaled by the wrong channel gain, a plausible
 * output with no fault.
 *
 * The product is int32 and saturates. Across 2^31 at a fixed accumulator, every inexact
 * test point implies the same limit, 2.147e9 to 2.158e9 against 2^31 = 2.1475e9, and none
 * wraps. A planner sizing C therefore has a bound to stay inside, not a cliff:
 * `|(acc + A) * C| <= INT32_MAX`.
 *
 * The OUT_CVT then applies one `(v*MUL)>>SHIFT` to every channel, so the composite
 * per-channel gain is `C[oc]*MUL/2^SHIFT`. A per-channel scale is therefore expressible,
 * and a per-channel shift is not. TFLite per-axis requant uses both, so this reproduces a
 * per-axis model closely and not bit-exactly. Two things bound the fidelity:
 *
 * - C is an integer, so a channel's gain resolution is 0.5/C[oc].
 * - The int32 clamp caps C[oc] at `INT32_MAX / max|acc + A|`, which falls as the layer's
 *   fan-in grows.
 *
 * Every entry must be non-zero, and the function rejects a 0 before submission. A zero C
 * disables the BS stage for that channel's whole 8-channel group. The DPU then writes an
 * empty output of the right size with nothing to fault on.
 *
 * Evidence: [HW sweep, H96 MAX M9, tests/rk3576_coeff_c.c]. The test compares C=1 against
 * C=2, with a known accumulator and a known `A`. `(acc + A)*C` explains 32 of 32 channels,
 * and `acc*C + A` explains only the 16 where C is 1.
 */
int rocket_rk3576_pack_coeff_perc(void *dst, size_t dst_bytes, const int32_t *bias,
                                  unsigned oc, const int16_t *b_term,
                                  const int16_t *c_term, int16_t multiplier);

/*
 * Sizes and packs the depthwise coefficient buffer. The depthwise path reads a different
 * group from the direct path, and these functions pack it. The group covers the same 8
 * output channels in 48 bytes, with no B field:
 *
 *     A[oc]  int32  at (oc%8)*4        per-channel bias term
 *     C[oc]  int16  at 32 + (oc%8)*2   per-channel multiplier
 *
 * Channel `oc` is again in group `oc/8`.
 *
 * Caution: a depthwise conv given the 64-byte direct group reads its two indices apart.
 * `A` strides 4 bytes per channel and `C` strides 2, so past the first eight channels each
 * reads out of a different group. Most channels then multiply a bias by zero and never
 * reach DDR. A few square their own bias, and one group's C multipliers are read as the
 * next group's biases. The output is correctly sized and mostly empty, with no fault, the
 * same failure a wrong geometry register gives [HW sweep, H96 MAX M9].
 *
 * The group has no place for a weight zero point, so there is no asymmetric form. None is
 * needed. The RK3576's int8 depthwise cube gives every (channel, tap) two live bytes, and
 * the datapath adds them. `w - w_zp` is therefore represented exactly in the cube over
 * [-256, 254].
 *
 * C is per output channel here exactly as in the 64-byte group, so a per-axis weight
 * quantization also applies on this path. `rocket_rk3576_pack_coeff_dw_perc()` takes the
 * per-channel ramp, and the two scalar entries are it with `c_term = NULL`.
 *
 * Caution: C must never be 0. A zero multiplier disables the whole 8-channel group's BS
 * stage, and the DPU writes an empty output of the right size with no fault.
 *
 * The caller must size the buffer with `rocket_rk3576_coeff_bytes_dw()`. It is smaller
 * than the direct buffer for the same channel count, and the emitter points BS_BASE_ADDR1
 * at the zeroed tail group this sizing includes.
 */
size_t rocket_rk3576_coeff_bytes_dw(unsigned oc);
int rocket_rk3576_pack_coeff_dw(void *dst, size_t dst_bytes, const int32_t *bias,
                                unsigned oc);
int rocket_rk3576_pack_coeff_dw_prec(void *dst, size_t dst_bytes, const int32_t *bias,
                                     unsigned oc, unsigned prec);
int rocket_rk3576_pack_coeff_dw_perc(void *dst, size_t dst_bytes, const int32_t *bias,
                                     unsigned oc, const int16_t *c_term,
                                     int16_t multiplier);

/*
 * Writes the DPU shift word into the tail group a packed coefficient buffer already
 * reserves, the one BS_BASE_ADDR1 points at. The program has `oc` output channels on the
 * direct (`dw` 0) or depthwise (`dw` 1) group layout. The caller must call it after the
 * pack, which zeroes that group.
 *
 * The stage then computes `sat32(rne(((acc + A)*C) >> s))`. The product is held wide, and
 * the shift rounds half to even [HW sweep, H96 MAX M9, tests/rk3576_coeff_c.c shift]. The
 * word contains one 6-bit shift per sign of the product: bits[5:0] for a non-negative one
 * and bits[13:8] for a negative one. `rocket_rk3576_pack_bs_shift()` writes `s` into both.
 *
 * Caution: a single field leaves the other sign at a different scale, and an output zero
 * point at the -128 rail hides the error.
 *
 * `s` is 0-63. Returns 0, or a negative value for a buffer too small or a shift out of
 * range.
 */
int rocket_rk3576_pack_bs_shift(void *dst, size_t dst_bytes, unsigned oc, int dw,
                                unsigned s);

/* ---- The DPU's elementwise stage: one cube in, requantized ---------------------
 *
 * Emits the vendor's elementwise op, which on the RK3576 is a program of its own. It uses
 * DPU and DPU_RDMA only, with no CNA or CORE, the same shape as the LUT table load, and
 * PC_OPERATION_ENABLE at 0x18. As measured on the RK3576, it computes:
 *
 *     out = sat8( ((ew + ew_offset) * ew_scale >> ew_shift) * out_scale >> out_shift )
 *
 * The program takes one operand, and applies an affine requantization to a cube in the
 * NC1HWC2 layout the conv path already packs. It is bit-exact over channel counts 16-320
 * and planes 1x1 to 28x28 (`tests/rk3576_add_probe.c gate`, 10 shapes). The final shift
 * rounds half to even, the same rule the direct path's OUT_CVT uses.
 *
 * A second operand is not reachable through this register set. Only 0x5024 has any
 * effect, and it injects a constant, because it is the DPU shift word, a live operand.
 *
 * A residual add is therefore lowered onto the convolution datapath, where it is
 * bit-exact. The two operands are concatenated along channels and convolved with a 1x1
 * kernel of two diagonal blocks. The convolution form supports different per-operand
 * scales (the ratio is in the weights) and both zero points (the second's in the bias),
 * which this stage cannot express. It also fuses into a producing convolution
 * (`tests/rk3576_residual_add.c`). `gen_ew_int8_rk3576()` remains a cheap standalone
 * requantization of one cube.
 *
 * Caution: the program has three hazards:
 *
 * - Every base the program leaves at zero reads IOVA 0, which is a real buffer in this
 *   stack (per-fd IOVA starts at zero). The caller must allocate a guard BO first, or the
 *   first operand allocated is what those bases read.
 * - DPU 0x40D0 must be the captured 0x0040FFFF verbatim. Written as a clamp pair, it leaves
 *   exactly half of every 16-channel group unwritten, silently. The clamps are
 *   OUT_CLAMP_MIN/MAX (0x40A4/0x40A8), and the no-clamp pair is INT32_MIN/MAX.
 * - OUT_CVT_SCALE is a signed 16-bit field, and 32768 flips the output's sign.
 *
 * Note: `Add(Conv(x), x)` does not produce a capture of this program. The vendor compiler
 * folds an identity skip into the convolution's own kernel, as the center tap of the
 * diagonal. Six such ONNX graphs compiled to a program indistinguishable from a plain
 * conv, and two convs into an add fold as well. A capture needs an operand the compiler
 * cannot reach: a graph input, a constant tensor, or a real bottleneck whose skip crosses
 * three convolutions.
 *
 * `mode` selects the vendor's ADD or MUL word, and the two differ in four registers. Only
 * ADD's arithmetic is tested.
 *
 * Evidence: the program is [source-confirmed: RKNN-Toolkit2 capture], and its arithmetic
 * is [HW sweep, H96 MAX M9, 2026-07-31]. The search for a second operand is
 * exhaustive, and `tests/rk3576_add_probe.c` records its six sweeps. They cover the zero
 * and unwritten registers, the output gain, the main DMA feed, the mode registers,
 * destination accumulation and the span the program reads. */
enum {
    ROCKET_RK3576_EW_ADD = 0,
    ROCKET_RK3576_EW_MUL          /* emitted, not tested */
};

typedef struct {
    uint16_t w, h, c;            /* the plane and channel count, c a multiple of 16 */
    uint8_t  mode;               /* ROCKET_RK3576_EW_*                             */
    uint32_t src_dma;            /* programmed, and measured not to be read        */
    uint32_t ew_dma;             /* the operand cube, NC1HWC2                      */
    uint32_t dst_dma;            /* the output surface                             */
    /* Elements per channel group. A value of 0 takes w*h, which the vendor's own programs
     * contain and a convolution's output surface has. A requantization joined to a
     * convolution's cube therefore needs no copy at either end. The RK3576 has one
     * source-side stride register, and the output has its own. */
    uint32_t surf_elems, dst_surf_elems;
    int32_t  ew_offset;          /* added to the operand before its scale          */
    uint16_t ew_scale;
    uint8_t  ew_shift;
    int32_t  out_offset;         /* OUT_CVT_OFFSET: emitted, not tested            */
    uint16_t out_scale;          /* signed 16-bit in the hardware                  */
    uint8_t  out_shift;
    /* OUT_CLAMP_MIN/MAX, applied before the int8 saturation, where a fused ReLU6's clamp
     * goes. The capture's no-clamp pair is INT32_MIN/INT32_MAX. */
    int32_t  clamp_lo, clamp_hi;
    uint64_t *tasks;             /* at least RK3576_EW_TASK_OPS words              */
    uint32_t task_count;
} ew_params_rk3576_t;

#define RK3576_EW_TASK_OPS 96    /* the emitted program is 93: the capture's 89 plus
                                  * the four-word PC trailer it stores apart        */

int gen_ew_int8_rk3576(ew_params_rk3576_t *p);

/* Splits one requantization gain across the two converters.
 *
 * The overall gain is (ew_scale >> ew_shift) * out_scale / 2^out_shift. The two pairs are
 * redundant, and together they reach a finer grid than either alone. The function fills
 * all four and returns the relative error scaled by 1e9, where 0 is exact. It returns a
 * negative value if no representable pair reaches the gain. It is pure and does not touch
 * the hardware. */
int64_t rocket_rk3576_ew_params(double gain,
                                uint16_t *ew_scale, uint8_t *ew_shift,
                                uint16_t *out_scale, uint8_t *out_shift);

/* ---- The matmul-form fp16 program ----------------------------------------------
 *
 * One task computes C[m][n] = sum_k A[m][k] * B[n][k] with fp16 operands and an fp32
 * output, contracting the whole of `K`. It is a different program from the fp16
 * convolution above, which contracts sixteen input channels a task. This program reads the
 * activation row-major, [m][k], with CNA 0x1090 = k/8 and bit 29 of CNA 0x100C set. It
 * takes the w4a16 output stage (CORE 0x3018 = 0x10000200).
 *
 * The program is charsiu's (gahingwoo/charsiu `524e10a`, `charsiu_emit_job` with no
 * `CHARSIU_*` environment variable set), transcribed word for word. It is 148 words in
 * charsiu's order, 23 of them closed forms of (m, k, n) and five of them buffer
 * addresses. The host half of `tests/rk3576_mm_fp16_gate.c` checks the stream, the
 * coefficient buffer and the weight permutation against charsiu's own output at every
 * tested shape.
 *
 * The program reads these buffers:
 *
 * - The activation, row-major fp16, m*k*2 bytes plus a 4 KiB tail
 * - The weights, [n/16][k/32][n%16][k%32] with two-byte elements, k*n*2 bytes
 * - The coefficient buffer, `rocket_rk3576_mm_fp16_coef_bytes(n)`, which depends on n
 *   alone and can be built once per n
 * - The output, m rows of n fp32 values, row-major, plus a 4 KiB tail
 *
 * One task takes at most `rocket_rk3576_mm_fp16_task_rows(k)` rows. Its input surface,
 * k/32 CBUF entries a row, must fit the 4096-entry data window. `k` runs to 6144 and `n`
 * to 8192 (n%16).
 *
 * Caution: a task past the data window produces wrong output on every row, and does not
 * fail.
 *
 * `rocket_rk3576_mm_fp16_task_ok()` checks those limits, and `gen_matmul_fp16_rk3576()`
 * refuses a task outside them. `rocket_rk3576_mm_fp16_shape_ok()` checks the limits of the
 * matmul entry, which tiles M over tasks. Each bound was measured on the RK3576, and the
 * evidence is beside the definitions in `npu_regcmd_rk3576.c`. */
#define RK3576_MM_FP16_OPS 148

unsigned rocket_rk3576_mm_fp16_task_rows(unsigned k);   /* 0 for a k no task takes */
int rocket_rk3576_mm_fp16_task_ok(unsigned m, unsigned k, unsigned n);
int rocket_rk3576_mm_fp16_shape_ok(unsigned m, unsigned k, unsigned n);
int gen_matmul_fp16_rk3576(uint64_t *ops, unsigned m, unsigned k, unsigned n,
                           uint32_t in_dma, uint32_t weight_dma, uint32_t out_dma,
                           uint32_t coef_dma);
/* Emits the same stream with no limits check, for probes that map where the limits can
 * go. It refuses only what the closed forms cannot express:
 *
 * - An `m` of 0
 * - A `k` under 32 or off a multiple of 32
 * - An `n` under 16 or off a multiple of 16
 *
 * Past charsiu's swept range (m 32, k 4096, n 1024), every shape word is the closed form
 * extrapolated, not a transcription. */
int gen_matmul_fp16_rk3576_unchecked(uint64_t *ops, unsigned m, unsigned k, unsigned n,
                                     uint32_t in_dma, uint32_t weight_dma,
                                     uint32_t out_dma, uint32_t coef_dma);
size_t rocket_rk3576_mm_fp16_in_bytes(unsigned m, unsigned k);
size_t rocket_rk3576_mm_fp16_weight_bytes(unsigned k, unsigned n);
size_t rocket_rk3576_mm_fp16_out_bytes(unsigned m, unsigned n);
size_t rocket_rk3576_mm_fp16_coef_bytes(unsigned n);
/* `B` is n rows of k fp16 bit patterns, and the pack is a permutation of them. */
int rocket_rk3576_mm_fp16_pack_weights(void *dst, size_t dst_bytes, const uint16_t *B,
                                       unsigned k, unsigned n);
int rocket_rk3576_mm_fp16_pack_coef(void *dst, size_t dst_bytes, unsigned n);

#ifdef __cplusplus
}
#endif
#endif /* NPU_REGCMD_RK3576_H */
