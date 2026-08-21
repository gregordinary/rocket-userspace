// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * npu_regcmd_rk3576.h — RK3576 CONV_2D register-command generator.
 *
 * The RK3576 NPU is the same IP family as the RK3588 and runs through the same
 * mainline `rocket` uAPI, but its CNA/CORE/DPU blocks use a DIFFERENT
 * geometry-register encoding at the same block bases: registers move, the
 * bit-packing differs at shared offsets, and the RK3576 drives offsets the RK3588
 * leaves at reset. A single address-translation table is therefore not enough —
 * the geometry-register emission itself is per-chip, which is what this file is.
 *
 * What stays shared with npu_regcmd.c: the descriptor structs (npu_cna_desc /
 * npu_core_desc / npu_dpu_desc), the NPUOP word format, the block targets, the
 * S_POINTER value, and the PC_OPERATION_ENABLE trailer that actually fires the job.
 *
 * Scope: int8/uint8 DIRECT convolution, one task or a row-window sequence, on both
 * of the CNA's feature datapaths — the normal one (in_ch > 4, an NC1HWC2 cube) and
 * the first-conv ARGB one (in_ch <= 4, a packed image), which is a separate
 * sub-encoding selected automatically by the channel count.
 *
 * fp16 direct convolution is here too. Its precision fields are transcribed from a
 * vendor float capture and its cube groups and output map are read off the part; it
 * computes bit-faithfully, delivering every output channel, and one task contracts
 * SIXTEEN input channels. rocket_rk3576_plan_ic() is the split that turns that into
 * a task sequence at an arbitrary channel count. See the fp16 block below.
 *
 * gen_conv2d_dw_int8_rk3576() is the depthwise sibling and computes bit-exactly, at
 * channel counts 8 to 256 including those that are not multiples of 32, kernels 1
 * through 7, both strides, SAME and VALID, planes to 112x112 including the odd ones,
 * and across a row window. Its register program came from manufactured vendor
 * captures; its two BUFFER layouts did not and could not — the coefficient group and
 * the int8 weight cube are read off the part, are each different from the direct
 * path's, and register fidelity to a capture says nothing about either. Both are
 * below: rocket_rk3576_pack_coeff_dw() and rocket_rk3576_weight_dw_int8().
 * [HW sweep, H96 MAX M9]
 */
#ifndef NPU_REGCMD_RK3576_H
#define NPU_REGCMD_RK3576_H

#include "npu_matmul.h"   /* conv_params_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Ops emitted per task: 139 register writes + the 4-word PC trailer, plus headroom
 * for the ROCKET_RK3576_ADD bring-up knob, which appends writes for registers the
 * vendor stream never touches. */
#define RK3576_CONV_TASK_OPS 175

/* int8/uint8 CONV_2D for the RK3576, one task. `_dw` is the depthwise variant.
 *
 * Geometry comes from conv_params_t. Two fields matter more here than on the
 * RK3588 because the RK3576 stream distinguishes the task's row window from the
 * full feature plane: set params->ih / params->oh to the rows THIS task reads and
 * writes, and params->ih_full / params->oh_full to the full plane.
 *
 * ih_full / oh_full may be left 0 ONLY when ih really is the whole plane, because
 * the emitter then derives the DDR channel-group stride from ih. A caller driving
 * rocket_rk3576_plan_rows() below must therefore set them from the plane on EVERY
 * task, including a plan of one — see that function for why a single-task plan can
 * still carry a short window, and what mis-setting this looks like.
 *
 * Returns 0 on success, <0 if the geometry is outside the transcribed envelope
 * (a field would overflow; an ARGB guard below fails).
 *
 * Pass CHANNEL COUNTS THAT SATISFY THE HARDWARE'S GRANULARITY — see
 * rocket_rk3576_pad_ic / _pad_oc below. A violation is warned, not rejected, since
 * the correction changes the caller's buffer sizes. The ARGB path is exempt: 3 is
 * its natural channel count and the granularity rule does not apply there.
 */
int gen_conv2d_int8_rk3576(conv_params_t *params);
int gen_conv2d_dw_int8_rk3576(conv_params_t *params);

/* The int8 direct convolution writing the DPU's RAW int32 accumulator rather than the
 * requantized int8 byte. This is what makes a MATMUL on this part exact past one
 * task's contraction: int8 partials cannot be summed without quantizing each one,
 * where int32 partials add exactly on the host.
 *
 * Two fields change and neither is geometry — DPU 0x4010's output WIDTH takes
 * precision_int32, and OUT_CVT is pinned to exact unity (the QNNPACK derivation lands
 * on scale 16385 for a unit conv scale, which the int8 requant divides away and the
 * raw word does not). The quant scale fields are therefore IGNORED here.
 *
 * The surface's BYTE layout is unchanged: the int8 writer puts 16 lanes in a 16-byte
 * atom and this one puts 4, so a channel group occupies the same bytes and every
 * stride register already holds the right value. What the caller changes is the
 * output BO, four times the size, and the de-scatter, which reads 32-bit words:
 *
 *     word = (oc/4) * ow*oh_full * 4 + 4 * (y*ow + x) + (oc%4)
 *
 * Decoded rather than assumed: driven with operands whose every accumulator is
 * distinct, each one names exactly one word in the raw BO, and every position read
 * that way fits the map. [HW sweep, H96 MAX M9] */
int gen_conv2d_int8_rk3576_i32out(conv_params_t *params);

/* The same raw-int32 convolution with the writer's byte budget DOUBLED, which halves
 * what the weight-cube scatter above costs: the delivered set becomes the first eight
 * output channels of every SIXTEEN rather than of every thirty-two, so a matmul
 * programs two times its output channels instead of four and one submit carries twice
 * the MACs.
 *
 * One field moves, and it is the DPU's own PROC_PRECISION rather than anything about
 * the operands: 0x4010[2:0] from int8 to int32. The operands stay int8, the arithmetic
 * is bit-identical, and the CNA and CORE programs are byte-unchanged. Two atoms per
 * (16-channel block, pixel) is the ceiling — int16, fp16 and bfloat16 reach it too, the
 * float ones destroying the arithmetic on the way, and int4 and int8 do not.
 *
 * The LAYOUT is not the narrow writer's. Address the surface with
 * rocket_rk3576_i32_wide_word(), which carries the bounds this mode adds (oc a multiple
 * of 32; only c%16 < 8 delivered), and size the BO for twice the narrow extent.
 * [HW sweep, H96 MAX M9] */
int gen_conv2d_int8_rk3576_i32out_wide(conv_params_t *params);

/* The word index gen_conv2d_int8_rk3576_i32out_wide() writes output channel `c` at
 * pixel `p` (row-major, p = y*ow + x) to, or -1 if `c` is not a delivered channel.
 * `oc` must be a multiple of 32; see the definition for what a partial group does. */
int rocket_rk3576_i32_wide_word(unsigned ow, unsigned oh_full, unsigned c, unsigned p);

/* ---- The first-conv ARGB datapath (params->ic <= 4) ----
 *
 * gen_conv2d_int8_rk3576() takes this path automatically when the input has 4 or
 * fewer channels, because that is not the same convolution at a small channel count
 * — it is a different CNA program. It is how a vision model's stem runs on this part
 * at all: the normal path needs ic to be a multiple of 32, and an image is 3.
 *
 * WHAT CHANGES FOR THE CALLER, and it is the feature buffer, not the registers:
 *
 *   - The input is a PACKED IMAGE, iw*ic interleaved bytes per row, straight out of
 *     a camera or a decoder. NOT the NC1HWC2 cube every other layer reads. Row
 *     stride is iw*ic bytes, so rocket_rk3576_plan_rows() returns feature offsets in
 *     those units on this path (the output is a normal cube either way).
 *   - Pixels are RAW UINT8. The CNA's converter — bypassed on every other layer —
 *     subtracts params->input_zero_point + 0x80 and hands the MAC a centred int8, so
 *     a plain 0..255 image with input_zero_point 0 is exactly right. The border pad
 *     is inserted before that stage, so the emitter pads in the raw byte domain too.
 *   - iw must be a MULTIPLE OF 16: both the DDR row stride and the CBUF row are
 *     counted in 16-byte granules. Rejected otherwise.
 *   - int8/uint8 only, and no depthwise form. Both rejected.
 *
 * THE WEIGHT CUBE FOR THIS PATH IS NOT TRANSCRIBED, and that is what stands between
 * this and a running first conv. The captures carry register programs, not weight
 * BOs, so what they pin is the cube's SIZE and stride and nothing about its byte
 * order: oc * kh * round16(4*kw) bytes, one 16-byte row per kernel row per output
 * channel, with 4*kw of those 16 bytes live. Packing it is a hardware bring-up item
 * — driving this emitter with a cube laid out by analogy with weight_conv_int8 is a
 * guess, and a wrong weight order here writes a full, correctly sized, wrong surface
 * with nothing to fault on.
 *
 * The one geometry inference: every ARGB capture is kw=3, so the "4 lanes per pixel
 * column, kw columns" fold that the two disagreeing channel counts (12 programmed,
 * 3 in the DMA) express cannot be told apart from a constant 12. The mechanism is
 * what makes 4*kw the reading; a first conv at another kernel width is UNVALIDATED.
 */

/*
 * fp16, AND THE GENERAL PRECISION ENTRY POINT.
 *
 * gen_conv2d_fp16_rk3576() emits ONE fp16 task, which is exactly 16 input channels,
 * and it REFUSES any other ic — any other count computes wrong silently (see the ic
 * split below, which is how a caller reaches an arbitrary channel count). What it
 * emits at ic=16 is bit-faithful against a CPU model at any kernel size, plane and
 * stride, and every programmed output channel reaches DDR.
 * gen_conv2d_rk3576_prec() is the unchecked bring-up entry beside it and applies no
 * envelope at all.
 *
 * The precision fields are transcribed from an RKNN-Toolkit2 float build for this
 * part (do_quantization=False on a float ONNX), which emits a genuine fp16 conv
 * program; they are constant across every geometry it emits. What they settle:
 *
 *   CNA  0x100C  bit21 | proc<<7 | in<<4 | conv_mode   proc AND in are both fp16, and
 *                bit 21 is a float enable the integer programs never set. All three
 *                parts are load-bearing: with any one of them wrong the contraction
 *                reads every feature surface twice and skips the odd ones. The proc
 *                field is the OPERAND width here, not the datapath's.
 *   DPU  0x4038  low half 0x0092, not the direct path's 0x0080. The high half is free
 *                but must be non-zero, or the DPU writes nothing at all.
 *   DPU  0x4050  bit 31 CLEAR. Set, with every other float field right, the
 *                accumulator sums the doubled reads and returns +inf.
 *   DPU  0x4010  out<<29 | in<<26 | proc        out selects a WIDTH: 5 the whole 32-bit
 *                epilogue word, 2 and 3 its low and high halves.
 *   DPU  0x40B0  fp32tofp16_en<<16 | scale      the float narrowing. With it, width 2
 *                writes fp16 and width 3 writes bfloat16; without it, width 2 writes
 *                the fp32 word's low mantissa bits, which are zero for small values.
 *
 * CORE 0x3018 pins the operand TYPE, where only 2 computes; CNA 0x100C's `in` field
 * pins the 2-byte WIDTH CLASS. That is what separates the two words, and it is why an
 * fp16 and a bf16 program are indistinguishable in the CNA one.
 *
 * The float WEIGHT CUBE is not the int8 one: it groups BOTH channel axes by 16
 * (rocket_rk3576_weight_conv_fp16), where the int8 cube groups both by 32. The
 * output group is observable only at k > 1 — the kernel index sits between it and
 * the (oc2, ic2) pair, so at k = 1 it cancels for every candidate value. The feature
 * cube keeps C2 = 8 fp16 lanes per 16-byte atom, and the element size enters the
 * feature side only through the CBUF entry count.
 *
 * THE EPILOGUE IS FLOAT, which decides the coefficient buffer: A (bias) is consumed
 * as fp32 and C (the multiplier) as fp16, so the integer C = 1 that means "no
 * scaling" on the int8 path is the denormal 6e-8 here and underflows the whole
 * surface to empty, with no fault. rocket_rk3576_pack_coeff_prec() folds in fp16 1.0
 * (0x3C00) — use it, not rocket_rk3576_pack_coeff().
 *
 * ONE BOUND REMAINS, and it is a single one rather than the two it once read as:
 * THE CONTRACTION IS SIXTEEN INPUT CHANNELS WIDE. The DPU's output element stride is
 * 16/ic words, so ic = 16 is the count at which an element occupies exactly its own
 * two bytes. Below it the writer spreads each element over a wider slot and half the
 * channels never reach DDR; above it two channels collide in one slot and the odd
 * ones are lost. The "output writer" defect was that stride seen at ic = 8, not a
 * property of the writer. rocket_rk3576_plan_ic() below is the split around the
 * bound.
 *
 * At precision_int8 every one of those fields is zero, so the int8 program is
 * byte-identical to the transcribed one and the capture gate still holds it.
 *
 * A wrong precision field does not fault: it writes a full, correctly sized, wrong
 * surface. Check any result against a CPU model. tests/rk3576_fp16_sweep.c is the
 * harness, and its probes read the lane, pixel and channel maps off the part.
 */
int gen_conv2d_fp16_rk3576(conv_params_t *params);
int gen_conv2d_rk3576_prec(conv_params_t *params, int dw, unsigned prec);

/* The channel counts to PROGRAM, given the logical ones: both round up to a multiple
 * of 32, the group weight_conv_int8 pads its two channel axes to. Convolutions whose
 * channel counts are not multiples of 32 compute WRONG otherwise, at every geometry.
 * No vendor capture shows this, because every captured geometry already satisfies it.
 * [HW sweep, H96 MAX M9]
 *
 * The weight cube needs nothing extra: weight_conv_int8 already pads both channel
 * axes to 32, and its index depends on either count only through ceil(n/32), so a
 * cube packed at the logical count is byte-identical to one packed at the padded
 * count. What the caller must do is size the surrounding buffers to match:
 *
 *   rocket_rk3576_pad_ic — pad the FEATURE CUBE out to the returned channel count
 *   with zeros. ic 8, 16, 17 and 48 all compute wrong while 32 and 64 are exact;
 *   ic=17 fails despite spanning two whole C2=16 surfaces, so the unit is the
 *   32-channel group and not the C2 surface.
 *
 *   rocket_rk3576_pad_oc — size the OUTPUT BO for the returned count, and pass that
 *   count to rocket_rk3576_coeff_bytes() / rocket_rk3576_pack_coeff() as well, so a
 *   padded channel gets a C term rather than a gated-off group.
 *
 * A partial oc group presents two different ways, which is worth knowing because
 * only one of them looks like a channel problem. At k=1 the DPU writes only output
 * ROW 0 of the trailing group and leaves the rest of the surface untouched. At k>1
 * the weights come out wrong instead: WEIGHT_BYTES = ic*oc*kh*kw describes a cube
 * tighter than the padded one, and the truncation drops whole (kh,kw) planes rather
 * than trimming oc inside each — so k=1 is the tolerant case, and an oc that looks
 * fine there (24, 56) is wrong at k=3.
 */
/* Element index into the FLOAT weight cube, 0-based on every axis. The float cube's
 * groups are measured, not inherited: output channels group by 8 and input channels
 * by 16, where the int8 cube groups both by 32. A cube packed at the int8 groups
 * makes the part contract the first 16 input channels twice and spill the rest into
 * the next output channel, which reads as an arithmetic fault rather than a packing
 * one. Returns -1 if the kernel position is out of range. */
int rocket_rk3576_weight_conv_fp16(unsigned oc_total, unsigned ic_total,
                                   unsigned kh_total, unsigned kw_total,
                                   unsigned oc, unsigned ic, unsigned kh, unsigned kw);

unsigned rocket_rk3576_pad_ic(unsigned ic);
unsigned rocket_rk3576_pad_oc(unsigned oc);

/* ---- The first conv's FLOAT weight cube -------------------------------------
 *
 * The packed-image datapath contracts FOUR LANES per pixel whatever the image
 * carries, so this cube has a lane axis of four rather than an input-channel axis,
 * and its taps stay on their own axes — no 4*kw fold and no round-up to sixteen.
 * Decoded from manufactured float captures at ic 3/4, k 1/3/5/7, oc 16/32/48/64:
 *
 *   slot(oc, c, kh, kw) = (oc/16) * (KH*KW*64)
 *                       + kh * (KW*64) + kw * 64
 *                       + (oc%16) * 4 + c
 *
 * 16-bit slots, four lanes per (output channel, tap) with lane c carrying image
 * channel c, output channels interleaved in groups of sixteen inside one tap, tap
 * axis kh-outer. Size the buffer with _bytes(), which rounds oc up to a whole group.
 *
 * This is the FLOAT cube; the int8 one below is a different object.
 * [source-confirmed, RKNN-Toolkit2 rk3576 float build] */
int rocket_rk3576_weight_argb_fp16(unsigned oc_total, unsigned ic_total,
                                   unsigned kh_total, unsigned kw_total,
                                   unsigned oc, unsigned ic, unsigned kh, unsigned kw);
size_t rocket_rk3576_weight_argb_fp16_bytes(unsigned oc, unsigned kh, unsigned kw);
int rocket_rk3576_argb_fp16_pack_weights(void *dst, size_t dst_bytes,
                                         const void *src, unsigned oc, unsigned ic,
                                         unsigned kh, unsigned kw);

/* ---- The first conv's INT8 weight cube ---------------------------------------
 *
 * A weight is ONE BYTE here, output channels group by THIRTY-TWO, and the tap ROW
 * sits outside that group while the tap COLUMN is folded into the same 16-byte row as
 * the four lanes — none of which the float cube does. Read off the part with an
 * impulse image and a one-byte cube (rk3576_conv_gate fcmap):
 *
 *   byte(oc, c, kh, kw) = (oc/32) * (KH * R * 32)
 *                       + kh * (32 * R)
 *                       + (oc%32) * R
 *                       + kw * 4
 *                       + c            R = round16(4*KW)
 *
 * Lane c carries image channel c and the lanes past `ic` are don't-care; 4*KW of each
 * R-byte row is live. A bijection over every live byte at oc 32 and 64, k 3/5/7 and
 * ic 3 and 4. The 32-channel group is observable only above one group — at oc=32 a
 * flat oc*R fits equally. [HW sweep, H96 MAX M9]
 *
 * TWO GEOMETRY BOUNDS COME WITH THE PATH, and no capture shows either because every
 * captured first conv is a 3x3 stride-2 SAME convolution:
 *
 *   THE LEFT PAD MUST BE NON-ZERO. At pad_left = 0 the DPU writes nothing at all —
 *   an untouched surface, not a wrong one — at every plane, stride, kernel and
 *   channel count. CNA_PAD_CON0's pad_left field decides it alone: forcing 0x0100
 *   into a zero-pad program makes that program write, and forcing 0 into a working
 *   one stops it, while the pad_top field does neither.
 *
 *   THE OUTPUT WIDTH MUST BE iw/stride. Anything else writes a full surface that is
 *   SHEARED — the tap a weight lands on drifts one output column per output row, the
 *   signature of a row-stride mismatch — and it shears for a narrow ow and a wide one
 *   alike. Together the two are what SAME padding means.
 *
 * The output-channel count follows the direct path's multiple-of-32 rule (oc=16
 * writes nothing) but NOT the float first conv's 32-channel per-program cap: one int8
 * program delivers 64 output channels. [HW sweep, H96 MAX M9] */
int rocket_rk3576_weight_argb_int8(unsigned oc_total, unsigned ic_total,
                                   unsigned kh_total, unsigned kw_total,
                                   unsigned oc, unsigned ic, unsigned kh, unsigned kw);
size_t rocket_rk3576_weight_argb_int8_bytes(unsigned oc, unsigned kh, unsigned kw);
int rocket_rk3576_argb_int8_pack_weights(void *dst, size_t dst_bytes,
                                         const void *src, unsigned oc, unsigned ic,
                                         unsigned kh, unsigned kw);

/* ---- The depthwise weight cube ----------------------------------------------
 *
 * Not the RK3588's, and not the direct path's. Decoded from vendor captures carrying
 * a unique weight value per (channel, tap), at C = 24, 32, 48, 64 and 128 and at
 * k = 3 and 5:
 *
 *   - a weight occupies a SIXTEEN-BIT slot, whatever the precision, so one geometry's
 *     float and int8 cubes are the same size;
 *   - channels group by 32, each group a contiguous block;
 *   - inside a group the order is TAP-MAJOR, kh outer — every channel of the group
 *     for one tap, then the next tap;
 *   - a trailing partial group is DENSE: its tap stride is the channels it holds, not
 *     32.
 *
 * rocket_rk3576_weight_dw() returns the 16-bit SLOT index (write a 16-bit element
 * there), rocket_rk3576_weight_dw_bytes() the buffer size — which is larger than the
 * slots handed out when C is not a multiple of 16, the tail being padding the DMA
 * still fetches. Returns -1 on an out-of-range position.
 * [source-confirmed, RKNN-Toolkit2 rk3576 depthwise builds] */
int rocket_rk3576_weight_dw(unsigned c_total, unsigned kh_total, unsigned kw_total,
                            unsigned c, unsigned kh, unsigned kw);

/*
 * The same cube at INT8, where a weight is ONE byte. Tap-major inside a channel
 * group, and a trailing partial group is dense, exactly as above — but the GROUP IS
 * SIXTY-FOUR CHANNELS rather than the float cube's 32, and the arrangement inside a
 * tap block is neither the float slot order nor its low byte. Measured on the part
 * (rk3576_conv_gate dwmap):
 * within a tap block the channel a byte carries is 2*(b/4) + (b%2), so channel c holds
 * two live bytes per tap, at 4*(c/2) + (c%2) and two further on, and BOTH contribute.
 * A single group and a k=1 kernel both hide the group size — C=64 k=3 is what separates
 * 64 from 32.
 * This hands out the first; leave the second at zero or the weight is added twice.
 *
 * Writing an int8 weight as a 16-bit value into rocket_rk3576_weight_dw()'s slot puts
 * its sign extension into the byte that belongs to the NEXT channel — a silent -1
 * weight on a neighbour rather than padding.
 *
 * Returns a BYTE offset, or -1 if the position is out of range.
 */
int rocket_rk3576_weight_dw_int8(unsigned c_total, unsigned kh_total, unsigned kw_total,
                                 unsigned c, unsigned kh, unsigned kw);

size_t rocket_rk3576_weight_dw_bytes(unsigned c_total, unsigned kh, unsigned kw);

/* ---- The ic split: an fp16 conv at an arbitrary input-channel count ----------
 *
 * gen_conv2d_fp16_rk3576() emits ONE task and one task contracts exactly 16 input
 * channels, so it REFUSES anything else. This is how a caller gets past that.
 *
 * The bound is not the arithmetic and it is not the surface index, which the float
 * precision fields close. It is the DPU's output element stride, 16/ic words: at
 * ic = 32 two channels land in one slot and the odd ones are lost, and the DPU
 * full, correctly sized surface with nothing to fault on. The split turns that into
 * a bound on a TASK: each slice takes exactly one contraction step, which is the
 * shape the part computes bit-faithfully.
 *
 * Verified bit-exact against a CPU model at ic 32/64/128, oc 32/64, k 1/3/5, planes
 * 8x8 through 32x32, stride 1 and 2. [HW sweep, H96 MAX M9]
 *
 * WHAT IT COSTS: ic/8 submits and a host accumulation. It does not recover the
 * output channels the fp16 writer drops (below) — those are a separate gap.
 *
 * The sequence, and what is per-slice versus shared:
 *
 *   - The FEATURE CUBE is shared and nothing is repacked. At C2 = 8 its channel
 *     groups are contiguous planes of iw*ih_full 16-byte atoms, so slice s is the
 *     same BO at `input_dma + t[s].feature_off`.
 *   - The WEIGHT CUBE is per-slice, and it is NOT a sub-cube of the whole conv's:
 *     each slice is its own convolution, so its group count follows the slice.
 *     rocket_rk3576_fp16_pack_slice_weights() builds it from row-major OIHW weights;
 *     size the BO with rocket_rk3576_fp16_slice_weight_bytes(), which sizes by the
 *     GROUPS (an 8-channel slice at an ic group of 16 needs a cube for 16).
 *   - The COEFFICIENT BUFFER is shared, packed once with
 *     rocket_rk3576_pack_coeff_prec(..., precision_float16) so C carries fp16 1.0.
 *     Give it to the FIRST slice only if it carries a bias, and pass NULL bias to
 *     the rest — every slice adds the whole A term, so a bias on all of them lands
 *     ic/8 times.
 *   - The OUTPUT BO is rewritten by every slice, so read it back between submits and
 *     accumulate with rocket_rk3576_fp16_accumulate().
 *
 * To emit slice s: copy the conv_params_t, set
 *     ic = t[s].ic,  input_dma += t[s].feature_off,  weights_dma = the slice cube,
 * and call gen_conv2d_fp16_rk3576().
 *
 * ACCUMULATION IS ON THE HOST. The DPU eltwise stage does this job on the RK3588
 * (ROCKET_KACC) and would remove ic/16 readbacks here. It is no longer blocked — the
 * partial it would read back is now a plain, dense, complete fp16 cube — so it is the
 * open lever on this path rather than a defect to compose.
 *
 * This plans the ic axis ONLY. The row window (rocket_rk3576_plan_rows) is a separate
 * split on a separate axis; a plane whose 16-channel slice still overflows the CBUF is
 * refused here, and the recourse is to compose the two — plan the rows of a slice, and
 * run the ic slices of each row task.
 *
 * Returns 0 with *count set, or <0 if ic or oc violates the granularity above, if the
 * sequence needs more than max_tasks, or if a slice does not fit one task.
 */
/* The input-channel count one fp16 task contracts. SIXTEEN is the part's working
 * point, and it is the only one: the DPU's output element stride is 16/ic words, so
 * ic=16 is where an element occupies exactly its own two bytes. Below it the writer
 * spreads each element over a wider slot and half the channels never reach DDR;
 * above it two channels collide in one slot and the odd ones are lost.
 * [HW sweep, H96 MAX M9] */
#define ROCKET_RK3576_FP16_IC_SLICE 16u

typedef struct {
    uint16_t ic0;          /* first input channel this task contracts          */
    uint16_t ic;           /* input channels this task contracts               */
    uint32_t feature_off;  /* byte offset into the feature cube for this task  */
} rocket_rk3576_ic_task;

int rocket_rk3576_plan_ic(const conv_params_t *p, rocket_rk3576_ic_task *out,
                          unsigned max_tasks, unsigned *count);

/* The channel counts an fp16 conv must PROGRAM, given the logical ones. Neither is
 * the int8 rule: rocket_rk3576_pad_ic/_pad_oc round to 32 because the integer cube
 * groups both axes there, and the float cube groups oc by 8 while its ic axis is
 * bounded by the slice instead. Pad the feature cube with zeros to _fp16_pad_ic and
 * size the output and coefficient buffers for _fp16_pad_oc.
 *
 * oc is confirmed at 16, 32 and 64; 8 is the group and is not separately measured. */
unsigned rocket_rk3576_fp16_pad_ic(unsigned ic);
unsigned rocket_rk3576_fp16_pad_oc(unsigned oc);

size_t rocket_rk3576_fp16_slice_weight_bytes(unsigned oc, unsigned ic,
                                             unsigned kh, unsigned kw);
int rocket_rk3576_fp16_pack_slice_weights(void *dst, size_t dst_bytes,
                                          const _Float16 *w_oihw,
                                          unsigned oc, unsigned ic_total,
                                          unsigned kh, unsigned kw,
                                          const rocket_rk3576_ic_task *t);

/* ---- Reading an fp16 surface back ----
 *
 * At the 16-channel contraction the writer is the plain native float cube: 8 channels
 * to a 16-byte atom, one atom per pixel, channel groups as contiguous planes, every
 * programmed channel present exactly once. [HW sweep, H96 MAX M9]
 *
 * The map is measured rather than fitted — a probe whose every output lane carries a
 * unique name decodes through it as a bijection. For channel c at pixel (y,x), in
 * 16-bit words: (ow*oh*8)*(c/8) + 8*(y*ow+x) + (c%8).
 *
 * rocket_rk3576_fp16_accumulate() de-scatters one surface through that map and ADDS
 * it into a row-major [out_channels][oh][ow] fp32 accumulator, which is what an ic
 * split needs; zero the accumulator before the first slice.
 */
unsigned rocket_rk3576_fp16_out_channels(unsigned oc);
size_t   rocket_rk3576_fp16_out_bytes(unsigned oc, unsigned oh, unsigned ow);
int rocket_rk3576_fp16_out_index(unsigned oh, unsigned ow, unsigned c,
                                 unsigned y, unsigned x);
int rocket_rk3576_fp16_accumulate(float *acc, const void *surface,
                                  size_t surface_bytes,
                                  unsigned oc, unsigned oh, unsigned ow);

/* CBUF budget planning.
 *
 * The RK3576's feature budget is programmable: CNA_CBUF_CON0 (0x1040) carries a
 * granule ALLOWANCE F in bits[16:27], and one task's feature plane must satisfy
 *
 *     ceil(iw*ic/64) granules per row  x  the task's input rows  <=  4096 + F
 *
 * F=0 is what the vendor's full-plane captures carry, so 4096 granules (256 KiB) is
 * the DEFAULT and not the hardware's limit. The data side reaches 6144 granules
 * (384 KiB), but data and weights share one ~448 KiB pool, so raising F costs the
 * weight path the same capacity: at F=0 a resident weight slice of 175 KiB computes
 * and 200 KiB does not; at F=2048 the slice must stay under 75 KiB. Driving F far
 * enough to leave the weight path nothing WEDGES the CNA, which then writes no output
 * at all. [HW sweep, H96 MAX M9]
 *
 * The emitter plans F from the operand shape on every task, so a caller that just
 * emits a conv gets a correct allowance without asking. These two are for a caller
 * that tiles: use rocket_rk3576_max_task_rows() to choose a row window, since a
 * window past the allowance computes WRONG with the DPU still writing a full surface
 * and no fault to catch it.
 *
 * rocket_rk3576_cbuf_f() returns 0 and writes the allowance to program, or <0 if one
 * task cannot hold this shape — the plane needs more than the data cap (window the
 * rows) or the weight slice does not fit even at F=0 (split ic). `oc`, `kh`, `kw` and
 * `dw` are needed because the weight side of the trade depends on them. Only F values
 * measured to deliver their face value are used (0, 256, 512, 1024, 2048): a
 * combination of allowance bits delivers LESS than their sum, so an arbitrary F would
 * program a budget the hardware does not honour and corrupt silently.
 *
 * AND THE TWO LOW RUNGS ARE PATH-CONDITIONAL. 256 and 512 deliver on the direct path
 * only where the resident slice is at or under 1 KiB, and on the DEPTHWISE path they
 * are not used at all — no threshold on the depthwise footprint fits the measurements,
 * which have 4 granules dead and 32 live. A rung the part does not honour delivers the
 * F=0 budget instead, so the surface is exact, then WRONG across the band of windows
 * that select that rung, then exact again above it: the failure is not monotone in the
 * window height, and a probe that bisects the window cannot see it. Declining the low
 * rungs costs nothing — 1024 is strictly larger, always live, and needs no extra
 * submit. [HW sweep, H96 MAX M9]
 *
 * The pool arithmetic is characterised at ONE output-channel group, and a conv driving
 * SEVERAL loses the trailing ones well before the slice reaches what the pool leaves —
 * one group at a time as the slice grows, with the leading groups bit-exact and a full
 * surface written, so it reads as an output-channel defect rather than a capacity one.
 * The governing quantity is the resident slice 32*ic*kh*kw itself: the same slice
 * behaves identically whichever (ic,kh,kw) produces it. Measured at F=0, at oc=128:
 * 144 KiB drives all four groups, 145-148 KiB three, 152-156 KiB two, 162 KiB one. So
 * ic*kh*kw <= 4608 computes at every oc, and cbuf_f() refuses past a small measured
 * table keyed on the group count rather than letting it corrupt silently — the
 * recourse is an ic split, which is the caller's to choose. [HW sweep, H96 MAX M9]
 *
 * rocket_rk3576_max_task_rows() returns the tallest row window one task can carry for
 * this operand shape, at the largest allowance the weight slice leaves room for, or 0
 * if no window fits.
 */
/* The feature plane's granule cost is per BYTE, so a 2-byte element doubles it and
 * the allowance has to be planned at the precision the conv will be emitted at. The
 * two entry points without a precision are these at int8, which is what every caller
 * of the int8 conv path wants; an fp16 caller must use the _prec forms, because an
 * allowance sized for half the plane computes WRONG with a full surface written.
 *
 * The WEIGHT side of the trade is modelled at int8 in both, deliberately: the float
 * cube's group is 8 output channels of 2-byte elements against the integer cube's 32
 * of 1 byte, so the integer form is an upper bound on the float slice rather than its
 * size, and the graded output-channel-loss table it feeds was measured on int8 only.
 * Over-estimating refuses early instead of corrupting, and it costs nothing at the
 * ic <= 8 the fp16 envelope allows. */
int rocket_rk3576_cbuf_f(unsigned iw, unsigned ic, unsigned ih, unsigned oc,
                         unsigned kh, unsigned kw, int dw, unsigned *f_out);
unsigned rocket_rk3576_max_task_rows(unsigned iw, unsigned ic, unsigned oc,
                                     unsigned kh, unsigned kw, int dw);
int rocket_rk3576_cbuf_f_prec(unsigned iw, unsigned ic, unsigned ih, unsigned oc,
                              unsigned kh, unsigned kw, int dw, unsigned prec,
                              unsigned *f_out);
unsigned rocket_rk3576_max_task_rows_prec(unsigned iw, unsigned ic, unsigned oc,
                                          unsigned kh, unsigned kw, int dw,
                                          unsigned prec);

/* ---- The row window: a conv too tall for one task, cut into a task sequence ----
 *
 * A feature plane over the CBUF allowance computes WRONG with the DPU still writing
 * a full surface, so a plane past rocket_rk3576_max_task_rows() is not a conv that
 * runs slowly — it is a conv that must be split. The split the hardware takes is by
 * INPUT ROWS: each task reads a row window of the full plane and writes the output
 * rows that window supports, with the full plane still described alongside the
 * window (conv_params_t ih/oh vs ih_full/oh_full). The vendor's own toolkit does the
 * same thing, and its depthwise capture is one such windowed task.
 *
 * This matters at ordinary vision-model geometry, not just at extremes: a 112x112
 * plane at ic=32 is 6272 granules against a 6144-granule data cap.
 *
 * rocket_rk3576_plan_rows() lays out the sequence. Each entry carries the window,
 * the leading pad that window consumes, and the BYTE OFFSETS into the feature and
 * output cubes — which is the whole of what the caller adds to its base addresses,
 * because the cubes are NC1HWC2 with a 16-byte channel atom and the CNA takes the
 * DDR group stride from the FULL plane (0x1094 = iw*ih_full). So one base plus a row
 * offset addresses every channel group correctly. On the ARGB path the feature
 * offset counts packed-image rows instead (iw*ic bytes); the output offset is
 * unchanged, because the output is a cube on both paths.
 *
 * To emit task t: copy the conv_params_t, set
 *     ih = t.ih, oh = t.oh, ih_full/oh_full = the full plane, pad_top = t.pad_top,
 *     input_dma  += t.feature_off,
 *     output_dma += t.output_off,
 * and call gen_conv2d_int8_rk3576() / _dw() as usual.
 *
 * SET ih_full / oh_full ON EVERY TASK, not only when *count > 1. A plan of ONE task
 * still carries a window shorter than the plane whenever the last output row does
 * not reach the last input row — which is ordinary geometry, not an edge case: any
 * stride > 1 whose output does not consume the plane leaves trailing input rows
 * unread, so a 32x32 k1 s2 VALID conv plans one task over 31 of its 32 rows. A
 * caller that sets these only for a split then hands the emitter a 31-row plane, and
 * the DDR channel-group stride comes out iw*31 instead of iw*32: every channel group
 * past the first reads at the wrong offset and the surface returns unrelated to the
 * input, with the DPU writing a full, correctly sized result and nothing to fault on.
 * It reads exactly like a broken geometry encoder. [HW, H96 MAX M9] The trailing pad the emitter
 * derives is the pad that task's last window actually consumes, so a middle task
 * takes none and the last one takes the plane's bottom pad.
 *
 * Returns 0 on success with *count set, or <0 if the geometry cannot be split (a
 * single output row's window already exceeds the allowance — the recourse is an ic
 * split, which is the caller's) or if the sequence needs more than max_tasks.
 *
 * ROCKET_RK3576_MAX_ROWS forces the per-task row cap below what the CBUF allows,
 * which is how a shape that fits one task is made to exercise the split against its
 * own single-task result.
 *
 * These tasks are INDEPENDENT SUBMITS, one per window, and cost about 1.2 ms each on
 * top of the compute. On a kernel whose `rocket` still programs PC_TASK_CON with the
 * RK3588 task-number width, only the first submit of each NPU power session computes
 * and every later one silently writes nothing, so a caller there has to check that
 * each task's own rows landed and let the part power-cycle between them. See
 * rockchip-npu-notes/chips/rk3576-regcmd.md.
 */
typedef struct {
    uint16_t iy0;          /* first input row this task reads, in the full plane  */
    uint16_t ih;           /* input rows this task reads                          */
    uint16_t oy0;          /* first output row this task writes                   */
    uint16_t oh;           /* output rows this task writes                        */
    uint8_t  pad_top;      /* leading pad rows this task's first window consumes  */
    uint32_t feature_off;  /* byte offset into the feature cube for this task     */
    uint32_t output_off;   /* byte offset into the output cube for this task      */
    /* CBUF row reuse — see gen_conv2d_int8_rk3576_reuse(). Ignored by the plain
     * entry points, which refetch the whole window. */
    uint16_t retained;     /* leading rows of this window the PREVIOUS task left  */
                           /* resident in the CBUF (0 on the first task)          */
    uint32_t cbuf_resident;/* CBUF granules the sequence has filled so far        */
} rocket_rk3576_row_task;

int rocket_rk3576_plan_rows(const conv_params_t *p, int dw,
                            rocket_rk3576_row_task *out, unsigned max_tasks,
                            unsigned *count);

/* The same plan at another precision, which changes three things and nothing else: the
 * CBUF entry count per row, the row cap the allowance affords, and — on the packed-image
 * first conv — the feature offsets, since a float packed image is `ic` interleaved
 * halfwords per pixel where an int8 one is bytes. rocket_rk3576_plan_rows() is this at
 * precision_int8. */
int rocket_rk3576_plan_rows_prec(const conv_params_t *p, int dw, unsigned prec,
                                 rocket_rk3576_row_task *out, unsigned max_tasks,
                                 unsigned *count);

/* ---- Emitting a windowed task that REUSES the previous task's overlap rows ----
 *
 * Consecutive row windows share kh - stride_y input rows, and the plain entry points
 * refetch them: each task fetches its whole window against a CBUF base of zero. That
 * is self-consistent and bit-exact, and it is what to use unless there is a reason
 * not to. The vendor does something else, and this is that something else.
 *
 * It keeps the shared rows resident and moves the CBUF write pointer instead:
 *
 *   the task FETCHES ih - retained rows, not ih;
 *   the CBUF window starts at `cbuf_resident - entries*retained` granules;
 *   fetching resumes at `cbuf_resident`;
 *   the FEATURE ADDRESS points at the first NEW row, not at the window start —
 *   so pass input_dma advanced by (iy0 + retained) rows, not by iy0.
 *
 * Reproduced exactly by all six of the vendor's two-task splits, at three kernel
 * sizes and both strides, including the k=1 case where the overlap is zero and the
 * vendor resets the base rather than continuing. A THREE-task sequence is the
 * natural extension of the same arithmetic and is not pinned by any capture.
 *
 * WHAT IT BUYS, so the risk is priced honestly: `entries * retained` granules of
 * feature DMA per continuation task — under a tenth of the window's traffic at the
 * captured geometries. A row window costs about 1.2 ms on this part and that is the
 * driver's completion poll, not the fetch, so this does not shorten a windowed conv
 * measurably. It is worth having because it is what the vendor's programs do, which
 * makes those programs a complete oracle; it is not worth defaulting to, because a
 * wrong base reads resident rows that are not there and corrupts silently.
 *
 * `index` is the task's position in the sequence: the register that marks a task as
 * a continuation is set for every task past the first, whether or not it retains
 * anything. Pass the task's own conv_params_t (window rows in ih/oh, plane in
 * ih_full/oh_full) exactly as for the plain entry point.
 */
int gen_conv2d_int8_rk3576_reuse(conv_params_t *p, int dw,
                                 const rocket_rk3576_row_task *t, unsigned index);

/* The coefficient buffer the RK3576 expects at BS_BASE_ADDR (`0x5020`), which is NOT
 * the RK3588's flat per-OC int32 bias array. It is groups of 64 bytes covering 8
 * output channels each:
 *
 *     A[oc]  int32  at (oc%8)*4        per-channel bias term
 *     B[oc]  int16  at 32 + (oc%8)*2   weight-zero-point correction
 *     C[oc]  int16  at 48 + (oc%8)*2   per-channel multiplier
 *
 * with `oc` living in group `oc/8`. Hand the part a flat int32 array instead and it
 * reads that array *as* this structure, which looks like a bias landing on the first
 * 16 channels only, on alternating channels, at 1024x the intended magnitude.
 *
 * `C` is NOT optional and NOT confined to the bias: it gates the whole BS stage, so
 * a buffer of zeros — which is what a caller who does not know about this layout
 * hands over — makes the DPU write a full, correctly sized, entirely EMPTY surface
 * no matter what the CNA and the MAC did. That failure reads exactly like "the
 * geometry encoder is wrong" and is the reason this packing lives here rather than
 * in each caller. [HW sweep, H96 MAX M9]
 *
 * `B` is pinned to 0, which is what symmetric quantization wants; it has not been
 * validated against a non-zero weight zero point.
 *
 * The A/B/C groups are followed by one more 64-byte group, left at zero, which the
 * emitter points BS_BASE_ADDR1 (`0x5024`) at. That register is a live operand base:
 * the DPU reads one 32-bit shift word through it and right-shifts the accumulator
 * by bits[5:0] where the result is non-negative and by bits[13:8] where it is
 * negative. Zeros mean no shift. Sizing the buffer without this tail points the
 * DPU past the end of the BO. [HW sweep, H96 MAX M9]
 *
 * rocket_rk3576_coeff_bytes() sizes the buffer (allocate exactly this much — the
 * groups are padded and the shift-word tail is included), rocket_rk3576_pack_coeff()
 * fills it. `bias` may be NULL for a bias-free conv, which still needs the buffer for
 * its `C` terms. Returns 0, or <0 if `dst_bytes` is too small.
 */
/*
 * Elements per OUTPUT SURFACE in DDR: the distance from one 16-channel output group to
 * the next. The plane on the direct path, and the plane ROUNDED UP TO FOUR on the
 * depthwise one.
 *
 * Size the output BO with this and de-scatter with it. Assuming ow*oh_full on a
 * depthwise conv lands every group past the first up to four elements early, and the
 * error is invisible at any plane whose ow*oh_full is already a multiple of four — so
 * 16x16 passes and 15x18, 17x19 and 19x19 do not.
 */
unsigned rocket_rk3576_out_surf_elems(unsigned ow, unsigned oh_full, int dw);

size_t rocket_rk3576_coeff_bytes(unsigned oc);
int rocket_rk3576_pack_coeff(void *dst, size_t dst_bytes, const int32_t *bias, unsigned oc);

/*
 * The same, for a program of a given precision. HOW `C` IS READ DEPENDS ON THE
 * PRECISION: the integer path takes 1, and a FLOAT program reads the field as fp16,
 * where the integer 1 is the denormal 6e-8 — it underflows the whole surface to empty,
 * the same silent signature C=0 gives on the integer path. fp16 1.0 is 0x3C00.
 *
 * Pass the precision the conv will be emitted at and the unit multiplier follows;
 * rocket_rk3576_pack_coeff() above is this at `precision_int8`.
 * [HW sweep, H96 MAX M9]
 */
int rocket_rk3576_pack_coeff_prec(void *dst, size_t dst_bytes, const int32_t *bias,
                                  unsigned oc, unsigned prec);

/*
 * The asymmetric form: a per-output-channel B term, and the C multiplier under the
 * caller's control. rocket_rk3576_pack_coeff() is this with `b_term = NULL` and
 * `multiplier = 1`.
 *
 * THE DPU ADDS IT: the accumulator correction is `+B*sum(x)`, where sum(x) is the sum
 * of the input elements the output contracted. Settled on the part against a CPU model
 * at B = 1, 40, 127, -37 and -128, each explaining every output of every direct shape
 * in the gate's envelope group exactly, where `-B*sum(x)` and an inert B explain only
 * the few percent that coincide. `ROCKET_G_WZP` in tests/rk3576_conv_gate.c drives it.
 * [HW sweep, H96 MAX M9]
 *
 * SO A WEIGHT ZERO POINT IS PROGRAMMED NEGATED. An asymmetric weight is
 * `w_true = w_stored - wzp`, whose correction is `-wzp*sum(x)`, so pass `B = -wzp`.
 * Getting that backwards produces a plausible surface with a bias-shaped error rather
 * than anything that faults.
 *
 * This is the DIRECT path only. The depthwise coefficient group has no B field at all,
 * so an asymmetric depthwise weight has to be folded into the bias.
 *
 * `multiplier` must not be 0: that gates the whole BS stage off for the group and
 * the DPU writes a full, correctly sized, entirely EMPTY surface. Rejected here.
 */
int rocket_rk3576_pack_coeff_asym(void *dst, size_t dst_bytes, const int32_t *bias,
                                  unsigned oc, const int16_t *b_term,
                                  int16_t multiplier);

/* ---- The PPU: pooling ---------------------------------------------------------
 *
 * Pooling is its OWN program on this part, not a convolution epilogue: 23 PPU writes
 * and 8 PPU_RDMA, no CNA, no CORE, no DPU. It reads and writes the same NC1HWC2 cube
 * (16-byte channel atom) the convolution path already packs, so a caller that has a
 * feature cube has a pool input.
 *
 * `GlobalAveragePool` is NOT this — it lowers onto convolutions, the same way the
 * RK3588 lowers a reduce. `GlobalMaxPool` IS pooling, as two cascaded passes.
 *
 * THE INPUT EXTENT IS WHAT THE WINDOWS CONSUME. The emitter derives it; a caller
 * passes the real plane in `iw`/`ih` and the real output in `ow`/`oh`. The strides are
 * the plane's and are the only place the full plane appears.
 *
 * THE DESTINATION BASE REGISTER IS NOT DECODED. Every capture stores zero for it,
 * exactly as it does for the feature, weight, output and bias bases, because the
 * vendor runtime patches those at load time — so which of the five PPU registers that
 * read zero in every capture receives it has to be read off the part. `ppu_dst_reg`
 * is that register; 0 takes the current best candidate. Until it is settled, this
 * emitter builds a program that is register-identical to the vendor's everywhere the
 * captures constrain it and is not yet known to write anything.
 * [Manufactured capture, RKNN-Toolkit2 for rk3576, 23 shapes] */
enum {
    ROCKET_RK3576_POOL_MAX = 0,
    ROCKET_RK3576_POOL_AVG,
    ROCKET_RK3576_POOL_AVG_NOPAD   /* average with the pad excluded from the divisor */
};

typedef struct {
    uint16_t iw, ih, c;          /* input plane and channels (the real plane)      */
    uint16_t ow, oh;             /* output plane                                   */
    uint8_t  kw, kh;             /* window; 4-bit fields, so at most 16            */
    uint8_t  stride_x, stride_y; /* likewise                                       */
    uint8_t  pad_left, pad_right, pad_top, pad_bottom;
    uint8_t  mode;               /* ROCKET_RK3576_POOL_*                           */
    int32_t  input_zero_point;   /* the average path's pad value; unused for max   */
    uint32_t input_dma, output_dma;
    uint16_t ppu_dst_reg;        /* 0 = the default candidate; see above           */
    /* Elements per input channel group, i.e. the source surface stride in the units
     * PPU_RDMA programs. 0 takes round4(iw*ih), which is what the vendor's own pooling
     * programs carry and what a cube this emitter allocates for itself uses. A caller
     * whose input cube comes from somewhere else — a convolution's OUTPUT SURFACE,
     * whose stride is ow*oh exactly — passes that stride instead, and the join needs no
     * copy. The field is honoured verbatim, so it is also the way to find out whether
     * the PPU takes a stride that is not a multiple of four. */
    uint32_t src_surf_elems;
    /* Elements per input ROW — the DDR line stride `0x7024` carries, which is a
     * different quantity from the plane. `iw` is what the WINDOWS consume against;
     * this is how far apart the rows holding it sit. 0 takes `iw`, so every program
     * that does not set it is byte-identical.
     *
     * A producer whose surface is WIDER than its tensor sets it: the packed-image
     * first conv materialises its pad columns and writes a 128-wide surface for a
     * 112-wide output, and a consumer told the pitch reads that surface in place
     * rather than making the producer de-scatter and re-scatter it. Only the PPU can
     * do this — the CNA's feature DMA carries the width and the line stride as one
     * quantity in two units, so a convolution consumer has no such register. */
    uint32_t src_line_elems;
    /* Elements per OUTPUT channel group — the destination surface stride `0x607C`
     * carries. 0 takes round4(ow*oh), which is what the vendor's own pooling programs
     * carry, so every program that does not set it is byte-identical.
     *
     * A caller writing into a buffer somebody else owns sets it: a shared concatenation
     * buffer's group stride is the plane EXACTLY, because a direct convolution's output
     * surface stride is `ow*oh` and the other operands are convolutions. The part
     * honours it at any value at or above the plane — including values that are neither
     * a multiple of four atoms nor of a 64-byte line, and odd ones: at a 5x3 plane
     * 240 bytes and 272 bytes both land every atom, while 224 (one atom UNDER the plane)
     * loses one, which is the control that says the register is live.
     * [HW sweep, H96 MAX M9, tests/rk3576_pool_probe.c dst]
     *
     * There is no destination LINE stride to go with it — `0x6084` is inert and the rows
     * are always the programmed output width — so a slice may be a plane inside a deeper
     * buffer but never a plane inside wider rows. */
    uint32_t dst_surf_elems;
    uint64_t *tasks;             /* at least RK3576_POOL_TASK_OPS words            */
    uint32_t task_count;         /* words written                                  */
} pool_params_rk3576_t;

#define RK3576_POOL_TASK_OPS 40

int gen_pool_rk3576(pool_params_rk3576_t *p);

/* ---- The DPU LUT: loading the two tables --------------------------------------
 *
 * A nonlinear activation on this part is TWO programs. The first is DPU + DPU_RDMA
 * only — no CNA, no CORE — and its work is loading the table rather than computing:
 * it drives a 1x1x16 dummy cube and bursts 1026 entries through a two-register
 * window. The second is an ordinary convolution with the LUT enabled, and that one
 * computes. This emits the first.
 *
 * The window: `0x4100` (LUT_ACCESS_CFG) selects a table and the start offset, and
 * every subsequent write to `0x4104` (LUT_ACCESS_DATA) stores one entry and advances.
 * `0x00020000` selects the low table, `0x00030000` the high one, 513 entries each —
 * 512 intervals with both endpoints, the shape a linear-interpolating LUT wants.
 *
 * The entries are the function's output; the vendor compiler writes Q15 of it and
 * pairs that with an OUT_CVT that carries Q15 to the op's own quantization.
 *
 * `scratch_dma` is where the dummy cube's 16 bytes land. The vendor stores zero,
 * which is a real IOVA on this stack — pass a BO of your own rather than inheriting
 * whatever holds IOVA 0.
 * [Manufactured capture, RKNN-Toolkit2 for rk3576] */
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
 * The LUT is the EW stage, between BN and the OUT_CVT, so a convolution that carries
 * this descriptor computes
 *
 *     out = OUT_CVT( LUT( (acc + A[oc] + B[oc]*sum(x)) * C[oc] ) )
 *
 * with the BN stage bypassed. Attach it to `conv_params_t.lut` and run
 * gen_lut_load_rk3576()'s program as an EARLIER TASK OF THE SAME JOB: the tables live
 * in the LUT RAM, and a separate submit can take a runtime-PM cycle in between.
 *
 * THE INPUT MAP IS `index = (value - le_start) / 2^sel`. The `0x00020000` table covers
 * [le_start, 0] and the `0x00030000` one [0, lo_end), 512 intervals each, and the
 * hardware INTERPOLATES LINEARLY between entries at the full resolution of the step.
 * The domain is HALF-OPEN AT THE TOP: `value == lo_end` takes `clamp_hi`, while
 * `value == le_start` is in domain and reads the low table's entry 0. A value below
 * `le_start` takes `clamp_lo`.
 *
 * The vendor never moves this window — it scales the value into a fixed one with the
 * BS stage's per-channel C instead, which is why every activation it compiles emits a
 * register-identical program with different table CONTENTS. All three registers are
 * live and this descriptor drives them.
 * [HW sweep, H96 MAX M9, tests/rk3576_lut_probe.c — four spans, 8192 samples each,
 *  bit-explained to at most one output count]
 *
 * A TRAP THAT LOOKS LIKE THE ANSWER: every vendor activation also carries BN_CFG
 * `0x4060` = 0x20, the BN stage ACTIVE. Transcribing it pins the LUT at the table join
 * for every input across twenty-two binades while the readout still tracks the table,
 * so it reads as a working LUT with no input map in any register. This emitter leaves
 * BN_CFG at 0x903 — bypassed — and the map appears. */
typedef struct lut_rk3576 {
    int32_t le_start;   /* the datapath value at the low table's entry 0            */
    int32_t lo_end;     /* one past the high table's domain; >= it takes clamp_hi   */
    uint8_t sel;        /* the index step is 2^sel                                  */
    int16_t clamp_lo;   /* what a value below le_start reads                        */
    int16_t clamp_hi;   /* what a value at or above lo_end reads                    */
} lut_rk3576_t;

/* The activations this part's table builder covers. Each is the function applied to
 * the REAL input value; the builder owns the quantization on both sides.
 *
 * `exp`, `gelu` and `sqrt` are deliberately absent: no vendor capture emits a 513-entry
 * table for any of them, so they lower some other way and a table for them here would
 * be an invention rather than a transcription. */
enum {
    ROCKET_RK3576_ACT_SIGMOID = 0,   /* 1/(1+e^-x)                                  */
    ROCKET_RK3576_ACT_TANH,
    ROCKET_RK3576_ACT_SWISH,         /* x*sigmoid(x)  (SiLU)                        */
    ROCKET_RK3576_ACT_HARDSWISH,     /* x*relu6(x+3)/6                              */
    ROCKET_RK3576_ACT_HARDSIGMOID,   /* relu6(x+3)/6                                */
    ROCKET_RK3576_ACT_ELU,           /* x >= 0 ? x : e^x - 1                        */
    ROCKET_RK3576_ACT_KINDS
};

/* The name, for a log line or a gate's table. NULL for an unknown kind. */
const char *rocket_rk3576_act_name(int kind);

/* Build the two tables and the window for `kind` over a datapath whose value `v` means
 * the real number `v * value_scale`, with the entries encoding the op's output in units
 * of `entry_scale` — that is, entry = f(v*value_scale) / entry_scale.
 *
 * `le` and `lo` are RK3576_LUT_ENTRIES each and `w` receives the window. `sel` picks the
 * step; the window is then the symmetric [-512*2^sel, 512*2^sel). The clamps are set to
 * the tables' own endpoints, which is the vendor's convention and what makes the
 * saturating tails of sigmoid/tanh exact outside the table.
 *
 * Returns 0, or <0 for an unknown kind or a null argument. Entries saturate at int16
 * rather than wrapping, so a caller that picked `entry_scale` too small gets a clamped
 * curve and not a folded one. */
int rocket_rk3576_lut_build(int kind, double value_scale, double entry_scale,
                            unsigned sel, int16_t *le, int16_t *lo, lut_rk3576_t *w);

/* The fp32 conv scale -> the OUT_CVT pair the emitter programs: `mul` is the uint16
 * multiplier and `shift` the REGISTER value (already pre-decremented). Exposed because
 * a per-channel requant has to size its C multipliers against the gain the emitter will
 * actually program, and re-deriving it independently is how the two drift apart. */
void rocket_rk3576_requant_params(float conv_scale, unsigned *mul, unsigned *shift);

/*
 * The PER-OUTPUT-CHANNEL form: `c_term[oc]` is that channel's C multiplier, and
 * `multiplier` is what a NULL `c_term` falls back to. This is the field a per-axis
 * weight quantization rides on.
 *
 * THE BS STAGE ADDS A AND THEN MULTIPLIES BY C: the surface is
 * `(acc + A[oc] + B[oc]*sum(x)) * C[oc]`, so a bias quantized in the accumulator domain
 * rides the per-channel gain for free and must NOT be pre-divided by it. The two
 * orders are the same buffer and different arithmetic, and the wrong one produces a
 * bias scaled by the wrong channel gain — a plausible surface, not a fault. Read off
 * the part at C=1 against C=2 with a known accumulator and a known A: `(acc + A)*C`
 * explains 32 of 32 channels and `acc*C + A` explains only the 16 where C is 1.
 *
 * THAT PRODUCT IS INT32 AND SATURATES. Walking it across 2^31 at a fixed accumulator,
 * every inexact cell implies the same ceiling — 2.147e9 to 2.158e9 against 2^31 =
 * 2.1475e9 — and none of them wraps. So a planner sizing C has a bound to stay inside
 * rather than a cliff: `|(acc + A) * C| <= INT32_MAX`.
 *
 * WHAT C CAN EXPRESS. The OUT_CVT then applies ONE `(v*MUL)>>SHIFT` to every channel,
 * so the composite per-channel gain is `C[oc]*MUL/2^SHIFT`. A per-channel SCALE is
 * therefore expressible and a per-channel SHIFT is not — TFLite per-axis requant
 * carries both, so this reproduces a per-axis model closely and not bit-exactly. Two
 * things bound the fidelity: C is an integer, so a channel's gain resolution is
 * 0.5/C[oc], and the int32 clamp caps C[oc] at `INT32_MAX / max|acc + A|`, which falls
 * as the layer's fan-in grows.
 *
 * No entry may be 0: a zero C gates the BS stage off for that channel's whole 8-channel
 * group and the DPU writes a full, correctly sized, entirely EMPTY surface with nothing
 * to fault on. Rejected here rather than submitted.
 * [HW sweep, H96 MAX M9, tests/rk3576_coeff_c.c]
 */
int rocket_rk3576_pack_coeff_perc(void *dst, size_t dst_bytes, const int32_t *bias,
                                  unsigned oc, const int16_t *b_term,
                                  const int16_t *c_term, int16_t multiplier);

/*
 * THE DEPTHWISE PATH READS A DIFFERENT GROUP, and this is the one to hand it. Same 8
 * output channels, but 48 bytes and no B field:
 *
 *     A[oc]  int32  at (oc%8)*4        per-channel bias term
 *     C[oc]  int16  at 32 + (oc%8)*2   per-channel multiplier
 *
 * again with `oc` in group `oc/8`. Give a depthwise conv the 64-byte group above and
 * the two indices drift apart — A strides 4 bytes per channel and C strides 2, so
 * past the first eight channels each reads out of a different group. Most channels
 * then multiply a bias by a zero and never reach DDR, a few square their own bias,
 * and one group's C multipliers are read as the next group's biases. The surface it
 * writes is correctly sized and mostly empty, with no fault to catch it — the same
 * signature a wrong geometry register gives. [HW sweep, H96 MAX M9]
 *
 * There is nowhere in this group to put a weight zero point, so there is no
 * asymmetric form. It does not need one: this part's int8 depthwise cube gives every
 * (channel, tap) two live bytes and the datapath ADDS them, so `w - w_zp` is carried
 * exactly in the CUBE over [-256, 254].
 *
 * C is per output channel here exactly as it is in the 64-byte group, so a per-axis
 * weight quantization rides on this path too: rocket_rk3576_pack_coeff_dw_perc() takes
 * the ramp, and the two scalar entries are it with `c_term = NULL`. C must never be
 * left at 0 — a zero multiplier gates the whole 8-channel group's BS stage and the DPU
 * writes a full but entirely empty surface with no fault.
 *
 * Size the buffer with rocket_rk3576_coeff_bytes_dw() — it is SMALLER than the direct
 * one for the same channel count, and the emitter points BS_BASE_ADDR1 at the zeroed
 * tail group this sizing includes.
 */
size_t rocket_rk3576_coeff_bytes_dw(unsigned oc);
int rocket_rk3576_pack_coeff_dw(void *dst, size_t dst_bytes, const int32_t *bias,
                                unsigned oc);
int rocket_rk3576_pack_coeff_dw_prec(void *dst, size_t dst_bytes, const int32_t *bias,
                                     unsigned oc, unsigned prec);
int rocket_rk3576_pack_coeff_dw_perc(void *dst, size_t dst_bytes, const int32_t *bias,
                                     unsigned oc, const int16_t *c_term,
                                     int16_t multiplier);

/* ---- The DPU's elementwise stage: one cube in, requantized ---------------------
 *
 * On this part the vendor's elementwise op is a program of its own: DPU + DPU_RDMA
 * only, no CNA and no CORE, the same shape as the LUT table load, with
 * PC_OPERATION_ENABLE at 0x18. What it computes, MEASURED on silicon:
 *
 *     out = sat8( ((ew + ew_offset) * ew_scale >> ew_shift) * out_scale >> out_shift )
 *
 * ONE operand — an affine requantization of a cube, in the NC1HWC2 layout the conv
 * path already packs. Bit-exact over channel counts 16-320 and planes 1x1 to 28x28
 * (`tests/rk3576_add_probe.c gate`, 10 shapes). The final shift ROUNDS HALF TO EVEN,
 * the same rule the direct path's OUT_CVT uses.
 *
 * A SECOND OPERAND IS NOT REACHABLE through this register set, and the search is
 * EXHAUSTIVE rather than an untried direction — six sweeps, set out in
 * tests/rk3576_add_probe.c: every register the program leaves at zero CROSSED with the
 * whole output gain ladder (17 x 32; only 0x5024 does anything, and it injects a
 * constant — it is the DPU shift word, a live operand), every register the program
 * never writes appended to it (29 candidates; the register file is not cleared between
 * jobs here, so this is not an empty set), the main DMA feed at every gain from 2^14
 * down to 2^-31, two joint mode grids, destination accumulation with 0x40C0 swept over
 * 29 values, and a delta probe showing the program reads exactly the addressed cube and
 * nothing within two cubes either side of it.
 *
 * SO A RESIDUAL ADD IS LOWERED ONTO THE CONVOLUTION DATAPATH, where it computes
 * bit-exactly: concatenate the two operands along channels and convolve with a 1x1
 * kernel of two diagonal blocks. That form carries different per-operand scales (the
 * ratio rides in the weights) and both zero points (the second's in the bias), neither
 * of which this stage can express, and it FUSES into a producing convolution — see
 * tests/rk3576_residual_add.c. This entry stays for what it is: a cheap standalone
 * requantization of one cube.
 *
 * TRAPS, both of which cost a round of the probe:
 *   - Every base the program leaves at zero READS IOVA 0, which is a real buffer on
 *     this stack (per-fd IOVA starts at zero). Allocate a guard BO first or the first
 *     operand allocated IS what those bases read.
 *   - DPU 0x40D0 must be the captured 0x0040FFFF verbatim. Reading it as a clamp pair
 *     left exactly half of every 16-channel group unwritten, silently. The clamps are
 *     OUT_CLAMP_MIN/MAX (0x40A4/0x40A8), and the no-clamp pair is INT32_MIN/MAX.
 *   - OUT_CVT_SCALE is a SIGNED 16-bit field: 32768 flips the output's sign.
 *   - `Add(Conv(x), x)` IS NOT A CAPTURE OF THIS. The vendor compiler folds an identity
 *     skip into the convolution's own kernel — the centre tap of the diagonal — so six
 *     such ONNX graphs compiled to a program indistinguishable from a plain conv, and
 *     two convs into an add fold as well. A capture needs an operand the compiler
 *     cannot reach: a graph input, a constant tensor, or a real bottleneck whose skip
 *     crosses three convolutions.
 *
 * `mode` selects the vendor's ADD or MUL word; the two differ in four registers. Only
 * ADD's arithmetic is gated.
 * [Manufactured capture + HW sweep, H96 MAX M9, 2026-07-31] */
enum {
    ROCKET_RK3576_EW_ADD = 0,
    ROCKET_RK3576_EW_MUL          /* emitted, not gated */
};

typedef struct {
    uint16_t w, h, c;            /* the plane and channel count; c a multiple of 16 */
    uint8_t  mode;               /* ROCKET_RK3576_EW_*                             */
    uint32_t src_dma;            /* programmed, and MEASURED NOT TO BE READ        */
    uint32_t ew_dma;             /* the operand cube, NC1HWC2                      */
    uint32_t dst_dma;            /* the output surface                             */
    /* Elements per channel group. 0 takes w*h, which is both what the vendor's own
     * programs carry and what a convolution's output surface has, so a cube-linked
     * requantization needs no copy at either end. There is ONE source-side stride
     * register on this part; the output has its own. */
    uint32_t surf_elems, dst_surf_elems;
    int32_t  ew_offset;          /* added to the operand BEFORE its scale          */
    uint16_t ew_scale;
    uint8_t  ew_shift;
    int32_t  out_offset;         /* OUT_CVT_OFFSET; emitted, not gated             */
    uint16_t out_scale;          /* SIGNED 16-bit in the hardware                  */
    uint8_t  out_shift;
    /* OUT_CLAMP_MIN/MAX, applied before the int8 saturation — where a fused ReLU6's
     * clamp goes. The capture's no-clamp pair is INT32_MIN/INT32_MAX. */
    int32_t  clamp_lo, clamp_hi;
    uint64_t *tasks;             /* at least RK3576_EW_TASK_OPS words              */
    uint32_t task_count;
} ew_params_rk3576_t;

#define RK3576_EW_TASK_OPS 96    /* the emitted program is 93: the capture's 89 plus
                                  * the four-word PC trailer it stores apart        */

int gen_ew_int8_rk3576(ew_params_rk3576_t *p);

/* Split one requantization gain across the two converters.
 *
 * The overall gain is (ew_scale >> ew_shift) * out_scale / 2^out_shift, so the two
 * pairs are redundant and together reach a finer grid than either alone. Fills all
 * four and returns the relative error scaled by 1e9 (0 is exact), or <0 if no
 * representable pair reaches the gain. Pure; nothing here touches the hardware. */
int64_t rocket_rk3576_ew_params(double gain,
                                uint16_t *ew_scale, uint8_t *ew_shift,
                                uint16_t *out_scale, uint8_t *out_shift);

#ifdef __cplusplus
}
#endif
#endif /* NPU_REGCMD_RK3576_H */
