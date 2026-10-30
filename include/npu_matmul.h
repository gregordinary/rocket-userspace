// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef NPU_MATMUL_H
#define NPU_MATMUL_H

#include <stdint.h>   /* uintN_t used in matmul_params_t (public API) */
#include "npu_dpu.h"   /* lut_epilogue_t (conv->activation fusion, conv_params_t.act) */

/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * prose-lint: off
 *
 * This file is part of rocket-userspace.
 *
 * rocket-userspace is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rocket-userspace is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rocket-userspace.  If not, see <https://www.gnu.org/licenses/>.
 *
 * prose-lint: on
 */

/*
 * Declares the matmul and CONV_2D register-program generators, their parameter structures,
 * and the host index helpers that scatter weights and features into the cube layout.
 */

#ifdef __cplusplus
extern "C" {
#endif
/* ============================================================================
 * `matmul_params_t`: the matmul generator parameters
 * ==========================================================================*/

typedef struct {
  uint16_t  m;
  uint16_t  k;
  uint16_t  n;

  uint32_t  input_dma;
  uint32_t  weights_dma;
  uint32_t  output_dma;

  uint64_t  *tasks;

  uint8_t   fp32tofp16;

  /* Selects K-accumulation, which is opt-in. With `accumulate=1`, the generator configures
   * the DPU eltwise-add path. The conv result is added to the tile already at `add_dma`, the
   * running partial sum, before write-back. `add_dma` is normally equal to `output_dma`. The
   * K-partials then accumulate in NPU memory, and the caller reads the tile once instead of
   * `nKt` times.
   *
   * `accumulate=0` selects the validated plain-conv path, byte for byte unchanged. `add_dma`
   * is the NPU IOVA of the partial to add in. */
  uint8_t   accumulate;
  uint32_t  add_dma;

  /* Selects the EW operation of the accumulate path. The generator reads it only with
   * `accumulate=1`:
   *
   *     0 (default) => eltwise-ADD: out = conv + add_dma  (the K-accum path)
   *     1           => eltwise-MUL: out = conv * add_dma  (EW_OP_TYPE=1)
   *
   * With an identity conv main feed (so conv == feature), MUL is a fully-on-NPU two-tensor
   * multiply. It is the building block of on-NPU HardSwish and SiLU (x * gate(x)), and it is
   * bit-exact [HW sweep] (`tests/ew_mul_rocket.c`). The `ROCKET_EW_CFG` environment variable
   * overrides the assembled `DPU_EW_CFG` word, for sweeps. */
  uint8_t   ew_mul;

  /* Selects an extended EW ALU operation. The generator reads it only with `accumulate=1`,
   * and a value >0 takes precedence over `ew_mul`. The DPU EW unit's ALU (BS, BN and EW are
   * NVDLA SDP X1, X2 and Y) has an algorithm field, `EW_ALU_ALGO` at `DPU_EW_CFG` bits
   * [17:16]: 0=MAX, 1=MIN, 2=SUM(add). Beyond add and mul, the same conv-main EW datapath
   * therefore computes the elementwise max or min of the conv result and the `add_dma`
   * operand:
   *
   *     0 => fall back to ew_mul (add/mul, the original path)
   *     2 => eltwise-MAX: out = max(conv, add_dma)   (EW_CFG 0x108002C0, algo 0)
   *     3 => eltwise-MIN: out = min(conv, add_dma)   (EW_CFG 0x108102C0, algo 1)
   *
   * It covers TFLite/ONNX Maximum and Minimum, and Clip and Relu with a constant operand
   * [HW sweep] (`tests/ew_minmax_rocket.c`). */
  uint8_t   ew_op;

  /* Selects CBUF operand reuse, which is opt-in. For each task, the CNA loads the weight
   * tile and the input feature tile from DRAM into the on-chip CBUF. The previous task on the
   * same core, earlier in the same submit, can already have loaded the same operand tile into
   * the same CBUF banks. These bits then tell the CNA to read it from CBUF instead of
   * re-fetching it from DRAM (`CNA_CBUF_CON0` bits 13/12). `weight_reuse` skips the weight
   * fetch and `data_reuse` skips the input feature fetch, and the two bits are independent.
   *
   * The default 0 gives byte-identical regcmd, and every tile re-fetches both operands. A bit's
   * precondition must hold before the caller sets it. The operand must be identical,
   * and so must the bank split, which means the same Mtile, Ntile and Ktile as the previous
   * task. The task must not be the first in the submit.
   *
   * Caution: a bit set without its precondition reads stale CBUF data, and the output is
   * silently wrong. */
  uint8_t   weight_reuse;
  uint8_t   data_reuse;

  /* Selects row-major operands, for `gen_matmul_fp16` only. The value 0 selects the cube
   * program, byte-identical.
   *
   * With `a_row_atoms` != 0, the task reads `A` as plain rows. One row of the full `A` is
   * this many 16-byte atoms (K_full*2/16). `input_dma` points at the task's first element
   * (row m0, column k0), so a task reads a `K` slice out of a wider `A`.
   *
   * With `c_row_atoms` != 0, the task writes `C` as plain rows. One row of the full `C` is
   * this many atoms (N_full*2/16 at fp16 out, N_full*4/16 at fp32). `output_dma` points at
   * (m0, n0), so a task writes an `N` tile into a wider `C` and nothing outside it.
   *
   * An accumulating task still reads its partial (`add_dma`) as a cube. The device cost
   * depends on C's row pitch: it is free to a 4 KiB row, and 1.4-1.6x at 16 KiB. The
   * generator refuses either field past its register (`a_row_atoms` 28 bits, `c_row_atoms`
   * 8192). The result is bit-exact against the cube program at every axis a tiled path
   * varies.
   *
   * Evidence: [HW sweep, RK3588] (`tests/rowmajor_tile_probe`). */
  uint32_t  a_row_atoms;
  uint32_t  c_row_atoms;

  /* OUT: the number of regcmd (NPUOP) words the generator writes to `tasks[]`. The count
   * varies by datatype and shape, because the DPU_RDMA block size differs. The caller must
   * submit this value, not a hard-coded constant. */
  uint32_t  task_count;
} matmul_params_t;

/* ============================================================================
 * `conv_params_t`: the CONV_2D generator parameters
 * ==========================================================================*/

/* Describes a general fp16 CONV_2D: KxK, with stride, pad and dilation. The generator
 * computes:
 *
 *     out[OC, OH, OW] = sum_{ic,kh,kw} W[OC,IC,KH,KW] *
 *                       in[IC, OH*stride_y + kh*dil_y - pad_top,
 *                              OW*stride_x + kw*dil_x - pad_left]
 *
 * An out-of-range input reads 0, which is what padding means. The host pre-scatters the
 * input feature into the NC1HWC2 cube (`feature_data`, C2=8 fp16). It scatters the weights
 * into the conv weight cube (`weight_conv_fp16`, the `oc1/ic1/kh/kw/oc2/ic2` reorder). The
 * generator programs the CNA conv engine to do the sliding-window MAC natively, with no
 * im2col. One generator call emits one NPU task, with no host-side spatial or OC tiling,
 * which is `rocket_conv2d_fp16`'s job.
 *
 * The CONV_2D generator is the convolution generalization of `gen_matmul_fp16`. A 1x1, stride-1, pad-0,
 * dilation-1, single-spatial-column conv is the matmul: at KH=KW=1, `weight_conv_fp16` ==
 * `weight_fp16`. The `gen_conv2d_*` generators are self-contained siblings with their own
 * task emitter (`gen_conv2d_task`). It also programs the `CONV_CON3` dilation fields and,
 * for depthwise, the `CORE_MISC_CFG` `DW_EN` and `conv_mode=3` grouping. */

/* Declares the RK3576's DPU LUT window descriptor. `npu_regcmd_rk3576.h` defines it and is
 * the only place that reads the field below. */
struct lut_rk3576;

typedef struct {
  uint16_t ic;   /* input channels  (must be %32==0 for the direct path)        */
  uint16_t ih;   /* input height                                                */
  uint16_t iw;   /* input width                                                 */
  uint16_t oc;   /* output channels (must be %16==0 for the direct path)        */
  uint16_t oh;   /* output height (caller-computed, see rocket_conv_out_dim)    */
  uint16_t ow;   /* output width                                                */
  uint16_t kh;   /* kernel height                                               */
  uint16_t kw;   /* kernel width                                                */
  uint8_t  stride_y;
  uint8_t  stride_x;
  uint8_t  dil_y;   /* dilation rate (1 = none), emitted as rate-1 in CONV_CON3 */
  uint8_t  dil_x;
  uint8_t  pad_top;
  uint8_t  pad_left;

  uint32_t input_dma;     /* NPU IOVA of the scattered feature cube  */
  uint32_t weights_dma;   /* NPU IOVA of the scattered weight cube   */
  uint32_t output_dma;    /* NPU IOVA of the output cube             */

  uint64_t *tasks;        /* OUT: regcmd op stream                   */
  uint8_t   fp32tofp16;   /* 1 = narrow output to fp16 (C2=8 cube), 0 = fp32 out */
  /* The depthwise channel group, the weight cube's innermost channel atom. The value 0
   * selects the default: 32 for `gen_conv2d_dw_fp16`, and 64, Mesa's int8 group, for
   * `gen_conv2d_dw_int8`. Only the depthwise generators read it, and the direct path ignores
   * it. The host weight scatter (`weight_conv_dw_fp16`) must use the same value. */
  uint8_t   dw_group;
  /* Selects the on-chip int8 requant, the int8-out conv path. `int8_out=0`, the default,
   * keeps the int32-raw datapath: `qd_en=0`, size_e=7/surf*8, int32 output and a host
   * requant. It is validated on the direct path (`tests/conv2d_int8_rocket.c`), and
   * unproven on the depthwise one (`gen_conv2d_dw_int8`). `int8_out=1` switches to Mesa's
   * int8-output writer, with no int32 readback. The writer programs:
   *
   * - QD_EN=1 and int8 out (DATA_FORMAT=0), at size_e=3/surf*4.
   * - A per-OC int32 bias add in the BS ALU.
   * - The OUT_CVT requant, computed from the quant scales below.
   * - CPEND (`DPU_BS_OW_OP`), which contains the weight zero point.
   *
   * The depthwise branch runs at size_e 3 / SURF_ADD x4, and the direct branch at size_e 1 /
   * SURF_ADD x2, Mesa's direct program. Each is bit-exact against a host model on the RK1
   * [HW sweep] (`tests/conv_i8out_probe.c` for the direct one, `tests/conv_dw_int8_runtime.c`
   * for the depthwise one).
   *
   * Caution: the direct program must run whole 32-kernel groups. At an OC that ends part way
   * through one, the task outlasts the watchdog and that group is unwritten. */
  uint8_t   int8_out;
  float     in_scale;            /* input  quant scale (per-tensor) */
  float     w_scale;             /* weight quant scale (per-tensor, which Teflon forces) */
  float     out_scale;           /* output quant scale (per-tensor) */
  int32_t   input_zero_point;    /* CNA_PAD_CON1 = (input_zp & 0xff) - 0x80 (border pad) */
  int32_t   output_zero_point;   /* OUT_CVT_OFFSET = output_zero_point - 0x80 */
  int32_t   weight_zero_point;   /* DPU_BS_OW_OP = 0x80 - weight_zero_point */
  uint32_t  bias_dma;            /* IOVA of the int32 per-OC bias cube (BRDMA reads it) */
  /* Selects the per-channel BS multiplier, for int8_out depthwise only. `bias_dma` then
   * addresses the per-channel coefficient cube, 64 bytes per 8 channels. The cube contains
   * int32 A, int16 B and int16 C. The BS stage multiplies by C[c] and then shifts by `bs_mul_shift`. The OUT_CVT
   * therefore applies one base gain for the job, and C applies the per-channel ratio to it.
   * See `npu_dpu_desc.bs_mul_src`. The value 0 selects per-tensor. */
  uint8_t   bs_mul_perc;
  uint8_t   bs_mul_shift;
  /* Selects the conv->activation fusion, on the fp16 path. NULL, the default, gives a plain
   * conv with byte-identical regcmd. A non-NULL value fuses f(x) into the conv's DPU
   * epilogue (BN-mul -> EW LUT -> affine OUT_CVT). The LUT result is fp16, so the fusion
   * requires the fp16-out writer, and the caller must set `fp32tofp16=1`. Build the epilogue
   * with `rocket_lut_epilogue_build`. It supports smooth single-pass kinds only (SiLU, tanh
   * and GELU), as `lut_epilogue_t` describes. */
  const lut_epilogue_t *act;
  /* The RK3576's DPU LUT, for a convolution whose EW stage applies a loaded table to its BS
   * output. NULL, the default, leaves the LUT bank zeroed and the stage bypassed, which
   * is byte-identical to a program without the field. Only the RK3576 generators read it. The
   * RK3588 path has its own `act` above, with a different field layout. A separate program
   * (`gen_lut_load_rk3576`) loads the two tables, and it must run in the same submit. */
  const struct lut_rk3576 *lut;
  /* The full-plane feature and output heights, for a task that covers only a row window of a
   * taller plane. The default 0 means the task is the whole plane, `ih` or `oh`. Only the
   * RK3576 generators (`npu_regcmd_rk3576.c`) read them, because their stream programs the
   * task window and the full plane in separate registers. The RK3588 generators never read
   * them, so their output is byte-identical. */
  uint16_t  ih_full;
  uint16_t  oh_full;
  /* The feature cube's channel-group stride, in elements. The default 0 means the plane,
   * `iw*ih_full`, which is the stride of every cube this library packs itself. A non-zero
   * value larger than the plane marks an input that another op wrote with a padded stride. A
   * depthwise producer, for example, writes round4(ow*oh) per group. The value goes to the
   * CNA's DDR group stride (0x1094) in place of the derived quantity.
   *
   * Only the RK3576 generators read it. The RK3588 generators never read it, so their output
   * is byte-identical. The generator refuses a stride smaller than the plane. */
  uint32_t  in_surf_elems;
  /* The feature plane's DDR row pitch, in elements. The default 0 means `iw`, with
   * contiguous rows, which is the pitch of every cube this library packs itself. A non-zero
   * value larger than `iw` places the iw-wide plane inside wider rows. A materializing
   * packed-image stem, for example, writes surplus columns that its caller's tensor never
   * had. The value goes to the CNA's DDR line stride (0x1090) in place of the derived
   * quantity. The fetched extent (0x1078's width half, 0x1080's two halves) and the CBUF row
   * granule count stay at `iw`.
   *
   * A caller that sets this must also set `in_surf_elems`. The channel-group jump is over the
   * pitched rows, which the derived plane does not describe. Only the RK3576
   * non-ARGB generators read it. The RK3588 generators never read it, so their output is
   * byte-identical. */
  uint32_t  in_pitch_w;
  /* Selects the CNA deconvolution mode, in which the hardware interior-dilates the input by
   * these strides. The default 0 or 1 is off and byte-identical. A value of 2, 4 or 8 turns
   * it on, and the generator refuses any other value. The caller must program `oh` and `ow`
   * as the transposed extent and pad by k-1-p. The caller must also flip the kernel,
   * transpose its channels and keep `stride_y` and `stride_x` at 1. Only the RK3588 fp16 direct generator (`gen_conv2d_fp16`)
   * reads it. */
  uint8_t   deconv_sy;
  uint8_t   deconv_sx;
  uint32_t  task_count;   /* OUT: number of NPUOP words written      */
} conv_params_t;

/* ============================================================================
 * Matmul regcmd generators, one per datatype
 * ==========================================================================*/

int gen_matmul_fp16(matmul_params_t *params);
int gen_matmul_int8(matmul_params_t *params);

/* Returns the fewest CBUF weight banks a streamed task needs. A task streams when its weight
 * cube is larger than its banks. It needs one whole 32-kernel atom, and never fewer than
 * two. With fewer, the task does not complete and its output is unwritten. An int8 task can
 * also hang at exactly this count, so `gen_matmul_int8` takes one bank more.
 * `gen_matmul_fp16` and `gen_matmul_int8` refuse a task below their floor (-3), and refuse
 * `weight_reuse` on a streamed cube.
 *
 * Evidence: [HW sweep, RK3588, 2026-10-21] (`tests/weight_stream_probe`,
 * `tests/matmul_int8_prepacked_rocket`). */
unsigned npu_stream_weight_banks(unsigned weight_bytes_per_kernel);
int gen_matmul_int4(matmul_params_t *params);
int gen_matmul_int16(matmul_params_t *params);
int gen_matmul_bf16(matmul_params_t *params);   /* precision 3 */
int gen_matmul_tf32(matmul_params_t *params);   /* precision 7, the first 4-byte input */

/* ============================================================================
 * CONV_2D regcmd generators, fp16 and int8
 * ==========================================================================*/

/* Generate a general fp16 CONV_2D register program. They are self-contained siblings of the
 * matmul path. `gen_conv2d_dw_fp16` is the depthwise variant (CONV_MODE=3 with DW_EN, one
 * input channel per output channel). Each returns 0 on success. If the weight or feature tile
 * overflows the CBUF, it returns a negative value, and the caller must then tile, as
 * `rocket_conv2d_fp16` does. */
int gen_conv2d_fp16(conv_params_t *params);
int gen_conv2d_dw_fp16(conv_params_t *params);

/* Generate a native int8 CONV_2D register program, as siblings of `gen_conv2d_fp16`. int8
 * features times int8 weights accumulate to int32 on the NPU. With `int8_out`=0, the
 * default, the host requantizes. The host packs a feature cube with C2=16 and a weight cube
 * with `weight_conv_int8` (oc-group 32, ic-group 32), and the output is an int32 cube with
 * C2=4. With `int8_out`=1 the DPU requantizes on chip, as `conv_params_t.int8_out` describes.
 *
 * `gen_conv2d_dw_int8` is the depthwise variant, and its two branches differ:
 *
 * - The int8-out branch (`int8_out`=1, `G`=64, size_e 3 / SURF_ADD x4) is bit-exact against
 *   a host model on the RK1 [HW sweep] (`tests/conv_dw_int8_runtime.c`,
 *   `tests/replay_dw_mesa.c`). `rocket_conv2d_dw_int8` and `rocket_conv2d_dw_int8_perc` run
 *   it.
 * - The int32-raw branch (`int8_out`=0) is unproven: it doubles the within-plane stride. Its
 *   output geometry and its group `G` are open for a hardware sweep, through the
 *   `ROCKET_CONV_DW_*` environment variables.
 *
 * Each generator returns 0 on success. If the weight or feature tile overflows the CBUF, it
 * returns a negative value, and the caller must then tile. */
int gen_conv2d_int8(conv_params_t *params);
int gen_conv2d_dw_int8(conv_params_t *params);

/* ============================================================================
 * Host tile-layout index helpers for the weight and feature scatter
 * ==========================================================================*/

/* Returns the conv weight element index (0-based) into the native conv weight cube. The
 * Mesa-confirmed reorder is the loop nest below, outer to inner, with N-group(oc) 16 and
 * K-group(ic) 32 for fp16:
 *
 *     oc1, ic1, kh, kw, oc2, ic2
 *
 * It is the matmul `weight_fp16` tile, generalized so that the kernel spatial (kh,kw) sits
 * between the ic1 and oc2 groups. OC is the output channels and IC the input channels, and
 * `oc`, `ic`, `kh` and `kw` are 1-based. */
int weight_conv_fp16(int OC, int IC, int KH, int KW, int oc, int ic, int kh, int kw);

/* Returns the depthwise conv weight index (0-based). It follows Mesa's depthwise reorder
 * (`rkt_coefs.c` with `output_channels` pinned to 1 and
 * `input_channel_groups = 2*WEIGHT_ATOMIC_SIZE`). The nest collapses to `ic1, kh, kw, ic2`
 * with channel group `G`. Mesa's int8 `G` is 64, and the fp16 `G` is 32 [HW sweep], half
 * the feature atom at the same 4x ratio. `C` is the depthwise channels (== OC == IC), and
 * `c`, `kh` and `kw` are 1-based. The cube shape is (C/G, KH, KW, G). */
int weight_conv_dw_fp16(int C, int KH, int KW, int G, int c, int kh, int kw);

/* Returns the int8 conv weight index (0-based). It is the int8 generalization of
 * `weight_conv_fp16`, with the int8 N-group(oc) 32 (fp16's is 16) and K-group(ic) 32. The
 * layout is `weight_int8`'s (N/32,K/32,32,32) tile with (kh,kw) between the ic1 and oc2
 * groups. At KH=KW=1 it collapses to `weight_int8`. OC is the output channels and IC the
 * input channels, and `oc`, `ic`, `kh` and `kw` are 1-based. */
int weight_conv_int8(int OC, int IC, int KH, int KW, int oc, int ic, int kh, int kw);

/* Returns the int8 depthwise conv weight index (0-based). The (C/G,KH,KW,G) element math is
 * that of `weight_conv_dw_fp16`, and only the group differs. Mesa's int8 `G` is 64 and the
 * fp16 `G` is 32, the latter confirmed on hardware [HW sweep]. The int8 `G` is open for a
 * hardware sweep. `C` is the depthwise channels, and `c`, `kh` and `kw` are 1-based. */
int weight_conv_dw_int8(int C, int KH, int KW, int G, int c, int kh, int kw);

int feature_data(int C, int H, int W, int C2, int c, int h, int w);
int weight_fp16(int C, int k, int c);
int weight_int8(int C, int k, int c);
/* Returns the int4 weight nibble index (0-based): byte = idx/2, nibble = idx&1. The weights
 * are nibble-packed, mirroring `weight_int8` (2 int4 per byte). */
int weight_int4(int C, int k, int c);
/* Returns the int16 weight element index (0-based). The layout is that of `weight_fp16`,
 * because int16 is 2 bytes like fp16: (N/16, K/32, 16, 32). */
int weight_int16(int C, int k, int c);
/* Returns the tf32 weight element index (0-based), for the first 4-byte-input matmul. The
 * layout is (N/16, K/16, 16, 16), a fixed 1024-byte hardware tile. For 4-byte elements the
 * K-group is 16, half of fp16's 32, and the N-group stays 16. `C` is `K` (the contraction
 * count), `k` is the N-index (kernel), and `c` is the K-index. */
int weight_tf32(int C, int k, int c);


#ifdef __cplusplus
}
#endif
#endif // NPU_MATMUL_H
