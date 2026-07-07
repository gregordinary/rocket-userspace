// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef NPU_MATMUL_H
#define NPU_MATMUL_H

#include <stdint.h>   /* uintN_t used in matmul_params_t (public API) */
#include "npu_dpu.h"   /* lut_epilogue_t (conv->activation fusion, conv_params_t.act) */

/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
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
 */


#ifdef __cplusplus
extern "C" {
#endif
/* ============================================================================
 * SECTION — matmul_params_t (regcmd generator parameters)
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

  /* K-ACCUMULATION (opt-in). When accumulate=1 the generator configures the DPU
   * eltwise-add path: the conv result is added to the tile already living at
   * add_dma (the running partial sum, normally == output_dma) before write-back,
   * so K-partials accumulate in NPU memory and the caller reads the tile ONCE
   * instead of nKt times. accumulate=0 => the validated plain-conv path, byte
   * for byte unchanged. add_dma is the NPU IOVA of the partial to add in. */
  uint8_t   accumulate;
  uint32_t  add_dma;

  /* EW operation for the accumulate path (only consulted when accumulate=1).
   * 0 (default) => eltwise-ADD: out = conv + add_dma  (the K-accum path).
   * 1           => eltwise-MUL: out = conv * add_dma  (EW_OP_TYPE=1). With an
   *                identity conv main (so conv == feature) this is a fully-on-NPU
   *                two-tensor multiply — the building block of on-NPU HardSwish/
   *                SiLU (x * gate(x)). HW-validated bit-exact (tests/ew_mul_rocket.c).
   * The ROCKET_EW_CFG env still overrides the assembled DPU_EW_CFG word for sweeps. */
  uint8_t   ew_mul;

  /* EXTENDED EW ALU op (only consulted when accumulate=1; takes precedence over ew_mul
   * when >0). The DPU EW unit's ALU (the BS/BN/EW == NVDLA SDP X1/X2/Y) has an algo field
   * EW_ALU_ALGO at DPU_EW_CFG bits [17:16]: 0=MAX, 1=MIN, 2=SUM(add). So beyond add/mul the
   * SAME conv-main EW datapath computes elementwise max/min of (conv_result, add_dma operand):
   *   0 => fall back to ew_mul (add/mul, the original path)
   *   2 => eltwise-MAX: out = max(conv, add_dma)   (EW_CFG 0x108002C0, algo 0)
   *   3 => eltwise-MIN: out = min(conv, add_dma)   (EW_CFG 0x108102C0, algo 1)
   * HW-validated (tests/ew_minmax_rocket.c). Covers TFLite/ONNX Maximum/Minimum and (with a
   * constant operand) Clip/Relu. */
  uint8_t   ew_op;

  /* CBUF OPERAND REUSE (opt-in). The CNA loads the weight tile + input
   * feature tile from DRAM into the on-chip CBUF for each task. When the PREVIOUS
   * task on the SAME core (i.e. earlier in the same submit batch/job) already
   * loaded the SAME operand tile into the SAME CBUF banks, these bits tell the CNA
   * to read it from CBUF instead of re-fetching from DRAM (CNA_CBUF_CON0 bits
   * 13/12). The CALLER is responsible for only setting a bit when that precondition
   * holds (identical operand AND identical bank split = identical Mtile/Ntile/Ktile
   * vs the previous task, and not first-in-batch) — setting it otherwise reads
   * STALE CBUF data -> silent garbage. Default 0 => byte-identical regcmd, every
   * tile re-fetches both operands. weight_reuse skips the weight fetch; data_reuse
   * skips the input feature fetch; they are independent bits. */
  uint8_t   weight_reuse;
  uint8_t   data_reuse;

  /* OUT: number of regcmd (NPUOP) words written to tasks[] by the generator.
   * The count varies by datatype and shape (the DPU_RDMA block size differs), so
   * callers must submit this value, not a hard-coded constant. */
  uint32_t  task_count;
} matmul_params_t;

/* ============================================================================
 * SECTION — conv_params_t (CONV_2D generator parameters)
 * ==========================================================================*/

/* CONV_2D params (general KxK / stride / pad / dilation fp16 convolution).
 *
 * Computes  out[OC, OH, OW] = sum_{ic,kh,kw} W[OC,IC,KH,KW] *
 *           in[IC, OH*stride_y + kh*dil_y - pad_top,
 *                  OW*stride_x + kw*dil_x - pad_left]
 * (out-of-range input reads 0 — that is what padding means). The host pre-scatters
 * the input feature into the NC1HWC2 cube (feature_data, C2=8 fp16) and the weights
 * into the conv weight cube (weight_conv_fp16, the oc1/ic1/kh/kw/oc2/ic2 reorder);
 * the generator programs the CNA conv engine to do the sliding-window MAC natively
 * (NO im2col). One generator, one NPU task (no host-side spatial/OC tiling — that
 * is rocket_conv2d_fp16's job).
 *
 * This is the convolution generalization of gen_matmul_fp16: a 1x1, stride-1,
 * pad-0, dilation-1, single-spatial-column conv IS the matmul (verify: at
 * KH=KW=1 weight_conv_fp16 == weight_fp16). The matmul generators are left
 * untouched; gen_conv2d_* are self-contained siblings with their own task emitter
 * (gen_conv2d_task) that additionally programs the CONV_CON3 dilation fields and
 * (depthwise) the CORE_MISC_CFG DW_EN / conv_mode=3 grouping. */
typedef struct {
  uint16_t ic;   /* input channels  (require %32==0 for the direct path)        */
  uint16_t ih;   /* input height                                                */
  uint16_t iw;   /* input width                                                 */
  uint16_t oc;   /* output channels (require %16==0 for the direct path)        */
  uint16_t oh;   /* output height (caller-computed; see rocket_conv_out_dim)    */
  uint16_t ow;   /* output width                                                */
  uint16_t kh;   /* kernel height                                               */
  uint16_t kw;   /* kernel width                                                */
  uint8_t  stride_y;
  uint8_t  stride_x;
  uint8_t  dil_y;   /* dilation rate (1 = none); emitted as rate-1 in CONV_CON3 */
  uint8_t  dil_x;
  uint8_t  pad_top;
  uint8_t  pad_left;

  uint32_t input_dma;     /* NPU IOVA of the scattered feature cube  */
  uint32_t weights_dma;   /* NPU IOVA of the scattered weight cube   */
  uint32_t output_dma;    /* NPU IOVA of the output cube             */

  uint64_t *tasks;        /* OUT: regcmd op stream                   */
  uint8_t   fp32tofp16;   /* 1 = narrow output to fp16 (C2=8 cube); 0 = fp32 out */
  /* Depthwise channel group (the weight cube's innermost channel atom). 0 ->
   * default 64 (Mesa's int8 group; the fp16 value is a HW-sweep item). Used only
   * by gen_conv2d_dw_fp16; ignored by the direct path. The host weight scatter
   * (weight_conv_dw_fp16) MUST use the same value. */
  uint8_t   dw_group;
  /* On-chip int8 requant (the int8-OUT conv path). int8_out=0 (default) keeps the
   * validated int32-raw datapath (qd_en=0, size_e=7/surf*8, int32 output, host
   * requant). int8_out=1 switches to Mesa's int8-output writer: QD_EN=1, int8 out
   * (DATA_FORMAT=0), size_e=3/surf*4, per-OC int32 bias add (BS ALU), and the
   * OUT_CVT requant computed from the quant scales below — bit-exact to the TFLite
   * int8 kernel, no int32 readback. Currently wired for the DEPTHWISE branch (the
   * Teflon-cracked path); direct int8-out is a follow-on A/B. */
  uint8_t   int8_out;
  float     in_scale;            /* input  quant scale (per-tensor) */
  float     w_scale;             /* weight quant scale (per-tensor; Teflon forces this) */
  float     out_scale;           /* output quant scale (per-tensor) */
  int32_t   input_zero_point;    /* CNA_PAD_CON1 = (input_zp & 0xff) - 0x80 (border pad) */
  int32_t   output_zero_point;   /* OUT_CVT_OFFSET = output_zero_point - 0x80 */
  int32_t   weight_zero_point;   /* DPU_BS_OW_OP = 0x80 - weight_zero_point */
  uint32_t  bias_dma;            /* IOVA of the int32 per-OC bias cube (BRDMA reads it) */
  /* conv->activation fusion (fp16 path). NULL (default) = plain conv, byte-identical
   * regcmd. Non-NULL fuses f(x) into the conv's DPU epilogue (BN-mul -> EW LUT ->
   * affine OUT_CVT). REQUIRES the fp16-out writer (set fp32tofp16=1) — the LUT result
   * is fp16. Build with rocket_lut_epilogue_build. SMOOTH single-pass kinds only
   * (SiLU/tanh/GELU); see lut_epilogue_t. */
  const lut_epilogue_t *act;
  uint32_t  task_count;   /* OUT: number of NPUOP words written      */
} conv_params_t;

/* ============================================================================
 * SECTION — Matmul regcmd generators (per datatype)
 * ==========================================================================*/

int gen_matmul_fp16(matmul_params_t *params);
int gen_matmul_int8(matmul_params_t *params);
int gen_matmul_int4(matmul_params_t *params);
int gen_matmul_int16(matmul_params_t *params);
int gen_matmul_bf16(matmul_params_t *params);   /* precision 3 */
int gen_matmul_tf32(matmul_params_t *params);   /* precision 7; first 4-byte input */

/* ============================================================================
 * SECTION — CONV_2D regcmd generators (fp16 / int8)
 * ==========================================================================*/

/* General fp16 CONV_2D regcmd generators (self-contained siblings of the matmul
 * path; the tuned matmul generators are untouched). _dw is the depthwise variant
 * (CONV_MODE=3 + DW_EN, one input channel per output channel). Return 0, or <0 if
 * the weight/feature tile overflows the CBUF (caller must tile — rocket_conv2d_fp16). */
int gen_conv2d_fp16(conv_params_t *params);
int gen_conv2d_dw_fp16(conv_params_t *params);

/* Native int8 CONV_2D regcmd generators (siblings of gen_conv2d_fp16; the fp16 path
 * is untouched). int8 features x int8 weights -> int32 accumulate on the NPU (host
 * requant). Host packing: feature cube C2=16, weight cube weight_conv_int8 (oc-group
 * 32 / ic-group 32), int32 output cube C2=4. _dw is depthwise (UNPROVEN — DW output
 * geometry + group G are HW-sweep items; ROCKET_CONV_DW_* knobs). Return 0, or <0 if
 * the weight/feature tile overflows the CBUF (caller must tile). */
int gen_conv2d_int8(conv_params_t *params);
int gen_conv2d_dw_int8(conv_params_t *params);

/* ============================================================================
 * SECTION — Host tile-layout index helpers (weight / feature scatter)
 * ==========================================================================*/

/* Conv weight element index (0-based) into the native conv weight cube. The
 * Mesa-confirmed reorder is the loop nest oc1, ic1, kh, kw, oc2, ic2 (outer->inner)
 * with N-group(oc) 16 and K-group(ic) 32 for fp16 — i.e. the matmul weight_fp16
 * tile generalized so the kernel spatial (kh,kw) sit between the ic1 and oc2
 * groups. OC=output channels, IC=input channels; oc/ic/kh/kw are 1-based. */
int weight_conv_fp16(int OC, int IC, int KH, int KW, int oc, int ic, int kh, int kw);

/* Depthwise conv weight index (0-based). Mesa's depthwise reorder (rkt_coefs.c with
 * output_channels pinned to 1 and input_channel_groups = 2*WEIGHT_ATOMIC_SIZE): the
 * nest collapses to ic1, kh, kw, ic2 with channel group G (Mesa int8 G=64; fp16
 * G=32, HW-confirmed — half the feature atom, same 4× ratio). C = depthwise channels (== OC == IC);
 * c/kh/kw are 1-based. Cube shape (C/G, KH, KW, G). */
int weight_conv_dw_fp16(int C, int KH, int KW, int G, int c, int kh, int kw);

/* int8 conv weight index (0-based). int8 generalization of weight_conv_fp16 with the
 * int8 N-group(oc) 32 (vs fp16's 16) and K-group(ic) 32 — i.e. weight_int8's
 * (N/32,K/32,32,32) tile with (kh,kw) between the ic1 and oc2 groups. At KH=KW=1 it
 * collapses to weight_int8. OC=output channels, IC=input channels; oc/ic/kh/kw 1-based. */
int weight_conv_int8(int OC, int IC, int KH, int KW, int oc, int ic, int kh, int kw);

/* int8 depthwise conv weight index (0-based). Same (C/G,KH,KW,G) element math as
 * weight_conv_dw_fp16; only the group differs (Mesa int8 G=64 vs fp16 G=32, the
 * latter HW-confirmed; int8 G is a HW-sweep item). C=depthwise channels; c/kh/kw 1-based. */
int weight_conv_dw_int8(int C, int KH, int KW, int G, int c, int kh, int kw);

int feature_data(int C, int H, int W, int C2, int c, int h, int w);
int weight_fp16(int C, int k, int c);
int weight_int8(int C, int k, int c);
/* int4 weight NIBBLE index (0-based): byte = idx/2, nibble = idx&1.
 * Nibble-packed, mirroring weight_int8 (2 int4 per byte). */
int weight_int4(int C, int k, int c);
/* int16 weight element index (0-based). Same layout as weight_fp16
 * (int16 is 2 bytes like fp16): (N/16, K/32, 16, 32). */
int weight_int16(int C, int k, int c);
/* tf32 weight element index (0-based). First 4-byte-input matmul. Layout
 * (N/16, K/16, 16, 16): a fixed 1024-byte HW tile, but for 4-byte elements the
 * K-group is 16 (half of fp16's 32) while the N-group stays 16. C = K (contraction
 * count), k = N-index (kernel), c = K-index. */
int weight_tf32(int C, int k, int c);


#ifdef __cplusplus
}
#endif
#endif // NPU_MATMUL_H
