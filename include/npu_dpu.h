// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef NPU_DPU_H
#define NPU_DPU_H

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

#include <stdint.h>

/* ============================================================================
 * SECTION — lut_epilogue_t (conv->activation LUT fusion descriptor)
 * ==========================================================================*/

/* DPU LUT-activation epilogue (conv->activation fusion). Built host-side
 * (rocket_lut_epilogue_build, rocket_activation.c) and attached to a conv via
 * conv_params_t.act / npu_dpu_desc.lut_ep so the conv's NPU job post-processes its
 * own result with f(x) (the NVDLA SDP BN-mul -> EW LUT -> affine OUT_CVT chain),
 * with NO second NPU round-trip. The register values mirror gen_lut_activation_fp16
 * (the standalone flying-mode op); only the input source differs (the conv CACC
 * accumulator instead of a flying MRDMA stream), so the SDP epilogue is identical.
 * SMOOTH single-pass activations only (SiLU / tanh / GELU); HardSwish's exactly-flat
 * x<=-3 tail trips the NVDLA LE/LO mux, so it stays on the host / 2-pass path. */

#ifdef __cplusplus
extern "C" {
#endif
typedef struct lut_epilogue {
  int      kind;                /* ROCKET_ACTIVATION_*; host reference/oracle ONLY —
                                 * the regcmd generator ignores it (uses the table). */
  const uint16_t *lut;          /* 1026 Q-format entries: LE[0..512], LO[513..1025] */
  uint16_t bn_mul_operand;      /* fp16 bits: conv result x -> LUT index scale       */
  uint32_t bn_alu_cfg;          /* BN ALU bias word (0x80000000 == fp32 -0.0 no-op)  */
  uint32_t out_cvt_offset;      /* affine OUT_CVT signed, pre-shift offset           */
  uint16_t out_cvt_scale;       /* OUT_CVT scale (1 on the validated path)           */
  uint8_t  out_cvt_minus_exp;   /* affine decode: out = (q+offset) * 2^-minus_exp    */
  uint8_t  out_cvt_cvt_type;    /* OUT_CVT cvt_type                                  */
  uint32_t lut_le_start;        /* LE table start (Q index, e.g. 0xffffc000 = -16384)*/
  uint32_t lut_lo_end;          /* LO table end   (e.g. 0x00004000 = 16384)          */
  uint16_t le_slope_scale;      /* LE (x < table) underflow extrapolation slope      */
  uint8_t  le_slope_shift;
  uint16_t lo_slope_scale;      /* LO (x > table) overflow extrapolation slope       */
  uint8_t  lo_slope_shift;
  uint8_t  le_index_select;     /* log2(step) for LE (5 => step 32)                  */
  uint8_t  lo_index_select;     /* log2(step) for LO                                 */
} lut_epilogue_t;

/* ============================================================================
 * SECTION — npu_dpu_desc (DPU datapath register descriptor)
 * ==========================================================================*/

typedef struct npu_dpu_desc {
 uint8_t burst_len;         // 0x400C
 uint8_t conv_mode;         // 0x400C
 uint8_t output_mode;       // 0x400C
 uint8_t flying_mode;       // 0x400C
 uint8_t out_precision;     // 0x4010
 uint8_t in_precision;      // 0x4010
 uint8_t proc_precision;    // 0x4010
 uint32_t dst_base_addr;    // 0x4020
 uint32_t dst_surf_stride;  // 0x4024
 uint16_t width;            // 0x4030
 uint16_t height;           // 0x4034
 uint16_t channel;          // 0x403C
 uint8_t bs_bypass;         // 0x4040
 uint8_t bs_alu_bypass;     // 0x4040
 uint8_t bs_mul_bypass;     // 0x4040
 uint8_t bs_relu_bypass;    // 0x4040
 uint8_t od_bypass;         // 0x4050
 uint8_t size_e_2;          // 0x4050
 uint8_t size_e_1;          // 0x4050
 uint8_t size_e_0;          // 0x4050
 uint16_t channel_wdma;     // 0x4058
 uint16_t height_wdma;      // 0x405C
 uint16_t width_wdma;       // 0x405C
 uint8_t bn_relu_bypass;    // 0x4060
 uint8_t bn_mul_bypass;     // 0x4060
 uint8_t bn_alu_bypass;     // 0x4060
 uint8_t bn_bypass;         // 0x4060
 uint8_t ew_bypass;         // 0x4070
 uint8_t ew_op_bypass;      // 0x4070
 uint8_t ew_lut_bypass;     // 0x4070
 uint8_t ew_op_cvt_bypass;  // 0x4070
 uint8_t ew_relu_bypass;    // 0x4070
 uint8_t fp32tofp16_en;     // 0x4084
 uint16_t out_cvt_scale;    // 0x4084
 uint32_t surf_add;         // 0x40C0
 uint32_t bs_ow_op;         // 0x4054  DPU_BS_OW_OP.OW_OP = 0x80 - weight_zp (depthwise)
 /* K-accumulation (eltwise-add) path. ew_accumulate=0 keeps the EW + DPU_RDMA
  * blocks in their validated bypass/disabled state. When 1, DPU_EW_CFG and the
  * ERDMA group are programmed from the pre-assembled *_val words so the fp16
  * precision fields stay env-tunable (see gen_matmul_fp16). */
 uint8_t  ew_accumulate;    // 1 => add ew_base_addr into the conv output
 uint32_t ew_base_addr;     // ERDMA EW source = running partial tile (0x5038)
 uint32_t ew_src_base_addr; // MRDMA SRC_BASE = running partial, NO offset (0x5018)
 uint32_t ew_cfg_val;       // assembled DPU_EW_CFG               (0x4070)
 uint32_t erdma_cfg_val;    // assembled DPU_RDMA_ERDMA_CFG       (0x5034)
 uint32_t ew_surf_stride;   // EW surface stride (raw, <<4 on emit) (0x5040)
 uint32_t rdma_fmc_val;     // assembled DPU_RDMA_FEATURE_MODE_CFG(0x5044)
 uint32_t ew_surf_notch;    // SURF_NOTCH / EW_SURF_NOTCH          (0x504C/0x506C)
 /* DPU output-writer controls (int16 path). Default 0 = the validated
  * behaviour for fp16/int8/int4; only gen_matmul_int16 sets them, to drive the
  * int16 transposed/truncated output path. Register fields:
  *   mc_surf_out  DPU_DATA_FORMAT bit3   0=16B/pixel one surface, 1=2/4 surf serial
  *   tp_precision DPU_WDMA_SIZE_0  bit27  transpose precision 0=8bit 1=16bit
  *   size_c_wdma  DPU_WDMA_SIZE_0  b26:16 Size_c for DPU_WDMA
  *   tp_org_en    DPU_BS_OW_CFG    bit27  enable original transpose */
 uint8_t  mc_surf_out;
 uint8_t  tp_precision;
 uint16_t size_c_wdma;
 uint8_t  tp_org_en;
 /* On-chip int8 requant output writer (the int8-OUT conv path; default 0 == the
  * validated int32-raw / float writers). When the generator programs these, the
  * shared emitters wire them instead of the hardcoded 0/scale/0. Mirrors Mesa's
  * NVDLA-style SDP converter:
  *   out_cvt_offset  0x4080  signed 32-bit; = output_zero_point - 0x80
  *   out_cvt_shift   0x4088  the QNNPACK multiplier shift (shift-1)
  * (out_cvt_scale at 0x4084 already exists above; the int8-out path sets it to the
  *  QNNPACK 15-bit multiplier instead of 1.) */
 uint32_t out_cvt_offset;   // 0x4080
 uint8_t  out_cvt_shift;    // 0x4088 bits[5:0] integer requant shift (int8-out path)
 /* Float-affine OUT_CVT (the LUT path's decode, generalized to the matmul int8->float
  * dequant fold). 0x4088 bit31 = cvt_type (0 integer-requant, 1 float-affine),
  * bits[19:12] = minus_exp. Default 0 keeps the integer/byte-identical path. The
  * effective output = (acc + out_cvt_offset) * out_cvt_scale * 2^-out_cvt_minus_exp,
  * with fp32tofp16_en narrowing to fp16 — the exact semantics are the OUT_CVT HW-RE
  * question (tests/matmul_int8_dequant_rocket.c sweeps them). */
 uint8_t  out_cvt_minus_exp;  // 0x4088 bits[19:12]
 uint8_t  out_cvt_cvt_type;   // 0x4088 bit31
 uint16_t bn_mul_operand;     // 0x4068 DPU_BN_MUL_CFG[31:16]: fp16 per-tensor scale
                              // (NVDLA SDP X-mul "channel/per-pixel"); 0 + bn_mul_bypass
                              // keeps the BN stage off (byte-identical). The dequant-fold alt path uses this.
 /* BS-stage int32 bias add (per output channel), read by BRDMA. bias_en=0 keeps
  * the BS block fully bypassed (the float/int32-raw default). bias_en=1 drives the
  * Mesa DW int8 word: BS_ALU_ALGO(2)|BS_ALU_SRC(1)|BS_RELU_BYPASS|BS_MUL_BYPASS and
  * arms BRDMA to fetch the bias cube from bias_base_addr. */
 uint8_t  bias_en;          // BS ALU add active
 uint32_t bias_base_addr;   // 0x5020 RDMA_BS_BASE_ADDR (int32 bias BO IOVA)
 /* conv->activation fusion: lut_en=0 (default) keeps the validated BN/EW/OUT_CVT
  * bypass path byte-identical. When 1, gen_conv2d_task uploads lut_ep->lut and
  * programs the BN-mul index scale + EW LUT + affine OUT_CVT so the conv result is
  * post-processed by f(x) in the same job (see lut_epilogue_t). */
 uint8_t  lut_en;
 const lut_epilogue_t *lut_ep;
} npu_dpu_desc;


#ifdef __cplusplus
}
#endif
#endif // NPU_DPU_H
