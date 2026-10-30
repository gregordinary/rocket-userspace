// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef NPU_DPU_H
#define NPU_DPU_H

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

#include <stdint.h>

/* ============================================================================
 * `lut_epilogue_t`: the conv->activation LUT fusion descriptor
 * ==========================================================================*/

/* Describes the DPU LUT-activation epilogue, which fuses an activation into a convolution.
 * `rocket_lut_epilogue_build` (`rocket_activation.c`) builds it on the host. The caller
 * attaches it to a conv through `conv_params_t.act` or `npu_dpu_desc.lut_ep`. The conv's NPU
 * task then post-processes its own result with f(x), with no second NPU round trip. The chain
 * is NVDLA's SDP sequence: BN-mul -> EW LUT -> affine OUT_CVT.
 *
 * The register values mirror `gen_lut_activation_fp16`, the standalone flying-mode op. Only
 * the input source differs: the conv's CACC accumulator instead of a flying MRDMA stream. The
 * SDP epilogue is therefore identical.
 *
 * The epilogue supports smooth single-pass activations only: SiLU, tanh and GELU.
 * HardSwish's exactly flat tail at x<=-3 trips the NVDLA LE/LO mux, so HardSwish stays on the
 * host or 2-pass path. */

#ifdef __cplusplus
extern "C" {
#endif
typedef struct lut_epilogue {
  int      kind;                /* ROCKET_ACTIVATION_*, for the host reference only.
                                 * The regcmd generator ignores it and uses the table. */
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
 * `npu_dpu_desc`: the DPU datapath register descriptor
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
 /* The K-accumulation (eltwise-add) path. `ew_accumulate=0` keeps the EW and DPU_RDMA
  * blocks in their validated bypass or disabled state. When it is 1, the generator programs
  * `DPU_EW_CFG` and the ERDMA group from the pre-assembled `*_val` words. The fp16 precision
  * fields can then be set from environment variables, as in `gen_matmul_fp16`. */
 uint8_t  ew_accumulate;    // 1 => add ew_base_addr into the conv output
 uint32_t ew_base_addr;     // ERDMA EW source = running partial tile (0x5038)
 uint32_t ew_src_base_addr; // MRDMA SRC_BASE = running partial, no offset (0x5018)
 uint32_t ew_cfg_val;       // assembled DPU_EW_CFG               (0x4070)
 uint32_t erdma_cfg_val;    // assembled DPU_RDMA_ERDMA_CFG       (0x5034)
 uint32_t ew_surf_stride;   // EW surface stride (raw, <<4 on emit) (0x5040)
 uint32_t rdma_fmc_val;     // assembled DPU_RDMA_FEATURE_MODE_CFG(0x5044)
 uint32_t ew_surf_notch;    // SURF_NOTCH / EW_SURF_NOTCH          (0x504C/0x506C)
 /* DPU output-writer controls for the int16 path. The default 0 is the validated behavior
  * for fp16, int8 and int4. Only `gen_matmul_int16` sets them, to drive the int16 transposed
  * and truncated output path. The register fields:
  *
  *     mc_surf_out  DPU_DATA_FORMAT bit3   0=16B/pixel one surface, 1=2/4 surf serial
  *     tp_precision DPU_WDMA_SIZE_0  bit27  transpose precision 0=8bit 1=16bit
  *     size_c_wdma  DPU_WDMA_SIZE_0  b26:16 Size_c for DPU_WDMA
  *     tp_org_en    DPU_BS_OW_CFG    bit27  enable original transpose */
 uint8_t  mc_surf_out;
 uint8_t  tp_precision;
 uint16_t size_c_wdma;
 uint8_t  tp_org_en;
 /* The on-chip int8 requant output writer, for the int8-out conv path. The default 0 selects
  * the validated int32-raw and float writers. When the generator programs these fields, the
  * shared emitters use them instead of the hardcoded 0/scale/0. The fields mirror Mesa's
  * NVDLA-style SDP converter:
  *
  *     out_cvt_offset  0x4080  signed 32-bit, = output_zero_point - 0x80
  *     out_cvt_shift   0x4088  the QNNPACK multiplier shift (shift-1)
  *
  * The int8-out path also sets `out_cvt_scale` (0x4084, above) to the QNNPACK 15-bit
  * multiplier instead of 1. */
 uint32_t out_cvt_offset;   // 0x4080
 uint8_t  out_cvt_shift;    // 0x4088 bits[5:0] integer requant shift (int8-out path)
 /* The float-affine OUT_CVT fields. In 0x4088, bit31 is `cvt_type` (0 integer-requant,
  * 1 float-affine) and bits[19:12] are `minus_exp`. The default 0 keeps the integer,
  * byte-identical path. Only the int8 matmul's dequant fold (`ROCKET_INT8_DEQ`) sets them in
  * this descriptor.
  *
  * On the raw int32 accumulator they have no effect. There the converter computes
  * `(acc * out_cvt_scale) >> out_cvt_shift` in the integer domain and rounds to an integer
  * before any float cast. A fractional dequant scale therefore cannot fold into it [HW sweep]
  * (`tests/matmul_int8_dequant_rocket.c`, with `ROCKET_INT8_DEQ=1` and its sweep variables).
  *
  * The float-affine decode applies on the LUT datapath, to the table's output q. With
  * `out_cvt_scale` at 1 and `fp32tofp16_en` narrowing the result to fp16, it is [HW sweep]:
  *
  *     out = (q + out_cvt_offset) * 2^-out_cvt_minus_exp
  *
  * The LUT paths program it from `lut_epilogue_t` and `lut_act_params_t`, not from these
  * fields. */
 uint8_t  out_cvt_minus_exp;  // 0x4088 bits[19:12]
 uint8_t  out_cvt_cvt_type;   // 0x4088 bit31
 /* Selects how the integer requant rounds an exact tie: 0x4088 bit30, `CVT_ROUND` in Mesa's
  * `registers.xml`. 0, every generator's value, rounds half to even. See
  * `npu_out_cvt_round_bit()`. */
 uint8_t  out_cvt_round;      // 0x4088 bit30
 uint16_t bn_mul_operand;     // 0x4068 DPU_BN_MUL_CFG[31:16]: fp16 per-tensor scale
                              // (NVDLA SDP X-mul "channel/per-pixel"). 0 with bn_mul_bypass
                              // keeps the BN stage off (byte-identical). The alternative
                              // dequant-fold path uses it.
 /* The BS-stage int32 bias add, per output channel, read by BRDMA. `bias_en=0` keeps the BS
  * block fully bypassed, the float and int32-raw default. `bias_en=1` programs the Mesa DW
  * int8 word, `BS_ALU_ALGO(2)|BS_ALU_SRC(1)|BS_RELU_BYPASS|BS_MUL_BYPASS`. It also enables
  * BRDMA to fetch the bias cube from `bias_base_addr`. */
 uint8_t  bias_en;          // BS ALU add active
 uint32_t bias_base_addr;   // 0x5020 RDMA_BS_BASE_ADDR (int32 bias BO IOVA)
 /* With `bias_en` set, `bs_mul_src` makes the BS multiplier live with a per-channel operand.
  * The operand is read from the same BRDMA cube, which is then 64 bytes per 8 output channels
  * under BRDMA data use 7. The cube contains int32 A x8 at 0, int16 B x8 at 32 and int16 C x8
  * at 48. The stage computes the following, with the product held wide:
  *
  *     sat32(rne(((acc + A[c]) * C[c]) >> s))
  *
  * The shift is per sign. `BS_MUL_CFG[13:8]` shifts a non-negative product and
  * `DATA_FORMAT[9:4]` a negative one, so the generator writes `bs_mul_shift` into both. 0
  * keeps the per-tensor bias-only word, byte-identical.
  *
  * Evidence: [HW sweep, RK1] (`tests/dw_perc_probe.c`). */
 uint8_t  bs_mul_src;       // 0x4048 BS_MUL_SRC, and BS_MUL_BYPASS cleared
 uint8_t  bs_mul_shift;     // 0x4048[13:8] and 0x4010[9:4]
 /* Enables the conv->activation fusion. `lut_en=0`, the default, keeps the validated
  * BN/EW/OUT_CVT bypass path byte-identical. When it is 1, `gen_conv2d_task` uploads
  * `lut_ep->lut` and programs the BN-mul index scale, the EW LUT and the affine OUT_CVT. The
  * conv result is then post-processed by f(x) in the same task (`lut_epilogue_t`). */
 uint8_t  lut_en;
 const lut_epilogue_t *lut_ep;
 /* Sets `DATA_CUBE_NOTCH_ADDR`, written to both halves (`NOTCH_ADDR_1` [28:16],
  * `NOTCH_ADDR_0` [12:0]). The value is the number of 16-byte atoms from one output row to
  * the next, minus one. With `dst_surf_stride` at one atom, the DPU writes a matmul's `C` as
  * plain [M][N] rows. `gen_matmul_fp16` sets this row-major form from
  * `matmul_params_t.c_row_atoms`. 0 is the cube program's value.
  *
  * Caution: either half alone leaves most of the tile unwritten [HW sweep]
  * (`tests/rowmajor_tile_probe`). */
 uint16_t notch;            // 0x4038
} npu_dpu_desc;

/* Returns the `OUT_CVT_SHIFT` bit-30 word for an integer requant: `requested` (the
 * descriptor's `out_cvt_round`) shifted into place. If `ROCKET_OUT_CVT_ROUND` is set, it
 * overrides `requested` on every integer `OUT_CVT_SHIFT` write of both encoders. The
 * environment variable exists for `tests/requant_round_probe.c`. The float-affine LUT writes
 * leave the bit clear. */
uint32_t npu_out_cvt_round_bit(unsigned requested);


#ifdef __cplusplus
}
#endif
#endif // NPU_DPU_H
