// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef NPU_ACTIVATION_H
#define NPU_ACTIVATION_H

#include <stdint.h>   /* fixed-width types used in the public params struct */

/*
 * Copyright (C) 2026  The rocket-userspace authors
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

/*
 * Declares the register-program generators for the standalone elementwise activation on the
 * DPU LUT block, with no convolution or matmul, and for the flying-mode elementwise multiply.
 *
 * The RK3588 DPU contains an NVDLA-style SDP LUT of two 513-entry tables. NVDLA names LE the
 * table whose index can be linear or exponential, and LO the linear-only table
 * [source-confirmed: NVDLA IAS, LUT programming]. The library's LUT programs use them as the
 * two halves of one index range:
 *
 * - LE covers [LE_START, 0].
 * - LO covers [0, LO_END].
 *
 * The BN-multiplied input indexes the tables. The hardware interpolates linearly between
 * adjacent entries and extrapolates by a slope outside the table range. A pure flying-mode
 * DPU pass reads the input cube through MRDMA, runs it through BN-mul -> LUT -> OUT_CVT and
 * writes the result. The op is elementwise, so the host can feed a flat fp16 vector and read a
 * flat fp16 vector back. The result is `out[i] = f(in[i])` for any partition of the data into
 * cube dimensions.
 *
 * `gen_lut_activation_fp16` emits that DPU and DPU_RDMA flying program into `params->tasks`
 * and sets `params->task_count`. Like `gen_gemm_fp16`, it takes shapes and addresses and
 * returns regcmd. The runtime (`rocket_activation.c`) allocates the BOs and supplies the LUT
 * table contents and the per-activation constants.
 */

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  uint64_t input_iova;     /* device VA of the fp16 input cube  */
  uint64_t output_iova;    /* device VA of the fp16 output cube */
  int      n;             /* element count, which must be a multiple of 8 (C2 atom) */

  /* The two LUT tables, Q-format per the activation. 1026 uint16 entries:
   *   lut[0   .. 512 ] = LE table (table id 0)
   *   lut[513 .. 1025] = LO table (table id 1)                                */
  const uint16_t *lut;

  uint16_t bn_mul_operand;   /* fp16 bits: input -> LUT-index scale            */
  uint32_t bn_alu_cfg;       /* BN ALU bias word (0x80000000 on the validated path) */

  /* Output converter: maps the LUT's fixed-point result back to fp16.         */
  uint32_t out_cvt_offset;   /* DPU_OUT_CVT_OFFSET (rounding bias)             */
  uint16_t out_cvt_scale;    /* DPU_OUT_CVT_SCALE.OUT_CVT_SCALE               */
  uint8_t  out_cvt_minus_exp;/* DPU_OUT_CVT_SHIFT.MINUS_EXP (>>2^exp)         */
  uint8_t  out_cvt_cvt_type; /* DPU_OUT_CVT_SHIFT.CVT_TYPE                     */

  /* LUT range + interpolation config (defaults below match the sigmoid path). */
  uint32_t lut_le_start;     /* 0xffffc000 = -16384 (Q index)                 */
  uint32_t lut_lo_end;       /* 0x00004000 =  16384                            */
  uint16_t le_slope_scale;   /* LE (x < table) underflow extrapolation slope scale */
  uint8_t  le_slope_shift;   /* ... and shift                                  */
  uint16_t lo_slope_scale;   /* LO (x > table) overflow extrapolation slope scale  */
  uint8_t  lo_slope_shift;   /* ... and shift (e.g. hardswish's linear x>3 tail)   */
  uint8_t  le_index_select;  /* log2(step) for the LE table (5 => step 32)     */
  uint8_t  lo_index_select;  /* log2(step) for the LO table (5 => step 32)     */

  uint64_t *tasks;           /* caller-allocated regcmd op buffer             */
  int       task_count;      /* OUT: number of 64-bit ops emitted             */
} lut_act_params_t;

/* Emits the flying-mode DPU LUT activation program. Returns 0 on success. It returns -1 when
 * the current hardware profile is not the RK3588's, because the program uses the RK3588's
 * geometry-register encoding. It returns -1 for an `n` that is not a positive multiple of 8.
 * It returns -2 for a cube width that overflows its 13-bit field (`n`/8 > 0x1FFF).
 *
 * Two environment variables override the program's framing, for bring-up.
 * `ROCKET_LUT_SPTR` sets the DPU and DPU_RDMA S_POINTER value (default 0xE).
 * `ROCKET_LUT_ENABLE` sets the PC_OPERATION_ENABLE block bitmap (default 0x18, DPU and
 * DPU_RDMA). */
int gen_lut_activation_fp16(lut_act_params_t *params);


#ifdef __cplusplus
}
#endif
#endif /* NPU_ACTIVATION_H */
