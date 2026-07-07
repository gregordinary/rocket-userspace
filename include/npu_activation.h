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
 * Standalone elementwise activation on the DPU LUT block (no conv/matmul).
 *
 * The RK3588 DPU carries an NVDLA-style SDP LUT: two 513-entry tables (LE =
 * "linear", covering [LE_START, 0]; LO = "exponential", covering [0, LO_END])
 * indexed by the BN-multiplied input, with hardware linear interpolation
 * between adjacent entries and a slope-based extrapolation outside the table
 * range. A pure flying-mode DPU pass reads the input cube via MRDMA, runs it
 * through BN-mul -> LUT -> OUT_CVT, and writes the result. Because the op is
 * elementwise, the host can feed a FLAT fp16 vector and read a flat fp16 vector
 * back: out[i] = f(in[i]) regardless of how the cube dims partition the data.
 *
 * This generator emits that DPU+DPU_RDMA flying program into params->tasks and
 * sets params->task_count. It is the activation sibling of gen_matmul_fp16:
 * shapes/addresses in, regcmd out; the runtime (rocket_activation.c) owns BOs,
 * the LUT table contents, and the per-activation constants.
 */

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  uint64_t input_dma;     /* device VA of the fp16 input cube  */
  uint64_t output_dma;    /* device VA of the fp16 output cube */
  int      n;             /* element count; MUST be a multiple of 8 (C2 atom) */

  /* The two LUT tables, Q-format per the activation. 1026 uint16 entries:
   *   lut[0   .. 512 ] = LE table (table id 0)
   *   lut[513 .. 1025] = LO table (table id 1)                                */
  const uint16_t *lut;

  uint16_t bn_mul_operand;   /* fp16 bits: input -> LUT-index scale            */
  uint32_t bn_alu_cfg;       /* BN ALU bias word (0x80000000 on the val. path) */

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

/* Emit the flying-mode DPU LUT activation program. Returns 0 on success,
 * -1 if n is not a positive multiple of 8, -2 if the cube width overflows. */
int gen_lut_activation_fp16(lut_act_params_t *params);

/*
 * SUPERSEDED (kept as the RE record). Elementwise fp16 multiply attempted with a
 * pure FLYING main: MRDMA reads input A, ERDMA reads input B, EW MUL combines
 * them. This CANNOT WORK on rocket — the DPU EW reads its 2nd operand only when
 * MRDMA is repurposed (COMB_USE(5)) to deliver the operand, which leaves no main
 * feed, so the operand reads 0 (Mesa Teflon add_tensor RE;).
 *
 * The WORKING fully-on-NPU multiply is `rocket_ew_mul_fp16` (rocket_activation.c):
 * an IDENTITY conv/matmul supplies the main feed (== A) and the EW MUL combines
 * the ERDMA operand (B) with it (gen_matmul_fp16 accumulate=1, ew_mul=1 =>
 * EW_OP_TYPE=1). HW-validated bit-exact (tests/ew_mul_rocket.c). This generator is
 * retained only behind ROCKET_ACT_EXPERIMENTAL for the negative-result record.
 */
typedef struct {
  uint64_t input_a_dma;   /* MRDMA main feed  */
  uint64_t input_b_dma;   /* ERDMA EW operand */
  uint64_t output_dma;
  int       n;            /* element count; MUST be a multiple of 8 */
  uint64_t *tasks;
  int       task_count;   /* OUT */
} ew_mul_params_t;

/* Emit the flying-mode DPU elementwise-multiply program. Same return codes. */
int gen_ew_mul_fp16(ew_mul_params_t *params);


#ifdef __cplusplus
}
#endif
#endif /* NPU_ACTIVATION_H */
