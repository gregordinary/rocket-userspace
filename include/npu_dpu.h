// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef NPU_DPU_H
#define NPU_DPU_H

/*
 * Declares the DPU's conv->activation epilogue descriptor and the OUT_CVT tie-rule word.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * `lut_epilogue_t`: the conv->activation LUT fusion descriptor
 * ==========================================================================*/

/* Describes the DPU LUT-activation epilogue, which fuses an activation into a convolution.
 * `rocket_lut_epilogue_build` (`rocket_activation.c`) builds it on the host. The caller
 * attaches it to a conv through `conv_params_t.act`. The conv's NPU task then post-processes
 * its own result with f(x), with no second NPU round trip. The chain is NVDLA's SDP sequence:
 * BN-mul -> EW LUT -> affine OUT_CVT.
 *
 * The register values mirror `gen_lut_activation_fp16`, the standalone flying-mode op. Only
 * the input source differs: the conv's CACC accumulator instead of a flying MRDMA stream. The
 * SDP epilogue is therefore identical.
 *
 * The epilogue supports smooth single-pass activations only: SiLU, tanh and GELU.
 * HardSwish's exactly flat tail at x<=-3 trips the NVDLA LE/LO mux, so HardSwish stays on the
 * host or 2-pass path. */
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

/* Returns the `OUT_CVT_SHIFT` bit-30 word for an integer requant: `requested` shifted into
 * place. Bit 30 (`CVT_ROUND`) selects how the requant rounds an exact tie, and 0, every
 * generator's value, rounds half to even. If `ROCKET_OUT_CVT_ROUND` is set, it overrides
 * `requested` on every integer `OUT_CVT_SHIFT` write of both encoders. The environment
 * variable exists for `tests/requant_round_probe.c`. The float-affine LUT writes leave the bit
 * clear. */
uint32_t npu_out_cvt_round_bit(unsigned requested);


#ifdef __cplusplus
}
#endif
#endif /* NPU_DPU_H */
