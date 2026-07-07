// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef NPU_POOL_H
#define NPU_POOL_H

#include <stdint.h>

/*
 * npu_pool.h — on-NPU POOLING (MaxPool / AveragePool) regcmd generator params.
 *
 * The PPU ("Planar Processing Unit") is the NPU's pooling engine (NVDLA PDP analog;
 * NOT a reshape engine). A pool
 * is a SELF-CONTAINED job on the PPU + PPU_RDMA blocks ONLY (no CNA/CORE/DPU, no
 * weights): PPU_RDMA reads the input NC1HWC2 cube, the PPU reduces each kernel window
 * (max or average) and writes the output cube in the SAME NC1HWC2 layout. Pooling is
 * per-channel independent (like depthwise) so the whole C is one job.
 *
 * GROUND TRUTH: the register program + the average RECIP_KERNEL format are HW-validated
 * (MAX bit-exact; AVG within fp16-reciprocal tolerance) and cross-checked against the
 * allbilly register-level pooling generator. Every
 * geometry field is (value - 1); strides are in bytes (16-aligned). The PC enable
 * trailer writes 0x60 = PPU_OP_EN|PPU_RDMA_OP_EN (the GLOBAL block-participation mask;
 * no per-block OPERATION_ENABLE is needed). FLYING_MODE=1 with DPU_FLYIN=0 == standalone
 * PPU job fed by PPU_RDMA.
 *
 * AVERAGE RECIP: the PPU has no divider; it multiplies the window sum by a per-axis
 * reciprocal, avg = sum * recip_w * recip_h * 2^-32 = sum/(kw*kh). The register encodes
 * recip_axis = fp16(65536.0 / k_axis) as the 17-bit field's fp16 bit pattern. Use
 * ppu_recip_kernel_fp16() to build it. k must be >= 2 (k=1 -> 65536 overflows fp16).
 */

/* PPU_OPERATION_MODE_CFG POOLING_METHOD encoding (NVDLA PDP lineage). */

#ifdef __cplusplus
extern "C" {
#endif
enum { POOL_METHOD_AVG = 0, POOL_METHOD_MAX = 1, POOL_METHOD_MIN = 2 };

typedef struct {
  uint16_t c;            /* channels (per-channel pooling; cube C2=8 for fp16)       */
  uint16_t ih, iw;       /* input height / width                                     */
  uint16_t oh, ow;       /* output height / width (caller-computed)                  */
  uint8_t  kh, kw;       /* pooling kernel height / width  (>=1; avg needs >=2)      */
  uint8_t  stride_y, stride_x;
  uint8_t  pad_top, pad_left, pad_bottom, pad_right;
  uint8_t  method;       /* POOL_METHOD_MAX / POOL_METHOD_AVG                         */

  /* Average-pool per-axis reciprocal, as the raw 17-bit register field (fp16 bits).
   * Ignored when method != AVG. Build with ppu_recip_kernel_fp16(k). */
  uint32_t recip_w, recip_h;

  uint32_t input_dma;    /* NPU IOVA of the input  NC1HWC2 cube  */
  uint32_t output_dma;   /* NPU IOVA of the output NC1HWC2 cube  */

  uint64_t *tasks;       /* OUT: regcmd op stream                */
  uint32_t  task_count;  /* OUT: number of NPUOP words written   */
} pool_params_t;

/* fp16 average reciprocal for a kernel extent k: the fp16 bit pattern of 65536.0/k,
 * placed in PPU_RECIP_KERNEL_WIDTH/HEIGHT. k>=2 (k<=1 saturates fp16). Pure helper. */
static inline uint32_t ppu_recip_kernel_fp16(int k)
{
    if (k < 1) k = 1;
    _Float16 h = (_Float16)(65536.0f / (float)k);
    uint16_t bits;
    __builtin_memcpy(&bits, &h, sizeof(bits));
    return (uint32_t)bits;
}

/* fp16 MaxPool / AveragePool over the NC1HWC2 cube (C2=8). Emits the PPU + PPU_RDMA
 * regcmd into params->tasks and sets params->task_count. Returns 0, <0 on bad params
 * (e.g. dim/kernel field overflow). Single job (no spatial tiling).
 *
 * NPU FACT: the PPU has NO native int8 pooling precision (PROC_PRECISION=int8 reads the
 * cube as fp16 — HW-confirmed; the allbilly reference emits PROC_PRECISION=fp16 for every
 * pool, with no int8 example). int8/uint8 pooling routes through THIS fp16 path with an
 * int8<->fp16 cube boundary (rocket_pool_int8/_uint8) — MAX is bit-exact because int8 is
 * exact in fp16. */
int gen_pool_fp16(pool_params_t *params);


#ifdef __cplusplus
}
#endif
#endif /* NPU_POOL_H */
