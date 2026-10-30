// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef NPU_POOL_H
#define NPU_POOL_H

#include <stdint.h>

/*
 * Declares the register-program generator for pooling on the NPU, MaxPool and AveragePool,
 * and its parameters.
 *
 * The PPU (Planar Processing Unit) is the NPU's pooling engine, the analog of NVDLA's PDP.
 * It is not a reshape engine. A pool is a self-contained task on the PPU and PPU_RDMA blocks
 * only, with no CNA, CORE or DPU and no weights. PPU_RDMA reads the input NC1HWC2 cube. The
 * PPU reduces each kernel window by max or average and writes the output cube in the same
 * NC1HWC2 layout. Pooling is independent per channel, like a depthwise convolution, so the
 * whole of `C` is one task.
 *
 * Max pooling is bit-exact, and average pooling is within fp16-reciprocal tolerance [HW sweep].
 * The register program and the average `RECIP_KERNEL` format agree with the allbilly
 * register-level pooling generator [source-confirmed].
 *
 * In the emitted program, every geometry field contains the value minus 1, and strides are in
 * bytes, 16-aligned. The PC enable trailer writes 0x60, `PPU_OP_EN|PPU_RDMA_OP_EN`, the global
 * block-participation mask. No per-block `OPERATION_ENABLE` is needed. `FLYING_MODE=1` with
 * `DPU_FLYIN=0` selects a standalone PPU task fed by PPU_RDMA.
 *
 * The PPU has no divider. An average pool multiplies the window sum by a per-axis reciprocal:
 *
 *     avg = sum * recip_w * recip_h * 2^-32 = sum/(kw*kh)
 *
 * The register encodes `recip_axis = fp16(65536.0 / k_axis)` as the 17-bit field's fp16 bit
 * pattern. The caller must build it with `ppu_recip_kernel_fp16()`. For an average pool, the
 * kernel extent `k` must be >= 2, because at `k=1` the value 65536 overflows fp16.
 */

/* The `POOLING_METHOD` encoding of `PPU_OPERATION_MODE_CFG`, inherited from NVDLA's PDP. */

#ifdef __cplusplus
extern "C" {
#endif
enum { POOL_METHOD_AVG = 0, POOL_METHOD_MAX = 1, POOL_METHOD_MIN = 2 };

typedef struct {
  uint16_t c;            /* channels (pooled per channel, fp16 cube C2=8)            */
  uint16_t ih, iw;       /* input height / width                                     */
  uint16_t oh, ow;       /* output height / width (caller-computed)                  */
  uint8_t  kh, kw;       /* pooling kernel height / width (>=1, average >=2)         */
  uint8_t  stride_y, stride_x;
  uint8_t  pad_top, pad_left, pad_bottom, pad_right;
  uint8_t  method;       /* POOL_METHOD_MAX / POOL_METHOD_AVG                         */

  /* The average pool's per-axis reciprocal, as the raw 17-bit register field (fp16 bits).
   * The generator ignores it unless `method` is AVG. Build it with `ppu_recip_kernel_fp16(k)`. */
  uint32_t recip_w, recip_h;

  uint32_t input_dma;    /* NPU IOVA of the input  NC1HWC2 cube  */
  uint32_t output_dma;   /* NPU IOVA of the output NC1HWC2 cube  */

  uint64_t *tasks;       /* OUT: regcmd op stream                */
  uint32_t  task_count;  /* OUT: number of NPUOP words written   */
} pool_params_t;

/* Returns the fp16 average reciprocal for a kernel extent `k`: the fp16 bit pattern of
 * 65536.0/k, placed in `PPU_RECIP_KERNEL_WIDTH/HEIGHT`. `k` must be >=2, because at `k<=1` the
 * value overflows fp16. The helper is pure. */
static inline uint32_t ppu_recip_kernel_fp16(int k)
{
    if (k < 1) k = 1;
    _Float16 h = (_Float16)(65536.0f / (float)k);
    uint16_t bits;
    __builtin_memcpy(&bits, &h, sizeof(bits));
    return (uint32_t)bits;
}

/* Generates an fp16 MaxPool or AveragePool over the NC1HWC2 cube (C2=8). It writes the PPU
 * and PPU_RDMA regcmd into `params->tasks` and sets `params->task_count`. The pool is one
 * task, with no spatial tiling. Returns 0 on success and a negative value on bad parameters,
 * such as a dimension or a kernel that overflows its field.
 *
 * The generator emits the fp16 precision pair only. The PPU also pools natively in int8. That
 * mode programs `PROC_PRECISION` 0 and `IN_PRECISION` 1 over a C2=16 byte cube, with an
 * integer Q16 reciprocal and the integer pad fill. It is bit-exact for MAX, MIN and AVG
 * [HW sweep], and this generator does not emit it. int8 and uint8 pooling
 * (`rocket_pool_int8`, `rocket_pool_uint8`) run through this fp16 path, converting the cube
 * between int8 and fp16 on each side. MAX is bit-exact there because int8 is exact in fp16. */
int gen_pool_fp16(pool_params_t *params);


#ifdef __cplusplus
}
#endif
#endif /* NPU_POOL_H */
