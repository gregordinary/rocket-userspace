// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * npu_regmap.h — the RK3588 register map the encoders and the register-level tests use.
 *
 * The map is Mesa's: rkt_registers.h is generated from the register database of Mesa's rocket
 * driver (third_party/mesa-rocket/registers.xml), so a program written with these names reads
 * against Mesa's decoder register for register. It declares, per register, an offset
 * REG_<BLOCK>_<REGISTER>, and per field a function <BLOCK>_<REGISTER>_<FIELD>(value) that
 * shifts the value into place and masks it to the field. A value wider than its field loses
 * its high bits, so a caller that must refuse an out-of-range value checks it first.
 *
 * This header adds what the database leaves unnamed, and nothing else:
 *
 * - CORE 0x3030 and DPU 0x40C4, which the database does not list. Every program writes both
 *   as 0, as Mesa's rkt_regcmd.c and the vendor's programs do.
 * - The per-block enable bits of the trailer's PC_OPERATION_ENABLE word. The database names
 *   that register's bits 31:1 RESERVED_0, but the word is read in the layout of the global
 *   enable register (0xF008), whose fields it does name: GLOBAL_OPERATION_ENABLE_*_OP_EN.
 *   Mesa writes a convolution's word as PC_OPERATION_ENABLE_RESERVED_0(14) |
 *   PC_OPERATION_ENABLE_OP_EN(1), the same 0x1D as NPU_ENABLE_CONV below.
 *
 * What the database does not hold is what a field means beyond its name and position: that
 * DPU OUT_CVT_SCALE bit 15 is a sign, that CVT_ROUND 0 rounds a tie to even, that
 * CNA_DMA_CON2.SURF_STRIDE is a 28-bit two's complement that goes negative below four rows,
 * that the CNA deconvolution strides hold the stride minus one. The encoders state each
 * such fact where they program the field.
 *
 * Private to the library and its tests, and never installed: the generated header defines
 * bare enumerators (PC, CNA, CORE, DPU, GLOBAL, ...) and a non-inline static function,
 * which would reach every translation unit of a frontend through the public headers.
 */
#ifndef NPU_REGMAP_H
#define NPU_REGMAP_H

#include <stdint.h>

/* rkt_get_target() is static and not inline, so a file that includes the map without
 * calling it would warn. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "rkt_registers.h"
#pragma GCC diagnostic pop

#include "npu_hw.h"   /* the regcmd word format: REGCMD, the block targets, the trailer words */

/* ---- registers the database does not list ---------------------------------------------- */

#define REG_CORE_3030 0x00003030u   /* undocumented; every program writes 0 */
#define REG_DPU_40C4  0x000040c4u   /* undocumented; every program writes 0 */

/* ---- the trailer's block enable word ---------------------------------------------------- */

/* The value of the trailer's PC_OPERATION_ENABLE write: the blocks the task runs. */
#define NPU_ENABLE_CONV (GLOBAL_OPERATION_ENABLE_CNA_OP_EN(1) |                  \
                         GLOBAL_OPERATION_ENABLE_CORE_OP_EN(1) |                 \
                         GLOBAL_OPERATION_ENABLE_DPU_OP_EN(1) |                  \
                         GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN(1))     /* 0x1D */
#define NPU_ENABLE_DPU  (GLOBAL_OPERATION_ENABLE_DPU_OP_EN(1) |                  \
                         GLOBAL_OPERATION_ENABLE_DPU_RDMA_OP_EN(1))     /* 0x18 */
#define NPU_ENABLE_PPU  (GLOBAL_OPERATION_ENABLE_PPU_OP_EN(1) |                  \
                         GLOBAL_OPERATION_ENABLE_PPU_RDMA_OP_EN(1))     /* 0x60 */

/* ---- the elementwise words the matmul's accumulate path writes -------------------------- */

/* DPU_EW_CFG for a two-operand EW ALU operation on 16-bit elements: the operand from ERDMA,
 * per pixel, with the LUT and the ReLU bypassed. `algo` is EW_ALU_ALGO, NVDLA's SDP X1 ALU
 * algorithm: 0 max, 1 min, 2 sum. EW_CFG_ALU(2) = 0x108202C0. */
#define EW_ALU_ALGO_MAX 0u
#define EW_ALU_ALGO_MIN 1u
#define EW_ALU_ALGO_ADD 2u
#define EW_CFG_ALU(algo) (DPU_EW_CFG_EW_DATA_MODE(1) | DPU_EW_CFG_EDATA_SIZE(2) |      \
                          DPU_EW_CFG_EW_ALU_ALGO(algo) | DPU_EW_CFG_EW_RELU_BYPASS(1) | \
                          DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_SRC(1))
/* The multiply: EW_OP_TYPE 1 with the operand converter bypassed, 0x108003C4. */
#define EW_CFG_MUL (DPU_EW_CFG_EW_DATA_MODE(1) | DPU_EW_CFG_EDATA_SIZE(2) |              \
                    DPU_EW_CFG_EW_OP_CVT_BYPASS(1) | DPU_EW_CFG_EW_RELU_BYPASS(1) |      \
                    DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_SRC(1) |              \
                    DPU_EW_CFG_EW_OP_TYPE(1))
/* The sum on 32-bit elements, 0x10C202C0. */
#define EW_CFG_ADD_32B (DPU_EW_CFG_EW_DATA_MODE(1) | DPU_EW_CFG_EDATA_SIZE(3) |          \
                        DPU_EW_CFG_EW_ALU_ALGO(EW_ALU_ALGO_ADD) |                         \
                        DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) |     \
                        DPU_EW_CFG_EW_OP_SRC(1))
/* DPU_RDMA_ERDMA_CFG reading the operand per pixel, 16-bit (0x40000008) or 32-bit
 * (0x4000000C) elements. */
#define ERDMA_CFG_16B (DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(1) |                       \
                       DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(2))
#define ERDMA_CFG_32B (DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(1) |                       \
                       DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(3))

#endif /* NPU_REGMAP_H */
