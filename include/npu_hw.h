// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef NPU_HW_H
#define NPU_HW_H

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

// Declares the RK3588 NPU's register offsets and field descriptions. Some descriptions are
// cryptic or missing.
//
// The offsets are the RK3588's and are specific to the IP revision. The block bases (PC
// 0x0xxx, CNA 0x1xxx, CORE 0x3xxx, DPU 0x4xxx) are common to every RKNPU. The datapath
// semantics are common too: the precision fields, the BS, BN, EW and LUT meanings, and the
// CNA->CORE->DPU sequence. The CNA, CORE and DPU geometry-register encoding differs. The
// RK3576 re-packs the CNA block at the same block base. Registers move, the bit-packing
// differs at shared offsets, and it writes offsets that are gaps here
// (0x1018/0x101c/0x103c/0x1090-0x1098/0x118c).
//
// Few CNA geometry registers coincide: 0x1014 (stride) and 0x1110 (weight address) do.
// The feature address does not. The RK3576 places it at 0x1088 on both its datapaths, and
// leaves 0x1070, the RK3588's slot, at zero. Several registers that look RK3576-only are
// these same ones moved: its 0x1048 is CNA_CVT_CON0 with identical packing, and its 0x108c
// is CNA_DMA_CON0 (the burst-length word).
//
// A second-chip port therefore needs a per-chip regcmd encoder for the geometry
// registers, not only an offset table. `npu_regcmd_rk3576.c` is that encoder for the
// RK3576. See `rocket_hw_profile.h`.

/* ============================================================================
 * PC, CNA and CORE register offsets
 * ==========================================================================*/

#define PC_OPERATION_ENABLE    0x0008 // Operation Enable
#define PC_BASE_ADDRESS        0x0010 // PC address register
#define PC_REGISTER_AMOUNTS    0x0014 // Register amount for each task

#define CNA_S_POINTER          0x1004 // Single register group pointer
#define CNA_CONV_CON1          0x100C // Convolution control register1
#define CNA_CONV_CON2          0x1010 // Convolution control register2
#define CNA_CONV_CON3          0x1014 // Convolution control register3
#define CNA_DATA_SIZE0         0x1020 // Feature data size control register0
#define CNA_DATA_SIZE1         0x1024 // Feature data size control register1
#define CNA_DATA_SIZE2         0x1028 // Feature data size control register2
#define CNA_DATA_SIZE3         0x102C // Feature data size control register3
#define CNA_WEIGHT_SIZE0       0x1030 // Weight size control 0
#define CNA_WEIGHT_SIZE1       0x1034 // Weight size control 1
#define CNA_WEIGHT_SIZE2       0x1038 // Weight size control 2
#define CNA_CBUF_CON0          0x1040 // CBUF control register 0
#define CNA_CBUF_CON1          0x1044 // CBUF control register 1
#define CNA_CVT_CON0           0x104C // Input convert control register0
#define CNA_CVT_CON1           0x1050 // Input convert control register1
#define CNA_CVT_CON2           0x1054 // Input convert control register2
#define CNA_CVT_CON3           0x1058 // Input convert control register3
#define CNA_CVT_CON4           0x105C // Input convert control register4
#define CNA_FC_CON0            0x1060 // Full connected control register0
#define CNA_FC_CON1            0x1064 // Full connected control register1
#define CNA_PAD_CON0           0x1068 // Pad control register0
#define CNA_FEATURE_DATA_ADDR  0x1070 // Base address for input feature data
#define CNA_FC_CON2            0x1074 // Full connected control register2
#define CNA_DMA_CON0           0x1078 // AXI control register 0
#define CNA_DMA_CON1           0x107C // AXI control register 1
#define CNA_DMA_CON2           0x1080 // AXI control register 2
#define CNA_FC_DATA_SIZE0      0x1084 // Full connected data size control register0
#define CNA_FC_DATA_SIZE1      0x1088 // Full connected data size control register1
#define CNA_CLK_GATE           0x1090 // Clock gating control on the RK3588, line stride on the RK3576
#define CNA_DCOMP_CTRL         0x1100 // Weight decompress control register
#define CNA_DCOMP_REGNUM       0x1104 // Weight decompress register number
#define CNA_DCOMP_ADDR0        0x1110 // Base address of the weight
#define CNA_DCOMP_AMOUNT       0x1140 // Amount of the weight decompress for the 0 decompress
#define CNA_DCOMP_AMOUNT1      0x1144 // Amount of the weight decompress for the 1 decompress
#define CNA_DCOMP_AMOUNT2      0x1148 // Amount of the weight decompress for the 2 decompress
#define CNA_DCOMP_AMOUNT3      0x114C // Amount of the weight decompress for the 3 decompress
#define CNA_DCOMP_AMOUNT4      0x1150 // Amount of the weight decompress for the 4 decompress
#define CNA_DCOMP_AMOUNT5      0x1154 // Amount of the weight decompress for the 5 decompress
#define CNA_DCOMP_AMOUNT6      0x1158 // Amount of the weight decompress for the 6 decompress
#define CNA_DCOMP_AMOUNT7      0x115C // Amount of the weight decompress for the 7 decompress
#define CNA_DCOMP_AMOUNT8      0x1160 // Amount of the weight decompress for the 8 decompress
#define CNA_DCOMP_AMOUNT9      0x1164 // Amount of the weight decompress for the 9 decompress
#define CNA_DCOMP_AMOUNT10     0x1168 // Amount of the weight decompress for the 10 decompress
#define CNA_DCOMP_AMOUNT11     0x116C // Amount of the weight decompress for the 11 decompress
#define CNA_DCOMP_AMOUNT12     0x1170 // Amount of the weight decompress for the 12 decompress
#define CNA_DCOMP_AMOUNT13     0x1174 // Amount of the weight decompress for the 13 decompress
#define CNA_DCOMP_AMOUNT14     0x1178 // Amount of the weight decompress for the 14 decompress
#define CNA_DCOMP_AMOUNT15     0x117C // Amount of the weight decompress for the 15 decompress
#define CNA_CVT_CON5           0x1180 // Input convert control register5
#define CNA_PAD_CON1           0x1184 // Pad controller register1

#define CORE_S_POINTER         0x3004 // Single register group pointer
#define CORE_MAC_GATING        0x300C // MAC soft clock gating, slcg_op_en [26:0], reset 0x07800800
#define CORE_MISC_CFG          0x3010 // Misc configuration register
#define CORE_DATAOUT_SIZE_0    0x3014 // Feature size register 0 of output
#define CORE_DATAOUT_SIZE_1    0x3018 // Feature size register 1 of output
#define CORE_CLIP_TRUNCATE     0x301C // Shift value register
#define CORE_3030              0x3030 // Undocumented, requirement not characterized

/* ============================================================================
 * DPU register offsets
 * ==========================================================================*/

#define DPU_S_POINTER            0x4004 // Single register group pointer
#define DPU_FEATURE_MODE_CFG     0x400C // Configuration of the feature mode
#define DPU_DATA_FORMAT          0x4010 // Configuration of the data format
#define DPU_OFFSET_PEND          0x4014 // Value of the offset pend
#define DPU_DST_BASE_ADD         0x4020 // Destination base address
#define DPU_DST_SURF_STRIDE      0x4024 // Destination surface size
#define DPU_DATA_CUBE_WIDTH      0x4030 // Width of the input cube
#define DPU_DATA_CUBE_HEIGHT     0x4034 // Height of the input cube  
#define DPU_DATA_CUBE_NOTCH_ADDR 0x4038 // Notch signal of the input cube
#define DPU_DATA_CUBE_CHANNEL    0x403C // Channel of the input cube
#define DPU_BS_CFG               0x4040 // Configuration of the BS
#define DPU_BS_ALU_CFG           0x4044 // Configuration of the BS ALU
#define DPU_BS_MUL_CFG           0x4048 // Configuration of the BS MUL
#define DPU_BS_RELUX_CMP_VALUE   0x404C // Value of the RELUX compare with
#define DPU_BS_OW_CFG            0x4050 // Configuration of the BS OW
#define DPU_BS_OW_OP             0x4054 // Ow op of the BS OW
#define DPU_WDMA_SIZE_0          0x4058 // Size 0 of the WDMA
#define DPU_WDMA_SIZE_1          0x405C // Size 1 of the WDMA
#define DPU_BN_CFG               0x4060 // Configuration of BN
#define DPU_BN_ALU_CFG           0x4064 // Configuration of the BN ALU
#define DPU_BN_MUL_CFG           0x4068 // Configuration of the BN MUL
#define DPU_BN_RELUX_CMP_VALUE   0x406C // Value of the RELUX compare with
#define DPU_EW_CFG               0x4070 // Configuration of EW
#define DPU_EW_CVT_OFFSET_VALUE  0x4074 // Offset of the EW input convert
#define DPU_EW_CVT_SCALE_VALUE   0x4078 // Scale of the EW input convert
#define DPU_EW_RELUX_CMP_VALUE   0x407C // Value of the RELUX compare with
#define DPU_OUT_CVT_OFFSET       0x4080 // Offset of the output converter
#define DPU_OUT_CVT_SCALE        0x4084 // Scale of the output converter
#define DPU_OUT_CVT_SHIFT        0x4088 // Shift of the output converter
#define DPU_EW_OP_VALUE_0        0x4090 // Configure operand0 of the EW
#define DPU_EW_OP_VALUE_1        0x4094 // Configure operand1 of the EW
#define DPU_EW_OP_VALUE_2        0x4098 // Configure operand2 of the EW
#define DPU_EW_OP_VALUE_3        0x409C // Configure operand3 of the EW
#define DPU_EW_OP_VALUE_4        0x40A0 // Configure operand4 of the EW
#define DPU_EW_OP_VALUE_5        0x40A4 // Configure operand5 of the EW
#define DPU_EW_OP_VALUE_6        0x40A8 // Configure operand6 of the EW
#define DPU_EW_OP_VALUE_7        0x40AC // Configure operand7 of the EW

/* ============================================================================
 * DPU EW configuration and ERDMA value builders
 * ==========================================================================*/

/* Build `DPU_EW_CFG` values. The EW two-operand configurations differ only in the
 * `EW_ALU_ALGO` field (`DPU_EW_CFG` bits[17:16]). `EW_CFG_ALU_BASE` is the validated
 * 16-bit-element two-operand configuration (`EW_DATA_MODE=1 | EDATA_SIZE=2 | …`) with
 * `EW_ALU_ALGO=0`, and `EW_CFG_ALU(algo)` sets the field. `EW_ALU_ALGO` is NVDLA's SDP X1
 * ALU algorithm field. The macros expand to the validated per-algorithm constants:
 *
 *     EW_CFG_ALU(MAX)=0x108002C0  EW_CFG_ALU(MIN)=0x108102C0  EW_CFG_ALU(ADD)=0x108202C0 */
#define EW_ALU_ALGO_SHIFT  16u
#define EW_ALU_ALGO_MAX    0u
#define EW_ALU_ALGO_MIN    1u
#define EW_ALU_ALGO_ADD    2u
#define EW_CFG_ALU_BASE    0x108002C0u
#define EW_CFG_ALU(algo)   (EW_CFG_ALU_BASE | ((uint32_t)(algo) << EW_ALU_ALGO_SHIFT))
/* EW configurations that are not ALU-algorithm variants of the base, because their
 * `EW_OP_TYPE` or `EDATA_SIZE` fields differ. They are kept as named constants. */
#define EW_CFG_MUL         0x108003C4u // EW_OP_TYPE=mul (clears EW_ALU_ALGO)
#define EW_CFG_ADD_32B     0x10C202C0u // EW_ALU_ALGO=add, EDATA_SIZE=3 (32-bit elems)
#define ERDMA_CFG_16B      0x40000008u // ERDMA_DATA_MODE=1 | ERDMA_DATA_SIZE=2 (16-bit)
#define ERDMA_CFG_32B      0x4000000Cu // ERDMA_DATA_SIZE=3 (32-bit)

#define DPU_SURFACE_ADD          0x40C0 // Value of the surface adder
#define DPU_40C4                 0x40C4 // Not documented      
#define DPU_LUT_ACCESS_CFG       0x4100 // LUT access address and type
#define DPU_LUT_ACCESS_DATA      0x4104 // Configuration of LUT access data
#define DPU_LUT_CFG              0x4108 // Configuration of the LUT
#define DPU_LUT_INFO             0x410C // LUT information register
#define DPU_LUT_LE_START         0x4110 // LE LUT start point
#define DPU_LUT_LE_END           0x4114 // LE LUT end point
#define DPU_LUT_LO_START         0x4118 // LO LUT start point
#define DPU_LUT_LO_END           0x411C // LO LUT end point
#define DPU_LUT_LE_SLOPE_SCALE   0x4120 // LE LUT slope scale
#define DPU_LUT_LE_SLOPE_SHIFT   0x4124 // LE LUT slope shift
#define DPU_LUT_LO_SLOPE_SCALE   0x4128 // LO LUT slope scale
#define DPU_LUT_LO_SLOPE_SHIFT   0x412C // LO LUT slope shift

/* ============================================================================
 * PPU and PPU-RDMA register offsets (pooling engine)
 * ==========================================================================*/

/* The PPU (Planar Processing Unit) is the NPU's pooling engine, the counterpart of
 * NVDLA's PDP. It reads a feature cube through PPU_RDMA and writes a pooled cube (max or
 * average) in the same NC1HWC2 layout. It has pooling kernel, stride and padding registers
 * and an average-pool reciprocal. It has no reshape, transpose or permute register, and
 * the RK3588 has no NVDLA RUBIK block: none appears in the register map. The PPU
 * therefore cannot de-tile the cube to row-major output.
 *
 * No register converts a layout, so software must pack the weights on the host. A
 * matmul's activation and output need not be packed.
 *
 * A stride-only form reads A and writes C as plain rows [HW sweep, 2026-09-26,
 * tests/rowmajor_matmul_probe, tests/rowmajor_tile_probe]. It programs CNA
 * `GROUP_LINE_OFF`, a line stride of one row of A, and a DPU surface stride of one atom
 * plus the notch. `gen_matmul_fp16` emits it from `matmul_params_t.a_row_atoms` and
 * `c_row_atoms`. The flash attention hands its scores and P between QK and AV that way
 * (`rocket_mm_batch_run_rows`). The other matmul paths pack A and de-tile C on the host.
 *
 * The register map is decoded from Mesa's rocket `registers.xml`, cross-checked against
 * allbilly's `rnpu_registers.h`. `gen_pool_fp16` (`npu_regcmd.c`) emits the PPU's
 * program, behind MaxPool and AveragePool (`rocket_pool.c`) and the spatial reductions
 * (`rocket_reduce.c`). The block IDs are `BLOCK_PPU` and `BLOCK_PPU_RDMA` (below), and the
 * register pages are 0x6000 and 0x7000.
 */
#define PPU_S_STATUS             0x6000
#define PPU_S_POINTER            0x6004 // Single register group pointer
#define PPU_OPERATION_ENABLE     0x6008 // Operation enable
#define PPU_DATA_CUBE_IN_WIDTH   0x600C
#define PPU_DATA_CUBE_IN_HEIGHT  0x6010
#define PPU_DATA_CUBE_IN_CHANNEL 0x6014
#define PPU_DATA_CUBE_OUT_WIDTH  0x6018
#define PPU_DATA_CUBE_OUT_HEIGHT 0x601C
#define PPU_DATA_CUBE_OUT_CHANNEL 0x6020
#define PPU_OPERATION_MODE_CFG   0x6024 // POOLING_METHOD / FLYING_MODE / INDEX_EN / USE_CNT
#define PPU_POOLING_KERNEL_CFG   0x6034 // kernel W/H + stride W/H
#define PPU_RECIP_KERNEL_WIDTH   0x6038 // 1/kW for average pooling
#define PPU_RECIP_KERNEL_HEIGHT  0x603C // 1/kH for average pooling
#define PPU_POOLING_PADDING_CFG  0x6040 // pad top/bottom/left/right
#define PPU_PADDING_VALUE_1_CFG  0x6044
#define PPU_PADDING_VALUE_2_CFG  0x6048
#define PPU_DST_BASE_ADDR        0x6070 // pooled output cube base (same NC1HWC2 layout)
#define PPU_DST_SURF_STRIDE      0x607C
#define PPU_DATA_FORMAT          0x6084 // DPU_FLYIN / INDEX_ADD / PROC_PRECISION
#define PPU_MISC_CTRL            0x60DC // BURST_LEN / MC_SURF_OUT / NONALIGN / SURF_LEN

#define PPU_RDMA_S_STATUS        0x7000
#define PPU_RDMA_S_POINTER       0x7004
#define PPU_RDMA_OPERATION_ENABLE 0x7008
#define PPU_RDMA_CUBE_IN_WIDTH   0x700C
#define PPU_RDMA_CUBE_IN_HEIGHT  0x7010
#define PPU_RDMA_CUBE_IN_CHANNEL 0x7014
#define PPU_RDMA_SRC_BASE_ADDR   0x701C
#define PPU_RDMA_SRC_LINE_STRIDE 0x7024
#define PPU_RDMA_SRC_SURF_STRIDE 0x7028
#define PPU_RDMA_DATA_FORMAT     0x7030

/* ============================================================================
 * Block IDs and PC op-enable encodings
 * ==========================================================================*/

// The NPU's capability is limited to the following units.
#define BLOCK_PC       0x0100
#define BLOCK_CNA      0x0200
#define BLOCK_CORE     0x0800
#define BLOCK_DPU      0x1000
#define BLOCK_DPU_RDMA 0x2000
#define BLOCK_PPU      0x4000
#define BLOCK_PPU_RDMA 0x8000

#define PC_OP_01     0x01  // reg (not characterized)
#define PC_OP_40     0x40  // Not characterized
#define PC_OP_ENABLE 0x80  // Enables block(s)  

#define OP_REG_PC   (BLOCK_PC | PC_OP_01)   // Not characterized
#define OP_REG_CNA  (BLOCK_CNA | PC_OP_01)  // Not characterized
#define OP_REG_CORE (BLOCK_CORE | PC_OP_01) // Not characterized
#define OP_REG_DPU  (BLOCK_DPU | PC_OP_01)  // Not characterized

#define OP_40     (PC_OP_40 | PC_OP_01)     // Not characterized
#define OP_ENABLE (PC_OP_ENABLE | PC_OP_01) // Not characterized
#define OP_NONE   0x0                       // Not characterized

#define PC_ENABLE      0x01  // Enable for this task
#define PC_ENABLE_CNA  0x04  // Interrupt (not characterized)
#define PC_ENABLE_DPU  0x08  // Interrupt (not characterized)
#define PC_ENABLE_PPU  0x10  // Interrupt (not characterized)

/* ============================================================================
 * NPUOP regcmd word builder and CBUF geometry
 * ==========================================================================*/

/* Builds one regcmd word from an op, a value and a register offset. Each parameter is
 * parenthesized, so a compound `value` such as `(2u<<16) | (1u<<6)` is masked as a whole
 * by `((value) & 0xffffffff)`, not only its last `|` term. The whole expansion is
 * parenthesized for safe use as a subexpression. */
#define NPUOP(op, value, reg) ((((uint64_t)((op) & 0xffff)) << 48) | (((uint64_t)((value) & 0xffffffff)) << 16) | (uint64_t)((reg) & 0xffff))

/* CBUF geometry: the RK3588's machine parameters, and the compile-time source that
 * `rocket_hw_rk3588.cbuf_bank_size` and `.cbuf_banks` mirror. Every consumer reads the
 * bank count from the profile rather than from `NPU_CBUF_BANKS`, through
 * `rocket_hw_current()`. That covers the tiling planners (`rocket_matmul.c`,
 * `rocket_conv.c`) and the regcmd bank-assignment arithmetic in `npu_regcmd.c`, whose
 * generators take a params struct and never see a context. `npu_regcmd.c` and
 * `rocket_prepacked_int4.c` still read `NPU_CBUF_BANK_SIZE` directly. The literals remain
 * as the RK3588 profile's initializers, so each value is defined once. See
 * `rocket_hw_profile.h`. */
#define NPU_CBUF_BANK_SIZE 32768
#define NPU_CBUF_BANKS 12


#ifdef __cplusplus
extern "C" {
#endif
/* ============================================================================
 * Convolution mode and precision field encodings
 * ==========================================================================*/

enum  { direct_convolution = 0};
/* The precision field encoding: 3 bits, in the CNA, CORE and DPU input, processing and
 * output fields. int8, fp16, int32 and fp32 are the documented values, and int16 = 1 is
 * the value from the NVDLA lineage. int4 = 6 [HW sweep]: precision 6 is the only value
 * that gives a bit-exact int4xint4->int16, where 3 saturates and 7 writes nothing.
 * bf16 = 3 and tf32 = 7 [HW sweep]: the CNA and DPU precision fields accept 3'd3 for bf16
 * and 3'd7 for tf32. bf16 runs at the same MAC rate as fp16.
 *
 * Six of the eight values are input datatypes: int4, int8, int16, fp16, bf16 and tf32.
 * Each has a matmul generator (`gen_matmul_*` in `npu_matmul.h`) validated on the RK3588.
 * int32 and fp32 are output and DPU precisions, and no matmul takes either as its input. */
enum  { precision_int8 = 0,
        precision_int16 = 1,
        precision_float16 = 2,
        precision_bfloat16 = 3,   /* 3'd3 */
        precision_int32 = 4,
        precision_float32 = 5,
        precision_int4 = 6,
        precision_tf32 = 7};      /* 3'd7, 4-byte input */


#ifdef __cplusplus
}
#endif
#endif //NPU_HW_H
