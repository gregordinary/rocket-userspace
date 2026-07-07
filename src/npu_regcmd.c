// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * npu_regcmd.c — matmul (as 1x1 convolution) register-command program generators
 * for the mainline *rocket* NPU driver.
 *
 * One generator per supported MAC datatype (fp16/int8/int4/int16/bf16/tf32). Each
 * fills the CNA/CORE/DPU/DPU_RDMA descriptors and emits the regcmd op stream the
 * rocket uAPI submits to the NPU. The register offsets (CNA 0x1xxx / CORE 0x3xxx /
 * DPU 0x4xxx / DPU_RDMA 0x5xxx), the target encoding (BLOCK|0x01), the S_POINTER
 * value (0xE) and the PC enable trailer (0x0041.. magic + 0x81 PC_OPERATION_ENABLE)
 * are public RK3588 NPU programming facts.
 *
 * THE DPU_RDMA / MRDMA TRAP: a 1x1-conv matmul with no element-wise/bias second
 * input MUST still configure the DPU read-DMA domain (0x5xxx) and set
 * RDMA_FEATURE_MODE_CFG.MRDMA_DISABLE. Omit it and the DPU read-DMA engine stays
 * armed waiting for a main-RDMA feed that never arrives: the DPU never raises
 * completion, and the job watchdog reports "NPU job timed out" with the output left
 * untouched. So these generators always arm RDMA_S_POINTER + the DPU_RDMA block and
 * set the matching PC enable mask (the no-bias / no-eltwise path).
 *
 * The fp16 generator is fp16 x fp16 -> fp16/fp32, single task (no tiling); the
 * other datatypes follow the same single-task shape.
 */

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "npu_hw.h"
#include "npu_cna.h"
#include "npu_dpu.h"
#include "npu_matmul.h"
#include "npu_activation.h"
#include "npu_pool.h"
#include "rocket_log.h"     // centralized log channel

/* DPU_RDMA domain (base 0x5xxx) — the registers the conv path touches. The write
 * target is BLOCK_DPU_RDMA|PC_OP_01, exactly as OP_REG_DPU is BLOCK_DPU|PC_OP_01. */
#define OP_REG_DPU_RDMA            (BLOCK_DPU_RDMA | PC_OP_01)   /* 0x2001 */

#define DPU_RDMA_S_POINTER          0x5004
#define DPU_RDMA_DATA_CUBE_WIDTH    0x500C
#define DPU_RDMA_DATA_CUBE_HEIGHT   0x5010
#define DPU_RDMA_DATA_CUBE_CHANNEL  0x5014
#define DPU_RDMA_SRC_BASE_ADDR      0x5018
#define DPU_RDMA_BRDMA_CFG          0x501C
#define DPU_RDMA_BS_BASE_ADDR       0x5020
#define DPU_RDMA_NRDMA_CFG          0x5028
#define DPU_RDMA_BN_BASE_ADDR       0x502C
#define DPU_RDMA_ERDMA_CFG          0x5034
#define DPU_RDMA_EW_BASE_ADDR       0x5038
#define DPU_RDMA_EW_SURF_STRIDE     0x5040
#define DPU_RDMA_FEATURE_MODE_CFG   0x5044
#define DPU_RDMA_SRC_DMA_CFG        0x5048
#define DPU_RDMA_SURF_NOTCH         0x504C
#define DPU_RDMA_PAD_CFG            0x5064
#define DPU_RDMA_WEIGHT             0x5068
#define DPU_RDMA_EW_SURF_NOTCH      0x506C

/* RDMA_FEATURE_MODE_CFG fields: BURST_LEN [14:11], MRDMA_DISABLE bit4,
 * MRDMA_FP16TOFP32_EN bit3, PROC_PRECISION [7:5], IN_PRECISION [17:15]. */
#define RDMA_FMC_BURST_LEN(x)        (((x) & 0xF) << 11)
#define RDMA_FMC_MRDMA_DISABLE(x)    (((x) & 0x1) << 4)
#define RDMA_FMC_MRDMA_FP16TOFP32(x) (((x) & 0x1) << 3)
/* IN_PRECISION [17:15], PROC_PRECISION [7:5] — govern how the RDMA block (incl.
 * ERDMA) interprets operand data. Left at 0 (=int8) for the plain-conv path
 * because the whole RDMA block is off there; MUST be fp16(2) once ERDMA is armed
 * for K-accumulation, else the fp16 partial is read as int8. */
#define RDMA_FMC_IN_PRECISION(x)     (((x) & 0x7) << 15)
#define RDMA_FMC_PROC_PRECISION(x)   (((x) & 0x7) << 5)
/* COMB_USE [10:8]: [0]=MRDMA & ERDMA read same data, [2]=read data to ERDMA.
 * The K-accumulation path uses COMB_USE(5) with MRDMA enabled — this is what
 * makes the EW combine the conv main-data with the ERDMA operand (vs the plain
 * path's MRDMA_DISABLE, which leaves the EW with no main feed). */
#define RDMA_FMC_COMB_USE(x)         (((x) & 0x7) << 8)
/* RDMA_WEIGHT fields: E[31:24] N[23:16] B[15:8] M[7:0] */
#define RDMA_WEIGHT_ALL1             (1u<<24 | 1u<<16 | 1u<<8 | 1u)
/* RDMA_ERDMA_CFG.ERDMA_DISABLE bit0 */
#define RDMA_ERDMA_DISABLE           0x1

/* CNA feature-surface (channel-plane) stride in bytes for a tile of `datain_height`
 * input rows with per-row `line_stride`. A full CNA block is 4 rows; datain_height < 4
 * is a partial (sub-block) height — the IH 4..7 regime where (IH/4)-1 == 0 and a zero
 * surf_stride is HW-proven. Without the clamp the (height/4)-1 term goes negative and
 * masks to a garbage 28-bit stride, corrupting any tile with fewer than 4 input rows
 * (e.g. an OW-column tile shrunk to one output row). The per-datatype matmul generators
 * (whose M%4 rule keeps datain_height a multiple of 4) all route the feature surface
 * through here so the clamp lives in one place. The general conv path needs a different,
 * non-truncating formula and clamps inline. */
/* ============================================================================
 * SECTION — Shared regcmd emitter and feature-stride helper
 * ==========================================================================*/

static inline int cna_feature_surf_stride(int line_stride, int datain_height) {
    int s = line_stride * ((datain_height / 4) - 1);
    return s < 0 ? 0 : s;
}

/*
 * Emit the regcmd program incrementally and return the op count. Beyond the
 * CNA/CORE/DPU config, this arms RDMA_S_POINTER (early) and the DPU_RDMA block
 * (late) and sets the enable mask that brings DPU_RDMA into op_en. See file header.
 */
static int gen_matmul_task(uint64_t *ops, npu_cna_desc *cna_desc,
                           npu_core_desc *core_desc, npu_dpu_desc *dpu_desc)
{
  uint32_t value;
  int i = 0;

  ops[i++] = NPUOP(OP_REG_DPU, 0xE, DPU_S_POINTER);
  /* Arm the DPU read-DMA single-register group right after DPU_S_POINTER. Same
   * 0xE bit pattern (POINTER_PP_MODE|EXECUTER_PP_EN|POINTER_PP_EN). */
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0xE, DPU_RDMA_S_POINTER);

  value = ((cna_desc->proc_precision & 0x7) <<7) | ((cna_desc->in_precision & 0x7)<<4) |
    (cna_desc->conv_mode & 0xf);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CONV_CON1);
  /* 10-bit CNA field. Normal path is m+1 (<=257), well within range; a diagnostic
   * ROCKET_*_GRAINS override >1023 would be truncated by the &0x3FF below and
   * mis-compute. Fail the emission unconditionally (NOT an assert: the Release build
   * compiles asserts out via -DNDEBUG, so the guard would vanish in the shipping lib). */
  if (cna_desc->feature_grains > 0x3FF) {
    ROCKET_LOGE("npu_regcmd: feature_grains %u exceeds 10-bit CNA field (max 1023)\n",
            cna_desc->feature_grains);
    return -1;
  }
  value = ((cna_desc->kernel_groups & 0xFF) << 16) | ((cna_desc->feature_grains & 0x3FF) << 4);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CONV_CON2);
  value = ((cna_desc->conv_y_stride & 0x7) << 3) | (cna_desc->conv_x_stride & 0x7);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CONV_CON3);
  value = ((cna_desc->datain_width) & 0x7FF) << 16 | (cna_desc->datain_height & 0x7FF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE0);
  value = ((cna_desc->datain_channel-1) & 0xFFFF) << 16 | (cna_desc->datain_channel & 0xFFFF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE1);
  value = cna_desc->dataout_width & 0x7FF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE2);
  value = cna_desc->dataout_atomics & 0x3FFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE3);
  /* CNA_WEIGHT_SIZE0 is a dedicated full-width register (total weight bytes), so —
   * unlike the packed/sub-field neighbours above and below — weight_bytes is emitted
   * UNMASKED on purpose. (The field is the full word for all validated matmul
   * tile sizes; re-check the field width before driving much larger tiles.) */
  value = cna_desc->weight_bytes;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_WEIGHT_SIZE0);
  value = cna_desc->weight_bytes_per_kernel & 0x7FFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_WEIGHT_SIZE1);
  value = ((cna_desc->weight_width & 0x1F) <<24) | ((cna_desc->weight_height & 0x1F) << 16) |
    (cna_desc->weight_kernels & 0x3FFF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_WEIGHT_SIZE2);
  /* CBUF operand reuse: bit13 WEIGHT_REUSE / bit12 DATA_REUSE skip the DRAM
   * re-fetch of the weight/input tile when it is already resident from the
   * previous task on this core. Both default 0 => the value is byte-identical
   * to the no-reuse program. */
  value = ((cna_desc->weight_bank & 0xF) << 4) | (cna_desc->data_bank & 0xF) |
          (cna_desc->weight_reuse ? 0x2000 : 0) | (cna_desc->data_reuse ? 0x1000 : 0);
  /* DEBUG sentinel-sweep knob (RE). FC_DATA_BANK is the 3-bit field [10:8] of
   * CNA_CBUF_CON0 the matmul leaves 0 (we drive the CNA in conv mode, not its
   * fully-connected mode, so the FC bank pointer should be a don't-care). This
   * forces it 0..7 on HW without a recompile to confirm it does not affect the
   * matmul datapath. Default unset = unchanged (==0). */
  {
    const char *e = getenv("ROCKET_FC_DATA_BANK");
    if (e) value |= ((uint32_t)(atoi(e) & 0x7)) << 8;
  }
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CBUF_CON0);
  value = cna_desc->data_entries & 0x1FFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CBUF_CON1);
  value = ((cna_desc->data_sign & 0x1) << 3) | ((cna_desc->cvt_type & 0x1)<< 1) | (cna_desc->cvt_bypass & 0x1);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON0);
  value = ((cna_desc->cvt_scale0 & 0xFFFF) << 16) | 0x0;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON1);
  value = ((cna_desc->cvt_scale1 & 0xFFFF) << 16) | 0x0;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON2);
  value = ((cna_desc->cvt_scale2 & 0xFFFF) << 16) | 0x0;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON3);
  value = ((cna_desc->cvt_scale3 & 0xFFFF) << 16) | 0x0;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON4);
  value = cna_desc->fc_skip_en & 0x1;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_FC_CON0);
  value = cna_desc->data_offset & 0x1FFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_FC_CON1);
  value = ((cna_desc->pad_left & 0xF) << 4) | (cna_desc->pad_top & 0xF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_PAD_CON0);
  ops[i++] = NPUOP(OP_REG_CNA, cna_desc->feature_base_addr, CNA_FEATURE_DATA_ADDR);
  value = cna_desc->weight_offset & 0x1FFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_FC_CON2);
  value = ((cna_desc->weight_burst_len & 0xF) << 16) | (cna_desc->data_burst_len & 0xF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DMA_CON0);
  value = cna_desc->line_stride & 0xFFFFFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DMA_CON1);
  value = cna_desc->surf_stride & 0xFFFFFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DMA_CON2);
  value = ((cna_desc->dma_width & 0x7FF) << 16) | (cna_desc->dma_height & 0x7FF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_FC_DATA_SIZE0);
  value = cna_desc->dma_channel & 0xFFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_FC_DATA_SIZE1);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_CTRL);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_REGNUM);
  ops[i++] = NPUOP(OP_REG_CNA, cna_desc->decompress_addr0, CNA_DCOMP_ADDR0);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT1);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT2);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT3);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT4);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT5);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT6);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT7);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT8);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT9);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT10);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT11);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT12);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT13);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT14);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT15);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_CVT_CON5);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_PAD_CON1);

  value = ((core_desc->proc_precision & 0x7) << 8) | (core_desc->qd_en & 0x1);
  ops[i++] = NPUOP(OP_REG_CORE, value, CORE_MISC_CFG);
  value = ((core_desc->dataout_height & 0xFFFF) << 16) | (core_desc->dataout_width & 0xFFFF);
  ops[i++] = NPUOP(OP_REG_CORE, value, CORE_DATAOUT_SIZE_0);
  value = core_desc->dataout_channel & 0xFFFF;
  ops[i++] = NPUOP(OP_REG_CORE, value, CORE_DATAOUT_SIZE_1);
  ops[i++] = NPUOP(OP_REG_CORE, 0x0, CORE_CLIP_TRUNCATE);
  ops[i++] = NPUOP(OP_REG_CORE, 0x0, CORE_3030);

  value = ((dpu_desc->burst_len & 0xF) << 5) | ((dpu_desc->conv_mode & 0x3) <<3) |
    ((dpu_desc->output_mode & 0x3) <<1) | (dpu_desc->flying_mode & 0x1);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_FEATURE_MODE_CFG);
  value = ((dpu_desc->out_precision & 0x7) << 29) | ((dpu_desc->in_precision & 0x7) << 26) |
    ((dpu_desc->mc_surf_out & 0x1) << 3) | (dpu_desc->proc_precision & 0x7);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_DATA_FORMAT);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_OFFSET_PEND);
  ops[i++] = NPUOP(OP_REG_DPU, dpu_desc->dst_base_addr, DPU_DST_BASE_ADD);
  value = (dpu_desc->dst_surf_stride & 0xFFFFFFF) << 4;
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_DST_SURF_STRIDE);
  value = dpu_desc->width & 0x1FFF;
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_DATA_CUBE_WIDTH);
  value = dpu_desc->height & 0x1FFF;
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_DATA_CUBE_HEIGHT);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_DATA_CUBE_NOTCH_ADDR);
  value = ((dpu_desc->channel & 0x1FFF) << 16) | (dpu_desc->channel & 0x1FFF);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_DATA_CUBE_CHANNEL);
  value = ((dpu_desc->bs_relu_bypass & 0x1) << 6) | ((dpu_desc->bs_mul_bypass & 0x1) << 4) |
    ((dpu_desc->bs_alu_bypass & 0x1) << 1) | (dpu_desc->bs_bypass & 0x1);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_BS_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_BS_ALU_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_BS_MUL_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_BS_RELUX_CMP_VALUE);
  value = ((dpu_desc->tp_org_en & 0x1) << 27) | ((dpu_desc->size_e_2 & 0x7) << 8) |
    ((dpu_desc->size_e_1 & 0x7) << 5) |
    ((dpu_desc->size_e_0 & 0x7) << 2) | ((dpu_desc->od_bypass & 0x1) << 1);
  ops[i++] = NPUOP(OP_REG_DPU, value,  DPU_BS_OW_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, (dpu_desc->bs_ow_op & 0xFFFF), DPU_BS_OW_OP);
  value = ((dpu_desc->tp_precision & 0x1) << 27) | ((dpu_desc->size_c_wdma & 0x7FF) << 16) |
    (dpu_desc->channel_wdma & 0x1FFF);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_WDMA_SIZE_0);
  value = ((dpu_desc->height_wdma & 0x1FFF) << 16) | (dpu_desc->width_wdma & 0x1FFF);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_WDMA_SIZE_1);
  value = ((dpu_desc->bn_relu_bypass & 0x1) << 6) | ((dpu_desc->bn_mul_bypass &0x1) << 4) |
    ((dpu_desc->bn_alu_bypass & 0x1) << 1) | (dpu_desc->bn_bypass & 0x1);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_BN_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_BN_ALU_CFG);
  /* BN MUL operand: 0 (+ bn_mul_bypass=1) keeps the BN stage off, byte-identical.
   * The dequant-fold alt path drives a per-tensor fp16 dequant scale here (NVDLA SDP X-mul). */
  ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)dpu_desc->bn_mul_operand & 0xFFFF) << 16, DPU_BN_MUL_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0,DPU_BN_RELUX_CMP_VALUE);
  /* K-accum: drive the EW ALU in add mode (pre-assembled, env-tunable) instead
   * of the bypass word. */
  if (dpu_desc->ew_accumulate) {
    value = dpu_desc->ew_cfg_val;
  } else {
    value = ((dpu_desc->ew_relu_bypass & 0x1) << 9) | ((dpu_desc->ew_op_cvt_bypass & 0x1) << 8) |
      ((dpu_desc->ew_lut_bypass & 0x1) <<7) | ((dpu_desc->ew_op_bypass & 0x1) << 1) |
      (dpu_desc->ew_bypass & 0x1);
  }
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_EW_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_CVT_OFFSET_VALUE);
  ops[i++] = NPUOP(OP_REG_DPU, 0x1, DPU_EW_CVT_SCALE_VALUE);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_RELUX_CMP_VALUE);
  ops[i++] = NPUOP(OP_REG_DPU, dpu_desc->out_cvt_offset, DPU_OUT_CVT_OFFSET);
  value = ((dpu_desc->fp32tofp16_en & 0x1) << 16) | (dpu_desc->out_cvt_scale & 0xFFFF);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_OUT_CVT_SCALE);
  /* OUT_CVT_SHIFT word: bit31 cvt_type (1=float-affine), bits[19:12] minus_exp,
   * bits[5:0] integer requant shift. All-zero default = the byte-identical integer
   * pass-through used by fp16/bf16/tf32/int4/int16 and plain int32-out int8. The
   * int8->float dequant fold sets cvt_type/minus_exp here. */
  value = (((uint32_t)dpu_desc->out_cvt_cvt_type & 0x1) << 31) |
          (((uint32_t)dpu_desc->out_cvt_minus_exp & 0xFF) << 12) |
          ((uint32_t)dpu_desc->out_cvt_shift & 0x3F);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_OUT_CVT_SHIFT);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_0);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_1);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_2);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_3);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_4);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_5);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_6);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_7);
  value = ((dpu_desc->surf_add & 0xFFFFFFF) << 4);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_SURFACE_ADD);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_40C4);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_ACCESS_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_ACCESS_DATA);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_INFO);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_START);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_END);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_START);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_END);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_SLOPE_SCALE);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_SLOPE_SHIFT);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_SLOPE_SCALE);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_SLOPE_SHIFT);

  /* Configure the DPU_RDMA block. With no eltwise/bias second feed, disable both
   * main- and eltwise-RDMA — this is what clears the "NPU job timed out" (see the
   * MRDMA trap in the file header). Output cube dims match the DPU output
   * (W=1,H=M,C=N). */
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (dpu_desc->width & 0x1FFF), DPU_RDMA_DATA_CUBE_WIDTH);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (dpu_desc->height & 0x1FFF), DPU_RDMA_DATA_CUBE_HEIGHT);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (dpu_desc->channel & 0x1FFF), DPU_RDMA_DATA_CUBE_CHANNEL);
  /* K-accum: COMB_USE(5) has MRDMA and ERDMA read the same partial, so SRC_BASE
   * (MRDMA) points at the add tensor too. Else 0 (MRDMA off). */
  ops[i++] = NPUOP(OP_REG_DPU_RDMA,
                   dpu_desc->ew_accumulate ? dpu_desc->ew_src_base_addr : 0x0,
                   DPU_RDMA_SRC_BASE_ADDR);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_BRDMA_CFG);   /* BS bypassed -> no bias read */
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_BS_BASE_ADDR);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_NRDMA_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_BN_BASE_ADDR);
  /* K-accum: arm the eltwise read-DMA to fetch the running partial from
   * ew_base_addr. Else keep ERDMA disabled, as on the plain-conv path. */
  if (dpu_desc->ew_accumulate) {
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, dpu_desc->erdma_cfg_val, DPU_RDMA_ERDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, dpu_desc->ew_base_addr,  DPU_RDMA_EW_BASE_ADDR);
    /* ew_surf_stride is the RAW register value (already shifted in gen). */
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, dpu_desc->ew_surf_stride, DPU_RDMA_EW_SURF_STRIDE);
  } else {
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, RDMA_ERDMA_DISABLE, DPU_RDMA_ERDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_EW_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_EW_SURF_STRIDE);
  }
  /* K-accum needs the RDMA block in fp16 so ERDMA reads the partial correctly;
   * keep MRDMA disabled (our main feed is the conv, not MRDMA). */
  if (dpu_desc->ew_accumulate) {
    value = dpu_desc->rdma_fmc_val;
  } else {
    value = RDMA_FMC_BURST_LEN(15) | RDMA_FMC_MRDMA_DISABLE(1) |
            (dpu_desc->fp32tofp16_en ? RDMA_FMC_MRDMA_FP16TOFP32(1) : 0);
  }
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, value, DPU_RDMA_FEATURE_MODE_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_SRC_DMA_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA,
                   dpu_desc->ew_accumulate ? dpu_desc->ew_surf_notch : 0x0,
                   DPU_RDMA_SURF_NOTCH);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_PAD_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, RDMA_WEIGHT_ALL1, DPU_RDMA_WEIGHT);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA,
                   dpu_desc->ew_accumulate ? dpu_desc->ew_surf_notch : 0x0,
                   DPU_RDMA_EW_SURF_NOTCH);

  /* PC trailer. The 0x0041.. magic word (required before op_en) then the
   * 0x81 PC_OPERATION_ENABLE. Enable mask 0x1D = RESERVED_0(14)|OP_EN = bits
   * 0,2,3,4; bit 4 is the one that brings the DPU_RDMA block into the op_en. */
  ops[i++] = NPUOP(OP_NONE, 0x0, 0x0);
  ops[i++] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
  ops[i++] = NPUOP(OP_40, 0x0, 0x0);
  ops[i++] = NPUOP(OP_ENABLE, 0x1D, PC_OPERATION_ENABLE);

  return i;
}

/* ============================================================================
 * SECTION — Matmul regcmd generators (fp16)
 * ==========================================================================*/

int gen_matmul_fp16(matmul_params_t *params)
{
   npu_cna_desc cna_desc;
   npu_core_desc core_desc;
   npu_dpu_desc dpu_desc = {0};   /* zero new output-writer fields; all used fields set below */

   unsigned int fd_bytes;
   unsigned int fd_banks;
   unsigned int weight_banks;
   int surf_stride;

   cna_desc.conv_mode = direct_convolution;
   cna_desc.in_precision = precision_float16;
   cna_desc.proc_precision = precision_float16;

   cna_desc.kernel_groups = 0;
   cna_desc.feature_grains = params->m+1;
   cna_desc.conv_x_stride = 1;
   cna_desc.conv_y_stride = 1;

   cna_desc.datain_width = 1;
   cna_desc.datain_height = params->m;
   cna_desc.datain_channel = params->k;
   cna_desc.dataout_width = 1;
   cna_desc.dataout_height = params->m;
   cna_desc.dataout_atomics = cna_desc.dataout_width * cna_desc.dataout_height;

   cna_desc.weight_width = 1;
   cna_desc.weight_height = 1;
   cna_desc.weight_kernels = params->n;
   cna_desc.weight_bytes_per_kernel = cna_desc.weight_width * cna_desc.weight_height *
     cna_desc.datain_channel * sizeof(_Float16);
   cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels;

   fd_bytes = cna_desc.datain_width * cna_desc.datain_height * cna_desc.datain_channel * sizeof(_Float16);
   fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
   fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks +1;
   weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
   weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE)==0) ? weight_banks : weight_banks + 1;
   if ((fd_banks) > NPU_CBUF_BANKS-1) {
     return -1;
   } else {
       if (cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE) {
        weight_banks = NPU_CBUF_BANKS - fd_banks;
       } else {
         return -2;
       }
   }

   cna_desc.weight_bank = weight_banks;
   cna_desc.data_bank = fd_banks;
   cna_desc.weight_reuse = params->weight_reuse & 0x1;   /* default 0 */
   cna_desc.data_reuse   = params->data_reuse   & 0x1;
   cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / 32;
   cna_desc.data_entries = (((cna_desc.datain_width * cna_desc.datain_channel) % 32) == 0) ?
     cna_desc.data_entries : cna_desc.data_entries +1;
   cna_desc.data_sign = 0x1;
   cna_desc.cvt_type  = 0x1;
   cna_desc.cvt_bypass = 0x1;
   cna_desc.cvt_scale0 = 0x1;
   cna_desc.cvt_scale1 = 0x1;
   cna_desc.cvt_scale2 = 0x1;
   cna_desc.cvt_scale3 = 0x1;
   cna_desc.fc_skip_en = 0;
   cna_desc.data_offset = 0x0;
   cna_desc.pad_left = 0;
   cna_desc.pad_top = 0;
   cna_desc.feature_base_addr = params->input_dma;
   cna_desc.weight_offset = 0;
   cna_desc.weight_burst_len = 0xf;
   cna_desc.data_burst_len = 0xf;
   cna_desc.line_stride = cna_desc.datain_width * 4;
   surf_stride = cna_feature_surf_stride(cna_desc.line_stride, cna_desc.datain_height);
   cna_desc.surf_stride = surf_stride;
   cna_desc.dma_width = cna_desc.datain_width;
   cna_desc.dma_height = cna_desc.datain_height;
   cna_desc.dma_channel = cna_desc.datain_channel;
   cna_desc.decompress_addr0 = params->weights_dma;

   core_desc.proc_precision = precision_float16;
   core_desc.qd_en = 1;
   core_desc.dataout_height = cna_desc.dataout_height - 1;
   core_desc.dataout_width = cna_desc.dataout_width - 1;
   core_desc.dataout_channel = cna_desc.weight_kernels -1;

   dpu_desc.burst_len = 0xf;
   dpu_desc.conv_mode = direct_convolution;
   dpu_desc.output_mode = 0x2;
   dpu_desc.flying_mode = 0x0;
   dpu_desc.out_precision = (params->fp32tofp16==0) ? precision_float32 : precision_float16;
   dpu_desc.in_precision = precision_float16;
   dpu_desc.proc_precision = precision_float16;
   dpu_desc.dst_base_addr = params->output_dma;
   dpu_desc.dst_surf_stride = cna_desc.dataout_height * cna_desc.dataout_width;
   dpu_desc.width = core_desc.dataout_width ;
   dpu_desc.height = core_desc.dataout_height;
   dpu_desc.channel = core_desc.dataout_channel;
   dpu_desc.bs_bypass = 1;
   dpu_desc.bs_alu_bypass = 1;
   dpu_desc.bs_mul_bypass = 1;
   dpu_desc.bs_relu_bypass = 1;
   dpu_desc.bn_bypass =1;
   dpu_desc.bn_alu_bypass = 1;
   dpu_desc.bn_mul_bypass = 1;
   dpu_desc.bn_relu_bypass = 1;
   dpu_desc.ew_bypass =1;
   dpu_desc.ew_op_bypass =1;
   dpu_desc.ew_lut_bypass =1;
   dpu_desc.ew_op_cvt_bypass =1;
   dpu_desc.ew_relu_bypass=1;
   dpu_desc.fp32tofp16_en = params->fp32tofp16 & 0x1;
   dpu_desc.out_cvt_scale =1;
   if (params->fp32tofp16 ==0) {
     dpu_desc.size_e_2 = 3;
     dpu_desc.size_e_1 = 3;
     dpu_desc.size_e_0 = 3;
   } else {
     dpu_desc.size_e_2 = 1;
     dpu_desc.size_e_1 = 1;
     dpu_desc.size_e_0 = 1;
   }
   dpu_desc.od_bypass = 1;
   dpu_desc.width_wdma = core_desc.dataout_width;
   dpu_desc.height_wdma = core_desc.dataout_height;
   dpu_desc.channel_wdma = core_desc.dataout_channel;
   dpu_desc.surf_add = (!params->fp32tofp16) ? dpu_desc.dst_surf_stride * 4 : dpu_desc.dst_surf_stride * 2;

   /* K-accumulation (eltwise-add) config: the conv result is the DPU main input,
    * MRDMA stays disabled, ERDMA adds the running partial. The fp16 EW/ERDMA
    * precision encodings (16-bit operands, not the int8 ones):
    *   DPU_EW_CFG        = EW_DATA_MODE(1)|EDATA_SIZE(2=16bit)|EW_ALU_ALGO(2=add)
    *                       |EW_RELU_BYPASS(1)|EW_LUT_BYPASS(1)|EW_OP_SRC(1)
    *                     = 0x108202C0  (EW_BYPASS/EW_OP_BYPASS clear => add active)
    *   DPU_RDMA_ERDMA_CFG= ERDMA_DATA_MODE(1)|ERDMA_DATA_SIZE(2=16bit) = 0x40000008
    * EDATA_SIZE/ERDMA_DATA_SIZE encoding: 0=4b 1=8b 2=16b 3=32b (so fp32 accum =>
    * use 3). Both words are env-overridable for a HW sweep without recompiling:
    *   ROCKET_EW_CFG, ROCKET_ERDMA_CFG. */
   dpu_desc.ew_accumulate = params->accumulate & 0x1;
   /* --- multi-surface K-accum geometry. The conv output is a surface-planar cube;
    * one position spans 16 bytes (8 fp16 ch, or 16 int8 ch) = ATOMIC_K_SIZE. All
    * stride/notch fields are reg bits [4:31], i.e. the geometric value <<4. For
    * matmul output_w=1, output_h=M:
    *   ew_stride  = MAX(output_w*output_h, 12) = MAX(M,12)
    *   surf_notch = ew_stride (single un-split task; split term is 0)
    *   ERDMA base = add_dma + output_w*output_h*16  (one surface)
    *   MRDMA SRC  = add_dma (no offset)
    *   COMB_USE(5) combines the MRDMA (surface n) + ERDMA (surface n+offset) feeds.
    * NOTE: MAX(.,12) over-states the surface stride for M<12 — use M>=12 (e.g.
    * M=16) to exercise the real, floor-free geometry. */
   {
     uint32_t M = (uint32_t)params->m;          /* output_h; output_w == 1      */
     uint32_t hw = M;                            /* output_w*output_h            */
     uint32_t ew_stride = hw > 12 ? hw : 12;     /* MAX(hw,12), geometric value  */
     uint32_t base_off  = hw * 16;               /* one surface in BYTES         */
     const char *e;
     if ((e = getenv("ROCKET_EW_SURF")))     ew_stride = (uint32_t)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_EW_BASE_OFF"))) base_off  = (uint32_t)strtoul(e, NULL, 0);
     uint32_t notch = ew_stride;
     if ((e = getenv("ROCKET_EW_NOTCH")))    notch = (uint32_t)strtoul(e, NULL, 0);
     dpu_desc.ew_src_base_addr = params->add_dma;             /* MRDMA, no offset */
     dpu_desc.ew_base_addr     = params->add_dma + base_off;  /* ERDMA, +1 surf   */
     dpu_desc.ew_surf_stride   = ew_stride << 4;              /* reg field [4:31] */
     dpu_desc.ew_surf_notch    = notch << 4;                  /* reg field [4:31] */
   }
   {
     /* RDMA feature-mode: COMB_USE(5), MRDMA enabled, fp16 precision (16-bit
      * operands). All env-overridable for a HW sweep:
      *   ROCKET_EW_CFG / ROCKET_ERDMA_CFG / ROCKET_RDMA_FMC.
      * FMC = BURST_LEN(15)|COMB_USE(5)|IN_PRECISION(2)|PROC_PRECISION(2) = 0x17D40. */
     /* PER-PIXEL defaults: ERDMA_DATA_MODE(1)=bit30, EW_DATA_MODE(1)=bit28.
      * (Dropping both mode bits, or running with notch=0 / no base offset, makes
      * the EW_SURF_STRIDE sweep inert.) EW_CFG 0x108202C0; ERDMA 0x40000008
      * (DATA_SIZE 2 = 16-bit fp16; int8 would use size 1 = 8-bit).
      * ew_mul=1 selects the eltwise-MULTIPLY encoding instead of ADD:
      *   0x108003C4 = EW_DATA_MODE(1)|EDATA_SIZE(2)|EW_OP_CVT_BYPASS(1)|
      *                EW_RELU_BYPASS(1)|EW_LUT_BYPASS(1)|EW_OP_SRC(1)|EW_OP_TYPE(1)
      * i.e. clear EW_ALU_ALGO(2=add) and set EW_OP_TYPE(1=mul) (the fp16
      * eltwise-multiply word). HW-validated bit-exact with an identity-conv main feed. */
     /* ew_op (>0) selects the EW ALU algo on the SAME conv-main datapath: 2=MAX (EW_ALU_ALGO
      * 0, 0x108002C0), 3=MIN (EW_ALU_ALGO 1, 0x108102C0); otherwise fall back to ew_mul
      * (add 0x108202C0 / mul 0x108003C4). EW_ALU_ALGO lives at DPU_EW_CFG bits[17:16]. */
     uint32_t ew;
     switch (params->ew_op) {
     case 2:  ew = EW_CFG_ALU(EW_ALU_ALGO_MAX); break;   /* MAX */
     case 3:  ew = EW_CFG_ALU(EW_ALU_ALGO_MIN); break;   /* MIN */
     default: ew = params->ew_mul ? EW_CFG_MUL : EW_CFG_ALU(EW_ALU_ALGO_ADD); break;
     }
     uint32_t erd = ERDMA_CFG_16B;
     uint32_t fmc = RDMA_FMC_BURST_LEN(15) | RDMA_FMC_COMB_USE(5) |
                    RDMA_FMC_IN_PRECISION(precision_float16) |
                    RDMA_FMC_PROC_PRECISION(precision_float16);
     const char *e;
     if ((e = getenv("ROCKET_EW_CFG")))    ew  = (uint32_t)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_ERDMA_CFG"))) erd = (uint32_t)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_RDMA_FMC")))  fmc = (uint32_t)strtoul(e, NULL, 0);
     dpu_desc.ew_cfg_val    = ew;
     dpu_desc.erdma_cfg_val = erd;
     dpu_desc.rdma_fmc_val  = fmc;
   }

   {
     int rc = gen_matmul_task(params->tasks, &cna_desc, &core_desc, &dpu_desc);
     if (rc < 0) return rc;
     params->task_count = (uint32_t)rc;
   }

   return 0;
}

/* ============================================================================
 * SECTION — LUT-activation and elementwise regcmd generators
 * ==========================================================================*/

/* Upload the two NVDLA LE/LO tables (513 Q-format entries each) through the DPU LUT
 * access port: DPU_LUT_ACCESS_CFG selects ACCESS_TYPE(1)=write + TABLE_ID (0=LE,
 * 1=LO) + ADDR(0), then each DPU_LUT_ACCESS_DATA write stores one entry and
 * auto-increments the address. Shared by the standalone activation op and the
 * conv->activation fusion. Returns the new op index. The 513 DATA writes per table
 * MUST stay contiguous (an intervening ACCESS_CFG resets the address). */
static int emit_lut_tables(uint64_t *ops, int i, const uint16_t *lut)
{
  ops[i++] = NPUOP(OP_REG_DPU, (1u<<17) | (0u<<16) | 0u, DPU_LUT_ACCESS_CFG);
  for (int j = 0; j <= 512; j++)
    ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)lut[j] & 0xFFFF, DPU_LUT_ACCESS_DATA);
  ops[i++] = NPUOP(OP_REG_DPU, (1u<<17) | (1u<<16) | 0u, DPU_LUT_ACCESS_CFG);
  for (int j = 0; j <= 512; j++)
    ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)lut[513 + j] & 0xFFFF, DPU_LUT_ACCESS_DATA);
  return i;
}

/* ---------------------------------------------------------------------------
 * DPU LUT activation (elementwise, flying mode) — gen_lut_activation_fp16.
 *
 * A standalone NVDLA-SDP LUT pass with NO conv (no CNA/CORE): MRDMA reads the
 * fp16 input cube (flying mode), the BN stage multiplies it by bn_mul_operand to
 * land it in the LUT index domain, the LE/LO hybrid table + hardware linear
 * interpolation produce f(x), OUT_CVT casts the fixed-point result back to fp16,
 * and the DPU writes the output cube. Only the DPU + DPU_RDMA blocks run.
 *
 * The op is elementwise, so the runtime feeds a flat fp16 vector and reads a
 * flat fp16 vector back (out[i]=f(in[i])): the cube dims (C2=8 channels, width
 * cols=n/8, height 1) only partition the data; the read/write strides are
 * identical so the element correspondence is preserved bit-for-bit.
 *
 * FRAMING mirrors gen_matmul_task (the rocket-proven task shape: DPU/DPU_RDMA
 * S_POINTER arm + the OP_NONE/PC/OP_40/OP_ENABLE trailer). REGISTER CONTENT is
 * the LUT path (register fields per Mesa registers.xml / our npu_hw.h; geometry HW-verified).
 * The enable word (default 0x18 = DPU|DPU_RDMA block bits) and S_POINTER value
 * are env-overridable (ROCKET_LUT_ENABLE / ROCKET_LUT_SPTR) for HW bring-up.
 * See include/npu_activation.h.
 * --------------------------------------------------------------------------- */
int gen_lut_activation_fp16(lut_act_params_t *p)
{
  if (p->n <= 0 || (p->n & 0x7)) return -1;     /* must be a +ve multiple of 8 */
  uint32_t cols  = (uint32_t)p->n / 8;          /* C2=8 fp16 per cube position */
  if (cols > 0x1FFF) return -2;                 /* DATA_CUBE_WIDTH is 13 bits  */
  uint32_t width = cols - 1;                    /* width register is W-1       */

  uint32_t sptr   = 0xE;     /* arm single-register group (matmul convention)  */
  uint32_t enable = 0x18;    /* RESERVED_0(12) => DPU+DPU_RDMA block bits 3,4  */
  const char *e;
  if ((e = getenv("ROCKET_LUT_SPTR")))   sptr   = (uint32_t)strtoul(e, NULL, 0);
  if ((e = getenv("ROCKET_LUT_ENABLE"))) enable = (uint32_t)strtoul(e, NULL, 0);

  uint64_t *ops = p->tasks;
  int i = 0;

  ops[i++] = NPUOP(OP_REG_DPU,      sptr, DPU_S_POINTER);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, sptr, DPU_RDMA_S_POINTER);

  /* Upload the two LUT tables (513 Q-format entries each) via the LUT access port. */
  i = emit_lut_tables(ops, i, p->lut);

  /* DPU feature/format: flying mode, output_mode 2, fp16 in/proc, fp16 out. */
  ops[i++] = NPUOP(OP_REG_DPU, (15u<<5) | (2u<<1) | (1u<<0), DPU_FEATURE_MODE_CFG);
  ops[i++] = NPUOP(OP_REG_DPU,
                   ((uint32_t)precision_float16 << 29) | ((uint32_t)precision_float16 << 26) |
                   ((uint32_t)precision_float16 << 0), DPU_DATA_FORMAT);
  ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)p->output_dma, DPU_DST_BASE_ADD);
  ops[i++] = NPUOP(OP_REG_DPU, (cols & 0xFFFFFFF) << 4, DPU_DST_SURF_STRIDE);
  ops[i++] = NPUOP(OP_REG_DPU, (width & 0x1FFF), DPU_DATA_CUBE_WIDTH);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_DATA_CUBE_HEIGHT);              /* rows-1 = 0 */
  ops[i++] = NPUOP(OP_REG_DPU, (7u<<16) | (7u<<0), DPU_DATA_CUBE_CHANNEL); /* C=8 */

  /* BS fully bypassed; OW writer: OD_BYPASS only. */
  ops[i++] = NPUOP(OP_REG_DPU, (1u<<6) | (1u<<4) | (1u<<1) | (1u<<0), DPU_BS_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, (1u<<1), DPU_BS_OW_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, (7u & 0x1FFF), DPU_WDMA_SIZE_0);        /* channel_wdma 7 */
  ops[i++] = NPUOP(OP_REG_DPU, (width & 0x1FFF), DPU_WDMA_SIZE_1);     /* width_wdma; h 0 */

  /* BN stage: ALU bias word + MUL by the index scale (this is what maps the
   * fp16 input onto the LUT index grid). RELU bypassed; ALU_ALGO 2. */
  ops[i++] = NPUOP(OP_REG_DPU, (2u<<16) | (1u<<6), DPU_BN_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, p->bn_alu_cfg, DPU_BN_ALU_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)p->bn_mul_operand & 0xFFFF) << 16, DPU_BN_MUL_CFG);

  /* EW stage: RELU + OP_CVT bypassed, but EW_LUT_BYPASS left CLEAR (bit7=0) so
   * the LUT runs here. EW op cvt scale = 1. */
  ops[i++] = NPUOP(OP_REG_DPU, (1u<<9) | (1u<<8), DPU_EW_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x1, DPU_EW_CVT_SCALE_VALUE);

  /* Output converter: fixed-point LUT result -> fp16. */
  ops[i++] = NPUOP(OP_REG_DPU, p->out_cvt_offset, DPU_OUT_CVT_OFFSET);
  ops[i++] = NPUOP(OP_REG_DPU, (1u<<16) | ((uint32_t)p->out_cvt_scale & 0xFFFF), DPU_OUT_CVT_SCALE);
  ops[i++] = NPUOP(OP_REG_DPU,
                   (((uint32_t)p->out_cvt_cvt_type & 0x1) << 31) |
                   (((uint32_t)p->out_cvt_minus_exp & 0xFF) << 12), DPU_OUT_CVT_SHIFT);

  ops[i++] = NPUOP(OP_REG_DPU, (cols & 0xFFFFFFF) << 4, DPU_SURFACE_ADD);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_40C4);

  /* LUT control: hybrid LE+LO mux, index selects, table range + extrapolation. */
  ops[i++] = NPUOP(OP_REG_DPU, (1u<<6) | (1u<<5) | ((2u & 0x3) << 2), DPU_LUT_CFG);
  ops[i++] = NPUOP(OP_REG_DPU,
                   (((uint32_t)p->lo_index_select & 0xFF) << 16) |
                   (((uint32_t)p->le_index_select & 0xFF) << 8), DPU_LUT_INFO);
  ops[i++] = NPUOP(OP_REG_DPU, p->lut_le_start, DPU_LUT_LE_START);
  ops[i++] = NPUOP(OP_REG_DPU, p->lut_lo_end,   DPU_LUT_LO_END);
  ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)p->le_slope_scale & 0xFFFF), DPU_LUT_LE_SLOPE_SCALE);
  ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)p->le_slope_shift & 0x1F),  DPU_LUT_LE_SLOPE_SHIFT);
  /* LO (overflow / x > table) slope extrapolation — e.g. hardswish's linear x>3 tail
   * (slope 1 in x). Left 0 for the saturating [0,1] kinds (sigmoid/tanh upper tail). */
  ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)p->lo_slope_scale & 0xFFFF), DPU_LUT_LO_SLOPE_SCALE);
  ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)p->lo_slope_shift & 0x1F),  DPU_LUT_LO_SLOPE_SHIFT);

  /* DPU_RDMA: the input feed. MRDMA flying, fp16->fp32 on read, ERDMA disabled. */
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (width & 0x1FFF), DPU_RDMA_DATA_CUBE_WIDTH);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0,              DPU_RDMA_DATA_CUBE_HEIGHT);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (7u & 0x1FFF),    DPU_RDMA_DATA_CUBE_CHANNEL);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (uint32_t)p->input_dma, DPU_RDMA_SRC_BASE_ADDR);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, RDMA_ERDMA_DISABLE,     DPU_RDMA_ERDMA_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA,
                   RDMA_FMC_IN_PRECISION(precision_float16) | RDMA_FMC_BURST_LEN(15) |
                   RDMA_FMC_PROC_PRECISION(precision_float16) | RDMA_FMC_MRDMA_FP16TOFP32(1) |
                   (1u<<0) /* FLYING_MODE */, DPU_RDMA_FEATURE_MODE_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, RDMA_WEIGHT_ALL1, DPU_RDMA_WEIGHT);

  /* PC trailer (rocket framing). enable = DPU|DPU_RDMA block bits. */
  ops[i++] = NPUOP(OP_NONE, 0x0, 0x0);
  ops[i++] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
  ops[i++] = NPUOP(OP_40, 0x0, 0x0);
  ops[i++] = NPUOP(OP_ENABLE, enable, PC_OPERATION_ENABLE);

  p->task_count = i;
  return 0;
}

/* ---------------------------------------------------------------------------
 * DPU elementwise fp16 multiply (flying mode) — gen_ew_mul_fp16.
 *
 * out = A (.*) B, both fp16 cubes, computed on the DPU EW unit with NO conv.
 * The DPU main feed (MRDMA, SRC_BASE=A) and the EW operand feed (ERDMA,
 * EW_BASE=B) are combined by the EW MUL sub-unit: EW_OP_TYPE=1 (mul),
 * EW_OP_SRC=1 (operand from ERDMA), per-pixel EW_DATA_MODE, 16-bit EDATA_SIZE.
 * The EW_OP_TYPE=1 mul word is HW-validated; framing matches the LUT op.
 * Like the LUT op the result is layout-agnostic (out[i]=A[i]*B[i]); the runtime
 * feeds flat fp16 vectors. --------------------------------------------------- */
int gen_ew_mul_fp16(ew_mul_params_t *p)
{
  if (p->n <= 0 || (p->n & 0x7)) return -1;
  uint32_t cols = (uint32_t)p->n / 8;
  if (cols > 0x1FFF) return -2;
  uint32_t width  = cols - 1;
  uint32_t stride = cols * 2;                 /* mul surf-stride field */

  uint32_t sptr   = 0xE;
  uint32_t enable = 0x18;                      /* DPU + DPU_RDMA block bits     */
  const char *e;
  if ((e = getenv("ROCKET_LUT_SPTR")))   sptr   = (uint32_t)strtoul(e, NULL, 0);
  if ((e = getenv("ROCKET_LUT_ENABLE"))) enable = (uint32_t)strtoul(e, NULL, 0);

  uint64_t *ops = p->tasks;
  int i = 0;

  ops[i++] = NPUOP(OP_REG_DPU,      sptr, DPU_S_POINTER);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, sptr, DPU_RDMA_S_POINTER);

  ops[i++] = NPUOP(OP_REG_DPU, (15u<<5) | (2u<<1) | (1u<<0), DPU_FEATURE_MODE_CFG);
  ops[i++] = NPUOP(OP_REG_DPU,
                   ((uint32_t)precision_float16 << 29) | ((uint32_t)precision_float16 << 26) |
                   ((uint32_t)precision_float16 << 0), DPU_DATA_FORMAT);
  ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)p->output_dma, DPU_DST_BASE_ADD);
  ops[i++] = NPUOP(OP_REG_DPU, (stride & 0xFFFFFFF) << 4, DPU_DST_SURF_STRIDE);
  ops[i++] = NPUOP(OP_REG_DPU, (width & 0x1FFF), DPU_DATA_CUBE_WIDTH);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_DATA_CUBE_HEIGHT);
  ops[i++] = NPUOP(OP_REG_DPU, (7u<<0), DPU_DATA_CUBE_CHANNEL);   /* C=8, orig 0 */

  ops[i++] = NPUOP(OP_REG_DPU, (7u & 0x1FFF), DPU_WDMA_SIZE_0);
  ops[i++] = NPUOP(OP_REG_DPU, (width & 0x1FFF), DPU_WDMA_SIZE_1);
  ops[i++] = NPUOP(OP_REG_DPU, (1u<<1), DPU_BS_OW_CFG);           /* OD_BYPASS */
  ops[i++] = NPUOP(OP_REG_DPU, (1u<<6) | (1u<<4) | (1u<<1) | (1u<<0), DPU_BS_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, (1u<<6) | (1u<<4) | (1u<<1) | (1u<<0), DPU_BN_CFG);

  /* EW MUL: per-pixel mode, 16-bit, RELU/OP_CVT/LUT bypassed, operand from
   * ERDMA (EW_OP_SRC=1), op = multiply (EW_OP_TYPE=1); EW/EW_OP not bypassed. */
  ops[i++] = NPUOP(OP_REG_DPU,
                   (1u<<28) | (2u<<22) | (1u<<9) | (1u<<8) | (1u<<7) | (1u<<6) | (1u<<2),
                   DPU_EW_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x1, DPU_EW_CVT_SCALE_VALUE);
  ops[i++] = NPUOP(OP_REG_DPU, (stride & 0xFFFFFFF) << 4, DPU_SURFACE_ADD);

  /* DPU_RDMA: A via MRDMA (SRC_BASE), B via ERDMA (EW_BASE). */
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (width & 0x1FFF), DPU_RDMA_DATA_CUBE_WIDTH);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0,              DPU_RDMA_DATA_CUBE_HEIGHT);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (7u & 0x1FFF),    DPU_RDMA_DATA_CUBE_CHANNEL);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (uint32_t)p->input_a_dma, DPU_RDMA_SRC_BASE_ADDR);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (1u<<30) | (2u<<2), DPU_RDMA_ERDMA_CFG); /* per-pixel, 16b */
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (uint32_t)p->input_b_dma, DPU_RDMA_EW_BASE_ADDR);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (stride & 0xFFFFFFF) << 4, DPU_RDMA_EW_SURF_STRIDE);
  /* SURF_NOTCH = the planar pointer-advance for the ERDMA operand. Per the fp16
   * K-accum finding, leaving it 0 makes the ERDMA never advance (reads only the
   * offset-0 atom), so B is effectively unread. */
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (stride & 0xFFFFFFF) << 4, DPU_RDMA_SURF_NOTCH);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (stride & 0xFFFFFFF) << 4, DPU_RDMA_EW_SURF_NOTCH);
  /* COMB_USE(5) is what makes the EW combine the MRDMA main feed (A) with the
   * ERDMA operand (B) on the rocket stack — the same field the fp16 K-accum
   * add-path needs. Env-tunable for a HW sweep. */
  {
    uint32_t comb = 5;
    const char *ec = getenv("ROCKET_EW_MUL_COMB");
    if (ec) comb = (uint32_t)strtoul(ec, NULL, 0);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA,
                     RDMA_FMC_IN_PRECISION(precision_float16) | RDMA_FMC_BURST_LEN(15) |
                     RDMA_FMC_PROC_PRECISION(precision_float16) | RDMA_FMC_MRDMA_FP16TOFP32(1) |
                     RDMA_FMC_COMB_USE(comb) | (1u<<0) /* FLYING_MODE */,
                     DPU_RDMA_FEATURE_MODE_CFG);
  }
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, RDMA_WEIGHT_ALL1, DPU_RDMA_WEIGHT);

  ops[i++] = NPUOP(OP_NONE, 0x0, 0x0);
  ops[i++] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
  ops[i++] = NPUOP(OP_40, 0x0, 0x0);
  ops[i++] = NPUOP(OP_ENABLE, enable, PC_OPERATION_ENABLE);

  p->task_count = i;
  return 0;
}

/* ============================================================================
 * SECTION — Integer and alternate-dtype matmul generators
 *           (int8 / int4 / int16 / bf16 / tf32)
 * ==========================================================================*/

/*
 * int8 x int8 -> int32 matmul (as 1x1 conv).
 *
 * Wired to this file's shared, MRDMA-safe gen_matmul_task: with ew_accumulate=0
 * it takes the plain-conv path (MRDMA disabled, ERDMA disabled), byte-for-byte the
 * same control flow as the fp16 plain matmul. (A generator without the DPU_RDMA /
 * MRDMA_DISABLE block hits the "NPU job timed out, output untouched" trap — see the
 * file header.)
 *
 * Design: HW does int8xint8->int32, raw (cvt_bypass=1, data_sign=1); the host
 * applies per-row activation + per-channel weight scales on dequant. K-tiling
 * accumulates the int32 partials on the HOST (each tile is a plain, non-
 * accumulating int8->int32 matmul), so every production caller keeps
 * ew_accumulate=0. (accumulate=1 arms an int32 EW add that is HW-DEAD — see the
 * ew_accumulate block below; it survives only as the
 * matmul_accum_int8_rocket.c sweep harness and is never taken in production.)
 *
 * Deltas vs gen_matmul_fp16 (everything else is identical):
 *   CNA:  in/proc_precision fp16->int8; element size 2 B->1 B; data_entries /32->/64
 *   CORE: proc_precision fp16->int8; qd_en 1->0
 *   DPU:  out_precision = int32; in/proc_precision = int8; fp32tofp16_en = 0
 *         size_e = 7, surf_add = stride*8  <-- see WARN below
 *
 * size_e / surf_add WARN — the int8 path uses size_e=7 / surf_add*8, despite
 * looking inconsistent with the fp16 path's "size_e=bytes-1, surf_add mult=bytes"
 * encoding (which would predict size_e=3, surf_add*4 for int32). 7/8 is HW-correct
 * and bit-exact; the fp16-pattern 3/4 halves the surface stride, leaving every
 * output column past the first few surfaces as the 0xAA sentinel (the NPU never
 * writes them). So int8's int32 output is strided as if 8 bytes/elem — do NOT
 * "fix" it to 4. The ROCKET_INT8_SIZE_E / ROCKET_INT8_SURF_MULT knobs are kept
 * only as a diagnostic; the defaults are the correct values.
 */
int gen_matmul_int8(matmul_params_t *params)
{
   npu_cna_desc cna_desc;
   npu_core_desc core_desc;
   npu_dpu_desc dpu_desc = {0};   /* zero new output-writer fields; all used fields set below */

   unsigned int fd_bytes;
   unsigned int fd_banks;
   unsigned int weight_banks;
   int surf_stride;

   cna_desc.conv_mode = direct_convolution;
   cna_desc.in_precision = precision_int8;
   cna_desc.proc_precision = precision_int8;

   cna_desc.kernel_groups = 0;
   cna_desc.feature_grains = params->m+1;
   cna_desc.conv_x_stride = 1;
   cna_desc.conv_y_stride = 1;

   cna_desc.datain_width = 1;
   cna_desc.datain_height = params->m;
   cna_desc.datain_channel = params->k;
   cna_desc.dataout_width = 1;
   cna_desc.dataout_height = params->m;
   cna_desc.dataout_atomics = cna_desc.dataout_width * cna_desc.dataout_height;

   cna_desc.weight_width = 1;
   cna_desc.weight_height = 1;
   cna_desc.weight_kernels = params->n;
   cna_desc.weight_bytes_per_kernel = cna_desc.weight_width * cna_desc.weight_height *
     cna_desc.datain_channel * sizeof(int8_t);
   cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels;

   fd_bytes = cna_desc.datain_width * cna_desc.datain_height * cna_desc.datain_channel * sizeof(int8_t);
   fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
   fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks +1;
   weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
   weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE)==0) ? weight_banks : weight_banks + 1;
   if ((fd_banks) > NPU_CBUF_BANKS-1) {
     return -1;
   } else {
       if (cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE) {
        weight_banks = NPU_CBUF_BANKS - fd_banks;
       } else {
         return -2;
       }
   }

   /* RESONANCE FIX: the int8 C2=16 feature-cube DMA reads ONE CBUF bank past the
    * ceil(bytes/bank) allocation at resonant (Mtile,Ktile) geometries, garbling the
    * tail rows (HW-proven: ROCKET_I8_FDBANK_EXTRA sweep — +1 fixes the whole 2D bad
    * set 86->0, -1 worsens it). fp16's 2-byte cube doesn't hit it, so the original
    * code (exact-fit banks, reused from fp16) was latently wrong for int8 only.
    * Reserve a slack bank for the feature; weight takes the rest. The int8 planner
    * (rocket_matmul_plan_int8) reserves the matching bank so fd_banks+1 <= BANKS-1.
    * A caller that hand-built matmul_params_t and bypassed that planner could land
    * here with fd_banks == BANKS-1, where silently clamping data_bank back to fd_banks
    * would re-create the exact int8 tail-row corruption the slack prevents — so fail
    * loudly and unconditionally instead of clamping. */
   if (fd_banks + 1 > (unsigned)NPU_CBUF_BANKS - 1) {
       ROCKET_LOGE("gen_matmul_int8: int8 CBUF bank-slack does not fit "
               "(fd_banks=%u, need fd_banks+1 <= %d) — use rocket_matmul_plan_int8\n",
               fd_banks, NPU_CBUF_BANKS - 1);
       return -1;
   }
   {
       unsigned data_bank = fd_banks + 1;
       cna_desc.data_bank   = data_bank;
       cna_desc.weight_bank = NPU_CBUF_BANKS - data_bank;
   }
   cna_desc.weight_reuse = params->weight_reuse & 0x1;   /* default 0 */
   cna_desc.data_reuse   = params->data_reuse   & 0x1;
   cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / 64;
   cna_desc.data_entries = (((cna_desc.datain_width * cna_desc.datain_channel) % 64) == 0) ?
     cna_desc.data_entries : cna_desc.data_entries +1;
   cna_desc.data_sign = 0x1;
   cna_desc.cvt_type  = 0x1;
   cna_desc.cvt_bypass = 0x1;
   cna_desc.cvt_scale0 = 0x1;
   cna_desc.cvt_scale1 = 0x1;
   cna_desc.cvt_scale2 = 0x1;
   cna_desc.cvt_scale3 = 0x1;
   cna_desc.fc_skip_en = 0;
   cna_desc.data_offset = 0x0;
   cna_desc.pad_left = 0;
   cna_desc.pad_top = 0;
   cna_desc.feature_base_addr = params->input_dma;
   cna_desc.weight_offset = 0;
   cna_desc.weight_burst_len = 0xf;
   cna_desc.data_burst_len = 0xf;
   cna_desc.line_stride = cna_desc.datain_width * 4;
   surf_stride = cna_feature_surf_stride(cna_desc.line_stride, cna_desc.datain_height);
   cna_desc.surf_stride = surf_stride;
   cna_desc.dma_width = cna_desc.datain_width;
   cna_desc.dma_height = cna_desc.datain_height;
   cna_desc.dma_channel = cna_desc.datain_channel;
   cna_desc.decompress_addr0 = params->weights_dma;

   core_desc.proc_precision = precision_int8;
   core_desc.qd_en = 0;
   core_desc.dataout_height = cna_desc.dataout_height - 1;
   core_desc.dataout_width = cna_desc.dataout_width - 1;
   core_desc.dataout_channel = cna_desc.weight_kernels -1;

   /* ROCKET_INT8_FP32_OUT: emit the int8 conv's accumulator as fp32 instead of raw
    * int32. The int32-integer DPU-EW add is HW-unsupported (the EW is a float-only
    * unit), so the only remaining NPU K-accum route is fp32-EW — which requires the
    * int8 conv to CAST its int32 MAC accumulator to fp32 on output. When set we
    * switch the DPU output side to the float32 geometry (out_precision=float32,
    * size_e=3, surf_add=stride*4), leaving the int8 INPUT side (CNA C2=16, weight
    * k-group 32) untouched. Default unset = the raw-int32 path. */
   int fp32_out = getenv("ROCKET_INT8_FP32_OUT") != NULL;

   /* Dequant-output mode: instead of reading back the raw int32 accumulator
    * and dequantizing on the host (C[m,n] * a_scale[m] * b_scale[n]), have the DPU
    * OUT_CVT apply the scale on-chip and emit a FLOAT result (fp32 reuses the proven
    * fp32-out geometry; fp16 halves the readback — its 2-byte writer geometry is the
    * sweep). The OUT_CVT affine + the optional BN-MUL per-tensor scale are fully
    * env-overridable so tests/matmul_int8_dequant_rocket.c can classify the exact HW
    * semantics before a clean API is committed. Default unset = byte-identical. */
   int deq = getenv("ROCKET_INT8_DEQ") != NULL;
   int deq_prec = precision_float32;
   {
     const char *de = getenv("ROCKET_INT8_DEQ_PREC");
     if (deq && de) {
       if      (!strcmp(de, "fp16")) deq_prec = precision_float16;
       else if (!strcmp(de, "fp32")) deq_prec = precision_float32;
       else                          deq_prec = (int)strtoul(de, NULL, 0) & 0x7;
     }
   }

   dpu_desc.burst_len = 0xf;
   dpu_desc.conv_mode = direct_convolution;
   dpu_desc.output_mode = 0x2;
   dpu_desc.flying_mode = 0x0;
   dpu_desc.out_precision = deq ? (uint8_t)deq_prec
                                : (fp32_out ? precision_float32 : precision_int32);
   dpu_desc.in_precision = precision_int8;
   dpu_desc.proc_precision = precision_int8;
   dpu_desc.dst_base_addr = params->output_dma;
   dpu_desc.dst_surf_stride = cna_desc.dataout_height * cna_desc.dataout_width;
   dpu_desc.width = core_desc.dataout_width ;
   dpu_desc.height = core_desc.dataout_height;
   dpu_desc.channel = core_desc.dataout_channel;
   dpu_desc.bs_bypass = 1;
   dpu_desc.bs_alu_bypass = 1;
   dpu_desc.bs_mul_bypass = 1;
   dpu_desc.bs_relu_bypass = 1;
   dpu_desc.bn_bypass =1;
   dpu_desc.bn_alu_bypass = 1;
   dpu_desc.bn_mul_bypass = 1;
   dpu_desc.bn_relu_bypass = 1;
   dpu_desc.ew_bypass =1;
   dpu_desc.ew_op_bypass =1;
   dpu_desc.ew_lut_bypass =1;
   dpu_desc.ew_op_cvt_bypass =1;
   dpu_desc.ew_relu_bypass=1;
   dpu_desc.fp32tofp16_en =0;
   dpu_desc.out_cvt_scale =1;
   if (deq) {
     /* Fold the dequant scale into the DPU output converter. The effective output
      * (hypothesis, to be HW-confirmed) is
      *   out_float = (acc_i32 + offset) * out_cvt_scale * 2^-minus_exp
      * with cvt_type=1 selecting the float-affine decode and fp32tofp16_en narrowing
      * to fp16. Defaults give scale=1, minus_exp=0 (identity * int->float cast); the
      * gate sweeps SCALE/MINEXP/CVTTYPE/OFFSET/FP32TOFP16 + the alt BN-MUL operand. */
     const char *de;
     dpu_desc.fp32tofp16_en   = (deq_prec == precision_float16) ? 1 : 0;
     dpu_desc.out_cvt_scale   = 1;
     dpu_desc.out_cvt_minus_exp = 0;
     dpu_desc.out_cvt_cvt_type  = 1;
     dpu_desc.out_cvt_offset    = 0;
     if ((de = getenv("ROCKET_INT8_DEQ_SCALE")))      dpu_desc.out_cvt_scale     = (uint16_t)strtoul(de, NULL, 0);
     if ((de = getenv("ROCKET_INT8_DEQ_SHIFT")))      dpu_desc.out_cvt_shift     = (uint8_t)strtoul(de, NULL, 0);
     if ((de = getenv("ROCKET_INT8_DEQ_MINEXP")))     dpu_desc.out_cvt_minus_exp = (uint8_t)strtoul(de, NULL, 0);
     if ((de = getenv("ROCKET_INT8_DEQ_CVTTYPE")))    dpu_desc.out_cvt_cvt_type  = (uint8_t)strtoul(de, NULL, 0) & 0x1;
     if ((de = getenv("ROCKET_INT8_DEQ_OFFSET")))     dpu_desc.out_cvt_offset    = (uint32_t)strtol(de, NULL, 0);
     if ((de = getenv("ROCKET_INT8_DEQ_FP32TOFP16"))) dpu_desc.fp32tofp16_en     = (uint8_t)strtoul(de, NULL, 0) & 0x1;
     if ((de = getenv("ROCKET_INT8_DEQ_BNMUL"))) {
       dpu_desc.bn_mul_operand = (uint16_t)strtoul(de, NULL, 0);
       dpu_desc.bn_bypass = 0; dpu_desc.bn_mul_bypass = 0;   /* arm the BN per-tensor MUL */
     }
   }
   {
     /* size_e/surf values; see WARN in the function header. Env-tunable so the
      * standalone test can sweep 7/8 vs 3/4 (the fp16 bytes-per-elem pattern for
      * int32) without a recompile. */
     /* size_e=7 / surf*8 is the int8-CONV DATAPATH quirk (HW-proven for int32-out),
      * and it carries over to fp32-out unchanged: SIZE_E=7 passes bit-exact and
      * SIZE_E=3 (the float-native value) gives garbage — the output type is just a
      * reinterpret of the same 4-byte int8-conv write. */
     unsigned se = 7, sm = 8;
     const char *e;
     if ((e = getenv("ROCKET_INT8_SIZE_E")))    se = (unsigned)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_INT8_SURF_MULT"))) sm = (unsigned)strtoul(e, NULL, 0);
     dpu_desc.size_e_2 = se & 0x7;
     dpu_desc.size_e_1 = se & 0x7;
     dpu_desc.size_e_0 = se & 0x7;
     dpu_desc.surf_add = dpu_desc.dst_surf_stride * sm;
   }
   dpu_desc.od_bypass = 1;
   dpu_desc.width_wdma = core_desc.dataout_width;
   dpu_desc.height_wdma = core_desc.dataout_height;
   dpu_desc.channel_wdma = core_desc.dataout_channel;

   /* NPU-side K-accumulation for int8 — accumulate=0 (default) keeps the plain-conv
    * (MRDMA/ERDMA-disabled) path, which matches the host-int64 oracle and is
    * MRDMA-trap-safe. This is the ONLY path production callers ever take.
    *
    * accumulate=1 arms the int32 EW add below. This is HW-DEAD: the DPU EW
    * operand DMA is <=16-bit, so EDATA_SIZE=3 (32-bit) reads the int32 operand back
    * as garbage (a true add of two int32 tiles returned an fp16 inf/NaN bit pattern,
    * NOT the integer sum). It is NOT bit-exact and must never be used in production —
    * rocket_matmul_int8 does not reach it. The encoding + env knobs (ROCKET_INT8_*)
    * are retained ONLY as the sweep harness driven by the standalone classifier
    * tests/matmul_accum_int8_rocket.c, kept per the note for any future EW attempt.
    * Integer K-accumulation is HW-dead — do not reattempt. The geometry mirrors
    * fp16 (one atom = 16 bytes for both:
    * 8 fp16 x 2B == 4 int32 x 4B; C2=8 vs C2=4) only so the classifier can sweep it:
    *   EW_CFG     EDATA_SIZE 2(16b)->3(32b): 0x108202C0 -> 0x10C202C0
    *   ERDMA_CFG  ERDMA_DATA_SIZE 2->3:       0x40000008 -> 0x4000000C
    *   RDMA_FMC   IN/PROC precision fp16(2)->int32(4): 0x17D40 -> 0x27D80
    * (EDATA_SIZE<<22, ERDMA_DATA_SIZE<<2.) */
   dpu_desc.ew_accumulate = params->accumulate & 0x1;
   {
     uint32_t M = (uint32_t)params->m;            /* output_h; output_w == 1 */
     uint32_t hw = M;
     uint32_t ew_stride = hw > 12 ? hw : 12;      /* MAX(hw,12), geometric value */
     uint32_t base_off  = hw * 16;                /* one surface BYTES (4 int32 x 4B) */
     const char *e;
     if ((e = getenv("ROCKET_INT8_EW_SURF")))     ew_stride = (uint32_t)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_INT8_EW_BASE_OFF"))) base_off  = (uint32_t)strtoul(e, NULL, 0);
     uint32_t notch = ew_stride;
     if ((e = getenv("ROCKET_INT8_EW_NOTCH")))    notch = (uint32_t)strtoul(e, NULL, 0);
     dpu_desc.ew_src_base_addr = params->add_dma;             /* MRDMA, no offset */
     dpu_desc.ew_base_addr     = params->add_dma + base_off;  /* ERDMA, +1 surface */
     dpu_desc.ew_surf_stride   = ew_stride << 4;
     dpu_desc.ew_surf_notch    = notch << 4;
   }
   {
     /* int32-out: EW float-add bits-as-float is WRONG (int32 EW unsupported).
      * fp32-out: operands are genuine fp32, so the float ALU-add
      * (0x10C202C0 = EW_DATA_MODE|EDATA_SIZE(3=32b)|EW_ALU_ALGO(2=add)|...) is
      * exactly right, with float32 RDMA precision (precision_float32=5). */
     uint32_t ew = EW_CFG_ADD_32B, erd = ERDMA_CFG_32B;
     uint32_t prec = fp32_out ? (uint32_t)precision_float32 : (uint32_t)precision_int32;
     uint32_t fmc = RDMA_FMC_BURST_LEN(15) | RDMA_FMC_COMB_USE(5) |
                    RDMA_FMC_IN_PRECISION(prec) |
                    RDMA_FMC_PROC_PRECISION(prec);
     const char *e;
     if ((e = getenv("ROCKET_INT8_EW_CFG")))    ew  = (uint32_t)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_INT8_ERDMA_CFG"))) erd = (uint32_t)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_INT8_RDMA_FMC")))  fmc = (uint32_t)strtoul(e, NULL, 0);
     dpu_desc.ew_cfg_val    = ew;
     dpu_desc.erdma_cfg_val = erd;
     dpu_desc.rdma_fmc_val  = fmc;
   }

   /* ---- DEBUG sentinel-sweep knobs (resonant-tile corruption RE). The int8
    * feature-DMA garbles the tail rows at specific (Mtile,Ktile). These let the
    * standalone test sweep the suspect CNA feature-DMA fields on HW without a
    * recompile, to localize the corrupted formula. Default unset = unchanged. */
   {
     const char *e;
     if ((e = getenv("ROCKET_I8_FDBANK_EXTRA"))) {
       int x = atoi(e);
       int nb = (int)cna_desc.data_bank + x;
       if (nb < 1) nb = 1;
       if (nb > (int)NPU_CBUF_BANKS - 1) nb = NPU_CBUF_BANKS - 1;
       cna_desc.data_bank   = nb;
       cna_desc.weight_bank = NPU_CBUF_BANKS - nb;
     }
     if ((e = getenv("ROCKET_I8_GRAINS")))     cna_desc.feature_grains = (unsigned)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_I8_SURF")))       cna_desc.surf_stride    = (int)strtol(e, NULL, 0);
     if ((e = getenv("ROCKET_I8_ENTRIES")))    cna_desc.data_entries   = (unsigned)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_I8_LINESTRIDE"))) cna_desc.line_stride    = (unsigned)strtoul(e, NULL, 0);
   }

   {
     int rc = gen_matmul_task(params->tasks, &cna_desc, &core_desc, &dpu_desc);
     if (rc < 0) return rc;
     params->task_count = (uint32_t)rc;
   }

   return 0;
}

/* ============================================================================
 * gen_matmul_int4 — int4 x int4 -> int16 single-task generator.
 *
 * Every uncertain register field is env-overridable so matmul_int4_rocket can
 * sweep + sentinel-classify it on HW (the same method that cracked the int8
 * size_e=7 quirk); the defaults below are HW-confirmed bit-exact. RK3588 int4
 * matmul facts:
 *   - int4(A) x int4(B) -> INT16(C)  (output is int16, NOT int32 like int8)
 *   - A native layout (K/32, M, 32): feature cube C2=32 (2x int8's 16)
 *   - K align 32, N align 64; K <= 10240 single-pass
 *   - 4x the fp16 MAC; needs Hadamard rotation
 *
 * Deltas vs gen_matmul_int8 (the structural sibling):
 *   precision: in/proc = int4 (ROCKET_INT4_PREC); DPU out = int16 (ROCKET_INT4_OUT_PREC)
 *   byte math: int4 = 0.5 B/elem -> weight_bytes = N*K/2, fd_bytes = M*K/2
 *   data_entries divisor: 128 (ROCKET_INT4_DENTRIES_DIV; int8 was /64, int4 2x density)
 *   output geom: int16 output follows the same integer-conv stride quirk as int8's
 *              int32 (size_e=7, surf*8, NOT the 2-byte 1/2 pattern); ROCKET_INT4_SIZE_E
 *              / ROCKET_INT4_SURF_MULT remain as diagnostics
 * The feature DMA strides (line_stride/surf_stride) are M-driven (datain_width=1),
 * so they carry over from int8 unchanged. Plain-conv path only (ew_accumulate=0,
 * MRDMA/ERDMA disabled — MRDMA-trap-safe). The single-task path never
 * K-accumulates, so no EW geometry is needed here. */
int gen_matmul_int4(matmul_params_t *params)
{
   npu_cna_desc cna_desc;
   npu_core_desc core_desc;
   npu_dpu_desc dpu_desc = {0};   /* zero new output-writer fields; all used fields set below */

   unsigned int fd_bytes, fd_banks, weight_banks;
   int surf_stride;
   const char *e;

   /* HW-confirmed encodings (matmul_int4_rocket, bit-exact). Kept env-overridable
    * as diagnostics for the tiling/K-accum bring-up.
    *   in/proc precision = int4 = 6 (the only value giving correct int4 MAC)
    *   out precision     = int16 = 1
    *   size_e = 7: integer-conv outputs stride size_e=7 (SAME quirk as int8's
    *               int32 output; SIZE_E 1->16 cols, 3->32, 7->all 64). surf*8
    *               like int8 (4/8/16 all passed at M=4; 8 matches the int8 path). */
   unsigned in_prec  = precision_int4;
   unsigned out_prec = precision_int16;
   unsigned se = 7, sm = 8;
   unsigned dentries_div = 128;
   if ((e = getenv("ROCKET_INT4_PREC")))         in_prec      = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT4_OUT_PREC")))     out_prec     = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT4_SIZE_E")))       se           = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT4_SURF_MULT")))    sm           = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT4_DENTRIES_DIV"))) dentries_div = (unsigned)strtoul(e, NULL, 0);
   if (dentries_div == 0) dentries_div = 128;

   cna_desc.conv_mode = direct_convolution;
   cna_desc.in_precision = in_prec & 0x7;
   cna_desc.proc_precision = in_prec & 0x7;

   cna_desc.kernel_groups = 0;
   cna_desc.feature_grains = params->m + 1;
   cna_desc.conv_x_stride = 1;
   cna_desc.conv_y_stride = 1;

   cna_desc.datain_width = 1;
   cna_desc.datain_height = params->m;
   cna_desc.datain_channel = params->k;
   cna_desc.dataout_width = 1;
   cna_desc.dataout_height = params->m;
   cna_desc.dataout_atomics = cna_desc.dataout_width * cna_desc.dataout_height;

   cna_desc.weight_width = 1;
   cna_desc.weight_height = 1;
   cna_desc.weight_kernels = params->n;
   /* int4 = 0.5 B/elem: weight bytes = W*H*K/2 per kernel (K even, K%32==0). */
   cna_desc.weight_bytes_per_kernel =
       (cna_desc.weight_width * cna_desc.weight_height * cna_desc.datain_channel + 1) / 2;
   cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels;

   fd_bytes = (cna_desc.datain_width * cna_desc.datain_height * cna_desc.datain_channel + 1) / 2;
   fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
   fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks + 1;
   weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
   weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE) == 0) ? weight_banks : weight_banks + 1;
   if (fd_banks > NPU_CBUF_BANKS - 1) {
     return -1;
   } else {
       if (cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE) {
        weight_banks = NPU_CBUF_BANKS - fd_banks;
       } else {
         return -2;
       }
   }

   cna_desc.weight_bank = weight_banks;
   cna_desc.data_bank = fd_banks;
   cna_desc.weight_reuse = params->weight_reuse & 0x1;
   cna_desc.data_reuse   = params->data_reuse   & 0x1;
   cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / dentries_div;
   cna_desc.data_entries = (((cna_desc.datain_width * cna_desc.datain_channel) % dentries_div) == 0) ?
     cna_desc.data_entries : cna_desc.data_entries + 1;
   cna_desc.data_sign = 0x1;
   cna_desc.cvt_type  = 0x1;
   cna_desc.cvt_bypass = 0x1;
   cna_desc.cvt_scale0 = 0x1;
   cna_desc.cvt_scale1 = 0x1;
   cna_desc.cvt_scale2 = 0x1;
   cna_desc.cvt_scale3 = 0x1;
   cna_desc.fc_skip_en = 0;
   cna_desc.data_offset = 0x0;
   cna_desc.pad_left = 0;
   cna_desc.pad_top = 0;
   cna_desc.feature_base_addr = params->input_dma;
   cna_desc.weight_offset = 0;
   cna_desc.weight_burst_len = 0xf;
   cna_desc.data_burst_len = 0xf;
   cna_desc.line_stride = cna_desc.datain_width * 4;
   surf_stride = cna_feature_surf_stride(cna_desc.line_stride, cna_desc.datain_height);
   cna_desc.surf_stride = surf_stride;
   cna_desc.dma_width = cna_desc.datain_width;
   cna_desc.dma_height = cna_desc.datain_height;
   cna_desc.dma_channel = cna_desc.datain_channel;
   cna_desc.decompress_addr0 = params->weights_dma;

   core_desc.proc_precision = in_prec & 0x7;
   core_desc.qd_en = 0;
   core_desc.dataout_height = cna_desc.dataout_height - 1;
   core_desc.dataout_width = cna_desc.dataout_width - 1;
   core_desc.dataout_channel = cna_desc.weight_kernels - 1;

   dpu_desc.burst_len = 0xf;
   dpu_desc.conv_mode = direct_convolution;
   dpu_desc.output_mode = 0x2;
   dpu_desc.flying_mode = 0x0;
   dpu_desc.out_precision = out_prec & 0x7;
   dpu_desc.in_precision = in_prec & 0x7;
   dpu_desc.proc_precision = in_prec & 0x7;
   dpu_desc.dst_base_addr = params->output_dma;
   dpu_desc.dst_surf_stride = cna_desc.dataout_height * cna_desc.dataout_width;
   dpu_desc.width = core_desc.dataout_width;
   dpu_desc.height = core_desc.dataout_height;
   dpu_desc.channel = core_desc.dataout_channel;
   dpu_desc.bs_bypass = 1;
   dpu_desc.bs_alu_bypass = 1;
   dpu_desc.bs_mul_bypass = 1;
   dpu_desc.bs_relu_bypass = 1;
   dpu_desc.bn_bypass = 1;
   dpu_desc.bn_alu_bypass = 1;
   dpu_desc.bn_mul_bypass = 1;
   dpu_desc.bn_relu_bypass = 1;
   dpu_desc.ew_bypass = 1;
   dpu_desc.ew_op_bypass = 1;
   dpu_desc.ew_lut_bypass = 1;
   dpu_desc.ew_op_cvt_bypass = 1;
   dpu_desc.ew_relu_bypass = 1;
   dpu_desc.fp32tofp16_en = 0;
   dpu_desc.out_cvt_scale = 1;
   dpu_desc.size_e_2 = se & 0x7;
   dpu_desc.size_e_1 = se & 0x7;
   dpu_desc.size_e_0 = se & 0x7;
   dpu_desc.surf_add = dpu_desc.dst_surf_stride * sm;
   dpu_desc.od_bypass = 1;
   dpu_desc.width_wdma = core_desc.dataout_width;
   dpu_desc.height_wdma = core_desc.dataout_height;
   dpu_desc.channel_wdma = core_desc.dataout_channel;

   /* Plain-conv path (no NPU K-accum): MRDMA/ERDMA disabled, MRDMA-trap-safe
    * control flow. int4's int16 output IS <=16-bit, so DPU-EW K-accum is FEASIBLE
    * here later (unlike int8) — deferred to the tiling step; the single-task path
    * does one K-pass, no accumulation. */
   dpu_desc.ew_accumulate = 0;
   {
     uint32_t M = (uint32_t)params->m;
     uint32_t hw = M;
     uint32_t ew_stride = hw > 12 ? hw : 12;
     dpu_desc.ew_src_base_addr = params->add_dma;
     dpu_desc.ew_base_addr     = params->add_dma + hw * 16;
     dpu_desc.ew_surf_stride   = ew_stride << 4;
     dpu_desc.ew_surf_notch    = ew_stride << 4;
     dpu_desc.ew_cfg_val    = EW_CFG_ALU(EW_ALU_ALGO_ADD);
     dpu_desc.erdma_cfg_val = ERDMA_CFG_16B;
     dpu_desc.rdma_fmc_val  = RDMA_FMC_BURST_LEN(15) | RDMA_FMC_COMB_USE(5) |
                              RDMA_FMC_IN_PRECISION(precision_float16) |
                              RDMA_FMC_PROC_PRECISION(precision_float16);
   }

   {
     int rc = gen_matmul_task(params->tasks, &cna_desc, &core_desc, &dpu_desc);
     if (rc < 0) return rc;
     params->task_count = (uint32_t)rc;
   }

   return 0;
}

/* ============================================================================
 * gen_matmul_int16 — int16 x int16 -> int32 single-task generator.
 *
 * RK3588 documents matmul modes only for fp16/int8/int4 — there is NO documented int16xint16 matmul mode. So int16 is
 * reverse-engineered from scratch, triangulated from two analogs:
 *   - fp16 for the GEOMETRY (int16 is 2 bytes, same as fp16): feature cube C2=8,
 *     weight layout (N/16,K/32,16,32) [weight_int16 == weight_fp16 reinterpreted],
 *     N-align 16, banks_for x2, data_entries /32, data_sign=1.
 *   - int8 for the INTEGER-OUTPUT handling: int16xint16 accumulates in the 48-bit
 *     CACC and writes int32; output cube C2=4, size_e=7 (the integer-output quirk,
 *     HW-confirmed for int8's int32 AND int4's int16 — almost certainly int16's
 *     int32 too), surf*8, host int64 K-accum (DPU-EW is DEAD: int32 partials
 *     exceed the EW's <=16-bit operand DMA, like int8 — do NOT attempt EW K-accum).
 *
 * PRIME HYPOTHESIS (sweep each on HW with a small shape):
 *   precision in/proc = precision_int16 = 1 (UNCONFIRMED on this path — recall
 *     int4's "obvious" 3 is wrong, 6 is right; rule out 3/7).
 *   DPU out precision = int32 = 4.
 *   size_e = 7, surf_mult = 8 (the integer-output quirk).
 *   data_entries divisor = 32 (2-byte element, like fp16).
 * Every uncertain field is env-overridable so matmul_int16_rocket can crack it on
 * HW by sweep + sentinel classification. Bake the confirmed values as defaults.
 *
 * int32-output SATURATION: int16xint16 <= 32767^2 ~= 1.07e9 per product, ~half
 * of int32 max — a sum of just TWO full-range products overflows int32. So a
 * bit-exact test must use SMALL int16 magnitudes; the gen is unaffected (it just
 * writes whatever the conv accumulates), but the tiling/backend must keep each
 * Kt-pass within int32. See matmul_int16_rocket for the overflow characterization.
 *
 * Plain-conv path only (ew_accumulate=0, MRDMA/ERDMA disabled — MRDMA-trap-safe).
 * The single-task path does one K-pass. */
int gen_matmul_int16(matmul_params_t *params)
{
   npu_cna_desc cna_desc;
   npu_core_desc core_desc;
   npu_dpu_desc dpu_desc = {0};   /* zero new output-writer fields; all used fields set below */

   unsigned int fd_bytes, fd_banks, weight_banks;
   int surf_stride;
   const char *e;

   /* HYPOTHESIS encodings (sweep on HW). Kept env-overridable as diagnostics.
    *   in/proc precision = int16 = 1 (rule out 3/7 by sweep)
    *   out precision     = int32 = 4 (int16xint16 -> int32, like int8's output)
    *   size_e = 7 / surf*8: the integer-conv output quirk (HW-proven for int8's
    *               int32 and int4's int16; near-certain for int16's int32 too)
    *   data_entries /32: 2-byte element, like fp16 (int8 /64, int4 /128). */
   unsigned in_prec  = precision_int16;
   unsigned out_prec = precision_int32;
   unsigned se = 7, sm = 8;
   unsigned dentries_div = 32;
   /* qd_en = 1 — HW-CONFIRMED REQUIRED for int16. Initially defaulted to 0
    * (copying int8), which on HW TRUNCATED the conv to one weight N-group
    * (16 output channels) regardless of M/N. The conv path must set
    * CORE_MISC_CFG_QD_EN(1) unconditionally (DW_EN is depthwise-only), and the fp16
    * path uses 1. int8's qd_en=0 happens to be benign for int8 but breaks int16's
    * N-group-16 iteration. Keep the ROCKET_INT16_QD_EN knob as a diagnostic. */
   unsigned qd_en = 1;
   if ((e = getenv("ROCKET_INT16_PREC")))         in_prec      = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT16_OUT_PREC")))     out_prec     = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT16_SIZE_E")))       se           = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT16_SURF_MULT")))    sm           = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT16_DENTRIES_DIV"))) dentries_div = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT16_QD_EN")))        qd_en        = (unsigned)strtoul(e, NULL, 0);
   if (dentries_div == 0) dentries_div = 32;
   /* CNA_CONV_CON2 iteration fields. On HW int16 (precision=1) emits only ONE
    * output tile (1 row x 16 ch) regardless of M/N — int8/int4 iterate fully with
    * the same descriptor. feature_grains (input rows/grains processed) and
    * kernel_groups (output-channel groups) are the prime suspects for that
    * iteration; sweep them. -1 = use the default (M+1 / 0, like fp16/int8). */
   int grains_ovr = -1, kgroups_ovr = -1;
   if ((e = getenv("ROCKET_INT16_GRAINS")))  grains_ovr  = (int)strtol(e, NULL, 0);
   if ((e = getenv("ROCKET_INT16_KGROUPS"))) kgroups_ovr = (int)strtol(e, NULL, 0);
   /* DPU output-writer controls (the "stuck on one 1x16 tile" cluster the
    * iteration-register sweep pointed at). All default 0 = current behaviour. */
   unsigned mc_surf = 0, tp_prec = 0, tp_org = 0, size_c = 0;
   if ((e = getenv("ROCKET_INT16_MC_SURF"))) mc_surf = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT16_TP_PREC"))) tp_prec = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT16_TP_ORG")))  tp_org  = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_INT16_SIZE_C")))  size_c  = (unsigned)strtoul(e, NULL, 0);

   cna_desc.conv_mode = direct_convolution;
   cna_desc.in_precision = in_prec & 0x7;
   cna_desc.proc_precision = in_prec & 0x7;

   cna_desc.kernel_groups = (kgroups_ovr >= 0) ? (unsigned)kgroups_ovr : 0;
   cna_desc.feature_grains = (grains_ovr > 0) ? (unsigned)grains_ovr : (unsigned)(params->m + 1);
   cna_desc.conv_x_stride = 1;
   cna_desc.conv_y_stride = 1;

   cna_desc.datain_width = 1;
   cna_desc.datain_height = params->m;
   cna_desc.datain_channel = params->k;
   cna_desc.dataout_width = 1;
   cna_desc.dataout_height = params->m;
   cna_desc.dataout_atomics = cna_desc.dataout_width * cna_desc.dataout_height;

   cna_desc.weight_width = 1;
   cna_desc.weight_height = 1;
   cna_desc.weight_kernels = params->n;
   /* int16 = 2 B/elem (like fp16): weight bytes = W*H*K * 2 per kernel. */
   cna_desc.weight_bytes_per_kernel = cna_desc.weight_width * cna_desc.weight_height *
     cna_desc.datain_channel * sizeof(int16_t);
   cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels;

   fd_bytes = cna_desc.datain_width * cna_desc.datain_height * cna_desc.datain_channel * sizeof(int16_t);
   fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
   fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks + 1;
   weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
   weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE) == 0) ? weight_banks : weight_banks + 1;
   if (fd_banks > NPU_CBUF_BANKS - 1) {
     return -1;
   } else {
       if (cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE) {
        weight_banks = NPU_CBUF_BANKS - fd_banks;
       } else {
         return -2;
       }
   }

   cna_desc.weight_bank = weight_banks;
   cna_desc.data_bank = fd_banks;
   cna_desc.weight_reuse = params->weight_reuse & 0x1;
   cna_desc.data_reuse   = params->data_reuse   & 0x1;
   cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / dentries_div;
   cna_desc.data_entries = (((cna_desc.datain_width * cna_desc.datain_channel) % dentries_div) == 0) ?
     cna_desc.data_entries : cna_desc.data_entries + 1;
   cna_desc.data_sign = 0x1;
   cna_desc.cvt_type  = 0x1;
   cna_desc.cvt_bypass = 0x1;
   cna_desc.cvt_scale0 = 0x1;
   cna_desc.cvt_scale1 = 0x1;
   cna_desc.cvt_scale2 = 0x1;
   cna_desc.cvt_scale3 = 0x1;
   cna_desc.fc_skip_en = 0;
   cna_desc.data_offset = 0x0;
   cna_desc.pad_left = 0;
   cna_desc.pad_top = 0;
   cna_desc.feature_base_addr = params->input_dma;
   cna_desc.weight_offset = 0;
   cna_desc.weight_burst_len = 0xf;
   cna_desc.data_burst_len = 0xf;
   cna_desc.line_stride = cna_desc.datain_width * 4;
   surf_stride = cna_feature_surf_stride(cna_desc.line_stride, cna_desc.datain_height);
   cna_desc.surf_stride = surf_stride;
   cna_desc.dma_width = cna_desc.datain_width;
   cna_desc.dma_height = cna_desc.datain_height;
   cna_desc.dma_channel = cna_desc.datain_channel;
   cna_desc.decompress_addr0 = params->weights_dma;

   core_desc.proc_precision = in_prec & 0x7;
   core_desc.qd_en = qd_en & 0x1;             /* ROCKET_INT16_QD_EN sweep (see above) */
   core_desc.dataout_height = cna_desc.dataout_height - 1;
   core_desc.dataout_width = cna_desc.dataout_width - 1;
   core_desc.dataout_channel = cna_desc.weight_kernels - 1;

   dpu_desc.burst_len = 0xf;
   dpu_desc.conv_mode = direct_convolution;
   dpu_desc.output_mode = 0x2;
   dpu_desc.flying_mode = 0x0;
   dpu_desc.out_precision = out_prec & 0x7;
   dpu_desc.in_precision = in_prec & 0x7;
   dpu_desc.proc_precision = in_prec & 0x7;
   dpu_desc.dst_base_addr = params->output_dma;
   dpu_desc.dst_surf_stride = cna_desc.dataout_height * cna_desc.dataout_width;
   dpu_desc.width = core_desc.dataout_width;
   dpu_desc.height = core_desc.dataout_height;
   dpu_desc.channel = core_desc.dataout_channel;
   dpu_desc.bs_bypass = 1;
   dpu_desc.bs_alu_bypass = 1;
   dpu_desc.bs_mul_bypass = 1;
   dpu_desc.bs_relu_bypass = 1;
   dpu_desc.bn_bypass = 1;
   dpu_desc.bn_alu_bypass = 1;
   dpu_desc.bn_mul_bypass = 1;
   dpu_desc.bn_relu_bypass = 1;
   dpu_desc.ew_bypass = 1;
   dpu_desc.ew_op_bypass = 1;
   dpu_desc.ew_lut_bypass = 1;
   dpu_desc.ew_op_cvt_bypass = 1;
   dpu_desc.ew_relu_bypass = 1;
   dpu_desc.fp32tofp16_en = 0;
   dpu_desc.out_cvt_scale = 1;
   dpu_desc.size_e_2 = se & 0x7;
   dpu_desc.size_e_1 = se & 0x7;
   dpu_desc.size_e_0 = se & 0x7;
   dpu_desc.surf_add = dpu_desc.dst_surf_stride * sm;
   dpu_desc.od_bypass = 1;
   dpu_desc.width_wdma = core_desc.dataout_width;
   dpu_desc.height_wdma = core_desc.dataout_height;
   dpu_desc.channel_wdma = core_desc.dataout_channel;

   /* DPU output-writer controls — the unswept register cluster the iteration
    * sweep pointed at for the int16 "one 1x16 tile" truncation. Default 0 (current
    * behaviour); sweep via ROCKET_INT16_MC_SURF / TP_PREC / TP_ORG / SIZE_C. */
   dpu_desc.mc_surf_out  = mc_surf & 0x1;
   dpu_desc.tp_precision = tp_prec & 0x1;
   dpu_desc.tp_org_en    = tp_org  & 0x1;
   dpu_desc.size_c_wdma  = (uint16_t)(size_c & 0x7FF);

   /* Plain-conv path (no NPU K-accum): MRDMA/ERDMA disabled, MRDMA-trap-safe
    * control flow. int16's int32 output exceeds the DPU-EW's
    * <=16-bit operand DMA (like int8's int32), so DPU-EW K-accum is DEAD here —
    * K-partials are summed on the HOST in int64. The block below is populated but
    * inert (ew_accumulate=0). */
   dpu_desc.ew_accumulate = 0;
   {
     uint32_t M = (uint32_t)params->m;
     uint32_t hw = M;
     uint32_t ew_stride = hw > 12 ? hw : 12;
     dpu_desc.ew_src_base_addr = params->add_dma;
     dpu_desc.ew_base_addr     = params->add_dma + hw * 16;
     dpu_desc.ew_surf_stride   = ew_stride << 4;
     dpu_desc.ew_surf_notch    = ew_stride << 4;
     dpu_desc.ew_cfg_val    = EW_CFG_ALU(EW_ALU_ALGO_ADD);
     dpu_desc.erdma_cfg_val = ERDMA_CFG_16B;
     dpu_desc.rdma_fmc_val  = RDMA_FMC_BURST_LEN(15) | RDMA_FMC_COMB_USE(5) |
                              RDMA_FMC_IN_PRECISION(precision_float16) |
                              RDMA_FMC_PROC_PRECISION(precision_float16);
   }

   {
     int rc = gen_matmul_task(params->tasks, &cna_desc, &core_desc, &dpu_desc);
     if (rc < 0) return rc;
     params->task_count = (uint32_t)rc;
   }

   return 0;
}

/*
 * gen_matmul_bf16 — bf16 x bf16 -> fp32 single-task generator.
 *
 * FEASIBILITY OUTLOOK STRONG (unlike int16). bf16 is a FIRST-CLASS MAC datatype
 * on RK3588: it runs at 512x3 bf16 MAC operations per cycle (the SAME rate as
 * fp16), and the precision-field tables name 3'd3 = bfloat16 in the CNA CONV_CON1,
 * the DPU data-format reg, AND the DPU_RDMA reg. So the encoding (precision value
 * 3) is well-established. The standalone test (matmul_bf16_rocket) answers the ONE
 * residual question: does feeding precision=3 + bf16-truncated 2-byte operands
 * actually compute a full M×N result on THIS path.
 *
 * This is a near-verbatim clone of gen_matmul_fp16's fp32-OUTPUT branch
 * (params->fp32tofp16==0). bf16 is 2 bytes like fp16, so the INPUT geometry is
 * IDENTICAL: feature cube C2=8, weight (N/16,K/32,16,32) == weight_fp16, 2-byte
 * whole-element stores, data_entries /32. bf16 multiply ACCUMULATES TO FP32, so the
 * output is fp32 — and we reuse the fp16 path's PROVEN fp32-out writer
 * (out_precision=float32, size_e=3, surf_add=stride*4; OUTPUT cube C2=4, the 4-byte
 * fp32 cube). The exact failure that killed int16 (int32-out writer iterates only
 * ONE tile) does NOT apply here: fp32-out is the same writer the fp16 prefill path
 * validates on every run.
 *
 * Deltas vs gen_matmul_fp16's fp32-out branch (everything else byte-identical):
 *   CNA / CORE / DPU in & proc precision: float16(2) -> bfloat16(3) [ROCKET_BF16_PREC]
 *   (out precision float32 / size_e 3 / surf*4 already == the fp16 fp32-out branch)
 *   fp32tofp16_en = 0 always (bf16 -> fp32; no fp16-narrowing variant here).
 *
 * Every uncertain field is env-overridable so matmul_bf16_rocket can sweep on HW:
 *   ROCKET_BF16_PREC      in/proc precision   (default 3 = bf16; sweep 7 = tf32 ctrl)
 *   ROCKET_BF16_OUT_PREC  output precision    (default 5 = fp32)
 *   ROCKET_BF16_SIZE_E    fp32-out stride exp (default 3, == fp16's fp32-out)
 *   ROCKET_BF16_SURF_MULT fp32-out surf mult  (default 4 = 4 bytes/elem)
 *   ROCKET_BF16_DENTRIES_DIV (default 32, 2-byte element like fp16)
 *
 * Plain-conv path only (ew_accumulate=0): MRDMA/ERDMA disabled — byte-identical EW
 * emission to the fp16 plain path (gen_matmul_task only reads the EW operand fields
 * when ew_accumulate=1). bf16 K-partials, if ever tiled, sum on the HOST in fp32
 * (do NOT attempt NPU bf16 EW K-accum without separate validation; the EW operand
 * DMA is <=16-bit, the same wall int8/int16 int32-accum hit). */
int gen_matmul_bf16(matmul_params_t *params)
{
   npu_cna_desc cna_desc;
   npu_core_desc core_desc;
   npu_dpu_desc dpu_desc = {0};   /* zero new output-writer + EW fields; inert at accumulate=0 */

   unsigned int fd_bytes, fd_banks, weight_banks;
   int surf_stride;
   const char *e;

   /* Established defaults (sweep-overridable, as int4/int16 bring-up taught us
    * to keep every uncertain field a knob until HW agrees). */
   unsigned in_prec  = precision_bfloat16;   /* 3 = bf16 (3'd3) */
   unsigned out_prec = precision_float32;    /* bf16 MAC -> fp32 accumulate -> fp32 out */
   unsigned se = 3, sm = 4;                   /* fp32-out stride: size_e=3, surf*4 (== fp16) */
   unsigned dentries_div = 32;                /* 2-byte element, like fp16 */
   if ((e = getenv("ROCKET_BF16_PREC")))         in_prec      = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_BF16_OUT_PREC")))     out_prec     = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_BF16_SIZE_E")))       se           = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_BF16_SURF_MULT")))    sm           = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_BF16_DENTRIES_DIV"))) dentries_div = (unsigned)strtoul(e, NULL, 0);
   if (dentries_div == 0) dentries_div = 32;

   cna_desc.conv_mode = direct_convolution;
   cna_desc.in_precision = in_prec & 0x7;
   cna_desc.proc_precision = in_prec & 0x7;

   cna_desc.kernel_groups = 0;
   cna_desc.feature_grains = params->m + 1;
   cna_desc.conv_x_stride = 1;
   cna_desc.conv_y_stride = 1;

   cna_desc.datain_width = 1;
   cna_desc.datain_height = params->m;
   cna_desc.datain_channel = params->k;
   cna_desc.dataout_width = 1;
   cna_desc.dataout_height = params->m;
   cna_desc.dataout_atomics = cna_desc.dataout_width * cna_desc.dataout_height;

   cna_desc.weight_width = 1;
   cna_desc.weight_height = 1;
   cna_desc.weight_kernels = params->n;
   /* bf16 = 2 B/elem (like fp16): weight bytes = W*H*K * 2 per kernel. */
   cna_desc.weight_bytes_per_kernel = cna_desc.weight_width * cna_desc.weight_height *
     cna_desc.datain_channel * sizeof(uint16_t);
   cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels;

   fd_bytes = cna_desc.datain_width * cna_desc.datain_height * cna_desc.datain_channel * sizeof(uint16_t);
   fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
   fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks + 1;
   weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
   weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE) == 0) ? weight_banks : weight_banks + 1;
   if (fd_banks > NPU_CBUF_BANKS - 1) {
     return -1;
   } else {
       if (cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE) {
        weight_banks = NPU_CBUF_BANKS - fd_banks;
       } else {
         return -2;
       }
   }

   cna_desc.weight_bank = weight_banks;
   cna_desc.data_bank = fd_banks;
   cna_desc.weight_reuse = params->weight_reuse & 0x1;
   cna_desc.data_reuse   = params->data_reuse   & 0x1;
   cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / dentries_div;
   cna_desc.data_entries = (((cna_desc.datain_width * cna_desc.datain_channel) % dentries_div) == 0) ?
     cna_desc.data_entries : cna_desc.data_entries + 1;
   cna_desc.data_sign = 0x1;
   cna_desc.cvt_type  = 0x1;
   cna_desc.cvt_bypass = 0x1;
   cna_desc.cvt_scale0 = 0x1;
   cna_desc.cvt_scale1 = 0x1;
   cna_desc.cvt_scale2 = 0x1;
   cna_desc.cvt_scale3 = 0x1;
   cna_desc.fc_skip_en = 0;
   cna_desc.data_offset = 0x0;
   cna_desc.pad_left = 0;
   cna_desc.pad_top = 0;
   cna_desc.feature_base_addr = params->input_dma;
   cna_desc.weight_offset = 0;
   cna_desc.weight_burst_len = 0xf;
   cna_desc.data_burst_len = 0xf;
   cna_desc.line_stride = cna_desc.datain_width * 4;
   surf_stride = cna_feature_surf_stride(cna_desc.line_stride, cna_desc.datain_height);
   cna_desc.surf_stride = surf_stride;
   cna_desc.dma_width = cna_desc.datain_width;
   cna_desc.dma_height = cna_desc.datain_height;
   cna_desc.dma_channel = cna_desc.datain_channel;
   cna_desc.decompress_addr0 = params->weights_dma;

   core_desc.proc_precision = in_prec & 0x7;
   core_desc.qd_en = 1;
   core_desc.dataout_height = cna_desc.dataout_height - 1;
   core_desc.dataout_width = cna_desc.dataout_width - 1;
   core_desc.dataout_channel = cna_desc.weight_kernels - 1;

   dpu_desc.burst_len = 0xf;
   dpu_desc.conv_mode = direct_convolution;
   dpu_desc.output_mode = 0x2;
   dpu_desc.flying_mode = 0x0;
   dpu_desc.out_precision = out_prec & 0x7;
   dpu_desc.in_precision = in_prec & 0x7;
   dpu_desc.proc_precision = in_prec & 0x7;
   dpu_desc.dst_base_addr = params->output_dma;
   dpu_desc.dst_surf_stride = cna_desc.dataout_height * cna_desc.dataout_width;
   dpu_desc.width = core_desc.dataout_width;
   dpu_desc.height = core_desc.dataout_height;
   dpu_desc.channel = core_desc.dataout_channel;
   dpu_desc.bs_bypass = 1;
   dpu_desc.bs_alu_bypass = 1;
   dpu_desc.bs_mul_bypass = 1;
   dpu_desc.bs_relu_bypass = 1;
   dpu_desc.bn_bypass = 1;
   dpu_desc.bn_alu_bypass = 1;
   dpu_desc.bn_mul_bypass = 1;
   dpu_desc.bn_relu_bypass = 1;
   dpu_desc.ew_bypass = 1;
   dpu_desc.ew_op_bypass = 1;
   dpu_desc.ew_lut_bypass = 1;
   dpu_desc.ew_op_cvt_bypass = 1;
   dpu_desc.ew_relu_bypass = 1;
   dpu_desc.fp32tofp16_en = 0;                /* bf16 -> fp32 out, no narrowing */
   dpu_desc.out_cvt_scale = 1;
   dpu_desc.size_e_2 = se & 0x7;
   dpu_desc.size_e_1 = se & 0x7;
   dpu_desc.size_e_0 = se & 0x7;
   dpu_desc.surf_add = dpu_desc.dst_surf_stride * sm;
   dpu_desc.od_bypass = 1;
   dpu_desc.width_wdma = core_desc.dataout_width;
   dpu_desc.height_wdma = core_desc.dataout_height;
   dpu_desc.channel_wdma = core_desc.dataout_channel;

   /* Plain-conv path: no NPU K-accum. With ew_accumulate=0, gen_matmul_task takes
    * the bypass branch and disables MRDMA/ERDMA, so the EW operand fields stay
    * inert (zeroed above) — byte-identical to the fp16 plain-conv emission. */
   dpu_desc.ew_accumulate = 0;

   {
     int rc = gen_matmul_task(params->tasks, &cna_desc, &core_desc, &dpu_desc);
     if (rc < 0) return rc;
     params->task_count = (uint32_t)rc;
   }

   return 0;
}

/*
 * gen_matmul_tf32 — tf32 x tf32 -> fp32 single-task generator. The FINAL datatype
 * rung; completes the RK3588 matrix (fp16/int8/int4/int16/bf16/tf32).
 *
 * tf32 = 1 sign + 8-bit exponent (fp32 RANGE) + 10-bit mantissa (fp16 PRECISION),
 * stored in a 4-BYTE fp32 container (NVIDIA-TF32 convention: 19 used bits can't fit
 * 2 bytes). You feed fp32; the MAC rounds the mantissa to 10 bits and accumulates
 * in fp32. So tf32 is the FIRST 4-byte-INPUT matmul on this stack (fp16/bf16/int16
 * are 2-byte, int8 1-byte, int4 0.5-byte; int32/fp32 have only ever been OUTPUTS).
 *
 * ENCODING ESTABLISHED, INPUT GEOMETRY FROM-SCRATCH. The precision-field tables name
 * "3'd7: tf32" in the CNA CONV_CON1 in/proc precision table, and tf32 runs at 256x3
 * MAC operations per cycle (HALF fp16/bf16's 512x3 rate). The OUTPUT side is the
 * known quantity: tf32 accumulates to fp32, so out_precision=5 / size_e=3 / surf*4
 * is the SAME proven fp32-out writer bf16 and fp16 ride (the DPU out enum has no
 * tf32 slot — expected, fp32 is the carrier). The REAL unknown is the 4-byte INPUT
 * geometry, which the standalone test (matmul_tf32_rocket) cracks on HW.
 *
 * Deltas vs gen_matmul_bf16 (everything else byte-identical to the fp32-out path):
 *   PRECISION is PER-STAGE (tf32=7 only at the front): CNA in/proc = 7, CORE proc =
 *     7 (the MAC), but DPU in/proc/out = fp32(5). bf16 could set all stages to 3
 *     because bf16=3 is a valid DPU code; tf32=7 is NOT in the DPU enum
 *     (out_precision lists 0..6), and the first HW run proved it — all-stages-7 RAN
 *     but the DPU wrote NOTHING. The DPU only sees the fp32 accumulator anyway.
 *     [ROCKET_TF32_PREC / _CORE_PREC / _DPU_PREC / _OUT_PREC]
 *   element size:  2 B (uint16_t) -> 4 B (fp32 container)      [ROCKET_TF32_ELEM_BYTES]
 *     -> weight_bytes_per_kernel = K*4, fd_bytes = M*K*4 (drives CBUF bank split).
 *   data_entries divisor: 32 (2-byte) -> 16 (4-byte)           [ROCKET_TF32_DENTRIES_DIV]
 *     (a "data entry" is a fixed 64 bytes: 32*2 fp16 = 64*1 int8 = 16*4 tf32, so
 *      divisor = 64/bytes = 16 for tf32; same family as fp16 /32, int8 /64).
 *   (out precision 5 / size_e 3 / surf*4 already == the bf16/fp16 fp32-out writer.)
 *
 * The feature-cube C2 (=4) and the weight tile (N/16, K/16, 16, 16 — HW-confirmed;
 * for 4-byte the K-group halves to 16, N-group stays 16) live in the PACKING (the
 * test / weight_tf32), NOT here: this generator only sets total byte sizes +
 * precision + data_entries (= K/16 = #K-groups). Every field is an env knob so
 * matmul_tf32_rocket can sweep without recompiling:
 *   ROCKET_TF32_PREC        CNA in/proc precision (default 7 = tf32; the operand read)
 *   ROCKET_TF32_CORE_PREC   CORE proc precision   (default 7 = tf32 MAC)
 *   ROCKET_TF32_DPU_PREC    DPU in/proc precision (default 5 = fp32; DPU has no tf32 code)
 *   ROCKET_TF32_OUT_PREC    DPU output precision  (default 5 = fp32)
 *   ROCKET_TF32_SIZE_E      fp32-out stride exp (default 3, == fp16/bf16 fp32-out)
 *   ROCKET_TF32_SURF_MULT   fp32-out surf mult  (default 4 = 4 bytes/elem)
 *   ROCKET_TF32_DENTRIES_DIV data-entry divisor (default 16 = 4-byte element)
 *   ROCKET_TF32_ELEM_BYTES  input element bytes (default 4; bank-split only)
 *
 * Plain-conv path only (ew_accumulate=0): MRDMA/ERDMA disabled, byte-identical EW
 * emission to the fp16/bf16 plain path. tf32 K-partials, if ever tiled, sum on the
 * HOST in fp32 (do NOT attempt NPU EW K-accum; the EW operand DMA is <=16-bit). */
int gen_matmul_tf32(matmul_params_t *params)
{
   npu_cna_desc cna_desc;
   npu_core_desc core_desc;
   npu_dpu_desc dpu_desc = {0};   /* zero new output-writer + EW fields; inert at accumulate=0 */

   unsigned int fd_bytes, fd_banks, weight_banks;
   int surf_stride;
   const char *e;

   /* PER-STAGE precision (the key tf32 lesson): tf32 = 7 is a
    * valid code only at the FRONT of the pipe. The DPU data-format
    * out_precision enum lists ONLY 0..6 (no tf32) — and the FIRST HW run confirmed
    * it: with DPU in/proc/out all = 7 the job RAN (no timeout) but the DPU wrote
    * NOTHING (output stayed 0xAA). tf32 must be 7 at the CNA (operand read) + CORE
    * (the MAC, per "256x3 tf32 MAC/cycle") but the DPU only ever sees the fp32
    * ACCUMULATOR, so DPU in/proc/out = fp32(5) — exactly the bf16/fp16 fp32-out
    * writer. All three stages are separate knobs so the sweep can isolate which
    * stage accepts 7 (int4/int16/bf16 lesson: keep every uncertain field a knob). */
   unsigned in_prec      = precision_tf32;      /* CNA in/proc (operand read): 7 (3'd7, CNA-valid) */
   unsigned core_prec    = precision_tf32;      /* CORE proc (the tf32 MAC): 7 */
   unsigned dpu_prec     = precision_float32;   /* DPU in/proc: 5 — DPU enum has NO 7; it rides the fp32 accumulator */
   unsigned out_prec     = precision_float32;   /* DPU out: 5 (fp32) */
   unsigned se = 3, sm = 4;                      /* fp32-out stride: size_e=3, surf*4 (== fp16/bf16) */
   unsigned dentries_div = 16;                   /* 4-byte element: 64-byte data entry / 4 */
   unsigned elem_bytes   = 4;                    /* 4-byte fp32 container (bank-split only) */
   if ((e = getenv("ROCKET_TF32_PREC")))         in_prec      = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_TF32_CORE_PREC")))    core_prec    = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_TF32_DPU_PREC")))     dpu_prec     = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_TF32_OUT_PREC")))     out_prec     = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_TF32_SIZE_E")))       se           = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_TF32_SURF_MULT")))    sm           = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_TF32_DENTRIES_DIV"))) dentries_div = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_TF32_ELEM_BYTES")))   elem_bytes   = (unsigned)strtoul(e, NULL, 0);
   if (dentries_div == 0) dentries_div = 16;
   if (elem_bytes == 0)   elem_bytes   = 4;

   /* MODE-BIT knobs (plausible N-lane / surface-routing levers, fixed at defaults
    * on the working dtypes but unswept for the tf32 4-byte path). feature_grains
    * default = M+1 (use -1 sentinel to
    * keep that); output_mode default 2; mc_surf_out (DPU_DATA_FORMAT bit 3) 0. */
   int fgrains_override = -1;
   unsigned output_mode = 0x2, mc_surf_out = 0;
   if ((e = getenv("ROCKET_TF32_FGRAINS")))     fgrains_override = (int)strtol(e, NULL, 0);
   if ((e = getenv("ROCKET_TF32_OUTPUT_MODE"))) output_mode = (unsigned)strtoul(e, NULL, 0);
   if ((e = getenv("ROCKET_TF32_MC_SURF")))     mc_surf_out = (unsigned)strtoul(e, NULL, 0);

   cna_desc.conv_mode = direct_convolution;
   cna_desc.in_precision = in_prec & 0x7;
   cna_desc.proc_precision = in_prec & 0x7;

   cna_desc.kernel_groups = 0;
   cna_desc.feature_grains = (fgrains_override >= 0) ? (unsigned)fgrains_override : (unsigned)(params->m + 1);
   cna_desc.conv_x_stride = 1;
   cna_desc.conv_y_stride = 1;

   cna_desc.datain_width = 1;
   cna_desc.datain_height = params->m;
   cna_desc.datain_channel = params->k;
   cna_desc.dataout_width = 1;
   cna_desc.dataout_height = params->m;
   cna_desc.dataout_atomics = cna_desc.dataout_width * cna_desc.dataout_height;

   cna_desc.weight_width = 1;
   cna_desc.weight_height = 1;
   cna_desc.weight_kernels = params->n;
   /* tf32 = 4 B/elem: weight bytes = W*H*K * 4 per kernel. */
   cna_desc.weight_bytes_per_kernel = cna_desc.weight_width * cna_desc.weight_height *
     cna_desc.datain_channel * elem_bytes;
   cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels;

   fd_bytes = cna_desc.datain_width * cna_desc.datain_height * cna_desc.datain_channel * elem_bytes;
   fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
   fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks + 1;
   weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
   weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE) == 0) ? weight_banks : weight_banks + 1;
   if (fd_banks > NPU_CBUF_BANKS - 1) {
     return -1;
   } else {
       if (cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE) {
        weight_banks = NPU_CBUF_BANKS - fd_banks;
       } else {
         return -2;
       }
   }

   cna_desc.weight_bank = weight_banks;
   cna_desc.data_bank = fd_banks;
   cna_desc.weight_reuse = params->weight_reuse & 0x1;
   cna_desc.data_reuse   = params->data_reuse   & 0x1;
   cna_desc.data_entries = (cna_desc.datain_width * cna_desc.datain_channel) / dentries_div;
   cna_desc.data_entries = (((cna_desc.datain_width * cna_desc.datain_channel) % dentries_div) == 0) ?
     cna_desc.data_entries : cna_desc.data_entries + 1;
   cna_desc.data_sign = 0x1;
   cna_desc.cvt_type  = 0x1;
   cna_desc.cvt_bypass = 0x1;
   cna_desc.cvt_scale0 = 0x1;
   cna_desc.cvt_scale1 = 0x1;
   cna_desc.cvt_scale2 = 0x1;
   cna_desc.cvt_scale3 = 0x1;
   cna_desc.fc_skip_en = 0;
   cna_desc.data_offset = 0x0;
   cna_desc.pad_left = 0;
   cna_desc.pad_top = 0;
   cna_desc.feature_base_addr = params->input_dma;
   cna_desc.weight_offset = 0;
   cna_desc.weight_burst_len = 0xf;
   cna_desc.data_burst_len = 0xf;
   cna_desc.line_stride = cna_desc.datain_width * 4;
   surf_stride = cna_feature_surf_stride(cna_desc.line_stride, cna_desc.datain_height);
   cna_desc.surf_stride = surf_stride;
   cna_desc.dma_width = cna_desc.datain_width;
   cna_desc.dma_height = cna_desc.datain_height;
   cna_desc.dma_channel = cna_desc.datain_channel;
   cna_desc.decompress_addr0 = params->weights_dma;

   core_desc.proc_precision = core_prec & 0x7;
   core_desc.qd_en = 1;
   core_desc.dataout_height = cna_desc.dataout_height - 1;
   core_desc.dataout_width = cna_desc.dataout_width - 1;
   core_desc.dataout_channel = cna_desc.weight_kernels - 1;

   dpu_desc.burst_len = 0xf;
   dpu_desc.conv_mode = direct_convolution;
   dpu_desc.output_mode = output_mode & 0x3;
   dpu_desc.flying_mode = 0x0;
   dpu_desc.mc_surf_out = mc_surf_out & 0x1;    /* DPU_DATA_FORMAT bit 3; N-lane/surface lever? */
   dpu_desc.out_precision = out_prec & 0x7;
   dpu_desc.in_precision = dpu_prec & 0x7;     /* DPU rides the fp32 accumulator (no tf32=7 DPU code) */
   dpu_desc.proc_precision = dpu_prec & 0x7;
   dpu_desc.dst_base_addr = params->output_dma;
   dpu_desc.dst_surf_stride = cna_desc.dataout_height * cna_desc.dataout_width;
   dpu_desc.width = core_desc.dataout_width;
   dpu_desc.height = core_desc.dataout_height;
   dpu_desc.channel = core_desc.dataout_channel;
   dpu_desc.bs_bypass = 1;
   dpu_desc.bs_alu_bypass = 1;
   dpu_desc.bs_mul_bypass = 1;
   dpu_desc.bs_relu_bypass = 1;
   dpu_desc.bn_bypass = 1;
   dpu_desc.bn_alu_bypass = 1;
   dpu_desc.bn_mul_bypass = 1;
   dpu_desc.bn_relu_bypass = 1;
   dpu_desc.ew_bypass = 1;
   dpu_desc.ew_op_bypass = 1;
   dpu_desc.ew_lut_bypass = 1;
   dpu_desc.ew_op_cvt_bypass = 1;
   dpu_desc.ew_relu_bypass = 1;
   dpu_desc.fp32tofp16_en = 0;                /* tf32 -> fp32 out, no narrowing */
   dpu_desc.out_cvt_scale = 1;
   dpu_desc.size_e_2 = se & 0x7;
   dpu_desc.size_e_1 = se & 0x7;
   dpu_desc.size_e_0 = se & 0x7;
   dpu_desc.surf_add = dpu_desc.dst_surf_stride * sm;
   dpu_desc.od_bypass = 1;
   dpu_desc.width_wdma = core_desc.dataout_width;
   dpu_desc.height_wdma = core_desc.dataout_height;
   dpu_desc.channel_wdma = core_desc.dataout_channel;

   /* Plain-conv path: no NPU K-accum (see gen_matmul_bf16's note). */
   dpu_desc.ew_accumulate = 0;

   {
     int rc = gen_matmul_task(params->tasks, &cna_desc, &core_desc, &dpu_desc);
     if (rc < 0) return rc;
     params->task_count = (uint32_t)rc;
   }

   return 0;
}

/* ============================================================================
 * SECTION — Tiled-layout index and weight-packing helpers
 * ==========================================================================*/

/* Hardware tiled-layout index math — NPU memory layout, not driver/format
 * dependent. */
int feature_data(int C, int H, int W, int C2, int c, int h, int w) {
  (void)C;   /* total channel count is not needed for the index (C2 is); kept for API symmetry */
  int plane = (c-1)/C2;
  int src = plane * H * W * C2;
  int offset = (c-1) % C2;
  int pos = src + C2 * ((h-1) * W + (w-1)) + offset;
  return pos;
}

int weight_fp16(int C, int k, int c) {
  int dst =0;
  int kpg = ((k-1)/16);
  int cpg = ((c-1)/32);
  dst = ((cpg*32)*16)+ (kpg*16*C);
  dst = dst + ((c-1)%32) + (((k-1)%16)*32);
  return dst;
}

/* Conv weight cube index (0-based). Mesa-confirmed reorder (rkt_coefs.c
 * rkt_fill_weights): loop nest oc1, ic1, kh, kw, oc2, ic2 (outer->inner). For fp16
 * the oc group is 16 and the ic (K) group is 32 (== weight_fp16's groups); the
 * kernel spatial (kh,kw) sit BETWEEN the ic1 and oc2 groups, so the contraction is
 * NOT a flat ic — it is structured (ic1, kh, kw, ic2). OC,IC are real (must be
 * %16 / %32); oc/ic/kh/kw are 1-based. At KH=KW=1 this collapses to weight_fp16
 * (the 1x1-conv == matmul identity). */
int weight_conv_fp16(int OC, int IC, int KH, int KW, int oc, int ic, int kh, int kw) {
  int oc1 = (oc - 1) / 16, oc2 = (oc - 1) % 16;
  int ic1 = (ic - 1) / 32, ic2 = (ic - 1) % 32;
  int nIC1 = (IC + 31) / 32;          /* number of ic groups */
  (void)OC;
  /* address = ((((oc1*nIC1 + ic1)*KH + (kh-1))*KW + (kw-1))*16 + oc2)*32 + ic2 */
  return ((((oc1 * nIC1 + ic1) * KH + (kh - 1)) * KW + (kw - 1)) * 16 + oc2) * 32 + ic2;
}

/* Depthwise conv weight cube index (0-based). Mesa's rkt_fill_weights with
 * output_channels==1 + input_channel_groups==2*WEIGHT_ATOMIC_SIZE collapses the
 * generic oc1/ic1/kh/kw/oc2/ic2 nest to ic1, kh, kw, ic2 (no oc dimension; one
 * filter per channel). Cube (C/G, KH, KW, G): G-grouped channels, kh outer / kw
 * inner spatial (matching the direct path), G channels innermost. G is the
 * depthwise channel group (Mesa int8 = 64; fp16 G = 32, HW-confirmed). C = channel
 * count; c/kh/kw are 1-based. */
int weight_conv_dw_fp16(int C, int KH, int KW, int G, int c, int kh, int kw) {
  int ic1 = (c - 1) / G, ic2 = (c - 1) % G;
  (void)C;
  return ((ic1 * KH + (kh - 1)) * KW + (kw - 1)) * G + ic2;
}

/* int8 conv weight cube index (0-based). The int8 generalization of
 * weight_conv_fp16: the SAME Mesa oc1/ic1/kh/kw/oc2/ic2 reorder, but with the int8
 * N-group(oc) = 32 (vs fp16's 16) and K-group(ic) = 32 — i.e. weight_int8's
 * (N/32,K/32,32,32) matmul tile generalized so the kernel spatial (kh,kw) sit
 * between the ic1 and oc2 groups. At KH=KW=1 this collapses to weight_int8 (the
 * 1x1-conv == int8-matmul identity; the algebra: oc1*nIC1*1024 == oc1*IC*32 when
 * IC%32==0). OC,IC are real (the driver zero-pads OC up to 32 / IC up to 32);
 * oc/ic/kh/kw are 1-based. */
int weight_conv_int8(int OC, int IC, int KH, int KW, int oc, int ic, int kh, int kw) {
  int oc1 = (oc - 1) / 32, oc2 = (oc - 1) % 32;
  int ic1 = (ic - 1) / 32, ic2 = (ic - 1) % 32;
  int nIC1 = (IC + 31) / 32;          /* number of ic groups */
  (void)OC;
  /* address = ((((oc1*nIC1 + ic1)*KH + (kh-1))*KW + (kw-1))*32 + oc2)*32 + ic2 */
  return ((((oc1 * nIC1 + ic1) * KH + (kh - 1)) * KW + (kw - 1)) * 32 + oc2) * 32 + ic2;
}

/* int8 depthwise conv weight cube index (0-based). Structurally IDENTICAL to
 * weight_conv_dw_fp16 (pure element-index math; (C/G,KH,KW,G) cube) — only the
 * channel group G differs: Mesa's int8 DW group is 64 (= feature-atom 16 × 4) vs
 * fp16's 32 (feature-atom 8 × 4). G is the caller's choice and is a HW-confirm item
 * (int8 DW is an open register-sweep, as the fp16 DW G crack was). C = depthwise
 * channels; c/kh/kw are 1-based. */
int weight_conv_dw_int8(int C, int KH, int KW, int G, int c, int kh, int kw) {
  int ic1 = (c - 1) / G, ic2 = (c - 1) % G;
  (void)C;
  return ((ic1 * KH + (kh - 1)) * KW + (kw - 1)) * G + ic2;
}

int weight_int8(int C, int k, int c) {
  int dst =0;
  int kpg = ((k-1)/32);
  int cpg = ((c-1)/32);
  dst = ((cpg*32)*32)+ (kpg*32*C);
  dst = dst + ((c-1)%32) + (((k-1)%32)*32);
  return dst;
}

/* int4 weight NIBBLE index (0-based): byte = idx/2, nibble = idx&1. HW-confirmed
 * RK3588 int4 weight native layout: (N/64, K/32, 64, 32) — i.e. N-group 64 (vs
 * int8's 32), K-group 32, then 64 N-within, then 32 K-within (innermost, two
 * consecutive-K int4 packed per byte). This DIFFERS from weight_int8 (N-group 32)
 * only for K>32: at K=32 the two layouts coincide (a single K-group). C = K
 * (contraction count, for the K-group stride), k = N-index (kernel), c = K-index. */
int weight_int4(int C, int k, int c) {
  int Ngrp    = (k-1)/64;
  int Nwithin = (k-1)%64;
  int Kgrp    = (c-1)/32;
  int Kwithin = (c-1)%32;
  int nKgrp   = (C + 31)/32;                 /* K/32 groups */
  return Ngrp*nKgrp*64*32 + Kgrp*64*32 + Nwithin*32 + Kwithin;
}

/* int16 weight element index (0-based). PRIME-SUSPECT layout == weight_fp16:
 * int16 is 2 bytes like fp16, so the native weight tile is the same (N/16, K/32,
 * 16, 32) — N-group 16, K-group 32 — only the stored bits are int16 not fp16.
 * The int16 bring-up (matmul_int16_rocket) confirms/corrects this on HW. C = K
 * (contraction count), k = N-index (kernel), c = K-index. */
int weight_int16(int C, int k, int c) {
  int dst = 0;
  int kpg = ((k-1)/16);
  int cpg = ((c-1)/32);
  dst = ((cpg*32)*16) + (kpg*16*C);
  dst = dst + ((c-1)%32) + (((k-1)%16)*32);
  return dst;
}

/* tf32 weight element index (0-based). FIRST 4-byte-input layout — HW-CONFIRMED
 * (matmul_tf32_rocket PAT=3 ncol gives correct=256 at K=64/128 with NG=16,KG=16,
 * and full random matmul passes). The weight block is the fixed 1024-byte HW tile
 * (16*16*4 = 1024 B), but for 4-byte tf32 it is shaped (N/16, K/16, 16, 16) —
 * N-group 16 (same as fp16), K-group **16** (halved from fp16's 32): for a 4-byte
 * element the K-group halves, the N-group does not. This is invisible at K=32 (a single
 * K-group for KG=32) and only testable at K>=64 — a half-rate (N-group 8, K-group 32)
 * layout looks correct there. data_entries = K/16 reads as K / K-group (# K-groups),
 * consistent with fp16's K/32 at KG=32. C = K (contraction count), k = N-index
 * (kernel), c = K-index. */
int weight_tf32(int C, int k, int c) {
  int dst = 0;
  int kpg = ((k-1)/16);
  int cpg = ((c-1)/16);
  dst = ((cpg*16)*16) + (kpg*16*C);
  dst = dst + ((c-1)%16) + (((k-1)%16)*16);
  return dst;
}

/* ============================================================================
 * SECTION — Convolution regcmd generators (fp16 and int8, direct + depthwise)
 * ==========================================================================*/

/* ============================================================================
 * General fp16 CONV_2D generators (gen_conv2d_fp16 / gen_conv2d_dw_fp16).
 *
 * The NPU CNA is a convolution engine; our matmul (gen_matmul_fp16) is its
 * degenerate 1x1 / stride-1 / pad-0 / dilation-1 / single-spatial-column case.
 * These siblings un-pin the conv knobs the matmul path hardcodes:
 *   weight_width/height 1 -> KW/KH        conv_x/y_stride 1 -> stride
 *   datain_width 1 -> IW, height M -> IH  pad_left/top 0 -> pad
 *   CONV_CON3 dilation 0 -> (dil-1)       (depthwise) CONV_MODE 0 -> 3 + DW_EN
 * The contraction K = KH*KW*IC is reduced natively by the conv (no im2col); the
 * host scatters the feature into the NC1HWC2 cube (feature_data, C2=8) and the
 * weights into the conv weight cube (weight_conv_fp16). The matmul generators and
 * the shared gen_matmul_task are left byte-for-byte untouched.
 *
 * gen_conv2d_task is a conv-aware copy of gen_matmul_task: identical CNA/CORE/DPU
 * sequence and the same MRDMA-disable plain-conv DPU_RDMA block (see the file
 * header's MRDMA trap), plus (a) the CONV_CON3 ATROUS dilation fields, (b) the
 * CORE_MISC_CFG DW_EN bit, and (c) the DPU_RDMA conv_mode for depthwise. It is the
 * plain path only (no NPU eltwise K-accum): the conv reduces all of K in one pass,
 * and rocket_conv2d_fp16 accumulates any OC/spatial split on the host.
 *
 * HW-VALIDATED 2026-06-19 on RK3588 hardware: the direct path is bit-exact vs the fp32
 * oracle across a multi-Kh/Kw/stride/pad/dilation/non-square sweep. The on-device
 * bring-up found ONE bug — the feature surface-stride (see the IW*(IH-4) note
 * below) — and confirmed every other field on the first run. Two CNA sizing fields
 * stayed env-overridable through the sweep (the int4/tf32 discipline); the defaults
 * below are now HW-confirmed and the knobs are kept only as diagnostics:
 *   feature_grains (CONV_CON2 [13:4]) default datain_height+1. HW-insensitive
 *     (9..65 all pass on a IH=8 shape) — it is a prefetch-depth floor, not exact.
 *     ROCKET_CONV_GRAINS overrides; Mesa's analog is a magic 50+stride_y+1.
 *   data_entries (CBUF_CON1 [12:0]) default ceil(IW*IC/32). HW-confirmed for fp16
 *     (a IW=8,IC=32 shape needs exactly 8: 4 fails, 8 passes, 16 reads inf).
 *     ROCKET_CONV_DENTRIES overrides. (Mesa's int8 ceil(IW*2*ceil(IC/16)/8) gives 4
 *     here and is WRONG for the 2-byte fp16 path.)
 * Dilation rate-1 and the stride x/y axis order are likewise HW-confirmed
 * (ROCKET_CONV_DIL_RAW / ROCKET_CONV_STRIDE_SWAP both make correct shapes fail). */
static int gen_conv2d_task(uint64_t *ops, npu_cna_desc *cna_desc,
                           npu_core_desc *core_desc, npu_dpu_desc *dpu_desc)
{
  uint32_t value;
  int i = 0;

  ops[i++] = NPUOP(OP_REG_DPU, 0xE, DPU_S_POINTER);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0xE, DPU_RDMA_S_POINTER);

  value = ((cna_desc->proc_precision & 0x7) <<7) | ((cna_desc->in_precision & 0x7)<<4) |
    (cna_desc->conv_mode & 0xf);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CONV_CON1);
  /* 10-bit CNA field. Normal path is m+1 (<=257), well within range; a diagnostic
   * ROCKET_*_GRAINS override >1023 would be truncated by the &0x3FF below and
   * mis-compute. Fail the emission unconditionally (NOT an assert: the Release build
   * compiles asserts out via -DNDEBUG, so the guard would vanish in the shipping lib). */
  if (cna_desc->feature_grains > 0x3FF) {
    ROCKET_LOGE("npu_regcmd: feature_grains %u exceeds 10-bit CNA field (max 1023)\n",
            cna_desc->feature_grains);
    return -1;
  }
  value = ((cna_desc->kernel_groups & 0xFF) << 16) | ((cna_desc->feature_grains & 0x3FF) << 4);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CONV_CON2);
  /* CONV_CON3: stride [5:0] PLUS the ATROUS dilation fields (the conv delta vs the
   * matmul emitter, which only sets stride). x-dilation [20:16], y-dilation
   * [25:21], stored as rate-1 (0 == no dilation == byte-identical to matmul). */
  value = ((cna_desc->atrous_y_dilation & 0x1F) << 21) |
          ((cna_desc->atrous_x_dilation & 0x1F) << 16) |
          ((cna_desc->conv_y_stride & 0x7) << 3) | (cna_desc->conv_x_stride & 0x7);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CONV_CON3);
  value = ((cna_desc->datain_width) & 0x7FF) << 16 | (cna_desc->datain_height & 0x7FF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE0);
  value = ((cna_desc->datain_channel-1) & 0xFFFF) << 16 | (cna_desc->datain_channel & 0xFFFF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE1);
  value = cna_desc->dataout_width & 0x7FF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE2);
  value = cna_desc->dataout_atomics & 0x3FFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DATA_SIZE3);
  /* CNA_WEIGHT_SIZE0 is a dedicated full-width register (total weight bytes), so —
   * unlike the packed/sub-field neighbours above and below — weight_bytes is emitted
   * UNMASKED on purpose. (The field is the full word for all validated matmul
   * tile sizes; re-check the field width before driving much larger tiles.) */
  value = cna_desc->weight_bytes;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_WEIGHT_SIZE0);
  value = cna_desc->weight_bytes_per_kernel & 0x7FFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_WEIGHT_SIZE1);
  value = ((cna_desc->weight_width & 0x1F) <<24) | ((cna_desc->weight_height & 0x1F) << 16) |
    (cna_desc->weight_kernels & 0x3FFF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_WEIGHT_SIZE2);
  value = ((cna_desc->weight_bank & 0xF) << 4) | (cna_desc->data_bank & 0xF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CBUF_CON0);
  value = cna_desc->data_entries & 0x1FFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CBUF_CON1);
  value = ((cna_desc->data_sign & 0x1) << 3) | ((cna_desc->cvt_type & 0x1)<< 1) | (cna_desc->cvt_bypass & 0x1);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON0);
  value = ((cna_desc->cvt_scale0 & 0xFFFF) << 16) | 0x0;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON1);
  value = ((cna_desc->cvt_scale1 & 0xFFFF) << 16) | 0x0;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON2);
  value = ((cna_desc->cvt_scale2 & 0xFFFF) << 16) | 0x0;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON3);
  value = ((cna_desc->cvt_scale3 & 0xFFFF) << 16) | 0x0;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_CVT_CON4);
  value = cna_desc->fc_skip_en & 0x1;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_FC_CON0);
  value = cna_desc->data_offset & 0x1FFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_FC_CON1);
  value = ((cna_desc->pad_left & 0xF) << 4) | (cna_desc->pad_top & 0xF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_PAD_CON0);
  ops[i++] = NPUOP(OP_REG_CNA, cna_desc->feature_base_addr, CNA_FEATURE_DATA_ADDR);
  value = cna_desc->weight_offset & 0x1FFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_FC_CON2);
  value = ((cna_desc->weight_burst_len & 0xF) << 16) | (cna_desc->data_burst_len & 0xF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DMA_CON0);
  value = cna_desc->line_stride & 0xFFFFFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DMA_CON1);
  value = cna_desc->surf_stride & 0xFFFFFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_DMA_CON2);
  value = ((cna_desc->dma_width & 0x7FF) << 16) | (cna_desc->dma_height & 0x7FF);
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_FC_DATA_SIZE0);
  value = cna_desc->dma_channel & 0xFFFF;
  ops[i++] = NPUOP(OP_REG_CNA, value, CNA_FC_DATA_SIZE1);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_CTRL);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_REGNUM);
  ops[i++] = NPUOP(OP_REG_CNA, cna_desc->decompress_addr0, CNA_DCOMP_ADDR0);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT1);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT2);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT3);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT4);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT5);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT6);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT7);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT8);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT9);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT10);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT11);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT12);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT13);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT14);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT15);
  ops[i++] = NPUOP(OP_REG_CNA, 0x0, CNA_CVT_CON5);
  ops[i++] = NPUOP(OP_REG_CNA, cna_desc->pad_con1, CNA_PAD_CON1);

  /* CORE_MISC_CFG: proc_precision + QD_EN, plus DW_EN for depthwise (the conv delta
   * vs the matmul emitter). DW_EN default 0 == matmul. */
  value = ((core_desc->proc_precision & 0x7) << 8) | ((core_desc->dw_en & 0x1) << 1) |
          (core_desc->qd_en & 0x1);
  ops[i++] = NPUOP(OP_REG_CORE, value, CORE_MISC_CFG);
  value = ((core_desc->dataout_height & 0xFFFF) << 16) | (core_desc->dataout_width & 0xFFFF);
  ops[i++] = NPUOP(OP_REG_CORE, value, CORE_DATAOUT_SIZE_0);
  value = core_desc->dataout_channel & 0xFFFF;
  ops[i++] = NPUOP(OP_REG_CORE, value, CORE_DATAOUT_SIZE_1);
  ops[i++] = NPUOP(OP_REG_CORE, 0x0, CORE_CLIP_TRUNCATE);
  ops[i++] = NPUOP(OP_REG_CORE, 0x0, CORE_3030);

  value = ((dpu_desc->burst_len & 0xF) << 5) | ((dpu_desc->conv_mode & 0x3) <<3) |
    ((dpu_desc->output_mode & 0x3) <<1) | (dpu_desc->flying_mode & 0x1);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_FEATURE_MODE_CFG);
  value = ((dpu_desc->out_precision & 0x7) << 29) | ((dpu_desc->in_precision & 0x7) << 26) |
    ((dpu_desc->mc_surf_out & 0x1) << 3) | (dpu_desc->proc_precision & 0x7);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_DATA_FORMAT);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_OFFSET_PEND);
  ops[i++] = NPUOP(OP_REG_DPU, dpu_desc->dst_base_addr, DPU_DST_BASE_ADD);
  value = (dpu_desc->dst_surf_stride & 0xFFFFFFF) << 4;
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_DST_SURF_STRIDE);
  value = dpu_desc->width & 0x1FFF;
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_DATA_CUBE_WIDTH);
  value = dpu_desc->height & 0x1FFF;
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_DATA_CUBE_HEIGHT);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_DATA_CUBE_NOTCH_ADDR);
  value = ((dpu_desc->channel & 0x1FFF) << 16) | (dpu_desc->channel & 0x1FFF);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_DATA_CUBE_CHANNEL);
  /* BS stage: when bias_en, drive the BS ALU in add mode reading the per-OC int32
   * bias from BRDMA (Mesa's int8 word: BS_ALU_ALGO(2)|BS_ALU_SRC(1)|RELU/MUL bypass
   * = 0x20150). Else the all-bypass word from the bs_* fields. */
  if (dpu_desc->bias_en) {
    value = (2u << 16) | (1u << 8) | (1u << 6) | (1u << 4);
  } else {
    value = ((dpu_desc->bs_relu_bypass & 0x1) << 6) | ((dpu_desc->bs_mul_bypass & 0x1) << 4) |
      ((dpu_desc->bs_alu_bypass & 0x1) << 1) | (dpu_desc->bs_bypass & 0x1);
  }
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_BS_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_BS_ALU_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_BS_MUL_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_BS_RELUX_CMP_VALUE);
  value = ((dpu_desc->tp_org_en & 0x1) << 27) | ((dpu_desc->size_e_2 & 0x7) << 8) |
    ((dpu_desc->size_e_1 & 0x7) << 5) |
    ((dpu_desc->size_e_0 & 0x7) << 2) | ((dpu_desc->od_bypass & 0x1) << 1);
  ops[i++] = NPUOP(OP_REG_DPU, value,  DPU_BS_OW_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, (dpu_desc->bs_ow_op & 0xFFFF), DPU_BS_OW_OP);
  value = ((dpu_desc->tp_precision & 0x1) << 27) | ((dpu_desc->size_c_wdma & 0x7FF) << 16) |
    (dpu_desc->channel_wdma & 0x1FFF);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_WDMA_SIZE_0);
  value = ((dpu_desc->height_wdma & 0x1FFF) << 16) | (dpu_desc->width_wdma & 0x1FFF);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_WDMA_SIZE_1);
  if (dpu_desc->lut_en) {
    /* conv->activation fusion: the BN stage maps the conv result x onto the LUT index
     * grid (ALU no-op add of the 0x80000000 == fp32 -0.0 bias, then MUL by the index
     * scale), exactly as the standalone gen_lut_activation_fp16. */
    ops[i++] = NPUOP(OP_REG_DPU, (2u<<16) | (1u<<6), DPU_BN_CFG);   /* BN_ALU_ALGO(2)|RELU_BYPASS */
    ops[i++] = NPUOP(OP_REG_DPU, dpu_desc->lut_ep->bn_alu_cfg, DPU_BN_ALU_CFG);
    ops[i++] = NPUOP(OP_REG_DPU,
                     ((uint32_t)dpu_desc->lut_ep->bn_mul_operand & 0xFFFF) << 16, DPU_BN_MUL_CFG);
  } else {
    value = ((dpu_desc->bn_relu_bypass & 0x1) << 6) | ((dpu_desc->bn_mul_bypass &0x1) << 4) |
      ((dpu_desc->bn_alu_bypass & 0x1) << 1) | (dpu_desc->bn_bypass & 0x1);
    ops[i++] = NPUOP(OP_REG_DPU, value, DPU_BN_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_BN_ALU_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_BN_MUL_CFG);
  }
  ops[i++] = NPUOP(OP_REG_DPU, 0x0,DPU_BN_RELUX_CMP_VALUE);
  if (dpu_desc->lut_en) {
    /* EW stage runs the LUT (EW_LUT_BYPASS left CLEAR, bit7=0); RELU + OP_CVT bypassed. */
    value = (1u<<9) | (1u<<8);
  } else {
    value = ((dpu_desc->ew_relu_bypass & 0x1) << 9) | ((dpu_desc->ew_op_cvt_bypass & 0x1) << 8) |
      ((dpu_desc->ew_lut_bypass & 0x1) <<7) | ((dpu_desc->ew_op_bypass & 0x1) << 1) |
      (dpu_desc->ew_bypass & 0x1);
  }
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_EW_CFG);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_CVT_OFFSET_VALUE);
  ops[i++] = NPUOP(OP_REG_DPU, 0x1, DPU_EW_CVT_SCALE_VALUE);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_RELUX_CMP_VALUE);
  /* OUT_CVT requant: the LUT-activation path affine-decodes the Q-result back to fp16
   * (out = (q + offset) * 2^-minus_exp, signed pre-shift offset); the int8-out path
   * programs offset/scale/shift from dpu_desc; the float/int32-raw default leaves
   * offset=0/shift=0 and scale=1 (or the fp32->fp16 narrowing scale). */
  if (dpu_desc->lut_en) {
    const lut_epilogue_t *L = dpu_desc->lut_ep;
    ops[i++] = NPUOP(OP_REG_DPU, L->out_cvt_offset, DPU_OUT_CVT_OFFSET);
    value = (1u << 16) | (L->out_cvt_scale & 0xFFFF);   /* FP32TOFP16_EN | scale */
    ops[i++] = NPUOP(OP_REG_DPU, value, DPU_OUT_CVT_SCALE);
    value = (((uint32_t)L->out_cvt_cvt_type & 0x1) << 31) |
            (((uint32_t)L->out_cvt_minus_exp & 0xFF) << 12);
    ops[i++] = NPUOP(OP_REG_DPU, value, DPU_OUT_CVT_SHIFT);
  } else {
    ops[i++] = NPUOP(OP_REG_DPU, dpu_desc->out_cvt_offset, DPU_OUT_CVT_OFFSET);
    value = ((dpu_desc->fp32tofp16_en & 0x1) << 16) | (dpu_desc->out_cvt_scale & 0xFFFF);
    ops[i++] = NPUOP(OP_REG_DPU, value, DPU_OUT_CVT_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, (dpu_desc->out_cvt_shift & 0x3F), DPU_OUT_CVT_SHIFT);
  }
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_0);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_1);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_2);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_3);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_4);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_5);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_6);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_7);
  value = ((dpu_desc->surf_add & 0xFFFFFFF) << 4);
  ops[i++] = NPUOP(OP_REG_DPU, value, DPU_SURFACE_ADD);
  ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_40C4);
  if (dpu_desc->lut_en) {
    /* conv->activation: upload the LE/LO tables here (inline, where the LUT control
     * registers live — the table writes are order-independent vs the other DPU config
     * since nothing fires until PC_OPERATION_ENABLE, and the 513 DATA writes per table
     * stay contiguous), then program the hybrid LE+LO mux + range + extrapolation. */
    const lut_epilogue_t *L = dpu_desc->lut_ep;
    i = emit_lut_tables(ops, i, L->lut);
    ops[i++] = NPUOP(OP_REG_DPU, (1u<<6) | (1u<<5) | ((2u & 0x3) << 2), DPU_LUT_CFG);
    ops[i++] = NPUOP(OP_REG_DPU,
                     (((uint32_t)L->lo_index_select & 0xFF) << 16) |
                     (((uint32_t)L->le_index_select & 0xFF) << 8), DPU_LUT_INFO);
    ops[i++] = NPUOP(OP_REG_DPU, L->lut_le_start, DPU_LUT_LE_START);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0,             DPU_LUT_LE_END);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0,             DPU_LUT_LO_START);
    ops[i++] = NPUOP(OP_REG_DPU, L->lut_lo_end,   DPU_LUT_LO_END);
    ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)L->le_slope_scale & 0xFFFF), DPU_LUT_LE_SLOPE_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)L->le_slope_shift & 0x1F),  DPU_LUT_LE_SLOPE_SHIFT);
    ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)L->lo_slope_scale & 0xFFFF), DPU_LUT_LO_SLOPE_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)L->lo_slope_shift & 0x1F),  DPU_LUT_LO_SLOPE_SHIFT);
  } else {
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_ACCESS_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_ACCESS_DATA);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_INFO);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_START);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_END);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_START);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_END);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_SLOPE_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LE_SLOPE_SHIFT);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_SLOPE_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, DPU_LUT_LO_SLOPE_SHIFT);
  }

  /* DPU_RDMA: plain-conv (no eltwise feed) — MRDMA + ERDMA disabled, the MRDMA
   * trap guard (see file header). Output cube dims = DPU output (W=OW,H=OH,C=OC). */
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (dpu_desc->width & 0x1FFF), DPU_RDMA_DATA_CUBE_WIDTH);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (dpu_desc->height & 0x1FFF), DPU_RDMA_DATA_CUBE_HEIGHT);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, (dpu_desc->channel & 0x1FFF), DPU_RDMA_DATA_CUBE_CHANNEL);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_SRC_BASE_ADDR);
  /* int8-out bias: arm BRDMA (BRDMA_DATA_USE(1) = bit1) to fetch the per-OC int32
   * bias cube from BS_BASE_ADDR. Else BS bypassed -> no bias read (float default). */
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, dpu_desc->bias_en ? (1u << 1) : 0x0, DPU_RDMA_BRDMA_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, dpu_desc->bias_en ? dpu_desc->bias_base_addr : 0x0, DPU_RDMA_BS_BASE_ADDR);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_NRDMA_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_BN_BASE_ADDR);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, RDMA_ERDMA_DISABLE, DPU_RDMA_ERDMA_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_EW_BASE_ADDR);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_EW_SURF_STRIDE);
  /* FMC: MRDMA disabled, fp16; plus conv_mode(3) at [2:1] for depthwise (the conv
   * delta vs the matmul emitter; dpu_desc->conv_mode is 0 for direct == matmul). */
  value = RDMA_FMC_BURST_LEN(15) | RDMA_FMC_MRDMA_DISABLE(1) |
          (dpu_desc->fp32tofp16_en ? RDMA_FMC_MRDMA_FP16TOFP32(1) : 0) |
          ((dpu_desc->conv_mode & 0x3) << 1);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, value, DPU_RDMA_FEATURE_MODE_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_SRC_DMA_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_SURF_NOTCH);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_PAD_CFG);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, RDMA_WEIGHT_ALL1, DPU_RDMA_WEIGHT);
  ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, DPU_RDMA_EW_SURF_NOTCH);

  ops[i++] = NPUOP(OP_NONE, 0x0, 0x0);
  ops[i++] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
  ops[i++] = NPUOP(OP_40, 0x0, 0x0);
  ops[i++] = NPUOP(OP_ENABLE, 0x1D, PC_OPERATION_ENABLE);

  return i;
}

/* Shared fill for both conv variants. depthwise=0 -> direct conv (OC kernels, full
 * IC reduction); depthwise=1 -> grouped conv (CONV_MODE=3 + DW_EN). Returns 0, or
 * <0 if a single feature/weight tile overflows the CBUF (caller must tile). */
static int gen_conv2d_fill(conv_params_t *params, int depthwise)
{
   npu_cna_desc cna_desc = {0};
   npu_core_desc core_desc = {0};
   npu_dpu_desc dpu_desc = {0};
   const char *e;

   unsigned IC = params->ic, IH = params->ih, IW = params->iw;
   unsigned OC = params->oc, OH = params->oh, OW = params->ow;
   unsigned KH = params->kh, KW = params->kw;

   unsigned int fd_bytes, fd_banks, weight_banks;
   int surf_stride;

   cna_desc.conv_mode = depthwise ? 3 : direct_convolution;
   cna_desc.in_precision = precision_float16;
   cna_desc.proc_precision = precision_float16;

   cna_desc.kernel_groups = 0;
   /* feature_grains / data_entries: HW-uncertain CBUF hints (see header). */
   cna_desc.feature_grains = IH + 1;
   if ((e = getenv("ROCKET_CONV_GRAINS"))) cna_desc.feature_grains = (uint16_t)strtoul(e, NULL, 0);
   /* Diagnostic knobs for the on-HW geometry bring-up (the non-square / dilation
    * cases that the square-symmetric set does not exercise):
    *   ROCKET_CONV_STRIDE_SWAP=1 — swap the conv_x/y_stride register fields (tests
    *     whether the engine's stride-x axis is our height, not width).
    *   ROCKET_CONV_DIL_RAW=1 — emit the ATROUS field as the raw dilation rate
    *     instead of rate-1 (the rate-1 encoding is a guess; Mesa never dilates). */
   {
     unsigned sx = params->stride_x, sy = params->stride_y;
     if (getenv("ROCKET_CONV_STRIDE_SWAP")) { unsigned t = sx; sx = sy; sy = t; }
     cna_desc.conv_x_stride = sx;
     cna_desc.conv_y_stride = sy;
   }
   {
     unsigned dx = params->dil_x > 0 ? params->dil_x : 1;
     unsigned dy = params->dil_y > 0 ? params->dil_y : 1;
     unsigned bias = getenv("ROCKET_CONV_DIL_RAW") ? 0 : 1;
     cna_desc.atrous_x_dilation = dx - bias;
     cna_desc.atrous_y_dilation = dy - bias;
   }

   cna_desc.datain_width = IW;
   cna_desc.datain_height = IH;
   cna_desc.datain_channel = IC;
   cna_desc.dataout_width = OW;
   cna_desc.dataout_height = OH;
   cna_desc.dataout_atomics = OW * OH;

   cna_desc.weight_width = KW;
   cna_desc.weight_height = KH;

   fd_bytes = IW * IH * IC * sizeof(_Float16);
   fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
   fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks + 1;
   if (fd_banks > NPU_CBUF_BANKS - 1)
     return -1;

   if (depthwise) {
     /* Depthwise: one filter per channel (no OC kernels). Mesa pins weight_kernels
      * to 1 and reorders into the (C/G, KH, KW, G) cube; the whole KH*KW*C weight
      * spans multiple CBUF banks (calc_weights_banks uses KW*KH*IC, NOT *OC). The
      * channel group G (Mesa int8 = 64) is the prime fp16 HW-sweep field. */
     /* fp16 depthwise channel group = 32 (HW-CONFIRMED 2026-06-20). Mesa's int8
      * DW group is 64 (= WEIGHT_ATOMIC_SIZE*2 = feature-atom 16 × 4); fp16 halves
      * the feature atom to 8, so the same 4× ratio gives 32. (G=64 fails HW:
      * max_abs=6/24; G=32 → max_abs=0 on all DW shapes once the size_e/surf_add/
      * grains/bs_ow_op regs above are correct.) */
     unsigned G = params->dw_group ? params->dw_group : 32;
     const char *eg = getenv("ROCKET_CONV_DW_GROUP");
     if (eg) G = (unsigned)strtoul(eg, NULL, 0);
     if (G == 0) G = 32;
     unsigned Cpad = ((IC + G - 1) / G) * G;            /* require IC%G==0 in plan */
     cna_desc.weight_kernels = 1;
     cna_desc.weight_bytes_per_kernel = KW * KH * IC * sizeof(_Float16);  /* Mesa WEIGHT_SIZE1 = KW*KH*IC */
     cna_desc.weight_bytes = Cpad * KH * KW * sizeof(_Float16);           /* G-padded cube bytes */
     weight_banks = (cna_desc.weight_bytes + NPU_CBUF_BANK_SIZE - 1) / NPU_CBUF_BANK_SIZE;
     if (fd_banks + weight_banks > NPU_CBUF_BANKS)
       return -2;
     weight_banks = NPU_CBUF_BANKS - fd_banks;   /* weight gets all remaining banks (== direct path) */
   } else {
     cna_desc.weight_kernels = OC;
     cna_desc.weight_bytes_per_kernel = KW * KH * IC * sizeof(_Float16);
     cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels;
     weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
     weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE) == 0) ? weight_banks : weight_banks + 1;
     if (cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE)
       weight_banks = NPU_CBUF_BANKS - fd_banks;
     else
       return -2;
   }

   cna_desc.weight_bank = weight_banks;
   cna_desc.data_bank = fd_banks;
   cna_desc.weight_reuse = 0;
   cna_desc.data_reuse = 0;
   cna_desc.data_entries = (IW * IC) / 32;
   cna_desc.data_entries = (((IW * IC) % 32) == 0) ? cna_desc.data_entries : cna_desc.data_entries + 1;
   if ((e = getenv("ROCKET_CONV_DENTRIES"))) cna_desc.data_entries = (uint16_t)strtoul(e, NULL, 0);
   cna_desc.data_sign = 0x1;
   cna_desc.cvt_type = 0x1;
   cna_desc.cvt_bypass = 0x1;
   cna_desc.cvt_scale0 = 0x1;
   cna_desc.cvt_scale1 = 0x1;
   cna_desc.cvt_scale2 = 0x1;
   cna_desc.cvt_scale3 = 0x1;
   cna_desc.fc_skip_en = 0;
   cna_desc.data_offset = 0x0;
   cna_desc.pad_left = params->pad_left;
   cna_desc.pad_top = params->pad_top;
   cna_desc.feature_base_addr = params->input_dma;
   cna_desc.weight_offset = 0;
   cna_desc.weight_burst_len = 0xf;
   cna_desc.data_burst_len = 0xf;
   cna_desc.line_stride = IW * 4;
   /* Feature surface (channel-plane) stride. The matmul uses
    * line_stride*((IH/4)-1), which is only exact when IH%4==0 (the matmul's M%4
    * rule) — the integer IH/4 truncates otherwise, under-stating the stride and
    * mis-sampling every conv with a non-%4 input height (HW-confirmed: IH in
    * {6,7,9,10} all failed, {8,12} passed). Mesa (rkt_task.c) computes it in float,
    * line_stride*(IH/4.0 - 1); since line_stride = IW*4 that is exactly IW*(IH-4),
    * which matches the matmul for IH%4==0 and is correct for any IH. */
   surf_stride = (int)((long)cna_desc.line_stride * ((int)IH - 4) / 4);
   /* Clamp the sub-4-row (no-full-block) case to 0: a negative surf_stride masks to a
    * garbage 28-bit stride and corrupts any tile with fewer than 4 input rows. */
   surf_stride = surf_stride < 0 ? 0 : surf_stride;
   cna_desc.surf_stride = surf_stride;
   cna_desc.dma_width = IW;
   cna_desc.dma_height = IH;
   cna_desc.dma_channel = IC;
   cna_desc.decompress_addr0 = params->weights_dma;

   core_desc.proc_precision = precision_float16;
   core_desc.qd_en = 1;
   core_desc.dw_en = depthwise ? 1 : 0;
   core_desc.dataout_height = OH - 1;
   core_desc.dataout_width = OW - 1;
   core_desc.dataout_channel = OC - 1;

   dpu_desc.burst_len = 0xf;
   dpu_desc.conv_mode = depthwise ? 3 : direct_convolution;
   dpu_desc.output_mode = 0x2;
   dpu_desc.flying_mode = 0x0;
   dpu_desc.out_precision = (params->fp32tofp16 == 0) ? precision_float32 : precision_float16;
   dpu_desc.in_precision = precision_float16;
   dpu_desc.proc_precision = precision_float16;
   dpu_desc.dst_base_addr = params->output_dma;
   dpu_desc.dst_surf_stride = OH * OW;
   dpu_desc.width = core_desc.dataout_width;
   dpu_desc.height = core_desc.dataout_height;
   dpu_desc.channel = core_desc.dataout_channel;
   dpu_desc.bs_bypass = 1;
   dpu_desc.bs_alu_bypass = 1;
   dpu_desc.bs_mul_bypass = 1;
   dpu_desc.bs_relu_bypass = 1;
   dpu_desc.bn_bypass = 1;
   dpu_desc.bn_alu_bypass = 1;
   dpu_desc.bn_mul_bypass = 1;
   dpu_desc.bn_relu_bypass = 1;
   dpu_desc.ew_bypass = 1;
   dpu_desc.ew_op_bypass = 1;
   dpu_desc.ew_lut_bypass = 1;
   dpu_desc.ew_op_cvt_bypass = 1;
   dpu_desc.ew_relu_bypass = 1;
   dpu_desc.fp32tofp16_en = params->fp32tofp16 & 0x1;
   dpu_desc.out_cvt_scale = 1;
   if (params->fp32tofp16 == 0) {
     dpu_desc.size_e_2 = 3; dpu_desc.size_e_1 = 3; dpu_desc.size_e_0 = 3;
   } else {
     dpu_desc.size_e_2 = 1; dpu_desc.size_e_1 = 1; dpu_desc.size_e_0 = 1;
   }
   dpu_desc.od_bypass = 1;
   dpu_desc.width_wdma = core_desc.dataout_width;
   dpu_desc.height_wdma = core_desc.dataout_height;
   dpu_desc.channel_wdma = core_desc.dataout_channel;
   dpu_desc.surf_add = (!params->fp32tofp16) ? dpu_desc.dst_surf_stride * 4 : dpu_desc.dst_surf_stride * 2;

   /* Depthwise output-writer fixes (Teflon ground-truth diff, 2026-06-20). Mesa's
    * rocket driver branches `if(depthwise)` for these output-geometry fields; a
    * depthwise job that copies the direct-conv values is channel-plausible-but-wrong
    * (the prior HW-FAIL signature). The fields below are baked from the captured Mesa
    * regcmd, ordered by confidence:
    *   HIGH (Mesa source-confirmed depthwise branches):
    *     size_e = 3   (rkt_regcmd.c:235) — overrides the fp16-out size_e=1 rule
    *.
    *     od_bypass = 0 — Mesa's BS_OW_CFG emits only SIZE_E, no OD_BYPASS bit; our
    *                  direct path's spurious OD_BYPASS=1 is dropped for DW.
    *     surf_add *= 2 (rkt_task.c:150, surfaces_per_row) — wider write-out surface.
    *   CANDIDATE (universal Mesa fields; insensitive for our HW-validated direct
    *   path, so only the DW path adopts them — overridable for the HW sweep):
    *     feature_grains = 50+stride_y+1 (rkt_regcmd.c:74) — Mesa's empirical magic.
    *     bs_ow_op = 0x80 - weight_zp (rkt_regcmd.c:244) — 128 for symmetric fp16.
    * Knobs (override the bake for single-variable HW isolation):
    *   ROCKET_CONV_DW_SIZEE, ROCKET_CONV_GRAINS, ROCKET_CONV_DW_BSOWOP,
    *   ROCKET_CONV_DW_OCPAD (pad OC Mesa-style: align(max(OC,32),32); *2 if OC<=32;
    *   align 64 — de-scatter still reads the real OC at the front). */
   if (depthwise) {
     const char *e;
     dpu_desc.size_e_2 = 3; dpu_desc.size_e_1 = 3; dpu_desc.size_e_0 = 3;  /* HIGH */
     dpu_desc.od_bypass = 0;                                              /* HIGH */
     dpu_desc.surf_add *= 2;                                              /* HIGH */
     if (!getenv("ROCKET_CONV_GRAINS"))
       cna_desc.feature_grains = 50 + params->stride_y + 1;              /* candidate */
     dpu_desc.bs_ow_op = 0x80;   /* 0x80 - weight_zp (symmetric fp16 zp=0) */  /* candidate */
     if ((e = getenv("ROCKET_CONV_DW_SIZEE"))) {
       unsigned se = (unsigned)strtoul(e, NULL, 0) & 0x7;
       dpu_desc.size_e_2 = se; dpu_desc.size_e_1 = se; dpu_desc.size_e_0 = se;
     }
     if ((e = getenv("ROCKET_CONV_DW_BSOWOP")))
       dpu_desc.bs_ow_op = (uint32_t)strtoul(e, NULL, 0);
     if (getenv("ROCKET_CONV_DW_OCPAD")) {
       unsigned oc_eff = ((OC + 31) / 32) * 32;
       if (OC <= 32) oc_eff *= 2;
       oc_eff = ((oc_eff + 63) / 64) * 64;
       core_desc.dataout_channel = oc_eff - 1;
       dpu_desc.channel          = oc_eff - 1;
       dpu_desc.channel_wdma     = oc_eff - 1;
     }
   }

   /* conv->activation fusion (fp16 path, DIRECT conv). When params->act is set, the
    * DPU epilogue post-processes the conv result with f(x) in the same job: BN-mul
    * index scale -> EW LUT -> affine OUT_CVT (see lut_epilogue_t / gen_conv2d_task).
    * The output is fp16 (the LUT result), so force the fp16-out writer geometry
    * (size_e=1, surf_add*2 narrowing) the same way fp32tofp16=1 already does above —
    * the LUT only changes the VALUE written, not the output cube layout, so readback
    * de-scatter is unchanged. SMOOTH single-pass kinds only (SiLU/tanh/GELU). */
   if (params->act) {
     dpu_desc.lut_en = 1;
     dpu_desc.lut_ep = params->act;
   }

   {
     int rc = gen_conv2d_task(params->tasks, &cna_desc, &core_desc, &dpu_desc);
     if (rc < 0) return rc;
     params->task_count = (uint32_t)rc;
   }
   return 0;
}

int gen_conv2d_fp16(conv_params_t *params)    { return gen_conv2d_fill(params, 0); }
int gen_conv2d_dw_fp16(conv_params_t *params) { return gen_conv2d_fill(params, 1); }

/* ============================================================================
 * Native int8 CONV_2D generators (gen_conv2d_int8 / gen_conv2d_dw_int8).
 *
 * The int8 sibling of gen_conv2d_fp16: int8 features x int8 weights reduced
 * NATIVELY by the CNA into an int32 accumulator written raw to DRAM (the host
 * applies the per-output-channel weight-scale x per-tensor act-scale requant). It
 * is to gen_conv2d_fp16 exactly what gen_matmul_int8 is to gen_matmul_fp16 — the
 * same shared gen_conv2d_task emitter, with the mechanical int8 deltas:
 *   CNA  in/proc precision fp16->int8; 1 B/elem; data_entries /32->/64
 *   CORE qd_en 1->0
 *   DPU  in/proc precision int8; OUT precision fp32/fp16->int32;
 *        size_e=7 / surf_add*8 — the int32-output stride quirk (== gen_matmul_int8;
 *        NOT the 4-byte "size_e=3, *4" float pattern: 7/8 is HW-proven for the int8
 *        matmul, 3/4 strides the int32 wrong).
 * The feature DMA strides (line_stride = IW*4, surf_stride = IW*(IH-4)) are
 * element-size-independent — the cube atom is 16 bytes for both fp16 (C2=8) and
 * int8 (C2=16) — so they carry over from the fp16 conv verbatim, exactly as the
 * matmul fp16/int8 paths share them.
 *
 * data_entries = ceil(IW*IC/64): the int8 analog of the fp16 conv's ceil(IW*IC/32),
 * and == Mesa's int8 conv formula ceil(IW*2*ceil(IC/16)/8) whenever IC%32==0 (the
 * direct path's alignment), so it is the Mesa-consistent value, not a fresh guess.
 *
 * Host packing (rocket_conv.c / the standalone gate): feature cube C2=16 (int8),
 * weight cube weight_conv_int8 (oc-group 32 / ic-group 32), int32 output cube C2=4.
 *
 * DEPTHWISE int8 is wired but UNPROVEN — the DW output-geometry regs (size_e /
 * surf_add / feature_grains / bs_ow_op) and the channel group G were HW-cracked for
 * fp16 and may differ for the int32 output; every uncertain DW field is
 * env-overridable (ROCKET_CONV_DW_*) for the on-HW sweep, exactly as the fp16 DW
 * crack used. The DIRECT path is the validated deliverable.
 */
static int gen_conv2d_int8_fill(conv_params_t *params, int depthwise)
{
   npu_cna_desc cna_desc = {0};
   npu_core_desc core_desc = {0};
   npu_dpu_desc dpu_desc = {0};
   const char *e;

   unsigned IC = params->ic, IH = params->ih, IW = params->iw;
   unsigned OC = params->oc, OH = params->oh, OW = params->ow;
   unsigned KH = params->kh, KW = params->kw;

   unsigned int fd_bytes, fd_banks, weight_banks;
   int surf_stride;

   cna_desc.conv_mode = depthwise ? 3 : direct_convolution;
   cna_desc.in_precision = precision_int8;
   cna_desc.proc_precision = precision_int8;

   cna_desc.kernel_groups = 0;
   cna_desc.feature_grains = IH + 1;
   if ((e = getenv("ROCKET_CONV_GRAINS"))) cna_desc.feature_grains = (uint16_t)strtoul(e, NULL, 0);
   {
     unsigned sx = params->stride_x, sy = params->stride_y;
     if (getenv("ROCKET_CONV_STRIDE_SWAP")) { unsigned t = sx; sx = sy; sy = t; }
     cna_desc.conv_x_stride = sx;
     cna_desc.conv_y_stride = sy;
   }
   {
     unsigned dx = params->dil_x > 0 ? params->dil_x : 1;
     unsigned dy = params->dil_y > 0 ? params->dil_y : 1;
     unsigned bias = getenv("ROCKET_CONV_DIL_RAW") ? 0 : 1;
     cna_desc.atrous_x_dilation = dx - bias;
     cna_desc.atrous_y_dilation = dy - bias;
   }

   cna_desc.datain_width = IW;
   cna_desc.datain_height = IH;
   cna_desc.datain_channel = IC;
   cna_desc.dataout_width = OW;
   cna_desc.dataout_height = OH;
   cna_desc.dataout_atomics = OW * OH;

   cna_desc.weight_width = KW;
   cna_desc.weight_height = KH;

   fd_bytes = IW * IH * IC * sizeof(int8_t);
   fd_banks = (fd_bytes / NPU_CBUF_BANK_SIZE);
   fd_banks = ((fd_bytes % NPU_CBUF_BANK_SIZE) == 0) ? fd_banks : fd_banks + 1;
   if (fd_banks > NPU_CBUF_BANKS - 1)
     return -1;

   if (depthwise) {
     /* int8 DW channel group: Mesa's value is 64 (feature-atom 16 × 4). HW-unconfirmed
      * (the fp16 G=32 crack was register-masked once) — ROCKET_CONV_DW_GROUP sweeps it. */
     unsigned G = params->dw_group ? params->dw_group : 64;
     const char *eg = getenv("ROCKET_CONV_DW_GROUP");
     if (eg) G = (unsigned)strtoul(eg, NULL, 0);
     if (G == 0) G = 64;
     unsigned Cpad = ((IC + G - 1) / G) * G;            /* require IC%G==0 in plan */
     cna_desc.weight_kernels = 1;
     cna_desc.weight_bytes_per_kernel = KW * KH * IC * sizeof(int8_t);  /* Mesa WEIGHT_SIZE1 = KW*KH*IC */
     cna_desc.weight_bytes = Cpad * KH * KW * sizeof(int8_t);           /* G-padded cube bytes */
     weight_banks = (cna_desc.weight_bytes + NPU_CBUF_BANK_SIZE - 1) / NPU_CBUF_BANK_SIZE;
     if (fd_banks + weight_banks > NPU_CBUF_BANKS)
       return -2;
     weight_banks = NPU_CBUF_BANKS - fd_banks;
   } else {
     cna_desc.weight_kernels = OC;
     cna_desc.weight_bytes_per_kernel = KW * KH * IC * sizeof(int8_t);
     cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels;
     weight_banks = (cna_desc.weight_bytes / NPU_CBUF_BANK_SIZE);
     weight_banks = ((cna_desc.weight_bytes % NPU_CBUF_BANK_SIZE) == 0) ? weight_banks : weight_banks + 1;
     if (cna_desc.weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE)
       weight_banks = NPU_CBUF_BANKS - fd_banks;
     else
       return -2;
   }

   cna_desc.weight_bank = weight_banks;
   cna_desc.data_bank = fd_banks;
   /* RESONANCE FIX: the int8 C2=16 feature-cube DMA
    * over-reads ONE CBUF bank past the ceil(bytes/bank) allocation at near-bank-full
    * feature geometries, garbling a tile's tail output rows. HW-proven for THIS conv
    * path (IW=1 == the matmul feature descriptor: IC=1184, IH=189..193 & 216..221 ->
    * last output rows ~3e5..9e5, rest bit-exact; +1 fixes the whole window, -1 worsens
    * it) — the identical signature gen_matmul_int8 has. Shared by direct AND depthwise
    * (same fd_banks). Reserve a slack bank for the feature; the weight takes the rest.
    * The int8 conv tiler (conv2d_int8_run) reserves the matching bank so
    * fd_banks+1 + weight_banks <= 12. fp16's 2-byte cube is immune (different fill).
    * As in gen_matmul_int8, the +1 slack must actually fit: if the conv tiler is
    * bypassed and fd_banks == BANKS-1, silently clamping data_bank back to fd_banks
    * re-creates the tail-row corruption — fail loudly and unconditionally instead. */
   if (fd_banks + 1 > (unsigned)NPU_CBUF_BANKS - 1) {
       ROCKET_LOGE("gen_conv2d_int8: int8 CBUF bank-slack does not fit "
               "(fd_banks=%u, need fd_banks+1 <= %d) — use the int8 conv tiler\n",
               fd_banks, NPU_CBUF_BANKS - 1);
       return -1;
   }
   {
       unsigned data_bank = fd_banks + 1;
       cna_desc.data_bank   = data_bank;
       cna_desc.weight_bank = NPU_CBUF_BANKS - data_bank;
   }
   /* AUDIT sentinel, kept for future RE (mirrors gen_matmul_int8's ROCKET_I8_FDBANK_EXTRA),
    * RELATIVE to the +1 base: EXTRA=-1 reproduces the old exact-fit (resonates), EXTRA=+1
    * over-reserves. Default unset = the fixed +1. */
   if ((e = getenv("ROCKET_CONV_I8_FDBANK_EXTRA"))) {
       int x = atoi(e);
       int nb = (int)cna_desc.data_bank + x;
       if (nb < 1) nb = 1;
       if (nb > (int)NPU_CBUF_BANKS - 1) nb = NPU_CBUF_BANKS - 1;
       cna_desc.data_bank   = (unsigned)nb;
       cna_desc.weight_bank = NPU_CBUF_BANKS - (unsigned)nb;
   }
   cna_desc.weight_reuse = 0;
   cna_desc.data_reuse = 0;
   cna_desc.data_entries = (IW * IC) / 64;
   cna_desc.data_entries = (((IW * IC) % 64) == 0) ? cna_desc.data_entries : cna_desc.data_entries + 1;
   if ((e = getenv("ROCKET_CONV_DENTRIES"))) cna_desc.data_entries = (uint16_t)strtoul(e, NULL, 0);
   cna_desc.data_sign = 0x1;
   cna_desc.cvt_type = 0x1;
   cna_desc.cvt_bypass = 0x1;
   cna_desc.cvt_scale0 = 0x1;
   cna_desc.cvt_scale1 = 0x1;
   cna_desc.cvt_scale2 = 0x1;
   cna_desc.cvt_scale3 = 0x1;
   cna_desc.fc_skip_en = 0;
   cna_desc.data_offset = 0x0;
   cna_desc.pad_left = params->pad_left;
   cna_desc.pad_top = params->pad_top;
   cna_desc.feature_base_addr = params->input_dma;
   cna_desc.weight_offset = 0;
   cna_desc.weight_burst_len = 0xf;
   cna_desc.data_burst_len = 0xf;
   cna_desc.line_stride = IW * 4;
   /* Same float-correct feature surface stride as the fp16 conv (IW*(IH-4)); the
    * cube atom is 16 B for both dtypes, so the stride is element-size-independent. */
   surf_stride = (int)((long)cna_desc.line_stride * ((int)IH - 4) / 4);
   surf_stride = surf_stride < 0 ? 0 : surf_stride;
   cna_desc.surf_stride = surf_stride;
   cna_desc.dma_width = IW;
   cna_desc.dma_height = IH;
   cna_desc.dma_channel = IC;
   cna_desc.decompress_addr0 = params->weights_dma;

   core_desc.proc_precision = precision_int8;
   core_desc.qd_en = 0;
   core_desc.dw_en = depthwise ? 1 : 0;
   core_desc.dataout_height = OH - 1;
   core_desc.dataout_width = OW - 1;
   core_desc.dataout_channel = OC - 1;

   dpu_desc.burst_len = 0xf;
   dpu_desc.conv_mode = depthwise ? 3 : direct_convolution;
   dpu_desc.output_mode = 0x2;
   dpu_desc.flying_mode = 0x0;
   dpu_desc.out_precision = precision_int32;
   dpu_desc.in_precision = precision_int8;
   dpu_desc.proc_precision = precision_int8;
   dpu_desc.dst_base_addr = params->output_dma;
   dpu_desc.dst_surf_stride = OH * OW;
   dpu_desc.width = core_desc.dataout_width;
   dpu_desc.height = core_desc.dataout_height;
   dpu_desc.channel = core_desc.dataout_channel;
   dpu_desc.bs_bypass = 1;
   dpu_desc.bs_alu_bypass = 1;
   dpu_desc.bs_mul_bypass = 1;
   dpu_desc.bs_relu_bypass = 1;
   dpu_desc.bn_bypass = 1;
   dpu_desc.bn_alu_bypass = 1;
   dpu_desc.bn_mul_bypass = 1;
   dpu_desc.bn_relu_bypass = 1;
   dpu_desc.ew_bypass = 1;
   dpu_desc.ew_op_bypass = 1;
   dpu_desc.ew_lut_bypass = 1;
   dpu_desc.ew_op_cvt_bypass = 1;
   dpu_desc.ew_relu_bypass = 1;
   dpu_desc.fp32tofp16_en = 0;
   dpu_desc.out_cvt_scale = 1;
   {
     /* int32-output stride quirk (== gen_matmul_int8): size_e=7, surf_add=stride*8.
      * Env-tunable for the on-HW geometry sweep — 7/8 is the HW-proven int8 default;
      * 3/4 (the float-native 4-byte pattern) strides the int32 wrong. */
     unsigned se = 7, sm = 8;
     if ((e = getenv("ROCKET_CONV_INT8_SIZE_E")))    se = (unsigned)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_CONV_INT8_SURF_MULT"))) sm = (unsigned)strtoul(e, NULL, 0);
     dpu_desc.size_e_2 = se & 0x7; dpu_desc.size_e_1 = se & 0x7; dpu_desc.size_e_0 = se & 0x7;
     dpu_desc.surf_add = dpu_desc.dst_surf_stride * sm;
   }
   dpu_desc.od_bypass = 1;
   dpu_desc.width_wdma = core_desc.dataout_width;
   dpu_desc.height_wdma = core_desc.dataout_height;
   dpu_desc.channel_wdma = core_desc.dataout_channel;

   if (depthwise) {
     /* DW int8 output geometry. Two datapaths:
      *
      *   int8_out=0 (diagnostic): int32-raw writer (size_e=7/surf*8) + Mesa's
      *     universal DW fields (od_bypass=0, feature_grains=50+sy+1, bs_ow_op=0x80).
      *     UNPROVEN — the int32-raw DW writer doubles the within-plane stride
      *     (got[2k]==ref[k]); kept only as the env-sweepable fallback.
      *
      *   int8_out=1 (the cracked path): Mesa's int8-OUTPUT writer with on-chip
      *     requant, captured bit-for-bit from Teflon. DATA_FORMAT=0 (int8 out/in/proc),
      *     QD_EN=1, size_e=3 / surf*4,
      *     a per-OC int32 bias added in the BS ALU (BRDMA), and the OUT_CVT requant
      *     out_i8 = sat8( round(acc_i32 * scale >> shift) + (out_zp - 0x80) ).
      *     This is int8-out (4x fewer readback bytes) AND correct — no int32 readback. */
     dpu_desc.od_bypass = 0;
     if (!getenv("ROCKET_CONV_GRAINS"))
       cna_desc.feature_grains = 50 + params->stride_y + 1;
     dpu_desc.bs_ow_op = 0x80 - (uint32_t)params->weight_zero_point;

     if (params->int8_out) {
       /* int8 output + on-chip requant (== Mesa's DW int8 regcmd). */
       core_desc.qd_en = 1;                       /* HW-required for the requant writer */
       /* CNA pads borders with the input zero-point in the uint8-centered domain
        * (Mesa rkt_regcmd.c: pad_con1 = input_zero_point - 0x80, input_zp as uint8). */
       cna_desc.pad_con1 = (uint32_t)(((uint32_t)params->input_zero_point & 0xff) - 0x80u);
       dpu_desc.out_precision = precision_int8;    /* DPU_DATA_FORMAT = 0 */
       /* int8-out writer stride: size_e=3, surf_add = dst_surf_stride*4 (NOT the
        * int32-raw 7/8). Matches the Teflon capture (SURFACE_ADD=256=OH*OW*4). */
       dpu_desc.size_e_2 = 3; dpu_desc.size_e_1 = 3; dpu_desc.size_e_0 = 3;
       dpu_desc.surf_add = dpu_desc.dst_surf_stride * 4;

       /* OUT_CVT requant triple, computed exactly as Mesa (rkt_regcmd.c):
        *   conv_scale = in_scale*w_scale/out_scale, reinterpreted as float bits;
        *   shift = 126 - exp + 16 - 1   (QNNPACK requantization.h derivation);
        *   scale = ((bits>>9)&0x7fff)+1, forced to have bit14 set;
        *   offset = out_zp - 0x80.
        * Per-tensor only (Teflon forces per-tensor quant). truncate_bits assumed 0
        * (Mesa's per-scale truncate hack-list is not reproduced — pick scales that
        * land truncate=0; the gate does). */
       union { float f; uint32_t u; } cv;
       cv.f = (params->in_scale * params->w_scale) / params->out_scale;
       uint32_t bits = cv.u;
       unsigned shift = 127u + 31u - 32u - (bits >> 23) + 16u;   /* == 126 - exp + 16 */
       unsigned scale = ((bits >> 9) & 0x7fffu) + 1u;
       if (scale < (1u << 14)) scale |= (1u << 14);
       dpu_desc.out_cvt_scale  = (uint16_t)scale;
       dpu_desc.out_cvt_shift  = (uint8_t)(shift - 1u);
       dpu_desc.out_cvt_offset = (uint32_t)(params->output_zero_point - 0x80);

       /* per-OC int32 bias add in the BS ALU, fetched by BRDMA. */
       dpu_desc.bias_en = 1;
       dpu_desc.bias_base_addr = params->bias_dma;
     }

     /* Env overrides for the on-HW geometry sweep (single-variable isolation). */
     if ((e = getenv("ROCKET_CONV_DW_SIZEE"))) {
       unsigned se = (unsigned)strtoul(e, NULL, 0) & 0x7;
       dpu_desc.size_e_2 = se; dpu_desc.size_e_1 = se; dpu_desc.size_e_0 = se;
     }
     if ((e = getenv("ROCKET_CONV_DW_SURF_MULT")))
       dpu_desc.surf_add = dpu_desc.dst_surf_stride * (uint32_t)strtoul(e, NULL, 0);
     if ((e = getenv("ROCKET_CONV_DW_BSOWOP")))
       dpu_desc.bs_ow_op = (uint32_t)strtoul(e, NULL, 0);
   }

   {
     int rc = gen_conv2d_task(params->tasks, &cna_desc, &core_desc, &dpu_desc);
     if (rc < 0) return rc;
     params->task_count = (uint32_t)rc;
   }
   return 0;
}

int gen_conv2d_int8(conv_params_t *params)    { return gen_conv2d_int8_fill(params, 0); }
int gen_conv2d_dw_int8(conv_params_t *params) { return gen_conv2d_int8_fill(params, 1); }

/* ============================================================================
 * SECTION — Pooling regcmd generator (PPU)
 * ==========================================================================*/

/* PPU / PPU_RDMA write targets (BLOCK_* | PC_OP_01), exactly as OP_REG_DPU is
 * BLOCK_DPU|PC_OP_01. */
#define OP_REG_PPU       (BLOCK_PPU | PC_OP_01)        /* 0x4001 */
#define OP_REG_PPU_RDMA  (BLOCK_PPU_RDMA | PC_OP_01)   /* 0x8001 */

/*
 * gen_pool_fp16 — MaxPool / AveragePool on the PPU (the NPU's pooling engine).
 *
 * A SELF-CONTAINED PPU job: PPU_RDMA reads the input NC1HWC2 cube (C2=8 fp16), the PPU
 * reduces each kernel window (max or average) and writes the output cube in the SAME
 * layout. NO CNA/CORE/DPU, no weights — so no MRDMA trap (those blocks are never
 * enabled). The register program + the average RECIP_KERNEL = fp16(65536/k) format are
 * HW-validated (MAX bit-exact, AVG within fp16-recip tolerance).
 *
 * Conventions (HW-confirmed): every geometry field is (value-1); cube strides are
 * BYTES (16-aligned, C2=8); the PC trailer enable mask 0x60 = PPU_OP_EN|PPU_RDMA_OP_EN
 * brings both blocks into op_en (the GLOBAL block-participation mask — no per-block
 * OPERATION_ENABLE is emitted). FLYING_MODE=1 with DPU_FLYIN=0 == standalone RDMA-fed.
 */
int gen_pool_fp16(pool_params_t *p)
{
  const int C2 = 8;     /* fp16 feature cube channel atom */
  const int esz = 2;    /* fp16 element bytes */
  uint64_t *ops = p->tasks;
  int i = 0;

  if (p->iw < 1 || p->ih < 1 || p->ow < 1 || p->oh < 1 || p->c < 1) return -1;
  if (p->kw < 1 || p->kh < 1 || p->stride_x < 1 || p->stride_y < 1) return -1;
  /* CUBE_* dims are 13-bit (value-1); kernel/stride are 4-bit (value-1). */
  if ((((uint32_t)p->iw-1)|((uint32_t)p->ih-1)|((uint32_t)p->ow-1)|
       ((uint32_t)p->oh-1)|((uint32_t)p->c-1)) >> 13) return -2;
  if ((((uint32_t)p->kw-1)|((uint32_t)p->kh-1)|
       ((uint32_t)p->stride_x-1)|((uint32_t)p->stride_y-1)) >> 4) return -3;

  const uint32_t in_line_stride  = (uint32_t)p->iw * C2 * esz;            /* bytes/row */
  const uint32_t in_surf_stride  = (uint32_t)p->ih * p->iw * C2 * esz;    /* bytes/C-plane */
  const uint32_t out_surf_stride = (uint32_t)p->oh * p->ow * C2 * esz;    /* bytes/C-plane */

  /* arm the PPU + PPU_RDMA single-register groups (same 0xE as the DPU path) */
  ops[i++] = NPUOP(OP_REG_PPU,      0xE, PPU_S_POINTER);
  ops[i++] = NPUOP(OP_REG_PPU_RDMA, 0xE, PPU_RDMA_S_POINTER);

  /* PPU input/output cube geometry (field = dim - 1) */
  ops[i++] = NPUOP(OP_REG_PPU, (p->iw - 1) & 0x1FFF, PPU_DATA_CUBE_IN_WIDTH);
  ops[i++] = NPUOP(OP_REG_PPU, (p->ih - 1) & 0x1FFF, PPU_DATA_CUBE_IN_HEIGHT);
  ops[i++] = NPUOP(OP_REG_PPU, (p->c  - 1) & 0x1FFF, PPU_DATA_CUBE_IN_CHANNEL);
  ops[i++] = NPUOP(OP_REG_PPU, (p->ow - 1) & 0x1FFF, PPU_DATA_CUBE_OUT_WIDTH);
  ops[i++] = NPUOP(OP_REG_PPU, (p->oh - 1) & 0x1FFF, PPU_DATA_CUBE_OUT_HEIGHT);
  ops[i++] = NPUOP(OP_REG_PPU, (p->c  - 1) & 0x1FFF, PPU_DATA_CUBE_OUT_CHANNEL);

  /* operation mode: FLYING_MODE (bit4) always; POOLING_METHOD (bits[1:0]) max=1/avg=0 */
  ops[i++] = NPUOP(OP_REG_PPU, (1u << 4) | ((uint32_t)p->method & 0x3),
                   PPU_OPERATION_MODE_CFG);

  /* kernel + stride (each field = value-1): KW[3:0] KH[11:8] SX[19:16] SY[23:20] */
  uint32_t kcfg = ((uint32_t)(p->kw - 1) & 0xF)
                | (((uint32_t)(p->kh - 1) & 0xF) << 8)
                | (((uint32_t)(p->stride_x - 1) & 0xF) << 16)
                | (((uint32_t)(p->stride_y - 1) & 0xF) << 20);
  ops[i++] = NPUOP(OP_REG_PPU, kcfg, PPU_POOLING_KERNEL_CFG);

  /* average reciprocal (per-axis fp16(65536/k), see ppu_recip_kernel_fp16); 0 for max */
  uint32_t rw = (p->method == POOL_METHOD_AVG) ? (p->recip_w & 0x1FFFF) : 0;
  uint32_t rh = (p->method == POOL_METHOD_AVG) ? (p->recip_h & 0x1FFFF) : 0;
  ops[i++] = NPUOP(OP_REG_PPU, rw, PPU_RECIP_KERNEL_WIDTH);
  ops[i++] = NPUOP(OP_REG_PPU, rh, PPU_RECIP_KERNEL_HEIGHT);

  /* padding (counts, NOT -1): L[2:0] T[6:4] R[10:8] B[14:12] */
  uint32_t pad = ((uint32_t)p->pad_left & 0x7)
               | (((uint32_t)p->pad_top & 0x7) << 4)
               | (((uint32_t)p->pad_right & 0x7) << 8)
               | (((uint32_t)p->pad_bottom & 0x7) << 12);
  ops[i++] = NPUOP(OP_REG_PPU, pad, PPU_POOLING_PADDING_CFG);
  /* pad fill: 0 for avg; -inf (fp16 0xFC00) for max so a padded cell never wins */
  ops[i++] = NPUOP(OP_REG_PPU,
                   (p->method == POOL_METHOD_MAX && pad) ? 0x0000FC00u : 0,
                   PPU_PADDING_VALUE_1_CFG);
  ops[i++] = NPUOP(OP_REG_PPU, 0x0, PPU_PADDING_VALUE_2_CFG);

  /* output cube write-out (DST_BASE_ADDR holds the raw IOVA; field is [31:4], BOs are
   * page-aligned). DATA_FORMAT mirrors the out surf stride into INDEX_ADD[31:4] and sets
   * PROC_PRECISION[2:0]=2 (fp16); out_surf_stride is 16-aligned so the OR is clean. */
  ops[i++] = NPUOP(OP_REG_PPU, p->output_dma, PPU_DST_BASE_ADDR);
  ops[i++] = NPUOP(OP_REG_PPU, out_surf_stride, PPU_DST_SURF_STRIDE);
  ops[i++] = NPUOP(OP_REG_PPU, out_surf_stride | 0x2, PPU_DATA_FORMAT);
  ops[i++] = NPUOP(OP_REG_PPU, 0x3, PPU_MISC_CTRL);          /* BURST_LEN=3 */

  /* PPU_RDMA: the input feed (own SRC_BASE + strides; IN_PRECISION=2 fp16) */
  ops[i++] = NPUOP(OP_REG_PPU_RDMA, (p->iw - 1) & 0x1FFF, PPU_RDMA_CUBE_IN_WIDTH);
  ops[i++] = NPUOP(OP_REG_PPU_RDMA, (p->ih - 1) & 0x1FFF, PPU_RDMA_CUBE_IN_HEIGHT);
  ops[i++] = NPUOP(OP_REG_PPU_RDMA, (p->c  - 1) & 0x1FFF, PPU_RDMA_CUBE_IN_CHANNEL);
  ops[i++] = NPUOP(OP_REG_PPU_RDMA, p->input_dma, PPU_RDMA_SRC_BASE_ADDR);
  ops[i++] = NPUOP(OP_REG_PPU_RDMA, in_line_stride, PPU_RDMA_SRC_LINE_STRIDE);
  ops[i++] = NPUOP(OP_REG_PPU_RDMA, in_surf_stride, PPU_RDMA_SRC_SURF_STRIDE);
  ops[i++] = NPUOP(OP_REG_PPU_RDMA, 0x2, PPU_RDMA_DATA_FORMAT);

  /* PC trailer: enable PPU + PPU_RDMA only. 0x60 = PPU_OP_EN(b5)|PPU_RDMA_OP_EN(b6)
   * (the GLOBAL block-participation mask; no per-block OPERATION_ENABLE is emitted).
   * Framing mirrors gen_matmul_task's validated rocket-path trailer. */
  ops[i++] = NPUOP(OP_NONE, 0x0, 0x0);
  ops[i++] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
  ops[i++] = NPUOP(OP_40, 0x0, 0x0);
  ops[i++] = NPUOP(OP_ENABLE, 0x60, PC_OPERATION_ENABLE);

  p->task_count = i;
  return 0;
}
