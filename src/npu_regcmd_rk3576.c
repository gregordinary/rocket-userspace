// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * npu_regcmd_rk3576.c — RK3576 int8 CONV_2D register-command generator.
 *
 * A sibling of npu_regcmd.c's gen_conv2d_task/gen_conv2d_int8_fill pair, for the
 * RK3576's own CNA/CORE/DPU geometry-register encoding. Same descriptor structs,
 * same NPUOP word, same block targets, same PC trailer — different registers and
 * different packing, because the RK3576 re-packs the geometry registers at the
 * shared block bases.
 *
 * PROVENANCE. The encoding is transcribed from RKNN-Toolkit2 register programs
 * captured for known conv geometries on the RK3576 (the .rknn captures decoded in
 * tests/data/rk3576-vendor-capture/), cross-read against Ga Hing Woo's differential
 * register map for the same part. Every geometry, channel and constant register
 * emitted below is pinned by at least one capture; the host-only gate
 * (tests/regcmd_rk3576_gate.c) diffs this emitter against those programs.
 *
 * SCOPE, and why it is this narrow:
 *   - int8/uint8 is the shipping path. The captures are int8, so every non-int8
 *     field is a hardware sweep on top of that baseline rather than a transcription;
 *     the float datapath now computes exactly and its output packing does not, which
 *     the fp16 entry point in the header sets out.
 *   - The vendor's "normal" datapath (in_ch > 4). The first-conv ARGB path
 *     (in_ch <= 4) packs several of the same registers differently — it is its own
 *     sub-encoding — and is rejected rather than approximated.
 *   - One task. The task's input row window and output row count are the caller's
 *     (conv_params_t ih/oh vs ih_full/oh_full), which is what the RK3576 stream
 *     itself distinguishes.
 *
 * THE TRAILER. The vendor stream carries no PC trailer at all: the vendor kernel
 * fires the job through its own enable-mask/task-number in the task descriptor.
 * Under `rocket` it is PC_OPERATION_ENABLE that triggers compute, so this emitter
 * appends the same 4-word trailer the RK3588 generators use. Transcribe the vendor
 * program without it and the pipeline is configured and never fires.
 *
 * CHANNEL GRANULARITY. BOTH channel counts must be programmed as a multiple of 32 —
 * the group weight_conv_int8 pads its two channel axes to — and the caller's buffers
 * must follow. rocket_rk3576_pad_ic() / rocket_rk3576_pad_oc() give the counts to
 * program. No capture shows this, because every captured geometry already satisfies
 * it. [HW sweep, H96 MAX M9]
 *   - ic: pad the FEATURE CUBE out to the returned count with zeros. A partial ic
 *     group computes wrong at every geometry and kernel size — ic 8, 16, 17 and 48
 *     all fail while 32 and 64 are exact. ic=17 fails despite spanning two whole
 *     C2=16 surfaces, so the unit is the 32-channel group, not surface parity.
 *   - oc: size the OUTPUT BO and the COEFFICIENT BUFFER for the returned count, and
 *     pass it to rocket_rk3576_coeff_bytes() too, so a padded channel gets a C term
 *     rather than a gated-off group.
 * A partial oc group trips two different mechanisms, which is why it presents two
 * ways. At k=1 the DPU writes only output ROW 0 of the trailing group and leaves the
 * rest of the surface untouched (oc 8, 16, 40, 48). At k>1 the weights themselves
 * come out wrong instead, because WEIGHT_BYTES (`0x101C`) = ic*oc*kh*kw describes a
 * cube tighter than the padded one the caller supplies, and the truncation lands
 * between whole (kh,kw) planes rather than trimming oc inside each — which is
 * exactly why k=1 is the tolerant case (oc=24 and oc=56 are exact at k=1 and wrong
 * at k=3, and both are exact once padded).
 *
 * The vendor programs a partial ic group verbatim — its conv2d capture is ic=16 with
 * weight bytes ic*oc*kh*kw = the TIGHT size — so it packs a 16-channel weight group
 * where weight_conv_int8 pads to 32. Padding is how this cube expresses the same
 * conv; a tighter partial-group weight packing would be the other way, at half the
 * weight bytes for such a layer.
 *
 * TWO CORRECTIONS to the published RK3576 map, both settled by re-reading the
 * captures and both load-bearing for an emitter:
 *   - The output-height field is NOT halved. It is the number of output rows THIS
 *     task writes, minus one. The "halved" reading comes from a capture whose task
 *     happened to cover half the plane; a capture of the same conv as a single task
 *     writes the full oh-1 (CORE 0x301c hi = 0x27 = 39 for a 40-row output).
 *   - The feature-data address is at 0x1088 on BOTH datapaths, not at 0x1070 on the
 *     ARGB path. 0x1070 reads zero in every capture, including the one program that
 *     provably needs a non-zero feature address (a second row-slice, where 0x1088
 *     carries exactly that slice's byte offset).
 */

#include <stdlib.h>
#include <string.h>

#include "npu_hw.h"
#include "npu_cna.h"
#include "npu_dpu.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_log.h"

#include <math.h>

#define OP_REG_DPU_RDMA (BLOCK_DPU_RDMA | PC_OP_01)   /* 0x2001 */

/* ============================================================================
 * SECTION — RK3576 register offsets
 *
 * Named by the function the captures pin them to. Where an RK3588 register plays
 * the same role the comment names it, so the two encodings can be read side by
 * side; the offsets and the packing are NOT the RK3588's.
 * ==========================================================================*/

#define R76_CNA_S_POINTER      0x1004  /* == RK3588 CNA_S_POINTER              */
#define R76_CNA_CONV_MODE      0x100C  /* bit0 = depthwise                     */
#define R76_CNA_CONV_CON2      0x1010  /* 0xFFF const across every capture     */
#define R76_CNA_STRIDE         0x1014  /* (sy<<3)|sx — coincides with RK3588   */
#define R76_CNA_FORMAT         0x1018  /* format/window word; see cna_format() */
#define R76_CNA_WEIGHT_BYTES   0x101C  /* total weight bytes  (RK3588 0x1030)  */
#define R76_CNA_WEIGHT_ELEMS   0x1020  /* (dw?oc:ic)*kh*kw                     */
#define R76_CNA_KERNEL_OC      0x1024  /* kernel word | oc-1                   */
#define R76_CNA_SURF_CHANNEL   0x1028  /* (surf*ih)<<16 | ic-1                 */
#define R76_CNA_DATAIN_SIZE    0x102C  /* (iw-1)<<16 | (ih_task-1)             */
#define R76_CNA_WBPK_OW        0x1030  /* weight bytes per kernel | ow-1       */
#define R76_CNA_DATAOUT_ATOMS  0x1034  /* ow*oh_task - 1  (RK3588 0x102C)      */
#define R76_CNA_WEIGHT_SIZE2   0x1038  /* 0x07 const for a single task         */
#define R76_CNA_CBUF_ENTRIES   0x103C  /* data_entries<<16 (RK3588 0x1044 lo)  */
#define R76_CNA_CBUF_CON0      0x1040  /* CBUF banking (RK3588 0x1040)         */
#define R76_CNA_IW_ENTRIES     0x1044  /* iw<<16 | data_entries                */
#define R76_CNA_CVT_CON0       0x1048  /* == RK3588 CNA_CVT_CON0 word, 0x104C  */
#define R76_CNA_CVT_SCALE01    0x104C  /* scale1<<16 | scale0                  */
#define R76_CNA_CVT_SCALE23    0x1050  /* scale3<<16 | scale2                  */
#define R76_CNA_CVT_OFFSET0    0x1054
#define R76_CNA_CVT_OFFSET1    0x1058
#define R76_CNA_CVT_OFFSET2    0x105C
#define R76_CNA_ZERO_1060      0x1060
#define R76_CNA_ZERO_1064      0x1064
#define R76_CNA_ZERO_1068      0x1068
#define R76_CNA_ZERO_106C      0x106C
#define R76_CNA_ZERO_1070      0x1070  /* NOT the feature address — see header  */
#define R76_CNA_ZERO_1074      0x1074
#define R76_CNA_DMA_SIZE       0x1078  /* (iw-1)<<16 | (ih_task-1)             */
#define R76_CNA_DMA_CHANNEL    0x107C  /* ic-1  (RK3588 CNA_FC_DATA_SIZE1)     */
#define R76_CNA_PAD_CON0       0x1080  /* right<<24|bottom<<16|left<<8|top     */
#define R76_CNA_PAD_CON1       0x1084  /* pad constant (RK3588 0x1184)         */
#define R76_CNA_FEATURE_ADDR   0x1088  /* feature data base (RK3588 0x1070)    */
#define R76_CNA_DMA_CON0       0x108C  /* burst lens (RK3588 CNA_DMA_CON0)     */
#define R76_CNA_LINE_STRIDE    0x1090  /* iw*4 (RK3588 CNA_DMA_CON1)           */
#define R76_CNA_SURF_FULL      0x1094  /* iw*ih_full                           */
#define R76_CNA_SURF_TASK      0x1098  /* round4(iw*ih_task)                   */
#define R76_CNA_ZERO_109C      0x109C
#define R76_CNA_DCOMP_CTRL     0x1100  /* == RK3588                            */
#define R76_CNA_DCOMP_REGNUM   0x1104  /* == RK3588                            */
#define R76_CNA_DCOMP_ADDR0    0x1110  /* weight base — coincides with RK3588  */
#define R76_CNA_DCOMP_AMOUNT   0x1140  /* == RK3588                            */
#define R76_CNA_DCOMP_AMOUNT1  0x1144  /* == RK3588                            */
#define R76_CNA_DATAIN_FULL    0x118C  /* (iw-1) in BOTH halves                */

#define R76_CORE_S_POINTER     0x3004  /* == RK3588 CORE_S_POINTER             */
#define R76_CORE_MISC_CFG      0x3018  /* RK3588 CORE_MISC_CFG is 0x3010       */
#define R76_CORE_DATAOUT_SIZE0 0x301C  /* (oh_task-1)<<16 | (ow-1)             */
#define R76_CORE_DATAOUT_SIZE1 0x3020  /* oc-1                                 */
#define R76_CORE_CLIP_TRUNCATE 0x3024

#define R76_DPU_S_POINTER      0x4004  /* == RK3588 DPU_S_POINTER              */
#define R76_DPU_FEATURE_MODE   0x400C  /* == RK3588 offset, different value    */
#define R76_DPU_DATA_FORMAT    0x4010  /* 0 for the all-int8 program           */
#define R76_DPU_OFFSET_PEND    0x4014
#define R76_DPU_DST_BASE_ADDR  0x4018  /* RK3588 DPU_DST_BASE_ADD is 0x4020    */
#define R76_DPU_DST_SURF       0x401C  /* ow*oh_full                           */
#define R76_DPU_CUBE_WIDTH     0x4020  /* ow-1                                 */
#define R76_DPU_CUBE_HEIGHT    0x4024  /* oh_task-1                            */
#define R76_DPU_CUBE_NOTCH     0x4028
#define R76_DPU_CUBE_CHANNEL   0x402C  /* oc-1                                 */
#define R76_DPU_WDMA_SIZE0     0x4030  /* (oc-1)<<16 | mode const              */
#define R76_DPU_WDMA_SIZE1     0x4034  /* (oh_task-1)<<16 | (ow-1)             */
#define R76_DPU_NOTCH_CFG      0x4038
#define R76_DPU_ZERO_403C      0x403C
#define R76_DPU_BS_ALU_CFG     0x4044
#define R76_DPU_BS_MIN         0x4048
#define R76_DPU_BS_MAX         0x404C
#define R76_DPU_BS_CFG         0x4050
#define R76_DPU_BN_MIN         0x4058
#define R76_DPU_BN_MAX         0x405C
#define R76_DPU_BN_CFG         0x4060
#define R76_DPU_EW_MIN         0x406C
#define R76_DPU_EW_MAX         0x4070
#define R76_DPU_EW_MIN2        0x4074
#define R76_DPU_EW_MAX2        0x4078
#define R76_DPU_EW_CFG         0x407C
/* EW_CFG with the stage's own bypass (bit 0) and EW_LUT_BYPASS (bit 7) cleared — the
 * RK3588's field layout at a moved offset. This is what turns the LUT on. */
#define R76_DPU_EW_CFG_LUT     0x01004140u
#define R76_DPU_EW_CVT_OFFSET  0x4080
#define R76_DPU_EW_CVT_SCALE   0x4084
#define R76_DPU_EW_CLAMP_MIN   0x4088
#define R76_DPU_EW_CLAMP_MAX   0x408C
#define R76_DPU_EW_OP_VALUE0   0x4090
#define R76_DPU_EW_OP_VALUE1   0x4094
#define R76_DPU_ZERO_409C      0x409C
#define R76_DPU_OUT_CLAMP_MIN  0x40A4
#define R76_DPU_OUT_CLAMP_MAX  0x40A8
#define R76_DPU_OUT_CVT_OFFSET 0x40AC  /* RK3588 DPU_OUT_CVT_OFFSET is 0x4080  */
#define R76_DPU_OUT_CVT_SCALE  0x40B0  /* RK3588 0x4084                        */
#define R76_DPU_OUT_CVT_SHIFT  0x40B4  /* RK3588 0x4088                        */
/* ---- the DPU LUT bank ----
 * `0x4100` selects a table and a start offset and every write to `0x4104` stores one
 * entry and advances — the load program's window. The rest carry the input map a
 * consuming convolution reads: the index step, the two tables' domains (four lanes
 * each) and the two overflow clamps. */
#define R76_LUT_ACCESS_CFG     0x4100
#define R76_LUT_ACCESS_DATA    0x4104
#define R76_LUT_CFG            0x4108
#define R76_LUT_INFO           0x410C  /* two index selects: the step is 2^sel  */
#define R76_LUT_LE_START       0x4110  /* four lanes, 0x4110-0x411C             */
#define R76_LUT_LO_END         0x4140  /* four lanes, 0x4140-0x414C             */
#define R76_LUT_UNDERFLOW      0x4188  /* the value below LE_START, and 0x418C  */
#define R76_LUT_OVERFLOW       0x4190  /* the value at or above LO_END, + 0x4194*/
#define R76_LUT_SELECT_LO      0x00020000u
#define R76_LUT_SELECT_HI      0x00030000u
/* The load program's own LUT_CFG: the table is open for writing, not for reading. */
#define R76_LUT_CFG_LOAD       0x00ff0000u
/* A consuming convolution's LUT_CFG. [Manufactured capture, every activation] */
#define R76_LUT_CFG_USE        0x02000006u

#define R76_DPU_SURFACE_ADD    0x40B8  /* RK3588 DPU_SURFACE_ADD is 0x40C0     */
#define R76_DPU_ZERO_40BC      0x40BC
#define R76_DPU_CONST_40C0     0x40C0
#define R76_DPU_ZERO_40C8      0x40C8
#define R76_DPU_ZERO_40CC      0x40CC
#define R76_DPU_CONST_40D0     0x40D0

/* The coefficient buffer at BS_BASE_ADDR: 64-byte groups of 8 output channels, A
 * (int32 bias) at (oc%8)*4, B (weight zero point) at +32, C (multiplier) at +48.
 *
 * THE DEPTHWISE PATH READS A DIFFERENT GROUP. It is 48 bytes for the same 8 channels
 * and carries no B at all: A (int32) at (oc%8)*4 and C (int16) at 32 + (oc%8)*2.
 * Handing it the 64-byte group makes every channel past the first eight read its A
 * and its C out of the wrong group — the two indices drift apart at different rates,
 * because A strides 4 bytes and C strides 2 — so most channels multiply a bias by a
 * zero and vanish, a few square their own bias, and the C multipliers of one group
 * are read as the biases of the next. [HW sweep, H96 MAX M9] */
#define R76_COEFF_GROUP_OC       8   /* output channels per group, both layouts */
#define R76_COEFF_GROUP_BYTES    64
#define R76_COEFF_B_OFFSET       32
#define R76_COEFF_C_OFFSET       48
#define R76_COEFF_GROUP_BYTES_DW 48
#define R76_COEFF_C_OFFSET_DW    32

/* Byte offset, inside that same buffer, of the zeroed group BS_BASE_ADDR1 reads. */
static size_t r76_shift_offset_p(unsigned oc, int dw)
{
    unsigned g = (oc + R76_COEFF_GROUP_OC - 1) / R76_COEFF_GROUP_OC;
    return (size_t)g * (dw ? R76_COEFF_GROUP_BYTES_DW : R76_COEFF_GROUP_BYTES);
}

static size_t r76_shift_offset(unsigned oc) { return r76_shift_offset_p(oc, 0); }

#define R76_RDMA_S_POINTER     0x5004  /* == RK3588 DPU_RDMA_S_POINTER         */
#define R76_RDMA_CUBE_WIDTH    0x500C  /* == RK3588 offset                     */
#define R76_RDMA_CUBE_HEIGHT   0x5010  /* == RK3588 offset                     */
#define R76_RDMA_CUBE_CHANNEL  0x5014  /* == RK3588 offset                     */
#define R76_RDMA_SRC_BASE_ADDR 0x5018  /* == RK3588 offset                     */
#define R76_RDMA_BRDMA_CFG     0x501C  /* == RK3588 offset, different value    */
#define R76_RDMA_BS_BASE_ADDR  0x5020  /* per-OC int32 bias cube               */
#define R76_RDMA_BS_BASE_ADDR1 0x5024
#define R76_RDMA_NRDMA_CFG     0x5028
#define R76_RDMA_BN_BASE_ADDR  0x502C
#define R76_RDMA_ZERO_5030     0x5030
#define R76_RDMA_ERDMA_CFG     0x5034
#define R76_RDMA_EW_BASE_ADDR  0x5038
#define R76_RDMA_EW_SURF       0x5040
#define R76_RDMA_FEATURE_MODE  0x5044
#define R76_RDMA_SRC_DMA_CFG   0x5048
#define R76_RDMA_SURF_NOTCH    0x504C
#define R76_RDMA_PAD_CFG       0x5064
#define R76_RDMA_EW_SURF_NOTCH 0x506C
#define R76_RDMA_ZERO_5078     0x5078
#define R76_RDMA_ZERO_507C     0x507C

/* ============================================================================
 * SECTION — mode constants
 *
 * Words that the captures show varying only between direct and depthwise, with no
 * further geometry dependence. Emitted verbatim rather than decomposed into
 * fields: the sub-bit meanings are not pinned by the captures, and inventing a
 * field decomposition would claim more than the evidence supports.
 *
 * ONE OF THEM IS NOT A CONSTANT. The depthwise BS config carries a 2-bit channel
 * field, which the original captures could not separate from the stride because
 * every C=32 program in them is stride 1 and every C=64 one is stride 2. Manufactured
 * captures across 22 channel counts at both strides separate the two: the field
 * tracks the CHANNEL COUNT and is flat in the stride. See r76_dw_bs_cfg().
 * ==========================================================================*/

#define R76_CORE_MISC_DIRECT   0x10000001u
#define R76_CORE_MISC_DW       0x1000000Au
#define R76_CORE_MISC_ARGB     0x10000081u  /* one extra bit below PROC_PRECISION */
#define R76_DPU_FMODE_DIRECT   0x40000004u
#define R76_DPU_FMODE_DW       0x4000000Cu
#define R76_DPU_WDMA0_DIRECT   0x00000710u
#define R76_DPU_WDMA0_DW       0x00000310u
#define R76_DPU_NOTCH_DIRECT   0x00120080u
#define R76_DPU_NOTCH_DW       0x00100092u
#define R76_DPU_BS_CFG_DIRECT  0x80011111u
/* The depthwise base word, at ONE 16-channel group. Bits[9:8] carry the group count
 * minus one, modulo 4 — see r76_dw_bs_cfg(), which is what the emitter calls. */
#define R76_DPU_BS_CFG_DW      0x00013033u

/* The float datapath's own two words, transcribed from a vendor float capture and
 * constant across every geometry it emits (ic 8-64, oc 32/64, k 1/3, planes 16 and
 * 32). Each is a ONE-BIT-CLASS change from the direct integer word beside it, and
 * each is load-bearing on its own: with the integer word in either slot the
 * contraction reads every feature surface TWICE and skips the odd ones.
 *
 *   0x4038  low half 0x0092 rather than 0x0080. The high half is free (0x0012 also
 *           computes) but must be non-zero, or the DPU writes nothing at all.
 *   0x4050  bit 31 CLEAR. That bit alone is the defect: leaving it set with every
 *           other float field right returns +inf, the accumulator having summed the
 *           doubled reads. The low nibbles are the write-extent field and do not
 *           enter here.
 * [source-confirmed, RKNN-Toolkit2 rk3576 float build; HW sweep, H96 MAX M9] */
#define R76_DPU_NOTCH_FLOAT    0x00100092u
#define R76_DPU_BS_CFG_FLOAT   0x00021111u
#define R76_RDMA_BRDMA_DIRECT  0x00000710u
#define R76_RDMA_BRDMA_DW      0x00000510u
#define R76_RDMA_FMODE_DIRECT  0x40000010u
#define R76_RDMA_FMODE_DW      0x40000012u
/* Two DPU_RDMA words the vendor's FLOAT programs carry a different constant in, on the
 * direct and the first-conv path alike and at every geometry. Both are reproduced and
 * both are bit-exact on hardware.
 *
 * The THIRD one is not, and it is the case that shows register fidelity is not the
 * goal — computing is. BRDMA_CFG (0x501C) reads 0x100 in every vendor float program
 * against the integer path's 0x710, and emitting the vendor's value makes the DPU
 * write NOTHING AT ALL: the surface comes back untouched, with no fault and nothing in
 * dmesg. It is the BS operand reader, and it does not stand alone — paired with the BS
 * ALU config the same programs carry (0x4044 = 2, where the integer path uses 1) the
 * writer runs again, but the arithmetic is then wrong on about a tenth of the surface
 * (1866/2048, worst relative error 0.016). So the vendor's float epilogue is a
 * different BS ARRANGEMENT, not a different constant, and this library's A/B/C
 * coefficient group is packed for the integer one — which is bit-exact at 0x710 with
 * either ALU mode. Keep 0x710 until the whole arrangement is decoded together.
 * [HW sweep, H96 MAX M9; leave-one-out over the six float-path registers] */
#define R76_RDMA_ERDMA_FLOAT   0x00000001u
#define R76_RDMA_FMODE_FLOAT   0x40010050u

/* ============================================================================
 * SECTION — precision
 *
 * EVERY CAPTURE IS int8. The precision fields are therefore all zero in them and
 * none of this is transcribed: it is the RK3588's own field packing carried over to
 * the RK3576 words that play the same role, which is the same reasoning that turned
 * out right for CNA_CVT_CON0 and wrong for nothing yet — but it is an inference, and
 * a wrong one here writes a full, correctly sized, WRONG surface rather than
 * faulting.
 *
 * Three of the four words decode consistently under the RK3588 packing at int8,
 * which is what makes it the leading candidate rather than a guess:
 *
 *   CNA 0x100C   RK3588 CNA_CONV_CON1 = proc<<7 | in<<4 | conv_mode.  The captures
 *                carry exactly conv_mode (0 direct, 1 depthwise) with both precision
 *                fields zero, as int8 requires.
 *   CORE 0x3018  RK3588 CORE_MISC_CFG = proc<<8 | qd_en.  Captured bits [10:8] are
 *                zero.  Bit 0 is SET in the RK3576 int8 word where the RK3588 clears
 *                qd_en for int8 — so either the bit means something else here or the
 *                low nibble is not qd_en at all.  Sweep it.
 *   DPU 0x4010   RK3588 DPU_DATA_FORMAT = out<<29 | in<<26 | proc.  Zero in every
 *                capture, which is all-int8 under that packing.
 *   DPU 0x400C   feature mode; carries conv_mode, not precision.
 *
 * Everything here is one register value, so the sweep harness moves any of it with
 * ROCKET_RK3576_SET rather than a rebuild. See tests/rk3576_fp16_sweep.c.
 * ==========================================================================*/

/* The value PROC_PRECISION carries for a given operand precision. It is the OPERAND
 * width, not the datapath width: a float conv programs float16 here and multiplies
 * fp16 operands, and programming the fp32 value instead reads each feature surface
 * twice. Transcribed from a vendor float capture; integer precisions keep their own
 * value, which is what every int8 capture holds.
 * [source-confirmed, RKNN-Toolkit2 rk3576 float build; HW sweep, H96 MAX M9] */
static inline unsigned r76_proc_precision(unsigned prec)
{
    return prec;
}

/* Bytes per feature/weight element. The int8 column is measured; the 2-byte column
 * is what the RK3588 uses for the same precisions. */
static inline unsigned r76_elem_bytes(unsigned prec)
{
    switch (prec) {
        case precision_float16:
        case precision_bfloat16:
        case precision_int16:   return 2;
        default:                return 1;   /* int8, int4 (packed by the caller) */
    }
}

/* Whether a precision runs the float datapath — which decides the cube groups, the
 * coefficient C unit and the DPU narrowing, none of which the integer path shares. */
static inline int r76_is_float(unsigned prec)
{
    return prec == precision_float16 || prec == precision_bfloat16;
}

/* The FLOAT weight cube's channel groups, and the float feature cube's lane count.
 * None of the three is the int8 value: the int8 cube groups both channel axes by 32
 * and packs 16 lanes into a 16-byte atom. See the float weight cube section below.
 *
 * The OUTPUT group is observable only at k > 1 and at a full input-channel group.
 * The kernel index sits between it and the (oc2, ic2) pair, so at k = 1 it cancels —
 * (oc/G)*G + oc%G = oc for any G — and a sweep there reports whatever it started
 * with while testing nothing. It is 16, read off the part at ic = 16 with k = 3 and
 * k = 5. [HW sweep, H96 MAX M9] */
#define R76_FP16_W_OC_GROUP 16u
#define R76_FP16_W_IC_GROUP 16u
#define R76_FP16_FEAT_C2    8u

/* The depthwise weight cube's channel group — 32, decoded from captures with a
 * unique value per (channel, tap). See rocket_rk3576_weight_dw(). */
#define R76_DW_W_GROUP      32u
/* The INT8 cube's channel group is SIXTY-FOUR, which the float one's 32 is not. Two
 * shapes hide the difference and both were the ones already passing: at one group the
 * two layouts coincide, and at k=1 there is a single tap so the group base and the tap
 * stride cannot be told apart. C=64 k=3 separates them, and there `rk3576_conv_gate
 * dwmap` predicts 576 of 576 (channel, tap) positions at 64 and 64 of 576 at 32.
 * [HW sweep, H96 MAX M9] */
#define R76_DW_W_GROUP_INT8 64u

/* ============================================================================
 * SECTION — derived geometry
 * ==========================================================================*/

/* CBUF data entries: the feature row's 64-byte granule count. Same quantity the
 * RK3588 emitter puts in CNA_CBUF_CON1; the RK3576 writes it twice (0x103c hi and
 * 0x1044 lo). Rounded UP — every capture has iw*ic a multiple of 64, so the gate
 * cannot tell ceil from floor, and rounding up is the side that cannot under-size
 * a DMA.
 *
 * The element size enters here and nowhere else on the feature side: the line
 * stride (0x1090, iw*4 in 4-byte units) and both surface strides (0x1094/0x1098, in
 * 16-byte units) count ATOMS, and an atom stays 16 bytes when C2 halves from 16
 * int8 lanes to 8 fp16 ones — so those three words are element-size independent.
 * The RK3588 makes the same change in the same one place, dividing by 32 instead of
 * 64 for fp16 against the same iw*ic. */
static inline unsigned r76_data_entries_p(unsigned iw, unsigned ic, unsigned elem)
{
    return (iw * ic * elem + 63u) / 64u;
}

static inline unsigned r76_data_entries(unsigned iw, unsigned ic)
{
    return r76_data_entries_p(iw, ic, 1u);
}

/* ============================================================================
 * SECTION — the depthwise channel granules
 *
 * The depthwise path rounds its channel count in TWO different places, and the two
 * roundings are not the same number. Both are transcribed from manufactured vendor
 * captures at 22 channel counts from 8 to 256 (tests/data/rk3576-vendor-capture/dw/,
 * rebuilt by mkdw.py). [source-confirmed, RKNN-Toolkit2 rk3576 depthwise builds]
 *
 *   r76_dw_cw  — the WEIGHT granule, C rounded up to 16. Sizes the weight cube
 *                (0x101C, 0x1020) and the per-kernel byte term (0x1030 hi).
 *
 *   r76_dw_cf  — the FEATURE granule. C rounded up to 16, and then, if that lands on
 *                a count whose 16-channel group index is 3 mod 4, up one group more.
 *                Sizes the CBUF entry count (0x1028, 0x103C, 0x1044). Read as
 *                hardware: channels are fetched in blocks of 64, and the trailing
 *                partial block holds 1, 2 or 4 sixteens — never 3 — so 48 rounds to
 *                64 and 112 to 128 while 80 and 144 stay put.
 *
 * The 2-bit field in the depthwise BS config (0x4050 bits[9:8]) is the group count
 * MINUS ONE, taken modulo 4, and computed on the UNROUNDED group count: C=48 carries
 * 2 there while its feature allocation is 64's. The emitter carried the C=32 word
 * unconditionally, which programmed the BS stage with a word the vendor never uses
 * above C=32 — and a misconfigured BS stage on this part writes a full but entirely
 * empty surface.
 * ==========================================================================*/

static inline unsigned r76_dw_cw(unsigned c) { return (c + 15u) & ~15u; }

static inline unsigned r76_dw_cf(unsigned c)
{
    unsigned g = r76_dw_cw(c) / 16u;
    return 16u * (g + ((g & 3u) == 3u ? 1u : 0u));
}

static inline uint32_t r76_dw_bs_cfg(unsigned c)
{
    unsigned g = r76_dw_cw(c) / 16u;
    return R76_DPU_BS_CFG_DW | (((g - 1u) & 3u) << 8);
}

/* Weight bytes per kernel (0x1030 hi). Direct is ic*kh*kw*2 across both non-dw
 * captures (k=3 and k=5). Depthwise is kh*kw*round16(ic)/8, transcribed at k=1, 3, 5
 * and 7 and at every channel count from 8 to 256. */
static inline unsigned r76_wbpk(unsigned ic, unsigned oc, unsigned kh, unsigned kw, int dw)
{
    (void)oc;
    return dw ? (kh * kw * r76_dw_cw(ic)) / 8u : (ic * kh * kw * 2u);
}

/* Trailing pad ACTUALLY CONSUMED by this task's last window, which is what the
 * RK3576 pad word carries — not the configured padding. The last window starts at
 * (out-1)*stride in padded coordinates and runs k wide; anything past
 * pad_lead + in_extent is pad. Reproduces all four captures, including the
 * depthwise one whose row window stops short of the image bottom (bottom pad 0
 * while right pad is 1). */
static inline unsigned r76_trail_pad(unsigned out, unsigned stride, unsigned k,
                                     unsigned pad_lead, unsigned in_extent)
{
    unsigned reach = (out - 1u) * stride + k;
    unsigned have  = pad_lead + in_extent;
    return reach > have ? reach - have : 0u;
}

/* 0x1018. Two values across the captures: 0x40000404 when the task covers the whole
 * feature plane, 0x40000505 when it reads a row window of a taller one, and the
 * vendor's sliced ARGB streams add 0x040b / 0x0606 / 0x060c. Reproduced here so the
 * emitted program stays byte-identical to the captures — but on the direct int8 path
 * only BIT 30 is live. Clearing it corrupts the whole surface; the low 16 bits are
 * don't-care, with 0x0404, 0x0505, 0x040b and 0x0000 all bit-exact on windowed and
 * un-windowed tasks alike. [HW sweep, H96 MAX M9] */
static inline uint32_t r76_cna_format(int windowed)
{
    return windowed ? 0x40000505u : 0x40000404u;
}

/* ============================================================================
 * SECTION — the first-conv ARGB sub-encoding
 *
 * A convolution whose input is a PACKED IMAGE — 3 or 4 interleaved bytes per pixel,
 * not the NC1HWC2 cube every other layer reads — runs on its own CNA datapath. The
 * RK3588 has the same one (Mesa drives it as CNA_CONV_CON1 NONALIGN_DMA |
 * GROUP_LINE_OFF | ARGB_IN for a 1-channel input); the RK3576 captures show its
 * 3-channel form. Twelve captured programs pin it, at two image sizes (224x224 and
 * 64x64) and two row splits.
 *
 * It matters because it is the only way a vision model's stem runs on this part at
 * all: the normal path needs ic to be a multiple of 32, and an RGB image is 3.
 *
 * WHAT THE DATAPATH DOES. The CNA reads the packed row straight out of DDR, and the
 * CVT — bypassed on every other layer — expands each pixel to FOUR int8 lanes and
 * applies a per-channel scale and offset while doing it. The kernel's horizontal
 * extent is then folded into the channel axis: the conv the MAC sees is kh x 1 over
 * 4*kw channels, which is why the programmed channel count (0x1028) and the DMA's
 * channel count (0x107C) disagree in these programs — 12 against 3 at kw=3.
 *
 * That fold is the one inference here that the captures cannot separate. Every ARGB
 * capture is kw=3, so `4*kw` and the constant 12 fit equally, as do several readings
 * of the 48-element weight kernel. The mechanism above is what makes 4*kw the
 * reading rather than a curve fit — the register pair only makes sense as "4 lanes
 * per pixel column, kw columns" — but a first conv at another kernel width is
 * UNVALIDATED and should be checked against a CPU model.
 *
 * THE WEIGHT CUBE. The found programs pin only its SIZE and stride —
 * oc * kh * round16(4*kw) bytes, one 16-byte row per kernel row per output channel —
 * because the weights live in a separate BO no capture carries. The FLOAT cube's own
 * layout is decoded, from manufactured captures whose weights are unique per position
 * (tests/data/rk3576-vendor-capture/argb/mkargb.py):
 *
 *     slot(oc, c, kh, kw) = (oc/16) * (KH*KW*64)
 *                         + kh * (KW*64)
 *                         + kw * 64
 *                         + (oc%16) * 4
 *                         + c
 *
 * a weight in a SIXTEEN-BIT slot, four lanes per (output channel, tap) with lane c
 * carrying image channel c and any lane past ic left don't-care, output channels
 * INTERLEAVED in groups of sixteen inside one tap, and the tap axis kh-outer. The 64
 * is 16 output channels x 4 lanes. Note that `4*kw` is NOT rounded up to 16 here, so
 * this layout's element count is not the byte size the register program declares.
 * Reproduced at ic 3 and 4, k 1/3/5 and oc 16/32. [source-confirmed, RKNN-Toolkit2
 * rk3576 float build]
 *
 * WHAT IS STILL NOT TRANSCRIBED: the INT8 cube. It is not this one — the depthwise
 * path already showed that this part's int8 and float cubes are different objects —
 * and a quantized ARGB capture does take the ARGB path (0x100C reads 0x2000a006,
 * exactly the found value) but its weights do not survive as a findable value set.
 * So the int8 first conv stays refused rather than shipping a guessed cube.
 *
 * ONE CORRECTION from these captures: the low field of 0x100C is not precision-
 * independent. The found int8 ARGB programs carry 0x2000a006 and the float ones
 * 0x2020a122, so it is 6 at int8 and 2 at fp16 — bit 2 clears on the float path while
 * GROUP_LINE_OFF and ARGB_IN stay where they are.
 * ==========================================================================*/

/* The CVT expands every pixel to this many int8 lanes, whatever the image carries. */
#define R76_ARGB_LANES 4u

/* CVT truncate: the scale is Q14, so the shift that returns (x + offset) unchanged
 * at unit gain is 14. Captured as 14 in all three live channel fields. */
#define R76_ARGB_CVT_TRUNCATE 14u
#define R76_ARGB_CVT_SCALE    (1u << R76_ARGB_CVT_TRUNCATE)   /* Q14 unity */

/* CNA_CONV_CON1 (0x100C) fields, which the RK3576 keeps at the RK3588's positions —
 * one of the few geometry-block registers that does. GROUP_LINE_OFF is bit 29 and
 * ARGB_IN is bits[15:12] on both parts, and the captured ARGB_IN of 0xA sits exactly
 * where Mesa's 1-channel 0x8 does, one per extra image channel. */
#define R76_CNA_CON1_GROUP_LINE_OFF (1u << 29)
#define R76_CNA_CON1_ARGB_IN_SHIFT  12
/* The mode nibble, beside direct (0) and depthwise (1). It is not precision-
 * independent: bit 2 is set on the int8 first conv and clear on the float one, which
 * reads as the fold — the int8 path folds the kernel width into the channel axis and
 * the float path does not. Both values are transcribed, neither is swept. */
#define R76_ARGB_CONV_MODE          6u
#define R76_ARGB_CONV_MODE_FLOAT    2u

/* Bit 21 of the same word, set in every vendor FLOAT program and in no integer one.
 * It is not optional and it is not implied by the two precision fields: with the
 * precisions right and this bit clear the contraction still reads each surface
 * twice, exactly as it does with the bit set and a precision wrong.
 * [source-confirmed, RKNN-Toolkit2 rk3576 float build; HW sweep, H96 MAX M9] */
#define R76_CNA_CON1_FLOAT_EN       (1u << 21)

/* The channel count PROGRAMMED at 0x1028.
 *
 * The two precisions do NOT agree here, and the difference is the whole shape of the
 * datapath. At INT8 the kernel's width is folded into the channel axis — 4 lanes per
 * pixel column, kw columns, so 12 at kw=3, which is what every int8 capture carries
 * against a feature DMA of 3. At FP16 there is no fold: the programmed count is the
 * image's own channel count and the taps stay on the kernel axis, which is also what
 * the decoded float weight cube says (it carries explicit kh and kw axes with four
 * lanes inside a tap, where a folded cube would carry the columns inside the lanes).
 * Separated by manufactured captures at kw 1/3/5/7, all of which program ic.
 * [source-confirmed, RKNN-Toolkit2 rk3576 float build] */
static inline unsigned r76_argb_ic_prog(unsigned kw, unsigned ic_img, int is_float)
{
    return is_float ? ic_img : R76_ARGB_LANES * kw;
}

/* Weight ELEMENTS per output channel. At int8 the fold pads each kernel row out to a
 * 16-byte row; at fp16 it is a dense 4 lanes per tap, which is why the float cube's
 * `4*kw` is not rounded up to 16. */
/* The int8 kernel ROW: 4 lanes per tap column padded up to a 16-byte row. It is the
 * stride of the int8 weight cube's output-channel axis as well as the emitter's own
 * declared row, which is why it has a name. Separated from a constant 16 by k=5 and
 * k=7, where it is 32. [HW sweep, H96 MAX M9] */
static inline unsigned r76_argb_weight_row(unsigned kw)
{
    return (R76_ARGB_LANES * kw + 15u) & ~15u;
}

static inline unsigned r76_argb_weight_elems(unsigned kh, unsigned kw, int is_float)
{
    return is_float ? R76_ARGB_LANES * kh * kw
                    : kh * r76_argb_weight_row(kw);
}

/* CBUF granules per feature row: the image row expanded to 4 lanes per pixel, in
 * 64-byte granules.
 *
 * The +1 is the INT8 fold's lookahead — the last output column of the row reads kw
 * pixels, which reach past the granule its own pixel sits in — so it goes with the
 * fold and the float path has none. Measured at three widths on the float path
 * (iw 16/32/64 give 2/4/8 with no k term at k 1/3/5/7), and both captured int8
 * widths carry exactly +1. */
static inline unsigned r76_argb_entries(unsigned iw, unsigned elem, int is_float)
{
    return iw * R76_ARGB_LANES * elem / 64u + (is_float ? 0u : 1u);
}

/* DDR row stride of the PACKED image, in 16-byte units. Not the normal path's
 * iw*4: that counts one NC1HWC2 channel-group row (iw*16 bytes) in 4-byte words,
 * and there is only one surface here. The element size enters because a float
 * packed image is 3 or 4 interleaved fp16 per pixel, not bytes. */
static inline unsigned r76_argb_line_stride(unsigned iw, unsigned ic_img, unsigned elem)
{
    return iw * ic_img * elem / 16u;
}

/* CNA_CVT_CON0 (0x1048), which packs exactly as the RK3588's: four 6-bit truncate
 * fields above the four mode bits. The live image channels take truncate 14 against
 * the Q14 scale; the unused lanes stay at 0, and CVT_BYPASS is CLEAR — this is the
 * only captured program in which the converter runs at all. */
static inline uint32_t r76_argb_cvt_con0(unsigned ic_img)
{
    uint32_t v = 0;
    unsigned c;
    for (c = 0; c < ic_img && c < 4u; c++)
        v |= (R76_ARGB_CVT_TRUNCATE & 0x3Fu) << (4 + 6 * c);
    return v;
}

/* ============================================================================
 * SECTION — address-placement RE knobs
 *
 * Every address register in the captures reads zero — the vendor patches them at
 * load time — so which offset carries which address is an inference, not a
 * measurement. One of them IS measured: a row-sliced capture puts that slice's
 * exact byte offset in 0x1088, which pins the feature address. The weight address
 * at 0x1110 is assumed by analogy with the RK3588, and the same kind of assumption
 * about the feature address (0x1070) turned out to be wrong.
 *
 * So the placement is sweepable on hardware:
 *   ROCKET_RK3576_WADDR_REG=0x____   put the weight address at this offset instead
 *   ROCKET_RK3576_FADDR_REG=0x____   likewise for the feature address
 *   ROCKET_RK3576_ADDR_SHOTGUN=1     write the weight address into EVERY register
 *                                    the captures leave at zero, at once — one run
 *                                    that says whether the address is the problem
 *                                    at all, before bisecting which offset it is.
 * ==========================================================================*/

/* Registers the vendor stream writes as zero: the candidate slots for an address
 * the captures cannot show us. */
static const uint16_t r76_zero_regs[] = {
    0x1060, 0x1064, 0x1068, 0x106C, 0x1070, 0x1074,
    0x109C, 0x1100, 0x1104, 0x1140, 0x1144,
};

static uint16_t r76_env_reg(const char *name, uint16_t dflt)
{
    const char *e = getenv(name);
    return e ? (uint16_t)strtoul(e, NULL, 0) : dflt;
}

/* Value for one of the normally-zero registers, after the knobs above have had
 * their say. Keeps the emit sequence a straight-line list. */
static uint32_t r76_zero_value(uint16_t reg, const uint32_t *zval)
{
    size_t i;
    for (i = 0; i < sizeof r76_zero_regs / sizeof r76_zero_regs[0]; i++)
        if (r76_zero_regs[i] == reg) return zval[i];
    return 0;
}

/* ============================================================================
 * SECTION — CBUF budget planning
 *
 * The feature budget is not a hardware constant. CNA_CBUF_CON0 (0x1040) carries a
 * granule ALLOWANCE in bits[16:27], and one task's feature plane must satisfy
 *
 *     ceil(iw*ic/64) granules per row  x  the task's input rows  <=  4096 + F
 *
 * with widths 16/32/64 breaking on the same granule total rather than the same row
 * count. The data side caps at 6144 granules (384 KiB) — F=3056 still measures 6144.
 *
 * ONLY SINGLE-BIT F VALUES DELIVER THEIR FACE VALUE. Each of F=256, 512, 1024 and
 * 2048 measures a ceiling of exactly 4096+F, but a combination delivers LESS than the
 * sum of its bits: F=1280 (256|1024) and F=1536 (512|1024) both measure 5152, not
 * 5376 and 5632. So this planner picks from the measured rungs below rather than
 * computing an arbitrary F, and rounds up to a rung. The cost is a little weight
 * headroom the plane did not need; the alternative is a budget the hardware does not
 * honour, which corrupts silently. [HW sweep, H96 MAX M9]
 *
 * F is not free. Data and weights share one pool of about 448 KiB (7168 granules),
 * so every granule handed to the data side is one the weight path loses:
 *
 *     F=0     data 256 KiB   weight slice 175 KiB computes, 200 KiB does not
 *     F=1024  data 320 KiB   weight slice 125 KiB computes, 150 KiB does not
 *     F=2048  data 384 KiB   weight slice below 75 KiB
 *
 * Every one of those brackets the pool minus the allowance at a whole 32 KiB bank
 * (6, 4, 2 banks), which is the model here. Driving F to 3072 leaves the weight path
 * nothing and WEDGES the CNA: the DPU then writes no output at all at any plane size,
 * which is indistinguishable from a wrong geometry encoder. So the planner never
 * emits an F that starves the weight path — it refuses the task instead, because the
 * caller's recourse (a shorter row window, or an ic split) is not the emitter's to
 * choose. [HW sweep, H96 MAX M9]
 *
 * The vendor's own two values are both rungs: its full-plane programs need 1600
 * granules and carry F=0, and its windowed depthwise task needs 56*91 = 5096 and
 * carries F=1024, which is the rung above its 1000-granule deficit. So a planned F
 * stays byte-identical to the captures.
 *
 * SCOPE OF THE WEIGHT GUARD. The trade above is measured at ONE output-channel group
 * (oc=32), where the weight-slice ceiling tracks pool-minus-allowance exactly. A
 * SEPARATE limit exists on the weight path that this guard does not model and that is
 * independent of F: at ic=192 k=5 (a 150 KiB slice) only the first 2 output-channel
 * groups compute, at F=0 as much as at F=2048 — the surface is exact through oc=63
 * and wrong past it — while ic=448 k=3 oc=128 (a 504 KiB cube, 126 KiB slice) is
 * exact on all four groups. So it is neither cube size, nor slice size, nor group
 * count alone, it predates this planner, and a caller working that region needs its
 * own validation. See rockchip-npu-notes/chips/rk3576-regcmd.md.
 * ==========================================================================*/
#define R76_GRANULE_BYTES        64u
/* THE ROW ALLOWANCE IS NOT ONE BUDGET — IT IS A LADDER, AND A RUNG THE PART DOES NOT
 * HONOUR DELIVERS THE BOTTOM OF IT.
 *
 * That makes the failure NON-MONOTONE in the window height, which is the trap: the
 * surface is exact below the rung's reach, wrong across the band of windows that SELECT
 * an unhonoured rung, and exact again above. Walked at 160x160 ic = oc = 32 k3
 * depthwise, one task per height, the shipped planner choosing:
 *
 *   45..51 rows  F=0     exact        (4096 granules covers a 51-row window at 80/row)
 *   52..54       F=256   WRONG        first row past 4096/entries in every window
 *   55..57       F=512   WRONG
 *   58..64       F=1024  exact
 *   65..76       F=2048  exact
 *
 * The same walk on the DIRECT path at the same plane and the same entries per row is
 * exact at every height. So a probe that BISECTS the window — which is what a capacity
 * bound would justify — reports whichever edge of the band it walks into, and a plan
 * whose windows straddle it fails on some tasks and not others.
 * [HW sweep, H96 MAX M9, `rk3576_conv_lib_gate rowmap`] */

#define R76_CBUF_BASE_GRANULES   4096u        /* what F=0 buys                      */
#define R76_CBUF_MAX_GRANULES    6144u        /* the data-side cap                  */
#define R76_CBUF_POOL_GRANULES   7168u        /* data + weight together, ~448 KiB   */
#define R76_CBUF_F_SHIFT         16           /* F occupies bits[16:27]             */
#define R76_CBUF_F_MASK          0xFFFu
#define R76_CBUF_CON0_BASE       0x10000000u  /* bit 28; a zero word corrupts       */

/* The F values measured to deliver exactly 4096+F granules, ascending. Not a formula,
 * because combinations of these bits do not deliver their sum (see above).
 *
 * AND TWO OF THEM DELIVER ONLY WHERE THE RESIDENT WEIGHT SLICE IS SMALL. 256 and 512
 * deliver exactly 4096 granules — the F=0 budget — the moment the slice passes 1 KiB, so
 * a task the planner put on one of them overruns its allowance and writes a full,
 * correctly sized surface with a WRONG TAIL: every output row past 4096/entries.
 *
 * THE GOVERNING QUANTITY IS THE SLICE, not the kernel, and a square-kernel sweep cannot
 * say which. The first characterisation of this held ic at 32 and moved the kernel — k=1
 * exact at five plane widths, k=3 and k=5 wrong at every one — which reads as "the rung
 * needs kh == 1" and is what shipped. Crossing the axes says otherwise: at ONE granule
 * total (4352, the F=256 rung exactly) and ONE kernel (k1x1), the same (row size, row
 * count) is bit-exact at ic 32 and wrong at ic 64 and 128 — 68 rows of 64 granules, wrong
 * from output row 64 in both. So `ic` at a fixed kernel and the kernel at a fixed `ic`
 * move the same thing, `32*ic*kh*kw`, and it is the same quantity r76_weight_slice_cap()
 * is already stated over. Live at 16 granules (ic 32, k=1); dead at 32 (ic 64), 64
 * (ic 128), 144 (ic 32, k=3) and 400 (ic 32, k=5).
 *
 * The delivered budget backs out to the row from the exact-element counts: the last
 * correct output row is always the one fed by input row 4096/entries. 1024 and 2048
 * deliver at every slice tried — the vendor's own windowed depthwise capture is a k=3
 * program at F=1024, and an ic=128 plane is bit-exact at F=1024 where it is wrong at 256
 * and 512 — so this is the two low rungs and not the field.
 *
 * THE QUANTITY IS ONE GROUP'S SLICE, and this charges the WHOLE CUBE anyway. Holding the
 * slice at the measured-live 16 granules and raising the group count separates the two: at
 * oc 64 and 96 — cubes of 32 and 48 granules — both low rungs still deliver. Charging the
 * cube is therefore conservative rather than wrong, and it stays because relaxing it buys
 * NOTHING: the rung it declines is replaced by a strictly larger one that also delivers, at
 * the same CBUF and with no extra submit. Being UNDER costs the next rung up or a shorter
 * row window (one more task); being OVER computes a silently wrong surface.
 *
 * The threshold between 16 and 32 granules stays bracketed, and this harness cannot narrow
 * it: ic is padded to a multiple of 32, so 32*ic*kh*kw moves in 1024-byte steps at every
 * shape a square kernel and a whole ic can build. A non-square kernel is what lands between.
 * Nothing in the corpus is affected either way: a rung is reachable only where the plane is
 * 4097-4608 granules AND the weights are under 1 KiB, and the matmul's own row planner is
 * past that at every K it runs. ROCKET_RK3576_CBUF_RUNGS is a comma-separated override,
 * for re-measuring this rather than for shipping.
 * [HW sweep, H96 MAX M9, tests/rk3576_conv_sym.c rung] */
static unsigned r76_cbuf_f_rungs[] = { 0u, 256u, 512u, 1024u, 2048u };
static unsigned r76_cbuf_f_nrungs = 5u;

/* The resident weight footprint at which the two low rungs were measured to deliver. */
#define R76_RUNG_WEIGHT_MAX_GRANULES 16u

/* THE FOOTPRINT ABOVE IS A DIRECT-PATH QUANTITY AND IT DOES NOT CARRY TO DEPTHWISE.
 *
 * Forcing each rung under one fixed window that F=0 does not buy — the direct question,
 * asked on both paths — says the two paths do not share a threshold, and that no
 * threshold on the depthwise weight footprint fits at all:
 *
 *   path      kernel  oc     footprint          F=256 / F=512
 *   direct    1x1     32     1024 B = 16 gran   deliver
 *   direct    1x1     64     2048 B = 32 gran   fall back to 4096
 *   depthwise 1x1     32       64 B =  1 gran   deliver
 *   depthwise 1x1    256      512 B =  8 gran   deliver
 *   depthwise 1x1   1024     2048 B = 32 gran   deliver
 *   depthwise 2x2     32      256 B =  4 gran   fall back to 4096
 *   depthwise 3x3     32      576 B =  9 gran   fall back to 4096
 *   depthwise 5x5     32     1600 B = 25 gran   fall back to 4096
 *
 * On the depthwise path 4 granules is dead where 32 is live, so the shipped predicate is
 * refuted in BOTH directions and the surviving description is the TAP COUNT: every
 * single-tap cell delivers at three channel counts spanning 32x, and every multi-tap one
 * falls back. A 64-channel-group footprint (the int8 depthwise cube's own group) against
 * the direct path's 16-granule threshold fits the k1, k3 and k5 cells and is refuted by
 * the 2x2 one, which is why that cell is in the probe.
 *
 * SO THE LOW RUNGS ARE SIMPLY NOT USED ON THIS PATH, rather than gated on a fitted
 * quantity. It costs nothing measurable to decline them: the planner takes the smallest
 * live rung that covers the window, 1024 and 2048 deliver at every footprint tried on
 * both paths, and 1024 is strictly larger — so the fallback is the same CBUF, the same
 * window, the same submit, and the pool check it must pass has room for any depthwise
 * cube a real shape carries (it would need oc*kh*kw past 65536 to bite).
 *
 * What a rung the part does NOT honour costs, in contrast, is a full, correctly sized
 * surface whose rows past 4096/entries are wrong, with nothing to fault on.
 * [HW sweep, H96 MAX M9, `rk3576_conv_lib_gate rowmap`] */

/* Whether this rung may be programmed beside a resident weight cube of this many granules.
 * Zero is "not weighed" rather than "none", so the rung is refused there too. */
static int r76_rung_live(unsigned f, unsigned wgran, int dw)
{
    if (f != 256u && f != 512u) return 1;
    if (dw) return 0;
    return wgran && wgran <= R76_RUNG_WEIGHT_MAX_GRANULES;
}

static void r76_cbuf_rungs_from_env(void)
{
    const char *e = getenv("ROCKET_RK3576_CBUF_RUNGS");
    unsigned n = 0;
    if (!e || !*e) return;
    while (*e && n < sizeof r76_cbuf_f_rungs / sizeof r76_cbuf_f_rungs[0]) {
        r76_cbuf_f_rungs[n++] = (unsigned)strtoul(e, (char **)&e, 0);
        while (*e == ',' || *e == ' ') e++;
    }
    if (n) r76_cbuf_f_nrungs = n;
}

/* ---- The CBUF base, and the RE knob that moves it ----
 *
 * The low half of CNA_CBUF_ENTRIES (0x103c) and of CNA_CBUF_CON0 (0x1040) is a
 * GRANULE OFFSET into the CBUF: where this task's window starts and where its fetch
 * resumes. The row-reuse path drives it across a plane's tasks; every plain entry
 * point leaves it at zero, so every job this library submits stages from granule 0.
 *
 * That matters beyond reuse, because the two NPU cores share one CBUF and two jobs
 * executing at once compute wrong answers. If the base is a real address into a
 * shared pool, then biasing one core's jobs away from the other's is a PARTITION —
 * and it is one userspace can express, since it is a field the encoder already
 * emits. If instead the base is per-core-relative, or ignored outside a
 * continuation, no bias separates anything.
 *
 * ROCKET_RK3576_CBUF_BIAS=<granules> adds a constant to both, which changes where a
 * task stages and nothing about what it computes: the window base and the fetch base
 * move together, so a task remains self-consistent at any bias the hardware honours.
 * A bias the hardware does NOT honour is therefore invisible on a solo job and shows
 * up only against a concurrent partner or past the end of the pool. Read
 * tests/rk3576_cbuf_base.c before drawing a conclusion from either.
 *
 * rocket_rk3576_set_cbuf_bias() overrides the variable PER THREAD, which is what a
 * concurrency probe needs: two workers in one process must be able to stage at
 * different bases, and an environment variable is process-wide. */
static __thread unsigned r76_cbuf_bias_tls = ~0u;   /* ~0u: unset, take the env */

void rocket_rk3576_set_cbuf_bias(unsigned granules)
{
    r76_cbuf_bias_tls = granules;
}

static unsigned r76_cbuf_bias(void)
{
    const char *s;
    if (r76_cbuf_bias_tls != ~0u) return r76_cbuf_bias_tls;
    s = getenv("ROCKET_RK3576_CBUF_BIAS");
    return s && *s ? (unsigned)strtoul(s, NULL, 0) : 0u;
}

/* The weight footprint that must be resident at once. The weight path stages per
 * output-channel group of 32 rather than per cube — a 441 KiB cube at oc=224
 * computes bit-exactly while one group's 200 KiB slice does not — so the cube size
 * is not the constraint and ic*kh*kw is. Depthwise carries one kernel per channel,
 * and its whole coefficient set is what the register calls WEIGHT_BYTES.
 * [HW sweep, H96 MAX M9] */
static inline unsigned r76_weight_resident_bytes(unsigned ic, unsigned oc,
                                                 unsigned kh, unsigned kw, int dw)
{
    return dw ? (oc * kh * kw * 2u) : (32u * ic * kh * kw);
}

static inline unsigned r76_granules(unsigned bytes)
{
    return (bytes + R76_GRANULE_BYTES - 1u) / R76_GRANULE_BYTES;
}

/* The largest resident weight slice at which EVERY output-channel group still
 * computes, as a function of how many groups the conv drives.
 *
 * The pool check below is not sufficient on its own: it models the slice against the
 * space the data allowance leaves (about 192 KiB at F=0), and the part loses groups
 * well before that. What it loses is graded — one group at a time as the slice grows
 * — and the surface comes back with the leading groups bit-exact and the trailing
 * ones wrong, which reads as an output-channel defect rather than a capacity one.
 *
 * Measured at F=0 on a 4x2 plane, sweeping ic at k=1 so the slice moves in 1 KiB
 * steps, and cross-checked at k=3 and k=5: the same slice behaves identically
 * whichever (ic, kh, kw) produces it, so the slice is the governing quantity.
 *
 *   slice  144 KiB -> 4 of 4 groups exact      152 KiB -> 2 of 4
 *          145 KiB -> 3 of 4                   156 KiB -> 2 of 4
 *          148 KiB -> 3 of 4                   162 KiB -> 1 of 4
 *
 * The table takes the last measured-good slice for each group count. A conv past it
 * is refused rather than run, because the recourse — splitting ic — is the caller's
 * to choose, and because a silent wrong answer here is indistinguishable from a
 * geometry bug.
 *
 * A SINGLE group is left to the pool check alone. There is nothing extra to model
 * there — the pool arithmetic already lands on the measured single-group boundary
 * (a 175 KiB slice computes at F=0 and 200 KiB does not, and 192 KiB is where
 * base + slice reaches the pool) — and the graded loss above is by definition a
 * multi-group effect. [HW sweep, H96 MAX M9] */
/* The two output-channel-axis bounds a matmul reaches and a convolution does not.
 * See r76_plan_cbuf for what each was measured against. */
#define R76_MAX_OC            2944u
#define R76_MAX_WEIGHT_CUBE   (6u * 1024u * 1024u)

static inline unsigned r76_weight_slice_cap(unsigned oc)
{
    unsigned groups = (oc + 31u) / 32u;
    if (groups >= 4u) return 144u * 1024u;
    if (groups == 3u) return 148u * 1024u;
    if (groups == 2u) return 156u * 1024u;
    return ~0u;                      /* one group: the pool check governs */
}

/* The allowance planner proper, driven by the two quantities that actually govern it:
 * the granules ONE ROW of the task's feature plane costs, and the weight footprint
 * that has to be resident beside it. Both differ on the ARGB path — its row is the
 * packed image expanded to 4 lanes, and its weight cube is folded — so the public
 * entry point below computes them the normal-path way and the ARGB emitter computes
 * its own. `wbytes` of 0 skips the weight-side guards (nothing resident to weigh). */
static void r76_cbuf_rungs_init(void)
{
    static int done;
    if (done) return;
    done = 1;
    r76_cbuf_rungs_from_env();
}

static int r76_plan_cbuf(unsigned entries, unsigned ih, unsigned oc, unsigned wbytes,
                         int dw, unsigned *f_out)
{
    unsigned need = entries * ih;
    unsigned wgran = r76_granules(wbytes);
    /* The RESIDENT footprint, which is what the rung liveness is stated over: a direct
     * conv holds one slice per output-channel group, a depthwise one holds its group. */
    unsigned wres = dw ? wgran : ((oc + 31u) / 32u) * wgran;
    unsigned f;
    size_t r;

    if (!entries || !ih) return -1;
    r76_cbuf_rungs_init();

    /* The lowest rung whose budget covers the plane. */
    f = r76_cbuf_f_rungs[r76_cbuf_f_nrungs - 1u];
    for (r = 0; r < r76_cbuf_f_nrungs; r++) {
        if (!r76_rung_live(r76_cbuf_f_rungs[r], wres, dw)) continue;
        if (R76_CBUF_BASE_GRANULES + r76_cbuf_f_rungs[r] >= need) {
            f = r76_cbuf_f_rungs[r];
            break;
        }
    }

    if (R76_CBUF_BASE_GRANULES + f > R76_CBUF_MAX_GRANULES ||
        need > R76_CBUF_BASE_GRANULES + f) {
        ROCKET_LOGE("rk3576 cbuf: the task's feature plane is %u granules and the data "
                    "side caps at %u — window the rows to at most %u\n",
                    need, R76_CBUF_MAX_GRANULES,
                    R76_CBUF_MAX_GRANULES / entries);
        return -1;
    }
    if (!dw && wbytes > r76_weight_slice_cap(oc)) {
        ROCKET_LOGE("rk3576 cbuf: a resident weight slice of %u KiB drives only the "
                    "leading output-channel groups at oc=%u (cap %u KiB) — split ic\n",
                    wbytes / 1024u, oc, r76_weight_slice_cap(oc) / 1024u);
        return -1;
    }
    /* Two bounds the slice cap above does not cover, because no conv shape reaches
     * them: they bite only at the very wide output channel counts a MATMUL drives,
     * where N is the free axis and buying MACs per submit means growing it. Past
     * either one the trailing output channels simply do not reach DDR — a full,
     * correctly sized surface with a clean prefix and no fault — so they are
     * refusals, not warnings.
     *
     * Both are measured on a k=1 conv, which is the matmul's own kernel, at the
     * feature-budget cap so the data side is out of the way. oc: 2944 computes at
     * every ic tried and 3072 is intermittent. Total cube: 6 MiB computes (ic 4096 x
     * oc 1536) and 6.75 MiB does not (ic 4608 x oc 1536), with ic 4096 x oc 2048
     * losing about a quarter of its channels. Stating the second as the whole cube
     * rather than per-kernel-tap makes it refuse EARLIER at k > 1 than anything
     * measured there, which is the safe direction. [HW sweep, H96 MAX M9] */
    if (!dw && oc > R76_MAX_OC) {
        ROCKET_LOGE("rk3576 cbuf: oc=%u is past the %u output channels one task "
                    "delivers — split the output-channel axis\n", oc, R76_MAX_OC);
        return -1;
    }
    if (!dw && wbytes) {
        uint64_t total = (uint64_t)((oc + 31u) / 32u) * wbytes;
        if (total > R76_MAX_WEIGHT_CUBE) {
            ROCKET_LOGE("rk3576 cbuf: a weight cube of %llu KiB at oc=%u drives only "
                        "the leading output-channel groups (cap %u KiB) — split the "
                        "output-channel axis\n",
                        (unsigned long long)(total / 1024u), oc,
                        R76_MAX_WEIGHT_CUBE / 1024u);
            return -1;
        }
    }
    if (R76_CBUF_BASE_GRANULES + f + wgran > R76_CBUF_POOL_GRANULES) {
        ROCKET_LOGE("rk3576 cbuf: %u granules of feature plane needs F=%u, which leaves "
                    "the weight path %u granules against a resident slice of %u "
                    "(%u KiB) — shorten the row window or split ic\n",
                    need, f, R76_CBUF_POOL_GRANULES - R76_CBUF_BASE_GRANULES - f,
                    wgran, wbytes / 1024u);
        return -1;
    }

    if (f_out) *f_out = f;
    return 0;
}

/* The feature plane's granule cost is per BYTE, so a 2-byte element doubles it. The
 * emitted registers have always scaled (r76_data_entries_p); these planners did not,
 * and an fp16 conv large enough to need a rung would therefore have programmed an
 * allowance sized for half its plane — which computes WRONG with a full surface
 * written and nothing to fault on.
 *
 * The WEIGHT side keeps the int8 form at every precision, deliberately. The float
 * cube's group is 8 output channels of 2-byte elements against the int8 cube's 32 of
 * 1 byte, so 32*ic*kh*kw is an UPPER BOUND on the float slice rather than its size,
 * and the graded group-loss table it feeds was measured on int8 only. Over-estimating
 * refuses early instead of corrupting, and it costs nothing on the shapes the fp16
 * envelope allows (ic <= 8 puts the slice under 7 KiB at k=5). */
int rocket_rk3576_cbuf_f_prec(unsigned iw, unsigned ic, unsigned ih, unsigned oc,
                              unsigned kh, unsigned kw, int dw, unsigned prec,
                              unsigned *f_out)
{
    if (!iw || !ic || !ih || !kh || !kw) return -1;
    if (ic <= R76_ARGB_LANES)            /* the packed-image first conv */
        return r76_plan_cbuf(r76_argb_entries(iw, r76_elem_bytes(prec),
                                              r76_is_float(prec)), ih, oc,
                             oc * r76_argb_weight_elems(kh, kw, r76_is_float(prec))
                                * r76_elem_bytes(prec), 0, f_out);
    return r76_plan_cbuf(r76_data_entries_p(iw, ic, r76_elem_bytes(prec)), ih, oc,
                         r76_weight_resident_bytes(ic, oc, kh, kw, dw), dw, f_out);
}

int rocket_rk3576_cbuf_f(unsigned iw, unsigned ic, unsigned ih, unsigned oc,
                         unsigned kh, unsigned kw, int dw, unsigned *f_out)
{
    return rocket_rk3576_cbuf_f_prec(iw, ic, ih, oc, kh, kw, dw,
                                     precision_int8, f_out);
}

unsigned rocket_rk3576_max_task_rows_prec(unsigned iw, unsigned ic, unsigned oc,
                                          unsigned kh, unsigned kw, int dw,
                                          unsigned prec)
{
    unsigned entries, wbytes, wgran, wres, f = 0;
    size_t r;

    if (!iw || !ic || !kh || !kw) return 0;
    r76_cbuf_rungs_init();

    if (ic <= R76_ARGB_LANES) {
        entries = r76_argb_entries(iw, r76_elem_bytes(prec), r76_is_float(prec));
        wbytes  = oc * r76_argb_weight_elems(kh, kw, r76_is_float(prec))
                     * r76_elem_bytes(prec);
        dw      = 0;
    } else {
        entries = r76_data_entries_p(iw, ic, r76_elem_bytes(prec));
        wbytes  = r76_weight_resident_bytes(ic, oc, kh, kw, dw);
    }
    wgran = r76_granules(wbytes);
    /* The RESIDENT footprint the rung liveness is stated over — every group's slice for a
     * direct conv. Taken before the charge below, which grows `wgran` to the same thing
     * only when it still leaves a rung and so cannot stand in for it. */
    wres = dw ? wgran : ((oc + 31u) / 32u) * wgran;
    if (!entries || R76_CBUF_BASE_GRANULES + wgran > R76_CBUF_POOL_GRANULES)
        return 0;                     /* the weight slice does not fit even at F=0 */
    if (!dw && wbytes > r76_weight_slice_cap(oc))
        return 0;                     /* it fits, but only the leading groups compute */
    if (!dw && (oc > R76_MAX_OC ||
                (uint64_t)((oc + 31u) / 32u) * wbytes > R76_MAX_WEIGHT_CUBE))
        return 0;                     /* the output-channel-axis bounds; see r76_plan_cbuf */

    /* THE WEIGHT SIDE IS THE WHOLE CUBE, NOT ONE GROUP'S SLICE. `wgran` is
     * `32*ic*kh*kw` and does not depend on `oc` at all, so charging it alone leaves the
     * data side an allowance the part does not honour once a conv drives more than one
     * output-channel group: a 224x224 k7 s2 at oc 64 computes to 48 input rows per task
     * and this said 54, and the task past the real allowance writes NOTHING — spending
     * the surface guard's eight power cycles before the entry gives up.
     *
     * Charging every group brings that to 45, which is under the measurement rather
     * than over it. It is a BOUND rather than the law: the exact budget is not pinned
     * (48 rows is 5376 granules where this model allows 5120 and the one-slice model
     * 6144), and it is applied only where it still leaves a rung, because a cube past
     * the pool at F=0 would otherwise refuse shapes that compute today. Both directions
     * are safe: this can only ever SHRINK a window, and a shorter window is a submit,
     * not a wrong answer. [HW sweep, `rk3576_conv_lib_gate rowbound`] */
    if (!dw) {
        unsigned cube = ((oc + 31u) / 32u) * wgran;
        if (R76_CBUF_BASE_GRANULES + cube <= R76_CBUF_POOL_GRANULES) wgran = cube;
    }

    /* The highest rung the weight cube still leaves room for, inside the data cap. */
    for (r = 0; r < r76_cbuf_f_nrungs; r++) {
        unsigned cand = r76_cbuf_f_rungs[r];
        if (R76_CBUF_BASE_GRANULES + cand > R76_CBUF_MAX_GRANULES) break;
        if (R76_CBUF_BASE_GRANULES + cand + wgran > R76_CBUF_POOL_GRANULES) break;
        if (!r76_rung_live(cand, wres, dw)) continue;
        f = cand;
    }

    return (R76_CBUF_BASE_GRANULES + f) / entries;
}

unsigned rocket_rk3576_max_task_rows(unsigned iw, unsigned ic, unsigned oc,
                                     unsigned kh, unsigned kw, int dw)
{
    return rocket_rk3576_max_task_rows_prec(iw, ic, oc, kh, kw, dw, precision_int8);
}

/* ============================================================================
 * SECTION — the row window
 *
 * A plane past the CBUF allowance is not a slow conv, it is a wrong one — the DPU
 * still writes a full surface and nothing faults — so the recourse the allowance
 * planner names has to exist. It is the same split the vendor's toolkit takes: cut
 * the OUTPUT rows into runs, give each run the input row window it reads, and keep
 * describing the full plane alongside the window.
 *
 * Two quantities per task are the caller's whole job, and both are byte offsets:
 * the feature cube's row offset and the output cube's. They are plain row strides
 * because the cubes are NC1HWC2 with a 16-byte channel atom and the CNA takes the
 * DDR group stride from the FULL plane (0x1094 = iw*ih_full, in 16-byte units), so
 * base + iy0*iw*16 addresses row iy0 of EVERY channel group, not just the first.
 * The vendor's sliced capture is the direct evidence for the feature side: its
 * fourth task reads input rows 111.. and 0x1088 carries exactly that row offset.
 * ==========================================================================*/
#define R76_C2_BYTES 16u          /* the int8 cube's channel atom, in bytes */

/* Input rows the output-row run [oy0, oy0+oh) reads, clipped to the plane. Output
 * row y reads padded rows [y*stride - pad_top, y*stride - pad_top + kh). */
static void r76_row_window(unsigned oy0, unsigned oh, unsigned stride, unsigned kh,
                           unsigned pad_top, unsigned ih_full,
                           unsigned *iy0, unsigned *ih, unsigned *ptop)
{
    long first = (long)oy0 * stride - (long)pad_top;
    long last  = ((long)oy0 + oh - 1) * stride - (long)pad_top + kh - 1;

    *ptop = first < 0 ? (unsigned)(-first) : 0u;
    if (first < 0) first = 0;
    if (last  < 0) last  = 0;
    if (last  > (long)ih_full - 1) last = (long)ih_full - 1;

    *iy0 = (unsigned)first;
    *ih  = last >= first ? (unsigned)(last - first + 1) : 1u;
}

/* Lay the windows out, taking at most `want` output rows per task (0 = as many as
 * the allowance permits). Returns the task count, or 0 if a single output row
 * already overflows. `out` may be NULL to count without recording. */
static unsigned r76_lay_rows(const conv_params_t *p, unsigned ih_full, unsigned oh_full,
                             unsigned cap, unsigned want, unsigned prec,
                             rocket_rk3576_row_task *out, unsigned max_tasks)
{
    unsigned n = 0, oy0;
    unsigned prev_end = 0, resident = 0, retained = 0, entries;
    unsigned elem = r76_elem_bytes(prec);

    entries = p->ic <= R76_ARGB_LANES
                  ? r76_argb_entries(p->iw, elem, r76_is_float(prec))
                  : r76_data_entries_p(p->iw, p->ic, elem);

    for (oy0 = 0; oy0 < oh_full; ) {
        unsigned iy0, ih, ptop, oh_task;

        /* The most output rows this run can carry, from the padded extent the
         * allowance affords: the leading pad rows it starts on plus a full window of
         * real ones. Deliberately NOT capped by the rows left in the plane — the
         * output rows at the bottom are served by the bottom pad, so a run there
         * reaches further than its real rows suggest. The clip to the plane and the
         * shrink loop below are what make the estimate exact. */
        r76_row_window(oy0, 1, p->stride_y, p->kh, p->pad_top, ih_full, &iy0, &ih, &ptop);
        {
            unsigned extent = ptop + cap;
            oh_task = extent >= p->kh ? (extent - p->kh) / p->stride_y + 1u : 0u;
        }
        if (want && oh_task > want) oh_task = want;
        if (oh_task > oh_full - oy0) oh_task = oh_full - oy0;   /* bottom pad tail */
        if (!oh_task) oh_task = 1;                              /* pad-only window */

        for (;;) {
            r76_row_window(oy0, oh_task, p->stride_y, p->kh, p->pad_top, ih_full,
                           &iy0, &ih, &ptop);
            if (ih <= cap || oh_task == 1) break;
            oh_task--;
        }
        if (ih > cap) return 0;
        /* Rows of this window the previous task already fetched. The vendor resets
         * the CBUF base rather than continuing when there is nothing to keep. */
        retained = (n && iy0 < prev_end) ? prev_end - iy0 : 0u;
        if (!retained) resident = 0;
        if (out) {
            if (n == max_tasks) return 0;
            out[n].iy0         = (uint16_t)iy0;
            out[n].ih          = (uint16_t)ih;
            out[n].oy0         = (uint16_t)oy0;
            out[n].oh          = (uint16_t)oh_task;
            out[n].pad_top     = (uint8_t)ptop;
            /* The feature row stride is the cube's channel atom on the normal path
             * and the PACKED image row on the ARGB one — iw*ic bytes, since that
             * datapath reads the image as it comes out of a camera or a decoder,
             * not as an NC1HWC2 cube. The output is a cube either way. */
            out[n].feature_off = (uint32_t)iy0 * p->iw *
                                 (p->ic <= R76_ARGB_LANES ? p->ic * elem : R76_C2_BYTES);
            out[n].output_off  = (uint32_t)oy0 * p->ow * R76_C2_BYTES;
            /* What a reuse-aware caller needs: how many of this window's leading rows
             * the previous task already fetched, and how full the CBUF is when this
             * task starts. Both are 0 on the first task and both are ignored by the
             * plain entry points. */
            out[n].retained      = (uint16_t)retained;
            out[n].cbuf_resident = resident;
        }
        resident += entries * (ih - retained);
        prev_end = iy0 + ih;
        n++;
        oy0 += oh_task;
    }
    return n;
}

int rocket_rk3576_plan_rows_prec(const conv_params_t *p, int dw, unsigned prec,
                                 rocket_rk3576_row_task *out, unsigned max_tasks,
                                 unsigned *count)
{
    unsigned ih_full, oh_full, cap, entries, n, t;
    const char *forced;

    if (!p || !out || !max_tasks || !count) return -1;
    if (!p->iw || !p->ic || !p->kh || !p->kw || !p->oh || !p->ow || !p->stride_y) return -1;

    ih_full = p->ih_full ? p->ih_full : p->ih;
    oh_full = p->oh_full ? p->oh_full : p->oh;
    /* The row window is charged at the DDR row ADVANCE, so a pitched feature buffer
     * plans shorter windows — the one cost of a pitch, and the same one the widened
     * lowering pays. */
    entries = p->ic <= R76_ARGB_LANES
                  ? r76_argb_entries(p->iw, r76_elem_bytes(prec), r76_is_float(prec))
                  : r76_data_entries_p(p->in_pitch_w ? p->in_pitch_w : p->iw,
                                       p->ic, r76_elem_bytes(prec));
    if (!entries) return -1;

    cap = rocket_rk3576_max_task_rows_prec(p->in_pitch_w ? p->in_pitch_w : p->iw,
                                           p->ic, p->oc, p->kh, p->kw, dw, prec);
    /* A PITCHED TASK IS CHARGED ITS DMA ROW BUDGET, NOT ITS ROWS. The budget is
     * `ceil(rows * entries_pitch / entries_plane)` rows and the CBUF is charged that
     * many at the PITCH's own entry count, so the window a pitched task may take is the
     * unpitched allowance scaled DOWN by that ratio, and by measurement it is scaled
     * down by the ratio TWICE. A 112-wide plane inside 128-wide rows has an allowance of
     * 48 rows: 40, 41, 42 and 47 all compute wrong and 16, 32 and 36 are bit-exact, so
     * the once-scaled 42 is over and the twice-scaled 36 is under. **It is a BOUND, not
     * the law** — the residue between 36 and 40 is unpinned and the extra margin has the
     * size of the retained rows a chained window carries, which is a charge this
     * arithmetic does not model. Safe in one direction only, like the unpitched
     * allowance it scales: too small is a submit, too large is a silently wrong surface.
     * [HW sweep, H96 MAX M9, tests/rk3576_conv_pitch.c] */
    if (p->in_pitch_w && p->in_pitch_w != p->iw) {
        unsigned ep = r76_data_entries_p(p->iw, p->ic, r76_elem_bytes(prec));
        if (ep && entries)
            cap = (unsigned)(((uint64_t)cap * ep * ep) / ((uint64_t)entries * entries));
        if (!cap) cap = 1u;
    }
    /* 0x1028 packs entries*rows in a 16-bit half, which can bite before the CBUF
     * does on a narrow, very deep plane. */
    if (cap > 0xFFFFu / entries) cap = 0xFFFFu / entries;
    forced = getenv("ROCKET_RK3576_MAX_ROWS");
    if (forced && *forced) {
        unsigned f = (unsigned)strtoul(forced, NULL, 0);
        if (f && f < cap) cap = f;
        ROCKET_LOGI("rk3576 rows: per-task row cap forced to %u\n", cap);
    }
    /* THE PROBE KNOB, and it is the only one here that can RAISE the cap. The shipped
     * allowance is a BOUND rather than the part's law — it charges the whole weight
     * cube, which is under the measurement — so pinning the law means asking the part
     * about windows the planner would never choose. Every window is still validated by
     * the emitter's own per-task allowance check below, so this reaches only into the
     * region that check permits, and a window past THAT is refused rather than emitted.
     * Never set in shipping code. [tests/rk3576_conv_lib_gate.c rowlaw] */
    forced = getenv("ROCKET_RK3576_ROW_CAP_PROBE");
    if (forced && *forced) {
        unsigned f = (unsigned)strtoul(forced, NULL, 0);
        if (f) cap = f;
        ROCKET_LOGI("rk3576 rows: per-task row cap set to %u by the RE probe knob\n", cap);
    }
    if (!cap) {
        ROCKET_LOGE("rk3576 rows: no row window fits — the weight slice alone "
                    "exceeds the CBUF pool at ic=%u k=%ux%u; split ic\n",
                    p->ic, p->kh, p->kw);
        return -1;
    }

    /* Count greedily, then spread the output rows EVENLY over that many tasks.
     * Greedy alone leaves a ragged tail — a 112-row plane at a 109-row cap comes out
     * 109 + 2 + 1 rather than two windows of 56 — and every extra task is a whole
     * submit, which on this part is a whole NPU power cycle. Taking the greedy count
     * as the target keeps the even pass from ever needing more tasks than greedy. */
    n = r76_lay_rows(p, ih_full, oh_full, cap, 0, prec, NULL, 0);
    if (!n) {
        ROCKET_LOGE("rk3576 rows: one output row already needs more than %u input "
                    "rows of allowance — split ic instead\n", cap);
        return -1;
    }
    if (n > max_tasks) {
        ROCKET_LOGE("rk3576 rows: the plane needs %u tasks, more than the %u offered\n",
                    n, max_tasks);
        return -1;
    }
    {
        unsigned even = (oh_full + n - 1u) / n;
        unsigned m = r76_lay_rows(p, ih_full, oh_full, cap, even, prec, out, max_tasks);
        if (!m || m > n)
            m = r76_lay_rows(p, ih_full, oh_full, cap, 0, prec, out, max_tasks);
        if (!m) return -1;
        n = m;
    }

    /* Every window has to satisfy the allowance planner too — same check the
     * emitter makes per task, made here so a plan is refused before any submit. */
    for (t = 0; t < n; t++)
        if (rocket_rk3576_cbuf_f_prec(p->iw, p->ic, out[t].ih, p->oc,
                                      p->kh, p->kw, dw, prec, NULL) < 0)
            return -1;

    *count = n;
    return 0;
}

/* The int8 plan, which is what every caller before the float paths wanted. */
int rocket_rk3576_plan_rows(const conv_params_t *p, int dw,
                            rocket_rk3576_row_task *out, unsigned max_tasks,
                            unsigned *count)
{
    return rocket_rk3576_plan_rows_prec(p, dw, precision_int8, out, max_tasks, count);
}

/* ============================================================================
 * SECTION — the emitter
 * ==========================================================================*/

/*
 * Emit one RK3576 conv task. Register order follows the vendor stream exactly
 * (interleaved S_POINTER preamble, then CNA, CORE, DPU, DPU_RDMA), so a dump of
 * this program lines up entry-for-entry with a vendor dump. Nothing fires until
 * the PC trailer, so the order is a diffability choice, not a correctness one —
 * except for the S_POINTER writes, which select the register group and must come
 * before their block's registers.
 */
/* CBUF row reuse, as the vendor's continuation tasks program it. All zero — which is
 * what the plain entry points pass — is "refetch the whole window at base 0", the
 * form validated on hardware. See gen_conv2d_int8_rk3576_reuse() in the header. */
struct r76_reuse {
    unsigned continuation;   /* this task is not the first of its plane        */
    unsigned retained_rows;  /* leading window rows already in the CBUF        */
    unsigned resident;       /* CBUF granules the sequence has filled so far   */
};

static int gen_conv2d_task_rk3576(uint64_t *ops, const npu_cna_desc *cna,
                                  const npu_core_desc *core, const npu_dpu_desc *dpu,
                                  int dw, int argb, unsigned ih_full, unsigned oh_full,
                                  unsigned pad_right, unsigned pad_bottom,
                                  unsigned cbuf_f, const struct r76_reuse *reuse,
                                  const lut_rk3576_t *lut, unsigned in_surf_elems,
                                  unsigned in_pitch_w)
{
    int i = 0;
    unsigned iw  = cna->datain_width,  ih  = cna->datain_height;
    unsigned ic  = cna->datain_channel;   /* on the ARGB path: the IMAGE's channels */
    unsigned ow  = cna->dataout_width, oh  = cna->dataout_height;
    unsigned oc  = (unsigned)core->dataout_channel + 1u;
    unsigned kh  = cna->weight_height, kw = cna->weight_width;
    unsigned elem = r76_elem_bytes(cna->in_precision);
    /* The float datapath is a mode, not a width: three words below take a different
     * value for it and each one alone decides whether the contraction reads every
     * feature surface once or twice. */
    int is_float = r76_is_float(cna->in_precision);
    /* The ARGB path programs a channel count that is not the image's (the kernel
     * width folded into 4 lanes per pixel column), a CBUF row that is the image row
     * expanded to those lanes, and a weight kernel padded to a 16-byte row per kernel
     * row. Everything downstream of the CNA is the direct path's. */
    unsigned ic_prog = argb ? r76_argb_ic_prog(kw, ic, is_float) : ic;
    /* The depthwise CBUF entry count is taken against the FEATURE granule, which
     * rounds the channel count past 16 in the one residue the hardware will not fetch
     * a partial block of. The weight cube rounds to 16 instead. */
    unsigned entries = argb ? r76_argb_entries(iw, elem, is_float)
                            : (dw ? r76_data_entries_p(iw, r76_dw_cf(ic), elem)
                                  : r76_data_entries_p(iw, ic, elem));
    /* THE FETCH LENGTH AND THE DDR ROW STRIDE ARE TWO REGISTERS, not one quantity in
     * two units. `entries` above is the row's granule count and sizes the CBUF at
     * 0x103C and 0x1028; the granule count at 0x1044's LOW half is what the feature
     * DMA ADVANCES BY between rows. They are equal in every capture because every
     * capture has a tight plane, which is why the pair reads as one number.
     * [HW sweep, H96 MAX M9, tests/rk3576_conv_pitch.c map mode: a 16-wide plane in
     * 20-wide rows reads at stride 16 with everything derived, at stride 20 with this
     * one half moved, and 0x1090 — the register a source read named as the line
     * stride — advances by its own value once every FOUR rows and cannot express a
     * pitch at all.] */
    unsigned entries_ddr = (!argb && !dw && in_pitch_w)
                             ? r76_data_entries_p(in_pitch_w, ic, elem) : entries;
    /* Weight ELEMENTS per output channel, and the two registers built from them.
     * 0x1020 carries the count scaled by the element size — the BYTES one output
     * channel occupies — and 0x1030's high half carries twice the raw count. The two
     * coincide at fp16, which is why one register can look like the other; at int8
     * they differ by the factor of two and at fp16 by nothing. Depthwise is its own
     * pair: a depthwise weight occupies a 16-bit slot whatever the precision, so its
     * element count is already the float one and must NOT be scaled again. */
    unsigned welem_count = argb ? r76_argb_weight_elems(kh, kw, is_float)
                                : (dw ? r76_dw_cw(oc) : ic) * kh * kw;
    unsigned welems = dw ? welem_count : welem_count * elem;
    unsigned wbpk = argb ? welem_count * 2u : r76_wbpk(ic, oc, kh, kw, dw);
    unsigned line_stride = argb ? r76_argb_line_stride(iw, ic, elem) : cna->line_stride;
    /* The rows this task actually FETCHES, and where in the CBUF they land. Without
     * reuse these reduce to the whole window at base 0. */
    unsigned fetch_rows = ih - (reuse->retained_rows < ih ? reuse->retained_rows : 0u);
    /* THE FEATURE DMA'S BUDGET IS A ROW COUNT, AND IT IS SPENT AT THE ROW ADVANCE.
     * The DMA stops after `fetch_rows * entries` granules of ADDRESS, not of data, so a
     * pitched walk — which advances `entries_ddr` per row while still staging `entries`
     * — runs out part way down the plane and every row past that reads whatever the
     * DMA had left. The map departs at exactly the atom where that total is spent, at
     * two geometries, and inflating this ONE field in proportion makes it affine to the
     * last row. The window grid does not follow it: the output extent and both derived
     * pads come from the real geometry, so a pitched task still synthesizes the
     * trailing pad an inflated PLANE would have read real bytes for.
     * [HW sweep, H96 MAX M9, tests/rk3576_conv_pitch.c] */
    unsigned dma_rows;
    /* A constant granule offset added to both, which moves WHERE in the CBUF this
     * task stages without changing anything about what it computes — see the
     * CBUF-base RE knob below. Zero unless a bring-up sweep sets it. */
    unsigned cbuf_bias        = r76_cbuf_bias();
    unsigned cbuf_fetch_base  = reuse->resident + cbuf_bias;
    unsigned cbuf_window_base = reuse->resident - entries * reuse->retained_rows
                                + cbuf_bias;
    int windowed = (ih < ih_full);
    /* THE TWO KERNEL AXES IN ONE WORD, AND THE LOW HALF IS THE HEIGHT.
     *
     * `0x1024`'s high half carries the kernel: bits [31:24] are the WIDTH and [23:16] the
     * HEIGHT. A SQUARE kernel cannot tell that map from its transpose, and every vendor
     * capture this was transcribed from is square, as is every cell of every conv gate
     * here and every kernel in the first four networks — so the transposed assignment
     * computed a full, correctly sized, entirely plausible surface and nothing could see
     * it. The same shape of trap as the PPU's pad nibbles, on the other block.
     *
     * Measured on Inception V3, which is the first graph here with a non-square kernel:
     * its 34 layers at 1x7, 7x1, 1x3 and 3x1 over planes 17 and 8 are wrong in every
     * element region with the halves the other way round and BIT-EXACT with this one,
     * while all 61 square-kernel convolutions are unmoved.
     * [HW sweep, H96 MAX M9, tests/rk3576_net_gate.c --net iv3]
     *
     * ROCKET_RK3576_KSWAP=1 puts the old assignment back; it is the control that made
     * this a measurement rather than a guess, and it is kept for the same reason. */
    const char *kswap_env = getenv("ROCKET_RK3576_KSWAP");
    int kswap = kswap_env && *kswap_env && *kswap_env != '0';
    uint32_t kword = kswap
                       ? (uint32_t)(((kh - 1u) & 0xFFu) << 8 | ((kw - 1u) & 0xFFu))
                       : (uint32_t)(((kw - 1u) & 0xFFu) << 8 | ((kh - 1u) & 0xFFu));
    /* Address placement — the default is the transcribed one; see the RE knobs. */
    uint16_t faddr_reg = r76_env_reg("ROCKET_RK3576_FADDR_REG", R76_CNA_FEATURE_ADDR);
    uint16_t waddr_reg = r76_env_reg("ROCKET_RK3576_WADDR_REG", R76_CNA_DCOMP_ADDR0);
    int shotgun = getenv("ROCKET_RK3576_ADDR_SHOTGUN") != NULL;
    uint32_t zval[sizeof r76_zero_regs / sizeof r76_zero_regs[0]];
    size_t zi;

    dma_rows = entries_ddr > entries
                 ? (fetch_rows * entries_ddr + entries - 1u) / entries : fetch_rows;
    if (dma_rows > 0xFFFFu) {
        ROCKET_LOGE("rk3576 conv: a pitched feature row needs a DMA budget of %u rows, "
                    "past the 16-bit field — the task needs a smaller row window\n",
                    dma_rows);
        return -1;
    }

    for (zi = 0; zi < sizeof zval / sizeof zval[0]; zi++) {
        uint32_t v = 0;
        if (shotgun) v = cna->decompress_addr0;
        if (r76_zero_regs[zi] == faddr_reg) v = cna->feature_base_addr;
        if (r76_zero_regs[zi] == waddr_reg) v = cna->decompress_addr0;
        zval[zi] = v;
    }
#define R76_ZERO(reg) r76_zero_value(reg, zval)

    /* The vendor's RK3576-only state init, replayed from the regcmd. Its kernel
     * carries a per-SoC `state_init` hook that the RK3588 does not have and that
     * mainline `rocket` therefore has no analog of: it seeds CNA 0x1024 with bit 31
     * in BOTH ping-pong register groups and then clears the pointer
     * (S_POINTER = POINTER_PP_CLEAR | POINTER_PP_MODE | EXECUTER_PP_EN |
     * POINTER_PP_EN). The kernel does it with the PC in slave mode after each
     * power-on; the PC-slave-mode write cannot be replayed from inside a regcmd the
     * PC is fetching, but the CNA writes can. Gated on the knob because it is a
     * cold-start-wall experiment, not part of the transcribed program. */
    if (getenv("ROCKET_RK3576_STATE_INIT")) {
        ops[i++] = NPUOP(OP_REG_CNA, 0x0,        R76_CNA_S_POINTER);
        ops[i++] = NPUOP(OP_REG_CNA, 0x80000000, R76_CNA_KERNEL_OC);
        ops[i++] = NPUOP(OP_REG_CNA, 0x1,        R76_CNA_S_POINTER);
        ops[i++] = NPUOP(OP_REG_CNA, 0x80000000, R76_CNA_KERNEL_OC);
        ops[i++] = NPUOP(OP_REG_CNA, 0x1E,       R76_CNA_S_POINTER);
    }

    /* Interleaved S_POINTER preamble, verbatim from the vendor stream. The early
     * 0x1038 write is the vendor's; it is rewritten in place inside the CNA block.
     *
     * A LUT-consuming task carries 0x30 in all four instead of 0xE — the one thing
     * outside the DPU's own LUT bank that separates a vendor activation from its
     * bare-conv control, and it is in every one of them. */
    ops[i++] = NPUOP(OP_REG_CNA,      lut ? 0x30u : 0xEu,  R76_CNA_S_POINTER);
    ops[i++] = NPUOP(OP_REG_CORE,     lut ? 0x30u : 0xEu,  R76_CORE_S_POINTER);
    ops[i++] = NPUOP(OP_REG_CNA,      0x07, R76_CNA_WEIGHT_SIZE2);
    ops[i++] = NPUOP(OP_REG_DPU,      lut ? 0x30u : 0xEu,  R76_DPU_S_POINTER);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, lut ? 0x30u : 0xEu,  R76_RDMA_S_POINTER);

    /* ---- CNA ----
     * Precision joins conv_mode in this word at the RK3588's CNA_CONV_CON1 offsets;
     * both fields are zero at int8, so the int8 program is byte-unchanged. */
    ops[i++] = NPUOP(OP_REG_CNA,
                     (is_float ? R76_CNA_CON1_FLOAT_EN : 0u) |
                     ((uint32_t)(cna->proc_precision & 0x7) << 7) |
                     ((uint32_t)(cna->in_precision   & 0x7) << 4) |
                     (argb ? (R76_CNA_CON1_GROUP_LINE_OFF |
                              ((8u | ((ic - 1u) & 0x7u)) << R76_CNA_CON1_ARGB_IN_SHIFT) |
                              (is_float ? R76_ARGB_CONV_MODE_FLOAT
                                        : R76_ARGB_CONV_MODE))
                           : (dw ? 1u : 0u)), R76_CNA_CONV_MODE);
    ops[i++] = NPUOP(OP_REG_CNA, 0xFFFu, R76_CNA_CONV_CON2);
    ops[i++] = NPUOP(OP_REG_CNA,
                     ((uint32_t)(cna->conv_y_stride & 0x7) << 3) |
                      (uint32_t)(cna->conv_x_stride & 0x7), R76_CNA_STRIDE);
    /* Bit 31 marks a task that RETAINS rows from the previous one; the low bits are
     * the vendor allocator's, and a hardware sweep found only bit 30 live on the
     * direct int8 path. */
    ops[i++] = NPUOP(OP_REG_CNA,
                     r76_cna_format(windowed) |
                     (reuse->retained_rows ? 0x80000000u : 0u), R76_CNA_FORMAT);
    /* Total weight bytes. The element size scales it — the depthwise form already
     * carries a 2 that is NOT the element size: a depthwise weight occupies a 16-bit
     * slot whatever its precision, which is why the float and int8 cubes for one
     * geometry are the same size. */
    ops[i++] = NPUOP(OP_REG_CNA,
                     argb ? (oc * welems)
                          : (dw ? (r76_dw_cw(oc) * kh * kw * 2u)
                                : (ic * oc * kh * kw) * elem),
                     R76_CNA_WEIGHT_BYTES);
    ops[i++] = NPUOP(OP_REG_CNA, welems, R76_CNA_WEIGHT_ELEMS);
    ops[i++] = NPUOP(OP_REG_CNA, (kword << 16) | (dw ? 1u : (oc - 1u)), R76_CNA_KERNEL_OC);
    /* The CBUF granules this task FILLS — its fetched rows, not its window. The two
     * differ only when the previous task left the leading rows resident. */
    ops[i++] = NPUOP(OP_REG_CNA, ((entries * fetch_rows) << 16) | (ic_prog - 1u),
                     R76_CNA_SURF_CHANNEL);
    ops[i++] = NPUOP(OP_REG_CNA, ((iw - 1u) << 16) | (ih - 1u), R76_CNA_DATAIN_SIZE);
    ops[i++] = NPUOP(OP_REG_CNA, (wbpk << 16) | (ow - 1u), R76_CNA_WBPK_OW);
    ops[i++] = NPUOP(OP_REG_CNA, ow * oh - 1u, R76_CNA_DATAOUT_ATOMS);
    /* 0x07 in every single-task capture; the sliced ARGB program varies its low bits.
     * Only bit 31 is live on the direct int8 path — 0x07, 0x010e and 0x10 are all
     * bit-exact, and setting bit 31 (0x80000010) corrupts the surface.
     * [HW sweep, H96 MAX M9] */
    ops[i++] = NPUOP(OP_REG_CNA, 0x07u | (reuse->continuation ? 0x80000000u : 0u),
                     R76_CNA_WEIGHT_SIZE2);
    /* Granules per feature row, and — in the low half — the CBUF granule where this
     * task's WINDOW begins, which is where its retained rows already sit. */
    ops[i++] = NPUOP(OP_REG_CNA, (entries << 16) | (cbuf_window_base & 0xFFFFu),
                     R76_CNA_CBUF_ENTRIES);
    /* The granule allowance, not a constant — see the CBUF planning section. The
     * captures' two values are F=0 and F=1024, so a planned F reproduces them. The
     * low half is where FETCHING resumes: the window base plus the retained rows. */
    ops[i++] = NPUOP(OP_REG_CNA,
                     R76_CBUF_CON0_BASE |
                     ((cbuf_f & R76_CBUF_F_MASK) << R76_CBUF_F_SHIFT) |
                     (cbuf_fetch_base & 0xFFFFu),
                     R76_CNA_CBUF_CON0);
    /* The low half is the entry count on the direct path and on the int8 first conv.
     * On the FLOAT first conv it is iw instead, measured at three widths — which is
     * also `entries * 8` and `iw*LANES*elem/8`, three readings that coincide at every
     * value the path can take, so the number is settled and the mechanism is not. */
    ops[i++] = NPUOP(OP_REG_CNA,
                     (iw << 16) | ((argb && is_float) ? iw : entries_ddr),
                     R76_CNA_IW_ENTRIES);
    /* Same field packing as RK3588 CNA_CVT_CON0, at a different offset: four 6-bit
     * truncate fields above the four mode bits. Every non-ARGB capture bypasses the
     * converter and leaves all four truncates at zero; the ARGB one runs it, which is
     * where the field layout is confirmed rather than assumed. */
    ops[i++] = NPUOP(OP_REG_CNA,
                     (argb ? r76_argb_cvt_con0(ic) : 0u) |
                     ((uint32_t)(cna->data_sign & 0x1) << 3) |
                     ((uint32_t)(cna->cvt_type  & 0x1) << 1) |
                      (uint32_t)(cna->cvt_bypass & 0x1), R76_CNA_CVT_CON0);
    /* The RK3576 packs two CVT scales per register where the RK3588 pairs each scale
     * with its offset; the offsets get three registers of their own below. */
    ops[i++] = NPUOP(OP_REG_CNA,
                     ((uint32_t)cna->cvt_scale1 << 16) | cna->cvt_scale0, R76_CNA_CVT_SCALE01);
    ops[i++] = NPUOP(OP_REG_CNA,
                     ((uint32_t)cna->cvt_scale3 << 16) | cna->cvt_scale2, R76_CNA_CVT_SCALE23);
    /* CVT_OFFSET0..2, one register per image channel, sign-extended. Zero on every
     * bypassed path; on the ARGB one they carry the uint8 zero point the converter
     * subtracts, which is how that datapath centres a raw camera image. A fourth
     * offset almost certainly lives at 0x1060 — the three here are consecutive and
     * there are four scales — but every capture leaves it zero, so it stays in the
     * address-shotgun list below rather than being claimed. */
    {
        int32_t off = (int32_t)(int16_t)cna->data_offset;
        ops[i++] = NPUOP(OP_REG_CNA, (uint32_t)(ic > 0u ? off : 0), R76_CNA_CVT_OFFSET0);
        ops[i++] = NPUOP(OP_REG_CNA, (uint32_t)(ic > 1u ? off : 0), R76_CNA_CVT_OFFSET1);
        ops[i++] = NPUOP(OP_REG_CNA, (uint32_t)(ic > 2u ? off : 0), R76_CNA_CVT_OFFSET2);
    }
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_ZERO_1060), R76_CNA_ZERO_1060);
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_ZERO_1064), R76_CNA_ZERO_1064);
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_ZERO_1068), R76_CNA_ZERO_1068);
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_ZERO_106C), R76_CNA_ZERO_106C);
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_ZERO_1070), R76_CNA_ZERO_1070);
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_ZERO_1074), R76_CNA_ZERO_1074);
    /* The feature DMA's own view of the fetch. On the normal path it repeats the
     * cube's width and height; on the ARGB path the width becomes the PACKED row's
     * granule count, because the DMA is walking interleaved image bytes and not a
     * channel-group plane. The row count is the BUDGET (see dma_rows above), which is
     * the fetched rows themselves everywhere but a pitched feature buffer. */
    ops[i++] = NPUOP(OP_REG_CNA,
                     ((argb ? line_stride - 1u : iw - 1u) << 16) | (dma_rows - 1u),
                     R76_CNA_DMA_SIZE);
    /* The DMA's channel count is the IMAGE's on the ARGB path — 3 for RGB — while
     * 0x1028 above carries the folded count the MAC sees. The two disagreeing is the
     * clearest signature of this sub-encoding in a capture. */
    ops[i++] = NPUOP(OP_REG_CNA, ic - 1u, R76_CNA_DMA_CHANNEL);
    ops[i++] = NPUOP(OP_REG_CNA,
                     ((pad_right  & 0xFFu) << 24) | ((pad_bottom & 0xFFu) << 16) |
                     (((uint32_t)cna->pad_left & 0xFFu) << 8) | ((uint32_t)cna->pad_top & 0xFFu),
                     R76_CNA_PAD_CON0);
    ops[i++] = NPUOP(OP_REG_CNA, cna->pad_con1, R76_CNA_PAD_CON1);
    ops[i++] = NPUOP(OP_REG_CNA,
                     faddr_reg == R76_CNA_FEATURE_ADDR ? cna->feature_base_addr : 0u,
                     R76_CNA_FEATURE_ADDR);
    ops[i++] = NPUOP(OP_REG_CNA,
                     ((uint32_t)(cna->weight_burst_len & 0xF) << 16) |
                      (uint32_t)(cna->data_burst_len & 0xF), R76_CNA_DMA_CON0);
    ops[i++] = NPUOP(OP_REG_CNA, line_stride, R76_CNA_LINE_STRIDE);
    /* The two surface strides. On the normal path they are the DDR channel-group jump
     * over the FULL plane and over this task's window, in 16-byte atoms. The ARGB
     * image has ONE surface — there is no second channel group to jump to — so both
     * carry the same quantity, the task's own packed rows: all twelve ARGB captures
     * hold line_stride*ih in both, at three window heights of the same image.
     *
     * The DDR one is a caller-supplied quantity when the feature cube is somebody
     * else's surface with a padded stride; `in_surf_elems` is zero everywhere else and
     * the derived plane stands. */
    ops[i++] = NPUOP(OP_REG_CNA,
                     argb ? line_stride * ih
                          : (in_surf_elems ? in_surf_elems : iw * ih_full),
                     R76_CNA_SURF_FULL);
    ops[i++] = NPUOP(OP_REG_CNA,
                     argb ? line_stride * fetch_rows
                          : ((iw * fetch_rows + 3u) & ~3u),
                     R76_CNA_SURF_TASK);
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_ZERO_109C), R76_CNA_ZERO_109C);
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_DCOMP_CTRL), R76_CNA_DCOMP_CTRL);
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_DCOMP_REGNUM), R76_CNA_DCOMP_REGNUM);
    ops[i++] = NPUOP(OP_REG_CNA,
                     (waddr_reg == R76_CNA_DCOMP_ADDR0 || shotgun) ? cna->decompress_addr0 : 0u,
                     R76_CNA_DCOMP_ADDR0);
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_DCOMP_AMOUNT), R76_CNA_DCOMP_AMOUNT);
    ops[i++] = NPUOP(OP_REG_CNA, R76_ZERO(R76_CNA_DCOMP_AMOUNT1), R76_CNA_DCOMP_AMOUNT1);
    /* The full plane's extent — and on the ARGB path, NOT that at all. There it is
     * the CBUF row's granule count and that count minus one, which is why a
     * normal-path decode of an ARGB capture reads a nonsense full-plane height. The
     * ARGB program carries no full-plane height anywhere: it does not need one,
     * because its single surface makes every stride the task's own. */
    /* BOTH halves carry iw-1 on the normal path. The low half read as the full plane
     * height for as long as every capture was a SQUARE plane, where the two cannot be
     * told apart; rectangular ones separate them, and 142 non-ARGB programs carry
     * iw-1 twice with no exception. The ARGB path's own two halves are granule
     * counts. */
    ops[i++] = NPUOP(OP_REG_CNA,
                     argb ? (is_float ? (((entries - 1u) << 16) | (entries - 1u))
                                      : (((entries - 1u) << 16) | (entries - 2u)))
                          : (((iw - 1u) << 16) | (iw - 1u)),
                     R76_CNA_DATAIN_FULL);

    /* ---- CORE ----
     * The mode word carries the processing precision at the RK3588's CORE_MISC_CFG
     * bits [10:8]; the rest of the word is transcribed and is zero there at int8.
     * The ARGB path sets one more bit (7) below that field, transcribed as its own
     * mode constant — everything past the CNA is otherwise the direct path's. */
    /* Bit 0 is an INT8 marker, not part of the mode: every float capture clears it on
     * both the direct and the first-conv path, where every integer one sets it.
     * [source-confirmed, RKNN-Toolkit2 rk3576 float build] */
    ops[i++] = NPUOP(OP_REG_CORE,
                     ((argb ? R76_CORE_MISC_ARGB
                            : (dw ? R76_CORE_MISC_DW : R76_CORE_MISC_DIRECT))
                      & (is_float ? ~1u : ~0u)) |
                     ((uint32_t)(core->proc_precision & 0x7) << 8),
                     R76_CORE_MISC_CFG);
    ops[i++] = NPUOP(OP_REG_CORE, ((oh - 1u) << 16) | (ow - 1u), R76_CORE_DATAOUT_SIZE0);
    ops[i++] = NPUOP(OP_REG_CORE, oc - 1u, R76_CORE_DATAOUT_SIZE1);
    ops[i++] = NPUOP(OP_REG_CORE, 0x0, R76_CORE_CLIP_TRUNCATE);

    /* ---- DPU ---- */
    ops[i++] = NPUOP(OP_REG_DPU, dw ? R76_DPU_FMODE_DW : R76_DPU_FMODE_DIRECT,
                     R76_DPU_FEATURE_MODE);
    /* Zero in every capture, which is all-int8 under the RK3588's DPU_DATA_FORMAT
     * packing — so the int8 program is byte-unchanged and the three fields are
     * placed where that packing puts them. */
    /* The middle field is the DPU's INPUT width, and the vendor leaves it at zero on
     * the float path — only the output width and the processing precision move there.
     * Ours used to carry fp16 in all three; that computes, so the field is inert at
     * these shapes, but the capture is what this emitter reproduces. */
    ops[i++] = NPUOP(OP_REG_DPU,
                     ((uint32_t)(dpu->out_precision  & 0x7) << 29) |
                     (is_float ? 0u : ((uint32_t)(dpu->in_precision & 0x7) << 26)) |
                      (uint32_t)(dpu->proc_precision & 0x7), R76_DPU_DATA_FORMAT);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_OFFSET_PEND);
    ops[i++] = NPUOP(OP_REG_DPU, dpu->dst_base_addr, R76_DPU_DST_BASE_ADDR);
    /* The destination surface, in output elements. On the DEPTHWISE path it is
     * ROUNDED UP TO FOUR: the rounding is invisible at every power-of-two plane, and
     * the vendor programs 272, 324 and 364 for ow*oh_full of 270, 323 and 361.
     *
     * The DIRECT path does NOT round, and that is a hardware result rather than a
     * transcription — every direct capture happens to have a plane that is already a
     * multiple of four, so the captures cannot say, but a VALID-padded oc=128 conv at
     * ow*oh_full = 105 is bit-exact with 105 here and wrong with 108. Both paths feed
     * the same value into the surface add below, so a wrong one there mis-places
     * every output channel group past the first. */
    ops[i++] = NPUOP(OP_REG_DPU, dw ? ((ow * oh_full + 3u) & ~3u) : (ow * oh_full),
                     R76_DPU_DST_SURF);
    ops[i++] = NPUOP(OP_REG_DPU, ow - 1u, R76_DPU_CUBE_WIDTH);
    ops[i++] = NPUOP(OP_REG_DPU, oh - 1u, R76_DPU_CUBE_HEIGHT);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_CUBE_NOTCH);
    ops[i++] = NPUOP(OP_REG_DPU, oc - 1u, R76_DPU_CUBE_CHANNEL);
    ops[i++] = NPUOP(OP_REG_DPU, ((oc - 1u) << 16) |
                     (dw ? R76_DPU_WDMA0_DW : R76_DPU_WDMA0_DIRECT), R76_DPU_WDMA_SIZE0);
    ops[i++] = NPUOP(OP_REG_DPU, ((oh - 1u) << 16) | (ow - 1u), R76_DPU_WDMA_SIZE1);
    ops[i++] = NPUOP(OP_REG_DPU,
                     is_float ? R76_DPU_NOTCH_FLOAT
                              : (dw ? R76_DPU_NOTCH_DW : R76_DPU_NOTCH_DIRECT),
                     R76_DPU_NOTCH_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_403C);
    ops[i++] = NPUOP(OP_REG_DPU, dw ? 0x0u : 0x1u, R76_DPU_BS_ALU_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_BS_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_BS_MAX);
    ops[i++] = NPUOP(OP_REG_DPU,
                     is_float ? R76_DPU_BS_CFG_FLOAT
                              : (dw ? r76_dw_bs_cfg(oc) : R76_DPU_BS_CFG_DIRECT),
                     R76_DPU_BS_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_BN_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_BN_MAX);
    /* BN STAYS BYPASSED even with the LUT on. Every vendor activation carries 0x20
     * here — the stage active — and copying that pins the LUT at the table join for
     * every input, because BN then multiplies by an operand register no vendor program
     * writes and this register file is not cleared between jobs. The readout still
     * tracks the table, so it reads as a working LUT with no input map anywhere. */
    ops[i++] = NPUOP(OP_REG_DPU, 0x903u, R76_DPU_BN_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_EW_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_EW_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_EW_MIN2);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_EW_MAX2);
    /* The LUT is the EW stage: EW_LUT_BYPASS is bit 7 here — the RK3588's field layout
     * at a moved offset — and clearing it, with bit 0 (the stage's own bypass), is what
     * turns the table on. */
    ops[i++] = NPUOP(OP_REG_DPU, lut ? R76_DPU_EW_CFG_LUT : 0x010041C1u,
                     R76_DPU_EW_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_EW_CVT_OFFSET);
    ops[i++] = NPUOP(OP_REG_DPU, 0x1u, R76_DPU_EW_CVT_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_EW_CLAMP_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_EW_CLAMP_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_EW_OP_VALUE0);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_EW_OP_VALUE1);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_409C);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_OUT_CLAMP_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_OUT_CLAMP_MAX);
    /* OUT_CVT triple: offset / scale / shift in three consecutive registers, the
     * same requant the RK3588 puts at 0x4080-0x4088. */
    ops[i++] = NPUOP(OP_REG_DPU, dpu->out_cvt_offset, R76_DPU_OUT_CVT_OFFSET);
    ops[i++] = NPUOP(OP_REG_DPU,
                     ((uint32_t)(dpu->fp32tofp16_en & 0x1) << 16) | dpu->out_cvt_scale,
                     R76_DPU_OUT_CVT_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, dpu->out_cvt_shift,  R76_DPU_OUT_CVT_SHIFT);
    /* The channel-group jump: a full destination surface PLUS the rows of it this
     * task does not write. The writer walks the task's rows and then adds this to
     * reach the same rows of the next group, so a windowed task has to be told about
     * the rows it skipped or every group past the first lands short.
     *
     * The plane term is a whole DESTINATION SURFACE — 0x401C above, which the
     * depthwise path rounds up to four and the direct path does not — taken twice on
     * the direct path and four times on the depthwise one.
     *
     * Pinned on hardware by driving an oc=64 conv (two output-channel groups, so
     * group 1 is exposed) through 32-, 16- and 8-row windows of the same 64-row
     * plane: only 2*surface - ow*oh_task is bit-exact at all three, and the whole
     * surface is wrong at any other value. [HW sweep, H96 MAX M9]
     *
     * DEPTHWISE DOUBLES THE PLANE TERM. Every one of 112 captured depthwise tasks
     * carries the 4 — channel counts 8 to 256, strides 1 and 2, kernels 1 to 7 and
     * seven window heights — and the direct form holds over the same window heights
     * at three channel counts. The kernel sweep is what separates the 4 from kh+1,
     * which the original all-k=3 captures could not.
     *
     * Read as an advance rather than a stride the pattern is one quantity: the
     * writer covers the task's own rows and then adds this, so it advances two
     * surfaces per direct pass and four per depthwise pass. At 8 bytes per unit that
     * is exactly one 16-channel output surface per direct pass and two of them — 32
     * channels — per depthwise pass, which is the width the depthwise datapath has no
     * input-channel reduction to spend. The unit is inferred; the values are
     * transcribed. */
    ops[i++] = NPUOP(OP_REG_DPU,
                     dw ? 4u * ((ow * oh_full + 3u) & ~3u) - ow * oh
                        : 2u * (ow * oh_full) - ow * oh,
                     R76_DPU_SURFACE_ADD);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_40BC);
    ops[i++] = NPUOP(OP_REG_DPU, 0x04440100u, R76_DPU_CONST_40C0);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_40C8);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_40CC);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0040FFFFu, R76_DPU_CONST_40D0);
    /* The LUT bank. Zero throughout in every convolution capture, which is the LUT
     * bypassed; a task that carries a window writes it here, and the WHOLE bank is
     * written either way so a stale value from the previous job cannot survive into
     * this one. LE_START and LO_END each occupy four consecutive lanes. */
    for (unsigned r = 0x4100; r <= 0x4120; r += 4) {
        uint32_t v = 0;
        if (lut) {
            if (r == R76_LUT_CFG)                    v = R76_LUT_CFG_USE;
            else if (r == R76_LUT_INFO)              v = ((uint32_t)lut->sel << 16) |
                                                         ((uint32_t)lut->sel << 8);
            else if (r >= R76_LUT_LE_START && r <= R76_LUT_LE_START + 0xC)
                                                     v = (uint32_t)lut->le_start;
        }
        ops[i++] = NPUOP(OP_REG_DPU, v, r);
    }
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4130);
    for (unsigned r = 0x4140; r <= 0x4154; r += 4)
        ops[i++] = NPUOP(OP_REG_DPU,
                         (lut && r <= R76_LUT_LO_END + 0xC) ? (uint32_t)lut->lo_end : 0u,
                         r);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4160);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4170);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4174);
    /* The two overflow clamps, in the vendor's own packing: the value alone in the
     * high half of the first register and doubled across the second. */
    for (unsigned r = 0x4184; r <= 0x4194; r += 4) {
        uint32_t v = 0;
        if (lut) {
            uint32_t c = (uint32_t)(uint16_t)(r <= R76_LUT_UNDERFLOW + 4
                                              ? lut->clamp_lo : lut->clamp_hi);
            if (r == R76_LUT_UNDERFLOW || r == R76_LUT_OVERFLOW)     v = c << 16;
            else if (r == R76_LUT_UNDERFLOW + 4 || r == R76_LUT_OVERFLOW + 4)
                                                                     v = (c << 16) | c;
        }
        ops[i++] = NPUOP(OP_REG_DPU, v, r);
    }

    /* ---- DPU_RDMA ---- */
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, ow - 1u, R76_RDMA_CUBE_WIDTH);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, oh - 1u, R76_RDMA_CUBE_HEIGHT);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, oc - 1u, R76_RDMA_CUBE_CHANNEL);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_SRC_BASE_ADDR);
    /* The integer value on every path, deliberately — see R76_RDMA_ERDMA_FLOAT. */
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, dw ? R76_RDMA_BRDMA_DW : R76_RDMA_BRDMA_DIRECT,
                     R76_RDMA_BRDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, dpu->bias_base_addr, R76_RDMA_BS_BASE_ADDR);
    /* BS_BASE_ADDR1 is a live operand base, not a spare word. The DPU reads one
     * 32-bit shift word through it per task and applies it to the accumulator:
     * bits[5:0] right-shift the non-negative results, bits[13:8] the negative ones.
     * The captures hold 0 here because it is an address the vendor runtime patches,
     * exactly like the feature/weight/output/bias bases — and 0 is a REAL buffer on
     * this stack, since the per-fd IOVA space bump-starts at 0. Leaving it at zero
     * therefore makes the DPU take two shift amounts out of whatever BO was
     * allocated first, which is normally the feature cube: its byte 0 and byte 1
     * silently attenuate the result by up to 2^-63 per sign. Point it at the zeroed
     * group past the A/B/C coefficients, which is what "no shift" means. */
    ops[i++] = NPUOP(OP_REG_DPU_RDMA,
                     dpu->bias_base_addr + (uint32_t)r76_shift_offset_p(oc, dw),
                     R76_RDMA_BS_BASE_ADDR1);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, lut ? 0x1Au : 0x0u, R76_RDMA_NRDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_BN_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_ZERO_5030);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, is_float ? R76_RDMA_ERDMA_FLOAT : 0x41u,
                     R76_RDMA_ERDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_EW_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_EW_SURF);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA,
                     is_float ? R76_RDMA_FMODE_FLOAT
                              : (dw ? R76_RDMA_FMODE_DW : R76_RDMA_FMODE_DIRECT),
                     R76_RDMA_FEATURE_MODE);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_SRC_DMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_SURF_NOTCH);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_PAD_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_EW_SURF_NOTCH);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_ZERO_5078);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_ZERO_507C);

    /* Registers the vendor stream never writes, added on request. The vendor program
     * leaves several per-stage DPU operand slots untouched (0x4040, 0x4054, 0x4064,
     * 0x4068) and this register file is not cleared between jobs, so those hold a
     * reset default or the previous job's value. ROCKET_RK3576_ADD="0x4054=0,..."
     * writes them, which is the only way to tell "the vendor relies on the reset
     * value" apart from "the vendor's own runtime wrote it earlier". The block is
     * taken from the offset — including the PC block below 0x1000, which is how the
     * cold-start wall was shown to be out of a regcmd's reach: PC_TASK_CON is the
     * cause, but the PC latches it before it starts fetching, so writing it from
     * inside the stream changes nothing — and the writes land before the trailer so
     * they take effect in the same task. */
    {
        const char *spec = getenv("ROCKET_RK3576_ADD");
        const char *p;
        for (p = spec ? spec : ""; *p && i < RK3576_CONV_TASK_OPS - 4; ) {
            char *end;
            unsigned long reg, val;
            uint16_t target;
            reg = strtoul(p, &end, 0);
            if (end == p || *end != '=') break;
            p = end + 1;
            val = strtoul(p, &end, 0);
            if (end == p) break;
            p = end;
            while (*p == ',' || *p == ' ') p++;
            switch (reg >> 12) {
            case 0x0: target = OP_REG_PC;       break;
            case 0x1: target = OP_REG_CNA;      break;
            case 0x3: target = OP_REG_CORE;     break;
            case 0x4: target = OP_REG_DPU;      break;
            case 0x5: target = OP_REG_DPU_RDMA; break;
            default:  ROCKET_LOGE("rk3576 add: no block owns register 0x%04lx\n", reg);
                      continue;
            }
            ops[i++] = NPUOP(target, (uint32_t)val, (uint16_t)reg);
            ROCKET_LOGI("rk3576 add: reg 0x%04lx = 0x%08lx\n", reg, val);
        }
    }

    /* The PC trailer the vendor stream does not carry. Without it `rocket` submits a
     * fully-configured pipeline that never starts. */
    ops[i++] = NPUOP(OP_NONE, 0x0, 0x0);
    ops[i++] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
    ops[i++] = NPUOP(OP_40, 0x0, 0x0);
    ops[i++] = NPUOP(OP_ENABLE, 0x1D, PC_OPERATION_ENABLE);

#undef R76_ZERO
    return i;
}

/* ============================================================================
 * SECTION — register override (bring-up RE)
 *
 * ROCKET_RK3576_SET="0x1018=0x40000505,0x1040=0x14000000" rewrites the value of
 * any already-emitted register before submit. Several words in this encoding are
 * pinned to one or two observed values rather than decoded into fields — the
 * format word 0x1018, the CBUF word 0x1040, the mode constants — and the only way
 * to decode them is to vary one at a time on the part. Overriding after emission
 * keeps the sweep out of the transcribed program: with the variable unset the
 * emitted bytes are exactly what the gate checks.
 * ==========================================================================*/
static void r76_apply_overrides(uint64_t *ops, int n)
{
    const char *spec = getenv("ROCKET_RK3576_SET");
    const char *p;
    if (!spec || !*spec) return;
    for (p = spec; *p; ) {
        char *end;
        unsigned long reg, val;
        int i, hits = 0;
        reg = strtoul(p, &end, 0);
        if (end == p || *end != '=') break;
        p = end + 1;
        val = strtoul(p, &end, 0);
        if (end == p) break;
        p = end;
        while (*p == ',' || *p == ' ') p++;
        for (i = 0; i < n; i++) {
            if ((uint16_t)(ops[i] & 0xFFFF) == (uint16_t)reg &&
                (uint16_t)(ops[i] >> 48) != 0) {
                ops[i] = (ops[i] & ~(0xFFFFFFFFull << 16)) |
                         ((uint64_t)(uint32_t)val << 16);
                hits++;
            }
        }
        ROCKET_LOGI("rk3576 override: reg 0x%04lx = 0x%08lx (%d write%s patched)\n",
                    reg, val, hits, hits == 1 ? "" : "s");
    }
}

/* The emitted program, one `BLK reg=value` a line, when ROCKET_RK3576_DUMP is set.
 * Two runs diffed against each other is how a geometry the part refuses to compute is
 * separated from one it does: the difference is a short list of registers rather than
 * a guess at which one carries the mode. Printed AFTER the overrides, so a sweep shows
 * what was actually submitted. */
static void r76_dump_program(const uint64_t *ops, int n)
{
    static const char *blk[] = { "?   ", "PC  ", "CNA ", "?   ", "CORE",
                                 "?   ", "?   ", "?   ", "DPU ", "RDMA" };
    int i;
    if (!getenv("ROCKET_RK3576_DUMP")) return;
    for (i = 0; i < n; i++) {
        unsigned op = (unsigned)(ops[i] >> 48);
        unsigned reg = (unsigned)(ops[i] & 0xFFFF);
        uint32_t val = (uint32_t)((ops[i] >> 16) & 0xFFFFFFFFu);
        unsigned b = (op >> 8) & 0xF;
        ROCKET_LOGI("REGCMD %3d %s 0x%04x = 0x%08x\n",
                    i, b < sizeof blk / sizeof blk[0] ? blk[b] : "?   ", reg, val);
    }
}

/* ============================================================================
 * SECTION — descriptor fill
 * ==========================================================================*/

/* `dpu_proc` is the DPU's own PROC_PRECISION, which is the operand width every other
 * block also carries — except on the raw-int32 writer, where widening it alone doubles
 * the writer's byte budget without touching the arithmetic. See
 * gen_conv2d_int8_rk3576_i32out_wide(). */
static int gen_conv2d_rk3576_fill(conv_params_t *p, int dw, unsigned prec,
                                  unsigned out_prec, unsigned dpu_proc,
                                  const struct r76_reuse *reuse)
{
    npu_cna_desc  cna  = {0};
    npu_core_desc core = {0};
    npu_dpu_desc  dpu  = {0};

    unsigned IC = p->ic, IH = p->ih, IW = p->iw;
    unsigned OC = p->oc, OH = p->oh, OW = p->ow;
    unsigned KH = p->kh, KW = p->kw;
    unsigned ih_full = p->ih_full ? p->ih_full : IH;
    unsigned oh_full = p->oh_full ? p->oh_full : OH;
    unsigned pad_right, pad_bottom;
    unsigned cbuf_f = 0;
    /* A packed-image input takes the first-conv sub-encoding, which is a different
     * CNA program rather than the same one at a small channel count. */
    int argb = (IC <= R76_ARGB_LANES);

    if (argb && dw) {
        ROCKET_LOGE("rk3576 conv: the first-conv ARGB datapath has no depthwise form "
                    "(no capture, and the channel fold leaves nothing to be depthwise "
                    "over)\n");
        return -1;
    }
    if (argb && prec != precision_int8 && prec != precision_float16) {
        ROCKET_LOGE("rk3576 conv: the first-conv ARGB datapath has an int8 and an fp16 "
                    "form and nothing else — precision %u has no transcribed mode "
                    "word\n", prec);
        return -1;
    }
    if (argb && (IW % 16u)) {
        ROCKET_LOGE("rk3576 conv: the first-conv ARGB datapath needs iw a multiple of "
                    "16 (iw=%u); its DDR row stride and CBUF row are both counted in "
                    "16-byte granules\n", IW);
        return -1;
    }
    if (IH > ih_full || OH > oh_full) {
        ROCKET_LOGE("rk3576 conv: task rows (ih=%u oh=%u) exceed the full plane "
                    "(ih_full=%u oh_full=%u)\n", IH, OH, ih_full, oh_full);
        return -1;
    }
    if (IW == 0 || IH == 0 || OW == 0 || OH == 0 || KH == 0 || KW == 0 || OC == 0) {
        ROCKET_LOGE("rk3576 conv: zero geometry\n");
        return -1;
    }
    if (IW > 0xFFFFu || ih_full > 0xFFFFu || OW > 0xFFFFu || oh_full > 0xFFFFu ||
        IC > 0xFFFFu || OC > 0xFFFFu) {
        ROCKET_LOGE("rk3576 conv: a geometry term exceeds the 16-bit register half\n");
        return -1;
    }
    /* A caller-supplied feature group stride only ever describes a buffer whose groups
     * sit FURTHER apart than the plane. A smaller one would make the groups overlap,
     * which is not a layout anything on this part produces, and the ARGB program has a
     * single surface with no group jump at all. */
    if (p->in_surf_elems) {
        if (argb) {
            ROCKET_LOGE("rk3576 conv: the first-conv ARGB datapath has one surface and "
                        "no channel-group stride to set\n");
            return -1;
        }
        if (p->in_surf_elems < IW * ih_full) {
            ROCKET_LOGE("rk3576 conv: a feature group stride of %u elements is shorter "
                        "than the %ux%u plane it walks\n",
                        p->in_surf_elems, IW, ih_full);
            return -1;
        }
    }
    /* A row pitch describes a plane sitting inside WIDER rows and nothing else. The
     * ARGB path's row stride is a granule count the emitter derives from the packed
     * image, and its single surface makes every stride the task's own. */
    if (p->in_pitch_w && p->in_pitch_w != IW) {
        if (argb || dw) {
            ROCKET_LOGE("rk3576 conv: only the DIRECT datapath takes a feature row "
                        "pitch — the ARGB one derives its row stride from the packed "
                        "image and the depthwise one has an unmeasured feature "
                        "granule\n");
            return -1;
        }
        /* BOTH widths are GRANULE counts and neither may round. The pitch is what the
         * DMA advances by and the plane is what it stages, and the budget that ties
         * them is `rows * plane_granules` — so a plane whose own row does not fill a
         * whole granule already carries a rounding the budget arithmetic cannot undo,
         * and every row past the first lands short. [HW sweep: iw=21 at ic=32 is 10.5
         * granules and computes wrong at every pitch, where iw=28 at the same ic is
         * bit-exact at four.] */
        if ((p->in_pitch_w * IC * r76_elem_bytes(prec)) % 64u ||
            (IW * IC * r76_elem_bytes(prec)) % 64u) {
            ROCKET_LOGE("rk3576 conv: a feature row pitch needs BOTH the %u-wide pitch "
                        "and the %u-wide plane to fill whole 64-byte DDR granules at "
                        "ic=%u (%u and %u bytes)\n", p->in_pitch_w, IW, IC,
                        p->in_pitch_w * IC * r76_elem_bytes(prec),
                        IW * IC * r76_elem_bytes(prec));
            return -1;
        }
        if (p->in_pitch_w < IW) {
            ROCKET_LOGE("rk3576 conv: a feature row pitch of %u elements is narrower "
                        "than the %u-wide plane it carries\n", p->in_pitch_w, IW);
            return -1;
        }
        if (p->in_pitch_w > 0xFFFFu) {
            ROCKET_LOGE("rk3576 conv: a feature row pitch of %u elements exceeds the "
                        "register field\n", p->in_pitch_w);
            return -1;
        }
        if (!p->in_surf_elems && p->in_pitch_w != IW) {
            ROCKET_LOGE("rk3576 conv: a feature row pitch of %u needs the channel-group "
                        "stride told too — the derived %ux%u plane does not describe a "
                        "pitched buffer\n", p->in_pitch_w, IW, ih_full);
            return -1;
        }
    }
    /* 0x1028 packs entries*ih in its high half. Against the SCALED entry count, which
     * is what the task programs — a 2-byte element reaches the field twice as fast. */
    {
        unsigned ent = argb ? r76_argb_entries(IW, r76_elem_bytes(prec),
                                              r76_is_float(prec))
                            : r76_data_entries_p(IW, IC, r76_elem_bytes(prec));
        if (ent * IH > 0xFFFFu) {
            ROCKET_LOGE("rk3576 conv: CBUF entries*rows (%u) exceeds the 16-bit field — "
                        "the task needs a smaller row window\n", ent * IH);
            return -1;
        }
    }
    if (dw && IC != OC) {
        ROCKET_LOGE("rk3576 dw conv: depthwise needs ic == oc (%u vs %u)\n", IC, OC);
        return -1;
    }
    /* The channel-granularity contract. Warned rather than rejected, because the
     * emitter must still reproduce the vendor's own programs verbatim for the
     * register-fidelity gate — the vendor's conv2d capture is ic=16 — and because
     * the correction changes the caller's buffer sizes, so it cannot be applied
     * silently here. See the header for what each violation looks like on hardware.
     *
     * The 32-channel rule is the INTEGER cube's, and the float cube does not share it:
     * its groups are 8 output channels and 16 input ones, and its input axis is bounded
     * by the surface-index defect long before granularity binds (ic=8, one contraction
     * step — gen_conv2d_fp16_rk3576() rejects past that). So the float path is warned
     * against its own oc group, not against 32. */
    if (r76_is_float(prec)) {
        if (!dw && (OC % R76_FP16_W_OC_GROUP))
            ROCKET_LOGW("rk3576 conv: oc=%u is a partial %u-channel float group and "
                        "will compute WRONG. Size the output and coefficient buffers "
                        "for %u channels and pass that count\n",
                        OC, R76_FP16_W_OC_GROUP,
                        ((OC + R76_FP16_W_OC_GROUP - 1u) / R76_FP16_W_OC_GROUP)
                        * R76_FP16_W_OC_GROUP);
    } else {
        if (!dw && !argb && IC != rocket_rk3576_pad_ic(IC))
            ROCKET_LOGW("rk3576 conv: ic=%u is a partial 32-channel group and will "
                        "compute WRONG. Pad the feature cube to %u channels and pass "
                        "that count (rocket_rk3576_pad_ic)\n",
                        IC, rocket_rk3576_pad_ic(IC));
        if (!dw && OC != rocket_rk3576_pad_oc(OC))
            ROCKET_LOGW("rk3576 conv: oc=%u is a partial 32-channel group and will "
                        "compute WRONG. Size the output and coefficient buffers for %u "
                        "channels and pass that count (rocket_rk3576_pad_oc)\n",
                        OC, rocket_rk3576_pad_oc(OC));
    }

    pad_right  = r76_trail_pad(OW, p->stride_x, KW, p->pad_left, IW);
    pad_bottom = r76_trail_pad(OH, p->stride_y, KH, p->pad_top,  IH);

    cna.conv_mode      = dw ? 3 : direct_convolution;
    cna.in_precision   = (uint8_t)prec;
    /* PROC_PRECISION is the OPERAND width, so a float conv programs float16 here
     * even though it accumulates in fp32. An earlier reading had this field naming
     * the datapath and wanting fp32; that came across from the RK3588 and is wrong
     * for this part, where fp32 makes every feature surface read twice. It survived
     * because the probes that endorsed it held the feature uniform in the channel
     * axis, where a doubled read cannot be told from a scale.
     * [source-confirmed, RKNN-Toolkit2 rk3576 float build; HW sweep, H96 MAX M9] */
    cna.proc_precision = (uint8_t)r76_proc_precision(prec);
    cna.conv_x_stride  = p->stride_x;
    cna.conv_y_stride  = p->stride_y;
    cna.datain_width   = (uint16_t)IW;
    cna.datain_height  = (uint16_t)IH;
    cna.datain_channel = (uint16_t)IC;
    cna.dataout_width  = (uint16_t)OW;
    cna.dataout_height = (uint16_t)OH;
    cna.weight_width   = (uint8_t)KW;
    cna.weight_height  = (uint8_t)KH;
    cna.weight_kernels = (uint16_t)OC;
    if (!argb) {
        /* CVT: pass the int8 feature straight through (unit scale, zero offset) —
         * the value every non-ARGB capture carries. */
        cna.data_sign   = 1;
        cna.cvt_type    = 1;
        cna.cvt_bypass  = 1;
        cna.cvt_scale0  = 1;
        cna.cvt_scale1  = 1;
        cna.cvt_scale2  = 1;
        cna.cvt_scale3  = 1;
        /* Border pad constant, in the uint8-centered domain, as on the RK3588. */
        cna.pad_con1    = (uint32_t)(((uint32_t)p->input_zero_point & 0xFFu) - 0x80u);
    } else {
        /* CVT: the converter is the ARGB path's uint8 -> int8 stage. Each live image
         * channel gets Q14 unity and the zero point as a subtracted offset, so the
         * lane the MAC sees is (pixel - zero_point) exactly. The unused lanes keep
         * the bypassed path's scale of 1, which is what the captures carry.
         *
         * The border then has to be padded in the RAW byte domain, since the pad is
         * inserted BEFORE the converter runs: the constant is the uint8 zero point
         * itself, replicated once per image channel, and the CVT maps it to 0. That
         * is the opposite of the normal path, where pad_con1 is already centred. */
        cna.cvt_scale0  = R76_ARGB_CVT_SCALE;
        cna.cvt_scale1  = (uint16_t)(IC > 1u ? R76_ARGB_CVT_SCALE : 1u);
        cna.cvt_scale2  = (uint16_t)(IC > 2u ? R76_ARGB_CVT_SCALE : 1u);
        cna.cvt_scale3  = (uint16_t)(IC > 3u ? R76_ARGB_CVT_SCALE : 1u);
        /* The centring is the INT8 path's alone, and on the float path the converter
         * is BYPASSED outright — a float packed image arrives as fp16 already in the
         * value domain the MAC wants. So the offsets read zero, the border pads with a
         * float zero, and CVT_CON0's mode bits are the direct path's minus cvt_type.
         * (The per-channel truncates and Q14 scales stay in the word: the vendor emits
         * them under bypass, where they are inert.) */
        if (r76_is_float(prec)) {
            cna.data_sign   = 1;
            cna.cvt_bypass  = 1;
            cna.data_offset = 0;
            cna.pad_con1    = 0;
        } else {
            cna.data_offset = (uint16_t)(int16_t)
                              -(int16_t)(((int)p->input_zero_point + 0x80) & 0xFF);
            {
                unsigned zp = (unsigned)(((int)p->input_zero_point + 0x80) & 0xFF), c;
                cna.pad_con1 = 0;
                for (c = 0; c < IC; c++) cna.pad_con1 |= zp << (8 * c);
            }
        }
    }
    cna.pad_left    = p->pad_left;
    cna.pad_top     = p->pad_top;
    cna.feature_base_addr = p->input_dma;
    cna.decompress_addr0  = p->weights_dma;
    cna.weight_burst_len  = 0xF;
    cna.data_burst_len    = 0xF;
    /* NOT the DDR row jump, whatever its RK3588 name says: a readout of the fetch map
     * has this register advancing the address by its own value once every FOUR rows
     * while the rows in between step by the fetched width, so it cannot express a
     * pitch and a pitched program leaves it derived. The row stride is 0x1044's low
     * half. [HW sweep, H96 MAX M9, tests/rk3576_conv_pitch.c map mode] */
    cna.line_stride       = IW * 4u;

    /* The CORE keeps the OPERAND precision, not the datapath one — the two words
     * disagree on this part. Carrying fp32 here as well returns a surface that is
     * wrong by an arbitrary factor. [HW sweep, H96 MAX M9] */
    core.proc_precision  = (uint8_t)prec;
    core.dw_en           = dw ? 1 : 0;
    core.dataout_channel = (uint16_t)(OC - 1u);
    core.dataout_width   = (uint16_t)(OW - 1u);
    core.dataout_height  = (uint16_t)(OH - 1u);

    dpu.conv_mode     = dw ? 3 : direct_convolution;
    dpu.out_precision = (uint8_t)out_prec;
    dpu.in_precision  = (uint8_t)prec;
    dpu.proc_precision = (uint8_t)dpu_proc;
    /* The fp32 accumulator reaches DDR as its own 32-bit word unless this narrows it.
     * The output WIDTH field (0x4010 [31:29]) selects how much of that word is
     * written and nothing else: at 5 the whole fp32, at 2 or 3 its low or high half,
     * so an fp16 program without the narrowing writes the low mantissa bits — zero
     * for every small value — and reads as a dead MAC. With the narrowing on, width 2
     * writes true fp16 and width 3 writes the top half, which is bfloat16.
     * [HW sweep, H96 MAX M9] */
    dpu.fp32tofp16_en = (prec == precision_float16 || prec == precision_bfloat16);
    dpu.dst_base_addr = p->output_dma;
    dpu.dst_surf_stride = OW * oh_full;
    dpu.bias_en         = 1;
    dpu.bias_base_addr  = p->bias_dma;

    /* OUT_CVT requant, identical arithmetic to the RK3588 int8-out path:
     *   out_i8 = sat8( round(acc_i32 * scale >> shift) + (out_zp - 0x80) )
     * with scale/shift derived from the fp32 conv scale exactly as the vendor
     * (QNNPACK) does. Per-tensor only.
     *
     * A float output has nothing to requant: the RK3588's fp16 path leaves the
     * converter at unity and the accumulator is already in the output format. Unity
     * here is scale 1 << 14 with the matching shift, which is what this arithmetic
     * produces for conv_scale == 1.0 — but the fp16 datapath is unvalidated on this
     * part, so the sweep drives 0x40AC-0x40B4 directly when it needs to. */
    if (prec == precision_float16 || prec == precision_bfloat16 ||
        out_prec == precision_int32) {
        /* Unity, and it has to be EXACTLY unity on the int32-out path: the QNNPACK
         * derivation below lands on scale 16385 for a conv_scale of 1.0, which drifts
         * the accumulator by acc/16384 — invisible on the int8 writer, where the
         * requant divides it away, and a silent off-by-one on the raw int32 word. */
        dpu.out_cvt_scale  = 1u << 14;
        dpu.out_cvt_shift  = 14;
        dpu.out_cvt_offset = 0;
    } else {
        unsigned shift, scale;
        float conv_scale = (p->out_scale != 0.0f)
            ? (p->in_scale * p->w_scale) / p->out_scale : 1.0f;
        rocket_rk3576_requant_params(conv_scale, &scale, &shift);
        dpu.out_cvt_scale  = (uint16_t)scale;
        dpu.out_cvt_shift  = (uint8_t)shift;
        dpu.out_cvt_offset = (uint32_t)(p->output_zero_point - 0x80);
    }

    /* The CBUF granule allowance. Planned rather than constant, because a plane over
     * the allowance computes WRONG with the DPU still writing a full surface — there
     * is no fault to catch it. ROCKET_RK3576_CBUF_F forces F for bring-up sweeps and
     * bypasses both fit checks, which is how the allowance was characterised. */
    {
        const char *e = getenv("ROCKET_RK3576_CBUF_F");
        if (e && *e) {
            cbuf_f = (unsigned)strtoul(e, NULL, 0);
            ROCKET_LOGI("rk3576 cbuf: F forced to %u (%u granules of feature budget)\n",
                        cbuf_f, R76_CBUF_BASE_GRANULES + cbuf_f);
        /* AGAINST THE PITCH, not the plane. The CBUF allowance is consumed at the
         * DDR row ADVANCE and not at the fetch length: a task told a pitch and planned
         * for the tight row runs out of granules part way down the plane, and every
         * row past that reads whatever the DMA had left — measured departing at
         * exactly `ent_iw * ih / ent_pitch` rows, at two geometries.
         * [HW sweep, H96 MAX M9, tests/rk3576_conv_pitch.c map mode] */
        } else if (rocket_rk3576_cbuf_f_prec(p->in_pitch_w ? p->in_pitch_w : IW,
                                             IC, IH, OC, KH, KW, dw, prec,
                                             &cbuf_f) < 0) {
            return -1;
        }
    }

    {
        int rc = gen_conv2d_task_rk3576(p->tasks, &cna, &core, &dpu, dw, argb,
                                        ih_full, oh_full, pad_right, pad_bottom,
                                        cbuf_f, reuse, p->lut, p->in_surf_elems,
                                        p->in_pitch_w);
        if (rc < 0) return rc;
        r76_apply_overrides(p->tasks, rc);
        r76_dump_program(p->tasks, rc);
        p->task_count = (uint32_t)rc;
    }
    return 0;
}

/* ============================================================================
 * SECTION — the float weight cube
 *
 * The int8 weight cube groups both channel axes by 32 (weight_conv_int8). The float
 * one does NOT inherit that: measured on the part, its output-channel group is 8 and
 * its input-channel group is 16, so a kernel spans 16 fp16 (32 bytes) and the group
 * block holds 8 of them. Driving a kernel that ramps over ic and reading the sum back
 * is what pins it — at the int8 groups the part contracts the first 16 lanes twice
 * and pushes the second 16 into the next output channel, and at (8, 16) the sum is
 * exact. [HW sweep, H96 MAX M9]
 * ==========================================================================*/
int rocket_rk3576_weight_conv_fp16(unsigned oc_total, unsigned ic_total,
                                   unsigned kh_total, unsigned kw_total,
                                   unsigned oc, unsigned ic, unsigned kh, unsigned kw)
{
    unsigned oc1 = oc / R76_FP16_W_OC_GROUP, oc2 = oc % R76_FP16_W_OC_GROUP;
    unsigned ic1 = ic / R76_FP16_W_IC_GROUP, ic2 = ic % R76_FP16_W_IC_GROUP;
    unsigned nic1 = (ic_total + R76_FP16_W_IC_GROUP - 1u) / R76_FP16_W_IC_GROUP;
    (void)oc_total;
    if (kh >= kh_total || kw >= kw_total) return -1;
    return (int)((((((oc1 * nic1 + ic1) * kh_total) + kh) * kw_total + kw)
                  * R76_FP16_W_OC_GROUP + oc2) * R76_FP16_W_IC_GROUP + ic2);
}

/* ============================================================================
 * SECTION — the FIRST CONV's float weight cube
 *
 * Neither of the two above. The packed-image datapath contracts four lanes per pixel
 * whatever the image carries, so its cube has a LANE axis of four rather than an
 * input-channel axis, and the taps stay on their own axes — there is no 4*kw fold in
 * the cube, and no round-up of the lane group to sixteen.
 *
 * Decoded from manufactured captures whose weights carry a unique value per position
 * (tests/data/rk3576-vendor-capture/argb/mkargb.py), at ic 3 and 4, k 1/3/5/7 and
 * oc 16/32/48/64:
 *
 *   - a weight occupies a SIXTEEN-BIT slot;
 *   - four lanes per (output channel, tap), lane c carrying image channel c, and any
 *     lane past ic left don't-care;
 *   - output channels INTERLEAVED in groups of sixteen inside one tap — sixteen
 *     channels x four lanes is the 64 that separates one tap from the next;
 *   - the tap axis kh-outer, and one oc group's whole tap plane before the next.
 *
 * [source-confirmed, RKNN-Toolkit2 rk3576 float build]
 * ==========================================================================*/

/* 16-bit SLOT index of the weight for (output channel, image channel, tap), or -1 if
 * the position is out of range. Write a 16-bit element there. */
int rocket_rk3576_weight_argb_fp16(unsigned oc_total, unsigned ic_total,
                                   unsigned kh_total, unsigned kw_total,
                                   unsigned oc, unsigned ic, unsigned kh, unsigned kw)
{
    const unsigned G = R76_FP16_W_OC_GROUP;      /* 16 output channels per tap block */
    const unsigned tap = G * R76_ARGB_LANES;     /* 64 slots */
    if (oc >= oc_total || ic >= ic_total || ic >= R76_ARGB_LANES ||
        kh >= kh_total || kw >= kw_total)
        return -1;
    return (int)((oc / G) * (kh_total * kw_total * tap)
                 + kh * (kw_total * tap)
                 + kw * tap
                 + (oc % G) * R76_ARGB_LANES
                 + ic);
}

/* Bytes to allocate for that cube. Sized by the output-channel GROUPS, so an oc that
 * is not a multiple of sixteen still gets a whole trailing group — which is also what
 * the emitter's weight-byte register describes only when oc IS a multiple of sixteen,
 * and why a partial group is warned about rather than packed around. */
size_t rocket_rk3576_weight_argb_fp16_bytes(unsigned oc, unsigned kh, unsigned kw)
{
    unsigned groups = (oc + R76_FP16_W_OC_GROUP - 1u) / R76_FP16_W_OC_GROUP;
    return (size_t)groups * kh * kw * R76_FP16_W_OC_GROUP * R76_ARGB_LANES * 2u;
}

/* Pack row-major OIHW fp16 weights into it. `src` is oc*ic*kh*kw 16-bit elements with
 * ic the IMAGE's channel count (1..4); the lanes past it are zeroed rather than left
 * undefined, since nothing in the captures says the part ignores them. */
int rocket_rk3576_argb_fp16_pack_weights(void *dst, size_t dst_bytes,
                                         const void *src, unsigned oc, unsigned ic,
                                         unsigned kh, unsigned kw)
{
    const uint16_t *w = (const uint16_t *)src;
    uint16_t *out = (uint16_t *)dst;
    unsigned o, c, y, x;

    if (!dst || !src || !oc || !ic || ic > R76_ARGB_LANES || !kh || !kw) return -1;
    if (dst_bytes < rocket_rk3576_weight_argb_fp16_bytes(oc, kh, kw)) return -1;
    memset(dst, 0, dst_bytes);
    for (o = 0; o < oc; o++)
        for (c = 0; c < ic; c++)
            for (y = 0; y < kh; y++)
                for (x = 0; x < kw; x++) {
                    int s = rocket_rk3576_weight_argb_fp16(oc, ic, kh, kw, o, c, y, x);
                    if (s < 0) return -1;
                    out[s] = w[((o * ic + c) * kh + y) * kw + x];
                }
    return 0;
}

/* ============================================================================
 * SECTION — the first conv's INT8 weight cube
 *
 * Not the float one, and the difference is the whole reason a quantized stem did not
 * run: the float cube puts a weight in a 16-bit slot, groups output channels by
 * sixteen and carries the tap axis OUTSIDE that group; this one is single bytes,
 * groups output channels by THIRTY-TWO, and carries the tap ROW outside the group
 * with the tap COLUMN folded into the same 16-byte row as the lanes. Read off the
 * part with an impulse image and a one-byte cube (tests/rk3576_conv_gate.c fcmap),
 * which names the output channel, both taps and the lane of every live byte at once:
 *
 *   byte(oc, c, kh, kw) = (oc/32) * (KH * R * 32)
 *                       + kh * (32 * R)
 *                       + (oc%32) * R
 *                       + kw * 4
 *                       + c
 *
 *      R = round16(4 * KW)   the padded 16-byte kernel row the emitter declares
 *
 * Four lanes per (output channel, tap column) with lane c carrying image channel c
 * and the lanes past `ic` don't-care; 4*KW of each R-byte row live and the rest
 * padding the DMA still fetches. A bijection over every live byte at oc 32 and 64,
 * k 3/5/7 and ic 3 and 4 — 1152, 2304, 3200, 6272 and 864 live bytes, each landing on
 * exactly one output position of one channel, with none left over and none doubled.
 * The oc group of 32 is observable only above one group: at OC=32 a flat `oc*R` fits
 * equally, and OC=64 is what separates them. [HW sweep, H96 MAX M9]
 *
 * TWO GEOMETRY BOUNDS COME WITH IT, and neither is visible in any capture because
 * every captured first conv is a 3x3 stride-2 SAME convolution:
 *
 *   - THE LEFT PAD MUST BE NON-ZERO. With CNA_PAD_CON0's pad_left field at zero the
 *     DPU writes nothing at all — not a wrong surface, an untouched one — at every
 *     plane, stride, kernel and channel count tried. The field alone decides it:
 *     forcing 0x0100 into a zero-pad program makes the same program write, and
 *     forcing 0x0000 into a working one stops it. The pad_top field does neither.
 *   - THE OUTPUT WIDTH MUST BE iw/stride. A narrower one still writes a full surface,
 *     and it is SHEARED: the tap a byte lands on drifts one output column per output
 *     row, exactly as a row-stride mismatch does. Both bounds are what "SAME padding"
 *     means, which is why the vendor's own stems satisfy them and nothing else does.
 *
 * The output-channel count follows the direct path's rule — a multiple of 32, with a
 * partial group writing nothing (measured at oc=16) — but NOT the float first conv's
 * 32-channel per-program cap: one int8 program delivers 64 output channels.
 * [HW sweep, H96 MAX M9] */
int rocket_rk3576_weight_argb_int8(unsigned oc_total, unsigned ic_total,
                                   unsigned kh_total, unsigned kw_total,
                                   unsigned oc, unsigned ic, unsigned kh, unsigned kw)
{
    const unsigned G = 32u;                                  /* output-channel group */
    unsigned R = r76_argb_weight_row(kw_total);              /* padded kernel row     */
    if (oc >= oc_total || ic >= ic_total || ic >= R76_ARGB_LANES ||
        kh >= kh_total || kw >= kw_total)
        return -1;
    return (int)((oc / G) * (kh_total * R * G)
                 + kh * (G * R)
                 + (oc % G) * R
                 + kw * R76_ARGB_LANES
                 + ic);
}

/* Bytes to allocate. Sized by the output-channel GROUPS, so an oc that is not a
 * multiple of 32 still gets a whole trailing group — which is also the only oc the
 * part computes at, so the rounding is a safety net rather than a packing choice. */
size_t rocket_rk3576_weight_argb_int8_bytes(unsigned oc, unsigned kh, unsigned kw)
{
    unsigned groups = (oc + 31u) / 32u;
    return (size_t)groups * 32u * kh * r76_argb_weight_row(kw);
}

/* Pack row-major OIHW int8 weights into it. `src` is oc*ic*kh*kw bytes with ic the
 * IMAGE's channel count (1..4); the lanes past it and the kernel-row padding are
 * zeroed rather than left undefined — the DMA fetches the whole row, and a stale byte
 * in a lane the image does not carry is a weight on a channel that does not exist. */
int rocket_rk3576_argb_int8_pack_weights(void *dst, size_t dst_bytes,
                                         const void *src, unsigned oc, unsigned ic,
                                         unsigned kh, unsigned kw)
{
    const int8_t *w = (const int8_t *)src;
    int8_t *out = (int8_t *)dst;
    unsigned o, c, y, x;

    if (!dst || !src || !oc || !ic || ic > R76_ARGB_LANES || !kh || !kw) return -1;
    if (dst_bytes < rocket_rk3576_weight_argb_int8_bytes(oc, kh, kw)) return -1;
    memset(dst, 0, dst_bytes);
    for (o = 0; o < oc; o++)
        for (c = 0; c < ic; c++)
            for (y = 0; y < kh; y++)
                for (x = 0; x < kw; x++) {
                    int s = rocket_rk3576_weight_argb_int8(oc, ic, kh, kw, o, c, y, x);
                    if (s < 0) return -1;
                    out[s] = w[((o * ic + c) * kh + y) * kw + x];
                }
    return 0;
}

/* ============================================================================
 * SECTION — the depthwise weight cube
 *
 * DECODED, not inherited. The RK3588's depthwise cube is 64-channel groups of
 * single-byte weights (weight_conv_dw_int8); this part's is neither, and driving it
 * with the RK3588 layout is why depthwise never computed here. Read out of vendor
 * captures whose weights carry a unique value per (channel, tap), at C = 24, 32, 48,
 * 64 and 128 and at k = 3 and 5:
 *
 *   - A weight occupies a SIXTEEN-BIT SLOT, which is why the cube is
 *     round16(C)*kh*kw*2 bytes for both precisions of one geometry.
 *   - Channels group by THIRTY-TWO, and a group is a contiguous block.
 *   - Inside a group the layout is TAP-MAJOR: all of the group's channels for tap
 *     (0,0), then all of them for (0,1), and so on in kh-outer order.
 *   - A TRAILING PARTIAL GROUP IS DENSE. Its tap stride is the channels it actually
 *     holds, not 32 — C=48 packs a 32-group then a 16-group whose taps stride by 16,
 *     and the whole cube is 432 slots rather than 576. The buffer is still sized to
 *     the round16 count; the slots past the last channel are don't-care, and the
 *     vendor leaves whatever was in memory there.
 * [source-confirmed, RKNN-Toolkit2 rk3576 depthwise builds]
 *
 * Returns the 16-bit SLOT index, not a byte offset; the caller writes a 16-bit
 * element there. Returns -1 if the position is out of range. */
int rocket_rk3576_weight_dw(unsigned c_total, unsigned kh_total, unsigned kw_total,
                            unsigned c, unsigned kh, unsigned kw)
{
    unsigned g, base, held;
    if (c >= c_total || kh >= kh_total || kw >= kw_total) return -1;
    g    = c / R76_DW_W_GROUP;
    base = g * R76_DW_W_GROUP * kh_total * kw_total;
    held = c_total - g * R76_DW_W_GROUP;
    if (held > R76_DW_W_GROUP) held = R76_DW_W_GROUP;
    return (int)(base + (kh * kw_total + kw) * held + (c % R76_DW_W_GROUP));
}

/* The same cube at INT8, where a weight is one byte and the slot above is two.
 *
 * The tap-major, group-of-32 block structure is shared — it is the arrangement
 * INSIDE a tap block that differs, and it is not "the low byte of the float slot".
 * Read off the part by `rk3576_conv_gate dwmap`, which drives an impulse feature
 * against a cube that is zero but for one byte and reports which output that byte
 * reaches: over a C=32 k=3 cube every one of the 576 bytes reaches exactly one
 * (channel, tap), and the channel a byte carries is
 *
 *     channel(b) = 2*(b/4) + (b%2)
 *
 * so channel `c` holds two bytes per tap, at 4*(c/2) + (c%2) and two further on. Both
 * are live and both contribute, which is why the whole cube reads as live rather than
 * half of it. This entry point hands out the FIRST of the two and the packer leaves
 * the second at zero — a weight written into both is added twice.
 *
 * The float slot map is NOT this: laying an int8 weight into slot `c` as a 16-bit
 * value puts its sign extension into the byte belonging to channel c+1, which is a
 * neighbour's weight of 0 or -1 rather than padding. [HW sweep, H96 MAX M9]
 *
 * Returns a BYTE offset, or -1 if the position is out of range. */
int rocket_rk3576_weight_dw_int8(unsigned c_total, unsigned kh_total, unsigned kw_total,
                                 unsigned c, unsigned kh, unsigned kw)
{
    unsigned g, base, held, j;
    if (c >= c_total || kh >= kh_total || kw >= kw_total) return -1;
    g    = c / R76_DW_W_GROUP_INT8;
    base = g * R76_DW_W_GROUP_INT8 * kh_total * kw_total * 2u;
    /* A trailing partial group strides by the channels it holds ROUNDED UP TO 16,
     * which is what makes the whole cube exactly weight_dw_bytes(). The float cube's
     * partial group is dense in the raw count instead — the two precisions do not
     * share this. Measured at C = 24, 48 and 72, where the raw remainder and its
     * round16 differ. [HW sweep, H96 MAX M9] */
    held = r76_dw_cw(c_total - g * R76_DW_W_GROUP_INT8);
    if (held > R76_DW_W_GROUP_INT8) held = R76_DW_W_GROUP_INT8;
    j    = c % R76_DW_W_GROUP_INT8;
    return (int)(base + (kh * kw_total + kw) * held * 2u + 4u * (j / 2u) + (j % 2u));
}

/* Bytes to allocate for a depthwise weight cube: the round16 channel count, two
 * bytes per weight. Larger than the slots rocket_rk3576_weight_dw() hands out
 * whenever C is not a multiple of 16 — the tail is padding the DMA still fetches. */
size_t rocket_rk3576_weight_dw_bytes(unsigned c_total, unsigned kh, unsigned kw)
{
    return (size_t)r76_dw_cw(c_total) * kh * kw * 2u;
}

/* ============================================================================
 * SECTION — the ic split: an fp16 conv at an arbitrary input-channel count
 *
 * One fp16 task contracts exactly 16 input channels, because the DPU's output
 * element stride is 16/ic words and 16 is where an element lands in its own two
 * bytes. The split is what makes that a bound on a TASK rather than on the
 * convolution: cut ic into 16-channel slices, run each as its own conv, and sum the
 * partial surfaces.
 *
 * The feature side costs nothing. At C2 = 8 the cube's channel groups are contiguous
 * planes of iw*ih_full 16-byte atoms, so slice k is the same BO at a base offset and
 * nothing is repacked. Only the weight cube is per-slice, and it is not a sub-cube of
 * the whole conv's: each slice is its own convolution, so its group count follows the
 * slice (nic1 = 1) rather than the total.
 *
 * Accumulation is on the HOST. The DPU's eltwise stage is what the RK3588 uses for
 * the same job (ROCKET_KACC) and it would remove ic/16 readbacks. Nothing blocks it
 * any more — at this contraction width the partial it would read back is a plain,
 * dense, complete fp16 cube — so it is the open lever on this path.
 * See rockchip-npu-notes/chips/rk3576-regcmd.md.
 * ==========================================================================*/

int rocket_rk3576_plan_ic(const conv_params_t *p, rocket_rk3576_ic_task *out,
                          unsigned max_tasks, unsigned *count)
{
    unsigned ih_full, n, s, slice;

    if (!p || !out || !max_tasks || !count) return -1;
    if (!p->iw || !p->ic || !p->oc || !p->kh || !p->kw || !p->oh || !p->ow) return -1;

    ih_full = p->ih_full ? p->ih_full : p->ih;

    if (p->ic % ROCKET_RK3576_FP16_IC_SLICE) {
        ROCKET_LOGE("rk3576 ic split: ic=%u is not a multiple of the %u-channel slice. "
                    "Zero-pad the feature cube to %u channels and pass that count "
                    "(rocket_rk3576_fp16_pad_ic)\n",
                    p->ic, ROCKET_RK3576_FP16_IC_SLICE,
                    rocket_rk3576_fp16_pad_ic(p->ic));
        return -1;
    }
    if (p->oc % R76_FP16_W_OC_GROUP) {
        ROCKET_LOGE("rk3576 ic split: oc=%u is a partial %u-channel float group\n",
                    p->oc, R76_FP16_W_OC_GROUP);
        return -1;
    }

    /* The slice is FIXED, not the widest that fits: 16 is the one contraction width
     * whose output element lands in its own two bytes, so a wider slice would not be
     * a cheaper task but a wrong one. The ic axis and the ROW axis are separate
     * splits and this plans only the first, so a plane that overflows the CBUF even
     * at one slice is refused here with a pointer at the row planner. */
    slice = ROCKET_RK3576_FP16_IC_SLICE;
    if (rocket_rk3576_cbuf_f_prec(p->iw, slice, p->ih, p->oc,
                                  p->kh, p->kw, 0, precision_float16, NULL) < 0) {
        ROCKET_LOGE("rk3576 ic split: a single %u-channel slice of this %ux%u plane "
                    "still exceeds the CBUF allowance — window the rows too "
                    "(rocket_rk3576_plan_rows) and compose the two splits\n",
                    slice, p->iw, p->ih);
        return -1;
    }

    n = p->ic / slice;
    if (n > max_tasks) {
        ROCKET_LOGE("rk3576 ic split: ic=%u needs %u slices of %u, more than the %u "
                    "offered\n", p->ic, n, slice, max_tasks);
        return -1;
    }

    for (s = 0; s < n; s++) {
        out[s].ic0 = (uint16_t)(s * slice);
        out[s].ic  = (uint16_t)slice;
        /* A channel group is one contiguous plane of 16-byte atoms, and an atom stays
         * 16 bytes when C2 halves from 16 int8 lanes to 8 fp16 ones — so the stride is
         * element-size independent and the offset is a plain multiple of it. */
        out[s].feature_off = (uint32_t)s * slice / R76_FP16_FEAT_C2
                             * p->iw * ih_full * R76_C2_BYTES;
    }
    *count = n;
    return 0;
}

unsigned rocket_rk3576_fp16_pad_ic(unsigned ic)
{
    return ((ic + ROCKET_RK3576_FP16_IC_SLICE - 1u) / ROCKET_RK3576_FP16_IC_SLICE)
           * ROCKET_RK3576_FP16_IC_SLICE;
}

unsigned rocket_rk3576_fp16_pad_oc(unsigned oc)
{
    return ((oc + R76_FP16_W_OC_GROUP - 1u) / R76_FP16_W_OC_GROUP)
           * R76_FP16_W_OC_GROUP;
}

size_t rocket_rk3576_fp16_slice_weight_bytes(unsigned oc, unsigned ic,
                                             unsigned kh, unsigned kw)
{
    /* Sized by the GROUPS, not by the slice's channel count: a partial input-channel
     * group still occupies a whole one, so an 8-channel slice at an ic group of 16
     * needs a cube for 16. Sizing from the slice under-allocates, and at k=1 the
     * overrun stays inside the BO's page and computes correctly anyway — it surfaces
     * only at a kernel large enough to run past the page, where it reads as "this
     * part cannot do k=3". */
    unsigned groups = (ic + R76_FP16_W_IC_GROUP - 1u) / R76_FP16_W_IC_GROUP;
    if (!groups) groups = 1u;
    return (size_t)rocket_rk3576_fp16_pad_oc(oc) * groups * R76_FP16_W_IC_GROUP
           * kh * kw * sizeof(_Float16);
}

int rocket_rk3576_fp16_pack_slice_weights(void *dst, size_t dst_bytes,
                                          const _Float16 *w_oihw,
                                          unsigned oc, unsigned ic_total,
                                          unsigned kh, unsigned kw,
                                          const rocket_rk3576_ic_task *t)
{
    size_t need = rocket_rk3576_fp16_slice_weight_bytes(oc, t->ic, kh, kw);
    _Float16 *cube = (_Float16 *)dst;
    unsigned o, i, y, x;

    if (!dst || !w_oihw || !t || !oc || !ic_total || !kh || !kw) return -1;
    if (dst_bytes < need) {
        ROCKET_LOGE("rk3576 ic split: a slice weight cube for oc=%u k=%ux%u needs %zu "
                    "bytes, %zu offered\n", oc, kh, kw, need, dst_bytes);
        return -1;
    }
    if ((unsigned)t->ic0 + t->ic > ic_total) return -1;

    memset(cube, 0, need);
    for (o = 0; o < oc; o++)
        for (i = 0; i < t->ic; i++)
            for (y = 0; y < kh; y++)
                for (x = 0; x < kw; x++) {
                    /* ic_total for the INDEX is the slice's own channel count — each
                     * slice is its own convolution, so nic1 follows the slice. */
                    int idx = rocket_rk3576_weight_conv_fp16(oc, t->ic, kh, kw,
                                                             o, i, y, x);
                    if (idx < 0) return -1;
                    cube[idx] = w_oihw[(((size_t)o * ic_total + t->ic0 + i) * kh + y)
                                       * kw + x];
                }
    return 0;
}

/* ---- reading the partial surface back ---- */

unsigned rocket_rk3576_fp16_out_channels(unsigned oc)
{
    return oc;
}

size_t rocket_rk3576_fp16_out_bytes(unsigned oc, unsigned oh, unsigned ow)
{
    return (size_t)rocket_rk3576_fp16_pad_oc(oc) * oh * ow * sizeof(_Float16);
}

/* The plain native float cube: 8 channels to a 16-byte atom, one atom per pixel,
 * channel groups as contiguous planes. Every programmed channel is present exactly
 * once, which is true at the 16-channel contraction the emitter runs and at no other
 * — see ROCKET_RK3576_FP16_IC_SLICE. [HW sweep, H96 MAX M9] */
int rocket_rk3576_fp16_out_index(unsigned oh, unsigned ow, unsigned c,
                                 unsigned y, unsigned x)
{
    if (y >= oh || x >= ow) return -1;
    return (int)((ow * oh * R76_FP16_FEAT_C2) * (c / R76_FP16_FEAT_C2)
                 + R76_FP16_FEAT_C2 * (y * ow + x) + (c % R76_FP16_FEAT_C2));
}

int rocket_rk3576_fp16_accumulate(float *acc, const void *surface,
                                  size_t surface_bytes,
                                  unsigned oc, unsigned oh, unsigned ow)
{
    const _Float16 *s = (const _Float16 *)surface;
    unsigned n = rocket_rk3576_fp16_out_channels(oc), c, y, x;

    if (!acc || !surface || !oc || !oh || !ow) return -1;
    if (surface_bytes < rocket_rk3576_fp16_out_bytes(oc, oh, ow)) return -1;

    for (c = 0; c < n; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                int idx = rocket_rk3576_fp16_out_index(oh, ow, c, y, x);
                if (idx < 0) return -1;
                acc[((size_t)c * oh + y) * ow + x] += (float)s[idx];
            }
    return 0;
}

/* No reuse: every task refetches its whole window against a CBUF base of zero. */
static const struct r76_reuse R76_NO_REUSE = { 0, 0, 0 };

int gen_conv2d_int8_rk3576(conv_params_t *p)
{
    return gen_conv2d_rk3576_fill(p, 0, precision_int8, precision_int8,
                                 precision_int8, &R76_NO_REUSE);
}

int gen_conv2d_dw_int8_rk3576(conv_params_t *p)
{
    return gen_conv2d_rk3576_fill(p, 1, precision_int8, precision_int8,
                                 precision_int8, &R76_NO_REUSE);
}

/* The same int8 direct convolution, writing the DPU's RAW 32-bit accumulator instead
 * of the requantized int8 byte. Only two things change and neither is geometry:
 * DPU 0x4010's output WIDTH field takes precision_int32, and the OUT_CVT is pinned to
 * exact unity. The surface's BYTE layout is unchanged — the int8 writer puts 16 lanes
 * in a 16-byte atom and this one puts 4, so a channel group occupies the same bytes
 * and every stride register already holds the right value. What changes for the
 * caller is only the output BO, which is four times the size, and the de-scatter,
 * which reads 32-bit words: (n/4)*ow*oh_full*4 + 4*(y*ow + x) + (n%4).
 * [HW sweep, H96 MAX M9 — decoded, not assumed: every accumulator was made distinct
 * and each one names exactly one word.] */
int gen_conv2d_int8_rk3576_i32out(conv_params_t *p)
{
    return gen_conv2d_rk3576_fill(p, 0, precision_int8, precision_int32,
                                 precision_int8, &R76_NO_REUSE);
}

/* The same raw-int32 convolution with the writer's byte budget DOUBLED.
 *
 * The DPU's write budget is one 16-byte atom per (16-channel block, pixel) whatever the
 * output element width is, which is what leaves the narrow writer above delivering the
 * first eight output channels of every thirty-two. That budget is a function of the
 * DPU's own PROC_PRECISION, not of the output width: driving 0x4010[2:0] from int8 to
 * any two-byte-or-wider operand precision makes it TWO atoms per (block, pixel), and
 * the delivered set becomes the first eight of every SIXTEEN. Nothing else moves — the
 * operands stay int8 everywhere, the arithmetic is bit-identical, and the CNA and CORE
 * blocks are byte-unchanged.
 *
 * Two atoms is the ceiling. Swept across all eight values of the field: int8 and int4
 * write one atom, int16 / fp16 / bfloat16 / int32 write two, and the float precisions
 * additionally reinterpret the operands so the arithmetic is destroyed. int32 is used
 * here because it is what the output width already says. [HW sweep, H96 MAX M9]
 *
 * WHAT MOVES IS THE LAYOUT. The narrow writer's delivered words land on the plain
 * RK3588 int32 cube; these do not. Use rocket_rk3576_i32_wide_word() to address the
 * surface, and size the BO for twice the narrow writer's extent. */
int gen_conv2d_int8_rk3576_i32out_wide(conv_params_t *p)
{
    return gen_conv2d_rk3576_fill(p, 0, precision_int8, precision_int32,
                                 precision_int32, &R76_NO_REUSE);
}

/* Where gen_conv2d_int8_rk3576_i32out_wide() puts output channel `c` at pixel `p`, as a
 * 32-bit word index into the output BO.
 *
 * Decoded on the part rather than fitted: the operands were drawn so that every
 * accumulator in the tile is distinct, so each written word names exactly one (channel,
 * pixel) or the map is not a map. Read off at oc 8/16/24/32/40/48/64/96 and at pixel
 * counts 4/5/6/7/8/12/16. [HW sweep, H96 MAX M9]
 *
 * The shape of it. Work in 32-channel SUPER-GROUPS, each 4*A atoms long, where A is the
 * surface's pixel count ow*oh_full. Inside one super-group the writer emits a single
 * linear STREAM indexed by
 *
 *     s = 2*p + j        j = the 16-channel block, 0 or 1
 *
 * and cuts that stream into runs of A atoms. The run a slot lands in also carries the
 * lane group L = (c%16)/4, which is the slower axis:
 *
 *     atom = 4*A*(c/32) + A*(2*(s/A) + L) + s%A
 *
 * The stream is what makes the plane's parity a non-event: nothing rounds, so an odd A
 * simply cuts the same stream at an odd place, and A = 1 works as readily as A = 16.
 *
 * TWO BOUNDS, and both fail silently rather than loudly:
 *   - `oc` must be a multiple of 32, because a partial super-group is not a truncated
 *     one. At oc = 16 (mod 32) the trailing group holds one 16-channel block instead of
 *     two and packs at one atom per pixel with no stream; at an oc that is not a
 *     multiple of 16 at all — 24 was the case read off — the delivered channel set
 *     ROTATES with the pixel and there is no block form to describe. Pad with
 *     rocket_rk3576_pad_oc() and leave the padding zero.
 *   - only channels with `c % 16 < 8` are written. Scatter real channels into those
 *     slots; everything else in the weight cube stays zero.
 * Returns -1 if `c` is not a delivered channel. */
int rocket_rk3576_i32_wide_word(unsigned ow, unsigned oh_full, unsigned c, unsigned p)
{
    unsigned A = rocket_rk3576_out_surf_elems(ow, oh_full, 0);
    unsigned s, atom;

    if ((c % 16u) >= 8u || !A || p >= A) return -1;
    s = 2u * p + ((c % 32u) / 16u);
    atom = 4u * A * (c / 32u)
         + A * (2u * (s / A) + (c % 16u) / 4u)
         + (s % A);
    return (int)(4u * atom + (c % 4u));
}

int gen_conv2d_int8_rk3576_reuse(conv_params_t *p, int dw,
                                 const rocket_rk3576_row_task *t, unsigned index)
{
    struct r76_reuse r;
    if (!t) return -1;
    if (index && t->retained >= t->ih) {
        ROCKET_LOGE("rk3576 reuse: task %u retains %u of its %u window rows — nothing "
                    "left to fetch\n", index, t->retained, t->ih);
        return -1;
    }
    /* The window base is the resident count minus the retained rows, so a resident
     * count that does not cover them is not a smaller base — it is a wrapped one,
     * and the CNA would then read rows from the far end of the CBUF. */
    if (index && t->retained) {
        unsigned entries = p->ic <= R76_ARGB_LANES ? r76_argb_entries(p->iw, 1u, 0)
                                                   : r76_data_entries(p->iw, p->ic);
        if ((unsigned)t->retained * entries > t->cbuf_resident) {
            ROCKET_LOGE("rk3576 reuse: task %u retains %u rows (%u granules) against a "
                        "resident count of %u — the window base would wrap\n",
                        index, t->retained, t->retained * entries, t->cbuf_resident);
            return -1;
        }
    }
    r.continuation  = index ? 1u : 0u;
    r.retained_rows = index ? t->retained : 0u;
    r.resident      = index ? t->cbuf_resident : 0u;
    return gen_conv2d_rk3576_fill(p, dw, precision_int8, precision_int8,
                                 precision_int8, &r);
}

/* ONE fp16 task, which is exactly 16 input channels.
 *
 * The bound is the feature surface index: the datapath takes ic/8 contraction steps
 * and the surface pointer advances at the int8 rate of one per 16 channels, so a
 * second step re-reads surface 0 and the channels past 8 are contracted against the
 * wrong lanes. That produces a full, correctly sized, WRONG surface with nothing to
 * fault on, which is why a larger ic is REFUSED here rather than warned about — a
 * caller must not be able to reach it silently.
 *
 * Anything past 8 goes through rocket_rk3576_plan_ic(), which is the same convolution
 * as a sequence of slices. gen_conv2d_rk3576_prec() is the unchecked bring-up entry
 * and emits whatever it is asked for; use that, not this, to sweep the defect itself.
 * [HW sweep, H96 MAX M9] */
int gen_conv2d_fp16_rk3576(conv_params_t *p)
{
    if (!p) return -1;
    /* The packed-image first conv is a different CNA program, not this one at a small
     * channel count: it contracts four lanes per pixel in one task, so the 16-channel
     * contraction bound below does not apply to it. gen_conv2d_rk3576_fill() selects
     * it from the same ic <= 4 test the int8 entry uses. */
    if (p->ic <= 4u)
        return gen_conv2d_rk3576_fill(p, 0, precision_float16, precision_float16,
                                      precision_float16, &R76_NO_REUSE);
    if (p->ic != ROCKET_RK3576_FP16_IC_SLICE) {
        ROCKET_LOGE("rk3576 fp16 conv: one task contracts exactly %u input channels "
                    "and ic=%u was asked for. The DPU's output element stride is 16/ic "
                    "words, so any other count writes a full, correctly sized, WRONG "
                    "surface with nothing to fault on. Split ic with "
                    "rocket_rk3576_plan_ic()\n",
                    ROCKET_RK3576_FP16_IC_SLICE, p->ic);
        return -1;
    }
    return gen_conv2d_rk3576_fill(p, 0, precision_float16, precision_float16,
                                 precision_float16, &R76_NO_REUSE);
}

/* The unchecked entry: a program at any precision and any geometry, for bring-up
 * sweeps and for the register-fidelity gate. It does NOT apply the fp16 envelope
 * above, so an fp16 conv emitted through here past ic=8 computes wrong silently. */
int gen_conv2d_rk3576_prec(conv_params_t *p, int dw, unsigned prec)
{
    return gen_conv2d_rk3576_fill(p, dw, prec, prec, prec, &R76_NO_REUSE);
}

/*
 * The fp32 conv scale -> the OUT_CVT (MUL, SHIFT) pair, the vendor's (QNNPACK)
 * derivation. `shift` comes back as the REGISTER value, already pre-decremented, so a
 * CPU model shifts by exactly what the DPU has. One copy, because a caller that has to
 * know the gain the emitter will actually program — a per-channel requant sizing its C
 * multipliers against it — must not re-derive it slightly differently.
 */
void rocket_rk3576_requant_params(float conv_scale, unsigned *mul, unsigned *shift)
{
    union { float f; uint32_t u; } cv;
    uint32_t bits;
    unsigned m;
    cv.f = conv_scale;
    bits = cv.u;
    *shift = 127u + 31u - 32u - (bits >> 23) + 16u - 1u;   /* == 125 - exp + 16 */
    m = ((bits >> 9) & 0x7FFFu) + 1u;
    if (m < (1u << 14)) m |= (1u << 14);
    *mul = m;
}

/* ============================================================================
 * SECTION — the coefficient buffer
 *
 * The A/B/C group layout the RK3576 reads at BS_BASE_ADDR. See the header for the
 * structure and for why C is emitted as 1 rather than left at zero.
 * ==========================================================================*/

/* The A/B/C groups are followed by one more 64-byte group, left at zero, which is
 * what BS_BASE_ADDR1 (0x5024) points at — the DPU shift word. See r76_shift_offset. */
/* Elements per OUTPUT SURFACE in DDR — what separates one 16-channel group from the
 * next. It is the plane on the direct path and the plane ROUNDED UP TO FOUR on the
 * depthwise one, which is the register the emitter puts at 0x401C.
 *
 * A caller has to know this, because it is the difference between an output cube whose
 * groups line up and one where every group past the first lands four elements early.
 * It is invisible at any plane whose ow*oh_full is already a multiple of four, which
 * is most of them: it shows at 15x18, 17x19 and 19x19 and hides at 16x16. Size the
 * output BO with it and de-scatter with it. [source-confirmed + HW sweep, H96 MAX M9]
 */
unsigned rocket_rk3576_out_surf_elems(unsigned ow, unsigned oh_full, int dw)
{
    unsigned e = ow * oh_full;
    return dw ? ((e + 3u) & ~3u) : e;
}

unsigned rocket_rk3576_pad_ic(unsigned ic)
{
    return ((ic + 31u) / 32u) * 32u;
}

unsigned rocket_rk3576_pad_oc(unsigned oc)
{
    return ((oc + 31u) / 32u) * 32u;
}

/* The unit C multiplier for a precision. The BS stage reads the field in the program's
 * own precision, so the integer 1 that means "no scaling" on the int8 path is the
 * denormal 6e-8 on a float one and empties the surface exactly as C=0 does — with no
 * fault either way. [HW sweep, H96 MAX M9] */
#define R76_COEFF_C_UNIT_FP16  ((int16_t)0x3C00)

size_t rocket_rk3576_coeff_bytes(unsigned oc)
{
    return r76_shift_offset(oc) + R76_COEFF_GROUP_BYTES;
}

int rocket_rk3576_pack_coeff(void *dst, size_t dst_bytes, const int32_t *bias, unsigned oc)
{
    return rocket_rk3576_pack_coeff_asym(dst, dst_bytes, bias, oc, NULL, 1);
}

/* ---------------------------------------------------------------------------
 * The DEPTHWISE coefficient group: 48 bytes for 8 output channels, A (int32) at
 * (oc%8)*4 and C (int16) at 32 + (oc%8)*2, with no B field between them.
 *
 * DECODED, not inherited. rk3576_conv_gate dwcoeff bumps one int16 of the buffer at a
 * time against a measured baseline and reports which raw output bytes move; on the
 * depthwise path every one of the fifty moves it finds, and every position that does
 * NOT move, is what this layout predicts and what the 64-byte one does not. With it
 * the raw output cube is the plain native one — channel c at lane c%16 of plane c/16,
 * every programmed channel present exactly once — where the 64-byte group left four
 * lanes of one plane carrying another group's biases and the rest at zero or the
 * clip. [HW sweep, H96 MAX M9]
 *
 * There is nowhere to put a weight zero point here, so there is no asymmetric form:
 * an asymmetric depthwise weight is carried in the CUBE instead, whose two live bytes
 * per (channel, tap) the datapath adds.
 *
 * C is per output channel here exactly as it is in the 64-byte group, so a per-axis
 * weight quantization is expressible on this path too — rocket_rk3576_pack_coeff_dw_perc()
 * is that form and the two entries below are it with a single multiplier.
 * ------------------------------------------------------------------------- */
size_t rocket_rk3576_coeff_bytes_dw(unsigned oc)
{
    return r76_shift_offset_p(oc, 1) + R76_COEFF_GROUP_BYTES_DW;
}

int rocket_rk3576_pack_coeff_dw_perc(void *dst, size_t dst_bytes, const int32_t *bias,
                                     unsigned oc, const int16_t *c_term,
                                     int16_t multiplier)
{
    uint8_t *b = (uint8_t *)dst;
    unsigned c;

    if (!dst || dst_bytes < rocket_rk3576_coeff_bytes_dw(oc)) {
        ROCKET_LOGE("rk3576 dw coeff: buffer is %zu bytes, %u output channels need %zu\n",
                    dst_bytes, oc, rocket_rk3576_coeff_bytes_dw(oc));
        return -1;
    }
    if (!c_term && multiplier == 0) {
        ROCKET_LOGE("rk3576 dw coeff: multiplier 0 gates the BS stage off — the DPU "
                    "would write a full but entirely empty surface\n");
        return -1;
    }
    memset(b, 0, rocket_rk3576_coeff_bytes_dw(oc));
    for (c = 0; c < oc; c++) {
        uint8_t *g = b + (size_t)(c / R76_COEFF_GROUP_OC) * R76_COEFF_GROUP_BYTES_DW;
        int32_t a = bias ? bias[c] : 0;
        int16_t mul = c_term ? c_term[c] : multiplier;
        if (mul == 0) {
            ROCKET_LOGE("rk3576 dw coeff: C[%u] is 0 — that gates the BS stage off for "
                        "the whole 8-channel group and the DPU writes an empty surface "
                        "with no fault\n", c);
            return -1;
        }
        memcpy(g + (c % R76_COEFF_GROUP_OC) * 4, &a, sizeof a);
        memcpy(g + R76_COEFF_C_OFFSET_DW + (c % R76_COEFF_GROUP_OC) * 2,
               &mul, sizeof mul);
    }
    return 0;
}

int rocket_rk3576_pack_coeff_dw_prec(void *dst, size_t dst_bytes, const int32_t *bias,
                                     unsigned oc, unsigned prec)
{
    int16_t unit = (prec == precision_float16 || prec == precision_bfloat16)
                   ? R76_COEFF_C_UNIT_FP16 : (int16_t)1;

    return rocket_rk3576_pack_coeff_dw_perc(dst, dst_bytes, bias, oc, NULL, unit);
}

int rocket_rk3576_pack_coeff_dw(void *dst, size_t dst_bytes, const int32_t *bias,
                                unsigned oc)
{
    return rocket_rk3576_pack_coeff_dw_prec(dst, dst_bytes, bias, oc, precision_int8);
}

int rocket_rk3576_pack_coeff_prec(void *dst, size_t dst_bytes, const int32_t *bias,
                                  unsigned oc, unsigned prec)
{
    int16_t unit = (prec == precision_float16 || prec == precision_bfloat16)
                   ? R76_COEFF_C_UNIT_FP16 : (int16_t)1;
    return rocket_rk3576_pack_coeff_asym(dst, dst_bytes, bias, oc, NULL, unit);
}

/*
 * B is the WEIGHT ZERO POINT and C the per-channel multiplier. Both are int16 and
 * both are per output channel, so an asymmetric weight quantization is expressible
 * here without touching the register program — which is what makes uint8 weights a
 * packing question rather than an encoding one.
 *
 * UNVALIDATED: every capture carries B = 0, because every captured model quantizes
 * its weights symmetrically. That B is where the weight zero point goes is read off
 * the field's position and width in the group, not off a program that uses it, and
 * the SIGN CONVENTION is unknown — the accumulator correction is either -B*sum(x) or
 * +B*sum(x) and nothing here says which. A wrong sign is a bias-shaped error that
 * looks like a plausible surface, so drive it against a CPU model before believing
 * it, at a weight zero point far from zero where the two signs cannot be confused.
 *
 * C defaults to 1 and must never be left at 0: a zero multiplier gates the whole BS
 * stage for that group and the DPU writes a full, correctly sized, entirely EMPTY
 * surface with no fault to catch it.
 *
 * HOW C IS READ DEPENDS ON THE PRECISION. On a FLOAT program the field is read as fp16,
 * where the integer 1 is the denormal 6e-8: it underflows the surface to empty, the same
 * signature C=0 gives on the integer path and just as silent. fp16 1.0 is 0x3C00.
 * rocket_rk3576_pack_coeff_prec() takes the precision and picks the unit multiplier, so
 * only a caller driving a non-unit C needs this entry point.
 * [HW sweep, H96 MAX M9, measured 2026-07-26]
 */
int rocket_rk3576_pack_coeff_asym(void *dst, size_t dst_bytes, const int32_t *bias,
                                  unsigned oc, const int16_t *b_term,
                                  int16_t multiplier)
{
    return rocket_rk3576_pack_coeff_perc(dst, dst_bytes, bias, oc, b_term,
                                         NULL, multiplier);
}

/*
 * The per-output-channel form. `c_term` gives every channel its own multiplier and
 * `multiplier` is the value used where it is NULL, so the scalar entries above are
 * this with `c_term = NULL`. See the header for what a per-channel C can and cannot
 * express.
 */
int rocket_rk3576_pack_coeff_perc(void *dst, size_t dst_bytes, const int32_t *bias,
                                  unsigned oc, const int16_t *b_term,
                                  const int16_t *c_term, int16_t multiplier)
{
    uint8_t *b = (uint8_t *)dst;
    unsigned c;

    if (!dst || dst_bytes < rocket_rk3576_coeff_bytes(oc)) {
        ROCKET_LOGE("rk3576 coeff: buffer is %zu bytes, %u output channels need %zu\n",
                    dst_bytes, oc, rocket_rk3576_coeff_bytes(oc));
        return -1;
    }
    if (!c_term && multiplier == 0) {
        ROCKET_LOGE("rk3576 coeff: multiplier 0 gates the BS stage off — the DPU "
                    "would write a full but entirely empty surface\n");
        return -1;
    }
    memset(b, 0, rocket_rk3576_coeff_bytes(oc));
    for (c = 0; c < oc; c++) {
        uint8_t *g = b + (size_t)(c / R76_COEFF_GROUP_OC) * R76_COEFF_GROUP_BYTES;
        int32_t a = bias ? bias[c] : 0;
        /* Straight into the field: the DPU ADDS B*sum(x), so a weight zero point
         * reaches here already negated. See the header. */
        int16_t wzp = b_term ? b_term[c] : 0;
        int16_t mul = c_term ? c_term[c] : multiplier;
        if (mul == 0) {
            ROCKET_LOGE("rk3576 coeff: C[%u] is 0 — that gates the BS stage off for "
                        "the whole 8-channel group and the DPU writes an empty "
                        "surface with no fault\n", c);
            return -1;
        }
        memcpy(g + (c % R76_COEFF_GROUP_OC) * 4, &a, sizeof a);
        memcpy(g + R76_COEFF_B_OFFSET + (c % R76_COEFF_GROUP_OC) * 2, &wzp, sizeof wzp);
        memcpy(g + R76_COEFF_C_OFFSET + (c % R76_COEFF_GROUP_OC) * 2,
               &mul, sizeof mul);
    }
    return 0;
}

/* ============================================================================
 * SECTION — the PPU: pooling
 *
 * Pooling is its own NPU program on this part and not an epilogue: 23 PPU writes and
 * 8 PPU_RDMA, no CNA, no CORE, no DPU. It reads and writes the SAME NC1HWC2 cube the
 * convolution path already packs, so it needs no new host layout.
 *
 * Every register below is read off manufactured vendor captures (an ONNX MaxPool or
 * AveragePool compiled for rk3576, `tests/data/rk3576-vendor-capture/pool/`) swept over
 * method, kernel, stride, plane, channel count and padding — 23 shapes, non-square and
 * odd planes included, which is what separates the two extents below.
 *
 * TWO TRAPS RIDE WITH IT.
 *
 *   THE INPUT EXTENT IS WHAT THE WINDOWS CONSUME, NOT THE PLANE. `0x600C`/`0x6010` and
 *   their PPU_RDMA twins carry `(ow-1)*sx + kw` and `(oh-1)*sy + kh` clamped to the
 *   plane, so a 19-wide plane pooled k2 s2 programs 18 and a row no window reaches is
 *   simply not described. A square capture cannot separate this from the plane; the
 *   non-square ones can, and do.
 *
 *   THE STRIDES ARE THE PLANE'S, and they are the only place the full plane appears.
 *   `0x7024` is the DDR line stride in bytes (`iw*16`) and `0x7028` the channel-group
 *   surface stride (`round4(iw*ih)*16`) — the same round-to-four the convolution's
 *   `0x401C` takes, invisible at every plane whose area is already a multiple of four.
 *
 * The destination base address is NOT in any capture: like every other base on this
 * part it is patched by the vendor runtime at load time and the stored program carries
 * zero. `ppu_dst_reg` is which register receives it, so the sweep that reads it off the
 * part can drive the candidates without a second emitter.
 * ==========================================================================*/

#define OP_REG_PPU      (BLOCK_PPU | PC_OP_01)        /* 0x4001 */
#define OP_REG_PPU_RDMA (BLOCK_PPU_RDMA | PC_OP_01)   /* 0x8001 */

#define R76_PPU_S_POINTER   0x6004
#define R76_PPU_IN_WIDTH    0x600C
#define R76_PPU_IN_HEIGHT   0x6010
#define R76_PPU_IN_CHANNEL  0x6014
#define R76_PPU_OUT_WIDTH   0x6018
#define R76_PPU_OUT_HEIGHT  0x601C
#define R76_PPU_OUT_CHANNEL 0x6020
#define R76_PPU_MODE        0x6024
#define R76_PPU_KERNEL      0x6034
#define R76_PPU_RECIP_W     0x6038
#define R76_PPU_RECIP_H     0x603C
#define R76_PPU_PAD_CFG     0x6040
#define R76_PPU_PAD_VAL0    0x6044
#define R76_PPU_PAD_VAL1    0x6048
#define R76_PPU_PAD_VAL2    0x604C
#define R76_PPU_PAD_VAL3    0x6050
#define R76_PPU_ZERO_6054   0x6054
#define R76_PPU_ZERO_6058   0x6058
#define R76_PPU_ZERO_605C   0x605C
#define R76_PPU_ZERO_6070   0x6070
#define R76_PPU_DST_SURF0   0x607C
#define R76_PPU_DST_SURF1   0x6084
#define R76_PPU_ZERO_60DC   0x60DC

#define R76_PPUR_S_POINTER  0x7004
#define R76_PPUR_IN_WIDTH   0x700C
#define R76_PPUR_IN_HEIGHT  0x7010
#define R76_PPUR_IN_CHANNEL 0x7014
#define R76_PPUR_SRC_BASE   0x701C
#define R76_PPUR_SRC_LINE   0x7024
#define R76_PPUR_SRC_SURF   0x7028
#define R76_PPUR_CONST_7030 0x7030   /* 0x40 in every capture */

/* PC_OPERATION_ENABLE is a per-block bitmap. The conv path's 0x1D and this 0x60 are
 * disjoint, so a pool carrying the conv's word starts nothing. */
#define R76_PPU_ENABLE_WORD     0x60

#define R76_PPU_MODE_MAX        0x11
#define R76_PPU_MODE_AVG        0x10
#define R76_PPU_MODE_AVG_NOPAD  0x18
/* -128 in the 19-bit sign-extended pad field the max path uses. */
#define R76_PPU_PAD_MIN         0x0007ff80u

static uint32_t r76_round4(uint32_t v) { return (v + 3u) & ~3u; }

int gen_pool_rk3576(pool_params_rk3576_t *p)
{
    uint64_t *ops = p ? p->tasks : NULL;
    unsigned iw, ih, c, ow, oh, in_w, in_h, creg, pitch, dst_surf;
    unsigned mode, dst_reg;
    int i = 0;

    if (!p || !ops) return -1;
    iw = p->iw; ih = p->ih; c = p->c;
    ow = p->ow; oh = p->oh;
    if (!iw || !ih || !c || !ow || !oh || !p->kw || !p->kh ||
        !p->stride_x || !p->stride_y) {
        ROCKET_LOGE("gen_pool_rk3576: a zero extent (%ux%u c%u k%ux%u s%ux%u -> %ux%u)\n",
                    iw, ih, c, p->kw, p->kh, p->stride_x, p->stride_y, ow, oh);
        return -1;
    }
    if (p->kw > 16u || p->kh > 16u || p->stride_x > 16u || p->stride_y > 16u) {
        ROCKET_LOGE("gen_pool_rk3576: the kernel and stride fields are four bits — "
                    "k%ux%u s%ux%u does not fit; cascade instead\n",
                    p->kw, p->kh, p->stride_x, p->stride_y);
        return -1;
    }
    switch (p->mode) {
    case ROCKET_RK3576_POOL_MAX:       mode = R76_PPU_MODE_MAX; break;
    case ROCKET_RK3576_POOL_AVG:       mode = R76_PPU_MODE_AVG; break;
    case ROCKET_RK3576_POOL_AVG_NOPAD: mode = R76_PPU_MODE_AVG_NOPAD; break;
    default:
        ROCKET_LOGE("gen_pool_rk3576: mode %u is not one of the three the captures "
                    "carry\n", p->mode);
        return -1;
    }

    /* What the windows consume FROM THE REAL PLANE — not the plane, and not the padded
     * plane either. The last window may hang off both ends; the pad it reads there is
     * synthesised and is not described here, so the leading pad comes off the span and
     * the result is clamped to the plane. Six captures pin it, including the two that
     * separate it from every simpler reading: a VALID k3 s2 over 16 programs 15, and a
     * SAME k3 s2 over the same 16 programs 16 rather than the 17 its windows span. */
    in_w = (ow - 1u) * p->stride_x + p->kw;
    in_h = (oh - 1u) * p->stride_y + p->kh;
    in_w = in_w > p->pad_left ? in_w - p->pad_left : 1u;
    in_h = in_h > p->pad_top  ? in_h - p->pad_top  : 1u;
    if (in_w > iw) in_w = iw;
    if (in_h > ih) in_h = ih;

    /* THE ROW PITCH IS NOT THE PLANE, and clamping the consumed extent against the
     * wrong one of the two is silent. `iw` above is the real plane, so a window that
     * hangs off its right edge reads a SYNTHESISED pad; a pitch passed in `iw` instead
     * would lift that clamp and the same window would read whatever the producer's
     * surplus columns hold. Two of four geometries in `tests/rk3576_row_pitch.c` differ
     * on exactly that and only that. */
    pitch = p->src_line_elems ? p->src_line_elems : iw;
    if (pitch < iw) {
        ROCKET_LOGE("gen_pool_rk3576: a row pitch of %u is shorter than the %u-wide "
                    "plane it holds\n", pitch, iw);
        return -1;
    }

    /* THE DESTINATION SURFACE STRIDE IS A REGISTER, and it takes the plane exactly. The
     * vendor's pooling programs all carry round4(ow*oh), which is what a cube this
     * emitter allocates for itself uses; a caller writing a SLICE of somebody else's
     * buffer passes that buffer's group stride instead, which for a shared concatenation
     * is `ow*oh` — a convolution's own surface stride. Below the plane the groups
     * overlap and atoms are lost. */
    dst_surf = p->dst_surf_elems ? p->dst_surf_elems : r76_round4(ow * oh);
    if (dst_surf < ow * oh) {
        ROCKET_LOGE("gen_pool_rk3576: a destination surface stride of %u is shorter than "
                    "the %ux%u plane it holds; the channel groups would overlap\n",
                    dst_surf, ow, oh);
        return -1;
    }

    creg = ((c + 15u) / 16u) * 16u;

    dst_reg = p->ppu_dst_reg ? p->ppu_dst_reg : R76_PPU_ZERO_6070;

    ops[i++] = NPUOP(OP_REG_PPU, 0xE, R76_PPU_S_POINTER);
    ops[i++] = NPUOP(OP_REG_PPU, in_w - 1u, R76_PPU_IN_WIDTH);
    ops[i++] = NPUOP(OP_REG_PPU, in_h - 1u, R76_PPU_IN_HEIGHT);
    ops[i++] = NPUOP(OP_REG_PPU, creg - 1u,  R76_PPU_IN_CHANNEL);
    ops[i++] = NPUOP(OP_REG_PPU, ow - 1u,    R76_PPU_OUT_WIDTH);
    ops[i++] = NPUOP(OP_REG_PPU, oh - 1u,    R76_PPU_OUT_HEIGHT);
    ops[i++] = NPUOP(OP_REG_PPU, creg - 1u,  R76_PPU_OUT_CHANNEL);
    ops[i++] = NPUOP(OP_REG_PPU, mode,       R76_PPU_MODE);
    ops[i++] = NPUOP(OP_REG_PPU,
                     ((uint32_t)(p->stride_y - 1u) << 20) |
                     ((uint32_t)(p->stride_x - 1u) << 16) |
                     ((uint32_t)(p->kh - 1u) << 8) |
                      (uint32_t)(p->kw - 1u), R76_PPU_KERNEL);
    /* 1/kw and 1/kh in Q16, and zero on the max path where nothing is divided. */
    if (p->mode == ROCKET_RK3576_POOL_MAX) {
        ops[i++] = NPUOP(OP_REG_PPU, 0x0, R76_PPU_RECIP_W);
        ops[i++] = NPUOP(OP_REG_PPU, 0x0, R76_PPU_RECIP_H);
    } else {
        ops[i++] = NPUOP(OP_REG_PPU, 0x10000u / p->kw, R76_PPU_RECIP_W);
        ops[i++] = NPUOP(OP_REG_PPU, 0x10000u / p->kh, R76_PPU_RECIP_H);
    }
    /* Four pad nibbles, and the LEADING pads sit in the two LOW ones: bits [3:0] are the
     * LEFT pad and bits [7:4] the TOP. Measured, not read off a capture — every capture's
     * pool is padded symmetrically, and a symmetric pad cannot distinguish this map from
     * its transpose, which is why an asymmetric one computed a full and plausible surface
     * that was wrong nearly everywhere.
     *
     * What pins it is a ONE-DIMENSIONAL sweep (`rk3576_pool_probe pad`): a single row,
     * a unit kernel on the other axis, and the part's line scored against the window grid
     * each candidate leading pad implies. The grid follows the nibble written here at
     * [3:0] on x and [7:4] on y, at pads 0, 1 and 2 on both axes.
     *
     * The trailing pads are the high pair. Nothing this emitter produces depends on them
     * — the window count is programmed explicitly in OUT_WIDTH/OUT_HEIGHT and the span in
     * IN_WIDTH/IN_HEIGHT — so they are written for the part's benefit, not read back. */
    ops[i++] = NPUOP(OP_REG_PPU,
                     ((uint32_t)(p->pad_bottom & 0xf) << 12) |
                     ((uint32_t)(p->pad_right  & 0xf) << 8) |
                     ((uint32_t)(p->pad_top    & 0xf) << 4) |
                      (uint32_t)(p->pad_left   & 0xf), R76_PPU_PAD_CFG);
    /* The four pad values. Max pads with the minimum so a padded window never wins;
     * average pads with the input zero point, scaled 1..4 — the capture's own ramp. */
    if (p->mode == ROCKET_RK3576_POOL_MAX) {
        ops[i++] = NPUOP(OP_REG_PPU, R76_PPU_PAD_MIN, R76_PPU_PAD_VAL0);
        ops[i++] = NPUOP(OP_REG_PPU, R76_PPU_PAD_MIN, R76_PPU_PAD_VAL1);
        ops[i++] = NPUOP(OP_REG_PPU, R76_PPU_PAD_MIN, R76_PPU_PAD_VAL2);
        ops[i++] = NPUOP(OP_REG_PPU, R76_PPU_PAD_MIN, R76_PPU_PAD_VAL3);
    } else {
        /* The zero point times one to four, each sign-extended into the 19-bit field:
         * the captures carry -35, -70, -105, -140 for a zero point of -35, so the
         * multiply happens before the truncation and a negative one must not be masked
         * first. Four values because a window can straddle up to four padded taps. */
        int32_t z = p->input_zero_point;
        ops[i++] = NPUOP(OP_REG_PPU, (uint32_t)(z * 1) & 0x7ffffu, R76_PPU_PAD_VAL0);
        ops[i++] = NPUOP(OP_REG_PPU, (uint32_t)(z * 2) & 0x7ffffu, R76_PPU_PAD_VAL1);
        ops[i++] = NPUOP(OP_REG_PPU, (uint32_t)(z * 3) & 0x7ffffu, R76_PPU_PAD_VAL2);
        ops[i++] = NPUOP(OP_REG_PPU, (uint32_t)(z * 4) & 0x7ffffu, R76_PPU_PAD_VAL3);
    }
    ops[i++] = NPUOP(OP_REG_PPU, dst_reg == R76_PPU_ZERO_6054 ? p->output_dma : 0u,
                     R76_PPU_ZERO_6054);
    ops[i++] = NPUOP(OP_REG_PPU, dst_reg == R76_PPU_ZERO_6058 ? p->output_dma : 0u,
                     R76_PPU_ZERO_6058);
    ops[i++] = NPUOP(OP_REG_PPU, dst_reg == R76_PPU_ZERO_605C ? p->output_dma : 0u,
                     R76_PPU_ZERO_605C);
    ops[i++] = NPUOP(OP_REG_PPU, dst_reg == R76_PPU_ZERO_6070 ? p->output_dma : 0u,
                     R76_PPU_ZERO_6070);
    ops[i++] = NPUOP(OP_REG_PPU, dst_surf * 16u, R76_PPU_DST_SURF0);
    ops[i++] = NPUOP(OP_REG_PPU, dst_surf * 16u, R76_PPU_DST_SURF1);
    ops[i++] = NPUOP(OP_REG_PPU, dst_reg == R76_PPU_ZERO_60DC ? p->output_dma : 0u,
                     R76_PPU_ZERO_60DC);

    ops[i++] = NPUOP(OP_REG_PPU_RDMA, 0xE, R76_PPUR_S_POINTER);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, in_w - 1u, R76_PPUR_IN_WIDTH);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, in_h - 1u, R76_PPUR_IN_HEIGHT);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, creg - 1u, R76_PPUR_IN_CHANNEL);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, p->input_dma, R76_PPUR_SRC_BASE);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, pitch * 16u, R76_PPUR_SRC_LINE);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA,
                     (p->src_surf_elems ? p->src_surf_elems
                                        : r76_round4(pitch * ih)) * 16u,
                     R76_PPUR_SRC_SURF);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, 0x40, R76_PPUR_CONST_7030);

    /* The same four-word PC trailer every program on this part ends with — and the
     * LAST word is not the same value.
     *
     * PC_OPERATION_ENABLE is a per-block bitmap, not a constant. A convolution enables
     * CNA/CORE/DPU/DPU_RDMA with 0x1D; the vendor's pooling program enables PPU and
     * PPU_RDMA with 0x60, and the two bit sets are disjoint. Carrying the convolution's
     * trailer over to a pool leaves a fully configured PPU that is never started: the
     * job completes in the usual time, faults nothing, and writes nothing — which is
     * indistinguishable by inspection from a wrong destination address, and cost one
     * whole register sweep before the raw capture was read word by word rather than
     * through the block classifier. Transcribe the WHOLE program, trailer included.
     * [Manufactured capture, RKNN-Toolkit2 for rk3576] */
    ops[i++] = NPUOP(OP_NONE, 0x0, 0x0);
    ops[i++] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
    ops[i++] = NPUOP(OP_40, 0x0, 0x0);
    ops[i++] = NPUOP(OP_ENABLE, R76_PPU_ENABLE_WORD, PC_OPERATION_ENABLE);

    p->task_count = (uint32_t)i;
    return 0;
}

/* ============================================================================
 * SECTION — the DPU LUT table load
 *
 * Transcribed from a manufactured activation capture (`tests/data/rk3576-vendor-
 * capture/lut/`). The program is DPU + DPU_RDMA only and computes nothing: it
 * configures a 1x1x16 cube so the DPU has a shape to be idle in, then bursts the two
 * tables through the access window. PC_OPERATION_ENABLE takes 0x18 — DPU and
 * DPU_RDMA — which is neither the convolution's 0x1D nor the pool's 0x60.
 * ==========================================================================*/

/* DPU + DPU_RDMA. [Manufactured capture] */
#define R76_LUT_ENABLE_WORD  0x18

int gen_lut_load_rk3576(lut_load_params_rk3576_t *p)
{
    uint64_t *ops = p ? p->tasks : NULL;
    unsigned e;
    int i = 0;

    if (!p || !ops || !p->lo || !p->hi) return -1;

    ops[i++] = NPUOP(OP_REG_DPU,      0xE, R76_DPU_S_POINTER);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0xE, R76_RDMA_S_POINTER);

    /* The dummy cube: one pixel, sixteen channels, all-int8 precision. Nothing here
     * is arithmetic — the surface exists so the DPU has a well-formed job while the
     * table burst goes through. */
    ops[i++] = NPUOP(OP_REG_DPU, 0x5u, R76_DPU_FEATURE_MODE);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_DATA_FORMAT);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_OFFSET_PEND);
    ops[i++] = NPUOP(OP_REG_DPU, p->scratch_dma, R76_DPU_DST_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU, 0x1u, R76_DPU_DST_SURF);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_CUBE_WIDTH);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_CUBE_HEIGHT);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_CUBE_NOTCH);
    ops[i++] = NPUOP(OP_REG_DPU, 0xFu, R76_DPU_CUBE_CHANNEL);
    ops[i++] = NPUOP(OP_REG_DPU, 0x000F0F00u, R76_DPU_WDMA_SIZE0);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_WDMA_SIZE1);
    ops[i++] = NPUOP(OP_REG_DPU, 0x00000053u, R76_DPU_NOTCH_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_403C);
    ops[i++] = NPUOP(OP_REG_DPU, 0x2u, R76_DPU_BS_ALU_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_BS_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_BS_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, 0x00020000u, R76_DPU_BS_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_BN_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_BN_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, 0x903u, R76_DPU_BN_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_EW_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_EW_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_EW_MIN2);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_EW_MAX2);
    ops[i++] = NPUOP(OP_REG_DPU, 0x010041C1u, R76_DPU_EW_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_EW_CVT_OFFSET);
    ops[i++] = NPUOP(OP_REG_DPU, 0x1u, R76_DPU_EW_CVT_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_EW_CLAMP_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_EW_CLAMP_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_EW_OP_VALUE0);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_EW_OP_VALUE1);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_409C);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_OUT_CLAMP_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_OUT_CLAMP_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_OUT_CVT_OFFSET);
    ops[i++] = NPUOP(OP_REG_DPU, 0x00010001u, R76_DPU_OUT_CVT_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_OUT_CVT_SHIFT);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_SURFACE_ADD);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_40BC);
    ops[i++] = NPUOP(OP_REG_DPU, 0x04440000u, R76_DPU_CONST_40C0);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_40C8);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_40CC);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0040FFFFu, R76_DPU_CONST_40D0);

    /* The preamble: one write through the window at offset zero, then LUT_CFG into
     * its load value. The order is the capture's. */
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_LUT_ACCESS_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_LUT_ACCESS_DATA);
    ops[i++] = NPUOP(OP_REG_DPU, R76_LUT_CFG_LOAD, R76_LUT_CFG);
    for (e = 0x410C; e <= 0x4120; e += 4)
        ops[i++] = NPUOP(OP_REG_DPU, 0x0, e);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4130);
    for (e = 0x4140; e <= 0x4154; e += 4)
        ops[i++] = NPUOP(OP_REG_DPU, 0x0, e);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4160);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4170);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4174);
    for (e = 0x4184; e <= 0x4194; e += 4)
        ops[i++] = NPUOP(OP_REG_DPU, 0x0, e);

    /* ---- DPU_RDMA: the dummy cube's source ---- */
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_CUBE_WIDTH);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_CUBE_HEIGHT);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0xFu, R76_RDMA_CUBE_CHANNEL);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, p->scratch_dma, R76_RDMA_SRC_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_BRDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_BS_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_BS_BASE_ADDR1);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_NRDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_BN_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_ZERO_5030);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x1u, R76_RDMA_ERDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_EW_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_EW_SURF);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x9u, R76_RDMA_FEATURE_MODE);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_SRC_DMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, 0x504C);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, 0x5064);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, 0x506C);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, 0x5078);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, 0x507C);

    /* The two bursts. The select word carries the table and the start offset; each
     * data write stores one entry and advances, so the 513 writes are consecutive
     * and their order IS the table order. */
    ops[i++] = NPUOP(OP_REG_DPU, R76_LUT_SELECT_LO, R76_LUT_ACCESS_CFG);
    for (e = 0; e < RK3576_LUT_ENTRIES; e++)
        ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)(uint16_t)p->lo[e], R76_LUT_ACCESS_DATA);
    ops[i++] = NPUOP(OP_REG_DPU, R76_LUT_SELECT_HI, R76_LUT_ACCESS_CFG);
    for (e = 0; e < RK3576_LUT_ENTRIES; e++)
        ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)(uint16_t)p->hi[e], R76_LUT_ACCESS_DATA);

    ops[i++] = NPUOP(OP_NONE, 0x0, 0x0);
    ops[i++] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
    ops[i++] = NPUOP(OP_40, 0x0, 0x0);
    ops[i++] = NPUOP(OP_ENABLE, R76_LUT_ENABLE_WORD, PC_OPERATION_ENABLE);

    r76_apply_overrides(ops, i);
    r76_dump_program(ops, i);

    p->task_count = (uint32_t)i;
    return 0;
}

/* ============================================================================
 * SECTION — the elementwise binary op
 *
 * Transcribed from manufactured captures (`tests/data/rk3576-vendor-capture/add/`).
 * The program is DPU + DPU_RDMA only, 89 writes, PC_OPERATION_ENABLE 0x18 — the LUT
 * load's shape, but this one computes. Five captures over two geometries emit it
 * IDENTICALLY apart from the plane, the channel count and the two converters, which
 * is what says the shape is the op rather than one graph's accident.
 *
 * The DPU reads its primary operand from DPU_RDMA's own source (BRDMA_CFG 0x1a, where
 * a convolution carries 0x710 and takes the CNA pipe instead) and the second from the
 * EW base (ERDMA_CFG 0x40000044 against a convolution's 0x41).
 * ==========================================================================*/

/* DPU + DPU_RDMA, as the LUT load. [Manufactured capture] */
#define R76_EW_ENABLE_WORD   0x18

/* EW_CFG. Add: the stage on (bit 0 clear), the LUT bypassed (bit 7), the operand from
 * memory (bit 6). Mul carries a different word entire — emitted, not decomposed, since
 * two captures do not pin nine bits. [Manufactured capture] */
#define R76_EW_CFG_ADD       0x8002C0C0u
#define R76_EW_CFG_MUL       0x810F4094u
#define R76_EW_BRDMA_ADD     0x1Au
#define R76_EW_BRDMA_MUL     0x2u
#define R76_EW_BS_ALU_MUL    0x2u
#define R76_EW_BS_CFG_MUL    0x00020000u
#define R76_EW_NOTCH_CFG     0x00100012u
#define R76_EW_ERDMA_CFG     0x40000044u
#define R76_EW_FEATURE_MODE  0x9u

int gen_ew_int8_rk3576(ew_params_rk3576_t *p)
{
    uint64_t *ops = p ? p->tasks : NULL;
    unsigned w, h, c, e, surf, dst_surf;
    int mul;
    int i = 0;

    if (!p || !ops) return -1;
    w = p->w; h = p->h; c = p->c;
    if (!w || !h || !c) return -1;
    if (p->mode != ROCKET_RK3576_EW_ADD && p->mode != ROCKET_RK3576_EW_MUL) return -1;
    /* The cube's channel granule. Every capture programs c-1 with no rounding, and a
     * count that is not a whole number of granules would leave the tail group's lanes
     * reading whatever the caller left there. */
    if (c % 16u) return -1;
    if (p->ew_shift > 31 || p->out_shift > 31) return -1;
    mul = (p->mode == ROCKET_RK3576_EW_MUL);

    /* ONE source-side stride register exists (0x5040), and every capture carries the
     * plane in it while both operands ARE the plane — so nothing here says whether it
     * is the EW operand's stride, the primary's, or both. It is programmed as one and
     * the two operands are required to agree, rather than inventing a second register
     * the vendor never writes. */
    surf = p->surf_elems ? p->surf_elems : (unsigned)w * h;
    dst_surf = p->dst_surf_elems ? p->dst_surf_elems : (unsigned)w * h;

    ops[i++] = NPUOP(OP_REG_DPU,      0xE, R76_DPU_S_POINTER);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0xE, R76_RDMA_S_POINTER);

    ops[i++] = NPUOP(OP_REG_DPU, 0x5u, R76_DPU_FEATURE_MODE);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_DATA_FORMAT);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_OFFSET_PEND);
    ops[i++] = NPUOP(OP_REG_DPU, p->dst_dma, R76_DPU_DST_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU, dst_surf, R76_DPU_DST_SURF);
    ops[i++] = NPUOP(OP_REG_DPU, w - 1u, R76_DPU_CUBE_WIDTH);
    ops[i++] = NPUOP(OP_REG_DPU, h - 1u, R76_DPU_CUBE_HEIGHT);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_CUBE_NOTCH);
    ops[i++] = NPUOP(OP_REG_DPU, c - 1u, R76_DPU_CUBE_CHANNEL);
    ops[i++] = NPUOP(OP_REG_DPU, ((c - 1u) << 16) | 0x0F00u, R76_DPU_WDMA_SIZE0);
    ops[i++] = NPUOP(OP_REG_DPU, ((h - 1u) << 16) | (w - 1u), R76_DPU_WDMA_SIZE1);
    ops[i++] = NPUOP(OP_REG_DPU, R76_EW_NOTCH_CFG, R76_DPU_NOTCH_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_403C);
    ops[i++] = NPUOP(OP_REG_DPU, mul ? R76_EW_BS_ALU_MUL : 0x0, R76_DPU_BS_ALU_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_BS_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_BS_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, mul ? R76_EW_BS_CFG_MUL : 0x0, R76_DPU_BS_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_BN_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_BN_MAX);
    /* 0x903 here as everywhere: the vendor's 0x20 pins the LUT at its table join and
     * makes the BN stage multiply by an operand no program writes. */
    ops[i++] = NPUOP(OP_REG_DPU, 0x903u, R76_DPU_BN_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_EW_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_EW_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_EW_MIN2);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_EW_MAX2);
    ops[i++] = NPUOP(OP_REG_DPU, mul ? R76_EW_CFG_MUL : R76_EW_CFG_ADD,
                     R76_DPU_EW_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)p->ew_offset, R76_DPU_EW_CVT_OFFSET);
    ops[i++] = NPUOP(OP_REG_DPU,
                     ((uint32_t)p->ew_shift << 16) | p->ew_scale,
                     R76_DPU_EW_CVT_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, 0x80000000u, R76_DPU_EW_CLAMP_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, 0x7FFFFFFFu, R76_DPU_EW_CLAMP_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_EW_OP_VALUE0);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_EW_OP_VALUE1);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_409C);
    ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)p->clamp_lo, R76_DPU_OUT_CLAMP_MIN);
    ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)p->clamp_hi, R76_DPU_OUT_CLAMP_MAX);
    ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)p->out_offset, R76_DPU_OUT_CVT_OFFSET);
    ops[i++] = NPUOP(OP_REG_DPU, 0x00010000u | p->out_scale, R76_DPU_OUT_CVT_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, p->out_shift, R76_DPU_OUT_CVT_SHIFT);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_SURFACE_ADD);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_40BC);
    ops[i++] = NPUOP(OP_REG_DPU, 0x04440000u, R76_DPU_CONST_40C0);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_40C8);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, R76_DPU_ZERO_40CC);
    /* Verbatim. Every capture of every op class on this part carries this word, and
     * its sub-bits are not pinned by any of them — reading it as a clamp pair and
     * writing 0x00407F80 instead saturated the whole surface to +127. */
    ops[i++] = NPUOP(OP_REG_DPU, 0x0040FFFFu, R76_DPU_CONST_40D0);

    /* The LUT bank, written to zero exactly as the capture does. The tables are not
     * loaded for this op and the bank must not carry a previous program's map: the
     * register file is NOT cleared between jobs on this part. */
    for (e = 0x4100; e <= 0x4120; e += 4)
        ops[i++] = NPUOP(OP_REG_DPU, 0x0, e);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4130);
    for (e = 0x4140; e <= 0x4154; e += 4)
        ops[i++] = NPUOP(OP_REG_DPU, 0x0, e);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4160);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4170);
    ops[i++] = NPUOP(OP_REG_DPU, 0x0, 0x4174);
    for (e = 0x4184; e <= 0x4194; e += 4)
        ops[i++] = NPUOP(OP_REG_DPU, 0x0, e);

    /* ---- DPU_RDMA: both operands ---- */
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, w - 1u, R76_RDMA_CUBE_WIDTH);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, h - 1u, R76_RDMA_CUBE_HEIGHT);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, c - 1u, R76_RDMA_CUBE_CHANNEL);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, p->src_dma, R76_RDMA_SRC_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, mul ? R76_EW_BRDMA_MUL : R76_EW_BRDMA_ADD,
                     R76_RDMA_BRDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_BS_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_BS_BASE_ADDR1);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_NRDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_BN_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_ZERO_5030);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, R76_EW_ERDMA_CFG, R76_RDMA_ERDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, p->ew_dma, R76_RDMA_EW_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, surf, R76_RDMA_EW_SURF);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, R76_EW_FEATURE_MODE, R76_RDMA_FEATURE_MODE);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_SRC_DMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_SURF_NOTCH);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_PAD_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_EW_SURF_NOTCH);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_ZERO_5078);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA, 0x0, R76_RDMA_ZERO_507C);

    ops[i++] = NPUOP(OP_NONE, 0x0, 0x0);
    ops[i++] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
    ops[i++] = NPUOP(OP_40, 0x0, 0x0);
    ops[i++] = NPUOP(OP_ENABLE, R76_EW_ENABLE_WORD, PC_OPERATION_ENABLE);

    r76_apply_overrides(ops, i);
    r76_dump_program(ops, i);

    p->task_count = (uint32_t)i;
    return 0;
}

/* Solve the two converter pairs. Pure arithmetic; see the header for the model.
 *
 * The primary gain is 16384 * out_scale >> out_shift and the EW gain is that times
 * (ew_scale >> ew_shift) / 16384, so out_* is fixed by sa/so alone and ew_* then only
 * has to reach sb/sa. Both scales are 16-bit and both shifts 5-bit fields; the search
 * is over the shift, taking the scale that rounds best at each. */
static int64_t r76_ew_fit(double gain, unsigned scale_bits, unsigned max_shift,
                          uint16_t *scale_out, uint8_t *shift_out)
{
    double best_err = -1.0;
    unsigned sh;
    /* The hardware field is SIGNED 16-bit: 32768 reads back as -32768 and flips the
     * output's sign, measured. So the usable maximum is 32767, not 65535. */
    uint32_t lim = (1u << (scale_bits - 1)) - 1u;

    if (!(gain > 0.0)) return -1;
    for (sh = 0; sh <= max_shift; sh++) {
        double s = gain * (double)(1u << sh);
        double got, err;
        uint32_t si;

        if (s < 1.0 || s > (double)lim) continue;
        si = (uint32_t)(s + 0.5);
        if (si < 1u || si > lim) continue;
        got = (double)si / (double)(1u << sh);
        err = fabs(got - gain) / gain;
        if (best_err < 0.0 || err < best_err) {
            best_err = err;
            *scale_out = (uint16_t)si;
            *shift_out = (uint8_t)sh;
        }
    }
    if (best_err < 0.0) return -1;
    return (int64_t)(best_err * 1e9 + 0.5);
}

int64_t rocket_rk3576_ew_params(double gain,
                                uint16_t *ew_scale, uint8_t *ew_shift,
                                uint16_t *out_scale, uint8_t *out_shift)
{
    int64_t best = -1;
    unsigned esh;

    if (!ew_scale || !ew_shift || !out_scale || !out_shift) return -1;
    if (!(gain > 0.0)) return -1;

    /* The EW pair can only scale by a whole number over a power of two, so it is the
     * coarse step; OUT then carries the remainder. Sweeping the EW shift and fitting
     * OUT to what is left reaches a finer grid than either pair alone. */
    for (esh = 0; esh <= 15; esh++) {
        uint16_t es = 0, os = 0;
        uint8_t osh = 0;
        double eg;
        int64_t err;

        /* Keep the EW factor near unity so the operand is not crushed before OUT. */
        double want = (double)(1u << esh);
        if (want < 1.0 || want > 65535.0) continue;
        es = (uint16_t)want;
        eg = (double)es / (double)(1u << esh);          /* == 1, by construction */
        err = r76_ew_fit(gain / eg / 16384.0, 16, 31, &os, &osh);
        if (err < 0) continue;
        if (best < 0 || err < best) {
            best = err;
            *ew_scale = es; *ew_shift = (uint8_t)esh;
            *out_scale = os; *out_shift = osh;
        }
    }
    return best;
}
