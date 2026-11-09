// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * npu_regcmd.c — the RK3588 register-program encoder: the matmul, convolution, DPU-only and
 * pooling generators, written on Mesa's rocket register map (npu_regmap.h).
 *
 * A convolution or matmul generator describes its task by what the task computes, in a
 * struct conv_task: the input cube and how the CNA reads it, the window, the weights, the
 * output cube, and the DPU stages the result passes through. emit_conv_task() derives every
 * register from that description through the generated field functions. A matmul is a 1x1
 * convolution over a feature cube one pixel wide and M rows high, with N kernels.
 *
 * The program follows the shape of Mesa's rkt_regcmd.c: the two S_POINTER writes, the CNA,
 * CORE, DPU and DPU_RDMA groups in Mesa's register order, then the PC trailer. A dump of one
 * of these programs reads against Mesa's decoder register for register. Where a value differs
 * from Mesa's, the comment at the field says what this program needs and why.
 *
 * The register offsets and the position of each field come from the database. What a field
 * means beyond its name is stated where the field is programmed.
 *
 * THE DPU_RDMA TRAP. A convolution or matmul with no elementwise or bias operand must still
 * program the DPU_RDMA group and set MRDMA_DISABLE in its FEATURE_MODE_CFG, and the trailer
 * must enable the DPU_RDMA block. Otherwise the DPU's read DMA waits for a main feed that
 * never arrives, the DPU never completes, and the job times out with its output untouched.
 * Every program here programs that group.
 */

#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "npu_regmap.h"
#include "npu_matmul.h"
#include "npu_activation.h"
#include "npu_requant.h"
#include "npu_pool.h"
#include "rocket_hw_profile.h"
#include "rocket_log.h"

/* The CBUF bank count of the active profile. The generators take a params struct and never
 * see a context, so they read the profile rocket_hw_current() resolves once. */
#define CBUF_BANKS ((unsigned)rocket_hw_current()->cbuf_banks)

/* ============================================================================
 * The program writer
 * ==========================================================================*/

typedef struct {
    uint64_t *w;
    int n;
} regcmd_t;

/* One register write: the target (the block select and the write bit) in bits 63:48, the
 * value in 47:16, the register offset in 15:0. */
static inline void emit_raw(regcmd_t *rc, uint32_t target, uint32_t reg, uint32_t value)
{
    rc->w[rc->n++] = ((uint64_t)target << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
}

/* A write to a register the database lists: the target is its block's. */
static inline void emit(regcmd_t *rc, uint32_t reg, uint32_t value)
{
    emit_raw(rc, rkt_get_target(reg) + 0x1, reg, value);
}

static inline void emit_word(regcmd_t *rc, uint64_t word) { rc->w[rc->n++] = word; }

/* The PC trailer that ends every task. The first word is an empty write, which a chained
 * submit replaces with a PC_BASE_ADDRESS write to the next task (rocket_chain.c). The
 * TRM requires the 0x0041 word before the enable. The enable write's value is the blocks the
 * task runs, in the global enable register's layout (npu_regmap.h). */
static void emit_trailer(regcmd_t *rc, uint32_t blocks)
{
    emit_word(rc, 0);
    emit(rc, REG_PC_REGISTER_AMOUNTS, 0);
    emit_word(rc, (uint64_t)TARGET_PRE_ENABLE << 48);
    emit_raw(rc, TARGET_ENABLE, REG_PC_OPERATION_ENABLE, blocks);
}

/* S_POINTER: the task's registers go to the group the pointer selects, the pointer and the
 * executer alternate between the two groups, and the executer follows the pointer. 0xE, as
 * Mesa programs it, for every block that has the register. */
#define SPTR_DPU      (DPU_S_POINTER_POINTER_PP_MODE(1) | DPU_S_POINTER_EXECUTER_PP_EN(1) |   \
                       DPU_S_POINTER_POINTER_PP_EN(1))
#define SPTR_DPU_RDMA (DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |                          \
                       DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |                           \
                       DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1))
#define SPTR_PPU      (PPU_S_POINTER_POINTER_PP_MODE(1) | PPU_S_POINTER_EXECUTER_PP_EN(1) |   \
                       PPU_S_POINTER_POINTER_PP_EN(1))
#define SPTR_PPU_RDMA (PPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |                          \
                       PPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |                           \
                       PPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1))

/* Every stage of the DPU's BS and BN units bypassed. */
#define BS_ALL_BYPASS (DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1) |           \
                       DPU_BS_CFG_BS_ALU_BYPASS(1) | DPU_BS_CFG_BS_BYPASS(1))
#define BN_ALL_BYPASS (DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |           \
                       DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1))
#define EW_ALL_BYPASS (DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |        \
                       DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) |             \
                       DPU_EW_CFG_EW_BYPASS(1))
#define RDMA_WEIGHTS_1 (DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |  \
                        DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1))

/* ============================================================================
 * The diagnostic environment variables
 * ==========================================================================*/

/* The generators read their ROCKET_* variables on every call, and that is deliberate. A
 * generator runs once per output tile, which looks like a hot-path getenv to hoist into a
 * cache, but neither half holds. The cost is not there: on the RK1 at M=512 K=3840 N=4096 the
 * whole `gen` bucket ROCKET_MM_PROFILE reports is 2-5 ms against a ~800 ms wall, and the
 * environ scans are a fraction of that. And a cache breaks the sweeps: the RE tests set a
 * variable between calls in one process (ew_mul_rocket selects add, sub or mul through
 * ROCKET_EW_CFG per operation, matmul_int8_dequant_rocket walks ROCKET_INT8_DEQ_*), and a cache
 * keyed on the first read repeats the first cell with a full, correctly sized, wrong output.
 * Caching them was tried, and those two tests caught it [HW sweep, RK1, 2026-08-22].
 *
 * The variables below are the exception: they are read once, through pthread_once, because
 * the fp16 matmul's emission runs on the fan-out's worker threads as well as the caller's and
 * the values must be published together. Unset, each leaves the program unchanged.
 *
 *   ROCKET_CNA_CLK_GATE, ROCKET_CORE_SOFT_GATING, ROCKET_CORE_MAC_GATING  The matmul programs
 *     never write CNA_CLK_GATE (0x1090: bits 0, 1, 2 and 4 disable the feature-fetch,
 *     weight-fetch, CSC and CBUF clock gates) or CORE_MAC_GATING (0x300C, reset 0x07800800),
 *     and write MISC_CFG.SOFT_GATING as 0. Each variable writes its value instead
 *     (tests/npu_core_diff_probe).
 *   ROCKET_CORE_3030  The value of CORE 0x3030 in a matmul program, 0 by default.
 *   ROCKET_MM_MWIDTH=1  The fp16 matmul lays its M rows along the CNA width at height 1. The
 *     bytes are the same: a C2=8 cube at H=1 is [C1][M][8] either way.
 *   ROCKET_MM_SURF_TRUNC=1  The fp16 matmul's surface stride as line*(H/4 - 1) clamped at 0,
 *     the truncating form that is wrong at every M%4 != 0.
 *   ROCKET_MM_GRAINS=n  The fp16 matmul's FEATURE_GRAINS in place of M+1.
 *   ROCKET_MM_LINE_STRIDE=n, ROCKET_MM_SURF_PER_M=s  The fp16 matmul's line stride raw, and its
 *     surface stride as s*M (an external userspace programs 0x200 and -3 with M along the
 *     width) (tests/msmall_probe).
 *   ROCKET_CONV_SURF_TRUNC=1  The convolutions clamp a surface stride below four rows at 0,
 *     which misreads every input of more than one channel group below four rows. */
static struct {
    long clk_gate, soft_gating, mac_gating, core_3030;
    long mm_mwidth, mm_surf_trunc, mm_grains, mm_line, mm_surf_per_m, mm_surf_per_m_set;
    long conv_surf_trunc;
} g_diag = { -1, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0 };
static pthread_once_t g_diag_once = PTHREAD_ONCE_INIT;

static void diag_read(void)
{
    const char *e;
    if ((e = getenv("ROCKET_CNA_CLK_GATE")))     g_diag.clk_gate    = (long)(strtoul(e, NULL, 0) & 0x17);
    if ((e = getenv("ROCKET_CORE_SOFT_GATING"))) g_diag.soft_gating = (long)(strtoul(e, NULL, 0) & 0x3F);
    if ((e = getenv("ROCKET_CORE_MAC_GATING")))  g_diag.mac_gating  = (long)(strtoul(e, NULL, 0) & 0x7FFFFFF);
    if ((e = getenv("ROCKET_CORE_3030")))        g_diag.core_3030   = (long)(strtoul(e, NULL, 0) & 0xFFFFFFFF);
    if ((e = getenv("ROCKET_MM_MWIDTH")))        g_diag.mm_mwidth   = atoi(e) != 0;
    if ((e = getenv("ROCKET_MM_SURF_TRUNC")))    g_diag.mm_surf_trunc = atoi(e) != 0;
    if ((e = getenv("ROCKET_MM_GRAINS")))        g_diag.mm_grains   = (long)(strtoul(e, NULL, 0) & 0x3FF);
    if ((e = getenv("ROCKET_MM_LINE_STRIDE")))   g_diag.mm_line     = (long)(strtoul(e, NULL, 0) & 0xFFFFFFF);
    if ((e = getenv("ROCKET_MM_SURF_PER_M"))) {
        g_diag.mm_surf_per_m = strtol(e, NULL, 0);
        g_diag.mm_surf_per_m_set = 1;
    }
    if ((e = getenv("ROCKET_CONV_SURF_TRUNC")))  g_diag.conv_surf_trunc = atoi(e) != 0;
}

static void diag_once(void) { pthread_once(&g_diag_once, diag_read); }

/* The variable as an unsigned (strtoul, base 0) when it is set, `def` otherwise. */
static unsigned long env_ul(const char *name, unsigned long def)
{
    const char *e = getenv(name);
    return e ? strtoul(e, NULL, 0) : def;
}

static long env_l(const char *name, long def)
{
    const char *e = getenv(name);
    return e ? strtol(e, NULL, 0) : def;
}

/* ============================================================================
 * Shared helpers
 * ==========================================================================*/

/* OUT_CVT_SHIFT.CVT_ROUND, the integer requant's tie rule: 0 rounds half to even. Declared in
 * npu_dpu.h and shared with the RK3576 encoder, which keeps the field at 0x40B4.
 * ROCKET_OUT_CVT_ROUND overrides `requested` and is read per emission, so a probe can flip it
 * between two submits of one process (tests/requant_round_probe.c). */
uint32_t npu_out_cvt_round_bit(unsigned requested)
{
    const char *e = getenv("ROCKET_OUT_CVT_ROUND");
    unsigned r = (e && *e) ? (unsigned)strtoul(e, NULL, 0) : requested;
    return DPU_OUT_CVT_SHIFT_CVT_ROUND(r & 1u);
}

/* A streamed weight cube (one larger than its banks) needs one whole 32-kernel atom resident,
 * and never fewer than two banks. Below that the task does not complete. */
unsigned npu_stream_weight_banks(unsigned weight_bytes_per_kernel)
{
    unsigned atom = (weight_bytes_per_kernel * 32 + RK3588_CBUF_BANK_BYTES - 1) / RK3588_CBUF_BANK_BYTES;
    return atom < 2 ? 2 : atom;
}

static unsigned ceil_div(unsigned a, unsigned b) { return a / b + (a % b != 0); }

/* The CNA's surface stride for a packed feature cube `h` rows high, in 16-byte atoms. The CNA
 * steps a cube in lines of four rows (the line stride is width*4 atoms) and reaches the next
 * surface by the line stride plus this stride, so a surface of h rows needs width*(h - 4).
 * Below four rows that is negative, and CNA_DMA_CON2.SURF_STRIDE takes it as a 28-bit two's
 * complement. The truncating line*(h/4 - 1) agrees at every h%4 == 0 and is wrong at every
 * other h. With this form fp16 and int8 tasks are exact at M 1-17, 31, 33, 63 and 64, along the
 * height or the width [HW sweep 2026-10-21, tests/msmall_probe]. */
static int32_t feature_surf_stride(unsigned line_stride, unsigned h)
{
    return (int32_t)((long)line_stride * ((long)h - 4) / 4);
}

/* The convolutions' surface stride, clamped at 0 below four rows under ROCKET_CONV_SURF_TRUNC. */
static int32_t conv_surf_stride(unsigned line_stride, unsigned h)
{
    diag_once();
    int32_t s = feature_surf_stride(line_stride, h);
    return g_diag.conv_surf_trunc && s < 0 ? 0 : s;
}

/* The RK3588's geometry-register encoding, refused on a part that does not run it. Another
 * revision re-packs the CNA, CORE and DPU geometry registers at the same block bases, so a
 * program emitted here reaches it as a valid-looking job that completes in the usual time,
 * faults nothing and writes nothing. The guard sits at the generators rather than at the op
 * entries because the op library composes these generators, so every op is covered by
 * construction. A chip with its own encoder dispatches ahead of this (npu_regcmd_rk3576.c). */
static int rk3588_encoding_only(const char *gen)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (!strcmp(hw->name, "rk3588")) return 0;
    ROCKET_LOGE("%s emits the RK3588 geometry-register encoding, which the %s does not "
                "run: the job would complete and write nothing. This datapath has no "
                "encoder for this part yet\n", gen, hw->name);
    return 1;
}

/* ============================================================================
 * The convolution task
 * ==========================================================================*/

enum bs_mode { BS_OFF, BS_BIAS, BS_BIAS_MUL };

/* One CNA -> CORE -> DPU task, described by what it computes. The extents are counts, not
 * the minus-one values the CORE and DPU registers hold, and everything a register repeats is
 * derived in emit_conv_task(). The 16-bit members are the width the generators have always
 * stored them at, so a diagnostic value past it wraps the same way. */
struct conv_task {
    /* The input feature cube, and how the CNA reads it. */
    unsigned in_w, in_h, in_c;
    unsigned in_prec, proc_prec;       /* CNA_CONV_CON1 */
    unsigned conv_mode;                /* 0 direct, 3 depthwise; the CNA, DPU and DPU_RDMA */
    int      rows_plain;               /* GROUP_LINE_OFF: read A as plain [M][K] rows */
    uint32_t in_addr;
    uint32_t line_stride;              /* atoms */
    int32_t  surf_stride;              /* atoms, a 28-bit two's complement in the register */
    uint16_t grains;                   /* FEATURE_GRAINS, refused past its 10 bits */
    uint8_t  kernel_group;
    uint16_t data_entries;             /* CBUF entries of one input row */
    unsigned data_bank, weight_bank, fc_data_bank;
    int      weight_reuse, data_reuse;

    /* The window. Each field holds its register value: a dilation or deconvolution stride is
     * the rate minus one. */
    unsigned kw, kh, stride_x, stride_y, dil_x, dil_y;
    int      deconv;
    unsigned deconv_x, deconv_y;
    unsigned pad_left, pad_top;
    uint32_t pad_value;                /* CNA_PAD_CON1, the value a border reads */

    /* The weights. */
    unsigned kernels;
    uint32_t weight_bytes, weight_bytes_per_kernel, weight_addr;

    /* The CORE. */
    unsigned core_prec;
    int      qd_en, dw_en;

    /* The output cube, and how the DPU writes it. */
    unsigned out_w, out_h, out_c;
    unsigned dpu_in_prec, dpu_proc_prec, out_prec, output_mode;
    int      fp16_out;               /* narrow to fp16; also MRDMA_FP16TOFP32_EN */
    unsigned size_e;
    int      od_bypass, mc_surf_out, tp_precision, tp_org_en;
    unsigned size_c;
    uint32_t out_addr, dst_surf_stride, surf_add;
    unsigned notch;                    /* atoms from one output row to the next, minus one */
    uint32_t ow_op;                    /* BS_OW_OP, CPEND: 0x80 minus the weight zero point */

    /* The DPU stages. */
    enum bs_mode bs;
    uint32_t bias_addr;
    unsigned bs_shift;
    int      bn_mul;                   /* a per-tensor BN multiply by bn_operand */
    uint16_t bn_operand;
    const lut_epilogue_t *lut;         /* the conv -> activation epilogue */
    struct {
        int on;
        uint32_t cfg, erdma_cfg, rdma_fmc, src_addr, ew_addr, surf_stride, notch;
    } ew;                              /* the accumulate path's second operand */
    struct {
        uint32_t offset;
        uint16_t scale;
        uint8_t  shift, minus_exp, cvt_type, round;
    } cvt;

    /* Whether the matmul-only diagnostic variables apply (CLK_GATE, the CORE gating, 0x3030,
     * FC_DATA_BANK). */
    int matmul_diag;
};

/* The CNA holds the input width and height and the output width in 11 bits each, the output
 * atomics in 18 (DATAOUT_ATOMICS is 22 bits in the database, and 18 is the range the
 * programs were validated over), and the DPU holds each extent minus one in 13. A value past
 * its field wraps, and the task computes a full, plausible output of the wrong extent (an
 * int8 conv tile 2500 columns wide computes 452), so the emitter refuses it. */
static int task_fits(const struct conv_task *t)
{
    const struct { const char *name; unsigned long v, max; } f[] = {
        { "CNA input width",    t->in_w,              0x7FF },
        { "CNA input height",   t->in_h,              0x7FF },
        { "CNA output width",   t->out_w,             0x7FF },
        { "CNA output atomics", (unsigned long)t->out_w * t->out_h, 0x3FFFF },
        { "DPU width",          (unsigned)(t->out_w - 1), 0x1FFF },
        { "DPU height",         (unsigned)(t->out_h - 1), 0x1FFF },
        { "DPU channel",        (unsigned)(t->out_c - 1), 0x1FFF },
    };
    for (size_t k = 0; k < sizeof f / sizeof f[0]; k++)
        if (f[k].v > f[k].max) {
            ROCKET_LOGE("npu_regcmd: %s %lu exceeds its field (max %lu); the task would "
                        "compute a wrapped extent\n", f[k].name, f[k].v, f[k].max);
            return 0;
        }
    /* A diagnostic override can ask for more grains than the 10-bit field holds. Refused here
     * rather than by an assert, which a release build compiles out. */
    if (t->grains > 0x3FF) {
        ROCKET_LOGE("npu_regcmd: feature_grains %u exceeds 10-bit CNA field (max 1023)\n",
                    t->grains);
        return 0;
    }
    return 1;
}

/* The LE and LO tables, 513 entries each, through the DPU LUT access port: a write-mode
 * ACCESS_CFG selects the table and address 0, and each ACCESS_DATA write stores one entry
 * and advances the address. The 513 writes of a table must stay contiguous, because another
 * ACCESS_CFG resets the address. */
static void emit_lut_tables(regcmd_t *rc, const uint16_t *lut)
{
    for (unsigned table = 0; table < 2; table++) {
        emit(rc, REG_DPU_LUT_ACCESS_CFG, DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE(1) |
                                         DPU_LUT_ACCESS_CFG_LUT_TABLE_ID(table));
        for (unsigned j = 0; j <= 512; j++)
            emit(rc, REG_DPU_LUT_ACCESS_DATA, DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA(lut[table * 513 + j]));
    }
}

/* The LUT's range and extrapolation, from an epilogue or an activation's parameters. The LE
 * table's slope beyond its start and the LO table's beyond its end are written to the low
 * half of each slope register, the field the database names UFLOW. */
static void emit_lut_control(regcmd_t *rc, unsigned le_sel, unsigned lo_sel,
                             uint32_t le_start, uint32_t lo_end, int with_le_end_lo_start,
                             unsigned le_scale, unsigned le_shift,
                             unsigned lo_scale, unsigned lo_shift)
{
    /* The hybrid LE/LO mux, with the overflow and hybrid priorities set. */
    emit(rc, REG_DPU_LUT_CFG, DPU_LUT_CFG_LUT_HYBRID_PRIORITY(1) |
                              DPU_LUT_CFG_LUT_OFLOW_PRIORITY(1) | DPU_LUT_CFG_LUT_LO_LE_MUX(2));
    emit(rc, REG_DPU_LUT_INFO, DPU_LUT_INFO_LUT_LO_INDEX_SELECT(lo_sel) |
                               DPU_LUT_INFO_LUT_LE_INDEX_SELECT(le_sel));
    emit(rc, REG_DPU_LUT_LE_START, le_start);
    if (with_le_end_lo_start) {
        emit(rc, REG_DPU_LUT_LE_END, 0);
        emit(rc, REG_DPU_LUT_LO_START, 0);
    }
    emit(rc, REG_DPU_LUT_LO_END, lo_end);
    emit(rc, REG_DPU_LUT_LE_SLOPE_SCALE, DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE(le_scale));
    emit(rc, REG_DPU_LUT_LE_SLOPE_SHIFT, DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT(le_shift));
    emit(rc, REG_DPU_LUT_LO_SLOPE_SCALE, DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_UFLOW_SCALE(lo_scale));
    emit(rc, REG_DPU_LUT_LO_SLOPE_SHIFT, DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_UFLOW_SHIFT(lo_shift));
}

/* Emits one task and returns its word count, or -1 when a value does not fit its field. */
static int emit_conv_task(uint64_t *words, const struct conv_task *t)
{
    regcmd_t rc = { words, 0 };
    const lut_epilogue_t *L = t->lut;
    unsigned ow1 = t->out_w - 1, oh1 = t->out_h - 1, oc1 = t->out_c - 1;

    if (!task_fits(t)) return -1;
    diag_once();
    int diag = t->matmul_diag;

    emit(&rc, REG_DPU_S_POINTER, SPTR_DPU);
    emit(&rc, REG_DPU_RDMA_RDMA_S_POINTER, SPTR_DPU_RDMA);

    /* ---- CNA ---- */
    emit(&rc, REG_CNA_CONV_CON1,
         CNA_CONV_CON1_GROUP_LINE_OFF(t->rows_plain ? 1 : 0) |
         CNA_CONV_CON1_DECONV(t->deconv ? 1 : 0) |
         CNA_CONV_CON1_PROC_PRECISION(t->proc_prec) |
         CNA_CONV_CON1_IN_PRECISION(t->in_prec) |
         CNA_CONV_CON1_CONV_MODE(t->conv_mode));
    emit(&rc, REG_CNA_CONV_CON2, CNA_CONV_CON2_KERNEL_GROUP(t->kernel_group) |
                                 CNA_CONV_CON2_FEATURE_GRAINS(t->grains));
    emit(&rc, REG_CNA_CONV_CON3,
         CNA_CONV_CON3_ATROUS_Y_DILATION(t->dil_y) | CNA_CONV_CON3_ATROUS_X_DILATION(t->dil_x) |
         CNA_CONV_CON3_DECONV_Y_STRIDE(t->deconv_y) | CNA_CONV_CON3_DECONV_X_STRIDE(t->deconv_x) |
         CNA_CONV_CON3_CONV_Y_STRIDE(t->stride_y) | CNA_CONV_CON3_CONV_X_STRIDE(t->stride_x));
    emit(&rc, REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(t->in_w) |
                                  CNA_DATA_SIZE0_DATAIN_HEIGHT(t->in_h));
    emit(&rc, REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(t->in_c - 1) |
                                  CNA_DATA_SIZE1_DATAIN_CHANNEL(t->in_c));
    emit(&rc, REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(t->out_w));
    emit(&rc, REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(t->out_w * t->out_h));
    emit(&rc, REG_CNA_WEIGHT_SIZE0, CNA_WEIGHT_SIZE0_WEIGHT_BYTES(t->weight_bytes));
    emit(&rc, REG_CNA_WEIGHT_SIZE1,
         CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(t->weight_bytes_per_kernel));
    emit(&rc, REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(t->kw) |
                                    CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(t->kh) |
                                    CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(t->kernels));
    /* WEIGHT_REUSE and DATA_REUSE read the operand from the CBUF the previous task on this core
     * filled instead of fetching it again (gemm_params_t). FC_DATA_BANK is a field of the
     * CNA's fully-connected mode, which these programs do not use. ROCKET_FC_DATA_BANK writes
     * it in a matmul program, to confirm it is inert (tests/fc_data_bank_sweep_rocket). */
    emit(&rc, REG_CNA_CBUF_CON0,
         CNA_CBUF_CON0_WEIGHT_REUSE(t->weight_reuse ? 1 : 0) |
         CNA_CBUF_CON0_DATA_REUSE(t->data_reuse ? 1 : 0) |
         CNA_CBUF_CON0_FC_DATA_BANK(t->fc_data_bank) |
         CNA_CBUF_CON0_WEIGHT_BANK(t->weight_bank) | CNA_CBUF_CON0_DATA_BANK(t->data_bank));
    emit(&rc, REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(t->data_entries));
    /* The input converter bypassed on signed data, the non-image path Mesa programs. */
    emit(&rc, REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) |
                                CNA_CVT_CON0_CVT_BYPASS(1));
    emit(&rc, REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
    emit(&rc, REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
    emit(&rc, REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
    emit(&rc, REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
    emit(&rc, REG_CNA_FC_CON0, 0);
    emit(&rc, REG_CNA_FC_CON1, 0);
    emit(&rc, REG_CNA_PAD_CON0, CNA_PAD_CON0_PAD_LEFT(t->pad_left) | CNA_PAD_CON0_PAD_TOP(t->pad_top));
    emit(&rc, REG_CNA_FEATURE_DATA_ADDR, t->in_addr);
    emit(&rc, REG_CNA_FC_CON2, 0);
    emit(&rc, REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
    emit(&rc, REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(t->line_stride));
    emit(&rc, REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE((uint32_t)t->surf_stride));
    emit(&rc, REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(t->in_w) |
                                     CNA_FC_DATA_SIZE0_DMA_HEIGHT(t->in_h));
    emit(&rc, REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(t->in_c));
    emit(&rc, REG_CNA_DCOMP_CTRL, 0);
    emit(&rc, REG_CNA_DCOMP_REGNUM, 0);
    emit(&rc, REG_CNA_DCOMP_ADDR0, t->weight_addr);
    emit(&rc, REG_CNA_DCOMP_AMOUNT0, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT1, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT2, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT3, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT4, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT5, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT6, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT7, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT8, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT9, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT10, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT11, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT12, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT13, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT14, 0);
    emit(&rc, REG_CNA_DCOMP_AMOUNT15, 0);
    emit(&rc, REG_CNA_CVT_CON5, 0);
    emit(&rc, REG_CNA_PAD_CON1, t->pad_value);
    if (diag && g_diag.clk_gate >= 0)
        emit(&rc, REG_CNA_CLK_GATE, (uint32_t)g_diag.clk_gate);

    /* ---- CORE ---- */
    emit(&rc, REG_CORE_MISC_CFG,
         CORE_MISC_CFG_PROC_PRECISION(t->core_prec) | CORE_MISC_CFG_DW_EN(t->dw_en ? 1 : 0) |
         CORE_MISC_CFG_QD_EN(t->qd_en ? 1 : 0) |
         (diag && g_diag.soft_gating >= 0 ? CORE_MISC_CFG_SOFT_GATING((uint32_t)g_diag.soft_gating) : 0));
    if (diag && g_diag.mac_gating >= 0)
        emit(&rc, REG_CORE_MAC_GATING, (uint32_t)g_diag.mac_gating);
    emit(&rc, REG_CORE_DATAOUT_SIZE_0, CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(oh1) |
                                       CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(ow1));
    emit(&rc, REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(oc1));
    emit(&rc, REG_CORE_CLIP_TRUNCATE, 0);
    emit_raw(&rc, CORE + 0x1, REG_CORE_3030, diag ? (uint32_t)g_diag.core_3030 : 0);

    /* ---- DPU ---- */
    emit(&rc, REG_DPU_FEATURE_MODE_CFG,
         DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_CONV_MODE(t->conv_mode) |
         DPU_FEATURE_MODE_CFG_OUTPUT_MODE(t->output_mode));
    /* With the per-channel multiplier the BS shift is per sign: BS_MUL_CFG's field shifts a
     * non-negative product and this one a negative product, so both take the same shift
     * [HW sweep, RK1, tests/dw_perc_probe.c]. */
    emit(&rc, REG_DPU_DATA_FORMAT,
         DPU_DATA_FORMAT_OUT_PRECISION(t->out_prec) | DPU_DATA_FORMAT_IN_PRECISION(t->dpu_in_prec) |
         DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG(t->bs == BS_BIAS_MUL ? t->bs_shift : 0) |
         DPU_DATA_FORMAT_MC_SURF_OUT(t->mc_surf_out ? 1 : 0) |
         DPU_DATA_FORMAT_PROC_PRECISION(t->dpu_proc_prec));
    emit(&rc, REG_DPU_OFFSET_PEND, 0);
    emit(&rc, REG_DPU_DST_BASE_ADDR, t->out_addr);
    emit(&rc, REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(t->dst_surf_stride));
    emit(&rc, REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(ow1));
    emit(&rc, REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(oh1));
    /* Both halves take the notch: either half alone leaves most of a row-major tile
     * unwritten [HW sweep, tests/rowmajor_tile_probe]. */
    emit(&rc, REG_DPU_DATA_CUBE_NOTCH_ADDR, DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1(t->notch) |
                                            DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0(t->notch));
    emit(&rc, REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(oc1) |
                                         DPU_DATA_CUBE_CHANNEL_CHANNEL(oc1));

    /* BS: the per-channel int32 bias added in the ALU from BRDMA's cube (Mesa's int8 word,
     * 0x20150), and with BS_BIAS_MUL the multiplier live as well, reading its per-channel
     * operand from the same cube. */
    uint32_t bs_cfg = BS_ALL_BYPASS;
    if (t->bs != BS_OFF)
        bs_cfg = DPU_BS_CFG_BS_ALU_ALGO(2) | DPU_BS_CFG_BS_ALU_SRC(1) |
                 DPU_BS_CFG_BS_RELU_BYPASS(1) |
                 DPU_BS_CFG_BS_MUL_BYPASS(t->bs == BS_BIAS ? 1 : 0);
    emit(&rc, REG_DPU_BS_CFG, bs_cfg);
    emit(&rc, REG_DPU_BS_ALU_CFG, 0);
    emit(&rc, REG_DPU_BS_MUL_CFG, t->bs == BS_BIAS_MUL
         ? DPU_BS_MUL_CFG_BS_MUL_SHIFT_VALUE(t->bs_shift) | DPU_BS_MUL_CFG_BS_MUL_SRC(1) : 0);
    emit(&rc, REG_DPU_BS_RELUX_CMP_VALUE, 0);
    emit(&rc, REG_DPU_BS_OW_CFG,
         DPU_BS_OW_CFG_TP_ORG_EN(t->tp_org_en ? 1 : 0) | DPU_BS_OW_CFG_SIZE_E_2(t->size_e) |
         DPU_BS_OW_CFG_SIZE_E_1(t->size_e) | DPU_BS_OW_CFG_SIZE_E_0(t->size_e) |
         DPU_BS_OW_CFG_OD_BYPASS(t->od_bypass ? 1 : 0));
    emit(&rc, REG_DPU_BS_OW_OP, DPU_BS_OW_OP_OW_OP(t->ow_op));
    emit(&rc, REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_TP_PRECISION(t->tp_precision ? 1 : 0) |
                                   DPU_WDMA_SIZE_0_SIZE_C_WDMA(t->size_c) |
                                   DPU_WDMA_SIZE_0_CHANNEL_WDMA(oc1));
    emit(&rc, REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(oh1) |
                                   DPU_WDMA_SIZE_1_WIDTH_WDMA(ow1));

    /* BN: with the LUT epilogue, the stage maps the result onto the table's index grid, an
     * ALU add of the epilogue's bias word (0x80000000, fp32 -0.0, a no-op) and a multiply by
     * its index scale, as the standalone activation does. With a per-tensor operand only the
     * multiply runs. */
    if (L) {
        emit(&rc, REG_DPU_BN_CFG, DPU_BN_CFG_BN_ALU_ALGO(2) | DPU_BN_CFG_BN_RELU_BYPASS(1));
        emit(&rc, REG_DPU_BN_ALU_CFG, L->bn_alu_cfg);
        emit(&rc, REG_DPU_BN_MUL_CFG, DPU_BN_MUL_CFG_BN_MUL_OPERAND(L->bn_mul_operand));
    } else {
        emit(&rc, REG_DPU_BN_CFG, t->bn_mul ? DPU_BN_CFG_BN_RELU_BYPASS(1) |
                                              DPU_BN_CFG_BN_ALU_BYPASS(1)
                                            : BN_ALL_BYPASS);
        emit(&rc, REG_DPU_BN_ALU_CFG, 0);
        emit(&rc, REG_DPU_BN_MUL_CFG, DPU_BN_MUL_CFG_BN_MUL_OPERAND(t->bn_operand));
    }
    emit(&rc, REG_DPU_BN_RELUX_CMP_VALUE, 0);

    /* EW: the accumulate path's operation on its ERDMA operand, the LUT (EW_LUT_BYPASS clear),
     * or every stage bypassed. */
    emit(&rc, REG_DPU_EW_CFG, t->ew.on ? t->ew.cfg
                            : L ? DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1)
                                : EW_ALL_BYPASS);
    emit(&rc, REG_DPU_EW_CVT_OFFSET_VALUE, 0);
    emit(&rc, REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
    emit(&rc, REG_DPU_EW_RELUX_CMP_VALUE, 0);

    /* OUT_CVT. The LUT epilogue decodes the table's fixed-point result to fp16 as
     * (q + offset) * 2^-minus_exp, with CVT_ROUND clear. Every other program writes the
     * integer requant's tie rule (npu_out_cvt_round_bit()). OUT_CVT_SCALE bit 15 is a sign:
     * the requant scale comes from npu_out_cvt_pair(), which never sets it. */
    if (L) {
        emit(&rc, REG_DPU_OUT_CVT_OFFSET, L->out_cvt_offset);
        emit(&rc, REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) |
                                         DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(L->out_cvt_scale));
        emit(&rc, REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_CVT_TYPE(L->out_cvt_cvt_type) |
                                         DPU_OUT_CVT_SHIFT_MINUS_EXP(L->out_cvt_minus_exp));
    } else {
        emit(&rc, REG_DPU_OUT_CVT_OFFSET, t->cvt.offset);
        emit(&rc, REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(t->fp16_out ? 1 : 0) |
                                         DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(t->cvt.scale));
        emit(&rc, REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_CVT_TYPE(t->cvt.cvt_type) |
                                         npu_out_cvt_round_bit(t->cvt.round) |
                                         DPU_OUT_CVT_SHIFT_MINUS_EXP(t->cvt.minus_exp) |
                                         DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(t->cvt.shift));
    }
    emit(&rc, REG_DPU_EW_OP_VALUE_0, 0);
    emit(&rc, REG_DPU_EW_OP_VALUE_1, 0);
    emit(&rc, REG_DPU_EW_OP_VALUE_2, 0);
    emit(&rc, REG_DPU_EW_OP_VALUE_3, 0);
    emit(&rc, REG_DPU_EW_OP_VALUE_4, 0);
    emit(&rc, REG_DPU_EW_OP_VALUE_5, 0);
    emit(&rc, REG_DPU_EW_OP_VALUE_6, 0);
    emit(&rc, REG_DPU_EW_OP_VALUE_7, 0);
    emit(&rc, REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(t->surf_add));
    emit_raw(&rc, DPU + 0x1, REG_DPU_40C4, 0);
    if (L) {
        /* The tables go where the LUT registers sit. Nothing runs before the trailer's
         * enable, so their position among the DPU writes does not matter. */
        emit_lut_tables(&rc, L->lut);
        emit_lut_control(&rc, L->le_index_select, L->lo_index_select, L->lut_le_start,
                         L->lut_lo_end, 1, L->le_slope_scale, L->le_slope_shift,
                         L->lo_slope_scale, L->lo_slope_shift);
    } else {
        emit(&rc, REG_DPU_LUT_ACCESS_CFG, 0);
        emit(&rc, REG_DPU_LUT_ACCESS_DATA, 0);
        emit(&rc, REG_DPU_LUT_CFG, 0);
        emit(&rc, REG_DPU_LUT_INFO, 0);
        emit(&rc, REG_DPU_LUT_LE_START, 0);
        emit(&rc, REG_DPU_LUT_LE_END, 0);
        emit(&rc, REG_DPU_LUT_LO_START, 0);
        emit(&rc, REG_DPU_LUT_LO_END, 0);
        emit(&rc, REG_DPU_LUT_LE_SLOPE_SCALE, 0);
        emit(&rc, REG_DPU_LUT_LE_SLOPE_SHIFT, 0);
        emit(&rc, REG_DPU_LUT_LO_SLOPE_SCALE, 0);
        emit(&rc, REG_DPU_LUT_LO_SLOPE_SHIFT, 0);
    }

    /* ---- DPU_RDMA ---- */
    /* The output cube's extents. With no second operand MRDMA and ERDMA are disabled (the
     * trap in the file header). The accumulate path reads its partial through both: MRDMA at
     * the partial, ERDMA one surface on, combined by COMB_USE 5 in its FEATURE_MODE_CFG. */
    emit(&rc, REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(ow1));
    emit(&rc, REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(oh1));
    emit(&rc, REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(oc1));
    emit(&rc, REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, t->ew.on ? t->ew.src_addr : 0);
    /* BRDMA_DATA_USE 1 reads the int32 bias alone, 7 the 64-byte group of bias and
     * multiplier per eight channels. */
    emit(&rc, REG_DPU_RDMA_RDMA_BRDMA_CFG, t->bs == BS_OFF ? 0
         : DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(t->bs == BS_BIAS_MUL ? 7 : 1));
    emit(&rc, REG_DPU_RDMA_RDMA_BS_BASE_ADDR, t->bs == BS_OFF ? 0 : t->bias_addr);
    emit(&rc, REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
    emit(&rc, REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);
    if (t->ew.on) {
        emit(&rc, REG_DPU_RDMA_RDMA_ERDMA_CFG, t->ew.erdma_cfg);
        emit(&rc, REG_DPU_RDMA_RDMA_EW_BASE_ADDR, t->ew.ew_addr);
        emit(&rc, REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
             DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(t->ew.surf_stride));
        emit(&rc, REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, t->ew.rdma_fmc);
    } else {
        emit(&rc, REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1));
        emit(&rc, REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0);
        emit(&rc, REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0);
        emit(&rc, REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
             DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) |
             DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1) |
             DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(t->fp16_out ? 1 : 0) |
             DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(t->conv_mode));
    }
    emit(&rc, REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);
    emit(&rc, REG_DPU_RDMA_RDMA_SURF_NOTCH,
         t->ew.on ? DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(t->ew.notch) : 0);
    emit(&rc, REG_DPU_RDMA_RDMA_PAD_CFG, 0);
    emit(&rc, REG_DPU_RDMA_RDMA_WEIGHT, RDMA_WEIGHTS_1);
    emit(&rc, REG_DPU_RDMA_RDMA_EW_SURF_NOTCH,
         t->ew.on ? DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(t->ew.notch) : 0);

    emit_trailer(&rc, NPU_ENABLE_CONV);
    return rc.n;
}

/* ============================================================================
 * Matmul: C = A·B as a 1x1 convolution
 * ==========================================================================*/

/* What every matmul task shares: A as a feature cube one pixel wide, M rows high and K
 * channels deep, B as N 1x1 kernels, C as a cube of M rows and N channels, and every DPU
 * stage bypassed. A datatype sets the precisions, the byte sizes, the CBUF split and the
 * output writer. */
static void mm_task_init(struct conv_task *t, const gemm_params_t *p)
{
    memset(t, 0, sizeof *t);
    t->in_w = 1;  t->in_h = p->m;  t->in_c = p->k;
    t->out_w = 1; t->out_h = p->m; t->out_c = p->n;
    t->kw = t->kh = 1;
    t->kernels = p->n;
    t->stride_x = t->stride_y = 1;
    t->grains = (uint16_t)(p->m + 1);
    t->in_addr = p->input_iova;
    t->weight_addr = p->weights_iova;
    t->out_addr = p->output_iova;
    t->weight_reuse = p->weight_reuse & 1;
    t->data_reuse = p->data_reuse & 1;
    t->output_mode = 2;
    t->od_bypass = 1;
    t->cvt.scale = 1;
    t->matmul_diag = 1;
    const char *e = getenv("ROCKET_FC_DATA_BANK");
    if (e) t->fc_data_bank = (unsigned)atoi(e) & 0x7;
}

/* The CBUF split: the input cube's banks, then the weights' from the rest, the split Mesa's
 * rkt_task.c programs. An input past BANKS-1 banks refuses with -1. A kernel past one bank
 * refuses with -2, a bound of this library's own. The planners tile both.
 *
 * `slack` reserves one bank past the input's. The int8 C2=16 feature cube's DMA reads one bank
 * past ceil(bytes/bank) at resonant tile geometries and garbles the tail rows; one more bank
 * fixes the whole set (86 bad tiles to 0) and one fewer makes it worse [HW sweep,
 * ROCKET_I8_FDBANK_EXTRA]. fp16's 2-byte cube does not resonate. The int8 planner reserves
 * the matching bank, so a caller that bypasses it and leaves no room is refused rather than
 * clamped back into the corruption.
 *
 * `stream_extra` >= 0 checks a streamed weight cube (one larger than its banks): the input
 * stays resident and the CNA fetches the kernel groups through the weight banks, so one task
 * covers all of N. That needs npu_stream_weight_banks(), plus `stream_extra`: with one bank
 * and an atom smaller than it (K 384) the task never completes [HW sweep 2026-10-21,
 * tests/weight_stream_probe], and int8 at exactly the floor ran to the watchdog at (M 256,
 * K 1120) and (M 200, K 1376) and was exact one bank above it. WEIGHT_REUSE would read a CBUF
 * holding only the last atoms as the whole tile. Both refuse with -3. */
static int mm_cbuf(struct conv_task *t, unsigned fd_bytes, int slack, int stream_extra,
                   int weight_reuse, const char *gen)
{
    unsigned fd_banks = ceil_div(fd_bytes, RK3588_CBUF_BANK_BYTES);
    if (fd_banks > CBUF_BANKS - 1) return -1;
    if (t->weight_bytes_per_kernel > RK3588_CBUF_BANK_BYTES) return -2;
    t->data_bank = fd_banks;
    if (slack) {
        if (fd_banks + 1 > CBUF_BANKS - 1) {
            ROCKET_LOGE("%s: int8 CBUF bank-slack does not fit (fd_banks=%u, need "
                        "fd_banks+1 <= %u); use rocket_matmul_plan_int8\n",
                        gen, fd_banks, CBUF_BANKS - 1);
            return -1;
        }
        t->data_bank = fd_banks + 1;
    }
    t->weight_bank = CBUF_BANKS - t->data_bank;
    if (stream_extra >= 0 && t->weight_bytes > t->weight_bank * RK3588_CBUF_BANK_BYTES &&
        (t->weight_bank < npu_stream_weight_banks(t->weight_bytes_per_kernel) +
                          (unsigned)stream_extra || weight_reuse))
        return -3;
    return 0;
}

/* The input cube's strides: four rows to a line, the surface stride from the height. */
static void mm_strides(struct conv_task *t)
{
    t->line_stride = t->in_w * 4;
    t->surf_stride = feature_surf_stride(t->line_stride, t->in_h);
}

/* One CBUF entry holds 64 bytes of one input row: 32 fp16, 64 int8, 128 int4, 16 tf32. */
static void mm_entries(struct conv_task *t, unsigned per_entry)
{
    t->data_entries = (uint16_t)ceil_div(t->in_w * t->in_c, per_entry);
}

static int mm_emit(gemm_params_t *p, const struct conv_task *t)
{
    int n = emit_conv_task(p->tasks, t);
    if (n < 0) return n;
    p->task_count = (uint32_t)n;
    return 0;
}

int gen_gemm_fp16(gemm_params_t *p)
{
    if (rk3588_encoding_only("gen_gemm_fp16")) return -1;
    struct conv_task t;
    mm_task_init(&t, p);
    t.in_prec = t.proc_prec = t.core_prec = PREC_FP16;
    t.dpu_in_prec = t.dpu_proc_prec = PREC_FP16;
    t.qd_en = 1;

    diag_once();
    if (g_diag.mm_mwidth) {
        t.in_w = t.out_w = p->m;
        t.in_h = t.out_h = 1;
    }
    if (g_diag.mm_grains >= 0) t.grains = (uint16_t)g_diag.mm_grains;

    t.weight_bytes_per_kernel = (uint32_t)p->k * 2;
    t.weight_bytes = t.weight_bytes_per_kernel * p->n;
    int rc = mm_cbuf(&t, t.in_w * t.in_h * t.in_c * 2, 0, 0, p->weight_reuse != 0, "gen_gemm_fp16");
    if (rc) return rc;
    mm_entries(&t, 32);

    mm_strides(&t);
    if (g_diag.mm_surf_trunc) {
        int32_t s = (int32_t)t.line_stride * ((int32_t)(t.in_h / 4) - 1);
        t.surf_stride = s < 0 ? 0 : s;
    }
    if (g_diag.mm_surf_per_m_set)
        t.surf_stride = (int32_t)((g_diag.mm_surf_per_m * (long)p->m) & 0xFFFFFFF);
    if (g_diag.mm_line >= 0) t.line_stride = (uint32_t)g_diag.mm_line;
    /* Row-major A: GROUP_LINE_OFF with one line one row of the full A, and no surfaces, reads
     * A as plain rows, a tile inside a wider A included [HW 2026-09-26,
     * tests/rowmajor_matmul_probe, tests/rowmajor_tile_probe]. */
    if (p->a_row_atoms) {
        if (p->a_row_atoms > 0xFFFFFFF) return -1;
        t.rows_plain = 1;
        t.line_stride = p->a_row_atoms;
        t.surf_stride = 0;
    }

    /* fp32 out writes a 4-byte cube (SIZE_E 3, SURFACE_ADD four atoms a pixel), fp16 out a
     * 2-byte one (SIZE_E 1, two). */
    int narrow = p->fp16_out != 0;
    t.out_prec = narrow ? PREC_FP16 : PREC_FP32;
    t.fp16_out = p->fp16_out & 1;
    t.size_e = narrow ? 1 : 3;
    t.dst_surf_stride = t.out_h * t.out_w;
    /* Row-major C: one atom from one channel group's run to the next, the notch from one row
     * to the next, SURFACE_ADD one output group's bytes. */
    if (p->c_row_atoms) {
        if (p->c_row_atoms > 0x2000) return -1;
        t.dst_surf_stride = 1;
        t.notch = p->c_row_atoms - 1;
    }
    t.surf_add = t.dst_surf_stride * (narrow ? 2 : 4);

    /* The accumulate path: the convolution's result is the DPU's main input, and the EW stage
     * combines it with the partial at add_iova. MRDMA reads the partial and ERDMA reads it one
     * surface on (M*16 bytes), and COMB_USE 5 combines the two feeds. The surface stride and
     * the notch are M: they place the third surface on, so N <= 16 never reads them, and a
     * stride floored at MAX(M, 12) misreads every M < 12 with N > 16 [HW sweep 2026-10-22,
     * tests/msmall_probe MSP_KACC]. The precision fields are fp16, or ERDMA reads the partial
     * as int8. ew_op picks MAX or MIN, otherwise ew_mul picks the multiply over the sum.
     * ROCKET_EW_SURF, _BASE_OFF and _NOTCH override the geometry, and ROCKET_EW_CFG,
     * ROCKET_ERDMA_CFG and ROCKET_RDMA_FMC the three words. */
    {
        uint32_t M = p->m;
        uint32_t stride = (uint32_t)env_ul("ROCKET_EW_SURF", M);
        uint32_t base_off = (uint32_t)env_ul("ROCKET_EW_BASE_OFF", M * 16);
        uint32_t notch = (uint32_t)env_ul("ROCKET_EW_NOTCH", stride);
        uint32_t cfg;
        switch (p->ew_op) {
        case 2:  cfg = EW_CFG_ALU(EW_ALU_ALGO_MAX); break;
        case 3:  cfg = EW_CFG_ALU(EW_ALU_ALGO_MIN); break;
        default: cfg = p->ew_mul ? EW_CFG_MUL : EW_CFG_ALU(EW_ALU_ALGO_ADD); break;
        }
        t.ew.on = p->accumulate & 1;
        t.ew.src_addr = p->add_iova;
        t.ew.ew_addr = p->add_iova + base_off;
        t.ew.surf_stride = stride;
        t.ew.notch = notch;
        t.ew.cfg = (uint32_t)env_ul("ROCKET_EW_CFG", cfg);
        t.ew.erdma_cfg = (uint32_t)env_ul("ROCKET_ERDMA_CFG", ERDMA_CFG_16B);
        t.ew.rdma_fmc = (uint32_t)env_ul("ROCKET_RDMA_FMC",
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(5) |
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(PREC_FP16) |
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(PREC_FP16));
    }
    return mm_emit(p, &t);
}

/*
 * int8 x int8 -> int32. The int32 accumulator is written raw, and the host applies the row
 * and column scales; a K split sums its partials on the host in int64.
 *
 * The output writer is SIZE_E 7 with SURFACE_ADD eight atoms a pixel, although fp16's pattern
 * (size_e = bytes - 1) predicts 3 and 4 for an int32. 7/8 is bit-exact; 3/4 halves the surface
 * stride and leaves every column past the first surfaces unwritten. The same 7/8 holds for
 * fp32 out, which is a reinterpretation of the same 4-byte write (SIZE_E 3 there gives
 * garbage), so ROCKET_INT8_FP32_OUT is an int->float cast, not a smaller readback.
 * ROCKET_INT8_SIZE_E and ROCKET_INT8_SURF_MULT remain as diagnostics.
 *
 * ROCKET_INT8_DEQ folds a scale into OUT_CVT and writes a float (fp32, or fp16 with
 * ROCKET_INT8_DEQ_PREC=fp16). On the raw int32 accumulator the converter is integer-only:
 * out = (float)((acc * scale) >> shift), narrowed to fp16 with FP32TOFP16_EN. The scale and the
 * BN operand are integer multipliers and the shift rounds before the cast, so an integer gain
 * folds exactly and a fractional dequant scale does not. minus_exp has no effect on this
 * accumulator. cvt_type selects where OUT_CVT_OFFSET enters (1, set here, adds it before the
 * multiply; 0 at the output scale before the rounding), so at the zero offset written here the
 * two compute the same value [HW sweep 2026-11-02]. tests/matmul_int8_dequant_rocket.c sweeps
 * the fields through
 * ROCKET_INT8_DEQ_SCALE, _SHIFT, _MINEXP, _CVTTYPE, _OFFSET, _FP32TOFP16 and _BNMUL, the last
 * arming the BN per-tensor multiply. The fp16 case is bit-exact only for small single-tile
 * shapes with |acc| <= 2048.
 *
 * accumulate=1 arms an int32 EW add that does not compute as programmed: EDATA_SIZE 3 with the
 * precision fields left at fp16 adds the int32 bit patterns as floats. No library path
 * reaches it. The encoding and ROCKET_INT8_EW_* stay as the sweep harness of
 * tests/matmul_accum_int8_rocket.c. A working integer add needs the whole precision bundle at
 * int32 (DPU 0x4010 and the RDMA feature mode). With ROCKET_INT8_FP32_OUT the operands are
 * real fp32 and the float sum is right.
 *
 * ROCKET_I8_FDBANK_EXTRA, _GRAINS, _SURF, _ENTRIES and _LINESTRIDE override the CNA feature
 * fields, for the bank-slack RE.
 */
int gen_gemm_int8(gemm_params_t *p)
{
    if (rk3588_encoding_only("gen_gemm_int8")) return -1;
    struct conv_task t;
    mm_task_init(&t, p);
    t.in_prec = t.proc_prec = t.core_prec = PREC_INT8;
    t.dpu_in_prec = t.dpu_proc_prec = PREC_INT8;

    t.weight_bytes_per_kernel = p->k;
    t.weight_bytes = t.weight_bytes_per_kernel * p->n;
    int rc = mm_cbuf(&t, t.in_w * t.in_h * t.in_c, 1, 1, p->weight_reuse != 0, "gen_gemm_int8");
    if (rc) return rc;
    mm_entries(&t, 64);
    mm_strides(&t);

    int fp32_out = getenv("ROCKET_INT8_FP32_OUT") != NULL;
    int deq = getenv("ROCKET_INT8_DEQ") != NULL;
    unsigned deq_prec = PREC_FP32;
    const char *e = getenv("ROCKET_INT8_DEQ_PREC");
    if (deq && e) {
        if (!strcmp(e, "fp16"))      deq_prec = PREC_FP16;
        else if (!strcmp(e, "fp32")) deq_prec = PREC_FP32;
        else                         deq_prec = (unsigned)strtoul(e, NULL, 0) & 0x7;
    }
    t.out_prec = deq ? deq_prec : fp32_out ? PREC_FP32 : PREC_INT32;
    if (deq) {
        t.fp16_out = deq_prec == PREC_FP16;
        t.cvt.cvt_type = 1;
        if ((e = getenv("ROCKET_INT8_DEQ_SCALE")))      t.cvt.scale     = (uint16_t)strtoul(e, NULL, 0);
        if ((e = getenv("ROCKET_INT8_DEQ_SHIFT")))      t.cvt.shift     = (uint8_t)strtoul(e, NULL, 0);
        if ((e = getenv("ROCKET_INT8_DEQ_MINEXP")))     t.cvt.minus_exp = (uint8_t)strtoul(e, NULL, 0);
        if ((e = getenv("ROCKET_INT8_DEQ_CVTTYPE")))    t.cvt.cvt_type  = (uint8_t)(strtoul(e, NULL, 0) & 1);
        if ((e = getenv("ROCKET_INT8_DEQ_OFFSET")))     t.cvt.offset    = (uint32_t)strtol(e, NULL, 0);
        if ((e = getenv("ROCKET_INT8_DEQ_FP32TOFP16"))) t.fp16_out    = (int)(strtoul(e, NULL, 0) & 1);
        if ((e = getenv("ROCKET_INT8_DEQ_BNMUL"))) {
            t.bn_operand = (uint16_t)strtoul(e, NULL, 0);
            t.bn_mul = 1;
        }
    }
    t.size_e = (unsigned)env_ul("ROCKET_INT8_SIZE_E", 7) & 0x7;
    t.dst_surf_stride = t.out_h * t.out_w;
    t.surf_add = t.dst_surf_stride * (uint32_t)env_ul("ROCKET_INT8_SURF_MULT", 8);

    {
        uint32_t M = p->m;
        uint32_t stride = (uint32_t)env_ul("ROCKET_INT8_EW_SURF", M > 12 ? M : 12);
        uint32_t base_off = (uint32_t)env_ul("ROCKET_INT8_EW_BASE_OFF", M * 16);
        uint32_t notch = (uint32_t)env_ul("ROCKET_INT8_EW_NOTCH", stride);
        uint32_t prec = fp32_out ? PREC_FP32 : PREC_INT32;
        t.ew.on = p->accumulate & 1;
        t.ew.src_addr = p->add_iova;
        t.ew.ew_addr = p->add_iova + base_off;
        t.ew.surf_stride = stride;
        t.ew.notch = notch;
        t.ew.cfg = (uint32_t)env_ul("ROCKET_INT8_EW_CFG", EW_CFG_ADD_32B);
        t.ew.erdma_cfg = (uint32_t)env_ul("ROCKET_INT8_ERDMA_CFG", ERDMA_CFG_32B);
        t.ew.rdma_fmc = (uint32_t)env_ul("ROCKET_INT8_RDMA_FMC",
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(5) |
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(prec) |
            DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(prec));
    }

    if ((e = getenv("ROCKET_I8_FDBANK_EXTRA"))) {
        int nb = (int)t.data_bank + atoi(e);
        if (nb < 1) nb = 1;
        if (nb > (int)CBUF_BANKS - 1) nb = (int)CBUF_BANKS - 1;
        t.data_bank = (unsigned)nb;
        t.weight_bank = CBUF_BANKS - (unsigned)nb;
    }
    if ((e = getenv("ROCKET_I8_GRAINS")))     t.grains       = (uint16_t)strtoul(e, NULL, 0);
    if ((e = getenv("ROCKET_I8_SURF")))       t.surf_stride  = (int32_t)strtol(e, NULL, 0);
    if ((e = getenv("ROCKET_I8_ENTRIES")))    t.data_entries = (uint16_t)strtoul(e, NULL, 0);
    if ((e = getenv("ROCKET_I8_LINESTRIDE"))) t.line_stride  = (uint32_t)strtoul(e, NULL, 0);
    return mm_emit(p, &t);
}

/*
 * int4 x int4 -> int16. Precision 6 is the only value that gives a bit-exact int4 product (3
 * saturates, 7 writes nothing). Two int4 to a byte, so a kernel is ceil(K/2) bytes and a CBUF
 * entry holds 128 elements. The int16 output strides like int8's int32: SIZE_E 7 and
 * SURFACE_ADD eight atoms a pixel (SIZE_E 1 writes 16 columns, 3 writes 32, 7 all 64; 4, 8 and
 * 16 atoms all passed at M=4). One K pass, no accumulation. ROCKET_INT4_PREC, _OUT_PREC,
 * _SIZE_E, _SURF_MULT and _DENTRIES_DIV remain as diagnostics (tests/matmul_int4_rocket).
 */
int gen_gemm_int4(gemm_params_t *p)
{
    if (rk3588_encoding_only("gen_gemm_int4")) return -1;
    unsigned in_prec = (unsigned)env_ul("ROCKET_INT4_PREC", PREC_INT4) & 0x7;
    unsigned out_prec = (unsigned)env_ul("ROCKET_INT4_OUT_PREC", PREC_INT16) & 0x7;
    unsigned size_e = (unsigned)env_ul("ROCKET_INT4_SIZE_E", 7);
    unsigned surf_mult = (unsigned)env_ul("ROCKET_INT4_SURF_MULT", 8);
    unsigned per_entry = (unsigned)env_ul("ROCKET_INT4_DENTRIES_DIV", 128);
    if (!per_entry) per_entry = 128;

    struct conv_task t;
    mm_task_init(&t, p);
    t.in_prec = t.proc_prec = t.core_prec = in_prec;
    t.dpu_in_prec = t.dpu_proc_prec = in_prec;
    t.out_prec = out_prec;
    t.weight_bytes_per_kernel = ((uint32_t)p->k + 1) / 2;
    t.weight_bytes = t.weight_bytes_per_kernel * p->n;
    int rc = mm_cbuf(&t, (t.in_w * t.in_h * t.in_c + 1) / 2, 0, -1, 0, "gen_gemm_int4");
    if (rc) return rc;
    mm_entries(&t, per_entry);
    mm_strides(&t);
    t.size_e = size_e & 0x7;
    t.dst_surf_stride = t.out_h * t.out_w;
    t.surf_add = t.dst_surf_stride * surf_mult;
    return mm_emit(p, &t);
}

/*
 * int16 x int16 -> int32. RKNN's matmul menu has no int16 pairing, but the RK3588 runs one: the
 * fp16 program with the precision fields at int16 (1) and the output at int32 (4).
 *   - fp16's input geometry, int16 being two bytes: C2=8, the (N/16, K/32, 16, 32) weight tile
 *     (gemm_weight_index_int16 == gemm_weight_index_fp16), 32 elements a CBUF entry, QD_EN 1. With QD_EN 0, which
 *     int8 uses, the convolution stops after one 16-kernel group whatever M and N are.
 *   - fp16 -> fp32's output geometry, both writing four bytes: SIZE_E 3, SURFACE_ADD four atoms
 *     a pixel. Bit-exact against the int64 dot product saturated to int32 at six non-square
 *     shapes, K 32-1024, full-range operands on either side (tests/int16_native_probe, RK1,
 *     2026-09-26).
 *   - Not int8's 7/8: on an int16 program it writes row 0's first sixteen channels and the
 *     task never completes, so the kernel retires it at its 500 ms watchdog and the output
 *     reads as one 1x16 tile.
 * Saturation: an int16 product reaches 2^30, so two full-range products can pass int32. The
 * accumulator is wider (>= 36 bits measured, 48 per NVDLA) and the writer clamps, so a task is
 * exact only while |sum| < 2^31 over its K. One K pass. ROCKET_INT16_* remain as diagnostics,
 * including the DPU's transposed-writer fields (MC_SURF_OUT, TP_PRECISION, TP_ORG_EN,
 * SIZE_C_WDMA), all 0 by default (tests/matmul_int16_rocket).
 */
int gen_gemm_int16(gemm_params_t *p)
{
    if (rk3588_encoding_only("gen_gemm_int16")) return -1;
    unsigned in_prec = (unsigned)env_ul("ROCKET_INT16_PREC", PREC_INT16) & 0x7;
    unsigned out_prec = (unsigned)env_ul("ROCKET_INT16_OUT_PREC", PREC_INT32) & 0x7;
    unsigned size_e = (unsigned)env_ul("ROCKET_INT16_SIZE_E", 3);
    unsigned surf_mult = (unsigned)env_ul("ROCKET_INT16_SURF_MULT", 4);
    unsigned per_entry = (unsigned)env_ul("ROCKET_INT16_DENTRIES_DIV", 32);
    unsigned qd_en = (unsigned)env_ul("ROCKET_INT16_QD_EN", 1);
    if (!per_entry) per_entry = 32;
    long grains = env_l("ROCKET_INT16_GRAINS", -1);
    long kgroups = env_l("ROCKET_INT16_KGROUPS", -1);
    unsigned mc_surf = (unsigned)env_ul("ROCKET_INT16_MC_SURF", 0);
    unsigned tp_prec = (unsigned)env_ul("ROCKET_INT16_TP_PREC", 0);
    unsigned tp_org = (unsigned)env_ul("ROCKET_INT16_TP_ORG", 0);
    unsigned size_c = (unsigned)env_ul("ROCKET_INT16_SIZE_C", 0);

    struct conv_task t;
    mm_task_init(&t, p);
    t.in_prec = t.proc_prec = t.core_prec = in_prec;
    t.dpu_in_prec = t.dpu_proc_prec = in_prec;
    t.out_prec = out_prec;
    t.qd_en = qd_en & 1;
    if (kgroups >= 0) t.kernel_group = (uint8_t)kgroups;
    if (grains > 0) t.grains = (uint16_t)grains;
    t.weight_bytes_per_kernel = (uint32_t)p->k * 2;
    t.weight_bytes = t.weight_bytes_per_kernel * p->n;
    int rc = mm_cbuf(&t, t.in_w * t.in_h * t.in_c * 2, 0, -1, 0, "gen_gemm_int16");
    if (rc) return rc;
    mm_entries(&t, per_entry);
    mm_strides(&t);
    t.size_e = size_e & 0x7;
    t.dst_surf_stride = t.out_h * t.out_w;
    t.surf_add = t.dst_surf_stride * surf_mult;
    t.mc_surf_out = mc_surf & 1;
    t.tp_precision = tp_prec & 1;
    t.tp_org_en = tp_org & 1;
    t.size_c = size_c & 0x7FF;
    return mm_emit(p, &t);
}

/*
 * bf16 x bf16 -> fp32. The RK3588 multiplies bf16 at fp16's rate (512x3 MAC a cycle), and the
 * CNA, DPU and DPU_RDMA precision tables name 3'd3 bf16. bf16 is two bytes, so the input
 * geometry is fp16's (C2=8, the gemm_weight_index_fp16 tile, 32 elements a CBUF entry), and its products
 * accumulate in fp32, so the output is fp16's fp32 writer (SIZE_E 3, SURFACE_ADD four atoms a
 * pixel). One K pass: a split sums on the host in fp32, because a bf16 EW accumulate would read
 * 32-bit operands, which only the fp16 accumulate has been validated with. ROCKET_BF16_PREC,
 * _OUT_PREC, _SIZE_E, _SURF_MULT and _DENTRIES_DIV remain as diagnostics
 * (tests/matmul_bf16_rocket).
 */
int gen_gemm_bf16(gemm_params_t *p)
{
    if (rk3588_encoding_only("gen_gemm_bf16")) return -1;
    unsigned in_prec = (unsigned)env_ul("ROCKET_BF16_PREC", PREC_BF16) & 0x7;
    unsigned out_prec = (unsigned)env_ul("ROCKET_BF16_OUT_PREC", PREC_FP32) & 0x7;
    unsigned size_e = (unsigned)env_ul("ROCKET_BF16_SIZE_E", 3);
    unsigned surf_mult = (unsigned)env_ul("ROCKET_BF16_SURF_MULT", 4);
    unsigned per_entry = (unsigned)env_ul("ROCKET_BF16_DENTRIES_DIV", 32);
    if (!per_entry) per_entry = 32;

    struct conv_task t;
    mm_task_init(&t, p);
    t.in_prec = t.proc_prec = t.core_prec = in_prec;
    t.dpu_in_prec = t.dpu_proc_prec = in_prec;
    t.out_prec = out_prec;
    t.qd_en = 1;
    t.weight_bytes_per_kernel = (uint32_t)p->k * 2;
    t.weight_bytes = t.weight_bytes_per_kernel * p->n;
    int rc = mm_cbuf(&t, t.in_w * t.in_h * t.in_c * 2, 0, -1, 0, "gen_gemm_bf16");
    if (rc) return rc;
    mm_entries(&t, per_entry);
    mm_strides(&t);
    t.size_e = size_e & 0x7;
    t.dst_surf_stride = t.out_h * t.out_w;
    t.surf_add = t.dst_surf_stride * surf_mult;
    return mm_emit(p, &t);
}

/*
 * tf32 x tf32 -> fp32, the one 4-byte input. tf32 is an fp32 container whose mantissa the MAC
 * truncates to 10 bits, at half fp16's rate (256x3 MAC a cycle). The precision is per stage:
 * tf32 (7) at the CNA's operand read and the CORE's MAC, fp32 (5) at the DPU, whose precision
 * field has no 7. With 7 at every stage the job runs and the DPU writes nothing. The output is
 * the fp32 writer (SIZE_E 3, SURFACE_ADD four atoms a pixel). Four bytes an element: a kernel is
 * 4K bytes and a CBUF entry holds 16 elements; the weight tile is (N/16, K/16, 16, 16)
 * (gemm_weight_index_tf32). One K pass. ROCKET_TF32_PREC, _CORE_PREC, _DPU_PREC, _OUT_PREC, _SIZE_E,
 * _SURF_MULT, _DENTRIES_DIV, _ELEM_BYTES, _FGRAINS, _OUTPUT_MODE and _MC_SURF remain as
 * diagnostics (tests/matmul_tf32_rocket).
 */
int gen_gemm_tf32(gemm_params_t *p)
{
    if (rk3588_encoding_only("gen_gemm_tf32")) return -1;
    unsigned in_prec = (unsigned)env_ul("ROCKET_TF32_PREC", PREC_TF32) & 0x7;
    unsigned core_prec = (unsigned)env_ul("ROCKET_TF32_CORE_PREC", PREC_TF32) & 0x7;
    unsigned dpu_prec = (unsigned)env_ul("ROCKET_TF32_DPU_PREC", PREC_FP32) & 0x7;
    unsigned out_prec = (unsigned)env_ul("ROCKET_TF32_OUT_PREC", PREC_FP32) & 0x7;
    unsigned size_e = (unsigned)env_ul("ROCKET_TF32_SIZE_E", 3);
    unsigned surf_mult = (unsigned)env_ul("ROCKET_TF32_SURF_MULT", 4);
    unsigned per_entry = (unsigned)env_ul("ROCKET_TF32_DENTRIES_DIV", 16);
    unsigned elem = (unsigned)env_ul("ROCKET_TF32_ELEM_BYTES", 4);
    if (!per_entry) per_entry = 16;
    if (!elem) elem = 4;
    long grains = env_l("ROCKET_TF32_FGRAINS", -1);
    unsigned output_mode = (unsigned)env_ul("ROCKET_TF32_OUTPUT_MODE", 2);
    unsigned mc_surf = (unsigned)env_ul("ROCKET_TF32_MC_SURF", 0);

    struct conv_task t;
    mm_task_init(&t, p);
    t.in_prec = t.proc_prec = in_prec;
    t.core_prec = core_prec;
    t.dpu_in_prec = t.dpu_proc_prec = dpu_prec;
    t.out_prec = out_prec;
    t.qd_en = 1;
    if (grains >= 0) t.grains = (uint16_t)grains;
    t.output_mode = output_mode & 0x3;
    t.mc_surf_out = mc_surf & 1;
    t.weight_bytes_per_kernel = (uint32_t)p->k * elem;
    t.weight_bytes = t.weight_bytes_per_kernel * p->n;
    int rc = mm_cbuf(&t, t.in_w * t.in_h * t.in_c * elem, 0, -1, 0, "gen_gemm_tf32");
    if (rc) return rc;
    mm_entries(&t, per_entry);
    mm_strides(&t);
    t.size_e = size_e & 0x7;
    t.dst_surf_stride = t.out_h * t.out_w;
    t.surf_add = t.dst_surf_stride * surf_mult;
    return mm_emit(p, &t);
}

/* ============================================================================
 * Convolution
 * ==========================================================================*/

/*
 * The general CONV_2D: a KxK window with stride, pad and dilation, direct or depthwise, reduced
 * natively by the CNA with no im2col. The matmul is its 1x1, stride-1, unpadded, one-column
 * case. The host scatters the feature into an NC1HWC2 cube (feature_index) and the weights into
 * the convolution's cube (conv_weight_index_fp16 / _int8, or the depthwise forms). One call emits one
 * task. The planners split a convolution that does not fit the CBUF.
 *
 * The fp16 direct program is bit-exact against an fp32 reference over kernel, stride, pad,
 * dilation and non-square sweeps [HW sweep, RK3588, 2026-06-19]. FEATURE_GRAINS is the input
 * height plus one, and the hardware is insensitive to it (9 to 65 all pass on an 8-row input),
 * so it acts as a prefetch depth; Mesa programs 50 + stride_y + 1. A CBUF entry holds 32 fp16
 * or 64 int8 of an input row, and the fp16 count is exact: an 8-wide, 32-channel input needs 8
 * (4 fails, 16 reads inf). The dilation fields hold the rate minus one, and the x and y
 * strides go where the database places them: ROCKET_CONV_DIL_RAW and ROCKET_CONV_STRIDE_SWAP
 * make correct shapes fail. ROCKET_CONV_GRAINS and ROCKET_CONV_DENTRIES override the two
 * counts.
 *
 * The deconvolution mode is CONV_CON1.DECONV with CONV_CON3's DECONV strides holding the stride
 * minus one: the CNA interior-dilates its input, and with the output extent programmed as the
 * transposed one a single task computes a whole ConvTranspose. Power-of-two strides only, on a
 * direct stride-1 fp16 program (rocket_conv_transpose2d_fp16). ROCKET_CNA_DECONV, _X and _Y
 * write the three fields raw in every convolution program, for the deconvolution probes. NVDLA
 * has no deconvolution mode, so its documentation says nothing about this one.
 */
static int conv_common(struct conv_task *t, const conv_params_t *p, int depthwise,
                       unsigned prec, unsigned elem)
{
    const char *e;
    memset(t, 0, sizeof *t);
    t->conv_mode = depthwise ? 3 : 0;
    t->in_prec = t->proc_prec = t->core_prec = prec;
    t->dpu_in_prec = t->dpu_proc_prec = prec;
    t->grains = (uint16_t)(p->ih + 1);
    if ((e = getenv("ROCKET_CONV_GRAINS"))) t->grains = (uint16_t)strtoul(e, NULL, 0);

    unsigned sx = p->stride_x, sy = p->stride_y;
    if (getenv("ROCKET_CONV_STRIDE_SWAP")) { unsigned s = sx; sx = sy; sy = s; }
    t->stride_x = sx;
    t->stride_y = sy;
    unsigned rate_bias = getenv("ROCKET_CONV_DIL_RAW") ? 0 : 1;
    t->dil_x = (p->dil_x > 0 ? p->dil_x : 1) - rate_bias;
    t->dil_y = (p->dil_y > 0 ? p->dil_y : 1) - rate_bias;

    t->in_w = p->iw; t->in_h = p->ih; t->in_c = p->ic;
    t->out_w = p->ow; t->out_h = p->oh; t->out_c = p->oc;
    t->kw = p->kw; t->kh = p->kh;
    t->pad_left = p->pad_left; t->pad_top = p->pad_top;
    t->in_addr = p->input_iova;
    t->weight_addr = p->weights_iova;
    t->out_addr = p->output_iova;
    t->dw_en = depthwise;
    t->output_mode = 2;
    t->od_bypass = 1;
    t->cvt.scale = 1;
    t->dst_surf_stride = (uint32_t)p->oh * p->ow;

    unsigned fd_banks = ceil_div((unsigned)p->iw * p->ih * p->ic * elem, RK3588_CBUF_BANK_BYTES);
    if (fd_banks > CBUF_BANKS - 1) return -1;
    t->data_bank = fd_banks;
    t->weight_bytes_per_kernel = (uint32_t)p->kw * p->kh * p->ic * elem;
    return 0;
}

/* The weights of a direct convolution, OC kernels, or of a depthwise one: one filter per
 * channel in a (C/G, KH, KW, G) cube, padded to the group G, with WEIGHT_SIZE1 the unpadded
 * KW*KH*C as Mesa programs it. The weights take every bank the input leaves. */
static int conv_weights(struct conv_task *t, const conv_params_t *p, int depthwise,
                        unsigned elem, unsigned group)
{
    if (depthwise) {
        unsigned cpad = ceil_div(p->ic, group) * group;
        t->kernels = 1;
        t->weight_bytes = cpad * p->kh * p->kw * elem;
        if (t->data_bank + ceil_div(t->weight_bytes, RK3588_CBUF_BANK_BYTES) > CBUF_BANKS) return -2;
    } else {
        t->kernels = p->oc;
        t->weight_bytes = t->weight_bytes_per_kernel * p->oc;
        if (t->weight_bytes_per_kernel > RK3588_CBUF_BANK_BYTES) return -2;
    }
    t->weight_bank = CBUF_BANKS - t->data_bank;
    return 0;
}

/* The input strides, and the raw deconvolution overrides every convolution program takes. */
static void conv_finish(struct conv_task *t, unsigned per_entry)
{
    const char *e;
    t->data_entries = (uint16_t)ceil_div(t->in_w * t->in_c, per_entry);
    if ((e = getenv("ROCKET_CONV_DENTRIES"))) t->data_entries = (uint16_t)strtoul(e, NULL, 0);
    t->line_stride = t->in_w * 4;
    t->surf_stride = conv_surf_stride(t->line_stride, t->in_h);
}

static int conv_emit(conv_params_t *p, struct conv_task *t)
{
    const char *dc = getenv("ROCKET_CNA_DECONV");
    const char *dx = getenv("ROCKET_CNA_DECONV_X");
    const char *dy = getenv("ROCKET_CNA_DECONV_Y");
    if (dc && *dc) t->deconv = strtoul(dc, NULL, 0) != 0;
    if (dx && *dx) t->deconv_x = (unsigned)strtoul(dx, NULL, 0);
    if (dy && *dy) t->deconv_y = (unsigned)strtoul(dy, NULL, 0);
    int n = emit_conv_task(p->tasks, t);
    if (n < 0) return n;
    p->task_count = (uint32_t)n;
    return 0;
}

/*
 * fp16. The depthwise channel group is 32, half Mesa's int8 64 at the same four feature atoms
 * (G=64 fails with max_abs 6/24, G=32 is exact on every depthwise shape) [HW sweep, RK3588,
 * 2026-06-20]. The depthwise output writer is Mesa's depthwise branch, from a diff against a
 * Teflon capture: SIZE_E 3 whatever the output type, OD_BYPASS clear, SURFACE_ADD doubled,
 * FEATURE_GRAINS 50 + stride_y + 1, and CPEND 0x80 (0x80 minus a zero weight zero point). A
 * depthwise program that copies the direct program's values computes plausible channels in
 * the wrong places. ROCKET_CONV_DW_GROUP, _SIZEE and _BSOWOP override the group and the two
 * writer fields, and ROCKET_CONV_DW_OCPAD pads the output channels as Mesa does (a multiple of
 * 32, doubled at 32 or fewer, a multiple of 64). params->act fuses an activation into the DPU
 * epilogue: BN maps the result onto the table's index grid, EW runs the LUT and OUT_CVT decodes
 * the result to fp16, so the caller must set fp16_out.
 */
static int conv_fp16(conv_params_t *p, int depthwise)
{
    const char *e;
    struct conv_task t;
    int rc = conv_common(&t, p, depthwise, PREC_FP16, 2);
    if (rc) return rc;

    if (p->deconv_sy > 1 || p->deconv_sx > 1) {
        unsigned dsy = p->deconv_sy > 1 ? p->deconv_sy : 1;
        unsigned dsx = p->deconv_sx > 1 ? p->deconv_sx : 1;
        if (depthwise || p->stride_y > 1 || p->stride_x > 1 ||
            (dsy & (dsy - 1)) || (dsx & (dsx - 1)) || dsy > 8 || dsx > 8) {
            ROCKET_LOGE("gen_conv2d_fp16: the deconvolution mode takes a direct stride-1 conv "
                        "and power-of-two strides up to 8 (got %u x %u)\n", dsy, dsx);
            return -1;
        }
        t.deconv = 1;
        t.deconv_y = dsy - 1;
        t.deconv_x = dsx - 1;
    }

    unsigned group = p->dw_group ? p->dw_group : 32;
    if ((e = getenv("ROCKET_CONV_DW_GROUP"))) group = (unsigned)strtoul(e, NULL, 0);
    if (!group) group = 32;
    if ((rc = conv_weights(&t, p, depthwise, 2, group))) return rc;
    conv_finish(&t, 32);

    t.qd_en = 1;
    int narrow = p->fp16_out != 0;
    t.out_prec = narrow ? PREC_FP16 : PREC_FP32;
    t.fp16_out = p->fp16_out & 1;
    t.size_e = narrow ? 1 : 3;
    t.surf_add = t.dst_surf_stride * (narrow ? 2 : 4);

    if (depthwise) {
        t.size_e = 3;
        t.od_bypass = 0;
        t.surf_add *= 2;
        if (!getenv("ROCKET_CONV_GRAINS")) t.grains = (uint16_t)(50 + p->stride_y + 1);
        t.ow_op = 0x80;
        if ((e = getenv("ROCKET_CONV_DW_SIZEE"))) t.size_e = (unsigned)strtoul(e, NULL, 0) & 0x7;
        if ((e = getenv("ROCKET_CONV_DW_BSOWOP"))) t.ow_op = (uint32_t)strtoul(e, NULL, 0);
        if (getenv("ROCKET_CONV_DW_OCPAD")) {
            unsigned oc = ceil_div(p->oc, 32) * 32;
            if (p->oc <= 32) oc *= 2;
            t.out_c = ceil_div(oc, 64) * 64;
        }
    }
    if (p->act) t.lut = p->act;
    return conv_emit(p, &t);
}

int gen_conv2d_fp16(conv_params_t *params)
{
    if (rk3588_encoding_only("gen_conv2d_fp16")) return -1;
    return conv_fp16(params, 0);
}

int gen_conv2d_dw_fp16(conv_params_t *params)
{
    if (rk3588_encoding_only("gen_conv2d_dw_fp16")) return -1;
    return conv_fp16(params, 1);
}

/* The int8-output writer's epilogue, which the direct and the depthwise int8 programs share
 * so the two cannot disagree. Mesa's program: QD_EN 1, int8 out, the per-channel int32 bias
 * added in the BS stage from BRDMA (with bs_mul_perc the same cube's per-channel multiplier),
 * the OUT_CVT requant and CPEND.
 *   - The CNA pads a border with the input zero point in the uint8-centered domain, one byte
 *     per lane. On the depthwise program the odd channels among a group's first 32 read byte 1
 *     of the word and the rest byte 0 [HW sweep, RK1], so the byte goes in every lane. Mesa
 *     writes the byte once, sign-extended.
 *   - OUT_CVT: the multiplier and shift npu_out_cvt_pair() derives from
 *     in_scale * w_scale / out_scale (QNNPACK's derivation, without the carry Mesa's copy drops
 *     at one mantissa pattern), and the offset out_zp - 0x80. Per tensor: Mesa's per-scale
 *     truncate list is not reproduced.
 *   - CPEND: BS_OW_OP = 0x80 - weight_zero_point, with OD_BYPASS clear.
 * The caller sets the output geometry, where the two programs differ. */
static void int8_out_epilogue(struct conv_task *t, const conv_params_t *p)
{
    uint32_t pad_byte = (((uint32_t)p->input_zero_point & 0xff) - 0x80u) & 0xffu;
    unsigned scale, shift;

    t->qd_en = 1;
    t->pad_value = pad_byte * 0x01010101u;
    t->out_prec = PREC_INT8;
    t->od_bypass = 0;
    t->ow_op = 0x80 - (uint32_t)p->weight_zero_point;
    npu_out_cvt_pair((p->in_scale * p->w_scale) / p->out_scale, &scale, &shift);
    t->cvt.scale = (uint16_t)scale;
    t->cvt.shift = (uint8_t)shift;
    t->cvt.offset = (uint32_t)(p->output_zero_point - 0x80);
    t->bs = p->bs_mul_perc ? BS_BIAS_MUL : BS_BIAS;
    t->bias_addr = p->bias_iova;
    t->bs_shift = p->bs_mul_shift;
}

/*
 * int8: int8 features times int8 weights into an int32 accumulator. By default (int8_out=0) it
 * is written raw, SIZE_E 7 with SURFACE_ADD eight atoms a pixel as for the int8 matmul, and the
 * host requantizes. The feature strides do not depend on the element size: the cube atom is 16
 * bytes at fp16 (C2=8) and int8 (C2=16). A CBUF entry holds 64 elements, which equals Mesa's
 * ceil(IW*2*ceil(IC/16)/8) whenever IC%32 == 0.
 *
 * The int8 C2=16 cube needs the matmul's slack bank: at near-full feature geometries the DMA
 * reads one bank past ceil(bytes/bank) and garbles a tile's last output rows (IW=1, IC=1184,
 * IH 189-193 and 216-221 wrote ~3e5..9e5 in the last rows; one more bank fixes the window and
 * one fewer worsens it), the direct and the depthwise path alike. The int8 conv tiler reserves
 * the bank. ROCKET_CONV_I8_FDBANK_EXTRA moves it relative to that +1.
 *
 * int8_out=1, direct: Mesa's direct writer, the shared epilogue with SIZE_E 1 and SURFACE_ADD
 * two atoms a pixel, writing an int8 C2=16 cube with no readback of the accumulator. It is what
 * rocket_conv2d_int8_q (and _perc) runs, bit-exact against a host model on the RK1
 * (conv_i8out_probe, conv2d_int8_q_rocket).
 *
 * Depthwise, int8_out=1: Mesa's depthwise int8 writer, SIZE_E 3 with SURFACE_ADD four atoms a
 * pixel and G=64, matching Teflon's capture (replay_dw_mesa) and bit-exact against a host model,
 * asymmetric weight zero points through CPEND included (conv_dw_int8_runtime, cpend_wzp_probe),
 * and with a per-channel multiplier (dw_perc_probe). The int32-raw depthwise writer
 * (int8_out=0) doubles the within-plane stride and is unproven, and stays overridable through
 * ROCKET_CONV_DW_GROUP, _SIZEE, _SURF_MULT and _BSOWOP. ROCKET_CONV_INT8_SIZE_E and _SURF_MULT
 * override the other two writers.
 */
static int conv_int8(conv_params_t *p, int depthwise)
{
    const char *e;
    struct conv_task t;
    int rc = conv_common(&t, p, depthwise, PREC_INT8, 1);
    if (rc) return rc;

    unsigned group = p->dw_group ? p->dw_group : 64;
    if ((e = getenv("ROCKET_CONV_DW_GROUP"))) group = (unsigned)strtoul(e, NULL, 0);
    if (!group) group = 64;
    if ((rc = conv_weights(&t, p, depthwise, 1, group))) return rc;

    if (t.data_bank + 1 > CBUF_BANKS - 1) {
        ROCKET_LOGE("gen_conv2d_int8: int8 CBUF bank-slack does not fit (fd_banks=%u, need "
                    "fd_banks+1 <= %u); use the int8 conv tiler\n", t.data_bank, CBUF_BANKS - 1);
        return -1;
    }
    t.data_bank += 1;
    t.weight_bank = CBUF_BANKS - t.data_bank;
    if ((e = getenv("ROCKET_CONV_I8_FDBANK_EXTRA"))) {
        int nb = (int)t.data_bank + atoi(e);
        if (nb < 1) nb = 1;
        if (nb > (int)CBUF_BANKS - 1) nb = (int)CBUF_BANKS - 1;
        t.data_bank = (unsigned)nb;
        t.weight_bank = CBUF_BANKS - (unsigned)nb;
    }
    conv_finish(&t, 64);

    t.out_prec = PREC_INT32;
    t.size_e = (unsigned)env_ul("ROCKET_CONV_INT8_SIZE_E", 7) & 0x7;
    t.surf_add = t.dst_surf_stride * (uint32_t)env_ul("ROCKET_CONV_INT8_SURF_MULT", 8);

    if (!depthwise && p->int8_out) {
        int8_out_epilogue(&t, p);
        t.size_e = (unsigned)env_ul("ROCKET_CONV_INT8_SIZE_E", 1) & 0x7;
        t.surf_add = t.dst_surf_stride * (uint32_t)env_ul("ROCKET_CONV_INT8_SURF_MULT", 2);
    }
    if (depthwise) {
        t.od_bypass = 0;
        if (!getenv("ROCKET_CONV_GRAINS")) t.grains = (uint16_t)(50 + p->stride_y + 1);
        t.ow_op = 0x80 - (uint32_t)p->weight_zero_point;
        if (p->int8_out) {
            int8_out_epilogue(&t, p);
            t.size_e = 3;
            t.surf_add = t.dst_surf_stride * 4;
        }
        if ((e = getenv("ROCKET_CONV_DW_SIZEE"))) t.size_e = (unsigned)strtoul(e, NULL, 0) & 0x7;
        if ((e = getenv("ROCKET_CONV_DW_SURF_MULT")))
            t.surf_add = t.dst_surf_stride * (uint32_t)strtoul(e, NULL, 0);
        if ((e = getenv("ROCKET_CONV_DW_BSOWOP"))) t.ow_op = (uint32_t)strtoul(e, NULL, 0);
    }
    return conv_emit(p, &t);
}

int gen_conv2d_int8(conv_params_t *params)
{
    if (rk3588_encoding_only("gen_conv2d_int8")) return -1;
    return conv_int8(params, 0);
}

int gen_conv2d_dw_int8(conv_params_t *params)
{
    if (rk3588_encoding_only("gen_conv2d_dw_int8")) return -1;
    return conv_int8(params, 1);
}

/* ============================================================================
 * The DPU-only programs: the LUT activation and the elementwise multiply
 * ==========================================================================*/

/* Both run the DPU in flying mode with no CNA or CORE: MRDMA streams an fp16 cube in, and the
 * DPU writes one out. The runtime passes a flat vector of n elements, n a multiple of 8, which
 * the program partitions as a cube of 8 channels, n/8 wide and 1 high. The read and write
 * strides are equal, so out[i] corresponds to in[i] whatever the cube's shape. The trailer
 * enables the DPU and DPU_RDMA blocks. ROCKET_LUT_SPTR and ROCKET_LUT_ENABLE override the
 * S_POINTER value and the enable word, for bring-up. */
static void dpu_only_head(regcmd_t *rc, uint32_t out_addr, uint32_t stride, uint32_t width,
                          int orig_channel)
{
    emit(rc, REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) |
                                       DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2) |
                                       DPU_FEATURE_MODE_CFG_FLYING_MODE(1));
    emit(rc, REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_OUT_PRECISION(PREC_FP16) |
                                  DPU_DATA_FORMAT_IN_PRECISION(PREC_FP16) |
                                  DPU_DATA_FORMAT_PROC_PRECISION(PREC_FP16));
    emit(rc, REG_DPU_DST_BASE_ADDR, out_addr);
    emit(rc, REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(stride));
    emit(rc, REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(width));
    emit(rc, REG_DPU_DATA_CUBE_HEIGHT, 0);
    emit(rc, REG_DPU_DATA_CUBE_CHANNEL, DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(orig_channel ? 7 : 0) |
                                        DPU_DATA_CUBE_CHANNEL_CHANNEL(7));
}

static void dpu_only_rdma(regcmd_t *rc, uint32_t width, uint32_t src)
{
    emit(rc, REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(width));
    emit(rc, REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, 0);
    emit(rc, REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(7));
    emit(rc, REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, src);
}

/* MRDMA in flying mode reading fp16 and widening it to fp32, at burst 15. */
#define RDMA_FMC_FLYING_FP16 (DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION(PREC_FP16) |  \
                              DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) |                    \
                              DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION(PREC_FP16) |\
                              DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN(1) |           \
                              DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE(1))

/*
 * The LUT activation, NVDLA's SDP LUT pass. BN multiplies the input by bn_mul_operand onto the
 * table's index grid, the LE/LO hybrid table with the hardware's linear interpolation produces
 * f(x), OUT_CVT decodes the fixed-point result to fp16, and the DPU writes it.
 */
int gen_lut_activation_fp16(lut_act_params_t *p)
{
    if (rk3588_encoding_only("gen_lut_activation_fp16")) return -1;
    if (p->n <= 0 || (p->n & 0x7)) return -1;
    uint32_t cols = (uint32_t)p->n / 8;
    if (cols > 0x1FFF) return -2;                  /* DATA_CUBE_WIDTH is 13 bits */
    uint32_t width = cols - 1;
    uint32_t sptr = (uint32_t)env_ul("ROCKET_LUT_SPTR", SPTR_DPU);
    uint32_t enable = (uint32_t)env_ul("ROCKET_LUT_ENABLE", NPU_ENABLE_DPU);

    regcmd_t rc = { p->tasks, 0 };
    emit(&rc, REG_DPU_S_POINTER, sptr);
    emit(&rc, REG_DPU_RDMA_RDMA_S_POINTER, sptr);
    emit_lut_tables(&rc, p->lut);
    dpu_only_head(&rc, (uint32_t)p->output_iova, cols, width, 1);
    emit(&rc, REG_DPU_BS_CFG, BS_ALL_BYPASS);
    emit(&rc, REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_OD_BYPASS(1));
    emit(&rc, REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(7));
    emit(&rc, REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_WIDTH_WDMA(width));
    emit(&rc, REG_DPU_BN_CFG, DPU_BN_CFG_BN_ALU_ALGO(2) | DPU_BN_CFG_BN_RELU_BYPASS(1));
    emit(&rc, REG_DPU_BN_ALU_CFG, p->bn_alu_cfg);
    emit(&rc, REG_DPU_BN_MUL_CFG, DPU_BN_MUL_CFG_BN_MUL_OPERAND(p->bn_mul_operand));
    /* The LUT runs in the EW stage: EW_LUT_BYPASS clear. */
    emit(&rc, REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1));
    emit(&rc, REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
    emit(&rc, REG_DPU_OUT_CVT_OFFSET, p->out_cvt_offset);
    emit(&rc, REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_FP32TOFP16_EN(1) |
                                     DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(p->out_cvt_scale));
    emit(&rc, REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_CVT_TYPE(p->out_cvt_cvt_type) |
                                     DPU_OUT_CVT_SHIFT_MINUS_EXP(p->out_cvt_minus_exp));
    emit(&rc, REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(cols));
    emit_raw(&rc, DPU + 0x1, REG_DPU_40C4, 0);
    emit_lut_control(&rc, p->le_index_select, p->lo_index_select, p->lut_le_start,
                     p->lut_lo_end, 0, p->le_slope_scale, p->le_slope_shift,
                     p->lo_slope_scale, p->lo_slope_shift);
    dpu_only_rdma(&rc, width, (uint32_t)p->input_iova);
    emit(&rc, REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1));
    emit(&rc, REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, RDMA_FMC_FLYING_FP16);
    emit(&rc, REG_DPU_RDMA_RDMA_WEIGHT, RDMA_WEIGHTS_1);
    emit_trailer(&rc, enable);
    p->task_count = rc.n;
    return 0;
}

/* ============================================================================
 * Pooling on the PPU
 * ==========================================================================*/

/*
 * MaxPool and AveragePool: PPU_RDMA reads the fp16 NC1HWC2 cube (C2=8), the PPU reduces each
 * window, and writes the result in the same layout. No CNA, CORE or DPU, so no DPU_RDMA trap.
 * Every extent field holds the count minus one, the padding fields the count. The strides are
 * in bytes, and the fields that hold them start at bit 4, so a 16-byte-aligned byte count is
 * the register value. The trailer enables PPU and PPU_RDMA, and FLYING_MODE 1 with DPU_FLYIN 0
 * is the standalone, RDMA-fed pool. Max is bit-exact; the average's reciprocal is
 * fp16(65536/k) per axis (ppu_recip_kernel_fp16()).
 */
int gen_pool_fp16(pool_params_t *p)
{
    if (rk3588_encoding_only("gen_pool_fp16")) return -1;
    if (p->iw < 1 || p->ih < 1 || p->ow < 1 || p->oh < 1 || p->c < 1) return -1;
    if (p->kw < 1 || p->kh < 1 || p->stride_x < 1 || p->stride_y < 1) return -1;
    if ((((uint32_t)p->iw - 1) | ((uint32_t)p->ih - 1) | ((uint32_t)p->ow - 1) |
         ((uint32_t)p->oh - 1) | ((uint32_t)p->c - 1)) >> 13) return -2;
    if ((((uint32_t)p->kw - 1) | ((uint32_t)p->kh - 1) |
         ((uint32_t)p->stride_x - 1) | ((uint32_t)p->stride_y - 1)) >> 4) return -3;

    const uint32_t atom = 16;                                    /* 8 fp16 */
    uint32_t in_line = (uint32_t)p->iw * atom;
    uint32_t in_surf = (uint32_t)p->ih * p->iw * atom;
    uint32_t out_surf = (uint32_t)p->oh * p->ow * atom;
    int avg = p->method == POOL_METHOD_AVG;
    uint32_t pad = PPU_POOLING_PADDING_CFG_PAD_LEFT(p->pad_left) |
                   PPU_POOLING_PADDING_CFG_PAD_TOP(p->pad_top) |
                   PPU_POOLING_PADDING_CFG_PAD_RIGHT(p->pad_right) |
                   PPU_POOLING_PADDING_CFG_PAD_BOTTOM(p->pad_bottom);

    regcmd_t rc = { p->tasks, 0 };
    emit(&rc, REG_PPU_S_POINTER, SPTR_PPU);
    emit(&rc, REG_PPU_RDMA_RDMA_S_POINTER, SPTR_PPU_RDMA);
    emit(&rc, REG_PPU_DATA_CUBE_IN_WIDTH, PPU_DATA_CUBE_IN_WIDTH_CUBE_IN_WIDTH(p->iw - 1));
    emit(&rc, REG_PPU_DATA_CUBE_IN_HEIGHT, PPU_DATA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT(p->ih - 1));
    emit(&rc, REG_PPU_DATA_CUBE_IN_CHANNEL, PPU_DATA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(p->c - 1));
    emit(&rc, REG_PPU_DATA_CUBE_OUT_WIDTH, PPU_DATA_CUBE_OUT_WIDTH_CUBE_OUT_WIDTH(p->ow - 1));
    emit(&rc, REG_PPU_DATA_CUBE_OUT_HEIGHT, PPU_DATA_CUBE_OUT_HEIGHT_CUBE_OUT_HEIGHT(p->oh - 1));
    emit(&rc, REG_PPU_DATA_CUBE_OUT_CHANNEL, PPU_DATA_CUBE_OUT_CHANNEL_CUBE_OUT_CHANNEL(p->c - 1));
    emit(&rc, REG_PPU_OPERATION_MODE_CFG, PPU_OPERATION_MODE_CFG_FLYING_MODE(1) |
                                          PPU_OPERATION_MODE_CFG_POOLING_METHOD(p->method));
    emit(&rc, REG_PPU_POOLING_KERNEL_CFG,
         PPU_POOLING_KERNEL_CFG_KERNEL_WIDTH(p->kw - 1u) |
         PPU_POOLING_KERNEL_CFG_KERNEL_HEIGHT(p->kh - 1u) |
         PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_WIDTH(p->stride_x - 1u) |
         PPU_POOLING_KERNEL_CFG_KERNEL_STRIDE_HEIGHT(p->stride_y - 1u));
    emit(&rc, REG_PPU_RECIP_KERNEL_WIDTH, avg ? PPU_RECIP_KERNEL_WIDTH_RECIP_KERNEL_WIDTH(p->recip_w) : 0);
    emit(&rc, REG_PPU_RECIP_KERNEL_HEIGHT, avg ? PPU_RECIP_KERNEL_HEIGHT_RECIP_KERNEL_HEIGHT(p->recip_h) : 0);
    emit(&rc, REG_PPU_POOLING_PADDING_CFG, pad);
    /* A max pool pads with -inf (fp16 0xFC00) so a padded cell never wins; an average pads
     * with 0. */
    emit(&rc, REG_PPU_PADDING_VALUE_1_CFG,
         p->method == POOL_METHOD_MAX && pad ? PPU_PADDING_VALUE_1_CFG_PAD_VALUE_0(0xFC00) : 0);
    emit(&rc, REG_PPU_PADDING_VALUE_2_CFG, 0);
    emit(&rc, REG_PPU_DST_BASE_ADDR, p->output_iova);
    emit(&rc, REG_PPU_DST_SURF_STRIDE, PPU_DST_SURF_STRIDE_DST_SURF_STRIDE(out_surf / 16));
    /* INDEX_ADD takes the output's surface stride too, and PROC_PRECISION is fp16. */
    emit(&rc, REG_PPU_DATA_FORMAT, PPU_DATA_FORMAT_INDEX_ADD(out_surf / 16) |
                                   PPU_DATA_FORMAT_PROC_PRECISION(PREC_FP16));
    emit(&rc, REG_PPU_MISC_CTRL, PPU_MISC_CTRL_BURST_LEN(3));
    emit(&rc, REG_PPU_RDMA_RDMA_CUBE_IN_WIDTH, PPU_RDMA_RDMA_CUBE_IN_WIDTH_CUBE_IN_WIDTH(p->iw - 1));
    emit(&rc, REG_PPU_RDMA_RDMA_CUBE_IN_HEIGHT, PPU_RDMA_RDMA_CUBE_IN_HEIGHT_CUBE_IN_HEIGHT(p->ih - 1));
    emit(&rc, REG_PPU_RDMA_RDMA_CUBE_IN_CHANNEL, PPU_RDMA_RDMA_CUBE_IN_CHANNEL_CUBE_IN_CHANNEL(p->c - 1));
    emit(&rc, REG_PPU_RDMA_RDMA_SRC_BASE_ADDR, p->input_iova);
    emit(&rc, REG_PPU_RDMA_RDMA_SRC_LINE_STRIDE, PPU_RDMA_RDMA_SRC_LINE_STRIDE_SRC_LINE_STRIDE(in_line / 16));
    emit(&rc, REG_PPU_RDMA_RDMA_SRC_SURF_STRIDE, PPU_RDMA_RDMA_SRC_SURF_STRIDE_SRC_SURF_STRIDE(in_surf / 16));
    emit(&rc, REG_PPU_RDMA_RDMA_DATA_FORMAT, PPU_RDMA_RDMA_DATA_FORMAT_IN_PRECISION(PREC_FP16));
    emit_trailer(&rc, NPU_ENABLE_PPU);
    p->task_count = rc.n;
    return 0;
}

/* ============================================================================
 * The host layout helpers
 * ==========================================================================*/

/* Element (c, h, w) of a feature map in the NC1HWC2 cube, 1-based: `group` channels to a
 * group, groups outermost, each group a height x width plane of group-element atoms. */
int feature_index(int channels, int height, int width, int group, int c, int h, int w)
{
    (void)channels;
    int g = (c - 1) / group, lane = (c - 1) % group;
    return ((g * height + (h - 1)) * width + (w - 1)) * group + lane;
}

/* Element (n, k) of a matmul weight tile cube, 0-based: N/ng groups of kernels, each K/kg
 * groups of an ng x kg block, the K index innermost. `K` is the contraction length the caller
 * laid the cube out for. */
static int weight_tile(int K, int ng, int kg, int n, int k)
{
    return (n / ng) * ng * K + (k / kg) * ng * kg + (n % ng) * kg + k % kg;
}

/* The matmul weight index helpers take the contraction length, the kernel (N) index and the
 * contraction (K) index, the last two 1-based. */
int gemm_weight_index_fp16(int K, int n, int k)  { return weight_tile(K, 16, 32, n - 1, k - 1); }
int gemm_weight_index_int8(int K, int n, int k)  { return weight_tile(K, 32, 32, n - 1, k - 1); }
int gemm_weight_index_int16(int K, int n, int k) { return weight_tile(K, 16, 32, n - 1, k - 1); }
int gemm_weight_index_tf32(int K, int n, int k)  { return weight_tile(K, 16, 16, n - 1, k - 1); }
/* int4 groups 64 kernels, and its kernel group stride counts whole 32-element K groups. The
 * index is a nibble: byte idx/2, nibble idx&1. */
int gemm_weight_index_int4(int K, int n, int k)  { return weight_tile((K + 31) / 32 * 32, 64, 32, n - 1, k - 1); }

/* The convolution weight cube, 0-based: Mesa's loop nest oc1, ic1, kh, kw, oc2, ic2, with the
 * kernel window between the channel groups (rkt_coefs.c). fp16 groups 16 kernels and int8
 * 32, and both group 32 input channels. At KH = KW = 1 each reduces to its matmul tile. */
static int weight_conv(int IC, int og, int oc, int ic, int KH, int KW, int kh, int kw)
{
    int n_ic = (IC + 31) / 32;
    return ((((oc / og) * n_ic + ic / 32) * KH + kh) * KW + kw) * og * 32 + (oc % og) * 32 + ic % 32;
}

int conv_weight_index_fp16(int OC, int IC, int KH, int KW, int oc, int ic, int kh, int kw)
{
    (void)OC;
    return weight_conv(IC, 16, oc - 1, ic - 1, KH, KW, kh - 1, kw - 1);
}

int conv_weight_index_int8(int OC, int IC, int KH, int KW, int oc, int ic, int kh, int kw)
{
    (void)OC;
    return weight_conv(IC, 32, oc - 1, ic - 1, KH, KW, kh - 1, kw - 1);
}

/* The depthwise cube (C/G, KH, KW, G): Mesa's nest with one filter per channel. */
int conv_dw_weight_index_fp16(int C, int KH, int KW, int G, int c, int kh, int kw)
{
    (void)C;
    return (((c - 1) / G * KH + (kh - 1)) * KW + (kw - 1)) * G + (c - 1) % G;
}

int conv_dw_weight_index_int8(int C, int KH, int KW, int G, int c, int kh, int kw)
{
    return conv_dw_weight_index_fp16(C, KH, KW, G, c, kh, kw);
}
