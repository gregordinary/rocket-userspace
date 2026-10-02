// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * dw_perc_probe.c — RK3588 board probe: does the int8-out depthwise program's BS stage
 * take a PER-CHANNEL multiplier from the BRDMA cube, and what arithmetic does it do?
 *
 * The int8 depthwise entry requantizes on chip with one OUT_CVT scale, so a per-axis
 * filter (TFLite's default for a quantized conv) stays on the fp16 route. The RK3576's
 * BS stage computes `(acc + A[c]) * C[c]` with a per-channel int16 C, held wide under a
 * nonzero shift word and rounded half to even after the multiply [HW sweep, H96,
 * chips/rk3576-regcmd.md]. RKNN's RK3588 per-channel program writes BRDMA data use 7
 * over a 64-byte group per 8 output channels, BS_CFG 0x20140, BS_MUL_CFG (s << 8) | 1
 * and the same s in DATA_FORMAT.BS_MUL_SHIFT_VALUE_NEG [source-confirmed, widgetii's
 * rnpu_regcmd.c, its layout read off an .rknn]. The BS/BN field meanings are IP-inherent,
 * so the RK3576 epilogue is the hypothesis [expected], and the probe does not assume it.
 *
 * Each arm runs gen_conv2d_dw_int8's int8-out program with those words patched in, a
 * coefficient BO in a stated layout, and a sentinel-prefilled output. It scores the
 * surface against every candidate reading at once and prints the count each explains:
 *
 *   rne     v = sat32(rne(((acc + A)*C) >> s))     the RK3576's epilogue
 *   up      v = sat32(floor(((acc + A)*C + 2^(s-1)) / 2^s))
 *   floor   v = sat32(((acc + A)*C) >> s)
 *   sat1st  v = rne(sat32((acc + A)*C) >> s)       the product saturated before the shift
 *   order   v = sat32(rne((acc*C + A) >> s))       the add after the multiply
 *
 * each under three coefficient layouts (group stride / A offset / C offset): 64/0/48
 * (RKNN's), 64/0/32, and 48/0/32 (the RK3576 depthwise group). `acc` is the raw
 * sum of xp*w the CNA forms (the input zero point on a padded tap) and A the folded bias,
 * so acc + A is TFLite's accumulator. v then goes through the unchanged OUT_CVT
 * (tests/requant_model.h). A control arm runs the emitted program (bias only, the
 * multiplier bypassed) and must be exact against the per-tensor model, or no other row
 * can be read.
 *
 * Exit 0 when every arm ran (a probe: the verdict is the table), 1 when the control is
 * not exact, 77 without a device.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "test_fill.h"
#include "requant_model.h"

#define PADB NPU_CBUF_BANK_SIZE
#define DW_G 64
#define OP_REG_DPU_RDMA_P (BLOCK_DPU_RDMA | PC_OP_01)
#define RDMA_BRDMA_CFG_P   0x501C
#define RDMA_BS_BASE_P     0x5020

static int rup(int a, int b) { return ((a + b - 1) / b) * b; }
static int out_dim(int in, int k, int s, int p) { return (in + 2 * p - k) / s + 1; }

typedef struct {
    const char *name;
    int C, IH, IW, KH, KW, sy, sx, pt, pl;
} shape_t;

/* ---- regcmd word patching ------------------------------------------------------- */
static int reg_find(const uint64_t *ops, unsigned n, unsigned blockop, unsigned reg)
{
    for (unsigned i = 0; i < n; i++)
        if ((unsigned)(ops[i] >> 48) == (blockop & 0xffffu) && (unsigned)(ops[i] & 0xffffu) == reg)
            return (int)i;
    return -1;
}
static uint32_t reg_get(const uint64_t *ops, int i) { return (uint32_t)(ops[i] >> 16); }
static int reg_set(uint64_t *ops, unsigned n, unsigned blockop, unsigned reg, uint32_t v)
{
    int k = 0;
    for (unsigned i = 0; i < n; i++)
        if ((unsigned)(ops[i] >> 48) == (blockop & 0xffffu) && (unsigned)(ops[i] & 0xffffu) == reg) {
            ops[i] = NPUOP(blockop, v, reg);
            k++;
        }
    return k;
}

/* ---- the coefficient layouts ---------------------------------------------------- */
typedef struct { const char *name; int stride, a_off, c_off, b_off; } layout_t;
static const layout_t LAYOUTS[] = {
    { "64/A0/C48", 64, 0, 48, 32 },   /* RKNN's RK3588 group */
    { "64/A0/C32", 64, 0, 32, 48 },
    { "48/A0/C32", 48, 0, 32, -1 },   /* the RK3576 depthwise group */
};
#define NLAYOUT (int)(sizeof(LAYOUTS) / sizeof(LAYOUTS[0]))

static void pack_coeff(const layout_t *L, int C, const int32_t *A, const int16_t *B,
                       const int16_t *Cm, uint8_t *dst)
{
    for (int c = 0; c < C; c++) {
        uint8_t *g = dst + (size_t)(c / 8) * L->stride;
        int i = c % 8;
        memcpy(g + L->a_off + 4 * i, &A[c], 4);
        memcpy(g + L->c_off + 2 * i, &Cm[c], 2);
        if (L->b_off >= 0) memcpy(g + L->b_off + 2 * i, &B[c], 2);
    }
}

static void read_coeff(const layout_t *L, int C, const uint8_t *src, int32_t *A, int16_t *Cm,
                       int16_t *B)
{
    for (int c = 0; c < C; c++) {
        const uint8_t *g = src + (size_t)(c / 8) * L->stride;
        int i = c % 8;
        memcpy(&A[c], g + L->a_off + 4 * i, 4);
        memcpy(&Cm[c], g + L->c_off + 2 * i, 2);
        B[c] = 0;
        if (L->b_off >= 0) memcpy(&B[c], g + L->b_off + 2 * i, 2);
    }
}

/* ---- the candidate epilogues ---------------------------------------------------- */
static int64_t sat32(int64_t v)
{
    if (v > INT32_MAX) return INT32_MAX;
    if (v < INT32_MIN) return INT32_MIN;
    return v;
}
static int64_t floor_shift(int64_t p, unsigned s) { return s ? (p >> s) : p; }
static int64_t up_shift(int64_t p, unsigned s) { return s ? ((p + ((int64_t)1 << (s - 1))) >> s) : p; }

enum { H_RNE, H_UP, H_FLOOR, H_SAT1ST, H_ORDER, H_SPLIT, H_BSUM, NHYP };
static const char *HNAME[NHYP] = { "rne", "up", "floor", "sat1st", "order", "split", "B*S" };

/* split: the non-negative product shifts by s (BS_MUL_CFG's field), the negative one by
 * sn (DATA_FORMAT's NEG field). B*S: the coefficient group's B times the window sum of
 * the CNA's input, added before the multiply (the RK3576 direct group's weight-zero term). */
static int64_t epilogue(int h, int64_t acc, int32_t A, int C, unsigned s, unsigned sn,
                        int B, int64_t S)
{
    int64_t p = (acc + A) * (int64_t)C;
    switch (h) {
    case H_RNE:    return sat32(requant_round_shift(p, s));
    case H_UP:     return sat32(up_shift(p, s));
    case H_FLOOR:  return sat32(floor_shift(p, s));
    case H_SAT1ST: return requant_round_shift(sat32(p), s);
    case H_ORDER:  return sat32(requant_round_shift(acc * (int64_t)C + A, s));
    case H_SPLIT:  return sat32(requant_round_shift(p, p >= 0 ? s : sn));
    case H_BSUM:   return sat32(requant_round_shift((acc + A + (int64_t)B * S) * (int64_t)C, s));
    }
    return 0;
}

/* The BN stage after the established BS stage: the same form with E for C and no add. */
static int64_t bn_stage(int64_t v1, int E, unsigned s, unsigned sn)
{
    int64_t p = v1 * (int64_t)E;
    return sat32(requant_round_shift(p, p >= 0 ? s : sn));
}

/* ---- one arm ---------------------------------------------------------------------- */
typedef struct {
    const char *label;
    int patch;          /* 0: the emitted program (control); 1: per-channel words */
    int layout;         /* index into LAYOUTS, the one PACKED */
    unsigned s_mul;     /* BS_MUL_CFG.BS_MUL_SHIFT_VALUE */
    unsigned s_neg;     /* DATA_FORMAT.BS_MUL_SHIFT_VALUE_NEG */
    unsigned data_use;  /* RDMA_BRDMA_CFG.BRDMA_DATA_USE */
    int xlo, xhi, wlo, whi, blo, bhi;
    int cmode;          /* 0: C=1; 1: uniform [clo,chi]; 2: ramp clo + c*(chi-clo)/(C-1) */
    int clo, chi;
    float conv_scale;   /* the OUT_CVT's scale */
    int zx, zo;
    int bhalf;          /* B in [-bhalf, bhalf] per channel (0: B = 0) */
    int ow_src;         /* -1: BS_OW_CFG as emitted; 0 or 1: its OW_SRC bit (bit 0) */
    int bn;             /* 1: the BN stage takes a second per-channel multiplier E from NRDMA */
    unsigned bn_s, bn_sn;   /* BN_MUL_CFG shift (non-negative) and DATA_FORMAT[15:10] (negative) */
    int emode;          /* 0: E = 2^(c % 11); 1: E uniform [1, 32767]; 2: E = 2^14 */
} arm_t;

static int g_ctrl_bad = 0;

static void run_arm(int fd, const shape_t *s, const arm_t *a, uint64_t seed)
{
    const int C = s->C, IH = s->IH, IW = s->IW, KH = s->KH, KW = s->KW;
    const int OH = out_dim(IH, KH, s->sy, s->pt), OW = out_dim(IW, KW, s->sx, s->pl);
    const int Cpad = rup(C, DW_G), IHj = IH < 4 ? 4 : IH;
    const size_t ni = (size_t)C * IH * IW, nw = (size_t)C * KH * KW, no = (size_t)C * OH * OW;
    int8_t *x = malloc(ni), *w = malloc(nw), *got = malloc(no);
    int32_t *bias = malloc(C * sizeof(int32_t)), *Afold = malloc(C * sizeof(int32_t));
    int16_t *Cm = malloc(C * sizeof(int16_t)), *Bm = calloc(C, sizeof(int16_t));
    int64_t *acc = malloc(no * 8), *wsum = malloc(no * 8);
    uint8_t *coeff = NULL;
    size_t coeff_bytes = (size_t)(Cpad / 8) * 64 + PADB;
    rocket_bo guard = {0}, in_bo = {0}, wt_bo = {0}, rc_bo = {0}, bs_bo = {0}, out_bo = {0}, bn_bo = {0};
    int16_t *Em = malloc(C * sizeof(int16_t));
    uint64_t regs[256] = {0};
    unsigned mul, shift, n = 0;
    int rc = -1;
    uint32_t w_bscfg = 0, w_mulcfg = 0, w_fmt = 0, w_brdma = 0;

    tf_fill_i8(x, ni, seed + 1, a->xlo, a->xhi);
    tf_fill_i8(w, nw, seed + 2, a->wlo, a->whi);
    tf_fill_i32(bias, C, seed + 3, a->blo, a->bhi);
    {
        int32_t tmp[4096];
        tf_fill_i32(tmp, C, seed + 4, a->clo, a->chi);
        for (int c = 0; c < C; c++) {
            int v = a->cmode == 0 ? 1
                  : a->cmode == 1 ? tmp[c]
                  : a->clo + (int)((int64_t)c * (a->chi - a->clo) / (C > 1 ? C - 1 : 1));
            Cm[c] = (int16_t)v;
        }
    }
    if (a->bhalf) {
        int32_t tb[4096];
        tf_fill_i32(tb, C, seed + 5, -a->bhalf, a->bhalf);
        for (int c = 0; c < C; c++) Bm[c] = (int16_t)tb[c];
    }
    {
        int32_t te[4096];
        tf_fill_i32(te, C, seed + 6, 1, 32767);
        for (int c = 0; c < C; c++)
            Em[c] = (int16_t)(a->emode == 0 ? (1 << (c % 11)) : a->emode == 1 ? te[c] : (1 << 14));
    }
    for (int c = 0; c < C; c++) {
        int32_t sw = 0;
        for (int k = 0; k < KH * KW; k++) sw += w[(size_t)c * KH * KW + k];
        Afold[c] = bias[c] - a->zx * sw;
    }
    /* acc = sum xp*w, xp the input or the zero point off the image (what the CNA forms) */
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t p = 0, sm = 0;
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++) {
                        int ih = oh * s->sy + kh - s->pt, iw = ow * s->sx + kw - s->pl;
                        int xp = (ih >= 0 && ih < IH && iw >= 0 && iw < IW)
                                 ? x[((size_t)c * IH + ih) * IW + iw] : a->zx;
                        p += (int64_t)xp * w[((size_t)c * KH + kh) * KW + kw];
                        sm += xp;
                    }
                acc[((size_t)c * OH + oh) * OW + ow] = p;
                wsum[((size_t)c * OH + oh) * OW + ow] = sm;
            }
    requant_params(a->conv_scale, &mul, &shift);

    if (rocket_bo_alloc32(fd, 4096, &guard) ||
        rocket_bo_alloc32(fd, (size_t)C * IHj * IW + PADB, &in_bo) ||
        rocket_bo_alloc32(fd, (size_t)Cpad * KH * KW + PADB, &wt_bo) ||
        rocket_bo_alloc32(fd, 256 * sizeof(uint64_t), &rc_bo) ||
        rocket_bo_alloc32(fd, coeff_bytes, &bs_bo) ||
        rocket_bo_alloc32(fd, (size_t)C * OH * OW + PADB, &out_bo) ||
        rocket_bo_alloc32(fd, coeff_bytes, &bn_bo)) {
        fprintf(stderr, "  BO alloc failed\n");
        goto out;
    }
    rocket_bo_prep(fd, &in_bo, 1, 0);
    memset(in_bo.ptr, 0, in_bo.size);
    for (int c = 0; c < C; c++)
        for (int ih = 0; ih < IH; ih++)
            for (int iw = 0; iw < IW; iw++)
                ((int8_t *)in_bo.ptr)[feature_data(C, IHj, IW, 16, c + 1, ih + 1, iw + 1)] =
                    x[((size_t)c * IH + ih) * IW + iw];
    rocket_bo_fini(fd, &in_bo);
    rocket_bo_prep(fd, &wt_bo, 1, 0);
    memset(wt_bo.ptr, 0, wt_bo.size);
    for (int c = 0; c < C; c++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++)
                ((int8_t *)wt_bo.ptr)[weight_conv_dw_int8(C, KH, KW, DW_G, c + 1, kh + 1, kw + 1)] =
                    w[((size_t)c * KH + kh) * KW + kw];
    rocket_bo_fini(fd, &wt_bo);
    rocket_bo_prep(fd, &bs_bo, 1, 0);
    memset(bs_bo.ptr, 0, bs_bo.size);
    if (a->patch) pack_coeff(&LAYOUTS[a->layout], C, Afold, Bm, Cm, (uint8_t *)bs_bo.ptr);
    else memcpy(bs_bo.ptr, Afold, (size_t)C * 4);
    coeff = malloc(bs_bo.size);
    memcpy(coeff, bs_bo.ptr, bs_bo.size);
    rocket_bo_fini(fd, &bs_bo);
    rocket_bo_prep(fd, &bn_bo, 1, 0);
    memset(bn_bo.ptr, 0, bn_bo.size);
    if (a->bn) {
        int32_t *zA = calloc(C, sizeof(int32_t));
        pack_coeff(&LAYOUTS[0], C, zA, Bm, Em, (uint8_t *)bn_bo.ptr);   /* A2 = 0, C slot = E */
        free(zA);
    }
    rocket_bo_fini(fd, &bn_bo);

    {
        /* w_scale 1 and out_scale 1/conv_scale: the emitter derives the OUT_CVT pair
         * from in*w/out, the model from conv_scale, both through the same formula. */
        conv_params_t p = {
            .ic = C, .ih = IHj, .iw = IW, .oc = C, .oh = OH, .ow = OW,
            .kh = KH, .kw = KW, .stride_y = s->sy, .stride_x = s->sx,
            .dil_y = 1, .dil_x = 1, .pad_top = s->pt, .pad_left = s->pl,
            .input_dma = (uint32_t)in_bo.dma_address, .weights_dma = (uint32_t)wt_bo.dma_address,
            .output_dma = (uint32_t)out_bo.dma_address, .tasks = regs,
            .dw_group = DW_G, .int8_out = 1,
            .in_scale = a->conv_scale, .w_scale = 1.0f, .out_scale = 1.0f,
            .input_zero_point = a->zx + 0x80, .output_zero_point = a->zo + 0x80,
            .weight_zero_point = 0x80,
            .bias_dma = (uint32_t)bs_bo.dma_address,
        };
        int g = gen_conv2d_dw_int8(&p);
        if (g != 0) { fprintf(stderr, "  gen_conv2d_dw_int8 failed (%d)\n", g); goto out; }
        n = p.task_count;
        if (a->patch) {
            int kf = reg_find(regs, n, OP_REG_DPU, DPU_DATA_FORMAT);
            if (kf < 0) { fprintf(stderr, "  no DATA_FORMAT word\n"); goto out; }
            uint32_t fmt = (reg_get(regs, kf) & ~(0x3fu << 4)) | ((a->s_neg & 0x3fu) << 4);
            reg_set(regs, n, OP_REG_DPU, DPU_DATA_FORMAT, fmt);
            reg_set(regs, n, OP_REG_DPU, DPU_BS_CFG, (2u << 16) | (1u << 8) | (1u << 6));
            reg_set(regs, n, OP_REG_DPU, DPU_BS_MUL_CFG, ((a->s_mul & 0x3fu) << 8) | 1u);
            reg_set(regs, n, OP_REG_DPU_RDMA_P, RDMA_BRDMA_CFG_P, (a->data_use & 0xfu) << 1);
            if (a->bn) {
                int kf2 = reg_find(regs, n, OP_REG_DPU, DPU_DATA_FORMAT);
                uint32_t fmt2 = (reg_get(regs, kf2) & ~(0x3fu << 10)) | ((a->bn_sn & 0x3fu) << 10);
                reg_set(regs, n, OP_REG_DPU, DPU_DATA_FORMAT, fmt2);
                reg_set(regs, n, OP_REG_DPU, DPU_BN_CFG, (2u << 16) | (1u << 8) | (1u << 6));
                reg_set(regs, n, OP_REG_DPU, DPU_BN_MUL_CFG, ((a->bn_s & 0x3fu) << 8) | 1u);
                reg_set(regs, n, OP_REG_DPU_RDMA_P, 0x5028, 7u << 1);
                reg_set(regs, n, OP_REG_DPU_RDMA_P, 0x502C, (uint32_t)bn_bo.dma_address);
            }
            if (a->ow_src >= 0) {
                int ko = reg_find(regs, n, OP_REG_DPU, DPU_BS_OW_CFG);
                if (ko >= 0)
                    reg_set(regs, n, OP_REG_DPU, DPU_BS_OW_CFG,
                            (reg_get(regs, ko) & ~1u) | (uint32_t)(a->ow_src & 1));
            }
        }
        w_bscfg  = reg_get(regs, reg_find(regs, n, OP_REG_DPU, DPU_BS_CFG));
        w_mulcfg = reg_get(regs, reg_find(regs, n, OP_REG_DPU, DPU_BS_MUL_CFG));
        w_fmt    = reg_get(regs, reg_find(regs, n, OP_REG_DPU, DPU_DATA_FORMAT));
        w_brdma  = reg_get(regs, reg_find(regs, n, OP_REG_DPU_RDMA_P, RDMA_BRDMA_CFG_P));

        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, regs, (size_t)n * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);
        if (tf_sentinel_bo(fd, &out_bo)) { fprintf(stderr, "  sentinel failed\n"); goto out; }
        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo.dma_address, .regcmd_count = n };
        uint32_t in_h[] = { in_bo.handle, wt_bo.handle, bs_bo.handle, rc_bo.handle, bn_bo.handle };
        uint32_t out_h[] = { out_bo.handle };
        if (rocket_submit_tasks(fd, &task, 1, in_h, 5, out_h, 1)) {
            fprintf(stderr, "  submit failed\n"); goto out;
        }
    }
    if (rocket_bo_prep(fd, &out_bo, 0, 2000000000ULL)) { rc = -2; goto out; }
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                got[((size_t)c * OH + oh) * OW + ow] =
                    ((const int8_t *)out_bo.ptr)[feature_data(C, OH, OW, 16, c + 1, oh + 1, ow + 1)];
    rocket_bo_fini(fd, &out_bo);
    rc = 0;
out:
    rocket_bo_free(fd, &bn_bo);
    rocket_bo_free(fd, &out_bo); rocket_bo_free(fd, &bs_bo); rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &wt_bo); rocket_bo_free(fd, &in_bo); rocket_bo_free(fd, &guard);

    if (rc) {
        printf("  %-8s %-26s %s\n", s->name, a->label, rc == -2 ? "TIMEOUT" : "FAILED");
        if (!a->patch) g_ctrl_bad = 1;
    } else if (!a->patch) {
        long eq = 0, sent = 0, maxd = 0;
        for (size_t o = 0; o < no; o++) {
            int c = (int)(o / ((size_t)OH * OW));
            int f = requant_apply_zp(acc[o] + Afold[c], mul, shift, a->zo);
            if ((uint8_t)got[o] == TF_SENTINEL_BYTE) sent++;
            long d = labs((long)got[o] - f);
            if (!d) eq++;
            if (d > maxd) maxd = d;
        }
        printf("  %-8s %-26s BS_CFG=0x%05x MUL_CFG=0x%04x FMT=0x%08x BRDMA=0x%x | "
               "per-tensor model exact %ld/%zu max %ld sent %ld\n",
               s->name, a->label, w_bscfg, w_mulcfg, w_fmt, w_brdma, eq, no, maxd, sent);
        if (eq != (long)no) g_ctrl_bad = 1;
    } else if (a->bn) {
        /* BN arms: the established BS stage, then candidate readings of BN */
        long k_bn = 0, k_inert = 0, k_bnnosplit = 0, k_bn_c32 = 0, sent = 0, rail = 0;
        for (size_t o = 0; o < no; o++) {
            int c = (int)(o / ((size_t)OH * OW));
            int64_t v1 = epilogue(H_SPLIT, acc[o], Afold[c], Cm[c], a->s_mul, a->s_neg, 0, 0);
            int m_bn = requant_apply_zp(bn_stage(v1, Em[c], a->bn_s, a->bn_sn), mul, shift, a->zo);
            if (m_bn == got[o]) k_bn++;
            if (m_bn == 127 || m_bn == -128) rail++;
            if (requant_apply_zp(v1, mul, shift, a->zo) == got[o]) k_inert++;
            if (requant_apply_zp(bn_stage(v1, Em[c], a->bn_s, a->bn_s), mul, shift, a->zo) == got[o])
                k_bnnosplit++;
            if (requant_apply_zp(bn_stage(v1, 0, a->bn_s, a->bn_sn), mul, shift, a->zo) == got[o])
                k_bn_c32++;   /* E read as 0 (the B slot, empty) */
            if ((uint8_t)got[o] == TF_SENTINEL_BYTE) sent++;
        }
        printf("  %-8s %-26s BS s=%u/%u BN s=%u/%u of %zu (rail %ld, sent %ld) | BS*BN %ld  "
               "BN inert %ld  BN no split %ld  BN E=0 %ld\n", s->name, a->label, a->s_mul, a->s_neg,
               a->bn_s, a->bn_sn, no, rail, sent, k_bn, k_inert, k_bnnosplit, k_bn_c32);
    } else {
        /* scored at each candidate shift: the MUL_CFG field's and the NEG field's */
        long cnt[2][NLAYOUT][NHYP] = {{{0}}}, sent = 0, over31 = 0, satout = 0;
        const unsigned sv[2] = { a->s_mul, a->s_neg };
        const int nsv = a->s_mul == a->s_neg ? 1 : 2;
        int32_t *LA = malloc(C * 4);
        int16_t *LC = malloc(C * 2), *LB = malloc(C * 2);
        for (int k = 0; k < nsv; k++)
            for (int l = 0; l < NLAYOUT; l++) {
                read_coeff(&LAYOUTS[l], C, coeff, LA, LC, LB);
                for (size_t o = 0; o < no; o++) {
                    int c = (int)(o / ((size_t)OH * OW));
                    for (int h = 0; h < NHYP; h++) {
                        int64_t v = epilogue(h, acc[o], LA[c], LC[c], sv[k], a->s_neg, LB[c], wsum[o]);
                        if (requant_apply_zp(v, mul, shift, a->zo) == got[o]) cnt[k][l][h]++;
                    }
                }
            }
        for (size_t o = 0; o < no; o++) {
            int c = (int)(o / ((size_t)OH * OW));
            int64_t p = (acc[o] + Afold[c]) * (int64_t)Cm[c];
            if (p > INT32_MAX || p < INT32_MIN) over31++;
            int m = requant_apply_zp(epilogue(H_SPLIT, acc[o], Afold[c], Cm[c], a->s_mul, a->s_neg, 0, 0),
                                     mul, shift, a->zo);
            if (m == 127 || m == -128) satout++;
            if ((uint8_t)got[o] == TF_SENTINEL_BYTE) sent++;
        }
        printf("  %-8s %-26s BS_CFG=0x%05x MUL_CFG=0x%04x FMT=0x%08x BRDMA=0x%x of %zu (|p|>2^31 %ld, "
               "rail %ld, sent %ld)\n", s->name, a->label, w_bscfg, w_mulcfg, w_fmt, w_brdma, no,
               over31, satout, sent);
        for (int k = 0; k < nsv; k++)
            for (int l = 0; l < NLAYOUT; l++) {
                printf("      s=%-2u %-10s", sv[k], LAYOUTS[l].name);
                for (int h = 0; h < NHYP; h++) printf("  %-6s %6ld", HNAME[h], cnt[k][l][h]);
                printf("\n");
            }
        free(LA); free(LC); free(LB);
    }
    free(x); free(w); free(got); free(bias); free(Afold); free(Cm); free(Bm); free(acc); free(wsum); free(coeff); free(Em);
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU device; skipping\n"); return 77; }

    const shape_t shapes[] = {
        { "c64k3",  64, 12, 12, 3, 3, 1, 1, 1, 1 },
        { "c128k5", 128, 11,  9, 5, 5, 2, 2, 2, 2 },
    };
    /* label, patch, layout, s_mul, s_neg, data_use, x, w, bias ranges, cmode, clo, chi, scale, zx, zo,
     * B half-range, OW_SRC */
    const arm_t arms[] = {
        { "control (emitted)",       0, 0,  0,  0, 0, -128, 127, -128, 127, -6000, 6000, 0, 0, 0,
          0.0005f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "s0 C=1",                  1, 0,  0,  0, 7, -128, 127, -128, 127, -6000, 6000, 0, 0, 0,
          0.0005f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "s0 C in 1..6",            1, 0,  0,  0, 7, -128, 127, -128, 127, -6000, 6000, 1, 1, 6,
          0.0001f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "s14 C in 8192..32767",    1, 0, 14, 14, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0003f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "s14 C ramp 1..32767",     1, 0, 14, 14, 7, -128, 127, -128, 127, -6000, 6000, 2, 1, 32767,
          0.0005f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "s8 ties, identity cvt",   1, 0,  8,  8, 7,   -4,   4,   -4,   4,   -50,   50, 1, 1, 200,
          1.0f, 0, 0, 0, -1, 0, 0, 0, 0 },
        { "s20 wide, C 16384..32767", 1, 0, 20, 20, 7, -128, 127, -128, 127, -400000, 400000, 1, 16384, 32767,
          0.005f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "s14 MUL_CFG only",        1, 0, 14,  0, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0003f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "s14 FMT_NEG only",        1, 0,  0, 14, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0003f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "s14 layout 48",           1, 2, 14, 14, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0003f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "split 14/6",              1, 0, 14,  6, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0003f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "split 6/14",              1, 0,  6, 14, 7, -128, 127, -128, 127, -6000, 6000, 1, 64, 255,
          0.0003f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "s0 sat32 after shift",    1, 0,  0,  0, 7, -128, 127, -128, 127, -400000, 400000, 1, 16384, 32767,
          3e-8f, -3, 5, 0, -1, 0, 0, 0, 0 },
        { "s14 B in +-20, OW_SRC as", 1, 0, 14, 14, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0003f, -3, 5, 20, -1, 0, 0, 0, 0 },
        { "s14 B in +-20, OW_SRC 1", 1, 0, 14, 14, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0003f, -3, 5, 20, 1, 0, 0, 0, 0 },
        { "BN identity E=2^14 s14",  1, 0,  6,  6, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0000015f, -3, 5, 0, -1, 1, 14, 14, 2 },
        { "BN E=2^(c%11) s10",       1, 0,  6,  6, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0000015f, -3, 5, 0, -1, 1, 10, 10, 0 },
        { "BN E uniform s15",        1, 0,  6,  6, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0000030f, -3, 5, 0, -1, 1, 15, 15, 1 },
        { "BN split 10/4",           1, 0,  6,  6, 7, -128, 127, -128, 127, -6000, 6000, 1, 8192, 32767,
          0.0000015f, -3, 5, 0, -1, 1, 10, 4, 0 },
        { "control again",           0, 0,  0,  0, 0, -128, 127, -128, 127, -6000, 6000, 0, 0, 0,
          0.0005f, -3, 5, 0, -1, 0, 0, 0, 0 },
    };
    printf("dw_perc_probe: per-channel BS multiplier on the RK3588 int8-out depthwise program\n");
    printf("  (each patched arm: how many elements each epilogue x layout explains)\n");
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++)
        for (size_t ai = 0; ai < sizeof(arms) / sizeof(arms[0]); ai++)
            run_arm(fd, &shapes[si], &arms[ai], 1000 + 17 * si + ai);
    rocket_close(fd);
    if (g_ctrl_bad) { printf("CONTROL NOT EXACT: no row can be read\n"); return 1; }
    return 0;
}
