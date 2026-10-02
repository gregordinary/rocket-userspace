// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_i8out_probe.c — RK3588 board probe: does the DIRECT int8 convolution compute with
 * Mesa's int8-OUTPUT writer, and does that program carry CPEND and the per-channel BS
 * multiplier?
 *
 * The library's direct int8 conv writes the raw int32 accumulator (size_e 7, SURF_ADD x8),
 * which the host de-scatters and requantizes, and it cannot carry an asymmetric weight zero
 * point: clearing OD_BYPASS on that writer makes every value wrong [HW sweep, RK1,
 * tests/cpend_wzp_probe]. Mesa runs Teflon's uint8 direct convs on a different writer, the
 * one the int8 depthwise entry uses: QD_EN 1, DATA_FORMAT 0, the per-OC int32 bias in the BS
 * ALU via BRDMA, the OUT_CVT requant, OD_BYPASS clear with OW_OP = 0x80 - Z_w, and for a
 * DIRECT conv size_e 1 and SURF_ADD = OH*OW*2 [source-confirmed, rkt_regcmd.c, rkt_task.c].
 * gen_conv2d_int8 emits that program at conv_params_t.int8_out = 1.
 *
 * Every arm runs one job twice, over an output BO prefilled with 0xAA and then 0x55, reads
 * the int8 C2=16 cube, and scores each element. An element the two runs disagree on was
 * not written (or not deterministically). The rows report magnitudes:
 *
 *   F        the intended function: TFLite's int32 accumulator
 *            bias + sum (xp - zx)(w - zw), xp the input or zx off the image, then the
 *            OUT_CVT model (tests/requant_model.h, ties to even).
 *   exact / off1 / off>1 / max   against F, over written elements
 *   unwr     elements the two prefills disagree on
 *   mset     the multiset overlap of the surface with F: how many of F's values occur in
 *            it at all. mset high with exact low is a PLACEMENT defect, not arithmetic.
 *   bad b/i  wrong elements whose window reaches the border / lies inside
 *   H(v)     the CPEND model at a candidate operand v: sum xp*w + v*S + A, A the folded
 *            bias bias - zx*sum(w - zw), S the window sum of the CNA's input. S_all counts
 *            the channel padding the program runs (the CNA pads every programmed channel
 *            with zx at the border); S_real counts the real channels only. v is read as
 *            signed 16 (the depthwise program's reading) and 0 (CPEND inert).
 *
 * Per-channel arms patch the BS stage as tests/dw_perc_probe does (BRDMA data use 7 over a
 * 64-byte group per 8 OC: int32 A at 0, int16 B at 32, int16 C at 48; BS_CFG with the
 * multiplier live; the shift in BS_MUL_CFG[13:8] for a non-negative product and in
 * DATA_FORMAT[9:4] for a negative one; BS_OW_CFG.OW_SRC for B) and are scored against:
 *
 *   split    (P + v*S + A[c]) * C[c], rounded half-even at s or sn by the product's sign
 *   B*S      (P + A[c] + B[c]*S) * C[c], same rounding (B as a per-channel CPEND)
 *   inert    P + v*S + A[c], the multiplier not applied
 *
 * then the unchanged OUT_CVT. Output zero points sit off the -128 rail except in one arm
 * (MobileDet's direct convs all run at uint8 zp 0, int8 -128), since at the rail a wrong
 * negative path passes.
 *
 * A control arm (the emitted symmetric program at a shape the first arm has shown exact)
 * runs first and last. Exit 0 when every arm ran (a probe: the verdict is the table), 1 when
 * a control is not exact (no other row can be read), 77 without a device.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "test_fill.h"
#include "requant_model.h"

#define PADB NPU_CBUF_BANK_SIZE
#define OP_REG_DPU_RDMA_P (BLOCK_DPU_RDMA | PC_OP_01)
#define RDMA_BRDMA_CFG_P   0x501C

static int rup(int a, int b) { return ((a + b - 1) / b) * b; }
static int out_dim(int in, int k, int s, int p) { return (in + 2 * p - k) / s + 1; }

typedef struct {
    const char *name;
    int IC, IH, IW, OC, KH, KW, sy, sx, pt, pl;
    int oc_prog;    /* the program's OC: 0 -> rup(OC, 32) with zero kernels; else as given
                     * (a partial 32-kernel group, packed as Mesa packs it) */
} shape_t;

typedef struct {
    const char *label;
    int zx, zw, zo;          /* int8-domain zero points */
    float conv_scale;
    int op_set; uint32_t op; /* 1: override BS_OW_OP (else the emitted 0x80 - (zw + 0x80)) */
    int perc;                /* 1: the per-channel BS multiplier patch */
    unsigned s_mul, s_neg;
    int clo, chi;            /* C uniform in [clo, chi] */
    int bhalf;               /* B uniform in [-bhalf, bhalf]; 0 -> B = 0 */
    int ow_src;              /* -1 as emitted, else BS_OW_CFG bit 0 */
    int se, sm;              /* -1 as emitted, else patched size_e / SURF_ADD multiple */
    int control;
    int centered;            /* 1: x in zx +- 100 and w in zw +- 100 (clamped to int8), so a
                              * nonzero weight zero point does not push the surface onto the rail */
} arm_t;

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

/* Mesa's direct weight packing (rkt_coefs.c): oc1, ic1, kh, kw, oc2, ic2 with the LAST
 * oc group holding only its real kernels. For a whole-group OC this is weight_conv_int8. */
static size_t wt_index(int OCp, int ICp, int KH, int KW, int oc, int ic, int kh, int kw)
{
    int g = oc / 32, ng = OCp - 32 * g < 32 ? OCp - 32 * g : 32;
    size_t base = (size_t)32 * g * ICp * KH * KW;
    return base + ((((size_t)(ic / 32) * KH + kh) * KW + kw) * ng + oc % 32) * 32 + ic % 32;
}

static int64_t sat32(int64_t v)
{
    if (v > INT32_MAX) return INT32_MAX;
    if (v < INT32_MIN) return INT32_MIN;
    return v;
}

static int cmp_i8(const void *a, const void *b) { return *(const int8_t *)a - *(const int8_t *)b; }

static int g_ctrl_bad = 0;

/* An OUT_CVT scale that spreads a K-deep accumulator of uniform int8 operands (std about
 * sqrt(K) * 74^2) over the int8 range with a little saturation: 0.0006 at K = 576. */
static float auto_scale(const shape_t *s)
{
    return 0.0006f * (float)sqrt(576.0 / (double)(s->IC * s->KH * s->KW));
}

/* One device run of the program at one prefill byte; got [OC][OH][OW] (real OC). */
static int run_once(int fd, const shape_t *s, const arm_t *a, const int8_t *x, const int8_t *w,
                    const uint8_t *coeff, size_t coeff_bytes, uint8_t fill, int8_t *got,
                    uint32_t *words)
{
    const int ICp = rup(s->IC, 32), OCp = s->oc_prog ? s->oc_prog : rup(s->OC, 32);
    const int IHj = s->IH < 4 ? 4 : s->IH;
    const int OH = out_dim(s->IH, s->KH, s->sy, s->pt), OW = out_dim(s->IW, s->KW, s->sx, s->pl);
    const int OCc = rup(OCp, 16);
    rocket_bo guard = {0}, in_bo = {0}, wt_bo = {0}, rc_bo = {0}, bs_bo = {0}, out_bo = {0};
    uint64_t regs[256] = {0};
    int ret = -1;

    if (rocket_bo_alloc32(fd, 4096, &guard) ||
        rocket_bo_alloc32(fd, (size_t)ICp * IHj * s->IW + PADB, &in_bo) ||
        rocket_bo_alloc32(fd, (size_t)OCp * ICp * s->KH * s->KW + PADB, &wt_bo) ||
        rocket_bo_alloc32(fd, 256 * sizeof(uint64_t), &rc_bo) ||
        rocket_bo_alloc32(fd, coeff_bytes + PADB, &bs_bo) ||
        rocket_bo_alloc32(fd, (size_t)OCc * OH * OW + PADB, &out_bo)) {
        fprintf(stderr, "  BO alloc failed\n");
        goto out;
    }
    rocket_bo_prep(fd, &in_bo, 1, 0);
    memset(in_bo.ptr, 0, in_bo.size);
    for (int ic = 0; ic < s->IC; ic++)
        for (int ih = 0; ih < s->IH; ih++)
            for (int iw = 0; iw < s->IW; iw++)
                ((int8_t *)in_bo.ptr)[feature_data(ICp, IHj, s->IW, 16, ic + 1, ih + 1, iw + 1)] =
                    x[((size_t)ic * s->IH + ih) * s->IW + iw];
    rocket_bo_fini(fd, &in_bo);
    rocket_bo_prep(fd, &wt_bo, 1, 0);
    memset(wt_bo.ptr, 0, wt_bo.size);
    for (int oc = 0; oc < s->OC && oc < OCp; oc++)
        for (int ic = 0; ic < s->IC; ic++)
            for (int kh = 0; kh < s->KH; kh++)
                for (int kw = 0; kw < s->KW; kw++)
                    ((int8_t *)wt_bo.ptr)[wt_index(OCp, ICp, s->KH, s->KW, oc, ic, kh, kw)] =
                        w[(((size_t)oc * s->IC + ic) * s->KH + kh) * s->KW + kw];
    rocket_bo_fini(fd, &wt_bo);
    rocket_bo_prep(fd, &bs_bo, 1, 0);
    memset(bs_bo.ptr, 0, bs_bo.size);
    memcpy(bs_bo.ptr, coeff, coeff_bytes);
    rocket_bo_fini(fd, &bs_bo);

    {
        conv_params_t p = {
            .ic = ICp, .ih = IHj, .iw = s->IW, .oc = OCp, .oh = OH, .ow = OW,
            .kh = s->KH, .kw = s->KW, .stride_y = s->sy, .stride_x = s->sx,
            .dil_y = 1, .dil_x = 1, .pad_top = s->pt, .pad_left = s->pl,
            .input_dma = (uint32_t)in_bo.dma_address, .weights_dma = (uint32_t)wt_bo.dma_address,
            .output_dma = (uint32_t)out_bo.dma_address, .tasks = regs,
            .int8_out = 1,
            .in_scale = a->conv_scale, .w_scale = 1.0f, .out_scale = 1.0f,
            .input_zero_point = a->zx + 0x80, .output_zero_point = a->zo + 0x80,
            .weight_zero_point = a->zw + 0x80,
            .bias_dma = (uint32_t)bs_bo.dma_address,
        };
        int g = gen_conv2d_int8(&p);
        if (g != 0) { fprintf(stderr, "  gen_conv2d_int8 failed (%d)\n", g); goto out; }
        unsigned n = p.task_count;
        if (a->op_set) reg_set(regs, n, OP_REG_DPU, DPU_BS_OW_OP, a->op & 0xffffu);
        if (a->perc) {
            int kf = reg_find(regs, n, OP_REG_DPU, DPU_DATA_FORMAT);
            uint32_t fmt = (reg_get(regs, kf) & ~(0x3fu << 4)) | ((a->s_neg & 0x3fu) << 4);
            reg_set(regs, n, OP_REG_DPU, DPU_DATA_FORMAT, fmt);
            reg_set(regs, n, OP_REG_DPU, DPU_BS_CFG, (2u << 16) | (1u << 8) | (1u << 6));
            reg_set(regs, n, OP_REG_DPU, DPU_BS_MUL_CFG, ((a->s_mul & 0x3fu) << 8) | 1u);
            reg_set(regs, n, OP_REG_DPU_RDMA_P, RDMA_BRDMA_CFG_P, 7u << 1);
        }
        {
            int ko = reg_find(regs, n, OP_REG_DPU, DPU_BS_OW_CFG);
            uint32_t v = reg_get(regs, ko);
            if (a->ow_src >= 0) v = (v & ~1u) | (uint32_t)(a->ow_src & 1);
            if (a->se >= 0) {
                unsigned se = (unsigned)a->se & 7u;
                v = (v & ~((7u << 8) | (7u << 5) | (7u << 2))) | (se << 8) | (se << 5) | (se << 2);
            }
            reg_set(regs, n, OP_REG_DPU, DPU_BS_OW_CFG, v);
            if (a->sm >= 0)
                reg_set(regs, n, OP_REG_DPU, DPU_SURFACE_ADD,
                        (((uint32_t)OH * OW * (uint32_t)a->sm) & 0xFFFFFFFu) << 4);
        }
        if (words) {
            words[0] = reg_get(regs, reg_find(regs, n, OP_REG_CORE, CORE_MISC_CFG));
            words[1] = reg_get(regs, reg_find(regs, n, OP_REG_DPU, DPU_DATA_FORMAT));
            words[2] = reg_get(regs, reg_find(regs, n, OP_REG_DPU, DPU_BS_CFG));
            words[3] = reg_get(regs, reg_find(regs, n, OP_REG_DPU, DPU_BS_OW_CFG));
            words[4] = reg_get(regs, reg_find(regs, n, OP_REG_DPU, DPU_BS_OW_OP));
            words[5] = reg_get(regs, reg_find(regs, n, OP_REG_DPU, DPU_SURFACE_ADD));
            words[6] = reg_get(regs, reg_find(regs, n, OP_REG_CNA, CNA_PAD_CON1));
        }
        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, regs, (size_t)n * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);
        rocket_bo_prep(fd, &out_bo, 1, 0);
        memset(out_bo.ptr, fill, out_bo.size);
        rocket_bo_fini(fd, &out_bo);
        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo.dma_address, .regcmd_count = n };
        uint32_t in_h[] = { in_bo.handle, wt_bo.handle, bs_bo.handle, rc_bo.handle };
        uint32_t out_h[] = { out_bo.handle };
        if (rocket_submit_tasks(fd, &task, 1, in_h, 4, out_h, 1)) {
            fprintf(stderr, "  submit failed\n"); goto out;
        }
    }
    if (rocket_bo_prep(fd, &out_bo, 0, 2000000000ULL)) { ret = -2; goto out; }
    for (int oc = 0; oc < s->OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                got[((size_t)oc * OH + oh) * OW + ow] =
                    ((const int8_t *)out_bo.ptr)[feature_data(OCc, OH, OW, 16, oc + 1, oh + 1, ow + 1)];
    rocket_bo_fini(fd, &out_bo);
    ret = 0;
out:
    rocket_bo_free(fd, &out_bo); rocket_bo_free(fd, &bs_bo); rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &wt_bo); rocket_bo_free(fd, &in_bo); rocket_bo_free(fd, &guard);
    return ret;
}

static void run_arm(int fd, const shape_t *s, const arm_t *a_in, uint64_t seed)
{
    arm_t a_loc = *a_in;
    if (a_loc.conv_scale <= 0.0f) a_loc.conv_scale = auto_scale(s);
    const arm_t *a = &a_loc;
    const int IC = s->IC, OC = s->OC, KH = s->KH, KW = s->KW;
    const int ICp = rup(IC, 32), OCp = s->oc_prog ? s->oc_prog : rup(OC, 32);
    const int OH = out_dim(s->IH, KH, s->sy, s->pt), OW = out_dim(s->IW, KW, s->sx, s->pl);
    const size_t ni = (size_t)IC * s->IH * s->IW, nw = (size_t)OC * IC * KH * KW;
    const size_t no = (size_t)OC * OH * OW;
    int8_t *x = malloc(ni), *w = malloc(nw), *g1 = malloc(no), *g2 = malloc(no);
    int8_t *fs = malloc(no), *gs = malloc(no);
    int32_t *bias = malloc(OC * 4), *A = malloc(OC * 4);
    int16_t *Cm = calloc(OC, 2), *Bm = calloc(OC, 2);
    int64_t *P = malloc(no * 8), *Sr = malloc(no * 8), *Sa = malloc(no * 8);
    char *bord = malloc(no);
    const size_t coeff_bytes = a->perc ? (size_t)rup(OCp, 8) / 8 * 64 : (size_t)OCp * 4;
    uint8_t *coeff = calloc(coeff_bytes, 1);
    unsigned mul, shift;
    uint32_t words[7] = {0};

    if (a->centered) {
        tf_fill_i8(x, ni, seed + 1, a->zx - 100 < -128 ? -128 : a->zx - 100,
                   a->zx + 100 > 127 ? 127 : a->zx + 100);
        tf_fill_i8(w, nw, seed + 2, a->zw - 100 < -128 ? -128 : a->zw - 100,
                   a->zw + 100 > 127 ? 127 : a->zw + 100);
    } else {
        tf_fill_i8(x, ni, seed + 1, -128, 127);
        tf_fill_i8(w, nw, seed + 2, -128, 127);
    }
    tf_fill_i32(bias, OC, seed + 3, -6000, 6000);
    if (a->perc) {
        int32_t t[4096];
        tf_fill_i32(t, OC, seed + 4, a->clo, a->chi);
        for (int c = 0; c < OC; c++) Cm[c] = (int16_t)t[c];
        if (a->bhalf) {
            tf_fill_i32(t, OC, seed + 5, -a->bhalf, a->bhalf);
            for (int c = 0; c < OC; c++) Bm[c] = (int16_t)t[c];
        }
    }
    for (int oc = 0; oc < OC; oc++) {
        int64_t sw = 0;
        for (size_t k = 0; k < (size_t)IC * KH * KW; k++) sw += w[(size_t)oc * IC * KH * KW + k] - a->zw;
        A[oc] = (int32_t)(bias[oc] - (int64_t)a->zx * sw);
    }
    if (a->perc) {
        for (int c = 0; c < OC; c++) {
            uint8_t *grp = coeff + (size_t)(c / 8) * 64;
            memcpy(grp + 4 * (c % 8), &A[c], 4);
            memcpy(grp + 32 + 2 * (c % 8), &Bm[c], 2);
            memcpy(grp + 48 + 2 * (c % 8), &Cm[c], 2);
        }
    } else {
        memcpy(coeff, A, (size_t)OC * 4);
    }
    /* host terms */
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t p = 0, sr = 0;
                int offimg = 0;
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++) {
                        int ih = oh * s->sy + kh - s->pt, iw = ow * s->sx + kw - s->pl;
                        int in = ih >= 0 && ih < s->IH && iw >= 0 && iw < s->IW;
                        if (!in) offimg++;
                        for (int ic = 0; ic < IC; ic++) {
                            int xp = in ? x[((size_t)ic * s->IH + ih) * s->IW + iw] : a->zx;
                            p += (int64_t)xp * w[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
                            sr += xp;
                        }
                    }
                size_t o = ((size_t)oc * OH + oh) * OW + ow;
                P[o] = p; Sr[o] = sr;
                Sa[o] = sr + (int64_t)(ICp - IC) * offimg * a->zx;
                bord[o] = offimg > 0;
            }
    requant_params(a->conv_scale, &mul, &shift);

    int rc = run_once(fd, s, a, x, w, coeff, coeff_bytes, 0xAA, g1, words);
    if (!rc) rc = run_once(fd, s, a, x, w, coeff, coeff_bytes, 0x55, g2, NULL);
    if (rc) {
        printf("  %-10s %-28s %s\n", s->name, a->label, rc == -2 ? "TIMEOUT" : "FAILED");
        if (a->control) g_ctrl_bad = 1;
        goto done;
    }
    {
        const int v = (int)(int16_t)(words[4] & 0xffffu);
        long ex = 0, off1 = 0, offm = 0, unwr = 0, bb = 0, bi = 0, maxd = 0, nb = 0;
        long hsr = 0, hsa = 0, h0 = 0, rail = 0;
        long k_split = 0, k_bs = 0, k_inert = 0, over31 = 0;
        for (size_t o = 0; o < no; o++) {
            int c = (int)(o / ((size_t)OH * OW));
            int64_t tfl = P[o] - (int64_t)a->zw * Sr[o] + A[c];
            int f = requant_apply_zp(tfl, mul, shift, a->zo);
            fs[o] = (int8_t)f;
            if (f == 127 || f == -128) rail++;
            nb += bord[o];
            if (g1[o] != g2[o]) { unwr++; continue; }
            int d = g1[o];
            if (requant_apply_zp(P[o] + (int64_t)v * Sr[o] + A[c], mul, shift, a->zo) == d) hsr++;
            if (requant_apply_zp(P[o] + (int64_t)v * Sa[o] + A[c], mul, shift, a->zo) == d) hsa++;
            if (requant_apply_zp(P[o] + A[c], mul, shift, a->zo) == d) h0++;
            if (a->perc) {
                int64_t pr = (P[o] + (int64_t)v * Sa[o] + A[c]) * (int64_t)Cm[c];
                if (pr > INT32_MAX || pr < INT32_MIN) over31++;
                int64_t sp = sat32(requant_round_shift(pr, pr >= 0 ? a->s_mul : a->s_neg));
                if (requant_apply_zp(sp, mul, shift, a->zo) == d) k_split++;
                int64_t pb = (P[o] + A[c] + (int64_t)Bm[c] * Sa[o]) * (int64_t)Cm[c];
                int64_t sb = sat32(requant_round_shift(pb, pb >= 0 ? a->s_mul : a->s_neg));
                if (requant_apply_zp(sb, mul, shift, a->zo) == d) k_bs++;
                if (requant_apply_zp(P[o] + (int64_t)v * Sa[o] + A[c], mul, shift, a->zo) == d) k_inert++;
                continue;
            }
            long ad = labs((long)d - f);
            if (ad > maxd) maxd = ad;
            if (!ad) { ex++; continue; }
            if (ad == 1) off1++; else offm++;
            if (bord[o]) bb++; else bi++;
        }
        memcpy(gs, g1, no);
        qsort(fs, no, 1, cmp_i8); qsort(gs, no, 1, cmp_i8);
        size_t i = 0, j = 0, ms = 0;
        while (i < no && j < no) {
            if (fs[i] < gs[j]) i++; else if (fs[i] > gs[j]) j++; else { ms++; i++; j++; }
        }
        printf("  %-10s %-28s MISC=0x%03x FMT=0x%08x BS=0x%05x OWCFG=0x%03x OP=0x%04x SURF=0x%x PAD=0x%08x\n",
               s->name, a->label, words[0], words[1], words[2], words[3], words[4], words[5], words[6]);
        if (!a->perc) {
            printf("      of %zu (rail %ld, border %ld): exact %ld off1 %ld off>1 %ld max %ld unwr %ld "
                   "mset %zu | bad border %ld interior %ld | H s16(%d) S_real %ld S_all %ld  inert %ld\n",
                   no, rail, nb, ex, off1, offm, maxd, unwr, ms, bb, bi, v, hsr, hsa, h0);
            if (a->control && ex != (long)no) g_ctrl_bad = 1;
        } else {
            printf("      of %zu (rail %ld, |p|>2^31 %ld): unwr %ld mset %zu | split(s=%u/%u) %ld  B*S %ld  "
                   "inert %ld  | H s16(%d) S_all %ld\n",
                   no, rail, over31, unwr, ms, a->s_mul, a->s_neg, k_split, k_bs, k_inert, v, hsa);
        }
    }
done:
    free(x); free(w); free(g1); free(g2); free(fs); free(gs); free(bias); free(A); free(Cm); free(Bm);
    free(P); free(Sr); free(Sa); free(bord); free(coeff);
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) { printf("SKIP: no NPU device\n"); return 77; }
    const char *only = argc > 1 ? argv[1] : NULL;   /* "geo", "cpend", "perc", "witness"; none = the first three */

    const shape_t shapes[] = {
        /* name        IC  IH  IW   OC KH KW sy sx pt pl oc_prog */
        { "c32k1",     32,  8,  8,  32, 1, 1, 1, 1, 0, 0, 0 },
        { "c64k3p1",   64, 10, 12,  64, 3, 3, 1, 1, 1, 1, 0 },
        { "c96k3s2",   96, 11,  9,  32, 3, 3, 2, 2, 1, 1, 0 },
        { "c32k5s2",   32, 12, 12,  64, 5, 5, 2, 2, 2, 2, 0 },
        { "c32oc160",  32,  9,  9, 160, 3, 3, 1, 1, 1, 1, 0 },
        { "c40k3oc160", 40, 10, 10, 160, 3, 3, 1, 1, 0, 0, 0 },   /* MobileDet's IC 40, pad 0 */
        { "big64",     64, 60, 56,  32, 3, 3, 1, 1, 1, 1, 0 },   /* 7 feature banks */
        { "c16p1",     16,  8,  8,  32, 3, 3, 1, 1, 1, 1, 0 },   /* padded IC under a CNA pad */
        { "oc48pad",   32,  8,  8,  48, 3, 3, 1, 1, 1, 1, 0 },   /* host-padded to 64 */
        { "oc48prog",  32,  8,  8,  48, 3, 3, 1, 1, 1, 1, 48 },  /* programmed at 48 */
        { "oc16prog",  32,  8,  8,  16, 1, 1, 1, 1, 0, 0, 16 },  /* programmed at 16 */
    };
    const int NS = (int)(sizeof shapes / sizeof shapes[0]);
    uint64_t seed = 0x18a0000ull;

    printf("conv_i8out_probe: the RK3588 direct int8 conv on Mesa's int8-out writer\n");

    /* control: the emitted symmetric program, at the smallest shape */
    const arm_t ctrl = { "control (emitted, zw 0)", 23, 0, -7, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 1, 0 };
    run_arm(fd, &shapes[0], &ctrl, seed);

    if (!only || !strcmp(only, "geo")) {
        printf(" -- geometry: symmetric weights, zx 23, zo -7 (and one arm at the -128 rail)\n");
        for (int si = 0; si < NS; si++) {
            const shape_t *s = &shapes[si];
            if (s->oc_prog) continue;   /* a partial group: the "witness" section */
            arm_t a = { "emitted (se 1, SURF x2)", 23, 0, -7, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0 };
            run_arm(fd, s, &a, seed + 100 * si);
        }
        {
            arm_t a = { "emitted, zo -128 (rail)", 23, 0, -128, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0 };
            run_arm(fd, &shapes[1], &a, seed + 7);
            arm_t b = { "emitted, zx -128 zo 0", -128, 0, 0, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0 };
            run_arm(fd, &shapes[1], &b, seed + 8);
        }
    }

    /* Witnesses, run only on request: each job below outlasts the driver's watchdog (one
     * kernel job timeout per run, two runs per arm) and leaves output unwritten [HW sweep,
     * RK1, 2026-09-27]. A program whose OC ends part way through a 32-kernel group, and the
     * depthwise writer's geometry (size_e 3, SURF_ADD x4) on the direct program. */
    if (only && !strcmp(only, "witness")) {
        printf(" -- witnesses (each job times out): a partial 32-kernel group; the depthwise geometry\n");
        for (int si = 0; si < NS; si++) {
            const shape_t *s = &shapes[si];
            if (!s->oc_prog) continue;
            arm_t a = { "emitted, partial group", 23, 0, -7, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0 };
            run_arm(fd, s, &a, seed + 100 * si);
        }
        arm_t a = { "se 3, SURF x4 (dw's)", 23, 0, -7, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0, -1, 3, 4, 0, 0 };
        run_arm(fd, &shapes[1], &a, seed + 9);
    }

    if (!only || !strcmp(only, "cpend")) {
        printf(" -- CPEND: OW_OP = -zw (emitted), zx 23, zo 5, data centred on the zero points\n");
        const int zws[] = { -62, -37, -11, 25, 90 };
        const int cs_idx[] = { 1, 2, 3, 4, 7 };
        for (int k = 0; k < 5; k++) {
            const shape_t *s = &shapes[cs_idx[k]];
            const float sc = 0.0f;   /* auto */
            for (int zi = 0; zi < 5; zi++) {
                arm_t a = { "op = -zw (emitted)", 23, zws[zi], 5, sc, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 1 };
                run_arm(fd, s, &a, seed + 1000 + 10 * k + zi);
            }
            arm_t r = { "repro: zw -37, op 0", 23, -37, 5, sc, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 1 };
            run_arm(fd, s, &r, seed + 1000 + 10 * k + 1);
            arm_t ws = { "zw -37, op +zw (wrong sign)", 23, -37, 5, sc, 1, (uint32_t)(-37) & 0xffffu,
                         0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 1 };
            run_arm(fd, s, &ws, seed + 1000 + 10 * k + 1);
        }
        /* MobileDet's direct convs: uint8 zero points in 125..147, weights 120..143, out 0 */
        arm_t md = { "MobileDet-like zx 1 zw 9 zo -128", 1, 9, -128, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 1 };
        run_arm(fd, &shapes[1], &md, seed + 1100);
        run_arm(fd, &shapes[5], &md, seed + 1101);
        arm_t md2 = { "MobileDet-like zx -128 zw 6", -128, 6, -128, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 1 };
        run_arm(fd, &shapes[1], &md2, seed + 1102);
    }

    if (!only || !strcmp(only, "perc")) {
        printf(" -- per-channel BS multiplier (64-byte group A0/B32/C48, data use 7), zw 0, zx 23, zo -7\n");
        const int pidx[] = { 1, 4 };
        for (int k = 0; k < 2; k++) {
            const shape_t *s = &shapes[pidx[k]];
            const float sc = 0.0f;   /* auto */
            arm_t a0 = { "s0 C 1..1 (identity)", 23, 0, -7, sc, 0, 0, 1, 0, 0, 1, 1, 0, -1, -1, -1, 0, 0 };
            arm_t a1 = { "s14 C 8192..32767", 23, 0, -7, sc, 0, 0, 1, 14, 14, 8192, 32767, 0, -1, -1, -1, 0, 0 };
            arm_t a2 = { "split 14/6", 23, 0, -7, sc, 0, 0, 1, 14, 6, 8192, 32767, 0, -1, -1, -1, 0, 0 };
            arm_t a3 = { "s14 B +-20, OW_SRC 0", 23, 0, -7, sc, 0, 0, 1, 14, 14, 8192, 32767, 20, 0, -1, -1, 0, 0 };
            arm_t a4 = { "s14 B +-20, OW_SRC 1", 23, 0, -7, sc, 0, 0, 1, 14, 14, 8192, 32767, 20, 1, -1, -1, 0, 0 };
            run_arm(fd, s, &a0, seed + 2000 + 10 * k);
            run_arm(fd, s, &a1, seed + 2001 + 10 * k);
            run_arm(fd, s, &a2, seed + 2002 + 10 * k);
            run_arm(fd, s, &a3, seed + 2003 + 10 * k);
            run_arm(fd, s, &a4, seed + 2004 + 10 * k);
        }
    }

    const arm_t ctrl2 = { "control again", 23, 0, -7, 0.0f, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 1, 0 };
    run_arm(fd, &shapes[0], &ctrl2, seed);
    rocket_close(fd);
    if (g_ctrl_bad) { printf("RESULT: a control arm was not exact; no other row can be read\n"); return 1; }
    printf("RESULT: all arms ran; controls exact\n");
    return 0;
}
