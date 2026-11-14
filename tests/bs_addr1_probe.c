// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * bs_addr1_probe.c — RK3588 board probe: is DPU_RDMA 0x5024 a live register here?
 *
 * On the RK3576, DPU_RDMA 0x5024 (BS_BASE_ADDR1) is the base of a per-task operand: the DPU
 * reads one 32-bit word through it and right-shifts the BS product by bits [5:0] where the
 * product is non-negative and by bits [13:8] where it is negative [HW sweep, H96]. The
 * RK3588's TRM register list has no entry at 0x5024 (it runs 0x5020 BS_BASE_ADDR, then
 * 0x5028 NRDMA_CFG), Mesa's registers.xml has none, and no RK3588 generator writes it. The
 * RK3588 carries the same per-sign shift in two registers instead, BS_MUL_CFG[13:8] and
 * DATA_FORMAT[9:4]. The register file persists across jobs and processes, so if 0x5024 is
 * live here a second client writing it would silently attenuate every BS-multiplied result
 * this library produces.
 *
 * The program is the direct int8 conv on the int8-out writer (gen_conv2d_int8, int8_out 1)
 * with the per-channel BS multiplier patched in as tests/conv_i8out_probe does (BRDMA data
 * use 7, a 64-byte group per 8 OC: int32 A at 0, int16 B at 32, int16 C at 48). Weight and
 * input zero points are 0, so CPEND is 0, and C is drawn from 1..4 so the product stays far
 * inside int32. One extra word, DPU_RDMA 0x5024, is inserted after BS_BASE_ADDR (0x5020).
 *
 * Every arm runs the job twice, over an output prefilled with 0xAA and then 0x55, and scores
 * each written element against four models of the BS output, then the same OUT_CVT:
 *
 *   regs     ((P + A[c]) * C[c]) >> (BS_MUL_CFG shift or DATA_FORMAT shift, by the sign)
 *   word     the same with the shifts from the 0x5024 word ([5:0] non-negative, [13:8] negative)
 *   sum      the same with the register and word shifts added
 *   inert    P + A[c], the multiplier not applied
 *
 * `disc` is how many elements the regs and word models disagree on: a count of matches is
 * evidence only where disc is large. The modes for 0x5024:
 *
 *   none     not written (the emitted program)
 *   word     the address of a zeroed BO whose first word is the arm's value
 *   iova0    the address of the first BO the process allocated (IOVA 0 on a mainline rocket
 *            stack), filled with 0x03 bytes, the RK3576's IOVA-0 trap
 *
 * A control (no 0x5024 write, register shift 0) runs first and last, and the very last job
 * writes 0x5024 = 0, its reset value, so a live register is not left pointing into a freed
 * address space. argv[1] = "brdma1e" or "brdma70e" runs ONE extra arm instead of the table,
 * with the word at 0x0202 and BRDMA_CFG at a candidate enable for the operand read: 0x1E sets
 * the fourth bit of the data-use field [4:1], 0x70E the reserved bits [10:8] where the RK3576
 * keeps its data-use value (0x710). An unknown operand-reader value can stall the job (on the
 * RK3576 a wrong BRDMA_CFG makes the DPU write nothing), so run each in its own process and
 * read the kernel log between them.
 *
 * Exit 0 when every arm ran (a probe: the table is the verdict), 1 when a control is not
 * exact against `regs`, 77 without a device.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "npu_regmap.h"   /* the register offsets and fields (src/) */
#include "test_fill.h"
#include "requant_model.h"
#include "rocket_hw_profile.h"

#define PADB RK3588_CBUF_BANK_BYTES
#define TARGET_DPU_RDMA_P (NPU_TARGET_DPU_RDMA | NPU_REG_WRITE)
#define RDMA_BRDMA_CFG_P   0x501C
#define RDMA_BS_BASE_P     0x5020
#define RDMA_BS_ADDR1_P    0x5024

static int rup(int a, int b) { return ((a + b - 1) / b) * b; }
static int out_dim(int in, int k, int s, int p) { return (in + 2 * p - k) / s + 1; }

typedef struct {
    const char *name;
    int IC, IH, IW, OC, KH, KW, sy, sx, pt, pl;
} shape_t;

enum { A1_NONE = 0, A1_WORD = 1, A1_IOVA0 = 2, A1_ZERO = 3 };

typedef struct {
    const char *label;
    unsigned s_mul, s_neg;   /* the register shifts */
    int a1_mode;
    uint32_t a1_word;        /* A1_WORD: the word; A1_IOVA0: the fill byte, replicated */
    uint32_t brdma;          /* 0: the emitted 7 << 1 */
    int control;
} arm_t;

/* ---- regcmd word patching (as tests/conv_i8out_probe.c) ----------------------------- */
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
            ops[i] = REGCMD(blockop, reg, v);
            k++;
        }
    return k;
}
/* Insert one word after the first write of `after`; returns the new count or -1. */
static int reg_insert_after(uint64_t *ops, unsigned n, unsigned cap, unsigned blockop,
                            unsigned after, unsigned reg, uint32_t v)
{
    int k = reg_find(ops, n, blockop, after);
    if (k < 0 || n + 1 > cap) return -1;
    memmove(&ops[k + 2], &ops[k + 1], (size_t)(n - (unsigned)k - 1) * sizeof(uint64_t));
    ops[k + 1] = REGCMD(blockop, reg, v);
    return (int)n + 1;
}

/* Mesa's direct weight packing for a whole-group OC (conv_weight_index_int8's order). */
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

static int g_ctrl_bad = 0;

/* One device run at one prefill byte; got [OC][OH][OW]. */
static int run_once(int fd, const shape_t *s, const arm_t *a, const int8_t *x, const int8_t *w,
                    const uint8_t *coeff, size_t coeff_bytes, float conv_scale, int zo,
                    uint8_t fill, int8_t *got, uint32_t *iova0_out)
{
    const int ICp = rup(s->IC, 32), OCp = rup(s->OC, 32);
    const int IHj = s->IH < 4 ? 4 : s->IH;
    const int OH = out_dim(s->IH, s->KH, s->sy, s->pt), OW = out_dim(s->IW, s->KW, s->sx, s->pl);
    const int OCc = rup(OCp, 16);
    rocket_bo first = {0}, in_bo = {0}, wt_bo = {0}, rc_bo = {0}, bs_bo = {0}, out_bo = {0},
              w1_bo = {0};
    uint64_t regs[256] = {0};
    int ret = -1;

    /* `first` is allocated before anything else on this fd, so it is the IOVA-0 buffer. */
    if (rocket_bo_alloc32(fd, 4096, &first) ||
        rocket_bo_alloc32(fd, (size_t)ICp * IHj * s->IW + PADB, &in_bo) ||
        rocket_bo_alloc32(fd, (size_t)OCp * ICp * s->KH * s->KW + PADB, &wt_bo) ||
        rocket_bo_alloc32(fd, 256 * sizeof(uint64_t), &rc_bo) ||
        rocket_bo_alloc32(fd, coeff_bytes + PADB, &bs_bo) ||
        rocket_bo_alloc32(fd, 4096, &w1_bo) ||
        rocket_bo_alloc32(fd, (size_t)OCc * OH * OW + PADB, &out_bo)) {
        fprintf(stderr, "  BO alloc failed\n");
        goto out;
    }
    if (iova0_out) *iova0_out = (uint32_t)first.dma_address;
    rocket_bo_prep(fd, &first, 1, 0);
    memset(first.ptr, a->a1_mode == A1_IOVA0 ? (int)(a->a1_word & 0xffu) : 0, first.size);
    rocket_bo_fini(fd, &first);
    rocket_bo_prep(fd, &w1_bo, 1, 0);
    memset(w1_bo.ptr, 0, w1_bo.size);
    if (a->a1_mode == A1_WORD) memcpy(w1_bo.ptr, &a->a1_word, 4);
    rocket_bo_fini(fd, &w1_bo);
    rocket_bo_prep(fd, &in_bo, 1, 0);
    memset(in_bo.ptr, 0, in_bo.size);
    for (int ic = 0; ic < s->IC; ic++)
        for (int ih = 0; ih < s->IH; ih++)
            for (int iw = 0; iw < s->IW; iw++)
                ((int8_t *)in_bo.ptr)[feature_index(ICp, IHj, s->IW, 16, ic + 1, ih + 1, iw + 1)] =
                    x[((size_t)ic * s->IH + ih) * s->IW + iw];
    rocket_bo_fini(fd, &in_bo);
    rocket_bo_prep(fd, &wt_bo, 1, 0);
    memset(wt_bo.ptr, 0, wt_bo.size);
    for (int oc = 0; oc < s->OC; oc++)
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
            .input_iova = (uint32_t)in_bo.dma_address, .weights_iova = (uint32_t)wt_bo.dma_address,
            .output_iova = (uint32_t)out_bo.dma_address, .tasks = regs,
            .int8_out = 1,
            .in_scale = conv_scale, .w_scale = 1.0f, .out_scale = 1.0f,
            .input_zero_point = 0x80, .output_zero_point = zo + 0x80,
            .weight_zero_point = 0x80,
            .bias_iova = (uint32_t)bs_bo.dma_address,
        };
        int g = gen_conv2d_int8(&p);
        if (g != 0) { fprintf(stderr, "  gen_conv2d_int8 failed (%d)\n", g); goto out; }
        unsigned n = p.task_count;
        {
            int kf = reg_find(regs, n, TARGET_DPU, REG_DPU_DATA_FORMAT);
            uint32_t fmt = (reg_get(regs, kf) & ~(0x3fu << 4)) | ((a->s_neg & 0x3fu) << 4);
            reg_set(regs, n, TARGET_DPU, REG_DPU_DATA_FORMAT, fmt);
            reg_set(regs, n, TARGET_DPU, REG_DPU_BS_CFG, (2u << 16) | (1u << 8) | (1u << 6));
            reg_set(regs, n, TARGET_DPU, REG_DPU_BS_MUL_CFG, ((a->s_mul & 0x3fu) << 8) | 1u);
            reg_set(regs, n, TARGET_DPU_RDMA_P, RDMA_BRDMA_CFG_P, a->brdma ? a->brdma : 7u << 1);
        }
        if (a->a1_mode != A1_NONE) {
            uint32_t v = a->a1_mode == A1_WORD  ? (uint32_t)w1_bo.dma_address
                       : a->a1_mode == A1_IOVA0 ? (uint32_t)first.dma_address : 0u;
            int nn = reg_insert_after(regs, n, 256, TARGET_DPU_RDMA_P, RDMA_BS_BASE_P,
                                      RDMA_BS_ADDR1_P, v);
            if (nn < 0) { fprintf(stderr, "  no BS_BASE_ADDR write to insert after\n"); goto out; }
            n = (unsigned)nn;
        }
        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, regs, (size_t)n * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);
        rocket_bo_prep(fd, &out_bo, 1, 0);
        memset(out_bo.ptr, fill, out_bo.size);
        rocket_bo_fini(fd, &out_bo);
        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo.dma_address, .regcmd_count = n };
        uint32_t in_h[] = { in_bo.handle, wt_bo.handle, bs_bo.handle, rc_bo.handle,
                            w1_bo.handle, first.handle };
        uint32_t out_h[] = { out_bo.handle };
        if (rocket_submit_tasks(fd, &task, 1, in_h, 6, out_h, 1)) {
            fprintf(stderr, "  submit failed\n"); goto out;
        }
    }
    if (rocket_bo_prep(fd, &out_bo, 0, 2000000000ULL)) { ret = -2; goto out; }
    for (int oc = 0; oc < s->OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                got[((size_t)oc * OH + oh) * OW + ow] =
                    ((const int8_t *)out_bo.ptr)[feature_index(OCc, OH, OW, 16, oc + 1, oh + 1, ow + 1)];
    rocket_bo_fini(fd, &out_bo);
    ret = 0;
out:
    rocket_bo_free(fd, &out_bo); rocket_bo_free(fd, &w1_bo); rocket_bo_free(fd, &bs_bo);
    rocket_bo_free(fd, &rc_bo); rocket_bo_free(fd, &wt_bo); rocket_bo_free(fd, &in_bo);
    rocket_bo_free(fd, &first);
    return ret;
}

static int model(int64_t PA, int16_t C, unsigned sp, unsigned sn, unsigned mul, unsigned shift, int zo)
{
    int64_t pr = PA * (int64_t)C;
    int64_t v = sat32(requant_round_shift(pr, pr >= 0 ? sp : sn));
    return requant_apply_zp(v, mul, shift, zo);
}

static void run_arm(int fd, const shape_t *s, const arm_t *a, uint64_t seed)
{
    const int IC = s->IC, OC = s->OC, KH = s->KH, KW = s->KW;
    const int OCp = rup(OC, 32);
    const int OH = out_dim(s->IH, KH, s->sy, s->pt), OW = out_dim(s->IW, KW, s->sx, s->pl);
    const size_t ni = (size_t)IC * s->IH * s->IW, nw = (size_t)OC * IC * KH * KW;
    const size_t no = (size_t)OC * OH * OW;
    const int zo = -7;
    /* the accumulator's spread times C's mean (2.5), over the int8 range */
    const float conv_scale = 0.0006f * (float)sqrt(576.0 / (double)(IC * KH * KW)) / 2.5f;
    int8_t *x = malloc(ni), *w = malloc(nw), *g1 = malloc(no), *g2 = malloc(no);
    int32_t *bias = malloc(OC * 4), *Ct = malloc(OC * 4);
    int16_t *Cm = calloc(OC, 2);
    int64_t *PA = malloc(no * 8);
    const size_t coeff_bytes = (size_t)rup(OCp, 8) / 8 * 64;
    uint8_t *coeff = calloc(coeff_bytes, 1);
    unsigned mul, shift;
    uint32_t iova0 = 0xffffffffu;

    tf_fill_i8(x, ni, seed + 1, -128, 127);
    tf_fill_i8(w, nw, seed + 2, -128, 127);
    tf_fill_i32(bias, OC, seed + 3, -6000, 6000);
    tf_fill_i32(Ct, OC, seed + 4, 1, 4);
    for (int c = 0; c < OC; c++) Cm[c] = (int16_t)Ct[c];
    for (int c = 0; c < OC; c++) {
        uint8_t *grp = coeff + (size_t)(c / 8) * 64;
        memcpy(grp + 4 * (c % 8), &bias[c], 4);
        memcpy(grp + 48 + 2 * (c % 8), &Cm[c], 2);
    }
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t p = 0;
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++) {
                        int ih = oh * s->sy + kh - s->pt, iw = ow * s->sx + kw - s->pl;
                        if (ih < 0 || ih >= s->IH || iw < 0 || iw >= s->IW) continue;
                        for (int ic = 0; ic < IC; ic++)
                            p += (int64_t)x[((size_t)ic * s->IH + ih) * s->IW + iw] *
                                 w[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
                    }
                PA[((size_t)oc * OH + oh) * OW + ow] = p + bias[oc];
            }
    requant_params(conv_scale, &mul, &shift);

    int rc = run_once(fd, s, a, x, w, coeff, coeff_bytes, conv_scale, zo, 0xAA, g1, &iova0);
    if (!rc) rc = run_once(fd, s, a, x, w, coeff, coeff_bytes, conv_scale, zo, 0x55, g2, NULL);
    if (rc) {
        printf("  %-8s %-36s %s\n", s->name, a->label, rc == -2 ? "TIMEOUT" : "FAILED");
        if (a->control) g_ctrl_bad = 1;
        goto done;
    }
    {
        /* the word the DPU would read, per mode */
        uint32_t wv = a->a1_mode == A1_WORD ? a->a1_word
                    : a->a1_mode == A1_IOVA0 ? (a->a1_word & 0xffu) * 0x01010101u : 0u;
        unsigned wp = wv & 0x3fu, wn = (wv >> 8) & 0x3fu;
        long unwr = 0, k_regs = 0, k_word = 0, k_sum = 0, k_inert = 0, k_none = 0, disc = 0, rail = 0;
        for (size_t o = 0; o < no; o++) {
            int c = (int)(o / ((size_t)OH * OW));
            int fr = model(PA[o], Cm[c], a->s_mul, a->s_neg, mul, shift, zo);
            int fw = model(PA[o], Cm[c], wp, wn, mul, shift, zo);
            int fs = model(PA[o], Cm[c], a->s_mul + wp, a->s_neg + wn, mul, shift, zo);
            int fi = requant_apply_zp(PA[o], mul, shift, zo);
            if (fr != fw) disc++;
            if (fr == 127 || fr == -128) rail++;
            if (g1[o] != g2[o]) { unwr++; continue; }
            int d = g1[o];
            int any = 0;
            if (d == fr) { k_regs++; any = 1; }
            if (d == fw) { k_word++; any = 1; }
            if (d == fs) { k_sum++; any = 1; }
            if (d == fi) { k_inert++; any = 1; }
            if (!any) k_none++;
        }
        printf("  %-8s %-36s 0x5024 %s word 0x%08x (iova0 0x%x)\n", s->name, a->label,
               a->a1_mode == A1_NONE ? "unwritten" : a->a1_mode == A1_WORD ? "->word"
               : a->a1_mode == A1_IOVA0 ? "->IOVA0" : "=0", wv, iova0);
        printf("      of %zu (rail %ld, disc %ld): unwr %ld | regs(%u/%u) %ld  word(%u/%u) %ld  "
               "sum %ld  inert %ld  none %ld\n",
               no, rail, disc, unwr, a->s_mul, a->s_neg, k_regs, wp, wn, k_word, k_sum, k_inert, k_none);
        if (a->control && (k_regs != (long)no)) g_ctrl_bad = 1;
    }
done:
    free(x); free(w); free(g1); free(g2); free(bias); free(Ct); free(Cm); free(PA); free(coeff);
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) { printf("SKIP: no NPU device\n"); return 77; }
    const uint32_t brdma = argc > 1 && !strcmp(argv[1], "brdma1e")  ? 0x1Eu
                         : argc > 1 && !strcmp(argv[1], "brdma70e") ? 0x70Eu : 0u;

    const shape_t shapes[] = {
        /* name       IC  IH  IW  OC KH KW sy sx pt pl */
        { "c64k3p1",  64, 10, 12, 64, 3, 3, 1, 1, 1, 1 },
        { "c32k1",    32,  8,  8, 32, 1, 1, 1, 1, 0, 0 },
    };
    uint64_t seed = 0x5024000ull;

    printf("bs_addr1_probe: is DPU_RDMA 0x5024 (the RK3576's BS shift-word base) live on this part?\n");
    printf("  rocket_hw: %s\n", rocket_hw_current()->name);

    const arm_t ctrl = { "control (regs s0, 0x5024 unwritten)", 0, 0, A1_NONE, 0, 0, 1 };
    run_arm(fd, &shapes[0], &ctrl, seed);

    const arm_t arms[] = {
        { "regs s0, word 0x0202",            0, 0, A1_WORD,  0x0202u, 0, 0 },
        { "regs s0, word 0x0300 (neg only)", 0, 0, A1_WORD,  0x0300u, 0, 0 },
        { "regs s0, word 0x0003 (pos only)", 0, 0, A1_WORD,  0x0003u, 0, 0 },
        { "regs s0, word 0x0808 (flush)",    0, 0, A1_WORD,  0x0808u, 0, 0 },
        { "regs s2/2, word 0x0000",          2, 2, A1_WORD,  0x0000u, 0, 0 },
        { "regs s2/2, unwritten",            2, 2, A1_NONE,  0,       0, 0 },
        { "regs s0, ->IOVA0 filled 0x03",    0, 0, A1_IOVA0, 0x03u,   0, 0 },
        { "regs s1/3, word 0x0102",          1, 3, A1_WORD,  0x0102u, 0, 0 },
    };
    if (brdma) {
        printf(" -- BRDMA_CFG 0x%x, word 0x0202\n", brdma);
        const arm_t b = { "BRDMA candidate, word 0x0202", 0, 0, A1_WORD, 0x0202u, brdma, 0 };
        run_arm(fd, &shapes[1], &b, seed + 900);
    } else {
        for (int si = 0; si < 2; si++)
            for (size_t k = 0; k < sizeof arms / sizeof arms[0]; k++)
                run_arm(fd, &shapes[si], &arms[k], seed + 100 * (uint64_t)si + k);
    }

    const arm_t ctrl2 = { "control again", 0, 0, A1_NONE, 0, 0, 1 };
    run_arm(fd, &shapes[0], &ctrl2, seed);
    /* leave 0x5024 at its reset value */
    const arm_t reset = { "0x5024 = 0 (reset value, last job)", 0, 0, A1_ZERO, 0, 0, 1 };
    run_arm(fd, &shapes[1], &reset, seed + 1);
    rocket_close(fd);
    if (g_ctrl_bad) { printf("RESULT: a control arm was not exact against regs; no other row can be read\n"); return 1; }
    printf("RESULT: all arms ran; controls exact\n");
    return 0;
}
