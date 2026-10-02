// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * cpend_wzp_probe.c — RK3588 board probe: does DPU_BS_OW_OP (the TRM's "CPEND operand")
 * carry an asymmetric weight zero point?
 *
 * A TFLite quantized conv computes acc = bias + sum (x - zx)(w - zw). With zw = 0 the
 * input zero point folds into the bias and the int8 depthwise entry is exact; a nonzero
 * zw adds -zw * sum(x), which depends on the input and cannot fold. Mesa's uint8 driver
 * writes DPU_BS_OW_OP = 0x80 - Z_w for every conv, with the weight cube holding u_w - 0x80,
 * which is exactly that term if CPEND adds OW_OP * (the window sum of the CNA's input,
 * border pad included) to the accumulator [expected]. In the int8 domain that is
 * OW_OP = -zw and a bias of bias - zx * sum(w - zw).
 *
 * The probe does not assume the reading. For each arm it runs the program with a chosen
 * OW_OP, sentinel-prefilled output, and scores the device output against:
 *
 *   F      the intended function: TFLite's int32 accumulator, then the OUT_CVT model
 *          (tests/requant_model.h). This is what the entry must compute.
 *   H(v)   the CPEND hypothesis at a candidate reading v of the 16-bit operand:
 *          bias' + sum xp*w + v * sum xp, then the same requant. v is read as signed 16,
 *          unsigned 16, signed 8 (low byte) and 0 (CPEND inert). H(-zw) == F algebraically.
 *
 * Each arm reports magnitudes: elements equal to F, one away, further, still holding the
 * sentinel, and how many equal each H(v). A reproduction arm (zw = 0 at OW_OP = 128, the
 * value measured wrong on 3962 of 4096) and a control that must succeed (zw = 0 at 0) are
 * part of the run, so no other arm is read without them.
 *
 * DW arms use the int8-out writer (gen_conv2d_dw_int8, int8_out=1). DIRECT arms use the
 * int32-raw writer (gen_conv2d_int8) with the BS stage and CPEND switched on by patching
 * the emitted words (BS_CFG, BS_OW_CFG.OD_BYPASS, BS_OW_OP, CORE_MISC_CFG.QD_EN), and are
 * scored in int32 against R = sum xp*w (the raw accumulator) and R + v * sum xp.
 *
 * The last two depthwise shapes (C 96 and 80) have a partial last 64-channel group, run
 * through the raw program at that channel count. On the RK1 the program leaves that group
 * UNWRITTEN at every arm, the symmetric control included: the elements still hold the
 * sentinel, 32 or 16 channels' worth, and the full groups are exact [HW sweep, RK1,
 * 2026-09-27]. rocket_conv2d_dw_int8 now programs each job at whole 64-channel groups for
 * that reason. Those rows are witnesses of the program's behaviour, not controls.
 *
 * Exit 0 when every arm ran (a probe, not a gate: the verdict is the table). Exit 1 when a
 * full-group control is not exact, since then no other row can be read. 77 without a device.
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

static int rup(int a, int b) { return ((a + b - 1) / b) * b; }

typedef struct {
    const char *name;
    int C, IH, IW, KH, KW, sy, sx, pt, pl;
} shape_t;

static int out_dim(int in, int k, int s, int p) { return (in + 2 * p - k) / s + 1; }

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

/* ---- host arithmetic ------------------------------------------------------------ */
/* Per output of a depthwise conv: A = TFLite's accumulator (bias included), P = sum xp*w,
 * S = sum xp, with xp the input or, off the image, zx (the CNA pads with the input zero
 * point, and a TFLite padded tap contributes nothing). */
static void dw_terms(const shape_t *s, const int8_t *x, const int8_t *w, const int32_t *bias,
                     int zx, int zw, int64_t *A, int64_t *P, int64_t *S)
{
    const int OH = out_dim(s->IH, s->KH, s->sy, s->pt), OW = out_dim(s->IW, s->KW, s->sx, s->pl);
    for (int c = 0; c < s->C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t a = bias[c], p = 0, sm = 0;
                for (int kh = 0; kh < s->KH; kh++)
                    for (int kw = 0; kw < s->KW; kw++) {
                        int ih = oh * s->sy + kh - s->pt, iw = ow * s->sx + kw - s->pl;
                        int xp = (ih >= 0 && ih < s->IH && iw >= 0 && iw < s->IW)
                                 ? x[((size_t)c * s->IH + ih) * s->IW + iw] : zx;
                        int wv = w[((size_t)c * s->KH + kh) * s->KW + kw];
                        a  += (int64_t)(xp - zx) * (wv - zw);
                        p  += (int64_t)xp * wv;
                        sm += xp;
                    }
                size_t o = ((size_t)c * OH + oh) * OW + ow;
                A[o] = a; P[o] = p; S[o] = sm;
            }
}

static int is_border(const shape_t *s, int oh, int ow)
{
    /* an output whose window reaches off the image */
    int ih0 = oh * s->sy - s->pt, iw0 = ow * s->sx - s->pl;
    return ih0 < 0 || iw0 < 0 || ih0 + s->KH > s->IH || iw0 + s->KW > s->IW;
}

/* ---- the DW int8-out device run ---------------------------------------------------- */
static int run_dw(int fd, const shape_t *s, const int8_t *x, const int8_t *w, const int32_t *bias_q,
                  float in_s, float w_s, float out_s, int zx, int zw, int zo,
                  int override, uint32_t op, uint32_t *op_emitted, int8_t *got)
{
    const int C = s->C, IH = s->IH, IW = s->IW, KH = s->KH, KW = s->KW;
    const int OH = out_dim(IH, KH, s->sy, s->pt), OW = out_dim(IW, KW, s->sx, s->pl);
    const int Cpad = rup(C, DW_G), IHj = IH < 4 ? 4 : IH;
    rocket_bo guard = {0}, in_bo = {0}, wt_bo = {0}, rc_bo = {0}, bs_bo = {0}, out_bo = {0};
    uint64_t regs[256] = {0};
    int ret = -1;

    if (rocket_bo_alloc32(fd, 4096, &guard) ||
        rocket_bo_alloc32(fd, (size_t)C * IHj * IW + PADB, &in_bo) ||
        rocket_bo_alloc32(fd, (size_t)Cpad * KH * KW + PADB, &wt_bo) ||
        rocket_bo_alloc32(fd, 256 * sizeof(uint64_t), &rc_bo) ||
        rocket_bo_alloc32(fd, (size_t)Cpad * sizeof(int32_t) + PADB, &bs_bo) ||
        rocket_bo_alloc32(fd, (size_t)C * OH * OW + PADB, &out_bo)) {
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

    /* bias - zx * sum(w - zw): the input zero point folded as the entry folds it, now with
     * the weight zero point inside the sum. */
    rocket_bo_prep(fd, &bs_bo, 1, 0);
    memset(bs_bo.ptr, 0, bs_bo.size);
    for (int c = 0; c < C; c++) {
        int32_t sw = 0;
        for (int k = 0; k < KH * KW; k++) sw += w[(size_t)c * KH * KW + k] - zw;
        ((int32_t *)bs_bo.ptr)[c] = bias_q[c] - zx * sw;
    }
    rocket_bo_fini(fd, &bs_bo);

    {
        conv_params_t p = {
            .ic = C, .ih = IHj, .iw = IW, .oc = C, .oh = OH, .ow = OW,
            .kh = KH, .kw = KW, .stride_y = s->sy, .stride_x = s->sx,
            .dil_y = 1, .dil_x = 1, .pad_top = s->pt, .pad_left = s->pl,
            .input_dma = (uint32_t)in_bo.dma_address, .weights_dma = (uint32_t)wt_bo.dma_address,
            .output_dma = (uint32_t)out_bo.dma_address, .tasks = regs,
            .dw_group = DW_G, .int8_out = 1,
            .in_scale = in_s, .w_scale = w_s, .out_scale = out_s,
            .input_zero_point = zx + 0x80, .output_zero_point = zo + 0x80,
            .weight_zero_point = zw + 0x80,
            .bias_dma = (uint32_t)bs_bo.dma_address,
        };
        int g = gen_conv2d_dw_int8(&p);
        if (g != 0) { fprintf(stderr, "  gen_conv2d_dw_int8 failed (%d)\n", g); goto out; }
        int k = reg_find(regs, p.task_count, OP_REG_DPU, DPU_BS_OW_OP);
        if (k < 0) { fprintf(stderr, "  no DPU_BS_OW_OP word\n"); goto out; }
        if (override) reg_set(regs, p.task_count, OP_REG_DPU, DPU_BS_OW_OP, op & 0xffffu);
        *op_emitted = reg_get(regs, k);

        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);
        if (tf_sentinel_bo(fd, &out_bo)) { fprintf(stderr, "  sentinel failed\n"); goto out; }

        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo.dma_address, .regcmd_count = p.task_count };
        uint32_t in_h[] = { in_bo.handle, wt_bo.handle, bs_bo.handle, rc_bo.handle };
        uint32_t out_h[] = { out_bo.handle };
        if ((ret = rocket_submit_tasks(fd, &task, 1, in_h, 4, out_h, 1)) != 0) {
            fprintf(stderr, "  submit failed (%d)\n", ret); ret = -1; goto out;
        }
    }
    if ((ret = rocket_bo_prep(fd, &out_bo, 0, 2000000000ULL)) != 0) {
        fprintf(stderr, "  wait timeout (%d)\n", ret); ret = -2; goto out;
    }
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                got[((size_t)c * OH + oh) * OW + ow] =
                    ((const int8_t *)out_bo.ptr)[feature_data(C, OH, OW, 16, c + 1, oh + 1, ow + 1)];
    rocket_bo_fini(fd, &out_bo);
    ret = 0;
out:
    rocket_bo_free(fd, &out_bo); rocket_bo_free(fd, &bs_bo); rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &wt_bo); rocket_bo_free(fd, &in_bo); rocket_bo_free(fd, &guard);
    return ret;
}

static int s16(uint32_t op) { return (int)(int16_t)(op & 0xffffu); }
static int s8v(uint32_t op) { return (int)(int8_t)(op & 0xffu); }

static int g_ctrl_bad = 0;

/* One DW arm: run, score, print one row. */
static void dw_arm(int fd, const shape_t *s, const char *label, int zx, int zw, int zo,
                   int override, uint32_t op, float in_s, float w_s, float out_s,
                   uint64_t seed, int is_control)
{
    const int OH = out_dim(s->IH, s->KH, s->sy, s->pt), OW = out_dim(s->IW, s->KW, s->sx, s->pl);
    const size_t ni = (size_t)s->C * s->IH * s->IW, nw = (size_t)s->C * s->KH * s->KW;
    const size_t no = (size_t)s->C * OH * OW;
    int8_t *x = malloc(ni), *w = malloc(nw), *got = malloc(no);
    int32_t *bias = malloc(s->C * sizeof(int32_t));
    int64_t *A = malloc(no * 8), *P = malloc(no * 8), *S = malloc(no * 8);
    unsigned mul, shift;
    uint32_t emitted = 0;

    tf_fill_i8(x, ni, seed + 1, -128, 127);
    tf_fill_i8(w, nw, seed + 2, -128, 127);
    tf_fill_i32(bias, s->C, seed + 3, -6000, 6000);
    requant_params(in_s * w_s / out_s, &mul, &shift);
    dw_terms(s, x, w, bias, zx, zw, A, P, S);

    int rc = run_dw(fd, s, x, w, bias, in_s, w_s, out_s, zx, zw, zo, override, op, &emitted, got);
    if (rc) {
        printf("  %-10s %-24s zw=%4d op=0x%04x  %s\n", s->name, label, zw, override ? op : 0,
               rc == -2 ? "TIMEOUT" : "FAILED");
        if (is_control) g_ctrl_bad = 1;
        goto done;
    }
    {
        long eq = 0, off1 = 0, offm = 0, sent = 0, bord_bad = 0, int_bad = 0, nb = 0, maxab = 0;
        long hs16 = 0, hu16 = 0, hs8 = 0, h0 = 0;
        const int vs16 = s16(emitted), vs8 = s8v(emitted);
        const long vu16 = (long)(emitted & 0xffffu);
        const int64_t bprime_delta = 0; /* bias' + P == A + zw*S (see header) */
        (void)bprime_delta;
        for (int c = 0; c < s->C; c++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++) {
                    size_t o = ((size_t)c * OH + oh) * OW + ow;
                    int f = requant_apply_zp(A[o], mul, shift, zo);
                    int d = got[o];
                    int b = is_border(s, oh, ow);
                    nb += b;
                    /* bias' + P = A + zw * S  (expand sum (xp - zx)(w - zw)) */
                    int64_t base = A[o] + (int64_t)zw * S[o];
                    if (d == requant_apply_zp(base + (int64_t)vs16 * S[o], mul, shift, zo)) hs16++;
                    if (d == requant_apply_zp(base + (int64_t)vu16 * S[o], mul, shift, zo)) hu16++;
                    if (d == requant_apply_zp(base + (int64_t)vs8 * S[o], mul, shift, zo)) hs8++;
                    if (d == requant_apply_zp(base, mul, shift, zo)) h0++;
                    if ((uint8_t)d == TF_SENTINEL_BYTE) sent++;
                    long ad = labs((long)d - f);
                    if (ad > maxab) maxab = ad;
                    if (!ad) { eq++; continue; }
                    if (ad == 1) off1++; else offm++;
                    if (b) bord_bad++; else int_bad++;
                }
        printf("  %-10s %-24s zw=%4d op=0x%04x | F: exact %5ld/%zu off1 %4ld off>1 %5ld max %3ld "
               "sent %4ld (bad: border %ld/%ld interior %ld) | H s16(%d) %5ld  u16 %5ld  "
               "s8(%d) %5ld  inert %5ld\n",
               s->name, label, zw, emitted, eq, no, off1, offm, maxab, sent, bord_bad, nb,
               int_bad, vs16, hs16, hu16, vs8, hs8, h0);
        if (is_control && eq != (long)no) g_ctrl_bad = 1;
    }
done:
    free(x); free(w); free(got); free(bias); free(A); free(P); free(S);
}

static int cmp64(const void *a, const void *b)
{
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return x < y ? -1 : x > y;
}

/* size of the multiset intersection of two sorted arrays */
static size_t ms_overlap(const int64_t *a, size_t na, const int64_t *b, size_t nb)
{
    size_t i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else { k++; i++; j++; }
    }
    return k;
}

/* the depthwise int32 writer's failure signature, got[2k] == ref[k] in the cube's
 * address space: count elements whose value sits at twice its cube offset */
static long stride2(const int32_t *raw, size_t nraw, const int64_t *R, const int64_t *S, int v,
                    int OC, int OH, int OW, int OCp)
{
    long k = 0;
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                size_t o = ((size_t)oc * OH + oh) * OW + ow;
                size_t a = (size_t)feature_data(OCp, OH, OW, 4, oc + 1, oh + 1, ow + 1) * 2;
                if (a < nraw && (int64_t)raw[a] == R[o] + (int64_t)v * S[o]) k++;
            }
    return k;
}

/* ---- the DIRECT int32-raw device run with CPEND patched on --------------------------- */
typedef struct {
    const char *label;
    int bs_on;      /* 1: BS_CFG = BS stage live, ALU/MUL/RELU bypassed; 0: as emitted */
    int od_on;      /* 1: clear OD_BYPASS (CPEND live); 0: as emitted (bypassed) */
    int op_set;     /* write BS_OW_OP */
    uint32_t op;
    int qd;         /* -1 as emitted, else CORE_MISC_CFG.QD_EN */
} direct_arm_t;

static void direct_arm(int fd, const direct_arm_t *a, uint64_t seed, int is_control)
{
    const int IC = 32, OC = 32, IH = 8, IW = 10, KH = 3, KW = 3, sy = 1, sx = 1, pt = 1, pl = 1;
    const int OH = out_dim(IH, KH, sy, pt), OW = out_dim(IW, KW, sx, pl);
    const int OCp = rup(OC, 32);
    const size_t ni = (size_t)IC * IH * IW, nw = (size_t)OCp * IC * KH * KW, no = (size_t)OC * OH * OW;
    int8_t *x = malloc(ni), *w = calloc(nw, 1);
    int32_t *got = malloc(no * 4);
    int64_t *R = malloc(no * 8), *S = malloc(no * 8);
    rocket_bo guard = {0}, in_bo = {0}, wt_bo = {0}, rc_bo = {0}, out_bo = {0};
    uint64_t regs[256] = {0};
    uint32_t op_em = 0, misc = 0, owcfg = 0, bscfg = 0;
    int32_t *raw = NULL;
    size_t nraw = 0;
    int rc = -1;

    tf_fill_i8(x, ni, seed + 11, -128, 127);
    tf_fill_i8(w, (size_t)OC * IC * KH * KW, seed + 12, -128, 127);
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t r = 0, sm = 0;
                for (int ic = 0; ic < IC; ic++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++) {
                            int ih = oh * sy + kh - pt, iw = ow * sx + kw - pl;
                            if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) continue;   /* pad 0 */
                            int xv = x[((size_t)ic * IH + ih) * IW + iw];
                            r  += (int64_t)xv * w[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
                            sm += xv;
                        }
                size_t o = ((size_t)oc * OH + oh) * OW + ow;
                R[o] = r; S[o] = sm;
            }

    if (rocket_bo_alloc32(fd, 4096, &guard) ||
        rocket_bo_alloc32(fd, ni + PADB, &in_bo) ||
        rocket_bo_alloc32(fd, nw + PADB, &wt_bo) ||
        rocket_bo_alloc32(fd, 256 * sizeof(uint64_t), &rc_bo) ||
        rocket_bo_alloc32(fd, (size_t)OCp * OH * OW * 4 + PADB, &out_bo)) {
        fprintf(stderr, "  BO alloc failed\n"); goto out;
    }
    rocket_bo_prep(fd, &in_bo, 1, 0);
    memset(in_bo.ptr, 0, in_bo.size);
    for (int ic = 0; ic < IC; ic++)
        for (int ih = 0; ih < IH; ih++)
            for (int iw = 0; iw < IW; iw++)
                ((int8_t *)in_bo.ptr)[feature_data(IC, IH, IW, 16, ic + 1, ih + 1, iw + 1)] =
                    x[((size_t)ic * IH + ih) * IW + iw];
    rocket_bo_fini(fd, &in_bo);
    rocket_bo_prep(fd, &wt_bo, 1, 0);
    memset(wt_bo.ptr, 0, wt_bo.size);
    for (int oc = 0; oc < OCp; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    ((int8_t *)wt_bo.ptr)[weight_conv_int8(OCp, IC, KH, KW, oc + 1, ic + 1, kh + 1, kw + 1)] =
                        w[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
    rocket_bo_fini(fd, &wt_bo);
    {
        conv_params_t p = {
            .ic = IC, .ih = IH, .iw = IW, .oc = OCp, .oh = OH, .ow = OW,
            .kh = KH, .kw = KW, .stride_y = sy, .stride_x = sx, .dil_y = 1, .dil_x = 1,
            .pad_top = pt, .pad_left = pl,
            .input_dma = (uint32_t)in_bo.dma_address, .weights_dma = (uint32_t)wt_bo.dma_address,
            .output_dma = (uint32_t)out_bo.dma_address, .tasks = regs,
        };
        int g = gen_conv2d_int8(&p);
        if (g != 0) { fprintf(stderr, "  gen_conv2d_int8 failed (%d)\n", g); goto out; }
        unsigned n = p.task_count;
        int kb = reg_find(regs, n, OP_REG_DPU, DPU_BS_CFG);
        int ko = reg_find(regs, n, OP_REG_DPU, DPU_BS_OW_CFG);
        int km = reg_find(regs, n, OP_REG_CORE, CORE_MISC_CFG);
        if (kb < 0 || ko < 0 || km < 0) { fprintf(stderr, "  register word missing\n"); goto out; }
        if (a->bs_on) reg_set(regs, n, OP_REG_DPU, DPU_BS_CFG, (1u << 6) | (1u << 4) | (1u << 1));
        if (a->od_on) reg_set(regs, n, OP_REG_DPU, DPU_BS_OW_CFG, reg_get(regs, ko) & ~(1u << 1));
        if (a->op_set) reg_set(regs, n, OP_REG_DPU, DPU_BS_OW_OP, a->op & 0xffffu);
        if (a->qd >= 0) reg_set(regs, n, OP_REG_CORE, CORE_MISC_CFG,
                                (reg_get(regs, km) & ~1u) | (uint32_t)(a->qd & 1));
        bscfg = reg_get(regs, kb); owcfg = reg_get(regs, ko); misc = reg_get(regs, km);
        op_em = reg_get(regs, reg_find(regs, n, OP_REG_DPU, DPU_BS_OW_OP));
        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, regs, (size_t)n * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);
        if (tf_sentinel_bo(fd, &out_bo)) goto out;
        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo.dma_address, .regcmd_count = n };
        uint32_t in_h[] = { in_bo.handle, wt_bo.handle, rc_bo.handle };
        uint32_t out_h[] = { out_bo.handle };
        if (rocket_submit_tasks(fd, &task, 1, in_h, 3, out_h, 1)) { fprintf(stderr, "  submit failed\n"); goto out; }
    }
    if (rocket_bo_prep(fd, &out_bo, 0, 2000000000ULL)) { rc = -2; goto out; }
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                got[((size_t)oc * OH + oh) * OW + ow] =
                    ((const int32_t *)out_bo.ptr)[feature_data(OCp, OH, OW, 4, oc + 1, oh + 1, ow + 1)];
    /* the whole BO, for the placement question: are the right values anywhere in it? */
    nraw = out_bo.size / 4;
    raw = malloc(out_bo.size);
    if (raw) memcpy(raw, out_bo.ptr, out_bo.size);
    rocket_bo_fini(fd, &out_bo);
    rc = 0;
out:
    rocket_bo_free(fd, &out_bo); rocket_bo_free(fd, &rc_bo); rocket_bo_free(fd, &wt_bo);
    rocket_bo_free(fd, &in_bo); rocket_bo_free(fd, &guard);
    if (rc) {
        printf("  direct     %-24s %s\n", a->label, rc == -2 ? "TIMEOUT" : "FAILED");
        if (is_control) g_ctrl_bad = 1;
    } else {
        long eqR = 0, eqH = 0, eqHu = 0, sent = 0;
        long long maxd = 0;
        int v = s16(op_em);
        for (size_t o = 0; o < no; o++) {
            if (TF_SENT_I32(got[o])) sent++;
            if (got[o] == R[o]) eqR++;
            if ((int64_t)got[o] == R[o] + (int64_t)v * S[o]) eqH++;
            if ((int64_t)got[o] == R[o] + (int64_t)(op_em & 0xffffu) * S[o]) eqHu++;
            long long dd = llabs((long long)got[o] - R[o]);
            if (dd > maxd) maxd = dd;
        }
        printf("  direct     %-24s BS_CFG=0x%05x OW_CFG=0x%03x MISC=0x%03x op=0x%04x | "
               "R exact %4ld/%zu  H s16(%d) %4ld  u16 %4ld  sent %4ld  max|d-R| %lld",
               a->label, bscfg, owcfg, misc, op_em, eqR, no, v, eqH, eqHu, sent, maxd);
        if (raw) {
            /* multiset overlap: how many of R's and H's values occur anywhere in the BO */
            int64_t *hv = malloc(no * 8), *rv = malloc(no * 8), *bv = malloc(nraw * 8);
            if (hv && rv && bv) {
                for (size_t o = 0; o < no; o++) { rv[o] = R[o]; hv[o] = R[o] + (int64_t)v * S[o]; }
                for (size_t o = 0; o < nraw; o++) bv[o] = raw[o];
                qsort(rv, no, 8, cmp64); qsort(hv, no, 8, cmp64); qsort(bv, nraw, 8, cmp64);
                printf("  | in BO anywhere: R %zu/%zu  H %zu/%zu  (2x-stride R %ld H %ld)",
                       ms_overlap(rv, no, bv, nraw), no, ms_overlap(hv, no, bv, nraw), no,
                       stride2(raw, nraw, R, S, 0, OC, OH, OW, OCp),
                       stride2(raw, nraw, R, S, v, OC, OH, OW, OCp));
            }
            free(hv); free(rv); free(bv);
        }
        printf("\n");
        if (is_control && eqR != (long)no) g_ctrl_bad = 1;
    }
    free(x); free(w); free(got); free(R); free(S); free(raw);
}

static long read_exact(const char *path, void *dst, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t k = fread(dst, 1, n, f);
    fclose(f);
    return k == n ? (long)k : -1;
}

/* `layer` mode, for tflite-rocket/tools/requant_chain.py: one real depthwise layer in the
 * int8 domain, its border pad already materialized by the caller (the program runs pad 0),
 * the weight zero point carried by CPEND at -zw. Channels go in chunks of 64 so each job's
 * feature stays within 8 CBUF banks, the library's depthwise budget. Exit 3 when a 64-channel
 * chunk does not fit (the caller keeps its host model for that layer), 0 on success.
 *
 *   layer C H W KH KW S in_s w_s out_s zx zw zo in.bin w.bin bias.bin out.bin
 *   in [C][H][W] int8, w [C][KH][KW] int8, bias [C] int32, out [C][OH][OW] int8 */
static int layer_mode(int fd, char **a)
{
    const int C = atoi(a[0]), H = atoi(a[1]), W = atoi(a[2]), KH = atoi(a[3]), KW = atoi(a[4]);
    const int S = atoi(a[5]);
    const float in_s = strtof(a[6], NULL), w_s = strtof(a[7], NULL), out_s = strtof(a[8], NULL);
    const int zx = atoi(a[9]), zw = atoi(a[10]), zo = atoi(a[11]);
    const int OH = (H - KH) / S + 1, OW = (W - KW) / S + 1;
    const int Hj = H < 4 ? 4 : H;
    int chunk = 0;
    for (int c = DW_G; c <= rup(C, DW_G); c += DW_G)
        if ((long)c * Hj * W <= 8L * NPU_CBUF_BANK_SIZE) chunk = c;
    if (!chunk || C % 16) { printf("does not fit one pass (C %d, %dx%d)\n", C, H, W); return 3; }
    int8_t *x = malloc((size_t)C * H * W), *w = malloc((size_t)C * KH * KW);
    int8_t *o = malloc((size_t)C * OH * OW);
    int32_t *b = malloc((size_t)C * 4);
    if (read_exact(a[12], x, (size_t)C * H * W) < 0 || read_exact(a[13], w, (size_t)C * KH * KW) < 0 ||
        read_exact(a[14], b, (size_t)C * 4) < 0) { printf("input files short\n"); return 2; }
    int jobs = 0;
    for (int c0 = 0; c0 < C; c0 += chunk) {
        const int cc = C - c0 < chunk ? C - c0 : chunk;
        shape_t s = { "layer", cc, H, W, KH, KW, S, S, 0, 0 };
        uint32_t em = 0;
        int rc = run_dw(fd, &s, x + (size_t)c0 * H * W, w + (size_t)c0 * KH * KW, b + c0,
                        in_s, w_s, out_s, zx, zw, zo, 0, 0, &em, o + (size_t)c0 * OH * OW);
        if (rc) { printf("device run failed (%d) at channel %d\n", rc, c0); return 2; }
        jobs++;
    }
    FILE *f = fopen(a[15], "wb");
    if (!f || fwrite(o, 1, (size_t)C * OH * OW, f) != (size_t)C * OH * OW) { printf("write failed\n"); return 2; }
    fclose(f);
    printf("%d job(s) of <= %d channels, OW_OP 0x%04x", jobs, chunk, (unsigned)(-zw) & 0xffffu);
    free(x); free(w); free(o); free(b);
    return 0;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) { printf("SKIP: no /dev/accel/accel0\n"); return 77; }
    if (argc == 18 && !strcmp(argv[1], "layer")) {
        int rc = layer_mode(fd, argv + 2);
        rocket_close(fd);
        return rc;
    }
    const int do_direct = !(argc > 1 && !strcmp(argv[1], "dw"));

    const shape_t shapes[] = {
        { "64x8x11",   64, 8, 11, 3, 3, 1, 1, 1, 1 },
        { "128x9x7s2", 128, 9, 7, 3, 3, 2, 1, 1, 1 },
        { "64x8x8k5",  64, 8, 8, 5, 5, 1, 1, 2, 2 },
        /* a partial last 64-channel group (C % 64 of 32 and 16): the shipped entry admits
         * these (it checks C % 16 only) */
        { "96x8x11",   96, 8, 11, 3, 3, 1, 1, 1, 1 },
        { "80x8x11",   80, 8, 11, 3, 3, 1, 1, 1, 1 },
    };
    /* conv scale 0.0025: outputs spread over the int8 range at full-range data and a
     * 9-bit effective weight, with a little saturation. */
    const float in_s = 0.05f, w_s = 0.01f, out_s = 0.2f;
    const int zx = 23, zo = -7;
    /* uint8 weight zero points 91, 117, 66 (MobileDet's range) are int8 -37, -11, -62;
     * 25 and 90 are the other sign and a large one. */
    const int zws[] = { -37, -11, -62, 25, 90 };

    printf("cpend_wzp_probe: DPU_BS_OW_OP as the weight zero point\n");
    printf("  zx=%d zo=%d scales %g %g %g; F = TFLite acc + OUT_CVT model; H(v) = bias' + "
           "sum xp*w + v*sum xp\n", zx, zo, in_s, w_s, out_s);

    for (size_t si = 0; si < sizeof shapes / sizeof shapes[0]; si++) {
        const shape_t *s = &shapes[si];
        uint64_t seed = 0x5eed0000ull + si * 101;
        /* a shape with a partial last 64-channel group is a defect witness, not a control */
        dw_arm(fd, s, "control zw=0 op=0",     zx, 0, zo, 0, 0,      in_s, w_s, out_s, seed,
               s->C % DW_G == 0);
        dw_arm(fd, s, "repro zw=0 op=128",     zx, 0, zo, 1, 0x80,   in_s, w_s, out_s, seed, 0);
        dw_arm(fd, s, "zw=0 op=-1",            zx, 0, zo, 1, 0xffff, in_s, w_s, out_s, seed, 0);
        for (size_t zi = 0; zi < sizeof zws / sizeof zws[0]; zi++) {
            int zw = zws[zi];
            dw_arm(fd, s, "mesa op=-zw (s16)",   zx, zw, zo, 0, 0, in_s, w_s, out_s, seed + zi, 0);
            dw_arm(fd, s, "op=0 (refused today)", zx, zw, zo, 1, 0, in_s, w_s, out_s, seed + zi, 0);
            dw_arm(fd, s, "op=+zw (wrong sign)", zx, zw, zo, 1, (uint32_t)zw & 0xffffu,
                   in_s, w_s, out_s, seed + zi, 0);
            dw_arm(fd, s, "op=-zw low byte",     zx, zw, zo, 1, (uint32_t)(-zw) & 0xffu,
                   in_s, w_s, out_s, seed + zi, 0);
        }
    }

    if (do_direct) {
        const direct_arm_t arms[] = {
            { "control (as emitted)",     0, 0, 0, 0,      -1 },
            { "BS live, OD bypass, op37", 1, 0, 1, 37,     -1 },
            { "BS live, CPEND live, op0", 1, 1, 1, 0,      -1 },
            { "CPEND live op=37 qd0",     1, 1, 1, 37,      0 },
            { "CPEND live op=37 qd1",     1, 1, 1, 37,      1 },
            { "CPEND live op=-37 qd1",    1, 1, 1, 0xffdb,  1 },
            { "CPEND live op=0 qd1",      1, 1, 1, 0,       1 },
        };
        for (size_t i = 0; i < sizeof arms / sizeof arms[0]; i++)
            direct_arm(fd, &arms[i], 0xd1ec0000ull, i == 0);
        /* the control again, after every patched job, so a latched state would show */
        direct_arm(fd, &arms[0], 0xd1ec0000ull, 1);
    }

    rocket_close(fd);
    if (g_ctrl_bad) { printf("RESULT: a control arm was not exact; no other row can be read\n"); return 1; }
    printf("RESULT: all arms ran; controls exact\n");
    return 0;
}
