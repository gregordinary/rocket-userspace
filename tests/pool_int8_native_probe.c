// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * pool_int8_native_probe.c — does the RK3588 PPU pool natively in int8?
 *
 * WHAT THIS RE-OPENS. This tree records "the RK3588 PPU has no native int8 pooling
 * precision", measured 2026-06-22 by programming PPU_DATA_FORMAT.PROC_PRECISION = 0 and
 * PPU_RDMA_DATA_FORMAT.IN_PRECISION = 0 over a packed int8 C2=16 cube and reading back
 * garbage. But IN_PRECISION is a STORAGE WIDTH, not a dtype selector: 0x7030[1:0]
 * enumerates 2'd0: 4bit, 2'd1: 8bit, 2'd2: 16bit, 2'd3: 32bit [TRM, RK3588 Part1;
 * registers.xml agrees on the field position]. So that probe asked for a 4-bit input
 * stream over a byte cube, and the garbage is what that implies rather than a property
 * of the silicon. The shipping fp16 path uses 2, which is correct under BOTH readings,
 * which is why no gate could ever have caught it.
 *
 * WHY THIS IS A MAP AND NOT A RETEST OF ONE GUESS. The correct int8 cell is a PAIR --
 * a processing precision and a storage width -- and the two readings disagree about
 * which pair it is (storage-width says in_prec 1, a dtype reading says 0). A sibling
 * part is evidence for neither: rocket_pool_rk3576.c pools int8 with 0x40 in its own
 * 0x7030, whose low bits are 0, on a part whose register packing is known to differ.
 * So sweep the pair and print what each cell computes. A single cell that passes is an
 * answer; a single cell that fails is not, which is exactly how the recorded negative
 * came about.
 *
 * MAX FIRST, AND UNPADDED FIRST. Max has no reciprocal and no rounding, so it separates
 * "the datapath reads int8 bytes" from "the average's divisor is encoded right", which
 * are different questions with different fixes. Unpadded first because the pad-fill
 * register is dtype-dependent too (the fp16 path writes -inf as 0xFC00) and a wrong
 * fill corrupts only the border -- a small, plausible, easily-missed error.
 *
 * SCORING. Against an independent integer golden computed here in plain C, never
 * against the library's fp16-routed int8 entry, which would agree with itself.
 *
 * Usage: pool_int8_native_probe [max|avg|min]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "npu_hw.h"
#include "npu_pool.h"
#include "npu_matmul.h"   /* feature_data */
#include "rocket_op.h"

/* The two PPU block opcodes, spelled as npu_regcmd.c spells them (they are defined in
 * the .c, not the header, so a probe that emits its own program restates them). */
#define OP_REG_PPU       (BLOCK_PPU | PC_OP_01)        /* 0x4001 */
#define OP_REG_PPU_RDMA  (BLOCK_PPU_RDMA | PC_OP_01)   /* 0x8001 */

#define C2_I8   16        /* int8 feature cube channel atom */
#define BO_SLACK 256

typedef struct {
    const char *name;
    int c, ih, iw, kh, kw, sy, sx, pt, pl, pb, pr;
    int method;
} shape_t;

static int out_dim(int i, int k, int s, int p0, int p1)
{ return (i + p0 + p1 - k) / s + 1; }

/* Independent integer golden. MAX: padding never contributes. AVG: count-include-pad,
 * out-of-range taps contribute 0, round-to-nearest (ties away from zero, which is what
 * lrintf-free integer rounding below does for a positive quotient and mirrors it for a
 * negative one). MIN mirrors MAX. */
/* THE CHIP'S OWN AVERAGE ROUNDING, as rocket_pool_rk3576.c models it for the sibling
 * part: round half away from zero, except that an EVEN window whose remainder is exactly
 * half steps back toward zero -- always when the Q16 reciprocal is inexact (a truncated
 * reciprocal sits a hair under 1/n, so a sum landing on a half falls to the smaller
 * magnitude), and only for an ODD quotient when it is exact, which is round-half-to-even.
 * Scored here as a SECOND column beside the naive model, because a difference between the
 * two is a rounding question and a difference from both is a datapath one. */
static int r76_avg_round(long sum, int dw, int dh)
{
    long n = (long)dw * dh, half = n / 2;
    uint64_t magic = (uint64_t)(((uint64_t)1 << 32) / (uint64_t)n) + 1u;
    int even = (n & 1) == 0;
    int exact_recip = (0x10000 % dw) == 0 && (0x10000 % dh) == 0;
    long a = sum >= 0 ? sum : -sum;
    long q = (long)(((uint64_t)(a + half) * magic) >> 32);
    if (sum < 0) q = -q;
    if (even) {
        long r = sum - q * n;
        if ((r == half || r == -half) && (!exact_recip || (q & 1)))
            q += (sum >= 0) ? -1 : 1;
    }
    return (int)q;
}

static void golden(const shape_t *s, const int8_t *in, int OH, int OW, int *out,
                   int chip_round)
{
    for (int c = 0; c < s->c; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int acc = (s->method == POOL_METHOD_MAX) ? -128
                        : (s->method == POOL_METHOD_MIN) ?  127 : 0;
                for (int kh = 0; kh < s->kh; kh++) {
                    int ih = oh*s->sy + kh - s->pt;
                    for (int kw = 0; kw < s->kw; kw++) {
                        int iw = ow*s->sx + kw - s->pl;
                        int oor = (ih < 0 || ih >= s->ih || iw < 0 || iw >= s->iw);
                        if (oor) continue;          /* AVG: adds 0; MAX/MIN: skipped */
                        int v = in[((size_t)c*s->ih + ih)*s->iw + iw];
                        if      (s->method == POOL_METHOD_MAX) { if (v > acc) acc = v; }
                        else if (s->method == POOL_METHOD_MIN) { if (v < acc) acc = v; }
                        else acc += v;
                    }
                }
                if (s->method == POOL_METHOD_AVG) {
                    int n = s->kh * s->kw;
                    acc = chip_round ? r76_avg_round(acc, s->kw, s->kh)
                        : ((acc >= 0) ? (2*acc + n) / (2*n) : -((-2*acc + n) / (2*n)));
                }
                out[((size_t)c*OH + oh)*OW + ow] = acc;
            }
}

/* The PPU program, with the two precision fields and the reciprocal encoding as
 * parameters. Everything else is gen_pool_fp16's validated sequence with the int8
 * cube's strides (C2=16, one byte an element). */
static uint32_t emit(uint64_t *ops, const shape_t *s, int OH, int OW,
                     uint32_t in_dma, uint32_t out_dma,
                     unsigned proc_prec, unsigned in_prec, int int_recip)
{
    uint32_t i = 0;
    const uint32_t in_line  = (uint32_t)s->iw * C2_I8;
    const uint32_t in_surf  = (uint32_t)s->ih * s->iw * C2_I8;
    const uint32_t out_surf = (uint32_t)OH * OW * C2_I8;

    ops[i++] = NPUOP(OP_REG_PPU,      0xE, PPU_S_POINTER);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, 0xE, PPU_RDMA_S_POINTER);

    ops[i++] = NPUOP(OP_REG_PPU, (s->iw - 1) & 0x1FFF, PPU_DATA_CUBE_IN_WIDTH);
    ops[i++] = NPUOP(OP_REG_PPU, (s->ih - 1) & 0x1FFF, PPU_DATA_CUBE_IN_HEIGHT);
    ops[i++] = NPUOP(OP_REG_PPU, (s->c  - 1) & 0x1FFF, PPU_DATA_CUBE_IN_CHANNEL);
    ops[i++] = NPUOP(OP_REG_PPU, (OW - 1) & 0x1FFF, PPU_DATA_CUBE_OUT_WIDTH);
    ops[i++] = NPUOP(OP_REG_PPU, (OH - 1) & 0x1FFF, PPU_DATA_CUBE_OUT_HEIGHT);
    ops[i++] = NPUOP(OP_REG_PPU, (s->c  - 1) & 0x1FFF, PPU_DATA_CUBE_OUT_CHANNEL);

    ops[i++] = NPUOP(OP_REG_PPU, (1u << 4) | ((uint32_t)s->method & 0x3),
                     PPU_OPERATION_MODE_CFG);

    ops[i++] = NPUOP(OP_REG_PPU,
                     ((uint32_t)(s->kw - 1) & 0xF)
                   | (((uint32_t)(s->kh - 1) & 0xF) << 8)
                   | (((uint32_t)(s->sx - 1) & 0xF) << 16)
                   | (((uint32_t)(s->sy - 1) & 0xF) << 20), PPU_POOLING_KERNEL_CFG);

    /* The average's per-axis reciprocal. The fp16 encoding is what ships; the integer
     * Q16 one is what an integer datapath wants and what the RK3576 pool programs. */
    uint32_t rw = 0, rh = 0;
    if (s->method == POOL_METHOD_AVG) {
        rw = int_recip ? (0x10000u / (unsigned)s->kw) : ppu_recip_kernel_fp16(s->kw);
        rh = int_recip ? (0x10000u / (unsigned)s->kh) : ppu_recip_kernel_fp16(s->kh);
    }
    ops[i++] = NPUOP(OP_REG_PPU, rw & 0x1FFFF, PPU_RECIP_KERNEL_WIDTH);
    ops[i++] = NPUOP(OP_REG_PPU, rh & 0x1FFFF, PPU_RECIP_KERNEL_HEIGHT);

    uint32_t pad = ((uint32_t)s->pl & 0x7)
                 | (((uint32_t)s->pt & 0x7) << 4)
                 | (((uint32_t)s->pr & 0x7) << 8)
                 | (((uint32_t)s->pb & 0x7) << 12);
    ops[i++] = NPUOP(OP_REG_PPU, pad, PPU_POOLING_PADDING_CFG);
    /* Pad fill. On the integer path the "never wins" value is the dtype minimum in the
     * sign-extended field the RK3576 pool uses (0x7ff80 == -128); on the fp16 path it is
     * -inf. Only reached when this shape actually pads. */
    uint32_t fill = 0;
    if (pad && s->method == POOL_METHOD_MAX) fill = (proc_prec == 2) ? 0x0000FC00u : 0x0007FF80u;
    if (pad && s->method == POOL_METHOD_MIN) fill = (proc_prec == 2) ? 0x00007C00u : 0x0000007Fu;
    ops[i++] = NPUOP(OP_REG_PPU, fill, PPU_PADDING_VALUE_1_CFG);
    ops[i++] = NPUOP(OP_REG_PPU, 0x0,  PPU_PADDING_VALUE_2_CFG);

    ops[i++] = NPUOP(OP_REG_PPU, out_dma, PPU_DST_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_PPU, out_surf, PPU_DST_SURF_STRIDE);
    /* INDEX_ADD[31:4] carries the out surf stride; PROC_PRECISION is [2:0]. The stride
     * is 16-aligned for a C2=16 byte cube, so the OR stays clean. */
    ops[i++] = NPUOP(OP_REG_PPU, out_surf | (proc_prec & 0x7), PPU_DATA_FORMAT);
    ops[i++] = NPUOP(OP_REG_PPU, 0x3, PPU_MISC_CTRL);

    ops[i++] = NPUOP(OP_REG_PPU_RDMA, (s->iw - 1) & 0x1FFF, PPU_RDMA_CUBE_IN_WIDTH);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, (s->ih - 1) & 0x1FFF, PPU_RDMA_CUBE_IN_HEIGHT);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, (s->c  - 1) & 0x1FFF, PPU_RDMA_CUBE_IN_CHANNEL);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, in_dma,   PPU_RDMA_SRC_BASE_ADDR);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, in_line,  PPU_RDMA_SRC_LINE_STRIDE);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, in_surf,  PPU_RDMA_SRC_SURF_STRIDE);
    ops[i++] = NPUOP(OP_REG_PPU_RDMA, in_prec & 0x3, PPU_RDMA_DATA_FORMAT);

    ops[i++] = NPUOP(OP_NONE, 0x0, 0x0);
    ops[i++] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
    ops[i++] = NPUOP(OP_40, 0x0, 0x0);
    ops[i++] = NPUOP(OP_ENABLE, 0x60, PC_OPERATION_ENABLE);
    return i;
}

/* Returns 0 and fills *maxerr / *nwrong, or <0 if the submit itself failed. */
static int run_cell(int fd, const shape_t *s, const int8_t *in, const int *gold,
                    const int *gold2, int OH, int OW, unsigned proc_prec,
                    unsigned in_prec, int int_recip,
                    int *maxerr, int *nwrong, int *allzero,
                    int *maxerr2, int *nwrong2)
{
    const int C1 = (s->c + C2_I8 - 1) / C2_I8;
    const size_t in_elems  = (size_t)C1 * s->ih * s->iw * C2_I8;
    const size_t out_elems = (size_t)C1 * OH * OW * C2_I8;
    rocket_bo guard = {0}, in_bo = {0}, rc_bo = {0}, out_bo = {0};
    uint64_t regs[64] = {0};
    int ret = -1;

    if (rocket_bo_alloc(fd, 4096, &guard) ||
        rocket_bo_alloc(fd, in_elems + BO_SLACK, &in_bo) ||
        rocket_bo_alloc(fd, sizeof(regs), &rc_bo) ||
        rocket_bo_alloc(fd, out_elems + BO_SLACK, &out_bo)) goto out;
    { rocket_bo *const set[] = { &in_bo, &rc_bo, &out_bo };
      if (rocket_op_iova_overflow("pool_int8_native_probe", set, 3)) goto out; }

    rocket_bo_prep(fd, &in_bo, 1, 0);
    memset(in_bo.ptr, 0, in_bo.size);
    { int8_t *dst = in_bo.ptr;
      for (int c = 0; c < s->c; c++)
        for (int h = 0; h < s->ih; h++)
          for (int w = 0; w < s->iw; w++)
            dst[feature_data(s->c, s->ih, s->iw, C2_I8, c+1, h+1, w+1)] =
                in[((size_t)c*s->ih + h)*s->iw + w]; }
    rocket_bo_fini(fd, &in_bo);

    uint32_t n = emit(regs, s, OH, OW, (uint32_t)in_bo.dma_address,
                      (uint32_t)out_bo.dma_address, proc_prec, in_prec, int_recip);

    /* A sentinel, not a zero fill: a program that writes NOTHING and one that writes
     * zeroes are the same table otherwise, and that confound is how a dead cell reads
     * as a computing one. 0x5A is outside any plausible pooled value here. */
    rocket_bo_prep(fd, &out_bo, 1, 0);
    memset(out_bo.ptr, 0x5A, out_bo.size);
    rocket_bo_fini(fd, &out_bo);

    { rocket_bo *const ins[] = { &in_bo };
      ret = rocket_op_submit_one_flags(fd, "pool_int8_native_probe", &rc_bo, regs, n,
                                       ins, 1, &out_bo,
                                       rocket_ppu_done_supported() ? ROCKET_JOB_PPU_DONE : 0u); }
    if (ret) goto out;

    *maxerr = 0; *nwrong = 0; *allzero = 1; *maxerr2 = 0; *nwrong2 = 0;
    { const int8_t *src = out_bo.ptr;
      for (int c = 0; c < s->c; c++)
        for (int oh = 0; oh < OH; oh++)
          for (int ow = 0; ow < OW; ow++) {
            size_t idx = ((size_t)c*OH + oh)*OW + ow;
            int v = src[feature_data(s->c, OH, OW, C2_I8, c+1, oh+1, ow+1)];
            int e = v - gold[idx];  if (e < 0) e = -e;
            int e2 = v - gold2[idx]; if (e2 < 0) e2 = -e2;
            if (v != 0) *allzero = 0;
            if (e)  { (*nwrong)++;  if (e  > *maxerr)  *maxerr  = e;  }
            if (e2) { (*nwrong2)++; if (e2 > *maxerr2) *maxerr2 = e2; }
          } }
    rocket_bo_fini(fd, &out_bo);
    ret = 0;
out:
    rocket_bo_free(fd, &out_bo); rocket_bo_free(fd, &rc_bo);
    rocket_bo_free(fd, &in_bo);  rocket_bo_free(fd, &guard);
    return ret;
}

int main(int argc, char **argv)
{
    int only = -1;
    if (argc > 1) {
        if      (!strcmp(argv[1], "max")) only = POOL_METHOD_MAX;
        else if (!strcmp(argv[1], "avg")) only = POOL_METHOD_AVG;
        else if (!strcmp(argv[1], "min")) only = POOL_METHOD_MIN;
    }

    static const shape_t SHAPES[] = {
      {"max-2x2-s2-c16",  16, 8, 8, 2,2, 2,2, 0,0,0,0, POOL_METHOD_MAX},
      {"max-3x3-s1-c16",  16, 8, 8, 3,3, 1,1, 0,0,0,0, POOL_METHOD_MAX},
      {"max-2x3-s1-c16",  16, 8, 8, 2,3, 1,1, 0,0,0,0, POOL_METHOD_MAX},
      {"max-2x2-s2-c32",  32, 8, 8, 2,2, 2,2, 0,0,0,0, POOL_METHOD_MAX},
      {"max-2x2-s2-pad1", 16, 7, 7, 2,2, 2,2, 1,1,0,0, POOL_METHOD_MAX},
      {"min-2x2-s2-c16",  16, 8, 8, 2,2, 2,2, 0,0,0,0, POOL_METHOD_MIN},
      {"avg-2x2-s2-c16",  16, 8, 8, 2,2, 2,2, 0,0,0,0, POOL_METHOD_AVG},
      {"avg-4x4-s4-c16",  16, 8, 8, 4,4, 4,4, 0,0,0,0, POOL_METHOD_AVG},
      {"avg-2x3-s1-c16",  16, 8, 8, 2,3, 1,1, 0,0,0,0, POOL_METHOD_AVG},
    };
    const int NS = (int)(sizeof(SHAPES)/sizeof(SHAPES[0]));

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no NPU device\n"); return 77; }
    printf("pool_int8_native_probe: PPU int8 precision map\n");
    printf("  proc = PPU_DATA_FORMAT[2:0], in = PPU_RDMA_DATA_FORMAT[1:0] (storage width:"
           " 0=4b 1=8b 2=16b 3=32b)\n");
    printf("  recip: I = integer Q16 0x10000/k, F = fp16(65536/k) bits (avg only)\n\n");

    int any_exact = 0;
    for (int si = 0; si < NS; si++) {
        const shape_t *s = &SHAPES[si];
        if (only >= 0 && s->method != only) continue;
        int OH = out_dim(s->ih, s->kh, s->sy, s->pt, s->pb);
        int OW = out_dim(s->iw, s->kw, s->sx, s->pl, s->pr);
        size_t ni = (size_t)s->c*s->ih*s->iw, no = (size_t)s->c*OH*OW;
        int8_t *in = malloc(ni); int *gold = malloc(no*sizeof(int));
        int *gold2 = malloc(no*sizeof(int));
        if (!in || !gold || !gold2) { free(in); free(gold); free(gold2); continue; }
        /* A varied, sign-crossing, non-periodic fill: a periodic one cannot see a tile
         * or channel mix-up, and an all-positive one cannot see a sign error. */
        for (size_t i = 0; i < ni; i++) in[i] = (int8_t)((int)((i*37 + i/13*11) % 251) - 125);
        golden(s, in, OH, OW, gold, 0);   /* naive: half away from zero      */
        golden(s, in, OH, OW, gold2, 1);  /* the chip's model (RK3576 divisor) */

        printf("%-16s c=%d %dx%d k=%dx%d s=%dx%d pad=%d%d%d%d -> %dx%d\n",
               s->name, s->c, s->ih, s->iw, s->kh, s->kw, s->sy, s->sx,
               s->pt, s->pl, s->pb, s->pr, OH, OW);
        for (unsigned pp = 0; pp <= 2; pp++) {
            for (unsigned ip = 0; ip <= 3; ip++) {
                int nrec = (s->method == POOL_METHOD_AVG) ? 2 : 1;
                for (int r = 0; r < nrec; r++) {
                    int me = 0, nw = 0, az = 0, me2 = 0, nw2 = 0;
                    int rc = run_cell(fd, s, in, gold, gold2, OH, OW, pp, ip, r == 0,
                                      &me, &nw, &az, &me2, &nw2);
                    const char *rl = (s->method == POOL_METHOD_AVG) ? (r == 0 ? " I" : " F") : "  ";
                    if (rc) { printf("   proc=%u in=%u%s  SUBMIT FAILED (%d)\n", pp, ip, rl, rc); continue; }
                    if (nw2 == 0) {
                        printf("   proc=%u in=%u%s  EXACT vs chip model%s\n", pp, ip, rl,
                               nw ? "  (naive model differs: rounding only)" : "");
                        any_exact = 1;
                    } else if (nw == 0) {
                        printf("   proc=%u in=%u%s  EXACT vs naive model\n", pp, ip, rl);
                        any_exact = 1;
                    } else {
                        printf("   proc=%u in=%u%s  naive wrong %d/%zu maxerr %d | chip wrong %d maxerr %d%s\n",
                               pp, ip, rl, nw, no, me, nw2, me2, az ? "  (all-zero output)" : "");
                    }
                }
            }
        }
        printf("\n");
        free(in); free(gold); free(gold2);
    }
    rocket_close(fd);
    printf("%s\n", any_exact ? "AT LEAST ONE CELL IS EXACT -> native int8 pooling exists"
                             : "no exact cell -> the recorded negative stands for these shapes");
    return 0;
}
