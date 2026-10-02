// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * ew_int_probe.c — does the RK3588 DPU eltwise ALU compute in INTEGER when the whole
 * precision bundle says integer?
 *
 * Our two int32-EW probes (matmul_accum_int8_rocket) moved DPU_EW_CFG and read "the
 * int32 bit patterns added as float". Neither set the precision triple, and DPU 0x4010
 * enumerates 3'd4 as Integer 32bit [TRM Part1]. allbilly/rk3588 measures same-format
 * integer ADD/MAX/MIN on this silicon with the whole bundle, but at INT32 operands
 * inside +-50000, where an int -> fp32 -> int datapath is also exact, so it cannot tell
 * an integer ALU from a float one. This probe can: its INT32 operands sit past 2^24 with
 * odd low bits.
 *
 * The program is DPU-only (flying mode: MRDMA reads the main operand from memory, the
 * ERDMA the second one), and it writes EVERY register it depends on, because a register
 * a program leaves out keeps whatever the last program wrote. The bundle:
 *   DPU 0x4010 out/in/proc precision P; DPU_RDMA FEATURE_MODE_CFG in/proc precision P;
 *   EW and ERDMA element size E; the EW input converter bypassed (INT32) at scale 1;
 *   BRDMA and NRDMA disabled; BS and BN bypassed; OUT_CVT scale 1 shift 0.
 * Cube: one row of W 16-byte atoms, C = 16 / sizeof(T) channels, so input, operand and
 * output are the same flat array and out[i] = op(a[i], b[i]).
 *
 * Each element is scored against six models, and the arm reports how many elements
 * each model explains:
 *   sat    the integer op, saturating to T         (the capability claim)
 *   wrap   the integer op, wrapping modulo 2^bits
 *   fval   T -> fp32, the op in fp32, round to nearest, saturate back to T
 *   fbits  the bits reinterpreted as fp32, the op, reinterpreted back (INT32 only)
 *   a      the main operand passed through (the second operand read as 0 for ADD)
 *   unw    the 0xAA sentinel, unwritten
 *   b16    (INT32) the op with the second operand read as signed int16
 * An fp16 arm (P 2, E 2) scores against the fp16 op instead; it also tests our older
 * reading that a flying main feed plus ERDMA reads the operand as 0.
 *
 * Arms (all by default):   ew_int_probe [dtype] [op] [variant]
 *   dtype   int8 int16 int32 fp16
 *   op      add max min minus mul
 *   variant full     the whole bundle (default)
 *           nodis    BRDMA/NRDMA left enabled (not written)
 *           oldprec  INT32 EW sizes with the precision triple at fp16 (our old probes)
 *   ew_int_probe oldgen   gen_ew_mul_fp16 (the library's retained flying multiply, the
 *           source of the "a two-buffer EW needs a conv main feed" reading) verbatim, after
 *           an fp16 arm and after an INT32 arm have left different OUT_CVT values, then with
 *           OUT_CVT written into it. Measured on both RK1s: with its COMB_USE(5) the job never
 *           completes (vendor: rc -110, nothing written), whatever OUT_CVT holds; with
 *           ROCKET_EW_MUL_COMB=0 or 4 it completes and writes the right product on lane 0 of
 *           every 8-channel atom and zero on the other seven, because its DATA_CUBE_CHANNEL
 *           leaves ORIG_CHANNEL at 0. The fp16 MUL arm above, which differs in those fields,
 *           is exact. So the reading was two register fields, not the datapath.
 * Env: ROCKET_EWI_OWCFG overrides DPU_BS_OW_CFG (default OD_BYPASS only),
 *      ROCKET_EWI_NOTCH writes RDMA SURF_NOTCH / EW_SURF_NOTCH (default 0),
 *      ROCKET_EWI_ORIGC sets DATA_CUBE_CHANNEL's ORIG_CHANNEL (default C-1),
 *      ROCKET_EWI_COMB sets FEATURE_MODE_CFG's COMB_USE (default 0),
 *      ROCKET_EWI_CVTBYP=0/1 forces the EW converter bypass bit.
 * Exit 0 when every "full" arm is exact under sat (fp16: exact fp16; INT32 MUL: under
 * b16, the product with the second operand read as signed int16), 1 otherwise, 2 with no
 * device.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npu_activation.h"
#include "npu_hw.h"
#include "rocket_npu.h"
#include "test_fill.h"

#define OP_REG_DPU_RDMA_P  (BLOCK_DPU_RDMA | PC_OP_01)
#define R_S_POINTER        0x5004
#define R_CUBE_W           0x500C
#define R_CUBE_H           0x5010
#define R_CUBE_C           0x5014
#define R_SRC_BASE         0x5018
#define R_BRDMA_CFG        0x501C
#define R_NRDMA_CFG        0x5028
#define R_ERDMA_CFG        0x5034
#define R_EW_BASE          0x5038
#define R_EW_SURF_STRIDE   0x5040
#define R_FEATURE_MODE     0x5044
#define R_SURF_NOTCH       0x504C
#define R_WEIGHT           0x5068
#define R_EW_SURF_NOTCH    0x506C

enum { P_INT8 = 0, P_INT16 = 1, P_FP16 = 2, P_INT32 = 4 };
enum { OPC_MAX = 0, OPC_MIN = 1, OPC_ADD = 2, OPC_MINUS = 4, OPC_MUL = 100 };
enum { V_FULL, V_NODIS, V_OLDPREC };

#define N_ELEM 4096

typedef struct { const char *name; int prec, esize, bytes; } dtype_t;
static const dtype_t DT[] = {
    {"int8", P_INT8, 1, 1}, {"int16", P_INT16, 2, 2}, {"int32", P_INT32, 3, 4},
    {"fp16", P_FP16, 2, 2},
};
typedef struct { const char *name; int alu; } op_t;
static const op_t OPS[] = {{"add", OPC_ADD}, {"max", OPC_MAX}, {"min", OPC_MIN},
                           {"minus", OPC_MINUS}, {"mul", OPC_MUL}};
static const char *VNAME[] = {"full", "nodis", "oldprec"};

static int build(uint64_t *ops, uint32_t out, uint32_t in, uint32_t ew, const dtype_t *d,
                 const op_t *o, int variant)
{
    const int C = 16 / d->bytes, W = N_ELEM / C;
    const int prec = (variant == V_OLDPREC) ? P_FP16 : d->prec;
    const uint32_t stride = (uint32_t)W << 4;       /* W atoms of 16 bytes */
    uint32_t owcfg = 1u << 1, notch = 0, origc = (uint32_t)(C - 1), comb = 0;
    const char *e;
    if ((e = getenv("ROCKET_EWI_ORIGC"))) origc = (uint32_t)strtoul(e, NULL, 0);
    if ((e = getenv("ROCKET_EWI_COMB")))  comb  = (uint32_t)strtoul(e, NULL, 0) & 7u;
    if ((e = getenv("ROCKET_EWI_OWCFG"))) owcfg = (uint32_t)strtoul(e, NULL, 0);
    if ((e = getenv("ROCKET_EWI_NOTCH"))) notch = (uint32_t)strtoul(e, NULL, 0);
    uint32_t ewcfg = (1u << 28) | ((uint32_t)d->esize << 22) | (1u << 9) | (1u << 7) |
                     (1u << 6);
    if (o->alu == OPC_MUL) ewcfg |= 1u << 2;          /* EW_OP_TYPE = MUL */
    else ewcfg |= (uint32_t)o->alu << 16;             /* EW_ALU_ALGO */
    if (d->prec == P_INT32) ewcfg |= 1u << 8;         /* EW input converter bypassed */
    /* fp16 MUL with the converter live times out on the vendor RK1 (rc -110, nothing
     * written); the shipping fp16 MUL word 0x108003C4 bypasses it. ROCKET_EWI_CVTBYP=0/1
     * forces the bit either way for every arm. */
    if (d->prec == P_FP16 && o->alu == OPC_MUL) ewcfg |= 1u << 8;
    if ((e = getenv("ROCKET_EWI_CVTBYP"))) ewcfg = atoi(e) ? (ewcfg | 1u << 8) : (ewcfg & ~(1u << 8));
    if (o->alu == OPC_MAX || o->alu == OPC_MIN) ewcfg |= 1u << 21;   /* EW_EQUAL_EN */

    int i = 0;
    ops[i++] = NPUOP(OP_REG_DPU, 0xE, DPU_S_POINTER);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, 0xE, R_S_POINTER);
    if (variant != V_NODIS) {
        ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, 1, R_BRDMA_CFG);   /* BRDMA_DISABLE */
        ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, 1, R_NRDMA_CFG);   /* NRDMA_DISABLE */
    }
    ops[i++] = NPUOP(OP_REG_DPU, (15u << 5) | (2u << 1) | 1u, DPU_FEATURE_MODE_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, ((uint32_t)prec << 29) | ((uint32_t)prec << 26) |
                     (uint32_t)prec, DPU_DATA_FORMAT);
    ops[i++] = NPUOP(OP_REG_DPU, out, DPU_DST_BASE_ADD);
    ops[i++] = NPUOP(OP_REG_DPU, stride, DPU_DST_SURF_STRIDE);
    ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)(W - 1), DPU_DATA_CUBE_WIDTH);
    ops[i++] = NPUOP(OP_REG_DPU, 0, DPU_DATA_CUBE_HEIGHT);
    ops[i++] = NPUOP(OP_REG_DPU, 0, DPU_DATA_CUBE_NOTCH_ADDR);
    ops[i++] = NPUOP(OP_REG_DPU, (origc << 16) | (uint32_t)(C - 1), DPU_DATA_CUBE_CHANNEL);
    ops[i++] = NPUOP(OP_REG_DPU, (1u << 6) | (1u << 4) | (1u << 1) | 1u, DPU_BS_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0, DPU_BS_ALU_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0, DPU_BS_MUL_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, owcfg, DPU_BS_OW_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0, DPU_BS_OW_OP);
    ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)(C - 1), DPU_WDMA_SIZE_0);
    ops[i++] = NPUOP(OP_REG_DPU, (uint32_t)(W - 1), DPU_WDMA_SIZE_1);
    ops[i++] = NPUOP(OP_REG_DPU, (1u << 6) | (1u << 4) | (1u << 1) | 1u, DPU_BN_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, ewcfg, DPU_EW_CFG);
    ops[i++] = NPUOP(OP_REG_DPU, 0, DPU_EW_CVT_OFFSET_VALUE);
    ops[i++] = NPUOP(OP_REG_DPU, 1, DPU_EW_CVT_SCALE_VALUE);
    ops[i++] = NPUOP(OP_REG_DPU, 0, DPU_OUT_CVT_OFFSET);
    ops[i++] = NPUOP(OP_REG_DPU, d->prec == P_FP16 ? (1u << 16) | 1u : 1u, DPU_OUT_CVT_SCALE);
    ops[i++] = NPUOP(OP_REG_DPU, 0, DPU_OUT_CVT_SHIFT);
    ops[i++] = NPUOP(OP_REG_DPU, stride, DPU_SURFACE_ADD);
    ops[i++] = NPUOP(OP_REG_DPU, 0, DPU_40C4);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, (uint32_t)(W - 1), R_CUBE_W);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, 0, R_CUBE_H);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, (uint32_t)(C - 1), R_CUBE_C);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, in, R_SRC_BASE);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, (1u << 30) | ((uint32_t)d->esize << 2), R_ERDMA_CFG);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, ew, R_EW_BASE);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, stride, R_EW_SURF_STRIDE);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, notch, R_SURF_NOTCH);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, notch, R_EW_SURF_NOTCH);
    /* fp16 reads through MRDMA_FP16TOFP32 (bit 3), as the working fp16 flying programs do */
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, ((uint32_t)prec << 15) | (15u << 11) |
                     ((uint32_t)prec << 5) | (d->prec == P_FP16 ? 1u << 3 : 0u) | (comb << 8) |
                     1u, R_FEATURE_MODE);
    ops[i++] = NPUOP(OP_REG_DPU_RDMA_P, (1u << 24) | (1u << 16) | (1u << 8) | 1u, R_WEIGHT);
    ops[i++] = NPUOP(OP_NONE, 0, 0);
    ops[i++] = NPUOP(OP_REG_PC, 0, PC_REGISTER_AMOUNTS);
    ops[i++] = NPUOP(OP_40, 0, 0);
    ops[i++] = NPUOP(OP_ENABLE, 0x18, PC_OPERATION_ENABLE);
    return i;
}

static int64_t load(const void *p, size_t i, int bytes)
{
    if (bytes == 1) return ((const int8_t *)p)[i];
    if (bytes == 2) return ((const int16_t *)p)[i];
    return ((const int32_t *)p)[i];
}

static int64_t sat(int64_t v, int bytes)
{
    int64_t hi = bytes == 1 ? 127 : bytes == 2 ? 32767 : 2147483647LL, lo = -hi - 1;
    return v < lo ? lo : v > hi ? hi : v;
}

static int64_t wrap(int64_t v, int bytes)
{
    if (bytes == 1) return (int8_t)(uint8_t)v;
    if (bytes == 2) return (int16_t)(uint16_t)v;
    return (int32_t)(uint32_t)v;
}

static int64_t iop(int alu, int64_t a, int64_t b)
{
    switch (alu) {
    case OPC_MAX: return a > b ? a : b;
    case OPC_MIN: return a < b ? a : b;
    case OPC_ADD: return a + b;
    case OPC_MUL: return a * b;
    default:      return a - b;
    }
}

static float fop(int alu, float a, float b)
{
    switch (alu) {
    case OPC_MAX: return a > b ? a : b;
    case OPC_MIN: return a < b ? a : b;
    case OPC_ADD: return a + b;
    case OPC_MUL: return a * b;
    default:      return a - b;
    }
}

static void fill(void *a, void *b, const dtype_t *d, uint64_t seed)
{
    for (size_t i = 0; i < N_ELEM; i++) {
        uint64_t h1 = tf_hash(seed, 2 * i), h2 = tf_hash(seed, 2 * i + 1);
        if (d->prec == P_FP16) {
            ((_Float16 *)a)[i] = (_Float16)((double)(int)(h1 % 2001) / 8.0 - 125.0);
            ((_Float16 *)b)[i] = (_Float16)((double)(int)(h2 % 2001) / 8.0 - 125.0);
        } else if (d->bytes == 4) {
            /* A quarter near the rails (so ADD/MINUS saturate or wrap), the rest in
             * +-2^27; every value odd, so none is exactly representable past 2^24. */
            int32_t va, vb;
            if ((i & 3) == 0) {
                va = (int32_t)(0x70000001u + (uint32_t)(h1 % 0x0FFFFFFEu));
                vb = (int32_t)(0x70000001u + (uint32_t)(h2 % 0x0FFFFFFEu));
                if (h1 & (1ull << 62)) va = -va;
                if (h2 & (1ull << 62)) vb = -vb;
            } else {
                va = (int32_t)((int64_t)(h1 % (1ull << 28)) - (1ll << 27));
                vb = (int32_t)((int64_t)(h2 % (1ull << 28)) - (1ll << 27));
            }
            ((int32_t *)a)[i] = va | 1;
            ((int32_t *)b)[i] = vb | 1;
        } else if (d->bytes == 2) {
            ((int16_t *)a)[i] = (int16_t)(uint16_t)h1;
            ((int16_t *)b)[i] = (int16_t)(uint16_t)h2;
        } else {
            ((int8_t *)a)[i] = (int8_t)(uint8_t)h1;
            ((int8_t *)b)[i] = (int8_t)(uint8_t)h2;
        }
    }
}

/* Returns 1 if the arm is exact under its capability model. */
static int run_arm(int fd, rocket_bo *bin, rocket_bo *bew, rocket_bo *bout, rocket_bo *breg,
                   const dtype_t *d, const op_t *o, int variant)
{
    if (variant == V_OLDPREC && d->prec != P_INT32) return 1;   /* only meaningful at INT32 */
    const size_t nbytes = (size_t)N_ELEM * d->bytes;
    rocket_bo_prep(fd, bin, 1, 0);
    rocket_bo_prep(fd, bew, 1, 0);
    rocket_bo_prep(fd, bout, 1, 0);
    fill(bin->ptr, bew->ptr, d, 0x6e77u + (uint64_t)d->prec * 16 + (uint64_t)o->alu);
    memset(bout->ptr, 0xAA, nbytes);
    rocket_bo_fini(fd, bin);
    rocket_bo_fini(fd, bew);
    rocket_bo_fini(fd, bout);

    rocket_bo_prep(fd, breg, 1, 0);
    int n = build((uint64_t *)breg->ptr, (uint32_t)bout->dma_address,
                  (uint32_t)bin->dma_address, (uint32_t)bew->dma_address, d, o, variant);
    rocket_bo_fini(fd, breg);

    rocket_task_desc t = {(uint32_t)breg->dma_address, (uint32_t)n};
    uint32_t in_h[3] = {breg->handle, bin->handle, bew->handle}, out_h[1] = {bout->handle};
    int rc = rocket_submit_tasks(fd, &t, 1, in_h, 3, out_h, 1);
    int wr = rocket_bo_prep(fd, bout, 0, 2000000000ull);

    long m_sat = 0, m_wrap = 0, m_fval = 0, m_fbits = 0, m_a = 0, m_unw = 0, m_none = 0, m_b16 = 0;
    long shown = 0;
    for (size_t i = 0; i < N_ELEM; i++) {
        int ok_sat, ok_wrap = 0, ok_fval = 0, ok_fbits = 0, ok_a, ok_unw = 0;
        if (d->prec == P_FP16) {
            _Float16 a = ((_Float16 *)bin->ptr)[i], b = ((_Float16 *)bew->ptr)[i];
            _Float16 g = ((_Float16 *)bout->ptr)[i];
            _Float16 want = (_Float16)fop(o->alu, (float)a, (float)b);
            ok_sat = memcmp(&g, &want, 2) == 0;
            ok_a = memcmp(&g, &a, 2) == 0;
            ok_unw = tf_is_sentinel_f16(g) || ((uint16_t *)bout->ptr)[i] == 0xAAAA;
        } else {
            int64_t a = load(bin->ptr, i, d->bytes), b = load(bew->ptr, i, d->bytes);
            int64_t g = load(bout->ptr, i, d->bytes);
            int64_t r = iop(o->alu, a, b);
            ok_sat = g == sat(r, d->bytes);
            ok_wrap = g == wrap(r, d->bytes);
            double fr = rint((double)fop(o->alu, (float)a, (float)b));
            ok_fval = g == sat((int64_t)(fr > 9e18 ? 9e18 : fr < -9e18 ? -9e18 : fr), d->bytes);
            if (d->bytes == 4) {
                float fa, fb, fg;
                int32_t ia = (int32_t)a, ib = (int32_t)b, ig;
                memcpy(&fa, &ia, 4); memcpy(&fb, &ib, 4);
                fg = fop(o->alu, fa, fb);
                memcpy(&ig, &fg, 4);
                ok_fbits = g == ig;
            }
            ok_a = g == a;
            if (d->bytes == 4) m_b16 += g == sat(iop(o->alu, a, (int16_t)(uint16_t)b), 4);
            uint8_t sb[4] = {0xAA, 0xAA, 0xAA, 0xAA};
            ok_unw = memcmp((const uint8_t *)bout->ptr + i * d->bytes, sb, d->bytes) == 0;
            if (!ok_sat && shown < 4 && variant == V_FULL) {
                printf("      [%zu] a=%lld b=%lld got=%lld want=%lld\n", i, (long long)a,
                       (long long)b, (long long)g, (long long)sat(r, d->bytes));
                shown++;
            }
        }
        m_sat += ok_sat; m_wrap += ok_wrap; m_fval += ok_fval; m_fbits += ok_fbits;
        m_a += ok_a; m_unw += ok_unw;
        m_none += !(ok_sat || ok_wrap || ok_fval || ok_fbits || ok_a || ok_unw);
    }
    rocket_bo_fini(fd, bout);
    const int int32_mul = d->prec == P_INT32 && o->alu == OPC_MUL;
    printf("  %-5s %-5s %-7s rc %d wait %d | of %d: sat %ld wrap %ld fval %ld fbits %ld "
           "a %ld unw %ld none %ld b16 %ld -> %s\n", d->name, o->name, VNAME[variant], rc, wr,
           N_ELEM, m_sat, m_wrap, m_fval, m_fbits, m_a, m_unw, m_none, m_b16,
           (rc == 0 && wr == 0 && m_sat == N_ELEM) ? "EXACT"
           : (rc == 0 && wr == 0 && int32_mul && m_b16 == N_ELEM) ? "EXACT (b16)" : "not exact");
    /* INT32 MUL is scored on the b16 model: its second operand is read as signed int16
     * (allbilly/rk3588's reading), so a full-int32 product is not the claim there. */
    if (int32_mul) return rc == 0 && wr == 0 && m_b16 == N_ELEM;
    return rc == 0 && wr == 0 && m_sat == N_ELEM;
}

/* gen_ew_mul_fp16 verbatim (insert_cvt 0), or with OUT_CVT offset 0 / scale fp32->fp16 x1 /
 * shift 0 written before its trailer (insert_cvt 1). Scored against the fp16 product. */
static void run_oldgen(int fd, rocket_bo *bin, rocket_bo *bew, rocket_bo *bout, rocket_bo *breg,
                       int insert_cvt, const char *label)
{
    rocket_bo_prep(fd, bin, 1, 0);
    rocket_bo_prep(fd, bew, 1, 0);
    rocket_bo_prep(fd, bout, 1, 0);
    fill(bin->ptr, bew->ptr, &DT[3], 0x01d6e5ull);
    memset(bout->ptr, 0xAA, (size_t)N_ELEM * 2);
    rocket_bo_fini(fd, bin);
    rocket_bo_fini(fd, bew);
    rocket_bo_fini(fd, bout);

    uint64_t ops[128];
    ew_mul_params_t p = {bin->dma_address, bew->dma_address, bout->dma_address, N_ELEM, ops, 0};
    if (gen_ew_mul_fp16(&p) != 0) { printf("  %s: gen_ew_mul_fp16 refused\n", label); return; }
    int n = p.task_count;
    if (insert_cvt) {   /* before the 4-op trailer */
        memmove(&ops[n - 1], &ops[n - 4], 4 * sizeof ops[0]);
        ops[n - 4] = NPUOP(OP_REG_DPU, 0, DPU_OUT_CVT_OFFSET);
        ops[n - 3] = NPUOP(OP_REG_DPU, (1u << 16) | 1u, DPU_OUT_CVT_SCALE);
        ops[n - 2] = NPUOP(OP_REG_DPU, 0, DPU_OUT_CVT_SHIFT);
        n += 3;
    }
    rocket_bo_prep(fd, breg, 1, 0);
    memcpy(breg->ptr, ops, (size_t)n * sizeof ops[0]);
    rocket_bo_fini(fd, breg);
    rocket_task_desc t = {(uint32_t)breg->dma_address, (uint32_t)n};
    uint32_t in_h[3] = {breg->handle, bin->handle, bew->handle}, out_h[1] = {bout->handle};
    int rc = rocket_submit_tasks(fd, &t, 1, in_h, 3, out_h, 1);
    int wr = rocket_bo_prep(fd, bout, 0, 2000000000ull);
    long ok = 0, zero = 0, a_pass = 0, unw = 0, ok_lane0 = 0;
    for (size_t i = 0; i < N_ELEM; i++) {
        _Float16 a = ((_Float16 *)bin->ptr)[i], b = ((_Float16 *)bew->ptr)[i];
        _Float16 g = ((_Float16 *)bout->ptr)[i], want = (_Float16)((float)a * (float)b);
        ok += memcmp(&g, &want, 2) == 0;
        ok_lane0 += (i % 8 == 0) && memcmp(&g, &want, 2) == 0;
        zero += (float)g == 0.0f;
        a_pass += memcmp(&g, &a, 2) == 0;
        unw += ((uint16_t *)bout->ptr)[i] == 0xAAAA;
    }
    rocket_bo_fini(fd, bout);
    printf("  %-44s rc %d wait %d | of %d: a*b %ld (lane 0 of each atom: %ld of %d) zero %ld "
           "a %ld unw %ld\n", label, rc, wr, N_ELEM, ok, ok_lane0, N_ELEM / 8, zero, a_pass, unw);
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); return 2; }

    rocket_bo bin, bew, bout, breg;
    if (rocket_bo_alloc32(fd, 64 * 1024, &bin) || rocket_bo_alloc32(fd, 64 * 1024, &bew) ||
        rocket_bo_alloc32(fd, 64 * 1024, &bout) || rocket_bo_alloc32(fd, 4096, &breg)) {
        fprintf(stderr, "bo alloc failed\n");
        return 1;
    }
    printf("columns: elements (of %d) each model explains; sat = the saturating integer op, "
           "wrap = wrapping, fval = via fp32 values, fbits = int32 bits read as fp32, "
           "a = main operand passed through, unw = unwritten, none = no model\n", N_ELEM);

    int fail = 0;
    if (argc > 1 && !strcmp(argv[1], "oldgen")) {
        run_arm(fd, &bin, &bew, &bout, &breg, &DT[3], &OPS[0], V_FULL);   /* fp16 OUT_CVT */
        run_oldgen(fd, &bin, &bew, &bout, &breg, 0, "gen_ew_mul_fp16 after the fp16 arm");
        run_arm(fd, &bin, &bew, &bout, &breg, &DT[2], &OPS[0], V_FULL);   /* int OUT_CVT */
        run_oldgen(fd, &bin, &bew, &bout, &breg, 0, "gen_ew_mul_fp16 after the int32 arm");
        run_arm(fd, &bin, &bew, &bout, &breg, &DT[2], &OPS[0], V_FULL);
        run_oldgen(fd, &bin, &bew, &bout, &breg, 1, "... with OUT_CVT written, after int32");
        rocket_bo_free(fd, &bin); rocket_bo_free(fd, &bew); rocket_bo_free(fd, &bout);
        rocket_bo_free(fd, &breg);
        rocket_close(fd);
        return 0;
    }
    for (size_t di = 0; di < sizeof DT / sizeof DT[0]; di++) {
        if (argc > 1 && strcmp(argv[1], DT[di].name)) continue;
        for (size_t oi = 0; oi < sizeof OPS / sizeof OPS[0]; oi++) {
            if (argc > 2 && strcmp(argv[2], OPS[oi].name)) continue;
            for (int v = V_FULL; v <= V_OLDPREC; v++) {
                if (argc > 3 && strcmp(argv[3], VNAME[v])) continue;
                int exact = run_arm(fd, &bin, &bew, &bout, &breg, &DT[di], &OPS[oi], v);
                if (v == V_FULL && !exact) fail = 1;
            }
        }
    }
    rocket_bo_free(fd, &bin); rocket_bo_free(fd, &bew); rocket_bo_free(fd, &bout);
    rocket_bo_free(fd, &breg);
    rocket_close(fd);
    printf("%s\n", fail ? "FAIL: at least one full-bundle arm is not exact"
                        : "PASS: every full-bundle arm is exact");
    return fail;
}
