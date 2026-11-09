// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * regcmd_golden_gate.c — host-only gate: does every regcmd generator still emit, word for
 * word, the program recorded in tests/data/regcmd-golden/golden.txt?
 *
 * No NPU. The gate runs each RK3588 generator (the six matmul datatypes, the four
 * convolution variants with their bias, per-channel, int8-out, LUT-epilogue and
 * deconvolution branches, the LUT activation, the elementwise multiply and the PPU pool)
 * and each RK3576 generator over a fixed sweep, hashes the words each configuration
 * writes, and compares the hashes against the golden file. A configuration a generator
 * refuses is recorded by its return code, so a refusal that starts or stops is a
 * difference too.
 *
 * The sweep runs once at default settings and once per profile below. A profile sets one
 * or more of the generators' ROCKET_* environment variables (the diagnostic overrides the
 * RE tests drive) and reruns the part of the sweep that reads them. Each profile runs in a
 * forked child with every inherited ROCKET_* variable cleared, because several of the
 * variables are read once per process and the chip profile is resolved once per process.
 *
 * The golden file holds one line per group: a generator at one shape (default sweep) or a
 * generator under one profile. Each line carries the configuration count, the refusals,
 * the words written and a 64-bit FNV-1a hash over every configuration's return code and
 * words, in sweep order. A mismatch names the group. `--dump FILTER` prints every word of
 * the configurations whose "profile tag" contains FILTER, so the same build run against
 * two encoders diffs to the register:
 *
 *     regcmd_golden_gate [GOLDEN]          check (exit 1 on a difference)
 *     regcmd_golden_gate --write GOLDEN    rewrite the golden file from this build
 *     regcmd_golden_gate --dump FILTER     print "profile tag i target reg value" lines
 *
 * What a pass does not show: the sweep is a finite set of shapes and settings, so a
 * generator that changes only outside it passes, and the hash says nothing about what the
 * hardware computes. The device gates (`ctest` on the RK1, `tools/gates-rk3576.sh`) are
 * what reach the rest.
 */
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "npu_hw.h"
#include "npu_matmul.h"
#include "npu_activation.h"
#include "npu_pool.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_log.h"

extern char **environ;

#define NOPS (1u << 16)
static uint64_t *ops;

/* ---- recording ------------------------------------------------------------------- */

static const char *g_profile = "default";
static const char *g_dump;          /* --dump filter, or NULL                          */
static FILE *g_out;

static struct {
    char     name[160];
    unsigned configs, refused;
    unsigned long long words;
    uint64_t h;
    int      open;
} g_grp;

#define FNV_OFF 0xcbf29ce484222325ull
#define FNV_PRIME 0x100000001b3ull

static uint64_t fnv_u64(uint64_t h, uint64_t v)
{
    for (int b = 0; b < 8; b++) {
        h ^= (v >> (8 * b)) & 0xFF;
        h *= FNV_PRIME;
    }
    return h;
}

static void group_close(void)
{
    if (!g_grp.open) return;
    if (!g_dump)
        fprintf(g_out, "%s %s configs=%u refused=%u words=%llu h=%016llx\n", g_profile,
                g_grp.name, g_grp.configs, g_grp.refused, g_grp.words,
                (unsigned long long)g_grp.h);
    g_grp.open = 0;
}

static void group_open(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void group_open(const char *fmt, ...)
{
    va_list ap;
    group_close();
    va_start(ap, fmt);
    vsnprintf(g_grp.name, sizeof g_grp.name, fmt, ap);
    va_end(ap);
    g_grp.configs = g_grp.refused = 0;
    g_grp.words = 0;
    g_grp.h = FNV_OFF;
    g_grp.open = 1;
}

/* One configuration: `rc` is the generator's return, `n` the words it reports. */
static void record(const char *tag, int rc, long n)
{
    if (rc < 0 || n < 0) n = 0;
    if ((unsigned long)n > NOPS) {
        fprintf(stderr, "regcmd_golden_gate: %s reports %ld words, past the buffer\n", tag, n);
        exit(3);
    }
    g_grp.configs++;
    if (rc < 0) g_grp.refused++;
    g_grp.words += (unsigned long long)n;
    g_grp.h = fnv_u64(g_grp.h, (uint64_t)(int64_t)rc);
    g_grp.h = fnv_u64(g_grp.h, (uint64_t)n);
    for (long i = 0; i < n; i++) g_grp.h = fnv_u64(g_grp.h, ops[i]);

    if (g_dump) {
        char full[320];
        snprintf(full, sizeof full, "%s %s/%s", g_profile, g_grp.name, tag);
        if (!strstr(full, g_dump)) return;
        fprintf(g_out, "%s rc=%d n=%ld\n", full, rc, n);
        for (long i = 0; i < n; i++)
            fprintf(g_out, "%s %4ld %04x %04x %08x\n", full, i, (unsigned)(ops[i] >> 48),
                    (unsigned)(ops[i] & 0xFFFF), (unsigned)(ops[i] >> 16));
    }
}

static void clear_ops(void) { memset(ops, 0, sizeof(uint64_t) * 8192); }

/* ---- the RK3588 sweep ------------------------------------------------------------- */

typedef int (*mm_fn)(gemm_params_t *);
static const struct { const char *name; mm_fn fn; } MM[] = {
    { "fp16", gen_gemm_fp16 }, { "int8", gen_gemm_int8 }, { "int4", gen_gemm_int4 },
    { "int16", gen_gemm_int16 }, { "bf16", gen_gemm_bf16 }, { "tf32", gen_gemm_tf32 },
};
#define NMM (sizeof MM / sizeof MM[0])

/* acc/ew pairs: plain, accumulate with ADD, MUL, MAX and MIN */
static const int ACC_EW[][3] = { {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 2}, {1, 0, 3} };

static void mm_one(mm_fn fn, int m, int k, int n, int f32, int ae, int reuse, int rm)
{
    gemm_params_t p;
    memset(&p, 0, sizeof p);
    p.m = (uint16_t)m; p.k = (uint16_t)k; p.n = (uint16_t)n;
    p.input_iova = 0x81000040u; p.weights_iova = 0x82000000u; p.output_iova = 0x83000080u;
    p.tasks = ops; p.fp16_out = (uint8_t)f32;
    p.accumulate = (uint8_t)ACC_EW[ae][0];
    p.add_iova = p.accumulate ? 0x84000100u : 0;
    p.ew_mul = (uint8_t)ACC_EW[ae][1]; p.ew_op = (uint8_t)ACC_EW[ae][2];
    p.weight_reuse = (uint8_t)(reuse & 1); p.data_reuse = (uint8_t)((reuse >> 1) & 1);
    if (rm & 1) p.a_row_atoms = (uint32_t)(k * 2 / 16 + 4);
    if (rm & 2) p.c_row_atoms = (uint32_t)(n * (f32 ? 2 : 4) / 16 + 4);
    char tag[96];
    snprintf(tag, sizeof tag, "m%d_k%d_n%d_f%d_ae%d_r%d_rm%d", m, k, n, f32, ae, reuse, rm);
    clear_ops();
    int rc = fn(&p);
    record(tag, rc, (long)p.task_count);
}

/* Every branch combination at a few shapes, then the default branches at many. */
static const int MM_BRANCH_SHAPES[][3] = { {4, 64, 16}, {17, 128, 32}, {64, 1024, 1024} };
static const int MM_SHAPES[][3] = {
    {1, 64, 64}, {2, 32, 16}, {3, 96, 48}, {5, 64, 64}, {16, 64, 16}, {31, 256, 80},
    {33, 512, 128}, {63, 64, 64}, {64, 256, 64}, {128, 2048, 512}, {200, 1376, 256},
    {256, 1120, 512}, {256, 8192, 256}, {512, 4096, 1024}, {32, 4096, 256}, {1022, 64, 64},
    {1023, 64, 64}, {2047, 32, 16}, {96, 16384, 64}, {160, 1024, 4096}, {8, 10240, 64},
    {64, 64, 8192},
};

static void sweep_mm(unsigned which_mask, int branches, int shapes)
{
    for (unsigned g = 0; g < NMM; g++) {
        if (!(which_mask & (1u << g))) continue;
        if (branches)
            for (unsigned s = 0; s < sizeof MM_BRANCH_SHAPES / sizeof MM_BRANCH_SHAPES[0]; s++) {
                const int *sh = MM_BRANCH_SHAPES[s];
                group_open("%s/m%d_k%d_n%d/branches", MM[g].name, sh[0], sh[1], sh[2]);
                for (int f32 = 0; f32 <= 1; f32++)
                    for (int ae = 0; ae < 5; ae++)
                        for (int reuse = 0; reuse <= 3; reuse++)
                            for (int rm = 0; rm <= 3; rm++)
                                mm_one(MM[g].fn, sh[0], sh[1], sh[2], f32, ae, reuse, rm);
            }
        if (shapes) {
            group_open("%s/shapes", MM[g].name);
            for (unsigned s = 0; s < sizeof MM_SHAPES / sizeof MM_SHAPES[0]; s++)
                for (int f32 = 0; f32 <= 1; f32++)
                    for (int ae = 0; ae < 2; ae++) {
                        const int *sh = MM_SHAPES[s];
                        mm_one(MM[g].fn, sh[0], sh[1], sh[2], f32, ae, 0, 0);
                    }
        }
    }
    group_close();
}

/* The profile mini-sweep: two shapes and the branches that read the overrides. */
static void sweep_mm_mini(unsigned which_mask)
{
    static const int SH[][3] = { {4, 64, 16}, {17, 128, 32}, {2, 64, 32}, {64, 1024, 1024} };
    for (unsigned g = 0; g < NMM; g++) {
        if (!(which_mask & (1u << g))) continue;
        group_open("%s/mini", MM[g].name);
        for (unsigned s = 0; s < sizeof SH / sizeof SH[0]; s++)
            for (int f32 = 0; f32 <= 1; f32++)
                for (int ae = 0; ae < 5; ae++)
                    for (int rm = 0; rm <= 3; rm += 3)
                        mm_one(MM[g].fn, SH[s][0], SH[s][1], SH[s][2], f32, ae, rm ? 3 : 0, rm);
    }
    group_close();
}

typedef int (*conv_fn)(conv_params_t *);
static const struct { const char *name; conv_fn fn; int int8, dw; } CONV[] = {
    { "conv_fp16", gen_conv2d_fp16, 0, 0 }, { "conv_dw_fp16", gen_conv2d_dw_fp16, 0, 1 },
    { "conv_int8", gen_conv2d_int8, 1, 0 }, { "conv_dw_int8", gen_conv2d_dw_int8, 1, 1 },
};
#define NCONV (sizeof CONV / sizeof CONV[0])

static uint16_t g_lut[1026];
static lut_epilogue_t g_epi;

/* ext: 0 plain, 1 bias (int8 out, per-tensor), 2 bias + per-channel multiplier, 3 the
 * int8-out writer with zero points, 4 LUT epilogue, 5 deconvolution, 6 explicit dw group */
static void conv_one(conv_fn fn, int int8, int ic, int oc, int ih, int iw, int kh, int kw,
                     int sy, int sx, int dy, int dx, int pad, int f32, int ext)
{
    conv_params_t p;
    memset(&p, 0, sizeof p);
    p.ic = (uint16_t)ic; p.oc = (uint16_t)oc; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.kh = (uint16_t)kh; p.kw = (uint16_t)kw;
    p.stride_y = (uint8_t)sy; p.stride_x = (uint8_t)sx;
    p.dil_y = (uint8_t)dy; p.dil_x = (uint8_t)dx;
    p.pad_top = (uint8_t)(pad ? kh / 2 : 0); p.pad_left = (uint8_t)(pad ? kw / 2 : 0);
    int eh = (kh - 1) * dy + 1, ew = (kw - 1) * dx + 1;
    int oh = (ih + 2 * p.pad_top - eh) / sy + 1, ow = (iw + 2 * p.pad_left - ew) / sx + 1;
    if (oh < 1 || ow < 1) return;
    p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
    p.input_iova = 0x81000000u; p.weights_iova = 0x82000040u; p.output_iova = 0x83000000u;
    p.tasks = ops; p.fp16_out = (uint8_t)f32;
    p.in_scale = 0.02f; p.w_scale = 0.01f; p.out_scale = 0.05f;
    p.input_zero_point = -3; p.output_zero_point = 7; p.weight_zero_point = 0;
    switch (ext) {
    case 1: p.bias_iova = 0x85000000u; p.int8_out = 1; break;
    case 2: p.bias_iova = 0x85000000u; p.int8_out = 1; p.bs_mul_perc = 1; p.bs_mul_shift = 9;
            break;
    case 3: p.bias_iova = 0x85000000u; p.int8_out = 1; p.input_zero_point = 100;
            p.output_zero_point = -20; p.weight_zero_point = 3; p.in_scale = 0.0123f;
            p.w_scale = 0.0031f; p.out_scale = 0.071f; break;
    case 4: p.act = &g_epi; break;
    case 5:
        if (sy != 1 || sx != 1 || dy != 1 || dx != 1) return;
        p.deconv_sy = p.deconv_sx = 2;
        p.oh = (uint16_t)(ih * 2); p.ow = (uint16_t)(iw * 2);
        break;
    case 6: p.dw_group = int8 ? 32 : 16; break;
    case 7:
        if (sy != 1 || sx != 1 || dy != 1 || dx != 1) return;
        p.deconv_sy = 4; p.deconv_sx = 2;
        p.oh = (uint16_t)(ih * 4); p.ow = (uint16_t)(iw * 2);
        break;
    }
    char tag[128];
    snprintf(tag, sizeof tag, "c%d_%d_h%d_w%d_k%dx%d_s%dx%d_d%dx%d_p%d_f%d_x%d", ic, oc, ih, iw,
             kh, kw, sy, sx, dy, dx, pad, f32, ext);
    clear_ops();
    int rc = fn(&p);
    record(tag, rc, (long)p.task_count);
}

/* (y, x) pairs: equal and unequal, so a swapped axis cannot hide behind a symmetric shape */
static const int AX[][2] = { {1, 1}, {2, 2}, {1, 2}, {2, 1} };
static const int KS[][2] = { {1, 1}, {3, 3}, {5, 5}, {1, 7}, {7, 1}, {2, 2} };
static const int CONV_SHAPES[][4] = {   /* ic, oc, ih, iw */
    {64, 64, 32, 32}, {32, 16, 3, 40}, {128, 32, 7, 9}, {64, 64, 1, 64}, {32, 32, 2, 2},
    {256, 128, 14, 14}, {64, 96, 6, 10}, {1184, 32, 189, 1}, {32, 64, 64, 64},
    {64, 64, 300, 20}, {32, 2048, 4, 4},
};

static void sweep_conv(unsigned which_mask, int full)
{
    for (unsigned g = 0; g < NCONV; g++) {
        if (!(which_mask & (1u << g))) continue;
        int nks = full ? 6 : 2;
        for (int k = 0; k < nks; k++) {
            group_open("%s/k%dx%d", CONV[g].name, KS[k][0], KS[k][1]);
            for (unsigned s = 0; s < (full ? sizeof CONV_SHAPES / sizeof CONV_SHAPES[0] : 3); s++) {
                const int *sh = CONV_SHAPES[s];
                int ic = sh[0], oc = CONV[g].dw ? sh[0] : sh[1];
                for (int st = 0; st < 4; st++)
                    for (int dil = 0; dil < 3; dil++)
                        for (int pad = 0; pad <= 1; pad++)
                            for (int f32 = 0; f32 <= 1; f32++)
                                for (int ext = 0; ext <= 7; ext++) {
                                    if (!full && (ext == 6 || dil > 0 || st > 2)) continue;
                                    conv_one(CONV[g].fn, CONV[g].int8, ic, oc, sh[2], sh[3],
                                             KS[k][0], KS[k][1], AX[st][0], AX[st][1],
                                             AX[dil][0], AX[dil][1], pad, f32, ext);
                                }
            }
        }
    }
    group_close();
}

static void sweep_dpu_only(int full)
{
    group_open("lut");
    for (int n = 8; n <= 65536; n *= (full ? 4 : 64)) {
        lut_act_params_t a;
        memset(&a, 0, sizeof a);
        a.input_iova = 0x81000000u; a.output_iova = 0x83000000u; a.n = n; a.lut = g_lut;
        a.bn_mul_operand = 0x3c00; a.bn_alu_cfg = 0x80000000u; a.out_cvt_offset = 0xfffffff0u;
        a.out_cvt_scale = 3; a.out_cvt_minus_exp = 2; a.out_cvt_cvt_type = 1;
        a.lut_le_start = 0xffffc000u; a.lut_lo_end = 0x4000; a.le_slope_scale = 0x1234;
        a.le_slope_shift = 5; a.lo_slope_scale = 0x4321; a.lo_slope_shift = 7;
        a.le_index_select = 5; a.lo_index_select = 6; a.tasks = ops;
        char tag[32]; snprintf(tag, sizeof tag, "n%d", n);
        clear_ops();
        int rc = gen_lut_activation_fp16(&a);
        record(tag, rc, a.task_count);
    }
    {
        lut_act_params_t a;
        memset(&a, 0, sizeof a);
        a.n = 12; a.lut = g_lut; a.tasks = ops;
        clear_ops();
        int rc = gen_lut_activation_fp16(&a);
        record("lut_n12", rc, a.task_count);
    }
    group_close();
}

static void sweep_pool(void)
{
    group_open("pool");
    static const int SH[][3] = { {64, 32, 32}, {8, 7, 5}, {256, 112, 112}, {16, 1, 9} };
    for (unsigned s = 0; s < sizeof SH / sizeof SH[0]; s++)
    for (int method = 0; method <= 1; method++)
    for (int k = 1; k <= 4; k++)
    for (int st = 1; st <= 3; st++)
    for (int pad = 0; pad <= 2; pad++) {
        pool_params_t p;
        memset(&p, 0, sizeof p);
        p.c = (uint16_t)SH[s][0]; p.ih = (uint16_t)SH[s][1]; p.iw = (uint16_t)SH[s][2];
        p.kh = (uint8_t)k; p.kw = (uint8_t)(k == 4 ? 2 : k);
        p.stride_y = (uint8_t)st; p.stride_x = (uint8_t)(st == 3 ? 1 : st);
        p.pad_top = p.pad_left = (uint8_t)pad; p.pad_bottom = p.pad_right = (uint8_t)(pad / 2);
        int oh = (p.ih + p.pad_top + p.pad_bottom - p.kh) / p.stride_y + 1;
        int ow = (p.iw + p.pad_left + p.pad_right - p.kw) / p.stride_x + 1;
        if (oh < 1 || ow < 1) continue;
        p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
        p.method = (uint8_t)method; p.recip_w = 0x10000 / p.kw; p.recip_h = 0x10000 / p.kh;
        p.input_iova = 0x81000000u; p.output_iova = 0x83000000u; p.tasks = ops;
        char tag[64];
        snprintf(tag, sizeof tag, "s%u_m%d_k%d_s%d_p%d", s, method, k, st, pad);
        clear_ops();
        int rc = gen_pool_fp16(&p);
        record(tag, rc, p.task_count);
    }
    {
        pool_params_t p;
        memset(&p, 0, sizeof p);
        p.c = 8; p.ih = p.iw = 4; p.oh = p.ow = 2; p.kh = p.kw = 17; p.stride_x = p.stride_y = 1;
        p.tasks = ops;
        clear_ops();
        int rc = gen_pool_fp16(&p);
        record("kernel17", rc, p.task_count);
    }
    group_close();
}

/* ---- the host layout helpers -------------------------------------------------------- */

/* Every index each helper returns over a range of shapes, written into the word buffer in
 * order and hashed like a program. Contraction lengths that are not a multiple of the group
 * are included, so a helper's arithmetic is pinned outside the shapes the planners pass. */
static void sweep_layout(void)
{
    static const int KS_[] = { 32, 40, 64, 96, 160 };
    static const int NS_[] = { 16, 32, 48, 64, 80, 128 };
    typedef int (*wfn)(int, int, int);
    static const struct { const char *name; wfn fn; } W[] = {
        { "gemm_weight_index_fp16", gemm_weight_index_fp16 }, { "gemm_weight_index_int8", gemm_weight_index_int8 },
        { "gemm_weight_index_int4", gemm_weight_index_int4 }, { "gemm_weight_index_int16", gemm_weight_index_int16 },
        { "gemm_weight_index_tf32", gemm_weight_index_tf32 },
    };
    for (unsigned f = 0; f < sizeof W / sizeof W[0]; f++) {
        group_open("layout/%s", W[f].name);
        for (unsigned a = 0; a < sizeof KS_ / sizeof KS_[0]; a++)
            for (unsigned b = 0; b < sizeof NS_ / sizeof NS_[0]; b++) {
                long n = 0;
                for (int k = 1; k <= NS_[b]; k++)
                    for (int c = 1; c <= KS_[a]; c++)
                        ops[n++] = (uint64_t)(int64_t)W[f].fn(KS_[a], k, c);
                char tag[32];
                snprintf(tag, sizeof tag, "K%d_N%d", KS_[a], NS_[b]);
                record(tag, 0, n);
            }
    }
    group_open("layout/feature_index");
    {
        static const int SH[][4] = { {16, 3, 5, 8}, {24, 4, 4, 16}, {7, 2, 9, 8}, {64, 1, 1, 16} };
        for (unsigned s = 0; s < sizeof SH / sizeof SH[0]; s++) {
            long n = 0;
            int C = SH[s][0], H = SH[s][1], W_ = SH[s][2], C2 = SH[s][3];
            for (int c = 1; c <= C; c++)
                for (int h = 1; h <= H; h++)
                    for (int w = 1; w <= W_; w++)
                        ops[n++] = (uint64_t)(int64_t)feature_index(C, H, W_, C2, c, h, w);
            char tag[48];
            snprintf(tag, sizeof tag, "C%d_H%d_W%d_C2%d", C, H, W_, C2);
            record(tag, 0, n);
        }
    }
    group_open("layout/weight_conv");
    {
        static const int SH[][4] = { {16, 32, 3, 3}, {32, 64, 1, 7}, {48, 40, 2, 2} };
        for (unsigned s = 0; s < sizeof SH / sizeof SH[0]; s++) {
            int OC = SH[s][0], IC = SH[s][1], KH = SH[s][2], KW = SH[s][3];
            long n = 0, m = 0;
            for (int oc = 1; oc <= OC; oc++)
                for (int ic = 1; ic <= IC; ic++)
                    for (int kh = 1; kh <= KH; kh++)
                        for (int kw = 1; kw <= KW; kw++) {
                            ops[n++] = (uint64_t)(int64_t)conv_weight_index_fp16(OC, IC, KH, KW, oc, ic, kh, kw);
                            ops[n++] = (uint64_t)(int64_t)conv_weight_index_int8(OC, IC, KH, KW, oc, ic, kh, kw);
                        }
            for (int c = 1; c <= IC; c++)
                for (int kh = 1; kh <= KH; kh++)
                    for (int kw = 1; kw <= KW; kw++) {
                        ops[n + m++] = (uint64_t)(int64_t)conv_dw_weight_index_fp16(IC, KH, KW, 32, c, kh, kw);
                        ops[n + m++] = (uint64_t)(int64_t)conv_dw_weight_index_int8(IC, KH, KW, 64, c, kh, kw);
                    }
            char tag[48];
            snprintf(tag, sizeof tag, "OC%d_IC%d_K%dx%d", OC, IC, KH, KW);
            record(tag, 0, n + m);
        }
    }
    group_close();
}

/* ---- the RK3576 sweep ------------------------------------------------------------- */

static void r76_conv_one(const char *gname, int (*fn)(conv_params_t *), int ic, int oc, int ih,
                         int iw, int kh, int kw, int sy, int sx, int pad, int bias, int lut,
                         int full)
{
    static const lut_rk3576_t L = { -16384, 16384, 5, -7, 9 };
    conv_params_t p;
    memset(&p, 0, sizeof p);
    p.ic = (uint16_t)ic; p.oc = (uint16_t)oc; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.kh = (uint16_t)kh; p.kw = (uint16_t)kw;
    p.stride_y = (uint8_t)sy; p.stride_x = (uint8_t)sx; p.dil_y = p.dil_x = 1;
    p.pad_top = (uint8_t)(pad ? kh / 2 : 0); p.pad_left = (uint8_t)(pad ? kw / 2 : 0);
    int oh = (ih + 2 * p.pad_top - kh) / sy + 1, ow = (iw + 2 * p.pad_left - kw) / sx + 1;
    if (oh < 1 || ow < 1) return;
    p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
    p.input_iova = 0x81000000u; p.weights_iova = 0x82000000u; p.output_iova = 0x83000000u;
    p.tasks = ops; p.in_scale = 0.02f; p.w_scale = 0.01f; p.out_scale = 0.05f;
    p.input_zero_point = -3; p.output_zero_point = 7; p.int8_out = 1;
    if (bias) { p.bias_iova = 0x85000000u; p.bs_mul_perc = bias == 2; p.bs_mul_shift = 7; }
    if (lut) p.lut = &L;
    if (full) { p.ih_full = p.ih; p.oh_full = p.oh; }
    char tag[96];
    snprintf(tag, sizeof tag, "%s_c%d_%d_h%d_w%d_k%dx%d_s%dx%d_p%d_b%d_l%d", gname, ic, oc, ih,
             iw, kh, kw, sy, sx, pad, bias, lut);
    memset(ops, 0, sizeof(uint64_t) * 8192);
    int rc = fn(&p);
    record(tag, rc, (long)p.task_count);
}

static int r76_dw(conv_params_t *p)   { return gen_conv2d_rk3576_prec(p, 1, PREC_INT8); }
static int r76_bf16(conv_params_t *p) { return gen_conv2d_rk3576_prec(p, 0, PREC_BF16); }

static void sweep_rk3576(void)
{
    static const struct { const char *name; int (*fn)(conv_params_t *); } G[] = {
        { "int8", gen_conv2d_int8_rk3576 }, { "dw_int8", gen_conv2d_dw_int8_rk3576 },
        { "i32out", gen_conv2d_int8_rk3576_i32out },
        { "i32wide", gen_conv2d_int8_rk3576_i32out_wide },
        { "i32dense", gen_conv2d_int8_rk3576_i32out_dense },
        { "fp16", gen_conv2d_fp16_rk3576 }, { "fp16_f32", gen_conv2d_fp16_rk3576_f32out },
        { "prec_dw", r76_dw }, { "prec_bf16", r76_bf16 },
    };
    static const int SH[][4] = { {64, 64, 16, 16}, {3, 32, 32, 32}, {16, 48, 7, 9},
                                 {32, 32, 56, 56}, {4, 16, 20, 24}, {128, 256, 7, 7} };
    for (unsigned g = 0; g < sizeof G / sizeof G[0]; g++) {
        group_open("r76/%s", G[g].name);
        static const int RK[][2] = { {1, 1}, {3, 3}, {1, 3}, {3, 1} };
        for (unsigned s = 0; s < sizeof SH / sizeof SH[0]; s++)
            for (int k = 0; k < 4; k++)
                for (int st = 0; st < 4; st++)
                    for (int pad = 0; pad <= 1; pad++)
                        for (int bias = 0; bias <= 2; bias++)
                            for (int lut = 0; lut <= 1; lut++) {
                                int oc = (g == 1 || g == 7) ? SH[s][0] : SH[s][1];
                                r76_conv_one(G[g].name, G[g].fn, SH[s][0], oc, SH[s][2],
                                             SH[s][3], RK[k][0], RK[k][1], AX[st][0], AX[st][1],
                                             pad, bias, lut, g == 4);
                            }
    }

    group_open("r76/rows");
    {
        static const int RS[][4] = { {32, 32, 112, 112}, {64, 64, 56, 56}, {3, 32, 224, 224} };
        for (unsigned s = 0; s < sizeof RS / sizeof RS[0]; s++)
            for (int dw = 0; dw <= 1; dw++) {
                conv_params_t p;
                rocket_rk3576_row_task t[64];
                unsigned cnt = 0;
                memset(&p, 0, sizeof p);
                p.ic = (uint16_t)RS[s][0]; p.oc = (uint16_t)(dw ? RS[s][0] : RS[s][1]);
                p.ih = (uint16_t)RS[s][2]; p.iw = (uint16_t)RS[s][3];
                p.kh = p.kw = 3; p.stride_x = p.stride_y = 1; p.dil_x = p.dil_y = 1;
                p.pad_top = p.pad_left = 1; p.oh = p.ih; p.ow = p.iw;
                p.in_scale = 0.02f; p.w_scale = 0.01f; p.out_scale = 0.05f; p.int8_out = 1;
                p.bias_iova = 0x85000000u;
                int rc = rocket_rk3576_plan_rows(&p, dw, t, 64, &cnt);
                char tag[64];
                snprintf(tag, sizeof tag, "plan_s%u_dw%d", s, dw);
                clear_ops();
                record(tag, rc, 0);
                if (rc != 0) continue;
                for (unsigned j = 0; j < cnt; j++) {
                    conv_params_t q = p;
                    q.ih_full = p.ih; q.oh_full = p.oh;
                    q.ih = t[j].ih; q.oh = t[j].oh; q.pad_top = t[j].pad_top;
                    q.input_iova = 0x81000000u + t[j].feature_off;
                    q.output_iova = 0x83000000u + t[j].output_off;
                    q.weights_iova = 0x82000000u; q.tasks = ops;
                    snprintf(tag, sizeof tag, "s%u_dw%d_t%u", s, dw, j);
                    clear_ops();
                    rc = dw ? gen_conv2d_dw_int8_rk3576(&q) : gen_conv2d_int8_rk3576(&q);
                    record(tag, rc, (long)q.task_count);
                    clear_ops();
                    q.tasks = ops;
                    rc = gen_conv2d_int8_rk3576_reuse(&q, dw, &t[j], j);
                    record(tag, rc, (long)q.task_count);
                }
            }
    }

    group_open("r76/pool");
    for (int mode = 0; mode <= 2; mode++)
        for (int k = 2; k <= 3; k++)
            for (int st = 1; st <= 2; st++)
                for (int pad = 0; pad <= 1; pad++) {
                    pool_params_rk3576_t q;
                    memset(&q, 0, sizeof q);
                    q.iw = 28; q.ih = 20; q.c = 32; q.kw = q.kh = (uint8_t)k;
                    q.stride_x = q.stride_y = (uint8_t)st;
                    q.pad_left = q.pad_top = (uint8_t)pad; q.pad_right = q.pad_bottom = 0;
                    q.ow = (uint16_t)((q.iw + pad - k) / st + 1);
                    q.oh = (uint16_t)((q.ih + pad - k) / st + 1);
                    q.mode = (uint8_t)mode; q.input_zero_point = -5;
                    q.input_iova = 0x81000000u; q.output_iova = 0x83000000u; q.tasks = ops;
                    if (pad) { q.src_surf_elems = 600; q.dst_surf_elems = 400; }
                    char tag[64];
                    snprintf(tag, sizeof tag, "m%d_k%d_s%d_p%d", mode, k, st, pad);
                    clear_ops();
                    int rc = gen_pool_rk3576(&q);
                    record(tag, rc, q.task_count);
                }

    group_open("r76/lut_ew_mm");
    {
        static int16_t lo[RK3576_LUT_ENTRIES], hi[RK3576_LUT_ENTRIES];
        for (int i = 0; i < RK3576_LUT_ENTRIES; i++) { lo[i] = (int16_t)(i * 7 - 300); hi[i] = (int16_t)(i * 3); }
        lut_load_params_rk3576_t l;
        memset(&l, 0, sizeof l);
        l.lo = lo; l.hi = hi; l.scratch_iova = 0x86000000u; l.tasks = ops;
        clear_ops();
        int rc = gen_lut_load_rk3576(&l);
        record("lut_load", rc, l.task_count);
        for (int mode = 0; mode <= 1; mode++) {
            ew_params_rk3576_t e;
            memset(&e, 0, sizeof e);
            e.w = 14; e.h = 9; e.c = 32; e.mode = (uint8_t)mode;
            e.src_iova = 0x81000000u; e.ew_iova = 0x82000000u; e.dst_iova = 0x83000000u;
            e.ew_offset = -12; e.ew_scale = 0x4321; e.ew_shift = 7; e.out_offset = 5;
            e.out_scale = 0x2345; e.out_shift = 9; e.clamp_lo = -100; e.clamp_hi = 100;
            e.tasks = ops;
            char tag[16]; snprintf(tag, sizeof tag, "ew%d", mode);
            clear_ops();
            rc = gen_ew_int8_rk3576(&e);
            record(tag, rc, e.task_count);
        }
        static const int MS[][3] = { {64, 64, 64}, {512, 1536, 256}, {7, 96, 48}, {2048, 6144, 8192} };
        for (unsigned s = 0; s < sizeof MS / sizeof MS[0]; s++) {
            char tag[48];
            snprintf(tag, sizeof tag, "mm_m%d_k%d_n%d", MS[s][0], MS[s][1], MS[s][2]);
            clear_ops();
            int rc = gen_gemm_fp16_rk3576(ops, (unsigned)MS[s][0], (unsigned)MS[s][1],
                                            (unsigned)MS[s][2], 0x81000000u, 0x82000000u,
                                            0x83000000u);
            record(tag, rc, rc > 0 ? rc : 0);
        }
        /* the RK3588 encoders refuse on this part */
        gemm_params_t p;
        memset(&p, 0, sizeof p);
        p.m = 4; p.k = 64; p.n = 16; p.tasks = ops;
        clear_ops();
        rc = gen_gemm_fp16(&p);
        record("refuse_mm_fp16", rc, (long)p.task_count);
    }
    group_close();
}

/* ---- profiles ------------------------------------------------------------------------ */

enum { F_FP16 = 1, F_INT8 = 2, F_INT4 = 4, F_INT16 = 8, F_BF16 = 16, F_TF32 = 32,
       F_MM = 63, F_CONV = 64, F_DPU = 128, F_R76 = 256 };

typedef struct { const char *env[4]; unsigned fam; } profile_t;

static const profile_t PROFILES[] = {
    /* no override: the mini-sweeps' reference, so a profile equal to it changed nothing */
    { { NULL }, F_MM | F_CONV | F_DPU },
    { { NULL }, F_R76 },
    /* the shared matmul emitter */
    { { "ROCKET_CNA_CLK_GATE=0x13" }, F_MM },
    { { "ROCKET_CORE_SOFT_GATING=0x2a" }, F_MM },
    { { "ROCKET_CORE_MAC_GATING=0x7800801" }, F_MM },
    { { "ROCKET_CORE_3030=0x5" }, F_MM | F_CONV },
    { { "ROCKET_FC_DATA_BANK=3" }, F_MM | F_CONV },
    { { "ROCKET_OUT_CVT_ROUND=1" }, F_MM | F_CONV | F_DPU },
    /* fp16 matmul */
    { { "ROCKET_MM_MWIDTH=1" }, F_MM },
    { { "ROCKET_MM_SURF_TRUNC=1" }, F_MM },
    { { "ROCKET_MM_GRAINS=77" }, F_MM },
    { { "ROCKET_MM_LINE_STRIDE=0x200" }, F_MM },
    { { "ROCKET_MM_SURF_PER_M=-3" }, F_MM },
    { { "ROCKET_EW_SURF=40" }, F_MM },
    { { "ROCKET_EW_BASE_OFF=0x300" }, F_MM },
    { { "ROCKET_EW_NOTCH=50" }, F_MM },
    { { "ROCKET_EW_CFG=0x108102C0" }, F_MM },
    { { "ROCKET_ERDMA_CFG=0x4000000C" }, F_MM },
    { { "ROCKET_RDMA_FMC=0x17D41" }, F_MM },
    /* int8 matmul */
    { { "ROCKET_INT8_FP32_OUT=1" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1", "ROCKET_INT8_DEQ_PREC=fp16" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1", "ROCKET_INT8_DEQ_PREC=fp32" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1", "ROCKET_INT8_DEQ_PREC=6" }, F_INT8 },
    { { "ROCKET_INT8_DEQ_PREC=fp16" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1", "ROCKET_INT8_DEQ_SCALE=3" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1", "ROCKET_INT8_DEQ_SHIFT=2" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1", "ROCKET_INT8_DEQ_MINEXP=4" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1", "ROCKET_INT8_DEQ_CVTTYPE=0" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1", "ROCKET_INT8_DEQ_OFFSET=-7" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1", "ROCKET_INT8_DEQ_FP32TOFP16=1" }, F_INT8 },
    { { "ROCKET_INT8_DEQ=1", "ROCKET_INT8_DEQ_BNMUL=0x3c00" }, F_INT8 },
    { { "ROCKET_INT8_SIZE_E=3" }, F_INT8 },
    { { "ROCKET_INT8_SURF_MULT=4" }, F_INT8 },
    { { "ROCKET_INT8_EW_SURF=20" }, F_INT8 },
    { { "ROCKET_INT8_EW_BASE_OFF=0x100" }, F_INT8 },
    { { "ROCKET_INT8_EW_NOTCH=33" }, F_INT8 },
    { { "ROCKET_INT8_EW_CFG=0x10C202C1" }, F_INT8 },
    { { "ROCKET_INT8_ERDMA_CFG=0x40000004" }, F_INT8 },
    { { "ROCKET_INT8_RDMA_FMC=0x27D81" }, F_INT8 },
    { { "ROCKET_INT8_FP32_OUT=1", "ROCKET_INT8_DEQ=1" }, F_INT8 },
    { { "ROCKET_I8_FDBANK_EXTRA=1" }, F_INT8 },
    { { "ROCKET_I8_FDBANK_EXTRA=-1" }, F_INT8 },
    { { "ROCKET_I8_FDBANK_EXTRA=-20" }, F_INT8 },
    { { "ROCKET_I8_GRAINS=99" }, F_INT8 },
    { { "ROCKET_I8_SURF=12" }, F_INT8 },
    { { "ROCKET_I8_ENTRIES=5" }, F_INT8 },
    { { "ROCKET_I8_LINESTRIDE=8" }, F_INT8 },
    /* int4 */
    { { "ROCKET_INT4_PREC=3" }, F_INT4 },
    { { "ROCKET_INT4_OUT_PREC=4" }, F_INT4 },
    { { "ROCKET_INT4_SIZE_E=3" }, F_INT4 },
    { { "ROCKET_INT4_SURF_MULT=4" }, F_INT4 },
    { { "ROCKET_INT4_DENTRIES_DIV=64" }, F_INT4 },
    { { "ROCKET_INT4_DENTRIES_DIV=0" }, F_INT4 },
    /* int16 */
    { { "ROCKET_INT16_PREC=2" }, F_INT16 },
    { { "ROCKET_INT16_OUT_PREC=5" }, F_INT16 },
    { { "ROCKET_INT16_SIZE_E=7" }, F_INT16 },
    { { "ROCKET_INT16_SURF_MULT=8" }, F_INT16 },
    { { "ROCKET_INT16_DENTRIES_DIV=64" }, F_INT16 },
    { { "ROCKET_INT16_DENTRIES_DIV=0" }, F_INT16 },
    { { "ROCKET_INT16_QD_EN=0" }, F_INT16 },
    { { "ROCKET_INT16_GRAINS=33" }, F_INT16 },
    { { "ROCKET_INT16_GRAINS=0" }, F_INT16 },
    { { "ROCKET_INT16_KGROUPS=2" }, F_INT16 },
    { { "ROCKET_INT16_MC_SURF=1" }, F_INT16 },
    { { "ROCKET_INT16_TP_PREC=1" }, F_INT16 },
    { { "ROCKET_INT16_TP_ORG=1" }, F_INT16 },
    { { "ROCKET_INT16_SIZE_C=15" }, F_INT16 },
    /* bf16 */
    { { "ROCKET_BF16_PREC=7" }, F_BF16 },
    { { "ROCKET_BF16_OUT_PREC=2" }, F_BF16 },
    { { "ROCKET_BF16_SIZE_E=1" }, F_BF16 },
    { { "ROCKET_BF16_SURF_MULT=2" }, F_BF16 },
    { { "ROCKET_BF16_DENTRIES_DIV=16" }, F_BF16 },
    /* tf32 */
    { { "ROCKET_TF32_PREC=2" }, F_TF32 },
    { { "ROCKET_TF32_CORE_PREC=5" }, F_TF32 },
    { { "ROCKET_TF32_DPU_PREC=2" }, F_TF32 },
    { { "ROCKET_TF32_OUT_PREC=2" }, F_TF32 },
    { { "ROCKET_TF32_SIZE_E=1" }, F_TF32 },
    { { "ROCKET_TF32_SURF_MULT=2" }, F_TF32 },
    { { "ROCKET_TF32_DENTRIES_DIV=32" }, F_TF32 },
    { { "ROCKET_TF32_ELEM_BYTES=2" }, F_TF32 },
    { { "ROCKET_TF32_ELEM_BYTES=0", "ROCKET_TF32_DENTRIES_DIV=0" }, F_TF32 },
    { { "ROCKET_TF32_FGRAINS=9" }, F_TF32 },
    { { "ROCKET_TF32_OUTPUT_MODE=1" }, F_TF32 },
    { { "ROCKET_TF32_MC_SURF=1" }, F_TF32 },
    /* convolution */
    { { "ROCKET_CONV_GRAINS=40" }, F_CONV },
    { { "ROCKET_CONV_STRIDE_SWAP=1" }, F_CONV },
    { { "ROCKET_CONV_DIL_RAW=1" }, F_CONV },
    { { "ROCKET_CONV_DENTRIES=9" }, F_CONV },
    { { "ROCKET_CONV_DW_GROUP=16" }, F_CONV },
    { { "ROCKET_CONV_DW_GROUP=0" }, F_CONV },
    { { "ROCKET_CONV_DW_SIZEE=1" }, F_CONV },
    { { "ROCKET_CONV_DW_BSOWOP=0x7f" }, F_CONV },
    { { "ROCKET_CONV_DW_OCPAD=1" }, F_CONV },
    { { "ROCKET_CONV_DW_SURF_MULT=8" }, F_CONV },
    { { "ROCKET_CONV_INT8_SIZE_E=3" }, F_CONV },
    { { "ROCKET_CONV_INT8_SURF_MULT=4" }, F_CONV },
    { { "ROCKET_CONV_I8_FDBANK_EXTRA=-1" }, F_CONV },
    { { "ROCKET_CONV_I8_FDBANK_EXTRA=30" }, F_CONV },
    { { "ROCKET_CONV_SURF_TRUNC=1" }, F_CONV },
    { { "ROCKET_CNA_DECONV=1" }, F_CONV },
    { { "ROCKET_CNA_DECONV=0" }, F_CONV },
    { { "ROCKET_CNA_DECONV_X=1" }, F_CONV },
    { { "ROCKET_CNA_DECONV_Y=3" }, F_CONV },
    /* the DPU-only programs */
    { { "ROCKET_LUT_SPTR=0xF" }, F_DPU },
    { { "ROCKET_LUT_ENABLE=0x1C" }, F_DPU },
    /* the RK3576 encoder's own overrides */
    { { "ROCKET_RK3576_KSWAP=1" }, F_R76 },
    { { "ROCKET_RK3576_ADDR_SHOTGUN=1" }, F_R76 },
    { { "ROCKET_RK3576_ADD=0x4050=0x55" }, F_R76 },
    { { "ROCKET_RK3576_SET=0x1010=0x1234" }, F_R76 },
    { { "ROCKET_OUT_CVT_ROUND=1" }, F_R76 },
};
#define NPROF (sizeof PROFILES / sizeof PROFILES[0])

static void clear_rocket_env(void)
{
    char *names[512];
    int n = 0;
    for (char **e = environ; *e && n < 512; e++)
        if (!strncmp(*e, "ROCKET_", 7)) {
            const char *eq = strchr(*e, '=');
            size_t len = eq ? (size_t)(eq - *e) : strlen(*e);
            names[n] = strndup(*e, len);
            if (names[n]) n++;
        }
    for (int i = 0; i < n; i++) { unsetenv(names[i]); free(names[i]); }
}

static void quiet_log(rocket_log_level l, const char *t, void *u) { (void)l; (void)t; (void)u; }

/* Runs in the child: the default sweep (prof < 0) or one profile's. */
static void run(int prof)
{
    int r76 = prof == -2 || (prof >= 0 && (PROFILES[prof].fam & F_R76));
    clear_rocket_env();
    setenv("ROCKET_CHIP", r76 ? "rk3576" : "rk3588", 1);
    rocket_log_set_callback(quiet_log, NULL);
    for (int i = 0; i < 1026; i++) g_lut[i] = (uint16_t)(i * 37);
    g_epi.lut = g_lut; g_epi.bn_mul_operand = 0x3c00; g_epi.bn_alu_cfg = 0x80000000u;
    g_epi.out_cvt_offset = 0xfffffff0u; g_epi.out_cvt_scale = 1; g_epi.out_cvt_minus_exp = 3;
    g_epi.out_cvt_cvt_type = 1; g_epi.lut_le_start = 0xffffc000u; g_epi.lut_lo_end = 0x4000;
    g_epi.le_slope_scale = 0x1234; g_epi.le_slope_shift = 5; g_epi.lo_slope_scale = 0x4321;
    g_epi.lo_slope_shift = 7; g_epi.le_index_select = 5; g_epi.lo_index_select = 6;

    if (prof == -1) {
        g_profile = "default";
        sweep_mm(F_MM, 1, 1);
        sweep_conv(0xF, 1);
        sweep_dpu_only(1);
        sweep_pool();
        sweep_layout();
        return;
    }
    if (prof == -2) {
        g_profile = "default";
        sweep_rk3576();
        return;
    }
    static char name[256];
    name[0] = 0;
    for (int i = 0; i < 4 && PROFILES[prof].env[i]; i++) {
        char kv[128];
        snprintf(kv, sizeof kv, "%s", PROFILES[prof].env[i]);
        char *eq = strchr(kv, '=');
        if (!eq) continue;
        *eq = 0;
        setenv(kv, eq + 1, 1);
        if (i) strncat(name, ",", sizeof name - strlen(name) - 1);
        strncat(name, PROFILES[prof].env[i], sizeof name - strlen(name) - 1);
    }
    g_profile = name[0] ? name : "baseline";
    unsigned f = PROFILES[prof].fam;
    if (f & F_MM) sweep_mm_mini(f & F_MM);
    if (f & F_CONV) sweep_conv(0xF, 0);
    if (f & F_DPU) sweep_dpu_only(0);
    if (f & F_R76) sweep_rk3576();
}

/* Fork, run one sweep in the child with its stdout on a pipe, append what it writes. */
static int collect(int prof, char **buf, size_t *len, size_t *cap)
{
    int fd[2];
    if (pipe(fd) != 0) { perror("pipe"); return -1; }
    fflush(NULL);
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return -1; }
    if (pid == 0) {
        close(fd[0]);
        g_out = fdopen(fd[1], "w");
        if (!g_out) _exit(4);
        run(prof);
        fclose(g_out);
        _exit(0);
    }
    close(fd[1]);
    for (;;) {
        if (*cap - *len < 65536) {
            *cap = *cap * 2 + 65536;
            char *nb = realloc(*buf, *cap);
            if (!nb) { close(fd[0]); return -1; }
            *buf = nb;
        }
        ssize_t r = read(fd[0], *buf + *len, *cap - *len - 1);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        *len += (size_t)r;
    }
    close(fd[0]);
    int st;
    if (waitpid(pid, &st, 0) < 0 || !WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        fprintf(stderr, "regcmd_golden_gate: sweep %d did not exit cleanly (status 0x%x)\n",
                prof, st);
        return -1;
    }
    (*buf)[*len] = 0;
    return 0;
}

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    size_t cap = 1 << 16, len = 0;
    char *b = malloc(cap);
    size_t r;
    while (b && (r = fread(b + len, 1, cap - len - 1, f)) > 0) {
        len += r;
        if (cap - len < 4096) { cap *= 2; b = realloc(b, cap); }
    }
    fclose(f);
    if (b) b[len] = 0;
    return b;
}

/* Compares line by line, skipping '#' lines in the golden file. */
static int compare(const char *want, const char *got)
{
    int bad = 0, lines = 0;
    const char *w = want, *g = got;
    for (;;) {
        while (*w == '#') { w = strchr(w, '\n'); w = w ? w + 1 : ""; }
        while (*g == '#') { g = strchr(g, '\n'); g = g ? g + 1 : ""; }
        if (!*w && !*g) break;
        const char *we = strchr(w, '\n'), *ge = strchr(g, '\n');
        size_t wl = we ? (size_t)(we - w) : strlen(w), gl = ge ? (size_t)(ge - g) : strlen(g);
        lines++;
        if (wl != gl || memcmp(w, g, wl)) {
            if (++bad <= 20) {
                printf("DIFF want: %.*s\n", (int)wl, *w ? w : "(end of golden file)");
                printf("     got:  %.*s\n", (int)gl, *g ? g : "(end of this sweep)");
            }
        }
        w = we ? we + 1 : w + wl;
        g = ge ? ge + 1 : g + gl;
    }
    printf("regcmd_golden_gate: %d groups, %d differ\n", lines, bad);
    return bad;
}

int main(int argc, char **argv)
{
    const char *golden = NULL, *write_to = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--write") && i + 1 < argc) write_to = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) g_dump = argv[++i];
        else golden = argv[i];
    }
    if (!golden && !write_to && !g_dump) golden = "tests/data/regcmd-golden/golden.txt";

    ops = calloc(NOPS, sizeof *ops);
    if (!ops) return 3;

    size_t len = 0, cap = 1 << 20;
    char *buf = malloc(cap);
    if (!buf) return 3;
    const char *hdr =
        "# regcmd_golden_gate: one line per group of configurations, in sweep order.\n"
        "# Regenerate with `regcmd_golden_gate --write FILE` only for an intended change.\n";
    if (!g_dump) { len = strlen(hdr); memcpy(buf, hdr, len); }

    if (collect(-1, &buf, &len, &cap) || collect(-2, &buf, &len, &cap)) return 3;
    for (int p = 0; p < (int)NPROF; p++)
        if (collect(p, &buf, &len, &cap)) return 3;

    if (g_dump) { fwrite(buf, 1, len, stdout); return 0; }
    if (write_to) {
        FILE *f = fopen(write_to, "w");
        if (!f || fwrite(buf, 1, len, f) != len || fclose(f)) {
            perror(write_to);
            return 3;
        }
        printf("regcmd_golden_gate: wrote %s\n", write_to);
        return 0;
    }
    char *want = slurp(golden);
    if (!want) { perror(golden); return 3; }
    return compare(want, buf) ? 1 : 0;
}
