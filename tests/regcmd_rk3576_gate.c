// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * regcmd_rk3576_gate.c — host-only gate: does the RK3576 conv emitter reproduce the
 * vendor's register program?
 *
 * No NPU, no RK3576 board. The oracle is a set of RKNN-Toolkit2 register programs
 * captured for known conv geometries on the RK3576 (tests/data/rk3576-vendor-capture/).
 * For each geometry the gate emits our program and diffs it against the capture,
 * entry for entry:
 *
 *   - order and length must match, register for register;
 *   - every geometry / channel / tiling / constant register must match exactly;
 *   - addresses and the data-dependent requant triple are excluded — they are
 *     per-run values the capture zeroes or takes from its own model's quant;
 *   - registers with a known-open formula are reported and do not fail the gate,
 *     so a mismatch there stays visible instead of being silently excluded.
 *
 * This is the acceptance test for everything downstream: an emitter that passes it
 * is programming the RK3576 the way the vendor does, before a single job is
 * submitted to real silicon.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "npu_hw.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rk3576_vendor_golden.h"
#include "rocket_log.h"

/* Per-run addresses: zero in the capture (the vendor kernel patches them), driven
 * by the caller in ours. */
static const uint16_t EXCLUDE_ADDR[] = {
    0x1088,  /* CNA feature data base   */
    0x1110,  /* CNA weight (DCOMP) base */
    0x4018,  /* DPU output base         */
    0x5020,  /* DPU_RDMA bias base      */
    0x5024,  /* DPU_RDMA shift-word base */
};

/* The requant triple is computed from the model's quant scales, which differ
 * between the capture's model and the gate's inputs. The gate checks that the
 * emitter puts requant HERE (offset/scale/shift, three consecutive registers), not
 * that it reproduces another model's numbers. */
static const uint16_t EXCLUDE_QUANT[] = { 0x40AC, 0x40B0, 0x40B4 };

/* DPU epilogue registers the captures show tracking the MODEL rather than the
 * geometry: they take one value per capture file and are constant across every
 * geometry and both modes inside it. The eltwise clamp pair carries the layer's
 * activation range (0/103564 in the chained model, full int32 in the single-op
 * ones); BN_CFG and the BS ALU config likewise follow which epilogue stages that
 * model uses. Our emitter clamps nothing and runs no BN, which is a valid program,
 * so a difference here is a different model, not a different encoder. */
static const uint16_t EXCLUDE_MODEL[] = {
    0x4044,  /* BS ALU config  — 1 with a bias stage, 0 in the isolated-op models */
    0x4060,  /* BN config      — 0x903 single-op, 0x902 chained                   */
    0x406C, 0x4070, 0x4074, 0x4078,  /* EW clamp min/max pairs = activation range */
};

/* A continuation window — a task that is not the first of its plane — reads fewer
 * feature rows than its window is tall, because the vendor keeps the kernel's
 * overlap rows resident in the CBUF from the previous task and rotates the CBUF
 * base instead of refetching them. Our emitter refetches the whole window against a
 * base of zero, which is self-consistent and bit-exact on hardware, so the three
 * fetch-count registers legitimately differ on those tasks only. */
static const uint16_t REFETCH_FIELDS[] = { 0x1028, 0x1078, 0x1098 };

/* Fields the vendor's own CBUF allocator chooses per task rather than deriving from
 * the geometry: the granule allowance F and the rotating CBUF base (0x1040), the
 * base echoed into the entries word (0x103C), and the format/window low bits
 * (0x1018, 0x1038) that a hardware sweep showed are don't-care on the int8 path —
 * only 0x1018 bit 30 and 0x1038 bit 31 are live. Our planner picks its own F for a
 * plane the vendor never ran, so a difference here is a different allocation, not a
 * different program. Reported so it stays visible, not fatal. */
static const uint16_t ALLOC_FIELDS[] = { 0x1040, 0x103C, 0x1018, 0x1038 };

/* Registers with a formula this emitter does not claim to have closed. Empty: the
 * last one was DPU 0x4050, whose depthwise channel field was confounded with the
 * stride for as long as every C=32 capture was stride 1 and every C=64 one stride 2.
 * Manufactured captures at 22 channel counts and both strides separated it.
 *
 * Kept as a mechanism rather than deleted — a register whose formula is open should
 * stay VISIBLE in this gate's output rather than be silently excluded, and the next
 * one will want it. */
static const uint16_t OPEN_FIELDS[] = { 0xFFFF };

/* Where the vendor's FLOAT program and this emitter differ ON PURPOSE, because the
 * vendor's value does not compute with the buffers this library packs.
 *
 * DPU_RDMA BRDMA_CFG (0x501C) is the BS operand reader. Every vendor float program
 * carries 0x100 where the integer ones carry 0x710, and emitting 0x100 makes the DPU
 * write nothing at all — no fault, no dmesg, an untouched surface. Paired with the BS
 * ALU config those same programs carry (0x4044 = 2) the writer runs again and the
 * arithmetic is wrong on about a tenth of the surface. The vendor's float epilogue is
 * a different BS arrangement, and this library's A/B/C coefficient group is packed for
 * the integer one, which is bit-exact under either ALU mode.
 *
 * Reported rather than excluded: this is a divergence with a measured reason, and it
 * should stay visible until the arrangement is decoded as a whole. */
static const uint16_t FLOAT_BS_FIELDS[] = { 0x501C };

static int in_set(uint16_t reg, const uint16_t *set, size_t n)
{
    for (size_t i = 0; i < n; i++) if (set[i] == reg) return 1;
    return 0;
}
#define IN(reg, set) in_set((reg), (set), sizeof(set)/sizeof((set)[0]))

static const char *blk(uint16_t t)
{
    switch (t) {
        case 0x0201: return "CNA ";
        case 0x0801: return "CORE";
        case 0x1001: return "DPU ";
        case 0x2001: return "RDMA";
        default:     return "??  ";
    }
}

struct gate_result { int fail, open, alloc, refetch, checked, excluded, fbs; };

/* Drive the emitter with a captured task's own geometry and diff the two programs. */
static int run_case(const struct rk3576_golden_case *c, struct gate_result *tot,
                    int verbose)
{
    uint64_t ops[256] = {0};
    int fail = 0, open_hits = 0, alloc_hits = 0, refetch_hits = 0, fbs_hits = 0;
    int checked = 0, excluded = 0, rc;
    size_t n_golden = c->n_ops;
    /* Not the first task of its plane: the vendor keeps the overlap rows resident. */
    int continuation = (c->ih_task < c->ih_full && c->pad_top == 0);
    /* The ARGB first-conv programs carry TWO channel counts, and the golden table
     * decodes the wrong one for a caller: `ic` there is 0x1028's folded count (4
     * lanes x kw), while the emitter's input is the image's own channel count, which
     * the program puts in the feature DMA at 0x107C. Read it back out of the capture.
     *
     * They carry no full-plane height at all — every stride in an ARGB program is the
     * task's own, because the packed image is a single surface — so the window is its
     * own plane as far as the CNA is concerned. The OUTPUT plane is still full-height
     * and still decoded (0x401C / ow), which is what the DPU needs. */
    unsigned ic_in = c->ic, ih_full = c->ih_full;
    /* The pad constant (0x1084) is the model's INPUT ZERO POINT, not geometry: the
     * vendor pads the border with it. Captures compiled from different calibration
     * data carry different ones, so drive the emitter with each capture's own rather
     * than excluding the register — that keeps the check on where the emitter puts
     * it. The ARGB path builds the word per channel from the same quantity, and every
     * ARGB capture is zp 0. */
    int zp = 0;
    if (!c->argb) {
        for (size_t k = 0; k < c->n_ops; k++)
            if (c->ops[k].target == 0x0201 && c->ops[k].reg == 0x1084)
                zp = (int)(((c->ops[k].val & 0xFFu) + 0x80u) & 0xFFu);
    }
    if (c->argb) {
        for (size_t k = 0; k < c->n_ops; k++)
            if (c->ops[k].target == 0x0201 && c->ops[k].reg == 0x107c)
                ic_in = c->ops[k].val + 1u;
        ih_full = c->ih_task;
        continuation = 0;
    }
    conv_params_t p = {
        .ic = ic_in, .ih = c->ih_task, .iw = c->iw,
        .oc = c->oc, .oh = c->oh_task, .ow = c->ow,
        .kh = c->kh, .kw = c->kw,
        .stride_y = c->sy, .stride_x = c->sx,
        .pad_top = c->pad_top, .pad_left = c->pad_left,
        .ih_full = ih_full, .oh_full = c->oh_full,
        .input_dma = 0, .weights_dma = 0, .output_dma = 0, .bias_dma = 0,
        .int8_out = 1, .input_zero_point = zp,
        .tasks = ops,
    };

    /* Force the vendor's own CBUF allowance so the comparison is of the program,
     * not of two allocators; a mismatch in F is reported separately below. */
    {
        static char fbuf[32];
        snprintf(fbuf, sizeof fbuf, "%u", c->cbuf_f);
        setenv("ROCKET_RK3576_CBUF_F", fbuf, 1);
    }
    /* The capture's own precision, read off CNA_CONV_CON1 bit 21 rather than assumed.
     * The float captures are manufactured (do_quantization=False on a float ONNX), and
     * they are what turns the transcribed float fields into a gated program. */
    rc = gen_conv2d_rk3576_prec(&p, (int)c->dw,
                                c->is_float ? precision_float16 : precision_int8);
    unsetenv("ROCKET_RK3576_CBUF_F");
    if (rc != 0) {
        printf("  FAIL %s: generator returned %d\n", c->name, rc);
        tot->fail++;
        return 1;
    }

    if (verbose) {
        printf("== %s == (%s)\n", c->name, c->source);
        printf("   %s ic=%u oc=%u k=%ux%u s=%u  in %ux%u of %u -> out %ux%u of %u\n",
               c->dw ? "depthwise" : (c->argb ? "ARGB first-conv" : "direct"),
               ic_in, c->oc, c->kh, c->kw, c->sy,
               c->iw, c->ih_task, ih_full, c->ow, c->oh_task, c->oh_full);
    }

    /* Two preamble variants. Most captures open with an early duplicate write of CNA
     * 0x1038, which the CNA block then rewrites in place; some carry 138 writes
     * instead of 139 because they omit it. Our emitter always writes it — the form
     * hardware is validated against — so skip it when comparing against a program
     * that does not. The value carried there is the same in both. */
    int preamble_skew = !(c->ops[2].target == 0x0201 && c->ops[2].reg == 0x1038);

    if (p.task_count != n_golden + 4 + (unsigned)preamble_skew) {
        printf("  FAIL %s: expected %zu register writes + 4 trailer words, got %u\n",
               c->name, n_golden + (size_t)preamble_skew, p.task_count);
        fail++;
    }

    for (size_t i = 0; i < n_golden && i < p.task_count; i++) {
        size_t j = i + (size_t)(preamble_skew && i >= 2);
        uint16_t gt = c->ops[i].target, gr = c->ops[i].reg;
        uint32_t gv = c->ops[i].val;
        uint16_t et = (uint16_t)(ops[j] >> 48), er = (uint16_t)(ops[j] & 0xFFFF);
        uint32_t ev = (uint32_t)((ops[j] >> 16) & 0xFFFFFFFF);

        if (et != gt || er != gr) {
            printf("  FAIL [%3zu] order: golden %s 0x%04x, emitted %s 0x%04x\n",
                   i, blk(gt), gr, blk(et), er);
            fail++;
            continue;
        }
        if (IN(gr, EXCLUDE_ADDR) || IN(gr, EXCLUDE_QUANT) || IN(gr, EXCLUDE_MODEL)) {
            excluded++; continue;
        }
        if (ev == gv) { checked++; continue; }
        /* CNA 0x1014 carries the stride pair in its low bits and, in the vendor's
         * multi-task program variant only, bit 28 as well. Every capture that sets
         * it is one of the 138-write programs, whose plane is also split into more
         * tasks — so it reads as a multi-task enable rather than a stride term. Our
         * single-task program leaves it clear; the stride nibbles are compared. */
        if (gr == 0x1014 && (gv & ~0x10000000u) == ev) {
            if (verbose)
                printf("  variant[%3zu] %s 0x%04x  golden 0x%08x  emitted 0x%08x\n",
                       i, blk(gt), gr, gv, ev);
            alloc_hits++;
            continue;
        }
        if (continuation && IN(gr, REFETCH_FIELDS)) {
            if (verbose)
                printf("  refetch[%3zu] %s 0x%04x  golden 0x%08x  emitted 0x%08x\n",
                       i, blk(gt), gr, gv, ev);
            refetch_hits++;
            continue;
        }
        if (IN(gr, ALLOC_FIELDS)) {
            if (verbose)
                printf("  alloc[%3zu] %s 0x%04x  golden 0x%08x  emitted 0x%08x\n",
                       i, blk(gt), gr, gv, ev);
            alloc_hits++;
            continue;
        }
        if (c->is_float && IN(gr, FLOAT_BS_FIELDS)) {
            if (verbose)
                printf("  float-bs[%3zu] %s 0x%04x  golden 0x%08x  emitted 0x%08x\n",
                       i, blk(gt), gr, gv, ev);
            fbs_hits++;
            continue;
        }
        if (IN(gr, OPEN_FIELDS)) {
            printf("  open [%3zu] %s 0x%04x  golden 0x%08x  emitted 0x%08x\n",
                   i, blk(gt), gr, gv, ev);
            open_hits++;
            continue;
        }
        printf("  FAIL %-34s [%3zu] %s 0x%04x  golden 0x%08x  emitted 0x%08x\n",
               c->name, i, blk(gt), gr, gv, ev);
        fail++;
    }

    /* The trailer the vendor stream does not carry, and `rocket` cannot run without. */
    if (p.task_count >= 4) {
        uint64_t last = ops[p.task_count - 1];
        uint16_t op = (uint16_t)(last >> 48), reg = (uint16_t)(last & 0xFFFF);
        if (op != OP_ENABLE || reg != PC_OPERATION_ENABLE) {
            printf("  FAIL %s: missing PC_OPERATION_ENABLE trailer "
                   "(last op 0x%04x reg 0x%04x)\n", c->name, op, reg);
            fail++;
        }
    }

    if (verbose)
        printf("   %d matched, %d excluded (address/requant/model), %d allocator, "
               "%d refetch, %d open, %d FAILED\n\n",
               checked, excluded, alloc_hits, refetch_hits, open_hits, fail);
    tot->fail += fail; tot->open += open_hits; tot->alloc += alloc_hits;
    tot->refetch += refetch_hits; tot->fbs += fbs_hits;
    tot->checked += checked; tot->excluded += excluded;
    return fail;
}

/* A sink for the diagnostics a section drives deliberately. */
static void gate_swallow_log(rocket_log_level level, const char *text, void *user)
{
    (void)level; (void)text; (void)user;
}

int main(int argc, char **argv)
{
    int verbose = (argc > 1 && strcmp(argv[1], "-v") == 0);
    struct gate_result tot = {0};
    int fail = 0;
    unsigned n_dw = 0, n_direct = 0, n_argb = 0, n_epi = 0;

    /* This gate drives the emitter with the vendor's own geometries verbatim, and
     * several captures are ic=16 or ic=12 — partial 32-channel groups, which the
     * emitter warns about because our weight cube cannot compute them. That warning
     * is the point of the capture here (register fidelity), not a defect, so keep it
     * out of the gate's output. */
    rocket_log_set_level(ROCKET_LOG_ERROR);

    printf("RK3576 regcmd emitter vs vendor capture\n");
    printf("%zu captured task programs (dedup'd across 8 .rknn captures)\n\n",
           sizeof(rk3576_golden_cases)/sizeof(rk3576_golden_cases[0]));

    for (size_t i = 0; i < sizeof(rk3576_golden_cases)/sizeof(rk3576_golden_cases[0]); i++) {
        const struct rk3576_golden_case *c = &rk3576_golden_cases[i];
        /* conv + a DPU epilogue stage: same geometry as the plain conv2d capture,
         * carried for the epilogue registers rather than as a conv oracle. */
        if (c->epilogue) { n_epi++; continue; }
        if (c->dw) n_dw++; else if (c->argb) n_argb++; else n_direct++;
        fail += run_case(c, &tot, verbose);
    }

    printf("== capture diff ==\n");
    printf("   %u direct + %u depthwise + %u ARGB first-conv programs compared "
           "(%u conv-with-epilogue carried for reference)\n",
           n_direct, n_dw, n_argb, n_epi);
    printf("   %d registers matched, %d excluded (address/requant/model),\n"
           "   %d allocator-choice, %d vendor CBUF row-reuse, %d float BS arrangement,\n"
           "   %d open, %d FAILED\n\n",
           tot.checked, tot.excluded, tot.alloc, tot.refetch, tot.fbs,
           tot.open, tot.fail);

    /* The ARGB first-conv datapath's guards. The captures pin an int8 and an fp16
     * form at 1, 3 and 4 image channels and a width that is a multiple of 16, so the
     * emitter has to say so rather than emit a program outside what any of them pins.
     * bf16 is the case that would be a guess: it shares fp16's width class in the CNA
     * word, and no capture separates the two on this path. */
    {
        int argb_fail = 0;
        struct { const char *what; unsigned ic, iw; int dw; unsigned prec; int want_ok; } g[] = {
            { "RGB 64x64 accepted",            3, 64, 0, precision_int8,    1 },
            { "RGBA accepted",                 4, 64, 0, precision_int8,    1 },
            { "grayscale accepted",            1, 64, 0, precision_int8,    1 },
            { "fp16 RGB accepted",             3, 64, 0, precision_float16, 1 },
            { "fp16 RGBA accepted",            4, 64, 0, precision_float16, 1 },
            { "iw not a multiple of 16",       3, 60, 0, precision_int8,    0 },
            { "no depthwise form",             3, 64, 1, precision_int8,    0 },
            { "int8 and fp16 only",            3, 64, 0, precision_bfloat16, 0 },
        };
        printf("== ARGB first-conv guards ==\n");
        for (size_t i = 0; i < sizeof(g)/sizeof(g[0]); i++) {
            uint64_t ops[256] = {0};
            conv_params_t p = { .ic = g[i].ic, .ih = 64, .iw = g[i].iw,
                                .oc = 32, .oh = 32, .ow = g[i].iw / 2,
                                .kh = 3, .kw = 3, .stride_y = 2, .stride_x = 2,
                                .pad_top = 1, .pad_left = 1, .tasks = ops };
            int rc = gen_conv2d_rk3576_prec(&p, g[i].dw, g[i].prec);
            if ((rc == 0) != (g[i].want_ok != 0)) {
                printf("   FAIL %-30s expected %s\n", g[i].what,
                       g[i].want_ok ? "acceptance" : "a refusal");
                argb_fail++;
            }
        }
        if (!argb_fail) printf("   %zu guards hold\n", sizeof(g)/sizeof(g[0]));
        printf("\n");
        fail += argb_fail;
    }

    /* CBUF row reuse. The vendor's continuation tasks fetch fewer rows than their
     * window is tall, keep the overlap resident and rotate the CBUF base; the plain
     * emitter refetches instead, so seven registers legitimately differ on those
     * tasks and run_case() above only reports them. Driven through the reuse entry
     * point they must match EXACTLY — which is what makes the vendor's split
     * programs a complete oracle rather than a partial one.
     *
     * Each row is the capture, the task before it, and the two numbers a caller
     * supplies: the leading window rows the previous task already fetched, and the
     * CBUF granules the sequence has filled. The vendor compiles the same graph
     * three times with different CBUF origins (0, 7168, 6144), which is why the
     * resident count is data here rather than derived — the origin is the
     * allocator's, not the geometry's. */
    {
        static const struct { const char *name, *prev; unsigned retained, resident; } rz[] = {
            /* per-layer splits that RETAIN, at both strides and both channel counts */
            { "dw_c32_32_k3_s1_112x23_o112x22",     "dw_c32_32_k3_s1_112x91_o112x90",    2,  5096 },
            { "dw_c64_64_k3_s2_112x45_o56x22",      "dw_c64_64_k3_s2_112x44_o56x22",     1,  4928 },
            { "dw_c32_32_k3_s1_112x23_o112x22_v3",  "dw_c32_32_k3_s1_112x91_o112x90_v3", 2,  5096 },
            /* the same splits at the 7168-granule CBUF origin */
            { "dw_c32_32_k3_s1_112x23_o112x22_v2",  "dw_c32_32_k3_s1_112x91_o112x90_v2", 2, 12264 },
            { "dw_c64_64_k3_s2_112x45_o56x22_v2",   "dw_c64_64_k3_s2_112x44_o56x22_v2",  1, 12096 },
            { "dw_c32_32_k3_s1_112x23_o112x22_v4",  "dw_c32_32_k3_s1_112x91_o112x90_v4", 2, 12264 },
            /* k=1: no overlap to keep, so the vendor resets the base to its origin */
            { "conv_c32_64_k1_s1_112x21_o112x21",   "conv_c32_64_k1_s1_112x91_o112x91",  0,     0 },
            { "conv_c32_64_k1_s1_112x21_o112x21_v2","conv_c32_64_k1_s1_112x91_o112x91_v2",0, 7168 },
            /* the whole-graph tile split: a second pass over the plane's other half,
             * which retains nothing because four other layers used the CBUF between */
            { "argb_c12_32_k3_s2_224x113_o112x56",  "argb_c12_32_k3_s2_224x112_o112x56", 0,  6144 },
            { "dw_c32_32_k3_s1_112x57_o112x56_v2",  "dw_c32_32_k3_s1_112x57_o112x56",    0,  6144 },
            { "conv_c32_64_k1_s1_112x56_o112x56_v2","conv_c32_64_k1_s1_112x56_o112x56",  0,  6144 },
            { "dw_c64_64_k3_s2_112x37_o56x18_v2",   "dw_c64_64_k3_s2_112x37_o56x18",     0,  6144 },
            { "conv_c16_128_k5_s2_80x41_o40x20",    "conv_c16_128_k5_s2_80x42_o40x20",   0,  6144 },
        };
        /* Everything the reuse path moves: the fetch counts, the two CBUF bases, and
         * the two bits that mark a task as a continuation and as retaining. */
        static const uint16_t REUSE_REGS[] = {
            0x1028, 0x1078, 0x1098,   /* what this task fetches      */
            0x103C, 0x1040,           /* where the window and the fetch sit */
            0x1018, 0x1038,           /* continuation / retain bits  */
        };
        int reuse_fail = 0, reuse_checked = 0;
        printf("== CBUF row reuse ==\n");
        for (size_t i = 0; i < sizeof(rz)/sizeof(rz[0]); i++) {
            const struct rk3576_golden_case *c = NULL;
            uint64_t ops[256] = {0};
            rocket_rk3576_row_task t = {0};
            conv_params_t p = {0};
            char fbuf[32];
            unsigned ic_in;

            for (size_t k = 0; k < sizeof(rk3576_golden_cases)/sizeof(rk3576_golden_cases[0]); k++)
                if (strcmp(rk3576_golden_cases[k].name, rz[i].name) == 0)
                    c = &rk3576_golden_cases[k];
            if (!c) {
                printf("   FAIL %-38s no such capture\n", rz[i].name);
                reuse_fail++; continue;
            }
            ic_in = c->ic;
            if (c->argb)
                for (size_t k = 0; k < c->n_ops; k++)
                    if (c->ops[k].target == 0x0201 && c->ops[k].reg == 0x107c)
                        ic_in = c->ops[k].val + 1u;

            p.ic = ic_in;  p.ih = c->ih_task; p.iw = c->iw;
            p.oc = c->oc;  p.oh = c->oh_task; p.ow = c->ow;
            p.kh = c->kh;  p.kw = c->kw;
            p.stride_y = c->sy; p.stride_x = c->sx;
            p.pad_top = c->pad_top; p.pad_left = c->pad_left;
            p.ih_full = c->argb ? c->ih_task : c->ih_full;
            p.oh_full = c->oh_full;
            p.int8_out = 1;
            p.tasks = ops;
            t.ih = c->ih_task;
            t.retained = (uint16_t)rz[i].retained;
            t.cbuf_resident = rz[i].resident;

            snprintf(fbuf, sizeof fbuf, "%u", c->cbuf_f);
            setenv("ROCKET_RK3576_CBUF_F", fbuf, 1);
            if (gen_conv2d_int8_rk3576_reuse(&p, c->dw, &t, 1) != 0) {
                printf("   FAIL %-38s generator refused\n", rz[i].name);
                unsetenv("ROCKET_RK3576_CBUF_F");
                reuse_fail++; continue;
            }
            unsetenv("ROCKET_RK3576_CBUF_F");

            /* Compare the LAST write of each register on both sides: 0x1038 is
             * written twice (an early preamble copy, then the CNA block), and only
             * the second one is the value the hardware runs with.
             *
             * The masks keep this check to the reuse mechanism. Three of the vendor's
             * bits here are its allocator's compile-variant choices, not reuse:
             * 0x1018's low 16, 0x1038's low bits and 0x1040's F field and bit 29 all
             * take one value per compile of the graph and were measured don't-care on
             * the int8 path. What is checked is the two bases, the fetch counts, and
             * the two bits that carry the mechanism. */
            for (size_t r = 0; r < sizeof(REUSE_REGS)/sizeof(REUSE_REGS[0]); r++) {
                uint16_t reg = REUSE_REGS[r];
                uint32_t gv = 0, ev = 0, mask;
                int have_g = 0, have_e = 0;
                for (size_t k = 0; k < c->n_ops; k++)
                    if (c->ops[k].target == 0x0201 && c->ops[k].reg == reg) {
                        gv = c->ops[k].val; have_g = 1;
                    }
                for (unsigned e = 0; e < p.task_count; e++)
                    if ((uint16_t)(ops[e] >> 48) == 0x0201 &&
                        (uint16_t)(ops[e] & 0xFFFF) == reg) {
                        ev = (uint32_t)((ops[e] >> 16) & 0xFFFFFFFF); have_e = 1;
                    }
                if (!have_g || !have_e) continue;
                mask = reg == 0x1018 ? 0xFFFF0000u
                     : reg == 0x1038 ? 0x80000000u
                     : reg == 0x1040 ? 0x1000FFFFu : 0xFFFFFFFFu;
                if ((ev & mask) != (gv & mask)) {
                    printf("   FAIL %-38s 0x%04x golden 0x%08x emitted 0x%08x "
                           "(mask 0x%08x)\n", rz[i].name, reg, gv, ev, mask);
                    reuse_fail++;
                } else {
                    reuse_checked++;
                }
            }
        }
        if (!reuse_fail)
            printf("   %zu continuation tasks reproduced, %d registers matched\n",
                   sizeof(rz)/sizeof(rz[0]), reuse_checked);
        printf("\n");
        fail += reuse_fail;
    }

    /* The CBUF allowance planner, off-device. Every expectation here is a hardware
     * measurement (see rockchip-npu-notes/chips/rk3576-regcmd.md): the budget is
     * 4096+F granules, only the rungs 0/256/512/1024/2048 deliver their face value,
     * the data side caps at 6144, and the weight path gets what is left of the ~448
     * KiB pool. A plane past its allowance computes WRONG with no fault to catch it,
     * so these are the arithmetic that stands between a caller and silent corruption.
     *
     * AND 256 AND 512 DELIVER ONLY AT kh == 1, so the planner offers them only there.
     * At kh > 1 each delivers 4096 granules — the F=0 budget — measured at one granule
     * total across five plane widths: k=1 exact at every width, k=3 wrong at every one,
     * k=5 wrong, and the F=0 controls exact at k=3. The pairs below assert both sides,
     * because a planner that forgot the kernel would pass the k=1 half alone.
     * [HW sweep, H96 MAX M9, tests/rk3576_conv_lib_gate.c rung256] */
    {
        struct { const char *what; unsigned iw, ic, ih, oc, kh, kw; int dw;
                 int want_ok; unsigned want_f; } pc[] = {
            /* the two captured geometries: the planner must reproduce their words */
            { "conv2d capture (1600 gr)",   80,  16,  80, 128, 5, 5, 0, 1,    0 },
            { "dw capture (5096 gr)",      112,  32,  91,  32, 3, 3, 1, 1, 1024 },
            /* rung selection: the lowest rung that covers the plane */
            { "4096 gr exactly",            16,  32, 512,  32, 1, 1, 0, 1,    0 },
            { "4097 gr k1 -> 256",          16,  32, 513,  32, 1, 1, 0, 1,  256 },
            { "4352 gr k1 -> 256 exactly",  16,  32, 544,  32, 1, 1, 0, 1,  256 },
            { "4608 gr k1 -> 512",          16,  32, 576,  32, 1, 1, 0, 1,  512 },
            /* the same totals at k=3, where those two rungs deliver only 4096 */
            { "4097 gr k3 -> 1024 not 256", 16,  32, 513,  32, 3, 3, 0, 1, 1024 },
            { "4352 gr k3 -> 1024 not 256", 16,  32, 544,  32, 3, 3, 0, 1, 1024 },
            { "4608 gr k3 -> 1024 not 512", 16,  32, 576,  32, 3, 3, 0, 1, 1024 },
            { "4096 gr k3 -> 0, unchanged", 16,  32, 512,  32, 3, 3, 0, 1,    0 },
            { "5120 gr -> 1024",            16,  32, 640,  32, 1, 1, 0, 1, 1024 },
            /* 5600 needs 1504, which is NOT a rung: 1536 would underdeliver on HW */
            { "5600 gr -> 2048 not 1536",   16,  32, 700,  32, 1, 1, 0, 1, 2048 },
            { "6144 gr -> 2048 (the cap)",  16,  32, 768,  32, 1, 1, 0, 1, 2048 },
            /* past the data cap, and the weight-path guard */
            { "6152 gr -> past the cap",     16, 32, 769,  32, 1, 1, 0, 0,    0 },
            { "192 KiB slice fits at F=0",    8,1536,  8,  32, 2, 2, 0, 1,    0 },
            { "196 KiB slice does not",       8,1568,  8,  32, 2, 2, 0, 0,    0 },
        };
        int planner_fail = 0;
        printf("== CBUF allowance planner ==\n");
        for (size_t i = 0; i < sizeof(pc)/sizeof(pc[0]); i++) {
            unsigned f = 0xFFFFFFFFu;
            int rc = rocket_rk3576_cbuf_f(pc[i].iw, pc[i].ic, pc[i].ih, pc[i].oc,
                                          pc[i].kh, pc[i].kw, pc[i].dw, &f);
            if ((rc == 0) != (pc[i].want_ok != 0)) {
                printf("   FAIL %-28s expected %s, got %s\n", pc[i].what,
                       pc[i].want_ok ? "a plan" : "a refusal",
                       rc == 0 ? "a plan" : "a refusal");
                planner_fail++;
            } else if (rc == 0 && f != pc[i].want_f) {
                printf("   FAIL %-28s expected F=%u, got F=%u\n",
                       pc[i].what, pc[i].want_f, f);
                planner_fail++;
            }
        }
        /* The tiler-facing helper: the tallest window one task can carry. At ic=32
         * k=1 the weight slice is 1 KiB, so the full 2048 rung is available and the
         * budget is the 6144-granule cap: 6144/8 = 768 rows at iw=16. */
        {
            unsigned rows = rocket_rk3576_max_task_rows(16, 32, 32, 1, 1, 0);
            if (rows != 768) {
                printf("   FAIL max_task_rows(iw16 ic32 k1) expected 768, got %u\n", rows);
                planner_fail++;
            }
        }
        /* A weight slice big enough to leave no rung holds the window at the base
         * budget. ic=1472 k=2 is a 184 KiB slice: it fits beside F=0 (4096+2944 of
         * 7168 granules) but not beside the 256 rung (7296), so the window is
         * 4096/entries and not one row more. The neighbouring ic=1408 (176 KiB) DOES
         * leave room for 256 — but both are k=2, where that rung delivers nothing, so
         * both now hold at the base budget. That is the COST of the kernel rule,
         * stated: a k>1 slice that used to buy 256 granules of window buys none, which
         * is one more row task and not a wrong answer. The k=1 half of the rule is
         * asserted by the 4352-granule pair in the table above, which is the same query
         * with the kernel as the only variable. */
        {
            unsigned rows = rocket_rk3576_max_task_rows(16, 1472, 32, 2, 2, 0);
            unsigned entries = (16u * 1472u + 63u) / 64u;      /* 368 granules/row */
            if (rows != 4096u / entries) {
                printf("   FAIL max_task_rows(no rung fits) expected %u, got %u\n",
                       4096u / entries, rows);
                planner_fail++;
            }
        }
        {
            unsigned rows = rocket_rk3576_max_task_rows(16, 1408, 32, 2, 2, 0);
            unsigned entries = (16u * 1408u + 63u) / 64u;      /* 352 granules/row */
            if (rows != 4096u / entries) {
                printf("   FAIL max_task_rows(k2, 256 rung dead) expected %u, got %u\n",
                       4096u / entries, rows);
                planner_fail++;
            }
        }
        if (!planner_fail)
            printf("   %zu plans + 3 window queries as measured\n",
                   sizeof(pc)/sizeof(pc[0]));
        printf("\n");
        fail += planner_fail;
    }

    /* The row window. Every property below is what makes a split COVER the plane —
     * a gap or an overlap in the output runs is a silently wrong conv, and a window
     * over the allowance is the silent corruption the whole planner exists to avoid.
     * All of it is arithmetic, so it belongs off-device; the shapes are the ones run
     * bit-exactly on the part. */
    {
        struct { const char *what; unsigned iw, ih, ic, oc, kh, s, pad;
                 unsigned cap; unsigned want_tasks; } rc_[] = {
            /* 112x112 at ic=32 is 6272 granules against a 6144 cap — the ordinary
             * vision-model geometry that has no single-task plan at all. */
            { "112x112 ic32 k3 s1 SAME",  112, 112,  32,  32, 3, 1, 1,   0, 2 },
            { "224x224 ic32 k3 s1 SAME",  224, 224,  32,  32, 3, 1, 1,   0, 5 },
            /* Forced caps: the same plane cut finer, which is how a shape that fits
             * one task is made to exercise the split against its own known result. */
            { "64x64 ic64 k3 SAME cap35",  64,  64,  64,  64, 3, 1, 1,  35, 2 },
            { "64x64 ic64 k3 SAME cap18",  64,  64,  64,  64, 3, 1, 1,  18, 4 },
            { "64x64 ic64 k3 SAME cap10",  64,  64,  64,  64, 3, 1, 1,  10, 8 },
            { "64x64 ic32 k5 s2 SAME cap7",64,  64,  32,  32, 5, 2, 1,   7,16 },
            { "32x32 ic32 k1 s1 cap4",     32,  32,  32,  32, 1, 1, 0,   4, 8 },
            { "fits in one task",          32,  32,  32,  32, 3, 1, 1,   0, 1 },
        };
        int rows_fail = 0;
        printf("== row window ==\n");
        for (size_t i = 0; i < sizeof(rc_)/sizeof(rc_[0]); i++) {
            conv_params_t p = {0};
            rocket_rk3576_row_task t[512];
            unsigned n = 0, k, covered = 0, cap = rc_[i].cap;
            char capenv[32];

            p.iw = (uint16_t)rc_[i].iw; p.ih = (uint16_t)rc_[i].ih;
            p.ic = (uint16_t)rc_[i].ic; p.oc = (uint16_t)rc_[i].oc;
            p.kh = (uint16_t)rc_[i].kh; p.kw = (uint16_t)rc_[i].kh;
            p.stride_y = (uint8_t)rc_[i].s; p.stride_x = (uint8_t)rc_[i].s;
            p.pad_top = (uint8_t)rc_[i].pad; p.pad_left = (uint8_t)rc_[i].pad;
            p.oh = (uint16_t)((rc_[i].ih + 2*rc_[i].pad - rc_[i].kh) / rc_[i].s + 1);
            p.ow = (uint16_t)((rc_[i].iw + 2*rc_[i].pad - rc_[i].kh) / rc_[i].s + 1);
            p.ih_full = p.ih; p.oh_full = p.oh;

            if (cap) { snprintf(capenv, sizeof capenv, "%u", cap);
                       setenv("ROCKET_RK3576_MAX_ROWS", capenv, 1); }
            else       unsetenv("ROCKET_RK3576_MAX_ROWS");

            if (rocket_rk3576_plan_rows(&p, 0, t, 512, &n) < 0) {
                printf("   FAIL %-30s refused\n", rc_[i].what);
                rows_fail++; continue;
            }
            if (n != rc_[i].want_tasks) {
                printf("   FAIL %-30s expected %u tasks, got %u\n",
                       rc_[i].what, rc_[i].want_tasks, n);
                rows_fail++;
            }
            for (k = 0; k < n; k++) {
                unsigned iy0, ih;
                /* The output runs must tile [0, oh_full) exactly: contiguous, in
                 * order, no gap and no overlap. */
                if (t[k].oy0 != covered) {
                    printf("   FAIL %-30s task %u starts at out row %u, expected %u\n",
                           rc_[i].what, k, t[k].oy0, covered);
                    rows_fail++; break;
                }
                covered += t[k].oh;
                /* The input window must be exactly the rows those output rows read,
                 * clipped to the plane, with the leading pad they do not find there. */
                {
                    long first = (long)t[k].oy0 * p.stride_y - (long)p.pad_top;
                    long last  = ((long)t[k].oy0 + t[k].oh - 1) * p.stride_y
                                 - (long)p.pad_top + p.kh - 1;
                    unsigned want_pt = first < 0 ? (unsigned)(-first) : 0u;
                    if (first < 0) first = 0;
                    if (last > (long)p.ih_full - 1) last = (long)p.ih_full - 1;
                    iy0 = (unsigned)first;
                    ih  = (unsigned)(last - first + 1);
                    if (t[k].iy0 != iy0 || t[k].ih != ih || t[k].pad_top != want_pt) {
                        printf("   FAIL %-30s task %u window %u+%u pt%u, expected %u+%u pt%u\n",
                               rc_[i].what, k, t[k].iy0, t[k].ih, t[k].pad_top,
                               iy0, ih, want_pt);
                        rows_fail++; break;
                    }
                }
                /* Every window has to satisfy the allowance, and the byte offsets are
                 * plain row strides in the 16-byte channel atom. */
                if (rocket_rk3576_cbuf_f(p.iw, p.ic, t[k].ih, p.oc, p.kh, p.kw, 0, NULL) < 0) {
                    printf("   FAIL %-30s task %u window is over the allowance\n",
                           rc_[i].what, k);
                    rows_fail++; break;
                }
                if (t[k].feature_off != t[k].iy0 * p.iw * 16u ||
                    t[k].output_off  != t[k].oy0 * p.ow * 16u) {
                    printf("   FAIL %-30s task %u offsets\n", rc_[i].what, k);
                    rows_fail++; break;
                }
                /* The reuse bookkeeping: retained is exactly the rows this window
                 * shares with the previous one, and the resident count accumulates
                 * the rows actually fetched — resetting whenever nothing is kept,
                 * which is what the vendor's k=1 continuation does. */
                {
                    unsigned entries = (p.iw * p.ic + 63u) / 64u;
                    unsigned want_ret = (k && t[k].iy0 < t[k-1].iy0 + t[k-1].ih)
                                        ? (unsigned)(t[k-1].iy0 + t[k-1].ih - t[k].iy0)
                                        : 0u;
                    unsigned want_res = 0;
                    if (want_ret)
                        want_res = t[k-1].cbuf_resident + entries * (t[k-1].ih - t[k-1].retained);
                    if (t[k].retained != want_ret || t[k].cbuf_resident != want_res) {
                        printf("   FAIL %-30s task %u reuse: retained %u resident %u, "
                               "expected %u / %u\n", rc_[i].what, k,
                               t[k].retained, t[k].cbuf_resident, want_ret, want_res);
                        rows_fail++; break;
                    }
                }
            }
            if (k == n && covered != p.oh_full) {
                printf("   FAIL %-30s covers %u of %u output rows\n",
                       rc_[i].what, covered, p.oh_full);
                rows_fail++;
            }
        }
        unsetenv("ROCKET_RK3576_MAX_ROWS");
        if (!rows_fail)
            printf("   %zu plans tile their planes exactly\n", sizeof(rc_)/sizeof(rc_[0]));
        printf("\n");
        fail += rows_fail;
    }

    /* ====================================================================
     * The fp16 ic split, as far as it can be checked without the part.
     *
     * The whole point of the split is that a caller CANNOT reach the shape that
     * computes wrong. That refusal is a host-side property and belongs in a gate
     * that always runs — an envelope enforced only by a board test is an envelope
     * that regresses the first time the board is unavailable.
     * ==================================================================*/
    {
        static const struct { unsigned ic, oc, iw, ih, k; unsigned want; } ics[] = {
            { 16, 32, 16, 16, 3, 1 },
            { 32, 32, 16, 16, 3, 2 },
            { 64, 32, 16, 16, 1, 4 },
            { 128,64, 32, 32, 5, 8 },
        };
        int ic_fail = 0;
        printf("== fp16 ic split ==\n");
        /* This section drives the refusals on purpose, so their diagnostics are the
         * expected result rather than noise to read. Swallow them; a real failure
         * still reports through printf below. */
        rocket_log_set_callback(gate_swallow_log, NULL);
        for (size_t i = 0; i < sizeof(ics)/sizeof(ics[0]); i++) {
            conv_params_t p = {0};
            rocket_rk3576_ic_task t[64];
            unsigned n = 0, s;

            p.ic = (uint16_t)ics[i].ic; p.oc = (uint16_t)ics[i].oc;
            p.iw = (uint16_t)ics[i].iw; p.ih = (uint16_t)ics[i].ih;
            p.kh = (uint16_t)ics[i].k;  p.kw = (uint16_t)ics[i].k;
            p.stride_y = 1; p.stride_x = 1;
            p.ow = (uint16_t)(ics[i].iw - ics[i].k + 1);
            p.oh = (uint16_t)(ics[i].ih - ics[i].k + 1);
            p.ih_full = p.ih; p.oh_full = p.oh;

            if (rocket_rk3576_plan_ic(&p, t, 64, &n) < 0 || n != ics[i].want) {
                printf("   FAIL ic=%u expected %u slices, got %u\n",
                       ics[i].ic, ics[i].want, n);
                ic_fail++; continue;
            }
            for (s = 0; s < n; s++) {
                /* Every slice is one contraction step, and the feature side is a
                 * plain base offset in the 16-byte channel atom — nothing repacked. */
                if (t[s].ic != ROCKET_RK3576_FP16_IC_SLICE ||
                    t[s].ic0 != s * ROCKET_RK3576_FP16_IC_SLICE ||
                    t[s].feature_off != s * (ROCKET_RK3576_FP16_IC_SLICE / 8u)
                                          * p.iw * p.ih_full * 16u) {
                    printf("   FAIL ic=%u slice %u: ic0 %u ic %u off %u\n",
                           ics[i].ic, s, t[s].ic0, t[s].ic, t[s].feature_off);
                    ic_fail++; break;
                }
            }
            /* The refusal. A task past one contraction step writes a full, correctly
             * sized, WRONG surface, so the emitter must reject rather than warn. */
            {
                conv_params_t q = p;
                uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
                q.tasks = ops;
                if (q.ic != ROCKET_RK3576_FP16_IC_SLICE &&
                    gen_conv2d_fp16_rk3576(&q) == 0) {
                    printf("   FAIL ic=%u reached the emitter instead of being "
                           "refused\n", ics[i].ic);
                    ic_fail++;
                }
                q.ic = (uint16_t)ROCKET_RK3576_FP16_IC_SLICE;
                if (gen_conv2d_fp16_rk3576(&q) != 0) {
                    printf("   FAIL ic=%u: the emitter refused a single slice\n",
                           ics[i].ic);
                    ic_fail++;
                }
            }
        }
        /* A partial slice and a partial float output group are both refusals. */
        {
            conv_params_t p = {0};
            rocket_rk3576_ic_task t[64];
            unsigned n = 0;
            p.ic = 12; p.oc = 32; p.iw = 16; p.ih = 16; p.kh = 1; p.kw = 1;
            p.ow = 16; p.oh = 16; p.stride_y = 1; p.stride_x = 1;
            if (rocket_rk3576_plan_ic(&p, t, 64, &n) == 0) {
                printf("   FAIL ic=12 planned instead of being refused\n"); ic_fail++;
            }
            p.ic = 32; p.oc = 12;
            if (rocket_rk3576_plan_ic(&p, t, 64, &n) == 0) {
                printf("   FAIL oc=12 planned instead of being refused\n"); ic_fail++;
            }
        }
        /* The output map is a BIJECTION ONTO the whole surface — the property it was
         * measured as, and the one an off-by-one breaks. At the 16-channel
         * contraction the writer spends exactly two bytes on each element and every
         * programmed channel is present, so the map is onto and not merely into. */
        {
            unsigned oc = 32, oh = 5, ow = 7;
            unsigned nout = rocket_rk3576_fp16_out_channels(oc);
            size_t words = rocket_rk3576_fp16_out_bytes(oc, oh, ow) / 2u;
            unsigned char *seen = calloc(words, 1);
            unsigned c, y, x, hits = 0;
            if (!seen) return 1;
            for (c = 0; c < nout; c++)
                for (y = 0; y < oh; y++)
                    for (x = 0; x < ow; x++) {
                        int w = rocket_rk3576_fp16_out_index(oh, ow, c, y, x);
                        if (w < 0 || (size_t)w >= words || seen[w]) {
                            printf("   FAIL out map c=%u (%u,%u) -> %d\n", c, y, x, w);
                            ic_fail++; c = nout; y = oh; break;
                        }
                        seen[w] = 1; hits++;
                    }
            if (hits != nout * oh * ow || hits != words) {
                printf("   FAIL out map covers %u words of %zu\n", hits, words);
                ic_fail++;
            }
            free(seen);
        }
        rocket_log_set_callback(NULL, NULL);
        if (!ic_fail)
            printf("   %zu plans, their refusals, and a bijective output map\n",
                   sizeof(ics)/sizeof(ics[0]));
        printf("\n");
        fail += ic_fail;
    }

    if (fail) { printf("RK3576 regcmd gate: FAIL (%d)\n", fail); return 1; }
    printf("RK3576 regcmd gate: PASS\n");
    return 0;
}
