// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_poison_probe.c — which programs poison the NEXT submit, and what carries it.
 *
 * A job on this part can leave the NPU in a state where the FOLLOWING submit completes
 * normally, in its usual ~1.4 ms, and writes nothing at all. It is cleared by the
 * runtime-PM autosuspend cycling the NPU power domain, not by elapsed time, so a
 * caller pays a power-idle guard per call — two orders of magnitude on the paths that
 * carry it.
 *
 * THE MEASUREMENT PROBLEM this probe exists to solve. Asking "did program A write?"
 * cannot separate the two ways A can come back empty:
 *
 *     A is DEAD          — its own register program computes nothing, and
 *     A was POISONED     — by whatever ran before it.
 *
 * Every register experiment on the hazard runs straight into that: override a register
 * to see whether the poisoning stops, and a value that merely breaks the program reads
 * exactly like a value that fixes it. So this probe never scores the program under
 * test. It runs
 *
 *     stamp A's surface and B's, submit A, wait `gap`, submit B, read both
 *
 * with B a CANARY that is known not to poison and known to write, and scores B. A may
 * compute nonsense; the question is only whether the part still works after it. That
 * decouples "does A compute" from "does A poison", and it is what makes a
 * leave-one-out over a register delta mean anything.
 *
 * A's own result is reported beside B's, because the pair is what classifies an
 * override: a register that leaves A computing AND stops it poisoning is the mechanism,
 * one that kills A and stops the poisoning only says a dead program poisons nothing.
 * A DEAD PROGRAM CAN ALSO POISON — driving `0x4038`'s low half to the int8 value while
 * the rest of the program stays float writes nothing and poisons anyway — so "it wrote
 * nothing" does not classify an override either way.
 *
 * WHAT THE SWEEPS FOUND. The two paths poison through different register conditions.
 * The int32 writers carry it in ONE register, DPU `0x4010`'s `out` field. The fp16 path
 * carries it in TWO BITS TOGETHER, DPU `0x4038[4]` and `0x4050[17]`, and neither alone
 * does anything — which is why every leave-one-out over that delta came back empty. Both
 * bits are load-bearing for the float arithmetic: cleared, the fp16 direct conv stops
 * poisoning, still writes, and computes wrong at every shape that was exact. So there is
 * no fp16 encoding that does not poison, and the guard cannot be encoded away.
 *
 * The state is not in the DPU register file: the canary already writes both registers at
 * their int8 values and is poisoned anyway, and adding the nine DPU registers neither
 * program writes changes nothing. Nor is the poisoning the same hazard as the missing DPU
 * completion that `rocket.dpu_grace_us` bounds — `time` separates them, the int32 writers
 * always paying the grace where the fp16 programs mostly retire at the poll floor, while
 * all four poison every time.
 *
 * Modes:
 *   rk3576_poison_probe scope        the full A-poisons-B matrix over the program kinds
 *   rk3576_poison_probe pair A B     one pair, at ROCKET_PP_GAP_MS
 *   rk3576_poison_probe scan A B     leave-one-out: apply each ROCKET_PP_REGS entry to
 *                                    A alone and re-measure B
 *   rk3576_poison_probe chain A B    how DEEP it goes: one A, then B over and over,
 *                                    reporting which position recovers
 *   rk3576_poison_probe regs A B     the register SET difference, including the registers
 *                                    A writes that B never does — the ones a leave-one-out
 *                                    cannot reach
 *   rk3576_poison_probe heal A B     the JOINT sweep: apply the whole counterpart delta to
 *                                    A, then minimise it by delta debugging
 *   rk3576_poison_probe xadd A B     whether a register WRITE in the next job clears it:
 *                                    add A's exclusive registers to the canary, minimise
 *   rk3576_poison_probe time         per-kind submit wall time, which separates the
 *                                    programs that raise a DPU completion from the ones
 *                                    the driver has to time out on
 *
 * Program kinds: int8fc fp16fc int8d fp16d i32 i32w dw
 *
 * Env knobs:
 *   ROCKET_PP_GAP_MS   gap between A's completion and B's submit (default 0)
 *   ROCKET_PP_REPS     A/B rounds per measurement (default 20)
 *   ROCKET_PP_REGS     "0x4010=0,0x4038=0x120080,…" — the leave-one-out list for `scan`
 *   ROCKET_PP_RECOVER  ms of idle BEFORE each round. Without it the A,B,A,B chain wedges
 *                      permanently the first time B poisons: A is then poisoned too and
 *                      neither ever writes again, so every later cell reads zero and the
 *                      matrix says nothing. The default tracks the driver, at four times
 *                      power/autosuspend_delay_ms plus 50 — the clear is the power domain
 *                      cycling rather than elapsed time, the working gap is twice the
 *                      delay, and a recovery merely AT that boundary leaves the program
 *                      under test itself short of the round count in the odd cell.
 *   ROCKET_PP_SETTLE   seconds of idle before the first submit (default 6). The part
 *                      carries the hazard across PROCESSES, so a probe that starts too
 *                      soon after the last one measures the previous process.
 *   ROCKET_PP_XREG     `heal` only: also offer A's EXCLUSIVE registers (the ones the
 *                      counterpart never writes) to the sweep, at 0. Off by default,
 *                      because zero is a guess at "neutral" where a counterpart value
 *                      is a fact.
 *   ROCKET_PP_ADD      `xadd` only: an explicit "0x4054=0,..." list to add to the canary
 *                      instead of A's exclusive set.
 *
 * Exit: 0, or 2 if there is no NPU (skip).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"

#define SENTINEL 0xAA
#define C2 16

enum { PK_INT8FC, PK_FP16FC, PK_INT8D, PK_FP16D, PK_I32, PK_I32W, PK_DW, PK_N };

static const char *const KIND_NAME[PK_N] = {
    "int8fc", "fp16fc", "int8d", "fp16d", "i32", "i32w", "dw"
};

/* One geometry per kind. The two first-conv kinds share theirs exactly, so the
 * register delta between them is precision and nothing else; likewise the direct
 * pair, except for the input-channel count each emitter contracts (16 at fp16, the
 * 32-channel MAC group at int8), which is not a free axis on either. */
struct geom { unsigned ic, oc, iw, ih, k, stride, pad; };
static const struct geom KIND_GEOM[PK_N] = {
    /* int8fc */ { 4, 32, 16, 16, 3, 1, 1 },
    /* fp16fc */ { 4, 32, 16, 16, 3, 1, 1 },
    /* int8d  */ { 32, 32, 8, 8, 1, 1, 0 },
    /* fp16d  */ { 16, 32, 8, 8, 1, 1, 0 },
    /* i32    */ { 32, 32, 8, 8, 1, 1, 0 },
    /* i32w   */ { 32, 32, 8, 8, 1, 1, 0 },
    /* dw     */ { 32, 32, 8, 8, 3, 1, 1 },
};

struct prog {
    int          kind;
    rocket_bo    in, w, b, o, r;
    conv_params_t p;
    uint64_t     ops[RK3576_CONV_TASK_OPS];
    uint32_t     in_h[4], out_h[1];
    size_t       obytes;      /* the whole output allocation, which is what is scanned */
    int          live;
};

static int env_int(const char *name, int dflt)
{
    const char *e = getenv(name);
    return (e && *e) ? (int)strtol(e, NULL, 0) : dflt;
}

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static void sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

static int kind_of(const char *s)
{
    int i;
    for (i = 0; i < PK_N; i++) if (!strcmp(s, KIND_NAME[i])) return i;
    return -1;
}

/* The non-poisoning program to diff a poisoning one against. For the two int32 writers
 * and for fp16fc the counterpart is the SAME geometry at int8, so the delta is the
 * output width and nothing else; fp16d's differs in the input-channel count each
 * emitter contracts, which is not a free axis on either side. -1 = no counterpart. */
static const int COUNTERPART[PK_N] = {
    /* int8fc */ -1, /* fp16fc */ PK_INT8FC, /* int8d */ -1, /* fp16d */ PK_INT8D,
    /* i32 */ PK_INT8D, /* i32w */ PK_INT8D, /* dw */ -1
};

/* Per-run addresses. Two programs here hold two sets of BOs, so these differ for a
 * reason that has nothing to do with the encoding. */
static int is_addr_reg(unsigned r)
{
    return r == 0x1088 || r == 0x1110 || r == 0x4018 || r == 0x5020 || r == 0x5024;
}

/* Emit `pr`'s program under whatever ROCKET_RK3576_SET is set right now, and stage it
 * in the regcmd BO. Split out from prog_init so `scan` can re-emit one program with an
 * override without disturbing the canary's. */
static int prog_emit(int fd, struct prog *pr)
{
    int rc;
    memset(pr->ops, 0, sizeof pr->ops);
    pr->p.tasks = pr->ops;
    pr->p.task_count = 0;
    switch (pr->kind) {
    case PK_I32:  rc = gen_conv2d_int8_rk3576_i32out(&pr->p); break;
    case PK_I32W: rc = gen_conv2d_int8_rk3576_i32out_wide(&pr->p); break;
    case PK_DW:   rc = gen_conv2d_dw_int8_rk3576(&pr->p); break;
    case PK_FP16FC:
    case PK_FP16D: rc = gen_conv2d_fp16_rk3576(&pr->p); break;
    default:      rc = gen_conv2d_int8_rk3576(&pr->p); break;
    }
    if (rc != 0) return -1;
    rocket_bo_prep(fd, &pr->r, 1, 0);
    memcpy(pr->r.ptr, pr->ops, pr->p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &pr->r);
    return 0;
}

static int prog_init(int fd, struct prog *pr, int kind)
{
    const struct geom *g = &KIND_GEOM[kind];
    int fp16 = (kind == PK_FP16FC || kind == PK_FP16D);
    int argb = (kind == PK_INT8FC || kind == PK_FP16FC);
    int dw   = (kind == PK_DW);
    unsigned esz = fp16 ? 2u : 1u;
    unsigned ow = (g->iw + g->pad - g->k) / g->stride + 1;
    unsigned oh = (g->ih + g->pad - g->k) / g->stride + 1;
    /* The fp16 direct emitter contracts EXACTLY sixteen input channels and refuses any
     * other count, so its register count is the raw one — the 32-channel MAC group is
     * an int8 rule. */
    unsigned icreg = (dw || kind == PK_FP16D) ? g->ic : rocket_rk3576_pad_ic(g->ic);
    unsigned ocreg = dw ? g->oc : rocket_rk3576_pad_oc(g->oc);
    unsigned surf_elems = rocket_rk3576_out_surf_elems(ow, oh, dw);
    size_t in_bytes, w_bytes, coeff, obase;
    int32_t *bias;
    unsigned i;

    memset(pr, 0, sizeof *pr);
    pr->kind = kind;

    if (argb) in_bytes = (size_t)g->ih * g->iw * g->ic * esz;
    else      in_bytes = (size_t)((icreg + 31) / 32) * 32 * g->ih * g->iw * esz;

    if (argb && fp16)      w_bytes = rocket_rk3576_weight_argb_fp16_bytes(g->oc, g->k, g->k);
    else if (argb)         w_bytes = (size_t)g->oc * g->k * ((4u * g->k + 15u) & ~15u);
    else if (dw)           w_bytes = rocket_rk3576_weight_dw_bytes(icreg, g->k, g->k);
    else                   w_bytes = (size_t)((ocreg + 31) / 32) * ((icreg + 31) / 32)
                                     * 32 * 32 * g->k * g->k * esz;

    coeff  = dw ? rocket_rk3576_coeff_bytes_dw(ocreg) : rocket_rk3576_coeff_bytes(ocreg);
    obase  = (size_t)((ocreg + C2 - 1) / C2) * surf_elems * C2;
    /* Eight times the int8 surface. The int32 writers emit four bytes an element and
     * the fp16 ones two, and a program writing past what this run sized for must land
     * inside the BO rather than fault — an allocation is cheap and a wedged IOMMU is
     * not. The whole allocation is what "did it write" scans. */
    pr->obytes = obase * 8;

    if (rocket_bo_alloc(fd, in_bytes, &pr->in) < 0) return -1;
    if (rocket_bo_alloc(fd, w_bytes,  &pr->w)  < 0) return -1;
    if (rocket_bo_alloc(fd, coeff,    &pr->b)  < 0) return -1;
    if (rocket_bo_alloc(fd, pr->obytes, &pr->o) < 0) return -1;
    if (rocket_bo_alloc(fd, sizeof pr->ops, &pr->r) < 0) return -1;

    /* Operands that cannot produce a zero surface. The layouts are the real cubes'
     * only where it costs nothing to make them so: this probe scores whether the part
     * WROTE, never what it computed, and a wrong scatter is a wrong number rather than
     * an empty buffer. What does matter is that nothing is zero, so "wrote" and
     * "correctly wrote zeros" stay distinguishable. */
    rocket_bo_prep(fd, &pr->in, 1, 0);
    if (fp16) for (i = 0; i < in_bytes / 2; i++) ((uint16_t *)pr->in.ptr)[i] = 0x3C00;
    else      memset(pr->in.ptr, 3, in_bytes);
    rocket_bo_fini(fd, &pr->in);

    rocket_bo_prep(fd, &pr->w, 1, 0);
    if (fp16) for (i = 0; i < w_bytes / 2; i++) ((uint16_t *)pr->w.ptr)[i] = 0x3C00;
    else      memset(pr->w.ptr, 1, w_bytes);
    rocket_bo_fini(fd, &pr->w);

    bias = calloc(ocreg, sizeof *bias);
    if (!bias) return -1;
    for (i = 0; i < ocreg; i++) bias[i] = (int32_t)i + 1;
    rocket_bo_prep(fd, &pr->b, 1, 0);
    if (dw)
        rocket_rk3576_pack_coeff_dw_prec(pr->b.ptr, coeff, bias, ocreg, precision_int8);
    else
        rocket_rk3576_pack_coeff_prec(pr->b.ptr, coeff, bias, ocreg,
                                      fp16 ? precision_float16 : precision_int8);
    rocket_bo_fini(fd, &pr->b);
    free(bias);

    pr->p.ic = (uint16_t)(argb ? g->ic : icreg);
    pr->p.oc = (uint16_t)ocreg;
    pr->p.ih = (uint16_t)g->ih; pr->p.iw = (uint16_t)g->iw;
    pr->p.oh = (uint16_t)oh;    pr->p.ow = (uint16_t)ow;
    pr->p.kh = (uint16_t)g->k;  pr->p.kw = (uint16_t)g->k;
    pr->p.stride_y = (uint8_t)g->stride; pr->p.stride_x = (uint8_t)g->stride;
    pr->p.pad_top = (uint8_t)g->pad; pr->p.pad_left = (uint8_t)g->pad;
    pr->p.int8_out = (uint8_t)(fp16 ? 0 : 1);
    pr->p.in_scale = 1.0f; pr->p.w_scale = 1.0f; pr->p.out_scale = 64.0f;
    pr->p.input_zero_point = 0; pr->p.output_zero_point = 0; pr->p.weight_zero_point = 0;
    pr->p.ih_full = (uint16_t)g->ih; pr->p.oh_full = (uint16_t)oh;
    pr->p.input_dma   = (uint32_t)pr->in.dma_address;
    pr->p.weights_dma = (uint32_t)pr->w.dma_address;
    pr->p.bias_dma    = (uint32_t)pr->b.dma_address;
    pr->p.output_dma  = (uint32_t)pr->o.dma_address;

    pr->in_h[0] = pr->in.handle; pr->in_h[1] = pr->w.handle;
    pr->in_h[2] = pr->b.handle;  pr->in_h[3] = pr->r.handle;
    pr->out_h[0] = pr->o.handle;

    if (prog_emit(fd, pr) != 0) return -1;
    pr->live = 1;
    return 0;
}

/* The fill is BRACKETED. A bare memset of an output BO leaves dirty cache lines that
 * race the DPU's DMA, and the writeback lands on top of the result — which reads as a
 * job that wrote nothing. */
static void prog_stamp(int fd, struct prog *pr)
{
    rocket_bo_prep(fd, &pr->o, 1, 0);
    memset(pr->o.ptr, SENTINEL, pr->obytes);
    rocket_bo_fini(fd, &pr->o);
}

static int prog_submit(int fd, struct prog *pr)
{
    return rocket_submit_matmul(fd, &pr->r, pr->p.task_count,
                                pr->in_h, 4, pr->out_h, 1, 2000);
}

/* `wait_us`, when asked for, is the time the PREP_BO blocked — which is where a submit
 * actually costs anything. The SUBMIT ioctl returns before the job runs, so timing it
 * measures the ioctl and not the part. */
static int prog_wrote_t(int fd, struct prog *pr, uint64_t *wait_us)
{
    const uint8_t *o;
    size_t i;
    int hit = 0;
    uint64_t t0 = now_us();
    if (rocket_bo_prep(fd, &pr->o, 0, 2000000000ull) < 0) return -1;
    if (wait_us) *wait_us = now_us() - t0;
    o = (const uint8_t *)pr->o.ptr;
    for (i = 0; i < pr->obytes; i++) if (o[i] != SENTINEL) { hit = 1; break; }
    rocket_bo_fini(fd, &pr->o);
    return hit;
}

static int prog_wrote(int fd, struct prog *pr)
{
    return prog_wrote_t(fd, pr, NULL);
}

/* The driver's autosuspend delay, which is what every idle here is really measured in.
 * -1 if it cannot be read. */
static int autosuspend_ms(void)
{
    static const char *const PATHS[] = {
        "/sys/bus/platform/drivers/rocket/27700000.npu/power/autosuspend_delay_ms",
        "/sys/bus/platform/drivers/rocket/27f00000.npu/power/autosuspend_delay_ms",
    };
    unsigned i;
    for (i = 0; i < sizeof PATHS / sizeof PATHS[0]; i++) {
        FILE *fp = fopen(PATHS[i], "r");
        long v;
        if (!fp) continue;
        if (fscanf(fp, "%ld", &v) == 1 && v >= 0 && v < 10000) { fclose(fp); return (int)v; }
        fclose(fp);
    }
    return -1;
}

/* ---------------------------------------------------------------------------
 * The register sets, and the subset machinery the joint sweep runs on.
 *
 * A leave-one-out asks "is any ONE register sufficient", and every negative it
 * returns is compatible with a condition of two. The sweeps below ask about
 * SUBSETS, which is the question a joint condition answers. Both directions
 * share one predicate: apply a subset to one program, and score the CANARY.
 * ------------------------------------------------------------------------- */
#define MAXSPEC 96

struct regval { unsigned reg; uint32_t val; };

/* Every register write a program carries, minus the per-run addresses and the
 * OP_NONE padding. */
static int prog_regs(const struct prog *pr, struct regval *out, int max)
{
    int i, n = 0;
    for (i = 0; i < (int)pr->p.task_count && n < max; i++) {
        unsigned reg = (unsigned)(pr->ops[i] & 0xFFFF);
        if ((unsigned)(pr->ops[i] >> 48) == 0) continue;
        if (is_addr_reg(reg)) continue;
        out[n].reg = reg;
        out[n].val = (uint32_t)(pr->ops[i] >> 16);
        n++;
    }
    return n;
}

static const struct regval *rv_find(const struct regval *v, int n, unsigned reg)
{
    int i;
    for (i = 0; i < n; i++) if (v[i].reg == reg) return &v[i];
    return NULL;
}

/* "0x4010=0x00000001,0x4038=..." for the subset picked out by `take`. */
static void spec_join(char *dst, size_t cap, const struct regval *v, int n,
                      const unsigned char *take)
{
    size_t used = 0;
    int i;
    dst[0] = '\0';
    for (i = 0; i < n; i++) {
        int w;
        if (!take[i]) continue;
        w = snprintf(dst + used, cap - used, "%s0x%04x=0x%08x",
                     used ? "," : "", v[i].reg, v[i].val);
        if (w < 0 || (size_t)w >= cap - used) return;
        used += (size_t)w;
    }
}

static void spec_print(const struct regval *v, int n, const unsigned char *take)
{
    int i, first = 1;
    for (i = 0; i < n; i++) {
        if (!take[i]) continue;
        printf("%s0x%04x", first ? "" : " ", v[i].reg);
        first = 0;
    }
    if (first) printf("(none)");
}

static int take_count(const unsigned char *take, int n)
{
    int i, c = 0;
    for (i = 0; i < n; i++) c += take[i] != 0;
    return c;
}

static int g_recover = 250;

/* One A-then-B round. The recovery idle comes FIRST, so A always runs on an unpoisoned
 * part and the round measures what A does to B rather than what the last round did to
 * A. Both surfaces are stamped before A runs, so the only thing between the two
 * submits is the gap. */
static int round_once(int fd, struct prog *a, struct prog *b, int gap,
                      int *a_wrote, int *b_wrote)
{
    sleep_ms(g_recover);
    prog_stamp(fd, a);
    prog_stamp(fd, b);
    if (prog_submit(fd, a) != 0) return -1;
    sleep_ms(gap);
    if (prog_submit(fd, b) != 0) return -1;
    *a_wrote = prog_wrote(fd, a);
    *b_wrote = prog_wrote(fd, b);
    return (*a_wrote < 0 || *b_wrote < 0) ? -1 : 0;
}

static int measure(int fd, struct prog *a, struct prog *b, int gap, int reps,
                   int *na, int *nb)
{
    int r, aw, bw;
    *na = *nb = 0;
    for (r = 0; r < reps; r++) {
        if (round_once(fd, a, b, gap, &aw, &bw) != 0) return -1;
        *na += aw; *nb += bw;
    }
    return 0;
}

/* The joint-sweep predicate. `which` selects the program the subset is applied to:
 * A for the heal direction (does restoring these stop the poisoning), the canary for
 * the xadd direction (does writing these clear it). Scoring is always the canary. */
enum { APPLY_SET_A, APPLY_ADD_B };

static int predicate(int fd, struct prog *a, struct prog *b, int apply,
                     const char *spec, int gap, int reps, int *na, int *nb)
{
    struct prog *tgt = (apply == APPLY_ADD_B) ? b : a;
    const char *var = (apply == APPLY_ADD_B) ? "ROCKET_RK3576_ADD" : "ROCKET_RK3576_SET";
    int rc;

    setenv(var, spec, 1);
    rc = prog_emit(fd, tgt);
    unsetenv(var);
    if (rc != 0) return -1;

    rc = measure(fd, a, b, gap, reps, na, nb);

    /* Leave both programs holding their unmodified encodings, so the next call in the
     * sweep starts from the base rather than from the last subset tried. */
    if (prog_emit(fd, tgt) != 0) return -1;
    return rc;
}

/* True when the subset stopped the poisoning WITHOUT merely killing the program it was
 * applied to. A dead program poisons nothing, and that reading is the trap this whole
 * probe exists to avoid. */
static int pred_ok(int apply, int reps, int na, int nb)
{
    if (nb != reps) return 0;
    return (apply == APPLY_ADD_B) ? (na == 0 || na > 0) : (na > 0);
}

/* Delta debugging (ddmin) over the subset that satisfies the predicate. A joint
 * condition of two or three registers is invisible to a leave-one-out and is found here
 * in a few rounds: the search splits the set, tries each part and each complement, and
 * narrows to whichever still satisfies it. */
static int ddmin(int fd, struct prog *a, struct prog *b, int apply,
                 const struct regval *v, int n, unsigned char *take,
                 int gap, int reps)
{
    char spec[MAXSPEC * 20];
    unsigned char trial[MAXSPEC];
    int gran = 2, na, nb;

    while (take_count(take, n) > 1) {
        int cnt = take_count(take, n), progress = 0, part;
        int chunk = (cnt + gran - 1) / gran;
        if (chunk < 1) break;

        /* Each part first, then each complement — a minimal set that straddles a split
         * is caught by the complement pass. */
        for (part = 0; part < gran * 2 && !progress; part++) {
            int want = part % gran, comp = part >= gran, seen = 0, i;
            for (i = 0; i < n; i++) {
                if (!take[i]) { trial[i] = 0; continue; }
                trial[i] = (unsigned char)(((seen / chunk) == want) ^ (comp ? 1 : 0));
                seen++;
            }
            if (take_count(trial, n) == 0 || take_count(trial, n) == cnt) continue;
            spec_join(spec, sizeof spec, v, n, trial);
            if (predicate(fd, a, b, apply, spec, gap, reps, &na, &nb) != 0) return -1;
            printf("    try %2d reg: ", take_count(trial, n));
            spec_print(v, n, trial);
            printf("  -> canary %d/%d, program %d/%d%s\n", nb, reps, na, reps,
                   pred_ok(apply, reps, na, nb) ? "   KEEP" : "");
            fflush(stdout);
            if (pred_ok(apply, reps, na, nb)) {
                memcpy(take, trial, (size_t)n);
                gran = comp ? (gran > 2 ? gran - 1 : 2) : 2;
                progress = 1;
            }
        }
        if (progress) continue;
        if (gran >= cnt) break;
        gran = gran * 2 > cnt ? cnt : gran * 2;
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "scope";
    int gap    = env_int("ROCKET_PP_GAP_MS", 0);
    int reps   = env_int("ROCKET_PP_REPS", 20);
    int settle = env_int("ROCKET_PP_SETTLE", 6);
    /* TWO independent instances of every kind. A and B are drawn from different sets so
     * the A==B cells are a real pair rather than one program reading its own surface
     * back — with a shared BO the diagonal reports "B wrote" whenever A did, which is
     * exactly the cell that says whether a program poisons a repeat of itself. */
    struct prog progs[PK_N], progs_b[PK_N];
    int fd, i, rc = 0;

    {
        int d = autosuspend_ms();
        g_recover = env_int("ROCKET_PP_RECOVER", d >= 0 ? 4 * d + 50 : 250);
    }

    fd = rocket_open();
    if (fd < 0) { printf("rk3576_poison_probe: no NPU — skipping\n"); return 2; }

    /* The hazard outlives the process. Without this the first measurement reports the
     * previous run's state, which is how a no-op override once read as fatal. */
    printf("settling %d s (the hazard crosses processes)\n", settle);
    sleep_ms(settle * 1000);

    for (i = 0; i < PK_N; i++) {
        if (prog_init(fd, &progs[i], i) != 0)
            printf("  %-7s: could not be built\n", KIND_NAME[i]);
        if (prog_init(fd, &progs_b[i], i) != 0) progs_b[i].live = 0;
    }

    if (!strcmp(mode, "scope")) {
        int a, b;
        printf("\nA-poisons-B, gap %d ms, %d rounds, %d ms recovery before each. Each cell "
               "is how many of\nthe %d B submits wrote; the bracketed number is how many "
               "of A's own did — a cell\nwhere A itself is short of %d says nothing about "
               "B.\n\n", gap, reps, g_recover, reps, reps);
        printf("%-8s", "A \\ B");
        for (b = 0; b < PK_N; b++) printf("%12s", KIND_NAME[b]);
        printf("\n");
        for (a = 0; a < PK_N; a++) {
            if (!progs[a].live) continue;
            printf("%-8s", KIND_NAME[a]);
            for (b = 0; b < PK_N; b++) {
                int na, nb;
                if (!progs_b[b].live) { printf("%12s", "-"); continue; }
                if (measure(fd, &progs[a], &progs_b[b], gap, reps, &na, &nb) != 0) {
                    printf("%12s", "ERR"); rc = 1; continue;
                }
                printf("%9d[%2d]", nb, na);
                fflush(stdout);
            }
            printf("\n");
        }
    } else if (!strcmp(mode, "pair") || !strcmp(mode, "scan")) {
        int ka = argc > 2 ? kind_of(argv[2]) : -1;
        int kb = argc > 3 ? kind_of(argv[3]) : -1;
        int na, nb;
        if (ka < 0 || kb < 0) { printf("usage: %s %s <A> <B>\n", argv[0], mode); rc = 1; goto out; }
        if (!progs[ka].live || !progs_b[kb].live) { printf("a program is not live\n"); rc = 1; goto out; }

        if (measure(fd, &progs[ka], &progs_b[kb], gap, reps, &na, &nb) != 0) { rc = 1; goto out; }
        printf("\nbase  %-7s -> %-7s : B wrote %d/%d, A wrote %d/%d  (gap %d ms)\n",
               KIND_NAME[ka], KIND_NAME[kb], nb, reps, na, reps, gap);

        if (!strcmp(mode, "scan")) {
            const char *regs = getenv("ROCKET_PP_REGS");
            char specs[64][32];
            int nspec = 0, s;

            if (regs && *regs) {
                char *list = strdup(regs), *tok, *save = NULL;
                if (!list) { rc = 1; goto out; }
                for (tok = strtok_r(list, ",", &save); tok && nspec < 64;
                     tok = strtok_r(NULL, ",", &save))
                    snprintf(specs[nspec++], sizeof specs[0], "%s", tok);
                free(list);
            } else {
                /* Derive the delta rather than transcribe it. A hand-copied list is
                 * pinned to the geometry it was dumped at, and this probe's is not
                 * that geometry — the override then reads as a register that does
                 * nothing when it is really a value from another shape. */
                int cp = COUNTERPART[ka];
                unsigned i8, i16;
                if (cp < 0 || !progs[cp].live) {
                    printf("scan: %s has no clean counterpart to diff against; set "
                           "ROCKET_PP_REGS\n", KIND_NAME[ka]);
                    rc = 1; goto out;
                }
                printf("\ndelta %s -> %s (the override value is %s's):\n",
                       KIND_NAME[ka], KIND_NAME[cp], KIND_NAME[cp]);
                for (i16 = 0; i16 < progs[ka].p.task_count && nspec < 64; i16++) {
                    unsigned reg = (unsigned)(progs[ka].ops[i16] & 0xFFFF);
                    uint32_t va = (uint32_t)(progs[ka].ops[i16] >> 16);
                    if ((unsigned)(progs[ka].ops[i16] >> 48) == 0) continue;
                    if (is_addr_reg(reg)) continue;
                    for (i8 = 0; i8 < progs[cp].p.task_count; i8++) {
                        uint32_t vb;
                        if ((unsigned)(progs[cp].ops[i8] & 0xFFFF) != reg) continue;
                        if ((unsigned)(progs[cp].ops[i8] >> 48) == 0) continue;
                        vb = (uint32_t)(progs[cp].ops[i8] >> 16);
                        if (vb != va) {
                            printf("    0x%04x  %-7s %08x   %-7s %08x\n", reg,
                                   KIND_NAME[ka], va, KIND_NAME[cp], vb);
                            snprintf(specs[nspec++], sizeof specs[0], "0x%04x=0x%08x",
                                     reg, vb);
                        }
                        break;
                    }
                }
                printf("  %d registers differ\n", nspec);
            }

            printf("\nleave-one-out on %s. A register that stops the poisoning shows B at "
                   "%d/%d;\none that merely kills A shows A at 0/%d beside it.\n\n",
                   KIND_NAME[ka], reps, reps, reps);
            printf("  %-24s %-14s %s\n", "override on A", "B wrote", "A wrote");
            for (s = 0; s < nspec; s++) {
                setenv("ROCKET_RK3576_SET", specs[s], 1);
                if (prog_emit(fd, &progs[ka]) != 0) {
                    printf("  %-24s emit failed\n", specs[s]); unsetenv("ROCKET_RK3576_SET"); continue;
                }
                unsetenv("ROCKET_RK3576_SET");
                if (measure(fd, &progs[ka], &progs_b[kb], gap, reps, &na, &nb) != 0) {
                    printf("  %-24s ERR\n", specs[s]); rc = 1; continue;
                }
                printf("  %-24s %5d/%-8d %5d/%d%s\n", specs[s], nb, reps, na, reps,
                       (nb == reps && na > 0) ? "   <-- STOPS THE POISONING" :
                       (nb == reps)           ? "   (but A is dead)" : "");
                fflush(stdout);
            }
            /* Re-emit the base program, so a caller reading the exit state sees the
             * unmodified one. */
            if (prog_emit(fd, &progs[ka]) != 0) rc = 1;
        }
    } else if (!strcmp(mode, "chain")) {
        /* Does one poisoned submit ABSORB it? If the hazard reaches exactly one
         * successor, a sacrificial job is the whole fix and costs a submit rather than
         * a power cycle. If it persists until the domain cycles, no amount of
         * submitting clears it and the guard has to stay an idle. */
        int ka = argc > 2 ? kind_of(argv[2]) : -1;
        int kb = argc > 3 ? kind_of(argv[3]) : -1;
        int depth = env_int("ROCKET_PP_DEPTH", 6);
        int *hit, r, d;
        if (ka < 0 || kb < 0) { printf("usage: %s chain <A> <B>\n", argv[0]); rc = 1; goto out; }
        if (!progs[ka].live || !progs_b[kb].live) { printf("a program is not live\n"); rc = 1; goto out; }
        hit = calloc((size_t)depth + 1, sizeof *hit);
        if (!hit) { rc = 1; goto out; }
        for (r = 0; r < reps; r++) {
            sleep_ms(g_recover);
            prog_stamp(fd, &progs[ka]);
            if (prog_submit(fd, &progs[ka]) != 0) { rc = 1; break; }
            hit[0] += (prog_wrote(fd, &progs[ka]) > 0);
            for (d = 0; d < depth; d++) {
                prog_stamp(fd, &progs_b[kb]);
                sleep_ms(gap);
                if (prog_submit(fd, &progs_b[kb]) != 0) { rc = 1; break; }
                hit[d + 1] += (prog_wrote(fd, &progs_b[kb]) > 0);
            }
        }
        printf("\n%s then %s x%d, gap %d ms, %d runs. Position 0 is A's own submit.\n"
               "If the hazard reaches exactly one successor, position 1 is 0 and every\n"
               "later position is %d.\n\n", KIND_NAME[ka], KIND_NAME[kb], depth, gap,
               reps, reps);
        for (d = 0; d <= depth; d++)
            printf("  position %d: wrote %d/%d\n", d, hit[d], reps);
        free(hit);
    } else if (!strcmp(mode, "regs")) {
        /* What a leave-one-out could never see. It diffs the registers both programs
         * write; a register only the poisoning one writes has no counterpart value to
         * restore, so it is skipped — and this part does NOT clear the register file
         * between jobs, which makes exactly that set the interesting one. */
        struct regval va[MAXSPEC], vb[MAXSPEC];
        int ka = argc > 2 ? kind_of(argv[2]) : -1;
        int kb = argc > 3 ? kind_of(argv[3]) : -1;
        int nva, nvb, i, nonly = 0, bonly = 0, ndiff = 0;
        if (ka < 0 || kb < 0) { printf("usage: %s regs <A> <B>\n", argv[0]); rc = 1; goto out; }
        if (!progs[ka].live || !progs[kb].live) { printf("a program is not live\n"); rc = 1; goto out; }
        nva = prog_regs(&progs[ka], va, MAXSPEC);
        nvb = prog_regs(&progs[kb], vb, MAXSPEC);
        printf("\n%s writes %d registers, %s writes %d.\n\n",
               KIND_NAME[ka], nva, KIND_NAME[kb], nvb);
        printf("  written by %s ONLY (no counterpart value; invisible to `scan`):\n", KIND_NAME[ka]);
        for (i = 0; i < nva; i++)
            if (!rv_find(vb, nvb, va[i].reg)) { printf("    0x%04x = %08x\n", va[i].reg, va[i].val); nonly++; }
        if (!nonly) printf("    (none)\n");
        printf("  written by %s ONLY:\n", KIND_NAME[kb]);
        for (i = 0; i < nvb; i++)
            if (!rv_find(va, nva, vb[i].reg)) { printf("    0x%04x = %08x\n", vb[i].reg, vb[i].val); bonly++; }
        if (!bonly) printf("    (none)\n");
        printf("  written by both, DIFFERENT value:\n");
        for (i = 0; i < nva; i++) {
            const struct regval *m = rv_find(vb, nvb, va[i].reg);
            if (m && m->val != va[i].val) {
                printf("    0x%04x  %-7s %08x   %-7s %08x\n", va[i].reg,
                       KIND_NAME[ka], va[i].val, KIND_NAME[kb], m->val);
                ndiff++;
            }
        }
        if (!ndiff) printf("    (none)\n");
        printf("\n  %d only-A, %d only-B, %d differing\n", nonly, bonly, ndiff);
    } else if (!strcmp(mode, "heal")) {
        /* The JOINT sweep. First question is one measurement: does the WHOLE counterpart
         * delta stop the poisoning? If it does not, no subset of it can, and the
         * mechanism is outside the registers the two programs share. If it does, ddmin
         * narrows it to the set that a one-at-a-time sweep is blind to. */
        struct regval va[MAXSPEC], vb[MAXSPEC], sel[MAXSPEC];
        unsigned char take[MAXSPEC];
        char spec[MAXSPEC * 20];
        int ka = argc > 2 ? kind_of(argv[2]) : -1;
        int kb = argc > 3 ? kind_of(argv[3]) : -1;
        int cp, nva, nvb, nsel = 0, i, na, nb;
        if (ka < 0 || kb < 0) { printf("usage: %s heal <A> <B>\n", argv[0]); rc = 1; goto out; }
        cp = COUNTERPART[ka];
        if (cp < 0 || !progs[cp].live) { printf("heal: %s has no clean counterpart\n", KIND_NAME[ka]); rc = 1; goto out; }
        nva = prog_regs(&progs[ka], va, MAXSPEC);
        nvb = prog_regs(&progs[cp], vb, MAXSPEC);
        for (i = 0; i < nva && nsel < MAXSPEC; i++) {
            const struct regval *m = rv_find(vb, nvb, va[i].reg);
            if (m && m->val != va[i].val) { sel[nsel].reg = va[i].reg; sel[nsel].val = m->val; nsel++; }
            else if (!m && env_int("ROCKET_PP_XREG", 0)) { sel[nsel].reg = va[i].reg; sel[nsel].val = 0; nsel++; }
        }
        memset(take, 1, (size_t)nsel);
        printf("\nheal %s -> %s, scoring %s. %d registers offered (the value is %s's;\n"
               "an only-in-%s register is offered at 0 when ROCKET_PP_XREG is set).\n\n",
               KIND_NAME[ka], KIND_NAME[cp], KIND_NAME[kb], nsel, KIND_NAME[cp], KIND_NAME[ka]);

        if (measure(fd, &progs[ka], &progs_b[kb], gap, reps, &na, &nb) != 0) { rc = 1; goto out; }
        printf("  base (no override)        canary %d/%d, %s %d/%d\n",
               nb, reps, KIND_NAME[ka], na, reps);
        spec_join(spec, sizeof spec, sel, nsel, take);
        if (predicate(fd, &progs[ka], &progs_b[kb], APPLY_SET_A, spec, gap, reps, &na, &nb) != 0) { rc = 1; goto out; }
        printf("  whole delta (%2d reg)      canary %d/%d, %s %d/%d  %s\n\n", nsel,
               nb, reps, KIND_NAME[ka], na, reps,
               pred_ok(APPLY_SET_A, reps, na, nb) ? "-- the mechanism IS in these registers"
                                                  : "-- NOT in these registers");
        if (!pred_ok(APPLY_SET_A, reps, na, nb)) goto out;
        printf("  minimising:\n");
        if (ddmin(fd, &progs[ka], &progs_b[kb], APPLY_SET_A, sel, nsel, take, gap, reps) != 0) { rc = 1; goto out; }
        printf("\n  MINIMAL SET (%d): ", take_count(take, nsel));
        spec_print(sel, nsel, take);
        printf("\n");
        for (i = 0; i < nsel; i++)
            if (take[i]) printf("    0x%04x: %s %08x -> %s %08x\n", sel[i].reg,
                                KIND_NAME[ka], rv_find(va, nva, sel[i].reg)->val,
                                KIND_NAME[cp], sel[i].val);
    } else if (!strcmp(mode, "xadd")) {
        /* The shippable question, and a different one from `heal`: not "what makes A
         * stop poisoning" — A's output width is not a free axis on the fp16 path — but
         * "can the NEXT job write its way out of it". A register write costs nothing
         * beside a power cycle, so a set that clears it here replaces the whole guard. */
        struct regval va[MAXSPEC], vb[MAXSPEC], sel[MAXSPEC];
        unsigned char take[MAXSPEC];
        char spec[MAXSPEC * 20];
        const char *explicit_add = getenv("ROCKET_PP_ADD");
        int ka = argc > 2 ? kind_of(argv[2]) : -1;
        int kb = argc > 3 ? kind_of(argv[3]) : -1;
        int nva, nvb, nsel = 0, i, na, nb;
        if (ka < 0 || kb < 0) { printf("usage: %s xadd <A> <B>\n", argv[0]); rc = 1; goto out; }
        if (!progs[ka].live || !progs_b[kb].live) { printf("a program is not live\n"); rc = 1; goto out; }
        nva = prog_regs(&progs[ka], va, MAXSPEC);
        nvb = prog_regs(&progs_b[kb], vb, MAXSPEC);
        if (explicit_add && *explicit_add) {
            char *list = strdup(explicit_add), *tok, *save = NULL;
            if (!list) { rc = 1; goto out; }
            for (tok = strtok_r(list, ",", &save); tok && nsel < MAXSPEC;
                 tok = strtok_r(NULL, ",", &save)) {
                char *eq = strchr(tok, '=');
                if (!eq) continue;
                *eq = '\0';
                sel[nsel].reg = (unsigned)strtoul(tok, NULL, 0);
                sel[nsel].val = (uint32_t)strtoul(eq + 1, NULL, 0);
                nsel++;
            }
            free(list);
        } else {
            /* A's exclusive registers, at 0. These are precisely the ones the canary
             * leaves holding whatever the last job put there. */
            for (i = 0; i < nva && nsel < MAXSPEC; i++)
                if (!rv_find(vb, nvb, va[i].reg)) { sel[nsel].reg = va[i].reg; sel[nsel].val = 0; nsel++; }
        }
        memset(take, 1, (size_t)nsel);
        printf("\nxadd: %s poisons, then the canary %s runs with %d extra register\n"
               "write%s. If the canary writes, the hazard is a register the canary was\n"
               "not writing, and the guard becomes those writes instead of a power cycle.\n\n",
               KIND_NAME[ka], KIND_NAME[kb], nsel, nsel == 1 ? "" : "s");
        if (nsel == 0) { printf("  nothing to add\n"); goto out; }
        for (i = 0; i < nsel; i++) printf("    0x%04x = %08x\n", sel[i].reg, sel[i].val);

        if (measure(fd, &progs[ka], &progs_b[kb], gap, reps, &na, &nb) != 0) { rc = 1; goto out; }
        printf("\n  base (canary unchanged)   canary %d/%d, %s %d/%d\n",
               nb, reps, KIND_NAME[ka], na, reps);
        spec_join(spec, sizeof spec, sel, nsel, take);
        if (predicate(fd, &progs[ka], &progs_b[kb], APPLY_ADD_B, spec, gap, reps, &na, &nb) != 0) { rc = 1; goto out; }
        printf("  whole add set (%2d reg)    canary %d/%d, %s %d/%d  %s\n\n", nsel,
               nb, reps, KIND_NAME[ka], na, reps,
               (nb == reps) ? "-- the canary WRITES ITS WAY OUT" : "-- no");
        if (nb != reps) goto out;
        printf("  minimising:\n");
        if (ddmin(fd, &progs[ka], &progs_b[kb], APPLY_ADD_B, sel, nsel, take, gap, reps) != 0) { rc = 1; goto out; }
        printf("\n  MINIMAL SET (%d): ", take_count(take, nsel));
        spec_print(sel, nsel, take);
        printf("\n");
    } else if (!strcmp(mode, "bits")) {
        /* Exhaustive over the BITS of a short register list, once `heal` has narrowed
         * the joint condition to one. A truth table over every combination says which
         * bits are load-bearing and which merely ride along, where a minimal set only
         * says "these registers together" — and the difference decides whether an fp16
         * program can be emitted that does not poison. */
        struct regval va[MAXSPEC], vb[MAXSPEC];
        struct { unsigned reg; int bit; } bits[16];
        char spec[MAXSPEC * 20];
        int ka = argc > 2 ? kind_of(argv[2]) : -1;
        int kb = argc > 3 ? kind_of(argv[3]) : -1;
        const char *reglist = argc > 4 ? argv[4] : NULL;
        int cp, nva, nvb, nbits = 0, i, na, nb;
        unsigned long combo, ncombo;
        char *list, *tok, *save = NULL;
        if (ka < 0 || kb < 0 || !reglist) {
            printf("usage: %s bits <A> <B> 0x4038,0x4050\n", argv[0]); rc = 1; goto out;
        }
        cp = COUNTERPART[ka];
        if (cp < 0 || !progs[cp].live) { printf("bits: %s has no clean counterpart\n", KIND_NAME[ka]); rc = 1; goto out; }
        nva = prog_regs(&progs[ka], va, MAXSPEC);
        nvb = prog_regs(&progs[cp], vb, MAXSPEC);
        list = strdup(reglist);
        if (!list) { rc = 1; goto out; }
        for (tok = strtok_r(list, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
            unsigned reg = (unsigned)strtoul(tok, NULL, 0);
            const struct regval *ra = rv_find(va, nva, reg), *rb = rv_find(vb, nvb, reg);
            uint32_t x;
            if (!ra || !rb) { printf("bits: 0x%04x is not written by both\n", reg); continue; }
            for (x = ra->val ^ rb->val, i = 0; i < 32 && nbits < 16; i++)
                if (x & (1u << i)) { bits[nbits].reg = reg; bits[nbits].bit = i; nbits++; }
        }
        free(list);
        if (nbits == 0 || nbits > 12) {
            printf("bits: %d differing bits is out of range (1..12)\n", nbits); rc = 1; goto out;
        }
        printf("\n%d differing bits over %s. A row is A's program with the listed bits\n"
               "taken from %s; `canary` is how many of %d %s submits then wrote.\n\n",
               nbits, reglist, KIND_NAME[cp], reps, KIND_NAME[kb]);
        for (i = 0; i < nbits; i++)
            printf("  bit %d = 0x%04x[%d]  %s %d -> %s %d\n", i, bits[i].reg, bits[i].bit,
                   KIND_NAME[ka], (rv_find(va, nva, bits[i].reg)->val >> bits[i].bit) & 1,
                   KIND_NAME[cp], (rv_find(vb, nvb, bits[i].reg)->val >> bits[i].bit) & 1);
        printf("\n  %-14s %-26s %-10s %s\n", "combination", "bits taken", "canary", "program");
        ncombo = 1ul << nbits;
        for (combo = 0; combo < ncombo; combo++) {
            struct regval put[8];
            unsigned char all[8];
            int nput = 0, j;
            char label[40];
            for (i = 0; i < nbits; i++) {
                const struct regval *ra, *rb;
                int slot = -1;
                if (!(combo & (1ul << i))) continue;
                for (j = 0; j < nput; j++) if (put[j].reg == bits[i].reg) slot = j;
                if (slot < 0) {
                    ra = rv_find(va, nva, bits[i].reg);
                    slot = nput++;
                    put[slot].reg = bits[i].reg;
                    put[slot].val = ra->val;
                }
                rb = rv_find(vb, nvb, bits[i].reg);
                put[slot].val = (put[slot].val & ~(1u << bits[i].bit)) |
                                (rb->val & (1u << bits[i].bit));
            }
            memset(all, 1, sizeof all);
            spec_join(spec, sizeof spec, put, nput, all);
            if (predicate(fd, &progs[ka], &progs_b[kb], APPLY_SET_A,
                          nput ? spec : "", gap, reps, &na, &nb) != 0) { rc = 1; goto out; }
            for (i = 0; i < nbits; i++) label[i] = (combo & (1ul << i)) ? '1' : '.';
            label[nbits] = '\0';
            printf("  %-14s %-26s %5d/%-4d %5d/%d%s\n", label,
                   nput ? spec : "(base)", nb, reps, na, reps,
                   (nb == reps && na > 0) ? "   CLEAN" : "");
            fflush(stdout);
        }
    } else if (!strcmp(mode, "time")) {
        /* The driver retires an RK3576 job on the DPU's own completion bits, and bounds
         * that wait by rocket.dpu_grace_us because not every task raises them. A program
         * that does not shows up here as a submit that costs the whole grace. Whether
         * that class is the SAME class as the poisoning one is one table away, and it
         * is the difference between "the DPU never finished" and "a register stuck". */
        int k, r;
        uint64_t t[64];
        if (reps > 64) reps = 64;
        printf("\nper-kind submit wall time, %d submits each, %d ms recovery before each.\n"
               "A kind whose median tracks rocket.dpu_grace_us raises no DPU completion.\n\n",
               reps, g_recover);
        printf("  %-8s %10s %10s %10s   %s\n", "kind", "min us", "median", "max", "wrote");
        for (k = 0; k < PK_N; k++) {
            int wrote = 0;
            if (!progs[k].live) continue;
            for (r = 0; r < reps; r++) {
                uint64_t t0, wait = 0;
                sleep_ms(g_recover);
                prog_stamp(fd, &progs[k]);
                t0 = now_us();
                if (prog_submit(fd, &progs[k]) != 0) { rc = 1; break; }
                wrote += (prog_wrote_t(fd, &progs[k], &wait) > 0);
                t[r] = now_us() - t0;
                (void)wait;
            }
            qsort(t, (size_t)reps, sizeof t[0], cmp_u64);
            printf("  %-8s %10llu %10llu %10llu   %d/%d\n", KIND_NAME[k],
                   (unsigned long long)t[0], (unsigned long long)t[reps / 2],
                   (unsigned long long)t[reps - 1], wrote, reps);
            fflush(stdout);
        }
    } else {
        printf("usage: %s scope | pair <A> <B> | scan <A> <B> | chain <A> <B>\n"
               "       %s regs <A> <B> | heal <A> <B> | xadd <A> <B> | time\n",
               argv[0], argv[0]);
        rc = 1;
    }

out:
    /* Leave the part unpoisoned: the hazard outlives this process, and the next thing
     * to run should not inherit it. */
    sleep_ms(300);
    rocket_close(fd);
    return rc;
}
