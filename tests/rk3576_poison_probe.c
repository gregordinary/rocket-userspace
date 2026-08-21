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
 *
 * Modes:
 *   rk3576_poison_probe scope        the full A-poisons-B matrix over the program kinds
 *   rk3576_poison_probe pair A B     one pair, at ROCKET_PP_GAP_MS
 *   rk3576_poison_probe scan A B     leave-one-out: apply each ROCKET_PP_REGS entry to
 *                                    A alone and re-measure B
 *   rk3576_poison_probe chain A B    how DEEP it goes: one A, then B over and over,
 *                                    reporting which position recovers
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

static int prog_wrote(int fd, struct prog *pr)
{
    const uint8_t *o;
    size_t i;
    int hit = 0;
    if (rocket_bo_prep(fd, &pr->o, 0, 2000000000ull) < 0) return -1;
    o = (const uint8_t *)pr->o.ptr;
    for (i = 0; i < pr->obytes; i++) if (o[i] != SENTINEL) { hit = 1; break; }
    rocket_bo_fini(fd, &pr->o);
    return hit;
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
    } else {
        printf("usage: %s scope | pair <A> <B> | scan <A> <B> | chain <A> <B>\n", argv[0]);
        rc = 1;
    }

out:
    /* Leave the part unpoisoned: the hazard outlives this process, and the next thing
     * to run should not inherit it. */
    sleep_ms(300);
    rocket_close(fd);
    return rc;
}
