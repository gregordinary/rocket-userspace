// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_add_probe.c — the DPU's elementwise stage on the RK3576, and its model.
 *
 * The program is transcribed from manufactured captures
 * (`tests/data/rk3576-vendor-capture/add/`, `mkadd.py` + `decode_add.py`): DPU +
 * DPU_RDMA only, 89 writes, PC_OPERATION_ENABLE 0x18. Five captures over two
 * geometries emit it identically apart from the plane, the channels and the two
 * converters. What no capture can carry is the three base addresses — the vendor
 * runtime patches those at load time — so the arithmetic is measured here, not read.
 *
 * WHAT IT COMPUTES, MEASURED: one operand, the EW cube, through its converter and the
 * OUT_CVT. `out = sat8(((ew + ew_offset) * ew_scale >> ew_shift) * out_scale >>
 * out_shift)`, exact over the shape table. THE SECOND OPERAND IS NOT REACHABLE THROUGH
 * THIS REGISTER SET, and the search is EXHAUSTIVE rather than a lead not yet chased:
 *
 *   - `joint` crosses every register the program leaves at zero with the whole output
 *     gain ladder, 17 bases x 32 rungs. Only 0x5024 does anything, and what it does is
 *     inject a constant — it is the DPU shift word, a live operand, not a spare.
 *     CROSSING THE AXES IS THE POINT: `where` swept placement at one gain and `mrdma`
 *     swept gain at one placement, and an operand that both sits somewhere unexpected
 *     AND enters the accumulator unscaled is invisible to either alone.
 *   - `unwritten` appends the operand's address at each register the program NEVER
 *     writes — the complement of every other sweep here, and not an empty set, because
 *     the register file is not cleared between jobs on this part. 29 candidates, two
 *     gains: three stop the write, none carries an operand.
 *   - `mrdma` runs the main DMA feed at every gain from 2^14 down to 2^-31. The program
 *     is configured to read one (0x400C bit 0 set = off-flying, 0x5044 bit 4 clear =
 *     MRDMA enabled, the NVDLA pair) and it contributes exactly zero at all 64 rungs.
 *   - `prim` and `comb` are JOINT grids — EW_CFG x BRDMA_CFG, and FEATURE_MODE x
 *     ERDMA_CFG driving the COMB_USE field the RK3588's own K-accumulation uses to
 *     combine two feeds. In both, only the captured word writes at all.
 *   - `acc` asks whether the write accumulates into its destination, with DPU 0x40C0
 *     swept over the captured word and 28 single-bit variants and each case run twice
 *     over differently pre-filled destinations. It always overwrites. SURFACE_ADD sits
 *     at that offset in the RK3588's map and is not an accumulate mode here.
 *   - `span` walks one non-zero atom over an operand buffer eight cubes long: the
 *     addressed cube's atoms map one to one onto the output and NOTHING outside it
 *     moves anything, so the two operands are not one allocation at a fixed offset
 *     either — which is how the RK3588's K-accumulation feeds its pair.
 *
 * What the captures say the vendor's program does have, which is what makes the negative
 * precise: `Sub` compiles to this same program with the operand converter's scale
 * NEGATED, in both operand orders, so the subtrahend is always the EW cube and the other
 * operand's weight is fixed at +1; and both operands share ONE quantization scale (two
 * graph inputs calibrated 64x apart still compile to a single converter).
 *
 * THE RESIDUAL ADD IS LOWERED ONTO THE CONVOLUTION DATAPATH INSTEAD, and it computes —
 * see tests/rk3576_residual_add.c.
 *
 * The modes:
 *   `probe`  does it write at all, with the bases programmed.
 *   `model`  the converter fields one at a time against the measured reference.
 *   `raw`    a small plane printed against its operands: the function, read off.
 *   `scan`   the OUT_CVT shift ladder, which is what measures the accumulator.
 *   `where`  the base-placement sweep above.
 *   `prim`   the joint mode-register grid above.
 *   `mrdma`  the OUT_CVT ladder with the primary as the only live operand — whether
 *            the main DMA feed is dead or merely enters the accumulator at a gain the
 *            earlier sweeps could not see.
 *   `acc`    does the write accumulate into the destination, with DPU 0x40C0 swept.
 *   `comb`   the joint FEATURE_MODE x ERDMA_CFG grid over the field the RK3588's own
 *            K-accumulation path uses to combine the two feeds.
 *   `joint`  every candidate base crossed with the output gain — the combination
 *            neither `where` nor `mrdma` reaches, since each holds the other fixed.
 *   `span`   a single non-zero atom walked over an operand buffer eight cubes long:
 *            what the program reads, rather than what it was handed.
 *   `unwritten` the operand address appended at each register the program leaves
 *            alone — the complement of every other sweep here, and not an empty set,
 *            since the register file is not cleared between jobs on this part.
 *   `gate`   the shape table against the measured model.
 *
 * TRAPS.
 *   - A GUARD BO IS ALLOCATED FIRST in every mode. Per-fd IOVA starts at zero here, so
 *     without it the first operand lands on IOVA 0 and every base left at the
 *     capture's stored zero points AT THAT OPERAND rather than being disabled. That is
 *     what made the surface read a constant through three rounds of this probe.
 *   - DPU 0x40D0 must be the captured 0x0040FFFF VERBATIM. Reading it as a clamp pair
 *     and writing 0x00407F80 left exactly half of every 16-channel group unwritten,
 *     silently — a coverage failure, not a wrong value.
 *   - OUT_CVT_SCALE is a SIGNED 16-bit field: 32768 flips the output's sign.
 *   - The output BO is stamped through PREP_BO/FINI_BO, never a bare memset.
 *   - `Add(Conv(x), x)` is not a capture of this op. The vendor compiler folds an
 *     identity skip into the convolution's own kernel. See mkadd.py.
 *
 * Usage:  rk3576_add_probe [probe|model|raw|scan|where|prim|mrdma|acc|comb|joint|span|
 *                           unwritten|gate]                              (default: gate)
 * Exit:   0, 1 on a failure, 2 no NPU (skip).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_hw_profile.h"

#define C2       16u
#define SENTINEL 0xA5

/* The DPU_RDMA block opcode. `npu_hw.h` names the other four blocks but not this one,
 * and `unwritten` appends a write to it. */
#define OP_REG_DPU_RDMA_ (BLOCK_DPU_RDMA | PC_OP_01)

/* What the output BO is filled with before the submit. -1 is SENTINEL, which is what
 * makes "this task never wrote" a property of the surface. A mode that asks whether the
 * program ACCUMULATES sets a real value here instead and runs the same case twice. */
static int g_prefill = -1;

/* WHAT THE PART WAS MEASURED TO COMPUTE, not what the register names suggest:
 *
 *     out = sat8( ((ew + ew_offset) * ew_scale >> ew_shift) * out_scale >> out_shift )
 *
 * ONE operand. The DPU-only program the vendor emits for an elementwise op passes the
 * EW cube through its converter and the OUT_CVT, and the primary contributes nothing —
 * see the header of this file and `where`/`prim` for the sweeps that establish it.
 *
 * The final shift ROUNDS HALF TO EVEN. The ladder said "rounds" — sweeping out_shift
 * at scale 1 over a fixed accumulator gives 97 48 24 12 6 3 2 1 0 at shifts 23..31 —
 * and the shape table said which way: against a round-half-up reference EVERY
 * disagreement was an exact tie, 25% of elements at a gain of one half and none at
 * unity. Same rule as the direct path's OUT_CVT. */
static int ew_ref(int ew, const ew_params_rk3576_t *p)
{
    int64_t v = ((int64_t)ew + p->ew_offset) * (int64_t)p->ew_scale;
    int64_t out;

    v >>= p->ew_shift;
    out = v * (int64_t)p->out_scale;
    /* Round half to EVEN — the DPU's requant rule, reproduced here independently:
     * every disagreement a round-half-up reference produced was an exact tie, on 25%
     * of elements at a gain of one half and none at unity. */
    {
        int64_t half = (int64_t)1 << p->out_shift, r;
        int64_t q = out >> p->out_shift;            /* floor */
        r = out - (q << p->out_shift);              /* 0 <= r < half */
        if (r * 2 > half) q += 1;
        else if (r * 2 == half && (q & 1)) q += 1;
        out = q;
    }
    if (out < p->clamp_lo) out = p->clamp_lo;
    if (out > p->clamp_hi) out = p->clamp_hi;
    if (out < -128) out = -128;
    if (out > 127) out = 127;
    return (int)out;
}

static size_t cube_index(unsigned surf, unsigned w, unsigned c,
                         unsigned y, unsigned x)
{
    return (size_t)(c / C2) * surf * C2 + (size_t)C2 * (y * w + x) + (c % C2);
}

/* Build, submit, read back. `a` and `b` are row-major CHW; `out` receives CHW too.
 * Returns 0 on a completed submit, <0 otherwise; *wrote says the surface moved. */
/* How much of the surface the last run left untouched, and where the untouched part
 * starts. A partial write is the failure mode this part shows most often, and it is
 * invisible in a per-element diff against a model. */
static size_t g_sent_bytes, g_total_bytes, g_first_sent;

static int run_ew(int fd, ew_params_rk3576_t *p, const int8_t *a, const int8_t *b,
                  int8_t *out, int *wrote)
{
    unsigned w = p->w, h = p->h, c = p->c;
    unsigned surf = p->surf_elems ? p->surf_elems : w * h;
    unsigned dsurf = p->dst_surf_elems ? p->dst_surf_elems : w * h;
    size_t in_bytes = (size_t)(c / C2) * surf * C2;
    size_t out_bytes = (size_t)(c / C2) * dsurf * C2;
    rocket_bo guard = {0}, bo_a = {0}, bo_b = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_EW_TASK_OPS];
    uint32_t in_h[4], out_h[1];
    unsigned ci, y, x, i;
    int rc = -1;

    *wrote = 0;
    memset(ops, 0, sizeof ops);

    /* THE GUARD MUST BE ALLOCATED FIRST. Per-fd IOVA starts at zero here, so without
     * it the first operand lands on IOVA 0 — and every base this program leaves at the
     * capture's stored zero then points AT THAT OPERAND rather than being disabled.
     * That is not a hypothetical: it is what made the whole surface read a constant. */
    if (rocket_bo_alloc(fd, 4096, &guard) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_a) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_b) < 0) goto done;
    if (rocket_bo_alloc(fd, out_bytes, &bo_o) < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    rocket_bo_prep(fd, &bo_a, 1, 0);
    rocket_bo_prep(fd, &bo_b, 1, 0);
    memset(bo_a.ptr, 0, in_bytes);
    memset(bo_b.ptr, 0, in_bytes);
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                size_t s = ((size_t)ci * h + y) * w + x;
                ((int8_t *)bo_a.ptr)[cube_index(surf, w, ci, y, x)] = a[s];
                ((int8_t *)bo_b.ptr)[cube_index(surf, w, ci, y, x)] = b[s];
            }
    rocket_bo_fini(fd, &bo_a);
    rocket_bo_fini(fd, &bo_b);

    p->src_dma = bo_a.dma_address;
    p->ew_dma = bo_b.dma_address;
    p->dst_dma = bo_o.dma_address;
    p->tasks = ops;
    if (gen_ew_int8_rk3576(p) != 0) { printf("    generator refused\n"); goto done; }

    rocket_bo_prep(fd, &bo_r, 1, 0);
    memcpy(bo_r.ptr, ops, p->task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &bo_r);

    rocket_bo_prep(fd, &bo_o, 1, 0);
    memset(bo_o.ptr, g_prefill < 0 ? SENTINEL : g_prefill, out_bytes);
    rocket_bo_fini(fd, &bo_o);

    in_h[0] = guard.handle; in_h[1] = bo_a.handle;
    in_h[2] = bo_b.handle; in_h[3] = bo_r.handle;
    out_h[0] = bo_o.handle;

    if (rocket_submit_matmul(fd, &bo_r, p->task_count, in_h, 4, out_h, 1, 2000) != 0) {
        printf("    submit failed\n"); goto done;
    }
    if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
        printf("    PREP_BO timed out\n"); goto done;
    }

    g_total_bytes = out_bytes;
    g_sent_bytes = 0;
    g_first_sent = out_bytes;
    for (i = 0; i < out_bytes; i++)
        if (((const uint8_t *)bo_o.ptr)[i] == SENTINEL) {
            g_sent_bytes++;
            if (g_first_sent == out_bytes) g_first_sent = i;
        } else {
            *wrote = 1;
        }

    if (out)
        for (ci = 0; ci < c; ci++)
            for (y = 0; y < h; y++)
                for (x = 0; x < w; x++)
                    out[((size_t)ci * h + y) * w + x] =
                        ((const int8_t *)bo_o.ptr)[cube_index(dsurf, w, ci, y, x)];
    rc = 0;
done:
    if (guard.ptr) rocket_bo_free(fd, &guard);
    if (bo_a.ptr) rocket_bo_free(fd, &bo_a);
    if (bo_b.ptr) rocket_bo_free(fd, &bo_b);
    if (bo_o.ptr) rocket_bo_free(fd, &bo_o);
    if (bo_r.ptr) rocket_bo_free(fd, &bo_r);
    return rc;
}

static void fill_ramp(int8_t *v, size_t n, int seed)
{
    size_t i;
    uint32_t s = 0x9E3779B9u ^ (uint32_t)seed;
    for (i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        v[i] = (int8_t)((int)((s >> 16) % 251u) - 125);
    }
}

/* One case: run it, compare every element against add_ref, report. */
static int check_case(int fd, const char *name, ew_params_rk3576_t *p,
                      const int8_t *a, const int8_t *b)
{
    size_t n = (size_t)p->c * p->h * p->w;
    int8_t *out = malloc(n);
    int wrote = 0, bad = 0, maxd = 0, shown = 0;
    size_t i;

    if (!out) return -1;
    if (run_ew(fd, p, a, b, out, &wrote) != 0) { free(out); return -1; }
    if (!wrote) {
        printf("  %-28s NOTHING WRITTEN\n", name);
        free(out);
        return 1;
    }
    for (i = 0; i < n; i++) {
        int want = ew_ref(b[i], p);
        int d = out[i] - want;
        if (d < 0) d = -d;
        if (d) {
            bad++;
            if (d > maxd) maxd = d;
            if (shown < 4) {
                printf("      [%zu] a=%d b=%d want %d got %d\n",
                       i, a[i], b[i], want, out[i]);
                shown++;
            }
        }
    }
    printf("  %-28s %s  (%d/%zu differ, max %d; %zu/%zu bytes still sentinel"
           ", first at %zu)\n", name, bad ? "FAIL" : "bit-exact", bad, n, maxd,
           g_sent_bytes, g_total_bytes, g_first_sent);
    free(out);
    return bad ? 1 : 0;
}

static void base_params(ew_params_rk3576_t *p, unsigned c, unsigned h, unsigned w)
{
    memset(p, 0, sizeof *p);
    p->w = (uint16_t)w; p->h = (uint16_t)h; p->c = (uint16_t)c;
    p->mode = ROCKET_RK3576_EW_ADD;
    p->clamp_lo = -2147483647-1; p->clamp_hi = 2147483647;
    /* Unity on both paths. The primary's gain is 16384*out_scale>>out_shift, so
     * (1, 14). The EW gain is (ew_scale>>ew_shift) times the SAME out pair, so it
     * needs ew_scale>>ew_shift == 16384 — shift ZERO, not 14. Getting that wrong is
     * an EW contribution of 1/16384, which reads as "the second operand does
     * nothing" rather than as a mis-set field. */
    p->ew_scale = 16384; p->ew_shift = 0;
    p->out_scale = 1; p->out_shift = 14;
}

static int mode_probe(int fd)
{
    ew_params_rk3576_t p;
    int8_t *a, *b;
    int wrote = 0, rc;

    base_params(&p, 32, 16, 16);
    a = malloc(32u * 16 * 16); b = malloc(32u * 16 * 16);
    if (!a || !b) { free(a); free(b); return -1; }
    fill_ramp(a, 32u * 16 * 16, 1);
    fill_ramp(b, 32u * 16 * 16, 2);

    printf("probe: the captured program verbatim, all three bases programmed\n");
    rc = run_ew(fd, &p, a, b, NULL, &wrote);
    if (rc == 0)
        printf("  32x16x16 -> %s\n", wrote ? "WROTE" : "nothing");
    free(a); free(b);
    return rc == 0 && wrote ? 0 : 1;
}

static int mode_model(int fd)
{
    ew_params_rk3576_t p;
    size_t n = 32u * 16 * 16;
    int8_t *a = malloc(n), *b = malloc(n), *z = calloc(n, 1);
    int fails = 0;

    if (!a || !b || !z) { free(a); free(b); free(z); return -1; }
    fill_ramp(a, n, 1);
    fill_ramp(b, n, 2);

    printf("model: separating the two paths\n");

    /* 1. The primary alone. EW scale zero contributes nothing whatever the operand
     *    is, so this is the implicit left shift against OUT_CVT and nothing else. */
    base_params(&p, 32, 16, 16);
    p.ew_scale = 0;
    fails += check_case(fd, "primary alone (ew_scale=0)", &p, a, b) > 0;

    /* 2. The EW operand alone, primary held at zero. */
    base_params(&p, 32, 16, 16);
    fails += check_case(fd, "ew alone (src=0)", &p, z, b) > 0;

    /* 3. Both, unity on each: the part must SUM them. */
    base_params(&p, 32, 16, 16);
    fails += check_case(fd, "both, unity gains", &p, a, b) > 0;

    /* 4. A gain that is not unity on either path, so a model that happens to work at
     *    1.0 is separated from the real one. Primary 1/2, EW 1/4. */
    base_params(&p, 32, 16, 16);
    p.out_scale = 1; p.out_shift = 15;      /* 16384>>15 = 0.5 */
    p.ew_scale = 8192; p.ew_shift = 14;     /* half again -> 0.25 overall */
    fails += check_case(fd, "primary 1/2, ew 1/4", &p, a, b) > 0;

    /* 5. The EW offset, which the header claims is added BEFORE the scale. At an EW
     *    gain of 1/4 an offset of 4 moves the output by exactly 1 if it is pre-scale
     *    and by 4 if it is post-scale, so the two are not confusable. */
    base_params(&p, 32, 16, 16);
    p.out_scale = 1; p.out_shift = 15;
    p.ew_scale = 8192; p.ew_shift = 14;
    p.ew_offset = 4;
    fails += check_case(fd, "ew_offset=4 (pre-scale?)", &p, a, b) > 0;

    /* 6. The OUT_CVT offset, applied to the sum. */
    base_params(&p, 32, 16, 16);
    p.out_offset = 3 * 16384;               /* three output counts at unity gain */
    fails += check_case(fd, "out_offset = 3<<14", &p, a, b) > 0;

    free(a); free(b); free(z);
    return fails ? 1 : 0;
}

/* What did the part actually compute? A small plane, printed against its operands, so
 * the function is read off rather than fitted. Also reports the write COVERAGE per
 * channel group, which is what separates "the model is wrong" from "the writer only
 * reached part of the surface". */
static int mode_raw(int fd)
{
    unsigned c = 32, h = 4, w = 4, ci, y, x, g, ngrp = c / C2;
    size_t n = (size_t)c * h * w;
    int8_t *a = malloc(n), *b = malloc(n), *o = malloc(n);
    ew_params_rk3576_t p;
    int wrote = 0;

    if (!a || !b || !o) { free(a); free(b); free(o); return -1; }
    /* Operands chosen so a, b, a+b and every scaled form are distinguishable. */
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                size_t s = ((size_t)ci * h + y) * w + x;
                a[s] = (int8_t)(1 + (int)(y * w + x));        /* 1..16      */
                b[s] = (int8_t)(-(int)(ci + 1));              /* -1..-32    */
            }

    base_params(&p, c, h, w);
    printf("raw: c=%u %ux%u, a = 1..16 by pixel, b = -(channel+1)\n", c, h, w);
    if (run_ew(fd, &p, a, b, o, &wrote) != 0) { free(a); free(b); free(o); return -1; }
    printf("  wrote=%d, %zu/%zu bytes still sentinel\n", wrote,
           g_sent_bytes, g_total_bytes);

    for (g = 0; g < ngrp; g++) {
        size_t sent = 0;
        for (ci = g * C2; ci < (g + 1) * C2; ci++)
            for (y = 0; y < h; y++)
                for (x = 0; x < w; x++)
                    if ((uint8_t)o[((size_t)ci * h + y) * w + x] == SENTINEL) sent++;
        printf("  channel group %u: %zu/%u still sentinel\n", g, sent, C2 * h * w);
    }
    for (ci = 0; ci < c; ci += 8) {
        printf("  c=%-3u a=%3d..%-3d b=%-4d ->", ci, a[(size_t)ci * h * w],
               a[(size_t)ci * h * w + h * w - 1], b[(size_t)ci * h * w]);
        for (x = 0; x < h * w && x < 8; x++)
            printf(" %4d", o[(size_t)ci * h * w + x]);
        printf("\n");
    }
    free(a); free(b); free(o);
    return 0;
}

/* Several (scale, shift) pairs that all mean the SAME gain. If the part agrees with
 * the model they are interchangeable; if it only computes near the magnitudes the
 * vendor uses, the multiplier has an implied normalization the model is missing. The
 * conv path already has one register of this kind — the coefficient C, where zero
 * silently gates a whole group — so it is worth asking before fitting anything. */
static int mode_scan(int fd)
{
    unsigned c = 32, h = 4, w = 4, ci, y, x, i;
    size_t n = (size_t)c * h * w;
    int8_t *a = malloc(n), *b = malloc(n), *o = malloc(n);
    int wrote = 0;

    if (!a || !b || !o) { free(a); free(b); free(o); return -1; }
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                size_t s = ((size_t)ci * h + y) * w + x;
                a[s] = (int8_t)(1 + (int)(y * w + x));
                b[s] = (int8_t)(-(int)(ci + 1));
            }

    printf("scan: OUT_CVT shift at scale 1, which measures the ACCUMULATOR rather\n"
           "      than fitting a gain.  a = 1..16 by pixel, b = -1 at c=0.\n"
           "      Under `acc = a<<14 + b<<14` the sum at c=0 is 0..7 times 16384,\n"
           "      so it should stop saturating around shift 14 and read 0 by 31.\n");
    for (i = 0; i <= 31; i++) {
        ew_params_rk3576_t p;
        base_params(&p, c, h, w);
        p.out_scale = 1;
        p.out_shift = (uint8_t)i;
        if (run_ew(fd, &p, a, b, o, &wrote) != 0) continue;
        printf("  shift %-2u  c=0 ->", i);
        for (x = 0; x < 6; x++) printf(" %5d", o[x]);
        printf("    c=8 ->");
        for (x = 0; x < 4; x++) printf(" %5d", o[(size_t)8 * h * w + x]);
        printf("\n");
    }
    free(a); free(b); free(o);
    return 0;
}

/* WHERE DO THE TWO OPERANDS GO? The captures store zero in every base, so this is the
 * one thing they cannot answer, and the program computes a CONSTANT with the obvious
 * assignment — primary at SRC_BASE_ADDR, second at EW_BASE_ADDR. Four DPU_RDMA
 * registers read zero in the add captures and could be carrying an address: 0x5018
 * (SRC), 0x5020 (BS), 0x502C (BN) and 0x5030.
 *
 * A GUARD BO IS ALLOCATED FIRST so that neither operand lands on IOVA 0. Per-fd IOVA
 * starts at zero on this stack, so the first BO of the run is at 0 and is
 * indistinguishable from "this register was never programmed" — which is exactly the
 * confound that makes a base-address sweep read as a null result. */
static int mode_where(int fd)
{
    /* Every register the add program leaves at zero, one at a time as the PRIMARY,
     * with the EW operand held at its known-good 0x5038. The DPU-side candidates are
     * included: a DPU-only program's feature read need not be a DPU_RDMA register. */
    static const struct { const char *what; uint16_t sreg, ereg; } PLACES[] = {
        { "0x5018 SRC",        0x5018, 0x5038 },
        { "0x5020 BS",         0x5020, 0x5038 },
        { "0x5024 BS1",        0x5024, 0x5038 },
        { "0x5028 NRDMA_CFG",  0x5028, 0x5038 },
        { "0x502C BN",         0x502C, 0x5038 },
        { "0x5030",            0x5030, 0x5038 },
        { "0x5048 SRC_DMA_CFG",0x5048, 0x5038 },
        { "0x504C SURF_NOTCH", 0x504C, 0x5038 },
        { "0x5064 PAD_CFG",    0x5064, 0x5038 },
        { "0x506C EW_NOTCH",   0x506C, 0x5038 },
        { "0x5078",            0x5078, 0x5038 },
        { "0x507C",            0x507C, 0x5038 },
        { "0x4010 DATA_FORMAT",0x4010, 0x5038 },
        { "0x4014 OFFSET_PEND",0x4014, 0x5038 },
        { "0x40BC",            0x40BC, 0x5038 },
        { "0x40C8",            0x40C8, 0x5038 },
        { "0x40CC",            0x40CC, 0x5038 },
    };
    unsigned c = 32, h = 4, w = 4, ci, y, x, i, k;
    unsigned surf = h * w;
    size_t n = (size_t)c * h * w;
    size_t in_bytes = (size_t)(c / C2) * surf * C2;
    rocket_bo guard = {0}, bo_a = {0}, bo_b = {0}, bo_o = {0}, bo_r = {0};
    int8_t *a = malloc(n), *b = malloc(n);
    uint64_t ops[RK3576_EW_TASK_OPS];
    int rc = -1;

    if (!a || !b) { free(a); free(b); return -1; }
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                size_t s = ((size_t)ci * h + y) * w + x;
                a[s] = (int8_t)(1 + (int)(y * w + x));
                b[s] = (int8_t)(-(int)(ci + 1));
            }

    /* The guard, then the two operands and the output — so no operand is at IOVA 0. */
    if (rocket_bo_alloc(fd, 4096, &guard) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_a) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_b) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_o) < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    rocket_bo_prep(fd, &bo_a, 1, 0);
    rocket_bo_prep(fd, &bo_b, 1, 0);
    memset(bo_a.ptr, 0, in_bytes);
    memset(bo_b.ptr, 0, in_bytes);
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                size_t s = ((size_t)ci * h + y) * w + x;
                ((int8_t *)bo_a.ptr)[cube_index(surf, w, ci, y, x)] = a[s];
                ((int8_t *)bo_b.ptr)[cube_index(surf, w, ci, y, x)] = b[s];
            }
    rocket_bo_fini(fd, &bo_a);
    rocket_bo_fini(fd, &bo_b);

    printf("where: which DPU_RDMA base carries each operand\n");
    printf("  guard BO holds IOVA 0; src=0x%08llx ew=0x%08llx dst=0x%08llx\n",
           (unsigned long long)bo_a.dma_address,
           (unsigned long long)bo_b.dma_address,
           (unsigned long long)bo_o.dma_address);
    printf("  a = 1..16 by pixel, b = -(channel+1); a UNITY sum reads 0 1 2 3 at c=0\n");

    for (i = 0; i < sizeof PLACES / sizeof *PLACES; i++) {
        ew_params_rk3576_t p;
        unsigned sent = 0;

        base_params(&p, c, h, w);
        /* Emit with every candidate base zeroed, then place the two by hand. */
        p.src_dma = 0; p.ew_dma = 0; p.dst_dma = bo_o.dma_address;
        p.out_scale = 1; p.out_shift = 14;
        p.tasks = ops;
        memset(ops, 0, sizeof ops);
        if (gen_ew_int8_rk3576(&p) != 0) continue;
        for (k = 0; k < p.task_count; k++) {
            uint16_t reg = (uint16_t)(ops[k] & 0xFFFF);
            uint32_t v = 0;
            if ((uint16_t)(ops[k] >> 48) == 0) continue;
            if (reg == PLACES[i].sreg)      v = bo_a.dma_address;
            else if (reg == PLACES[i].ereg) v = bo_b.dma_address;
            else continue;
            ops[k] = (ops[k] & ~(0xFFFFFFFFull << 16)) | ((uint64_t)v << 16);
        }

        rocket_bo_prep(fd, &bo_r, 1, 0);
        memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &bo_r);
        rocket_bo_prep(fd, &bo_o, 1, 0);
        memset(bo_o.ptr, SENTINEL, in_bytes);
        rocket_bo_fini(fd, &bo_o);

        {
            uint32_t in_h[4] = { guard.handle, bo_a.handle, bo_b.handle, bo_r.handle };
            uint32_t out_h[1] = { bo_o.handle };
            if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 4,
                                     out_h, 1, 2000) != 0) {
                printf("  %-44s submit failed\n", PLACES[i].what);
                continue;
            }
        }
        if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
            printf("  %-44s PREP_BO timed out\n", PLACES[i].what);
            continue;
        }
        for (k = 0; k < in_bytes; k++)
            if (((const uint8_t *)bo_o.ptr)[k] == SENTINEL) sent++;

        printf("  %-44s c=0 ->", PLACES[i].what);
        for (x = 0; x < 4; x++)
            printf(" %5d", ((const int8_t *)bo_o.ptr)[cube_index(surf, w, 0, 0, x)]);
        printf("   c=8 ->");
        for (x = 0; x < 2; x++)
            printf(" %5d", ((const int8_t *)bo_o.ptr)[cube_index(surf, w, 8, 0, x)]);
        printf("   (%u sentinel)\n", sent);
    }
    rc = 0;
done:
    free(a); free(b);
    if (guard.ptr) rocket_bo_free(fd, &guard);
    if (bo_a.ptr) rocket_bo_free(fd, &bo_a);
    if (bo_b.ptr) rocket_bo_free(fd, &bo_b);
    if (bo_o.ptr) rocket_bo_free(fd, &bo_o);
    if (bo_r.ptr) rocket_bo_free(fd, &bo_r);
    return rc;
}

/* WHAT TURNS THE PRIMARY OPERAND ON? With the guard in place the transcribed program
 * passes the EW operand through exactly and the primary contributes NOTHING, so one of
 * the mode words is gating the DPU's own feature read. The captures differ from a
 * convolution's in five: DPU 0x400C, DPU 0x4038, and DPU_RDMA 0x501C / 0x5034 / 0x5044.
 *
 * This is a JOINT grid over the two most likely — EW_CFG's two RK3576-only high bits
 * and the DPU_RDMA read config — rather than a one-at-a-time sweep. A single-register
 * sweep cannot see a condition of two, and on this part that has already cost one
 * decode twice over (the fp16 mode is three registers, the poisoning two bits).
 *
 * The operands are chosen so each possible answer is a DIFFERENT printout: a varies by
 * pixel only, b by channel only, so `a`, `b`, `a+b` and a constant are four distinct
 * readings at c=0 x=0..3 / c=8. */
static int mode_prim(int fd)
{
    static const uint32_t EWCFG[] = { 0x8002C0C0u, 0x0002C0C0u, 0x4002C0C0u,
                                      0xC002C0C0u, 0x80024040u };
    static const uint32_t BRDMA[] = { 0x1Au, 0x2u, 0x0u, 0x710u };
    unsigned c = 32, h = 4, w = 4, ci, y, x;
    size_t n = (size_t)c * h * w;
    int8_t *a = malloc(n), *b = malloc(n), *o = malloc(n);
    unsigned i, j;
    int wrote = 0;

    if (!a || !b || !o) { free(a); free(b); free(o); return -1; }
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                size_t s = ((size_t)ci * h + y) * w + x;
                a[s] = (int8_t)(1 + (int)(y * w + x));   /* varies by PIXEL   */
                b[s] = (int8_t)(-(int)(ci + 1));         /* varies by CHANNEL */
            }

    printf("prim: joint grid, EW_CFG x BRDMA_CFG.  a varies by pixel, b by channel.\n");
    printf("      a alone reads 1 2 3 4 / 1 2;  b alone -1 -1 -1 -1 / -9 -9;\n");
    printf("      a+b reads 0 1 2 3 / -8 -7.  Anything else is neither.\n");
    for (i = 0; i < sizeof EWCFG / sizeof *EWCFG; i++)
        for (j = 0; j < sizeof BRDMA / sizeof *BRDMA; j++) {
            ew_params_rk3576_t p;
            char spec[96];
            base_params(&p, c, h, w);
            snprintf(spec, sizeof spec, "0x407c=0x%08x,0x501c=0x%08x",
                     EWCFG[i], BRDMA[j]);
            setenv("ROCKET_RK3576_SET", spec, 1);
            if (run_ew(fd, &p, a, b, o, &wrote) != 0) continue;
            printf("  EW_CFG %08x  BRDMA %08x ->", EWCFG[i], BRDMA[j]);
            for (x = 0; x < 4; x++) printf(" %4d", o[x]);
            printf("  /");
            for (x = 0; x < 2; x++) printf(" %4d", o[(size_t)8 * h * w + x]);
            printf("   (%zu sentinel)\n", g_sent_bytes);
        }
    unsetenv("ROCKET_RK3576_SET");
    free(a); free(b); free(o);
    return 0;
}

/* IS THE PRIMARY READ AT ALL, at some gain the earlier sweeps did not reach?
 *
 * "The primary contributes nothing" was measured at ONE gain, and that is a weaker
 * statement than it reads as. The program is CONFIGURED to read a primary from memory:
 * DPU 0x400C bit 0 is the NVDLA feature-mode FLYING bit and the add sets it (off-flying
 * — take the main feed from DMA, not from the MAC array), where a convolution clears
 * it; and DPU_RDMA 0x5044 bit 4 is MRDMA_DISABLE in the same lineage's map and the add
 * CLEARS it where a convolution sets it. Both say the main DMA feed is armed.
 *
 * The EW operand reaches the accumulator scaled by 2^14 — that is what makes its unity
 * gain (out_scale 1, out_shift 14). If the primary enters RAW instead, then at the gain
 * that makes EW unity it is worth 1/16384 of an output count and is indistinguishable
 * from absent. A LADDER over the OUT_CVT shift is what separates those two, and it is
 * the same instrument that measured the accumulator in `scan`.
 *
 * The operands are chosen so one run answers both halves: the primary varies BY PIXEL
 * and is the same in every channel, and the EW operand is ZERO below channel 16 and a
 * constant above it. So the c=0 row is the primary alone, and the c=16 row is the
 * primary plus a known EW contribution — a live EW at every rung is the control that
 * says the rung ran at all. */
static int mode_mrdma(int fd)
{
    static const uint16_t SCALES[] = { 1, 16384 };
    unsigned c = 32, h = 4, w = 4, ci, y, x, i, s;
    size_t n = (size_t)c * h * w;
    int8_t *a = malloc(n), *b = malloc(n), *o = malloc(n);
    int wrote = 0;

    if (!a || !b || !o) { free(a); free(b); free(o); return -1; }
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                size_t k = ((size_t)ci * h + y) * w + x;
                a[k] = (int8_t)(1 + (int)(y * w + x));     /* 1..16, PIXEL only  */
                b[k] = (int8_t)(ci < C2 ? 0 : -8);         /* EW off below c=16  */
            }

    printf("mrdma: the OUT_CVT ladder with the primary the only live operand below\n"
           "       c=16.  primary = 1..16 by pixel, EW = 0 (c<16) / -8 (c>=16).\n"
           "       A live raw primary reads 1 2 3 4 at scale 1 shift 0; a primary\n"
           "       that enters scaled by 2^14 reads it at shift 14.  The c=16 column\n"
           "       is the control: EW is armed at every rung.\n");
    for (s = 0; s < sizeof SCALES / sizeof *SCALES; s++)
        for (i = 0; i <= 31; i++) {
            ew_params_rk3576_t p;
            int varies = 0;
            base_params(&p, c, h, w);
            p.out_scale = SCALES[s];
            p.out_shift = (uint8_t)i;
            if (run_ew(fd, &p, a, b, o, &wrote) != 0) continue;
            for (x = 1; x < h * w; x++) if (o[x] != o[0]) varies = 1;
            printf("  scale %-5u shift %-2u  c=0 ->", (unsigned)SCALES[s], i);
            for (x = 0; x < 6; x++) printf(" %5d", o[x]);
            printf("   c=16 ->");
            for (x = 0; x < 3; x++) printf(" %5d", o[(size_t)C2 * h * w + x]);
            printf("   %s\n", varies ? "VARIES BY PIXEL" : "");
        }
    free(a); free(b); free(o);
    return 0;
}

/* DOES THE PROGRAM ACCUMULATE INTO ITS DESTINATION?
 *
 * If it does, the vendor's add is two passes over one surface — pass one writes operand
 * A, pass two adds operand B into it — and that is a residual add today, with no second
 * base to find. It is also what the capture's shape hints at: every add capture carries
 * the dpu-only program TWICE, register for register identical, and the only thing that
 * can differ between two identical programs is the addresses the runtime patches in.
 *
 * The classifier is the DESTINATION'S PRIOR CONTENTS, not a sentinel: each candidate
 * runs twice, once over a destination pre-filled with one value and once with another.
 * Same output both times means the write is destination-independent; an output that
 * moves by exactly the difference means it accumulates; an output equal to its own
 * prefill both times means nothing was written. A sentinel cannot separate the first
 * two, which is why it is not used here.
 *
 * DPU 0x40C0 is the register swept: it reads 0x04440000 in every add capture against a
 * convolution's 0x04440100, and its RK3588 twin is SURFACE_ADD. Single bits are tried
 * on top of the captured word rather than whole invented values — the captures carry it
 * as a constant, and inventing a field decomposition for one of those has already cost
 * this probe two rounds. */
static int mode_acc(int fd)
{
    static const uint32_t C40C0[] = {
        0x04440000u,            /* the add capture's own word: the control  */
        0x04440100u,            /* a convolution's                          */
        0x04440001u, 0x04440002u, 0x04440004u, 0x04440008u,
        0x04440010u, 0x04440020u, 0x04440040u, 0x04440080u,
        0x04440200u, 0x04440400u, 0x04440800u, 0x04441000u,
        0x04442000u, 0x04444000u, 0x04448000u,
        0x04450000u, 0x04460000u, 0x04480000u, 0x04540000u,
        0x04640000u, 0x05440000u, 0x06440000u, 0x0C440000u,
        0x14440000u, 0x24440000u, 0x44440000u, 0x84440000u,
    };
    unsigned c = 32, h = 4, w = 4, ci, y, x, i;
    size_t n = (size_t)c * h * w;
    int8_t *a = malloc(n), *b = malloc(n);
    int8_t *o1 = malloc(n), *o2 = malloc(n);
    const int P1 = 0, P2 = 16;
    int wrote = 0, rc = 0;

    if (!a || !b || !o1 || !o2) { free(a); free(b); free(o1); free(o2); return -1; }
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                size_t k = ((size_t)ci * h + y) * w + x;
                a[k] = (int8_t)(1 + (int)(y * w + x));
                b[k] = 2;                       /* a small constant: f(ew) == 2 */
            }

    printf("acc: does the write accumulate into the destination?  EW operand is a\n"
           "     constant 2 at unity gain, so an overwrite reads 2 whatever the\n"
           "     destination held.  Prefills %d and %d.\n", P1, P2);
    for (i = 0; i < sizeof C40C0 / sizeof *C40C0; i++) {
        char spec[64];
        ew_params_rk3576_t p;
        int same = 1, accum = 1, dead = 1, k;

        snprintf(spec, sizeof spec, "0x40c0=0x%08x", C40C0[i]);
        setenv("ROCKET_RK3576_SET", spec, 1);

        base_params(&p, c, h, w);
        g_prefill = P1;
        if (run_ew(fd, &p, a, b, o1, &wrote) != 0) { g_prefill = -1; continue; }
        base_params(&p, c, h, w);
        g_prefill = P2;
        if (run_ew(fd, &p, a, b, o2, &wrote) != 0) { g_prefill = -1; continue; }
        g_prefill = -1;

        for (k = 0; k < (int)n; k++) {
            if (o1[k] != o2[k]) same = 0;
            if (o2[k] - o1[k] != P2 - P1) accum = 0;
            if (o1[k] != P1 || o2[k] != P2) dead = 0;
        }
        printf("  0x40C0 %08x -> p%d %4d %4d %4d  p%d %4d %4d %4d   %s\n",
               C40C0[i], P1, o1[0], o1[1], o1[2], P2, o2[0], o2[1], o2[2],
               dead ? "NOTHING WRITTEN" :
               accum ? "ACCUMULATES" : same ? "overwrite" : "neither");
    }
    unsetenv("ROCKET_RK3576_SET");
    free(a); free(b); free(o1); free(o2);
    return rc;
}

/* WHAT ARMS THE MAIN DMA FEED? A JOINT GRID over the three words that carry it.
 *
 * The RK3588's own K-accumulation path is the same IP block doing the same thing, and
 * it needs THREE fields together: the DPU's feature mode off-flying, MRDMA enabled in
 * DPU_RDMA_FEATURE_MODE_CFG, and COMB_USE in that same word combining the MRDMA feed
 * with the ERDMA operand. The RK3576 add capture already carries the first two — 0x400C
 * bit 0 set, 0x5044 bit 4 clear — and carries ZERO where the RK3588 puts COMB_USE, so
 * that field is the candidate this grid drives.
 *
 * Jointly with the ERDMA config, because a field that gates a combine can be inert
 * while the operand path it combines into is wrong; a one-at-a-time sweep cannot see a
 * condition of two, and on this part that has cost a decode three times over.
 *
 * The gain is the one that makes a RAW primary readable (out_scale 1, out_shift 0), not
 * the one that makes the EW operand unity — at unity the primary would be 1/16384 of a
 * count whatever the grid does. `mrdma` is what fixes that choice; run it first. */
static int mode_comb(int fd)
{
    static const uint32_t FMC[]   = { 0x00000009u, 0x00000109u, 0x00000209u,
                                      0x00000309u, 0x00000409u, 0x00000509u,
                                      0x00000609u, 0x00000709u, 0x00000001u,
                                      0x00000509u | 0x8u };
    static const uint32_t ERDMA[] = { 0x40000044u, 0x40000041u, 0x00000044u };
    unsigned c = 32, h = 4, w = 4, ci, y, x, i, j;
    size_t n = (size_t)c * h * w;
    int8_t *a = malloc(n), *b = malloc(n), *o = malloc(n);
    int wrote = 0;

    if (!a || !b || !o) { free(a); free(b); free(o); return -1; }
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                size_t k = ((size_t)ci * h + y) * w + x;
                a[k] = (int8_t)(1 + (int)(y * w + x));     /* varies by PIXEL   */
                b[k] = (int8_t)(ci < C2 ? 0 : -8);         /* EW off below c=16 */
            }

    printf("comb: joint grid, DPU_RDMA 0x5044 x 0x5034, at a gain a RAW primary shows\n"
           "      at.  primary 1..16 by pixel, EW 0 below c=16 and -8 above it.\n"
           "      A live primary reads 1 2 3 4 at c=0.  c=16 is the EW control.\n");
    for (i = 0; i < sizeof FMC / sizeof *FMC; i++)
        for (j = 0; j < sizeof ERDMA / sizeof *ERDMA; j++) {
            ew_params_rk3576_t p;
            char spec[96];
            int varies = 0;
            base_params(&p, c, h, w);
            p.out_scale = 1; p.out_shift = 0;
            snprintf(spec, sizeof spec, "0x5044=0x%08x,0x5034=0x%08x", FMC[i], ERDMA[j]);
            setenv("ROCKET_RK3576_SET", spec, 1);
            if (run_ew(fd, &p, a, b, o, &wrote) != 0) {
                printf("  FMC %08x  ERDMA %08x   submit refused\n", FMC[i], ERDMA[j]);
                continue;
            }
            for (x = 1; x < h * w; x++) if (o[x] != o[0]) varies = 1;
            printf("  FMC %08x  ERDMA %08x  c=0 ->", FMC[i], ERDMA[j]);
            for (x = 0; x < 4; x++) printf(" %5d", o[x]);
            printf("   c=16 ->");
            for (x = 0; x < 2; x++) printf(" %5d", o[(size_t)C2 * h * w + x]);
            printf("   (%zu sentinel)%s\n", g_sent_bytes, varies ? " VARIES" : "");
        }
    unsetenv("ROCKET_RK3576_SET");
    free(a); free(b); free(o);
    return 0;
}

/* PLACEMENT CROSSED WITH GAIN. `where` swept every base the program leaves at zero at
 * ONE output gain, and `mrdma` swept the gain at ONE placement — so the combination has
 * never been tried, and a first operand that both sits somewhere unexpected AND enters
 * the accumulator unscaled is invisible to either sweep alone.
 *
 * The manufactured captures say there IS a first operand and what its weight is. `Sub`
 * compiles to this same program with the EW converter's scale NEGATED, and `Sub` with
 * its operands swapped negates it too — so the SUBTRAHEND is always the EW cube, the
 * other operand rides a different feed, and that feed's weight is fixed at +1 because no
 * register carries a second scale. An asymmetric-scale capture confirms the shape of it:
 * with the two graph inputs calibrated 64x apart the program still has ONE operand
 * converter, and the OUT_CVT gain comes out as the ratio of ONE shared input scale to
 * the output's — the compiler quantizes BOTH operands to a COMMON scale.
 *
 * The suspect this adds over `where` is DPU_RDMA 0x5020, the BS-stage operand. The add
 * arms BRDMA (0x501C = 0x1A, and the `prim` grid found that word load-bearing) and
 * leaves DPU 0x4050 at ZERO where a convolution writes 0x80011111 — a word of repeating
 * nibbles that reads as a bank of bypass fields. A conv bypasses the BS stage; this
 * program does not.
 *
 * The report is one line per (base, rung) that put anything non-zero at c=0, so a live
 * feed shows as a run of rungs and a dead one prints nothing. The c=16 column is the EW
 * control: it is armed at every rung, so a base whose rung prints nothing at c=0 while
 * c=16 tracks is a placement that did not carry, not a submit that did not run. */
static int mode_joint(int fd)
{
    static const uint16_t PLACES[] = {
        0x5018, 0x5020, 0x5024, 0x5028, 0x502C, 0x5030, 0x5048, 0x504C,
        0x5064, 0x506C, 0x5078, 0x507C, 0x4010, 0x4014, 0x40BC, 0x40C8, 0x40CC,
    };
    unsigned c = 32, h = 4, w = 4, ci, y, x, i, k, sh;
    unsigned surf = h * w;
    size_t in_bytes = (size_t)(c / C2) * surf * C2;
    rocket_bo guard = {0}, bo_a = {0}, bo_b = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_EW_TASK_OPS];
    int rc = -1;

    /* The guard first, so no operand lands on IOVA 0 and no unprogrammed base reads a
     * live buffer. */
    if (rocket_bo_alloc(fd, 4096, &guard) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_a) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_b) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_o) < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    rocket_bo_prep(fd, &bo_a, 1, 0);
    rocket_bo_prep(fd, &bo_b, 1, 0);
    memset(bo_a.ptr, 0, in_bytes);
    memset(bo_b.ptr, 0, in_bytes);
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                ((int8_t *)bo_a.ptr)[cube_index(surf, w, ci, y, x)] =
                    (int8_t)(1 + (int)(y * w + x));            /* by PIXEL      */
                ((int8_t *)bo_b.ptr)[cube_index(surf, w, ci, y, x)] =
                    (int8_t)(ci < C2 ? 0 : -8);                /* EW control    */
            }
    rocket_bo_fini(fd, &bo_a);
    rocket_bo_fini(fd, &bo_b);

    printf("joint: every candidate base x the OUT_CVT ladder.  first operand = 1..16 by\n"
           "       pixel, EW = 0 below c=16 and -8 above it.  Only rungs that put\n"
           "       something non-zero at c=0 are printed.\n");
    for (i = 0; i < sizeof PLACES / sizeof *PLACES; i++) {
        unsigned hits = 0;
        for (sh = 0; sh <= 31; sh++) {
            ew_params_rk3576_t p;
            const int8_t *o;
            int varies = 0, live = 0;

            base_params(&p, c, h, w);
            p.out_scale = 1; p.out_shift = (uint8_t)sh;
            p.src_dma = 0; p.ew_dma = 0; p.dst_dma = bo_o.dma_address;
            p.tasks = ops;
            memset(ops, 0, sizeof ops);
            if (gen_ew_int8_rk3576(&p) != 0) continue;
            for (k = 0; k < p.task_count; k++) {
                uint16_t reg = (uint16_t)(ops[k] & 0xFFFF);
                uint32_t v;
                if ((uint16_t)(ops[k] >> 48) == 0) continue;
                if (reg == PLACES[i])    v = bo_a.dma_address;
                else if (reg == 0x5038)  v = bo_b.dma_address;
                else continue;
                ops[k] = (ops[k] & ~(0xFFFFFFFFull << 16)) | ((uint64_t)v << 16);
            }

            rocket_bo_prep(fd, &bo_r, 1, 0);
            memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
            rocket_bo_fini(fd, &bo_r);
            rocket_bo_prep(fd, &bo_o, 1, 0);
            memset(bo_o.ptr, SENTINEL, in_bytes);
            rocket_bo_fini(fd, &bo_o);
            {
                uint32_t in_h[4] = { guard.handle, bo_a.handle, bo_b.handle, bo_r.handle };
                uint32_t out_h[1] = { bo_o.handle };
                if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 4,
                                         out_h, 1, 2000) != 0) continue;
            }
            if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) continue;

            o = (const int8_t *)bo_o.ptr;
            for (x = 0; x < surf; x++) {
                int v = o[cube_index(surf, w, 0, x / w, x % w)];
                if ((uint8_t)v == SENTINEL) { live = 0; break; }
                if (v) live = 1;
                if (v != o[cube_index(surf, w, 0, 0, 0)]) varies = 1;
            }
            if (!live) continue;
            hits++;
            printf("  base %04x  shift %-2u  c=0 ->", PLACES[i], sh);
            for (x = 0; x < 4; x++)
                printf(" %5d", o[cube_index(surf, w, 0, 0, x)]);
            printf("   c=16 ->");
            for (x = 0; x < 2; x++)
                printf(" %5d", o[cube_index(surf, w, C2, 0, x)]);
            printf("   %s\n", varies ? "VARIES BY PIXEL" : "constant");
        }
        if (!hits) printf("  base %04x  nothing at c=0 at any of 32 rungs\n", PLACES[i]);
    }
    rc = 0;
done:
    if (guard.ptr) rocket_bo_free(fd, &guard);
    if (bo_a.ptr) rocket_bo_free(fd, &bo_a);
    if (bo_b.ptr) rocket_bo_free(fd, &bo_b);
    if (bo_o.ptr) rocket_bo_free(fd, &bo_o);
    if (bo_r.ptr) rocket_bo_free(fd, &bo_r);
    return rc;
}

/* WHAT DOES THE PROGRAM ACTUALLY READ? A DELTA PROBE OVER THE OPERAND BUFFER.
 *
 * Every base the program leaves at zero has now been tried as a first operand, at every
 * output gain, and none carries one — so if a first operand exists it does not arrive
 * through a base of its own. The remaining way for two operands to reach one DPU program
 * with ONE base register is for them to be ONE ALLOCATION, the second at a fixed offset
 * from the first. That is exactly how the RK3588's own K-accumulation feeds its two
 * operands: one buffer, the main feed at the base and the elementwise feed one surface
 * further on.
 *
 * A probe that only ever hands the base a buffer the size of one cube cannot see that —
 * the second read lands past the end, the IOMMU absorbs it, and the operand comes back
 * as though it were alone. Which is what every measurement here has done.
 *
 * So: give the base a buffer EIGHT cubes long, point it at the middle, and walk a single
 * non-zero 16-byte atom over the whole span. Whatever the program reads shows up as an
 * output atom that moved. If only the addressed cube is read, the atoms inside it map
 * one to one and nothing outside it does anything — and that closes the operand question
 * for good. If something outside it moves an output atom, the offset it moved from IS
 * the second operand's placement. */
static int mode_span(int fd)
{
    unsigned c = 32, h = 4, w = 4, x, k;
    unsigned surf = h * w;
    unsigned groups = c / C2;
    size_t cube = (size_t)groups * surf * C2;      /* one operand cube, bytes    */
    unsigned atoms = (unsigned)(cube / C2);        /* 16-byte atoms in one cube  */
    unsigned span = atoms * 8;                     /* the buffer, in atoms       */
    unsigned base_atom = atoms * 2;                /* point the base at cube 2   */
    rocket_bo guard = {0}, bo_b = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_EW_TASK_OPS];
    unsigned found_in = 0, found_out = 0;
    int rc = -1;

    if (rocket_bo_alloc(fd, 4096, &guard) < 0) goto done;
    if (rocket_bo_alloc(fd, (size_t)span * C2, &bo_b) < 0) goto done;
    if (rocket_bo_alloc(fd, cube, &bo_o) < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    printf("span: one non-zero atom walked over an operand buffer %u atoms long, with\n"
           "      the base at atom %u.  Unity gain, so an output atom reads the source\n"
           "      atom that fed it.  Atoms %u..%u are the addressed cube.\n",
           span, base_atom, base_atom, base_atom + atoms - 1);

    for (k = 0; k < span; k++) {
        ew_params_rk3576_t p;
        const int8_t *o;
        int hit = -1, nhit = 0;

        rocket_bo_prep(fd, &bo_b, 1, 0);
        memset(bo_b.ptr, 0, (size_t)span * C2);
        memset((uint8_t *)bo_b.ptr + (size_t)k * C2, 1, C2);
        rocket_bo_fini(fd, &bo_b);

        base_params(&p, c, h, w);
        p.src_dma = guard.dma_address;
        p.ew_dma = bo_b.dma_address + (uint32_t)((size_t)base_atom * C2);
        p.dst_dma = bo_o.dma_address;
        p.tasks = ops;
        memset(ops, 0, sizeof ops);
        if (gen_ew_int8_rk3576(&p) != 0) continue;

        rocket_bo_prep(fd, &bo_r, 1, 0);
        memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &bo_r);
        rocket_bo_prep(fd, &bo_o, 1, 0);
        memset(bo_o.ptr, SENTINEL, cube);
        rocket_bo_fini(fd, &bo_o);
        {
            uint32_t in_h[3] = { guard.handle, bo_b.handle, bo_r.handle };
            uint32_t out_h[1] = { bo_o.handle };
            if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 3,
                                     out_h, 1, 2000) != 0) continue;
        }
        if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) continue;

        o = (const int8_t *)bo_o.ptr;
        for (x = 0; x < cube; x++)
            if (o[x] != 0 && (uint8_t)o[x] != SENTINEL) {
                nhit++;
                if (hit < 0) hit = (int)(x / C2);
            }
        if (nhit) {
            int inside = k >= base_atom && k < base_atom + atoms;
            if (inside) found_in++; else found_out++;
            printf("  src atom %3u (%s%+d) -> output atom %d, %d byte(s)%s\n",
                   k, inside ? "in cube " : "OUTSIDE", (int)k - (int)base_atom,
                   hit, nhit, inside ? "" : "   <== READ BEYOND THE ADDRESSED CUBE");
        }
    }
    printf("== %u atom(s) inside the addressed cube reached the output, %u outside ==\n",
           found_in, found_out);
    rc = 0;
done:
    if (guard.ptr) rocket_bo_free(fd, &guard);
    if (bo_b.ptr) rocket_bo_free(fd, &bo_b);
    if (bo_o.ptr) rocket_bo_free(fd, &bo_o);
    if (bo_r.ptr) rocket_bo_free(fd, &bo_r);
    return rc;
}

/* THE REGISTERS THE PROGRAM NEVER WRITES.
 *
 * Every sweep so far has moved a register the add program WRITES — as a zero, as a mode
 * word, as a base. The complement has never been touched, and on this part it is not an
 * empty set: the register file is NOT cleared between jobs, so a base the vendor
 * programs in an earlier task is still standing when the add runs. A first operand
 * living there would be invisible in every capture (the add program does not carry it)
 * and dead in every standalone submit here (nothing ever set it) — which is exactly the
 * pair of facts this probe has been accumulating.
 *
 * So: append a write of the operand's address at each offset in the DPU and DPU_RDMA
 * blocks that the program leaves alone, one at a time. The extra write goes in BEFORE
 * the four-word trailer, because the trailer is what the driver reads to size and start
 * the kick, and a write after it is not part of the program.
 *
 * Two rungs each: the gain that makes the EW operand unity, and the one that makes a raw
 * operand unity. A feed that enters unscaled is invisible at the first and obvious at
 * the second, and holding only one of them is how this question stayed open. */
static int mode_unwritten(int fd)
{
    static const uint16_t CAND[] = {
        /* DPU: every 4-byte offset in 0x4000-0x40FC the program does not write. */
        0x4008, 0x4040, 0x4054, 0x4064, 0x4068, 0x4098, 0x40A0, 0x40C4,
        0x40D4, 0x40D8, 0x40DC, 0x40E0, 0x40E4, 0x40E8, 0x40EC, 0x40F0,
        0x40F4, 0x40F8, 0x40FC,
        /* DPU_RDMA: the same, over 0x5000-0x507C. */
        0x5008, 0x503C, 0x5050, 0x5054, 0x5058, 0x505C, 0x5060, 0x5068,
        0x5070, 0x5074,
    };
    static const uint8_t SHIFTS[] = { 14, 0 };
    unsigned c = 32, h = 4, w = 4, ci, y, x, i, j, k;
    unsigned surf = h * w;
    size_t in_bytes = (size_t)(c / C2) * surf * C2;
    rocket_bo guard = {0}, bo_a = {0}, bo_b = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_EW_TASK_OPS + 4];
    int rc = -1;

    if (rocket_bo_alloc(fd, 4096, &guard) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_a) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_b) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &bo_o) < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    rocket_bo_prep(fd, &bo_a, 1, 0);
    rocket_bo_prep(fd, &bo_b, 1, 0);
    memset(bo_a.ptr, 0, in_bytes);
    memset(bo_b.ptr, 0, in_bytes);
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                ((int8_t *)bo_a.ptr)[cube_index(surf, w, ci, y, x)] =
                    (int8_t)(1 + (int)(y * w + x));
                ((int8_t *)bo_b.ptr)[cube_index(surf, w, ci, y, x)] =
                    (int8_t)(ci < C2 ? 0 : -8);
            }
    rocket_bo_fini(fd, &bo_a);
    rocket_bo_fini(fd, &bo_b);

    printf("unwritten: the operand address appended at each register the program leaves\n"
           "           alone.  first operand = 1..16 by pixel, EW = 0 below c=16.\n");
    for (i = 0; i < sizeof CAND / sizeof *CAND; i++)
        for (j = 0; j < sizeof SHIFTS / sizeof *SHIFTS; j++) {
            ew_params_rk3576_t p;
            const int8_t *o;
            unsigned n, sent = 0;
            int varies = 0, live = 0;

            base_params(&p, c, h, w);
            p.out_scale = 1; p.out_shift = SHIFTS[j];
            p.src_dma = 0;
            p.ew_dma = bo_b.dma_address;
            p.dst_dma = bo_o.dma_address;
            p.tasks = ops;
            memset(ops, 0, sizeof ops);
            if (gen_ew_int8_rk3576(&p) != 0) continue;
            n = p.task_count;
            /* Insert before the four-word trailer, which must stay last. */
            memmove(&ops[n - 3], &ops[n - 4], 4 * sizeof ops[0]);
            ops[n - 4] = NPUOP(OP_REG_DPU, bo_a.dma_address, CAND[i]);
            if (CAND[i] >= 0x5000)
                ops[n - 4] = NPUOP(OP_REG_DPU_RDMA_, bo_a.dma_address, CAND[i]);
            n++;

            rocket_bo_prep(fd, &bo_r, 1, 0);
            memcpy(bo_r.ptr, ops, n * sizeof(uint64_t));
            rocket_bo_fini(fd, &bo_r);
            rocket_bo_prep(fd, &bo_o, 1, 0);
            memset(bo_o.ptr, SENTINEL, in_bytes);
            rocket_bo_fini(fd, &bo_o);
            {
                uint32_t in_h[4] = { guard.handle, bo_a.handle, bo_b.handle, bo_r.handle };
                uint32_t out_h[1] = { bo_o.handle };
                if (rocket_submit_matmul(fd, &bo_r, n, in_h, 4, out_h, 1, 2000) != 0) {
                    printf("  %04x shift %-2u  submit failed\n", CAND[i], SHIFTS[j]);
                    continue;
                }
            }
            if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
                printf("  %04x shift %-2u  PREP_BO timed out\n", CAND[i], SHIFTS[j]);
                continue;
            }
            o = (const int8_t *)bo_o.ptr;
            for (k = 0; k < in_bytes; k++)
                if ((uint8_t)o[k] == SENTINEL) sent++;
            for (x = 0; x < surf; x++) {
                int v = o[cube_index(surf, w, 0, x / w, x % w)];
                if ((uint8_t)v == SENTINEL) continue;
                if (v) live = 1;
                if (v != o[cube_index(surf, w, 0, 0, 0)]) varies = 1;
            }
            printf("  %04x shift %-2u  c=0 ->", CAND[i], SHIFTS[j]);
            for (x = 0; x < 4; x++)
                printf(" %5d", o[cube_index(surf, w, 0, 0, x)]);
            printf("   c=16 ->");
            for (x = 0; x < 2; x++)
                printf(" %5d", o[cube_index(surf, w, C2, 0, x)]);
            printf("   (%u sentinel)%s%s\n", sent, live ? " LIVE" : "",
                   varies ? " VARIES" : "");
        }
    rc = 0;
done:
    if (guard.ptr) rocket_bo_free(fd, &guard);
    if (bo_a.ptr) rocket_bo_free(fd, &bo_a);
    if (bo_b.ptr) rocket_bo_free(fd, &bo_b);
    if (bo_o.ptr) rocket_bo_free(fd, &bo_o);
    if (bo_r.ptr) rocket_bo_free(fd, &bo_r);
    return rc;
}

typedef struct { const char *name; unsigned c, h, w; } shape;

static const shape SHAPES[] = {
    { "c16 16x16",   16, 16, 16 },
    { "c32 16x16",   32, 16, 16 },
    { "c48 16x16",   48, 16, 16 },
    { "c64 28x28",   64, 28, 28 },
    { "c96 14x14",   96, 14, 14 },
    { "c32 15x18",   32, 15, 18 },
    { "c32 7x7",     32,  7,  7 },
    { "c160 7x7",   160,  7,  7 },
    { "c320 1x1",   320,  1,  1 },
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof *SHAPES))

static int mode_gate(int fd)
{
    int i, fails = 0, ran = 0;

    printf("gate: the shape table against the model\n");
    for (i = 0; i < N_SHAPES; i++) {
        ew_params_rk3576_t p;
        size_t n = (size_t)SHAPES[i].c * SHAPES[i].h * SHAPES[i].w;
        int8_t *a = malloc(n), *b = malloc(n);
        if (!a || !b) { free(a); free(b); return -1; }
        fill_ramp(a, n, i + 1);
        fill_ramp(b, n, i + 101);

        base_params(&p, SHAPES[i].c, SHAPES[i].h, SHAPES[i].w);
        /* Halve both so an ordinary sum does not saturate, which is what a real
         * residual quantization does anyway. */
        p.out_shift = 15;
        fails += check_case(fd, SHAPES[i].name, &p, a, b) > 0;
        ran++;
        free(a); free(b);
    }

    /* Saturation, deliberately: unity gains on data that sums past int8. */
    {
        ew_params_rk3576_t p;
        size_t n = 32u * 16 * 16;
        int8_t *a = malloc(n), *b = malloc(n);
        size_t k;
        if (!a || !b) { free(a); free(b); return -1; }
        for (k = 0; k < n; k++) { a[k] = (int8_t)(k & 1 ? 120 : -120);
                                  b[k] = (int8_t)(k & 1 ? 100 : -100); }
        base_params(&p, 32, 16, 16);
        fails += check_case(fd, "saturation, unity gains", &p, a, b) > 0;
        ran++;
        free(a); free(b);
    }

    printf("== %d shape(s), %d failed ==\n", ran, fails);
    return fails ? 1 : 0;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *mode = argc > 1 ? argv[1] : "gate";
    int fd, rc;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_add_probe: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_add_probe: no NPU device — skipping\n"); return 2; }

    if (!strcmp(mode, "probe"))      rc = mode_probe(fd);
    else if (!strcmp(mode, "model")) rc = mode_model(fd);
    else if (!strcmp(mode, "raw"))   rc = mode_raw(fd);
    else if (!strcmp(mode, "scan"))  rc = mode_scan(fd);
    else if (!strcmp(mode, "where")) rc = mode_where(fd);
    else if (!strcmp(mode, "prim"))  rc = mode_prim(fd);
    else if (!strcmp(mode, "mrdma")) rc = mode_mrdma(fd);
    else if (!strcmp(mode, "acc"))   rc = mode_acc(fd);
    else if (!strcmp(mode, "comb"))  rc = mode_comb(fd);
    else if (!strcmp(mode, "joint")) rc = mode_joint(fd);
    else if (!strcmp(mode, "span"))  rc = mode_span(fd);
    else if (!strcmp(mode, "unwritten")) rc = mode_unwritten(fd);
    else if (!strcmp(mode, "gate"))  rc = mode_gate(fd);
    else { printf("usage: %s [probe|model|raw|scan|where|prim|mrdma|acc|comb|joint|span|unwritten|gate]\n",
                  argv[0]); rc = 1; }

    rocket_close(fd);
    return rc;
}
