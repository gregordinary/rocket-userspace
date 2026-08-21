// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_lut_probe.c — read the DPU LUT's INPUT map off the part.
 *
 * The output side of the table is decoded from manufactured vendor captures: two
 * 513-entry tables, entries Q15 of the function's output, joined into one monotone
 * curve, with output clamps and tail slopes beside them. What no capture can give is
 * the map from a datapath value onto a table index — the table-LOAD program is
 * register-identical across every activation, and at fp16 sigmoid and tanh differ in
 * NOTHING but their entries while their tables span 6.25 and 3.02 input units. So the
 * compiler expresses each function's span in the table CONTENTS, and a diff of vendor
 * programs cannot yield an equation for the map however many captures it is given.
 *
 * The instrument is our own program on silicon. Load a table whose entries encode
 * their own index, feed values whose position in the datapath is known exactly, and
 * read the map off directly.
 *
 * WHAT ENABLES THE LUT, diffed against the bare-conv control of the same capture set
 * (not across activations, which is what hid it): DPU EW_CFG `0x407c` 0x010041c1 ->
 * 0x01004140, which is EW_LUT_BYPASS clearing — so the LUT is the EW stage and its
 * input is what leaves BN. Then LUT_CFG `0x4108` -> 0x02000006, LUT_INFO `0x410c` ->
 * 0x00050500, LE_START `0x4110`-`0x411c` -> 0xffffc000, LO_END `0x4140`-`0x414c` ->
 * 0x00004000, DPU_RDMA NRDMA_CFG `0x5028` -> 0x1a, and every block's `0x*004` from
 * 0x0e to 0x30.
 *
 * AND ONE REGISTER IN THAT DIFF IS A TRAP. Every vendor activation also carries
 * BN_CFG `0x4060` = 0x20 — the BN stage ACTIVE — and transcribing it pins the LUT at
 * the table join whatever the accumulator holds, because BN then multiplies by an
 * operand register no vendor program writes and this register file is not cleared
 * between jobs. Left at the generator's own 0x903 the BN stage is bypassed, the LUT
 * reads the BS output directly, and the map appears. That is the whole of the earlier
 * "the input map is not in any register" finding: it is in registers, the vendor
 * never moves them, and copying the vendor's stage configuration hides them.
 *
 * THE SWEEP. One 1x1 convolution, 32 input channels of which only channel 0 is
 * non-zero, weight 1 into every output channel, so the accumulator IS the feature
 * byte: 256 pixels carry -128..127. Each of the 32 output channels then applies its
 * own A and C from the coefficient group, so one submit samples 8192 (value -> index)
 * pairs. The table is a RAMP in the index and the two clamps sit outside its range, so
 * the readout says three things at once: below the domain, inside it (the index,
 * scaled), and above it.
 *
 * Modes:
 *   load    does the DPU-only table-load program execute at all
 *   tables  the same sweep under eight different tables — a readout that tracks the
 *           table is a loaded table, one that does not is not there
 *   one     one ramp run, one line; ROCKET_LUT_SKIP / _SET / _BN / _ADD drive it,
 *           which is the register hunt
 *   map     the index as a function of the datapath value, per channel
 *   interp  does it interpolate between entries, or step
 *   gate    the whole map against a CPU model, at four different spans
 *
 * Usage:  rk3576_lut_probe [load|tables|one|map|interp|gate]   (default: gate)
 * Exit:   0, 1 on a failure, 2 no NPU (skip).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_hw_profile.h"

#define C2        16u
#define SENTINEL  0xAA
#define IC        32u
#define OC        32u
#define PLANE     16u          /* 16x16 = 256 pixels, one per int8 value */
#define NPIX      (PLANE * PLANE)
#define NENT      (2 * RK3576_LUT_ENTRIES)   /* 1026 */

/* The readout: OUT_CVT is programmed as a >>8 of the LUT result, so a table entry
 * reads back as entry>>8 and the two clamps read back as the extremes. */
#define RAMP_STEP   31         /* 31*1025 = 31775, inside int16 */
#define CLAMP_LO    (-32768)   /* reads back as -128 */
#define CLAMP_HI    ( 32767)   /* reads back as  127 */

/* Registers the conv generator never emits, appended through its own hook. Empty by
 * default: the four the vendor's programs also leave alone (0x4040, 0x4054, 0x4064,
 * 0x4068) were each driven here and none of them moves the LUT. */
#define LUT_ADD_DEFAULT ""

static void sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

/* The OUT_CVT's own arithmetic: a right shift that rounds to nearest and breaks ties
 * to EVEN. Modelling it as an arithmetic shift disagrees on every value within half a
 * count, which is a quarter of a dense sweep — enough to read as a broken control. */
static long rte_shift(long v, unsigned s)
{
    long half = 1L << (s - 1), q = v >> s, r = v - (q << s);
    if (r > half || (r == half && (q & 1))) q++;
    return q;
}

static size_t out_index(unsigned surf_elems, unsigned ow, unsigned c,
                        unsigned y, unsigned x)
{
    return (size_t)(c / C2) * surf_elems * C2 + (size_t)C2 * (y * ow + x) + (c % C2);
}

/* Rewrite an already-emitted register. The conv generator writes the whole LUT bank as
 * zeros because every convolution capture leaves it zero; turning the LUT on is a
 * short list of values on top of that, and patching after emission keeps the probe out
 * of the transcribed program. */
static int patch_reg(uint64_t *ops, unsigned n, unsigned reg, uint32_t val)
{
    unsigned i, hits = 0;
    for (i = 0; i < n; i++)
        if ((uint16_t)(ops[i] & 0xFFFFu) == (uint16_t)reg &&
            (uint16_t)(ops[i] >> 48) != 0) {
            ops[i] = (ops[i] & ~(0xFFFFFFFFull << 16)) | ((uint64_t)val << 16);
            hits++;
        }
    return (int)hits;
}

/* The span, as the three registers carry it: the index step is 2^sel, and the two
 * tables cover [le_start, 0] and [0, lo_end]. The vendor never moves any of them —
 * it places the value inside this fixed window with the BS stage's own per-channel
 * multiplier instead — but they are live, and the gate drives them. */
static int     g_sel      = 5;
static int32_t g_le_start = -16384;
static int32_t g_lo_end   =  16384;

/* Is `reg` in the comma list ROCKET_LUT_SKIP? Leaving one member of the enable set at
 * the generator's own value is the leave-one-out over that set — which is worth having
 * only because the whole set was applied FIRST and is known to reach the table. */
static int skipped(unsigned reg)
{
    const char *p = getenv("ROCKET_LUT_SKIP");
    while (p && *p) {
        char *end;
        unsigned long v = strtoul(p, &end, 0);
        if (end == p) break;
        if (v == reg) return 1;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return 0;
}

/* ROCKET_LUT_SET rewrites any already-emitted register after the enable set is on. */
static void apply_set(uint64_t *ops, unsigned n)
{
    const char *p = getenv("ROCKET_LUT_SET");
    while (p && *p) {
        char *end;
        unsigned long reg, val;
        int hits;
        reg = strtoul(p, &end, 0);
        if (end == p || *end != '=') break;
        p = end + 1;
        val = strtoul(p, &end, 0);
        if (end == p) break;
        p = end;
        while (*p == ',' || *p == ' ') p++;
        hits = patch_reg(ops, n, (unsigned)reg, (uint32_t)val);
        printf("  set 0x%04lx = 0x%08lx (%d write%s)\n", reg, val,
               hits, hits == 1 ? "" : "s");
    }
}

/* Every register the vendor's consuming convolution carries that its own bare-conv
 * control does not, applied as a SET rather than one at a time — the fp16 mode on this
 * part is three registers each inert while either of the others is wrong, and a
 * leave-one-out over a set like that returns nothing anywhere. ROCKET_LUT_SKIP then
 * takes members back out once the whole set is known to work.
 *
 * `lo`/`hi` are the table's endpoint values, which the vendor puts in the clamps; this
 * probe puts something outside the table's range there instead, so "below the domain"
 * and "above it" are visible in the output. */
static int enable_lut(uint64_t *ops, unsigned n, int16_t lo, int16_t hi)
{
    unsigned r;
    int miss = 0;
#define PATCH(reg, val)  do { if (!skipped(reg)) \
        miss += patch_reg(ops, n, (reg), (uint32_t)(val)) != 1; } while (0)
    PATCH(0x1004, 0x30);
    PATCH(0x3004, 0x30);
    PATCH(0x4004, 0x30);
    PATCH(0x5004, 0x30);
    /* BN_CFG is NOT part of enabling the LUT, and the vendor's value for it is a
     * trap. Every vendor activation carries 0x20 there — the BN stage ACTIVE — and
     * copying that pins the LUT at the table join whatever the accumulator holds,
     * because BN then multiplies by an operand register no vendor program writes and
     * this register file is not cleared between jobs. Left at the generator's own
     * 0x903, the BN stage is bypassed and the LUT reads the BS output directly, which
     * is where the per-output-channel C already places it. ROCKET_LUT_BN puts the
     * vendor's value back. */
    {
        const char *bn = getenv("ROCKET_LUT_BN");
        if (bn && *bn)
            miss += patch_reg(ops, n, 0x4060,
                              (uint32_t)strtoul(bn, NULL, 0)) != 1;
    }
    PATCH(0x407c, 0x01004140u);       /* EW_CFG: EW_LUT_BYPASS clear            */
    PATCH(0x4108, 0x02000006u);       /* LUT_CFG                                */
    PATCH(0x410c, ((uint32_t)g_sel << 16) | ((uint32_t)g_sel << 8));
    for (r = 0x4110; r <= 0x411c; r += 4)
        PATCH(r, (uint32_t)g_le_start);   /* LE_START, four lanes               */
    for (r = 0x4140; r <= 0x414c; r += 4)
        PATCH(r, (uint32_t)g_lo_end);     /* LO_END,   four lanes               */
    PATCH(0x4188, (uint32_t)((uint16_t)lo) << 16);
    PATCH(0x418c, ((uint32_t)((uint16_t)lo) << 16) | (uint32_t)(uint16_t)lo);
    PATCH(0x4190, (uint32_t)((uint16_t)hi) << 16);
    PATCH(0x4194, ((uint32_t)((uint16_t)hi) << 16) | (uint32_t)(uint16_t)hi);
    PATCH(0x5028, 0x0000001au);       /* NRDMA_CFG                              */
#undef PATCH
    apply_set(ops, n);
    return miss;
}

/* ------------------------------------------------------------------ the harness */

typedef struct {
    int fd;
    rocket_bo scratch, in, w, coeff, out, rc;
    unsigned surf_elems;
    size_t out_bytes;
    uint64_t *lut_ops;     /* RK3576_LUT_TASK_OPS  */
    uint64_t *conv_ops;    /* RK3576_CONV_TASK_OPS */
    uint32_t lut_count, conv_count;
} harness;

static void harness_free(harness *h)
{
    if (h->scratch.ptr) rocket_bo_free(h->fd, &h->scratch);
    if (h->in.ptr)      rocket_bo_free(h->fd, &h->in);
    if (h->w.ptr)       rocket_bo_free(h->fd, &h->w);
    if (h->coeff.ptr)   rocket_bo_free(h->fd, &h->coeff);
    if (h->out.ptr)     rocket_bo_free(h->fd, &h->out);
    if (h->rc.ptr)      rocket_bo_free(h->fd, &h->rc);
    free(h->lut_ops);
    free(h->conv_ops);
}

/* The conv the table feeds: 1x1 over a 16x16 plane, 32 in, 32 out, and only input
 * channel 0 live with a weight of 1, so the accumulator at every pixel is that
 * pixel's own byte. A and C then place it. */
static int harness_init(harness *h, int fd)
{
    size_t in_bytes  = (size_t)(IC / C2) * PLANE * PLANE * C2;
    size_t w_bytes   = (size_t)(OC / 32) * (IC / 32) * 32 * 32;
    size_t coeff_bytes = rocket_rk3576_coeff_bytes(OC);
    int8_t *in_cube = NULL, *w_cube = NULL;
    unsigned n, c;
    int rc = -1;

    memset(h, 0, sizeof *h);
    h->fd = fd;
    h->surf_elems = rocket_rk3576_out_surf_elems(PLANE, PLANE, 0);
    h->out_bytes  = (size_t)(OC / C2) * h->surf_elems * C2;

    h->lut_ops  = calloc(RK3576_LUT_TASK_OPS,  sizeof(uint64_t));
    h->conv_ops = calloc(RK3576_CONV_TASK_OPS, sizeof(uint64_t));
    in_cube = calloc(in_bytes, 1);
    w_cube  = calloc(w_bytes, 1);
    if (!h->lut_ops || !h->conv_ops || !in_cube || !w_cube) goto done;

    /* The scratch BO FIRST, so it owns IOVA 0 — the address the vendor's table-load
     * program stores for its dummy cube. On this stack IOVA 0 is a real buffer. */
    if (rocket_bo_alloc(fd, 4096, &h->scratch) < 0) goto done;
    if (rocket_bo_alloc(fd, in_bytes, &h->in) < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes, &h->w) < 0) goto done;
    if (rocket_bo_alloc(fd, coeff_bytes, &h->coeff) < 0) goto done;
    if (rocket_bo_alloc(fd, h->out_bytes, &h->out) < 0) goto done;
    if (rocket_bo_alloc(fd, (RK3576_LUT_TASK_OPS + RK3576_CONV_TASK_OPS) *
                            sizeof(uint64_t), &h->rc) < 0) goto done;

    for (n = 0; n < NPIX; n++)
        in_cube[feature_data((int)IC, (int)PLANE, (int)PLANE, (int)C2,
                             1, (int)(n / PLANE) + 1, (int)(n % PLANE) + 1)] =
            (int8_t)((int)n - 128);
    for (c = 0; c < OC; c++)
        w_cube[weight_conv_int8((int)OC, (int)IC, 1, 1, (int)c + 1, 1, 1, 1)] = 1;

    rocket_bo_prep(fd, &h->in, 1, 0);
    memcpy(h->in.ptr, in_cube, in_bytes);
    rocket_bo_fini(fd, &h->in);
    rocket_bo_prep(fd, &h->w, 1, 0);
    memcpy(h->w.ptr, w_cube, w_bytes);
    rocket_bo_fini(fd, &h->w);
    rc = 0;
done:
    free(in_cube);
    free(w_cube);
    if (rc) harness_free(h);
    return rc;
}

/* Build the two programs and run them as ONE job: the table lives in the LUT RAM, and
 * a separate submit could take a runtime-PM cycle between the load and the use. */
static int run_pair(harness *h, const int16_t *lo, const int16_t *hi,
                    const int32_t *A, const int16_t *C, int lut_on,
                    int16_t clamp_lo, int16_t clamp_hi, int8_t *out)
{
    lut_load_params_rk3576_t lp = {0};
    conv_params_t p = {0};
    rocket_task_desc task[2];
    uint32_t in_h[4], out_h[1];
    int miss;

    lp.lo = lo; lp.hi = hi;
    lp.scratch_dma = h->scratch.dma_address;
    lp.tasks = h->lut_ops;
    if (gen_lut_load_rk3576(&lp) != 0) {
        printf("  the table-load generator refused\n");
        return -1;
    }
    h->lut_count = lp.task_count;

    rocket_bo_prep(h->fd, &h->coeff, 1, 0);
    if (rocket_rk3576_pack_coeff_perc(h->coeff.ptr,
                                      rocket_rk3576_coeff_bytes(OC),
                                      A, OC, NULL, C, 1) != 0) {
        printf("  the coefficient packer refused\n");
        rocket_bo_fini(h->fd, &h->coeff);
        return -1;
    }
    rocket_bo_fini(h->fd, &h->coeff);

    p.ic = IC; p.ih = PLANE; p.iw = PLANE;
    p.oc = OC; p.oh = PLANE; p.ow = PLANE;
    p.kh = 1;  p.kw = 1;
    p.stride_y = 1; p.stride_x = 1;
    p.ih_full = PLANE; p.oh_full = PLANE;
    p.int8_out = 1;
    /* conv_scale 1/256, so OUT_CVT is a plain >>8 of whatever leaves the LUT. */
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 256.0f;
    p.input_zero_point  = 0x80;
    p.output_zero_point = 0x80;
    p.weight_zero_point = 0x80;
    p.tasks       = h->conv_ops;
    p.input_dma   = h->in.dma_address;
    p.weights_dma = h->w.dma_address;
    p.bias_dma    = h->coeff.dma_address;
    p.output_dma  = h->out.dma_address;

    /* The BN stage sits between the BS output and the LUT, and the two registers that
     * configure its multiply are ones the vendor's programs never write — so on a part
     * whose register file is not cleared between jobs they hold whatever the last job
     * left. On the RK3588 that multiply IS the LUT's index scale. ROCKET_LUT_ADD drives
     * them (and anything else) through the generator's own append hook. */
    if (lut_on) {
        const char *add = getenv("ROCKET_LUT_ADD");
        setenv("ROCKET_RK3576_ADD", add && *add ? add : LUT_ADD_DEFAULT, 1);
    }
    if (gen_conv2d_int8_rk3576(&p) != 0) {
        printf("  the conv generator refused\n");
        unsetenv("ROCKET_RK3576_ADD");
        return -1;
    }
    unsetenv("ROCKET_RK3576_ADD");
    h->conv_count = p.task_count;

    if (lut_on) {
        miss = enable_lut(h->conv_ops, h->conv_count, clamp_lo, clamp_hi);
        if (miss) {
            printf("  %d LUT register(s) were not present in the emitted conv — the "
                   "patch list and the generator have drifted apart\n", miss);
            return -1;
        }
    }

    rocket_bo_prep(h->fd, &h->rc, 1, 0);
    memcpy((uint64_t *)h->rc.ptr, h->lut_ops, h->lut_count * sizeof(uint64_t));
    memcpy((uint64_t *)h->rc.ptr + RK3576_LUT_TASK_OPS, h->conv_ops,
           h->conv_count * sizeof(uint64_t));
    rocket_bo_fini(h->fd, &h->rc);

    /* Bracketed, never a bare memset: dirty CPU lines race the DPU's write DMA. */
    rocket_bo_prep(h->fd, &h->out, 1, 0);
    memset(h->out.ptr, SENTINEL, h->out_bytes);
    rocket_bo_fini(h->fd, &h->out);
    rocket_bo_prep(h->fd, &h->scratch, 1, 0);
    memset(h->scratch.ptr, SENTINEL, 4096);
    rocket_bo_fini(h->fd, &h->scratch);

    task[0].regcmd = h->rc.dma_address;
    task[0].regcmd_count = h->lut_count;
    task[1].regcmd = h->rc.dma_address + RK3576_LUT_TASK_OPS * sizeof(uint64_t);
    task[1].regcmd_count = h->conv_count;

    in_h[0] = h->in.handle; in_h[1] = h->w.handle;
    in_h[2] = h->coeff.handle; in_h[3] = h->rc.handle;
    out_h[0] = h->out.handle;

    if (rocket_submit_tasks(h->fd, task, 2, in_h, 4, out_h, 1) != 0) {
        printf("  submit failed\n");
        return -1;
    }
    if (rocket_bo_prep(h->fd, &h->out, 0, 2000000000ull) < 0) {
        printf("  PREP_BO on the output timed out\n");
        return -1;
    }
    memcpy(out, h->out.ptr, h->out_bytes);
    rocket_bo_fini(h->fd, &h->out);
    return 0;
}

/* ------------------------------------------------------------------ load mode */

/* Does the DPU-only program run at all? Its dummy cube writes 16 bytes, so the
 * scratch BO moving off its sentinel is the positive control — and a faulted job
 * retires cleanly on this part, so the check has to be on the surface. */
static int mode_load(harness *h)
{
    lut_load_params_rk3576_t lp = {0};
    int16_t *lo = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int16_t *hi = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    rocket_task_desc task[1];
    uint32_t in_h[1], out_h[1];
    unsigned i, moved = 0;
    int rc = 1;

    if (!lo || !hi) goto done;
    for (i = 0; i < RK3576_LUT_ENTRIES; i++) {
        lo[i] = (int16_t)(i * RAMP_STEP);
        hi[i] = (int16_t)((RK3576_LUT_ENTRIES + i) * RAMP_STEP);
    }

    lp.lo = lo; lp.hi = hi;
    lp.scratch_dma = h->scratch.dma_address;
    lp.tasks = h->lut_ops;
    if (gen_lut_load_rk3576(&lp) != 0) { printf("generator refused\n"); goto done; }

    printf("load: the table-load program is %u words (the vendor's is 1117 plus a "
           "4-word trailer)\n", lp.task_count);
    printf("      scratch BO at IOVA 0x%llx%s\n",
           (unsigned long long)h->scratch.dma_address,
           h->scratch.dma_address ? "" : " (IOVA 0, as the capture stores)");

    rocket_bo_prep(h->fd, &h->rc, 1, 0);
    memcpy(h->rc.ptr, h->lut_ops, lp.task_count * sizeof(uint64_t));
    rocket_bo_fini(h->fd, &h->rc);
    rocket_bo_prep(h->fd, &h->scratch, 1, 0);
    memset(h->scratch.ptr, SENTINEL, 4096);
    rocket_bo_fini(h->fd, &h->scratch);

    task[0].regcmd = h->rc.dma_address;
    task[0].regcmd_count = lp.task_count;
    in_h[0] = h->rc.handle;
    out_h[0] = h->scratch.handle;
    if (rocket_submit_tasks(h->fd, task, 1, in_h, 1, out_h, 1) != 0) {
        printf("      submit failed\n"); goto done;
    }
    if (rocket_bo_prep(h->fd, &h->scratch, 0, 2000000000ull) < 0) {
        printf("      PREP_BO timed out\n"); goto done;
    }
    for (i = 0; i < 64; i++)
        if (((const uint8_t *)h->scratch.ptr)[i] != SENTINEL) moved++;
    printf("      the dummy cube wrote %u of its first 64 bytes\n", moved);
    printf("      %s\n", moved ? "the DPU-only program RUNS"
                               : "nothing written — the program did not execute, or "
                                 "its dummy cube writes nowhere");
    rc = moved ? 0 : 1;
done:
    free(lo); free(hi);
    return rc;
}

/* ------------------------------------------------------------------ map mode */

/* The per-channel placement of the sweep. C spans fifteen binades so that whatever
 * range the table covers, some channel straddles its edge; A breaks the symmetry of
 * the accumulator about zero, which a C-only sweep cannot see. */
static void sweep_schedule(int32_t *A, int16_t *C)
{
    unsigned c;
    for (c = 0; c < 30; c++) {
        C[c] = (int16_t)(1 << (c / 2));
        A[c] = (c & 1) ? 64 : 0;
    }
    C[30] = 1; A[30] =  65536;
    C[31] = 1; A[31] = -65536;
}

static int mode_map(harness *h)
{
    int16_t *lo = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int16_t *hi = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int8_t  *out = calloc(h->out_bytes, 1);
    int8_t  *ctl = calloc(h->out_bytes, 1);
    int32_t A[OC]; int16_t C[OC];
    unsigned c, i;
    int rc = 1;

    if (!lo || !hi || !out || !ctl) goto done;
    for (i = 0; i < RK3576_LUT_ENTRIES; i++) {
        lo[i] = (int16_t)(i * RAMP_STEP);
        hi[i] = (int16_t)((RK3576_LUT_ENTRIES + i) * RAMP_STEP);
    }
    sweep_schedule(A, C);

    /* The control first: the same conv with the LUT bank left as the generator emits
     * it. Its output is the BS result requantised, which is what says the sweep's
     * arithmetic is what this claims before any of it is read as a LUT index. */
    printf("map: control — the same convolution with the LUT off\n");
    if (run_pair(h, lo, hi, A, C, 0, CLAMP_LO, CLAMP_HI, ctl) != 0) goto done;
    {
        int bad = 0, shown = 0;
        for (c = 0; c < OC; c++)
            for (i = 0; i < NPIX; i++) {
                long v = ((long)((int)i - 128) + A[c]) * C[c];
                long q = rte_shift(v, 8);              /* OUT_CVT, before clamping */
                int want = q > 127 ? 127 : (q < -128 ? -128 : (int)q);
                int have = ctl[out_index(h->surf_elems, PLANE, c,
                                         i / PLANE, i % PLANE)];
                if (have != want) {
                    if (++bad, shown < 5) {
                        printf("     oc=%-2u p=%-4d  (acc+A)*C=%-12ld want %4d got %4d\n",
                               c, (int)i - 128, v, want, have);
                        shown++;
                    }
                }
            }
        printf("     %d of %u disagree with (acc + A[oc]) * C[oc] >> 8\n",
               bad, (unsigned)(OC * NPIX));
        if (bad > (int)(OC * NPIX) / 2) {
            printf("     the control does not hold — the sweep's own arithmetic is "
                   "wrong, so nothing read through the LUT would mean anything\n");
            goto done;
        }
    }

    sleep_ms(200);
    printf("map: the LUT on, a ramp of %d per index, clamps %d / %d\n",
           RAMP_STEP, CLAMP_LO, CLAMP_HI);
    if (run_pair(h, lo, hi, A, C, 1, CLAMP_LO, CLAMP_HI, out) != 0) goto done;

    {
        int moved = 0;
        for (i = 0; i < h->out_bytes; i++)
            if ((uint8_t)out[i] != SENTINEL) { moved = 1; break; }
        if (!moved) {
            printf("     nothing written. The LUT-enabled program did not compute — "
                   "an untouched surface is never evidence about the encoding\n");
            goto done;
        }
    }

    /* What the surface actually holds, before any of it is read as a map. A LUT that
     * never loaded and a LUT whose whole domain lands on one entry are both a constant
     * surface, and a summary that only fits a slope cannot tell either from a working
     * one. */
    {
        int seen[256]; unsigned k, distinct = 0;
        memset(seen, 0, sizeof seen);
        for (c = 0; c < OC; c++)
            for (i = 0; i < NPIX; i++)
                seen[(unsigned char)out[out_index(h->surf_elems, PLANE, c,
                                                  i / PLANE, i % PLANE)]]++;
        printf("     distinct output values:");
        for (k = 0; k < 256; k++)
            if (seen[k]) {
                if (distinct < 12)
                    printf(" %d(x%d)", (int)(int8_t)k, seen[k]);
                distinct++;
            }
        printf("%s  [%u distinct]\n", distinct > 12 ? " ..." : "", distinct);
        printf("     oc=0  p=-128..-113:");
        for (i = 0; i < 16; i++)
            printf(" %d", out[out_index(h->surf_elems, PLANE, 0, i / PLANE, i % PLANE)]);
        printf("\n     oc=14 p=-128..-113:");
        for (i = 0; i < 16; i++)
            printf(" %d", out[out_index(h->surf_elems, PLANE, 14, i / PLANE, i % PLANE)]);
        printf("\n     the same pixels with the LUT OFF, oc=0:");
        for (i = 0; i < 16; i++)
            printf(" %d", ctl[out_index(h->surf_elems, PLANE, 0, i / PLANE, i % PLANE)]);
        printf("\n");
    }

    /* Per channel: where the readout leaves the low clamp, where it reaches the high
     * one, and how the index moves with the value in between. */
    printf("\n  %-3s %-7s %-8s   %-13s %-13s  %s\n",
           "oc", "C", "A", "first in-domain", "last in-domain", "index per unit");
    for (c = 0; c < OC; c++) {
        long v_first = 0, v_last = 0, sx = 0, sy = 0, n = 0;
        double sxx = 0, sxy = 0, mx, my, slope = 0, icpt = 0, maxres = 0;
        int have_first = 0, ndiff = 0;
        int prev = -1000;
        for (i = 0; i < NPIX; i++) {
            long v = ((long)((int)i - 128) + A[c]) * C[c];
            int o = out[out_index(h->surf_elems, PLANE, c, i / PLANE, i % PLANE)];
            if (o == -128 || o == 127) continue;       /* clamped: outside the table */
            if (!have_first) { v_first = v; have_first = 1; }
            v_last = v;
            if (o != prev) { ndiff++; prev = o; }
            sx += v; sy += o; n++;
        }
        if (n >= 2) {
            mx = (double)sx / (double)n; my = (double)sy / (double)n;
            for (i = 0; i < NPIX; i++) {
                long v = ((long)((int)i - 128) + A[c]) * C[c];
                int o = out[out_index(h->surf_elems, PLANE, c, i / PLANE, i % PLANE)];
                if (o == -128 || o == 127) continue;
                sxx += ((double)v - mx) * ((double)v - mx);
                sxy += ((double)v - mx) * ((double)o - my);
            }
            if (sxx > 0) { slope = sxy / sxx; icpt = my - slope * mx; }
            for (i = 0; i < NPIX; i++) {
                long v = ((long)((int)i - 128) + A[c]) * C[c];
                int o = out[out_index(h->surf_elems, PLANE, c, i / PLANE, i % PLANE)];
                double r;
                if (o == -128 || o == 127) continue;
                r = (double)o - (slope * (double)v + icpt);
                if (r < 0) r = -r;
                if (r > maxres) maxres = r;
            }
        }
        printf("  %-3u %-7d %-8d   %-13ld %-13ld  ", c, (int)C[c], A[c],
               n ? v_first : 0, n ? v_last : 0);
        if (n < 2) {
            printf("(%ld sample%s in domain)\n", n, n == 1 ? "" : "s");
        } else {
            /* One output count is 256/RAMP_STEP indices, so the slope in indices per
             * datapath unit is that times the fitted slope. */
            printf("%.6g   [%ld pts, %d levels, max residual %.2f]\n",
                   slope * 256.0 / (double)RAMP_STEP, n, ndiff, maxres);
        }
    }
    rc = 0;
done:
    free(lo); free(hi); free(out); free(ctl);
    return rc;
}

/* ------------------------------------------------------------------ tables mode */

/* A constant surface under one table says nothing on its own: a table that never
 * loaded and a table whose whole domain lands on one entry look identical. Vary the
 * TABLE with the sweep held fixed and the two separate — a readout that tracks the
 * entry at some index is a loaded table being indexed, and one that does not is a
 * table that is not there. */
typedef struct { const char *name; int kind; int arg; } table_case;

static void build_table(const table_case *tc, int16_t *lo, int16_t *hi)
{
    unsigned i;
    for (i = 0; i < RK3576_LUT_ENTRIES; i++) {
        unsigned il = i, ih = RK3576_LUT_ENTRIES + i;
        switch (tc->kind) {
        case 0:  /* a ramp of `arg` per index across both tables */
            lo[i] = (int16_t)(il * (unsigned)tc->arg);
            hi[i] = (int16_t)(ih * (unsigned)tc->arg);
            break;
        case 1:  /* one constant everywhere */
            lo[i] = hi[i] = (int16_t)tc->arg;
            break;
        case 2:  /* a step at index `arg` */
            lo[i] = (int16_t)(il < (unsigned)tc->arg ? 0 : 32000);
            hi[i] = (int16_t)(ih < (unsigned)tc->arg ? 0 : 32000);
            break;
        case 3:  /* the low table one value, the high table another */
            lo[i] = (int16_t)tc->arg;
            hi[i] = (int16_t)(-tc->arg);
            break;
        }
    }
}

static const table_case TABLES[] = {
    { "ramp 31/index",     0, 31    },
    { "ramp 17/index",     0, 17    },
    { "flat 8192",         1, 8192  },
    { "flat 16384",        1, 16384 },
    { "step at 512",       2, 512   },
    { "step at 1",         2, 1     },
    { "step at 1025",      2, 1025  },
    { "lo +8192, hi -8192",3, 8192  },
};

static int mode_tables(harness *h)
{
    int16_t *lo = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int16_t *hi = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int8_t  *out = calloc(h->out_bytes, 1);
    int32_t A[OC]; int16_t C[OC];
    unsigned t, c, i;
    int rc = 1;

    if (!lo || !hi || !out) goto done;
    sweep_schedule(A, C);
    printf("tables: the same sweep under different tables. A readout that tracks the\n"
           "        table is a loaded table; one that does not is not there.\n");
    for (t = 0; t < sizeof TABLES / sizeof *TABLES; t++) {
        int seen[256]; unsigned distinct = 0, shown = 0;
        int varies_pixel = 0, varies_channel = 0;
        build_table(&TABLES[t], lo, hi);
        if (run_pair(h, lo, hi, A, C, 1, 0, 32000, out) != 0) goto done;
        memset(seen, 0, sizeof seen);
        for (c = 0; c < OC; c++)
            for (i = 0; i < NPIX; i++)
                seen[(unsigned char)out[out_index(h->surf_elems, PLANE, c,
                                                  i / PLANE, i % PLANE)]]++;
        for (i = 1; i < NPIX; i++)
            if (out[out_index(h->surf_elems, PLANE, 0, i / PLANE, i % PLANE)] !=
                out[out_index(h->surf_elems, PLANE, 0, 0, 0)]) varies_pixel = 1;
        for (c = 1; c < OC; c++)
            if (out[out_index(h->surf_elems, PLANE, c, 0, 0)] !=
                out[out_index(h->surf_elems, PLANE, 0, 0, 0)]) varies_channel = 1;
        printf("  %-20s ->", TABLES[t].name);
        for (i = 0; i < 256; i++)
            if (seen[i]) {
                if (shown < 8) printf(" %d(x%d)", (int)(int8_t)i, seen[i]);
                shown++; distinct++;
            }
        printf("%s   [%u distinct, varies with pixel: %s, with channel: %s]\n",
               distinct > 8 ? " ..." : "", distinct,
               varies_pixel ? "yes" : "no", varies_channel ? "yes" : "no");
        sleep_ms(150);
    }
    rc = 0;
done:
    free(lo); free(hi); free(out);
    return rc;
}

/* ------------------------------------------------------------------ one mode */

/* One ramp-table run, one line of output, everything else from the environment. The
 * ramp makes the SELECTED INDEX readable directly — a table entry is 31 per index and
 * the OUT_CVT is a >>8, so the readout times 256/31 is the index — which means a
 * register that moves the index shows up even while the surface is still constant.
 * Driven from a shell loop, this is the register hunt. */
static int mode_one(harness *h)
{
    int16_t *lo = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int16_t *hi = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int8_t  *out = calloc(h->out_bytes, 1);
    int32_t A[OC]; int16_t C[OC];
    unsigned c, i;
    int rc = 1, mn = 127, mx = -128, distinct = 0;
    int seen[256];

    if (!lo || !hi || !out) goto done;
    sweep_schedule(A, C);
    for (i = 0; i < RK3576_LUT_ENTRIES; i++) {
        lo[i] = (int16_t)(i * RAMP_STEP);
        hi[i] = (int16_t)((RK3576_LUT_ENTRIES + i) * RAMP_STEP);
    }
    if (run_pair(h, lo, hi, A, C, 1, CLAMP_LO, CLAMP_HI, out) != 0) goto done;
    memset(seen, 0, sizeof seen);
    for (c = 0; c < OC; c++)
        for (i = 0; i < NPIX; i++) {
            int o = out[out_index(h->surf_elems, PLANE, c, i / PLANE, i % PLANE)];
            if (o < mn) mn = o;
            if (o > mx) mx = o;
            seen[(unsigned char)o]++;
        }
    for (i = 0; i < 256; i++) if (seen[i]) distinct++;
    printf("ONE add=[%s] out %d..%d  index %d..%d  distinct %d\n",
           getenv("ROCKET_LUT_ADD") ? getenv("ROCKET_LUT_ADD") : LUT_ADD_DEFAULT,
           mn, mx, (int)(mn * 256 / RAMP_STEP), (int)(mx * 256 / RAMP_STEP), distinct);
    rc = distinct > 1 ? 0 : 1;
done:
    free(lo); free(hi); free(out);
    return rc;
}


/* ------------------------------------------------------------------ gate mode */

/* What the part should compute, given the three registers that carry the span and the
 * two tables. LE covers [le_start, 0] and LO covers [0, lo_end], each 512 segments of
 * 2^sel datapath units, and the hardware interpolates linearly between entries. */
static double lut_model(long v, const int16_t *le, const int16_t *lo,
                        double clamp_lo, double clamp_hi)
{
    double step = (double)(1 << g_sel), pos;
    const int16_t *t;
    int i;
    /* HALF-OPEN AT THE TOP. `v == LO_END` takes the overflow clamp, not the last
     * table entry — measured at four different C on the vendor's own span and again
     * at each of three others, every disagreement in the whole sweep sitting on that
     * one value. `v == LE_START` is in domain and reads LE[0]. */
    if (v <  g_le_start) return clamp_lo;
    if (v >= g_lo_end)   return clamp_hi;
    if (v < 0) { t = le; pos = (double)(v - g_le_start) / step; }
    else       { t = lo; pos = (double)v / step; }
    if (pos < 0) pos = 0;
    if (pos > RK3576_LUT_ENTRIES - 1) pos = RK3576_LUT_ENTRIES - 1;
    i = (int)pos;
    if (i >= RK3576_LUT_ENTRIES - 1) return t[RK3576_LUT_ENTRIES - 1];
    return t[i] + (pos - i) * (t[i + 1] - t[i]);
}

/* C and A per output channel: fifteen windows over the domain at four different
 * sample steps, two that straddle each edge, and four at step 1 so a whole index
 * interval is walked one datapath unit at a time — which is the only way the
 * interpolation is visible at all. */
static void gate_schedule(int32_t *A, int16_t *C)
{
    static const int32_t sched[OC][2] = {
        {   32, -384 }, {   32, -128 }, {   32,  128 }, {   32,  384 },
        {    1,    0 }, {    1, 4096 }, {    1,-4096 }, {    1, 8192 },
        {    4,    0 }, {    4, 2048 }, {    4,-2048 }, {    4, 3000 },
        {    1,16300 }, {    1,-16300}, {  128,    0 }, {  128,   64 },
        {   64,    0 }, {   64,  128 }, {   16,    0 }, {   16,  512 },
        {    8,    0 }, {    8, 1024 }, {    2,    0 }, {    2, 4096 },
        {  256,    0 }, {  512,    0 }, {    1,  100 }, {    1, -100 },
        {   64, -128 }, {   16, -512 }, {   32,    0 }, {    1,12345 },
    };
    unsigned c;
    for (c = 0; c < OC; c++) { C[c] = (int16_t)sched[c][0]; A[c] = sched[c][1]; }
}

typedef struct { const char *name; int sel; int32_t le_start, lo_end; } span_case;

static const span_case SPANS[] = {
    { "the vendor's span", 5, -16384,  16384 },
    { "half the step",     4,  -8192,   8192 },
    { "twice the step",    6, -32768,  32768 },
    { "asymmetric",        5,  -8192,  16384 },
};

static int mode_gate(harness *h)
{
    int16_t *le = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int16_t *lo = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int8_t  *out = calloc(h->out_bytes, 1);
    int32_t A[OC]; int16_t C[OC];
    unsigned mul, shift, t, c, i;
    int rc = 1, fails = 0;

    if (!le || !lo || !out) goto done;
    gate_schedule(A, C);
    rocket_rk3576_requant_params(1.0f / 256.0f, &mul, &shift);
    /* A ramp is the one table whose interpolation the model can state exactly: any
     * linear function is its own interpolant, so a residual is the hardware's and not
     * the model's. */
    for (i = 0; i < RK3576_LUT_ENTRIES; i++) {
        le[i] = (int16_t)(i * RAMP_STEP);
        lo[i] = (int16_t)((RK3576_LUT_ENTRIES + i) * RAMP_STEP);
    }
    printf("gate: index = (value - LE_START) / 2^sel, against the part\n");
    printf("      OUT_CVT mul %u shift %u, table ramp %d per index\n",
           mul, shift, RAMP_STEP);

    for (t = 0; t < sizeof SPANS / sizeof *SPANS; t++) {
        int worst = 0, over1 = 0, n = 0, shown = 0;
        g_sel = SPANS[t].sel;
        g_le_start = SPANS[t].le_start;
        g_lo_end = SPANS[t].lo_end;
        if (run_pair(h, le, lo, A, C, 1, CLAMP_LO, CLAMP_HI, out) != 0) goto done;
        for (c = 0; c < OC; c++)
            for (i = 0; i < NPIX; i++) {
                long v = ((long)((int)i - 128) + A[c]) * C[c];
                double m = lut_model(v, le, lo, (double)CLAMP_LO, (double)CLAMP_HI);
                long q = rte_shift((long)(m < 0 ? m - 0.5 : m + 0.5) * (long)mul, shift);
                int want = q > 127 ? 127 : (q < -128 ? -128 : (int)q);
                int have = out[out_index(h->surf_elems, PLANE, c,
                                         i / PLANE, i % PLANE)];
                int d = have - want; if (d < 0) d = -d;
                if (d > worst) worst = d;
                if (d > 1) {
                    over1++;
                    if (shown < 6) {
                        printf("     oc=%-2u C=%-5d A=%-7d v=%-8ld want %4d got %4d\n",
                               c, (int)C[c], A[c], v, want, have);
                        shown++;
                    }
                }
                n++;
            }
        printf("  %-18s sel=%d LE_START=%-7d LO_END=%-6d  max |diff| %d, "
               "%d of %d past one count  %s\n",
               SPANS[t].name, g_sel, g_le_start, g_lo_end, worst, over1, n,
               over1 ? "FAIL" : "PASS");
        if (over1) fails++;
        sleep_ms(150);
    }
    printf("== %d of %d spans explained by the model ==\n",
           (int)(sizeof SPANS / sizeof *SPANS) - fails,
           (int)(sizeof SPANS / sizeof *SPANS));
    rc = fails ? 1 : 0;
done:
    free(le); free(lo); free(out);
    return rc;
}


/* ------------------------------------------------------------------ interp mode */

/* Does the hardware INTERPOLATE between entries, or step? The gate cannot tell: at 31
 * per index a fractional position is worth an eighth of an output count. Put the whole
 * table's range into ONE interval — every entry below index T at zero and every entry
 * from T up at the top — and walk the datapath one unit at a time across it. A step
 * gives two values and an interpolation gives 2^sel of them.
 */
static int mode_interp(harness *h)
{
    const unsigned T = 600;                       /* in the LO table: index 87 */
    long centre = (long)(T - RK3576_LUT_ENTRIES) * 32;
    int16_t *le = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int16_t *lo = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int8_t  *out = calloc(h->out_bytes, 1);
    int32_t A[OC]; int16_t C[OC];
    unsigned c, i;
    int rc = 1, levels = 0, prev = -1000;

    if (!le || !lo || !out) goto done;
    for (i = 0; i < RK3576_LUT_ENTRIES; i++) {
        le[i] = (int16_t)(i < T ? 0 : 32000);
        lo[i] = (int16_t)(RK3576_LUT_ENTRIES + i < T ? 0 : 32000);
    }
    for (c = 0; c < OC; c++) { C[c] = 1; A[c] = (int32_t)centre; }
    g_sel = 5; g_le_start = -16384; g_lo_end = 16384;
    if (run_pair(h, le, lo, A, C, 1, 0, 32000, out) != 0) goto done;

    printf("interp: a step table at index %u (value %ld), walked one unit at a time\n",
           T, centre);
    for (i = 0; i < NPIX; i++) {
        long v = (long)((int)i - 128) + centre;
        int o = out[out_index(h->surf_elems, PLANE, 0, i / PLANE, i % PLANE)];
        if (v >= centre - 40 && v <= centre + 8)
            printf("   v=%-6ld out=%d\n", v, o);
        if (o != prev) { levels++; prev = o; }
    }
    printf("   %d distinct runs across the transition — a step would give 2, an\n"
           "   interpolation over one interval gives up to %d\n", levels, 32);
    rc = 0;
done:
    free(le); free(lo); free(out);
    return rc;
}

/* ------------------------------------------------------------------ edge mode */

/* A step table: every entry below T is the low clamp's value and every entry from T up
 * is the high one. Whatever the interpolation does, the readout's transition brackets
 * exactly one interval — so the value at which it moves gives the interval's position
 * and the width of the transition gives the step. Immune to the resolution the ramp
 * readout loses to eight bits of output. */
static int mode_edge(harness *h)
{
    static const unsigned TS[] = { 1, 128, 256, 512, 513, 640, 1024 };
    int16_t *lo = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int16_t *hi = calloc(RK3576_LUT_ENTRIES, sizeof(int16_t));
    int8_t  *out = calloc(h->out_bytes, 1);
    int32_t A[OC]; int16_t C[OC];
    unsigned t, c, i;
    int rc = 1;

    if (!lo || !hi || !out) goto done;
    sweep_schedule(A, C);

    printf("edge: a step table, the value at which the readout crosses index T\n");
    for (t = 0; t < sizeof TS / sizeof *TS; t++) {
        unsigned T = TS[t];
        for (i = 0; i < RK3576_LUT_ENTRIES; i++) {
            lo[i] = (int16_t)(i < T ? 0 : 32767);
            hi[i] = (int16_t)(RK3576_LUT_ENTRIES + i < T ? 0 : 32767);
        }
        if (run_pair(h, lo, hi, A, C, 1, 0, 32767, out) != 0) goto done;
        printf("  T=%-5u", T);
        for (c = 0; c < OC; c++) {
            long v_lo = 0, v_hi = 0;
            int seen_lo = 0, seen_hi = 0;
            for (i = 0; i < NPIX; i++) {
                long v = ((long)((int)i - 128) + A[c]) * C[c];
                int o = out[out_index(h->surf_elems, PLANE, c, i / PLANE, i % PLANE)];
                if (o <= 0 && !seen_hi) { v_lo = v; seen_lo = 1; }
                if (o >= 127 && !seen_hi) { v_hi = v; seen_hi = 1; }
            }
            if (seen_lo && seen_hi && C[c] <= 4096)
                printf(" c%u:(%ld,%ld]", c, v_lo, v_hi);
        }
        printf("\n");
        sleep_ms(150);
    }
    rc = 0;
done:
    free(lo); free(hi); free(out);
    return rc;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *mode = argc > 1 ? argv[1] : "gate";
    harness h;
    int fd, rc;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_lut_probe: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_lut_probe: no NPU device — skipping\n"); return 2; }
    if (harness_init(&h, fd) != 0) {
        printf("rk3576_lut_probe: could not build the harness\n");
        rocket_close(fd);
        return 1;
    }

    if (!strcmp(mode, "one"))         rc = mode_one(&h);
    else if (!strcmp(mode, "gate"))   rc = mode_gate(&h);
    else if (!strcmp(mode, "interp")) rc = mode_interp(&h);
    else if (!strcmp(mode, "map"))    rc = mode_map(&h);
    else if (!strcmp(mode, "tables")) rc = mode_tables(&h);
    else if (!strcmp(mode, "edge"))   rc = mode_edge(&h);
    else                            rc = mode_load(&h);

    harness_free(&h);
    rocket_close(fd);
    return rc;
}
