// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_pool_probe.c — make the PPU write, and gate what it writes.
 *
 * The pooling program is decoded from manufactured vendor captures down to one
 * register: the DESTINATION BASE ADDRESS. Every capture stores zero for it, exactly as
 * it does for the feature, weight, output and bias bases, because the vendor runtime
 * patches those at load time and a `.rknn`-derived program is the STORED register set.
 * So the one thing a capture cannot carry is the one thing left, and it has to be read
 * off the part.
 *
 * Five PPU registers read zero in every capture — 0x6054, 0x6058, 0x605C, 0x6070 and
 * 0x60DC — and one of them is it. The probe drives them:
 *
 *   `sweep`  each candidate in turn, then all five at once. What is being asked is
 *            only "did the output BO change from its sentinel", so a candidate that
 *            is not an address costs a submit and nothing else. All-five FIRST is
 *            deliberate: a one-at-a-time sweep cannot see a condition of two, and on
 *            this part that has already cost one decode (the fp16 mode is three
 *            registers, each inert while either of the others is wrong).
 *
 *   `gate`   with the register settled, the shape table against a CPU model.
 *
 *   `lib`    the LIBRARY entry over row-major tensors, each shape run twice — once
 *            with the source surface stride the vendor's programs carry (round4 of
 *            the plane) and once with the plane's own element count. The second is
 *            what a CUBE-IN join hands PPU_RDMA, because a direct convolution's output
 *            surface stride is `ow*oh` exactly, so this is the gate that says whether
 *            that join is available at a plane whose element count is not a multiple
 *            of four. Every shape also runs prepacked twice, since the second call is
 *            the first to reuse a held BO.
 *
 * A CUBE-IN JOIN IS NOT AN ALIGNMENT ASSUMPTION. It was written before the stride
 * question was asked, on the belief that the producer's surface would be round4 too;
 * it is not, and a wrong stride computes a full, plausible surface rather than
 * faulting. Gate the stride, not the plumbing.
 *
 * The output BO is stamped with a sentinel through PREP_BO/FINI_BO rather than a bare
 * memset: dirty CPU lines race the DPU's write DMA and the writeback lands on top of
 * the result. And a faulted job retires cleanly at every layer on this part, so an
 * untouched BO is evidence that nothing wrote — never evidence about the encoding.
 *
 *   `bound`  the two axes that could carry a capacity limit — the PLANE and the CHANNEL
 *            count — over a window with a leading pad and no trailing one, which is the
 *            shape a classifier's first pool has and which nothing in `gate` or `lib`
 *            reaches. The plan function carries no capacity bound at all, so what a
 *            plane too large does is a question worth asking of the part.
 *
 *   `pad`    the PAD NIBBLES, one axis at a time. `0x6040` packs four four-bit pads and
 *            every vendor pool is padded symmetrically, so a capture cannot say which
 *            nibble is which edge — and the wrong assignment computes a full, plausible
 *            surface. This runs a single line with a unit kernel on the other axis and
 *            scores the part against the window grid each candidate leading pad implies.
 *            Measured: bits [3:0] are LEFT and bits [7:4] are TOP.
 *
 *   `avg`    a padded average pool run many times over one input, with every failing
 *            iteration scored against four candidate divisor functions. What it reads
 *            out is WHICH function the part evaluated when it evaluated the wrong one,
 *            which a wrong-element count cannot say. It ASSERTS as well: the library
 *            checks the positions where the divisor lag can show and redoes the submit,
 *            so a wrong iteration here is one that check missed.
 *            ROCKET_RK3576_POOL_LAGCHECK=0 turns the check off and the raw rate comes
 *            back — about 1% of calls at these shapes, every occurrence explained by
 *            the previous window's divisor and by nothing else.
 *
 *   `rate`   the same hazard's RATE against the GAP between submits, which is the one
 *            axis that differs between this probe's ~1% and the graph's ~30%. The
 *            library's check is turned off inside it, so the raw rate comes back.
 *
 * Usage:  rk3576_pool_probe [sweep|gate|lib|split|bound|rect|pad|avg|rate]  (default: sweep)
 * Env:    ROCKET_POOLB_LEAD / ROCKET_POOLB_TRAIL   the pads `bound` uses (1 and 0)
 *         ROCKET_POOLP_OTHER                       `pad`'s non-swept extent (4)
 *         ROCKET_POOLA_N                           `avg`'s iterations a cell (30)
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
#include "rocket_pool.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"

#define C2       16u
#define SENTINEL 0xAA

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);

static unsigned round4(unsigned v) { return (v + 3u) & ~3u; }

static size_t out_index(unsigned surf_elems, unsigned ow, unsigned c,
                        unsigned y, unsigned x)
{
    return (size_t)(c / C2) * surf_elems * C2 + (size_t)C2 * (y * ow + x) + (c % C2);
}

static void sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

typedef struct {
    const char *name;
    unsigned c, iw, ih, k, stride, pad;
    unsigned mode;
} pool_case;

/* What the part should compute, on the CPU. int8 in, int8 out, no requant in the
 * path — the PPU is a window reduction and nothing else. */
static int pool_ref(const pool_case *pc, const int8_t *in, unsigned ci,
                    unsigned oy, unsigned ox, unsigned ow, unsigned oh)
{
    long best = -1000, sum = 0;
    unsigned n, ky, kx, dw = pc->k, dh = pc->k;
    (void)ow; (void)oh;
    /* The two per-axis divisors. They are the window unless the pad is excluded, and the
     * taps that landed inside the plane when it is — kept as a PAIR because the PPU's
     * reciprocal is per axis, so whether a tie survives is a question about each. */
    if (pc->mode == ROCKET_RK3576_POOL_AVG_NOPAD) {
        long lo, hi;
        lo = (long)(ox * pc->stride) - (long)pc->pad; hi = lo + (long)pc->k;
        if (lo < 0) lo = 0; if (hi > (long)pc->iw) hi = (long)pc->iw;
        dw = hi > lo ? (unsigned)(hi - lo) : 0u;
        lo = (long)(oy * pc->stride) - (long)pc->pad; hi = lo + (long)pc->k;
        if (lo < 0) lo = 0; if (hi > (long)pc->ih) hi = (long)pc->ih;
        dh = hi > lo ? (unsigned)(hi - lo) : 0u;
    }
    n = dw * dh;
    for (ky = 0; ky < pc->k; ky++) {
        for (kx = 0; kx < pc->k; kx++) {
            long iy = (long)(oy * pc->stride + ky) - (long)pc->pad;
            long ix = (long)(ox * pc->stride + kx) - (long)pc->pad;
            int v;
            if (iy < 0 || ix < 0 || iy >= (long)pc->ih || ix >= (long)pc->iw) {
                if (pc->mode == ROCKET_RK3576_POOL_MAX) v = -128;
                else if (pc->mode == ROCKET_RK3576_POOL_AVG) v = 0;
                else continue;                       /* AVG_NOPAD drops the tap */
            } else {
                v = in[((size_t)ci * pc->ih + (size_t)iy) * pc->iw + (size_t)ix];
            }
            if (v > best) best = v;
            sum += v;
        }
    }
    if (pc->mode == ROCKET_RK3576_POOL_MAX) return (int)best;
    if (!n) return 0;
    /* AN EXACT TIE BREAKS TO EVEN WHEN THE RECIPROCAL IS EXACT AND TOWARD ZERO WHEN IT IS
     * TRUNCATED. The PPU multiplies by 0x10000/dw and 0x10000/dh, both floored, so unless
     * each divisor divides 65536 the product is a hair under 1/n and a sum landing exactly
     * on a half falls to the smaller magnitude. Against round-half-away-from-zero a k2
     * average disagrees on one output in eight — sum = +/-2 (mod 4), half of which land on
     * an odd quotient — which is the 254 of 2048 the first run of this gate reported. */
    {
        long half = (long)n / 2, q, r;
        q = sum >= 0 ? (sum + half) / (long)n
                     : -(((-sum) + half) / (long)n);
        /* An ODD window has no exact half, so there is no tie to break — testing the
         * residue against n/2 there fires on an ordinary value and moves it. */
        if ((n & 1u) == 0u) {
            int exact_recip = (0x10000u % dw) == 0u && (0x10000u % dh) == 0u;
            r = sum - q * (long)n;
            if ((r == half || r == -half) && (!exact_recip || (q & 1)))
                q += (sum >= 0) ? -1 : 1;
        }
        return (int)q;
    }
}

/* Build, submit, and report whether the output BO moved off its sentinel. `diffs` is
 * filled with the mismatch count against the CPU model when `check` is set. */
/* Where the last checked run's wrong outputs were. A border-only failure is a statement
 * about the pad taps; an interior one is a statement about the geometry, and a total
 * cannot tell them apart. */
static int WRONG_BORDER, WRONG_INTERIOR, MAXD_BORDER, MAXD_INTERIOR;

static int run_pool(int fd, const pool_case *pc, unsigned dst_reg, int check,
                    int *wrote, int *diffs, int *maxdiff, int verbose)
{
    unsigned iw = pc->iw, ih = pc->ih, c = pc->c;
    unsigned ow = (iw + 2 * pc->pad - pc->k) / pc->stride + 1;
    unsigned oh = (ih + 2 * pc->pad - pc->k) / pc->stride + 1;
    unsigned creg = ((c + 15u) / 16u) * 16u;
    unsigned in_surf = round4(iw * ih), out_surf = round4(ow * oh);
    size_t in_bytes  = (size_t)(creg / C2) * in_surf * C2;
    size_t out_bytes = (size_t)(creg / C2) * out_surf * C2;
    rocket_bo bo_in = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_POOL_TASK_OPS] = {0};
    uint32_t in_h[2], out_h[1];
    pool_params_rk3576_t p = {0};
    int8_t *src = NULL;
    unsigned i, y, x, ci;
    int rc = -1;

    *wrote = 0; if (diffs) *diffs = 0; if (maxdiff) *maxdiff = 0;

    src = malloc((size_t)c * ih * iw);
    if (!src) goto done;
    {
        unsigned seed = 0x9E3779B9u ^ (c * 31 + iw * 7 + ih * 3 + pc->k);
        for (i = 0; i < (unsigned)((size_t)c * ih * iw); i++) {
            seed = seed * 1103515245u + 12345u;
            src[i] = (int8_t)((int)((seed >> 16) % 251u) - 125);
        }
    }

    if (rocket_bo_alloc(fd, in_bytes, &bo_in) < 0) goto done;
    if (rocket_bo_alloc(fd, out_bytes, &bo_o) < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    rocket_bo_prep(fd, &bo_in, 1, 0);
    memset(bo_in.ptr, 0, in_bytes);
    {
        int8_t *cube = (int8_t *)bo_in.ptr;
        for (ci = 0; ci < c; ci++)
            for (y = 0; y < ih; y++)
                for (x = 0; x < iw; x++)
                    cube[(size_t)(ci / C2) * in_surf * C2 +
                         (size_t)C2 * (y * iw + x) + (ci % C2)] =
                        src[((size_t)ci * ih + y) * iw + x];
    }
    rocket_bo_fini(fd, &bo_in);

    p.iw = (uint16_t)iw; p.ih = (uint16_t)ih; p.c = (uint16_t)c;
    p.ow = (uint16_t)ow; p.oh = (uint16_t)oh;
    p.kw = (uint8_t)pc->k; p.kh = (uint8_t)pc->k;
    p.stride_x = (uint8_t)pc->stride; p.stride_y = (uint8_t)pc->stride;
    p.pad_left = p.pad_right = p.pad_top = p.pad_bottom = (uint8_t)pc->pad;
    p.mode = (uint8_t)pc->mode;
    p.input_zero_point = 0;
    p.input_dma  = bo_in.dma_address;
    p.output_dma = bo_o.dma_address;
    p.ppu_dst_reg = (uint16_t)dst_reg;
    p.tasks = ops;

    if (gen_pool_rk3576(&p) != 0) { printf("    generator refused\n"); goto done; }

    rocket_bo_prep(fd, &bo_r, 1, 0);
    memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &bo_r);

    /* Bracketed, so "never written" is a property of the surface and not a guess. */
    rocket_bo_prep(fd, &bo_o, 1, 0);
    memset(bo_o.ptr, SENTINEL, out_bytes);
    rocket_bo_fini(fd, &bo_o);

    in_h[0] = bo_in.handle; in_h[1] = bo_r.handle;
    out_h[0] = bo_o.handle;

    if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 2, out_h, 1, 2000) != 0) {
        printf("    submit failed\n"); goto done;
    }
    if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
        printf("    PREP_BO timed out\n"); goto done;
    }

    {
        const uint8_t *o = (const uint8_t *)bo_o.ptr;
        for (i = 0; i < out_bytes; i++)
            if (o[i] != SENTINEL) { *wrote = 1; break; }
    }
    if (check && *wrote) {
        const int8_t *o = (const int8_t *)bo_o.ptr;
        int shown = 0;
        WRONG_BORDER = WRONG_INTERIOR = MAXD_BORDER = MAXD_INTERIOR = 0;
        for (ci = 0; ci < c; ci++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    int want = pool_ref(pc, src, ci, y, x, ow, oh);
                    int have = o[out_index(out_surf, ow, ci, y, x)];
                    int d = have - want; if (d < 0) d = -d;
                    /* A BORDER output is one whose window reaches outside the plane —
                     * the only place a divisor question can show. Splitting the count
                     * this way is what separates "the pad taps are counted wrong" from
                     * "this geometry does not read its own plane". */
                    int bord = ((long)(y * pc->stride) - (long)pc->pad < 0) ||
                               ((long)(x * pc->stride) - (long)pc->pad < 0) ||
                               ((long)(y * pc->stride + pc->k) - (long)pc->pad >
                                (long)pc->ih) ||
                               ((long)(x * pc->stride + pc->k) - (long)pc->pad >
                                (long)pc->iw);
                    if (d) {
                        (*diffs)++;
                        if (d > *maxdiff) *maxdiff = d;
                        if (bord) {
                            WRONG_BORDER++;
                            if (d > MAXD_BORDER) MAXD_BORDER = d;
                        } else {
                            WRONG_INTERIOR++;
                            if (d > MAXD_INTERIOR) MAXD_INTERIOR = d;
                        }
                        if (verbose && shown < 6) {
                            printf("      c=%u (%u,%u) %s want %d got %d\n",
                                   ci, y, x, bord ? "border" : "interior", want, have);
                            shown++;
                        }
                    }
                }
    }
    rc = 0;
done:
    free(src);
    if (bo_in.ptr) rocket_bo_free(fd, &bo_in);
    if (bo_o.ptr)  rocket_bo_free(fd, &bo_o);
    if (bo_r.ptr)  rocket_bo_free(fd, &bo_r);
    return rc;
}

static const pool_case PROBE_CASE = { "probe", 32, 16, 16, 2, 2, 0,
                                      ROCKET_RK3576_POOL_MAX };

static const unsigned CANDIDATES[] = { 0x6054, 0x6058, 0x605C, 0x6070, 0x60DC };
#define N_CAND ((int)(sizeof CANDIDATES / sizeof *CANDIDATES))

static int probe_sweep(int fd)
{
    int wrote, i, found = -1, any = 0;

    printf("sweep: which register carries the destination base?\n");
    printf("  the control first — every candidate at zero, which is the capture "
           "verbatim\n");
    if (run_pool(fd, &PROBE_CASE, 0xFFFF, 0, &wrote, NULL, NULL, 0) != 0) return -1;
    printf("    all zero            -> %s\n", wrote ? "WROTE" : "nothing");
    if (wrote) {
        printf("    it wrote with no base programmed at all — the destination is not "
               "one of these five, and IOVA 0 is a real buffer on this stack\n");
        return -1;
    }

    sleep_ms(200);
    for (i = 0; i < N_CAND; i++) {
        if (run_pool(fd, &PROBE_CASE, CANDIDATES[i], 0, &wrote, NULL, NULL, 0) != 0)
            return -1;
        printf("    0x%04x              -> %s\n", CANDIDATES[i],
               wrote ? "WROTE" : "nothing");
        if (wrote) { any++; if (found < 0) found = i; }
        sleep_ms(200);
    }

    if (!any) {
        printf("  none of the five alone makes it write. A one-at-a-time sweep cannot "
               "see a condition of two, so the next step is a JOINT one — and the "
               "candidate set may simply be wrong, since a register that reads zero in "
               "a capture need not be an address at all.\n");
        return 1;
    }
    printf("  DESTINATION BASE: 0x%04x%s\n", CANDIDATES[found],
           any > 1 ? " (and more than one wrote — narrow before believing it)" : "");
    return 0;
}

static const pool_case GATE[] = {
    { "max-k2s2",     32, 16, 16, 2, 2, 0, ROCKET_RK3576_POOL_MAX },
    { "max-k3s1",     32, 16, 16, 3, 1, 0, ROCKET_RK3576_POOL_MAX },
    { "max-k3s2",     32, 16, 16, 3, 2, 0, ROCKET_RK3576_POOL_MAX },
    { "max-k3s3-15",  32, 15, 15, 3, 3, 0, ROCKET_RK3576_POOL_MAX },
    { "max-nonsq",    32, 19, 17, 2, 2, 0, ROCKET_RK3576_POOL_MAX },
    { "max-c8",        8, 16, 16, 2, 2, 0, ROCKET_RK3576_POOL_MAX },
    { "max-c64",      64, 16, 16, 2, 2, 0, ROCKET_RK3576_POOL_MAX },
    { "max-k5",       32, 16, 16, 5, 1, 0, ROCKET_RK3576_POOL_MAX },
    { "max-pad1",     32, 16, 16, 3, 2, 1, ROCKET_RK3576_POOL_MAX },
    { "avg-k2s2",     32, 16, 16, 2, 2, 0, ROCKET_RK3576_POOL_AVG },
    { "avg-k3s1",     32, 16, 16, 3, 1, 0, ROCKET_RK3576_POOL_AVG },
    /* THE PADDED AVERAGE, both divisors. The PPU has no divider — it multiplies the
     * window sum by two per-axis Q16 reciprocals — so POOL_METHOD_AVG divides by kh*kw
     * whatever the padding excluded (count-include-pad = TRUE) and AVG_NOPAD's own mode
     * bit is meant to drop the pad taps from the divisor as well as the sum, which is
     * what TFLite's AVERAGE_POOL_2D computes. Every cell above is UNPADDED, where the
     * two are the same function, so nothing here had ever driven the bit.
     *
     * The pair at pad 0 is the control that must SUCCEED on both: a mode bit that
     * changed an unpadded pool would be a different op rather than a different divisor.
     * `avg-nopad-35` is Inception V3's own branch-pool geometry. */
    { "avg-p1",       32, 16, 16, 3, 1, 1, ROCKET_RK3576_POOL_AVG },
    { "avg-nopad-p0", 32, 16, 16, 3, 1, 0, ROCKET_RK3576_POOL_AVG_NOPAD },
    { "avg-nopad-p1", 32, 16, 16, 3, 1, 1, ROCKET_RK3576_POOL_AVG_NOPAD },
    { "avg-nopad-s2", 32, 16, 16, 3, 2, 1, ROCKET_RK3576_POOL_AVG_NOPAD },
    /* NO CELL HERE IS WIDER THAN THE PER-TASK OUTPUT-WIDTH ALLOWANCE, deliberately. This
     * mode drives the emitter directly and so bypasses the plan function's refusal; past
     * the allowance the part writes a full surface that is wrong from column 0 in EVERY
     * mode, max included, at every height and both pads. That wall is read out by `rect`
     * and its refusal is asserted by `lib`. Putting a cell past it here would only make
     * this gate permanently red.
     *
     * The pair at pad 0 above is the control that must SUCCEED on both divisors: a mode
     * bit that changed an unpadded pool would be a different op rather than a different
     * divisor. */
    /* An EXACT reciprocal on both axes (0x10000/1 and /2 divide evenly), so this cell
     * asks about the divisor with the truncation confound removed. */
    { "avg-nopad-k2", 32, 16, 16, 2, 1, 1, ROCKET_RK3576_POOL_AVG_NOPAD },
};
#define N_GATE ((int)(sizeof GATE / sizeof *GATE))

static int probe_gate(int fd, unsigned dst_reg, int verbose)
{
    int i, fails = 0;
    printf("gate: the shape table against a CPU model, destination base at 0x%04x\n",
           dst_reg);
    for (i = 0; i < N_GATE; i++) {
        int wrote = 0, diffs = 0, maxd = 0;
        unsigned ow = (GATE[i].iw + 2 * GATE[i].pad - GATE[i].k) / GATE[i].stride + 1;
        unsigned oh = (GATE[i].ih + 2 * GATE[i].pad - GATE[i].k) / GATE[i].stride + 1;
        if (run_pool(fd, &GATE[i], dst_reg, 1, &wrote, &diffs, &maxd, verbose) != 0) {
            fails++; continue;
        }
        printf("  %-4s %-12s c=%-3u %2ux%-2u k%u s%u pad%u -> %2ux%-2u  %s\n",
               (!wrote || diffs) ? "FAIL" : "PASS", GATE[i].name, GATE[i].c,
               GATE[i].iw, GATE[i].ih, GATE[i].k, GATE[i].stride, GATE[i].pad, ow, oh,
               !wrote ? "NOTHING WRITTEN"
                      : diffs ? "differs from the model" : "exact");
        if (diffs)
            printf("       %d wrong, max |diff| %d — border %d (max %d), "
                   "interior %d (max %d)\n", diffs, maxd,
                   WRONG_BORDER, MAXD_BORDER, WRONG_INTERIOR, MAXD_INTERIOR);
        if (!wrote || diffs) fails++;
        sleep_ms(150);
    }
    printf("== %d passed, %d failed ==\n", N_GATE - fails, fails);
    return fails ? 1 : 0;
}

/* The library entry over row-major tensors. `iw*ih` deliberately includes planes whose
 * element count is not a multiple of four — 7x7, 5x5, 3x3 — because that is the only
 * place the source-surface stride can be seen at all. */
struct lib_case {
    const char *name;
    unsigned c, ih, iw, kh, kw, sy, sx, pad;
    int method, in_zp, exclude_pad;
    /* 1 = the ENTRY MUST REFUSE this shape. Past the per-task output-width allowance the
     * part writes a full surface that is wrong from column 0 — but that is no longer a
     * refusal: the entry splits the plane by COLUMNS and runs one task a slice, so those
     * shapes are asserted EXACT here. What is still refused is a plane needing more
     * slices than a handle carries. */
    int refuse;
};

static const struct lib_case LIBGATE[] = {
    { "avg-7x7-k7",     1024, 7,  7,  7, 7, 1, 1, 0, POOL_METHOD_AVG, -128 },
    { "avg-7x7-k7-z0",    64, 7,  7,  7, 7, 1, 1, 0, POOL_METHOD_AVG,    0 },
    { "max-7x7-k3s2",     64, 7,  7,  3, 3, 2, 2, 0, POOL_METHOD_MAX,    0 },
    { "avg-5x5-k5",       32, 5,  5,  5, 5, 1, 1, 0, POOL_METHOD_AVG,   17 },
    { "max-3x3-k3",       16, 3,  3,  3, 3, 1, 1, 0, POOL_METHOD_MAX,    0 },
    { "avg-14x14-k7s7",  512, 14, 14, 7, 7, 7, 7, 0, POOL_METHOD_AVG,  -12 },
    { "max-16x16-k2s2",   32, 16, 16, 2, 2, 2, 2, 0, POOL_METHOD_MAX,    0 },
    { "max-19x17-k3s2p1", 32, 19, 17, 3, 3, 2, 2, 1, POOL_METHOD_MAX,    0 },
    /* INCEPTION V3's BRANCH POOLS: a PADDED 3x3 stride-1 average on an odd plane, at all
     * three of that graph's widths, on both divisors. The raw single-task path in
     * `gate` computes a 35x35 plane wrong in EVERY mode including max and including pad
     * 0, so these ask the same geometry of the entry that owns the row plan — the one a
     * network actually calls. */
    { "avg-35-k3s1p1",    32, 35, 35, 3, 3, 1, 1, 1, POOL_METHOD_AVG,    0 },
    { "avgnp-35-k3s1p1",  32, 35, 35, 3, 3, 1, 1, 1, POOL_METHOD_AVG, -128, 1 },
    /* THE PAIRS THAT SEPARATE THE ZERO POINT FROM THE GEOMETRY. Every AVG_NOPAD cell in
     * `gate` runs at zero point 0, where a pad tap that is still SUMMED contributes
     * nothing and an excluded-divisor mode is indistinguishable from a correct one.
     * Inception V3's branch pools carry -128. Same geometry, both zero points. */
    { "avgnp-17-k3s1p1",  64, 17, 17, 3, 3, 1, 1, 1, POOL_METHOD_AVG, -128, 1 },
    { "avgnp-17-z0",      64, 17, 17, 3, 3, 1, 1, 1, POOL_METHOD_AVG,    0, 1 },
    { "avgnp-8-k3s1p1",  128,  8,  8, 3, 3, 1, 1, 1, POOL_METHOD_AVG, -128, 1 },
    { "avgnp-8-z0",      128,  8,  8, 3, 3, 1, 1, 1, POOL_METHOD_AVG,    0, 1 },
    { "avgnp-8-z40",     128,  8,  8, 3, 3, 1, 1, 1, POOL_METHOD_AVG,   40, 1 },
    { "max-35-k3s1p1",    32, 35, 35, 3, 3, 1, 1, 1, POOL_METHOD_MAX,    0 },
    { "max-147-k3s2",     64, 147, 147, 3, 3, 2, 2, 0, POOL_METHOD_MAX, -128 },
    /* THE COLUMN SPLIT'S OWN AXES, which the four above do not vary: three slices rather
     * than two, an EVEN plane, a plane whose slices divide it unevenly, and the padded
     * divisor over a split — where a slice after the first carries no leading pad and
     * only the last carries the trailing one, so the tap count at a slice boundary is a
     * question about the cut and not about the part. */
    { "avgnp-71-k3s1p1",  32, 71, 71, 3, 3, 1, 1, 1, POOL_METHOD_AVG, -128, 1 },
    { "max-96-k3s1p1",    16, 96, 96, 3, 3, 1, 1, 1, POOL_METHOD_MAX,    0 },
    { "avg-64-k2s1",      16, 64, 64, 2, 2, 1, 1, 0, POOL_METHOD_AVG,    0 },
    { "avgnp-147-k3s1p1", 16, 147, 147, 3, 3, 1, 1, 1, POOL_METHOD_AVG,  -7, 1 },
    /* THE TWO HALVES OF THE PPU's KERNEL WORD, and of its stride pair. `0x6038` packs
     * (sy-1)<<20 | (sx-1)<<16 | (kh-1)<<8 | (kw-1), and every pooling cell ever run here
     * — every vendor capture, every cell above, every pool of the five networks — is
     * SQUARE in both. A transposed assignment computes a full, correctly sized, entirely
     * plausible surface, which is exactly how the CNA's kernel word and the PPU's own pad
     * nibbles were each wrong for several sessions.
     *
     * The cells come in PAIRS at one plane: a transpose maps each onto the other, so
     * both being exact is what says the map is right. The plane is rectangular too, so a
     * transposed EXTENT would not hide inside a square one either. */
    { "max-24x40-k3x1",   32, 24, 40, 3, 1, 1, 1, 0, POOL_METHOD_MAX,    0 },
    { "max-24x40-k1x3",   32, 24, 40, 1, 3, 1, 1, 0, POOL_METHOD_MAX,    0 },
    { "avg-24x40-k4x2",   32, 24, 40, 4, 2, 1, 1, 0, POOL_METHOD_AVG,  -12 },
    { "avg-24x40-k2x4",   32, 24, 40, 2, 4, 1, 1, 0, POOL_METHOD_AVG,  -12 },
    { "max-24x40-s2x1",   32, 24, 40, 3, 3, 2, 1, 0, POOL_METHOD_MAX,    0 },
    { "max-24x40-s1x2",   32, 24, 40, 3, 3, 1, 2, 0, POOL_METHOD_MAX,    0 },
    /* A padded pair needs both axes at 2 or more, or the pad swallows a unit axis
     * whole and the cell is asking about a divisor of zero rather than about the map.
     * 3x2 is also the first even window with an inexact reciprocal, so these carry the
     * tie rule as well as the transpose. */
    { "avgnp-24x40-k3x2", 32, 24, 40, 3, 2, 1, 1, 1, POOL_METHOD_AVG, -128, 1 },
    { "avgnp-24x40-k2x3", 32, 24, 40, 2, 3, 1, 1, 1, POOL_METHOD_AVG, -128, 1 },
};
#define N_LIBGATE ((int)(sizeof LIBGATE / sizeof *LIBGATE))

static int lib_gate(int fd)
{
    int i, fails = 0;
    const int packed = getenv("ROCKET_RK3576_POOL_PACK_SRC") &&
                       *getenv("ROCKET_RK3576_POOL_PACK_SRC") != '0';

    printf("lib: the library entry over row-major tensors, source surface stride = %s\n",
           packed ? "the plane's own element count (what a cube-in join hands it)"
                  : "round4 of the plane (what the vendor's programs carry)");
    for (i = 0; i < N_LIBGATE; i++) {
        const struct lib_case *L = &LIBGATE[i];
        rocket_pool_desc d;
        rocket_pool_int8_rk3576_handle *h;
        size_t in_n = (size_t)L->c * L->ih * L->iw, out_n;
        int8_t *in, *out, *out2, *ref;
        unsigned oh, ow;
        size_t k;
        int bad = 0, bad2 = 0, maxd = 0, rc;

        memset(&d, 0, sizeof d);
        d.c = (int)L->c; d.ih = (int)L->ih; d.iw = (int)L->iw;
        d.kh = (int)L->kh; d.kw = (int)L->kw;
        d.stride_y = (int)L->sy; d.stride_x = (int)L->sx;
        d.pad_top = d.pad_left = d.pad_bottom = d.pad_right = (int)L->pad;
        d.method = L->method;
        d.avg_exclude_pad = L->exclude_pad;
        oh = (unsigned)rocket_pool_oh(&d);
        ow = (unsigned)rocket_pool_ow(&d);
        out_n = (size_t)L->c * oh * ow;

        in = malloc(in_n); out = malloc(out_n); out2 = malloc(out_n); ref = malloc(out_n);
        if (!in || !out || !out2 || !ref) { fails++; free(in); free(out); free(out2);
                                            free(ref); continue; }
        /* A value per position rather than a constant: a probe uniform in an axis cannot
         * see a stride wrong in that axis. */
        for (k = 0; k < in_n; k++)
            in[k] = (int8_t)((int)((k * 37u + (k >> 5) * 11u) % 251u) - 125);
        memset(out, 0x5A, out_n); memset(out2, 0x5A, out_n);
        rocket_pool_ref_int8_rk3576(&d, L->in_zp, in, ref);

        h = rocket_pool_int8_pack_rk3576(fd, &d, L->in_zp);
        if (!h || L->refuse) {
            int ok = L->refuse ? (h == NULL) : 0;
            if (h) rocket_pool_int8_free_rk3576(fd, h);
            printf("  %-4s %-16s c=%-4u %2ux%-2u k%ux%u s%ux%u pad%u  %s\n",
                   ok ? "PASS" : "FAIL", L->name, L->c, L->iw, L->ih, L->kw, L->kh,
                   L->sx, L->sy, L->pad,
                   L->refuse ? (h ? "the entry ACCEPTED a shape past the output-width "
                                    "allowance, where the part is silently wrong"
                                  : "refused, as the output-width allowance requires")
                             : "the handle refused the shape");
            if (!ok) fails++;
            free(in); free(out); free(out2); free(ref); continue;
        }
        rc  = rocket_pool_int8_prepacked_rk3576(fd, h, in, out);
        /* Twice, because the second call is the first to reuse a held surface. */
        rc |= rocket_pool_int8_prepacked_rk3576(fd, h, in, out2);
        rocket_pool_int8_free_rk3576(fd, h);

        for (k = 0; k < out_n; k++) {
            int dd = out[k] - ref[k];
            if (dd) { bad++; if (dd < 0) dd = -dd; if (dd > maxd) maxd = dd; }
            if (out2[k] != out[k]) bad2++;
        }
        printf("  %-4s %-16s c=%-4u %2ux%-2u k%ux%u s%ux%u pad%u -> %2ux%-2u  %s\n",
               (rc || bad || bad2) ? "FAIL" : "PASS", L->name, L->c, L->iw, L->ih,
               L->kw, L->kh, L->sx, L->sy, L->pad, ow, oh,
               rc ? "the entry returned an error"
                  : bad ? "differs from the model"
                  : bad2 ? "the second prepacked call differs from the first" : "exact");
        if (bad) printf("       %d of %zu wrong, max |diff| %d\n", bad, out_n, maxd);
        if (rc || bad || bad2) fails++;
        free(in); free(out); free(out2); free(ref);
        sleep_ms(120);
    }
    printf("== %d passed, %d failed ==\n", N_LIBGATE - fails, fails);
    return fails ? 1 : 0;
}

/* ---- the CAPACITY axis ----------------------------------------------------------
 * Every shape in the tables above is small — the largest plane is 19x17 — and
 * rocket_pool_int8_rk3576_plan() carries no capacity bound at all. A real classifier's
 * first pool is 112x112 at 64 channels, which is three orders of magnitude more feature
 * bytes than anything the gate had run, so what a plane too large does for one PPU
 * program is a question no gate has asked. This sweeps the two axes that could carry
 * the bound — the PLANE and the CHANNEL COUNT — and prints the wrong-element count for
 * each cell, so the law is read off the part rather than reasoned to.
 *
 * The window is held at k3 s2 with the pad a stride-2 SAME-on-an-even-plane produces
 * (one leading, none trailing), which is the shape a network's first pool has.
 */
/* The lowest output COLUMN that disagrees, over every channel and row, and how many
 * distinct columns do. A wall that is a per-pass column budget and a wall that is a
 * capacity look the same in a total; they do not look the same in this. -1 = none. */
static int FIRST_BAD_COL, N_BAD_COLS;

static int pool_run_one(int fd, const rocket_pool_desc *d, int in_zp,
                        int *bad, int *maxd)
{
    size_t in_n = (size_t)d->c * d->ih * d->iw, k;
    unsigned oh = (unsigned)rocket_pool_oh(d), ow = (unsigned)rocket_pool_ow(d);
    size_t out_n = (size_t)d->c * oh * ow;
    int8_t *in = malloc(in_n), *out = malloc(out_n), *ref = malloc(out_n);
    unsigned char *colbad = calloc(ow ? ow : 1, 1);
    int rc = ROCKET_E_NOMEM;

    *bad = 0; *maxd = 0; FIRST_BAD_COL = -1; N_BAD_COLS = 0;
    if (in && out && ref && colbad) {
        for (k = 0; k < in_n; k++)
            in[k] = (int8_t)((int)((k * 37u + (k >> 5) * 11u) % 251u) - 125);
        memset(out, 0x5A, out_n);
        rocket_pool_ref_int8_rk3576(d, in_zp, in, ref);
        rc = rocket_pool_int8_rk3576(fd, d, in_zp, in, out);
        if (rc == ROCKET_OK)
            for (k = 0; k < out_n; k++) {
                int dd = out[k] - ref[k];
                if (dd) {
                    (*bad)++; colbad[k % ow] = 1;
                    if (dd < 0) dd = -dd; if (dd > *maxd) *maxd = dd;
                }
            }
        for (k = 0; k < ow; k++)
            if (colbad[k]) { if (FIRST_BAD_COL < 0) FIRST_BAD_COL = (int)k; N_BAD_COLS++; }
    }
    free(colbad);
    free(in); free(out); free(ref);
    return rc;
}

/* `rect` reads out a wall the ENTRY refuses, so it asks past the refusal through the
 * library's own RE escape. Everything else about the call is identical. */
static int pool_run_one_raw(int fd, const rocket_pool_desc *d, int in_zp,
                            int *bad, int *maxd)
{
    int rc;
    setenv("ROCKET_RK3576_POOL_OW_PROBE", "1", 1);
    rc = pool_run_one(fd, d, in_zp, bad, maxd);
    unsetenv("ROCKET_RK3576_POOL_OW_PROBE");
    return rc;
}

static int bound_sweep(int fd)
{
    static const unsigned CH[] = { 16, 32, 64, 128 };
    /* Odd widths and a few large ones, because every plane this ever swept was EVEN and
     * at most 112 — and Inception V3's pools are 147, 71, 35, 17 and 8. */
    static const unsigned PLANE[] = { 8, 15, 16, 17, 24, 32, 35, 40, 48, 56, 64, 71,
                                      80, 96, 112, 147 };
    unsigned ci, pi;
    int fails = 0;
    const char *el = getenv("ROCKET_POOLB_LEAD"), *et = getenv("ROCKET_POOLB_TRAIL");
    const char *ek = getenv("ROCKET_POOLB_K"),   *es = getenv("ROCKET_POOLB_S");
    const int lead = el ? atoi(el) : 1, trail = et ? atoi(et) : 0;
    const int kk = ek ? atoi(ek) : 3, ss = es ? atoi(es) : 2;

    printf("bound: max k%d s%d, %d leading pad and %d trailing, over plane x channels."
           "\n       Each cell is the wrong-element count; \".\" is exact.\n",
           kk, ss, lead, trail);
    printf("  %6s", "c\\wxh");
    for (pi = 0; pi < sizeof PLANE / sizeof *PLANE; pi++)
        printf(" %7u", PLANE[pi]);
    printf("\n");
    for (ci = 0; ci < sizeof CH / sizeof *CH; ci++) {
        printf("  %6u", CH[ci]);
        for (pi = 0; pi < sizeof PLANE / sizeof *PLANE; pi++) {
            rocket_pool_desc d;
            int bad, maxd, rc;
            memset(&d, 0, sizeof d);
            d.c = (int)CH[ci]; d.ih = d.iw = (int)PLANE[pi];
            d.kh = d.kw = kk; d.stride_y = d.stride_x = ss;
            d.pad_top = d.pad_left = lead;
            d.pad_bottom = d.pad_right = trail;
            d.method = POOL_METHOD_MAX;
            /* A shape past the per-task output-width allowance is REFUSED by the
             * entry, which is the contract — so it is reported as such and is not a
             * wrong cell. The wall itself is read out by `rect`. */
            if (rocket_pool_int8_rk3576_plan(&d) != ROCKET_OK) {
                printf(" %7s", "ref"); fflush(stdout); continue;
            }
            rc = pool_run_one(fd, &d, 0, &bad, &maxd);
            if (rc != ROCKET_OK) { printf(" %7s", "err"); fails++; }
            else if (bad)        { printf(" %7d", bad); fails++; }
            else                   printf(" %7s", ".");
            fflush(stdout);
            sleep_ms(60);
        }
        printf("\n");
    }
    printf("== %d cell(s) wrong ==  \"ref\" = past the output-width allowance, refused\n",
           fails);
    return fails ? 1 : 0;
}

/*
 * WHICH AXIS THE POOLING WALL IS ON. `bound` says a k3 s1 pool is exact to a 32-wide
 * square plane and wrong from 35, and a k3 s2 one is exact to 112 and wrong at 147 — two
 * different limits, so it is neither an input width nor an output width on its own. Every
 * cell of that sweep is SQUARE, which cannot separate a width bound from a height bound
 * from a product, so this one holds one axis while it moves the other.
 *
 * Reported as a table rather than a boundary: a bisection over a bound that may be
 * intermittent reports the wrong number, and the shape is what names the mechanism.
 */
static int rect_sweep(int fd)
{
    static const unsigned W[] = { 16, 32, 33, 34, 35, 40, 48, 56, 64, 65, 66, 71, 96,
                                  112, 113, 128, 129, 130, 147 };
    static const unsigned HALL[] = { 2, 4, 8, 16, 32, 35, 64, 112 };
    static unsigned HONE[1];
    const unsigned *HH = HALL;
    unsigned nh = sizeof HALL / sizeof *HALL, wi, hi;
    int fails = 0;
    const char *ek = getenv("ROCKET_POOLB_K"), *es = getenv("ROCKET_POOLB_S");
    const char *el = getenv("ROCKET_POOLB_LEAD"), *eh = getenv("ROCKET_POOLB_H");
    /* THE TWO AXES SEPARATELY, because the wall was characterised at kh == kw and a
     * square kernel cannot say which of them it is keyed on. Defaulting each to the
     * square knob leaves every existing invocation reading what it always did. */
    const char *ekh = getenv("ROCKET_POOLB_KH"), *ekw = getenv("ROCKET_POOLB_KW");
    const char *esy = getenv("ROCKET_POOLB_SY"), *esx = getenv("ROCKET_POOLB_SX");
    const int kk = ek ? atoi(ek) : 3, ss = es ? atoi(es) : 1;
    const int kh = ekh ? atoi(ekh) : kk, kw = ekw ? atoi(ekw) : kk;
    const int sy = esy ? atoi(esy) : ss, sx = esx ? atoi(esx) : ss;
    const int lead = el ? atoi(el) : 1;

    /* The height axis is settled — the wall does not move with it — so one row is the
     * whole table once that is known, and it makes a (k, stride, pad) sweep affordable. */
    if (eh) { HONE[0] = (unsigned)atoi(eh); HH = HONE; nh = 1; }

    printf("rect: max k%dx%d (hxw) s%dx%d lead %d, c=32, WIDTH across x HEIGHT down.\n"
           "      Each cell is the wrong-element count; \".\" is exact.\n",
           kh, kw, sy, sx, lead);
    printf("  %6s", "h\\w");
    for (wi = 0; wi < sizeof W / sizeof *W; wi++) printf(" %7u", W[wi]);
    printf("\n");
    for (hi = 0; hi < nh; hi++) {
        printf("  %6u", HH[hi]);
        for (wi = 0; wi < sizeof W / sizeof *W; wi++) {
            rocket_pool_desc d;
            int bad, maxd, rc;
            memset(&d, 0, sizeof d);
            d.c = 32; d.iw = (int)W[wi]; d.ih = (int)HH[hi];
            d.kh = kh; d.kw = kw; d.stride_y = sy; d.stride_x = sx;
            d.pad_top = d.pad_left = lead;
            d.method = POOL_METHOD_MAX;
            if (rocket_pool_oh(&d) < 1 || rocket_pool_ow(&d) < 1) {
                printf(" %7s", "-"); continue;
            }
            /* This mode deliberately drives shapes the ENTRY refuses — that is the
             * wall it exists to read out — so the probe asks the emitter directly
             * through the RE escape rather than being stopped by the plan. */
            rc = pool_run_one_raw(fd, &d, 0, &bad, &maxd);
            if (rc != ROCKET_OK) { printf(" %7s", "err"); fails++; }
            else if (bad)        { printf(" %4d@%-2d", N_BAD_COLS, FIRST_BAD_COL);
                                   fails++; }
            else                   printf(" %7s", ".");
            fflush(stdout);
            sleep_ms(60);
        }
        printf("\n");
    }
    printf("== %d cell(s) wrong ==  cells read \"<columns wrong>@<first wrong column>\"\n",
           fails);
    return fails ? 1 : 0;
}

/* ---- what an ASYMMETRIC pad actually does ----------------------------------------
 * `bound` says a symmetric pad computes and an asymmetric one does not, at every plane
 * and every channel count, which is the emitter or the part and not a capacity. This
 * reads the answer off a ONE-DIMENSIONAL case — one row, a unit kernel on the y axis —
 * so the whole disagreement fits on a line and can be scored against candidate window
 * placements instead of guessed at.
 */
static int pad_map(int fd)
{
    static const int PADL[] = { 0, 1, 2 }, PADR[] = { 0, 1, 2 };
    const unsigned C = 16, IW = 12, KW = 3, SX = 2;
    const char *eo = getenv("ROCKET_POOLP_OTHER");
    const unsigned OTHER = eo ? (unsigned)atoi(eo) : 4;
    unsigned li, ri, x, k;
    int fails = 0;
    int8_t *in = malloc((size_t)C * IW * OTHER);

    if (!in) return 1;
    for (k = 0; k < (size_t)C * IW * OTHER; k++)
        in[k] = (int8_t)(-120 + (int)(k % 61u) * 3);

    printf("pad: k%ux1 s%ux1 over a %u-long axis, run once along x and once along y.\n"
           "     For each (lead, trail) the part's line is scored against the window\n"
           "     grid each candidate leading pad implies.\n", KW, SX, IW);
    for (li = 0; li < 2 * (sizeof PADL / sizeof *PADL); li++)
    for (ri = 0; ri < sizeof PADR / sizeof *PADR; ri++) {
        const int ax_y = li >= sizeof PADL / sizeof *PADL;
        const int lead_v = PADL[li % (sizeof PADL / sizeof *PADL)], trail_v = PADR[ri];
        rocket_pool_desc d;
        int8_t *out, *ref;
        unsigned ow;
        int rc, hit_left = 0, hit_right = 0, hit_zero = 0;

        memset(&d, 0, sizeof d);
        d.c = (int)C;
        d.method = POOL_METHOD_MAX;
        /* The OTHER axis is four long, not one: a unit extent is degenerate here (the
         * line and surface strides collapse onto each other) and reads values that are
         * in no input, which would be scored as a pad defect. */
        if (ax_y) {
            d.ih = (int)IW; d.iw = (int)OTHER;
            d.kh = (int)KW; d.kw = 1; d.stride_y = (int)SX; d.stride_x = 1;
            d.pad_top = lead_v; d.pad_bottom = trail_v;
            ow = (unsigned)rocket_pool_oh(&d);
        } else {
            d.ih = (int)OTHER; d.iw = (int)IW;
            d.kh = 1; d.kw = (int)KW; d.stride_y = 1; d.stride_x = (int)SX;
            d.pad_left = lead_v; d.pad_right = trail_v;
            ow = (unsigned)rocket_pool_ow(&d);
        }
        out = malloc((size_t)C * ow * OTHER); ref = malloc((size_t)C * ow * OTHER);
        if (!out || !ref) { free(out); free(ref); fails++; continue; }
        memset(out, 0x5A, (size_t)C * ow * OTHER);
        rc = rocket_pool_int8_rk3576(fd, &d, 0, in, out);
        if (rc != ROCKET_OK) {
            printf("  l%d r%d  the entry returned %d\n", PADL[li], PADR[ri], rc);
            free(out); free(ref); fails++; continue;
        }
        /* Three candidate placements for output x: the window starts at x*s minus the
         * LEFT pad (what the library computes), minus the RIGHT one (the low nibble read
         * as the leading edge), or at x*s with no leading pad at all. */
        /* The swept axis is the innermost one on x and strided by OTHER on y; both
         * tensors are channel-major, so one pair of strides covers both. */
        {
        const unsigned istep = ax_y ? OTHER : 1, ostep = ax_y ? OTHER : 1;
        for (k = 0; k < 3; k++) {
            int lead = k == 0 ? lead_v : k == 1 ? trail_v : 0, ok = 1;
            unsigned ch;
            for (ch = 0; ch < C && ok; ch++)
                for (x = 0; x < ow && ok; x++) {
                    int best = -128, t;
                    for (t = 0; t < (int)KW; t++) {
                        int sx = (int)x * (int)SX + t - lead;
                        if (sx >= 0 && sx < (int)IW &&
                            in[ch * IW * OTHER + (unsigned)sx * istep] > best)
                            best = in[ch * IW * OTHER + (unsigned)sx * istep];
                    }
                    if (out[ch * ow * OTHER + x * ostep] != (int8_t)best) ok = 0;
                }
            if (k == 0) hit_left = ok; else if (k == 1) hit_right = ok; else hit_zero = ok;
        }
        }
        printf("  %c lead%d trail%d -> %u:  lead=the lead pad %s  lead=the TRAIL pad %s  "
               "lead=0 %s\n", ax_y ? 'y' : 'x', lead_v, trail_v, ow,
               hit_left ? "MATCH" : "-", hit_right ? "MATCH" : "-",
               hit_zero ? "MATCH" : "-");
        if (!hit_left && !hit_right && !hit_zero) {
            /* 0xA5 is the library's own sentinel, so an element carrying it was never
             * WRITTEN — a different fact from an element computed wrong, and the one
             * that says the disagreement is not about the pad at all. */
            unsigned ch, unwritten = 0, wrong = 0;
            for (ch = 0; ch < C; ch++)
                for (x = 0; x < ow; x++) {
                    int8_t v = out[ch * ow * OTHER + x * (ax_y ? OTHER : 1)];
                    int best = -128, t;
                    for (t = 0; t < (int)KW; t++) {
                        int sx = (int)x * (int)SX + t - lead_v;
                        if (sx >= 0 && sx < (int)IW &&
                            in[ch * IW * OTHER + (unsigned)sx * (ax_y ? OTHER : 1)] > best)
                            best = in[ch * IW * OTHER + (unsigned)sx * (ax_y ? OTHER : 1)];
                    }
                    if (v == (int8_t)best) continue;
                    if ((uint8_t)v == 0xA5u) unwritten++; else wrong++;
                }
            printf("       against the leading pad: %u never written (the sentinel), "
                   "%u computed wrong\n", unwritten, wrong);
            printf("       part ch0:");
            for (x = 0; x < ow; x++) printf(" %4d", out[x * (ax_y ? OTHER : 1)]);
            printf("\n       in   ch0:");
            for (x = 0; x < IW; x++) printf(" %4d", in[x * (ax_y ? OTHER : 1)]);
            printf("\n");
            fails++;
        }
        free(out); free(ref);
        sleep_ms(60);
    }
    free(in);
    /* THIS PROBE REPORTS; IT DOES NOT ASSERT. The pad map it decodes is asserted by
     * `bound` (a plane x channel sweep at a real 2-D window, exact in all 40 cells) and
     * by the net gate's first max pool. What this form cannot assert is its own y axis:
     * a UNIT kernel on the other axis is unstable on this part — the same cell reports
     * "never written", "computed wrong" and "MATCH" across runs, and it raises the
     * occasional backstop hit — and no network has a pooling shape of that form, so it is
     * an open observation rather than a regression. */
    printf("== %d cell(s) matched no candidate; this probe REPORTS, the assertion is in "
           "`bound` ==\n", fails);
    return 0;
}

/* ===========================================================================
 * `avg` — WHAT FUNCTION A PADDED AVERAGE POOL COMPUTED WHEN IT COMPUTED THE
 * WRONG ONE.
 *
 * Inception V3's oracle pass is intermittently wrong on one or two of its nine padded
 * 3x3 stride-1 average pools, always at maxdiff 213, and a re-run of the same layer is
 * clean. That signature is a DETERMINISTIC wrong surface produced INTERMITTENTLY, which
 * is neither a dropped atom (those read as the surface's fill) nor a rounding drift.
 *
 * A count of wrong elements cannot say which function the part evaluated, so this scores
 * the part's surface against four CPU models at once and reports how many of the wrong
 * elements each one explains:
 *
 *   lib      the divisor the library asks for — the taps that fell inside the plane
 *   incl     count-include-pad, the whole window, which is the other mode bit
 *   lag-x    the in-plane COLUMN count taken one output column late
 *   lag-y    the same one output ROW late
 *
 * The two `lag` models are not guesses about a mechanism; they are the shape the graph's
 * own failures have. A wrong set confined to output columns 1 and ow-1, with the part
 * reading -128*6/9 where the model reads -128, is exactly a column count that is right
 * everywhere except one column early and one column late.
 *
 * Both mode bits are run over the same geometries, because "the residual is not the new
 * mode" is a claim that needs the OTHER arm measured rather than argued.
 * ==========================================================================*/
enum { AVGM_LIB = 0, AVGM_INCL, AVGM_LAGX, AVGM_LAGY, AVGM_PREV, AVGM_N };
static const char *AVGM_NAME[AVGM_N] = { "lib", "incl", "lag-x", "lag-y", "prev-window" };

/* The in-plane tap count on one axis at output position `o`. Negative `o` clamps to 0,
 * which is what makes the lag models agree with `lib` at the leading edge — and the
 * graph's failures leave output column 0 exact, so that agreement is the observation. */
static int avg_inplane(int o, int stride, int k, int pad, int extent)
{
    int lo, hi;
    if (o < 0) o = 0;
    lo = o * stride - pad;
    hi = lo + k;
    if (lo < 0) lo = 0;
    if (hi > extent) hi = extent;
    return hi > lo ? hi - lo : 0;
}

/* The library reference's own rounding, over a divisor pair this caller chooses. Kept
 * byte-identical to `rocket_pool_ref_int8_rk3576()` — including the wrap to int8 with no
 * saturation, which is what makes a divisor of 6 where 9 was meant read as +79 rather
 * than as a clamp at -128. */
static int8_t avg_round(long sum, int dw, int dh)
{
    long n = (long)dw * dh, half, q, r;
    if (n <= 0) { n = 1; dw = dh = 1; }
    half = n / 2;
    q = sum >= 0 ? (sum + half) / n : -(((-sum) + half) / n);
    if ((n & 1) == 0) {
        int exact_recip = (0x10000 % dw) == 0 && (0x10000 % dh) == 0;
        r = sum - q * n;
        if ((r == half || r == -half) && (!exact_recip || (q & 1)))
            q += (sum >= 0) ? -1 : 1;
    }
    return (int8_t)q;
}

static void avg_model(const rocket_pool_desc *d, const int8_t *in, int8_t *out, int model)
{
    int oh = rocket_pool_oh(d), ow = rocket_pool_ow(d);
    int c, y, x, kh, kw;

    for (c = 0; c < d->c; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                long sum = 0;
                int dw, dh;
                for (kh = 0; kh < d->kh; kh++)
                    for (kw = 0; kw < d->kw; kw++) {
                        int iy = y * d->stride_y + kh - d->pad_top;
                        int ix = x * d->stride_x + kw - d->pad_left;
                        /* The pad value is zero on every arm: the mode bit moves the
                         * DIVISOR and not the sum, so a tap outside the plane adds
                         * nothing whichever divisor is under test. */
                        if (iy < 0 || ix < 0 || iy >= d->ih || ix >= d->iw) continue;
                        sum += in[((size_t)c * d->ih + iy) * d->iw + ix];
                    }
                dw = avg_inplane(x, d->stride_x, d->kw, d->pad_left, d->iw);
                dh = avg_inplane(y, d->stride_y, d->kh, d->pad_top, d->ih);
                switch (model) {
                case AVGM_INCL: dw = d->kw; dh = d->kh; break;
                case AVGM_LAGX:
                    dw = avg_inplane(x - 1, d->stride_x, d->kw, d->pad_left, d->iw);
                    break;
                case AVGM_LAGY:
                    dh = avg_inplane(y - 1, d->stride_y, d->kh, d->pad_top, d->ih);
                    break;
                case AVGM_PREV: {
                    /* The divisor belonging to the previous window in RASTER order —
                     * one output position late, wrapping at the row boundary. */
                    int py = y, px = x - 1;
                    if (px < 0) { px = ow - 1; py = y - 1; }
                    if (py < 0) { py = y; px = x; }
                    dw = avg_inplane(px, d->stride_x, d->kw, d->pad_left, d->iw);
                    dh = avg_inplane(py, d->stride_y, d->kh, d->pad_top, d->ih);
                    break;
                }
                default: break;
                }
                out[((size_t)c * oh + y) * ow + x] = avg_round(sum, dw, dh);
            }
}

typedef struct { unsigned c, plane; } avg_case;

/* WHAT RAN BEFORE IT. The pool alone reproduces nothing over sixty iterations, and the
 * graph reproduces it several times in ten — so the trigger is not the pooling program
 * and the arms have to name the submit in front of it. A convolution is what precedes
 * every pool in a network; an int32-output matmul is the one job class already known to
 * leave the next submit of any kind broken. */
enum { AVG_PRE_NONE = 0, AVG_PRE_WIDE, AVG_PRE_N };
static const char *AVG_PRE_NAME[AVG_PRE_N] = { "nothing", "an int32-output matmul" };

static int avg_pre_submit(int fd, int which)
{
    if (which == AVG_PRE_WIDE) {
        enum { PM = 32, PK = 64, PN = 64 };
        static int8_t *pa, *pb;
        static int32_t *pc;
        if (!pa) {
            pa = calloc((size_t)PM * PK, 1);
            pb = calloc((size_t)PN * PK, 1);
            pc = calloc((size_t)PM * PN, sizeof *pc);
            if (!pa || !pb || !pc) return -1;
            pa[0] = 1; pb[0] = 1;
        }
        return rocket_matmul_int8_rk3576_i32(fd, PM, PK, PN, pa, pb, NULL, pc);
    }
    return 0;
}

static int avg_hazard(int fd)
{
    /* Inception V3's own padded averages: 3x3 stride 1 SAME at three planes, with the
     * channel counts the graph carries. The two that have been seen to fail are the
     * 17x17 at 768 channels and the 8x8 at 1280. */
    static const avg_case CASES[] = { { 1280, 8 }, { 2048, 8 }, { 768, 17 } };
    const char *en = getenv("ROCKET_POOLA_N");
    const unsigned iters = en ? (unsigned)atoi(en) : 30u;
    /* WHICH SUBMIT THIS MODE MAKES, selectable. The mode runs two program classes — the
     * padded average pool itself, and (in the AVG_PRE_WIDE arm) an int32-output matmul,
     * which is the one class already known to poison the next submit of any kind across
     * processes. Anything that names this whole mode as the cause of some later effect is
     * naming both at once; these restrict it to one arm without moving the default. */
    const char *ep = getenv("ROCKET_POOLA_PRE");
    const char *ex = getenv("ROCKET_POOLA_EXCL");
    const char *eo = getenv("ROCKET_POOLA_PRE_ONLY");
    const int only_pre  = ep && *ep ? atoi(ep) : -1;
    const int only_excl = ex && *ex ? atoi(ex) : -1;
    /* The no-pool cell of the joint sweep: the SAME pre-submit, the same binary and the
     * same count, with the pooling program left out. Without it "the pool" and "the
     * int32 matmul" can only be varied together, which is how they were confounded. */
    const int pre_only  = eo && *eo ? atoi(eo) : 0;
    unsigned ci, it;
    int excl, pre, fails = 0;

    printf("avg: a padded 3x3 s1 average, %u iterations a cell, the same input every\n"
           "     time. A failing iteration is scored against every divisor model.\n",
           iters);
    for (pre = 0; pre < AVG_PRE_N; pre++)
    for (excl = 1; excl >= 0; excl--) {
        if (only_pre  >= 0 && pre  != only_pre)  continue;
        if (only_excl >= 0 && excl != only_excl) continue;
        printf("\n  before each pool: %s;  avg_exclude_pad = %d (%s)\n",
               AVG_PRE_NAME[pre], excl,
               excl ? "the mode bit the graph uses" : "count-include-pad, the control");
        for (ci = 0; ci < sizeof CASES / sizeof *CASES; ci++) {
            rocket_pool_desc d;
            size_t in_n, out_n, k;
            int8_t *in, *out, *mdl[AVGM_N];
            unsigned ow, oh, m, wrong_iters = 0;
            /* The arm's OWN arithmetic is the baseline: with the mode bit clear the part
             * is asked for count-include-pad, so scoring it against the in-plane divisor
             * would report the whole border as a failure. */
            const int base = excl ? AVGM_LIB : AVGM_INCL;

            memset(&d, 0, sizeof d);
            d.c = (int)CASES[ci].c;
            d.ih = d.iw = (int)CASES[ci].plane;
            d.kh = d.kw = 3; d.stride_y = d.stride_x = 1;
            d.pad_top = d.pad_left = 1; d.pad_bottom = d.pad_right = 1;
            d.method = POOL_METHOD_AVG;
            d.avg_exclude_pad = excl;
            oh = (unsigned)rocket_pool_oh(&d); ow = (unsigned)rocket_pool_ow(&d);
            in_n = (size_t)d.c * d.ih * d.iw;
            out_n = (size_t)d.c * oh * ow;
            in = malloc(in_n); out = malloc(out_n);
            for (m = 0; m < AVGM_N; m++) mdl[m] = malloc(out_n);
            if (!in || !out || !mdl[AVGM_N - 1]) { printf("   out of memory\n"); return 1; }
            /* Saturated data on purpose: the graph's own failures sit where the true
             * average is at or near -128, which is where a divisor of 6 against 9 is
             * furthest from a rounding difference. */
            for (k = 0; k < in_n; k++)
                in[k] = (int8_t)(((k * 37u + (k >> 6) * 101u) % 40u) - 128u + 0u);
            for (m = 0; m < AVGM_N; m++) avg_model(&d, in, mdl[m], (int)m);

            printf("   c%-5u %2ux%-2u -> %2ux%-2u: ", d.c, d.iw, d.ih, ow, oh);
            fflush(stdout);
            for (it = 0; it < iters; it++) {
                long wrong = 0, expl[AVGM_N];
                unsigned char *col = calloc(ow, 1);
                long chan_lo = -1, chan_hi = -1, chans = 0;
                int rc;
                for (m = 0; m < AVGM_N; m++) expl[m] = 0;
                memset(out, 0x5A, out_n);
                if (avg_pre_submit(fd, pre) < 0) { printf("P"); free(col); continue; }
                if (pre_only) { printf("p"); free(col); fflush(stdout); continue; }
                rc = rocket_pool_int8_rk3576(fd, &d, 0, in, out);
                if (rc != ROCKET_OK) { printf("E"); free(col); fails++; continue; }
                for (k = 0; k < out_n; k++) {
                    long ch = (long)(k / ((size_t)oh * ow));
                    if (out[k] == mdl[base][k]) continue;
                    wrong++;
                    col[k % ow] = 1;
                    if (ch != chan_hi) { chans++; chan_hi = ch; }
                    if (chan_lo < 0) chan_lo = ch;
                    for (m = 0; m < AVGM_N; m++) if (out[k] == mdl[m][k]) expl[m]++;
                }
                if (!wrong) { printf("."); free(col); fflush(stdout); sleep_ms(5); continue; }
                printf("X");
                wrong_iters++; fails++;
                printf("\n      iteration %u: %ld wrong of %zu, channels %ld..%ld (%ld "
                       "distinct), columns", it, wrong, out_n, chan_lo, chan_hi, chans);
                for (m = 0; m < ow; m++) if (col[m]) printf(" %u", m);
                printf("\n      explained:");
                for (m = 0; m < AVGM_N; m++)
                    printf("  %s %ld/%ld", AVGM_NAME[m], expl[m], wrong);
                printf("\n   ");
                free(col);
                fflush(stdout);
                sleep_ms(5);
            }
            printf("  %u of %u iterations wrong\n", wrong_iters, iters);
            free(in); free(out);
            for (m = 0; m < AVGM_N; m++) free(mdl[m]);
        }
    }
    printf("\n== %d wrong iteration(s) ==  with the library's divisor check on this "
           "ASSERTS: a wrong surface here is one the check did not catch\n", fails);
    return fails ? 1 : 0;
}

/* ===========================================================================
 * `rate` — HOW OFTEN THE DIVISOR LAGS, AS A FUNCTION OF THE GAP BETWEEN SUBMITS.
 *
 * The hazard's RATE is what every cost about it is a function of: the check is paid per
 * call, a redo is paid per occurrence, and a coverage item's price is the rate times the
 * blast radius. Two rates were on record and they differ by thirty times — about 1% of
 * calls from `avg` above, and about a third of them inside Inception V3 — with nothing
 * separating the two contexts, so the difference was carried as a property of "the graph".
 *
 * The one axis that differs between them and is not the geometry is TIME: `avg` sleeps
 * between iterations and a graph submits back to back. This runs one geometry at a
 * sequence of gaps and reports the rate at each, which makes the rate an OUTPUT rather
 * than a number attached to a context.
 *
 * The library's own check is turned OFF here, because with it on every occurrence is
 * repaired before the caller sees it and the rate reads as zero. That is also why this is
 * a PROBE and not a gate: it measures a hazard rather than asserting its absence.
 *
 * Env: ROCKET_POOLA_N (iterations a cell, 60), ROCKET_POOLR_GAPS (the gap list in ms).
 * ==========================================================================*/
static int rate_map(int fd)
{
    static const avg_case CASES[] = { { 768, 17 }, { 288, 35 }, { 2048, 8 } };
    static const int GAPS_DEF[] = { 0, 1, 2, 5, 10, 20 };
    int gaps[16], ngap = 0;
    const char *eg = getenv("ROCKET_POOLR_GAPS");
    const char *en = getenv("ROCKET_POOLA_N");
    const unsigned iters = en ? (unsigned)atoi(en) : 60u;
    unsigned ci;
    int g;

    if (eg && *eg) {
        char buf[128], *t;
        snprintf(buf, sizeof buf, "%s", eg);
        for (t = strtok(buf, ","); t && ngap < 16; t = strtok(NULL, ","))
            gaps[ngap++] = atoi(t);
    }
    if (!ngap) {
        for (g = 0; g < (int)(sizeof GAPS_DEF / sizeof *GAPS_DEF); g++)
            gaps[ngap++] = GAPS_DEF[g];
    }

    /* Not cached inside the library, so setting it here reaches every call below. */
    setenv("ROCKET_RK3576_POOL_LAGCHECK", "0", 1);
    printf("rate: a padded 3x3 s1 average, %u calls a cell, the library's divisor check\n"
           "      OFF so the raw rate comes back. The gap is a sleep between calls.\n\n",
           iters);
    printf("  %-18s", "geometry");
    for (g = 0; g < ngap; g++) printf("  %4d ms", gaps[g]);
    printf("\n");

    for (ci = 0; ci < sizeof CASES / sizeof *CASES; ci++) {
        rocket_pool_desc d;
        size_t in_n, out_n, k;
        int8_t *in, *out, *mdl, *prv;
        unsigned ow, oh, it;

        memset(&d, 0, sizeof d);
        d.c = (int)CASES[ci].c;
        d.ih = d.iw = (int)CASES[ci].plane;
        d.kh = d.kw = 3; d.stride_y = d.stride_x = 1;
        d.pad_top = d.pad_left = 1; d.pad_bottom = d.pad_right = 1;
        d.method = POOL_METHOD_AVG;
        d.avg_exclude_pad = 1;
        oh = (unsigned)rocket_pool_oh(&d); ow = (unsigned)rocket_pool_ow(&d);
        in_n = (size_t)d.c * d.ih * d.iw;
        out_n = (size_t)d.c * oh * ow;
        in = malloc(in_n); out = malloc(out_n); mdl = malloc(out_n); prv = malloc(out_n);
        if (!in || !out || !mdl || !prv) { printf("   out of memory\n"); return 1; }
        for (k = 0; k < in_n; k++)
            in[k] = (int8_t)(((k * 37u + (k >> 6) * 101u) % 40u) - 128u);
        avg_model(&d, in, mdl, AVGM_LIB);
        avg_model(&d, in, prv, AVGM_PREV);

        printf("  c%-5u %3ux%-3u  ", d.c, d.iw, d.ih);
        fflush(stdout);
        for (g = 0; g < ngap; g++) {
            unsigned wrong_iters = 0;
            for (it = 0; it < iters; it++) {
                memset(out, 0x5A, out_n);
                if (rocket_pool_int8_rk3576(fd, &d, 0, in, out) != ROCKET_OK) continue;
                if (memcmp(out, mdl, out_n) != 0) {
                    wrong_iters++;
                    /* WHICH FUNCTION, not how many elements: a cell that is wrong for
                     * some other reason must not be counted as this hazard. */
                    if (wrong_iters == 1) {
                        long w = 0, e = 0;
                        for (k = 0; k < out_n; k++)
                            if (out[k] != mdl[k]) { w++; if (out[k] == prv[k]) e++; }
                        printf("[%ld/%ld prev]", e, w);
                    }
                }
                sleep_ms(gaps[g]);
            }
            printf("  %3u/%-3u", wrong_iters, iters);
            fflush(stdout);
        }
        printf("\n");
        free(in); free(out); free(mdl); free(prv);
    }
    printf("\n  A rate that falls with the gap says the hazard is a function of TIME "
           "between\n  submits and not of the pooling program.\n");
    return 0;
}

/* ===========================================================================
 * `place` — A POOL WRITES A SLICE OF SOMEBODY ELSE'S BUFFER.
 *
 * A concatenation whose operands already carry the output's quantization is placement
 * rather than arithmetic: one buffer, a slice per operand, and the consumer reads the
 * whole thing as its feature cube. The buffer's channel-group stride is the PLANE, because
 * the other operands are convolutions and a direct convolution's output surface stride is
 * `ow*oh` exactly — so a pool joining that set has to write at the plane rather than at the
 * `round4(ow*oh)` its own surface uses.
 *
 * It can, and this asserts it end to end at planes whose element count is NOT a multiple of
 * four, which is the only case where the two strides differ. Three things are checked per
 * cell, because a stride that is merely accepted proves nothing:
 *
 *   - the de-scattered result is bit-exact against the CPU model, twice (the second call is
 *     the first to reuse a held surface);
 *   - every byte OUTSIDE the slice is still the poison the buffer was filled with, which is
 *     what says the groups landed where the stride says and not on a neighbour's operand;
 *   - a stride BELOW the plane is refused, which is the arm that must show a positive.
 * ==========================================================================*/
struct place_case {
    const char *name;
    unsigned c, ih, iw, kh, kw, sy, sx;
    unsigned pad;
    int      method, exclude_pad, in_zp;
    unsigned bufc, slot;      /* the shared buffer's channels, and this operand's first */
};

static const struct place_case PLACECASE[] = {
    /* Inception V3's layer 39, in miniature: the max pool of a reduction module writing
     * the tail of a concatenation whose plane is 17x17 = 289 elements, where the pool's
     * own surface would be 292. Placed after two other operands. */
    { "plane-289",    32u, 35u, 35u, 3u,3u, 2u,2u, 0u, POOL_METHOD_MAX, 0, -12,  96u, 64u },
    /* An ODD plane: 7x7 = 49, the stride a 7x7 tail already hands a consumer. */
    { "plane-49",     16u, 14u, 14u, 2u,2u, 2u,2u, 0u, POOL_METHOD_MAX, 0,   0,  64u, 16u },
    /* A padded average with the pad EXCLUDED — the one mode whose divisor varies with
     * position — so the lag check runs against a placed surface too. */
    { "plane-289-avg",32u, 17u, 17u, 3u,3u, 1u,1u, 1u, POOL_METHOD_AVG, 1,   0,  64u, 32u },
    /* THE CONTROL: a plane already a multiple of four, so the placed stride and the
     * derived one agree. It exercises the same path and must pass. */
    { "plane-64-ctl", 32u, 16u, 16u, 2u,2u, 2u,2u, 0u, POOL_METHOD_MAX, 0,   9,  64u, 32u },
};
#define N_PLACECASE ((int)(sizeof PLACECASE / sizeof PLACECASE[0]))

#define PLACE_POISON 0x3C

/* Score the slice IN CUBE LAYOUT against the row-major reference. Returns the number of
 * wrong elements and raises `*maxd`. Brackets its own read of the buffer. */
static int place_score(rocket_rk3576_cube *buf, unsigned slot, unsigned c,
                       unsigned oh, unsigned ow, const int8_t *ref, int *maxd)
{
    const int8_t *b;
    unsigned ch, px;
    int bad = 0;

    rocket_bo_prep(buf->fd, &buf->bo, 0, 2000000000ull);
    b = (const int8_t *)buf->bo.ptr;
    for (ch = 0; ch < c; ch++) {
        unsigned g = (slot + ch) / C2, lane = (slot + ch) % C2;
        for (px = 0; px < oh * ow; px++) {
            int8_t got = b[(size_t)g * buf->surf_elems * C2 + (size_t)px * C2 + lane];
            int8_t want = ref[(size_t)ch * oh * ow + px];
            int dd = got - want;
            if (dd < 0) dd = -dd;
            if (dd > *maxd) *maxd = dd;
            if (got != want) bad++;
        }
    }
    rocket_bo_fini(buf->fd, &buf->bo);
    return bad;
}

static int place_gate(int fd)
{
    int i, fails = 0;

    printf("place: a pool writes a SLICE of a shared buffer at the buffer's own\n"
           "       channel-group stride — the PLANE, which is what a concatenation's\n"
           "       convolution operands write\n");
    for (i = 0; i < N_PLACECASE; i++) {
        const struct place_case *P = &PLACECASE[i];
        rocket_pool_desc d;
        rocket_pool_int8_rk3576_handle *h = NULL;
        rocket_rk3576_cube buf, slice, narrow;
        int8_t *in = NULL, *ref = NULL, *got = NULL, *got2 = NULL;
        size_t in_n, out_n, k, slice_off, slice_bytes, buf_bytes;
        unsigned oh, ow, groups;
        int bad = 0, bad2 = 0, spill = 0, refused_short = 0, maxd = 0, placed = 0;

        memset(&buf, 0, sizeof buf);
        memset(&d, 0, sizeof d);
        d.c = (int)P->c; d.ih = (int)P->ih; d.iw = (int)P->iw;
        d.kh = (int)P->kh; d.kw = (int)P->kw;
        d.stride_y = (int)P->sy; d.stride_x = (int)P->sx;
        d.pad_top = d.pad_left = d.pad_bottom = d.pad_right = (int)P->pad;
        d.method = P->method;
        d.avg_exclude_pad = P->exclude_pad;
        oh = (unsigned)rocket_pool_oh(&d);
        ow = (unsigned)rocket_pool_ow(&d);
        in_n  = (size_t)P->c * P->ih * P->iw;
        out_n = (size_t)P->c * oh * ow;
        groups = (P->c + C2 - 1u) / C2;

        in = malloc(in_n); ref = malloc(out_n); got = malloc(out_n); got2 = malloc(out_n);
        if (!in || !ref || !got || !got2) { fails++; goto done; }
        for (k = 0; k < in_n; k++)
            in[k] = (int8_t)((int)((k * 37u + (k >> 5) * 11u) % 251u) - 125);
        rocket_pool_ref_int8_rk3576(&d, P->in_zp, in, ref);

        h = rocket_pool_int8_pack_rk3576(fd, &d, P->in_zp);
        if (!h) { printf("  FAIL %-14s the handle would not pack\n", P->name);
                  fails++; goto done; }
        if (rocket_rk3576_cube_alloc(fd, P->bufc, oh, ow, &buf) != ROCKET_OK) {
            printf("  FAIL %-14s the shared buffer would not allocate\n", P->name);
            fails++; goto done;
        }
        buf_bytes   = (size_t)((P->bufc + C2 - 1u) / C2) * buf.surf_elems * C2;
        slice_off   = (size_t)(P->slot / C2) * buf.surf_elems * C2;
        slice_bytes = (size_t)groups * buf.surf_elems * C2;
        /* Poison the WHOLE buffer, bracketed — a bare memset races the PPU's DMA. */
        rocket_bo_prep(fd, &buf.bo, 1, 0);
        memset(buf.bo.ptr, PLACE_POISON, buf_bytes);
        rocket_bo_fini(fd, &buf.bo);

        /* THE ARM THAT MUST SHOW A POSITIVE: a stride one element under the plane. */
        if (rocket_rk3576_cube_slice(&buf, P->slot, P->c, &narrow) == ROCKET_OK) {
            narrow.surf_elems = (size_t)oh * ow - 1u;
            refused_short = rocket_pool_int8_cube_out_at_rk3576(h, &narrow) != ROCKET_OK;
        }

        if (rocket_rk3576_cube_slice(&buf, P->slot, P->c, &slice) != ROCKET_OK) {
            printf("  FAIL %-14s the slice would not cut\n", P->name);
            fails++; goto done;
        }
        placed = rocket_pool_int8_cube_out_at_rk3576(h, &slice) == ROCKET_OK;
        if (!placed) {
            printf("  FAIL %-14s c=%-3u %2ux%-2u -> %2ux%-2u (plane %u, round4 %u)  the "
                   "entry REFUSED the placement\n",
                   P->name, P->c, P->iw, P->ih, ow, oh, oh * ow, round4(oh * ow));
            fails++; goto done;
        }

        /* A PLACED HANDLE LEAVES ITS OUTPUT IN THE CUBE and de-scatters nothing, so the
         * comparison is against the cube the model implies rather than against a row-major
         * tensor: channel `slot + ch` of the buffer, at the buffer's own group stride. */
        if (rocket_pool_int8_prepacked_rk3576(fd, h, in, NULL) != ROCKET_OK) {
            printf("  FAIL %-14s the first placed call failed\n", P->name);
            fails++; goto done;
        }
        bad = place_score(&buf, P->slot, P->c, oh, ow, ref, &maxd);
        /* Twice, because the second call is the first to reuse a held surface. */
        if (rocket_pool_int8_prepacked_rk3576(fd, h, in, NULL) != ROCKET_OK) {
            printf("  FAIL %-14s the second placed call failed\n", P->name);
            fails++; goto done;
        }
        bad2 = place_score(&buf, P->slot, P->c, oh, ow, ref, &maxd);
        /* NOTHING OUTSIDE THE SLICE MOVED. */
        rocket_bo_prep(fd, &buf.bo, 0, 2000000000ull);
        for (k = 0; k < buf_bytes; k++) {
            if (k >= slice_off && k < slice_off + slice_bytes) continue;
            if (((const unsigned char *)buf.bo.ptr)[k] != PLACE_POISON) spill++;
        }
        rocket_bo_fini(fd, &buf.bo);
        (void)got; (void)got2;

        printf("  %-4s %-14s c=%-3u %2ux%-2u -> %2ux%-2u  plane %-4u round4 %-4u  "
               "%s, %d and %d wrong of %zu scored (maxdiff %d), %d byte(s) outside the "
               "slice moved, a short stride %s\n",
               (!bad && !bad2 && !spill && refused_short) ? "PASS" : "FAIL",
               P->name, P->c, P->iw, P->ih, ow, oh, oh * ow, round4(oh * ow),
               oh * ow == round4(oh * ow) ? "stride == round4 (control)"
                                          : "stride BELOW round4",
               bad, bad2, out_n, maxd, spill,
               refused_short ? "is refused" : "was ACCEPTED");
        if (bad || bad2 || spill || !refused_short) fails++;
done:
        if (h) rocket_pool_int8_free_rk3576(fd, h);
        if (buf.bo.ptr) rocket_rk3576_cube_free(fd, &buf);
        free(in); free(ref); free(got); free(got2);
    }
    printf("\n== place: %d cell(s) failed ==\n", fails);
    return fails;
}

/* ===========================================================================
 * `dst` — WHERE THE PPU PUTS EACH ATOM, read out rather than swept.
 *
 * A pooling task's output width is capped, so a wide plane has to be split by COLUMNS —
 * and whether the slices can write into ONE surface, or have to be stitched, is a
 * question about the DESTINATION address function. The emitter writes two registers with
 * the same plane-derived value, `0x607C` and `0x6084`, and neither has ever been driven
 * to anything else.
 *
 * A max pool at kernel 1 stride 1 is an identity copy, so a cube whose every atom carries
 * its own index makes the destination map an OUTPUT: scan the surface for each code and
 * the address function falls out. The arms move one register at a time and then both, so
 * a stride that only takes effect in company is visible — that has already cost one
 * decode on this part.
 *
 * What is being asked: is either register a DDR LINE stride (rows jump by it), a channel
 * GROUP stride (groups jump by it), or inert?
 *
 * AND AT WHICH VALUES. The first reading of this map was taken at an 8x4 plane, whose 32
 * elements are already a multiple of four, and every arm moved the register by four atoms
 * at a time — so "honoured at any value" was a claim over multiples of 64 bytes. What a
 * shared concatenation buffer wants is the plane EXACTLY: a convolution's output surface
 * stride is `ow*oh` and every vendor pooling program carries `round4(ow*oh)`, so a 17x17
 * plane is 289 against 292 and one operand of the buffer disagrees with the rest. The
 * second geometry here is 5x3 — 15 elements, 240 bytes, neither a multiple of four atoms
 * nor of a 64-byte line — and its `= the plane` arm is that question.
 * ==========================================================================*/
#define DST_MAX_PX  32
#define DST_MAX_G    2

/* One arm: patch the two registers to the given values IN ATOMS and read back where each
 * atom landed. Returns the number of atoms that never appeared. */
static int dst_arm(int fd, int diw, int dih, int dc, const char *name,
                   int v0_atoms, int v1_atoms)
{
    const int npx = diw * dih, ng = dc / (int)C2;
    const unsigned out_surf = round4((unsigned)npx);
    const size_t nat = (size_t)out_surf * (unsigned)ng * 4;  /* four times the room */
    const size_t out_bytes = nat * C2;
    rocket_bo bo_in = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_POOL_TASK_OPS] = {0};
    uint32_t in_h[2], out_h[1];
    pool_params_rk3576_t p = {0};
    long off[DST_MAX_G][DST_MAX_PX];
    unsigned i;
    int g, px, missing = ng * npx;

    if (npx > DST_MAX_PX || ng > DST_MAX_G) { printf("   geometry too large\n"); return -1; }
    if (rocket_bo_alloc(fd, (size_t)out_surf * C2 * (unsigned)ng, &bo_in) < 0) return -1;
    if (rocket_bo_alloc(fd, out_bytes, &bo_o) < 0) return -1;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) return -1;

    /* Atom (g, p) carries the code g*npx + p + 1 in all sixteen lanes. */
    rocket_bo_prep(fd, &bo_in, 1, 0);
    for (g = 0; g < ng; g++)
        for (px = 0; px < npx; px++)
            memset((char *)bo_in.ptr + ((size_t)g * out_surf + (unsigned)px) * C2,
                   g * npx + px + 1, C2);
    rocket_bo_fini(fd, &bo_in);

    p.iw = (uint16_t)diw; p.ih = (uint16_t)dih; p.c = (uint16_t)dc;
    p.ow = (uint16_t)diw; p.oh = (uint16_t)dih;
    p.kw = p.kh = 1; p.stride_x = p.stride_y = 1;
    p.mode = ROCKET_RK3576_POOL_MAX;
    p.input_dma = bo_in.dma_address;
    p.output_dma = bo_o.dma_address;
    p.ppu_dst_reg = 0x6070;
    p.tasks = ops;
    if (gen_pool_rk3576(&p) != 0) { printf("   generator refused\n"); return -1; }
    /* Patch the two words in place: an NPUOP is op<<48 | value<<16 | reg. */
    for (i = 0; i < p.task_count; i++) {
        unsigned reg = (unsigned)(ops[i] & 0xffffu);
        if (reg == 0x607Cu || reg == 0x6084u)
            ops[i] = (ops[i] & 0xffff00000000ffffull) |
                     ((uint64_t)(uint32_t)((reg == 0x607Cu ? v0_atoms : v1_atoms)
                                           * (int)C2) << 16);
    }

    rocket_bo_prep(fd, &bo_r, 1, 0);
    memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &bo_r);
    rocket_bo_prep(fd, &bo_o, 1, 0);
    memset(bo_o.ptr, SENTINEL, out_bytes);
    rocket_bo_fini(fd, &bo_o);

    in_h[0] = bo_in.handle; in_h[1] = bo_r.handle;
    out_h[0] = bo_o.handle;
    if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 2, out_h, 1, 2000) != 0) {
        printf("   submit failed\n"); return -1;
    }
    if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
        printf("   PREP_BO timed out\n"); return -1;
    }

    for (g = 0; g < ng; g++) for (px = 0; px < npx; px++) off[g][px] = -1;
    {
        const unsigned char *o = (const unsigned char *)bo_o.ptr;
        size_t at;
        for (at = 0; at + C2 <= out_bytes; at += C2) {
            unsigned char v = o[at];
            unsigned k, same = 1;
            if (v == SENTINEL || v == 0 || v > (unsigned char)(ng * npx)) continue;
            for (k = 1; k < C2; k++) if (o[at + k] != v) { same = 0; break; }
            if (!same) continue;
            off[(v - 1) / npx][(v - 1) % npx] = (long)at;
        }
    }
    {
        long a00 = off[0][0], astride = -1, gstride = -1, rstride = -1;
        int landed = 0;
        for (g = 0; g < ng; g++)
            for (px = 0; px < npx; px++) if (off[g][px] >= 0) landed++;
        if (off[0][1] >= 0 && a00 >= 0) astride = off[0][1] - a00;
        if (ng > 1 && off[1][0] >= 0 && a00 >= 0) gstride = off[1][0] - a00;
        if (npx > diw && off[0][diw] >= 0 && a00 >= 0) rstride = off[0][diw] - a00;
        missing = ng * npx - landed;
        printf("   %-30s 0x607C = %4d B  0x6084 = %4d B   %2d of %2d atoms; "
               "atom +%ld, row +%ld, GROUP +%ld\n",
               name, v0_atoms * (int)C2, v1_atoms * (int)C2, landed, ng * npx,
               astride, rstride, gstride);
    }
    rocket_bo_free(fd, &bo_in); rocket_bo_free(fd, &bo_o); rocket_bo_free(fd, &bo_r);
    sleep_ms(60);
    return missing;
}

static int dst_map(int fd)
{
    /* GEOMETRY A — 8x4, 32 elements, already a multiple of four atoms. The original
     * readout; reproduced so the two are one table. */
    const int aw = 8, ah = 4, ac = 32, anp = aw * ah, ader = (int)round4((unsigned)anp);
    /* GEOMETRY B — 5x3, 15 elements. round4 is 16, so `= the plane` is the arm that has
     * never been driven: 240 bytes, neither four atoms nor a 64-byte line. */
    const int bw = 5, bh = 3, bc = 32, bnp = bw * bh, bder = (int)round4((unsigned)bnp);
    int lost = 0, r;

    printf("dst: a k1 s1 max pool is an identity copy, so every atom's destination is\n"
           "     readable. Two planes: %dx%d (%d elements, round4 = %d) and %dx%d (%d,\n"
           "     round4 = %d). The arms move ONE register, then both.\n\n",
           aw, ah, anp, ader, bw, bh, bnp, bder);

    printf("  %dx%d c%d — derived stride %d bytes\n", aw, ah, ac, ader * (int)C2);
    if ((r = dst_arm(fd, aw, ah, ac, "control (both derived)", ader, ader)) < 0) return 1;
    lost += r;
    if ((r = dst_arm(fd, aw, ah, ac, "0x607C + 4 atoms", ader + 4, ader)) < 0) return 1;
    lost += r;                                    /* above the plane: none may go missing */
    if ((r = dst_arm(fd, aw, ah, ac, "0x6084 + 4 atoms", ader, ader + 4)) < 0) return 1;
    lost += r;                                    /* 0x6084 is inert: none may go missing */
    if ((r = dst_arm(fd, aw, ah, ac, "both + 4 atoms", ader + 4, ader + 4)) < 0) return 1;
    lost += r;
    if ((r = dst_arm(fd, aw, ah, ac, "0x607C = one row (POSITIVE)", aw, ader)) < 0) return 1;
    if (r == 0) { printf("      ^ this arm is UNDER the plane and must lose atoms\n"); lost++; }

    printf("\n  %dx%d c%d — derived stride %d bytes; the plane is %d\n",
           bw, bh, bc, bder * (int)C2, bnp * (int)C2);
    if ((r = dst_arm(fd, bw, bh, bc, "control (both derived)", bder, bder)) < 0) return 1;
    lost += r;
    if ((r = dst_arm(fd, bw, bh, bc, "0x607C = THE PLANE", bnp, bder)) < 0) return 1;
    lost += r;
    if ((r = dst_arm(fd, bw, bh, bc, "both = THE PLANE", bnp, bnp)) < 0) return 1;
    lost += r;
    if ((r = dst_arm(fd, bw, bh, bc, "0x607C = plane + 2 (odd)", bnp + 2, bder)) < 0) return 1;
    lost += r;
    if ((r = dst_arm(fd, bw, bh, bc, "0x607C = plane - 1 (POSITIVE)", bnp - 1, bder)) < 0)
        return 1;
    if (r == 0) { printf("      ^ this arm is UNDER the plane and must lose atoms\n"); lost++; }

    printf("\n== %d atom(s) went missing where every one had to land ==  this probe\n"
           "   REPORTS a map; it is not a gate. A `GROUP +` equal to the programmed\n"
           "   stride is the register being honoured at that value.\n", lost);
    return 0;
}


/* ---- a COLUMN-SPLIT plane takes a cube IN ---------------------------------------
 * A plane wider than one pooling task's output-width allowance is packed as several
 * handles, each owning a column window at both ends. The refusal that used to sit on the
 * whole handle was stated over its OUTPUT — "one surface per slice" — and that is true and
 * is still refused; the INPUT side is a different question, because a slice's window of a
 * producer's surface is the ordinary pitched cube. The PPU carries what the windows consume
 * (`0x600C`) separately from the DDR line stride (`0x7024`) and honours a base part way
 * into a row, so a slice reads its columns in place.
 *
 * The cube is built HERE rather than taken from a producer handle, so the reference is the
 * CPU model of the part's own arithmetic and not another program that could be wrong the
 * same way. Every cell scores against that model, and the row-major arm of the SAME handle
 * is the arm that must succeed — when it fails the finding is about the harness.
 *
 * The surplus columns of a pitched cell are filled with a value no live column carries, so
 * a program that read them would be wrong rather than lucky.
 */
struct split_case {
    const char *name;
    unsigned c, ih, iw, kh, kw, sy, sx;
    unsigned pad;                 /* on every edge */
    int      method, exclude_pad, in_zp;
    unsigned gap, col;            /* the cube's row pitch surplus, and its first column */
};

static const struct split_case SPLITCASE[] = {
    /* The CONTROL: inside the allowance, so nothing splits and the cube-in path is the
     * one that has always run. It must pass or the harness is what is being measured. */
    { "unsplit-ctl",  64u,  32u,  32u, 3u,3u, 1u,1u, 1u, POOL_METHOD_AVG, 1, 0,   0u, 0u },
    /* Inception V3's first max pool: 147 wide at VALID pad, two slices. */
    { "iv3-maxpool",  64u, 147u, 147u, 3u,3u, 2u,2u, 0u, POOL_METHOD_MAX, 0, -12, 0u, 0u },
    /* Inception V3's module average pool: 35 wide, padded, pad EXCLUDED — the one mode
     * whose divisor varies with position, so it also exercises the lag check per slice. */
    { "iv3-avgnopad",192u,  35u,  35u, 3u,3u, 1u,1u, 1u, POOL_METHOD_AVG, 1, 7,   0u, 0u },
    /* A count-include-pad average at the same width: a different divisor, same split. */
    { "avg-incl",     32u,  35u,  35u, 3u,3u, 1u,1u, 1u, POOL_METHOD_AVG, 0, 0,   0u, 0u },
    /* MORE THAN TWO SLICES, and an odd width so the remainder is spread. */
    { "five-slice",   32u,  40u, 147u, 3u,3u, 1u,1u, 1u, POOL_METHOD_MAX, 0, 5,   0u, 0u },
    /* The COMPOSITION: a producer whose surface is wider than its tensor and whose plane
     * starts part way into the row, split as well. Neither quantity is the split's. */
    { "pitched-split",64u,  24u,  70u, 2u,2u, 1u,1u, 0u, POOL_METHOD_MAX, 0, 0,  11u, 3u },
};
#define N_SPLITCASE ((int)(sizeof SPLITCASE / sizeof SPLITCASE[0]))

static int split_gate(int fd)
{
    int i, fails = 0;

    printf("split: a column-split pool reads a producer's cube; its OUTPUT stays refused\n");
    for (i = 0; i < N_SPLITCASE; i++) {
        const struct split_case *S = &SPLITCASE[i];
        rocket_pool_desc d;
        rocket_pool_int8_rk3576_handle *h = NULL;
        rocket_rk3576_cube cube;
        rocket_bo bo;
        int8_t *in = NULL, *ref = NULL, *rowm = NULL, *got = NULL, *got2 = NULL;
        size_t in_n, out_n, k, cube_bytes;
        unsigned oh, ow, slices, groups, pitch, surf_elems, g, y, x;
        int bad = 0, bad2 = 0, badrm = 0, maxd = 0, rc = 0, refused_out = 0;

        memset(&d, 0, sizeof d);
        memset(&bo, 0, sizeof bo);
        d.c = (int)S->c; d.ih = (int)S->ih; d.iw = (int)S->iw;
        d.kh = (int)S->kh; d.kw = (int)S->kw;
        d.stride_y = (int)S->sy; d.stride_x = (int)S->sx;
        d.pad_top = d.pad_left = d.pad_bottom = d.pad_right = (int)S->pad;
        d.method = S->method;
        d.avg_exclude_pad = S->exclude_pad;
        oh = (unsigned)rocket_pool_oh(&d);
        ow = (unsigned)rocket_pool_ow(&d);
        slices = rocket_pool_int8_rk3576_ow_slices(&d);
        if (ow <= rocket_pool_int8_rk3576_max_ow(&d)) slices = 0u;
        in_n  = (size_t)S->c * S->ih * S->iw;
        out_n = (size_t)S->c * oh * ow;

        in = malloc(in_n); ref = malloc(out_n); rowm = malloc(out_n);
        got = malloc(out_n); got2 = malloc(out_n);
        if (!in || !ref || !rowm || !got || !got2) { fails++; goto done; }
        for (k = 0; k < in_n; k++)
            in[k] = (int8_t)((int)((k * 37u + (k >> 5) * 11u) % 251u) - 125);
        rocket_pool_ref_int8_rk3576(&d, S->in_zp, in, ref);

        h = rocket_pool_int8_pack_rk3576(fd, &d, S->in_zp);
        if (!h) { printf("  FAIL %-14s the handle would not pack\n", S->name);
                  fails++; goto done; }

        /* THE ARM THAT MUST SUCCEED. */
        memset(rowm, 0x5A, out_n);
        if (rocket_pool_int8_prepacked_rk3576(fd, h, in, rowm) != ROCKET_OK) {
            printf("  FAIL %-14s the ROW-MAJOR arm failed — this is about the harness\n",
                   S->name);
            fails++; goto done;
        }
        for (k = 0; k < out_n; k++) if (rowm[k] != ref[k]) badrm++;

        /* THE OUTPUT SIDE STAYS REFUSED, and that is asserted rather than assumed: the
         * destination surface stride is derived from the plane with no register to move
         * it, so the slices cannot write one plane between them. */
        if (slices) {
            rocket_rk3576_cube outc;
            refused_out = (rocket_pool_int8_cube_out_rk3576(h, 1) != ROCKET_OK) &&
                          (rocket_pool_int8_cube_of_rk3576(h, &outc) != ROCKET_OK);
        } else refused_out = 1;

        /* The cube a producer would hand it, built by hand so the reference is the model. */
        pitch      = S->iw + S->gap;
        surf_elems = (size_t)S->ih * pitch > 0xFFFFFFu ? 0u : S->ih * pitch;
        groups     = (S->c + C2 - 1u) / C2;
        cube_bytes = (size_t)groups * surf_elems * C2;
        if (!surf_elems || rocket_bo_alloc(fd, cube_bytes, &bo) < 0) {
            printf("  FAIL %-14s the cube would not allocate\n", S->name);
            fails++; goto done;
        }
        rocket_bo_prep(fd, &bo, 1, 0);
        memset(bo.ptr, 0x7F, cube_bytes);          /* poison every surplus column */
        for (g = 0; g < S->c; g++)
            for (y = 0; y < S->ih; y++)
                for (x = 0; x < S->iw; x++)
                    ((int8_t *)bo.ptr)[(size_t)(g / C2) * surf_elems * C2 +
                                       (size_t)C2 * ((size_t)y * pitch + x + S->col) +
                                       (g % C2)] =
                        in[(size_t)g * S->ih * S->iw + (size_t)y * S->iw + x];
        rocket_bo_fini(fd, &bo);

        memset(&cube, 0, sizeof cube);
        cube.fd = fd; cube.c = S->c; cube.h = S->ih; cube.w = S->iw;
        cube.groups = groups; cube.surf_elems = surf_elems;
        cube.bo = bo; cube.off = 0;
        cube.pitch_w = (S->gap || S->col) ? pitch : 0u;
        cube.col_off = S->col;

        rc = rocket_pool_int8_cube_in_rk3576(h, &cube);
        if (rc != ROCKET_OK) {
            printf("  FAIL %-14s c=%-4u %ux%-3u k%ux%u s%ux%u pad%u -> %ux%-3u  "
                   "%u slice(s): the cube-in was REFUSED\n",
                   S->name, S->c, S->iw, S->ih, S->kw, S->kh, S->sx, S->sy, S->pad,
                   ow, oh, slices);
            fails++; goto done;
        }
        /* TWICE: the second call is the first to reuse a held surface and a held stamp. */
        memset(got, 0x5A, out_n); memset(got2, 0x5A, out_n);
        if (rocket_pool_int8_prepacked_rk3576(fd, h, NULL, got) != ROCKET_OK ||
            rocket_pool_int8_prepacked_rk3576(fd, h, NULL, got2) != ROCKET_OK) {
            printf("  FAIL %-14s the cube-in arm returned an error\n", S->name);
            fails++; goto done;
        }
        for (k = 0; k < out_n; k++) {
            int dd = got[k] - ref[k];
            if (dd) { bad++; if (dd < 0) dd = -dd; if (dd > maxd) maxd = dd; }
            if (got2[k] != got[k]) bad2++;
        }
        printf("  %-4s %-14s c=%-4u %ux%-3u k%ux%u s%ux%u pad%u -> %ux%-3u  %u slice(s)"
               "%s%s  %s\n",
               (bad || bad2 || badrm || !refused_out) ? "FAIL" : "PASS",
               S->name, S->c, S->iw, S->ih, S->kw, S->kh, S->sx, S->sy, S->pad, ow, oh,
               slices, S->gap ? ", pitched" : "", S->col ? ", offset" : "",
               badrm ? "the ROW-MAJOR arm differs from the model"
                     : !refused_out ? "the OUTPUT side was ACCEPTED for a split handle"
                     : bad ? "the cube-in arm differs from the model"
                     : bad2 ? "the second call differs from the first"
                            : "bit-exact against the model, both calls");
        if (bad) printf("       %d of %zu wrong, max |diff| %d\n", bad, out_n, maxd);
        if (bad || bad2 || badrm || !refused_out) fails++;
done:
        if (h) rocket_pool_int8_free_rk3576(fd, h);
        if (bo.ptr) rocket_bo_free(fd, &bo);
        free(in); free(ref); free(rowm); free(got); free(got2);
        sleep_ms(120);
    }
    printf("== %d passed, %d failed ==\n", N_SPLITCASE - fails, fails);
    return fails ? 1 : 0;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *mode = argc > 1 ? argv[1] : "sweep";
    int fd, rc;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_pool_probe: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_pool_probe: no NPU device — skipping\n"); return 2; }

    if (!strcmp(mode, "lib")) {
        rc = lib_gate(fd);
    } else if (!strcmp(mode, "split")) {
        rc = split_gate(fd);
    } else if (!strcmp(mode, "place")) {
        rc = place_gate(fd);
    } else if (!strcmp(mode, "pad")) {
        rc = pad_map(fd);
    } else if (!strcmp(mode, "bound")) {
        rc = bound_sweep(fd);
    } else if (!strcmp(mode, "rect")) {
        rc = rect_sweep(fd);
    } else if (!strcmp(mode, "avg")) {
        rc = avg_hazard(fd);
    } else if (!strcmp(mode, "rate")) {
        rc = rate_map(fd);
    } else if (!strcmp(mode, "dst")) {
        rc = dst_map(fd);
    } else if (!strcmp(mode, "gate")) {
        const char *e = getenv("ROCKET_POOL_DST_REG");
        unsigned reg = e && *e ? (unsigned)strtol(e, NULL, 0) : 0x6070u;
        rc = probe_gate(fd, reg, getenv("ROCKET_POOL_VERBOSE") != NULL);
    } else {
        rc = probe_sweep(fd);
    }
    rocket_close(fd);
    return rc > 0 ? 1 : (rc < 0 ? 1 : 0);
}
