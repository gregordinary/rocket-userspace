// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_drain_probe.c — which RK3576 programs raise the DPU's completion, and whether a
 * job that returns has finished writing.
 *
 * The RK3576 `rocket` driver retires a job when its writing block raises its completion.
 * ROCKET_JOB_NO_DPU_DONE tells it the program raises none, and it then waits a blind
 * settle past PC_DONE instead (`dpu_blind_us`, 250 us). PC_DONE on a one-task job is the
 * kick being over, not the convolution: the CNA/CORE/DPU pipeline still has the whole
 * program to run. So the hint on a program that DOES raise a completion ends the wait
 * while the DPU is still writing, and the caller reads a full, correctly sized surface
 * whose last rows are the output buffer's previous contents. Without the hint, rocket
 * 1.6.0 (`dpu_grace_us` 0) waits for the block; 1.5.0 and a non-zero grace put a deadline
 * on that wait instead, which is the same failure at a longer threshold.
 *
 * Two modes.
 *
 *   raw SPEC...   the mechanism, per program. One program, built with the library's own
 *                 generator and helpers, operands 1.0 (int8: 3 and 1), submitted alone
 *                 after ROCKET_DP_RECOVER_MS of idle (the power domain suspends, so the
 *                 part is unpoisoned), its output stamped 0xA5 first. Per rep: the
 *                 submit-to-fence wait, the output elements still holding the stamp at
 *                 the fence, and the same count plus a content hash after
 *                 ROCKET_DP_SETTLE_MS. Per arm: the kernel's `retiring it` lines. A
 *                 program that raises its completion waits its execution and never the
 *                 125 ms backstop; one that raises none waits the backstop without the
 *                 hint and logs a line per submit. A tail at the fence that is gone after
 *                 the settle, with the hash moved, is a job that returned mid-drain.
 *
 *   gate [SHAPE...]  the regression gate: rocket_conv2d_fp16_rk3576() at the drain-heavy
 *                 shapes, ROCKET_DP_REPS calls each, two alternating inputs, every
 *                 element of every call scored against an exact host sum (small-integer
 *                 operands, every partial an exact fp32, the one rounding the final fp16
 *                 narrowing on both sides). Any wrong element or any refusal fails it.
 *
 * SPEC  = kind:ic:oc:ih:iw:k[:h]    stride 1, pad k/2
 *   kind  fp16d   the ic-split program (ic 16, oc 16 or 32)
 *         fp16fc  the packed-image first conv (ic <= 4, iw % 16 == 0, oc 16 or 32)
 *         i32     the narrow int32 writer (ic, oc multiples of 32, k 1)
 *         i32w    the wide int32 writer (the same)
 *         int8d   the one-byte direct program, the control (ic, oc multiples of 32)
 *   :h    submit with ROCKET_JOB_NO_DPU_DONE, where the kernel takes it
 * SHAPE = ic:oc:ih:iw:k             stride 1, pad k/2 (the gate's default list if none)
 *
 * The int32 writers' stamp count is over the whole buffer and they deliver only part of
 * it (the narrow writer 8 of every 32 channels), so for them read "wrote" and the wait,
 * not the count.
 *
 * Env: ROCKET_DP_REPS (raw 10, gate 6), ROCKET_DP_RECOVER_MS (300), ROCKET_DP_SETTLE_MS
 * (50), ROCKET_DP_POISON=1 (raw: submit the program once, flags 0, straight before each
 * measured submit, so the measured one follows a wide-output job with no power cycle),
 * ROCKET_DP_LAG=1 (raw, fp16 kinds: instead of the fence-time scan, poll the last element
 * of every 8-channel plane from the fence until it lands, and report how long that took),
 * ROCKET_DP_GAP_MS (gate: idle before each call, 0; past the power domain's autosuspend
 * each call is priced as an isolated one rather than as one of a back-to-back loop).
 *
 * What it does NOT show. The raw mode's operands are uniform, so it says whether and when
 * elements land, never what they compute; the gate says that. The mechanism is read on one
 * core (core 0, the only bound one on this board) at one clock.
 *
 * Exit: 0 ran (raw) or passed (gate), 1 a gate failure or a setup error, 2 no NPU or not
 * an RK3576.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"

#define STAMP 0xA5u

enum { K_FP16D, K_FP16FC, K_I32, K_I32W, K_INT8D, K_N };
static const char *const KNAME[K_N] = { "fp16d", "fp16fc", "i32", "i32w", "int8d" };

static int env_int(const char *name, int dflt)
{
    const char *e = getenv(name);
    return (e && *e) ? (int)strtol(e, NULL, 0) : dflt;
}

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static void sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* The kernel's backstop retirements this boot, read through the journal (group adm). The
 * sleep lets the journal catch up with a line the last job just caused. -1 unreadable. */
static long klog_retired(void)
{
    FILE *f;
    long n = -1;
    sleep_ms(200);
    f = popen("journalctl -b -k -q --no-pager 2>/dev/null | grep -c 'retiring it'", "r");
    if (!f) return -1;
    if (fscanf(f, "%ld", &n) != 1) n = -1;
    if (pclose(f) == -1) n = -1;
    return n;
}

/* ---- raw: one program ------------------------------------------------------------ */

struct prog {
    int          kind;
    unsigned     ic, oc, ih, iw, k;
    rocket_bo    in, w, b, o, r;
    conv_params_t p;
    uint64_t     ops[RK3576_CONV_TASK_OPS];
    uint32_t     in_h[4], out_h[1];
    size_t       ext;         /* bytes of the output the program owns (fp16: the surface) */
    unsigned     esz;         /* element size the stamp count reads                       */
};

static void fill_u16(rocket_bo *bo, size_t bytes, uint16_t v)
{
    size_t i;
    for (i = 0; i < bytes / 2; i++) ((uint16_t *)bo->ptr)[i] = v;
}

static int prog_build(int fd, struct prog *pr)
{
    const unsigned pad = pr->k / 2, oh = pr->ih, ow = pr->iw;
    size_t in_bytes, w_bytes, coeff, obytes;
    unsigned ocreg, icreg;
    int fp16 = pr->kind == K_FP16D || pr->kind == K_FP16FC;
    int32_t *bias = NULL;
    int rc;

    if (fp16) {
        if (pr->oc % 16u || pr->oc > 32u) {
            fprintf(stderr, "fp16 kinds take oc 16 or 32 (one program writes 32)\n");
            return -1;
        }
        if (pr->kind == K_FP16D && pr->ic != ROCKET_RK3576_FP16_IC_SLICE) {
            fprintf(stderr, "fp16d contracts exactly %u input channels\n",
                    ROCKET_RK3576_FP16_IC_SLICE);
            return -1;
        }
        if (pr->kind == K_FP16FC && (pr->ic > 4u || pr->iw % 16u)) {
            fprintf(stderr, "fp16fc takes ic <= 4 and iw a multiple of 16\n");
            return -1;
        }
        ocreg = rocket_rk3576_fp16_pad_oc(pr->oc);
        icreg = pr->ic;
        in_bytes = pr->kind == K_FP16D
                 ? (size_t)(ROCKET_RK3576_FP16_IC_SLICE / 8u) * pr->ih * pr->iw * 8u * 2u
                 : (size_t)pr->ih * pr->iw * pr->ic * 2u;
        w_bytes = pr->kind == K_FP16D
                ? rocket_rk3576_fp16_slice_weight_bytes(pr->oc, pr->ic, pr->k, pr->k)
                : rocket_rk3576_weight_argb_fp16_bytes(ocreg, pr->k, pr->k);
        coeff = rocket_rk3576_coeff_bytes(ocreg);
        pr->ext = rocket_rk3576_fp16_out_bytes(pr->oc, oh, ow);
        obytes = pr->ext;
        pr->esz = 2;
    } else {
        if (pr->ic % 32u || pr->oc % 32u) {
            fprintf(stderr, "the int8/int32 kinds take ic and oc multiples of 32\n");
            return -1;
        }
        if ((pr->kind == K_I32 || pr->kind == K_I32W) && pr->k != 1u) {
            fprintf(stderr, "the int32 writers are the matmul's 1x1 program\n");
            return -1;
        }
        ocreg = pr->oc; icreg = pr->ic;
        in_bytes = (size_t)icreg * pr->ih * pr->iw;
        w_bytes = (size_t)ocreg * icreg * pr->k * pr->k;
        coeff = rocket_rk3576_coeff_bytes(ocreg);
        pr->ext = (size_t)(ocreg / 16u) * rocket_rk3576_out_surf_elems(ow, oh, 0) * 16u;
        /* The int32 writers emit four bytes an element and the wide one twice the atoms,
         * so the buffer is eight times the int8 surface, as the net gate's injection
         * sizes it: a program writing past what was sized lands in the BO, not a fault. */
        if (pr->kind != K_INT8D) pr->ext *= 8u;
        obytes = pr->ext;
        pr->esz = pr->kind == K_INT8D ? 1u : 4u;
    }

    if (rocket_bo_alloc32(fd, in_bytes, &pr->in) < 0 ||
        rocket_bo_alloc32(fd, w_bytes, &pr->w) < 0 ||
        rocket_bo_alloc32(fd, coeff, &pr->b) < 0 ||
        rocket_bo_alloc32(fd, obytes, &pr->o) < 0 ||
        rocket_bo_alloc32(fd, sizeof pr->ops, &pr->r) < 0) {
        fprintf(stderr, "BO allocation failed\n");
        return -1;
    }

    rocket_bo_prep(fd, &pr->in, 1, 0);
    if (fp16) fill_u16(&pr->in, in_bytes, 0x3C00u);
    else      memset(pr->in.ptr, 3, in_bytes);
    rocket_bo_fini(fd, &pr->in);
    rocket_bo_prep(fd, &pr->w, 1, 0);
    if (fp16) fill_u16(&pr->w, w_bytes, 0x3C00u);
    else      memset(pr->w.ptr, 1, w_bytes);
    rocket_bo_fini(fd, &pr->w);

    if (!fp16) {
        unsigned i;
        bias = calloc(ocreg, sizeof *bias);
        if (!bias) return -1;
        for (i = 0; i < ocreg; i++) bias[i] = (int32_t)i + 1;
    }
    rocket_bo_prep(fd, &pr->b, 1, 0);
    rc = rocket_rk3576_pack_coeff_prec(pr->b.ptr, coeff, bias, ocreg,
                                       fp16 ? precision_float16 : precision_int8);
    rocket_bo_fini(fd, &pr->b);
    free(bias);
    if (rc < 0) { fprintf(stderr, "coefficient pack refused\n"); return -1; }

    memset(&pr->p, 0, sizeof pr->p);
    pr->p.ic = (uint16_t)icreg; pr->p.oc = (uint16_t)ocreg;
    pr->p.ih = (uint16_t)pr->ih; pr->p.iw = (uint16_t)pr->iw;
    pr->p.oh = (uint16_t)oh;     pr->p.ow = (uint16_t)ow;
    pr->p.kh = (uint16_t)pr->k;  pr->p.kw = (uint16_t)pr->k;
    pr->p.stride_y = 1; pr->p.stride_x = 1;
    pr->p.pad_top = (uint8_t)pad; pr->p.pad_left = (uint8_t)pad;
    pr->p.ih_full = (uint16_t)pr->ih; pr->p.oh_full = (uint16_t)oh;
    pr->p.int8_out = (uint8_t)(fp16 ? 0 : 1);
    pr->p.in_scale = 1.0f; pr->p.w_scale = 1.0f; pr->p.out_scale = fp16 ? 1.0f : 64.0f;
    pr->p.input_dma   = (uint32_t)pr->in.dma_address;
    pr->p.weights_dma = (uint32_t)pr->w.dma_address;
    pr->p.bias_dma    = (uint32_t)pr->b.dma_address;
    pr->p.output_dma  = (uint32_t)pr->o.dma_address;
    pr->p.tasks = pr->ops;
    switch (pr->kind) {
    case K_FP16D: case K_FP16FC: rc = gen_conv2d_fp16_rk3576(&pr->p); break;
    case K_I32:   rc = gen_conv2d_int8_rk3576_i32out(&pr->p); break;
    case K_I32W:  rc = gen_conv2d_int8_rk3576_i32out_wide(&pr->p); break;
    default:      rc = gen_conv2d_int8_rk3576(&pr->p); break;
    }
    if (rc != 0) { fprintf(stderr, "the generator refused this geometry\n"); return -1; }
    rocket_bo_prep(fd, &pr->r, 1, 0);
    memcpy(pr->r.ptr, pr->ops, pr->p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &pr->r);
    pr->in_h[0] = pr->in.handle; pr->in_h[1] = pr->w.handle;
    pr->in_h[2] = pr->b.handle;  pr->in_h[3] = pr->r.handle;
    pr->out_h[0] = pr->o.handle;
    return 0;
}

static void prog_free(int fd, struct prog *pr)
{
    rocket_bo_free(fd, &pr->in); rocket_bo_free(fd, &pr->w); rocket_bo_free(fd, &pr->b);
    rocket_bo_free(fd, &pr->o);  rocket_bo_free(fd, &pr->r);
}

/* Elements of the owned extent still holding the stamp, the lowest of them, and an FNV-1a
 * hash of the extent. Inside a bracket the caller holds.
 *
 * BACKWARDS, from the last element. The DPU writes the surface front to back, so a job
 * that returned mid-drain has its unwritten elements at the END, and a forward scan reads
 * them last, by which time the writer may have caught up: a forward scan of a k9 surface
 * that the hint returned at 750 us of its 2.1 ms read no tail in 10 of 10. The reader's
 * order is part of this instrument. */
static size_t scan(const struct prog *pr, size_t *first, uint64_t *hash)
{
    const unsigned char *o = (const unsigned char *)pr->o.ptr;
    size_t i, n = pr->ext / pr->esz, left = 0;
    uint64_t h = 1469598103934665603ull;
    *first = (size_t)-1;
    for (i = n; i-- > 0;) {
        const unsigned char *e = o + i * pr->esz;
        unsigned j, st = 1;
        for (j = 0; j < pr->esz; j++) if (e[j] != STAMP) { st = 0; break; }
        if (st) { *first = i; left++; }
    }
    for (i = 0; i < pr->ext; i++) { h ^= o[i]; h *= 1099511628211ull; }
    *hash = h;
    return left;
}

/* How long past the fence the last element of every 8-channel plane took to land, in us:
 * each poll syncs one cache line per plane (the ranged PREP, interface 1.5) and reads its
 * last halfword. `at_fence` is the planes unwritten at the first poll. -1 if a plane was
 * still unwritten `budget_us` after the fence. fp16 kinds only. */
static double tail_lag(int fd, struct prog *pr, double t_fence, unsigned *at_fence,
                       double budget_us)
{
    size_t plane = (size_t)pr->iw * pr->ih * 8u * 2u;   /* bytes per 8-channel plane */
    unsigned np = (unsigned)(pr->ext / plane), j, left, first = 1;
    rocket_bo_range rg[8];
    double t;

    if (np > 8u) np = 8u;
    for (j = 0; j < np; j++) {
        uint64_t last = (uint64_t)(j + 1u) * plane - 2u;
        rg[j].offset = last & ~(uint64_t)63u;
        rg[j].size = 64u;
    }
    *at_fence = 0;
    for (;;) {
        rocket_bo_prep_ranges(fd, &pr->o, rg, np, 2000000000ull);
        left = 0;
        for (j = 0; j < np; j++) {
            const uint16_t *e = (const uint16_t *)((const unsigned char *)pr->o.ptr +
                                                   (size_t)(j + 1u) * plane - 2u);
            if (*e == (uint16_t)(STAMP * 0x0101u)) left++;
        }
        rocket_bo_fini_ranges(fd, &pr->o, rg, np);
        t = now_us();
        if (first) { *at_fence = left; first = 0; }
        if (!left) return t - t_fence;
        if (t - t_fence > budget_us) return -1.0;
    }
}

static int submit(int fd, struct prog *pr, uint32_t flags)
{
    return rocket_submit_matmul_flags(fd, &pr->r, pr->p.task_count, pr->in_h, 4,
                                      pr->out_h, 1, flags);
}

static int raw_arm(int fd, const char *spec)
{
    struct prog pr;
    char kind[16] = {0}, tail[8] = {0};
    int reps = env_int("ROCKET_DP_REPS", 10), recover = env_int("ROCKET_DP_RECOVER_MS", 300);
    int settle = env_int("ROCKET_DP_SETTLE_MS", 50), poison = env_int("ROCKET_DP_POISON", 0);
    int lag = env_int("ROCKET_DP_LAG", 0);
    int n, r, tails = 0, landed = 0, wrote = 0, hashmoved = 0, i;
    uint32_t flags = 0;
    double *wt;
    long k0, k1;
    size_t owned;

    memset(&pr, 0, sizeof pr);
    n = sscanf(spec, "%15[^:]:%u:%u:%u:%u:%u:%7s", kind, &pr.ic, &pr.oc, &pr.ih, &pr.iw,
               &pr.k, tail);
    if (n < 6) { fprintf(stderr, "bad SPEC '%s'\n", spec); return -1; }
    pr.kind = -1;
    for (i = 0; i < K_N; i++) if (!strcmp(kind, KNAME[i])) pr.kind = i;
    if (pr.kind < 0) { fprintf(stderr, "unknown kind '%s'\n", kind); return -1; }
    if (n == 7 && !strcmp(tail, "h")) {
        if (!rocket_no_dpu_done_supported()) {
            fprintf(stderr, "this kernel does not take ROCKET_JOB_NO_DPU_DONE\n");
            return -1;
        }
        flags = ROCKET_JOB_NO_DPU_DONE;
    }
    if (reps < 1) reps = 1;
    if (prog_build(fd, &pr) != 0) { prog_free(fd, &pr); return -1; }
    owned = pr.ext / pr.esz;
    wt = calloc((size_t)reps, sizeof *wt);
    if (!wt) { prog_free(fd, &pr); return -1; }

    printf("%s ic %u oc %u %ux%u k%u (%.1f MMAC)%s%s: %zu elements of %u B, %d reps, "
           "%d ms recover, %d ms settle\n", KNAME[pr.kind], pr.ic, pr.oc, pr.ih, pr.iw,
           pr.k, (double)pr.ih * pr.iw * pr.oc * pr.ic * pr.k * pr.k / 1e6,
           flags ? ", HINT" : ", no hint", poison ? ", POISONED (same program first)" : "",
           owned, pr.esz, reps, recover, settle);
    fflush(stdout);

    k0 = klog_retired();
    for (r = 0; r < reps; r++) {
        size_t f0 = (size_t)-1, f1, left0, left1;
        uint64_t h0, h1;
        double t0, tf, lag_us = 0.0;
        unsigned lag_planes = 0;
        sleep_ms(recover);
        if (poison) {
            if (submit(fd, &pr, 0) != 0 || rocket_bo_prep(fd, &pr.o, 0, 2000000000ull) < 0) {
                fprintf(stderr, "poisoning submit failed\n");
                break;
            }
            rocket_bo_fini(fd, &pr.o);
        }
        rocket_bo_prep(fd, &pr.o, 1, 0);
        memset(pr.o.ptr, STAMP, pr.o.size);
        rocket_bo_fini(fd, &pr.o);
        t0 = now_us();
        if (submit(fd, &pr, flags) != 0) { fprintf(stderr, "submit failed\n"); break; }
        if (rocket_bo_prep(fd, &pr.o, 0, 2000000000ull) < 0) {
            fprintf(stderr, "PREP_BO timed out\n");
            break;
        }
        tf = now_us();
        wt[r] = tf - t0;
        if (lag && pr.esz == 2) {
            /* The lag replaces the fence-time scan: both read the tail, and whichever
             * reads it first is the one that sees it unwritten. */
            rocket_bo_fini(fd, &pr.o);
            lag_us = tail_lag(fd, &pr, tf, &lag_planes, 50000.0);
            left0 = lag_planes ? 1u : 0u;
            h0 = 0;
        } else {
            left0 = scan(&pr, &f0, &h0);
            rocket_bo_fini(fd, &pr.o);
        }
        sleep_ms(settle);
        rocket_bo_prep(fd, &pr.o, 0, 2000000000ull);
        left1 = scan(&pr, &f1, &h1);
        rocket_bo_fini(fd, &pr.o);

        if (lag && pr.esz == 2) {
            tails += lag_planes > 0;
            landed += lag_planes > 0 && lag_us >= 0.0;
            wrote += left1 < owned;
            printf("  rep %2d: wait %9.0f us  planes whose last element was unwritten at "
                   "the fence %u, all landed %s%.0f us after it; %zu stamp-holding after "
                   "the settle\n", r, wt[r], lag_planes, lag_us < 0 ? "NOT within " : "",
                   lag_us < 0 ? 50000.0 : lag_us, left1);
            fflush(stdout);
            continue;
        }
        wrote += left0 < owned;
        tails += left0 > 0 && left0 < owned;
        landed += left0 > 0 && left1 < left0;
        hashmoved += h0 != h1;
        printf("  rep %2d: wait %9.0f us  stamp-holding %7zu at the fence, %7zu after the "
               "settle%s", r, wt[r], left0, left1, h0 != h1 ? "  (hash moved)" : "");
        if (left0 && pr.esz == 2) {
            /* fp16 surface: (ow*oh*8)*(c/8) + 8*(y*ow+x) + (c%8), in halfwords. */
            size_t plane = (size_t)pr.iw * pr.ih * 8u;
            size_t px = (f0 % plane) / 8u;
            printf("  lowest at 8-ch plane %zu row %zu", f0 / plane, px / pr.iw);
        }
        printf("\n");
        fflush(stdout);
    }
    k1 = klog_retired();
    if (r == reps) {
        qsort(wt, (size_t)reps, sizeof *wt, cmp_d);
        printf("  => wait median %.0f us (%.0f .. %.0f); wrote %d/%d; a tail at the fence "
               "%d/%d, landed after it %d, hash moved %d; kernel 'retiring it' %ld\n",
               wt[reps / 2], wt[0], wt[reps - 1], wrote, reps, tails, reps, landed,
               hashmoved, (k0 >= 0 && k1 >= 0) ? k1 - k0 : -1L);
    }
    fflush(stdout);
    free(wt);
    prog_free(fd, &pr);
    return r == reps ? 0 : -1;
}

/* ---- gate: the library entry ------------------------------------------------------ */

/* The shapes the gate drives, ic:oc:ih:iw:k at stride 1, pad k/2. Chosen where the entry's
 * programs run longest per submit: 32-channel tasks at k9-k13 on a plane the planner
 * programs at F 0, one or two input-channel slices. */
static const char *const GATE_SHAPES[] = {
    "16:32:120:64:9",
    "16:32:120:64:11",
    "32:32:120:64:9",
};

static void fp16_fill(_Float16 *in, _Float16 *W, unsigned ic, unsigned oc, unsigned ih,
                      unsigned iw, unsigned k, unsigned salt)
{
    unsigned c, y, x, seed = 0x85EBCA6Bu ^ (ic * 31u + oc * 17u + iw * 7u + ih * 3u + k)
                                         ^ (salt * 0x9E3779B9u);
    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                in[((size_t)c * ih + y) * iw + x] =
                    (_Float16)((int)((c * 7 + y * 13 + x * 3 + salt) % 5) - 2);
    for (c = 0; c < oc * ic * k * k; c++) {
        seed = seed * 1103515245u + 12345u;
        W[c] = (_Float16)((int)((seed >> 16) % 3u) - 1);
    }
}

/* The exact host sum. |sum| <= 2048 everywhere is checked, so every partial is an exact
 * fp32 and the narrowing is the one rounding. Returns -1 if a sum leaves that range.
 * Operands are small integers (fp16_fill), so it runs in int32 over whole rows: a tap is
 * one shifted row-by-row multiply-add of a plane. */
static int fp16_ref(const _Float16 *in, const _Float16 *W, _Float16 *ref, unsigned ic,
                    unsigned oc, unsigned ih, unsigned iw, unsigned k)
{
    const int pad = (int)(k / 2);
    const size_t px = (size_t)ih * iw;
    int32_t *acc = malloc(px * sizeof *acc);
    int8_t *xi = malloc((size_t)ic * px);
    unsigned c, i, kh, kw;
    size_t p;
    int rc = 0;

    if (!acc || !xi) { free(acc); free(xi); return -1; }
    for (p = 0; p < (size_t)ic * px; p++) xi[p] = (int8_t)(int)in[p];
    for (c = 0; c < oc && !rc; c++) {
        memset(acc, 0, px * sizeof *acc);
        for (i = 0; i < ic; i++)
            for (kh = 0; kh < k; kh++)
                for (kw = 0; kw < k; kw++) {
                    int w = (int)W[(((size_t)c * ic + i) * k + kh) * k + kw];
                    int dy = (int)kh - pad, dx = (int)kw - pad, y;
                    int y0 = dy < 0 ? -dy : 0, y1 = dy > 0 ? (int)ih - dy : (int)ih;
                    int x0 = dx < 0 ? -dx : 0, x1 = dx > 0 ? (int)iw - dx : (int)iw;
                    if (!w) continue;
                    for (y = y0; y < y1; y++) {
                        const int8_t *src = xi + (size_t)i * px + (size_t)(y + dy) * iw + dx;
                        int32_t *dst = acc + (size_t)y * iw;
                        int x;
                        for (x = x0; x < x1; x++) dst[x] += w * src[x];
                    }
                }
        for (p = 0; p < px; p++) {
            if (acc[p] > 2048 || acc[p] < -2048) { rc = -1; break; }
            ref[(size_t)c * px + p] = (_Float16)(float)acc[p];
        }
    }
    free(acc); free(xi);
    return rc;
}

static int gate_shape(int fd, const char *shape, int reps, int *bad_calls)
{
    unsigned ic, oc, ih, iw, k, v, c;
    _Float16 *in[2] = {0}, *W[2] = {0}, *ref[2] = {0}, *out = NULL;
    rocket_conv2d_desc d;
    size_t n, nin, nw, i;
    uint64_t u0, w0, u1, w1;
    long k0, k1;
    double tmin = 1e30, tmax = 0, tsum = 0;
    int r, wrong_calls = 0, refused = 0, rc = 0, gap = env_int("ROCKET_DP_GAP_MS", 0);

    if (sscanf(shape, "%u:%u:%u:%u:%u", &ic, &oc, &ih, &iw, &k) != 5) {
        fprintf(stderr, "bad SHAPE '%s'\n", shape);
        return -1;
    }
    n = (size_t)oc * ih * iw; nin = (size_t)ic * ih * iw; nw = (size_t)oc * ic * k * k;
    out = malloc(n * sizeof *out);
    for (v = 0; v < 2; v++) {
        in[v] = malloc(nin * sizeof(_Float16));
        W[v] = malloc(nw * sizeof(_Float16));
        ref[v] = malloc(n * sizeof(_Float16));
        if (!in[v] || !W[v] || !ref[v]) { rc = -1; goto done; }
        fp16_fill(in[v], W[v], ic, oc, ih, iw, k, v);
        if (fp16_ref(in[v], W[v], ref[v], ic, oc, ih, iw, k) != 0) {
            fprintf(stderr, "%s: a host sum leaves the exact range\n", shape);
            rc = -1; goto done;
        }
    }
    if (!out) { rc = -1; goto done; }
    if (!memcmp(ref[0], ref[1], n * sizeof(_Float16))) {
        fprintf(stderr, "%s: the two inputs have one answer, so the check is vacuous\n",
                shape);
        rc = -1; goto done;
    }

    memset(&d, 0, sizeof d);
    d.ic = (int)ic; d.oc = (int)oc; d.ih = (int)ih; d.iw = (int)iw;
    d.kh = (int)k; d.kw = (int)k; d.stride_y = 1; d.stride_x = 1;
    d.pad_top = (int)(k / 2); d.pad_left = (int)(k / 2); d.dil_y = 1; d.dil_x = 1;

    printf("gate fp16 ic %u oc %u %ux%u k%u: %d calls, two alternating inputs\n",
           ic, oc, ih, iw, k, reps);
    rocket_rk3576_retired_counts(&u0, &w0);
    k0 = klog_retired();
    for (r = 0; r < reps; r++) {
        size_t wrong = 0, first = (size_t)-1;
        double t0, ms;
        int e;
        v = (unsigned)r & 1u;
        for (i = 0; i < n; i++) out[i] = (_Float16)12345.0f;
        sleep_ms(gap);
        t0 = now_us();
        e = rocket_conv2d_fp16_rk3576(fd, &d, in[v], W[v], out);
        ms = (now_us() - t0) / 1e3;
        if (e != ROCKET_OK) {
            printf("  call %2d: REFUSED rc %d after %.1f ms\n", r, e, ms);
            refused++;
            continue;
        }
        tsum += ms; if (ms < tmin) tmin = ms; if (ms > tmax) tmax = ms;
        for (i = 0; i < n; i++)
            if (out[i] != ref[v][i]) { if (!wrong) first = i; wrong++; }
        if (wrong) {
            size_t plane = (size_t)ih * iw;
            c = (unsigned)(first / plane);
            printf("  call %2d: WRONG %zu of %zu elements, first at channel %u row %zu "
                   "(%.1f ms)\n", r, wrong, n, c, (first % plane) / iw, ms);
            wrong_calls++;
        } else {
            printf("  call %2d: exact (%.1f ms)\n", r, ms);
        }
        fflush(stdout);
    }
    k1 = klog_retired();
    rocket_rk3576_retired_counts(&u1, &w1);
    printf("  => %d of %d calls wrong, %d refused; wall %.1f .. %.1f ms (mean %.1f); "
           "retired: %llu scored (%llu unwritten), kernel %ld\n",
           wrong_calls, reps, refused, reps - refused ? tmin : 0.0,
           reps - refused ? tmax : 0.0, reps - refused ? tsum / (reps - refused) : 0.0,
           (unsigned long long)((u1 - u0) + (w1 - w0)), (unsigned long long)(u1 - u0),
           (k0 >= 0 && k1 >= 0) ? k1 - k0 : -1L);
    fflush(stdout);
    *bad_calls += wrong_calls + refused;

done:
    for (v = 0; v < 2; v++) { free(in[v]); free(W[v]); free(ref[v]); }
    free(out);
    return rc;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw;
    int fd, a, rc = 0;

    if (argc < 2 || (strcmp(argv[1], "raw") && strcmp(argv[1], "gate"))) {
        fprintf(stderr, "usage: %s raw kind:ic:oc:ih:iw:k[:h]...\n"
                        "       %s gate [ic:oc:ih:iw:k...]\n", argv[0], argv[0]);
        return 1;
    }
    fd = rocket_open();
    if (fd < 0) { printf("no NPU; skipping\n"); return 2; }
    hw = rocket_hw_current();
    if (!hw || !hw->name || strcmp(hw->name, "rk3576")) {
        printf("not an RK3576; skipping\n");
        rocket_close(fd);
        return 2;
    }

    if (!strcmp(argv[1], "raw")) {
        if (argc < 3) { fprintf(stderr, "raw needs at least one SPEC\n"); rc = 1; }
        for (a = 2; a < argc && !rc; a++)
            if (raw_arm(fd, argv[a]) != 0) rc = 1;
    } else {
        int reps = env_int("ROCKET_DP_REPS", 6), bad = 0;
        if (argc > 2) {
            for (a = 2; a < argc && !rc; a++)
                if (gate_shape(fd, argv[a], reps, &bad) != 0) rc = 1;
        } else {
            for (a = 0; a < (int)(sizeof GATE_SHAPES / sizeof GATE_SHAPES[0]) && !rc; a++)
                if (gate_shape(fd, GATE_SHAPES[a], reps, &bad) != 0) rc = 1;
        }
        if (!rc && bad) rc = 1;
        printf("%s: %d wrong or refused call(s)\n", rc ? "FAIL" : "PASS", bad);
    }
    /* Leave the part unpoisoned for whatever runs next: the hazard outlives the process,
     * and the power domain clears it once it has been idle past its autosuspend delay. */
    sleep_ms(300);
    rocket_close(fd);
    return rc;
}
