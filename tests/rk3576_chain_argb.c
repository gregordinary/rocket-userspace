// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_chain_argb.c — may a PACKED-IMAGE first conv head a chained stream?
 *
 * The packed-image (ARGB) sub-encoding programs 8.0x fewer MACs than the direct lowering
 * of a three-channel stem and is 7.2x on `exec + drain`, a measured 1.566 ms at 224x224
 * k7 s2. Every graph in the corpus is ONE hardware kick in ONE submit, so that saving is
 * only reachable if the packed program can sit inside the kick: a stem that cannot chain
 * splits the run and pays a 784 KiB join the per-join sweep priced at 6.80 ms, which
 * would make the cheaper program a large net LOSS.
 *
 * The reason to think it chains is that a chained stream already carries a POOL, whose
 * `PC_OPERATION_ENABLE` bitmap (0x60) is DISJOINT from a convolution's (0x1D) — and this
 * program's is a convolution's. The reason to ask anyway is that the packed sub-encoding
 * is the one place the CNA's front end is configured differently within a conv program:
 * a folded channel count that disagrees with the DMA's, a live CVT stage, a packed row
 * stride instead of a cube's, and a feature DMA walking interleaved image bytes. If the
 * PC carries any of that across a chained boundary, the program AFTER it is where it
 * shows, which is why the consumer is scored and not only the stem.
 *
 * THE MEASUREMENT IS THREE-WAY at the boundary, because "the chained answer is wrong",
 * "the chained answer is stale" and "the program never ran" are three findings a two-way
 * comparison cannot tell apart:
 *
 *   P1 packed first conv   image -> X       the stem, at ic 3
 *   P2 direct 1x1 conv     X     -> Y       an ordinary program reading the stem's cube
 *
 *   separate   two submits, which is what a graph with an unchainable stem does
 *   stale      P2 alone over a sentinel X
 *   chained    both as ONE batched job, X and Y sentinel-filled first
 *
 * X is read as well as Y, so a failure is attributed to the boundary it happened at.
 * The sentinel is what separates "never ran" from "ran on stale input": a fresh BO's
 * zeros could not.
 *
 * The cells are single-row-task geometries, so the question is the CROSS-LAYER boundary
 * alone and not the packed path's own row splitting. The packed entry submits one job per
 * row task today, which is a separate lever and a separate measurement.
 *
 * This is a PROBE and reports; the packed path's arithmetic is gated by
 * rk3576_argb_pad and rk3576_argb_extent.
 *
 * Usage: rk3576_chain_argb [iterations]     (default 8)
 * Exit:  0 the question is answered either way, 1 the probe could not run it,
 *        2 no NPU or the wrong chip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_chain.h"
#include "rocket_hw_profile.h"
#include "npu_regcmd_rk3576.h"

#define C2        16u
#define SENTINEL  0xA5
#define IC        3u

/* Plane, kernel, stride and output channels. Every cell satisfies the packed path's own
 * bounds — a non-zero left pad, ow == iw/stride, ow a multiple of 16, iw a multiple of 16
 * — and is small enough to plan into ONE row task, which is checked rather than assumed.
 * The kernel is swept because the packed weight cube's row is round16(4*kw): a stream that
 * only ever carried a k3 program would not say whether the cube's size is what matters. */
static const struct { unsigned iw, k, s, oc; } CASES[] = {
    {  32u, 3u, 2u, 32u },
    {  64u, 3u, 2u, 32u },
    {  64u, 7u, 2u, 32u },
    {  64u, 7u, 2u, 64u },
    { 128u, 3u, 2u, 32u },
};
#define N_CASES ((int)(sizeof CASES / sizeof CASES[0]))

struct prog {
    rocket_bo     w, b, r;         /* weights, coefficients, its own regcmd BO */
    conv_params_t p;
    uint64_t      ops[RK3576_CONV_TASK_OPS];
};

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

/* A bracketed fill. A bare memset of a BO the NPU is about to write leaves dirty cache
 * lines that race the DPU's DMA and land on top of the result. */
static void bo_fill(int fd, rocket_bo *bo, size_t n, int byte)
{
    rocket_bo_prep(fd, bo, 1, 0);
    memset(bo->ptr, byte, n);
    rocket_bo_fini(fd, bo);
}

static void fill_bytes(void *p, size_t n, unsigned seed, int lo, int span)
{
    int8_t *b = p;
    size_t i;
    for (i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        b[i] = (int8_t)((int)((seed >> 16) % (unsigned)span) + lo);
    }
}

static int coeff_init(int fd, struct prog *pr, unsigned ocreg, unsigned seed)
{
    size_t coeff = rocket_rk3576_coeff_bytes(ocreg);
    int32_t *bias;
    unsigned i;

    if (rocket_bo_alloc(fd, coeff, &pr->b) < 0) return -1;
    bias = calloc(ocreg, sizeof *bias);
    if (!bias) return -1;
    for (i = 0; i < ocreg; i++) bias[i] = (int32_t)((seed + i) % 7u) - 3;
    rocket_bo_prep(fd, &pr->b, 1, 0);
    rocket_rk3576_pack_coeff_prec(pr->b.ptr, coeff, bias, ocreg, precision_int8);
    rocket_bo_fini(fd, &pr->b);
    free(bias);
    return 0;
}

/* THE PROGRAM UNDER TEST: the packed-image first conv, ic 3, in its own cube-writing
 * form. Its feature buffer is an interleaved image and its weight cube is the packed
 * path's own object, which is why it cannot share conv_init below. */
static int argb_init(int fd, struct prog *pr, unsigned iw, unsigned k, unsigned s,
                     unsigned oc, uint32_t in_dma, uint32_t out_dma, unsigned *nrow_out)
{
    unsigned ow = iw / s, pad = (k - 1u) / 2u;
    size_t w_bytes = rocket_rk3576_weight_argb_int8_bytes(oc, k, k);
    int8_t *wsrc;
    rocket_rk3576_row_task rows[64];
    unsigned nrow = 0;

    memset(pr, 0, sizeof *pr);
    if (rocket_bo_alloc(fd, w_bytes, &pr->w) < 0) return -1;
    if (rocket_bo_alloc(fd, sizeof pr->ops, &pr->r) < 0) return -1;
    if (coeff_init(fd, pr, oc, 5u) != 0) return -1;

    wsrc = malloc((size_t)oc * IC * k * k);
    if (!wsrc) return -1;
    fill_bytes(wsrc, (size_t)oc * IC * k * k, 0x2545F491u, -4, 9);
    rocket_bo_prep(fd, &pr->w, 1, 0);
    if (rocket_rk3576_argb_int8_pack_weights(pr->w.ptr, w_bytes, wsrc, oc, IC, k, k) < 0) {
        rocket_bo_fini(fd, &pr->w); free(wsrc);
        printf("   the packed weight cube refused oc=%u k%u\n", oc, k);
        return -1;
    }
    rocket_bo_fini(fd, &pr->w);
    free(wsrc);

    pr->p.ic = (uint16_t)IC;      pr->p.oc = (uint16_t)oc;
    pr->p.ih = (uint16_t)iw;      pr->p.iw = (uint16_t)iw;
    pr->p.oh = (uint16_t)ow;      pr->p.ow = (uint16_t)ow;
    pr->p.kh = (uint16_t)k;       pr->p.kw = (uint16_t)k;
    pr->p.stride_y = (uint8_t)s;  pr->p.stride_x = (uint8_t)s;
    pr->p.pad_top = (uint8_t)pad; pr->p.pad_left = (uint8_t)pad;
    pr->p.int8_out = 1;
    pr->p.in_scale = 1.0f; pr->p.w_scale = 1.0f; pr->p.out_scale = 512.0f;
    /* uint8-centered, as the library entry programs them. */
    pr->p.input_zero_point = 0x80; pr->p.output_zero_point = 0x80;
    pr->p.weight_zero_point = 0x80;
    pr->p.ih_full = (uint16_t)iw; pr->p.oh_full = (uint16_t)ow;
    pr->p.input_dma   = in_dma;
    pr->p.weights_dma = (uint32_t)pr->w.dma_address;
    pr->p.bias_dma    = (uint32_t)pr->b.dma_address;
    pr->p.output_dma  = out_dma;
    pr->p.tasks = pr->ops;
    pr->p.task_count = 0;

    /* ONE row task or the cell is skipped: this probe is about the CROSS-LAYER boundary,
     * and a multi-task stem would put the packed path's own row split in the same
     * measurement. */
    if (rocket_rk3576_plan_rows_prec(&pr->p, 0, precision_int8, rows,
                                     (unsigned)(sizeof rows / sizeof rows[0]), &nrow) < 0)
        return -1;
    *nrow_out = nrow;
    if (nrow != 1) return 1;

    if (gen_conv2d_int8_rk3576(&pr->p) != 0) {
        printf("   the generator refused the packed first conv (%ux%u k%u s%u oc=%u)\n",
               iw, iw, k, s, oc);
        return -1;
    }
    return 0;
}

/* The consumer: one 1x1 int8 direct convolution over the stem's output cube. */
static int conv_init(int fd, struct prog *pr, unsigned plane, unsigned c,
                     uint32_t in_dma, uint32_t out_dma)
{
    unsigned icreg = rocket_rk3576_pad_ic(c), ocreg = rocket_rk3576_pad_oc(c);
    size_t w_bytes = (size_t)((ocreg + 31u) / 32u) * ((icreg + 31u) / 32u) * 32u * 32u;

    memset(pr, 0, sizeof *pr);
    if (rocket_bo_alloc(fd, w_bytes, &pr->w) < 0) return -1;
    if (rocket_bo_alloc(fd, sizeof pr->ops, &pr->r) < 0) return -1;
    if (coeff_init(fd, pr, ocreg, 11u) != 0) return -1;

    rocket_bo_prep(fd, &pr->w, 1, 0);
    fill_bytes(pr->w.ptr, w_bytes, 0x9E3779B9u, -4, 9);
    rocket_bo_fini(fd, &pr->w);

    pr->p.ic = (uint16_t)icreg;   pr->p.oc = (uint16_t)ocreg;
    pr->p.ih = (uint16_t)plane;   pr->p.iw = (uint16_t)plane;
    pr->p.oh = (uint16_t)plane;   pr->p.ow = (uint16_t)plane;
    pr->p.kh = 1;                 pr->p.kw = 1;
    pr->p.stride_y = 1;           pr->p.stride_x = 1;
    pr->p.pad_top = 0;            pr->p.pad_left = 0;
    pr->p.int8_out = 1;
    pr->p.in_scale = 1.0f; pr->p.w_scale = 1.0f; pr->p.out_scale = 256.0f;
    pr->p.input_zero_point = 0; pr->p.output_zero_point = 0; pr->p.weight_zero_point = 0;
    pr->p.ih_full = (uint16_t)plane; pr->p.oh_full = (uint16_t)plane;
    pr->p.input_dma   = in_dma;
    pr->p.weights_dma = (uint32_t)pr->w.dma_address;
    pr->p.bias_dma    = (uint32_t)pr->b.dma_address;
    pr->p.output_dma  = out_dma;
    pr->p.tasks = pr->ops;
    pr->p.task_count = 0;
    if (gen_conv2d_int8_rk3576(&pr->p) != 0) {
        printf("   the int8 conv generator refused 1x1 %ux%u at %u channels\n",
               plane, plane, c);
        return -1;
    }
    return 0;
}

static void prog_free(int fd, struct prog *pr)
{
    if (pr->w.ptr) rocket_bo_free(fd, &pr->w);
    if (pr->b.ptr) rocket_bo_free(fd, &pr->b);
    if (pr->r.ptr) rocket_bo_free(fd, &pr->r);
    memset(pr, 0, sizeof *pr);
}

static int run_alone(int fd, rocket_bo *rc, const uint64_t *ops, uint32_t count,
                     const uint32_t *in_h, uint32_t n_in, uint32_t out_h)
{
    rocket_bo_prep(fd, rc, 1, 0);
    memcpy(rc->ptr, ops, (size_t)count * sizeof(uint64_t));
    rocket_bo_fini(fd, rc);
    return rocket_submit_matmul(fd, rc, count, in_h, n_in, &out_h, 1, 2000) == 0 ? 0 : -1;
}

static int all_sentinel(const void *p, size_t n)
{
    const unsigned char *b = p;
    size_t i;
    for (i = 0; i < n; i++) if (b[i] != SENTINEL) return 0;
    return 1;
}

static const char *verdict(const void *got, const void *ref, const void *stale, size_t n)
{
    if (all_sentinel(got, n))   return "NEVER RAN (still the sentinel)";
    if (!memcmp(got, ref, n))   return "correct";
    if (stale && !memcmp(got, stale, n)) return "STALE (read the pre-chain buffer)";
    return "matches NEITHER";
}

/* Returns 0 answered, 1 could not ask, 2 the cell does not plan into one task. */
static int run_case(int fd, unsigned iw, unsigned k, unsigned s, unsigned oc, int iters,
                    int *ok_stem, int *ok_next)
{
    struct prog p1, p2;
    rocket_bo IMG, X, Y, rcc;
    unsigned ow = iw / s;
    unsigned ocreg = rocket_rk3576_pad_oc(oc);
    unsigned go = (ocreg + C2 - 1u) / C2;
    unsigned x_elems = rocket_rk3576_out_surf_elems(ow, ow, 0);   /* == ow*ow */
    size_t img_bytes = (size_t)iw * iw * IC;
    size_t x_bytes = (size_t)go * x_elems * C2;
    size_t y_bytes = x_bytes;
    uint8_t *xref = NULL, *yref = NULL, *ystale = NULL, *xc = NULL, *yc = NULL;
    uint32_t in_h[8];
    int it, could_not = 0, n_stem = 0, n_next = 0, rc1;
    unsigned nrow = 0;

    memset(&p1, 0, sizeof p1); memset(&p2, 0, sizeof p2);
    memset(&IMG, 0, sizeof IMG); memset(&X, 0, sizeof X);
    memset(&Y, 0, sizeof Y); memset(&rcc, 0, sizeof rcc);

    if (rocket_bo_alloc(fd, img_bytes, &IMG) < 0 || rocket_bo_alloc(fd, x_bytes, &X) < 0 ||
        rocket_bo_alloc(fd, y_bytes, &Y) < 0 ||
        rocket_bo_alloc(fd, 4u * RK3576_CONV_TASK_OPS * sizeof(uint64_t), &rcc) < 0) {
        printf("   allocation failed\n");
        could_not = 1; goto out;
    }

    /* A packed image the stem cannot turn into a uniform surface, so a stale read
     * downstream is distinguishable from a correct one. */
    rocket_bo_prep(fd, &IMG, 1, 0);
    fill_bytes(IMG.ptr, img_bytes, 0x1234567u, -20, 41);
    rocket_bo_fini(fd, &IMG);

    rc1 = argb_init(fd, &p1, iw, k, s, oc, (uint32_t)IMG.dma_address,
                    (uint32_t)X.dma_address, &nrow);
    if (rc1 == 1) {
        printf("-- %ux%u k%u s%u oc=%u: SKIPPED, it plans into %u row tasks and this probe "
               "wants one --\n", iw, iw, k, s, oc, nrow);
        could_not = 2; goto out;
    }
    if (rc1 != 0) { could_not = 1; goto out; }
    if (conv_init(fd, &p2, ow, ocreg, (uint32_t)X.dma_address,
                  (uint32_t)Y.dma_address) != 0) { could_not = 1; goto out; }

    printf("\n-- %ux%u k%u s%u oc=%u: packed first conv -> 1x1 conv, X %zu B, Y %zu B, "
           "programs %u + %u words --\n", iw, iw, k, s, oc, x_bytes, y_bytes,
           p1.p.task_count, p2.p.task_count);

    xref = malloc(x_bytes); yref = malloc(y_bytes); ystale = malloc(y_bytes);
    xc = malloc(x_bytes); yc = malloc(y_bytes);
    if (!xref || !yref || !ystale || !xc || !yc) { could_not = 1; goto out; }

    for (it = 0; it < iters; it++) {
        uint64_t t_sep, t_chain;

        /* ---- SEPARATE: two submits, the path a graph with an unchainable stem runs ---- */
        bo_fill(fd, &X, x_bytes, SENTINEL);
        bo_fill(fd, &Y, y_bytes, SENTINEL);
        t_sep = now_us();
        in_h[0] = IMG.handle; in_h[1] = p1.w.handle; in_h[2] = p1.b.handle;
        in_h[3] = p1.r.handle;
        if (run_alone(fd, &p1.r, p1.ops, p1.p.task_count, in_h, 4, X.handle) != 0) {
            could_not = 1; break;
        }
        if (rocket_bo_prep(fd, &X, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(xref, X.ptr, x_bytes);
        rocket_bo_fini(fd, &X);

        in_h[0] = X.handle; in_h[1] = p2.w.handle; in_h[2] = p2.b.handle;
        in_h[3] = p2.r.handle;
        if (run_alone(fd, &p2.r, p2.ops, p2.p.task_count, in_h, 4, Y.handle) != 0) {
            could_not = 1; break;
        }
        if (rocket_bo_prep(fd, &Y, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(yref, Y.ptr, y_bytes);
        rocket_bo_fini(fd, &Y);
        t_sep = now_us() - t_sep;

        /* THE ARM THAT MUST SUCCEED. If the unchained path leaves either buffer untouched
         * the probe is measuring its own geometry rather than the chain. */
        if (it == 0 && (all_sentinel(xref, x_bytes) || all_sentinel(yref, y_bytes))) {
            printf("   the SEPARATE arm left %s untouched — this geometry does not run "
                   "even unchained, so nothing below would be about chaining\n",
                   all_sentinel(xref, x_bytes) ? "the stem's output" : "the consumer's");
            could_not = 1; break;
        }

        /* ---- STALE: the consumer alone over an input nothing wrote ---- */
        bo_fill(fd, &X, x_bytes, SENTINEL);
        bo_fill(fd, &Y, y_bytes, SENTINEL);
        in_h[0] = X.handle; in_h[1] = p2.w.handle; in_h[2] = p2.b.handle;
        in_h[3] = p2.r.handle;
        if (run_alone(fd, &p2.r, p2.ops, p2.p.task_count, in_h, 4, Y.handle) != 0) {
            could_not = 1; break;
        }
        if (rocket_bo_prep(fd, &Y, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(ystale, Y.ptr, y_bytes);
        rocket_bo_fini(fd, &Y);

        /* ---- CHAINED: both programs, ONE kick ---- */
        bo_fill(fd, &X, x_bytes, SENTINEL);
        bo_fill(fd, &Y, y_bytes, SENTINEL);
        {
            rocket_task_desc td[2];
            uint32_t out_h[2];
            size_t o0 = 0;
            size_t o1 = o0 + rkt_chain_words(p1.p.task_count);
            int bad = 0;

            /* MIXED LENGTHS, so each trailer carries its SUCCESSOR's count: the packed
             * program and the 1x1 one are not the same size, and a stream that told the PC
             * one length for the other would fetch a short segment as a long one, with
             * nothing to fault on. */
            rocket_bo_prep(fd, &rcc, 1, 0);
            bad |= rkt_chain_pack_at(&rcc, td, 0, o0, o1, p1.ops, p1.p.task_count,
                                     p2.p.task_count);
            bad |= rkt_chain_pack_at(&rcc, td, 1, o1, o1, p2.ops, p2.p.task_count,
                                     p2.p.task_count);
            rocket_bo_fini(fd, &rcc);
            if (bad) {
                printf("   a program's trailer is not the shape the chain layout claims — "
                       "the stream would be malformed, so it is NOT submitted\n");
                could_not = 1; break;
            }

            /* Each BO ONCE across both lists: X is the stem's output and the consumer's
             * input, and a handle in both is rejected with EALREADY. */
            in_h[0] = IMG.handle;  in_h[1] = p1.w.handle; in_h[2] = p1.b.handle;
            in_h[3] = p2.w.handle; in_h[4] = p2.b.handle; in_h[5] = rcc.handle;
            out_h[0] = X.handle; out_h[1] = Y.handle;
            t_chain = now_us();
            if (rocket_submit_tasks_flags(fd, td, 2, in_h, 6, out_h, 2,
                                          ROCKET_JOB_BATCHED) != 0) {
                printf("   the chained submit was REJECTED\n");
                could_not = 1; break;
            }
            if (rocket_bo_prep(fd, &Y, 0, 2000000000ull) < 0) { could_not = 1; break; }
            memcpy(yc, Y.ptr, y_bytes);
            rocket_bo_fini(fd, &Y);
            t_chain = now_us() - t_chain;
            if (rocket_bo_prep(fd, &X, 0, 2000000000ull) < 0) { could_not = 1; break; }
            memcpy(xc, X.ptr, x_bytes);
            rocket_bo_fini(fd, &X);
        }

        {
            const char *vx = verdict(xc, xref, NULL, x_bytes);
            const char *vy = verdict(yc, yref, ystale, y_bytes);

            if (!strcmp(vx, "correct")) n_stem++;
            if (!strcmp(vy, "correct")) n_next++;
            printf("   %2d: packed stem %-30s consumer %-34s  2 submits %llu us, "
                   "1 kick %llu us\n", it, vx, vy,
                   (unsigned long long)t_sep, (unsigned long long)t_chain);
        }

        if (it == 0 && !memcmp(yref, ystale, y_bytes))
            printf("       NOTE: the consumer's written and unwritten inputs give the SAME "
                   "output here, so its column cannot distinguish them\n");
    }

    *ok_stem += n_stem;
    *ok_next += n_next;
out:
    free(xref); free(yref); free(ystale); free(xc); free(yc);
    prog_free(fd, &p1); prog_free(fd, &p2);
    if (IMG.ptr) rocket_bo_free(fd, &IMG);
    if (X.ptr)   rocket_bo_free(fd, &X);
    if (Y.ptr)   rocket_bo_free(fd, &Y);
    if (rcc.ptr) rocket_bo_free(fd, &rcc);
    return could_not;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int iters = argc > 1 ? atoi(argv[1]) : 8;
    int fd, i, ok_stem = 0, ok_next = 0, ran = 0, skipped = 0, could_not = 0;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_chain_argb: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    if (!rkt_chain_enabled())
        printf("NOTE: ROCKET_BATCH_SUBMIT is off in this process, but this probe lays its "
               "own stream out explicitly, so the chained arm still runs.\n");
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_chain_argb: no NPU device — skipping\n"); return 2; }

    printf("== RK3576: may a packed-image first conv head a chained stream? "
           "%d iterations per cell ==\n", iters);

    for (i = 0; i < N_CASES; i++) {
        int rc = run_case(fd, CASES[i].iw, CASES[i].k, CASES[i].s, CASES[i].oc, iters,
                          &ok_stem, &ok_next);
        if (rc == 2) { skipped++; continue; }
        if (rc) { could_not++; continue; }
        ran += iters;
    }

    printf("\n== %d chained iterations: stem correct %d, consumer correct %d; "
           "%d cells skipped, %d could not run ==\n",
           ran, ok_stem, ok_next, skipped, could_not);
    if (ran && ok_stem == ran && ok_next == ran)
        printf("== A PACKED-IMAGE FIRST CONV RUNS INSIDE A CHAINED STREAM, and the "
               "program after it reads what it wrote ==\n");
    else if (ran)
        printf("== it does NOT chain cleanly — read the per-iteration verdicts for the "
               "boundary it failed at ==\n");
    rocket_close(fd);
    return could_not && !ran ? 1 : 0;
}
