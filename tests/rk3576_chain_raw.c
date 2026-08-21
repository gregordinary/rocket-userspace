// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_chain_raw.c — does a CHAINED stream honour read-after-write between its
 * programs?
 *
 * Chaining row tasks into one hardware kick is worth `(n-1) * 439 us` per call and is
 * already measured. Chaining ACROSS LAYERS would be worth much more on a graph, and it is
 * a different question in one respect that decides the whole refactor: a row task set
 * writes DISJOINT rows of one surface, while layer n+1 READS the surface layer n writes.
 * If the PC streams the second program before the first program's writes have landed, a
 * cross-layer chain computes on stale memory and no amount of userspace bookkeeping fixes
 * it.
 *
 * There is a prior that says it will not: `PC_DONE` means the program counter finished
 * ISSUING, not that the DPU's writes have drained — measured 82-343 us apart on this part,
 * scaling with the bytes in flight. The PC advancing to the next program in the chain is
 * the same event.
 *
 * THE MEASUREMENT IS THREE-WAY, because "the chained answer is wrong" and "the chained
 * answer is stale" are different findings and a two-way comparison cannot tell them apart:
 *
 *   fresh    P2 run alone against an X that P1 has just written        (RAW honoured)
 *   stale    P2 run alone against an X holding the sentinel           (RAW ignored)
 *   chained  P1 and P2 as ONE batched job, X sentinel-filled first
 *
 * Exactly one of the first two should equal the third. Anything else — matching neither —
 * is a third finding: the chain corrupts rather than reorders.
 *
 * THE TWO PROGRAMS ARE A CUBE CHAIN. P1's output surface IS P2's feature cube, byte for
 * byte, which is the property the cross-layer refactor would rest on: a direct conv's
 * surface stride is `ow*oh` exactly and a feature cube's channel-group stride is `ih*iw`,
 * equal here at 1x1 stride 1. So this probe asks the hardware question in the shape the
 * refactor would actually take, not in a proxy.
 *
 * Usage: rk3576_chain_raw [iterations]     (default 8)
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
#include "npu_cna.h"

#define C2        16u
#define SENTINEL  0xA5

/* One 1x1 int8 direct conv over a 32-channel square plane.
 *
 * THE PLANE IS SWEPT, and it has to be. The DPU's writes become visible 82-343 us after
 * PC_DONE on this part, SCALING WITH THE BYTES IN FLIGHT — so a small intermediate surface
 * is the case most likely to look ordered whether the hardware orders it or not. A pass at
 * 8x8 alone would say nothing about a real layer boundary; the answer has to hold at a
 * plane the size of one. */
#define G_IC   32u
#define G_OC   32u
/* 110 is as large as ONE program goes here: the feature plane costs ih*iw/2 CBUF granules
 * against a 6144 data-side cap, so 112x112 needs a row window and is refused rather than
 * split — a probe that quietly windowed would be asking about row tasks again. */
static const unsigned PLANES[] = { 8u, 32u, 56u, 96u, 110u };
#define N_PLANES ((int)(sizeof PLANES / sizeof PLANES[0]))
static unsigned G_IW = 8u, G_IH = 8u;

struct prog {
    rocket_bo     w, b, r;         /* weights, coefficients, regcmd            */
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

static int prog_init(int fd, struct prog *pr, unsigned seed,
                     uint32_t in_dma, uint32_t out_dma)
{
    unsigned icreg = rocket_rk3576_pad_ic(G_IC), ocreg = rocket_rk3576_pad_oc(G_OC);
    unsigned surf_elems = rocket_rk3576_out_surf_elems(G_IW, G_IH, 0);
    size_t w_bytes = (size_t)((ocreg + 31u) / 32u) * ((icreg + 31u) / 32u) * 32u * 32u;
    size_t coeff   = rocket_rk3576_coeff_bytes(ocreg);
    int32_t *bias;
    unsigned i;

    memset(pr, 0, sizeof *pr);
    if (rocket_bo_alloc(fd, w_bytes, &pr->w) < 0) return -1;
    if (rocket_bo_alloc(fd, coeff,   &pr->b) < 0) return -1;
    if (rocket_bo_alloc(fd, sizeof pr->ops, &pr->r) < 0) return -1;

    /* Weights that make the two programs DIFFERENT functions, so a chained P2 reading the
     * wrong buffer cannot come out right by symmetry. */
    rocket_bo_prep(fd, &pr->w, 1, 0);
    {
        int8_t *w = (int8_t *)pr->w.ptr;
        unsigned s = seed;
        for (i = 0; i < w_bytes; i++) {
            s = s * 1103515245u + 12345u;
            w[i] = (int8_t)((int)((s >> 16) % 9u) - 4);
        }
    }
    rocket_bo_fini(fd, &pr->w);

    bias = calloc(ocreg, sizeof *bias);
    if (!bias) return -1;
    for (i = 0; i < ocreg; i++) bias[i] = (int32_t)(i % 7u) - 3;
    rocket_bo_prep(fd, &pr->b, 1, 0);
    rocket_rk3576_pack_coeff_prec(pr->b.ptr, coeff, bias, ocreg, precision_int8);
    rocket_bo_fini(fd, &pr->b);
    free(bias);

    pr->p.ic = (uint16_t)icreg;  pr->p.oc = (uint16_t)ocreg;
    pr->p.ih = (uint16_t)G_IH;   pr->p.iw = (uint16_t)G_IW;
    pr->p.oh = (uint16_t)G_IH;   pr->p.ow = (uint16_t)G_IW;
    pr->p.kh = 1;                pr->p.kw = 1;
    pr->p.stride_y = 1;          pr->p.stride_x = 1;
    pr->p.pad_top = 0;           pr->p.pad_left = 0;
    pr->p.int8_out = 1;
    pr->p.in_scale = 1.0f; pr->p.w_scale = 1.0f; pr->p.out_scale = 256.0f;
    pr->p.input_zero_point = 0; pr->p.output_zero_point = 0; pr->p.weight_zero_point = 0;
    pr->p.ih_full = (uint16_t)G_IH; pr->p.oh_full = (uint16_t)G_IH;
    pr->p.input_dma   = in_dma;
    pr->p.weights_dma = (uint32_t)pr->w.dma_address;
    pr->p.bias_dma    = (uint32_t)pr->b.dma_address;
    pr->p.output_dma  = out_dma;
    pr->p.tasks = pr->ops;
    pr->p.task_count = 0;
    if (gen_conv2d_int8_rk3576(&pr->p) != 0) {
        fprintf(stderr, "the int8 conv generator refused a 1x1 32x32 8x8\n");
        return -1;
    }
    (void)surf_elems;
    return 0;
}

/* Stage one program's regcmd in its own BO and submit it as a single unchained task. */
static int prog_run_alone(int fd, struct prog *pr, const uint32_t *in_h, uint32_t n_in,
                          const uint32_t *out_h)
{
    rocket_bo_prep(fd, &pr->r, 1, 0);
    memcpy(pr->r.ptr, pr->ops, (size_t)pr->p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &pr->r);
    if (rocket_submit_matmul(fd, &pr->r, pr->p.task_count, in_h, n_in, out_h, 1, 2000) != 0)
        return -1;
    return 0;
}

/* The whole three-way comparison at one plane. Returns 0 answered, 1 could not ask. */
static int run_plane(int fd, int iters, int *verdict_fresh, int *verdict_stale)
{
    int it;
    struct prog p1, p2;
    rocket_bo A, X, Y, rc_chain;
    unsigned icreg = rocket_rk3576_pad_ic(G_IC), ocreg = rocket_rk3576_pad_oc(G_OC);
    unsigned surf_elems = rocket_rk3576_out_surf_elems(G_IW, G_IH, 0);
    size_t cube_bytes = (size_t)((icreg + C2 - 1u) / C2) * G_IH * G_IW * C2;
    size_t surf_bytes = (size_t)((ocreg + C2 - 1u) / C2) * surf_elems * C2;
    uint8_t *fresh = NULL, *stale = NULL, *chained = NULL, *xfresh = NULL;
    int n_fresh = 0, n_stale = 0, n_other = 0, could_not = 0;
    uint32_t in_h[8], out_h[2];
    unsigned i;

    printf("\n-- plane %ux%u: X is %zu bytes, P1's output surface AND P2's feature cube --\n",
           G_IW, G_IH, cube_bytes);

    /* X has to serve as both a %zu-byte surface and a %zu-byte cube. They are equal here
     * by the cube-chain identity; assert it rather than assume it, because a mismatch
     * would make every comparison below meaningless. */
    if (cube_bytes != surf_bytes) {
        printf("   the surface is %zu bytes and the cube %zu — this geometry is not a "
               "cube chain, so the probe would be asking a different question\n",
               surf_bytes, cube_bytes);
        rocket_close(fd);
        return 1;
    }

    if (rocket_bo_alloc(fd, cube_bytes, &A) < 0 ||
        rocket_bo_alloc(fd, cube_bytes, &X) < 0 ||
        rocket_bo_alloc(fd, surf_bytes, &Y) < 0 ||
        rocket_bo_alloc(fd, 4u * RK3576_CONV_TASK_OPS * sizeof(uint64_t), &rc_chain) < 0) {
        printf("allocation failed\n"); rocket_close(fd); return 1;
    }

    /* A feature cube that cannot produce a zero surface and is not uniform in any axis. */
    rocket_bo_prep(fd, &A, 1, 0);
    for (i = 0; i < cube_bytes; i++)
        ((int8_t *)A.ptr)[i] = (int8_t)((int)((i * 7u + 11u) % 41u) - 20);
    rocket_bo_fini(fd, &A);

    if (prog_init(fd, &p1, 0x2545F491u, (uint32_t)A.dma_address,
                  (uint32_t)X.dma_address) != 0 ||
        prog_init(fd, &p2, 0x9E3779B9u, (uint32_t)X.dma_address,
                  (uint32_t)Y.dma_address) != 0) {
        rocket_close(fd); return 1;
    }

    fresh   = malloc(surf_bytes);
    stale   = malloc(surf_bytes);
    chained = malloc(surf_bytes);
    xfresh  = malloc(cube_bytes);
    if (!fresh || !stale || !chained || !xfresh) { rocket_close(fd); return 1; }

    for (it = 0; it < iters; it++) {
        uint64_t t_ctl, t_chain;
        int same_fresh, same_stale, x_wrote;

        /* ---- FRESH: two separate submits, which is what the graph does today ---- */
        bo_fill(fd, &X, cube_bytes, SENTINEL);
        bo_fill(fd, &Y, surf_bytes, SENTINEL);
        in_h[0] = A.handle; in_h[1] = p1.w.handle; in_h[2] = p1.b.handle;
        in_h[3] = p1.r.handle;
        out_h[0] = X.handle;
        t_ctl = now_us();
        if (prog_run_alone(fd, &p1, in_h, 4, out_h) != 0) { could_not = 1; break; }
        if (rocket_bo_prep(fd, &X, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(xfresh, X.ptr, cube_bytes);
        rocket_bo_fini(fd, &X);

        in_h[0] = X.handle; in_h[1] = p2.w.handle; in_h[2] = p2.b.handle;
        in_h[3] = p2.r.handle;
        out_h[0] = Y.handle;
        if (prog_run_alone(fd, &p2, in_h, 4, out_h) != 0) { could_not = 1; break; }
        if (rocket_bo_prep(fd, &Y, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(fresh, Y.ptr, surf_bytes);
        rocket_bo_fini(fd, &Y);
        t_ctl = now_us() - t_ctl;

        /* ---- STALE: P2 alone against an X that nothing wrote ---- */
        bo_fill(fd, &X, cube_bytes, SENTINEL);
        bo_fill(fd, &Y, surf_bytes, SENTINEL);
        in_h[0] = X.handle; in_h[1] = p2.w.handle; in_h[2] = p2.b.handle;
        in_h[3] = p2.r.handle;
        out_h[0] = Y.handle;
        if (prog_run_alone(fd, &p2, in_h, 4, out_h) != 0) { could_not = 1; break; }
        if (rocket_bo_prep(fd, &Y, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(stale, Y.ptr, surf_bytes);
        rocket_bo_fini(fd, &Y);

        /* ---- CHAINED: both programs, one kick, X sentinel-filled first ---- */
        bo_fill(fd, &X, cube_bytes, SENTINEL);
        bo_fill(fd, &Y, surf_bytes, SENTINEL);
        {
            rocket_task_desc td[2];
            rocket_bo_prep(fd, &rc_chain, 1, 0);
            rkt_chain_pack(1, &rc_chain, td, 0, p1.ops, p1.p.task_count, 0);
            rkt_chain_pack(1, &rc_chain, td, 1, p2.ops, p2.p.task_count, 0);
            rkt_chain_seal(1, &rc_chain, 2, p1.p.task_count);
            rocket_bo_fini(fd, &rc_chain);

            /* X IS LISTED ONCE, as an output. It is P1's output and P2's input, and a
             * handle that appears in both lists is rejected with EALREADY — the driver
             * locks each BO once per job through drm_exec, and locking one twice is that
             * error. Which list it goes in only affects the implicit sync and cache
             * maintenance the kernel does around the job; the regcmd carries the IOVA, and
             * this probe brackets every access itself. The constraint is real for a
             * cross-layer chain and is worth knowing before the refactor: an intermediate
             * surface names itself once. */
            in_h[0] = A.handle;  in_h[1] = p1.w.handle; in_h[2] = p1.b.handle;
            in_h[3] = p2.w.handle; in_h[4] = p2.b.handle; in_h[5] = rc_chain.handle;
            out_h[0] = Y.handle; out_h[1] = X.handle;
            t_chain = now_us();
            if (rocket_submit_tasks_flags(fd, td, 2, in_h, 6, out_h, 2,
                                          ROCKET_JOB_BATCHED) != 0) {
                printf("   the chained submit was rejected\n");
                could_not = 1; break;
            }
            if (rocket_bo_prep(fd, &Y, 0, 2000000000ull) < 0) { could_not = 1; break; }
            memcpy(chained, Y.ptr, surf_bytes);
            rocket_bo_fini(fd, &Y);
            t_chain = now_us() - t_chain;
            if (rocket_bo_prep(fd, &X, 0, 2000000000ull) < 0) { could_not = 1; break; }
            x_wrote = memcmp(X.ptr, xfresh, cube_bytes) == 0;
            rocket_bo_fini(fd, &X);
        }

        same_fresh = memcmp(chained, fresh, surf_bytes) == 0;
        same_stale = memcmp(chained, stale, surf_bytes) == 0;
        if (same_fresh) n_fresh++;
        else if (same_stale) n_stale++;
        else n_other++;

        printf("   %2d: chained %-28s  P1's own output %s  2 submits %llu us, 1 kick "
               "%llu us\n", it,
               same_fresh ? "== FRESH (RAW honoured)"
                          : same_stale ? "== STALE (RAW ignored)" : "matches NEITHER",
               x_wrote ? "correct" : "WRONG/absent",
               (unsigned long long)t_ctl, (unsigned long long)t_chain);

        /* A P1 that did not write leaves nothing to be ordered against, so the verdict
         * above would be about the wrong thing. Said once, loudly. */
        if (!x_wrote && it == 0)
            printf("       NOTE: P1's surface does not match its solo run, so the chain's "
                   "FIRST program is the problem and the ordering question is moot until "
                   "that is understood\n");
    }

    if (!could_not && memcmp(fresh, stale, surf_bytes) == 0) {
        printf("\n   the two references are IDENTICAL, so nothing here can distinguish "
               "them — the sentinel X and the written X happen to give the same surface, "
               "and the probe proves nothing. Change the operands.\n");
        could_not = 1;
    }

    if (!could_not)
        printf("   => %d fresh, %d stale, %d matching neither, of %d\n",
               n_fresh, n_stale, n_other, iters);
    *verdict_fresh = n_fresh;
    *verdict_stale = n_stale;

    free(fresh); free(stale); free(chained); free(xfresh);
    rocket_bo_free(fd, &A); rocket_bo_free(fd, &X); rocket_bo_free(fd, &Y);
    rocket_bo_free(fd, &rc_chain);
    rocket_bo_free(fd, &p1.w); rocket_bo_free(fd, &p1.b); rocket_bo_free(fd, &p1.r);
    rocket_bo_free(fd, &p2.w); rocket_bo_free(fd, &p2.b); rocket_bo_free(fd, &p2.r);
    return could_not ? 1 : 0;
}

int main(int argc, char **argv)
{
    int fd, iters = argc > 1 ? atoi(argv[1]) : 8;
    int i, tot_fresh = 0, tot_stale = 0, planes_asked = 0, planes_skipped = 0;

    setvbuf(stdout, NULL, _IOLBF, 0);
    if (iters < 1) iters = 1;

    fd = rocket_open();
    if (fd < 0) { printf("no /dev/accel — SKIP\n"); return 2; }
    {
        const struct rocket_hw_profile *hw = rocket_hw_current();
        if (!hw || !hw->name || strcmp(hw->name, "rk3576")) {
            printf("chip is '%s', not rk3576 — SKIP\n", hw && hw->name ? hw->name : "?");
            rocket_close(fd);
            return 2;
        }
    }
    if (!rocket_batched_submit_supported()) {
        printf("this kernel does not honor DRM_ROCKET_JOB_BATCHED (needs "
               "patches/rk3576/npu/0015-0016) — the question cannot be asked here\n");
        rocket_close(fd);
        return 1;
    }

    printf("== does a chained stream honour read-after-write between its programs? ==\n");
    printf("   P1: A -> X, P2: X -> Y, two 1x1 int8 %ux%u convs in ONE batched job,\n"
           "   against the same pair run as two submits and against P2 alone on an\n"
           "   unwritten X. The plane is swept because the DPU's write visibility scales\n"
           "   with the bytes in flight.\n", G_IC, G_OC);

    for (i = 0; i < N_PLANES; i++) {
        int nf = 0, ns = 0;
        G_IW = G_IH = PLANES[i];
        if (run_plane(fd, iters, &nf, &ns) != 0) { planes_skipped++; continue; }
        planes_asked++;
        tot_fresh += nf; tot_stale += ns;
    }

    printf("\n");
    if (!planes_asked) {
        printf("== the probe could not answer the question at any plane ==\n");
        rocket_close(fd);
        return 1;
    }
    if (tot_fresh == planes_asked * iters) {
        printf("== A CHAINED STREAM HONOURS READ-AFTER-WRITE: %d of %d iterations over %d "
               "plane(s) gave the same surface as two separate submits ==\n",
               tot_fresh, planes_asked * iters, planes_asked);
        printf("   So a cross-layer chain is a correctness-preserving refactor, and what "
               "is left is the graph-level regcmd BO and the per-layer poison guard.\n");
    } else if (tot_stale == planes_asked * iters) {
        printf("== A CHAINED STREAM DOES NOT HONOUR READ-AFTER-WRITE: every one of %d "
               "iterations read the value X held BEFORE the chain ran ==\n", tot_stale);
        printf("   The PC advances when a program finishes ISSUING, not when its writes "
               "have drained. Chaining is then a lever for INDEPENDENT tasks only.\n");
    } else {
        printf("== MIXED: %d of %d iterations fresh, %d stale ==\n",
               tot_fresh, planes_asked * iters, tot_stale);
        printf("   A race rather than a rule, so a cross-layer chain is unsafe: read the "
               "per-plane lines above — if the small planes are ordered and the large ones "
               "are not, the drain is what decides and no plane size is safe by rule.\n");
    }
    if (planes_skipped)
        printf("   (%d plane(s) could not be asked and are NOT counted)\n", planes_skipped);

    rocket_close(fd);
    return 0;
}
