// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_chain_pool.c — may a POOL program sit inside a chained convolution stream?
 *
 * A cross-layer kick is a contiguous regcmd stream of convolution programs, and every
 * classifier puts a pooling layer in the middle of one. Today a pool breaks the run: the
 * chain holds convolution handles, so a caller marks the pool as unchainable and the graph
 * pays a submit for the pool and another for whatever follows it. On ResNet-18 that is what
 * isolates the stem and the classifier — four submits of eleven, before the run finder was
 * widened, and two of five after.
 *
 * The reason to doubt it is `PC_OPERATION_ENABLE`, which is a per-block BITMAP rather than a
 * constant: a convolution enables CNA/CORE/DPU with 0x1D and a pool enables PPU/PPU_RDMA
 * with 0x60, and the two sets are disjoint. If the PC carries any block state across a
 * chained boundary — a configuration it expects to still hold, a completion it expects from
 * a block the next program never starts — a mixed stream is where that shows. Against it:
 * each task carries its OWN program including its own trailer, and the trailer is what says
 * which blocks to start, so there may be no obstacle at all.
 *
 * THE MEASUREMENT IS THREE-WAY at each of the two boundaries, because "the chained answer is
 * wrong", "the chained answer is stale" and "the program never ran" are three different
 * findings and a two-way comparison cannot tell them apart:
 *
 *   P1 conv   A -> X          the producer the pool reads
 *   P2 pool   X -> Y          the program under test
 *   P3 conv   Y -> Z          a convolution AFTER a pool in the same stream
 *
 *   separate   all three as three submits, which is what a graph does today
 *   stale      P2 alone on a sentinel X, and P3 alone on a sentinel Y
 *   chained    all three as ONE batched job, X/Y/Z sentinel-filled first
 *
 * Both intermediates are read, so a failure is attributed to the boundary it happened at
 * rather than to the stream as a whole. The sentinel is what separates "never ran" from
 * "ran on stale input": a fresh BO's zeros could not.
 *
 * This is a PROBE and reports; it asserts nothing about the pool's arithmetic, which
 * rk3576_pool_probe already gates.
 *
 * Usage: rk3576_chain_pool [iterations]     (default 8)
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

/* The plane is swept for the same reason rk3576_chain_raw sweeps it: the DPU's writes
 * become visible 82-343 us after PC_DONE on this part, scaling with the bytes in flight, so
 * a small intermediate is the case most likely to look ordered whether the hardware orders
 * it or not. The channel count is swept too, because the PPU's own program is a function of
 * the channel groups and a single width would not say whether a wider one still fits in a
 * stream. 110 is as large as ONE convolution program goes here (the feature plane costs
 * ih*iw/2 CBUF granules against a 6144 data-side cap), and the pool halves it. */
static const struct { unsigned w, c; } CASES[] = {
    {  16u, 32u },
    {  32u, 32u },
    {  56u, 32u },
    {  56u, 64u },
    { 110u, 32u },
};
#define N_CASES ((int)(sizeof CASES / sizeof CASES[0]))

static unsigned round4(unsigned v) { return (v + 3u) & ~3u; }

struct conv_prog {
    rocket_bo     w, b, r;         /* weights, coefficients, its own regcmd BO */
    conv_params_t p;
    uint64_t      ops[RK3576_CONV_TASK_OPS];
};

struct pool_prog {
    rocket_bo            r;
    pool_params_rk3576_t p;
    uint64_t             ops[RK3576_POOL_TASK_OPS];
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

/* One 1x1 int8 direct convolution over `plane` x `plane` at `c` channels. `in_surf` is the
 * feature cube's channel-group stride, 0 for the plane itself — the program after a pool
 * reads a surface the PPU wrote at round4(ow*oh), which is not the plane. */
static int conv_init(int fd, struct conv_prog *pr, unsigned seed, unsigned plane,
                     unsigned c, unsigned in_surf, uint32_t in_dma, uint32_t out_dma)
{
    unsigned icreg = rocket_rk3576_pad_ic(c), ocreg = rocket_rk3576_pad_oc(c);
    size_t w_bytes = (size_t)((ocreg + 31u) / 32u) * ((icreg + 31u) / 32u) * 32u * 32u;
    size_t coeff   = rocket_rk3576_coeff_bytes(ocreg);
    int32_t *bias;
    unsigned i;

    memset(pr, 0, sizeof *pr);
    if (rocket_bo_alloc(fd, w_bytes, &pr->w) < 0) return -1;
    if (rocket_bo_alloc(fd, coeff,   &pr->b) < 0) return -1;
    if (rocket_bo_alloc(fd, sizeof pr->ops, &pr->r) < 0) return -1;

    /* Weights that make the two convolutions DIFFERENT functions, so a chained P3 reading
     * the wrong buffer cannot come out right by symmetry. */
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
    pr->p.in_surf_elems = in_surf;
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

/* A max pool, k2 s2, no pad, reading a convolution's OUTPUT SURFACE — whose channel-group
 * stride is `iw*ih` exactly, not the round4 a vendor pooling program carries. */
static int pool_init(int fd, struct pool_prog *pr, unsigned plane, unsigned c,
                     uint32_t in_dma, uint32_t out_dma)
{
    memset(pr, 0, sizeof *pr);
    if (rocket_bo_alloc(fd, sizeof pr->ops, &pr->r) < 0) return -1;

    pr->p.iw = (uint16_t)plane;      pr->p.ih = (uint16_t)plane;
    pr->p.c  = (uint16_t)c;
    pr->p.ow = (uint16_t)(plane / 2u); pr->p.oh = (uint16_t)(plane / 2u);
    pr->p.kw = 2; pr->p.kh = 2;
    pr->p.stride_x = 2; pr->p.stride_y = 2;
    pr->p.mode = ROCKET_RK3576_POOL_MAX;
    pr->p.src_surf_elems = plane * plane;
    pr->p.input_dma  = in_dma;
    pr->p.output_dma = out_dma;
    pr->p.tasks = pr->ops;
    pr->p.task_count = 0;
    if (gen_pool_rk3576(&pr->p) != 0) {
        printf("   the pool generator refused max k2 s2 over %ux%u at %u channels\n",
               plane, plane, c);
        return -1;
    }
    return 0;
}

/* Stage one program in its own BO and submit it as a single unchained task. */
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

/* Name the verdict at one boundary. */
static const char *verdict(const void *got, const void *ref, const void *stale, size_t n)
{
    if (all_sentinel(got, n))            return "NEVER RAN (still the sentinel)";
    if (!memcmp(got, ref, n))            return "correct";
    if (!memcmp(got, stale, n))          return "STALE (read the pre-chain buffer)";
    return "matches NEITHER";
}

/* Returns 0 answered, 1 could not ask. */
static int run_case(int fd, unsigned plane, unsigned c, int iters,
                    int *ok_pool, int *ok_conv)
{
    struct conv_prog p1, p3;
    struct pool_prog p2;
    rocket_bo A, X, Y, Z, rcc;
    unsigned icreg = rocket_rk3576_pad_ic(c), ocreg = rocket_rk3576_pad_oc(c);
    unsigned gi = (icreg + C2 - 1u) / C2, go = (ocreg + C2 - 1u) / C2;
    unsigned op = plane / 2u;
    unsigned x_elems = rocket_rk3576_out_surf_elems(plane, plane, 0);   /* == plane*plane */
    unsigned y_elems = round4(op * op);          /* what the PPU writes per channel group */
    unsigned z_elems = rocket_rk3576_out_surf_elems(op, op, 0);
    size_t a_bytes = (size_t)gi * plane * plane * C2;
    size_t x_bytes = (size_t)go * x_elems * C2;
    size_t y_bytes = (size_t)go * y_elems * C2;
    size_t z_bytes = (size_t)go * z_elems * C2;
    uint8_t *xref = NULL, *yref = NULL, *zref = NULL, *ystale = NULL, *zstale = NULL;
    uint8_t *xc = NULL, *yc = NULL, *zc = NULL;
    uint32_t in_h[8];
    int it, could_not = 0, n_pool_ok = 0, n_conv_ok = 0;
    unsigned i;

    printf("\n-- %ux%u at %u channels: conv -> pool(k2 s2) -> conv, X %zu B, Y %zu B, "
           "Z %zu B --\n", plane, plane, c, x_bytes, y_bytes, z_bytes);

    if (rocket_bo_alloc(fd, a_bytes, &A) < 0 || rocket_bo_alloc(fd, x_bytes, &X) < 0 ||
        rocket_bo_alloc(fd, y_bytes, &Y) < 0 || rocket_bo_alloc(fd, z_bytes, &Z) < 0 ||
        rocket_bo_alloc(fd, 8u * RK3576_CONV_TASK_OPS * sizeof(uint64_t), &rcc) < 0) {
        printf("   allocation failed\n");
        return 1;
    }

    /* A feature cube that cannot produce a uniform surface, so a max pool over it is not
     * the same value everywhere and a stale read is distinguishable. */
    rocket_bo_prep(fd, &A, 1, 0);
    for (i = 0; i < a_bytes; i++)
        ((int8_t *)A.ptr)[i] = (int8_t)((int)((i * 7u + 11u) % 41u) - 20);
    rocket_bo_fini(fd, &A);

    if (conv_init(fd, &p1, 0x2545F491u, plane, c, 0,
                  (uint32_t)A.dma_address, (uint32_t)X.dma_address) != 0 ||
        pool_init(fd, &p2, plane, c,
                  (uint32_t)X.dma_address, (uint32_t)Y.dma_address) != 0 ||
        conv_init(fd, &p3, 0x9E3779B9u, op, c, y_elems,
                  (uint32_t)Y.dma_address, (uint32_t)Z.dma_address) != 0)
        return 1;

    xref = malloc(x_bytes); yref = malloc(y_bytes); zref = malloc(z_bytes);
    ystale = malloc(y_bytes); zstale = malloc(z_bytes);
    xc = malloc(x_bytes); yc = malloc(y_bytes); zc = malloc(z_bytes);
    if (!xref || !yref || !zref || !ystale || !zstale || !xc || !yc || !zc) return 1;

    for (it = 0; it < iters; it++) {
        uint64_t t_sep, t_chain;

        /* ---- SEPARATE: three submits, which is what a graph does today ---- */
        bo_fill(fd, &X, x_bytes, SENTINEL);
        bo_fill(fd, &Y, y_bytes, SENTINEL);
        bo_fill(fd, &Z, z_bytes, SENTINEL);
        t_sep = now_us();
        in_h[0] = A.handle; in_h[1] = p1.w.handle; in_h[2] = p1.b.handle;
        in_h[3] = p1.r.handle;
        if (run_alone(fd, &p1.r, p1.ops, p1.p.task_count, in_h, 4, X.handle) != 0) {
            could_not = 1; break;
        }
        if (rocket_bo_prep(fd, &X, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(xref, X.ptr, x_bytes);
        rocket_bo_fini(fd, &X);

        in_h[0] = X.handle; in_h[1] = p2.r.handle;
        if (run_alone(fd, &p2.r, p2.ops, p2.p.task_count, in_h, 2, Y.handle) != 0) {
            could_not = 1; break;
        }
        if (rocket_bo_prep(fd, &Y, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(yref, Y.ptr, y_bytes);
        rocket_bo_fini(fd, &Y);

        in_h[0] = Y.handle; in_h[1] = p3.w.handle; in_h[2] = p3.b.handle;
        in_h[3] = p3.r.handle;
        if (run_alone(fd, &p3.r, p3.ops, p3.p.task_count, in_h, 4, Z.handle) != 0) {
            could_not = 1; break;
        }
        if (rocket_bo_prep(fd, &Z, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(zref, Z.ptr, z_bytes);
        rocket_bo_fini(fd, &Z);
        t_sep = now_us() - t_sep;

        /* THE ARM THAT MUST SUCCEED. Three separate submits are the path the graph runs
         * today, so if this one leaves an untouched buffer the probe is measuring its own
         * geometry rather than the chain. */
        if (it == 0 && (all_sentinel(yref, y_bytes) || all_sentinel(zref, z_bytes))) {
            printf("   the SEPARATE arm left %s untouched — this geometry does not run "
                   "even unchained, so nothing below would be about chaining\n",
                   all_sentinel(yref, y_bytes) ? "the pool's output" : "the last conv's");
            could_not = 1; break;
        }

        /* ---- STALE: each consumer alone against an input nothing wrote ---- */
        bo_fill(fd, &X, x_bytes, SENTINEL);
        bo_fill(fd, &Y, y_bytes, SENTINEL);
        in_h[0] = X.handle; in_h[1] = p2.r.handle;
        if (run_alone(fd, &p2.r, p2.ops, p2.p.task_count, in_h, 2, Y.handle) != 0) {
            could_not = 1; break;
        }
        if (rocket_bo_prep(fd, &Y, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(ystale, Y.ptr, y_bytes);
        rocket_bo_fini(fd, &Y);

        bo_fill(fd, &Y, y_bytes, SENTINEL);
        bo_fill(fd, &Z, z_bytes, SENTINEL);
        in_h[0] = Y.handle; in_h[1] = p3.w.handle; in_h[2] = p3.b.handle;
        in_h[3] = p3.r.handle;
        if (run_alone(fd, &p3.r, p3.ops, p3.p.task_count, in_h, 4, Z.handle) != 0) {
            could_not = 1; break;
        }
        if (rocket_bo_prep(fd, &Z, 0, 2000000000ull) < 0) { could_not = 1; break; }
        memcpy(zstale, Z.ptr, z_bytes);
        rocket_bo_fini(fd, &Z);

        /* ---- CHAINED: all three programs, ONE kick ---- */
        bo_fill(fd, &X, x_bytes, SENTINEL);
        bo_fill(fd, &Y, y_bytes, SENTINEL);
        bo_fill(fd, &Z, z_bytes, SENTINEL);
        {
            rocket_task_desc td[3];
            uint32_t out_h[3];
            size_t o0 = 0;
            size_t o1 = o0 + rkt_chain_words(p1.p.task_count);
            size_t o2 = o1 + rkt_chain_words(p2.p.task_count);
            int bad = 0;

            /* MIXED LENGTHS, so the offsets are explicit and each trailer carries its
             * SUCCESSOR's count: a conv program and a pool program are not the same size,
             * and a stream that told the PC a pool's length for a conv would fetch a short
             * segment as a long one, with nothing to fault on. */
            rocket_bo_prep(fd, &rcc, 1, 0);
            bad |= rkt_chain_pack_at(&rcc, td, 0, o0, o1, p1.ops, p1.p.task_count,
                                     p2.p.task_count);
            bad |= rkt_chain_pack_at(&rcc, td, 1, o1, o2, p2.ops, p2.p.task_count,
                                     p3.p.task_count);
            bad |= rkt_chain_pack_at(&rcc, td, 2, o2, o2, p3.ops, p3.p.task_count,
                                     p3.p.task_count);
            rocket_bo_fini(fd, &rcc);
            if (bad) {
                printf("   a program's trailer is not the shape the chain layout claims — "
                       "the stream would be malformed, so it is NOT submitted\n");
                could_not = 1; break;
            }

            /* Each BO ONCE across both lists: X and Y are each a producer's output and a
             * consumer's input, and a handle in both is rejected with EALREADY. */
            in_h[0] = A.handle;    in_h[1] = p1.w.handle; in_h[2] = p1.b.handle;
            in_h[3] = p3.w.handle; in_h[4] = p3.b.handle; in_h[5] = rcc.handle;
            out_h[0] = X.handle; out_h[1] = Y.handle; out_h[2] = Z.handle;
            t_chain = now_us();
            if (rocket_submit_tasks_flags(fd, td, 3, in_h, 6, out_h, 3,
                                          ROCKET_JOB_BATCHED) != 0) {
                printf("   the chained submit was REJECTED\n");
                could_not = 1; break;
            }
            if (rocket_bo_prep(fd, &Z, 0, 2000000000ull) < 0) { could_not = 1; break; }
            memcpy(zc, Z.ptr, z_bytes);
            rocket_bo_fini(fd, &Z);
            t_chain = now_us() - t_chain;
            if (rocket_bo_prep(fd, &X, 0, 2000000000ull) < 0) { could_not = 1; break; }
            memcpy(xc, X.ptr, x_bytes);
            rocket_bo_fini(fd, &X);
            if (rocket_bo_prep(fd, &Y, 0, 2000000000ull) < 0) { could_not = 1; break; }
            memcpy(yc, Y.ptr, y_bytes);
            rocket_bo_fini(fd, &Y);
        }

        {
            /* X has no stale form — nothing before it in the stream — so it is scored
             * against its own solo run alone. */
            const char *vx = all_sentinel(xc, x_bytes) ? "NEVER RAN (still the sentinel)"
                           : (!memcmp(xc, xref, x_bytes) ? "correct" : "WRONG");
            const char *vy = verdict(yc, yref, ystale, y_bytes);
            const char *vz = verdict(zc, zref, zstale, z_bytes);

            if (!strcmp(vy, "correct")) n_pool_ok++;
            if (!strcmp(vz, "correct")) n_conv_ok++;
            printf("   %2d: conv %-30s pool %-34s conv-after-pool %-34s  "
                   "3 submits %llu us, 1 kick %llu us\n",
                   it, vx, vy, vz,
                   (unsigned long long)t_sep, (unsigned long long)t_chain);
        }

        /* Both discriminators, said once. A reference equal to its stale twin scores
         * "correct" for the wrong reason. */
        if (it == 0) {
            if (!memcmp(yref, ystale, y_bytes))
                printf("       NOTE: the pool's written and unwritten inputs give the SAME "
                       "output here, so its column cannot distinguish them\n");
            if (!memcmp(zref, zstale, z_bytes))
                printf("       NOTE: the last conv's written and unwritten inputs give the "
                       "SAME output here, so its column cannot distinguish them\n");
        }
    }

    if (!could_not)
        printf("   => pool correct in %d of %d, conv-after-pool correct in %d of %d\n",
               n_pool_ok, iters, n_conv_ok, iters);
    *ok_pool = n_pool_ok;
    *ok_conv = n_conv_ok;

    free(xref); free(yref); free(zref); free(ystale); free(zstale);
    free(xc); free(yc); free(zc);
    rocket_bo_free(fd, &A); rocket_bo_free(fd, &X); rocket_bo_free(fd, &Y);
    rocket_bo_free(fd, &Z); rocket_bo_free(fd, &rcc);
    rocket_bo_free(fd, &p1.w); rocket_bo_free(fd, &p1.b); rocket_bo_free(fd, &p1.r);
    rocket_bo_free(fd, &p2.r);
    rocket_bo_free(fd, &p3.w); rocket_bo_free(fd, &p3.b); rocket_bo_free(fd, &p3.r);
    return could_not ? 1 : 0;
}

int main(int argc, char **argv)
{
    int fd, iters = argc > 1 ? atoi(argv[1]) : 8;
    int i, asked = 0, skipped = 0, tot_pool = 0, tot_conv = 0;

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

    printf("== may a POOL program sit inside a chained convolution stream? ==\n");
    printf("   conv -> pool -> conv as ONE batched job, against the same three as three\n"
           "   submits and against each consumer alone on an unwritten input. Both\n"
           "   intermediates are read, so a failure names the boundary it happened at.\n"
           "   PC_OPERATION_ENABLE is a per-block bitmap and a pool's 0x60 is disjoint\n"
           "   from a convolution's 0x1D, which is the reason to doubt it.\n");

    for (i = 0; i < N_CASES; i++) {
        int np = 0, nc = 0;
        if (run_case(fd, CASES[i].w, CASES[i].c, iters, &np, &nc) != 0) {
            skipped++;
            continue;
        }
        asked++;
        tot_pool += np;
        tot_conv += nc;
    }

    printf("\n");
    if (!asked) {
        printf("== the probe could not answer the question at any geometry ==\n");
        rocket_close(fd);
        return 1;
    }
    if (tot_pool == asked * iters && tot_conv == asked * iters) {
        printf("== A POOL PROGRAM RUNS INSIDE A CHAINED CONVOLUTION STREAM: %d of %d "
               "iterations over %d geometr(ies) gave the same pool output AND the same "
               "output from the convolution after it as separate submits ==\n",
               tot_pool, asked * iters, asked);
        printf("   So a per-block bitmap does not partition a stream, and a run finder may "
               "carry a pooling layer instead of breaking at one.\n");
    } else if (!tot_pool) {
        printf("== A POOL PROGRAM DOES NOT RUN INSIDE A CHAINED STREAM: 0 of %d "
               "iterations ==\n", asked * iters);
        printf("   Read the per-iteration column: 'NEVER RAN' says the PC did not start "
               "the PPU, 'STALE' says it ran too early, and the two want different fixes.\n");
    } else {
        printf("== MIXED: the pool was correct in %d of %d and the convolution after it in "
               "%d of %d ==\n", tot_pool, asked * iters, tot_conv, asked * iters);
        printf("   A race or a geometry dependence rather than a rule, so a pool in a "
               "stream is unsafe until the per-geometry lines above are explained.\n");
    }
    if (skipped)
        printf("   (%d geometr(ies) could not be asked and are NOT counted)\n", skipped);

    rocket_close(fd);
    return 0;
}
