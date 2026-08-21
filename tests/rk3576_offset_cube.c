// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_offset_cube.c — is a feature cube's base a plain ADDRESS on both sides of a
 * convolution, so that a program can read or write at a CHANNEL-GROUP OFFSET inside a
 * larger buffer?
 *
 * What wants it: MobileNetV2's residual topology fragments a cube chain far more than the
 * channel alignment does — of 31 refused joins, 10 are an ADD whose operands the host has
 * to concatenate and 10 a producer that must leave a row-major tensor for a skip several
 * layers later. Both want one thing: a producer writing its surface into a slice of a
 * bigger allocation, so that two producers' outputs ARE the concatenated operand and a
 * project convolution can take its skip as the tail of its own input cube.
 *
 * The arithmetic says it is free: the de-scatter reads channel group `g` at
 * `g*surf_elems*16` and a row task is already programmed as `output_dma + output_off`, so
 * a group offset is the same addition on the same register. But the row-task offset proves
 * offsets work along ONE axis, and the emitter derives the DDR group stride from the FULL
 * plane rather than from the base — so whether the base may sit inside a larger buffer is
 * a hardware question, and this asks it before anything is built on the answer.
 *
 * The hardware question, asked three ways on raw programs:
 *
 *   OUT   the same convolution, its output base moved by G channel groups inside one big
 *         BO. The slice must be bit-identical to the reference run at offset 0, and every
 *         byte outside it must still hold the sentinel — a writer that ignored the offset,
 *         or one that scattered past its slice, both show up here.
 *   IN    the same convolution, its FEATURE base moved by G groups inside a bigger cube.
 *         The output must be bit-identical to the reference.
 *   ROWS  the same, at a plane the planner splits into row tasks, so the group offset and
 *         the row offset compose on the same register.
 *
 * Then the LIBRARY question, which is what a caller actually gets: three layers where two
 * producers write their own slices of one buffer and the third reads all of it, against
 * the same three run through row-major tensors. Byte-identity or nothing. And a join whose
 * channel count is not a multiple of 32, which the cube path refuses only at a non-zero
 * weight zero point — the padding channels' weights are zero, so the coefficient group's B
 * term is the one thing that reads them.
 *
 * Usage: rk3576_offset_cube [gate]      (`gate` skips the raw probes)
 * Exit:  0 every part answered YES, 1 a part failed, 2 no NPU or the wrong chip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_pool.h"
#include "rocket_hw_profile.h"
#include "npu_regcmd_rk3576.h"
#include "npu_cna.h"

#define C2        16u
#define SENTINEL  0xA5
#define MAX_ROWS  64u

/* One int8 direct convolution, everything but its two base addresses frozen. */
struct prog {
    rocket_bo w, b, r;
    unsigned  ic, oc, ih, iw, kh, kw, sx, sy, pt, pl;
    unsigned  icreg, ocreg, ow, oh, surf_elems;
    size_t    cube_bytes, surf_bytes;
    uint64_t  ops[MAX_ROWS * RK3576_CONV_TASK_OPS];
};

static void bo_fill(int fd, rocket_bo *bo, size_t off, size_t n, int byte)
{
    rocket_bo_prep(fd, bo, 1, 0);
    memset((char *)bo->ptr + off, byte, n);
    rocket_bo_fini(fd, bo);
}

static int prog_init(int fd, struct prog *p, unsigned ic, unsigned oc,
                     unsigned iw, unsigned ih, unsigned k, unsigned s)
{
    size_t w_bytes, coeff;
    int32_t *bias;
    unsigned i;

    memset(p, 0, sizeof *p);
    p->ic = ic; p->oc = oc; p->iw = iw; p->ih = ih;
    p->kh = p->kw = k; p->sx = p->sy = s;
    p->pt = p->pl = k / 2u;
    p->icreg = rocket_rk3576_pad_ic(ic);
    p->ocreg = rocket_rk3576_pad_oc(oc);
    p->ow = (iw + 2u * p->pl - k) / s + 1u;
    p->oh = (ih + 2u * p->pt - k) / s + 1u;
    p->surf_elems = rocket_rk3576_out_surf_elems(p->ow, p->oh, 0);
    p->cube_bytes = (size_t)((p->icreg + C2 - 1u) / C2) * ih * iw * C2;
    p->surf_bytes = (size_t)((p->ocreg + C2 - 1u) / C2) * p->surf_elems * C2;

    w_bytes = (size_t)((p->ocreg + 31u) / 32u) * ((p->icreg + 31u) / 32u) *
              32u * 32u * p->kh * p->kw;
    coeff = rocket_rk3576_coeff_bytes(p->ocreg);
    if (rocket_bo_alloc(fd, w_bytes, &p->w) < 0 ||
        rocket_bo_alloc(fd, coeff, &p->b) < 0 ||
        rocket_bo_alloc(fd, sizeof p->ops, &p->r) < 0)
        return -1;

    /* Weights with no symmetry in any axis, so a surface that landed in the wrong place
     * cannot come out equal to one that landed in the right place. */
    rocket_bo_prep(fd, &p->w, 1, 0);
    {
        int8_t *w = (int8_t *)p->w.ptr;
        unsigned st = 0x2545F491u;
        for (i = 0; i < w_bytes; i++) {
            st = st * 1103515245u + 12345u;
            w[i] = (int8_t)((int)((st >> 16) % 9u) - 4);
        }
    }
    rocket_bo_fini(fd, &p->w);

    bias = calloc(p->ocreg, sizeof *bias);
    if (!bias) return -1;
    for (i = 0; i < p->ocreg; i++) bias[i] = (int32_t)(i % 7u) - 3;
    rocket_bo_prep(fd, &p->b, 1, 0);
    rocket_rk3576_pack_coeff_prec(p->b.ptr, coeff, bias, p->ocreg, precision_int8);
    rocket_bo_fini(fd, &p->b);
    free(bias);
    return 0;
}

static void prog_params(const struct prog *p, conv_params_t *q,
                        uint32_t in_dma, uint32_t out_dma)
{
    memset(q, 0, sizeof *q);
    q->ic = (uint16_t)p->icreg; q->oc = (uint16_t)p->ocreg;
    q->ih = (uint16_t)p->ih;    q->iw = (uint16_t)p->iw;
    q->oh = (uint16_t)p->oh;    q->ow = (uint16_t)p->ow;
    q->kh = (uint16_t)p->kh;    q->kw = (uint16_t)p->kw;
    q->stride_y = (uint8_t)p->sy; q->stride_x = (uint8_t)p->sx;
    q->pad_top = (uint8_t)p->pt;  q->pad_left = (uint8_t)p->pl;
    q->ih_full = (uint16_t)p->ih; q->oh_full = (uint16_t)p->oh;
    q->int8_out = 1;
    q->in_scale = 1.0f; q->w_scale = 1.0f; q->out_scale = 256.0f;
    q->input_zero_point = 0; q->output_zero_point = 0; q->weight_zero_point = 0;
    q->input_dma   = in_dma;
    q->weights_dma = (uint32_t)p->w.dma_address;
    q->bias_dma    = (uint32_t)p->b.dma_address;
    q->output_dma  = out_dma;
}

/* Emit and submit the whole row plan for one placement. Returns the task count, or -1. */
static int prog_run(int fd, struct prog *p, rocket_bo *in, size_t in_off,
                    rocket_bo *out, size_t out_off)
{
    rocket_rk3576_row_task plan[MAX_ROWS];
    rocket_task_desc td[MAX_ROWS];
    conv_params_t base, q;
    unsigned ntask = 1u, t;
    uint32_t task_ops = 0, in_h[4], out_h[1];
    size_t slot;

    prog_params(p, &base, (uint32_t)(in->dma_address + in_off),
                (uint32_t)(out->dma_address + out_off));
    q = base;
    if (rocket_rk3576_plan_rows(&q, 0, plan, MAX_ROWS, &ntask) < 0) return -1;

    slot = RK3576_CONV_TASK_OPS;
    for (t = 0; t < ntask; t++) {
        q = base;
        q.ih = plan[t].ih; q.oh = plan[t].oh;
        q.pad_top = plan[t].pad_top;
        q.input_dma  = base.input_dma  + plan[t].feature_off;
        q.output_dma = base.output_dma + plan[t].output_off;
        q.ih_full = (uint16_t)p->ih; q.oh_full = (uint16_t)p->oh;
        q.tasks = p->ops + (size_t)t * slot;
        q.task_count = 0;
        if (gen_conv2d_int8_rk3576(&q) != 0) return -1;
        if (!t) task_ops = q.task_count;
        else if (q.task_count != task_ops) return -1;
    }

    rocket_bo_prep(fd, &p->r, 1, 0);
    memcpy(p->r.ptr, p->ops, (size_t)ntask * slot * sizeof(uint64_t));
    rocket_bo_fini(fd, &p->r);
    for (t = 0; t < ntask; t++) {
        td[t].regcmd = (uint32_t)(p->r.dma_address + t * slot * sizeof(uint64_t));
        td[t].regcmd_count = task_ops;
    }

    in_h[0] = in->handle; in_h[1] = p->w.handle;
    in_h[2] = p->b.handle; in_h[3] = p->r.handle;
    out_h[0] = out->handle;
    if (ntask > 1u) {
        if (rocket_submit_tasks_flags(fd, td, ntask, in_h, 4, out_h, 1, 0) != 0) return -1;
    } else if (rocket_submit_matmul(fd, &p->r, task_ops, in_h, 4, out_h, 1, 2000) != 0) {
        return -1;
    }
    return (int)ntask;
}

/* Is every byte of [lo, hi) still the sentinel? Returns the first index that is not. */
static long first_touched(const unsigned char *o, size_t lo, size_t hi)
{
    size_t i;
    for (i = lo; i < hi; i++) if (o[i] != SENTINEL) return (long)i;
    return -1;
}

/* One geometry, all three questions. Returns 0 pass, 1 fail, -1 could not ask. */
static int run_case(int fd, unsigned ic, unsigned oc, unsigned iw, unsigned ih,
                    unsigned k, unsigned s, const unsigned *offs, unsigned n_offs)
{
    struct prog p;
    rocket_bo cube, ref_out, big_out, big_cube, ref2;
    unsigned char *ref = NULL;
    unsigned gsurf, gcube, gbig, i, o;
    size_t big_bytes, bigcube_bytes;
    int bad = 0, ntask;

    if (prog_init(fd, &p, ic, oc, iw, ih, k, s) != 0) {
        printf("   allocation/pack failed\n");
        return -1;
    }
    gsurf = (p.ocreg + C2 - 1u) / C2;
    gcube = (p.icreg + C2 - 1u) / C2;
    gbig  = gsurf + offs[n_offs - 1u];
    big_bytes     = (size_t)gbig * p.surf_elems * C2;
    bigcube_bytes = (size_t)(gcube + offs[n_offs - 1u]) * p.ih * p.iw * C2;

    if (rocket_bo_alloc(fd, p.cube_bytes, &cube) < 0 ||
        rocket_bo_alloc(fd, p.surf_bytes, &ref_out) < 0 ||
        rocket_bo_alloc(fd, p.surf_bytes, &ref2) < 0 ||
        rocket_bo_alloc(fd, big_bytes, &big_out) < 0 ||
        rocket_bo_alloc(fd, bigcube_bytes, &big_cube) < 0) {
        printf("   allocation failed\n");
        return -1;
    }

    rocket_bo_prep(fd, &cube, 1, 0);
    for (i = 0; i < p.cube_bytes; i++)
        ((int8_t *)cube.ptr)[i] = (int8_t)((int)((i * 7u + 11u) % 41u) - 20);
    rocket_bo_fini(fd, &cube);

    /* The reference: the same convolution, both bases at offset 0 in buffers of their
     * own. Everything below is compared against this and nothing else. */
    bo_fill(fd, &ref_out, 0, p.surf_bytes, SENTINEL);
    ntask = prog_run(fd, &p, &cube, 0, &ref_out, 0);
    if (ntask < 0) { printf("   the reference run failed\n"); return -1; }
    rocket_bo_prep(fd, &ref_out, 0, 2000000000ull);
    ref = malloc(p.surf_bytes);
    if (!ref) return -1;
    memcpy(ref, ref_out.ptr, p.surf_bytes);
    rocket_bo_fini(fd, &ref_out);
    if (first_touched(ref, 0, p.surf_bytes) < 0) {
        printf("   the reference surface is entirely sentinel — nothing wrote, so no "
               "comparison below means anything\n");
        return -1;
    }

    printf("   ic=%u oc=%u %ux%u k%u s%u: %u surface group(s) of %u elems, %u row task(s)\n",
           ic, oc, iw, ih, k, s, gsurf, p.surf_elems, (unsigned)ntask);

    /* ---- OUT: the surface base moved by G channel groups ---- */
    for (o = 0; o < n_offs; o++) {
        unsigned G = offs[o];
        size_t off = (size_t)G * p.surf_elems * C2;
        long touched;
        int same;

        bo_fill(fd, &big_out, 0, big_bytes, SENTINEL);
        if (prog_run(fd, &p, &cube, 0, &big_out, off) < 0) {
            printf("   OUT  G=%-2u the run failed\n", G);
            bad = 1; continue;
        }
        rocket_bo_prep(fd, &big_out, 0, 2000000000ull);
        same = memcmp((char *)big_out.ptr + off, ref, p.surf_bytes) == 0;
        touched = first_touched((unsigned char *)big_out.ptr, 0, off);
        if (touched < 0)
            touched = first_touched((unsigned char *)big_out.ptr,
                                    off + p.surf_bytes, big_bytes);
        rocket_bo_fini(fd, &big_out);
        printf("   OUT  G=%-2u byte %-8zu slice %s   outside %s\n", G, off,
               same ? "== reference" : "DIFFERS    ",
               touched < 0 ? "untouched" : "WRITTEN");
        if (!same || touched >= 0) bad = 1;
    }

    /* ---- IN: the feature base moved by G channel groups ---- */
    for (o = 0; o < n_offs; o++) {
        unsigned G = offs[o];
        size_t off = (size_t)G * p.ih * p.iw * C2;
        int same;

        rocket_bo_prep(fd, &big_cube, 1, 0);
        memset(big_cube.ptr, 0x5A, bigcube_bytes);
        memcpy((char *)big_cube.ptr + off, cube.ptr, p.cube_bytes);
        rocket_bo_fini(fd, &big_cube);
        bo_fill(fd, &ref2, 0, p.surf_bytes, SENTINEL);
        if (prog_run(fd, &p, &big_cube, off, &ref2, 0) < 0) {
            printf("   IN   G=%-2u the run failed\n", G);
            bad = 1; continue;
        }
        rocket_bo_prep(fd, &ref2, 0, 2000000000ull);
        same = memcmp(ref2.ptr, ref, p.surf_bytes) == 0;
        rocket_bo_fini(fd, &ref2);
        printf("   IN   G=%-2u byte %-8zu output %s\n", G, off,
               same ? "== reference" : "DIFFERS");
        if (!same) bad = 1;
    }

    /* ---- BOTH: input and output each at their own offset, in one run ---- */
    if (n_offs > 1u) {
        unsigned Gi = offs[1], Go = offs[n_offs - 1u];
        size_t ioff = (size_t)Gi * p.ih * p.iw * C2;
        size_t ooff = (size_t)Go * p.surf_elems * C2;
        int same;
        long touched;

        rocket_bo_prep(fd, &big_cube, 1, 0);
        memset(big_cube.ptr, 0x5A, bigcube_bytes);
        memcpy((char *)big_cube.ptr + ioff, cube.ptr, p.cube_bytes);
        rocket_bo_fini(fd, &big_cube);
        bo_fill(fd, &big_out, 0, big_bytes, SENTINEL);
        if (prog_run(fd, &p, &big_cube, ioff, &big_out, ooff) < 0) {
            printf("   BOTH the run failed\n");
            bad = 1;
        } else {
            rocket_bo_prep(fd, &big_out, 0, 2000000000ull);
            same = memcmp((char *)big_out.ptr + ooff, ref, p.surf_bytes) == 0;
            touched = first_touched((unsigned char *)big_out.ptr, 0, ooff);
            if (touched < 0)
                touched = first_touched((unsigned char *)big_out.ptr,
                                        ooff + p.surf_bytes, big_bytes);
            rocket_bo_fini(fd, &big_out);
            printf("   BOTH in G=%u out G=%u  slice %s   outside %s\n", Gi, Go,
                   same ? "== reference" : "DIFFERS    ",
                   touched < 0 ? "untouched" : "WRITTEN");
            if (!same || touched >= 0) bad = 1;
        }
    }

    free(ref);
    rocket_bo_free(fd, &cube); rocket_bo_free(fd, &ref_out);
    rocket_bo_free(fd, &ref2); rocket_bo_free(fd, &big_out);
    rocket_bo_free(fd, &big_cube);
    rocket_bo_free(fd, &p.w); rocket_bo_free(fd, &p.b); rocket_bo_free(fd, &p.r);
    return bad;
}

/* ============================================================================
 * THE LIBRARY GATE — three layers, once through row-major tensors and once with the two
 * producers writing SLICES of one buffer the consumer reads whole.
 *
 * That is the concatenation an add's operands need and the tail a fused project
 * convolution takes its skip from, in the smallest shape that has both: L1 writes the low
 * half, L2 reads the low half and writes the HIGH half of the same buffer, and L3 reads
 * all of it. The two runs must be BYTE-IDENTICAL — the slice is a placement decision and
 * nothing about the arithmetic — and the chained form must agree with both.
 * ==========================================================================*/
#define LP  28u          /* the plane */
#define LC  32u          /* channels per producer */

struct layer {
    rocket_conv2d_int8_weights_rk3576 *h;
    rocket_conv2d_desc d;
    int8_t  *W;
    int32_t *bias;
    float    in_s, w_s, out_s;
    int      in_zp, w_zp, out_zp;
};

static void layer_desc_p(struct layer *L, unsigned ic, unsigned oc, unsigned k,
                         unsigned plane)
{
    memset(&L->d, 0, sizeof L->d);
    L->d.ic = (int)ic; L->d.oc = (int)oc;
    L->d.ih = L->d.iw = (int)plane;
    L->d.kh = L->d.kw = (int)k;
    L->d.stride_y = L->d.stride_x = 1;
    L->d.pad_top = L->d.pad_left = (int)(k / 2u);
    L->d.dil_y = L->d.dil_x = 1;
}

static void layer_desc(struct layer *L, unsigned ic, unsigned oc, unsigned k)
{
    layer_desc_p(L, ic, oc, k, LP);
}

static void layer_desc_dw(struct layer *L, unsigned c, unsigned k, unsigned plane)
{
    layer_desc_p(L, c, c, k, plane);
    L->d.depthwise = 1;
}

static int layer_pack(int fd, struct layer *L, unsigned seed)
{
    size_t n = L->d.depthwise ? (size_t)L->d.oc * L->d.kh * L->d.kw
                              : (size_t)L->d.oc * L->d.ic * L->d.kh * L->d.kw;
    size_t i;
    unsigned st = seed;

    L->W = malloc(n);
    L->bias = calloc((size_t)L->d.oc, sizeof *L->bias);
    if (!L->W || !L->bias) return -1;
    for (i = 0; i < n; i++) {
        st = st * 1103515245u + 12345u;
        L->W[i] = (int8_t)((int)((st >> 16) % 11u) - 5);
    }
    for (i = 0; i < (size_t)L->d.oc; i++) L->bias[i] = (int32_t)(i % 13u) - 6;
    /* Scales that keep the output well inside int8 without saturating it flat — a
     * saturated surface would compare equal whatever the placement did. */
    L->in_s = 0.02f; L->w_s = 0.015f; L->out_s = 0.35f;
    /* THE PRODUCER'S OUTPUT ZERO POINT IS THE CONSUMER'S INPUT ZERO POINT, because it is
     * one tensor's quantization — which is the whole of why a producer's padded output
     * channels carry the border constant a consumer's fold assumes. A gate that gave the
     * two layers unrelated zero points would be testing a graph that cannot exist. */
    L->in_zp = 3; L->out_zp = 3;
    L->h = rocket_conv2d_int8_pack_rk3576(fd, &L->d, L->W, L->bias,
                                          L->in_s, L->w_s, NULL, L->out_s,
                                          L->in_zp, L->w_zp, L->out_zp);
    return L->h ? 0 : -1;
}

static void layer_drop(int fd, struct layer *L)
{
    if (L->h) rocket_conv2d_int8_weights_free_rk3576(fd, L->h);
    free(L->W); free(L->bias);
    memset(L, 0, sizeof *L);
}

static int lib_gate(int fd)
{
    struct layer L1, L2, L3;
    rocket_rk3576_cube buf, lo, hi;
    int8_t *in = NULL, *t1 = NULL, *t2 = NULL, *cat = NULL;
    int8_t *ref = NULL, *got = NULL;
    size_t px = (size_t)LP * LP, obytes = (size_t)LC * px;
    unsigned i;
    int bad = 0, rc;

    memset(&L1, 0, sizeof L1); memset(&L2, 0, sizeof L2); memset(&L3, 0, sizeof L3);
    layer_desc(&L1, LC, LC, 1u);
    layer_desc(&L2, LC, LC, 3u);
    layer_desc(&L3, 2u * LC, LC, 1u);
    if (layer_pack(fd, &L1, 0x1234567u) || layer_pack(fd, &L2, 0x89ABCDEu) ||
        layer_pack(fd, &L3, 0xFEDCBA9u)) {
        printf("   a layer would not pack\n");
        return -1;
    }

    in  = malloc(obytes);
    t1  = malloc(obytes); t2 = malloc(obytes);
    cat = malloc(2u * obytes);
    ref = malloc(obytes); got = malloc(obytes);
    if (!in || !t1 || !t2 || !cat || !ref || !got) return -1;
    for (i = 0; i < obytes; i++) in[i] = (int8_t)((int)((i * 13u + 7u) % 61u) - 30);

    /* ---- the reference: row-major throughout, the host doing the concatenation ---- */
    if (rocket_conv2d_int8_prepacked_rk3576(fd, L1.h, in, t1) != ROCKET_OK ||
        rocket_conv2d_int8_prepacked_rk3576(fd, L2.h, t1, t2) != ROCKET_OK) {
        printf("   the row-major producers failed\n"); return -1;
    }
    memcpy(cat, t1, obytes);
    memcpy(cat + obytes, t2, obytes);
    if (rocket_conv2d_int8_prepacked_rk3576(fd, L3.h, cat, ref) != ROCKET_OK) {
        printf("   the row-major consumer failed\n"); return -1;
    }

    /* ---- the slices ---- */
    if (rocket_rk3576_cube_alloc(fd, 2u * LC, LP, LP, &buf) != ROCKET_OK ||
        rocket_rk3576_cube_slice(&buf, 0u, LC, &lo) != ROCKET_OK ||
        rocket_rk3576_cube_slice(&buf, LC, LC, &hi) != ROCKET_OK) {
        printf("   the buffer or its slices would not build\n"); return -1;
    }
    printf("   buffer %u channels of %ux%u, slices at byte %zu and %zu\n",
           buf.c, buf.w, buf.h, lo.off, hi.off);
    if (rocket_conv2d_int8_cube_out_at_rk3576(L1.h, &lo) != ROCKET_OK ||
        rocket_conv2d_int8_cube_in_rk3576(L2.h, &lo) != ROCKET_OK ||
        rocket_conv2d_int8_cube_out_at_rk3576(L2.h, &hi) != ROCKET_OK ||
        rocket_conv2d_int8_cube_in_rk3576(L3.h, &buf) != ROCKET_OK) {
        printf("   the cube links would not be made\n"); return -1;
    }

    /* TWICE, because the second call is the first to reuse a held surface and a held
     * sentinel — the failure a single call cannot see is a stamp landing on a slice
     * somebody else has already written. */
    for (i = 0; i < 2u; i++) {
        memset(got, 0, obytes);
        if (rocket_conv2d_int8_prepacked_rk3576(fd, L1.h, in, NULL) != ROCKET_OK ||
            rocket_conv2d_int8_prepacked_rk3576(fd, L2.h, NULL, NULL) != ROCKET_OK ||
            rocket_conv2d_int8_prepacked_rk3576(fd, L3.h, NULL, got) != ROCKET_OK) {
            printf("   the sliced run failed\n"); bad = 1; break;
        }
        rc = memcmp(got, ref, obytes);
        printf("   per-layer, call %u: %s\n", i + 1u,
               rc ? "DIFFERS from the row-major reference" : "byte-identical");
        if (rc) bad = 1;
    }

    /* ---- the same two producers as ONE hardware kick ---- */
    {
        rocket_conv2d_int8_weights_rk3576 *hs[2];
        rocket_conv2d_int8_chain_rk3576 *ch;
        hs[0] = L1.h; hs[1] = L2.h;
        ch = rocket_conv2d_int8_chain_new_rk3576(fd, hs, 2u);
        if (!ch) {
            printf("   the chain would not build over two layers sharing one buffer\n");
            bad = 1;
        } else {
            for (i = 0; i < 2u; i++) {
                memset(got, 0, obytes);
                if (rocket_conv2d_int8_chain_run_rk3576(fd, ch, in, NULL) != ROCKET_OK ||
                    rocket_conv2d_int8_prepacked_rk3576(fd, L3.h, NULL, got) != ROCKET_OK) {
                    printf("   the chained run failed\n"); bad = 1; break;
                }
                rc = memcmp(got, ref, obytes);
                printf("   chained, call %u: %s (%u kick(s))\n", i + 1u,
                       rc ? "DIFFERS from the row-major reference" : "byte-identical",
                       rocket_conv2d_int8_chain_kicks_rk3576(ch));
                if (rc) bad = 1;
            }
            rocket_conv2d_int8_chain_free_rk3576(fd, ch);
        }
    }

    /* ---- the PPU reads a slice too ----
     * The pooling entry takes the same cube descriptor, so a pool consuming a producer that
     * was placed in a buffer is the same join — and it is a separate program with its own
     * DMA, so the offset is asked of the PPU rather than of the CNA. */
    {
        rocket_pool_desc pd;
        rocket_pool_int8_rk3576_handle *ph;
        int8_t *pref = NULL, *pgot = NULL;
        size_t pn = (size_t)LC * (LP / 2u) * (LP / 2u);

        memset(&pd, 0, sizeof pd);
        pd.c = (int)LC; pd.ih = (int)LP; pd.iw = (int)LP;
        pd.kh = pd.kw = 2; pd.stride_y = pd.stride_x = 2;
        pd.method = POOL_METHOD_AVG;
        ph = rocket_pool_int8_pack_rk3576(fd, &pd, 0);
        pref = malloc(pn); pgot = malloc(pn);
        if (!ph || !pref || !pgot) {
            printf("   the pool handle would not pack\n"); bad = 1;
        } else if (rocket_pool_int8_prepacked_rk3576(fd, ph, t2, pref) != ROCKET_OK) {
            printf("   the row-major pool failed\n"); bad = 1;
        } else if (rocket_pool_int8_cube_in_rk3576(ph, &hi) != ROCKET_OK) {
            printf("   the pool refused the high slice\n"); bad = 1;
        } else if (rocket_pool_int8_prepacked_rk3576(fd, ph, NULL, pgot) != ROCKET_OK) {
            printf("   the cube-in pool failed\n"); bad = 1;
        } else {
            rc = memcmp(pgot, pref, pn);
            printf("   pool over the high slice: %s\n",
                   rc ? "DIFFERS from the row-major pool" : "byte-identical");
            if (rc) bad = 1;
        }
        if (ph) rocket_pool_int8_free_rk3576(fd, ph);
        free(pref); free(pgot);
    }

    /* ---- the refusals: a slice that is not this handle's shape ---- */
    {
        rocket_rk3576_cube odd;
        if (rocket_rk3576_cube_slice(&buf, 8u, LC, &odd) == ROCKET_OK) {
            printf("   a slice starting mid-group was ACCEPTED\n"); bad = 1;
        }
        if (rocket_rk3576_cube_slice(&buf, LC, 2u * LC, &odd) == ROCKET_OK) {
            printf("   a slice running past the buffer was ACCEPTED\n"); bad = 1;
        }
        if (rocket_rk3576_cube_slice(&buf, LC, LC, &odd) == ROCKET_OK) {
            odd.h = LP + 1u;
            if (rocket_conv2d_int8_cube_out_at_rk3576(L1.h, &odd) == ROCKET_OK) {
                printf("   a slice of the wrong plane was ACCEPTED\n"); bad = 1;
            }
        }
        /* Put L1 back where the loop above left it, in case anything is added below. */
        if (rocket_conv2d_int8_cube_out_at_rk3576(L1.h, &lo) != ROCKET_OK) bad = 1;
    }

    free(in); free(t1); free(t2); free(cat); free(ref); free(got);
    layer_drop(fd, &L1); layer_drop(fd, &L2); layer_drop(fd, &L3);
    rocket_rk3576_cube_free(fd, &buf);
    return bad;
}

/* An UNALIGNED join: a producer of `mid` output channels feeding a consumer of `mid` input
 * channels, where `mid` is not a multiple of 32. The consumer's feature DMA walks the
 * round-32 count, so the channels past `mid` come from the producer's own padded output
 * rather than from the border constant this handle would have filled. Their weights are
 * zero, so the only thing that can see them is the coefficient group's B term — which a
 * symmetric weight quantization does not have at all, and which an ASYMMETRIC one meets
 * because a direct producer's partial output group lands on its output zero point, the
 * consumer's own border constant.
 *
 * `w_zp` is the CONSUMER'S, and it is the axis: at zero the padding content cannot be read
 * and at anything else it is the whole question. `dw` makes the consumer depthwise, which
 * is the other side of the same join — its coefficient group has no B field and its weight
 * zero point rides in the cube instead.
 *
 * The claim is bit-identity with the row-major path; anything else says B is not the whole
 * story. `expect_refuse` is for the producer whose surface does NOT declare a tail. */
static int unaligned_gate_p(int fd, unsigned ic, unsigned mid, unsigned oc, int w_zp,
                            int dw_prod, int dw, int expect_refuse, unsigned plane,
                            int prod_w_zp)
{
    struct layer L1, L2;
    rocket_rk3576_cube c;
    int8_t *in = NULL, *t = NULL, *ref = NULL, *got = NULL;
    size_t px = (size_t)plane * plane;
    unsigned i;
    int bad = 0, rc;

    memset(&L1, 0, sizeof L1); memset(&L2, 0, sizeof L2);
    if (dw_prod) { layer_desc_dw(&L1, mid, 3u, plane); ic = mid; }
    else           layer_desc_p(&L1, ic, mid, 1u, plane);
    if (dw) { layer_desc_dw(&L2, mid, 3u, plane); oc = mid; }
    else      layer_desc_p(&L2, mid, oc, 3u, plane);
    L1.w_zp = prod_w_zp;
    L2.w_zp = w_zp;
    if (layer_pack(fd, &L1, 0x51A7C0DEu) || layer_pack(fd, &L2, 0x0FFCBE55u)) {
        printf("   a layer would not pack\n"); return -1;
    }
    in = malloc((size_t)ic * px); t = malloc((size_t)mid * px);
    ref = malloc((size_t)oc * px); got = malloc((size_t)oc * px);
    if (!in || !t || !ref || !got) return -1;
    for (i = 0; i < (unsigned)((size_t)ic * px); i++)
        in[i] = (int8_t)((int)((i * 13u + 7u) % 61u) - 30);

    if (rocket_conv2d_int8_prepacked_rk3576(fd, L1.h, in, t) != ROCKET_OK ||
        rocket_conv2d_int8_prepacked_rk3576(fd, L2.h, t, ref) != ROCKET_OK) {
        printf("   the row-major pair failed\n"); return -1;
    }
    if (rocket_conv2d_int8_cube_of_rk3576(L1.h, &c) != ROCKET_OK ||
        rocket_conv2d_int8_cube_in_rk3576(L2.h, &c) != ROCKET_OK ||
        rocket_conv2d_int8_cube_out_rk3576(L1.h, 1) != ROCKET_OK) {
        printf("   ic=%-3u w_zp=%-4d %s<-%s: %s\n", mid, w_zp,
               dw ? "dw    " : "direct", dw_prod ? "dw    " : "direct",
               expect_refuse ? "refused, as a producer that declares no tail requires"
                             : "THE JOIN WAS REFUSED");
        layer_drop(fd, &L1); layer_drop(fd, &L2);
        free(in); free(t); free(ref); free(got);
        return expect_refuse ? 0 : 1;
    }
    if (expect_refuse) {
        printf("   ic=%-3u w_zp=%-4d %s<-%s: ACCEPTED a cube whose tail nothing "
               "declares\n", mid, w_zp, dw ? "dw    " : "direct",
               dw_prod ? "dw    " : "direct");
        layer_drop(fd, &L1); layer_drop(fd, &L2);
        free(in); free(t); free(ref); free(got);
        return 1;
    }
    for (i = 0; i < 2u; i++) {
        memset(got, 0, (size_t)oc * px);
        if (rocket_conv2d_int8_prepacked_rk3576(fd, L1.h, in, NULL) != ROCKET_OK ||
            rocket_conv2d_int8_prepacked_rk3576(fd, L2.h, NULL, got) != ROCKET_OK) {
            printf("   the cube-linked pair failed\n"); bad = 1; break;
        }
        rc = memcmp(got, ref, (size_t)oc * px);
        printf("   ic=%-3u w_zp=%-4d %s<-%s %ux%u call %u: %s\n", mid, w_zp,
               dw ? "dw    " : "direct", dw_prod ? "dw    " : "direct", plane, plane,
               i + 1u, rc ? "DIFFERS from the row-major reference" : "byte-identical");
        if (rc) bad = 1;
    }
    free(in); free(t); free(ref); free(got);
    layer_drop(fd, &L1); layer_drop(fd, &L2);
    return bad;
}

static int unaligned_gate(int fd, unsigned ic, unsigned mid, unsigned oc, int w_zp,
                          int dw_prod, int dw, int expect_refuse)
{
    return unaligned_gate_p(fd, ic, mid, oc, w_zp, dw_prod, dw, expect_refuse, LP, 0);
}

int main(int argc, char **argv)
{
    static const unsigned OFFS[] = { 0u, 1u, 2u, 3u, 5u };
    int fd, bad = 0, rc, only_lib = (argc > 1 && !strcmp(argv[1], "gate"));
    rocket_bo guard;

    fd = rocket_open();
    if (fd < 0) { printf("no /dev/accel/accel0\n"); return 2; }
    {
        const struct rocket_hw_profile *hw = rocket_hw_current();
        if (!hw || !hw->name || strcmp(hw->name, "rk3576")) {
            printf("not an RK3576\n"); rocket_close(fd); return 2;
        }
    }
    /* IOVA 0 is a real buffer on this part, so nothing else may land there. */
    if (rocket_bo_alloc(fd, 4096, &guard) < 0) { rocket_close(fd); return 2; }

    if (!only_lib) {
        printf("== is a cube base a plain address? ==\n");
        printf("-- one row task --\n");
        rc = run_case(fd, 32, 32, 32, 32, 1, 1, OFFS, 5u);
        if (rc) bad = 1;
        printf("-- a taller kernel, still one task --\n");
        rc = run_case(fd, 64, 64, 28, 28, 3, 1, OFFS, 5u);
        if (rc) bad = 1;
        printf("-- a plane the planner splits into row tasks --\n");
        rc = run_case(fd, 32, 32, 112, 112, 3, 1, OFFS, 4u);
        if (rc) bad = 1;
    }

    printf("\n== the library entries: two producers, one buffer ==\n");
    rc = lib_gate(fd);
    if (rc) bad = 1;

    printf("\n== a join whose channel count is not a multiple of 32 ==\n");
    /* A SYMMETRIC weight quantization: the B term does not exist, so the padding channels
     * cannot be read whatever they hold. */
    rc = unaligned_gate(fd, 32u, 24u, 32u, 0, 0, 0, 0);
    if (rc) bad = 1;
    rc = unaligned_gate(fd, 32u, 144u, 32u, 0, 0, 0, 0);
    if (rc) bad = 1;
    rc = unaligned_gate(fd, 32u, 16u, 32u, 0, 0, 0, 0);
    if (rc) bad = 1;
    /* AN ASYMMETRIC one, which is what a legacy uint8 model carries. The B term sums the
     * padding channels, and what makes the join sound is that a direct producer's partial
     * output group lands on its own output zero point — the consumer's border constant. */
    rc = unaligned_gate(fd, 32u, 24u, 32u, -24, 0, 0, 0);
    if (rc) bad = 1;
    rc = unaligned_gate(fd, 32u, 16u, 32u, 16, 0, 0, 0);
    if (rc) bad = 1;
    rc = unaligned_gate(fd, 32u, 144u, 32u, -76, 0, 0, 0);
    if (rc) bad = 1;
    /* The same join into a DEPTHWISE consumer, whose coefficient group has no B field and
     * whose weight zero point rides in the cube instead. */
    rc = unaligned_gate(fd, 32u, 144u, 0u, -6, 0, 1, 0);
    if (rc) bad = 1;
    rc = unaligned_gate(fd, 32u, 24u, 0u, 15, 0, 1, 0);
    if (rc) bad = 1;
    /* THE NEGATIVE CONTROL. A DEPTHWISE producer is programmed with the raw output count
     * and leaves everything past it untouched, so its surface declares no tail and the
     * same join must still be refused. */
    rc = unaligned_gate(fd, 24u, 24u, 32u, -24, 1, 0, 1);
    if (rc) bad = 1;
    /* THE SAME JOINS AT A PLANE THE PLANNER SPLITS INTO ROW TASKS, which is where a real
     * graph's unaligned joins are — MobileNetV2's are at 112x112 and 56x56, not at the
     * 28x28 a per-op gate reaches for. */
    rc = unaligned_gate_p(fd, 32u, 16u, 96u, -1, 0, 0, 0, 112u, 0);
    if (rc) bad = 1;
    rc = unaligned_gate_p(fd, 24u, 144u, 24u, -76, 0, 0, 0, 56u, 0);
    if (rc) bad = 1;
    rc = unaligned_gate_p(fd, 24u, 144u, 0u, -6, 0, 1, 0, 56u, 0);
    if (rc) bad = 1;
    /* AND THE PRODUCER'S OWN weight zero point, which is the axis a gate whose producers
     * were all symmetric could not see. Its B term reaches the padding output channels
     * exactly as it reaches the live ones, so unless the tail is given none, what it
     * carries is `requant(B*sum(x))` — data-dependent, and a consumer folding it as a
     * constant computes a wrong answer everywhere. MobileNetV2's layer 2 -> 3 is this. */
    rc = unaligned_gate_p(fd, 32u, 16u, 96u, -1, 0, 0, 0, 112u, 12);
    if (rc) bad = 1;
    rc = unaligned_gate_p(fd, 32u, 24u, 32u, -24, 0, 0, 0, 28u, -24);
    if (rc) bad = 1;
    rc = unaligned_gate_p(fd, 24u, 144u, 0u, -6, 0, 1, 0, 56u, 16);
    if (rc) bad = 1;

    rocket_bo_free(fd, &guard);
    rocket_close(fd);
    printf("\n%s\n", bad ? "FAIL: a base is not a plain address on some path"
                         : "PASS: a group offset on either base is honoured, a writer "
                           "stays inside its slice, and the library entries are "
                           "byte-identical to the row-major path");
    return bad;
}
