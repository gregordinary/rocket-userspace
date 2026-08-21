// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_surf_stride.c — will the CNA read a feature cube whose channel groups sit
 * FURTHER APART than the plane?
 *
 * What wants it: a depthwise convolution writes `round4(ow*oh)` elements per output
 * channel group where a direct one writes `ow*oh` exactly, so a depthwise producer's
 * surface is a direct consumer's input cube only when the plane is already a multiple of
 * four. MobileNetV2's four 7x7 joins are not — 52 against 49 — and the cube chain refuses
 * them, materialising a row-major tensor at each and paying a submit for it.
 *
 * The refusal reads as geometric, but the quantity a consumer needs is a REGISTER the
 * emitter derives rather than a constant: `R76_CNA_SURF_FULL` (`0x1094`) carries the DDR
 * channel-group jump in 16-byte atoms and is programmed `iw * ih_full`. Whether the part
 * honours a value LARGER than the plane is a hardware question and nothing decoded so far
 * answers it — the PPU accepting a stride SMALLER than `round4` on the same two numbers is
 * a different register on a different block and says nothing about this one.
 *
 * The question, asked on raw programs:
 *
 *   STRIDE   the same convolution, its feature cube laid out with P extra elements
 *            between channel groups and `0x1094` told the padded stride. The output must
 *            be bit-identical to the reference run over a tight cube. The gap elements
 *            hold a poison byte, so a program that read at the tight stride would pick
 *            them up and cannot come out equal by luck.
 *   CONTROL  the same padded buffer with the stride NOT told — `0x1094` left at the
 *            derived plane. This must DIFFER. If it does not, the register is not what
 *            carries the group jump at this geometry and the STRIDE result above is
 *            vacuous, whatever it says.
 *   ROWS     both of the above at a plane the planner splits into row tasks, which is
 *            where `R76_CNA_SURF_TASK` (`0x1098`, `round4(iw*fetch_rows)`) would be read
 *            if the task's own window carried a DDR quantity too. It is left derived
 *            here, so a pass says the group jump is the one register.
 *
 * Every case runs at more than one channel group, because a single-group cube has no
 * group jump to get wrong and would pass whatever the register did.
 *
 * Usage: rk3576_surf_stride
 * Exit:  0 the part honours a padded group stride, 1 it does not, 2 no NPU or wrong chip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "npu_regcmd_rk3576.h"
#include "npu_cna.h"

#define C2        16u
#define POISON    0x5A
#define MAX_ROWS  128u

/* One int8 direct convolution, everything but its feature stride frozen. */
struct prog {
    rocket_bo w, b, r;
    unsigned  ic, oc, ih, iw, kh, kw, sx, sy, pt, pl;
    unsigned  icreg, ocreg, ow, oh, surf_elems, gcube;
    size_t    surf_bytes;
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
    p->gcube = (p->icreg + C2 - 1u) / C2;
    p->surf_bytes = (size_t)((p->ocreg + C2 - 1u) / C2) * p->surf_elems * C2;

    w_bytes = (size_t)((p->ocreg + 31u) / 32u) * ((p->icreg + 31u) / 32u) *
              32u * 32u * p->kh * p->kw;
    coeff = rocket_rk3576_coeff_bytes(p->ocreg);
    if (rocket_bo_alloc(fd, w_bytes, &p->w) < 0 ||
        rocket_bo_alloc(fd, coeff, &p->b) < 0 ||
        rocket_bo_alloc(fd, sizeof p->ops, &p->r) < 0)
        return -1;

    /* Weights with no symmetry in any axis, so a surface built from the wrong bytes
     * cannot come out equal to one built from the right ones. */
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
                        uint32_t in_dma, uint32_t out_dma, unsigned in_surf)
{
    memset(q, 0, sizeof *q);
    q->ic = (uint16_t)p->icreg; q->oc = (uint16_t)p->ocreg;
    q->ih = (uint16_t)p->ih;    q->iw = (uint16_t)p->iw;
    q->oh = (uint16_t)p->oh;    q->ow = (uint16_t)p->ow;
    q->kh = (uint16_t)p->kh;    q->kw = (uint16_t)p->kw;
    q->stride_y = (uint8_t)p->sy; q->stride_x = (uint8_t)p->sx;
    q->pad_top = (uint8_t)p->pt;  q->pad_left = (uint8_t)p->pl;
    q->ih_full = (uint16_t)p->ih; q->oh_full = (uint16_t)p->oh;
    q->in_surf_elems = in_surf;
    q->int8_out = 1;
    q->in_scale = 1.0f; q->w_scale = 1.0f; q->out_scale = 256.0f;
    q->input_zero_point = 0; q->output_zero_point = 0; q->weight_zero_point = 0;
    q->input_dma   = in_dma;
    q->weights_dma = (uint32_t)p->w.dma_address;
    q->bias_dma    = (uint32_t)p->b.dma_address;
    q->output_dma  = out_dma;
}

/* Emit and submit the whole row plan. `in_surf` of 0 leaves the emitter to derive the
 * group stride from the plane. Returns the task count, or -1. */
static int prog_run(int fd, struct prog *p, rocket_bo *in, rocket_bo *out,
                    unsigned in_surf)
{
    rocket_rk3576_row_task plan[MAX_ROWS];
    rocket_task_desc td[MAX_ROWS];
    conv_params_t base, q;
    unsigned ntask = 1u, t;
    uint32_t task_ops = 0, in_h[4], out_h[1];
    size_t slot;

    prog_params(p, &base, (uint32_t)in->dma_address, (uint32_t)out->dma_address,
                in_surf);
    q = base;
    if (rocket_rk3576_plan_rows(&q, 0, plan, MAX_ROWS, &ntask) < 0) return -1;

    slot = RK3576_CONV_TASK_OPS;
    for (t = 0; t < ntask; t++) {
        q = base;
        q.ih = plan[t].ih; q.oh = plan[t].oh;
        q.pad_top = plan[t].pad_top;
        /* A row offset lands INSIDE a channel group, so it is the same addition
         * whatever the group jump is. */
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
        if (rocket_submit_tasks_flags(fd, td, ntask, in_h, 4, out_h, 1, 0) != 0)
            return -1;
    } else if (rocket_submit_matmul(fd, &p->r, task_ops, in_h, 4, out_h, 1, 2000) != 0) {
        return -1;
    }
    return (int)ntask;
}

/* The feature byte at group g, row y, column x, channel lane c, in a cube of `stride`
 * elements per group. */
static int8_t feat_byte(unsigned g, unsigned y, unsigned x, unsigned c)
{
    unsigned v = g * 37u + y * 11u + x * 5u + c * 3u;
    return (int8_t)((int)(v % 61u) - 30);
}

static void cube_fill(int fd, rocket_bo *bo, const struct prog *p, size_t stride)
{
    unsigned g, y, x, c;

    rocket_bo_prep(fd, bo, 1, 0);
    memset(bo->ptr, POISON, (size_t)p->gcube * stride * C2);
    for (g = 0; g < p->gcube; g++)
        for (y = 0; y < p->ih; y++)
            for (x = 0; x < p->iw; x++)
                for (c = 0; c < C2; c++)
                    ((int8_t *)bo->ptr)[((size_t)g * stride + (size_t)y * p->iw + x)
                                        * C2 + c] = feat_byte(g, y, x, c);
    rocket_bo_fini(fd, bo);
}

/* One geometry. `pads` are the EXTRA elements between channel groups. Returns 0 pass,
 * 1 fail, -1 could not ask. */
static int run_case(int fd, unsigned ic, unsigned oc, unsigned iw, unsigned ih,
                    unsigned k, unsigned s, const unsigned *pads, unsigned npads,
                    unsigned force_rows)
{
    struct prog p;
    rocket_bo tight, padded, out;
    unsigned char *ref = NULL;
    unsigned plane, i, maxpad = 0;
    int bad = 0, ntask, ctl_ntask;
    char buf[32];

    if (force_rows) {
        snprintf(buf, sizeof buf, "%u", force_rows);
        setenv("ROCKET_RK3576_MAX_ROWS", buf, 1);
    } else {
        unsetenv("ROCKET_RK3576_MAX_ROWS");
    }

    if (prog_init(fd, &p, ic, oc, iw, ih, k, s) != 0) {
        printf("   allocation/pack failed\n");
        unsetenv("ROCKET_RK3576_MAX_ROWS");
        return -1;
    }
    plane = p.iw * p.ih;
    for (i = 0; i < npads; i++) if (pads[i] > maxpad) maxpad = pads[i];

    if (rocket_bo_alloc(fd, (size_t)p.gcube * plane * C2, &tight) < 0 ||
        rocket_bo_alloc(fd, (size_t)p.gcube * (plane + maxpad) * C2, &padded) < 0 ||
        rocket_bo_alloc(fd, p.surf_bytes, &out) < 0) {
        printf("   allocation failed\n");
        unsetenv("ROCKET_RK3576_MAX_ROWS");
        return -1;
    }

    /* ---- the reference: a tight cube, the stride derived as always ---- */
    cube_fill(fd, &tight, &p, plane);
    bo_fill(fd, &out, 0, p.surf_bytes, POISON);
    ntask = prog_run(fd, &p, &tight, &out, 0);
    if (ntask < 0) { printf("   the reference run failed\n"); bad = -1; goto done; }
    rocket_bo_prep(fd, &out, 0, 2000000000ull);
    ref = malloc(p.surf_bytes);
    if (!ref) { bad = -1; goto done; }
    memcpy(ref, out.ptr, p.surf_bytes);
    rocket_bo_fini(fd, &out);
    for (i = 0; i < p.surf_bytes; i++) if (ref[i] != POISON) break;
    if (i == p.surf_bytes) {
        printf("   the reference surface is entirely poison — nothing wrote, so no "
               "comparison below means anything\n");
        bad = -1; goto done;
    }

    printf("   ic=%u oc=%u %ux%u k%u s%u: %u feature group(s), plane %u elems, "
           "%u row task(s)\n", ic, oc, iw, ih, k, s, p.gcube, plane, (unsigned)ntask);

    for (i = 0; i < npads; i++) {
        unsigned S = plane + pads[i];
        int same;

        cube_fill(fd, &padded, &p, S);
        bo_fill(fd, &out, 0, p.surf_bytes, POISON);
        if (prog_run(fd, &p, &padded, &out, S) < 0) {
            printf("   STRIDE  %-6u (+%-3u) the run failed\n", S, pads[i]);
            bad = 1; continue;
        }
        rocket_bo_prep(fd, &out, 0, 2000000000ull);
        same = memcmp(out.ptr, ref, p.surf_bytes) == 0;
        rocket_bo_fini(fd, &out);
        printf("   STRIDE  %-6u (+%-3u) output %s\n", S, pads[i],
               same ? "== reference" : "DIFFERS");
        if (!same) bad = 1;
    }

    /* ---- THE CONTROL. The same padded buffer, the stride left derived. ---- */
    {
        unsigned S = plane + pads[0];
        int same;

        cube_fill(fd, &padded, &p, S);
        bo_fill(fd, &out, 0, p.surf_bytes, POISON);
        ctl_ntask = prog_run(fd, &p, &padded, &out, 0);
        if (ctl_ntask < 0) {
            printf("   CONTROL %-6u        the run failed\n", S);
            bad = 1;
        } else {
            rocket_bo_prep(fd, &out, 0, 2000000000ull);
            same = memcmp(out.ptr, ref, p.surf_bytes) == 0;
            rocket_bo_fini(fd, &out);
            printf("   CONTROL %-6u        stride not told: output %s\n", S,
                   same ? "== reference, so 0x1094 IS NOT the group jump here and the "
                          "STRIDE results above mean nothing"
                        : "DIFFERS, as it must");
            if (same) bad = 1;
        }
    }

done:
    free(ref);
    rocket_bo_free(fd, &tight); rocket_bo_free(fd, &padded); rocket_bo_free(fd, &out);
    rocket_bo_free(fd, &p.w); rocket_bo_free(fd, &p.b); rocket_bo_free(fd, &p.r);
    unsetenv("ROCKET_RK3576_MAX_ROWS");
    return bad;
}

/* ============================================================================
 * THE LIBRARY GATE — a DEPTHWISE producer at a plane whose element count is not a multiple
 * of four, feeding a consumer through a cube instead of through a row-major tensor.
 *
 * That is the join MobileNetV2 refuses four times: a 7x7 depthwise writes 52 elements per
 * channel group and its project convolution reads a plane of 49. The claim is byte-identity
 * with the row-major path — the stride is a layout fact and nothing about the arithmetic —
 * and it is asked at a non-zero weight zero point too, because that is the one thing that
 * reads a producer's padded output channels.
 * ==========================================================================*/
struct layer {
    rocket_conv2d_int8_weights_rk3576 *h;
    rocket_conv2d_desc d;
    int8_t  *W;
    int32_t *bias;
    int      w_zp;
};

static void layer_desc(struct layer *L, unsigned ic, unsigned oc, unsigned k,
                       unsigned plane, int dw)
{
    memset(&L->d, 0, sizeof L->d);
    L->d.ic = (int)ic; L->d.oc = (int)oc;
    L->d.ih = L->d.iw = (int)plane;
    L->d.kh = L->d.kw = (int)k;
    L->d.stride_y = L->d.stride_x = 1;
    L->d.pad_top = L->d.pad_left = (int)(k / 2u);
    L->d.dil_y = L->d.dil_x = 1;
    L->d.depthwise = dw;
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
    /* One tensor's quantization, so the producer's output zero point IS the consumer's
     * input zero point — which is what makes a padded tail the border constant. */
    L->h = rocket_conv2d_int8_pack_rk3576(fd, &L->d, L->W, L->bias,
                                          0.02f, 0.015f, NULL, 0.35f, 3, L->w_zp, 3);
    return L->h ? 0 : -1;
}

static void layer_drop(int fd, struct layer *L)
{
    if (L->h) rocket_conv2d_int8_weights_free_rk3576(fd, L->h);
    free(L->W); free(L->bias);
    memset(L, 0, sizeof *L);
}

static int join_gate(int fd, unsigned plane, unsigned c, unsigned oc, unsigned k,
                     int dw_prod, int dw_cons, int w_zp)
{
    struct layer L1, L2;
    rocket_rk3576_cube cube;
    int8_t *in = NULL, *t = NULL, *ref = NULL, *got = NULL;
    size_t px = (size_t)plane * plane;
    unsigned i;
    int bad = 0, rc;

    memset(&L1, 0, sizeof L1); memset(&L2, 0, sizeof L2);
    layer_desc(&L1, c, c, dw_prod ? 3u : 1u, plane, dw_prod);
    if (dw_cons) oc = c;
    layer_desc(&L2, c, oc, k, plane, dw_cons);
    L2.w_zp = w_zp;
    if (layer_pack(fd, &L1, 0x51A7C0DEu) || layer_pack(fd, &L2, 0x0FFCBE55u)) {
        printf("   a layer would not pack\n");
        layer_drop(fd, &L1); layer_drop(fd, &L2);
        return -1;
    }
    in = malloc((size_t)c * px); t = malloc((size_t)c * px);
    ref = malloc((size_t)oc * px); got = malloc((size_t)oc * px);
    if (!in || !t || !ref || !got) { bad = -1; goto done; }
    for (i = 0; i < (unsigned)((size_t)c * px); i++)
        in[i] = (int8_t)((int)((i * 13u + 7u) % 61u) - 30);

    if (rocket_conv2d_int8_prepacked_rk3576(fd, L1.h, in, t) != ROCKET_OK ||
        rocket_conv2d_int8_prepacked_rk3576(fd, L2.h, t, ref) != ROCKET_OK) {
        printf("   the row-major pair failed\n"); bad = -1; goto done;
    }
    if (rocket_conv2d_int8_cube_of_rk3576(L1.h, &cube) != ROCKET_OK ||
        rocket_conv2d_int8_cube_in_rk3576(L2.h, &cube) != ROCKET_OK ||
        rocket_conv2d_int8_cube_out_rk3576(L1.h, 1) != ROCKET_OK) {
        printf("   %ux%u c=%-3u %s<-%s w_zp=%-4d: THE JOIN WAS REFUSED\n",
               plane, plane, c, dw_cons ? "dw    " : "direct",
               dw_prod ? "dw    " : "direct", w_zp);
        bad = 1; goto done;
    }
    /* Twice, because the second call is the first to reuse a held surface. */
    for (i = 0; i < 2u; i++) {
        memset(got, 0, (size_t)oc * px);
        if (rocket_conv2d_int8_prepacked_rk3576(fd, L1.h, in, NULL) != ROCKET_OK ||
            rocket_conv2d_int8_prepacked_rk3576(fd, L2.h, NULL, got) != ROCKET_OK) {
            printf("   the cube-linked pair failed\n"); bad = 1; break;
        }
        rc = memcmp(got, ref, (size_t)oc * px);
        printf("   %ux%u c=%-3u %s<-%s w_zp=%-4d stride %-4zu call %u: %s\n",
               plane, plane, c, dw_cons ? "dw    " : "direct",
               dw_prod ? "dw    " : "direct", w_zp, cube.surf_elems, i + 1u,
               rc ? "DIFFERS from the row-major reference" : "byte-identical");
        if (rc) bad = 1;
    }

done:
    free(in); free(t); free(ref); free(got);
    layer_drop(fd, &L1); layer_drop(fd, &L2);
    return bad;
}

int main(void)
{
    /* round4 padding is the case a depthwise producer's surface actually presents; the
     * wider ones say whether the register is a general stride or only tolerates that. */
    static const unsigned PADS[]  = { 3u, 16u, 64u };
    static const unsigned PADS1[] = { 3u };
    int fd, bad = 0, rc;
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

    printf("== will the CNA read a feature cube with a padded group stride? ==\n");
    printf("-- 7x7, the plane MobileNetV2's depthwise joins refuse at --\n");
    rc = run_case(fd, 32, 32, 7, 7, 1, 1, PADS, 3u, 0);   if (rc > 0) bad = 1;
    rc = run_case(fd, 64, 32, 7, 7, 3, 1, PADS, 3u, 0);   if (rc > 0) bad = 1;
    /* The real width at that plane: MobileNetV2's last block projects 960 -> 320. */
    rc = run_case(fd, 960, 32, 7, 7, 1, 1, PADS1, 1u, 0); if (rc > 0) bad = 1;
    printf("-- other planes whose element count is not a multiple of four --\n");
    rc = run_case(fd, 32, 32, 5, 5, 3, 1, PADS, 3u, 0);   if (rc > 0) bad = 1;
    rc = run_case(fd, 32, 32, 13, 13, 1, 1, PADS, 3u, 0); if (rc > 0) bad = 1;
    printf("-- a plane the planner splits into row tasks: is 0x1098 read too? --\n");
    rc = run_case(fd, 32, 32, 56, 56, 3, 1, PADS1, 1u, 8u);  if (rc > 0) bad = 1;
    rc = run_case(fd, 64, 32, 28, 28, 3, 1, PADS1, 1u, 4u);  if (rc > 0) bad = 1;

    printf("\n== the library entries: a join off a producer with a padded stride ==\n");
    /* The refused join itself: a depthwise producer at 7x7 (52 elements per group) into
     * the project convolution that reads a plane of 49. */
    if (join_gate(fd, 7u, 32u, 64u, 1u, 1, 0, 0)   > 0) bad = 1;
    if (join_gate(fd, 7u, 32u, 64u, 1u, 1, 0, -24) > 0) bad = 1;
    if (join_gate(fd, 7u, 96u, 32u, 1u, 1, 0, 12)  > 0) bad = 1;
    /* A 3x3 consumer, and a DEPTHWISE one — the emitter programs the same register on
     * both, but neither is the direct 1x1 the raw probe asked at. */
    if (join_gate(fd, 7u, 64u, 32u, 3u, 1, 0, -6)  > 0) bad = 1;
    if (join_gate(fd, 7u, 32u, 0u,  3u, 1, 1, 0)   > 0) bad = 1;
    /* A plane whose element count IS a multiple of four, where nothing is padded — the
     * control that says the change did not alter a join that already worked. */
    if (join_gate(fd, 8u, 32u, 64u, 1u, 1, 0, 0)   > 0) bad = 1;
    if (join_gate(fd, 7u, 32u, 64u, 1u, 0, 0, 0)   > 0) bad = 1;

    rocket_bo_free(fd, &guard);
    rocket_close(fd);
    printf("\n%s\n", bad
        ? "FAIL: the part does not honour a feature group stride larger than the plane"
        : "PASS: the CNA's DDR group stride is a caller-supplied quantity, the task "
          "window stride is not read as one, a padded cube computes bit-exactly, and a "
          "depthwise producer's surface joins like any other");
    return bad;
}
