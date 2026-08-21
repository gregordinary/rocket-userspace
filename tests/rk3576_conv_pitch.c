// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_conv_pitch.c — the CNA reads a feature plane sitting inside WIDER rows, and
 * this is what it takes.
 *
 * The PPU was shown to honour a row pitch; the CNA was said not to, on the grounds that
 * its width field (`0x1078` high half, `iw-1`) and its line stride (`0x1090`, `iw*4`
 * words) are "one quantity in two units". That was a source read of our own emitter —
 * both are derived from `iw` because every capture has a tight plane — and it is wrong.
 * Three registers carry it, and `0x1090` is not one of them:
 *
 *   0x1044 low half   the DDR row ADVANCE, in 64-byte granules. `0x103C` stays at the
 *                     plane's count: that one is the FETCH LENGTH, and the CBUF is
 *                     filled contiguously with it, so the window walking the CBUF at
 *                     `iw` atoms a row lines up only while the two differ.
 *   0x1094            the channel-group jump, already a caller-supplied quantity.
 *   0x1078 low half   the DMA's row BUDGET, which is spent at the ADVANCE and not at
 *                     the fetch length — so a pitched walk needs it inflated to
 *                     `ceil(rows * entries_pitch / entries_plane)` or it runs out part
 *                     way down the plane. The window grid does NOT follow it: the
 *                     output extent and both derived pads come from the real geometry,
 *                     which is what lets a pitched task still synthesize the trailing
 *                     pad that the cheap route below reads real bytes for.
 *
 * `0x1090` advances the address by its own value once every FOUR rows while the rows
 * between step by the fetched width, so it cannot express a pitch at any value.
 *
 * Two geometry bounds ride with it, and one is silent. BOTH the pitch and the plane must
 * fill whole 64-byte DDR granules (`iw * ic * elem % 64`) — an odd plane at ic=32 is
 * 10.5 granules a row and computes wrong at every pitch. And a pitched task's ROW WINDOW
 * is the unpitched allowance scaled down by the granule ratio TWICE; past it the surface
 * is written and wrong, exactly like the unpitched allowance.
 *
 * The modes:
 *
 *   pitch    (default) the gate. Each geometry's plane is laid out inside rows of
 *            `iw + gap` elements with the surplus poisoned, and the output must be
 *            bit-identical to a reference over a tight cube.
 *            CONTROL   the same pitched buffer with the row stride left derived, which
 *                      must DIFFER — otherwise the register is not what carries the row
 *                      jump here and every cell above it is vacuous.
 *            COLUMN    the plane placed `d` elements into each row, the base advanced
 *                      `d` atoms: the feature base is a plain address at atom
 *                      granularity, as the PPU's source base is.
 *            WIDE      the route that needs no decode at all — program the PITCH as the
 *                      plane and let the output extent limit the windows. Reported, not
 *                      asserted, and scored BY REGION: it is exact wherever the window
 *                      grid stays inside the caller's plane and wrong on the TRAILING
 *                      columns wherever a trailing pad is derived, because widening the
 *                      plane lifts the clamp and a window that should read a synthesized
 *                      pad reads the producer's surplus columns instead.
 *
 *   map [wic [arm [iw ih pitch]]]   the readout that decoded all of the above. A 1x1
 *            convolution with two live weights copies a 16-bit position code from the
 *            feature buffer to the output, so the output at (y,x) NAMES the atom the
 *            part fetched and the addressing is an output rather than a pass/fail.
 *            `wic` 16 puts the live weight on channel group 1, which is what makes the
 *            group jump readable; a single-group map cannot see it.
 *
 * TRAP: THE ARMS OF THIS PROBE ARE NOT INDEPENDENT MEASUREMENTS. Two of them leave the
 * part writing nothing for every submit that follows, in this process and the next, and
 * a preceding arm can change a following one's map even when both emit byte-identical
 * streams. Run one arm per process — `map <wic> <index>` and `ROCKET_CP_GAP=<n>` — before
 * believing anything a sequence says.
 *
 * Usage: rk3576_conv_pitch [pitch | map [wic [arm [iw ih pitch]]]]
 * Exit:  0 the CNA honours a row pitch, 1 it does not, 2 no NPU or wrong chip.
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

/* One int8 direct convolution, everything but its feature layout frozen. */
struct prog {
    rocket_bo w, b, r;
    unsigned  ic, oc, ih, iw, kh, kw, sx, sy, pt, pl;
    unsigned  icreg, ocreg, ow, oh, surf_elems, gcube;
    size_t    surf_bytes;
    uint64_t  ops[MAX_ROWS * RK3576_CONV_TASK_OPS];
};

/* How the feature buffer is laid out for one arm. */
struct layout {
    unsigned pitch;    /* elements per row in DDR                        */
    unsigned col;      /* elements into each row where the plane starts  */
};

/* What the program is TOLD. `wide` programs the pitch as the plane. */
struct told {
    unsigned pitch;    /* 0 = leave 0x1090 derived from iw               */
    unsigned surf;     /* 0 = leave 0x1094 derived from the plane        */
    unsigned wide;     /* 1 = iw := pitch, ow unchanged                  */
    unsigned unit;     /* 1 = requant the identity: out_i8 = sat8(acc)   */
    unsigned task;     /* non-zero = patch 0x1098 to round4(pitch*rows)  */
};

/* Patch a register in an emitted task's op words. Returns the number of hits — zero
 * means the emitter never wrote that register and the arm asked nothing. */
static unsigned ops_patch(uint64_t *ops, unsigned n, uint16_t reg, uint32_t val)
{
    unsigned i, hits = 0;
    for (i = 0; i < n; i++) {
        if ((uint16_t)(ops[i] & 0xFFFFu) != reg) continue;
        ops[i] = (ops[i] & ~0xFFFFFFFF0000ull) | ((uint64_t)val << 16);
        hits++;
    }
    return hits;
}

static void bo_fill(int fd, rocket_bo *bo, size_t n, int byte)
{
    rocket_bo_prep(fd, bo, 1, 0);
    memset(bo->ptr, byte, n);
    rocket_bo_fini(fd, bo);
}

static int prog_init(int fd, struct prog *p, unsigned ic, unsigned oc,
                     unsigned iw, unsigned ih, unsigned k, unsigned s, unsigned pad)
{
    size_t w_bytes, coeff;
    int32_t *bias;
    unsigned i;

    memset(p, 0, sizeof *p);
    p->ic = ic; p->oc = oc; p->iw = iw; p->ih = ih;
    p->kh = p->kw = k; p->sx = p->sy = s;
    p->pt = p->pl = pad;
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
                        uint32_t in_dma, uint32_t out_dma, const struct told *t)
{
    memset(q, 0, sizeof *q);
    q->ic = (uint16_t)p->icreg; q->oc = (uint16_t)p->ocreg;
    q->ih = (uint16_t)p->ih;    q->iw = (uint16_t)(t->wide ? t->wide : p->iw);
    q->oh = (uint16_t)p->oh;    q->ow = (uint16_t)p->ow;
    q->kh = (uint16_t)p->kh;    q->kw = (uint16_t)p->kw;
    q->stride_y = (uint8_t)p->sy; q->stride_x = (uint8_t)p->sx;
    q->pad_top = (uint8_t)p->pt;  q->pad_left = (uint8_t)p->pl;
    q->ih_full = (uint16_t)p->ih; q->oh_full = (uint16_t)p->oh;
    q->in_surf_elems = t->surf;
    q->in_pitch_w    = t->pitch;
    q->int8_out = 1;
    /* out_i8 = sat8( round(acc * in*w/out) + (out_zp - 0x80) ), so `unit` is a plain
     * copy of the accumulator and the readout decodes straight to a byte. */
    q->in_scale = 1.0f; q->w_scale = 1.0f;
    q->out_scale = t->unit ? 1.0f : 256.0f;
    q->input_zero_point = 0; q->weight_zero_point = 0;
    q->output_zero_point = t->unit ? 0x80 : 0;
    q->input_dma   = in_dma;
    q->weights_dma = (uint32_t)p->w.dma_address;
    q->bias_dma    = (uint32_t)p->b.dma_address;
    q->output_dma  = out_dma;
}

/* Emit and submit the whole row plan. `row_pitch` is the element count a row task's
 * feature offset advances by — the buffer's pitch, not the plane's width. Returns the
 * task count, or -1. */
static int prog_run(int fd, struct prog *p, rocket_bo *in, size_t in_off,
                    rocket_bo *out, const struct told *t, unsigned row_pitch)
{
    rocket_rk3576_row_task plan[MAX_ROWS];
    rocket_task_desc td[MAX_ROWS];
    conv_params_t base, q;
    unsigned ntask = 1u, i;
    uint32_t task_ops = 0, in_h[4], out_h[1];
    size_t slot;

    prog_params(p, &base, (uint32_t)(in->dma_address + in_off),
                (uint32_t)out->dma_address, t);
    q = base;
    if (rocket_rk3576_plan_rows(&q, 0, plan, MAX_ROWS, &ntask) < 0) return -1;

    slot = RK3576_CONV_TASK_OPS;
    for (i = 0; i < ntask; i++) {
        q = base;
        q.ih = plan[i].ih; q.oh = plan[i].oh;
        q.pad_top = plan[i].pad_top;
        /* The row offset is a row COUNT times the buffer's pitch, not the plane's
         * width — the one place a pitched buffer changes the caller's arithmetic. */
        q.input_dma  = base.input_dma + (uint32_t)plan[i].iy0 * row_pitch * C2;
        q.output_dma = base.output_dma + plan[i].output_off;
        q.ih_full = (uint16_t)p->ih; q.oh_full = (uint16_t)p->oh;
        q.tasks = p->ops + (size_t)i * slot;
        q.task_count = 0;
        if (gen_conv2d_int8_rk3576(&q) != 0) return -1;
        /* The task WINDOW's group stride, which the emitter derives as
         * round4(iw*fetch_rows). Asked as its own arm because a pitched buffer is the
         * first layout where it could be a DDR quantity — the padded-group-stride
         * sweep that left it derived never moved the ROW layout. */
        if (t->task)
            ops_patch(q.tasks, q.task_count, 0x1098,
                      (t->task * plan[i].ih + 3u) & ~3u);
        if (!i) task_ops = q.task_count;
        else if (q.task_count != task_ops) return -1;
    }

    rocket_bo_prep(fd, &p->r, 1, 0);
    memcpy(p->r.ptr, p->ops, (size_t)ntask * slot * sizeof(uint64_t));
    rocket_bo_fini(fd, &p->r);
    for (i = 0; i < ntask; i++) {
        td[i].regcmd = (uint32_t)(p->r.dma_address + i * slot * sizeof(uint64_t));
        td[i].regcmd_count = task_ops;
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

/* The feature byte at group g, PLANE column x, row y, lane c. Plane-relative, so the
 * same data lands in a tight buffer and in a pitched one. */
static int8_t feat_byte(unsigned g, unsigned y, unsigned x, unsigned c)
{
    unsigned v = g * 37u + y * 11u + x * 5u + c * 3u;
    return (int8_t)((int)(v % 61u) - 30);
}

static void cube_fill(int fd, rocket_bo *bo, const struct prog *p,
                      const struct layout *L, size_t bytes)
{
    size_t S = (size_t)L->pitch * p->ih;   /* channel-group jump, in atoms */
    unsigned g, y, x, c;

    rocket_bo_prep(fd, bo, 1, 0);
    memset(bo->ptr, POISON, bytes);
    for (g = 0; g < p->gcube; g++)
        for (y = 0; y < p->ih; y++)
            for (x = 0; x < p->iw; x++)
                for (c = 0; c < C2; c++)
                    ((int8_t *)bo->ptr)[(g * S + (size_t)y * L->pitch + L->col + x) * C2 + c]
                        = feat_byte(g, y, x, c);
    rocket_bo_fini(fd, bo);
}

/* Compare against the reference and describe WHERE it differs, in output columns —
 * a geometry answer is scored by region, not by total. */
static void diff_report(const unsigned char *got, const unsigned char *ref,
                        const struct prog *p, unsigned *nbad, unsigned *cmin,
                        unsigned *cmax)
{
    size_t i;
    unsigned og = (p->ocreg + C2 - 1u) / C2;

    *nbad = 0; *cmin = p->ow; *cmax = 0;
    for (i = 0; i < p->surf_bytes; i++) {
        unsigned e, col;
        if (got[i] == ref[i]) continue;
        e = (unsigned)((i / C2) % p->surf_elems);
        if (e >= p->ow * p->oh) continue;
        col = e % p->ow;
        (*nbad)++;
        if (col < *cmin) *cmin = col;
        if (col > *cmax) *cmax = col;
    }
    (void)og;
}

/* One geometry. Returns 0 pass, 1 fail, -1 could not ask. */
static int run_case(const char *tag, int fd, unsigned ic, unsigned oc,
                    unsigned iw, unsigned ih, unsigned k, unsigned s, unsigned pad,
                    const unsigned *gaps, unsigned ngaps, unsigned force_rows)
{
    struct prog p;
    rocket_bo tight, wide, out;
    unsigned char *ref = NULL, *got = NULL;
    struct layout L;
    struct told T;
    size_t tight_bytes, wide_bytes;
    unsigned maxgap = 0, mult = 1u, i, j, nbad, cmin, cmax;
    int only_gap = -1;
    int bad = 0, ntask;
    char buf[32];

    if (force_rows) {
        snprintf(buf, sizeof buf, "%u", force_rows);
        setenv("ROCKET_RK3576_MAX_ROWS", buf, 1);
    } else {
        unsetenv("ROCKET_RK3576_MAX_ROWS");
    }

    if (prog_init(fd, &p, ic, oc, iw, ih, k, s, pad) != 0) {
        printf("   allocation/pack failed\n");
        unsetenv("ROCKET_RK3576_MAX_ROWS");
        return -1;
    }
    for (i = 0; i < ngaps; i++) if (gaps[i] > maxgap) maxgap = gaps[i];

    tight_bytes = (size_t)p.gcube * p.iw * p.ih * C2;
    /* One row of slack: an arm whose base is d atoms in fetches d atoms past the last
     * row's own extent. */
    wide_bytes  = (size_t)p.gcube * (p.iw + maxgap) * (p.ih + 1u) * C2;
    if (rocket_bo_alloc(fd, tight_bytes, &tight) < 0 ||
        rocket_bo_alloc(fd, wide_bytes, &wide) < 0 ||
        rocket_bo_alloc(fd, p.surf_bytes, &out) < 0) {
        printf("   allocation failed\n");
        unsetenv("ROCKET_RK3576_MAX_ROWS");
        return -1;
    }
    ref = malloc(p.surf_bytes);
    got = malloc(p.surf_bytes);
    if (!ref || !got) { bad = -1; goto done; }

    /* ---- the reference: a tight plane, everything derived ---- */
    L.pitch = p.iw; L.col = 0;
    memset(&T, 0, sizeof T);
    cube_fill(fd, &tight, &p, &L, tight_bytes);
    bo_fill(fd, &out, p.surf_bytes, POISON);
    ntask = prog_run(fd, &p, &tight, 0, &out, &T, p.iw);
    if (ntask < 0) { printf("   the reference run failed\n"); bad = -1; goto done; }
    rocket_bo_prep(fd, &out, 0, 2000000000ull);
    memcpy(ref, out.ptr, p.surf_bytes);
    rocket_bo_fini(fd, &out);
    for (i = 0; i < p.surf_bytes; i++) if (ref[i] != POISON) break;
    if (i == p.surf_bytes) {
        printf("   the reference surface is entirely poison — nothing wrote, so no "
               "comparison below means anything\n");
        bad = -1; goto done;
    }

    printf("\n   [%s] ic=%u oc=%u %ux%u k%u s%u pad%u -> %ux%u: %u feature group(s), "
           "%u row task(s)\n", tag, ic, oc, iw, ih, k, s, pad, p.ow, p.oh, p.gcube,
           (unsigned)ntask);

    /* The row stride is a 64-byte GRANULE count, so a pitch is expressible only at a
     * multiple of 64/gcd(ic,64) elements — 1 at ic 64 and above, 2 at ic 32, 4 at
     * ic 16. Each requested gap is rounded up to the nearest one. */
    {
        unsigned g = p.icreg % 64u ? p.icreg : 64u, b = 64u;
        while (g) { unsigned t = b % g; b = g; g = t; }
        mult = 64u / b;
    }

    /* ---- PITCH: the row stride told, the plane at column 0 ----
     * ROCKET_CP_GAP runs ONE gap and nothing else, so an arm's answer is its own and
     * not its predecessor's: this part's CNA carries state between submits and a
     * sequence of RE arms is not a sequence of independent measurements. */
    {
        const char *e = getenv("ROCKET_CP_GAP");
        if (e && *e) { only_gap = (int)strtol(e, NULL, 0); }
    }
    for (i = 0; i < ngaps; i++) {
        if (only_gap >= 0 && (unsigned)only_gap != i) continue;
        L.pitch = ((p.iw + gaps[i] + mult - 1u) / mult) * mult; L.col = 0;
        cube_fill(fd, &wide, &p, &L, wide_bytes);
        /* Two arms per gap: the row stride alone, and the row stride with the task
         * window's own group jump moved with it. */
        for (j = 0; j < 2u; j++) {
            T.pitch = L.pitch; T.surf = L.pitch * p.ih; T.wide = 0;
            T.task = j ? L.pitch : 0u;
            bo_fill(fd, &out, p.surf_bytes, POISON);
            if (prog_run(fd, &p, &wide, 0, &out, &T, L.pitch) < 0) {
                printf("   PITCH%s +%-3u (%4u wide)  the run failed\n",
                       j ? "+T" : "  ", L.pitch - p.iw, L.pitch);
                bad = 1; continue;
            }
            rocket_bo_prep(fd, &out, 0, 2000000000ull);
            memcpy(got, out.ptr, p.surf_bytes);
            rocket_bo_fini(fd, &out);
            diff_report(got, ref, &p, &nbad, &cmin, &cmax);
            printf("   PITCH%s +%-3u (%4u wide)  %s\n", j ? "+T" : "  ",
                   L.pitch - p.iw, L.pitch,
                   nbad ? "DIFFERS from the tight-plane reference" : "== reference");
            if (nbad) { printf("        %u element(s) differ, output columns %u..%u\n",
                               nbad, cmin, cmax); if (j) bad = 1; }
        }
    }

    /* ---- THE CONTROL: the same pitched buffer, the row stride left derived ---- */
    if (only_gap < 0) {
        L.pitch = ((p.iw + gaps[0] + mult - 1u) / mult) * mult; L.col = 0;
        T.pitch = 0; T.surf = L.pitch * p.ih; T.wide = 0;
        cube_fill(fd, &wide, &p, &L, wide_bytes);
        bo_fill(fd, &out, p.surf_bytes, POISON);
        if (prog_run(fd, &p, &wide, 0, &out, &T, L.pitch) < 0) {
            printf("   CONTROL +%-3u              the run failed\n", gaps[0]);
            bad = 1;
        } else {
            rocket_bo_prep(fd, &out, 0, 2000000000ull);
            memcpy(got, out.ptr, p.surf_bytes);
            rocket_bo_fini(fd, &out);
            diff_report(got, ref, &p, &nbad, &cmin, &cmax);
            printf("   CONTROL +%-3u              row stride not told: %s\n", gaps[0],
                   nbad ? "DIFFERS, as it must"
                        : "== reference, so 0x1090 IS NOT the row jump here and every "
                          "PITCH cell above means nothing");
            if (!nbad) bad = 1;
        }
    }

    /* ---- COLUMN: the plane part way into the row, the base advanced d atoms ---- */
    if (only_gap < 0 && gaps[ngaps - 1u] >= 4u) {
        static const unsigned DS[] = { 1u, 3u, 4u };
        unsigned j;
        L.pitch = ((p.iw + gaps[ngaps - 1u] + mult - 1u) / mult) * mult;
        for (j = 0; j < sizeof DS / sizeof DS[0]; j++) {
            if (DS[j] > gaps[ngaps - 1u]) continue;
            L.col = DS[j];
            T.pitch = L.pitch; T.surf = L.pitch * p.ih; T.wide = 0;
            cube_fill(fd, &wide, &p, &L, wide_bytes);
            bo_fill(fd, &out, p.surf_bytes, POISON);
            if (prog_run(fd, &p, &wide, (size_t)DS[j] * C2, &out, &T, L.pitch) < 0) {
                printf("   COLUMN  d=%-2u              the run failed\n", DS[j]);
                bad = 1; continue;
            }
            rocket_bo_prep(fd, &out, 0, 2000000000ull);
            memcpy(got, out.ptr, p.surf_bytes);
            rocket_bo_fini(fd, &out);
            diff_report(got, ref, &p, &nbad, &cmin, &cmax);
            printf("   COLUMN  d=%-2u (%4u wide)  %s\n", DS[j], L.pitch,
                   nbad ? "DIFFERS from the tight-plane reference" : "== reference");
            if (nbad) { printf("        %u element(s) differ, output columns %u..%u\n",
                               nbad, cmin, cmax); bad = 1; }
        }
    }

    /* ---- WIDE: the pitch programmed AS the plane, the extent limiting the windows.
     * Reported, not asserted: the prediction is that it is exact where the window grid
     * stays inside the caller's plane and wrong on the trailing columns where a
     * trailing pad is derived. */
    if (only_gap < 0) {
        unsigned reach = (p.ow - 1u) * s + k;
        unsigned overhang = reach > p.pl + p.iw ? reach - (p.pl + p.iw) : 0u;

        L.pitch = ((p.iw + gaps[ngaps - 1u] + mult - 1u) / mult) * mult; L.col = 0;
        T.pitch = 0; T.surf = 0; T.wide = L.pitch;
        cube_fill(fd, &wide, &p, &L, wide_bytes);
        bo_fill(fd, &out, p.surf_bytes, POISON);
        if (prog_run(fd, &p, &wide, 0, &out, &T, L.pitch) < 0) {
            printf("   WIDE    (%4u as plane)   the run failed\n", L.pitch);
        } else {
            rocket_bo_prep(fd, &out, 0, 2000000000ull);
            memcpy(got, out.ptr, p.surf_bytes);
            rocket_bo_fini(fd, &out);
            diff_report(got, ref, &p, &nbad, &cmin, &cmax);
            if (!nbad)
                printf("   WIDE    (%4u as plane)   == reference   (this geometry "
                       "derives no trailing pad: reach %u <= %u)\n",
                       L.pitch, reach, p.pl + p.iw);
            else
                printf("   WIDE    (%4u as plane)   DIFFERS: %u element(s), output "
                       "columns %u..%u of %u  (trailing pad %u would have been "
                       "synthesized)\n", L.pitch, nbad, cmin, cmax, p.ow, overhang);
        }
    }

done:
    free(ref); free(got);
    rocket_bo_free(fd, &tight); rocket_bo_free(fd, &wide); rocket_bo_free(fd, &out);
    rocket_bo_free(fd, &p.w); rocket_bo_free(fd, &p.b); rocket_bo_free(fd, &p.r);
    unsetenv("ROCKET_RK3576_MAX_ROWS");
    return bad;
}

/* ============================================================================
 * THE READOUT — what row stride did the part ACTUALLY use?
 *
 * A pass/fail sweep says a candidate register did not work; it cannot say what the
 * DMA did instead, and it cannot see a condition of two registers. This mode makes
 * the addressing an OUTPUT.
 *
 * A 1x1 convolution whose only live weight is w[0][0] = 1, at unit requant, copies
 * input channel 0 to output channel 0. Fill input channel 0 of EVERY atom in the
 * buffer — the plane's and the surplus — with its own linear position, and the
 * output at (y, x) names the buffer atom the part fetched for it. The implied row
 * stride falls straight out of the map.
 *
 * Arms are register OVERRIDES applied to the emitted stream, so any joint condition
 * can be asked. The one that must SUCCEED is the last: every `iw`-derived register
 * moved together, which is the WIDE arm and is known to fetch correctly.
 * ==========================================================================*/

/* Small enough that TWO channel groups' position codes fit disjointly in one int8:
 * group 0 carries `pos`, group 1 carries `pos + 96`, both decodable. That is what
 * makes the group JUMP readable as well as the row stride — a map that populates
 * only group 0 cannot see it, which is exactly how the row stride came out right
 * while the sweep was still wrong. */


struct ovr { uint16_t reg; uint32_t val; };

struct arm {
    const char *name;
    unsigned    wide;      /* re-emit at iw := pitch instead of overriding */
    unsigned    emit;      /* re-emit through the LIBRARY's own in_pitch_w  */
    struct ovr  o[6];
};

/* The pitch the `wide` arms re-emit at; set once per map_mode() call. */
static unsigned map_pitch;



static int map_run(int fd, struct prog *p, rocket_bo *in, rocket_bo *out,
                   const struct arm *a, unsigned *miss)
{
    conv_params_t q;
    struct told T;
    uint32_t in_h[4], out_h[1];
    unsigned i;

    memset(&T, 0, sizeof T);
    T.unit = 1;
    T.wide  = a->wide ? map_pitch : 0u;
    T.pitch = a->emit ? map_pitch : 0u;
    T.surf  = a->emit ? map_pitch * p->ih : 0u;
    prog_params(p, &q, (uint32_t)in->dma_address, (uint32_t)out->dma_address, &T);
    q.tasks = p->ops;
    q.task_count = 0;
    if (gen_conv2d_int8_rk3576(&q) != 0) return -1;

    *miss = 0;
    for (i = 0; i < sizeof a->o / sizeof a->o[0]; i++) {
        if (!a->o[i].reg) break;
        if (!ops_patch(p->ops, q.task_count, a->o[i].reg, a->o[i].val)) (*miss)++;
    }

    rocket_bo_prep(fd, &p->r, 1, 0);
    memcpy(p->r.ptr, p->ops, (size_t)q.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &p->r);

    in_h[0] = in->handle; in_h[1] = p->w.handle;
    in_h[2] = p->b.handle; in_h[3] = p->r.handle;
    out_h[0] = out->handle;
    return rocket_submit_matmul(fd, &p->r, q.task_count, in_h, 4, out_h, 1, 2000);
}

static int map_mode(int fd, unsigned wic, int only, unsigned iw, unsigned ih,
                    unsigned pitch)
{
    /* Every register the emitter derives from `iw` on the direct int8 path, plus the
     * joint conditions around the two that move something. The row jump is one of
     * these or is not programmable at all. */
    unsigned ent_iw = (iw * 32u + 63u) / 64u, ent_p = (pitch * 32u + 63u) / 64u;
    const struct arm ARMS[] = {
      { "nothing told (the control: the tight stride)", 0, 0, {{0,0}} },
      { "the LIBRARY's own pitched program (in_pitch_w + in_surf_elems)", 0, 1,
        {{0,0}} },
      { "the whole plane re-emitted at iw := pitch (must succeed)", 1, 0, {{0,0}} },
      { "0x1090 line stride = pitch*4", 0, 0, {{0x1090, pitch * 4u}, {0,0}} },
      { "0x1094 group stride = pitch*ih", 0, 0, {{0x1094, pitch * ih}, {0,0}} },
      { "0x1044 HIGH half = pitch (its low half left at the fetch)", 0, 0,
        {{0x1044, (pitch << 16) | ent_iw}, {0,0}} },
      { "0x1044 LOW half = the pitch's granule count (high left at iw)", 0, 0,
        {{0x1044, (iw << 16) | ent_p}, {0,0}} },
      { "0x103C (the CBUF row) = the pitch's granule count", 0, 0,
        {{0x103C, ent_p << 16}, {0,0}} },
      { "0x1028 (the CBUF footprint) at the pitch", 0, 0,
        {{0x1028, ((ent_p * ih) << 16) | 31u}, {0,0}} },
      { "JOINT 0x1044 low + 0x1094  (row stride AND group jump)", 0, 0,
        {{0x1044, (iw << 16) | ent_p}, {0x1094, pitch * ih}, {0,0}} },
      { "JOINT 0x1044 low + 0x1094 + its high half", 0, 0,
        {{0x1044, (pitch << 16) | ent_p}, {0x1094, pitch * ih}, {0,0}} },
      { "JOINT 0x1044 low + 0x1094 + 0x103C + 0x1028 (the whole CBUF set)", 0, 0,
        {{0x1044, (iw << 16) | ent_p}, {0x1094, pitch * ih},
         {0x103C, ent_p << 16}, {0x1028, ((ent_p * ih) << 16) | 31u}, {0,0}} },
      { "JOINT 0x1044 low + 0x1094 + 0x1090", 0, 0,
        {{0x1044, (iw << 16) | ent_p}, {0x1094, pitch * ih},
         {0x1090, pitch * 4u}, {0,0}} },
      /* The TOTAL the task fetches, in granules. The pitched walk runs out of it at
       * exactly the row where `ent_iw * ih / ent_p` lands, which is what makes the
       * map depart mid-plane with everything else told correctly. */
      { "JOINT 0x1044 low + 0x1094 + 0x1028 (the fetch TOTAL)", 0, 0,
        {{0x1044, (iw << 16) | ent_p}, {0x1094, pitch * ih},
         {0x1028, ((ent_p * ih) << 16) | 31u}, {0,0}} },
      { "JOINT that, plus 0x103C (the CBUF row) at the pitch", 0, 0,
        {{0x1044, (iw << 16) | ent_p}, {0x1094, pitch * ih},
         {0x1028, ((ent_p * ih) << 16) | 31u},
         {0x103C, ent_p << 16}, {0,0}} },
      { "JOINT 0x1044 low + 0x1094 + 0x1098 (both surface strides)", 0, 0,
        {{0x1044, (iw << 16) | ent_p}, {0x1094, pitch * ih},
         {0x1098, (pitch * ih + 3u) & ~3u}, {0,0}} },
      /* The budget the pitched walk runs out of is `ent_iw * ih` GRANULES, consumed at
       * the ADVANCE — the map departs at exactly the atom where that total is spent, at
       * two geometries. These ask which register carries it. */
      { "JOINT 0x1044 low + 0x1094 + 0x1028 at FOUR times the total", 0, 0,
        {{0x1044, (iw << 16) | ent_p}, {0x1094, pitch * ih},
         {0x1028, ((4u * ent_iw * ih) << 16) | 31u}, {0,0}} },
      { "JOINT 0x1044 low + 0x1094 + 0x1078 rows inflated to pitch*ih/iw", 0, 0,
        {{0x1044, (iw << 16) | ent_p}, {0x1094, pitch * ih},
         {0x1078, ((iw - 1u) << 16) | ((ih * ent_p + ent_iw - 1u) / ent_iw - 1u)},
         {0,0}} },
      { "JOINT that, and 0x1028 raised with the rows", 0, 0,
        {{0x1044, (iw << 16) | ent_p}, {0x1094, pitch * ih},
         {0x1078, ((iw - 1u) << 16) | ((ih * ent_p + ent_iw - 1u) / ent_iw - 1u)},
         {0x1028, ((ent_p * ih) << 16) | 31u}, {0,0}} },
      /* LAST, because these are destructive: an extent or a fetch wider than the CBUF
       * row leaves the part writing nothing for every submit that follows, in this
       * process and the next. */
      { "0x118C both halves = pitch-1  (DESTRUCTIVE, run last)", 0, 0,
        {{0x118C, ((pitch - 1u) << 16) | (pitch - 1u)}, {0,0}} },
      { "0x1078 width half = pitch-1  (DESTRUCTIVE, run last)", 0, 0,
        {{0x1078, ((pitch - 1u) << 16) | (ih - 1u)}, {0,0}} },
    };
    struct prog p;
    rocket_bo in, out;
    size_t in_bytes;
    unsigned a, y, x, miss;
    int bad = 0;

    if (pitch * ih > 4096u) {
        printf("   the buffer needs %u positions and the code carries 4096\n",
               pitch * ih);
        return -1;
    }
    if ((pitch * 32u) % 64u) {
        printf("   a %u-wide row at ic=32 is not a whole 64-byte DDR granule\n", pitch);
        return -1;
    }
    map_pitch = pitch;
    if (prog_init(fd, &p, 32u, 32u, iw, ih, 1u, 1u, 0u) != 0) {
        printf("   allocation/pack failed\n");
        return -1;
    }
    /* TWO live weights: w[oc 0][ic wic] and w[oc 1][ic wic+1], so output channels 0
     * and 1 carry the low and high byte of a 16-bit position code. A one-byte code
     * caps the map at 256 atoms, which is smaller than every geometry the sweep asks
     * about — and the geometry is exactly what the two disagree on. */
    rocket_bo_prep(fd, &p.w, 1, 0);
    memset(p.w.ptr, 0, (size_t)((p.ocreg + 31u) / 32u) * ((p.icreg + 31u) / 32u) *
                       32u * 32u);
    ((int8_t *)p.w.ptr)[(wic % 32u)] = 1;                 /* oc 0 <- ic wic     */
    ((int8_t *)p.w.ptr)[32u + (wic % 32u) + 1u] = 1;      /* oc 1 <- ic wic+1   */
    rocket_bo_fini(fd, &p.w);
    {
        size_t coeff = rocket_rk3576_coeff_bytes(p.ocreg);
        int32_t *z = calloc(p.ocreg, sizeof *z);
        if (!z) return -1;
        rocket_bo_prep(fd, &p.b, 1, 0);
        rocket_rk3576_pack_coeff_prec(p.b.ptr, coeff, z, p.ocreg, precision_int8);
        rocket_bo_fini(fd, &p.b);
        free(z);
    }

    in_bytes = (size_t)p.gcube * pitch * (ih + 1u) * C2;
    if (rocket_bo_alloc(fd, in_bytes, &in) < 0 ||
        rocket_bo_alloc(fd, p.surf_bytes, &out) < 0) {
        printf("   allocation failed\n");
        return -1;
    }
    /* Lanes `wic` and `wic+1` of every atom carry its own position in the buffer,
     * plane and surplus alike, with the channel GROUP in the code's high nibble. */
    rocket_bo_prep(fd, &in, 1, 0);
    memset(in.ptr, 0, in_bytes);
    {
        unsigned g, l0 = wic % C2, l1 = l0 + 1u;
        for (g = 0; g < p.gcube; g++)
            for (y = 0; y < ih; y++)
                for (x = 0; x < pitch; x++) {
                    size_t at = ((size_t)g * pitch * ih + y * pitch + x) * C2;
                    unsigned code = g * 4096u + y * pitch + x;
                    ((int8_t *)in.ptr)[at + l0] = (int8_t)((int)(code & 0xFFu) - 128);
                    ((int8_t *)in.ptr)[at + l1] = (int8_t)((int)(code >> 8) - 128);
                }
    }
    rocket_bo_fini(fd, &in);

    printf("== what feature address does the CNA actually use? ==\n");
    printf("   a %ux%u plane inside %u-wide rows, %u channel groups; the live weights "
           "are on input channels %u/%u (group %u)\n", iw, ih, pitch, p.gcube,
           wic, wic + 1u, wic / C2);
    printf("   each cell below is  g:pos  — the channel group and the position inside "
           "it that the part fetched\n");
    printf("   the tight plane reads row stride %u; the pitched buffer wants %u, and "
           "the group it must read is %u\n\n", iw, pitch, wic / C2);

    for (a = 0; a < sizeof ARMS / sizeof ARMS[0]; a++) {
        int8_t *o;
        int rstep, fits = 1;
        unsigned try, wrote = 0, first = 0;

        if (only >= 0 && (unsigned)only != a) continue;
        /* A program that writes NOTHING is not a map, and the previous arm can be why:
         * this part poisons the next submit after some programs. Retry once so a
         * silent surface is the arm's own answer rather than its predecessor's. */
        for (try = 0; try < 2u && !wrote; try++) {
            unsigned j;
            bo_fill(fd, &out, p.surf_bytes, POISON);
            if (map_run(fd, &p, &in, &out, &ARMS[a], &miss) != 0) {
                printf("   %-56s the run failed\n", ARMS[a].name);
                bad = 1; break;
            }
            rocket_bo_prep(fd, &out, 0, 2000000000ull);
            for (j = 0; j < p.surf_bytes; j++)
                if (((unsigned char *)out.ptr)[j] != POISON) { wrote = 1; break; }
            if (!wrote) rocket_bo_fini(fd, &out);
        }
        if (!wrote) {
            printf("   %-56s wrote NOTHING (twice) — no map\n", ARMS[a].name);
            continue;
        }
        o = (int8_t *)out.ptr;
#define MAP_AT(yy, xx) ((unsigned)((int)o[((size_t)(yy) * p.ow + (xx)) * C2] + 128) | \
                        ((unsigned)((int)o[((size_t)(yy) * p.ow + (xx)) * C2 + 1] + 128) << 8))
        first = MAP_AT(0, 0);
        rstep = p.oh > 1u ? (int)MAP_AT(1, 0) - (int)first : 0;
        for (y = 0; y < p.oh && fits; y++)
            for (x = 0; x < p.ow; x++)
                if (MAP_AT(y, x) != first + y * (unsigned)rstep + x) { fits = 0; break; }
        rocket_bo_fini(fd, &out);

        printf("   %-56s row step %4d%s%s\n", ARMS[a].name, rstep,
               fits ? "  AFFINE" : "  NOT affine",
               miss ? "  [an override named a register the emitter never wrote]" : "");
        printf("        col 0 down the rows:");
        for (y = 0; y < p.oh && y < 10u; y++)
            printf("  %u:%-4u", MAP_AT(y, 0) >> 12, MAP_AT(y, 0) & 4095u);
        printf("\n        row 0 across:       ");
        for (x = 0; x < p.ow && x < 10u; x++)
            printf("  %u:%-4u", MAP_AT(0, x) >> 12, MAP_AT(0, x) & 4095u);
        printf("\n");
        if (!fits) {
            /* Where it first departs, which is the whole point of a readout. */
            unsigned yy, xx, shown = 0;
            printf("        first departures:   ");
            for (yy = 0; yy < p.oh && shown < 8u; yy++)
                for (xx = 0; xx < p.ow && shown < 8u; xx++)
                    if (MAP_AT(yy, xx) != first + yy * (unsigned)rstep + xx) {
                        printf("  (%u,%u)->%u:%u want %u", yy, xx,
                               MAP_AT(yy, xx) >> 12, MAP_AT(yy, xx) & 4095u,
                               (first + yy * (unsigned)rstep + xx) & 4095u);
                        shown++;
                    }
            printf("\n");
        }
#undef MAP_AT
    }

    rocket_bo_free(fd, &in); rocket_bo_free(fd, &out);
    rocket_bo_free(fd, &p.w); rocket_bo_free(fd, &p.b); rocket_bo_free(fd, &p.r);
    return bad;
}

int main(int argc, char **argv)
{
    static const unsigned GAPS[]  = { 1u, 4u, 16u, 64u };
    static const unsigned GAPS1[] = { 16u };
    int fd, bad = 0, rc;
    rocket_bo guard;
    const char *mode = argc > 1 ? argv[1] : "pitch";

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

    if (!strcmp(mode, "map")) {
        bad = map_mode(fd, argc > 2 ? (unsigned)strtoul(argv[2], NULL, 0) : 0u,
                       argc > 3 ? (int)strtol(argv[3], NULL, 0) : -1,
                       argc > 4 ? (unsigned)strtoul(argv[4], NULL, 0) : 8u,
                       argc > 5 ? (unsigned)strtoul(argv[5], NULL, 0) : 10u,
                       argc > 6 ? (unsigned)strtoul(argv[6], NULL, 0) : 12u) > 0;
        rocket_bo_free(fd, &guard);
        rocket_close(fd);
        return bad;
    }

    printf("== does the CNA read a feature plane sitting inside WIDER rows? ==\n");
    printf("-- 1x1, no pad: the window grid cannot overhang, so WIDE should hold too --\n");
    rc = run_case("1x1", fd, 32, 32, 28, 28, 1, 1, 0, GAPS, 4u, 0);  if (rc > 0) bad = 1;
    rc = run_case("1x1 g4", fd, 64, 64, 14, 14, 1, 1, 0, GAPS, 4u, 0); if (rc > 0) bad = 1;
    printf("\n-- 3x3 SAME: a trailing pad IS derived, which is where WIDE should fail --\n");
    rc = run_case("3x3", fd, 32, 32, 28, 28, 3, 1, 1, GAPS, 4u, 0);  if (rc > 0) bad = 1;
    rc = run_case("3x3 g4", fd, 64, 32, 16, 16, 3, 1, 1, GAPS, 4u, 0); if (rc > 0) bad = 1;
    printf("\n-- stride 2, and a non-square plane --\n");
    rc = run_case("3x3 s2", fd, 32, 64, 28, 28, 3, 2, 1, GAPS, 4u, 0); if (rc > 0) bad = 1;
    /* An ODD plane width at ic=32 is 10.5 granules a row and the emitter refuses a
     * pitch on it — the run failing here is the refusal, which is the answer. */
    rc = run_case("5x5 rect", fd, 32, 32, 22, 13, 5, 1, 2, GAPS, 4u, 0); if (rc > 0) bad = 1;
    printf("\n-- ResNet-18's own shape: a 112-wide plane inside the stem's 128-wide rows --\n");
    rc = run_case("112/128", fd, 64, 64, 112, 112, 3, 1, 1, GAPS1, 1u, 0); if (rc > 0) bad = 1;
    rc = run_case("112/128 r16", fd, 64, 64, 112, 112, 3, 1, 1, GAPS1, 1u, 16u);
    if (rc > 0) bad = 1;
    rc = run_case("112/128 r32", fd, 64, 64, 112, 112, 3, 1, 1, GAPS1, 1u, 32u);
    if (rc > 0) bad = 1;
    printf("\n-- split into row tasks: is 0x1098 read as a DDR quantity too? --\n");
    rc = run_case("rows", fd, 32, 32, 56, 56, 3, 1, 1, GAPS1, 1u, 8u); if (rc > 0) bad = 1;
    rc = run_case("rows g4", fd, 64, 32, 28, 28, 3, 1, 1, GAPS1, 1u, 4u); if (rc > 0) bad = 1;

    rocket_bo_free(fd, &guard);
    rocket_close(fd);
    printf("\n%s\n", bad
        ? "FAIL: the CNA does not read a feature plane inside wider rows"
        : "PASS: the CNA's DDR row stride is a caller-supplied quantity separate from "
          "the fetched extent, the feature base is a plain address at atom granularity, "
          "and the task window stride is not read as a DDR quantity");
    return bad;
}
