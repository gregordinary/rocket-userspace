// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_argb_f32out_probe.c — the packed-image (first-conv) fp16 program on the whole
 * output stage: does it compute on random operands, and where does a row window's
 * channel-group jump (DPU 0x40B8) have to point?
 *
 * Runs on an RK3576 board. rocket_conv2d_fp16_rk3576() runs a convolution of four or fewer
 * input channels as the packed-image first conv, on its own fp16 writer, which poisons the
 * next submit and so pays a power cycle per task. Under the whole output stage (the nine DPU
 * words gen_conv2d_fp16_rk3576_f32out() puts on the slice program) the first conv poisoned
 * nothing and wrote all 7200 fp32 words of a 15x15 by 32 surface, on ALL-ONES operands. Ones
 * cannot see a program that reads one surface twice and skips another, and that entry runs
 * one program per row window, which the stage has never been run at. This asks both.
 *
 * The program is gen_conv2d_fp16_rk3576() at ic <= 4, with the nine stage words patched in
 * place after emission (all nine must be found, as r76_whole_output_stage() requires). The
 * operands are random small integers held in fp16 (input [-30,30], weights [-8,8]), so every
 * fp32 partial sum is exact and the score is element for element against a CPU model. The
 * output is read through the plain fp32 cube map, word = 4*((c/4)*A + p) + c%4, A = ow*oh.
 * Rows are planned by rocket_rk3576_plan_rows_prec(PREC_FP16) exactly as the library entry
 * plans them; `rows` in the shape table forces a finer cap (ROCKET_RK3576_MAX_ROWS). Output
 * channels go 32 to a program, the first conv's measured bound, each 32-channel tile its own
 * job of one task per window.
 *
 * Arms (one per process):
 *   stage    windows write into the whole plane's cube at row offset oy0*ow*16;
 *            0x40B8 = 4*S - ow*oh_task, S = 0x401C = ow*oh_full as emitted. On one window
 *            this is 3S, the measured whole-plane value.
 *   stage3t  the same, with 0x40B8 = 3*ow*oh_task (the stage helper applied to a window).
 *   own      each window its own convolution with its own surface (oh_full = oh_task),
 *            0x40B8 = 3*ow*oh_task, the windows' surfaces laid end to end.
 *   time     fc224_k3s2_p1 only: a call on the stage (image and weights packed, stamp, one job,
 *            read, narrow to fp16) against rocket_conv2d_fp16_rk3576() on the same operands,
 *            whose fp16 writer power-cycles per task; medians, and the two outputs compared.
 *
 * After every shape a CANARY runs: an int8 direct conv (ic 32, oc 32, 12x10, k3) scored
 * against its own CPU model. A canary that comes back wrong is a poisoned part.
 *
 *   rk3576_argb_f32out_probe <arm> [substr]
 *
 * WHAT IT CANNOT SEE: more than 32 output channels in one program, an asymmetric pad, a
 * VALID first conv, a chained stream, fp16 values past the exact-integer range, and the
 * library entry itself. tests/rk3576_fc_stage_gate.c gates the entry, which now takes this
 * form by default.
 *
 * Exit: 0 every shape exact with its canary, 1 otherwise, 2 no NPU or not RK3576.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"
#include "requant_model.h"

int feature_index(int C, int H, int W, int C2_, int c, int h, int w);
int conv_weight_index_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

#define STAMP    0xA5u
#define OC_TILE  32u
#define MAXWIN   256u

typedef struct {
    const char *name;
    unsigned ic, oc, iw, ih, k, stride, rows;
    int pad;           /* -1: TFLite SAME (lead pad, trailing implied); else symmetric */
} fshape_t;

static const fshape_t SHAPES[] = {
    { "ctl16_ic4",     4, 32,  16,  16, 3, 1, 0, -1 },
    { "rgb32_k3",      3, 32,  32,  32, 3, 1, 0, -1 },
    { "rgb32_k7",      3, 32,  32,  32, 7, 1, 0, -1 },
    { "rgb32_s2",      3, 32,  32,  32, 3, 2, 0, -1 },
    { "gray32_oc16",   1, 16,  32,  32, 3, 1, 0, -1 },
    { "rgb32_oc24",    3, 24,  32,  32, 3, 1, 0, -1 },
    { "rgb32_k3_r8",   3, 32,  32,  32, 3, 1, 8, -1 },
    { "rgb32_s2_r3",   3, 32,  32,  32, 3, 2, 3, -1 },
    { "rgb48_k5_r7",   3, 32,  48,  36, 5, 1, 7, -1 },
    { "fc224_k3s2",    3, 32, 224, 224, 3, 2, 0, -1 },
    { "fc224_k7s2",    3, 32, 224, 224, 7, 2, 0, -1 },
    { "fc224_k3s2_oc64", 3, 64, 224, 224, 3, 2, 0, -1 },
    /* The library's descriptor is symmetric: pad 1 gives 112x112 here, the time arm's shape. */
    { "fc224_k3s2_p1", 3, 32, 224, 224, 3, 2, 0, 1 },
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof SHAPES[0]))

enum { ARM_STAGE, ARM_STAGE3T, ARM_OWN, ARM_TIME };
static const char *ARMS[] = { "stage", "stage3t", "own", "time" };

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}
static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* The nine stage words over an emitted program; returns how many were found. */
static unsigned stage_patch(uint64_t *ops, unsigned n, uint32_t v40b8)
{
    unsigned i, found = 0;
    for (i = 0; i < n; i++) {
        uint64_t op = ops[i];
        uint32_t v = (uint32_t)((op >> 16) & 0xFFFFFFFFu), nv;
        if ((op >> 48) == 0) continue;
        switch ((unsigned)(op & 0xFFFFu)) {
        case 0x4010: nv = 0xA0000002u; break;
        case 0x4030: nv = (v & 0xFFFF0000u) | 0x0310u; break;
        case 0x4038: nv = 0x00000053u; break;
        case 0x4044: nv = 0x00000002u; break;
        case 0x4050: nv = 0x00023333u; break;
        case 0x40AC: nv = 0u; break;
        case 0x40B0: nv = 1u; break;
        case 0x40B4: nv = 0u; break;
        case 0x40B8: nv = v40b8; break;
        default: continue;
        }
        ops[i] = (op & ~(0xFFFFFFFFull << 16)) | ((uint64_t)nv << 16);
        found++;
    }
    return found;
}

static uint32_t reg_value(const uint64_t *ops, unsigned n, unsigned reg)
{
    unsigned i;
    for (i = 0; i < n; i++)
        if ((ops[i] >> 48) && (unsigned)(ops[i] & 0xFFFFu) == reg)
            return (uint32_t)((ops[i] >> 16) & 0xFFFFFFFFu);
    return 0xFFFFFFFFu;
}

/* ---- the int8 canary ---- */
static int canary(int fd)
{
    const unsigned ic = 32, oc = 32, iw = 12, ih = 10, k = 3, pad = 1, C2 = 16;
    unsigned ow = iw, oh = ih, divisor = 1, scale, shift, c, i, y, x, kh, kw, seed = 0x1234567u;
    size_t in_bytes = (size_t)(ic / C2) * ih * iw * C2;
    size_t w_bytes = (size_t)(oc / 32) * (ic / 32) * 32 * 32 * k * k;
    unsigned surf = rocket_rk3576_out_surf_elems(ow, oh, 0);
    size_t obytes = (size_t)(oc / C2) * surf * C2, cb = rocket_rk3576_coeff_bytes(oc);
    int8_t in[32 * 10 * 12], w[32 * 32 * 9];
    int32_t bias[32];
    rocket_bo ib = {0}, wb = {0}, bb = {0}, ob = {0}, rb = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p = {0};
    int exact = -1;

    while ((double)divisor < 2.0 * sqrt((double)(ic * k * k))) divisor *= 2;
    requant_params(1.0f / (float)divisor, &scale, &shift);
    for (i = 0; i < sizeof in; i++) { seed = seed * 1103515245u + 12345u; in[i] = (int8_t)((int)((seed >> 16) % 61u) - 30); }
    for (i = 0; i < sizeof w; i++) { seed = seed * 1103515245u + 12345u; w[i] = (int8_t)((int)((seed >> 16) % 17u) - 8); }
    for (c = 0; c < oc; c++) bias[c] = ((int)c - 16) * 8;
    if (rocket_bo_alloc(fd, in_bytes, &ib) < 0 || rocket_bo_alloc(fd, w_bytes, &wb) < 0 ||
        rocket_bo_alloc(fd, cb, &bb) < 0 || rocket_bo_alloc(fd, obytes, &ob) < 0 ||
        rocket_bo_alloc(fd, sizeof ops, &rb) < 0)
        goto out;
    rocket_bo_prep(fd, &ib, 1, 0);
    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                ((int8_t *)ib.ptr)[feature_index((int)ic, (int)ih, (int)iw, (int)C2, (int)c + 1,
                                                 (int)y + 1, (int)x + 1)] = in[(c * ih + y) * iw + x];
    rocket_bo_fini(fd, &ib);
    rocket_bo_prep(fd, &wb, 1, 0);
    memset(wb.ptr, 0, w_bytes);
    for (c = 0; c < oc; c++)
        for (i = 0; i < ic; i++)
            for (kh = 0; kh < k; kh++)
                for (kw = 0; kw < k; kw++)
                    ((int8_t *)wb.ptr)[conv_weight_index_int8((int)oc, (int)ic, (int)k, (int)k,
                                                              (int)c + 1, (int)i + 1, (int)kh + 1,
                                                              (int)kw + 1)] =
                        w[((c * ic + i) * k + kh) * k + kw];
    rocket_bo_fini(fd, &wb);
    rocket_bo_prep(fd, &bb, 1, 0);
    rocket_rk3576_pack_coeff(bb.ptr, cb, bias, oc);
    rocket_bo_fini(fd, &bb);
    rocket_bo_prep(fd, &ob, 1, 0); memset(ob.ptr, 0xAA, obytes); rocket_bo_fini(fd, &ob);
    p.ic = ic; p.oc = oc; p.ih = ih; p.iw = iw; p.oh = oh; p.ow = ow; p.kh = k; p.kw = k;
    p.stride_y = 1; p.stride_x = 1; p.pad_top = pad; p.pad_left = pad;
    p.ih_full = ih; p.oh_full = oh; p.int8_out = 1;
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = (float)divisor;
    p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
    p.input_iova = ib.dma_address; p.weights_iova = wb.dma_address;
    p.bias_iova = bb.dma_address; p.output_iova = ob.dma_address; p.tasks = ops;
    if (gen_conv2d_int8_rk3576(&p) != 0) goto out;
    rocket_bo_prep(fd, &rb, 1, 0); memcpy(rb.ptr, ops, p.task_count * 8u); rocket_bo_fini(fd, &rb);
    in_h[0] = ib.handle; in_h[1] = wb.handle; in_h[2] = bb.handle; in_h[3] = rb.handle;
    out_h[0] = ob.handle;
    if (rocket_submit_matmul(fd, &rb, p.task_count, in_h, 4, out_h, 1, 2000) != 0) goto out;
    rocket_bo_prep(fd, &ob, 0, 2000000000ull);
    exact = 0;
    for (c = 0; c < oc; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                int64_t acc = bias[c];
                for (kh = 0; kh < k; kh++)
                    for (kw = 0; kw < k; kw++) {
                        int iy = (int)(y + kh) - (int)pad, ix = (int)(x + kw) - (int)pad;
                        if (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw) continue;
                        for (i = 0; i < ic; i++)
                            acc += (int64_t)in[(i * ih + iy) * iw + ix] * w[((c * ic + i) * k + kh) * k + kw];
                    }
                if (((int8_t *)ob.ptr)[(size_t)(c / C2) * surf * C2 + (size_t)C2 * (y * ow + x) + c % C2]
                    == requant_apply(acc, scale, shift))
                    exact++;
            }
    rocket_bo_fini(fd, &ob);
out:
    rocket_bo_free(fd, &ib); rocket_bo_free(fd, &wb); rocket_bo_free(fd, &bb);
    rocket_bo_free(fd, &ob); rocket_bo_free(fd, &rb);
    return exact;   /* of 32*10*12 = 3840 */
}

/* ---- one shape ---- */
struct geo {
    unsigned ic, oc, iw, ih, k, s, pad, ow, oh, A, nwin;
    rocket_rk3576_row_task rows[MAXWIN];
};

static int plan(const fshape_t *S, struct geo *g)
{
    conv_params_t q = {0};
    memset(g, 0, sizeof *g);
    g->ic = S->ic; g->oc = S->oc; g->iw = S->iw; g->ih = S->ih; g->k = S->k; g->s = S->stride;
    if (S->pad < 0) {
        g->ow = (S->iw + S->stride - 1) / S->stride;
        g->oh = (S->ih + S->stride - 1) / S->stride;
        g->pad = ((g->ow - 1) * S->stride + S->k > S->iw) ? ((g->ow - 1) * S->stride + S->k - S->iw) / 2 : 0;
    } else {
        g->pad = (unsigned)S->pad;
        g->ow = (S->iw + 2u * g->pad - S->k) / S->stride + 1u;
        g->oh = (S->ih + 2u * g->pad - S->k) / S->stride + 1u;
    }
    g->A = g->ow * g->oh;
    q.ic = (uint16_t)S->ic; q.ih = (uint16_t)S->ih; q.iw = (uint16_t)S->iw;
    q.oc = (uint16_t)OC_TILE; q.oh = (uint16_t)g->oh; q.ow = (uint16_t)g->ow;
    q.kh = (uint16_t)S->k; q.kw = (uint16_t)S->k;
    q.stride_y = (uint8_t)S->stride; q.stride_x = (uint8_t)S->stride;
    q.pad_top = (uint8_t)g->pad; q.pad_left = (uint8_t)g->pad;
    q.ih_full = (uint16_t)S->ih; q.oh_full = (uint16_t)g->oh;
    if (S->rows) {
        char b[16]; snprintf(b, sizeof b, "%u", S->rows); setenv("ROCKET_RK3576_MAX_ROWS", b, 1);
    } else {
        unsetenv("ROCKET_RK3576_MAX_ROWS");
    }
    if (rocket_rk3576_plan_rows_prec(&q, 0, PREC_FP16, g->rows, MAXWIN, &g->nwin) < 0 || !g->nwin) {
        unsetenv("ROCKET_RK3576_MAX_ROWS");
        return -1;
    }
    unsetenv("ROCKET_RK3576_MAX_ROWS");
    return 0;
}

/* Emits every window of one 32-channel tile into `rbo` (one slot per window) and fills
 * `td`. Output placement per arm. Returns the window count, or -1. */
static int emit_tile(const struct geo *g, int arm, unsigned npad, uint32_t in_iova,
                     uint32_t w_iova, uint32_t b_iova, uint32_t out_iova, rocket_bo *rbo,
                     rocket_task_desc *td, int verbose)
{
    unsigned r, slot = RK3576_CONV_TASK_SLOT;
    size_t own_off = 0;
    uint64_t *base = (uint64_t *)rbo->ptr;
    for (r = 0; r < g->nwin; r++) {
        conv_params_t p = {0};
        uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
        uint32_t v40b8, s401c;
        unsigned found, ohf;
        p.ic = (uint16_t)g->ic; p.iw = (uint16_t)g->iw;
        p.ih = g->rows[r].ih; p.oh = g->rows[r].oh;
        p.oc = (uint16_t)npad; p.ow = (uint16_t)g->ow;
        p.kh = (uint16_t)g->k; p.kw = (uint16_t)g->k;
        p.stride_y = (uint8_t)g->s; p.stride_x = (uint8_t)g->s;
        p.pad_top = g->rows[r].pad_top; p.pad_left = (uint8_t)g->pad;
        p.ih_full = (uint16_t)g->ih;
        ohf = (arm == ARM_OWN) ? g->rows[r].oh : g->oh;
        p.oh_full = (uint16_t)ohf;
        p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;
        p.input_iova = in_iova + g->rows[r].feature_off;
        p.weights_iova = w_iova; p.bias_iova = b_iova;
        p.output_iova = (arm == ARM_OWN) ? out_iova + (uint32_t)own_off
                                         : out_iova + (uint32_t)((size_t)g->rows[r].oy0 * g->ow * 16u);
        p.tasks = ops;
        if (gen_conv2d_fp16_rk3576(&p) != 0) return -1;
        s401c = reg_value(ops, p.task_count, 0x401C);
        if (arm == ARM_STAGE) v40b8 = 4u * s401c - (uint32_t)g->ow * g->rows[r].oh;
        else                  v40b8 = 3u * (uint32_t)g->ow * g->rows[r].oh;
        found = stage_patch(ops, p.task_count, v40b8);
        if (found != 9u) { printf("  stage patch found %u of 9 words\n", found); return -1; }
        if (verbose)
            printf("    win %u: in rows %u+%u pad_top %u -> out rows %u+%u, 0x401C %u 0x40B8 %u\n",
                   r, g->rows[r].iy0, g->rows[r].ih, g->rows[r].pad_top, g->rows[r].oy0,
                   g->rows[r].oh, s401c, v40b8);
        memcpy(base + (size_t)r * slot, ops, p.task_count * sizeof(uint64_t));
        td[r].regcmd = (uint32_t)(rbo->dma_address + (size_t)r * slot * sizeof(uint64_t));
        td[r].regcmd_count = p.task_count;
        own_off += (size_t)((npad + 3u) / 4u) * g->ow * g->rows[r].oh * 16u;
    }
    return (int)g->nwin;
}

/* Where (channel c of the tile, pixel y,x) sits, in fp32 words. */
static size_t word_of(const struct geo *g, int arm, unsigned c, unsigned y, unsigned x)
{
    if (arm != ARM_OWN)
        return 4u * ((size_t)(c / 4u) * g->A + (size_t)y * g->ow + x) + c % 4u;
    {
        unsigned r = 0, quads = (OC_TILE + 3u) / 4u;
        size_t base = 0;
        while (r + 1u < g->nwin && y >= g->rows[r + 1].oy0) {
            base += (size_t)quads * g->ow * g->rows[r].oh * 4u;
            r++;
        }
        return base + 4u * ((size_t)(c / 4u) * g->ow * g->rows[r].oh +
                            (size_t)(y - g->rows[r].oy0) * g->ow + x) + c % 4u;
    }
}

struct bufs {
    rocket_bo in, w, b, o, r;
    size_t obytes, wbytes;
};

static int run_shape(int fd, const fshape_t *S, int arm, int verbose, int *n_ok)
{
    struct geo g;
    struct bufs B;
    _Float16 *in = NULL, *W = NULL, *wt = NULL;
    float *ref = NULL;
    unsigned c, y, x, i, kh, kw, oc0, seed;
    size_t in_bytes, cbytes;
    rocket_task_desc td[MAXWIN];
    uint32_t in_h[4], out_h[1];
    int rc = -1;
    unsigned long bad = 0, total = 0, unwritten = 0, beyond = 0;

    memset(&B, 0, sizeof B);
    if (plan(S, &g) < 0) { printf("  %-16s no row plan\n", S->name); return -1; }
    in  = malloc((size_t)g.ic * g.ih * g.iw * sizeof *in);
    W   = malloc((size_t)g.oc * g.ic * g.k * g.k * sizeof *W);
    wt  = calloc((size_t)OC_TILE * g.ic * g.k * g.k, sizeof *wt);
    ref = malloc((size_t)g.oc * g.oh * g.ow * sizeof *ref);
    if (!in || !W || !wt || !ref) goto done;
    seed = 0x9E3779B9u ^ (g.ic * 31u + g.oc * 17u + g.iw * 7u + g.ih * 3u + g.k);
    for (i = 0; i < g.ic * g.ih * g.iw; i++) {
        seed = seed * 1103515245u + 12345u; in[i] = (_Float16)((int)((seed >> 16) % 61u) - 30);
    }
    for (i = 0; i < g.oc * g.ic * g.k * g.k; i++) {
        seed = seed * 1103515245u + 12345u; W[i] = (_Float16)((int)((seed >> 16) % 17u) - 8);
    }
    for (c = 0; c < g.oc; c++)
        for (y = 0; y < g.oh; y++)
            for (x = 0; x < g.ow; x++) {
                float acc = 0.0f;
                for (kh = 0; kh < g.k; kh++)
                    for (kw = 0; kw < g.k; kw++) {
                        int iy = (int)(y * g.s + kh) - (int)g.pad, ix = (int)(x * g.s + kw) - (int)g.pad;
                        if (iy < 0 || iy >= (int)g.ih || ix < 0 || ix >= (int)g.iw) continue;
                        for (i = 0; i < g.ic; i++)
                            acc += (float)in[((size_t)i * g.ih + iy) * g.iw + ix] *
                                   (float)W[(((size_t)c * g.ic + i) * g.k + kh) * g.k + kw];
                    }
                ref[((size_t)c * g.oh + y) * g.ow + x] = acc;
            }

    in_bytes = (size_t)g.ih * g.iw * g.ic * sizeof(_Float16);
    B.wbytes = rocket_rk3576_weight_argb_fp16_bytes(OC_TILE, g.k, g.k);
    cbytes = rocket_rk3576_coeff_bytes(OC_TILE);
    B.obytes = 2u * (size_t)(OC_TILE / 4u) * g.A * 16u;    /* twice one tile's fp32 cube */
    if (rocket_bo_alloc32(fd, in_bytes, &B.in) < 0 || rocket_bo_alloc32(fd, B.wbytes, &B.w) < 0 ||
        rocket_bo_alloc32(fd, cbytes, &B.b) < 0 || rocket_bo_alloc32(fd, B.obytes, &B.o) < 0 ||
        rocket_bo_alloc32(fd, (size_t)g.nwin * RK3576_CONV_TASK_SLOT * 8u, &B.r) < 0)
        goto done;
    rocket_bo_prep(fd, &B.in, 1, 0);
    for (y = 0; y < g.ih; y++)
        for (x = 0; x < g.iw; x++)
            for (c = 0; c < g.ic; c++)
                ((_Float16 *)B.in.ptr)[((size_t)y * g.iw + x) * g.ic + c] = in[((size_t)c * g.ih + y) * g.iw + x];
    rocket_bo_fini(fd, &B.in);
    rocket_bo_prep(fd, &B.b, 1, 0);
    rocket_rk3576_pack_coeff_prec(B.b.ptr, cbytes, NULL, OC_TILE, PREC_FP16);
    rocket_bo_fini(fd, &B.b);
    in_h[0] = B.in.handle; in_h[1] = B.w.handle; in_h[2] = B.b.handle; in_h[3] = B.r.handle;
    out_h[0] = B.o.handle;

    for (oc0 = 0; oc0 < g.oc; oc0 += OC_TILE) {
        unsigned n = g.oc - oc0 < OC_TILE ? g.oc - oc0 : OC_TILE;
        unsigned npad = rocket_rk3576_fp16_pad_oc(n);
        int nw;
        memset(wt, 0, (size_t)OC_TILE * g.ic * g.k * g.k * sizeof *wt);
        memcpy(wt, W + (size_t)oc0 * g.ic * g.k * g.k, (size_t)n * g.ic * g.k * g.k * sizeof *wt);
        rocket_bo_prep(fd, &B.w, 1, 0);
        memset(B.w.ptr, 0, B.wbytes);
        if (rocket_rk3576_argb_fp16_pack_weights(B.w.ptr, B.wbytes, wt, npad, g.ic, g.k, g.k) < 0) {
            rocket_bo_fini(fd, &B.w); printf("  weight pack refused\n"); goto done;
        }
        rocket_bo_fini(fd, &B.w);
        rocket_bo_prep(fd, &B.r, 1, 0);
        nw = emit_tile(&g, arm, npad, (uint32_t)B.in.dma_address, (uint32_t)B.w.dma_address,
                       (uint32_t)B.b.dma_address, (uint32_t)B.o.dma_address, &B.r, td,
                       verbose && oc0 == 0);
        rocket_bo_fini(fd, &B.r);
        if (nw < 0) { printf("  %-16s the generator refused\n", S->name); goto done; }
        rocket_bo_prep(fd, &B.o, 1, 0); memset(B.o.ptr, STAMP, B.obytes); rocket_bo_fini(fd, &B.o);
        if (rocket_submit_tasks_flags(fd, td, (uint32_t)nw, in_h, 4, out_h, 1, 0) != 0) {
            printf("  %-16s submit failed\n", S->name); goto done;
        }
        rocket_bo_prep(fd, &B.o, 0, 4000000000ull);
        {
            const uint32_t *wd = (const uint32_t *)B.o.ptr;
            char *hit = calloc(B.obytes / 4u, 1);
            size_t i2, wmax = 0;
            for (c = 0; c < n; c++)
                for (y = 0; y < g.oh; y++)
                    for (x = 0; x < g.ow; x++) {
                        size_t wi = word_of(&g, arm, c, y, x);
                        float v, want = ref[((size_t)(oc0 + c) * g.oh + y) * g.ow + x];
                        memcpy(&v, &wd[wi], 4);
                        if (hit) hit[wi] = 1;
                        if (wi > wmax) wmax = wi;
                        total++;
                        if (wd[wi] == 0xA5A5A5A5u) unwritten++;
                        if (v != want) {
                            if (verbose && bad < 6)
                                printf("    mism c=%u y=%u x=%u want %g got %g (word %zu)\n",
                                       oc0 + c, y, x, (double)want, (double)v, wi);
                            bad++;
                        }
                    }
            /* Words written that no output element owns: past the map, or stray. */
            for (i2 = 0; hit && i2 < B.obytes / 4u; i2++)
                /* A partial tile's padding channels own words inside the map; past
                 * the map's last word, or anywhere on a whole tile, a write is stray. */
                if (!hit[i2] && wd[i2] != 0xA5A5A5A5u && (i2 > wmax || n == npad))
                    beyond++;
            free(hit);
        }
        rocket_bo_fini(fd, &B.o);
    }
    {
        int can = canary(fd);
        printf("  %-16s %ux%u ic%u oc%-3u k%u s%u -> %ux%u, %2u win | %8lu/%-8lu exact, unwritten %lu, "
               "stray %lu | canary %d/3840 %s\n", S->name, g.iw, g.ih, g.ic, g.oc, g.k, g.s,
               g.ow, g.oh, g.nwin, total - bad, total, unwritten, beyond, can,
               (!bad && can == 3840) ? "EXACT" : (can != 3840 ? "CANARY-WRONG" : "WRONG"));
        *n_ok = (!bad && can == 3840);
    }
    rc = 0;
done:
    rocket_bo_free(fd, &B.in); rocket_bo_free(fd, &B.w); rocket_bo_free(fd, &B.b);
    rocket_bo_free(fd, &B.o); rocket_bo_free(fd, &B.r);
    free(in); free(W); free(wt); free(ref);
    return rc;
}

/* ---- the time arm: fc224_k3s2, a call on the stage against the library's guarded entry ---- */
static int time_arm(int fd)
{
    const fshape_t *S = &SHAPES[N_SHAPES - 1];   /* fc224_k3s2_p1 */
    struct geo g;
    struct bufs B;
    _Float16 *in, *W, *out_lib, *out_st;
    rocket_task_desc td[MAXWIN];
    uint32_t in_h[4], out_h[1];
    double t_st[12], t_lib[6];
    unsigned i, c, y, x, seed = 0xC0FFEEu, rep;
    size_t in_bytes, cbytes;
    rocket_conv2d_desc d = {0};
    unsigned long diff = 0;

    memset(&B, 0, sizeof B);
    if (plan(S, &g) < 0) return -1;
    in = malloc((size_t)g.ic * g.ih * g.iw * 2); W = malloc((size_t)g.oc * g.ic * g.k * g.k * 2);
    out_lib = malloc((size_t)g.oc * g.A * 2); out_st = malloc((size_t)g.oc * g.A * 2);
    for (i = 0; i < g.ic * g.ih * g.iw; i++) { seed = seed * 1103515245u + 12345u; in[i] = (_Float16)((int)((seed >> 16) % 61u) - 30); }
    for (i = 0; i < g.oc * g.ic * g.k * g.k; i++) { seed = seed * 1103515245u + 12345u; W[i] = (_Float16)((int)((seed >> 16) % 17u) - 8); }
    in_bytes = (size_t)g.ih * g.iw * g.ic * 2;
    B.wbytes = rocket_rk3576_weight_argb_fp16_bytes(OC_TILE, g.k, g.k);
    cbytes = rocket_rk3576_coeff_bytes(OC_TILE);
    B.obytes = (size_t)(OC_TILE / 4u) * g.A * 16u;
    if (rocket_bo_alloc32(fd, in_bytes, &B.in) < 0 || rocket_bo_alloc32(fd, B.wbytes, &B.w) < 0 ||
        rocket_bo_alloc32(fd, cbytes, &B.b) < 0 || rocket_bo_alloc32(fd, B.obytes, &B.o) < 0 ||
        rocket_bo_alloc32(fd, (size_t)g.nwin * RK3576_CONV_TASK_SLOT * 8u, &B.r) < 0)
        return -1;
    rocket_bo_prep(fd, &B.b, 1, 0);
    rocket_rk3576_pack_coeff_prec(B.b.ptr, cbytes, NULL, OC_TILE, PREC_FP16);
    rocket_bo_fini(fd, &B.b);
    rocket_bo_prep(fd, &B.r, 1, 0);
    emit_tile(&g, ARM_STAGE, OC_TILE, (uint32_t)B.in.dma_address, (uint32_t)B.w.dma_address,
              (uint32_t)B.b.dma_address, (uint32_t)B.o.dma_address, &B.r, td, 0);
    rocket_bo_fini(fd, &B.r);
    in_h[0] = B.in.handle; in_h[1] = B.w.handle; in_h[2] = B.b.handle; in_h[3] = B.r.handle;
    out_h[0] = B.o.handle;

    for (rep = 0; rep < 12; rep++) {      /* a call: pack image + weights, stamp, one job, read */
        double t0 = now_ms();
        rocket_bo_prep(fd, &B.in, 1, 0);
        for (y = 0; y < g.ih; y++)
            for (x = 0; x < g.iw; x++)
                for (c = 0; c < g.ic; c++)
                    ((_Float16 *)B.in.ptr)[((size_t)y * g.iw + x) * g.ic + c] = in[((size_t)c * g.ih + y) * g.iw + x];
        rocket_bo_fini(fd, &B.in);
        rocket_bo_prep(fd, &B.w, 1, 0);
        rocket_rk3576_argb_fp16_pack_weights(B.w.ptr, B.wbytes, W, OC_TILE, g.ic, g.k, g.k);
        rocket_bo_fini(fd, &B.w);
        rocket_bo_prep(fd, &B.o, 1, 0); memset(B.o.ptr, STAMP, B.obytes); rocket_bo_fini(fd, &B.o);
        if (rocket_submit_tasks_flags(fd, td, g.nwin, in_h, 4, out_h, 1, 0) != 0) return -1;
        rocket_bo_prep(fd, &B.o, 0, 4000000000ull);
        for (c = 0; c < g.oc; c++)
            for (i = 0; i < g.A; i++) {
                float v;
                memcpy(&v, (const uint32_t *)B.o.ptr + 4u * ((size_t)(c / 4u) * g.A + i) + c % 4u, 4);
                out_st[(size_t)c * g.A + i] = (_Float16)v;
            }
        rocket_bo_fini(fd, &B.o);
        t_st[rep] = now_ms() - t0;
    }
    d.ic = (int)g.ic; d.oc = (int)g.oc; d.ih = (int)g.ih; d.iw = (int)g.iw;
    d.kh = (int)g.k; d.kw = (int)g.k; d.stride_y = (int)g.s; d.stride_x = (int)g.s;
    d.pad_top = (int)g.pad; d.pad_left = (int)g.pad; d.dil_y = 1; d.dil_x = 1;
    for (rep = 0; rep < 6; rep++) {
        double t0 = now_ms();
        int rc = rocket_conv2d_fp16_rk3576(fd, &d, in, W, out_lib);
        t_lib[rep] = now_ms() - t0;
        if (rc != ROCKET_OK) { printf("  library entry returned %d\n", rc); return -1; }
    }
    for (i = 0; i < g.oc * g.A; i++) diff += (*(uint16_t *)&out_lib[i] != *(uint16_t *)&out_st[i]);
    {   /* both outputs against the CPU model, narrowed to fp16 the same way */
        unsigned long bad_st = 0, bad_lib = 0, kh, kw, ii;
        for (c = 0; c < g.oc; c++)
            for (y = 0; y < g.oh; y++)
                for (x = 0; x < g.ow; x++) {
                    float acc = 0.0f;
                    _Float16 want;
                    for (kh = 0; kh < g.k; kh++)
                        for (kw = 0; kw < g.k; kw++) {
                            int iy = (int)(y * g.s + kh) - (int)g.pad, ix = (int)(x * g.s + kw) - (int)g.pad;
                            if (iy < 0 || iy >= (int)g.ih || ix < 0 || ix >= (int)g.iw) continue;
                            for (ii = 0; ii < g.ic; ii++)
                                acc += (float)in[((size_t)ii * g.ih + iy) * g.iw + ix] *
                                       (float)W[(((size_t)c * g.ic + ii) * g.k + kh) * g.k + kw];
                        }
                    want = (_Float16)acc;
                    bad_st  += out_st[(size_t)c * g.A + y * g.ow + x] != want;
                    bad_lib += out_lib[(size_t)c * g.A + y * g.ow + x] != want;
                }
        printf("  against the CPU model: stage path %lu wrong, library entry %lu wrong, of %u\n",
               bad_st, bad_lib, g.oc * g.A);
    }
    qsort(t_st + 2, 10, sizeof(double), cmp_d);
    qsort(t_lib + 1, 5, sizeof(double), cmp_d);
    printf("  fc224_k3s2_p1 (3->32, %u windows): stage call median %.2f ms (min %.2f max %.2f, 10 after 2)"
           " | library guarded entry median %.1f ms (min %.1f max %.1f, 5 after 1) | %.1fx | "
           "outputs differ at %lu of %u fp16 elements\n",
           g.nwin, t_st[2 + 5], t_st[2], t_st[11], t_lib[1 + 2], t_lib[1], t_lib[5],
           t_lib[1 + 2] / t_st[2 + 5], diff, g.oc * g.A);
    rocket_bo_free(fd, &B.in); rocket_bo_free(fd, &B.w); rocket_bo_free(fd, &B.b);
    rocket_bo_free(fd, &B.o); rocket_bo_free(fd, &B.r);
    free(in); free(W); free(out_lib); free(out_st);
    return diff ? 1 : 0;
}

int main(int argc, char **argv)
{
    int arm = -1, fd, bad = 0, i;
    const char *filt = argc > 2 ? argv[2] : NULL;
    int verbose = getenv("ROCKET_AF_VERBOSE") ? atoi(getenv("ROCKET_AF_VERBOSE")) : 0;

    for (i = 0; argc > 1 && i < 4; i++) if (!strcmp(argv[1], ARMS[i])) arm = i;
    if (arm < 0) { fprintf(stderr, "usage: %s stage|stage3t|own|time [substr]\n", argv[0]); return 2; }
    if (strcmp(rocket_hw_current()->name, "rk3576") != 0) { printf("not an RK3576\n"); return 2; }
    fd = rocket_open();
    if (fd < 0) { printf("no NPU\n"); return 2; }
    printf("arm %s\n", ARMS[arm]);
    if (arm == ARM_TIME) {
        bad = time_arm(fd) != 0;
    } else {
        int c0 = canary(fd);
        printf("  canary before any shape: %d/3840\n", c0);
        for (i = 0; i < N_SHAPES; i++) {
            int ok = 0;
            if (filt && !strstr(SHAPES[i].name, filt)) continue;
            if (run_shape(fd, &SHAPES[i], arm, verbose, &ok) != 0 || !ok) bad = 1;
        }
    }
    rocket_close(fd);
    return bad;
}
