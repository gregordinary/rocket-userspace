// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_fc_stage_gate.c — the packed-image fp16 first conv through rocket_conv2d_fp16_rk3576()
 * on the whole output stage: exact on random operands, at row windows and output-channel
 * tiles, back to back, and what it costs against the fp16 writer.
 *
 * Runs on an RK3576 board. A convolution of four or fewer input channels takes the
 * packed-image first conv. By default its programs write fp32 through the whole output
 * stage, which poisons nothing, so a 32-channel tile's row windows go out as one job and the
 * call ends with no power cycle. ROCKET_RK3576_FP16_F32OUT=0 restores the fp16 writer, which
 * cycles the power domain before every program after the first and once on exit.
 *
 * The operands are random small integers held in fp16 (input [-30,30], weights [-8,8]), so
 * every fp32 partial sum is exact, and each element is scored against a CPU fp32 model
 * narrowed to fp16 the way the entry narrows. After every shape an int8 direct conv (ic 32,
 * oc 32, 12x10, k3) runs as a canary against its own CPU model: a canary that comes back
 * wrong is a poisoned part. The entry plans its own row windows; a shape's `rows` forces a
 * finer plan through ROCKET_RK3576_MAX_ROWS for that call.
 *
 * Arms (one per process):
 *   gate   every shape below once, on the default path, each followed by the canary.
 *   long   back to back with no idle: 200 calls at the 224x224 k3 s2 stem (3 into 32) on
 *          two alternating inputs, then 100 at the same stem into 64 channels with the
 *          windows forced to 8 rows (two tiles of 14 windows); every call against its own
 *          reference; the canary at the end.
 *   time   per shape, blocks of 6 calls on each path in turn (4 blocks a path), the first
 *          call of a block discarded; medians, and the two paths' outputs compared.
 *
 *   rk3576_fc_stage_gate <arm> [substr]
 *
 * WHAT IT CANNOT SEE: an asymmetric pad (the fp16 entry refuses an extent it did not
 * derive, so TFLite's SAME at an even input and stride 2 never reaches this path), values
 * past fp16's exact-integer range or real image statistics (the one rounding is still the
 * narrowing, but a non-integer sum is not scored here), a stream chained with other layers,
 * and kernels other than the one it runs on. Which path ran is not observable from the
 * output; the time arm's ratio and the gate row's empty backstop count are the witnesses.
 *
 * Exit: 0 every shape exact with its canary, 1 otherwise, 2 no NPU or not an RK3576.
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

typedef struct {
    const char *name;
    unsigned ic, oc, iw, ih, k, stride, pad;
    unsigned rows;   /* the INPUT rows a window may hold (ROCKET_RK3576_MAX_ROWS); 0: the entry's plan */
} fshape_t;

static const fshape_t SHAPES[] = {
    { "ctl16_ic4",        4, 32,  16,  16, 3, 1, 1, 0 },
    { "rgb32_k3",         3, 32,  32,  32, 3, 1, 1, 0 },
    { "rgb32_k5",         3, 32,  32,  32, 5, 1, 2, 0 },
    { "rgb32_k7",         3, 32,  32,  32, 7, 1, 3, 0 },
    { "rgb32_s2",         3, 32,  32,  32, 3, 2, 1, 0 },
    { "rgb32_s2_valid",   3, 32,  32,  32, 3, 2, 0, 0 },
    { "rgb32_valid",      3, 16,  32,  32, 3, 1, 0, 0 },
    { "gray32_oc16",      1, 16,  32,  32, 3, 1, 1, 0 },
    { "two32_oc16",       2, 16,  32,  32, 3, 1, 1, 0 },
    { "rgb32_oc24",       3, 24,  32,  32, 3, 1, 1, 0 },
    { "rgb32_oc40",       3, 40,  32,  32, 3, 1, 1, 0 },
    { "rgb32_oc48",       3, 48,  32,  32, 3, 1, 1, 0 },
    { "rgb32_k3_r8",      3, 32,  32,  32, 3, 1, 1, 8 },
    { "rgb32_k3_r3",      3, 32,  32,  32, 3, 1, 1, 3 },   /* one output row a window */
    { "rgb32_s2_r3",      3, 32,  32,  32, 3, 2, 1, 3 },
    { "rgb32_oc64_r5",    3, 64,  32,  32, 3, 1, 1, 5 },
    { "rgb48_k5_r7",      3, 32,  48,  36, 5, 1, 2, 7 },
    { "rgb128_k3",        3, 32, 128, 128, 3, 1, 1, 0 },
    { "fc224_k3s2_p1",    3, 32, 224, 224, 3, 2, 1, 0 },
    { "fc224_k3s2_p0",    3, 32, 224, 224, 3, 2, 0, 0 },
    { "fc224_k7s2_p3",    3, 32, 224, 224, 7, 2, 3, 0 },
    { "fc224_k3s2_oc64",  3, 64, 224, 224, 3, 2, 1, 0 },
    { "fc224_k3s2_oc96",  3, 96, 224, 224, 3, 2, 1, 0 },
    { "fc224_k7s2_oc64",  3, 64, 224, 224, 7, 2, 3, 0 },
    { "fc224_oc64_r3",    3, 64, 224, 224, 3, 2, 1, 3 },   /* 112 windows a tile */
    { "fc224_rgba_oc64",  4, 64, 224, 224, 3, 2, 1, 0 },
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof SHAPES[0]))

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

/* ---- the int8 canary (the same program as tests/rk3576_argb_f32out_probe.c) ---- */
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

/* ---- one shape's operands, reference and call ---- */
struct fc {
    rocket_conv2d_desc d;
    unsigned ow, oh;
    _Float16 *in[2], *W, *out, *ref[2];
};

static void fc_free(struct fc *f)
{
    free(f->in[0]); free(f->in[1]); free(f->W); free(f->out); free(f->ref[0]); free(f->ref[1]);
    memset(f, 0, sizeof *f);
}

static void fc_ref(const struct fc *f, const _Float16 *in, _Float16 *ref)
{
    const rocket_conv2d_desc *d = &f->d;
    unsigned c, y, x, kh, kw, i;
    for (c = 0; c < (unsigned)d->oc; c++)
        for (y = 0; y < f->oh; y++)
            for (x = 0; x < f->ow; x++) {
                float acc = 0.0f;
                for (kh = 0; kh < (unsigned)d->kh; kh++)
                    for (kw = 0; kw < (unsigned)d->kw; kw++) {
                        int iy = (int)(y * d->stride_y + kh) - d->pad_top;
                        int ix = (int)(x * d->stride_x + kw) - d->pad_left;
                        if (iy < 0 || iy >= d->ih || ix < 0 || ix >= d->iw) continue;
                        for (i = 0; i < (unsigned)d->ic; i++)
                            acc += (float)in[((size_t)i * d->ih + iy) * d->iw + ix] *
                                   (float)f->W[(((size_t)c * d->ic + i) * d->kh + kh) * d->kw + kw];
                    }
                ref[((size_t)c * f->oh + y) * f->ow + x] = (_Float16)acc;
            }
}

static int fc_make(const fshape_t *S, struct fc *f, int two)
{
    unsigned i, v, seed;
    size_t nin, nw, nout;
    memset(f, 0, sizeof *f);
    f->d.ic = (int)S->ic; f->d.oc = (int)S->oc; f->d.ih = (int)S->ih; f->d.iw = (int)S->iw;
    f->d.kh = (int)S->k; f->d.kw = (int)S->k;
    f->d.stride_y = (int)S->stride; f->d.stride_x = (int)S->stride;
    f->d.pad_top = (int)S->pad; f->d.pad_left = (int)S->pad;
    f->d.dil_y = 1; f->d.dil_x = 1;
    f->ow = (unsigned)rocket_conv2d_ow(&f->d);
    f->oh = (unsigned)rocket_conv2d_oh(&f->d);
    if (!f->ow || !f->oh) return -1;
    nin = (size_t)S->ic * S->ih * S->iw; nw = (size_t)S->oc * S->ic * S->k * S->k;
    nout = (size_t)S->oc * f->oh * f->ow;
    f->W = malloc(nw * sizeof *f->W); f->out = malloc(nout * sizeof *f->out);
    if (!f->W || !f->out) return -1;
    seed = 0x9E3779B9u ^ (S->ic * 31u + S->oc * 17u + S->iw * 7u + S->ih * 3u + S->k);
    for (i = 0; i < nw; i++) {
        seed = seed * 1103515245u + 12345u; f->W[i] = (_Float16)((int)((seed >> 16) % 17u) - 8);
    }
    for (v = 0; v < (two ? 2u : 1u); v++) {
        f->in[v] = malloc(nin * sizeof *f->in[v]);
        f->ref[v] = malloc(nout * sizeof *f->ref[v]);
        if (!f->in[v] || !f->ref[v]) return -1;
        for (i = 0; i < nin; i++) {
            seed = seed * 1103515245u + 12345u;
            f->in[v][i] = (_Float16)((int)((seed >> 16) % 61u) - 30);
        }
        fc_ref(f, f->in[v], f->ref[v]);
    }
    return 0;
}

static void set_rows(unsigned rows)
{
    if (rows) {
        char b[16];
        snprintf(b, sizeof b, "%u", rows);
        setenv("ROCKET_RK3576_MAX_ROWS", b, 1);
    } else {
        unsetenv("ROCKET_RK3576_MAX_ROWS");
    }
}

/* One call; returns the entry's rc and the wrong-element count against ref[v]. */
static int fc_call(int fd, struct fc *f, unsigned v, unsigned rows, size_t *wrong, double *ms)
{
    size_t i, n = (size_t)f->d.oc * f->oh * f->ow;
    double t0;
    int rc;
    memset(f->out, 0, n * sizeof *f->out);
    set_rows(rows);
    t0 = now_ms();
    rc = rocket_conv2d_fp16_rk3576(fd, &f->d, f->in[v], f->W, f->out);
    *ms = now_ms() - t0;
    set_rows(0);
    *wrong = 0;
    if (rc != ROCKET_OK) return rc;
    for (i = 0; i < n; i++)
        if (*(const uint16_t *)&f->out[i] != *(const uint16_t *)&f->ref[v][i]) (*wrong)++;
    return rc;
}

static int arm_gate(int fd, const char *filt)
{
    int i, bad = 0, ran = 0;
    for (i = 0; i < N_SHAPES; i++) {
        const fshape_t *S = &SHAPES[i];
        struct fc f;
        size_t wrong = 0;
        double ms = 0;
        int rc, can;
        if (filt && !strstr(S->name, filt)) continue;
        if (fc_make(S, &f, 0) < 0) { printf("  %-18s setup failed\n", S->name); fc_free(&f); bad = 1; continue; }
        rc = fc_call(fd, &f, 0, S->rows, &wrong, &ms);
        can = canary(fd);
        printf("  %-18s %3ux%-3u ic%u oc%-3u k%u s%u p%u rows %-2u -> %3ux%-3u | rc %d, %zu of %zu "
               "wrong, %7.2f ms | canary %d/3840 %s\n", S->name, S->iw, S->ih, S->ic, S->oc, S->k,
               S->stride, S->pad, S->rows, f.ow, f.oh, rc, wrong,
               (size_t)S->oc * f.oh * f.ow, ms, can,
               (rc == ROCKET_OK && !wrong && can == 3840) ? "EXACT"
               : (can != 3840 ? "CANARY-WRONG" : "WRONG"));
        if (rc != ROCKET_OK || wrong || can != 3840) bad = 1;
        ran++;
        fc_free(&f);
    }
    printf("gate: %d shapes, %s\n", ran, bad ? "FAIL" : "all exact, every canary exact");
    return bad;
}

static int long_run(int fd, const fshape_t *S, unsigned calls, unsigned *bad_calls)
{
    struct fc f;
    double *t = calloc(calls, sizeof *t), med;
    unsigned c, worst_call = 0;
    size_t wrong_total = 0;
    int rc_bad = 0;
    *bad_calls = 0;
    if (!t || fc_make(S, &f, 1) < 0) { free(t); fc_free(&f); return -1; }
    for (c = 0; c < calls; c++) {
        size_t wrong = 0;
        int rc = fc_call(fd, &f, c & 1u, S->rows, &wrong, &t[c]);
        if (rc != ROCKET_OK) { rc_bad++; (*bad_calls)++; continue; }
        if (wrong) {
            (*bad_calls)++;
            wrong_total += wrong;
            if (*bad_calls <= 5)
                printf("    call %u: %zu wrong (%.2f ms)\n", c, wrong, t[c]);
        }
        if (t[c] > t[worst_call]) worst_call = c;
    }
    {
        double worst = t[worst_call];
        qsort(t, calls, sizeof *t, cmp_d);
        med = t[calls / 2];
        printf("  %-18s rows %-2u: %u calls, two alternating inputs, %u wrong calls (%zu "
               "elements, %d rc), median %.2f ms, min %.2f, max %.2f (call %u)\n", S->name,
               S->rows, calls, *bad_calls, wrong_total, rc_bad, med, t[0], worst, worst_call);
    }
    free(t);
    fc_free(&f);
    return 0;
}

static int arm_long(int fd)
{
    static const fshape_t L1 = { "long_stem_oc32",  3, 32, 224, 224, 3, 2, 1, 0 };
    static const fshape_t L2 = { "long_stem_oc64_r8", 3, 64, 224, 224, 3, 2, 1, 8 };
    unsigned b1 = 0, b2 = 0;
    int can;
    if (long_run(fd, &L1, 200, &b1) < 0 || long_run(fd, &L2, 100, &b2) < 0) return 1;
    can = canary(fd);
    printf("long: %u + %u wrong calls, canary %d/3840 %s\n", b1, b2, can,
           (!b1 && !b2 && can == 3840) ? "PASS" : "FAIL");
    return (b1 || b2 || can != 3840) ? 1 : 0;
}

/* Blocks of calls on one path, the paths taking turns. A call on the fp16 writer ends with
 * a power cycle and the stage's does not, so a strict per-call alternation would bill every
 * stage call the power-up the previous writer call left it; a block is what a caller running
 * either path in a loop sees. */
#define TB 6
#define TBLOCKS 4
static int arm_time(int fd, const char *filt)
{
    static const fshape_t TS[] = {
        { "t_stem_k3_oc32",  3, 32, 224, 224, 3, 2, 1, 0 },
        { "t_stem_k3_oc64",  3, 64, 224, 224, 3, 2, 1, 0 },
        { "t_stem_k7_oc64",  3, 64, 224, 224, 7, 2, 3, 0 },
        { "t_rgb32_k3",      3, 32,  32,  32, 3, 1, 1, 0 },
    };
    unsigned s;
    int bad = 0;
    for (s = 0; s < sizeof TS / sizeof TS[0]; s++) {
        const fshape_t *S = &TS[s];
        struct fc f;
        double ts[TBLOCKS * (TB - 1)], tw[TBLOCKS * (TB - 1)], first_s = 0, first_w = 0;
        unsigned ns = 0, nw = 0, blk, c, wrong_s = 0, wrong_w = 0;
        _Float16 *keep;
        size_t n, i, differ = 0;
        if (filt && !strstr(S->name, filt)) continue;
        if (fc_make(S, &f, 0) < 0) { fc_free(&f); return 1; }
        n = (size_t)S->oc * f.oh * f.ow;
        keep = malloc(n * sizeof *keep);
        for (blk = 0; blk < 2u * TBLOCKS; blk++) {
            int writer = (int)(blk & 1u);   /* stage first */
            if (writer) setenv("ROCKET_RK3576_FP16_F32OUT", "0", 1);
            else unsetenv("ROCKET_RK3576_FP16_F32OUT");
            for (c = 0; c < TB; c++) {
                size_t wrong = 0;
                double ms = 0;
                int rc = fc_call(fd, &f, 0, 0, &wrong, &ms);
                if (rc != ROCKET_OK || wrong) {
                    if (writer) wrong_w++; else wrong_s++;
                }
                if (!c) { if (writer) first_w += ms; else first_s += ms; continue; }
                if (writer) tw[nw++] = ms; else ts[ns++] = ms;
            }
            if (blk == 0 && keep) memcpy(keep, f.out, n * sizeof *keep);
            if (blk == 1 && keep)
                for (i = 0; i < n; i++)
                    differ += *(const uint16_t *)&keep[i] != *(const uint16_t *)&f.out[i];
        }
        unsetenv("ROCKET_RK3576_FP16_F32OUT");
        qsort(ts, ns, sizeof *ts, cmp_d);
        qsort(tw, nw, sizeof *tw, cmp_d);
        printf("  %-16s %ux%u ic%u oc%-3u k%u s%u -> %ux%u | stage median %.2f ms (%.2f-%.2f, n %u) "
               "| fp16 writer median %.2f ms (%.2f-%.2f, n %u) | stage/writer %.3f | first of a "
               "block: stage %.2f, writer %.2f ms | wrong calls %u/%u | outputs differ at %zu\n",
               S->name, S->iw, S->ih, S->ic, S->oc, S->k, S->stride, f.ow, f.oh,
               ts[ns / 2], ts[0], ts[ns - 1], ns, tw[nw / 2], tw[0], tw[nw - 1], nw,
               ts[ns / 2] / tw[nw / 2], first_s / TBLOCKS, first_w / TBLOCKS, wrong_s, wrong_w,
               differ);
        if (wrong_s || wrong_w || differ) bad = 1;
        free(keep);
        fc_free(&f);
    }
    return bad;
}

int main(int argc, char **argv)
{
    int fd, bad;
    const char *filt = argc > 2 ? argv[2] : NULL;
    const char *arm = argc > 1 ? argv[1] : "gate";

    if (strcmp(arm, "gate") && strcmp(arm, "long") && strcmp(arm, "time")) {
        fprintf(stderr, "usage: %s gate|long|time [substr]\n", argv[0]);
        return 2;
    }
    if (strcmp(rocket_hw_current()->name, "rk3576") != 0) { printf("not an RK3576\n"); return 2; }
    fd = rocket_open();
    if (fd < 0) { printf("no NPU\n"); return 2; }
    printf("arm %s, ROCKET_RK3576_FP16_F32OUT=%s, canary before any shape: %d/3840\n", arm,
           getenv("ROCKET_RK3576_FP16_F32OUT") ? getenv("ROCKET_RK3576_FP16_F32OUT") : "(unset)",
           canary(fd));
    if (!strcmp(arm, "gate")) bad = arm_gate(fd, filt);
    else if (!strcmp(arm, "long")) bad = arm_long(fd);
    else bad = arm_time(fd, filt);
    rocket_close(fd);
    return bad;
}
