// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_fp16_fp32out_probe.c — does the fp16 direct conv program compute when its DPU
 * output stage is the whole wide stage, and where does each fp32 element land?
 *
 * The fp16 direct program writes fp16 and poisons the next submit, and the two bits it
 * poisons through (DPU 0x4038[4], 0x4050[17]) are its float arithmetic, so clearing them
 * is not a cure. The whole wide output stage (the one that makes the int8 direct program a
 * dense int32 writer: 0x4010 0xa0000002, 0x4030 low half 0x0310, 0x4038 0x53, 0x4044 2,
 * 0x4050 0x00023333, 0x40AC/B0/B4 0/1/0, 0x40B8 at three task surfaces) keeps both bits set
 * and poisons nothing on it. On the fp16 program it writes 16.0 on all-ones operands and
 * poisons nothing either. All-ones operands cannot see a program that reads one feature
 * surface twice and skips another, which is the float path's known failure, so this asks
 * with operands that can.
 *
 *   decode   k 1, constructed operands: two live input lanes give each output the distinct
 *            value (n + 1) + m*(N + 1), so each fp32 word names one (pixel, channel) or
 *            none. Reports how many are found, how many sit where the int8 dense writer's
 *            map puts them, and the word each of the first few occupies.
 *   check    random small-integer operands at k 1 and 3, every element read through the
 *            plain int32 cube map and scored against a CPU model. The fp16 program with
 *            its own output stage runs first as the harness's control.
 *
 * One task, ic 16 (the fp16 program's contraction), oc 16 or 32, stride 1, pad k/2, no
 * bias. Every accumulation is of small integers and exact in fp32.
 *
 * Env: ROCKET_FP_IW / _IH (plane, default 8x4), ROCKET_FP_OC (32), ROCKET_FP_GAP_MS (300,
 * idle before each submit, past the autosuspend that clears a poisoned part).
 *
 * Exit: 0 ran (decode) or every element exact (check), 1 otherwise, 2 no NPU or not RK3576.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"
#include "test_fill.h"

static int env_int(const char *name, int dflt)
{
    const char *e = getenv(name);
    return (e && *e) ? (int)strtol(e, NULL, 0) : dflt;
}

static void sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

/* The int8 dense writer's map: A pixels, g = c/32, j = (c%32)/16, L = (c%16)/4,
 * s = 2p + j, atom = 8A*g + 4A*(s/A) + A*L + s%A, word = 4*atom + c%4. */
static size_t dense_word(unsigned A, unsigned c, unsigned p)
{
    unsigned g = c / 32u, j = (c % 32u) / 16u, L = (c % 16u) / 4u, s = 2u * p + j;
    size_t atom = (size_t)8u * A * g + (size_t)4u * A * (s / A) + (size_t)A * L + s % A;
    return 4u * atom + c % 4u;
}

/* The plain int32 cube, which the fp16 program writes under the same stage: four fp32
 * lanes to a 16-byte atom, one atom per pixel, each quad of channels a plane of A atoms. */
static size_t cube_word(unsigned A, unsigned c, unsigned p)
{
    return 4u * ((size_t)(c / 4u) * A + p) + c % 4u;
}

struct run {
    unsigned iw, ih, oc, k;
    int wide;              /* 1 = the whole wide stage, 0 = the program as emitted */
    _Float16 *in, *w;      /* [16][ih][iw], OIHW [oc][16][k][k] */
    float *out;            /* [oc][oh][ow] as read back */
    unsigned written;      /* output words or halves not holding the stamp */
    uint32_t *raw;         /* when set, the raw output words land here */
    size_t raw_words;
};

#define STAMP 0xA5u

static int run_once(int fd, struct run *r)
{
    const unsigned IC = ROCKET_RK3576_FP16_IC_SLICE, pad = r->k / 2u;
    const unsigned oh = r->ih, ow = r->iw, A = oh * ow;
    const unsigned ocreg = rocket_rk3576_fp16_pad_oc(r->oc);
    size_t in_bytes = (size_t)(IC / 8u) * r->ih * r->iw * 8u * 2u;
    size_t w_bytes = rocket_rk3576_fp16_slice_weight_bytes(r->oc, IC, r->k, r->k);
    size_t coeff = rocket_rk3576_coeff_bytes(ocreg);
    size_t obytes = (size_t)ocreg * A * 4u * 2u;   /* twice a dense fp32 surface */
    rocket_bo in = {0}, w = {0}, b = {0}, o = {0}, rb = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p;
    rocket_rk3576_ic_task t = { 0, (uint16_t)IC, 0 };
    char spec[400];
    unsigned c, y, x;
    int rc = -1;

    if (rocket_bo_alloc32(fd, in_bytes, &in) < 0 || rocket_bo_alloc32(fd, w_bytes, &w) < 0 ||
        rocket_bo_alloc32(fd, coeff, &b) < 0 || rocket_bo_alloc32(fd, obytes, &o) < 0 ||
        rocket_bo_alloc32(fd, sizeof ops, &rb) < 0) {
        fprintf(stderr, "BO allocation failed\n");
        goto out;
    }

    /* Feature cube: C2 = 8 channel planes of iw*ih 16-byte atoms. */
    rocket_bo_prep(fd, &in, 1, 0);
    memset(in.ptr, 0, in_bytes);
    for (c = 0; c < IC; c++)
        for (y = 0; y < r->ih; y++)
            for (x = 0; x < r->iw; x++)
                ((_Float16 *)in.ptr)[(size_t)(c / 8u) * r->ih * r->iw * 8u +
                                     (size_t)(y * r->iw + x) * 8u + c % 8u] =
                    r->in[((size_t)c * r->ih + y) * r->iw + x];
    rocket_bo_fini(fd, &in);

    rocket_bo_prep(fd, &w, 1, 0);
    if (rocket_rk3576_fp16_pack_slice_weights(w.ptr, w_bytes, r->w, r->oc, IC, r->k, r->k,
                                              &t) < 0) {
        rocket_bo_fini(fd, &w);
        fprintf(stderr, "weight pack refused\n");
        goto out;
    }
    rocket_bo_fini(fd, &w);

    rocket_bo_prep(fd, &b, 1, 0);
    if (rocket_rk3576_pack_coeff_prec(b.ptr, coeff, NULL, ocreg, precision_float16) < 0) {
        rocket_bo_fini(fd, &b);
        fprintf(stderr, "coefficient pack refused\n");
        goto out;
    }
    rocket_bo_fini(fd, &b);

    rocket_bo_prep(fd, &o, 1, 0);
    memset(o.ptr, STAMP, obytes);
    rocket_bo_fini(fd, &o);

    memset(&p, 0, sizeof p);
    p.ic = (uint16_t)IC; p.oc = (uint16_t)ocreg;
    p.ih = (uint16_t)r->ih; p.iw = (uint16_t)r->iw;
    p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
    p.kh = (uint16_t)r->k; p.kw = (uint16_t)r->k;
    p.stride_y = 1; p.stride_x = 1;
    p.pad_top = (uint8_t)pad; p.pad_left = (uint8_t)pad;
    p.ih_full = (uint16_t)r->ih; p.oh_full = (uint16_t)oh;
    p.int8_out = 0;
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;
    p.input_dma = (uint32_t)in.dma_address;
    p.weights_dma = (uint32_t)w.dma_address;
    p.bias_dma = (uint32_t)b.dma_address;
    p.output_dma = (uint32_t)o.dma_address;
    p.tasks = ops;

    if (r->wide) {
        snprintf(spec, sizeof spec,
                 "0x4010=0xa0000002,0x4030=0x%08x,0x4038=0x00000053,0x4044=0x2,"
                 "0x4050=0x00023333,0x40ac=0,0x40b0=1,0x40b4=0,0x40b8=0x%x",
                 ((ocreg - 1u) << 16) | 0x0310u, 3u * A);
        setenv("ROCKET_RK3576_SET", spec, 1);
    }
    rc = gen_conv2d_fp16_rk3576(&p);
    if (r->wide) unsetenv("ROCKET_RK3576_SET");
    if (rc != 0) { fprintf(stderr, "the generator refused this geometry\n"); rc = -1; goto out; }

    rocket_bo_prep(fd, &rb, 1, 0);
    memcpy(rb.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &rb);
    in_h[0] = in.handle; in_h[1] = w.handle; in_h[2] = b.handle; in_h[3] = rb.handle;
    out_h[0] = o.handle;

    sleep_ms(env_int("ROCKET_FP_GAP_MS", 300));
    if (rocket_submit_matmul(fd, &rb, p.task_count, in_h, 4, out_h, 1, 4000) != 0) {
        fprintf(stderr, "submit failed\n"); rc = -1; goto out;
    }
    rocket_bo_prep(fd, &o, 0, 2000000000ull);
    r->written = 0;
    if (r->wide) {
        const uint32_t *wd = (const uint32_t *)o.ptr;
        size_t i;
        for (i = 0; i < obytes / 4u; i++) r->written += wd[i] != 0xA5A5A5A5u;
        for (c = 0; c < r->oc; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    float v;
                    memcpy(&v, &wd[cube_word(A, c, y * ow + x)], 4);
                    r->out[((size_t)c * oh + y) * ow + x] = v;
                }
    } else {
        const uint16_t *hd = (const uint16_t *)o.ptr;
        size_t i;
        for (i = 0; i < obytes / 2u; i++) r->written += hd[i] != 0xA5A5u;
        for (c = 0; c < r->oc; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    _Float16 h;
                    int idx = rocket_rk3576_fp16_out_index(oh, ow, c, y, x);
                    memcpy(&h, &hd[idx], 2);
                    r->out[((size_t)c * oh + y) * ow + x] = (float)h;
                }
    }
    /* The raw words, for the decode. */
    if (r->raw) memcpy(r->raw, o.ptr, r->raw_words * 4u < obytes ? r->raw_words * 4u : obytes);
    rocket_bo_fini(fd, &o);
    rc = 0;
out:
    rocket_bo_free(fd, &in); rocket_bo_free(fd, &w); rocket_bo_free(fd, &b);
    rocket_bo_free(fd, &o); rocket_bo_free(fd, &rb);
    return rc;
}

static void ref_conv(const struct run *r, float *want)
{
    const unsigned IC = ROCKET_RK3576_FP16_IC_SLICE, pad = r->k / 2u;
    unsigned c, y, x, ic, ky, kx;
    for (c = 0; c < r->oc; c++)
        for (y = 0; y < r->ih; y++)
            for (x = 0; x < r->iw; x++) {
                float acc = 0.0f;
                for (ic = 0; ic < IC; ic++)
                    for (ky = 0; ky < r->k; ky++)
                        for (kx = 0; kx < r->k; kx++) {
                            int iy = (int)(y + ky) - (int)pad, ix = (int)(x + kx) - (int)pad;
                            if (iy < 0 || ix < 0 || iy >= (int)r->ih || ix >= (int)r->iw) continue;
                            acc += (float)r->in[((size_t)ic * r->ih + iy) * r->iw + ix] *
                                   (float)r->w[(((size_t)c * IC + ic) * r->k + ky) * r->k + kx];
                        }
                want[((size_t)c * r->ih + y) * r->iw + x] = acc;
            }
}

static int alloc_run(struct run *r)
{
    const unsigned IC = ROCKET_RK3576_FP16_IC_SLICE;
    r->in = calloc((size_t)IC * r->ih * r->iw, sizeof(_Float16));
    r->w = calloc((size_t)r->oc * IC * r->k * r->k, sizeof(_Float16));
    r->out = calloc((size_t)r->oc * r->ih * r->iw, sizeof(float));
    return (r->in && r->w && r->out) ? 0 : -1;
}

static void free_run(struct run *r) { free(r->in); free(r->w); free(r->out); }

static int mode_decode(int fd, unsigned iw, unsigned ih, unsigned oc)
{
    struct run r = { iw, ih, oc, 1u, 1, NULL, NULL, NULL, 0, NULL, 0 };
    const unsigned M = iw * ih, IC = ROCKET_RK3576_FP16_IC_SLICE;
    unsigned m, n, found = 0, fits = 0, fits_cube = 0, shown = 0;
    size_t words = (size_t)rocket_rk3576_fp16_pad_oc(oc) * M * 2u, i;
    uint32_t *raw = calloc(words, 4);

    if (!raw || alloc_run(&r) < 0) { free(raw); return 1; }
    r.raw = raw; r.raw_words = words;
    /* Two live lanes: in[0] = 1, in[1] = m; w[n][0] = n + 1, w[n][1] = N + 1. */
    for (m = 0; m < M; m++) {
        r.in[(size_t)0 * M + m] = (_Float16)1.0f;
        r.in[(size_t)1 * M + m] = (_Float16)(float)m;
    }
    for (n = 0; n < oc; n++) {
        r.w[(size_t)n * IC + 0] = (_Float16)(float)(n + 1u);
        r.w[(size_t)n * IC + 1] = (_Float16)(float)(oc + 1u);
    }
    if (run_once(fd, &r) != 0) { free(raw); free_run(&r); return 1; }
    for (m = 0; m < M; m++)
        for (n = 0; n < oc; n++) {
            float want = (float)(n + 1u) + (float)m * (float)(oc + 1u);
            unsigned hits = 0;
            size_t at = 0;
            for (i = 0; i < words; i++) {
                float v;
                memcpy(&v, &raw[i], 4);
                if (v == want) { hits++; at = i; }
            }
            if (hits == 1) {
                found++;
                if (at == dense_word(M, n, m)) fits++;
                if (at == cube_word(M, n, m)) fits_cube++;
                else if (shown < 8) {
                    printf("    C[%u][%u] at word %zu, the cube map says %zu\n", m, n, at,
                           cube_word(M, n, m));
                    shown++;
                }
            }
        }
    printf("decode %ux%u oc %u: %u words written, %u/%u values found once, %u/%u where the "
           "int32 cube map puts them, %u/%u where the int8 dense map does\n", iw, ih, oc,
           r.written, found, M * oc, fits_cube, found, fits, found);
    free(raw);
    free_run(&r);
    return 0;
}

static int mode_check(int fd, unsigned iw, unsigned ih, unsigned oc, unsigned k, int wide,
                      uint64_t seed)
{
    struct run r = { iw, ih, oc, k, wide, NULL, NULL, NULL, 0, NULL, 0 };
    const unsigned IC = ROCKET_RK3576_FP16_IC_SLICE;
    size_t n_out = (size_t)oc * ih * iw, i, bad = 0;
    float *want;
    if (alloc_run(&r) < 0) return 1;
    tf_fill_f16_int(r.in, (size_t)IC * ih * iw, seed, -3, 3);
    tf_fill_f16_int(r.w, (size_t)oc * IC * k * k, seed ^ 0x55u, -3, 3);
    want = calloc(n_out, sizeof(float));
    if (!want || run_once(fd, &r) != 0) { free(want); free_run(&r); return 1; }
    ref_conv(&r, want);
    for (i = 0; i < n_out; i++) bad += r.out[i] != want[i];
    printf("check %-14s %2ux%-2u oc %2u k%u: %6u written, %zu of %zu wrong%s\n",
           wide ? "wide stage" : "fp16 (control)", iw, ih, oc, k, r.written, bad, n_out,
           bad ? "" : "  EXACT");
    if (bad) {
        size_t shown = 0;
        for (i = 0; i < n_out && shown < 4; i++)
            if (r.out[i] != want[i]) {
                printf("    (c %zu, p %zu): got %g want %g\n", i / ((size_t)ih * iw),
                       i % ((size_t)ih * iw), (double)r.out[i], (double)want[i]);
                shown++;
            }
    }
    free(want);
    free_run(&r);
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "check";
    unsigned iw = (unsigned)env_int("ROCKET_FP_IW", 8), ih = (unsigned)env_int("ROCKET_FP_IH", 4);
    unsigned oc = (unsigned)env_int("ROCKET_FP_OC", 32);
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, fails = 0;

    if (!hw || strcmp(hw->name, "rk3576") != 0) { printf("not an RK3576, skipping\n"); return 2; }
    fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }
    if (!strcmp(mode, "decode")) {
        fails = mode_decode(fd, iw, ih, oc);
    } else {
        unsigned k;
        for (k = 1; k <= 3; k += 2) {
            fails += mode_check(fd, iw, ih, oc, k, 0, 0xF16u + k);
            fails += mode_check(fd, iw, ih, oc, k, 1, 0xF16u + k);
        }
    }
    rocket_close(fd);
    return fails ? 1 : 0;
}
