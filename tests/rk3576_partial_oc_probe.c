// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_partial_oc_probe.c — can a direct int8 conv PROGRAM a partial 32-channel output
 * group, instead of padding it with rocket_rk3576_pad_oc()?
 *
 * Runs on an RK3576 board. The emitter's direct path programs DPU 0x4050 as the even-count
 * word 0x80011111 at every count, and the library pads `oc` to a multiple of 32 so the
 * programmed 16-channel atom count is always even. gahingwoo/mesa-rk3576 programs `oc`
 * unpadded and clears 0x4050 bit 8 where DIV_ROUND_UP(oc, 16) is odd (0x80011011). The
 * vendor compiler does not pad either: its weight cube holds a 16-channel trailing group
 * where conv_weight_index_int8 pads it to 32, and its WEIGHT_BYTES (0x101C) is
 * ic*oc*kh*kw, the unpadded size. Three things therefore differ between our padded program
 * and a partial one: the programmed `oc` (every register the emitter derives from it), the
 * 0x4050 word, and the weight cube's trailing group. This probe crosses the last two at an
 * unpadded `oc`:
 *
 *   arm        oc programmed   0x4050         weight cube
 *   ctl        pad_oc(oc)      as emitted     conv_weight_index_int8 (32-wide groups)
 *   even_pad   oc              as emitted     32-wide groups
 *   odd_pad    oc              0x80011011     32-wide groups
 *   odd_vend   oc              0x80011011     trailing group oc%32 wide (the vendor form)
 *   even_vend  oc              as emitted     trailing group oc%32 wide
 *   auto_vend  oc              odd word where DIV_ROUND_UP(oc,16) is odd (gahingwoo's
 *                              rule), as emitted otherwise; trailing group oc%32 wide
 *
 * The "vendor form" is an inference, not a capture: the cube nest of
 * conv_weight_index_int8, [oc/32][ic/32][kh][kw][oc%32][ic%32], with the LAST oc group's
 * width og = oc - 32*(oc/32) in place of 32. Its size is then exactly ic*oc*kh*kw, which
 * is what WEIGHT_BYTES says the CNA fetches. At k=1 with one 32-wide ic group the two cubes
 * are byte-identical over the fetched bytes, so the shapes carry ic=64 to separate them.
 *
 * Each shape runs the arm, scores every real output channel bit-exactly against the CPU
 * model (the same model and operands as rk3576_conv_gate), counts the sentinel bytes the
 * arm left in its own channels and the bytes it wrote into the padding channels' surfaces
 * (oc..pad_oc-1, which a partial program must not touch), and then runs a CANARY: the
 * padded control program on the same operands, which must be exact. A canary that comes
 * back wrong means the arm left the part poisoned for the next submit.
 *
 * Usage:
 *   rk3576_partial_oc_probe <arm> [substr]   run one arm over the shape table
 *   rk3576_partial_oc_probe -l               list the shapes
 * One arm per process: a program that poisons the next submit must not reach the next arm.
 * Idle 3 s between processes after an arm that poisoned.
 *
 * Env: ROCKET_G_VERBOSE=1 prints the first mismatches per shape.
 *
 * The `_r<n>` shapes force a row cap of n, so the partial group crosses a row window, each
 * window its own submit as rk3576_conv_gate drives it.
 *
 * WHAT IT CANNOT SEE: the int32 writers (dense and wide), whose generators refuse an `oc`
 * that is not a multiple of 32; the depthwise path, which programs its own count; fp16;
 * a chained (one-kick) stream; a partial INPUT group. A green arm says nothing about those.
 *
 * Exit: 0 every shape of the arm exact with its canary, 1 any shape wrong, 2 no NPU.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_hw_profile.h"
#include "requant_model.h"

#define C2       16
#define SENTINEL 0xAA
#define ODD_WORD 0x80011011u

int feature_index(int C, int H, int W, int C2_, int c, int h, int w);
int conv_weight_index_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

typedef struct {
    const char *name;
    unsigned ic, oc, iw, ih, k, stride, same;
    unsigned rows;     /* non-zero: force the per-task row cap (ROCKET_RK3576_MAX_ROWS) */
} pshape_t;

static const pshape_t shapes[] = {
    { "oc48_k1_ic64",   64, 48, 14, 14, 1, 1, 0, 0 },
    { "oc48_k3_ic64",   64, 48, 14, 14, 3, 1, 1, 0 },
    { "oc16_k1_ic64",   64, 16, 12, 10, 1, 1, 0, 0 },
    { "oc16_k3_ic32",   32, 16, 12, 10, 3, 1, 1, 0 },
    { "oc80_k3_ic64",   64, 80, 12, 12, 3, 1, 1, 0 },
    { "oc48_k3s2_ic32", 32, 48, 15, 13, 3, 2, 1, 0 },
    { "oc112_k5_ic32",  32, 112, 10, 9, 5, 1, 1, 0 },
    /* oc not a multiple of 16: the trailing group is og = oc%32 wide, the atom count
     * DIV_ROUND_UP(oc,16) is even at 24 and 56 and odd at 8 and 40. */
    { "oc8_k3_ic32",    32,   8, 12, 10, 3, 1, 1, 0 },
    { "oc24_k1_ic64",   64,  24, 12, 10, 1, 1, 0, 0 },
    { "oc24_k3_ic32",   32,  24, 12, 10, 3, 1, 1, 0 },
    { "oc40_k3_ic64",   64,  40, 11, 13, 3, 1, 1, 0 },
    { "oc56_k3_ic32",   32,  56, 12, 12, 3, 1, 1, 0 },
    /* More atoms, and wider contractions. */
    { "oc176_k3_ic96",  96, 176, 14, 14, 3, 1, 1, 0 },
    { "oc208_k1_ic128", 128, 208, 9, 8, 1, 1, 0, 0 },
    { "oc48_k1s2_ic32", 32,  48, 15, 13, 1, 2, 0, 0 },
    /* Row windows: the same partial group across a forced split. */
    { "oc48_k3_r4",     64,  48, 20, 20, 3, 1, 1, 4 },
    { "oc80_k3_r6",     32,  80, 18, 22, 3, 1, 1, 6 },
    { "oc24_k5_r5",     32,  24, 16, 17, 5, 1, 1, 5 },
    { "oc48_k3s2_r3",   32,  48, 21, 19, 3, 2, 1, 3 },
};
#define NSHAPES (sizeof shapes / sizeof shapes[0])

enum { ARM_CTL, ARM_EVEN_PAD, ARM_ODD_PAD, ARM_ODD_VEND, ARM_EVEN_VEND, ARM_AUTO_VEND };
static const char *arm_names[] = { "ctl", "even_pad", "odd_pad", "odd_vend", "even_vend",
                                   "auto_vend" };

static int env_int(const char *name, int dflt)
{
    const char *e = getenv(name);
    return (e && *e) ? (int)strtol(e, NULL, 0) : dflt;
}

static size_t out_index(unsigned surf_elems, unsigned c, unsigned pix)
{
    return (size_t)(c / C2) * surf_elems * C2 + (size_t)C2 * pix + (c % C2);
}

/* The vendor-form index: conv_weight_index_int8's nest with the last oc group og wide. */
static size_t vend_index(unsigned OC, unsigned IC, unsigned K,
                         unsigned oc, unsigned ic, unsigned kh, unsigned kw)
{
    unsigned n_ic = (IC + 31u) / 32u, g = oc / 32u;
    unsigned og = (g == OC / 32u) ? (OC % 32u) : 32u;
    size_t base = (size_t)g * n_ic * K * K * 32u * 32u;   /* every earlier group is full */
    return base + ((((size_t)(ic / 32u) * K + kh) * K + kw) * og + (oc % 32u)) * 32u
           + ic % 32u;
}

struct res {
    int exact, total, maxdiff, untouched, pad_written, tasks, odd;
    int c_exact, c_total;
};

/* One program on prepared buffers, split into row tasks by the library's planner (forced
 * finer with `rows`). Each task is its own submit, as rk3576_conv_gate drives it.
 * Returns the task count, or -1. */
static int submit_conv(int fd, conv_params_t *p, rocket_bo *r_bo, rocket_bo *o_bo,
                       uint32_t *in_h, uint32_t *out_h, unsigned ih, unsigned oh,
                       unsigned rows)
{
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    rocket_rk3576_row_task plan[256];
    unsigned n = 1, t;
    conv_params_t base = *p;

    base.ih_full = (uint16_t)ih; base.oh_full = (uint16_t)oh;
    if (rows) {
        char buf[16];
        snprintf(buf, sizeof buf, "%u", rows);
        setenv("ROCKET_RK3576_MAX_ROWS", buf, 1);
    } else {
        unsetenv("ROCKET_RK3576_MAX_ROWS");
    }
    if (rocket_rk3576_plan_rows(&base, 0, plan, 256, &n) < 0) {
        unsetenv("ROCKET_RK3576_MAX_ROWS");
        return -1;
    }
    for (t = 0; t < n; t++) {
        conv_params_t q = base;
        q.ih = plan[t].ih; q.oh = plan[t].oh;
        q.pad_top = plan[t].pad_top;
        q.input_iova  = base.input_iova + plan[t].feature_off;
        q.output_iova = base.output_iova + plan[t].output_off;
        q.tasks = ops;
        q.task_count = 0;
        if (gen_conv2d_int8_rk3576(&q) != 0) { unsetenv("ROCKET_RK3576_MAX_ROWS"); return -1; }
        rocket_bo_prep(fd, r_bo, 1, 0);
        memcpy(r_bo->ptr, ops, q.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, r_bo);
        if (rocket_submit_matmul(fd, r_bo, q.task_count, in_h, 4, out_h, 1, 2000)) {
            unsetenv("ROCKET_RK3576_MAX_ROWS");
            return -1;
        }
        /* Wait for this task before the next one rewrites the regcmd BO it reads. */
        rocket_bo_prep(fd, o_bo, 0, 2000000000ull);
        rocket_bo_fini(fd, o_bo);
    }
    unsetenv("ROCKET_RK3576_MAX_ROWS");
    return (int)n;
}

static int run_shape(int fd, const pshape_t *s, int arm, struct res *R)
{
    unsigned ic = s->ic, oc = s->oc, iw = s->iw, ih = s->ih, k = s->k, st = s->stride;
    unsigned icreg = rocket_rk3576_pad_ic(ic), ocpad = rocket_rk3576_pad_oc(oc);
    unsigned ocprog = (arm == ARM_CTL) ? ocpad : oc;
    int vend = (arm == ARM_ODD_VEND || arm == ARM_EVEN_VEND || arm == ARM_AUTO_VEND);
    int odd = (arm == ARM_ODD_PAD || arm == ARM_ODD_VEND) ||
              (arm == ARM_AUTO_VEND && ((oc + 15u) / 16u) % 2u);
    unsigned ow, oh, pad, divisor = 1, scale, shift, surf_elems, c, i, y, x, kh, kw;
    size_t in_bytes, w_bytes_pad, w_bytes_vend, obytes, cb_prog, cb_pad;
    int8_t *in_plain, *w_plain, *in_cube, *w_pad, *w_vend, *out;
    int32_t *bias;
    rocket_bo in_bo = {0}, wp_bo = {0}, wv_bo = {0}, bp_bo = {0}, bc_bo = {0}, o_bo = {0},
              r_bo = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p;
    unsigned seed;
    int verbose = env_int("ROCKET_G_VERBOSE", 0), shown = 0, rc = -1, pass;

    memset(R, 0, sizeof *R);
    if (s->same) {
        ow = (iw + st - 1) / st; oh = (ih + st - 1) / st;
        pad = ((ow - 1) * st + k > iw) ? ((ow - 1) * st + k - iw) / 2 : 0;
    } else {
        ow = (iw - k) / st + 1; oh = (ih - k) / st + 1; pad = 0;
    }
    while ((double)divisor < 2.0 * sqrt((double)(ic * k * k))) divisor *= 2;
    requant_params(1.0f / (float)divisor, &scale, &shift);

    in_bytes     = (size_t)(icreg / C2) * ih * iw * C2;
    w_bytes_pad  = (size_t)(ocpad / 32u) * (icreg / 32u) * 32u * 32u * k * k;
    w_bytes_vend = (size_t)oc * icreg * k * k;
    surf_elems   = rocket_rk3576_out_surf_elems(ow, oh, 0);
    obytes       = (size_t)(ocpad / C2) * surf_elems * C2;
    cb_prog      = rocket_rk3576_coeff_bytes(ocprog);
    cb_pad       = rocket_rk3576_coeff_bytes(ocpad);

    in_plain = calloc((size_t)ic * ih * iw, 1);
    w_plain  = calloc((size_t)oc * ic * k * k, 1);
    in_cube  = calloc(in_bytes, 1);
    w_pad    = calloc(w_bytes_pad, 1);
    w_vend   = calloc(w_bytes_vend, 1);
    out      = calloc(obytes, 1);
    bias     = calloc(ocpad, sizeof *bias);
    if (!in_plain || !w_plain || !in_cube || !w_pad || !w_vend || !out || !bias) goto done;

#define INP(c_, y_, x_)      in_plain[(((size_t)(c_) * ih) + (y_)) * iw + (x_)]
#define WD(o_, i_, h_, w_)   w_plain[((((size_t)(o_) * ic + (i_)) * k + (h_)) * k + (w_))]
    seed = 0x9E3779B9u ^ (ic * 31u + oc * 17u + iw * 7u + ih * 3u + k);
    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++) {
                seed = seed * 1103515245u + 12345u;
                INP(c, y, x) = (int8_t)((int)((seed >> 16) % 61u) - 30);
            }
    for (c = 0; c < oc; c++)
        for (i = 0; i < ic; i++)
            for (kh = 0; kh < k; kh++)
                for (kw = 0; kw < k; kw++) {
                    seed = seed * 1103515245u + 12345u;
                    WD(c, i, kh, kw) = (int8_t)((int)((seed >> 16) % 17u) - 8);
                }
    for (c = 0; c < oc; c++) bias[c] = ((int)c - (int)oc / 2) * 8;

    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                in_cube[feature_index((int)ic, (int)ih, (int)iw, C2, (int)c + 1,
                                      (int)y + 1, (int)x + 1)] = INP(c, y, x);
    for (c = 0; c < oc; c++)
        for (i = 0; i < ic; i++)
            for (kh = 0; kh < k; kh++)
                for (kw = 0; kw < k; kw++) {
                    w_pad[conv_weight_index_int8((int)oc, (int)ic, (int)k, (int)k, (int)c + 1,
                                                 (int)i + 1, (int)kh + 1, (int)kw + 1)] =
                        WD(c, i, kh, kw);
                    w_vend[vend_index(oc, icreg, k, c, i, kh, kw)] = WD(c, i, kh, kw);
                }

    if (rocket_bo_alloc(fd, in_bytes, &in_bo) < 0 ||
        rocket_bo_alloc(fd, w_bytes_pad, &wp_bo) < 0 ||
        rocket_bo_alloc(fd, w_bytes_vend, &wv_bo) < 0 ||
        rocket_bo_alloc(fd, cb_prog, &bp_bo) < 0 ||
        rocket_bo_alloc(fd, cb_pad, &bc_bo) < 0 ||
        rocket_bo_alloc(fd, obytes, &o_bo) < 0 ||
        rocket_bo_alloc(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &r_bo) < 0)
        goto done;
    rocket_bo_prep(fd, &in_bo, 1, 0); memcpy(in_bo.ptr, in_cube, in_bytes); rocket_bo_fini(fd, &in_bo);
    rocket_bo_prep(fd, &wp_bo, 1, 0); memcpy(wp_bo.ptr, w_pad, w_bytes_pad); rocket_bo_fini(fd, &wp_bo);
    rocket_bo_prep(fd, &wv_bo, 1, 0); memcpy(wv_bo.ptr, w_vend, w_bytes_vend); rocket_bo_fini(fd, &wv_bo);
    /* The coefficient buffer is packed for the PROGRAMMED count: the emitter points
     * 0x5024 at the shift-word tail after ceil(ocprog/8) groups, so a buffer packed for
     * the padded count would hand an unpadded program a live C term as its shift word. */
    rocket_bo_prep(fd, &bp_bo, 1, 0);
    rocket_rk3576_pack_coeff(bp_bo.ptr, cb_prog, bias, ocprog);
    rocket_bo_fini(fd, &bp_bo);
    rocket_bo_prep(fd, &bc_bo, 1, 0);
    rocket_rk3576_pack_coeff(bc_bo.ptr, cb_pad, bias, ocpad);
    rocket_bo_fini(fd, &bc_bo);

    for (pass = 0; pass < 2; pass++) {   /* 0: the arm, 1: the padded canary */
        int is_arm = (pass == 0);
        unsigned prog_oc = is_arm ? ocprog : ocpad;
        rocket_bo *w_bo = (is_arm && vend) ? &wv_bo : &wp_bo;
        rocket_bo *b_bo = is_arm ? &bp_bo : &bc_bo;
        int ex = 0, tot = 0;

        rocket_bo_prep(fd, &o_bo, 1, 0); memset(o_bo.ptr, SENTINEL, obytes); rocket_bo_fini(fd, &o_bo);
        memset(&p, 0, sizeof p);
        p.ic = (uint16_t)icreg; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
        p.oc = (uint16_t)prog_oc; p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
        p.kh = (uint16_t)k; p.kw = (uint16_t)k;
        p.stride_y = (uint8_t)st; p.stride_x = (uint8_t)st;
        p.pad_top = (uint8_t)pad; p.pad_left = (uint8_t)pad;
        p.int8_out = 1;
        p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = (float)divisor;
        p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
        p.input_iova = in_bo.dma_address; p.weights_iova = w_bo->dma_address;
        p.bias_iova = b_bo->dma_address; p.output_iova = o_bo.dma_address;
        in_h[0] = in_bo.handle; in_h[1] = w_bo->handle; in_h[2] = b_bo->handle;
        in_h[3] = r_bo.handle; out_h[0] = o_bo.handle;

        if (is_arm && odd) {
            char spec[48];
            snprintf(spec, sizeof spec, "0x4050=0x%08x", ODD_WORD);
            setenv("ROCKET_RK3576_SET", spec, 1);
        } else {
            unsetenv("ROCKET_RK3576_SET");
        }
        rc = submit_conv(fd, &p, &r_bo, &o_bo, in_h, out_h, ih, oh, s->rows);
        unsetenv("ROCKET_RK3576_SET");
        if (is_arm) { R->tasks = rc; R->odd = is_arm && odd; }
        if (rc < 0) { printf("  %s: submit failed\n", is_arm ? "arm" : "canary"); goto done; }
        rocket_bo_prep(fd, &o_bo, 0, 2000000000ull);
        memcpy(out, o_bo.ptr, obytes);
        rocket_bo_fini(fd, &o_bo);

        for (c = 0; c < oc; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    int64_t acc = bias[c];
                    int want, got, d;
                    for (kh = 0; kh < k; kh++)
                        for (kw = 0; kw < k; kw++) {
                            int iy = (int)(y * st + kh) - (int)pad;
                            int ix = (int)(x * st + kw) - (int)pad;
                            if (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw) continue;
                            for (i = 0; i < ic; i++)
                                acc += (int64_t)INP(i, iy, ix) * WD(c, i, kh, kw);
                        }
                    want = requant_apply(acc, scale, shift);
                    got = out[out_index(surf_elems, c, y * ow + x)];
                    tot++;
                    if (got == want) { ex++; continue; }
                    if (is_arm) {
                        d = got > want ? got - want : want - got;
                        if (d > R->maxdiff) R->maxdiff = d;
                        if ((uint8_t)got == SENTINEL) R->untouched++;
                        if (verbose && shown < 6) {
                            printf("    mism c=%u y=%u x=%u want %d got %d\n", c, y, x,
                                   want, got);
                            shown++;
                        }
                    }
                }
        if (is_arm) {
            R->exact = ex; R->total = tot;
            /* Bytes the arm wrote into the padding channels' lanes. */
            for (c = oc; c < ocpad; c++)
                for (i = 0; i < ow * oh; i++)
                    if ((uint8_t)out[out_index(surf_elems, c, i)] != SENTINEL)
                        R->pad_written++;
        } else {
            R->c_exact = ex; R->c_total = tot;
        }
    }
    rc = 0;
#undef INP
#undef WD
done:
    if (r_bo.ptr)  rocket_bo_free(fd, &r_bo);
    if (o_bo.ptr)  rocket_bo_free(fd, &o_bo);
    if (bc_bo.ptr) rocket_bo_free(fd, &bc_bo);
    if (bp_bo.ptr) rocket_bo_free(fd, &bp_bo);
    if (wv_bo.ptr) rocket_bo_free(fd, &wv_bo);
    if (wp_bo.ptr) rocket_bo_free(fd, &wp_bo);
    if (in_bo.ptr) rocket_bo_free(fd, &in_bo);
    free(in_plain); free(w_plain); free(in_cube); free(w_pad); free(w_vend); free(out);
    free(bias);
    return rc;
}

int main(int argc, char **argv)
{
    int arm = -1, fd, bad = 0;
    unsigned n;
    const char *filt = argc > 2 ? argv[2] : NULL;

    if (argc > 1 && !strcmp(argv[1], "-l")) {
        for (n = 0; n < NSHAPES; n++)
            printf("%-16s ic=%u oc=%u (pad %u) %ux%u k%u s%u %s rows %u\n", shapes[n].name,
                   shapes[n].ic, shapes[n].oc, rocket_rk3576_pad_oc(shapes[n].oc),
                   shapes[n].iw, shapes[n].ih, shapes[n].k, shapes[n].stride,
                   shapes[n].same ? "SAME" : "VALID", shapes[n].rows);
        return 0;
    }
    for (n = 0; argc > 1 && n < sizeof arm_names / sizeof arm_names[0]; n++)
        if (!strcmp(argv[1], arm_names[n])) arm = (int)n;
    if (arm < 0) {
        fprintf(stderr, "usage: %s ctl|even_pad|odd_pad|odd_vend|even_vend|auto_vend [substr] | -l\n",
                argv[0]);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("no NPU\n"); return 2; }
    if (rocket_hw_current() != &rocket_hw_rk3576) {
        printf("not an RK3576\n"); rocket_close(fd); return 2;
    }
    printf("arm %s\n", arm_names[arm]);
    for (n = 0; n < NSHAPES; n++) {
        struct res R;
        const pshape_t *s = &shapes[n];
        if (filt && !strstr(s->name, filt)) continue;
        if (run_shape(fd, s, arm, &R) != 0) {
            printf("  %-16s could not run\n", s->name);
            bad = 1;
            continue;
        }
        printf("  %-16s %s t%-2d arm %5d/%-5d exact maxdiff %3d untouched %5d pad_written %4d | "
               "canary %5d/%-5d %s\n", s->name, R.odd ? "odd " : "even", R.tasks, R.exact,
               R.total, R.maxdiff, R.untouched, R.pad_written, R.c_exact, R.c_total,
               (R.exact == R.total && R.c_exact == R.c_total) ? "EXACT" :
               (R.c_exact != R.c_total ? "CANARY-WRONG" : "WRONG"));
        if (R.exact != R.total || R.c_exact != R.c_total) bad = 1;
    }
    rocket_close(fd);
    return bad;
}
