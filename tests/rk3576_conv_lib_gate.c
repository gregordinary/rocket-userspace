// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_conv_lib_gate.c — the RK3576 convolution envelope, through the LIBRARY.
 *
 * rk3576_conv_gate.c asks whether the register emitter computes. This asks whether a
 * caller can reach it: the same shape table, driven through rocket_conv2d_int8_rk3576()
 * and rocket_conv2d_dw_int8_rk3576() with row-major tensors and nothing chip-specific
 * on the caller's side. The harness there builds cubes, plans rows, submits, checks each
 * task wrote, and de-scatters; here none of that is in the test, which is the whole
 * point — every line of it moved into the library, so a pass means the chip is usable
 * rather than that the encoding is understood.
 *
 * THE ENVELOPE IS NOT THE SAME, and the difference is a result rather than a
 * convenience. A single conv program past the weight-slice cap for its output-channel
 * group count loses its trailing groups, so the emitter refuses it. The library splits
 * the output channels instead, which lowers the group count per submit and so RAISES
 * the slice the part tolerates — 144 KiB at four groups, 148 at three, 156 at two, and
 * the CBUF pool alone at one. Fourteen of the emitter's refusals are expected to compute
 * here for that reason, and one is expected to survive: 200 KiB is past the pool at a
 * single group, where there is no group count left to spend.
 *
 * ZERO POINTS are a group of their own. Every shape the emitter gate drives is
 * symmetric, and the library's public contract is not — it takes model-domain signed
 * zero points and folds them, the input's and the weight's into the coefficient buffer
 * and the output's into the DPU's OUT_CVT offset. Folded arithmetic that is never
 * exercised is a claim, so the `zp` group exercises each of the three separately and
 * together, scored against a CPU model that applies them the caller's way.
 *
 * Usage:
 *   rk3576_conv_lib_gate [group ...]   groups: envelope window surface weight dw zp
 *                                      (default: all)
 *   rk3576_conv_lib_gate -l            list without running
 *
 * Env: ROCKET_LG_FILTER=<substr>   run only shapes whose name contains this
 *      ROCKET_LG_VERBOSE=1         per-shape detail and the first mismatches
 *
 * Exit: 0 all pass, 1 a shape failed, 2 no NPU or the wrong chip (skip).
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
#include "rk3576_conv_shapes.h"

/* The zero-point group. Same driving logic, three quant parameters that the symmetric
 * table leaves at zero — run on shapes small enough that a failure is legible. */
typedef struct {
    const char *name;
    unsigned ic, oc, iw, ih, k, stride;
    int same, dw;
    int in_zp, w_zp, out_zp;
} zp_shape_t;

static const zp_shape_t ZP_SHAPES[] = {
    {"zp-in",           32,  32, 16, 16, 3, 1, 1, 0,  -7,   0,   0},
    {"zp-w",            32,  32, 16, 16, 3, 1, 1, 0,   0,  11,   0},
    {"zp-out",          32,  32, 16, 16, 3, 1, 1, 0,   0,   0,  23},
    {"zp-all",          32,  32, 16, 16, 3, 1, 1, 0, -21,  13, -9},
    {"zp-all-valid",    64,  64, 16, 16, 3, 1, 0, 0,  40, -37,  5},
    {"zp-all-k1",       32,  64, 16, 16, 1, 1, 0, 0, -128, 127, 0},
    {"zp-all-s2",       64,  32, 24, 24, 3, 2, 1, 0,  19, -5,  -3},
    {"zp-dw-in",        32,  32, 16, 16, 3, 1, 1, 1,  -7,   0,   0},
    {"zp-dw-out",       32,  32, 16, 16, 3, 1, 1, 1,   0,   0,  23},
    {"zp-dw-all",       64,  64, 16, 16, 3, 1, 1, 1,  31,   0, -12},
};
#define N_ZP ((int)(sizeof ZP_SHAPES / sizeof ZP_SHAPES[0]))

static int env_int(const char *name, int dflt)
{
    const char *e = getenv(name);
    return (e && *e) ? (int)strtol(e, NULL, 0) : dflt;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* The requant the emitter programs: out = sat8( (acc*scale + half) >> shift ). Derived
 * from the fp32 conv scale exactly as the vendor (QNNPACK) does, with shift-1 written to
 * the register, so the model shifts by the same value the DPU does. */
static void requant_params(float conv_scale, unsigned *scale, unsigned *shift_reg)
{
    union { float f; uint32_t u; } cv;
    uint32_t bits;
    cv.f = conv_scale;
    bits = cv.u;
    *shift_reg = 127u + 31u - 32u - (bits >> 23) + 16u - 1u;
    *scale = ((bits >> 9) & 0x7FFFu) + 1u;
    if (*scale < (1u << 14)) *scale |= (1u << 14);
}

static int requant_apply(int64_t acc, unsigned scale, unsigned shift_reg, int out_zp)
{
    int64_t half = shift_reg ? ((int64_t)1 << (shift_reg - 1)) : 0;
    int64_t v = ((acc * (int64_t)scale + half) >> shift_reg) + out_zp;
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int)v;
}

struct lg_stat {
    int    exact, total, maxdiff;
    double ms;
};

/* One shape, end to end, through the public per-chip entry.
 * Returns 0 exact, 1 wrong, 2 skip, 3 refused by the library. */
static int run_one(int fd, const char *name, unsigned ic, unsigned oc, unsigned iw,
                   unsigned ih, unsigned k, unsigned stride, int same, int dw,
                   unsigned max_rows, int in_zp, int w_zp, int out_zp,
                   struct lg_stat *st)
{
    rocket_conv2d_desc d = {0};
    int8_t *in = NULL, *W = NULL, *out = NULL;
    int32_t *bias = NULL;
    unsigned ow, oh, terms, divisor, scale, shift_reg, seed;
    int pad_lead, rc, shown = 0;
    int verbose = env_int("ROCKET_LG_VERBOSE", 0);
    unsigned c, y, x, kh, kw, i;

    memset(st, 0, sizeof *st);

    /* SAME follows TFLite: the output covers ceil(in/stride) and the total pad splits
     * with the smaller half leading. rocket_conv2d_desc carries one pad per axis and
     * derives the output from it, which is the same geometry. */
    if (same) {
        ow = (iw + stride - 1) / stride;
        oh = (ih + stride - 1) / stride;
        pad_lead = (int)(((ow - 1) * stride + k - iw) / 2);
        if (pad_lead < 0) pad_lead = 0;
    } else {
        if (iw < k || ih < k) return 2;
        pad_lead = 0;
    }
    d.ic = (int)ic; d.oc = (int)oc; d.ih = (int)ih; d.iw = (int)iw;
    d.kh = (int)k;  d.kw = (int)k;
    d.stride_y = (int)stride; d.stride_x = (int)stride;
    d.pad_top = pad_lead; d.pad_left = pad_lead;
    d.dil_y = 1; d.dil_x = 1;
    d.depthwise = dw;
    ow = (unsigned)rocket_conv2d_ow(&d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    if (!ow || !oh) return 2;

    /* Keep most accumulators inside the int8 output range: a surface clamped everywhere
     * hides the arithmetic under the saturation. */
    terms   = dw ? (k * k) : (ic * k * k);
    divisor = 1;
    while ((double)divisor < 2.0 * sqrt((double)terms)) divisor *= 2;
    requant_params(1.0f / (float)divisor, &scale, &shift_reg);

    in   = calloc((size_t)ic * ih * iw, 1);
    W    = calloc(dw ? (size_t)ic * k * k : (size_t)oc * ic * k * k, 1);
    out  = calloc((size_t)oc * oh * ow, 1);
    bias = calloc(oc, sizeof *bias);
    if (!in || !W || !out || !bias) { rc = 2; goto done; }

#define INP(c_, y_, x_)      in[(((size_t)(c_) * ih) + (y_)) * iw + (x_)]
#define WD(oc_, ic_, h_, w_) W[((((size_t)(oc_) * ic + (ic_)) * k + (h_)) * k + (w_))]
#define WW(c_, h_, w_)       W[(((size_t)(c_) * k + (h_)) * k + (w_))]

    /* Deterministic operands varying on every axis: a feature flat along an axis proves
     * nothing about that axis's stride. The stored values are centered on the caller's
     * zero point, so an asymmetric run is a realistic one rather than a shifted one. */
    seed = 0x9E3779B9u ^ (unsigned)(ic * 31 + oc * 17 + iw * 7 + ih * 3 + k);
    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++) {
                int v = ((int)((c * 7 + y * 13 + x * 3) % 61)) - 30 + in_zp;
                INP(c, y, x) = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
            }
    if (dw) {
        for (c = 0; c < ic; c++)
            for (kh = 0; kh < k; kh++)
                for (kw = 0; kw < k; kw++) {
                    int v;
                    seed = seed * 1103515245u + 12345u;
                    v = (int)((seed >> 16) % 17u) - 8 + w_zp;
                    WW(c, kh, kw) = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
                }
    } else {
        for (c = 0; c < oc; c++)
            for (i = 0; i < ic; i++)
                for (kh = 0; kh < k; kh++)
                    for (kw = 0; kw < k; kw++) {
                        int v;
                        seed = seed * 1103515245u + 12345u;
                        v = (int)((seed >> 16) % 17u) - 8 + w_zp;
                        WD(c, i, kh, kw) = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
                    }
    }
    /* A per-channel bias, so the BS stage carries a value the MAC cannot produce. */
    for (c = 0; c < oc; c++)
        bias[c] = (int32_t)((int)c - (int)oc / 2) * 8;

    if (max_rows) {
        char buf[16];
        snprintf(buf, sizeof buf, "%u", max_rows);
        setenv("ROCKET_RK3576_MAX_ROWS", buf, 1);
    } else {
        unsetenv("ROCKET_RK3576_MAX_ROWS");
    }

    {
        double t0 = now_ms();
        rc = dw ? rocket_conv2d_dw_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f,
                                               (float)divisor, in_zp, w_zp, out_zp, out)
                : rocket_conv2d_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f,
                                            (float)divisor, in_zp, w_zp, out_zp, out);
        st->ms = now_ms() - t0;
    }
    unsetenv("ROCKET_RK3576_MAX_ROWS");
    if (rc == ROCKET_E_UNSUPPORTED) { rc = 3; goto done; }
    if (rc != ROCKET_OK) {
        printf("  %s: the entry returned %d\n", name, rc);
        rc = 1; goto done;
    }

    /* ---- the CPU model, the caller's way ---- */
    st->total = (int)(oc * oh * ow);
    for (c = 0; c < oc; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                int64_t acc = bias[c];
                int got, want, diff;
                for (kh = 0; kh < k; kh++)
                    for (kw = 0; kw < k; kw++) {
                        int iy = (int)(y * stride + kh) - pad_lead;
                        int ix = (int)(x * stride + kw) - pad_lead;
                        if (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw) continue;
                        if (dw) {
                            acc += (int64_t)(INP(c, iy, ix) - in_zp) *
                                   (WW(c, kh, kw) - w_zp);
                        } else {
                            for (i = 0; i < ic; i++)
                                acc += (int64_t)(INP(i, iy, ix) - in_zp) *
                                       (WD(c, i, kh, kw) - w_zp);
                        }
                    }
                want = requant_apply(acc, scale, shift_reg, out_zp);
                got  = out[(((size_t)c * oh) + y) * ow + x];
                diff = got > want ? got - want : want - got;
                if (diff > st->maxdiff) st->maxdiff = diff;
                if (got == want) st->exact++;
                else if (verbose && shown < 8) {
                    printf("    mism c=%u y=%u x=%u: want %d got %d (acc %lld)\n",
                           c, y, x, want, got, (long long)acc);
                    shown++;
                }
            }
    rc = (st->exact == st->total) ? 0 : 1;

#undef INP
#undef WD
#undef WW
done:
    free(in); free(W); free(out); free(bias);
    return rc;
}

int main(int argc, char **argv)
{
    const char *filter = getenv("ROCKET_LG_FILTER");
    int fd, i, a, list = 0;
    int passed = 0, failed = 0, skipped = 0, refused = 0, wrong_refusal = 0;
    const char *groups[16];
    int ngroups = 0;

    for (a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-l")) list = 1;
        else if (!strcmp(argv[a], "all")) ngroups = 0;
        else if (ngroups < 16) groups[ngroups++] = argv[a];
    }

    if (list) {
        for (i = 0; i < N_SHAPES; i++)
            printf("%-9s %-20s ic=%-5u oc=%-4u %ux%u k%u s%u %s%s%s\n",
                   SHAPES[i].group, SHAPES[i].name, SHAPES[i].ic, SHAPES[i].oc,
                   SHAPES[i].iw, SHAPES[i].ih, SHAPES[i].k, SHAPES[i].stride,
                   SHAPES[i].same ? "SAME" : "VALID", SHAPES[i].dw ? " dw" : "",
                   SHAPES[i].lib_refuse ? "  [must refuse]" : "");
        for (i = 0; i < N_ZP; i++)
            printf("%-9s %-20s ic=%-5u oc=%-4u %ux%u k%u s%u  zp %d/%d/%d\n",
                   "zp", ZP_SHAPES[i].name, ZP_SHAPES[i].ic, ZP_SHAPES[i].oc,
                   ZP_SHAPES[i].iw, ZP_SHAPES[i].ih, ZP_SHAPES[i].k,
                   ZP_SHAPES[i].stride, ZP_SHAPES[i].in_zp, ZP_SHAPES[i].w_zp,
                   ZP_SHAPES[i].out_zp);
        return 0;
    }

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

    printf("== the RK3576 conv envelope, through the library entries ==\n");
    printf("Row-major tensors in, row-major tensors out; the cubes, the row plan, the\n"
           "output-channel split and the submit-loop retries are all the library's.\n\n");

    for (i = 0; i < N_SHAPES; i++) {
        const shape_t *s = &SHAPES[i];
        struct lg_stat st;
        int rc, want_group = (ngroups == 0);

        for (a = 0; a < ngroups; a++) if (!strcmp(groups[a], s->group)) want_group = 1;
        if (!want_group) continue;
        if (filter && *filter && !strstr(s->name, filter)) continue;

        rc = run_one(fd, s->name, s->ic, s->oc, s->iw, s->ih, s->k, s->stride,
                     s->same, s->dw, s->max_rows, 0, 0, 0, &st);
        if (rc == 2) { printf("  SKIP   %-9s %-20s\n", s->group, s->name); skipped++; continue; }
        if (rc == 3) {
            if (s->lib_refuse) {
                printf("  PASS   %-9s %-20s refused, as it must\n", s->group, s->name);
                refused++;
            } else {
                printf("  FAIL   %-9s %-20s REFUSED and should compute\n",
                       s->group, s->name);
                failed++;
            }
            continue;
        }
        if (s->lib_refuse) {
            printf("  FAIL   %-9s %-20s COMPUTED and should have refused\n",
                   s->group, s->name);
            wrong_refusal++; failed++;
            continue;
        }
        if (rc == 0) {
            printf("  PASS   %-9s %-20s %d/%d exact  %.1f ms%s\n", s->group, s->name,
                   st.exact, st.total, st.ms, s->refuse ? "   (emitter refuses this)" : "");
            passed++;
        } else {
            printf("  FAIL   %-9s %-20s %d/%d exact, maxdiff %d\n", s->group, s->name,
                   st.exact, st.total, st.maxdiff);
            failed++;
        }
    }

    {
        int want_zp = (ngroups == 0);
        for (a = 0; a < ngroups; a++) if (!strcmp(groups[a], "zp")) want_zp = 1;
        if (want_zp)
            for (i = 0; i < N_ZP; i++) {
                const zp_shape_t *z = &ZP_SHAPES[i];
                struct lg_stat st;
                int rc;
                if (filter && *filter && !strstr(z->name, filter)) continue;
                rc = run_one(fd, z->name, z->ic, z->oc, z->iw, z->ih, z->k, z->stride,
                             z->same, z->dw, 0, z->in_zp, z->w_zp, z->out_zp, &st);
                if (rc == 2) { printf("  SKIP   %-9s %-20s\n", "zp", z->name); skipped++; }
                else if (rc == 3) {
                    printf("  FAIL   %-9s %-20s REFUSED (zp %d/%d/%d)\n", "zp", z->name,
                           z->in_zp, z->w_zp, z->out_zp);
                    failed++;
                } else if (rc == 0) {
                    printf("  PASS   %-9s %-20s %d/%d exact  zp %d/%d/%d  %.1f ms\n",
                           "zp", z->name, st.exact, st.total, z->in_zp, z->w_zp,
                           z->out_zp, st.ms);
                    passed++;
                } else {
                    printf("  FAIL   %-9s %-20s %d/%d exact, maxdiff %d  zp %d/%d/%d\n",
                           "zp", z->name, st.exact, st.total, st.maxdiff,
                           z->in_zp, z->w_zp, z->out_zp);
                    failed++;
                }
            }
    }

    rocket_close(fd);
    printf("\n== %d passed, %d refused as required, %d failed, %d skipped ==\n",
           passed, refused, failed, skipped);
    if (wrong_refusal)
        printf("   (%d of those computed where the table says the part cannot)\n",
               wrong_refusal);
    return failed ? 1 : 0;
}
