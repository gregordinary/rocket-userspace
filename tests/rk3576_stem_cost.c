// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_stem_cost.c — is a convolution's EXECUTION its MACs or its feature DMA?
 *
 * A 3-channel 224x224 stem is one program of 29-81 in a graph and about a third of
 * the part's whole execution on a 7x7 one. Whether that is reducible is a question
 * about the ENCODING, and it cannot be asked until the term is attributed: the int8
 * direct datapath contracts 32 input channels because that is its MAC group, and it
 * also streams a 32-channel cube through the CBUF, so a narrower encoding would
 * shrink BOTH and neither the wall nor the submit bucket says which one it was
 * paying for.
 *
 * This separates them by crossing the two axes rather than sweeping one:
 *
 *   A — MACs vary, feature bytes FIXED. Kernel 1/3/5/7 on one 224x224 plane with
 *       SAME padding, so every arm reads the same cube and writes the same 112x112
 *       surface while the MAC count moves 49x. A pure feature-DMA bound predicts a
 *       FLAT column; a pure MAC bound predicts 49x.
 *   B — MACs FIXED, feature bytes vary 4x. `224x224 k7 s2` and `112x112 k7 s1` have
 *       the same kernel, the same channel counts and the same 112x112 output, so the
 *       same MAC count, against 1.606 MB and 0.401 MB of feature cube.
 *   C — the oc axis at fixed feature bytes, as a second, independent slope on the
 *       MAC term.
 *
 * The two real stems the graphs run are the controls.
 *
 * WHAT THE COLUMNS ARE. `wall` is the whole prepacked entry — the feature scatter,
 * the submit, the guard and the de-scatter — because that is what a caller pays.
 * The part's own time is the SUBMIT bucket inside it: run with
 * ROCKET_RK3576_INT8_PROF=1 for the split, or load the job-timing debug module
 * (rocket-userspace/tools/rk3576-job-timing-debug.patch) and read `stat_exec_us`,
 * which is the only instrument that sees inside the fence wait. `bytes` is the
 * feature cube this call's row plan actually READS — the sum of the task windows,
 * not the plane — so arm A's fixed-bytes claim is checked here rather than assumed.
 *
 * Weights are RESIDENT on EVERY arm (packed once, outside the timed loop), so no arm
 * pays the weight arithmetic and the k axis does not smuggle a host term in beside the
 * MACs. That includes the PACKED arms, which used to run the transient entry because the
 * encoding had no handle — a per-call BO teardown of 1.0-1.8 ms that no direct arm paid,
 * and enough on its own to reverse the comparison the E and F groups are for.
 *
 * THIS IS A PROBE: it reports, it does not assert. Exit 0 ran, 2 no NPU or the
 * wrong chip.
 *
 * Usage: rk3576_stem_cost [arm-substring]
 * Env:   ROCKET_SC_REPS=<n>    timed repeats after a discarded first (default 20)
 *        ROCKET_SC_CHECK=1     score each arm against a scalar CPU reference first
 *                              (slow — 1.3 GMAC single-threaded on the k7 arms)
 *        ROCKET_SC_ZP=<n>      input zero point (default 0). A real graph's stem
 *                              carries a non-zero one — ResNet-18's is -128 — and the
 *                              entry now MATERIALISES the pad columns there rather
 *                              than refusing, so a packed arm at a non-zero zero point
 *                              runs, is exact, and prices the wider program: k7 s2
 *                              oc64 measures 1.41 ms against 1.26 at zp 0, both
 *                              against the direct lowering's 3.13. The check still
 *                              splits its mismatches into windows that touch the pad
 *                              and windows that do not, which is what would name a
 *                              regression in that extension.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "requant_model.h"

#define MAX_TASKS 64

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

static void fill_i8(int8_t *p, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        p[i] = (int8_t)((int)((seed >> 16) & 0x1Fu) - 16);
    }
}

typedef struct {
    const char *group;
    const char *name;
    unsigned ic, oc, ih, iw, k, stride, pad;
    /* 1 = leave `direct_datapath` CLEAR so an ic <= 4 layer routes to the packed-image
     * first conv instead, which is the one narrower encoding this part has. Both
     * encodings are resident, so the wall column compares the same quantity. */
    unsigned packed;
    /* An explicit output extent, which is how TFLite's asymmetric SAME is expressed.
     * 0 = whatever the symmetric formula derives. */
    unsigned oh, ow;
} arm_t;

/* Every arm in A and C writes a 112x112 surface off a 224x224 plane, so the output
 * de-scatter and the guard are constant down those columns and only the MAC count
 * moves. B holds the MAC count and moves the plane. */
static const arm_t ARMS[] = {
    {"A", "k1 224 s2 oc64",      3,  64, 224, 224, 1, 2, 0, 0, 0, 0},
    {"A", "k3 224 s2 oc64",      3,  64, 224, 224, 3, 2, 1, 0, 0, 0},
    {"A", "k5 224 s2 oc64",      3,  64, 224, 224, 5, 2, 2, 0, 0, 0},
    {"A", "k7 224 s2 oc64",      3,  64, 224, 224, 7, 2, 3, 0, 0, 0},
    {"B", "k7 224 s2 oc64",      3,  64, 224, 224, 7, 2, 3, 0, 0, 0},
    {"B", "k7 112 s1 oc64",      3,  64, 112, 112, 7, 1, 3, 0, 0, 0},
    {"C", "k7 224 s2 oc16",      3,  16, 224, 224, 7, 2, 3, 0, 0, 0},
    {"C", "k7 224 s2 oc32",      3,  32, 224, 224, 7, 2, 3, 0, 0, 0},
    {"C", "k7 224 s2 oc64",      3,  64, 224, 224, 7, 2, 3, 0, 0, 0},
    /* D — the INPUT-channel axis, which is the one the cap is actually about and which
     * A and C do not touch. The feature bytes move with it, so D alone confounds the
     * two terms; it is readable only BESIDE B, which has already priced a byte. k3
     * keeps the weight slice inside `ic*kh*kw <= 4608` at every rung. */
    {"D", "k3 224 s2 oc32 ic32", 32,  32, 224, 224, 3, 2, 1, 0, 0, 0},
    {"D", "k3 224 s2 oc32 ic64", 64,  32, 224, 224, 3, 2, 1, 0, 0, 0},
    {"D", "k3 224 s2 oc32 ic128",128,  32, 224, 224, 3, 2, 1, 0, 0, 0},
    /* E — the one NARROWER ENCODING this part has, at the two stem geometries, against
     * the direct path at the same geometry. The packed-image first conv interleaves the
     * channels per pixel instead of contracting a 32-lane MAC group, so if the MAC term
     * is what the direct stem pays, this is where a cheaper stem would show up. Both
     * geometries carry a non-zero left pad, which is what that encoding requires. */
    {"E", "packed k3 s2 oc32",    3,  32, 224, 224, 3, 2, 1, 1, 0, 0},
    {"E", "direct k3 s2 oc32",    3,  32, 224, 224, 3, 2, 1, 0, 0, 0},
    {"E", "packed k7 s2 oc64",    3,  64, 224, 224, 7, 2, 3, 1, 0, 0},
    {"E", "direct k7 s2 oc64",    3,  64, 224, 224, 7, 2, 3, 0, 0, 0},
    /* F — the four graphs' OWN stem descriptors, read out of the .rnet blobs. Only
     * Inception V1's satisfies the packed-image encoding's decoded bounds (a non-zero
     * left pad, an output width of iw/stride, that width a multiple of 16, and a ZERO
     * input zero point): both MobileNets carry TFLite's `pad_left = 0` and ResNet-18
     * carries `in_zp = -128`. What Inception V1 also carries is an asymmetric SAME
     * extent — pad_before 2, pad_after 3 — which the packed encoding honours: the
     * refusal that used to stand here was the LIBRARY'S and not the part's
     * [tests/rk3576_argb_extent.c]. So this pair now prices the lever on the one graph
     * of four that can take it. */
    {"F", "IV1 stem packed",      3,  64, 224, 224, 7, 2, 2, 1, 112, 112},
    {"F", "IV1 stem direct",      3,  64, 224, 224, 7, 2, 2, 0, 112, 112},
    /* The two stems the graphs actually run, and the widened form of one of them:
     * ic 3 and ic 32 are the SAME register program on the direct datapath, so the
     * pair is the sanity check that this probe's ic axis is a host fact only. */
    {"ctl", "V1 stem k3 s2 oc32", 3,  32, 224, 224, 3, 2, 1, 0, 0, 0},
    {"ctl", "R18 stem k7 s2 oc64", 3, 64, 224, 224, 7, 2, 3, 0, 0, 0},
    {"ctl", "ic32 k7 s2 oc64",   32,  64, 224, 224, 7, 2, 3, 0, 0, 0},
};
#define N_ARMS ((int)(sizeof ARMS / sizeof ARMS[0]))

/* The programmed MAC count. The direct int8 cube is a 32-channel MAC group at every
 * count, so an ic of 3 and an ic of 32 contract the same 32 lanes — which is the
 * whole reason this question exists. */
static unsigned prog_ic(const arm_t *a)
{
    /* The packed-image encoding carries FOUR interleaved lanes per pixel, not a
     * 32-channel MAC group — which is the whole reason it is in this table. */
    return a->packed ? 4u : ((a->ic + 31u) / 32u) * 32u;
}

static double prog_macs(const arm_t *a, unsigned oh, unsigned ow)
{
    return (double)a->oc * oh * ow * prog_ic(a) * a->k * a->k;
}

/* The feature bytes this call's ROW PLAN reads: the sum of the task windows times the
 * cube's row size. A split re-reads its overlap rows, so this is above the plane. */
static int plan_feature_bytes(const arm_t *a, unsigned oh, unsigned ow,
                              unsigned *tasks, double *bytes)
{
    rocket_rk3576_row_task rt[MAX_TASKS];
    conv_params_t p;
    /* The planner routes on `ic` itself: at four or fewer it plans the packed-image
     * row size, above it the 32-lane cube's. So a packed arm must hand it the real
     * count and a direct one the count the cube programs. */
    unsigned icp = a->packed ? a->ic : ((a->ic + 31u) / 32u) * 32u;
    unsigned n = 0, t;
    double sum = 0.0;

    memset(&p, 0, sizeof p);
    p.ic = (uint16_t)icp; p.oc = (uint16_t)a->oc;
    p.ih = (uint16_t)a->ih; p.iw = (uint16_t)a->iw;
    p.oh = (uint16_t)oh;    p.ow = (uint16_t)ow;
    p.kh = (uint16_t)a->k;  p.kw = (uint16_t)a->k;
    p.stride_y = (uint8_t)a->stride; p.stride_x = (uint8_t)a->stride;
    p.dil_y = 1; p.dil_x = 1;
    p.pad_top = (uint8_t)a->pad; p.pad_left = (uint8_t)a->pad;

    if (rocket_rk3576_plan_rows(&p, 0, rt, MAX_TASKS, &n) < 0 || !n) return -1;
    for (t = 0; t < n; t++) sum += (double)rt[t].ih * a->iw * icp;
    *tasks = n;
    *bytes = sum;
    return 0;
}

/* The chip's arithmetic: an out-of-range tap reads the INPUT ZERO POINT, which the CNA
 * substitutes as a border constant, and the whole accumulation is over (x - in_zp).
 * `touches_pad` marks the outputs whose window reaches outside the plane — the set the
 * packed-image encoding's fourth bound is stated over. */
static void cpu_conv_int8(const int8_t *in, const int8_t *W, int32_t *acc,
                          uint8_t *touches_pad,
                          unsigned ic, unsigned oc, unsigned ih, unsigned iw,
                          unsigned k, unsigned stride, unsigned pad,
                          unsigned oh, unsigned ow, int in_zp)
{
    unsigned c, y, x, kh, kw, i;
    for (c = 0; c < oc; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                int32_t s = 0;
                int pad_hit = 0;
                for (kh = 0; kh < k; kh++) {
                    int iy = (int)(y * stride + kh) - (int)pad;
                    for (kw = 0; kw < k; kw++) {
                        int ix = (int)(x * stride + kw) - (int)pad;
                        int oob = (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw);
                        if (oob) pad_hit = 1;
                        for (i = 0; i < ic; i++) {
                            int v = oob ? in_zp
                                        : (int)in[((size_t)i * ih + iy) * iw + ix];
                            s += (int32_t)(v - in_zp) *
                                 W[(((size_t)c * ic + i) * k + kh) * k + kw];
                        }
                    }
                }
                acc[((size_t)c * oh + y) * ow + x] = s;
                if (touches_pad) touches_pad[(size_t)y * ow + x] = (uint8_t)pad_hit;
            }
}

/* One inference, on whichever path this arm asked for. */
static int run_once(int fd, const arm_t *a, const rocket_conv2d_desc *d,
                    rocket_conv2d_int8_weights_rk3576 *h,
                    const int8_t *in, const int8_t *W, const int32_t *bias,
                    int8_t *out, int in_zp)
{
    (void)d; (void)W; (void)bias; (void)in_zp;
    return rocket_conv2d_int8_prepacked_rk3576(fd, h, in, out);
}

/* Returns 0 ran, 1 refused, 2 out of memory. */
static int run_arm(int fd, const arm_t *a, int reps, int check, int in_zp,
                   double *ms, unsigned *tasks, double *fbytes, double *macs,
                   int *bad_border, int *bad_interior, int *n_border)
{
    uint8_t *touches = NULL;
    rocket_conv2d_desc d;
    rocket_conv2d_int8_weights_rk3576 *h = NULL;
    int8_t *in = NULL, *W = NULL, *out = NULL;
    int32_t *bias = NULL, *acc = NULL;
    unsigned oh, ow, i;
    size_t nin, nw, nout;
    double t;
    int rc = 2;

    memset(&d, 0, sizeof d);
    d.ic = (int)a->ic; d.oc = (int)a->oc; d.ih = (int)a->ih; d.iw = (int)a->iw;
    d.kh = (int)a->k;  d.kw = (int)a->k;
    d.stride_y = (int)a->stride; d.stride_x = (int)a->stride;
    d.pad_top = (int)a->pad; d.pad_left = (int)a->pad;
    d.dil_y = 1; d.dil_x = 1;
    /* The whole point: hold the layer on the direct datapath at ic 3, which is what a
     * graph's stem does, so the programmed channel count is 32 on every arm — unless
     * this arm is deliberately asking for the packed-image encoding instead. */
    d.direct_datapath = !a->packed;
    d.oh = (int)a->oh; d.ow = (int)a->ow;

    oh = (unsigned)rocket_conv2d_oh(&d);
    ow = (unsigned)rocket_conv2d_ow(&d);
    nin  = (size_t)a->ic * a->ih * a->iw;
    nw   = (size_t)a->oc * a->ic * a->k * a->k;
    nout = (size_t)a->oc * oh * ow;

    *macs = prog_macs(a, oh, ow);
    if (plan_feature_bytes(a, oh, ow, tasks, fbytes) < 0) { *tasks = 0; *fbytes = 0.0; }

    in   = calloc(nin, 1);
    W    = calloc(nw, 1);
    out  = calloc(nout, 1);
    bias = calloc(a->oc, sizeof *bias);
    if (!in || !W || !out || !bias) goto done;
    fill_i8(in, nin, 1u);
    fill_i8(W, nw, 7u);

    /* BOTH ENCODINGS ARE RESIDENT, so the wall column compares the same quantity on every
     * row. It did not used to be: the packed form had no handle, so its arms ran the
     * transient entry and carried a per-call BO teardown of 1.0-1.8 ms that no direct arm
     * paid — a difference large enough to reverse the comparison the table is for. */
    h = rocket_conv2d_int8_pack_rk3576(fd, &d, W, bias, 1.0f, 1.0f, NULL,
                                       256.0f, in_zp, 0, 0);
    if (!h) { rc = 1; goto done; }

    /* Correctness first when asked: a row window past the part's real allowance
     * writes a full WRONG surface rather than nothing, and a wrong surface is also a
     * time this probe would report as if it meant something. */
    if (check) {
        acc = calloc(nout, sizeof *acc);
        touches = calloc((size_t)oh * ow, 1);
        if (!acc || !touches) goto done;
        if (run_once(fd, a, &d, h, in, W, bias, out, in_zp) != ROCKET_OK) {
            rc = 1; goto done;
        }
        cpu_conv_int8(in, W, acc, touches, a->ic, a->oc, a->ih, a->iw, a->k,
                      a->stride, a->pad, oh, ow, in_zp);
        /* The chip's own arithmetic, not a float divide: this OUT_CVT rounds ties to
         * EVEN through a 15-bit multiplier, so a truncating reference would report a
         * mismatch on rounding noise and hide the failure this check is for. */
        for (i = 0; i < nout; i++) {
            unsigned px = i % (oh * ow);
            if ((int)out[i] == requant_scale(acc[i], 1.0f * 1.0f / 256.0f)) continue;
            if (touches[px]) (*bad_border)++; else (*bad_interior)++;
        }
        for (i = 0; i < oh * ow; i++) if (touches[i]) (*n_border)++;
        *n_border *= (int)a->oc;
    }

    for (i = 0; i < (unsigned)reps + 1u; i++) {
        double t0;
        if (i == 1) *ms = 0.0;
        t0 = now_ms();
        if (run_once(fd, a, &d, h, in, W, bias, out, in_zp) != ROCKET_OK) {
            rc = 1; goto done;
        }
        t = now_ms() - t0;
        if (i) *ms += t;
    }
    *ms /= reps;
    rc = 0;
done:
    if (h) rocket_conv2d_int8_weights_free_rk3576(fd, h);
    free(in); free(W); free(out); free(bias); free(acc); free(touches);
    return rc;
}

int main(int argc, char **argv)
{
    const char *filter = argc > 1 ? argv[1] : NULL;
    int reps = env_int("ROCKET_SC_REPS", 20);
    int check = env_int("ROCKET_SC_CHECK", 0);
    int in_zp = env_int("ROCKET_SC_ZP", 0);
    const char *group = NULL;
    int fd, i;

    fd = rocket_open();
    if (fd < 0) { printf("no /dev/accel/accel0 (%d) — SKIP\n", fd); return 2; }
    {
        const struct rocket_hw_profile *hw = rocket_hw_current();
        if (!hw || !hw->name || strcmp(hw->name, "rk3576")) {
            printf("chip is '%s', not rk3576 — SKIP\n", hw && hw->name ? hw->name : "?");
            rocket_close(fd);
            return 2;
        }
    }

    printf("RK3576 stem cost — is a convolution's execution its MACs or its feature DMA?\n"
           "%d timed repeats after a discarded first, RESIDENT weights, direct datapath\n"
           "(so every arm programs 32 input channels). `wall` is the whole prepacked\n"
           "entry; run under ROCKET_RK3576_INT8_PROF=1 for the submit bucket inside it.\n"
           "`bytes` is what the ROW PLAN reads, not the plane.\n\n", reps);
    printf("  %-3s %-22s %6s %10s %9s %9s %8s\n",
           "grp", "arm", "tasks", "MMAC", "feat KiB", "wall ms", "ns/MAC");
    printf("  %-3s %-22s %6s %10s %9s %9s %8s\n",
           "---", "----------------------", "------", "----------", "---------",
           "---------", "--------");

    for (i = 0; i < N_ARMS; i++) {
        const arm_t *a = &ARMS[i];
        double ms = 0.0, bytes = 0.0, macs = 0.0;
        unsigned tasks = 0;
        int bad_b = 0, bad_i = 0, n_b = 0, rc;
        char verdict[96];

        if (filter && !strstr(a->name, filter) && strcmp(a->group, filter)) continue;
        if (group && strcmp(group, a->group)) printf("\n");
        group = a->group;

        rc = run_arm(fd, a, reps, check, in_zp, &ms, &tasks, &bytes, &macs,
                     &bad_b, &bad_i, &n_b);
        if (rc == 1) { printf("  %-3s %-22s %6s\n", a->group, a->name, "refused"); continue; }
        if (rc == 2) { printf("  %-3s %-22s %6s\n", a->group, a->name, "no mem"); continue; }
        verdict[0] = '\0';
        if (check) {
            if (!bad_b && !bad_i) snprintf(verdict, sizeof verdict, "  exact");
            else snprintf(verdict, sizeof verdict,
                          "  WRONG: border %d/%d, interior %d", bad_b, n_b, bad_i);
        }
        printf("  %-3s %-22s %6u %10.1f %9.0f %9.3f %8.3f%s\n",
               a->group, a->name, tasks, macs / 1e6, bytes / 1024.0, ms,
               ms * 1e6 / (macs > 0.0 ? macs : 1.0), verdict);
        fflush(stdout);
    }

    printf("\nHow to read it. Down group A the feature bytes are constant and the MAC\n"
           "count moves 49x: a FLAT wall column is a feature-DMA bound, a 49x one is a\n"
           "MAC bound, and anything between is a two-term sum whose slope is the ns/MAC\n"
           "column's limit. Group B holds the MACs and moves the bytes 4x, which is the\n"
           "only arm here that prices a byte. The wall carries the host scatter, which\n"
           "is itself proportional to the plane — so read group B's SUBMIT bucket, not\n"
           "its wall.\n");

    rocket_close(fd);
    return 0;
}
