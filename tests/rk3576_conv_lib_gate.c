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
 *   rk3576_conv_lib_gate [group ...]   groups: envelope window surface weight dw zp fq
 *                                      nic fc  (default: all)
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
#include "requant_model.h"
#include "npu_regcmd_rk3576.h"
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
    /* A DEPTHWISE WEIGHT ZERO POINT, which the coefficient group cannot carry and the
     * weight cube can: every (channel, tap) has two live bytes and the datapath adds
     * them, so `w - w_zp` is exact across the pair. The group spans both signs, both
     * channel-group boundaries (32 and 64), a kernel that is not 3, and the two ends of
     * the range the pair reaches — a per-tensor TFLite depthwise filter always has one,
     * so this is the form a real network hits rather than a corner. */
    {"zp-dw-w",         32,  32, 16, 16, 3, 1, 1, 1,   0,  83,   0},
    {"zp-dw-w-neg",     32,  32, 16, 16, 3, 1, 1, 1,   0, -34,   0},
    {"zp-dw-w-all",     64,  64, 16, 16, 3, 1, 1, 1, -128, 45, -12},
    {"zp-dw-w-g64",    128, 128, 14, 14, 3, 1, 1, 1, -128, -18,  7},
    {"zp-dw-w-k5",      64,  64, 16, 16, 5, 1, 1, 1,  -9,  120, -3},
    {"zp-dw-w-s2",      64,  64, 24, 24, 3, 2, 1, 1,  12, -127,  5},
};
#define N_ZP ((int)(sizeof ZP_SHAPES / sizeof ZP_SHAPES[0]))

/* The QUANTIZED FIRST CONV group. Same CNA sub-encoding as the fp16 one below, a
 * different weight cube, and — uniquely on this path — two GEOMETRY bounds that are
 * silent when violated, so the group asserts both directions:
 *
 *   the left pad must be NON-ZERO       at zero the DPU writes an untouched surface
 *   the output width must be iw/stride  anything else writes a surface sheared by one
 *                                       column per row
 *
 * ONE image channel is NOT one of them, though the emitted program will not fetch it:
 * the entry programs a single-channel image as two, a zero lane against zero weights,
 * so `fq-gray-*` must compute rather than refuse.
 *
 * Padding here is ONNX-style symmetric (k-1)/2 rather than the TFLite SAME the rest of
 * the gate uses, because TFLite puts the odd pad byte on the TRAILING edge and a 3x3
 * stride-2 stem then has pad_left = 0 — the one case the part will not compute. That
 * difference is the whole reason this group carries its own pad column. */
typedef struct {
    const char *name;
    unsigned ic, oc, iw, ih, k, stride, pad;
    int refuse;            /* the measured boundary says this one must be refused */
} fq_shape_t;

static const fq_shape_t FQ_SHAPES[] = {
    {"fq-rgb-k3",        3, 32,  32,  32, 3, 1, 1, 0}, /* the shape the cube was decoded at */
    {"fq-rgb-k3-oc64",   3, 64,  32,  32, 3, 1, 1, 0}, /* two output-channel groups        */
    {"fq-rgb-k3-oc96",   3, 96,  32,  32, 3, 1, 1, 0}, /* past one program's 64            */
    {"fq-rgba-k3",       4, 32,  32,  32, 3, 1, 1, 0}, /* the fourth lane is the image's   */
    {"fq-gray-k3",       1, 32,  32,  32, 3, 1, 1, 0}, /* one channel, programmed as two   */
    {"fq-gray-s2",       1, 32,  64,  64, 3, 2, 1, 0}, /* and through the row window       */
    {"fq-2ch-k3",        2, 32,  32,  32, 3, 1, 1, 0},
    {"fq-rgb-k5",        3, 32,  32,  32, 5, 1, 2, 0}, /* R = round16(4*kw) becomes 32     */
    {"fq-rgb-k7",        3, 32,  32,  32, 7, 1, 3, 0},
    {"fq-rgb-s2",        3, 32,  32,  32, 3, 2, 1, 0}, /* the vision stem's stride         */
    {"fq-rgb-s2-k7",     3, 32,  32,  32, 7, 2, 3, 0},
    {"fq-rgb-16w",       3, 32,  16,  32, 3, 1, 1, 0}, /* iw alone                         */
    {"fq-rgb-64w",       3, 32,  64,  32, 3, 1, 1, 0},
    {"fq-rgb-r48",       3, 32,  32,  48, 3, 1, 1, 0}, /* ih alone                         */
    {"fq-rgb-112",       3, 32, 112, 112, 3, 2, 1, 1}, /* ow 56: the output granule        */
    /* The output-width axis on its own: iw is a multiple of 16 in all of these and ow
     * is not always, which is the axis the plane shapes above cannot separate. */
    {"fq-rgb-112-s1",    3, 32, 112, 112, 3, 1, 1, 0}, /* ow 112                           */
    {"fq-rgb-96-s2",     3, 32,  96,  96, 3, 2, 1, 0}, /* ow 48                            */
    {"fq-rgb-128-s2",    3, 32, 128, 128, 3, 2, 1, 0}, /* ow 64                            */
    {"fq-rgb-160-s2",    3, 32, 160, 160, 3, 2, 1, 0}, /* ow 80                            */
    {"fq-rgb-48w-s2",    3, 32,  48,  32, 3, 2, 1, 1}, /* ow 24: refused for the same       */
    {"fq-rgb-192-s2",    3, 32, 192, 192, 3, 2, 1, 0}, /* ow 96, a stem-sized plane        */
    {"fq-rgb-224",       3, 32, 224, 224, 3, 2, 1, 0}, /* the stem: needs the row window   */
    {"fq-rgb-224-oc64",  3, 64, 224, 224, 3, 2, 1, 0}, /* rows AND output channels         */
    {"fq-rgb-224-k7",    3, 32, 224, 224, 7, 2, 3, 0}, /* a k7 stem, deeper windows        */
    /* The boundary, in both directions. */
    {"fq-rgb-valid",     3, 32,  32,  32, 3, 1, 0, 1}, /* pad_left 0: writes NOTHING       */
    {"fq-rgb-k1",        3, 32,  32,  32, 1, 1, 0, 1}, /* k=1 cannot have a leading pad    */
    {"fq-rgb-oc16",      3, 16,  32,  32, 3, 1, 1, 1}, /* a partial 32-channel group       */
    {"fq-rgb-iw24",      3, 32,  24,  32, 3, 1, 1, 1}, /* iw not a multiple of 16          */
};
#define N_FQ ((int)(sizeof FQ_SHAPES / sizeof FQ_SHAPES[0]))

/* The NARROW-IC group: four or fewer input channels on the DIRECT datapath, which is
 * what rocket_conv2d_desc.direct_datapath asks for.
 *
 * The int8 direct cube is a 32-channel MAC group at every count — rocket_rk3576_pad_ic(3)
 * and _pad_ic(32) are both 32 — so the register program, the weight-cube size and the
 * feature-cube size at ic 3 are the ones at ic 32, and channels 3..31 are the cube's own
 * zero padding. What could still be wrong is the ARITHMETIC: the zero-point fold is over
 * the LIVE tap count, ic*kh*kw, and getting that count from the register value instead
 * would be wrong by in_zp*w_zp*(32-ic)*kh*kw — invisible at a symmetric quantization,
 * which is why this group carries zero points rather than only shapes.
 *
 * It also carries the geometry the PACKED-IMAGE path refuses, because that is the reason
 * to want this at all: a zero leading pad, an output width that is not iw/stride, and one
 * that is not a multiple of 16. And `ext` asks for TFLite's own SAME extent, so a TFLite
 * stem is gated here directly rather than by resemblance. */
typedef struct {
    const char *name;
    unsigned ic, oc, iw, ih, k, stride, pad;
    int ext;                       /* TFLite's SAME extent, so the trailing pad is derived */
    int in_zp, w_zp, out_zp;
} nic_shape_t;

static const nic_shape_t NIC_SHAPES[] = {
    {"nic-rgb-k3",       3, 32,  32,  32, 3, 1, 1, 0,   0,   0,   0},
    {"nic-rgb-k3-valid", 3, 32,  32,  32, 3, 1, 0, 0,   0,   0,   0}, /* pad 0: fq refuses */
    {"nic-rgb-k1",       3, 32,  32,  32, 1, 1, 0, 0,   0,   0,   0}, /* fq refuses        */
    {"nic-rgb-iw24",     3, 32,  24,  32, 3, 1, 1, 0,   0,   0,   0}, /* iw%16: fq refuses */
    {"nic-rgb-oc16",     3, 16,  32,  32, 3, 1, 1, 0,   0,   0,   0}, /* fq refuses        */
    {"nic-gray-k3",      1, 32,  32,  32, 3, 1, 1, 0,   0,   0,   0}, /* one live channel  */
    {"nic-2ch-k3",       2, 32,  32,  32, 3, 1, 1, 0,   0,   0,   0},
    {"nic-rgba-k3",      4, 32,  32,  32, 3, 1, 1, 0,   0,   0,   0},
    {"nic-rgb-k5",       3, 32,  32,  32, 5, 1, 2, 0,   0,   0,   0},
    {"nic-rgb-k7",       3, 32,  32,  32, 7, 1, 3, 0,   0,   0,   0},
    {"nic-rgb-s2",       3, 32,  32,  32, 3, 2, 1, 0,   0,   0,   0},
    {"nic-rgb-oc96",     3, 96,  32,  32, 3, 1, 1, 0,   0,   0,   0}, /* an oc split       */
    /* THE ZERO POINTS, which is where a tap count taken from the register value shows. */
    {"nic-zp-in",        3, 32,  32,  32, 3, 1, 1, 0, -17,   0,   0},
    {"nic-zp-w",         3, 32,  32,  32, 3, 1, 1, 0,   0,  23,   0},
    {"nic-zp-all",       3, 32,  32,  32, 3, 1, 1, 0, -21,  13,  -9},
    {"nic-zp-all-k1",    3, 64,  32,  32, 1, 1, 0, 0, -128, 127,  0},
    {"nic-zp-gray",      1, 32,  32,  32, 3, 1, 1, 0,  40, -37,   5},
    /* ic ABOVE FOUR, which already reached the direct path before the flag existed and
     * is padded to the same 32-channel group. These say whether a padded channel count
     * with both zero points is a property of the group or of the packed-image boundary:
     * if the pad substitution covers the whole PROGRAMMED group, every one of these is
     * wrong on its border too and the flag merely found it. */
    {"nic-ic8-zp",       8, 32,  32,  32, 3, 1, 1, 0, -21,  13,  -9},
    {"nic-ic16-zp",     16, 32,  32,  32, 3, 1, 1, 0, -21,  13,  -9},
    {"nic-ic24-zp",     24, 32,  32,  32, 3, 1, 1, 0, -21,  13,  -9},
    /* THE TFLITE STEM's own geometry: an even plane at stride two, so the leading pad is
     * zero and the extent is one larger than it derives. The packed-image path cannot
     * express this at all — it is the whole reason the flag exists. */
    {"nic-tfl-s2",       3, 32,  32,  32, 3, 2, 0, 1,   0,   0,   0},
    {"nic-tfl-s2-zp",    3, 32,  32,  32, 3, 2, 0, 1, -31,  11,  -5},
    {"nic-tfl-224",      3, 32, 224, 224, 3, 2, 0, 1,   0,   0,   0}, /* the stem, rows too */
    {"nic-tfl-224-oc64", 3, 64, 224, 224, 3, 2, 0, 1, -21,   0,   7},
};
#define N_NIC ((int)(sizeof NIC_SHAPES / sizeof NIC_SHAPES[0]))

/* The FIRST CONV group at fp16. A packed image of four or fewer channels runs the
 * CNA's own sub-encoding; this group is the float form of it, and the int8 form is
 * the group above.
 *
 * Operands are small INTEGERS held in fp16. Every partial sum is then exact in the
 * DPU's fp32 accumulator whatever order it reduces in, so the one rounding is the
 * final narrowing to fp16 and both sides do it to the same value — which makes
 * "bit-exact" a claim about the datapath rather than about a tolerance. */
typedef struct {
    const char *name;
    unsigned ic, oc, iw, ih, k, stride;
    int same;
} fc_shape_t;

static const fc_shape_t FC_SHAPES[] = {
    {"fc-rgb-k3",        3, 16, 32, 32, 3, 1, 1},   /* the shape the cube was decoded at */
    {"fc-rgb-k3-oc32",   3, 32, 32, 32, 3, 1, 1},   /* a second oc group                 */
    {"fc-rgb-k3-oc48",   3, 48, 32, 32, 3, 1, 1},   /* three                             */
    {"fc-rgb-k3-oc64",   3, 64, 32, 32, 3, 1, 1},   /* four                              */
    {"fc-rgb-k3-oc96",   3, 96, 32, 32, 3, 1, 1},   /* six                               */
    {"fc-rgb-k3-oc24",   3, 24, 32, 32, 3, 1, 1},   /* a partial group                   */
    {"fc-rgb-k1",        3, 16, 32, 32, 1, 1, 0},   /* no tap axis to get wrong          */
    {"fc-rgb-k5",        3, 16, 32, 32, 5, 1, 1},
    {"fc-rgb-k7",        3, 16, 32, 32, 7, 1, 1},   /* past every captured kernel        */
    {"fc-rgba-k3",       4, 16, 32, 32, 3, 1, 1},   /* the fourth lane is the image's    */
    {"fc-gray-k3",       1, 16, 32, 32, 3, 1, 1},   /* one live lane, three don't-care   */
    {"fc-2ch-k3",        2, 16, 32, 32, 3, 1, 1},   /* the count the vendor compiles direct */
    {"fc-rgb-s2",        3, 32, 32, 32, 3, 2, 1},   /* the vision stem's stride          */
    {"fc-rgb-s2-k7",     3, 16, 32, 32, 7, 2, 1},
    {"fc-rgb-valid",     3, 16, 32, 32, 3, 1, 0},   /* no border pad                     */
    {"fc-rgb-16w",       3, 16, 16, 32, 3, 1, 1},   /* iw alone                          */
    {"fc-rgb-64w",       3, 16, 64, 32, 3, 1, 1},
    {"fc-rgb-r48",       3, 16, 32, 48, 3, 1, 1},   /* ih alone                          */
    {"fc-rgb-112",       3, 32, 112, 112, 3, 2, 1}, /* a stem-sized plane                */
    {"fc-rgb-128",       3, 16, 128, 128, 3, 1, 1},
    {"fc-rgb-224",       3, 32, 224, 224, 3, 2, 1}, /* the stem: needs the row window     */
    {"fc-rgb-224-oc64",  3, 64, 224, 224, 3, 2, 1}, /* rows AND output channels together  */
    {"fc-rgb-224-k7",    3, 32, 224, 224, 7, 2, 1}, /* a k7 stem, deeper windows          */
};
#define N_FC ((int)(sizeof FC_SHAPES / sizeof FC_SHAPES[0]))

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


struct lg_stat {
    int    exact, total, maxdiff;
    int    chans;      /* output channels exact over their whole plane */
    double ms;
    /* What a wrong element LOOKS like, which is what separates the two hazards this
     * path carries. An atom the DPU never emitted reads as the calloc'd zero, so a
     * failure whose wrong elements are all zero-where-a-value-was-wanted is a DROP;
     * one with wrong non-zero values is arithmetic. Both counted, plus the span, so
     * an intermittent names itself rather than being re-run blind. */
    int    wrong_zero, wrong_val;
    int    span_c0, span_c1, span_y0, span_y1, span_x0, span_x1;
};

/* Every shape that failed, so a capture of the tail names them. An intermittent that
 * shows up once in a few hundred shapes is otherwise a count. */
#define LG_MAX_FAILED 64
static char lg_failed[LG_MAX_FAILED][96];
static int  lg_nfailed;
/* How many shapes went through the resident A/B, so a run that silently skipped it cannot
 * read as one that passed it. */
static int  lg_resident_ab;

static void lg_note_failure(const char *group, const char *name,
                            const struct lg_stat *st, const char *reason)
{
    if (lg_nfailed >= LG_MAX_FAILED) { lg_nfailed++; return; }
    if (st && st->total)
        snprintf(lg_failed[lg_nfailed], sizeof lg_failed[0],
                 "%s/%s  %d/%d exact, %d dropped-to-zero, %d wrong-valued",
                 group, name, st->exact, st->total, st->wrong_zero, st->wrong_val);
    else
        snprintf(lg_failed[lg_nfailed], sizeof lg_failed[0], "%s/%s  %s",
                 group, name, reason ? reason : "failed");
    lg_nfailed++;
}

/* Record one wrong element: which kind, and how far the damage spreads. */
static void lg_wrong(struct lg_stat *st, int got_is_zero, unsigned c, unsigned y, unsigned x)
{
    if (got_is_zero) st->wrong_zero++; else st->wrong_val++;
    if (st->wrong_zero + st->wrong_val == 1) {
        st->span_c0 = st->span_c1 = (int)c;
        st->span_y0 = st->span_y1 = (int)y;
        st->span_x0 = st->span_x1 = (int)x;
        return;
    }
    if ((int)c < st->span_c0) st->span_c0 = (int)c;
    if ((int)c > st->span_c1) st->span_c1 = (int)c;
    if ((int)y < st->span_y0) st->span_y0 = (int)y;
    if ((int)y > st->span_y1) st->span_y1 = (int)y;
    if ((int)x < st->span_x0) st->span_x0 = (int)x;
    if ((int)x > st->span_x1) st->span_x1 = (int)x;
}

/* The one line a chase needs: what kind of wrong, and where. */
static void lg_print_signature(const struct lg_stat *st)
{
    if (!st->wrong_zero && !st->wrong_val) return;
    printf("         %d dropped to zero, %d wrong-valued; c %d..%d y %d..%d x %d..%d\n",
           st->wrong_zero, st->wrong_val, st->span_c0, st->span_c1,
           st->span_y0, st->span_y1, st->span_x0, st->span_x1);
}

/* One shape, end to end, through the public per-chip entry.
 * `direct` sets rocket_conv2d_desc.direct_datapath; `extent` asks for TFLite's SAME
 * output extent (ceil(in/stride)) rather than the one the leading pad derives, which is
 * how an asymmetric pad is expressed.
 * Returns 0 exact, 1 wrong, 2 skip, 3 refused by the library. */
static int run_one_pad(int fd, const char *name, unsigned ic, unsigned oc, unsigned iw,
                       unsigned ih, unsigned k, unsigned stride, int same, int dw,
                       unsigned max_rows, int in_zp, int w_zp, int out_zp,
                       int pad_override, int direct, int extent, struct lg_stat *st)
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
    /* The first conv wants ONNX-style symmetric padding rather than TFLite's
     * smaller-half-leading SAME, because its measured bound is on the LEADING pad. */
    if (pad_override >= 0) pad_lead = pad_override;
    d.ic = (int)ic; d.oc = (int)oc; d.ih = (int)ih; d.iw = (int)iw;
    d.kh = (int)k;  d.kw = (int)k;
    d.stride_y = (int)stride; d.stride_x = (int)stride;
    d.pad_top = pad_lead; d.pad_left = pad_lead;
    d.dil_y = 1; d.dil_x = 1;
    d.depthwise = dw;
    d.direct_datapath = direct;
    /* TFLite's SAME at an even plane and stride two: the odd row and column go at the END,
     * so the leading pad is zero and the extent is one larger than that pad derives. The
     * CNA takes the pad its last window CONSUMES, so the extent is all it needs told. */
    if (extent) {
        d.oh = (int)((ih + stride - 1) / stride);
        d.ow = (int)((iw + stride - 1) / stride);
    }
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
                want = requant_apply_zp(acc, scale, shift_reg, out_zp);
                got  = out[(((size_t)c * oh) + y) * ow + x];
                diff = got > want ? got - want : want - got;
                if (diff > st->maxdiff) st->maxdiff = diff;
                if (got == want) st->exact++;
                else {
                    lg_wrong(st, got == 0, c, y, x);
                    if (verbose && shown < 8) {
                        printf("    mism c=%u y=%u x=%u: want %d got %d (acc %lld)\n",
                               c, y, x, want, got, (long long)acc);
                        shown++;
                    }
                }
            }
    rc = (st->exact == st->total) ? 0 : 1;

    /* ---- the RESIDENT path, against the transient one element for element ----
     *
     * The two share their arithmetic by construction (one packing routine, one submit
     * loop), so what this actually asks is whether HOLDING the operands across calls is
     * safe: the feature cube is filled without its zeroing memset from the second call on,
     * and each tile's output surface carries the previous call's result when the DPU starts
     * writing. TWICE for that reason — a single prepacked call would not notice either.
     *
     * ROCKET_LG_RESIDENT=1. Off by default so the shape table's own timings stay
     * comparable run to run. */
    if (!rc && env_int("ROCKET_LG_RESIDENT", 0) &&
        /* ic<=4 without the flag is the packed-image conv; no handle packs that cube. */
        (dw || ic > 4 || direct)) {
        rocket_conv2d_int8_weights_rk3576 *h;
        int8_t *out2 = calloc((size_t)oc * oh * ow, 1);
        int pass;

        lg_resident_ab++;

        if (!out2) { rc = 2; goto done; }
        h = rocket_conv2d_int8_pack_rk3576(fd, &d, W, bias, 1.0f, 1.0f, NULL,
                                           (float)divisor, in_zp, w_zp, out_zp);
        if (!h) {
            printf("  %s: the transient call computed but pack refused\n", name);
            free(out2); rc = 1; goto done;
        }
        for (pass = 0; pass < 2 && !rc; pass++) {
            int prc;
            memset(out2, 0x5A, (size_t)oc * oh * ow);
            prc = rocket_conv2d_int8_prepacked_rk3576(fd, h, in, out2);
            if (prc != ROCKET_OK) {
                printf("  %s: prepacked pass %d returned %d\n", name, pass + 1, prc);
                rc = 1;
            } else if (memcmp(out, out2, (size_t)oc * oh * ow)) {
                size_t n = (size_t)oc * oh * ow, j, bad = 0, first = 0;
                for (j = 0; j < n; j++)
                    if (out[j] != out2[j]) { if (!bad) first = j; bad++; }
                printf("  %s: resident pass %d differs from transient on %zu of %zu "
                       "elements, first at %zu (transient %d, resident %d)\n",
                       name, pass + 1, bad, n, first, out[first], out2[first]);
                rc = 1;
            }
        }
        rocket_conv2d_int8_weights_free_rk3576(fd, h);
        free(out2);
    }

#undef INP
#undef WD
#undef WW
done:
    free(in); free(W); free(out); free(bias);
    return rc;
}

static int run_one(int fd, const char *name, unsigned ic, unsigned oc, unsigned iw,
                   unsigned ih, unsigned k, unsigned stride, int same, int dw,
                   unsigned max_rows, int in_zp, int w_zp, int out_zp,
                   struct lg_stat *st)
{
    return run_one_pad(fd, name, ic, oc, iw, ih, k, stride, same, dw, max_rows,
                       in_zp, w_zp, out_zp, -1, 0, 0, st);
}

/* One first-conv shape, through rocket_conv2d_fp16_rk3576() with row-major CHW fp16
 * tensors — the library owns the packed image, the lane cube and the de-scatter. */
static int run_fc(int fd, const fc_shape_t *s, struct lg_stat *st)
{
    rocket_conv2d_desc d = {0};
    _Float16 *in = NULL, *W = NULL, *out = NULL;
    unsigned ow, oh, seed, c, y, x, kh, kw, i;
    int pad_lead, rc, shown = 0;
    int verbose = env_int("ROCKET_LG_VERBOSE", 0);
    unsigned ic = s->ic, oc = s->oc, iw = s->iw, ih = s->ih, k = s->k, stride = s->stride;

    memset(st, 0, sizeof *st);

    if (s->same) {
        ow = (iw + stride - 1) / stride;
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
    ow = (unsigned)rocket_conv2d_ow(&d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    if (!ow || !oh) return 2;

    in  = calloc((size_t)ic * ih * iw, sizeof *in);
    W   = calloc((size_t)oc * ic * k * k, sizeof *W);
    out = calloc((size_t)oc * oh * ow, sizeof *out);
    if (!in || !W || !out) { rc = 2; goto done; }

#define FINP(c_, y_, x_)      in[(((size_t)(c_) * ih) + (y_)) * iw + (x_)]
#define FWD(oc_, ic_, h_, w_) W[((((size_t)(oc_) * ic + (ic_)) * k + (h_)) * k + (w_))]

    seed = 0x9E3779B9u ^ (unsigned)(ic * 31 + oc * 17 + iw * 7 + ih * 3 + k);
    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                FINP(c, y, x) = (_Float16)(int)(((c * 7 + y * 13 + x * 3) % 61) - 30);
    for (c = 0; c < oc; c++)
        for (i = 0; i < ic; i++)
            for (kh = 0; kh < k; kh++)
                for (kw = 0; kw < k; kw++) {
                    seed = seed * 1103515245u + 12345u;
                    FWD(c, i, kh, kw) = (_Float16)((int)((seed >> 16) % 17u) - 8);
                }

    {
        double t0 = now_ms();
        rc = rocket_conv2d_fp16_rk3576(fd, &d, in, W, out);
        st->ms = now_ms() - t0;
    }
    if (rc == ROCKET_E_UNSUPPORTED) { rc = 3; goto done; }
    if (rc != ROCKET_OK) {
        printf("  %s: the entry returned %d\n", s->name, rc);
        rc = 1; goto done;
    }

    st->total = (int)(oc * oh * ow);
    for (c = 0; c < oc; c++) {
        int chan_exact = 1;
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                float acc = 0.0f;
                _Float16 want, got;
                for (kh = 0; kh < k; kh++)
                    for (kw = 0; kw < k; kw++) {
                        int iy = (int)(y * stride + kh) - pad_lead;
                        int ix = (int)(x * stride + kw) - pad_lead;
                        if (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw) continue;
                        for (i = 0; i < ic; i++)
                            acc += (float)FINP(i, iy, ix) * (float)FWD(c, i, kh, kw);
                    }
                want = (_Float16)acc;
                got  = out[(((size_t)c * oh) + y) * ow + x];
                if (got == want) st->exact++;
                else {
                    float dv = (float)got - (float)want;
                    int dq = (int)(dv < 0 ? -dv : dv);
                    chan_exact = 0;
                    if (dq > st->maxdiff) st->maxdiff = dq;
                    lg_wrong(st, (float)got == 0.0f, c, y, x);
                    if (verbose && shown < 8) {
                        printf("    mism c=%u y=%u x=%u: want %g got %g\n",
                               c, y, x, (double)(float)want, (double)(float)got);
                        shown++;
                    }
                }
            }
        if (chan_exact) st->chans++;
    }
    rc = (st->exact == st->total) ? 0 : 1;

#undef FINP
#undef FWD
done:
    free(in); free(W); free(out);
    return rc;
}

/* ---- the row allowance the part actually honours ---------------------------------
 * rocket_rk3576_max_task_rows() says how many INPUT rows one row task may hold, and
 * every shape that has ever gated it has a kernel of 1, 3 or 5. A 7x7 stem does not:
 * the planner's answer is too large there, the task past the real allowance writes
 * NOTHING at all — deterministically, and with the whole surface guard's eight power
 * cycles spent on it — and the conv falls back to the host.
 *
 * So this measures the allowance instead of asserting it. For each geometry it BISECTS
 * the forced cap and reports the largest value that computes bit-exactly, next to what
 * the planner predicted. Bisection assumes only that a smaller window is never worse,
 * which is what a capacity bound means; a shape whose two agree costs one run, and a
 * failing one costs eight power cycles per probe, so a linear walk is not affordable.
 */
struct rowbound_case { const char *name; unsigned ic, oc, iw, ih, k, stride; int same; };

static const struct rowbound_case ROWBOUND[] = {
    /* The ResNet-18 stem, as the graph runs it (three image channels widened to eight)
     * and with a full 32 behind it, so the deficit can be read against `entries`. */
    { "stem-k7-ic8",     8,  64, 224, 224, 7, 2, 1 },
    { "stem-k7-ic32",   32,  64, 224, 224, 7, 2, 1 },
    { "k7-s1-112",      32,  32, 112, 112, 7, 1, 1 },
    { "k7-s1-56-ic64",  64,  64,  56,  56, 7, 1, 1 },
    { "k5-s1-112",      32,  32, 112, 112, 5, 1, 1 },
    { "k3-s1-112",      32,  32, 112, 112, 3, 1, 1 },
    { "k3-s1-224",      32,  32, 224, 224, 3, 1, 1 },
    { "k1-s1-224",      32,  32, 224, 224, 1, 1, 0 },
    /* The axes that separate the stem from every shape above it, one at a time: the
     * STRIDE, the output-channel group count, the plane width, and the kernel. */
    { "k3-s2-224",      32,  32, 224, 224, 3, 2, 1 },
    { "k5-s2-224",      32,  32, 224, 224, 5, 2, 1 },
    { "k7-s2-224-oc32", 32,  32, 224, 224, 7, 2, 1 },
    { "k7-s1-224",      32,  32, 224, 224, 7, 1, 1 },
    { "k7-s2-112",      32,  32, 112, 112, 7, 2, 1 },
};
#define N_ROWBOUND ((int)(sizeof ROWBOUND / sizeof *ROWBOUND))

static int rowbound(int fd)
{
    int i, bad = 0;

    printf("rowbound: the largest forced row cap that computes, against the planner's\n"
           "          own answer. `entries` is the feature row's granule count.\n");
    for (i = 0; i < N_ROWBOUND; i++) {
        const struct rowbound_case *s = &ROWBOUND[i];
        unsigned predicted = rocket_rk3576_max_task_rows(s->iw, s->ic, s->oc,
                                                         s->k, s->k, 0);
        unsigned entries = (s->iw * s->ic + 63u) / 64u;
        unsigned cap, measured = 0;
        struct lg_stat st;

        if (!predicted) { printf("  %-16s the planner refuses the shape\n", s->name);
                          continue; }
        {
            unsigned lo = s->k, hi = predicted;
            int rc = run_one(fd, s->name, s->ic, s->oc, s->iw, s->ih, s->k, s->stride,
                             s->same, 0, hi, 0, 0, 0, &st);
            /* A BISECTION OVER AN INTERMITTENT FAILURE REPORTS THE WRONG BOUND, and the
             * atom-drop hazard makes one here: a single unlucky probe at the top sends
             * the search down and it never comes back. Two shapes have swung 4.5x and
             * 1.6x between runs this way. So the top of the range is probed TWICE before
             * the search starts, and only a twice-confirmed failure is believed. */
            if (rc != 0)
                rc = run_one(fd, s->name, s->ic, s->oc, s->iw, s->ih, s->k, s->stride,
                             s->same, 0, hi, 0, 0, 0, &st);
            if (rc == 0) measured = hi;
            else while (lo <= hi) {                       /* the largest cap that works */
                cap = lo + (hi - lo) / 2u;
                rc = run_one(fd, s->name, s->ic, s->oc, s->iw, s->ih, s->k, s->stride,
                             s->same, 0, cap, 0, 0, 0, &st);
                if (rc == 0) { measured = cap; lo = cap + 1u; }
                else { if (cap == s->k) break; hi = cap - 1u; }
            }
        }
        printf("  %-16s ic=%-3u oc=%-3u %ux%u k%u s%u  entries %-4u  planner %-4u  "
               "part %-4u  %s\n", s->name, s->ic, s->oc, s->iw, s->ih, s->k, s->stride,
               entries, predicted, measured,
               measured == predicted ? "agree"
               : measured ? "THE PLANNER IS OVER" : "nothing computed");
        if (measured != predicted) bad++;
    }
    /* THIS MEASURES; IT DOES NOT ASSERT. One geometry is KNOWN to be over and the
     * shipped bound deliberately does not reach it — `56x56 k7 ic 64 oc 64` has a weight
     * cube past the CBUF pool at F=0, where charging every group would refuse a shape
     * that computes, so it keeps the one-slice allowance. Failing here would make that
     * open item a standing red rather than a tracked one; what asserts the allowance is
     * the conv and net gates, which run real shapes. */
    printf("== %d geometr%s where the planner is over; this probe MEASURES, the "
           "assertion is in the conv and net gates ==\n", bad, bad == 1 ? "y" : "ies");
    return 0;
}

int main(int argc, char **argv)
{
    const char *filter = getenv("ROCKET_LG_FILTER");
    int fd, i, a, list = 0;

    /* The library logs to stderr and this gate reports on stdout. Redirected to one
     * file, a block-buffered stdout reorders the two and a diagnostic ends up filed
     * under the wrong shape — which is exactly what an intermittent needs to survive. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    int passed = 0, failed = 0, skipped = 0, refused = 0, wrong_refusal = 0;
    const char *groups[16];
    int ngroups = 0;

    int want_rowbound = 0;
    for (a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-l")) list = 1;
        else if (!strcmp(argv[a], "rowbound")) want_rowbound = 1;
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

    if (want_rowbound) {
        int rc = rowbound(fd);
        rocket_close(fd);
        return rc;
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
                lg_note_failure(s->group, s->name, NULL, "REFUSED and should compute");
                failed++;
            }
            continue;
        }
        if (s->lib_refuse) {
            printf("  FAIL   %-9s %-20s COMPUTED and should have refused\n",
                   s->group, s->name);
            lg_note_failure(s->group, s->name, NULL, "COMPUTED and should have refused");
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
            lg_print_signature(&st);
            lg_note_failure(s->group, s->name, &st, NULL);
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
                    lg_note_failure("zp", z->name, NULL, "REFUSED");
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
                    lg_print_signature(&st);
                    lg_note_failure("zp", z->name, &st, NULL);
                    failed++;
                }
            }
    }

    {
        int want_fq = (ngroups == 0);
        for (a = 0; a < ngroups; a++) if (!strcmp(groups[a], "fq")) want_fq = 1;
        if (want_fq)
            for (i = 0; i < N_FQ; i++) {
                const fq_shape_t *q = &FQ_SHAPES[i];
                struct lg_stat st;
                int rc;
                if (filter && *filter && !strstr(q->name, filter)) continue;
                rc = run_one_pad(fd, q->name, q->ic, q->oc, q->iw, q->ih, q->k,
                                 q->stride, 0, 0, 0, 0, 0, 0, (int)q->pad, 0, 0, &st);
                if (rc == 2) { printf("  SKIP   %-9s %-20s\n", "fq", q->name); skipped++; }
                else if (rc == 3 || q->refuse) {
                    /* Both directions: a shape past the measured boundary must be
                     * refused, and one inside it must not be. The boundary is silent
                     * on this path — a zero left pad writes nothing and a wrong output
                     * width writes a sheared surface — so the refusal IS the result. */
                    int ok = (rc == 3) == (q->refuse != 0);
                    printf("  %s   %-9s %-20s %s\n", ok ? "PASS" : "FAIL", "fq", q->name,
                           rc == 3 ? (q->refuse
                                      ? "refused, as the measured boundary requires"
                                      : "REFUSED but the boundary says it computes")
                                   : "COMPUTED but the measured boundary says it cannot");
                    if (ok) { if (rc == 3) refused++; else passed++; }
                    else    { failed++; wrong_refusal++;
                              lg_note_failure("fq", q->name, NULL,
                                              rc == 3 ? "REFUSED and should compute"
                                                      : "COMPUTED and should have refused"); }
                } else if (rc == 0) {
                    printf("  PASS   %-9s %-20s %d/%d exact  %.1f ms\n", "fq",
                           q->name, st.exact, st.total, st.ms);
                    passed++;
                } else {
                    printf("  FAIL   %-9s %-20s %d/%d exact, maxdiff %d\n", "fq",
                           q->name, st.exact, st.total, st.maxdiff);
                    lg_print_signature(&st);
                    lg_note_failure("fq", q->name, &st, NULL);
                    failed++;
                }
            }
    }

    {
        int want_nic = (ngroups == 0);
        for (a = 0; a < ngroups; a++) if (!strcmp(groups[a], "nic")) want_nic = 1;
        if (want_nic)
            for (i = 0; i < N_NIC; i++) {
                const nic_shape_t *n = &NIC_SHAPES[i];
                struct lg_stat st;
                int rc;
                if (filter && *filter && !strstr(n->name, filter)) continue;
                rc = run_one_pad(fd, n->name, n->ic, n->oc, n->iw, n->ih, n->k,
                                 n->stride, 0, 0, 0, n->in_zp, n->w_zp, n->out_zp,
                                 (int)n->pad, 1, n->ext, &st);
                if (rc == 2) { printf("  SKIP   %-9s %-20s\n", "nic", n->name); skipped++; }
                else if (rc == 3) {
                    printf("  FAIL   %-9s %-20s REFUSED (ic=%u, and the direct cube is a "
                           "32-channel group at every count)\n", "nic", n->name, n->ic);
                    lg_note_failure("nic", n->name, NULL, "REFUSED");
                    failed++;
                } else if (rc == 0) {
                    printf("  PASS   %-9s %-20s %d/%d exact  zp %d/%d/%d  %.1f ms\n",
                           "nic", n->name, st.exact, st.total, n->in_zp, n->w_zp,
                           n->out_zp, st.ms);
                    passed++;
                } else {
                    printf("  FAIL   %-9s %-20s %d/%d exact, maxdiff %d  zp %d/%d/%d\n",
                           "nic", n->name, st.exact, st.total, st.maxdiff,
                           n->in_zp, n->w_zp, n->out_zp);
                    lg_print_signature(&st);
                    lg_note_failure("nic", n->name, &st, NULL);
                    failed++;
                }
            }
    }

    {
        int want_fc = (ngroups == 0);
        for (a = 0; a < ngroups; a++) if (!strcmp(groups[a], "fc")) want_fc = 1;
        if (want_fc)
            for (i = 0; i < N_FC; i++) {
                const fc_shape_t *f = &FC_SHAPES[i];
                struct lg_stat st;
                int rc;
                if (filter && *filter && !strstr(f->name, filter)) continue;
                rc = run_fc(fd, f, &st);
                if (rc == 2) { printf("  SKIP   %-9s %-20s\n", "fc", f->name); skipped++; }
                else if (rc == 3) {
                    printf("  FAIL   %-9s %-20s REFUSED (ic=%u %ux%u k%u s%u)\n", "fc",
                           f->name, f->ic, f->iw, f->ih, f->k, f->stride);
                    lg_note_failure("fc", f->name, NULL, "REFUSED");
                    failed++;
                } else if (rc == 0) {
                    printf("  PASS   %-9s %-20s %d/%d exact  %.1f ms\n", "fc",
                           f->name, st.exact, st.total, st.ms);
                    passed++;
                } else {
                    printf("  FAIL   %-9s %-20s %d/%d exact, %d/%u whole channels, "
                           "maxdiff %d\n", "fc", f->name, st.exact, st.total,
                           st.chans, f->oc, st.maxdiff);
                    lg_print_signature(&st);
                    lg_note_failure("fc", f->name, &st, NULL);
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
    if (env_int("ROCKET_LG_RESIDENT", 0))
        printf("   resident A/B: %d shape(s) also run through pack + 2 prepacked calls "
               "and compared to the transient output element for element\n",
               lg_resident_ab);
    /* The count alone cannot be chased; the names can. Printed last so a captured
     * tail carries them, and with the drop-vs-arithmetic signature attached. */
    if (lg_nfailed) {
        int n = lg_nfailed < LG_MAX_FAILED ? lg_nfailed : LG_MAX_FAILED;
        printf("   failing shapes:\n");
        for (i = 0; i < n; i++) printf("     %s\n", lg_failed[i]);
        if (lg_nfailed > LG_MAX_FAILED)
            printf("     ... and %d more\n", lg_nfailed - LG_MAX_FAILED);
    }
    return failed ? 1 : 0;
}
