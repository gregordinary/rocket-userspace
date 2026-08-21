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
 *   rk3576_conv_lib_gate rowbound|rowlaw|rowmap    the row-allowance probes; these
 *                                      MEASURE and print a map, they do not assert
 *   rk3576_conv_lib_gate slicemap      where the trailing output-channel group is lost
 *                                      as the resident weight SLICE grows, at a forced
 *                                      feature allowance; also a map, not an assertion
 *
 * Env: ROCKET_LG_FILTER=<substr>   run only shapes whose name contains this
 *      ROCKET_LG_VERBOSE=1         per-shape detail and the first mismatches
 *      ROCKET_LG_MAP_LO/_HI        narrow rowmap's height walk
 *      ROCKET_LG_MAP_REPS=N        run each height N times and report the RATE, which is
 *                                  what an intermittent wall needs (default: one run,
 *                                  with a wrong result confirmed twice)
 *      ROCKET_LG_MAP_ONLY=1        skip rowmap's forced-rung section
 *      ROCKET_LG_MAP_SEQ=1         with REPS, print one line PER REP: the dropped-to-zero
 *                                  and wrong-valued counts and the span of each, in order.
 *                                  The aggregate reports the widest span over the reps, so
 *                                  it cannot see two reps failing in different FORMS, nor
 *                                  where in the sequence a form occurs.
 *      ROCKET_LG_MAP_F=N           force one CBUF F across the whole map walk, so the
 *                                  programmed allowance is constant between cells; on
 *                                  slicemap it selects ONE of the two forced arms
 *      ROCKET_LG_SLICE_LO/_HI      slicemap's slice walk, in KiB (default 80..116)
 *      ROCKET_LG_SLICE_OC          slicemap's output channels, i.e. its group count
 *      ROCKET_LG_SLICE_IW/_IH      slicemap's plane (default 4x2, where nothing drains)
 *      ROCKET_LG_SLICE_K           slicemap's kernel (default 1). The slice is
 *                                  32*ic*k*k and ic must be a multiple of 32, so the
 *                                  reachable slices are the multiples of k*k KiB —
 *                                  which is what decides whether a given k can put a
 *                                  slice on a 32 KiB boundary at all
 *      ROCKET_LG_SLICE_PASTPOOL=1  run slicemap's cells past the pool too, where the
 *                                  failure stops being quiet (no completion at all)
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
     * path carries. THREE classes, not two, because the library stamps every output
     * surface with 0xA5 before the submit (`rocket_rk3576_sentinel_on`, on by default):
     *   - still the SENTINEL (-91 as int8)  -> the part never wrote that element.
     *   - zero where a value was wanted     -> written, and written as nothing.
     *   - any other wrong value             -> written, with wrong data.
     * Reading a stamped surface as though it were the calloc'd zero collapses the first
     * class into the third and makes a dropped write indistinguishable from a wrong one;
     * `wrong_zero` alone therefore says nothing about whether the part reached an element.
     * All three counted, plus the span, so an intermittent names itself rather than being
     * re-run blind. */
    int    wrong_zero, wrong_val, wrong_stamp;
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
static void lg_wrong(struct lg_stat *st, int got_is_zero, int got_is_stamp,
                     unsigned c, unsigned y, unsigned x)
{
    if (got_is_zero) st->wrong_zero++; else st->wrong_val++;
    /* The library's own stamp byte, read back through the output surface. Counted
     * independently of the two above so the existing columns keep their meaning. */
    if (got_is_stamp) st->wrong_stamp++;
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
    printf("         %d dropped to zero, %d wrong-valued (%d still the stamp); "
           "c %d..%d y %d..%d x %d..%d\n",
           st->wrong_zero, st->wrong_val, st->wrong_stamp, st->span_c0, st->span_c1,
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
                    lg_wrong(st, got == 0, got == (int)(int8_t)0xA5, c, y, x);
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
                    lg_wrong(st, (float)got == 0.0f, 0, c, y, x);
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
struct rowbound_case { const char *name; unsigned ic, oc, iw, ih, k, stride; int same, dw; };

static const struct rowbound_case ROWBOUND[] = {
    /* The ResNet-18 stem, as the graph runs it (three image channels widened to eight)
     * and with a full 32 behind it, so the deficit can be read against `entries`. */
    { "stem-k7-ic8",     8,  64, 224, 224, 7, 2, 1, 0 },
    { "stem-k7-ic32",   32,  64, 224, 224, 7, 2, 1, 0 },
    { "k7-s1-112",      32,  32, 112, 112, 7, 1, 1, 0 },
    { "k7-s1-56-ic64",  64,  64,  56,  56, 7, 1, 1, 0 },
    { "k5-s1-112",      32,  32, 112, 112, 5, 1, 1, 0 },
    { "k3-s1-112",      32,  32, 112, 112, 3, 1, 1, 0 },
    { "k3-s1-224",      32,  32, 224, 224, 3, 1, 1, 0 },
    { "k1-s1-224",      32,  32, 224, 224, 1, 1, 0, 0 },
    /* The axes that separate the stem from every shape above it, one at a time: the
     * STRIDE, the output-channel group count, the plane width, and the kernel. */
    { "k3-s2-224",      32,  32, 224, 224, 3, 2, 1, 0 },
    { "k5-s2-224",      32,  32, 224, 224, 5, 2, 1, 0 },
    { "k7-s2-224-oc32", 32,  32, 224, 224, 7, 2, 1, 0 },
    { "k7-s1-224",      32,  32, 224, 224, 7, 1, 1, 0 },
    { "k7-s2-112",      32,  32, 112, 112, 7, 2, 1, 0 },
    /* THE DEPTHWISE AXIS, which nothing above varies and which a real model reaches.
     * `dw-k3-s1-160` is EfficientDet-Lite0's `blocks_0` and `dw-k3-s1-150` SSD MobileNet
     * V2's first depthwise, at the same channels and kernel; the rest walk the plane at a
     * fixed ic and the ic at a fixed plane.
     *
     * WHAT THESE TEN CELLS MEASURE IS THE PLAN, NOT THE SHAPE, and reading them the other
     * way is what cost a session. A cell's window is the planner's even spread and not its
     * cap — `dw-k3-s1-112` stages 3192 granules against a cap of 6104, so it never reaches
     * a rung at all — and the two that failed were the only two whose window selected a
     * CBUF rung the depthwise path does not honour. `dw-k5-s1-160` was read as refuting a
     * capacity bound because it needs MORE and computes; it computes BECAUSE it needs more,
     * which pushed it up one rung. The rung question itself is `rowmap`. */
    { "dw-k3-s1-112",   32,  32, 112, 112, 3, 1, 1, 1 },
    { "dw-k3-s1-150",   32,  32, 150, 150, 3, 1, 1, 1 },
    { "dw-k3-s1-160",   32,  32, 160, 160, 3, 1, 1, 1 },
    { "dw-k3-s1-176",   32,  32, 176, 176, 3, 1, 1, 1 },
    { "dw-k3-s1-224",   32,  32, 224, 224, 3, 1, 1, 1 },
    { "dw-k3-s1-160-ic64",  64,  64, 160, 160, 3, 1, 1, 1 },
    { "dw-k3-s1-160-ic96",  96,  96, 160, 160, 3, 1, 1, 1 },
    { "dw-k3-s1-160-ic128",128, 128, 160, 160, 3, 1, 1, 1 },
    { "dw-k5-s1-160",   32,  32, 160, 160, 5, 1, 1, 1 },
    { "dw-k3-s2-160",   32,  32, 160, 160, 3, 2, 1, 1 },
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
                                                         s->k, s->k, s->dw);
        unsigned entries = (s->iw * s->ic + 63u) / 64u;
        unsigned cap, measured = 0;
        struct lg_stat st;

        if (!predicted) { printf("  %-20s the planner refuses the shape\n", s->name);
                          continue; }
        {
            unsigned lo = s->k, hi = predicted;
            int rc = run_one(fd, s->name, s->ic, s->oc, s->iw, s->ih, s->k, s->stride,
                             s->same, s->dw, hi, 0, 0, 0, &st);
            /* A BISECTION OVER AN INTERMITTENT FAILURE REPORTS THE WRONG BOUND, and the
             * atom-drop hazard makes one here: a single unlucky probe at the top sends
             * the search down and it never comes back. Two shapes have swung 4.5x and
             * 1.6x between runs this way. So the top of the range is probed TWICE before
             * the search starts, and only a twice-confirmed failure is believed. */
            if (rc != 0)
                rc = run_one(fd, s->name, s->ic, s->oc, s->iw, s->ih, s->k, s->stride,
                             s->same, s->dw, hi, 0, 0, 0, &st);
            if (rc == 0) measured = hi;
            else while (lo <= hi) {                       /* the largest cap that works */
                cap = lo + (hi - lo) / 2u;
                rc = run_one(fd, s->name, s->ic, s->oc, s->iw, s->ih, s->k, s->stride,
                             s->same, s->dw, cap, 0, 0, 0, &st);
                if (rc == 0) { measured = cap; lo = cap + 1u; }
                else { if (cap == s->k) break; hi = cap - 1u; }
            }
        }
        printf("  %-20s %-6s ic=%-3u oc=%-3u %ux%u k%u s%u  entries %-4u  planner %-4u  "
               "part %-4u  %s\n", s->name, s->dw ? "dw" : "direct",
               s->ic, s->oc, s->iw, s->ih, s->k, s->stride,
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

/* ---- the row allowance as a LAW rather than a bound -------------------------------
 * `rowbound` above measures the shipped planner against the part on the shapes a graph
 * runs. This asks the question underneath it: WHAT does the part charge the feature
 * side for the weights beside it? The shipped model charges the whole weight cube —
 * groups x slice — which is under the one measurement that exists and is therefore a
 * safe bound and not the rule. Two quantities a real fix needs: whether the deficit
 * scales with the output-channel GROUP COUNT or is a fixed reserve, and where the
 * pool's own ceiling is.
 *
 * THE INSTRUMENT IS THE PLANE HEIGHT, NOT A FORCED ROW CAP. A cap on a tall plane does
 * not become the window: the planner counts greedily and then spreads the output rows
 * EVENLY over that many tasks, so a 112-row plane at a 96-row cap comes out as two
 * 62-row windows and the footprint under test is never staged. So each case is a plane
 * of exactly the height being probed, VALID-padded at stride one, which lays out as ONE
 * task whose feature footprint is exactly `ih * entries` granules. The RE knob
 * ROCKET_RK3576_ROW_CAP_PROBE is what lets the planner emit a window it would not have
 * chosen; the emitter's own per-task allowance check still governs, so the reachable
 * region ends at the data-side cap of 6144 granules.
 *
 * The two models this separates, at a fixed slice and a rising group count:
 *
 *   whole cube      ceiling = POOL - groups * slice     falls by one slice per group
 *   fixed reserve   ceiling = POOL - slice - R          does not move at all
 *
 * and a third the stem already refutes — that the charge is against the PROGRAMMED
 * allowance 4096+F rather than the staged footprint — since 48 rows of a 112-granule
 * row compute at F=2048 where that model allows 45.
 *
 * MEASURES, does not assert: it prints the ceiling, the granule arithmetic and the pool
 * each model backs out to. A failure at the ceiling costs the surface guard's eight
 * power cycles, and the bisection confirms every failure twice, because a bisection over
 * an intermittent reports the wrong bound.
 */
struct rowlaw_case { const char *name; unsigned ic, oc, iw, k; };

static const struct rowlaw_case ROWLAW[] = {
    /* A — the output-channel GROUP COUNT, with the slice held at 49 KiB. */
    { "k7-ic32-g1",  32,  32, 112, 7 },
    { "k7-ic32-g2",  32,  64, 112, 7 },
    { "k7-ic32-g3",  32,  96, 112, 7 },
    { "k7-ic32-g4",  32, 128, 112, 7 },
    { "k7-ic32-g5",  32, 160, 112, 7 },
    { "k7-ic32-g6",  32, 192, 112, 7 },
    /* B — the SLICE, at group counts that put the ceiling inside the reachable
     *     window. If the deficit is the cube, the step per group tracks the slice. */
    { "k5-ic32-g4",  32, 128, 112, 5 },
    { "k5-ic32-g6",  32, 192, 112, 5 },
    { "k5-ic32-g8",  32, 256, 112, 5 },
    { "k3-ic64-g4",  64, 128, 112, 3 },
    { "k3-ic64-g6",  64, 192, 112, 3 },
    { "k3-ic64-g8",  64, 256, 112, 3 },
    /* C — the same cube at a different WIDTH, so a per-row term would show up as a
     *     ceiling that moves with `entries` rather than with the granule total. */
    { "k7-ic32-g4-w56",  32, 128,  56, 7 },
    { "k7-ic32-g4-w224", 32, 128, 224, 7 },
};
#define N_ROWLAW ((int)(sizeof ROWLAW / sizeof *ROWLAW))

/* One plane height, run as a single task. 0 = computed bit-exactly. */
static int rowlaw_try_st(int fd, const struct rowlaw_case *s, unsigned ih,
                         struct lg_stat *st)
{
    char buf[32];
    int rc;

    snprintf(buf, sizeof buf, "%u", ih);
    setenv("ROCKET_RK3576_ROW_CAP_PROBE", buf, 1);
    rc = run_one(fd, s->name, s->ic, s->oc, s->iw, ih, s->k, 1, 0 /*VALID*/, 0,
                 0, 0, 0, 0, st);
    unsetenv("ROCKET_RK3576_ROW_CAP_PROBE");
    return rc;
}

static int rowlaw_try(int fd, const struct rowlaw_case *s, unsigned ih)
{
    struct lg_stat st;
    return rowlaw_try_st(fd, s, ih, &st);
}

static int rowlaw(int fd)
{
    int i;

    printf("rowlaw: the tallest single-task plane the part stages, in granules, against\n"
           "        the weight cube beside it. `slice` is 32*ic*kh*kw and `cube` is\n"
           "        groups*slice, both in 64-byte granules. POOL is what each model\n"
           "        backs out to: feature + cube, and feature + one slice.\n");
    printf("  %-18s %6s %6s %4s %6s  %5s %7s  %7s %7s\n",
           "case", "entries", "slice", "grp", "cube", "rows", "feature", "+cube", "+slice");
    for (i = 0; i < N_ROWLAW; i++) {
        const struct rowlaw_case *s = &ROWLAW[i];
        unsigned entries = (s->iw * s->ic + 63u) / 64u;
        unsigned slice   = (32u * s->ic * s->k * s->k + 63u) / 64u;
        unsigned groups  = (s->oc + 31u) / 32u;
        unsigned cube    = groups * slice;
        unsigned lo = s->k, hi = 6144u / entries, mid, measured = 0;
        int censored;

        if (hi < lo) { printf("  %-18s a single row is past the data cap\n", s->name);
                       continue; }
        /* The top of the range is the data-side cap, which the emitter refuses past.
         * Confirm it twice before believing it, then bisect. */
        if (rowlaw_try(fd, s, hi) == 0 && rowlaw_try(fd, s, hi) == 0) {
            measured = hi;
        } else {
            while (lo <= hi) {
                mid = lo + (hi - lo) / 2u;
                if (rowlaw_try(fd, s, mid) == 0) { measured = mid; lo = mid + 1u; }
                else if (rowlaw_try(fd, s, mid) == 0) { measured = mid; lo = mid + 1u; }
                else { if (mid == s->k) break; hi = mid - 1u; }
            }
        }
        censored = measured && measured == 6144u / entries;
        if (!measured) { printf("  %-18s nothing computed\n", s->name); continue; }
        printf("  %-18s %6u %6u %4u %6u  %5u %7u  %7u %7u%s\n",
               s->name, entries, slice, groups, cube, measured, measured * entries,
               measured * entries + cube, measured * entries + slice,
               censored ? "  (at the data cap — a lower bound)" : "");
        /* WHAT PAST THE ALLOWANCE LOOKS LIKE. A task that never wrote is caught by the
         * surface guard and costs a fallback; one that writes a WRONG surface is silent,
         * and the two are a different item entirely. `wrong_zero` is an element the DPU
         * never emitted (the calloc'd zero) and `wrong_val` is arithmetic. */
        if (!censored) {
            struct lg_stat st;
            int rc = rowlaw_try_st(fd, s, measured + 1u, &st);
            if (rc == 0)
                printf("  %-18s   ONE ROW PAST IT COMPUTED — the bound above is not "
                       "reproducing\n", "");
            else if (rc != 1 || !st.total)
                printf("  %-18s   one row past it: the entry refused (rc=%d)\n", "", rc);
            else
                printf("  %-18s   one row past it: %d/%d exact, %d never emitted, "
                       "%d wrong-valued, maxdiff %d, rows %d-%d of %u\n", "",
                       st.exact, st.total, st.wrong_zero, st.wrong_val, st.maxdiff,
                       st.span_y0, st.span_y1, measured + 1u - s->k + 1u);
        }
    }
    printf("== rowlaw MEASURES; the assertion is in the conv and net gates ==\n");
    return 0;
}

/* ---- which CBUF rung the part honours, on the path the caller is on -----------------
 * `rowbound` and `rowlaw` both BISECT, which assumes a smaller window is never worse.
 * That is what a capacity bound means, and it is why neither can see the failure the
 * depthwise path actually has: the F rungs are a LADDER, the planner picks the smallest
 * one that covers the window, and a rung the part does not honour delivers the F=0
 * budget instead. So the surface is exact below the rung's reach, WRONG across the band
 * of windows that select it, and exact again above — non-monotone, and a bisection over
 * a band reports whichever edge it walks into.
 *
 * Two readouts, both linear, both printing the map:
 *
 *   rungs  fixes one window that needs more than F=0 buys and FORCES each rung under it
 *          (ROCKET_RK3576_CBUF_F). This is the measurement — it asks the part directly
 *          which rungs deliver on this path, the same question tests/rk3576_conv_sym.c
 *          `rung` asked on the DIRECT path, asked again where the footprint differs.
 *   map    walks the plane height over the whole reachable range with the SHIPPED
 *          planner choosing F, so the band is read where a caller would meet it.
 *
 * Each case is a plane of exactly the probed height, VALID at stride one, which lays out
 * as ONE task whose feature footprint is exactly `ih * entries` granules — the same
 * construction `rowlaw` uses, and for the same reason: a forced cap on a tall plane is
 * spread evenly and the footprint under test is never staged.
 */
struct rowmap_case { const char *name; unsigned ic, oc, iw, k; int dw; };

static const struct rowmap_case ROWMAP[] = {
    /* EfficientDet-Lite0's `blocks_0` geometry. Depthwise resident footprint
     * oc*kh*kw*2 = 576 B = 9 granules, which passes the <=16-granule liveness the
     * DIRECT path measured, so the shipped planner programs F=512 here. */
    { "dw-160-ic32-k3",  32,  32, 160, 3, 1 },
    /* The same plane and channels at ONE tap: 64 B = 1 granule. Separates "the low
     * rungs are dead on the depthwise path" from "the depthwise footprint is larger
     * than the shipped one" — the second predicts this cell delivers. */
    { "dw-160-ic32-k1",  32,  32, 160, 1, 1 },
    /* 1600 B = 25 granules, past the liveness bound under either reading. The cell
     * `rowbound` reports as agreeing, because its SHIPPED window needs 4640 granules
     * and lands on F=1024 — which is a property of the plan, not of the shape. */
    { "dw-160-ic32-k5",  32,  32, 160, 5, 1 },
    /* THE CONTROL, and it is the exact point the low rungs were measured LIVE at:
     * direct, slice 32*ic*kh*kw = 1024 B = 16 granules, same plane, same entries per
     * row, same rung sequence. A band here would mean the reading is about neither
     * path and the instrument is the suspect. */
    { "dir-160-ic32-k1", 32,  32, 160, 1, 0 },
    /* The direct path one step past its own measured threshold — 2048 B = 32 granules,
     * recorded dead by tests/rk3576_conv_sym.c `rung`. Here so the two instruments are
     * known to agree about the same point before either is read about a new one. */
    { "dir-160-ic64-k1", 64,  64, 160, 1, 0 },
    /* ---- what the depthwise footprint IS, over the three models the four cells above
     * leave standing. The shipped one, `oc*kh*kw*2`, is already refuted: 9 granules is
     * dead where the direct path's 16 is live.
     *
     *   A  one channel GROUP of 64, 2-byte coefficients: 64*kh*kw*2, threshold 16
     *      granules — the same threshold as the direct path, and 64 is the int8
     *      depthwise cube's own group (R76_DW_W_GROUP_INT8)
     *   B  the low rungs are live on this path only at a single tap
     *   C  the shipped footprint with a lower threshold, somewhere in 2..8 granules
     *
     * A 2x2 kernel separates B from the other two: 8 granules under A, 4 under C, and
     * more than one tap under B. */
    { "dw-160-ic32-k2",  32,  32, 160, 2, 1 },
    /* And the CHANNEL count separates A from C, because A does not depend on it. Both
     * planes are 160 granules a row, so the window under test is the same 26 rows.
     * 256 channels: 2 granules under A, 8 under C — both live, so this one only
     * brackets C's threshold. 1024 channels: still 2 under A, 32 under C, which is the
     * direct path's own measured-dead point. That cell is the discriminator. */
    { "dw-40-ic256-k1", 256, 256,  40, 1, 1 },
    { "dw-10-ic1024-k1",1024,1024, 10, 1, 1 },
    /* ---- and the DIRECT path's own unpinned half: is the quantity one output-channel
     * GROUP's slice, or the whole resident CUBE? Every cell that ever reached a rung had
     * oc 32 — one group, where the two are the same number — so the shipped rule takes
     * the cube, which is the smaller envelope. These hold the slice at the measured-live
     * 16 granules and raise the group count, so the cube alone moves: 32 granules at
     * oc 64 and 48 at oc 96, both past the threshold if the cube is what is charged.
     * The planner would decline these rungs on its own, so only the FORCED readout
     * reaches them. */
    { "dir-160-ic32-k1-oc64", 32,  64, 160, 1, 0 },
    { "dir-160-ic32-k1-oc96", 32,  96, 160, 1, 0 },
    /* ---- THE DIRECT PATH'S ROW ALLOWANCE, at the output-channel group counts where the
     * shipped planner is over. A bisection (`rowlaw`) put the ceiling at 79 / 57-61 /
     * 44-45 / 38-39 / 32-34 input rows at 2-6 groups — falling with the group count, far
     * below any rung boundary, with only the last few output rows wrong and a 1-4 row
     * swing between runs. A bisection over a band or an intermittent reports whichever
     * edge it walks into, so these are read as a MAP, and `ROCKET_LG_MAP_REPS` makes each
     * height a RATE rather than one sample.
     *
     * `iw 112 ic 32 k7` is `rowlaw`'s own geometry — 56 granules a feature row, a
     * 784-granule weight slice per output-channel group — so the only axis that moves
     * across the six is `oc`. g1 and g2 are the controls: the shipped planner agrees with
     * the part at g1 and the conv gate's own k7 shapes sit at 1 and 2 groups. */
    { "k7g1-112",  32,  32, 112, 7, 0 },
    { "k7g2-112",  32,  64, 112, 7, 0 },
    { "k7g3-112",  32,  96, 112, 7, 0 },
    { "k7g4-112",  32, 128, 112, 7, 0 },
    { "k7g5-112",  32, 160, 112, 7, 0 },
    { "k7g6-112",  32, 192, 112, 7, 0 },
    /* The axes the six above hold constant, so the onset can be FITTED rather than
     * described. The SLICE at a fixed group count and width, through the kernel (k5 is
     * 400 granules, k3 is 144, against k7's 784) and through `ic` — the two must move the
     * onset the same way or the governing quantity is not the slice. And the WIDTH at a
     * fixed cube, where a per-ROW term would show up as an onset that moves with the row
     * count rather than with the granule total. Each is narrow enough to map around a
     * predicted onset in about a minute with ROCKET_LG_MAP_LO/_HI. */
    { "k5g4-112",   32, 128, 112, 5, 0 },
    { "k3g4-112",   32, 128, 112, 3, 0 },
    { "k7g4-56",    32, 128,  56, 7, 0 },
    { "k7g4-224",   32, 128, 224, 7, 0 },
    { "k7ic64-g2",  64,  64, 112, 7, 0 },
};
#define N_ROWMAP ((int)(sizeof ROWMAP / sizeof *ROWMAP))

/* One plane height at a forced F. `f_force` of ~0u leaves the planner its own choice.
 * Returns 0 exact, 1 wrong, 2 skip, 3 refused. */
static int rowmap_try(int fd, const struct rowmap_case *s, unsigned ih, unsigned f_force,
                      struct lg_stat *st)
{
    char buf[32];
    int rc;

    snprintf(buf, sizeof buf, "%u", ih);
    setenv("ROCKET_RK3576_ROW_CAP_PROBE", buf, 1);
    if (f_force != ~0u) {
        snprintf(buf, sizeof buf, "%u", f_force);
        setenv("ROCKET_RK3576_CBUF_F", buf, 1);
    }
    rc = run_one(fd, s->name, s->ic, s->oc, s->iw, ih, s->k, 1, 0 /*VALID*/, s->dw,
                 0, 0, 0, 0, st);
    unsetenv("ROCKET_RK3576_CBUF_F");
    unsetenv("ROCKET_RK3576_ROW_CAP_PROBE");
    return rc;
}

static const unsigned ROWMAP_RUNGS[] = { 0u, 256u, 512u, 1024u, 2048u };
#define N_ROWMAP_RUNGS ((int)(sizeof ROWMAP_RUNGS / sizeof *ROWMAP_RUNGS))

static int rowmap(int fd)
{
    const char *filter = getenv("ROCKET_LG_FILTER");
    int i, r;

#define ROWMAP_SKIP(s) (filter && *filter && !strstr((s)->name, filter))
    printf("rowmap: which CBUF F rung the part honours on this path, and where the\n"
           "        shipped planner's own choice lands. `need` is ih*entries granules;\n"
           "        a rung that delivers covers a window up to 4096+F.\n");

    if (env_int("ROCKET_LG_MAP_ONLY", 0)) goto map_only;
    printf("\n-- rungs: ONE window, each rung FORCED under it --\n");
    for (i = 0; i < N_ROWMAP; i++) {
        const struct rowmap_case *s = &ROWMAP[i];
        unsigned entries, resident;

        if (ROWMAP_SKIP(s)) continue;
        entries = (s->iw * s->ic + 63u) / 64u;
        resident = s->dw ? (s->oc * s->k * s->k * 2u)
                                  : (32u * s->ic * s->k * s->k);
        /* The shortest window F=0 does NOT buy, so every rung above it is separable:
         * a rung that delivers computes and one that falls back to 4096 does not. */
        unsigned ih = 4096u / entries + 1u;
        struct lg_stat st;

        if (ih < s->k) ih = s->k;
        printf("  %-16s %s ic=%-4u k%u  entries %u/row  resident %u B = %u granule(s)\n",
               s->name, s->dw ? "dw    " : "direct", s->ic, s->k, entries,
               resident, (resident + 63u) / 64u);
        printf("      window %u rows = %u granules, which needs a rung of at least %u\n",
               ih, ih * entries,
               ih * entries > 4096u ? ih * entries - 4096u : 0u);
        for (r = 0; r < N_ROWMAP_RUNGS; r++) {
            unsigned f = ROWMAP_RUNGS[r];
            int rc = rowmap_try(fd, s, ih, f, &st);
            /* A BISECTION IS NOT THE ONLY THING AN INTERMITTENT FOOLS. Confirm a
             * failure twice before printing it, the same rule the two probes above use. */
            if (rc == 1) rc = rowmap_try(fd, s, ih, f, &st);
            printf("        F=%-5u budget %-5u  %s", f, 4096u + f,
                   rc == 0 ? "EXACT — the rung delivers"
                   : rc == 3 ? "the entry refused"
                   : rc == 2 ? "skipped" : "WRONG");
            if (rc == 1)
                printf(" — %d/%d exact, %d never emitted, %d wrong-valued, rows %d-%d",
                       st.exact, st.total, st.wrong_zero, st.wrong_val,
                       st.span_y0, st.span_y1);
            printf("\n");
        }
    }

    /* The band as a caller meets it: the planner choosing, the height walking. Linear,
     * because the whole point is that the failing region has an upper edge. */
map_only:
    printf("\n-- map: the shipped planner choosing F, plane height walking --\n");
    for (i = 0; i < N_ROWMAP; i++) {
        const struct rowmap_case *s = &ROWMAP[i];
        unsigned entries, hi, ih;

        if (ROWMAP_SKIP(s)) continue;
        entries = (s->iw * s->ic + 63u) / 64u;
        hi = 6144u / entries;
        unsigned lo = env_int("ROCKET_LG_MAP_LO", 0) > 0
                          ? (unsigned)env_int("ROCKET_LG_MAP_LO", 0) : s->k;
        struct lg_stat st;

        if (env_int("ROCKET_LG_MAP_HI", 0) > 0 &&
            (unsigned)env_int("ROCKET_LG_MAP_HI", 0) < hi)
            hi = (unsigned)env_int("ROCKET_LG_MAP_HI", 0);
        printf("  %-16s %s entries %u/row, reachable heights %u..%u\n",
               s->name, s->dw ? "dw    " : "direct", entries, lo, hi);
        int reps = env_int("ROCKET_LG_MAP_REPS", 0);
        /* The planner's own F moves BETWEEN cells — a cube that fits at F=0 lets it charge
         * the whole cube and take a lower rung — so a walk that lets it choose is two
         * experiments at once and the honoured window cannot be fitted against the cube.
         * Forcing one rung across every cell holds the programmed allowance fixed and
         * leaves the cube as the only thing that moves. */
        int f_env = env_int("ROCKET_LG_MAP_F", -1);
        unsigned f_force = f_env < 0 ? ~0u : (unsigned)f_env;

        for (ih = lo; ih <= hi; ih++) {
            unsigned f = 0;
            int planned = rocket_rk3576_cbuf_f(s->iw, s->ic, ih, s->oc, s->k, s->k,
                                               s->dw, &f);
            if (f_env >= 0) f = (unsigned)f_env;
            if (reps < 2) {
                int rc = rowmap_try(fd, s, ih, f_force, &st);
                if (rc == 1) rc = rowmap_try(fd, s, ih, f_force, &st);
                printf("      ih %-4u need %-5u  planner F=%-5u  %s\n", ih, ih * entries,
                       planned < 0 ? 0u : f,
                       rc == 0 ? "exact"
                       : rc == 3 ? "refused" : rc == 2 ? "skip" : "WRONG");
                continue;
            }
            /* A RATE, because a wall that swings between runs is not read by a
             * confirm-twice: that turns a hazard which fires half the time into one that
             * prints "exact" three cells in four. Each height is run `reps` times and the
             * count of wrong runs is the cell; the spans are the WIDEST any wrong run
             * reached, since which rows and which channels go wrong is what separates a
             * row-tail wall from a channel-group one. */
            {
                int r2, n_wrong = 0, refused = 0, worst = 0;
                int y0 = 0, y1 = 0, c0 = 0, c1 = 0;
                /* The aggregate below reports the WIDEST span over the reps, which is what
                 * a wall's extent needs — but it cannot see whether two reps failed
                 * DIFFERENTLY, and this path has two failure forms (a row tail of the
                 * trailing group, and a surface wrong across every row and channel). The
                 * per-rep line separates them and carries the two things that discriminate
                 * a cut drain from a write landing in the wrong job's window: the
                 * zero/wrong-valued split, and the rep's POSITION. Off by default so no
                 * gate expectation moves. */
                int seq = env_int("ROCKET_LG_MAP_SEQ", 0);

                for (r2 = 0; r2 < reps; r2++) {
                    int rc = rowmap_try(fd, s, ih, f_force, &st);
                    if (seq && rc == 1)
                        printf("      seq ih %-4u rep %-3d %6d zero %6d val %6d stamp  "
                               "y %d-%d c %d-%d x %d-%d\n",
                               ih, r2, st.wrong_zero, st.wrong_val, st.wrong_stamp,
                               st.span_y0, st.span_y1, st.span_c0, st.span_c1,
                               st.span_x0, st.span_x1);
                    else if (seq)
                        printf("      seq ih %-4u rep %-3d %s\n", ih, r2,
                               rc == 0 ? "exact" : rc == 3 ? "refused" : "skip");
                    if (rc != 1) { if (rc != 0) refused++; continue; }
                    if (!n_wrong) {
                        y0 = st.span_y0; y1 = st.span_y1;
                        c0 = st.span_c0; c1 = st.span_c1;
                    } else {
                        if (st.span_y0 < y0) y0 = st.span_y0;
                        if (st.span_y1 > y1) y1 = st.span_y1;
                        if (st.span_c0 < c0) c0 = st.span_c0;
                        if (st.span_c1 > c1) c1 = st.span_c1;
                    }
                    if (st.wrong_zero + st.wrong_val > worst)
                        worst = st.wrong_zero + st.wrong_val;
                    n_wrong++;
                }
                printf("      ih %-4u out %-4u need %-5u  %sF=%-5u  %d/%d wrong",
                       ih, ih - s->k + 1u, ih * entries,
                       f_env >= 0 ? "forced " : "planner ", planned < 0 ? 0u : f,
                       n_wrong, reps);
                if (n_wrong)
                    printf("   rows %d-%d of %u, chans %d-%d, worst %d elements",
                           y0, y1, ih - s->k + 1u, c0, c1, worst);
                if (refused) printf("   (%d refused or skipped)", refused);
                printf("\n");
            }
        }
    }
    printf("== rowmap MEASURES; the assertion is in the conv and net gates ==\n");
#undef ROWMAP_SKIP
    return 0;
}

/* ---- the SLICE boundary as a function of the feature allowance ---------------------
 *
 * r76_weight_slice_cap() is the space at F=0 and r76_weight_slice_cap_at() subtracts F
 * from it granule for granule. That subtraction is 1:1 by MECHANISM — one pool, and a
 * granule is spent once — but the two cells that established the pool reading BRACKET it
 * rather than pin it: 98 KiB wrong at two groups and 49 KiB exact at four. A slope
 * steeper than 1:1 leaves the shipped rule accepting shapes the part computes wrong,
 * which is this file's usual quiet failure.
 *
 * So read the boundary directly, at one group count, with F the only thing that moves.
 * The plane is 4x2 — far inside every budget, nothing drains, and it is the geometry the
 * F=0 table itself was measured on — and `k=1` makes the slice `32*ic`, so an `ic` step
 * of 32 is a 1 KiB step. `ROCKET_RK3576_CBUF_F` forces the allowance and bypasses both
 * fit checks, which is what makes a boundary readable at all: the shipped plan would
 * refuse every cell past its own cap.
 *
 * Stay under the POOL as well as under the cap. At F the weight path has
 * 7168-4096-F granules, so the quiet graded loss only exists below that; past it the job
 * raises no completion and the surface guard spends its power cycles. At F=1024 that
 * ceiling is 128 KiB, so the default sweep stops at 116.
 */
static int slicemap(int fd)
{
    unsigned lo_kib = (unsigned)env_int("ROCKET_LG_SLICE_LO", 80);
    unsigned hi_kib = (unsigned)env_int("ROCKET_LG_SLICE_HI", 116);
    unsigned oc     = (unsigned)env_int("ROCKET_LG_SLICE_OC", 64);
    unsigned iw     = (unsigned)env_int("ROCKET_LG_SLICE_IW", 4);
    unsigned ih     = (unsigned)env_int("ROCKET_LG_SLICE_IH", 2);
    unsigned kk     = (unsigned)env_int("ROCKET_LG_SLICE_K", 1);
    int reps        = env_int("ROCKET_LG_MAP_REPS", 3);
    int f_env       = env_int("ROCKET_LG_MAP_F", -1);
    /* ROCKET_LG_MAP_F selects ONE forced allowance rather than filtering the default
     * pair — a value the pair does not contain would otherwise skip every arm and print
     * an empty section, which reads as "no cells" and not as "you asked for nothing". */
    unsigned fs_one[1], kib;
    const unsigned *FS;
    int n_fs, fi;
    static const unsigned FS_DEFAULT[] = { 0u, 1024u };
    int past_pool = env_int("ROCKET_LG_SLICE_PASTPOOL", 0);

    if (f_env >= 0) { fs_one[0] = (unsigned)f_env; FS = fs_one; n_fs = 1; }
    else            { FS = FS_DEFAULT; n_fs = (int)(sizeof FS_DEFAULT / sizeof *FS_DEFAULT); }

    if (reps < 1) reps = 1;
    if (kk < 1) kk = 1;
    /* The slice is 32*ic*k*k bytes and the direct path demands ic be a multiple of 32
     * (rocket_rk3576_pad_ic), so the reachable slices are exactly the multiples of
     * k*k KiB — 1 KiB apart at k=1, 4 at k=2, 9 at k=3, 16 at k=4. That lattice is
     * what decides which k can drive a question about a slice SIZE: a 32 KiB-multiple
     * slice needs lcm(k*k, 32) = 32, so k=1, 2 and 4 reach one and k=3 and k=5 cannot
     * reach one below 288 and 800 KiB. A cell off the lattice is printed rather than
     * silently skipped, because an absent row reads as a cell that ran and passed. */
    printf("slicemap: where the trailing output-channel group is lost, as a function of\n"
           "          the resident weight SLICE, at a FORCED feature allowance. Plane\n"
           "          %ux%u, k=%u, oc=%u (%u group(s)); slice = 32*ic*k*k, so %u KiB\n"
           "          per step and only multiples of %u KiB are reachable.\n",
           iw, ih, kk, oc, (oc + 31u) / 32u, kk * kk, kk * kk);
    if (iw < kk || ih < kk) {
        printf("      the %ux%u plane is smaller than k=%u at VALID padding: every cell\n"
               "      would skip. Raise ROCKET_LG_SLICE_IW/_IH.\n", iw, ih, kk);
        return 0;
    }

    for (fi = 0; fi < n_fs; fi++) {
        unsigned f = FS[fi];
        unsigned pool_kib;

        pool_kib = (7168u - 4096u - f) / 16u;   /* granules left to the weight path */
        printf("\n-- forced F=%u: budget %u granules, the weight path has %u (%u KiB) --\n",
               f, 4096u + f, 7168u - 4096u - f, pool_kib);
        for (kib = lo_kib; kib <= hi_kib; kib++) {
            unsigned ic;
            struct lg_stat st;
            int r, n_wrong = 0, refused = 0, worst = 0, nocomp = 0;
            int c0 = 0, c1 = 0;
            unsigned nout = oc * (iw - kk + 1u) * (ih - kk + 1u);   /* VALID, stride 1 */
            char nm[48];

            if (kib % (kk * kk)) {
                printf("      slice %-4u KiB  --      off the k=%u lattice, no legal ic\n",
                       kib, kk);
                continue;
            }
            ic = kib / (kk * kk) * 32u;
            if (kib > pool_kib && !past_pool) {
                printf("      slice %-4u KiB  ic %-5u  past the pool, not run "
                       "(ROCKET_LG_SLICE_PASTPOOL=1 runs it)\n", kib, ic);
                continue;
            }
            snprintf(nm, sizeof nm, "slice-%uKiB-oc%u-F%u-k%u", kib, oc, f, kk);
            for (r = 0; r < reps; r++) {
                int rc;
                char buf[32];

                snprintf(buf, sizeof buf, "%u", f);
                setenv("ROCKET_RK3576_CBUF_F", buf, 1);
                rc = run_one(fd, nm, ic, oc, iw, ih, kk, 1, 0 /*VALID*/, 0,
                             0, 0, 0, 0, &st);
                unsetenv("ROCKET_RK3576_CBUF_F");
                if (rc != 1) { if (rc != 0) refused++; continue; }
                /* The entry failing outright comes back as `wrong` with nothing scored,
                 * which otherwise prints as "N/N wrong, worst 0" and reads as a wrong
                 * SURFACE. It is the opposite kind of cell — past the pool the program
                 * does not execute at all — so it gets its own column. */
                if (!st.total) { nocomp++; continue; }
                if (!n_wrong) { c0 = st.span_c0; c1 = st.span_c1; }
                else {
                    if (st.span_c0 < c0) c0 = st.span_c0;
                    if (st.span_c1 > c1) c1 = st.span_c1;
                }
                if (st.wrong_zero + st.wrong_val > worst)
                    worst = st.wrong_zero + st.wrong_val;
                n_wrong++;
            }
            printf("      slice %-4u KiB  ic %-5u  %d/%d wrong", kib, ic, n_wrong, reps);
            if (n_wrong)
                printf("   chans %d-%d of %u, worst %d of %u elements",
                       c0, c1, oc, worst, nout);
            if (nocomp) printf("   (%d did not COMPLETE — the entry failed, no surface "
                               "to score)", nocomp);
            if (refused) printf("   (%d refused or skipped)", refused);
            printf("\n");
        }
    }
    printf("== slicemap MEASURES; the assertion is claimplan's `one pool` cells ==\n");
    return 0;
}

/* ---- the CLAIM-TIME plan against the RUN's own verdict -----------------------------
 *
 * rocket_conv2d_int8_plan_rk3576() exists so a frontend can decline a shape while it is
 * still deciding what to CLAIM, where the cost is one node the framework runs itself —
 * against a refusal at pack or run time, which fails the caller's whole model. That is
 * only worth anything if the two agree, and they are separate code: the plan reads the
 * descriptor, the run reaches the same bound through the emitter's per-task allowance.
 *
 * So this asserts the agreement over the WHOLE envelope table rather than over cells
 * written for it. `lib_refuse` is the run's own verdict, gated on the part by every other
 * cell in this file, so the table is already the ground truth and needs no new column.
 *
 * The two failures are not the same failure. A plan that ACCEPTS what the run refuses is
 * the defect this entry exists to prevent — a claim that fails Prepare. A plan that
 * REFUSES what the run computes costs a claim and nothing else, so it is reported as an
 * over-refusal rather than a failure; a pure predicate that is conservative in that
 * direction is doing its job, and one that drifts far enough to matter shows up as a
 * count here rather than as a silent loss of coverage in a frontend.
 *
 * Pure: no submit, no device work, no allocation past one row-task array.
 */
static int claimplan(void)
{
    int i, fail = 0, over = 0, n = 0;

    printf("== the pure claim-time plan against the run's own verdict, %d shapes ==\n",
           N_SHAPES);
    for (i = 0; i < N_SHAPES; i++) {
        const shape_t *s = &SHAPES[i];
        rocket_conv2d_desc d;
        int rc, refused;

        memset(&d, 0, sizeof d);
        d.ic = (int)s->ic; d.oc = (int)s->oc;
        d.ih = (int)s->ih; d.iw = (int)s->iw;
        d.kh = (int)s->k;  d.kw = (int)s->k;
        d.stride_y = (int)s->stride; d.stride_x = (int)s->stride;
        d.pad_top = s->same ? (int)(s->k / 2u) : 0;
        d.pad_left = d.pad_top;
        d.dil_y = 1; d.dil_x = 1;
        d.depthwise = s->dw;

        rc = rocket_conv2d_int8_plan_rk3576(&d);
        refused = rc != ROCKET_OK;
        n++;
        if (refused && !s->lib_refuse) {
            printf("  OVER   %-9s %-20s the plan refuses (%d) a shape the library "
                   "computes — a lost claim, not a wrong answer\n",
                   s->group, s->name, rc);
            over++;
        } else if (!refused && s->lib_refuse) {
            printf("  FAIL   %-9s %-20s the plan ACCEPTS a shape the library refuses — "
                   "a claim that fails Prepare\n", s->group, s->name);
            lg_note_failure(s->group, s->name, NULL, "claim-time plan accepts a refusal");
            fail++;
        }
    }
    printf("== claimplan: %d shapes, %d accept-a-refusal, %d over-refusal ==\n",
           n, fail, over);

    /* ---- the feature allowance and the weight slice are one pool, asserted PURELY.
     *
     * `4096+F` granules are programmed for the feature side whatever the plane's height
     * is, so a rung is a claim on the same granules the resident slice needs, and the
     * measured slice cap is the space at F=0. Both shapes below were wrong ON THE PART at
     * the rung the planner used to take, in the trailing output-channel groups and at every
     * row — 1 of 2 groups at 98 KiB beside F=1024, 3 of 4 at 49 KiB beside F=2048 — and the
     * CBUF pool check accepts both (6688 and 6928 granules against 7168), so it is not what
     * covers them. `k7ic64-g2` is the one that had been recorded as a dead F=1024 rung.
     *
     * These are the WINDOW's side of it. A rung whose F-aware cap no longer covers the
     * slice must be skipped rather than taken, or the row planner hands out exactly the
     * failing band: 45 input rows for the first shape is F=1024's window and every height
     * from 37 to 45 was wrong. Pure arithmetic, so this asserts off-device too, which is
     * the whole point — nothing in the corpus reaches either geometry.
     * [HW sweep, H96 MAX M9, `rowmap` with ROCKET_LG_MAP_F] */
    {
        static const struct { const char *name; unsigned iw, ic, oc, k, rows, f_at; }
        POOLSHARE[] = {
            /* 98 KiB slice, 2 groups: F=1024's cap is 92 KiB, so F=0 and 36 rows. */
            { "k7 ic64 oc64  112", 112,  64,  64, 7, 36, 0u },
            /* 49 KiB slice, 4 groups: F=1024's cap is 80 KiB and F=2048's is 16, so
             * F=1024 and 91 rows — where charging the pool alone would take 2048. */
            { "k7 ic32 oc128 112", 112,  32, 128, 7, 91, 1024u },
        };
        size_t j;
        printf("== the feature allowance and the weight slice are one pool ==\n");
        for (j = 0; j < sizeof POOLSHARE / sizeof POOLSHARE[0]; j++) {
            unsigned got = rocket_rk3576_max_task_rows(POOLSHARE[j].iw, POOLSHARE[j].ic,
                                                       POOLSHARE[j].oc, POOLSHARE[j].k,
                                                       POOLSHARE[j].k, 0);
            unsigned f = ~0u;
            int rc = rocket_rk3576_cbuf_f(POOLSHARE[j].iw, POOLSHARE[j].ic,
                                          POOLSHARE[j].rows, POOLSHARE[j].oc,
                                          POOLSHARE[j].k, POOLSHARE[j].k, 0, &f);
            int bad = got != POOLSHARE[j].rows || rc != 0 || f != POOLSHARE[j].f_at;
            printf("  %-6s %-18s window %u row(s) (want %u), F=%d at that window "
                   "(want %u)\n", bad ? "FAIL" : "ok", POOLSHARE[j].name, got,
                   POOLSHARE[j].rows, rc == 0 ? (int)f : -1, POOLSHARE[j].f_at);
            if (bad) {
                lg_note_failure("poolshare", POOLSHARE[j].name, NULL,
                                "the rung and the weight slice are not charged together");
                fail++;
            }
        }
        /* And the height one past that window must REFUSE rather than compute, which is
         * the difference between a shorter window and a silently wrong surface. */
        {
            unsigned f = ~0u;
            int rc = rocket_rk3576_cbuf_f(112, 64, 37, 64, 7, 7, 0, &f);
            printf("  %-6s %-18s one row past the window refuses (rc=%d)\n",
                   rc == 0 ? "FAIL" : "ok", "k7 ic64 oc64  112", rc);
            if (rc == 0) {
                lg_note_failure("poolshare", "k7 ic64 oc64 112 ih37", NULL,
                                "the plan accepts a rung the slice does not leave room for");
                fail++;
            }
        }
    }
    return fail ? 1 : 0;
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

    int want_rowbound = 0, want_rowlaw = 0, want_rowmap = 0, want_claimplan = 0;
    int want_slicemap = 0;
    for (a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-l")) list = 1;
        else if (!strcmp(argv[a], "rowbound")) want_rowbound = 1;
        else if (!strcmp(argv[a], "rowlaw")) want_rowlaw = 1;
        else if (!strcmp(argv[a], "rowmap")) want_rowmap = 1;
        else if (!strcmp(argv[a], "slicemap")) want_slicemap = 1;
        else if (!strcmp(argv[a], "claimplan")) want_claimplan = 1;
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

    /* The claim-time plan is PURE, so it runs before the device is opened and on a host
     * with no NPU at all — which is where a frontend asks it. */
    if (want_claimplan) return claimplan();

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

    if (want_rowbound || want_rowlaw || want_rowmap || want_slicemap) {
        int rc = want_rowbound ? rowbound(fd) : 0;
        if (want_rowlaw) rc |= rowlaw(fd);
        if (want_rowmap) rc |= rowmap(fd);
        if (want_slicemap) rc |= slicemap(fd);
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
