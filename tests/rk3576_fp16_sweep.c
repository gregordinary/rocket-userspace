// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_fp16_sweep.c — settle the RK3576's non-int8 encodings on the part.
 *
 * Every vendor capture is int8, so nothing about an fp16 program is transcribed:
 * the precision fields, the element-size scaling and the converter settings the
 * emitter uses are the RK3588's own encoding carried across. That makes fp16 a
 * HARDWARE SWEEP rather than a transcription, and this is the sweep.
 *
 * The failure mode it is built around: a wrong precision field does not fault. The
 * part writes a full, correctly sized, wrong surface, which is indistinguishable by
 * inspection from a right one. So every candidate is scored against a CPU model of
 * the same convolution, and "the BO was written" is reported separately from "the
 * BO was written correctly" — an untouched buffer, an all-zero surface and a wrong
 * surface are three different results and each points somewhere else.
 *
 * WHAT IT SWEEPS
 *
 * Each candidate is a set of whole register values, applied through the emitter's
 * own ROCKET_RK3576_SET override, so nothing here needs a rebuild to move and a
 * candidate that wins can be read straight back into the emitter. `bitsweep` walks
 * one register bit by bit, or over an explicit ROCKET_FS_SWEEP_VALS list, which is
 * what finds a field this part re-packed rather than inherited.
 *
 * WHAT IT HAS SETTLED. The float datapath computes exactly: CNA 0x100C's proc field
 * is the MAC datapath width and wants fp32 (at fp16 the part reads the weight cube a
 * byte per lane), CORE 0x3018 keeps the operand precision instead, the float weight
 * cube groups oc by 8 and ic by 16, and an fp16 OUTPUT needs the float narrowing at
 * DPU 0x40B0 bit 16.
 *
 * TWO GAPS ARE STILL OPEN, and each bounds a different axis. The output packing: the
 * DPU writes each result dword twice, so a 16-byte atom carries 4 output channels
 * instead of 8 and only the first oc/2 reach DDR. And the feature surface index,
 * which advances at the int8 rate while the datapath steps at the fp16 one, so one
 * task contracts 8 input channels and no more. `icsplit` is the split around the
 * second, driven through the library that now owns it.
 *
 * THE OUTPUT WIDTH IS NOT THE OUTPUT FORMAT. DPU 0x4010 bits [31:29] select how many
 * bytes of the epilogue's 32-bit result word reach DDR, and the epilogue is a FLOAT
 * path: at 5 the whole fp32 word lands, at 2 or 3 only its low or high half. The low
 * half is zero for every value a small integer test pattern can produce, so an fp16
 * program without the narrowing reads as an all-zero surface whether or not the MAC
 * ran. `f32out` takes the full 32-bit word instead, which is what separates a MAC that
 * contributes nothing from a truncation that discards everything it did.
 *
 * ASYMMETRIC WEIGHTS are the other half, and they are a packing question rather
 * than an encoding one: B in the coefficient group is a per-channel int16 that the
 * captures always leave at zero. `asym` drives a non-zero weight zero point at a
 * value far from zero and reports which SIGN CONVENTION reproduces the CPU model,
 * -B*sum(x) or +B*sum(x) — the two are indistinguishable near zero, which is why
 * the default probe uses a large one.
 *
 * Usage:
 *   rk3576_fp16_sweep              sweep the fp16 candidates
 *   rk3576_fp16_sweep icsplit      the LIBRARY's ic split, scored end to end
 *   rk3576_fp16_sweep map          decode the output map off the part (probe 8)
 *   rk3576_fp16_sweep bitsweep     walk one register (ROCKET_FS_SWEEP_*)
 *   rk3576_fp16_sweep f32out       the fp16 datapath read back as fp32 (out width 5)
 *   rk3576_fp16_sweep int8         the control: the same harness on the working path
 *   rk3576_fp16_sweep asym         int8 with a non-zero weight zero point
 *
 * Env knobs:
 *   ROCKET_FS_IC / _OC / _IW / _IH / _K / _STRIDE   geometry (default 32/32/8/8/1/1)
 *   ROCKET_FS_PROBE                1 uniform (values via _PIN/_PW), 2 a single
 *                                  element, 3 the pixel ramp on every channel,
 *                                  5 kernel c weighted c+1, 6 the input-channel
 *                                  pairing, 7 the weight-lane count
 *   ROCKET_FS_C2/_WOC/_WIC/_OUT_C2 cube geometry, one axis at a time
 *   ROCKET_FS_OUT_LAYOUT           0 the native output cube, 1 the `dup4` map the
 *                                  part's fp16 writer actually produces
 *   ROCKET_FS_SWEEP_REG/_BASE/_VALS/_WITH/_F16   the bitsweep mode
 *   ROCKET_FS_WZP                  weight zero point for `asym` (default 64)
 *   ROCKET_FS_GAP_MS               gap between submits (default 0)
 *   ROCKET_FS_ONLY=<substr>        run only candidates whose name contains this
 *   ROCKET_FS_VERBOSE=1            per-channel first mismatches
 *
 * Exit: 0 a candidate reproduced the model, 1 none did, 2 no NPU (skip).
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
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_hw_profile.h"

#define SENTINEL 0xAA

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);
int weight_conv_fp16(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);
int weight_conv_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

static int env_int(const char *k, int dflt)
{
    const char *v = getenv(k);
    return (v && *v) ? atoi(v) : dflt;
}

static double env_dbl(const char *k, double dflt)
{
    const char *v = getenv(k);
    return (v && *v) ? strtod(v, NULL) : dflt;
}

/* Register numbers and register values arrive as hex. atoi() reads "0x100c" as 0 and
 * then the sweep silently patches register 0 and reports that nothing moved, so every
 * knob that carries a register or a whole word goes through this instead. */
static uint32_t env_u32(const char *k, uint32_t dflt)
{
    const char *v = getenv(k);
    return (v && *v) ? (uint32_t)strtoul(v, NULL, 0) : dflt;
}

/* Geometry, set once in main. */
static unsigned IC, OC, IW, IH, K, STRIDE, OW, OH;

/* What the DPU writes per output element. The width is a register field, so it is
 * independent of the datapath precision: an fp16 program can be read back as the
 * full 32-bit epilogue word. */
enum out_kind { OUT_I8 = 0, OUT_F16, OUT_F32 };

/* The DPU output-width field, bits [31:29] of 0x4010. 5 is the whole word. */
#define R76_OUTW_WORD32 5u

/* Which cube layout the buffers are scattered into.
 *
 * The RK3588's fp16 cube keeps the ATOM at 16 bytes and halves its lane count when
 * the element doubles: C2 = 8 fp16 lanes, weight oc-group 16. The other reading is
 * that the geometry is fixed in LANES rather than in bytes — 16 feature channels and
 * an oc-group of 32 whatever the element size, which is what this part's int8 path
 * uses — and then an fp16 atom is 32 bytes wide. Nothing in an int8 capture separates
 * the two, so it is a knob and the part decides. */
enum cube_layout { CUBE_RK3588_FP16 = 0, CUBE_LANES_FIXED = 1 };
static int LAYOUT;
/* Feature lanes per atom, and the weight cube's output- and input-channel groups.
 * LAYOUT picks a named pair; the three knobs move each independently, which is what
 * an unknown group size needs — the RK3588 fp16 cube is (C2 8, oc 16, ic 32) and the
 * part's own int8 cube is (16, 32, 32), and nothing says fp16 here is either. */
static unsigned FEAT_C2, W_OC_GROUP, W_IC_GROUP;

static double fs_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Was this surface written at all? An untouched buffer, an all-zero one and a wrong
 * one are three different results and each points somewhere else, so the first has to
 * be separable from the other two before anything is scored. */
static int fs_all_sentinel(const void *buf, size_t bytes)
{
    const uint8_t *b = (const uint8_t *)buf;
    size_t i;
    for (i = 0; i < bytes; i++)
        if (b[i] != SENTINEL) return 0;
    return 1;
}

static int fs_feature_idx(unsigned c, unsigned y, unsigned x, unsigned c2)
{
    (void)c2;
    return feature_data((int)IC, (int)IH, (int)IW, (int)FEAT_C2,
                        (int)c + 1, (int)y + 1, (int)x + 1);
}

/* Which map the OUTPUT surface is read back through.
 *
 * `native` is the cube the emitter is aiming at — the same feature_data() scatter the
 * input side uses, at the output's own lane count.
 *
 * `dup4` is what the part actually writes at fp16 today: it spends FOUR bytes on a
 * two-byte element, so a 16-byte atom carries 4 output channels instead of 8 and each
 * result dword lands twice. Reading through it answers a different question from
 * "is the packing fixed" — namely whether the values that DO reach DDR are correct
 * and in a known place, which decides whether fp16 is usable under a channel
 * constraint while the register is still open. It is the LIBRARY's map — the same
 * one a caller reads a surface back through — so a pass here is a pass for callers.
 */
enum out_layout { OUTL_NATIVE = 0, OUTL_DUP4 = 1 };
static int OUT_LAYOUT;

static int fs_output_idx(unsigned c, unsigned y, unsigned x, unsigned c2)
{
    if (OUT_LAYOUT == OUTL_DUP4)
        return rocket_rk3576_fp16_out_index(OH, OW, c, y, x);
    return feature_data((int)OC, (int)OH, (int)OW, (int)c2,
                        (int)c + 1, (int)y + 1, (int)x + 1);
}

/* Mesa's oc1/ic1/kh/kw/oc2/ic2 reorder with both group sizes left open. At the
 * published groups this reproduces weight_conv_fp16 and weight_conv_int8 exactly. */
static int fs_weight_idx(unsigned oc, unsigned ic, unsigned kh, unsigned kw, int fp16)
{
    unsigned goc = W_OC_GROUP, gic = W_IC_GROUP;
    unsigned oc1 = oc / goc, oc2 = oc % goc;
    unsigned ic1 = ic / gic, ic2 = ic % gic;
    unsigned nic1 = (IC + gic - 1u) / gic;
    if (!fp16)
        return weight_conv_int8((int)OC, (int)IC, (int)K, (int)K,
                                (int)oc + 1, (int)ic + 1, (int)kh + 1, (int)kw + 1);
    return (int)((((((oc1 * nic1 + ic1) * K) + kh) * K + kw) * goc + oc2) * gic + ic2);
}

/* ============================================================================
 * SECTION — the candidate table
 *
 * A candidate is a name and an override string. The empty string is the emitter's
 * own program, which is the RK3588-derived encoding; every other entry moves one
 * thing away from it so a win names what was wrong rather than only that something
 * was. The values that depend on geometry are filled in at run time.
 * ==========================================================================*/
enum cand_id {
    CAND_BASELINE = 0,
    CAND_NO_PREC,
    CAND_QDEN,
    CAND_WBYTES_X1,
    CAND_WBPK_X2,
    CAND_ENTRIES_X1,
    CAND_MANUAL,
};

typedef struct {
    enum cand_id id;
    const char *name;
    const char *why;
} candidate_t;

static const candidate_t FP16_CANDIDATES[] = {
    { CAND_BASELINE, "baseline",
      "the emitter's own program: the HW-swept float encoding" },
    { CAND_NO_PREC, "no-precision-fields",
      "this part may not carry precision in these words at all" },
    { CAND_QDEN, "qden-set",
      "RK3588 sets CORE qd_en for fp16; the RK3576 int8 word already has bit 0" },
    { CAND_WBYTES_X1, "wbytes-x1",
      "total weight bytes may count elements rather than bytes" },
    { CAND_WBPK_X2, "wbpk-x2",
      "weight bytes per kernel may scale with the element size too" },
    { CAND_ENTRIES_X1, "entries-x1",
      "the CBUF entry count may be in elements rather than bytes" },
    { CAND_MANUAL, "manual",
      "whatever ROCKET_FS_SET holds; the hand-driven probe (empty = baseline)" },
};

/* ============================================================================
 * SECTION — the CPU model
 * ==========================================================================*/
static float model_out(const _Float16 *in, const _Float16 *w, const float *bias,
                       unsigned oc, unsigned y, unsigned x)
{
    float acc = bias ? bias[oc] : 0.0f;
    unsigned ic, kh, kw;
    for (ic = 0; ic < IC; ic++)
        for (kh = 0; kh < K; kh++)
            for (kw = 0; kw < K; kw++) {
                unsigned iy = y * STRIDE + kh, ix = x * STRIDE + kw;
                acc += (float)in[(ic * IH + iy) * IW + ix] *
                       (float)w[((oc * IC + ic) * K + kh) * K + kw];
            }
    return acc;
}

/* Build a candidate's override string against this run's geometry. Whole register
 * values, so a candidate that wins can be read straight back into the emitter. */
static void build_set(char *dst, size_t n, enum cand_id id)
{
    unsigned ent1 = (IW * IC + 63u) / 64u;          /* the int8-sized entry count */

    dst[0] = 0;
    switch (id) {
    case CAND_BASELINE:
        break;
    case CAND_NO_PREC:
        /* conv_mode alone in 0x100C, the transcribed direct CORE word, zero DPU. */
        snprintf(dst, n, "0x100c=0x0,0x3018=0x10000001,0x4010=0x0");
        break;
    case CAND_QDEN:
        /* proc_precision fp16 at bits [10:8], qd_en set, over the direct word. */
        snprintf(dst, n, "0x3018=0x%x", 0x10000001u | (2u << 8));
        break;
    case CAND_WBYTES_X1:
        snprintf(dst, n, "0x101c=0x%x", IC * OC * K * K);
        break;
    case CAND_WBPK_X2:
        snprintf(dst, n, "0x1030=0x%x", ((IC * K * K * 4u) << 16) | (OW - 1u));
        break;
    case CAND_ENTRIES_X1:
        snprintf(dst, n, "0x103c=0x%x,0x1044=0x%x,0x1028=0x%x",
                 ent1 << 16, (IW << 16) | ent1, ((ent1 * IH) << 16) | (IC - 1u));
        break;
    case CAND_MANUAL: {
        const char *s = getenv("ROCKET_FS_SET");
        if (s && *s) snprintf(dst, n, "%s", s);
        break;
    }
    }
}

/* ============================================================================
 * SECTION — one run
 * ==========================================================================*/
struct bufs {
    rocket_bo in, w, b, o, r;
};

static void bufs_free(int fd, struct bufs *B)
{
    if (B->r.ptr)  rocket_bo_free(fd, &B->r);
    if (B->o.ptr)  rocket_bo_free(fd, &B->o);
    if (B->b.ptr)  rocket_bo_free(fd, &B->b);
    if (B->w.ptr)  rocket_bo_free(fd, &B->w);
    if (B->in.ptr) rocket_bo_free(fd, &B->in);
}

/* Score a surface against the model. Reports the three outcomes separately.
 * `lim` is the allocation in elements of the read type, so a layout that indexes
 * past the buffer is reported rather than read. */
static int score(const void *out, enum out_kind kind, const _Float16 *in_p,
                 const _Float16 *w_p, const float *bias, unsigned c2,
                 size_t lim, double *max_rel, int verbose)
{
    unsigned c, y, x, exact = 0, total = 0, untouched = 0, zero = 0, oob = 0;
    double worst = 0.0;

    for (c = 0; c < OC; c++)
        for (y = 0; y < OH; y++)
            for (x = 0; x < OW; x++) {
                int idx = fs_output_idx(c, y, x, c2);
                /* score() indexes the OUTPUT cube, whose lane count is its own knob:
                 * c2 arrives already resolved by the caller. */
                double got, want = model_out(in_p, w_p, bias, c, y, x);
                double rel;
                total++;
                if (idx < 0 || (size_t)idx >= lim) { oob++; continue; }
                /* The integer path saturates into its output type; the model has to
                 * or every clamped lane reads as an encoding error. */
                if (kind == OUT_I8)
                    want = want < -128.0 ? -128.0 : (want > 127.0 ? 127.0 : want);
                if (kind == OUT_F32) {
                    if (((const uint32_t *)out)[idx] == 0xAAAAAAAAu) { untouched++; continue; }
                    got = (double)((const float *)out)[idx];
                } else if (kind == OUT_F16) {
                    const _Float16 *o = (const _Float16 *)out;
                    got = (double)o[idx];
                    /* An untouched fp16 lane reads as the sentinel bit pattern. */
                    if (((const uint16_t *)out)[idx] == 0xAAAAu) { untouched++; continue; }
                } else {
                    const int8_t *o = (const int8_t *)out;
                    if ((uint8_t)o[idx] == SENTINEL) { untouched++; continue; }
                    got = (double)o[idx];
                }
                /* Count every lane that reads zero, not only the ones the model
                 * disagrees with: an all-zero surface whose model happens to be zero
                 * in places otherwise scores as a partial match and hides the
                 * gated-BS signature behind it. */
                if (got == 0.0) zero++;
                rel = fabs(want) > 1e-6 ? fabs(got - want) / fabs(want)
                                        : fabs(got - want);
                if (rel < 1e-2) exact++;
                else {
                    if (rel > worst) worst = rel;
                    if (verbose && exact + untouched < 8)
                        printf("      c=%u y=%u x=%u  got %g want %g\n", c, y, x, got, want);
                }
            }
    *max_rel = worst;
    if (oob)
        printf("      %u/%u lanes index past the allocation — the layout under test "
               "does not fit the buffer\n", oob, total);
    if (untouched == total) { printf("      surface UNTOUCHED\n"); return -1; }
    if (zero + untouched == total) {
        /* Two causes, same surface: an epilogue that gated or underflowed (C wrong for
         * this precision), or a MAC contributing nothing. The bias probe separates
         * them — it removes the MAC, so anything landing there is the epilogue. */
        printf("      surface written but ENTIRELY ZERO — either the epilogue gated or\n"
               "      underflowed, or the MAC contributed nothing. ROCKET_FS_BIAS=1\n"
               "      separates them.\n");
        return -1;
    }
    printf("      %u/%u within 1%%, %u untouched, %u zero, worst relative error %.3g\n",
           exact, total, untouched, zero, worst);
    return (exact == total && !oob) ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "fp16";
    /* `bitsweep` is `f32out` with the candidate table replaced by one register walked
     * a bit at a time, which is how a re-packed field is found rather than guessed. */
    int bitsweep = !strcmp(mode, "bitsweep");
    /* `map` does not score. It drives the unique-naming probe and DECODES the surface,
     * so the output map is read off the part rather than inferred from which model
     * values a guessed map happens to reproduce. */
    int mapmode = !strcmp(mode, "map");
    /* `icsplit` works around the feature-surface defect instead of fixing it. The part
     * reduces correctly while a task takes ONE contraction step, so a conv is split into
     * ic slices of the feature cube's own lane count and the partial surfaces are summed
     * on the host. The slices are free to address: at C2 = 8 the cube's channel groups
     * are contiguous, so slice k is the same BO at a base offset. */
    int icsplit = !strcmp(mode, "icsplit");
    /* A sweep normally reads the 32-bit epilogue word, which is unambiguous but only
     * covers half the surface. Once the fp16 output narrows correctly, ROCKET_FS_SWEEP_F16
     * puts the sweep on the real output so a whole surface can be scored. */
    int sweep_f16 = bitsweep && getenv("ROCKET_FS_SWEEP_F16") != NULL;
    int out32 = !strcmp(mode, "f32out") || (bitsweep && !sweep_f16);
    int fp16 = !strcmp(mode, "fp16") || out32 || bitsweep || mapmode || icsplit;
    uint16_t sweep_reg = (uint16_t)env_u32("ROCKET_FS_SWEEP_REG", 0x100c);
    uint32_t sweep_base = env_u32("ROCKET_FS_SWEEP_BASE", 0x120);
    /* Overrides carried alongside every swept value, so one sweep can hold a second
     * register at a candidate setting instead of needing a rebuild per pair. */
    const char *sweep_with = getenv("ROCKET_FS_SWEEP_WITH");
    uint32_t sweep_vals[256];
    unsigned n_vals = 0, n_runs;
    int asym = !strcmp(mode, "asym");
    int probe = env_int("ROCKET_FS_PROBE", mapmode ? 8 : 0);
    int verbose = env_int("ROCKET_FS_VERBOSE", 0);
    const char *only = getenv("ROCKET_FS_ONLY");
    int gap = env_int("ROCKET_FS_GAP_MS", 0);
    unsigned c2 = fp16 ? 8u : 16u;
    unsigned elem = fp16 ? 2u : 1u;
    /* The output side is sized by the WIDTH FIELD, not by the datapath precision.
     * A 4-byte output element still fills a 16-byte atom, so C2 falls to 4. */
    enum out_kind okind = out32 ? OUT_F32 : (fp16 ? OUT_F16 : OUT_I8);
    unsigned out_c2 = (unsigned)env_int("ROCKET_FS_OUT_C2", (int)(out32 ? 4u : c2));
    unsigned out_elem = out32 ? 4u : elem;
    int fd, rc = 1, won = 0;
    size_t in_bytes, w_bytes, o_bytes, o_alloc, coeff;
    _Float16 *in_p = NULL, *w_p = NULL;
    float *biasf = NULL;
    int32_t *biasi = NULL;
    struct bufs B = {0};
    int16_t cmul;
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    unsigned i, c, y, x, kh, kw;

    LAYOUT = env_int("ROCKET_FS_LAYOUT", CUBE_RK3588_FP16);
    OUT_LAYOUT = env_int("ROCKET_FS_OUT_LAYOUT", OUTL_NATIVE);
    FEAT_C2    = (unsigned)env_int("ROCKET_FS_C2",
                                   LAYOUT == CUBE_LANES_FIXED ? 16 : (fp16 ? 8 : 16));
    /* The measured float groups. LAYOUT still selects the two published pairs so a
     * regression can be shown against them, but the default is what the part does. */
    W_OC_GROUP = (unsigned)env_int("ROCKET_FS_WOC",
                                   LAYOUT == CUBE_LANES_FIXED ? 32 : 8);
    W_IC_GROUP = (unsigned)env_int("ROCKET_FS_WIC",
                                   LAYOUT == CUBE_LANES_FIXED ? 32 : 16);
    IC     = (unsigned)env_int("ROCKET_FS_IC", 32);
    OC     = (unsigned)env_int("ROCKET_FS_OC", 32);
    IW     = (unsigned)env_int("ROCKET_FS_IW", 8);
    IH     = (unsigned)env_int("ROCKET_FS_IH", 8);
    K      = (unsigned)env_int("ROCKET_FS_K", 1);
    STRIDE = (unsigned)env_int("ROCKET_FS_STRIDE", 1);
    OW = (IW - K) / STRIDE + 1;
    OH = (IH - K) / STRIDE + 1;

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

    in_bytes = (size_t)(IC / c2) * IH * IW * c2 * elem;
    /* The weight cube is sized by its GROUP, not by ic: a partial input-channel group
     * still occupies a whole one, so ic=8 at an ic group of 16 needs a cube for 16.
     * Sizing this from ic alone under-allocates, and at k=1 the overrun stays inside
     * the BO's page and computes anyway — it only surfaces at a kernel big enough to
     * run past the page, which reads as "k=3 does not compute". */
    {
        unsigned gic = W_IC_GROUP > 1u ? W_IC_GROUP : 1u;
        unsigned goc = W_OC_GROUP > 1u ? W_OC_GROUP : 1u;
        unsigned ic_pad = ((IC + gic - 1u) / gic) * gic;
        unsigned oc_pad = ((OC + goc - 1u) / goc) * goc;
        if (!fp16) { ic_pad = ((IC + 31u) / 32u) * 32u; oc_pad = ((OC + 31u) / 32u) * 32u; }
        w_bytes = (size_t)ic_pad * oc_pad * K * K * elem;
    }
    o_bytes  = (size_t)((OC + out_c2 - 1) / out_c2) * OH * OW * out_c2 * out_elem;
    /* Always allocate for the widest output the width field can select. A sweep that
     * drives 0x4010 to a wider word than the mode expects then overruns the sentinel
     * fill rather than the buffer. */
    o_alloc  = (size_t)((OC + 3u) / 4u) * OH * OW * 4u * 4u;
    if (o_alloc < o_bytes) o_alloc = o_bytes;
    coeff    = rocket_rk3576_coeff_bytes(OC);

    in_p  = calloc((size_t)IC * IH * IW, sizeof *in_p);
    w_p   = calloc((size_t)OC * IC * K * K, sizeof *w_p);
    biasf = calloc(OC, sizeof *biasf);
    biasi = calloc(OC, sizeof *biasi);
    if (!in_p || !w_p || !biasf || !biasi) goto done;

    /* Small magnitudes: an fp16 accumulator over IC*K*K terms has to stay exact
     * enough that a 1% check is a real check and not a rounding argument. */
    for (c = 0; c < IC; c++)
        for (y = 0; y < IH; y++)
            for (x = 0; x < IW; x++)
                in_p[(c * IH + y) * IW + x] = (_Float16)(((int)(c + y + x) % 7) - 3);
    for (c = 0; c < OC; c++)
        for (i = 0; i < IC; i++)
            for (kh = 0; kh < K; kh++)
                for (kw = 0; kw < K; kw++)
                    w_p[((c * IC + i) * K + kh) * K + kw] =
                        (_Float16)(((int)(c * 3 + i * 5 + kh + kw) % 5) - 2);

    /* MAC-side probes. The bias probe below removes the MAC to read the epilogue;
     * these do the opposite — the bias stays zero, so every non-zero lane is the MAC's
     * and nothing else's.
     *
     *   1  uniform ones. The cheapest question: does the MAC produce ANYTHING under
     *      this program. Every output is the same value (ic*k*k), so a yes says the
     *      datapath runs and says nothing about where anything lands.
     *   2  one non-zero element on each side, at the origin. Exactly one output pixel
     *      of one channel is non-zero, so a yes turns into a position.
     *   3  a spatial ramp on one input channel. Output (0,y,x) names its own pixel,
     *      which is the mapping a bias probe cannot reach.
     */
    if (probe) {
        memset(in_p, 0, (size_t)IC * IH * IW * sizeof *in_p);
        memset(w_p,  0, (size_t)OC * IC * K * K * sizeof *w_p);
        if (probe == 1) {
            /* Uniform, but each side's value is its own knob. A datapath that reads a
             * side ONE BYTE AT A TIME answers to that side's fp16 BYTES rather than to
             * its value, so driving the two independently says which side is misread:
             * the value that the result tracks is the one read as fp16, and the value
             * whose high byte the result tracks is the one read as int8. */
            float pin = (float)env_dbl("ROCKET_FS_PIN", 1.0);
            float pw  = (float)env_dbl("ROCKET_FS_PW",  1.0);
            uint16_t bin, bw;
            _Float16 hin = (_Float16)pin, hw = (_Float16)pw;
            memcpy(&bin, &hin, sizeof bin);
            memcpy(&bw,  &hw,  sizeof bw);
            for (i = 0; i < IC * IH * IW; i++) in_p[i] = hin;
            for (i = 0; i < OC * IC * K * K; i++) w_p[i] = hw;
            printf("probe 1: feature %g (fp16 %04x), weights %g (fp16 %04x) — every "
                   "output is %g.\n", (double)pin, bin, (double)pw, bw,
                   (double)((float)hin * (float)hw * (float)(IC * K * K)));
        } else if (probe == 2) {
            in_p[0] = (_Float16)1.0f;
            w_p[0]  = (_Float16)1.0f;
            printf("probe 2: in[0,0,0] = w[0,0,0,0] = 1.0 — one output lane is 1.0, "
                   "the rest are 0.\n");
        } else if (probe == 3) {
            /* PIXEL naming. Every input channel carries the same spatial ramp, so the
             * answer does not depend on which feature lane the weight lane pairs with
             * — a lane mapping that is wrong still returns the pixel. That makes the
             * output pixel map readable on its own, before the lane map is known. */
            /* ROCKET_FS_TAP=-1 taps EVERY input channel instead of ic=0, which is the
             * one combination the rest of the probe set never makes: a feature that
             * varies per PIXEL contracted over MANY input channels at once. Every
             * other probe is uniform in one of those two axes or one-hot in the
             * weight, so all of them pass under a datapath that mixes pixels across
             * the input-channel reduction. */
            int tap3 = env_int("ROCKET_FS_TAP", 0);
            unsigned nt = tap3 < 0 ? IC : 1u, t0 = tap3 < 0 ? 0u : (unsigned)tap3;
            if (t0 >= IC) t0 = 0;
            for (c = 0; c < IC; c++)
                for (y = 0; y < IH; y++)
                    for (x = 0; x < IW; x++)
                        in_p[(c * IH + y) * IW + x] = (_Float16)(float)(y * IW + x + 1);
            for (c = 0; c < OC; c++)
                for (i = 0; i < nt; i++)
                    w_p[(c * IC + t0 + i) * K * K] = (_Float16)1.0f;
            printf("probe 3: every input channel carries the pixel ramp, %u tap%s at "
                   "ic=%u — output (c,y,x) = %u*(y*%u+x+1), whatever the lane map.\n",
                   nt, nt == 1 ? "" : "s", t0, nt, IW);
        } else if (probe == 6) {
            /* Which INPUT CHANNELS the one non-zero weight lane pairs with. Each input
             * channel carries its own index; the kernel taps ic=0 alone. The output is
             * then the sum of the channel indices the part actually read, so it names
             * the pairing instead of only showing that it is wrong. */
            for (c = 0; c < IC; c++)
                for (i = 0; i < IH * IW; i++)
                    in_p[c * IH * IW + i] = (_Float16)(float)(c + 1);
            for (c = 0; c < OC; c++) w_p[c * IC * K * K] = (_Float16)1.0f;
            printf("probe 6: input channel c carries c+1, kernel taps ic=0 — output "
                   "names the channels the part paired with lane 0 (want 1).\n");
        } else if (probe == 7) {
            /* The same question from the weight side: a uniform feature and a kernel
             * that ramps over ic. The output is the sum of the weight lanes the part
             * read, so 1+2+...+ic is the correct answer and anything else counts the
             * lanes it actually walked. */
            for (i = 0; i < IC * IH * IW; i++) in_p[i] = (_Float16)1.0f;
            for (c = 0; c < OC; c++)
                for (i = 0; i < IC; i++)
                    w_p[(c * IC + i) * K * K] = (_Float16)(float)(i + 1);
            printf("probe 7: uniform feature, kernel ramps over ic — output is the sum "
                   "of the weight lanes read (want %u).\n", IC * (IC + 1) / 2);
        } else if (probe == 8) {
            /* UNIQUE naming — the map probe. Every output lane carries a different
             * value, so a word read off the part names the (channel, pixel) it holds
             * instead of merely agreeing or disagreeing with a model. That is what
             * separates "the layout is wrong" from "the layout is wrong THIS way",
             * which no uniform probe can do: probes 3 and 5 are each uniform in the
             * axis they do not name, so either one passes under a map that is wrong
             * in the other axis.
             *
             * The value is ow*oh*c + p + 1, which runs 1..oc*ow*oh. fp16 represents
             * every integer up to 2048 exactly and none of the odd ones above it, so
             * the probe is only unambiguous while oc*ow*oh <= 2048 — checked below,
             * because a silently rounded name decodes as a neighbouring lane. */
            for (y = 0; y < IH; y++)
                for (x = 0; x < IW; x++) {
                    in_p[y * IW + x] = (_Float16)(float)(y * IW + x + 1);
                    in_p[IH * IW + y * IW + x] = (_Float16)1.0f;
                }
            for (c = 0; c < OC; c++) {
                w_p[c * IC * K * K] = (_Float16)1.0f;
                w_p[c * IC * K * K + K * K] = (_Float16)(float)(OW * OH * c);
            }
            printf("probe 8: unique naming — output (c,y,x) = %u*c + (y*%u+x+1), so "
                   "every lane names itself.\n", OW * OH, IW);
        } else if (probe == 10) {
            /* INPUT-CHANNEL MULTIPLICITY. One input channel carries 1.0, the rest are
             * zero, and every weight lane is 1.0 — so the output is the number of
             * times the part read that channel, and nothing else. Sweeping
             * ROCKET_FS_TAP over the channel count returns the multiplicity of each,
             * which is what separates "the reduction is wrong" from "the reduction
             * reads channel k twice and channel j never".
             *
             * Every other probe is blind to this: a uniform feature makes the total
             * depend only on how MANY lanes were read, and an ic-constant weight makes
             * the contraction invariant under any permutation of the channel axis. */
            unsigned tap = (unsigned)env_int("ROCKET_FS_TAP", 0);
            if (tap >= IC) tap = 0;
            for (i = 0; i < IH * IW; i++) in_p[tap * IH * IW + i] = (_Float16)1.0f;
            for (c = 0; c < OC; c++)
                for (i = 0; i < IC; i++)
                    w_p[(c * IC + i) * K * K] = (_Float16)1.0f;
            printf("probe 10: only input channel %u carries 1.0, every weight lane is "
                   "1.0 — the output IS that channel's multiplicity (want 1).\n", tap);
        } else if (probe == 9) {
            /* FEATURE-CUBE playback. Every input lane carries a unique name and output
             * channel c taps input channel c alone, so out(c,p) IS whatever the part
             * read at (c,p) — the input cube handed back through the datapath, under
             * the same decode as probe 8.
             *
             * This is the probe the uniform ones cannot replace. Probes 3, 5, 6 and 7
             * are each uniform in one of the two feature axes, so a cube that confuses
             * the channel axis with the pixel axis satisfies all of them and still
             * contracts the wrong lanes the moment the feature varies in both. */
            for (c = 0; c < IC; c++)
                for (y = 0; y < IH; y++)
                    for (x = 0; x < IW; x++)
                        in_p[(c * IH + y) * IW + x] =
                            (_Float16)(float)(OW * OH * c + y * IW + x + 1);
            for (c = 0; c < OC && c < IC; c++)
                w_p[(c * IC + c) * K * K] = (_Float16)1.0f;
            printf("probe 9: every input lane names itself and output channel c taps "
                   "input channel c — out(c,p) is the cube read back.\n");
        } else if (probe == 5) {
            /* CHANNEL naming, the same way round: a uniform feature makes the answer
             * independent of the lane map, and each kernel scales by its own index. */
            /* ROCKET_FS_TAP moves the single live weight lane off ic=0. That is the
             * one axis the probe set otherwise leaves dark: an output channel only
             * reaches DDR for c < 16 today, so every probe that taps ic=0 — or ic=c —
             * exercises the weight cube's FIRST input-channel group and nothing else.
             * A cube whose ic1>0 blocks are misplaced satisfies all of them and still
             * contracts the wrong weights as soon as a real kernel spans ic >= 16. */
            unsigned tap = (unsigned)env_int("ROCKET_FS_TAP", 0);
            if (tap >= IC) tap = 0;
            for (i = 0; i < IC * IH * IW; i++) in_p[i] = (_Float16)1.0f;
            for (c = 0; c < OC; c++)
                w_p[(c * IC + tap) * K * K] = (_Float16)(float)(c + 1);
            printf("probe 5: uniform feature, kernel c weighted c+1 at ic=%u — output "
                   "(c,y,x) = c+1, whatever the lane map.\n", tap);
        } else {
            /* Self-naming. Input channel 0 carries the pixel ramp and channel 1 a
             * constant 100; kernel c weights them 1 and c. Every output value is then
             * 100*c + pixel, so a lane READ OFF THE PART says which channel and which
             * pixel it holds — the one thing a uniform or single-element probe cannot,
             * and the only way to settle a cube layout rather than assume one. The
             * terms stay exact in an fp32 accumulator, so a mismatch is a layout fault
             * and never a rounding one. */
            for (y = 0; y < IH; y++)
                for (x = 0; x < IW; x++) {
                    in_p[y * IW + x] = (_Float16)(float)(y * IW + x + 1);
                    in_p[(IH + y) * IW + x] = (_Float16)100.0f;
                }
            for (c = 0; c < OC; c++) {
                w_p[c * IC * K * K] = (_Float16)1.0f;
                w_p[c * IC * K * K + K * K] = (_Float16)(float)c;
            }
            printf("probe 4: self-naming — output (c,y,x) = 100*c + (y*%u+x+1).\n", IW);
        }
    }

    /* Bias probe. Zero the weights and drive a per-channel bias, so the whole answer
     * comes from the DPU epilogue and nothing from the MAC. This separates the two
     * readings of an all-zero surface that scoring alone cannot: a gated BS stage
     * (the bias does not land either) from a MAC contributing nothing (it does). */
    if (env_int("ROCKET_FS_BIAS", 0)) {
        /* A is a 32-bit field and the packer fills it with an integer. Which TYPE the
         * epilogue reads it as is the question: at ROCKET_FS_BIAS=2 the same value
         * goes in as its fp32 bit pattern instead, so an integer datapath returns
         * garbage and a float one returns the value. */
        int as_float = env_int("ROCKET_FS_BIAS", 0) == 2;
        memset(w_p, 0, (size_t)OC * IC * K * K * sizeof *w_p);
        for (c = 0; c < OC; c++) {
            float f = (float)(c + 1);
            biasf[c] = f;
            if (as_float) memcpy(&biasi[c], &f, sizeof f);
            else          biasi[c] = (int32_t)(c + 1);
        }
        printf("bias probe: weights zeroed, bias[c] = c+1 as %s — the surface is the "
               "epilogue alone.\n", as_float ? "fp32 BITS" : "an integer");
    }

    if (rocket_bo_alloc(fd, in_bytes, &B.in) < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes,  &B.w)  < 0) goto done;
    if (rocket_bo_alloc(fd, coeff,    &B.b)  < 0) goto done;
    if (rocket_bo_alloc(fd, o_alloc,  &B.o)  < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &B.r) < 0) goto done;

    /* Scatter into the cubes. fp16 takes C2=8 and weight_conv_fp16; the int8 control
     * takes C2=16 and weight_conv_int8, which is the layout the part is known to
     * want, so a failure in that mode is the harness and not the encoding. */
    rocket_bo_prep(fd, &B.in, 1, 0);
    memset(B.in.ptr, 0, in_bytes);
    for (c = 0; c < IC; c++)
        for (y = 0; y < IH; y++)
            for (x = 0; x < IW; x++) {
                int idx = fs_feature_idx(c, y, x, c2);
                float v = (float)in_p[(c * IH + y) * IW + x];
                if (fp16) ((_Float16 *)B.in.ptr)[idx] = (_Float16)v;
                else      ((int8_t   *)B.in.ptr)[idx] = (int8_t)v;
            }
    rocket_bo_fini(fd, &B.in);

    rocket_bo_prep(fd, &B.w, 1, 0);
    memset(B.w.ptr, 0, w_bytes);
    for (c = 0; c < OC; c++)
        for (i = 0; i < IC; i++)
            for (kh = 0; kh < K; kh++)
                for (kw = 0; kw < K; kw++) {
                    float v = (float)w_p[((c * IC + i) * K + kh) * K + kw];
                    /* weight_conv_* take 1-based oc/ic/kh/kw, the same convention as
                     * feature_data above; at 0 the group remainder goes negative and
                     * the whole cube is misplaced. */
                    int idx = fs_weight_idx(c, i, kh, kw, fp16);
                    if (fp16) ((_Float16 *)B.w.ptr)[idx] = (_Float16)v;
                    else      ((int8_t   *)B.w.ptr)[idx] = (int8_t)v;
                }
    rocket_bo_fini(fd, &B.w);

    /* The coefficient buffer. `asym` is the whole point of the third mode: drive a
     * weight zero point far from zero, where the two sign conventions cannot be
     * confused, and report which one the part implements. */
    /* The C multiplier is an int16 bit pattern in the group, and nothing says the BS
     * stage reads it in the same precision the int8 path does. At fp16 the integer 1
     * is the denormal 6e-8, which scales the accumulator to zero and produces a full
     * but entirely empty surface — the same signature a gated stage gives. fp16 1.0
     * is 0x3C00. ROCKET_FS_CMUL drives it so the two readings can be separated. */
    cmul = (int16_t)env_int("ROCKET_FS_CMUL", fp16 ? 0x3C00 : 1);
    printf("coefficient C multiplier: 0x%04x (%d)\n", (uint16_t)cmul, cmul);

    rocket_bo_prep(fd, &B.b, 1, 0);
    if (asym) {
        int16_t wzp = (int16_t)env_int("ROCKET_FS_WZP", 64);
        int16_t *zps = calloc(OC, sizeof *zps);
        if (!zps) { rocket_bo_fini(fd, &B.b); goto done; }
        for (c = 0; c < OC; c++) zps[c] = wzp;
        rocket_rk3576_pack_coeff_asym(B.b.ptr, coeff, biasi, OC, zps, cmul);
        free(zps);
        printf("asym: weight zero point %d driven into B for every channel.\n"
               "      The model below assumes NO correction, so a surface that\n"
               "      matches it means B is inert; a surface off by -zp*sum(x)\n"
               "      or +zp*sum(x) names the sign convention.\n", wzp);
    } else {
        rocket_rk3576_pack_coeff_asym(B.b.ptr, coeff, biasi, OC, NULL, cmul);
    }
    rocket_bo_fini(fd, &B.b);

    in_h[0] = B.in.handle; in_h[1] = B.w.handle;
    in_h[2] = B.b.handle;  in_h[3] = B.r.handle;
    out_h[0] = B.o.handle;

    printf("RK3576 %s sweep — ic=%u oc=%u %ux%u k=%u s=%u -> %ux%u, feature C2=%u "
           "at %u B/element, output C2=%u at %u B/element\n",
           mode, IC, OC, IW, IH, K, STRIDE, OW, OH, c2, elem, out_c2, out_elem);
    printf("A wrong precision field does not fault: it writes a full, correctly sized,\n"
           "WRONG surface. Every line below is scored against a CPU model.\n\n");

    /* ============================================================================
     * ic-SPLIT — the working fp16 conv at an arbitrary input-channel count.
     * ==========================================================================*/
    if (icsplit) {
        /* Driven entirely through the LIBRARY: rocket_rk3576_plan_ic() lays the
         * slices out, rocket_rk3576_fp16_pack_slice_weights() builds each slice's
         * cube and rocket_rk3576_fp16_accumulate() de-scatters the partials. Nothing
         * here reimplements the split — a pass is the library's, not the harness's. */
        rocket_rk3576_ic_task slices[256];
        unsigned nslice = 0, s;
        unsigned nout = rocket_rk3576_fp16_out_channels(OC);
        /* Sized for the WIDEST slice the planner can hand back, which is the whole
         * channel count — the planner narrows only when the CBUF makes it. */
        size_t slice_w = rocket_rk3576_fp16_slice_weight_bytes(OC, IC, K, K);
        size_t surf = rocket_rk3576_fp16_out_bytes(OC, OH, OW);
        float *acc = calloc((size_t)nout * OH * OW, sizeof *acc);
        unsigned exact = 0, total = 0, retries = 0;
        double worst = 0.0;
        /* What the split costs, split three ways: the whole conv, the part of it the
         * NPU is actually running, and the host packing and de-scattering that the
         * ic slices force. The gap knob is probe pacing and is excluded from all
         * three — run at ROCKET_FS_GAP_MS=0 for a number that means anything. */
        double t_all = 0.0, t_npu = 0.0, t_host = 0.0;
        conv_params_t base = {0};

        if (!acc) goto done;
        base.ic = (uint16_t)IC; base.ih = (uint16_t)IH; base.iw = (uint16_t)IW;
        base.oc = (uint16_t)OC; base.oh = (uint16_t)OH; base.ow = (uint16_t)OW;
        base.kh = (uint16_t)K;  base.kw = (uint16_t)K;
        base.stride_y = (uint8_t)STRIDE; base.stride_x = (uint8_t)STRIDE;
        base.in_scale = 1.0f; base.w_scale = 1.0f; base.out_scale = 1.0f;
        base.input_zero_point = 0x80; base.output_zero_point = 0x80;
        base.weight_zero_point = 0x80;
        base.ih_full = (uint16_t)IH; base.oh_full = (uint16_t)OH;

        if (rocket_rk3576_plan_ic(&base, slices,
                                  sizeof slices / sizeof slices[0], &nslice) < 0) {
            printf("icsplit: the library refused to plan ic=%u\n", IC);
            free(acc); rc = 2; goto done;
        }
        if (slice_w > w_bytes || surf > o_alloc) {
            printf("icsplit: the harness under-allocated (weights %zu/%zu, surface "
                   "%zu/%zu)\n", slice_w, w_bytes, surf, o_alloc);
            free(acc); rc = 2; goto done;
        }
        printf("icsplit: %u library slices of %u input channels, summed on the host; "
               "the writer delivers %u of %u output channels.\n\n",
               nslice, (unsigned)slices[0].ic, nout, OC);

        for (s = 0; s < nslice; s++) {
            conv_params_t p = base;
            unsigned tries;
            double t0;

            /* This slice's weight cube: its own conv, so its own group count. */
            t0 = fs_now();
            rocket_bo_prep(fd, &B.w, 1, 0);
            if (rocket_rk3576_fp16_pack_slice_weights(B.w.ptr, w_bytes, w_p,
                                                      OC, IC, K, K, &slices[s]) < 0) {
                rocket_bo_fini(fd, &B.w);
                printf("  slice %u: the library refused to pack its weights\n", s);
                free(acc); goto done;
            }
            rocket_bo_fini(fd, &B.w);
            t_host += fs_now() - t0;

            p.ic          = slices[s].ic;
            p.tasks       = ops;
            p.input_dma   = B.in.dma_address + slices[s].feature_off;
            p.weights_dma = B.w.dma_address;
            p.bias_dma    = B.b.dma_address;
            p.output_dma  = B.o.dma_address;

            if (gen_conv2d_fp16_rk3576(&p) != 0) {
                printf("  slice %u: generator refused\n", s); free(acc); goto done;
            }
            rocket_bo_prep(fd, &B.r, 1, 0);
            memcpy(B.r.ptr, ops, p.task_count * sizeof(uint64_t));
            rocket_bo_fini(fd, &B.r);

            /* Back-to-back submits return untouched surfaces at a rate that depends
             * on the gap, so a no-write is a measurement to repeat rather than one to
             * record. The sentinel is what tells the two apart. */
            for (tries = 0; tries < 3u; tries++) {
                rocket_bo_prep(fd, &B.o, 1, 0);
                memset(B.o.ptr, SENTINEL, o_alloc);
                rocket_bo_fini(fd, &B.o);
                if (gap > 0) {
                    struct timespec ts = { gap / 1000, (long)(gap % 1000) * 1000000L };
                    nanosleep(&ts, NULL);
                }
                t0 = fs_now();
                if (rocket_submit_matmul(fd, &B.r, p.task_count, in_h, 4, out_h, 1,
                                         2000) != 0) {
                    printf("  slice %u: submit FAILED\n", s); free(acc); goto done;
                }
                if (rocket_bo_prep(fd, &B.o, 0, 2000000000ull) < 0) {
                    printf("  slice %u: PREP_BO timed out\n", s); free(acc); goto done;
                }
                if (!fs_all_sentinel(B.o.ptr, surf)) { t_npu += fs_now() - t0; break; }
                rocket_bo_fini(fd, &B.o);
                retries++;
                printf("  slice %u: surface UNTOUCHED, retrying\n", s);
            }
            if (tries == 3u) {
                printf("  slice %u: the surface stayed untouched over 3 submits — "
                       "raise ROCKET_FS_GAP_MS, and check dmesg for an IOMMU wedge\n", s);
                free(acc); goto done;
            }
            t0 = fs_now();
            if (rocket_rk3576_fp16_accumulate(acc, B.o.ptr, surf, OC, OH, OW) < 0) {
                rocket_bo_fini(fd, &B.o);
                printf("  slice %u: the library refused the surface\n", s);
                free(acc); goto done;
            }
            t_host += fs_now() - t0;
            rocket_bo_fini(fd, &B.o);
            printf("  slice %u of %u done\n", s, nslice);
        }
        t_all = t_npu + t_host;

        /* Score the SUM against a model of the whole convolution, over the channels
         * the writer reaches. Scoring past them would count the writer's own gap as
         * an ic-split failure. */
        for (c = 0; c < nout; c++)
            for (y = 0; y < OH; y++)
                for (x = 0; x < OW; x++) {
                    double want = model_out(in_p, w_p, NULL, c, y, x);
                    double got = acc[(c * OH + y) * OW + x];
                    double rel;
                    total++;
                    rel = fabs(want) > 1e-6 ? fabs(got - want) / fabs(want)
                                            : fabs(got - want);
                    if (rel < 1e-2) exact++;
                    else if (rel > worst) worst = rel;
                }
        printf("\n  %u/%u within 1%% over the %u channels the writer reaches "
               "(%u past it), worst relative error %.3g\n",
               exact, total, nout, OC - nout, worst);
        /* Priced against the ic/8 submits the split spends, not against one task —
         * the split is what the conv IS on this part today. Retries are reported
         * because they are excluded from the timing and would otherwise hide. */
        printf("  cost: %.2f ms over %u submits (%.2f ms each) — %.2f ms NPU, "
               "%.2f ms host packing and de-scatter; %u retries excluded\n",
               t_all * 1e3, nslice, t_all * 1e3 / (double)nslice,
               t_npu * 1e3, t_host * 1e3, retries);
        if (exact == total && total) {
            printf("  REPRODUCES THE MODEL — an fp16 conv at ic=%u via %u library "
                   "slices\n", IC, nslice);
            rc = 0;
        }
        free(acc);
        goto done;
    }

    {
        const char *lv = getenv("ROCKET_FS_SWEEP_VALS");
        while (lv && *lv && n_vals < 256) {
            char *end;
            sweep_vals[n_vals] = (uint32_t)strtoul(lv, &end, 0);
            if (end == lv) break;
            n_vals++;
            lv = end;
            while (*lv == ',' || *lv == ' ') lv++;
        }
    }
    n_runs = bitsweep ? (n_vals ? n_vals : 33u)
                      : ((fp16 && !mapmode)
                         ? sizeof FP16_CANDIDATES / sizeof FP16_CANDIDATES[0] : 1u);
    if (mapmode && OC * OW * OH > 2048u) {
        printf("map: oc*ow*oh = %u exceeds the 2048 integers fp16 represents exactly, "
               "so a lane name would round into its neighbour. Shrink the geometry.\n",
               OC * OW * OH);
        rc = 2; goto done;
    }
    if (bitsweep)
        printf("bit sweep of 0x%04x over base 0x%08x, %u submits.\n\n",
               sweep_reg, sweep_base, n_runs);

    for (i = 0; i < n_runs; i++) {
        const candidate_t *cand =
            (fp16 && !bitsweep && !mapmode) ? &FP16_CANDIDATES[i] : NULL;
        char set[256];
        conv_params_t p = {0};
        double worst = 0.0;
        int r;

        if (cand && only && !strstr(cand->name, only)) continue;
        if (bitsweep) {
            /* Either an explicit value list, or — with no list — one bit at a time on
             * top of the emitted word. A 3-bit precision field holding fp16 (2) IS a
             * single set bit, so walking the bits covers every position such a field
             * could sit at; a list is what walks a field through values it cannot
             * reach by setting one more bit. */
            uint32_t val;
            if (n_vals) {
                val = sweep_vals[i];
                printf("  == 0x%04x = 0x%08x ==\n", sweep_reg, val);
            } else {
                val = (i == 0) ? sweep_base : (sweep_base | (1u << (i - 1)));
                if (i && (sweep_base & (1u << (i - 1)))) {
                    printf("  == 0x%04x bit %2u == already set in the base, skipped\n",
                           sweep_reg, i - 1);
                    continue;
                }
                if (i == 0) printf("  == 0x%04x base  == 0x%08x\n", sweep_reg, val);
                else printf("  == 0x%04x bit %2u == 0x%08x\n", sweep_reg, i - 1, val);
            }
            if (sweep_f16)
                snprintf(set, sizeof set, "0x%04x=0x%x%s%s", sweep_reg, val,
                         sweep_with ? "," : "", sweep_with ? sweep_with : "");
            else
                snprintf(set, sizeof set, "0x4010=0x%x,0x%04x=0x%x%s%s",
                         (R76_OUTW_WORD32 << 29) |
                         ((unsigned)precision_float16 << 26) |
                         (unsigned)precision_float16, sweep_reg, val,
                         sweep_with ? "," : "", sweep_with ? sweep_with : "");
            printf("     overrides: %s\n", set);
            setenv("ROCKET_RK3576_SET", set, 1);
        } else if (cand) {
            char body[256];
            build_set(body, sizeof body, cand->id);
            /* The width override goes FIRST, so a candidate that names 0x4010 itself
             * still wins: the emitter applies the overrides in order and the last
             * write to a register is the one that stands. */
            if (out32)
                snprintf(set, sizeof set, "0x4010=0x%x%s%s",
                         (R76_OUTW_WORD32 << 29) |
                         ((unsigned)precision_float16 << 26) |
                         (unsigned)precision_float16,
                         *body ? "," : "", body);
            else
                snprintf(set, sizeof set, "%s", body);
            printf("  == %s ==\n     %s\n", cand->name, cand->why);
            if (*set) printf("     overrides: %s\n", set);
            if (*set) setenv("ROCKET_RK3576_SET", set, 1);
            else      unsetenv("ROCKET_RK3576_SET");
        } else {
            /* The non-candidate modes still take ROCKET_FS_SET, so `map` and `int8`
             * can be run against an override without a candidate table entry. */
            const char *s = getenv("ROCKET_FS_SET");
            printf("  == %s control ==\n", mode);
            if (s && *s) {
                printf("     overrides: %s\n", s);
                setenv("ROCKET_RK3576_SET", s, 1);
            } else {
                unsetenv("ROCKET_RK3576_SET");
            }
        }

        p.ic = (uint16_t)IC; p.ih = (uint16_t)IH; p.iw = (uint16_t)IW;
        p.oc = (uint16_t)OC; p.oh = (uint16_t)OH; p.ow = (uint16_t)OW;
        p.kh = (uint16_t)K;  p.kw = (uint16_t)K;
        p.stride_y = (uint8_t)STRIDE; p.stride_x = (uint8_t)STRIDE;
        /* An integer output is requantized and saturated; a float one is not. At
         * out_scale 1.0 the requant is the identity, so the only difference the
         * model has to carry is the int8 clamp, which score() applies. */
        p.int8_out = fp16 ? 0 : 1;
        p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;
        p.input_zero_point = 0x80; p.output_zero_point = 0x80;
        p.weight_zero_point = 0x80;
        p.ih_full = (uint16_t)IH; p.oh_full = (uint16_t)OH;
        p.tasks       = ops;
        p.input_dma   = B.in.dma_address;
        p.weights_dma = B.w.dma_address;
        p.bias_dma    = B.b.dma_address;
        p.output_dma  = B.o.dma_address;

        /* The UNCHECKED entry, deliberately: gen_conv2d_fp16_rk3576() refuses past
         * ic=8, and the defect that bound it there is exactly what these modes drive.
         * The caller-facing path is exercised by `icsplit` above. */
        r = fp16 ? gen_conv2d_rk3576_prec(&p, 0, precision_float16)
                 : gen_conv2d_int8_rk3576(&p);
        if (r != 0) { printf("     generator refused: %d\n", r); continue; }

        rocket_bo_prep(fd, &B.r, 1, 0);
        memcpy(B.r.ptr, ops, p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &B.r);

        /* Submit, and RE-SUBMIT a surface that came back untouched. Consecutive
         * submits alternate written / not-written on this part, so a sweep that takes
         * one submit per value silently tests only half of them — every no-write reads
         * as "this value changed nothing". Retrying is what makes a bit sweep mean
         * anything; a value that never writes across all attempts is reported as such
         * and is a result in its own right. */
        {
            int attempt, wrote = 0;
            for (attempt = 0; attempt < 3 && !wrote; attempt++) {
                unsigned j, n = (unsigned)(o_alloc / 4);
                const uint32_t *raw;

                rocket_bo_prep(fd, &B.o, 1, 0);
                memset(B.o.ptr, SENTINEL, o_alloc);
                rocket_bo_fini(fd, &B.o);

                if (gap > 0) {
                    struct timespec ts = { gap / 1000, (long)(gap % 1000) * 1000000L };
                    nanosleep(&ts, NULL);
                }
                if (rocket_submit_matmul(fd, &B.r, p.task_count, in_h, 4, out_h, 1,
                                         2000) != 0) {
                    printf("     submit FAILED\n"); break;
                }
                if (rocket_bo_prep(fd, &B.o, 0, 2000000000ull) < 0) {
                    printf("     PREP_BO timed out\n"); break;
                }
                raw = (const uint32_t *)B.o.ptr;
                for (j = 0; j < n; j++)
                    if (raw[j] != 0xAAAAAAAAu) { wrote = 1; break; }
                if (!wrote) {
                    rocket_bo_fini(fd, &B.o);
                    if (attempt + 1 < 3) printf("     no write — resubmitting\n");
                }
            }
            if (!wrote) { printf("     surface UNTOUCHED after 3 submits\n\n"); continue; }
        }
        /* Raw surface dump. Scoring reads the buffer through the layout under test, so
         * it cannot show a layout that disagrees with it; the raw words can. Paired
         * with a bias probe whose value names its channel, this reads the output
         * atom straight off the part instead of inferring it from a mismatch. */
        {
            int dump = env_int("ROCKET_FS_DUMP", 0);
            int off  = env_int("ROCKET_FS_DUMP_OFF", 0);
            if (dump > 0 && okind == OUT_F32) {
                const uint32_t *raw = (const uint32_t *)B.o.ptr;
                int total_w = (int)(o_alloc / 4), n = dump, j;
                if (off > total_w) off = total_w;
                if (off + n > total_w) n = total_w - off;
                printf("     raw output dwords [%d,%d) of %d (fp32 value alongside):\n",
                       off, off + n, total_w);
                for (j = 0; j < n; j++) {
                    float f;
                    memcpy(&f, &raw[off + j], sizeof f);
                    if (j % 4 == 0) printf("       %4d:", off + j);
                    printf("  %08x %-12g", raw[off + j],
                           raw[off + j] == 0xAAAAAAAAu ? 0.0 : (double)f);
                    if (j % 4 == 3) printf("\n");
                }
                if (n % 4) printf("\n");
            } else if (dump > 0) {
                const uint16_t *raw = (const uint16_t *)B.o.ptr;
                int total_w = (int)(o_alloc / 2), n = dump, j;
                if (off > total_w) off = total_w;
                if (off + n > total_w) n = total_w - off;
                printf("     raw output words [%d,%d) of %d:\n", off, off + n, total_w);
                for (j = 0; j < n; j++) {
                    if (j % 16 == 0) printf("       %4d:", off + j);
                    printf(" %04x", raw[off + j]);
                    if (j % 16 == 15) printf("\n");
                }
                if (n % 16) printf("\n");
            }
        }
        /* Surface scan. A probe that puts ONE non-zero value on each side produces one
         * non-zero output lane, and its index is the layout — but only if something
         * reports where it landed, which scoring through an assumed layout cannot. The
         * written extent is the other half of the answer: a DPU that stops short has a
         * geometry or stride fault rather than an arithmetic one. */
        {
            int scan = env_int("ROCKET_FS_SCAN", 0);
            if (scan > 0) {
                const uint32_t *raw = (const uint32_t *)B.o.ptr;
                unsigned n = (unsigned)(o_alloc / 4), j;
                unsigned written = 0, nz = 0, first = n, last = 0, shown = 0;
                for (j = 0; j < n; j++) {
                    if (raw[j] == 0xAAAAAAAAu) continue;
                    written++;
                    if (j < first) first = j;
                    if (j > last)  last = j;
                    if (raw[j] == 0) continue;
                    nz++;
                    if (shown < (unsigned)scan) {
                        float f;
                        memcpy(&f, &raw[j], sizeof f);
                        printf("     nonzero dword %u = %08x (%g)\n", j, raw[j], (double)f);
                        shown++;
                    }
                }
                printf("     scan: %u of %u dwords written [%u..%u], %u non-zero\n",
                       written, n, written ? first : 0, written ? last : 0, nz);
            }
        }
        /* The map read straight off the part. Every lane names itself, so this walks
         * the surface once and reports, per output channel, WHERE that channel's
         * pixels landed and how many times each was written — the duplication, the
         * pixel order and the extent in one table, none of it assumed. */
        if (mapmode) {
            const uint16_t *raw = (const uint16_t *)B.o.ptr;
            unsigned nwords = (unsigned)(o_alloc / 2), j;
            unsigned npix = OW * OH, undecoded = 0, nwritten = 0;
            int *firstw = malloc((size_t)OC * npix * sizeof *firstw);
            unsigned *nhit = calloc((size_t)OC * npix, sizeof *nhit);

            if (!firstw || !nhit) { free(firstw); free(nhit); goto done; }
            for (j = 0; j < OC * npix; j++) firstw[j] = -1;
            for (j = 0; j < nwords; j++) {
                _Float16 h;
                double v;
                long n;
                if (raw[j] == 0xAAAAu) continue;
                nwritten++;
                memcpy(&h, &raw[j], sizeof h);
                v = (double)h;
                n = lround(v);
                if (fabs(v - (double)n) > 0.25 || n < 1 ||
                    (unsigned long)n > (unsigned long)OC * npix) {
                    if (undecoded < 8)
                        printf("     word %u does not name a lane: %g (%04x)\n",
                               j, v, raw[j]);
                    undecoded++;
                    continue;
                }
                {
                    unsigned c_ = (unsigned)(n - 1) / npix, p_ = (unsigned)(n - 1) % npix;
                    unsigned k = c_ * npix + p_;
                    if (firstw[k] < 0) firstw[k] = (int)j;
                    nhit[k]++;
                }
            }
            printf("     %u of %u words written, %u did not decode\n",
                   nwritten, nwords, undecoded);
            printf("     channel: [words holding pixel 0, pixel 1, pixel 2]  "
                   "copies/lane  pixels present\n");
            for (c = 0; c < OC; c++) {
                unsigned present = 0, copies = 0, p_;
                for (p_ = 0; p_ < npix; p_++)
                    if (nhit[c * npix + p_]) { present++; copies += nhit[c * npix + p_]; }
                printf("       c%-3u [%6d %6d %6d]  %-4.1f  %u/%u\n", c,
                       firstw[c * npix + 0],
                       npix > 1 ? firstw[c * npix + 1] : -1,
                       npix > 2 ? firstw[c * npix + 2] : -1,
                       present ? (double)copies / present : 0.0, present, npix);
            }
            free(firstw); free(nhit);
            rocket_bo_fini(fd, &B.o);
            won = 1;
            printf("\n");
            continue;
        }
        r = score(B.o.ptr, okind, in_p, w_p, biasf, out_c2,
                  o_alloc / (okind == OUT_F32 ? 4u : (okind == OUT_F16 ? 2u : 1u)),
                  &worst, verbose);
        rocket_bo_fini(fd, &B.o);
        if (r == 0) {
            printf("     REPRODUCES THE MODEL\n");
            won = 1;
            if (cand) printf("     -> fold this into the emitter: %s\n",
                             *set ? set : "the default program is correct");
        }
        printf("\n");
    }

    unsetenv("ROCKET_RK3576_SET");
    rc = won ? 0 : 1;
    if (!won && fp16)
        printf("No candidate reproduced the model. What is left is the OUTPUT PACKING, "
               "not the arithmetic:\n"
               "the contraction is exact (ROCKET_FS_PROBE=1/6/7 all return the model), "
               "and the DPU writes\neach result dword TWICE, so a 16-byte atom carries "
               "4 output channels where it should\ncarry 8 and only the first 16 "
               "channels reach DDR. ROCKET_FS_PROBE=5 reads that off the\npart: the "
               "groups hold channels 1-4, 5-8, 9-12, 13-16 and stop.\n");
done:
    bufs_free(fd, &B);
    free(in_p); free(w_p); free(biasf); free(biasi);
    rocket_close(fd);
    return rc;
}
