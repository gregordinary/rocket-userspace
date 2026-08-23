// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_pool_rk3576.c — POOLING for the RK3576, behind a library entry.
 *
 * gen_pool_rk3576() emits the PPU's own 31-write program and computes bit-exactly, and
 * for as long as it was reachable only from tests/rk3576_pool_probe.c a caller with a
 * row-major tensor could not use it: the NC1HWC2 cube, the sentinel, the submit and the
 * de-scatter all lived in the probe. This file is that layer, and it is the same shape
 * as rocket_conv2d_rk3576.c's — one entry, row-major in and out, the part's submit
 * discipline owned here rather than by the caller.
 *
 * WHAT THE PART COMPUTES, and so what this entry states:
 *
 *   int8 -> int8, a window reduction with NO requant in the path. The PPU is not a
 *   convolution epilogue on this part: it is 23 PPU writes and 8 PPU_RDMA, no CNA, no
 *   CORE, no DPU, reading and writing the same 16-byte-atom cube the convolution path
 *   already packs.
 *
 *   THE AVERAGE ROUNDS HALF TO EVEN, which is where this diverges from the RK3588's
 *   rocket_pool_int8(): that one truncates toward zero. Two different roundings are two
 *   different functions, so the RK3588 entry refuses here and names this one rather than
 *   dispatching — the same answer rocket_conv2d_int8() and rocket_matmul_int8() give.
 *
 *   THE DIVISOR IS THE WINDOW, NOT THE TAP COUNT. The PPU has no divider; it multiplies
 *   the window sum by a per-axis Q16 reciprocal, 0x10000/kw and 0x10000/kh. So
 *   POOL_METHOD_AVG divides by kh*kw whatever the padding excluded — TFLite's
 *   count-include-pad = TRUE — and ROCKET_RK3576_POOL_AVG_NOPAD is the mode bit that
 *   drops the pad taps instead. THAT RECIPROCAL IS TRUNCATED, and the error it leaves
 *   grows with the window sum: at k2/k3/k5 it is far under half a count over any int8
 *   window and the result is the exact rounded average, and at larger windows it is not
 *   guaranteed to be. rocket_pool_int8_rk3576_exact() answers that question for a
 *   descriptor without running it, from the reciprocal the emitter will program and the
 *   worst-case int8 window sum.
 *
 * THE PAD VALUE IS THE INPUT ZERO POINT on the average path, which is why this entry
 * takes one where the RK3588's does not: a quantized average pool whose input zero point
 * is not zero pads with that value, and the emitter carries the field.
 *
 * A HANDLE IS THE SHAPE, NOT AN OPTIMIZATION OF IT. Everything below runs through
 * rocket_pool_int8_pack_rk3576(): the transient entry packs, runs once and frees, so
 * there is one arithmetic and one submit discipline rather than two that have to be kept
 * in agreement. What a handle holds is what does not depend on the input — the three BOs,
 * the register program, the plan — and what it saves on a graph is the BO churn, which is
 * what the convolution path's residency lever turned out to be.
 *
 * ROCKET_RK3576_POOL_PROF=1 logs one line per call, at ROCKET_LOG_INFO.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_pool.h"
#include "rocket_conv.h"     /* rocket_rk3576_cube */
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define R76P_HAVE_NEON 1
#endif

#define C2 16u      /* the int8 feature/output channel atom */

static unsigned r76p_round4(unsigned v) { return (v + 3u) & ~3u; }

/* The number of taps that fell INSIDE the plane on one axis, at output position `o`. A
 * negative position clamps to zero, which is what the leading edge needs. */
static int r76p_inplane(int o, int stride, int k, int pad, int extent)
{
    int lo, hi;
    if (o < 0) o = 0;
    lo = o * stride - pad;
    hi = lo + k;
    if (lo < 0) lo = 0;
    if (hi > extent) hi = extent;
    return hi > lo ? hi - lo : 0;
}

/*
 * AN EXACT TIE BREAKS TOWARD ZERO WHEN THE RECIPROCAL IS TRUNCATED, and to even when it
 * is not. The PPU multiplies by 0x10000/kw and 0x10000/kh, both floored, so unless each
 * divisor divides 65536 the product is a hair UNDER 1/n and a sum that lands exactly on a
 * half falls to the smaller magnitude — where an exact reciprocal leaves the tie standing
 * for the round-half-to-even rule to break.
 *
 * Nothing before the padded average could see this: every window in the corpus is either
 * odd (no tie exists) or a power of two on both axes (the reciprocal is exact). Excluding
 * the pad reaches 3x2, which is the first even window with a truncated reciprocal.
 *
 * The wrap to int8 is deliberate and is not a clamp: it is what the part does, and it is
 * what makes a window divided by 6 where 9 was meant read as +79 rather than as -128.
 * [HW sweep, H96 MAX M9, tests/rk3576_pool_probe.c gate]
 */
/* ONE ARITHMETIC, TWO CALLING CONVENTIONS. The divisor lag's check scores tens of
 * thousands of elements per call at a handful of output POSITIONS, and everything here
 * that costs — the quotient and the two modulos behind `exact_recip` — is a function of
 * the position alone. So the divisor is a value a caller may build once and reuse
 * (r76p_div_init) and this entry is the same function with the build inlined, rather than
 * a second implementation to keep in agreement. */
struct r76p_div { long n, half; uint64_t magic; int even, exact_recip; };

static void r76p_div_init(struct r76p_div *v, int dw, int dh)
{
    long n = (long)dw * dh;
    if (n <= 0) { n = 1; dw = dh = 1; }
    v->n = n;
    v->half = n / 2;
    /* AN EXACT RECIPROCAL OVER THIS DOMAIN, not an approximation: with M = 2^32/n + 1 the
     * error in `x*M >> 32` is under `x / 2^32`, so the quotient is right for every
     * x < 2^32/n. A window sum plus its half is under 2^16 and n is a pooling window, so
     * the margin is four orders of magnitude. */
    v->magic = (uint64_t)(((uint64_t)1 << 32) / (uint64_t)n) + 1u;
    v->even = (n & 1) == 0;
    v->exact_recip = (0x10000 % dw) == 0 && (0x10000 % dh) == 0;
}

static int8_t r76p_avg_round_div(long sum, const struct r76p_div *v)
{
    long a = sum >= 0 ? sum : -sum;
    long q = (long)(((uint64_t)(a + v->half) * v->magic) >> 32);
    if (sum < 0) q = -q;
    if (v->even) {
        long r = sum - q * v->n;
        if ((r == v->half || r == -v->half) && (!v->exact_recip || (q & 1)))
            q += (sum >= 0) ? -1 : 1;
    }
    return (int8_t)q;
}

static int8_t r76p_avg_round(long sum, int dw, int dh)
{
    struct r76p_div v;
    r76p_div_init(&v, dw, dh);
    return r76p_avg_round_div(sum, &v);
}

/* What a handle freezes and why each is not bookkeeping — the same three the convolution
 * path's handle does. The fd, because a BO belongs to its file and an IOVA is per-fd. The
 * geometry, because the register program is generated from it once. The input zero point,
 * because the average path's pad value is folded into that program.
 *
 * `src` is a BORROWED cube: when cube_in is set the handle reads a producer's output
 * surface directly and its own feature cube is dropped, so an inference does no scatter
 * and no cache maintenance on the input side at all. */
struct rocket_pool_int8_rk3576_handle {
    int              fd;
    rocket_pool_desc d;
    int              in_zp;
    unsigned         c, ih, iw, oh, ow, creg;
    unsigned         in_surf, out_surf;      /* elements per channel group */
    size_t           src_off;                /* the input cube's byte offset in `src` */
    size_t           in_bytes, out_bytes;
    rocket_bo        in, out, rc;            /* `in` unallocated when cube_in */
    rocket_bo        src;                    /* borrowed; valid when cube_in */
    int              cube_in;
    unsigned         src_pitch;              /* elements per source row; 0 = iw */
    unsigned         src_col;                /* first column of the plane in those rows */
    /* The output surface is left where the PPU wrote it and a consumer reads it as its
     * feature cube, so there is no de-scatter and `out` may be NULL. `out_ext` is a
     * caller's buffer this handle writes its own slice of — borrowed, never freed here,
     * and while it is in use this handle's own `out` is released. */
    int              cube_out;
    rocket_bo        out_ext;
    size_t           out_off;
    uint32_t         task_count;
    /* A COLUMN SLICE reads a window of the caller's row-major input and writes a window
     * of its output, so a plane too wide for one task is N handles over ONE tensor with
     * no copy at either end. Zero pitch means "the plane is the tensor", which is every
     * handle that is not a slice. */
    unsigned         rm_in_col, rm_in_pitch;
    unsigned         rm_out_col, rm_out_pitch;
    /* HOW OFTEN THIS HANDLE'S DIVISOR HAS LAGGED, over its whole life. The hazard's rate
     * is what its cost is a function of, and a rate averaged over a geometry cannot say
     * whether every pool of that shape lags or one of them does — which is the difference
     * between a property of the part and a property of one layer's placement. Counted per
     * HANDLE so a caller can print it per layer. */
    unsigned         lag_fires, lag_calls;
    unsigned long    lag_discr;       /* summed over every check, RE readout only */
    /* A PARENT owns slices and no BOs. Its geometry is the caller's whole plane, which is
     * what rocket_pool_int8_rk3576_ow_slices() and the refusals below are stated over. */
    unsigned         nslice;
    struct rocket_pool_int8_rk3576_handle *slice[ROCKET_RK3576_POOL_MAX_SLICES];
};

struct r76p_prof {
    int      on;
    double   pack_us, scatter_us, stamp_us, submit_us, read_us, descat_us, free_us;
    unsigned attempts;
    /* THE DIVISOR LAG, SPLIT INTO ITS TWO COSTS, because they are levered separately: the
     * CHECK is arithmetic over the observable positions plus (on a cube join) one bracket
     * over the producer's surface, and is paid on every call the hazard can show on; a
     * REDO is a whole submit and is paid at the hazard's RATE. A knob that lowered the rate
     * would collect the second and none of the first; a cheaper check collects the first
     * and none of the second. `lag_us` is inside `read_us` as well, which is the interval
     * it sits in — subtract it to read the write check alone. */
    double   lag_us;
    unsigned lag_checks, lag_redos;
    /* AND THE WRITE CHECK SPLIT THE SAME WAY, because `read_us` minus `lag_us` is not one
     * quantity either. The interval holds the FENCE WAIT (a PREP_BO that is also the
     * invalidate, and which a row-major-out call would pay for its de-scatter regardless),
     * the sentinel SCAN (early-exit, so it stops at the first byte the program wrote), and
     * the closing FINI. Only the last two plus `stamp_us` are the guard; the wait is the
     * part finishing. `surf_kib` is what the two brackets walk, so the bracket cost model
     * prices them instead of a guess. */
    double   wait_us, scan_us, fini_us;
    double   surf_kib;
};

static int r76p_prof_on(void)
{
    const char *e = getenv("ROCKET_RK3576_POOL_PROF");
    return (e && *e && *e != '0');
}

static double r76p_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

#define R76P_T(p)          ((p).on ? r76p_now_us() : 0.0)
#define R76P_ACC(p, f, t0) do { if ((p).on) (p).f += r76p_now_us() - (t0); } while (0)

/* How many times a task that wrote nothing is redone. The guard a wide-output job leaves
 * behind is probabilistic — a confirmed power-domain collapse clears it about 87% of the
 * time — so the redo count is the lever. Eight matches the conv and matmul paths. */
static unsigned r76p_attempts(void)
{
    const char *e = getenv("ROCKET_RK3576_TASK_ATTEMPTS");
    long v = e && *e ? strtol(e, NULL, 0) : 8;
    if (v < 1) v = 1;
    if (v > 64) v = 64;
    return (unsigned)v;
}

/* Whether a handle that owns its own feature cube packs it at the plane's own element
 * count rather than the round4 the vendor's pooling programs carry. It exists to make
 * the alignment a GATED question rather than an assumption: a cube-in join hands
 * PPU_RDMA a producer's surface stride verbatim, and a direct convolution's is `ow*oh`
 * exactly, so whether the PPU takes a stride that is not a multiple of four decides
 * whether the join is available at all. ROCKET_RK3576_POOL_PACK_SRC=1. */
static int r76p_pack_src(void)
{
    const char *e = getenv("ROCKET_RK3576_POOL_PACK_SRC");
    return e && *e && *e != '0';
}

/* WHERE THIS HANDLE'S OUTPUT SURFACE IS: its own BO, or the slice of a caller's buffer it
 * was placed in. The stamp, the write check and the de-scatter all address it through
 * here, so a placed handle never touches the bytes outside its own slice — the rest of
 * that buffer belongs to another producer and has already been written this inference. */
static rocket_bo *r76p_surf(struct rocket_pool_int8_rk3576_handle *h, size_t *off)
{
    if (h->out_ext.ptr) { *off = h->out_off; return &h->out_ext; }
    *off = 0;
    return &h->out;
}

static int r76p_is_this_chip(const char *entry)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (hw == &rocket_hw_rk3576) return 1;
    ROCKET_LOGE("%s emits the RK3576 PPU program and the active profile is %s\n",
                entry, hw && hw->name ? hw->name : "unknown");
    return 0;
}

/*
 * HOW MANY OUTPUT COLUMNS ONE POOLING TASK MAY PRODUCE.
 *
 * Past this the part writes a full, correctly sized surface that is WRONG FROM COLUMN 0,
 * at every height and every channel count — so it is refused here rather than computed.
 * The wall is on the WIDTH alone: a 33-wide plane is exact at every height up to 112, and
 * a 34-wide one is wrong at a height of 2.
 *
 * MEASURED, as the largest exact output width and the smallest wrong one
 * (`tests/rk3576_pool_probe.c rect`, max, c=32, both pads):
 *
 *     k3 s1   ow 32 exact, 33 wrong        k2 s1   ow 64 exact, 65 wrong
 *     k3 s2   ow 64 exact, 65 wrong        k5 s1   ow 14 exact, 30 wrong
 *     k3 s3   ow 49 exact (no wall found in a plane of 147)
 *
 * What fits all five is an allowance that HALVES for each additional output window that
 * overlaps a given input column, from 128 at no overlap — a line-buffer budget shared
 * across the `ceil(kw/stride_x)` columns in flight [hypothesis; five (k, stride) points,
 * and the k5 rung is bracketed rather than pinned]. The formula is used as a BOUND and is
 * deliberately under the measurement where the two differ: at k5 s1 it allows 8 where 14
 * is exact, because being under costs a refusal and being over costs a silent wrong
 * surface. Nothing in the corpus before Inception V3 reached it — every pool ever gated
 * here is either stride 2 at a plane of 112 or less, or a global window.
 */
unsigned rocket_pool_int8_rk3576_max_ow(const rocket_pool_desc *d)
{
    unsigned overlap, allow;
    if (!d || d->kh <= 0 || d->stride_y <= 0) return 0u;
    /* THE AXIS IS THE KERNEL HEIGHT, NOT THE WIDTH, and a square kernel cannot say which:
     * every cell the wall was first characterised on had kh == kw, so the two readings
     * agreed everywhere. Separated by running the halves as a PAIR at one plane
     * (`rk3576_pool_probe rect` with ROCKET_POOLB_KH / _KW), VALID pad, height 8, c=32 —
     * last exact / first wrong output width:
     *
     *   k3x1 (hxw)  32 / 33      the width reading allows 128, so it was over by 4x
     *   k1x3        no wall to 145
     *   k4x2        33 / 34      the width reading allows 64, over by 2x
     *   k2x4        63 / 68
     *
     * Keyed on the height this reproduces all four AND every square cell the original
     * table carries (k3 s1 -> 32, k2 s1 -> 64, k3 s2 -> 64, k5 s1 exact to 14, k3 s3 no
     * wall in 147), and it is conservative rather than over wherever the two differ. It
     * also fits the line-buffer story the halving came from: what shares the budget is the
     * ceil(kh / stride_y) output ROWS in flight over a given input row.
     * [HW sweep, H96 MAX M9, 2026-08-03] */
    overlap = ((unsigned)d->kh + (unsigned)d->stride_y - 1u) / (unsigned)d->stride_y;
    if (overlap < 1u) overlap = 1u;
    if (overlap > 8u) return 1u;
    allow = 128u >> (overlap - 1u);
    return allow ? allow : 1u;
}

static int r76p_ow_probe(void)
{
    const char *e = getenv("ROCKET_RK3576_POOL_OW_PROBE");
    return e && *e && *e != '0';
}

int rocket_pool_int8_rk3576_plan(const rocket_pool_desc *d)
{
    if (!d) return ROCKET_E_SHAPE;
    if (d->c <= 0 || d->ih <= 0 || d->iw <= 0 || d->kh <= 0 || d->kw <= 0 ||
        d->stride_y <= 0 || d->stride_x <= 0)
        return ROCKET_E_SHAPE;
    if (d->pad_top < 0 || d->pad_left < 0 || d->pad_bottom < 0 || d->pad_right < 0)
        return ROCKET_E_SHAPE;
    if (d->method != POOL_METHOD_MAX && d->method != POOL_METHOD_AVG)
        return ROCKET_E_SHAPE;
    if (d->avg_exclude_pad && d->method != POOL_METHOD_AVG)
        return ROCKET_E_SHAPE;      /* nothing to exclude from a max */
    /* The kernel and stride fields are four bits each; a larger window cascades. */
    if (d->kh > 16 || d->kw > 16 || d->stride_y > 16 || d->stride_x > 16)
        return ROCKET_E_UNSUPPORTED;
    if (d->pad_top > 255 || d->pad_left > 255 ||
        d->pad_bottom > 255 || d->pad_right > 255)
        return ROCKET_E_UNSUPPORTED;
    if (rocket_pool_oh(d) <= 0 || rocket_pool_ow(d) <= 0)
        return ROCKET_E_SHAPE;
    /* PAST THE ALLOWANCE THE PLANE IS SPLIT BY COLUMNS, so the only refusal left here is
     * a plane needing more slices than a handle carries. ROCKET_RK3576_POOL_OW_PROBE
     * forces ONE task at the caller's full width so the wall can still be READ OUT — an
     * RE escape and not a mode: what it admits writes a full surface that is wrong from
     * column 0. */
    if (r76p_ow_probe()) return ROCKET_OK;
    if (rocket_pool_int8_rk3576_ow_slices(d) == 0u) {
        ROCKET_LOGE("rocket_pool_int8_rk3576: an output width of %d needs more than %d "
                    "column slices at this part's per-task allowance of %u for a %dx%d "
                    "window at stride %d\n",
                    rocket_pool_ow(d), ROCKET_RK3576_POOL_MAX_SLICES,
                    rocket_pool_int8_rk3576_max_ow(d),
                    d->kh, d->kw, d->stride_x);
        return ROCKET_E_UNSUPPORTED;
    }
    return ROCKET_OK;
}

/*
 * THE COLUMN SPLIT'S GEOMETRY, in one place because three callers need it: the plan
 * function's refusal, the pack that builds the slices, and any caller pricing the shape.
 *
 * Slice `s` produces output columns [x0, x0+ow_s). The input columns its windows reach
 * are [x0*stride - pad_left, (x0+ow_s-1)*stride + kw - pad_left), so a slice after the
 * first starts at a real column and carries NO leading pad, and only the last carries the
 * trailing one. Its own descriptor's derived width comes back out as ow_s exactly, which
 * is what makes a slice an ordinary handle rather than a special case.
 */
static unsigned r76p_slice_plan(const rocket_pool_desc *d, unsigned s,
                                unsigned *x0, unsigned *ow_s, unsigned *in_col,
                                unsigned *iw_s, int *pad_l, int *pad_r)
{
    unsigned ow = (unsigned)rocket_pool_ow(d);
    unsigned allow = rocket_pool_int8_rk3576_max_ow(d);
    unsigned n, i, x = 0, w;
    if (!allow) return 0;
    n = (ow + allow - 1u) / allow;
    if (n == 0u) n = 1u;
    if (s >= n) return n;
    /* Evenly sized slices, the remainder spread over the leading ones, so no slice is a
     * single column and every one is inside the allowance. */
    for (i = 0; i < s; i++) x += ow / n + (i < ow % n ? 1u : 0u);
    w = ow / n + (s < ow % n ? 1u : 0u);
    {
        long lo = (long)x * d->stride_x - d->pad_left;
        long hi = ((long)x + (long)w - 1) * d->stride_x + d->kw - d->pad_left;
        int pl = 0, pr = 0;
        if (lo < 0) { pl = (int)(-lo); lo = 0; }
        if (hi > d->iw) { pr = (int)(hi - d->iw); hi = d->iw; }
        *x0 = x; *ow_s = w;
        *in_col = (unsigned)lo;
        *iw_s = (unsigned)(hi - lo);
        *pad_l = pl; *pad_r = pr;
    }
    return n;
}

unsigned rocket_pool_int8_rk3576_ow_slices(const rocket_pool_desc *d)
{
    unsigned x0, ows, col, iws, n, s;
    int pl, pr;
    if (!d) return 0u;
    n = r76p_slice_plan(d, 0, &x0, &ows, &col, &iws, &pl, &pr);
    if (!n || n > ROCKET_RK3576_POOL_MAX_SLICES) return 0u;
    /* Every slice's own descriptor must derive back to the width it was cut to; if one
     * does not, the shape is refused rather than run on a geometry nobody checked. */
    for (s = 0; s < n; s++) {
        rocket_pool_desc sd;
        r76p_slice_plan(d, s, &x0, &ows, &col, &iws, &pl, &pr);
        sd = *d;
        sd.iw = (int)iws; sd.pad_left = pl; sd.pad_right = pr;
        if ((unsigned)rocket_pool_ow(&sd) != ows) return 0u;
    }
    return n;
}

/*
 * Whether the programmed reciprocal can reproduce the exactly-rounded average for this
 * window, over EVERY int8 input. The PPU multiplies the window sum by 0x10000/kw and
 * 0x10000/kh, both truncated, so the computed quotient is low by
 *
 *     |sum| * (1/(kh*kw) - (floor(2^16/kw)/2^16)*(floor(2^16/kh)/2^16))
 *
 * and the largest |sum| an int8 window can reach is 128*kh*kw. The rounded average moves
 * when that error can cross a half-count boundary; the closest a quotient of an integer
 * by n can come to a half without being one is 1/(2n), so the test is against that.
 * Exact for max, which divides nothing.
 */
static int r76p_recip_exact(unsigned kw, unsigned kh)
{
    unsigned n = kh * kw;
    double rw = (double)(0x10000u / kw) / 65536.0;
    double rh = (double)(0x10000u / kh) / 65536.0;
    double err = (1.0 / (double)n - rw * rh) * (128.0 * (double)n);
    /* AN EVEN WINDOW HAS TIES, and a truncated reciprocal breaks every one of them
     * toward zero rather than to even — a full count on any sum that lands on a half,
     * which the error bound below cannot see because it asks how close a NON-tie can
     * come to one. So an even window needs each axis to divide 65536 exactly. */
    if ((n & 1u) == 0u && (0x10000u % kw || 0x10000u % kh)) return 0;
    if (err < 0.0) err = -err;
    return err < 1.0 / (2.0 * (double)n);
}

int rocket_pool_int8_rk3576_exact(const rocket_pool_desc *d)
{
    if (rocket_pool_int8_rk3576_plan(d) != ROCKET_OK) return 0;
    if (d->method == POOL_METHOD_MAX) return 1;
    /* With the pad excluded the divisor is a per-POSITION quantity — a border window
     * divides by the taps that fell inside the plane — so the predicate has to hold for
     * every extent a window can present, not just the full one. Stated over the pair
     * because the PPU's reciprocal is per axis and a border's valid count is separable. */
    if (d->avg_exclude_pad) {
        unsigned a, b;
        for (b = 1; b <= (unsigned)d->kh; b++)
            for (a = 1; a <= (unsigned)d->kw; a++)
                if (!r76p_recip_exact(a, b)) return 0;
        return 1;
    }
    return r76p_recip_exact((unsigned)d->kw, (unsigned)d->kh);
}

/* Build the register program into `ops`, returning its word count in `*nops`. Split out of
 * r76p_gen so that a cross-layer chain, which lays every program of a run out contiguously
 * in one BO of its own, gets the same words from the same code rather than a second
 * transcription of the geometry. */
static int r76p_build(const struct rocket_pool_int8_rk3576_handle *h, const char *entry,
                      uint64_t *ops, uint32_t *nops)
{
    pool_params_rk3576_t p;

    memset(&p, 0, sizeof p);
    memset(ops, 0, RK3576_POOL_TASK_OPS * sizeof *ops);
    p.iw = (uint16_t)h->iw; p.ih = (uint16_t)h->ih; p.c = (uint16_t)h->c;
    p.ow = (uint16_t)h->ow; p.oh = (uint16_t)h->oh;
    p.kw = (uint8_t)h->d.kw; p.kh = (uint8_t)h->d.kh;
    p.stride_x = (uint8_t)h->d.stride_x; p.stride_y = (uint8_t)h->d.stride_y;
    p.pad_left   = (uint8_t)h->d.pad_left;   p.pad_right  = (uint8_t)h->d.pad_right;
    p.pad_top    = (uint8_t)h->d.pad_top;    p.pad_bottom = (uint8_t)h->d.pad_bottom;
    p.mode = (uint8_t)(h->d.method == POOL_METHOD_MAX ? ROCKET_RK3576_POOL_MAX
                       : h->d.avg_exclude_pad         ? ROCKET_RK3576_POOL_AVG_NOPAD
                                                      : ROCKET_RK3576_POOL_AVG);
    /* THE MODE BIT MOVES THE DIVISOR AND NOT THE SUM. AVG_NOPAD divides a border window
     * by the taps that landed inside the plane, but the taps that did not are still
     * ADDED, at the pad value — so at a zero point of 0 it is exactly TFLite's
     * AVERAGE_POOL_2D and at any other zero point it is a third function that is neither
     * that nor count-include-pad. Measured as EXACTLY the border outputs and no others:
     * 4096 of 18496 at 17x17 c=64 (64 border positions x 64 channels) and 3584 of 8192 at
     * 8x8 c=128, wrong at zero points -128 and +40 and exact at 0, same geometry.
     * [HW sweep, H96 MAX M9, tests/rk3576_pool_probe.c lib]
     *
     * So the pad value is programmed as ZERO here rather than as the input zero point,
     * which makes the sum the valid taps' own and the quotient their average — which is
     * what TFLite computes, since its quantized average pool averages the STORED values
     * over the valid count and requires the input and output quantization to match. */
    p.input_zero_point = h->d.avg_exclude_pad ? 0 : h->in_zp;
    p.input_dma  = h->cube_in
                     ? h->src.dma_address + h->src_off + (size_t)h->src_col * C2
                     : h->in.dma_address;
    p.output_dma = (uint32_t)((h->out_ext.ptr ? h->out_ext.dma_address + h->out_off
                                             : h->out.dma_address));
    p.src_surf_elems = h->in_surf;
    p.src_line_elems = h->src_pitch;
    p.dst_surf_elems = h->out_surf;
    p.tasks = ops;

    if (gen_pool_rk3576(&p) != 0) {
        ROCKET_LOGE("%s: the generator refused (%ux%u c%u k%ux%u s%ux%u -> %ux%u)\n",
                    entry, h->iw, h->ih, h->c, h->d.kw, h->d.kh,
                    h->d.stride_x, h->d.stride_y, h->ow, h->oh);
        return ROCKET_E_UNSUPPORTED;
    }
    *nops = p.task_count;
    return ROCKET_OK;
}

/* (Re)generate the register program into the handle's regcmd BO. Called at pack time and
 * again whenever the input BO changes, which is the only thing in the program that a
 * caller can move after packing. */
static int r76p_gen(struct rocket_pool_int8_rk3576_handle *h, const char *entry)
{
    uint64_t ops[RK3576_POOL_TASK_OPS];
    uint32_t nops = 0;
    int rc = r76p_build(h, entry, ops, &nops);

    if (rc != ROCKET_OK) return rc;
    h->task_count = nops;
    rocket_bo_prep(h->fd, &h->rc, 1, 0);
    memcpy(h->rc.ptr, ops, (size_t)nops * sizeof(uint64_t));
    rocket_bo_fini(h->fd, &h->rc);
    return ROCKET_OK;
}

/* THE CHAIN'S VIEW OF THIS HANDLE. See rocket_rk3576_internal.h for why a pool may sit in
 * a convolution stream and why only as an INTERIOR node. Everything here is read off the
 * frozen handle; nothing on it is written, and the program goes into the caller's array
 * rather than the handle's own regcmd BO, whose trailer the chain would otherwise rewrite
 * under the per-layer entry that shares it. */
int rocket_rk3576_pool_link(struct rocket_pool_int8_rk3576_handle *h,
                            struct rocket_rk3576_pool_link *out, uint64_t *ops)
{
    static const char entry[] = "rocket_rk3576_pool_link";
    uint32_t nops = 0;
    int rc;

    /* A COLUMN-SPLIT HANDLE HAS ONE SURFACE PER SLICE, so there is no single cube for a
     * producer to write into and nothing a chained stream could carry. That is the OUTPUT
     * side alone — the input side is a pitched window per slice and is taken. Refused here
     * rather than at the submit, where a caller would already have wired it. */
    if (h && h->nslice) {
        ROCKET_LOGE("%s: this shape is split into %u column slices, each writing its own "
                    "surface, so it cannot be the PRODUCER end of a cube join nor a chain "
                    "node; its input side takes a cube\n", entry, h->nslice);
        return ROCKET_E_UNSUPPORTED;
    }
    if (!h || !out || !ops) return ROCKET_E_SHAPE;
    /* Somewhere to write, which is its OWN surface or a caller's buffer it writes a slice
     * of — a placed handle releases `out`, so asking for that one alone refuses every
     * pool whose output was placed. */
    if (!h->rc.ptr || (!h->out.ptr && !h->out_ext.ptr)) {
        ROCKET_LOGE("%s: this handle was never packed\n", entry);
        return ROCKET_E_SHAPE;
    }
    rc = r76p_build(h, entry, ops, &nops);
    if (rc != ROCKET_OK) return rc;

    memset(out, 0, sizeof *out);
    out->fd         = h->fd;
    out->feat_dma   = h->cube_in ? h->src.dma_address + h->src_off : h->in.dma_address;
    out->surf_dma   = h->out_ext.ptr ? h->out_ext.dma_address + h->out_off
                                     : h->out.dma_address;
    out->cube_in    = h->cube_in;
    out->cube_out   = h->cube_out;
    out->surf       = h->out_ext.ptr ? &h->out_ext : &h->out;
    out->surf_off   = h->out_ext.ptr ? h->out_off : 0;
    out->groups     = h->creg / C2;
    out->surf_elems = h->out_surf;
    /* The PPU writes ow*oh elements per group into a stride that is round4(ow*oh) by
     * default and the buffer's own where the surface is PLACED, so the write guard must
     * ask about the live prefix and not the padding — which the PPU never touches and
     * which would therefore read as "this task wrote nothing". */
    out->live_elems = h->ow * h->oh;
    out->nops       = nops;
    return ROCKET_OK;
}

rocket_pool_int8_rk3576_handle *
rocket_pool_int8_pack_rk3576(int fd, const rocket_pool_desc *d, int in_zp)
{
    static const char entry[] = "rocket_pool_int8_pack_rk3576";
    struct rocket_pool_int8_rk3576_handle *h;

    if (!d || fd < 0) return NULL;
    if (!r76p_is_this_chip(entry)) return NULL;
    if (rocket_pool_int8_rk3576_plan(d) != ROCKET_OK) return NULL;
    if (in_zp < -128 || in_zp > 127) return NULL;

    /* A PLANE TOO WIDE FOR ONE TASK IS N HANDLES, and this is where that is decided —
     * once, at pack time, so an inference is the same loop whatever the shape. The parent
     * holds the caller's geometry and no BOs; every slice below is an ordinary handle
     * with a column window at each end. */
    if (!r76p_ow_probe() &&
        (unsigned)rocket_pool_ow(d) > rocket_pool_int8_rk3576_max_ow(d)) {
        unsigned n = rocket_pool_int8_rk3576_ow_slices(d), s;
        struct rocket_pool_int8_rk3576_handle *par;
        if (!n) return NULL;
        par = (struct rocket_pool_int8_rk3576_handle *)calloc(1, sizeof *par);
        if (!par) return NULL;
        par->fd = fd; par->d = *d; par->in_zp = in_zp;
        par->c = (unsigned)d->c; par->ih = (unsigned)d->ih; par->iw = (unsigned)d->iw;
        par->oh = (unsigned)rocket_pool_oh(d);
        par->ow = (unsigned)rocket_pool_ow(d);
        par->nslice = n;
        for (s = 0; s < n; s++) {
            rocket_pool_desc sd = *d;
            unsigned x0, ows, col, iws;
            int pl, pr;
            r76p_slice_plan(d, s, &x0, &ows, &col, &iws, &pl, &pr);
            sd.iw = (int)iws; sd.pad_left = pl; sd.pad_right = pr;
            par->slice[s] = rocket_pool_int8_pack_rk3576(fd, &sd, in_zp);
            if (!par->slice[s]) {
                rocket_pool_int8_free_rk3576(fd, par);
                return NULL;
            }
            par->slice[s]->rm_in_col   = col;
            par->slice[s]->rm_in_pitch = (unsigned)d->iw;
            par->slice[s]->rm_out_col  = x0;
            par->slice[s]->rm_out_pitch = par->ow;
        }
        return par;
    }

    h = (struct rocket_pool_int8_rk3576_handle *)calloc(1, sizeof *h);
    if (!h) return NULL;
    h->fd = fd;
    h->d = *d;
    h->in_zp = in_zp;
    h->c  = (unsigned)d->c; h->ih = (unsigned)d->ih; h->iw = (unsigned)d->iw;
    h->oh = (unsigned)rocket_pool_oh(d);
    h->ow = (unsigned)rocket_pool_ow(d);
    h->creg = ((h->c + C2 - 1u) / C2) * C2;
    h->in_surf  = r76p_pack_src() ? h->iw * h->ih : r76p_round4(h->iw * h->ih);
    h->out_surf = r76p_round4(h->ow * h->oh);
    h->in_bytes  = (size_t)(h->creg / C2) * h->in_surf * C2;
    h->out_bytes = (size_t)(h->creg / C2) * h->out_surf * C2;

    if (rocket_bo_alloc32(fd, h->in_bytes, &h->in) < 0 ||
        rocket_bo_alloc32(fd, h->out_bytes, &h->out) < 0 ||
        rocket_bo_alloc32(fd, RK3576_POOL_TASK_OPS * sizeof(uint64_t), &h->rc) < 0) {
        ROCKET_LOGE("%s: BO allocation failed\n", entry);
        rocket_pool_int8_free_rk3576(fd, h);
        return NULL;
    }
    /* The whole cube is zeroed once here rather than per call. Pooling reduces WITHIN a
     * channel, so what the padding channels of the last group hold never reaches a live
     * channel's output and is never de-scattered — but the PPU reads them, so they have
     * to hold something defined. */
    rocket_bo_prep(fd, &h->in, 1, 0);
    memset(h->in.ptr, 0, h->in_bytes);
    rocket_bo_fini(fd, &h->in);

    if (r76p_gen(h, entry) != ROCKET_OK) {
        rocket_pool_int8_free_rk3576(fd, h);
        return NULL;
    }
    return h;
}

int rocket_pool_int8_cube_in_rk3576(rocket_pool_int8_rk3576_handle *h,
                                    const rocket_rk3576_cube *src)
{
    static const char entry[] = "rocket_pool_int8_cube_in_rk3576";

    if (!h) return ROCKET_E_SHAPE;

    /* A COLUMN-SPLIT HANDLE TAKES A CUBE IN, and "one surface per slice" is a property of
     * the OUTPUT. Each slice already reads a column window of the caller's tensor — the
     * host scatter walks `rm_in_col` of `rm_in_pitch`-wide rows — and the PPU carries the
     * same two quantities as registers: the DDR line stride (`0x7024`) separately from what
     * the windows consume (`0x600C`), honoured at a pitch above the extent and at a base
     * part way into a row [HW sweep, 16 + 12 cells, tests/rk3576_row_pitch.c]. So a slice's
     * window of a producer's surface is the ordinary pitched cube every other consumer
     * takes, and the split costs the input side nothing. The OUTPUT stays refused: the
     * destination surface stride is derived from the plane and no register moves it, so
     * the slices cannot write one plane between them. */
    if (h->nslice) {
        unsigned s;
        int rc;
        if (!src) {
            for (s = 0; s < h->nslice; s++)
                if ((rc = rocket_pool_int8_cube_in_rk3576(h->slice[s], NULL)) != ROCKET_OK)
                    return rc;
            h->cube_in = 0;
            return ROCKET_OK;
        }
        /* Checked against the CALLER'S tensor first, so a mismatch names the geometry the
         * caller passed rather than a slice's window it never chose. */
        if (src->c != h->c || src->h != h->ih || src->w != h->iw) {
            ROCKET_LOGE("%s: the cube is %ux%ux%u and this handle reads %ux%ux%u\n",
                        entry, src->c, src->h, src->w, h->c, h->ih, h->iw);
            return ROCKET_E_SHAPE;
        }
        for (s = 0; s < h->nslice; s++) {
            rocket_rk3576_cube sv = *src;
            /* The slice's window, expressed the way any pitched consumer expresses one:
             * its own width, at its own column of the caller's rows. A cube that already
             * declares a pitch keeps it — a producer whose surface is wider than its
             * tensor composes with the split rather than excluding it. */
            sv.w       = h->slice[s]->iw;
            sv.pitch_w = src->pitch_w ? src->pitch_w : src->w;
            sv.col_off = src->col_off + h->slice[s]->rm_in_col;
            if ((rc = rocket_pool_int8_cube_in_rk3576(h->slice[s], &sv)) != ROCKET_OK) {
                unsigned u;
                for (u = 0; u < s; u++) rocket_pool_int8_cube_in_rk3576(h->slice[u], NULL);
                return rc;
            }
        }
        h->cube_in = 1;
        return ROCKET_OK;
    }
    if (!src) {
        if (!h->cube_in) return ROCKET_OK;
        if (rocket_bo_alloc32(h->fd, h->in_bytes, &h->in) < 0) return ROCKET_E_NOMEM;
        rocket_bo_prep(h->fd, &h->in, 1, 0);
        memset(h->in.ptr, 0, h->in_bytes);
        rocket_bo_fini(h->fd, &h->in);
        h->cube_in = 0;
        h->src_off = 0;
        h->src_pitch = 0;
        h->src_col = 0;
        memset(&h->src, 0, sizeof h->src);
        h->in_surf = r76p_pack_src() ? h->iw * h->ih : r76p_round4(h->iw * h->ih);
        return r76p_gen(h, entry);
    }
    if (src->fd != h->fd) {
        ROCKET_LOGE("%s: the cube belongs to fd %d and this handle to fd %d; an IOVA is "
                    "per-fd\n", entry, src->fd, h->fd);
        return ROCKET_E_SHAPE;
    }
    if (src->c != h->c || src->h != h->ih || src->w != h->iw) {
        ROCKET_LOGE("%s: the cube is %ux%ux%u and this handle reads %ux%ux%u\n",
                    entry, src->c, src->h, src->w, h->c, h->ih, h->iw);
        return ROCKET_E_SHAPE;
    }
    /* The channel count has to round the same way at both ends. Below a multiple of 16
     * the handle's own cube carries padding channels it fills itself and a producer does
     * not, and the PPU reads the whole programmed group — so the result of a live channel
     * is unaffected but the padding is a producer's business, and there is no producer
     * this library builds whose surface would not simply be a multiple of 16. */
    if (h->c % C2) {
        ROCKET_LOGE("%s: %u channels is not a multiple of %u, so this handle's own cube "
                    "carries padding channels a producer does not write\n",
                    entry, h->c, C2);
        return ROCKET_E_UNSUPPORTED;
    }
    if (src->groups * C2 < h->creg ||
        src->bo.size < src->off + (size_t)src->groups * src->surf_elems * C2) {
        ROCKET_LOGE("%s: the cube carries %u channel group(s) of %zu bytes and this "
                    "handle's PPU_RDMA walks %u channels\n",
                    entry, src->groups, src->surf_elems * C2, h->creg);
        return ROCKET_E_SHAPE;
    }
    if (src->surf_elems > 0xFFFFFFFFu / C2) return ROCKET_E_SHAPE;

    /* A cube may be a SLICE of a bigger buffer, and the base is a plain address the PPU
     * honours like every other. */
    if (src->off % ((size_t)src->surf_elems * C2)) {
        ROCKET_LOGE("%s: the cube starts at byte %zu, which is not a whole channel group "
                    "of %zu bytes into its buffer\n",
                    entry, src->off, (size_t)src->surf_elems * C2);
        return ROCKET_E_SHAPE;
    }
    /* A ROW PITCH IS A REGISTER HERE. The PPU carries what the windows consume
     * (`0x600C`) and the DDR line stride (`0x7024`) separately and honours a pitch above
     * the extent — 16 cells over four geometries at gaps of 1/4/16/64, each against a
     * control that leaves the pitch derived and differs
     * [HW sweep, H96 MAX M9, tests/rk3576_row_pitch.c]. So a producer whose surface is
     * WIDER than its tensor — the packed-image first conv, which materialises its pad
     * columns — is read in place instead of being de-scattered and scattered back. */
    if (src->pitch_w && src->pitch_w < src->col_off + src->w) {
        ROCKET_LOGE("%s: the cube declares a %u-wide plane at column %u of %u-element "
                    "rows\n", entry, src->w, src->col_off, src->pitch_w);
        return ROCKET_E_SHAPE;
    }
    if (src->col_off && !src->pitch_w) {
        ROCKET_LOGE("%s: the cube starts at column %u of rows whose pitch it does not "
                    "declare\n", entry, src->col_off);
        return ROCKET_E_SHAPE;
    }
    h->src = src->bo;
    h->src_off = src->off;
    h->cube_in = 1;
    h->src_pitch = src->pitch_w && src->pitch_w != src->w ? src->pitch_w : 0u;
    /* A COLUMN OFFSET IS AN ADDRESS, and it is added at the PROGRAM rather than folded
     * into `src_off`. The chain's run rule asserts a link by comparing a consumer's
     * feature base against a producer's surface base for exact equality — deliberately,
     * since with slices an address is a base plus an offset and a consumer reading the
     * whole of a concatenation buffer is not linked to the producer of its second half.
     * Folding the column in would move that address off the producer's by one atom and
     * break every run this join is meant to create. */
    h->src_col = src->col_off;
    /* The producer's surface stride is what PPU_RDMA has to walk, and a direct
     * convolution's is `ow*oh` exactly rather than the round4 this handle would allocate
     * for itself. The register takes it verbatim. */
    h->in_surf = (unsigned)src->surf_elems;
    if (h->in.ptr) rocket_bo_free(h->fd, &h->in);
    memset(&h->in, 0, sizeof h->in);
    return r76p_gen(h, entry);
}

int rocket_pool_int8_cube_out_rk3576(rocket_pool_int8_rk3576_handle *h, int on)
{
    static const char entry[] = "rocket_pool_int8_cube_out_rk3576";

    /* A COLUMN-SPLIT HANDLE HAS ONE SURFACE PER SLICE, so there is no single cube for a
     * producer to write into and nothing a chained stream could carry. That is the OUTPUT
     * side alone — the input side is a pitched window per slice and is taken. Refused here
     * rather than at the submit, where a caller would already have wired it. */
    if (h && h->nslice) {
        ROCKET_LOGE("%s: this shape is split into %u column slices, each writing its own "
                    "surface, so it cannot be the PRODUCER end of a cube join nor a chain "
                    "node; its input side takes a cube\n", entry, h->nslice);
        return ROCKET_E_UNSUPPORTED;
    }
    if (!h) return ROCKET_E_SHAPE;
    if (!on) { h->cube_out = 0; return ROCKET_OK; }
    /* The same rounding rule the input side has, for the same reason: pooling reduces
     * WITHIN a channel, so a partial group's channels carry whatever the input cube's
     * padding held and this handle cannot say what a consumer would read there. */
    if (h->c % C2) {
        ROCKET_LOGE("%s: %u channels is not a multiple of %u, so the last group's tail is "
                    "the input cube's padding and not a value this handle can declare\n",
                    entry, h->c, C2);
        return ROCKET_E_UNSUPPORTED;
    }
    h->cube_out = 1;
    return ROCKET_OK;
}

int rocket_pool_int8_cube_out_at_rk3576(rocket_pool_int8_rk3576_handle *h,
                                        const struct rocket_rk3576_cube *dst)
{
    static const char entry[] = "rocket_pool_int8_cube_out_at_rk3576";
    unsigned need;
    size_t bytes;

    /* A COLUMN-SPLIT HANDLE HAS ONE SURFACE PER SLICE, so there is no single cube for a
     * producer to write into and nothing a chained stream could carry. That is the OUTPUT
     * side alone — the input side is a pitched window per slice and is taken. Refused here
     * rather than at the submit, where a caller would already have wired it. */
    if (h && h->nslice) {
        ROCKET_LOGE("%s: this shape is split into %u column slices, each writing its own "
                    "surface, so it cannot be the PRODUCER end of a cube join nor a chain "
                    "node; its input side takes a cube\n", entry, h->nslice);
        return ROCKET_E_UNSUPPORTED;
    }
    if (!h) return ROCKET_E_SHAPE;
    if (!dst) {
        if (!h->out_ext.ptr) { h->cube_out = 0; return ROCKET_OK; }
        /* Back onto a surface of its own, which is the stride the emitter derives. */
        h->out_surf  = r76p_round4(h->ow * h->oh);
        h->out_bytes = (size_t)(h->creg / C2) * h->out_surf * C2;
        if (rocket_bo_alloc32(h->fd, h->out_bytes, &h->out) < 0) return ROCKET_E_NOMEM;
        memset(&h->out_ext, 0, sizeof h->out_ext);
        h->out_off = 0;
        h->cube_out = 0;
        return r76p_gen(h, entry);
    }
    if (rocket_pool_int8_cube_out_rk3576(h, 1) != ROCKET_OK) return ROCKET_E_UNSUPPORTED;
    if (dst->fd != h->fd) {
        ROCKET_LOGE("%s: the buffer belongs to fd %d and this handle to fd %d; an IOVA is "
                    "per-fd\n", entry, dst->fd, h->fd);
        return ROCKET_E_SHAPE;
    }
    if (dst->h != h->oh || dst->w != h->ow) {
        ROCKET_LOGE("%s: the slice is %ux%u and this handle writes %ux%u\n",
                    entry, dst->w, dst->h, h->ow, h->oh);
        return ROCKET_E_SHAPE;
    }
    /* A WRITER IGNORES A PITCH IT WAS NEVER TOLD. Both destination strides here are
     * derived from the output plane, so a slice whose rows sit further apart than its width
     * — or whose plane starts part way into a row — would be written contiguously and read
     * back skewed. Nothing this library allocates carries either (`rocket_rk3576_cube_alloc()`
     * zeroes them and a slice inherits), so this is a refusal rather than a case. */
    if ((dst->pitch_w && dst->pitch_w != dst->w) || dst->col_off) {
        ROCKET_LOGE("%s: the slice is a %u-wide plane at column %u of %u-element rows, and "
                    "this handle writes a contiguous plane\n",
                    entry, dst->w, dst->col_off, dst->pitch_w ? dst->pitch_w : dst->w);
        return ROCKET_E_UNSUPPORTED;
    }
    /* THE DESTINATION SURFACE STRIDE IS A REGISTER (`0x607C`) AND IT TAKES THE PLANE.
     * The emitter's default is round4(ow*oh) — the shape every vendor pooling program
     * carries and what this handle allocates for itself — but a slice of somebody else's
     * buffer has that buffer's stride, and a shared concatenation's is the plane exactly,
     * because its other operands are convolutions and a direct convolution writes `ow*oh`.
     * The part honours any value at or above the plane, including ones that are neither a
     * multiple of four atoms nor of a 64-byte line, and odd ones: at a 5x3 plane, 240 and
     * 272 bytes both land every atom where 224 — one atom under the plane — loses one
     * [HW sweep, H96 MAX M9, tests/rk3576_pool_probe.c dst]. Below the plane the channel
     * groups overlap, so that is the one refusal left.
     *
     * There is no destination LINE stride to go with it, which is what the pitch/column
     * clause above is about: a slice may be a plane inside a DEEPER buffer, never a plane
     * inside wider rows. */
    if (dst->surf_elems < (size_t)h->ow * h->oh) {
        ROCKET_LOGE("%s: the slice's channel-group stride is %zu elements and this handle "
                    "writes a %ux%u plane; below the plane the groups overlap\n",
                    entry, dst->surf_elems, h->ow, h->oh);
        return ROCKET_E_UNSUPPORTED;
    }
    if (dst->surf_elems > 0xFFFFFFFFu / C2) return ROCKET_E_SHAPE;
    need = h->creg / C2;
    bytes = (size_t)need * dst->surf_elems * C2;
    if (dst->groups < need || dst->bo.size < dst->off + bytes) {
        ROCKET_LOGE("%s: this handle writes %u channel group(s) (%zu bytes) and the slice "
                    "carries %u at byte %zu of a %zu-byte buffer\n",
                    entry, need, bytes, dst->groups, dst->off, dst->bo.size);
        return ROCKET_E_SHAPE;
    }
    if (dst->off % ((size_t)dst->surf_elems * C2)) {
        ROCKET_LOGE("%s: the slice starts at byte %zu, which is not a whole channel group "
                    "of %zu bytes into its buffer\n",
                    entry, dst->off, (size_t)dst->surf_elems * C2);
        return ROCKET_E_SHAPE;
    }
    /* The program, the sentinel stamp and the de-scatter all walk the stride the surface
     * really has, so it is the handle's from here rather than the derived one. */
    h->out_surf  = (unsigned)dst->surf_elems;
    h->out_bytes = bytes;
    h->out_ext = dst->bo;
    h->out_off = dst->off;
    /* The handle's own surface is dead weight now — the point of the shared buffer is to
     * pay for one allocation and not two. */
    if (h->out.ptr) rocket_bo_free(h->fd, &h->out);
    memset(&h->out, 0, sizeof h->out);
    return r76p_gen(h, entry);
}

int rocket_pool_int8_cube_of_rk3576(const rocket_pool_int8_rk3576_handle *h,
                                    struct rocket_rk3576_cube *out)
{
    static const char entry[] = "rocket_pool_int8_cube_of_rk3576";

    if (h && h->nslice) {
        ROCKET_LOGE("%s: a column-split pool has one surface per slice, so it has no cube "
                    "to hand a consumer\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    if (!h || !out) return ROCKET_E_SHAPE;
    if (h->c % C2) {
        ROCKET_LOGE("%s: %u channels is not a multiple of %u\n", entry, h->c, C2);
        return ROCKET_E_UNSUPPORTED;
    }
    memset(out, 0, sizeof *out);
    out->fd = h->fd;
    out->c = h->c;
    out->h = h->oh;
    out->w = h->ow;
    out->groups = h->creg / C2;
    /* A surface of the handle's own is round4(ow*oh) per channel group where a direct
     * convolution writes the plane exactly; a PLACED one carries the stride of the buffer
     * it was given. Either way that is the consumer's DDR channel-group jump, which is a
     * register and is honoured at any value at or above the plane. */
    out->surf_elems = h->out_surf;
    if (h->out_ext.ptr) { out->bo = h->out_ext; out->off = h->out_off; }
    else                { out->bo = h->out;     out->off = 0; }
    return ROCKET_OK;
}

/* ===========================================================================
 * THE DIVISOR LAG, and the check that costs three columns.
 *
 * A padded average pool on this part intermittently divides an output window by the tap
 * count belonging to the PREVIOUS window in RASTER order. The sum is right; only the
 * divisor is one output position late, and once it starts it persists over a long run of
 * channel groups. It is measured, not inferred: every wrong element of every occurrence
 * — 18368 of 18368 on one, and the same on the graph's — is reproduced exactly by that
 * model and by no other, including the two one-axis lags that fit 89% of it.
 *
 * IT IS ONLY OBSERVABLE WHERE THE PAD IS EXCLUDED, because that is the only mode whose
 * divisor varies with position: at a constant divisor the previous window's is this
 * window's. So it has been under every max pool and every count-include-pad average ever
 * run here and could not have shown, and a whole-surface write check cannot see it — the
 * surface is written, in full, and is wrong.
 *
 * What it costs to catch: the positions where the model and the part CAN disagree are
 * exactly those whose raster predecessor has a different tap count, which for a SAME
 * window is two columns of every row plus a cell or two at a row where the vertical count
 * changes. Recomputing those on the host is O(rows) an image rather than O(rows*columns),
 * and any occurrence lights up thousands of them at once. A failing check redoes the
 * submit — the same loop the write check uses, without the power cycle, since this is not
 * the poisoning and a domain collapse is not what clears it.
 *
 * A CUBE JOIN IS COVERED TOO, and the input it reads is the producer's SURFACE rather
 * than a row-major tensor — the same values in the layout the PPU itself fetched them
 * from, so the check is the same arithmetic over a different accessor. What it costs is
 * one cache-maintenance bracket on that surface, which the cube path otherwise never
 * pays: the whole point of a join is that the input side does no maintenance at all.
 * Only a pool this hazard can SHOW on pays it — a padded average, and nothing else.
 *
 * NOT COVERED: nothing, once a chain runs it too (rocket_pool_int8_rk3576_lag_check(),
 * called from the kick's verify bracket). ROCKET_RK3576_POOL_LAGCHECK=0 turns it off.
 * ==========================================================================*/
static int r76p_lagcheck_on(void)
{
    const char *e = getenv("ROCKET_RK3576_POOL_LAGCHECK");
    return (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 1;
}

/* HOW MANY REDOS THE LAG GETS, and it is NOT the poisoning's eight.
 *
 * The two hazards cost different things to retry and clear at different rates. A poisoning
 * redo cycles the NPU's power domain, which is why its budget is small; a lag redo is a
 * resubmit and nothing else. And the lag is COMMON where it is observable at all: on
 * Inception V3's nine padded average pools it repairs five to eleven calls per inference
 * set — roughly a third of them — so a run of failures is ordinary rather than a sign that
 * something else is wrong. Eight was measured to run out: a cross-layer kick redoes every
 * pool of the run at once, so two observable pools in one stream need both clean on the
 * same attempt, and six of eight attempts have been seen used [HW sweep, H96 MAX M9].
 * Sized from that rate rather than from a wish: at a per-call clean rate of ~0.4 for a
 * pair, thirty-two attempts leaves ~1e-7. Each one costs a submit, and only a shape the
 * hazard can show on ever spends one. */
static unsigned r76p_lag_attempts(void)
{
    const char *e = getenv("ROCKET_RK3576_POOL_LAG_ATTEMPTS");
    long v = (e && *e) ? strtol(e, NULL, 0) : 0;
    return (v >= 1 && v <= 4096) ? (unsigned)v : 32u;
}

/* SCORING THE LAST CHANNEL GROUP ONLY, which is the shipped check and is a COVERAGE TRADE.
 *
 * The check's cost is one pass over the source cube and the surface — memory, not
 * arithmetic — so the only large lever on it is scoring fewer channels, and whether that
 * is sound is the question "does an occurrence always reach the last group". Measured, by
 * recording every occurrence's reached SET rather than its element count: over **1109
 * occurrences** across six geometries of Inception V3 (channel counts 256..2048, planes
 * 8x8 to 35x19), every one is a CONTIGUOUS run of channel groups ENDING AT THE LAST ONE,
 * and no element matched neither divisor. The onset varies and is near-constant per shape
 * (group 7 of 48, 44 or 108 of 128, 1 or 5 of 18); the end never varies.
 *
 * So the last group alone detects every occurrence seen, at 1/ngroups of the cost — 6.0 ms
 * of an Inception V3 inference down to about 1, and its wall **38.2 -> 33.4 ms (12.6%)** on
 * the shipped `ondemand` governor, three interleaved repeats an arm at a spread of 0.4
 * (3.53 ms of it with the governor pinned, where the arms' differing host work cannot move
 * the clock), at a redo count indistinguishable from the full check's (0.9-1.9 per inference
 * against 1.2-1.7, three runs of 100 each way, the within-arm spread larger than the
 * difference). ZERO on the other four networks, whose pools are chain-run nodes so this
 * entry never runs [HW sweep, H96 MAX M9].
 *
 * What that is NOT is a proof: the suffix is an observation about how the part's divisor
 * state fails, not a register, and a lag that stopped short of the last group would be
 * missed and would leave a full, plausible, wrong surface. It ships because the evidence
 * bounds the class it would miss — 1109 of 1109 occurrences reach the last group — which
 * is the distinction from the write guard's cheaper forms, whose missed class the
 * instrument cannot induce at all. ROCKET_RK3576_POOL_LAG_TAIL=0 scores every group. */
static int r76p_lag_tail(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_POOL_LAG_TAIL");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 1;
    }
    return cached;
}

/* Whether to count how many scored elements could tell the two divisors apart — an RE
 * readout that doubles the rounding work, so it is off unless asked for. */
static int r76p_lag_discr(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_POOL_LAG_DISCR");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 0;
    }
    return cached;
}

/* Whether this descriptor has any position at all where the lag could show. A pool with
 * no padding, or a max pool, or a count-include-pad average, has none. */
static int r76p_lag_observable(const rocket_pool_desc *d)
{
    return d->method == POOL_METHOD_AVG && d->avg_exclude_pad &&
           (d->pad_left || d->pad_top || d->pad_right || d->pad_bottom);
}

/* The observable positions, with BOTH divisors each one has to be scored against: the
 * window's own and its raster predecessor's. Neither depends on the channel, so they are
 * computed once here rather than per element — the check runs over every channel, and on a
 * 768-channel plane that is the difference between a scan and a cost. Returns the count.
 * Cheap to build (one pass over the output plane) and reused over every channel. */
struct r76p_lag_pos { unsigned short y, x; unsigned char dw, dh, pw, ph; };

static unsigned r76p_lag_positions(const struct rocket_pool_int8_rk3576_handle *h,
                                   struct r76p_lag_pos *list, unsigned cap)
{
    const rocket_pool_desc *d = &h->d;
    unsigned y, x, n = 0;
    for (y = 0; y < h->oh; y++)
        for (x = 0; x < h->ow; x++) {
            int py = (int)y, px = (int)x - 1;
            int dw, dh, pw, ph;
            if (px < 0) { px = (int)h->ow - 1; py = (int)y - 1; }
            if (py < 0) { py = (int)y; px = (int)x; }
            dw = r76p_inplane((int)x, d->stride_x, d->kw, d->pad_left, (int)h->iw);
            dh = r76p_inplane((int)y, d->stride_y, d->kh, d->pad_top, (int)h->ih);
            pw = r76p_inplane(px, d->stride_x, d->kw, d->pad_left, (int)h->iw);
            ph = r76p_inplane(py, d->stride_y, d->kh, d->pad_top, (int)h->ih);
            if (dw == pw && dh == ph) continue;
            if (n >= cap) return n;
            list[n].y = (unsigned short)y; list[n].x = (unsigned short)x;
            list[n].dw = (unsigned char)dw; list[n].dh = (unsigned char)dh;
            list[n].pw = (unsigned char)pw; list[n].ph = (unsigned char)ph;
            n++;
        }
    return n;
}

/* ONE INPUT ELEMENT, IN WHICHEVER LAYOUT THIS HANDLE READS. `in` is the caller's
 * row-major tensor — a column slice reads it through its own window rather than at its
 * own width — and NULL on a cube join, where the producer's surface is read in place at
 * the same base, group stride, row pitch and column offset the register program was
 * given. Both forms are the SAME addressing the entry already uses (the scatter loop and
 * r76p_params respectively), which is what keeps the check a second reading of the
 * program's own input rather than a second model of it. */
/* `base` is channel `ci`'s first element; `row` and `step` walk (y, x) inside that
 * channel; `lane` reaches the next channel of the same sixteen-channel group. */
struct r76p_in_view { const int8_t *base; size_t row, lane; unsigned step; };

/* ONE CHANNEL'S VIEW, hoisted out of the position loop. Both layouts are affine in (y, x)
 * once the channel is fixed — a row-major channel is a plane at `pitch` per row and a cube
 * channel is one lane of every atom — so the per-tap work is two multiplies and no divide.
 * A per-tap `ci / C2` cost more than the arithmetic it addressed. */
static void r76p_in_view(const struct rocket_pool_int8_rk3576_handle *h, const int8_t *in,
                         unsigned ci, struct r76p_in_view *v)
{
    if (in) {
        const unsigned pitch = h->rm_in_pitch ? h->rm_in_pitch : h->iw;
        v->base = in + (size_t)ci * h->ih * pitch + h->rm_in_col;
        v->row  = pitch;
        v->step = 1u;
        v->lane = (size_t)h->ih * pitch;      /* the next channel is the next plane */
    } else {
        const unsigned pitch = h->src_pitch ? h->src_pitch : h->iw;
        v->base = (const int8_t *)h->src.ptr + h->src_off +
                  (size_t)(ci / C2) * h->in_surf * C2 +
                  (size_t)C2 * h->src_col + (ci % C2);
        v->row  = (size_t)C2 * pitch;
        v->step = C2;
        v->lane = 1u;                         /* the next channel is the next byte */
    }
}

/* ONE OUTPUT POSITION'S WINDOW SUM, FOR SIXTEEN CHANNELS AT ONCE.
 *
 * THIS IS THE CHECK'S WHOLE COST and it is host arithmetic, not the part: on Inception V3
 * the divisor check is 6.8 ms of a 43.1 ms inference — the largest host term in that graph,
 * and larger than every redo it saves put together — and all of it is here. It is the whole
 * of the pooling path's expense too: the WRITE guard beside it is 1.5 ms and its sentinel
 * scan is zero, the scan stopping at the first byte the program wrote [HW sweep,
 * H96 MAX M9].
 *
 * On a CUBE the sixteen channels of a group are the sixteen BYTES of an atom, so a tap is
 * one 16-byte load and the accumulation is two widening adds. Walked a lane at a time it is
 * sixteen scalar loads over the same line and sixteen dependent adds — the same shape as
 * the cube transpose that was 145 MB/s before it took a vector block. Row-major input keeps
 * the scalar loop: there the sixteen lanes are sixteen different PLANES and there is no
 * vector to load. Every pool that pays this on a real graph reads a cube.
 *
 * int16 accumulation is exact for any window a pooling program can carry — the worst case
 * is `taps * 128` and the field holds 32767 — so the guard is on the tap count and no
 * shape on this part comes near it. */
static void r76p_window_sums(const rocket_pool_desc *d,
                             const struct rocket_pool_int8_rk3576_handle *h,
                             const struct r76p_in_view *v,
                             const struct r76p_lag_pos *q, unsigned lanes, long *sum)
{
    int kh, kw;
    unsigned l;
#if R76P_HAVE_NEON
    if (v->lane == 1 && lanes == (unsigned)C2 && d->kh * d->kw <= 255) {
        int16x8_t s0 = vdupq_n_s16(0), s1 = vdupq_n_s16(0);
        int16_t acc[C2];
        for (kh = 0; kh < d->kh; kh++)
            for (kw = 0; kw < d->kw; kw++) {
                const int8_t *t;
                int8x16_t t8;
                int iy = (int)q->y * d->stride_y + kh - d->pad_top;
                int ix = (int)q->x * d->stride_x + kw - d->pad_left;
                if (iy < 0 || ix < 0 || iy >= (int)h->ih || ix >= (int)h->iw) continue;
                t = v->base + (size_t)iy * v->row + (size_t)ix * v->step;
                t8 = vld1q_s8(t);
                s0 = vaddw_s8(s0, vget_low_s8(t8));
                s1 = vaddw_s8(s1, vget_high_s8(t8));
            }
        vst1q_s16(acc, s0);
        vst1q_s16(acc + 8, s1);
        for (l = 0; l < (unsigned)C2; l++) sum[l] = acc[l];
        return;
    }
#endif
    for (l = 0; l < lanes; l++) sum[l] = 0;
    for (kh = 0; kh < d->kh; kh++)
        for (kw = 0; kw < d->kw; kw++) {
            const int8_t *t;
            int iy = (int)q->y * d->stride_y + kh - d->pad_top;
            int ix = (int)q->x * d->stride_x + kw - d->pad_left;
            if (iy < 0 || ix < 0 || iy >= (int)h->ih || ix >= (int)h->iw) continue;
            t = v->base + (size_t)iy * v->row + (size_t)ix * v->step;
            for (l = 0; l < lanes; l++) sum[l] += t[(size_t)l * v->lane];
        }
}

/* WHAT THE PART COMPUTED, NOT WHETHER IT AGREED. A wrong-element count cannot say which
 * function the surface came from, and that distinction is the whole of this hazard: the
 * lag was only ever decoded by scoring a failing surface against candidate DIVISORS. So
 * the check keeps the same score — how many observable elements match the right divisor,
 * how many match the PREVIOUS window's, and how many match neither — and the last bucket
 * is what says a failure is something else (or that this accessor is reading the wrong
 * bytes) rather than the lag. */
/* WHICH CHANNEL GROUPS AN OCCURRENCE REACHES is recorded too, because it is the only
 * thing that could make a CHEAPER check sound. The check's cost is one pass over the
 * source cube and the surface — memory, not arithmetic — so the only large lever on it is
 * scoring fewer channels, and whether that is a coverage trade or a free saving is exactly
 * the question "does an occurrence reach every group, or a run of them, or one". Recorded
 * on failure only, so a passing call pays nothing for it. */
struct r76p_lag_score {
    unsigned agree, prev, neither;
    unsigned groups, ngroups, g_lo, g_hi;
    /* HOW MANY OF THE SCORED ELEMENTS COULD HAVE TOLD THE TWO DIVISORS APART on this
     * call's data. A "rate" counted from occurrences is a rate of OBSERVATION, and a
     * layer whose border windows round the same either way would read as a layer that
     * never lags — so the discriminating count is what separates a placement difference
     * from a data one. It costs a second rounding per element, so it is behind
     * ROCKET_RK3576_POOL_LAG_DISCR=1 and is an RE readout, not a shipped cost. */
    unsigned discr;
};

/* 1 when the surface agrees with the arithmetic at every observable position. `sc` is
 * optional and is filled whatever the verdict. */
static int r76p_divisor_ok(const struct rocket_pool_int8_rk3576_handle *h,
                           const int8_t *in, const int8_t *surf,
                           const struct r76p_lag_pos *pos, unsigned npos,
                           struct r76p_lag_score *sc)
{
    const rocket_pool_desc *d = &h->d;
    unsigned ci, p;
    int ok = 1;
    /* One reciprocal per position for each of the two divisors it is scored against,
     * built once and reused over every channel group — the divisions are a property of
     * the output POSITION and the check's cost is per ELEMENT. */
    struct r76p_div *dv = (struct r76p_div *)malloc((size_t)npos * 2 * sizeof *dv);
    if (!dv) return 1;                        /* a check that cannot run is not a failure */
    for (p = 0; p < npos; p++) {
        r76p_div_init(&dv[2 * p],     pos[p].dw, pos[p].dh);
        r76p_div_init(&dv[2 * p + 1], pos[p].pw, pos[p].ph);
    }
    const int discr = r76p_lag_discr();
    if (sc) {
        sc->agree = sc->prev = sc->neither = sc->groups = sc->discr = 0;
        sc->ngroups = (h->c + (unsigned)C2 - 1u) / (unsigned)C2;
        sc->g_lo = sc->ngroups; sc->g_hi = 0;
    }
    /* SIXTEEN CHANNELS AT A TIME, because that is what one atom holds. Walked a channel at
     * a time the cube side reads one useful byte per sixteen and the surface side one per
     * sixteen too, so a 768-channel plane is sixteen strided passes over the same lines;
     * summed as a group it is one sequential pass. Measured worth ~20 ms of Inception V3's
     * wall on its own [HW sweep, H96 MAX M9]. Row-major input has the same shape with the
     * lanes a plane apart, which costs nothing extra — the sums are per channel either
     * way. */
    for (ci = r76p_lag_tail() ? ((h->c - 1u) / (unsigned)C2) * (unsigned)C2 : 0u;
         ci < h->c; ci += C2) {
        struct r76p_in_view v;
        unsigned lanes = h->c - ci < (unsigned)C2 ? h->c - ci : (unsigned)C2, l;
        const int8_t *srow = surf + (size_t)(ci / C2) * h->out_surf * C2;
        int gbad = 0;
        r76p_in_view(h, in, ci, &v);
        for (p = 0; p < npos; p++) {
            const struct r76p_lag_pos *q = &pos[p];
            long sum[C2];
            const int8_t *sq = srow + (size_t)C2 * ((size_t)q->y * h->ow + q->x);
            r76p_window_sums(d, h, &v, q, lanes, sum);
            for (l = 0; l < lanes; l++) {
                int got = sq[l];
                if (discr && sc &&
                    r76p_avg_round_div(sum[l], &dv[2 * p]) !=
                    r76p_avg_round_div(sum[l], &dv[2 * p + 1]))
                    sc->discr++;
                if (got == r76p_avg_round_div(sum[l], &dv[2 * p])) {
                    if (sc) sc->agree++;
                    continue;
                }
                ok = 0;
                gbad = 1;
                if (!sc) { free(dv); return 0; }
                /* The raster PREDECESSOR's window, which is the model this hazard IS. */
                if (got == r76p_avg_round_div(sum[l], &dv[2 * p + 1])) sc->prev++;
                else                                                   sc->neither++;
            }
        }
        if (gbad && sc) {
            unsigned g = ci / (unsigned)C2;
            sc->groups++;
            if (g < sc->g_lo) sc->g_lo = g;
            if (g > sc->g_hi) sc->g_hi = g;
        }
    }
    free(dv);
    return ok;
}

static void r76p_lag_report(const char *entry, const struct r76p_lag_score *sc)
{
    ROCKET_LOGD("%s: %u observable element(s) agree, %u are the PREVIOUS window's divisor "
                "(which is this hazard) and %u are neither (which is not); it reaches %u "
                "of %u channel group(s), %u..%u%s\n",
                entry, sc->agree, sc->prev, sc->neither, sc->groups, sc->ngroups,
                sc->g_lo, sc->g_hi,
                r76p_lag_tail() ? " (the LAST group only was scored)" : "");
}

/* THE SAME CHECK, FOR A CALLER THAT OWNS THE SUBMIT — a cross-layer kick, where the
 * pool's program is one of a stream and nothing runs between them. The kick's verify
 * bracket already has this layer's surface synced for the CPU, so `surf` comes from
 * there; the SOURCE cube is bracketed here, because a chain has no other reason to read
 * it and only an observable pool makes it worth a bracket at all.
 *
 * 1 when the surface is right or the hazard cannot show on this handle, 0 when the
 * divisor lagged — the caller then redoes its kick, with no power cycle, exactly as the
 * per-call path redoes its submit. */
int rocket_pool_int8_rk3576_lag_check(int fd, rocket_pool_int8_rk3576_handle *h,
                                      const void *surf)
{
    struct r76p_lag_pos *pos;
    unsigned npos, cap;
    int ok = 1;

    if (!h || !surf || !h->cube_in || !h->src.ptr) return 1;
    if (!r76p_lagcheck_on() || !r76p_lag_observable(&h->d)) return 1;
    cap = h->oh * h->ow;
    pos = (struct r76p_lag_pos *)malloc((size_t)cap * sizeof *pos);
    if (!pos) return 1;                       /* a check that cannot run is not a failure */
    npos = r76p_lag_positions(h, pos, cap);
    if (npos) {
        struct r76p_lag_score sc;
        rocket_bo_prep(fd, &h->src, 0, 2000000000ull);
        ok = r76p_divisor_ok(h, NULL, (const int8_t *)surf, pos, npos, &sc);
        rocket_bo_fini(fd, &h->src);
        h->lag_calls++;
        h->lag_discr += sc.discr;
        if (!ok) { h->lag_fires++; r76p_lag_report("rocket_pool_int8_rk3576_lag_check", &sc); }
    }
    free(pos);
    return ok;
}

/* Whether the divisor lag can show on this handle at all, so a caller that has to
 * arrange something for the check — a chain holding its producers' surfaces intact
 * across the verify bracket — only pays where there is a check to arrange. */
int rocket_pool_int8_rk3576_lag_observable(const rocket_pool_int8_rk3576_handle *h)
{
    return h && r76p_lagcheck_on() && r76p_lag_observable(&h->d);
}

/* THE SAME QUESTION WITHOUT THE CHECK'S KNOB IN IT. A caller that decides where to PLACE
 * this pool — a run finder — must read the geometry alone: ROCKET_RK3576_POOL_LAGCHECK=0 is
 * the arm that prices the check, and if it also moved the run structure that arm would
 * price two things and neither. */
int rocket_pool_int8_rk3576_lag_can_show(const rocket_pool_int8_rk3576_handle *h)
{
    return h && r76p_lag_observable(&h->d);
}

/* How many redos a caller that owns the submit should give the lag. See r76p_lag_attempts():
 * it is not the poisoning's budget, because a lag redo costs a submit and a poisoning redo
 * costs a power cycle, and the lag is common where it can show at all. */
unsigned rocket_pool_int8_rk3576_lag_attempts(void)
{
    return r76p_lag_attempts();
}

/* The BO a lag check reads as its input, so a caller can tell whether a surface it holds
 * is one. NULL when the check does not apply. */
uint32_t rocket_pool_int8_rk3576_lag_src_handle(const rocket_pool_int8_rk3576_handle *h)
{
    if (!rocket_pool_int8_rk3576_lag_observable(h) || !h->cube_in) return 0u;
    return h->src.handle;
}

/* How many bytes that bracket walks, so a caller profiling the check can price it against
 * the bracket cost model rather than guessing. Zero when there is no check to pay for. */
size_t rocket_pool_int8_rk3576_lag_src_bytes(const rocket_pool_int8_rk3576_handle *h)
{
    if (!rocket_pool_int8_rk3576_lag_observable(h) || !h->cube_in) return 0u;
    return h->src.size;
}

/* HOW OFTEN THIS HANDLE HAS SEEN THE HAZARD, and over how many checks — a caller printing
 * a per-layer table reads the RATE off this rather than off a per-geometry average. A
 * PARENT handle sums its column slices, each of which is its own submit and its own
 * occurrence. */
void rocket_pool_int8_rk3576_lag_counts(const rocket_pool_int8_rk3576_handle *h,
                                        unsigned *fires, unsigned *calls)
{
    unsigned f = 0, c = 0, s;
    if (h) {
        f = h->lag_fires; c = h->lag_calls;
        for (s = 0; s < h->nslice; s++)
            if (h->slice[s]) { f += h->slice[s]->lag_fires; c += h->slice[s]->lag_calls; }
    }
    if (fires) *fires = f;
    if (calls) *calls = c;
}

/* The discriminating-element total behind those counts (ROCKET_RK3576_POOL_LAG_DISCR=1;
 * zero otherwise). A layer with none could not have reported an occurrence whatever the
 * part did. */
unsigned long rocket_pool_int8_rk3576_lag_discr(const rocket_pool_int8_rk3576_handle *h)
{
    unsigned long v = 0; unsigned s;
    if (h) {
        v = h->lag_discr;
        for (s = 0; s < h->nslice; s++) if (h->slice[s]) v += h->slice[s]->lag_discr;
    }
    return v;
}

int rocket_pool_int8_prepacked_rk3576(int fd, rocket_pool_int8_rk3576_handle *h,
                                      const int8_t *in, int8_t *out)
{
    static const char entry[] = "rocket_pool_int8_prepacked_rk3576";
    uint32_t in_h[2], out_h[1];
    rocket_bo *osurf;
    size_t ooff;
    unsigned attempt, attempts, ci, y, x;
    unsigned char stamp;
    struct r76p_prof prof;
    double t0, tw;
    int rc, cycled = 0, confirmed = 0;
    struct r76p_lag_pos *lagpos = NULL;
    unsigned nlag = 0, lagged = 0;

    if (!h) return ROCKET_E_SHAPE;
    if (!out && !h->cube_out) return ROCKET_E_SHAPE;
    if (!in && !h->cube_in) return ROCKET_E_SHAPE;
    if (fd != h->fd) {
        ROCKET_LOGE("%s: this handle's buffers belong to fd %d, not %d — an IOVA is "
                    "per-fd, so a foreign fd would submit successfully against addresses "
                    "that mean nothing there\n", entry, h->fd, fd);
        return ROCKET_E_SHAPE;
    }

    memset(&prof, 0, sizeof prof);
    prof.on = r76p_prof_on();

    /* A SPLIT HANDLE IS THE SAME CALL, N TIMES. Each slice owns its window at both ends,
     * so there is no gather before and no stitch after — the loop is the whole of it. */
    if (h->nslice) {
        unsigned s;
        if (h->cube_out) {
            ROCKET_LOGE("%s: a column-split pool has one surface per slice, so it cannot be "
                        "a cube producer\n", entry);
            return ROCKET_E_UNSUPPORTED;
        }
        /* A cube IN is per slice — each one holds its own window of the producer's surface
         * and skips its scatter — so the loop is unchanged and `in` is simply unread. */
        for (s = 0; s < h->nslice; s++) {
            int src = rocket_pool_int8_prepacked_rk3576(fd, h->slice[s], in, out);
            if (src != ROCKET_OK) return src;
        }
        return ROCKET_OK;
    }

    if (!h->cube_in) {
        /* The feature cube. A row-major CHW tensor and a cube are a TRANSPOSE, and on a
         * graph that does not join into the pool this is its largest cost by far — an
         * element at a time it measured about 145 MB/s against the shared 16x16 NEON
         * block's multiple GB/s [HW sweep, H96 MAX M9].
         *
         * A FULL group goes through the block and a partial trailing one keeps the scalar
         * loop. That is the PADDING contract rather than a tail: the block stores whole
         * atoms, so it would fill this cube's dead lanes with a value the PPU reduces into
         * output channels past `c`, and what a consumer may find there is a question this
         * entry has never had to answer. The scalar loop leaves them as they were. Every
         * pooling layer of the four networks here has a multiple of sixteen channels. */
        const unsigned pitch = h->rm_in_pitch ? h->rm_in_pitch : h->iw;
        const unsigned col   = h->rm_in_col;
        const size_t px = (size_t)h->ih * pitch;      /* a channel of the CALLER's tensor */
        const int windowed = (pitch != h->iw || col != 0u);
        unsigned g = 0;
        t0 = R76P_T(prof);
        rocket_bo_prep(fd, &h->in, 1, 0);
        {
            int8_t *cube = (int8_t *)h->in.ptr;
            for (; g + C2 <= h->c; g += C2) {
                const int8_t *sp[C2];
                unsigned i;
                if (!windowed) {
                    for (i = 0; i < C2; i++) sp[i] = in + (size_t)(g + i) * px;
                    rocket_rk3576_c2_pack(cube + (size_t)(g / C2) * h->in_surf * C2,
                                          sp, C2, (size_t)h->ih * h->iw, 0);
                    continue;
                }
                /* A COLUMN SLICE READS ROWS, NOT A PLANE. The same 16x16 block, one row
                 * at a time — the rows are `pitch` apart in the caller's tensor and
                 * `iw` apart in this slice's cube. */
                for (y = 0; y < h->ih; y++) {
                    for (i = 0; i < C2; i++)
                        sp[i] = in + (size_t)(g + i) * px + (size_t)y * pitch + col;
                    rocket_rk3576_c2_pack(cube + (size_t)(g / C2) * h->in_surf * C2 +
                                              (size_t)C2 * y * h->iw,
                                          sp, C2, h->iw, 0);
                }
            }
            for (ci = g; ci < h->c; ci++) {
                int8_t *dst = cube + (size_t)(ci / C2) * h->in_surf * C2 + (ci % C2);
                for (y = 0; y < h->ih; y++) {
                    const int8_t *src = in + (size_t)ci * px + (size_t)y * pitch + col;
                    for (x = 0; x < h->iw; x++)
                        dst[(size_t)C2 * (y * h->iw + x)] = *src++;
                }
            }
        }
        rocket_bo_fini(fd, &h->in);
        R76P_ACC(prof, scatter_us, t0);
    }

    osurf = r76p_surf(h, &ooff);
    in_h[0] = h->cube_in ? h->src.handle : h->in.handle;
    in_h[1] = h->rc.handle;
    out_h[0] = osurf->handle;

    /* One PPU program, so the write check is over the whole surface. The stamp is what
     * makes "never written" a fact rather than a guess — a fresh BO's zeros cannot tell
     * an unwritten atom from a legitimately zero one — and it is BRACKETED, because a
     * bare memset leaves dirty lines that race the PPU's write DMA. */
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;
    attempts = r76p_attempts();
    rc = ROCKET_E_DEVICE;

    /* The divisor lag is invisible to the write check, so it gets its own — over the
     * positions where it can show, which is a list this shape has whether or not the
     * hazard fires. On a CUBE JOIN the input is the producer's surface rather than a
     * row-major tensor and the check reads it there, at the cost of one bracket the join
     * otherwise never pays. */
    if ((in || (h->cube_in && h->src.ptr)) &&
        r76p_lagcheck_on() && r76p_lag_observable(&h->d)) {
        unsigned cap = h->oh * h->ow;
        lagpos = (struct r76p_lag_pos *)malloc((size_t)cap * sizeof *lagpos);
        if (lagpos) nlag = r76p_lag_positions(h, lagpos, cap);
        else        ROCKET_LOGW("%s: no memory for the %u-position divisor-lag list; "
                                "the padded-average lag check is OFF for this call\n",
                                entry, cap);
    }
    /* `attempt` counts POISONING redos, each of which cycles the power domain. A LAG redo
     * spends nothing but a submit and is counted separately, against its own budget. */
    for (attempt = 0; attempt < attempts; ) {
        size_t i;
        int wrote, lag_fail;

        prof.attempts++;
        if (stamp) {
            t0 = R76P_T(prof);
            rocket_bo_prep(fd, osurf, 1, 0);
            memset((char *)osurf->ptr + ooff, stamp, h->out_bytes);
            rocket_bo_fini(fd, osurf);
            R76P_ACC(prof, stamp_us, t0);
        }
        t0 = R76P_T(prof);
        /* THE COMPLETION IS THE PPU'S, NOT THE DPU'S. A pool program is 23 PPU writes
         * and 8 PPU_RDMA with no DPU stage at all, so the DPU bits the driver waits on
         * by default can never set and the job retires on the grace period in full —
         * measured at 640 us of a 1.48 ms call, and tracking dpu_grace_us count for
         * count. Naming the class makes it the PPU's own completion instead.
         *
         * GATED on the kernel, not sent blind: the submit ioctl REJECTS a flag word it
         * does not recognise, so an older kernel would fail the submit rather than fall
         * back to the grace period. [HW sweep, H96 MAX M9] */
        if (rocket_submit_matmul_flags(fd, &h->rc, h->task_count, in_h, 2, out_h, 1,
                                       rocket_ppu_done_supported()
                                           ? ROCKET_JOB_PPU_DONE : 0u) != 0) {
            ROCKET_LOGE("%s: submit failed\n", entry);
            /* The stamp opened a PREP/FINI bracket on osurf above and this exit is inside
             * it. Close it before leaving: this file's own hazard note is that a CPU write
             * to an output BO outside the bracket races the DPU's DMA, and an error path is
             * not exempt — the caller is failing out, but the BO outlives the call. */
            if (stamp) rocket_bo_fini(fd, osurf);
            rc = ROCKET_E_DEVICE;
            goto out;
        }
        R76P_ACC(prof, submit_us, t0);
        t0 = R76P_T(prof);
        tw = R76P_T(prof);
        if (rocket_bo_prep(fd, osurf, 0, 2000000000ull) < 0) {
            ROCKET_LOGE("%s: PREP_BO on the output timed out\n", entry);
            rc = ROCKET_E_DEVICE;
            goto out;
        }
        R76P_ACC(prof, wait_us, tw);
        prof.surf_kib += (double)h->out_bytes / 1024.0;
        wrote = 1;
        lag_fail = 0;
        if (stamp) {
            tw = R76P_T(prof);
            wrote = 0;
            for (i = 0; i < h->out_bytes; i++)
                if (((const unsigned char *)osurf->ptr)[ooff + i] != stamp) {
                    wrote = 1; break;
                }
            R76P_ACC(prof, scan_us, tw);
        }
        if (wrote && nlag) {
            /* A cube join's input lives in a BO the device wrote and nothing has synced
             * for this CPU, so the read needs its own bracket. `in` is already coherent. */
            int ok;
            struct r76p_lag_score sc;
            double tl = R76P_T(prof);
            if (!in) rocket_bo_prep(fd, &h->src, 0, 2000000000ull);
            ok = r76p_divisor_ok(h, in, (const int8_t *)osurf->ptr + ooff, lagpos, nlag,
                                 &sc);
            if (!in) rocket_bo_fini(fd, &h->src);
            R76P_ACC(prof, lag_us, tl);
            prof.lag_checks++;
            h->lag_calls++;
            h->lag_discr += sc.discr;
            if (!ok) {
                prof.lag_redos++;
                h->lag_fires++;
                r76p_lag_report(entry, &sc);
                /* Written, in full, and wrong: the divisor is one output position late.
                 * Redoing the submit is what clears it — this is not the poisoning and a
                 * power cycle is neither necessary nor sufficient for it. */
                lag_fail = 1;
                lagged++;
                ROCKET_LOGD("%s: the divisor lagged one output position on attempt %u; "
                            "redoing the submit\n", entry, attempt + 1u);
            }
        }
        if (stamp) {
            tw = R76P_T(prof);
            rocket_bo_fini(fd, osurf);
            R76P_ACC(prof, fini_us, tw);
        }
        R76P_ACC(prof, read_us, t0);
        if (wrote && !lag_fail) { rc = ROCKET_OK; break; }
        if (lag_fail) {
            /* Not the poisoning: no power cycle, just redo — and out of its OWN budget,
             * since a lag redo costs a submit where a poisoning redo costs a power cycle. */
            if (lagged >= r76p_lag_attempts()) break;
            continue;
        }
        ROCKET_LOGD("%s: the program wrote nothing on attempt %u; cycling the power "
                    "domain and redoing it\n", entry, attempt + 1u);
        attempt++;
        cycled++;
        confirmed += rocket_rk3576_power_idle();
    }
out:
    free(lagpos);
    if (rc != ROCKET_OK) {
        if (lagged)
            ROCKET_LOGE("%s: the divisor lagged on %u redo(s) and the surface was never "
                        "right (ROCKET_RK3576_POOL_LAG_ATTEMPTS=%u)\n",
                        entry, lagged, r76p_lag_attempts());
        else
            ROCKET_LOGE("%s: the program wrote nothing over %u attempts (%d power cycles, "
                        "%d of them confirmed to reach suspended)\n",
                        entry, attempts, cycled, confirmed);
        return rc;
    }
    if (lagged)
        ROCKET_LOGD("%s: the divisor lag cost %u redo(s)\n", entry, lagged);

    /* CUBE OUT: the surface stays where the PPU wrote it and the next layer's CNA reads
     * it, so there is nothing to de-scatter. The fence was already waited on above. */
    t0 = R76P_T(prof);
    if (!h->cube_out) {
        const int8_t *o = (const int8_t *)osurf->ptr + ooff;
        const unsigned opitch = h->rm_out_pitch ? h->rm_out_pitch : h->ow;
        const unsigned ocol   = h->rm_out_col;
        const size_t px = (size_t)h->oh * opitch;   /* a channel of the CALLER's tensor */
        const int windowed = (opitch != h->ow || ocol != 0u);
        unsigned g = 0;
        /* The same block the other way round. Unpacking reads whole atoms and writes only
         * the live lanes, so a partial group has no padding question — it takes the block
         * too and the scalar loop is the pixel tail inside it. A column slice walks rows
         * for the same reason the scatter does. */
        for (; g < h->c; g += C2) {
            unsigned n = h->c - g < C2 ? h->c - g : C2;
            const int8_t *sg = o + (size_t)(g / C2) * h->out_surf * C2;
            int8_t *dp[C2];
            unsigned i;
            if (!windowed) {
                for (i = 0; i < n; i++) dp[i] = out + (size_t)(g + i) * px;
                rocket_rk3576_c2_unpack(dp, n, sg, (size_t)h->oh * h->ow);
                continue;
            }
            for (y = 0; y < h->oh; y++) {
                for (i = 0; i < n; i++)
                    dp[i] = out + (size_t)(g + i) * px + (size_t)y * opitch + ocol;
                rocket_rk3576_c2_unpack(dp, n, sg + (size_t)C2 * y * h->ow, h->ow);
            }
        }
    }
    R76P_ACC(prof, descat_us, t0);

    if (prof.on)
        ROCKET_LOGI("pool %ux%ux%u k%ux%u -> %ux%u: scatter %.0f stamp %.0f submit %.0f "
                    "read %.0f (wait %.0f scan %.0f fini %.0f, lag check %.0f over %u, "
                    "%u redo%s) de-scatter %.0f us over %.0f KiB (%u attempt%s%s%s)\n",
                    h->c, h->ih, h->iw, h->d.kh, h->d.kw, h->oh, h->ow,
                    prof.scatter_us, prof.stamp_us, prof.submit_us, prof.read_us,
                    prof.wait_us, prof.scan_us, prof.fini_us,
                    prof.lag_us, prof.lag_checks, prof.lag_redos,
                    prof.lag_redos == 1u ? "" : "s",
                    prof.descat_us, prof.surf_kib,
                    prof.attempts, prof.attempts == 1u ? "" : "s",
                    h->cube_in ? ", cube in" : "",
                    h->cube_out ? ", cube out" : "");
    return ROCKET_OK;
}

void rocket_pool_int8_free_rk3576(int fd, rocket_pool_int8_rk3576_handle *h)
{
    if (!h) return;
    if (h->nslice) {
        unsigned s;
        for (s = 0; s < h->nslice; s++)
            rocket_pool_int8_free_rk3576(fd, h->slice[s]);
        free(h);
        return;
    }
    if (h->rc.ptr)  rocket_bo_free(fd, &h->rc);
    if (h->out.ptr) rocket_bo_free(fd, &h->out);
    if (h->in.ptr)  rocket_bo_free(fd, &h->in);
    free(h);
}

/*
 * The transient entry: pack, run once, free. It is deliberately not a second
 * implementation — a caller that runs this shape once pays the BO churn a handle exists
 * to amortize, and a caller that runs it repeatedly holds the handle instead.
 */
int rocket_pool_int8_rk3576(int fd, const rocket_pool_desc *d, int in_zp,
                            const int8_t *in, int8_t *out)
{
    static const char entry[] = "rocket_pool_int8_rk3576";
    struct rocket_pool_int8_rk3576_handle *h;
    struct r76p_prof prof;
    double t0;
    int rc;

    if (!d || !in || !out) return ROCKET_E_SHAPE;
    if (!r76p_is_this_chip(entry)) return ROCKET_E_UNSUPPORTED;
    rc = rocket_pool_int8_rk3576_plan(d);
    if (rc != ROCKET_OK) return rc;
    if (in_zp < -128 || in_zp > 127) return ROCKET_E_SHAPE;
    if (fd < 0) return ROCKET_E_SHAPE;

    memset(&prof, 0, sizeof prof);
    prof.on = r76p_prof_on();

    t0 = R76P_T(prof);
    h = rocket_pool_int8_pack_rk3576(fd, d, in_zp);
    R76P_ACC(prof, pack_us, t0);
    if (!h) return ROCKET_E_UNSUPPORTED;

    rc = rocket_pool_int8_prepacked_rk3576(fd, h, in, out);

    t0 = R76P_T(prof);
    rocket_pool_int8_free_rk3576(fd, h);
    R76P_ACC(prof, free_us, t0);

    if (prof.on)
        ROCKET_LOGI("pool transient: pack %.0f (3 BOs + the program) free %.0f us\n",
                    prof.pack_us, prof.free_us);
    return rc;
}

/*
 * The CPU model of the above: signed integer max, or an average over the WINDOW that
 * rounds half to even. Against round-half-away-from-zero a k2 average disagrees on one
 * output in eight, so this is the rule and not a detail. An odd window has no exact
 * half and so no tie to break.
 *
 * With avg_exclude_pad the divisor is the number of taps that fell inside the plane, so
 * `n` is a per-position quantity rather than the window — which is also why the tie rule
 * has to be evaluated against that `n` and not against kh*kw.
 */
void rocket_pool_ref_int8_rk3576(const rocket_pool_desc *d, int in_zp,
                                 const int8_t *in, int8_t *out)
{
    int oh = rocket_pool_oh(d), ow = rocket_pool_ow(d);
    int c, y, x, kh, kw;

    for (c = 0; c < d->c; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                long best = -1000, sum = 0;
                /* The two per-axis divisors, which are the window unless the pad is
                 * excluded and the taps that landed inside the plane when it is. Kept as
                 * a PAIR rather than as their product because the PPU's reciprocal is per
                 * axis, so whether the tie survives is a question about each of them. */
                int dw = d->kw, dh = d->kh;
                long n;
                if (d->avg_exclude_pad) {
                    int lo, hi;
                    lo = x * d->stride_x - d->pad_left;
                    hi = lo + d->kw; if (lo < 0) lo = 0; if (hi > d->iw) hi = d->iw;
                    dw = hi > lo ? hi - lo : 0;
                    lo = y * d->stride_y - d->pad_top;
                    hi = lo + d->kh; if (lo < 0) lo = 0; if (hi > d->ih) hi = d->ih;
                    dh = hi > lo ? hi - lo : 0;
                }
                n = (long)dw * dh;
                for (kh = 0; kh < d->kh; kh++)
                    for (kw = 0; kw < d->kw; kw++) {
                        int iy = y * d->stride_y + kh - d->pad_top;
                        int ix = x * d->stride_x + kw - d->pad_left;
                        int v;
                        if (iy < 0 || ix < 0 || iy >= d->ih || ix >= d->iw) {
                            if (d->avg_exclude_pad) continue;
                            v = (d->method == POOL_METHOD_MAX) ? -128 : in_zp;
                        } else {
                            v = in[((size_t)c * d->ih + iy) * d->iw + ix];
                        }
                        if (v > best) best = v;
                        sum += v;
                    }
                (void)n;
                if (d->method == POOL_METHOD_MAX)
                    out[((size_t)c * oh + y) * ow + x] = (int8_t)best;
                else
                    out[((size_t)c * oh + y) * ow + x] = r76p_avg_round(sum, dw, dh);
            }
}
