// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_matmul_rk3576.c — the int8 matmul for the RK3576.
 *
 * A matmul is a 1x1 convolution over this part's CNA->CORE->DPU blocks, exactly as it
 * is on the RK3588; what differs is the geometry-register encoding, the operand cubes,
 * and the envelope. The register program is not the new part here — the RK3576 already
 * computes 1x1 convolutions bit-exactly — so this file is the userspace layer around
 * one: the operand scatter, the output de-scatter, the tiling planner, and the
 * per-chip dispatch point.
 *
 * WHICH PRECISION, which is the opposite of the RK3588's answer. int8 is the GEMM
 * precision here. One int8 task takes ic*kh*kw up to 4608, so a K of 4608 lands in a
 * single submit. The fp16 matmul form below (rocket_matmul_fp16_rk3576) contracts K to
 * 6144 in one task too, but at a prefill's M its resident call costs about twice the
 * resident int8 one, 17.9 against 8.6 ms at 512x1536x1536 with the BO pool off. The
 * fp16 CONVOLUTION program contracts sixteen input channels a task, a bound of that
 * program and not of the part. [HW sweep, H96 MAX M9, rocket 1.6.0, 786 MHz, 2026-09-25]
 *
 * WHAT ONE SUBMIT COSTS, and what that makes the planner's job. A submit's device time
 * rises with what it carries. At the CBUF feature budget (M*K = 262144 int8 elements),
 * submit to fence is about 0.12 ms at N 32 and 0.91 ms at N 2048, so the device's rate
 * rises almost linearly with N, from about 140 GOP/s to about 1.2 TOP/s. The whole
 * resident call, host work included, is 1.2 ms at N 32 and 6.7 ms at N 2048. M*K is
 * capped by that budget and K by the resident weight slice at 4608, so N is what the
 * planner should spend. [HW sweep, H96 MAX M9, rocket 1.6.0, 786 MHz, resident weight,
 * pool off, 2026-09-25]
 *
 * THE M AXIS CARRIES NO CONSTRAINT, which is also the opposite of the RK3588. There
 * rows are the conv's spatial height, a height under 4 mis-computes, and software pads
 * M==1 to 4. Here M=1, 2, 3, 5, 7, 17 and 31 are each bit-exact, and so is every
 * factorization of M into a plane — a single row of pixels, a single column, and
 * everything between. The plane is a TILING choice, not a correctness one.
 * [HW sweep, H96 MAX M9]
 *
 * THE OUTPUT IS int8, THROUGH THE DPU's REQUANT, at every K one task can contract, and
 * that is the fast path. The RK3588's int8 matmul writes int32 and sums K partials on
 * the host; this part's DPU will do that too, but its 32-bit writer delivers only some
 * of the output channels — the byte budget is a function of the DPU's operand width and
 * not of the output element's. The way round it is the weight cube rather than a
 * register — see the int32 section below — and it costs the output-channel axis twice
 * over. So K past one task's contraction is SPLIT rather than refused, through that
 * path, and the requant moves to the host.
 */
#include <dirent.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_sysfs.h"      /* the one enumeration of the bound NPU cores */
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"

#define C2 16

/* The idle a poisoned submit needs and whether the output surface is stamped before it
 * runs are properties of the CHIP, not of the matmul: the convolution entry points hit
 * the same hazard through the same driver. Both live in rocket_rk3576_internal.h, and
 * the int32 section below is where they are documented and defined. */
#define R76_SENTINEL_BYTE ROCKET_RK3576_SENTINEL_BYTE

/* Attempts a row task gets before the call refuses. Each retry is one submit plus, when
 * the failure is the poisoning, one power cycle — cheap against a wrong answer, and the
 * measured need is real: on a 128-task shape the redo fires on 1-12 tasks a run and
 * heals all of them, while at four attempts about one run in thirty had a task it did
 * not recover. */
#define R76_I32_TASK_ATTEMPTS 8

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);
int weight_conv_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

/* ============================================================================
 * SECTION — phase profiling
 *
 * A THIRD accumulator, on the same ROCKET_MM_PROFILE knob and with its own exit line,
 * for the same reason the int8 path has one separate from the fp16 path: the buckets are
 * not the same SET. `g_prof`/`g_prof_i8` in rocket_matmul.c split an RK3588 job into
 * pack/gen/sync/submit/wait/read, and three of this route's real terms have no bucket
 * there at all — the O(K*N) `sum_abs_w` pass, the per-tile per-column ramp plan, and the
 * poisoning stamp-and-check. A term with no bucket is charged to whichever neighbour the
 * timer happens to span, which is how a profile reports a plausible wrong split.
 *
 * The unit is ONE CALL of the entry, and the tile loop's buckets accumulate ACROSS tiles
 * within it, so `calls` counts entry calls and not tiles or submits. `submits` and
 * `tiles` are carried beside them because a per-submit or per-tile term cannot be read
 * off a total that does not say how many there were.
 *
 * WHAT IT CANNOT SEE: anything the caller does around the entry — the frontend's
 * quantize, its rotation and its dequantize are in ggml-rocket and are timed there, on
 * this same knob; and the wall this route is quoted against also carries attention, the
 * norms, rope and the scheduler, none of which reach this file. So these buckets sum to
 * the entry's own time, NOT to the model's.
 * ==========================================================================*/
static int r76_mm_profile(void)
{
    static int v = -1;
    if (v < 0) v = getenv("ROCKET_MM_PROFILE") != NULL;
    return v;
}

static pthread_mutex_t g_prof_r76_mu = PTHREAD_MUTEX_INITIALIZER;
static struct {
    double sumabs, packA, packB, coeff, gen, alloc, stamp, submit, wait, read;
    double setup, bofree, wall;
    long calls, tiles, submits, redos;
} g_prof_r76;
static int g_prof_r76_armed = 0;

static void r76_mm_prof_dump(void)
{
    const double t = g_prof_r76.sumabs + g_prof_r76.packA + g_prof_r76.packB
                   + g_prof_r76.coeff + g_prof_r76.gen + g_prof_r76.alloc
                   + g_prof_r76.stamp + g_prof_r76.submit + g_prof_r76.wait
                   + g_prof_r76.read + g_prof_r76.setup + g_prof_r76.bofree;
    /* `wall` spans the entry from its first statement to the fold, so wall-accounted is
     * the in-entry time no bucket owns — printed here so pricing it never needs a second
     * instrument's wall subtracted from this one's sum. */
    ROCKET_LOGI("ROCKET rk3576 matmul profile total(ms): wall=%.0f accounted=%.0f "
                "sumabs=%.0f packA=%.0f packB=%.0f coeff=%.0f gen=%.0f alloc=%.0f "
                "stamp=%.0f submit=%.0f wait=%.0f read=%.0f setup=%.0f bofree=%.0f  "
                "over %ld calls / %ld tiles / %ld submits (%ld redos)\n",
                g_prof_r76.wall, t, g_prof_r76.sumabs, g_prof_r76.packA, g_prof_r76.packB,
                g_prof_r76.coeff, g_prof_r76.gen, g_prof_r76.alloc, g_prof_r76.stamp,
                g_prof_r76.submit, g_prof_r76.wait, g_prof_r76.read,
                g_prof_r76.setup, g_prof_r76.bofree,
                g_prof_r76.calls, g_prof_r76.tiles, g_prof_r76.submits,
                g_prof_r76.redos);
}

static double r76_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* One accumulator struct per call, folded in under the mutex at the end, so the timers
 * themselves cost no lock and a concurrent caller cannot interleave partial phases. */
struct r76_mm_prof {
    double sumabs, packA, packB, coeff, gen, alloc, stamp, submit, wait, read;
    double setup, bofree, wall;
    long tiles, submits, redos;
};

static void r76_mm_prof_fold(const struct r76_mm_prof *p)
{
    pthread_mutex_lock(&g_prof_r76_mu);
    if (!g_prof_r76_armed) { atexit(r76_mm_prof_dump); g_prof_r76_armed = 1; }
    g_prof_r76.sumabs += p->sumabs; g_prof_r76.packA += p->packA;
    g_prof_r76.packB  += p->packB;  g_prof_r76.coeff += p->coeff;
    g_prof_r76.gen    += p->gen;    g_prof_r76.alloc += p->alloc;
    g_prof_r76.stamp  += p->stamp;  g_prof_r76.submit += p->submit;
    g_prof_r76.wait   += p->wait;   g_prof_r76.read  += p->read;
    g_prof_r76.setup  += p->setup;  g_prof_r76.bofree += p->bofree;
    g_prof_r76.wall   += p->wall;
    g_prof_r76.calls++;
    g_prof_r76.tiles   += p->tiles;
    g_prof_r76.submits += p->submits;
    g_prof_r76.redos   += p->redos;
    pthread_mutex_unlock(&g_prof_r76_mu);
}

/* PROF_T0/PROF_ADD: no-ops when the knob is off, so the un-profiled path pays one
 * predictable branch per phase and no clock_gettime at all. */
#define PROF_T0()        (prof ? r76_now_ms() : 0.0)
#define PROF_ADD(f, t0)  do { if (prof) pr.f += r76_now_ms() - (t0); } while (0)

/* Both machine parameters come from the profile, not from literals here: the feature
 * budget in int8 elements is the CBUF data allocation, and the output-channel tile is
 * the profile's max_tile. ROCKET_RK3576_MM_NT overrides the tile per run. */
static unsigned r76_mm_feature_elems(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    return (unsigned)(hw->cbuf_banks * hw->cbuf_bank_size);
}

/* ============================================================================
 * SECTION — the plane
 *
 * M rows become a plane of M pixels, and any (iw, ih) with iw*ih == M carries them.
 * What the choice costs is granules: a feature row is ceil(iw*K/64) of them, against
 * the profile's CBUF data allocation, so the pixels one task can carry come out the
 * same however the plane is cut — PROVIDED iw*K is a whole number of granules. When it
 * is not, every row of the plane rounds up and the waste is real, so the widest
 * divisor of M that divides evenly is the one to take.
 * ==========================================================================*/
static void r76_mm_plane(unsigned M, unsigned K, unsigned *iw_out, unsigned *ih_out)
{
    unsigned budget_px = r76_mm_feature_elems() / K;
    unsigned iw, best = 1;
    /* ROCKET_RK3576_MM_IW forces the plane width, for probing questions whose answer
     * depends on how the same M is cut — the wide int32 writer's surface-height bound
     * among them, since a bound over rows and a bound over atoms are the same number
     * until the plane changes. It is a probe knob: it skips the granule rule above, so
     * a width that is not a whole number of granules wastes CBUF rather than failing. */
    {
        const char *e = getenv("ROCKET_RK3576_MM_IW");
        unsigned forced = (e && *e) ? (unsigned)strtoul(e, NULL, 0) : 0u;
        if (forced && forced <= M && M % forced == 0u) {
            *iw_out = forced;
            *ih_out = M / forced;
            return;
        }
    }

    if (!budget_px) budget_px = 1;
    for (iw = 1; iw <= M; iw++) {
        if (M % iw) continue;
        if (iw > budget_px) break;
        if ((iw * K) % 64u) continue;
        best = iw;
    }
    *iw_out = best;
    *ih_out = M / best;
}

static unsigned r76_mm_nt(void)
{
    const char *e = getenv("ROCKET_RK3576_MM_NT");
    long v = (e && *e) ? strtol(e, NULL, 0) : 0;
    if (v > 0) return (unsigned)v;
    return (unsigned)rocket_hw_current()->max_tile;
}

/* Whether this entry may fall onto the int32 K-split route. Off by default, and the
 * reason is in the refusal's own message. Read per call, so a probe can bracket one
 * shape with it. */
static int r76_mm_ksplit_opt_in(void)
{
    const char *e = getenv("ROCKET_RK3576_MM_KSPLIT");
    return e && *e && *e != '0';
}

/* ============================================================================
 * SECTION — the plan
 * ==========================================================================*/
/* The output-channel tile this shape can actually run, or 0 if none can.
 *
 * Shrinking rather than refusing is the whole of the planner's job on this axis. The
 * emitter's guards — the resident weight slice, the 2944-channel bound, the 6 MiB
 * weight cube — are all functions of the TILE's oc, so a tile that is too wide is not
 * an unsupported shape, it is a tile to halve. What no tile can fix is K: it is the
 * contraction, the output is int8 through the requant, and a partial cannot be summed
 * without quantizing it. */
static unsigned r76_mm_fit_nt_mult(unsigned iw, unsigned K, unsigned N,
                                   unsigned oc_mult, unsigned *rows_out)
{
    unsigned nt = r76_mm_nt();

    if (nt > N) nt = N;
    nt = (nt / 32u) * 32u;
    if (!nt) nt = 32u;

    for (;;) {
        unsigned rows = rocket_rk3576_max_task_rows(iw, K, nt * oc_mult, 1u, 1u, 0);
        if (rows) { if (rows_out) *rows_out = rows; return nt; }
        if (nt <= 32u) return 0;
        nt = ((nt / 2u) / 32u) * 32u;
        if (!nt) nt = 32u;
    }
}

static unsigned r76_mm_fit_nt(unsigned iw, unsigned K, unsigned N, unsigned *rows_out)
{
    return r76_mm_fit_nt_mult(iw, K, N, 1u, rows_out);
}

/* THE TILE AND ITS ROW WINDOW ARE CHOSEN SO NO JOB STALLS. A job's output-channel group g
 * completes only if its weight slice, staged at its 64 KiB phase, ends inside the weight
 * area the task's feature allowance leaves (rocket_rk3576_weight_phase_groups()); the first
 * group that does not is where the job stops, unwritten past it and retired at the
 * backstop. The planner's widest row window programs the highest allowance and so the
 * smallest weight area, which is how a K in 2049-2560 at a two-row window stalls where a
 * one-row window of the same tile does not. Two plans avoid it: a shorter row window whose
 * allowance holds every group of the tile, or a tile no wider than the groups the planner's
 * own window holds. This takes whichever needs fewer jobs, a tie to the full width, because
 * a job's device time rises with what it carries and the per-job cost does not. Measured
 * at M 512 N 1536, first call of a process: K 2240 23.5-25.6 ms with a one-row window
 * against 27.5-29.4 with a 320-channel tile (8 jobs against 20), K 2432 24.7-30.4 against
 * 35.6-37.6 (8 against 40), where the narrowing below reaches 31-42 ms after a first call
 * of 0.76-1.02 s. Past K 4096 even F=0 stalls at some K, and only the tile cap is left:
 * K 4480 at 160 channels, 78-92 ms against the narrowing's 89-92 after 0.82-1.28 s.
 * [HW sweep, H96 MAX M9, rocket 1.6.0, pool off, tests/rk3576_retire_probe, 2026-09-27]
 *
 * The narrowing stays behind this as the fallback for a stall the rule does not predict.
 * ROCKET_RK3576_MM_PHASE=0 turns this off, which leaves the narrowing to find the width. */
static int r76_mm_phase_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_MM_PHASE");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 1;
    }
    return cached;
}

/* The widest window of an even spread of `ih` rows under a cap of `cap`: the row planner
 * counts ceil(ih/cap) tasks and spreads the rows evenly over them. */
static unsigned r76_mm_spread_max(unsigned ih, unsigned cap)
{
    unsigned n = (ih + cap - 1u) / cap;
    return (ih + n - 1u) / n;
}

/* Whether a task of `rows` rows at this plane and tile holds every group of the tile. */
static int r76_mm_rows_hold(unsigned iw, unsigned ih, unsigned K, unsigned nt,
                            unsigned rows, unsigned *groups_out)
{
    unsigned f, g;
    if (rocket_rk3576_cbuf_f(iw, K, r76_mm_spread_max(ih, rows), nt, 1u, 1u, 0, &f) < 0) {
        if (groups_out) *groups_out = 0;
        return 0;
    }
    g = rocket_rk3576_weight_phase_groups(32u * K, f);
    if (groups_out) *groups_out = g;
    return g >= (nt + 31u) / 32u;
}

/* `*nt` in, the stall-free tile out; `*row_cap` the row window to cap the plan at (0 = the
 * planner's own). Both only ever shrink. */
static void r76_mm_phase(unsigned iw, unsigned ih, unsigned K, unsigned N,
                         unsigned *nt, unsigned *row_cap)
{
    unsigned iter, cur = *nt;

    *row_cap = 0;
    if (!r76_mm_phase_on()) return;
    for (iter = 0; iter < 8u; iter++) {
        unsigned rows = rocket_rk3576_max_task_rows(iw, K, cur, 1u, 1u, 0);
        unsigned r, full = 0, g = 0, capped;
        unsigned long jobs_full = ~0ul, jobs_cap = ~0ul;

        if (!rows) break;                 /* no plan: the planner's refusal speaks */
        if (rows > ih) rows = ih;
        if (r76_mm_rows_hold(iw, ih, K, cur, rows, &g)) break;   /* already holds */
        for (r = rows - 1u; r >= 1u; r--)
            if (r76_mm_rows_hold(iw, ih, K, cur, r, NULL)) { full = r; break; }
        if (full)
            jobs_full = (unsigned long)((N + cur - 1u) / cur) * ((ih + full - 1u) / full);
        capped = g ? 32u * g : 0u;
        if (capped && capped < cur)
            jobs_cap = (unsigned long)((N + capped - 1u) / capped) * ((ih + rows - 1u) / rows);
        if (full && jobs_full <= jobs_cap) { *row_cap = full; break; }
        if (!capped || capped >= cur) break;   /* nothing holds: the narrowing is left */
        cur = capped;                          /* a narrower tile may plan more rows */
    }
    *nt = cur;
}

int rocket_matmul_plan_int8_rk3576(int M, int K, int N, int *Mt, int *Kt, int *Nt)
{
    unsigned iw, ih, nt, rows = 0, row_cap = 0;

    if (M <= 0 || K <= 0 || N <= 0) return ROCKET_E_SHAPE;
    if (K % 32 || N % 32) {
        ROCKET_LOGE("rk3576 matmul: K=%d N=%d — both must be multiples of 32 (the "
                    "int8 weight cube groups each channel axis by 32)\n", K, N);
        return ROCKET_E_SHAPE;
    }

    r76_mm_plane((unsigned)M, (unsigned)K, &iw, &ih);
    nt = r76_mm_fit_nt(iw, (unsigned)K, (unsigned)N, &rows);
    if (nt) {
        r76_mm_phase(iw, ih, (unsigned)K, (unsigned)N, &nt, &row_cap);
        rows = rocket_rk3576_max_task_rows(iw, (unsigned)K, nt, 1u, 1u, 0);
        if (row_cap && row_cap < rows) rows = row_cap;
    }
    if (!nt) {
        /* Not an error here: this entry reports the SINGLE-TASK plan and there is none.
         * rocket_matmul_int8_rk3576() declines the shape when this returns E_SHAPE — the
         * K-split route it used to fall onto wedges the device — so the caller's own
         * refusal, not this line, is where the message that matters is. */
        ROCKET_LOGI("rk3576 matmul: K=%d does not fit one task even at a 32-channel "
                    "output tile (the boundary is K>=6176), so there is no single-task "
                    "plan to report and rocket_matmul_int8_rk3576() will decline it\n", K);
        return ROCKET_E_SHAPE;
    }

    if (Mt) *Mt = (int)(rows * iw);
    if (Kt) *Kt = K;
    if (Nt) *Nt = (int)nt;
    return (int)(((unsigned)N + nt - 1u) / nt * ((ih + rows - 1u) / rows));
}

/* ============================================================================
 * SECTION — the transient-BO pool
 *
 * The entry allocates and frees its feature, weight, coefficient, output and regcmd
 * BOs per tile per call — `alloc` is its largest bucket — and the same sizes recur on
 * every call of a given shape. ROCKET_RK3576_BO_POOL=1 keeps freed BOs on a small
 * free list keyed by (fd, size) and hands them back on the next matching request;
 * ROCKET_RK3576_BO_POOL_MB caps the held bytes (default 64). Default OFF: holding
 * device memory across calls is an exposure — whether the per-call free is what keeps
 * IOMMU mapping churn inside what the BO-lifetime fixes were written for is
 * unmeasured — so the pool is a measured opt-in, not the shipped path.
 *
 * The knob is read on EVERY allocation (not latched), so an A/B can flip it between
 * arms inside one process. Correctness does not depend on the pool: a pooled BO is
 * fully rewritten before use by the same packs that fill a fresh one, and the
 * poisoning sentinel stamps the output BO per task either way — what a pooled BO
 * skips is only the kernel's zero-fill, which the sentinel never relied on.
 * ==========================================================================*/
/* Both knobs are read on EVERY pool get and put, so they are resolved once — the rest
 * of this file caches its knobs for exactly this reason. */
static int r76_bo_pool_on(void)
{
    static _Atomic int v = -1;
    int c = atomic_load_explicit(&v, memory_order_relaxed);
    if (c < 0) {
        const char *e = getenv("ROCKET_RK3576_BO_POOL");
        c = e && atoi(e) != 0;
        atomic_store_explicit(&v, c, memory_order_relaxed);
    }
    return c;
}
static size_t r76_bo_pool_cap(void)
{
    static _Atomic long v = -1;
    long c = atomic_load_explicit(&v, memory_order_relaxed);
    if (c < 0) {
        const char *e = getenv("ROCKET_RK3576_BO_POOL_MB");
        long mb = e ? atol(e) : 64;
        if (mb < 0) mb = 0;
        c = mb << 20;
        atomic_store_explicit(&v, c, memory_order_relaxed);
    }
    return (size_t)c;
}
#define R76_BO_POOL_SLOTS 16
static pthread_mutex_t g_r76_bo_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static struct { int fd; rocket_bo bo; } g_r76_bo_pool[R76_BO_POOL_SLOTS];
static size_t g_r76_bo_pool_bytes;

/* Smallest pooled BO that covers the request wins; a larger BO is always safe (the
 * consumers bound every access by the size they asked for). It is not free: a pooled BO
 * keeps its allocated size, so once the mix of shapes or entries changes a small request
 * can settle onto a far larger BO, and PREP_BO/FINI_BO sync the whole of it every call.
 * The fp16 per-call entry's 8 KiB weight request settles onto a 257 KiB coefficient BO
 * after one shape change [HW sweep 2026-09-25, rk3576_mm_fp16_cost; the matching is
 * inferred from the sizes]. */
static int r76_bo_get(int fd, size_t size, rocket_bo *bo)
{
    if (r76_bo_pool_on()) {
        int i, best = -1;
        pthread_mutex_lock(&g_r76_bo_pool_mu);
        for (i = 0; i < R76_BO_POOL_SLOTS; i++) {
            if (!g_r76_bo_pool[i].bo.ptr || g_r76_bo_pool[i].fd != fd) continue;
            if (g_r76_bo_pool[i].bo.size < size) continue;
            if (best < 0 || g_r76_bo_pool[i].bo.size < g_r76_bo_pool[best].bo.size)
                best = i;
        }
        if (best >= 0) {
            *bo = g_r76_bo_pool[best].bo;
            g_r76_bo_pool_bytes -= g_r76_bo_pool[best].bo.size;
            memset(&g_r76_bo_pool[best], 0, sizeof g_r76_bo_pool[best]);
            pthread_mutex_unlock(&g_r76_bo_pool_mu);
            return 0;
        }
        pthread_mutex_unlock(&g_r76_bo_pool_mu);
    }
    return rocket_bo_alloc32(fd, size, bo);
}
static void r76_bo_put(int fd, rocket_bo *bo)
{
    if (bo->ptr && r76_bo_pool_on()) {
        int i;
        pthread_mutex_lock(&g_r76_bo_pool_mu);
        if (g_r76_bo_pool_bytes + bo->size <= r76_bo_pool_cap()) {
            for (i = 0; i < R76_BO_POOL_SLOTS; i++) {
                if (g_r76_bo_pool[i].bo.ptr) continue;
                g_r76_bo_pool[i].fd = fd;
                g_r76_bo_pool[i].bo = *bo;
                g_r76_bo_pool_bytes += bo->size;
                memset(bo, 0, sizeof *bo);
                pthread_mutex_unlock(&g_r76_bo_pool_mu);
                return;
            }
        }
        pthread_mutex_unlock(&g_r76_bo_pool_mu);
    }
    if (bo->ptr) rocket_bo_free(fd, bo);
}
void rocket_rk3576_bo_pool_drain(int fd)
{
    int i;
    pthread_mutex_lock(&g_r76_bo_pool_mu);
    for (i = 0; i < R76_BO_POOL_SLOTS; i++) {
        if (!g_r76_bo_pool[i].bo.ptr || g_r76_bo_pool[i].fd != fd) continue;
        g_r76_bo_pool_bytes -= g_r76_bo_pool[i].bo.size;
        rocket_bo_free(fd, &g_r76_bo_pool[i].bo);
        memset(&g_r76_bo_pool[i], 0, sizeof g_r76_bo_pool[i]);
    }
    pthread_mutex_unlock(&g_r76_bo_pool_mu);
}

/* ============================================================================
 * SECTION — the resident weight cube
 *
 * One weight, packed once into the int8 weight-cube layout in a device BO. The pack
 * puts column n's 32-byte k-runs at (n/32)*nK1*1024 + (n%32)*32 + (k/32)*1024, and
 * the tile loop steps n0 by a multiple of 32, so the whole-N cube is exactly the
 * concatenation of the per-tile cubes and one BO serves every N tile at every M —
 * a tile's program just offsets weights_dma by (n0/32)*nK1*1024. The contract, and
 * what a stale object computes, are on the declarations in rocket_matmul.h.
 * ==========================================================================*/
struct rocket_rk3576_wbo {
    rocket_bo bo;
    int K, N;
};

int rocket_rk3576_wbo_create(int fd, int K, int N, const int8_t *B,
                             struct rocket_rk3576_wbo **out)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    struct rocket_rk3576_wbo *w;
    unsigned nK1, k1, n;
    size_t bytes;

    if (out) *out = NULL;
    if (fd < 0 || !B || !out) return ROCKET_E_SHAPE;
    if (strcmp(hw->name, "rk3576") != 0) {
        ROCKET_LOGE("rocket_rk3576_wbo_create: this is the RK3576 encoding and the "
                    "active profile is %s\n", hw->name);
        return ROCKET_E_UNSUPPORTED;
    }
    if (K <= 0 || N <= 0 || (K % 32) || (N % 32)) {
        ROCKET_LOGE("rk3576 wbo: K=%d N=%d — both must be multiples of 32 (the int8 "
                    "weight cube groups each channel axis by 32)\n", K, N);
        return ROCKET_E_SHAPE;
    }
    nK1 = ((unsigned)K + 31u) / 32u;
    bytes = (size_t)((unsigned)N / 32u) * nK1 * 1024u;
    w = calloc(1, sizeof *w);
    if (!w) return ROCKET_E_NOMEM;
    if (rocket_bo_alloc32(fd, bytes, &w->bo) < 0) { free(w); return ROCKET_E_NOMEM; }
    rocket_bo_prep(fd, &w->bo, 1, 0);
    /* The same blocked copy the per-call path uses, over the whole N. K and N are
     * multiples of 32, so every 1024-byte (n-group, k-group) block is fully written
     * and no memset is needed. */
    {
        int8_t *cube = (int8_t *)w->bo.ptr;
        for (n = 0; n < (unsigned)N; n++) {
            const int8_t *src = B + (size_t)n * (unsigned)K;
            int8_t *dst = cube + (size_t)(n / 32u) * nK1 * 1024u
                               + (size_t)(n % 32u) * 32u;
            for (k1 = 0; k1 < nK1; k1++)
                memcpy(dst + (size_t)k1 * 1024u, src + (size_t)k1 * 32u, 32);
        }
    }
    rocket_bo_fini(fd, &w->bo);
    w->K = K; w->N = N;
    *out = w;
    return ROCKET_OK;
}

void rocket_rk3576_wbo_free(int fd, struct rocket_rk3576_wbo *w)
{
    if (!w) return;
    if (w->bo.ptr) rocket_bo_free(fd, &w->bo);
    free(w);
}

/* ============================================================================
 * SECTION — one N tile
 * ==========================================================================*/
struct r76_mm_bos {
    rocket_bo in, w, coeff, out, rc;
};

static void r76_mm_free(int fd, struct r76_mm_bos *b)
{
    if (b->rc.ptr)    r76_bo_put(fd, &b->rc);
    if (b->out.ptr)   r76_bo_put(fd, &b->out);
    if (b->coeff.ptr) r76_bo_put(fd, &b->coeff);
    if (b->w.ptr)     r76_bo_put(fd, &b->w);
    if (b->in.ptr)    r76_bo_put(fd, &b->in);
    memset(b, 0, sizeof *b);
}

/* The shipped body, with `scale_n` optional.
 *
 * NULL is the per-tensor entry: one output scale, programmed straight into the OUT_CVT
 * pair. Non-NULL is the per-output-channel entry: `scale_n[n]` is column n's own output
 * scale, carried by the coefficient group's C ramp over a shared (MUL, SHIFT). The two
 * differ in exactly two places — the coefficient packing and what `out_scale` is — which
 * is why they are one function rather than two: everything else about a tile, from the
 * weight cube to the write guard to the row plan, is the same program. */
/* A TILE WHOSE JOB STALLS IS RE-PLANNED NARROWER — the FALLBACK behind the plan. At some K
 * an int8 matmul job stops at a fixed output channel of its tile and the driver retires it
 * at the backstop with the channels before it written: 320 at K=2240, 384 at 2208, 288 at
 * 2272, 160 at 2432, at Mt 128 [HW sweep, H96 MAX M9, 2026-09-27]. Where the job stops is
 * the weight-slice phase rule (rocket_rk3576_weight_phase_groups()), and r76_mm_phase()
 * plans every tile and row window so that no group breaks it, so on a shape the rule
 * describes this never fires. It stays for a stall the rule does not predict: a row task
 * whose job is retired with its surface partly written halves the tile's width, down to 32
 * channels, and the tile is redone. An UNWRITTEN retirement is the poisoning and keeps its
 * power-cycled redo. The width that completed is kept per K for the rest of the process, so
 * a caller that repeats the shape pays the retirements once. ROCKET_RK3576_MM_NARROW=0 turns
 * it off. */
static int r76_mm_narrow_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_MM_NARROW");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 1;
    }
    return cached;
}

enum { R76_NT_CAPS = 16 };
static struct { int k; unsigned nt; } g_r76_nt_cap[R76_NT_CAPS];
static pthread_mutex_t g_r76_nt_cap_mu = PTHREAD_MUTEX_INITIALIZER;

static unsigned r76_nt_cap_get(int K)
{
    unsigned i, cap = 0;
    pthread_mutex_lock(&g_r76_nt_cap_mu);
    for (i = 0; i < R76_NT_CAPS; i++)
        if (g_r76_nt_cap[i].k == K) { cap = g_r76_nt_cap[i].nt; break; }
    pthread_mutex_unlock(&g_r76_nt_cap_mu);
    return cap;
}

static void r76_nt_cap_set(int K, unsigned nt)
{
    unsigned i, slot = R76_NT_CAPS;
    pthread_mutex_lock(&g_r76_nt_cap_mu);
    for (i = 0; i < R76_NT_CAPS; i++) {
        if (g_r76_nt_cap[i].k == K) { slot = i; break; }
        if (!g_r76_nt_cap[i].k && slot == R76_NT_CAPS) slot = i;
    }
    if (slot < R76_NT_CAPS) { g_r76_nt_cap[slot].k = K; g_r76_nt_cap[slot].nt = nt; }
    pthread_mutex_unlock(&g_r76_nt_cap_mu);
}

/* WHICH PER-COLUMN RAMP. On by default: the DPU shift word carries the gain, so the tile's
 * largest column scale sits at C = 32767 whatever the contraction depth. Under a nonzero
 * shift the BS stage holds `(acc + bias)*C` wide and rounds it half to even after the
 * multiply [HW sweep, H96 MAX M9, tests/rk3576_coeff_c.c shift], so the product is bounded
 * by the shift rather than by int32. `ROCKET_RK3576_MM_PC_SHIFT=0` restores the shift-0
 * ramp exactly as it shipped before — the word at zero, C capped at INT32_MAX over the
 * column's accumulator bound, the OUT_CVT gain passed as the reciprocal of its reciprocal —
 * which is the arm to A/B against. The convolution's ROCKET_RK3576_PC_SHIFT does not reach
 * this entry, so the two A/B independently. Read per call, so a probe can bracket one call. */
static int r76_mm_pc_shift_on(void)
{
    const char *e = getenv("ROCKET_RK3576_MM_PC_SHIFT");
    return !(e && *e == '0');
}

/* The per-column plan of one N tile, from whichever ramp is selected. The shift ramp is the
 * convolution's own planner with every column live (`sum_w` NULL): one copy of the rule. */
int rocket_rk3576_plan_percol(unsigned n0, unsigned tile_n, unsigned nreg,
                              int32_t *tile_bias, const int64_t *sum_abs_w,
                              const float *scale_n, int16_t *C,
                              struct rocket_rk3576_percol_plan *pl)
{
    memset(pl, 0, sizeof *pl);
    if (r76_mm_pc_shift_on()) {
        struct rocket_rk3576_pc_plan pc;
        if (rocket_rk3576_plan_perchannel_bs("rk3576 matmul", n0, tile_n, nreg, tile_bias,
                                             NULL, sum_abs_w, 1.0f, scale_n, 1.0f, 0,
                                             NULL, C, &pc) != 0)
            return -1;
        pl->shift_ramp  = 1;
        pl->bs_shift    = pc.bs_shift;
        pl->gain        = pc.base_scale;
        pl->max_rel_err = pc.max_rel_err;
        pl->n_clamp     = pc.n_clamp;
        return 0;
    }
    {
        /* The shift-0 ramp exactly as it shipped, and at its cost: it does not count its
         * clamped columns (a column below half the base computes at the base's gain), and
         * its worst_rel_err includes them. */
        float base = 0.0f;
        double err = 0.0;
        if (rocket_rk3576_plan_perchannel("rk3576 matmul", n0, tile_n, nreg, tile_bias,
                                          sum_abs_w, 1.0f, scale_n, 1.0f, NULL, C, &base,
                                          &err) != 0)
            return -1;
        pl->gain        = base;
        pl->max_rel_err = err;
    }
    return 0;
}

static int r76_mm_int8(int fd, int M, int K, int N,
                       const int8_t *A, const int8_t *B,
                       const int32_t *bias, float scale, const float *scale_n,
                       const int64_t *sum_abs_in,
                       const struct rocket_rk3576_wbo *wbo,
                       int8_t *C, double *worst_rel_err)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    struct r76_mm_bos b = {0};
    uint64_t *ops = NULL;
    int32_t *tile_bias = NULL;
    int16_t *cmul = NULL;
    /* `sum_abs_own` is this call's, and is freed; `sum_abs_w` is what the planner reads
     * and may point at the caller's, which is not. */
    int64_t *sum_abs_own = NULL;
    const int64_t *sum_abs_w = NULL;
    rocket_rk3576_row_task *plan = NULL;
    unsigned iw, ih, nt, n0, adv, surf_elems, max_tasks, row_cap = 0;
    size_t in_bytes;
    unsigned char blank = rocket_rk3576_sentinel_on() ? (unsigned char)R76_SENTINEL_BYTE : 0;
    int rc = ROCKET_E_SHAPE;
    const int prof = r76_mm_profile();
    struct r76_mm_prof pr = {0};
    double pt0;
    const double wall_t0 = PROF_T0();

    if (fd < 0 || !A || (!B && !wbo) || !C) return ROCKET_E_SHAPE;
    if (worst_rel_err) *worst_rel_err = 0.0;
    if (strcmp(hw->name, "rk3576") != 0) {
        ROCKET_LOGE("rocket_matmul_int8_rk3576: this is the RK3576 encoding and the "
                    "active profile is %s\n", hw->name);
        return ROCKET_E_UNSUPPORTED;
    }
    /* K and N are the one cheap consistency check a resident weight admits: the packed
     * cube cannot be validated against anything per call (checking it would be the pass
     * the object exists to skip), but an object of the wrong shape is always the
     * caller's bookkeeping and refusing beats indexing off the end of its BO. */
    if (wbo && (wbo->K != K || wbo->N != N)) {
        ROCKET_LOGE("rk3576 matmul: resident weight is K=%d N=%d and the call is "
                    "K=%d N=%d — the object was created from a different weight\n",
                    wbo->K, wbo->N, K, N);
        return ROCKET_E_SHAPE;
    }
    if (!scale_n && scale <= 0.0f) {
        ROCKET_LOGE("rk3576 matmul: scale must be positive (the DPU's OUT_CVT gates "
                    "the whole BS stage off at zero and writes an empty surface)\n");
        return ROCKET_E_SHAPE;
    }
    /* Checked here rather than per tile: a non-positive column scale is the caller's
     * error and the plan below would report it as a shape refusal instead. */
    if (scale_n) {
        int n;
        for (n = 0; n < N; n++) {
            if (!(scale_n[n] > 0.0f)) {
                ROCKET_LOGE("rk3576 matmul: scale_n[%d] is %g — every per-column output "
                            "scale must be positive (a zero C term gates the BS stage "
                            "off for its whole eight-channel group and the DPU writes a "
                            "full but empty surface)\n", n, (double)scale_n[n]);
                return ROCKET_E_SHAPE;
            }
        }
    }
    /* Ahead of the plan call, because the plan refuses malformed shapes and K-split ones
     * with the same code: without this, an unaligned K reports the K>=6176 bound below
     * and sends the caller looking for a contraction limit it is nowhere near. */
    if (M <= 0 || K <= 0 || N <= 0 || (K % 32) || (N % 32)) {
        ROCKET_LOGE("rk3576 matmul: M=%d K=%d N=%d — K and N must be multiples of 32 "
                    "(the int8 weight cube groups each channel axis by 32)\n", M, K, N);
        return ROCKET_E_SHAPE;
    }
    pt0 = PROF_T0();
    {
        int ntile = 0;
        if (rocket_matmul_plan_int8_rk3576(M, K, N, NULL, NULL, &ntile) < 0) {
            /* A K with no single-task plan can only be run by splitting it through the
             * int32 writer and requantizing on the host. That route TAKES THE DEVICE DOWN
             * at prefill scale: at M=512 K=8192 N=2048 the write guard spends its eight
             * power cycles, the entry returns -4, the part raises the driver's DMA-error
             * WARN_ON, and the NPU then computes nothing in ANY process until the board is
             * rebooted — a module reload leaves core 0 failing to probe at -22. So this
             * entry declines instead of routing, and a caller falls back to its own CPU
             * path with the device intact. [HW, H96 MAX M9, one shape, 2026-08-07]
             *
             * The boundary is where the single-task planner gives up, which is K >= 6176
             * for every (M, N) enumerated — a condition on K alone, because the planner
             * shrinks the output tile to 32 channels before it refuses and what is left
             * is the CBUF pool against a one-group weight cube.
             *
             * Small cells in this class do compute — M=32 K=8192 N=128 and M=16 K=16384
             * N=64 have run green in the gate list every session, and M=64 K=8192 N=32
             * scored exact three times — so this is a bound on the SHAPES MEASURED, not a
             * decoded mechanism, and it is set where it is because the cost of declining
             * a shape that would have worked is a CPU fallback and the cost of accepting
             * one that does not is a reboot. ROCKET_RK3576_MM_KSPLIT=1 restores the route
             * for measurement; rocket_matmul_int8_rk3576_i32() still reaches it directly
             * and carries the same warning. */
            int32_t *acc;
            int m, rc32;
            if (wbo) {
                ROCKET_LOGE("rk3576 matmul: K=%d has no single-task plan and the "
                            "resident-weight entry does not carry the row-major B the "
                            "int32 K-split fallback needs — declining\n", K);
                return ROCKET_E_UNSUPPORTED;
            }
            if (!r76_mm_ksplit_opt_in()) {
                ROCKET_LOGE("rk3576 matmul: K=%d has no single-task plan (the boundary is "
                            "K>=6176), and the int32 K-split route that would run it has "
                            "been seen to wedge the NPU across processes until a reboot. "
                            "Declining — run this K on the host, or set "
                            "ROCKET_RK3576_MM_KSPLIT=1 to take the route anyway\n", K);
                return ROCKET_E_UNSUPPORTED;
            }
            if (M <= 0 || N <= 0) return ROCKET_E_SHAPE;
            acc = calloc((size_t)M * N, sizeof *acc);
            if (!acc) return ROCKET_E_NOMEM;
            rc32 = rocket_matmul_int8_rk3576_i32(fd, M, K, N, A, B, bias, acc);
            if (rc32 == ROCKET_OK) {
                for (m = 0; m < M * N; m++) {
                    float v = (float)acc[m] * (scale_n ? scale_n[m % N] : scale);
                    long r = (long)(v < 0.0f ? v - 0.5f : v + 0.5f);
                    C[m] = (int8_t)(r < -128 ? -128 : (r > 127 ? 127 : r));
                }
            }
            free(acc);
            return rc32;
        }
        nt = (unsigned)ntile;
    }
    r76_mm_plane((unsigned)M, (unsigned)K, &iw, &ih);
    r76_mm_phase(iw, ih, (unsigned)K, (unsigned)N, &nt, &row_cap);
    {
        unsigned cap = r76_mm_narrow_on() ? r76_nt_cap_get(K) : 0u;
        if (cap && nt > cap) nt = cap;
    }

    surf_elems = rocket_rk3576_out_surf_elems(iw, ih, 0);
    in_bytes   = (size_t)(((unsigned)K + C2 - 1) / C2) * ih * iw * C2;
    max_tasks  = ih + 1u;

    ops   = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    plan  = calloc(max_tasks, sizeof *plan);
    if (!ops || !plan) { rc = ROCKET_E_NOMEM; goto done; }
    PROF_ADD(setup, pt0);

    /* THE C RAMP IS CAPPED BY THE ACCUMULATOR, so the plan needs each column's own
     * sum of |weight| — the actual one. The int8 envelope (128*K*127) is one to two
     * orders of magnitude looser than a real weight row and the difference IS the
     * available precision, which is why this O(N*K) pass is worth its cost. It is one
     * pass over the same bytes the weight cube copies.
     *
     * A caller holding the weight across calls has usually computed this already, and it
     * is `M`-independent, so `_perc_sa` lets it hand the sums over instead — the same
     * numbers, one pass fewer. The supplied array is NOT validated: checking it is the
     * pass being skipped, and what a wrong one does is on that entry's declaration. */
    if (scale_n) {
        if (sum_abs_in) {
            sum_abs_w = sum_abs_in;
        } else if (!B) {
            /* The resident-weight wrapper refuses NULL sums before it gets here; this
             * is the backstop for an internal caller, because the recompute below is a
             * pass over the row-major B this call does not have. */
            ROCKET_LOGE("rk3576 matmul: per-column scales with no sum_abs_w and no "
                        "row-major B to derive them from\n");
            rc = ROCKET_E_SHAPE; goto done;
        } else {
            int n;
            pt0 = PROF_T0();
            sum_abs_own = calloc((size_t)N, sizeof *sum_abs_own);
            if (!sum_abs_own) { rc = ROCKET_E_NOMEM; goto done; }
            for (n = 0; n < N; n++) {
                const int8_t *w = B + (size_t)n * K;
                int64_t s = 0;
                int k;
                for (k = 0; k < K; k++) s += w[k] < 0 ? -(int64_t)w[k] : (int64_t)w[k];
                sum_abs_own[n] = s;
            }
            sum_abs_w = sum_abs_own;
            PROF_ADD(sumabs, pt0);
        }
    }

    /* The FEATURE cube is packed once and shared by every N tile: the tiling is on the
     * output-channel axis, which the feature side does not see.
     *
     * BLOCKED, NOT PER ELEMENT. feature_data() at C2=16 puts channel k of pixel m at
     * (k/16)*ih*iw*16 + 16*m + (k%16), so sixteen consecutive channels are sixteen
     * consecutive bytes at BOTH ends — a copy rather than a scatter. K is a multiple of
     * 32 on this path, so every run is a whole sixteen. The per-element form is what
     * the index function above spells out and is the reference for this one. */
    /* Straight into the mapping, not via a host stage buffer. The BO is acquired FIRST
     * so the pack has somewhere to land: staging it and then copying costs a calloc and
     * a free of in_bytes plus a second full pass over them, and the profiler attributed
     * both halves to packA, which is where they hid. The memset is what the calloc used
     * to do — ih*iw exceeds M, and the pixels past M must read zero. */
    pt0 = PROF_T0();
    if (r76_bo_get(fd, in_bytes, &b.in) < 0) { rc = ROCKET_E_NOMEM; goto done; }
    PROF_ADD(alloc, pt0);
    pt0 = PROF_T0();
    rocket_bo_prep(fd, &b.in, 1, 0);
    memset(b.in.ptr, 0, in_bytes);
    {
        size_t plane = (size_t)ih * iw * C2;
        unsigned k1, ngrp = (unsigned)K / C2;
        int m;
        for (m = 0; m < M; m++) {
            const int8_t *src = A + (size_t)m * K;
            int8_t *dst = (int8_t *)b.in.ptr + (size_t)C2 * (unsigned)m;
            for (k1 = 0; k1 < ngrp; k1++)
                memcpy(dst + (size_t)k1 * plane, src + (size_t)k1 * C2, C2);
        }
    }
    rocket_bo_fini(fd, &b.in);
    PROF_ADD(packA, pt0);

    pt0 = PROF_T0();
    if (r76_bo_get(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }
    PROF_ADD(alloc, pt0);

    for (n0 = 0; n0 < (unsigned)N; n0 += adv) {
        unsigned tile_n = (unsigned)N - n0 < nt ? (unsigned)N - n0 : nt;
        unsigned nreg = rocket_rk3576_pad_oc(tile_n);
        size_t w_bytes = (size_t)((nreg + 31) / 32) * (((unsigned)K + 31) / 32) * 32 * 32;
        int narrow = 0;
        adv = 0;
        size_t coeff_bytes = rocket_rk3576_coeff_bytes(nreg);
        size_t obytes = (size_t)((nreg + C2 - 1) / C2) * surf_elems * C2;
        conv_params_t p = {0};
        uint32_t in_h[4], out_h[1];
        unsigned ntask = 1, t, n;
        float base_scale;
        unsigned bs_shift = 0;      /* the DPU shift word this tile's coefficients carry */
        int shift_ramp = 0;         /* the OUT_CVT gain goes in unrounded under the shift */

        pr.tiles++;
        pt0 = PROF_T0();
        if (b.w.ptr)     r76_bo_put(fd, &b.w);
        if (b.coeff.ptr) r76_bo_put(fd, &b.coeff);
        if (b.out.ptr)   r76_bo_put(fd, &b.out);
        memset(&b.w, 0, sizeof b.w);
        memset(&b.coeff, 0, sizeof b.coeff);
        memset(&b.out, 0, sizeof b.out);

        if ((!wbo && r76_bo_get(fd, w_bytes, &b.w) < 0) ||
            r76_bo_get(fd, coeff_bytes, &b.coeff) < 0 ||
            r76_bo_get(fd, obytes, &b.out) < 0) { rc = ROCKET_E_NOMEM; goto done; }
        PROF_ADD(alloc, pt0);

        /* The WEIGHT cube is per-tile and each tile is its own convolution, so its
         * group count follows the tile rather than the whole N — unless the caller
         * handed the weight in resident, in which case the cached whole-N cube IS the
         * concatenation of these per-tile ones and the program below just offsets into
         * it. */
        if (!wbo) {
            pt0 = PROF_T0();
            rocket_bo_prep(fd, &b.w, 1, 0);
            memset(b.w.ptr, 0, w_bytes);
            /* BLOCKED, NOT PER ELEMENT — and this is the one that mattered. At kh=kw=1
             * weight_conv_int8() reduces to (n/32)*nK1*1024 + (k/32)*1024 + (n%32)*32 +
             * (k%32), so thirty-two consecutive k are thirty-two consecutive bytes at
             * both ends. A K*N element-at-a-time scatter through the index function is
             * a million calls at K=N=1024, which was most of that shape's wall clock;
             * the memset above still covers the output-channel tail of a partial
             * group. */
            {
                unsigned nK1 = ((unsigned)K + 31u) / 32u, k1;
                int8_t *w = (int8_t *)b.w.ptr;
                for (n = 0; n < tile_n; n++) {
                    const int8_t *src = B + (size_t)(n0 + n) * K;
                    int8_t *dst = w + (size_t)(n / 32u) * nK1 * 1024u
                                    + (size_t)(n % 32u) * 32u;
                    for (k1 = 0; k1 < nK1; k1++)
                        memcpy(dst + (size_t)k1 * 1024u, src + (size_t)k1 * 32u, 32);
                }
            }
            rocket_bo_fini(fd, &b.w);
            PROF_ADD(packB, pt0);
        }

        /* The COEFFICIENT buffer is NOT a flat int32 bias array on this part, and a
         * zeroed one makes the DPU write a full but entirely empty surface whatever
         * the MAC did — the C term gates the BS stage. Pad the tail channels of a
         * partial group with a zero bias so they carry a C term too. */
        pt0 = PROF_T0();
        free(tile_bias);
        tile_bias = calloc(nreg, sizeof *tile_bias);
        if (!tile_bias) { rc = ROCKET_E_NOMEM; goto done; }
        if (bias)
            for (n = 0; n < tile_n; n++) tile_bias[n] = bias[n0 + n];
        /* THE PER-COLUMN RAMP IS PLANNED PER TILE, because the (MUL, SHIFT) it rides on
         * is per TASK and every N tile is its own program. A tile whose columns share a
         * scale gets a flat ramp and loses nothing; a tile spanning a wide spread pays
         * its resolution there and nowhere else. Under the shift word (the default) the
         * ramp spans 32767:1 in every tile; on the shift-0 ramp it spans only the int32
         * headroom, a few hundred at a GEMM's depth. The planner is the convolution
         * path's — one copy, because the bounds it balances are the part's and not the
         * op's. */
        base_scale = scale;
        if (scale_n) {
            struct rocket_rk3576_percol_plan pp;
            free(cmul);
            cmul = calloc(nreg, sizeof *cmul);
            if (!cmul) { rc = ROCKET_E_NOMEM; goto done; }
            if (rocket_rk3576_plan_percol(n0, tile_n, nreg, tile_bias, sum_abs_w, scale_n,
                                          cmul, &pp) != 0) {
                rc = ROCKET_E_SHAPE; goto done;
            }
            base_scale = pp.gain;
            bs_shift   = pp.bs_shift;
            shift_ramp = pp.shift_ramp;
            if (worst_rel_err && pp.max_rel_err > *worst_rel_err)
                *worst_rel_err = pp.max_rel_err;
        }
        rocket_bo_prep(fd, &b.coeff, 1, 0);
        if (cmul)
            rocket_rk3576_pack_coeff_perc(b.coeff.ptr, coeff_bytes, tile_bias, nreg,
                                          NULL, cmul, 1);
        else
            rocket_rk3576_pack_coeff(b.coeff.ptr, coeff_bytes, tile_bias, nreg);
        /* The shift word, in the tail group the pack just zeroed (BS_BASE_ADDR1 reads it),
         * in both sign fields: the stage shifts a negative product by its own field, and
         * one field alone leaves every negative output at another scale. */
        if (bs_shift &&
            rocket_rk3576_pack_bs_shift(b.coeff.ptr, coeff_bytes, nreg, 0, bs_shift) != 0) {
            rocket_bo_fini(fd, &b.coeff);
            rc = ROCKET_E_SHAPE; goto done;
        }
        rocket_bo_fini(fd, &b.coeff);
        PROF_ADD(coeff, pt0);

        p.ic = (uint16_t)K;    p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
        p.oc = (uint16_t)nreg; p.oh = (uint16_t)ih; p.ow = (uint16_t)iw;
        p.kh = 1; p.kw = 1;
        p.stride_y = 1; p.stride_x = 1;
        p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)ih;
        p.int8_out = 1;
        /* The requant the caller asked for. conv_scale = in*w/out is what the emitter
         * derives OUT_CVT from. The OUT_CVT itself has no per-channel multiplier: on the
         * per-column entry `base` is the shared gain the ramp rides on and the per-column
         * part is in the coefficient group's C, not here.
         *
         * Under the shift ramp the gain goes in as in_scale with unit w and out, so the
         * emitter derives its pair from exactly the float the planner built C against.
         * The per-tensor entry and the shift-0 ramp keep out = 1/base, as they shipped:
         * 1/(1/x) is not x for about 17% of floats, and for 0.033% of them the derived
         * (MUL, SHIFT) moves by one multiplier unit (2^-14 of the gain) [host sweep of
         * every float over 25 exponents]. */
        if (shift_ramp) {
            p.in_scale = base_scale; p.w_scale = 1.0f; p.out_scale = 1.0f;
        } else {
            p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f / base_scale;
        }
        p.input_zero_point = 0x80;   /* symmetric int8: every zero-point term cancels */
        p.output_zero_point = 0x80;
        p.weight_zero_point = 0x80;
        p.tasks       = ops;
        p.input_dma   = b.in.dma_address;
        p.weights_dma = wbo ? wbo->bo.dma_address
                              + (size_t)(n0 / 32u) * (((unsigned)K + 31u) / 32u) * 1024u
                            : b.w.dma_address;
        p.bias_dma    = b.coeff.dma_address;
        p.output_dma  = b.out.dma_address;

        pt0 = PROF_T0();
        {
            conv_params_t q = p;
            if (rocket_rk3576_plan_rows_cap(&q, 0, row_cap, plan, max_tasks, &ntask) < 0) {
                ROCKET_LOGE("rk3576 matmul: no row plan for M=%d K=%d N tile %u "
                            "(plane %ux%u)\n", M, K, tile_n, iw, ih);
                rc = ROCKET_E_SHAPE; goto done;
            }
        }
        PROF_ADD(gen, pt0);

        in_h[0] = b.in.handle; in_h[1] = wbo ? wbo->bo.handle : b.w.handle;
        in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
        out_h[0] = b.out.handle;

        /* THE POISONING REACHES THIS PATH TOO, and it used to pass straight through.
         * An int32-output job leaves the next submit of ANY kind writing nothing, across
         * calls and across processes, so the plain int8 matmul inherits the hazard from
         * whatever ran before it — its first submit comes back untouched and the caller
         * reads a correctly sized, entirely stale tile with no fault and normal timing.
         * Measured in a soak: one whole gate run in twenty failed its first shape this
         * way, at 1 element of 256 correct.
         *
         * Stamping the surface is what makes that detectable without false positives. A
         * region that is still the sentinel throughout is a submit that did not write;
         * an all-zero region is a legitimate int8 result after the requant and must not
         * be read as failure. The stamp is bracketed by PREP_BO and FINI_BO so no dirty
         * line is left to race the DPU's DMA. */
        if (blank) {
            pt0 = PROF_T0();
            rocket_bo_prep(fd, &b.out, 1, 0);
            memset(b.out.ptr, blank, obytes);
            rocket_bo_fini(fd, &b.out);
            PROF_ADD(stamp, pt0);
        }

        for (t = 0; t < ntask; t++) {
            unsigned tattempt;
            int task_ok;
            conv_params_t q = p;
            q.ih = plan[t].ih; q.oh = plan[t].oh;
            q.pad_top = plan[t].pad_top;
            q.input_dma  = p.input_dma  + plan[t].feature_off;
            q.output_dma = p.output_dma + plan[t].output_off;
            /* ALWAYS, not only when the plan split: the emitter derives the DDR
             * channel-group stride from the FULL plane, and leaving these at the
             * window makes every group past the first read at the wrong offset. */
            q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
            pt0 = PROF_T0();
            if (gen_conv2d_int8_rk3576(&q) != 0) { rc = ROCKET_E_SHAPE; goto done; }
            PROF_ADD(gen, pt0);
            /* PER TASK, not per tile. One poisoned submit among several leaves its own
             * rows stale while its siblings are full, so a check over the whole tile
             * reads "something was written" and passes the hole through. */
            task_ok = 0;
            for (tattempt = 0; tattempt < R76_I32_TASK_ATTEMPTS; tattempt++) {
                uint64_t tjob, jns;
                pr.submits++;
                if (tattempt) pr.redos++;
                pt0 = PROF_T0();
                rocket_bo_prep(fd, &b.rc, 1, 0);
                memcpy(b.rc.ptr, ops, q.task_count * sizeof(uint64_t));
                rocket_bo_fini(fd, &b.rc);
                tjob = rocket_rk3576_job_clock();
                if (rocket_submit_matmul(fd, &b.rc, q.task_count, in_h, 4, out_h, 1, 4000) != 0) {
                    rc = ROCKET_E_DEVICE; goto done;
                }
                PROF_ADD(submit, pt0);
                pt0 = PROF_T0();
                if (rocket_bo_prep(fd, &b.out, 0, 2000000000ull) < 0) {
                    rc = ROCKET_E_DEVICE; goto done;
                }
                jns = rocket_rk3576_job_ns(tjob);
                PROF_ADD(wait, pt0);
                if (blank) {
                    unsigned char *sp = (unsigned char *)b.out.ptr + plan[t].output_off;
                    size_t tb = (size_t)plan[t].oh * iw * C2, si;
                    int twrote = 0, retired;
                    pt0 = PROF_T0();
                    for (si = 0; si < tb; si++) if (sp[si] != blank) { twrote = 1; break; }
                    /* A RETIRED JOB IS REDONE EVEN IF IT WROTE: one byte satisfies the
                     * check above, and a retirement can leave a partial surface. The
                     * rows the check reads go back to the sentinel inside this bracket,
                     * so a redo that writes nothing cannot pass on this attempt's bytes. */
                    retired = rocket_rk3576_score_retired("rk3576 matmul", jns, twrote);
                    if (retired && twrote) memset(sp, blank, tb);
                    rocket_bo_fini(fd, &b.out);
                    PROF_ADD(stamp, pt0);
                    if (twrote && !retired) { task_ok = 1; break; }
                    /* Written in part and retired: the stall past a fixed channel, which a
                     * redo of the same tile repeats. The tile is redone narrower instead. */
                    if (retired && twrote && nt > 32u && r76_mm_narrow_on()) {
                        narrow = 1;
                        break;
                    }
                    ROCKET_LOGD("rk3576 matmul: n0=%u row task %u %s, idling and redoing "
                                "it\n", n0, t, retired ? "was retired" : "wrote nothing");
                    rocket_rk3576_power_idle();
                    continue;
                }
                rocket_bo_fini(fd, &b.out);
                if (rocket_rk3576_score_retired("rk3576 matmul", jns, 1)) {
                    rocket_rk3576_power_idle();
                    continue;
                }
                task_ok = 1;
                break;
            }
            if (narrow) break;
            /* A task the retries did not recover is a hole in the surface, and quietly
             * de-scattering it hands the caller a matrix that is exactly one row task
             * stale — no fault, normal timing, plausible values. Refuse instead. */
            if (!task_ok) {
                ROCKET_LOGE("rk3576 matmul: n0=%u row task %u wrote nothing or was "
                            "retired on each of %d attempts — refusing to return a "
                            "partial surface\n", n0, t, R76_I32_TASK_ATTEMPTS);
                rc = ROCKET_E_DEVICE; goto done;
            }
        }

        if (narrow) {
            unsigned was = nt;
            nt = ((nt / 2u) / 32u) * 32u;
            if (nt < 32u) nt = 32u;
            r76_nt_cap_set(K, nt);
            ROCKET_LOGW("rk3576 matmul: K=%d: a job of a %u-channel tile was retired with "
                        "its surface partly written, the stall this K has past a fixed "
                        "output channel; redoing the tile at %u channels\n", K, was, nt);
            rocket_rk3576_power_idle();
            continue;                       /* adv is 0: the same n0, narrower */
        }

        /* De-scatter this tile's channels straight into the caller's row-major C. */
        pt0 = PROF_T0();
        rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
        /* BLOCKED, NOT PER ELEMENT, the same way round: y*iw + x IS m, so the source
         * index is (n/16)*surf_elems*16 + 16*m + (n%16) and sixteen consecutive output
         * channels are sixteen consecutive bytes at both ends. */
        {
            const int8_t *o = (const int8_t *)b.out.ptr;
            size_t surf = (size_t)surf_elems * C2;
            int m;
            for (m = 0; m < M; m++) {
                const int8_t *src = o + (size_t)C2 * (unsigned)m;
                int8_t *dst = C + (size_t)m * N + n0;
                for (n = 0; n < tile_n; n += C2) {
                    unsigned run = tile_n - n < C2 ? tile_n - n : (unsigned)C2;
                    memcpy(dst + n, src + (size_t)(n / C2) * surf, run);
                }
            }
        }
        rocket_bo_fini(fd, &b.out);
        PROF_ADD(read, pt0);
        adv = tile_n;
    }
    rc = ROCKET_OK;

done:
    free(ops); free(plan); free(tile_bias);
    free(cmul); free(sum_abs_own);
    pt0 = PROF_T0();
    r76_mm_free(fd, &b);
    PROF_ADD(bofree, pt0);
    /* Folded on EVERY exit that reaches here, including the refusals: a call that
     * returned -4 still spent its pack and its submits, and dropping those would report
     * a clean split over a run that was not clean. Folded LAST so `wall` covers the BO
     * teardown the bucket above just priced. */
    if (prof) { pr.wall = r76_now_ms() - wall_t0; r76_mm_prof_fold(&pr); }
    return rc;
}

int rocket_matmul_int8_rk3576(int fd, int M, int K, int N,
                              const int8_t *A, const int8_t *B,
                              const int32_t *bias, float scale, int8_t *C)
{
    return r76_mm_int8(fd, M, K, N, A, B, bias, scale, NULL, NULL, NULL, C, NULL);
}

int rocket_matmul_int8_rk3576_perc(int fd, int M, int K, int N,
                                   const int8_t *A, const int8_t *B,
                                   const int32_t *bias, const float *scale_n,
                                   int8_t *C, double *worst_rel_err)
{
    if (!scale_n) return ROCKET_E_SHAPE;
    return r76_mm_int8(fd, M, K, N, A, B, bias, 0.0f, scale_n, NULL, NULL, C,
                       worst_rel_err);
}

int rocket_matmul_int8_rk3576_perc_sa(int fd, int M, int K, int N,
                                      const int8_t *A, const int8_t *B,
                                      const int32_t *bias, const float *scale_n,
                                      const int64_t *sum_abs_w,
                                      int8_t *C, double *worst_rel_err)
{
    if (!scale_n) return ROCKET_E_SHAPE;
    return r76_mm_int8(fd, M, K, N, A, B, bias, 0.0f, scale_n, sum_abs_w, NULL, C,
                       worst_rel_err);
}

int rocket_matmul_int8_rk3576_perc_wbo(int fd, int M, int K, int N,
                                       const int8_t *A,
                                       const struct rocket_rk3576_wbo *wbo,
                                       const int32_t *bias, const float *scale_n,
                                       const int64_t *sum_abs_w,
                                       int8_t *C, double *worst_rel_err)
{
    if (!scale_n || !wbo) return ROCKET_E_SHAPE;
    if (!sum_abs_w) {
        ROCKET_LOGE("rk3576 matmul: the resident-weight entry needs sum_abs_w — the "
                    "row-major B is not an argument here and re-deriving the sums from "
                    "the packed cube would be the O(N*K) pass this entry removes\n");
        return ROCKET_E_SHAPE;
    }
    return r76_mm_int8(fd, M, K, N, A, NULL, bias, 0.0f, scale_n, sum_abs_w, wbo, C,
                       worst_rel_err);
}

/* ============================================================================
 * SECTION — the int32 output, and the K split it buys
 *
 * The DPU will emit its raw 32-bit accumulator on an integer program. What it will not
 * do is write all of them. Its write budget is one 16-byte atom per (16-channel block,
 * pixel) — the INT8 surface — whatever the OUTPUT element width is, so at four bytes an
 * element most channels never reach DDR. What that budget IS a function of is the DPU's
 * own operand width, PROC_PRECISION, and widening that to int32 doubles it. Two writers
 * come out of it, and the arithmetic is bit-identical in both:
 *
 *   gen_conv2d_int8_rk3576_i32out()       one atom:  the first 8 channels of every 32
 *   gen_conv2d_int8_rk3576_i32out_wide()  two atoms: the first 8 channels of every 16
 *
 * Two atoms is the ceiling — int16, fp16 and bfloat16 reach it too, the float ones
 * destroying the arithmetic on the way, and int4 and int8 do not. [HW sweep, H96 MAX M9]
 *
 * THE WAY AROUND IT IS THE WEIGHT CUBE, not a register. Program a multiple of the output
 * channels and put real channel n in a slot the writer delivers, leaving the rest zero.
 * Every real channel then lands in a delivered slot and the surface reads back as a
 * plain cube. The narrow writer wants 4x and slot 32*(n/8)+n%8, which collapses to the
 * RK3588's int32 cube `(n/4)*ow*oh_full*4 + 4*p + n%4`; the wide one wants 2x and slot
 * 16*(n/8)+n%8, addressed with rocket_rk3576_i32_wide_word().
 *
 * WHAT IT COSTS is the output-channel axis, spent over. The bytes are not wasted — the
 * budget comes out exactly the int32 surface — but the resident weight slice, the N tile
 * and the 2944-channel bound are all functions of the PROGRAMMED oc, so the multiplier
 * is paid in MACs per submit. Halving it from 4 to 2 doubles the N tile and so halves
 * the submits, which is what this path is billed in: measured 4 submits to 2 at
 * M=64 K=1024 N=2048, and 8 to 4 at M=32 K=1024 N=4096. Below the tile cap the two cost
 * the same. [HW sweep, H96 MAX M9]
 *
 * THE WIDE WRITER'S SURFACE HEIGHT IS BOUNDED, at oh_full * oc < 4096, and past it the
 * map silently stops being the map. The bound is measured; the mechanism is not decoded.
 * A row task here is a standalone 1x1 convolution with its own surface, so the planner
 * below honours the bound by splitting the row plan further.
 * ==========================================================================*/

/* The idle an i32out job needs before the NEXT submit will write.
 *
 * IT IS NOT A SETTLING TIME. What clears the hazard is the driver's runtime-PM
 * autosuspend cycling the NPU power domain, so the idle that works is the driver's
 * autosuspend delay plus the suspend/resume round trip — and with runtime suspend
 * turned off entirely (`power/control` = `on`) no amount of idle clears it at all. See
 * the submit loop for the measurement.
 *
 * So the default is read from the driver rather than fixed. The law is TWICE the
 * autosuspend delay: over a 5 x 7 sweep of delay against gap, the gap at which a chain
 * of wide-output submits first reaches 20 of 20 is ~100 ms at a 50 ms delay, ~40 at 20,
 * ~20 at 10, ~10 at 5 and ~5 at 1. The small constant beside it is margin, and it is what
 * dominates once the delay is low. ROCKET_RK3576_MM_GAP_MS overrides the result outright.
 *
 * LOWERING THE DRIVER'S DELAY IS THE LEVER, and it is a system policy rather than
 * anything this library can set: mainline picks 50 ms as ~3 frames at 60 Hz, for an NPU
 * feeding a display pipeline, and `power/autosuspend_delay_ms` is writable. At 1 ms every
 * gate still passes bit-exactly, the per-submit dispatch floor is unchanged (1079 -> 1104
 * us, which a back-to-back chain never lets the timer touch), and the fp16 first conv's
 * gate group falls from 122-513 ms a shape to 1.3-122: 224x224 k3 s2 goes 257 -> 20 ms.
 * [HW sweep, H96 MAX M9]
 *
 * EVERY BOUND CORE HOLDS THE DOMAIN, so every one of them is driven. The part's two NPU
 * cores are two platform devices with one shared power domain between them, and the
 * domain collapses only once BOTH are runtime-suspended. On an image whose device tree
 * enables core 1 the kernel fans an fd's work across both, so EITHER can be the active
 * one at the moment the guard looks — sampled once a second through a matmul gate run,
 * the two take turns. A guard that waited on the first device therefore returned while
 * the other still held the domain up, and the next wide-output submit wrote nothing:
 * measured as 21 of 163 library-conv cases failing with core 1 at the stock delay and 1
 * with its delay forced to zero. So the devices are enumerated from the driver's sysfs
 * directory rather than named, and the kick writes, waits on and restores all of them.
 * [HW sweep, H96 MAX M9] */
#define R76_PM_MAX_DEVS 8

/* One "<dev>/power/" prefix per bound rocket device, discovered once. */
static char r76_pm_dir[R76_PM_MAX_DEVS][192];
static int r76_pm_ndev = -1;            /* -1 unprobed */

/* Open `<r76_pm_dir[i]><leaf>`. */
static FILE *r76_pm_open(int i, const char *leaf, const char *mode)
{
    char path[256];
    if (snprintf(path, sizeof path, "%s%s", r76_pm_dir[i], leaf) >= (int)sizeof path)
        return NULL;
    return fopen(path, mode);
}

/* Fill r76_pm_dir[] from /sys/bus/platform/drivers/rocket/<dev>, and return the delay the
 * devices are set to (they share one policy, so the first readable one is it), or -1. */
static int r76_pm_probe(void)
{
    char names[ROCKET_SYSFS_MAX_DEVS][ROCKET_SYSFS_NAME_MAX];
    int ndev = rocket_sysfs_bound_devices(names, ROCKET_SYSFS_MAX_DEVS);
    int delay = -1, i;

    r76_pm_ndev = 0;
    if (ndev <= 0) return -1;
    for (i = 0; i < ndev && r76_pm_ndev < R76_PM_MAX_DEVS; i++) {
        FILE *fp;
        long v;
        if (snprintf(r76_pm_dir[r76_pm_ndev], sizeof r76_pm_dir[0],
                     "/sys/bus/platform/drivers/rocket/%s/power/",
                     names[i]) >= (int)sizeof r76_pm_dir[0])
            continue;
        fp = r76_pm_open(r76_pm_ndev, "autosuspend_delay_ms", "r");
        if (!fp) continue;
        if (fscanf(fp, "%ld", &v) == 1 && v >= 0 && v < 10000) {
            if (delay < 0) delay = (int)v;
            r76_pm_ndev++;
        }
        fclose(fp);
    }
    return r76_pm_ndev ? delay : -1;
}

static int r76_autosuspend_ms(void)
{
    if (r76_pm_ndev < 0) return r76_pm_probe();
    if (!r76_pm_ndev) return -1;
    {
        FILE *fp = r76_pm_open(0, "autosuspend_delay_ms", "r");
        long v;
        int got = fp && fscanf(fp, "%ld", &v) == 1 && v >= 0 && v < 10000;
        if (fp) fclose(fp);
        return got ? (int)v : -1;
    }
}

/* ============================================================================
 * SECTION — the PM kick: pay the power cycle instead of waiting for it
 *
 * The guard below is an idle only because the driver's autosuspend timer is what
 * cycles the power domain, and that timer is sized for a media pipeline rather than
 * for this hazard. Lowering the delay system-wide fixes the wide-output paths and
 * charges everything else: a resume costs about 230 us, and any workload whose
 * inter-job gap falls between the new delay and the old one pays it on every job,
 * a 60 Hz frame interval included. [HW sweep, H96 MAX M9]
 *
 * So drive the transition instead of waiting for it. Write a zero delay, wait for
 * `runtime_status` to actually read `suspended`, and put the caller's delay back. The
 * guard then costs one real power cycle rather than a worst-case idle, the steady-state
 * policy every other consumer of this NPU sees is unchanged, and the wait is a poll on
 * the thing being waited for rather than a timer.
 *
 * It needs WRITE access to that sysfs file, which a process without privilege does not
 * have — so it is attempted, checked, and falls back to the plain idle. A udev rule
 * granting a group write on `power/autosuspend_delay_ms` is what makes it available
 * unprivileged. ROCKET_RK3576_PM_KICK=0 turns it off.
 *
 * THE WINDOW IS THE ONE COST. Between writing zero and writing the caller's value back
 * there are a few milliseconds in which a kill would leave the system's delay at zero.
 * That is a degraded power policy rather than a broken one, and it is self-healing:
 * a delay of zero is not a value any system sets deliberately, so it is read as an
 * interrupted kick and restored to mainline's 50 rather than preserved.
 * ==========================================================================*/
/* What mainline's rocket_core.c sets, and what an interrupted kick is healed to. */
#define R76_PM_DELAY_DEFAULT_MS 50

/* Write `ms` to every bound device. 1 only if all of them took it: one device left at
 * the old delay is one core left holding the shared domain up. */
static int r76_pm_write_delay(int ms)
{
    int i, ok = r76_pm_ndev > 0;
    for (i = 0; i < r76_pm_ndev; i++) {
        FILE *fp = r76_pm_open(i, "autosuspend_delay_ms", "w");
        if (!fp) { ok = 0; continue; }
        if (fprintf(fp, "%d\n", ms) <= 0) ok = 0;
        if (fclose(fp) != 0) ok = 0;
    }
    return ok;
}

/* 1 once EVERY device reports `suspended`, 0 if they have not within `budget_ms`. The
 * power domain is shared, so a single active core means it never collapsed. */
static int r76_pm_wait_suspended(int budget_ms)
{
    int waited = 0;

    if (r76_pm_ndev <= 0) return 0;
    for (;;) {
        int i, all = 1;
        for (i = 0; i < r76_pm_ndev && all; i++) {
            FILE *fp = r76_pm_open(i, "runtime_status", "r");
            char st[32] = {0};
            int got = 0;
            if (fp) {
                got = fscanf(fp, "%31s", st) == 1;
                fclose(fp);
            }
            if (!got || strcmp(st, "suspended")) all = 0;
        }
        if (all) return 1;
        if (waited >= budget_ms) return 0;
        {
            struct timespec ts = { 0, 500000L };   /* 0.5 ms */
            nanosleep(&ts, NULL);
        }
        waited++;   /* a poll is ~0.5 ms, so this bounds at ~2x budget_ms in wall time */
    }
}

int rocket_rk3576_power_idle(void)
{
    static int cached = -2;
    static int kick = -2;          /* -2 unprobed, -1 unavailable, else the delay to restore */
    const char *g = getenv("ROCKET_RK3576_MM_GAP_MS");
    int gms;
    struct timespec ts;

    if (g && *g) {
        gms = (int)strtol(g, NULL, 0);
    } else {
        if (cached == -2) {
            int d = r76_autosuspend_ms();
            cached = d >= 0 ? 2 * d + 5 : 150;
            if (cached < 5) cached = 5;
            if (cached > 300) cached = 300;
            ROCKET_LOGI("rk3576 matmul i32: the 32-bit writer's hazard is cleared by the "
                        "NPU power domain cycling, so the inter-submit idle is sized from "
                        "the driver's autosuspend delay (%d ms) at %d ms\n", d, cached);
        }
        gms = cached;
    }

    /* Drive the power cycle rather than waiting for the autosuspend timer, when the
     * sysfs delay is writable. Probed once: a process without privilege falls back to
     * the idle below and pays what it always paid. */
    if (kick == -2) {
        int d = r76_autosuspend_ms();
        kick = -1;
        if (d >= 0 && getenv("ROCKET_RK3576_PM_KICK") &&
            strcmp(getenv("ROCKET_RK3576_PM_KICK"), "0") == 0) {
            /* explicitly off */
        } else if (d >= 0 && r76_pm_write_delay(d ? d : R76_PM_DELAY_DEFAULT_MS)) {
            /* A zero here is an earlier kick that was killed mid-window, not a policy —
             * heal it rather than adopting it as the value to restore. */
            kick = d ? d : R76_PM_DELAY_DEFAULT_MS;
            ROCKET_LOGI("rk3576: the power-domain cycle the wide-output guard waits for is "
                        "driven directly on all %d bound core(s) (autosuspend delay %d ms "
                        "is writable), so the guard costs one cycle rather than the "
                        "timer's worst case\n", r76_pm_ndev, kick);
        }
    }
    if (kick >= 0) {
        if (r76_pm_write_delay(0)) {
            int done = r76_pm_wait_suspended(gms);
            r76_pm_write_delay(kick);
            if (done) return 1;
            /* It did not suspend inside the budget — fall through to the plain idle
             * rather than returning with the hazard possibly uncleared. */
            ROCKET_LOGD("rk3576: the power domain did not report suspended within %d ms; "
                        "falling back to a blind idle\n", gms);
        }
    }

    if (gms <= 0) return 0;
    ts.tv_sec = gms / 1000;
    ts.tv_nsec = (long)(gms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
    return 0;
}

/* How many programmed output channels one real channel costs.
 *
 * FOUR is the default: the narrow writer delivers the first eight output channels of
 * every thirty-two. TWO is the wide writer (DPU PROC_PRECISION widened to int32), which
 * delivers the first eight of every sixteen and so halves the programmed oc — worth
 * double the N tile and half the submits. `ROCKET_RK3576_I32_OC_MULT=2` selects it.
 *
 * IT IS OPT-IN BECAUSE IT INTERMITTENTLY EMITS ZEROS. On about one run in ten of the
 * shape that shows it most, the wide writer emits a contiguous block of one 32-channel
 * super-group's stream with zero data while the rest of the surface is exact.
 *
 * IT DOES NOT DROP WRITES, which is what this looked like before the surface was stamped.
 * Against a sentinel written into the output BO beforehand — and verified to reach DDR —
 * the bad atoms come back ZERO rather than holding the sentinel, so the writer reached
 * them and the data was wrong. It is not a readback race either: a second fence and a
 * second read return byte-identical contents. And it is not the poisoning above, whose
 * signature is an empty region, which the power domain cycling clears, and which this is
 * indifferent to — 3 failures in 30 at no idle against 6 in 30 at 800 ms.
 * [HW sweep, H96 MAX M9]
 *
 * IT IS REPAIRED BY REDOING THE TASK, and r76_i32_wide_suspect() is what sees it. The
 * corruption is contiguous in the writer's EMISSION order and in no other, lies inside a
 * single super-group, and ends two emissions short of that group's last; a redo lands the
 * task exactly, with no idle in front of it. Measured over 40 runs of the shape that
 * fails most: 0 failures, 3 runs redoing one task. [HW sweep, H96 MAX M9]
 *
 * The narrow writer is also the negative control for the wide one: a result that differs
 * between the two says one of the two output maps is wrong and takes the arithmetic out
 * of the question. */
static unsigned r76_i32_oc_mult_env(void)
{
    const char *e = getenv("ROCKET_RK3576_I32_OC_MULT");
    if (!e || !*e) return 0u;                       /* unset: the planner chooses */
    return strtol(e, NULL, 0) == 2 ? 2u : 4u;
}

/* Programmed output channel carrying real output channel n. Both writers deliver the
 * low eight of a group; the group is 32 channels wide on the narrow one and 16 on the
 * wide one. */
static unsigned r76_i32_prog_oc(unsigned n, unsigned mult)
{
    return (mult == 4u ? 32u : 16u) * (n / 8u) + (n % 8u);
}
/* What the last int32 call's detector saw, for rocket_rk3576_i32_last_stats(). The
 * entry repairs both of its failure modes silently, so without this the only record of
 * what a shape actually cost is a log line. Thread-local so a multi-fd caller does not
 * read another worker's call; the accessor documents that it is per thread. */
static _Thread_local rocket_rk3576_i32_stats r76_i32_stats;

void rocket_rk3576_i32_last_stats(rocket_rk3576_i32_stats *out)
{
    if (out) *out = r76_i32_stats;
}

/* One task's contraction is bounded by the resident weight slice, which the 4x oc
 * multiplier eats into directly. Slices descend by halves from this. */
#define R76_I32_KS_MAX    4608u
/* The wide writer's surface bound, as `iw * oh_full * oc_prog` — the whole PLANE times
 * the programmed channel count, which is the surface in half-bytes, so this is 8 KiB per
 * row task. Past it the writer emits zeros and the task comes back part right.
 *
 * IT IS THE PLANE AND NOT THE ROWS, and reading it as rows is a silent wrong answer
 * rather than a lost bound. A plan's row split can only shorten `oh_full`; it cannot
 * narrow `iw`, and the plane chooser takes the WIDEST divisor of M it can — so on a
 * shape whose plane comes out `Mx1` a rows-only cap is satisfied by every task while the
 * surface is M times over. Measured with the writer forced on eleven such shapes: wrong
 * on ten of them at the planner's own plane, exact on all eleven at iw=1, where the two
 * readings coincide. [HW sweep, H96 MAX M9] */
#define R76_I32_WIDE_SURF  4095u


/* The bound in force, so a probe can RAISE it and find where the map actually breaks.
 * Honouring the bound is what keeps the wide writer correct, so the override exists to
 * measure the boundary rather than to be set in anger: past it a tile comes back part
 * right and silently. ROCKET_RK3576_I32_WIDE_SURF. */
static unsigned r76_i32_wide_surf(void)
{
    const char *e = getenv("ROCKET_RK3576_I32_WIDE_SURF");
    unsigned v = (e && *e) ? (unsigned)strtoul(e, NULL, 0) : 0u;
    return v ? v : R76_I32_WIDE_SURF;
}

/* Rows one wide row task may cover at this plane width and programmed channel count, or
 * 0 if the plane is already too wide for a single row — in which case the wide writer
 * cannot run this tile at all, because there is no shorter task to split into. */
static unsigned r76_i32_wide_rows(unsigned iw, unsigned oc_prog)
{
    unsigned per_row = oc_prog * (iw ? iw : 1u);
    return per_row ? r76_i32_wide_surf() / per_row : 0u;
}

/* ROCKET_RK3576_I32_DIAG locates the wide writer's dropped atoms in the coordinates of
 * its OWN map rather than in (m, n).
 *
 * The detector needs no expected values. Every atom of a wide surface is a delivered
 * atom — the writer emits `oc/16 * 2` atoms per pixel and the map covers exactly those —
 * and the scatter puts a real output channel in every delivered slot, so with operands
 * that are not degenerate each 16-byte atom holds four accumulators that are all zero
 * only by coincidence. An all-zero atom is therefore a write that did not happen, and
 * the scan below names it as (super-group, run, offset-in-run, lane group, pixel).
 *
 * At level 2 each task's surface is also given a guard band of its own, so an atom that
 * was written PAST its task's surface shows up as a nonzero guard rather than as
 * nothing at all — that separates a dropped write from an address-generation slip. */
static int r76_i32_diag(void)
{
    const char *e = getenv("ROCKET_RK3576_I32_DIAG");
    return (e && *e) ? (int)strtol(e, NULL, 0) : 0;
}
#define R76_DIAG_TASK_GUARD 256u

/* The surface is stamped with a sentinel before the tasks run, so "this atom was never
 * written" is a property of the surface rather than an inference from its value.
 *
 * That is what makes the wide writer's dropped atoms repairable. A fresh BO arrives
 * zeroed, and zero is also a legitimate accumulator, so a zeroed surface can only
 * answer "did this task write anything at all" — enough to catch a poisoned submit,
 * blind to a task that wrote all but thirteen of its atoms. Against a sentinel the
 * question becomes per atom and the answer is exact, so the retry that already covers
 * the poisoning covers the drop as well.
 *
 * Stamping it is safe here for the same reason the operand buffers are: the fill is
 * bracketed by PREP_BO and FINI_BO, so the lines are written back before the submit and
 * none are left dirty to race the DPU's DMA. Zeroing an output BO with a bare memset —
 * no FINI_BO — is the trap, and it is a different thing.
 *
 * The value is chosen to be an implausible accumulator rather than an impossible one. A
 * real accumulator that happens to equal it costs one wasted submit and stays correct. */
int rocket_rk3576_sentinel_on(void)
{
    const char *e = getenv("ROCKET_RK3576_I32_SENTINEL");
    /* On by default on the wide path: it is what makes that path's dropped atoms
     * visible, and so what lets the retry heal them. =0 turns it off. */
    return (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 1;
}

/* The completion half of the write guard; the contract is in rocket_rk3576_internal.h
 * and the counters' in rocket_npu.h. */
static _Atomic uint64_t g_r76_retired_unwritten, g_r76_retired_written;

static uint64_t r76_backstop_ns(void)
{
    static _Atomic int64_t cached = -1;
    int64_t v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("ROCKET_RK3576_BACKSTOP_US");
        long long us = (e && *e) ? strtoll(e, NULL, 0) : 0;
        if (us <= 0) {
            int ms = rocket_hw_current()->backstop_ms;
            us = (long long)(ms > 0 ? ms : 125) * 1000;
        }
        v = (int64_t)us * 1000;
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return (uint64_t)v;
}

uint64_t rocket_rk3576_job_clock(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

uint64_t rocket_rk3576_job_ns(uint64_t t0)
{
    return rocket_rk3576_job_clock() - t0;
}

int rocket_rk3576_past_backstop(uint64_t elapsed_ns)
{
    return elapsed_ns >= r76_backstop_ns();
}

int rocket_rk3576_score_retired(const char *entry, uint64_t elapsed_ns, int wrote)
{
    if (!rocket_rk3576_past_backstop(elapsed_ns)) return 0;
    atomic_fetch_add_explicit(wrote ? &g_r76_retired_written : &g_r76_retired_unwritten,
                              1, memory_order_relaxed);
    ROCKET_LOGW("%s: a job took %.3f ms from its submit to its fence, at or past the "
                "driver's %.3f ms backstop, so the driver retired it rather than it "
                "completing; its surface reads %s. Redoing it.\n",
                entry, (double)elapsed_ns / 1e6, (double)r76_backstop_ns() / 1e6,
                wrote ? "WRITTEN, which a write check alone would have passed"
                      : "unwritten");
    return 1;
}

void rocket_rk3576_retired_counts(uint64_t *unwritten, uint64_t *written)
{
    if (unwritten)
        *unwritten = atomic_load_explicit(&g_r76_retired_unwritten, memory_order_relaxed);
    if (written)
        *written = atomic_load_explicit(&g_r76_retired_written, memory_order_relaxed);
}

void rocket_rk3576_retired_counts_reset(void)
{
    atomic_store_explicit(&g_r76_retired_unwritten, 0, memory_order_relaxed);
    atomic_store_explicit(&g_r76_retired_written, 0, memory_order_relaxed);
}

/* One task's surface, scanned for atoms the writer never wrote. Returns how many. */
static unsigned r76_i32_diag_task(const unsigned char *surf, unsigned A,
                                  unsigned oc_prog, unsigned atoms_per_px,
                                  unsigned k0, unsigned n0, unsigned t,
                                  unsigned char want, const char *label, int quiet)
{
    unsigned a, natoms = atoms_per_px * A, nzero = 0;
    long prev = -2;

    for (a = 0; a < natoms; a++) {
        const unsigned char *ap = surf + (size_t)a * C2;
        unsigned i, sg, rem, r, off, s, p, j, L, c0;
        for (i = 0; i < C2; i++) if (ap[i] != want) break;
        if (i != C2) continue;
        if (quiet) { nzero++; continue; }
        sg  = A ? a / (4u * A) : 0u;
        rem = a - sg * 4u * A;
        r   = rem / A;
        off = rem % A;
        L   = r % 2u;
        s   = (r / 2u) * A + off;
        p   = s / 2u;
        j   = s % 2u;
        c0  = 32u * sg + 16u * j + 4u * L;
        ROCKET_LOGI("rk3576 i32 diag: k0=%u n0=%u task=%u %s atom %u "
                    "byte %zu (mod64 %zu) A=%u oc=%u | super-group %u run %u "
                    "off-in-run %u lane-group %u | s=%u pixel %u block %u "
                    "channels %u..%u%s\n",
                    k0, n0, t, label, a, (size_t)a * C2, ((size_t)a * C2) % 64u, A, oc_prog,
                    sg, r, off, L, s, p, j, c0, c0 + 3u,
                    prev == (long)a - 1 ? " (contiguous with the last)" : "");
        prev = (long)a;
        nzero++;
    }
    return nzero;
}

/* One run of consecutive zero emissions, in the coordinates the signature is stated in.
 * `tail=at-2A-2` is the predicted end; anything else is a counter-example to it. */
static void r76_i32_run_log(unsigned sg, unsigned e0, unsigned e1, unsigned A)
{
    ROCKET_LOGD("rk3576 i32 wide: zero run super-group %u emissions %u..%u "
                "(s %u..%u, lane groups %u..%u) len %u of %u | A=%u 2A-2=%u "
                "tail=%s\n",
                sg, e0, e1, e0 / 2u, e1 / 2u, e0 % 2u, e1 % 2u,
                e1 - e0 + 1u, 4u * A, A, 2u * A - 2u,
                e1 == 4u * A - 3u ? "at-2A-2" : "elsewhere");
}

/* The attempt from which a task whose only defect is zero atoms is settled EXACTLY, by
 * asking the operands rather than the part.
 *
 * A dropped atom and a legitimately zero accumulator are the same bytes, and no property
 * of the surface separates them: repetition does not, because the drop is correlated
 * with the memory system and repeats often enough to leak a wrong answer — measured, its
 * rate goes from 8.5% of row tasks to 42.5% under host streaming load while the
 * never-emitted count barely moves. What does separate them is the arithmetic, and this
 * entry has both operands. So the first attempt is simply redone (the drop usually does
 * not survive one redo, and a redo is far cheaper than a dot product), and from the
 * second the zero atoms are checked against the CPU. [HW sweep, H96 MAX M9] */
#define R76_I32_CPU_CHECK_FROM 1u

/* The wide writer's corruption signature, in the coordinates of its own emission order.
 *
 * WHAT THE DEFECT IS. On a minority of runs the wide writer emits a contiguous block of
 * one 32-channel super-group's stream with ZERO data. It is not a dropped write: on a
 * surface stamped with a sentinel beforehand — and the stamp is verified to reach DDR —
 * those atoms come back zero rather than holding the sentinel, so the writer reached
 * them and the data was wrong. Nor is it a readback race, since a second fence and a
 * second read return byte-identical contents; nor the poisoning, whose signature is an
 * empty region and which is cleared by the power domain cycling, where this is
 * indifferent to the idle ahead of it (3 failures in 30 at no idle, 6 in 30 at 800 ms).
 * [HW sweep, H96 MAX M9]
 *
 * WHAT IT LOOKS LIKE. The writer emits two atoms per stream position s = 2*pixel +
 * block, one per lane group, so its emission order is (s, L) ascending. The corrupt
 * block is contiguous in THAT order and in no other — in address order it appears as two
 * separate holes — it lies inside a single super-group, and it always ends at s = 2A-2,
 * two emissions short of that super-group's last. Its start varies run to run, from two
 * emissions to most of the group. Measured over 40 runs of one shape, every failure had
 * that shape. [HW sweep, H96 MAX M9]
 *
 * WHAT THIS FUNCTION DOES with it. Two adjacent emissions coming back zero is the
 * smallest corruption seen and is implausible as arithmetic — eight output channels at
 * one pixel, all exactly zero — so it is the signal, and the task is redone. A tile
 * whose data really does hold such a pair costs a wasted submit and stays correct.
 * Returns the number of atoms in runs of two or more consecutive zero emissions.
 *
 * Each run's EXTENT is logged at debug level, in emission coordinates, against the
 * signature's prediction that a run ends at s = 2A-2 (emission 4A-3). That is what a
 * tighter detector would key on, so it is what has to be confirmed across shapes before
 * one is written: run ROCKET_LOG_LEVEL=debug and read the "tail" field. */
static unsigned r76_i32_wide_suspect(const unsigned char *surf, unsigned A,
                                     unsigned oc_prog)
{
    unsigned sg, nsg = (oc_prog + 31u) / 32u, total = 0;

    for (sg = 0; sg < nsg; sg++) {
        unsigned e, run = 0;
        for (e = 0; e < 4u * A; e++) {
            unsigned s = e / 2u, L = e % 2u;
            unsigned atom = 4u * A * sg + A * (2u * (s / A) + L) + s % A;
            const unsigned char *ap = surf + (size_t)atom * C2;
            unsigned i;
            for (i = 0; i < C2; i++) if (ap[i]) break;
            if (i == C2) { run++; continue; }
            if (run >= 2u) { r76_i32_run_log(sg, e - run, e - 1u, A); total += run; }
            run = 0;
        }
        if (run >= 2u) { r76_i32_run_log(sg, 4u * A - run, 4u * A - 1u, A); total += run; }
    }
    return total;
}

/* Are this task's all-zero atoms the arithmetic?
 *
 * Walks the same atoms the detector walks, and for every one that came back entirely
 * zero computes the four accumulators it should hold — over THIS K slice only, since a
 * task's surface carries one slice's partial and the host sums the slices. Returns 0 as
 * soon as one of them should have been non-zero, which is a dropped atom; 1 if every
 * zero atom is genuinely zero, which is data and the surface is correct.
 *
 * Both writers put four consecutive accumulators in an atom's 16 bytes, and on both of
 * them every atom of the surface is delivered — what differs is which four:
 *   narrow, the plain int32 cube — channel block `blk` holds real channels 4*blk..+3 at
 *   pixel `px`, the blocks laid `surf_elems` pixels apart;
 *   wide — the stream map, whose atom decomposition gives a programmed channel c0 and a
 *   pixel, with c0 % 16 always below 8 so all four are delivered.
 */
static int r76_i32_zeros_are_data(const unsigned char *surf, unsigned mult,
                                  unsigned iw, unsigned oy0, unsigned oh,
                                  unsigned atoms_per_px, unsigned surf_elems,
                                  unsigned n0, unsigned tile_n,
                                  unsigned k0, unsigned kslice,
                                  int K, const int8_t *A, const int8_t *B)
{
    unsigned px0 = oy0 * iw, npx = oh * iw;
    unsigned a, natoms;

    /* One accumulator, over this slice's K range. */
#define R76_ACC_NONZERO(m_, n_)                                                        \
    ({ int32_t acc_ = 0; unsigned k_;                                                  \
       for (k_ = 0; k_ < kslice; k_++)                                                 \
           acc_ += (int32_t)A[(size_t)(m_) * K + k0 + k_] *                            \
                   (int32_t)B[(size_t)(n_) * K + k0 + k_];                             \
       acc_ != 0; })

    if (mult == 4u) {
        unsigned blk, px;
        for (blk = 0; blk < atoms_per_px; blk++)
            for (px = 0; px < npx; px++) {
                const unsigned char *ap =
                    surf + ((size_t)blk * surf_elems + px0 + px) * C2;
                unsigned i, jj;
                for (i = 0; i < C2; i++) if (ap[i]) break;
                if (i != C2) continue;                      /* not a zero atom */
                for (jj = 0; jj < 4u; jj++) {
                    unsigned n = 4u * blk + jj;
                    if (n >= tile_n) break;                 /* past the tile: padding */
                    if (R76_ACC_NONZERO(px0 + px, n0 + n)) return 0;
                }
            }
        return 1;
    }

    natoms = atoms_per_px * rocket_rk3576_out_surf_elems(iw, oh, 0);
    for (a = 0; a < natoms; a++) {
        const unsigned char *ap = surf + (size_t)a * C2;
        unsigned i, sg, rem, r, off, s, p, j, L, c0, jj, Aw = iw * oh;
        for (i = 0; i < C2; i++) if (ap[i]) break;
        if (i != C2) continue;
        if (!Aw) return 0;
        sg  = a / (4u * Aw);
        rem = a - sg * 4u * Aw;
        r   = rem / Aw;
        off = rem % Aw;
        L   = r % 2u;
        s   = (r / 2u) * Aw + off;
        p   = s / 2u;
        j   = s % 2u;
        c0  = 32u * sg + 16u * j + 4u * L;
        for (jj = 0; jj < 4u; jj++) {
            unsigned c = c0 + jj, n = 8u * (c / 16u) + (c % 16u);
            if (n >= tile_n) break;
            if (R76_ACC_NONZERO(px0 + p, n0 + n)) return 0;
        }
    }
    return 1;
#undef R76_ACC_NONZERO
}

/* The largest K slice this shape can run, or 0 if none can. The plane is chosen for
 * the LARGEST slice and then held fixed across every slice, because the output surface
 * the partials accumulate into has to be the same one. */
static unsigned r76_i32_plan(unsigned M, unsigned K, unsigned N, unsigned mult,
                             unsigned *iw_out, unsigned *ih_out, unsigned *nt_out)
{
    unsigned ks;
    unsigned cap = R76_I32_KS_MAX;
    const char *e = getenv("ROCKET_RK3576_MM_KS");

    if (e && *e) {
        long v = strtol(e, NULL, 0);
        if (v >= 32) cap = (unsigned)(v / 32 * 32);
    }
    for (ks = K < cap ? K : cap; ks >= 32u; ks = (ks / 2u / 32u) * 32u) {
        unsigned iw, ih, nt;
        r76_mm_plane(M, ks, &iw, &ih);
        nt = r76_mm_fit_nt_mult(iw, ks, N, mult, NULL);
        if (nt) {
            *iw_out = iw; *ih_out = ih; *nt_out = nt;
            return ks;
        }
        if (ks <= 32u) break;
    }
    return 0;
}

/* How many submits a writer costs on this shape, from the plan alone and with nothing
 * allocated on the device. Mirrors the tile loop below: K slices x N tiles x row tasks,
 * with the wide writer's height bound splitting those tasks further. 0 = it cannot run
 * this shape at all. */
static unsigned r76_i32_submit_count(unsigned M, unsigned K, unsigned N, unsigned mult)
{
    unsigned iw = 0, ih = 0, nt = 0, ks, n0, per_slice = 0;
    rocket_rk3576_row_task *plan;
    unsigned max_tasks;

    ks = r76_i32_plan(M, K, N, mult, &iw, &ih, &nt);
    if (!ks) return 0;
    max_tasks = ih + 1u;
    plan = calloc(max_tasks, sizeof *plan);
    if (!plan) return 0;

    for (n0 = 0; n0 < N; n0 += nt) {
        unsigned tile_n = N - n0 < nt ? N - n0 : nt;
        unsigned oc_prog = rocket_rk3576_pad_oc(mult * tile_n);
        unsigned ntask = 1;
        conv_params_t q = {0};

        q.ic = (uint16_t)((K < ks ? K : ks));
        q.ih = (uint16_t)ih; q.iw = (uint16_t)iw;
        q.oc = (uint16_t)oc_prog; q.oh = (uint16_t)ih; q.ow = (uint16_t)iw;
        q.kh = 1; q.kw = 1; q.stride_y = 1; q.stride_x = 1;
        q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
        if (rocket_rk3576_plan_rows(&q, 0, plan, max_tasks, &ntask) < 0) {
            free(plan);
            return 0;
        }
        if (mult != 4u) {
            /* A plane too wide for even a one-row task cannot be split into one, so the
             * wide writer simply cannot run this shape — say so by costing it nothing,
             * which is what makes the chooser fall back to the narrow writer instead of
             * running a task the bound does not cover. */
            unsigned cap = r76_i32_wide_rows(iw, oc_prog), src, split = 0;
            if (!cap) { free(plan); return 0; }
            for (src = 0; src < ntask; src++)
                split += (plan[src].oh + cap - 1u) / cap;
            ntask = split;
        }
        per_slice += ntask;
    }
    free(plan);
    return per_slice * ((K + ks - 1u) / ks);
}

/* WHICH WRITER, chosen per shape rather than globally.
 *
 * The wide writer's whole benefit is submits: at PROC_PRECISION int32 the DPU's byte
 * budget is two atoms per (16-channel block, pixel) instead of one, which doubles the N
 * tile the weight-cube scatter can drive. Where that halves the tile count it is worth
 * 1.2-1.4x; where the surface bound (iw * oh_full * oc_prog < 4096) forces the row tasks
 * back apart it is worth nothing and costs an extra submit, and it also scans every atom
 * of the surface on readback where the narrow writer stops at the first non-blank byte.
 * Both effects are visible in the submit count, and that count is computable before
 * anything is submitted — so the planner counts both and takes the cheaper.
 *
 * Ties go to the narrow writer, which needs a quarter of the wide one's row tasks at a
 * wide output tile. Neither is the one without the dropped-atom defect — both drop, and
 * both are checked per atom against a sentinel and redone.
 *
 * ONCE THE SURFACE BOUND IS STATED OVER THE PLANE, THE WIDE WRITER STOPS WINNING. The
 * bound costs it exactly the submits its wider N tile was reached for, and it costs them
 * on the N-heavy shapes where the tile would have paid: swept over 315 natural shapes,
 * M 4-256, K 64-2048, N 32-2048, the chooser took the narrow writer on every one. It is
 * kept because it is the instrument that decoded the 32-bit writer's map and because a
 * shape outside that sweep may still reach it, not because the default path uses it.
 * [HW sweep, H96 MAX M9]
 * ROCKET_RK3576_I32_OC_MULT=2 or =4 forces either. [HW sweep, H96 MAX M9] */
static unsigned r76_i32_oc_mult(unsigned M, unsigned K, unsigned N)
{
    unsigned forced = r76_i32_oc_mult_env(), narrow, wide;

    if (forced) return forced;
    narrow = r76_i32_submit_count(M, K, N, 4u);
    wide   = r76_i32_submit_count(M, K, N, 2u);
    if (!wide) return 4u;
    if (!narrow || wide < narrow) {
        ROCKET_LOGD("rk3576 matmul i32: wide writer chosen, %u submits against %u\n",
                    wide, narrow);
        return 2u;
    }
    return 4u;
}

/* THE DENSE WRITER, the int32 entry's default. ROCKET_RK3576_I32_DENSE=0, or forcing a
 * writer with ROCKET_RK3576_I32_OC_MULT, takes the narrow and wide writers below instead.
 *
 * gen_conv2d_int8_rk3576_i32out_dense() writes every accumulator of a task as a raw int32
 * and poisons nothing, where the narrow and wide writers deliver a quarter and a half of
 * the programmed channels and poison the next submit. So this path programs the real
 * output channels (rounded to 32), submits its tasks back to back with no idle, and leaves
 * the part clean on exit. What it keeps from the writers' path is the guard: a sentinel
 * stamp, a per-atom scan of each task's surface, a retired job redone, and a power cycle
 * before a redo only when an atom was never written, which a FOREIGN job's poisoning still
 * causes. A task that still has holes after every attempt is refused, not returned. Each row
 * task is its own 1x1 convolution with its own surface, laid end to end, because the
 * stage's 0x40B8 at a window of a larger surface is not measured. The bias goes on here,
 * on the host: the stage adds none. */
static int r76_i32_dense_on(void)
{
    const char *e = getenv("ROCKET_RK3576_I32_DENSE");
    return !(e && *e == '0');
}

static int r76_i32_dense(int fd, int M, int K, int N, const int8_t *A, const int8_t *B,
                         const int32_t *bias, int32_t *C)
{
    struct r76_mm_bos b = {0};
    uint64_t *ops = NULL;
    rocket_rk3576_row_task *plan = NULL;
    size_t *task_off = NULL, in_slot = 0;
    unsigned iw = 0, ih = 0, nt = 0, ks, k0, n0, max_tasks, nslices, submits = 0;
    unsigned char blank = rocket_rk3576_sentinel_on() ? (unsigned char)R76_SENTINEL_BYTE : 0;
    int rc = ROCKET_E_SHAPE;

    ks = r76_i32_plan((unsigned)M, (unsigned)K, (unsigned)N, 1u, &iw, &ih, &nt);
    ROCKET_LOGI("rk3576 matmul i32: M=%d K=%d N=%d -> plane %ux%u, K slice %u, N tile %u "
                "(dense writer)\n", M, K, N, iw, ih, ks, nt);
    if (!ks) {
        ROCKET_LOGE("rk3576 matmul i32: M=%d K=%d N=%d does not fit one task even at a "
                    "32-channel output tile and a 32-deep K slice\n", M, K, N);
        return ROCKET_E_SHAPE;
    }
    max_tasks = ih + 1u;
    ops      = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    plan     = calloc(max_tasks, sizeof *plan);
    task_off = calloc(max_tasks + 1u, sizeof *task_off);
    if (!ops || !plan || !task_off) { rc = ROCKET_E_NOMEM; goto done; }
    memset(C, 0, (size_t)M * N * sizeof *C);

    /* Every K slice's feature cube is packed up front at its own offset, as on the
     * writers' path: a slice rewritten in place between submits is read stale. */
    nslices = ((unsigned)K + ks - 1u) / ks;
    in_slot = ((size_t)((ks + C2 - 1) / C2) * ih * iw * C2 + 63u) & ~(size_t)63u;
    if (rocket_bo_alloc32(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0 ||
        rocket_bo_alloc32(fd, in_slot * nslices, &b.in) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }
    rocket_bo_prep(fd, &b.in, 1, 0);
    memset(b.in.ptr, 0, in_slot * nslices);
    for (k0 = 0; k0 < (unsigned)K; k0 += ks) {
        unsigned kslice = (unsigned)K - k0 < ks ? (unsigned)K - k0 : ks, k;
        int8_t *slot = (int8_t *)b.in.ptr + (size_t)(k0 / ks) * in_slot;
        int m;
        for (m = 0; m < M; m++)
            for (k = 0; k < kslice; k++)
                slot[feature_data((int)kslice, (int)ih, (int)iw, C2, (int)k + 1,
                                  m / (int)iw + 1, m % (int)iw + 1)] = A[(size_t)m * K + k0 + k];
    }
    rocket_bo_fini(fd, &b.in);

    for (k0 = 0; k0 < (unsigned)K; k0 += ks) {
        unsigned kslice = (unsigned)K - k0 < ks ? (unsigned)K - k0 : ks;
        for (n0 = 0; n0 < (unsigned)N; n0 += nt) {
            unsigned tile_n = (unsigned)N - n0 < nt ? (unsigned)N - n0 : nt;
            unsigned oc_prog = rocket_rk3576_pad_oc(tile_n), ntask = 1, t, n, k;
            size_t w_bytes = (size_t)(oc_prog / 32) * ((kslice + 31) / 32) * 32 * 32;
            size_t coeff_bytes = rocket_rk3576_coeff_bytes(oc_prog), surf_bytes, guard_off;
            conv_params_t p = {0};
            uint32_t in_h[4], out_h[1];
            {
                conv_params_t q = {0};
                q.ic = (uint16_t)kslice; q.ih = (uint16_t)ih; q.iw = (uint16_t)iw;
                q.oc = (uint16_t)oc_prog; q.oh = (uint16_t)ih; q.ow = (uint16_t)iw;
                q.kh = 1; q.kw = 1; q.stride_y = 1; q.stride_x = 1;
                q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
                if (rocket_rk3576_plan_rows(&q, 0, plan, max_tasks, &ntask) < 0) {
                    ROCKET_LOGE("rk3576 matmul i32: no row plan for M=%d K slice %u N tile %u "
                                "(plane %ux%u)\n", M, kslice, tile_n, iw, ih);
                    rc = ROCKET_E_SHAPE; goto done;
                }
            }
            /* One surface per row task, end to end, then a band that stays zero: nothing
             * writes this BO from the CPU but the stamp, and the band is past the stamp. */
            surf_bytes = 0;
            for (t = 0; t < ntask; t++) {
                task_off[t] = surf_bytes;
                surf_bytes += (size_t)oc_prog * iw * plan[t].oh * sizeof(int32_t);
            }
            guard_off = (surf_bytes + 63u) & ~(size_t)63u;

            if (b.w.ptr)     rocket_bo_free(fd, &b.w);
            if (b.coeff.ptr) rocket_bo_free(fd, &b.coeff);
            if (b.out.ptr)   rocket_bo_free(fd, &b.out);
            memset(&b.w, 0, sizeof b.w);
            memset(&b.coeff, 0, sizeof b.coeff);
            memset(&b.out, 0, sizeof b.out);
            if (rocket_bo_alloc32(fd, w_bytes, &b.w) < 0 ||
                rocket_bo_alloc32(fd, coeff_bytes, &b.coeff) < 0 ||
                rocket_bo_alloc32(fd, guard_off + 4096u, &b.out) < 0) {
                rc = ROCKET_E_NOMEM; goto done;
            }
            rocket_bo_prep(fd, &b.w, 1, 0);
            memset(b.w.ptr, 0, w_bytes);
            for (n = 0; n < tile_n; n++)
                for (k = 0; k < kslice; k++)
                    ((int8_t *)b.w.ptr)[weight_conv_int8((int)oc_prog, (int)kslice, 1, 1,
                                                         (int)n + 1, (int)k + 1, 1, 1)] =
                        B[(size_t)(n0 + n) * K + k0 + k];
            rocket_bo_fini(fd, &b.w);
            rocket_bo_prep(fd, &b.coeff, 1, 0);
            rocket_rk3576_pack_coeff(b.coeff.ptr, coeff_bytes, NULL, oc_prog);
            rocket_bo_fini(fd, &b.coeff);
            if (blank) {
                rocket_bo_prep(fd, &b.out, 1, 0);
                memset(b.out.ptr, blank, surf_bytes);
                rocket_bo_fini(fd, &b.out);
            }

            p.ic = (uint16_t)kslice; p.oc = (uint16_t)oc_prog;
            p.iw = (uint16_t)iw; p.ow = (uint16_t)iw;
            p.kh = 1; p.kw = 1; p.stride_y = 1; p.stride_x = 1;
            p.ih_full = (uint16_t)ih;
            p.int8_out = 1;
            p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;
            p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
            p.tasks = ops;
            p.weights_dma = b.w.dma_address;
            p.bias_dma    = b.coeff.dma_address;
            in_h[0] = b.in.handle; in_h[1] = b.w.handle;
            in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
            out_h[0] = b.out.handle;

            for (t = 0; t < ntask; t++) {
                unsigned attempt;
                int task_ok = 0;
                for (attempt = 0; attempt < R76_I32_TASK_ATTEMPTS && !task_ok; attempt++) {
                    conv_params_t q = p;
                    uint64_t tjob, jns;
                    unsigned unwritten = 0, a, natoms;
                    const unsigned char *sp;
                    q.ih = plan[t].oh; q.oh = plan[t].oh; q.oh_full = plan[t].oh;
                    q.pad_top = 0;
                    q.input_dma  = b.in.dma_address + (uint32_t)((size_t)(k0 / ks) * in_slot) +
                                   (uint32_t)plan[t].oy0 * iw * C2;
                    q.output_dma = b.out.dma_address + (uint32_t)task_off[t];
                    if (gen_conv2d_int8_rk3576_i32out_dense(&q) != 0) {
                        rc = ROCKET_E_SHAPE; goto done;
                    }
                    rocket_bo_prep(fd, &b.rc, 1, 0);
                    memcpy(b.rc.ptr, ops, q.task_count * sizeof(uint64_t));
                    rocket_bo_fini(fd, &b.rc);
                    tjob = rocket_rk3576_job_clock();
                    submits++;
                    if (rocket_submit_matmul(fd, &b.rc, q.task_count, in_h, 4, out_h, 1,
                                             2000) != 0) {
                        rc = ROCKET_E_DEVICE; goto done;
                    }
                    if (rocket_bo_prep(fd, &b.out, 0, 2000000000ull) < 0) {
                        rc = ROCKET_E_DEVICE; goto done;
                    }
                    jns = rocket_rk3576_job_ns(tjob);
                    sp = (const unsigned char *)b.out.ptr + task_off[t];
                    natoms = oc_prog * iw * plan[t].oh / 4u;
                    for (a = 0; blank && a < natoms; a++) {
                        const unsigned char *ap = sp + (size_t)a * C2;
                        unsigned i;
                        for (i = 0; i < C2 && ap[i] == blank; i++) ;
                        if (i == C2) unwritten++;
                    }
                    rocket_bo_fini(fd, &b.out);
                    /* A retired job is redone whatever it wrote. A poisoned part, which only
                     * a foreign job can leave now, reads as a retired job that wrote nothing,
                     * and only the power cycle clears it, so every redo is idled. */
                    if (!rocket_rk3576_score_retired("rk3576 matmul i32", jns, !unwritten) &&
                        !unwritten) { task_ok = 1; break; }
                    r76_i32_stats.redo_empty++;
                    r76_i32_stats.atoms_empty += unwritten;
                    ROCKET_LOGD("rk3576 matmul i32: k0=%u n0=%u row task %u — %u atoms never "
                                "written or the job retired; cycling the power domain and "
                                "redoing it\n", k0, n0, t, unwritten);
                    rocket_rk3576_power_idle();
                }
                if (!task_ok) {
                    r76_i32_stats.refused++;
                    ROCKET_LOGE("rk3576 matmul i32: k0=%u n0=%u row task %u still has "
                                "unwritten atoms or was retired after %d attempts — refusing "
                                "to return a partial surface\n", k0, n0, t,
                                R76_I32_TASK_ATTEMPTS);
                    rc = ROCKET_E_DEVICE; goto done;
                }
                r76_i32_stats.tasks++;
            }
            r76_i32_stats.submits = submits;

            rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
            {
                const unsigned char *guard = (const unsigned char *)b.out.ptr + guard_off;
                const int32_t *o32 = (const int32_t *)b.out.ptr;
                unsigned g;
                int m;
                for (g = 0; g < 4096u; g++)
                    if (guard[g] != 0) {
                        ROCKET_LOGE("rk3576 matmul i32: the dense writer reached past its "
                                    "surfaces (%zu bytes) at programmed oc=%u\n", surf_bytes,
                                    oc_prog);
                        rocket_bo_fini(fd, &b.out);
                        rc = ROCKET_E_DEVICE; goto done;
                    }
                t = 0;
                for (m = 0; m < M; m++) {
                    unsigned y = (unsigned)m / iw, px, A_t;
                    while (t + 1u < ntask && y >= (unsigned)plan[t + 1].oy0) t++;
                    while (t && y < (unsigned)plan[t].oy0) t--;
                    px  = (unsigned)m - (unsigned)plan[t].oy0 * iw;
                    A_t = iw * plan[t].oh;
                    for (n = 0; n < tile_n; n++)
                        C[(size_t)m * N + n0 + n] +=
                            o32[task_off[t] / 4u +
                                (size_t)rocket_rk3576_i32_dense_word(A_t, n, px)];
                }
            }
            rocket_bo_fini(fd, &b.out);
        }
    }
    if (bias) {
        int m, n;
        for (m = 0; m < M; m++)
            for (n = 0; n < N; n++) C[(size_t)m * N + n] += bias[n];
    }
    ROCKET_LOGI("rk3576 matmul i32: M=%d K=%d N=%d done in %u submits (dense writer)\n",
                M, K, N, submits);
    rc = ROCKET_OK;
done:
    free(ops); free(plan); free(task_off);
    r76_mm_free(fd, &b);
    return rc;
}

int rocket_matmul_int8_rk3576_i32(int fd, int M, int K, int N,
                                  const int8_t *A, const int8_t *B,
                                  const int32_t *bias, int32_t *C)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    struct r76_mm_bos b = {0};
    uint64_t *ops = NULL;
    int8_t *stage = NULL;
    rocket_rk3576_row_task *plan = NULL, *wplan = NULL;
    size_t *task_off = NULL;
    unsigned iw = 0, ih = 0, nt = 0, ks, k0, n0, surf_elems, max_tasks;
    unsigned mult;
    int diag = r76_i32_diag();
    unsigned diag_dropped = 0, heals = 0;
    unsigned char blank = 0;
    size_t o_surf_max, guard_off, in_slot = 0;
    unsigned nslices = 1, submits = 0;
    int rc = ROCKET_E_SHAPE;

    memset(&r76_i32_stats, 0, sizeof r76_i32_stats);
    if (fd < 0 || !A || !B || !C || M <= 0 || K <= 0 || N <= 0) return ROCKET_E_SHAPE;
    if (strcmp(hw->name, "rk3576") != 0) {
        ROCKET_LOGE("rocket_matmul_int8_rk3576_i32: this is the RK3576 encoding and the "
                    "active profile is %s\n", hw->name);
        return ROCKET_E_UNSUPPORTED;
    }
    if (K % 32 || N % 32) {
        ROCKET_LOGE("rk3576 matmul i32: K=%d N=%d — both must be multiples of 32 (the "
                    "int8 weight cube groups each channel axis by 32)\n", K, N);
        return ROCKET_E_SHAPE;
    }
    if (r76_i32_dense_on() && !r76_i32_oc_mult_env())
        return r76_i32_dense(fd, M, K, N, A, B, bias, C);

    mult = r76_i32_oc_mult((unsigned)M, (unsigned)K, (unsigned)N);
    ks = r76_i32_plan((unsigned)M, (unsigned)K, (unsigned)N, mult, &iw, &ih, &nt);
    ROCKET_LOGI("rk3576 matmul i32: M=%d K=%d N=%d -> plane %ux%u, K slice %u "
                "(%u slices, last %u), N tile %u (programmed oc %u, %ux writer)\n",
                M, K, N, iw, ih, ks,
                ks ? ((unsigned)K + ks - 1u) / ks : 0u,
                ks ? ((unsigned)K % ks ? (unsigned)K % ks : ks) : 0u,
                nt, mult * nt, mult);
    if (!ks) {
        ROCKET_LOGE("rk3576 matmul i32: M=%d K=%d N=%d does not fit one task even at a "
                    "32-channel output tile and a 32-deep K slice\n", M, K, N);
        return ROCKET_E_SHAPE;
    }

    /* BOTH WRITERS GET THE SENTINEL. The narrow one drops atoms too — sparsely, and with
     * the same signature: on M=128 K=256 N=2048 six calls in twelve came back with a
     * handful of elements out of 262144 reading ZERO and the rest exact. Against a
     * zeroed BO that is invisible, because the only question a zeroed surface can answer
     * is "did this task write anything at all", and a task that wrote all but eight of
     * its atoms answers yes. [HW sweep, H96 MAX M9]
     *
     * Every atom of a task's surface is a delivered atom on both paths — the narrow
     * writer's cube is `tile_n/4` channel blocks of `surf_elems` pixels and the
     * de-scatter reads every one of them — so "still the sentinel" is a per-atom fact on
     * both, and the retry that already covers the poisoning covers this as well. */
    if (rocket_rk3576_sentinel_on()) blank = (unsigned char)R76_SENTINEL_BYTE;

    surf_elems = rocket_rk3576_out_surf_elems(iw, ih, 0);
    /* One task per output ROW is the worst case the wide writer's height bound can
     * force, so the plan array is sized for it rather than for the row planner alone. */
    max_tasks  = ih + 1u;
    ops   = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    plan  = calloc(max_tasks, sizeof *plan);
    wplan = calloc(max_tasks, sizeof *wplan);
    task_off = calloc(max_tasks, sizeof *task_off);
    stage = calloc((size_t)(ks + C2 - 1) / C2 * ih * iw * C2, 1);
    if (!ops || !plan || !wplan || !task_off || !stage) { rc = ROCKET_E_NOMEM; goto done; }

    memset(C, 0, (size_t)M * N * sizeof *C);

    /* TWO buffer rules here, and both were paid for on the part.
     *
     * The FEATURE cube for every K slice is packed into ONE buffer before any job runs,
     * and each slice's task is pointed at its own offset. Writing one slice at a time
     * between submits — the obvious way — leaves the next job reading what the last one
     * saw, so it recomputes the previous slice byte for byte with no error anywhere.
     *
     * Everything the DPU WRITES, and everything a task reads that changes with the
     * task, is a FRESH buffer per submit. Rewriting one in place between submits does
     * not take effect either, and the failure is the same silent one: the job runs, the
     * timing is normal, nothing faults, and the surface still holds the previous
     * result. The int8 path above allocates per tile for the same reason. Pacing the
     * submits about 200 ms apart also hides it, which is what makes it read as a timing
     * problem rather than a buffer-lifetime one. */
    {
        size_t in_max = (size_t)((ks + C2 - 1) / C2) * ih * iw * C2;
        in_slot = (in_max + 63u) & ~(size_t)63u;
        nslices = ((unsigned)K + ks - 1u) / ks;
        if (rocket_bo_alloc32(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0 ||
            rocket_bo_alloc32(fd, in_slot * nslices, &b.in) < 0) {
            rc = ROCKET_E_NOMEM; goto done;
        }
    }

    rocket_bo_prep(fd, &b.in, 1, 0);
    memset(b.in.ptr, 0, in_slot * nslices);
    for (k0 = 0; k0 < (unsigned)K; k0 += ks) {
        unsigned kslice = (unsigned)K - k0 < ks ? (unsigned)K - k0 : ks;
        int8_t *slot = (int8_t *)b.in.ptr + (size_t)(k0 / ks) * in_slot;
        int m;
        for (m = 0; m < M; m++) {
            unsigned k;
            for (k = 0; k < kslice; k++)
                slot[feature_data((int)kslice, (int)ih, (int)iw, C2,
                                  (int)k + 1, m / (int)iw + 1, m % (int)iw + 1)] =
                    A[(size_t)m * K + k0 + k];
        }
    }
    rocket_bo_fini(fd, &b.in);

    for (k0 = 0; k0 < (unsigned)K; k0 += ks) {
        unsigned kslice = (unsigned)K - k0 < ks ? (unsigned)K - k0 : ks;

        for (n0 = 0; n0 < (unsigned)N; n0 += nt) {
            unsigned tile_n = (unsigned)N - n0 < nt ? (unsigned)N - n0 : nt;
            unsigned oc_prog = rocket_rk3576_pad_oc(mult * tile_n);
            size_t w_bytes = (size_t)((oc_prog + 31) / 32) * ((kslice + 31) / 32) * 32 * 32;
            size_t coeff_bytes = rocket_rk3576_coeff_bytes(oc_prog);
            /* The writer's byte budget is one 16-byte atom per (16-channel block,
             * pixel) on the narrow writer and TWO on the wide one, whatever the output
             * element width is. Either way it comes out at exactly the int32 cube for
             * `tile_n` real channels, because the programmed count absorbs the rest. */
            size_t atoms_per_px = (mult == 4u ? 1u : 2u) * ((oc_prog + C2 - 1) / C2);
            size_t surf_bytes;
            conv_params_t p = {0};
            uint32_t in_h[4], out_h[1];
            unsigned ntask = 1, t, n, k, attempt;
            int wrote = 0;

            /* The row plan comes FIRST here, because on the wide writer it SIZES the
             * output BO: each row task carries its own surface rather than a window
             * into a shared one. */
            {
                conv_params_t q = {0};
                q.ic = (uint16_t)kslice; q.ih = (uint16_t)ih; q.iw = (uint16_t)iw;
                q.oc = (uint16_t)oc_prog; q.oh = (uint16_t)ih; q.ow = (uint16_t)iw;
                q.kh = 1; q.kw = 1; q.stride_y = 1; q.stride_x = 1;
                q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
                if (rocket_rk3576_plan_rows(&q, 0, plan, max_tasks, &ntask) < 0) {
                    ROCKET_LOGE("rk3576 matmul i32: no row plan for M=%d K slice %u "
                                "N tile %u (plane %ux%u, programmed oc %u)\n",
                                M, kslice, tile_n, iw, ih, oc_prog);
                    rc = ROCKET_E_SHAPE; goto done;
                }
            }
            if (mult == 4u) {
                surf_bytes  = atoms_per_px * surf_elems * C2;
                task_off[0] = 0;
            } else {
                /* THE WIDE WRITER'S SURFACE IS BOUNDED AT 8 KiB PER TASK, and this cap is
                 * that bound exactly rather than a conservative stand-in for it.
                 *
                 * State it in the PROGRAMMED channel count: a task is correct while
                 * `A * oc_prog < 4096`, where A = ow*oh_full. Since a wide task writes
                 * `oc_prog/8` atoms a pixel at 16 bytes each, that product is the surface
                 * in half-bytes — the bound is 8192 bytes, or 512 atoms, whichever way it
                 * is read. Swept at K=32 with the cap lifted, the first wrong height is
                 * the smallest oh with `oh*oc_prog >= 4096` at every one of the eight
                 * programmed counts 64/128/192/256/320/384/448/512 — heights 64, 32, 22,
                 * 16, 13, 11, 10 and 8, predicted and observed. [HW sweep, H96 MAX M9]
                 *
                 * The bound reads as a mispredicting rule if it is stated in the REAL
                 * channel count, which is half the programmed one on this path, or if it
                 * is swept at a single mid-range K.
                 *
                 * IT IS THE PLANE, NOT THE ROWS, and the difference is a wrong answer
                 * rather than a lost bound. `A` carries `iw`, a row split can only
                 * shorten `oh_full`, and the plane chooser takes the WIDEST divisor of M
                 * that fits its granule rule — so on a shape whose plane comes out `Mx1`
                 * a cap that divides only by `oc_prog` is met by every task while the
                 * surface is M times over it. Forced onto eleven shapes whose planner
                 * plane is `Mx1`, the writer was wrong on ten; at iw=1, where the two
                 * readings coincide, all eleven are exact and every zero run disappears.
                 * [HW sweep, H96 MAX M9]
                 *
                 * K RELAXES IT, and only ever outward. The same oc 128 that is wrong from
                 * oh 16 at K <= 128 is wrong at only 31, 32 and 40 at K=512 and exact to
                 * oh 40 at K=1024 and K=2048. So 8 KiB is the worst-case floor, reached
                 * once K is small enough, and enforcing it unconditionally is what makes
                 * the path correct for every K.
                 *
                 * WHAT THE FAILURE IS. Every wrong element comes back ZERO — over the
                 * whole map, not one aliases another element's value and not one holds
                 * anything else — so the writer emits empty atoms rather than misplacing
                 * full ones. That is the same signature as the intermittent corruption
                 * r76_i32_wide_suspect() repairs, and it is consistent with a fill/drain
                 * race on a buffer of that size: past 8 KiB the producer laps the drain
                 * unless a long contraction slows it down. Redoing the task does not help
                 * once the surface is over the bound, because the timing repeats.
                 *
                 * A row task here is a standalone 1x1 convolution with its own surface,
                 * so honouring the cap costs nothing but submits: split the plan's
                 * tasks further until each one's height fits. What it costs in submits
                 * is real — an honest cap is what makes the chooser prefer the narrow
                 * writer on the N-heavy shapes the wide one was reached for. */
                unsigned cap = r76_i32_wide_rows(iw, oc_prog);
                unsigned src;
                /* No row count fits, so there is nothing to split into. Refusing is the
                 * only correct answer: the chooser already avoids this shape, and a
                 * caller who forced the wide writer onto it would otherwise get a
                 * silently part-right surface. */
                if (!cap) {
                    ROCKET_LOGE("rk3576 matmul i32: the wide writer's surface bound "
                                "cannot be met at plane width %u with programmed oc %u "
                                "(one row is already %u of %u) — use the narrow "
                                "writer\n", iw, oc_prog, iw * oc_prog,
                                r76_i32_wide_surf());
                    rc = ROCKET_E_SHAPE; goto done;
                }
                surf_bytes = 0;
                for (src = 0, t = 0; src < ntask; src++) {
                    unsigned done_rows = 0;
                    while (done_rows < plan[src].oh) {
                        unsigned take = plan[src].oh - done_rows;
                        if (take > cap) take = cap;
                        if (t == max_tasks) {
                            ROCKET_LOGE("rk3576 matmul i32: the wide writer's row bound "
                                        "needs more than %u tasks at programmed oc %u\n",
                                        max_tasks, oc_prog);
                            rc = ROCKET_E_SHAPE; goto done;
                        }
                        wplan[t].oy0 = (uint16_t)(plan[src].oy0 + done_rows);
                        wplan[t].oh  = (uint16_t)take;
                        wplan[t].iy0 = wplan[t].oy0;
                        wplan[t].ih  = (uint16_t)take;
                        task_off[t]  = surf_bytes;
                        surf_bytes  += atoms_per_px *
                                       rocket_rk3576_out_surf_elems(iw, take, 0) * C2;
                        /* A band between one task's surface and the next, so an atom
                         * written past a task's extent lands somewhere observable
                         * instead of in its neighbour's first run. */
                        if (diag >= 2) surf_bytes += R76_DIAG_TASK_GUARD;
                        done_rows += take;
                        t++;
                    }
                }
                ntask = t;
                for (t = 0; t < ntask; t++) plan[t] = wplan[t];
            }

            if (b.w.ptr)     rocket_bo_free(fd, &b.w);
            if (b.coeff.ptr) rocket_bo_free(fd, &b.coeff);
            if (b.out.ptr)   rocket_bo_free(fd, &b.out);
            memset(&b.w, 0, sizeof b.w);
            memset(&b.coeff, 0, sizeof b.coeff);
            memset(&b.out, 0, sizeof b.out);

            /* NOTHING writes the output buffer from the CPU — not the surface, and not
             * the band past it. A fresh BO arrives zeroed, which is already the
             * sentinel: the band staying zero is the overrun check, and the DPU writes
             * the whole surface when it writes at all.
             *
             * Zeroing it by hand is the trap. Those dirty cache lines race the DPU's
             * DMA and the writeback lands on top of the result, so the surface comes
             * back all zeros — intermittently, on the first submit as readily as a
             * later one, with no fault anywhere and normal timing. It reads exactly
             * like an arithmetic failure. The band is 64-byte aligned so nothing in it
             * can share a line with a surface byte either. */
            guard_off = (surf_bytes + 63u) & ~(size_t)63u;
            o_surf_max = surf_bytes;
            if (rocket_bo_alloc32(fd, w_bytes, &b.w) < 0 ||
                rocket_bo_alloc32(fd, coeff_bytes, &b.coeff) < 0 ||
                rocket_bo_alloc32(fd, guard_off + (size_t)surf_elems * C2, &b.out) < 0) {
                rc = ROCKET_E_NOMEM; goto done;
            }

            /* The scatter: real channel n lives at programmed channel 32*(n/8)+n%8,
             * which is a slot the 32-bit writer delivers. Everything else stays zero. */
            rocket_bo_prep(fd, &b.w, 1, 0);
            memset(b.w.ptr, 0, w_bytes);
            for (n = 0; n < tile_n; n++)
                for (k = 0; k < kslice; k++)
                    ((int8_t *)b.w.ptr)[weight_conv_int8((int)oc_prog, (int)kslice, 1, 1,
                                                         (int)r76_i32_prog_oc(n, mult) + 1,
                                                         (int)k + 1, 1, 1)] =
                        B[(size_t)(n0 + n) * K + k0 + k];
            rocket_bo_fini(fd, &b.w);

            /* No bias on the device here: a K split would add it once per slice. The C
             * term still has to be there — it gates the BS stage, and a zeroed
             * coefficient buffer returns a full but empty surface — which is what
             * pack_coeff writes for a NULL bias array. */
            rocket_bo_prep(fd, &b.coeff, 1, 0);
            rocket_rk3576_pack_coeff(b.coeff.ptr, coeff_bytes, NULL, oc_prog);
            rocket_bo_fini(fd, &b.coeff);

            p.ic = (uint16_t)kslice; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
            p.oc = (uint16_t)oc_prog; p.oh = (uint16_t)ih; p.ow = (uint16_t)iw;
            p.kh = 1; p.kw = 1;
            p.stride_y = 1; p.stride_x = 1;
            p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)ih;
            p.int8_out = 1;
            /* The i32out emitter pins OUT_CVT to exact unity itself; these only keep the
             * shared derivation away from a divide by zero. */
            p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;
            p.input_zero_point = 0x80;
            p.output_zero_point = 0x80;
            p.weight_zero_point = 0x80;
            p.tasks       = ops;
            p.input_dma   = b.in.dma_address + (uint32_t)((size_t)(k0 / ks) * in_slot);
            p.weights_dma = b.w.dma_address;
            p.bias_dma    = b.coeff.dma_address;
            p.output_dma  = b.out.dma_address;

            in_h[0] = b.in.handle; in_h[1] = b.w.handle;
            in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
            out_h[0] = b.out.handle;

          for (attempt = 0; attempt < 2; attempt++) {
            /* Stamp the surface, but not the band past it — the band staying zero is the
             * overrun check, and it has to keep meaning that. */
            if (blank) {
                rocket_bo_prep(fd, &b.out, 1, 0);
                memset(b.out.ptr, blank, surf_bytes);
                rocket_bo_fini(fd, &b.out);
                /* The stamp is only evidence if it reaches DDR. Reading an atom back as
                 * ZERO rather than as the sentinel says the writer emitted it with the
                 * wrong data — but only once the sentinel is known to have landed, since
                 * a fresh BO arrives zeroed and a stamp that never left the cache would
                 * read the same way. */
                if (diag >= 3) {
                    size_t si, bad = 0;
                    const unsigned char *sp;
                    rocket_bo_prep(fd, &b.out, 0, 0);
                    sp = (const unsigned char *)b.out.ptr;
                    for (si = 0; si < surf_bytes; si++) if (sp[si] != blank) bad++;
                    ROCKET_LOGI("rk3576 i32 diag: stamp check k0=%u n0=%u: %zu of %zu "
                                "bytes are not the sentinel before any submit\n",
                                k0, n0, bad, surf_bytes);
                    rocket_bo_fini(fd, &b.out);
                }
            }
            for (t = 0; t < ntask; t++) {
              /* PER-TASK retry, and it has to be per task on the wide path. Each row
               * task there owns its own sub-surface, so one poisoned submit leaves that
               * region empty while its siblings are full — and a check over the whole
               * tile then reads "something was written" and passes the hole through.
               * At the stock autosuspend delay this never fires; at a short one it does,
               * which is exactly the configuration that made it visible. */
              unsigned tattempt;
              int task_ok = 0;
              for (tattempt = 0; tattempt < R76_I32_TASK_ATTEMPTS; tattempt++) {
                conv_params_t q = p;
                uint64_t tjob, jns;
                q.ih = plan[t].ih; q.oh = plan[t].oh;
                q.pad_top = plan[t].pad_top;
                q.input_dma  = p.input_dma  + plan[t].feature_off;
                if (mult == 4u) {
                    /* The narrow writer's surface is one shared cube and the row plan's
                     * output offset counts int8 atoms; a 32-bit element moves the same
                     * 16-byte atom, so the offset carries over unchanged. */
                    q.output_dma = p.output_dma + plan[t].output_off;
                    q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
                } else {
                    /* The WIDE writer's surface cannot be entered part-way. Its map is
                     * one stream, s = 2*pixel + block, cut into runs of ow*oh_full — so
                     * a task starting at row oy0 is at stream position 2*oy0*ow, which
                     * crosses a run boundary at the surface's midpoint and stops being
                     * expressible as a base offset. A shared cube with a shifted base
                     * therefore computes correctly and lands in the wrong words, which
                     * is exactly what a row-split tile did before this.
                     *
                     * The kernel here is 1x1 at stride 1, so a row task is not a window
                     * into a larger convolution at all — it is a smaller convolution.
                     * Each one gets its OWN full surface, laid end to end in the same
                     * BO, and the de-scatter walks them. */
                    q.input_dma  = p.input_dma +
                                   (uint32_t)plan[t].iy0 * iw * C2;
                    q.output_dma = p.output_dma + (uint32_t)task_off[t];
                    /* ih_full stays the FULL plane — it is the feature cube's
                     * channel-group stride, and the cube in the BO is the whole one.
                     * Only oh_full moves, which is the output surface's. */
                    q.ih_full = (uint16_t)ih;
                    q.oh_full = plan[t].oh;
                    q.pad_top = 0;
                }
                if ((mult == 4u ? gen_conv2d_int8_rk3576_i32out(&q)
                                : gen_conv2d_int8_rk3576_i32out_wide(&q)) != 0) {
                    rc = ROCKET_E_SHAPE; goto done;
                }
                ROCKET_LOGD("rk3576 matmul i32: submit k0=%u n0=%u ic=%u oc=%u "
                            "in@%08x[%d %d %d %d] w@%08x[%d %d %d %d] out@%08x\n",
                            k0, n0, kslice, oc_prog, q.input_dma,
                            ((const int8_t *)b.in.ptr)[(size_t)(k0 / ks) * in_slot + 0],
                            ((const int8_t *)b.in.ptr)[(size_t)(k0 / ks) * in_slot + 1],
                            ((const int8_t *)b.in.ptr)[(size_t)(k0 / ks) * in_slot + 2],
                            ((const int8_t *)b.in.ptr)[(size_t)(k0 / ks) * in_slot + 3],
                            q.weights_dma,
                            ((const int8_t *)b.w.ptr)[0], ((const int8_t *)b.w.ptr)[1],
                            ((const int8_t *)b.w.ptr)[2], ((const int8_t *)b.w.ptr)[3],
                            q.output_dma);
                /* AN i32out JOB POISONS THE NEXT SUBMIT, and what clears it is a
                 * POWER CYCLE rather than elapsed time.
                 *
                 * The job that does not write completes normally — no fault, no IOMMU
                 * error, no timeout, the usual ~1.4 ms — and leaves the output buffer
                 * untouched, so it reads as the previous result rather than as a
                 * failure. Fresh buffers per submit, a different feature address and a
                 * different output address were each tried and none of them helps.
                 *
                 * What does: the driver's runtime-PM autosuspend cycling the NPU power
                 * domain. Write `on` to the device's `power/control` and no gap clears
                 * it — 600 ms still writes nothing — and the working gap tracks
                 * `power/autosuspend_delay_ms` one for one across 5, 10, 20 and 50 ms.
                 * The "50-100 ms of idle" this was first read as is that file's stock
                 * value plus the suspend round trip, and nothing about the silicon.
                 *
                 * What poisons is DPU 0x4010, the DATA_FORMAT word: driving its output
                 * WIDTH field to int32 does it, and driving its PROC_PRECISION field to
                 * int32 does it separately. Pinning OUT_CVT to unity — the other thing
                 * the i32out program changes — does not, and neither does any plain
                 * int8 job. The next job programs 0x4010 back to all-int8 and still
                 * comes back empty, so this is latched datapath state and not a stale
                 * register. [HW sweep, H96 MAX M9]
                 *
                 * So the cost is a submit per power cycle, and it is SETTABLE: lowering
                 * the driver's autosuspend delay shortens it proportionally, and a
                 * kernel-side reset of the DPU at job start would remove it. */
                /* The FIRST submit of a call is not idled — a caller that enters with the
                 * part already poisoned is covered by the per-task check below, which
                 * costs an idle only when it fires, where a leading idle would cost one
                 * every call. Sweeping the idle ahead of that first submit changes
                 * nothing about the wide writer's zero-data defect either: 3 failures in
                 * 30 at no idle against 6 in 30 at 800 ms. [HW sweep, H96 MAX M9] */
                if (submits++) rocket_rk3576_power_idle();
                r76_i32_stats.submits = submits;
                rocket_bo_prep(fd, &b.rc, 1, 0);
                memcpy(b.rc.ptr, ops, q.task_count * sizeof(uint64_t));
                rocket_bo_fini(fd, &b.rc);
                tjob = rocket_rk3576_job_clock();
                /* The 32-bit writers raise no DPU completion: without the hint each job
                 * waits out the driver's 125 ms backstop and is logged retired, 5 of 5
                 * each [HW sweep, H96 MAX M9, rocket 1.6.0, tests/rk3576_drain_probe
                 * raw, 2026-09-27]. So the class is named and the wait is the settle
                 * past PC_DONE — which is also all the fence says: a program still
                 * writing when it ends returns with atoms unwritten, and the per-atom
                 * scan below, not the fence, is what says the surface landed. */
                if (rocket_submit_matmul_flags(fd, &b.rc, q.task_count, in_h, 4,
                                               out_h, 1,
                                               rocket_no_dpu_done_supported()
                                                   ? ROCKET_JOB_NO_DPU_DONE : 0u) != 0) {
                    rc = ROCKET_E_DEVICE; goto done;
                }
                if (rocket_bo_prep(fd, &b.out, 0, 2000000000ull) < 0) {
                    rc = ROCKET_E_DEVICE; goto done;
                }
                jns = rocket_rk3576_job_ns(tjob);
                {
                    /* Did THIS task's region get written? A fresh BO arrives zeroed and
                     * the DPU fills the whole region when it writes at all, so a region
                     * still entirely zero is a poisoned submit. A legitimately zero
                     * region costs one wasted redo and stays correct.
                     *
                     * It has to be per TASK. One poisoned submit among several leaves
                     * its own rows empty while its siblings are full, so a check over
                     * the whole tile reads "something was written" and passes the hole
                     * straight through to the caller — a tile that is exactly one row
                     * task short, with no fault and normal timing.
                     *
                     * The region differs by writer. The wide one gives each task its own
                     * surface, so it is that surface. The narrow one shares a cube, so
                     * its task occupies the same rows of every channel block and the
                     * scan below walks them; the contiguous rows of CHANNEL BLOCK 0 are
                     * only enough to tell a dead submit from a live one. */
                    size_t off = mult == 4u ? (size_t)plan[t].output_off : task_off[t];
                    size_t tb  = mult == 4u
                               ? (size_t)plan[t].oh * iw * C2
                               : atoms_per_px *
                                 rocket_rk3576_out_surf_elems(iw, plan[t].oh, 0) * C2;
                    const unsigned char *sp = (const unsigned char *)b.out.ptr + off;
                    size_t si;
                    int twrote = 0;
                    unsigned holes = 0, unwritten = 0, suspect = 0;
                    if (blank && mult == 4u) {
                        /* THE NARROW WRITER DROPS ATOMS TOO, a handful at a time, and
                         * the only thing that sees them is a per-atom scan of every
                         * channel block this task wrote into. Its cube is `atoms_per_px`
                         * blocks of `surf_elems` pixels laid end to end, and a row task
                         * owns pixels [oy0*iw, oy0*iw + oh*iw) of each block.
                         *
                         * Both failures land the same way here — the atom still holds
                         * the sentinel — so the recovery is the conservative one: redo,
                         * and idle first, because the poisoning is the failure that
                         * needs it and idling a drop only costs time. */
                        unsigned blk, px0 = (unsigned)plan[t].oy0 * iw;
                        unsigned npx = (unsigned)plan[t].oh * iw;
                        for (blk = 0; blk < atoms_per_px; blk++) {
                            const unsigned char *bp =
                                (const unsigned char *)b.out.ptr +
                                ((size_t)blk * surf_elems + px0) * C2;
                            for (unsigned px = 0; px < npx; px++) {
                                const unsigned char *ap = bp + (size_t)px * C2;
                                unsigned i, nz = 0;
                                for (i = 0; i < C2; i++) {
                                    if (ap[i] != blank) break;
                                }
                                if (i == C2) { unwritten++; continue; }
                                for (i = 0; i < C2; i++) if (ap[i]) { nz = 1; break; }
                                if (!nz) suspect++;
                            }
                        }
                        holes  = unwritten + suspect;
                        twrote = holes == 0;
                    } else if (blank && mult != 4u) {
                        /* AGAINST A SENTINEL THE QUESTION IS PER ATOM, and that is what
                         * covers the wide writer's dropped atoms as well as a dead
                         * submit. The drop is a contiguous block in the writer's
                         * EMISSION order that simply never reaches DDR — a second read
                         * after an idle returns byte-identical contents — so redoing the
                         * task is the repair, and it lands every atom on the retry.
                         * [HW sweep, H96 MAX M9] */
                        /* Two failures to catch here, and they want different recovery.
                         * An atom still holding the sentinel was never emitted, which is
                         * the poisoning, and only the power domain cycling clears that —
                         * so that redo has to be idled. An emission stream that goes zero
                         * for a stretch was emitted with the wrong data, which is the
                         * wide writer's own defect and is indifferent to idle, so that
                         * redo is immediate and costs one submit. */
                        unsigned npx = iw * plan[t].oh;   /* pixels this task wrote */
                        unwritten = r76_i32_diag_task(sp, npx, oc_prog,
                                                      (unsigned)atoms_per_px, k0, n0, t,
                                                      blank, "unwritten", 1);
                        suspect = r76_i32_wide_suspect(sp, npx, oc_prog);
                        holes   = unwritten + suspect;
                        twrote  = holes == 0;
                    } else {
                        for (si = 0; si < tb; si++)
                            if (sp[si] != blank) { twrote = 1; break; }
                    }
                    /* SETTLE ZERO ATOMS AGAINST THE OPERANDS, not against the surface,
                     * and while the mapping is still the CPU's. From the second attempt
                     * on — the first redo is cheaper than a dot product and usually
                     * enough — a task whose only defect is zero atoms is decided
                     * exactly: if every one of them should be zero the surface is right
                     * and this is data, and if any should not it is a drop and the task
                     * is redone. A caller whose operands legitimately produce a zero
                     * atom (a padded batch, a masked token, an all-zero embedding) is
                     * answered on the arithmetic rather than made to spend its attempts.
                     * An atom still holding the sentinel was never written at all, so it
                     * is never data and does not reach this. */
                    if (!twrote && !unwritten && tattempt >= R76_I32_CPU_CHECK_FROM &&
                        r76_i32_zeros_are_data(mult == 4u
                                                   ? (const unsigned char *)b.out.ptr
                                                   : sp,
                                               mult, iw, (unsigned)plan[t].oy0,
                                               (unsigned)plan[t].oh,
                                               (unsigned)atoms_per_px, surf_elems,
                                               n0, tile_n, k0, kslice, K, A, B)) {
                        ROCKET_LOGD("rk3576 matmul i32: k0=%u n0=%u row task %u — %u "
                                    "zero atom(s) are the arithmetic; keeping the "
                                    "surface\n", k0, n0, t, suspect);
                        r76_i32_stats.accepted_zero++;
                        twrote = 1;
                    }
                    rocket_bo_fini(fd, &b.out);
                    /* A RETIRED JOB IS REDONE whatever its surface reads, and ahead of
                     * the zero-atom acceptance below, which would otherwise keep it. The
                     * per-atom check already sees a partial surface; this adds a job the
                     * driver gave up on that happens to read complete. No re-stamp: an
                     * atom this attempt left is still the sentinel, and the idle at the
                     * top of the loop is the power cycle the redo needs. */
                    if (rocket_rk3576_score_retired("rk3576 matmul i32", jns, twrote))
                        continue;
                    if (twrote) { task_ok = 1; break; }
                    /* SETTLE ZERO ATOMS AGAINST THE OPERANDS, not against the surface.
                     * From the second attempt on — the first redo is cheaper than a dot
                     * product and usually enough — a task whose only defect is zero
                     * atoms is decided exactly: if every one of them should be zero the
                     * surface is right and this is data, and if any should not it is a
                     * drop and the task is redone. An atom still holding the sentinel is
                     * never data, so it does not reach this. */
                    if (!unwritten && tattempt >= R76_I32_CPU_CHECK_FROM &&
                        r76_i32_zeros_are_data(mult == 4u
                                                   ? (const unsigned char *)b.out.ptr
                                                   : sp,
                                               mult, iw, (unsigned)plan[t].oy0,
                                               (unsigned)plan[t].oh,
                                               (unsigned)atoms_per_px, surf_elems,
                                               n0, tile_n, k0, kslice, K, A, B)) {
                        ROCKET_LOGD("rk3576 matmul i32: k0=%u n0=%u row task %u — %u "
                                    "zero atom(s) are the arithmetic; keeping the "
                                    "surface\n", k0, n0, t, suspect);
                        r76_i32_stats.accepted_zero++;
                        task_ok = 1;
                        break;
                    }
                    ROCKET_LOGD("rk3576 matmul i32: k0=%u n0=%u row task %u — %u atoms "
                                "never emitted, %u emitted zero; redoing it\n",
                                k0, n0, t, unwritten, suspect);
                    if (suspect) heals++;
                    if (unwritten) r76_i32_stats.redo_empty++;
                    if (suspect)   r76_i32_stats.redo_zeroed++;
                    r76_i32_stats.atoms_empty  += unwritten;
                    r76_i32_stats.atoms_zeroed += suspect;
                    /* Only the poisoning needs the power cycle. Idling on a zero-data
                     * redo would put 120 ms on a failure that does not care about it. */
                    if (unwritten) rocket_rk3576_power_idle();
                }
              }
              /* EXHAUSTING THE RETRIES IS AN ERROR, not a result. The sentinel makes
               * "this atom was never written" a fact about the surface rather than an
               * inference, so a task that still has holes after every attempt is a
               * surface this call cannot stand behind — and returning it quietly is how
               * one dead row task among a hundred reaches a caller as a plausible
               * matrix. Measured on M=32 K=1024 N=4096, where the per-task redo fires on
               * 1-12 of the 128 tasks a run and heals all of them: about one run in
               * thirty had a task that four attempts did not recover, and that run was
               * the only wrong answer in the set. [HW sweep, H96 MAX M9] */
              if (!task_ok) {
                  r76_i32_stats.refused++;
                  ROCKET_LOGE("rk3576 matmul i32: k0=%u n0=%u row task %u still has "
                              "unwritten atoms or was retired after %d attempts — "
                              "refusing to return a "
                              "partial surface\n", k0, n0, t, R76_I32_TASK_ATTEMPTS);
                  rc = ROCKET_E_DEVICE; goto done;
              }
              r76_i32_stats.tasks++;
            }
            /* A poisoned job leaves the surface exactly as it found it, and a fresh BO
             * arrives zeroed — so "still all zero" is the signal, and redoing the job
             * after an idle is the recovery. This covers the poisoning that crosses
             * CALLS, which the inter-submit idle above cannot see: the last i32out job
             * of a previous matmul poisons the first submit of this one. A surface that
             * is legitimately all zero costs one wasted retry and stays correct. */
            rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
            wrote = 0;
            {
                const unsigned char *sp = (const unsigned char *)b.out.ptr;
                size_t si;
                for (si = 0; si < surf_bytes; si++)
                    if (sp[si] != blank) { wrote = 1; break; }
            }
            rocket_bo_fini(fd, &b.out);
            if (wrote) break;
            if (attempt == 0) {
                ROCKET_LOGD("rk3576 matmul i32: slice k0=%u tile n0=%u wrote nothing, "
                            "idling and redoing it\n", k0, n0);
                rocket_rk3576_power_idle();
            }
          }

            rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
            {
                const unsigned char *guard = (const unsigned char *)b.out.ptr + guard_off;
                size_t g;
                for (g = 0; g < (size_t)surf_elems * C2; g++)
                    if (guard[g] != 0) {
                        ROCKET_LOGE("rk3576 matmul i32: the 32-bit writer reached past "
                                    "the largest surface any tile of this run claims "
                                    "(%zu bytes) at programmed oc=%u — the extent model "
                                    "is wrong for this shape\n", o_surf_max, oc_prog);
                        rocket_bo_fini(fd, &b.out);
                        rc = ROCKET_E_DEVICE; goto done;
                    }
            }
            if (diag && mult != 4u) {
                const unsigned char *sp = (const unsigned char *)b.out.ptr;
                unsigned dropped = 0, zeros = 0;
                for (t = 0; t < ntask; t++) {
                    unsigned npx = iw * plan[t].oh;   /* pixels this task wrote */
                    dropped += r76_i32_diag_task(sp + task_off[t], npx, oc_prog,
                                                 (unsigned)atoms_per_px, k0, n0, t,
                                                 blank, "UNWRITTEN (still the sentinel)",
                                                 0);
                    if (blank)
                        zeros += r76_i32_diag_task(sp + task_off[t], npx, oc_prog,
                                                   (unsigned)atoms_per_px, k0, n0, t,
                                                   0, "ZERO (emitted, wrong data)", 0);
                    if (diag >= 2) {
                        size_t gb = task_off[t] + (size_t)atoms_per_px * npx * C2, g;
                        for (g = 0; g < R76_DIAG_TASK_GUARD; g++)
                            if (sp[gb + g] != blank) {
                                ROCKET_LOGI("rk3576 i32 diag: k0=%u n0=%u task=%u wrote "
                                            "%u bytes PAST its surface (first at +%zu) — "
                                            "this is address generation, not a dropped "
                                            "write\n", k0, n0, t,
                                            (unsigned)(R76_DIAG_TASK_GUARD - g), g);
                                break;
                            }
                    }
                }
                diag_dropped += dropped;
                if (zeros)
                    ROCKET_LOGI("rk3576 i32 diag: k0=%u n0=%u %u atoms came back ZERO on a "
                                "stamped surface — the writer emitted them and the data is "
                                "wrong, rather than skipping them\n", k0, n0, zeros);
                diag_dropped += zeros;
                if (dropped)
                    ROCKET_LOGI("rk3576 i32 diag: k0=%u n0=%u dropped %u atoms over %u "
                                "row task%s (plane %ux%u, programmed oc %u)\n",
                                k0, n0, dropped, ntask, ntask == 1 ? "" : "s",
                                iw, ih, oc_prog);
            }
            /* ROCKET_RK3576_I32_RECHECK re-reads the surface after a second fence and
             * says whether anything changed. It separates the two ways a hole in the
             * output can arise — a word the DPU never wrote, and a word this side read
             * stale — which look identical from the accumulate below. */
            if (getenv("ROCKET_RK3576_I32_RECHECK")) {
                unsigned char *copy = malloc(surf_bytes);
                if (copy) {
                    size_t di, ndiff = 0, nzero = 0;
                    const unsigned char *sp;
                    memcpy(copy, b.out.ptr, surf_bytes);
                    rocket_bo_fini(fd, &b.out);
                    rocket_rk3576_power_idle();
                    rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
                    sp = (const unsigned char *)b.out.ptr;
                    for (di = 0; di < surf_bytes; di++) {
                        if (sp[di] != copy[di]) ndiff++;
                        if (!sp[di]) nzero++;
                    }
                    ROCKET_LOGI("rk3576 matmul i32: recheck k0=%u n0=%u: %zu of %zu "
                                "bytes changed on a second read, %zu still zero\n",
                                k0, n0, ndiff, surf_bytes, nzero);
                    free(copy);
                }
            }
            /* Accumulate this slice's partials into the caller's row-major C. */
            {
                const int32_t *o32 = (const int32_t *)b.out.ptr;
                int m;
                ROCKET_LOGD("rk3576 matmul i32: slice k0=%u(%u) tile n0=%u: "
                            "first word %d\n", k0, kslice, n0, o32[0]);
                t = 0;
                for (m = 0; m < M; m++) {
                    unsigned p_idx = (unsigned)m, y = p_idx / iw;
                    size_t base = 0;
                    if (mult != 4u) {
                        /* Which row task's surface this row landed in, and where it
                         * sits inside it. */
                        while (t + 1u < ntask && y >= (unsigned)plan[t + 1].oy0) t++;
                        while (t && y < (unsigned)plan[t].oy0) t--;
                        base = task_off[t] / 4u;
                        p_idx -= (unsigned)plan[t].oy0 * iw;
                    }
                    for (n = 0; n < tile_n; n++) {
                        int w;
                        if (mult == 4u) {
                            /* The narrow writer's scatter collapses to the plain
                             * RK3588 int32 cube. */
                            w = (int)((size_t)(n / 4u) * surf_elems * 4u +
                                      4u * p_idx + (n % 4u));
                        } else {
                            w = rocket_rk3576_i32_wide_word(iw, plan[t].oh,
                                                            r76_i32_prog_oc(n, mult),
                                                            p_idx);
                            if (w < 0) { rc = ROCKET_E_SHAPE; goto done; }
                        }
                        C[(size_t)m * N + n0 + n] += o32[base + (size_t)w];
                    }
                }
            }
            rocket_bo_fini(fd, &b.out);
        }
    }

    if (bias) {
        int m;
        unsigned n;
        for (m = 0; m < M; m++)
            for (n = 0; n < (unsigned)N; n++) C[(size_t)m * N + n] += bias[n];
    }
    /* Leave the part unpoisoned. The hazard an i32out job creates outlives the job and
     * outlives the PROCESS, because it lives in the NPU rather than in this fd: the next
     * submit of ANY kind writes nothing, and the plain int8 matmul in another program is
     * as exposed as this one — its output BO comes back untouched and the caller reads a
     * correctly sized, entirely stale surface. The entry that creates the hazard is the
     * one that should absorb it, so this idles once on the way out rather than leaving
     * the next caller to discover it. */
    rocket_rk3576_power_idle();
    /* Submits is the cost that matters on this path: the poisoning workaround pays an
     * idle per submit, so what a change buys is measured in submits and not in MACs. */
    ROCKET_LOGI("rk3576 matmul i32: M=%d K=%d N=%d done in %u submits (%ux writer)\n",
                M, K, N, submits, mult);
    if (diag && mult != 4u)
        ROCKET_LOGI("rk3576 i32 diag: M=%d K=%d N=%d dropped %u atoms in total, %u task%s "
                    "redone for a partial write\n",
                    M, K, N, diag_dropped, heals, heals == 1 ? "" : "s");
    else if (heals)
        ROCKET_LOGI("rk3576 matmul i32: %u row task%s redone — the wide writer left a "
                    "partial surface and the retry completed it\n",
                    heals, heals == 1 ? " was" : "s were");
    rc = ROCKET_OK;

done:
    free(ops); free(plan); free(wplan); free(task_off); free(stage);
    r76_mm_free(fd, &b);
    return rc;
}

/* ============================================================================
 * The matmul-form fp16 program: one job, the whole of K, an fp32 output.
 *
 * The register program is gen_matmul_fp16_rk3576() in npu_regcmd_rk3576.c; this is the
 * userspace around it. There is no tiling and no scatter: the activation goes in
 * row-major as it arrives, and the output comes back m rows of n fp32 values. What this
 * layer adds is the weight permutation, the coefficient buffer and the write check.
 *
 * THE WRITE CHECK IS PER ELEMENT. An output element still holding the sentinel stamp
 * after the fence is one the job did not write. The whole call is then redone after a
 * power cycle, up to the int32 path's attempt budget, and refused past it, because a
 * partly written surface returns plausible values beside stale ones. An fp32 result
 * whose bit pattern equals the stamp reads as unwritten; with a nonzero stamp byte
 * that pattern is one value in 2^32, and the stamp is 0xA5A5A5A5, about -2.9e-16.
 * ==========================================================================*/
static _Atomic unsigned g_r76_mmf16_attempts;

unsigned rocket_rk3576_mm_fp16_last_attempts(void)
{
    return atomic_load_explicit(&g_r76_mmf16_attempts, memory_order_relaxed);
}

/* The resident weight and coefficient buffer. The weight BO is the per-call path's
 * permutation of B over the whole N, and the coefficient buffer depends on N alone, so
 * both are what one call used to build; the contract is on the declarations in
 * rocket_matmul.h. */
struct rocket_rk3576_wbo_fp16 {
    rocket_bo w, coef;
    int K, N;
};

int rocket_rk3576_wbo_fp16_create(int fd, int K, int N, const _Float16 *B,
                                  struct rocket_rk3576_wbo_fp16 **out)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    struct rocket_rk3576_wbo_fp16 *h;
    int rc;

    if (out) *out = NULL;
    if (fd < 0 || !B || !out || K <= 0 || N <= 0) return ROCKET_E_SHAPE;
    if (strcmp(hw->name, "rk3576") != 0) {
        ROCKET_LOGE("rocket_rk3576_wbo_fp16_create: this is the RK3576 encoding and the "
                    "active profile is %s\n", hw->name);
        return ROCKET_E_UNSUPPORTED;
    }
    if (!rocket_rk3576_mm_fp16_shape_ok(1, (unsigned)K, (unsigned)N)) {
        ROCKET_LOGE("rocket_rk3576_wbo_fp16_create: K=%d N=%d is outside the gated "
                    "envelope\n", K, N);
        return ROCKET_E_UNSUPPORTED;
    }
    h = calloc(1, sizeof *h);
    if (!h) return ROCKET_E_NOMEM;
    if (rocket_bo_alloc32(fd, rocket_rk3576_mm_fp16_weight_bytes((unsigned)K, (unsigned)N),
                          &h->w) != 0 ||
        rocket_bo_alloc32(fd, rocket_rk3576_mm_fp16_coef_bytes((unsigned)N), &h->coef) != 0) {
        rocket_rk3576_wbo_fp16_free(fd, h);
        return ROCKET_E_NOMEM;
    }
    rocket_bo_prep(fd, &h->w, 1, 0);
    rc = rocket_rk3576_mm_fp16_pack_weights(h->w.ptr, h->w.size, (const uint16_t *)B,
                                            (unsigned)K, (unsigned)N);
    rocket_bo_fini(fd, &h->w);
    if (!rc) {
        rocket_bo_prep(fd, &h->coef, 1, 0);
        rc = rocket_rk3576_mm_fp16_pack_coef(h->coef.ptr, h->coef.size, (unsigned)N);
        rocket_bo_fini(fd, &h->coef);
    }
    if (rc) { rocket_rk3576_wbo_fp16_free(fd, h); return ROCKET_E_SHAPE; }
    h->K = K; h->N = N;
    *out = h;
    return ROCKET_OK;
}

void rocket_rk3576_wbo_fp16_free(int fd, struct rocket_rk3576_wbo_fp16 *h)
{
    if (!h) return;
    if (h->coef.ptr) rocket_bo_free(fd, &h->coef);
    if (h->w.ptr) rocket_bo_free(fd, &h->w);
    free(h);
}

/* The M tiling. One task carries at most rocket_rk3576_mm_fp16_task_rows(K) rows, since
 * its input surface must fit the CBUF data window, so M is cut into tasks of equal size
 * (the last one no larger than the rest). Each task reads its rows of the row-major
 * activation and writes its rows of the row-major output, so a tile is an offset into
 * both buffers and nothing is re-laid out. The tasks go out as one job, one drm task
 * descriptor each: the driver kicks them back to back, each on its own completion, and
 * the caller makes one ioctl and waits on one fence. */
#define R76_MMF16_MAX_TASKS 128u

/* The fp16 entry's buckets, under the same ROCKET_MM_PROFILE knob as the int8 entry's and
 * on a line of their own, so a process that ran both does not blend them. `scan` is the
 * copy out to C with the stamp check folded into it; `read` is the FINI_BO after it. */
struct r76_f16_prof {
    double alloc, packB, coeff, packA, gen, stamp, submit, wait, scan, read, bofree, wall;
    long tasks, submits;
};
static struct r76_f16_prof g_prof_f16;
static long g_prof_f16_calls;
static int g_prof_f16_armed;

static void r76_f16_prof_dump(void)
{
    const struct r76_f16_prof *g = &g_prof_f16;
    ROCKET_LOGI("ROCKET rk3576 fp16 matmul profile total(ms): wall=%.1f alloc=%.1f "
                "packB=%.1f coeff=%.1f packA=%.1f gen=%.1f stamp=%.1f submit=%.1f "
                "wait=%.1f scan=%.1f read=%.1f bofree=%.1f  over %ld calls / %ld tasks / "
                "%ld submits\n", g->wall, g->alloc, g->packB, g->coeff, g->packA, g->gen,
                g->stamp, g->submit, g->wait, g->scan, g->read, g->bofree,
                g_prof_f16_calls, g->tasks, g->submits);
}

static void r76_f16_prof_fold(const struct r76_f16_prof *p)
{
    struct r76_f16_prof *g = &g_prof_f16;
    pthread_mutex_lock(&g_prof_r76_mu);
    if (!g_prof_f16_armed) { atexit(r76_f16_prof_dump); g_prof_f16_armed = 1; }
    g->alloc += p->alloc; g->packB += p->packB; g->coeff += p->coeff;
    g->packA += p->packA; g->gen += p->gen; g->stamp += p->stamp;
    g->submit += p->submit; g->wait += p->wait; g->scan += p->scan;
    g->read += p->read; g->bofree += p->bofree; g->wall += p->wall;
    g->tasks += p->tasks; g->submits += p->submits;
    g_prof_f16_calls++;
    pthread_mutex_unlock(&g_prof_r76_mu);
}

static unsigned r76_mmf16_tile(unsigned M, unsigned K, unsigned *rows)
{
    unsigned cap = rocket_rk3576_mm_fp16_task_rows(K), n, mt;
    if (!cap) return 0;
    n = (M + cap - 1u) / cap;
    mt = (M + n - 1u) / n;
    *rows = mt;
    return (M + mt - 1u) / mt;
}

/* The one body. `wbo` NULL is the per-call entry, which builds the weight and
 * coefficient BOs from B; non-NULL takes both from the handle and never reads B. */
static int r76_mm_fp16(int fd, int M, int K, int N, const _Float16 *A,
                       const _Float16 *B, const struct rocket_rk3576_wbo_fp16 *wbo,
                       float *C)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    rocket_bo in = {0}, w = {0}, coef = {0}, out = {0}, rc = {0};
    const rocket_bo *wp = &w, *cp = &coef;
    rocket_task_desc td[R76_MMF16_MAX_TASKS];
    uint32_t in_h[4], out_h[1];
    unsigned char blank = rocket_rk3576_sentinel_on() ? (unsigned char)R76_SENTINEL_BYTE : 0;
    size_t ib, ob, oelems, e;
    unsigned attempt, ntask, mt, t;
    int ret = ROCKET_E_SHAPE;
    const int prof = r76_mm_profile();
    struct r76_f16_prof pr = {0};
    double pt0, wall_t0;

    if (fd < 0 || !A || (!B && !wbo) || !C || M <= 0 || K <= 0 || N <= 0)
        return ROCKET_E_SHAPE;
    if (strcmp(hw->name, "rk3576") != 0) {
        ROCKET_LOGE("rocket_matmul_fp16_rk3576: this is the RK3576 encoding and the "
                    "active profile is %s\n", hw->name);
        return ROCKET_E_UNSUPPORTED;
    }
    if (!rocket_rk3576_mm_fp16_shape_ok((unsigned)M, (unsigned)K, (unsigned)N)) {
        ROCKET_LOGE("rocket_matmul_fp16_rk3576: M=%d K=%d N=%d is outside the gated "
                    "envelope\n", M, K, N);
        return ROCKET_E_UNSUPPORTED;
    }
    if (wbo && (wbo->K != K || wbo->N != N)) {
        ROCKET_LOGE("rocket_matmul_fp16_rk3576_wbo: the handle holds K=%d N=%d and the "
                    "call asks K=%d N=%d\n", wbo->K, wbo->N, K, N);
        return ROCKET_E_SHAPE;
    }
    ntask = r76_mmf16_tile((unsigned)M, (unsigned)K, &mt);
    if (!ntask || ntask > R76_MMF16_MAX_TASKS) {
        ROCKET_LOGE("rocket_matmul_fp16_rk3576: M=%d K=%d plans into %u tasks, over the "
                    "%u one call lays out\n", M, K, ntask, R76_MMF16_MAX_TASKS);
        return ROCKET_E_SHAPE;
    }
    wall_t0 = PROF_T0();
    ib = rocket_rk3576_mm_fp16_in_bytes((unsigned)M, (unsigned)K);
    ob = rocket_rk3576_mm_fp16_out_bytes((unsigned)M, (unsigned)N);
    oelems = (size_t)M * (size_t)N;
    pt0 = PROF_T0();
    if (r76_bo_get(fd, ib, &in) || r76_bo_get(fd, ob, &out) ||
        r76_bo_get(fd, (size_t)ntask * RK3576_MM_FP16_OPS * sizeof(uint64_t), &rc)) {
        ret = ROCKET_E_NOMEM; goto done;
    }
    PROF_ADD(alloc, pt0);
    if (wbo) {
        wp = &wbo->w; cp = &wbo->coef;
    } else {
        pt0 = PROF_T0();
        if (r76_bo_get(fd, rocket_rk3576_mm_fp16_weight_bytes((unsigned)K, (unsigned)N), &w) ||
            r76_bo_get(fd, rocket_rk3576_mm_fp16_coef_bytes((unsigned)N), &coef)) {
            ret = ROCKET_E_NOMEM; goto done;
        }
        PROF_ADD(alloc, pt0);
        pt0 = PROF_T0();
        rocket_bo_prep(fd, &w, 1, 0);
        ret = rocket_rk3576_mm_fp16_pack_weights(w.ptr, w.size, (const uint16_t *)B,
                                                 (unsigned)K, (unsigned)N);
        rocket_bo_fini(fd, &w);
        PROF_ADD(packB, pt0);
        if (ret) { ret = ROCKET_E_SHAPE; goto done; }
        pt0 = PROF_T0();
        rocket_bo_prep(fd, &coef, 1, 0);
        ret = rocket_rk3576_mm_fp16_pack_coef(coef.ptr, coef.size, (unsigned)N);
        rocket_bo_fini(fd, &coef);
        PROF_ADD(coeff, pt0);
        if (ret) { ret = ROCKET_E_SHAPE; goto done; }
    }

    pt0 = PROF_T0();
    rocket_bo_prep(fd, &in, 1, 0);
    memcpy(in.ptr, A, (size_t)M * K * 2);
    memset((unsigned char *)in.ptr + (size_t)M * K * 2, 0, ib - (size_t)M * K * 2);
    rocket_bo_fini(fd, &in);
    PROF_ADD(packA, pt0);

    pt0 = PROF_T0();
    rocket_bo_prep(fd, &rc, 1, 0);
    for (t = 0; t < ntask; t++) {
        unsigned r0 = t * mt, rows = (unsigned)M - r0 < mt ? (unsigned)M - r0 : mt;
        uint64_t *slot = (uint64_t *)rc.ptr + (size_t)t * RK3576_MM_FP16_OPS;
        if (gen_matmul_fp16_rk3576(slot, rows, (unsigned)K, (unsigned)N,
                                   (uint32_t)(in.dma_address + (uint64_t)r0 * K * 2),
                                   (uint32_t)wp->dma_address,
                                   (uint32_t)(out.dma_address + (uint64_t)r0 * N * 4),
                                   (uint32_t)cp->dma_address) != RK3576_MM_FP16_OPS) {
            rocket_bo_fini(fd, &rc);
            ret = ROCKET_E_SHAPE; goto done;
        }
        td[t].regcmd = (uint32_t)(rc.dma_address +
                                  (uint64_t)t * RK3576_MM_FP16_OPS * sizeof(uint64_t));
        td[t].regcmd_count = RK3576_MM_FP16_OPS;
    }
    rocket_bo_fini(fd, &rc);
    PROF_ADD(gen, pt0);
    in_h[0] = in.handle; in_h[1] = wp->handle; in_h[2] = cp->handle; in_h[3] = rc.handle;
    out_h[0] = out.handle;
    pr.tasks = ntask;

    ret = ROCKET_E_DEVICE;
    atomic_store_explicit(&g_r76_mmf16_attempts, 0, memory_order_relaxed);
    for (attempt = 0; attempt < R76_I32_TASK_ATTEMPTS; attempt++) {
        size_t stale = 0;
        int srv, retired;
        uint64_t tjob, jns;
        atomic_store_explicit(&g_r76_mmf16_attempts, attempt + 1, memory_order_relaxed);
        if (blank) {
            pt0 = PROF_T0();
            rocket_bo_prep(fd, &out, 1, 0);
            memset(out.ptr, blank, oelems * 4);
            rocket_bo_fini(fd, &out);
            PROF_ADD(stamp, pt0);
        }
        pt0 = PROF_T0();
        tjob = rocket_rk3576_job_clock();
        srv = ntask == 1
            ? rocket_submit_matmul(fd, &rc, RK3576_MM_FP16_OPS, in_h, 4, out_h, 1, 4000)
            : rocket_submit_tasks(fd, td, ntask, in_h, 4, out_h, 1);
        PROF_ADD(submit, pt0);
        pr.submits++;
        if (srv != 0) goto done;
        pt0 = PROF_T0();
        if (rocket_bo_prep(fd, &out, 0, 2000000000ull) < 0) goto done;
        jns = rocket_rk3576_job_ns(tjob);
        PROF_ADD(wait, pt0);
        /* The stamp check rides the copy out. Each is one read of an output that has
         * just been invalidated from the cache, and at 512x1536 outputs each pass cost
         * about 1 ms at DRAM rate, so a separate scan was a second read of the same
         * bytes. C holds a rejected surface until the redo overwrites it. */
        pt0 = PROF_T0();
        if (blank) {
            const uint32_t *o = (const uint32_t *)out.ptr;
            const uint32_t pat = (uint32_t)blank * 0x01010101u;
            uint32_t *c = (uint32_t *)C;
            for (e = 0; e < oelems; e++) {
                uint32_t v = o[e];
                c[e] = v;
                stale += v == pat;
            }
        } else {
            memcpy(C, out.ptr, oelems * 4);
        }
        PROF_ADD(scan, pt0);
        /* A retired job is redone even when no output reads stale: the driver gave up on
         * it, and the stamp at the top of the next attempt covers the whole surface. */
        retired = rocket_rk3576_score_retired("rk3576 fp16 matmul", jns, !stale);
        if (!stale && !retired) {
            pt0 = PROF_T0();
            rocket_bo_fini(fd, &out);
            PROF_ADD(read, pt0);
            ret = ROCKET_OK;
            break;
        }
        rocket_bo_fini(fd, &out);
        ROCKET_LOGD("rk3576 fp16 matmul: %zu of %zu outputs unwritten over %u task(s)%s, "
                    "idling and redoing\n", stale, oelems, ntask,
                    retired ? " and the job was retired" : "");
        rocket_rk3576_power_idle();
    }
    if (ret != ROCKET_OK)
        ROCKET_LOGE("rk3576 fp16 matmul: M=%d K=%d N=%d left outputs unwritten or was "
                    "retired on each of %d attempts — refusing to return a partial "
                    "surface\n", M, K, N, R76_I32_TASK_ATTEMPTS);

done:
    pt0 = PROF_T0();
    if (rc.ptr)   r76_bo_put(fd, &rc);
    if (out.ptr)  r76_bo_put(fd, &out);
    if (coef.ptr) r76_bo_put(fd, &coef);
    if (w.ptr)    r76_bo_put(fd, &w);
    if (in.ptr)   r76_bo_put(fd, &in);
    if (prof) {
        PROF_ADD(bofree, pt0);
        pr.wall = r76_now_ms() - wall_t0;
        r76_f16_prof_fold(&pr);
    }
    return ret;
}

int rocket_matmul_fp16_rk3576(int fd, int M, int K, int N,
                              const _Float16 *A, const _Float16 *B, float *C)
{
    if (!B) return ROCKET_E_SHAPE;
    return r76_mm_fp16(fd, M, K, N, A, B, NULL, C);
}

int rocket_matmul_fp16_rk3576_wbo(int fd, int M, int K, int N, const _Float16 *A,
                                  const struct rocket_rk3576_wbo_fp16 *wbo, float *C)
{
    if (!wbo) return ROCKET_E_SHAPE;
    return r76_mm_fp16(fd, M, K, N, A, NULL, wbo, C);
}
