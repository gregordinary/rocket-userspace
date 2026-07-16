// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_matmul.c — tiled fp16 matmul on the rocket NPU. See rocket_matmul.h.
 *
 * Throughput techniques, cumulative:
 *   - BIGGER TILES (Mt,Nt up to 256, K split to fit) — fewer tiles, less DRAM
 *     traffic (the NPU doesn't cache).
 *   - PRE-PACK ONCE — A and B are scattered into NPU tile layout exactly once into
 *     two big BOs; the compute loop only points DMA at the right slot.
 *   - MULTI-TASK BATCHING — instead of one job (one fence) per tile, pack up to
 *     BATCH tiles into a single job via rocket_submit_tasks(). The NPU runs them
 *     back-to-back under ONE completion fence, amortising the submit syscall +
 *     fence/IRQ + cache-sync (~1.5ms/job of overhead).
 *
 * The matmul splits into reusable phases (rocket_matmul_internal.h): pack weights
 * -> pack input -> compute. This one-shot entry point does all three per call;
 * rocket_prepacked.c reuses the phases to keep weights/scratch resident.
 *
 * K is split with CPU fp32 accumulation (fp16-out partials). 3-core concurrency
 * lives in rocket_matmul_mt.c / rocket_prepacked.c.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define ROCKET_HAVE_NEON 1
#endif

#include "rocket_npu.h"
#include "rocket_hw_profile.h"   /* machine parameters (cbuf_banks / max_tile / dtype mask) */
#include "npu_matmul.h"
#include "rocket_matmul.h"
#include "rocket_matmul_internal.h"
#include "rocket_chain.h"        /* contiguous self-chaining regcmd layout (shared) */
#include "rocket_log.h"     // centralized log channel

/* ############################################################################
 * PART 1 — Config, env-knob resolution, profiling, and shared layout helpers
 * ##########################################################################*/

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Output-fence wait deadline. Default 8s; ROCKET_WAIT_MS overrides. Sized to ride out a
 * multi-second CPU-side stall — e.g. a 24GB model faulting in from NVMe under tight RAM —
 * that can outlast a short wait on both the prepacked path and its mt fallback, expiring
 * into a FATAL timeout instead of a graceful slow op. 8s cannot mask a true hardware hang:
 * the kernel's own NPU job watchdog fires long before then and logs to dmesg, so a wait
 * that elapses without a watchdog entry is CPU-side latency, not a wedged device. The
 * >500ms SLOW-job log below surfaces any build-up toward the deadline. */
static long rocket_wait_ns(void) {
    static _Atomic long ns = -1;
    if (ns < 0) {
        const char *e = getenv("ROCKET_WAIT_MS");
        long ms = e ? atol(e) : 8000;
        if (ms < 1) ms = 8000;
        ns = ms * 1000000L;
    }
    return ns;
}

/* Cached ROCKET_MM_PROFILE: read ONCE rather than a getenv (a locked environ scan)
 * per matmul / worker call. The knob is process-start config and the profiler arms
 * on first use, so caching matches rocket_wait_ns / reuse_policy above. */
static int mm_profile(void) {
    static _Atomic int v = -1;
    if (v < 0) v = getenv("ROCKET_MM_PROFILE") != NULL;
    return v;
}

/* Phase profiling: set ROCKET_MM_PROFILE=1 to print an AGGREGATE breakdown (one
 * summary line at exit) across every matmul + worker. Workers run concurrently so
 * the accumulator is mutex-guarded; the atexit dump is armed on first use. */
static pthread_mutex_t g_prof_mu = PTHREAD_MUTEX_INITIALIZER;
static struct { double pack, packA, packB, gen, sync, submit, wait, read; long calls; } g_prof;
static int g_prof_armed = 0;
static void mm_prof_dump(void) {
    /* packA = input(A) scatter, packB = weight(B) scatter; on the mt path A is
     * packed redundantly per worker (lever 2 target) while B is unavoidable
     * (each weight used once per prefill). Their split decides which to attack. */
    ROCKET_LOGI("ROCKET profile total(ms): pack=%.0f (packA=%.0f packB=%.0f) "
            "gen=%.0f sync=%.0f submit=%.0f wait=%.0f read=%.0f  over %ld job-batches\n",
            g_prof.pack, g_prof.packA, g_prof.packB, g_prof.gen, g_prof.sync,
            g_prof.submit, g_prof.wait, g_prof.read, g_prof.calls);
}
static void mm_prof_arm_locked(void) {
    if (!g_prof_armed) { atexit(mm_prof_dump); g_prof_armed = 1; }
}
/* Attribute pack time done OUTSIDE mm_compute to the aggregate. kind: 'A' = input
 * scatter, 'B' = weight scatter, 0 = unclassified (still counted in the total). */
static void mm_prof_add_pack_kind(double ms, char kind) {
    if (!mm_profile()) return;
    pthread_mutex_lock(&g_prof_mu);
    mm_prof_arm_locked();
    g_prof.pack += ms;
    if (kind == 'A') g_prof.packA += ms;
    else if (kind == 'B') g_prof.packB += ms;
    pthread_mutex_unlock(&g_prof_mu);
}
/* Attribute pack time done OUTSIDE mm_compute (e.g. the prepacked path's single
 * shared A-pack) to the aggregate, so the summary stays honest. */
void mm_prof_add_pack(double ms)   { mm_prof_add_pack_kind(ms, 'A'); }
void mm_prof_add_pack_b(double ms) { mm_prof_add_pack_kind(ms, 'B'); }

/* int8-path profiler — a SEPARATE accumulator from g_prof. In a hybrid
 * int8 run, over-budget weights fall back to the fp16 streaming path (which feeds
 * g_prof), while the int8 one-shot path feeds this; keeping them distinct stops the
 * two from contaminating each other. Same ROCKET_MM_PROFILE knob; its own exit line. */
static struct { double packA, packB, sync, submit, wait, read; long calls; } g_prof_i8;
static int g_prof_i8_armed = 0;
static void mm_prof_i8_dump(void) {
    ROCKET_LOGI("ROCKET int8 profile total(ms): pack=%.0f (packA=%.0f packB=%.0f) "
            "sync=%.0f submit=%.0f wait=%.0f read=%.0f  over %ld job-batches\n",
            g_prof_i8.packA + g_prof_i8.packB, g_prof_i8.packA, g_prof_i8.packB,
            g_prof_i8.sync, g_prof_i8.submit, g_prof_i8.wait, g_prof_i8.read, g_prof_i8.calls);
}

/* fp16 NPU-side K-accumulation — the operating mode, DEFAULT-ON. Accumulates the
 * nKt K-partials on the NPU (DPU eltwise-add) instead of reading each partial back
 * and summing on the CPU: +19% warm prefill, `wait` −21% (HW-confirmed), coherent
 * greedy output. Opt OUT with ROCKET_KACC=0 or ROCKET_NO_KACC (both select the
 * byte-identical CPU-accum mm_compute oracle). Read once and cached.
 *
 * fp16 ONLY. int8/int4 on-device K-accum is a hard HW dead end (the DPU eltwise
 * operand DMA is ≤16-bit; int32 partials don't fit) — those dtype paths live in
 * separate translation units and never consult this knob. Don't inherit it. */
static int kacc_on(void) {
    static _Atomic int v = -1;
    if (v < 0) {
        const char *e = getenv("ROCKET_KACC");
        if (e) v = (atoi(e) != 0);                 /* explicit: 0 = off, nonzero = on */
        else   v = (getenv("ROCKET_NO_KACC") == NULL);   /* default ON unless opted out */
    }
    return v;
}

/* Cross-ki KACC chaining (ROCKET_KACC_CHAIN, default 0/OFF). The KACC path
 * accumulates a tile's nKt K-partials on the NPU, one fenced submit per ki-step
 * (each ki>0 reads the prior partial via the fp16 EW-add). This knob instead
 * chains the whole ki sequence — and the tiles within each ki — into ONE
 * self-chained kick (the rocket_chain PC_BASE_ADDRESS redirect), collapsing nKt
 * fences to one. Cutting fences always trims the submit/sync terms; the catch is
 * the wait term: the ki-steps are serially dependent (each reads the prior
 * partial), so a chained kick can only pipeline the INDEPENDENT tiles WITHIN each
 * ki-block. With gcap = BATCH/nKt >= 3 that intra-block overlap hides the serial
 * stalls and the fence savings win (~5% at nKt 12-21 [HW sweep]); at gcap <= 2 the
 * stalls dominate and wait balloons (+18% at nKt 40, gcap 1). So:
 *   1 = adaptive: chain only in the winning regime (gcap >= 3), else per-ki — never
 *       regresses, the production setting.
 *   2 = force: chain for any fitting nKt (<= BATCH); for the correctness gate, which
 *       must exercise the fully-serial gcap=1 case (the strictest RAW test).
 * Read FRESH per matmul call (not cached) so one process can A/B mid-run (the gate +
 * perf sweep setenv between calls); once per matmul, not per tile, so cost is nil. */
static int kacc_chain_on(void) {
    const char *e = getenv("ROCKET_KACC_CHAIN");
    int v = e ? atoi(e) : 0;
    if (v <= 0)
        return 0;
    /*
     * Chaining is a joint contract with the kernel (we self-chain the regcmds and
     * set DRM_ROCKET_JOB_BATCHED; the kernel sets TASK_NUMBER = task_count). On a
     * kernel that does not honor the flag the submit is rejected (a contract kernel
     * refuses the unknown trailing field with -E2BIG) or the chained layout runs
     * down the per-task path and corrupts. Either way, fall back to the per-ki
     * fenced KACC path, which is correct on any kernel. Same gate as
     * rkt_chain_enabled() -- see rocket_batched_submit_supported(). */
    if (!rocket_batched_submit_supported())
        return 0;
    return v;
}

/* Asymmetric-Nt tiling (ROCKET_MM_ASYM, default ON; set =0 to opt out). The planner maximizes
 * a SYMMETRIC Mt=Nt=MAX_TILE tile, which fixes Kt at the symmetric CBUF fit (384 on RK3588). For
 * a shape that tiles BOTH N and K, halving Nt to MAX_TILE/2 (keeping Mt full) frees CBUF so Kt
 * grows (384→512), and the asymmetric Mt>Nt tile runs the NPU datapath markedly more efficiently
 * — `wait` −~10%. End-to-end warm pp2048 [HW sweep, 600 MHz]: standalone matmuls 1024² +7.5% /
 * Gemma FFN-down +4% [2026-06-30]; whole-model llama.cpp Qwen3.5-9B-F16 +9.5%, Gemma-4-12B-F16
 * +5.7%, Qwen3.5-9B-Q4_K +1.3% (quant is dequant-bound so the datapath win dilutes to ~noise)
 * [2026-07-01]. Win-or-wash across every tested shape/model, never a regression, and bit-exact
 * (tiling never changes the result; gated by matmul_correctness_matrix under both settings) →
 * default-on. `ROCKET_MM_ASYM=0` forces the symmetric plan. Read once per matmul (planner is
 * per-call, not per-tile). */
static int asym_on(void) {
    const char *e = getenv("ROCKET_MM_ASYM");
    return e ? atoi(e) > 0 : 1;   /* default ON */
}

/* CBUF operand-reuse policy. ROCKET_REUSE = 0 off / 1 WEIGHT_REUSE / 2 DATA_REUSE
 * FORCES that mode for every shape (per-shape A/B + the mm_compute_reuse oracle both
 * need the exact mode). UNSET defaults to AUTO when KACC is the operating mode (pick
 * per shape, see reuse_mode_for) and to 0 (off) otherwise — the byte-identical
 * no-reuse CPU-accum path. Cached on first read.
 *
 * Why reuse pays: a job's consecutive same-operand tiles set the CNA
 * WEIGHT_REUSE/DATA_REUSE bit and read that operand from CBUF instead of re-fetching
 * it from DRAM. DATA_REUSE measured +7% in-model warm pp2048 (13.5→14.5), `wait`
 * −21% HW-confirmed; correct (coherent greedy, VERIFY nonfinite=0).
 * Returns: -1 AUTO, 0 off, 1 WEIGHT, 2 DATA. */
static int reuse_policy(void) {
    static int p = -2;
    if (p == -2) {
        const char *e = getenv("ROCKET_REUSE");
        if (e) {
            p = atoi(e);
            if (p < 0 || p > 2) p = 0;
        } else {
            p = kacc_on() ? -1 : 0;   /* AUTO rides with KACC */
        }
    }
    return p;
}

/* Resolve the reuse policy to a concrete CNA mode (0/1/2) for ONE tile geometry.
 * Reuse only ever spans tiles WITHIN a single job, so its value is the length of the
 * consecutive same-operand run: DATA_REUSE chains nNt tiles sharing the (mi,ki)
 * input, WEIGHT_REUSE chains nMt sharing the (ni,ki) weight. The two operand tiles
 * are the same size at the Mt==Nt cap, so the deeper run wins (more CBUF re-hits,
 * more skipped DRAM fetches) — and a deeper run is never worse, so AUTO can't
 * regress vs the static DATA default. AUTO picks it; ties -> DATA_REUSE (the
 * validated default + measured winner at the common nMt<=nNt prefill shapes, e.g.
 * any M @ a wide FFN/attn weight). A FORCED 0/1/2 passes through unchanged. */
/* ROCKET_REUSE_DEBUG: log each shape's resolved reuse mode (the AUTO decision is
 * data-dependent, so a one-line trace per matmul is the field check that AUTO picked
 * the intended mode). Cached on first read like the other knobs — off by default, so
 * the steady state is one branch, no per-call getenv. */
static int reuse_debug(void) {
    static _Atomic int v = -1;
    if (v < 0) v = getenv("ROCKET_REUSE_DEBUG") != NULL;
    return v;
}
static int reuse_mode_for(int nMt, int nNt) {
    int p = reuse_policy();
    int m = (p != -1) ? p                  /* off or forced */
                      : (nNt >= nMt) ? 2 : 1;   /* AUTO: deeper consecutive run wins */
    if (reuse_debug())
        ROCKET_LOGI("reuse: %s mode=%d (%s) nMt=%d nNt=%d\n",
                    p == -1 ? "AUTO" : "forced", m,
                    m == 0 ? "off" : m == 2 ? "DATA" : "WEIGHT", nMt, nNt);
    return m;
}

/* Shared M==1 GEMV shim for the six dtype one-shot paths. The HW height-1 conv
 * geometry mis-computes (see rocket_matmul_plan), so pad the single input row to a
 * height-4 tile, run the dtype's matmul at M=4, and copy back row 0. AT/CT are the
 * A/C element types and MM the dtype's matmul entry; the macro expands to the
 * per-dtype idiom VERBATIM (calloc'd 4xK in / 4xN out, the M=4 call, the row-0
 * copy, free, and an early return from the caller) so each path stays byte-exact.
 * One place to fix now instead of six. */
#define MM_PAD_M1(fd, K, N, A, B, C, AT, CT, MM) do {              \
    AT *Ap_ = calloc((size_t)4 * (K), sizeof(AT));                 \
    CT *Cp_ = calloc((size_t)4 * (N), sizeof(CT));                 \
    if (!Ap_ || !Cp_) { free(Ap_); free(Cp_); return -1; }         \
    memcpy(Ap_, (A), (size_t)(K) * sizeof(AT));                    \
    int r_ = MM((fd), 4, (K), (N), Ap_, (B), Cp_);                 \
    if (!r_) memcpy((C), Cp_, (size_t)(N) * sizeof(CT));           \
    free(Ap_); free(Cp_);                                          \
    return r_;                                                     \
} while (0)

/* CBUF bank SIZE stays a compile-time constant pointed at the single npu_hw.h source
 * (the banks_for_*() helpers want a constant). The bank COUNT and the Mt/Nt cap are
 * read from the active hardware profile (rocket_hw_current(): cbuf_banks / max_tile),
 * so those chip-specific values live in ONE place rather than as bare 12 / 256 literals
 * duplicated per planner (the "edit one, miss the others" mis-tile trap). Each planner
 * below pulls them into `const int CBUF_BANKS` / `MAX_TILE` locals. */
#define CBUF_BANK   NPU_CBUF_BANK_SIZE
#define BATCH       64           /* tiles (tasks) per NPU job                      */
#define RC_STRIDE   128          /* u64 words reserved per task in the regcmd BO   */

static int rup(int x, int a) { return ((x + a - 1) / a) * a; }
static int banks_for(int rows, int Kt) {
    return ((long)rows * Kt * 2 + CBUF_BANK - 1) / CBUF_BANK;
}

/* ── Batched (chained) submit — the dispatch-floor lever ───────────────────────
 * A tiled matmul submits one HW kick (one submit + one completion IRQ + one
 * waiter wakeup) per output tile. Batched submit instead lays a job's per-tile
 * regcmds CONTIGUOUSLY in the regcmd BO so the NPU PC streams straight through
 * them and fires a single IRQ for the whole job: the kernel programs only task 0
 * and sets PC_TASK_CON.TASK_NUMBER = N, which gates that one interrupt (it fires
 * when the PC task counter reaches N).
 *
 * This is the userspace half. It REQUIRES the coordinated kernel change
 * (module param rocket_batch_submit=1); against a stock kernel the regcmds chain
 * but TASK_NUMBER stays 1, so only task 0 runs and the job times out — enable
 * BOTH halves together, or NEITHER. The default (mode 0) is the stock gapped
 * layout with one IRQ per task, byte-for-byte unchanged.
 *
 * Stock lays each task's regcmd in a fixed RC_STRIDE-u64 slot with a gap and
 * leaves the trailer PC_REGISTER_AMOUNTS op at 0. Chained packing drops the gap:
 * each task occupies exactly its (even-rounded) word count, which is also how far
 * the PC advances per task — it reads (regcmd_count+1)/2 128-bit chunks — and each
 * task's trailer is rewritten to carry the NEXT task's address (an embedded
 * PC_BASE_ADDRESS op) and stream length (PC_REGISTER_AMOUNTS), so after each
 * OP_ENABLE the PC follows the link to the next task. The PC_BASE_ADDRESS redirect
 * is load-bearing: with the address left pointing at task 0 the PC runs task 0 and
 * stops, regardless of the amount. Every matmul tile shares one data-independent
 * regcmd word count, so the stride is uniform and tasks[i].regcmd = base+i*stride.
 *
 * ROCKET_BATCH_SUBMIT = 0 (default, stock per-task) or 1 (chained). A chained batch
 * must be uniform-length (the matmul's op count is data-independent); a caller
 * mixing regcmd lengths in one job must keep it off. */
/* The chain layout primitives now live in rocket_chain.{h,c} so every submit path
 * (matmul fp16/int8/int4, conv) shares one implementation. These thin shims bind
 * the matmul's gapped slot stride (RC_STRIDE) so the call sites below are
 * unchanged. */
static inline int  mm_batch_chained(void) { return rkt_chain_enabled(); }
static inline void mm_pack_regcmd(int chained, rocket_bo *rcbo, rocket_task_desc *tasks,
                                  int nb, const uint64_t *src, uint32_t count) {
    rkt_chain_pack(chained, rcbo, tasks, nb, src, count, (size_t)RC_STRIDE);
}
static inline void mm_seal_chain(int chained, rocket_bo *rcbo, int nb, uint32_t count) {
    rkt_chain_seal(chained, rcbo, nb, count);
}

/* Inlined NPU layout-index math — identical to feature_data()/weight_fp16() in
 * npu_regcmd.c, but inlined to kill the per-element out-of-line call+divide
 * in the pack/readback loops (those run tens of millions of times and were ~67%
 * of wall time). feat: input/output cube (W=1, C2=8). wt: (N/16,K/32,16,32). */
static inline size_t feat_idx(int H, int ch, int h) {
    return ((size_t)(ch - 1) / 8) * (size_t)H * 8 + 8 * (size_t)(h - 1) + (ch - 1) % 8;
}
static inline size_t wt_idx(int C, int k, int c) {
    return (size_t)((c - 1) / 32) * 32 * 16 + (size_t)((k - 1) / 16) * 16 * C
         + (size_t)((c - 1) % 32) + (size_t)((k - 1) % 16) * 32;
}

/* Readback de-tile gather. Accumulate one output tile's Ntile columns x
 * Mtile rows from the NPU output cube `slot` into the row-major fp32 `acc` (the
 * largest single CPU component of the matmul: read ~= 214 ms of a ~1.1 s call,
 * index/gather-bound). The output cube is C2=8 (feat_idx): for a fixed row h, the
 * 8 columns of each group g=(nn-1)/8 are 8 CONTIGUOUS fp16 at
 * slot[g*H*8 + 8*(h-1) + 0..7], and their acc destinations are 8 CONTIGUOUS
 * floats. So each group is one 128-bit fp16 load -> 2x fp32 convert -> 2x fp32
 * load+add+store. Nt is a multiple of 16 (N%16==0, Nt rounded to 16) so Ntile is
 * always %8 — the scalar tail is a safety net, not the steady state. Bit-identical
 * to the scalar `acc[...] += (float)slot[feat_idx(H,nn,h)]` it replaces (same adds,
 * same order), so the cosine-sim correctness matrix gates it. */
static inline void detile_accum_f16(float *restrict acc, int N,
                                    const _Float16 *restrict slot,
                                    int bm0, int bn0, int Mtile, int Ntile) {
    const int H = Mtile;
    for (int h = 1; h <= Mtile; h++) {
        float *d = acc + (size_t)(bm0 + h - 1) * N + bn0;
        int g = 0;
#ifdef ROCKET_HAVE_NEON
        const _Float16 *s = slot + 8 * (size_t)(h - 1);
        const int ng = Ntile / 8;
        for (; g < ng; g++) {
            const __fp16 *src = (const __fp16 *)(s + (size_t)g * H * 8);
            float *dst = d + (size_t)g * 8;
            float16x8_t v  = vld1q_f16(src);
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(v));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(v));
            vst1q_f32(dst,     vaddq_f32(vld1q_f32(dst),     lo));
            vst1q_f32(dst + 4, vaddq_f32(vld1q_f32(dst + 4), hi));
        }
        g *= 8;   /* columns already done = ng*8 */
#endif
        for (int nn = g + 1; nn <= Ntile; nn++)   /* scalar tail (and the non-NEON build) */
            d[nn - 1] += (float)slot[feat_idx(H, nn, h)];
    }
}

/* Final narrowing of the de-tiled fp32 accumulator into the caller's fp16 output.
 * detile_accum_f16 leaves M*N floats in acc[]; this pass narrows them. Both read and
 * write are contiguous (a pure convert, not a gather), but the cast loop is scalar at
 * -O2, so vectorize it like the de-tile: fcvtn 8 lanes/iter, scalar tail + non-NEON
 * fallback. Free win on large M*N. */
static inline void narrow_f32_to_f16(_Float16 *restrict C, const float *restrict acc, size_t n) {
    size_t i = 0;
#ifdef ROCKET_HAVE_NEON
    for (; i + 8 <= n; i += 8) {
        float16x4_t lo = vcvt_f16_f32(vld1q_f32(acc + i));
        float16x4_t hi = vcvt_f16_f32(vld1q_f32(acc + i + 4));
        vst1q_f16((__fp16 *)(C + i), vcombine_f16(lo, hi));
    }
#endif
    for (; i < n; i++) C[i] = (_Float16)acc[i];   /* tail + non-NEON build */
}

/* KACC readback de-tile. The K-accumulation already happened on the NPU (eltwise add),
 * so each output tile's cube is read ONCE and de-tiled straight into the caller's fp16
 * `C` — a pure reorder gather (no accumulate, no convert): the non-accumulating
 * fp16->fp16 sibling of detile_accum_f16. Same C2=8 feat_idx layout: for a fixed row h,
 * group g=(nn-1)/8 is 8 CONTIGUOUS fp16 at slot[g*H*8 + 8*(h-1) + 0..7], landing at 8
 * CONTIGUOUS output columns -> one 128-bit fp16 load -> one 128-bit fp16 store. Nt is a
 * multiple of 16 (N%16==0) so Ntile is always %8 — the scalar tail is a safety net, not
 * the steady state. Bit-identical to the scalar `C[..] = slot[feat_idx(Mtile,nn,h)]` it
 * replaces (same elements, same order), so the cosine-sim matrix + mm_nt_det gate it. */
static inline void detile_store_f16(_Float16 *restrict C, int N,
                                    const _Float16 *restrict slot,
                                    int m0, int n0, int Mtile, int Ntile) {
    const int H = Mtile;
    for (int h = 1; h <= Mtile; h++) {
        int g = 0;
#ifdef ROCKET_HAVE_NEON
        const _Float16 *s = slot + 8 * (size_t)(h - 1);
        _Float16 *d = C + (size_t)(m0 + h - 1) * N + n0;
        const int ng = Ntile / 8;
        for (; g < ng; g++)
            vst1q_f16((__fp16 *)(d + (size_t)g * 8),
                      vld1q_f16((const __fp16 *)(s + (size_t)g * H * 8)));
        g *= 8;   /* columns already done = ng*8 */
#endif
        for (int nn = g + 1; nn <= Ntile; nn++)   /* scalar tail (and the non-NEON build) */
            C[(size_t)(m0 + h - 1) * N + (n0 + nn - 1)] = slot[feat_idx(H, nn, h)];
    }
}

/* packA scatter — the INVERSE of detile_store_f16: one row-major A[m0.., k0..] tile ->
 * NPU input cube. For a fixed row h, group g=(c-1)/8 is 8 CONTIGUOUS fp16 in the
 * row-major source (arow[g*8 .. +7]) AND 8 CONTIGUOUS fp16 in the cube at
 * slot[g*H*8 + 8*(h-1) + 0..7] -> one 128-bit fp16 load -> one 128-bit fp16 store,
 * hoisting the per-element feat_idx divide/mod out of the hot loop (it was the bulk of
 * packA wall time). Bit-identical to the scalar
 * slot[feat_idx(H,c,h)] = A[(m0+h-1)*Astride + (k0+c-1)] it replaces (same elements,
 * same values), so the cosine-sim correctness matrix gates it. */
static inline void feat_scatter_tile(_Float16 *restrict slot, const _Float16 *restrict A,
                                     size_t Astride, int m0, int k0,
                                     int Mtile, int Ktile) {
    const int H = Mtile;
    for (int h = 1; h <= Mtile; h++) {
        const _Float16 *arow = A + (size_t)(m0 + h - 1) * Astride + k0;
        int g = 0;
#ifdef ROCKET_HAVE_NEON
        _Float16 *s = slot + 8 * (size_t)(h - 1);
        const int ng = Ktile / 8;
        for (; g < ng; g++)
            vst1q_f16((__fp16 *)(s + (size_t)g * H * 8),
                      vld1q_f16((const __fp16 *)(arow + (size_t)g * 8)));
        g *= 8;   /* columns already done = ng*8 */
#endif
        for (int c = g + 1; c <= Ktile; c++)   /* scalar tail (and the non-NEON build) */
            slot[feat_idx(H, c, h)] = arow[c - 1];
    }
}

/* packB scatter of ONE weight row `k` from a row-major source `brow` (already offset
 * to [.., k0]) into the (N/16,K/32,16,32) wt_idx cube. Each 32-wide K-block is 32
 * CONTIGUOUS fp16 in both source and cube -> four 128-bit fp16 copies per block,
 * hoisting the per-element wt_idx divide/mod out of the hot loop. Bit-identical to the
 * scalar slot[wt_idx(Ktile,k,c)] = brow[c-1]. Shared by wt_scatter_tile (contiguous
 * weight) and mm_pack_weights_seg (per-row segment-resolved weight). Ktile is normally
 * a multiple of 32 (K%32 offload constraint); the tail covers the remainder. */
static inline void wt_scatter_row(_Float16 *restrict slot, const _Float16 *restrict brow,
                                  int k, int Ktile) {
    const int C = Ktile;
    const size_t kbase = (size_t)((k - 1) / 16) * 16 * C + (size_t)((k - 1) % 16) * 32;
    int c = 1;
    for (; c + 31 <= Ktile; c += 32) {   /* full 32-block: 32 contiguous fp16 */
        _Float16 *d = slot + (size_t)((c - 1) / 32) * 32 * 16 + kbase;
        const _Float16 *sB = brow + (c - 1);
#ifdef ROCKET_HAVE_NEON
        vst1q_f16((__fp16 *)(d),      vld1q_f16((const __fp16 *)(sB)));
        vst1q_f16((__fp16 *)(d + 8),  vld1q_f16((const __fp16 *)(sB + 8)));
        vst1q_f16((__fp16 *)(d + 16), vld1q_f16((const __fp16 *)(sB + 16)));
        vst1q_f16((__fp16 *)(d + 24), vld1q_f16((const __fp16 *)(sB + 24)));
#else
        for (int j = 0; j < 32; j++) d[j] = sB[j];
#endif
    }
    for (; c <= Ktile; c++)   /* partial-block tail (and the non-NEON build) */
        slot[wt_idx(Ktile, k, c)] = brow[c - 1];
}

/* packB scatter of a contiguous B[N,K] weight tile -> wt_idx cube (the weight sibling
 * of feat_scatter_tile). Row-parallel over wt_scatter_row. */
static inline void wt_scatter_tile(_Float16 *restrict slot, const _Float16 *restrict B,
                                   size_t Bstride, int n0, int k0,
                                   int Ntile, int Ktile) {
    for (int k = 1; k <= Ntile; k++) {
        const _Float16 *brow = B + (size_t)(n0 + k - 1) * Bstride + k0;
        wt_scatter_row(slot, brow, k, Ktile);
    }
}

/* Upper bound on a single matmul dimension. The tile-count math below (M+Mt-1 etc.,
 * with Mt/Nt <= MAX_TILE and Kt <= 16384) is computed in signed int, so a dimension
 * within a tile of INT_MAX would overflow BEFORE the nm*nn*nk product guard catches it.
 * This cap sits far above any real matmul dim (~64x the largest model vocab/FFN) and far
 * below that overflow edge, so it only rejects pathological/adversarial inputs. */
#define ROCKET_MM_DIM_MAX (1 << 24)

/* ############################################################################
 * PART 2 — Tiling / planning and the pack-scatter phases (weights, input, bias)
 * ##########################################################################*/

int rocket_matmul_plan(int M, int K, int N, int *pMt, int *pKt, int *pNt)
{
    /* Machine parameters from the active hardware profile (chip-agnostic by
     * construction; RK3588 today), pulled into locals for the tiling math below. */
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const int MAX_TILE = hw->max_tile, CBUF_BANKS = hw->cbuf_banks;
    /* Reject non-positive / absurd dimensions before the int tile-count arithmetic. */
    if (M <= 0 || K <= 0 || N <= 0 ||
        M > ROCKET_MM_DIM_MAX || K > ROCKET_MM_DIM_MAX || N > ROCKET_MM_DIM_MAX)
        return ROCKET_E_SHAPE;
    /* M%4==0 only. M==1 (a height-1 GEMV) mis-computes on the HW height-1 conv
     * geometry, so this path rejects it; the cosine-sim correctness matrix gates
     * this. The one-shot rocket_matmul_fp16 pads M==1->4 before planning, so it never
     * asks the plan for M==1; every other caller gets -1 rather than a wrong result.
     * Single-vector callers: pad M to 4. */
    if (K % 32 || N % 16 || M % 4 != 0)
        return ROCKET_E_SHAPE;

    int Mt = (M < MAX_TILE) ? M : MAX_TILE;
    int Nt = (N < MAX_TILE) ? N : MAX_TILE;

    /* Mt/Nt overrides are applied FIRST so Kt is then maximized for the chosen
     * tile. Shrinking Mt/Nt lets Kt grow to fill the CBUF (12 banks * 32KB), which
     * is the lever that matters: the conv accumulates over K natively in one pass,
     * so a bigger Kt means fewer K-passes (nKt) and proportionally less readback. */
    const char *e;
    if ((e = getenv("ROCKET_MM_MT"))) { int v = atoi(e); if (v >= 4)  { Mt = (v/4)*4;   if (Mt > M) Mt = M; } }
    if ((e = getenv("ROCKET_MM_NT"))) { int v = atoi(e); if (v >= 16) { Nt = (v/16)*16; if (Nt > N) Nt = N; } }

    /* maximize Kt to fill the CBUF for the final Mt,Nt */
    int Kt = (K < 16384) ? K : 16384;
    Kt = (Kt / 32) * 32; if (Kt < 32) Kt = 32;
    while (Kt > 32 && banks_for(Mt, Kt) + banks_for(Nt, Kt) > CBUF_BANKS) Kt -= 32;

    /* Asymmetric-Nt heuristic (opt-in): when N AND K both tile, halving the FULL Nt frees
     * CBUF so Kt grows (the symmetric fit caps it), and the asymmetric Mt>Nt tile runs the
     * datapath more efficiently. Only fires on the symmetric default (no Nt override, Nt
     * still at the cap, N>cap) and only when the symmetric plan actually K-tiles (else a
     * bigger Kt is moot and the extra N-tiles are pure loss). Re-maximizes Kt for the halved
     * Nt. See asym_on(). The explicit MM_KT override below still wins if set. */
    if (asym_on() && !getenv("ROCKET_MM_NT") && Nt == MAX_TILE && N > MAX_TILE &&
        MAX_TILE / 2 >= 16 && (K + Kt - 1) / Kt > 1) {
        Nt = MAX_TILE / 2;
        Kt = (K < 16384) ? K : 16384;
        Kt = (Kt / 32) * 32; if (Kt < 32) Kt = 32;
        while (Kt > 32 && banks_for(Mt, Kt) + banks_for(Nt, Kt) > CBUF_BANKS) Kt -= 32;
    }

    /* if even Kt=32 won't fit alongside the tile, shrink Nt then Mt */
    while (banks_for(Mt, Kt) + banks_for(Nt, Kt) > CBUF_BANKS && Nt > 16) Nt -= 16;
    while (banks_for(Mt, Kt) + banks_for(Nt, Kt) > CBUF_BANKS && Mt > 4)  Mt -= 4;

    /* explicit Kt override (clamped to K and to what fits the CBUF) */
    if ((e = getenv("ROCKET_MM_KT"))) {
        int v = atoi(e);
        if (v >= 32) {
            Kt = (v/32)*32; if (Kt > (K/32)*32) Kt = (K/32)*32;
            while (Kt > 32 && banks_for(Mt, Kt) + banks_for(Nt, Kt) > CBUF_BANKS) Kt -= 32;
        }
    }

    if (pMt) *pMt = Mt;
    if (pKt) *pKt = Kt;
    if (pNt) *pNt = Nt;

    int nm = (M + Mt - 1) / Mt, nn = (N + Nt - 1) / Nt, nk = (K + Kt - 1) / Kt;
    /* M4: reject shapes whose int tile-count product would overflow before the
     * size_t BO math (and the `total = nMt*nNt*nKt` in the compute loops below). */
    if ((int64_t)nm * nn * nk > INT32_MAX) return ROCKET_E_SHAPE;
    return nm * nn * nk;
}

/* ---- reusable phases (shared with rocket_prepacked.c) ---- */

int mm_plan_init(mm_plan *pl, int M, int K, int N)
{
    int Mt, Kt, Nt;
    if (rocket_matmul_plan(M, K, N, &Mt, &Kt, &Nt) < 0)
        return -1;
    pl->M = M; pl->K = K; pl->N = N;
    pl->Mt = Mt; pl->Kt = Kt; pl->Nt = Nt;
    pl->nMt = (M + Mt - 1) / Mt;
    pl->nNt = (N + Nt - 1) / Nt;
    pl->nKt = (K + Kt - 1) / Kt;
    pl->in_slot  = (size_t)rup(Mt, 4)  * rup(Kt, 32);
    pl->wt_slot  = (size_t)rup(Nt, 16) * rup(Kt, 32);
    pl->out_slot = (size_t)rup(Mt, 4)  * rup(Nt, 16);
    return 0;
}

/* Tiling planner with a PINNED Nt and/or Kt — the cross-op-fused sibling of
 * rocket_matmul_plan. The pinned dim is fixed; the free dims (and a free Kt) shrink to
 * fit the CBUF. No ROCKET_MM_* env overrides (the pin is the contract). See
 * mm_plan_init_pin. Returns the job count, or ROCKET_E_SHAPE if it cannot fit. */
static int matmul_plan_pin(int M, int K, int N, int pin_Nt, int pin_Kt,
                           int *pMt, int *pKt, int *pNt)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const int MAX_TILE = hw->max_tile, CBUF_BANKS = hw->cbuf_banks;
    if (M <= 0 || K <= 0 || N <= 0 ||
        M > ROCKET_MM_DIM_MAX || K > ROCKET_MM_DIM_MAX || N > ROCKET_MM_DIM_MAX)
        return ROCKET_E_SHAPE;
    if (K % 32 || N % 16 || M % 4 != 0) return ROCKET_E_SHAPE;
    if (pin_Nt && (pin_Nt % 32 || pin_Nt > N)) return ROCKET_E_SHAPE;
    if (pin_Kt && (pin_Kt % 32 || pin_Kt > K)) return ROCKET_E_SHAPE;

    int Mt = (M < MAX_TILE) ? M : MAX_TILE;
    int Nt = pin_Nt ? pin_Nt : ((N < MAX_TILE) ? N : MAX_TILE);
    int Kt;
    if (pin_Kt) {
        Kt = pin_Kt;
    } else {
        Kt = (K < 16384) ? K : 16384;
        Kt = (Kt / 32) * 32; if (Kt < 32) Kt = 32;
        while (Kt > 32 && banks_for(Mt, Kt) + banks_for(Nt, Kt) > CBUF_BANKS) Kt -= 32;
    }
    /* shrink the FREE dims (never a pinned one) until the tile fits the CBUF */
    while (banks_for(Mt, Kt) + banks_for(Nt, Kt) > CBUF_BANKS) {
        if (!pin_Nt && Nt > 16)      Nt -= 16;
        else if (Mt > 4)             Mt -= 4;
        else if (!pin_Kt && Kt > 32) Kt -= 32;
        else return ROCKET_E_SHAPE;   /* pinned dims alone overflow — caller falls back */
    }

    if (pMt) *pMt = Mt;
    if (pKt) *pKt = Kt;
    if (pNt) *pNt = Nt;

    int nm = (M + Mt - 1) / Mt, nn = (N + Nt - 1) / Nt, nk = (K + Kt - 1) / Kt;
    if ((int64_t)nm * nn * nk > INT32_MAX) return ROCKET_E_SHAPE;
    return nm * nn * nk;
}

int mm_plan_init_pin(mm_plan *pl, int M, int K, int N, int pin_Nt, int pin_Kt)
{
    int Mt, Kt, Nt;
    if (matmul_plan_pin(M, K, N, pin_Nt, pin_Kt, &Mt, &Kt, &Nt) < 0)
        return -1;
    pl->M = M; pl->K = K; pl->N = N;
    pl->Mt = Mt; pl->Kt = Kt; pl->Nt = Nt;
    pl->nMt = (M + Mt - 1) / Mt;
    pl->nNt = (N + Nt - 1) / Nt;
    pl->nKt = (K + Kt - 1) / Kt;
    pl->in_slot  = (size_t)rup(Mt, 4)  * rup(Kt, 32);
    pl->wt_slot  = (size_t)rup(Nt, 16) * rup(Kt, 32);
    pl->out_slot = (size_t)rup(Mt, 4)  * rup(Nt, 16);
    return 0;
}

size_t mm_cube_elems(const mm_plan *pl)
{
    return (size_t)pl->nMt * pl->nNt * pl->out_slot;
}

void mm_scatter_bias_cube(const mm_plan *pl, _Float16 *restrict dst, const _Float16 *restrict bias)
{
    const int Mt = pl->Mt, Nt = pl->Nt, N = pl->N, M = pl->M;
    const int nMt = pl->nMt, nNt = pl->nNt;
    const size_t out_slot = pl->out_slot;
    memset(dst, 0, mm_cube_elems(pl) * sizeof(_Float16));
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            _Float16 *slot = dst + (size_t)(mi * nNt + ni) * out_slot;   /* canonical tile order */
            /* feat_idx uses H=Mtile (the tile's real row count), matching how the DPU writes
             * the output cube + how mm_pack_input lays the input cube — so a ragged last
             * m-tile (Mtile<Mt, e.g. Whisper's M=1500) still aligns lane-for-lane. */
            for (int h = 1; h <= Mtile; h++)                            /* broadcast over rows */
                for (int ch = 1; ch <= Ntile; ch++)
                    slot[feat_idx(Mtile, ch, h)] = bias[(size_t)n0 + ch - 1];
        }
    }
}

int mm_bos_alloc(int fd, const mm_plan *pl, mm_bos *b)
{
    memset(b, 0, sizeof(*b));
    size_t in_sz  = (size_t)pl->nMt * pl->nKt * pl->in_slot  * sizeof(_Float16) + CBUF_BANK;
    size_t wt_sz  = (size_t)pl->nNt * pl->nKt * pl->wt_slot  * sizeof(_Float16) + CBUF_BANK;
    size_t rc_sz  = (size_t)BATCH * RC_STRIDE * sizeof(uint64_t);
    size_t out_sz = (size_t)BATCH * pl->out_slot * sizeof(_Float16) + CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096, &b->guard);             /* push allocs off IOVA 0 */
    ret |= rocket_bo_alloc(fd, rc_sz,  &b->regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &b->in_all);
    ret |= rocket_bo_alloc(fd, wt_sz,  &b->wt_all);
    ret |= rocket_bo_alloc(fd, out_sz, &b->out_all);
    if (ret) {
        ROCKET_LOGE("rocket_matmul: BO alloc failed (in=%zuMB wt=%zuMB out=%zuMB)\n",
                in_sz >> 20, wt_sz >> 20, out_sz >> 20);
        mm_bos_free(fd, b);
        return -1;
    }
    if (((b->in_all.dma_address + in_sz) | (b->wt_all.dma_address + wt_sz) |
         (b->out_all.dma_address + out_sz) | (b->regcmd.dma_address + rc_sz)) >> 32) {
        ROCKET_LOGE("rocket_matmul: a BO dma_address exceeds 32 bits\n");
        mm_bos_free(fd, b);
        return -1;
    }

    /* Capability flags fixed at alloc time from env; the compute dispatcher branches
     * on these (never a fresh getenv), so the chosen path always matches the scratch
     * that was actually allocated here. */
    b->has_pong = kacc_on();
    b->has_pipe = (getenv("ROCKET_MM_PIPE") != NULL);

    /* Right-sized KACC output ping-pong (okacc0 + pong), only when K-accum is
     * enabled. The kacc path issues one NPU job per K-tile and uses only nMt*nNt
     * output tiles per job (never the full BATCH), so these BOs are sized to
     * nMt*nNt — NOT BATCH. PREP_BO/FINI_BO cache-sync cost is proportional to BO
     * size, so this cuts the per-ki `sync` term by ~(nMt*nNt)/BATCH: measured
     * sync 127->15 ms and +18% fp16 throughput (455->539 GOP/s) on 512x3840x4096.
     * out_all stays BATCH-sized for the rare tiny-M mm_compute fallback. */
    if (b->has_pong) {
        int ktiles = pl->nMt * pl->nNt;
        if (ktiles > BATCH) ktiles = BATCH;
        if (ktiles < 1)     ktiles = 1;
        /* Diagnostic/bisect escape hatch: force the old full-BATCH output sizing
         * (pre-right-size behavior) to A/B the sync win or fall back on a regression. */
        if (getenv("ROCKET_KACC_BATCHOUT")) ktiles = BATCH;
        size_t outk_sz = (size_t)ktiles * pl->out_slot * sizeof(_Float16) + CBUF_BANK;
        if (rocket_bo_alloc(fd, outk_sz, &b->okacc0) ||
            (b->okacc0.dma_address + outk_sz) >> 32 ||
            rocket_bo_alloc(fd, outk_sz, &b->pong) ||
            (b->pong.dma_address + outk_sz) >> 32) {
            ROCKET_LOGE("rocket_matmul: KACC output BO alloc failed\n");
            mm_bos_free(fd, b);
            return -1;
        }
    }

    /* 2nd regcmd + 2nd output BO for the mm_compute_pipe double buffer, only under
     * ROCKET_MM_PIPE. Neither needs pre-zeroing (regcmd2 is
     * overwritten per batch; out2 is fully written by the NPU and only its live
     * tiles are read back). */
    if (b->has_pipe) {
        if (rocket_bo_alloc(fd, rc_sz,  &b->regcmd2) ||
            rocket_bo_alloc(fd, out_sz, &b->out2) ||
            ((b->regcmd2.dma_address + rc_sz) | (b->out2.dma_address + out_sz)) >> 32) {
            ROCKET_LOGE("rocket_matmul: PIPE BO alloc failed\n");
            mm_bos_free(fd, b);
            return -1;
        }
    }

    /* Zero the input/weight scratch ONCE here. For a fixed plan the scatter
     * writes the SAME live tile positions on every (re)pack and never touches the
     * padding, so padding zeroed now STAYS zero across all reuses of these
     * (persistent) BOs — the per-call pack then skips a full-BO memset. out_all is
     * fully (re)written by the NPU and only its live region is read back; regcmd is
     * overwritten per batch — neither needs zeroing. */
    if (rocket_bo_prep(fd, &b->in_all, 1, 0) != 0) { mm_bos_free(fd, b); return -1; }
    memset(b->in_all.ptr, 0, b->in_all.size);
    rocket_bo_fini(fd, &b->in_all);
    if (rocket_bo_prep(fd, &b->wt_all, 1, 0) != 0) { mm_bos_free(fd, b); return -1; }
    memset(b->wt_all.ptr, 0, b->wt_all.size);
    rocket_bo_fini(fd, &b->wt_all);
    b->prezeroed = 1;

    /* Allocate the resident per-job compute scratch ONCE (BATCH-sized, fixed;
     * tiny). `acc` is left NULL and sized lazily on first mm_compute. */
    b->tasks  = malloc(BATCH * sizeof(*b->tasks));
    b->bm0    = malloc(BATCH * sizeof(int));
    b->bn0    = malloc(BATCH * sizeof(int));
    b->bMtile = malloc(BATCH * sizeof(int));
    b->bNtile = malloc(BATCH * sizeof(int));
    b->bmi    = malloc(BATCH * sizeof(int));
    b->bni    = malloc(BATCH * sizeof(int));
    b->boff   = malloc(BATCH * sizeof(size_t));
    b->submit_dt = malloc(rocket_submit_scratch_size(BATCH));   /* resident submit scratch */
    if (!b->tasks || !b->bm0 || !b->bn0 || !b->bMtile || !b->bNtile ||
        !b->bmi || !b->bni || !b->boff || !b->submit_dt) {
        ROCKET_LOGE("rocket_matmul: scratch alloc failed\n");
        mm_bos_free(fd, b);
        return -1;
    }

    /* pipe metadata (2 slots) + 2nd task array, only under ROCKET_MM_PIPE. */
    if (b->has_pipe) {
        b->tasks2 = malloc(BATCH * sizeof(*b->tasks2));
        b->pm0    = malloc(2 * BATCH * sizeof(int));
        b->pn0    = malloc(2 * BATCH * sizeof(int));
        b->pMtile = malloc(2 * BATCH * sizeof(int));
        b->pNtile = malloc(2 * BATCH * sizeof(int));
        b->poff   = malloc(2 * BATCH * sizeof(size_t));
        if (!b->tasks2 || !b->pm0 || !b->pn0 || !b->pMtile || !b->pNtile || !b->poff) {
            ROCKET_LOGE("rocket_matmul: pipe scratch alloc failed\n");
            mm_bos_free(fd, b);
            return -1;
        }
    }
    return 0;
}

void mm_bos_free(int fd, mm_bos *b)
{
    rocket_bo_free(fd, &b->guard);
    rocket_bo_free(fd, &b->regcmd);
    rocket_bo_free(fd, &b->in_all);
    rocket_bo_free(fd, &b->wt_all);
    rocket_bo_free(fd, &b->out_all);
    rocket_bo_free(fd, &b->okacc0);
    rocket_bo_free(fd, &b->pong);
    /* resident host scratch */
    free(b->tasks);  b->tasks  = NULL;
    free(b->bm0);    b->bm0    = NULL;
    free(b->bn0);    b->bn0    = NULL;
    free(b->bMtile); b->bMtile = NULL;
    free(b->bNtile); b->bNtile = NULL;
    free(b->bmi);    b->bmi    = NULL;
    free(b->bni);    b->bni    = NULL;
    free(b->boff);   b->boff   = NULL;
    free(b->acc);    b->acc    = NULL;  b->acc_cap = 0;
    free(b->submit_dt); b->submit_dt = NULL;
    /* pipe scratch */
    rocket_bo_free(fd, &b->regcmd2);
    rocket_bo_free(fd, &b->out2);
    free(b->tasks2); b->tasks2 = NULL;
    free(b->pm0);    b->pm0    = NULL;
    free(b->pn0);    b->pn0    = NULL;
    free(b->pMtile); b->pMtile = NULL;
    free(b->pNtile); b->pNtile = NULL;
    free(b->poff);   b->poff   = NULL;
}

double mm_pack_weights(int fd, const mm_plan *pl, mm_bos *b, const _Float16 *B)
{
    double ts = now_ms();
    if (rocket_bo_prep(fd, &b->wt_all, 1, 0) != 0) return -1.0;
    /* Padding is already zero (mm_bos_alloc pre-zeroes; the scatter below never
     * writes padding lanes), so skip the per-call full-BO memset. The
     * guard keeps any future non-prezeroed allocator correct. */
    if (!b->prezeroed) memset(b->wt_all.ptr, 0, b->wt_all.size);
    for (int ni = 0; ni < pl->nNt; ni++) {
        int n0 = ni * pl->Nt, Ntile = (pl->N - n0 < pl->Nt) ? (pl->N - n0) : pl->Nt;
        for (int ki = 0; ki < pl->nKt; ki++) {
            int k0 = ki * pl->Kt, Ktile = (pl->K - k0 < pl->Kt) ? (pl->K - k0) : pl->Kt;
            _Float16 *slot = (_Float16 *)b->wt_all.ptr + (size_t)(ni * pl->nKt + ki) * pl->wt_slot;
            wt_scatter_tile(slot, B, pl->K, n0, k0, Ntile, Ktile);
        }
    }
    rocket_bo_fini(fd, &b->wt_all);
    return now_ms() - ts;
}

/* Like mm_pack_weights, but scatter a worker's column slice of a CONCATENATED
 * weight (fused matmul). The worker owns combined-N global columns
 * [worker_g0, worker_g0 + pl->N); each global column is resolved to its source
 * segment and scattered into the SAME (N/16,K/32,16,32) tile layout as
 * mm_pack_weights. Avoids materializing a concatenated [sumN,K] B on the host.
 * Writes only live lanes (padding stays zero from the prezeroed BO). Returns ms. */
double mm_pack_weights_seg(int fd, const mm_plan *pl, mm_bos *b,
                           const mm_wt_seg *segs, int nseg, int worker_g0)
{
    double ts = now_ms();
    if (rocket_bo_prep(fd, &b->wt_all, 1, 0) != 0) return -1.0;
    if (!b->prezeroed) memset(b->wt_all.ptr, 0, b->wt_all.size);
    for (int ni = 0; ni < pl->nNt; ni++) {
        int n0 = ni * pl->Nt, Ntile = (pl->N - n0 < pl->Nt) ? (pl->N - n0) : pl->Nt;
        for (int ki = 0; ki < pl->nKt; ki++) {
            int k0 = ki * pl->Kt, Ktile = (pl->K - k0 < pl->Kt) ? (pl->K - k0) : pl->Kt;
            _Float16 *slot = (_Float16 *)b->wt_all.ptr + (size_t)(ni * pl->nKt + ki) * pl->wt_slot;
            for (int kk = 1; kk <= Ntile; kk++) {
                /* worker-local column (n0+kk-1) -> combined-N global column -> segment */
                int g = worker_g0 + n0 + (kk - 1);
                const _Float16 *Brow = NULL;
                for (int s = 0; s < nseg; s++)
                    if (g >= segs[s].g0 && g < segs[s].g0 + segs[s].Nseg) {
                        Brow = segs[s].Bseg + (size_t)(g - segs[s].g0) * pl->K;
                        break;
                    }
                if (!Brow) continue;     /* column beyond all segments: leave zero */
                wt_scatter_row(slot, Brow + k0, kk, Ktile);
            }
        }
    }
    rocket_bo_fini(fd, &b->wt_all);
    return now_ms() - ts;
}

double mm_pack_input(int fd, const mm_plan *pl, mm_bos *b, const _Float16 *A)
{
    double ts = now_ms();
    if (rocket_bo_prep(fd, &b->in_all, 1, 0) != 0) return -1.0;
    /* Padding already zero (see mm_pack_weights); skip the per-call memset. */
    if (!b->prezeroed) memset(b->in_all.ptr, 0, b->in_all.size);
    for (int mi = 0; mi < pl->nMt; mi++) {
        int m0 = mi * pl->Mt, Mtile = (pl->M - m0 < pl->Mt) ? (pl->M - m0) : pl->Mt;
        for (int ki = 0; ki < pl->nKt; ki++) {
            int k0 = ki * pl->Kt, Ktile = (pl->K - k0 < pl->Kt) ? (pl->K - k0) : pl->Kt;
            _Float16 *slot = (_Float16 *)b->in_all.ptr + (size_t)(mi * pl->nKt + ki) * pl->in_slot;
            feat_scatter_tile(slot, A, pl->K, m0, k0, Mtile, Ktile);
        }
    }
    rocket_bo_fini(fd, &b->in_all);
    return now_ms() - ts;
}

/* Size (in fp16 elems) of the used region of in_all for a plan — the canonical
 * packed-A layout, shared by all workers when their plans agree. */
size_t mm_input_elems(const mm_plan *pl)
{
    return (size_t)pl->nMt * pl->nKt * pl->in_slot;
}

/* The bare A[M,K] -> NPU-tile scatter (the expensive feat_idx index math), into a
 * plain CPU buffer `dst` (no BO/prep). Writes only the live tile lanes; padding is
 * left untouched, so the caller owns zeroing it. dst spans mm_input_elems(pl). */
static void feat_scatter_into(const mm_plan *pl, _Float16 *restrict dst, const _Float16 *restrict A)
{
    for (int mi = 0; mi < pl->nMt; mi++) {
        int m0 = mi * pl->Mt, Mtile = (pl->M - m0 < pl->Mt) ? (pl->M - m0) : pl->Mt;
        for (int ki = 0; ki < pl->nKt; ki++) {
            int k0 = ki * pl->Kt, Ktile = (pl->K - k0 < pl->Kt) ? (pl->K - k0) : pl->Kt;
            _Float16 *slot = dst + (size_t)(mi * pl->nKt + ki) * pl->in_slot;
            feat_scatter_tile(slot, A, pl->K, m0, k0, Mtile, Ktile);
        }
    }
}

/* The bare B[N,K] -> NPU weight-tile scatter (the wt_idx (N/16,K/32,16,32) index
 * math), into a plain CPU buffer `dst` (no BO/prep). The weight sibling of
 * feat_scatter_into: writes only the live tile lanes (padding left untouched, the
 * caller owns zeroing it). dst spans nNt*nKt*wt_slot fp16 elems. Identical scatter
 * to mm_pack_weights' inner loop, factored so the batched matmul can pack many
 * items' weights into one BO at per-item offsets. */
static void wt_scatter_into(const mm_plan *pl, _Float16 *restrict dst, const _Float16 *restrict B)
{
    for (int ni = 0; ni < pl->nNt; ni++) {
        int n0 = ni * pl->Nt, Ntile = (pl->N - n0 < pl->Nt) ? (pl->N - n0) : pl->Nt;
        for (int ki = 0; ki < pl->nKt; ki++) {
            int k0 = ki * pl->Kt, Ktile = (pl->K - k0 < pl->Kt) ? (pl->K - k0) : pl->Kt;
            _Float16 *slot = dst + (size_t)(ni * pl->nKt + ki) * pl->wt_slot;
            wt_scatter_tile(slot, B, pl->K, n0, k0, Ntile, Ktile);
        }
    }
}

/* Scatter A ONCE into a CPU buffer so workers can memcpy it (>> repeating the
 * scatter). Zeroes `dst` first — use for a fresh/unzeroed buffer. */
double mm_pack_input_buf(const mm_plan *pl, _Float16 *dst, const _Float16 *A)
{
    double ts = now_ms();
    memset(dst, 0, mm_input_elems(pl) * sizeof(_Float16));
    feat_scatter_into(pl, dst, A);
    return now_ms() - ts;
}

/* Like mm_pack_input_buf but does NOT zero `dst` — the caller guarantees the
 * padding lanes are already zero (a persistent calloc'd buffer reused across
 * calls with a fixed plan: live lanes are fully overwritten, padding stays zero
 * from the one-time calloc). Skips the per-call full-buffer memset. */
double mm_scatter_input(const mm_plan *pl, _Float16 *dst, const _Float16 *A)
{
    double ts = now_ms();
    feat_scatter_into(pl, dst, A);
    return now_ms() - ts;
}

/* Load a pre-scattered canonical buffer into this worker's in_all BO (memcpy +
 * cache flush) — replaces the per-worker scatter when plans agree. */
double mm_load_input(int fd, const mm_plan *pl, mm_bos *b, const _Float16 *packed)
{
    double ts = now_ms();
    if (rocket_bo_prep(fd, &b->in_all, 1, 0) != 0) return -1.0;
    memcpy(b->in_all.ptr, packed, mm_input_elems(pl) * sizeof(_Float16));
    rocket_bo_fini(fd, &b->in_all);
    return now_ms() - ts;
}

/* ############################################################################
 * PART 3 — Core compute / submit: CPU-accum, reuse, pipe, and KACC phases
 *          (and their readback / de-tile gather)
 * ##########################################################################*/

int mm_compute(int fd, const mm_plan *pl, mm_bos *b, _Float16 *C, double t_pack)
{
    const int M = pl->M, K = pl->K, N = pl->N;
    const int Mt = pl->Mt, Kt = pl->Kt, Nt = pl->Nt;
    const int nMt = pl->nMt, nNt = pl->nNt, nKt = pl->nKt;
    const size_t in_slot = pl->in_slot, wt_slot = pl->wt_slot, out_slot = pl->out_slot;

    int ret = 0;
    int prof = mm_profile();
    double t_gen = 0, t_sync = 0, t_submit = 0, t_wait = 0, t_read = 0, ts;
    uint64_t npu_regs[256] = {0};

    /* Resident scratch (allocated once in mm_bos_alloc). `acc` is sized
     * lazily here and grown only on shape change; cleared per call (it accumulates
     * the K-partials, so it must start at zero — calloc's job, now an explicit
     * memset of the reused buffer). */
    size_t need = (size_t)M * N;
    if (b->acc_cap < need) {
        free(b->acc);
        b->acc = malloc(need * sizeof(float));
        b->acc_cap = b->acc ? need : 0;
    }
    float *acc            = b->acc;
    rocket_task_desc *tasks = b->tasks;
    int *bm0    = b->bm0;
    int *bn0    = b->bn0;
    int *bMtile = b->bMtile;
    int *bNtile = b->bNtile;
    size_t *boff = b->boff;
    if (!acc || !tasks || !bm0 || !bn0 || !bMtile || !bNtile || !boff) { ret = -1; goto done; }
    memset(acc, 0, need * sizeof(float));

    int batch_chained = mm_batch_chained();   /* stock per-task (0) vs contiguous one-kick (1) */
    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;

    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;

                if (nb == 0) { ts = now_ms(); rocket_bo_prep(fd, &b->regcmd, 1, 0);
                               t_sync += now_ms() - ts; }

                ts = now_ms();
                size_t out_off = (size_t)nb * out_slot;
                matmul_params_t p = {
                    .m = Mtile, .k = Ktile, .n = Ntile,
                    .input_dma  = (uint32_t)(b->in_all.dma_address +
                                  (size_t)(mi * nKt + ki) * in_slot * sizeof(_Float16)),
                    .weights_dma = (uint32_t)(b->wt_all.dma_address +
                                  (size_t)(ni * nKt + ki) * wt_slot * sizeof(_Float16)),
                    .output_dma = (uint32_t)(b->out_all.dma_address + out_off * sizeof(_Float16)),
                    .tasks = npu_regs, .fp32tofp16 = 1,
                };
                if ((ret = gen_matmul_fp16(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul: gen failed (%d)\n", ret); goto done;
                }
                if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto done; }
                mm_pack_regcmd(batch_chained, &b->regcmd, tasks, nb, npu_regs, p.task_count);
                bm0[nb] = m0; bn0[nb] = n0; bMtile[nb] = Mtile; bNtile[nb] = Ntile;
                boff[nb] = out_off;
                t_gen += now_ms() - ts;
                nb++; done_tiles++;

                if (nb == BATCH || done_tiles == total) {
                    mm_seal_chain(batch_chained, &b->regcmd, nb, tasks[0].regcmd_count);
                    ts = now_ms();
                    rocket_bo_fini(fd, &b->regcmd);                  /* flush regcmds       */
                    rocket_bo_prep(fd, &b->out_all, 1, 0);
                    rocket_bo_fini(fd, &b->out_all);                 /* hand output to device */
                    t_sync += now_ms() - ts;

                    uint32_t in_h[]  = { b->in_all.handle, b->wt_all.handle, b->regcmd.handle };
                    uint32_t out_h[] = { b->out_all.handle };
                    ts = now_ms();
                    /* batched flag tracks the regcmd layout: chained => one kick. */
                    ret = rocket_submit_tasks_pre(fd, b->submit_dt, tasks, nb, in_h, 3, out_h, 1, batch_chained);
                    t_submit += now_ms() - ts;
                    if (ret) goto done;

                    ts = now_ms();
                    ret = rocket_bo_prep(fd, &b->out_all, 0, rocket_wait_ns());
                    double w_ms = now_ms() - ts;
                    t_wait += w_ms;
                    if (ret) {
                        ROCKET_LOGE("rocket_matmul: WAIT TIMEOUT (%d) after %.0fms "
                                "fd=%d out_h=%u out_dma=0x%llx M=%d K=%d N=%d batch=%d tiles=%d/%d\n",
                                ret, w_ms, fd, b->out_all.handle,
                                (unsigned long long)b->out_all.dma_address,
                                M, K, N, nb, done_tiles, total);
                        goto done;
                    }
                    /* a wait that nearly hit the deadline = contention building toward
                     * the tip-over; surface it even on success. */
                    if (w_ms > 500.0)
                        ROCKET_LOGE("rocket_matmul: SLOW job %.0fms fd=%d M=%d K=%d N=%d batch=%d\n",
                                w_ms, fd, M, K, N, nb);

                    ts = now_ms();
                    _Float16 *ob = b->out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        _Float16 *slot = ob + boff[j];
                        detile_accum_f16(acc, N, slot, bm0[j], bn0[j],
                                         bMtile[j], bNtile[j]);
                    }
                    rocket_bo_fini(fd, &b->out_all);
                    t_read += now_ms() - ts;
                    nb = 0;
                }
            }
        }
    }

    if (prof) {
        pthread_mutex_lock(&g_prof_mu);
        mm_prof_arm_locked();
        g_prof.pack += t_pack; g_prof.gen += t_gen; g_prof.sync += t_sync;
        g_prof.submit += t_submit; g_prof.wait += t_wait; g_prof.read += t_read;
        g_prof.calls++;
        pthread_mutex_unlock(&g_prof_mu);
    }

    narrow_f32_to_f16(C, acc, (size_t)M * N);

done:
    /* Scratch (acc/tasks/bm0/.../boff) is resident in mm_bos — freed in
     * mm_bos_free, not here. */
    return ret;
}

/* ============================================================================
 * CBUF operand-reuse CPU-accum compute (ROCKET_REUSE=1 weight / 2 data).
 *
 * THE LOAD-BEARING CORRECTNESS CHECK for CBUF reuse. Structurally a clone of
 * mm_compute (CPU fp32-accum, read every K-partial back), but it (a) REORDERS the
 * tile loop so consecutive tasks in a job share the operand we want resident, and
 * (b) sets the WEIGHT_REUSE / DATA_REUSE CNA bit on every task after the first of
 * each same-operand run, so the CNA reads that operand from CBUF instead of
 * re-fetching it from DRAM. If the CBUF does NOT actually persist the operand
 * across our batched tasks the way the single-core, single-job model assumes,
 * the reuse tasks read STALE banks -> the de-tiled partial is wrong -> max_abs
 * blows up. So a PASS here proves the mechanism before any perf claim.
 *
 * Loop reorder (vs mm_compute's mi,ni,ki):
 *   WEIGHT_REUSE: ni-outer, ki-mid, mi-inner  -> consecutive tasks share (ni,ki)
 *                 weight tile; reuse depth = nMt.
 *   DATA_REUSE:   mi-outer, ki-mid, ni-inner  -> consecutive tasks share (mi,ki)
 *                 input tile;  reuse depth = nNt.
 * Either way, for a FIXED output element (mi,ni) the K-partials are still summed
 * into acc[] in ASCENDING ki order (ki sits above the innermost index), so the
 * fp32 result is BIT-IDENTICAL to mm_compute when the CBUF persists correctly.
 *
 * Reuse is gated on (1) not first-in-batch (a new job resets the CBUF) and (2) the
 * previous task sharing the same operand group AND identical Mtile/Ntile/Ktile
 * (so the bank split is identical and the resident tile is still valid). The
 * partition data=[0,fd_banks)/weight=[fd_banks,12) depends only on Mtile,Ktile;
 * guarding on all three dims is the safe superset. mm_compute is left untouched
 * (the oracle); ROCKET_REUSE=0 never routes here. ==========================*/
int mm_compute_reuse(int fd, const mm_plan *pl, mm_bos *b, _Float16 *C,
                     double t_pack, int mode)
{
    const int M = pl->M, K = pl->K, N = pl->N;
    const int Mt = pl->Mt, Kt = pl->Kt, Nt = pl->Nt;
    const int nMt = pl->nMt, nNt = pl->nNt, nKt = pl->nKt;
    const size_t in_slot = pl->in_slot, wt_slot = pl->wt_slot, out_slot = pl->out_slot;

    int ret = 0;
    int prof = mm_profile();
    double t_gen = 0, t_sync = 0, t_submit = 0, t_wait = 0, t_read = 0, ts;
    uint64_t npu_regs[256] = {0};

    size_t need = (size_t)M * N;
    if (b->acc_cap < need) {
        free(b->acc);
        b->acc = malloc(need * sizeof(float));
        b->acc_cap = b->acc ? need : 0;
    }
    float *acc            = b->acc;
    rocket_task_desc *tasks = b->tasks;
    int *bm0    = b->bm0;
    int *bn0    = b->bn0;
    int *bMtile = b->bMtile;
    int *bNtile = b->bNtile;
    size_t *boff = b->boff;
    if (!acc || !tasks || !bm0 || !bn0 || !bMtile || !bNtile || !boff) { ret = -1; goto done; }
    memset(acc, 0, need * sizeof(float));

    int total = nMt * nNt * nKt, nb = 0;
    /* previous task's operand-group key + dims (for the reuse decision). */
    int p_mi = -1, p_ni = -1, p_ki = -1, p_Mt = -1, p_Nt = -1, p_Kt = -1;

    for (int t = 0; t < total; t++) {
        int mi, ni, ki;
        if (mode == 1) {                 /* WEIGHT_REUSE: ni-outer, ki-mid, mi-inner */
            ni = t / (nKt * nMt); int rem = t % (nKt * nMt); ki = rem / nMt; mi = rem % nMt;
        } else {                         /* DATA_REUSE:   mi-outer, ki-mid, ni-inner */
            mi = t / (nKt * nNt); int rem = t % (nKt * nNt); ki = rem / nNt; ni = rem % nNt;
        }
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
        int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;

        if (nb == 0) { ts = now_ms(); rocket_bo_prep(fd, &b->regcmd, 1, 0);
                       t_sync += now_ms() - ts; }

        /* reuse only when not first-in-batch, same operand group, identical dims. */
        int dims_ok = (Mtile == p_Mt && Ntile == p_Nt && Ktile == p_Kt);
        int wreuse = 0, dreuse = 0;
        if (nb > 0 && dims_ok) {
            if (mode == 1 && ni == p_ni && ki == p_ki) wreuse = 1;   /* same weight tile */
            if (mode == 2 && mi == p_mi && ki == p_ki) dreuse = 1;   /* same input tile  */
        }

        ts = now_ms();
        size_t out_off = (size_t)nb * out_slot;
        matmul_params_t p = {
            .m = Mtile, .k = Ktile, .n = Ntile,
            .input_dma  = (uint32_t)(b->in_all.dma_address +
                          (size_t)(mi * nKt + ki) * in_slot * sizeof(_Float16)),
            .weights_dma = (uint32_t)(b->wt_all.dma_address +
                          (size_t)(ni * nKt + ki) * wt_slot * sizeof(_Float16)),
            .output_dma = (uint32_t)(b->out_all.dma_address + out_off * sizeof(_Float16)),
            .tasks = npu_regs, .fp32tofp16 = 1,
            .weight_reuse = wreuse, .data_reuse = dreuse,
        };
        if ((ret = gen_matmul_fp16(&p)) != 0) {
            ROCKET_LOGE("rocket_matmul(reuse): gen failed (%d)\n", ret); goto done;
        }
        if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto done; }
        memcpy((uint64_t *)b->regcmd.ptr + (size_t)nb * RC_STRIDE, npu_regs,
               (size_t)p.task_count * sizeof(uint64_t));
        tasks[nb].regcmd = (uint32_t)(b->regcmd.dma_address +
                                      (size_t)nb * RC_STRIDE * sizeof(uint64_t));
        tasks[nb].regcmd_count = p.task_count;
        bm0[nb] = m0; bn0[nb] = n0; bMtile[nb] = Mtile; bNtile[nb] = Ntile;
        boff[nb] = out_off;
        t_gen += now_ms() - ts;
        nb++;
        p_mi = mi; p_ni = ni; p_ki = ki; p_Mt = Mtile; p_Nt = Ntile; p_Kt = Ktile;

        if (nb == BATCH || t == total - 1) {
            ts = now_ms();
            rocket_bo_fini(fd, &b->regcmd);
            rocket_bo_prep(fd, &b->out_all, 1, 0);
            rocket_bo_fini(fd, &b->out_all);
            t_sync += now_ms() - ts;

            uint32_t in_h[]  = { b->in_all.handle, b->wt_all.handle, b->regcmd.handle };
            uint32_t out_h[] = { b->out_all.handle };
            ts = now_ms();
            ret = rocket_submit_tasks_pre(fd, b->submit_dt, tasks, nb, in_h, 3, out_h, 1, 0);
            t_submit += now_ms() - ts;
            if (ret) goto done;

            ts = now_ms();
            ret = rocket_bo_prep(fd, &b->out_all, 0, rocket_wait_ns());
            double w_ms = now_ms() - ts;
            t_wait += w_ms;
            if (ret) {
                ROCKET_LOGE("rocket_matmul(reuse): WAIT TIMEOUT (%d) after %.0fms "
                        "M=%d K=%d N=%d batch=%d\n", ret, w_ms, M, K, N, nb);
                goto done;
            }
            if (w_ms > 500.0)
                ROCKET_LOGE("rocket_matmul(reuse): SLOW job %.0fms M=%d K=%d N=%d batch=%d\n",
                        w_ms, M, K, N, nb);

            ts = now_ms();
            _Float16 *ob = b->out_all.ptr;
            for (int j = 0; j < nb; j++) {
                _Float16 *slot = ob + boff[j];
                detile_accum_f16(acc, N, slot, bm0[j], bn0[j],
                                 bMtile[j], bNtile[j]);
            }
            rocket_bo_fini(fd, &b->out_all);
            t_read += now_ms() - ts;
            nb = 0;
            /* new job next batch => CBUF reset; invalidate the reuse-prev key. */
            p_mi = p_ni = p_ki = p_Mt = p_Nt = p_Kt = -1;
        }
    }

    if (prof) {
        pthread_mutex_lock(&g_prof_mu);
        mm_prof_arm_locked();
        g_prof.pack += t_pack; g_prof.gen += t_gen; g_prof.sync += t_sync;
        g_prof.submit += t_submit; g_prof.wait += t_wait; g_prof.read += t_read;
        g_prof.calls++;
        pthread_mutex_unlock(&g_prof_mu);
    }

    narrow_f32_to_f16(C, acc, (size_t)M * N);

done:
    return ret;
}

/* ============================================================================
 * Software-pipelined CPU-accum compute (ROCKET_MM_PIPE=1).
 *
 * mm_compute runs each tile batch strictly serially: build regcmd -> submit ->
 * WAIT (CPU idle) -> de-tile/accumulate. The host de-tile (∝ M·N·nKt) is the
 * CPU-accum path's dominant cost and sits entirely OUTSIDE NPU compute. This
 * sibling double-buffers the regcmd + output BOs so that, while the NPU computes
 * batch N, the CPU de-tiles batch N-1 and builds batch N+1's regcmd — hiding the
 * readback behind compute. The de-tile is index-math (not bandwidth) bound, so it
 * overlaps NPU LPDDR traffic without much contention.
 *
 * Batches (groups of <=BATCH tiles in the SAME mi,ni,ki order as mm_compute) are
 * assigned alternating slots 0/1. Per-element accumulation order into acc[] is
 * preserved (batches de-tiled in submission order), so the fp32 result is
 * BIT-IDENTICAL to mm_compute (the oracle). One producer/consumer on one worker
 * thread — no locks. mm_compute stays byte-for-byte unchanged.
 *
 * Slot safety (2 regcmd + 2 out BOs): batch b uses slot b&1. We rebuild regcmd
 * slot (b+1)&1 only AFTER waiting batch b-1 (which used that same slot), so the
 * NPU is never reading a regcmd BO we overwrite; likewise out slot (b-1)&1 is
 * de-tiled before batch b+1 reuses it. ==========================================*/

/* Build one batch into `slot` (regcmd BO + that slot's task array + metadata),
 * consuming up to BATCH tiles from the linear iterator *t_iter. Returns the tile
 * count (0 = none left), or -1 on gen failure. */
static int pipe_build_batch(int fd, const mm_plan *pl, mm_bos *b, int slot,
                            int *t_iter, double *t_gen, double *t_sync)
{
    const int M = pl->M, K = pl->K, N = pl->N;
    const int Mt = pl->Mt, Kt = pl->Kt, Nt = pl->Nt;
    const int nNt = pl->nNt, nKt = pl->nKt;
    const size_t in_slot = pl->in_slot, wt_slot = pl->wt_slot, out_slot = pl->out_slot;
    const int total = pl->nMt * nNt * nKt;

    rocket_bo *rcbuf = slot ? &b->regcmd2 : &b->regcmd;
    rocket_bo *obuf  = slot ? &b->out2    : &b->out_all;
    rocket_task_desc *tk = slot ? b->tasks2 : b->tasks;
    int *pm0 = b->pm0 + slot * BATCH, *pn0 = b->pn0 + slot * BATCH;
    int *pMt = b->pMtile + slot * BATCH, *pNt = b->pNtile + slot * BATCH;
    size_t *poff = b->poff + slot * BATCH;
    uint64_t npu_regs[256] = {0};
    double ts;

    ts = now_ms(); rocket_bo_prep(fd, rcbuf, 1, 0); *t_sync += now_ms() - ts;

    int nb = 0;
    ts = now_ms();
    while (*t_iter < total && nb < BATCH) {
        int t = *t_iter;
        int mi = t / (nNt * nKt), rem = t % (nNt * nKt), ni = rem / nKt, ki = rem % nKt;
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
        int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
        size_t out_off = (size_t)nb * out_slot;
        matmul_params_t p = {
            .m = Mtile, .k = Ktile, .n = Ntile,
            .input_dma  = (uint32_t)(b->in_all.dma_address +
                          (size_t)(mi * nKt + ki) * in_slot * sizeof(_Float16)),
            .weights_dma = (uint32_t)(b->wt_all.dma_address +
                          (size_t)(ni * nKt + ki) * wt_slot * sizeof(_Float16)),
            .output_dma = (uint32_t)(obuf->dma_address + out_off * sizeof(_Float16)),
            .tasks = npu_regs, .fp32tofp16 = 1,
        };
        if (gen_matmul_fp16(&p) != 0) {
            ROCKET_LOGE("rocket_matmul(pipe): gen failed\n");
            *t_gen += now_ms() - ts;
            return -1;
        }
        if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) return -1;
        memcpy((uint64_t *)rcbuf->ptr + (size_t)nb * RC_STRIDE, npu_regs,
               (size_t)p.task_count * sizeof(uint64_t));
        tk[nb].regcmd = (uint32_t)(rcbuf->dma_address + (size_t)nb * RC_STRIDE * sizeof(uint64_t));
        tk[nb].regcmd_count = p.task_count;
        pm0[nb] = m0; pn0[nb] = n0; pMt[nb] = Mtile; pNt[nb] = Ntile; poff[nb] = out_off;
        nb++; (*t_iter)++;
    }
    *t_gen += now_ms() - ts;

    ts = now_ms(); rocket_bo_fini(fd, rcbuf); *t_sync += now_ms() - ts;
    b->pnb[slot] = nb;
    return nb;
}

/* Wait for `slot`'s output BO and accumulate its batch into acc[]. */
static int pipe_detile(int fd, const mm_plan *pl, mm_bos *b, int slot, float *acc,
                       double *t_wait, double *t_read)
{
    const int N = pl->N;
    rocket_bo *obuf = slot ? &b->out2 : &b->out_all;
    double ts = now_ms();
    int ret = rocket_bo_prep(fd, obuf, 0, rocket_wait_ns());
    double w_ms = now_ms() - ts;
    *t_wait += w_ms;
    if (ret) {
        ROCKET_LOGE("rocket_matmul(pipe): WAIT TIMEOUT (%d) after %.0fms slot=%d\n",
                ret, w_ms, slot);
        return ret;
    }
    if (w_ms > 500.0)
        ROCKET_LOGE("rocket_matmul(pipe): SLOW job %.0fms slot=%d\n", w_ms, slot);

    ts = now_ms();
    int *pm0 = b->pm0 + slot * BATCH, *pn0 = b->pn0 + slot * BATCH;
    int *pMt = b->pMtile + slot * BATCH, *pNt = b->pNtile + slot * BATCH;
    size_t *poff = b->poff + slot * BATCH;
    _Float16 *ob = obuf->ptr;
    for (int j = 0; j < b->pnb[slot]; j++) {
        _Float16 *s = ob + poff[j];
        detile_accum_f16(acc, N, s, pm0[j], pn0[j], pMt[j], pNt[j]);
    }
    rocket_bo_fini(fd, obuf);
    *t_read += now_ms() - ts;
    return 0;
}

int mm_compute_pipe(int fd, const mm_plan *pl, mm_bos *b, _Float16 *C, double t_pack)
{
    const int M = pl->M, N = pl->N;

    int ret = 0;
    int prof = mm_profile();
    double t_gen = 0, t_sync = 0, t_submit = 0, t_wait = 0, t_read = 0, ts;

    /* resident fp32 accumulator (lazy-sized like mm_compute; cleared per call) */
    size_t need = (size_t)M * N;
    if (b->acc_cap < need) {
        free(b->acc);
        b->acc = malloc(need * sizeof(float));
        b->acc_cap = b->acc ? need : 0;
    }
    float *acc = b->acc;
    if (!acc || !b->regcmd2.ptr || !b->out2.ptr || !b->tasks2 ||
        !b->pm0 || !b->pn0 || !b->pMtile || !b->pNtile || !b->poff) { ret = -1; goto done; }
    memset(acc, 0, need * sizeof(float));

    rocket_bo *ob[2] = { &b->out_all, &b->out2 };
    rocket_bo *rcb[2] = { &b->regcmd, &b->regcmd2 };
    rocket_task_desc *pt[2] = { b->tasks, b->tasks2 };

    int t_iter = 0;
    int cur = 0;                                  /* slot of the batch to submit   */
    int cur_nb = pipe_build_batch(fd, pl, b, cur, &t_iter, &t_gen, &t_sync);
    if (cur_nb < 0) { ret = -1; goto done; }
    if (cur_nb == 0) goto finish;                 /* empty problem                 */

    int have_prev = 0, prev = 0;
    for (;;) {
        /* hand cur's output BO to the device and submit (async) */
        ts = now_ms();
        rocket_bo_prep(fd, ob[cur], 1, 0);
        rocket_bo_fini(fd, ob[cur]);
        t_sync += now_ms() - ts;
        uint32_t in_h[]  = { b->in_all.handle, b->wt_all.handle, rcb[cur]->handle };
        uint32_t out_h[] = { ob[cur]->handle };
        ts = now_ms();
        ret = rocket_submit_tasks_pre(fd, b->submit_dt, pt[cur], cur_nb, in_h, 3, out_h, 1, 0);
        t_submit += now_ms() - ts;
        if (ret) goto done;

        /* consume prev (frees its regcmd+out slot = the slot we build next into),
         * then build next — both overlap cur's NPU compute. */
        if (have_prev && (ret = pipe_detile(fd, pl, b, prev, acc, &t_wait, &t_read)))
            goto done;

        int next = cur ^ 1;
        int next_nb = pipe_build_batch(fd, pl, b, next, &t_iter, &t_gen, &t_sync);
        if (next_nb < 0) { ret = -1; goto done; }

        have_prev = 1; prev = cur;
        if (next_nb == 0) break;                  /* cur was the last batch        */
        cur = next; cur_nb = next_nb;
    }
    /* drain: de-tile the final in-flight batch */
    if ((ret = pipe_detile(fd, pl, b, prev, acc, &t_wait, &t_read))) goto done;

finish:
    narrow_f32_to_f16(C, acc, (size_t)M * N);

    if (prof) {
        pthread_mutex_lock(&g_prof_mu);
        mm_prof_arm_locked();
        g_prof.pack += t_pack; g_prof.gen += t_gen; g_prof.sync += t_sync;
        g_prof.submit += t_submit; g_prof.wait += t_wait; g_prof.read += t_read;
        g_prof.calls++;
        pthread_mutex_unlock(&g_prof_mu);
    }

done:
    return ret;
}

/* ============================================================================
 * NPU-side K-accumulation (ROCKET_KACC=1). Self-contained sibling of
 * mm_compute (the tuned CPU-accum path stays byte-for-byte unchanged = zero
 * regression; CPU-accum is the oracle). Instead of reading back every K-partial
 * and summing on the CPU (read ∝ M·N·nKt), the nKt partials of each output tile
 * are accumulated ON the NPU via the DPU eltwise-add (validated geometry from
 * matmul_accum_rocket), and only the FINAL tile is read back (read ∝ M·N).
 *
 * Loop order is ki-OUTER, tiles-inner: at each K-step we submit ALL tiles in the
 * current group as independent jobs (one fence), then advance ki. Within a tile
 * the K-chain is serial (each accumulate reads the previous partial), so we
 * PING-PONG between two output BOs (NEVER in-place: ERDMA-read and WDMA-write must
 * be different BOs). ki=0 is a plain conv (accumulate=0); ki>0 adds the prev buf.
 *
 * Tiles are processed in groups of <=BATCH so the regcmd BO and the two ping-pong
 * BOs stay bounded. The EW geometry needs Mtile>=12 (MAX(.,12) floor) — true for
 * all real prefill tiles (Mt=256); a tiny padded last M-tile would be off, so we
 * fall back to CPU-accum for any plan whose last M-tile is <12. ==========*/
int mm_compute_kacc(int fd, const mm_plan *pl, mm_bos *b, _Float16 *C, double t_pack)
{
    const int N = pl->N;
    const int Mt = pl->Mt, Kt = pl->Kt, Nt = pl->Nt;
    const int nMt = pl->nMt, nNt = pl->nNt, nKt = pl->nKt;
    const size_t in_slot = pl->in_slot, wt_slot = pl->wt_slot, out_slot = pl->out_slot;

    /* EW MAX(M,12) floor: a last M-tile < 12 would read a mis-strided operand.
     * Cheap, exact guard — let those (rare, tiny-M) shapes use the CPU oracle. */
    int lastM = pl->M - (nMt - 1) * Mt;
    /* ROCKET_E_TILING here is a routing sentinel, NOT a hard error: the caller
     * (and the prepacked workers / tests) treat == ROCKET_E_TILING as "fall back to
     * the CPU-accum mm_compute oracle for this tiny-M shape". */
    if (lastM < 12 && lastM != pl->M) return ROCKET_E_TILING;
    if (pl->M < 12 && nMt == 1 && nKt > 1) return ROCKET_E_TILING;

    int ret = 0;
    int prof = mm_profile();
    double t_gen = 0, t_sync = 0, t_submit = 0, t_wait = 0, t_read = 0, ts;

    /* Right-sized KACC ping-pong: prefer the resident okacc0/pong (sized to
     * nMt*nNt tiles by mm_bos_alloc — the per-ki sync win, see there). Fall back to
     * local right-sized allocs only if a caller ran kacc without provisioning them
     * (kacc normally runs only when has_pong, which allocates both). */
    rocket_bo ping_local = {0}, pong_local = {0};
    rocket_bo *ping = &b->okacc0, *pong = &b->pong;
    if (ping->handle == 0 || pong->handle == 0) {
        int ktiles = nMt * nNt;
        if (ktiles > BATCH) ktiles = BATCH;
        if (ktiles < 1)     ktiles = 1;
        size_t need = (size_t)ktiles * out_slot * sizeof(_Float16) + CBUF_BANK;
        if (ping->handle == 0) {
            if (rocket_bo_alloc(fd, need, &ping_local) < 0) return -1;
            ping = &ping_local;
        }
        if (pong->handle == 0) {
            if (rocket_bo_alloc(fd, need, &pong_local) < 0) {
                if (ping_local.handle) rocket_bo_free(fd, &ping_local);
                return -1;
            }
            pong = &pong_local;
        }
    }
    rocket_bo *bufs[2] = { ping, pong };

    /* Resident scratch (allocated once in mm_bos_alloc). */
    rocket_task_desc *tasks = b->tasks;
    int *bmi = b->bmi;
    int *bni = b->bni;
    if (!tasks || !bmi || !bni) { ret = -1; goto done; }

    int nTiles = nMt * nNt;
    uint64_t npu_regs[256] = {0};
    int batch_chained = mm_batch_chained();   /* gapped per-task (0) vs contiguous one-kick (1) */

    /* CBUF operand reuse. Each ki is a SEPARATE job (own submit+fence) =>
     * the CBUF resets between ki-steps, so reuse only ever spans tiles WITHIN one
     * ki-job and the prev-key resets each ki. Two compatible orderings within the
     * job (ki fixed): DATA_REUSE keeps the natural ni-innermost decode (consecutive
     * tiles share the (mi,ki) input, depth nNt); WEIGHT_REUSE swaps to mi-innermost
     * (consecutive tiles share the (ni,ki) weight, depth nMt). Output ping-pong is
     * unaffected — tile gi maps to the same (mi,ni) across all ki for either decode.
     * reuse_mode_for() picks the deeper run per shape (AUTO default); an explicit
     * ROCKET_REUSE forces one mode. */
    int rmode = reuse_mode_for(nMt, nNt);

    /* ── Cross-ki chaining (ROCKET_KACC_CHAIN=1) ──────────────────────────────
     * Replace the nKt fenced submits of one tile-group with a SINGLE self-chained
     * kick spanning the whole [ki][gi] sequence. The chain runs the tasks strictly
     * in BO order, so within the kick each ki>0 tile reads the prior ki's WDMA
     * output (the EW-add K-accumulation RAW) and the ping-pong WAR (ki and ki+2
     * share a buffer) both resolve under that serial order — exactly the
     * dependency the per-ki fences guarantee, now expressed in one fence.
     *
     * Layout order is ki-OUTER, gi-inner (matching the unchained loop): all g
     * tiles of ki=0 (plain conv), then all g tiles of ki=1 (+= ki=0's buffer),
     * etc. So a tile's ki(j-1) write always precedes its ki(j) read, and every
     * read of ki(j-1)'s buffer (by ki(j)) precedes ki(j+1)'s overwrite of it.
     *
     * Budget: nKt*g tasks share one regcmd BO + tasks[] + submit scratch (all
     * BATCH-sized) and the chained stride (count, <= RC_STRIDE) packs tighter than
     * the gapped reserve, so the single cap nKt*g <= BATCH keeps every buffer in
     * bounds. The win concentrates where nTiles*nKt is within a few BATCH (a
     * K-tiled moderate matmul does nKt under-filled submits today -> one full
     * kick); huge nTiles already fills each unchained submit, so it converges to
     * the same kick count. accumulate=0 (ki=0) and accumulate=1 (ki>0) emit the
     * same regcmd op count (the gen_matmul_task accumulate branches are
     * op-count-symmetric), so the uniform chained stride is valid across the ki
     * boundary. fp16 EW-add accumulation is immune to the int8 CACC-clear failure
     * that makes integer chaining HW-dead. */
    int chain_mode = kacc_chain_on();              /* 0 off, 1 adaptive, 2 force-all */
    int chain_engage = (nKt > 1 && nKt <= BATCH) &&
                       (chain_mode == 2 || (chain_mode == 1 && nKt * 3 <= BATCH));
    if (chain_engage) {
        int gcap = BATCH / nKt;                    /* tiles/group: nKt*gcap <= BATCH */
        if (gcap > nTiles) gcap = nTiles;
        for (int base = 0; base < nTiles; base += gcap) {
            int g = (nTiles - base < gcap) ? (nTiles - base) : gcap;
            int nt = 0;                            /* chained task slot counter */

            ts = now_ms(); rocket_bo_prep(fd, &b->regcmd, 1, 0); t_sync += now_ms() - ts;

            ts = now_ms();
            for (int ki = 0; ki < nKt; ki++) {
                int k0 = ki * Kt, Ktile = (pl->K - k0 < Kt) ? (pl->K - k0) : Kt;
                rocket_bo *dst = bufs[ki & 1], *src = bufs[(ki + 1) & 1];
                int p_mi = -1, p_ni = -1, p_Mt = -1, p_Nt = -1;  /* prev-key resets per ki-block */
                for (int gi = 0; gi < g; gi++) {
                    int tile = base + gi;
                    int mi = (rmode == 1) ? tile % nMt : tile / nNt;
                    int ni = (rmode == 1) ? tile / nMt : tile % nNt;
                    int m0 = mi * Mt, Mtile = (pl->M - m0 < Mt) ? (pl->M - m0) : Mt;
                    int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
                    size_t out_off = (size_t)gi * out_slot;
                    int dims_ok = (Mtile == p_Mt && Ntile == p_Nt);
                    int wreuse = (rmode == 1 && gi > 0 && dims_ok && ni == p_ni);
                    int dreuse = (rmode == 2 && gi > 0 && dims_ok && mi == p_mi);
                    matmul_params_t p = {
                        .m = Mtile, .k = Ktile, .n = Ntile,
                        .input_dma  = (uint32_t)(b->in_all.dma_address +
                                      (size_t)(mi * nKt + ki) * in_slot * sizeof(_Float16)),
                        .weights_dma = (uint32_t)(b->wt_all.dma_address +
                                      (size_t)(ni * nKt + ki) * wt_slot * sizeof(_Float16)),
                        .output_dma = (uint32_t)(dst->dma_address + out_off * sizeof(_Float16)),
                        .tasks = npu_regs, .fp32tofp16 = 1,
                        .accumulate = (ki > 0),
                        .add_dma = (ki > 0) ? (uint32_t)(src->dma_address + out_off * sizeof(_Float16)) : 0,
                        .weight_reuse = wreuse, .data_reuse = dreuse,
                    };
                    p_mi = mi; p_ni = ni; p_Mt = Mtile; p_Nt = Ntile;
                    if ((ret = gen_matmul_fp16(&p)) != 0) {
                        ROCKET_LOGE("rocket_matmul(kacc-chain): gen failed (%d)\n", ret); goto done;
                    }
                    if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto done; }
                    rkt_chain_pack(1, &b->regcmd, tasks, nt, npu_regs, p.task_count, (size_t)RC_STRIDE);
                    if (ki == 0) { bmi[gi] = mi; bni[gi] = ni; }   /* (mi,ni) per tile, ki-invariant */
                    nt++;
                }
            }
            rkt_chain_seal(1, &b->regcmd, nt, tasks[0].regcmd_count);
            t_gen += now_ms() - ts;

            ts = now_ms();
            rocket_bo_fini(fd, &b->regcmd);
            rocket_bo_prep(fd, ping, 1, 0); rocket_bo_fini(fd, ping);
            rocket_bo_prep(fd, pong, 1, 0); rocket_bo_fini(fd, pong);   /* both r/w in the kick */
            t_sync += now_ms() - ts;

            /* ping and pong are both WRITTEN by the kick (and read intra-kick by the
             * device via the EW-add chain); they are the job's outputs. List them only
             * in out_h — the intra-kick read is device-internal (the in-list is for
             * read-only inputs from OTHER jobs). A handle in BOTH lists makes the kernel
             * mis-handle the job (it signals completion but executes nothing). */
            uint32_t in_h[3]  = { b->in_all.handle, b->wt_all.handle, b->regcmd.handle };
            uint32_t out_h[2] = { ping->handle, pong->handle };
            ts = now_ms();
            ret = rocket_submit_tasks_pre(fd, b->submit_dt, tasks, nt, in_h, 3, out_h, 2, 1);
            t_submit += now_ms() - ts;
            if (ret) goto done;

            rocket_bo *fin = bufs[(nKt - 1) & 1];   /* final partial lands here */
            ts = now_ms();
            ret = rocket_bo_prep(fd, fin, 0, rocket_wait_ns());
            t_wait += now_ms() - ts;
            if (ret) {
                ROCKET_LOGE("rocket_matmul(kacc-chain): WAIT TIMEOUT (%d) base=%d g=%d nKt=%d "
                        "M=%d K=%d N=%d\n", ret, base, g, nKt, pl->M, pl->K, N);
                goto done;
            }
            ts = now_ms();
            _Float16 *ob = fin->ptr;
            for (int gi = 0; gi < g; gi++) {
                int mi = bmi[gi], ni = bni[gi];
                int m0 = mi * Mt, Mtile = (pl->M - m0 < Mt) ? (pl->M - m0) : Mt;
                int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
                _Float16 *slot = ob + (size_t)gi * out_slot;
                detile_store_f16(C, N, slot, m0, n0, Mtile, Ntile);
            }
            rocket_bo_fini(fd, fin);
            t_read += now_ms() - ts;
        }
        goto kacc_profile;
    }

    for (int base = 0; base < nTiles; base += BATCH) {
        int g = (nTiles - base < BATCH) ? (nTiles - base) : BATCH;

        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (pl->K - k0 < Kt) ? (pl->K - k0) : Kt;
            rocket_bo *dst = bufs[ki & 1], *src = bufs[(ki + 1) & 1];
            int p_mi = -1, p_ni = -1, p_Mt = -1, p_Nt = -1;   /* reuse prev-key (per ki-job) */

            ts = now_ms(); rocket_bo_prep(fd, &b->regcmd, 1, 0); t_sync += now_ms() - ts;

            ts = now_ms();
            for (int gi = 0; gi < g; gi++) {
                int tile = base + gi;
                int mi = (rmode == 1) ? tile % nMt : tile / nNt;
                int ni = (rmode == 1) ? tile / nMt : tile % nNt;
                int m0 = mi * Mt, Mtile = (pl->M - m0 < Mt) ? (pl->M - m0) : Mt;
                int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
                size_t out_off = (size_t)gi * out_slot;
                /* reuse vs the previous task in THIS ki-job (Ktile is constant within
                 * the job, so the dim guard only needs Mtile/Ntile). */
                int dims_ok = (Mtile == p_Mt && Ntile == p_Nt);
                int wreuse = (rmode == 1 && gi > 0 && dims_ok && ni == p_ni);
                int dreuse = (rmode == 2 && gi > 0 && dims_ok && mi == p_mi);
                matmul_params_t p = {
                    .m = Mtile, .k = Ktile, .n = Ntile,
                    .input_dma  = (uint32_t)(b->in_all.dma_address +
                                  (size_t)(mi * nKt + ki) * in_slot * sizeof(_Float16)),
                    .weights_dma = (uint32_t)(b->wt_all.dma_address +
                                  (size_t)(ni * nKt + ki) * wt_slot * sizeof(_Float16)),
                    .output_dma = (uint32_t)(dst->dma_address + out_off * sizeof(_Float16)),
                    .tasks = npu_regs, .fp32tofp16 = 1,
                    .accumulate = (ki > 0),
                    .add_dma = (ki > 0) ? (uint32_t)(src->dma_address + out_off * sizeof(_Float16)) : 0,
                    .weight_reuse = wreuse, .data_reuse = dreuse,
                };
                p_mi = mi; p_ni = ni; p_Mt = Mtile; p_Nt = Ntile;
                if ((ret = gen_matmul_fp16(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul(kacc): gen failed (%d)\n", ret); goto done;
                }
                if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto done; }
                /* Lay this ki-job's g tiles either gapped (per-task) or contiguous +
                 * self-chained (one kick). fp16 KACC accumulates via fp16 eltwise-add
                 * across ki-JOBS (separate fenced submits), not the int32 CACC, so
                 * chaining the independent tiles WITHIN a ki-job is accumulation-safe. */
                mm_pack_regcmd(batch_chained, &b->regcmd, tasks, gi, npu_regs, p.task_count);
                bmi[gi] = mi; bni[gi] = ni;
            }
            mm_seal_chain(batch_chained, &b->regcmd, g, tasks[0].regcmd_count);
            t_gen += now_ms() - ts;

            ts = now_ms();
            rocket_bo_fini(fd, &b->regcmd);
            rocket_bo_prep(fd, dst, 1, 0);
            rocket_bo_fini(fd, dst);               /* hand dst to device */
            t_sync += now_ms() - ts;

            /* in-list: input, weights, regcmd, (+ src when accumulating: ERDMA reads it) */
            uint32_t in_h[4] = { b->in_all.handle, b->wt_all.handle, b->regcmd.handle, src->handle };
            uint32_t out_h[] = { dst->handle };
            ts = now_ms();
            ret = rocket_submit_tasks_pre(fd, b->submit_dt, tasks, g, in_h, (ki > 0) ? 4 : 3, out_h, 1, batch_chained);
            t_submit += now_ms() - ts;
            if (ret) goto done;

            ts = now_ms();
            ret = rocket_bo_prep(fd, dst, 0, rocket_wait_ns());
            t_wait += now_ms() - ts;
            if (ret) {
                ROCKET_LOGE("rocket_matmul(kacc): WAIT TIMEOUT (%d) ki=%d/%d M=%d K=%d N=%d\n",
                        ret, ki, nKt, pl->M, pl->K, N);
                goto done;
            }
            rocket_bo_fini(fd, dst);
        }

        /* read each tile's FINAL buffer ONCE (read ∝ M·N, independent of nKt) */
        rocket_bo *fin = bufs[(nKt - 1) & 1];
        ts = now_ms();
        rocket_bo_prep(fd, fin, 0, 0);
        _Float16 *ob = fin->ptr;
        for (int gi = 0; gi < g; gi++) {
            int mi = bmi[gi], ni = bni[gi];
            int m0 = mi * Mt, Mtile = (pl->M - m0 < Mt) ? (pl->M - m0) : Mt;
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            _Float16 *slot = ob + (size_t)gi * out_slot;
            detile_store_f16(C, N, slot, m0, n0, Mtile, Ntile);
        }
        rocket_bo_fini(fd, fin);
        t_read += now_ms() - ts;
    }

kacc_profile:
    if (prof) {
        pthread_mutex_lock(&g_prof_mu);
        mm_prof_arm_locked();
        g_prof.pack += t_pack; g_prof.gen += t_gen; g_prof.sync += t_sync;
        g_prof.submit += t_submit; g_prof.wait += t_wait; g_prof.read += t_read;
        g_prof.calls++;
        pthread_mutex_unlock(&g_prof_mu);
    }

done:
    if (ping_local.handle) rocket_bo_free(fd, &ping_local);
    if (pong_local.handle) rocket_bo_free(fd, &pong_local);
    /* tasks/bmi/bni are resident in mm_bos — freed in mm_bos_free. */
    return ret;
}

/* Cross-op KACC compute: same NPU-side fp16 K-accumulation as mm_compute_kacc, but the
 * COMPLETE output cube is left in `cube` (canonical [nMt][nNt] tile order) for the next op
 * to read directly, with NO de-tile to row-major. See the header for the contract. */
int mm_compute_kacc_cube(int fd, const mm_plan *pl, mm_bos *b, rocket_bo *cube, double t_pack)
{
    const int N = pl->N;
    const int Mt = pl->Mt, Kt = pl->Kt, Nt = pl->Nt;
    const int nMt = pl->nMt, nNt = pl->nNt, nKt = pl->nKt;
    const size_t in_slot = pl->in_slot, wt_slot = pl->wt_slot, out_slot = pl->out_slot;
    (void)Nt;

    int nTiles = nMt * nNt;
    /* The full cube must be one NPU batch so every tile co-exists in `cube` (the
     * de-tiling mm_compute_kacc tolerates multi-batch by reading each batch before the
     * next reuses the ping-pong; here the whole cube has to survive). */
    if (nTiles > BATCH) return ROCKET_E_TILING;
    /* Same tiny-M EW MAX(M,12) stride floor guard as mm_compute_kacc. */
    int lastM = pl->M - (nMt - 1) * Mt;
    if (lastM < 12 && lastM != pl->M) return ROCKET_E_TILING;
    if (pl->M < 12 && nMt == 1 && nKt > 1) return ROCKET_E_TILING;

    if (cube->size < (size_t)nTiles * out_slot * sizeof(_Float16)) return ROCKET_E_NOMEM;

    int ret = 0;
    int prof = mm_profile();
    double t_gen = 0, t_sync = 0, t_submit = 0, t_wait = 0, t_read = 0, ts;

    /* Ping-pong across the nKt K-tile jobs. Arrange parity so the FINAL buffer (after
     * nKt-1 accumulations) is `cube`; the other slot is a local scratch BO (only needed
     * when nKt>1 — a single K-tile writes `cube` directly with no accumulation). */
    rocket_bo pong_local = {0};
    rocket_bo *bufs[2];
    int finidx = (nKt - 1) & 1;
    bufs[finidx] = cube;
    if (nKt > 1) {
        size_t need = (size_t)nTiles * out_slot * sizeof(_Float16) + CBUF_BANK;
        if (rocket_bo_alloc(fd, need, &pong_local) < 0) return -1;
        if ((pong_local.dma_address + need) >> 32) { rocket_bo_free(fd, &pong_local); return -1; }
        bufs[finidx ^ 1] = &pong_local;
    } else {
        bufs[finidx ^ 1] = cube;   /* unused (accumulate is always 0 at nKt==1) */
    }

    rocket_task_desc *tasks = b->tasks;
    int *bmi = b->bmi, *bni = b->bni;
    if (!tasks || !bmi || !bni) { ret = -1; goto done; }

    uint64_t npu_regs[256] = {0};
    int batch_chained = mm_batch_chained();
    /* DATA reuse (mode 2) is mandatory here, not a perf knob: it makes tile gi map to
     * (mi = gi/nNt, ni = gi%nNt) so the output offset gi*out_slot IS the canonical
     * (mi*nNt+ni)*out_slot the consumer's input cube expects. (Correctness-neutral and
     * gated bit-identical; see mm_compute_reuse.) */
    int g = nTiles;

    for (int ki = 0; ki < nKt; ki++) {
        int k0 = ki * Kt, Ktile = (pl->K - k0 < Kt) ? (pl->K - k0) : Kt;
        rocket_bo *dst = bufs[ki & 1], *src = bufs[(ki + 1) & 1];
        int p_mi = -1, p_Mt = -1, p_Nt = -1;

        ts = now_ms(); rocket_bo_prep(fd, &b->regcmd, 1, 0); t_sync += now_ms() - ts;

        ts = now_ms();
        for (int gi = 0; gi < g; gi++) {
            int mi = gi / nNt, ni = gi % nNt;            /* DATA-reuse decode == canonical */
            int m0 = mi * Mt, Mtile = (pl->M - m0 < Mt) ? (pl->M - m0) : Mt;
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            size_t out_off = (size_t)gi * out_slot;       /* == (mi*nNt+ni)*out_slot */
            int dims_ok = (Mtile == p_Mt && Ntile == p_Nt);
            int dreuse  = (gi > 0 && dims_ok && mi == p_mi);   /* consecutive ni share (mi,ki) input */
            matmul_params_t p = {
                .m = Mtile, .k = Ktile, .n = Ntile,
                .input_dma  = (uint32_t)(b->in_all.dma_address +
                              (size_t)(mi * nKt + ki) * in_slot * sizeof(_Float16)),
                .weights_dma = (uint32_t)(b->wt_all.dma_address +
                              (size_t)(ni * nKt + ki) * wt_slot * sizeof(_Float16)),
                .output_dma = (uint32_t)(dst->dma_address + out_off * sizeof(_Float16)),
                .tasks = npu_regs, .fp32tofp16 = 1,
                .accumulate = (ki > 0),
                .add_dma = (ki > 0) ? (uint32_t)(src->dma_address + out_off * sizeof(_Float16)) : 0,
                .data_reuse = dreuse,
            };
            p_mi = mi; p_Mt = Mtile; p_Nt = Ntile;
            if ((ret = gen_matmul_fp16(&p)) != 0) {
                ROCKET_LOGE("rocket_matmul(kacc_cube): gen failed (%d)\n", ret); goto done;
            }
            if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto done; }
            mm_pack_regcmd(batch_chained, &b->regcmd, tasks, gi, npu_regs, p.task_count);
            bmi[gi] = mi; bni[gi] = ni;
        }
        mm_seal_chain(batch_chained, &b->regcmd, g, tasks[0].regcmd_count);
        t_gen += now_ms() - ts;

        ts = now_ms();
        rocket_bo_fini(fd, &b->regcmd);
        rocket_bo_prep(fd, dst, 1, 0);
        rocket_bo_fini(fd, dst);
        t_sync += now_ms() - ts;

        uint32_t in_h[4] = { b->in_all.handle, b->wt_all.handle, b->regcmd.handle, src->handle };
        uint32_t out_h[] = { dst->handle };
        ts = now_ms();
        ret = rocket_submit_tasks_pre(fd, b->submit_dt, tasks, g, in_h, (ki > 0) ? 4 : 3, out_h, 1, batch_chained);
        t_submit += now_ms() - ts;
        if (ret) goto done;

        ts = now_ms();
        ret = rocket_bo_prep(fd, dst, 0, rocket_wait_ns());
        t_wait += now_ms() - ts;
        if (ret) {
            ROCKET_LOGE("rocket_matmul(kacc_cube): WAIT TIMEOUT (%d) ki=%d/%d M=%d K=%d N=%d\n",
                    ret, ki, nKt, pl->M, pl->K, N);
            goto done;
        }
        rocket_bo_fini(fd, dst);
    }

    /* Leave the complete cube CPU-visible (the on-cube element-wise op reads cube->ptr).
     * No de-tile: `cube` already holds the canonical tile cube the next matmul consumes. */
    ts = now_ms();
    rocket_bo_prep(fd, cube, 0, 0);
    t_read += now_ms() - ts;

    if (prof) {
        pthread_mutex_lock(&g_prof_mu);
        mm_prof_arm_locked();
        g_prof.pack += t_pack; g_prof.gen += t_gen; g_prof.sync += t_sync;
        g_prof.submit += t_submit; g_prof.wait += t_wait; g_prof.read += t_read;
        g_prof.calls++;
        pthread_mutex_unlock(&g_prof_mu);
    }

done:
    if (pong_local.handle) rocket_bo_free(fd, &pong_local);
    return ret;
}

/* ############################################################################
 * PART 4 — fp16 public API entry points (one-shot + same-shape batch)
 * ##########################################################################*/

/* ---- one-shot public entry point: pack weights + input + compute, per call ---- */
int rocket_matmul_fp16(int fd, int M, int K, int N,
                       const _Float16 *A, const _Float16 *B, _Float16 *C)
{
    /* dtype capability gate: a chip declares its usable datatype menu in the hw
     * profile. RK3588's mask is all-ones (the datatype matrix is complete), so this
     * never fires here; it states the chip-dependency by construction. */
    if (!rocket_hw_dtype_supported(rocket_hw_current(), precision_float16))
        return ROCKET_E_UNSUPPORTED;
    /* M==1 (single-vector GEMV) has a broken HW geometry at feature-height 1: the
     * matmul-as-1x1-conv produces wrong output below height 4 (M%4==0 is proven;
     * M=2,3 are rejected by the plan). Honor the documented single-vector case by
     * padding to a height-4 tile (3 zero rows) and returning only row 0 — cheap
     * (M=1 is tiny) and it reuses the validated M%4==0 path verbatim. The LLM
     * backend never reaches here at M<4 (ROCKET_MIN_M=4 routes GEMV to the CPU);
     * the multicore/streaming/prepacked paths do NOT pad and reject M==1. */
    if (M == 1)
        MM_PAD_M1(fd, K, N, A, B, C, _Float16, _Float16, rocket_matmul_fp16);
    mm_plan pl;
    if (mm_plan_init(&pl, M, K, N) < 0) {
        ROCKET_LOGE("rocket_matmul: unsupported shape M=%d K=%d N=%d\n", M, K, N);
        return -1;
    }
    mm_bos b;
    if (mm_bos_alloc(fd, &pl, &b) < 0) return -1;

    /* Report A- and B-pack separately so the profiler can tell whether pack time
     * is the redundant per-worker input scatter (lever 2) or unavoidable weight
     * scatter. mm_compute is told 0 so the total isn't double-counted. */
    double t_pa = mm_pack_input(fd, &pl, &b, A);
    double t_pb = mm_pack_weights(fd, &pl, &b, B);
    if (t_pa < 0 || t_pb < 0) { mm_bos_free(fd, &b); return -1; }  /* cache-sync failed (logged) */
    mm_prof_add_pack_kind(t_pa, 'A');
    mm_prof_add_pack_kind(t_pb, 'B');
    int ret;
    if (b.has_pong) {
        ret = mm_compute_kacc(fd, &pl, &b, C, 0.0);   /* NPU-side K-accum (reuse-aware) */
        if (ret == -2) ret = mm_compute(fd, &pl, &b, C, 0.0);  /* tiny-M: CPU oracle */
    } else if (reuse_policy() != 0) {
        /* Reached only with KACC off + ROCKET_REUSE forced (AUTO implies KACC implies
         * has_pong above), so reuse_mode_for resolves to that forced 1/2 here. */
        ret = mm_compute_reuse(fd, &pl, &b, C, 0.0, reuse_mode_for(pl.nMt, pl.nNt));
    } else if (b.has_pipe) {
        ret = mm_compute_pipe(fd, &pl, &b, C, 0.0);   /* pipelined overlap */
    } else {
        ret = mm_compute(fd, &pl, &b, C, 0.0);
    }

    mm_bos_free(fd, &b);
    return ret;
}

/* ============================================================================
 * Batched same-shape fp16 matmul — the attention dispatch-floor lever.
 *
 * Runs `nbatch` INDEPENDENT matmuls C[i] = A[i]·B[i]^T, all sharing one (M,K,N)
 * shape, as a SINGLE NPU job stream: every item's output tiles flow through one
 * rocket_submit_tasks_pre job (BATCH tiles per kick), so a caller with N small
 * matmuls pays ONE submit syscall + ONE fence wait for the whole group instead of
 * N submit/wait round-trips. With ROCKET_BATCH_SUBMIT=1 (and the coordinated kernel
 * half) the per-item regcmds also chain contiguously, so the group fires ONE
 * completion IRQ instead of N — the contiguous self-chaining mechanism, extended
 * from one matmul's tiles to a batch of same-shape matmuls.
 *
 * BIT-IDENTICAL to calling rocket_matmul_fp16 per item: one mm_plan for the shared
 * shape, the same fp16 tile pack, the same CPU fp32 K-accumulation in ascending ki
 * order, the same de-tile + narrow. So the flash-attention cos-sim gate that holds
 * for the per-head path holds here unchanged; this path is a submit-batching
 * rearrangement, not a numeric change. (It is its own CPU-fp32-accum sibling — it
 * does NOT consult ROCKET_KACC/REUSE/PIPE, which are the single-matmul perf
 * variants; the batch lever is orthogonal and stacks with ROCKET_BATCH_SUBMIT.)
 *
 * The use: flash attention fans the query heads across worker fds, and within a
 * worker the per-head QK matmuls share one shape (and separately the per-head AV
 * matmuls), so each batches into ONE job per worker per op — collapsing the per-head
 * submit+wait serialization that is the small-GEMM dispatch floor. The score/prob
 * matrices stay host-resident BETWEEN the QK and AV groups, so the additive mask +
 * (host) softmax sit between the two batched jobs — the host-softmax-compatible
 * chaining, NOT a chained QK->softmax->AV that would force the softmax back on-NPU.
 *
 * BO sizing: in_all/wt_all hold all nbatch items' packed tiles (the caller bounds
 * nbatch so this stays modest — flash attention caps the head group by score-matrix
 * size); out_all is the BATCH-tile rolling submit buffer, de-tiled per job-batch
 * into the per-item fp32 accumulators. Allocates its BOs per call (the caller holds
 * the fd, not these BOs). Requires M%4, K%32, N%16 (the plan rejects otherwise);
 * nbatch==1 falls through to the one-shot rocket_matmul_fp16 verbatim. Returns 0 or
 * a negative rocket_status.
 * ==========================================================================*/
/* Persistent batched-matmul context: the resident BOs + host scratch reused across
 * same-shape batch runs. in_all/wt_all/out_all grow only on a larger shape; the
 * regcmd BO + guard + the BATCH-sized host arrays are fixed (allocated once at
 * create). `prezeroed` + last_* track whether in_all/wt_all are zero AND match the
 * last layout, so a repeated (M,K,N,nbatch) skips the full-BO zero. One context
 * tracks ONE layout — a caller alternating shapes uses one context per shape. */
struct rocket_mm_batch {
    int fd;
    rocket_bo guard, regcmd, in_all, wt_all, out_all;
    int    *bit_, *bm0, *bn0, *bMtile, *bNtile;
    size_t *boff;
    rocket_task_desc *tasks;
    void   *submit_dt;
    float  *acc;
    size_t  acc_cap;                 /* allocated fp32 elem count in acc            */
    int     prezeroed;               /* in_all/wt_all zero AND match last_* layout  */
    int     last_M, last_K, last_N, last_nbatch;
};

/* Grow a BO to at least `need` bytes (free + realloc), re-checking the 32-bit IOVA
 * ceiling. Returns 1 if (re)allocated (the caller re-zeroes if it relies on zeroed
 * padding), 0 if the existing BO already fits, <0 on failure (BO left freed). */
static int mm_bo_ensure(int fd, rocket_bo *bo, size_t need)
{
    if (bo->handle && bo->size >= need) return 0;
    rocket_bo_free(fd, bo);
    if (rocket_bo_alloc(fd, need, bo) != 0) return ROCKET_E_NOMEM;
    if ((bo->dma_address + bo->size) >> 32) { rocket_bo_free(fd, bo); return ROCKET_E_DEVICE; }
    return 1;
}

rocket_mm_batch *rocket_mm_batch_create(int fd)
{
    rocket_mm_batch *b = calloc(1, sizeof *b);
    if (!b) return NULL;
    b->fd = fd;
    b->last_M = b->last_K = b->last_N = b->last_nbatch = -1;

    size_t rc_sz = (size_t)BATCH * RC_STRIDE * sizeof(uint64_t);
    if (rocket_bo_alloc(fd, 4096, &b->guard) != 0 ||           /* push allocs off IOVA 0 */
        rocket_bo_alloc(fd, rc_sz, &b->regcmd) != 0 ||
        (b->regcmd.dma_address + b->regcmd.size) >> 32) {
        rocket_mm_batch_free(b);
        return NULL;
    }
    b->bit_   = malloc((size_t)BATCH * sizeof(int));
    b->bm0    = malloc((size_t)BATCH * sizeof(int));
    b->bn0    = malloc((size_t)BATCH * sizeof(int));
    b->bMtile = malloc((size_t)BATCH * sizeof(int));
    b->bNtile = malloc((size_t)BATCH * sizeof(int));
    b->boff   = malloc((size_t)BATCH * sizeof(size_t));
    b->tasks  = malloc((size_t)BATCH * sizeof(*b->tasks));
    b->submit_dt = malloc(rocket_submit_scratch_size(BATCH));
    if (!b->bit_ || !b->bm0 || !b->bn0 || !b->bMtile || !b->bNtile ||
        !b->boff || !b->tasks || !b->submit_dt) {
        rocket_mm_batch_free(b);
        return NULL;
    }
    return b;
}

void rocket_mm_batch_free(rocket_mm_batch *b)
{
    if (!b) return;
    rocket_bo_free(b->fd, &b->out_all);
    rocket_bo_free(b->fd, &b->wt_all);
    rocket_bo_free(b->fd, &b->in_all);
    rocket_bo_free(b->fd, &b->regcmd);
    rocket_bo_free(b->fd, &b->guard);
    free(b->bit_); free(b->bm0); free(b->bn0); free(b->bMtile); free(b->bNtile);
    free(b->boff); free(b->tasks); free(b->submit_dt); free(b->acc);
    free(b);
}

int rocket_mm_batch_run(rocket_mm_batch *b, int M, int K, int N, int nbatch,
                        const _Float16 *const *A, const _Float16 *const *B,
                        _Float16 *const *C)
{
    if (!b) return ROCKET_E_SHAPE;
    if (nbatch < 1 || !A || !B || !C) return ROCKET_E_SHAPE;
    if (nbatch == 1) return rocket_matmul_fp16(b->fd, M, K, N, A[0], B[0], C[0]);
    if (!rocket_hw_dtype_supported(rocket_hw_current(), precision_float16))
        return ROCKET_E_UNSUPPORTED;
    const int fd = b->fd;

    mm_plan pl;
    if (mm_plan_init(&pl, M, K, N) < 0) {
        ROCKET_LOGE("rocket_matmul: batch unsupported shape M=%d K=%d N=%d\n", M, K, N);
        return ROCKET_E_SHAPE;
    }
    const int nMt = pl.nMt, nNt = pl.nNt, nKt = pl.nKt;
    const size_t in_slot = pl.in_slot, wt_slot = pl.wt_slot, out_slot = pl.out_slot;
    const size_t in_item = (size_t)nMt * nKt * in_slot;   /* packed fp16 elems / item */
    const size_t wt_item = (size_t)nNt * nKt * wt_slot;

    size_t in_sz  = (size_t)nbatch * in_item * sizeof(_Float16) + CBUF_BANK;
    size_t wt_sz  = (size_t)nbatch * wt_item * sizeof(_Float16) + CBUF_BANK;
    size_t out_sz = (size_t)BATCH * out_slot * sizeof(_Float16) + CBUF_BANK;

    /* Grow the resident BOs to this shape (no-op when it already fits). out_all never
     * needs zeroing (the NPU writes it, only its live region is read back). */
    int gi = mm_bo_ensure(fd, &b->in_all,  in_sz);
    int gw = mm_bo_ensure(fd, &b->wt_all,  wt_sz);
    int go = mm_bo_ensure(fd, &b->out_all, out_sz);
    if (gi < 0 || gw < 0 || go < 0) {
        ROCKET_LOGE("rocket_matmul: batch BO alloc failed (in=%zuMB wt=%zuMB out=%zuMB n=%d)\n",
                in_sz >> 20, wt_sz >> 20, out_sz >> 20, nbatch);
        b->prezeroed = 0;
        return (gi == ROCKET_E_DEVICE || gw == ROCKET_E_DEVICE || go == ROCKET_E_DEVICE)
               ? ROCKET_E_DEVICE : ROCKET_E_NOMEM;
    }

    /* Per-item fp32 K-accumulators (grow-only); zeroed every run (they accumulate). */
    size_t acc_n = (size_t)nbatch * M * N;
    if (b->acc_cap < acc_n) {
        float *na = realloc(b->acc, acc_n * sizeof(float));
        if (!na) { b->prezeroed = 0; return ROCKET_E_NOMEM; }
        b->acc = na; b->acc_cap = acc_n;
    }
    memset(b->acc, 0, acc_n * sizeof(float));
    float *acc = b->acc;

    /* Skip the full-BO zero whenever the layout repeats and the BOs were not just
     * (re)allocated: for a fixed (M,K,N,nbatch) the scatter rewrites the SAME live
     * lanes and never touches padding, so padding zeroed once stays zero. This is
     * the per-call `pack` cost the persistent context exists to reclaim. */
    int layout_changed = (M != b->last_M || K != b->last_K ||
                          N != b->last_N || nbatch != b->last_nbatch);
    int need_zero = gi || gw || layout_changed || !b->prezeroed;

    int    *bit_ = b->bit_, *bm0 = b->bm0, *bn0 = b->bn0, *bMtile = b->bMtile, *bNtile = b->bNtile;
    size_t *boff = b->boff;
    rocket_task_desc *tasks = b->tasks;
    void   *submit_dt = b->submit_dt;
    int ret = 0;

    int prof = mm_profile();
    double t_packA = 0, t_packB = 0, t_gen = 0, t_sync = 0, t_submit = 0, t_wait = 0, t_read = 0, ts;

    /* Pack every item's A and B into its slice of the shared in_all/wt_all BO (zero
     * the whole BO first only when need_zero), then scatter each item at its per-item
     * element offset. */
    ts = now_ms();
    if (rocket_bo_prep(fd, &b->in_all, 1, 0) != 0) { ret = ROCKET_E_DEVICE; goto fail; }
    if (need_zero) memset(b->in_all.ptr, 0, b->in_all.size);
    for (int it = 0; it < nbatch; it++)
        feat_scatter_into(&pl, (_Float16 *)b->in_all.ptr + (size_t)it * in_item, A[it]);
    rocket_bo_fini(fd, &b->in_all);
    t_packA = now_ms() - ts;

    ts = now_ms();
    if (rocket_bo_prep(fd, &b->wt_all, 1, 0) != 0) { ret = ROCKET_E_DEVICE; goto fail; }
    if (need_zero) memset(b->wt_all.ptr, 0, b->wt_all.size);
    for (int it = 0; it < nbatch; it++)
        wt_scatter_into(&pl, (_Float16 *)b->wt_all.ptr + (size_t)it * wt_item, B[it]);
    rocket_bo_fini(fd, &b->wt_all);
    t_packB = now_ms() - ts;

    /* in_all/wt_all now hold this layout's live lanes with zero padding. */
    b->prezeroed = 1;
    b->last_M = M; b->last_K = K; b->last_N = N; b->last_nbatch = nbatch;

    uint64_t npu_regs[256] = {0};
    int batch_chained = mm_batch_chained();   /* gapped per-task (0) vs contiguous one-kick (1) */
    long total = (long)nbatch * nMt * nNt * nKt, done = 0;
    int nb = 0;

    /* (item, mi, ni, ki): ki innermost so a fixed (item,mi,ni) output's K-partials
     * accumulate into acc in ascending ki order — identical to mm_compute. */
    for (int it = 0; it < nbatch; it++) {
        const size_t in_base = (size_t)it * in_item, wt_base = (size_t)it * wt_item;
        for (int mi = 0; mi < nMt; mi++) {
            int m0 = mi * pl.Mt, Mtile = (M - m0 < pl.Mt) ? (M - m0) : pl.Mt;
            for (int ni = 0; ni < nNt; ni++) {
                int n0 = ni * pl.Nt, Ntile = (N - n0 < pl.Nt) ? (N - n0) : pl.Nt;
                for (int ki = 0; ki < nKt; ki++) {
                    int k0 = ki * pl.Kt, Ktile = (K - k0 < pl.Kt) ? (K - k0) : pl.Kt;

                    if (nb == 0) { ts = now_ms(); rocket_bo_prep(fd, &b->regcmd, 1, 0);
                                   t_sync += now_ms() - ts; }
                    ts = now_ms();
                    size_t out_off = (size_t)nb * out_slot;
                    matmul_params_t p = {
                        .m = Mtile, .k = Ktile, .n = Ntile,
                        .input_dma   = (uint32_t)(b->in_all.dma_address +
                                       (in_base + (size_t)(mi * nKt + ki) * in_slot) * sizeof(_Float16)),
                        .weights_dma = (uint32_t)(b->wt_all.dma_address +
                                       (wt_base + (size_t)(ni * nKt + ki) * wt_slot) * sizeof(_Float16)),
                        .output_dma  = (uint32_t)(b->out_all.dma_address + out_off * sizeof(_Float16)),
                        .tasks = npu_regs, .fp32tofp16 = 1,
                    };
                    if ((ret = gen_matmul_fp16(&p)) != 0) {
                        ROCKET_LOGE("rocket_matmul: batch gen failed (%d)\n", ret); goto fail;
                    }
                    if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = ROCKET_E_DEVICE; goto fail; }
                    mm_pack_regcmd(batch_chained, &b->regcmd, tasks, nb, npu_regs, p.task_count);
                    bit_[nb] = it; bm0[nb] = m0; bn0[nb] = n0;
                    bMtile[nb] = Mtile; bNtile[nb] = Ntile; boff[nb] = out_off;
                    t_gen += now_ms() - ts;
                    nb++; done++;

                    if (nb == BATCH || done == total) {
                        mm_seal_chain(batch_chained, &b->regcmd, nb, tasks[0].regcmd_count);
                        ts = now_ms();
                        rocket_bo_fini(fd, &b->regcmd);
                        rocket_bo_prep(fd, &b->out_all, 1, 0);
                        rocket_bo_fini(fd, &b->out_all);
                        t_sync += now_ms() - ts;

                        uint32_t in_h[]  = { b->in_all.handle, b->wt_all.handle, b->regcmd.handle };
                        uint32_t out_h[] = { b->out_all.handle };
                        ts = now_ms();
                        ret = rocket_submit_tasks_pre(fd, submit_dt, tasks, nb, in_h, 3, out_h, 1, batch_chained);
                        t_submit += now_ms() - ts;
                        if (ret) goto fail;

                        ts = now_ms();
                        ret = rocket_bo_prep(fd, &b->out_all, 0, rocket_wait_ns());
                        double w_ms = now_ms() - ts;
                        t_wait += w_ms;
                        if (ret) {
                            ROCKET_LOGE("rocket_matmul: batch WAIT TIMEOUT (%d) after %.0fms "
                                    "M=%d K=%d N=%d nbatch=%d batch=%d tiles=%ld/%ld\n",
                                    ret, w_ms, M, K, N, nbatch, nb, done, total);
                            goto fail;
                        }

                        ts = now_ms();
                        _Float16 *ob = b->out_all.ptr;
                        for (int j = 0; j < nb; j++)
                            detile_accum_f16(acc + (size_t)bit_[j] * M * N, N, ob + boff[j],
                                             bm0[j], bn0[j], bMtile[j], bNtile[j]);
                        rocket_bo_fini(fd, &b->out_all);
                        t_read += now_ms() - ts;
                        nb = 0;
                    }
                }
            }
        }
    }

    for (int it = 0; it < nbatch; it++)
        narrow_f32_to_f16(C[it], acc + (size_t)it * M * N, (size_t)M * N);

    if (prof) {
        pthread_mutex_lock(&g_prof_mu);
        mm_prof_arm_locked();
        g_prof.pack += t_packA + t_packB; g_prof.packA += t_packA; g_prof.packB += t_packB;
        g_prof.gen += t_gen; g_prof.sync += t_sync; g_prof.submit += t_submit;
        g_prof.wait += t_wait; g_prof.read += t_read; g_prof.calls++;
        pthread_mutex_unlock(&g_prof_mu);
    }
    return ROCKET_OK;

fail:
    /* The resident BOs are kept (the context owns them); the packed in/wt still hold a
     * consistent zero-padded layout, so prezeroed/last_* stay valid for the next run. */
    return ret;
}

/* One-shot batched matmul: a transient persistent-context (alloc + zero + run + free
 * per call), so the BO lifetime matches the original behaviour for callers that do not
 * hold a context (the matmul_batch gate, the stateless flash-attention paths). */
int rocket_matmul_fp16_batch(int fd, int M, int K, int N, int nbatch,
                             const _Float16 *const *A, const _Float16 *const *B,
                             _Float16 *const *C)
{
    if (nbatch < 1 || !A || !B || !C) return ROCKET_E_SHAPE;
    if (nbatch == 1) return rocket_matmul_fp16(fd, M, K, N, A[0], B[0], C[0]);
    rocket_mm_batch *b = rocket_mm_batch_create(fd);
    if (!b) return ROCKET_E_NOMEM;
    int ret = rocket_mm_batch_run(b, M, K, N, nbatch, A, B, C);
    rocket_mm_batch_free(b);
    return ret;
}

/* ############################################################################
 * PART 5 — Non-fp16 dtype matmul paths (int8 / int4 / int16 / bf16 / fp32-out / tf32)
 * ##########################################################################*/

/* ============================================================================
 * int8 x int8 -> int32 tiled matmul.
 *
 * Self-contained sibling of the fp16 path above — deliberately NOT a dtype
 * parameterization of mm_*, so the tuned fp16 path stays byte-for-byte unchanged
 * (zero regression risk). To be DRY'd up when the int8 streaming/prepacked path
 * is wired for the backend. Deltas vs fp16: in/wt BOs are 1 B (int8), out BO is
 * 4 B (int32); banks_for is x1 not x2 (so Kt ~2x); input cube C2=16, weight
 * k-group 32 (weight_int8), int32-output cube C2=4; gen_matmul_int8; K-partials
 * summed on the host in int64. A/B are pre-quantized, C is the raw int32 result.
 * ==========================================================================*/

/* CBUF banks for `rows` x Kt int8 (1 B/elem) — the x1 vs fp16's x2 is what lets
 * Kt grow ~2x, cutting K-passes (nKt) and readback. */
static int banks_for_i8(int rows, int Kt) {
    return ((long)rows * Kt + CBUF_BANK - 1) / CBUF_BANK;
}

/* The starting N-tile for an int8 plan: the SMALLEST tile that still reaches the fewest
 * tiles the hardware's tile cap allows.
 *
 * max_tile fixes the tile COUNT — nNt = ceil(N/max_tile) is the fewest possible, and that
 * count is what sets the task count and the dispatch. But ANY Nt >= ceil(N/nNt) reaches that
 * same count, so defaulting Nt to max_tile buys nothing and leaves the tail tile mostly
 * empty. Take the smallest such Nt instead: same tiles, same tasks, same real DMA — but the
 * ALLOCATED columns drop from nNt*max_tile to nNt*Nt, and for a RESIDENT weight those columns
 * are memory held for the process lifetime.
 *
 * At the resident-MoE operating point that is not a rounding error. A resident weight's N is
 * split across the worker fds, so each worker plans on a SLICE: gpt-oss's N=2880 over 5
 * workers gives a 576-wide slice, which at Nt=256 stores 3*256 = 768 columns to hold 576 — a
 * 33% padding tax on every resident expert, when residency is the entire product of the
 * native-quant expert route. At Nt=192 the same 3 tiles store exactly 576.
 *
 * Rounding UP to the 32-column int8 N-alignment is load-bearing: rounding DOWN can drop the
 * tile below ceil(N/nNt) and cost an EXTRA tile — a real dispatch cost, to save nothing.
 *
 * Callers may still shrink Nt afterwards to fit the CBUF; that stays correct (it only costs
 * tiles), it just may no longer divide N evenly.
 */
static int i8_pick_nt(int N, int max_tile) {
    int Nt;
    if (N <= max_tile) {
        Nt = N;
    } else {
        const int nNt = (N + max_tile - 1) / max_tile;   /* fewest tiles max_tile allows */
        Nt = (N + nNt - 1) / nNt;                        /* smallest tile reaching them   */
    }
    Nt = (Nt + 31) / 32 * 32;                            /* int8 N-align — UP, see above  */
    if (Nt > max_tile) Nt = (max_tile / 32) * 32;
    if (Nt < 32)       Nt = 32;
    return Nt;
}

/* int8 NPU layout index math (cf. feat_idx/wt_idx). Input feature cube C2=16,
 * weight k-group 32 (== weight_int8()), int32-output cube C2=4. */
static inline size_t feat_idx_i8(int H, int ch, int h) {   /* input, C2=16 */
    return ((size_t)(ch - 1) / 16) * (size_t)H * 16 + 16 * (size_t)(h - 1) + (ch - 1) % 16;
}
static inline size_t wt_idx_i8(int C, int k, int c) {      /* weight, k-group 32 */
    return (size_t)((c - 1) / 32) * 32 * 32 + (size_t)((k - 1) / 32) * 32 * C
         + (size_t)((c - 1) % 32) + (size_t)((k - 1) % 32) * 32;
}
static inline size_t out_idx_i8(int H, int ch, int h) {    /* output, C2=4 */
    return ((size_t)(ch - 1) / 4) * (size_t)H * 4 + 4 * (size_t)(h - 1) + (ch - 1) % 4;
}

int rocket_matmul_plan_int8(int M, int K, int N, int *pMt, int *pKt, int *pNt)
{
    /* Machine parameters from the active hardware profile (see rocket_matmul_plan). */
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const int MAX_TILE = hw->max_tile, CBUF_BANKS = hw->cbuf_banks;
    /* M%4==0 only — M==1 (height-1 GEMV) is broken on HW (the cosine-sim correctness matrix); the
     * one-shot rocket_matmul_int8 pads M==1->4 before planning, non-padding
     * callers (resident/int16_exact pre-check) get an honest -1. */
    if (K % 32 || N % 32 || M % 4 != 0)
        return ROCKET_E_SHAPE;

    int Mt = (M < MAX_TILE) ? M : MAX_TILE;
    /* Deliberately NOT i8_pick_nt here, unlike the group-wise twin below. This planner's Kt
     * is unconstrained (it starts at K and shrinks to fit the CBUF), so a smaller Nt frees
     * CBUF banks and lets Kt GROW — at the resident MoE shape it would move Kt 640 -> 768 and
     * re-tile K on a shipped, HW-validated path, for a padding win this plan does not need
     * (the dense W8A8 path holds one weight per tensor, not thousands of experts). The
     * group-wise planner has no such coupling: its Kt can never exceed the quant group.
     * Applying it here is a real lever for dense-int8 model fit, but it is its own change,
     * with its own gate run — int8 CBUF edges are where the feature-DMA resonance lived. */
    int Nt = (N < MAX_TILE) ? N : MAX_TILE;
    Nt = (Nt / 32) * 32; if (Nt < 32) Nt = 32;   /* int8 N-align is 32 */

    const char *e;
    if ((e = getenv("ROCKET_MM_MT"))) { int v = atoi(e); if (v >= 4)  { Mt = (v/4)*4;   if (Mt > M) Mt = M; } }
    if ((e = getenv("ROCKET_MM_NT"))) { int v = atoi(e); if (v >= 32) { Nt = (v/32)*32; if (Nt > N) Nt = N; } }

    /* Reserve ONE feature slack bank: gen_matmul_int8 sets data_bank = fd_banks+1
     * to dodge the int8 feature-DMA tail-row resonance, so the planner must keep
     * feature+weight within BANKS-1 (the +1th bank is the feature's slack). */
    const int I8_BUDGET = CBUF_BANKS - 1;
    int Kt = (K < 16384) ? K : 16384;
    Kt = (Kt / 32) * 32; if (Kt < 32) Kt = 32;
    while (Kt > 32 && banks_for_i8(Mt, Kt) + banks_for_i8(Nt, Kt) > I8_BUDGET) Kt -= 32;
    while (banks_for_i8(Mt, Kt) + banks_for_i8(Nt, Kt) > I8_BUDGET && Nt > 32) Nt -= 32;
    while (banks_for_i8(Mt, Kt) + banks_for_i8(Nt, Kt) > I8_BUDGET && Mt > 4)  Mt -= 4;

    if ((e = getenv("ROCKET_MM_KT"))) {
        int v = atoi(e);
        if (v >= 32) {
            Kt = (v/32)*32; if (Kt > (K/32)*32) Kt = (K/32)*32;
            while (Kt > 32 && banks_for_i8(Mt, Kt) + banks_for_i8(Nt, Kt) > I8_BUDGET) Kt -= 32;
        }
    }

    if (pMt) *pMt = Mt;
    if (pKt) *pKt = Kt;
    if (pNt) *pNt = Nt;

    int nm = (M + Mt - 1) / Mt, nn = (N + Nt - 1) / Nt, nk = (K + Kt - 1) / Kt;
    /* M4: reject shapes whose int tile-count product would overflow before the
     * size_t BO math (and the `total = nMt*nNt*nKt` in the compute loops below). */
    if ((int64_t)nm * nn * nk > INT32_MAX) return ROCKET_E_SHAPE;
    return nm * nn * nk;
}

int rocket_matmul_int8(int fd, int M, int K, int N,
                       const int8_t *A, const int8_t *B, int32_t *C)
{
    if (!rocket_hw_dtype_supported(rocket_hw_current(), precision_int8))
        return ROCKET_E_UNSUPPORTED;
    /* M==1 GEMV: height-1 HW geometry is broken (same as fp16, see
     * rocket_matmul_fp16). Pad to a 4-row tile (zero pad rows contribute 0 to the
     * integer dot products — no saturation risk), return row 0. */
    if (M == 1) MM_PAD_M1(fd, K, N, A, B, C, int8_t, int32_t, rocket_matmul_int8);
    int Mt, Kt, Nt;
    if (rocket_matmul_plan_int8(M, K, N, &Mt, &Kt, &Nt) < 0) {
        ROCKET_LOGE("rocket_matmul_int8: unsupported shape M=%d K=%d N=%d "
                "(need K%%32, N%%32, M%%4||1)\n", M, K, N);
        return -1;
    }
    int nMt = (M + Mt - 1) / Mt, nNt = (N + Nt - 1) / Nt, nKt = (K + Kt - 1) / Kt;
    size_t in_slot  = (size_t)rup(Mt, 4)  * rup(Kt, 32);   /* int8 elems */
    size_t wt_slot  = (size_t)rup(Nt, 32) * rup(Kt, 32);   /* int8 elems */
    size_t out_slot = (size_t)rup(Mt, 4)  * rup(Nt, 16);   /* int32 elems */

    /* ROCKET_MM_PROFILE timers (mirror mm_compute; feed g_prof_i8 at the end). */
    int prof_i8 = mm_profile();
    double tA = 0, tB = 0, t_sync = 0, t_submit = 0, t_wait = 0, t_read = 0, tsx = 0;

    /* ---- allocate BOs (int8 in/wt @1B, int32 out @4B) ---- */
    rocket_bo guard = {0}, regcmd = {0}, in_all = {0}, wt_all = {0}, out_all = {0};
    size_t in_sz  = (size_t)nMt * nKt * in_slot  + CBUF_BANK;
    size_t wt_sz  = (size_t)nNt * nKt * wt_slot  + CBUF_BANK;
    size_t rc_sz  = (size_t)BATCH * RC_STRIDE * sizeof(uint64_t);
    size_t out_sz = (size_t)BATCH * out_slot * sizeof(int32_t) + CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096, &guard);                 /* push allocs off IOVA 0 */
    ret |= rocket_bo_alloc(fd, rc_sz,  &regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &in_all);
    ret |= rocket_bo_alloc(fd, wt_sz,  &wt_all);
    ret |= rocket_bo_alloc(fd, out_sz, &out_all);
    if (ret) { ROCKET_LOGE("rocket_matmul_int8: BO alloc failed\n"); ret = -1; goto free_bos; }
    if (((in_all.dma_address + in_sz) | (wt_all.dma_address + wt_sz) |
         (out_all.dma_address + out_sz) | (regcmd.dma_address + rc_sz)) >> 32) {
        ROCKET_LOGE("rocket_matmul_int8: a BO dma_address exceeds 32 bits\n");
        ret = -1; goto free_bos;
    }

    /* ---- pack weights B[N,K] -> (N/32,K/32,32,32) int8 tile layout ---- */
    tB = now_ms();
    if (rocket_bo_prep(fd, &wt_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(wt_all.ptr, 0, wt_all.size);
    for (int ni = 0; ni < nNt; ni++) {
        int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            int8_t *slot = (int8_t *)wt_all.ptr + (size_t)(ni * nKt + ki) * wt_slot;
            for (int kk = 1; kk <= Ntile; kk++)
                for (int c = 1; c <= Ktile; c++)
                    slot[wt_idx_i8(Ktile, kk, c)] = B[(size_t)(n0 + kk - 1) * K + (k0 + c - 1)];
        }
    }
    rocket_bo_fini(fd, &wt_all);
    tB = now_ms() - tB;

    /* ---- pack input A[M,K] -> (M,K) int8 feature cube (C2=16) ---- */
    tA = now_ms();
    if (rocket_bo_prep(fd, &in_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(in_all.ptr, 0, in_all.size);
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            int8_t *slot = (int8_t *)in_all.ptr + (size_t)(mi * nKt + ki) * in_slot;
            for (int h = 1; h <= Mtile; h++)
                for (int c = 1; c <= Ktile; c++)
                    slot[feat_idx_i8(Mtile, c, h)] = A[(size_t)(m0 + h - 1) * K + (k0 + c - 1)];
        }
    }
    rocket_bo_fini(fd, &in_all);
    tA = now_ms() - tA;

    /* ---- batched tile compute: host int64 K-accumulation (bit-exact oracle).
     * NPU-side int8 K-accum (the DPU-EW int32 add) is HW-DEAD: the EW operand DMA
     * is <=16-bit, so int32 partials read back as garbage (a true add of two int32
     * tiles returned an fp16 inf/NaN bit pattern). Sum the partials on the host.
     * Do not reattempt. ---- */
    rocket_task_desc *tasks = malloc(BATCH * sizeof(*tasks));
    uint64_t npu_regs[256] = {0};
    int64_t *acc = NULL;
    int *bm0 = NULL, *bn0 = NULL, *bMtile = NULL, *bNtile = NULL;
    size_t *boff = NULL;
    if (!tasks) { ret = -1; goto free_host; }

    acc = calloc((size_t)M * N, sizeof(int64_t));
    bm0 = malloc(BATCH * sizeof(int)); bn0 = malloc(BATCH * sizeof(int));
    bMtile = malloc(BATCH * sizeof(int)); bNtile = malloc(BATCH * sizeof(int));
    boff = malloc(BATCH * sizeof(size_t));
    if (!acc || !bm0 || !bn0 || !bMtile || !bNtile || !boff) { ret = -1; goto free_host; }

    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;

                if (nb == 0) { tsx = now_ms(); rocket_bo_prep(fd, &regcmd, 1, 0); t_sync += now_ms() - tsx; }

                size_t out_off = (size_t)nb * out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma  = (uint32_t)(in_all.dma_address + (size_t)(mi*nKt+ki) * in_slot),
                    .weights_dma = (uint32_t)(wt_all.dma_address + (size_t)(ni*nKt+ki) * wt_slot),
                    .output_dma = (uint32_t)(out_all.dma_address + out_off * sizeof(int32_t)),
                    .tasks = npu_regs,
                };
                if ((ret = gen_matmul_int8(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul_int8: gen failed (%d)\n", ret); goto free_host;
                }
                if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto free_host; }
                memcpy((uint64_t *)regcmd.ptr + (size_t)nb * RC_STRIDE, npu_regs,
                       (size_t)p.task_count * sizeof(uint64_t));
                tasks[nb].regcmd = (uint32_t)(regcmd.dma_address + (size_t)nb * RC_STRIDE * sizeof(uint64_t));
                tasks[nb].regcmd_count = p.task_count;
                bm0[nb] = m0; bn0[nb] = n0; bMtile[nb] = Mtile; bNtile[nb] = Ntile; boff[nb] = out_off;
                nb++; done_tiles++;

                if (nb == BATCH || done_tiles == total) {
                    tsx = now_ms();
                    rocket_bo_fini(fd, &regcmd);
                    rocket_bo_prep(fd, &out_all, 1, 0);
                    rocket_bo_fini(fd, &out_all);
                    t_sync += now_ms() - tsx;

                    uint32_t in_h[]  = { in_all.handle, wt_all.handle, regcmd.handle };
                    uint32_t out_h[] = { out_all.handle };
                    tsx = now_ms();
                    ret = rocket_submit_tasks(fd, tasks, nb, in_h, 3, out_h, 1);
                    t_submit += now_ms() - tsx;
                    if (ret) goto free_host;

                    tsx = now_ms();
                    ret = rocket_bo_prep(fd, &out_all, 0, rocket_wait_ns());
                    t_wait += now_ms() - tsx;
                    if (ret) {
                        ROCKET_LOGE("rocket_matmul_int8: WAIT TIMEOUT (%d) M=%d K=%d N=%d "
                                "batch=%d tiles=%d/%d\n", ret, M, K, N, nb, done_tiles, total);
                        goto free_host;
                    }

                    tsx = now_ms();
                    int32_t *ob = (int32_t *)out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        int32_t *slot = ob + boff[j];
                        for (int h = 1; h <= bMtile[j]; h++)
                            for (int nn = 1; nn <= bNtile[j]; nn++)
                                acc[(size_t)(bm0[j] + h - 1) * N + (bn0[j] + nn - 1)] +=
                                    (int64_t)slot[out_idx_i8(bMtile[j], nn, h)];
                    }
                    rocket_bo_fini(fd, &out_all);
                    t_read += now_ms() - tsx;
                    nb = 0;
                }
            }
        }
    }
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = (int32_t)acc[i];

    if (prof_i8) {
        pthread_mutex_lock(&g_prof_mu);
        if (!g_prof_i8_armed) { atexit(mm_prof_i8_dump); g_prof_i8_armed = 1; }
        g_prof_i8.packA += tA; g_prof_i8.packB += tB; g_prof_i8.sync += t_sync;
        g_prof_i8.submit += t_submit; g_prof_i8.wait += t_wait; g_prof_i8.read += t_read;
        g_prof_i8.calls++;
        pthread_mutex_unlock(&g_prof_mu);
    }

free_host:
    free(acc); free(tasks); free(bm0); free(bn0); free(bMtile); free(bNtile); free(boff);
free_bos:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &in_all);
    rocket_bo_free(fd, &wt_all); rocket_bo_free(fd, &out_all);
    return ret;
}

/* ============================================================================
 * GROUP-WISE int8 matmul: C_f[M,N] = sum_g a_scale[m,g]*b_scale[n,g] * (int32
 * partial of K-group g), accumulated in fp32.
 *
 * The int8 sibling of rocket_matmul_int4_groupwise, and the primitive a natively
 * quantized weight needs. Such a weight (a GGUF MXFP4 / Q8_0 / Q4_K block) carries
 * one scale per K-block, and the NPU cannot apply a K-blocked scale on-chip: at the
 * output stage K is fully contracted, so nothing in the DPU is indexed by a K-block,
 * for any dtype. But the integer partials ALREADY leave the chip at every K-tile
 * boundary — on-device integer K-accum is HW-dead (see rocket_matmul_int8 above) —
 * so the block scale is free at a boundary that is already being paid for. Keep each
 * K-tile inside one quant group, multiply its int32 partial by that group's scale,
 * accumulate in fp32 on the host. The NPU-side work is identical to
 * rocket_matmul_int8; the only delta is `float += scale * int32` on readback instead
 * of `int64 += int32`.
 *
 * A, B are PRE-QUANTIZED int8 (row-major); a_scale is [M*nG] and b_scale is [N*nG]
 * (row-major, nG = K/group); Cf[M,N] is overwritten. Alignment is rocket_matmul_int8's
 * minus the M==1 pad: K%32, N%32, M%4 — a padded M==1 would need a padded a_scale
 * too, so single vectors are padded to M=4 caller-side (as the resident paths require).
 *
 * There is NO saturation bound here. The int4 twin needs `49*group < 32767` because
 * its NPU output is int16; int8's is int32, and a K-tile partial is bounded by
 * |group * 127 * 127| = 16129*group — well inside int32 for any group the CBUF can
 * hold (Kt <= 640 on RK3588). That headroom is exactly why int8, not int4, is the
 * dtype for a GGUF-quantized weight.
 * ==========================================================================*/

/* Group-wise int8 tiling (pure). Kt is constrained to lie inside one quant group,
 * which the free-Kt search in rocket_matmul_plan_int8 cannot express, so the CBUF is
 * re-fitted here around that constraint.
 *
 * Kt need only DIVIDE the group, not equal it: each K-tile must lie wholly inside one
 * group so its partial can take that group's single scale, but it does not have to BE
 * the group. So readback (~ M*N*nKt, and these paths are readback-bound) is set by the
 * CBUF's Kt cap, and `group` only upper-bounds it. Take the largest divisor that fits.
 *
 * Do NOT do what rocket_matmul_int4_groupwise does — overwrite the planner's Kt with
 * `group` and re-check nothing. int4 gets away with that only because its int16
 * saturation bound already caps group at 668 AND its nibble packing halves the bytes.
 * int8 has neither (the int32 accumulator is the whole point), so an un-rechecked wide
 * group would silently overflow the CBUF. */
int rocket_matmul_plan_int8_gw(int M, int K, int N, int group,
                               int *pMt, int *pKt, int *pNt)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const int MAX_TILE = hw->max_tile, CBUF_BANKS = hw->cbuf_banks;
    if (K % 32 || N % 32 || M % 4 != 0) return ROCKET_E_SHAPE;
    if (group < 32 || group % 32 || K % group) return ROCKET_E_SHAPE;

    /* Reserve ONE feature slack bank, exactly as rocket_matmul_plan_int8 does:
     * gen_matmul_int8 sets data_bank = fd_banks+1 to dodge the int8 feature-DMA
     * tail-row resonance, so feature+weight must stay within BANKS-1. */
    const int I8_BUDGET = CBUF_BANKS - 1;

    int Mt = (M < MAX_TILE) ? M : MAX_TILE;
    int Nt = i8_pick_nt(N, MAX_TILE);

    const char *e;
    if ((e = getenv("ROCKET_MM_MT"))) { int v = atoi(e); if (v >= 4)  { Mt = (v/4)*4;   if (Mt > M) Mt = M; } }
    if ((e = getenv("ROCKET_MM_NT"))) { int v = atoi(e); if (v >= 32) { Nt = (v/32)*32; if (Nt > N) Nt = N; } }
    /* ROCKET_MM_KT is deliberately NOT honored: Kt is pinned by the quant group. */

    int Kt = 0;
    for (;;) {
        for (int d = group; d >= 32; d -= 32) {
            if (group % d) continue;             /* a K-tile must never straddle a group */
            if (banks_for_i8(Mt, d) + banks_for_i8(Nt, d) <= I8_BUDGET) { Kt = d; break; }
        }
        if (Kt) break;
        /* Not even a 32-wide K-tile fits at this tile size. Unreachable on RK3588
         * (Mt,Nt <= 256 costs 2 banks at Kt=32, against a budget of 11); defensive
         * against a future profile with a large max_tile or a small CBUF. */
        if      (Nt > 32) Nt -= 32;
        else if (Mt > 4)  Mt -= 4;
        else return ROCKET_E_SHAPE;
    }

    if (pMt) *pMt = Mt;
    if (pKt) *pKt = Kt;
    if (pNt) *pNt = Nt;

    int nm = (M + Mt - 1) / Mt, nn = (N + Nt - 1) / Nt, nk = K / Kt;   /* Kt | group | K */
    if ((int64_t)nm * nn * nk > INT32_MAX) return ROCKET_E_SHAPE;
    return nm * nn * nk;
}

int rocket_matmul_int8_groupwise(int fd, int M, int K, int N,
                                 const int8_t *A, const int8_t *B,
                                 const float *a_scale, const float *b_scale,
                                 float *Cf, int group)
{
    if (!rocket_hw_dtype_supported(rocket_hw_current(), precision_int8))
        return ROCKET_E_UNSUPPORTED;
    int Mt, Kt, Nt;
    if (rocket_matmul_plan_int8_gw(M, K, N, group, &Mt, &Kt, &Nt) < 0) {
        ROCKET_LOGE("rocket_matmul_int8_groupwise: unsupported shape M=%d K=%d N=%d "
                "group=%d (need K%%32, N%%32, M%%4, group%%32, group|K)\n", M, K, N, group);
        return ROCKET_E_SHAPE;
    }
    int nG = K / group;                                        /* quant groups along K */
    int nMt = (M + Mt - 1) / Mt, nNt = (N + Nt - 1) / Nt, nKt = K / Kt;
    int kt_per_group = group / Kt;                             /* K-tiles inside one group */
    size_t in_slot  = (size_t)rup(Mt, 4)  * rup(Kt, 32);       /* int8 elems  */
    size_t wt_slot  = (size_t)rup(Nt, 32) * rup(Kt, 32);       /* int8 elems  */
    size_t out_slot = (size_t)rup(Mt, 4)  * rup(Nt, 16);       /* int32 elems */

    /* ---- allocate BOs (int8 in/wt @1B, int32 out @4B) — geometry identical to
     * rocket_matmul_int8, which is what makes this a drop-in scale-on-readback. ---- */
    rocket_bo guard = {0}, regcmd = {0}, in_all = {0}, wt_all = {0}, out_all = {0};
    size_t in_sz  = (size_t)nMt * nKt * in_slot  + CBUF_BANK;
    size_t wt_sz  = (size_t)nNt * nKt * wt_slot  + CBUF_BANK;
    size_t rc_sz  = (size_t)BATCH * RC_STRIDE * sizeof(uint64_t);
    size_t out_sz = (size_t)BATCH * out_slot * sizeof(int32_t) + CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096, &guard);                  /* push allocs off IOVA 0 */
    ret |= rocket_bo_alloc(fd, rc_sz,  &regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &in_all);
    ret |= rocket_bo_alloc(fd, wt_sz,  &wt_all);
    ret |= rocket_bo_alloc(fd, out_sz, &out_all);
    if (ret) { ROCKET_LOGE("rocket_matmul_int8_groupwise: BO alloc failed\n"); ret = -1; goto free_bos; }
    if (((in_all.dma_address + in_sz) | (wt_all.dma_address + wt_sz) |
         (out_all.dma_address + out_sz) | (regcmd.dma_address + rc_sz)) >> 32) {
        ROCKET_LOGE("rocket_matmul_int8_groupwise: a BO dma_address exceeds 32 bits\n");
        ret = -1; goto free_bos;
    }

    /* ---- pack weights B[N,K] -> (N/32,K/32,32,32) int8 tile layout ---- */
    if (rocket_bo_prep(fd, &wt_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(wt_all.ptr, 0, wt_all.size);
    for (int ni = 0; ni < nNt; ni++) {
        int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            int8_t *slot = (int8_t *)wt_all.ptr + (size_t)(ni * nKt + ki) * wt_slot;
            for (int kk = 1; kk <= Ntile; kk++)
                for (int c = 1; c <= Ktile; c++)
                    slot[wt_idx_i8(Ktile, kk, c)] = B[(size_t)(n0 + kk - 1) * K + (k0 + c - 1)];
        }
    }
    rocket_bo_fini(fd, &wt_all);

    /* ---- pack input A[M,K] -> (M,K) int8 feature cube (C2=16) ---- */
    if (rocket_bo_prep(fd, &in_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(in_all.ptr, 0, in_all.size);
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            int8_t *slot = (int8_t *)in_all.ptr + (size_t)(mi * nKt + ki) * in_slot;
            for (int h = 1; h <= Mtile; h++)
                for (int c = 1; c <= Ktile; c++)
                    slot[feat_idx_i8(Mtile, c, h)] = A[(size_t)(m0 + h - 1) * K + (k0 + c - 1)];
        }
    }
    rocket_bo_fini(fd, &in_all);

    /* ---- batched tile compute: host fp32 K-accumulation with per-group scales ---- */
    rocket_task_desc *tasks = malloc(BATCH * sizeof(*tasks));
    uint64_t npu_regs[256] = {0};
    int *bm0 = malloc(BATCH * sizeof(int)), *bn0 = malloc(BATCH * sizeof(int));
    int *bMtile = malloc(BATCH * sizeof(int)), *bNtile = malloc(BATCH * sizeof(int));
    int *bg = malloc(BATCH * sizeof(int));
    size_t *boff = malloc(BATCH * sizeof(size_t));
    if (!tasks || !bm0 || !bn0 || !bMtile || !bNtile || !bg || !boff) { ret = -1; goto free_host; }
    memset(Cf, 0, (size_t)M * N * sizeof(float));

    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;

                if (nb == 0) rocket_bo_prep(fd, &regcmd, 1, 0);

                size_t out_off = (size_t)nb * out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma   = (uint32_t)(in_all.dma_address + (size_t)(mi*nKt+ki) * in_slot),
                    .weights_dma = (uint32_t)(wt_all.dma_address + (size_t)(ni*nKt+ki) * wt_slot),
                    .output_dma  = (uint32_t)(out_all.dma_address + out_off * sizeof(int32_t)),
                    .tasks = npu_regs,
                };
                if ((ret = gen_matmul_int8(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul_int8_groupwise: gen failed (%d)\n", ret); goto free_host;
                }
                if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto free_host; }
                memcpy((uint64_t *)regcmd.ptr + (size_t)nb * RC_STRIDE, npu_regs,
                       (size_t)p.task_count * sizeof(uint64_t));
                tasks[nb].regcmd = (uint32_t)(regcmd.dma_address + (size_t)nb * RC_STRIDE * sizeof(uint64_t));
                tasks[nb].regcmd_count = p.task_count;
                bm0[nb] = m0; bn0[nb] = n0; bMtile[nb] = Mtile; bNtile[nb] = Ntile;
                bg[nb] = ki / kt_per_group;    /* Kt | group, so the tile lies in ONE group */
                boff[nb] = out_off;
                nb++; done_tiles++;

                if (nb == BATCH || done_tiles == total) {
                    rocket_bo_fini(fd, &regcmd);
                    rocket_bo_prep(fd, &out_all, 1, 0);
                    rocket_bo_fini(fd, &out_all);

                    uint32_t in_h[]  = { in_all.handle, wt_all.handle, regcmd.handle };
                    uint32_t out_h[] = { out_all.handle };
                    if ((ret = rocket_submit_tasks(fd, tasks, nb, in_h, 3, out_h, 1)) != 0) goto free_host;
                    if ((ret = rocket_bo_prep(fd, &out_all, 0, rocket_wait_ns())) != 0) {
                        ROCKET_LOGE("rocket_matmul_int8_groupwise: WAIT TIMEOUT (%d) M=%d K=%d N=%d "
                                "batch=%d tiles=%d/%d\n", ret, M, K, N, nb, done_tiles, total);
                        goto free_host;
                    }

                    int32_t *ob = (int32_t *)out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        int32_t *slot = ob + boff[j];
                        int g = bg[j];
                        for (int h = 1; h <= bMtile[j]; h++) {
                            int mrow = bm0[j] + h - 1;
                            float as = a_scale[(size_t)mrow * nG + g];
                            for (int nn = 1; nn <= bNtile[j]; nn++) {
                                int ncol = bn0[j] + nn - 1;
                                Cf[(size_t)mrow * N + ncol] +=
                                    as * b_scale[(size_t)ncol * nG + g] *
                                    (float)slot[out_idx_i8(bMtile[j], nn, h)];
                            }
                        }
                    }
                    rocket_bo_fini(fd, &out_all);
                    nb = 0;
                }
            }
        }
    }

free_host:
    free(tasks); free(bm0); free(bn0); free(bMtile); free(bNtile); free(bg); free(boff);
free_bos:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &in_all);
    rocket_bo_free(fd, &wt_all); rocket_bo_free(fd, &out_all);
    return ret;
}

/* ============================================================================
 * int4 x int4 -> int16 tiled matmul.
 *
 * Self-contained sibling of the int8 path above (same zero-regression stance).
 * Deltas vs int8: in/wt are NIBBLE-packed (2 int4 per byte); feature cube C2=32
 * (vs 16); weight layout (N/64,K/32,64,32) [weight_int4, N-group 64 vs int8's 32];
 * output is int16 (2 B, cube C2=8) NOT int32. K-partials are read back as int16
 * and host-summed in int64 -> int32 C. int8's int32-out quirk size_e=7 carries to
 * int16 (HW-confirmed). NPU K-accum (DPU-EW) is deferred (int16 output IS <=16-bit
 * so it is FEASIBLE here, unlike int8 — a later perf lever); this is the host-accum
 * oracle. Encodings live in gen_matmul_int4 (npu_regcmd.c).
 *
 * Each K-tile's int16 output SATURATES if |Kt-partial| > 32767; the host can't
 * recover a saturated partial, so a bit-exact test must keep Kt small enough (or
 * the data magnitude low enough) that no partial saturates. The CBUF Kt limit for
 * int4 is ~2x int8's (half the bytes), so many shapes are single-pass (nKt=1).
 * ==========================================================================*/

/* CBUF banks for `rows` x Kt int4 (0.5 B/elem) — half int8's bytes, so Kt ~2x. */
static int banks_for_i4(int rows, int Kt) {
    long bytes = (long)rows * Kt / 2;
    return (bytes + CBUF_BANK - 1) / CBUF_BANK;
}

/* int4 NPU layout index math (NIBBLE indices for in/wt; int16 elems for out). */
static inline size_t feat_idx_i4(int H, int ch, int h) {   /* input nibble, C2=32 */
    return ((size_t)(ch - 1) / 32) * (size_t)H * 32 + 32 * (size_t)(h - 1) + (ch - 1) % 32;
}
static inline size_t wt_idx_i4(int C, int k, int c) {      /* weight nibble, (N/64,K/32,64,32) */
    size_t nKgrp   = (size_t)((C + 31) / 32);
    size_t Ngrp    = (size_t)(k - 1) / 64, Nwithin = (size_t)(k - 1) % 64;
    size_t Kgrp    = (size_t)(c - 1) / 32, Kwithin = (size_t)(c - 1) % 32;
    return Ngrp * nKgrp * 64 * 32 + Kgrp * 64 * 32 + Nwithin * 32 + Kwithin;
}
static inline size_t out_idx_i4(int H, int ch, int h) {    /* output int16 elem, C2=8 */
    return ((size_t)(ch - 1) / 8) * (size_t)H * 8 + 8 * (size_t)(h - 1) + (ch - 1) % 8;
}
/* set nibble `idx` (byte idx/2; even=low, odd=high) in a packed buffer. */
static inline void put_nib(uint8_t *buf, size_t idx, int8_t v) {
    uint8_t nib = (uint8_t)(v & 0xF);
    if (idx & 1) buf[idx >> 1] = (uint8_t)((buf[idx >> 1] & 0x0F) | (nib << 4));
    else         buf[idx >> 1] = (uint8_t)((buf[idx >> 1] & 0xF0) | nib);
}

int rocket_matmul_plan_int4(int M, int K, int N, int *pMt, int *pKt, int *pNt)
{
    /* Machine parameters from the active hardware profile (see rocket_matmul_plan). */
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const int MAX_TILE = hw->max_tile, CBUF_BANKS = hw->cbuf_banks;
    if (K % 32 || N % 64 || M % 4 != 0)   /* int4 N-align is 64; M%4 (M==1 padded in one-shot) */
        return ROCKET_E_SHAPE;

    int Mt = (M < MAX_TILE) ? M : MAX_TILE;
    int Nt = (N < MAX_TILE) ? N : MAX_TILE;
    Nt = (Nt / 64) * 64; if (Nt < 64) Nt = 64;         /* int4 N-align 64 */

    const char *e;
    if ((e = getenv("ROCKET_MM_MT"))) { int v = atoi(e); if (v >= 4)  { Mt = (v/4)*4;   if (Mt > M) Mt = M; } }
    if ((e = getenv("ROCKET_MM_NT"))) { int v = atoi(e); if (v >= 64) { Nt = (v/64)*64; if (Nt > N) Nt = N; } }

    int Kt = (K < 16384) ? K : 16384;
    Kt = (Kt / 32) * 32; if (Kt < 32) Kt = 32;
    while (Kt > 32 && banks_for_i4(Mt, Kt) + banks_for_i4(Nt, Kt) > CBUF_BANKS) Kt -= 32;
    while (banks_for_i4(Mt, Kt) + banks_for_i4(Nt, Kt) > CBUF_BANKS && Nt > 64) Nt -= 64;
    while (banks_for_i4(Mt, Kt) + banks_for_i4(Nt, Kt) > CBUF_BANKS && Mt > 4)  Mt -= 4;

    if ((e = getenv("ROCKET_MM_KT"))) {
        int v = atoi(e);
        if (v >= 32) {
            Kt = (v/32)*32; if (Kt > (K/32)*32) Kt = (K/32)*32;
            while (Kt > 32 && banks_for_i4(Mt, Kt) + banks_for_i4(Nt, Kt) > CBUF_BANKS) Kt -= 32;
        }
    }

    if (pMt) *pMt = Mt;
    if (pKt) *pKt = Kt;
    if (pNt) *pNt = Nt;

    int nm = (M + Mt - 1) / Mt, nn = (N + Nt - 1) / Nt, nk = (K + Kt - 1) / Kt;
    /* M4: reject shapes whose int tile-count product would overflow before the
     * size_t BO math (and the `total = nMt*nNt*nKt` in the compute loops below). */
    if ((int64_t)nm * nn * nk > INT32_MAX) return ROCKET_E_SHAPE;
    return nm * nn * nk;
}

int rocket_matmul_int4_ex(int fd, int M, int K, int N,
                          const int8_t *A, const int8_t *B, int32_t *C, int kt_cap)
{
    if (!rocket_hw_dtype_supported(rocket_hw_current(), precision_int4))
        return ROCKET_E_UNSUPPORTED;
    if (M == 1)   /* height-1 GEMV broken on HW; pad to 4 rows, return row 0 */
        MM_PAD_M1(fd, K, N, A, B, C, int8_t, int32_t, rocket_matmul_int4);
    int Mt, Kt, Nt;
    if (rocket_matmul_plan_int4(M, K, N, &Mt, &Kt, &Nt) < 0) {
        ROCKET_LOGE("rocket_matmul_int4: unsupported shape M=%d K=%d N=%d "
                "(need K%%32, N%%64, M%%4||1)\n", M, K, N);
        return -1;
    }
    /* int16-saturation cap: each K-tile partial is read back as int16, and [-8,7]
     * inputs make |partial| <= 64*Kt, so an uncapped Kt (the plan can pick K) overflows
     * int16 for large K. kt_cap>0 caps Kt (rounded to %32) so 64*Kt < 32767; kt_cap=0
     * disables the cap (the raw entry point, valid only for small-magnitude data). */
    if (kt_cap >= 32 && Kt > kt_cap) { Kt = (kt_cap / 32) * 32; if (Kt < 32) Kt = 32; }
    int nMt = (M + Mt - 1) / Mt, nNt = (N + Nt - 1) / Nt, nKt = (K + Kt - 1) / Kt;
    size_t in_slot  = (size_t)rup(Mt, 4)  * rup(Kt, 32);   /* int4 nibbles */
    size_t wt_slot  = (size_t)rup(Nt, 64) * rup(Kt, 32);   /* int4 nibbles */
    size_t out_slot = (size_t)rup(Mt, 4)  * rup(Nt, 16);   /* int16 elems  */

    rocket_bo guard = {0}, regcmd = {0}, in_all = {0}, wt_all = {0}, out_all = {0};
    size_t in_sz  = (size_t)nMt * nKt * (in_slot / 2) + CBUF_BANK;   /* nibbles -> bytes */
    size_t wt_sz  = (size_t)nNt * nKt * (wt_slot / 2) + CBUF_BANK;
    size_t rc_sz  = (size_t)BATCH * RC_STRIDE * sizeof(uint64_t);
    size_t out_sz = (size_t)BATCH * out_slot * sizeof(int16_t) + CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096, &guard);
    ret |= rocket_bo_alloc(fd, rc_sz,  &regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &in_all);
    ret |= rocket_bo_alloc(fd, wt_sz,  &wt_all);
    ret |= rocket_bo_alloc(fd, out_sz, &out_all);
    if (ret) { ROCKET_LOGE("rocket_matmul_int4: BO alloc failed\n"); ret = -1; goto free_bos; }
    if (((in_all.dma_address + in_sz) | (wt_all.dma_address + wt_sz) |
         (out_all.dma_address + out_sz) | (regcmd.dma_address + rc_sz)) >> 32) {
        ROCKET_LOGE("rocket_matmul_int4: a BO dma_address exceeds 32 bits\n");
        ret = -1; goto free_bos;
    }

    /* ---- pack weights B[N,K] -> (N/64,K/32,64,32) int4 nibble layout ---- */
    if (rocket_bo_prep(fd, &wt_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(wt_all.ptr, 0, wt_all.size);
    for (int ni = 0; ni < nNt; ni++) {
        int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            uint8_t *slot = (uint8_t *)wt_all.ptr + (size_t)(ni * nKt + ki) * (wt_slot / 2);
            for (int kk = 1; kk <= Ntile; kk++)
                for (int c = 1; c <= Ktile; c++)
                    put_nib(slot, wt_idx_i4(Ktile, kk, c), B[(size_t)(n0 + kk - 1) * K + (k0 + c - 1)]);
        }
    }
    rocket_bo_fini(fd, &wt_all);

    /* ---- pack input A[M,K] -> (K/32,M,32) int4 nibble feature cube ---- */
    if (rocket_bo_prep(fd, &in_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(in_all.ptr, 0, in_all.size);
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            uint8_t *slot = (uint8_t *)in_all.ptr + (size_t)(mi * nKt + ki) * (in_slot / 2);
            for (int h = 1; h <= Mtile; h++)
                for (int c = 1; c <= Ktile; c++)
                    put_nib(slot, feat_idx_i4(Mtile, c, h), A[(size_t)(m0 + h - 1) * K + (k0 + c - 1)]);
        }
    }
    rocket_bo_fini(fd, &in_all);

    /* ---- batched tile compute: host int64 K-accumulation of int16 partials ---- */
    rocket_task_desc *tasks = malloc(BATCH * sizeof(*tasks));
    uint64_t npu_regs[256] = {0};
    int64_t *acc = calloc((size_t)M * N, sizeof(int64_t));
    int *bm0 = malloc(BATCH * sizeof(int)), *bn0 = malloc(BATCH * sizeof(int));
    int *bMtile = malloc(BATCH * sizeof(int)), *bNtile = malloc(BATCH * sizeof(int));
    size_t *boff = malloc(BATCH * sizeof(size_t));
    if (!tasks || !acc || !bm0 || !bn0 || !bMtile || !bNtile || !boff) { ret = -1; goto free_host; }

    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;

                if (nb == 0) rocket_bo_prep(fd, &regcmd, 1, 0);

                size_t out_off = (size_t)nb * out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma   = (uint32_t)(in_all.dma_address + (size_t)(mi*nKt+ki) * (in_slot / 2)),
                    .weights_dma = (uint32_t)(wt_all.dma_address + (size_t)(ni*nKt+ki) * (wt_slot / 2)),
                    .output_dma  = (uint32_t)(out_all.dma_address + out_off * sizeof(int16_t)),
                    .tasks = npu_regs,
                };
                if ((ret = gen_matmul_int4(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul_int4: gen failed (%d)\n", ret); goto free_host;
                }
                if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto free_host; }
                memcpy((uint64_t *)regcmd.ptr + (size_t)nb * RC_STRIDE, npu_regs,
                       (size_t)p.task_count * sizeof(uint64_t));
                tasks[nb].regcmd = (uint32_t)(regcmd.dma_address + (size_t)nb * RC_STRIDE * sizeof(uint64_t));
                tasks[nb].regcmd_count = p.task_count;
                bm0[nb] = m0; bn0[nb] = n0; bMtile[nb] = Mtile; bNtile[nb] = Ntile; boff[nb] = out_off;
                nb++; done_tiles++;

                if (nb == BATCH || done_tiles == total) {
                    rocket_bo_fini(fd, &regcmd);
                    rocket_bo_prep(fd, &out_all, 1, 0);
                    rocket_bo_fini(fd, &out_all);

                    uint32_t in_h[]  = { in_all.handle, wt_all.handle, regcmd.handle };
                    uint32_t out_h[] = { out_all.handle };
                    if ((ret = rocket_submit_tasks(fd, tasks, nb, in_h, 3, out_h, 1)) != 0) goto free_host;
                    if ((ret = rocket_bo_prep(fd, &out_all, 0, rocket_wait_ns())) != 0) {
                        ROCKET_LOGE("rocket_matmul_int4: WAIT TIMEOUT (%d) M=%d K=%d N=%d "
                                "batch=%d tiles=%d/%d\n", ret, M, K, N, nb, done_tiles, total);
                        goto free_host;
                    }

                    int16_t *ob = (int16_t *)out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        int16_t *slot = ob + boff[j];
                        for (int h = 1; h <= bMtile[j]; h++)
                            for (int nn = 1; nn <= bNtile[j]; nn++)
                                acc[(size_t)(bm0[j] + h - 1) * N + (bn0[j] + nn - 1)] +=
                                    (int64_t)slot[out_idx_i4(bMtile[j], nn, h)];
                    }
                    rocket_bo_fini(fd, &out_all);
                    nb = 0;
                }
            }
        }
    }
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = (int32_t)acc[i];

free_host:
    free(acc); free(tasks); free(bm0); free(bn0); free(bMtile); free(bNtile); free(boff);
free_bos:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &in_all);
    rocket_bo_free(fd, &wt_all); rocket_bo_free(fd, &out_all);
    return ret;
}

/* Raw int4 matmul: no extra Kt cap (single-pass K where CBUF allows -- valid for
 * small-magnitude/[-2,2] data, e.g. the bit-exact gates). In-model callers with
 * [-7,7] data use rocket_matmul_int4_ex(..., kt_cap=480) for saturation safety. */
int rocket_matmul_int4(int fd, int M, int K, int N,
                       const int8_t *A, const int8_t *B, int32_t *C)
{
    return rocket_matmul_int4_ex(fd, M, K, N, A, B, C, 0);
}

/* ============================================================================
 * GROUP-WISE int4 matmul: C_f[M,N] = sum_g a_scale[m,g]*b_scale[n,g] * (int4
 * partial of K-group g), accumulated in fp32. The W4 quality lever over the
 * per-channel/per-tensor scales of rocket_matmul_int4: a separate dequant scale
 * per K-group (group = the K-tile size) bounds the int4 quantization error to a
 * narrow slice of K, so int4 weights approach int8 fidelity (the GPTQ/AWQ regime).
 *
 * A, B are PRE-QUANTIZED int4 in [-7,7] (one per int8_t); a_scale is [M*nG],
 * b_scale is [N*nG] (row-major, nG = K/group), the caller's per-group quant scales
 * (incl. Hadamard, applied to the full K row before the per-group quant). The K-tile
 * is forced to `group`, so each tile IS one quant group and its int16 partial is
 * scaled by that group's a*b scale during the host fp32 accumulation -- no extra
 * NPU work vs the raw int4 path, just a per-tile multiply on readback. group must
 * divide K, be %32, and be <= the int16-saturation bound (49*group < 32767 for
 * [-7,7]; the backend uses 128). Returns 0, or <0 on unsupported shape.
 * ==========================================================================*/
int rocket_matmul_int4_groupwise(int fd, int M, int K, int N,
                                 const int8_t *A, const int8_t *B,
                                 const float *a_scale, const float *b_scale,
                                 float *Cf, int group)
{
    if (!rocket_hw_dtype_supported(rocket_hw_current(), precision_int4))
        return ROCKET_E_UNSUPPORTED;
    if (group < 32 || group % 32 || K % group || M % 4 || 49 * group >= 32767) {
        ROCKET_LOGE("rocket_matmul_int4_groupwise: bad group=%d (need %%32, |K, "
                "49*group<32767, M%%4) K=%d M=%d\n", group, K, M);
        return ROCKET_E_SHAPE;
    }
    int Mt, Kt, Nt;
    if (rocket_matmul_plan_int4(M, K, N, &Mt, &Kt, &Nt) < 0) {
        ROCKET_LOGE("rocket_matmul_int4_groupwise: unsupported shape M=%d K=%d N=%d\n", M, K, N);
        return -1;
    }
    Kt = group;                                   /* one K-tile == one quant group */
    int nG = K / group;
    int nMt = (M + Mt - 1) / Mt, nNt = (N + Nt - 1) / Nt, nKt = nG;
    size_t in_slot  = (size_t)rup(Mt, 4)  * rup(Kt, 32);
    size_t wt_slot  = (size_t)rup(Nt, 64) * rup(Kt, 32);
    size_t out_slot = (size_t)rup(Mt, 4)  * rup(Nt, 16);

    rocket_bo guard = {0}, regcmd = {0}, in_all = {0}, wt_all = {0}, out_all = {0};
    size_t in_sz  = (size_t)nMt * nKt * (in_slot / 2) + CBUF_BANK;
    size_t wt_sz  = (size_t)nNt * nKt * (wt_slot / 2) + CBUF_BANK;
    size_t rc_sz  = (size_t)BATCH * RC_STRIDE * sizeof(uint64_t);
    size_t out_sz = (size_t)BATCH * out_slot * sizeof(int16_t) + CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096, &guard);
    ret |= rocket_bo_alloc(fd, rc_sz,  &regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &in_all);
    ret |= rocket_bo_alloc(fd, wt_sz,  &wt_all);
    ret |= rocket_bo_alloc(fd, out_sz, &out_all);
    if (ret) { ROCKET_LOGE("rocket_matmul_int4_groupwise: BO alloc failed\n"); ret = -1; goto free_bos; }
    if (((in_all.dma_address + in_sz) | (wt_all.dma_address + wt_sz) |
         (out_all.dma_address + out_sz) | (regcmd.dma_address + rc_sz)) >> 32) {
        ROCKET_LOGE("rocket_matmul_int4_groupwise: a BO dma_address exceeds 32 bits\n");
        ret = -1; goto free_bos;
    }

    /* ---- pack weights B[N,K] -> (N/64,K/32,64,32) int4 nibble layout ---- */
    if (rocket_bo_prep(fd, &wt_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(wt_all.ptr, 0, wt_all.size);
    for (int ni = 0; ni < nNt; ni++) {
        int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            uint8_t *slot = (uint8_t *)wt_all.ptr + (size_t)(ni * nKt + ki) * (wt_slot / 2);
            for (int kk = 1; kk <= Ntile; kk++)
                for (int c = 1; c <= Ktile; c++)
                    put_nib(slot, wt_idx_i4(Ktile, kk, c), B[(size_t)(n0 + kk - 1) * K + (k0 + c - 1)]);
        }
    }
    rocket_bo_fini(fd, &wt_all);

    /* ---- pack input A[M,K] -> (K/32,M,32) int4 nibble feature cube ---- */
    if (rocket_bo_prep(fd, &in_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(in_all.ptr, 0, in_all.size);
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            uint8_t *slot = (uint8_t *)in_all.ptr + (size_t)(mi * nKt + ki) * (in_slot / 2);
            for (int h = 1; h <= Mtile; h++)
                for (int c = 1; c <= Ktile; c++)
                    put_nib(slot, feat_idx_i4(Mtile, c, h), A[(size_t)(m0 + h - 1) * K + (k0 + c - 1)]);
        }
    }
    rocket_bo_fini(fd, &in_all);

    /* ---- batched tile compute: host fp32 K-accumulation with per-group scales ---- */
    rocket_task_desc *tasks = malloc(BATCH * sizeof(*tasks));
    uint64_t npu_regs[256] = {0};
    int *bm0 = malloc(BATCH * sizeof(int)), *bn0 = malloc(BATCH * sizeof(int));
    int *bMtile = malloc(BATCH * sizeof(int)), *bNtile = malloc(BATCH * sizeof(int)), *bki = malloc(BATCH * sizeof(int));
    size_t *boff = malloc(BATCH * sizeof(size_t));
    if (!tasks || !bm0 || !bn0 || !bMtile || !bNtile || !bki || !boff) { ret = -1; goto free_host; }
    memset(Cf, 0, (size_t)M * N * sizeof(float));

    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
                if (nb == 0) rocket_bo_prep(fd, &regcmd, 1, 0);
                size_t out_off = (size_t)nb * out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma   = (uint32_t)(in_all.dma_address + (size_t)(mi*nKt+ki) * (in_slot / 2)),
                    .weights_dma = (uint32_t)(wt_all.dma_address + (size_t)(ni*nKt+ki) * (wt_slot / 2)),
                    .output_dma  = (uint32_t)(out_all.dma_address + out_off * sizeof(int16_t)),
                    .tasks = npu_regs,
                };
                if ((ret = gen_matmul_int4(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul_int4_groupwise: gen failed (%d)\n", ret); goto free_host;
                }
                if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto free_host; }
                memcpy((uint64_t *)regcmd.ptr + (size_t)nb * RC_STRIDE, npu_regs,
                       (size_t)p.task_count * sizeof(uint64_t));
                tasks[nb].regcmd = (uint32_t)(regcmd.dma_address + (size_t)nb * RC_STRIDE * sizeof(uint64_t));
                tasks[nb].regcmd_count = p.task_count;
                bm0[nb] = m0; bn0[nb] = n0; bMtile[nb] = Mtile; bNtile[nb] = Ntile;
                bki[nb] = ki; boff[nb] = out_off;
                nb++; done_tiles++;

                if (nb == BATCH || done_tiles == total) {
                    rocket_bo_fini(fd, &regcmd);
                    rocket_bo_prep(fd, &out_all, 1, 0);
                    rocket_bo_fini(fd, &out_all);
                    uint32_t in_h[]  = { in_all.handle, wt_all.handle, regcmd.handle };
                    uint32_t out_h[] = { out_all.handle };
                    if ((ret = rocket_submit_tasks(fd, tasks, nb, in_h, 3, out_h, 1)) != 0) goto free_host;
                    if ((ret = rocket_bo_prep(fd, &out_all, 0, rocket_wait_ns())) != 0) {
                        ROCKET_LOGE("rocket_matmul_int4_groupwise: WAIT TIMEOUT (%d) M=%d K=%d N=%d\n", ret, M, K, N);
                        goto free_host;
                    }
                    int16_t *ob = (int16_t *)out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        int16_t *slot = ob + boff[j];
                        int g = bki[j];
                        for (int h = 1; h <= bMtile[j]; h++) {
                            int mrow = bm0[j] + h - 1;
                            float as = a_scale[(size_t)mrow * nG + g];
                            for (int nn = 1; nn <= bNtile[j]; nn++) {
                                int ncol = bn0[j] + nn - 1;
                                Cf[(size_t)mrow * N + ncol] +=
                                    as * b_scale[(size_t)ncol * nG + g] *
                                    (float)slot[out_idx_i4(bMtile[j], nn, h)];
                            }
                        }
                    }
                    rocket_bo_fini(fd, &out_all);
                    nb = 0;
                }
            }
        }
    }

free_host:
    free(tasks); free(bm0); free(bn0); free(bMtile); free(bNtile); free(bki); free(boff);
free_bos:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &in_all);
    rocket_bo_free(fd, &wt_all); rocket_bo_free(fd, &out_all);
    return ret;
}

/* ============================================================================
 * int16 NPU geometry: the tiling plan + cube index helpers.
 *
 * int16 = fp16's INPUT geometry (in/wt are 2 B whole elements; feature cube C2=8;
 * weight layout (N/16,K/32,16,32) [weight_int16 == weight_fp16, N-group 16];
 * banks_for_i16 is x2, 2 B/elem) with a 4 B output cube (C2=4, == int8). The bf16
 * and tf32 tiled paths below reuse this plan and these index helpers — they share
 * the 2-byte input geometry and the C2=4 output de-tile, differing only in dtype.
 *
 * There is NO native int16 x int16 -> int32 matmul on RK3588: the int32-output conv
 * writes a single 1x16 output tile and never iterates over M-rows or N-groups (the
 * iteration registers were swept exhaustively; only the SATURATING int16-output
 * transposed primitive iterates, N<=32 — see matmul_int16_rocket). The
 * full-precision int16 matmul is rocket_matmul_int16_exact (int8 byte-decomposition,
 * below). NPU K-accum (DPU-EW) is DEAD for int16 too (int32 partials exceed the EW's
 * <=16-bit operand DMA, exactly like int8).
 * ==========================================================================*/

/* CBUF banks for `rows` x Kt int16 (2 B/elem) — same as fp16's banks_for. */
static int banks_for_i16(int rows, int Kt) {
    return ((long)rows * Kt * 2 + CBUF_BANK - 1) / CBUF_BANK;
}

/* int16 NPU layout index math. Feature cube C2=8 (== fp16); weight (N/16,K/32,
 * 16,32) (== weight_fp16); int32-output cube C2=4 (== int8). */
static inline size_t feat_idx_i16(int H, int ch, int h) {   /* input, C2=8 */
    return ((size_t)(ch - 1) / 8) * (size_t)H * 8 + 8 * (size_t)(h - 1) + (ch - 1) % 8;
}
static inline size_t wt_idx_i16(int C, int k, int c) {      /* weight, (N/16,K/32,16,32) */
    return (size_t)((c - 1) / 32) * 32 * 16 + (size_t)((k - 1) / 16) * 16 * C
         + (size_t)((c - 1) % 32) + (size_t)((k - 1) % 16) * 32;
}
static inline size_t out_idx_i16(int H, int ch, int h) {    /* output int32 elem, C2=4 */
    return ((size_t)(ch - 1) / 4) * (size_t)H * 4 + 4 * (size_t)(h - 1) + (ch - 1) % 4;
}

int rocket_matmul_plan_int16(int M, int K, int N, int *pMt, int *pKt, int *pNt)
{
    /* Machine parameters from the active hardware profile (see rocket_matmul_plan). */
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const int MAX_TILE = hw->max_tile, CBUF_BANKS = hw->cbuf_banks;
    if (K % 32 || N % 16 || M % 4 != 0)   /* int16/bf16 N-align 16; M%4 (M==1 padded in one-shot) */
        return ROCKET_E_SHAPE;

    int Mt = (M < MAX_TILE) ? M : MAX_TILE;
    int Nt = (N < MAX_TILE) ? N : MAX_TILE;
    Nt = (Nt / 16) * 16; if (Nt < 16) Nt = 16;         /* int16 N-align 16 */

    const char *e;
    if ((e = getenv("ROCKET_MM_MT"))) { int v = atoi(e); if (v >= 4)  { Mt = (v/4)*4;   if (Mt > M) Mt = M; } }
    if ((e = getenv("ROCKET_MM_NT"))) { int v = atoi(e); if (v >= 16) { Nt = (v/16)*16; if (Nt > N) Nt = N; } }

    int Kt = (K < 16384) ? K : 16384;
    Kt = (Kt / 32) * 32; if (Kt < 32) Kt = 32;
    while (Kt > 32 && banks_for_i16(Mt, Kt) + banks_for_i16(Nt, Kt) > CBUF_BANKS) Kt -= 32;
    while (banks_for_i16(Mt, Kt) + banks_for_i16(Nt, Kt) > CBUF_BANKS && Nt > 16) Nt -= 16;
    while (banks_for_i16(Mt, Kt) + banks_for_i16(Nt, Kt) > CBUF_BANKS && Mt > 4)  Mt -= 4;

    if ((e = getenv("ROCKET_MM_KT"))) {
        int v = atoi(e);
        if (v >= 32) {
            Kt = (v/32)*32; if (Kt > (K/32)*32) Kt = (K/32)*32;
            while (Kt > 32 && banks_for_i16(Mt, Kt) + banks_for_i16(Nt, Kt) > CBUF_BANKS) Kt -= 32;
        }
    }

    if (pMt) *pMt = Mt;
    if (pKt) *pKt = Kt;
    if (pNt) *pNt = Nt;

    int nm = (M + Mt - 1) / Mt, nn = (N + Nt - 1) / Nt, nk = (K + Kt - 1) / Kt;
    /* M4: reject shapes whose int tile-count product would overflow before the
     * size_t BO math (and the `total = nMt*nNt*nKt` in the compute loops below). */
    if ((int64_t)nm * nn * nk > INT32_MAX) return ROCKET_E_SHAPE;
    return nm * nn * nk;
}

/* ---- bit-exact int16 x int16 -> int64 via int8 byte-decomposition -------------
 * The RK3588 NPU has NO native int16 matmul (the documented matmul modes are
 * int8->int32 and fp16->fp32 only). The native int16 conv path can be
 * coaxed into an int16->int16 (SATURATING) transposed-output primitive via
 * tp_org_en (see matmul_int16_rocket / out_idx_i16_tp), but that cannot give a
 * full-precision int32/int64 product. This routine is the full-precision route:
 * decompose each int16 into two signed bytes and run FOUR proven int8 matmuls,
 * recombining in int64 (no saturation). ~4x int8 cost; completeness, not speed.
 *
 * Balanced signed split x = xh*256 + xl, xh,xl in [-128,127] (round-to-nearest):
 *   xl = ((x+128)&0xFF)-128;  xh = (x-xl)>>8.  Avoids the unsigned-low-byte
 *   sign-correction matmuls, at the cost of the top 128 int16 codes: two signed
 *   bytes span [-32896, 32639], so inputs MUST be in [-32768, 32639] (this routine
 *   clamps and is bit-exact within that domain). C = 65536*(Ah.Bh) +
 *   256*(Ah.Bl + Al.Bh) + (Al.Bl). Alignment follows the int8 path: K%32, N%32,
 *   (M%4||1). Returns 0, or <0 on a bad shape / int8-path / alloc error. */
int rocket_matmul_int16_exact(int fd, int M, int K, int N,
                              const int16_t *A, const int16_t *B, int64_t *C)
{
    /* int16_exact decomposes into FOUR int8 matmuls, so its hardware requirement is
     * int8 (not a native int16 output, which RK3588 lacks). Gate on int8. */
    if (!rocket_hw_dtype_supported(rocket_hw_current(), precision_int8))
        return ROCKET_E_UNSUPPORTED;
    if (M == 1)   /* height-1 GEMV broken on HW; pad to 4 rows, return row 0 */
        MM_PAD_M1(fd, K, N, A, B, C, int16_t, int64_t, rocket_matmul_int16_exact);
    int Mt, Kt, Nt;
    if (rocket_matmul_plan_int8(M, K, N, &Mt, &Kt, &Nt) < 0) {
        ROCKET_LOGE("rocket_matmul_int16_exact: unsupported shape M=%d K=%d N=%d "
                "(need K%%32, N%%32, M%%4||1)\n", M, K, N);
        return -1;
    }
    size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;
    int8_t  *Ah = malloc(na), *Al = malloc(na), *Bh = malloc(nb), *Bl = malloc(nb);
    int32_t *HH = malloc(nc*sizeof(int32_t)), *HL = malloc(nc*sizeof(int32_t));
    int32_t *LH = malloc(nc*sizeof(int32_t)), *LL = malloc(nc*sizeof(int32_t));
    int ret = -1;
    if (!Ah||!Al||!Bh||!Bl||!HH||!HL||!LH||!LL) { ROCKET_LOGE("int16_exact: oom\n"); goto done; }

    for (size_t i = 0; i < na; i++) {
        int x = A[i]; if (x > 32639) x = 32639; if (x < -32768) x = -32768;
        int l = ((x + 128) & 0xFF) - 128;  Al[i] = (int8_t)l;  Ah[i] = (int8_t)((x - l) >> 8);
    }
    for (size_t i = 0; i < nb; i++) {
        int x = B[i]; if (x > 32639) x = 32639; if (x < -32768) x = -32768;
        int l = ((x + 128) & 0xFF) - 128;  Bl[i] = (int8_t)l;  Bh[i] = (int8_t)((x - l) >> 8);
    }

    ret = 0;
    ret |= rocket_matmul_int8(fd, M, K, N, Ah, Bh, HH);
    ret |= rocket_matmul_int8(fd, M, K, N, Ah, Bl, HL);
    ret |= rocket_matmul_int8(fd, M, K, N, Al, Bh, LH);
    ret |= rocket_matmul_int8(fd, M, K, N, Al, Bl, LL);
    if (ret) { ret = -1; goto done; }

    for (size_t i = 0; i < nc; i++)
        C[i] = 65536LL*(int64_t)HH[i] + 256LL*((int64_t)HL[i] + (int64_t)LH[i]) + (int64_t)LL[i];

done:
    free(Ah); free(Al); free(Bh); free(Bl); free(HH); free(HL); free(LH); free(LL);
    return ret;
}

/* ============================================================================
 * bf16 x bf16 -> fp32 tiled matmul.
 *
 * bf16 shares the int16 NPU GEOMETRY exactly (2-byte input, feature cube C2=8,
 * weight (N/16,K/32,16,32), 4-byte output cube C2=4), so it reuses banks_for_i16 /
 * feat_idx_i16 / wt_idx_i16 / out_idx_i16 and rocket_matmul_plan_int16. Unlike
 * int16 it has a working native output. It differs only in DTYPE: the output is
 * fp32 (not int32), the K-partials are summed on the HOST in double, and there is
 * NO saturation — fp32 output is the bf16 MAC's natural fp32 accumulation,
 * proven exact-product + fp32-accumulate (matmul_bf16_rocket, max_rel ~1e-6).
 *
 * bf16's prize: with fp32's 8-bit exponent it needs NO activation scaling (the
 * fp16 path's per-row amax-scan hack), so the interface takes fp32 A/B straight
 * through and TRUNCATES to bf16 during the tile scatter (the scatter already
 * touches every element -> the truncation is free; no separate pack pass, no
 * scale/unscale). C is fp32 [M*N] row-major. Alignment K%32, N%16, (M%4||1) —
 * same as fp16/int16. gen_matmul_bf16 carries the HW-validated encodings.
 * ==========================================================================*/

/* fp32 -> bf16 by truncation (high 16 bits). Matches matmul_bf16_rocket's
 * reference; the NPU then does exact bf16 products + fp32 accumulate. */
static inline uint16_t f32_to_bf16(float f) {
    uint32_t b; memcpy(&b, &f, sizeof b); return (uint16_t)(b >> 16);
}

int rocket_matmul_plan_bf16(int M, int K, int N, int *pMt, int *pKt, int *pNt)
{
    /* identical 2-byte-in / 4-byte-out geometry + alignment as int16. */
    return rocket_matmul_plan_int16(M, K, N, pMt, pKt, pNt);
}

int rocket_matmul_bf16(int fd, int M, int K, int N,
                       const float *A, const float *B, float *C)
{
    if (!rocket_hw_dtype_supported(rocket_hw_current(), precision_bfloat16))
        return ROCKET_E_UNSUPPORTED;
    if (M == 1)   /* height-1 GEMV broken on HW; pad to 4 rows, return row 0 */
        MM_PAD_M1(fd, K, N, A, B, C, float, float, rocket_matmul_bf16);
    int Mt, Kt, Nt;
    if (rocket_matmul_plan_bf16(M, K, N, &Mt, &Kt, &Nt) < 0) {
        ROCKET_LOGE("rocket_matmul_bf16: unsupported shape M=%d K=%d N=%d "
                "(need K%%32, N%%16, M%%4||1)\n", M, K, N);
        return -1;
    }
    int nMt = (M + Mt - 1) / Mt, nNt = (N + Nt - 1) / Nt, nKt = (K + Kt - 1) / Kt;
    size_t in_slot  = (size_t)rup(Mt, 4)  * rup(Kt, 32);   /* bf16 elems (uint16_t) */
    size_t wt_slot  = (size_t)rup(Nt, 16) * rup(Kt, 32);   /* bf16 elems */
    size_t out_slot = (size_t)rup(Mt, 4)  * rup(Nt, 16);   /* fp32 elems */

    rocket_bo guard = {0}, regcmd = {0}, in_all = {0}, wt_all = {0}, out_all = {0};
    size_t in_sz  = (size_t)nMt * nKt * in_slot * sizeof(uint16_t) + CBUF_BANK;
    size_t wt_sz  = (size_t)nNt * nKt * wt_slot * sizeof(uint16_t) + CBUF_BANK;
    size_t rc_sz  = (size_t)BATCH * RC_STRIDE * sizeof(uint64_t);
    size_t out_sz = (size_t)BATCH * out_slot * sizeof(float) + CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096, &guard);
    ret |= rocket_bo_alloc(fd, rc_sz,  &regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &in_all);
    ret |= rocket_bo_alloc(fd, wt_sz,  &wt_all);
    ret |= rocket_bo_alloc(fd, out_sz, &out_all);
    if (ret) { ROCKET_LOGE("rocket_matmul_bf16: BO alloc failed\n"); ret = -1; goto free_bos; }
    if (((in_all.dma_address + in_sz) | (wt_all.dma_address + wt_sz) |
         (out_all.dma_address + out_sz) | (regcmd.dma_address + rc_sz)) >> 32) {
        ROCKET_LOGE("rocket_matmul_bf16: a BO dma_address exceeds 32 bits\n");
        ret = -1; goto free_bos;
    }

    /* ---- pack weights B[N,K] -> (N/16,K/32,16,32) bf16 tile layout (truncate) ---- */
    if (rocket_bo_prep(fd, &wt_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(wt_all.ptr, 0, wt_all.size);
    for (int ni = 0; ni < nNt; ni++) {
        int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            uint16_t *slot = (uint16_t *)wt_all.ptr + (size_t)(ni * nKt + ki) * wt_slot;
            for (int kk = 1; kk <= Ntile; kk++)
                for (int c = 1; c <= Ktile; c++)
                    slot[wt_idx_i16(Ktile, kk, c)] = f32_to_bf16(B[(size_t)(n0 + kk - 1) * K + (k0 + c - 1)]);
        }
    }
    rocket_bo_fini(fd, &wt_all);

    /* ---- pack input A[M,K] -> (K/8,M,8) bf16 feature cube (truncate) ---- */
    if (rocket_bo_prep(fd, &in_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(in_all.ptr, 0, in_all.size);
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            uint16_t *slot = (uint16_t *)in_all.ptr + (size_t)(mi * nKt + ki) * in_slot;
            for (int h = 1; h <= Mtile; h++)
                for (int c = 1; c <= Ktile; c++)
                    slot[feat_idx_i16(Mtile, c, h)] = f32_to_bf16(A[(size_t)(m0 + h - 1) * K + (k0 + c - 1)]);
        }
    }
    rocket_bo_fini(fd, &in_all);

    /* ---- batched tile compute: host fp32 (double) K-accumulation of fp32 partials ---- */
    rocket_task_desc *tasks = malloc(BATCH * sizeof(*tasks));
    uint64_t npu_regs[256] = {0};
    double *acc = calloc((size_t)M * N, sizeof(double));
    int *bm0 = malloc(BATCH * sizeof(int)), *bn0 = malloc(BATCH * sizeof(int));
    int *bMtile = malloc(BATCH * sizeof(int)), *bNtile = malloc(BATCH * sizeof(int));
    size_t *boff = malloc(BATCH * sizeof(size_t));
    if (!tasks || !acc || !bm0 || !bn0 || !bMtile || !bNtile || !boff) { ret = -1; goto free_host; }

    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;

                if (nb == 0) rocket_bo_prep(fd, &regcmd, 1, 0);

                size_t out_off = (size_t)nb * out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma   = (uint32_t)(in_all.dma_address + (size_t)(mi*nKt+ki) * in_slot * sizeof(uint16_t)),
                    .weights_dma = (uint32_t)(wt_all.dma_address + (size_t)(ni*nKt+ki) * wt_slot * sizeof(uint16_t)),
                    .output_dma  = (uint32_t)(out_all.dma_address + out_off * sizeof(float)),
                    .tasks = npu_regs,
                };
                if ((ret = gen_matmul_bf16(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul_bf16: gen failed (%d)\n", ret); goto free_host;
                }
                if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto free_host; }
                memcpy((uint64_t *)regcmd.ptr + (size_t)nb * RC_STRIDE, npu_regs,
                       (size_t)p.task_count * sizeof(uint64_t));
                tasks[nb].regcmd = (uint32_t)(regcmd.dma_address + (size_t)nb * RC_STRIDE * sizeof(uint64_t));
                tasks[nb].regcmd_count = p.task_count;
                bm0[nb] = m0; bn0[nb] = n0; bMtile[nb] = Mtile; bNtile[nb] = Ntile; boff[nb] = out_off;
                nb++; done_tiles++;

                if (nb == BATCH || done_tiles == total) {
                    rocket_bo_fini(fd, &regcmd);
                    rocket_bo_prep(fd, &out_all, 1, 0);
                    rocket_bo_fini(fd, &out_all);

                    uint32_t in_h[]  = { in_all.handle, wt_all.handle, regcmd.handle };
                    uint32_t out_h[] = { out_all.handle };
                    if ((ret = rocket_submit_tasks(fd, tasks, nb, in_h, 3, out_h, 1)) != 0) goto free_host;
                    if ((ret = rocket_bo_prep(fd, &out_all, 0, rocket_wait_ns())) != 0) {
                        ROCKET_LOGE("rocket_matmul_bf16: WAIT TIMEOUT (%d) M=%d K=%d N=%d "
                                "batch=%d tiles=%d/%d\n", ret, M, K, N, nb, done_tiles, total);
                        goto free_host;
                    }

                    float *ob = (float *)out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        float *slot = ob + boff[j];
                        for (int h = 1; h <= bMtile[j]; h++)
                            for (int nn = 1; nn <= bNtile[j]; nn++)
                                acc[(size_t)(bm0[j] + h - 1) * N + (bn0[j] + nn - 1)] +=
                                    (double)slot[out_idx_i16(bMtile[j], nn, h)];
                    }
                    rocket_bo_fini(fd, &out_all);
                    nb = 0;
                }
            }
        }
    }
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = (float)acc[i];

free_host:
    free(acc); free(tasks); free(bm0); free(bn0); free(bMtile); free(bNtile); free(boff);
free_bos:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &in_all);
    rocket_bo_free(fd, &wt_all); rocket_bo_free(fd, &out_all);
    return ret;
}

/* ============================================================================
 * fp16 x fp16 -> FP32 OUTPUT tiled matmul.
 *
 * Self-contained sibling of rocket_matmul_fp16 (same idiom as the int8/bf16/tf32
 * paths: a separate function so the tuned fp16 path stays byte-for-byte unchanged,
 * zero regression risk). IDENTICAL fp16 input packing (feat_idx / wt_idx, no
 * truncation) and the same rocket_matmul_plan tiling — the plan sizes the CBUF for
 * the 2-byte fp16 INPUTS, which output precision does not change (the result leaves
 * via WDMA to DRAM, not the CBUF). The ONLY deltas vs the fp16 path are on the
 * output side:
 *   - gen_matmul_fp16 with fp32tofp16=0  -> DPU emits the full fp32 accumulator
 *     (size_e=3, surf_add*4) instead of narrowing it to fp16 (size_e=1, surf_add*2);
 *   - the output cube is therefore C2=4 (16 B / 4-byte elem), read back with
 *     out_idx_i16 (the same C2=4 de-tile the int32/bf16/tf32 paths use), not the
 *     C2=8 fp16 detile;
 *   - the out BO is 4 B/elem; host K-accum is fp64 (double), like bf16/tf32.
 *
 * Net: the fp16 path rounds each K-partial to fp16 *before* the host fp32 sum and
 * then narrows the final result to fp16; this path keeps every partial in fp32 and
 * returns fp32 — the genuine fp16-input dot product. Costs 2x output readback bytes
 * (fp32 vs fp16 cube), so it's opt-in (the prefill path is readback-bound).
 * ==========================================================================*/
int rocket_matmul_fp16_f32out(int fd, int M, int K, int N,
                              const _Float16 *A, const _Float16 *B, float *C)
{
    if (!rocket_hw_dtype_supported(rocket_hw_current(), precision_float16))
        return ROCKET_E_UNSUPPORTED;
    if (M == 1)   /* height-1 GEMV broken on HW; pad to 4 rows, return row 0 */
        MM_PAD_M1(fd, K, N, A, B, C, _Float16, float, rocket_matmul_fp16_f32out);
    int Mt, Kt, Nt;
    if (rocket_matmul_plan(M, K, N, &Mt, &Kt, &Nt) < 0) {
        ROCKET_LOGE("rocket_matmul_fp16_f32out: unsupported shape M=%d K=%d N=%d "
                "(need K%%32, N%%16, M%%4||1)\n", M, K, N);
        return -1;
    }
    int nMt = (M + Mt - 1) / Mt, nNt = (N + Nt - 1) / Nt, nKt = (K + Kt - 1) / Kt;
    size_t in_slot  = (size_t)rup(Mt, 4)  * rup(Kt, 32);   /* fp16 elems          */
    size_t wt_slot  = (size_t)rup(Nt, 16) * rup(Kt, 32);   /* fp16 elems          */
    size_t out_slot = (size_t)rup(Mt, 4)  * rup(Nt, 16);   /* fp32 elems (C2=4)   */

    rocket_bo guard = {0}, regcmd = {0}, in_all = {0}, wt_all = {0}, out_all = {0};
    size_t in_sz  = (size_t)nMt * nKt * in_slot * sizeof(_Float16) + CBUF_BANK;
    size_t wt_sz  = (size_t)nNt * nKt * wt_slot * sizeof(_Float16) + CBUF_BANK;
    size_t rc_sz  = (size_t)BATCH * RC_STRIDE * sizeof(uint64_t);
    size_t out_sz = (size_t)BATCH * out_slot * sizeof(float) + CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096, &guard);
    ret |= rocket_bo_alloc(fd, rc_sz,  &regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &in_all);
    ret |= rocket_bo_alloc(fd, wt_sz,  &wt_all);
    ret |= rocket_bo_alloc(fd, out_sz, &out_all);
    if (ret) { ROCKET_LOGE("rocket_matmul_fp16_f32out: BO alloc failed\n"); ret = -1; goto free_bos; }
    if (((in_all.dma_address + in_sz) | (wt_all.dma_address + wt_sz) |
         (out_all.dma_address + out_sz) | (regcmd.dma_address + rc_sz)) >> 32) {
        ROCKET_LOGE("rocket_matmul_fp16_f32out: a BO dma_address exceeds 32 bits\n");
        ret = -1; goto free_bos;
    }

    /* ---- pack weights B[N,K] -> fp16 (N/16,K/32,16,32) tile layout (no truncation) ---- */
    if (rocket_bo_prep(fd, &wt_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(wt_all.ptr, 0, wt_all.size);
    for (int ni = 0; ni < nNt; ni++) {
        int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            _Float16 *slot = (_Float16 *)wt_all.ptr + (size_t)(ni * nKt + ki) * wt_slot;
            wt_scatter_tile(slot, B, K, n0, k0, Ntile, Ktile);
        }
    }
    rocket_bo_fini(fd, &wt_all);

    /* ---- pack input A[M,K] -> fp16 (K/8,M,8) feature cube (no truncation) ---- */
    if (rocket_bo_prep(fd, &in_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(in_all.ptr, 0, in_all.size);
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            _Float16 *slot = (_Float16 *)in_all.ptr + (size_t)(mi * nKt + ki) * in_slot;
            feat_scatter_tile(slot, A, K, m0, k0, Mtile, Ktile);
        }
    }
    rocket_bo_fini(fd, &in_all);

    /* ---- batched tile compute: fp32 NPU output, host fp64 K-accumulation ---- */
    rocket_task_desc *tasks = malloc(BATCH * sizeof(*tasks));
    uint64_t npu_regs[256] = {0};
    double *acc = calloc((size_t)M * N, sizeof(double));
    int *bm0 = malloc(BATCH * sizeof(int)), *bn0 = malloc(BATCH * sizeof(int));
    int *bMtile = malloc(BATCH * sizeof(int)), *bNtile = malloc(BATCH * sizeof(int));
    size_t *boff = malloc(BATCH * sizeof(size_t));
    if (!tasks || !acc || !bm0 || !bn0 || !bMtile || !bNtile || !boff) { ret = -1; goto free_host; }

    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;

                if (nb == 0) rocket_bo_prep(fd, &regcmd, 1, 0);

                size_t out_off = (size_t)nb * out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma   = (uint32_t)(in_all.dma_address + (size_t)(mi*nKt+ki) * in_slot * sizeof(_Float16)),
                    .weights_dma = (uint32_t)(wt_all.dma_address + (size_t)(ni*nKt+ki) * wt_slot * sizeof(_Float16)),
                    .output_dma  = (uint32_t)(out_all.dma_address + out_off * sizeof(float)),
                    .tasks = npu_regs, .fp32tofp16 = 0,   /* <-- full fp32 accumulator out */
                };
                if ((ret = gen_matmul_fp16(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul_fp16_f32out: gen failed (%d)\n", ret); goto free_host;
                }
                if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto free_host; }
                memcpy((uint64_t *)regcmd.ptr + (size_t)nb * RC_STRIDE, npu_regs,
                       (size_t)p.task_count * sizeof(uint64_t));
                tasks[nb].regcmd = (uint32_t)(regcmd.dma_address + (size_t)nb * RC_STRIDE * sizeof(uint64_t));
                tasks[nb].regcmd_count = p.task_count;
                bm0[nb] = m0; bn0[nb] = n0; bMtile[nb] = Mtile; bNtile[nb] = Ntile; boff[nb] = out_off;
                nb++; done_tiles++;

                if (nb == BATCH || done_tiles == total) {
                    rocket_bo_fini(fd, &regcmd);
                    rocket_bo_prep(fd, &out_all, 1, 0);
                    rocket_bo_fini(fd, &out_all);

                    uint32_t in_h[]  = { in_all.handle, wt_all.handle, regcmd.handle };
                    uint32_t out_h[] = { out_all.handle };
                    if ((ret = rocket_submit_tasks(fd, tasks, nb, in_h, 3, out_h, 1)) != 0) goto free_host;
                    if ((ret = rocket_bo_prep(fd, &out_all, 0, rocket_wait_ns())) != 0) {
                        ROCKET_LOGE("rocket_matmul_fp16_f32out: WAIT TIMEOUT (%d) M=%d K=%d N=%d "
                                "batch=%d tiles=%d/%d\n", ret, M, K, N, nb, done_tiles, total);
                        goto free_host;
                    }

                    float *ob = (float *)out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        float *slot = ob + boff[j];
                        for (int h = 1; h <= bMtile[j]; h++)
                            for (int nn = 1; nn <= bNtile[j]; nn++)
                                acc[(size_t)(bm0[j] + h - 1) * N + (bn0[j] + nn - 1)] +=
                                    (double)slot[out_idx_i16(bMtile[j], nn, h)];
                    }
                    rocket_bo_fini(fd, &out_all);
                    nb = 0;
                }
            }
        }
    }
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = (float)acc[i];

free_host:
    free(acc); free(tasks); free(bm0); free(bn0); free(bMtile); free(bNtile); free(boff);
free_bos:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &in_all);
    rocket_bo_free(fd, &wt_all); rocket_bo_free(fd, &out_all);
    return ret;
}

/* ============================================================================
 * tf32 x tf32 -> fp32 tiled matmul — the LAST datatype rung; closes the
 * RK3588 native matrix (fp16/int8/int4/int16/bf16/tf32) end-to-end.
 *
 * The FIRST 4-byte-INPUT tiled path. tf32 = fp32 RANGE (8-bit exp) + fp16
 * PRECISION (10-bit mantissa) in a 4-byte fp32 container: you feed RAW fp32, the
 * MAC rounds the mantissa to 10 bits and accumulates in fp32. Like bf16 it needs
 * NO activation scaling and is a pure FLOAT path (output fp32, host double K-accum,
 * NO saturation), so the interface mirrors rocket_matmul_bf16 — fp32 A/B straight
 * through. The ONLY difference from bf16 is the 4-byte geometry: feature cube C2=4
 * (vs bf16's 8), weight tile (N/16, K/16, 16, 16) — K-group HALVES to 16 for 4-byte
 * (N-group stays 16), banks count 4 B/elem, AND there is NO truncation on the
 * scatter (bf16 narrowed to its 16 high bits; tf32 stores the raw fp32, the HW
 * rounds). gen_matmul_tf32 carries the per-stage precision (CNA/CORE=7, DPU=fp32)
 * + data_entries=K/16. Alignment K%16, N%16, (M%4||1). HW-validated
 * (single-task test matmul_tf32_rocket; this is its tiled sibling).
 *
 * tf32 is the LOWEST-value rung (half-rate MAC, and bf16 already gives fp32 range
 * at full speed) — completeness, not a workload. No in-model backend; bf16 wins.
 * ==========================================================================*/

/* CBUF banks for `rows` x Kt tf32 (4 B/elem — x2 vs bf16's 2 B). */
static int banks_for_tf32(int rows, int Kt) {
    return ((long)rows * Kt * 4 + CBUF_BANK - 1) / CBUF_BANK;
}

/* tf32 NPU layout index math. Feature cube C2=4 (4-byte atom); weight
 * (N/16,K/16,16,16) (== weight_tf32). The fp32 output reuses out_idx_i16 (cube
 * C2=4) — identical to the int16/bf16 fp32-out writer. */
static inline size_t feat_idx_tf32(int H, int ch, int h) {   /* input, C2=4 */
    return ((size_t)(ch - 1) / 4) * (size_t)H * 4 + 4 * (size_t)(h - 1) + (ch - 1) % 4;
}
static inline size_t wt_idx_tf32(int C, int k, int c) {      /* weight, (N/16,K/16,16,16) */
    return (size_t)((c - 1) / 16) * 16 * 16 + (size_t)((k - 1) / 16) * 16 * C
         + (size_t)((c - 1) % 16) + (size_t)((k - 1) % 16) * 16;
}

int rocket_matmul_plan_tf32(int M, int K, int N, int *pMt, int *pKt, int *pNt)
{
    /* Machine parameters from the active hardware profile (see rocket_matmul_plan). */
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const int MAX_TILE = hw->max_tile, CBUF_BANKS = hw->cbuf_banks;
    if (K % 16 || N % 16 || M % 4 != 0)   /* tf32: K/N-group 16; M%4 (M==1 padded in one-shot) */
        return ROCKET_E_SHAPE;

    int Mt = (M < MAX_TILE) ? M : MAX_TILE;
    int Nt = (N < MAX_TILE) ? N : MAX_TILE;
    Nt = (Nt / 16) * 16; if (Nt < 16) Nt = 16;

    const char *e;
    if ((e = getenv("ROCKET_MM_MT"))) { int v = atoi(e); if (v >= 4)  { Mt = (v/4)*4;   if (Mt > M) Mt = M; } }
    if ((e = getenv("ROCKET_MM_NT"))) { int v = atoi(e); if (v >= 16) { Nt = (v/16)*16; if (Nt > N) Nt = N; } }

    /* 4-byte halves the Kt ceiling: weight_bytes_per_kernel = Kt*4 must be <= 32768
     * (gen_matmul_tf32 returns -2 above that), so Kt <= 8192. K-group is 16. */
    int Kt = (K < 8192) ? K : 8192;
    Kt = (Kt / 16) * 16; if (Kt < 16) Kt = 16;
    while (Kt > 16 && banks_for_tf32(Mt, Kt) + banks_for_tf32(Nt, Kt) > CBUF_BANKS) Kt -= 16;
    while (banks_for_tf32(Mt, Kt) + banks_for_tf32(Nt, Kt) > CBUF_BANKS && Nt > 16) Nt -= 16;
    while (banks_for_tf32(Mt, Kt) + banks_for_tf32(Nt, Kt) > CBUF_BANKS && Mt > 4)  Mt -= 4;

    if ((e = getenv("ROCKET_MM_KT"))) {
        int v = atoi(e);
        if (v >= 16) {
            Kt = (v/16)*16; if (Kt > (K/16)*16) Kt = (K/16)*16; if (Kt > 8192) Kt = 8192;
            while (Kt > 16 && banks_for_tf32(Mt, Kt) + banks_for_tf32(Nt, Kt) > CBUF_BANKS) Kt -= 16;
        }
    }

    if (pMt) *pMt = Mt;
    if (pKt) *pKt = Kt;
    if (pNt) *pNt = Nt;

    int nm = (M + Mt - 1) / Mt, nn = (N + Nt - 1) / Nt, nk = (K + Kt - 1) / Kt;
    /* M4: reject shapes whose int tile-count product would overflow before the
     * size_t BO math (and the `total = nMt*nNt*nKt` in the compute loops below). */
    if ((int64_t)nm * nn * nk > INT32_MAX) return ROCKET_E_SHAPE;
    return nm * nn * nk;
}

int rocket_matmul_tf32(int fd, int M, int K, int N,
                       const float *A, const float *B, float *C)
{
    if (!rocket_hw_dtype_supported(rocket_hw_current(), precision_tf32))
        return ROCKET_E_UNSUPPORTED;
    if (M == 1)   /* height-1 GEMV broken on HW; pad to 4 rows, return row 0 */
        MM_PAD_M1(fd, K, N, A, B, C, float, float, rocket_matmul_tf32);
    int Mt, Kt, Nt;
    if (rocket_matmul_plan_tf32(M, K, N, &Mt, &Kt, &Nt) < 0) {
        ROCKET_LOGE("rocket_matmul_tf32: unsupported shape M=%d K=%d N=%d "
                "(need K%%16, N%%16, M%%4||1)\n", M, K, N);
        return -1;
    }
    int nMt = (M + Mt - 1) / Mt, nNt = (N + Nt - 1) / Nt, nKt = (K + Kt - 1) / Kt;
    size_t in_slot  = (size_t)rup(Mt, 4)  * rup(Kt, 16);   /* fp32 elems (K-align 16) */
    size_t wt_slot  = (size_t)rup(Nt, 16) * rup(Kt, 16);   /* fp32 elems */
    size_t out_slot = (size_t)rup(Mt, 4)  * rup(Nt, 16);   /* fp32 elems */

    rocket_bo guard = {0}, regcmd = {0}, in_all = {0}, wt_all = {0}, out_all = {0};
    size_t in_sz  = (size_t)nMt * nKt * in_slot * sizeof(float) + CBUF_BANK;
    size_t wt_sz  = (size_t)nNt * nKt * wt_slot * sizeof(float) + CBUF_BANK;
    size_t rc_sz  = (size_t)BATCH * RC_STRIDE * sizeof(uint64_t);
    size_t out_sz = (size_t)BATCH * out_slot * sizeof(float) + CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096, &guard);
    ret |= rocket_bo_alloc(fd, rc_sz,  &regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &in_all);
    ret |= rocket_bo_alloc(fd, wt_sz,  &wt_all);
    ret |= rocket_bo_alloc(fd, out_sz, &out_all);
    if (ret) { ROCKET_LOGE("rocket_matmul_tf32: BO alloc failed\n"); ret = -1; goto free_bos; }
    if (((in_all.dma_address + in_sz) | (wt_all.dma_address + wt_sz) |
         (out_all.dma_address + out_sz) | (regcmd.dma_address + rc_sz)) >> 32) {
        ROCKET_LOGE("rocket_matmul_tf32: a BO dma_address exceeds 32 bits\n");
        ret = -1; goto free_bos;
    }

    /* ---- pack weights B[N,K] -> (N/16,K/16,16,16) fp32 tile layout (RAW, no trunc) ---- */
    if (rocket_bo_prep(fd, &wt_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(wt_all.ptr, 0, wt_all.size);
    for (int ni = 0; ni < nNt; ni++) {
        int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            float *slot = (float *)wt_all.ptr + (size_t)(ni * nKt + ki) * wt_slot;
            for (int kk = 1; kk <= Ntile; kk++)
                for (int c = 1; c <= Ktile; c++)
                    slot[wt_idx_tf32(Ktile, kk, c)] = B[(size_t)(n0 + kk - 1) * K + (k0 + c - 1)];
        }
    }
    rocket_bo_fini(fd, &wt_all);

    /* ---- pack input A[M,K] -> (K/4,M,4) fp32 feature cube (RAW, no trunc) ---- */
    if (rocket_bo_prep(fd, &in_all, 1, 0) != 0) { ret = -1; goto free_bos; }
    memset(in_all.ptr, 0, in_all.size);
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            float *slot = (float *)in_all.ptr + (size_t)(mi * nKt + ki) * in_slot;
            for (int h = 1; h <= Mtile; h++)
                for (int c = 1; c <= Ktile; c++)
                    slot[feat_idx_tf32(Mtile, c, h)] = A[(size_t)(m0 + h - 1) * K + (k0 + c - 1)];
        }
    }
    rocket_bo_fini(fd, &in_all);

    /* ---- batched tile compute: host fp32 (double) K-accumulation of fp32 partials ---- */
    rocket_task_desc *tasks = malloc(BATCH * sizeof(*tasks));
    uint64_t npu_regs[256] = {0};
    double *acc = calloc((size_t)M * N, sizeof(double));
    int *bm0 = malloc(BATCH * sizeof(int)), *bn0 = malloc(BATCH * sizeof(int));
    int *bMtile = malloc(BATCH * sizeof(int)), *bNtile = malloc(BATCH * sizeof(int));
    size_t *boff = malloc(BATCH * sizeof(size_t));
    if (!tasks || !acc || !bm0 || !bn0 || !bMtile || !bNtile || !boff) { ret = -1; goto free_host; }

    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (N - n0 < Nt) ? (N - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;

                if (nb == 0) rocket_bo_prep(fd, &regcmd, 1, 0);

                size_t out_off = (size_t)nb * out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma   = (uint32_t)(in_all.dma_address + (size_t)(mi*nKt+ki) * in_slot * sizeof(float)),
                    .weights_dma = (uint32_t)(wt_all.dma_address + (size_t)(ni*nKt+ki) * wt_slot * sizeof(float)),
                    .output_dma  = (uint32_t)(out_all.dma_address + out_off * sizeof(float)),
                    .tasks = npu_regs,
                };
                if ((ret = gen_matmul_tf32(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul_tf32: gen failed (%d)\n", ret); goto free_host;
                }
                if (MM_REGCMD_OVERFLOWS(p.task_count, RC_STRIDE)) { ret = -1; goto free_host; }
                memcpy((uint64_t *)regcmd.ptr + (size_t)nb * RC_STRIDE, npu_regs,
                       (size_t)p.task_count * sizeof(uint64_t));
                tasks[nb].regcmd = (uint32_t)(regcmd.dma_address + (size_t)nb * RC_STRIDE * sizeof(uint64_t));
                tasks[nb].regcmd_count = p.task_count;
                bm0[nb] = m0; bn0[nb] = n0; bMtile[nb] = Mtile; bNtile[nb] = Ntile; boff[nb] = out_off;
                nb++; done_tiles++;

                if (nb == BATCH || done_tiles == total) {
                    rocket_bo_fini(fd, &regcmd);
                    rocket_bo_prep(fd, &out_all, 1, 0);
                    rocket_bo_fini(fd, &out_all);

                    uint32_t in_h[]  = { in_all.handle, wt_all.handle, regcmd.handle };
                    uint32_t out_h[] = { out_all.handle };
                    if ((ret = rocket_submit_tasks(fd, tasks, nb, in_h, 3, out_h, 1)) != 0) goto free_host;
                    if ((ret = rocket_bo_prep(fd, &out_all, 0, rocket_wait_ns())) != 0) {
                        ROCKET_LOGE("rocket_matmul_tf32: WAIT TIMEOUT (%d) M=%d K=%d N=%d "
                                "batch=%d tiles=%d/%d\n", ret, M, K, N, nb, done_tiles, total);
                        goto free_host;
                    }

                    float *ob = (float *)out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        float *slot = ob + boff[j];
                        for (int h = 1; h <= bMtile[j]; h++)
                            for (int nn = 1; nn <= bNtile[j]; nn++)
                                acc[(size_t)(bm0[j] + h - 1) * N + (bn0[j] + nn - 1)] +=
                                    (double)slot[out_idx_i16(bMtile[j], nn, h)];
                    }
                    rocket_bo_fini(fd, &out_all);
                    nb = 0;
                }
            }
        }
    }
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = (float)acc[i];

free_host:
    free(acc); free(tasks); free(bm0); free(bn0); free(bMtile); free(bNtile); free(boff);
free_bos:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &in_all);
    rocket_bo_free(fd, &wt_all); rocket_bo_free(fd, &out_all);
    return ret;
}
