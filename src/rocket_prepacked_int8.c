// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_prepacked_int8.c — resident-weight int8 (W8A8) matmul path.
 *
 * The int8 sibling of rocket_prepacked.c's fp16 pack-weights-once path. A
 * pre-quantized int8 weight B[N,K] is scattered into resident per-worker NPU BOs
 * ONCE (rocket_i8_weights_pack); every later matmul reuses them and only packs A
 * (rocket_matmul_int8_prepacked). This removes the per-call int8 weight scatter
 * ("packB") that the one-shot rocket_matmul_int8 pays every call — the same lever
 * proved for fp16 (~9% of per-matmul wall), now on the int8 path whose
 * 11GB whole-model footprint fits across the 5 worker fds' 4GB IOVA windows.
 *
 * DELIBERATELY self-contained (it reimplements the int8 tiled host-accum compute
 * here) rather than threading int8 through rocket_prepacked.c's fp16 rkw_run, so
 * the tuned fp16 path stays byte-for-byte unchanged — the same zero-regression
 * stance the one-shot int8 path took in rocket_matmul.c. The compute mirrors
 * rocket_matmul_int8's host-int64 K-accum branch (int8 NPU K-accum is HW-dead:
 * the DPU EW operand DMA is <=16-bit, so int32 partials can't accumulate on-chip;
 * they are summed on the host in int64, which is integer-EXACT -> bit-identical
 * to the one-shot path and to the int64 CPU reference).
 *
 * TWO MODES, one compute core (rki_thread), selected by the weight's `group`:
 *
 *   group == 0  PER-CHANNEL. Raw int32 C[M,N], host int64 K-accum. The W8A8
 *               in-model path; the backend owns the per-row/per-channel dequant.
 *   group  > 0  GROUP-WISE. Each K-tile is kept inside ONE quant group, so its
 *               int32 partial can be scaled by that group's a_scale[m,g]*b_scale[n,g]
 *               as it is read back, and fp32-accumulated into Cf[M,N]. This is what a
 *               NATIVELY quantized weight needs — a GGUF MXFP4/Q8_0/Q4_K block carries
 *               one scale per K-block, and the NPU cannot apply a K-blocked scale
 *               on-chip (at the output stage K is fully contracted). The integer partials
 *               are already read back to the host at every K-tile boundary (on-device
 *               integer K-accum is HW-dead), so the block scale rides along a readback
 *               that is being paid for regardless: it fuses into the accumulate loop and
 *               measures +0.6% over the per-channel mode. Resident-int8 + group-wise is
 *               the combination that deletes the per-micro-batch host dequant for a
 *               GGUF-quant weight: the int8 codes never leave the NPU.
 *
 * Both modes share the A-pack, regcmd, batching and submit; they differ only in the
 * readback accumulate and the final scatter. Each is bit-exact against its one-shot oracle
 * (matmul_int8_prepacked_rocket, matmul_int8_prepacked_gw_rocket).
 *
 * Layout: weights resident per-NAME (the per-worker int8 wt BOs, the only
 * per-weight alloc); input/output/regcmd/host-accum scratch is shared per-(M,K,N,group)
 * shape on the ctx (reused every call — the two modes tile K differently, so they
 * cannot share a scratch). N is fanned across worker fds (one
 * drm_sched entity per fd -> the 3 NPU cores run in parallel). Each worker owns
 * its scratch and runs serially, so the shared scratch is race-free.
 *
 * int8 tile-layout constants (cf. rocket_matmul.c): input feature cube C2=16,
 * weight k-group 32 (== weight_int8()), int32-output cube C2=4. The index helpers
 * below are duplicated verbatim from rocket_matmul.c (feat_idx_i8/wt_idx_i8/
 * out_idx_i8) and pinned bit-exact by the standalone test vs the one-shot oracle.
 */
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "npu_matmul.h"
#include "rocket_matmul.h"
#include "rocket_affinity.h"
#include "rocket_log.h"     // centralized log channel
#include "rocket_chain.h"   // contiguous self-chaining regcmd layout (batched submit)

#define RKI_MAX_WORKERS 8
#define RKI_MAX_SLOTS   32
#define I8_BATCH        64    /* tiles (tasks) per NPU job — matches rocket_matmul.c */
#define I8_RC_STRIDE    128   /* u64 words reserved per task in the regcmd BO        */
#define I8_CBUF_BANK    32768 /* tail pad so a tile's DMA never runs off the BO end  */

/* ============================================================================
 * SECTION — int8 tile-layout index math and small helpers
 * ==========================================================================*/

/* Round x up to a multiple of a, in size_t (these feed slot/BO allocation sizes;
 * computing in int would truncate before the widen at the call site). */
static size_t i8_rup(int x, int a) { return (((size_t)x + a - 1) / a) * a; }

/* Output-fence wait deadline (ns). Mirrors rocket_matmul.c's rocket_wait_ns. */
static long i8_wait_ns(void) {
    static _Atomic long ns = -1;
    if (ns < 0) {
        const char *e = getenv("ROCKET_WAIT_MS");
        long ms = e ? atol(e) : 8000;
        if (ms < 1) ms = 8000;
        ns = ms * 1000000L;
    }
    return ns;
}

/* int8 NPU tile-layout index math — IDENTICAL to rocket_matmul.c's static
 * inlines (input feature cube C2=16, weight k-group 32, int32-output cube C2=4).
 * 1-based indices, matching weight_int8(). Kept in sync by the bit-exact test. */
static inline size_t i8_feat_idx(int H, int ch, int h) {   /* input, C2=16 */
    return ((size_t)(ch - 1) / 16) * (size_t)H * 16 + 16 * (size_t)(h - 1) + (ch - 1) % 16;
}
static inline size_t i8_wt_idx(int C, int k, int c) {      /* weight, k-group 32 */
    return (size_t)((c - 1) / 32) * 32 * 32 + (size_t)((k - 1) / 32) * 32 * C
         + (size_t)((c - 1) % 32) + (size_t)((k - 1) % 32) * 32;
}
static inline size_t i8_out_idx(int H, int ch, int h) {    /* output, C2=4 */
    return ((size_t)(ch - 1) / 4) * (size_t)H * 4 + 4 * (size_t)(h - 1) + (ch - 1) % 4;
}

/* One worker's N-slice: its column range, resolved int8 tiling plan, slot sizes,
 * and its shared per-shape scratch BOs + host-accum arrays. */
typedef struct {
    int n0, nsub;                 /* output-column slice C[:, n0:n0+nsub)            */
    int Mt, Kt, Nt;
    int nMt, nNt, nKt;
    int kt_per_group;             /* K-tiles inside one quant group (group-wise; 0 else) */
    size_t in_slot, wt_slot, out_slot;   /* int8 / int8 / int32 elems per tile       */

    rocket_bo guard, regcmd, in_all, out_all;   /* shared scratch (NOT the weight)   */
    rocket_task_desc *tasks;                     /* I8_BATCH descriptors              */
    int64_t *acc;                                /* M*nsub int64 K-accum (per-channel) */
    float   *facc;                               /* M*nsub fp32 K-accum (group-wise)  */
    int    *bm0, *bn0, *bMtile, *bNtile, *bg;    /* I8_BATCH each; bg = tile's K-group */
    size_t *boff;                                /* I8_BATCH                          */
} rki_worker;

/* Per-(M,K,N,group) compute scratch, cached on the ctx, shared by every same-shape
 * resident weight (the resident wt BO is the only per-weight alloc). group is part of
 * the key: the two modes tile K differently (group-wise pins a K-tile inside a quant
 * group), so their tile geometry — and hence every slot size here — differs. */
typedef struct {
    int M, K, N, nt, group;
    rki_worker w[RKI_MAX_WORKERS];
} rki_scratch;

/* Per-NAME resident int8 weight: just the scattered per-worker int8 wt BOs + a
 * borrowed pointer to the shared per-shape scratch. group == 0 is per-channel. */
struct rocket_i8_weights {
    int M, K, N, nt, group;
    rocket_bo   wt[RKI_MAX_WORKERS];
    rki_scratch *sc;
};

struct rocket_i8_ctx {
    int nthreads;
    const struct rocket_hw_profile *hw; /* active machine-parameter profile (the multi-chip profile seam) */
    int fd[RKI_MAX_WORKERS];
    rki_scratch *scache[RKI_MAX_SLOTS];
    int nscache;
};

/* ============================================================================
 * SECTION — Context lifecycle (worker fds) and shared per-shape scratch
 * ==========================================================================*/

/* N over nthreads, rounded up to a multiple of 32 (int8 weight k-group / N-align;
 * rocket_matmul_plan_int8 requires N%32). The last slice = N - n0 is a multiple
 * of 32 when N%32==0. */
static int rki_nstep(int N, int nthreads)
{
    int s = ((N + nthreads - 1) / nthreads + 31) / 32 * 32;
    return s < 32 ? 32 : s;
}

rocket_i8_ctx *rocket_i8_ctx_create(int nthreads)
{
    if (nthreads < 1) nthreads = 1;
    if (nthreads > RKI_MAX_WORKERS) nthreads = RKI_MAX_WORKERS;

    rocket_i8_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->nthreads = nthreads;
    ctx->hw = rocket_hw_current();
    for (int t = 0; t < nthreads; t++) ctx->fd[t] = -1;

    for (int t = 0; t < nthreads; t++) {
        ctx->fd[t] = rocket_open();
        if (ctx->fd[t] < 0) {
            ROCKET_LOGE("rocket_i8_ctx_create: rocket_open worker %d failed (%d)\n", t, ctx->fd[t]);
            for (int u = 0; u < t; u++) rocket_close(ctx->fd[u]);
            free(ctx);
            return NULL;
        }
    }
    return ctx;
}

static void rki_worker_free(int fd, rki_worker *w)
{
    rocket_bo_free(fd, &w->guard);
    rocket_bo_free(fd, &w->regcmd);
    rocket_bo_free(fd, &w->in_all);
    rocket_bo_free(fd, &w->out_all);
    free(w->tasks);  free(w->acc);   free(w->facc);
    free(w->bm0);    free(w->bn0);
    free(w->bMtile); free(w->bNtile); free(w->bg); free(w->boff);
}

static void rki_scratch_free(rocket_i8_ctx *ctx, rki_scratch *sc)
{
    if (!sc) return;
    for (int t = 0; t < sc->nt; t++) rki_worker_free(ctx->fd[t], &sc->w[t]);
    free(sc);
}

void rocket_i8_ctx_free(rocket_i8_ctx *ctx)
{
    if (!ctx) return;
    for (int i = 0; i < ctx->nscache; i++) rki_scratch_free(ctx, ctx->scache[i]);
    for (int t = 0; t < ctx->nthreads; t++)
        if (ctx->fd[t] >= 0) rocket_close(ctx->fd[t]);
    free(ctx);
}

/* True if BO [addr, addr+size) escapes the low-4GB IOVA window the regcmd's
 * 32-bit address fields can encode. */
static int i8_iova_overflow(const rocket_bo *bo, size_t size)
{
    return ((bo->dma_address + size) >> 32) != 0;
}

/* Allocate one worker's N-slice plan + shared scratch BOs (NOT the weight BO).
 * group == 0 per-channel, > 0 group-wise (see the file header). */
static int rki_worker_alloc(int fd, rki_worker *w, int M, int tileM, int K, int nsub,
                            int n0, int group)
{
    w->n0 = n0; w->nsub = nsub;
    /* Plan the tiling at the CANONICAL tileM (= MAX_TILE), not the actual row count M,
     * so Mt/Kt/Nt — and hence the resident weight scatter layout (wt_slot/nNt/nKt) — are
     * M-INDEPENDENT. A weight packed at one M is then valid to compute against at ANY
     * other M (the warmup-M-serves-prefill-M reuse: no short-prompt re-pack). int8's host
     * int32 K-accum is exact regardless of K-tiling, so cross-M reuse stays bit-exact.
     * The actual M sets only nMt and the host/BO sizing.
     *
     * Group-wise plans through rocket_matmul_plan_int8_gw, which re-fits the CBUF around
     * the "a K-tile lies inside one quant group" constraint. Do NOT instead overwrite Kt
     * with `group` the way the int4 twin does: int4 gets away with an un-rechecked Kt only
     * because its int16-saturation bound already caps group at 668 AND its nibble packing
     * halves the bytes. int8 has neither, so a wide group would silently overflow the CBUF. */
    int rc = group > 0 ? rocket_matmul_plan_int8_gw(tileM, K, nsub, group, &w->Mt, &w->Kt, &w->Nt)
                       : rocket_matmul_plan_int8(tileM, K, nsub, &w->Mt, &w->Kt, &w->Nt);
    if (rc < 0) {
        ROCKET_LOGE("rki_worker_alloc: unsupported slice tileM=%d K=%d N=%d group=%d\n",
                tileM, K, nsub, group);
        return -1;
    }
    w->kt_per_group = group > 0 ? group / w->Kt : 0;   /* plan_int8_gw guarantees Kt | group */
    w->nMt = (M + w->Mt - 1) / w->Mt;
    w->nNt = (nsub + w->Nt - 1) / w->Nt;
    w->nKt = (K + w->Kt - 1) / w->Kt;
    w->in_slot  = (size_t)i8_rup(w->Mt, 4)  * i8_rup(w->Kt, 32);   /* int8  */
    w->wt_slot  = (size_t)i8_rup(w->Nt, 32) * i8_rup(w->Kt, 32);   /* int8  */
    w->out_slot = (size_t)i8_rup(w->Mt, 4)  * i8_rup(w->Nt, 16);   /* int32 */

    size_t in_sz  = (size_t)w->nMt * w->nKt * w->in_slot + I8_CBUF_BANK;
    size_t rc_sz  = (size_t)I8_BATCH * I8_RC_STRIDE * sizeof(uint64_t);
    size_t out_sz = (size_t)I8_BATCH * w->out_slot * sizeof(int32_t) + I8_CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096,  &w->guard);    /* push allocs off IOVA 0 */
    ret |= rocket_bo_alloc(fd, rc_sz,  &w->regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &w->in_all);
    ret |= rocket_bo_alloc(fd, out_sz, &w->out_all);
    if (ret) { ROCKET_LOGE("rki_worker_alloc: scratch BO alloc failed\n"); goto fail; }
    if (i8_iova_overflow(&w->in_all, in_sz) || i8_iova_overflow(&w->out_all, out_sz) ||
        i8_iova_overflow(&w->regcmd, rc_sz)) {
        ROCKET_LOGE("rki_worker_alloc: scratch BO dma_address exceeds 32 bits\n");
        goto fail;
    }

    w->tasks  = malloc(I8_BATCH * sizeof(*w->tasks));
    if (group > 0) { w->facc = malloc((size_t)M * nsub * sizeof(float));
                     w->bg   = malloc(I8_BATCH * sizeof(int)); }
    else             w->acc  = malloc((size_t)M * nsub * sizeof(int64_t));
    w->bm0    = malloc(I8_BATCH * sizeof(int));
    w->bn0    = malloc(I8_BATCH * sizeof(int));
    w->bMtile = malloc(I8_BATCH * sizeof(int));
    w->bNtile = malloc(I8_BATCH * sizeof(int));
    w->boff   = malloc(I8_BATCH * sizeof(size_t));
    if (!w->tasks || (group > 0 ? (!w->facc || !w->bg) : !w->acc) ||
        !w->bm0 || !w->bn0 || !w->bMtile || !w->bNtile || !w->boff) {
        ROCKET_LOGE("rki_worker_alloc: host scratch alloc failed\n");
        goto fail;
    }
    return 0;

fail:
    rki_worker_free(fd, w);
    memset(w, 0, sizeof(*w));
    return -1;
}

static rki_scratch *rki_scratch_alloc(rocket_i8_ctx *ctx, int M, int K, int N, int group)
{
    rki_scratch *sc = calloc(1, sizeof(*sc));
    if (!sc) return NULL;
    sc->M = M; sc->K = K; sc->N = N; sc->group = group;

    int Nstep = rki_nstep(N, ctx->nthreads);
    int t = 0;
    for (; t < ctx->nthreads; t++) {
        int n0 = t * Nstep;
        if (n0 >= N) break;                          /* fewer slices than fds */
        int nsub = (n0 + Nstep > N) ? (N - n0) : Nstep;
        if (rki_worker_alloc(ctx->fd[t], &sc->w[t], M, ctx->hw->max_tile, K, nsub, n0, group) < 0)
            goto fail;
    }
    sc->nt = t;
    return sc;

fail:
    for (int u = 0; u < t; u++) rki_worker_free(ctx->fd[u], &sc->w[u]);
    free(sc);
    return NULL;
}

static rki_scratch *rki_ctx_scratch(rocket_i8_ctx *ctx, int M, int K, int N, int group)
{
    for (int i = 0; i < ctx->nscache; i++)
        if (ctx->scache[i]->M == M && ctx->scache[i]->K == K &&
            ctx->scache[i]->N == N && ctx->scache[i]->group == group)
            return ctx->scache[i];
    if (ctx->nscache >= RKI_MAX_SLOTS) return NULL;
    rki_scratch *sc = rki_scratch_alloc(ctx, M, K, N, group);
    if (!sc) return NULL;
    ctx->scache[ctx->nscache++] = sc;
    return sc;
}

/* True iff a weight scattered for the `packed` scratch's per-worker tile layout is valid
 * to compute against the `sc` scratch (a different call-M). The weight bytes depend only
 * on the N-split + Nt/Kt/nNt/nKt/wt_slot; canonical tiling makes these M-independent, so
 * this holds for every M — but check defensively (e.g. a ROCKET_MM_* env override between
 * pack and compute) so a genuine mismatch returns -2 instead of miscomputing. */
static int rki_weight_fits(const rki_scratch *packed, const rki_scratch *sc)
{
    if (!packed || !sc) return 0;
    if (packed->K != sc->K || packed->N != sc->N ||
        packed->group != sc->group || packed->nt != sc->nt) return 0;
    for (int t = 0; t < sc->nt; t++) {
        const rki_worker *a = &packed->w[t], *b = &sc->w[t];
        if (a->n0  != b->n0  || a->nsub != b->nsub || a->Nt != b->Nt || a->Kt != b->Kt ||
            a->nNt != b->nNt || a->nKt  != b->nKt  || a->wt_slot != b->wt_slot) return 0;
    }
    return 1;
}

/* ============================================================================
 * SECTION — Resident int8 weight packing (scatter B into per-worker BOs once)
 * ==========================================================================*/

/* Scatter a pre-quantized int8 weight B[N,K] into the resident per-worker NPU BOs,
 * (Nslice/32, K/32, 32, 32) tile layout per K-tile. Shared by both pack variants; the
 * worker's Kt/nKt tiling (CBUF-max, or group-constrained) is already fixed by the plan,
 * and the scatter is identical either way — only the K-tile boundaries move. */
static int rki_scatter_weights(const char *who, rocket_i8_ctx *ctx, rki_scratch *sc,
                               rocket_i8_weights *w, const int8_t *B, int K)
{
    int t = 0;
    for (; t < sc->nt; t++) {
        rki_worker *ww = &sc->w[t];
        size_t wt_sz = (size_t)ww->nNt * ww->nKt * ww->wt_slot + I8_CBUF_BANK;
        if (rocket_bo_alloc(ctx->fd[t], wt_sz, &w->wt[t]) != 0) {
            ROCKET_LOGE("%s: wt BO alloc failed (worker %d, %zuMB)\n", who, t, wt_sz >> 20);
            goto fail;
        }
        if (i8_iova_overflow(&w->wt[t], wt_sz)) {
            ROCKET_LOGE("%s: wt BO dma_address exceeds 32 bits (worker %d)\n", who, t);
            rocket_bo_free(ctx->fd[t], &w->wt[t]);
            goto fail;
        }

        /* B's column-slice [n0, n0+nsub): local n into the slice maps to global row
         * n0 + (n-1). */
        if (rocket_bo_prep(ctx->fd[t], &w->wt[t], 1, 0) != 0) {  /* sync failed (logged) */
            rocket_bo_free(ctx->fd[t], &w->wt[t]); goto fail;    /* fail frees [0,t); free t here */
        }
        memset(w->wt[t].ptr, 0, w->wt[t].size);
        const int8_t *restrict Bslice = B + (size_t)ww->n0 * K;
        for (int ni = 0; ni < ww->nNt; ni++) {
            int n0 = ni * ww->Nt, Ntile = (ww->nsub - n0 < ww->Nt) ? (ww->nsub - n0) : ww->Nt;
            for (int ki = 0; ki < ww->nKt; ki++) {
                int k0 = ki * ww->Kt, Ktile = (K - k0 < ww->Kt) ? (K - k0) : ww->Kt;
                int8_t *restrict slot = (int8_t *)w->wt[t].ptr + (size_t)(ni * ww->nKt + ki) * ww->wt_slot;
                for (int kk = 1; kk <= Ntile; kk++)
                    for (int c = 1; c <= Ktile; c++)
                        slot[i8_wt_idx(Ktile, kk, c)] = Bslice[(size_t)(n0 + kk - 1) * K + (k0 + c - 1)];
            }
        }
        rocket_bo_fini(ctx->fd[t], &w->wt[t]);
    }
    return 0;

fail:
    for (int u = 0; u < t; u++) rocket_bo_free(ctx->fd[u], &w->wt[u]);
    return -1;
}

/* Shape contract shared by both pack variants — checked here rather than left to fail
 * deep inside rki_worker_alloc with an opaque "unsupported slice". N%32 (the int8 weight
 * N-group) keeps every N-slice aligned; K%32 the K-group; M%4 (M==1 height-1 GEMV is
 * broken on HW and the resident path can't pad — pad M to 4 caller-side). */
static int rki_shape_ok(const char *who, int M, int K, int N)
{
    if (N % 32 != 0 || K % 32 != 0 || M % 4 != 0) {
        ROCKET_LOGE("%s: unsupported shape M=%d K=%d N=%d (need N%%32, K%%32, M%%4)\n",
                who, M, K, N);
        return 0;
    }
    return 1;
}

rocket_i8_weights *rocket_i8_weights_pack(rocket_i8_ctx *ctx, int M, int K, int N,
                                          const int8_t *B)
{
    if (!ctx) return NULL;
    if (!rki_shape_ok("rocket_i8_weights_pack", M, K, N)) return NULL;

    rki_scratch *sc = rki_ctx_scratch(ctx, M, K, N, 0);
    if (!sc) return NULL;

    rocket_i8_weights *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->M = M; w->K = K; w->N = N; w->nt = sc->nt; w->group = 0; w->sc = sc;
    if (rki_scatter_weights("rocket_i8_weights_pack", ctx, sc, w, B, K) < 0) { free(w); return NULL; }
    return w;
}

/* Group-wise resident pack: like rocket_i8_weights_pack, but the K-tiling is
 * constrained so no K-tile straddles a quant group (see rocket_matmul_plan_int8_gw).
 * The scattered BYTES are the same int8 codes either way — only the tile boundaries
 * differ — so this is a re-tiling, not a requantization. Partner:
 * rocket_matmul_int8_prepacked_gw. Unlike the int4 twin there is no saturation bound
 * on `group` (int8's NPU output is int32, not int16), and a group wider than the CBUF
 * cap is legal: the planner then uses the largest divisor of it that fits. */
rocket_i8_weights *rocket_i8_weights_pack_gw(rocket_i8_ctx *ctx, int M, int K, int N,
                                             const int8_t *B, int group)
{
    if (!ctx) return NULL;
    if (!rki_shape_ok("rocket_i8_weights_pack_gw", M, K, N)) return NULL;
    if (group < 32 || group % 32 || K % group) {
        ROCKET_LOGE("rocket_i8_weights_pack_gw: bad group=%d (need >=32, %%32, |K) K=%d\n",
                group, K);
        return NULL;
    }
    rki_scratch *sc = rki_ctx_scratch(ctx, M, K, N, group);
    if (!sc) return NULL;

    rocket_i8_weights *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->M = M; w->K = K; w->N = N; w->nt = sc->nt; w->group = group; w->sc = sc;
    if (rki_scatter_weights("rocket_i8_weights_pack_gw", ctx, sc, w, B, K) < 0) { free(w); return NULL; }
    return w;
}

size_t rocket_i8_weights_bytes(const rocket_i8_weights *w)
{
    if (!w) return 0;
    /* Resident NPU-BO footprint = the per-worker weight tiles (the shared scratch
     * is cached across weights and not counted here). */
    size_t bytes = 0;
    for (int t = 0; t < w->nt; t++) bytes += w->wt[t].size;
    return bytes;
}

void rocket_i8_weights_free(rocket_i8_ctx *ctx, rocket_i8_weights *w)
{
    if (!ctx || !w) return;
    for (int t = 0; t < w->nt; t++) rocket_bo_free(ctx->fd[t], &w->wt[t]);
    free(w);                                /* sc is shared, owned by the ctx */
}

/* ============================================================================
 * SECTION — Per-worker compute thread (pack A, batched tile submit, host K-accum)
 * ==========================================================================*/

typedef struct {
    int fd;
    rki_worker *ww;
    const rocket_bo *wt;     /* resident int8 weight for this worker's N-slice */
    const int8_t *A;         /* full int8 A[M,K] */
    int32_t *C;              /* full int32 C[M,N]  (per-channel; this worker's columns) */
    float   *Cf;             /* full fp32  Cf[M,N] (group-wise;  this worker's columns) */
    const float *a_scale;    /* [M*nG]  (group-wise only) */
    const float *b_scale;    /* [N*nG]  (group-wise only, GLOBAL column index) */
    int M, K, N;
    int group;               /* 0 = per-channel, > 0 = group-wise */
    int idx;
    int ret;
    int core_base;           /* big-core rotation base inherited from the caller thread */
} rki_arg;

static void *rki_thread(void *a)
{
    rki_arg *t = (rki_arg *)a;
    rocket_pin_worker_based(t->idx, t->core_base);
    rki_worker *w = t->ww;
    int fd = t->fd;
    int M = t->M, K = t->K, N = t->N;
    int nsub = w->nsub;
    int Mt = w->Mt, Kt = w->Kt, Nt = w->Nt;
    int nMt = w->nMt, nNt = w->nNt, nKt = w->nKt;
    const int gw = t->group > 0;              /* group-wise: fp32 scaled K-accum */
    const int nG = gw ? K / t->group : 0;     /* quant groups along K */
    t->ret = 0;

    /* ---- pack input A[M,K] -> (M,K) int8 feature cube (C2=16) into in_all.
     * A is N-independent, so every worker scatters the full A into its own
     * scratch (cheap int8 scatter; the shared-A optimization is a later lever). */
    if (rocket_bo_prep(fd, &w->in_all, 1, 0) != 0) { t->ret = -1; return NULL; }  /* sync failed (logged) */
    memset(w->in_all.ptr, 0, w->in_all.size);
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            int8_t *restrict slot = (int8_t *)w->in_all.ptr + (size_t)(mi * nKt + ki) * w->in_slot;
            for (int h = 1; h <= Mtile; h++)
                for (int c = 1; c <= Ktile; c++)
                    slot[i8_feat_idx(Mtile, c, h)] = t->A[(size_t)(m0 + h - 1) * K + (k0 + c - 1)];
        }
    }
    /* Flush the packed input to the device before it is read by the NPU. An
     * un-propagated sync failure would let a tile run on stale/un-synced data. */
    if ((t->ret = rocket_bo_fini(fd, &w->in_all)) != 0) return NULL;

    /* ---- batched tile compute: host K-accumulation (int8 NPU K-accum is HW-dead).
     * Accumulate over this worker's N-slice into acc[M, nsub] (per-channel: int64,
     * integer-EXACT) or facc[M, nsub] (group-wise: fp32, each tile's int32 partial
     * pre-scaled by its quant group's a_scale*b_scale). */
    int64_t *restrict acc  = w->acc;
    float   *restrict facc = w->facc;
    if (gw) memset(facc, 0, (size_t)M * nsub * sizeof(float));
    else    memset(acc,  0, (size_t)M * nsub * sizeof(int64_t));
    uint64_t npu_regs[256] = {0};
    rocket_task_desc *tasks = w->tasks;
    /* Submit layout. The resident int8 path already batches a job's tiles into ONE
     * ioctl (lever 1, the submit-overhead win); each gapped task is a SEPARATE HW
     * kick, so the integer int32 accumulator (CACC) clears between tasks.
     *
     * Contiguous CHAINING (lever 2: one kick / one IRQ for the whole batch) is
     * HW-BLOCKED for the integer datapath and forced off here. The chained layout
     * is byte-identical to fp16's (chain_layout passes for int8/int4), but running
     * the tasks back-to-back in one kick computes the FIRST task correctly and
     * garbles every subsequent one — the CACC clears per-kick, not per-task, so
     * task N+1 accumulates onto task N's residual int32 (HW sweep 2026-06-28:
     * M/N/K-tiled all show first-tile-exact, rest-garbage; fp16 chains fine because
     * it does not carry the integer accumulator). Same root as the
     * no-cross-op-int32-accumulate ceiling. Re-enable only if a per-task CACC clear
     * is found. */
    int chained = 0;   /* integer chaining HW-blocked; gapped batch = lever 1 only */
    (void)rkt_chain_enabled;

    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (nsub - n0 < Nt) ? (nsub - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;

                if (nb == 0 && (t->ret = rocket_bo_prep(fd, &w->regcmd, 1, 0)) != 0) return NULL;

                size_t out_off = (size_t)nb * w->out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma   = (uint32_t)(w->in_all.dma_address + (size_t)(mi*nKt+ki) * w->in_slot),
                    .weights_dma = (uint32_t)(t->wt->dma_address     + (size_t)(ni*nKt+ki) * w->wt_slot),
                    .output_dma  = (uint32_t)(w->out_all.dma_address + out_off * sizeof(int32_t)),
                    .tasks = npu_regs,
                };
                if ((t->ret = gen_matmul_int8(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul_int8_prepacked: gen failed (%d)\n", t->ret);
                    return NULL;
                }
                if (p.task_count > I8_RC_STRIDE) {  /* unconditional: -DNDEBUG strips asserts */
                    ROCKET_LOGE("rocket_matmul_int8_prepacked: regcmd slot overflow "
                            "(task_count %u > %d words)\n", p.task_count, I8_RC_STRIDE);
                    t->ret = -1; return NULL;
                }
                rkt_chain_pack(chained, &w->regcmd, tasks, nb, npu_regs,
                               p.task_count, I8_RC_STRIDE);
                w->bm0[nb] = m0; w->bn0[nb] = n0; w->bMtile[nb] = Mtile; w->bNtile[nb] = Ntile;
                w->boff[nb] = out_off;
                /* Kt divides the group, so this K-tile lies wholly inside ONE group —
                 * its partial takes that group's single scale. */
                if (gw) w->bg[nb] = ki / w->kt_per_group;
                nb++; done_tiles++;

                if (nb == I8_BATCH || done_tiles == total) {
                    /* Seal the chain (last task's link -> 0) then flush the regcmd
                     * batch + ready the output BO before submit. Propagate
                     * staging-sync failures (un-synced regcmd / a not-yet-
                     * invalidated output BO would silently corrupt the result). */
                    rkt_chain_seal(chained, &w->regcmd, nb, tasks[0].regcmd_count);
                    if ((t->ret = rocket_bo_fini(fd, &w->regcmd))   != 0) return NULL;
                    if ((t->ret = rocket_bo_prep(fd, &w->out_all, 1, 0)) != 0) return NULL;
                    if ((t->ret = rocket_bo_fini(fd, &w->out_all))  != 0) return NULL;

                    uint32_t in_h[]  = { w->in_all.handle, t->wt->handle, w->regcmd.handle };
                    uint32_t out_h[] = { w->out_all.handle };
                    if ((t->ret = rocket_submit_tasks(fd, tasks, nb, in_h, 3, out_h, 1)) != 0)
                        return NULL;

                    if ((t->ret = rocket_bo_prep(fd, &w->out_all, 0, i8_wait_ns())) != 0) {
                        ROCKET_LOGE("rocket_matmul_int8_prepacked: WAIT TIMEOUT (%d) "
                                "M=%d K=%d N=%d slice=%d batch=%d tiles=%d/%d\n",
                                t->ret, M, K, nsub, w->n0, nb, done_tiles, total);
                        return NULL;
                    }

                    int32_t *restrict ob = (int32_t *)w->out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        int32_t *restrict slot = ob + w->boff[j];
                        if (gw) {
                            int g = w->bg[j];                    /* this tile's K-group */
                            for (int h = 1; h <= w->bMtile[j]; h++) {
                                int mrow = w->bm0[j] + h - 1;
                                float as = t->a_scale[(size_t)mrow * nG + g];
                                for (int nn = 1; nn <= w->bNtile[j]; nn++) {
                                    int ncol_loc = w->bn0[j] + nn - 1;   /* within this N-slice */
                                    int ncol     = w->n0 + ncol_loc;     /* global (b_scale)    */
                                    facc[(size_t)mrow * nsub + ncol_loc] +=
                                        as * t->b_scale[(size_t)ncol * nG + g] *
                                        (float)slot[i8_out_idx(w->bMtile[j], nn, h)];
                                }
                            }
                        } else {
                            for (int h = 1; h <= w->bMtile[j]; h++)
                                for (int nn = 1; nn <= w->bNtile[j]; nn++)
                                    acc[(size_t)(w->bm0[j] + h - 1) * nsub + (w->bn0[j] + nn - 1)] +=
                                        (int64_t)slot[i8_out_idx(w->bMtile[j], nn, h)];
                        }
                    }
                    if ((t->ret = rocket_bo_fini(fd, &w->out_all)) != 0) return NULL;
                    nb = 0;
                }
            }
        }
    }

    /* scatter this worker's column slice into the full output. */
    for (int m = 0; m < M; m++)
        for (int n = 0; n < nsub; n++) {
            if (gw) t->Cf[(size_t)m * N + (w->n0 + n)] = facc[(size_t)m * nsub + n];
            else    t->C [(size_t)m * N + (w->n0 + n)] = (int32_t)acc[(size_t)m * nsub + n];
        }

    return NULL;
}

/* ============================================================================
 * SECTION — Public API: prepacked int8 (W8A8) matmul, per-channel and group-wise
 * ==========================================================================*/

/* Resolve the scratch a call will run against, and check the resident weight is valid
 * for it. Uses the scratch for the CALL's M (not the pack-time M): the resident weight
 * is M-independent (canonical tiling), so it is reused across M with no re-pack. Returns
 * 0, -1 (bad shape / OOM), or -2 = genuine tiling mismatch, meaning "re-pack" rather
 * than a wrong answer. */
static int rki_resolve(rocket_i8_ctx *ctx, const char *who, int M, int K, int N,
                       const rocket_i8_weights *w, rki_scratch **psc)
{
    /* The CALL's M must satisfy M%4, and nothing upstream checks it. The pack-time
     * shape check only saw the PACK's M, and rki_worker_alloc plans at the canonical
     * tileM (= MAX_TILE), so rocket_matmul_plan_int8's own M%4 guard never sees this M
     * either. M-independence is exactly what makes that gap reachable: the whole point
     * is that callers vary M freely against one packed weight. An unaligned M is not
     * merely suboptimal — it silently MISCOMPUTES (HW sweep: M = 1/2/3/5/6 all return
     * garbage with rc=0; only M%4 is correct), so it must be rejected, not padded here:
     * padding M would need a matching pad of a_scale, which only the caller can supply.
     * Callers with a ragged row count pad to 4 (see rocket_pad_m). */
    if (M % 4 != 0 || M <= 0) {
        ROCKET_LOGE("%s: M=%d must be a positive multiple of 4 (an unaligned M "
                "miscomputes on HW) — pad rows caller-side\n", who, M);
        return -1;
    }
    if (K != w->K || N != w->N) {
        ROCKET_LOGE("%s: shape K=%d N=%d != packed %d/%d\n", who, K, N, w->K, w->N);
        return -1;
    }
    rki_scratch *sc = rki_ctx_scratch(ctx, M, K, N, w->group);
    if (!sc) return -1;
    if (!rki_weight_fits(w->sc, sc)) {
        ROCKET_LOGE("%s: weight tiling (packed M=%d) incompatible with M=%d — "
                "re-pack needed\n", who, w->M, M);
        return -2;
    }
    *psc = sc;
    return 0;
}

/* One thread per worker (N-slice); `proto` carries the call-invariant fields and the
 * per-worker ones are filled in here. Shared by both entry points. */
static int rki_run(rocket_i8_ctx *ctx, rki_scratch *sc, rocket_i8_weights *w,
                   const rki_arg *proto)
{
    pthread_t th[RKI_MAX_WORKERS];
    rki_arg   args[RKI_MAX_WORKERS];
    int joinable[RKI_MAX_WORKERS] = {0};
    int base = rocket_affinity_get_base();      /* spread in-process pools across the cluster */

    for (int t = 0; t < sc->nt; t++) {
        args[t] = *proto;
        args[t].fd  = ctx->fd[t];
        args[t].ww  = &sc->w[t];
        args[t].wt  = &w->wt[t];
        args[t].idx = t;
        args[t].ret = 0;
        args[t].core_base = base;
        if (pthread_create(&th[t], NULL, rki_thread, &args[t]) == 0)
            joinable[t] = 1;
        /* else: run inline AFTER the spawn loop (see rocket_matmul_fp16_mt) so a create
         * failure doesn't serialize the remaining workers behind its ~8s NPU wait. */
    }

    for (int t = 0; t < sc->nt; t++)
        if (!joinable[t]) rki_thread(&args[t]);

    int ret = 0;
    for (int t = 0; t < sc->nt; t++) {
        if (joinable[t]) pthread_join(th[t], NULL);
        if (args[t].ret) ret = args[t].ret;
    }
    return ret;
}

int rocket_matmul_int8_prepacked(rocket_i8_ctx *ctx, int M, int K, int N,
                                 const int8_t *A, int32_t *C, rocket_i8_weights *w)
{
    if (!ctx || !w || !w->sc || !A || !C) return -1;
    if (w->group != 0) {
        ROCKET_LOGE("rocket_matmul_int8_prepacked: weight is group-wise (group=%d) — "
                "use rocket_matmul_int8_prepacked_gw\n", w->group);
        return -1;
    }
    rki_scratch *sc;
    int rc = rki_resolve(ctx, "rocket_matmul_int8_prepacked", M, K, N, w, &sc);
    if (rc) return rc;

    rki_arg proto = { .A = A, .C = C, .M = M, .K = K, .N = N, .group = 0 };
    return rki_run(ctx, sc, w, &proto);
}

int rocket_matmul_int8_prepacked_gw(rocket_i8_ctx *ctx, int M, int K, int N,
                                    const int8_t *A, const float *a_scale,
                                    const float *b_scale, float *Cf, rocket_i8_weights *w)
{
    if (!ctx || !w || !w->sc || !A || !a_scale || !b_scale || !Cf) return -1;
    if (w->group <= 0) {
        ROCKET_LOGE("rocket_matmul_int8_prepacked_gw: weight is per-channel — "
                "use rocket_matmul_int8_prepacked\n");
        return -1;
    }
    rki_scratch *sc;
    int rc = rki_resolve(ctx, "rocket_matmul_int8_prepacked_gw", M, K, N, w, &sc);
    if (rc) return rc;

    rki_arg proto = { .A = A, .Cf = Cf, .a_scale = a_scale, .b_scale = b_scale,
                      .M = M, .K = K, .N = N, .group = w->group };
    return rki_run(ctx, sc, w, &proto);
}
