// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_prepacked_int4.c — resident-weight int4 (W4A4) matmul path.
 *
 * The int4 sibling of rocket_prepacked_int8.c: a pre-quantized int4 weight B[N,K]
 * (values one-per-int8_t in [-7,7]) is scattered into resident per-worker NPU BOs
 * ONCE, every matmul reuses them and only packs A. N fanned across worker fds.
 * Built to MEASURE int4's raw multicore throughput against the resident fp16/int8
 * bars at matched shapes (no model) — in particular whether int4's single-pass-K
 * (nKt=1, achievable by shrinking the tiles, ROCKET_MM_MT/NT) + 4x MAC beats the
 * readback-bound int8 path.
 *
 * int4 deltas vs rocket_prepacked_int8.c: in/wt NIBBLE-packed (2 int4/byte, BO
 * bytes = nibbles/2); feature cube C2=32; weight layout (N/64,K/32,64,32)
 * [weight_int4]; output int16 (cube C2=8), host-summed in int64 -> int32 C; Nt%64
 * (int4 N-group is 64). Host int64 K-accum (DPU-EW int16 K-accum is a later lever).
 * Self-contained (fp16/int8 paths untouched). Index helpers duplicated from
 * rocket_matmul.c, pinned by the bit-exact test (matmul_int4_tiled_rocket).
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

#define RK4_MAX_WORKERS 8
#define RK4_MAX_SLOTS   32
#define I4_BATCH        64
#define I4_RC_STRIDE    128
#define I4_CBUF_BANK    32768

/* ============================================================================
 * SECTION — int4 tile-layout index math and small helpers
 * ==========================================================================*/

/* Round x up to a multiple of a, in size_t (these feed slot/BO allocation sizes;
 * computing in int would truncate before the widen at the call site). */
static size_t i4_rup(int x, int a) { return (((size_t)x + a - 1) / a) * a; }

static long i4_wait_ns(void) {
    static _Atomic long ns = -1;
    if (ns < 0) {
        const char *e = getenv("ROCKET_WAIT_MS");
        long ms = e ? atol(e) : 8000;
        if (ms < 1) ms = 8000;
        ns = ms * 1000000L;
    }
    return ns;
}

/* int4 NPU layout index math — identical to rocket_matmul.c's static inlines. */
static inline size_t i4_feat_idx(int H, int ch, int h) {   /* input nibble, C2=32 */
    return ((size_t)(ch - 1) / 32) * (size_t)H * 32 + 32 * (size_t)(h - 1) + (ch - 1) % 32;
}
static inline size_t i4_wt_idx(int C, int k, int c) {      /* weight nibble, (N/64,K/32,64,32) */
    size_t nKgrp   = (size_t)((C + 31) / 32);
    size_t Ngrp    = (size_t)(k - 1) / 64, Nwithin = (size_t)(k - 1) % 64;
    size_t Kgrp    = (size_t)(c - 1) / 32, Kwithin = (size_t)(c - 1) % 32;
    return Ngrp * nKgrp * 64 * 32 + Kgrp * 64 * 32 + Nwithin * 32 + Kwithin;
}
static inline size_t i4_out_idx(int H, int ch, int h) {    /* output int16 elem, C2=8 */
    return ((size_t)(ch - 1) / 8) * (size_t)H * 8 + 8 * (size_t)(h - 1) + (ch - 1) % 8;
}
static inline void i4_put_nib(uint8_t *buf, size_t idx, int8_t v) {
    uint8_t nib = (uint8_t)(v & 0xF);
    if (idx & 1) buf[idx >> 1] = (uint8_t)((buf[idx >> 1] & 0x0F) | (nib << 4));
    else         buf[idx >> 1] = (uint8_t)((buf[idx >> 1] & 0xF0) | nib);
}

typedef struct {
    int n0, nsub;
    int Mt, Kt, Nt, nMt, nNt, nKt;
    size_t in_slot, wt_slot, out_slot;       /* in/wt nibbles, out int16 elems */
    rocket_bo guard, regcmd, in_all, out_all;
    rocket_task_desc *tasks;
    void   *submit_dt;                       /* resident drm_rocket_task[] submit scratch */
    int64_t *acc;                            /* M*nsub (per-channel/raw-int32 path) */
    float   *facc;                           /* M*nsub (group-wise fp32 path) */
    int *bm0, *bn0, *bMtile, *bNtile, *bki;  /* bki = a tile's K-group index (group-wise) */
    size_t *boff;
} rk4_worker;

/* group: 0 = per-channel (raw int32 out); >0 = group-wise (Kt forced to group,
 * fp32 per-group-scaled out). A scratch is keyed on (M,K,N,group) — the two modes
 * use different K-tiling so they cannot share scratch. */
typedef struct { int M, K, N, nt, group; rk4_worker w[RK4_MAX_WORKERS]; } rk4_scratch;

struct rocket_i4_weights {
    int M, K, N, nt, group;
    rocket_bo   wt[RK4_MAX_WORKERS];
    rk4_scratch *sc;
};

struct rocket_i4_ctx {
    int nthreads;
    const struct rocket_hw_profile *hw; /* active machine-parameter profile (the multi-chip profile seam) */
    int fd[RK4_MAX_WORKERS];
    rk4_scratch *scache[RK4_MAX_SLOTS];
    int nscache;
};

/* ============================================================================
 * SECTION — Context lifecycle (worker fds) and shared per-shape scratch
 * ==========================================================================*/

/* N over nthreads, rounded up to a multiple of 64 (int4 N-group). */
static int rk4_nstep(int N, int nthreads) {
    int s = ((N + nthreads - 1) / nthreads + 63) / 64 * 64;
    return s < 64 ? 64 : s;
}

rocket_i4_ctx *rocket_i4_ctx_create(int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > RK4_MAX_WORKERS) nthreads = RK4_MAX_WORKERS;
    rocket_i4_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->nthreads = nthreads;
    ctx->hw = rocket_hw_current();
    for (int t = 0; t < nthreads; t++) ctx->fd[t] = -1;
    for (int t = 0; t < nthreads; t++) {
        ctx->fd[t] = rocket_open();
        if (ctx->fd[t] < 0) {
            ROCKET_LOGE("rocket_i4_ctx_create: rocket_open worker %d failed (%d)\n", t, ctx->fd[t]);
            for (int u = 0; u < t; u++) rocket_close(ctx->fd[u]);
            free(ctx); return NULL;
        }
    }
    return ctx;
}

static void rk4_worker_free(int fd, rk4_worker *w) {
    rocket_bo_free(fd, &w->guard);  rocket_bo_free(fd, &w->regcmd);
    rocket_bo_free(fd, &w->in_all); rocket_bo_free(fd, &w->out_all);
    free(w->tasks); free(w->submit_dt); free(w->acc); free(w->facc);
    free(w->bm0); free(w->bn0); free(w->bMtile); free(w->bNtile); free(w->bki); free(w->boff);
}

static void rk4_scratch_free(rocket_i4_ctx *ctx, rk4_scratch *sc) {
    if (!sc) return;
    for (int t = 0; t < sc->nt; t++) rk4_worker_free(ctx->fd[t], &sc->w[t]);
    free(sc);
}

void rocket_i4_ctx_free(rocket_i4_ctx *ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->nscache; i++) rk4_scratch_free(ctx, ctx->scache[i]);
    for (int t = 0; t < ctx->nthreads; t++)
        if (ctx->fd[t] >= 0) rocket_close(ctx->fd[t]);
    free(ctx);
}

static int i4_iova_of(const rocket_bo *bo, size_t size) { return ((bo->dma_address + size) >> 32) != 0; }

static int rk4_worker_alloc(int fd, rk4_worker *w, int M, int tileM, int K, int nsub, int n0, int group) {
    w->n0 = n0; w->nsub = nsub;
    /* Plan the tiling at the CANONICAL tileM (= MAX_TILE), not the actual row count M,
     * so Mt/Kt/Nt — and hence the resident weight scatter layout (wt_slot/nNt/nKt) — are
     * M-INDEPENDENT. A weight packed at one M is then valid to compute against at ANY
     * other M, killing the short-prompt re-pack stall (warmup M=512 → small-M prefill
     * re-packed the whole model). The actual M sets only nMt and the host/BO sizing. */
    if (rocket_matmul_plan_int4(tileM, K, nsub, &w->Mt, &w->Kt, &w->Nt) < 0) {
        ROCKET_LOGE("rk4_worker_alloc: unsupported slice tileM=%d K=%d N=%d\n", tileM, K, nsub);
        return -1;
    }
    if (group > 0) w->Kt = group;   /* one K-tile == one quant group (also int16-saturation-safe) */
    w->nMt = (M + w->Mt - 1) / w->Mt;
    w->nNt = (nsub + w->Nt - 1) / w->Nt;
    w->nKt = (K + w->Kt - 1) / w->Kt;
    w->in_slot  = (size_t)i4_rup(w->Mt, 4)  * i4_rup(w->Kt, 32);   /* int4 nibbles */
    w->wt_slot  = (size_t)i4_rup(w->Nt, 64) * i4_rup(w->Kt, 32);   /* int4 nibbles */
    w->out_slot = (size_t)i4_rup(w->Mt, 4)  * i4_rup(w->Nt, 16);   /* int16 elems  */

    size_t in_sz  = (size_t)w->nMt * w->nKt * (w->in_slot / 2) + I4_CBUF_BANK;
    size_t rc_sz  = (size_t)I4_BATCH * I4_RC_STRIDE * sizeof(uint64_t);
    /* NB: the output BO is intentionally left at I4_BATCH (NOT right-sized to the
     * live tile count). The fp16-KACC right-size wins because that
     * path syncs its output BO nKt times per matmul (sync-bound). int4 host-
     * accumulates and syncs its output BO only ONCE per job, so the sync term is
     * already negligible — right-sizing it is bit-exact but buys nothing measurable
     * (int4 resident throughput is noisy ~410-580 GOP/s across box state; the change
     * sat inside that band). The lever is specific to multi-sync paths. */
    size_t out_sz = (size_t)I4_BATCH * w->out_slot * sizeof(int16_t) + I4_CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096,  &w->guard);
    ret |= rocket_bo_alloc(fd, rc_sz,  &w->regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &w->in_all);
    ret |= rocket_bo_alloc(fd, out_sz, &w->out_all);
    if (ret) { ROCKET_LOGE("rk4_worker_alloc: scratch BO alloc failed\n"); goto fail; }
    if (i4_iova_of(&w->in_all, in_sz) || i4_iova_of(&w->out_all, out_sz) || i4_iova_of(&w->regcmd, rc_sz)) {
        ROCKET_LOGE("rk4_worker_alloc: scratch BO dma_address exceeds 32 bits\n"); goto fail;
    }
    w->tasks  = malloc(I4_BATCH * sizeof(*w->tasks));
    w->submit_dt = malloc(rocket_submit_scratch_size(I4_BATCH));  /* reused every submit; no per-job calloc */
    if (group > 0) { w->facc = malloc((size_t)M * nsub * sizeof(float));   w->bki = malloc(I4_BATCH * sizeof(int)); }
    else             w->acc  = malloc((size_t)M * nsub * sizeof(int64_t));
    w->bm0    = malloc(I4_BATCH * sizeof(int));   w->bn0    = malloc(I4_BATCH * sizeof(int));
    w->bMtile = malloc(I4_BATCH * sizeof(int));   w->bNtile = malloc(I4_BATCH * sizeof(int));
    w->boff   = malloc(I4_BATCH * sizeof(size_t));
    if (!w->tasks || !w->submit_dt || (group > 0 ? (!w->facc || !w->bki) : !w->acc) ||
        !w->bm0 || !w->bn0 || !w->bMtile || !w->bNtile || !w->boff) {
        ROCKET_LOGE("rk4_worker_alloc: host scratch alloc failed\n"); goto fail;
    }
    return 0;
fail:
    rk4_worker_free(fd, w); memset(w, 0, sizeof(*w)); return -1;
}

static rk4_scratch *rk4_scratch_alloc(rocket_i4_ctx *ctx, int M, int K, int N, int group) {
    rk4_scratch *sc = calloc(1, sizeof(*sc));
    if (!sc) return NULL;
    sc->M = M; sc->K = K; sc->N = N; sc->group = group;
    int Nstep = rk4_nstep(N, ctx->nthreads);
    int t = 0;
    for (; t < ctx->nthreads; t++) {
        int n0 = t * Nstep;
        if (n0 >= N) break;
        int nsub = (n0 + Nstep > N) ? (N - n0) : Nstep;
        if (rk4_worker_alloc(ctx->fd[t], &sc->w[t], M, ctx->hw->max_tile, K, nsub, n0, group) < 0) goto fail;
    }
    sc->nt = t;
    return sc;
fail:
    for (int u = 0; u < t; u++) rk4_worker_free(ctx->fd[u], &sc->w[u]);
    free(sc); return NULL;
}

static rk4_scratch *rk4_ctx_scratch(rocket_i4_ctx *ctx, int M, int K, int N, int group) {
    for (int i = 0; i < ctx->nscache; i++)
        if (ctx->scache[i]->M == M && ctx->scache[i]->K == K &&
            ctx->scache[i]->N == N && ctx->scache[i]->group == group)
            return ctx->scache[i];
    if (ctx->nscache >= RK4_MAX_SLOTS) return NULL;
    rk4_scratch *sc = rk4_scratch_alloc(ctx, M, K, N, group);
    if (!sc) return NULL;
    ctx->scache[ctx->nscache++] = sc;
    return sc;
}

/* The CALL's M must satisfy M%4, and nothing upstream checks it: the pack-time shape
 * check only saw the PACK's M, and rk4_worker_alloc plans at the canonical tileM (=
 * MAX_TILE), so rocket_matmul_plan_int4's own M%4 guard never sees the call's M either.
 * M-independence is what makes the gap reachable — varying M against one packed weight is
 * the feature. An unaligned M silently MISCOMPUTES (the int8 twin's HW sweep: M =
 * 1/2/3/5/6 all return garbage with rc=0), so reject rather than pad: padding M would
 * need a matching pad of a_scale, which only the caller can supply. */
static int rk4_call_m_ok(const char *who, int M) {
    if (M % 4 != 0 || M <= 0) {
        ROCKET_LOGE("%s: M=%d must be a positive multiple of 4 (an unaligned M "
                "miscomputes on HW) — pad rows caller-side\n", who, M);
        return 0;
    }
    return 1;
}

/* True iff a weight scattered for the `packed` scratch's per-worker tile layout is valid
 * to compute against the `sc` scratch (a different call-M). The weight bytes depend only
 * on the N-split + Nt/Kt/nNt/nKt/wt_slot; canonical tiling makes these M-independent, so
 * this holds for every M — but check defensively (e.g. a ROCKET_MM_* env override between
 * pack and compute) so a genuine mismatch returns -2 instead of miscomputing. */
static int rk4_weight_fits(const rk4_scratch *packed, const rk4_scratch *sc) {
    if (!packed || !sc) return 0;
    if (packed->K != sc->K || packed->N != sc->N ||
        packed->group != sc->group || packed->nt != sc->nt) return 0;
    for (int t = 0; t < sc->nt; t++) {
        const rk4_worker *a = &packed->w[t], *b = &sc->w[t];
        if (a->n0  != b->n0  || a->nsub != b->nsub || a->Nt != b->Nt || a->Kt != b->Kt ||
            a->nNt != b->nNt || a->nKt  != b->nKt  || a->wt_slot != b->wt_slot) return 0;
    }
    return 1;
}

/* ============================================================================
 * SECTION — Resident int4 weight packing (per-channel and group-wise variants)
 * ==========================================================================*/

/* Scatter a pre-quantized int4 weight B[N,K] into the resident per-worker NPU BOs
 * (nibble-packed, (N/64,K/32,64,32) layout per K-tile). Shared by both pack
 * variants; the worker's Kt/nKt tiling (CBUF-max or =group) is already fixed. */
static int rk4_scatter_weights(rocket_i4_ctx *ctx, rk4_scratch *sc, rocket_i4_weights *w,
                               const int8_t *B, int K) {
    int t = 0;
    for (; t < sc->nt; t++) {
        rk4_worker *ww = &sc->w[t];
        size_t wt_sz = (size_t)ww->nNt * ww->nKt * (ww->wt_slot / 2) + I4_CBUF_BANK;
        if (rocket_bo_alloc(ctx->fd[t], wt_sz, &w->wt[t]) != 0 || i4_iova_of(&w->wt[t], wt_sz)) {
            ROCKET_LOGE("rocket_i4_weights_pack: wt BO alloc/IOVA failed (worker %d, %zuMB)\n", t, wt_sz >> 20);
            if (w->wt[t].handle) rocket_bo_free(ctx->fd[t], &w->wt[t]);
            goto fail;
        }
        if (rocket_bo_prep(ctx->fd[t], &w->wt[t], 1, 0) != 0) {  /* sync failed (logged) */
            rocket_bo_free(ctx->fd[t], &w->wt[t]); goto fail;    /* fail frees [0,t); free t here */
        }
        memset(w->wt[t].ptr, 0, w->wt[t].size);
        const int8_t *Bslice = B + (size_t)ww->n0 * K;
        for (int ni = 0; ni < ww->nNt; ni++) {
            int n0 = ni * ww->Nt, Ntile = (ww->nsub - n0 < ww->Nt) ? (ww->nsub - n0) : ww->Nt;
            for (int ki = 0; ki < ww->nKt; ki++) {
                int k0 = ki * ww->Kt, Ktile = (K - k0 < ww->Kt) ? (K - k0) : ww->Kt;
                uint8_t *slot = (uint8_t *)w->wt[t].ptr + (size_t)(ni * ww->nKt + ki) * (ww->wt_slot / 2);
                for (int kk = 1; kk <= Ntile; kk++)
                    for (int c = 1; c <= Ktile; c++)
                        i4_put_nib(slot, i4_wt_idx(Ktile, kk, c), Bslice[(size_t)(n0 + kk - 1) * K + (k0 + c - 1)]);
            }
        }
        rocket_bo_fini(ctx->fd[t], &w->wt[t]);
    }
    return 0;
fail:
    for (int u = 0; u < t; u++) rocket_bo_free(ctx->fd[u], &w->wt[u]);
    return -1;
}

rocket_i4_weights *rocket_i4_weights_pack(rocket_i4_ctx *ctx, int M, int K, int N, const int8_t *B) {
    if (!ctx) return NULL;
    /* Up-front shape contract — else rk4_worker_alloc fails deep inside with an
     * opaque "unsupported slice". N%64 (the int4 weight N-group) keeps every
     * N-slice aligned; K%32 the K-group; M%4 (or the M==1 GEMV case). */
    if (N % 64 != 0 || K % 32 != 0 || M % 4 != 0) {   /* M==1 broken on HW; pad caller-side */
        ROCKET_LOGE("rocket_i4_weights_pack: unsupported shape M=%d K=%d N=%d "
                "(need N%%64, K%%32, M%%4)\n", M, K, N);
        return NULL;
    }
    rk4_scratch *sc = rk4_ctx_scratch(ctx, M, K, N, 0);
    if (!sc) return NULL;
    rocket_i4_weights *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->M = M; w->K = K; w->N = N; w->nt = sc->nt; w->group = 0; w->sc = sc;
    if (rk4_scatter_weights(ctx, sc, w, B, K) < 0) { free(w); return NULL; }
    return w;
}

/* Group-wise resident pack: like rocket_i4_weights_pack but with the K-tile forced
 * to `group` (so each tile is one quant group). The matmul partner is
 * rocket_matmul_int4_prepacked_gw (per-group fp32 dequant). Hadamard, when used,
 * is baked into B by the caller (the rotation is product-preserving). */
rocket_i4_weights *rocket_i4_weights_pack_gw(rocket_i4_ctx *ctx, int M, int K, int N,
                                             const int8_t *B, int group) {
    if (!ctx) return NULL;
    if (N % 64 != 0 || K % 32 != 0 || M % 4 != 0) {
        ROCKET_LOGE("rocket_i4_weights_pack_gw: unsupported shape M=%d K=%d N=%d "
                "(need N%%64, K%%32, M%%4)\n", M, K, N);
        return NULL;
    }
    if (group < 32 || group % 32 || K % group || 49 * group >= 32767) {
        ROCKET_LOGE("rocket_i4_weights_pack_gw: bad group=%d (need %%32, |K, 49*group<32767) K=%d\n",
                group, K);
        return NULL;
    }
    rk4_scratch *sc = rk4_ctx_scratch(ctx, M, K, N, group);
    if (!sc) return NULL;
    rocket_i4_weights *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->M = M; w->K = K; w->N = N; w->nt = sc->nt; w->group = group; w->sc = sc;
    if (rk4_scatter_weights(ctx, sc, w, B, K) < 0) { free(w); return NULL; }
    return w;
}

void rocket_i4_weights_free(rocket_i4_ctx *ctx, rocket_i4_weights *w) {
    if (!ctx || !w) return;
    for (int t = 0; t < w->nt; t++) rocket_bo_free(ctx->fd[t], &w->wt[t]);
    free(w);
}

size_t rocket_i4_weights_bytes(const rocket_i4_weights *w) {
    if (!w) return 0;
    /* Resident NPU-BO footprint = the per-worker nibble-packed weight tiles (the
     * shared scratch is cached across weights and not counted here). */
    size_t bytes = 0;
    for (int t = 0; t < w->nt; t++) bytes += w->wt[t].size;
    return bytes;
}

/* ============================================================================
 * SECTION — Per-channel compute thread and public API (raw int32 output)
 * ==========================================================================*/

typedef struct {
    int fd; rk4_worker *ww; const rocket_bo *wt;
    const int8_t *A; int32_t *C;
    int M, K, N, idx, ret;
} rk4_arg;

static void *rk4_thread(void *a) {
    rk4_arg *t = (rk4_arg *)a;
    rocket_pin_worker(t->idx);
    rk4_worker *w = t->ww;
    int fd = t->fd, M = t->M, K = t->K, N = t->N, nsub = w->nsub;
    int Mt = w->Mt, Kt = w->Kt, Nt = w->Nt, nMt = w->nMt, nNt = w->nNt, nKt = w->nKt;
    t->ret = 0;

    /* pack A[M,K] -> (K/32,M,32) int4 nibble feature cube */
    if (rocket_bo_prep(fd, &w->in_all, 1, 0) != 0) { t->ret = -1; return NULL; }  /* sync failed (logged) */
    memset(w->in_all.ptr, 0, w->in_all.size);
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            uint8_t *slot = (uint8_t *)w->in_all.ptr + (size_t)(mi * nKt + ki) * (w->in_slot / 2);
            for (int h = 1; h <= Mtile; h++)
                for (int c = 1; c <= Ktile; c++)
                    i4_put_nib(slot, i4_feat_idx(Mtile, c, h), t->A[(size_t)(m0 + h - 1) * K + (k0 + c - 1)]);
        }
    }
    /* Flush the packed input before the NPU reads it (propagate sync failure). */
    if ((t->ret = rocket_bo_fini(fd, &w->in_all)) != 0) return NULL;

    int64_t *acc = w->acc;
    memset(acc, 0, (size_t)M * nsub * sizeof(int64_t));
    uint64_t npu_regs[256] = {0};
    rocket_task_desc *tasks = w->tasks;
    /* Contiguous chaining is HW-blocked for the integer datapath (the int32 CACC
     * clears per-kick, not per-task, so chained tasks accumulate onto the previous
     * task's residual — first-tile-exact, rest-garbage). Force gapped; the resident
     * path still batches into one ioctl (lever 1). See rocket_prepacked_int8.c. */
    int chained = 0;
    (void)rkt_chain_enabled;
    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (nsub - n0 < Nt) ? (nsub - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int Ktile = (K - ki * Kt < Kt) ? (K - ki * Kt) : Kt;
                if (nb == 0 && (t->ret = rocket_bo_prep(fd, &w->regcmd, 1, 0)) != 0) return NULL;
                size_t out_off = (size_t)nb * w->out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma   = (uint32_t)(w->in_all.dma_address + (size_t)(mi*nKt+ki) * (w->in_slot / 2)),
                    .weights_dma = (uint32_t)(t->wt->dma_address     + (size_t)(ni*nKt+ki) * (w->wt_slot / 2)),
                    .output_dma  = (uint32_t)(w->out_all.dma_address + out_off * sizeof(int16_t)),
                    .tasks = npu_regs,
                };
                if ((t->ret = gen_matmul_int4(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul_int4_prepacked: gen failed (%d)\n", t->ret); return NULL;
                }
                if (p.task_count > I4_RC_STRIDE) {  /* unconditional: -DNDEBUG strips asserts */
                    ROCKET_LOGE("rocket_matmul_int4_prepacked: regcmd slot overflow "
                            "(task_count %u > %d words)\n", p.task_count, I4_RC_STRIDE);
                    t->ret = -1; return NULL;
                }
                rkt_chain_pack(chained, &w->regcmd, tasks, nb, npu_regs,
                               p.task_count, I4_RC_STRIDE);
                w->bm0[nb] = m0; w->bn0[nb] = n0; w->bMtile[nb] = Mtile; w->bNtile[nb] = Ntile; w->boff[nb] = out_off;
                nb++; done_tiles++;
                if (nb == I4_BATCH || done_tiles == total) {
                    /* Seal the chain then flush regcmd + ready output before submit. */
                    rkt_chain_seal(chained, &w->regcmd, nb, tasks[0].regcmd_count);
                    if ((t->ret = rocket_bo_fini(fd, &w->regcmd))   != 0) return NULL;
                    if ((t->ret = rocket_bo_prep(fd, &w->out_all, 1, 0)) != 0) return NULL;
                    if ((t->ret = rocket_bo_fini(fd, &w->out_all))  != 0) return NULL;
                    uint32_t in_h[]  = { w->in_all.handle, t->wt->handle, w->regcmd.handle };
                    uint32_t out_h[] = { w->out_all.handle };
                    if ((t->ret = rocket_submit_tasks_pre(fd, w->submit_dt, tasks, nb, in_h, 3, out_h, 1, 0)) != 0) return NULL;
                    if ((t->ret = rocket_bo_prep(fd, &w->out_all, 0, i4_wait_ns())) != 0) {
                        ROCKET_LOGE("rocket_matmul_int4_prepacked: WAIT TIMEOUT (%d) M=%d K=%d N=%d slice=%d\n",
                                t->ret, M, K, nsub, w->n0);
                        return NULL;
                    }
                    int16_t *ob = (int16_t *)w->out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        int16_t *slot = ob + w->boff[j];
                        for (int h = 1; h <= w->bMtile[j]; h++)
                            for (int nn = 1; nn <= w->bNtile[j]; nn++)
                                acc[(size_t)(w->bm0[j] + h - 1) * nsub + (w->bn0[j] + nn - 1)] +=
                                    (int64_t)slot[i4_out_idx(w->bMtile[j], nn, h)];
                    }
                    if ((t->ret = rocket_bo_fini(fd, &w->out_all)) != 0) return NULL;
                    nb = 0;
                }
            }
        }
    }
    for (int m = 0; m < M; m++)
        for (int n = 0; n < nsub; n++)
            t->C[(size_t)m * N + (w->n0 + n)] = (int32_t)acc[(size_t)m * nsub + n];
    return NULL;
}

int rocket_matmul_int4_prepacked(rocket_i4_ctx *ctx, int M, int K, int N,
                                 const int8_t *A, int32_t *C, rocket_i4_weights *w) {
    if (!ctx || !w || !w->sc) return -1;
    if (!rk4_call_m_ok("rocket_matmul_int4_prepacked", M)) return -1;
    if (K != w->K || N != w->N) {
        ROCKET_LOGE("rocket_matmul_int4_prepacked: shape K=%d N=%d != packed %d/%d\n",
                K, N, w->K, w->N);
        return -1;
    }
    /* Use the scratch for the CALL's M (not the pack-time M): the resident weight is
     * M-independent (canonical tiling), so it is reused across M with no re-pack. Reject
     * only a genuine tiling mismatch (-2) so the caller re-packs. */
    rk4_scratch *sc = rk4_ctx_scratch(ctx, M, K, N, w->group);
    if (!sc) return -1;
    if (!rk4_weight_fits(w->sc, sc)) {
        ROCKET_LOGE("rocket_matmul_int4_prepacked: weight tiling (packed M=%d) "
                "incompatible with M=%d — re-pack needed\n", w->M, M);
        return -2;
    }
    pthread_t th[RK4_MAX_WORKERS];
    rk4_arg   args[RK4_MAX_WORKERS];
    int joinable[RK4_MAX_WORKERS] = {0};
    for (int t = 0; t < sc->nt; t++) {
        args[t] = (rk4_arg){ ctx->fd[t], &sc->w[t], &w->wt[t], A, C, M, K, N, t, 0 };
        if (pthread_create(&th[t], NULL, rk4_thread, &args[t]) == 0) joinable[t] = 1;
        /* else: run inline AFTER the loop (see rocket_matmul_fp16_mt) so a create
         * failure doesn't serialize the remaining workers behind its ~8s NPU wait. */
    }
    for (int t = 0; t < sc->nt; t++)
        if (!joinable[t]) rk4_thread(&args[t]);
    int ret = 0;
    for (int t = 0; t < sc->nt; t++) {
        if (joinable[t]) pthread_join(th[t], NULL);
        if (args[t].ret) ret = args[t].ret;
    }
    return ret;
}

/* ============================================================================
 * SECTION — Group-wise int4 (W4A4): per-group fp32 dequant compute and API
 * ==========================================================================*/

/* ===========================================================================
 * Group-wise resident int4: the int4-resident sibling of rocket_matmul_int4_groupwise.
 * Each worker tiles K at Kt=group, so a tile's int16 partial is exactly one quant
 * group's contribution; it is dequantized in fp32 by a_scale[m,g]*b_scale[n,g] and
 * accumulated across groups on the host. Weight resident (rocket_i4_weights_pack_gw);
 * only A is packed per call. This is the in-model W4A4 path: the ggml-rocket backend
 * bakes Hadamard into the resident B once and rotates A per call (product-preserving).
 * ===========================================================================*/
typedef struct {
    int fd; rk4_worker *ww; const rocket_bo *wt;
    const int8_t *A; const float *a_scale; const float *b_scale; float *Cf;
    int M, K, N, group, idx, ret;
} rk4_arg_gw;

static void *rk4_thread_gw(void *a) {
    rk4_arg_gw *t = (rk4_arg_gw *)a;
    rocket_pin_worker(t->idx);
    rk4_worker *w = t->ww;
    int fd = t->fd, M = t->M, K = t->K, nsub = w->nsub, group = t->group;
    int Mt = w->Mt, Kt = w->Kt, Nt = w->Nt, nMt = w->nMt, nNt = w->nNt, nKt = w->nKt;
    int nG = K / group;                          /* == nKt (one tile == one group) */
    t->ret = 0;

    /* pack A[M,K] -> (group/32,M,32) int4 nibble feature cube (Kt=group tiling) */
    if (rocket_bo_prep(fd, &w->in_all, 1, 0) != 0) { t->ret = -1; return NULL; }  /* sync failed (logged) */
    memset(w->in_all.ptr, 0, w->in_all.size);
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ki = 0; ki < nKt; ki++) {
            int k0 = ki * Kt, Ktile = (K - k0 < Kt) ? (K - k0) : Kt;
            uint8_t *slot = (uint8_t *)w->in_all.ptr + (size_t)(mi * nKt + ki) * (w->in_slot / 2);
            for (int h = 1; h <= Mtile; h++)
                for (int c = 1; c <= Ktile; c++)
                    i4_put_nib(slot, i4_feat_idx(Mtile, c, h), t->A[(size_t)(m0 + h - 1) * K + (k0 + c - 1)]);
        }
    }
    if ((t->ret = rocket_bo_fini(fd, &w->in_all)) != 0) return NULL;

    float *facc = w->facc;
    memset(facc, 0, (size_t)M * nsub * sizeof(float));
    uint64_t npu_regs[256] = {0};
    rocket_task_desc *tasks = w->tasks;
    /* Contiguous chaining is HW-blocked for the integer datapath (the int32 CACC
     * clears per-kick, not per-task, so chained tasks accumulate onto the previous
     * task's residual — first-tile-exact, rest-garbage). Force gapped; the resident
     * path still batches into one ioctl (lever 1). See rocket_prepacked_int8.c. */
    int chained = 0;
    (void)rkt_chain_enabled;
    int total = nMt * nNt * nKt, done_tiles = 0, nb = 0;
    for (int mi = 0; mi < nMt; mi++) {
        int m0 = mi * Mt, Mtile = (M - m0 < Mt) ? (M - m0) : Mt;
        for (int ni = 0; ni < nNt; ni++) {
            int n0 = ni * Nt, Ntile = (nsub - n0 < Nt) ? (nsub - n0) : Nt;
            for (int ki = 0; ki < nKt; ki++) {
                int Ktile = (K - ki * Kt < Kt) ? (K - ki * Kt) : Kt;
                if (nb == 0 && (t->ret = rocket_bo_prep(fd, &w->regcmd, 1, 0)) != 0) return NULL;
                size_t out_off = (size_t)nb * w->out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma   = (uint32_t)(w->in_all.dma_address + (size_t)(mi*nKt+ki) * (w->in_slot / 2)),
                    .weights_dma = (uint32_t)(t->wt->dma_address     + (size_t)(ni*nKt+ki) * (w->wt_slot / 2)),
                    .output_dma  = (uint32_t)(w->out_all.dma_address + out_off * sizeof(int16_t)),
                    .tasks = npu_regs,
                };
                if ((t->ret = gen_matmul_int4(&p)) != 0) {
                    ROCKET_LOGE("rocket_matmul_int4_prepacked_gw: gen failed (%d)\n", t->ret); return NULL;
                }
                if (p.task_count > I4_RC_STRIDE) {  /* unconditional: -DNDEBUG strips asserts */
                    ROCKET_LOGE("rocket_matmul_int4_prepacked_gw: regcmd slot overflow "
                            "(task_count %u > %d words)\n", p.task_count, I4_RC_STRIDE);
                    t->ret = -1; return NULL;
                }
                rkt_chain_pack(chained, &w->regcmd, tasks, nb, npu_regs,
                               p.task_count, I4_RC_STRIDE);
                w->bm0[nb] = m0; w->bn0[nb] = n0; w->bMtile[nb] = Mtile; w->bNtile[nb] = Ntile;
                w->bki[nb] = ki; w->boff[nb] = out_off;
                nb++; done_tiles++;
                if (nb == I4_BATCH || done_tiles == total) {
                    /* Seal the chain then flush regcmd + ready output before submit. */
                    rkt_chain_seal(chained, &w->regcmd, nb, tasks[0].regcmd_count);
                    if ((t->ret = rocket_bo_fini(fd, &w->regcmd))   != 0) return NULL;
                    if ((t->ret = rocket_bo_prep(fd, &w->out_all, 1, 0)) != 0) return NULL;
                    if ((t->ret = rocket_bo_fini(fd, &w->out_all))  != 0) return NULL;
                    uint32_t in_h[]  = { w->in_all.handle, t->wt->handle, w->regcmd.handle };
                    uint32_t out_h[] = { w->out_all.handle };
                    if ((t->ret = rocket_submit_tasks_pre(fd, w->submit_dt, tasks, nb, in_h, 3, out_h, 1, 0)) != 0) return NULL;
                    if ((t->ret = rocket_bo_prep(fd, &w->out_all, 0, i4_wait_ns())) != 0) {
                        ROCKET_LOGE("rocket_matmul_int4_prepacked_gw: WAIT TIMEOUT (%d) M=%d K=%d N=%d slice=%d\n",
                                t->ret, M, K, nsub, w->n0);
                        return NULL;
                    }
                    int16_t *ob = (int16_t *)w->out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        int16_t *slot = ob + w->boff[j];
                        int g = w->bki[j];                       /* this tile's K-group */
                        for (int h = 1; h <= w->bMtile[j]; h++) {
                            int mrow = w->bm0[j] + h - 1;
                            float as = t->a_scale[(size_t)mrow * nG + g];
                            for (int nn = 1; nn <= w->bNtile[j]; nn++) {
                                int ncol_loc = w->bn0[j] + nn - 1;       /* within this worker's N-slice */
                                int ncol     = w->n0 + ncol_loc;         /* global N index (b_scale) */
                                facc[(size_t)mrow * nsub + ncol_loc] +=
                                    as * t->b_scale[(size_t)ncol * nG + g] *
                                    (float)slot[i4_out_idx(w->bMtile[j], nn, h)];
                            }
                        }
                    }
                    if ((t->ret = rocket_bo_fini(fd, &w->out_all)) != 0) return NULL;
                    nb = 0;
                }
            }
        }
    }
    for (int m = 0; m < M; m++)
        for (int n = 0; n < nsub; n++)
            t->Cf[(size_t)m * t->N + (w->n0 + n)] = facc[(size_t)m * nsub + n];
    return NULL;
}

int rocket_matmul_int4_prepacked_gw(rocket_i4_ctx *ctx, int M, int K, int N,
                                    const int8_t *A, const float *a_scale,
                                    const float *b_scale, float *Cf, rocket_i4_weights *w) {
    if (!ctx || !w || !w->sc || w->group <= 0) return -1;
    if (!rk4_call_m_ok("rocket_matmul_int4_prepacked_gw", M)) return -1;
    if (K != w->K || N != w->N) {
        ROCKET_LOGE("rocket_matmul_int4_prepacked_gw: shape K=%d N=%d != packed %d/%d\n",
                K, N, w->K, w->N);
        return -1;
    }
    /* Scratch for the CALL's M; the resident weight is M-independent (canonical tiling),
     * so reuse it across M with no re-pack. Reject a genuine tiling mismatch (-2). */
    rk4_scratch *sc = rk4_ctx_scratch(ctx, M, K, N, w->group);
    if (!sc) return -1;
    if (!rk4_weight_fits(w->sc, sc)) {
        ROCKET_LOGE("rocket_matmul_int4_prepacked_gw: weight tiling (packed M=%d) "
                "incompatible with M=%d — re-pack needed\n", w->M, M);
        return -2;
    }
    pthread_t th[RK4_MAX_WORKERS];
    rk4_arg_gw args[RK4_MAX_WORKERS];
    int joinable[RK4_MAX_WORKERS] = {0};
    for (int t = 0; t < sc->nt; t++) {
        args[t] = (rk4_arg_gw){ ctx->fd[t], &sc->w[t], &w->wt[t], A, a_scale, b_scale, Cf,
                                M, K, N, w->group, t, 0 };
        if (pthread_create(&th[t], NULL, rk4_thread_gw, &args[t]) == 0) joinable[t] = 1;
    }
    for (int t = 0; t < sc->nt; t++)
        if (!joinable[t]) rk4_thread_gw(&args[t]);
    int ret = 0;
    for (int t = 0; t < sc->nt; t++) {
        if (joinable[t]) pthread_join(th[t], NULL);
        if (args[t].ret) ret = args[t].ret;
    }
    return ret;
}
