// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_prepacked.c — pack-weights-once + streaming matmul paths (the pack-once path
 * an LLM/Whisper backend consumes). Both fan out N across per-core worker fds (rocket_matmul_fp16_mt's
 * split) and hoist the per-call fixed costs (fd open/close, BO alloc) out of the loop.
 *
 * SHARED SCRATCH. The NPU regcmd addresses BOs with 32-bit fields,
 * so resident weight BOs must fit each fd's low-4GB IOVA window (~20GB across 5 fds).
 * The compute SCRATCH (input/output/regcmd/ping-pong + host arrays) is ~2x the weight
 * bytes and is identical for every same-shape matmul, so giving each resident weight
 * its own scratch wasted ~67% of that window (~30% of a 12B model fit). Fix: cache ONE
 * `mm_scratch` per (M,K,N) on the rocket_ctx, shared by streaming AND prepacked; a
 * prepacked weight then owns only its resident `wt` BO and borrows the shared scratch,
 * pointing the scratch's `wt_all` at its weight for the duration of the compute. So:
 *
 *   - mm_scratch  : per-shape compute scratch (per-worker mm_bos + shared A-pack),
 *                   cached on the ctx. Streaming re-packs B into its wt_all each call.
 *   - rocket_weights : a per-NAME prepacked handle = resident wt BOs + an M-independent
 *                      layout signature. NOT pinned to one scratch — the compute borrows
 *                      whichever per-shape scratch matches the CALL's M, so a weight
 *                      packed at warmup-M is reused at prefill-M when the tiling matches.
 *
 * The compute primitives (rocket_matmul.c) are untouched -- a worker temporarily swaps
 * the resident weight into its (per-worker, serially-used) scratch bos and restores it,
 * so teardown frees the scratch's own wt_all, not a borrowed weight. The shared-scratch
 * ownership (correctness + no cross-weight aliasing + capacity) was validated standalone
 * in tests/prototype_shared_scratch_rocket.c before this refactor.
 *
 * The pool shape, the slot cache, the fence deadline, the column split and the
 * spawn/join orchestrator are rocket_fanout.h's, shared with the int8/int4/bf16 paths.
 * What stays per-dtype is the COMPUTE core, because the operand element types, the BO
 * geometry and the K-accumulator type genuinely differ -- fp16 accumulates K on the NPU
 * through a ping-pong the integer paths cannot use at all.
 *
 * Lifetime: free every rocket_weights (its resident wt BOs) before rocket_ctx_free,
 * which frees the shared scratch (BOs) and then closes the fds.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "rocket_matmul.h"
#include "rocket_matmul_internal.h"
#include "rocket_affinity.h"
#include "rocket_fanout.h"
#include "rocket_log.h"     // centralized log channel

#define RKW_MAX_WORKERS ROCKET_FANOUT_MAX_WORKERS

/* one worker's N-slice: its column range, tiling plan, and shared compute scratch. */
typedef struct {
    int     n0, nsub;     /* this worker's output-column slice C[:, n0:n0+nsub)        */
    mm_plan pl;           /* tiling plan for (M, K, nsub)                              */
    mm_bos  bos;          /* shared scratch on ctx->fd[worker]; bos.wt_all is the
                           * streaming re-pack target / temporarily swapped to a
                           * prepacked weight's resident BO during its compute.        */
} rkw_worker;

/* Per-(M,K,N) compute scratch, cached on the ctx and shared by every same-shape
 * matmul (streaming or prepacked). This is the only thing holding the bulky
 * input/output/regcmd/ping-pong BOs; resident weights borrow it. */
typedef struct {
    int M, K, N, nt;
    rkw_worker w[RKW_MAX_WORKERS];
    _Float16 *a_buf;          /* persistent shared A-pack buffer, zeroed ONCE;
                               * sized to mm_input_elems(w[0].pl), fixed per shape.    */
} mm_scratch;

/* Per-NAME prepacked resident weight: the scattered weight BOs (per worker N-slice) +
 * the layout signature. NOT bound to a single (M,K,N) scratch: the compute uses the
 * scratch for the CALL's M, reusing this weight whenever the tiling matches. */
struct rocket_weights {
    int M, K, N, nt;                   /* M = the pack-time M (reference only)                  */
    rocket_bo   wt[RKW_MAX_WORKERS];   /* resident weight per worker (the only per-name alloc)  */
    rocket_wsig sig[RKW_MAX_WORKERS];  /* per-worker layout signature for compatibility checks  */
};

struct rocket_ctx {
    int nthreads;
    const struct rocket_hw_profile *hw; /* active machine-parameter profile (RK3588 today);
                                         * the per-device autodetect seam for multi-chip support.       */
    int fd[RKW_MAX_WORKERS];
    rocket_slot_cache scache;          /* shared per-shape scratch (streaming + prepacked) */
};

/* ============================================================================
 * SECTION — Context lifecycle (worker fds) and shared per-shape scratch
 * ==========================================================================*/

rocket_ctx *rocket_ctx_create(int nthreads)
{
    if (nthreads < 1) nthreads = 1;
    if (nthreads > RKW_MAX_WORKERS) nthreads = RKW_MAX_WORKERS;

    rocket_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->nthreads = nthreads;
    ctx->hw = rocket_hw_current();
    for (int t = 0; t < nthreads; t++) ctx->fd[t] = -1;

    for (int t = 0; t < nthreads; t++) {
        ctx->fd[t] = rocket_open();
        if (ctx->fd[t] < 0) {
            ROCKET_LOGE("rocket_ctx_create: rocket_open worker %d failed (%d)\n", t, ctx->fd[t]);
            for (int u = 0; u < t; u++) rocket_close(ctx->fd[u]);
            free(ctx);
            return NULL;
        }
    }
    return ctx;
}

static void scratch_free(rocket_ctx *ctx, mm_scratch *sc)
{
    if (!sc) return;
    for (int t = 0; t < sc->nt; t++) mm_bos_free(ctx->fd[t], &sc->w[t].bos);
    free(sc->a_buf);
    free(sc);
}

static mm_scratch *scratch_alloc(rocket_ctx *ctx, int M, int K, int N);

/* How the shared slot cache builds and releases one of our per-shape scratches. */
static void *rkw_slot_alloc(void *owner, const rocket_shape_key *k)
{
    return scratch_alloc((rocket_ctx *)owner, k->M, k->K, k->N);
}
static void rkw_slot_release(void *owner, void *slot)
{
    scratch_free((rocket_ctx *)owner, (mm_scratch *)slot);
}
static const rocket_slot_ops rkw_slot_ops = { rkw_slot_alloc, rkw_slot_release };

void rocket_ctx_free(rocket_ctx *ctx)
{
    if (!ctx) return;
    /* free shared scratch (holds BOs on the fds) BEFORE closing the fds */
    rocket_slot_cache_clear(&ctx->scache, ctx, &rkw_slot_ops);
    for (int t = 0; t < ctx->nthreads; t++)
        if (ctx->fd[t] >= 0) rocket_close(ctx->fd[t]);
    free(ctx);
}

/* Allocate the per-worker N-slice plans + shared scratch BOs for (M,K,N). */
static mm_scratch *scratch_alloc(rocket_ctx *ctx, int M, int K, int N)
{
    mm_scratch *sc = calloc(1, sizeof(*sc));
    if (!sc) return NULL;
    sc->M = M; sc->K = K; sc->N = N;

    int Nstep = rocket_fanout_nstep(N, ctx->nthreads, 16);  /* the generator needs N%16 */
    int t = 0;
    for (; t < ctx->nthreads; t++) {
        int n0 = t * Nstep;
        if (n0 >= N) break;                 /* fewer slices than fds when N small */
        int nsub = (n0 + Nstep > N) ? (N - n0) : Nstep;

        rkw_worker *ww = &sc->w[t];
        ww->n0 = n0; ww->nsub = nsub;
        if (mm_plan_init(&ww->pl, M, K, nsub) < 0) {
            ROCKET_LOGE("scratch_alloc: unsupported slice M=%d K=%d N=%d\n", M, K, nsub);
            goto fail;
        }
        if (mm_bos_alloc(ctx->fd[t], &ww->pl, &ww->bos) < 0)
            goto fail;
    }
    sc->nt = t;
    return sc;

fail:
    for (int u = 0; u < t; u++) mm_bos_free(ctx->fd[u], &sc->w[u].bos);
    free(sc);
    return NULL;
}

/* Find (or build + cache) the shared per-shape scratch. NULL only on an alloc failure:
 * a full cache recycles its least recently used shape rather than refusing. */
static mm_scratch *ctx_scratch(rocket_ctx *ctx, int M, int K, int N)
{
    const rocket_shape_key k = { M, K, N, 0 };
    return rocket_slot_get(&ctx->scache, ctx, &k, &rkw_slot_ops);
}

/* ============================================================================
 * SECTION — Resident weight packing (pack-once: scatter B into per-worker BOs)
 * ==========================================================================*/

/* The body of both resident-weight packs. `B` is a whole [N,K] weight; `segs`/`nseg` are
 * a concatenated-along-N one, resolved per global column so no host concat buffer is
 * needed. Exactly one of the two is non-NULL -- everything else about the two entries
 * (scratch lookup, the per-worker BO alloc + prezero, the layout signature, the failure
 * unwind) is identical, so it is written once. */
static rocket_weights *rkw_weights_pack(rocket_ctx *ctx, int M, int K, int N,
                                        const _Float16 *B,
                                        const mm_wt_seg *segs, int nseg)
{
    if (!ctx) return NULL;
    mm_scratch *sc = ctx_scratch(ctx, M, K, N);
    if (!sc) return NULL;

    rocket_weights *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->M = M; w->K = K; w->N = N; w->nt = sc->nt;

    int t = 0;
    for (; t < sc->nt; t++) {
        rkw_worker *ww = &sc->w[t];
        /* Record the M-independent layout signature so a later compute at a different
         * (compatible) M can reuse this resident weight without re-packing. */
        w->sig[t] = (rocket_wsig){ ww->n0, ww->nsub, ww->pl.Kt, ww->pl.Nt,
                                   ww->pl.nKt, ww->pl.nNt, ww->pl.wt_slot,
                                   ww->bos.wt_all.size };
        /* resident weight BO for this worker's N-slice: same size/layout as the shared
         * scratch's wt_all (same plan). Prezero it once (mirrors mm_bos_alloc) so the
         * scatter below -- which runs with the scratch's prezeroed flag set -- writes
         * only live tile lanes and leaves the padding zero. */
        if (rocket_bo_alloc(ctx->fd[t], ww->bos.wt_all.size, &w->wt[t]) != 0) {
            ROCKET_LOGE("rocket_weights_pack: wt BO alloc failed (worker %d, %zuMB)\n",
                    t, ww->bos.wt_all.size >> 20);
            goto fail;
        }
        if (rocket_bo_prep(ctx->fd[t], &w->wt[t], 1, 0) != 0) {  /* sync failed (logged) */
            rocket_bo_free(ctx->fd[t], &w->wt[t]); goto fail;    /* fail frees [0,t); free t here */
        }
        memset(w->wt[t].ptr, 0, w->wt[t].size);
        rocket_bo_fini(ctx->fd[t], &w->wt[t]);

        /* scatter B's column-slice into the resident weight: borrow the scratch bos,
         * point its wt_all at our BO, pack, restore. Serial (pack time), so safe. */
        rocket_bo saved = ww->bos.wt_all;
        ww->bos.wt_all = w->wt[t];
        double mb = segs
            ? mm_pack_weights_seg(ctx->fd[t], &ww->pl, &ww->bos, segs, nseg, ww->n0)
            : mm_pack_weights(ctx->fd[t], &ww->pl, &ww->bos, B + (size_t)ww->n0 * K);
        ww->bos.wt_all = saved;
        if (mb < 0) { rocket_bo_free(ctx->fd[t], &w->wt[t]); goto fail; }  /* sync failed (logged) */
    }
    return w;

fail:
    for (int u = 0; u < t; u++) rocket_bo_free(ctx->fd[u], &w->wt[u]);
    free(w);
    return NULL;
}

rocket_weights *rocket_weights_pack(rocket_ctx *ctx, int M, int K, int N, const _Float16 *B)
{
    if (!B) return NULL;
    return rkw_weights_pack(ctx, M, K, N, B, NULL, 0);
}

rocket_weights *rocket_weights_pack_seg(rocket_ctx *ctx, int M, int K, int N,
                                        const _Float16 *const *Bs, const int *Ns, int nseg)
{
    if (!Bs || !Ns || nseg < 1 || nseg > ROCKET_MAX_FUSE) return NULL;
    mm_wt_seg segs[ROCKET_MAX_FUSE];
    int g0 = 0;
    for (int i = 0; i < nseg; i++) {
        if (!Bs[i] || Ns[i] <= 0) return NULL;
        segs[i] = (mm_wt_seg){ Bs[i], g0, Ns[i] };
        g0 += Ns[i];
    }
    if (g0 != N) return NULL;              /* N must be the sum of the segments */
    return rkw_weights_pack(ctx, M, K, N, NULL, segs, nseg);
}

void rocket_weights_free(rocket_ctx *ctx, rocket_weights *w)
{
    if (!ctx || !w) return;
    for (int t = 0; t < w->nt; t++) rocket_bo_free(ctx->fd[t], &w->wt[t]);
    free(w);                                /* sc is shared, owned by the ctx */
}

/* ============================================================================
 * SECTION — Per-worker compute thread and the multicore orchestrator
 * ==========================================================================*/

typedef struct {
    int fd;
    rkw_worker *ww;
    const _Float16 *A;       /* full A[M,K] (used when packed == NULL) */
    const _Float16 *packed;  /* pre-scattered canonical A buffer, or NULL */
    const _Float16 *B;       /* full B[N,K] to RE-PACK this call (stream), or NULL */
    const rocket_bo *wt;     /* prepacked resident weight to use (swapped into bos), or NULL */
    _Float16 *C;             /* full C[M,N], scattered into this worker's columns */
    int M, N;
    int ret;
    int idx;                 /* worker index, for big-core pinning */
    const mm_wt_seg *segs;   /* fused: concatenated-weight segments, or NULL */
    int nseg;
    int kacc;                /* use NPU-side K-accum compute (ROCKET_KACC) */
    int pipe;                /* use the pipelined CPU-accum compute */
    int core_base;           /* big-core rotation base inherited from the caller thread */
} rkw_arg;

static void *rkw_thread(void *a)
{
    rkw_arg *t = (rkw_arg *)a;
    rocket_pin_worker_based(t->idx, t->core_base);   /* keep the pack/readback off the A55s */
    rkw_worker *ww = t->ww;
    int nsub = ww->nsub;

    /* Resident, not a per-call malloc: at a prefill shape this is megabytes, and
     * faulting them in on every matmul cost more than the copy out of them does. */
    _Float16 *Csub = mm_csub(&ww->bos, (size_t)t->M * nsub);
    if (!Csub) { t->ret = -1; return NULL; }

    /* Prepacked: use this weight's RESIDENT BO for the compute. Swap it into the
     * shared scratch's wt_all and restore afterwards, so the scratch's own wt_all
     * (the streaming re-pack target) is intact for teardown / the next call. This
     * worker owns ww exclusively (one fd/worker) and matmuls run serially, so the
     * swap is race-free (validated in prototype_shared_scratch_rocket). */
    rocket_bo saved_wt = ww->bos.wt_all;
    if (t->wt) ww->bos.wt_all = *t->wt;

    /* Streaming: weights aren't resident, so re-pack this worker's B column-slice into
     * the scratch's weight BO before compute. Concurrent across workers, so the B
     * scatter overlaps like the prepacked path's A pack. (NULL => prepacked above.) */
    if (t->segs || t->B) {
        double mb = t->segs
            ? mm_pack_weights_seg(t->fd, &ww->pl, &ww->bos, t->segs, t->nseg, ww->n0)
            : mm_pack_weights(t->fd, &ww->pl, &ww->bos, t->B + (size_t)ww->n0 * ww->pl.K);
        if (mb < 0) {   /* weight cache-sync failed (logged): restore the swap and bail */
            if (t->wt) ww->bos.wt_all = saved_wt;
            t->ret = -1; return NULL;
        }
        mm_prof_add_pack_b(mb);
    }

    /* When all workers share the input layout, A was scattered once upstream and
     * we just memcpy it in (memcpy >> the feat_idx scatter); else pack per-worker. */
    double t_pack = t->packed ? mm_load_input(t->fd, &ww->pl, &ww->bos, t->packed)
                              : mm_pack_input(t->fd, &ww->pl, &ww->bos, t->A);
    if (t_pack < 0) {   /* input cache-sync failed (logged): restore the swap and bail */
        if (t->wt) ww->bos.wt_all = saved_wt;
        t->ret = -1; return NULL;
    }
    if (t->kacc) {
        t->ret = mm_compute_kacc(t->fd, &ww->pl, &ww->bos, Csub, t_pack);
        if (t->ret == -2)   /* tiny-M tile: fall back to the CPU-accum oracle */
            t->ret = mm_compute(t->fd, &ww->pl, &ww->bos, Csub, t_pack);
    } else if (t->pipe) {
        t->ret = mm_compute_pipe(t->fd, &ww->pl, &ww->bos, Csub, t_pack);
    } else {
        t->ret = mm_compute(t->fd, &ww->pl, &ww->bos, Csub, t_pack);
    }

    if (t->wt) ww->bos.wt_all = saved_wt;   /* restore the scratch's own weight BO */

    if (t->ret == 0)
        for (int m = 0; m < t->M; m++)
            memcpy(t->C + (size_t)m * t->N + ww->n0,
                   Csub + (size_t)m * nsub,
                   (size_t)nsub * sizeof(_Float16));
    return NULL;
}

/* Shared multicore orchestrator. `rw` non-NULL => prepacked (each worker uses
 * rw->wt[t], B must be NULL). `B`/`segs` non-NULL => streaming (re-pack into the
 * scratch's own weight BO). */
static int rkw_run(rocket_ctx *ctx, int M, int K, int N,
                   const _Float16 *A, const _Float16 *B, _Float16 *C, mm_scratch *sc,
                   rocket_weights *rw, const mm_wt_seg *segs, int nseg)
{
    if (!ctx || !sc) return -1;
    if (M != sc->M || K != sc->K || N != sc->N) {
        ROCKET_LOGE("rkw_run: shape M=%d K=%d N=%d != scratch %d/%d/%d\n",
                M, K, N, sc->M, sc->K, sc->N);
        return -1;
    }

    /* If every worker has the same input tiling plan, A packs identically for all
     * of them — scatter it ONCE here and hand workers the buffer to memcpy.
     * ROCKET_NO_SHARED_PACK=1 forces per-worker mm_pack_input instead. */
    const mm_plan *p0 = &sc->w[0].pl;
    int shared = rocket_fanout_shared_pack();
    for (int t = 1; shared && t < sc->nt; t++) {
        const mm_plan *pt = &sc->w[t].pl;
        if (pt->Mt != p0->Mt || pt->Kt != p0->Kt ||
            pt->nMt != p0->nMt || pt->nKt != p0->nKt || pt->in_slot != p0->in_slot) {
            shared = 0; break;
        }
    }

    _Float16 *packed = NULL;
    if (shared) {
        /* Persistent shared A buffer, allocated + zeroed ONCE per shape. */
        if (!sc->a_buf)
            sc->a_buf = calloc(mm_input_elems(p0), sizeof(_Float16));
        packed = sc->a_buf;
        if (packed) mm_prof_add_pack(mm_scatter_input(p0, packed, A));
        /* alloc failure -> packed == NULL -> per-worker packing fallback */
    }

    rkw_arg args[RKW_MAX_WORKERS];
    /* Mode is fixed by what the scratch was ALLOCATED with (mm_bos_alloc set these
     * from env), not a fresh env read — so the compute path always matches the BOs
     * present (no silent mm_compute_pipe -1 / .handle==0 pong if env changed since
     * the scratch was built). All workers' bos share the same alloc-time env. */
    int kacc = sc->w[0].bos.has_pong;           /* read once, not per worker */
    /* pipelined CPU-accum, only when KACC is off. */
    int pipe = !kacc && sc->w[0].bos.has_pipe;
    int base = rocket_affinity_get_base();      /* spread in-process pools across the cluster */

    for (int t = 0; t < sc->nt; t++) {
        const rocket_bo *wt = rw ? &rw->wt[t] : NULL;
        args[t] = (rkw_arg){ ctx->fd[t], &sc->w[t], A, packed, B, wt, C,
                             M, N, 0, t, segs, nseg, kacc, pipe, base };
    }
    rocket_fanout_run(sc->nt, args, sizeof args[0], rkw_thread);

    int ret = 0;
    for (int t = 0; t < sc->nt; t++)
        if (args[t].ret) ret = args[t].ret;
    return ret;
}

/* True iff the resident weight `w` was packed in the same per-worker tile layout that
 * shape (M,K,N) plans now — i.e. the weight bytes are valid to compute against at this M.
 * Holds for any two M that yield the same Kt/Nt/N-split (notably ALL M >= MAX_TILE, where
 * Mt is capped and the tiling is M-independent). */
static int weights_fit_scratch(const rocket_weights *w, const mm_scratch *sc)
{
    if (w->K != sc->K || w->N != sc->N || w->nt != sc->nt) return 0;
    for (int t = 0; t < sc->nt; t++) {
        const rkw_worker *ww = &sc->w[t];
        const rocket_wsig now = { ww->n0, ww->nsub, ww->pl.Kt, ww->pl.Nt,
                                  ww->pl.nKt, ww->pl.nNt, ww->pl.wt_slot,
                                  ww->bos.wt_all.size };
        if (!rocket_wsig_eq(&w->sig[t], &now)) return 0;
    }
    return 1;
}

/* ============================================================================
 * SECTION — Public API: prepacked (resident-weight) matmul
 * ==========================================================================*/

int rocket_matmul_fp16_prepacked(rocket_ctx *ctx, int M, int K, int N,
                                 const _Float16 *A, _Float16 *C, rocket_weights *w)
{
    if (!ctx || !w) return -1;
    if (K != w->K || N != w->N) return -1;
    /* Use the scratch for the CALL's M (not the pack-time M): a resident weight is no
     * longer pinned to the M it was packed at. Reuses the cached scratch when M matches
     * a prior call, or allocs one for a new M. */
    mm_scratch *sc = ctx_scratch(ctx, M, K, N);
    if (!sc) return -1;
    /* The weight scatter is M-independent; reject only a genuine tiling mismatch (e.g. a
     * weight packed at M>=256 used at M<256, where Kt grows) so the caller can re-pack. */
    if (!weights_fit_scratch(w, sc)) {
        ROCKET_LOGE("rocket_matmul_fp16_prepacked: weight tiling (packed M=%d) "
                "incompatible with M=%d — re-pack needed\n", w->M, M);
        return -2;
    }
    /* weights resident in w->wt[]; pass rw=w (workers swap them in), B=NULL. */
    return rkw_run(ctx, M, K, N, A, /*B=*/NULL, C, sc, /*rw=*/w, /*segs=*/NULL, 0);
}

/* ============================================================================
 * SECTION — Public API: streaming (re-pack B per call) matmul
 * ==========================================================================*/

/* ---- streaming path: persistent fds + shared per-shape scratch, B re-packed every
 * call. See rocket_matmul.h. -------------------------------------------------- */
struct rocket_stream {
    rocket_ctx *ctx;
};

rocket_stream *rocket_stream_create(int nthreads)
{
    rocket_stream *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->ctx = rocket_ctx_create(nthreads);
    if (!s->ctx) { free(s); return NULL; }
    return s;
}

void rocket_stream_free(rocket_stream *s)
{
    if (!s) return;
    rocket_ctx_free(s->ctx);                /* frees shared scratch (BOs) then fds */
    free(s);
}

int rocket_matmul_fp16_stream(rocket_stream *s, int M, int K, int N,
                              const _Float16 *A, const _Float16 *B, _Float16 *C)
{
    if (!s) return -1;
    mm_scratch *sc = ctx_scratch(s->ctx, M, K, N);
    if (!sc) return -1;
    return rkw_run(s->ctx, M, K, N, A, B, C, sc, /*rw=*/NULL, /*segs=*/NULL, 0); /* B -> re-pack */
}

int rocket_matmul_fp16_stream_fused(rocket_stream *s, int M, int K,
                                    const _Float16 *const *Bs, const int *Ns, int nseg,
                                    const _Float16 *A, _Float16 *C)
{
    if (!s || nseg < 1 || nseg > ROCKET_MAX_FUSE) return -1;

    /* Lay the weights out contiguously along the combined N and validate. */
    mm_wt_seg segs[ROCKET_MAX_FUSE];
    int Ntot = 0;
    for (int i = 0; i < nseg; i++) {
        if (!Bs[i] || Ns[i] <= 0) return -1;
        segs[i].Bseg = Bs[i]; segs[i].g0 = Ntot; segs[i].Nseg = Ns[i];
        Ntot += Ns[i];
    }

    /* One matmul over the combined N; the shared scratch is keyed on (M,K,Ntot) just
     * like a plain matmul of that shape. Each worker re-packs its column slice from
     * the segments (segs != NULL -> mm_pack_weights_seg into the scratch's wt_all). */
    mm_scratch *sc = ctx_scratch(s->ctx, M, K, Ntot);
    if (!sc) return -1;
    return rkw_run(s->ctx, M, K, Ntot, A, /*B=*/NULL, C, sc, /*rw=*/NULL, segs, nseg);
}
