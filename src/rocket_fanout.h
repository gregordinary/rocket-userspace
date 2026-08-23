// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_fanout.h — the machinery every resident-weight matmul path shares.
 *
 * Four dtype paths (fp16 rocket_prepacked.c, int8 rocket_prepacked_int8.c, int4
 * rocket_prepacked_int4.c, bf16 rocket_bf16_stream.c) fan a matmul's N across per-core
 * worker fds and cache a compute scratch per (M,K,N,group). Their COMPUTE cores differ
 * for real — the operand element types, the BO geometry and the K-accumulator type are
 * not the same algorithm wearing different names — but everything around those cores is
 * one mechanism, and it was written four times:
 *
 *   the pool shape (8 workers, 32 shape slots), the output-fence deadline, the size_t
 *   round-up, the N column split, the per-shape slot cache, and the spawn / run-inline-
 *   on-spawn-failure / join orchestrator.
 *
 * Four copies drift, and they had: one path's slot cache keyed on a field the others
 * did not, one path's deadline read the environment on every call where the others
 * cached it, and the big-core rotation base was threaded into some workers and not
 * others. This header is where those live once.
 *
 * NOT thread-safe, and does not need to be: a rocket_ctx is single-in-flight by contract
 * (see rocket_matmul.h), so the cache is touched only from the calling thread, between
 * fan-outs rather than during one.
 */
#ifndef ROCKET_FANOUT_H
#define ROCKET_FANOUT_H

#include <stddef.h>
#include <stdint.h>

/* Worker fds per context. One drm_sched entity per fd, so this is what lets the NPU
 * cores run in parallel; 8 is well above the 3 cores an RK3588 has. */
#define ROCKET_FANOUT_MAX_WORKERS 8
/* Distinct (M,K,N,group) shapes a context keeps scratch for. Past this the least
 * recently used slot is recycled — see rocket_slot_get. */
#define ROCKET_FANOUT_MAX_SLOTS   32
/* Tiles (tasks) per NPU job, and the u64 words reserved per task in the regcmd BO.
 * Both match rocket_matmul.c's one-shot paths, which is what makes a prepacked result
 * bit-comparable to its one-shot oracle. */
#define ROCKET_FANOUT_BATCH       64
#define ROCKET_FANOUT_RC_STRIDE   128

/* Output-fence wait deadline in ns, from ROCKET_WAIT_MS (default 8000). Read ONCE
 * rather than a getenv (a locked environ scan) per worker call. */
long rocket_fanout_wait_ns(void);

/* Is the shared A-pack enabled? When every worker plans the same input tiling, A is
 * scattered once and each worker memcpy's it in, which beats a per-worker scatter.
 * ROCKET_NO_SHARED_PACK=<anything> forces the per-worker form. Cached, for the same
 * reason as the deadline: this sits on the per-matmul path. */
int rocket_fanout_shared_pack(void);

/* Round x up to a multiple of a, in size_t — these feed slot and BO allocation sizes,
 * and computing in int would truncate before the widen at the call site. */
static inline size_t rocket_rup_sz(int x, int a)
{
    return (((size_t)x + (size_t)a - 1) / (size_t)a) * (size_t)a;
}

/* Split N across `nthreads` workers, each slice rounded up to `align` — the N
 * granularity the dtype's generator requires (16 fp16/bf16, 32 int8, 64 int4). The
 * last slice is N - n0, which is a multiple of `align` whenever N is. */
int rocket_fanout_nstep(int N, int nthreads, int align);

/* ── The per-shape scratch cache ────────────────────────────────────────────────
 *
 * A context caches one compute scratch per (M,K,N,group) so a repeated shape pays the
 * BO allocation once. `group` is 0 for the paths with no quant group; it is part of the
 * key because a group-wise plan tiles K differently and cannot share a slot.
 *
 * FULL means RECYCLE, not refuse. The cache used to return NULL at slot 33 and every
 * caller turned that into a bare failure with no log — so the 33rd distinct shape was
 * indistinguishable from an unsupported one. A caller that sweeps M (variable prompt
 * lengths, a micro-batch tail, per-expert token counts) reaches that in normal use, and
 * a hard failure is the wrong answer when a re-alloc is available. */
typedef struct { int M, K, N, group; } rocket_shape_key;

typedef struct {
    /* Build the scratch for `k`, or NULL. `owner` is the caller's context. */
    void *(*alloc)(void *owner, const rocket_shape_key *k);
    /* Release a scratch built by alloc(). Must tolerate NULL. */
    void  (*release)(void *owner, void *slot);
} rocket_slot_ops;

typedef struct {
    rocket_shape_key key[ROCKET_FANOUT_MAX_SLOTS];
    void            *slot[ROCKET_FANOUT_MAX_SLOTS];
    uint64_t         stamp[ROCKET_FANOUT_MAX_SLOTS];   /* last use, for the LRU pick */
    uint64_t         clock;
    int              n;
} rocket_slot_cache;

/* Find, or build and cache, the scratch for `k`. When the cache is full the least
 * recently used slot is released and reused. NULL only if alloc() failed.
 *
 * A slot may be RECYCLED under a caller holding a resident weight packed against it, so
 * a weight must never hold a pointer INTO a slot: record the layout with rocket_wsig
 * and re-check it against whatever slot the call resolves. */
void *rocket_slot_get(rocket_slot_cache *c, void *owner,
                      const rocket_shape_key *k, const rocket_slot_ops *ops);

/* Release every cached slot. Call before closing the fds the slots hold BOs on. */
void rocket_slot_cache_clear(rocket_slot_cache *c, void *owner, const rocket_slot_ops *ops);

/* ── Resident-weight layout signature ───────────────────────────────────────────
 *
 * Where a weight element lands in its resident BO is fixed by the N-split and the K/N
 * tiling, all of which are M-independent — so a weight packed at one M is valid at any
 * other M that plans the same way (notably every M >= max_tile, where Mt is capped).
 * Recording those determinants, rather than a pointer to the scratch that produced
 * them, is what makes both reuse-across-M and slot recycling safe.
 *
 * The last two fields are implied by the first six, so comparing them costs nothing and
 * cannot refuse a layout the others accept — they are there to catch a determinant this
 * struct has not learned about yet. */
typedef struct {
    int    n0, nsub;      /* output-column slice C[:, n0 : n0+nsub)            */
    int    Kt, Nt;        /* the tiling that fixes the weight scatter position */
    int    nKt, nNt;
    size_t wt_slot;       /* elements per weight tile                          */
    size_t wt_bytes;      /* the resident weight BO's byte size                */
} rocket_wsig;

static inline int rocket_wsig_eq(const rocket_wsig *a, const rocket_wsig *b)
{
    return a->n0  == b->n0  && a->nsub    == b->nsub &&
           a->Kt  == b->Kt  && a->Nt      == b->Nt   &&
           a->nKt == b->nKt && a->nNt     == b->nNt  &&
           a->wt_slot == b->wt_slot && a->wt_bytes == b->wt_bytes;
}

/* ── The worker fan-out ─────────────────────────────────────────────────────────
 *
 * Spawn `n` workers over `args` (an array of `stride`-byte structs) and join them.
 * A worker whose pthread_create fails runs INLINE, but only after the whole spawn
 * loop — running it in place would serialize every later worker behind its NPU wait,
 * which on a timeout is ~8 s.
 *
 * The worker itself must pin with rocket_pin_worker_based() and a base its spawner
 * read: the rotation base is __thread, so a fresh worker always reads 0. */
void rocket_fanout_run(int n, void *args, size_t stride, void *(*fn)(void *));

#endif /* ROCKET_FANOUT_H */
