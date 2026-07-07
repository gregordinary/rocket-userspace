// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/* rocket_scratch.h — a small bump/arena allocator for per-call host scratch.
 *
 * The tiled op paths need several short-lived host buffers per call (band-size
 * arrays, materialized sub-input / sub-output tiles). Allocating each with malloc
 * and freeing each on every error path is correct but fragile: one missed free on a
 * newly-added early return leaks. An arena carves all of a call's scratch from ONE
 * block with ONE cleanup point, so error paths cannot leak, and — when backed by a
 * resident context — reuses a grow-only high-water block across calls so the steady
 * state does no per-call malloc at all.
 *
 * Two backings, one push/carve API:
 *   - rocket_arena_open(&a, NULL, need)  — a call-local block (malloc'd; freed on close).
 *   - rocket_arena_open(&a, &pool, need) — borrow a caller-owned grow-only pool; close()
 *     only resets the offset (no free), so the pages stay resident for the next call.
 *
 * Bit-exactness: the arena changes only WHERE scratch lives, never its size, layout, or
 * contents. Each carved sub-buffer is 16-byte aligned (>= glibc malloc's guarantee), so
 * every consumer sees byte-identical buffers.
 *
 * Threading: a pool is not internally synchronized — it inherits the single-thread-use
 * contract of whatever owns it (e.g. a rocket_conv_ctx and its BOs are used by one thread
 * at a time; each multicore worker owns a distinct ctx, hence a distinct pool). A pool is
 * a single high-water block, NOT a stack: only one frame may be open on a given pool at a
 * time — never open a second pool-backed frame while the first is live (it would realloc
 * and alias the live buffers). Nested/recursive scratch must use call-local (NULL) frames.
 */
#ifndef ROCKET_SCRATCH_H
#define ROCKET_SCRATCH_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* A context-owned, grow-only high-water scratch block. Zero-initialize it (a {0} /
 * calloc'd owner is ready to use); the owner calls rocket_scratch_pool_free() once at
 * teardown. */
typedef struct {
    void  *base;
    size_t cap;
} rocket_scratch_pool;

/* One open scratch frame: a bump cursor over either a borrowed pool or an owned block. */
typedef struct {
    unsigned char *base;   /* start of this frame's block                       */
    size_t         cap;    /* usable bytes                                       */
    size_t         off;    /* bump cursor                                        */
    int            owned;  /* 1 => free(base) on close; 0 => borrowed from a pool */
} rocket_arena;

static inline size_t rocket_arena_align_up(size_t x, size_t a)
{
    return (x + (a - 1)) & ~(a - 1);
}

/* Reserve `need` bytes for the frame. With pool != NULL, grow the pool to >= need
 * (realloc is safe here: a pool has at most one open frame, so there are no live carved
 * pointers to invalidate) and borrow it. With pool == NULL, malloc a fresh owned block.
 * Returns 0 on success, -1 on OOM (the arena is left closed/zeroed). */
static inline int rocket_arena_open(rocket_arena *a, rocket_scratch_pool *pool, size_t need)
{
    a->off = 0;
    if (need == 0) need = 1;
    if (pool) {
        if (pool->cap < need) {
            void *p = realloc(pool->base, need);
            if (!p) { a->base = NULL; a->cap = 0; a->owned = 0; return -1; }
            pool->base = p;
            pool->cap  = need;
        }
        a->base  = (unsigned char *)pool->base;
        a->cap   = pool->cap;
        a->owned = 0;
    } else {
        void *p = malloc(need);
        if (!p) { a->base = NULL; a->cap = 0; a->owned = 0; return -1; }
        a->base  = (unsigned char *)p;
        a->cap   = need;
        a->owned = 1;
    }
    return 0;
}

/* Carve `n` bytes, 16-byte aligned. Returns NULL only if the frame was under-reserved
 * (a caller bug) — callers size `need` to cover every push, then treat a NULL push as a
 * checked error so it degrades safely instead of overrunning. */
static inline void *rocket_arena_push(rocket_arena *a, size_t n)
{
    size_t o = rocket_arena_align_up(a->off, 16);
    if (o + n > a->cap) return NULL;
    a->off = o + n;
    return a->base + o;
}

/* Sum of aligned push sizes for a frame that will carve buffers of the given byte sizes,
 * so a caller can reserve exactly enough in one rocket_arena_open. */
static inline size_t rocket_arena_reserve(const size_t *sizes, int n)
{
    size_t t = 0;
    for (int i = 0; i < n; i++) t += rocket_arena_align_up(sizes[i], 16);
    return t;
}

/* Release the frame: free an owned block, or just reset a borrowed pool's cursor. Safe on
 * a zeroed arena (no-op), so a single `goto done; ... done: rocket_arena_close(&a);` covers
 * every error path, including ones taken before the frame was opened. */
static inline void rocket_arena_close(rocket_arena *a)
{
    if (a->owned) free(a->base);
    a->base = NULL;
    a->cap  = 0;
    a->off  = 0;
    a->owned = 0;
}

/* Free a context-owned pool at owner teardown. */
static inline void rocket_scratch_pool_free(rocket_scratch_pool *pool)
{
    free(pool->base);
    pool->base = NULL;
    pool->cap  = 0;
}

#endif /* ROCKET_SCRATCH_H */
