// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_fanout.c — the shared worker-pool / per-shape-scratch machinery. See
 * rocket_fanout.h for what it is and why it is one file rather than four copies.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_fanout.h"
#include "rocket_log.h"

long rocket_fanout_wait_ns(void)
{
    static _Atomic long ns = -1;
    long v = atomic_load_explicit(&ns, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("ROCKET_WAIT_MS");
        long ms = e ? atol(e) : 8000;
        if (ms < 1) ms = 8000;
        v = ms * 1000000L;
        atomic_store_explicit(&ns, v, memory_order_relaxed);
    }
    return v;
}

int rocket_fanout_shared_pack(void)
{
    static _Atomic int on = -1;
    int v = atomic_load_explicit(&on, memory_order_relaxed);
    if (v < 0) {
        v = getenv("ROCKET_NO_SHARED_PACK") == NULL;
        atomic_store_explicit(&on, v, memory_order_relaxed);
    }
    return v;
}

int rocket_fanout_nstep(int N, int nthreads, int align)
{
    if (nthreads < 1) nthreads = 1;
    if (align < 1) align = 1;
    int s = ((N + nthreads - 1) / nthreads + align - 1) / align * align;
    return s < align ? align : s;
}

/* ============================================================================
 * SECTION — The per-shape scratch cache
 * ==========================================================================*/

static int key_eq(const rocket_shape_key *a, const rocket_shape_key *b)
{
    return a->M == b->M && a->K == b->K && a->N == b->N && a->group == b->group;
}

void *rocket_slot_get(rocket_slot_cache *c, void *owner,
                      const rocket_shape_key *k, const rocket_slot_ops *ops)
{
    for (int i = 0; i < c->n; i++)
        if (c->slot[i] && key_eq(&c->key[i], k)) {
            c->stamp[i] = ++c->clock;
            return c->slot[i];
        }

    int at = c->n;
    if (at >= ROCKET_FANOUT_MAX_SLOTS) {
        /* Full: recycle the least recently used slot. Freeing BEFORE the alloc is
         * deliberate — this cache exists to conserve each fd's 4 GB IOVA window, and
         * holding the old scratch across the new allocation is exactly the peak the
         * window cannot afford on a large resident weight. */
        at = 0;
        for (int i = 1; i < c->n; i++)
            if (c->stamp[i] < c->stamp[at]) at = i;
        ROCKET_LOGD("scratch cache full (%d shapes): recycling %d/%d/%d group=%d "
                    "for %d/%d/%d group=%d\n", ROCKET_FANOUT_MAX_SLOTS,
                    c->key[at].M, c->key[at].K, c->key[at].N, c->key[at].group,
                    k->M, k->K, k->N, k->group);
        ops->release(owner, c->slot[at]);
        c->slot[at] = NULL;
    }

    void *s = ops->alloc(owner, k);
    if (!s) {
        /* A failed alloc into a recycled slot leaves a hole; compact it away so the
         * cache does not carry a NULL that every later lookup has to skip. */
        if (at < c->n) { c->slot[at] = c->slot[--c->n]; c->key[at] = c->key[c->n];
                         c->stamp[at] = c->stamp[c->n]; }
        return NULL;
    }
    c->slot[at]  = s;
    c->key[at]   = *k;
    c->stamp[at] = ++c->clock;
    if (at == c->n) c->n++;
    return s;
}

void rocket_slot_cache_clear(rocket_slot_cache *c, void *owner, const rocket_slot_ops *ops)
{
    for (int i = 0; i < c->n; i++) {
        ops->release(owner, c->slot[i]);
        c->slot[i] = NULL;
    }
    c->n = 0;
}

/* ============================================================================
 * SECTION — The worker fan-out
 * ==========================================================================*/

void rocket_fanout_run(int n, void *args, size_t stride, void *(*fn)(void *))
{
    if (n < 1) return;
    if (n > ROCKET_FANOUT_MAX_WORKERS) n = ROCKET_FANOUT_MAX_WORKERS;

    pthread_t th[ROCKET_FANOUT_MAX_WORKERS];
    int joinable[ROCKET_FANOUT_MAX_WORKERS] = {0};
    char *base = (char *)args;

    for (int t = 0; t < n; t++)
        if (pthread_create(&th[t], NULL, fn, base + (size_t)t * stride) == 0)
            joinable[t] = 1;

    /* Spawn failures run here, AFTER the loop, so one of them cannot serialize the
     * workers behind it onto its own NPU wait. */
    for (int t = 0; t < n; t++)
        if (!joinable[t]) fn(base + (size_t)t * stride);

    for (int t = 0; t < n; t++)
        if (joinable[t]) pthread_join(th[t], NULL);
}
