// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_fence_watch.c — the fenced-wait counters. See rocket_fence_watch.h for why the
 * duration of a wait is worth keeping, and rocket_npu.h for how a caller reads it.
 */
#include <stdatomic.h>
#include <stdlib.h>

#include "rocket_fence_watch.h"
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "rocket_npu.h"

static _Atomic uint64_t g_waits;
static _Atomic uint64_t g_slow_waits;
static _Atomic uint64_t g_max_wait_ns;
static _Atomic long     g_slow_ms = -1;   /* -1 = unresolved; resolved once from the env */

/* The mark sits under the driver's retirement of a job that never completes, which is a
 * property of the part's driver: 450 ms under the RK3588's 500 ms watchdog, 110 ms under
 * the RK3576 series' 125 ms backstop (the profile's slow_wait_ms), leaving room for the time
 * a caller spends between its submit and its wait. No single healthy job outlasts the
 * retirement, so a wait past the mark is a retired job, or a wait that also covered other
 * jobs queued on the same core ahead of it. The kernel log tells the two apart; this
 * counter cannot. And a caller that waited late can see a retired job end under the mark,
 * so zero here is not proof that nothing timed out. */
static long slow_mark_ms(void)
{
    long v = atomic_load_explicit(&g_slow_ms, memory_order_relaxed);
    if (v < 0) {
        const long dflt = rocket_hw_current()->slow_wait_ms > 0
                        ? rocket_hw_current()->slow_wait_ms : 450;
        const char *e = getenv("ROCKET_SLOW_WAIT_MS");
        v = (e && *e) ? strtol(e, NULL, 10) : dflt;
        if (v <= 0) v = dflt;
        atomic_store_explicit(&g_slow_ms, v, memory_order_relaxed);
    }
    return v;
}

void rkt_fence_wait_note(uint64_t ns, uint32_t handle)
{
    atomic_fetch_add_explicit(&g_waits, 1, memory_order_relaxed);

    uint64_t prev = atomic_load_explicit(&g_max_wait_ns, memory_order_relaxed);
    while (ns > prev &&
           !atomic_compare_exchange_weak_explicit(&g_max_wait_ns, &prev, ns,
                                                  memory_order_relaxed, memory_order_relaxed))
        ;

    long mark = slow_mark_ms();
    if (ns < (uint64_t)mark * 1000000ull)
        return;
    atomic_fetch_add_explicit(&g_slow_waits, 1, memory_order_relaxed);
    /* The wording is matched by the test registration (FAIL_REGULAR_EXPRESSION in
     * CMakeLists.txt), so a change here must change it there too. */
    ROCKET_LOGW("rocket: a fenced wait on BO %u took %.1f ms, past the %ld ms slow-wait "
                "mark. The kernel retires a job that outlasts its watchdog and signals its "
                "fence as if it had completed, so this output may be unwritten.\n",
                handle, (double)ns / 1e6, mark);
}

uint64_t rocket_fence_wait_count(void)
{
    return atomic_load_explicit(&g_waits, memory_order_relaxed);
}

uint64_t rocket_fence_wait_slow_count(void)
{
    return atomic_load_explicit(&g_slow_waits, memory_order_relaxed);
}

uint64_t rocket_fence_wait_max_us(void)
{
    return atomic_load_explicit(&g_max_wait_ns, memory_order_relaxed) / 1000u;
}

void rocket_fence_wait_counters_reset(void)
{
    atomic_store_explicit(&g_waits, 0, memory_order_relaxed);
    atomic_store_explicit(&g_slow_waits, 0, memory_order_relaxed);
    atomic_store_explicit(&g_max_wait_ns, 0, memory_order_relaxed);
}
