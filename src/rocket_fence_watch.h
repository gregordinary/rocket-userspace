// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_fence_watch.h — how long each fenced wait took, shared across submit providers.
 *
 * The mainline `rocket` driver retires a job that runs 500 ms (JOB_TIMEOUT_MS), resets
 * the core and signals the job's fence. PREP_BO then returns 0 exactly as it does for a
 * job that completed, so from userspace a hung job reads as a slow one whose output is
 * whatever the BO held before. The duration is the only trace the wait leaves, so the
 * provider reports every fenced wait here and the counters in rocket_npu.h read it back.
 *
 * It sits on the core side of the submit seam for the same reason as rocket_busy_poll.h:
 * the bookkeeping is host-side and identical for every provider, and a provider that
 * does not call it only leaves the counters at zero rather than failing to link.
 */
#ifndef ROCKET_FENCE_WATCH_H
#define ROCKET_FENCE_WATCH_H

#include <stdint.h>

/* One fenced wait on `handle` took `ns`. Counts it, and logs a warning when it reached
 * the slow-wait mark (ROCKET_SLOW_WAIT_MS, default 450). */
void rkt_fence_wait_note(uint64_t ns, uint32_t handle);

#endif /* ROCKET_FENCE_WATCH_H */
