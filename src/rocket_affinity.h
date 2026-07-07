// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_affinity.h — CPU pinning policy for the matmul fan-out workers.
 *
 * The mt/streaming paths spawn worker pthreads that do the dominant fp16 pack
 * (A/B scatter) and readback — the ~67% CPU slice in the on-hardware profile.
 * RK3588 is heterogeneous (4×A76 "big" + 4×A55 "little"); if the OS parks a
 * worker on an A55, the pthread_join waits on the slow core and drags the whole
 * matmul. Pinning the heavy workers to the A76s removes that tail.
 *
 * NOT a public header (lives in src/, not installed).
 */
#ifndef ROCKET_AFFINITY_H
#define ROCKET_AFFINITY_H

/* Pin the CALLING thread (a matmul worker) to a "big" CPU. `worker_idx` selects
 * which big core via round-robin (worker_idx % n_big), so the N workers spread
 * across the big cluster instead of stacking on one core.
 *
 * The big-core set is, in order of precedence:
 *   1. ROCKET_CPU_AFFINITY, a CPU list ("4-7", "4,5,6,7", "0,4-7"). The special
 *      values "off"/"none"/"0"/"" disable pinning entirely.
 *   2. else auto-detected as the CPUs whose cpufreq cpuinfo_max_freq equals the
 *      global max (the A76s on RK3588). Generic across big.LITTLE; no hardcoded
 *      core numbers.
 *
 * No-op (leaves affinity untouched) when pinning is disabled, when no distinct
 * big cluster is found (all CPUs same max freq, or cpufreq not exposed), or when
 * pthread_setaffinity_np fails. Detection runs once (pthread_once); thread-safe.
 * Logs the detected set and each pin under ROCKET_DEBUG. */
void rocket_pin_worker(int worker_idx);

/* Same as rocket_pin_worker but with an explicit rotation BASE: pins to
 * g_big[(core_base + worker_idx) % n_big]. rocket_pin_worker(idx) ==
 * rocket_pin_worker_based(idx, <calling thread's affinity base>). Worker threads
 * carry the base from their spawning context (see rocket_affinity_set_base in
 * rocket_npu.h) so N in-process context pools spread across the big cluster
 * instead of every pool's worker 0 stacking on the same core. */
void rocket_pin_worker_based(int worker_idx, int core_base);

#endif /* ROCKET_AFFINITY_H */
