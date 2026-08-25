// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_busy_poll.h — the completion-wait spin budget, shared across submit providers.
 *
 * `rocket_busy_poll_set_us()` and ROCKET_BUSY_POLL are a HOST-side knob: they say how
 * long a waiter may spin before it blocks, which is a property of the caller's latency
 * tolerance rather than of any one kernel driver. Every provider that offers a
 * non-blocking completion probe reads the same budget from here, so an in-process A/B
 * set by one caller means the same thing whichever driver is carrying the submits.
 */
#ifndef ROCKET_BUSY_POLL_H
#define ROCKET_BUSY_POLL_H

/* Microseconds a completion wait may spin before blocking. Resolved once from
 * ROCKET_BUSY_POLL unless rocket_busy_poll_set_us() got there first; 0 = never spin. */
long rkt_busy_poll_us(void);

#endif /* ROCKET_BUSY_POLL_H */
