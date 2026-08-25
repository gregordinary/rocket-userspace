// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_busy_poll.c — the completion-wait spin budget. See rocket_busy_poll.h for why
 * this sits on the core side of the submit seam rather than inside a provider.
 */
#include <stdatomic.h>
#include <stdlib.h>

#include "rocket_busy_poll.h"
#include "rocket_npu.h"

static _Atomic long g_busy_poll_us = -1;   /* -1 = unresolved; resolve lazily from env (also set by rocket_busy_poll_set_us) */

void rocket_busy_poll_set_us(long us)
{
    g_busy_poll_us = us < 0 ? 0 : us;
}

long rkt_busy_poll_us(void)
{
    if (g_busy_poll_us < 0) {
        const char *e = getenv("ROCKET_BUSY_POLL");
        long v = (e && *e) ? strtol(e, NULL, 10) : 0;
        g_busy_poll_us = v < 0 ? 0 : v;
    }
    return g_busy_poll_us;
}
