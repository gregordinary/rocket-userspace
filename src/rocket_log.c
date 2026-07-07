// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_log.c — the one diagnostic channel for librocketnpu. See rocket_log.h.
 *
 * A formatted message is dropped early if its level is above the active
 * threshold, then handed to the installed sink (the stderr default unless a host
 * has called rocket_log_set_callback). Formatting uses a stack buffer with a heap
 * fallback for the rare long line, so a log call never allocates on the hot path.
 */
#include "rocket_log.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static rocket_log_callback g_cb       = NULL;
static void               *g_cb_user  = NULL;
static _Atomic int         g_level    = -1; /* <0 = not yet resolved from the env */
static _Atomic int         g_tee      = -1; /* ROCKET_LOG_STDERR: <0 = unresolved */

/*
 * ROCKET_LOG_STDERR tees each emitted line to stderr *in addition to* the
 * installed callback. It exists for the case a host silences its own logger:
 * llama-bench installs a no-op ggml callback (unless -v), which — because our
 * driver channel forwards into ggml — swallows every rocket diagnostic,
 * including the resident-budget decision that changes the benchmarked number.
 * The tee fires ONLY when a callback is installed (g_cb != NULL); with the
 * default stderr sink there is nothing to tee, so no double-print.
 */
static int stderr_tee_enabled(void)
{
    if (g_tee < 0) {
        const char *s = getenv("ROCKET_LOG_STDERR");
        g_tee = (s && *s && strcmp(s, "0")) ? 1 : 0;
    }
    return g_tee;
}

static rocket_log_level resolve_level_from_env(void)
{
    const char *s = getenv("ROCKET_LOG_LEVEL");
    rocket_log_level lvl = ROCKET_LOG_INFO; /* default: errors + warnings + info */

    if (s && *s) {
        if      (!strcmp(s, "error") || !strcmp(s, "0")) lvl = ROCKET_LOG_ERROR;
        else if (!strcmp(s, "warn")  || !strcmp(s, "1")) lvl = ROCKET_LOG_WARN;
        else if (!strcmp(s, "info")  || !strcmp(s, "2")) lvl = ROCKET_LOG_INFO;
        else if (!strcmp(s, "debug") || !strcmp(s, "3")) lvl = ROCKET_LOG_DEBUG;
    }
    /* ROCKET_DEBUG raises the floor to DEBUG, preserving the historical knob. */
    if (getenv("ROCKET_DEBUG") && lvl < ROCKET_LOG_DEBUG)
        lvl = ROCKET_LOG_DEBUG;

    return lvl;
}

static int active_level(void)
{
    if (g_level < 0)
        g_level = (int)resolve_level_from_env();
    return g_level;
}

void rocket_log_set_callback(rocket_log_callback cb, void *user_data)
{
    g_cb      = cb;
    g_cb_user = user_data;
}

void rocket_log_set_level(rocket_log_level level)
{
    g_level = (int)level;
}

rocket_log_level rocket_log_get_level(void)
{
    return (rocket_log_level)active_level();
}

static void default_sink(rocket_log_level level, const char *text, void *user_data)
{
    (void)level;
    (void)user_data;
    fputs(text, stderr);
}

void rocket_log(rocket_log_level level, const char *fmt, ...)
{
    if ((int)level > active_level())
        return;

    char  buf[2048];
    char *msg = buf;

    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    if (n < 0)
        return;

    /* Rare: the message did not fit the stack buffer — format it on the heap. */
    if ((size_t)n >= sizeof(buf)) {
        char *big = (char *)malloc((size_t)n + 1);
        if (big) {
            va_start(ap, fmt);
            vsnprintf(big, (size_t)n + 1, fmt, ap);
            va_end(ap);
            msg = big;
        }
        /* malloc failure: fall back to the truncated stack buffer. */
    }

    if (g_cb) {
        g_cb(level, msg, g_cb_user);
        /* Host callback may silence output (e.g. llama-bench's no-op logger);
         * tee to stderr when the operator asked to see the channel regardless. */
        if (stderr_tee_enabled())
            fputs(msg, stderr);
    } else {
        default_sink(level, msg, g_cb_user);
    }

    if (msg != buf)
        free(msg);
}
