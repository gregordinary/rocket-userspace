// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_log.h — centralized logging for librocketnpu.
 *
 * Every diagnostic the library emits (errors, warnings, the ROCKET_MM_PROFILE
 * breakdown, the ROCKET_DEBUG traces) flows through one channel so a host
 * application (llama.cpp, a TFLite delegate, Frigate) can intercept, redirect,
 * or silence it instead of having raw fprintf(stderr) pollute its own output.
 *
 * Default behavior with no host hook installed: ERROR/WARN/INFO go to stderr,
 * DEBUG is suppressed — matching the historical "errors always print, traces
 * only under ROCKET_DEBUG" behavior. A host overrides this by installing a
 * callback (see rocket_log_set_callback), much like ggml_log_set_callback.
 *
 * Environment control (read once, on first use):
 *   ROCKET_LOG_LEVEL = error | warn | info | debug   (or 0..3)  — the threshold.
 *   ROCKET_DEBUG set                                  — raises the threshold to
 *                                                       at least DEBUG (back-compat).
 *   ROCKET_LOG_STDERR set (non-"0")                   — tee every emitted line to
 *                                                       stderr as well, but only when
 *                                                       a host callback is installed.
 *                                                       Escape hatch for a host that
 *                                                       silences its own logger (e.g.
 *                                                       llama-bench without -v).
 */
#ifndef ROCKET_LOG_H
#define ROCKET_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ROCKET_LOG_ERROR = 0,  /* a failure the caller must handle */
    ROCKET_LOG_WARN  = 1,  /* a recoverable anomaly / fallback taken */
    ROCKET_LOG_INFO  = 2,  /* requested informational output (e.g. ROCKET_MM_PROFILE) */
    ROCKET_LOG_DEBUG = 3,  /* developer trace, off by default */
} rocket_log_level;

/*
 * Sink for one already-formatted, newline-terminated-as-emitted message.
 * `text` is owned by the library and only valid for the duration of the call.
 * Install with rocket_log_set_callback; pass NULL to restore the stderr default.
 * The contract is set-once-before-first-use (no internal locking).
 */
typedef void (*rocket_log_callback)(rocket_log_level level, const char *text, void *user_data);

void             rocket_log_set_callback(rocket_log_callback cb, void *user_data);
void             rocket_log_set_level(rocket_log_level level);
rocket_log_level rocket_log_get_level(void);

/* printf-style entry point. Prefer the ROCKET_LOG* convenience macros below. */
void rocket_log(rocket_log_level level, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#define ROCKET_LOGE(...) rocket_log(ROCKET_LOG_ERROR, __VA_ARGS__)
#define ROCKET_LOGW(...) rocket_log(ROCKET_LOG_WARN,  __VA_ARGS__)
#define ROCKET_LOGI(...) rocket_log(ROCKET_LOG_INFO,  __VA_ARGS__)
#define ROCKET_LOGD(...) rocket_log(ROCKET_LOG_DEBUG, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* ROCKET_LOG_H */
