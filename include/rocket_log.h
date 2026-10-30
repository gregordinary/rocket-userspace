// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * Declares the centralized logging of librocketnpu. Every diagnostic the library emits
 * flows through one channel: errors, warnings, the ROCKET_MM_PROFILE breakdown and the
 * ROCKET_DEBUG traces. A host application (llama.cpp, a TFLite delegate, Frigate) can then
 * intercept, redirect or silence them, instead of having raw fprintf(stderr) output mixed
 * into its own.
 *
 * With no host hook installed, ERROR, WARN and INFO go to stderr, and DEBUG is suppressed.
 * Errors always print, and traces print only under ROCKET_DEBUG. A host overrides this by
 * installing a callback with `rocket_log_set_callback`, much like ggml_log_set_callback.
 *
 * The library reads these environment variables once, on first use:
 *
 * - `ROCKET_LOG_LEVEL`: the threshold, one of error, warn, info or debug (or 0..3).
 * - `ROCKET_DEBUG`: any value raises the threshold to at least DEBUG, for backward
 *   compatibility.
 * - `ROCKET_LOG_STDERR`: a value other than "0" tees every emitted line to stderr as well,
 *   but only while a host callback is installed. It serves a host that silences its own
 *   logger, such as llama-bench without -v.
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
 * Receives one already-formatted message, newline-terminated as emitted. `text` belongs to
 * the library and is valid only for the duration of the call. Install the callback with
 * `rocket_log_set_callback`, and pass NULL to restore the stderr default.
 *
 * An install is safe while other threads log. The callback and `user_data` are published
 * together, so a concurrent message reaches either the old pair or the new one, never a
 * mix. A message that loaded the old pair before the call can still reach it afterwards.
 * The host must therefore keep the old callback and its `user_data` valid. Each install
 * of a non-NULL callback allocates a small record that the library never frees. If that
 * allocation fails, the current callback stays installed.
 */
typedef void (*rocket_log_callback)(rocket_log_level level, const char *text, void *user_data);

void             rocket_log_set_callback(rocket_log_callback cb, void *user_data);
void             rocket_log_set_level(rocket_log_level level);
rocket_log_level rocket_log_get_level(void);

/* Logs a printf-style message at `level`. Prefer the ROCKET_LOG* convenience macros below. */
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
