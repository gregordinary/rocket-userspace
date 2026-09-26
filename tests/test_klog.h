// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * test_klog.h — read the kernel log from a mark, for a gate whose verdict is in it.
 *
 * Some defects fail no syscall: a use-after-free shows as a WARN backtrace, and a job
 * the kernel's watchdog retired signals its fence like one that completed. A gate for
 * those reads the kernel log, and the read is part of the instrument. Counting `dmesg`
 * lines is not a safe mark. Under dmesg_restrict an unprivileged read comes back empty
 * and the gate passes on nothing, and once the ring buffer is full it rolls, so the line
 * count stops moving while new lines still arrive.
 *
 * So the mark here is a journal CURSOR: the position of the last kernel entry when the
 * mark was taken, which later reads resume after exactly. An empty read is not a clean
 * one. Any running system has kernel entries, so a mark that finds no cursor means the
 * reader has no access, and the caller must skip rather than pass. Access is tried as
 * the caller first, which works for the `adm` and `systemd-journal` groups, then through
 * `sudo -n`, which never prompts.
 */
#ifndef ROCKET_TESTS_TEST_KLOG_H
#define ROCKET_TESTS_TEST_KLOG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char cursor[512];
    const char *via;          /* "" or "sudo -n ": the prefix the mark read through */
} tk_mark;

/* Read the cursor of the last kernel journal entry through `via`. 0 if one came back. */
static inline int tk_read_cursor(const char *via, char *cursor, size_t cap)
{
    char cmd[256], line[1024];
    FILE *f;
    int found = 0;

    snprintf(cmd, sizeof cmd,
             "%sjournalctl -k -q --no-pager -n 1 --show-cursor -o cat 2>/dev/null", via);
    f = popen(cmd, "r");
    if (!f) return -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "-- cursor: ", 11) == 0) {
            size_t n = strcspn(line + 11, "\r\n");
            if (n > 0 && n < cap && !strchr(line + 11, '\'')) {
                memcpy(cursor, line + 11, n);
                cursor[n] = 0;
                found = 1;
            }
        }
    }
    pclose(f);
    return found ? 0 : -1;
}

/* Take a mark. 0 on success, -1 if this process cannot read the kernel journal. */
static inline int tk_mark_take(tk_mark *m)
{
    static const char *const vias[] = { "", "sudo -n " };
    for (size_t i = 0; i < sizeof vias / sizeof vias[0]; i++) {
        if (tk_read_cursor(vias[i], m->cursor, sizeof m->cursor) == 0) {
            m->via = vias[i];
            return 0;
        }
    }
    m->cursor[0] = 0;
    m->via = NULL;
    return -1;
}

/* Scan every kernel journal line after the mark. hits[i] counts the lines containing
 * sigs[i], and up to `keep` matching lines are copied into `sample`. Returns 0, or -1
 * if the log could not be read back, which a gate must treat as a skip. */
static inline int tk_scan(const tk_mark *m, const char *const *sigs, int nsig, long *hits,
                          char (*sample)[200], int keep, int *nsample)
{
    char cmd[768], line[1024];
    FILE *f;
    int i, rc;

    if (!m->via || !m->cursor[0]) return -1;
    snprintf(cmd, sizeof cmd,
             "%sjournalctl -k -q --no-pager -o short-monotonic --after-cursor='%s' "
             "2>/dev/null", m->via, m->cursor);
    f = popen(cmd, "r");
    if (!f) return -1;
    for (i = 0; i < nsig; i++) hits[i] = 0;
    *nsample = 0;
    while (fgets(line, sizeof line, f)) {
        int matched = 0;
        for (i = 0; i < nsig; i++)
            if (strstr(line, sigs[i])) { hits[i]++; matched = 1; }
        if (matched && *nsample < keep) {
            size_t n = strcspn(line, "\n");
            if (n >= sizeof sample[0]) n = sizeof sample[0] - 1;
            memcpy(sample[*nsample], line, n);
            sample[*nsample][n] = 0;
            (*nsample)++;
        }
    }
    rc = pclose(f);
    return rc == 0 ? 0 : -1;
}

/* Print the kernel journal lines after the mark that match `egrep` (grep -iE), at most
 * `max` of them. For a probe whose report is the log. Returns -1 if unreadable. */
static inline int tk_print_since(const tk_mark *m, const char *egrep, int max)
{
    char cmd[1024];
    if (!m->via || !m->cursor[0]) return -1;
    snprintf(cmd, sizeof cmd,
             "%sjournalctl -k -q --no-pager -o short-monotonic --after-cursor='%s' "
             "2>/dev/null | grep -iE '%s' | head -%d", m->via, m->cursor, egrep, max);
    fflush(stdout);
    if (system(cmd) != 0) { /* grep found nothing, which is a result */ }
    return 0;
}

#endif /* ROCKET_TESTS_TEST_KLOG_H */
