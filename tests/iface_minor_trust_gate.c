// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * iface_minor_trust_gate.c — host-only GATE: a private interface minor counts only on a
 * kernel that carries the patch series it names.
 *
 * Every capability probe keys on the DRM interface minor, and the minors 1.1 and up are
 * this library's patch series. Upstream numbers its own interface, so a minor it raises for
 * its own reasons would read as our job flags, and a flag word the kernel does not know
 * fails every submit. rocket_sysfs_trusted_minor() keeps a minor above 0 only where the
 * series' `rocket_batch_submit` module parameter exists. The boards here all carry the
 * series, so no device test ever takes the other branch. This one builds the sysfs trees
 * in a temporary directory and takes every branch:
 *
 *   series      module/rocket/parameters/rocket_batch_submit present: the minor stands
 *   stock       module/ visible, rocket loaded with no such parameter: the minor reads 0
 *   no rocket   module/ visible, no rocket entry at all (a built-in driver with no
 *               parameters has none): the minor reads 0
 *   no sysfs    no module/ directory: the reported minor stands
 *   pass-through 0 and -1 (stock, not rocket) are returned unchanged in every tree, and
 *               INT_MAX (a major above 1) obeys the same rule as a minor
 *
 * No NPU and no device file. Exit 0 or 1, never skips.
 */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rocket_sysfs.h"

static int g_fail, g_checks;

static void check(const char *what, int got, int want)
{
    g_checks++;
    if (got != want) {
        g_fail++;
        printf("FAIL  %-44s got %d want %d\n", what, got, want);
    }
}

static void mkdirs(const char *root, const char *rel)
{
    char p[512];
    snprintf(p, sizeof p, "%s/%s", root, rel);
    for (char *s = p + strlen(root) + 1; *s; s++)
        if (*s == '/') { *s = 0; mkdir(p, 0755); *s = '/'; }
    mkdir(p, 0755);
}

static void touch(const char *root, const char *rel)
{
    char p[512];
    snprintf(p, sizeof p, "%s/%s", root, rel);
    FILE *f = fopen(p, "w");
    if (f) { fputs("1\n", f); fclose(f); }
}

/* Each tree is checked at every reported value the rule distinguishes. */
static void tree(const char *name, const char *root, int keeps)
{
    static const int reported[] = { 1, 3, 6, INT_MAX };
    char what[96];
    for (unsigned i = 0; i < sizeof reported / sizeof reported[0]; i++) {
        snprintf(what, sizeof what, "%s, reported %d", name, reported[i]);
        check(what, rocket_sysfs_trusted_minor(reported[i], root), keeps ? reported[i] : 0);
    }
    snprintf(what, sizeof what, "%s, reported 0", name);
    check(what, rocket_sysfs_trusted_minor(0, root), 0);
    snprintf(what, sizeof what, "%s, reported -1", name);
    check(what, rocket_sysfs_trusted_minor(-1, root), -1);
}

int main(void)
{
    char base[] = "/tmp/iface_minor_trust_XXXXXX";
    if (!mkdtemp(base)) { perror("mkdtemp"); return 1; }
    char series[600], stock[600], norocket[600], nosysfs[600];
    snprintf(series, sizeof series, "%s/series", base);
    snprintf(stock, sizeof stock, "%s/stock", base);
    snprintf(norocket, sizeof norocket, "%s/norocket", base);
    snprintf(nosysfs, sizeof nosysfs, "%s/nosysfs", base);

    mkdirs(base, "series/module/rocket/parameters");
    touch(base, "series/module/rocket/parameters/rocket_batch_submit");
    touch(base, "series/module/rocket/parameters/rocket_npu_clk_hz");
    mkdirs(base, "stock/module/rocket/parameters");      /* another parameter, not ours */
    touch(base, "stock/module/rocket/parameters/some_upstream_knob");
    mkdirs(base, "norocket/module/drm/parameters");
    mkdirs(base, "nosysfs/bus");

    tree("series", series, 1);
    tree("stock", stock, 0);
    tree("no rocket entry", norocket, 0);
    tree("no sysfs", nosysfs, 1);

    char cmd[700];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", base);
    if (system(cmd) != 0) printf("note: could not remove %s\n", base);

    printf("%s: %d of %d checks passed\n", g_fail ? "FAIL" : "PASS", g_checks - g_fail, g_checks);
    return g_fail ? 1 : 0;
}
