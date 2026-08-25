// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_sysfs.c — the one enumeration of the bound NPU cores.
 *
 * This is sysfs enumeration, not an ioctl, so it sits on the CORE side of the submit
 * seam rather than inside a provider. rocket_hw_profile.c calls it to find the bound
 * device's `compatible` string, which means chip detection would otherwise carry a
 * link-time dependency on whichever submit provider is compiled in — and a provider
 * that drives a different kernel driver has no reason to own the answer to "which
 * cores has the mainline rocket driver bound".
 *
 * The contract, and why the power/ subdirectory is the test, are in rocket_sysfs.h.
 */
#include <dirent.h>
#include <stdatomic.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "rocket_sysfs.h"

int rocket_sysfs_bound_devices(char names[][ROCKET_SYSFS_NAME_MAX], int max)
{
    static const char *dir = "/sys/bus/platform/drivers/rocket";
    DIR *d = opendir(dir);
    struct dirent *e;
    int n = 0;

    if (!d) return 0;
    while ((e = readdir(d)) && n < max) {
        char path[512];
        struct stat st;
        if (e->d_name[0] == '.') continue;
        /* The directory also holds bind/unbind/uevent and a `module` link; a bound
         * device is whatever carries a power/ subdirectory. */
        if (snprintf(path, sizeof path, "%s/%s/power", dir, e->d_name) >= (int)sizeof path)
            continue;
        if (stat(path, &st) != 0 || !S_ISDIR(st.st_mode)) continue;
        if (names) {
            /* Truncate rather than skip: a name this long is not a device we can use
             * either way, and a truncated one simply fails to open. */
            snprintf(names[n], ROCKET_SYSFS_NAME_MAX, "%.*s",
                     ROCKET_SYSFS_NAME_MAX - 1, e->d_name);
        }
        n++;
    }
    closedir(d);
    return n;
}

int rocket_sysfs_bound_core_count(void)
{
    /* Cached: the bind set does not change under a running process that is using the
     * device, and the RK3576 fd guard asks on every open. _Atomic because that guard
     * can be reached from several threads at once; the probe is idempotent, so a race
     * costs a second enumeration and nothing else. */
    static _Atomic int cached = -1;
    int c = atomic_load_explicit(&cached, memory_order_relaxed);
    if (c < 0) {
        c = rocket_sysfs_bound_devices(NULL, ROCKET_SYSFS_MAX_DEVS);
        atomic_store_explicit(&cached, c, memory_order_relaxed);
    }
    return c;
}
