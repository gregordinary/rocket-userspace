// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_sysfs.h — which NPU cores the rocket driver has bound, asked once.
 *
 * Three places needed the answer and each invented its own test for "which entries of
 * /sys/bus/platform/drivers/rocket are devices": one looked for a power/ subdirectory,
 * one for a readable power/autosuspend_delay_ms, and one skipped a hardcoded list of
 * bind/unbind/uevent/module. Three answers to one question, and the question is
 * load-bearing -- the RK3576 dual-core corruption guard refuses a second fd based on it.
 *
 * The power/ subdirectory is the test kept: every bound platform device has one, the
 * driver's own control files do not, and it does not depend on knowing the names of
 * files a future kernel might add to that directory.
 */
#ifndef ROCKET_SYSFS_H
#define ROCKET_SYSFS_H

#include <stddef.h>

#define ROCKET_SYSFS_MAX_DEVS 8
#define ROCKET_SYSFS_NAME_MAX 64

/* Fill `names` with the bound devices' directory names and return how many there were.
 * Returns 0 when the driver is absent, unbound, or sysfs is not mounted -- a count, not
 * an error, because "no cores are bound" is a real state a caller must handle anyway.
 * A count above `max` is truncated to `max`; the return value is what was STORED. */
int rocket_sysfs_bound_devices(char names[][ROCKET_SYSFS_NAME_MAX], int max);

/* How many NPU cores the driver has bound. */
int rocket_sysfs_bound_core_count(void);

/* The DRM interface minor the capability probes may act on, given the one the kernel
 * `reported` and the sysfs root (`"/sys"`; a test passes a fake tree).
 *
 * The minors 1.1 and up name this library's kernel patch series: each private job flag,
 * the ranged cache sync and the PPU completion appear at a minor of that series, and both
 * the RK3588 and the RK3576 series take 1.1 from the batched-submit patch, which also
 * creates the `rocket_batch_submit` module parameter. Upstream numbers its own interface,
 * and a minor it raises for its own reasons names none of those features: sending a flag
 * word to a kernel that does not know it fails every submit with -EINVAL. So a minor
 * above 0 counts only where `<sys_root>/module/rocket/parameters/rocket_batch_submit`
 * exists, and reads as 0 where the module directory is visible and the parameter is not.
 * Where `<sys_root>/module` itself is absent (sysfs not mounted, as in some containers)
 * there is nothing to ask, and the reported minor stands.
 *
 * A `reported` of 0 or below (stock, or not rocket) is returned unchanged. */
int rocket_sysfs_trusted_minor(int reported, const char *sys_root);

#endif /* ROCKET_SYSFS_H */
