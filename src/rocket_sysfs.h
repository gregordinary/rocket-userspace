// SPDX-License-Identifier: GPL-3.0-or-later
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

#endif /* ROCKET_SYSFS_H */
