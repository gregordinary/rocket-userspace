// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * uapi_bo_ranges_rocket.c — the ranged cache-maintenance ioctls: what they
 * accept, what they refuse, and that a refusal comes back to userspace.
 *
 * DRM_ROCKET_PREP_BO_RANGES / _FINI_BO_RANGES (interface 1.5) sync named byte
 * ranges of a BO instead of the whole object. They are a new UNPRIVILEGED entry
 * point — /dev/accel/accel0 is group `render` — and they take a user pointer, a
 * count and offsets the kernel must bound-check, which is the shape of every
 * client-reachable defect this driver has had. So the refusals are gated, not
 * just the happy path.
 *
 * Each case runs in a FORKED CHILD, so a kernel that dies on one still leaves a
 * parent to say which. The probe stops at the first child the kernel kills.
 *
 * A TEST THAT DRIVES REJECTION PATHS CAN BE CORRUPTING THE KERNEL WHILE PASSING:
 * every case here returns to userspace on a stock kernel too, because an unknown
 * ioctl is refused before it reaches any of this code. Run it at least once under
 * `slub_debug=FZPU slab_nomerge` and read the dmesg, not the exit code.
 *
 * The DATA half — that a ranged sync actually makes the CPU see the NPU's writes
 * — is not gated here. It is gated by the four networks in rk3576_net_gate, whose
 * write guard runs on ranges under ROCKET_RK3576_GUARD_NARROW=1 and which are
 * bit-exact per layer; a sync that missed bytes would show up there as a surface
 * that reads as unwritten.
 *
 * GATE (registered in CTest): exit 0 = every case behaved, 1 = a case failed or
 * the kernel killed a child, 2 = no NPU device or a kernel without the ioctls.
 */
#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/wait.h>

#include <libdrm/drm.h>
#include <drm/rocket_accel.h>

#include "rocket_npu.h"

#ifndef DRM_ROCKET_PREP_BO_RANGES
#define DRM_ROCKET_PREP_BO_RANGES  0x04
#define DRM_ROCKET_FINI_BO_RANGES  0x05
struct drm_rocket_bo_range {
    __u64 offset;
    __u64 size;
};
struct drm_rocket_prep_bo_ranges {
    __u32 handle;
    __u32 range_count;
    __u64 ranges;
    __s64 timeout_ns;
    __u64 reserved;
};
struct drm_rocket_fini_bo_ranges {
    __u32 handle;
    __u32 range_count;
    __u64 ranges;
    __u64 reserved;
};
#define DRM_IOCTL_ROCKET_PREP_BO_RANGES \
    DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO_RANGES, struct drm_rocket_prep_bo_ranges)
#define DRM_IOCTL_ROCKET_FINI_BO_RANGES \
    DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO_RANGES, struct drm_rocket_fini_bo_ranges)
#endif

#define BO_BYTES (256u << 10)

/* What the kernel is expected to do with a case: accept it, or refuse it with
 * this errno. A refusal that comes back with the WRONG errno is still a pass on
 * the crash contract and a failure on the interface contract, so both are
 * reported separately. */
struct rcase {
    const char *name;
    const char *why;
    int         want_err;      /* 0 = must succeed */
    unsigned    count;
    int         bad_handle;
    int         null_ranges;
    uint64_t    reserved;
    unsigned    ndesc;         /* how many of r[] are literal; the rest is a ladder */
    struct drm_rocket_bo_range r[4];
};

static int run_case(int fd, uint32_t handle, size_t bo_size, const struct rcase *c,
                    int fini, int *out_err)
{
    struct drm_rocket_bo_range *r = NULL;
    int rc;

    (void)bo_size;
    if (c->count && !c->null_ranges) {
        r = calloc(c->count ? c->count : 1, sizeof *r);
        if (!r) return -1;
        /* Only the described ones are literal; a large count is filled out with a
         * legal ascending ladder so the COUNT is what is under test. Keyed on
         * ndesc rather than on a non-zero size, because a zero-size range is one
         * of the malformations. */
        for (unsigned i = 0; i < c->count; i++) {
            if (i < c->ndesc) {
                r[i] = c->r[i];
            } else {
                r[i].offset = (uint64_t)i * 128u;
                r[i].size   = 64u;
            }
        }
    }

    errno = 0;
    if (fini) {
        struct drm_rocket_fini_bo_ranges a = {
            .handle      = c->bad_handle ? 0x7fffffffu : handle,
            .range_count = c->count,
            .ranges      = (uint64_t)(uintptr_t)r,
            .reserved    = c->reserved,
        };
        rc = ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO_RANGES, &a);
    } else {
        struct drm_rocket_prep_bo_ranges a = {
            .handle      = c->bad_handle ? 0x7fffffffu : handle,
            .range_count = c->count,
            .ranges      = (uint64_t)(uintptr_t)r,
            .timeout_ns  = 0,
            .reserved    = c->reserved,
        };
        rc = ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO_RANGES, &a);
    }
    *out_err = rc < 0 ? errno : 0;
    free(r);
    return rc;
}

int main(void)
{
    static const struct rcase cases[] = {
        { "whole object", "range_count 0 is the plain PREP_BO/FINI_BO", 0, 0, 0, 0, 0, 0,
          {{0}} },
        { "one range", "the ordinary case", 0, 1, 0, 0, 0, 1,
          {{ 0, 64 }} },
        { "one range, tail", "a range ending exactly at the BO's end", 0, 1, 0, 0, 0, 1,
          {{ BO_BYTES - 64, 64 }} },
        { "one range, whole", "a range covering the whole object", 0, 1, 0, 0, 0, 1,
          {{ 0, BO_BYTES }} },
        { "four ranges", "ascending and disjoint", 0, 4, 0, 0, 0, 4,
          {{ 0, 64 }, { 4096, 64 }, { 65536, 128 }, { 131072, 64 }} },
        { "adjacent ranges", "touching but not overlapping", 0, 2, 0, 0, 0, 2,
          {{ 0, 64 }, { 64, 64 }} },
        { "many ranges", "the cap itself", 0, 1024, 0, 0, 0, 0, {{0}} },

        { "zero-size range", "an empty range is a caller bug, not a no-op", EINVAL, 1, 0, 0, 0, 1,
          {{ 0, 0 }} },
        { "past the end", "offset + size beyond the BO", EINVAL, 1, 0, 0, 0, 1,
          {{ BO_BYTES - 32, 64 }} },
        { "offset past the end", "wholly outside the BO", EINVAL, 1, 0, 0, 0, 1,
          {{ BO_BYTES + 4096, 64 }} },
        { "size overflow", "offset + size wraps u64", EINVAL, 1, 0, 0, 0, 1,
          {{ 4096, ~0ull - 1024 }} },
        { "descending", "the second range starts before the first", EINVAL, 2, 0, 0, 0, 2,
          {{ 4096, 64 }, { 0, 64 }} },
        { "overlapping", "the second range starts inside the first", EINVAL, 2, 0, 0, 0, 2,
          {{ 0, 128 }, { 64, 64 }} },
        { "over the cap", "more ranges than the kernel will copy", EINVAL, 1025, 0, 0, 0, 0,
          {{0}} },
        { "bad handle", "no such GEM object", ENOENT, 1, 1, 0, 0, 1,
          {{ 0, 64 }} },
        { "reserved set", "the reserved field must be zero", EINVAL, 1, 0, 0, 1, 1,
          {{ 0, 64 }} },
        { "null ranges pointer", "a count with no array", EFAULT, 1, 0, 1, 0, 0,
          {{0}} },
    };
    const int n = (int)(sizeof cases / sizeof cases[0]);
    int fd, failures = 0, killed = 0;
    rocket_bo guard, bo;

    fd = rocket_open();
    if (fd < 0) {
        printf("uapi_bo_ranges_rocket: no NPU device — skipping\n");
        return 2;
    }
    if (!rocket_bo_ranges_supported()) {
        printf("uapi_bo_ranges_rocket: kernel interface < 1.5, no ranged ioctls — skipping\n");
        rocket_close(fd);
        return 2;
    }
    /* A guard BO first: IOVA 0 is a real address on this part. */
    if (rocket_bo_alloc(fd, 4096, &guard) != 0 ||
        rocket_bo_alloc(fd, BO_BYTES, &bo) != 0) {
        fprintf(stderr, "uapi_bo_ranges_rocket: BO alloc failed\n");
        return 1;
    }

    printf("uapi_bo_ranges_rocket: %d case(s) x {PREP, FINI} on a %u KiB BO\n",
           n, BO_BYTES >> 10);

    for (int i = 0; i < n && !killed; i++) {
        const struct rcase *c = &cases[i];

        for (int fini = 0; fini < 2 && !killed; fini++) {
            const char *which = fini ? "FINI" : "PREP";
            pid_t pid = fork();
            int status = 0, err = 0;

            if (pid == 0) {
                int rc = run_case(fd, bo.handle, BO_BYTES, c, fini, &err);
                _exit(rc < 0 ? (err & 0x7f) : 0);
            }
            if (pid < 0) {
                fprintf(stderr, "fork failed\n");
                return 1;
            }
            waitpid(pid, &status, 0);
            if (!WIFEXITED(status)) {
                printf("  %-22s %s  KERNEL KILLED THE CHILD (signal %d) — %s\n",
                       c->name, which, WTERMSIG(status), c->why);
                killed = 1;
                failures++;
                break;
            }
            err = WEXITSTATUS(status);
            if (c->want_err == 0 && err != 0) {
                printf("  %-22s %s  FAIL: refused with errno %d, expected success — %s\n",
                       c->name, which, err, c->why);
                failures++;
            } else if (c->want_err != 0 && err == 0) {
                printf("  %-22s %s  FAIL: accepted, expected errno %d — %s\n",
                       c->name, which, c->want_err, c->why);
                failures++;
            } else if (c->want_err != 0 && err != c->want_err) {
                /* Came back, which is the crash contract; wrong errno is an
                 * interface failure and is reported as one. */
                printf("  %-22s %s  FAIL: errno %d, expected %d — %s\n",
                       c->name, which, err, c->want_err, c->why);
                failures++;
            } else {
                printf("  %-22s %s  ok (%s)\n", c->name, which,
                       c->want_err ? "refused" : "accepted");
            }
        }
    }

    rocket_bo_free(fd, &bo);
    rocket_bo_free(fd, &guard);
    rocket_close(fd);

    if (killed)
        printf("STOPPED: the kernel died on a rejection path. Do not keep running gates.\n");
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
