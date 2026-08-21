// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * uapi_submit_errpath_rocket.c — every REJECTED submit must return an errno,
 * not kill the kernel.
 *
 * rocket_ioctl_submit_job() builds its rocket_job with kzalloc and only takes
 * the IOMMU domain reference AFTER the task copy and both BO lookups have
 * succeeded. Every rejection before that point unwinds through
 * rocket_job_cleanup(), which puts job->domain unconditionally — so a client
 * that hands the ioctl a malformed job walks the kernel into
 * rocket_iommu_domain_put(NULL) and oopses. /dev/accel/accel0 is group
 * `render`, so the trigger is unprivileged.
 *
 * The malformations below are one per reachable rejection site, in the order
 * the ioctl reaches them. Each runs in a FORKED CHILD: an oops kills the child
 * and the parent still reports which site did it. The probe STOPS at the first
 * child the kernel kills — once one site is known unfixed there is nothing to
 * learn from crashing the box four more times.
 *
 * A rejected submit's errno is NOT checked here. Stock rocket_ioctl_submit()
 * discards rocket_ioctl_submit_job()'s return value and reports success for a
 * job it threw away; that is a separate defect (fixed by the RK3588 series'
 * 085). This probe pins only the crash contract: the caller comes back.
 *
 * GATE (registered in CTest): exit 0 = every rejection returned to userspace,
 * 1 = the kernel killed a child, 2 = no NPU device (skip).
 *
 * Build: linked against rocketnpu by CMake. Run: ./uapi_submit_errpath_rocket
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

#include <drm/rocket_accel.h>
#include <drm/drm.h>

#include "rocket_npu.h"

/* Declare the sizes a normal client declares -- its own. Anything smaller can be
 * refused by rocket_ioctl_submit()'s size gate, which sits AHEAD of the rejection
 * sites under test, and a probe that gets refused there proves nothing about
 * them. The v1 minimum (offsetofend of the last original field, which no appended
 * field can move) is used only where a deliberately-undersized value is the
 * malformation. */
#define JOB_SZ  sizeof(struct drm_rocket_job)
#define TASK_SZ sizeof(struct drm_rocket_task)
#define TASK_V1 (offsetof(struct drm_rocket_task, regcmd_count) + \
                 sizeof(((struct drm_rocket_task *)0)->regcmd_count))

struct errpath {
    const char *name;
    const char *site;      /* the driver function that rejects it */
    const char *expect;    /* the errno the driver's own code returns there */
};

/* In the order rocket_ioctl_submit_job() reaches them. Every one of these is
 * rejected BEFORE rjob->domain is assigned. */
enum {
    CASE_TASK_STRUCT_SIZE,
    CASE_TASK_EFAULT,
    CASE_REGCMD_COUNT_ZERO,
    CASE_BAD_IN_BO,
    CASE_BAD_OUT_BO,
    CASE_COUNT
};

static const struct errpath CASES[CASE_COUNT] = {
    [CASE_TASK_STRUCT_SIZE]  = { "task_struct_size below the v1 minimum",
                                 "rocket_copy_tasks", "EINVAL" },
    [CASE_TASK_EFAULT]       = { "unreadable drm_rocket_job.tasks pointer",
                                 "rocket_copy_tasks (copy_from_user)", "EFAULT" },
    [CASE_REGCMD_COUNT_ZERO] = { "drm_rocket_task.regcmd_count == 0",
                                 "rocket_copy_tasks", "EINVAL" },
    [CASE_BAD_IN_BO]         = { "no such in_bo handle",
                                 "drm_gem_objects_lookup (in)", "ENOENT" },
    [CASE_BAD_OUT_BO]        = { "no such out_bo handle",
                                 "drm_gem_objects_lookup (out)", "ENOENT" },
};

/* CASE_NONE builds a job that rocket_ioctl_submit_job() rejects at its very
 * first statement -- task_count == 0, refused BEFORE the kzalloc, so it cannot
 * reach the cleanup path however broken the kernel is. It is the control that
 * says the probe got past the ioctl-level size gate and into the function whose
 * rejection sites are under test. */
#define CASE_NONE (-1)

/* Build and fire the malformed submit for one case. Runs in the child. */
static int fire(int fd, int which, uint32_t good_handle)
{
    uint32_t bogus  = 0xFFFFFFF0u;              /* no such GEM handle */
    struct drm_rocket_task t = { .regcmd = 0, .regcmd_count = 1 };
    struct drm_rocket_job  j = {
        .tasks               = (uint64_t)(uintptr_t)&t,
        .task_count          = (which == CASE_NONE) ? 0u : 1u,
        .task_struct_size    = (uint32_t)TASK_SZ,
        .in_bo_handles       = (uint64_t)(uintptr_t)&good_handle,
        .in_bo_handle_count  = 1,
        .out_bo_handles      = (uint64_t)(uintptr_t)&good_handle,
        .out_bo_handle_count = 1,
    };

    switch (which) {
    case CASE_TASK_STRUCT_SIZE:
        j.task_struct_size = (uint32_t)(TASK_V1 - 4);
        break;
    case CASE_TASK_EFAULT:
        /* A userspace address that is guaranteed not to be mapped. */
        j.tasks = (uint64_t)0xdeadbeef000ULL;
        break;
    case CASE_REGCMD_COUNT_ZERO:
        t.regcmd_count = 0;
        break;
    case CASE_BAD_IN_BO:
        j.in_bo_handles = (uint64_t)(uintptr_t)&bogus;
        break;
    case CASE_BAD_OUT_BO:
        j.out_bo_handles = (uint64_t)(uintptr_t)&bogus;
        break;
    }

    struct drm_rocket_submit s = {
        .jobs = (uint64_t)(uintptr_t)&j, .job_count = 1,
        .job_struct_size = (uint32_t)JOB_SZ,
    };

    errno = 0;
    int rc = ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, &s);
    return rc < 0 ? errno : 0;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no rocket device (skip)\n"); return 2; }

    char name[32] = {0};
    struct drm_version dv = { .name = name, .name_len = sizeof(name)-1 };
    ioctl(fd, DRM_IOCTL_VERSION, &dv);

    printf("== rocket submit error-path probe ==\n");
    printf("  info : driver \"%s\" v%d.%d.%d\n", name,
           dv.version_major, dv.version_minor, dv.version_patchlevel);
    printf("  info : every case below is a submit the driver MUST reject; the\n"
           "         contract under test is that the caller comes back at all\n");

    /* One real BO, so a case that is not testing the BO lookup passes it. */
    rocket_bo bo = {0};
    if (rocket_bo_alloc(fd, 4096, &bo) != 0) {
        fprintf(stderr, "  FAIL : CREATE_BO for the control handle\n");
        rocket_close(fd);
        return 1;
    }

    /* Control: task_count == 0 is refused at the top of rocket_ioctl_submit_job(),
     * before the allocation, so it cannot crash any kernel. If THIS comes back
     * -EINVAL on a driver that reports per-job errnos it is indistinguishable from
     * the ioctl-level size gate, and the cases below would be measuring the gate
     * rather than the sites. Report what it did and let the reader see it. */
    {
        int e = fire(fd, CASE_NONE, bo.handle);
        printf("  info : control (task_count == 0, refused before the alloc) -> %s\n",
               e ? strerror(e) : "0, per-job errno discarded by the ioctl");
        if (e == EINVAL)
            printf("  info : -EINVAL here is also what the ioctl's size gate returns; if every\n"
                   "         case below also reports EINVAL, check the header against the kernel\n");
    }

    int fails = 0;
    for (int i = 0; i < CASE_COUNT; i++) {
        /* Announce BEFORE firing: if the kernel takes the whole box down
         * rather than just the child, the console names the site. */
        printf("  ---- : %s (rejected at %s, %s)\n",
               CASES[i].name, CASES[i].site, CASES[i].expect);
        fflush(stdout);

        pid_t pid = fork();
        if (pid < 0) { perror("fork"); fails++; break; }

        if (pid == 0) {
            int e = fire(fd, i, bo.handle);
            _exit(e == 0 ? 100 : 101);   /* survived; 100/101 = errno or not */
        }

        int status = 0;
        waitpid(pid, &status, 0);

        if (WIFSIGNALED(status)) {
            printf("  FAIL : %s -- the kernel killed the caller (signal %d)\n",
                   CASES[i].name, WTERMSIG(status));
            printf("  info : this is the rocket_iommu_domain_put(NULL) oops; "
                   "check dmesg for rocket_job_cleanup\n");
            printf("  info : stopping -- the remaining cases would only oops "
                   "the same kernel again\n");
            fails++;
            break;
        }

        printf("  ok   : %s -- returned to userspace (%s)\n", CASES[i].name,
               WEXITSTATUS(status) == 101 ? "errno set"
                                          : "reported success, errno discarded");
    }

    rocket_bo_free(fd, &bo);
    rocket_close(fd);

    printf("\n%s: %d/%d rejection sites returned to userspace\n",
           fails ? "FAIL" : "PASS", CASE_COUNT - fails, CASE_COUNT);
    return fails ? 1 : 0;
}
