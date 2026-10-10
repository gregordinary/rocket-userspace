// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * open_guard_race_rocket.c — the RK3576 second-fd guard holds when two threads open at once.
 *
 * On an RK3576 with more than one NPU core bound, rocket_open() refuses a second fd with
 * -EBUSY, because two fds is what puts two jobs in flight and two jobs in flight compute
 * wrong answers on that part. A guard that READS the open count and adds to it in a separate
 * step passes both of two concurrent opens: each reads zero before either has added. This
 * gate releases two threads from one barrier into rocket_open(), round after round, and
 * scores every round:
 *
 *   guard armed     (RK3576, >= 2 bound cores, ROCKET_RK3576_ALLOW_MULTI_FD unset)
 *                   exactly one fd and exactly one -EBUSY, every round
 *   guard not armed (any other part, one bound core, or the override set)
 *                   both opens succeed, every round — the guard must not fire there
 *
 * The race window is a few instructions wide, so a single round says little either way;
 * the round count is what gives a check-then-act guard the chance to show. Neither thread
 * submits anything.
 *
 * Usage: open_guard_race_rocket [rounds]   (default 500)
 * Exits 0 when every round scored, 1 on any wrong round, 2 with no rocket device.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "rocket_sysfs.h"          /* the bound-core count the guard itself reads */

struct opener {
    pthread_barrier_t *bar;
    int fd;
};

static void *open_after_barrier(void *p)
{
    struct opener *o = (struct opener *)p;
    pthread_barrier_wait(o->bar);
    o->fd = rocket_open();
    return NULL;
}

int main(int argc, char **argv)
{
    const int rounds = argc > 1 ? atoi(argv[1]) : 500;
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *allow = getenv("ROCKET_RK3576_ALLOW_MULTI_FD");
    const int override = allow && *allow && *allow != '0';
    int bound, armed, r, wrong = 0, other = 0;

    if (rounds < 1) { fprintf(stderr, "open_guard_race_rocket: rounds must be >= 1\n"); return 1; }

    /* One open alone first: no device, or not the rocket driver, is a skip. */
    {
        int fd = rocket_open();
        if (fd < 0) {
            printf("open_guard_race_rocket: no rocket NPU device (rocket_open %d) — skipping\n", fd);
            return 2;
        }
        rocket_close(fd);
    }

    bound = rocket_sysfs_bound_core_count();
    armed = hw && strcmp(hw->name, "rk3576") == 0 && bound > 1 && !override;
    printf("open_guard_race_rocket: profile %s, %d bound core(s), override %s -> guard %s\n",
           hw ? hw->name : "(none)", bound, override ? "set" : "unset",
           armed ? "ARMED: each round must give one fd and one -EBUSY"
                 : "not armed: each round must give two fds");

    for (r = 0; r < rounds; r++) {
        pthread_barrier_t bar;
        struct opener o[2];
        pthread_t th[2];
        int i, ok = 0, busy = 0;

        if (pthread_barrier_init(&bar, NULL, 2) != 0) {
            fprintf(stderr, "open_guard_race_rocket: pthread_barrier_init failed\n");
            return 1;
        }
        for (i = 0; i < 2; i++) {
            o[i].bar = &bar;
            o[i].fd = -1;
            if (pthread_create(&th[i], NULL, open_after_barrier, &o[i]) != 0) {
                fprintf(stderr, "open_guard_race_rocket: pthread_create failed\n");
                return 1;
            }
        }
        for (i = 0; i < 2; i++) pthread_join(th[i], NULL);
        pthread_barrier_destroy(&bar);

        for (i = 0; i < 2; i++) {
            if (o[i].fd >= 0)           ok++;
            else if (o[i].fd == -EBUSY) busy++;
            else                        other++;
        }
        if (armed ? (ok != 1 || busy != 1) : ok != 2) {
            if (wrong < 10)
                printf("  FAIL   round %d: %d fd(s), %d -EBUSY (rc %d, %d)\n",
                       r, ok, busy, o[0].fd, o[1].fd);
            wrong++;
        }
        for (i = 0; i < 2; i++) if (o[i].fd >= 0) rocket_close(o[i].fd);
    }

    printf("open_guard_race_rocket: %d round(s), %d wrong%s\n", rounds, wrong,
           other ? " (some opens failed with an error other than -EBUSY)" : "");
    return wrong ? 1 : 0;
}
