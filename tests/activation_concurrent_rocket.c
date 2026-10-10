// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * activation_concurrent_rocket.c — the parametric LUT activations under concurrent callers.
 *
 * rocket_leaky_relu_fp16() builds a table that depends on its slope, and the ELU/SELU body
 * builds one that depends on alpha and lambda — ELU and SELU share that body and build
 * different tables. Each table is written, then laid into the call's own regcmd and BOs.
 * Built in storage shared across calls, two callers on two fds overwrite each other's
 * table between those steps, and the device computes a full, correctly sized surface of
 * the OTHER caller's function. The serial gates cannot see that: they call one at a time.
 *
 * Two threads, each on its own fd, are released from one barrier per round and alternate:
 *
 *   thread 0   LeakyReLU slope 0.01, ELU alpha 1.0, LeakyReLU 0.01, ...
 *   thread 1   LeakyReLU slope 0.3,  SELU,          LeakyReLU 0.3,  ...
 *
 * so every round runs two slopes at once, then ELU beside SELU. Each result is scored
 * against the same configuration run serially on the main fd before the threads start.
 * The tolerance is two fp16 units in the last place, which admits a different core's
 * rounding and nothing a different table produces: the slopes differ thirtyfold on every
 * negative input and SELU is ELU scaled by 1.0507 with alpha 1.673.
 *
 * The LUT encoding is the RK3588's, so the gate skips on any other part.
 *
 * Usage: activation_concurrent_rocket [rounds]   (default 64)
 * Exits 0 when every result matched, 1 on any mismatch or failed call, 2 with no device or
 * on another part.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "rocket_activation.h"
#include "test_fill.h"

#define N_ELEM 4096

enum { CFG_LEAKY_A, CFG_LEAKY_B, CFG_ELU, CFG_SELU, CFG_N };
static const char *const cfg_name[CFG_N] = { "leaky 0.01", "leaky 0.3", "elu 1.0", "selu" };

static _Float16 g_in[N_ELEM];
static _Float16 g_ref[CFG_N][N_ELEM];

static int run_cfg(int fd, int cfg, _Float16 *out)
{
    switch (cfg) {
    case CFG_LEAKY_A: return rocket_leaky_relu_fp16(fd, 0.01f, g_in, out, N_ELEM);
    case CFG_LEAKY_B: return rocket_leaky_relu_fp16(fd, 0.3f, g_in, out, N_ELEM);
    case CFG_ELU:     return rocket_elu_fp16(fd, 1.0f, g_in, out, N_ELEM);
    default:          return rocket_selu_fp16(fd, g_in, out, N_ELEM);
    }
}

static int ulp_diff(_Float16 a, _Float16 b)
{
    int16_t x, y;
    memcpy(&x, &a, 2); memcpy(&y, &b, 2);
    if (x < 0) x = (int16_t)(0x8000 - x);
    if (y < 0) y = (int16_t)(0x8000 - y);
    return abs(x - y);
}

/* Elements more than two ulp from the configuration's serial result. */
static long mismatches(int cfg, const _Float16 *out)
{
    long bad = 0;
    for (int i = 0; i < N_ELEM; i++) bad += ulp_diff(out[i], g_ref[cfg][i]) > 2;
    return bad;
}

struct worker {
    pthread_barrier_t *bar;
    int fd, rounds, cfg_even, cfg_odd;
    long calls, bad_calls, failed_calls;
};

static void *worker_main(void *p)
{
    struct worker *w = (struct worker *)p;
    _Float16 *out = malloc(sizeof(_Float16) * N_ELEM);
    for (int r = 0; r < w->rounds; r++) {
        const int cfg = (r & 1) ? w->cfg_odd : w->cfg_even;
        pthread_barrier_wait(w->bar);
        if (!out) { w->failed_calls++; continue; }
        memset(out, 0, sizeof(_Float16) * N_ELEM);
        w->calls++;
        if (run_cfg(w->fd, cfg, out) != 0) { w->failed_calls++; continue; }
        const long bad = mismatches(cfg, out);
        if (bad) {
            if (w->bad_calls < 4)
                printf("  FAIL   round %d %-10s: %ld of %d elements off its serial result\n",
                       r, cfg_name[cfg], bad, N_ELEM);
            w->bad_calls++;
        }
    }
    free(out);
    return NULL;
}

int main(int argc, char **argv)
{
    const int rounds = argc > 1 ? atoi(argv[1]) : 64;
    const struct rocket_hw_profile *hw = rocket_hw_current();
    pthread_barrier_t bar;
    struct worker w[2];
    pthread_t th[2];
    int fd, i, fail = 0;

    if (rounds < 2) { fprintf(stderr, "activation_concurrent_rocket: rounds must be >= 2\n"); return 1; }
    if (!hw || strcmp(hw->name, "rk3588") != 0) {
        printf("activation_concurrent_rocket: profile is %s; the LUT activations are the "
               "RK3588 encoding — skipping\n", hw ? hw->name : "(none)");
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) {
        printf("activation_concurrent_rocket: no NPU device (%d) — skipping\n", fd);
        return 2;
    }

    /* Both branches of every function, past the x~0 band the entries repair on the host. */
    tf_fill_f16(g_in, N_ELEM, 0xAC71u, -6.0, 6.0);
    for (i = 0; i < CFG_N; i++) {
        int rc = run_cfg(fd, i, g_ref[i]);
        if (rc != 0) {
            printf("  FAIL   serial %s returned %d\n", cfg_name[i], rc);
            rocket_close(fd);
            return 1;
        }
    }
    /* The configurations must be told apart at all, or a swapped table would score. */
    if (!mismatches(CFG_LEAKY_A, g_ref[CFG_LEAKY_B]) || !mismatches(CFG_ELU, g_ref[CFG_SELU])) {
        printf("  FAIL   two configurations gave the same serial surface; the gate cannot "
               "see a swapped table\n");
        rocket_close(fd);
        return 1;
    }

    if (pthread_barrier_init(&bar, NULL, 2) != 0) { rocket_close(fd); return 1; }
    memset(w, 0, sizeof w);
    w[0].cfg_even = CFG_LEAKY_A; w[0].cfg_odd = CFG_ELU;
    w[1].cfg_even = CFG_LEAKY_B; w[1].cfg_odd = CFG_SELU;
    for (i = 0; i < 2; i++) {
        w[i].bar = &bar;
        w[i].rounds = rounds;
        w[i].fd = rocket_open();
        if (w[i].fd < 0) {
            printf("  FAIL   worker %d could not open its own fd (%d)\n", i, w[i].fd);
            fail = 1;
        }
    }
    if (!fail) {
        for (i = 0; i < 2; i++) pthread_create(&th[i], NULL, worker_main, &w[i]);
        for (i = 0; i < 2; i++) pthread_join(th[i], NULL);
    }
    pthread_barrier_destroy(&bar);
    for (i = 0; i < 2; i++) {
        if (w[i].fd >= 0) rocket_close(w[i].fd);
        if (w[i].bad_calls || w[i].failed_calls) fail = 1;
        printf("activation_concurrent_rocket: worker %d: %ld calls, %ld off their serial "
               "result, %ld failed\n", i, w[i].calls, w[i].bad_calls, w[i].failed_calls);
    }
    rocket_close(fd);
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail;
}
