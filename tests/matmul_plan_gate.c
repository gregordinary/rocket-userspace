// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_plan_gate.c — host-only GATE on the shared matmul tiling policy.
 *
 * Every dtype entry point plans through one parameterised policy (mm_plan_geom in
 * rocket_matmul.c). This gate asserts the invariants that policy owes its callers, over
 * every shape any of them could be handed and every ROCKET_MM_* override:
 *
 *   1. A degenerate dimension is REFUSED, never planned. This is the one that matters:
 *      Mt has no floor (Nt and Kt floor at their alignment), so M == 0 used to reach
 *      `nm = (M + Mt - 1) / Mt` with Mt == 0 and take the process down with SIGFPE. It
 *      is wider than M — an MM_NT / MM_KT override clamps Nt / Kt back DOWN to a zero
 *      N / K, so the "the other two are floored" argument only holds without the knobs.
 *   2. An accepted plan's tiles are POSITIVE and correctly ALIGNED, and no tile exceeds
 *      the dimension it tiles rounded up to its own granularity.
 *   2b. On the DEFAULT path, no tile exceeds the profile's max_tile. ROCKET_MM_MT /
 *      ROCKET_MM_NT are deliberately NOT held to that: they clamp against the dimension
 *      only, so ROCKET_MM_MT=384 plans Mt=320 on a part whose cap is 256. That is an RE
 *      escape hatch, not an accident — the knobs exist to probe past the default — but it
 *      does mean a value above the cap asks the generator for a tile the profile says the
 *      hardware does not have. Set them knowing that.
 *   3. The returned job count is exactly nMt * nNt * nKt, positive, and free of the
 *      signed overflow ROCKET_MM_DIM_MAX exists to keep out of the tile-count math.
 *   4. A group-wise plan's Kt DIVIDES the quant group — a K-tile that straddles a group
 *      would silently apply one group's scale to another's products — and a group-wise
 *      planner refuses a missing group rather than quietly planning dense.
 *
 * No NPU and no allocation: this is pure arithmetic, so it never skips.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

static long checked, failed;

static void fail(const char *what, const char *dt, const char *tag,
                 int M, int K, int N, int rc, int Mt, int Kt, int Nt)
{
    if (failed < 20)
        printf("FAIL [%s] %s %s M=%d K=%d N=%d -> rc=%d (Mt=%d Kt=%d Nt=%d)\n",
               what, dt, tag, M, K, N, rc, Mt, Kt, Nt);
    failed++;
}

/* One accepted plan's invariants. `group` == 0 for the dense planners. */
static void check_plan(const char *dt, const char *tag, int M, int K, int N, int group,
                       int nalign, int kalign, int strict_cap,
                       int rc, int Mt, int Kt, int Nt)
{
    const int max_tile = rocket_hw_current()->max_tile;

    if (Mt <= 0 || Kt <= 0 || Nt <= 0)
        return fail("non-positive tile", dt, tag, M, K, N, rc, Mt, Kt, Nt);
    if (Mt % 4 || Nt % nalign || Kt % kalign)
        return fail("misaligned tile", dt, tag, M, K, N, rc, Mt, Kt, Nt);
    if (strict_cap && (Mt > max_tile || Nt > max_tile))
        return fail("tile over the cap", dt, tag, M, K, N, rc, Mt, Kt, Nt);
    /* A tile may exceed its dimension only by that dimension's own rounding. */
    if (Mt > M + 3 || Nt > N + nalign - 1 || Kt > K + kalign - 1)
        return fail("tile over its dimension", dt, tag, M, K, N, rc, Mt, Kt, Nt);
    if (group && (Kt > group || group % Kt))
        return fail("K-tile straddles a quant group", dt, tag, M, K, N, rc, Mt, Kt, Nt);

    long nm = (M + Mt - 1) / Mt, nn = (N + Nt - 1) / Nt, nk = (K + Kt - 1) / Kt;
    if (nm * nn * nk != (long)rc)
        return fail("job count != nMt*nNt*nKt", dt, tag, M, K, N, rc, Mt, Kt, Nt);
    if (rc <= 0)
        return fail("non-positive job count", dt, tag, M, K, N, rc, Mt, Kt, Nt);
}

typedef int (*planfn)(int, int, int, int *, int *, int *);
static const struct { const char *name; planfn fn; int nalign, kalign; } dense[] = {
    { "fp16",  rocket_matmul_plan,       16, 32 },
    { "int8",  rocket_matmul_plan_int8,  32, 32 },
    { "int4",  rocket_matmul_plan_int4,  64, 32 },
    { "int16", rocket_matmul_plan_int16, 16, 32 },
    { "bf16",  rocket_matmul_plan_bf16,  16, 32 },
    { "tf32",  rocket_matmul_plan_tf32,  16, 16 },
};

/* Dimensions chosen to straddle every alignment (16/32/64), the tile cap, the Kt
 * ceilings (8192 / 16384) and ROCKET_MM_DIM_MAX, plus the degenerate values. */
static const int dims[] = {
    -64, -1, 0, 1, 2, 3, 4, 8, 12, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128,
    129, 160, 192, 255, 256, 257, 320, 384, 512, 576, 640, 768, 1024, 1152, 1536, 2048,
    2880, 4096, 8192, 8208, 11008, 16384, 16416, 65536, (1 << 24), (1 << 24) + 32,
};
#define NDIM ((int)(sizeof dims / sizeof dims[0]))

/* `strict_cap`: hold the plan to the profile's max_tile. Off when an MM_MT / MM_NT
 * override is in force — see invariant 2b. */
static void sweep(const char *tag, int strict_cap)
{
    for (size_t d = 0; d < sizeof dense / sizeof dense[0]; d++)
        for (int a = 0; a < NDIM; a++)
            for (int b = 0; b < NDIM; b++)
                for (int c = 0; c < NDIM; c++) {
                    int M = dims[a], K = dims[b], N = dims[c];
                    int Mt = -1, Kt = -1, Nt = -1;
                    int rc = dense[d].fn(M, K, N, &Mt, &Kt, &Nt);
                    checked++;
                    if (M <= 0 || K <= 0 || N <= 0) {
                        if (rc >= 0)
                            fail("degenerate dim ACCEPTED", dense[d].name, tag,
                                 M, K, N, rc, Mt, Kt, Nt);
                        continue;
                    }
                    if (rc < 0) continue;                     /* an honest refusal */
                    check_plan(dense[d].name, tag, M, K, N, 0, dense[d].nalign,
                               dense[d].kalign, strict_cap, rc, Mt, Kt, Nt);
                }

    static const int groups[] = { -32, 0, 1, 16, 31, 32, 33, 64, 128, 256, 512, 1024, 4096 };
    for (int a = 0; a < NDIM; a++)
        for (int b = 0; b < NDIM; b++)
            for (int c = 0; c < NDIM; c++)
                for (size_t g = 0; g < sizeof groups / sizeof groups[0]; g++) {
                    int M = dims[a], K = dims[b], N = dims[c], grp = groups[g];
                    int Mt = -1, Kt = -1, Nt = -1;
                    int rc = rocket_matmul_plan_int8_gw(M, K, N, grp, &Mt, &Kt, &Nt);
                    checked++;
                    if (M <= 0 || K <= 0 || N <= 0 || grp <= 0) {
                        if (rc >= 0)
                            fail("degenerate dim/group ACCEPTED", "int8gw", tag,
                                 M, K, N, rc, Mt, Kt, Nt);
                        continue;
                    }
                    if (rc < 0) continue;
                    check_plan("int8gw", tag, M, K, N, grp, 32, 32,
                               strict_cap, rc, Mt, Kt, Nt);
                }
}

int main(void)
{
    /* The RK3588 planners refuse by construction on another profile, which would make
     * every shape an honest refusal and the gate vacuous. Say so rather than pass. */
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (!hw || strcmp(hw->name, "rk3588") != 0) {
        printf("SKIP: active profile is '%s'; these planners are RK3588\n",
               hw ? hw->name : "(none)");
        return 2;
    }

    static const char *knobs[] = { "ROCKET_MM_MT", "ROCKET_MM_NT",
                                   "ROCKET_MM_KT", "ROCKET_MM_ASYM" };
    static const char *vals[] = { "0", "1", "4", "16", "32", "64", "128",
                                  "192", "256", "384", "999999" };
    /* A knob left set by the environment would silently retune every sweep below. */
    for (size_t k = 0; k < sizeof knobs / sizeof knobs[0]; k++) unsetenv(knobs[k]);

    sweep("default", /*strict_cap=*/1);
    for (size_t k = 0; k < sizeof knobs / sizeof knobs[0]; k++) {
        /* MM_MT / MM_NT name a tile directly, so they can exceed the cap (2b). MM_KT and
         * MM_ASYM cannot: neither touches Mt or Nt. */
        int strict = strcmp(knobs[k], "ROCKET_MM_MT") != 0 &&
                     strcmp(knobs[k], "ROCKET_MM_NT") != 0;
        for (size_t v = 0; v < sizeof vals / sizeof vals[0]; v++) {
            char tag[64];
            snprintf(tag, sizeof tag, "%s=%s", knobs[k], vals[v]);
            setenv(knobs[k], vals[v], 1);
            sweep(tag, strict);
            unsetenv(knobs[k]);
        }
    }

    printf("matmul_plan_gate: %ld plans checked, %ld failures\n", checked, failed);
    return failed ? 1 : 0;
}
