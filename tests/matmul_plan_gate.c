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
 * `matmul_plan_gate rk3576` holds the RK3576 int8 planner (rocket_matmul_plan_int8_rk3576)
 * to its own envelope instead, with the profile forced to that part:
 *
 *   5. Its job count is exactly ceil(N/Nt) * ceil(M/Mt) wherever it plans — M is the plane
 *      iw*ih and Mt one task's rows times iw — so a count that wrapped cannot pass.
 *   6. A PRIME M returns at once: the plane-width search is bounded by the CBUF budget,
 *      and a search bounded only by M walks every width up to M. The primes here sit just
 *      under ROCKET_MM_DIM_MAX, inside the envelope, so they reach the search rather than
 *      the dimension refusal.
 *   7. A plan whose job count passes INT_MAX is REFUSED with ROCKET_E_TILING, never
 *      returned as a negative int that reads as some other rocket_status.
 *   8. A dimension past ROCKET_MM_DIM_MAX, and a plane taller than the 16-bit field it is
 *      programmed into, are refused with ROCKET_E_SHAPE.
 *
 * No NPU and no allocation: this is pure arithmetic, so the RK3588 run skips only on another
 * part and the RK3576 run never skips.
 */
#define _GNU_SOURCE
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "rocket_npu.h"

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

/* Every refusal below is `continue`d as honest, so a planner that refused EVERY shape
 * would pass the sweep with nothing checked. This shape is the anchor: a real prefill
 * GEMM that every dense planner, and the group-wise one at a 128 group, must plan under
 * every knob setting the sweep visits. */
#define ANCHOR_M 512
#define ANCHOR_K 4096
#define ANCHOR_N 4096
#define ANCHOR_GROUP 128

static void require_anchor(const char *tag, int strict_cap)
{
    for (size_t d = 0; d < sizeof dense / sizeof dense[0]; d++) {
        int Mt = -1, Kt = -1, Nt = -1;
        int rc = dense[d].fn(ANCHOR_M, ANCHOR_K, ANCHOR_N, &Mt, &Kt, &Nt);
        checked++;
        if (rc < 0)
            fail("the anchor shape was REFUSED", dense[d].name, tag,
                 ANCHOR_M, ANCHOR_K, ANCHOR_N, rc, Mt, Kt, Nt);
        else
            check_plan(dense[d].name, tag, ANCHOR_M, ANCHOR_K, ANCHOR_N, 0, dense[d].nalign,
                       dense[d].kalign, strict_cap, rc, Mt, Kt, Nt);
    }
    int Mt = -1, Kt = -1, Nt = -1;
    int rc = rocket_matmul_plan_int8_gw(ANCHOR_M, ANCHOR_K, ANCHOR_N, ANCHOR_GROUP,
                                        &Mt, &Kt, &Nt);
    checked++;
    if (rc < 0)
        fail("the anchor shape was REFUSED", "int8gw", tag,
             ANCHOR_M, ANCHOR_K, ANCHOR_N, rc, Mt, Kt, Nt);
    else
        check_plan("int8gw", tag, ANCHOR_M, ANCHOR_K, ANCHOR_N, ANCHOR_GROUP, 32, 32,
                   strict_cap, rc, Mt, Kt, Nt);
}

/* `strict_cap`: hold the plan to the profile's max_tile. Off when an MM_MT / MM_NT
 * override is in force — see invariant 2b. */
static void sweep(const char *tag, int strict_cap)
{
    long accepted = 0;
    require_anchor(tag, strict_cap);
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
                    accepted++;
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
                    accepted++;
                    check_plan("int8gw", tag, M, K, N, grp, 32, 32,
                               strict_cap, rc, Mt, Kt, Nt);
                }
    if (accepted == 0)
        fail("no shape planned: every invariant above went unchecked", "all", tag,
             0, 0, 0, -1, -1, -1, -1);
}

/* ============================================================================
 * SECTION — the RK3576 int8 planner (`matmul_plan_gate rk3576`)
 * ==========================================================================*/

/* The refusals below are the point, and each logs; keep the gate's own lines readable. */
static void quiet_log(rocket_log_level level, const char *text, void *user)
{
    (void)level; (void)text; (void)user;
}

static double cpu_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* One RK3576 plan: invariant 5 if it planned, the expected code if it should not have. */
static int r76_plan(const char *what, int M, int K, int N, int want)
{
    int Mt = -1, Kt = -1, Nt = -1;
    int rc = rocket_matmul_plan_int8_rk3576(M, K, N, &Mt, &Kt, &Nt);
    checked++;
    if (want < 0) {
        if (rc != want) {
            printf("FAIL [%s] rk3576 M=%d K=%d N=%d -> rc=%d, expected the refusal %d\n",
                   what, M, K, N, rc, want);
            failed++;
        }
        return rc;
    }
    if (rc <= 0) {
        if (want > 0) {
            printf("FAIL [%s] rk3576 M=%d K=%d N=%d was refused (%d)\n", what, M, K, N, rc);
            failed++;
        }
        return rc;
    }
    if (Mt <= 0 || Nt <= 0 || Nt % 32 || Kt != K ||
        (uint64_t)rc != (uint64_t)((N + Nt - 1) / Nt) * (uint64_t)((M + Mt - 1) / Mt)) {
        printf("FAIL [%s] rk3576 M=%d K=%d N=%d -> rc=%d (Mt=%d Kt=%d Nt=%d): the job count "
               "is not ceil(N/Nt)*ceil(M/Mt)\n", what, M, K, N, rc, Mt, Kt, Nt);
        failed++;
    }
    return rc;
}

static int rk3576_main(void)
{
    static const int primes[] = { 16777213, 16777199, 16777183, 16777153 };
    static const int Ms[] = { 1, 4, 7, 64, 100, 196, 512, 1000, 2048, 3136, 12544, 65521 };
    static const int KNs[] = { 32, 64, 96, 256, 1024, 1536, 2048, 4096 };
    static const char *knobs[] = { "ROCKET_RK3576_MM_IW", "ROCKET_RK3576_MM_NT",
                                   "ROCKET_RK3576_MM_PHASE" };
    const struct rocket_hw_profile *hw;
    double t0, spent;
    int planned = 0;

    setenv("ROCKET_CHIP", "rk3576", 1);
    for (size_t k = 0; k < sizeof knobs / sizeof knobs[0]; k++) unsetenv(knobs[k]);
    rocket_log_set_callback(quiet_log, NULL);
    hw = rocket_hw_current();
    if (!hw || strcmp(hw->name, "rk3576") != 0) {
        printf("FAIL: ROCKET_CHIP=rk3576 did not select the RK3576 profile\n");
        return 1;
    }

    /* 5, over ordinary shapes; and something must plan, or the sweep checked nothing. */
    for (size_t a = 0; a < sizeof Ms / sizeof Ms[0]; a++)
        for (size_t b = 0; b < sizeof KNs / sizeof KNs[0]; b++)
            for (size_t c = 0; c < sizeof KNs / sizeof KNs[0]; c++)
                planned += r76_plan("sweep", Ms[a], KNs[b], KNs[c], 0) > 0;
    r76_plan("anchor", 512, 1536, 1536, 1);
    if (!planned) {
        printf("FAIL [sweep] rk3576: no shape planned, so invariant 5 went unchecked\n");
        failed++;
    }

    /* 6. Each prime's only plane is one pixel wide, which invariant 8 then refuses — but the
     * search that finds it runs first, and it is the search being timed. Unbounded, these
     * 64 calls are about a billion divisions (1.4 s of CPU on an x86 host); bounded by the
     * CBUF budget, about half a million. */
    t0 = cpu_s();
    for (int rep = 0; rep < 16; rep++)
        for (size_t p = 0; p < sizeof primes / sizeof primes[0]; p++)
            r76_plan("prime M", primes[p], 32, 32, ROCKET_E_SHAPE);
    spent = cpu_s() - t0;
    checked++;
    if (spent > 0.2) {
        printf("FAIL [prime M] rk3576: 64 plans of a prime M took %.3f s of CPU — the "
               "plane-width search is not bounded by the CBUF budget\n", spent);
        failed++;
    }
    /* A prime short enough for the field plans, as a one-pixel-wide plane. */
    r76_plan("prime M in the field", 65521, 64, 64, 1);

    /* 7. A 51-pixel plane 65535 rows tall at K 5120, whose tile is 32 channels and whose
     * row tasks are one row each: N/32 * 65535 jobs. N 2^20 is 2147450880 jobs, under
     * INT_MAX, and must plan and count exactly; one more 32-channel tile is 2147516415,
     * over it, and must refuse. The edge is read off the plan at N/64, whose own tiles
     * must be the 51-row, 32-channel ones, or the shape is not testing what it says. */
    {
        const int M = 65535 * 51, K = 5120;
        int Mt = -1, Kt = -1, Nt = -1;
        int rc = rocket_matmul_plan_int8_rk3576(M, K, 1 << 18, &Mt, &Kt, &Nt);
        checked++;
        if (rc <= 0 || Mt != 51 || Nt != 32) {
            printf("FAIL [job count] rk3576 M=%d K=%d N=%d -> rc=%d (Mt=%d Nt=%d): not the "
                   "51-row, 32-channel plan the INT_MAX edge below is computed from\n",
                   M, K, 1 << 18, rc, Mt, Nt);
            failed++;
        }
        if (r76_plan("job count at INT_MAX - 32767", M, K, 1 << 20, 1) != 2147450880) {
            printf("FAIL [job count] rk3576 M=%d K=%d N=%d did not plan 2147450880 jobs\n",
                   M, K, 1 << 20);
            failed++;
        }
        r76_plan("job count past INT_MAX", M, K, (1 << 20) + 32, ROCKET_E_TILING);
        r76_plan("job count far past INT_MAX", M, K, 1 << 24, ROCKET_E_TILING);
    }

    /* 8. */
    r76_plan("M past ROCKET_MM_DIM_MAX", 1000000007, 32, 32, ROCKET_E_SHAPE);
    r76_plan("M at INT_MAX", INT_MAX, 32, 32, ROCKET_E_SHAPE);
    r76_plan("N past ROCKET_MM_DIM_MAX", 512, 32, (1 << 24) + 32, ROCKET_E_SHAPE);
    r76_plan("K past ROCKET_MM_DIM_MAX", 512, (1 << 24) + 32, 32, ROCKET_E_SHAPE);
    r76_plan("plane past 16 bits", 65537 * 2, 4096, 32, ROCKET_E_SHAPE);

    rocket_log_set_callback(NULL, NULL);
    printf("matmul_plan_gate rk3576: %ld plans checked (%d planned in the sweep), %ld "
           "failures, %.3f s CPU for the prime M plans\n", checked, planned, failed, spent);
    return failed ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "rk3576") == 0) return rk3576_main();
    if (argc > 1) {
        fprintf(stderr, "usage: matmul_plan_gate [rk3576]\n");
        return 1;
    }

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
