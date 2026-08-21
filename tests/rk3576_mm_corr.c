// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_mm_corr.c — does an arbitrary int8 matmul shape lose surface, and how often?
 *
 * The RK3576's drain deadline — the driver retiring a job while the DPU is still
 * writing — reaches the matmul entry, and its rate is a per-shape, intermittent
 * quantity: one prefill shape read 12 wrong reps of 14 at the shipped
 * `dpu_grace_us`=500 while six others read 0 of 5, and 0 of 5 bounds nothing about a
 * hazard that intermittent. This probe scores EVERY repetition of an arbitrary shape
 * so a rate can be read instead of a single sample, and it reports where in the output
 * the wrong elements sit, because a dropped write's signature is a row tail of the
 * trailing output-channel group rather than a scatter.
 *
 * The instrument for the hazard is the module parameter, not the repetition count:
 * `dpu_grace_us` dials the cause. On the SHIPPED module 0 retires on the next tick
 * (maximum exposure) and a large value waits (the cure); on a kernel carrying the
 * drain-deadline patch 0 means WAIT instead, so read
 * `dmesg | grep "Initialized rocket"` before believing either end. A correctness A/B
 * over an intermittent hazard has to be INTERLEAVED — the control that earns the claim
 * is the failure coming back when the parameter goes back, more than once.
 *
 * The host buffer is stamped with a sentinel before every repetition, so an element the
 * entry never wrote is distinguishable from one it wrote wrongly; without the stamp a
 * reused buffer hands back the previous repetition's correct value and the hazard is
 * invisible. Both counts are reported.
 *
 * WHAT THIS PROBE CAN AND CANNOT SCORE, because the two are very far apart and the
 * difference is not visible in its output.
 *
 * The default generators are periodic in the FLATTENED index — period 17 for A and 13
 * for B — so the reference depends only on (m mod 17, n mod 13) and takes at most 221
 * distinct values however large M and N are. Worse, each 221 consecutive k covers every
 * residue pair exactly once and both generators are zero-mean over a full period, so
 * every whole 221-term block of the contraction sums to EXACTLY zero and only K mod 221
 * terms survive. Enumerated over all 221 residues of K — which is every K there is —
 * the reference never leaves +-1 of zero at any of the 221 cells. **The expected output
 * surface is identically zero, at every shape.**
 *
 * So this probe cannot tell a datapath that computed the right answer from one that
 * wrote zeros, and it cannot see any wrong value of magnitude below 2. What it detects
 * well is the thing it was built for — a DROPPED write — and only because the library
 * stamps the output BO with 0xA5 by default, which reads back as -91 against a reference
 * of 0. That makes drop detection exact: every dropped element scores wrong.
 *
 * The dependency is worth stating because it is invisible and it is a debug knob:
 * **with ROCKET_RK3576_I32_SENTINEL=0 the output BO is zero-filled instead, a dropped
 * element reads back 0, the reference is 0, and this probe reports a clean run for a
 * datapath that wrote nothing at all.** It prints the sentinel's state for that reason.
 * `tools/rk3576-mm-corr-refmap.py` and `-kmap.py` derive all of the above off-device.
 *
 * This is a PROBE, not a gate: it exits 0 whenever the shape ran, whatever the surface
 * said. It refuses only when the entry refuses or the shape will not allocate.
 *
 * Run with `sudo -E`, pinned, one shape per process:
 *   sudo -E taskset -c 4 ./build/rk3576_mm_corr 256 2048 2048 5
 *
 * ROCKET_MM_CORR_ROWMAP=1 adds one line per losing row — its lowest and highest wrong
 * column and the count — which is what tells a diagonal front from a half-full rectangle.
 * The bounding box and the fill cannot: 0.49-0.56 is consistent with both. Off by default
 * because a wide loss prints hundreds of lines.
 *
 * The CPU reference is computed once, so the per-repetition cost is the matmul plus an
 * M*N compare; a large M pays a few seconds up front.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

#define SENTINEL ((int8_t)0x5A)          /* host-side: the de-scatter never ran */
#define DEV_STAMP ((int8_t)0xA5)         /* library-side: the DPU never wrote this atom */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 256;
    int K = argc > 2 ? atoi(argv[2]) : 2048;
    int N = argc > 3 ? atoi(argv[3]) : 2048;
    int reps = argc > 4 ? atoi(argv[4]) : 5;

    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_mm_corr: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    if (M <= 0 || K <= 0 || N <= 0 || reps <= 0) {
        fprintf(stderr, "usage: rk3576_mm_corr M K N reps\n");
        return 1;
    }

    int fd = rocket_open();
    if (fd < 0) {
        fprintf(stderr, "rk3576_mm_corr: rocket_open failed (%d)\n", fd);
        return 1;
    }

    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)N * K);
    int8_t *C = malloc((size_t)M * N);
    int8_t *ref = malloc((size_t)M * N);
    if (!A || !B || !C || !ref) {
        fprintf(stderr, "rk3576_mm_corr: out of memory for %dx%dx%d\n", M, K, N);
        return 1;
    }

    for (size_t i = 0; i < (size_t)M * K; i++)
        A[i] = (int8_t)((i * 7 + 3) % 17 - 8);
    for (size_t i = 0; i < (size_t)N * K; i++)
        B[i] = (int8_t)((i * 5 + 1) % 13 - 6);

    const float scale = 1.0f / 512.0f;

    for (int m = 0; m < M; m++) {
        const int8_t *a = A + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *b = B + (size_t)n * K;
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)a[k] * (int32_t)b[k];
            float f = acc * scale;
            int want = (int)(f < 0 ? f - 0.5f : f + 0.5f);
            if (want > 127) want = 127;
            if (want < -128) want = -128;
            ref[(size_t)m * N + n] = (int8_t)want;
        }
    }

    /* Declare what this probe's discriminating power rests on, because both halves are
     * invisible in the wrong-element count. The reference's span says whether the
     * surface can witness a wrong VALUE at all; the sentinel says whether it can
     * witness a dropped write. Same rule as the library's rocket_rk3576_sentinel_on(). */
    {
        int rmin = 127, rmax = -128;
        const char *e = getenv("ROCKET_RK3576_I32_SENTINEL");
        int stamped = (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 1;
        for (size_t i = 0; i < (size_t)M * N; i++) {
            if (ref[i] < rmin) rmin = ref[i];
            if (ref[i] > rmax) rmax = ref[i];
        }
        int degenerate = (rmin >= -1 && rmax <= 1);
        printf("reference span %d..%d%s; output BO stamp %s\n", rmin, rmax,
               degenerate
                   ? " (DEGENERATE: within tolerance of zero everywhere, so a wrong"
                     " VALUE is unscorable and only a dropped write can be seen)" : "",
               stamped ? "0xA5 (drops are scorable)"
                       : "OFF -- A DROPPED WRITE READS BACK 0 AND SCORES CLEAN");
        /* Both halves gone is the one configuration with NO discriminating power at all:
         * a degenerate reference cannot witness a wrong value, and an unstamped BO cannot
         * witness a dropped one, so every rep scores clean whatever the datapath did --
         * including writing nothing. Refuse rather than print that as a result. The
         * geometry gates (offset_cube, surf_stride, conv_pitch) take the same line when
         * their reference surface comes back entirely sentinel. */
        if (degenerate && !stamped) {
            fprintf(stderr, "rk3576_mm_corr: reference is degenerate AND the output stamp "
                    "is off -- this probe cannot distinguish a correct datapath from one "
                    "that wrote nothing. Unset ROCKET_RK3576_I32_SENTINEL.\n");
            return 1;
        }
    }

    const char *rm = getenv("ROCKET_MM_CORR_ROWMAP");
    int rowmap = (rm && *rm) ? (int)strtol(rm, NULL, 0) != 0 : 0;

    /* Warm: the clock parks at idle, and the first call also faults in the mappings. */
    for (int i = 0; i < 2; i++) {
        int rc = rocket_matmul_int8_rk3576(fd, M, K, N, A, B, NULL, scale, C);
        if (rc != 0) {
            fprintf(stderr, "rk3576_mm_corr: entry refused (%d) for %dx%dx%d\n",
                    rc, M, K, N);
            return 1;
        }
    }

    int wrong_reps = 0, refused_reps = 0;
    long long total_wrong = 0;

    for (int r = 0; r < reps; r++) {
        memset(C, SENTINEL, (size_t)M * N);

        double t0 = now_ms();
        int rc = rocket_matmul_int8_rk3576(fd, M, K, N, A, B, NULL, scale, C);
        double ms = now_ms() - t0;

        if (rc != 0) {
            printf("rep %d: REFUSED rc %d  %.3f ms\n", r, rc, ms);
            refused_reps++;
            continue;
        }

        long long wrong = 0, unwritten = 0, dropped = 0, zeroed = 0, computed = 0;
        int m_lo = M, m_hi = -1, n_lo = N, n_hi = -1;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                int got = C[(size_t)m * N + n];
                int want = ref[(size_t)m * N + n];
                if (got >= want - 1 && got <= want + 1)
                    continue;
                wrong++;
                /* A count is not a mechanism. What the wrong element HOLDS separates a
                 * write the DPU never made from one it made wrongly, and the two have
                 * different cures: the stamp means the job retired while the block was
                 * still writing, an arbitrary value means the datapath computed it. */
                if (got == SENTINEL)
                    unwritten++;       /* de-scatter never ran over this element */
                else if (got == DEV_STAMP)
                    dropped++;         /* DPU never wrote this atom */
                else if (got == 0)
                    zeroed++;          /* written, but written as zero */
                else
                    computed++;        /* written with a wrong value */
                if (m < m_lo) m_lo = m;
                if (m > m_hi) m_hi = m;
                if (n < n_lo) n_lo = n;
                if (n > n_hi) n_hi = n;
            }
        }

        if (wrong) {
            wrong_reps++;
            total_wrong += wrong;
            printf("rep %d: %lld of %lld wrong (drop %lld, zero %lld, computed %lld,"
                   " no-descatter %lld), rows %d..%d, cols %d..%d, %.3f ms\n",
                   r, wrong, (long long)M * N, dropped, zeroed, computed, unwritten,
                   m_lo, m_hi, n_lo, n_hi, ms);
            /* A bounding box and a count cannot tell a triangle from a half-full
             * rectangle, and on this entry the fill is 0.49-0.56 every time — which is
             * what a straight diagonal cut gives, and also what plenty of other shapes
             * give. ROCKET_MM_CORR_ROWMAP=1 prints the per-row column extent so the front
             * is read rather than inferred. One line per row that lost anything: the
             * lowest and highest wrong column and how many. */
            if (rowmap) {
                for (int m = m_lo; m <= m_hi; m++) {
                    int lo = N, hi = -1;
                    long long cnt = 0;
                    for (int n = 0; n < N; n++) {
                        int got = C[(size_t)m * N + n];
                        int want = ref[(size_t)m * N + n];
                        if (got >= want - 1 && got <= want + 1)
                            continue;
                        if (n < lo) lo = n;
                        if (n > hi) hi = n;
                        cnt++;
                    }
                    if (cnt)
                        printf("  rowmap rep %d row %d: cols %d..%d, %lld wrong\n",
                               r, m, lo, hi, cnt);
                }
            }
        } else {
            printf("rep %d: clean, %.3f ms\n", r, ms);
        }
    }

    printf("MMCORR M %d K %d N %d reps %d wrong_reps %d refused %d total_wrong %lld\n",
           M, K, N, reps, wrong_reps, refused_reps, total_wrong);

    free(A); free(B); free(C); free(ref);
    rocket_close(fd);
    return 0;
}
