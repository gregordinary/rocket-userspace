// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * reuse_follower_rocket.c — score the job that runs AFTER a reuse chain, and the LUT
 * job that runs after a different LUT job.
 *
 * allbilly/rk3588 reports two state-hygiene failures on this silicon. After a two-task
 * CNA spatial split with WEIGHT_REUSE, the first following known-good DPU job read stale
 * data and a second job recovered. And LUT tables written by one submit were not there
 * for the next submit that did not write them. A gate that scores only the reusing job
 * cannot see the first failure, because the reusing job itself is exact; so this gate
 * scores the FOLLOWER.
 *
 *   part 1  A matmul whose plan has several M, N and K tiles, so a reuse run and the
 *           K-accumulation are live under this process's ROCKET_REUSE / ROCKET_KACC.
 *           Straight after it, two followers: a small integer matmul (exact in fp16, so
 *           it is scored against the CPU on every element) and a sigmoid LUT job (scored
 *           against its own bytes from a solo run). The followers swap order every
 *           trial, so each takes the first-follower slot half the time.
 *   part 2  sigmoid, tanh, sigmoid as three separate submits, each scored against its
 *           solo bytes. Every LUT program here writes both tables into its own regcmd,
 *           so this holds that property rather than probing for a cache.
 *
 * ctest runs it under the default (AUTO reuse), ROCKET_REUSE=1 (WEIGHT) and
 * ROCKET_REUSE=2 (DATA). The chain's own output is scored too, exactly while every
 * reference element stays inside fp16's integer range.
 *
 *   reuse_follower_rocket [trials]   (default 20)
 * Exit 0 pass, 1 fail, 2 skip (no device).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_activation.h"
#include "rocket_matmul.h"
#include "rocket_npu.h"
#include "test_fill.h"

/* The chain: several tiles on every axis at the default plan. */
#define BM 512
#define BK 2048
#define BN 512
/* The follower matmul: one tile, integers in [-3, 3], |C| <= 576. */
#define FM 64
#define FK 64
#define FN 64
/* The LUT jobs. */
#define LN 4096

static void ref_mm(int M, int K, int N, const _Float16 *A, const _Float16 *B, float *C)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++)
                s += (double)(float)A[(size_t)m * K + k] * (double)(float)B[(size_t)n * K + k];
            C[(size_t)m * N + n] = (float)s;
        }
}

/* Elements of got that differ from ref (compared as fp16 values; a NaN never matches). */
static long count_bad_mm(const _Float16 *got, const float *ref, size_t n)
{
    long bad = 0;
    for (size_t i = 0; i < n; i++)
        if (!((float)got[i] == ref[i])) bad++;
    return bad;
}

static long count_bad_bytes(const _Float16 *got, const _Float16 *want, size_t n)
{
    long bad = 0;
    for (size_t i = 0; i < n; i++)
        if (memcmp(&got[i], &want[i], sizeof got[i])) bad++;
    return bad;
}

int main(int argc, char **argv)
{
    int trials = argc > 1 ? atoi(argv[1]) : 20;
    if (trials < 1) trials = 1;

    int fd = rocket_open();
    if (fd < 0) {
        printf("no NPU (%d) -> SKIP\n", fd);
        return 2;
    }

    const char *reuse = getenv("ROCKET_REUSE"), *kacc = getenv("ROCKET_KACC");
    int Mt, Kt, Nt;
    int njobs = rocket_matmul_plan(BM, BK, BN, &Mt, &Kt, &Nt);
    if (njobs < 0) {
        printf("chain shape %dx%dx%d unsupported (%d) -> FAIL\n", BM, BK, BN, njobs);
        rocket_close(fd);
        return 1;
    }
    int nMt = (BM + Mt - 1) / Mt, nKt = (BK + Kt - 1) / Kt, nNt = (BN + Nt - 1) / Nt;
    printf("ROCKET_REUSE=%s ROCKET_KACC=%s; chain %dx%dx%d plans Mt %d Kt %d Nt %d "
           "(%d x %d x %d tiles, %d jobs)\n", reuse ? reuse : "(unset)",
           kacc ? kacc : "(unset)", BM, BK, BN, Mt, Kt, Nt, nMt, nKt, nNt, njobs);
    if (nMt < 2 || nNt < 2 || nKt < 2) {
        printf("the chain does not tile every axis, so no reuse run is guaranteed -> FAIL "
               "(pick a larger shape)\n");
        rocket_close(fd);
        return 1;
    }

    _Float16 *A = malloc((size_t)BM * BK * 2), *B = malloc((size_t)BN * BK * 2);
    _Float16 *C = malloc((size_t)BM * BN * 2);
    float *Cref = malloc((size_t)BM * BN * sizeof(float));
    _Float16 *fA = malloc(FM * FK * 2), *fB = malloc(FN * FK * 2), *fC = malloc(FM * FN * 2);
    float *fref = malloc(FM * FN * sizeof(float));
    _Float16 *x = malloc(LN * 2), *y = malloc(LN * 2);
    _Float16 *sig = malloc(LN * 2), *tnh = malloc(LN * 2);
    if (!A || !B || !C || !Cref || !fA || !fB || !fC || !fref || !x || !y || !sig || !tnh) {
        fprintf(stderr, "oom\n");
        return 1;
    }

    /* The chain in {-1, 0, 1}: exact while every |reference| < 2048, checked below. */
    tf_fill_f16_int(A, (size_t)BM * BK, tf_hash(0x6733, 1), -1, 1);
    tf_fill_f16_int(B, (size_t)BN * BK, tf_hash(0x6733, 2), -1, 1);
    ref_mm(BM, BK, BN, A, B, Cref);
    double peak = 0;
    for (size_t i = 0; i < (size_t)BM * BN; i++)
        if (fabs(Cref[i]) > peak) peak = fabs(Cref[i]);
    const int chain_exact = peak < 2048.0;
    printf("chain max|ref| %.0f -> scored %s\n", peak,
           chain_exact ? "exactly" : "for determinism only (fp16 cannot hold it)");

    tf_fill_f16_int(fA, FM * FK, tf_hash(0x6733, 3), -3, 3);
    tf_fill_f16_int(fB, FN * FK, tf_hash(0x6733, 4), -3, 3);
    ref_mm(FM, FK, FN, fA, fB, fref);

    for (int i = 0; i < LN; i++) x[i] = (_Float16)(-8.0 + 16.0 * i / (LN - 1));

    /* Solo LUT bytes: each activation twice in a row must agree with itself first. */
    int rc_solo = 0;
    tf_sentinel_f16(sig, LN);
    rc_solo |= rocket_activation_fp16(fd, ROCKET_ACTIVATION_SIGMOID, x, sig, LN);
    tf_sentinel_f16(y, LN);
    rc_solo |= rocket_activation_fp16(fd, ROCKET_ACTIVATION_SIGMOID, x, y, LN);
    long solo_sig = count_bad_bytes(y, sig, LN);
    tf_sentinel_f16(tnh, LN);
    rc_solo |= rocket_activation_fp16(fd, ROCKET_ACTIVATION_TANH, x, tnh, LN);
    tf_sentinel_f16(y, LN);
    rc_solo |= rocket_activation_fp16(fd, ROCKET_ACTIVATION_TANH, x, y, LN);
    long solo_tnh = count_bad_bytes(y, tnh, LN);
    long sent = 0;
    for (int i = 0; i < LN; i++) sent += tf_is_sentinel_f16(sig[i]) + tf_is_sentinel_f16(tnh[i]);
    printf("solo: rc %d, sigmoid twice differs on %ld, tanh twice on %ld, unwritten %ld\n",
           rc_solo, solo_sig, solo_tnh, sent);
    if (rc_solo || solo_sig || solo_tnh || sent) {
        printf("the solo LUT runs do not agree with themselves, so nothing after can be "
               "scored against them -> FAIL\n");
        rocket_close(fd);
        return 1;
    }

    /* Part 1. */
    long chain_bad = 0, chain_nondet = 0, f_mm_bad[2] = {0, 0}, f_lut_bad[2] = {0, 0};
    int rc_bad = 0, f_mm_trials_bad = 0, f_lut_trials_bad = 0;
    _Float16 *C0 = malloc((size_t)BM * BN * 2);
    if (!C0) return 1;
    for (int t = 0; t < trials; t++) {
        tf_sentinel_f16(C, (size_t)BM * BN);
        int r = rocket_matmul_fp16(fd, BM, BK, BN, A, B, C);
        const int lut_first = t & 1;
        long bm = 0, bl = 0;
        for (int step = 0; step < 2; step++) {
            if ((step == 0) == lut_first) {
                tf_sentinel_f16(y, LN);
                r |= rocket_activation_fp16(fd, ROCKET_ACTIVATION_SIGMOID, x, y, LN);
                bl = count_bad_bytes(y, sig, LN);
                f_lut_bad[step] += bl;
            } else {
                tf_sentinel_f16(fC, FM * FN);
                r |= rocket_matmul_fp16(fd, FM, FK, FN, fA, fB, fC);
                bm = count_bad_mm(fC, fref, FM * FN);
                f_mm_bad[step] += bm;
            }
        }
        if (r) rc_bad++;
        if (bm) f_mm_trials_bad++;
        if (bl) f_lut_trials_bad++;
        long cb = chain_exact ? count_bad_mm(C, Cref, (size_t)BM * BN) : 0;
        chain_bad += cb;
        if (t == 0) memcpy(C0, C, (size_t)BM * BN * 2);
        else chain_nondet += count_bad_bytes(C, C0, (size_t)BM * BN);
        if (r || bm || bl || cb)
            printf("  trial %2d (%s first): rc %d, chain %ld wrong, follower matmul %ld of %d "
                   "wrong, follower sigmoid %ld of %d differ\n", t, lut_first ? "LUT" : "matmul",
                   r, cb, bm, FM * FN, bl, LN);
    }
    printf("part 1: %d trials; rc!=0 in %d; chain %ld elements wrong, %ld differ from trial 0; "
           "follower matmul wrong in %d trials (%ld first-slot, %ld second-slot elements); "
           "follower sigmoid wrong in %d trials (%ld first-slot, %ld second-slot elements)\n",
           trials, rc_bad, chain_bad, chain_nondet, f_mm_trials_bad, f_mm_bad[0], f_mm_bad[1],
           f_lut_trials_bad, f_lut_bad[0], f_lut_bad[1]);

    /* Part 2. */
    long lut_bad = 0;
    int lut_rc = 0;
    for (int t = 0; t < trials; t++) {
        static const int kinds[3] = {ROCKET_ACTIVATION_SIGMOID, ROCKET_ACTIVATION_TANH,
                                     ROCKET_ACTIVATION_SIGMOID};
        for (int j = 0; j < 3; j++) {
            tf_sentinel_f16(y, LN);
            lut_rc |= rocket_activation_fp16(fd, kinds[j], x, y, LN);
            long b = count_bad_bytes(y, kinds[j] == ROCKET_ACTIVATION_TANH ? tnh : sig, LN);
            if (b) printf("  rotation %2d step %d: %ld of %d differ from solo\n", t, j, b, LN);
            lut_bad += b;
        }
    }
    printf("part 2: %d rotations of sigmoid, tanh, sigmoid; rc %d; %ld elements differ from "
           "solo\n", trials, lut_rc, lut_bad);

    int fail = rc_bad || chain_bad || f_mm_trials_bad || f_lut_trials_bad || lut_rc || lut_bad;
    printf("%s\n", fail ? "FAIL" : "PASS");
    free(A); free(B); free(C); free(C0); free(Cref); free(fA); free(fB); free(fC); free(fref);
    free(x); free(y); free(sig); free(tnh);
    rocket_close(fd);
    return fail;
}
