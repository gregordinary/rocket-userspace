// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_kacc_chain_rocket.c — gate the cross-ki KACC chaining (ROCKET_KACC_CHAIN).
 *
 * The fp16 NPU-side K-accumulation (ROCKET_KACC, default on) splits a tile's K into
 * nKt partials and EW-adds them on the NPU. By default each ki-step is its own fenced
 * submit (ki>0 reads the prior partial). ROCKET_KACC_CHAIN=1 instead chains the whole
 * [ki][tile] sequence into ONE self-chained kick, so a tile's read-after-write
 * K-accumulation — and the ping-pong write-after-read — must be honored by the HW
 * within a single fence.
 *
 * PRIMARY GATE (HW-RE): chained must equal unchained BYTE-FOR-BYTE. Both paths run the
 * identical fp16 EW-add ops in the identical ki order; only the submit topology differs.
 * So fp16 rounding is irrelevant — any byte mismatch means the chained kick did NOT
 * serialize the cross-task dependency (the RAW/WAR hazard the per-ki fences guarantee),
 * which would make the lever HW-dead. An exact memcmp is the right gate.
 *
 * SECONDARY: both results are within fp16 tolerance of a CPU fp32 reference (absolute
 * correctness; the unchained KACC path is already CPU-ref-gated by matmul_tiled_rocket,
 * so this is a belt-and-suspenders check on the smallest shape).
 *
 * Shapes are chosen to force K-tiling (nKt>1) so chaining actually engages; the
 * smallest also fits one chained kick (the tightest RAW test). Usage:
 *   matmul_kacc_chain_rocket [M K N]   (default: an internal K-tiled suite)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

/* Run one shape: unchained vs chained byte-exact, plus an optional CPU fp32 ref.
 * Returns 0 PASS, 1 FAIL, 2 SKIP (shape unsupported / chaining inactive). */
static int run_one(int fd, int M, int K, int N, int do_ref)
{
    int Mt, Kt, Nt;
    int njobs = rocket_matmul_plan(M, K, N, &Mt, &Kt, &Nt);
    if (njobs < 0) {
        printf("  [%d x %d x %d] unsupported shape -> skip\n", M, K, N);
        return 2;
    }
    int nMt = (M + Mt - 1) / Mt, nNt = (N + Nt - 1) / Nt, nKt = (K + Kt - 1) / Kt;
    /* The chained path packs nKt tasks per tile into one kick; the cap is nKt <= BATCH
     * (64). Larger nKt falls back to the per-ki path (still correct — chained==unchained
     * trivially — but it does not exercise the chaining). */
    const int CHAIN_BATCH = 64;
    printf("  [%d x %d x %d] Mt=%d Kt=%d Nt=%d  tiles=%dx%dx%d  nKt=%d  %s\n",
           M, K, N, Mt, Kt, Nt, nMt, nNt, nKt, nKt,
           nKt < 2 ? "(nKt=1: no K-accum)" :
           nKt > CHAIN_BATCH ? "(nKt>BATCH: per-ki fallback, chaining inactive)" :
                               "CHAINS");
    if (nKt < 2) {
        printf("    nKt=1 -> no K-accumulation; chaining is a no-op here, skipping\n");
        return 2;
    }

    size_t mn = (size_t)M * N;
    _Float16 *A  = malloc((size_t)M * K * sizeof(_Float16));
    _Float16 *B  = malloc((size_t)N * K * sizeof(_Float16));
    _Float16 *C0 = malloc(mn * sizeof(_Float16));   /* unchained */
    _Float16 *C1 = malloc(mn * sizeof(_Float16));   /* chained   */
    float    *R  = do_ref ? malloc(mn * sizeof(float)) : NULL;
    if (!A || !B || !C0 || !C1 || (do_ref && !R)) {
        fprintf(stderr, "    host alloc failed\n");
        free(A); free(B); free(C0); free(C1); free(R);
        return 1;
    }

    /* Small ints keep the fp16 EW-accum away from inf even at large K; the byte-exact
     * comparison does not depend on this, but the CPU-ref tolerance check does. */
    srand(0xC0FFEE ^ (M * 2654435761u) ^ (K * 40503u) ^ N);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (_Float16)(rand() % 3);
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (_Float16)(rand() % 3);

    /* Unchained KACC (per-ki fences). */
    setenv("ROCKET_KACC_CHAIN", "0", 1);
    memset(C0, 0, mn * sizeof(_Float16));
    int r0 = rocket_matmul_fp16(fd, M, K, N, A, B, C0);

    /* Chained KACC, FORCE mode (=2): chain for any fitting nKt regardless of the
     * adaptive win-regime gate, so the correctness test covers the fully-serial gcap=1
     * case (large nKt) — the strictest cross-task RAW test, not just the win regime. */
    setenv("ROCKET_KACC_CHAIN", "2", 1);
    memset(C1, 0, mn * sizeof(_Float16));
    int r1 = rocket_matmul_fp16(fd, M, K, N, A, B, C1);

    int fail = 0;
    if (r0 || r1) {
        fprintf(stderr, "    matmul failed: unchained=%d chained=%d\n", r0, r1);
        fail = 1;
        goto done;
    }

    /* PRIMARY: byte-for-byte identical. */
    long nbyte = 0; size_t first_bad = 0;
    for (size_t i = 0; i < mn; i++) {
        if (memcmp(&C0[i], &C1[i], sizeof(_Float16)) != 0) {
            if (nbyte == 0) first_bad = i;
            nbyte++;
        }
    }
    if (nbyte) {
        printf("    BYTE-EXACT FAIL: %ld/%zu elems differ (first [%zu] unchained=%.3f chained=%.3f)\n",
               nbyte, mn, first_bad, (float)C0[first_bad], (float)C1[first_bad]);
        fail = 1;
    } else {
        printf("    byte-exact: chained == unchained over %zu elems  OK\n", mn);
    }

    /* SECONDARY: chained within fp16 tolerance of the CPU fp32 reference. */
    if (do_ref) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float s = 0.f;
                const _Float16 *a = &A[(size_t)m * K], *b = &B[(size_t)n * K];
                for (int k = 0; k < K; k++) s += (float)a[k] * (float)b[k];
                R[(size_t)m * N + n] = s;
            }
        double max_abs = 0, max_rel = 0; long nbad = 0;
        for (size_t i = 0; i < mn; i++) {
            float got = (float)C1[i], exp = (float)(_Float16)R[i];
            double ad = fabs(got - exp), rd = ad / (fabs(exp) + 1e-6);
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) max_rel = rd;
            if (rd > 0.02 && ad > 1.0) nbad++;
        }
        printf("    cpu-ref: max_abs=%.2f max_rel=%.4f nbad=%ld  %s\n",
               max_abs, max_rel, nbad, nbad ? "FAIL" : "OK");
        if (nbad) fail = 1;
    }

done:
    free(A); free(B); free(C0); free(C1); free(R);
    return fail;
}

int main(int argc, char **argv)
{
    setenv("ROCKET_KACC", "1", 1);   /* force the KACC path on regardless of ambient env */

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); return 2; }

    int fails = 0, ran = 0;
    if (argc == 4) {
        int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
        int r = run_one(fd, M, K, N, 1);
        if (r != 2) { ran++; fails += (r == 1); }
    } else {
        /* Internal K-tiled suite, all with nKt in [2,64] so chaining engages: the
         * tightest cross-ki RAW (nKt=2), a moderate chain, two deeper chains, and a
         * realistic Gemma FFN-down. The big one is byte-exact only (CPU ref too slow). */
        struct { int M, K, N, ref; } S[] = {
            { 256,   768,  256, 1 },   /* nKt=2: tightest cross-ki RAW (ki=0 conv -> ki=1 EW-add) */
            { 256,  4608,  512, 1 },   /* nKt=12: moderate */
            { 256, 16384,  512, 1 },   /* nKt=43: deep chain, one tile-group */
            { 512, 16384, 1024, 1 },   /* nKt=43: multiple tiles per group */
            { 512, 15360, 3840, 0 },   /* nKt=40: Gemma FFN-down (byte-exact only) */
        };
        for (size_t i = 0; i < sizeof(S) / sizeof(S[0]); i++) {
            int r = run_one(fd, S[i].M, S[i].K, S[i].N, S[i].ref);
            if (r != 2) { ran++; fails += (r == 1); }
        }
    }
    rocket_close(fd);

    if (ran == 0) { printf("no K-tiled shape exercised chaining -> SKIP\n"); return 2; }
    printf("\nkacc-chain gate: %d shape(s) run, %d failed -> %s\n",
           ran, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
