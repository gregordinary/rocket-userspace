// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * crossop_chain_rocket.c — does one fp16 matmul's NPU output cube feed the NEXT
 * fp16 matmul's input DIRECTLY, with no host de-tile/re-tile of the intermediate?
 *
 * This is the load-bearing experiment for CROSS-OP submit chaining (chaining
 * DIFFERENT graph ops, not tiles of one op). The tile-layout reference says the
 * fp16 input feature cube and the fp16-narrowed output cube are the SAME layout —
 * both `feat_idx`, C2=8, `(dim/8, M, 8)` (rocket_matmul.c: feat_idx is commented
 * "input/output cube"). So for a chain D = (X·W1^T)·W2^T the intermediate
 * C1 = X·W1^T, once the DPU has written it into an output BO, should ALREADY be in
 * the exact byte layout the second matmul's CNA wants for its input — letting the
 * second matmul read the first's output BO at the same IOVA, with the host never
 * touching the M×N intermediate.
 *
 * "Should" is a layout-index argument; the hardware is the authority. This gate
 * proves it three ways, on the single-tile shape that isolates the cube question
 * from the multi-tile arrangement question (N a multiple of 32 so the output's
 * Nt-pad-to-16 slot equals the next input's Kt-pad-to-32 slot — see the slot
 * formulas in rocket_matmul.c):
 *
 *   1. BYTE EQUIVALENCE — the first matmul's raw output BO == the de-tiled
 *      intermediate re-packed as the second matmul's input (memcmp, off the NPU's
 *      own write). Proves the DPU writes exactly what the CNA reads.
 *   2. ALIASED COMPUTE — run the second matmul with its input BO ALIASED to the
 *      first's output BO (the real cross-op handoff: same IOVA, zero host touch of
 *      the intermediate) and compare to the host-round-trip reference. Bit-exact.
 *   3. ORACLE — both NPU paths vs an fp64 reference (cosine), so a layout that is
 *      self-consistent but wrong is still caught.
 *
 * Sentinel-classified per the RE method: the second output is pre-filled with a
 * sentinel, so "unwritten" (engine declined) is distinct from "wrong" (ran,
 * mis-read) and "exact".
 *
 *   sudo -E ./crossop_chain_rocket            # default single-tile shape
 *   sudo -E ./crossop_chain_rocket 64 128 64 96
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_matmul_internal.h"

static _Float16 x_of(size_t i) { return (_Float16)(((int)((i * 7) % 13) - 6) * 0.05f); }
static _Float16 w1_of(size_t i){ return (_Float16)(((int)((i * 5) % 11) - 5) * 0.04f); }
static _Float16 w2_of(size_t i){ return (_Float16)(((int)((i * 3) % 9 ) - 4) * 0.06f); }

/* cosine similarity of an fp16 result vs an fp64 reference. Skips any non-finite
 * lane (an fp16 overflow->inf would otherwise NaN the whole score) and reports the
 * count, so a layout that is bit-exact but range-overflowed is still legible. */
static double cosine(const _Float16 *got, const double *ref, size_t n, size_t *nonfinite) {
    double dot = 0, ng = 0, nr = 0; size_t nf = 0;
    for (size_t i = 0; i < n; i++) {
        double g = (double)got[i], r = ref[i];
        if (!isfinite(g) || !isfinite(r)) { nf++; continue; }
        dot += g * r; ng += g * g; nr += r * r;
    }
    if (nonfinite) *nonfinite = nf;
    if (ng == 0 || nr == 0) return (ng == 0 && nr == 0) ? 1.0 : 0.0;
    return dot / (sqrt(ng) * sqrt(nr));
}

int main(int argc, char **argv) {
    /* A: C1[M,N] = X[M,K]·W1[N,K]^T ; B: D[M,N2] = C1[M,N]·W2[N2,N]^T.
     * Defaults keep every op single-tile (<=256) and make N a multiple of 32 so
     * out_slot(A)=rup(Mt,4)*rup(N,16) == in_slot(B)=rup(Mt,4)*rup(N,32). */
    int M  = argc >= 5 ? atoi(argv[1]) : 64;
    int K  = argc >= 5 ? atoi(argv[2]) : 128;
    int N  = argc >= 5 ? atoi(argv[3]) : 64;    /* == B's contraction dim K' */
    int N2 = argc >= 5 ? atoi(argv[4]) : 96;
    printf("cross-op chain probe: D[%d,%d] = (X[%d,%d]·W1[%d,%d]^T)·W2[%d,%d]^T  (intermediate C1[%d,%d])\n",
           M, N2, M, K, N, K, N2, N, M, N);

    if (N % 32 != 0) {
        fprintf(stderr, "N must be %%32 (slot-match precondition); got %d\n", N);
        return 1;
    }

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no NPU (rocket_open failed) -> SKIP\n"); return 2; }

    /* ---- inputs ---- */
    _Float16 *X  = malloc((size_t)M * K * sizeof(_Float16));
    _Float16 *W1 = malloc((size_t)N * K * sizeof(_Float16));
    _Float16 *W2 = malloc((size_t)N2 * N * sizeof(_Float16));
    for (size_t i = 0; i < (size_t)M * K;  i++) X[i]  = x_of(i);
    for (size_t i = 0; i < (size_t)N * K;  i++) W1[i] = w1_of(i);
    for (size_t i = 0; i < (size_t)N2 * N; i++) W2[i] = w2_of(i);

    /* ---- fp64 oracle ---- */
    double *C1ref = malloc((size_t)M * N  * sizeof(double));
    double *Dref  = malloc((size_t)M * N2 * sizeof(double));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double a = 0;
            for (int k = 0; k < K; k++) a += (double)X[(size_t)m*K+k] * (double)W1[(size_t)n*K+k];
            C1ref[(size_t)m*N+n] = a;
        }
    for (int m = 0; m < M; m++)
        for (int j = 0; j < N2; j++) {
            double a = 0;
            for (int n = 0; n < N; n++) a += C1ref[(size_t)m*N+n] * (double)W2[(size_t)j*N+n];
            Dref[(size_t)m*N2+j] = a;
        }

    /* ---- reference NPU path: two independent matmuls, host round-trip ---- */
    _Float16 *C1   = malloc((size_t)M * N  * sizeof(_Float16));
    _Float16 *Dref_npu = malloc((size_t)M * N2 * sizeof(_Float16));
    if (rocket_matmul_fp16(fd, M, K, N,  X,  W1, C1) != 0 ||
        rocket_matmul_fp16(fd, M, N, N2, C1, W2, Dref_npu) != 0) {
        fprintf(stderr, "reference matmul path failed\n"); return 1;
    }

    /* ---- chained NPU path: B reads A's OUTPUT BO directly as its INPUT BO ---- */
    mm_plan plA, plB;
    if (mm_plan_init(&plA, M, K, N) < 0 || mm_plan_init(&plB, M, N, N2) < 0) {
        fprintf(stderr, "plan init failed\n"); return 1;
    }
    /* require single-tile so the raw BO is one contiguous cube (multi-tile tile
     * arrangement is the next gate, not this one). */
    if (plA.nMt*plA.nNt*plA.nKt != 1 || plB.nMt*plB.nNt*plB.nKt != 1) {
        fprintf(stderr, "shape not single-tile (A:%dx%dx%d B:%dx%dx%d) -- pick smaller dims\n",
                plA.nMt,plA.nNt,plA.nKt, plB.nMt,plB.nNt,plB.nKt);
        return 1;
    }

    mm_bos bosA, bosB;
    if (mm_bos_alloc(fd, &plA, &bosA) < 0 || mm_bos_alloc(fd, &plB, &bosB) < 0) {
        fprintf(stderr, "bos alloc failed\n"); return 1;
    }

    /* matmul A: pack X + W1, compute. C1chk de-tiled; bosA.out_all holds the raw cube. */
    _Float16 *C1chk = malloc((size_t)M * N * sizeof(_Float16));
    mm_pack_input(fd, &plA, &bosA, X);
    mm_pack_weights(fd, &plA, &bosA, W1);
    if (mm_compute(fd, &plA, &bosA, C1chk, 0.0) != 0) { fprintf(stderr,"A compute failed\n"); return 1; }

    /* CHECK 1 — byte equivalence: A's raw output cube == C1chk re-packed as B's input. */
    size_t nin = mm_input_elems(&plB);
    _Float16 *packedB = calloc(nin, sizeof(_Float16));
    mm_pack_input_buf(&plB, packedB, C1chk);
    int byte_equal = memcmp(packedB, bosA.out_all.ptr, nin * sizeof(_Float16)) == 0;
    /* count differing fp16 lanes for diagnostics */
    size_t ndiff = 0; const _Float16 *raw = (const _Float16 *)bosA.out_all.ptr;
    for (size_t i = 0; i < nin; i++) if (packedB[i] != raw[i]) ndiff++;

    /* matmul B: pack W2, then ALIAS its input BO to A's output BO and compute. */
    _Float16 *Dchain = malloc((size_t)M * N2 * sizeof(_Float16));
    for (size_t i = 0; i < (size_t)M*N2; i++) ((uint16_t*)Dchain)[i] = 0xAAAA; /* sentinel */
    mm_pack_weights(fd, &plB, &bosB, W2);
    rocket_bo in_saved = bosB.in_all;
    bosB.in_all = bosA.out_all;                 /* <-- the cross-op handoff (same IOVA) */
    int rcB = mm_compute(fd, &plB, &bosB, Dchain, 0.0);
    bosB.in_all = in_saved;                      /* restore so mm_bos_free owns it */
    if (rcB != 0) { fprintf(stderr, "B (aliased) compute failed\n"); return 1; }

    /* sentinel scan: any element still 0xAAAA == unwritten */
    size_t unwritten = 0;
    for (size_t i = 0; i < (size_t)M*N2; i++) if (((uint16_t*)Dchain)[i] == 0xAAAA) unwritten++;

    /* CHECK 2 — aliased vs host-round-trip reference: bit-exact expected. */
    double max_abs_vs_ref = 0; size_t nbad = 0;
    for (size_t i = 0; i < (size_t)M*N2; i++) {
        double d = fabs((double)Dchain[i] - (double)Dref_npu[i]);
        if (d > max_abs_vs_ref) max_abs_vs_ref = d;
        if (Dchain[i] != Dref_npu[i]) nbad++;
    }

    /* CHECK 3 — both NPU paths vs fp64 oracle (cosine). */
    size_t nf_ref = 0, nf_chain = 0;
    double cos_ref   = cosine(Dref_npu, Dref, (size_t)M*N2, &nf_ref);
    double cos_chain = cosine(Dchain,   Dref, (size_t)M*N2, &nf_chain);

    printf("CHECK1 byte-equiv : A.out_all == C1 repacked-as-B.in : %s (%zu/%zu lanes differ)\n",
           byte_equal ? "EXACT" : "DIFFER", ndiff, nin);
    printf("CHECK2 aliased    : D_chain vs D_ref(round-trip) : nbad=%zu max_abs=%.6f unwritten=%zu -> %s\n",
           nbad, max_abs_vs_ref, unwritten,
           (unwritten ? "UNWRITTEN" : (nbad == 0 ? "BIT-EXACT" : "MISMATCH")));
    printf("CHECK3 oracle     : cos(ref)=%.6f cos(chain)=%.6f (nonfinite ref=%zu chain=%zu)\n",
           cos_ref, cos_chain, nf_ref, nf_chain);

    /* The proof is CHECK1 (DPU writes exactly what the CNA reads) + CHECK2 (the
     * aliased BO-to-BO compute is bit-identical to the host round-trip). The oracle
     * guards against a self-consistent-but-wrong layout; require it finite + ~1. */
    int pass = byte_equal && nbad == 0 && unwritten == 0
            && nf_chain == 0 && cos_chain > 0.9999;
    printf("RESULT: %s\n", pass ? "PASS (cross-op on-chip chaining is bit-exact)"
                                : "FAIL (intermediate cannot be passed BO-to-BO)");

    mm_bos_free(fd, &bosA); mm_bos_free(fd, &bosB);
    free(X); free(W1); free(W2); free(C1ref); free(Dref); free(C1); free(Dref_npu);
    free(C1chk); free(packedB); free(Dchain);
    rocket_close(fd);
    return pass ? 0 : 3;
}
