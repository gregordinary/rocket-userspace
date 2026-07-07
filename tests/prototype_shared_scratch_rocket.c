// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * prototype_shared_scratch_rocket.c — de-risk the shared-scratch refactor.
 *
 * The blocker: the prepacked path gives EACH resident weight tensor its own
 * in_all/out_all/pong scratch (~67% of the footprint), so the NPU's per-fd 4 GB IOVA
 * (see iova_ceiling_rocket) holds only ~30% of a 12B model. The fix is to keep one
 * resident WEIGHT BO per tensor but SHARE one scratch set per (shape, worker). Before
 * rewiring the backend hot path (resident-BO territory == the gibberish-saga bug), this
 * prototype proves the ownership pattern INLINE on the driver primitives:
 *
 *   - ONE shared `mm_bos` scratch  (mm_bos_alloc: in_all/out_all/pong/regcmd/host arrays)
 *   - N per-weight `wt_all` BOs    (rocket_bo_alloc, sized like the scratch's wt_all)
 *   - compute weight i by ASSEMBLING: scratch.wt_all = wt[i]; mm_compute*(... &scratch).
 *
 * It checks (a) every weight computes CORRECTLY against a CPU ref while sharing scratch,
 * (b) a re-run pass is BIT-IDENTICAL to the first (no cross-weight scratch contamination
 * -- the aliasing failure mode), and (c) how many weights actually fit one fd with shared
 * vs per-tensor scratch. Honors ROCKET_KACC (run it in the operating mode).
 *
 *   ROCKET_KACC=1 sudo -E ./prototype_shared_scratch_rocket            # ffn-down worker slice
 *   ROCKET_KACC=1 sudo -E ./prototype_shared_scratch_rocket 1024 3840 768 16
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul_internal.h"

#define NMAX 64

static _Float16 a_of(size_t i){ return (_Float16)(((i * 7) % 13 - 6) * 0.05f); }
/* distinct weight per index w: pattern shifts with w so a mix-up is caught */
static _Float16 b_of(int w, size_t i){ return (_Float16)((((i * 5 + w * 3) % 11) - 5) * 0.05f); }

int main(int argc, char **argv)
{
    int M = argc>=4 ? atoi(argv[1]) : 1024;
    int K = argc>=4 ? atoi(argv[2]) : 15360;
    int N = argc>=4 ? atoi(argv[3]) : 768;     /* a per-worker N-slice of ffn-down (3840/5) */
    int W = argc>=5 ? atoi(argv[4]) : 16;      /* weights sharing one scratch */
    if (W > NMAX) W = NMAX;
    int kacc = getenv("ROCKET_KACC") != NULL;
    printf("shared-scratch prototype: %d weights share 1 scratch  C[%d,%d]=A[%d,%d]xB[%d,%d]^T  kacc=%d\n",
           W, M, N, M, K, N, K, kacc);

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open failed\n"); return 1; }

    mm_plan pl;
    if (mm_plan_init(&pl, M, K, N) < 0) { fprintf(stderr, "unsupported shape\n"); return 1; }

    /* one shared scratch (this also allocates a wt_all we won't use -- save+restore it
     * so mm_bos_free still owns it; per-weight wt[] are allocated separately below). */
    mm_bos sc;
    if (mm_bos_alloc(fd, &pl, &sc) < 0) { fprintf(stderr, "scratch alloc failed\n"); return 1; }
    const rocket_bo scratch_wt = sc.wt_all;       /* the scratch's own (unused) weight BO */

    const _Float16 *A = malloc((size_t)M*K*sizeof(_Float16));
    _Float16 *Abuf = (_Float16*)A;
    for (size_t i=0;i<(size_t)M*K;i++) Abuf[i] = a_of(i);

    /* CPU reference for the first ref_rows rows of every weight */
    int ref_rows = M < 8 ? M : 8;
    float *ref = malloc((size_t)W*ref_rows*N*sizeof(float));
    _Float16 *Brow = malloc((size_t)N*K*sizeof(_Float16));   /* one weight at a time */
    for (int w=0; w<W; w++) {
        for (size_t i=0;i<(size_t)N*K;i++) Brow[i] = b_of(w, i);
        for (int m=0;m<ref_rows;m++) for (int n=0;n<N;n++){
            float a=0; for (int k=0;k<K;k++) a += (float)Abuf[(size_t)m*K+k]*(float)Brow[(size_t)n*K+k];
            ref[((size_t)w*ref_rows+m)*N+n] = a;
        }
    }

    /* alloc + prezero + pack each weight into its OWN wt BO, all sharing `sc`. */
    rocket_bo wt[NMAX];
    for (int w=0; w<W; w++) {
        if (rocket_bo_alloc(fd, scratch_wt.size, &wt[w]) != 0) { fprintf(stderr,"wt[%d] alloc failed\n",w); return 1; }
        rocket_bo_prep(fd, &wt[w], 1, 0); memset(wt[w].ptr, 0, wt[w].size); rocket_bo_fini(fd, &wt[w]);
        for (size_t i=0;i<(size_t)N*K;i++) Brow[i] = b_of(w, i);
        sc.wt_all = wt[w];                         /* assemble: scratch + this weight */
        mm_pack_weights(fd, &pl, &sc, Brow);       /* sc.prezeroed=1, wt prezeroed above */
    }

    /* PASS 1: compute every weight (sharing sc), verify vs CPU ref, keep each C. */
    _Float16 *C  = malloc((size_t)M*N*sizeof(_Float16));
    _Float16 *C1 = malloc((size_t)W*M*N*sizeof(_Float16));   /* first-pass results */
    float worst_ref = 0, worst_rel = 0; int verify_fail = 0;
    for (int w=0; w<W; w++) {
        sc.wt_all = wt[w];
        double tp = mm_pack_input(fd, &pl, &sc, A);
        int rc = kacc ? mm_compute_kacc(fd, &pl, &sc, C, tp) : mm_compute(fd, &pl, &sc, C, tp);
        if (rc == -2) rc = mm_compute(fd, &pl, &sc, C, tp);  /* tiny-M KACC fallback */
        if (rc) { fprintf(stderr, "compute w=%d failed (%d)\n", w, rc); return 1; }
        memcpy(C1 + (size_t)w*M*N, C, (size_t)M*N*sizeof(_Float16));
        /* abs-OR-rel like matmul_mt: KACC's fp16 running-sum drifts with K magnitude,
         * so a large abs at small rel is fp16 rounding, NOT a shared-scratch bug. */
        float ma=0, mr=0;
        for (int m=0;m<ref_rows;m++) for (int n=0;n<N;n++){
            float want = ref[((size_t)w*ref_rows+m)*N+n];
            float ad = fabsf((float)C[(size_t)m*N+n] - want);
            float rd = ad / (fabsf(want) + 1e-6f);
            if (ad>ma) ma=ad;
            if (rd>mr) mr=rd;
        }
        if (ma > worst_ref) worst_ref = ma;
        if (mr > worst_rel) worst_rel = mr;
        if (ma >= 0.5f && mr >= 0.05f) verify_fail++;
    }

    /* PASS 2: recompute in REVERSE order; must be BIT-IDENTICAL to pass 1 (else the
     * shared scratch is leaking state between weights -- the aliasing failure mode). */
    int alias_fail = 0; float worst_rerun = 0;
    for (int w=W-1; w>=0; w--) {
        sc.wt_all = wt[w];
        double tp = mm_pack_input(fd, &pl, &sc, A);
        int rc = kacc ? mm_compute_kacc(fd, &pl, &sc, C, tp) : mm_compute(fd, &pl, &sc, C, tp);
        if (rc == -2) rc = mm_compute(fd, &pl, &sc, C, tp);
        if (rc) { fprintf(stderr, "re-compute w=%d failed (%d)\n", w, rc); return 1; }
        float md=0;
        for (size_t i=0;i<(size_t)M*N;i++){
            float d = fabsf((float)C[i] - (float)C1[(size_t)w*M*N + i]);
            if (d>md) md=d;
        }
        if (md > worst_rerun) worst_rerun = md;
        if (md != 0.0f) alias_fail++;
    }

    printf("PASS1 correctness : worst vs CPU ref over %d weights: max_abs=%.4f max_rel=%.4f -> %s\n",
           W, worst_ref, worst_rel, verify_fail ? "FAIL" : "PASS");
    printf("PASS2 no-aliasing : worst |rerun-first| = %.4f (expect 0.0000) -> %s\n",
           worst_rerun, alias_fail ? "FAIL (scratch contaminated)" : "PASS (bit-identical)");

    /* ---- footprint + capacity: weight vs scratch, shared vs per-tensor ---- */
    size_t wt_sz = scratch_wt.size;
    size_t sc_sz = sc.in_all.size + sc.out_all.size + sc.regcmd.size + sc.guard.size
                 + (sc.pong.handle ? sc.pong.size : 0);
    const size_t WIN = (size_t)4 << 30;                 /* per-fd 32-bit window */
    long shared_cap = (long)((WIN - sc_sz) / wt_sz);    /* 1 scratch + N weights */
    long pertensor_cap = (long)(WIN / (wt_sz + sc_sz)); /* each weight drags its own scratch */
    printf("footprint/fd      : weight=%zuMB  scratch=%zuMB (in=%zu out=%zu pong=%zu rc=%zu MB)\n",
           wt_sz>>20, sc_sz>>20, sc.in_all.size>>20, sc.out_all.size>>20,
           (sc.pong.handle?sc.pong.size:0)>>20, sc.regcmd.size>>20);
    printf("capacity (calc)   : SHARED %ld weights/fd (%.1fGB) vs PER-TENSOR %ld/fd (%.1fGB) = %.1fx\n",
           shared_cap, shared_cap*(double)wt_sz/(double)(1u<<30),
           pertensor_cap, pertensor_cap*(double)(wt_sz+sc_sz)/(double)(1u<<30),
           pertensor_cap? (double)shared_cap/pertensor_cap : 0.0);

    /* Capacity is answered by the `calc` line above + iova_ceiling's per-fd 4GB window.
     * (An empirical alloc-until-fail probe is intentionally omitted: BOs that are never
     * prep'd/submitted aren't assigned a device IOVA on this path, so a dma_address
     * 32-bit guard never trips and the loop overshoots into host RAM until ENOMEM —
     * which can wedge a swapless box. The math + the dedicated iova probe suffice.) */
    for (int w=0; w<W; w++) rocket_bo_free(fd, &wt[w]);

    sc.wt_all = scratch_wt;          /* restore so mm_bos_free frees the scratch's own wt */
    mm_bos_free(fd, &sc);
    free((void*)A); free(ref); free(Brow); free(C); free(C1);
    rocket_close(fd);
    return (verify_fail || alias_fail) ? 2 : 0;
}
