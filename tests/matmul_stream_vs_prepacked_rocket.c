// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_stream_vs_prepacked_rocket.c — isolate the per-call weight-scatter (packB).
 *
 * THE LOAD-BEARING QUESTION: does eliminating the per-call weight scatter
 * (packB) actually raise throughput, or is it already hidden behind the NPU `wait`
 * (overlapped across the T concurrent workers, as the pre-KACC envelope assumed)?
 *
 * The full-MODEL resident-weight path can't answer this: the NPU regcmd encodes BO
 * addresses as 32-bit fields (npu_regcmd.c: weights_dma & 0xFFFFFFFF), so all
 * resident weight BOs must live in the low-4GB IOVA window — far below a 22GB model
 * (ROCKET_CACHE_MB=12000 overflows it: "a BO dma_address exceeds 32 bits"). But ONE
 * representative shape needs just a single resident weight (tens of MB), so we can
 * compare directly, with the REAL T-worker overlap, the only two paths that differ
 * solely in whether B is re-scattered each call:
 *
 *   STREAM    (rocket_matmul_fp16_stream)    — persistent fds + per-shape scratch,
 *                                              but RE-PACKS B every call   (packB paid)
 *   PREPACKED (rocket_matmul_fp16_prepacked) — B scattered ONCE, resident  (packB ~0)
 *
 * Both share the A-pack across workers and run the SAME compute, so (stream_ms -
 * prepacked_ms) is the per-call packB cost left ON the critical path after worker
 * overlap — i.e. the ceiling on what resident pre-tiled weights could buy for
 * this shape. ~0 => packB is hidden (resident weights are a non-lever); large => packB is exposed.
 *
 * Honors ROCKET_KACC / ROCKET_REUSE / ROCKET_MM_PROFILE from the env — RUN IT IN THE
 * OPERATING MODE so the overlap matches the model:
 *   ROCKET_KACC=1 sudo -E ./matmul_stream_vs_prepacked_rocket 1024 3840  4096 5 20
 *   ROCKET_KACC=1 sudo -E ./matmul_stream_vs_prepacked_rocket 1024 15360 3840 5 20  # ffn-down
 *   ROCKET_KACC=1 sudo -E ./matmul_stream_vs_prepacked_rocket  512 3840  4096 5 20  # smaller M
 * (packB/wait ~ FLOPS/(M*BW) is M-bound, so smaller M exposes more packB.)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "rocket_matmul.h"

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6; }

/* max abs error of the first ref_rows rows vs the CPU fp32 reference */
static float verify(const _Float16 *C, const float *ref, int M, int N, int ref_rows){
    (void)M;
    float max_abs=0;
    for (int m=0;m<ref_rows;m++) for (int n=0;n<N;n++){
        float ad=fabsf((float)C[(size_t)m*N+n]-ref[(size_t)m*N+n]);
        if (ad>max_abs) max_abs=ad;
    }
    return max_abs;
}

int main(int argc, char **argv)
{
    int M     = argc>=4 ? atoi(argv[1]) : 1024;
    int K     = argc>=4 ? atoi(argv[2]) : 3840;
    int N     = argc>=4 ? atoi(argv[3]) : 4096;
    int T     = argc>=5 ? atoi(argv[4]) : 5;
    int iters = argc>=6 ? atoi(argv[5]) : 20;
    printf("stream-vs-prepacked  C[%d,%d]=A[%d,%d]xB[%d,%d]^T  T=%d iters=%d\n",
           M, N, M, K, N, K, T, iters);

    _Float16 *A = malloc((size_t)M*K*sizeof(_Float16));
    _Float16 *B = malloc((size_t)N*K*sizeof(_Float16));
    _Float16 *C = malloc((size_t)M*N*sizeof(_Float16));
    float    *ref = malloc((size_t)M*N*sizeof(float));
    if (!A||!B||!C||!ref) { fprintf(stderr,"oom\n"); return 1; }
    for (size_t i=0;i<(size_t)M*K;i++) A[i]=(_Float16)(((i*7)%13-6)*0.05f);
    for (size_t i=0;i<(size_t)N*K;i++) B[i]=(_Float16)(((i*5)%11-5)*0.05f);

    int ref_rows = M<=64 ? M : 64;
    for (int m=0;m<ref_rows;m++) for (int n=0;n<N;n++){
        float a=0; for (int k=0;k<K;k++) a+=(float)A[(size_t)m*K+k]*(float)B[(size_t)n*K+k];
        ref[(size_t)m*N+n]=a;
    }
    double gflop = 2.0*(double)M*N*K/1e9;

    /* ---- STREAM: re-pack B every call (current ship path for K>2048) ---- */
    rocket_stream *s = rocket_stream_create(T);
    if (!s) { fprintf(stderr,"stream_create failed\n"); return 1; }
    for (int i=0;i<3;i++)   /* warm: ramp the clock 200->600 before timing */
        if (rocket_matmul_fp16_stream(s,M,K,N,A,B,C)) { fprintf(stderr,"stream warmup failed\n"); return 1; }
    float sa = verify(C,ref,M,N,ref_rows);
    double ts0 = now_ms();
    for (int i=0;i<iters;i++)
        if (rocket_matmul_fp16_stream(s,M,K,N,A,B,C)) { fprintf(stderr,"stream iter %d failed\n",i); return 1; }
    double sdt = (now_ms()-ts0)/iters;
    rocket_stream_free(s);

    /* ---- PREPACKED: scatter B once into resident BOs, reuse every call ---- */
    rocket_ctx *ctx = rocket_ctx_create(T);
    if (!ctx) { fprintf(stderr,"ctx_create failed\n"); return 1; }
    rocket_weights *w = rocket_weights_pack(ctx,M,K,N,B);
    if (!w) { fprintf(stderr,"weights_pack failed (IOVA? this is a SINGLE weight)\n"); rocket_ctx_free(ctx); return 1; }
    for (int i=0;i<3;i++)   /* warm (clock is already up from the stream loop, but be safe) */
        if (rocket_matmul_fp16_prepacked(ctx,M,K,N,A,C,w)) { fprintf(stderr,"prepacked warmup failed\n"); return 1; }
    float pa = verify(C,ref,M,N,ref_rows);
    double tp0 = now_ms();
    for (int i=0;i<iters;i++)
        if (rocket_matmul_fp16_prepacked(ctx,M,K,N,A,C,w)) { fprintf(stderr,"prepacked iter %d failed\n",i); return 1; }
    double pdt = (now_ms()-tp0)/iters;
    rocket_weights_free(ctx,w);
    rocket_ctx_free(ctx);

    printf("STREAM    : %8.2f ms/call  %7.1f GFLOP/s  verify max_abs=%.3f\n",
           sdt, gflop/(sdt/1000.0), sa);
    printf("PREPACKED : %8.2f ms/call  %7.1f GFLOP/s  verify max_abs=%.3f\n",
           pdt, gflop/(pdt/1000.0), pa);
    /* (stream - prepacked) = the per-call packB left on the critical path after the
     * T-worker overlap. >0 and large => resident weights would help THIS shape
     * (up to the ~4GB IOVA cap); ~0 => packB is hidden under wait, resident is a non-lever. */
    printf("packB on critical path: %+.2f ms/call (%.1f%% of stream) -> prepacked is %.3fx %s\n",
           sdt - pdt, (sdt>0 ? (sdt-pdt)/sdt*100.0 : 0.0), (pdt>0 ? sdt/pdt : 0.0),
           pdt < sdt ? "faster" : "(no gain)");

    /* This is a benchmark, but a path that computes GARBAGE fast is worse than
     * useless: gate on the verify so a broken stream/prepacked path FAILS instead of
     * reporting a fast-but-wrong number. (Exhaustive per-row correctness is
     * matmul_correctness_matrix_rocket's job; the first ref_rows here are a sanity
     * floor sized to catch gross layout/scatter breakage.) Override with VERIFY_TOL. */
    double tol = getenv("VERIFY_TOL") ? atof(getenv("VERIFY_TOL")) : 1.0;
    int bad = (sa > tol) || (pa > tol);
    printf("verify (tol=%.3f): stream max_abs=%.3f prepacked max_abs=%.3f -> %s\n",
           tol, sa, pa, bad ? "FAIL" : "PASS");

    free(A); free(B); free(C); free(ref);
    return bad ? 1 : 0;
}
