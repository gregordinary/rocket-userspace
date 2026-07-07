// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * moe_spike.c — MUL_MAT_ID de-risk micro-bench (NOT a shipped gate).
 *
 * OUTCOME (2026-07-03): this spike's "GO" was OVER-OPTIMISTIC. It times F16 GEMMs
 * (A and B are already fp16 -- NO dequant), but real MoE GGUFs are QUANTIZED, so the
 * in-model handler pays a per-expert host dequant every micro-batch that this bench
 * excludes. The built ggml-rocket MUL_MAT_ID handler is bit-faithful but a net LOSS on
 * quant experts (gpt-oss 0.40-0.90x, DeepSeek 0.25-0.59x CPU even at -ub 2048) -> ships
 * OPT-IN (ROCKET_MOE=1, default off). See ggml-rocket + NPU_SETTLED.md. Kept as the
 * F16-GEMM-throughput reference (which does hold); it just isn't the whole in-model cost.
 *
 * MoE routed-expert prefill lowers to many SMALL matmuls: one [M_e,K]x[N,K]^T
 * per active expert, where M_e = n_tokens * n_expert_used / n_expert is well
 * below the dense rocket_min_m() offload floor (512). This bench answers the
 * single go/no-go question for the MUL_MAT_ID handler: at that small-M_e expert
 * shape, does the STREAMING (pack-per-call, weights can't stay resident across
 * 32 experts x 24 layers) NPU path still beat a multi-core CPU fp16 GEMM?
 *
 * The NPU side is rocket_matmul_fp16_mt (the realistic first-handler path,
 * includes per-call weight pack). The CPU side is an OpenMP cache-blocked
 * fp16->fp32 GEMM. NOTE: the CPU kernel is reasonable but NOT ggml-tuned
 * (no repack / tinyBLAS), so the NPU/CPU ratio here is an UPPER bound for the
 * NPU -- if the NPU does not win here, it will not win in-model.
 *
 * Build (on the RK1, from rocket-userspace/):
 *   gcc -O3 -march=native -fopenmp -Iinclude tests/moe_spike.c \
 *       src/rocket_npu.c src/npu_regcmd.c src/rocket_matmul.c \
 *       src/rocket_matmul_mt.c src/rocket_hw_profile.c src/rocket_log.c \
 *       -o build/moe_spike -lm -lpthread
 * Run:
 *   ./build/moe_spike M K N [reps] [nth]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "rocket_npu.h"
#include "rocket_matmul.h"

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6; }

/* OpenMP GEMM: C[M,N] = A[M,K] * B[N,K]^T, fp16 in / fp32 accumulate, using the
 * NEON vfmlal (fp16->fp32 widening FMA) inner loop that ggml's ARM fp16 GEMV/GEMM
 * path uses -- the fair, tuned CPU baseline (K%8 handled; K=2880 is %8). */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
static void cpu_gemm(int M, int K, int N,
                     const _Float16 *A, const _Float16 *B, float *C) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const float16_t *ar = (const float16_t *)A + (size_t)i*K;
        for (int j = 0; j < N; j++) {
            const float16_t *br = (const float16_t *)B + (size_t)j*K;
            float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
            int k=0;
            for (; k+8<=K; k+=8) {   // widen fp16->fp32 (asimdhp), fp32 FMA -- ggml's no-FHM path
                float16x8_t a = vld1q_f16(ar+k), b = vld1q_f16(br+k);
                s0 = vfmaq_f32(s0, vcvt_f32_f16(vget_low_f16(a)),  vcvt_f32_f16(vget_low_f16(b)));
                s1 = vfmaq_f32(s1, vcvt_f32_f16(vget_high_f16(a)), vcvt_f32_f16(vget_high_f16(b)));
            }
            float acc = vaddvq_f32(vaddq_f32(s0,s1));
            for (; k<K; k++) acc += (float)ar[k]*(float)br[k];
            C[(size_t)i*N+j] = acc;
        }
    }
}
#define CPU_KERNEL "neon-fp16-widen-fp32fma"
#else
static void cpu_gemm(int M, int K, int N,
                     const _Float16 *A, const _Float16 *B, float *C) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc=0; for (int k=0;k<K;k++) acc+=(float)A[(size_t)i*K+k]*(float)B[(size_t)j*K+k];
            C[(size_t)i*N+j]=acc;
        }
}
#define CPU_KERNEL "scalar-fallback"
#endif

int main(int argc, char **argv)
{
    int M   = argc>1 ? atoi(argv[1]) : 256;
    int K   = argc>2 ? atoi(argv[2]) : 2880;
    int N   = argc>3 ? atoi(argv[3]) : 2880;
    int reps= argc>4 ? atoi(argv[4]) : 6;
    int nth = argc>5 ? atoi(argv[5]) : 4;   /* NPU worker fds */
#ifdef _OPENMP
    int cpu_th = omp_get_max_threads();
#else
    int cpu_th = 1;
#endif
    printf("shape M=%d K=%d N=%d  reps=%d  npu_nth=%d  cpu_omp_th=%d\n",
           M, K, N, reps, nth, cpu_th);

    double gflop = 2.0*(double)M*K*N/1e9;

    _Float16 *A = malloc((size_t)M*K*sizeof(_Float16));
    _Float16 *B = malloc((size_t)N*K*sizeof(_Float16));
    _Float16 *C = malloc((size_t)M*N*sizeof(_Float16));
    float *Ccpu = malloc((size_t)M*N*sizeof(float));
    for (size_t i=0;i<(size_t)M*K;i++) A[i]=(_Float16)(((i*7)%13-6)*0.05f);
    for (size_t i=0;i<(size_t)N*K;i++) B[i]=(_Float16)(((i*5)%11-5)*0.05f);
    printf("cpu_kernel=%s\n", CPU_KERNEL);

    /* ---- CPU (NEON fp16 GEMM, ggml-style) ---- */
    double cpu_best = 1e30;
    for (int r=0;r<reps;r++){ double t0=now_ms(); cpu_gemm(M,K,N,A,B,Ccpu);
        double dt=now_ms()-t0; if(r>0 && dt<cpu_best) cpu_best=dt; }
    printf("CPU  best %8.3f ms  (%6.1f GFLOP/s)\n", cpu_best, gflop/(cpu_best/1e3));

    /* ---- NPU (streaming mt, pack-per-call) ---- */
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> CPU-only\n", fd); return 2; }
    rocket_close(fd);
    double npu_best = 1e30;
    for (int r=0;r<reps;r++){ double t0=now_ms();
        int rc = rocket_matmul_fp16_mt(M, K, N, A, B, C, nth);
        double dt=now_ms()-t0;
        if (rc!=0){ printf("NPU mt rc=%d\n", rc); return 1; }
        if(r>0 && dt<npu_best) npu_best=dt; }
    printf("NPU  best %8.3f ms  (%6.1f GFLOP/s)\n", npu_best, gflop/(npu_best/1e3));
    printf("==> NPU/CPU speedup: %.2fx  (>1 = NPU wins this shape)\n", cpu_best/npu_best);
    return 0;
}
