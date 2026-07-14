// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * moe_gw_bench.c — resident GROUP-WISE int8 throughput at the MoE expert shape.
 *
 * Times rocket_matmul_int8_prepacked_gw against the resident PER-CHANNEL
 * rocket_matmul_int8_prepacked at the same shape. The two run identical NPU work — same
 * tiles, same DMA, same readback — and differ only in the host readback loop, which does
 * `float += a_scale*b_scale*int32` instead of `int64 += int32`. So the gap between the two
 * columns is the cost of carrying per-K-group quant scales, isolated: it prices the
 * group-wise mode against its own dtype rather than against a different one.
 *
 * It comes out at ~0.6%. The scales ride a readback the integer datapath is already forced
 * to pay (on-device integer K-accum is HW-dead, so every K-tile partial crosses to the host
 * regardless), and the multiply-add fuses into an accumulate loop that is already there and
 * already memory-bound. Keep it that way: applying the scales in a SEPARATE pass over the
 * int32 output would turn a rounding error into a real cost.
 *
 * The default shape is one gpt-oss-20b expert projection at a 2048-token micro-batch
 * (M=256 = 2048 * 4 active / 32 experts, K=N=2880), at group=576 — the largest quant group
 * the CBUF admits, hence the fewest K-tiles (nKt=5) and the least readback, which is what
 * these paths are bound by.
 *
 * MEASUREMENT DISCIPLINE (or the numbers are noise): the NPU clock parks at 200 MHz on an
 * idle gap, so a bursty harness flaps 200<->600 MHz and swings run-to-run by ~5x. Pin the
 * power domain before running and restore it after:
 *   for d in fdab0000 fdac0000 fdad0000; do echo on | sudo tee /sys/devices/platform/$d.npu/power/control; done
 *   sudo -E ./moe_gw_bench
 *   for d in fdab0000 fdac0000 fdad0000; do echo auto | sudo tee /sys/devices/platform/$d.npu/power/control; done
 * Run 1 is always discarded (cold clock); the reported figure is the median of the rest.
 *
 * Usage: moe_gw_bench [M K N group [iters]]     (default 256 2880 2880 576, 7 iters)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

#define MAX_ITERS 64

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Median of the WARM runs (run 1 discarded: the clock parks at idle, so a cold run reads
 * ~15% low and a mean would carry it). */
static double median_warm(double *t, int n)
{
    if (n <= 1) return t[0];
    qsort(t + 1, (size_t)(n - 1), sizeof(double), cmp_d);
    return t[1 + (n - 1) / 2];
}

int main(int argc, char **argv)
{
    int M = 256, K = 2880, N = 2880, group = 576, iters = 7;
    if (argc >= 5) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); group = atoi(argv[4]); }
    if (argc == 6) iters = atoi(argv[5]);
    if (iters < 2) iters = 2;
    if (iters > MAX_ITERS) iters = MAX_ITERS;
    if (M % 4 || K % 32 || N % 32 || group < 32 || group % 32 || K % group) {
        fprintf(stderr, "bad shape: need M%%4, K%%32, N%%32, group>=32, group%%32, K%%group\n");
        return -1;
    }
    const int nG = K / group;
    const char *te = getenv("ROCKET_N_THREADS");
    int nthreads = te ? atoi(te) : 3;   /* saturates at the 3 physical NPU cores */

    int Mt, Kt, Nt;
    int tiles = rocket_matmul_plan_int8_gw(M, K, N, group, &Mt, &Kt, &Nt);
    if (tiles < 0) { fprintf(stderr, "planner rejects the shape (%d)\n", tiles); return -1; }
    const double gop = 2.0 * (double)M * K * N / 1e9;

    printf("MoE expert GEMM  M=%d K=%d N=%d  group=%d (nG=%d)  workers=%d  iters=%d\n",
           M, K, N, group, nG, nthreads, iters);
    printf("  one-shot plan: Mt=%d Kt=%d Nt=%d  nKt=%d  tiles=%d  |  %.2f GOP/matmul\n",
           Mt, Kt, Nt, K / Kt, tiles, gop);
    printf("  readback: %.1f MB/matmul (M*N*nKt at the 8 B/elem int32 output stride)\n\n",
           (double)M * N * (K / Kt) * 8.0 / (1024 * 1024));

    size_t Asz = (size_t)M * K, Bsz = (size_t)N * K, Csz = (size_t)M * N;
    int8_t  *A  = malloc(Asz), *B = malloc(Bsz);
    float   *as = malloc((size_t)M * nG * sizeof(float));
    float   *bs = malloc((size_t)N * nG * sizeof(float));
    float   *Cf = malloc(Csz * sizeof(float));
    int32_t *C32 = malloc(Csz * sizeof(int32_t));
    double  *t_gw = malloc(MAX_ITERS * sizeof(double)), *t_pc = malloc(MAX_ITERS * sizeof(double));
    if (!A || !B || !as || !bs || !Cf || !C32 || !t_gw || !t_pc) { fprintf(stderr, "alloc\n"); return -1; }

    srand(20260714);
    for (size_t i = 0; i < Asz; i++) A[i] = (int8_t)(rand() % 255 - 127);
    for (size_t i = 0; i < Bsz; i++) B[i] = (int8_t)(rand() % 255 - 127);
    for (size_t i = 0; i < (size_t)M * nG; i++) as[i] = 0.01f + (rand() % 100) / 10000.0f;
    for (size_t i = 0; i < (size_t)N * nG; i++) bs[i] = 0.01f + (rand() % 100) / 10000.0f;

    rocket_i8_ctx *ctx = rocket_i8_ctx_create(nthreads);
    if (!ctx) { fprintf(stderr, "ctx create failed — no NPU, skipping\n"); return 2; }

    /* Pack both residencies up front — the pack is the ONE-TIME cost this whole path
     * exists to hoist out of the forward pass, so it is deliberately outside the timer.
     * (Its size is reported: that is the RAM an expert occupies while resident.) */
    double t0 = now_ms();
    rocket_i8_weights *wgw = rocket_i8_weights_pack_gw(ctx, M, K, N, B, group);
    double t_pack_gw = now_ms() - t0;
    t0 = now_ms();
    rocket_i8_weights *wpc = rocket_i8_weights_pack(ctx, M, K, N, B);
    double t_pack_pc = now_ms() - t0;
    if (!wgw || !wpc) { fprintf(stderr, "pack failed\n"); return -1; }
    printf("  resident weight: %zu MB   (pack once: group-wise %.1f ms, per-channel %.1f ms)\n\n",
           rocket_i8_weights_bytes(wgw) >> 20, t_pack_gw, t_pack_pc);

    for (int i = 0; i < iters; i++) {
        t0 = now_ms();
        if (rocket_matmul_int8_prepacked_gw(ctx, M, K, N, A, as, bs, Cf, wgw)) {
            fprintf(stderr, "prepacked_gw failed\n"); return -1; }
        t_gw[i] = now_ms() - t0;

        t0 = now_ms();
        if (rocket_matmul_int8_prepacked(ctx, M, K, N, A, C32, wpc)) {
            fprintf(stderr, "prepacked failed\n"); return -1; }
        t_pc[i] = now_ms() - t0;
        printf("  run %d%-8s group-wise %7.2f ms   per-channel %7.2f ms\n",
               i + 1, i == 0 ? " (cold)" : "", t_gw[i], t_pc[i]);
    }

    double mgw = median_warm(t_gw, iters), mpc = median_warm(t_pc, iters);
    printf("\n  median (warm)   group-wise %7.2f ms (%6.1f GOP/s)\n", mgw, gop / (mgw / 1e3));
    printf("                  per-channel %7.2f ms (%6.1f GOP/s)\n", mpc, gop / (mpc / 1e3));
    printf("  scale-accumulate cost (the ONLY delta between them): %+.2f ms (%+.1f%%)\n",
           mgw - mpc, 100.0 * (mgw - mpc) / mpc);

    rocket_i8_weights_free(ctx, wgw);
    rocket_i8_weights_free(ctx, wpc);
    rocket_i8_ctx_free(ctx);
    free(A); free(B); free(as); free(bs); free(Cf); free(C32); free(t_gw); free(t_pc);
    return 0;
}
