// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * bytes_moved_rocket.c — ANALYTICAL DRAM-traffic model for a tiled matmul.
 *
 * The real RK3588 DDR/DMA byte counters are dead (reading the 0x2xxx page
 * hard-locks the SoC; the 0x80xx page is config-only). RKNN's "Total Memory R/W per frame"
 * is likewise *computed from the graph*, not read from HW. So this tool computes
 * the bytes each phase moves analytically from the shape + the real tiling
 * (rocket_matmul_plan, pure/no-HW) + the dtype + the reuse mode — a traffic metric
 * to drive the readback/dispatch-floor work with NO hardware counter.
 *
 * It quantifies the "not MAC-bound" finding per shape: pair these byte figures with a
 * ROCKET_MM_PROFILE timing line (pack / wait / read ms) to get achieved GB/s per
 * phase and see which phase is bandwidth-bound. It also makes the int8 readback
 * floor explicit: int32 partials, no on-NPU K-accum, so readback scales with nKt.
 *
 * Phase  -> ROCKET_MM_PROFILE bucket it feeds, and the formula:
 *   packB    (host weight scatter, "packB")  = N*K*ein                 (streaming: per call)
 *   packA    (host input  scatter, "packA")  = M*K*ein
 *   wDMA     (NPU weight DRAM->CBUF, in wait) = nMt * N*K*ein           (no weight reuse)
 *   fDMA     (NPU feature DRAM->CBUF, in wait)= M*K*ein   (data_reuse)  | nNt*M*K*ein (no reuse)
 *   oWDMA    (NPU output  CBUF->DRAM, in wait)= M*N*eout  (KACC, once)  | nKt*M*N*eout (no KACC)
 *   ewRD     (NPU EW-accum operand re-read)   = (nKt-1)*M*N*eout        (KACC only)
 *   readback (host de-tile gather, "read")    = M*N*eout (KACC)         | nKt*M*N*eout (no KACC)
 *
 * Build (CTest bytes_moved_rocket; pure, runs anywhere — no NPU):
 *   gcc -O2 -Iinclude tests/bytes_moved_rocket.c -L build -lrocketnpu -o bytes_moved_rocket -lm
 * Run:
 *   ./bytes_moved_rocket 512 3840 4096                 # fp16 KACC+data_reuse (production)
 *   ./bytes_moved_rocket 512 3840 4096 int8            # int8 (no KACC -> nKt readback floor)
 *   ./bytes_moved_rocket 512 3840 4096 fp16 --no-kacc --no-reuse
 *   ./bytes_moved_rocket 512 3840 4096 fp16 --bw 12.0  # + LPDDR-bound time floor at 12 GB/s
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_matmul.h"

/* bytes per input / output element by dtype (output = the de-tiled readback elem) */
static void elem_bytes(const char *dt, double *ein, double *eout, int *kacc_ok)
{
    if      (!strcmp(dt,"fp16")) { *ein=2;   *eout=2; *kacc_ok=1; } /* fp16 K-accum on NPU */
    else if (!strcmp(dt,"int8")) { *ein=1;   *eout=4; *kacc_ok=0; } /* int32 out, EW-accum HW-dead */
    else if (!strcmp(dt,"int4")) { *ein=0.5; *eout=2; *kacc_ok=0; } /* int16 out, host-accum */
    else if (!strcmp(dt,"bf16")) { *ein=2;   *eout=4; *kacc_ok=0; } /* fp32 out, host-accum */
    else if (!strcmp(dt,"tf32")) { *ein=4;   *eout=4; *kacc_ok=0; }
    else                         { *ein=2;   *eout=2; *kacc_ok=1; }
}

static void human(double b, char *out)
{
    const char *u[] = {"B","KB","MB","GB"}; int i=0; double v=b;
    while (v >= 1024.0 && i < 3) { v/=1024.0; i++; }
    sprintf(out, "%7.2f %s", v, u[i]);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s M K N [fp16|int8|int4|bf16|tf32] [--no-kacc] [--no-reuse] [--bw GBs]\n", argv[0]);
        return 2;
    }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    const char *dt = (argc > 4 && argv[4][0] != '-') ? argv[4] : "fp16";

    double ein, eout; int kacc_ok;
    elem_bytes(dt, &ein, &eout, &kacc_ok);
    int kacc = kacc_ok, data_reuse = 1; double bw = 0;
    for (int i = 4; i < argc; i++) {
        if      (!strcmp(argv[i], "--no-kacc"))  kacc = 0;
        else if (!strcmp(argv[i], "--no-reuse")) data_reuse = 0;
        else if (!strcmp(argv[i], "--bw") && i+1 < argc) bw = atof(argv[++i]);
    }

    int Mt=0, Kt=0, Nt=0;
    /* plan returns njobs (>0) on success, -1 on unsupported shape */
    int njobs_plan = rocket_matmul_plan(M, K, N, &Mt, &Kt, &Nt);
    if (njobs_plan < 0 || Mt<=0 || Kt<=0 || Nt<=0) {
        fprintf(stderr, "rocket_matmul_plan rejected %dx%dx%d (need K%%32, N%%16, M%%4)\n", M, K, N);
        return 2;
    }
    int nMt = (M + Mt - 1) / Mt, nKt = (K + Kt - 1) / Kt, nNt = (N + Nt - 1) / Nt;
    long njobs = (long)nMt * nNt * nKt;
    /* self-check: our tile-count derivation must match the planner's job count */
    if (njobs != njobs_plan) {
        fprintf(stderr, "FAIL: njobs mismatch (model %ld vs plan %d)\n", njobs, njobs_plan);
        return 1;
    }

    /* per-phase DRAM bytes (see header) */
    double packB    = (double)N * K * ein;
    double packA    = (double)M * K * ein;
    double wDMA     = (double)nMt * N * K * ein;
    double fDMA     = data_reuse ? (double)M * K * ein : (double)nNt * M * K * ein;
    double oWDMA    = kacc ? (double)M * N * eout : (double)nKt * M * N * eout;
    double ewRD     = kacc ? (double)(nKt - 1) * M * N * eout : 0.0;
    double readback = kacc ? (double)M * N * eout : (double)nKt * M * N * eout;

    double host_bytes = packB + packA + readback;             /* pack + read buckets   */
    double npu_bytes  = wDMA + fDMA + oWDMA + ewRD;            /* the "wait" DMA traffic */
    double total      = host_bytes + npu_bytes;
    double flop       = 2.0 * M * N * K;

    char h[8][32];
    human(packB,&h[0][0]); human(packA,&h[1][0]); human(wDMA,&h[2][0]); human(fDMA,&h[3][0]);
    human(oWDMA,&h[4][0]); human(ewRD,&h[5][0]); human(readback,&h[6][0]); human(total,&h[7][0]);

    printf("matmul C[%d,%d] = A[%d,%d] . B[%d,%d]^T   dtype=%s  KACC=%d  data_reuse=%d\n",
           M, N, M, K, N, K, dt, kacc, data_reuse);
    printf("tiling: Mt=%d Kt=%d Nt=%d -> nMt=%d nKt=%d nNt=%d  (%ld jobs)\n",
           Mt, Kt, Nt, nMt, nKt, nNt, njobs);
    printf("  in-elem=%.1fB out-elem=%.0fB\n", ein, eout);
    printf("\nDRAM traffic by phase (-> ROCKET_MM_PROFILE bucket):\n");
    printf("  packB    weight scatter (host, pack) : %s\n", &h[0][0]);
    printf("  packA    input  scatter (host, pack) : %s\n", &h[1][0]);
    printf("  wDMA     weight DRAM->CBUF (wait)     : %s   x nMt=%d\n", &h[2][0], nMt);
    printf("  fDMA     feature DRAM->CBUF (wait)    : %s   %s\n", &h[3][0],
           data_reuse ? "(data_reuse: read once)" : "(no reuse: x nNt)");
    printf("  oWDMA    output CBUF->DRAM (wait)     : %s   %s\n", &h[4][0],
           kacc ? "(KACC: write once)" : "(no KACC: x nKt partials)");
    if (kacc) printf("  ewRD     EW-accum operand re-read     : %s   (KACC: x (nKt-1))\n", &h[5][0]);
    printf("  readback de-tile gather (host, read) : %s   %s\n", &h[6][0],
           kacc ? "(KACC: read once)" : "(no KACC: x nKt -> the int readback FLOOR)");
    printf("\n  host  (pack+read) : %9.2f MB\n", host_bytes/1048576.0);
    printf("  npu   (wait DMA)  : %9.2f MB\n", npu_bytes/1048576.0);
    printf("  TOTAL             : %s   (%.1f MB)\n", &h[7][0], total/1048576.0);
    printf("  arithmetic intensity: %.2f FLOP/byte   (%.1f GFLOP)\n",
           flop/total, flop/1e9);

    if (bw > 0) {
        printf("\nbandwidth-bound floor @ %.1f GB/s (one phase at a time):\n", bw);
        double s = 1e9 * bw;
        printf("  packB %.2f ms | wDMA %.2f ms | fDMA %.2f ms | readback %.2f ms | TOTAL %.2f ms\n",
               packB/s*1e3, wDMA/s*1e3, fDMA/s*1e3, readback/s*1e3, total/s*1e3);
        printf("  (compare each to its ROCKET_MM_PROFILE bucket ms => achieved GB/s = bytes/measured_s)\n");
    }
    printf("\nmodel self-check (njobs %ld == plan): PASS\n", njobs);
    return 0;
}
