// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int4_prepacked_rocket.c — the RESIDENT int4 path (rocket_i4_ctx /
 * rocket_matmul_int4_prepacked) must be BIT-EXACT to the one-shot rocket_matmul_int4
 * oracle, and must not alias across distinct resident weights sharing one scratch.
 * The int4 sibling of matmul_int8_prepacked_rocket — validates rocket_i4_ctx before
 * the perf comparison trusts it. Reduced int4 range [-3,3] so no Kt-partial
 * saturates int16 (see matmul_int4_tiled_rocket).
 *
 * Usage: matmul_int4_prepacked_rocket [M K N [W]]   (default 512 3840 4096, W=3)
 *   K%32, N%64, (M%4||1).
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int8_t rand_i4(void) { return (int8_t)(rand() % 7 - 3); }

static int oracle(int M, int K, int N, const int8_t *A, const int8_t *B, int32_t *C) {
    int fd = rocket_open();
    if (fd < 0) return fd;
    int ret = rocket_matmul_int4(fd, M, K, N, A, B, C);
    rocket_close(fd);
    return ret;
}

static int cmp_exact(const char *tag, const int32_t *got, const int32_t *ref, size_t n) {
    int bad = 0; long maxabs = 0;
    for (size_t i = 0; i < n; i++) {
        long d = labs((long)got[i] - (long)ref[i]);
        if (d > maxabs) maxabs = d;
        if (got[i] != ref[i] && bad < 6) { printf("  [%s] [%zu] exp=%d got=%d\n", tag, i, ref[i], got[i]); bad++; }
    }
    printf("  [%s] max_abs_diff=%ld -> %s\n", tag, maxabs, bad ? "FAIL" : "PASS (bit-exact)");
    return bad ? -1 : 0;
}

int main(int argc, char **argv) {
    int M = 512, K = 3840, N = 4096, W = 3;
    if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    if (argc == 5) W = atoi(argv[4]);
    if (W < 1) W = 1;
    if (W > 8) W = 8;

    int Mt, Kt, Nt;
    if (rocket_matmul_plan_int4(M, K, N, &Mt, &Kt, &Nt) < 0) {
        fprintf(stderr, "unsupported (need K%%32, N%%64, M%%4||1)\n"); return -1; }
    printf("resident int4 C[%d,%d]=A[%d,%d]xB[%d,%d]^T (W=%d) tiling Mt=%d Kt=%d Nt=%d nKt=%d\n",
           M, N, M, K, N, K, W, Mt, Kt, Nt, (K+Kt-1)/Kt);

    const char *te = getenv("ROCKET_N_THREADS");
    int nthreads = te ? atoi(te) : 5;
    size_t Asz = (size_t)M*K, Bsz = (size_t)N*K, Csz = (size_t)M*N;
    int8_t *A = malloc(Asz);
    int8_t **B = malloc(W * sizeof(*B));
    int32_t *Cr = malloc(Csz*sizeof(int32_t)), *Co = malloc(Csz*sizeof(int32_t));
    if (!A || !B || !Cr || !Co) { fprintf(stderr, "alloc\n"); return -1; }
    srand(20260617);
    for (size_t i = 0; i < Asz; i++) A[i] = rand_i4();
    for (int w = 0; w < W; w++) { B[w] = malloc(Bsz); for (size_t i = 0; i < Bsz; i++) B[w][i] = rand_i4(); }

    rocket_i4_ctx *ctx = rocket_i4_ctx_create(nthreads);
    if (!ctx) { fprintf(stderr, "ctx create failed — no NPU, skipping\n"); return 2; }
    rocket_i4_weights **rw = malloc(W * sizeof(*rw));
    for (int w = 0; w < W; w++) {
        rw[w] = rocket_i4_weights_pack(ctx, M, K, N, B[w]);
        if (!rw[w]) { fprintf(stderr, "pack %d failed\n", w); return -1; }
    }

    int fails = 0;
    for (int w = 0; w < W; w++) {
        if (oracle(M, K, N, A, B[w], Co)) { fprintf(stderr, "oracle %d failed\n", w); return -1; }
        memset(Cr, 0, Csz*sizeof(int32_t));
        if (rocket_matmul_int4_prepacked(ctx, M, K, N, A, Cr, rw[w])) { fprintf(stderr, "prepacked %d failed\n", w); return -1; }
        char tag[16]; snprintf(tag, sizeof(tag), "w%d", w);
        if (cmp_exact(tag, Cr, Co, Csz)) fails++;
    }
    /* reuse w0 */
    if (!oracle(M, K, N, A, B[0], Co)) {
        memset(Cr, 0, Csz*sizeof(int32_t));
        if (!rocket_matmul_int4_prepacked(ctx, M, K, N, A, Cr, rw[0]))
            if (cmp_exact("reuse-w0", Cr, Co, Csz)) fails++;
    }

    for (int w = 0; w < W; w++) rocket_i4_weights_free(ctx, rw[w]);
    rocket_i4_ctx_free(ctx);
    printf("\n==> %s (%d/%d failed)\n", fails ? "FAIL" : "ALL PASS", fails, W+1);
    free(A); free(Cr); free(Co); free(rw);
    for (int w = 0; w < W; w++) free(B[w]);
    free(B);
    return fails ? -1 : 0;
}
