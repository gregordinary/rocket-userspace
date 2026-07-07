// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int8_prepacked_rocket.c — the RESIDENT int8 weight path
 * (rocket_i8_ctx / rocket_i8_weights_pack / rocket_matmul_int8_prepacked) must be
 * BIT-EXACT to the one-shot rocket_matmul_int8 oracle, and must not alias across
 * distinct resident weights that share one per-shape scratch.
 *
 * This is the int8 analogue of matmul_prepacked_rocket / prototype_shared_scratch
 * (which de-risked the fp16 shared-scratch refactor). It proves the resident
 * int8 layout (per-worker N-split weight BO + shared in/out/regcmd scratch + host
 * int64 K-accum) composes correctly before the backend is wired to it.
 *
 * Three checks:
 *   1. resident vs one-shot oracle, bit-exact (==), single resident weight.
 *   2. W distinct resident weights of the SAME shape sharing one scratch: each
 *      must match ITS OWN oracle (catches cross-weight scratch aliasing).
 *   3. re-run check 1's weight a 2nd time (resident reuse) -> identical (no churn).
 *
 * Usage: matmul_int8_prepacked_rocket [M K N [W]]   (default 512 3840 4096, W=4)
 *   Needs K%32==0, N%32==0, (M%4==0 || M==1). Try the Gemma FFN shapes:
 *     512 15360 3840   (ffn-down: deep K -> host int32 accum, nKt large)
 *     512 3840 15360   (ffn-up:  many N-tiles, fanned across workers)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}
static int8_t rand_i8(void) { return (int8_t)(rand() % 256 - 128); }

/* one-shot oracle: persistent fd, raw int32 out. */
static int oracle(int M, int K, int N, const int8_t *A, const int8_t *B, int32_t *C) {
    int fd = rocket_open();
    if (fd < 0) return fd;
    int ret = rocket_matmul_int8(fd, M, K, N, A, B, C);
    rocket_close(fd);
    return ret;
}

static int cmp_exact(const char *tag, const int32_t *got, const int32_t *ref, size_t n) {
    int bad = 0; long maxabs = 0;
    for (size_t i = 0; i < n; i++) {
        long d = labs((long)got[i] - (long)ref[i]);
        if (d > maxabs) maxabs = d;
        if (got[i] != ref[i] && bad < 6) {
            printf("  [%s] mismatch [%zu] exp=%d got=%d\n", tag, i, ref[i], got[i]);
            bad++;
        }
    }
    printf("  [%s] max_abs_diff=%ld -> %s\n", tag, maxabs, bad ? "FAIL" : "PASS (bit-exact)");
    return bad ? -1 : 0;
}

int main(int argc, char **argv) {
    int M = 512, K = 3840, N = 4096, W = 4;
    if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    if (argc == 5) W = atoi(argv[4]);
    if (argc != 1 && argc != 4 && argc != 5) {
        printf("usage: %s [M K N [W]]\n", argv[0]); return -1;
    }
    if (W < 1) W = 1;
    if (W > 8) W = 8;

    int Mt, Kt, Nt;
    if (rocket_matmul_plan_int8(M, K, N, &Mt, &Kt, &Nt) < 0) {
        fprintf(stderr, "unsupported shape (need K%%32==0, N%%32==0, M%%4==0||1)\n");
        return -1;
    }
    printf("resident int8 C[%d,%d] = A[%d,%d] x B[%d,%d]^T  (W=%d distinct weights)\n",
           M, N, M, K, N, K, W);
    printf("tiling: Mt=%d Kt=%d Nt=%d  nKt=%d\n", Mt, Kt, Nt, (K + Kt - 1) / Kt);

    const char *te = getenv("ROCKET_N_THREADS");
    int nthreads = te ? atoi(te) : 5;

    size_t Asz = (size_t)M * K, Bsz = (size_t)N * K, Csz = (size_t)M * N;
    int8_t  *A  = malloc(Asz);
    int8_t **B  = malloc(W * sizeof(*B));
    int32_t *Cr = malloc(Csz * sizeof(int32_t));   /* resident result */
    int32_t *Co = malloc(Csz * sizeof(int32_t));   /* one-shot oracle */
    if (!A || !B || !Cr || !Co) { fprintf(stderr, "host alloc failed\n"); return -1; }

    srand(20260617);
    for (size_t i = 0; i < Asz; i++) A[i] = rand_i8();
    for (int w = 0; w < W; w++) {
        B[w] = malloc(Bsz);
        if (!B[w]) { fprintf(stderr, "host alloc failed\n"); return -1; }
        for (size_t i = 0; i < Bsz; i++) B[w][i] = rand_i8();
    }

    rocket_i8_ctx *ctx = rocket_i8_ctx_create(nthreads);
    if (!ctx) { fprintf(stderr, "rocket_i8_ctx_create failed — no NPU, skipping\n"); return 2; }

    /* pack all W distinct weights resident (they share one per-shape scratch). */
    rocket_i8_weights **rw = malloc(W * sizeof(*rw));
    for (int w = 0; w < W; w++) {
        rw[w] = rocket_i8_weights_pack(ctx, M, K, N, B[w]);
        if (!rw[w]) { fprintf(stderr, "rocket_i8_weights_pack(%d) failed\n", w); return -1; }
    }

    int fails = 0;

    /* Check 1+2: each resident weight vs its own one-shot oracle. */
    for (int w = 0; w < W; w++) {
        if (oracle(M, K, N, A, B[w], Co)) { fprintf(stderr, "oracle(%d) failed\n", w); return -1; }
        memset(Cr, 0, Csz * sizeof(int32_t));
        int64_t t0 = now_us();
        if (rocket_matmul_int8_prepacked(ctx, M, K, N, A, Cr, rw[w])) {
            fprintf(stderr, "rocket_matmul_int8_prepacked(%d) failed\n", w); return -1;
        }
        int64_t us = now_us() - t0;
        double gop = 2.0 * M * K * N / 1e9;
        char tag[32]; snprintf(tag, sizeof(tag), "w%d", w);
        printf("weight %d: resident %.2f ms (%.1f GOP/s)\n", w, us / 1000.0, gop / (us / 1e6));
        if (cmp_exact(tag, Cr, Co, Csz)) fails++;
    }

    /* Check 3: re-run weight 0 (resident reuse must be stable). */
    if (oracle(M, K, N, A, B[0], Co)) return -1;
    memset(Cr, 0, Csz * sizeof(int32_t));
    if (rocket_matmul_int8_prepacked(ctx, M, K, N, A, Cr, rw[0])) return -1;
    if (cmp_exact("reuse-w0", Cr, Co, Csz)) fails++;

    for (int w = 0; w < W; w++) rocket_i8_weights_free(ctx, rw[w]);
    rocket_i8_ctx_free(ctx);

    printf("\n==> %s (%d/%d checks failed)\n", fails ? "FAIL" : "ALL PASS", fails, W + 1);

    free(A); free(Cr); free(Co); free(rw);
    for (int w = 0; w < W; w++) free(B[w]);
    free(B);
    return fails ? -1 : 0;
}
