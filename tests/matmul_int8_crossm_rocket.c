// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int8_crossm_rocket.c — cross-M reuse gate for the RESIDENT int8 path: a weight
 * packed at ONE M is REUSED at other M (incl. a small short-prompt M) with no re-pack,
 * and stays BIT-EXACT (int32) vs the one-shot rocket_matmul_int8 at each M.
 *
 * The resident int8 weight scatter depends only on the N-split + Nt/Kt tiling, which the
 * canonical-tileM plan (rki_worker_alloc plans at MAX_TILE, not the actual M) makes
 * M-INDEPENDENT. int8's host int32 K-accum is exact for any K-tiling, so the result is
 * bit-identical to the one-shot regardless of M. So rocket_matmul_int8_prepacked at a
 * different M than pack must (a) NOT return -2 (the re-pack signal) and (b) equal the
 * one-shot exactly. Guards the warmup-M-serves-prefill-M fix (W8A8 prefill / detection):
 * pack at M=PACKM, compute at 512/256/768/64/8 against the SAME resident weight.
 *
 * CTest target matmul_int8_crossm_rocket; skip-code 2 off-device.
 *   sudo ./matmul_int8_crossm_rocket            # K=3840 N=256, pack M=512
 *   sudo ./matmul_int8_crossm_rocket 7680 512 512
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int8_t rand_i8(void) { return (int8_t)(rand() % 255 - 127); }

static int oneshot(int M, int K, int N, const int8_t *A, const int8_t *B, int32_t *C) {
    int fd = rocket_open();
    if (fd < 0) return fd;
    int rc = rocket_matmul_int8(fd, M, K, N, A, B, C);
    rocket_close(fd);
    return rc;
}

static int cmp_exact(const char *tag, const int32_t *got, const int32_t *ref, size_t n) {
    long nbad = 0; int32_t maxd = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t d = got[i] - ref[i]; if (d < 0) d = -d;
        if (d > maxd) maxd = d;
        if (d != 0 && nbad < 6) printf("  [%s] [%zu] oneshot=%d resident=%d\n", tag, i, ref[i], got[i]);
        if (d != 0) nbad++;
    }
    printf("  [%s] vs one-shot: max_abs_diff=%d nbad=%ld -> %s\n",
           tag, maxd, nbad, nbad ? "FAIL" : "PASS (bit-exact)");
    return nbad ? -1 : 0;
}

int main(int argc, char **argv) {
    int K = 3840, N = 256, PACKM = 512;
    if (argc >= 3) { K = atoi(argv[1]); N = atoi(argv[2]); }
    if (argc >= 4) PACKM = atoi(argv[3]);
    if (K % 32 || N % 32 || PACKM % 4) {
        fprintf(stderr, "bad shape: need K%%32, N%%32, PACKM%%4\n");
        return -1;
    }
    int Ms[] = { 512, 256, 768, 64, 8 };
    const int nMs = (int)(sizeof(Ms) / sizeof(Ms[0]));
    int Mmax = PACKM;
    for (int i = 0; i < nMs; i++) if (Ms[i] > Mmax) Mmax = Ms[i];

    printf("cross-M resident int8: pack M=%d, K=%d N=%d, reuse at", PACKM, K, N);
    for (int i = 0; i < nMs; i++) printf(" %d", Ms[i]);
    printf("\n");

    const char *te = getenv("ROCKET_N_THREADS");
    int nthreads = te ? atoi(te) : 5;

    int8_t  *A  = malloc((size_t)Mmax * K);
    int8_t  *B  = malloc((size_t)N * K);
    int32_t *Co = malloc((size_t)Mmax * N * sizeof(int32_t));
    int32_t *Cr = malloc((size_t)Mmax * N * sizeof(int32_t));
    if (!A || !B || !Co || !Cr) { fprintf(stderr, "alloc\n"); return -1; }
    srand(20260627);
    for (size_t i = 0; i < (size_t)Mmax * K; i++) A[i] = rand_i8();
    for (size_t i = 0; i < (size_t)N * K;    i++) B[i] = rand_i8();

    rocket_i8_ctx *ctx = rocket_i8_ctx_create(nthreads);
    if (!ctx) { fprintf(stderr, "ctx create failed — no NPU, skipping\n"); return 2; }

    rocket_i8_weights *rw = rocket_i8_weights_pack(ctx, PACKM, K, N, B);
    if (!rw) { fprintf(stderr, "pack failed\n"); return -1; }

    int fails = 0;
    for (int i = 0; i < nMs; i++) {
        int M = Ms[i];
        char tag[24]; snprintf(tag, sizeof(tag), "M=%d", M);
        if (oneshot(M, K, N, A, B, Co)) { fprintf(stderr, "oneshot %s failed\n", tag); return -1; }
        memset(Cr, 0, (size_t)M * N * sizeof(int32_t));
        int rc = rocket_matmul_int8_prepacked(ctx, M, K, N, A, Cr, rw);
        if (rc == -2) { printf("  [%s] RE-PACK SIGNALLED (-2) — weight not reused -> FAIL\n", tag); fails++; continue; }
        if (rc)       { fprintf(stderr, "  [%s] prepacked failed (%d)\n", tag, rc); return -1; }
        if (cmp_exact(tag, Cr, Co, (size_t)M * N)) fails++;
    }

    rocket_i8_weights_free(ctx, rw);
    rocket_i8_ctx_free(ctx);
    printf("\n==> %s (%d checks failed)\n", fails ? "FAIL" : "ALL PASS", fails);
    free(A); free(B); free(Co); free(Cr);
    return fails ? -1 : 0;
}
