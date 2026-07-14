// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int8_crossm_gw_rocket.c — cross-M reuse gate for the RESIDENT GROUP-WISE int8
 * path: a weight packed at ONE M is REUSED at other M (including a small short-prompt M)
 * with no re-pack, and stays BIT-EXACT vs the one-shot rocket_matmul_int8_groupwise at
 * each M.
 *
 * This is what makes a natively-quantized MoE expert practical. Prefill runs at whatever
 * M the router hands each expert — it varies per micro-batch and per layer — and an int8
 * expert stack is gigabytes. If the resident weight had to be re-packed whenever M
 * changed, the pack cost would land back on the hot path and swallow the win the
 * residency is there to buy.
 *
 * The scatter layout depends only on the N-split and the Nt/Kt tiling, which the
 * canonical-tileM plan (rki_worker_alloc plans at MAX_TILE, not the actual M) makes
 * M-INDEPENDENT. So rocket_matmul_int8_prepacked_gw at a different M than the pack must
 * (a) NOT return -2 (the re-pack signal) and (b) equal the one-shot exactly.
 *
 * group=576 (the MoE operating point) fits the CBUF at the worst-case tile, so Kt == group
 * in both paths and the agreement is bit-exact — see matmul_int8_prepacked_gw_rocket.c for
 * why that is a property of the group, not a given.
 *
 * CTest target matmul_int8_crossm_gw_rocket; skip-code 2 off-device.
 *   sudo ./matmul_int8_crossm_gw_rocket              # K=2880 N=512 group=576, pack M=512
 *   sudo ./matmul_int8_crossm_gw_rocket 5760 256 288 256
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int8_t rand_i8(void)    { return (int8_t)(rand() % 255 - 127); }
static float  rand_scale(void) { return 0.5f + (rand() % 1000) / 1000.0f; }

static int oneshot(int M, int K, int N, const int8_t *A, const int8_t *B,
                   const float *as, const float *bs, float *Cf, int group)
{
    int fd = rocket_open();
    if (fd < 0) return fd;
    int rc = rocket_matmul_int8_groupwise(fd, M, K, N, A, B, as, bs, Cf, group);
    rocket_close(fd);
    return rc;
}

static int cmp_exact(const char *tag, const float *got, const float *ref, size_t n)
{
    long nbad = 0; double maxd = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)ref[i]);
        if (d > maxd) maxd = d;
        if (d != 0.0 && nbad < 6)
            printf("  [%s] [%zu] oneshot=%.6f resident=%.6f\n", tag, i, ref[i], got[i]);
        if (d != 0.0) nbad++;
    }
    printf("  [%s] vs one-shot: max_abs_diff=%.6g nbad=%ld -> %s\n",
           tag, maxd, nbad, nbad ? "FAIL" : "PASS (bit-exact)");
    return nbad ? -1 : 0;
}

int main(int argc, char **argv)
{
    int K = 2880, N = 512, group = 576, PACKM = 512;
    if (argc >= 4) { K = atoi(argv[1]); N = atoi(argv[2]); group = atoi(argv[3]); }
    if (argc >= 5) PACKM = atoi(argv[4]);
    if (K % 32 || N % 32 || PACKM % 4 || group < 32 || group % 32 || K % group) {
        fprintf(stderr, "bad shape: need K%%32, N%%32, PACKM%%4, group>=32, group%%32, K%%group\n");
        return -1;
    }
    int Ms[] = { 512, 256, 768, 64, 8 };
    const int nMs = (int)(sizeof(Ms) / sizeof(Ms[0]));
    int Mmax = PACKM;
    for (int i = 0; i < nMs; i++) if (Ms[i] > Mmax) Mmax = Ms[i];
    const int nG = K / group;

    printf("cross-M resident GROUPWISE int8: pack M=%d, K=%d N=%d group=%d (nG=%d), reuse at",
           PACKM, K, N, group, nG);
    for (int i = 0; i < nMs; i++) printf(" %d", Ms[i]);
    printf("\n");

    const char *te = getenv("ROCKET_N_THREADS");
    int nthreads = te ? atoi(te) : 5;

    int8_t *A  = malloc((size_t)Mmax * K);
    int8_t *B  = malloc((size_t)N * K);
    float  *as = malloc((size_t)Mmax * nG * sizeof(float));
    float  *bs = malloc((size_t)N * nG * sizeof(float));
    float  *Co = malloc((size_t)Mmax * N * sizeof(float));
    float  *Cr = malloc((size_t)Mmax * N * sizeof(float));
    if (!A || !B || !as || !bs || !Co || !Cr) { fprintf(stderr, "alloc\n"); return -1; }
    srand(20260714);
    for (size_t i = 0; i < (size_t)Mmax * K;  i++) A[i]  = rand_i8();
    for (size_t i = 0; i < (size_t)N * K;     i++) B[i]  = rand_i8();
    for (size_t i = 0; i < (size_t)Mmax * nG; i++) as[i] = rand_scale();
    for (size_t i = 0; i < (size_t)N * nG;    i++) bs[i] = rand_scale();

    rocket_i8_ctx *ctx = rocket_i8_ctx_create(nthreads);
    if (!ctx) { fprintf(stderr, "ctx create failed — no NPU, skipping\n"); return 2; }

    /* a_scale is row-major [M][nG], so the first M rows serve any M <= Mmax */
    rocket_i8_weights *rw = rocket_i8_weights_pack_gw(ctx, PACKM, K, N, B, group);
    if (!rw) { fprintf(stderr, "pack_gw failed\n"); return -1; }

    int fails = 0;
    for (int i = 0; i < nMs; i++) {
        int M = Ms[i];
        char tag[24]; snprintf(tag, sizeof(tag), "M=%d", M);
        if (oneshot(M, K, N, A, B, as, bs, Co, group)) {
            fprintf(stderr, "oneshot %s failed\n", tag); return -1; }
        memset(Cr, 0, (size_t)M * N * sizeof(float));
        int rc = rocket_matmul_int8_prepacked_gw(ctx, M, K, N, A, as, bs, Cr, rw);
        if (rc == -2) { printf("  [%s] RE-PACK SIGNALLED (-2) — weight not reused -> FAIL\n", tag);
                        fails++; continue; }
        if (rc)       { fprintf(stderr, "  [%s] prepacked_gw failed (%d)\n", tag, rc); return -1; }
        if (cmp_exact(tag, Cr, Co, (size_t)M * N)) fails++;
    }

    rocket_i8_weights_free(ctx, rw);
    rocket_i8_ctx_free(ctx);
    printf("\n==> %s (%d checks failed)\n", fails ? "FAIL" : "ALL PASS", fails);
    free(A); free(B); free(as); free(bs); free(Co); free(Cr);
    return fails ? -1 : 0;
}
