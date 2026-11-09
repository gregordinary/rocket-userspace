// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * cube_index_gate.c — rocket_cube.h's inline index helpers and blocked weight scatters
 * against the out-of-line layout helpers in npu_regcmd.c. Host only, no device.
 *
 * regcmd_golden_gate pins feature_index() and weight_*() to recorded hashes; this holds
 * every inline wrapper the pack and readback loops use to the out-of-line helper for the
 * same cube, element by element, so the two cannot drift apart without a host failure.
 * It then runs the blocked weight scatters (MM_WT_SCATTER at each granule that uses it,
 * and the int4 nibble scatter) against a per-element scatter through the index helper,
 * which is what checks the row-base and block-stride arithmetic the blocked forms add.
 *
 * The contraction lengths cover one block, several, and every remainder class a tail
 * takes; the kernel counts cross each kernel-group boundary.
 *
 * Exit 0 when every case matches, 1 otherwise.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npu_matmul.h"
#include "rocket_cube.h"

static const int KLEN[] = { 1, 7, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 100, 128, 160, 257 };
static const int NCNT[] = { 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129 };
#define NKL (int)(sizeof KLEN / sizeof KLEN[0])
#define NNC (int)(sizeof NCNT / sizeof NCNT[0])

static int fails;

static void report(const char *what, int a, int b, int c, size_t got, size_t want)
{
    if (fails++ < 20)
        fprintf(stderr, "FAIL %s (%d, %d, %d): %zu, want %zu\n", what, a, b, c, got, want);
}

/* ---- the feature cubes against feature_index() at W = 1 ---------------------------- */

static void check_cubes(void)
{
    static const struct {
        const char *name;
        size_t (*fn)(int, int, int);
        int g;
    } C[] = {
        { "feat_idx", feat_idx, 8 },          { "feat_idx_i8", feat_idx_i8, 16 },
        { "feat_idx_i4", feat_idx_i4, 32 },   { "feat_idx_i16", feat_idx_i16, 8 },
        { "feat_idx_tf32", feat_idx_tf32, 4 }, { "out_idx_i8", out_idx_i8, 4 },
        { "out_idx_i4", out_idx_i4, 8 },      { "out_idx_i16", out_idx_i16, 4 },
    };
    static const int HS[] = { 1, 3, 4, 5, 17, 64 };
    for (size_t f = 0; f < sizeof C / sizeof C[0]; f++)
        for (size_t t = 0; t < sizeof HS / sizeof HS[0]; t++)
            for (int ci = 0; ci < NKL; ci++) {
                int H = HS[t], CH = KLEN[ci];
                for (int n = 1; n <= CH; n++)
                    for (int h = 1; h <= H; h++) {
                        size_t want = (size_t)feature_index(CH, H, 1, C[f].g, n, h, 1);
                        size_t got = C[f].fn(H, n, h);
                        if (got != want) report(C[f].name, H, n, h, got, want);
                    }
            }
}

/* ---- the weight cubes against weight_*() ------------------------------------------- */

static void check_weights(void)
{
    static const struct {
        const char *name;
        size_t (*fn)(int, int, int);
        int (*ref)(int, int, int);
    } W[] = {
        { "wt_idx", wt_idx, gemm_weight_index_fp16 },
        { "wt_idx_i8", wt_idx_i8, gemm_weight_index_int8 },
        { "wt_idx_i4", wt_idx_i4, gemm_weight_index_int4 },
        { "wt_idx_i16", wt_idx_i16, gemm_weight_index_int16 },
        { "wt_idx_tf32", wt_idx_tf32, gemm_weight_index_tf32 },
    };
    for (size_t f = 0; f < sizeof W / sizeof W[0]; f++)
        for (int ki = 0; ki < NKL; ki++)
            for (int ni = 0; ni < NNC; ni++) {
                int K = KLEN[ki], N = NCNT[ni];
                for (int n = 1; n <= N; n++)
                    for (int k = 1; k <= K; k++) {
                        size_t want = (size_t)W[f].ref(K, n, k);
                        size_t got = W[f].fn(K, n, k);
                        if (got != want) report(W[f].name, K, n, k, got, want);
                    }
            }
}

/* ---- the blocked weight scatters against a per-element scatter ---------------------- */

static uint32_t rng = 20261108u;
static uint32_t next(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

/* The cube's extent: N rounded up to its kernel group, K to its contraction group. */
static size_t cube_elems(int N, int K, int ng, int kg)
{
    return (size_t)((N + ng - 1) / ng * ng) * (size_t)((K + kg - 1) / kg * kg);
}

#define CHECK_WT_SCATTER(T, KG, NG, IDX, NAME)                                            \
    do {                                                                                  \
        for (int ki = 0; ki < NKL; ki++)                                                  \
            for (int ni = 0; ni < NNC; ni++) {                                            \
                int K = KLEN[ki], N = NCNT[ni];                                           \
                size_t ne = cube_elems(N, K, NG, KG);                                     \
                T *src = malloc((size_t)N * K * sizeof(T));                               \
                T *got = malloc(ne * sizeof(T)), *ref = malloc(ne * sizeof(T));           \
                for (size_t i = 0; i < (size_t)N * K; i++) src[i] = (T)(next() & 0x7f);   \
                memset(got, 0xA5, ne * sizeof(T));                                        \
                memset(ref, 0xA5, ne * sizeof(T));                                       \
                for (int n = 1; n <= N; n++)                                              \
                    for (int k = 1; k <= K; k++)                                          \
                        ref[IDX(K, n, k)] = src[(size_t)(n - 1) * K + (k - 1)];           \
                MM_WT_SCATTER(T, T, KG, NG, IDX, got, src, K, 0, 0, N, K, MM_ID);         \
                if (memcmp(got, ref, ne * sizeof(T))) report(NAME, K, N, 0, 1, 0);        \
                free(src); free(got); free(ref);                                          \
            }                                                                             \
    } while (0)

static void check_i4_scatter(void)
{
    for (int ki = 0; ki < NKL; ki++)
        for (int ni = 0; ni < NNC; ni++) {
            int K = KLEN[ki], N = NCNT[ni];
            size_t bytes = cube_elems(N, K, 64, 32) / 2;
            int8_t *src = malloc((size_t)N * K);
            uint8_t *got = malloc(bytes), *ref = malloc(bytes);
            for (size_t i = 0; i < (size_t)N * K; i++) src[i] = (int8_t)((int)(next() & 0xf) - 8);
            memset(got, 0, bytes);
            memset(ref, 0, bytes);
            for (int n = 1; n <= N; n++)
                for (int k = 1; k <= K; k++)
                    put_nib(ref, wt_idx_i4(K, n, k), src[(size_t)(n - 1) * K + (k - 1)]);
            i4_wt_scatter(got, src, (size_t)K, 0, 0, N, K);
            if (memcmp(got, ref, bytes)) report("i4_wt_scatter", K, N, 0, 1, 0);
            free(src); free(got); free(ref);
        }
}

int main(void)
{
    check_cubes();
    check_weights();
    if (fails) {   /* the scatter checks index through these helpers; a wrong one can write past the cube */
        fprintf(stderr, "cube_index_gate: %d index mismatch(es); scatters not run\n", fails);
        return 1;
    }
    CHECK_WT_SCATTER(int8_t, 32, 32, wt_idx_i8, "MM_WT_SCATTER int8");
    CHECK_WT_SCATTER(uint16_t, 32, 16, wt_idx_i16, "MM_WT_SCATTER int16");
    CHECK_WT_SCATTER(float, 16, 16, wt_idx_tf32, "MM_WT_SCATTER tf32");
    check_i4_scatter();
    if (fails) {
        fprintf(stderr, "cube_index_gate: %d mismatch(es)\n", fails);
        return 1;
    }
    printf("cube_index_gate: every inline index and blocked weight scatter matches\n");
    return 0;
}
