// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_correctness_matrix_rocket.c  (packed-layout correctness matrix)
 *
 * The existing matmul gates (matmul_tiled_rocket.c etc.) feed INTEGER inputs
 * (rand()%3) whose dot products land EXACTLY on fp16 grid points. That hides
 * layout/scatter/readback corruption: a transposed tile or a +1-bank over-read
 * can still produce an exactly-representable (just wrong) value, and an
 * integer-only "max_rel<=0.02 OR max_abs<=1.0" check can pass on garbage.
 *
 * This harness instead uses REALISTIC random inputs and validates with COSINE
 * SIMILARITY (the cosine-similarity metric): a correct result tracks the reference to
 * cos ~1; ANY index/stride/tail-row corruption collapses cosine far below the
 * threshold even when a few values look plausible.
 *
 * It is DTYPE-AGNOSTIC: fp16 / int8 / int4 / int16(-exact) / bf16 / tf32, each
 * with the right reference arithmetic (exact integer for the int paths -> a
 * correct result is cos==1.0 bit-exact; fp32 for the float paths). It exercises
 * the parts it validates: feat_idx/wt_idx packing + the readback de-tile across an
 * M/K/N grid, M%4!=0 via host padding, K>8192, and the resident-weight paths.
 *
 * To keep the CPU reference affordable at FFN scale, the reference + metrics are
 * computed over a SAMPLE of output rows that always includes the first 4 and last
 * 4 rows (tail-row corruption lives there), BOTH rows on either side of every
 * M-tile boundary (where layout corruption clusters), plus an even spread. Set
 * FULL_ROWS=1 to verify every row exhaustively (opt-in; costs a full O(M*N*K) ref).
 *
 * Usage:  matmul_correctness_matrix_rocket M K N [mode] [dtype] [nthreads]
 *   mode   = tiled | mt | stream | prepacked     (default tiled)
 *            (mt/stream are fp16-only; prepacked = resident, fp16/int8/int4)
 *   dtype  = fp16 | int8 | int4 | int16 | bf16 | tf32   (default fp16)
 *            (int16 = the bit-exact int16->int64 path, rocket_matmul_int16_exact)
 *   nthreads = workers for mt/stream/prepacked   (default 4)
 * Env: COS_THRESH (default 0.9995), SAMPLE_ROWS (default 96), SEED (default 0),
 *      FULL_ROWS=1 (verify every row, ignores SAMPLE_ROWS).
 *
 * Prints one machine-parseable RESULT line + PASS/FAIL; exit 0 on PASS or SKIP.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

/* deterministic, shape-seeded PRNG (splitmix64) */
static uint64_t g_state;
static inline uint64_t sm64(void) {
    uint64_t z = (g_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline float frand(void) {  /* [-0.5, 0.5) */
    return (float)(sm64() >> 40) / (float)(1u << 24) - 0.5f;
}
static inline int irand(int lo, int hi) {  /* [lo, hi] inclusive */
    return lo + (int)(sm64() % (uint64_t)(hi - lo + 1));
}

/* bf16 truncation of an fp32 (the scatter drops the low 16 mantissa bits). */
static inline float to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4); u &= 0xFFFF0000u; memcpy(&f, &u, 4); return f;
}

/* Shared metrics over sampled rows: got[]/ref[] are nr*N doubles. */
static int report(const char *mode, const char *dtype, int M, int K, int N,
                  int Mpad, int Kt, int njobs, int nr,
                  const double *got, const double *ref, double cos_thresh) {
    double dot = 0, na = 0, nb = 0, max_abs = 0, max_rel = 0, sse = 0;
    long ncmp = (long)nr * N, nbad = 0;
    for (long i = 0; i < ncmp; i++) {
        double g = got[i], r = ref[i], ad = fabs(g - r);
        dot += g * r; na += g * g; nb += r * r;
        double rd = ad / (fabs(r) + 1e-3);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        sse += ad * ad;
        if (rd > 0.05 && ad > 0.05) nbad++;
    }
    /* Degenerate norms: only call it a match when BOTH sides are zero. The old
     * "nb==0 -> 1.0" passed a garbage NON-zero output against an all-zero reference
     * (cos is undefined there); require na==0 too so ref-zero/got-nonzero FAILs. */
    double cos = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb))
                                    : ((na == 0 && nb == 0) ? 1.0 : 0.0);
    int pass = (cos >= cos_thresh);
    printf("RESULT dt=%-5s mode=%-9s M=%-5d K=%-6d N=%-6d Mpad=%-5d Kt=%-5d njobs=%-4d rows=%-3d "
           "cos=%.6f max_abs=%.3f rmse=%.4f frac_bad=%.4f -> %s\n",
           dtype, mode, M, K, N, Mpad, Kt, njobs, nr,
           cos, max_abs, sqrt(sse / (double)ncmp), (double)nbad / (double)ncmp,
           pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("usage: %s M K N [tiled|mt|stream|prepacked] [fp16|int8|int4|int16|bf16|tf32] [nthreads]\n",
               argv[0]);
        return 2;
    }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    const char *mode  = (argc > 4) ? argv[4] : "tiled";
    const char *dtype = (argc > 5) ? argv[5] : "fp16";
    int nthreads = (argc > 6) ? atoi(argv[6]) : 4;

    double cos_thresh = getenv("COS_THRESH") ? atof(getenv("COS_THRESH")) : 0.9995;
    int sample_rows   = getenv("SAMPLE_ROWS") ? atoi(getenv("SAMPLE_ROWS")) : 96;
    uint64_t seed     = getenv("SEED") ? strtoull(getenv("SEED"), NULL, 0) : 0;
    g_state = seed ^ ((uint64_t)M * 0x100000001B3ULL) ^ ((uint64_t)K << 21)
                   ^ ((uint64_t)N << 42) ^ ((uint64_t)dtype[0] << 8);

    int Mpad = (M == 1) ? 1 : ((M + 3) & ~3);   /* M%4!=0 -> caller pads to 4 */

    /* plan is REPORT-ONLY (Kt/njobs); fp16 plan is a fine proxy for the tile count. */
    int Mt = 0, Kt = 0, Nt = 0;
    int njobs = rocket_matmul_plan(Mpad, K, N, &Mt, &Kt, &Nt);

    /* Rows to verify. Default = a SAMPLE that always pins the first 4, the last 4,
     * BOTH rows straddling every M-tile boundary (tile-edge / tail-row corruption is
     * the dominant failure mode, so we hit those exactly instead of hoping an even
     * spread lands on them), then an even spread to fill out sample_rows. FULL_ROWS=1
     * verifies EVERY row (exhaustive; an O(M*N*K) reference, so opt-in for large M). */
    int full_rows = getenv("FULL_ROWS") ? atoi(getenv("FULL_ROWS")) : 0;
    int want = full_rows ? M : (sample_rows < M ? sample_rows : M);
    int *rows = malloc((size_t)M * sizeof(int));       /* at most M distinct rows */
    char *picked = calloc((size_t)M, 1);               /* O(1) dedup */
    if (!rows || !picked) { fprintf(stderr, "host alloc failed\n"); return 3; }
    int nr = 0;
#define ADD_ROW(I) do { int _i=(I); if (_i>=0 && _i<M && !picked[_i]) { picked[_i]=1; rows[nr++]=_i; } } while (0)
    if (full_rows) {
        for (int i = 0; i < M; i++) ADD_ROW(i);
    } else {
        for (int i = 0; i < 4; i++) ADD_ROW(i);                       /* first 4 */
        for (int i = M - 4; i < M; i++) ADD_ROW(i);                   /* last 4  */
        if (Mt > 0) for (int b = Mt; b < M; b += Mt) { ADD_ROW(b-1); ADD_ROW(b); }  /* tile edges */
        for (int s = 0; nr < want && s < want; s++)                   /* even spread fills the rest */
            ADD_ROW((int)(((int64_t)(s + 1) * M) / (want + 1)));
    }
#undef ADD_ROW
    free(picked);

    double *got = calloc((size_t)nr * N, sizeof(double));
    double *ref = calloc((size_t)nr * N, sizeof(double));
    if (!got || !ref) { fprintf(stderr, "host alloc failed\n"); return 3; }

    int ret = 0, skip = 0;

#define SKIP_IF(cond, msg) do { if (cond) { \
        printf("RESULT dt=%-5s mode=%-9s M=%-5d K=%-6d N=%-6d -> SKIP (%s)\n", \
               dtype, mode, M, K, N, msg); skip = 1; goto done; } } while (0)

    if (!strcmp(dtype, "fp16")) {
        _Float16 *A = calloc((size_t)Mpad * K, sizeof(_Float16));
        _Float16 *B = malloc((size_t)N * K * sizeof(_Float16));
        _Float16 *C = malloc((size_t)Mpad * N * sizeof(_Float16));
        if (!A || !B || !C) { fprintf(stderr, "oom\n"); return 3; }
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (_Float16)frand();
        for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (_Float16)frand();
        int r;
        if (!strcmp(mode, "tiled")) {
            int fd = rocket_open(); if (fd < 0) return 3;
            r = rocket_matmul_fp16(fd, Mpad, K, N, A, B, C); rocket_close(fd);
        } else if (!strcmp(mode, "mt")) {
            r = rocket_matmul_fp16_mt(Mpad, K, N, A, B, C, nthreads);
        } else if (!strcmp(mode, "stream")) {
            rocket_stream *s = rocket_stream_create(nthreads); if (!s) return 3;
            r = rocket_matmul_fp16_stream(s, Mpad, K, N, A, B, C); rocket_stream_free(s);
        } else if (!strcmp(mode, "prepacked")) {
            rocket_ctx *ctx = rocket_ctx_create(nthreads); if (!ctx) return 3;
            rocket_weights *w = rocket_weights_pack(ctx, Mpad, K, N, B);
            if (!w) { free(A);free(B);free(C); rocket_ctx_free(ctx); SKIP_IF(1, "weights_pack declined"); }
            r = rocket_matmul_fp16_prepacked(ctx, Mpad, K, N, A, C, w);
            rocket_weights_free(ctx, w); rocket_ctx_free(ctx);
        } else { fprintf(stderr, "bad mode\n"); return 2; }
        if (r) { free(A);free(B);free(C); SKIP_IF(1, "path declined"); }
        for (int ri = 0; ri < nr; ri++) { int m = rows[ri];
            for (int n = 0; n < N; n++) { float s = 0;
                for (int k = 0; k < K; k++) s += (float)A[(size_t)m*K+k] * (float)B[(size_t)n*K+k];
                ref[(size_t)ri*N+n] = s; got[(size_t)ri*N+n] = (double)(float)C[(size_t)m*N+n]; } }
        free(A); free(B); free(C);

    } else if (!strcmp(dtype, "int8") || !strcmp(dtype, "int4")) {
        int is4 = !strcmp(dtype, "int4");
        int lo = is4 ? -2 : -64, hi = is4 ? 2 : 64;   /* int4: keep |partial|<32767 */
        int8_t *A = calloc((size_t)Mpad * K, 1);
        int8_t *B = malloc((size_t)N * K);
        int32_t *C = malloc((size_t)Mpad * N * sizeof(int32_t));
        if (!A || !B || !C) { fprintf(stderr, "oom\n"); return 3; }
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)irand(lo, hi);
        for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (int8_t)irand(lo, hi);
        int r;
        if (!strcmp(mode, "tiled")) {
            int fd = rocket_open(); if (fd < 0) return 3;
            r = is4 ? rocket_matmul_int4(fd, Mpad, K, N, A, B, C)
                    : rocket_matmul_int8(fd, Mpad, K, N, A, B, C);
            rocket_close(fd);
        } else if (!strcmp(mode, "prepacked")) {
            if (is4) {
                rocket_i4_ctx *ctx = rocket_i4_ctx_create(nthreads); if (!ctx) return 3;
                rocket_i4_weights *w = rocket_i4_weights_pack(ctx, Mpad, K, N, B);
                if (!w) { free(A);free(B);free(C); rocket_i4_ctx_free(ctx); SKIP_IF(1,"i4 pack declined"); }
                r = rocket_matmul_int4_prepacked(ctx, Mpad, K, N, A, C, w);
                rocket_i4_weights_free(ctx, w); rocket_i4_ctx_free(ctx);
            } else {
                rocket_i8_ctx *ctx = rocket_i8_ctx_create(nthreads); if (!ctx) return 3;
                rocket_i8_weights *w = rocket_i8_weights_pack(ctx, Mpad, K, N, B);
                if (!w) { free(A);free(B);free(C); rocket_i8_ctx_free(ctx); SKIP_IF(1,"i8 pack declined"); }
                r = rocket_matmul_int8_prepacked(ctx, Mpad, K, N, A, C, w);
                rocket_i8_weights_free(ctx, w); rocket_i8_ctx_free(ctx);
            }
        } else { free(A);free(B);free(C); SKIP_IF(1, "mode N/A for int dtype"); }
        if (r) { free(A);free(B);free(C); SKIP_IF(1, "path declined"); }
        for (int ri = 0; ri < nr; ri++) { int m = rows[ri];
            for (int n = 0; n < N; n++) { int64_t s = 0;
                for (int k = 0; k < K; k++) s += (int)A[(size_t)m*K+k] * (int)B[(size_t)n*K+k];
                ref[(size_t)ri*N+n] = (double)s; got[(size_t)ri*N+n] = (double)C[(size_t)m*N+n]; } }
        free(A); free(B); free(C);

    } else if (!strcmp(dtype, "int16")) {   /* bit-exact int16->int64 */
        SKIP_IF(strcmp(mode, "tiled"), "int16 = one-shot only");
        int16_t *A = calloc((size_t)Mpad * K, sizeof(int16_t));
        int16_t *B = malloc((size_t)N * K * sizeof(int16_t));
        int64_t *C = malloc((size_t)Mpad * N * sizeof(int64_t));
        if (!A || !B || !C) { fprintf(stderr, "oom\n"); return 3; }
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int16_t)irand(-1000, 1000);
        for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (int16_t)irand(-1000, 1000);
        int fd = rocket_open(); if (fd < 0) return 3;
        int r = rocket_matmul_int16_exact(fd, Mpad, K, N, A, B, C); rocket_close(fd);
        if (r) { free(A);free(B);free(C); SKIP_IF(1, "path declined"); }
        for (int ri = 0; ri < nr; ri++) { int m = rows[ri];
            for (int n = 0; n < N; n++) { int64_t s = 0;
                for (int k = 0; k < K; k++) s += (int64_t)A[(size_t)m*K+k] * (int64_t)B[(size_t)n*K+k];
                ref[(size_t)ri*N+n] = (double)s; got[(size_t)ri*N+n] = (double)C[(size_t)m*N+n]; } }
        free(A); free(B); free(C);

    } else if (!strcmp(dtype, "bf16") || !strcmp(dtype, "tf32")) {
        int isbf = !strcmp(dtype, "bf16");
        SKIP_IF(strcmp(mode, "tiled"), "bf16/tf32 = one-shot only");
        float *A = calloc((size_t)Mpad * K, sizeof(float));
        float *B = malloc((size_t)N * K * sizeof(float));
        float *C = malloc((size_t)Mpad * N * sizeof(float));
        if (!A || !B || !C) { fprintf(stderr, "oom\n"); return 3; }
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = frand();
        for (size_t i = 0; i < (size_t)N * K; i++) B[i] = frand();
        int fd = rocket_open(); if (fd < 0) return 3;
        int r = isbf ? rocket_matmul_bf16(fd, Mpad, K, N, A, B, C)
                     : rocket_matmul_tf32(fd, Mpad, K, N, A, B, C);
        rocket_close(fd);
        if (r) { free(A);free(B);free(C); SKIP_IF(1, "path declined"); }
        for (int ri = 0; ri < nr; ri++) { int m = rows[ri];
            for (int n = 0; n < N; n++) { float s = 0;
                for (int k = 0; k < K; k++) {
                    float a = A[(size_t)m*K+k], b = B[(size_t)n*K+k];
                    if (isbf) { a = to_bf16(a); b = to_bf16(b); }   /* match the scatter trunc */
                    s += a * b;
                }
                ref[(size_t)ri*N+n] = s; got[(size_t)ri*N+n] = (double)C[(size_t)m*N+n]; } }
        free(A); free(B); free(C);

    } else { fprintf(stderr, "unknown dtype '%s'\n", dtype); return 2; }

    ret = report(mode, dtype, M, K, N, Mpad, Kt, njobs, nr, got, ref, cos_thresh);

done:
    free(rows); free(got); free(ref);
    return skip ? 0 : ret;
}
