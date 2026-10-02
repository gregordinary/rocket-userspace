// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * npu_core_diff_probe.c — how often do two RK3588 NPU cores compute the same fp16 matmul
 * differently, and on what data?
 *
 * A probe, not a gate. On two
 * RK3588s, the third core (fdad0000, mask 4) computes one element of one flash-attention AV
 * tile 2^-16 low in its fp32 accumulator, where cores 0 and 1 give the correctly rounded sum.
 * That tile's A operand is a softmax row, about half fp16 subnormals. This sizes the effect:
 * it runs fp16 x fp16 -> fp32 tiles through rocket_matmul_fp16_f32out at the AV pass-0 shape
 * (M 512, K 384, N 256), writes each output to a file, and a second mode diffs two runs made
 * on two different cores.
 *
 * Input distributions, each seeded per tile so every run computes the same tiles:
 *   rand  A and B full-mantissa uniform in [-1, 1)
 *   soft  A a softmax-like row, exp(-x) for x uniform in [0, 24] rounded to fp16, so about
 *         a quarter zero and a third subnormal; B uniform in [-1, 1)
 *   sub   A every element a random positive fp16 subnormal; B uniform in [-1, 1)
 *   int8  A and B full-range int8 through rocket_matmul_int8 (int32 out), the control: an
 *         integer accumulation is exact, so every core must agree with every other and
 *         with the exact product
 *   perm  only with NPU_CORE_DIFF_OPS naming a directory of fa_av_{a,b}.f16 (one head's real
 *         AV operands, P 512x1024 and V 256x1024, tests/fa_av_dump_wrap.c): K pass 0, with
 *         one random permutation of its 384 K indices applied to both operands per tile.
 *         Every exact dot product is unchanged and the grouping of products the hardware
 *         sums together is not. Tile 0 is the identity, the pass the FA replay flips on.
 *         NPU_CORE_DIFF_PERM picks the permutation: "all" (the default, any order), "in:B"
 *         (products move only within their own aligned B-wide K block) or "blk:B" (whole
 *         B-wide blocks move, each block's contents kept in order).
 *         NPU_CORE_DIFF_XOR=x then remaps every aligned 32-wide K block's lanes, k -> k ^ x,
 *         on top: a run on one core with x set against a run on another without it asks
 *         whether the two cores sum a 32-product group in lane orders that differ by x.
 *
 * The core is chosen OUTSIDE the probe: on the vendor kernel through the rknpu-submit
 * provider, RKNPU_CORE_MASK=1, 2 or 4 pins every job of the process to core 0, 1 or 2. The
 * mainline `rocket` driver has no such pin, so there the run needs a trace to say which core
 * ran what.
 *
 * What it does NOT show: other shapes or K depths, the fp16-output path's narrowing, or the
 * mechanism. A difference reads as a count and a size, not a cause.
 *
 * NPU_CORE_DIFF_ONLY=<name> runs and compares one distribution only.
 *
 * raw mode runs operands the host built: INDIR/raw_NNN.a16 (M x K fp16, row-major) and
 * INDIR/raw_NNN.b16 (N x K fp16), one call each, output OUTDIR/raw_NNN.f32 (M x N fp32).
 * Placing variant v of one dot product at (v, v) packs 256 experiments into a call; every
 * off-diagonal element is a further dot product of the same operands. Scoring is host-side,
 * against a model rather than against the exact sum: tools/npu_core_diff_score.c holds the
 * base model (each 32-product K group summed exactly and rounded once to fp32, the groups
 * accumulated in K order in fp32), which every core reproduces bit for bit off output lane
 * n % 16 == 0, and tools/npu_core_diff_runs.py builds the lane-0 experiments and reads them.
 * The raw mode cannot see why a lane-0 element differs, only that it does and by how much.
 * npu_regcmd.c's debug knobs ROCKET_CNA_CLK_GATE, ROCKET_CORE_MAC_GATING and
 * ROCKET_CORE_SOFT_GATING reach its calls; tools/npu_core_diff_gate.sh runs one setting per
 * core against a stored control.
 *
 * Note: perm tiles are seeded by the distribution's index, and perm moved from index 3 to 4
 * when the int8 control was added (7545e47); runs made before it need that seed to be
 * regenerated.
 *
 * Usage:
 *   npu_core_diff_probe run OUTDIR NTILES     run NTILES tiles of each distribution
 *   npu_core_diff_probe raw INDIR OUTDIR NTILES
 *   npu_core_diff_probe compare DIRA DIRB     diff two runs, per distribution
 * run prints, per distribution, elements that differ from the exact sum rounded to fp32
 * (a double sum of exact products). compare prints the differing elements' count, rate per
 * million and size, and which side is the correctly rounded one.
 * Exit: 0, 1 on a failure, 2 no NPU.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "test_fill.h"

enum { M = 512, K = 384, N = 256 };
static const char *const DIST[] = { "rand", "soft", "sub", "int8", "perm" };
enum { D_INT8 = 3, D_PERM = 4 };
static int g_ndist = 4;
static _Float16 *g_P, *g_V;          /* the real operands, K pass 0, when given */

static int load_ops(void)
{
    const char *dir = getenv("NPU_CORE_DIFF_OPS");
    enum { OK = 1024 };
    char path[4096];
    _Float16 *p = malloc((size_t)M * OK * 2), *v = malloc((size_t)N * OK * 2);
    FILE *f;
    int m, n;
    if (!dir || !*dir) return 0;
    if (!p || !v) return -1;
    snprintf(path, sizeof path, "%s/fa_av_a.f16", dir);
    f = fopen(path, "rb");
    if (!f || fread(p, 2, (size_t)M * OK, f) != (size_t)M * OK) return -1;
    fclose(f);
    snprintf(path, sizeof path, "%s/fa_av_b.f16", dir);
    f = fopen(path, "rb");
    if (!f || fread(v, 2, (size_t)N * OK, f) != (size_t)N * OK) return -1;
    fclose(f);
    g_P = malloc((size_t)M * K * 2); g_V = malloc((size_t)N * K * 2);
    if (!g_P || !g_V) return -1;
    for (m = 0; m < M; m++) memcpy(g_P + (size_t)m * K, p + (size_t)m * OK, (size_t)K * 2);
    for (n = 0; n < N; n++) memcpy(g_V + (size_t)n * K, v + (size_t)n * OK, (size_t)K * 2);
    free(p); free(v);
    g_ndist = 5;
    return 0;
}

static void make_tile(int dist, int tile, _Float16 *A, _Float16 *B)
{
    const uint64_t sa = 1000003ull * (uint64_t)(dist + 1) + (uint64_t)tile * 7919ull;
    size_t i;
    if (dist == D_PERM) {
        int pi[K], k, j, m, n, bw = 0, kind = 0;      /* kind 0 all, 1 in-block, 2 blocks */
        const char *e = getenv("NPU_CORE_DIFF_PERM");
        if (e && !strncmp(e, "in:", 3))  { kind = 1; bw = atoi(e + 3); }
        if (e && !strncmp(e, "blk:", 4)) { kind = 2; bw = atoi(e + 4); }
        if (kind && (bw <= 0 || K % bw)) { kind = 0; bw = 0; }
        for (k = 0; k < K; k++) pi[k] = k;
        if (tile && kind == 0)
            for (k = K - 1; k > 0; k--) {             /* Fisher-Yates; tile 0 identity */
                j = (int)(tf_hash(sa, (uint64_t)k) % (uint64_t)(k + 1));
                int t = pi[k]; pi[k] = pi[j]; pi[j] = t;
            }
        if (tile && kind == 1)
            for (int b0 = 0; b0 < K; b0 += bw)
                for (k = bw - 1; k > 0; k--) {
                    j = (int)(tf_hash(sa, (uint64_t)(b0 + k)) % (uint64_t)(k + 1));
                    int t = pi[b0 + k]; pi[b0 + k] = pi[b0 + j]; pi[b0 + j] = t;
                }
        if (tile && kind == 2) {
            int nb = K / bw, ord[K], b;
            for (b = 0; b < nb; b++) ord[b] = b;
            for (b = nb - 1; b > 0; b--) {
                j = (int)(tf_hash(sa, (uint64_t)b) % (uint64_t)(b + 1));
                int t = ord[b]; ord[b] = ord[j]; ord[j] = t;
            }
            for (b = 0; b < nb; b++)
                for (k = 0; k < bw; k++) pi[b * bw + k] = ord[b] * bw + k;
        }
        {
            const char *xe = getenv("NPU_CORE_DIFF_XOR");
            int x = xe ? atoi(xe) & 31 : 0, q[K];
            if (x) {
                for (k = 0; k < K; k++) q[k] = pi[(k & ~31) | ((k & 31) ^ x)];
                memcpy(pi, q, sizeof q);
            }
        }
        for (m = 0; m < M; m++)
            for (k = 0; k < K; k++) A[(size_t)m * K + k] = g_P[(size_t)m * K + pi[k]];
        for (n = 0; n < N; n++)
            for (k = 0; k < K; k++) B[(size_t)n * K + k] = g_V[(size_t)n * K + pi[k]];
        return;
    }
    for (i = 0; i < (size_t)N * K; i++)
        B[i] = (_Float16)tf_real(sa + 1, i, -1.0, 1.0);
    for (i = 0; i < (size_t)M * K; i++) {
        switch (dist) {
        case 0: A[i] = (_Float16)tf_real(sa, i, -1.0, 1.0); break;
        case 1: A[i] = (_Float16)exp(-tf_real(sa, i, 0.0, 24.0)); break;
        default: {
            /* A positive subnormal: exponent field 0, mantissa 1..1023. */
            uint16_t b = (uint16_t)tf_int(sa, i, 1, 1023);
            memcpy(&A[i], &b, 2);
        }
        }
    }
}

/* NPU_CORE_DIFF_ONLY=<name> runs one distribution. */
static int skip(int d)
{
    const char *e = getenv("NPU_CORE_DIFF_ONLY");
    return e && *e && strcmp(e, DIST[d]) != 0;
}

static void make_int8(int tile, int8_t *A, int8_t *B)
{
    const uint64_t sa = 7777777ull + (uint64_t)tile * 7919ull;
    tf_fill_i8(A, (size_t)M * K, sa, -128, 127);
    tf_fill_i8(B, (size_t)N * K, sa + 1, -128, 127);
}

static void exact_i32(const int8_t *A, const int8_t *B, int32_t *X)
{
    int m, n, k;
    for (m = 0; m < M; m++)
        for (n = 0; n < N; n++) {
            int64_t s = 0;
            for (k = 0; k < K; k++) s += (int64_t)A[(size_t)m * K + k] * B[(size_t)n * K + k];
            X[(size_t)m * N + n] = (int32_t)s;
        }
}

static void exact_f32(const _Float16 *A, const _Float16 *B, float *X)
{
    int m, n, k;
    for (m = 0; m < M; m++)
        for (n = 0; n < N; n++) {
            double s = 0.0;
            const _Float16 *a = A + (size_t)m * K, *b = B + (size_t)n * K;
            for (k = 0; k < K; k++) s += (double)a[k] * (double)b[k];
            X[(size_t)m * N + n] = (float)s;
        }
}

static uint32_t bits32(float v) { uint32_t b; memcpy(&b, &v, 4); return b; }

static int run(const char *dir, int ntiles)
{
    _Float16 *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    float *C = malloc((size_t)M * N * 4), *X = malloc((size_t)M * N * 4);
    char path[4096];
    int fd, d, t;
    if (!A || !B || !C || !X) { fprintf(stderr, "out of memory\n"); return 1; }
    fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no NPU\n"); return 2; }
    for (d = 0; d < g_ndist; d++) {
        long mis = 0;
        if (skip(d)) continue;
        size_t sub = 0, zero = 0, i;
        for (t = 0; t < ntiles; t++) {
            FILE *f;
            if (d == D_INT8) {
                int8_t *a8 = (int8_t *)A, *b8 = (int8_t *)B;
                int32_t *c32 = (int32_t *)C, *x32 = (int32_t *)X;
                make_int8(t, a8, b8);
                if (rocket_matmul_int8(fd, M, K, N, a8, b8, c32) != 0) {
                    fprintf(stderr, "int8 tile %d: matmul failed\n", t);
                    return 1;
                }
                exact_i32(a8, b8, x32);
                for (i = 0; i < (size_t)M * N; i++) mis += c32[i] != x32[i];
                goto write;
            }
            make_tile(d, t, A, B);
            if (t == 0)
                for (i = 0; i < (size_t)M * K; i++) {
                    uint16_t b; memcpy(&b, &A[i], 2);
                    if ((b & 0x7fff) == 0) zero++;
                    else if ((b & 0x7c00) == 0) sub++;
                }
            if (rocket_matmul_fp16_f32out(fd, M, K, N, A, B, C) != 0) {
                fprintf(stderr, "%s tile %d: matmul failed\n", DIST[d], t);
                return 1;
            }
            exact_f32(A, B, X);
            for (i = 0; i < (size_t)M * N; i++) mis += bits32(C[i]) != bits32(X[i]);
        write:
            snprintf(path, sizeof path, "%s/%s_%03d.f32", dir, DIST[d], t);
            f = fopen(path, "wb");
            if (!f || fwrite(C, 4, (size_t)M * N, f) != (size_t)M * N) {
                fprintf(stderr, "%s: write failed\n", path);
                return 1;
            }
            fclose(f);
        }
        printf("%-5s %d tiles: A zero %.1f%% subnormal %.1f%% (tile 0); %ld of %ld outputs "
               "differ from the exact %s\n", DIST[d], ntiles,
               100.0 * zero / ((double)M * K), 100.0 * sub / ((double)M * K), mis,
               (long)ntiles * M * N,
               d == D_INT8 ? "int32 product" : "sum rounded to fp32");
    }
    rocket_close(fd);
    free(A); free(B); free(C); free(X);
    return 0;
}

static float *load(const char *dir, int d, int t)
{
    char path[4096];
    float *p = malloc((size_t)M * N * 4);
    FILE *f;
    snprintf(path, sizeof path, "%s/%s_%03d.f32", dir, DIST[d], t);
    f = fopen(path, "rb");
    if (!p || !f || fread(p, 4, (size_t)M * N, f) != (size_t)M * N) {
        if (f) fclose(f);
        free(p);
        return NULL;
    }
    fclose(f);
    return p;
}

static int compare(const char *da, const char *db)
{
    _Float16 *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    float *X = malloc((size_t)M * N * 4);
    int d, t;
    if (!A || !B || !X) { fprintf(stderr, "out of memory\n"); return 1; }
    for (d = 0; d < g_ndist; d++) {
        if (skip(d)) continue;
        long diff = 0, a_exact = 0, b_exact = 0, neither = 0, shown = 0, n = 0;
        double maxrel = 0.0;
        for (t = 0;; t++) {
            float *ca = load(da, d, t), *cb = load(db, d, t);
            size_t i;
            if (!ca || !cb) { free(ca); free(cb); break; }
            if (d == D_INT8) {
                int32_t *x32 = (int32_t *)X;
                make_int8(t, (int8_t *)A, (int8_t *)B);
                exact_i32((const int8_t *)A, (const int8_t *)B, x32);
                for (i = 0; i < (size_t)M * N; i++) {
                    n++;
                    if (bits32(ca[i]) == bits32(cb[i])) continue;
                    diff++;
                    if (bits32(ca[i]) == (uint32_t)x32[i]) a_exact++;
                    else if (bits32(cb[i]) == (uint32_t)x32[i]) b_exact++;
                    else neither++;
                }
                free(ca); free(cb);
                continue;
            }
            make_tile(d, t, A, B);
            exact_f32(A, B, X);
            for (i = 0; i < (size_t)M * N; i++) {
                n++;
                if (bits32(ca[i]) == bits32(cb[i])) continue;
                diff++;
                if (bits32(ca[i]) == bits32(X[i])) a_exact++;
                else if (bits32(cb[i]) == bits32(X[i])) b_exact++;
                else neither++;
                {
                    double rel = fabs((double)ca[i] - (double)cb[i]) /
                                 (fabs((double)X[i]) > 0 ? fabs((double)X[i]) : 1.0);
                    if (rel > maxrel) maxrel = rel;
                }
                if (shown < 6) {
                    printf("  %-5s tile %d (%zu,%zu): %08x %.9g vs %08x %.9g, exact %.9g, "
                           "difference %.3g (2^%.2f)\n", DIST[d], t, i / N, i % N,
                           bits32(ca[i]), ca[i], bits32(cb[i]), cb[i], X[i],
                           (double)ca[i] - (double)cb[i],
                           log2(fabs((double)ca[i] - (double)cb[i])));
                    shown++;
                }
            }
            free(ca); free(cb);
        }
        printf("%-5s %ld elements over %ld tiles: %ld differ (%.2f per million); of those "
               "the first run is the exact fp32 rounding in %ld, the second in %ld, neither "
               "in %ld; largest difference %.3g of the value\n", DIST[d], n,
               n / ((long)M * N), diff, n ? 1e6 * diff / n : 0.0, a_exact, b_exact,
               neither, maxrel);
    }
    free(A); free(B); free(X);
    return 0;
}

static int read_all(const char *path, void *p, size_t n)
{
    FILE *f = fopen(path, "rb");
    size_t got;
    if (!f) return -1;
    got = fread(p, 1, n, f);
    fclose(f);
    return got == n ? 0 : -1;
}

/* One call per host-built operand pair; no reference, the host scores the output. */
static int run_raw(const char *in, const char *out, int ntiles)
{
    _Float16 *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    float *C = malloc((size_t)M * N * 4);
    char path[4096];
    int fd, t;
    if (!A || !B || !C) { fprintf(stderr, "out of memory\n"); return 1; }
    fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no NPU\n"); return 2; }
    for (t = 0; t < ntiles; t++) {
        FILE *f;
        snprintf(path, sizeof path, "%s/raw_%03d.a16", in, t);
        if (read_all(path, A, (size_t)M * K * 2)) { fprintf(stderr, "%s: read failed\n", path); return 1; }
        snprintf(path, sizeof path, "%s/raw_%03d.b16", in, t);
        if (read_all(path, B, (size_t)N * K * 2)) { fprintf(stderr, "%s: read failed\n", path); return 1; }
        if (rocket_matmul_fp16_f32out(fd, M, K, N, A, B, C) != 0) {
            fprintf(stderr, "raw tile %d: matmul failed\n", t);
            return 1;
        }
        snprintf(path, sizeof path, "%s/raw_%03d.f32", out, t);
        f = fopen(path, "wb");
        if (!f || fwrite(C, 4, (size_t)M * N, f) != (size_t)M * N) {
            fprintf(stderr, "%s: write failed\n", path);
            return 1;
        }
        fclose(f);
    }
    printf("raw %d tiles written to %s\n", ntiles, out);
    rocket_close(fd);
    free(A); free(B); free(C);
    return 0;
}

int main(int argc, char **argv)
{
    if (load_ops() < 0) { fprintf(stderr, "NPU_CORE_DIFF_OPS: cannot load the operands\n"); return 1; }
    if (argc == 5 && !strcmp(argv[1], "raw")) return run_raw(argv[2], argv[3], atoi(argv[4]));
    if (argc == 4 && !strcmp(argv[1], "run")) return run(argv[2], atoi(argv[3]));
    if (argc == 4 && !strcmp(argv[1], "compare")) return compare(argv[2], argv[3]);
    fprintf(stderr, "usage: %s run OUTDIR NTILES | raw INDIR OUTDIR NTILES | compare DIRA DIRB\n", argv[0]);
    return 1;
}
