// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_i32_route_cost.c — what an exact int32 route would cost an LLM layer on the RK3576,
 * against the int8-out route ggml-rocket ships.
 *
 * A probe, not a gate: it prices, and it checks the int32 arms' exactness. At Qwen2.5-1.5B's
 * projection shapes (hidden 1536, kv 256, FFN 8960) and M 512, per call:
 *
 *   i8r     the route as it ships: rocket_matmul_int8_rk3576_perc_wbo() with the weight in a
 *           device cube and its column sums supplied, both made once outside the timing,
 *           then the route's dequant (a copy of ggml-rocket's rk76_dequant, per-tensor
 *           activation scale) into f32.
 *   i8p     the same with the weight packed per call (rocket_matmul_int8_rk3576_perc_sa()),
 *           the route's form for its calibration forwards and for a weight with no cube.
 *   i32     rocket_matmul_int8_rk3576_i32() (the dense writer, K split inside the entry),
 *           then a convert of the exact int32 accumulator to f32 with exact per-column scales:
 *           out = (float)acc * a_scale * b_scale[n].
 *
 * At ffn_down's K 8960 the int8 entry refuses (K >= 6176), so its route arm is the frontend's
 * host K-split: chunks of 2304, 2304, 2304 and 2048, each its own resident GEMM, dequantized
 * and summed in f32 (`i8r_split`). The int32 side runs it whole (`i32`) and as the same four
 * chunks summed in int32 before one convert (`i32_split`).
 *
 * Two host-only arms attribute the int32 entry's host terms by copying its loops: `pack` is
 * the per-call weight scatter through conv_weight_index_int8(), `gather` is the readback
 * through rocket_rk3576_i32_dense_word() for one task of M pixels and N channels. They are
 * replicas, priced at the shape, not measurements of the entry.
 *
 * The activation quantize and rotate (7.4-7.6 ms a call at K 1536, recorded) is common to both
 * routes and is not timed here; neither is the int8 route's calibration (its first two
 * forwards), which the int32 route would not need.
 *
 * Arms are interleaved and their order rotates each round. The first ROCKET_IRC_WARM rounds
 * (default 3) are discarded; the median of the rest is reported. Pin the governor first.
 * The BO pool is whatever ROCKET_RK3576_BO_POOL says (default off); the int32 entry does not
 * use it.
 *
 * Exactness: on the first timed round each int32 arm's accumulator is compared with an exact
 * host reference (int32 inner sums, |acc| <= K*127*127 < 2^31, held in int64), every element.
 *
 *   rk3576_i32_route_cost [rounds] [shape-filter]     default 12 timed rounds
 *
 * Exit 0, 1 if an int32 arm was inexact or an entry failed unexpectedly, 2 with no RK3576.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "npu_regcmd_rk3576.h"

int conv_weight_index_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

#define M_ROWS 512
#define MAXCH  4

typedef struct { const char *name; int K, N, count; } shape_t;
static const shape_t SHAPES[] = {
    { "q,o   1536x1536", 1536, 1536, 2 },
    { "k,v   1536x256",  1536,  256, 2 },
    { "gate,up 1536x8960", 1536, 8960, 2 },
    { "down  8960x1536", 8960, 1536, 1 },
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof SHAPES[0]))
static const int DOWN_CHUNKS[] = { 2304, 2304, 2304, 2048 };

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof *v, cmp_d);
    return n ? (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0;
}

/* ---- the exact reference, row-partitioned over four threads ---- */
struct refjob { const int8_t *A, *B; int64_t *R; int M, K, N, m0, m1; };
static void *ref_thread(void *arg)
{
    struct refjob *j = arg;
    for (int m = j->m0; m < j->m1; m++) {
        const int8_t *a = j->A + (size_t)m * j->K;
        for (int n = 0; n < j->N; n++) {
            const int8_t *b = j->B + (size_t)n * j->K;
            int32_t acc = 0;
            for (int k = 0; k < j->K; k++) acc += (int32_t)a[k] * (int32_t)b[k];
            j->R[(size_t)m * j->N + n] = acc;
        }
    }
    return NULL;
}
static void ref_gemm(const int8_t *A, const int8_t *B, int64_t *R, int M, int K, int N)
{
    pthread_t th[4];
    struct refjob j[4];
    for (int t = 0; t < 4; t++) {
        j[t] = (struct refjob){ A, B, R, M, K, N, M * t / 4, M * (t + 1) / 4 };
        pthread_create(&th[t], NULL, ref_thread, &j[t]);
    }
    for (int t = 0; t < 4; t++) pthread_join(th[t], NULL);
}

/* ---- ggml-rocket's rk76_dequant, per-tensor branch, expression for expression ---- */
static void dequant8(const int8_t *src, float *dst, int M, int N, float a_scale,
                     const float *b_scale, const float *scale_n, int acc, float *f,
                     uint64_t *sat_out)
{
    uint64_t sat = 0;
    for (int n = 0; n < N; n++) f[n] = a_scale * b_scale[n] / scale_n[n];
    for (int m = 0; m < M; m++) {
        const int8_t *srow = src + (size_t)m * N;
        float *drow = dst + (size_t)m * N;
        if (acc) {
            for (int n = 0; n < N; n++) {
                const int v = srow[n];
                if (v >= 127 || v <= -127) sat++;
                drow[n] += (float)v * f[n];
            }
        } else {
            for (int n = 0; n < N; n++) {
                const int v = srow[n];
                if (v >= 127 || v <= -127) sat++;
                drow[n] = (float)v * f[n];
            }
        }
    }
    *sat_out += sat;
}

/* ---- the exact route's convert: one multiply by an exact per-column scale ---- */
static void convert32(const int32_t *src, float *dst, int M, int N, float a_scale,
                      const float *b_scale, float *f)
{
    for (int n = 0; n < N; n++) f[n] = a_scale * b_scale[n];
    for (int m = 0; m < M; m++) {
        const int32_t *srow = src + (size_t)m * N;
        float *drow = dst + (size_t)m * N;
        for (int n = 0; n < N; n++) drow[n] = (float)srow[n] * f[n];
    }
}

enum { A_I8R, A_I8P, A_I32, A_I32S, A_PACK, A_GATHER, N_ARMS };
static const char *ARM[N_ARMS] = { "i8r", "i8p", "i32", "i32_split", "pack", "gather" };

struct chunk {
    int K0, Kc;
    int8_t *A, *B;                 /* contiguous M x Kc and N x Kc slabs */
    int64_t *sumabs;
    float *scale_n;
    struct rocket_rk3576_wbo *wbo;
};

int main(int argc, char **argv)
{
    int rounds = argc > 1 ? atoi(argv[1]) : 12;
    const char *filt = argc > 2 ? argv[2] : NULL;
    int warm = getenv("ROCKET_IRC_WARM") ? atoi(getenv("ROCKET_IRC_WARM")) : 3;
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, bad = 0;
    const int M = M_ROWS;
    double layer_i8r = 0, layer_i32 = 0, layer_i32s = 0, layer_i8r_nodown = 0;

    if (strcmp(hw->name, "rk3576") != 0) { printf("not an RK3576\n"); return 2; }
    fd = rocket_open();
    if (fd < 0) { printf("no NPU\n"); return 2; }
    {
        const char *p = getenv("ROCKET_RK3576_BO_POOL");
        printf("rk3576_i32_route_cost: M %d, %d timed rounds after %d warm, BO pool %s\n",
               M, rounds, warm, (p && *p && *p != '0') ? "ON" : "off");
    }

    for (int si = 0; si < N_SHAPES; si++) {
        const shape_t *S = &SHAPES[si];
        int K = S->K, N = S->N, down = K >= 6176, nch = down ? 4 : 1;
        int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K);
        int64_t *R = malloc((size_t)M * N * sizeof *R), *Rc = malloc((size_t)M * N * sizeof *Rc);
        int8_t *C8 = malloc((size_t)M * N);
        int32_t *C32 = malloc((size_t)M * N * sizeof *C32);
        int32_t *C32b = malloc((size_t)M * N * sizeof *C32b);
        float *F = malloc((size_t)M * N * sizeof *F), *f = malloc((size_t)N * sizeof *f);
        float *b_scale = malloc((size_t)N * sizeof *b_scale);
        struct chunk ch[MAXCH];
        double t_ent[N_ARMS][64], t_host[N_ARMS][64];
        int nt = 0, have[N_ARMS] = {0};
        unsigned seed = 0x2545F491u ^ (unsigned)(K * 31 + N);
        const float a_scale = 0.0123f;
        uint64_t sat = 0;
        int i8_whole_rc;

        if (filt && !strstr(S->name, filt)) continue;
        if (!A || !B || !R || !Rc || !C8 || !C32 || !C32b || !F || !f || !b_scale) {
            printf("out of memory\n"); return 1;
        }
        memset(C32b, 0, (size_t)M * N * sizeof *C32b);
        for (size_t i = 0; i < (size_t)M * K; i++) {
            seed = seed * 1103515245u + 12345u; A[i] = (int8_t)((int)((seed >> 16) % 255u) - 127);
        }
        for (size_t i = 0; i < (size_t)N * K; i++) {
            seed = seed * 1103515245u + 12345u; B[i] = (int8_t)((int)((seed >> 16) % 255u) - 127);
        }
        for (int n = 0; n < N; n++) b_scale[n] = 0.001f * (1.0f + (float)(n % 97) / 97.0f);

        /* chunks: the route's cut for ffn_down, else the whole K */
        memset(R, 0, (size_t)M * N * sizeof *R);
        for (int c = 0, k0 = 0; c < nch; c++) {
            struct chunk *q = &ch[c];
            q->K0 = k0; q->Kc = down ? DOWN_CHUNKS[c] : K; k0 += q->Kc;
            q->A = malloc((size_t)M * q->Kc); q->B = malloc((size_t)N * q->Kc);
            q->sumabs = calloc((size_t)N, sizeof *q->sumabs);
            q->scale_n = malloc((size_t)N * sizeof *q->scale_n);
            q->wbo = NULL;
            for (int m = 0; m < M; m++) memcpy(q->A + (size_t)m * q->Kc, A + (size_t)m * K + q->K0, (size_t)q->Kc);
            for (int n = 0; n < N; n++) {
                memcpy(q->B + (size_t)n * q->Kc, B + (size_t)n * K + q->K0, (size_t)q->Kc);
                for (int k = 0; k < q->Kc; k++) {
                    int v = q->B[(size_t)n * q->Kc + k]; q->sumabs[n] += v < 0 ? -v : v;
                }
            }
            /* the chunk's own exact accumulator, for its oracle column scale */
            ref_gemm(q->A, q->B, Rc, M, q->Kc, N);
            for (int n = 0; n < N; n++) {
                int64_t mx = 1;
                for (int m = 0; m < M; m++) {
                    int64_t v = Rc[(size_t)m * N + n]; if (v < 0) v = -v; if (v > mx) mx = v;
                }
                q->scale_n[n] = (float)(127.0 / ((double)mx * 3.0));   /* CALSAFE 3.0 */
            }
            for (size_t i = 0; i < (size_t)M * N; i++) R[i] += Rc[i];
            if (rocket_rk3576_wbo_create(fd, q->Kc, N, q->B, &q->wbo) != 0) {
                printf("  %s: wbo_create refused at Kc %d\n", S->name, q->Kc); q->wbo = NULL;
            }
        }

        /* The int8 entry at the whole K: report a refusal, never work around it. */
        i8_whole_rc = rocket_matmul_int8_rk3576_perc_sa(fd, M, K, N, A, B, NULL, ch[0].scale_n,
                                                        ch[0].sumabs, C8, NULL);
        if (down)
            printf("\n== %s: int8 entry at the whole K returns %d (expected a refusal: "
                   "K >= 6176); the route arm is the 4-chunk host split ==\n", S->name,
                   i8_whole_rc);
        else
            printf("\n== %s ==\n", S->name);

        have[A_I8R] = 1; have[A_I8P] = !down; have[A_I32] = 1; have[A_I32S] = down;
        have[A_PACK] = 1; have[A_GATHER] = 1;

        for (int r = 0; r < warm + rounds; r++) {
            for (int ai = 0; ai < N_ARMS; ai++) {
                int a = (ai + r) % N_ARMS;
                double t0, t1, t2;
                int rc = 0;
                if (!have[a]) continue;
                t0 = now_ms();
                t1 = t0;
                switch (a) {
                case A_I8R:
                case A_I8P:
                    t1 = t0;
                    {
                        double ent = 0, host = 0;
                        for (int c = 0; c < nch; c++) {
                            struct chunk *q = &ch[c];
                            double s0 = now_ms(), s1;
                            rc = (a == A_I8R && q->wbo)
                                ? rocket_matmul_int8_rk3576_perc_wbo(fd, M, q->Kc, N, q->A, q->wbo,
                                                                     NULL, q->scale_n, q->sumabs,
                                                                     C8, NULL)
                                : rocket_matmul_int8_rk3576_perc_sa(fd, M, q->Kc, N, q->A, q->B,
                                                                    NULL, q->scale_n, q->sumabs,
                                                                    C8, NULL);
                            s1 = now_ms();
                            if (rc) break;
                            dequant8(C8, F, M, N, a_scale, b_scale, q->scale_n, c != 0, f, &sat);
                            ent += s1 - s0; host += now_ms() - s1;
                        }
                        t1 = t0 + ent; t2 = t1 + host;
                    }
                    break;
                case A_I32:
                    rc = rocket_matmul_int8_rk3576_i32(fd, M, K, N, A, B, NULL, C32);
                    t1 = now_ms();
                    if (!rc) convert32(C32, F, M, N, a_scale, b_scale, f);
                    t2 = now_ms();
                    if (!rc && r == warm) {
                        size_t nb = 0;
                        for (size_t i = 0; i < (size_t)M * N; i++) nb += (C32[i] != R[i]);
                        printf("  exactness i32       : %zu of %zu elements differ\n", nb,
                               (size_t)M * N);
                        if (nb) bad = 1;
                    }
                    break;
                case A_I32S:
                    {
                        double ent = 0, host = 0;
                        for (int c = 0; c < nch && !rc; c++) {
                            struct chunk *q = &ch[c];
                            double s0 = now_ms(), s1;
                            rc = rocket_matmul_int8_rk3576_i32(fd, M, q->Kc, N, q->A, q->B, NULL,
                                                               c ? C32b : C32);
                            s1 = now_ms();
                            if (rc) break;
                            if (c) for (size_t i = 0; i < (size_t)M * N; i++) C32[i] += C32b[i];
                            ent += s1 - s0; host += now_ms() - s1;
                        }
                        if (!rc) {
                            double s1 = now_ms();
                            convert32(C32, F, M, N, a_scale, b_scale, f);
                            host += now_ms() - s1;
                        }
                        t1 = t0 + ent; t2 = t1 + host;
                        if (!rc && r == warm) {
                            size_t nb = 0;
                            for (size_t i = 0; i < (size_t)M * N; i++) nb += (C32[i] != R[i]);
                            printf("  exactness i32_split : %zu of %zu elements differ\n", nb,
                                   (size_t)M * N);
                            if (nb) bad = 1;
                        }
                    }
                    break;
                case A_PACK:
                    {   /* the entry's per-call weight scatter, one 32-padded tile of all N */
                        int ocp = (N + 31) / 32 * 32;
                        size_t wb = (size_t)(ocp / 32) * ((K + 31) / 32) * 32 * 32;
                        int8_t *w = malloc(wb);
                        memset(w, 0, wb);
                        for (int n = 0; n < N; n++)
                            for (int k = 0; k < K; k++)
                                w[conv_weight_index_int8(ocp, K, 1, 1, n + 1, k + 1, 1, 1)] =
                                    B[(size_t)n * K + k];
                        t1 = now_ms();
                        free(w);
                        t2 = t1;
                    }
                    break;
                case A_GATHER:
                    {   /* the entry's readback loop, one task of M pixels */
                        const int32_t *o32 = C32b;
                        for (int m = 0; m < M; m++)
                            for (int n = 0; n < N; n++)
                                C32[(size_t)m * N + n] +=
                                    o32[rocket_rk3576_i32_dense_word((unsigned)M, (unsigned)n,
                                                                     (unsigned)m)];
                        t1 = now_ms(); t2 = t1;
                    }
                    break;
                }
                if (rc) {
                    printf("  %s returned %d\n", ARM[a], rc);
                    have[a] = 0; bad = 1; continue;
                }
                if (r >= warm) { t_ent[a][nt] = t1 - t0; t_host[a][nt] = t2 - t1; }
            }
            if (r >= warm) nt++;
        }

        printf("  %-10s %9s %9s %9s   (median of %d, ms a call)\n", "arm", "entry", "host",
               "total", nt);
        for (int a = 0; a < N_ARMS; a++) {
            double e, h, tot[64];
            if (!have[a]) continue;
            for (int i = 0; i < nt; i++) tot[i] = t_ent[a][i] + t_host[a][i];
            e = median(t_ent[a], nt); h = median(t_host[a], nt);
            {
                double tt = median(tot, nt);
                printf("  %-10s %9.2f %9.2f %9.2f%s\n", ARM[a], e, h, tt,
                       a >= A_PACK ? "   (host replica, attribution only)" : "");
                if (a == A_I8R) { layer_i8r += tt * S->count; if (!down) layer_i8r_nodown += tt * S->count; }
                if (a == A_I32) { layer_i32 += tt * S->count; if (!down) layer_i32s += tt * S->count; }
                if (a == A_I32S) layer_i32s += tt * S->count;
            }
        }
        if (sat) printf("  i8 dequant saturated codes seen: %llu\n", (unsigned long long)sat);

        for (int c = 0; c < nch; c++) {
            if (ch[c].wbo) rocket_rk3576_wbo_free(fd, ch[c].wbo);
            free(ch[c].A); free(ch[c].B); free(ch[c].sumabs); free(ch[c].scale_n);
        }
        free(A); free(B); free(R); free(Rc); free(C8); free(C32); free(C32b); free(F); free(f);
        free(b_scale);
    }

    if (!filt) {
        printf("\n== one layer: q,o + k,v + gate,up (two of each) + down ==\n");
        printf("  int8-out route, down on the CPU (the default):     %8.1f ms (+ CPU down)\n",
               layer_i8r_nodown);
        printf("  int8-out route, down as the 4-chunk split:         %8.1f ms\n", layer_i8r);
        printf("  exact int32, down whole:                           %8.1f ms\n", layer_i32);
        printf("  exact int32, down as 4 chunks summed in int32:     %8.1f ms\n", layer_i32s);
    }
    rocket_rk3576_bo_pool_drain(fd);
    rocket_close(fd);
    return bad;
}
