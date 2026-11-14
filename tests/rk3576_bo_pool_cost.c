// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_bo_pool_cost.c — what the RK3576 transient-BO pool's oversize costs a matmul call.
 *
 * A probe, not a gate. ROCKET_RK3576_BO_POOL=1 hands a request the smallest pooled BO that
 * covers it, with no bound on the mismatch, and PREP_BO/FINI_BO sync the whole BO. So a
 * small request served from a large BO pays the large BO's cache walk on every sync. This
 * runs one call SEQUENCE per process, so the pool's state is the sequence's alone, and
 * prints each shape's median call. The oversize's price is the difference between a
 * shape's median inside a sequence and the same shape's median run alone (`fixed`), pooled
 * both times. Run each sequence pooled and unpooled, in separate processes.
 *
 * Two entries:
 *   i8   rocket_matmul_int8_rk3576_perc_wbo(), the entry ggml-rocket's RK3576 route calls,
 *        with the weight in a device cube and the column sums made once, outside the
 *        timing. It takes the feature, coefficient, output and regcmd BOs per call.
 *   f16  rocket_matmul_fp16_rk3576(), the per-call fp16 entry, which also takes a weight BO
 *        per call. No frontend calls it on an LLM's shapes.
 *
 * Sequences (argv[1]):
 *   i8:fixed:<s>    one i8 shape, `n` calls (s = q, k, gate, down)
 *   i8:layer        Qwen2.5-1.5B's layer at M 512: q, k, v, o (K 1536; N 1536, 256, 256,
 *                   1536), gate, up (N 8960), then ffn_down as the frontend's host K-split,
 *                   four chunks of K 2304, 2304, 2304, 2048 into N 1536; `n` layers
 *   i8:LS / i8:SL   one gate call then `n` k calls, and one k call then `n` gate calls
 *   i8:DQ           one down chunk then `n` q calls (the feature BO 1.15 MiB against 0.75)
 *   f16:fixed:<s>   one f16 shape, `n` calls (s = S: 16x256x64, L: 64x2048x2048)
 *   f16:LS / f16:SL one L call then `n` S calls, and the reverse
 *
 *   rk3576_bo_pool_cost <sequence> [n]        (default n 100; the first 5 calls of each
 *                                              shape are dropped from its median)
 *
 * ROCKET_MM_PROFILE=1 adds the pool's own line at exit (gets, hits, requested against
 * served bytes, allocation and release time, peak held bytes) and the entries' phase
 * totals. Pin the governor first if the numbers are to be compared with another run.
 *
 * WHAT IT CANNOT SEE: a sequence it does not run (another model's shapes, two frontends
 * sharing one process), the pool's slot and byte caps under a longer mix, and correctness:
 * each call's return code is checked, its output is not.
 *
 * Exit 0, 1 if an entry failed, 2 with no RK3576 device.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

#define M_I8 512
#define DROP 5

typedef struct {
    const char *name;
    int M, K, N;
    int f16;
    /* i8 */
    int8_t *A, *B, *C;
    float *scale_n;
    int64_t *sumabs;
    struct rocket_rk3576_wbo *wbo;
    /* f16 */
    _Float16 *Ah, *Bh;
    float *Cf;
    /* timings */
    double *t;
    int nt, cap;
} shape_t;

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

static unsigned g_seed = 0x2545F491u;
static int rnd(int span) { g_seed = g_seed * 1103515245u + 12345u; return (int)((g_seed >> 16) % (unsigned)span); }

static int shape_init(int fd, shape_t *s, const char *name, int M, int K, int N, int f16, int cap)
{
    size_t i;
    memset(s, 0, sizeof *s);
    s->name = name; s->M = M; s->K = K; s->N = N; s->f16 = f16; s->cap = cap;
    s->t = calloc((size_t)cap, sizeof *s->t);
    if (!s->t) return -1;
    if (f16) {
        s->Ah = malloc((size_t)M * K * sizeof *s->Ah);
        s->Bh = malloc((size_t)K * N * sizeof *s->Bh);
        s->Cf = malloc((size_t)M * N * sizeof *s->Cf);
        if (!s->Ah || !s->Bh || !s->Cf) return -1;
        for (i = 0; i < (size_t)M * K; i++) s->Ah[i] = (_Float16)(rnd(17) - 8);
        for (i = 0; i < (size_t)K * N; i++) s->Bh[i] = (_Float16)(rnd(17) - 8);
        return 0;
    }
    s->A = malloc((size_t)M * K);
    s->B = malloc((size_t)N * K);
    s->C = malloc((size_t)M * N);
    s->scale_n = malloc((size_t)N * sizeof *s->scale_n);
    s->sumabs = calloc((size_t)N, sizeof *s->sumabs);
    if (!s->A || !s->B || !s->C || !s->scale_n || !s->sumabs) return -1;
    for (i = 0; i < (size_t)M * K; i++) s->A[i] = (int8_t)(rnd(255) - 127);
    for (i = 0; i < (size_t)N * K; i++) s->B[i] = (int8_t)(rnd(255) - 127);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            int v = s->B[(size_t)n * K + k];
            s->sumabs[n] += v < 0 ? -v : v;
        }
        s->scale_n[n] = 127.0f / ((float)s->sumabs[n] * 127.0f / 8.0f);
    }
    return rocket_rk3576_wbo_create(fd, K, N, s->B, &s->wbo) == ROCKET_OK ? 0 : -1;
}

static void shape_free(int fd, shape_t *s)
{
    if (s->wbo) rocket_rk3576_wbo_free(fd, s->wbo);
    free(s->A); free(s->B); free(s->C); free(s->scale_n); free(s->sumabs);
    free(s->Ah); free(s->Bh); free(s->Cf); free(s->t);
    memset(s, 0, sizeof *s);
}

static int call(int fd, shape_t *s)
{
    double t0 = now_ms();
    int rc = s->f16
        ? rocket_matmul_fp16_rk3576(fd, s->M, s->K, s->N, s->Ah, s->Bh, s->Cf)
        : rocket_matmul_int8_rk3576_perc_wbo(fd, s->M, s->K, s->N, s->A, s->wbo, NULL,
                                             s->scale_n, s->sumabs, s->C, NULL);
    double dt = now_ms() - t0;
    if (s->nt < s->cap) s->t[s->nt] = dt;
    s->nt++;
    return rc;
}

static void report(shape_t *s)
{
    int n = s->nt < s->cap ? s->nt : s->cap, k = n > DROP ? n - DROP : n;
    double *v = s->t + (n > DROP ? DROP : 0), tot = 0;
    if (!n) return;
    for (int i = 0; i < k; i++) tot += v[i];
    qsort(v, (size_t)k, sizeof *v, cmp_d);
    printf("  %-12s M %4d K %5d N %5d: %4d calls, median %8.3f ms, p10 %8.3f, p90 %8.3f, "
           "mean %8.3f (first %d dropped)\n", s->name, s->M, s->K, s->N, s->nt, v[k / 2],
           v[k / 10], v[(9 * k) / 10], tot / k, n > DROP ? DROP : 0);
}

int main(int argc, char **argv)
{
    const char *seq = argc > 1 ? argv[1] : "i8:layer";
    int n = argc > 2 ? atoi(argv[2]) : 100, fd, rc = 0;
    shape_t sh[8];
    int ns = 0;
    double t0, wall;

    if (strcmp(rocket_hw_current()->name, "rk3576") != 0) { printf("not an RK3576\n"); return 2; }
    fd = rocket_open();
    if (fd < 0) { printf("no NPU\n"); return 2; }
    memset(sh, 0, sizeof sh);
    printf("sequence %s, n %d, ROCKET_RK3576_BO_POOL=%s\n", seq, n,
           getenv("ROCKET_RK3576_BO_POOL") ? getenv("ROCKET_RK3576_BO_POOL") : "(unset)");

#define ADD_I8(nm, K_, N_, cap_) do { if (shape_init(fd, &sh[ns], nm, M_I8, K_, N_, 0, cap_)) goto setup_fail; ns++; } while (0)
#define ADD_F16(nm, M_, K_, N_, cap_) do { if (shape_init(fd, &sh[ns], nm, M_, K_, N_, 1, cap_)) goto setup_fail; ns++; } while (0)

    t0 = now_ms();
    if (!strncmp(seq, "i8:fixed:", 9)) {
        const char *w = seq + 9;
        if (!strcmp(w, "q"))         ADD_I8("q", 1536, 1536, n);
        else if (!strcmp(w, "k"))    ADD_I8("k", 1536, 256, n);
        else if (!strcmp(w, "gate")) ADD_I8("gate", 1536, 8960, n);
        else if (!strcmp(w, "down")) ADD_I8("down2304", 2304, 1536, n);
        else goto usage;
        t0 = now_ms();
        for (int i = 0; i < n && !rc; i++) rc = call(fd, &sh[0]);
    } else if (!strcmp(seq, "i8:layer")) {
        ADD_I8("q,o", 1536, 1536, 2 * n);
        ADD_I8("k,v", 1536, 256, 2 * n);
        ADD_I8("gate,up", 1536, 8960, 2 * n);
        ADD_I8("down2304", 2304, 1536, 3 * n);
        ADD_I8("down2048", 2048, 1536, n);
        t0 = now_ms();
        for (int l = 0; l < n && !rc; l++) {
            int order[] = { 0, 1, 1, 0, 2, 2, 3, 3, 3, 4 };
            for (int j = 0; j < 10 && !rc; j++) rc = call(fd, &sh[order[j]]);
        }
    } else if (!strcmp(seq, "i8:LS") || !strcmp(seq, "i8:SL") || !strcmp(seq, "i8:DQ")) {
        int ls = !strcmp(seq, "i8:LS"), dq = !strcmp(seq, "i8:DQ");
        if (dq) { ADD_I8("down2304", 2304, 1536, n); ADD_I8("q", 1536, 1536, n); }
        else if (ls) { ADD_I8("gate", 1536, 8960, n); ADD_I8("k", 1536, 256, n); }
        else { ADD_I8("k", 1536, 256, n); ADD_I8("gate", 1536, 8960, n); }
        t0 = now_ms();
        rc = call(fd, &sh[0]);
        for (int i = 0; i < n && !rc; i++) rc = call(fd, &sh[1]);
    } else if (!strncmp(seq, "f16:fixed:", 10)) {
        if (!strcmp(seq + 10, "S"))      ADD_F16("S", 16, 256, 64, n);
        else if (!strcmp(seq + 10, "L")) ADD_F16("L", 64, 2048, 2048, n);
        else goto usage;
        t0 = now_ms();
        for (int i = 0; i < n && !rc; i++) rc = call(fd, &sh[0]);
    } else if (!strcmp(seq, "f16:LS") || !strcmp(seq, "f16:SL")) {
        if (!strcmp(seq, "f16:LS")) { ADD_F16("L", 64, 2048, 2048, n); ADD_F16("S", 16, 256, 64, n); }
        else                        { ADD_F16("S", 16, 256, 64, n); ADD_F16("L", 64, 2048, 2048, n); }
        t0 = now_ms();
        rc = call(fd, &sh[0]);
        for (int i = 0; i < n && !rc; i++) rc = call(fd, &sh[1]);
    } else {
        goto usage;
    }
    wall = now_ms() - t0;
    if (rc) printf("  an entry returned %d\n", rc);
    for (int i = 0; i < ns; i++) report(&sh[i]);
    printf("  sequence wall %.1f ms\n", wall);
    for (int i = 0; i < ns; i++) shape_free(fd, &sh[i]);
    rocket_close(fd);
    return rc ? 1 : 0;

setup_fail:
    printf("setup failed at shape %d\n", ns);
    for (int i = 0; i <= ns && i < 8; i++) shape_free(fd, &sh[i]);
    rocket_close(fd);
    return 1;
usage:
    fprintf(stderr, "usage: %s i8:fixed:{q,k,gate,down}|i8:layer|i8:LS|i8:SL|i8:DQ|"
            "f16:fixed:{S,L}|f16:LS|f16:SL [n]\n", argv[0]);
    rocket_close(fd);
    return 2;
}
