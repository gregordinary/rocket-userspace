// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * r76_w8a8_hostterm.c -- what a W8A8 ggml path must ADD to the int8 matmul entry,
 * timed on the part's own A72s. No NPU, no library.
 *
 * The sgemm A/B's 2.39x is a cap over the ENTRY. A ggml W8A8 prefill path also pays,
 * per micro-batch and per call, everything below; a cap that does not carry these is
 * a cap over a term rather than over the lever.
 *
 *   quant     the activation quantization the entry's int8 A comes from: one pass to
 *             find the max, one to scale-round-store [M,K] fp32 -> int8
 *   fwht      the Hadamard rotation along K that the accuracy arm says the entry needs
 *             (fp32, in place, log2(K) passes, K-contiguous so it stays in L1)
 *   deq       the output side: [M,N] int8 -> fp32 with a row scale and a column scale
 *
 * Build: gcc -O3 -march=armv8-a -o r76_w8a8_hostterm r76_w8a8_hostterm.c -lm -lpthread
 * Usage: r76_w8a8_hostterm [M K N reps threads]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int M = 512, K = 2048, N = 2048, REPS = 5, THREADS = 4;
static float *A;            /* [M,K] activations */
static int8_t *Aq;          /* [M,K] quantized  */
static int8_t *C8;          /* [M,N] int8 result */
static float *Cf;           /* [M,N] dequantized */
static float *rs, *cs;      /* row and column scales */
static int LOGK;

static void rows_of(int t, int *lo, int *hi)
{
    int per = (M + THREADS - 1) / THREADS;
    *lo = t * per;
    *hi = (*lo + per > M) ? M : *lo + per;
    if (*lo > M) *lo = M;
}

static void *w_quant(void *arg)
{
    int lo, hi, m, k;
    rows_of((int)(intptr_t)arg, &lo, &hi);
    for (m = lo; m < hi; m++) {
        const float *a = A + (size_t)m * K;
        float mx = 0.0f;
        for (k = 0; k < K; k++) { float v = fabsf(a[k]); if (v > mx) mx = v; }
        float s = (mx > 0.0f) ? 127.0f / mx : 0.0f;
        int8_t *o = Aq + (size_t)m * K;
        for (k = 0; k < K; k++) {
            float v = a[k] * s;
            int q = (int)nearbyintf(v);
            o[k] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
        }
        rs[m] = (mx > 0.0f) ? mx / 127.0f : 0.0f;
    }
    return NULL;
}

static void *w_fwht(void *arg)
{
    int lo, hi, m, s, i, j;
    rows_of((int)(intptr_t)arg, &lo, &hi);
    const float inv = 1.0f / sqrtf((float)K);
    for (m = lo; m < hi; m++) {
        float *a = A + (size_t)m * K;
        for (s = 1; s < K; s <<= 1)
            for (i = 0; i < K; i += s << 1)
                for (j = i; j < i + s; j++) {
                    float x = a[j], y = a[j + s];
                    a[j] = x + y; a[j + s] = x - y;
                }
        for (j = 0; j < K; j++) a[j] *= inv;
    }
    return NULL;
}

static void *w_deq(void *arg)
{
    int lo, hi, m, n;
    rows_of((int)(intptr_t)arg, &lo, &hi);
    for (m = lo; m < hi; m++) {
        const int8_t *c = C8 + (size_t)m * N;
        float *o = Cf + (size_t)m * N;
        float r = rs[m];
        for (n = 0; n < N; n++) o[n] = (float)c[n] * r * cs[n];
    }
    return NULL;
}

static double run(void *(*fn)(void *))
{
    pthread_t th[64];
    int t;
    double t0 = now_ms();
    for (t = 1; t < THREADS; t++) pthread_create(&th[t], NULL, fn, (void *)(intptr_t)t);
    fn((void *)(intptr_t)0);
    for (t = 1; t < THREADS; t++) pthread_join(th[t], NULL);
    return now_ms() - t0;
}

static double best(void *(*fn)(void *), const char *nm)
{
    int r;
    double b = 1e30;
    for (r = 0; r < REPS; r++) {
        double d = run(fn);
        if (r && d < b) b = d;          /* discard the first, take the best of the rest */
    }
    printf("  %-6s %7.3f ms\n", nm, b);
    return b;
}

int main(int argc, char **argv)
{
    size_t i;
    if (argc > 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    if (argc > 4) REPS = atoi(argv[4]);
    if (argc > 5) THREADS = atoi(argv[5]);
    for (LOGK = 0; (1 << LOGK) < K; LOGK++) ;
    if ((1 << LOGK) != K) { fprintf(stderr, "K must be a power of two\n"); return 2; }

    A  = malloc(sizeof *A * (size_t)M * K);
    Aq = malloc((size_t)M * K);
    C8 = malloc((size_t)M * N);
    Cf = malloc(sizeof *Cf * (size_t)M * N);
    rs = malloc(sizeof *rs * M);
    cs = malloc(sizeof *cs * N);
    if (!A || !Aq || !C8 || !Cf || !rs || !cs) return 2;
    for (i = 0; i < (size_t)M * K; i++) A[i] = (float)((i * 37 % 1000) - 500) / 500.0f;
    for (i = 0; i < (size_t)M * N; i++) C8[i] = (int8_t)((i * 13) % 255 - 128);
    for (i = 0; i < (size_t)N; i++) cs[i] = 1e-3f;

    printf("M=%d K=%d N=%d  threads=%d  best of %d after a discarded first\n",
           M, K, N, THREADS, REPS);
    double q = best(w_quant, "quant");
    double h = best(w_fwht,  "fwht");
    double d = best(w_deq,   "deq");
    printf("  ----   %7.3f ms  TOTAL host term the W8A8 port ADDS per call\n", q + h + d);
    return 0;
}
