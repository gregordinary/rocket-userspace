// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * whisper_decode_step_probe.c — what one Whisper decoder step costs with every matmul on the
 * NPU through resident fp16 weights, at the M=1 a decode step has (padded to 4).
 *
 * Two modes:
 *
 *   shapes <tiny|base|small> [reps]   each distinct matmul of the step alone, warm median at
 *                                     M=4, on a three-worker and a one-worker ctx, and a
 *                                     least-squares fit of call time against weight bytes:
 *                                     the intercept is what a call costs beyond its weight
 *                                     stream, the slope the stream's rate.
 *   step <tiny|base|small> [steps]    the whole step: Q/K/V fused, cross-attention Q and O,
 *                                     the MLP and the tied vocabulary projection on the NPU;
 *                                     LayerNorm, GELU and both attentions on the host, one
 *                                     thread. Per step: wall, the NPU calls' wall, the host
 *                                     rest, and process CPU time (every thread). Each step's
 *                                     logits are scored against an fp32 host step fed the same
 *                                     tokens: cosine, and whether the argmax agrees.
 *
 * The weights are random (uniform, std 1/sqrt(K)), and the cross-attention K/V a random
 * [1500][d] per layer standing in for the encoder's. Timing does not depend on the values.
 * ROCKET_DEC_WORKERS sets the step mode's worker count (default 3).
 *
 * Probe, not a gate: no verdict on the timing. The step mode fails only if a call fails or a
 * step's logits cosine falls below 0.99.
 *
 *   sudo -E ROCKET_KACC=1 ./whisper_decode_step_probe shapes base 31
 *   sudo -E ROCKET_KACC=1 ./whisper_decode_step_probe step base 64
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include "rocket_npu.h"
#include "rocket_matmul.h"

typedef _Float16 f16;

#define M_PAD 4        /* one decode row, padded to the resident path's minimum */
#define S_ENC 1500     /* encoder frames, a 30 s window */
#define N_CTX 448      /* the decoder's positional table */
#define VOCAB 51865
#define VOCAB_PAD 51872 /* N % 32 */
#define D_HEAD 64

typedef struct { const char *name; int d, L; } model_t;
static const model_t MODELS[] = { { "tiny", 384, 4 }, { "base", 512, 6 }, { "small", 768, 12 } };

static double now_ms(clockid_t c)
{
    struct timespec t;
    clock_gettime(c, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}
#define WALL() now_ms(CLOCK_MONOTONIC)
#define CPU()  now_ms(CLOCK_PROCESS_CPUTIME_ID)

static double sys_ms(void)   /* kernel time of every thread: the ioctls and cache maintenance */
{
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_stime.tv_sec * 1e3 + r.ru_stime.tv_usec * 1e-3;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n)
{
    qsort(v, n, sizeof *v, cmp_d);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static uint32_t g_seed = 0x5eed;
static float urand(void)   /* uniform in [-1, 1) */
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float)(g_seed >> 8) / 8388608.0f - 1.0f;
}

static f16 *rand_w(int N, int K)   /* [N][K], std 1/sqrt(K) */
{
    f16 *w = malloc((size_t)N * K * sizeof *w);
    float s = sqrtf(3.0f / K);
    for (size_t i = 0; i < (size_t)N * K; i++) w[i] = (f16)(urand() * s);
    return w;
}

static float *rand_v(int n, float scale, float offset)
{
    float *v = malloc((size_t)n * sizeof *v);
    for (int i = 0; i < n; i++) v[i] = offset + urand() * scale;
    return v;
}

/* ---- host kernels -------------------------------------------------------------------------- */

static void layernorm(const float *x, const float *g, const float *b, float *y, int d)
{
    double m = 0, v = 0;
    for (int i = 0; i < d; i++) m += x[i];
    m /= d;
    for (int i = 0; i < d; i++) v += (x[i] - m) * (x[i] - m);
    float r = 1.0f / sqrtf((float)(v / d) + 1e-5f);
    for (int i = 0; i < d; i++) y[i] = (float)((x[i] - m) * r) * g[i] + b[i];
}

/* dot of an fp32 query head with one fp16 key head, D_HEAD wide */
static float dot_h(const float *q, const f16 *k)
{
#ifdef __aarch64__
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    for (int j = 0; j < D_HEAD; j += 8) {
        float16x8_t kk = vld1q_f16((const __fp16 *)(k + j));
        a0 = vfmaq_f32(a0, vld1q_f32(q + j), vcvt_f32_f16(vget_low_f16(kk)));
        a1 = vfmaq_f32(a1, vld1q_f32(q + j + 4), vcvt_high_f32_f16(kk));
    }
    return vaddvq_f32(vaddq_f32(a0, a1));
#else
    float s = 0;
    for (int j = 0; j < D_HEAD; j++) s += q[j] * (float)k[j];
    return s;
#endif
}

/* acc[0..D_HEAD) += p * v (fp16 head) */
static void axpy_h(float *acc, float p, const f16 *v)
{
#ifdef __aarch64__
    float32x4_t pp = vdupq_n_f32(p);
    for (int j = 0; j < D_HEAD; j += 8) {
        float16x8_t vv = vld1q_f16((const __fp16 *)(v + j));
        vst1q_f32(acc + j, vfmaq_f32(vld1q_f32(acc + j), pp, vcvt_f32_f16(vget_low_f16(vv))));
        vst1q_f32(acc + j + 4, vfmaq_f32(vld1q_f32(acc + j + 4), pp, vcvt_high_f32_f16(vv)));
    }
#else
    for (int j = 0; j < D_HEAD; j++) acc[j] += p * (float)v[j];
#endif
}

/* One query row against n keys/values stored [n][d] fp16, all heads; out [d]. */
static void attend(const float *q, const f16 *K, const f16 *V, int n, int d, float *out, float *sc)
{
    int H = d / D_HEAD;
    const float scale = 1.0f / sqrtf((float)D_HEAD);
    for (int h = 0; h < H; h++) {
        const float *qh = q + h * D_HEAD;
        float mx = -INFINITY;
        for (int s = 0; s < n; s++) {
            sc[s] = dot_h(qh, K + (size_t)s * d + h * D_HEAD) * scale;
            if (sc[s] > mx) mx = sc[s];
        }
        float sum = 0;
        for (int s = 0; s < n; s++) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
        float *o = out + h * D_HEAD;
        memset(o, 0, D_HEAD * sizeof *o);
        for (int s = 0; s < n; s++) axpy_h(o, sc[s] / sum, V + (size_t)s * d + h * D_HEAD);
    }
}

static float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }

/* fp32 host matmul for the reference: y[N] = W[N][K] x[K] */
static void ref_mm(const f16 *W, const float *x, int K, int N, float *y)
{
    for (int n = 0; n < N; n++) {
        const f16 *w = W + (size_t)n * K;
        float s = 0;
        for (int k = 0; k < K; k++) s += (float)w[k] * x[k];
        y[n] = s;
    }
}

/* ---- the model ----------------------------------------------------------------------------- */

typedef struct {
    float *ln1g, *ln1b, *ln2g, *ln2b, *ln3g, *ln3b;
    f16 *wq, *wk, *wv, *wo, *cq, *co, *w1, *w2;     /* [N][K] */
    float *bq, *bv, *bo, *cbq, *cbo, *b1, *b2;
    f16 *ck, *cv;                                    /* cross K/V [S][d] */
    rocket_weights *qkv, *o, *cqw, *cow, *fc1, *fc2;
} layer_t;

typedef struct {
    const model_t *m;
    layer_t *ly;
    float *lnfg, *lnfb, *pos;   /* pos [N_CTX][d] */
    f16 *emb;                   /* [VOCAB_PAD][d], rows past VOCAB zero */
    rocket_weights *vocab;
    rocket_ctx *ctx;
} net_t;

static void net_init(net_t *nt, const model_t *m)
{
    int d = m->d;
    nt->m = m;
    nt->ly = calloc(m->L, sizeof *nt->ly);
    for (int l = 0; l < m->L; l++) {
        layer_t *y = &nt->ly[l];
        y->ln1g = rand_v(d, 0.2f, 1); y->ln1b = rand_v(d, 0.1f, 0);
        y->ln2g = rand_v(d, 0.2f, 1); y->ln2b = rand_v(d, 0.1f, 0);
        y->ln3g = rand_v(d, 0.2f, 1); y->ln3b = rand_v(d, 0.1f, 0);
        y->wq = rand_w(d, d); y->wk = rand_w(d, d); y->wv = rand_w(d, d); y->wo = rand_w(d, d);
        y->cq = rand_w(d, d); y->co = rand_w(d, d);
        y->w1 = rand_w(4 * d, d); y->w2 = rand_w(d, 4 * d);
        y->bq = rand_v(d, 0.1f, 0); y->bv = rand_v(d, 0.1f, 0); y->bo = rand_v(d, 0.1f, 0);
        y->cbq = rand_v(d, 0.1f, 0); y->cbo = rand_v(d, 0.1f, 0);
        y->b1 = rand_v(4 * d, 0.1f, 0); y->b2 = rand_v(d, 0.1f, 0);
        y->ck = malloc((size_t)S_ENC * d * sizeof(f16));
        y->cv = malloc((size_t)S_ENC * d * sizeof(f16));
        for (size_t i = 0; i < (size_t)S_ENC * d; i++) { y->ck[i] = (f16)urand(); y->cv[i] = (f16)urand(); }
    }
    nt->lnfg = rand_v(d, 0.2f, 1); nt->lnfb = rand_v(d, 0.1f, 0);
    nt->pos = rand_v(N_CTX * d, 0.5f, 0);
    nt->emb = calloc((size_t)VOCAB_PAD * d, sizeof(f16));
    float s = sqrtf(3.0f / d);
    for (size_t i = 0; i < (size_t)VOCAB * d; i++) nt->emb[i] = (f16)(urand() * s);
}

static int net_pack(net_t *nt, int workers)
{
    int d = nt->m->d;
    nt->ctx = rocket_ctx_create_ex(workers, ROCKET_CTX_TILING_CANONICAL);
    if (!nt->ctx) return -1;
    for (int l = 0; l < nt->m->L; l++) {
        layer_t *y = &nt->ly[l];
        const f16 *const qkv[3] = { y->wq, y->wk, y->wv };
        const int ns[3] = { d, d, d };
        y->qkv = rocket_weights_pack_seg(nt->ctx, M_PAD, d, 3 * d, qkv, ns, 3);
        y->o = rocket_weights_pack(nt->ctx, M_PAD, d, d, y->wo);
        y->cqw = rocket_weights_pack(nt->ctx, M_PAD, d, d, y->cq);
        y->cow = rocket_weights_pack(nt->ctx, M_PAD, d, d, y->co);
        y->fc1 = rocket_weights_pack(nt->ctx, M_PAD, d, 4 * d, y->w1);
        y->fc2 = rocket_weights_pack(nt->ctx, M_PAD, 4 * d, d, y->w2);
        if (!y->qkv || !y->o || !y->cqw || !y->cow || !y->fc1 || !y->fc2) return -1;
    }
    nt->vocab = rocket_weights_pack(nt->ctx, M_PAD, d, VOCAB_PAD, nt->emb);
    return nt->vocab ? 0 : -1;
}

/* Per-sequence state: the self-attention cache for one path. */
typedef struct { f16 *k, *v; } cache_t;   /* per layer [N_CTX][d] */

typedef struct {
    double mm, host;   /* ms */
    double sattn, xattn;   /* ms of the host part in the two attentions */
    int calls;
    int err;
} acct_t;

/* y[N] = W x through the NPU: x is one row, padded to M_PAD with zeros. */
static void npu_mm(net_t *nt, rocket_weights *w, int K, int N, const float *x, float *y,
                   const float *bias, f16 *A, f16 *C, acct_t *ac)
{
    for (int k = 0; k < K; k++) A[k] = (f16)x[k];
    memset(A + K, 0, (size_t)(M_PAD - 1) * K * sizeof *A);   /* the padding rows */
    double t = WALL();
    int r = rocket_matmul_fp16_prepacked(nt->ctx, M_PAD, K, N, A, C, w);
    ac->mm += WALL() - t;
    ac->calls++;
    if (r) { if (!ac->err) fprintf(stderr, "prepacked K=%d N=%d: %d\n", K, N, r); ac->err = r; }
    for (int n = 0; n < N; n++) y[n] = (float)C[n] + (bias ? bias[n] : 0.0f);
}

static void ref_lin(const f16 *W, int K, int N, const float *x, float *y, const float *bias)
{
    ref_mm(W, x, K, N, y);
    if (bias) for (int n = 0; n < N; n++) y[n] += bias[n];
}

/* One decoder step at position pos. npu != 0 runs the matmuls on the NPU. logits [VOCAB_PAD]. */
static void step(net_t *nt, cache_t *c, int token, int pos, int npu, float *logits, acct_t *ac)
{
    int d = nt->m->d, d4 = 4 * d;
    static float *x, *h, *t3, *t, *sc;
    static f16 *A, *C;
    static int cap;
    if (cap < d) {
        x = malloc(d * sizeof *x); h = malloc(d * sizeof *h); t = malloc(d4 * sizeof *t);
        t3 = malloc(3 * d * sizeof *t3); sc = malloc(S_ENC * sizeof *sc);
        A = calloc((size_t)M_PAD * d4, sizeof *A);
        C = malloc((size_t)M_PAD * VOCAB_PAD * sizeof *C);
        cap = d;
    }
    double t0 = WALL(), mm0 = ac->mm;
    for (int i = 0; i < d; i++) x[i] = (float)nt->emb[(size_t)token * d + i] + nt->pos[(size_t)pos * d + i];
    for (int l = 0; l < nt->m->L; l++) {
        layer_t *y = &nt->ly[l];
        /* self-attention */
        layernorm(x, y->ln1g, y->ln1b, h, d);
        if (npu) npu_mm(nt, y->qkv, d, 3 * d, h, t3, NULL, A, C, ac);
        else { ref_mm(y->wq, h, d, d, t3); ref_mm(y->wk, h, d, d, t3 + d); ref_mm(y->wv, h, d, d, t3 + 2 * d); }
        for (int i = 0; i < d; i++) {
            t3[i] += y->bq[i];
            t3[2 * d + i] += y->bv[i];
            c[l].k[(size_t)pos * d + i] = (f16)t3[d + i];
            c[l].v[(size_t)pos * d + i] = (f16)t3[2 * d + i];
        }
        double ta = WALL();
        attend(t3, c[l].k, c[l].v, pos + 1, d, h, sc);
        ac->sattn += WALL() - ta;
        if (npu) npu_mm(nt, y->o, d, d, h, t, y->bo, A, C, ac); else ref_lin(y->wo, d, d, h, t, y->bo);
        for (int i = 0; i < d; i++) x[i] += t[i];
        /* cross-attention */
        layernorm(x, y->ln2g, y->ln2b, h, d);
        if (npu) npu_mm(nt, y->cqw, d, d, h, t3, y->cbq, A, C, ac); else ref_lin(y->cq, d, d, h, t3, y->cbq);
        ta = WALL();
        attend(t3, y->ck, y->cv, S_ENC, d, h, sc);
        ac->xattn += WALL() - ta;
        if (npu) npu_mm(nt, y->cow, d, d, h, t, y->cbo, A, C, ac); else ref_lin(y->co, d, d, h, t, y->cbo);
        for (int i = 0; i < d; i++) x[i] += t[i];
        /* MLP */
        layernorm(x, y->ln3g, y->ln3b, h, d);
        if (npu) npu_mm(nt, y->fc1, d, d4, h, t, y->b1, A, C, ac); else ref_lin(y->w1, d, d4, h, t, y->b1);
        for (int i = 0; i < d4; i++) t[i] = gelu(t[i]);
        if (npu) npu_mm(nt, y->fc2, d4, d, t, h, y->b2, A, C, ac); else ref_lin(y->w2, d4, d, t, h, y->b2);
        for (int i = 0; i < d; i++) x[i] += h[i];
    }
    layernorm(x, nt->lnfg, nt->lnfb, h, d);
    if (npu) npu_mm(nt, nt->vocab, d, VOCAB_PAD, h, logits, NULL, A, C, ac);
    else ref_mm(nt->emb, h, d, VOCAB_PAD, logits);
    ac->host += (WALL() - t0) - (ac->mm - mm0);
}

static int argmax(const float *v, int n)
{
    int a = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[a]) a = i;
    return a;
}

static int cmd_step(const model_t *m, int nsteps)
{
    const char *we = getenv("ROCKET_DEC_WORKERS");
    int workers = we ? atoi(we) : 3;
    net_t nt;
    double t = WALL();
    net_init(&nt, m);
    if (net_pack(&nt, workers)) { fprintf(stderr, "pack failed\n"); return 1; }
    int d = m->d;
    long wbytes = 2L * m->L * (6L * d * d + 8L * d * d) + 2L * d * VOCAB_PAD;
    printf("%s: d %d L %d, %d workers, %.1f MB resident weights, packed in %.0f ms\n", m->name, d,
           m->L, workers, wbytes / 1e6, WALL() - t);
    cache_t *cn = calloc(m->L, sizeof *cn), *cr = calloc(m->L, sizeof *cr);
    for (int l = 0; l < m->L; l++) {
        cn[l].k = calloc((size_t)N_CTX * d, sizeof(f16)); cn[l].v = calloc((size_t)N_CTX * d, sizeof(f16));
        cr[l].k = calloc((size_t)N_CTX * d, sizeof(f16)); cr[l].v = calloc((size_t)N_CTX * d, sizeof(f16));
    }
    float *ln = malloc(VOCAB_PAD * sizeof *ln), *lr = malloc(VOCAB_PAD * sizeof *lr);
    double *w = malloc(nsteps * sizeof *w), *mm = malloc(nsteps * sizeof *mm),
           *hr = malloc(nsteps * sizeof *hr), *cpu = malloc(nsteps * sizeof *cpu),
           *sy = malloc(nsteps * sizeof *sy), *sa = malloc(nsteps * sizeof *sa),
           *xa = malloc(nsteps * sizeof *xa);
    const int warm = 4;
    if (nsteps > N_CTX - warm) nsteps = N_CTX - warm;
    int fail = 0, agree = 0, scored = 0;
    double worst = 1;
    for (int s = -warm; s < nsteps; s++) {
        int pos = s + warm, token = (pos * 7919 + 50258) % VOCAB;
        if (pos >= N_CTX) break;
        acct_t ac = { 0 };
        double c0 = CPU(), s0 = sys_ms(), t0 = WALL();
        step(&nt, cn, token, pos, 1, ln, &ac);
        double t1 = WALL(), c1 = CPU(), s1 = sys_ms();
        if (ac.err) { fail = 1; break; }
        acct_t ar = { 0 };
        step(&nt, cr, token, pos, 0, lr, &ar);   /* the reference, untimed, same token */
        double dot = 0, nn = 0, nr = 0;
        for (int i = 0; i < VOCAB; i++) { dot += (double)ln[i] * lr[i]; nn += (double)ln[i] * ln[i]; nr += (double)lr[i] * lr[i]; }
        double cosv = dot / sqrt(nn * nr);
        if (cosv < worst) worst = cosv;
        if (cosv < 0.99) fail = 1;
        scored++;
        agree += argmax(ln, VOCAB) == argmax(lr, VOCAB);
        if (s >= 0) { w[s] = t1 - t0; mm[s] = ac.mm; hr[s] = ac.host; cpu[s] = c1 - c0; sy[s] = s1 - s0; sa[s] = ac.sattn; xa[s] = ac.xattn; }
        if (s == 0) printf("calls per step %d\n", ac.calls);
    }
    printf("RESULT step %s workers %d steps %d step_med %.3f mm_med %.3f host_med %.3f cpu_med %.3f "
           "sys_med %.3f sattn_med %.3f xattn_med %.3f cos_min %.5f argmax_agree %d/%d\n", m->name, workers,
           nsteps, median(w, nsteps), median(mm, nsteps), median(hr, nsteps), median(cpu, nsteps),
           median(sy, nsteps), median(sa, nsteps), median(xa, nsteps), worst, agree, scored);
    return fail;
}

/* ---- shapes mode --------------------------------------------------------------------------- */

static double time_shape(rocket_ctx *ctx, int K, int N, int reps)
{
    f16 *B = rand_w(N, K), *A = calloc((size_t)M_PAD * K, sizeof *A), *C = malloc((size_t)M_PAD * N * sizeof *C);
    for (int k = 0; k < K; k++) A[k] = (f16)urand();
    rocket_weights *w = rocket_weights_pack(ctx, M_PAD, K, N, B);
    double *t = malloc(reps * sizeof *t), med = -1;
    if (!w) goto out;
    for (int r = 0; r < 3; r++) rocket_matmul_fp16_prepacked(ctx, M_PAD, K, N, A, C, w);
    for (int r = 0; r < reps; r++) {
        double t0 = WALL();
        if (rocket_matmul_fp16_prepacked(ctx, M_PAD, K, N, A, C, w)) goto out;
        t[r] = WALL() - t0;
    }
    med = median(t, reps);
out:
    if (w) rocket_weights_free(ctx, w);
    free(A); free(B); free(C); free(t);
    return med;
}

static int cmd_shapes(const model_t *m, int reps)
{
    int d = m->d;
    const int sh[][2] = { { d, d }, { d, 3 * d }, { d, 4 * d }, { 4 * d, d }, { d, VOCAB_PAD } };
    const int ns = sizeof sh / sizeof sh[0];
    const int wk[] = { 3, 1 };
    for (int wi = 0; wi < 2; wi++) {
        rocket_ctx *ctx = rocket_ctx_create_ex(wk[wi], ROCKET_CTX_TILING_CANONICAL);
        if (!ctx) { fprintf(stderr, "ctx\n"); return 1; }
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < ns; i++) {
            int K = sh[i][0], N = sh[i][1];
            double mb = 2.0 * K * N / 1e6, ms = time_shape(ctx, K, N, reps);
            if (ms < 0) { fprintf(stderr, "K=%d N=%d failed\n", K, N); return 1; }
            printf("  %s workers %d K %-5d N %-6d %7.2f MB %8.3f ms  %6.2f GB/s\n", m->name, wk[wi], K,
                   N, mb, ms, mb / ms);
            sx += mb; sy += ms; sxx += mb * mb; sxy += mb * ms;
        }
        double slope = (ns * sxy - sx * sy) / (ns * sxx - sx * sx), icpt = (sy - slope * sx) / ns;
        printf("RESULT shapes %s workers %d fixed_ms %.3f stream_GBps %.2f\n", m->name, wk[wi], icpt,
               1.0 / slope);
        rocket_ctx_free(ctx);
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s shapes|step tiny|base|small [reps|steps]\n", argv[0]);
        return 1;
    }
    const model_t *m = NULL;
    for (size_t i = 0; i < sizeof MODELS / sizeof MODELS[0]; i++)
        if (!strcmp(argv[2], MODELS[i].name)) m = &MODELS[i];
    if (!m) { fprintf(stderr, "model %s?\n", argv[2]); return 1; }
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no NPU (%d) -> SKIP\n", fd); return 2; }
    rocket_close(fd);
    int n = argc > 3 ? atoi(argv[3]) : 0;
    if (!strcmp(argv[1], "shapes")) return cmd_shapes(m, n > 0 ? n : 31);
    if (!strcmp(argv[1], "step")) return cmd_step(m, n > 0 ? n : 64);
    fprintf(stderr, "mode %s?\n", argv[1]);
    return 1;
}
