// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_modernbert_encoder.c: the ModernBERT text encoder (and an optional torch pre-norm
 * layer stack after it) resident on the NPU. See rocket_modernbert.h for the graph, the
 * numeric plan and the batch contract.
 *
 * Per call, every per-token projection runs ONCE over the live rows of all B sequences
 * stacked into one M, against weights packed at create on an M-independent ctx. The
 * attention runs per sequence through the resident flash-attention fan-out, at the
 * sequence's own length, so padding needs no mask and costs nothing. Host work (LayerNorm,
 * RoPE fused into the head-major relayout, GeGLU, bias, residual) is row-parallel over the
 * A76 cores.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#include "rocket_modernbert.h"
#include "rocket_matmul.h"      /* rocket_ctx_create_ex + rocket_weights_pack + _prepacked */
#include "rocket_attn.h"        /* rocket_flash_attn_fp16_ctx / _ref_fp16                  */
#include "rocket_npu.h"         /* rocket_affinity_get_base                                */
#include "rocket_affinity.h"    /* rocket_pin_worker / _based                              */
#include "rocket_log.h"

#define MB_PACK_M        256    /* pack M; any M serves on the canonical ctx               */
#define MB_DOWN_LIMIT    1024.f /* down-projection input scaled to at most this magnitude  */
#define MB_DOWN_RETRIES  3

/* The weights a layer packs, in order: encoder layer l at 4l + {0 qkv, 1 o, 2 i, 3 down};
 * post layer j at 4 n_layers + 4j + {0 qkv, 1 o, 2 w1, 3 w2}. */
enum { W_QKV = 0, W_O = 1, W_UP = 2, W_DOWN = 3 };

typedef struct { const float *f; int L; float *cs; } mb_rope;   /* cs: [L][half][cos, sin] */

struct rocket_mb_ctx {
    rocket_modernbert_model m;
    int host;                     /* 1: host mode (no device)                   */
    int ht;                       /* host worker threads                        */
    rocket_ctx     *mm;
    rocket_fa_ctx  *fa;
    rocket_weights **w;
    int nw;

    size_t capM, capL;            /* grown scratch capacities (rows, tokens)    */
    int    wmax;                  /* widest GEMM operand, in elements           */
    float    *x;                  /* [capM][d] residual (fp32)                  */
    float    *p;                  /* [capM][wmax] down-projection input (fp32)  */
    _Float16 *a16;                /* [capM][wmax] GEMM A operand                */
    _Float16 *c16;                /* [capM][wmax] GEMM output                   */
    _Float16 *cc;                 /* [capM][d] attention context               */
    _Float16 *Qh, *Kh, *Vh, *Oh;  /* [capL][d] head-major (H x L x dh)          */
    _Float16 *mask; int mask_L;   /* [L][L] sliding mask, built for mask_L      */
    float    *Qf, *Kf, *Vf;       /* [capL][d] fp32 head-major (H x L x dh), host attention */
    int       attn_host;          /* 1: host attention (ROCKET_MB_ATTN=host)    */
    int       band_tile;          /* banded sliding attention's query tile, 0 off */
    _Float16 *bQ, *bK, *bV, *bO, *bmask;   /* banded call operands, grown     */
    size_t    b_cap, bm_cap;
    mb_rope   rope[4];
    float    *hw;                 /* host mode: fp32 weight staging             */
    size_t    hw_cap;
};

/* ============================================================================
 * SECTION: host parallel-for (pinned to the big cores, like the other encoders)
 * ==========================================================================*/

typedef struct {
    void (*fn)(void *, int, int); void *arg; int lo, hi, idx, core_base;
} mb_range;

static void *mb_thunk(void *p)
{
    mb_range *r = p;
    rocket_pin_worker_based(r->idx, r->core_base);
    r->fn(r->arg, r->lo, r->hi);
    return NULL;
}

static void pfor(int n, int nt, void (*fn)(void *, int, int), void *arg)
{
    if (nt < 2 || n < 2 * nt) { fn(arg, 0, n); return; }
    if (nt > 8) nt = 8;
    const int base = rocket_affinity_get_base();
    pthread_t th[8]; mb_range r[8];
    int chunk = (n + nt - 1) / nt, spawned = 0;
    for (int i = 0; i < nt; i++) {
        int lo = i * chunk, hi = lo + chunk;
        if (lo >= n) break;
        if (hi > n) hi = n;
        r[spawned] = (mb_range){ fn, arg, lo, hi, spawned, base };
        if (pthread_create(&th[spawned], NULL, mb_thunk, &r[spawned]) == 0) spawned++;
        else fn(arg, lo, hi);
    }
    for (int i = 0; i < spawned; i++) pthread_join(th[i], NULL);
}

static double mb_ms(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ============================================================================
 * SECTION: exact erf GELU of an fp16 value, as a 65536-entry fp32 table
 * ==========================================================================*/

static float g_gelu[65536];
static pthread_once_t g_gelu_once = PTHREAD_ONCE_INIT;
static void gelu_build(void)
{
    for (int u = 0; u < 65536; u++) {
        uint16_t bits = (uint16_t)u; _Float16 h; memcpy(&h, &bits, sizeof h);
        double v = (double)h;
        g_gelu[u] = (float)(0.5 * v * (1.0 + erf(v * 0.70710678118654752440)));
    }
}
static inline float gelu16(_Float16 h)
{
    uint16_t bits; memcpy(&bits, &h, sizeof bits);
    return g_gelu[bits];
}

/* ============================================================================
 * SECTION: row kernels
 * ==========================================================================*/

/* LayerNorm over d of fp32 rows: statistics in double, affine from fp16 g (NULL: identity,
 * which also skips the normalization) and b (NULL: none). Writes fp16 and/or fp32. */
typedef struct {
    const float *x; int d; const _Float16 *g, *b; float eps;
    _Float16 *o16; float *o32; int identity;
} ln_arg;

static void ln_rows(void *a, int lo, int hi)
{
    const ln_arg *s = a;
    const int d = s->d;
    for (int r = lo; r < hi; r++) {
        const float *x = s->x + (size_t)r * d;
        _Float16 *o16 = s->o16 ? s->o16 + (size_t)r * d : NULL;
        float    *o32 = s->o32 ? s->o32 + (size_t)r * d : NULL;
        if (s->identity) {
            for (int j = 0; j < d; j++) { if (o16) o16[j] = (_Float16)x[j]; if (o32) o32[j] = x[j]; }
            continue;
        }
        double mean = 0, var = 0;
        for (int j = 0; j < d; j++) mean += x[j];
        mean /= d;
        for (int j = 0; j < d; j++) { double t = x[j] - mean; var += t * t; }
        const double inv = 1.0 / sqrt(var / d + s->eps);
        for (int j = 0; j < d; j++) {
            double v = (x[j] - mean) * inv * (double)s->g[j];
            if (s->b) v += (double)s->b[j];
            if (o16) o16[j] = (_Float16)v;
            if (o32) o32[j] = (float)v;
        }
    }
}

static void layernorm(rocket_mb_ctx *c, int M, const float *x, const _Float16 *g,
                      const _Float16 *b, float eps, _Float16 *o16, float *o32)
{
    ln_arg a = { x, c->m.d, g, b, eps, o16, o32, g == NULL };
    pfor(M, c->ht, ln_rows, &a);
}

/* x[r] += scale * (y[r] + bias), fp32; flags a non-finite y. */
typedef struct {
    float *x; const _Float16 *y; const _Float16 *bias; int n, ystride; float scale; int bad;
} res_arg;

static void res_rows(void *a, int lo, int hi)
{
    res_arg *s = a;
    int bad = 0;
    for (int r = lo; r < hi; r++) {
        float *x = s->x + (size_t)r * s->n;
        const _Float16 *y = s->y + (size_t)r * s->ystride;
        for (int j = 0; j < s->n; j++) {
            float v = (float)y[j];
            if (!isfinite(v)) bad = 1;
            x[j] += s->scale * v + (s->bias ? (float)s->bias[j] : 0.f);
        }
    }
    if (bad) __atomic_store_n(&s->bad, 1, __ATOMIC_RELAXED);
}

/* A running max over the pfor chunks, merged once per chunk. */
typedef struct { pthread_mutex_t mu; float v; } mb_max;
static void max_merge(mb_max *m, float v)
{
    pthread_mutex_lock(&m->mu);
    if (v > m->v) m->v = v;
    pthread_mutex_unlock(&m->mu);
}

/* GeGLU: p[r][j] = GELU(c[r][j]) * c[r][j + dff] for the encoder MLP. */
typedef struct { const _Float16 *g; float *p; int dff; mb_max amax; } geglu_arg;

static void geglu_rows(void *a, int lo, int hi)
{
    geglu_arg *s = a;
    const int dff = s->dff;
    float mx = 0;
    for (int r = lo; r < hi; r++) {
        const _Float16 *in = s->g + (size_t)r * 2 * dff;
        float *p = s->p + (size_t)r * dff;
        for (int j = 0; j < dff; j++) {
            float v = gelu16(in[j]) * (float)in[dff + j];
            p[j] = v;
            float av = fabsf(v); if (av > mx) mx = av;
        }
    }
    max_merge(&s->amax, mx);
}

/* ReLU / GELU with bias for the post FFN: p[r][j] = act(c[r][j] + b[j]). */
typedef struct { const _Float16 *c; const _Float16 *b; float *p; int n, act; mb_max amax; } act_arg;

static void act_rows(void *a, int lo, int hi)
{
    act_arg *s = a;
    float mx = 0;
    for (int r = lo; r < hi; r++) {
        const _Float16 *in = s->c + (size_t)r * s->n;
        float *p = s->p + (size_t)r * s->n;
        for (int j = 0; j < s->n; j++) {
            float v = (float)in[j] + (float)s->b[j];
            if (s->act == 0) v = v > 0.f ? v : 0.f;
            else v = (float)(0.5 * v * (1.0 + erf((double)v * 0.70710678118654752440)));
            p[j] = v;
            float av = fabsf(v); if (av > mx) mx = av;
        }
    }
    max_merge(&s->amax, mx);
}

/* o16[r][j] = p[r][j] * s, the scaled down-projection operand. */
typedef struct { const float *p; _Float16 *o; int n; float s; } scale_arg;
static void scale_rows(void *a, int lo, int hi)
{
    const scale_arg *s = a;
    for (size_t i = (size_t)lo * s->n; i < (size_t)hi * s->n; i++) s->o[i] = (_Float16)(s->p[i] * s->s);
}

/* ============================================================================
 * SECTION: GEMMs (NPU resident, or the host reference in the same operand plan)
 * ==========================================================================*/

typedef struct { const _Float16 *A; const float *W; _Float16 *C; int K, N; } hg_arg;
static void hg_rows(void *a, int lo, int hi)
{
    const hg_arg *s = a;
    float *arow = malloc((size_t)s->K * sizeof(float));
    if (!arow) return;
    for (int r = lo; r < hi; r++) {
        const _Float16 *ar = s->A + (size_t)r * s->K;
        for (int k = 0; k < s->K; k++) arow[k] = (float)ar[k];
        for (int n = 0; n < s->N; n++) {
            const float *w = s->W + (size_t)n * s->K;
            float acc = 0;
            for (int k = 0; k < s->K; k++) acc += arow[k] * w[k];
            s->C[(size_t)r * s->N + n] = (_Float16)acc;
        }
    }
    free(arow);
}

typedef struct { const _Float16 *W; float *o; int K; } cvt_arg;
static void cvt_rows(void *a, int lo, int hi)
{
    const cvt_arg *s = a;
    for (size_t i = (size_t)lo * s->K; i < (size_t)hi * s->K; i++) s->o[i] = (float)s->W[i];
}

/* C[M][N] = A[M][K] . W[N][K]^T. Rows [Mlive, M) of A must be zero (the caller pads). */
static int gemm(rocket_mb_ctx *c, int wi, int M, int K, int N, const _Float16 *A,
                _Float16 *C, const _Float16 *W)
{
    if (!c->host) {
        int r = rocket_matmul_fp16_prepacked(c->mm, M, K, N, A, C, c->w[wi]);
        if (r) ROCKET_LOGE("rocket_modernbert: GEMM %d (M=%d K=%d N=%d) failed (%d)\n", wi, M, K, N, r);
        return r ? -3 : 0;
    }
    size_t need = (size_t)N * K;
    if (need > c->hw_cap) {
        float *nw = realloc(c->hw, need * sizeof(float));
        if (!nw) return -2;
        c->hw = nw; c->hw_cap = need;
    }
    cvt_arg ca = { W, c->hw, K };
    pfor(N, c->ht, cvt_rows, &ca);
    hg_arg ha = { A, c->hw, C, K, N };
    pfor(M, c->ht, hg_rows, &ha);
    return 0;
}

/* ============================================================================
 * SECTION: attention (RoPE fused into the head-major relayout)
 * ==========================================================================*/

static const float *rope_table(rocket_mb_ctx *c, const float *f, int L, int half)
{
    for (int i = 0; i < 4; i++)
        if (c->rope[i].f == f && c->rope[i].L >= L) return c->rope[i].cs;
    int slot = 0;
    for (int i = 0; i < 4; i++) if (c->rope[i].f == f || !c->rope[i].f) { slot = i; break; }
    mb_rope *r = &c->rope[slot];
    float *cs = realloc(r->cs, (size_t)L * half * 2 * sizeof(float));
    if (!cs) return NULL;
    /* fp32 argument then cos/sin, the way the exported graph computes its tables */
    for (int t = 0; t < L; t++)
        for (int i = 0; i < half; i++) {
            float a = (float)t * f[i];
            cs[((size_t)t * half + i) * 2]     = cosf(a);
            cs[((size_t)t * half + i) * 2 + 1] = sinf(a);
        }
    r->f = f; r->L = L; r->cs = cs;
    return cs;
}

typedef struct {
    const _Float16 *qkv;          /* rows r0.., stride 3d        */
    const _Float16 *bias;         /* [3d] or NULL                 */
    const float *cs;              /* rope table or NULL           */
    _Float16 *Q, *K, *V;
    int L, d, dh; float qs;
} relay_arg;

static void relay_heads(void *a, int lo, int hi)
{
    const relay_arg *s = a;
    const int d = s->d, dh = s->dh, half = dh / 2, L = s->L;
    float q[256], k[256];
    for (int h = lo; h < hi; h++) {
        _Float16 *Qd = s->Q + (size_t)h * L * dh, *Kd = s->K + (size_t)h * L * dh;
        _Float16 *Vd = s->V + (size_t)h * dh * L;
        for (int t = 0; t < L; t++) {
            const _Float16 *row = s->qkv + (size_t)t * 3 * d;
            for (int i = 0; i < dh; i++) {
                q[i] = (float)row[h * dh + i];
                k[i] = (float)row[d + h * dh + i];
                float v = (float)row[2 * d + h * dh + i];
                if (s->bias) {
                    q[i] += (float)s->bias[h * dh + i];
                    k[i] += (float)s->bias[d + h * dh + i];
                    v    += (float)s->bias[2 * d + h * dh + i];
                }
                Vd[(size_t)i * L + t] = (_Float16)v;
            }
            if (s->cs) {
                const float *cs = s->cs + (size_t)t * half * 2;
                for (int i = 0; i < half; i++) {
                    const float co = cs[2 * i], si = cs[2 * i + 1];
                    const float q0 = q[i], q1 = q[i + half], k0 = k[i], k1 = k[i + half];
                    q[i] = q0 * co - q1 * si;  q[i + half] = q1 * co + q0 * si;
                    k[i] = k0 * co - k1 * si;  k[i + half] = k1 * co + k0 * si;
                }
            }
            for (int i = 0; i < dh; i++) {
                Qd[(size_t)t * dh + i] = (_Float16)(q[i] * s->qs);
                Kd[(size_t)t * dh + i] = (_Float16)k[i];
            }
        }
    }
}

typedef struct { const _Float16 *O; _Float16 *cc; int L, d, dh; } unlay_arg;
static void unlay_heads(void *a, int lo, int hi)
{
    const unlay_arg *s = a;
    for (int h = lo; h < hi; h++)
        for (int t = 0; t < s->L; t++)
            memcpy(s->cc + (size_t)t * s->d + (size_t)h * s->dh,
                   s->O + ((size_t)h * s->L + t) * s->dh, (size_t)s->dh * sizeof(_Float16));
}

/* ---- host attention (ROCKET_MB_ATTN=host) ------------------------------------------------
 * fp32 attention on the A76s: q.k over the live keys only (|i-j| <= window on a sliding
 * layer), a vectorized exp, p.v accumulated in fp32. Q arrives scaled, K and V row-major per
 * head. Opt-in: it ties the NPU path at 58 tokens (241.9 against 243.6 ms, a 3-question
 * mmBERT batch) and loses at 639 (2411 against 2300 ms, ModernBERT-large), one query row at
 * a time running ~5 GFLOP/s over four A76s [HW 2026-09-27, RK1]. It is also the host mode's
 * attention. */
typedef float   mb_f4 __attribute__((vector_size(16)));
typedef int32_t mb_i4 __attribute__((vector_size(16)));

static inline mb_f4 mb_sel(mb_i4 m, mb_f4 a, mb_f4 b) { return (mb_f4)(((mb_i4)a & m) | ((mb_i4)b & ~m)); }

static inline mb_f4 mb_vexp(mb_f4 x)   /* Cephes range reduction + degree-6 polynomial, ~1 ulp */
{
    const mb_f4 hi = { 88.3762626647949f, 88.3762626647949f, 88.3762626647949f, 88.3762626647949f };
    const mb_f4 lo = -hi;
    x = mb_sel(x > hi, hi, x);
    x = mb_sel(x < lo, lo, x);
    mb_f4 fx = x * 1.44269504088896341f + 0.5f;
    mb_i4 n = __builtin_convertvector(fx, mb_i4);
    mb_f4 fn = __builtin_convertvector(n, mb_f4);
    n = n + (mb_i4)(fn > fx);
    fn = __builtin_convertvector(n, mb_f4);
    x = x - fn * 0.693359375f - fn * -2.12194440e-4f;
    mb_f4 z = x * x;
    mb_f4 y = x * 1.9875691500e-4f + 1.3981999507e-3f;
    y = y * x + 8.3334519073e-3f;
    y = y * x + 4.1665795894e-2f;
    y = y * x + 1.6666665459e-1f;
    y = y * x + 5.0000001201e-1f;
    y = y * z + x + 1.0f;
    return y * (mb_f4)((n + 127) << 23);
}

typedef struct {
    const float *Q, *K, *V;       /* [H][L][dh] */
    _Float16 *cc;                 /* [L][d] output rows */
    int L, d, dh, nblk, rb, window;   /* window < 0: global */
} hattn_arg;

static void hattn_items(void *a, int lo, int hi)
{
    const hattn_arg *s = a;
    const int L = s->L, dh = s->dh, nv = dh / 4;
    float *sc = malloc(((size_t)L + 4) * sizeof(float));
    if (!sc) return;
    for (int w = lo; w < hi; w++) {
        const int h = w / s->nblk, b = w % s->nblk;
        const float *Q = s->Q + (size_t)h * L * dh, *K = s->K + (size_t)h * L * dh, *V = s->V + (size_t)h * L * dh;
        const int r0 = b * s->rb, r1 = r0 + s->rb < L ? r0 + s->rb : L;
        for (int i = r0; i < r1; i++) {
            const int j0 = s->window < 0 ? 0 : (i - s->window > 0 ? i - s->window : 0);
            const int j1 = s->window < 0 ? L : (i + s->window + 1 < L ? i + s->window + 1 : L);
            const int n = j1 - j0;
            mb_f4 q[64];                                 /* dh <= 256 */
            memcpy(q, Q + (size_t)i * dh, (size_t)dh * sizeof(float));
            float mx = -INFINITY;
            for (int j = 0; j < n; j++) {
                const float *kr = K + (size_t)(j0 + j) * dh;
                mb_f4 acc = { 0, 0, 0, 0 };
                for (int v = 0; v < nv; v++) { mb_f4 k4; memcpy(&k4, kr + 4 * v, sizeof k4); acc += q[v] * k4; }
                const float sv = acc[0] + acc[1] + acc[2] + acc[3];
                sc[j] = sv;
                if (sv > mx) mx = sv;
            }
            for (int j = n; j < ((n + 3) & ~3); j++) sc[j] = -INFINITY;
            const mb_f4 m4 = { mx, mx, mx, mx };
            mb_f4 vs = { 0, 0, 0, 0 };
            for (int j = 0; j < n; j += 4) {
                mb_f4 v4; memcpy(&v4, sc + j, sizeof v4);
                v4 = mb_vexp(v4 - m4);
                memcpy(sc + j, &v4, sizeof v4);
                vs += v4;
            }
            const float inv = 1.f / (vs[0] + vs[1] + vs[2] + vs[3]);
            mb_f4 o[64];
            for (int v = 0; v < nv; v++) o[v] = (mb_f4){ 0, 0, 0, 0 };
            for (int j = 0; j < n; j++) {
                const float *vr = V + (size_t)(j0 + j) * dh;
                const mb_f4 p4 = { sc[j], sc[j], sc[j], sc[j] };
                for (int v = 0; v < nv; v++) { mb_f4 v4; memcpy(&v4, vr + 4 * v, sizeof v4); o[v] += p4 * v4; }
            }
            _Float16 *dst = s->cc + (size_t)i * s->d + (size_t)h * dh;
            for (int v = 0; v < nv; v++)
                for (int e = 0; e < 4; e++) dst[4 * v + e] = (_Float16)(o[v][e] * inv);
        }
    }
    free(sc);
}

typedef struct {
    const _Float16 *qkv; const _Float16 *bias; const float *cs;
    float *Q, *K, *V; int L, d, dh; float qs;
} relayf_arg;

static void relayf_heads(void *a, int lo, int hi)
{
    const relayf_arg *s = a;
    const int d = s->d, dh = s->dh, half = dh / 2, L = s->L;
    for (int h = lo; h < hi; h++) {
        float *Qd = s->Q + (size_t)h * L * dh, *Kd = s->K + (size_t)h * L * dh, *Vd = s->V + (size_t)h * L * dh;
        for (int t = 0; t < L; t++) {
            const _Float16 *row = s->qkv + (size_t)t * 3 * d;
            float *q = Qd + (size_t)t * dh, *k = Kd + (size_t)t * dh, *v = Vd + (size_t)t * dh;
            for (int i = 0; i < dh; i++) {
                q[i] = (float)row[h * dh + i];
                k[i] = (float)row[d + h * dh + i];
                v[i] = (float)row[2 * d + h * dh + i];
                if (s->bias) {
                    q[i] += (float)s->bias[h * dh + i];
                    k[i] += (float)s->bias[d + h * dh + i];
                    v[i] += (float)s->bias[2 * d + h * dh + i];
                }
            }
            if (s->cs) {
                const float *cs = s->cs + (size_t)t * half * 2;
                for (int i = 0; i < half; i++) {
                    const float co = cs[2 * i], si = cs[2 * i + 1];
                    const float q0 = q[i], q1 = q[i + half], k0 = k[i], k1 = k[i + half];
                    q[i] = q0 * co - q1 * si;  q[i + half] = q1 * co + q0 * si;
                    k[i] = k0 * co - k1 * si;  k[i + half] = k1 * co + k0 * si;
                }
            }
            for (int i = 0; i < dh; i++) q[i] *= s->qs;
        }
    }
}


/* ---- banded sliding attention (ROCKET_MB_BAND_TILE, default 128) ------------------------
 * A sliding layer's query i sees keys |i-j| <= w. Split the queries into tiles of T rows; an
 * interior tile [q0, q0+T) needs keys [q0-w, q0+T+w), and its band RELATIVE to that key window
 * is the same for every interior tile. So all interior tiles of all heads run as ONE flash-
 * attention call whose heads are the (tile, head) pairs and whose one mask is that band; the
 * first and last tiles, whose windows clip at the sequence ends, take a call each. */
typedef struct {
    rocket_mb_ctx *c; int L, dh, H, nq, nk; const int *q0, *k0; int d;
    _Float16 *cc;
} band_arg;

static void band_pack(void *a, int lo, int hi)   /* items [lo,hi): item = tile*H + head */
{
    const band_arg *s = a;
    const rocket_mb_ctx *c = s->c;
    const int L = s->L, dh = s->dh, nq = s->nq, nk = s->nk;
    for (int it = lo; it < hi; it++) {
        const int t = it / s->H, h = it % s->H, q0 = s->q0[t], k0 = s->k0[t];
        memcpy(c->bQ + (size_t)it * nq * dh, c->Qh + ((size_t)h * L + q0) * dh, (size_t)nq * dh * sizeof(_Float16));
        memcpy(c->bK + (size_t)it * nk * dh, c->Kh + ((size_t)h * L + k0) * dh, (size_t)nk * dh * sizeof(_Float16));
        for (int e = 0; e < dh; e++)
            memcpy(c->bV + ((size_t)it * dh + e) * nk, c->Vh + ((size_t)h * dh + e) * L + k0,
                   (size_t)nk * sizeof(_Float16));
    }
}

static void band_unpack(void *a, int lo, int hi)
{
    const band_arg *s = a;
    const int dh = s->dh, nq = s->nq;
    for (int it = lo; it < hi; it++) {
        const int t = it / s->H, h = it % s->H;
        for (int i = 0; i < nq; i++)
            memcpy(s->cc + (size_t)(s->q0[t] + i) * s->d + (size_t)h * dh,
                   s->c->bO + ((size_t)it * nq + i) * dh, (size_t)dh * sizeof(_Float16));
    }
}

/* One call over `nt` tiles that share (nq, nk, q0-k0). */
static int band_call(rocket_mb_ctx *c, int L, int n_head, int dh, float fa_scale, int nt,
                     const int *q0, const int *k0, int nq, int nk, _Float16 *cc_rows)
{
    const int items = nt * n_head, w = c->m.window, rel = q0[0] - k0[0];
    const size_t need = (size_t)items * (nq > nk ? nq : nk) * dh;
    if (need > c->b_cap) {
        free(c->bQ); free(c->bK); free(c->bV); free(c->bO);
        c->bQ = malloc(need * sizeof(_Float16)); c->bK = malloc(need * sizeof(_Float16));
        c->bV = malloc(need * sizeof(_Float16)); c->bO = malloc(need * sizeof(_Float16));
        c->b_cap = (c->bQ && c->bK && c->bV && c->bO) ? need : 0;
        if (!c->b_cap) return -2;
    }
    if ((size_t)nq * nk > c->bm_cap) {
        free(c->bmask);
        c->bmask = malloc((size_t)nq * nk * sizeof(_Float16));
        c->bm_cap = c->bmask ? (size_t)nq * nk : 0;
        if (!c->bmask) return -2;
    }
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < nk; j++)
            c->bmask[(size_t)i * nk + j] = (abs(i + rel - j) <= w) ? (_Float16)0.f : (_Float16)-INFINITY;
    band_arg ba = { c, L, dh, n_head, nq, nk, q0, k0, c->m.d, cc_rows };
    pfor(items, c->ht, band_pack, &ba);
    const int r = rocket_flash_attn_fp16_ctx(c->fa, nq, nk, dh, dh, items, items, fa_scale, 0.f,
                                             c->bQ, c->bK, c->bV, c->bmask, c->bO);
    if (r) { ROCKET_LOGE("rocket_modernbert: banded attention (%d x %d, %d items) failed (%d)\n", nq, nk, items, r); return -4; }
    pfor(items, c->ht, band_unpack, &ba);
    return 0;
}

/* The banded plan for a sliding layer of L tokens, Qh/Kh/Vh already relaid. */
static int band_attend(rocket_mb_ctx *c, int L, int n_head, int dh, float fa_scale, _Float16 *cc_rows)
{
    const int T = c->band_tile, w = c->m.window;
    int q0[64], k0[64], nt = 0;
    /* first tile: no left halo */
    q0[0] = 0; k0[0] = 0;
    int r = band_call(c, L, n_head, dh, fa_scale, 1, q0, k0, T, T + w, cc_rows);
    if (r) return r;
    /* interior tiles: a full window on both sides */
    int q = T;
    for (; q + T + w <= L && nt < 64; q += T) { q0[nt] = q; k0[nt] = q - w; nt++; }
    if (nt && (r = band_call(c, L, n_head, dh, fa_scale, nt, q0, k0, T, T + 2 * w, cc_rows))) return r;
    /* last tile: the rest, window clipped at L */
    if (q < L) { q0[0] = q; k0[0] = q - w; r = band_call(c, L, n_head, dh, fa_scale, 1, q0, k0, L - q, L - (q - w), cc_rows); }
    return r;
}

static const _Float16 *sliding_mask(rocket_mb_ctx *c, int L)
{
    if (c->mask_L == L) return c->mask;
    const int w = c->m.window;
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            c->mask[(size_t)i * L + j] = (abs(i - j) <= w) ? (_Float16)0.f : (_Float16)-INFINITY;
    c->mask_L = L;
    return c->mask;
}

/* One sequence's attention over its L live rows. qkv points at its first row. */
static int attend(rocket_mb_ctx *c, const _Float16 *qkv, const _Float16 *bias, const float *inv_freq,
                  int sliding, int n_head, int L, _Float16 *cc_rows)
{
    const int d = c->m.d, dh = d / n_head;
    const float scale = 1.f / sqrtf((float)dh);
    int e2; const float mant = frexpf(scale, &e2);
    const int exact = (mant == 0.5f);        /* 1/sqrt(dh) a power of two: fold it into q */
    const float *cs = NULL;
    if (inv_freq) { cs = rope_table(c, inv_freq, L, dh / 2); if (!cs) return -2; }
    if (c->host || c->attn_host) {
        relayf_arg fa = { qkv, bias, cs, c->Qf, c->Kf, c->Vf, L, d, dh, scale };
        pfor(n_head, c->ht, relayf_heads, &fa);
        const int rb = 32, nblk = (L + rb - 1) / rb;
        hattn_arg ha = { c->Qf, c->Kf, c->Vf, cc_rows, L, d, dh, nblk, rb, sliding ? c->m.window : -1 };
        pfor(n_head * nblk, c->ht, hattn_items, &ha);
        return 0;
    }
    relay_arg ra = { qkv, bias, cs, c->Qh, c->Kh, c->Vh, L, d, dh, exact ? scale : 1.f };
    pfor(n_head, c->ht, relay_heads, &ra);
    /* banded when the band leaves most of the score matrix out: 64 interior tiles cap L */
    if (sliding && c->band_tile > 0 && L >= 2 * c->band_tile + 2 * c->m.window &&
        L <= 65 * c->band_tile)
        return band_attend(c, L, n_head, dh, exact ? 1.f : scale, cc_rows);
    const _Float16 *mask = sliding ? sliding_mask(c, L) : NULL;
    const float fa_scale = exact ? 1.f : scale;
    const int r = rocket_flash_attn_fp16_ctx(c->fa, L, L, dh, dh, n_head, n_head, fa_scale, 0.f,
                                             c->Qh, c->Kh, c->Vh, mask, c->Oh);
    if (r) { ROCKET_LOGE("rocket_modernbert: attention (L=%d H=%d) failed (%d)\n", L, n_head, r); return -4; }
    unlay_arg ua = { c->Oh, cc_rows, L, d, dh };
    pfor(n_head, c->ht, unlay_heads, &ua);
    return 0;
}

/* ============================================================================
 * SECTION: lifecycle
 * ==========================================================================*/

static int mb_check(const rocket_modernbert_model *m)
{
    if (!m || m->d <= 0 || m->d % 32 || m->n_head <= 0 || m->d % m->n_head) return -1;
    if ((m->d / m->n_head) % 32 || m->d / m->n_head > 256) return -1;
    if (m->n_layers < 1 || m->n_layers > ROCKET_MB_MAX_LAYERS) return -1;
    if (m->d_ff <= 0 || m->d_ff % 32 || m->window < 0) return -1;
    if (!m->emb_norm_g || !m->final_norm_g) return -1;
    for (int l = 0; l < m->n_layers; l++) {
        const rocket_mb_layer *L = &m->layers[l];
        if (!L->Wqkv || !L->Wo || !L->mlp_norm_g || !L->Wi || !L->Wo_mlp || !L->inv_freq) return -1;
    }
    if (m->n_post < 0 || m->n_post > ROCKET_MB_MAX_POST) return -1;
    if (m->n_post) {
        if (m->post_n_head <= 0 || m->d % m->post_n_head) return -1;
        if ((m->d / m->post_n_head) % 32 || m->d / m->post_n_head > 256) return -1;
        if (m->post_d_ff <= 0 || m->post_d_ff % 32 || m->post_act < 0 || m->post_act > 1) return -1;
        for (int j = 0; j < m->n_post; j++) {
            const rocket_mb_post_layer *P = &m->post[j];
            if (!P->ln1_g || !P->ln1_b || !P->Wqkv || !P->bqkv || !P->Wo || !P->bo ||
                !P->ln2_g || !P->ln2_b || !P->W1 || !P->b1 || !P->W2 || !P->b2) return -1;
        }
    }
    return 0;
}

rocket_mb_ctx *rocket_modernbert_ctx_create(const rocket_modernbert_model *m, int nthreads)
{
    if (mb_check(m)) { ROCKET_LOGE("rocket_modernbert_ctx_create: invalid model\n"); return NULL; }
    rocket_mb_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->m = *m;
    c->host = (nthreads == 0);
    c->ht = 4;
    c->mask_L = -1;
    {
        const char *am = getenv("ROCKET_MB_ATTN");
        c->attn_host = am && !strcmp(am, "host");
    }
    {
        const char *bt = getenv("ROCKET_MB_BAND_TILE");
        c->band_tile = bt ? atoi(bt) : 128;
        if (c->band_tile < 0) c->band_tile = 0;
    }
    c->wmax = 3 * m->d;
    if (2 * m->d_ff > c->wmax) c->wmax = 2 * m->d_ff;
    if (m->n_post && m->post_d_ff > c->wmax) c->wmax = m->post_d_ff;
    pthread_once(&g_gelu_once, gelu_build);
    if (c->host) return c;

    const int d = m->d, dff = m->d_ff;
    c->nw = 4 * (m->n_layers + m->n_post);
    c->w = calloc((size_t)c->nw, sizeof(*c->w));
    c->mm = rocket_ctx_create_ex(nthreads, ROCKET_CTX_TILING_CANONICAL);
    c->fa = rocket_fa_ctx_create(nthreads);
    if (!c->w || !c->mm || !c->fa) goto fail;
    for (int l = 0; l < m->n_layers; l++) {
        const rocket_mb_layer *L = &m->layers[l];
        rocket_weights **w = c->w + 4 * l;
        w[W_QKV]  = rocket_weights_pack(c->mm, MB_PACK_M, d,   3 * d,   L->Wqkv);
        w[W_O]    = rocket_weights_pack(c->mm, MB_PACK_M, d,   d,       L->Wo);
        w[W_UP]   = rocket_weights_pack(c->mm, MB_PACK_M, d,   2 * dff, L->Wi);
        w[W_DOWN] = rocket_weights_pack(c->mm, MB_PACK_M, dff, d,       L->Wo_mlp);
        if (!w[0] || !w[1] || !w[2] || !w[3]) goto fail;
    }
    for (int j = 0; j < m->n_post; j++) {
        const rocket_mb_post_layer *P = &m->post[j];
        rocket_weights **w = c->w + 4 * (m->n_layers + j);
        w[W_QKV]  = rocket_weights_pack(c->mm, MB_PACK_M, d, 3 * d, P->Wqkv);
        w[W_O]    = rocket_weights_pack(c->mm, MB_PACK_M, d, d, P->Wo);
        w[W_UP]   = rocket_weights_pack(c->mm, MB_PACK_M, d, m->post_d_ff, P->W1);
        w[W_DOWN] = rocket_weights_pack(c->mm, MB_PACK_M, m->post_d_ff, d, P->W2);
        if (!w[0] || !w[1] || !w[2] || !w[3]) goto fail;
    }
    return c;
fail:
    ROCKET_LOGE("rocket_modernbert_ctx_create: device setup or weight pack failed\n");
    rocket_modernbert_ctx_free(c);
    return NULL;
}

void rocket_modernbert_ctx_free(rocket_mb_ctx *c)
{
    if (!c) return;
    if (c->mm && c->w)
        for (int i = 0; i < c->nw; i++) if (c->w[i]) rocket_weights_free(c->mm, c->w[i]);
    free(c->w);
    if (c->fa) rocket_fa_ctx_free(c->fa);
    if (c->mm) rocket_ctx_free(c->mm);
    free(c->x); free(c->p); free(c->a16); free(c->c16); free(c->cc);
    free(c->Qh); free(c->Kh); free(c->Vh); free(c->Oh); free(c->mask);
    free(c->Qf); free(c->Kf); free(c->Vf);
    free(c->bQ); free(c->bK); free(c->bV); free(c->bO); free(c->bmask);
    for (int i = 0; i < 4; i++) free(c->rope[i].cs);
    free(c->hw);
    free(c);
}

static int grow(rocket_mb_ctx *c, size_t M, size_t L)
{
    const size_t d = c->m.d, W = c->wmax;
    if (M > c->capM) {
        free(c->x); free(c->p); free(c->a16); free(c->c16); free(c->cc);
        c->x = malloc(M * d * sizeof(float));
        c->p = malloc(M * W * sizeof(float));
        c->a16 = calloc(M * W, sizeof(_Float16));
        c->c16 = malloc(M * W * sizeof(_Float16));
        c->cc = calloc(M * d, sizeof(_Float16));
        c->capM = (c->x && c->p && c->a16 && c->c16 && c->cc) ? M : 0;
        if (!c->capM) return -2;
    }
    if (L > c->capL) {
        free(c->Qh); free(c->Kh); free(c->Vh); free(c->Oh); free(c->mask);
        free(c->Qf); free(c->Kf); free(c->Vf);
        c->Qh = malloc(L * d * sizeof(_Float16)); c->Kh = malloc(L * d * sizeof(_Float16));
        c->Vh = malloc(L * d * sizeof(_Float16)); c->Oh = malloc(L * d * sizeof(_Float16));
        c->Qf = malloc(L * d * sizeof(float)); c->Kf = malloc(L * d * sizeof(float));
        c->Vf = malloc(L * d * sizeof(float));
        c->mask = malloc(L * L * sizeof(_Float16));
        c->mask_L = -1;
        c->capL = (c->Qh && c->Kh && c->Vh && c->Oh && c->mask && c->Qf && c->Kf && c->Vf) ? L : 0;
        if (!c->capL) return -2;
    }
    return 0;
}

/* ============================================================================
 * SECTION: encode
 * ==========================================================================*/

/* Zero the A operand's pad rows [Mlive, M) at width K. */
static void pad_rows(_Float16 *A, int Mlive, int M, int K)
{
    if (M > Mlive) memset(A + (size_t)Mlive * K, 0, (size_t)(M - Mlive) * K * sizeof(_Float16));
}

/* Scatter the stacked live rows back to [B][T][d], zero past len[b]. */
static void emit(const rocket_mb_ctx *c, int B, int T, const int *len, const int *off,
                 const float *x, float *dst)
{
    const size_t d = c->m.d;
    for (int b = 0; b < B; b++) {
        float *o = dst + (size_t)b * T * d;
        memcpy(o, x + (size_t)off[b] * d, (size_t)len[b] * d * sizeof(float));
        memset(o + (size_t)len[b] * d, 0, (size_t)(T - len[b]) * d * sizeof(float));
    }
}

/* x[0..Mlive) += scale * (down . p) with the operand scaled into fp16's range first. */
static int down_proj(rocket_mb_ctx *c, int wi, int Mlive, int M, int K, const float *p, float amax,
                     const _Float16 *W, const _Float16 *bias)
{
    const int d = c->m.d;
    int e = 0;
    while (e < 24 && amax * ldexpf(1.f, -e) > MB_DOWN_LIMIT) e++;
    for (int attempt = 0; attempt <= MB_DOWN_RETRIES; attempt++, e += 4) {
        scale_arg sa = { p, c->a16, K, ldexpf(1.f, -e) };
        pfor(Mlive, c->ht, scale_rows, &sa);
        pad_rows(c->a16, Mlive, M, K);
        int r = gemm(c, wi, M, K, d, c->a16, c->c16, W);
        if (r) return r;
        /* check first, so a retry adds nothing twice */
        int bad = 0;
        for (size_t i = 0; i < (size_t)Mlive * d && !bad; i++) bad = !isfinite((float)c->c16[i]);
        if (bad) {
            ROCKET_LOGW("rocket_modernbert: down-projection %d non-finite at scale 2^-%d, retrying\n", wi, e);
            continue;
        }
        res_arg ra = { c->x, c->c16, bias, d, d, ldexpf(1.f, e), 0 };
        pfor(Mlive, c->ht, res_rows, &ra);
        return 0;
    }
    ROCKET_LOGE("rocket_modernbert: down-projection %d stays non-finite\n", wi);
    return -5;
}

int rocket_modernbert_encode(rocket_mb_ctx *c, int B, int T, const int *len,
                             const float *emb, const float *post_bias, float *out, float *hidden)
{
    if (!c || B < 1 || T < 1 || !len || !emb || !out) return -1;
    const rocket_modernbert_model *m = &c->m;
    const int d = m->d, dff = m->d_ff, nL = m->n_layers;
    int *off = malloc((size_t)B * sizeof(int));
    if (!off) return -2;
    int Mlive = 0, Lmax = 0;
    for (int b = 0; b < B; b++) {
        if (len[b] < 1 || len[b] > T) { free(off); return -1; }
        off[b] = Mlive; Mlive += len[b];
        if (len[b] > Lmax) Lmax = len[b];
    }
    const int M = (Mlive + 3) & ~3;
    if (grow(c, (size_t)M, (size_t)Lmax)) { free(off); return -2; }
    if (!c->host) rocket_pin_worker(0);

    const int prof = getenv("ROCKET_MB_PROF") != NULL;
    double tb[8] = {0}, t0 = 0;
#define TIC (t0 = prof ? mb_ms() : 0)
#define TOC(i) do { if (prof) tb[i] += mb_ms() - t0; } while (0)
    const size_t slab = (size_t)B * T * d;   /* one hidden tap */
    int rc = 0;

    /* embeddings -> LayerNorm, in place on the stacked rows */
    for (int b = 0; b < B; b++)
        memcpy(c->x + (size_t)off[b] * d, emb + (size_t)b * T * d, (size_t)len[b] * d * sizeof(float));
    TIC; layernorm(c, Mlive, c->x, m->emb_norm_g, NULL, m->eps, NULL, c->x); TOC(0);
    if (hidden) emit(c, B, T, len, off, c->x, hidden);

    for (int l = 0; l < nL; l++) {
        const rocket_mb_layer *Ly = &m->layers[l];
        const int wb = 4 * l;
        /* attention */
        TIC; layernorm(c, Mlive, c->x, Ly->attn_norm_g, NULL, m->eps, c->a16, NULL);
        pad_rows(c->a16, Mlive, M, d); TOC(0);
        TIC; if ((rc = gemm(c, wb + W_QKV, M, d, 3 * d, c->a16, c->c16, Ly->Wqkv))) goto done; TOC(1);
        TIC;
        for (int b = 0; b < B; b++)
            if ((rc = attend(c, c->c16 + (size_t)off[b] * 3 * d, NULL, Ly->inv_freq, Ly->sliding,
                             m->n_head, len[b], c->cc + (size_t)off[b] * d))) goto done;
        TOC(2);
        pad_rows(c->cc, Mlive, M, d);
        TIC; if ((rc = gemm(c, wb + W_O, M, d, d, c->cc, c->c16, Ly->Wo))) goto done; TOC(1);
        TIC;
        {
            res_arg ra = { c->x, c->c16, NULL, d, d, 1.f, 0 };
            pfor(Mlive, c->ht, res_rows, &ra);
            if (ra.bad) { ROCKET_LOGE("rocket_modernbert: layer %d attention output non-finite\n", l); rc = -5; goto done; }
        }
        /* GeGLU MLP */
        layernorm(c, Mlive, c->x, Ly->mlp_norm_g, NULL, m->eps, c->a16, NULL);
        pad_rows(c->a16, Mlive, M, d); TOC(0);
        TIC; if ((rc = gemm(c, wb + W_UP, M, d, 2 * dff, c->a16, c->c16, Ly->Wi))) goto done; TOC(1);
        TIC;
        geglu_arg ga = { c->c16, c->p, dff, { PTHREAD_MUTEX_INITIALIZER, 0.f } };
        pfor(Mlive, c->ht, geglu_rows, &ga);
        TOC(3);
        TIC; if ((rc = down_proj(c, wb + W_DOWN, Mlive, M, dff, c->p, ga.amax.v, Ly->Wo_mlp, NULL))) goto done; TOC(4);
        if (hidden) emit(c, B, T, len, off, c->x, hidden + (size_t)(l + 1) * slab);
    }

    /* final norm, then the post bias */
    TIC; layernorm(c, Mlive, c->x, m->final_norm_g, NULL, m->eps, NULL, c->x); TOC(0);
    if (hidden) emit(c, B, T, len, off, c->x, hidden + (size_t)(nL + 1) * slab);
    if (post_bias)
        for (int b = 0; b < B; b++)
            for (int t = 0; t < len[b]; t++) {
                float *xr = c->x + (size_t)(off[b] + t) * d;
                const float *pb = post_bias + (size_t)b * d;
                for (int j = 0; j < d; j++) xr[j] += pb[j];
            }

    for (int j = 0; j < m->n_post; j++) {
        const rocket_mb_post_layer *P = &m->post[j];
        const int wb = 4 * (nL + j), pff = m->post_d_ff;
        TIC; layernorm(c, Mlive, c->x, P->ln1_g, P->ln1_b, m->post_eps, c->a16, NULL);
        pad_rows(c->a16, Mlive, M, d); TOC(0);
        TIC; if ((rc = gemm(c, wb + W_QKV, M, d, 3 * d, c->a16, c->c16, P->Wqkv))) goto done; TOC(1);
        TIC;
        for (int b = 0; b < B; b++)
            if ((rc = attend(c, c->c16 + (size_t)off[b] * 3 * d, P->bqkv, NULL, 0, m->post_n_head,
                             len[b], c->cc + (size_t)off[b] * d))) goto done;
        TOC(2);
        pad_rows(c->cc, Mlive, M, d);
        TIC; if ((rc = gemm(c, wb + W_O, M, d, d, c->cc, c->c16, P->Wo))) goto done; TOC(1);
        TIC;
        {
            res_arg ra = { c->x, c->c16, P->bo, d, d, 1.f, 0 };
            pfor(Mlive, c->ht, res_rows, &ra);
            if (ra.bad) { ROCKET_LOGE("rocket_modernbert: post layer %d attention output non-finite\n", j); rc = -5; goto done; }
        }
        layernorm(c, Mlive, c->x, P->ln2_g, P->ln2_b, m->post_eps, c->a16, NULL);
        pad_rows(c->a16, Mlive, M, d); TOC(0);
        TIC; if ((rc = gemm(c, wb + W_UP, M, d, pff, c->a16, c->c16, P->W1))) goto done; TOC(1);
        TIC;
        act_arg aa = { c->c16, P->b1, c->p, pff, m->post_act, { PTHREAD_MUTEX_INITIALIZER, 0.f } };
        pfor(Mlive, c->ht, act_rows, &aa);
        TOC(3);
        TIC; if ((rc = down_proj(c, wb + W_DOWN, Mlive, M, pff, c->p, aa.amax.v, P->W2, P->b2))) goto done; TOC(4);
        if (hidden) emit(c, B, T, len, off, c->x, hidden + (size_t)(nL + 2 + j) * slab);
    }

    emit(c, B, T, len, off, c->x, out);
    rc = 0;
    if (prof) {
        static const char *nm[5] = { "layernorm", "gemm", "attention", "act", "down+res" };
        double tot = 0; for (int i = 0; i < 5; i++) tot += tb[i];
        ROCKET_LOGE("[modernbert prof] B=%d Mlive=%d Lmax=%d total %.1f ms\n", B, Mlive, Lmax, tot);
        for (int i = 0; i < 5; i++)
            ROCKET_LOGE("    %-10s %8.1f ms (%4.1f%%)\n", nm[i], tb[i], 100.0 * tb[i] / (tot + 1e-9));
    }
#undef TIC
#undef TOC
done:
    free(off);
    return rc;
}
