// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_siglip_encoder.c — the SigLIP-B/16 vision encoder end to end on the NPU,
 * glue over the validated primitives. See rocket_siglip.h for the graph + blob format.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <math.h>
#include <time.h>
#include <pthread.h>

#include "rocket_siglip.h"
#include "rocket_matmul.h"      /* rocket_matmul_fp16 (C=A·B^T) + prepacked/stream */
#include "rocket_encoder.h"     /* rocket_encoder_block_fp16    */
#include "rocket_attn.h"        /* rocket_flash_attn_fp16_ctx (multicore MHA)    */
#include "rocket_norm.h"        /* rocket_layernorm_fp16        */
#include "rocket_softmax.h"     /* rocket_softmax_fp16          */
#include "rocket_npu.h"         /* rocket_open / rocket_close (aux fd) */
#include "rocket_log.h"     // centralized log channel

#define SIGLIP_MAGIC   0x53474C50
#define SIGLIP_VERSION 1

/* ============================================================================
 * SECTION — Model blob load / free
 * ==========================================================================*/

/* per-layer fp16 element count (must match siglip_extract.py's order/sizes) */
static size_t layer_stride_elems(int d, int d_ff)
{
    return (size_t)2 * d                               /* ln1_g, ln1_b           */
         + 4 * ((size_t)d * d + d)                     /* Wq,bq Wk,bk Wv,bv Wo,bo */
         + (size_t)2 * d                               /* ln2_g, ln2_b           */
         + ((size_t)d_ff * d + d_ff)                   /* Wf1, bf1               */
         + ((size_t)d * d_ff + d);                     /* Wf2, bf2               */
}

int rocket_siglip_load(const char *path, rocket_siglip_model *m)
{
    if (!path || !m) return -1;
    memset(m, 0, sizeof(*m));
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 64) { close(fd); return -2; }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return -2;

    const int32_t *h = (const int32_t *)map;
    if (h[0] != SIGLIP_MAGIC || h[1] != SIGLIP_VERSION) { munmap(map, st.st_size); return -3; }
    m->map = map; m->map_size = (size_t)st.st_size;
    m->d = h[2]; m->n_layers = h[3]; m->n_head = h[4]; m->d_ff = h[5];
    m->L = h[6]; m->patch_dim = h[7];
    m->ic = h[8]; m->kh = h[9]; m->kw = h[10]; m->stride = h[11]; m->image_size = h[12];
    memcpy(&m->eps, &h[13], sizeof(float));

    if (m->d <= 0 || m->n_layers <= 0 || m->n_head <= 0 || m->d_ff <= 0 ||
        m->L <= 0 || m->patch_dim <= 0 || (m->d % m->n_head)) { munmap(map, st.st_size); return -4; }

    const _Float16 *w = (const _Float16 *)((const char *)map + 64);
    size_t off = 0;
    m->patch_W = w + off; off += (size_t)m->d * m->patch_dim;
    m->patch_b = w + off; off += m->d;
    m->pos     = w + off; off += (size_t)m->L * m->d;
    m->layers  = w + off;
    m->layer_stride = layer_stride_elems(m->d, m->d_ff);
    off += (size_t)m->n_layers * m->layer_stride;
    m->post_g  = w + off; off += m->d;
    m->post_b  = w + off; off += m->d;

    /* verify the blob is exactly the size the dims imply (header + fp16 region) */
    size_t want = 64 + off * sizeof(_Float16);
    if (want != m->map_size) { munmap(map, st.st_size); memset(m, 0, sizeof(*m)); return -4; }
    return 0;
}

void rocket_siglip_free(rocket_siglip_model *m)
{
    if (m && m->map) { munmap(m->map, m->map_size); m->map = NULL; }
}

/* ============================================================================
 * SECTION — Simple encode path (per-call, single-fd)
 * ==========================================================================*/

/* host fp32-accumulate matmul C[M][N] = A[M][K]·B[N][K]^T (the fd<0 patch-embed path) */
static void host_matmul(int M, int K, int N, const _Float16 *A, const _Float16 *B, _Float16 *C)
{
    for (int i = 0; i < M; i++) {
        const _Float16 *ar = A + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const _Float16 *br = B + (size_t)j * K;
            double a = 0;
            for (int k = 0; k < K; k++) a += (double)ar[k] * (double)br[k];
            C[(size_t)i * N + j] = (_Float16)a;
        }
    }
}

int rocket_siglip_encode(int fd, const rocket_siglip_model *m,
                         const _Float16 *pixels_chw, _Float16 *out, _Float16 *hidden_opt)
{
    if (!m || !m->map || !pixels_chw || !out) return -1;
    const int d = m->d, L = m->L, pdim = m->patch_dim, dff = m->d_ff, nh = m->n_head;
    const int side = m->image_size / m->stride, H = m->image_size, W = m->image_size;
    const int ic = m->ic, kh = m->kh, kw = m->kw, stride = m->stride;
    const size_t Ld = (size_t)L * d;
    int rc = -2;

    _Float16 *patches = malloc((size_t)L * pdim * sizeof(_Float16));
    _Float16 *xa = malloc(Ld * sizeof(_Float16));
    _Float16 *xb = malloc(Ld * sizeof(_Float16));
    if (!patches || !xa || !xb) goto done;

    /* --- im2col: patches[p][ci*kh*kw + r*kw + c] = pixels[ci][ph*stride+r][pw*stride+c] ---
     * matches patch_W's [oc][ic][kh][kw] row-major flatten (patchify: stride==kernel, pad 0). */
    for (int ph = 0; ph < side; ph++)
        for (int pw = 0; pw < side; pw++) {
            int p = ph * side + pw;
            _Float16 *row = patches + (size_t)p * pdim;
            for (int ci = 0; ci < ic; ci++) {
                const _Float16 *chan = pixels_chw + (size_t)ci * H * W;
                for (int r = 0; r < kh; r++) {
                    const _Float16 *src = chan + (size_t)(ph * stride + r) * W + pw * stride;
                    _Float16 *dst = row + (size_t)ci * kh * kw + (size_t)r * kw;
                    for (int c = 0; c < kw; c++) dst[c] = src[c];
                }
            }
        }

    /* --- patch projection x = patches·patch_W^T  (NPU GEMM; host fp32 for fd<0) --- */
    if (fd >= 0) { if ((rc = rocket_matmul_fp16(fd, L, pdim, d, patches, m->patch_W, xa)) != 0) goto done; }
    else         host_matmul(L, pdim, d, patches, m->patch_W, xa);

    /* --- + patch bias (broadcast) + position embedding (host glue) --- */
    for (int p = 0; p < L; p++) {
        _Float16 *xr = xa + (size_t)p * d;
        const _Float16 *pr = m->pos + (size_t)p * d;
        for (int j = 0; j < d; j++)
            xr[j] = (_Float16)((float)xr[j] + (float)m->patch_b[j] + (float)pr[j]);
    }
    if (hidden_opt) memcpy(hidden_opt, xa, Ld * sizeof(_Float16));

    /* --- 12 × pre-norm encoder block (ping-pong xa -> xb) --- */
    for (int l = 0; l < m->n_layers; l++) {
        const _Float16 *lb = m->layers + (size_t)l * m->layer_stride;
        const _Float16 *ln1_g = lb,          *ln1_b = lb + d;
        const _Float16 *Wq = ln1_b + d,      *bq = Wq + (size_t)d * d;
        const _Float16 *Wk = bq + d,         *bk = Wk + (size_t)d * d;
        const _Float16 *Wv = bk + d,         *bv = Wv + (size_t)d * d;
        const _Float16 *Wo = bv + d,         *bo = Wo + (size_t)d * d;
        const _Float16 *ln2_g = bo + d,      *ln2_b = ln2_g + d;
        const _Float16 *Wf1 = ln2_b + d,     *bf1 = Wf1 + (size_t)dff * d;
        const _Float16 *Wf2 = bf1 + dff,     *bf2 = Wf2 + (size_t)d * dff;
        if ((rc = rocket_encoder_block_fp16(fd, L, d, nh, dff, xa,
                     ln1_g, ln1_b, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
                     ln2_g, ln2_b, Wf1, bf1, Wf2, bf2, m->eps, xb)) != 0) goto done;
        _Float16 *t = xa; xa = xb; xb = t;
        if (hidden_opt) memcpy(hidden_opt + (size_t)(l + 1) * Ld, xa, Ld * sizeof(_Float16));
    }

    /* --- post LayerNorm over d -> out --- */
    rc = rocket_layernorm_fp16(fd, L, d, xa, m->post_g, m->post_b, m->eps, out);

done:
    free(patches); free(xa); free(xb);
    return rc;
}

/* ============================================================================
 * SECTION — Resident (prepacked, multicore) path
 * ==========================================================================*/

struct rocket_siglip_ctx {
    const rocket_siglip_model *m;
    rocket_ctx     *mm;       /* prepacked static weights, multicore */
    rocket_stream  *strm;     /* computed attention matmuls (scores, P·V) */
    int             aux_fd;   /* softmax */
    rocket_weights *w_patch;
    rocket_weights **wq, **wk, **wv, **wo, **wf1, **wf2;   /* [n_layers] */
    int ht;   /* host worker threads for softmax/GELU (memory-bound, embarrassingly parallel) */
    /* multicore MHA: the heads are independent, so fan them across worker fds (3 cores)
     * via the validated flash-attention primitive instead of the single-worker per-head
     * stream. Unmasked (mask=NULL), n_kv_heads==n_head (plain MHA). Built unless
     * ROCKET_SIGLIP_FA=0; NULL => the single-stream per-head path. */
    rocket_fa_ctx  *fa;
};

/* per-layer weight pointers from the blob (same offsets as the encode loop above) */
typedef struct {
    const _Float16 *ln1_g, *ln1_b, *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo,
                   *ln2_g, *ln2_b, *Wf1, *bf1, *Wf2, *bf2;
} layer_ptrs;

static layer_ptrs unpack_layer(const rocket_siglip_model *m, int l)
{
    const int d = m->d, dff = m->d_ff;
    const _Float16 *lb = m->layers + (size_t)l * m->layer_stride;
    layer_ptrs p;
    p.ln1_g = lb;            p.ln1_b = lb + d;
    p.Wq = p.ln1_b + d;      p.bq = p.Wq + (size_t)d * d;
    p.Wk = p.bq + d;         p.bk = p.Wk + (size_t)d * d;
    p.Wv = p.bk + d;         p.bv = p.Wv + (size_t)d * d;
    p.Wo = p.bv + d;         p.bo = p.Wo + (size_t)d * d;
    p.ln2_g = p.bo + d;      p.ln2_b = p.ln2_g + d;
    p.Wf1 = p.ln2_b + d;     p.bf1 = p.Wf1 + (size_t)dff * d;
    p.Wf2 = p.bf1 + dff;     p.bf2 = p.Wf2 + (size_t)d * dff;
    return p;
}

/* ============================================================================
 * SECTION — Resident path host elementwise helpers (layernorm/GELU/softmax)
 * ==========================================================================*/

/* host elementwise (memory-bound, faster than an NPU round-trip once de-tiled) */
static void h_layernorm(int M, int d, const _Float16 *x, const _Float16 *g,
                        const _Float16 *b, float eps, _Float16 *out)
{
    for (int i = 0; i < M; i++) {
        const _Float16 *xr = x + (size_t)i * d;
        double mean = 0; for (int j = 0; j < d; j++) mean += (double)xr[j]; mean /= d;
        double var = 0; for (int j = 0; j < d; j++) { double t = (double)xr[j] - mean; var += t * t; }
        double inv = 1.0 / sqrt(var / d + eps);
        _Float16 *o = out + (size_t)i * d;
        for (int j = 0; j < d; j++)
            o[j] = (_Float16)(((double)xr[j] - mean) * inv * (double)g[j] + (double)b[j]);
    }
}
/* host parallel-for over [0,n): split into <=nt contiguous chunks (the host elementwise
 * ops — softmax, GELU — are memory-bound and embarrassingly parallel over rows/elements). */
typedef struct { void (*fn)(void *, int, int); void *arg; int lo, hi; } prange_t;
static void *prange_thunk(void *p) { prange_t *r = p; r->fn(r->arg, r->lo, r->hi); return NULL; }
static void parallel_for(int n, int nt, void (*fn)(void *, int, int), void *arg)
{
    if (nt < 2 || n < 2 * nt) { fn(arg, 0, n); return; }
    if (nt > 8) nt = 8;
    pthread_t th[8]; prange_t r[8];
    int chunk = (n + nt - 1) / nt, spawned = 0;
    for (int i = 0; i < nt; i++) {
        int lo = i * chunk, hi = lo + chunk;
        if (lo >= n) break;
        if (hi > n) hi = n;
        r[spawned].fn = fn; r[spawned].arg = arg; r[spawned].lo = lo; r[spawned].hi = hi;
        if (pthread_create(&th[spawned], NULL, prange_thunk, &r[spawned]) == 0) spawned++;
        else fn(arg, lo, hi);   /* spawn failed: run this chunk inline */
    }
    for (int i = 0; i < spawned; i++) pthread_join(th[i], NULL);
}

/* SigLIP's gelu_pytorch_tanh, exactly (matches the training activation) */
static inline _Float16 gelu1(_Float16 xv)
{
    const float c = 0.7978845608028654f;   /* sqrt(2/pi) */
    float v = (float)xv;
    float t = tanhf(c * (v + 0.044715f * v * v * v));
    return (_Float16)(0.5f * v * (1.f + t));
}

/* fp16 GELU is a function of a 16-bit value, so all 65536 outputs fit one table. The LUT
 * is BIT-EXACT to the scalar tanhf path (it tabulates gelu1 over every fp16 bit pattern,
 * incl. inf/nan) and turns the hot per-element tanhf into a load — the host GELU is the
 * resident encode's #2 cost once attention is multicore. Built once (single-threaded entry
 * in h_gelu_tanh, before the parallel fan-out). ROCKET_SIGLIP_GELU_SCALAR=1 forces the
 * scalar path (A/B). */
static _Float16 g_gelu_lut[65536];
static int      g_gelu_lut_ready;
static void gelu_lut_build(void)
{
    for (int u = 0; u < 65536; u++) {
        uint16_t bits = (uint16_t)u; _Float16 h; memcpy(&h, &bits, sizeof h);
        g_gelu_lut[u] = gelu1(h);
    }
    g_gelu_lut_ready = 1;
}
static void gelu_range(void *a, int lo, int hi)
{
    _Float16 *x = a;
    for (int i = lo; i < hi; i++) {
        uint16_t bits; memcpy(&bits, &x[i], sizeof bits);
        x[i] = g_gelu_lut[bits];
    }
}
static void gelu_range_scalar(void *a, int lo, int hi)
{
    _Float16 *x = a;
    for (int i = lo; i < hi; i++) x[i] = gelu1(x[i]);
}
static void h_gelu_tanh(size_t n, _Float16 *x, int nt)
{
    if (getenv("ROCKET_SIGLIP_GELU_SCALAR")) { parallel_for((int)n, nt, gelu_range_scalar, x); return; }
    if (!g_gelu_lut_ready) gelu_lut_build();   /* single-threaded, before the fan-out */
    parallel_for((int)n, nt, gelu_range, x);
}

static void h_add_bias(int M, int N, _Float16 *C, const _Float16 *b)
{
    if (!b) return;
    for (int i = 0; i < M; i++) { _Float16 *r = C + (size_t)i * N; for (int j = 0; j < N; j++) r[j] = (_Float16)((float)r[j] + (float)b[j]); }
}
/* row-wise softmax in place (host). The scores already live in host memory after the
 * stream matmul, so this beats packing 12M elements to the NPU softmax and back. */
typedef struct { _Float16 *x; int n; } sm_arg;
static void sm_rows(void *a, int lo, int hi)
{
    sm_arg *s = a; int n = s->n;
    for (int r = lo; r < hi; r++) {
        _Float16 *row = s->x + (size_t)r * n;
        float mx = -INFINITY;
        for (int j = 0; j < n; j++) { float v = (float)row[j]; if (v > mx) mx = v; }
        float sum = 0;
        for (int j = 0; j < n; j++) { float e = expf((float)row[j] - mx); row[j] = (_Float16)e; sum += e; }
        float inv = 1.f / sum;
        for (int j = 0; j < n; j++) row[j] = (_Float16)((float)row[j] * inv);
    }
}
static void h_softmax_rows(int rows, int n, _Float16 *x, int nt)
{ sm_arg a = { x, n }; parallel_for(rows, nt, sm_rows, &a); }
static void h_residual(size_t n, _Float16 *acc, const _Float16 *add)
{ for (size_t i = 0; i < n; i++) acc[i] = (_Float16)((float)acc[i] + (float)add[i]); }

static double prof_ms(void)
{ struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6; }

/* computed matmul (both operands are activations): resident stream, fall back per-call */
static int cmatmul(rocket_siglip_ctx *c, int M, int K, int N,
                   const _Float16 *A, const _Float16 *B, _Float16 *C)
{
    if (rocket_matmul_fp16_stream(c->strm, M, K, N, A, B, C) == 0) return 0;
    return rocket_matmul_fp16(c->aux_fd, M, K, N, A, B, C);   /* fallback */
}

rocket_siglip_ctx *rocket_siglip_ctx_create(const rocket_siglip_model *m, int nthreads)
{
    if (!m || !m->map) return NULL;
    rocket_siglip_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->m = m;
    c->ht = (nthreads <= 1) ? 1 : 4;   /* host softmax/GELU across the 4 A76 big cores */
    const int d = m->d, L = m->L, pdim = m->patch_dim, dff = m->d_ff, nL = m->n_layers;

    c->mm = rocket_ctx_create(nthreads);
    c->strm = rocket_stream_create(1);   /* attention matmuls are small (K=64/N=64) — N-split hurts */
    c->aux_fd = rocket_open();
    c->wq = calloc(nL, sizeof(*c->wq));  c->wk = calloc(nL, sizeof(*c->wk));
    c->wv = calloc(nL, sizeof(*c->wv));  c->wo = calloc(nL, sizeof(*c->wo));
    c->wf1 = calloc(nL, sizeof(*c->wf1)); c->wf2 = calloc(nL, sizeof(*c->wf2));
    if (!c->mm || !c->strm || c->aux_fd < 0 ||
        !c->wq || !c->wk || !c->wv || !c->wo || !c->wf1 || !c->wf2) goto fail;

    c->w_patch = rocket_weights_pack(c->mm, L, pdim, d, m->patch_W);
    if (!c->w_patch) goto fail;
    for (int l = 0; l < nL; l++) {
        layer_ptrs p = unpack_layer(m, l);
        c->wq[l]  = rocket_weights_pack(c->mm, L, d,   d,   p.Wq);
        c->wk[l]  = rocket_weights_pack(c->mm, L, d,   d,   p.Wk);
        c->wv[l]  = rocket_weights_pack(c->mm, L, d,   d,   p.Wv);
        c->wo[l]  = rocket_weights_pack(c->mm, L, d,   d,   p.Wo);
        c->wf1[l] = rocket_weights_pack(c->mm, L, d,   dff, p.Wf1);
        c->wf2[l] = rocket_weights_pack(c->mm, L, dff, d,   p.Wf2);
        if (!c->wq[l] || !c->wk[l] || !c->wv[l] || !c->wo[l] || !c->wf1[l] || !c->wf2[l]) goto fail;
    }

    /* multicore MHA via the flash-attention fan-out (opt out with ROCKET_SIGLIP_FA=0).
     * The per-head QK/softmax/AV is the resident path's largest slice and was running on a
     * single worker; fanning the heads across `nthreads` worker fds parallelizes it over the
     * NPU cores. NULL (e.g. nthreads<=1 or a worker fd fails) keeps the single-stream path. */
    {
        const char *ae = getenv("ROCKET_SIGLIP_FA");
        if ((!ae || atoi(ae) != 0) && nthreads > 1)
            c->fa = rocket_fa_ctx_create(nthreads);   /* NULL => single-stream fallback */
    }
    return c;
fail:
    rocket_siglip_ctx_free(c);
    return NULL;
}

void rocket_siglip_ctx_free(rocket_siglip_ctx *c)
{
    if (!c) return;
    if (c->mm) {
        if (c->w_patch) rocket_weights_free(c->mm, c->w_patch);
        for (int l = 0; c->wq && l < c->m->n_layers; l++) {
            if (c->wq && c->wq[l])  rocket_weights_free(c->mm, c->wq[l]);
            if (c->wk && c->wk[l])  rocket_weights_free(c->mm, c->wk[l]);
            if (c->wv && c->wv[l])  rocket_weights_free(c->mm, c->wv[l]);
            if (c->wo && c->wo[l])  rocket_weights_free(c->mm, c->wo[l]);
            if (c->wf1 && c->wf1[l]) rocket_weights_free(c->mm, c->wf1[l]);
            if (c->wf2 && c->wf2[l]) rocket_weights_free(c->mm, c->wf2[l]);
        }
    }
    if (c->fa) rocket_fa_ctx_free(c->fa);
    free(c->wq); free(c->wk); free(c->wv); free(c->wo); free(c->wf1); free(c->wf2);
    if (c->strm) rocket_stream_free(c->strm);
    if (c->mm) rocket_ctx_free(c->mm);
    if (c->aux_fd >= 0) rocket_close(c->aux_fd);
    free(c);
}

int rocket_siglip_encode_ctx(rocket_siglip_ctx *c, const _Float16 *pixels_chw,
                             _Float16 *out, _Float16 *hidden_opt)
{
    if (!c || !pixels_chw || !out) return -1;
    const rocket_siglip_model *m = c->m;
    const int d = m->d, L = m->L, pdim = m->patch_dim, dff = m->d_ff, nh = m->n_head, dh = d / nh;
    const int side = m->image_size / m->stride, H = m->image_size, W = m->image_size;
    const int ic = m->ic, kh = m->kh, kw = m->kw, stride = m->stride;
    const float scale = 1.f / sqrtf((float)dh);
    const size_t Ld = (size_t)L * d;
    int rc = -2;

    const int use_fa = (c->fa != NULL);
    _Float16 *patches = malloc((size_t)L * pdim * sizeof(_Float16));
    _Float16 *x  = malloc(Ld * sizeof(_Float16));
    _Float16 *ln = malloc(Ld * sizeof(_Float16));
    _Float16 *q  = malloc(Ld * sizeof(_Float16));
    _Float16 *k  = malloc(Ld * sizeof(_Float16));
    _Float16 *v  = malloc(Ld * sizeof(_Float16));
    _Float16 *cc = malloc(Ld * sizeof(_Float16));      /* concat per-head context */
    _Float16 *y  = malloc((size_t)L * dff * sizeof(_Float16));  /* o-proj / fc1 / fc2 out */
    /* FA path: head-major Q/K/V/OUT for the flash-attention fan-out (each Ld elems).
     * single-stream path: per-head gather scratch qh/khh/vhT + all-heads scores sc. */
    _Float16 *Qh = NULL, *Kh = NULL, *Vh = NULL, *Oh = NULL;
    _Float16 *qh = NULL, *khh = NULL, *vhT = NULL, *sc = NULL;
    int alloc_ok;
    if (use_fa) {
        Qh = malloc(Ld * sizeof(_Float16)); Kh = malloc(Ld * sizeof(_Float16));
        Vh = malloc(Ld * sizeof(_Float16)); Oh = malloc(Ld * sizeof(_Float16));
        alloc_ok = Qh && Kh && Vh && Oh;
    } else {
        qh  = malloc((size_t)L * dh * sizeof(_Float16));
        khh = malloc((size_t)L * dh * sizeof(_Float16));
        vhT = malloc((size_t)dh * L * sizeof(_Float16));
        /* all heads' scores in one buffer; softmaxed in place on the host (one pass over
         * [nh*L, L] rows), so no NPU softmax round-trip and PV reads sc directly. */
        sc  = malloc((size_t)nh * L * L * sizeof(_Float16));
        alloc_ok = qh && khh && vhT && sc;
    }
    if (!patches || !x || !ln || !q || !k || !v || !cc || !y || !alloc_ok) goto done;

    int prof = getenv("ROCKET_SIGLIP_PROF") != NULL;
    double tb[16] = {0}, t0 = 0;
#define TIC (t0 = prof_ms())
#define TOC(i) do { if (prof) tb[i] += prof_ms() - t0; } while (0)

    /* im2col (same ordering as the simple path) */
    TIC;
    for (int ph = 0; ph < side; ph++)
        for (int pw = 0; pw < side; pw++) {
            int p = ph * side + pw;
            _Float16 *row = patches + (size_t)p * pdim;
            for (int ci = 0; ci < ic; ci++) {
                const _Float16 *chan = pixels_chw + (size_t)ci * H * W;
                for (int r = 0; r < kh; r++) {
                    const _Float16 *srcp = chan + (size_t)(ph * stride + r) * W + pw * stride;
                    _Float16 *dst = row + (size_t)ci * kh * kw + (size_t)r * kw;
                    for (int cx = 0; cx < kw; cx++) dst[cx] = srcp[cx];
                }
            }
        }
    TOC(0);

    /* patch projection (prepacked) + bias + pos (host) */
    TIC;
    if ((rc = rocket_matmul_fp16_prepacked(c->mm, L, pdim, d, patches, x, c->w_patch)) != 0) goto done;
    for (int p = 0; p < L; p++) {
        _Float16 *xr = x + (size_t)p * d; const _Float16 *pr = m->pos + (size_t)p * d;
        for (int j = 0; j < d; j++) xr[j] = (_Float16)((float)xr[j] + (float)m->patch_b[j] + (float)pr[j]);
    }
    TOC(1);
    if (hidden_opt) memcpy(hidden_opt, x, Ld * sizeof(_Float16));

    for (int l = 0; l < m->n_layers; l++) {
        layer_ptrs p = unpack_layer(m, l);

        /* --- attention sublayer: x += Wo·MHA(LN1(x)) --- */
        TIC; h_layernorm(L, d, x, p.ln1_g, p.ln1_b, m->eps, ln); TOC(2);
        TIC;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, L, d, d, ln, q, c->wq[l])) != 0) goto done;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, L, d, d, ln, k, c->wk[l])) != 0) goto done;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, L, d, d, ln, v, c->wv[l])) != 0) goto done;
        h_add_bias(L, d, q, p.bq); h_add_bias(L, d, k, p.bk); h_add_bias(L, d, v, p.bv);
        if (!use_fa)
            for (size_t i = 0; i < Ld; i++) q[i] = (_Float16)((float)q[i] * scale);  /* fold 1/sqrt(dh) */
        TOC(3);

        if (use_fa) {
            /* re-lay q/k/v [L,d] (heads interleaved) into head-major operands and run all
             * heads across the worker fds: Q/K=[nh][L][dh], V=[nh][dh][L] (the AV B-operand),
             * OUT=[nh][L][dh]. scale is applied inside flash-attn (q is NOT pre-scaled). */
            TIC;
            for (int h = 0; h < nh; h++) {
                int off = h * dh;
                _Float16 *Qd = Qh + (size_t)h * L * dh, *Kd = Kh + (size_t)h * L * dh;
                _Float16 *Vd = Vh + (size_t)h * dh * L;
                for (int t = 0; t < L; t++) {
                    memcpy(Qd + (size_t)t * dh, q + (size_t)t * d + off, dh * sizeof(_Float16));
                    memcpy(Kd + (size_t)t * dh, k + (size_t)t * d + off, dh * sizeof(_Float16));
                    for (int cx = 0; cx < dh; cx++) Vd[(size_t)cx * L + t] = v[(size_t)t * d + off + cx];
                }
            }
            TOC(4);
            TIC;
            if ((rc = rocket_flash_attn_fp16_ctx(c->fa, L, L, dh, dh, nh, nh, scale, 0.f,
                                                 Qh, Kh, Vh, NULL, Oh)) != 0) goto done;
            TOC(5);   /* scores + softmax + PV, fanned across cores */
            TIC;
            for (int h = 0; h < nh; h++) {
                int off = h * dh;
                const _Float16 *Od = Oh + (size_t)h * L * dh;
                for (int t = 0; t < L; t++)
                    memcpy(cc + (size_t)t * d + off, Od + (size_t)t * dh, dh * sizeof(_Float16));
            }
            TOC(8);
        } else {
        /* phase 1: every head's scores q_h·k_h^T into sc[h] (stacked by row) */
        for (int h = 0; h < nh; h++) {
            int off = h * dh;
            TIC;
            for (int t = 0; t < L; t++) {
                memcpy(qh  + (size_t)t * dh, q + (size_t)t * d + off, dh * sizeof(_Float16));
                memcpy(khh + (size_t)t * dh, k + (size_t)t * d + off, dh * sizeof(_Float16));
            }
            TOC(4);
            TIC; if ((rc = cmatmul(c, L, dh, L, qh, khh, sc + (size_t)h * L * L)) != 0) goto done; TOC(5);
        }
        /* phase 2: row-wise softmax on the host, in place (no NPU round-trip), threaded */
        TIC; h_softmax_rows(nh * L, L, sc, c->ht); TOC(6);
        /* phase 3: every head's ctx_h = P_h·v_h, scattered into cc columns [off,off+dh) */
        for (int h = 0; h < nh; h++) {
            int off = h * dh;
            TIC;
            for (int t = 0; t < L; t++)
                for (int cx = 0; cx < dh; cx++) vhT[(size_t)cx * L + t] = v[(size_t)t * d + off + cx];
            TOC(4);
            TIC; if ((rc = cmatmul(c, L, L, dh, sc + (size_t)h * L * L, vhT, qh)) != 0) goto done; TOC(7);
            TIC;
            for (int t = 0; t < L; t++) memcpy(cc + (size_t)t * d + off, qh + (size_t)t * dh, dh * sizeof(_Float16));
            TOC(8);
        }
        }
        TIC;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, L, d, d, cc, y, c->wo[l])) != 0) goto done;
        h_add_bias(L, d, y, p.bo);
        TOC(9);
        TIC; h_residual(Ld, x, y); TOC(13);

        /* --- FFN sublayer: x += fc2(GELU(fc1(LN2(x)))) --- */
        TIC; h_layernorm(L, d, x, p.ln2_g, p.ln2_b, m->eps, ln); TOC(2);
        TIC;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, L, d, dff, ln, y, c->wf1[l])) != 0) goto done;
        h_add_bias(L, dff, y, p.bf1);
        TOC(10);
        TIC; h_gelu_tanh((size_t)L * dff, y, c->ht); TOC(11);
        TIC;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, L, dff, d, y, cc, c->wf2[l])) != 0) goto done;
        h_add_bias(L, d, cc, p.bf2);
        TOC(12);
        TIC; h_residual(Ld, x, cc); TOC(13);

        if (hidden_opt) memcpy(hidden_opt + (size_t)(l + 1) * Ld, x, Ld * sizeof(_Float16));
    }

    TIC; h_layernorm(L, d, x, m->post_g, m->post_b, m->eps, out); TOC(2);
    rc = 0;
    if (prof) {
        const char *nm[14] = {"im2col","patch+pos","layernorm","qkv-proj","head-xpose",
            "scores-mm","softmax","pv-mm","ctx-scatter","o-proj","fc1","gelu","fc2","residual"};
        double tot = 0; for (int i = 0; i < 14; i++) tot += tb[i];
        ROCKET_LOGE("[siglip prof] total %.1f ms:\n", tot);
        for (int i = 0; i < 14; i++)
            ROCKET_LOGE("    %-12s %8.1f ms  (%4.1f%%)\n", nm[i], tb[i], 100.0 * tb[i] / (tot + 1e-9));
    }
#undef TIC
#undef TOC
done:
    free(patches); free(x); free(ln); free(q); free(k); free(v); free(cc); free(y);
    free(qh); free(khh); free(vhT); free(sc);
    free(Qh); free(Kh); free(Vh); free(Oh);
    return rc;
}
