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
#include <sys/resource.h>
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
#include "rocket_affinity.h"    /* rocket_pin_worker_based (keep host workers off the A55s) */
#include "rocket_cube.h"        /* MM_NO_OVERLAP */
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
    /* This per-call path handles only the SigLIP configuration. A CLIP-shaped model (prefix/cls
     * token, pre-encoder LayerNorm, quick-GELU, or a bias-less patch Conv / no sequence post-LN)
     * must go through the resident ctx path, which open-codes the block and supports those. */
    if (m->n_prefix != 0 || m->prefix || m->pre_g || m->act_kind != 0 || !m->patch_b ||
        !m->post_g || m->n_taps != 0)
        return -3;
    const int d = m->d, L = m->L, pdim = m->patch_dim, dff = m->d_ff, nh = m->n_head;
    /* matmul K for the patch projection: patch_dim rounded up to %32 (see rocket_siglip.h).
     * patch_W is laid out at this stride with a zero tail; the im2col below zero-fills to match. */
    const int pdp = m->patch_dim_pad ? m->patch_dim_pad : pdim;
    const int side = m->image_size / m->stride, H = m->image_size, W = m->image_size;
    const int ic = m->ic, kh = m->kh, kw = m->kw, stride = m->stride;
    const size_t Ld = (size_t)L * d;
    int rc = -2;

    _Float16 *patches = calloc((size_t)L * pdp, sizeof(_Float16));   /* calloc: [pdim,pdp) tail 0 */
    _Float16 *xa = malloc(Ld * sizeof(_Float16));
    _Float16 *xb = malloc(Ld * sizeof(_Float16));
    if (!patches || !xa || !xb) goto done;

    /* --- im2col: patches[p][ci*kh*kw + r*kw + c] = pixels[ci][ph*stride+r][pw*stride+c] ---
     * matches patch_W's [oc][ic][kh][kw] row-major flatten (patchify: stride==kernel, pad 0). */
    for (int ph = 0; ph < side; ph++)
        for (int pw = 0; pw < side; pw++) {
            int p = ph * side + pw;
            _Float16 *row = patches + (size_t)p * pdp;
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
    if (fd >= 0) { if ((rc = rocket_matmul_fp16(fd, L, pdp, d, patches, m->patch_W, xa)) != 0) goto done; }
    else         host_matmul(L, pdp, d, patches, m->patch_W, xa);

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
    /* Per-image host scratch, allocated zeroed once at create and reused by every encode. The
     * zero regions it relies on are never written afterwards: the im2col tail columns
     * [patch_dim, pdp) and the pad rows past L / ntok (host writes stop at ntok, and an NPU
     * matmul writes row i from input row i, so a zero pad row stays zero). Allocating these per
     * call cost a page fault per 4 KiB whenever another allocation in the process left glibc
     * returning them to the kernel between images. */
    _Float16 *patches, *xp, *x, *ln, *q, *k, *v, *cc, *y;
    _Float16 *Qh, *Kh, *Vh, *Oh;        /* FA path: head-major operands */
    _Float16 *qh, *khh, *vhT, *sc;      /* single-stream path: per-head gather + all scores */
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

/* The row sums run as LN_LANES independent float partial sums combined in a fixed order: GCC
 * vectorizes that without reassociating anything, where one float (or double) accumulator stays
 * scalar. The result does not depend on the thread or the run. */
#define LN_LANES 8
static float ln_sum(const _Float16 *x, int d)
{
    float acc[LN_LANES] = {0}, s = 0;
    int j = 0;
    for (; j + LN_LANES <= d; j += LN_LANES)
        for (int k = 0; k < LN_LANES; k++) acc[k] += (float)x[j + k];
    for (int k = 0; k < LN_LANES; k++) s += acc[k];
    for (; j < d; j++) s += (float)x[j];
    return s;
}
static float ln_sqdev(const _Float16 *x, int d, float mean)
{
    float acc[LN_LANES] = {0}, s = 0;
    int j = 0;
    for (; j + LN_LANES <= d; j += LN_LANES)
        for (int k = 0; k < LN_LANES; k++) { float t = (float)x[j + k] - mean; acc[k] += t * t; }
    for (int k = 0; k < LN_LANES; k++) s += acc[k];
    for (; j < d; j++) { float t = (float)x[j] - mean; s += t * t; }
    return s;
}
static void h_layernorm(int M, int d, const _Float16 *x, const _Float16 *g,
                        const _Float16 *b, float eps, _Float16 *out)
{
    for (int i = 0; i < M; i++) {
        const _Float16 *xr = x + (size_t)i * d;
        const float mean = ln_sum(xr, d) / d;
        const float inv = 1.f / sqrtf(ln_sqdev(xr, d, mean) / d + eps);
        _Float16 *o = out + (size_t)i * d;
        MM_NO_OVERLAP
        for (int j = 0; j < d; j++)
            o[j] = (_Float16)(((float)xr[j] - mean) * inv * (float)g[j] + (float)b[j]);
    }
}
/* host parallel-for over [0,n): split into <=nt contiguous chunks (the host elementwise
 * ops — softmax, GELU — are memory-bound and embarrassingly parallel over rows/elements).
 * Each worker pins to an A76 big core, exactly like the NPU fan-out workers — otherwise the
 * scheduler scatters them onto the A55 little cores and the join barrier stalls on the slow
 * straggler (measured ~1.1x slower + far noisier on the SigLIP-B/16 encode). Default-on;
 * ROCKET_SIGLIP_PIN=0 reverts (e.g. a stream pool spreading across all 8 cores, which should
 * instead set a per-process rocket_affinity_set_base). */
static int sl_pin = 1;
static pthread_once_t sl_pin_once = PTHREAD_ONCE_INIT;
static void sl_pin_init(void) { const char *e = getenv("ROCKET_SIGLIP_PIN"); if (e && atoi(e) == 0) sl_pin = 0; }
static int sl_pin_enabled(void) { pthread_once(&sl_pin_once, sl_pin_init); return sl_pin; }

typedef struct {
    void (*fn)(void *, int, int); void *arg; int lo, hi, idx;
    int core_base;   /* big-core rotation base inherited from the CALLING thread */
} prange_t;
static void *prange_thunk(void *p)
{
    prange_t *r = p;
    /* _based, not rocket_pin_worker: the rotation base is __thread, so a freshly created
     * worker always reads 0 and two in-process pools would collide on the same A76s. */
    if (sl_pin_enabled()) rocket_pin_worker_based(r->idx, r->core_base);
    r->fn(r->arg, r->lo, r->hi);
    return NULL;
}
static void parallel_for(int n, int nt, void (*fn)(void *, int, int), void *arg)
{
    if (nt < 2 || n < 2 * nt) { fn(arg, 0, n); return; }
    if (nt > 8) nt = 8;
    const int core_base = rocket_affinity_get_base();   /* read on the CALLING thread */
    pthread_t th[8]; prange_t r[8];
    int chunk = (n + nt - 1) / nt, spawned = 0;
    for (int i = 0; i < nt; i++) {
        int lo = i * chunk, hi = lo + chunk;
        if (lo >= n) break;
        if (hi > n) hi = n;
        r[spawned].fn = fn; r[spawned].arg = arg; r[spawned].lo = lo; r[spawned].hi = hi;
        r[spawned].idx = spawned; r[spawned].core_base = core_base;
        const int rc = sl_pin_enabled()
            ? rocket_thread_create_pinned(&th[spawned], prange_thunk, &r[spawned], spawned, core_base)
            : pthread_create(&th[spawned], NULL, prange_thunk, &r[spawned]);
        if (rc == 0) spawned++;
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
 * resident encode's #2 cost once attention is multicore. Built once, through pthread_once
 * in h_gelu_tanh before the parallel fan-out, so two encodes on two threads neither build
 * it twice nor read it half-built. ROCKET_SIGLIP_GELU_SCALAR=1 forces the scalar path
 * (A/B). */
static _Float16 g_gelu_lut[65536];
static pthread_once_t g_gelu_once = PTHREAD_ONCE_INIT;
static void gelu_lut_build(void)
{
    for (int u = 0; u < 65536; u++) {
        uint16_t bits = (uint16_t)u; _Float16 h; memcpy(&h, &bits, sizeof h);
        g_gelu_lut[u] = gelu1(h);
    }
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
    pthread_once(&g_gelu_once, gelu_lut_build);   /* before the fan-out */
    parallel_for((int)n, nt, gelu_range, x);
}

/* CLIP's quick-GELU: x*sigmoid(1.702x). Same fp16-input LUT trick as gelu-tanh (all 65536
 * outputs fit one table, bit-exact to the scalar path incl inf/nan), so the hot activation is
 * a load. Built once before the parallel fan-out. */
static inline _Float16 quickgelu1(_Float16 xv)
{
    float v = (float)xv;
    return (_Float16)(v / (1.f + expf(-1.702f * v)));
}
static _Float16 g_qgelu_lut[65536];
static pthread_once_t g_qgelu_once = PTHREAD_ONCE_INIT;
static void qgelu_lut_build(void)
{
    for (int u = 0; u < 65536; u++) {
        uint16_t bits = (uint16_t)u; _Float16 h; memcpy(&h, &bits, sizeof h);
        g_qgelu_lut[u] = quickgelu1(h);
    }
}
static void qgelu_range(void *a, int lo, int hi)
{
    _Float16 *x = a;
    for (int i = lo; i < hi; i++) {
        uint16_t bits; memcpy(&bits, &x[i], sizeof bits);
        x[i] = g_qgelu_lut[bits];
    }
}
static void qgelu_range_scalar(void *a, int lo, int hi)
{
    _Float16 *x = a;
    for (int i = lo; i < hi; i++) x[i] = quickgelu1(x[i]);
}
static void h_quick_gelu(size_t n, _Float16 *x, int nt)
{
    if (getenv("ROCKET_SIGLIP_GELU_SCALAR")) { parallel_for((int)n, nt, qgelu_range_scalar, x); return; }
    pthread_once(&g_qgelu_once, qgelu_lut_build);
    parallel_for((int)n, nt, qgelu_range, x);
}

/* DINOv2's exact erf GELU x*Phi(x) (what Depth Anything v2 exports as an Erf node). Same
 * fp16-input LUT trick, bit-exact to the scalar path. */
static inline _Float16 erfgelu1(_Float16 xv)
{
    const float inv_sqrt2 = 0.70710678118654752f;
    float v = (float)xv;
    return (_Float16)(0.5f * v * (1.f + erff(v * inv_sqrt2)));
}
static _Float16 g_egelu_lut[65536];
static pthread_once_t g_egelu_once = PTHREAD_ONCE_INIT;
static void egelu_lut_build(void)
{
    for (int u = 0; u < 65536; u++) {
        uint16_t bits = (uint16_t)u; _Float16 h; memcpy(&h, &bits, sizeof h);
        g_egelu_lut[u] = erfgelu1(h);
    }
}
static void egelu_range(void *a, int lo, int hi)
{
    _Float16 *x = a;
    for (int i = lo; i < hi; i++) {
        uint16_t bits; memcpy(&bits, &x[i], sizeof bits);
        x[i] = g_egelu_lut[bits];
    }
}
static void egelu_range_scalar(void *a, int lo, int hi)
{
    _Float16 *x = a;
    for (int i = lo; i < hi; i++) x[i] = erfgelu1(x[i]);
}
static void h_erf_gelu(size_t n, _Float16 *x, int nt)
{
    if (getenv("ROCKET_SIGLIP_GELU_SCALAR")) { parallel_for((int)n, nt, egelu_range_scalar, x); return; }
    pthread_once(&g_egelu_once, egelu_lut_build);
    parallel_for((int)n, nt, egelu_range, x);
}

/* the model's MLP nonlinearity (host, in place over n elements) */
static void h_mlp_act(const rocket_siglip_model *m, size_t n, _Float16 *x, int nt)
{
    if      (m->act_kind == 1) h_quick_gelu(n, x, nt);
    else if (m->act_kind == 2) h_erf_gelu(n, x, nt);
    else                       h_gelu_tanh(n, x, nt);
}

static void h_add_bias(int M, int N, _Float16 *C, const _Float16 *b)
{
    if (!b) return;
    for (int i = 0; i < M; i++) {
        _Float16 *r = C + (size_t)i * N;
        MM_NO_OVERLAP   /* C is scratch and b a model weight */
        for (int j = 0; j < N; j++) r[j] = (_Float16)((float)r[j] + (float)b[j]);
    }
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
/* dst[c*ld_dst + t] = src[t*ld_src + c] for t < rows, c < cols: one head's V columns [ntok][dh]
 * to the AV operand's [dh][ntok]. 8x8 halfword blocks write whole lines. One element at a time is
 * a store every 2740 bytes at Depth Anything's 1370 tokens, and its cost moved 15-25 ms an image
 * with where the process's buffers landed. */
static void h_transpose_u16(uint16_t *dst, size_t ld_dst, const uint16_t *src, size_t ld_src,
                            int rows, int cols)
{
    int t = 0;
#if defined(__aarch64__)
    for (; t + 8 <= rows; t += 8) {
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            uint16x8_t blk[8];
            for (int r = 0; r < 8; r++) blk[r] = vld1q_u16(src + (size_t)(t + r) * ld_src + c);
            rc_tr8x8_u16(blk);
            for (int k = 0; k < 8; k++) vst1q_u16(dst + (size_t)(c + k) * ld_dst + t, blk[k]);
        }
        for (; c < cols; c++)
            for (int r = 0; r < 8; r++) dst[(size_t)c * ld_dst + t + r] = src[(size_t)(t + r) * ld_src + c];
    }
#endif
    for (; t < rows; t++)
        for (int c = 0; c < cols; c++) dst[(size_t)c * ld_dst + t] = src[(size_t)t * ld_src + c];
}

/* acc and add are always two of the ctx's scratch buffers. Those come from the ctx, so GCC at -O2
 * cannot see they are distinct and will not version the loop for it: unmarked it runs scalar. */
static void h_residual(size_t n, _Float16 *acc, const _Float16 *add)
{
    MM_NO_OVERLAP
    for (size_t i = 0; i < n; i++) acc[i] = (_Float16)((float)acc[i] + (float)add[i]);
}

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
    c->aux_fd = -1;   /* a failure before rocket_open() must not close fd 0 */
    c->ht = (nthreads <= 1) ? 1 : 4;   /* host softmax/GELU across the 4 A76 big cores */
    const int d = m->d, L = m->L, pdim = m->patch_dim, dff = m->d_ff, nL = m->n_layers;
    const int pdp = m->patch_dim_pad ? m->patch_dim_pad : pdim;   /* patch matmul K (%32) */
    if (m->n_taps < 0 || m->n_taps > ROCKET_SIGLIP_MAX_TAPS) goto fail;
    for (int t = 0; t < m->n_taps; t++)
        if (m->tap_layer[t] < 0 || m->tap_layer[t] >= nL ||
            (t && m->tap_layer[t] <= m->tap_layer[t - 1])) goto fail;   /* ascending, in range */
    /* The token-projection matmuls run over ntok = L + n_prefix rows (the cls token joins the
     * sequence); the patch projection runs over L patch rows. Both need M % 4 == 0, so pack the
     * resident weights at the padded M the prepacked calls will use (the layout is M-independent
     * for M >= 256, but Mp/Lp < 256 here, so pack-M and call-M must agree exactly). */
    const int Mp = ((L + m->n_prefix) + 3) & ~3;
    const int Lp = (L + 3) & ~3;

    c->mm = rocket_ctx_create(nthreads);
    c->strm = rocket_stream_create(1);   /* attention matmuls are small (K=64/N=64) — N-split hurts */
    c->aux_fd = rocket_open();
    c->wq = calloc(nL, sizeof(*c->wq));  c->wk = calloc(nL, sizeof(*c->wk));
    c->wv = calloc(nL, sizeof(*c->wv));  c->wo = calloc(nL, sizeof(*c->wo));
    c->wf1 = calloc(nL, sizeof(*c->wf1)); c->wf2 = calloc(nL, sizeof(*c->wf2));
    if (!c->mm || !c->strm || c->aux_fd < 0 ||
        !c->wq || !c->wk || !c->wv || !c->wo || !c->wf1 || !c->wf2) goto fail;

    c->w_patch = rocket_weights_pack(c->mm, Lp, pdp, d, m->patch_W);
    if (!c->w_patch) goto fail;
    for (int l = 0; l < nL; l++) {
        layer_ptrs p = unpack_layer(m, l);
        c->wq[l]  = rocket_weights_pack(c->mm, Mp, d,   d,   p.Wq);
        c->wk[l]  = rocket_weights_pack(c->mm, Mp, d,   d,   p.Wk);
        c->wv[l]  = rocket_weights_pack(c->mm, Mp, d,   d,   p.Wv);
        c->wo[l]  = rocket_weights_pack(c->mm, Mp, d,   d,   p.Wo);
        c->wf1[l] = rocket_weights_pack(c->mm, Mp, d,   dff, p.Wf1);
        c->wf2[l] = rocket_weights_pack(c->mm, Mp, dff, d,   p.Wf2);
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

    /* Token buffers hold Mp rows and the patch buffers Lp; calloc leaves the pad rows and the
     * im2col tail zero for the life of the ctx (see the struct). */
    const int ntok = L + m->n_prefix, dh = d / m->n_head;
    const size_t Td = (size_t)ntok * d;
    c->patches = calloc((size_t)Lp * pdp, sizeof(_Float16));
    c->xp = calloc((size_t)Lp * d, sizeof(_Float16));
    c->x  = calloc((size_t)Mp * d, sizeof(_Float16));
    c->ln = calloc((size_t)Mp * d, sizeof(_Float16));
    c->q  = calloc((size_t)Mp * d, sizeof(_Float16));
    c->k  = calloc((size_t)Mp * d, sizeof(_Float16));
    c->v  = calloc((size_t)Mp * d, sizeof(_Float16));
    c->cc = calloc((size_t)Mp * d, sizeof(_Float16));
    c->y  = calloc((size_t)Mp * dff, sizeof(_Float16));
    if (!c->patches || !c->xp || !c->x || !c->ln || !c->q || !c->k || !c->v || !c->cc || !c->y)
        goto fail;
    if (c->fa) {
        c->Qh = calloc(Td, sizeof(_Float16)); c->Kh = calloc(Td, sizeof(_Float16));
        c->Vh = calloc(Td, sizeof(_Float16)); c->Oh = calloc(Td, sizeof(_Float16));
        if (!c->Qh || !c->Kh || !c->Vh || !c->Oh) goto fail;
    } else {
        c->qh  = malloc((size_t)ntok * dh * sizeof(_Float16));
        c->khh = malloc((size_t)ntok * dh * sizeof(_Float16));
        c->vhT = malloc((size_t)dh * ntok * sizeof(_Float16));
        c->sc  = malloc((size_t)m->n_head * ntok * ntok * sizeof(_Float16));
        if (!c->qh || !c->khh || !c->vhT || !c->sc) goto fail;
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
    free(c->patches); free(c->xp); free(c->x); free(c->ln); free(c->q); free(c->k); free(c->v);
    free(c->cc); free(c->y);
    free(c->Qh); free(c->Kh); free(c->Vh); free(c->Oh);
    free(c->qh); free(c->khh); free(c->vhT); free(c->sc);
    free(c);
}

int rocket_siglip_encode_ctx(rocket_siglip_ctx *c, const _Float16 *pixels_chw,
                             _Float16 *out, _Float16 *hidden_opt)
{
    if (!c || !pixels_chw || !out) return -1;
    /* rocket_pin_worker, not _based: this pins the CALLING thread, which is the one
     * case where reading the __thread base is right. Workers use _based. */
    if (sl_pin_enabled()) rocket_pin_worker(0);   /* keep this thread's serial parts on an A76 */
    const rocket_siglip_model *m = c->m;
    const int d = m->d, L = m->L, pdim = m->patch_dim, dff = m->d_ff, nh = m->n_head, dh = d / nh;
    const int pdp = m->patch_dim_pad ? m->patch_dim_pad : pdim;   /* patch matmul K (%32) */
    const int npre = m->n_prefix;
    const int ntok = L + npre;                    /* token-sequence length (cls prefix + patches) */
    const int Mp = (ntok + 3) & ~3;               /* token-projection matmul M % 4 == 0            */
    const int Lp = (L + 3) & ~3;                  /* patch-projection matmul M % 4 == 0            */
    const int side = m->image_size / m->stride, H = m->image_size, W = m->image_size;
    const int ic = m->ic, kh = m->kh, kw = m->kw, stride = m->stride;
    const float scale = 1.f / sqrtf((float)dh);
    const size_t Td = (size_t)ntok * d;           /* real token-sequence element count */
    int rc = -2, tap_next = 0;                    /* next multi-tap block to emit into `out` */

    const int use_fa = (c->fa != NULL);
    /* Token buffers hold Mp rows (padded to M%4); host elementwise touches only the first ntok, and
     * every projection matmul runs at M=Mp with the pad rows held at zero, so each matmul's pad-row
     * output stays zero (never poisons a real token, which the attention reads at n_tokens=ntok).
     * patches/xp hold Lp patch rows. All of it is the ctx's scratch, zeroed once at create.
     * FA path: head-major Q/K/V/OUT for the flash-attention fan-out (each ntok*d elems).
     * single-stream path: per-head gather scratch qh/khh/vhT + all-heads scores sc, softmaxed in
     * place on the host (one pass over [nh*ntok, ntok] rows), so PV reads sc directly. */
    _Float16 *const patches = c->patches, *const xp = c->xp, *const x = c->x, *const ln = c->ln;
    _Float16 *const q = c->q, *const k = c->k, *const v = c->v, *const cc = c->cc, *const y = c->y;
    _Float16 *const Qh = c->Qh, *const Kh = c->Kh, *const Vh = c->Vh, *const Oh = c->Oh;
    _Float16 *const qh = c->qh, *const khh = c->khh, *const vhT = c->vhT, *const sc = c->sc;

    int prof = getenv("ROCKET_SIGLIP_PROF") != NULL;
    long flt0 = 0;
    if (prof) { struct rusage ru; getrusage(RUSAGE_SELF, &ru); flt0 = ru.ru_minflt; }
    double tb[16] = {0}, t0 = 0;
#define TIC (t0 = prof_ms())
#define TOC(i) do { if (prof) tb[i] += prof_ms() - t0; } while (0)

    /* im2col (same ordering as the simple path); patches has L rows, pad rows L..Lp stay zero */
    TIC;
    for (int ph = 0; ph < side; ph++)
        for (int pw = 0; pw < side; pw++) {
            int p = ph * side + pw;
            _Float16 *row = patches + (size_t)p * pdp;
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

    /* patch projection (prepacked, M=Lp) -> xp, then assemble the token sequence: prefix (cls) rows
     * = prefix + pos; patch rows = patch_proj + patch_b (if any) + pos. pos is [ntok][d]. The
     * patch-row add keeps the SigLIP order (proj + patch_b + pos) so that path stays bit-identical. */
    TIC;
    if ((rc = rocket_matmul_fp16_prepacked(c->mm, Lp, pdp, d, patches, xp, c->w_patch)) != 0) goto done;
    for (int j = 0; j < npre; j++) {
        _Float16 *xr = x + (size_t)j * d;
        const _Float16 *cr = m->prefix + (size_t)j * d, *pr = m->pos + (size_t)j * d;
        for (int e = 0; e < d; e++) xr[e] = (_Float16)((float)cr[e] + (float)pr[e]);
    }
    for (int p = 0; p < L; p++) {
        _Float16 *xr = x + (size_t)(npre + p) * d;
        const _Float16 *sr = xp + (size_t)p * d, *pr = m->pos + (size_t)(npre + p) * d;
        for (int e = 0; e < d; e++) {
            float acc = (float)sr[e];
            if (m->patch_b) acc += (float)m->patch_b[e];
            acc += (float)pr[e];
            xr[e] = (_Float16)acc;
        }
    }
    /* optional pre-encoder LayerNorm (CLIP pre_layrnorm) over the token rows, in place */
    if (m->pre_g) { h_layernorm(ntok, d, x, m->pre_g, m->pre_b, m->eps, ln); memcpy(x, ln, Td * sizeof(_Float16)); }
    TOC(1);
    if (hidden_opt) memcpy(hidden_opt, x, Td * sizeof(_Float16));

    for (int l = 0; l < m->n_layers; l++) {
        layer_ptrs p = unpack_layer(m, l);

        /* --- attention sublayer: x += Wo·MHA(LN1(x)) --- */
        TIC; h_layernorm(ntok, d, x, p.ln1_g, p.ln1_b, m->eps, ln); TOC(2);
        TIC;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, Mp, d, d, ln, q, c->wq[l])) != 0) goto done;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, Mp, d, d, ln, k, c->wk[l])) != 0) goto done;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, Mp, d, d, ln, v, c->wv[l])) != 0) goto done;
        TOC(3);
        TIC;
        h_add_bias(ntok, d, q, p.bq); h_add_bias(ntok, d, k, p.bk); h_add_bias(ntok, d, v, p.bv);
        if (!use_fa)
            for (size_t i = 0; i < Td; i++) q[i] = (_Float16)((float)q[i] * scale);  /* fold 1/sqrt(dh) */
        TOC(14);

        if (use_fa) {
            /* re-lay q/k/v [ntok,d] (heads interleaved) into head-major operands and run all heads
             * across the worker fds: Q/K=[nh][ntok][dh], V=[nh][dh][ntok] (the AV B-operand),
             * OUT=[nh][ntok][dh]. scale is applied inside flash-attn (q is NOT pre-scaled). The FA
             * primitive pads the odd token count internally and masks the pad keys. */
            TIC;
            for (int h = 0; h < nh; h++) {
                int off = h * dh;
                _Float16 *Qd = Qh + (size_t)h * ntok * dh, *Kd = Kh + (size_t)h * ntok * dh;
                _Float16 *Vd = Vh + (size_t)h * dh * ntok;
                for (int t = 0; t < ntok; t++) {
                    memcpy(Qd + (size_t)t * dh, q + (size_t)t * d + off, dh * sizeof(_Float16));
                    memcpy(Kd + (size_t)t * dh, k + (size_t)t * d + off, dh * sizeof(_Float16));
                }
                h_transpose_u16((uint16_t *)Vd, (size_t)ntok, (const uint16_t *)(v + off), (size_t)d,
                                ntok, dh);
            }
            TOC(4);
            TIC;
            if ((rc = rocket_flash_attn_fp16_ctx(c->fa, ntok, ntok, dh, dh, nh, nh, scale, 0.f,
                                                 Qh, Kh, Vh, NULL, Oh)) != 0) goto done;
            TOC(5);   /* scores + softmax + PV, fanned across cores */
            TIC;
            for (int h = 0; h < nh; h++) {
                int off = h * dh;
                const _Float16 *Od = Oh + (size_t)h * ntok * dh;
                for (int t = 0; t < ntok; t++)
                    memcpy(cc + (size_t)t * d + off, Od + (size_t)t * dh, dh * sizeof(_Float16));
            }
            TOC(8);
        } else {
        /* phase 1: every head's scores q_h·k_h^T into sc[h] (stacked by row) */
        for (int h = 0; h < nh; h++) {
            int off = h * dh;
            TIC;
            for (int t = 0; t < ntok; t++) {
                memcpy(qh  + (size_t)t * dh, q + (size_t)t * d + off, dh * sizeof(_Float16));
                memcpy(khh + (size_t)t * dh, k + (size_t)t * d + off, dh * sizeof(_Float16));
            }
            TOC(4);
            TIC; if ((rc = cmatmul(c, ntok, dh, ntok, qh, khh, sc + (size_t)h * ntok * ntok)) != 0) goto done; TOC(5);
        }
        /* phase 2: row-wise softmax on the host, in place (no NPU round-trip), threaded */
        TIC; h_softmax_rows(nh * ntok, ntok, sc, c->ht); TOC(6);
        /* phase 3: every head's ctx_h = P_h·v_h, scattered into cc columns [off,off+dh) */
        for (int h = 0; h < nh; h++) {
            int off = h * dh;
            TIC;
            h_transpose_u16((uint16_t *)vhT, (size_t)ntok, (const uint16_t *)(v + off), (size_t)d,
                            ntok, dh);
            TOC(4);
            TIC; if ((rc = cmatmul(c, ntok, ntok, dh, sc + (size_t)h * ntok * ntok, vhT, qh)) != 0) goto done; TOC(7);
            TIC;
            for (int t = 0; t < ntok; t++) memcpy(cc + (size_t)t * d + off, qh + (size_t)t * dh, dh * sizeof(_Float16));
            TOC(8);
        }
        }
        TIC;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, Mp, d, d, cc, y, c->wo[l])) != 0) goto done;
        TOC(9);
        TIC; h_add_bias(ntok, d, y, p.bo); TOC(14);
        TIC; h_residual(Td, x, y); TOC(13);

        /* --- FFN sublayer: x += fc2(ACT(fc1(LN2(x)))) --- ACT = gelu-tanh (SigLIP) / quick-GELU (CLIP) */
        TIC; h_layernorm(ntok, d, x, p.ln2_g, p.ln2_b, m->eps, ln); TOC(2);
        TIC;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, Mp, d, dff, ln, y, c->wf1[l])) != 0) goto done;
        TOC(10);
        TIC; h_add_bias(ntok, dff, y, p.bf1); TOC(14);
        TIC; h_mlp_act(m, (size_t)ntok * dff, y, c->ht); TOC(11);
        TIC;
        if ((rc = rocket_matmul_fp16_prepacked(c->mm, Mp, dff, d, y, cc, c->wf2[l])) != 0) goto done;
        TOC(12);
        TIC; h_add_bias(ntok, d, cc, p.bf2); TOC(14);
        TIC; h_residual(Td, x, cc); TOC(13);

        if (hidden_opt) memcpy(hidden_opt + (size_t)(l + 1) * Td, x, Td * sizeof(_Float16));
        /* multi-tap exit (DPT heads): emit this layer's output NOW, while it is the live
         * residual -- the next layer adds into x in place. */
        if (tap_next < m->n_taps && m->tap_layer[tap_next] == l) {
            TIC;
            _Float16 *ob = out + (size_t)tap_next * Td;
            if (m->post_g) h_layernorm(ntok, d, x, m->post_g, m->post_b, m->eps, ob);
            else           memcpy(ob, x, Td * sizeof(_Float16));
            TOC(2);
            tap_next++;
        }
    }

    /* single exit (SigLIP/CLIP): the optional post-LayerNorm over the sequence. CLIP leaves
     * last_hidden_state un-normed (its post_layernorm applies only to the pooled cls row, done
     * on the host tail). A multi-tap model wrote every block inside the loop above. */
    if (m->n_taps == 0) {
        TIC;
        if (m->post_g) h_layernorm(ntok, d, x, m->post_g, m->post_b, m->eps, out);
        else           memcpy(out, x, Td * sizeof(_Float16));
        TOC(2);
    }
    rc = 0;
    if (prof) {
        const char *nm[15] = {"im2col","patch+pos","layernorm","qkv-proj","head-xpose",
            "scores-mm","softmax","pv-mm","ctx-scatter","o-proj","fc1","act","fc2","residual","bias"};
        double tot = 0; for (int i = 0; i < 15; i++) tot += tb[i];
        struct rusage ru; getrusage(RUSAGE_SELF, &ru);
        ROCKET_LOGE("[siglip prof] total %.1f ms, %ld minor faults:\n", tot, ru.ru_minflt - flt0);
        for (int i = 0; i < 15; i++)
            ROCKET_LOGE("    %-12s %8.1f ms  (%4.1f%%)\n", nm[i], tb[i], 100.0 * tb[i] / (tot + 1e-9));
    }
#undef TIC
#undef TOC
done:
    return rc;
}
