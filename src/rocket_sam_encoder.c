// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_sam_encoder.c — the SAM ViT-Det image encoder (facebook/sam-vit-base) end to end
 * on the NPU, glue over the validated primitives. See rocket_sam.h for the graph, the blob
 * format, and the decomposed rel-pos math.
 *
 * This file holds the model blob load/free, the one-shot per-call encode (rocket_sam_encode:
 * the correctness reference; fd<0 is a pure-host datapath), and the resident multicore ctx
 * (rocket_sam_encode_ctx: the latency path). The two share the datapath helpers below.
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

#include "rocket_sam.h"
#include "rocket_matmul.h"    /* rocket_matmul_fp16 + prepacked/stream */
#include "rocket_conv.h"      /* rocket_conv2d_fp16 (neck conv3x3)     */
#include "rocket_npu.h"       /* rocket_open / rocket_close            */
#include "rocket_affinity.h"  /* rocket_pin_worker                    */
#include "rocket_log.h"

#define SAM_MAGIC   0x53414D42   /* "SAMB" */
#define SAM_VERSION 1

/* ============================================================================
 * SECTION — Model blob load / free
 * ==========================================================================*/

int rocket_sam_load(const char *path, rocket_sam_model *m)
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
    if (h[0] != SAM_MAGIC || h[1] != SAM_VERSION) { munmap(map, st.st_size); return -3; }
    m->map = map; m->map_size = (size_t)st.st_size;
    m->d = h[2]; m->n_layers = h[3]; m->n_head = h[4]; m->d_ff = h[5];
    m->grid = h[6]; m->win = h[7]; m->patch = h[8]; m->ic = h[9]; m->neck_out = h[10];
    memcpy(&m->eps, &h[11], sizeof(float));
    m->image_size = h[12]; m->dhead = h[13]; m->patch_dim = h[14];

    if (m->d <= 0 || m->n_layers <= 0 || m->n_layers > ROCKET_SAM_MAX_LAYERS ||
        m->n_head <= 0 || m->d_ff <= 0 || m->grid <= 0 || m->win <= 0 ||
        m->dhead <= 0 || (m->d % m->n_head) || m->dhead != m->d / m->n_head ||
        m->patch_dim != m->ic * m->patch * m->patch) { munmap(map, st.st_size); return -4; }

    const int32_t *flags = h + 16;
    for (int l = 0; l < m->n_layers; l++) {
        m->windowed[l] = flags[l] ? 1 : 0;
        m->Sl[l] = m->windowed[l] ? m->win : m->grid;
    }

    const int d = m->d, dff = m->d_ff, dh = m->dhead, no = m->neck_out, gg = m->grid * m->grid;
    const char *base = (const char *)map + 64 + (size_t)m->n_layers * sizeof(int32_t);
    const _Float16 *w = (const _Float16 *)base;
    size_t off = 0;
    m->patch_W = w + off; off += (size_t)d * m->patch_dim;
    m->patch_b = w + off; off += d;
    m->pos     = w + off; off += (size_t)gg * d;
    for (int l = 0; l < m->n_layers; l++) {
        const int S = m->Sl[l];
        m->ln1_g[l] = w + off; off += d;   m->ln1_b[l] = w + off; off += d;
        m->Wqkv[l]  = w + off; off += (size_t)3 * d * d;  m->bqkv[l] = w + off; off += (size_t)3 * d;
        m->Wo[l]    = w + off; off += (size_t)d * d;      m->bo[l]   = w + off; off += d;
        m->ln2_g[l] = w + off; off += d;   m->ln2_b[l] = w + off; off += d;
        m->Wf1[l]   = w + off; off += (size_t)dff * d;    m->bf1[l]  = w + off; off += dff;
        m->Wf2[l]   = w + off; off += (size_t)d * dff;    m->bf2[l]  = w + off; off += d;
        m->Rh[l]    = w + off; off += (size_t)S * S * dh;
        m->Rw[l]    = w + off; off += (size_t)S * S * dh;
    }
    m->neck_c1    = w + off; off += (size_t)no * d;
    m->neck_ln1_g = w + off; off += no;  m->neck_ln1_b = w + off; off += no;
    m->neck_c2    = w + off; off += (size_t)no * no * 3 * 3;
    m->neck_ln2_g = w + off; off += no;  m->neck_ln2_b = w + off; off += no;

    size_t want = 64 + (size_t)m->n_layers * sizeof(int32_t) + off * sizeof(_Float16);
    if (want != m->map_size) { munmap(map, st.st_size); memset(m, 0, sizeof(*m)); return -4; }
    return 0;
}

void rocket_sam_free(rocket_sam_model *m)
{
    if (m && m->map) { munmap(m->map, m->map_size); m->map = NULL; }
}

/* ============================================================================
 * SECTION — Host datapath helpers (shared by one-shot + resident)
 * ==========================================================================*/

/* host fp32-accumulate matmul C[M][N] = A[M][K]·B[N][K]^T (fp16 store). */
static void host_matmul(int M, int K, int N, const _Float16 *A, const _Float16 *B, _Float16 *C)
{
    for (int i = 0; i < M; i++) {
        const _Float16 *ar = A + (size_t)i * K;
        _Float16 *cr = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const _Float16 *br = B + (size_t)j * K;
            float a = 0;
            for (int k = 0; k < K; k++) a += (float)ar[k] * (float)br[k];
            cr[j] = (_Float16)a;
        }
    }
}

/* one-shot matmul: NPU when fd>=0, else host reference (C=A·B^T, fp16). */
static int mm(int fd, int M, int K, int N, const _Float16 *A, const _Float16 *B, _Float16 *C)
{
    if (fd >= 0) return rocket_matmul_fp16(fd, M, K, N, A, B, C);
    host_matmul(M, K, N, A, B, C);
    return 0;
}

/* LayerNorm over the last axis (d), fp32 reduce, fp16 store. */
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

/* channels-first LayerNorm (SamLayerNorm): for each spatial position p, normalize over the C
 * channels (stride P in an [C,P] channels-major buffer), affine per channel. eps 1e-6. */
static void h_layernorm2d(int C, int P, const _Float16 *x, const _Float16 *g,
                          const _Float16 *b, float eps, _Float16 *out)
{
    for (int p = 0; p < P; p++) {
        double mean = 0; for (int c = 0; c < C; c++) mean += (double)x[(size_t)c * P + p]; mean /= C;
        double var = 0; for (int c = 0; c < C; c++) { double t = (double)x[(size_t)c * P + p] - mean; var += t * t; }
        double inv = 1.0 / sqrt(var / C + eps);
        for (int c = 0; c < C; c++)
            out[(size_t)c * P + p] = (_Float16)(((double)x[(size_t)c * P + p] - mean) * inv *
                                                (double)g[c] + (double)b[c]);
    }
}

/* erf-GELU LUT (exact 0.5*x*(1+erf(x/sqrt2)); all 65536 fp16 inputs). Built once. */
static _Float16 g_gelu_lut[65536];
static int      g_gelu_lut_ready;
static pthread_once_t g_gelu_once = PTHREAD_ONCE_INIT;
static void gelu_lut_build(void)
{
    const float inv_sqrt2 = 0.70710678118654752f;
    for (int u = 0; u < 65536; u++) {
        uint16_t bits = (uint16_t)u; _Float16 hv; memcpy(&hv, &bits, sizeof hv);
        float v = (float)hv;
        g_gelu_lut[u] = (_Float16)(0.5f * v * (1.f + erff(v * inv_sqrt2)));
    }
    g_gelu_lut_ready = 1;
}
static void h_gelu_erf(size_t n, _Float16 *x)
{
    pthread_once(&g_gelu_once, gelu_lut_build);
    for (size_t i = 0; i < n; i++) { uint16_t b; memcpy(&b, &x[i], sizeof b); x[i] = g_gelu_lut[b]; }
}

static void h_add_bias(int M, int N, _Float16 *C, const _Float16 *b)
{
    for (int i = 0; i < M; i++) { _Float16 *r = C + (size_t)i * N; for (int j = 0; j < N; j++) r[j] = (_Float16)((float)r[j] + (float)b[j]); }
}
static void h_residual(size_t n, _Float16 *acc, const _Float16 *add)
{ for (size_t i = 0; i < n; i++) acc[i] = (_Float16)((float)acc[i] + (float)add[i]); }

/* round up to a multiple of 32 (the key/value pad for the QK N% 16 + PV K% 32 alignment). */
static inline int round32(int n) { return (n + 31) & ~31; }

/* ============================================================================
 * SECTION — Attention (windowed + global), shared inner over a "matmul provider"
 * ==========================================================================*/

/* A minimal matmul provider so the one-shot (fd) and resident (ctx stream) paths share the
 * exact attention datapath. `qk` computes C[M,K,N]=A·B^T for a computed (both-activation)
 * matmul; the caller supplies whichever backing (one-shot rocket_matmul_fp16 or a stream). */
typedef struct { void *self; int (*run)(void *self, int M, int K, int N,
                                         const _Float16 *A, const _Float16 *B, _Float16 *C); } mm_provider;

/* SHARED scratch for one attention sublayer (produced/consumed outside the parallel region). */
typedef struct {
    _Float16 *hw;      /* window-partitioned LN1 output [nWinMax*S2max][d] (windowed) */
    _Float16 *qkv;     /* [Mtok][3d]  (read-only during the parallel head loop)       */
    _Float16 *ctxf;    /* per-token attention context [grid*grid][d] (disjoint writes) */
    int      *ip, *jp; /* key row/col index tables [S2max] (read-only)                 */
} attn_scratch;

/* PER-HEAD scratch: one per worker thread, so heads run concurrently without sharing state. */
typedef struct {
    _Float16 *qh, *kh, *vh; /* per-head q/k/v gather [S2][dhead]     */
    _Float16 *khp;          /* padded key [Kpad][dhead]             */
    _Float16 *vtp;          /* padded v^T [dhead][Kpad]             */
    _Float16 *sc;           /* scores [S2][Kpad]                    */
    _Float16 *outh;         /* per-head context [S2][dhead]         */
    _Float16 *relh, *relw;  /* rel_h / rel_w [S2][S]                */
    _Float16 *qcol, *blk;   /* rel-pos small-matmul scratch         */
} head_scratch;

/* rel-pos: fill hs->relh[t][i'] and hs->relw[t][j'] for hs->qh[S2][dhead], side S. Host. */
static void rel_pos(head_scratch *hs, const _Float16 *Rh, const _Float16 *Rw, int S, int dh)
{
    const _Float16 *qh = hs->qh;
    /* rel_h: rows i*S..(i+1)*S of qh are contiguous -> block = qh_row_i @ Rh[i]^T [S(j),S(i')] */
    for (int i = 0; i < S; i++)
        host_matmul(S, dh, S, qh + (size_t)i * S * dh, Rh + (size_t)i * S * dh, hs->relh + (size_t)i * S * S);
    /* rel_w: gather qh column j (rows j, j+S, ...) -> block = qcol @ Rw[j]^T [S(i),S(j')] */
    for (int j = 0; j < S; j++) {
        for (int i = 0; i < S; i++)
            memcpy(hs->qcol + (size_t)i * dh, qh + (size_t)(i * S + j) * dh, (size_t)dh * sizeof(_Float16));
        host_matmul(S, dh, S, hs->qcol, Rw + (size_t)j * S * dh, hs->blk);   /* [S(i)][S(j')] */
        for (int i = 0; i < S; i++)
            memcpy(hs->relw + (size_t)(i * S + j) * S, hs->blk + (size_t)i * S, (size_t)S * sizeof(_Float16));
    }
}

/* biased softmax over one query row: p[t'] = softmax_t'( scale*sc[t'] + relh[i'] + relw[j'] ),
 * pad columns [S2,Kpad) set to 0 (excluded). sc_row overwritten with p. */
static void attn_softmax_row(_Float16 *sc_row, int S2, int Kpad, float scale,
                             const _Float16 *relh_t, const _Float16 *relw_t,
                             const int *ip, const int *jp)
{
    float mx = -INFINITY;
    for (int t = 0; t < S2; t++) {
        float v = (float)sc_row[t] * scale + (float)relh_t[ip[t]] + (float)relw_t[jp[t]];
        sc_row[t] = (_Float16)v; if (v > mx) mx = v;
    }
    float sum = 0;
    for (int t = 0; t < S2; t++) { float e = expf((float)sc_row[t] - mx); sc_row[t] = (_Float16)e; sum += e; }
    for (int t = S2; t < Kpad; t++) sc_row[t] = 0;
    float inv = 1.f / sum;
    for (int t = 0; t < S2; t++) sc_row[t] = (_Float16)((float)sc_row[t] * inv);
}

/* One attention head over one window: hs->qh/kh/vh [S2][dhead] pre-gathered, side S. Writes
 * hs->outh[S2][dhead] = proj-free per-head context. `P` runs QK / PV; `ip`/`jp` are shared. */
static int attn_one_head(mm_provider *P, head_scratch *hs, const _Float16 *Rh, const _Float16 *Rw,
                         int S, int dh, float scale, const int *ip, const int *jp)
{
    const int S2 = S * S, Kpad = round32(S2);
    /* pad keys to Kpad rows (zero) and v^T to Kpad cols (zero) */
    memset(hs->khp, 0, (size_t)Kpad * dh * sizeof(_Float16));
    memcpy(hs->khp, hs->kh, (size_t)S2 * dh * sizeof(_Float16));
    memset(hs->vtp, 0, (size_t)dh * Kpad * sizeof(_Float16));
    for (int t = 0; t < S2; t++) for (int c = 0; c < dh; c++) hs->vtp[(size_t)c * Kpad + t] = hs->vh[(size_t)t * dh + c];
    /* scores = qh · khp^T  [S2][Kpad] (unscaled; scale folded in the softmax) */
    if (P->run(P->self, S2, dh, Kpad, hs->qh, hs->khp, hs->sc) != 0) return -1;
    rel_pos(hs, Rh, Rw, S, dh);
    for (int t = 0; t < S2; t++)
        attn_softmax_row(hs->sc + (size_t)t * Kpad, S2, Kpad, scale,
                         hs->relh + (size_t)t * S, hs->relw + (size_t)t * S, ip, jp);
    /* context = P · v  [S2][dhead] = sc · vtp^T */
    if (P->run(P->self, S2, Kpad, dh, hs->sc, hs->vtp, hs->outh) != 0) return -1;
    return 0;
}

/* gather head h's q/k/v for window w out of the fused qkv [Mtok][3d] into hs->qh/kh/vh. */
static void gather_head(head_scratch *hs, const _Float16 *qkv, int w, int h, int S2, int d, int dh)
{
    const _Float16 *q = qkv, *k = qkv + d, *v = qkv + 2 * d;
    const int off = h * dh;
    for (int t = 0; t < S2; t++) {
        const size_t b = (size_t)(w * S2 + t) * 3 * d + off;
        memcpy(hs->qh + (size_t)t * dh, q + b, (size_t)dh * sizeof(_Float16));
        memcpy(hs->kh + (size_t)t * dh, k + b, (size_t)dh * sizeof(_Float16));
        memcpy(hs->vh + (size_t)t * dh, v + b, (size_t)dh * sizeof(_Float16));
    }
}

/* ============================================================================
 * SECTION — Windowing (host)
 * ==========================================================================*/

/* window_partition: LN1 output h[grid*grid][d] -> hw[nWin*S2][d], window-major, pad tokens 0.
 * nWin = (gp/win)^2 where gp = grid padded up to a multiple of win. Returns nWin. */
static int window_partition(const _Float16 *h, int grid, int win, int d, _Float16 *hw)
{
    const int gp = ((grid + win - 1) / win) * win, nw = gp / win, S2 = win * win;
    memset(hw, 0, (size_t)nw * nw * S2 * d * sizeof(_Float16));
    for (int wr = 0; wr < nw; wr++)
        for (int wc = 0; wc < nw; wc++) {
            _Float16 *wdst = hw + (size_t)(wr * nw + wc) * S2 * d;
            for (int i = 0; i < win; i++) {
                int gr = wr * win + i; if (gr >= grid) continue;
                for (int j = 0; j < win; j++) {
                    int gc = wc * win + j; if (gc >= grid) continue;
                    memcpy(wdst + (size_t)(i * win + j) * d, h + (size_t)(gr * grid + gc) * d,
                           (size_t)d * sizeof(_Float16));
                }
            }
        }
    return nw * nw;
}

/* window_unpartition: attention context cc[nWin*S2][d] -> ctxf[grid*grid][d] (drop pad tokens). */
static void window_unpartition(const _Float16 *cc, int grid, int win, int d, _Float16 *ctxf)
{
    const int gp = ((grid + win - 1) / win) * win, nw = gp / win, S2 = win * win;
    for (int wr = 0; wr < nw; wr++)
        for (int wc = 0; wc < nw; wc++) {
            const _Float16 *wsrc = cc + (size_t)(wr * nw + wc) * S2 * d;
            for (int i = 0; i < win; i++) {
                int gr = wr * win + i; if (gr >= grid) continue;
                for (int j = 0; j < win; j++) {
                    int gc = wc * win + j; if (gc >= grid) continue;
                    memcpy(ctxf + (size_t)(gr * grid + gc) * d, wsrc + (size_t)(i * win + j) * d,
                           (size_t)d * sizeof(_Float16));
                }
            }
        }
}

/* ============================================================================
 * SECTION — One-shot encode (rocket_sam_encode)
 * ==========================================================================*/

/* provider bound to a single fd (or host when fd<0). */
typedef struct { int fd; } os_mm;
static int os_run(void *self, int M, int K, int N, const _Float16 *A, const _Float16 *B, _Float16 *C)
{ return mm(((os_mm *)self)->fd, M, K, N, A, B, C); }

/* im2col patchify (stride==kernel, pad 0): patches[p][ci*kh*kw + r*kw + c]. */
static void im2col(const rocket_sam_model *m, const _Float16 *pix, _Float16 *patches)
{
    const int side = m->grid, ic = m->ic, kh = m->patch, kw = m->patch, W = m->image_size;
    const int pdim = m->patch_dim;
    for (int ph = 0; ph < side; ph++)
        for (int pw = 0; pw < side; pw++) {
            _Float16 *row = patches + (size_t)(ph * side + pw) * pdim;
            for (int ci = 0; ci < ic; ci++) {
                const _Float16 *chan = pix + (size_t)ci * W * W;
                for (int r = 0; r < kh; r++) {
                    const _Float16 *src = chan + (size_t)(ph * kh + r) * W + pw * kw;
                    _Float16 *dst = row + (size_t)ci * kh * kw + (size_t)r * kw;
                    for (int c = 0; c < kw; c++) dst[c] = src[c];
                }
            }
        }
}

/* build the key row/col index tables for side S (t' -> (i',j')). */
static void fill_ij(int S, int *ip, int *jp)
{ for (int t = 0; t < S * S; t++) { ip[t] = t / S; jp[t] = t % S; } }

static int alloc_scratch(attn_scratch *s, int grid, int win, int d)
{
    memset(s, 0, sizeof(*s));
    const int Smax = grid > win ? grid : win, S2max = Smax * Smax;
    const int gp = ((grid + win - 1) / win) * win, nwMax = (gp / win) * (gp / win);
    const int Mwin = nwMax * win * win;                 /* windowed padded-grid token count */
    s->hw   = malloc((size_t)Mwin * d * sizeof(_Float16));
    s->qkv  = malloc((size_t)Mwin * 3 * d * sizeof(_Float16));
    s->ctxf = malloc((size_t)grid * grid * d * sizeof(_Float16));
    s->ip   = malloc((size_t)S2max * sizeof(int));
    s->jp   = malloc((size_t)S2max * sizeof(int));
    return s->hw && s->qkv && s->ctxf && s->ip && s->jp;
}
static void free_scratch(attn_scratch *s)
{ free(s->hw); free(s->qkv); free(s->ctxf); free(s->ip); free(s->jp); }

static int alloc_head_scratch(head_scratch *hs, int grid, int win, int dh)
{
    memset(hs, 0, sizeof(*hs));
    const int Smax = grid > win ? grid : win, S2max = Smax * Smax, Kpad = round32(S2max);
    hs->qh   = malloc((size_t)S2max * dh * sizeof(_Float16));
    hs->kh   = malloc((size_t)S2max * dh * sizeof(_Float16));
    hs->vh   = malloc((size_t)S2max * dh * sizeof(_Float16));
    hs->khp  = malloc((size_t)Kpad * dh * sizeof(_Float16));
    hs->vtp  = malloc((size_t)dh * Kpad * sizeof(_Float16));
    hs->sc   = malloc((size_t)S2max * Kpad * sizeof(_Float16));
    hs->outh = malloc((size_t)S2max * dh * sizeof(_Float16));
    hs->relh = malloc((size_t)S2max * Smax * sizeof(_Float16));
    hs->relw = malloc((size_t)S2max * Smax * sizeof(_Float16));
    hs->qcol = malloc((size_t)Smax * dh * sizeof(_Float16));
    hs->blk  = malloc((size_t)Smax * Smax * sizeof(_Float16));
    return hs->qh && hs->kh && hs->vh && hs->khp && hs->vtp && hs->sc && hs->outh &&
           hs->relh && hs->relw && hs->qcol && hs->blk;
}
static void free_head_scratch(head_scratch *hs)
{
    free(hs->qh); free(hs->kh); free(hs->vh); free(hs->khp); free(hs->vtp); free(hs->sc);
    free(hs->outh); free(hs->relh); free(hs->relw); free(hs->qcol); free(hs->blk);
}

/* the neck: x_final[grid*grid][d] token-major -> out[neck_out][grid][grid]. Uses `provider`
 * for conv1 (1x1 matmul) and rocket_conv2d_fp16(fd) for conv3x3. */
static int sam_neck(int fd, mm_provider *P, const rocket_sam_model *m, const _Float16 *xf, _Float16 *out)
{
    const int d = m->d, no = m->neck_out, g = m->grid, gg = g * g;
    int rc = -1;
    _Float16 *y1  = malloc((size_t)gg * no * sizeof(_Float16));     /* conv1 out token-major */
    _Float16 *ncw = malloc((size_t)no * gg * sizeof(_Float16));     /* channels-first [no][gg] */
    _Float16 *ln  = malloc((size_t)no * gg * sizeof(_Float16));
    _Float16 *c2  = malloc((size_t)no * gg * sizeof(_Float16));
    if (!y1 || !ncw || !ln || !c2) goto done;
    /* conv1 1x1: y1[gg][no] = xf[gg][d] · neck_c1[no][d]^T */
    if (P->run(P->self, gg, d, no, xf, m->neck_c1, y1) != 0) goto done;
    /* -> channels-first [no][gg] */
    for (int p = 0; p < gg; p++) for (int c = 0; c < no; c++) ncw[(size_t)c * gg + p] = y1[(size_t)p * no + c];
    h_layernorm2d(no, gg, ncw, m->neck_ln1_g, m->neck_ln1_b, 1e-6f, ln);
    /* conv2 3x3 pad1, no bias */
    rocket_conv2d_desc dc = { .ic = no, .ih = g, .iw = g, .oc = no, .kh = 3, .kw = 3,
                              .stride_y = 1, .stride_x = 1, .pad_top = 1, .pad_left = 1,
                              .dil_y = 1, .dil_x = 1, .depthwise = 0 };
    if (rocket_conv2d_fp16(fd, &dc, ln, m->neck_c2, c2) != 0) goto done;
    h_layernorm2d(no, gg, c2, m->neck_ln2_g, m->neck_ln2_b, 1e-6f, out);
    rc = 0;
done:
    free(y1); free(ncw); free(ln); free(c2);
    return rc;
}

/* the shared per-layer body. qkv is packed into s->qkv [Mtok][3d]; this runs attention + FFN,
 * updating x in place. `P` provides every matmul (projections + computed QK/PV). */
static int sam_layer(mm_provider *P, const rocket_sam_model *m, int l,
                     _Float16 *x, attn_scratch *s)
{
    const int d = m->d, dff = m->d_ff, nh = m->n_head, dh = m->dhead, g = m->grid, gg = g * g;
    const int win = m->win, S = m->Sl[l], S2 = S * S;
    const float scale = 1.f / sqrtf((float)dh);
    const int windowed = m->windowed[l];
    int rc = -1;

    _Float16 *ln = malloc((size_t)gg * d * sizeof(_Float16));
    _Float16 *y  = malloc((size_t)gg * (dff > d ? dff : d) * sizeof(_Float16));
    if (!ln || !y) { free(ln); free(y); return -1; }

    /* --- attention: x += o( attn( LN1(x) ) ) --- */
    h_layernorm(gg, d, x, m->ln1_g[l], m->ln1_b[l], m->eps, ln);

    int nWin, Mtok;
    _Float16 *src;   /* token buffer feeding qkv: [Mtok][d] */
    if (windowed) {
        nWin = window_partition(ln, g, win, d, s->hw);   /* [nWin*S2][d] */
        Mtok = nWin * S2;
        src = s->hw;
    } else {
        nWin = 1; Mtok = gg; src = ln;
    }
    /* fused qkv: [Mtok][3d] */
    if (P->run(P->self, Mtok, d, 3 * d, src, m->Wqkv[l], s->qkv) != 0) goto done;
    h_add_bias(Mtok, 3 * d, s->qkv, m->bqkv[l]);
    fill_ij(S, s->ip, s->jp);

    /* per window, per head: attention with rel-pos; scatter into cc (window-major). */
    head_scratch hs;
    if (!alloc_head_scratch(&hs, g, win, dh)) { free_head_scratch(&hs); goto done; }
    _Float16 *cc = windowed ? malloc((size_t)Mtok * d * sizeof(_Float16)) : s->ctxf;
    if (!cc) { free_head_scratch(&hs); goto done; }
    for (int w = 0; w < nWin; w++)
        for (int h = 0; h < nh; h++) {
            gather_head(&hs, s->qkv, w, h, S2, d, dh);
            if (attn_one_head(P, &hs, m->Rh[l], m->Rw[l], S, dh, scale, s->ip, s->jp) != 0) {
                if (windowed) free(cc);
                free_head_scratch(&hs);
                goto done;
            }
            for (int t = 0; t < S2; t++)
                memcpy(cc + (size_t)(w * S2 + t) * d + h * dh, hs.outh + (size_t)t * dh,
                       (size_t)dh * sizeof(_Float16));
        }
    free_head_scratch(&hs);

    /* un-partition (windowed) then o-proj at M=grid*grid */
    _Float16 *ctxf = windowed ? s->ctxf : cc;
    if (windowed) { window_unpartition(cc, g, win, d, ctxf); free(cc); }
    if (P->run(P->self, gg, d, d, ctxf, m->Wo[l], y) != 0) goto done;
    h_add_bias(gg, d, y, m->bo[l]);
    h_residual((size_t)gg * d, x, y);

    /* --- FFN: x += fc2( gelu( fc1( LN2(x) ) ) ) --- */
    h_layernorm(gg, d, x, m->ln2_g[l], m->ln2_b[l], m->eps, ln);
    if (P->run(P->self, gg, d, dff, ln, m->Wf1[l], y) != 0) goto done;
    h_add_bias(gg, dff, y, m->bf1[l]);
    h_gelu_erf((size_t)gg * dff, y);
    if (P->run(P->self, gg, dff, d, y, m->Wf2[l], ln) != 0) goto done;   /* reuse ln as fc2 out */
    h_add_bias(gg, d, ln, m->bf2[l]);
    h_residual((size_t)gg * d, x, ln);
    rc = 0;
done:
    free(ln); free(y);
    return rc;
}

int rocket_sam_encode(int fd, const rocket_sam_model *m,
                      const _Float16 *pixels_chw, _Float16 *out, _Float16 *hidden_opt)
{
    if (!m || !pixels_chw || !out) return -1;
    const int d = m->d, gg = m->grid * m->grid, pdim = m->patch_dim;
    const size_t Gd = (size_t)gg * d;
    int rc = -2;

    os_mm om = { fd };
    mm_provider P = { &om, os_run };
    attn_scratch s;
    if (!alloc_scratch(&s, m->grid, m->win, d)) { free_scratch(&s); return -3; }

    _Float16 *patches = malloc((size_t)gg * pdim * sizeof(_Float16));
    _Float16 *x       = malloc(Gd * sizeof(_Float16));
    if (!patches || !x) goto done;

    /* stem: im2col -> patch proj -> + patch_b + pos */
    im2col(m, pixels_chw, patches);
    if (P.run(P.self, gg, pdim, d, patches, m->patch_W, x) != 0) goto done;
    for (int p = 0; p < gg; p++) {
        _Float16 *xr = x + (size_t)p * d;
        const _Float16 *pr = m->pos + (size_t)p * d;
        for (int j = 0; j < d; j++) xr[j] = (_Float16)((float)xr[j] + (float)m->patch_b[j] + (float)pr[j]);
    }
    if (hidden_opt) memcpy(hidden_opt, x, Gd * sizeof(_Float16));

    for (int l = 0; l < m->n_layers; l++) {
        if ((rc = sam_layer(&P, m, l, x, &s)) != 0) goto done;
        if (hidden_opt) memcpy(hidden_opt + (size_t)(l + 1) * Gd, x, Gd * sizeof(_Float16));
    }

    rc = sam_neck(fd, &P, m, x, out);
done:
    free(patches); free(x); free_scratch(&s);
    return rc;
}

/* ============================================================================
 * SECTION — Resident (prepacked, multicore) path
 * ==========================================================================*/

#define SAM_MAX_WORKERS 8

struct rocket_sam_ctx {
    const rocket_sam_model *m;
    rocket_ctx     *cw;        /* resident static weights (patch, per-layer proj, neck c1) */
    int             aux_fd;    /* neck conv3x3 (+ stream fallback)                          */
    /* per-worker computed-matmul streams + per-head scratch, for the head-parallel attention.
     * The heads are independent, so each worker owns a stream (its own fd -> its own NPU core)
     * and a head_scratch, and processes a disjoint set of (window,head) tasks. */
    int             nworkers;
    rocket_stream  *strm[SAM_MAX_WORKERS];
    head_scratch    hs[SAM_MAX_WORKERS];
    rocket_weights *w_patch, *w_neck_c1;
    rocket_weights *wqkv[ROCKET_SAM_MAX_LAYERS], *wo[ROCKET_SAM_MAX_LAYERS];
    rocket_weights *wf1[ROCKET_SAM_MAX_LAYERS],  *wf2[ROCKET_SAM_MAX_LAYERS];
};

/* per-worker computed-matmul provider (bound to one stream). No aux-fd fallback: the workers run
 * concurrently, so a shared-fd fallback would race; the fixed QK/PV shape set never exhausts the
 * stream's shape cache, so a stream error is a real failure and propagates. */
typedef struct { rocket_stream *s; int aux_fd; } wk_prov;
static int wk_run(void *self, int M, int K, int N, const _Float16 *A, const _Float16 *B, _Float16 *C)
{
    wk_prov *w = self;
    return rocket_matmul_fp16_stream(w->s, M, K, N, A, B, C);
}

rocket_sam_ctx *rocket_sam_ctx_create(const rocket_sam_model *m, int nthreads)
{
    if (!m) return NULL;
    rocket_sam_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->m = m;
    const int d = m->d, dff = m->d_ff, no = m->neck_out, gg = m->grid * m->grid, pdim = m->patch_dim;
    const int gp = ((m->grid + m->win - 1) / m->win) * m->win;
    const int Mwin = (gp / m->win) * (gp / m->win) * m->win * m->win;   /* windowed qkv M */
    c->nworkers = nthreads < 1 ? 1 : (nthreads > SAM_MAX_WORKERS ? SAM_MAX_WORKERS : nthreads);
    c->cw = rocket_ctx_create(nthreads);
    c->aux_fd = rocket_open();
    if (!c->cw || c->aux_fd < 0) goto fail;
    for (int w = 0; w < c->nworkers; w++) {
        c->strm[w] = rocket_stream_create(1);   /* one core per worker (heads run in parallel) */
        if (!c->strm[w] || !alloc_head_scratch(&c->hs[w], m->grid, m->win, m->dhead)) goto fail;
    }
    /* pack the static weights once (M-independent for M>=256, so windowed 4900 + global 4096 share) */
    c->w_patch   = rocket_weights_pack(c->cw, gg, pdim, d, m->patch_W);
    c->w_neck_c1 = rocket_weights_pack(c->cw, gg, d, no, m->neck_c1);
    if (!c->w_patch || !c->w_neck_c1) goto fail;
    for (int l = 0; l < m->n_layers; l++) {
        int Mqkv = m->windowed[l] ? Mwin : gg;
        c->wqkv[l] = rocket_weights_pack(c->cw, Mqkv, d,   3 * d, m->Wqkv[l]);
        c->wo[l]   = rocket_weights_pack(c->cw, gg,   d,   d,     m->Wo[l]);
        c->wf1[l]  = rocket_weights_pack(c->cw, gg,   d,   dff,   m->Wf1[l]);
        c->wf2[l]  = rocket_weights_pack(c->cw, gg,   dff, d,     m->Wf2[l]);
        if (!c->wqkv[l] || !c->wo[l] || !c->wf1[l] || !c->wf2[l]) goto fail;
    }
    return c;
fail:
    rocket_sam_ctx_free(c);
    return NULL;
}

void rocket_sam_ctx_free(rocket_sam_ctx *c)
{
    if (!c) return;
    if (c->cw) {
        if (c->w_patch)   rocket_weights_free(c->cw, c->w_patch);
        if (c->w_neck_c1) rocket_weights_free(c->cw, c->w_neck_c1);
        for (int l = 0; l < c->m->n_layers; l++) {
            if (c->wqkv[l]) rocket_weights_free(c->cw, c->wqkv[l]);
            if (c->wo[l])   rocket_weights_free(c->cw, c->wo[l]);
            if (c->wf1[l])  rocket_weights_free(c->cw, c->wf1[l]);
            if (c->wf2[l])  rocket_weights_free(c->cw, c->wf2[l]);
        }
    }
    for (int w = 0; w < c->nworkers; w++) {
        if (c->strm[w]) rocket_stream_free(c->strm[w]);
        free_head_scratch(&c->hs[w]);
    }
    if (c->cw) rocket_ctx_free(c->cw);
    if (c->aux_fd >= 0) rocket_close(c->aux_fd);
    free(c);
}

/* one worker's slice of a layer's (window,head) attention tasks. Reads the shared qkv, writes its
 * own (window,head) slices of cc (disjoint), using its own stream + head scratch -> no locking. */
typedef struct {
    rocket_sam_ctx *c; int tid, lo, hi;
    const _Float16 *qkv; _Float16 *cc;
    const _Float16 *Rh, *Rw;
    int S, S2, d, dh, nh; float scale;
    const int *ip, *jp; int rc;
} sam_attn_arg;

static void *sam_attn_worker(void *p)
{
    sam_attn_arg *a = p;
    rocket_pin_worker(a->tid);
    wk_prov wp = { a->c->strm[a->tid], a->c->aux_fd };
    mm_provider P = { &wp, wk_run };
    head_scratch *hs = &a->c->hs[a->tid];
    a->rc = 0;
    for (int task = a->lo; task < a->hi; task++) {
        int w = task / a->nh, h = task % a->nh;
        gather_head(hs, a->qkv, w, h, a->S2, a->d, a->dh);
        if (attn_one_head(&P, hs, a->Rh, a->Rw, a->S, a->dh, a->scale, a->ip, a->jp) != 0) { a->rc = -1; break; }
        for (int t = 0; t < a->S2; t++)
            memcpy(a->cc + (size_t)(w * a->S2 + t) * a->d + h * a->dh,
                   hs->outh + (size_t)t * a->dh, (size_t)a->dh * sizeof(_Float16));
    }
    return NULL;
}

/* Resident encode: same datapath as the one-shot, but the projections run prepacked (one weight
 * handle per role) and the per-head QK/PV + rel-pos/softmax fan out across the worker streams. */
static int ctx_layer(rocket_sam_ctx *c, int l, _Float16 *x, attn_scratch *s)
{
    const rocket_sam_model *m = c->m;
    const int d = m->d, dff = m->d_ff, nh = m->n_head, dh = m->dhead, g = m->grid, gg = g * g;
    const int win = m->win, S = m->Sl[l], S2 = S * S, windowed = m->windowed[l];
    const float scale = 1.f / sqrtf((float)dh);
    int rc = -1;

    _Float16 *ln = malloc((size_t)gg * d * sizeof(_Float16));
    _Float16 *y  = malloc((size_t)gg * (dff > d ? dff : d) * sizeof(_Float16));
    if (!ln || !y) { free(ln); free(y); return -1; }

    h_layernorm(gg, d, x, m->ln1_g[l], m->ln1_b[l], m->eps, ln);
    int nWin, Mtok; _Float16 *psrc;
    if (windowed) { nWin = window_partition(ln, g, win, d, s->hw); Mtok = nWin * S2; psrc = s->hw; }
    else { nWin = 1; Mtok = gg; psrc = ln; }
    if (rocket_matmul_fp16_prepacked(c->cw, Mtok, d, 3 * d, psrc, s->qkv, c->wqkv[l]) != 0) goto done;
    h_add_bias(Mtok, 3 * d, s->qkv, m->bqkv[l]);
    fill_ij(S, s->ip, s->jp);

    _Float16 *cc = windowed ? malloc((size_t)Mtok * d * sizeof(_Float16)) : s->ctxf;
    if (!cc) goto done;
    /* fan the (window,head) tasks across the worker streams (heads independent) */
    {
        const int ntask = nWin * nh;
        int nw = c->nworkers; if (nw > ntask) nw = ntask;
        pthread_t th[SAM_MAX_WORKERS]; sam_attn_arg wa[SAM_MAX_WORKERS]; int started[SAM_MAX_WORKERS] = {0};
        const int per = (ntask + nw - 1) / nw;
        for (int w = 0; w < nw; w++) {
            int lo = w * per, hi = lo + per; if (lo >= ntask) { nw = w; break; } if (hi > ntask) hi = ntask;
            wa[w] = (sam_attn_arg){ c, w, lo, hi, s->qkv, cc, m->Rh[l], m->Rw[l], S, S2, d, dh, nh, scale,
                                    s->ip, s->jp, 0 };
            if (nw > 1 && pthread_create(&th[w], NULL, sam_attn_worker, &wa[w]) == 0) started[w] = 1;
            else sam_attn_worker(&wa[w]);   /* nw==1 or spawn failed: run inline */
        }
        int aok = 1;
        for (int w = 0; w < nw; w++) { if (started[w]) pthread_join(th[w], NULL); if (wa[w].rc) aok = 0; }
        if (!aok) { if (windowed) free(cc); goto done; }
    }
    _Float16 *ctxf = windowed ? s->ctxf : cc;
    if (windowed) { window_unpartition(cc, g, win, d, ctxf); free(cc); }
    if (rocket_matmul_fp16_prepacked(c->cw, gg, d, d, ctxf, y, c->wo[l]) != 0) goto done;
    h_add_bias(gg, d, y, m->bo[l]);
    h_residual((size_t)gg * d, x, y);

    h_layernorm(gg, d, x, m->ln2_g[l], m->ln2_b[l], m->eps, ln);
    if (rocket_matmul_fp16_prepacked(c->cw, gg, d, dff, ln, y, c->wf1[l]) != 0) goto done;
    h_add_bias(gg, dff, y, m->bf1[l]);
    h_gelu_erf((size_t)gg * dff, y);
    if (rocket_matmul_fp16_prepacked(c->cw, gg, dff, d, y, ln, c->wf2[l]) != 0) goto done;
    h_add_bias(gg, d, ln, m->bf2[l]);
    h_residual((size_t)gg * d, x, ln);
    rc = 0;
done:
    free(ln); free(y);
    return rc;
}

int rocket_sam_encode_ctx(rocket_sam_ctx *c, const _Float16 *pixels_chw,
                          _Float16 *out, _Float16 *hidden_opt)
{
    if (!c || !pixels_chw || !out) return -1;
    const rocket_sam_model *m = c->m;
    const int d = m->d, gg = m->grid * m->grid, pdim = m->patch_dim, no = m->neck_out, g = m->grid;
    const size_t Gd = (size_t)gg * d;
    int rc = -2;

    attn_scratch s;
    if (!alloc_scratch(&s, m->grid, m->win, d)) { free_scratch(&s); return -3; }
    _Float16 *patches = malloc((size_t)gg * pdim * sizeof(_Float16));
    _Float16 *x       = malloc(Gd * sizeof(_Float16));
    /* neck scratch */
    _Float16 *y1  = malloc((size_t)gg * no * sizeof(_Float16));
    _Float16 *ncw = malloc((size_t)no * gg * sizeof(_Float16));
    _Float16 *nln = malloc((size_t)no * gg * sizeof(_Float16));
    _Float16 *c2  = malloc((size_t)no * gg * sizeof(_Float16));
    if (!patches || !x || !y1 || !ncw || !nln || !c2) goto done;

    im2col(m, pixels_chw, patches);
    if (rocket_matmul_fp16_prepacked(c->cw, gg, pdim, d, patches, x, c->w_patch) != 0) goto done;
    for (int p = 0; p < gg; p++) {
        _Float16 *xr = x + (size_t)p * d; const _Float16 *pr = m->pos + (size_t)p * d;
        for (int j = 0; j < d; j++) xr[j] = (_Float16)((float)xr[j] + (float)m->patch_b[j] + (float)pr[j]);
    }
    if (hidden_opt) memcpy(hidden_opt, x, Gd * sizeof(_Float16));
    for (int l = 0; l < m->n_layers; l++) {
        if ((rc = ctx_layer(c, l, x, &s)) != 0) goto done;
        if (hidden_opt) memcpy(hidden_opt + (size_t)(l + 1) * Gd, x, Gd * sizeof(_Float16));
    }
    /* neck: conv1 prepacked, conv3x3 on aux fd */
    if (rocket_matmul_fp16_prepacked(c->cw, gg, d, no, x, y1, c->w_neck_c1) != 0) goto done;
    for (int p = 0; p < gg; p++) for (int cc = 0; cc < no; cc++) ncw[(size_t)cc * gg + p] = y1[(size_t)p * no + cc];
    h_layernorm2d(no, gg, ncw, m->neck_ln1_g, m->neck_ln1_b, 1e-6f, nln);
    rocket_conv2d_desc dc = { .ic = no, .ih = g, .iw = g, .oc = no, .kh = 3, .kw = 3,
                              .stride_y = 1, .stride_x = 1, .pad_top = 1, .pad_left = 1,
                              .dil_y = 1, .dil_x = 1, .depthwise = 0 };
    if (rocket_conv2d_fp16(c->aux_fd, &dc, nln, m->neck_c2, c2) != 0) goto done;
    h_layernorm2d(no, gg, c2, m->neck_ln2_g, m->neck_ln2_b, 1e-6f, out);
    rc = 0;
done:
    free(patches); free(x); free(y1); free(ncw); free(nln); free(c2); free_scratch(&s);
    return rc;
}
