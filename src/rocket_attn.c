// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_attn.c — multi-head self-attention on the NPU, composed from the validated matmul +
 * the new on-NPU softmax. See rocket_attn.h for the math and the NPU/host work split.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include "rocket_attn.h"
#include "rocket_npu.h"       /* rocket_open / rocket_close (the mt worker fds)  */
#include "rocket_matmul.h"    /* rocket_matmul_fp16 (C = A·B^T) */
#include "rocket_softmax.h"   /* rocket_softmax_fp16 (row-wise) */
#include "rocket_affinity.h"  /* rocket_pin_worker (keep mt workers off the A55s) */

/* ############################################################################
 * PART 1 — Encoder multi-head self-attention (q/k/v + on-NPU/host softmax)
 * ##########################################################################*/

/* Optional host-side softmax (ROCKET_ATTN_HOST_SOFTMAX=1). The attention scores are already
 * host-resident after the scores matmul, so a host softmax skips the three NPU jobs the on-NPU
 * softmax submits (exp / row-sum / scale) and their per-call host<->NPU transfers of the [Tp,Tn]
 * score matrix — which DOMINATE the on-NPU softmax (it is transfer-bound, not exp-compute-bound:
 * a host expf softmax measured ~3x faster than the NPU LUT path on a Whisper-base encoder). Default
 * off, so attention stays fully on the NPU; a fully on-NPU RESIDENT softmax (scores kept in a BO,
 * on-NPU row-max) would remove the transfers without the CPU cost. */
static _Atomic int g_attn_host_sm = -1;
static void host_softmax_rows(int M, int N, const _Float16 *in, _Float16 *out){
    for (int m = 0; m < M; m++) {
        const _Float16 *xp = in + (size_t)m*N; _Float16 *op = out + (size_t)m*N;
        float mx = -INFINITY; for (int n=0;n<N;n++){ float v=(float)xp[n]; if (v>mx) mx=v; }
        float s = 0.f; for (int n=0;n<N;n++){ float e=expf((float)xp[n]-mx); op[n]=(_Float16)e; s+=e; }
        float inv = s>0.f ? 1.f/s : 0.f; for (int n=0;n<N;n++) op[n]=(_Float16)((float)op[n]*inv);
    }
}

/* ---- CPU fp64 reference (the golden oracle) ----------------------------------- */
void rocket_mha_self_ref_fp16(int T, int d, int n_head, const _Float16 *x,
                              const _Float16 *Wq, const _Float16 *bq,
                              const _Float16 *Wk, const _Float16 *bk,
                              const _Float16 *Wv, const _Float16 *bv,
                              const _Float16 *Wo, const _Float16 *bo, _Float16 *out)
{
    const int dh = d / n_head;
    const double scale = 1.0 / sqrt((double)dh);
    double *q = malloc((size_t)T*d*sizeof(double));
    double *k = malloc((size_t)T*d*sizeof(double));
    double *v = malloc((size_t)T*d*sizeof(double));
    double *ctx = malloc((size_t)T*d*sizeof(double));
    double *sc  = malloc((size_t)T*sizeof(double));
    /* projections q/k/v = x·W^T + b */
    for (int t = 0; t < T; t++) for (int o = 0; o < d; o++) {
        double aq=0, ak=0, av=0; const _Float16 *xr = x + (size_t)t*d;
        const _Float16 *wq=Wq+(size_t)o*d,*wk=Wk+(size_t)o*d,*wv=Wv+(size_t)o*d;
        for (int i = 0; i < d; i++) { double xi=(double)xr[i]; aq+=xi*(double)wq[i]; ak+=xi*(double)wk[i]; av+=xi*(double)wv[i]; }
        q[(size_t)t*d+o]=aq+(bq?(double)bq[o]:0); k[(size_t)t*d+o]=ak+(bk?(double)bk[o]:0); v[(size_t)t*d+o]=av+(bv?(double)bv[o]:0);
    }
    for (int h = 0; h < n_head; h++) {
        int off = h*dh;
        for (int i = 0; i < T; i++) {
            double mx = -INFINITY;
            for (int j = 0; j < T; j++) {
                double s = 0; for (int c = 0; c < dh; c++) s += q[(size_t)i*d+off+c]*k[(size_t)j*d+off+c];
                s *= scale; sc[j] = s; if (s > mx) mx = s;
            }
            double sum = 0; for (int j = 0; j < T; j++){ sc[j]=exp(sc[j]-mx); sum+=sc[j]; }
            for (int c = 0; c < dh; c++) {
                double a = 0; for (int j = 0; j < T; j++) a += sc[j]*v[(size_t)j*d+off+c];
                ctx[(size_t)i*d+off+c] = a/sum;
            }
        }
    }
    /* out = ctx·Wo^T + bo */
    for (int t = 0; t < T; t++) for (int o = 0; o < d; o++) {
        double a=0; const double *cr = ctx+(size_t)t*d; const _Float16 *wo=Wo+(size_t)o*d;
        for (int i = 0; i < d; i++) a += cr[i]*(double)wo[i];
        out[(size_t)t*d+o] = (_Float16)(a + (bo?(double)bo[o]:0));
    }
    free(q);free(k);free(v);free(ctx);free(sc);
}

/* add bias b[N] (broadcast over the M rows) into C[M*N], in place, on the host (O(MN) glue). */
static void add_bias(_Float16 *C, const _Float16 *b, int M, int N)
{
    if (!b) return;
    for (int m = 0; m < M; m++) { _Float16 *r = C+(size_t)m*N; for (int n=0;n<N;n++) r[n]=(_Float16)((float)r[n]+(float)b[n]); }
}

int rocket_mha_self_fp16(int fd, int T, int d, int n_head, const _Float16 *x,
                         const _Float16 *Wq, const _Float16 *bq,
                         const _Float16 *Wk, const _Float16 *bk,
                         const _Float16 *Wv, const _Float16 *bv,
                         const _Float16 *Wo, const _Float16 *bo, _Float16 *out)
{
    if (T < 1 || d < 1 || n_head < 1 || d % n_head) return -1;
    if (fd < 0) { rocket_mha_self_ref_fp16(T,d,n_head,x,Wq,bq,Wk,bk,Wv,bv,Wo,bo,out); return 0; }

    const int dh = d / n_head;
    const int Tp = (T + 3) & ~3;                 /* matmul M%4 (query rows)            */
    const int Tn = (T + 31) & ~31;               /* matmul N/K align (key count)       */
    const float scale = 1.f / sqrtf((float)dh);
    int rc = -2;

    _Float16 *xp = NULL, *q=NULL,*k=NULL,*v=NULL,*ctx=NULL;
    _Float16 *qh=NULL,*kh=NULL,*vhT=NULL,*sc=NULL,*P=NULL;

    /* zero-pad x rows to Tp (pad rows -> garbage q/k/v rows, never read into the output) */
    xp = calloc((size_t)Tp*d, sizeof(_Float16));
    q  = malloc((size_t)Tp*d*sizeof(_Float16));
    k  = malloc((size_t)Tp*d*sizeof(_Float16));
    v  = malloc((size_t)Tp*d*sizeof(_Float16));
    ctx= calloc((size_t)Tp*d, sizeof(_Float16));
    qh = calloc((size_t)Tp*dh, sizeof(_Float16));   /* zero pad rows T..Tp (copied wholesale below) */
    kh = calloc((size_t)Tn*dh, sizeof(_Float16));   /* keys padded to Tn rows (zero pad) */
    vhT= calloc((size_t)dh*Tn, sizeof(_Float16));   /* v^T padded to Tn cols (zero pad)  */
    sc = malloc((size_t)Tp*Tn*sizeof(_Float16));
    P  = malloc((size_t)Tp*Tn*sizeof(_Float16));
    if (!xp||!q||!k||!v||!ctx||!qh||!kh||!vhT||!sc||!P) goto out;
    memcpy(xp, x, (size_t)T*d*sizeof(_Float16));

    /* 1. q/k/v = x·W^T (+bias). M=Tp, K=d, N=d. */
    if ((rc = rocket_matmul_fp16(fd, Tp, d, d, xp, Wq, q)) != 0) goto out;
    if ((rc = rocket_matmul_fp16(fd, Tp, d, d, xp, Wk, k)) != 0) goto out;
    if ((rc = rocket_matmul_fp16(fd, Tp, d, d, xp, Wv, v)) != 0) goto out;
    add_bias(q, bq, T, d); add_bias(k, bk, T, d); add_bias(v, bv, T, d);

    /* 2. per head: scores = scale·(q_h·k_h^T) -> softmax -> ctx_h = P·v_h.
     *    The key count T is padded to Tn (matmul N/K alignment); the pad key columns get a
     *    score of 0 from the zero keys, which we MASK to a large negative before softmax so
     *    they contribute ~0 (exp underflow) — and the zero pad v rows add nothing to ctx. */
    if (g_attn_host_sm < 0) g_attn_host_sm = getenv("ROCKET_ATTN_HOST_SOFTMAX") ? 1 : 0;
    for (int h = 0; h < n_head; h++) {
        int off = h*dh;
        for (int t = 0; t < Tp; t++) memcpy(qh+(size_t)t*dh, q+(size_t)t*d+off, dh*sizeof(_Float16));
        for (int t = 0; t < T;  t++) memcpy(kh+(size_t)t*dh, k+(size_t)t*d+off, dh*sizeof(_Float16));
        /* v_h^T [dh,Tn] so the ctx matmul C=P·(v_h^T)^T = P·v_h (pad key cols stay 0) */
        for (int t = 0; t < T; t++) for (int c = 0; c < dh; c++)
            vhT[(size_t)c*Tn+t] = v[(size_t)t*d+off+c];

        /* scores[Tp,Tn] = qh·kh^T  (M=Tp,K=dh,N=Tn) */
        if ((rc = rocket_matmul_fp16(fd, Tp, dh, Tn, qh, kh, sc)) != 0) goto out;
        for (int i = 0; i < Tp; i++) {
            _Float16 *row = sc + (size_t)i*Tn;
            for (int j = 0; j < T;  j++) row[j] = (_Float16)((float)row[j]*scale);
            for (int j = T; j < Tn; j++) row[j] = (_Float16)(-30000.f);  /* mask pad keys */
        }

        /* P = softmax(scores) row-wise over the Tn columns; pad columns -> ~0. On-NPU by default;
         * host softmax when ROCKET_ATTN_HOST_SOFTMAX=1 (the on-NPU softmax is transfer-bound). */
        if (g_attn_host_sm) host_softmax_rows(Tp, Tn, sc, P);
        else if ((rc = rocket_softmax_fp16(fd, Tp, Tn, sc, P)) != 0) goto out;

        /* ctx_h[Tp,dh] = P·v_h  (M=Tp,K=Tn,N=dh) -> scatter into ctx columns [off,off+dh) */
        if ((rc = rocket_matmul_fp16(fd, Tp, Tn, dh, P, vhT, qh)) != 0) goto out;  /* reuse qh */
        for (int t = 0; t < T; t++) memcpy(ctx+(size_t)t*d+off, qh+(size_t)t*dh, dh*sizeof(_Float16));
    }

    /* 3. out = ctx·Wo^T (+bo).  M=Tp,K=d,N=d; take the first T rows. */
    if ((rc = rocket_matmul_fp16(fd, Tp, d, d, ctx, Wo, q)) != 0) goto out;   /* reuse q as out buf */
    add_bias(q, bo, T, d);
    memcpy(out, q, (size_t)T*d*sizeof(_Float16));
    rc = 0;

out:
    free(P);free(sc);free(vhT);free(kh);free(qh);free(ctx);free(v);free(k);free(q);free(xp);
    return rc;
}

/* ############################################################################
 * PART 2 — Masked GQA (flash) attention: reference, knobs, scratch
 * ##########################################################################*/

/* ---- masked grouped-query (flash) attention -------------------------------------
 * Decoder attention given already-projected Q/K/V + an additive mask. See rocket_attn.h
 * for the layouts and math. Score = softcap·tanh(scale·QKᵀ/softcap) + mask (softcap==0
 * skips the tanh). The pre-softmax masked score clamps at -30000 so a masked / pad key
 * underflows to ~0 in fp16. */
static float fa_score(float qk, float scale, float softcap, float m)
{
    float s = qk * scale;
    if (softcap != 0.0f) s = softcap * tanhf(s / softcap);
    s += m;
    return s < -30000.0f ? -30000.0f : s;
}

void rocket_flash_attn_ref_fp16(int n_tokens, int n_kv, int head_dim, int dv,
                                int n_head, int n_kv_heads, float scale, float softcap,
                                const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                                const _Float16 *mask, _Float16 *out)
{
    const int gqa = n_head / n_kv_heads;
    double *sc = malloc((size_t)n_kv * sizeof(double));
    if (!sc) return;
    for (int h = 0; h < n_head; h++) {
        const int hk = h / gqa;
        const _Float16 *Qh = Q + (size_t)h  * n_tokens * head_dim;   /* [n_tokens,head_dim] (dk) */
        const _Float16 *Kh = K + (size_t)hk * n_kv     * head_dim;   /* [n_kv,head_dim]     (dk) */
        const _Float16 *Vh = V + (size_t)hk * dv       * n_kv;       /* [dv,n_kv]               */
        _Float16       *Oh = out + (size_t)h * n_tokens * dv;
        for (int t = 0; t < n_tokens; t++) {
            const _Float16 *qr = Qh + (size_t)t * head_dim;
            const _Float16 *mr = mask ? mask + (size_t)t * n_kv : NULL;
            double mx = -INFINITY;
            for (int j = 0; j < n_kv; j++) {
                double qk = 0; const _Float16 *kr = Kh + (size_t)j * head_dim;
                for (int c = 0; c < head_dim; c++) qk += (double)qr[c] * (double)kr[c];
                double s = fa_score((float)qk, scale, softcap, mr ? (float)mr[j] : 0.0f);
                sc[j] = s; if (s > mx) mx = s;
            }
            double sum = 0; for (int j = 0; j < n_kv; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
            const double inv = sum > 0 ? 1.0 / sum : 0.0;
            for (int c = 0; c < dv; c++) {
                double a = 0; for (int j = 0; j < n_kv; j++) a += sc[j] * (double)Vh[(size_t)c * n_kv + j];
                Oh[(size_t)t * dv + c] = (_Float16)(a * inv);
            }
        }
    }
    free(sc);
}

/* Flash-attn softmax placement. Unlike the encoder MHA path (g_attn_host_sm, on-NPU by
 * default), the FA primitive ALWAYS brings the scores to the host for the mandatory additive
 * mask + scale + soft-cap, so the on-NPU softmax is a pure round-trip here (upload scores, 3
 * submits, download P, then re-upload P for the AV matmul) — measured ~5-10% slower than a
 * host pass, the gap growing with n_kv. So the FA default is HOST softmax;
 * ROCKET_ATTN_HOST_SOFTMAX=0 forces the on-NPU path for comparison. */
static int fa_host_softmax(void)
{
    static _Atomic int v = -1;
    if (v < 0) { const char *e = getenv("ROCKET_ATTN_HOST_SOFTMAX"); v = e ? atoi(e) : 1; }
    return v;
}

/* Per-worker QK/AV submit chaining (ROCKET_FA_CHAIN, default ON). A worker's per-head
 * QK matmuls all share one (Tp,dh,Kn) shape (and its AV matmuls one (Tp,Kn,dh)), so each
 * set batches into a SINGLE NPU job via the worker's resident rocket_mm_batch context —
 * one submit + one fence wait for the whole head range instead of one per head, and
 * (with ROCKET_BATCH_SUBMIT=1 + the kernel half) one completion IRQ instead of one per
 * head. This attacks the small-GEMM dispatch floor that makes short/mid-context FA-NPU
 * lose to CPU: per-head attention GEMMs are tiny and dispatch-bound, so collapsing the
 * per-head submit+wait serialization is the lever. With the persistent batched context
 * (BOs + score scratch held resident, prezeroed once) it nearly doubles the FA-op
 * throughput vs the per-call batch, moving the FA-NPU-vs-CPU prefill crossover from ~6K
 * down to ~2K (parity <=1K) — so it defaults ON. The softmax + additive mask sit BETWEEN
 * the batched QK and AV jobs (not inside a chained QK->softmax->AV), so this is compatible
 * with the host-softmax default — it never forces the softmax back on-NPU. Bit-identical
 * to the per-head path; ROCKET_FA_CHAIN=0 forces the per-head path back for comparison. */
static int fa_chain(void)
{
    static _Atomic int v = -1;
    if (v < 0) { const char *e = getenv("ROCKET_FA_CHAIN"); v = e ? atoi(e) : 1; }
    return v;
}

/* Max score-matrix elements (Tp*Kn padded) per chained head group. Bounds the batched group
 * scratch (G heads' sc/P/k/v tiles held at once): Gmax = budget / (Tp*Kn), capped at the heads
 * in the worker's range, so the scratch stays bounded as context grows (the group shrinks
 * toward 1 at long context).
 *
 * Collapsing the per-head QK/AV submits PAYS at long context, contrary to an earlier guess that
 * each head's GEMM already fills a submit batch there: on this dispatch-bound NPU, batching a
 * worker's whole head range into one QK + one AV job (vs one submit+fence per head) measured
 * 1.10x at 4K, 1.47x at 8K, 1.32x at 16K, 1.16x at 32K at a 512-token ubatch [HW sweep
 * 2026-06-29], with no short-context change (already batched at <=2K). So the default is 32M
 * elems (not 4M): it batches a worker's ~3-head range up to ~20K context, bounding the batched
 * score scratch to ~150-200 MB/worker. Above that (or at a 2048-token ubatch, where each head's
 * score alone exceeds the budget and the win shrinks to ~1.05x anyway) it falls back to per-head
 * — raise ROCKET_FA_CHAIN_ELEMS (e.g. 64M) to chain further, at a higher resident-scratch cost. */
static long fa_chain_elems(void)
{
    static _Atomic long v = -1;
    if (v < 0) { const char *e = getenv("ROCKET_FA_CHAIN_ELEMS"); v = e ? atol(e) : (32L << 20); if (v < 1) v = 1; }
    return v;
}

/* Online/tiled long-context flash attention (ROCKET_FA_TILE_KV). The materialized path
 * computes scores over the FULL key axis in one QK matmul, holding the [Tp,Kn] score matrix
 * (32 MB/head at 32K) host-side for the mask+softmax — so the score traffic streams to DRAM
 * and back (NPU write -> host mask read -> host softmax write -> NPU AV read) and grows with
 * n_kv. The tiled path walks the key axis in blocks of width Kbp, carrying the FlashAttention-2
 * running softmax state (max m[Tp], denom l[Tp], unnormalized output acc[Tp,dh], all fp32), so
 * the working score tile is only [Tp,Kbp] and stays cache-resident across that read/modify/read
 * cycle, and the resident scratch is bounded (no 32-64 MB/worker score allocation). The total
 * MACs and the host exp/row-reduce work are unchanged; the win is cache locality + a flat
 * memory footprint at long context, traded against more (smaller) per-tile NPU submits.
 *
 * fa_tile_kv_width(Kn) returns the padded tile width (a multiple of 32 = the sc/kh/vh/P tile
 * buffer width) when the tiled path should engage, else 0 (use the materialized path). It
 * engages only above ROCKET_FA_TILE_MIN_KV (short/mid context keeps the materialized+chained
 * path, where the small-GEMM dispatch floor — not the score matrix — dominates) and when the
 * tile is a real subdivision (>1 tile). Both fa_scratch_ensure (buffer sizing) and
 * fa_heads_range (dispatch) call it with the SAME padded Kn so they always agree. Read fresh
 * each call (per heads_range, cheap) so a gate/sweep can toggle the knobs mid-process. */
#define ROCKET_FA_TILE_KV_DEFAULT     0      /* 0 = disabled (opt-in until the crossover is set) */
#define ROCKET_FA_TILE_MIN_KV_DEFAULT 8192   /* engage only above this padded key count          */
static int fa_tile_kv_width(int kv_padded)
{
    const char *ek = getenv("ROCKET_FA_TILE_KV");
    int kb = ek ? atoi(ek) : ROCKET_FA_TILE_KV_DEFAULT;
    if (kb <= 0) return 0;
    const char *em = getenv("ROCKET_FA_TILE_MIN_KV");
    int min_kv = em ? atoi(em) : ROCKET_FA_TILE_MIN_KV_DEFAULT;
    if (kv_padded <= min_kv) return 0;
    int kbp = (kb + 31) & ~31;
    if (kbp < 32) kbp = 32;
    if (kbp >= kv_padded) return 0;          /* one tile == full: no subdivision, no benefit */
    return kbp;
}

/* Per-worker host scratch for one head range: the padded matmul tiles + the score/prob
 * matrices. Sized by (Tp,Kn,dh); GROW-ONLY (reallocated up, never shrunk) so a persistent
 * context reuses one allocation across calls instead of mallocing — and at long context the
 * sc/P score matrices (8-16 MB) cross glibc's mmap threshold, so a per-call malloc/free would
 * mmap+fault+munmap every layer. The per-head loop memset-zeros qh/kh/vh before filling, so
 * the buffers need no zero-init on (re)alloc; sc/P are fully written before use. */
typedef struct {
    _Float16 *qh, *kh, *vh, *sc, *P, *ctx;
    size_t c_qh, c_kh, c_vh, c_sc, c_P, c_ctx;   /* allocated element counts */

    /* Chained-batch (ROCKET_FA_CHAIN) group scratch: the host gather buffers for a
     * group of up to Gmax heads (the per-head buffers above hold one head), grown to
     * the largest (Gmax,Tp,Kn,dh) seen, plus the matmul operand-pointer arrays. Held
     * resident so the chained path stops mallocing the 8-16 MB score matrices per call
     * — the same per-call-malloc cost the per-head path avoided. */
    _Float16 *bqh, *bkh, *bsc, *bP, *bvh, *bcx;
    size_t c_bqh, c_bkh, c_bsc, c_bP, c_bvh, c_bcx;
    const _Float16 **pA, **pB;
    _Float16 **pC;
    size_t c_ptr;                                /* allocated pointer-array length     */

    /* Per-worker persistent batched-matmul contexts (one per shape: QK and AV
     * alternate, so a shared context would re-zero every op). NULL on the stateless /
     * single-fd paths, where the chained matmul allocates its BOs per call. */
    rocket_mm_batch *qk, *av;

    /* Online/tiled (long-context) running-softmax accumulators, all fp32 for stability:
     * running max fm[Tp], running denom fl[Tp], unnormalized output facc[Tp*dh]. Small
     * (<=Tp*(dh+2) floats), so grown unconditionally; the materialized path leaves them
     * unread. When the tiled path engages, qh/kh/vh/sc/P/ctx above are sized to ONE key
     * tile [.,Kbp] rather than the full [.,Kn]. */
    float  *fm, *fl, *facc;
    size_t  c_fm, c_fl, c_facc;
} fa_scratch;

static int fa_grow(_Float16 **buf, size_t *cap, size_t need)
{
    if (*cap >= need) return 0;
    _Float16 *n = realloc(*buf, need * sizeof(_Float16));
    if (!n) return -1;
    *buf = n; *cap = need;
    return 0;
}

static int fa_grow_f(float **buf, size_t *cap, size_t need)
{
    if (*cap >= need) return 0;
    float *n = realloc(*buf, need * sizeof(float));
    if (!n) return -1;
    *buf = n; *cap = need;
    return 0;
}

/* Ensure the per-worker scratch for shape (Tp,Kn,dh). The score/key/value tiles (sc/P/kh/vh)
 * are sized to the FULL padded key count Kn for the materialized path, or to ONE key tile of
 * width Kbp for the online/tiled path (fa_tile_kv_width>0) — the whole point of tiling is to
 * never hold the [Tp,Kn] score matrix. fa_heads_range makes the matching dispatch from the
 * same fa_tile_kv_width(Kn), so the buffers it uses are always sized for the path it takes. The
 * fp32 running-softmax accumulators are small and grown unconditionally. */
static int fa_scratch_ensure(fa_scratch *s, int Tp, int Kn, int dk, int dv)
{
    /* Tiling assumes a single head dim; disable it for MLA (dk != dv) so the buffer
     * sizing here matches fa_heads_range's path choice (which also forces materialized). */
    const int Kbp = (dk == dv) ? fa_tile_kv_width(Kn) : 0;
    const int sw  = Kbp ? Kbp : Kn;                  /* score/key/value tile column width */
    const size_t qd = (size_t)Tp * dk, vd = (size_t)Tp * dv, tk = (size_t)Tp * sw;
    const size_t kkd = (size_t)sw * dk, vkd = (size_t)sw * dv;   /* kh: [sw,dk]; vh: [dv,sw] */
    return (fa_grow(&s->qh, &s->c_qh, qd) || fa_grow(&s->kh, &s->c_kh, kkd) ||
            fa_grow(&s->vh, &s->c_vh, vkd) || fa_grow(&s->sc, &s->c_sc, tk) ||
            fa_grow(&s->P,  &s->c_P,  tk) || fa_grow(&s->ctx,&s->c_ctx,vd) ||
            fa_grow_f(&s->fm, &s->c_fm, (size_t)Tp) || fa_grow_f(&s->fl, &s->c_fl, (size_t)Tp) ||
            fa_grow_f(&s->facc, &s->c_facc, vd)) ? -1 : 0;
}

/* Grow the chained-batch group buffers + pointer arrays for `Gmax` heads at (Tp,Kn,dh).
 * The gather steps memset their target region before filling and sc/P/cx are fully
 * written, so the buffers need no zero-init on (re)alloc. Returns 0 / -1. */
static int fa_batch_ensure(fa_scratch *s, int Gmax, int Tp, int Kn, int dh)
{
    const size_t qd = (size_t)Tp * dh, kd = (size_t)Kn * dh, tk = (size_t)Tp * Kn;
    if (fa_grow(&s->bqh, &s->c_bqh, (size_t)Gmax * qd) ||
        fa_grow(&s->bkh, &s->c_bkh, (size_t)Gmax * kd) ||
        fa_grow(&s->bsc, &s->c_bsc, (size_t)Gmax * tk) ||
        fa_grow(&s->bP,  &s->c_bP,  (size_t)Gmax * tk) ||
        fa_grow(&s->bvh, &s->c_bvh, (size_t)Gmax * kd) ||
        fa_grow(&s->bcx, &s->c_bcx, (size_t)Gmax * qd))
        return -1;
    if (s->c_ptr < (size_t)Gmax) {
        const _Float16 **na = realloc(s->pA, (size_t)Gmax * sizeof(*na));
        const _Float16 **nb = realloc(s->pB, (size_t)Gmax * sizeof(*nb));
        _Float16       **nc = realloc(s->pC, (size_t)Gmax * sizeof(*nc));
        if (na) s->pA = na;
        if (nb) s->pB = nb;
        if (nc) s->pC = nc;
        if (!na || !nb || !nc) return -1;
        s->c_ptr = Gmax;
    }
    return 0;
}

static void fa_scratch_free(fa_scratch *s)
{
    rocket_mm_batch_free(s->qk);
    rocket_mm_batch_free(s->av);
    free(s->qh); free(s->kh); free(s->vh); free(s->sc); free(s->P); free(s->ctx);
    free(s->bqh); free(s->bkh); free(s->bsc); free(s->bP); free(s->bvh); free(s->bcx);
    free(s->pA); free(s->pB); free(s->pC);
    free(s->fm); free(s->fl); free(s->facc);
    memset(s, 0, sizeof *s);
}

/* ############################################################################
 * PART 3 — Per-head host glue + head-range compute (batched / tiled / per-head)
 * ##########################################################################*/

/* ---- per-head host glue (shared by the per-head and the chained-batch paths) ----
 * The gather/mask steps are identical whether a head's QK/AV runs as its own submit
 * or as one item of a chained group; factoring them keeps the two paths byte-for-byte
 * the same on these (the flash_attn gate covers both). */

/* gather head h's query rows into the padded matmul tile qh[Tp,dh] (rows T..Tp zero). */
static void fa_gather_q(_Float16 *qh, const _Float16 *Qh, int n_tokens, int Tp, int dh)
{
    memset(qh, 0, (size_t)Tp * dh * sizeof(_Float16));
    for (int t = 0; t < n_tokens; t++)
        memcpy(qh + (size_t)t * dh, Qh + (size_t)t * dh, dh * sizeof(_Float16));
}

/* gather head hk's key rows into kh[Kn,dh] (the QK B-operand; rows n_kv..Kn zero). */
static void fa_gather_k(_Float16 *kh, const _Float16 *Kh, int n_kv, int Kn, int dh)
{
    memset(kh, 0, (size_t)Kn * dh * sizeof(_Float16));
    for (int j = 0; j < n_kv; j++)
        memcpy(kh + (size_t)j * dh, Kh + (size_t)j * dh, dh * sizeof(_Float16));
}

/* gather head hk's values into vh[dh,Kn] = v^T (the AV B-operand; cols n_kv..Kn zero). */
static void fa_gather_v(_Float16 *vh, const _Float16 *Vh, int n_kv, int Kn, int dh)
{
    memset(vh, 0, (size_t)dh * Kn * sizeof(_Float16));
    for (int c = 0; c < dh; c++)
        memcpy(vh + (size_t)c * Kn, Vh + (size_t)c * n_kv, n_kv * sizeof(_Float16));
}

/* scale + soft-cap + additive mask, in place on sc[Tp,Kn]; pad key columns -> -inf. */
static void fa_mask_scores(_Float16 *sc, const _Float16 *mask, int n_tokens, int n_kv,
                           int Tp, int Kn, float scale, float softcap)
{
    for (int i = 0; i < Tp; i++) {
        _Float16 *row = sc + (size_t)i * Kn;
        const _Float16 *mr = (mask && i < n_tokens) ? mask + (size_t)i * n_kv : NULL;
        for (int j = 0; j < n_kv; j++)
            row[j] = (_Float16)fa_score((float)row[j], scale, softcap, mr ? (float)mr[j] : 0.0f);
        for (int j = n_kv; j < Kn; j++) row[j] = (_Float16)(-30000.0f);
    }
}

/* gather kv head hk's value COLUMNS [j0,j0+w) into vh[dh,wp] = v_tile^T (the AV B-operand for
 * one key tile; cols w..wp zero). Vh is [dh,n_kv] with full row stride n_kv. */
static void fa_gather_v_tile(_Float16 *vh, const _Float16 *Vh, int j0, int w, int wp,
                             int dh, int v_stride)
{
    memset(vh, 0, (size_t)dh * wp * sizeof(_Float16));
    for (int c = 0; c < dh; c++)
        memcpy(vh + (size_t)c * wp, Vh + (size_t)c * v_stride + j0, (size_t)w * sizeof(_Float16));
}

/* scale + soft-cap + additive mask for ONE key tile [j0,j0+w) into sc[Tp,wp]; pad columns
 * w..wp -> -inf. mask is the full [n_tokens,n_kv] additive mask (row i, key j0+jj). */
static void fa_mask_scores_tile(_Float16 *sc, const _Float16 *mask, int n_tokens, int n_kv,
                                int j0, int w, int Tp, int wp, float scale, float softcap)
{
    for (int i = 0; i < Tp; i++) {
        _Float16 *row = sc + (size_t)i * wp;
        const _Float16 *mr = (mask && i < n_tokens) ? mask + (size_t)i * n_kv + j0 : NULL;
        for (int jj = 0; jj < w; jj++)
            row[jj] = (_Float16)fa_score((float)row[jj], scale, softcap, mr ? (float)mr[jj] : 0.0f);
        for (int jj = w; jj < wp; jj++) row[jj] = (_Float16)(-30000.0f);
    }
}

/* ---- chained-batch head range (ROCKET_FA_CHAIN=1) -------------------------------
 * Compute heads [h0,h1) on ONE fd like fa_heads_range, but batch the per-head QK
 * matmuls into a SINGLE NPU job (one submit + one fence, one IRQ under the chained
 * kernel) and, after the host softmax, the per-head AV matmuls into a second single
 * job — via rocket_matmul_fp16_batch (all of a worker's QK share one (Tp,dh,Kn), all
 * its AV one (Tp,Kn,dh)). Heads run in groups of <= Gmax (a memory bound; see
 * fa_chain_elems), all group scratch held in one allocation reused across groups.
 * The mask + softmax sit between the two batched jobs, so the host-softmax default is
 * preserved. Bit-identical to fa_heads_range (same gather/mask/softmax/scatter; the
 * batched matmul is bit-identical to per-item rocket_matmul_fp16). Returns 0 / <0. */
static int fa_heads_range_batched(int fd, int n_tokens, int n_kv, int head_dim,
                                  int n_head, int n_kv_heads, float scale, float softcap,
                                  const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                                  const _Float16 *mask, _Float16 *out,
                                  fa_scratch *s, int h0, int h1, int host_sm, int Gmax)
{
    const int gqa = n_head / n_kv_heads, dh = head_dim;
    const int Tp = (n_tokens + 3) & ~3, Kn = (n_kv + 31) & ~31;
    const size_t qd = (size_t)Tp * dh, kd = (size_t)Kn * dh, tk = (size_t)Tp * Kn;

    /* resident group scratch (Gmax heads), reused across groups AND across calls; pA/pB/
     * pC are refilled for QK then AV (qh,kh->sc ; P,vh->cx). s->qk/s->av are the resident
     * batched-matmul contexts (one per shape) on the persistent path; on the stateless /
     * single-fd path they are NULL and the matmul allocates its BOs per call. */
    if (fa_batch_ensure(s, Gmax, Tp, Kn, dh)) return -2;
    _Float16 *qh = s->bqh, *kh = s->bkh, *sc = s->bsc, *P = s->bP, *vh = s->bvh, *cx = s->bcx;
    const _Float16 **pA = s->pA, **pB = s->pB;
    _Float16       **pC = s->pC;
    int rc = 0;

    for (int gh = h0; gh < h1 && !rc; gh += Gmax) {
        const int G = (h1 - gh < Gmax) ? (h1 - gh) : Gmax;

        /* gather the group's QK operands; sc[g] receives qh[g]·kh[g]^T */
        for (int g = 0; g < G; g++) {
            const int h = gh + g, hk = h / gqa;
            fa_gather_q(qh + (size_t)g * qd, Q + (size_t)h  * n_tokens * dh, n_tokens, Tp, dh);
            fa_gather_k(kh + (size_t)g * kd, K + (size_t)hk * n_kv     * dh, n_kv, Kn, dh);
            pA[g] = qh + (size_t)g * qd; pB[g] = kh + (size_t)g * kd; pC[g] = sc + (size_t)g * tk;
        }
        rc = s->qk ? rocket_mm_batch_run(s->qk, Tp, dh, Kn, G, pA, pB, pC)
                   : rocket_matmul_fp16_batch(fd, Tp, dh, Kn, G, pA, pB, pC);
        if (rc) break;

        /* per head: scale+mask -> softmax -> gather V; P[g],vh[g] feed the batched AV */
        for (int g = 0; g < G; g++) {
            const int h = gh + g, hk = h / gqa;
            _Float16 *scg = sc + (size_t)g * tk, *Pg = P + (size_t)g * tk;
            fa_mask_scores(scg, mask, n_tokens, n_kv, Tp, Kn, scale, softcap);
            if (host_sm) host_softmax_rows(Tp, Kn, scg, Pg);
            else if ((rc = rocket_softmax_fp16(fd, Tp, Kn, scg, Pg)) != 0) break;
            fa_gather_v(vh + (size_t)g * kd, V + (size_t)hk * dh * n_kv, n_kv, Kn, dh);
            pA[g] = Pg; pB[g] = vh + (size_t)g * kd; pC[g] = cx + (size_t)g * qd;
        }
        if (rc) break;
        rc = s->av ? rocket_mm_batch_run(s->av, Tp, Kn, dh, G, pA, pB, pC)
                   : rocket_matmul_fp16_batch(fd, Tp, Kn, dh, G, pA, pB, pC);
        if (rc) break;

        /* scatter each head's ctx into its output slice */
        for (int g = 0; g < G; g++) {
            _Float16 *Oh = out + (size_t)(gh + g) * n_tokens * dh;
            const _Float16 *cg = cx + (size_t)g * qd;
            for (int t = 0; t < n_tokens; t++)
                memcpy(Oh + (size_t)t * dh, cg + (size_t)t * dh, dh * sizeof(_Float16));
        }
    }
    return rc;
}

/* ---- online/tiled head range (ROCKET_FA_TILE_KV, long context) ------------------
 * Compute heads [h0,h1) on ONE fd, walking the key axis in tiles of width Kbp and carrying
 * the FlashAttention-2 running softmax (fm/fl/facc, fp32) instead of materializing the full
 * [Tp,Kn] score matrix. Per key tile [j0,j0+w): QK matmul -> tile scores sc[Tp,wp]; scale +
 * soft-cap + mask; per row, fold the tile into the running max/denom and rescale the running
 * output; AV matmul P_tile·V_tile -> add to the running output. Finalize out = acc/l.
 *
 * The softmax always runs HOST-side here: the running max/rescale is inherently sequential
 * across tiles and needs the scores in fp32, so an on-NPU per-tile softmax would be a pure
 * round-trip (this matches the materialized path's host-softmax default). A query row whose
 * every key in a tile is masked is SKIPPED for that tile (its P row zeroed so the batched AV
 * adds nothing and its running state is untouched) — required for causal / sliding-window,
 * where a tile can be wholly outside a row's visible range (then the tile's max is the mask
 * clamp and exp() of it would otherwise wrongly weight the masked keys). Numerically at least
 * as accurate as fa_heads_range (fp32 accumulation; same fp16 NPU matmuls). Returns 0 / <0. */
static int fa_heads_range_tiled(int fd, int n_tokens, int n_kv, int head_dim,
                                int n_head, int n_kv_heads, float scale, float softcap,
                                const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                                const _Float16 *mask, _Float16 *out,
                                fa_scratch *s, int h0, int h1, int Kbp)
{
    const int gqa = n_head / n_kv_heads, dh = head_dim;
    const int Tp  = (n_tokens + 3) & ~3;
    _Float16 *qh = s->qh, *kh = s->kh, *vh = s->vh, *sc = s->sc, *P = s->P, *pv = s->ctx;
    float *m = s->fm, *l = s->fl, *acc = s->facc;
    int rc;

    for (int h = h0; h < h1; h++) {
        const int hk = h / gqa;
        const _Float16 *Qh = Q + (size_t)h  * n_tokens * dh;   /* [n_tokens,dh] */
        const _Float16 *Kh = K + (size_t)hk * n_kv     * dh;   /* [n_kv,dh]     */
        const _Float16 *Vh = V + (size_t)hk * dh       * n_kv; /* [dh,n_kv]     */

        fa_gather_q(qh, Qh, n_tokens, Tp, dh);
        for (int i = 0; i < Tp; i++) { m[i] = -INFINITY; l[i] = 0.0f; }
        memset(acc, 0, (size_t)Tp * dh * sizeof(float));

        for (int j0 = 0; j0 < n_kv; j0 += Kbp) {
            const int w  = (n_kv - j0 < Kbp) ? (n_kv - j0) : Kbp;  /* real keys this tile */
            const int wp = (w + 31) & ~31;                         /* matmul N/K %32      */

            fa_gather_k(kh, Kh + (size_t)j0 * dh, w, wp, dh);
            fa_gather_v_tile(vh, Vh, j0, w, wp, dh, n_kv);

            /* tile scores sc[Tp,wp] = qh·kh^T  (M=Tp,K=dh,N=wp) */
            if ((rc = rocket_matmul_fp16(fd, Tp, dh, wp, qh, kh, sc)) != 0) return rc;
            fa_mask_scores_tile(sc, mask, n_tokens, n_kv, j0, w, Tp, wp, scale, softcap);

            /* fold the tile into the running softmax -> P[Tp,wp] (fp16, feeds the AV matmul) */
            for (int i = 0; i < Tp; i++) {
                const _Float16 *sr = sc + (size_t)i * wp;
                _Float16 *pr = P + (size_t)i * wp;
                float tmax = -INFINITY;
                for (int j = 0; j < wp; j++) { float v = (float)sr[j]; if (v > tmax) tmax = v; }
                if (tmax <= -29999.0f) {           /* every key masked for this row in this tile */
                    memset(pr, 0, (size_t)wp * sizeof(_Float16));  /* P·V adds 0; state untouched */
                    continue;
                }
                float mnew = m[i] > tmax ? m[i] : tmax;
                float corr = (m[i] == -INFINITY) ? 0.0f : expf(m[i] - mnew);
                float rs = 0.0f;
                for (int j = 0; j < wp; j++) { float e = expf((float)sr[j] - mnew); pr[j] = (_Float16)e; rs += e; }
                l[i] = l[i] * corr + rs;
                if (corr != 1.0f) { float *ar = acc + (size_t)i * dh; for (int c = 0; c < dh; c++) ar[c] *= corr; }
                m[i] = mnew;
            }

            /* running output += P_tile·V_tile  (M=Tp,K=wp,N=dh); B=vh[dh,wp] */
            if ((rc = rocket_matmul_fp16(fd, Tp, wp, dh, P, vh, pv)) != 0) return rc;
            for (int i = 0; i < Tp; i++) {
                float *ar = acc + (size_t)i * dh; const _Float16 *pr = pv + (size_t)i * dh;
                for (int c = 0; c < dh; c++) ar[c] += (float)pr[c];
            }
        }

        /* finalize out_h = acc / l (rows with no visible key -> l==0 -> 0) */
        _Float16 *Oh = out + (size_t)h * n_tokens * dh;
        for (int t = 0; t < n_tokens; t++) {
            float inv = l[t] > 0.0f ? 1.0f / l[t] : 0.0f;
            const float *ar = acc + (size_t)t * dh; _Float16 *orow = Oh + (size_t)t * dh;
            for (int c = 0; c < dh; c++) orow[c] = (_Float16)(ar[c] * inv);
        }
    }
    return 0;
}

/* Compute attention heads [h0,h1) on ONE fd, writing each head's output slice, using the
 * caller-supplied scratch `s` (pre-ensured for this (Tp,Kn,dh)). host_sm selects the host vs
 * on-NPU softmax. Shared by the single-fd entry, each stateless mt worker, and the persistent
 * context; heads are independent (disjoint output slices) so a head subset computes in
 * isolation. Reentrant as long as each concurrent caller passes its OWN scratch. */
static int fa_heads_range(int fd, int n_tokens, int n_kv, int head_dim, int dv,
                          int n_head, int n_kv_heads, float scale, float softcap,
                          const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                          const _Float16 *mask, _Float16 *out,
                          fa_scratch *s, int h0, int h1, int host_sm)
{
    const int gqa = n_head / n_kv_heads;
    const int dh  = head_dim;                /* dk: the QK contraction / query-key head dim */
    const int Tp  = (n_tokens + 3) & ~3;     /* matmul M%4 (query rows)             */
    const int Kn  = (n_kv + 31) & ~31;       /* matmul N (QK) and K (AV): %32 (=>%16) */
    _Float16 *qh = s->qh, *kh = s->kh, *vh = s->vh, *sc = s->sc, *P = s->P, *ctx = s->ctx;
    int rc;

    /* MLA (dv != dh): the chained-batch and online-tiled paths below assume a single head
     * dim (one shape for QK and AV, one scratch tile width), so they only run for dv == dh;
     * MLA takes the materialized per-head path. Correctness-first — chaining/tiling MLA is a
     * later perf follow-up. fa_scratch_ensure disables tiling on the same dv != dh condition,
     * so the scratch it sized (full Kn) matches the path taken here. */
    const int mla = (dv != dh);

    /* Long context: the online/tiled path (never materialize the [Tp,Kn] score matrix).
     * Checked FIRST — it owns the scratch sizing above (sc/kh/vh sized to one Kbp tile, not
     * Kn), so the chained-batch path below (which would grow its own Kn-wide group buffers)
     * must not run here. fa_tile_kv_width matches fa_scratch_ensure's decision exactly. */
    const int Kbp = mla ? 0 : fa_tile_kv_width(Kn);
    if (Kbp)
        return fa_heads_range_tiled(fd, n_tokens, n_kv, head_dim, n_head, n_kv_heads,
                                    scale, softcap, Q, K, V, mask, out, s, h0, h1, Kbp);

    /* Dispatch a multi-head range to the chained-batch path when ROCKET_FA_CHAIN is on
     * and a group of >1 heads fits the score-matrix budget (Gmax>1) — collapsing the
     * per-head QK/AV submits into one job each. Gmax==1 (long context) gains nothing, so
     * stay on the per-head path. The batched path uses its own scratch, not `s`. */
    if (!mla && fa_chain() && (h1 - h0) > 1) {
        const long per_head = (long)Tp * Kn;
        int Gmax = per_head > 0 ? (int)(fa_chain_elems() / per_head) : (h1 - h0);
        if (Gmax > (h1 - h0)) Gmax = h1 - h0;
        if (Gmax > 1)
            return fa_heads_range_batched(fd, n_tokens, n_kv, head_dim, n_head, n_kv_heads,
                                          scale, softcap, Q, K, V, mask, out, s, h0, h1, host_sm, Gmax);
    }

    for (int h = h0; h < h1; h++) {
        const int hk = h / gqa;
        const _Float16 *Qh = Q + (size_t)h  * n_tokens * dh;     /* [n_tokens,dh] (dk)  */
        const _Float16 *Kh = K + (size_t)hk * n_kv     * dh;     /* [n_kv,dh]     (dk)  */
        const _Float16 *Vh = V + (size_t)hk * dv       * n_kv;   /* [dv,n_kv]           */

        /* gather this head's operands into the padded matmul tiles */
        fa_gather_q(qh, Qh, n_tokens, Tp, dh);
        fa_gather_k(kh, Kh, n_kv, Kn, dh);
        fa_gather_v(vh, Vh, n_kv, Kn, dv);

        /* scores[Tp,Kn] = qh·kh^T  (M=Tp,K=dh,N=Kn) */
        if ((rc = rocket_matmul_fp16(fd, Tp, dh, Kn, qh, kh, sc)) != 0) return rc;

        /* scale + soft-cap + mask, in place; pad key columns -> -inf */
        fa_mask_scores(sc, mask, n_tokens, n_kv, Tp, Kn, scale, softcap);

        /* P = softmax(scores) over the Kn columns (pad columns -> ~0) */
        if (host_sm) host_softmax_rows(Tp, Kn, sc, P);
        else if ((rc = rocket_softmax_fp16(fd, Tp, Kn, sc, P)) != 0) return rc;

        /* ctx[Tp,dv] = P·v_hk  (M=Tp,K=Kn,N=dv); B=vh[dv,Kn] so B[n=c][k=j]=V[c,j] */
        if ((rc = rocket_matmul_fp16(fd, Tp, Kn, dv, P, vh, ctx)) != 0) return rc;

        _Float16 *Oh = out + (size_t)h * n_tokens * dv;
        for (int t = 0; t < n_tokens; t++) memcpy(Oh + (size_t)t * dv, ctx + (size_t)t * dv, dv * sizeof(_Float16));
    }
    (void)rc;
    return 0;
}

int rocket_flash_attn_fp16(int fd, int n_tokens, int n_kv, int head_dim, int dv,
                           int n_head, int n_kv_heads, float scale, float softcap,
                           const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                           const _Float16 *mask, _Float16 *out)
{
    if (n_tokens < 1 || n_kv < 1 || head_dim < 1 || dv < 1 || n_head < 1 || n_kv_heads < 1) return -1;
    if (n_head % n_kv_heads != 0 || head_dim % 32 != 0 || dv % 16 != 0) return -1;   /* GQA + QK K%32 + AV N%16 */
    if (fd < 0) {
        rocket_flash_attn_ref_fp16(n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads,
                                   scale, softcap, Q, K, V, mask, out);
        return 0;
    }
    fa_scratch s = {0};
    if (fa_scratch_ensure(&s, (n_tokens + 3) & ~3, (n_kv + 31) & ~31, head_dim, dv)) {
        fa_scratch_free(&s); return -2;
    }
    int rc = fa_heads_range(fd, n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads,
                            scale, softcap, Q, K, V, mask, out, &s, 0, n_head, fa_host_softmax());
    fa_scratch_free(&s);
    return rc;
}

/* ############################################################################
 * PART 4 — Flash-attn public API: single-fd, multicore fan-out, persistent ctx
 * ##########################################################################*/

/* ---- multi-core flash attention: fan the heads across worker fds ----------------
 * Running all heads on one fd would serialize the per-head QK/softmax/AV submits
 * onto one NPU core — and attention is hundreds of small, dispatch-bound
 * per-head GEMMs, the regime where this NPU is slowest. The heads are independent (each
 * writes its own output slice and reads only Q/K/V/mask), so they fan out cleanly: split
 * the n_head heads into contiguous ranges, one worker fd per range. The rocket driver makes
 * one scheduling entity per fd and an entity pins to one core while it has queued work, so
 * independent fds let the kernel dispatch the head ranges across the NPU cores in parallel
 * (the rocket_matmul_fp16_mt fan-out, by head instead of by output column). Each worker also
 * runs its own host-side gather + softmax, so that one-core bottleneck parallelizes too. */
typedef struct {
    int fd;                /* own_fd ? unused : the persistent worker fd to use            */
    fa_scratch *s;         /* NULL -> the worker allocates+frees a local scratch (stateless) */
    int own_fd;            /* 1 -> the worker opens/closes its own fd (stateless mt path)   */
    int n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads;
    float scale, softcap;
    const _Float16 *Q, *K, *V, *mask;
    _Float16 *out;
    int h0, h1, host_sm, idx;
    int ret;
} fa_mt_arg;

static void *fa_mt_worker(void *a)
{
    fa_mt_arg *w = (fa_mt_arg *)a;
    rocket_pin_worker(w->idx);          /* keep the gather/softmax off the A55 littles */
    int fd = w->own_fd ? rocket_open() : w->fd;
    if (fd < 0) { w->ret = fd; return NULL; }

    fa_scratch local = {0};
    fa_scratch *s = w->s;
    if (!s) {                            /* stateless: this worker's own per-call scratch */
        if (fa_scratch_ensure(&local, (w->n_tokens + 3) & ~3, (w->n_kv + 31) & ~31, w->head_dim, w->dv)) {
            w->ret = -2;
            if (w->own_fd) rocket_close(fd);
            return NULL;
        }
        s = &local;
    }
    w->ret = fa_heads_range(fd, w->n_tokens, w->n_kv, w->head_dim, w->dv, w->n_head, w->n_kv_heads,
                            w->scale, w->softcap, w->Q, w->K, w->V, w->mask, w->out,
                            s, w->h0, w->h1, w->host_sm);
    if (s == &local) fa_scratch_free(&local);
    if (w->own_fd) rocket_close(fd);
    return NULL;
}

/* Fan the heads across `nt` workers. Stateless path (fds==NULL, scratch==NULL): each worker
 * opens its own fd and allocates its own per-call scratch. Persistent path (fds + scratch
 * arrays, one per worker, scratch pre-ensured for this shape): each worker reuses them. The
 * heads split into balanced contiguous ranges; threads spawn, any pthread_create failure runs
 * inline after the spawn loop (the rocket_matmul_fp16_mt idiom — inline HERE would block every
 * not-yet-spawned worker behind its NPU wait). */
static int fa_fan_heads(const int *fds, fa_scratch *scratch, int nt,
                        int n_tokens, int n_kv, int head_dim, int dv, int n_head, int n_kv_heads,
                        float scale, float softcap, const _Float16 *Q, const _Float16 *K,
                        const _Float16 *V, const _Float16 *mask, _Float16 *out, int host_sm)
{
    pthread_t th[8];
    fa_mt_arg args[8];
    int joinable[8] = {0};
    const int base = n_head / nt, rem = n_head % nt;
    int h0 = 0, n = 0;
    for (int t = 0; t < nt; t++) {
        const int cnt = base + (t < rem ? 1 : 0);
        if (cnt <= 0) break;
        args[n] = (fa_mt_arg){
            .fd = fds ? fds[t] : -1, .s = scratch ? &scratch[t] : NULL, .own_fd = fds ? 0 : 1,
            .n_tokens = n_tokens, .n_kv = n_kv, .head_dim = head_dim, .dv = dv,
            .n_head = n_head, .n_kv_heads = n_kv_heads, .scale = scale, .softcap = softcap,
            .Q = Q, .K = K, .V = V, .mask = mask, .out = out,
            .h0 = h0, .h1 = h0 + cnt, .host_sm = host_sm, .idx = n, .ret = 0 };
        if (pthread_create(&th[n], NULL, fa_mt_worker, &args[n]) == 0)
            joinable[n] = 1;
        h0 += cnt;
        n++;
    }
    for (int t = 0; t < n; t++)
        if (!joinable[t]) fa_mt_worker(&args[t]);

    int ret = 0;
    for (int t = 0; t < n; t++) {
        if (joinable[t]) pthread_join(th[t], NULL);
        if (args[t].ret) ret = args[t].ret;
    }
    return ret;
}

int rocket_flash_attn_fp16_mt(int fd, int n_tokens, int n_kv, int head_dim, int dv,
                              int n_head, int n_kv_heads, float scale, float softcap,
                              const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                              const _Float16 *mask, _Float16 *out, int nthreads)
{
    if (n_tokens < 1 || n_kv < 1 || head_dim < 1 || dv < 1 || n_head < 1 || n_kv_heads < 1) return -1;
    if (n_head % n_kv_heads != 0 || head_dim % 32 != 0 || dv % 16 != 0) return -1;   /* GQA + QK K%32 + AV N%16 */
    if (fd < 0) {
        rocket_flash_attn_ref_fp16(n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads,
                                   scale, softcap, Q, K, V, mask, out);
        return 0;
    }
    const int host_sm = fa_host_softmax();

    if (nthreads < 1) nthreads = 1;
    if (nthreads > 8) nthreads = 8;
    if (nthreads > n_head) nthreads = n_head;       /* never more workers than heads */
    if (nthreads == 1) {
        fa_scratch s = {0};
        if (fa_scratch_ensure(&s, (n_tokens + 3) & ~3, (n_kv + 31) & ~31, head_dim, dv)) {
            fa_scratch_free(&s); return -2;
        }
        int rc = fa_heads_range(fd, n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads,
                                scale, softcap, Q, K, V, mask, out, &s, 0, n_head, host_sm);
        fa_scratch_free(&s);
        return rc;
    }
    return fa_fan_heads(NULL, NULL, nthreads, n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads,
                        scale, softcap, Q, K, V, mask, out, host_sm);
}

/* ---- persistent flash-attention context -----------------------------------------
 * The mt entry above opens/closes `nthreads` worker fds, spawns/joins threads, and (in each
 * worker) mallocs the per-head scratch on EVERY call — i.e. per layer per forward. At long
 * context the sc/P score matrices are 8-16 MB, so each call mmaps+faults+munmaps them; the fd
 * open also re-attaches an IOMMU domain. A persistent context holds the worker fds open and
 * keeps the per-worker scratch resident (grown to the largest shape seen), so a forward only
 * pays the thread spawn/join. This is the rocket_ctx/rocket_stream pattern applied to FA; it
 * pays off exactly where FA-NPU is competitive (long context, where the per-call mmap bites).
 * Numerically identical to rocket_flash_attn_fp16_mt — the same fa_heads_range body. */
struct rocket_fa_ctx {
    int        nthreads;
    int        fd[8];
    fa_scratch sc[8];
};

rocket_fa_ctx *rocket_fa_ctx_create(int nthreads)
{
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 8) nthreads = 8;
    rocket_fa_ctx *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->nthreads = nthreads;
    for (int t = 0; t < 8; t++) c->fd[t] = -1;
    for (int t = 0; t < nthreads; t++) {
        c->fd[t] = rocket_open();
        if (c->fd[t] < 0) { rocket_fa_ctx_free(c); return NULL; }
    }
    /* When chaining is enabled, give each worker its own resident QK and AV batched-
     * matmul contexts (one per shape — QK and AV alternate, so a shared context would
     * re-zero every op), bound to that worker's fd. They lazily allocate their BOs on
     * first use, so this only opens the small guard/regcmd BOs when ROCKET_FA_CHAIN=1;
     * the per-call paths leave them NULL. */
    if (fa_chain()) {
        for (int t = 0; t < nthreads; t++) {
            c->sc[t].qk = rocket_mm_batch_create(c->fd[t]);
            c->sc[t].av = rocket_mm_batch_create(c->fd[t]);
            if (!c->sc[t].qk || !c->sc[t].av) { rocket_fa_ctx_free(c); return NULL; }
        }
    }
    return c;
}

void rocket_fa_ctx_free(rocket_fa_ctx *c)
{
    if (!c) return;
    for (int t = 0; t < c->nthreads; t++) {
        fa_scratch_free(&c->sc[t]);
        if (c->fd[t] >= 0) rocket_close(c->fd[t]);
    }
    free(c);
}

int rocket_flash_attn_fp16_ctx(rocket_fa_ctx *c, int n_tokens, int n_kv, int head_dim, int dv,
                               int n_head, int n_kv_heads, float scale, float softcap,
                               const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                               const _Float16 *mask, _Float16 *out)
{
    if (!c) return -1;
    if (n_tokens < 1 || n_kv < 1 || head_dim < 1 || dv < 1 || n_head < 1 || n_kv_heads < 1) return -1;
    if (n_head % n_kv_heads != 0 || head_dim % 32 != 0 || dv % 16 != 0) return -1;   /* GQA + QK K%32 + AV N%16 */
    const int host_sm = fa_host_softmax();

    int nt = c->nthreads;
    if (nt > n_head) nt = n_head;                   /* never more workers than heads */
    const int Tp = (n_tokens + 3) & ~3, Kn = (n_kv + 31) & ~31;
    /* Grow each worker's resident scratch up-front on this thread (each worker touches only its
     * own sc[t], so the workers never realloc concurrently). */
    for (int t = 0; t < nt; t++)
        if (fa_scratch_ensure(&c->sc[t], Tp, Kn, head_dim, dv)) return -2;

    if (nt == 1)
        return fa_heads_range(c->fd[0], n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads,
                              scale, softcap, Q, K, V, mask, out, &c->sc[0], 0, n_head, host_sm);

    return fa_fan_heads(c->fd, c->sc, nt, n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads,
                        scale, softcap, Q, K, V, mask, out, host_sm);
}
