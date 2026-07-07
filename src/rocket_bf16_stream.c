// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_bf16_stream.c — fast bf16 x bf16 -> fp32 matmul: multicore fan-out plus a
 * streaming (persistent-fd, resident-scratch) context. The single-fd
 * rocket_matmul_bf16 (rocket_matmul.c) opens nothing persistent and allocates +
 * frees all BOs every call, so an in-model bf16 prefill paid the fd-serialised one
 * core plus a full BO alloc/free per matmul. These paths hoist both out of the loop:
 *
 *   - rocket_matmul_bf16_mt   : split the output columns N across `nthreads` worker
 *                               fds (one scheduling entity per fd => the 3 NPU cores
 *                               run in parallel), each running the UNCHANGED single-fd
 *                               rocket_matmul_bf16 on its slice. No resident state.
 *   - rocket_matmul_bf16_stream: the same N-split, but a persistent context keeps the
 *                               worker fds AND the per-(M,K,N) scratch BOs (input/
 *                               weight/output/regcmd) resident, re-packing only A and
 *                               B per call. The bf16 sibling of rocket_matmul_fp16_stream
 *                               (rocket_prepacked.c); bf16 weights are used once per
 *                               prefill token batch, so there is no resident-weight
 *                               (prepacked) variant — streaming is the right shape.
 *
 * bf16 reuses int16's NPU geometry exactly (2-byte input cube C2=8, weight
 * (N/16,K/32,16,32), 4-byte fp32 output cube C2=4) — identical index math to the fp16
 * input path, differing only in the bf16 truncation on the scatter, the fp32 output
 * cube, and gen_matmul_bf16 (precision 3). K-partials accumulate on the HOST in double
 * (no NPU K-accum: the DPU eltwise operand is <=16-bit and the bf16 output is fp32).
 *
 * The streaming output is bit-identical to single-fd rocket_matmul_bf16 at nthreads=1
 * (one worker, nsub==N => identical tiling and host-accum order); at nthreads>1 it is
 * the same valid bf16 product within float-accum reassociation (gated against an exact
 * double reference). matmul_bf16_stream_rocket is the HW gate.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "rocket_matmul.h"
#include "npu_matmul.h"        // matmul_params_t + gen_matmul_bf16
#include "rocket_affinity.h"
#include "rocket_log.h"

#define BFS_MAX_WORKERS 8
#define BFS_MAX_SLOTS   32
#define BFS_BATCH       64           /* tiles (tasks) per NPU job (== rocket_matmul.c)  */
#define BFS_RC_STRIDE   128          /* u64 words reserved per task in the regcmd BO    */
#define BFS_CBUF_BANK   NPU_CBUF_BANK_SIZE

/* ============================================================================
 * SECTION — bf16 tile-layout index math and small helpers
 * ==========================================================================*/

static int rup(int x, int a) { return ((x + a - 1) / a) * a; }

/* bf16 input geometry == fp16/int16 input geometry (feature cube C2=8, weight tile
 * (N/16,K/32,16,32)); output cube is the 4-byte C2=4 the int32/fp32-out paths use. */
static inline size_t bfs_feat_idx(int H, int ch, int h) {   /* input, C2=8 */
    return ((size_t)(ch - 1) / 8) * (size_t)H * 8 + 8 * (size_t)(h - 1) + (ch - 1) % 8;
}
static inline size_t bfs_wt_idx(int C, int k, int c) {      /* weight, (N/16,K/32,16,32) */
    return (size_t)((c - 1) / 32) * 32 * 16 + (size_t)((k - 1) / 16) * 16 * C
         + (size_t)((c - 1) % 32) + (size_t)((k - 1) % 16) * 32;
}
static inline size_t bfs_out_idx(int H, int ch, int h) {    /* output fp32 elem, C2=4 */
    return ((size_t)(ch - 1) / 4) * (size_t)H * 4 + 4 * (size_t)(h - 1) + (ch - 1) % 4;
}
/* fp32 -> bf16 by truncation (high 16 bits) — matches rocket_matmul_bf16. */
static inline uint16_t f32_to_bf16(float f) {
    uint32_t b; memcpy(&b, &f, sizeof b); return (uint16_t)(b >> 16);
}
static long bfs_wait_ns(void) {
    const char *e = getenv("ROCKET_WAIT_MS");
    long ms = e ? atol(e) : 8000;
    if (ms < 1) ms = 8000;
    return ms * 1000000L;
}

/* ============================================================================
 * SECTION — Tiling plan, BO scratch, and pack/compute primitives
 * ==========================================================================*/

/* Resolved tiling plan + per-tile slot sizes (input/weight in bf16 elems, output in
 * fp32 elems) for one worker's (M,K,nsub). */
typedef struct {
    int M, K, N;
    int Mt, Kt, Nt, nMt, nNt, nKt;
    size_t in_slot, wt_slot, out_slot;   /* elems */
} bfs_plan;

static int bfs_plan_init(bfs_plan *pl, int M, int K, int N)
{
    int Mt, Kt, Nt;
    if (rocket_matmul_plan_bf16(M, K, N, &Mt, &Kt, &Nt) < 0) return -1;
    pl->M = M; pl->K = K; pl->N = N;
    pl->Mt = Mt; pl->Kt = Kt; pl->Nt = Nt;
    pl->nMt = (M + Mt - 1) / Mt;
    pl->nNt = (N + Nt - 1) / Nt;
    pl->nKt = (K + Kt - 1) / Kt;
    pl->in_slot  = (size_t)rup(Mt, 4)  * rup(Kt, 32);
    pl->wt_slot  = (size_t)rup(Nt, 16) * rup(Kt, 32);
    pl->out_slot = (size_t)rup(Mt, 4)  * rup(Nt, 16);
    return 0;
}

/* The five BOs a tiled bf16 matmul drives + the resident host-side compute scratch. */
typedef struct {
    rocket_bo guard, regcmd, in_all, wt_all, out_all;
    int prezeroed;                       /* in_all/wt_all zeroed once at alloc */
    rocket_task_desc *tasks;             /* BATCH                              */
    int    *bm0, *bn0, *bMtile, *bNtile; /* BATCH each                         */
    size_t *boff;                        /* BATCH                              */
    double *acc;                         /* M*N fp32 (double) accumulator      */
} bfs_bos;

static void bfs_bos_free(int fd, bfs_bos *b)
{
    rocket_bo_free(fd, &b->guard);
    rocket_bo_free(fd, &b->regcmd);
    rocket_bo_free(fd, &b->in_all);
    rocket_bo_free(fd, &b->wt_all);
    rocket_bo_free(fd, &b->out_all);
    free(b->tasks); free(b->bm0); free(b->bn0); free(b->bMtile);
    free(b->bNtile); free(b->boff); free(b->acc);
    memset(b, 0, sizeof(*b));
}

static int bfs_bos_alloc(int fd, const bfs_plan *pl, bfs_bos *b)
{
    memset(b, 0, sizeof(*b));
    size_t in_sz  = (size_t)pl->nMt * pl->nKt * pl->in_slot * sizeof(uint16_t) + BFS_CBUF_BANK;
    size_t wt_sz  = (size_t)pl->nNt * pl->nKt * pl->wt_slot * sizeof(uint16_t) + BFS_CBUF_BANK;
    size_t rc_sz  = (size_t)BFS_BATCH * BFS_RC_STRIDE * sizeof(uint64_t);
    size_t out_sz = (size_t)BFS_BATCH * pl->out_slot * sizeof(float) + BFS_CBUF_BANK;

    int ret = 0;
    ret |= rocket_bo_alloc(fd, 4096,   &b->guard);
    ret |= rocket_bo_alloc(fd, rc_sz,  &b->regcmd);
    ret |= rocket_bo_alloc(fd, in_sz,  &b->in_all);
    ret |= rocket_bo_alloc(fd, wt_sz,  &b->wt_all);
    ret |= rocket_bo_alloc(fd, out_sz, &b->out_all);
    if (ret) { ROCKET_LOGE("bf16_stream: BO alloc failed\n"); goto fail; }
    if (((b->in_all.dma_address + in_sz) | (b->wt_all.dma_address + wt_sz) |
         (b->out_all.dma_address + out_sz) | (b->regcmd.dma_address + rc_sz)) >> 32) {
        ROCKET_LOGE("bf16_stream: a BO dma_address exceeds 32 bits\n"); ret = -1; goto fail;
    }

    /* Prezero the input/weight BOs ONCE so the per-call pack writes only live tile
     * lanes (the padding stays zero across calls — same trick as mm_bos_alloc). */
    if (rocket_bo_prep(fd, &b->in_all, 1, 0) != 0) { ret = -1; goto fail; }
    memset(b->in_all.ptr, 0, b->in_all.size); rocket_bo_fini(fd, &b->in_all);
    if (rocket_bo_prep(fd, &b->wt_all, 1, 0) != 0) { ret = -1; goto fail; }
    memset(b->wt_all.ptr, 0, b->wt_all.size); rocket_bo_fini(fd, &b->wt_all);
    b->prezeroed = 1;

    b->tasks  = malloc(BFS_BATCH * sizeof(*b->tasks));
    b->bm0    = malloc(BFS_BATCH * sizeof(int));
    b->bn0    = malloc(BFS_BATCH * sizeof(int));
    b->bMtile = malloc(BFS_BATCH * sizeof(int));
    b->bNtile = malloc(BFS_BATCH * sizeof(int));
    b->boff   = malloc(BFS_BATCH * sizeof(size_t));
    b->acc    = calloc((size_t)pl->M * pl->N, sizeof(double));
    if (!b->tasks || !b->bm0 || !b->bn0 || !b->bMtile || !b->bNtile || !b->boff || !b->acc) {
        ret = -1; goto fail;
    }
    return 0;
fail:
    bfs_bos_free(fd, b);
    return ret;
}

/* Scatter A[M,K] -> bf16 feature cube into a host buffer `dst` (truncate). dst must
 * hold nMt*nKt*in_slot uint16 and have its padding pre-zeroed (caller calloc's it). */
static void bfs_scatter_A(const bfs_plan *pl, uint16_t *dst, const float *A)
{
    for (int mi = 0; mi < pl->nMt; mi++) {
        int m0 = mi * pl->Mt, Mtile = (pl->M - m0 < pl->Mt) ? (pl->M - m0) : pl->Mt;
        for (int ki = 0; ki < pl->nKt; ki++) {
            int k0 = ki * pl->Kt, Ktile = (pl->K - k0 < pl->Kt) ? (pl->K - k0) : pl->Kt;
            uint16_t *slot = dst + (size_t)(mi * pl->nKt + ki) * pl->in_slot;
            for (int h = 1; h <= Mtile; h++)
                for (int c = 1; c <= Ktile; c++)
                    slot[bfs_feat_idx(Mtile, c, h)] =
                        f32_to_bf16(A[(size_t)(m0 + h - 1) * pl->K + (k0 + c - 1)]);
        }
    }
}

/* memcpy a shared pre-scattered A buffer into this worker's resident in_all BO. */
static int bfs_load_A(int fd, const bfs_plan *pl, bfs_bos *b, const uint16_t *packed)
{
    if (rocket_bo_prep(fd, &b->in_all, 1, 0) != 0) return -1;
    memcpy(b->in_all.ptr, packed,
           (size_t)pl->nMt * pl->nKt * pl->in_slot * sizeof(uint16_t));
    rocket_bo_fini(fd, &b->in_all);
    return 0;
}

/* Pack A[M,K] directly into this worker's resident in_all BO (per-worker fallback when
 * the shared A-pack is disabled or workers have heterogeneous input tiling). */
static int bfs_pack_A(int fd, const bfs_plan *pl, bfs_bos *b, const float *A)
{
    if (rocket_bo_prep(fd, &b->in_all, 1, 0) != 0) return -1;
    if (!b->prezeroed) memset(b->in_all.ptr, 0, b->in_all.size);
    bfs_scatter_A(pl, (uint16_t *)b->in_all.ptr, A);
    rocket_bo_fini(fd, &b->in_all);
    return 0;
}

/* Pack this worker's weight slice Bslice[N,K] (row-major, N==pl->N) into wt_all. */
static int bfs_pack_B(int fd, const bfs_plan *pl, bfs_bos *b, const float *Bslice)
{
    if (rocket_bo_prep(fd, &b->wt_all, 1, 0) != 0) return -1;
    if (!b->prezeroed) memset(b->wt_all.ptr, 0, b->wt_all.size);
    for (int ni = 0; ni < pl->nNt; ni++) {
        int n0 = ni * pl->Nt, Ntile = (pl->N - n0 < pl->Nt) ? (pl->N - n0) : pl->Nt;
        for (int ki = 0; ki < pl->nKt; ki++) {
            int k0 = ki * pl->Kt, Ktile = (pl->K - k0 < pl->Kt) ? (pl->K - k0) : pl->Kt;
            uint16_t *slot = (uint16_t *)b->wt_all.ptr + (size_t)(ni * pl->nKt + ki) * pl->wt_slot;
            for (int kk = 1; kk <= Ntile; kk++)
                for (int c = 1; c <= Ktile; c++)
                    slot[bfs_wt_idx(Ktile, kk, c)] =
                        f32_to_bf16(Bslice[(size_t)(n0 + kk - 1) * pl->K + (k0 + c - 1)]);
        }
    }
    rocket_bo_fini(fd, &b->wt_all);
    return 0;
}

/* Batched tile compute over already-packed in_all/wt_all: host double K-accum into
 * acc, then narrow to fp32 Csub[M,N] row-major. Mirrors rocket_matmul_bf16's loop. */
static int bfs_compute(int fd, const bfs_plan *pl, bfs_bos *b, float *Csub)
{
    int M = pl->M, N = pl->N;
    memset(b->acc, 0, (size_t)M * N * sizeof(double));
    uint64_t npu_regs[256] = {0};
    int total = pl->nMt * pl->nNt * pl->nKt, done = 0, nb = 0, ret = 0;

    for (int mi = 0; mi < pl->nMt; mi++) {
        int m0 = mi * pl->Mt, Mtile = (M - m0 < pl->Mt) ? (M - m0) : pl->Mt;
        for (int ni = 0; ni < pl->nNt; ni++) {
            int n0 = ni * pl->Nt, Ntile = (N - n0 < pl->Nt) ? (N - n0) : pl->Nt;
            for (int ki = 0; ki < pl->nKt; ki++) {
                int k0 = ki * pl->Kt, Ktile = (pl->K - k0 < pl->Kt) ? (pl->K - k0) : pl->Kt;

                if (nb == 0) rocket_bo_prep(fd, &b->regcmd, 1, 0);

                size_t out_off = (size_t)nb * pl->out_slot;
                matmul_params_t p = {
                    .m = (uint16_t)Mtile, .k = (uint16_t)Ktile, .n = (uint16_t)Ntile,
                    .input_dma   = (uint32_t)(b->in_all.dma_address  + (size_t)(mi*pl->nKt+ki) * pl->in_slot * sizeof(uint16_t)),
                    .weights_dma = (uint32_t)(b->wt_all.dma_address  + (size_t)(ni*pl->nKt+ki) * pl->wt_slot * sizeof(uint16_t)),
                    .output_dma  = (uint32_t)(b->out_all.dma_address + out_off * sizeof(float)),
                    .tasks = npu_regs,
                };
                if ((ret = gen_matmul_bf16(&p)) != 0) {
                    ROCKET_LOGE("bf16_stream: gen failed (%d)\n", ret); return ret;
                }
                if ((size_t)p.task_count > (size_t)BFS_RC_STRIDE) {
                    ROCKET_LOGE("bf16_stream: regcmd slot overflow %u > %d\n", p.task_count, BFS_RC_STRIDE);
                    return -1;
                }
                memcpy((uint64_t *)b->regcmd.ptr + (size_t)nb * BFS_RC_STRIDE, npu_regs,
                       (size_t)p.task_count * sizeof(uint64_t));
                b->tasks[nb].regcmd = (uint32_t)(b->regcmd.dma_address + (size_t)nb * BFS_RC_STRIDE * sizeof(uint64_t));
                b->tasks[nb].regcmd_count = p.task_count;
                b->bm0[nb] = m0; b->bn0[nb] = n0; b->bMtile[nb] = Mtile; b->bNtile[nb] = Ntile;
                b->boff[nb] = out_off;
                nb++; done++;

                if (nb == BFS_BATCH || done == total) {
                    rocket_bo_fini(fd, &b->regcmd);
                    rocket_bo_prep(fd, &b->out_all, 1, 0);
                    rocket_bo_fini(fd, &b->out_all);

                    uint32_t in_h[]  = { b->in_all.handle, b->wt_all.handle, b->regcmd.handle };
                    uint32_t out_h[] = { b->out_all.handle };
                    if ((ret = rocket_submit_tasks(fd, b->tasks, nb, in_h, 3, out_h, 1)) != 0) return ret;
                    if ((ret = rocket_bo_prep(fd, &b->out_all, 0, bfs_wait_ns())) != 0) {
                        ROCKET_LOGE("bf16_stream: WAIT TIMEOUT (%d) M=%d K=%d N=%d batch=%d %d/%d\n",
                                ret, M, pl->K, N, nb, done, total);
                        return ret;
                    }
                    float *ob = (float *)b->out_all.ptr;
                    for (int j = 0; j < nb; j++) {
                        float *slot = ob + b->boff[j];
                        for (int h = 1; h <= b->bMtile[j]; h++)
                            for (int nn = 1; nn <= b->bNtile[j]; nn++)
                                b->acc[(size_t)(b->bm0[j] + h - 1) * N + (b->bn0[j] + nn - 1)] +=
                                    (double)slot[bfs_out_idx(b->bMtile[j], nn, h)];
                    }
                    rocket_bo_fini(fd, &b->out_all);
                    nb = 0;
                }
            }
        }
    }
    for (size_t i = 0; i < (size_t)M * N; i++) Csub[i] = (float)b->acc[i];
    return 0;
}

/* ============================================================================
 * SECTION — Public API: multicore fan-out (rocket_matmul_bf16_mt)
 * ==========================================================================*/

typedef struct {
    int M, K, N, n0, nsub;
    const float *A, *B;
    float *C;
    int ret, idx;
} bfs_mt_arg;

static void *bfs_mt_worker(void *a)
{
    bfs_mt_arg *w = (bfs_mt_arg *)a;
    rocket_pin_worker(w->idx);
    int fd = rocket_open();
    if (fd < 0) { w->ret = fd; return NULL; }
    float *Csub = malloc((size_t)w->M * w->nsub * sizeof(float));
    if (!Csub) { rocket_close(fd); w->ret = -1; return NULL; }

    w->ret = rocket_matmul_bf16(fd, w->M, w->K, w->nsub,
                                w->A, w->B + (size_t)w->n0 * w->K, Csub);
    if (w->ret == 0)
        for (int m = 0; m < w->M; m++)
            memcpy(w->C + (size_t)m * w->N + w->n0,
                   Csub + (size_t)m * w->nsub, (size_t)w->nsub * sizeof(float));
    free(Csub);
    rocket_close(fd);
    return NULL;
}

static int bfs_nstep(int N, int nthreads)
{
    int s = ((N + nthreads - 1) / nthreads + 15) / 16 * 16;
    return s < 16 ? 16 : s;
}

int rocket_matmul_bf16_mt(int M, int K, int N,
                          const float *A, const float *B, float *C, int nthreads)
{
    if (nthreads < 1) nthreads = 1;
    if (nthreads > BFS_MAX_WORKERS) nthreads = BFS_MAX_WORKERS;
    int Nstep = bfs_nstep(N, nthreads);

    pthread_t th[BFS_MAX_WORKERS];
    bfs_mt_arg args[BFS_MAX_WORKERS];
    int joinable[BFS_MAX_WORKERS] = {0};
    int nt = 0;
    for (int t = 0; t < nthreads; t++) {
        int n0 = t * Nstep;
        if (n0 >= N) break;
        int n1 = n0 + Nstep; if (n1 > N) n1 = N;
        args[nt] = (bfs_mt_arg){ M, K, N, n0, n1 - n0, A, B, C, 0, nt };
        if (pthread_create(&th[nt], NULL, bfs_mt_worker, &args[nt]) == 0) joinable[nt] = 1;
        nt++;
    }
    for (int t = 0; t < nt; t++) if (!joinable[t]) bfs_mt_worker(&args[t]);
    int ret = 0;
    for (int t = 0; t < nt; t++) {
        if (joinable[t]) pthread_join(th[t], NULL);
        if (args[t].ret) ret = args[t].ret;
    }
    return ret;
}

/* ============================================================================
 * SECTION — Streaming context: persistent fds, resident scratch, public API
 * ==========================================================================*/

typedef struct {
    int     n0, nsub;
    bfs_plan pl;
    bfs_bos  bos;
} bfs_worker;

typedef struct {
    int M, K, N, nt;
    bfs_worker w[BFS_MAX_WORKERS];
    uint16_t *a_buf;          /* shared pre-scattered A (bf16), zeroed once; or NULL */
} bfs_scratch;

struct rocket_bf16_stream {
    int nthreads;
    int fd[BFS_MAX_WORKERS];
    bfs_scratch *scache[BFS_MAX_SLOTS];
    int nscache;
};

rocket_bf16_stream *rocket_bf16_stream_create(int nthreads)
{
    if (nthreads < 1) nthreads = 1;
    if (nthreads > BFS_MAX_WORKERS) nthreads = BFS_MAX_WORKERS;
    rocket_bf16_stream *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->nthreads = nthreads;
    for (int t = 0; t < nthreads; t++) s->fd[t] = -1;
    for (int t = 0; t < nthreads; t++) {
        s->fd[t] = rocket_open();
        if (s->fd[t] < 0) {
            ROCKET_LOGE("rocket_bf16_stream_create: rocket_open worker %d failed (%d)\n", t, s->fd[t]);
            for (int u = 0; u < t; u++) rocket_close(s->fd[u]);
            free(s);
            return NULL;
        }
    }
    return s;
}

static void bfs_scratch_free(rocket_bf16_stream *s, bfs_scratch *sc)
{
    if (!sc) return;
    for (int t = 0; t < sc->nt; t++) bfs_bos_free(s->fd[t], &sc->w[t].bos);
    free(sc->a_buf);
    free(sc);
}

void rocket_bf16_stream_free(rocket_bf16_stream *s)
{
    if (!s) return;
    for (int i = 0; i < s->nscache; i++) bfs_scratch_free(s, s->scache[i]);
    for (int t = 0; t < s->nthreads; t++)
        if (s->fd[t] >= 0) rocket_close(s->fd[t]);
    free(s);
}

static bfs_scratch *bfs_scratch_alloc(rocket_bf16_stream *s, int M, int K, int N)
{
    bfs_scratch *sc = calloc(1, sizeof(*sc));
    if (!sc) return NULL;
    sc->M = M; sc->K = K; sc->N = N;

    int Nstep = bfs_nstep(N, s->nthreads);
    int t = 0;
    for (; t < s->nthreads; t++) {
        int n0 = t * Nstep;
        if (n0 >= N) break;
        int nsub = (n0 + Nstep > N) ? (N - n0) : Nstep;
        bfs_worker *ww = &sc->w[t];
        ww->n0 = n0; ww->nsub = nsub;
        if (bfs_plan_init(&ww->pl, M, K, nsub) < 0) {
            ROCKET_LOGE("bf16_stream: unsupported slice M=%d K=%d N=%d\n", M, K, nsub);
            goto fail;
        }
        if (bfs_bos_alloc(s->fd[t], &ww->pl, &ww->bos) < 0) goto fail;
    }
    sc->nt = t;
    return sc;
fail:
    for (int u = 0; u < t; u++) bfs_bos_free(s->fd[u], &sc->w[u].bos);
    free(sc);
    return NULL;
}

static bfs_scratch *bfs_ctx_scratch(rocket_bf16_stream *s, int M, int K, int N)
{
    for (int i = 0; i < s->nscache; i++)
        if (s->scache[i]->M == M && s->scache[i]->K == K && s->scache[i]->N == N)
            return s->scache[i];
    if (s->nscache >= BFS_MAX_SLOTS) return NULL;
    bfs_scratch *sc = bfs_scratch_alloc(s, M, K, N);
    if (!sc) return NULL;
    s->scache[s->nscache++] = sc;
    return sc;
}

typedef struct {
    int fd;
    bfs_worker *ww;
    const float *A;            /* full A[M,K] (per-worker pack when packed==NULL) */
    const uint16_t *packed;    /* shared pre-scattered A, or NULL                 */
    const float *B;            /* full B[N,K] to re-pack this call                */
    float *C;                  /* full C[M,N]                                      */
    int M, N, ret, idx;
} bfs_run_arg;

static void *bfs_run_worker(void *a)
{
    bfs_run_arg *t = (bfs_run_arg *)a;
    rocket_pin_worker(t->idx);
    bfs_worker *ww = t->ww;
    int nsub = ww->nsub;

    float *Csub = malloc((size_t)t->M * nsub * sizeof(float));
    if (!Csub) { t->ret = -1; return NULL; }

    if (bfs_pack_B(t->fd, &ww->pl, &ww->bos, t->B + (size_t)ww->n0 * ww->pl.K) < 0) {
        t->ret = -1; free(Csub); return NULL;
    }
    int prc = t->packed ? bfs_load_A(t->fd, &ww->pl, &ww->bos, t->packed)
                        : bfs_pack_A(t->fd, &ww->pl, &ww->bos, t->A);
    if (prc < 0) { t->ret = -1; free(Csub); return NULL; }

    t->ret = bfs_compute(t->fd, &ww->pl, &ww->bos, Csub);
    if (t->ret == 0)
        for (int m = 0; m < t->M; m++)
            memcpy(t->C + (size_t)m * t->N + ww->n0,
                   Csub + (size_t)m * nsub, (size_t)nsub * sizeof(float));
    free(Csub);
    return NULL;
}

int rocket_matmul_bf16_stream(rocket_bf16_stream *s, int M, int K, int N,
                              const float *A, const float *B, float *C)
{
    if (!s) return -1;
    bfs_scratch *sc = bfs_ctx_scratch(s, M, K, N);
    if (!sc) return -1;

    /* Share the A scatter across workers when every worker plans the SAME input
     * tiling (Mt/Kt/nMt/nKt): scatter A once, hand workers the buffer to memcpy.
     * ROCKET_NO_SHARED_PACK=1 forces per-worker packing. */
    const bfs_plan *p0 = &sc->w[0].pl;
    int shared = (getenv("ROCKET_NO_SHARED_PACK") == NULL);
    for (int t = 1; shared && t < sc->nt; t++) {
        const bfs_plan *pt = &sc->w[t].pl;
        if (pt->Mt != p0->Mt || pt->Kt != p0->Kt ||
            pt->nMt != p0->nMt || pt->nKt != p0->nKt || pt->in_slot != p0->in_slot) {
            shared = 0; break;
        }
    }
    const uint16_t *packed = NULL;
    if (shared) {
        if (!sc->a_buf)
            sc->a_buf = calloc((size_t)p0->nMt * p0->nKt * p0->in_slot, sizeof(uint16_t));
        if (sc->a_buf) { bfs_scatter_A(p0, sc->a_buf, A); packed = sc->a_buf; }
        /* alloc failure -> packed stays NULL -> per-worker packing fallback */
    }

    pthread_t th[BFS_MAX_WORKERS];
    bfs_run_arg args[BFS_MAX_WORKERS];
    int joinable[BFS_MAX_WORKERS] = {0};
    for (int t = 0; t < sc->nt; t++) {
        args[t] = (bfs_run_arg){ s->fd[t], &sc->w[t], A, packed, B, C, M, N, 0, t };
        if (pthread_create(&th[t], NULL, bfs_run_worker, &args[t]) == 0) joinable[t] = 1;
    }
    for (int t = 0; t < sc->nt; t++) if (!joinable[t]) bfs_run_worker(&args[t]);
    int ret = 0;
    for (int t = 0; t < sc->nt; t++) {
        if (joinable[t]) pthread_join(th[t], NULL);
        if (args[t].ret) ret = args[t].ret;
    }
    return ret;
}
