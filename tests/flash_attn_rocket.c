// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * flash_attn_rocket.c — HW gate for masked grouped-query (flash) attention on the NPU
 * (rocket_flash_attn_fp16): the decoder / LLM-prefill attention sublayer that an
 * llama.cpp GGML_OP_FLASH_ATTN_EXT lowers to. Already-projected Q/K/V + an additive
 * causal (or sliding-window) mask -> per-head scale·QKᵀ -> mask -> softmax -> P·V, with
 * GQA (n_head > n_kv_heads). Validated against an fp64 attention oracle at Gemma-4-12B
 * shapes (head_dim 256, 16 q-heads, 8 kv-heads local / 1 kv-head global).
 *
 * Metrics: cosine similarity (the per-head criterion) + max abs vs the oracle. The score
 * round-trips through fp16, so the bar is the same cos>=0.9995 band as the other attention
 * gates. Off-device: ref self-check + SKIP.
 *
 * Usage: flash_attn_rocket                                  (sweep)
 *        flash_attn_rocket T n_kv head_dim n_head n_kv_heads (one shape, causal)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_attn.h"

static int g_fail = 0;

static void fill(_Float16 *v, size_t n, float amp, uint32_t seed)
{
    uint32_t st = 0x9e3779b9u ^ seed;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)((st >> 8) & 0xffff) / 65535.f;
        v[i] = (_Float16)((u * 2.f - 1.f) * amp);
    }
}

/* additive causal (+ optional sliding-window) mask [n_tokens][n_kv]; key j aligns to the
 * tail of the cache, i.e. query token t is key index (n_kv - n_tokens + t). window<=0 = full
 * causal. -INFINITY where masked (the kernel clamps it to a large negative before softmax). */
static void make_mask(_Float16 *m, int n_tokens, int n_kv, int window)
{
    for (int t = 0; t < n_tokens; t++) {
        int pos = n_kv - n_tokens + t;            /* this query's absolute position */
        for (int j = 0; j < n_kv; j++) {
            int keep = (j <= pos) && (window <= 0 || j > pos - window);
            m[(size_t)t * n_kv + j] = (_Float16)(keep ? 0.0f : -INFINITY);
        }
    }
}

/* c!=NULL exercises the persistent context (rocket_flash_attn_fp16_ctx); otherwise nthreads<=0
 * exercises the single-fd entry (the primitive oracle) and nthreads>0 the multi-core fan-out
 * (rocket_flash_attn_fp16_mt) — all numerically identical, so the same cos bar gates them. */
/* dh = key/query head dim (QK contraction); dv = value/output head dim. dv == dh is
 * standard GQA/MHA; dv != dh is MLA (e.g. DeepSeek dh=192, dv=128). */
static int test_fa(int fd, rocket_fa_ctx *c, int T, int n_kv, int dh, int dv, int nh, int nkvh,
                   int window, float softcap, int nthreads)
{
    const float scale = 1.0f / sqrtf((float)dh);
    size_t qn = (size_t)nh * T * dh, kn = (size_t)nkvh * n_kv * dh;
    size_t vn = (size_t)nkvh * n_kv * dv, on = (size_t)nh * T * dv, mn = (size_t)T * n_kv;
    _Float16 *Q = malloc(qn * sizeof(_Float16));
    _Float16 *K = malloc(kn * sizeof(_Float16));
    _Float16 *V = malloc(vn * sizeof(_Float16));
    _Float16 *M = malloc(mn * sizeof(_Float16));
    _Float16 *got = malloc(on * sizeof(_Float16)), *ref = malloc(on * sizeof(_Float16));
    if (!Q || !K || !V || !M || !got || !ref) { fprintf(stderr, "oom\n"); return 1; }
    fill(Q, qn, 1.0f, T * 7 + dh);   /* post-norm unit-scale activations */
    fill(K, kn, 1.0f, n_kv * 3 + 1);
    fill(V, vn, 1.0f, n_kv * 5 + 2);
    make_mask(M, T, n_kv, window);

    rocket_flash_attn_ref_fp16(T, n_kv, dh, dv, nh, nkvh, scale, softcap, Q, K, V, M, ref);
    int rc = c
        ? rocket_flash_attn_fp16_ctx(c, T, n_kv, dh, dv, nh, nkvh, scale, softcap, Q, K, V, M, got)
        : nthreads > 0
        ? rocket_flash_attn_fp16_mt(fd, T, n_kv, dh, dv, nh, nkvh, scale, softcap, Q, K, V, M, got, nthreads)
        : rocket_flash_attn_fp16   (fd, T, n_kv, dh, dv, nh, nkvh, scale, softcap, Q, K, V, M, got);

    char tag[144];
    snprintf(tag, sizeof tag, "fa T=%d n_kv=%d dh=%d dv=%d nh=%d nkvh=%d win=%d cap=%.0f %s=%d",
             T, n_kv, dh, dv, nh, nkvh, window, softcap, c ? "ctx" : "mt", nthreads);
    if (rc) { printf("  %s: call=%d -> FAIL\n", tag, rc);
              free(Q);free(K);free(V);free(M);free(got);free(ref); return 1; }

    double dot=0, ng=0, nr=0, max_abs=0, maxv=0;
    for (size_t i = 0; i < on; i++) {
        double g=(double)got[i], r=(double)ref[i];
        dot+=g*r; ng+=g*g; nr+=r*r;
        double ad=fabs(g-r); if(ad>max_abs)max_abs=ad; if(fabs(r)>maxv)maxv=fabs(r);
    }
    double cos = dot/(sqrt(ng)*sqrt(nr)+1e-30);
    int ok = (cos >= 0.9995) && (max_abs <= 0.02*maxv + 1e-3);
    printf("  %s: cos=%.6f max_abs=%.4g (maxv=%.3g) -> %s\n", tag, cos, max_abs, maxv, ok?"PASS":"FAIL");
    free(Q);free(K);free(V);free(M);free(got);free(ref);
    return ok?0:1;
}

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec*1e3 + (double)ts.tv_nsec*1e-6; }

/* Time the persistent-ctx flash-attn path at one shape under the CURRENTLY-SET env (the caller
 * toggles ROCKET_FA_TILE_KV between calls to compare materialized vs tiled). Writes the last
 * output into `out` (nh*T*dh elems) for a tiled-vs-materialized cross-check. Returns the min
 * ms/call over `iters` warm iterations (min = least host/LPDDR contention), or -1 on error. */
static double bench_ctx(int T,int n_kv,int dh,int nh,int nkvh,int window,int nthreads,int iters,_Float16 *out)
{
    const float scale=1.0f/sqrtf((float)dh);
    size_t qn=(size_t)nh*T*dh, kn=(size_t)nkvh*n_kv*dh, mn=(size_t)T*n_kv;
    _Float16 *Q=malloc(qn*sizeof(_Float16)),*K=malloc(kn*sizeof(_Float16)),*V=malloc(kn*sizeof(_Float16));
    _Float16 *M=malloc(mn*sizeof(_Float16));
    if(!Q||!K||!V||!M){fprintf(stderr,"oom\n");free(Q);free(K);free(V);free(M);return -1;}
    fill(Q,qn,1.0f,7);fill(K,kn,1.0f,11);fill(V,kn,1.0f,13);make_mask(M,T,n_kv,window);
    rocket_fa_ctx *c=rocket_fa_ctx_create(nthreads);
    if(!c){free(Q);free(K);free(V);free(M);return -1;}
    for(int i=0;i<2;i++) rocket_flash_attn_fp16_ctx(c,T,n_kv,dh,dh,nh,nkvh,scale,0.0f,Q,K,V,M,out); /* warm */
    double best=1e30;
    for(int i=0;i<iters;i++){
        double t0=now_ms();
        rocket_flash_attn_fp16_ctx(c,T,n_kv,dh,dh,nh,nkvh,scale,0.0f,Q,K,V,M,out);
        double dt=now_ms()-t0; if(dt<best)best=dt;
    }
    rocket_fa_ctx_free(c);
    free(Q);free(K);free(V);free(M);
    return best;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd);

    /* bench T n_kv [Kb=2048] [iters=5] : time the _ctx path materialized vs tiled, Gemma-4-12B
     * local-layer shape (head_dim 256, 16 q / 8 kv heads, 5 workers), + a cross-check cos. */
    if (argc >= 4 && strcmp(argv[1], "bench") == 0) {
        if (fd < 0) { printf("==== SKIP ====\n"); return 2; }
        int T=atoi(argv[2]), n_kv=atoi(argv[3]);
        int Kb=argc>4?atoi(argv[4]):2048, iters=argc>5?atoi(argv[5]):5;
        int dh=256,nh=16,nkvh=8,window=0,nthreads=5;
        size_t qn=(size_t)nh*T*dh;
        _Float16 *om=malloc(qn*sizeof(_Float16)), *ot=malloc(qn*sizeof(_Float16));
        char kbs[32]; snprintf(kbs,sizeof kbs,"%d",Kb);
        unsetenv("ROCKET_FA_TILE_KV");
        double tm=bench_ctx(T,n_kv,dh,nh,nkvh,window,nthreads,iters,om);
        setenv("ROCKET_FA_TILE_MIN_KV","256",1); setenv("ROCKET_FA_TILE_KV",kbs,1);
        double tt=bench_ctx(T,n_kv,dh,nh,nkvh,window,nthreads,iters,ot);
        double dot=0,na=0,nb=0; for(size_t i=0;i<qn;i++){double a=om[i],b=ot[i];dot+=a*b;na+=a*a;nb+=b*b;}
        double cos=dot/(sqrt(na)*sqrt(nb)+1e-30);
        printf("bench T=%d n_kv=%d dh=%d nh=%d nkvh=%d nthr=%d iters=%d\n",T,n_kv,dh,nh,nkvh,nthreads,iters);
        printf("  materialized : %8.2f ms/call\n", tm);
        printf("  tiled Kb=%-5d: %8.2f ms/call  (%.2fx vs mat)  cos(tiled,mat)=%.6f\n", Kb, tt, tm/tt, cos);
        free(om);free(ot); rocket_close(fd);
        return 0;
    }

    if (argc == 6) {   /* T n_kv head_dim n_head n_kv_heads (GQA: dv = head_dim) */
        int dh = atoi(argv[3]);
        if (fd>=0) g_fail |= test_fa(fd, NULL, atoi(argv[1]), atoi(argv[2]), dh, dh,
                                     atoi(argv[4]), atoi(argv[5]), 0, 0.0f, 0);
        if (fd>=0) rocket_close(fd);
        printf("==== %s ====\n", g_fail?"FAIL":"PASS");
        return g_fail?1:(fd<0?2:0);
    }

    if (fd >= 0) {
        /* Gemma-4-12B: head_dim 256, 16 q-heads. Local layers: 8 kv-heads (GQA 2:1),
         * sliding window 1024. Global layers: 1 kv-head (MQA), full causal. */
        g_fail |= test_fa(fd, NULL, 128, 128, 256, 256, 16, 8, 0,    0.0f, 0);  /* local-shape, full causal */
        g_fail |= test_fa(fd, NULL, 128, 128, 256, 256, 16, 8, 64,   0.0f, 0);  /* sliding window < n_kv     */
        g_fail |= test_fa(fd, NULL, 128, 128, 256, 256, 16, 1, 0,    0.0f, 0);  /* global layer: MQA         */
        g_fail |= test_fa(fd, NULL, 100, 100, 256, 256, 16, 8, 0,    0.0f, 0);  /* T%4!=0, n_kv%32!=0        */
        g_fail |= test_fa(fd, NULL, 64,  512, 256, 256, 16, 8, 0,    0.0f, 0);  /* n_kv>T (cached prefix)    */
        g_fail |= test_fa(fd, NULL, 128, 128, 256, 256, 16, 8, 0,   50.0f, 0);  /* logit soft-cap path       */

        /* MLA (dv != dh): DeepSeek-V2-Lite shape — dh=192 (128 nope + 64 rope), dv=128, and
         * MLA is MQA-like (n_kv_heads=1 on the compressed latent). Forces the materialized
         * per-head path (chaining/tiling gate to dv==dh). Covers single-fd, mt fan-out, the
         * persistent ctx, the unaligned/pad path, sliding window, and n_kv>T (cached prefix). */
        g_fail |= test_fa(fd, NULL, 128, 128, 192, 128, 16, 1, 0,    0.0f, 0);  /* MLA single-fd            */
        g_fail |= test_fa(fd, NULL, 128, 128, 192, 128, 16, 1, 0,    0.0f, 5);  /* MLA mt fan-out           */
        g_fail |= test_fa(fd, NULL, 100, 100, 192, 128, 16, 1, 0,    0.0f, 5);  /* MLA unaligned T/n_kv     */
        g_fail |= test_fa(fd, NULL, 128, 128, 192, 128, 16, 1, 64,   0.0f, 5);  /* MLA sliding window       */
        g_fail |= test_fa(fd, NULL, 64,  512, 192, 128, 16, 1, 0,    0.0f, 5);  /* MLA n_kv>T (cached)      */
        rocket_fa_ctx *cm = rocket_fa_ctx_create(5);
        if (!cm) { printf("  fa MLA ctx: create FAILED\n"); g_fail = 1; }
        else {
            g_fail |= test_fa(fd, cm, 128, 128, 192, 128, 16, 1, 0,  0.0f, 5);  /* MLA persistent ctx       */
            g_fail |= test_fa(fd, cm, 100, 100, 192, 128, 16, 1, 0,  0.0f, 5);  /* MLA ctx reuse, unaligned */
            rocket_fa_ctx_free(cm);
        }

        /* multi-core fan-out: same shapes, heads split across worker fds. nthreads=5 is the
         * backend default; the GQA/MQA cases check a worker's head range maps to the right kv
         * head (h/gqa), and 100/100 keeps the pad path. nthreads=3 splits 16 unevenly (6/5/5). */
        g_fail |= test_fa(fd, NULL, 128, 128, 256, 256, 16, 8, 0,    0.0f, 5);  /* GQA 2:1, 5 workers       */
        g_fail |= test_fa(fd, NULL, 128, 128, 256, 256, 16, 1, 0,    0.0f, 5);  /* MQA, 5 workers           */
        g_fail |= test_fa(fd, NULL, 128, 128, 256, 256, 16, 8, 64,   0.0f, 3);  /* window, uneven 3-way     */
        g_fail |= test_fa(fd, NULL, 100, 100, 256, 256, 16, 8, 0,    0.0f, 5);  /* unaligned T/n_kv, mt     */
        g_fail |= test_fa(fd, NULL, 64,  512, 256, 256, 16, 8, 0,    0.0f, 8);  /* n_kv>T, more fds than... */
        g_fail |= test_fa(fd, NULL, 64, 8192, 256, 256, 16, 8, 0,    0.0f, 5);  /* LONG context (n_kv past the
                                                                        offload gate): big score
                                                                        matrix + softmax over 8192 */

        /* persistent context: the same shapes through one reused rocket_fa_ctx (worker fds +
         * resident scratch held across calls — the backend's per-layer reuse). The repeated
         * calls also exercise the grow-only scratch (n_kv jumps 128->8192) and reuse-at-same
         * shape. Numerically identical to _mt, so the same cos bar gates it. */
        rocket_fa_ctx *c = rocket_fa_ctx_create(5);
        if (!c) { printf("  fa ctx: create FAILED\n"); g_fail = 1; }
        else {
            g_fail |= test_fa(fd, c, 128, 128, 256, 256, 16, 8, 0,    0.0f, 5);  /* GQA 2:1            */
            g_fail |= test_fa(fd, c, 128, 128, 256, 256, 16, 1, 0,    0.0f, 5);  /* MQA                */
            g_fail |= test_fa(fd, c, 128, 128, 256, 256, 16, 8, 64,   0.0f, 5);  /* sliding window     */
            g_fail |= test_fa(fd, c, 100, 100, 256, 256, 16, 8, 0,    0.0f, 5);  /* unaligned T/n_kv   */
            g_fail |= test_fa(fd, c, 64, 8192, 256, 256, 16, 8, 0,    0.0f, 5);  /* grow to long ctx   */
            g_fail |= test_fa(fd, c, 128, 128, 256, 256, 16, 8, 0,    0.0f, 5);  /* shrink-reuse scratch */
            rocket_fa_ctx_free(c);
        }

        /* ---- chained long-context path (ROCKET_FA_CHAIN_ELEMS) ----
         * Force a high chain budget so a worker's whole head range batches into one QK + one AV
         * job even at long context (the default 32M already does this up to ~20K; this pins it
         * independent of the default). Bit-identical to the per-head path, so the same
         * cos>=0.9995 bar vs the fp64 oracle; Tp kept modest so the oracle stays quick. */
        setenv("ROCKET_FA_CHAIN_ELEMS", "134217728", 1);                   /* 128M: batch all heads */
        g_fail |= test_fa(fd, NULL, 128, 16384, 256, 256, 16, 8, 0,    0.0f, 0); /* chained, single-fd, 16 heads/job */
        g_fail |= test_fa(fd, NULL, 128, 16384, 256, 256, 16, 8, 0,    0.0f, 5); /* chained, mt 3-4 heads/worker     */
        g_fail |= test_fa(fd, NULL, 128, 16384, 256, 256, 16, 8, 2048, 0.0f, 5); /* chained + sliding window         */
        unsetenv("ROCKET_FA_CHAIN_ELEMS");                                  /* restore default budget */

        /* ---- online/tiled long-context path (ROCKET_FA_TILE_KV) ----
         * Force the tiled path on at a small (512-key) tile so several tiles cover each shape,
         * and re-gate the SAME cos>=0.9995 bar vs the fp64 oracle. The window cases are the
         * load-bearing ones: a tile wholly outside a query row's visible range must be skipped
         * PER ROW (else its masked keys would wrongly weight); window>tile with a tall T makes a
         * tile visible for some rows and fully masked for others in the same AV batch. The
         * setenv mid-process exercises the read-fresh knob. */
        setenv("ROCKET_FA_TILE_MIN_KV", "256", 1);
        setenv("ROCKET_FA_TILE_KV", "512", 1);                              /* 512-key tiles */
        g_fail |= test_fa(fd, NULL, 128, 2048, 256, 256, 16, 8, 0,    0.0f, 0);  /* causal, 4 tiles, single-fd  */
        g_fail |= test_fa(fd, NULL, 512, 2048, 256, 256, 16, 8, 640,  0.0f, 0);  /* window>tile: per-row skip   */
        g_fail |= test_fa(fd, NULL, 128, 2048, 256, 256, 16, 1, 0,    0.0f, 5);  /* MQA, tiled, mt fan-out      */
        g_fail |= test_fa(fd, NULL, 100, 2000, 256, 256, 16, 8, 0,    0.0f, 5);  /* unaligned T/n_kv, tiled mt  */
        g_fail |= test_fa(fd, NULL, 96,  600,  256, 256, 16, 8, 0,    0.0f, 0);  /* short last tile (600%512=88)*/
        g_fail |= test_fa(fd, NULL, 64,  8192, 256, 256, 16, 8, 0,    0.0f, 5);  /* long ctx, 16 tiles, tiled mt*/
        rocket_fa_ctx *ct = rocket_fa_ctx_create(5);                        /* tiled persistent ctx */
        if (!ct) { printf("  fa tiled ctx: create FAILED\n"); g_fail = 1; }
        else {
            g_fail |= test_fa(fd, ct, 128, 2048, 256, 256, 16, 8, 0,   0.0f, 5);
            g_fail |= test_fa(fd, ct, 64,  8192, 256, 256, 16, 8, 0,   0.0f, 5); /* grow across shapes */
            g_fail |= test_fa(fd, ct, 128, 2048, 256, 256, 16, 8, 0,   0.0f, 5); /* reuse-at-shape     */
            rocket_fa_ctx_free(ct);
        }
        unsetenv("ROCKET_FA_TILE_KV");                                      /* back to materialized */

        rocket_close(fd);
    }

    printf("==== %s ====\n", g_fail?"FAIL":"PASS");
    return g_fail?1:(fd<0?2:0);
}
