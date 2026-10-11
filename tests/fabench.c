// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * fabench — primitive-level A/B for the flash-attention QK/AV submit-chaining lever.
 * Times rocket_flash_attn_fp16_ctx (the backend's per-layer path), by default at a Gemma-4
 * prefill shape (head_dim 256, 16 q-heads, 8 kv-heads, causal mask), warm, over `iters` reps. Run twice —
 * ROCKET_FA_CHAIN=0 then 1 — to isolate the per-worker QK/AV batching; the chained path
 * also reuses the worker's resident batched-matmul context (rocket_mm_batch), so this
 * also measures the persistent-scratch reclaim. A tool, not a CTest gate (correctness is
 * flash_attn_rocket); prints mean + best ms.
 *
 * Usage: fabench [n_tokens] [n_kv] [nthreads] [iters] [head_dim] [n_head] [n_kv_heads] [mask]
 *        (defaults 512 2048 5 30 256 16 8 1). mask 0 runs unmasked, as a ViT encoder calls it:
 *        Depth Anything v2-small's layer is `fabench 1370 1370 3 30 64 6 6 0`.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "rocket_npu.h"
#include "rocket_attn.h"

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6; }
static void fill(_Float16 *v, size_t n, float amp, uint32_t s){
    uint32_t st=0x9e3779b9u^s; for(size_t i=0;i<n;i++){ st=st*1664525u+1013904223u;
        float u=(float)((st>>8)&0xffff)/65535.f; v[i]=(_Float16)((u*2.f-1.f)*amp);} }

int main(int argc, char **argv){
    int T    = argc>1 ? atoi(argv[1]) : 512;
    int n_kv = argc>2 ? atoi(argv[2]) : 2048;
    int nth  = argc>3 ? atoi(argv[3]) : 5;
    int iters= argc>4 ? atoi(argv[4]) : 30;
    const int dh  = argc>5 ? atoi(argv[5]) : 256;
    const int nh  = argc>6 ? atoi(argv[6]) : 16;
    const int nkvh= argc>7 ? atoi(argv[7]) : 8;
    const int use_mask = argc>8 ? atoi(argv[8]) : 1;
    if(dh<=0||nh<=0||nkvh<=0||nh%nkvh){printf("bad head geometry\n");return 1;}
    const float scale=1.f/sqrtf((float)dh), softcap=0.f;
    size_t qn=(size_t)nh*T*dh, kn=(size_t)nkvh*n_kv*dh, mn=(size_t)T*n_kv;
    _Float16 *Q=malloc(qn*2),*K=malloc(kn*2),*V=malloc(kn*2),*M=malloc(mn*2),*O=malloc(qn*2);
    if(!Q||!K||!V||!M||!O){printf("oom\n");return 1;}
    fill(Q,qn,1.f,1); fill(K,kn,1.f,2); fill(V,kn,1.f,3);
    for(int t=0;t<T;t++){ int pos=n_kv-T+t; for(int j=0;j<n_kv;j++)
        M[(size_t)t*n_kv+j]=(_Float16)((j<=pos)?0.f:-INFINITY); }
    int fd=rocket_open(); if(fd<0){printf("no NPU -> SKIP\n");return 2;}
    rocket_fa_ctx *c=rocket_fa_ctx_create(nth);
    if(!c){printf("ctx create failed\n");return 1;}
    const char *ch=getenv("ROCKET_FA_CHAIN"); ch=ch?ch:"1 (default)";   /* the library defaults it on */
    /* Every call's rc is checked: a call that fails fast would otherwise time as a win. */
    int errs=0;
    const _Float16 *Mk = use_mask ? M : NULL;
    for(int i=0;i<8;i++) errs += rocket_flash_attn_fp16_ctx(c,T,n_kv,dh,dh,nh,nkvh,scale,softcap,Q,K,V,Mk,O)!=0; /* warm */
    double best=1e30, sum=0;
    for(int i=0;i<iters;i++){ double t0=now_ms();
        errs += rocket_flash_attn_fp16_ctx(c,T,n_kv,dh,dh,nh,nkvh,scale,softcap,Q,K,V,Mk,O)!=0;
        double dt=now_ms()-t0; sum+=dt; if(dt<best)best=dt; }
    if(errs){ printf("%d call(s) failed: the timing below is not a measurement\n",errs); }
    /* FABENCH_DUMP=path writes the last call's output, to compare two builds bit for bit. */
    const char *dp=getenv("FABENCH_DUMP");
    if(dp){ FILE *f=fopen(dp,"wb"); if(!f||fwrite(O,2,qn,f)!=qn){printf("dump failed\n");errs++;} if(f)fclose(f); }
    printf("T=%d n_kv=%d nth=%d dh=%d nh=%d nkvh=%d mask=%d chain=%s : mean=%.3f ms  best=%.3f ms  (%d iters)\n",
           T,n_kv,nth,dh,nh,nkvh,use_mask,ch,sum/iters,best,iters);
    rocket_fa_ctx_free(c); rocket_close(fd);
    free(Q);free(K);free(V);free(M);free(O); return errs ? 1 : 0;
}
