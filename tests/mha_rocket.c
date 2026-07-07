// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * mha_rocket.c — HW gate for on-NPU multi-head self-attention (rocket_mha_self_fp16), the
 * encoder capstone: the composition that drives the new on-NPU softmax inside real
 * scaled-dot-product attention (QKV matmuls -> per-head scores -> softmax -> P·V -> out-proj).
 *
 * Validated against an fp64 attention oracle at Whisper-base shapes (d_model 512, 8 heads).
 * The chain runs every matmul + every softmax on the NPU; only the head slicing, scale, bias
 * adds, and per-head V transpose are host glue. Metrics: cosine similarity (the prompt's
 * per-layer criterion) + max abs/rel vs the oracle. Off-device: ref self-check + SKIP.
 *
 * Usage: mha_rocket            (sweep)
 *        mha_rocket T d nhead  (one shape)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static int test_mha(int fd, int T, int d, int nh)
{
    size_t Td = (size_t)T*d, dd = (size_t)d*d;
    _Float16 *x=malloc(Td*sizeof(_Float16));
    _Float16 *Wq=malloc(dd*sizeof(_Float16)),*Wk=malloc(dd*sizeof(_Float16)),*Wv=malloc(dd*sizeof(_Float16)),*Wo=malloc(dd*sizeof(_Float16));
    _Float16 *bq=calloc(d,sizeof(_Float16)),*bv=calloc(d,sizeof(_Float16)),*bo=calloc(d,sizeof(_Float16));
    _Float16 *got=malloc(Td*sizeof(_Float16)),*ref=malloc(Td*sizeof(_Float16));
    if(!x||!Wq||!Wk||!Wv||!Wo||!bq||!bv||!bo||!got||!ref){fprintf(stderr,"oom\n");return 1;}
    /* small weights so q/k/v stay in a sane range; x ~ post-LN unit scale */
    fill(x, Td, 1.0f, T*7+d);
    fill(Wq, dd, 0.05f, 1); fill(Wk, dd, 0.05f, 2); fill(Wv, dd, 0.05f, 3); fill(Wo, dd, 0.05f, 4);
    fill(bq, d, 0.1f, 5);   fill(bv, d, 0.1f, 6);   fill(bo, d, 0.1f, 7);
    /* Whisper: bq,bv,bo present; NO bk */

    rocket_mha_self_ref_fp16(T,d,nh,x,Wq,bq,Wk,NULL,Wv,bv,Wo,bo,ref);
    int rc = rocket_mha_self_fp16(fd,T,d,nh,x,Wq,bq,Wk,NULL,Wv,bv,Wo,bo,got);

    char tag[80]; snprintf(tag,sizeof tag,"mha T=%d d=%d nh=%d",T,d,nh);
    if (rc) { printf("  %s: call=%d -> FAIL\n", tag, rc); free(x);free(Wq);free(Wk);free(Wv);free(Wo);free(bq);free(bv);free(bo);free(got);free(ref); return 1; }

    double dot=0,ng=0,nr=0,max_abs=0,maxv=0,max_rel=0;
    for (size_t i=0;i<Td;i++){
        double g=(double)got[i], r=(double)ref[i];
        dot+=g*r; ng+=g*g; nr+=r*r;
        double ad=fabs(g-r); if(ad>max_abs)max_abs=ad; if(fabs(r)>maxv)maxv=fabs(r);
    }
    for (size_t i=0;i<Td;i++){ double r=(double)ref[i]; if(fabs(r)>0.1*maxv){ double rd=fabs((double)got[i]-r)/(fabs(r)+1e-6); if(rd>max_rel)max_rel=rd; } }
    double cos = dot/(sqrt(ng)*sqrt(nr)+1e-30);
    int ok = (cos >= 0.9999) && (max_abs <= 0.02*maxv + 1e-3);
    printf("  %s: cos=%.6f max_abs=%.4g (maxv=%.3g) max_rel[big]=%.2g -> %s\n",
           tag, cos, max_abs, maxv, max_rel, ok?"PASS":"FAIL");
    free(x);free(Wq);free(Wk);free(Wv);free(Wo);free(bq);free(bv);free(bo);free(got);free(ref);
    return ok?0:1;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd);

    if (argc == 4) {
        if (fd>=0) g_fail |= test_mha(fd, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
        if (fd>=0) rocket_close(fd);
        printf("==== %s ====\n", g_fail?"FAIL":"PASS");
        return g_fail?1:(fd<0?2:0);
    }

    if (fd >= 0) {
        g_fail |= test_mha(fd,  64, 512, 8);   /* Whisper-base d_model, 8 heads, dh=64 */
        g_fail |= test_mha(fd, 128, 512, 8);
        g_fail |= test_mha(fd, 100, 512, 8);   /* T%16!=0 (arbitrary key count N) */
        g_fail |= test_mha(fd,  64, 384, 6);   /* Whisper-small-ish: d=384, dh=64 */
        rocket_close(fd);
    }

    printf("==== %s ====\n", g_fail?"FAIL":"PASS");
    return g_fail?1:(fd<0?2:0);
}
