// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * encoder_block_rocket.c — HW gate for one full Whisper/transformer encoder block on the NPU
 * (rocket_encoder_block_fp16): LN1 -> MHA -> residual -> LN2 -> MLP(GELU) -> residual.
 * step-4 capstone — proves the validated primitives compose into a complete pre-norm block.
 *
 * Validated against an fp64 block oracle (cosine similarity, the prompt's per-layer criterion,
 * + max abs/rel). Everything runs on the NPU except the MLP's elementwise GELU (the documented
 * host seam — the on-NPU GELU is the fused matmul->act epilogue, TRACK B). Off-device: SKIP.
 *
 * Usage: encoder_block_rocket            (sweep)
 *        encoder_block_rocket T d nhead  (one shape, d_ff=4d)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_encoder.h"

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

static int test_block(int fd, int T, int d, int nh, int d_ff)
{
    size_t Td=(size_t)T*d, dd=(size_t)d*d, df=(size_t)d_ff*d;
    /* inputs calloc'd (zero-init, then fill()) — keeps GCC -Wmaybe-uninitialized quiet:
     * the heuristic can't see through fill() and flags a different subset of these many
     * buffers on x86 vs aarch64. got/ref below are outputs (written by the calls), so
     * malloc is fine for them. */
    _Float16 *x=calloc(Td,sizeof(_Float16));
    _Float16 *g1=calloc(d,sizeof(_Float16)),*b1=calloc(d,sizeof(_Float16));
    _Float16 *Wq=calloc(dd,sizeof(_Float16)),*Wk=calloc(dd,sizeof(_Float16)),*Wv=calloc(dd,sizeof(_Float16)),*Wo=calloc(dd,sizeof(_Float16));
    _Float16 *bq=calloc(d,sizeof(_Float16)),*bv=calloc(d,sizeof(_Float16)),*bo=calloc(d,sizeof(_Float16));
    _Float16 *g2=calloc(d,sizeof(_Float16)),*b2=calloc(d,sizeof(_Float16));
    _Float16 *Wf1=calloc(df,sizeof(_Float16)),*bf1=calloc(d_ff,sizeof(_Float16));
    _Float16 *Wf2=calloc(df,sizeof(_Float16)),*bf2=calloc(d,sizeof(_Float16));
    _Float16 *got=malloc(Td*sizeof(_Float16)),*ref=malloc(Td*sizeof(_Float16));
    if(!x||!g1||!b1||!Wq||!Wk||!Wv||!Wo||!bq||!bv||!bo||!g2||!b2||!Wf1||!bf1||!Wf2||!bf2||!got||!ref){fprintf(stderr,"oom\n");return 1;}
    fill(x,Td,1.0f,T+d);
    for(int i=0;i<d;i++){ g1[i]=(_Float16)(0.8f+0.4f*((i*53%100)/99.f)); b1[i]=(_Float16)(0.1f*(((i*31%100)/99.f)-0.5f));
                          g2[i]=(_Float16)(0.8f+0.4f*((i*37%100)/99.f)); b2[i]=(_Float16)(0.1f*(((i*17%100)/99.f)-0.5f)); }
    fill(Wq,dd,0.05f,1);fill(Wk,dd,0.05f,2);fill(Wv,dd,0.05f,3);fill(Wo,dd,0.05f,4);
    fill(bq,d,0.1f,5);fill(bv,d,0.1f,6);fill(bo,d,0.1f,7);
    fill(Wf1,df,0.04f,8);fill(bf1,d_ff,0.1f,9);fill(Wf2,df,0.04f,10);fill(bf2,d,0.1f,11);
    const float eps=1e-5f;

    rocket_encoder_block_ref_fp16(T,d,nh,d_ff,x,g1,b1,Wq,bq,Wk,NULL,Wv,bv,Wo,bo,g2,b2,Wf1,bf1,Wf2,bf2,eps,ref);
    int rc = rocket_encoder_block_fp16(fd,T,d,nh,d_ff,x,g1,b1,Wq,bq,Wk,NULL,Wv,bv,Wo,bo,g2,b2,Wf1,bf1,Wf2,bf2,eps,got);

    char tag[80]; snprintf(tag,sizeof tag,"encblock T=%d d=%d nh=%d dff=%d",T,d,nh,d_ff);
    if (rc) { printf("  %s: call=%d -> FAIL\n",tag,rc); free(x);free(g1);free(b1);free(Wq);free(Wk);free(Wv);free(Wo);free(bq);free(bv);free(bo);free(g2);free(b2);free(Wf1);free(bf1);free(Wf2);free(bf2);free(got);free(ref); return 1; }

    double dot=0,ng=0,nr=0,max_abs=0,maxv=0,max_rel=0;
    for(size_t i=0;i<Td;i++){ double gg=(double)got[i],r=(double)ref[i]; dot+=gg*r;ng+=gg*gg;nr+=r*r; double ad=fabs(gg-r); if(ad>max_abs)max_abs=ad; if(fabs(r)>maxv)maxv=fabs(r); }
    for(size_t i=0;i<Td;i++){ double r=(double)ref[i]; if(fabs(r)>0.1*maxv){ double rd=fabs((double)got[i]-r)/(fabs(r)+1e-6); if(rd>max_rel)max_rel=rd; } }
    double cos=dot/(sqrt(ng)*sqrt(nr)+1e-30);
    int ok=(cos>=0.9995)&&(max_abs<=0.03*maxv+2e-3);
    printf("  %s: cos=%.6f max_abs=%.4g (maxv=%.3g) max_rel[big]=%.2g -> %s\n",tag,cos,max_abs,maxv,max_rel,ok?"PASS":"FAIL");
    free(x);free(g1);free(b1);free(Wq);free(Wk);free(Wv);free(Wo);free(bq);free(bv);free(bo);free(g2);free(b2);free(Wf1);free(bf1);free(Wf2);free(bf2);free(got);free(ref);
    return ok?0:1;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd);

    if (argc == 4) {
        int T=atoi(argv[1]),d=atoi(argv[2]),nh=atoi(argv[3]);
        if (fd>=0) g_fail |= test_block(fd,T,d,nh,4*d);
        if (fd>=0) rocket_close(fd);
        printf("==== %s ====\n",g_fail?"FAIL":"PASS");
        return g_fail?1:(fd<0?2:0);
    }

    if (fd >= 0) {
        g_fail |= test_block(fd,  64, 256, 4, 1024);   /* small block */
        g_fail |= test_block(fd,  64, 512, 8, 2048);   /* Whisper-base d_model/heads/d_ff */
        g_fail |= test_block(fd, 100, 512, 8, 2048);   /* T%16!=0 (key-pad path) */
        rocket_close(fd);
    }

    printf("==== %s ====\n",g_fail?"FAIL":"PASS");
    return g_fail?1:(fd<0?2:0);
}
