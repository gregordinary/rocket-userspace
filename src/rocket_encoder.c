// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_encoder.c — one Whisper/transformer encoder block, assembled from the validated on-NPU
 * primitives (LayerNorm, multi-head attention, the residual add, the projection matmuls, and the
 * 2-pass GELU). Runs FULLY on the NPU — the MLP's GELU is the on-NPU 2-pass x·Φ(x). See
 * rocket_encoder.h.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "rocket_encoder.h"
#include "rocket_norm.h"        /* rocket_layernorm_fp16 */
#include "rocket_attn.h"        /* rocket_mha_self_fp16  */
#include "rocket_matmul.h"      /* rocket_matmul_fp16 (C=A·B^T) */
#include "rocket_activation.h"  /* rocket_ew_add_fp16 (residual) */
#include "rocket_ffn.h"         /* rocket_mlp_fp16_fused (cross-op cube-resident encoder MLP) */

static double gelu_d(double x) { return 0.5 * x * (1.0 + erf(x * M_SQRT1_2)); }

void rocket_encoder_block_ref_fp16(int T, int d, int n_head, int d_ff,
                               const _Float16 *x,
                               const _Float16 *ln1_g, const _Float16 *ln1_b,
                               const _Float16 *Wq, const _Float16 *bq,
                               const _Float16 *Wk, const _Float16 *bk,
                               const _Float16 *Wv, const _Float16 *bv,
                               const _Float16 *Wo, const _Float16 *bo,
                               const _Float16 *ln2_g, const _Float16 *ln2_b,
                               const _Float16 *Wf1, const _Float16 *bf1,
                               const _Float16 *Wf2, const _Float16 *bf2,
                               float eps, _Float16 *out)
{
    size_t Td = (size_t)T*d;
    _Float16 *ln = malloc(Td*sizeof(_Float16)), *attn = malloc(Td*sizeof(_Float16));
    _Float16 *xa = malloc(Td*sizeof(_Float16));
    double *f1 = malloc((size_t)d_ff*sizeof(double));
    /* x = x + MHA(LN1(x)) */
    rocket_layernorm_ref_fp16(T,d,x,ln1_g,ln1_b,eps,ln);
    rocket_mha_self_ref_fp16(T,d,n_head,ln,Wq,bq,Wk,bk,Wv,bv,Wo,bo,attn);
    for (size_t i=0;i<Td;i++) xa[i]=(_Float16)((double)x[i]+(double)attn[i]);
    /* x = x + MLP(LN2(x)) */
    rocket_layernorm_ref_fp16(T,d,xa,ln2_g,ln2_b,eps,ln);
    for (int t=0;t<T;t++){
        const _Float16 *hr = ln+(size_t)t*d;
        for (int o=0;o<d_ff;o++){ double a=0; const _Float16 *w=Wf1+(size_t)o*d; for(int i=0;i<d;i++) a+=(double)hr[i]*(double)w[i]; f1[o]=gelu_d(a+(bf1?(double)bf1[o]:0)); }
        _Float16 *orow = out+(size_t)t*d;
        for (int o=0;o<d;o++){ double a=0; const _Float16 *w=Wf2+(size_t)o*d_ff; for(int i=0;i<d_ff;i++) a+=f1[i]*(double)w[i]; orow[o]=(_Float16)((double)xa[(size_t)t*d+o]+a+(bf2?(double)bf2[o]:0)); }
    }
    free(ln);free(attn);free(xa);free(f1);
}

/* add bias b[N] broadcast over M rows (host glue) */
static void add_bias(_Float16 *C, const _Float16 *b, int M, int N)
{
    if (!b) return;
    for (int m=0;m<M;m++){ _Float16 *r=C+(size_t)m*N; for(int n=0;n<N;n++) r[n]=(_Float16)((float)r[n]+(float)b[n]); }
}

int rocket_encoder_block_fp16(int fd, int T, int d, int n_head, int d_ff,
                               const _Float16 *x,
                               const _Float16 *ln1_g, const _Float16 *ln1_b,
                               const _Float16 *Wq, const _Float16 *bq,
                               const _Float16 *Wk, const _Float16 *bk,
                               const _Float16 *Wv, const _Float16 *bv,
                               const _Float16 *Wo, const _Float16 *bo,
                               const _Float16 *ln2_g, const _Float16 *ln2_b,
                               const _Float16 *Wf1, const _Float16 *bf1,
                               const _Float16 *Wf2, const _Float16 *bf2,
                               float eps, _Float16 *out)
{
    if (T<1||d<1||n_head<1||d_ff<1||d%n_head) return -1;
    if (fd<0) { rocket_encoder_block_ref_fp16(T,d,n_head,d_ff,x,ln1_g,ln1_b,Wq,bq,Wk,bk,Wv,bv,Wo,bo,ln2_g,ln2_b,Wf1,bf1,Wf2,bf2,eps,out); return 0; }

    const int Tp = (T+3)&~3;                        /* matmul M%4 */
    /* Tp*d_ff is passed as int to the GELU/EW ops below; reject a truncating product. */
    if ((size_t)Tp * d_ff > INT_MAX) return -1;
    const size_t Td = (size_t)T*d;
    int rc = -2;
    _Float16 *ln=NULL,*attn=NULL,*xa=NULL,*hp=NULL,*f1=NULL,*act=NULL,*f2=NULL;

    ln   = malloc(Td*sizeof(_Float16));
    attn = malloc(Td*sizeof(_Float16));
    xa   = malloc(Td*sizeof(_Float16));
    hp   = calloc((size_t)Tp*d, sizeof(_Float16));         /* LN2 output, row-padded for matmul */
    f1   = malloc((size_t)Tp*d_ff*sizeof(_Float16));
    act  = malloc((size_t)Tp*d_ff*sizeof(_Float16));
    f2   = malloc((size_t)Tp*d*sizeof(_Float16));
    if (!ln||!attn||!xa||!hp||!f1||!act||!f2) goto out;

    /* --- attention sublayer: x = x + MHA(LN1(x)) --- */
    if ((rc = rocket_layernorm_fp16(fd,T,d,x,ln1_g,ln1_b,eps,ln)) != 0) goto out;
    if ((rc = rocket_mha_self_fp16(fd,T,d,n_head,ln,Wq,bq,Wk,bk,Wv,bv,Wo,bo,attn)) != 0) goto out;
    if ((rc = rocket_ew_add_fp16(fd,x,attn,xa,(int)Td)) != 0) goto out;

    /* --- feed-forward sublayer: x = x + MLP(LN2(x)) --- */
    if ((rc = rocket_layernorm_fp16(fd,T,d,xa,ln2_g,ln2_b,eps,ln)) != 0) goto out;
    memcpy(hp, ln, Td*sizeof(_Float16));                   /* pad rows Tp-T are zero */

    /* f2 = GELU(LN2(x)·Wf1^T + bf1)·Wf2^T.
     *
     * CROSS-OP CUBE-RESIDENT path (default): rocket_mlp_fp16_fused keeps the [Tp,d_ff]
     * intermediate in NPU cube layout across fc1 -> +bf1 -> GELU -> fc2 — the bf1 add and the
     * 2-pass GELU run element-wise on the cube and fc2 reads it directly, so the host never
     * de-tiles/re-tiles the activation. This is the lever for the TRANSFORM-BOUND encoder
     * (SigLIP/Whisper ~80% transform-to-compute). For shapes it can't tiling-match it falls
     * back internally to the host-handoff fc1/GELU/fc2 below. ROCKET_ENCODER_NOFUSE forces the
     * host-handoff path (A/B). The pad rows (T..Tp) are matmul-of-zero and never read out. */
    if (!getenv("ROCKET_ENCODER_NOFUSE")) {
        if ((rc = rocket_mlp_fp16_fused(fd, Tp, d, d_ff, d, hp, Wf1, bf1, Wf2,
                                        ROCKET_ACTIVATION_GELU, f2)) != 0) goto out;
    } else {
        /* host-handoff: f1 = LN2(x)·Wf1^T + bf1; act = GELU(f1) (2-pass: Φ-gate then f1·Φ);
         * f2 = act·Wf2^T (each op de-tiles to row-major and re-scatters). */
        if ((rc = rocket_matmul_fp16(fd, Tp, d, d_ff, hp, Wf1, f1)) != 0) goto out;
        add_bias(f1, bf1, T, d_ff);
        if ((rc = rocket_activation_fp16(fd, ROCKET_ACTIVATION_GELU_GATE, f1, act, Tp*d_ff)) != 0) goto out;
        if ((rc = rocket_ew_mul_fp16(fd, f1, act, act, Tp*d_ff)) != 0) goto out;
        if ((rc = rocket_matmul_fp16(fd, Tp, d_ff, d, act, Wf2, f2)) != 0) goto out;
    }
    add_bias(f2, bf2, T, d);

    /* out = xa + f2  (NPU residual add over the real T rows) */
    {
        _Float16 *f2c = malloc(Td*sizeof(_Float16));
        if (!f2c) { rc = -2; goto out; }
        for (int t=0;t<T;t++) memcpy(f2c+(size_t)t*d, f2+(size_t)t*d, (size_t)d*sizeof(_Float16));
        rc = rocket_ew_add_fp16(fd, xa, f2c, out, (int)Td);
        free(f2c);
    }

out:
    free(f2);free(act);free(f1);free(hp);free(xa);free(attn);free(ln);
    return rc;
}
