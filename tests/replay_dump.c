// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * replay_dump.c — replay the EXACT failing in-context matmul captured by the
 * ggml-rocket backend (ROCKET_DUMP=1 -> /tmp/rk_dump.bin).
 *
 * The dump (all _Float16) is: hdr[4]={M,K,N,Mp}, A16[Mp*K], B[N*K],
 * Cctx_pp[Mp*N] (in-context prepacked), Cctx_mt[Mp*N] (in-context mt).
 *
 * We recompute prepacked (shared + non-shared A-pack) and mt vs a CPU fp32 ref
 * over ALL rows, and report WHERE the divergence is (row/col), so the prepacked
 * bug (data-specific, only with real weights) can be localized and fixed.
 *
 * Build: test foreach in CMakeLists.txt.  Run: sudo ./replay_dump
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_matmul.h"

static int g_M, g_N, g_Mp, g_K;
static const float *g_ref;

/* max |C-ref| over all Mp rows, with location + per-row bad count */
static void report(const char *tag, const _Float16 *C)
{
    double gmax = 0; int gr = 0, gc = 0;
    int bad_rows = 0;
    for (int m = 0; m < g_Mp; m++) {
        double rmax = 0;
        for (int n = 0; n < g_N; n++) {
            double d = fabs((double)C[(size_t)m*g_N+n] - (double)g_ref[(size_t)m*g_N+n]);
            if (d > rmax) rmax = d;
            if (d > gmax) { gmax = d; gr = m; gc = n; }
        }
        if (rmax > 1.0) bad_rows++;
    }
    printf("  %-22s max_abs=%10.2f at (row %d,col %d)  bad_rows(>1)=%d/%d\n",
           tag, gmax, gr, gc, bad_rows, g_Mp);
    /* per-row max for the first 16 rows */
    printf("      per-row max:");
    for (int m = 0; m < g_Mp && m < 16; m++) {
        double rmax = 0;
        for (int n = 0; n < g_N; n++) {
            double d = fabs((double)C[(size_t)m*g_N+n] - (double)g_ref[(size_t)m*g_N+n]);
            if (d > rmax) rmax = d;
        }
        printf(" %.0f", rmax);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    const char *path = argc>=2 ? argv[1] : "/tmp/rk_dump.bin";
    FILE *f = fopen(path,"rb");
    if (!f) { fprintf(stderr,"cannot open %s\n",path); return 1; }
    int hdr[4];
    if (fread(hdr,sizeof(int),4,f) != 4) { fprintf(stderr,"replay: short read (header)\n"); fclose(f); return 1; }
    int M=hdr[0], K=hdr[1], N=hdr[2], Mp=hdr[3];
    g_M=M; g_N=N; g_Mp=Mp; g_K=K;
    printf("replay M=%d K=%d N=%d Mp=%d\n", M,K,N,Mp);

    _Float16 *A=malloc((size_t)Mp*K*2), *B=malloc((size_t)N*K*2);
    _Float16 *Cpp=malloc((size_t)Mp*N*2), *Cmt=malloc((size_t)Mp*N*2);
    size_t rd  = fread(A,2,(size_t)Mp*K,f);    /* separate statements: freads must stay sequential */
    rd        += fread(B,2,(size_t)N*K,f);
    rd        += fread(Cpp,2,(size_t)Mp*N,f);
    rd        += fread(Cmt,2,(size_t)Mp*N,f);
    if (rd != (size_t)Mp*K + (size_t)N*K + 2*(size_t)Mp*N) {
        fprintf(stderr,"replay: short read (data)\n"); fclose(f); return 1;
    }
    fclose(f);

    float *ref=malloc((size_t)Mp*N*sizeof(float)); g_ref=ref;
    for (int m=0;m<Mp;m++) for(int n=0;n<N;n++){
        float a=0; for(int k=0;k<K;k++) a+=(float)A[(size_t)m*K+k]*(float)B[(size_t)n*K+k];
        ref[(size_t)m*N+n]=a;
    }

    printf("\n-- in-context (from dump) --\n");
    report("dump prepacked", Cpp);
    report("dump mt",        Cmt);

    _Float16 *Cr=malloc((size_t)Mp*N*2);

    printf("\n-- standalone recompute (same bytes) --\n");
    rocket_matmul_fp16_mt(Mp,K,N,A,B,Cr,4);              report("mt", Cr);

    unsetenv("ROCKET_NO_SHARED_PACK");
    rocket_ctx *c1=rocket_ctx_create(4);
    rocket_weights *w1=rocket_weights_pack(c1,Mp,K,N,B);
    rocket_matmul_fp16_prepacked(c1,Mp,K,N,A,Cr,w1);     report("prepacked SHARED", Cr);
    rocket_weights_free(c1,w1); rocket_ctx_free(c1);

    setenv("ROCKET_NO_SHARED_PACK","1",1);
    rocket_ctx *c2=rocket_ctx_create(4);
    rocket_weights *w2=rocket_weights_pack(c2,Mp,K,N,B);
    rocket_matmul_fp16_prepacked(c2,Mp,K,N,A,Cr,w2);     report("prepacked NOSHARE", Cr);
    rocket_weights_free(c2,w2); rocket_ctx_free(c2);

    /* single-thread (one fd, one N-slice -> exercises full N in one plan) */
    setenv("ROCKET_NO_SHARED_PACK","1",1);
    rocket_ctx *c3=rocket_ctx_create(1);
    rocket_weights *w3=rocket_weights_pack(c3,Mp,K,N,B);
    rocket_matmul_fp16_prepacked(c3,Mp,K,N,A,Cr,w3);     report("prepacked T=1 NOSHARE", Cr);
    rocket_weights_free(c3,w3); rocket_ctx_free(c3);

    free(A);free(B);free(Cpp);free(Cmt);free(ref);free(Cr);
    return 0;
}
