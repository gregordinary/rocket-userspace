// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * norm_vision_rocket.c — HW gate for the on-NPU VISION normalization family
 * (rocket_batchnorm/groupnorm/instancenorm/l2norm_fp16, rocket_normvision.h).
 *
 * Each op runs its O(elements) work on the NPU (square via ew_mul, the per-group reduce via
 * the stacked feature-reduce, the affine via ew_mul+ew_add / per-row scale) and its O(rows)
 * mean/var/rsqrt tail exact on the host, then is compared to an independent fp64 oracle. The
 * sweep covers the M/row-tile boundary, channels not a multiple of the matmul N-group, the
 * GroupNorm group counts (incl. G=C == InstanceNorm and G=1 == LayerNorm-over-CHW), spatial
 * P>1 and P==1 ([N,C] tensors), the no-affine (gamma/beta NULL) case, and a large-magnitude
 * input that exercises the fp16-square overflow prescale.
 *
 * Usage: norm_vision_rocket        (full sweep)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_normvision.h"

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

/* positive values for the variance (var) stat of BatchNorm */
static void fill_pos(_Float16 *v, size_t n, float lo, float hi, uint32_t seed)
{
    uint32_t st = 0x85ebca6bu ^ seed;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)((st >> 8) & 0xffff) / 65535.f;
        v[i] = (_Float16)(lo + u * (hi - lo));
    }
}

static int compare(const char *tag, const _Float16 *got, const _Float16 *ref,
                   size_t n, double rel_tol)
{
    double maxv = 0;
    for (size_t i = 0; i < n; i++) { double a = fabs((double)ref[i]); if (a > maxv) maxv = a; }
    const double abs_tol = rel_tol * maxv + 1e-4;
    double max_abs = 0, max_rel = 0; int bad = 0;
    for (size_t i = 0; i < n; i++) {
        double ad = fabs((double)got[i] - (double)ref[i]);
        double rd = ad / (fabs((double)ref[i]) + 1e-9);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (rd > rel_tol && ad > abs_tol) {
            if (bad < 5) printf("    [%zu] ref=%.5g got=%.5g d=%.4g\n",
                                i, (double)ref[i], (double)got[i], ad);
            bad++;
        }
    }
    int ok = (bad == 0);
    printf("  %s: maxv=%.4g max_abs=%.4g max_rel=%.2g bad=%d -> %s\n",
           tag, maxv, max_abs, max_rel, bad, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* ---- BatchNorm ---- */
static int test_bn(int fd, int N, int C, int P, float amp, int use_gamma, int use_beta)
{
    size_t T = (size_t)N * C * P;
    _Float16 *x = malloc(T*2), *g = malloc((size_t)C*2), *b = malloc((size_t)C*2);
    _Float16 *mean = malloc((size_t)C*2), *var = malloc((size_t)C*2);
    _Float16 *got = malloc(T*2), *ref = malloc(T*2);
    if (!x||!g||!b||!mean||!var||!got||!ref){fprintf(stderr,"oom\n");return 1;}
    fill(x, T, amp, (uint32_t)(N*7+C*131+P*17));
    for (int c=0;c<C;c++){ g[c]=(_Float16)(0.5f+((c*53%100)/99.f)); b[c]=(_Float16)(((c*29%100)/99.f)-0.5f); }
    fill(mean, C, amp*0.3f, 99u+C);
    fill_pos(var, C, 0.2f, 4.0f, 7u+C);         /* running var > 0 */
    const float eps = 1e-3f;
    rocket_batchnorm_ref_fp16(N,C,P,x, use_gamma?g:NULL, use_beta?b:NULL, mean,var, eps, ref);
    int rc = rocket_batchnorm_fp16(fd,N,C,P,x, use_gamma?g:NULL, use_beta?b:NULL, mean,var, eps, got);
    char tag[96]; snprintf(tag,sizeof tag,"batchnorm N=%d C=%d P=%d amp=%.0f g=%d b=%d",N,C,P,amp,use_gamma,use_beta);
    int fail = rc ? (printf("  %s: call=%d -> FAIL\n",tag,rc),1) : compare(tag,got,ref,T,0.03);
    free(x);free(g);free(b);free(mean);free(var);free(got);free(ref);
    return fail;
}

/* ---- GroupNorm / InstanceNorm ---- */
static int test_gn(int fd, int N, int C, int G, int P, float amp, int use_beta)
{
    size_t T = (size_t)N * C * P;
    _Float16 *x = malloc(T*2), *g = malloc((size_t)C*2), *b = malloc((size_t)C*2);
    _Float16 *got = malloc(T*2), *ref = malloc(T*2);
    if (!x||!g||!b||!got||!ref){fprintf(stderr,"oom\n");return 1;}
    fill(x, T, amp, (uint32_t)(N*7+C*131+G*13+P*17+(uint32_t)amp));
    for (int c=0;c<C;c++){ g[c]=(_Float16)(0.5f+((c*53%100)/99.f)); b[c]=(_Float16)(((c*29%100)/99.f)-0.5f); }
    const float eps = 1e-5f;
    const _Float16 *beta = use_beta ? b : NULL;
    rocket_groupnorm_ref_fp16(N,C,G,P,x,g,beta,eps,ref);
    int rc = rocket_groupnorm_fp16(fd,N,C,G,P,x,g,beta,eps,got);
    char tag[96]; snprintf(tag,sizeof tag,"groupnorm N=%d C=%d G=%d P=%d amp=%.0f beta=%d",N,C,G,P,amp,use_beta);
    int fail = rc ? (printf("  %s: call=%d -> FAIL\n",tag,rc),1) : compare(tag,got,ref,T,0.03);
    free(x);free(g);free(b);free(got);free(ref);
    return fail;
}

static int test_in(int fd, int N, int C, int P, float amp)
{
    size_t T = (size_t)N * C * P;
    _Float16 *x = malloc(T*2), *g = malloc((size_t)C*2), *b = malloc((size_t)C*2);
    _Float16 *got = malloc(T*2), *ref = malloc(T*2);
    if (!x||!g||!b||!got||!ref){fprintf(stderr,"oom\n");return 1;}
    fill(x, T, amp, (uint32_t)(N*3+C*101+P*19));
    for (int c=0;c<C;c++){ g[c]=(_Float16)(0.5f+((c*53%100)/99.f)); b[c]=(_Float16)(((c*29%100)/99.f)-0.5f); }
    const float eps = 1e-5f;
    rocket_instancenorm_ref_fp16(N,C,P,x,g,b,eps,ref);
    int rc = rocket_instancenorm_fp16(fd,N,C,P,x,g,b,eps,got);
    char tag[96]; snprintf(tag,sizeof tag,"instancenorm N=%d C=%d P=%d amp=%.0f",N,C,P,amp);
    int fail = rc ? (printf("  %s: call=%d -> FAIL\n",tag,rc),1) : compare(tag,got,ref,T,0.03);
    free(x);free(g);free(b);free(got);free(ref);
    return fail;
}

/* ---- L2-Normalize ---- */
static int test_l2(int fd, int M, int H, float amp)
{
    size_t T = (size_t)M * H;
    _Float16 *x = malloc(T*2), *got = malloc(T*2), *ref = malloc(T*2);
    if (!x||!got||!ref){fprintf(stderr,"oom\n");return 1;}
    fill(x, T, amp, (uint32_t)(M*131+H*17+(uint32_t)amp));
    const float eps = 1e-12f;
    rocket_l2norm_ref_fp16(M,H,x,eps,ref);
    int rc = rocket_l2norm_fp16(fd,M,H,x,eps,got);
    char tag[96]; snprintf(tag,sizeof tag,"l2norm M=%d H=%d amp=%.0f",M,H,amp);
    int fail = rc ? (printf("  %s: call=%d -> FAIL\n",tag,rc),1) : compare(tag,got,ref,T,0.03);
    free(x);free(got);free(ref);
    return fail;
}

/* host self-check: GroupNorm with gamma==1,beta==0 -> each group has mean~0, var~1 */
static int ref_selfcheck(void)
{
    const int N=2,C=8,G=2,P=16; size_t T=(size_t)N*C*P;
    _Float16 *x=malloc(T*2), *g=malloc((size_t)C*2), *o=malloc(T*2);
    if(!x||!g||!o){free(x);free(g);free(o);return 1;}
    fill(x,T,3.f,5); for(int c=0;c<C;c++) g[c]=(_Float16)1.0f;
    rocket_groupnorm_ref_fp16(N,C,G,P,x,g,NULL,0.f,o);
    int Cg=C/G,Pg=Cg*P,bad=0;
    for(int n=0;n<N;n++)for(int gg=0;gg<G;gg++){
        double sm=0,ss=0;
        for(int cc=0;cc<Cg;cc++){int c=gg*Cg+cc;for(int p=0;p<P;p++){double v=(double)o[((size_t)n*C+c)*P+p];sm+=v;ss+=v*v;}}
        double mean=sm/Pg,var=ss/Pg-mean*mean;
        if(fabs(mean)>0.02||fabs(var-1.0)>0.05)bad++;
    }
    free(x);free(g);free(o);
    printf("ref self-check (groupnorm out mean~0 var~1 when gamma==1): %s\n\n", bad?"FAIL":"PASS");
    return bad?1:0;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — ref self-check + SKIP\n\n", fd);

    g_fail |= ref_selfcheck();

    if (fd >= 0) {
        printf("== BatchNorm ==\n");
        g_fail |= test_bn(fd,  2,  64,  49, 4.f, 1, 1);   /* 7x7 map */
        g_fail |= test_bn(fd,  4, 128,   1, 4.f, 1, 1);   /* P==1, [N,C] */
        g_fail |= test_bn(fd,  1, 100, 196, 4.f, 1, 0);   /* C%32!=0, no beta */
        g_fail |= test_bn(fd,  2,  32, 256, 1000.f, 1, 1);/* large |x| */

        printf("== GroupNorm ==\n");
        g_fail |= test_gn(fd,  2,  32,  4, 196, 4.f, 1);  /* 4 groups, 14x14 */
        g_fail |= test_gn(fd,  2,  64,  1, 49,  4.f, 1);  /* G=1 == LayerNorm over CHW */
        g_fail |= test_gn(fd,  1, 128, 32, 64,  4.f, 0);  /* 32 groups, no beta */
        g_fail |= test_gn(fd,  3,  96,  3,  1,  4.f, 1);  /* P==1 */
        g_fail |= test_gn(fd,  2,  48,  4, 256, 1000.f, 1);/* large |x| prescale */

        printf("== InstanceNorm (G=C) ==\n");
        g_fail |= test_in(fd,  2,  64, 196, 4.f);         /* 14x14 per (n,c) */
        g_fail |= test_in(fd,  4,  32,  49, 4.f);

        printf("== L2-Normalize ==\n");
        g_fail |= test_l2(fd,  16,   64, 4.f);
        g_fail |= test_l2(fd, 256,  512, 4.f);            /* row-tile boundary */
        g_fail |= test_l2(fd,  12,  100, 4.f);            /* small M, H%32!=0 */
        g_fail |= test_l2(fd,  32, 1024, 1000.f);         /* large |x| prescale */

        rocket_close(fd);
    }

    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : (fd < 0 ? 2 : 0);
}
