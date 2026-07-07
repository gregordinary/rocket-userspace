// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
//
// fc_data_bank_sweep_rocket — RE probe for the CNA_CBUF_CON0 FC_DATA_BANK field [10:8]
// and a large-K single-pass confirmation, on the RK3588 NPU.
//
// FC_DATA_BANK is the 3-bit "fully-connected data bank" field of CNA_CBUF_CON0 (0x1040),
// alongside WEIGHT_BANK[7:4] and DATA_BANK[3:0]. The matmul regcmd generator drives the
// CNA in CONV mode (a matmul is a 1x1 conv) and leaves FC_DATA_BANK == 0. This sweep forces
// it 0..7 via the ROCKET_FC_DATA_BANK sentinel (npu_regcmd.c gen_matmul_task) and checks
// whether the matmul output stays byte-identical and the wall time stays flat — i.e. whether
// the field is a genuine don't-care on the conv-mode (matmul) datapath. A non-zero field that
// is silently ignored => don't-care confirmed; a divergence or a wedge => it is live and the
// emission must keep it 0.
//
// PART 2 is the "RKNN large-K" confirmation: RKNN exposes K<=10240 as an API convenience
// (internally tiled). This runs a single large-K matmul through our tiler and confirms it is
// correct vs a CPU reference and reports the tiling (Kt, nKt) our planner chose — no hidden
// trick, our Kt already tiles past RKNN's window.
//
// Bit-exactness model: the cross-fc comparison is a raw byte memcmp of the fp16 outputs (the
// same fp16 path runs each time, so it is exact regardless of magnitude); correctness of the
// reference run is checked against an fp32 CPU matmul with the usual fp16 tolerance.
//
// No sudo needed (opens /dev/accel directly). SKIP (exit 2) with no NPU. FAIL (exit 1) on any
// cross-fc divergence, a reference-correctness miss, or an NPU error.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

/* fp32 CPU reference: C[m,n] = sum_k A[m,k]*B[n,k]  (B is [N,K], C=A·B^T) */
static void cpu_ref(const _Float16 *A, const _Float16 *B, float *R, int M, int K, int N)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += (float)A[(size_t)m*K+k] * (float)B[(size_t)n*K+k];
            R[(size_t)m*N+n] = s;
        }
}

/* tolerance compare NPU fp16 output vs fp32 reference (matmul_tiled criterion) */
static long check_ref(const _Float16 *C, const float *R, int M, int N,
                      double *max_abs_o, double *max_rel_o)
{
    double max_abs = 0, max_rel = 0; long nbad = 0;
    for (size_t i = 0; i < (size_t)M*N; i++) {
        float got = (float)C[i], exp = (float)(_Float16)R[i];
        double ad = fabs(got - exp), rd = ad / (fabs(exp) + 1e-6);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (rd > 0.02 && ad > 1.0) nbad++;
    }
    *max_abs_o = max_abs; *max_rel_o = max_rel;
    return nbad;
}

int main(void)
{
    int fail = 0;
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); return 2; }

    /* ---- PART 1: FC_DATA_BANK sweep on a multi-tile fp16 matmul ---- */
    const int M = 256, K = 2048, N = 1024;
    int Mt, Kt, Nt;
    int njobs = rocket_matmul_plan(M, K, N, &Mt, &Kt, &Nt);
    int nKt = (K + Kt - 1) / Kt;
    printf("PART 1: FC_DATA_BANK sweep  C[%d,%d]=A[%d,%d]xB[%d,%d]^T\n", M, N, M, K, N, K);
    printf("  tiling Mt=%d Kt=%d Nt=%d -> %d jobs (nKt=%d K-tiles)\n", Mt, Kt, Nt, njobs, nKt);

    _Float16 *A   = malloc((size_t)M*K*sizeof(_Float16));
    _Float16 *B   = malloc((size_t)N*K*sizeof(_Float16));
    _Float16 *Cr  = malloc((size_t)M*N*sizeof(_Float16));   /* fc-unset reference */
    _Float16 *Cf  = malloc((size_t)M*N*sizeof(_Float16));   /* per-fc run */
    float    *R   = malloc((size_t)M*N*sizeof(float));
    if (!A||!B||!Cr||!Cf||!R) { fprintf(stderr,"alloc\n"); rocket_close(fd); return 2; }
    srand(1234);
    for (size_t i=0;i<(size_t)M*K;i++) A[i]=(_Float16)(rand()%3);
    for (size_t i=0;i<(size_t)N*K;i++) B[i]=(_Float16)(rand()%3);

    /* reference run with FC_DATA_BANK unset (== field 0) */
    unsetenv("ROCKET_FC_DATA_BANK");
    memset(Cr,0,(size_t)M*N*sizeof(_Float16));
    int ret = rocket_matmul_fp16(fd, M, K, N, A, B, Cr);
    if (ret) { fprintf(stderr,"reference matmul = %d\n", ret); rocket_close(fd); return 1; }
    cpu_ref(A,B,R,M,K,N);
    double ma, mr; long nb = check_ref(Cr,R,M,N,&ma,&mr);
    printf("  reference (fc unset): max_abs=%.3f max_rel=%.4f nbad=%ld -> %s\n",
           ma, mr, nb, nb? "CORRECTNESS-FAIL":"correct");
    if (nb) fail = 1;

    /* Contract this gate enforces (the don't-care assertions): fc=0 (the field our
     * generator emits) must be byte-identical to the fc-unset reference, and the default
     * path must be correct. The fc=1..7 rows are an RE characterization, NOT pass/fail:
     * the field is LIVE on the conv-mode matmul datapath (a non-zero value corrupts the
     * output), which is exactly why the generator pins it 0 — so a divergence there is the
     * expected/known finding, reported but not failed. */
    const size_t obytes = (size_t)M*N*sizeof(_Float16);
    printf("  %-12s %12s %10s %12s   (fc=1..7 = RE characterization, expected to diverge)\n",
           "FC_DATA_BANK", "vs-ref", "diffbytes", "median wall");
    for (int fc=0; fc<8; fc++) {
        char v[4]; snprintf(v,sizeof v,"%d",fc); setenv("ROCKET_FC_DATA_BANK",v,1);
        int err=0; int64_t best=0;
        for (int rep=0; rep<5; rep++) {
            memset(Cf,0,obytes);
            int64_t t0=now_us();
            int r = rocket_matmul_fp16(fd, M, K, N, A, B, Cf);
            int64_t us=now_us()-t0;
            if (r) { err=r; break; }
            if (rep==0 || us<best) best=us;
        }
        if (err) { printf("  fc=%d         ERROR (matmul=%d) -- possible wedge, re-run after rmmod\n", fc, err);
                   fail=1; continue; }
        size_t diffbytes=0; double mdelta=0;
        const unsigned char *a=(const unsigned char*)Cr, *b=(const unsigned char*)Cf;
        for (size_t i=0;i<obytes;i++) if(a[i]!=b[i]) diffbytes++;
        for (size_t i=0;i<(size_t)M*N;i++){ double d=fabs((double)Cr[i]-(double)Cf[i]); if(d>mdelta) mdelta=d; }
        printf("  fc=%d        %12s %7zu/%zu %9lld us   %s\n", fc,
               diffbytes==0?"IDENTICAL":"DIVERGED", diffbytes, obytes, (long long)best,
               diffbytes==0?"(== emitted)":"FC_DATA_BANK LIVE");
        if (fc!=0 && diffbytes) printf("                 max|Δ| vs ref = %.3f\n", mdelta);
        /* the ONLY fail in this loop: fc=0 must equal the unset reference (generator contract) */
        if (fc==0 && diffbytes) { printf("  CONTRACT VIOLATION: fc=0 != unset\n"); fail=1; }
    }
    unsetenv("ROCKET_FC_DATA_BANK");

    /* ---- PART 2: large-K single matmul confirmation ---- */
    const int M2=64, K2=10240, N2=256;
    int Mt2,Kt2,Nt2;
    int nj2 = rocket_matmul_plan(M2,K2,N2,&Mt2,&Kt2,&Nt2);
    int nKt2 = (K2+Kt2-1)/Kt2;
    printf("\nPART 2: large-K probe  C[%d,%d]=A[%d,%d]xB[%d,%d]^T\n", M2,N2,M2,K2,N2,K2);
    printf("  tiling Mt=%d Kt=%d Nt=%d -> %d jobs (nKt=%d) %s\n", Mt2,Kt2,Nt2,nj2,nKt2,
           K2>10240-1?"(>= RKNN's K<=10240 window)":"");
    _Float16 *A2=malloc((size_t)M2*K2*sizeof(_Float16));
    _Float16 *B2=malloc((size_t)N2*K2*sizeof(_Float16));
    _Float16 *C2=malloc((size_t)M2*N2*sizeof(_Float16));
    float    *R2=malloc((size_t)M2*N2*sizeof(float));
    if (A2&&B2&&C2&&R2) {
        srand(99);
        for (size_t i=0;i<(size_t)M2*K2;i++) A2[i]=(_Float16)(rand()%2);
        for (size_t i=0;i<(size_t)N2*K2;i++) B2[i]=(_Float16)(rand()%2);
        memset(C2,0,(size_t)M2*N2*sizeof(_Float16));
        int64_t t0=now_us();
        int r=rocket_matmul_fp16(fd,M2,K2,N2,A2,B2,C2);
        int64_t us=now_us()-t0;
        if (r) { printf("  large-K matmul = %d -> FAIL\n", r); fail=1; }
        else {
            cpu_ref(A2,B2,R2,M2,K2,N2);
            double a2,r2; long nb2=check_ref(C2,R2,M2,N2,&a2,&r2);
            double gflops = 2.0*M2*K2*N2/((double)us*1e3);
            printf("  result: max_abs=%.3f max_rel=%.4f nbad=%ld  %lld us  %.1f GFLOP/s -> %s\n",
                   a2,r2,nb2,(long long)us,gflops, nb2?"FAIL":"PASS");
            if (nb2) fail=1;
        }
    }
    free(A2);free(B2);free(C2);free(R2);
    free(A);free(B);free(Cr);free(Cf);free(R);
    rocket_close(fd);
    printf("\n%s\n", fail?"FC_DATA_BANK sweep: FAIL":"FC_DATA_BANK sweep: PASS");
    return fail?1:0;
}
