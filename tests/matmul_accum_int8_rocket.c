// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_accum_int8_rocket.c — isolate the int32 DPU eltwise-add for int8 K-accum.
 *
 * The int8 sibling of matmul_accum_rocket.c. Where that test validated the fp16
 * EW-add (and it now drives the working fp16 ROCKET_KACC), THIS test isolates the
 * unsolved piece: can the DPU EW ALU add two int32 partials (conv int32 result +
 * an int32 operand read back by the ERDMA), bit-exact?
 *
 * Why a fresh test instead of more in-tiled-driver sweeping: the geometry in
 * gen_matmul_int8's accumulate path is BYTE-IDENTICAL to the working fp16 add
 * geometry (base_off = M*ATOMIC_K_SIZE(16), ew_stride = MAX(M,12), surf_notch =
 * ew_stride), and the int32 output really is C2=4 / 16 bytes-per-position (the
 * non-accum path reads it back bit-exact via out_idx_i8). So if the accumulate
 * result is wrong, the suspect is NOT the surface geometry but whether the EW ALU
 * does a 32-bit INTEGER add at all (every known EW example uses EDATA_SIZE=2
 * i.e. 16-bit; there is no 32-bit EW exemplar).
 *
 *   single mode  — ONE accumulate job, full K, add operand = a CONSTANT int32.
 *                  A constant operand is layout-INVARIANT, so this cleanly
 *                  separates "does the int32 EW add work" from "is the surface
 *                  addressing right". The readback self-classifies:
 *                    out==ref+COP -> int32 EW-add WORKS (then geometry is fine too)
 *                    out==ref     -> conv works, ERDMA read as 0 (operand not read)
 *                    out==COP     -> conv dropped, EW passes operand
 *                    out==sentinel-> WDMA didn't write under accumulate
 *                    saturation/garbage -> int32 add itself is broken (EDATA_SIZE)
 *   chain  mode  — the real scenario: split K into nKt tiles, ping-pong an
 *                  accumulate chain, read the final buf once; bit-EXACT vs int64
 *                  CPU ref (int32 EW-add is integer-exact, unlike fp16's running
 *                  round). Use once `single` passes to confirm multi-surface +
 *                  depth. sudo ... chain M K N nKt  (K%nKt==0, (K/nKt)%32==0).
 *
 * The int32 EW precision/geometry knobs are inherited from gen_matmul_int8:
 *   ROCKET_INT8_EW_CFG  ROCKET_INT8_ERDMA_CFG  ROCKET_INT8_RDMA_FMC
 *   ROCKET_INT8_EW_BASE_OFF  ROCKET_INT8_EW_SURF  ROCKET_INT8_EW_NOTCH
 * Run the integer-add candidate as:
 *   ROCKET_INT8_EW_CFG=0x10C203C4 sudo -E ./matmul_accum_int8_rocket single 4 64 32
 *
 * NB: a wrong EW config can WEDGE the NPU (WAIT TIMEOUT -110) that persists to the
 * next run — reload the rocket module between failing attempts.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

/* int8 layout via the exported feature_data(): input cube C2=16, output cube
 * C2=4 (== feat_idx_i8/out_idx_i8 in rocket_matmul.c); weights via weight_int8. */
static inline int in_idx(int K, int M, int c, int h)  { return feature_data(K, M, 1, 16, c, h, 1); }
static inline int out_idx(int N, int M, int n, int h) { return feature_data(N, M, 1, 4,  n, h, 1); }

static int8_t small_i8(size_t i) { return (int8_t)((int)(i % 9) - 4); }  /* [-4,4], keeps ref modest */

/* run one int8 conv tile: out[M,N] (+)= A_part[M,Kt] @ B_part[N,Kt]^T (int32).
 * add_bo MUST differ from out (ping-pong; in-place ERDMA/WDMA corrupts). */
static int run_partial(int fd, int M, int Kt, int N,
                       rocket_bo *in, rocket_bo *wt, rocket_bo *out,
                       rocket_bo *regcmd, int accumulate, rocket_bo *add_bo)
{
    uint64_t ops[256] = {0};
    matmul_params_t p = {
        .m = (uint16_t)M, .k = (uint16_t)Kt, .n = (uint16_t)N,
        .input_dma   = (uint32_t)in->dma_address,
        .weights_dma = (uint32_t)wt->dma_address,
        .output_dma  = (uint32_t)out->dma_address,
        .tasks = ops,
        .accumulate = (uint8_t)accumulate,
        .add_dma = accumulate ? (uint32_t)add_bo->dma_address : 0,
    };
    int r = gen_matmul_int8(&p);
    if (r) { fprintf(stderr, "gen_matmul_int8 failed (%d)\n", r); return -1; }

    if (accumulate) {   /* echo the regs THIS binary generated, to defeat stale builds */
        uint32_t ew=0, erd=0, fmc=0, ews=0;
        for (uint32_t z = 0; z < p.task_count; z++) {
            uint16_t rg = ops[z] & 0xFFFF; uint32_t v = (ops[z] >> 16) & 0xFFFFFFFF;
            if (rg==0x4070) ew=v; else if (rg==0x5034) erd=v;
            else if (rg==0x5040) ews=v; else if (rg==0x5044) fmc=v;
        }
        fprintf(stderr, "[regs] EW_CFG=0x%08x ERDMA=0x%08x EW_SURF_STRIDE=0x%08x FMC=0x%08x\n",
                ew, erd, ews, fmc);
    }

    rocket_bo_prep(fd, regcmd, 1, 0);
    memcpy(regcmd->ptr, ops, (size_t)p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, regcmd);

    rocket_task_desc t = { (uint32_t)regcmd->dma_address, p.task_count };
    uint32_t add_h       = add_bo ? add_bo->handle : 0;
    uint32_t in_h_acc[]  = { in->handle, wt->handle, regcmd->handle, add_h };
    uint32_t in_h_pln[]  = { in->handle, wt->handle, regcmd->handle };
    uint32_t out_h[]     = { out->handle };

    if (accumulate) r = rocket_submit_tasks(fd, &t, 1, in_h_acc, 4, out_h, 1);
    else            r = rocket_submit_tasks(fd, &t, 1, in_h_pln, 3, out_h, 1);
    if (r) { fprintf(stderr, "submit failed (%d)\n", r); return -1; }

    r = rocket_bo_prep(fd, out, 0, 2000000000LL);
    if (r) { fprintf(stderr, "job timed out (%d) -- a wrong EW cfg can wedge the NPU; "
                             "reload the rocket module before the next try\n", r); return -1; }
    return 0;
}

static void pack_in(int fd, rocket_bo *bo, const int8_t *A, int M, int K, int k0, int Kt) {
    rocket_bo_prep(fd, bo, 1, 0); memset(bo->ptr, 0, bo->size);
    int8_t *ip = bo->ptr;
    for (int h = 1; h <= M; h++)
        for (int c = 1; c <= Kt; c++)
            ip[in_idx(Kt, M, c, h)] = A[(size_t)(h-1)*K + (k0 + c-1)];
    rocket_bo_fini(fd, bo);
}
static void pack_wt(int fd, rocket_bo *bo, const int8_t *B, int N, int K, int k0, int Kt) {
    rocket_bo_prep(fd, bo, 1, 0); memset(bo->ptr, 0, bo->size);
    int8_t *wp = bo->ptr;
    for (int kk = 1; kk <= N; kk++)
        for (int c = 1; c <= Kt; c++)
            wp[weight_int8(Kt, kk, c)] = B[(size_t)(kk-1)*K + (k0 + c-1)];
    rocket_bo_fini(fd, bo);
}

/* generous int8 in / int8 wt / int32 out byte sizes (pad to be safe) */
#define IN_SZ(M,Kt)  ((size_t)((M+3)/4*4)   * ((Kt+31)/32*32) * sizeof(int8_t)  + 4096)
#define WT_SZ(N,Kt)  ((size_t)((N+31)/32*32) * ((Kt+31)/32*32) * sizeof(int8_t)  + 4096)
#define OUT_SZ(M,N)  ((size_t)((M+3)/4*4)    * ((N+31)/32*32)  * sizeof(int32_t) + 4096)

int main(int argc, char **argv)
{
    /* ---------------- convf32 mode: int8-conv -> fp32 OUTPUT, NO EW ----------------
     * THE load-bearing A-probe. Run with ROCKET_INT8_FP32_OUT set: one plain int8
     * conv, output emitted as fp32. Classifies whether the DPU casts its int32 MAC
     * accumulator to fp32 on output (the make-or-break for fp32-EW K-accum):
     *   got == (float)ref            -> int8-conv->fp32 CAST WORKS (A is feasible)
     *   int32_bits(got) == ref       -> NO cast, raw int32 written (out_prec ignored)
     *   else                         -> garbage. */
    if (argc >= 2 && strcmp(argv[1], "convf32") == 0) {
        int M = argc>2?atoi(argv[2]):4, K = argc>3?atoi(argv[3]):64, N = argc>4?atoi(argv[4]):32;
        if (K % 32 || N % 32 || (M % 4 && M != 1)) {
            fprintf(stderr, "convf32: need K%%32==0, N%%32==0, M%%4==0||1\n"); return 2;
        }
        if (!getenv("ROCKET_INT8_FP32_OUT"))
            fprintf(stderr, "convf32: WARN ROCKET_INT8_FP32_OUT not set -> conv emits int32, not fp32\n");
        printf("int8->fp32 conv probe: C[%d,%d] = A[%d,%d] x B[%d,%d]^T (plain conv, fp32 out)\n",
               M, N, M, K, N, K);

        int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)N*K);
        int32_t *ref = calloc((size_t)M*N, sizeof(int32_t));
        if (!A || !B || !ref) { fprintf(stderr, "host alloc failed\n"); return 1; }
        for (size_t i = 0; i < (size_t)M*K; i++) A[i] = small_i8(i*7);
        for (size_t i = 0; i < (size_t)N*K; i++) B[i] = small_i8(i*5);
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
            int64_t a = 0;
            for (int k = 0; k < K; k++) a += (int32_t)A[(size_t)m*K+k]*(int32_t)B[(size_t)n*K+k];
            ref[(size_t)m*N+n] = (int32_t)a;
        }

        int fd = rocket_open();
        if (fd < 0) { fprintf(stderr, "rocket_open failed (%d)\n", fd); return 1; }
        rocket_bo inF={0}, wtF={0}, outF={0}, rcF={0};
        if (rocket_bo_alloc(fd, IN_SZ(M,K), &inF) || rocket_bo_alloc(fd, WT_SZ(N,K), &wtF) ||
            rocket_bo_alloc(fd, OUT_SZ(M,N), &outF) || rocket_bo_alloc(fd, 256*sizeof(uint64_t), &rcF)) {
            fprintf(stderr, "BO alloc failed\n"); return 1;
        }
        pack_in(fd, &inF, A, M, K, 0, K);
        pack_wt(fd, &wtF, B, N, K, 0, K);
        rocket_bo_prep(fd, &outF, 1, 0);
        for (size_t z = 0; z < outF.size/sizeof(int32_t); z++) ((int32_t*)outF.ptr)[z] = 123456789;
        rocket_bo_fini(fd, &outF);

        if (run_partial(fd, M, K, N, &inF, &wtF, &outF, &rcF, 0, NULL)) return 1;

        rocket_bo_prep(fd, &outF, 0, 0);
        float   *obf = (float *)outF.ptr;     /* interpret as fp32 */
        int32_t *obi = (int32_t *)outF.ptr;   /* interpret as int32 */
        double e_castf = 0, e_rawi = 0; long maxabs_i = 0;
        for (int m=0;m<M;m++) for (int n=0;n<N;n++){
            size_t o = out_idx(N,M,n+1,m+1);
            e_castf += fabs((double)obf[o] - (double)ref[(size_t)m*N+n]);  /* fp32 == ref? */
            long di = labs((long)obi[o] - (long)ref[(size_t)m*N+n]);       /* int32 == ref? */
            e_rawi += di; if (di > maxabs_i) maxabs_i = di;
        }
        float s0=obf[out_idx(N,M,1,1)], s1=obf[out_idx(N,M,2,1)], s2=obf[out_idx(N,M,3,1)];
        rocket_bo_fini(fd, &outF);
        printf("convf32: sum|fp32-ref|=%.3f  sum|int32bits-ref|=%.0f (maxabs_i=%ld)\n",
               e_castf, e_rawi, maxabs_i);
        printf("convf32: -> %s\n",
            e_castf < 1.0 ? "int8-conv->fp32 CAST WORKS -> fp32-EW K-accum is FEASIBLE" :
            e_rawi  == 0  ? "NO cast: raw int32 written despite out_precision=fp32 -> A dead via this route" :
            "garbage -> int8-conv with fp32 out not behaving; A blocked here");
        printf("convf32: sample fp32 got: %.2f %.2f %.2f | ref: %d %d %d\n",
               s0, s1, s2, ref[0], ref[1], ref[2]);
        rocket_bo_free(fd,&inF); rocket_bo_free(fd,&wtF); rocket_bo_free(fd,&outF); rocket_bo_free(fd,&rcF);
        rocket_close(fd); free(A); free(B); free(ref);
        return 0;
    }

    /* ---------------- chainf32 mode: fp32-EW K-accum (TEST 2) ----------------
     * Run with ROCKET_INT8_FP32_OUT set. int8 conv -> fp32 out, accumulate the
     * fp32 partials via the PROVEN float EW-add (gen defaults to the float ALU-add
     * + float32 RDMA precision under fp32_out). nKt=2 is the minimal single-EW
     * case. Verifies vs the int64 CPU ref cast to float (fp32 holds modest
     * partials exactly; deep sums round ~1/16M, far below int8 quant error). */
    if (argc >= 2 && strcmp(argv[1], "chainf32") == 0) {
        int M = argc>2?atoi(argv[2]):256, K = argc>3?atoi(argv[3]):1536,
            N = argc>4?atoi(argv[4]):256, nKt = argc>5?atoi(argv[5]):2;
        if (nKt < 1 || K % nKt || (K/nKt) % 32 || N % 32 || (M % 4 && M != 1)) {
            fprintf(stderr, "chainf32: need K%%nKt==0, (K/nKt)%%32==0, N%%32==0, M%%4==0\n");
            return 2;
        }
        if (!getenv("ROCKET_INT8_FP32_OUT"))
            fprintf(stderr, "chainf32: WARN ROCKET_INT8_FP32_OUT not set -> wrong (int32) EW path\n");
        int Kt = K / nKt;
        printf("int8 fp32-EW chain: C[%d,%d]=A[%d,%d]xB[%d,%d]^T, %d K-tiles of %d (fp32 accum)\n",
               M, N, M, K, N, K, nKt, Kt);

        int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)N*K);
        int32_t *ref = calloc((size_t)M*N, sizeof(int32_t));
        if (!A || !B || !ref) { fprintf(stderr, "host alloc failed\n"); return 1; }
        for (size_t i = 0; i < (size_t)M*K; i++) A[i] = small_i8(i*7);
        for (size_t i = 0; i < (size_t)N*K; i++) B[i] = small_i8(i*5);
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
            int64_t a = 0;
            for (int k = 0; k < K; k++) a += (int32_t)A[(size_t)m*K+k]*(int32_t)B[(size_t)n*K+k];
            ref[(size_t)m*N+n] = (int32_t)a;
        }

        int fd = rocket_open();
        if (fd < 0) { fprintf(stderr, "rocket_open failed (%d)\n", fd); return 1; }
        rocket_bo *ins = calloc(nKt, sizeof(rocket_bo)), *wts = calloc(nKt, sizeof(rocket_bo));
        rocket_bo bufA={0}, bufB={0}, rc={0};
        int e = 0;
        for (int t = 0; t < nKt; t++) { e |= rocket_bo_alloc(fd, IN_SZ(M,Kt), &ins[t]);
                                        e |= rocket_bo_alloc(fd, WT_SZ(N,Kt), &wts[t]); }
        e |= rocket_bo_alloc(fd, OUT_SZ(M,N), &bufA);
        e |= rocket_bo_alloc(fd, OUT_SZ(M,N), &bufB);
        e |= rocket_bo_alloc(fd, 256*sizeof(uint64_t), &rc);
        if (e) { fprintf(stderr, "BO alloc failed\n"); return 1; }
        for (int t = 0; t < nKt; t++) { pack_in(fd,&ins[t],A,M,K,t*Kt,Kt); pack_wt(fd,&wts[t],B,N,K,t*Kt,Kt); }

        rocket_bo *bufs[2] = { &bufA, &bufB };
        if (run_partial(fd, M, Kt, N, &ins[0], &wts[0], bufs[0], &rc, 0, NULL)) return 1;
        for (int t = 1; t < nKt; t++) {
            rocket_bo *dst = bufs[t & 1], *src = bufs[(t-1) & 1];
            if (run_partial(fd, M, Kt, N, &ins[t], &wts[t], dst, &rc, 1, src)) return 1;
        }
        rocket_bo *fin = bufs[(nKt-1) & 1];
        rocket_bo_prep(fd, fin, 0, 0);
        float *ob = (float *)fin->ptr;
        double maxabs = 0, maxrel = 0; size_t bad = 0; int shown = 0;
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
            double got = ob[out_idx(N, M, n+1, m+1)], r = ref[(size_t)m*N+n];
            double ad = fabs(got - r), rd = ad / (fabs(r) + 1e-6);
            if (ad > maxabs) maxabs = ad;
            if (rd > maxrel) maxrel = rd;
            if (ad > 0.5 && shown < 6) { printf("  mismatch [%d,%d] exp=%d got=%.2f\n",
                                                m, n, ref[(size_t)m*N+n], got); shown++; }
            if (ad > 0.5) bad++;
        }
        rocket_bo_fini(fd, fin);
        printf("chainf32 verify (nKt=%d): max_abs=%.4f max_rel=%.6f mism=%zu -> %s\n",
               nKt, maxabs, maxrel, bad, (maxabs < 0.5 || maxrel < 1e-4) ? "PASS" : "FAIL");
        for (int t = 0; t < nKt; t++) { rocket_bo_free(fd,&ins[t]); rocket_bo_free(fd,&wts[t]); }
        rocket_bo_free(fd,&bufA); rocket_bo_free(fd,&bufB); rocket_bo_free(fd,&rc);
        free(ins); free(wts); rocket_close(fd); free(A); free(B); free(ref);
        return bad ? 1 : 0;
    }

    /* ---------------- singlef32: fp32-EW add with a CONSTANT operand ----------------
     * Run with ROCKET_INT8_FP32_OUT. One accumulate job, full K, operand = const
     * fp32. Constant => layout-INVARIANT, so this isolates "does the float EW-add
     * mechanism work" from "is the fp32 operand addressing right":
     *   out==ref+COP -> float EW-add WORKS (remaining bug is purely operand geometry)
     *   else         -> the float EW path itself is misconfigured. */
    if (argc >= 2 && strcmp(argv[1], "singlef32") == 0) {
        int M = argc>2?atoi(argv[2]):4, K = argc>3?atoi(argv[3]):64, N = argc>4?atoi(argv[4]):32;
        if (K % 32 || N % 32 || (M % 4 && M != 1)) {
            fprintf(stderr, "singlef32: need K%%32==0, N%%32==0, M%%4==0||1\n"); return 2;
        }
        if (!getenv("ROCKET_INT8_FP32_OUT"))
            fprintf(stderr, "singlef32: WARN ROCKET_INT8_FP32_OUT not set\n");
        const float COP = 1000.0f, SENT = 7654321.0f;
        printf("[singlef32] full K=%d, operand=const %.0f, sentinel=%.0f\n", K, COP, SENT);

        int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)N*K);
        int32_t *ref = calloc((size_t)M*N, sizeof(int32_t));
        if (!A || !B || !ref) { fprintf(stderr, "host alloc failed\n"); return 1; }
        for (size_t i = 0; i < (size_t)M*K; i++) A[i] = small_i8(i*7);
        for (size_t i = 0; i < (size_t)N*K; i++) B[i] = small_i8(i*5);
        for (int m=0;m<M;m++) for (int n=0;n<N;n++){ int64_t a=0;
            for (int k=0;k<K;k++) a+=(int32_t)A[(size_t)m*K+k]*(int32_t)B[(size_t)n*K+k];
            ref[(size_t)m*N+n]=(int32_t)a; }

        int fd = rocket_open();
        if (fd < 0) { fprintf(stderr, "rocket_open failed (%d)\n", fd); return 1; }
        rocket_bo inF={0}, wtF={0}, outF={0}, opF={0}, rcF={0};
        if (rocket_bo_alloc(fd, IN_SZ(M,K), &inF) || rocket_bo_alloc(fd, WT_SZ(N,K), &wtF) ||
            rocket_bo_alloc(fd, OUT_SZ(M,N), &outF) || rocket_bo_alloc(fd, OUT_SZ(M,N), &opF) ||
            rocket_bo_alloc(fd, 256*sizeof(uint64_t), &rcF)) { fprintf(stderr,"alloc failed\n"); return 1; }
        pack_in(fd,&inF,A,M,K,0,K); pack_wt(fd,&wtF,B,N,K,0,K);
        rocket_bo_prep(fd,&opF,1,0);
        for (size_t z=0; z<opF.size/sizeof(float); z++) ((float*)opF.ptr)[z]=COP;
        rocket_bo_fini(fd,&opF);
        rocket_bo_prep(fd,&outF,1,0);
        for (size_t z=0; z<outF.size/sizeof(float); z++) ((float*)outF.ptr)[z]=SENT;
        rocket_bo_fini(fd,&outF);

        if (run_partial(fd, M, K, N, &inF, &wtF, &outF, &rcF, 1, &opF)) return 1;

        rocket_bo_prep(fd,&outF,0,0);
        float *ob=(float*)outF.ptr;
        double e_full=0,e_fullcop=0,e_cop=0,e_sent=0;
        for (int m=0;m<M;m++) for (int n=0;n<N;n++){
            double got=ob[out_idx(N,M,n+1,m+1)], r=ref[(size_t)m*N+n];
            e_full+=fabs(got-r); e_fullcop+=fabs(got-(r+COP));
            e_cop+=fabs(got-COP); e_sent+=fabs(got-SENT);
        }
        float g0=ob[out_idx(N,M,1,1)],g1=ob[out_idx(N,M,2,1)],g2=ob[out_idx(N,M,3,1)];
        rocket_bo_fini(fd,&outF);
        printf("[singlef32] sum|err|: full=%.1f full+op=%.1f op=%.1f sentinel=%.1f\n",
               e_full,e_fullcop,e_cop,e_sent);
        printf("[singlef32] -> %s\n",
            e_fullcop<1.0 ? "float EW-ADD WORKS (out==ref+const) -> only operand GEOMETRY remains" :
            e_full<1.0    ? "conv works, operand read as 0 (ERDMA off)" :
            e_cop<1.0     ? "conv dropped, EW passes operand only" :
            e_sent<1.0    ? "WDMA didn't write" : "float EW path misconfigured");
        printf("[singlef32] sample got: %.1f %.1f %.1f | ref+op: %d %d %d\n",
               g0,g1,g2, ref[0]+(int)COP, ref[1]+(int)COP, ref[2]+(int)COP);
        rocket_bo_free(fd,&inF);rocket_bo_free(fd,&wtF);rocket_bo_free(fd,&outF);
        rocket_bo_free(fd,&opF);rocket_bo_free(fd,&rcF);
        rocket_close(fd); free(A);free(B);free(ref);
        return 0;
    }

    int single = (argc >= 2 && strcmp(argv[1], "single") == 0);

    /* ---------------- chain mode ---------------- */
    if (argc >= 2 && strcmp(argv[1], "chain") == 0) {
        int M = argc>2?atoi(argv[2]):512, K = argc>3?atoi(argv[3]):768,
            N = argc>4?atoi(argv[4]):256, nKt = argc>5?atoi(argv[5]):2;
        if (nKt < 1 || K % nKt || (K/nKt) % 32 || N % 32 || (M % 4 && M != 1)) {
            fprintf(stderr, "chain: need K%%nKt==0, (K/nKt)%%32==0, N%%32==0, M%%4==0\n");
            return 2;
        }
        int Kt = K / nKt;
        printf("int8 chain: C[%d,%d] = A[%d,%d] x B[%d,%d]^T, %d K-tiles of %d (int32 accum)\n",
               M, N, M, K, N, K, nKt, Kt);

        int8_t  *A = malloc((size_t)M * K);
        int8_t  *B = malloc((size_t)N * K);
        int32_t *ref = calloc((size_t)M * N, sizeof(int32_t));
        if (!A || !B || !ref) { fprintf(stderr, "host alloc failed\n"); return 1; }
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = small_i8(i * 7);
        for (size_t i = 0; i < (size_t)N * K; i++) B[i] = small_i8(i * 5);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int64_t a = 0;
                for (int k = 0; k < K; k++) a += (int32_t)A[(size_t)m*K+k] * (int32_t)B[(size_t)n*K+k];
                ref[(size_t)m*N+n] = (int32_t)a;
            }

        int fd = rocket_open();
        if (fd < 0) { fprintf(stderr, "rocket_open failed (%d)\n", fd); return 1; }

        rocket_bo *ins = calloc(nKt, sizeof(rocket_bo));
        rocket_bo *wts = calloc(nKt, sizeof(rocket_bo));
        rocket_bo bufA = {0}, bufB = {0}, rc = {0};
        int e = 0;
        for (int t = 0; t < nKt; t++) { e |= rocket_bo_alloc(fd, IN_SZ(M,Kt), &ins[t]);
                                        e |= rocket_bo_alloc(fd, WT_SZ(N,Kt), &wts[t]); }
        e |= rocket_bo_alloc(fd, OUT_SZ(M,N), &bufA);
        e |= rocket_bo_alloc(fd, OUT_SZ(M,N), &bufB);
        e |= rocket_bo_alloc(fd, 256 * sizeof(uint64_t), &rc);
        if (e) { fprintf(stderr, "BO alloc failed\n"); return 1; }

        for (int t = 0; t < nKt; t++) {
            pack_in(fd, &ins[t], A, M, K, t*Kt, Kt);
            pack_wt(fd, &wts[t], B, N, K, t*Kt, Kt);
        }

        rocket_bo *bufs[2] = { &bufA, &bufB };
        if (run_partial(fd, M, Kt, N, &ins[0], &wts[0], bufs[0], &rc, 0, NULL)) return 1;
        for (int t = 1; t < nKt; t++) {
            rocket_bo *dst = bufs[t & 1], *src = bufs[(t-1) & 1];
            if (run_partial(fd, M, Kt, N, &ins[t], &wts[t], dst, &rc, 1, src)) return 1;
        }
        rocket_bo *fin = bufs[(nKt-1) & 1];

        rocket_bo_prep(fd, fin, 0, 0);
        int32_t *ob = fin->ptr;
        long maxabs = 0; size_t bad = 0; int shown = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int32_t got = ob[out_idx(N, M, n+1, m+1)];
                long d = labs((long)got - (long)ref[(size_t)m*N+n]);
                if (d > maxabs) maxabs = d;
                if (got != ref[(size_t)m*N+n]) { bad++;
                    if (shown < 6) { printf("  mismatch [%d,%d] exp=%d got=%d\n",
                                            m, n, ref[(size_t)m*N+n], got); shown++; } }
            }
        rocket_bo_fini(fd, fin);
        printf("int8 chain verify (nKt=%d): max_abs_diff=%ld mismatches=%zu -> %s\n",
               nKt, maxabs, bad, bad ? "FAIL" : "PASS (bit-exact)");

        for (int t = 0; t < nKt; t++) { rocket_bo_free(fd, &ins[t]); rocket_bo_free(fd, &wts[t]); }
        rocket_bo_free(fd, &bufA); rocket_bo_free(fd, &bufB); rocket_bo_free(fd, &rc);
        free(ins); free(wts); rocket_close(fd); free(A); free(B); free(ref);
        return bad ? 1 : 0;
    }

    /* ---------------- single / default 2-job modes ---------------- */
    int aoff = single ? 1 : 0;
    int M = 4, K = 64, N = 32;
    if (argc >= 4 + aoff) { M = atoi(argv[1+aoff]); K = atoi(argv[2+aoff]); N = atoi(argv[3+aoff]); }

    int Kh = K / 2;
    if (K % 64 || N % 32 || (M % 4 && M != 1)) {
        fprintf(stderr, "need K%%64==0 (so K/2%%32==0), N%%32==0, M%%4==0||1\n");
        return 2;
    }
    printf("int8 accum test: C[%d,%d] = A[%d,%d] x B[%d,%d]^T, K split %d+%d\n",
           M, N, M, K, N, K, Kh, K - Kh);

    int8_t  *A = malloc((size_t)M * K);
    int8_t  *B = malloc((size_t)N * K);
    int32_t *ref = calloc((size_t)M * N, sizeof(int32_t));
    if (!A || !B || !ref) { fprintf(stderr, "host alloc failed\n"); return 1; }
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = small_i8(i * 7);
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = small_i8(i * 5);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64_t a = 0;
            for (int k = 0; k < K; k++) a += (int32_t)A[(size_t)m*K+k] * (int32_t)B[(size_t)n*K+k];
            ref[(size_t)m*N+n] = (int32_t)a;
        }

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open failed (%d)\n", fd); return 1; }

    if (single) {
        /* SEPARATE constant int32 operand (layout-invariant) + sentinel out, one
         * accumulate job over full K. This is THE int32-EW-add classifier. */
        const int32_t COP = 1000000, SENT = 123456789;
        printf("[single] full K=%d, operand=const %d, out sentinel=%d\n", K, COP, SENT);
        rocket_bo inF={0}, wtF={0}, outF={0}, opF={0}, rcF={0};
        if (rocket_bo_alloc(fd, IN_SZ(M,K), &inF) || rocket_bo_alloc(fd, WT_SZ(N,K), &wtF) ||
            rocket_bo_alloc(fd, OUT_SZ(M,N), &outF) || rocket_bo_alloc(fd, OUT_SZ(M,N), &opF) ||
            rocket_bo_alloc(fd, 256*sizeof(uint64_t), &rcF)) {
            fprintf(stderr, "BO alloc failed\n"); return 1;
        }
        pack_in(fd, &inF, A, M, K, 0, K);
        pack_wt(fd, &wtF, B, N, K, 0, K);
        rocket_bo_prep(fd, &opF, 1, 0);
        for (size_t z = 0; z < opF.size/sizeof(int32_t); z++) ((int32_t*)opF.ptr)[z] = COP;
        rocket_bo_fini(fd, &opF);
        rocket_bo_prep(fd, &outF, 1, 0);
        for (size_t z = 0; z < outF.size/sizeof(int32_t); z++) ((int32_t*)outF.ptr)[z] = SENT;
        rocket_bo_fini(fd, &outF);

        if (run_partial(fd, M, K, N, &inF, &wtF, &outF, &rcF, 1, &opF)) return 1;

        rocket_bo_prep(fd, &outF, 0, 0);
        int32_t *ob = outF.ptr;
        long e_full=0,e_fullcop=0,e_cop=0,e_sent=0,e_zero=0;
        for (int m=0;m<M;m++) for (int n=0;n<N;n++){
            long got=ob[out_idx(N,M,n+1,m+1)]; long r=ref[(size_t)m*N+n];
            e_full+=labs(got-r); e_fullcop+=labs(got-(r+COP));
            e_cop+=labs(got-COP); e_sent+=labs(got-SENT); e_zero+=labs(got);
        }
        int32_t g0=ob[out_idx(N,M,1,1)], g1=ob[out_idx(N,M,2,1)], g2=ob[out_idx(N,M,3,1)];
        rocket_bo_fini(fd, &outF);
        printf("[single] sum|err|: full=%ld full+op=%ld op(=%d)=%ld sentinel(=%d)=%ld zero=%ld\n",
               e_full, e_fullcop, COP, e_cop, SENT, e_sent, e_zero);
        const char *verdict =
            e_fullcop==0 ? "int32 EW-ADD WORKS (out==ref+operand) -> geometry is fine; wire it" :
            e_sent  ==0 ? "JOB DID NOT WRITE (out==sentinel) -> WDMA off under accumulate" :
            e_full  ==0 ? "conv works, operand read as 0 -> ERDMA not reading the int32 operand" :
            e_cop   ==0 ? "conv DROPPED, EW passes operand only" :
            e_zero  ==0 ? "all zero" :
            "garbage/saturation -> int32 EW-add itself broken (EDATA_SIZE / 16-bit lane?)";
        printf("[single] -> %s\n", verdict);
        printf("[single] sample got: %d %d %d | ref: %d %d %d  (ref+op: %d %d %d)\n",
               g0, g1, g2, ref[0], ref[1], ref[2], ref[0]+COP, ref[1]+COP, ref[2]+COP);
        rocket_bo_free(fd,&inF); rocket_bo_free(fd,&wtF); rocket_bo_free(fd,&outF);
        rocket_bo_free(fd,&opF); rocket_bo_free(fd,&rcF);
        rocket_close(fd); free(A); free(B); free(ref);
        return 0;
    }

    /* default: 2-job ping-pong (job0 plain -> out; job1 out2 = partial1 + out) */
    rocket_bo in0={0}, in1={0}, wt0={0}, wt1={0}, out={0}, out2={0}, rc={0};
    int e = 0;
    e |= rocket_bo_alloc(fd, IN_SZ(M,Kh), &in0);  e |= rocket_bo_alloc(fd, IN_SZ(M,Kh), &in1);
    e |= rocket_bo_alloc(fd, WT_SZ(N,Kh), &wt0);  e |= rocket_bo_alloc(fd, WT_SZ(N,Kh), &wt1);
    e |= rocket_bo_alloc(fd, OUT_SZ(M,N), &out);  e |= rocket_bo_alloc(fd, OUT_SZ(M,N), &out2);
    e |= rocket_bo_alloc(fd, 256 * sizeof(uint64_t), &rc);
    if (e) { fprintf(stderr, "BO alloc failed\n"); return 1; }

    pack_in(fd, &in0, A, M, K, 0,  Kh); pack_wt(fd, &wt0, B, N, K, 0,  Kh);
    pack_in(fd, &in1, A, M, K, Kh, Kh); pack_wt(fd, &wt1, B, N, K, Kh, Kh);
    rocket_bo_prep(fd, &out, 1, 0); memset(out.ptr, 0, out.size); rocket_bo_fini(fd, &out);

    if (run_partial(fd, M, Kh, N, &in0, &wt0, &out,  &rc, 0, NULL)) return 1;
    if (run_partial(fd, M, Kh, N, &in1, &wt1, &out2, &rc, 1, &out)) return 1;

    rocket_bo_prep(fd, &out2, 0, 0);
    int32_t *ob = out2.ptr;
    int32_t *p0 = calloc((size_t)M*N, sizeof(int32_t));
    int32_t *p1 = calloc((size_t)M*N, sizeof(int32_t));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64_t a0=0, a1=0;
            for (int k = 0;  k < Kh; k++) a0 += (int32_t)A[(size_t)m*K+k]*(int32_t)B[(size_t)n*K+k];
            for (int k = Kh; k < K;  k++) a1 += (int32_t)A[(size_t)m*K+k]*(int32_t)B[(size_t)n*K+k];
            p0[(size_t)m*N+n]=(int32_t)a0; p1[(size_t)m*N+n]=(int32_t)a1;
        }
    long maxabs=0, e_full=0, e_p0=0, e_p1=0; size_t bad=0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t got = ob[out_idx(N, M, n+1, m+1)];
            e_full += labs((long)got - ref[(size_t)m*N+n]);
            e_p0   += labs((long)got - p0[(size_t)m*N+n]);
            e_p1   += labs((long)got - p1[(size_t)m*N+n]);
            long d = labs((long)got - ref[(size_t)m*N+n]);
            if (d > maxabs) maxabs = d;
            if (got != ref[(size_t)m*N+n]) bad++;
        }
    rocket_bo_fini(fd, &out2);
    printf("verify: max_abs_diff=%ld mismatches=%zu -> %s\n", maxabs, bad, bad ? "FAIL" : "PASS (bit-exact)");
    if (bad)
        printf("  classify: vs full(p0+p1)=%ld  vs p0-only=%ld  vs p1-only=%ld  -> %s\n",
               e_full, e_p0, e_p1,
               e_p1 < e_full ? "out==p1: add NOT happening (ERDMA off/ignored)" :
               e_p0 < e_full ? "out==p0: job1 conv 0 / wrong operand" :
               "garbage (sweep ROCKET_INT8_EW_CFG/ERDMA/RDMA_FMC)");
    free(p0); free(p1);
    rocket_bo_free(fd,&in0); rocket_bo_free(fd,&in1); rocket_bo_free(fd,&wt0); rocket_bo_free(fd,&wt1);
    rocket_bo_free(fd,&out); rocket_bo_free(fd,&out2); rocket_bo_free(fd,&rc);
    rocket_close(fd); free(A); free(B); free(ref);
    return bad ? 1 : 0;
}
