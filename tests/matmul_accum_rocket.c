// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_accum_rocket.c — minimal validation of NPU-side K-accumulation.
 *
 * Proves the DPU eltwise-add path works on the rocket driver, on the smallest
 * possible case, before it is wired into the tiled driver:
 *
 *   out  = A[:,0:Kh] @ B[:,0:Kh]^T          (job 0, plain conv, accumulate=0)
 *   out += A[:,Kh:K] @ B[:,Kh:K]^T          (job 1, eltwise-add, accumulate=1,
 *                                            add_dma = out)
 *   verify out == A @ B^T   (full K, CPU reference)
 *
 * TWO SEPARATE JOBS with a fence between them, so a PASS isolates "does the
 * eltwise-add path work" from "does intra-job WDMA->ERDMA ordering hold". Once
 * this passes, the batched single-job variant (all K-partials as consecutive
 * tasks) is the next test, then the tiled-loop rewrite.
 *
 * Everything is fp16 (accumulator included) to keep this minimal — only the EW
 * add is under test. Switch the accumulator to fp32 (fp32tofp16=0) for the real
 * 15-deep case after the path is validated.
 *
 * If it FAILS with the NPU producing job-0's result only (no add) or garbage,
 * sweep the undocumented fp16 EW precision fields without recompiling:
 *   ROCKET_EW_CFG=<hex>   ROCKET_ERDMA_CFG=<hex>   (see npu_regcmd.c)
 *
 * Build:
 *   gcc -O2 -Iinclude tests/matmul_accum_rocket.c src/rocket_npu.c \
 *       src/npu_regcmd.c -o matmul_accum_rocket -lm
 * Run:
 *   sudo ./matmul_accum_rocket            # M=4 K=64 N=16, split at K/2
 *   sudo ./matmul_accum_rocket 8 128 32   # M K N (N%16==0, K split in half, K/2%32==0)
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

/* run one conv tile: out[M,N] (+)= A_part[M,Kt] @ B_part[N,Kt]^T  */
/* out = conv(in,wt) [+ add_bo if accumulate]. add_bo MUST differ from out:
 * in-place (add_bo==out) corrupts — the NPU can't read the EW operand from the
 * buffer WDMA is writing. So K-accumulation ping-pongs between two buffers. */
static int run_partial(int fd, int M, int Kt, int N,
                       rocket_bo *in, rocket_bo *wt, rocket_bo *out,
                       rocket_bo *regcmd, int accumulate, rocket_bo *add_bo)
{
    uint64_t ops[256] = {0};
    matmul_params_t p = {
        .m = M, .k = Kt, .n = N,
        .input_dma   = (uint32_t)in->dma_address,
        .weights_dma = (uint32_t)wt->dma_address,
        .output_dma  = (uint32_t)out->dma_address,
        .tasks = ops, .fp32tofp16 = 1,
        .accumulate = accumulate,
        .add_dma = accumulate ? (uint32_t)add_bo->dma_address : 0,
    };
    int r = gen_matmul_fp16(&p);
    if (r) { fprintf(stderr, "gen_matmul_fp16 failed (%d)\n", r); return -1; }

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

    /* accumulate => NPU READS add_bo (ERDMA) and WRITES out (WDMA): both on lists.
     * Guard the deref — job0 passes add_bo=NULL. */
    uint32_t add_h       = add_bo ? add_bo->handle : 0;
    uint32_t in_h_acc[]  = { in->handle, wt->handle, regcmd->handle, add_h };
    uint32_t in_h_pln[]  = { in->handle, wt->handle, regcmd->handle };
    uint32_t out_h[]     = { out->handle };

    if (accumulate) r = rocket_submit_tasks(fd, &t, 1, in_h_acc, 4, out_h, 1);
    else            r = rocket_submit_tasks(fd, &t, 1, in_h_pln, 3, out_h, 1);
    if (r) { fprintf(stderr, "submit failed (%d)\n", r); return -1; }

    r = rocket_bo_prep(fd, out, 0, 2000000000LL);   /* wait + invalidate for CPU */
    if (r) { fprintf(stderr, "job timed out (%d)\n", r); return -1; }
    return 0;
}

/* pack A[M,Kt] (rows, cols k0..k0+Kt) into NPU input cube layout */
static void pack_in(int fd, rocket_bo *bo, const _Float16 *A, int M, int K, int k0, int Kt) {
    rocket_bo_prep(fd, bo, 1, 0); memset(bo->ptr, 0, bo->size);
    _Float16 *ip = bo->ptr;
    for (int h = 1; h <= M; h++)
        for (int c = 1; c <= Kt; c++)
            ip[feature_data(Kt, M, 1, 8, c, h, 1)] = A[(size_t)(h-1)*K + (k0 + c-1)];
    rocket_bo_fini(fd, bo);
}
static void pack_wt(int fd, rocket_bo *bo, const _Float16 *B, int N, int K, int k0, int Kt) {
    rocket_bo_prep(fd, bo, 1, 0); memset(bo->ptr, 0, bo->size);
    _Float16 *wp = bo->ptr;
    for (int kk = 1; kk <= N; kk++)
        for (int c = 1; c <= Kt; c++)
            wp[weight_fp16(Kt, kk, c)] = B[(size_t)(kk-1)*K + (k0 + c-1)];
    rocket_bo_fini(fd, bo);
}

int main(int argc, char **argv)
{
    /* "single" mode: ONE accumulate job over full K with the add operand
     * pre-zeroed (out = conv + 0). Isolates "does the conv feed the EW ALU at
     * all under accumulate?" from the cross-job operand read. out==full => conv
     * reaches EW; out==0 => conv dropped when EW active. */
    int single = (argc >= 2 && strcmp(argv[1], "single") == 0);
    /* "chain" mode: the REAL K-accum scenario — split K into nKt equal tiles and
     * ping-pong an accumulate chain (job0 plain -> bufA; job j accumulates the
     * prev buf into the other buf), reading the final buf ONCE. This is what
     * mm_compute will do per output tile. Verifies bit-CLOSE vs the CPU fp32
     * full-K reference at depth nKt, exposing fp16-intermediate drift (if any).
     *   sudo ./matmul_accum_rocket chain M K N nKt   (each K/nKt must be %32) */
    if (argc >= 2 && strcmp(argv[1], "chain") == 0) {
        int M = argc>2?atoi(argv[2]):512, K = argc>3?atoi(argv[3]):640,
            N = argc>4?atoi(argv[4]):384, nKt = argc>5?atoi(argv[5]):10;
        if (nKt < 1 || K % nKt || (K/nKt) % 32 || N % 16 || (M % 4 && M != 1)) {
            fprintf(stderr, "chain: need K%%nKt==0, (K/nKt)%%32==0, N%%16==0, M%%4==0\n");
            return 2;
        }
        int Kt = K / nKt;
        printf("chain test: C[%d,%d] = A[%d,%d] x B[%d,%d]^T, %d K-tiles of %d (fp16 accum)\n",
               M, N, M, K, N, K, nKt, Kt);

        _Float16 *A = malloc((size_t)M * K * sizeof(_Float16));
        _Float16 *B = malloc((size_t)N * K * sizeof(_Float16));
        float *ref  = calloc((size_t)M * N, sizeof(float));
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (_Float16)(((i * 7) % 13 - 6) * 0.1f);
        for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (_Float16)(((i * 5) % 11 - 5) * 0.1f);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float a = 0;
                for (int k = 0; k < K; k++) a += (float)A[(size_t)m*K+k] * (float)B[(size_t)n*K+k];
                ref[m*N+n] = a;
            }

        int fd = rocket_open();
        if (fd < 0) { fprintf(stderr, "rocket_open failed (%d)\n", fd); return 1; }

        size_t in_sz  = (size_t)((M+3)/4*4)  * ((Kt+31)/32*32) * sizeof(_Float16) + 4096;
        size_t wt_sz  = (size_t)((N+15)/16*16)* ((Kt+31)/32*32) * sizeof(_Float16) + 4096;
        size_t out_sz = (size_t)((M+3)/4*4)  * ((N+15)/16*16)  * sizeof(_Float16) + 4096;
        rocket_bo *ins = calloc(nKt, sizeof(rocket_bo));
        rocket_bo *wts = calloc(nKt, sizeof(rocket_bo));
        rocket_bo bufA = {0}, bufB = {0}, rc = {0};
        int e = 0;
        for (int t = 0; t < nKt; t++) { e |= rocket_bo_alloc(fd, in_sz, &ins[t]);
                                        e |= rocket_bo_alloc(fd, wt_sz, &wts[t]); }
        e |= rocket_bo_alloc(fd, out_sz, &bufA);
        e |= rocket_bo_alloc(fd, out_sz, &bufB);
        e |= rocket_bo_alloc(fd, 256 * sizeof(uint64_t), &rc);
        if (e) { fprintf(stderr, "BO alloc failed\n"); return 1; }

        for (int t = 0; t < nKt; t++) {
            pack_in(fd, &ins[t], A, M, K, t*Kt, Kt);
            pack_wt(fd, &wts[t], B, N, K, t*Kt, Kt);
        }

        /* chain: job0 plain -> bufA; job t reads prev buf, writes the OTHER buf.
         * after job t the live result is in bufA if t even, bufB if t odd. */
        rocket_bo *bufs[2] = { &bufA, &bufB };
        if (run_partial(fd, M, Kt, N, &ins[0], &wts[0], bufs[0], &rc, 0, NULL)) return 1;
        for (int t = 1; t < nKt; t++) {
            rocket_bo *dst = bufs[t & 1], *src = bufs[(t-1) & 1];
            if (run_partial(fd, M, Kt, N, &ins[t], &wts[t], dst, &rc, 1, src)) return 1;
        }
        rocket_bo *fin = bufs[(nKt-1) & 1];

        rocket_bo_prep(fd, fin, 0, 0);
        _Float16 *ob = fin->ptr;
        float max_abs = 0, max_rel = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float got = (float)ob[feature_data(N, M, 1, 8, n+1, m+1, 1)];
                float ad = fabsf(got - ref[m*N+n]);
                float rd = ad / (fabsf(ref[m*N+n]) + 1e-6f);
                if (ad > max_abs) max_abs = ad;
                if (rd > max_rel) max_rel = rd;
            }
        rocket_bo_fini(fd, fin);
        int pass = (max_abs < 0.5f || max_rel < 0.05f);
        printf("chain verify (nKt=%d): max_abs=%.4f max_rel=%.4f -> %s\n",
               nKt, max_abs, max_rel, pass ? "PASS" : "FAIL");

        for (int t = 0; t < nKt; t++) { rocket_bo_free(fd, &ins[t]); rocket_bo_free(fd, &wts[t]); }
        rocket_bo_free(fd, &bufA); rocket_bo_free(fd, &bufB); rocket_bo_free(fd, &rc);
        free(ins); free(wts);
        rocket_close(fd); free(A); free(B); free(ref);
        return pass ? 0 : 1;
    }
    int aoff = single ? 1 : 0;

    int M = 4, K = 64, N = 16;
    if (argc >= 4 + aoff) { M = atoi(argv[1+aoff]); K = atoi(argv[2+aoff]); N = atoi(argv[3+aoff]); }

    int Kh = K / 2;
    if (K % 64 || N % 16 || (M % 4 && M != 1)) {
        fprintf(stderr, "need K%%64==0 (so K/2%%32==0), N%%16==0, M%%4==0\n");
        return 2;
    }
    printf("accum test: C[%d,%d] = A[%d,%d] x B[%d,%d]^T, K split %d+%d\n",
           M, N, M, K, N, K, Kh, K - Kh);

    /* host operands */
    _Float16 *A = malloc((size_t)M * K * sizeof(_Float16));
    _Float16 *B = malloc((size_t)N * K * sizeof(_Float16));
    float *ref  = calloc((size_t)M * N, sizeof(float));
    for (int i = 0; i < M * K; i++) A[i] = (_Float16)(((i * 7) % 13 - 6) * 0.1f);
    for (int i = 0; i < N * K; i++) B[i] = (_Float16)(((i * 5) % 11 - 5) * 0.1f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float a = 0;
            for (int k = 0; k < K; k++) a += (float)A[m*K+k] * (float)B[n*K+k];
            ref[m*N+n] = a;
        }

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open failed (%d)\n", fd); return 1; }

    if (single) {
        /* Disambiguating experiment: SEPARATE constant operand (all = COP) + out
         * pre-filled with a SENTINEL. One accumulate job, full K, add_dma=operand.
         * The readback classifies every hypothesis at once:
         *   out==sentinel -> job did NOT write (WDMA off in accumulate mode)
         *   out==full+COP -> conv AND operand both work (2-job bug = coherency)
         *   out==full     -> conv works, operand read as 0 (ERDMA not reading)
         *   out==COP      -> conv dropped, EW passes operand
         *   out==0        -> everything zero */
        const float COP = 2.0f, SENT = 9.0f;
        printf("[single] full K=%d, operand=const %.1f, out sentinel=%.1f\n", K, COP, SENT);
        rocket_bo inF={0}, wtF={0}, outF={0}, opF={0}, rcF={0};
        size_t in_sz  = (size_t)((M+3)/4*4)  * ((K+31)/32*32)  * sizeof(_Float16) + 4096;
        size_t wt_sz  = (size_t)((N+15)/16*16)* ((K+31)/32*32)  * sizeof(_Float16) + 4096;
        size_t out_sz = (size_t)((M+3)/4*4)  * ((N+15)/16*16)  * sizeof(_Float16) + 4096;
        if (rocket_bo_alloc(fd, in_sz, &inF) || rocket_bo_alloc(fd, wt_sz, &wtF) ||
            rocket_bo_alloc(fd, out_sz, &outF) || rocket_bo_alloc(fd, out_sz, &opF) ||
            rocket_bo_alloc(fd, 256*sizeof(uint64_t), &rcF)) {
            fprintf(stderr, "BO alloc failed\n"); return 1;
        }
        pack_in(fd, &inF, A, M, K, 0, K);
        pack_wt(fd, &wtF, B, N, K, 0, K);
        /* fill operand BO entirely with COP and out entirely with SENT (constant
         * => NPU layout irrelevant). */
        rocket_bo_prep(fd, &opF, 1, 0);
        for (size_t z = 0; z < opF.size/sizeof(_Float16); z++) ((_Float16*)opF.ptr)[z] = (_Float16)COP;
        rocket_bo_fini(fd, &opF);
        rocket_bo_prep(fd, &outF, 1, 0);
        for (size_t z = 0; z < outF.size/sizeof(_Float16); z++) ((_Float16*)outF.ptr)[z] = (_Float16)SENT;
        rocket_bo_fini(fd, &outF);

        /* accumulate job: add_dma = the SEPARATE operand buffer, output -> outF */
        uint64_t ops[256] = {0};
        matmul_params_t p = { .m=M,.k=K,.n=N,
            .input_dma=(uint32_t)inF.dma_address, .weights_dma=(uint32_t)wtF.dma_address,
            .output_dma=(uint32_t)outF.dma_address, .tasks=ops, .fp32tofp16=1,
            .accumulate=1, .add_dma=(uint32_t)opF.dma_address };
        if (gen_matmul_fp16(&p)) { fprintf(stderr,"gen failed\n"); return 1; }
        rocket_bo_prep(fd, &rcF, 1, 0); memcpy(rcF.ptr, ops, p.task_count*sizeof(uint64_t)); rocket_bo_fini(fd, &rcF);
        rocket_task_desc t = { (uint32_t)rcF.dma_address, p.task_count };
        uint32_t inh[] = { inF.handle, wtF.handle, rcF.handle, opF.handle };
        uint32_t outh[] = { outF.handle };
        if (rocket_submit_tasks(fd, &t, 1, inh, 4, outh, 1)) { fprintf(stderr,"submit failed\n"); return 1; }
        if (rocket_bo_prep(fd, &outF, 0, 2000000000LL)) { fprintf(stderr,"timeout\n"); return 1; }

        _Float16 *ob = outF.ptr;
        float e_full=0,e_fullcop=0,e_cop=0,e_sent=0,e_zero=0;
        for (int m=0;m<M;m++) for (int n=0;n<N;n++){
            float got=(float)ob[feature_data(N,M,1,8,n+1,m+1,1)];
            e_full+=fabsf(got-ref[m*N+n]); e_fullcop+=fabsf(got-(ref[m*N+n]+COP));
            e_cop+=fabsf(got-COP); e_sent+=fabsf(got-SENT); e_zero+=fabsf(got);
        }
        rocket_bo_fini(fd, &outF);
        printf("[single] |err|: full=%.2f full+op=%.2f op(=%.0f)=%.2f sentinel(=%.0f)=%.2f zero=%.2f\n",
               e_full, e_fullcop, COP, e_cop, SENT, e_sent, e_zero);
        const char *verdict =
            e_sent   < 1.0f ? "JOB DID NOT WRITE (out==sentinel) -> WDMA not firing in accum mode" :
            e_fullcop< 1.0f ? "conv+operand BOTH work -> 2-job failure is cross-job coherency" :
            e_full   < 1.0f ? "conv works, operand read as 0 -> ERDMA not reading" :
            e_cop    < 1.0f ? "conv DROPPED, EW passes operand" :
            e_zero   < 1.0f ? "all zero" : "garbage";
        printf("[single] -> %s\n", verdict);
        printf("[single] sample got %.3f %.3f %.3f | full %.3f %.3f %.3f\n",
               (float)ob[feature_data(N,M,1,8,1,1,1)], (float)ob[feature_data(N,M,1,8,2,1,1)],
               (float)ob[feature_data(N,M,1,8,3,1,1)], ref[0], ref[1], ref[2]);
        rocket_bo_free(fd,&inF); rocket_bo_free(fd,&wtF); rocket_bo_free(fd,&outF);
        rocket_bo_free(fd,&opF); rocket_bo_free(fd,&rcF);
        rocket_close(fd); free(A); free(B); free(ref);
        return 0;
    }

    rocket_bo in0 = {0}, in1 = {0}, wt0 = {0}, wt1 = {0}, out = {0}, rc = {0};
    /* generous slots: rup(M,4)*rup(Kt,32) elems in / rup(N,16)*rup(Kt,32) wt /
     * rup(M,4)*rup(N,16) out — pad to be safe. */
    size_t in_sz  = (size_t)((M+3)/4*4) * ((Kh+31)/32*32) * sizeof(_Float16) + 4096;
    size_t wt_sz  = (size_t)((N+15)/16*16) * ((Kh+31)/32*32) * sizeof(_Float16) + 4096;
    size_t out_sz = (size_t)((M+3)/4*4) * ((N+15)/16*16) * sizeof(_Float16) + 4096;
    int e = 0;
    e |= rocket_bo_alloc(fd, in_sz,  &in0);
    e |= rocket_bo_alloc(fd, in_sz,  &in1);
    e |= rocket_bo_alloc(fd, wt_sz,  &wt0);
    e |= rocket_bo_alloc(fd, wt_sz,  &wt1);
    rocket_bo out2 = {0};
    e |= rocket_bo_alloc(fd, out_sz, &out);
    e |= rocket_bo_alloc(fd, out_sz, &out2);   /* ping-pong target (NOT in-place) */
    e |= rocket_bo_alloc(fd, 256 * sizeof(uint64_t), &rc);
    if (e) { fprintf(stderr, "BO alloc failed\n"); return 1; }

    /* pack each K-half into NPU tile layout (helpers exported by the generator) */
    rocket_bo *ins[2] = { &in0, &in1 }, *wts[2] = { &wt0, &wt1 };
    for (int part = 0; part < 2; part++) {
        int k0 = part * Kh;
        rocket_bo_prep(fd, ins[part], 1, 0);
        memset(ins[part]->ptr, 0, ins[part]->size);
        _Float16 *ip = ins[part]->ptr;
        for (int h = 1; h <= M; h++)
            for (int c = 1; c <= Kh; c++)
                ip[feature_data(Kh, M, 1, 8, c, h, 1)] = A[(size_t)(h-1)*K + (k0 + c-1)];
        rocket_bo_fini(fd, ins[part]);

        rocket_bo_prep(fd, wts[part], 1, 0);
        memset(wts[part]->ptr, 0, wts[part]->size);
        _Float16 *wp = wts[part]->ptr;
        for (int kk = 1; kk <= N; kk++)
            for (int c = 1; c <= Kh; c++)
                wp[weight_fp16(Kh, kk, c)] = B[(size_t)(kk-1)*K + (k0 + c-1)];
        rocket_bo_fini(fd, wts[part]);
    }

    rocket_bo_prep(fd, &out, 1, 0); memset(out.ptr, 0, out.size); rocket_bo_fini(fd, &out);

    /* PING-PONG (not in-place): job0 out=partial0 (plain); job1 out2=partial1+out. */
    if (run_partial(fd, M, Kh, N, &in0, &wt0, &out,  &rc, 0, NULL)) return 1;
    if (run_partial(fd, M, Kh, N, &in1, &wt1, &out2, &rc, 1, &out)) return 1;

    /* read the final buffer (out2) once and verify against full-K CPU reference */
    rocket_bo_prep(fd, &out2, 0, 0);
    _Float16 *ob = out2.ptr;

    /* CPU references for each hypothesis so a FAIL self-classifies:
     *   p0 = A0·B0^T (job0 alone), p1 = A1·B1^T, full = p0+p1 (ref). */
    float *p0 = calloc((size_t)M*N, sizeof(float));
    float *p1 = calloc((size_t)M*N, sizeof(float));
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float a0 = 0, a1 = 0;
            for (int k = 0; k < Kh; k++)   a0 += (float)A[m*K+k]      * (float)B[n*K+k];
            for (int k = Kh; k < K; k++)   a1 += (float)A[m*K+k]      * (float)B[n*K+k];
            p0[m*N+n] = a0; p1[m*N+n] = a1;
        }

    float e_full=0, e_p0=0, e_p1=0, max_abs=0, max_rel=0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float got = (float)ob[feature_data(N, M, 1, 8, n+1, m+1, 1)];
            e_full += fabsf(got - ref[m*N+n]);
            e_p0   += fabsf(got - p0[m*N+n]);
            e_p1   += fabsf(got - p1[m*N+n]);
            float ad = fabsf(got - ref[m*N+n]);
            float rd = ad / (fabsf(ref[m*N+n]) + 1e-6f);
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) max_rel = rd;
        }
    int pass = (max_abs < 0.5f || max_rel < 0.05f);
    printf("verify: max_abs=%.4f max_rel=%.4f -> %s\n", max_abs, max_rel, pass ? "PASS" : "FAIL");
    if (!pass) {
        printf("  classify (sum|err|): vs full(p0+p1)=%.3f  vs p0-only=%.3f  vs p1-only=%.3f\n",
               e_full, e_p0, e_p1);
        printf("  -> %s\n",
               e_p1 < e_full*0.5f ? "out==p1: add NOT happening (job1 overwrote, ERDMA off/ignored)" :
               e_p0 < e_full*0.5f ? "out==p0: job1 conv produced 0 / wrong operand" :
               "neither clean partial: scaled or garbage operand (sweep ROCKET_EW_CFG/ERDMA/RDMA_FMC)");
        /* The operand the NPU actually read back == got - p1. Compare to the
         * expected p0 to expose the layout error (shift/transpose/wrong row). */
        printf("  operand_read (=got-p1) vs expected p0, per row m:\n");
        for (int m = 0; m < M; m++) {
            printf("   m=%d read:", m);
            for (int n = 0; n < N; n++) {
                float got = (float)ob[feature_data(N, M, 1, 8, n+1, m+1, 1)];
                printf(" %6.2f", got - p1[m*N+n]);
            }
            printf("\n   m=%d  p0 :", m);
            for (int n = 0; n < N; n++) printf(" %6.2f", p0[m*N+n]);
            printf("\n");
        }
    }
    rocket_bo_fini(fd, &out2);
    free(p0); free(p1);

    rocket_bo_free(fd, &in0); rocket_bo_free(fd, &in1);
    rocket_bo_free(fd, &wt0); rocket_bo_free(fd, &wt1);
    rocket_bo_free(fd, &out); rocket_bo_free(fd, &out2); rocket_bo_free(fd, &rc);
    rocket_close(fd);
    free(A); free(B); free(ref);
    return 0;
}
