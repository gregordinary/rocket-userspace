// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * ew_mul_rocket.c — fully-on-NPU elementwise binary op via a CONV MAIN feed.
 *
 * The blocker for a fully-on-NPU HardSwish/SiLU (x * gate(x)) is a two-tensor
 * elementwise MULTIPLY on the DPU. The Teflon RE
 * settled that the rocket DPU EW unit reads its 2nd operand ONLY combined with a
 * conv/CACC main feed (ERDMA EW_BASE + MRDMA SRC_BASE + COMB_USE(5)); a pure
 * flying-MRDMA-main + ERDMA-operand pair (the old gen_ew_mul_fp16) reads the
 * operand as 0. So this gate drives the proven conv-main EW path — the SAME
 * machinery as the fp16 K-accum eltwise-ADD (gen_matmul_fp16, accumulate=1) — and
 * just swaps the EW op:
 *
 *   main feed   = an IDENTITY matmul  C[M,N] = A[M,K] · I[N,K]^T  (K==N, W=I)
 *                 => C[m,n] = A[m,n]  (the conv reproduces A into CACC exactly)
 *   EW operand  = B  (read via ERDMA, output-cube layout, COMB_USE(5))
 *   EW op       = ADD  -> out = A + B   (the known-good K-accum encoding, de-risk)
 *               = MUL  -> out = A * B   (EW_OP_TYPE=1, the allbilly alu_case_mul
 *                                        encoding 0x108003C4)
 *
 * Because the EW op is selected purely by DPU_EW_CFG (env ROCKET_EW_CFG, already
 * plumbed through gen_matmul_fp16), this reuses the entire validated conv+ERDMA
 * path and only changes one register word. ADD must reproduce the K-accum result
 * (proving the operand layout is right); MUL is then the one new bit.
 *
 *   EW_CFG ADD = 0x108202C0  EW_ALU_ALGO(2) EW_OP_TYPE(0)        (rocket K-accum)
 *   EW_CFG MUL = 0x108003C4  EW_ALU_ALGO(0) EW_OP_TYPE(1) +OP_CVT_BYPASS (allbilly)
 *
 * The output self-classifies (out == A+B / A*B / A / const) so a FAIL says why.
 *
 * Build (CTest target ew_mul_rocket; skip-code 2 off-device):
 *   gcc -O2 -Iinclude tests/ew_mul_rocket.c src/rocket_npu.c src/npu_regcmd.c \
 *       -o ew_mul_rocket -lm
 * Run:
 *   sudo ./ew_mul_rocket            # mul, M=16 N=K=32, random A,B
 *   sudo ./ew_mul_rocket add        # the de-risk add (must match K-accum)
 *   sudo ./ew_mul_rocket mul 32 32  # custom M,N
 *   sudo ROCKET_EW_CFG=0x... ./ew_mul_rocket mul   # sweep an EW_CFG candidate
 *   sudo ROCKET_EW_OPERAND_CONST=2.0 ./ew_mul_rocket mul  # constant operand
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "rocket_activation.h"   /* rocket_ew_add_fp16 / rocket_ew_mul_fp16 (runtime) */

#define EW_CFG_ADD 0x108202C0u   /* rocket fp16 K-accum eltwise-ADD             */
#define EW_CFG_MUL 0x108003C4u   /* allbilly alu_case_mul (EW_OP_TYPE=1) for fp16 */

/* Runtime-level check of the public flat-vector ops (rocket_ew_add_fp16 /
 * rocket_ew_mul_fp16) — the residual-add + gate-multiply primitives. n crosses the
 * internal M_TILE so the M-tiling path is exercised. Bit-exact vs the fp16 host op. */
static int runtime_check(int fd)
{
    /* the gen-test above setenv()'s ROCKET_EW_CFG (gen_matmul_fp16 honors it as an
     * override) — clear it so the runtimes select the op from their own ew_mul flag. */
    unsetenv("ROCKET_EW_CFG");
    const int NS[] = { 256, 512, 2048, 40000 };  /* single-tile small..multi-tile */
    int fail = 0;
    enum { OP_ADD, OP_MUL, OP_SUB };
    struct { const char *nm; int op; } ops[] = { {"add", OP_ADD}, {"mul", OP_MUL}, {"sub", OP_SUB} };
    for (int s = 0; s < (int)(sizeof(NS)/sizeof(NS[0])); s++) {
        int n = NS[s];
        _Float16 *a = malloc((size_t)n * sizeof(_Float16));
        _Float16 *b = malloc((size_t)n * sizeof(_Float16));
        _Float16 *o = malloc((size_t)n * sizeof(_Float16));
        if (!a || !b || !o) { free(a); free(b); free(o); return 1; }
        for (int i = 0; i < n; i++) {
            a[i] = (_Float16)(((i * 7) % 13 - 6) * 0.25f);
            b[i] = (_Float16)(((i * 5) % 11 - 5) * 0.3f);
        }
        for (int k = 0; k < (int)(sizeof(ops)/sizeof(ops[0])); k++) {
            memset(o, 0, (size_t)n * sizeof(_Float16));
            int r = ops[k].op == OP_MUL ? rocket_ew_mul_fp16(fd, a, b, o, n)
                  : ops[k].op == OP_SUB ? rocket_ew_sub_fp16(fd, a, b, o, n)
                                        : rocket_ew_add_fp16(fd, a, b, o, n);
            if (r) { printf("runtime ew_%s(n=%d): ret=%d -> FAIL\n", ops[k].nm, n, r); fail = 1; continue; }
            double max_abs = 0; int bad = 0;
            for (int i = 0; i < n; i++) {
                float want = ops[k].op == OP_MUL ? (float)(_Float16)((float)a[i] * (float)b[i])
                           : ops[k].op == OP_SUB ? (float)(_Float16)((float)a[i] - (float)b[i])
                                                 : (float)(_Float16)((float)a[i] + (float)b[i]);
                double ad = fabs((float)o[i] - want);
                if (ad > max_abs) max_abs = ad;
                if (ad > 1e-3) bad++;
            }
            printf("runtime ew_%s(n=%d): max_abs=%.5f bad=%d -> %s\n",
                   ops[k].nm, n, max_abs, bad, bad ? "FAIL" : "PASS");
            if (bad) fail = 1;
        }
        free(a); free(b); free(o);
    }
    return fail;
}

int main(int argc, char **argv)
{
    int is_mul = 1;                 /* default: test the multiply              */
    int ai = 1;
    if (argc > 1 && (strcmp(argv[1], "add") == 0 || strcmp(argv[1], "mul") == 0)) {
        is_mul = (strcmp(argv[1], "mul") == 0);
        ai = 2;
    }
    int M = (argc > ai)   ? atoi(argv[ai])   : 16;
    int N = (argc > ai+1) ? atoi(argv[ai+1]) : 32;   /* N == K (square identity) */
    int K = N;

    if (K % 32 || N % 16 || (M % 4)) {
        fprintf(stderr, "need K%%32==0, N%%16==0, M%%4==0 (K==N here)\n");
        return 2;
    }

    /* Pick the EW op encoding. gen_matmul_fp16 reads ROCKET_EW_CFG; default it to
     * the op under test but let an explicit env override win (for sweeps). */
    if (!getenv("ROCKET_EW_CFG"))
        setenv("ROCKET_EW_CFG", is_mul ? "0x108003C4" : "0x108202C0", 1);

    const char *ce = getenv("ROCKET_EW_OPERAND_CONST");
    int use_const = ce != NULL;
    float cop = use_const ? (float)atof(ce) : 0.0f;

    printf("ew_%s test: out[%d,%d] = A %s B  (identity main, EW operand%s)\n",
           is_mul ? "mul" : "add", M, N, is_mul ? "*" : "+",
           use_const ? " = const" : "");

    /* host operands */
    _Float16 *A = malloc((size_t)M * N * sizeof(_Float16));
    _Float16 *B = malloc((size_t)M * N * sizeof(_Float16));
    for (int i = 0; i < M * N; i++) {
        A[i] = (_Float16)(((i * 7) % 13 - 6) * 0.25f);          /* ~[-1.5,1.5] */
        B[i] = use_const ? (_Float16)cop
                         : (_Float16)(((i * 5) % 11 - 5) * 0.3f); /* ~[-1.5,1.5] */
    }

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open failed (%d) — no NPU, skipping\n", fd);
                  free(A); free(B); return 2; }   /* SKIP_RETURN_CODE off-device */

    size_t in_sz  = (size_t)((M+3)/4*4)   * ((K+31)/32*32) * sizeof(_Float16) + 4096;
    size_t wt_sz  = (size_t)((N+15)/16*16) * ((K+31)/32*32) * sizeof(_Float16) + 4096;
    size_t out_sz = (size_t)((M+3)/4*4)   * ((N+15)/16*16) * sizeof(_Float16) + 4096;

    rocket_bo inb = {0}, wtb = {0}, opb = {0}, outb = {0}, rcb = {0};
    int e = 0;
    e |= rocket_bo_alloc(fd, in_sz,  &inb);
    e |= rocket_bo_alloc(fd, wt_sz,  &wtb);
    e |= rocket_bo_alloc(fd, out_sz, &opb);    /* the EW operand B (output layout) */
    e |= rocket_bo_alloc(fd, out_sz, &outb);   /* result (separate from operand)   */
    e |= rocket_bo_alloc(fd, 256 * sizeof(uint64_t), &rcb);
    if (e) { fprintf(stderr, "BO alloc failed\n"); return 1; }

    /* pack A into the input feature cube */
    rocket_bo_prep(fd, &inb, 1, 0); memset(inb.ptr, 0, inb.size);
    { _Float16 *ip = inb.ptr;
      for (int h = 1; h <= M; h++)
          for (int c = 1; c <= K; c++)
              ip[feature_data(K, M, 1, 8, c, h, 1)] = A[(size_t)(h-1)*N + (c-1)]; }
    rocket_bo_fini(fd, &inb);

    /* identity weights I[N,K] (N==K): W[n,k] = (n==k) */
    rocket_bo_prep(fd, &wtb, 1, 0); memset(wtb.ptr, 0, wtb.size);
    { _Float16 *wp = wtb.ptr;
      for (int kk = 1; kk <= N; kk++)
          for (int c = 1; c <= K; c++)
              wp[weight_fp16(K, kk, c)] = (_Float16)(kk == c ? 1.0f : 0.0f); }
    rocket_bo_fini(fd, &wtb);

    /* operand B packed in the OUTPUT cube layout (same indexing as the readback) */
    rocket_bo_prep(fd, &opb, 1, 0); memset(opb.ptr, 0, opb.size);
    { _Float16 *bp = opb.ptr;
      for (int m = 0; m < M; m++)
          for (int n = 0; n < N; n++)
              bp[feature_data(N, M, 1, 8, n+1, m+1, 1)] = B[(size_t)m*N + n]; }
    rocket_bo_fini(fd, &opb);

    rocket_bo_prep(fd, &outb, 1, 0); memset(outb.ptr, 0, outb.size); rocket_bo_fini(fd, &outb);

    /* one accumulate job: main = identity matmul (=A), EW operand = B */
    uint64_t ops[256] = {0};
    matmul_params_t p = { .m = M, .k = K, .n = N,
        .input_dma   = (uint32_t)inb.dma_address,
        .weights_dma = (uint32_t)wtb.dma_address,
        .output_dma  = (uint32_t)outb.dma_address,
        .tasks = ops, .fp32tofp16 = 1,
        .accumulate = 1, .add_dma = (uint32_t)opb.dma_address };
    if (gen_matmul_fp16(&p)) { fprintf(stderr, "gen failed\n"); return 1; }

    { uint32_t ew=0, erd=0, fmc=0;
      for (uint32_t z = 0; z < p.task_count; z++) {
          uint16_t rg = ops[z] & 0xFFFF; uint32_t v = (ops[z] >> 16) & 0xFFFFFFFF;
          if (rg==0x4070) ew=v; else if (rg==0x5034) erd=v; else if (rg==0x5044) fmc=v;
      }
      printf("[regs] EW_CFG=0x%08x ERDMA=0x%08x FMC=0x%08x\n", ew, erd, fmc); }

    rocket_bo_prep(fd, &rcb, 1, 0);
    memcpy(rcb.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &rcb);

    rocket_task_desc t = { (uint32_t)rcb.dma_address, p.task_count };
    uint32_t inh[]  = { inb.handle, wtb.handle, rcb.handle, opb.handle };
    uint32_t outh[] = { outb.handle };
    if (rocket_submit_tasks(fd, &t, 1, inh, 4, outh, 1)) { fprintf(stderr, "submit failed\n"); return 1; }
    if (rocket_bo_prep(fd, &outb, 0, 2000000000LL)) { fprintf(stderr, "job timed out\n"); return 1; }

    /* classify: sum|err| of the readback vs each hypothesis */
    _Float16 *ob = outb.ptr;
    double e_add = 0, e_mul = 0, e_ident = 0, max_abs = 0;
    float want_max = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float got = (float)ob[feature_data(N, M, 1, 8, n+1, m+1, 1)];
            float a = (float)A[(size_t)m*N+n], b = (float)B[(size_t)m*N+n];
            float r_add = (float)(_Float16)(a + b);
            float r_mul = (float)(_Float16)(a * b);
            e_add   += fabsf(got - r_add);
            e_mul   += fabsf(got - r_mul);
            e_ident += fabsf(got - a);
            float want = is_mul ? r_mul : r_add;
            float ad = fabsf(got - want);
            if (ad > max_abs) max_abs = ad;
            if (fabsf(want) > want_max) want_max = fabsf(want);
        }
    rocket_bo_fini(fd, &outb);

    printf("sum|err|: vs A+B=%.3f  vs A*B=%.3f  vs A(operand ignored)=%.3f\n",
           e_add, e_mul, e_ident);
    const char *got_op =
        e_mul   < 0.5 ? "A*B (MULTIPLY)" :
        e_add   < 0.5 ? "A+B (ADD)" :
        e_ident < 0.5 ? "A (operand read as 0 / EW inert)" : "garbage";
    printf("NPU computed: %s\n", got_op);

    /* PASS iff the readback matches the REQUESTED op within fp16 tolerance */
    int pass = (max_abs < 0.05f * (want_max + 1.0f));
    printf("ew_%s verify: max_abs=%.4f -> %s\n",
           is_mul ? "mul" : "add", max_abs, pass ? "PASS" : "FAIL");

    rocket_bo_free(fd, &inb); rocket_bo_free(fd, &wtb); rocket_bo_free(fd, &opb);
    rocket_bo_free(fd, &outb); rocket_bo_free(fd, &rcb);
    free(A); free(B);

    /* also exercise the public flat-vector runtimes (residual add + gate mul) */
    int rt = runtime_check(fd);
    rocket_close(fd);
    return (pass && rt == 0) ? 0 : 1;
}
