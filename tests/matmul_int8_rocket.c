// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int8_rocket.c — standalone int8 x int8 -> int32 readiness test on the
 * mainline *rocket* driver. THIS IS THE FOUNDATIONAL int8 TEST: nothing downstream
 * (int8 tiling, backend quant/dequant, W8A8+Hadamard) proceeds until this PASSES on
 * the NPU. Same prove-the-HW-first discipline that de-risked the fp16 path.
 *
 * It is matmul_fp16_rocket.c's submit skeleton (rocket_npu shim: open / bo_alloc
 * / prep / fini / submit_matmul) with the int8 specifics:
 *   - int8 input + int8 weights BOs (1 B/elem), int32 output BO (4 B/elem)
 *   - gen_matmul_int8() instead of gen_matmul_fp16()
 *   - feature-data input atom C2=16 (int8) and weight k-group 32 (weight_int8)
 *   - int32-output cube C2=4 on readback
 *   - CPU reference matmul_int (int8xint8 -> int32)
 *
 * Usage: matmul_int8_rocket <M> <K> <N>
 *   M multiple of 4 (or 1), K multiple of 32, N multiple of 32, and the
 *   single-task generator must fit the 12 x 32KB CBUF banks (gen returns <0).
 *
 *   int8 N-alignment is 32, NOT 16 (fp16's value). The int8 weight k-group is
 *   32 (weight_int8), so weight_int8 only packs into exactly N*K when N%32==0;
 *   with N=16 it reserves a full 32-kernel group, over-runs the logical buffer,
 *   and the NPU (reading N=16 kernels) sees a layout that disagrees -> garbage.
 *   Confirmed on HW: 4x64x16 FAILED (N=16), 256x384x256 PASSED.
 *
 * size_e / surf_add open question: gen_matmul_int8 defaults to the legacy
 * template's size_e=7 / surf_add*8 (== 8-byte output element), which is
 * inconsistent with int32=4 B and the fp16 path's "size_e=bytes-1, mult=bytes"
 * encoding. If this test FAILS (esp. output untouched, or values right but
 * strided/duplicated), re-run with the fp16-consistent values:
 *     ROCKET_INT8_SIZE_E=3 ROCKET_INT8_SURF_MULT=4 sudo -E ./build/matmul_int8_rocket M K N
 * One build, both hypotheses. See npu_regcmd.c gen_matmul_int8 header.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

/* Report any nonzero (negative-errno) return from a shim call. */
#define CK(call) do { int _r = (call); \
    if (_r) fprintf(stderr, "%s -> %d (%s)\n", #call, _r, strerror(-_r)); } while (0)

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

/* int8 x int8 -> int32 reference. A is [M,K] row-major, B is [N,K] row-major
 * (each weight kernel n is a contiguous K-vector), C is [M,N] row-major.
 * Accumulate in int64 — NOT float like the legacy matmul_int(): with full int8
 * range and K=4096 a column sum reaches ~67M, past float's 2^24 exact limit, so
 * a float reference would mis-compare against the EXACT int32 the HW produces.
 * (int32 itself doesn't overflow here: 128*128*4096 = 67M < 2^31.) */
static void ref_matmul_int8(int m, int k, int n,
                            const int8_t *A, const int8_t *B, int32_t *C) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int64_t sum = 0;
            for (int l = 0; l < k; l++)
                sum += (int32_t)A[i*k + l] * (int32_t)B[j*k + l];
            C[i*n + j] = (int32_t)sum;
        }
}

/* full signed int8 range, exercises data_sign=1 */
static int8_t rand_i8(void) { return (int8_t)(rand() % 256 - 128); }

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s <M> <K> <N>\n", argv[0]); return -1; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);

    /* int8 single-task alignment (see header). Fail fast & loud instead of
     * silently scattering weights/features into a layout the NPU can't read. */
    if (M <= 0 || (M % 4 != 0 && M != 1)) {
        fprintf(stderr, "M=%d invalid: must be a multiple of 4 (or 1)\n", M); return -1; }
    if (K <= 0 || K % 32 != 0) {
        fprintf(stderr, "K=%d invalid: must be a multiple of 32\n", K); return -1; }
    if (N <= 0 || N % 32 != 0) {
        fprintf(stderr, "N=%d invalid: int8 weight k-group is 32 -> N must be a "
                "multiple of 32 (fp16's 16 is NOT enough)\n", N); return -1; }

    int fd = rocket_open();
    if (fd < 0) return fd;

    rocket_bo guard = {0}, regcmd = {0}, input = {0}, weights = {0}, output = {0};
    int ret = 0;
    /* Reserve NPU IOVA 0 with a throwaway BO so every real buffer is nonzero
     * (BASE_ADDRESS=0 reads as "no program"). Same guard as the fp16 test. */
    rocket_bo_alloc(fd, 4096, &guard);
    ret |= rocket_bo_alloc(fd, 4096,                          &regcmd);
    ret |= rocket_bo_alloc(fd, (size_t)M*K*sizeof(int8_t),    &input);
    ret |= rocket_bo_alloc(fd, (size_t)N*K*sizeof(int8_t),    &weights);
    ret |= rocket_bo_alloc(fd, (size_t)M*N*sizeof(int32_t),   &output);
    if (ret) { fprintf(stderr, "bo alloc failed\n"); goto out; }
    printf("dma  in=%lx  w=%lx  out=%lx  regcmd=%lx\n",
           input.dma_address, weights.dma_address, output.dma_address, regcmd.dma_address);

    /* Addresses are programmed into 32-bit NPUOP value fields / uint32_t *_dma —
     * every BO VA must fit the low 4GB of NPU IOVA space (else silent truncation). */
    if ((input.dma_address | weights.dma_address | output.dma_address | regcmd.dma_address) >> 32) {
        fprintf(stderr, "ERROR: a BO dma_address exceeds 32 bits — see fp16 test note\n");
        ret = -1; goto out;
    }

    /* Build the int8 register program (npu_regcmd.c). */
    uint64_t npu_regs[256] = {0};
    matmul_params_t p = {
        .m = (uint16_t)M, .k = (uint16_t)K, .n = (uint16_t)N,
        .input_dma = (uint32_t)input.dma_address,
        .weights_dma = (uint32_t)weights.dma_address,
        .output_dma = (uint32_t)output.dma_address,
        .tasks = npu_regs,
    };
    if ((ret = gen_matmul_int8(&p)) != 0) {
        fprintf(stderr, "gen_matmul_int8 = %d (likely exceeds CBUF banks — needs tiling)\n", ret);
        goto out;
    }
    printf("regcmd ops = %u (int8, incl. DPU_RDMA block)\n", p.task_count);

    /* host reference inputs: A [M,K], B [N,K] row-major */
    int8_t  *A = malloc((size_t)M*K);
    int8_t  *B = malloc((size_t)N*K);
    int32_t *C = malloc((size_t)M*N*sizeof(int32_t));
    if (!A || !B || !C) { fprintf(stderr, "host malloc failed\n"); ret = -1; goto out_free; }
    srand(time(NULL));
    for (int i = 0; i < M*K; i++) A[i] = rand_i8();
    for (int i = 0; i < N*K; i++) B[i] = rand_i8();

    /* Pack into NPU native int8 layouts: weights k-group 32 (weight_int8),
     * feature atom C2=16. Bracket the CPU writes with prep(write)/fini(flush). */
    CK(rocket_bo_prep(fd, &regcmd, 1, 0));
    CK(rocket_bo_prep(fd, &input,  1, 0));
    CK(rocket_bo_prep(fd, &weights,1, 0));

    memcpy(regcmd.ptr, npu_regs, (size_t)p.task_count * sizeof(uint64_t));

    int8_t *wdst = weights.ptr, *idst = input.ptr;
    memset(input.ptr, 0, input.size); memset(weights.ptr, 0, weights.size);
    for (int n = 1; n <= N; n++)
        for (int k = 1; k <= K; k++) wdst[weight_int8(K, n, k)] = B[(n-1)*K + (k-1)];
    for (int m = 1; m <= M; m++)
        for (int k = 1; k <= K; k++) idst[feature_data(K, M, 1, 16, k, m, 1)] = A[(m-1)*K + (k-1)];

    CK(rocket_bo_fini(fd, &regcmd));
    CK(rocket_bo_fini(fd, &input));
    CK(rocket_bo_fini(fd, &weights));

    /* Sentinel-fill output: 0xAA intact -> NPU never wrote; all-zero -> wrote but
     * computed nothing; correct int32 -> success. */
    CK(rocket_bo_prep(fd, &output, 1, 0));
    memset(output.ptr, 0xAA, output.size);
    CK(rocket_bo_fini(fd, &output));

    ref_matmul_int8(M, K, N, A, B, C);

    /* Submit: NPU reads input+weights+regcmd, writes int32 output. */
    uint32_t in_h[]  = { input.handle, weights.handle, regcmd.handle };
    uint32_t out_h[] = { output.handle };
    int64_t t0 = now_us();
    ret = rocket_submit_matmul(fd, &regcmd, p.task_count, in_h, 3, out_h, 1, 6000);
    int64_t us = now_us() - t0;
    printf("ROCKET_SUBMIT = %d  (%.3f ms)\n", ret, us / 1000.0);
    if (ret) goto out_free;

    /* SUBMIT is async; PREP_BO(output, read) blocks on the completion fence and
     * invalidates cache so we see the NPU's writes. Needs a real timeout. */
    int prc = rocket_bo_prep(fd, &output, 0, 2000000000LL); /* 2s */
    if (prc) {
        fprintf(stderr, "PREP_BO(output) = %d (%s): job never completed — DPU "
                "completion IRQ likely never fired (block-enable / HW path / MRDMA)\n",
                prc, strerror(-prc));
        ret = prc; goto out_free;
    }

    /* Did the NPU touch the output at all? (0xAA == 0xAAAAAAAA as int32) */
    int touched = 0;
    for (size_t i = 0; i < output.size; i++)
        if (((uint8_t *)output.ptr)[i] != 0xAA) { touched = 1; break; }
    if (!touched)
        printf("DIAGNOSTIC: output still 0xAA sentinel — NPU did not write output\n");

    int32_t *od = output.ptr;
    int bad = 0;
    for (int m = 1; m <= M && bad < 10; m++)
        for (int n = 1; n <= N; n++) {
            int32_t act = od[feature_data(N, M, 1, 4, n, m, 1)];   /* int32 cube C2=4 */
            int32_t exp = C[(m-1)*N + (n-1)];
            if (act != exp) {
                printf("mismatch m=%d n=%d exp=%d act=%d\n", m, n, exp, act);
                if (++bad >= 10) break;
            }
        }
    rocket_bo_fini(fd, &output);
    if (bad) { printf("FAILED (%d mismatches shown)\n", bad); ret = -1; }
    else     printf("OK: [%d,%d]x[%d,%d] int8->int32\n", M, K, N, K);

out_free:
    free(A); free(B); free(C);
out:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &input);
    rocket_bo_free(fd, &weights); rocket_bo_free(fd, &output);
    rocket_close(fd);
    return ret;
}
