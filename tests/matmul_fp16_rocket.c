// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_fp16_rocket.c — standalone fp16 x fp16 -> fp16/fp32 single-task test on
 * the rocket NPU. Exercises gen_matmul_fp16() / weight_fp16() / feature_data()
 * (npu_regcmd.c) through the rocket_npu shim. Usage: matmul_fp16_rocket <M> <K> <N>
 *
 * Constraints inherited from the single-task generator (no tiling yet):
 *   M multiple of 4 (or 1), K multiple of 32, N multiple of 16 (or 1),
 *   and M*K + N*K must fit the 12 x 32KB CBUF banks (gen returns <0 otherwise).
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

/* regcmd_count = the TOTAL number of NPUOP words the PC must execute, INCLUDING
 * the 4-op control trailer ops[104..107] (OP_NONE / PC_REGISTER_AMOUNTS / OP_40
 * / OP_ENABLE). ops[107] = OP_ENABLE writes PC_OPERATION_ENABLE = 0x0d
 * (PC_ENABLE|PC_ENABLE_CNA|PC_ENABLE_DPU) which is what actually TRIGGERS the
 * compute; the kernel's own OPERATION_ENABLE=0x1 write only starts the PC
 * sequencer. If the PC stops before ops[107] the pipeline is configured but
 * never fired -> DPU never completes -> "NPU job timed out" (observed with 104).
 *
 * The hardware PC_DATA_AMOUNT confirms the count (scale = 2):
 *   (regcmd_count + 1)/2 - 1 = (108 + 1)/2 - 1 = 53. */
#define REGCMD_COUNT 108

/* Report any nonzero (negative-errno) return from a shim call. */
#define CK(call) do { int _r = (call); \
    if (_r) fprintf(stderr, "%s -> %d (%s)\n", #call, _r, strerror(-_r)); } while (0)

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

static void ref_matmul(int m, int k, int n, _Float16 *A, _Float16 *B, _Float16 *C) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int l = 0; l < k; l++) s += (float)A[i*k+l] * (float)B[j*k+l];
            C[i*n+j] = (_Float16)s;
        }
}

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s <M> <K> <N>\n", argv[0]); return -1; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);

    int fd = rocket_open();
    if (fd < 0) return fd;

    rocket_bo guard = {0}, regcmd = {0}, input = {0}, weights = {0}, output = {0};
    int ret = 0;
    /* PROBE (IOVA-0 hypothesis): the first allocation lands at NPU IOVA 0x0
     * (regcmd=0 in the first runs). If the PC treats BASE_ADDRESS=0 as "no
     * program" or IOVA 0 is unmapped, it executes an empty program and writes
     * nothing — exactly the observed untouched (0xAA) output. Reserve IOVA 0
     * with a throwaway BO so every real buffer gets a nonzero address. */
    rocket_bo_alloc(fd, 4096, &guard);
    ret |= rocket_bo_alloc(fd, 4096,                  &regcmd);
    ret |= rocket_bo_alloc(fd, (size_t)M*K*sizeof(_Float16), &input);
    ret |= rocket_bo_alloc(fd, (size_t)N*K*sizeof(_Float16), &weights);
    ret |= rocket_bo_alloc(fd, (size_t)M*N*sizeof(_Float16), &output);
    if (ret) { fprintf(stderr, "bo alloc failed\n"); goto out; }
    printf("dma  in=%lx  w=%lx  out=%lx  regcmd=%lx\n",
           input.dma_address, weights.dma_address, output.dma_address, regcmd.dma_address);

    /* The NPU programs addresses into 32-bit registers (NPUOP value is 32-bit,
     * and matmul_params_t stores *_dma as uint32_t). Every BO's device VA must
     * therefore fit in the low 4GB of the NPU IOVA space. Verify rather than
     * silently truncate. If this
     * trips, the rocket driver is handing out >32-bit IOVAs and the address
     * programming needs revisiting (possible shift, or a CREATE_BO flag to
     * request a 32-bit-addressable BO). */
    if ((input.dma_address | weights.dma_address | output.dma_address | regcmd.dma_address) >> 32) {
        fprintf(stderr, "ERROR: a BO dma_address exceeds 32 bits — see note above\n");
        ret = -1; goto out;
    }

    /* Build the register program (npu_regcmd.c).
     * Sized for ~108 ops + the DPU_RDMA block. */
    uint64_t npu_regs[256] = {0};
    matmul_params_t p = {
        .m = M, .k = K, .n = N,
        .input_dma = input.dma_address, .weights_dma = weights.dma_address,
        .output_dma = output.dma_address, .tasks = npu_regs, .fp32tofp16 = 1,
    };
    if ((ret = gen_matmul_fp16(&p)) != 0) {
        fprintf(stderr, "gen_matmul_fp16 = %d (likely exceeds CBUF banks — needs tiling)\n", ret);
        goto out;
    }
    printf("regcmd ops = %u (incl. DPU_RDMA block)\n", p.task_count);

    /* Fill device buffers (CPU writes -> flush via fini before submit).
     * These BOs have no in-flight job yet, so timeout 0 is fine here. */
    CK(rocket_bo_prep(fd, &regcmd, 1, 0));
    CK(rocket_bo_prep(fd, &input,  1, 0));
    CK(rocket_bo_prep(fd, &weights,1, 0));

    memcpy(regcmd.ptr, npu_regs, (size_t)p.task_count * sizeof(uint64_t));

    /* host reference inputs */
    _Float16 *A = malloc((size_t)M*K*2), *B = malloc((size_t)N*K*2), *C = malloc((size_t)M*N*2);
    /* Log (and allow overriding) the RNG seed so a failing run is reproducible:
     * re-run with ROCKET_TEST_SEED=<printed value> to regenerate the exact inputs. */
    const char *seed_env = getenv("ROCKET_TEST_SEED");
    unsigned seed = seed_env ? (unsigned)strtoul(seed_env, NULL, 0) : (unsigned)time(NULL);
    printf("seed = %u%s\n", seed, seed_env ? " (from ROCKET_TEST_SEED)" : "");
    srand(seed);
    for (int i = 0; i < M*K; i++) A[i] = (_Float16)(int)(10.0 * rand()/(float)RAND_MAX);
    for (int i = 0; i < N*K; i++) B[i] = (_Float16)(int)(10.0 * rand()/(float)RAND_MAX);

    /* repack into NPU native layouts (reused from npu_matmul.c) */
    _Float16 *wdst = weights.ptr, *idst = input.ptr;
    memset(input.ptr, 0, input.size); memset(weights.ptr, 0, weights.size);
    for (int n = 1; n <= N; n++)
        for (int k = 1; k <= K; k++) wdst[weight_fp16(K, n, k)] = B[(n-1)*K + (k-1)];
    for (int m = 1; m <= M; m++)
        for (int k = 1; k <= K; k++) idst[feature_data(K, M, 1, 8, k, m, 1)] = A[(m-1)*K + (k-1)];

    CK(rocket_bo_fini(fd, &regcmd));
    CK(rocket_bo_fini(fd, &input));
    CK(rocket_bo_fini(fd, &weights));

    /* Sentinel-fill output so the next run distinguishes the failure modes:
     *   pattern 0xAA intact -> NPU never wrote output (job didn't run/route)
     *   all zeros           -> NPU wrote, but computed zeros (addr/enable)
     *   correct values      -> success */
    CK(rocket_bo_prep(fd, &output, 1, 0));
    memset(output.ptr, 0xAA, output.size);
    CK(rocket_bo_fini(fd, &output));

    ref_matmul(M, K, N, A, B, C);

    /* Submit: NPU reads input+weights+regcmd, writes output. */
    uint32_t in_h[]  = { input.handle, weights.handle, regcmd.handle };
    uint32_t out_h[] = { output.handle };
    int64_t t0 = now_us();
    ret = rocket_submit_matmul(fd, &regcmd, p.task_count,
                               in_h, 3, out_h, 1, 6000);
    int64_t us = now_us() - t0;
    printf("ROCKET_SUBMIT = %d  (%.3f ms)\n", ret, us / 1000.0);
    if (ret) goto out_free;

    /* SUBMIT is asynchronous (note the sub-ms "submit" time above): it returns
     * once the job is queued. PREP_BO on the output BO is what blocks on the
     * job's completion fence, so it MUST get a real timeout — passing 0 does not
     * wait and reads the BO before the NPU writes it (the original all-zeros). */
    int prc = rocket_bo_prep(fd, &output, 0, 2000000000LL); /* 2s */
    if (prc) {
        fprintf(stderr, "PREP_BO(output) = %d (%s): job never completed — DPU "
                "completion IRQ likely never fired (block-enable / HW path)\n",
                prc, strerror(-prc));
        ret = prc; goto out_free;
    }

    /* Did the NPU touch the output at all? */
    int touched = 0;
    for (size_t i = 0; i < output.size; i++)
        if (((uint8_t *)output.ptr)[i] != 0xAA) { touched = 1; break; }
    if (!touched)
        printf("DIAGNOSTIC: output still 0xAA sentinel — NPU did not write output\n");

    _Float16 *od = output.ptr;
    int bad = 0;
    for (int m = 1; m <= M && bad < 10; m++)
        for (int n = 1; n <= N; n++) {   /* 1-based: must include the final column n==N */
            _Float16 act = od[feature_data(N, M, 1, 8, n, m, 1)];
            _Float16 exp = C[(m-1)*N + (n-1)];
            if (act != exp) {
                printf("mismatch m=%d n=%d exp=%f act=%f\n", m, n, (float)exp, (float)act);
                if (++bad >= 10) break;
            }
        }
    rocket_bo_fini(fd, &output);
    printf(bad ? "FAILED (%d mismatches shown)\n" : "OK: [%d,%d]x[%d,%d]\n",
           bad ? bad : M, bad ? 0 : K, N, K);

out_free:
    free(A); free(B); free(C);
out:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &input);
    rocket_bo_free(fd, &weights); rocket_bo_free(fd, &output);
    rocket_close(fd);
    return ret;
}
