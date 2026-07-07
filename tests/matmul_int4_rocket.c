// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int4_rocket.c — standalone int4 x int4 -> int16 readiness
 * test + ENCODING SWEEP harness on the rocket driver. The int4 analogue of
 * matmul_int8_rocket, but int4's register encodings are NOT documented (no legacy
 * generator), so this is built to CRACK them on HW by sweep, staged so each
 * unknown is isolated:
 *
 *   STAGE 1 — does an int4 program RUN and WRITE int16 output? (precision value +
 *     output geometry, LAYOUT-INDEPENDENT). Watch for: WAIT TIMEOUT (bad/invalid
 *     precision value -> DPU never completes) vs "output still 0xAA" (ran but
 *     wrote nothing) vs "output touched" (precision + DPU enable correct). Sweep:
 *       ROCKET_INT4_PREC      in/proc precision (try 3, 6, 7)
 *       ROCKET_INT4_OUT_PREC  output precision  (try 1=int16)
 *       ROCKET_INT4_SIZE_E / ROCKET_INT4_SURF_MULT  int16 output stride (try 1/2, then 3/4, 7/8)
 *     A "touched, int16-strided" result (even with WRONG values) clears stage 1.
 *
 *   STAGE 2 — are the VALUES correct? (weight/feature NIBBLE packing + C2=32
 *     feature / C2=? output cube). Once stage 1 writes, fix the packing:
 *       ROCKET_INT4_NIBBLE_HILO  swap which logical elem is the low vs high nibble
 *       ROCKET_INT4_OUT_C2       int16 output cube (try 8, then 4, 16)
 *     The weight_int4 ordinal layout (npu_regcmd.c, currently == weight_int8)
 *     is the prime suspect — int4 N-align is 64 (vs the guessed 32).
 *
 * Small shapes keep the int16 output UNSATURATED so the compare to the int32
 * reference is exact: with K<=256 and int4 in [-8,7], |sum| <= 64*256 = 16384 <
 * 32767. Use M=4 K=32 N=64 to start (one K-group, one int4 N-group of 64).
 *
 * Usage: matmul_int4_rocket <M> <K> <N>   (M%4||1, K%32, N%32; N=64 recommended)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

#define CK(call) do { int _r = (call); \
    if (_r) fprintf(stderr, "%s -> %d (%s)\n", #call, _r, strerror(-_r)); } while (0)

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}
/* signed int4 in [-8, 7] */
static int8_t rand_i4(void) { return (int8_t)(rand() % 16 - 8); }

static int env_int(const char *name, int def) {
    const char *e = getenv(name); return e ? (int)strtol(e, NULL, 0) : def;
}

/* Write a signed int4 value into a nibble-packed buffer at nibble index `idx`.
 * byte = idx/2; even idx -> low nibble, odd -> high (swappable via hilo). */
static void put_nibble(uint8_t *buf, size_t idx, int8_t v, int hilo) {
    uint8_t nib = (uint8_t)(v & 0xF);
    size_t byte = idx >> 1;
    int high = (int)(idx & 1);
    if (hilo) high = !high;
    if (high) buf[byte] = (uint8_t)((buf[byte] & 0x0F) | (nib << 4));
    else      buf[byte] = (uint8_t)((buf[byte] & 0xF0) | nib);
}

/* int4 x int4 -> int32 reference (exact; int16 output equals this when unsaturated) */
static void ref_matmul_int4(int m, int k, int n,
                            const int8_t *A, const int8_t *B, int32_t *C) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int32_t sum = 0;
            for (int l = 0; l < k; l++) sum += (int32_t)A[i*k + l] * (int32_t)B[j*k + l];
            C[i*n + j] = sum;
        }
}

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s <M> <K> <N>  (try 4 32 64)\n", argv[0]); return -1; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    if (M <= 0 || (M % 4 != 0 && M != 1)) { fprintf(stderr, "M must be %%4 or 1\n"); return -1; }
    if (K <= 0 || K % 32 != 0) { fprintf(stderr, "K must be %%32\n"); return -1; }
    if (N <= 0 || N % 32 != 0) { fprintf(stderr, "N must be %%32 (int4 prefers %%64)\n"); return -1; }

    const int hilo   = env_int("ROCKET_INT4_NIBBLE_HILO", 0);
    const int out_c2 = env_int("ROCKET_INT4_OUT_C2", 8);   /* int16 output cube guess */

    printf("int4 TEST: C[%d,%d] = A[%d,%d] x B[%d,%d]^T (int4xint4->int16)\n", M, N, M, K, N, K);
    printf("knobs: PREC=%d OUT_PREC=%d SIZE_E=%d SURF_MULT=%d DENTRIES_DIV=%d NIBBLE_HILO=%d OUT_C2=%d\n",
           env_int("ROCKET_INT4_PREC", 3), env_int("ROCKET_INT4_OUT_PREC", 1),
           env_int("ROCKET_INT4_SIZE_E", 1), env_int("ROCKET_INT4_SURF_MULT", 2),
           env_int("ROCKET_INT4_DENTRIES_DIV", 128), hilo, out_c2);

    int fd = rocket_open();
    if (fd < 0) return fd;

    rocket_bo guard = {0}, regcmd = {0}, input = {0}, weights = {0}, output = {0};
    int ret = 0;
    size_t in_bytes  = (size_t)(M * K + 1) / 2;   /* int4 nibbles -> bytes */
    size_t wt_bytes  = (size_t)(N * K + 1) / 2;
    size_t out_bytes = (size_t)M * N * sizeof(int16_t);
    rocket_bo_alloc(fd, 4096, &guard);
    ret |= rocket_bo_alloc(fd, 4096,      &regcmd);
    ret |= rocket_bo_alloc(fd, in_bytes,  &input);
    ret |= rocket_bo_alloc(fd, wt_bytes,  &weights);
    ret |= rocket_bo_alloc(fd, out_bytes, &output);
    if (ret) { fprintf(stderr, "bo alloc failed\n"); goto out; }
    printf("dma  in=%lx w=%lx out=%lx regcmd=%lx  (in=%zuB wt=%zuB out=%zuB)\n",
           input.dma_address, weights.dma_address, output.dma_address, regcmd.dma_address,
           in_bytes, wt_bytes, out_bytes);
    if ((input.dma_address | weights.dma_address | output.dma_address | regcmd.dma_address) >> 32) {
        fprintf(stderr, "ERROR: a BO dma_address exceeds 32 bits\n"); ret = -1; goto out;
    }

    uint64_t npu_regs[256] = {0};
    matmul_params_t p = {
        .m = (uint16_t)M, .k = (uint16_t)K, .n = (uint16_t)N,
        .input_dma = (uint32_t)input.dma_address,
        .weights_dma = (uint32_t)weights.dma_address,
        .output_dma = (uint32_t)output.dma_address,
        .tasks = npu_regs,
    };
    if ((ret = gen_matmul_int4(&p)) != 0) {
        fprintf(stderr, "gen_matmul_int4 = %d (CBUF banks? -1 fd, -2 weight/kernel)\n", ret);
        goto out;
    }
    printf("regcmd ops = %u\n", p.task_count);

    int8_t  *A = malloc((size_t)M * K);
    int8_t  *B = malloc((size_t)N * K);
    int32_t *C = malloc((size_t)M * N * sizeof(int32_t));
    if (!A || !B || !C) { fprintf(stderr, "host malloc failed\n"); ret = -1; goto out_free; }
    srand(1234);
    for (int i = 0; i < M*K; i++) A[i] = rand_i4();
    for (int i = 0; i < N*K; i++) B[i] = rand_i4();

    CK(rocket_bo_prep(fd, &regcmd, 1, 0));
    CK(rocket_bo_prep(fd, &input,  1, 0));
    CK(rocket_bo_prep(fd, &weights,1, 0));
    memcpy(regcmd.ptr, npu_regs, (size_t)p.task_count * sizeof(uint64_t));
    memset(input.ptr, 0, input.size);
    memset(weights.ptr, 0, weights.size);
    /* pack into NPU native int4 nibble layouts: weight_int4, feature cube C2=32 */
    for (int n = 1; n <= N; n++)
        for (int k = 1; k <= K; k++)
            put_nibble(weights.ptr, (size_t)weight_int4(K, n, k), B[(n-1)*K + (k-1)], hilo);
    for (int m = 1; m <= M; m++)
        for (int k = 1; k <= K; k++)
            put_nibble(input.ptr, (size_t)feature_data(K, M, 1, 32, k, m, 1), A[(m-1)*K + (k-1)], hilo);
    CK(rocket_bo_fini(fd, &regcmd));
    CK(rocket_bo_fini(fd, &input));
    CK(rocket_bo_fini(fd, &weights));

    CK(rocket_bo_prep(fd, &output, 1, 0));
    memset(output.ptr, 0xAA, output.size);     /* sentinel */
    CK(rocket_bo_fini(fd, &output));

    ref_matmul_int4(M, K, N, A, B, C);

    uint32_t in_h[]  = { input.handle, weights.handle, regcmd.handle };
    uint32_t out_h[] = { output.handle };
    int64_t t0 = now_us();
    ret = rocket_submit_matmul(fd, &regcmd, p.task_count, in_h, 3, out_h, 1, 6000);
    printf("ROCKET_SUBMIT = %d  (%.3f ms)\n", ret, (now_us() - t0) / 1000.0);
    if (ret) goto out_free;

    int prc = rocket_bo_prep(fd, &output, 0, 2000000000LL);
    if (prc) {
        fprintf(stderr, "STAGE1 FAIL: PREP_BO(output) = %d (%s) — job never completed "
                "(precision value likely invalid; sweep ROCKET_INT4_PREC=3/6/7)\n",
                prc, strerror(-prc));
        ret = prc; goto out_free;
    }

    /* STAGE 1: touched? where? */
    size_t first = out_bytes; int touched = 0;
    for (size_t i = 0; i < out_bytes; i++)
        if (((uint8_t *)output.ptr)[i] != 0xAA) { touched = 1; first = i; break; }
    if (!touched) {
        printf("STAGE1: output still 0xAA — ran but wrote NOTHING (DPU enable / out_precision). "
               "Sweep ROCKET_INT4_OUT_PREC, then SIZE_E/SURF_MULT.\n");
        rocket_bo_fini(fd, &output); ret = -1; goto out_free;
    }
    printf("STAGE1 OK: output TOUCHED (first non-0xAA byte @ %zu of %zu)\n", first, out_bytes);

    /* STAGE 2: values. read int16 at the guessed output cube C2. */
    int16_t *od = output.ptr;
    int bad = 0, shown = 0; long maxabs = 0; int any_match = 0;
    int row0_prefix = 0, prefix_open = 1;    /* contiguous correct cols in row m=1 */
    int row0_correct = 0;                    /* total correct cols in row m=1      */
    int sentinel = 0;                        /* outputs left as 0xAAAA (untouched)  */
    for (int m = 1; m <= M; m++)
        for (int n = 1; n <= N; n++) {
            int16_t act = od[feature_data(N, M, 1, out_c2, n, m, 1)];
            int32_t exp = C[(m-1)*N + (n-1)];
            if ((uint16_t)act == 0xAAAA) sentinel++;
            if ((int32_t)act == exp) {
                any_match = 1;
                if (m == 1) { row0_correct++; if (prefix_open) row0_prefix++; }
            } else if (m == 1) {
                prefix_open = 0;
            }
            long d = labs((long)act - (long)exp);
            if (d > maxabs) maxabs = d;
            if ((int32_t)act != exp) {
                if (shown < 8) { printf("  mismatch m=%d n=%d exp=%d act=%d\n", m, n, exp, (int)act); shown++; }
                bad++;
            }
        }
    rocket_bo_fini(fd, &output);
    printf("STAGE2: %d/%d wrong, max_abs=%ld, any_match=%d | row0: %d/%d correct "
           "(contiguous prefix n=1..%d) | %d sentinel(0xAAAA)\n",
           bad, M*N, maxabs, any_match, row0_correct, N, row0_prefix, sentinel);
    if (bad == 0) { printf("==> PASS (bit-exact int4xint4->int16)\n"); ret = 0; }
    else {
        printf("==> values wrong (stage1 cleared). Crack packing: ROCKET_INT4_NIBBLE_HILO=1, "
               "ROCKET_INT4_OUT_C2=4/16, and the weight_int4 layout (N-group 64?).\n");
        ret = -1;
    }

out_free:
    free(A); free(B); free(C);
out:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &input);
    rocket_bo_free(fd, &weights); rocket_bo_free(fd, &output);
    rocket_close(fd);
    return ret;
}
