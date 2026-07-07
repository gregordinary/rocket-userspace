// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_bf16_rocket.c — standalone bf16 x bf16 -> fp32 FEASIBILITY
 * test + encoding sweep on the rocket driver. The bf16 analogue of
 * matmul_int16_rocket, but FLOAT: it compares against an fp32 reference within a
 * TOLERANCE (HW tiling reorders the sum), not bit-exact.
 *
 * WHY bf16 (the payoff): bf16 has fp32's 8-bit EXPONENT at fp16's 2-byte cost. If
 * the NPU MAC does bf16 *matmul*, we can DELETE the per-row activation amax-scan +
 * scale/unscale that exists only because fp16's 5-bit exponent overflows on Gemma
 * activations. So this is a correctness simplification, not just a new dtype.
 *
 * FEASIBILITY OUTLOOK (Step-0 recon): STRONG, unlike int16. bf16 runs at 512x3
 * bf16 MAC operations per cycle (same rate as fp16), and the precision-field tables
 * name "3'd3: bfloat 16" in the CNA CONV_CON1, DPU data-format, and DPU_RDMA
 * tables — so bf16 = precision value 3 is well-established. bf16 accumulates to fp32,
 * so the output reuses the fp16 path's PROVEN fp32-out writer (which iterates full
 * M×N) — sidestepping the exact failure that made int16 output-only-1-tile.
 *
 *   STAGE 1 — does a bf16 program RUN and WRITE fp32 output? (precision value +
 *     output geometry, LAYOUT-INDEPENDENT). Watch for: WAIT TIMEOUT (PREP_BO on the
 *     output never completes -> precision value rejected on this path; sweep next)
 *     vs "output still 0xAA" (ran but wrote nothing) vs "touched" (cleared stage 1).
 *
 *   STAGE 2 — are the VALUES right (within bf16 tolerance)? classify each element:
 *       correct(|rel|<=tol) / nonfinite(inf|nan where ref finite -> range fail) /
 *       sentinel(0xAAAAAAAA, unwritten) / wrong(written, finite, out of tol).
 *     A clean PASS = all M*N correct within tol; the printed max_rel reveals the
 *     ACCUMULATION REGIME (≈1e-5 => exact products + fp32 accumulate; ≈0.4% =>
 *     product or accumulate rounded to bf16). Either way bf16 matmul is real.
 *
 * Sweep knobs (passed through to gen_matmul_bf16 via env; this harness only prints
 * them + sizes the output BO):
 *   ROCKET_BF16_PREC      in/proc precision (default 3 = bf16; try 7 = tf32 control)
 *   ROCKET_BF16_OUT_PREC  output precision  (default 5 = fp32)
 *   ROCKET_BF16_SIZE_E / ROCKET_BF16_SURF_MULT   fp32-out stride (default 3 / 4)
 * Harness-local knobs:
 *   ROCKET_BF16_C2        feature input cube (default 8 = fp16's 2-byte atom)
 *   ROCKET_BF16_OUT_C2    fp32 output cube   (default 4 = 4-byte fp32 cube, HW-known)
 *   ROCKET_BF16_TOL       relative tolerance (default 0.03; raise for large K)
 *   ROCKET_BF16_BIG       1 => drive |values| past fp16's 65504 max (range proof)
 *   ROCKET_BF16_BIG_MAG   magnitude for BIG mode (default 1e5, > fp16 max)
 *   ROCKET_BF16_MAG       magnitude for normal mode (default 10, Gemma-ish)
 *
 * bf16 is represented as uint16_t = (fp32_bits >> 16) (TRUNCATE — no compiler
 * __bf16 dependency). The reference reads each operand back as bf16->fp32 so the
 * host matmul sees EXACTLY what the HW sees; the only difference is summation order.
 *
 * Usage: matmul_bf16_rocket <M> <K> <N>   (M%4||1, K%32, N%16; try 4 32 64)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
static int env_int(const char *name, int def) {
    const char *e = getenv(name); return e ? (int)strtol(e, NULL, 0) : def;
}
static double env_dbl(const char *name, double def) {
    const char *e = getenv(name); return e ? strtod(e, NULL) : def;
}

/* bf16 <-> fp32 by truncation (bf16 = high 16 bits of the fp32 bit pattern). */
static inline uint16_t fp32_to_bf16(float f) {
    uint32_t b; memcpy(&b, &f, sizeof b); return (uint16_t)(b >> 16);
}
static inline float bf16_to_fp32(uint16_t h) {
    uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, sizeof f); return f;
}

/* fp32 output read index (cube C2, default 4 = the 4-byte fp32 cube). 1-based n,m
 * to match the pack loops, like matmul_fp16.c / matmul_int16_rocket.c. */
static size_t out_index(int N, int M, int out_c2, int n, int m) {
    return (size_t)feature_data(N, M, 1, out_c2, n, m, 1);
}

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s <M> <K> <N>  (try 4 32 64)\n", argv[0]); return -1; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    if (M <= 0 || (M % 4 != 0 && M != 1)) { fprintf(stderr, "M must be %%4 or 1\n"); return -1; }
    if (K <= 0 || K % 32 != 0) { fprintf(stderr, "K must be %%32\n"); return -1; }
    if (N <= 0 || N % 16 != 0) { fprintf(stderr, "N must be %%16 (fp16/bf16 N-group)\n"); return -1; }

    const int   c2     = env_int("ROCKET_BF16_C2", 8);       /* feature input cube  */
    const int   out_c2 = env_int("ROCKET_BF16_OUT_C2", 4);   /* fp32 output cube    */
    const int   big    = env_int("ROCKET_BF16_BIG", 0);      /* range-proof mode    */
    const float tol    = (float)env_dbl("ROCKET_BF16_TOL", 0.03);
    float mag          = (float)env_dbl("ROCKET_BF16_MAG", 10.0);
    if (big) mag       = (float)env_dbl("ROCKET_BF16_BIG_MAG", 1e5);  /* > fp16 max 65504 */

    printf("bf16 TEST%s: C[%d,%d] = A[%d,%d] x B[%d,%d]^T (bf16xbf16->fp32)\n",
           big ? " [BIG/range>fp16]" : "", M, N, M, K, N, K);
    printf("knobs: PREC=%d OUT_PREC=%d SIZE_E=%d SURF_MULT=%d | C2=%d OUT_C2=%d tol=%g mag=%g%s\n",
           env_int("ROCKET_BF16_PREC", 3), env_int("ROCKET_BF16_OUT_PREC", 5),
           env_int("ROCKET_BF16_SIZE_E", 3), env_int("ROCKET_BF16_SURF_MULT", 4),
           c2, out_c2, tol, mag,
           big ? "  (|values| EXCEED fp16's 65504 max -> would be inf in fp16)" : "");

    int fd = rocket_open();
    if (fd < 0) return fd;

    rocket_bo guard = {0}, regcmd = {0}, input = {0}, weights = {0}, output = {0};
    int ret = 0;
    size_t in_bytes  = (size_t)M * K * sizeof(uint16_t);
    size_t wt_bytes  = (size_t)N * K * sizeof(uint16_t);
    /* size the fp32 output BO for the padded cube (align M->4, N->32), generous. */
    size_t out_bytes = (size_t)((M + 3) / 4 * 4) * ((N + 31) / 32 * 32) * sizeof(float) + 4096;

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
    if ((ret = gen_matmul_bf16(&p)) != 0) {
        fprintf(stderr, "gen_matmul_bf16 = %d (CBUF banks? -1 fd, -2 weight/kernel)\n", ret);
        goto out;
    }
    printf("regcmd ops = %u\n", p.task_count);

    /* host operands: generate fp32, truncate to bf16, keep the bf16->fp32 value as
     * the reference input so the host sees EXACTLY the HW's operands. */
    uint16_t *Ab = malloc((size_t)M * K * sizeof(uint16_t));   /* bf16 bits (packed) */
    uint16_t *Bb = malloc((size_t)N * K * sizeof(uint16_t));
    float    *Ar = malloc((size_t)M * K * sizeof(float));      /* bf16->fp32 ref */
    float    *Br = malloc((size_t)N * K * sizeof(float));
    float    *C  = malloc((size_t)M * N * sizeof(float));      /* fp32 reference */
    if (!Ab || !Bb || !Ar || !Br || !C) { fprintf(stderr, "host malloc failed\n"); ret = -1; goto out_free; }
    srand(1234);
    for (int i = 0; i < M*K; i++) {
        float f = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * mag;
        Ab[i] = fp32_to_bf16(f); Ar[i] = bf16_to_fp32(Ab[i]);
    }
    for (int i = 0; i < N*K; i++) {
        float f = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * mag;
        Bb[i] = fp32_to_bf16(f); Br[i] = bf16_to_fp32(Bb[i]);
    }
    /* fp32 reference dot products over the bf16-truncated operands. */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int l = 0; l < K; l++) s += Ar[i*K + l] * Br[j*K + l];
            C[i*N + j] = s;
        }

    CK(rocket_bo_prep(fd, &regcmd, 1, 0));
    CK(rocket_bo_prep(fd, &input,  1, 0));
    CK(rocket_bo_prep(fd, &weights,1, 0));
    memcpy(regcmd.ptr, npu_regs, (size_t)p.task_count * sizeof(uint64_t));
    memset(input.ptr, 0, input.size);
    memset(weights.ptr, 0, weights.size);
    /* pack bf16 bits into the NPU native layouts: weight_fp16 (bf16 == fp16 2-byte
     * layout), feature cube C2 (default 8 = fp16's 2-byte atom). Raw 2-byte stores. */
    uint16_t *wdst = weights.ptr, *idst = input.ptr;
    for (int n = 1; n <= N; n++)
        for (int k = 1; k <= K; k++)
            wdst[weight_fp16(K, n, k)] = Bb[(n-1)*K + (k-1)];
    for (int m = 1; m <= M; m++)
        for (int k = 1; k <= K; k++)
            idst[feature_data(K, M, 1, c2, k, m, 1)] = Ab[(m-1)*K + (k-1)];
    CK(rocket_bo_fini(fd, &regcmd));
    CK(rocket_bo_fini(fd, &input));
    CK(rocket_bo_fini(fd, &weights));

    CK(rocket_bo_prep(fd, &output, 1, 0));
    memset(output.ptr, 0xAA, output.size);     /* sentinel 0xAAAAAAAA (a tiny -ve fp32) */
    CK(rocket_bo_fini(fd, &output));

    uint32_t in_h[]  = { input.handle, weights.handle, regcmd.handle };
    uint32_t out_h[] = { output.handle };
    int64_t t0 = now_us();
    ret = rocket_submit_matmul(fd, &regcmd, p.task_count, in_h, 3, out_h, 1, 6000);
    printf("ROCKET_SUBMIT = %d  (%.3f ms)\n", ret, (now_us() - t0) / 1000.0);
    if (ret) goto out_free;

    int prc = rocket_bo_prep(fd, &output, 0, 2000000000LL);
    if (prc) {
        fprintf(stderr, "STAGE1 FAIL: PREP_BO(output) = %d (%s) — job never completed "
                "(precision value likely rejected; sweep ROCKET_BF16_PREC, default 3=bf16)\n",
                prc, strerror(-prc));
        ret = prc; goto out_free;
    }

    /* STAGE 1: touched? (sentinel 0xAA byte) */
    size_t first = out_bytes; int touched = 0;
    for (size_t i = 0; i < out_bytes; i++)
        if (((uint8_t *)output.ptr)[i] != 0xAA) { touched = 1; first = i; break; }
    if (!touched) {
        printf("STAGE1: output still 0xAA — ran but wrote NOTHING (DPU enable / out_precision). "
               "Sweep ROCKET_BF16_OUT_PREC=5, SIZE_E/SURF_MULT=3/4.\n");
        rocket_bo_fini(fd, &output); ret = -1; goto out_free;
    }
    printf("STAGE1 OK: output TOUCHED (first non-0xAA byte @ %zu of %zu)\n", first, out_bytes);

    /* STAGE 2: values within bf16 tolerance. */
    const float *od = output.ptr;
    int n_correct = 0, n_wrong = 0, n_sent = 0, n_nonfin = 0, shown = 0;
    int row0_prefix = 0, prefix_open = 1, row0_correct = 0;
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int m = 1; m <= M; m++)
        for (int n = 1; n <= N; n++) {
            float act = od[out_index(N, M, out_c2, n, m)];
            float exp = C[(m-1)*N + (n-1)];
            uint32_t raw; memcpy(&raw, &act, sizeof raw);
            int ok = 0;
            if (raw == 0xAAAAAAAAu) {
                n_sent++;
            } else if (!isfinite(act)) {
                n_nonfin++;
                if (shown < 8) { printf("  nonfinite m=%d n=%d exp=%g act=%g (raw=0x%08x)\n",
                                        m, n, exp, act, raw); shown++; }
            } else {
                float ad = fabsf(act - exp);
                float rd = ad / (fabsf(exp) + 1e-6f);
                if (ad > max_abs) max_abs = ad;
                if (rd > max_rel) max_rel = rd;
                if (rd <= tol) { ok = 1; n_correct++; }
                else {
                    n_wrong++;
                    if (shown < 8) { printf("  wrong m=%d n=%d exp=%g act=%g (rel=%.4f)\n",
                                            m, n, exp, act, rd); shown++; }
                }
            }
            if (m == 1) {
                if (ok) { row0_correct++; if (prefix_open) row0_prefix++; }
                else prefix_open = 0;
            }
        }
    printf("STAGE2: correct=%d wrong=%d nonfinite=%d sentinel=%d (of %d) | "
           "max_abs=%.4g max_rel=%.4g | row0 %d/%d correct (prefix %d)\n",
           n_correct, n_wrong, n_nonfin, n_sent, M*N, max_abs, max_rel,
           row0_correct, N, row0_prefix);
    /* sample magnitudes (sanity that we're reading the right cube/scale) */
    printf("sample got vs ref: [%.3g|%.3g] [%.3g|%.3g] [%.3g|%.3g]\n",
           od[out_index(N,M,out_c2,1,1)], C[0],
           N>=2 ? od[out_index(N,M,out_c2,2,1)] : 0.0f, N>=2 ? C[1] : 0.0f,
           N>=3 ? od[out_index(N,M,out_c2,3,1)] : 0.0f, N>=3 ? C[2] : 0.0f);

    if (n_correct == M*N) {
        printf("==> PASS (bf16xbf16->fp32 within tol; max_rel=%.4g => %s)\n", max_rel,
               max_rel < 1e-3 ? "exact products + fp32 accumulate" :
               max_rel < 2e-2 ? "bf16-rounded products/accumulate (expected)" :
               "loose but tracking — inspect regime");
        if (big) printf("==> RANGE PROVEN: correct at |values|>fp16-max ⇒ bf16 can drop activation scaling\n");
        ret = 0;
    } else if (n_nonfin > 0 && n_sent == 0 && n_wrong == 0) {
        printf("==> NONFINITE: wrote inf/nan where ref is finite — precision/accum NOT bf16 "
               "(overflow like fp16 would). Sweep ROCKET_BF16_PREC; this is the int16-style fail.\n");
        ret = -1;
    } else {
        printf("==> values wrong (stage1 cleared). Crack geometry: ROCKET_BF16_OUT_C2 (try 8), "
               "ROCKET_BF16_C2 (try 8/16), then ROCKET_BF16_PREC. A nonzero row0-prefix that "
               "stops mid-row usually means the OUTPUT CUBE (OUT_C2) is wrong.\n");
        ret = -1;
    }
    rocket_bo_fini(fd, &output);

out_free:
    free(Ab); free(Bb); free(Ar); free(Br); free(C);
out:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &input);
    rocket_bo_free(fd, &weights); rocket_bo_free(fd, &output);
    rocket_close(fd);
    return ret;
}
