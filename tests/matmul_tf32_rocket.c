// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_tf32_rocket.c — standalone tf32 x tf32 -> fp32 FEASIBILITY
 * test + 4-byte-input geometry sweep + PRECISION CHARACTERIZATION on the rocket
 * driver. The tf32 analogue of matmul_bf16_rocket, but tf32 is the FIRST 4-byte-
 * INPUT matmul on this stack — so the input cube / weight layout / data-entry
 * divisor are from-scratch, and this harness sweeps them.
 *
 * WHAT tf32 IS (NVIDIA-TF32 convention): 1 sign + 8-bit exponent (fp32 RANGE) +
 * 10-bit mantissa (fp16 PRECISION) = 19 used bits, stored in a 4-BYTE fp32
 * container (19 bits can't fit 2 bytes). You feed fp32; the MAC rounds the mantissa
 * to 10 bits, then accumulates in fp32. So we feed RAW fp32 operands and let the HW
 * round — that is exactly what the characterization measures.
 *
 * FEASIBILITY OUTLOOK (Step-0 recon): encoding STRONG, input geometry UNKNOWN. The
 * precision-field tables name "3'd7: tf32" in the CNA CONV_CON1 in/proc precision
 * table, and tf32 runs at 256x3 MAC operations per cycle (HALF fp16/bf16's 512x3). tf32
 * accumulates to FP32, so the OUTPUT reuses the fp16/bf16 PROVEN fp32-out writer
 * (out_precision=5, size_e=3, surf*4, output cube C2=4) — that risk is low. The REAL
 * unknown is the 4-byte INPUT cube + weight layout + data-entry divisor (the int16
 * lesson: a dtype can be a listed MAC op yet have no usable matmul path).
 *
 *   STAGE 1 — does a tf32 program RUN and WRITE fp32 output? (precision value +
 *     output geometry, LAYOUT-INDEPENDENT). Watch for: WAIT TIMEOUT (PREP_BO on the
 *     output never completes -> precision value rejected on this 4-byte path; the
 *     int4 sweep saw precision=7 "wrote nothing" for NIBBLE data — tf32 needs 4-byte
 *     fp32-laid-out operands) vs "output still 0xAA" (ran, wrote nothing) vs
 *     "touched" (cleared stage 1).
 *
 *   STAGE 2 — are the VALUES right, and AT WHAT PRECISION? classify each element:
 *       correct(|rel_tf32|<=tol) / nonfinite / sentinel(0xAAAAAAAA) / wrong.
 *     Then CHARACTERIZE: compare the HW output against BOTH a tf32-rounded reference
 *     AND a full-fp32 reference (inputs are full-23-bit-mantissa random fp32, so the
 *     two refs differ by ~2^-11 ~ 5e-4 = the "ref gap", which gives the test its
 *     discriminating power). Whichever ref the HW tracks reveals the actual internal
 *     mantissa: 10-bit tf32, full fp32, or other. This is the interesting RE result
 *     (analogous to int16's saturate/wrap characterization).
 *
 * Sweep knobs read by gen_matmul_tf32 (this harness only prints them):
 *   ROCKET_TF32_PREC        in/proc precision   (default 7 = tf32; sweep if STAGE1 fails)
 *   ROCKET_TF32_OUT_PREC    output precision    (default 5 = fp32)
 *   ROCKET_TF32_SIZE_E / ROCKET_TF32_SURF_MULT  fp32-out stride (default 3 / 4)
 *   ROCKET_TF32_DENTRIES_DIV data-entry divisor (default 16 = 4-byte element)
 *   ROCKET_TF32_ELEM_BYTES  input element bytes (default 4)
 * Harness-local knobs (the from-scratch 4-byte INPUT geometry — sweep these first):
 *   ROCKET_TF32_C2          feature input cube  (default 4 = 16-byte atom / 4 bytes)
 *   ROCKET_TF32_OUT_C2      fp32 output cube     (default 4 = 4-byte fp32 cube, proven)
 *   ROCKET_TF32_WT_NGROUP   weight N-group       (default 8 = 1024-byte tile / (32*4))
 *   ROCKET_TF32_TOL         relative tolerance   (default 2e-3; tf32 ~2^-11 + K-accum)
 *   ROCKET_TF32_MAG         operand magnitude    (default 10, Gemma-ish)
 *   ROCKET_TF32_BIG / ROCKET_TF32_BIG_MAG  drive |values| past fp16's 65504 (range proof)
 *
 * Usage: matmul_tf32_rocket <M> <K> <N>   (M%4||1, K%32, N%16; try 4 32 64)
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

/* Round an fp32 to NVIDIA-tf32 precision: keep sign + 8-bit exp + top 10 mantissa
 * bits, zero the low 13 (23-bit fp32 mantissa -> 10-bit tf32 mantissa). This is
 * what the HW MAC is hypothesized to do internally before the fp32 multiply. */
static inline float round_to_tf32(float f) {
    uint32_t b; memcpy(&b, &f, sizeof b);
    b &= 0xFFFFE000u;                 /* keep bits 31..13, zero bits 12..0 */
    float r; memcpy(&r, &b, sizeof r); return r;
}

/* tf32 weight element index, N-group (ng) AND K-group (kg) parameterized. Mirrors
 * weight_fp16's block shape with both group sizes swept for the 4-byte HW tile RE
 * (ng=8,kg=32 == weight_tf32() in npu_regcmd.c). kg is now sweepable too:
 * tf32's weight K-group could be 16 (not 32) for 4-byte, and this only becomes
 * testable at K>kg (K=32 is a single K-group => row-major for ANY ng/kg, which is
 * why the K=32 case could not test the weight layout at all). */
static int wt_tf32_idx(int C, int k, int c, int ng, int kg) {
    int kpg = (k-1)/ng;
    int cpg = (c-1)/kg;
    int dst = (cpg*kg)*ng + kpg*ng*C;
    dst += ((c-1)%kg) + (((k-1)%ng)*kg);
    return dst;
}

/* fp32 (de)cube index (element index; cube C2). 1-based n,m to match pack loops. */
static size_t cube_index(int N, int M, int c2, int n, int m) {
    return (size_t)feature_data(N, M, 1, c2, n, m, 1);
}

static int rup(int x, int a) { return (x + a - 1) / a * a; }

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s <M> <K> <N>  (try 4 32 64)\n", argv[0]); return -1; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    if (M <= 0 || (M % 4 != 0 && M != 1)) { fprintf(stderr, "M must be %%4 or 1\n"); return -1; }
    if (K <= 0 || K % 32 != 0) { fprintf(stderr, "K must be %%32\n"); return -1; }
    if (N <= 0 || N % 16 != 0) { fprintf(stderr, "N must be %%16 (fp32-out writer N-group)\n"); return -1; }

    const int   c2     = env_int("ROCKET_TF32_C2", 4);        /* feature input cube (16B/4) */
    const int   out_c2 = env_int("ROCKET_TF32_OUT_C2", 4);    /* fp32 output cube           */
    const int   wt_ng  = env_int("ROCKET_TF32_WT_NGROUP", 16);/* weight N-group (HW: 16, == fp16) */
    const int   wt_kg  = env_int("ROCKET_TF32_WT_KGROUP", 16);/* weight K-group (HW: 16, HALVED vs fp16's 32) */
    const int   big    = env_int("ROCKET_TF32_BIG", 0);       /* range-proof mode           */
    const float tol    = (float)env_dbl("ROCKET_TF32_TOL", 2e-3);
    float mag          = (float)env_dbl("ROCKET_TF32_MAG", 10.0);
    if (big) mag       = (float)env_dbl("ROCKET_TF32_BIG_MAG", 1e5);  /* > fp16 max 65504 */
    const int ones = env_int("ROCKET_TF32_ONES", 0);  /* shorthand for ROCKET_TF32_PAT=1 */
    const int pat  = env_int("ROCKET_TF32_PAT", ones ? 1 : 0);  /* operand pattern (see fill below) */
    const int kimp = env_int("ROCKET_TF32_KIMP", 1);  /* impulse K-index (1-based) for pat 5/6 */
    const int dump = env_int("ROCKET_TF32_DUMP", 0);  /* dump raw fp32 output buffer (layout probe) */
    static const char *patname[] = {"random", "ones(C=K)", "kramp(C=Sk^2)", "ncol(C=K*n)",
                                    "mrow(C=K*m)", "aimp(A delta@kimp)", "bimp(B delta@kimp)"};

    printf("tf32 TEST%s [pat=%s]: C[%d,%d] = A[%d,%d] x B[%d,%d]^T (tf32xtf32->fp32, 4-byte input)\n",
           big ? " [BIG/range>fp16]" : "", (pat >= 0 && pat <= 6) ? patname[pat] : "?",
           M, N, M, K, N, K);
    if (pat == 5 || pat == 6) printf("  (impulse kimp=%d -> correct C should = %d everywhere; "
                                     "deviation reveals the K-lane permutation)\n", kimp, kimp);
    printf("knobs: PREC=%d CORE_PREC=%d DPU_PREC=%d OUT_PREC=%d SIZE_E=%d SURF_MULT=%d "
           "DENTRIES_DIV=%d ELEM_BYTES=%d | C2=%d OUT_C2=%d WT_NGROUP=%d WT_KGROUP=%d tol=%g mag=%g%s\n",
           env_int("ROCKET_TF32_PREC", 7), env_int("ROCKET_TF32_CORE_PREC", 7),
           env_int("ROCKET_TF32_DPU_PREC", 5), env_int("ROCKET_TF32_OUT_PREC", 5),
           env_int("ROCKET_TF32_SIZE_E", 3), env_int("ROCKET_TF32_SURF_MULT", 4),
           env_int("ROCKET_TF32_DENTRIES_DIV", 16), env_int("ROCKET_TF32_ELEM_BYTES", 4),
           c2, out_c2, wt_ng, wt_kg, tol, mag,
           big ? "  (|values| EXCEED fp16's 65504 max)" : "");

    int fd = rocket_open();
    if (fd < 0) return fd;

    rocket_bo guard = {0}, regcmd = {0}, input = {0}, weights = {0}, output = {0};
    int ret = 0;
    /* 4-byte fp32 operands; size generously for the padded cube/weight tile so a
     * geometry sweep (bigger C2 / N-group) can't write out of bounds. */
    size_t in_bytes  = (size_t)rup(M, 4)  * rup(K, 32) * sizeof(float) + 4096;
    size_t wt_bytes  = (size_t)rup(N, 64) * rup(K, 32) * sizeof(float) + 4096;
    size_t out_bytes = (size_t)rup(M, 4)  * rup(N, 32) * sizeof(float) + 4096;

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
    if ((ret = gen_matmul_tf32(&p)) != 0) {
        fprintf(stderr, "gen_matmul_tf32 = %d (CBUF banks? -1 fd, -2 weight/kernel)\n", ret);
        goto out;
    }
    printf("regcmd ops = %u\n", p.task_count);

    /* host operands: full-23-bit-mantissa random fp32 (fed RAW to the HW — the MAC
     * is what rounds to tf32). Two references: (a) inputs ROUNDED to tf32 then fp32
     * matmul; (b) full-fp32 inputs, fp32 matmul. They differ by ~2^-11 (the "ref
     * gap"), which lets STAGE2 tell whether the HW really rounds to 10-bit tf32. */
    float *A   = malloc((size_t)M * K * sizeof(float));   /* raw fp32 fed to HW */
    float *B   = malloc((size_t)N * K * sizeof(float));
    float *Atf = malloc((size_t)M * K * sizeof(float));   /* tf32-rounded copy   */
    float *Btf = malloc((size_t)N * K * sizeof(float));
    float *Ctf = malloc((size_t)M * N * sizeof(float));   /* tf32-rounded ref    */
    float *Cf  = malloc((size_t)M * N * sizeof(float));   /* full-fp32 ref       */
    if (!A || !B || !Atf || !Btf || !Ctf || !Cf) { fprintf(stderr, "host malloc failed\n"); ret = -1; goto out_free; }
    /* Operand patterns (ROCKET_TF32_PAT; ONES=1 is the shorthand for pat 1). Each
     * makes the correct C predictable so the raw dump is interpretable, and each
     * isolates one layout axis:
     *   0 random (default)  — precision/characterization once geometry is solved
     *   1 ones   A=B=1       -> C=K everywhere   (reduction COUNT + which cells written)
     *   2 kramp  A=B=k       -> C=sum k^2 (uniform) (A/B K-ALIGNMENT: deviation => K mispair)
     *   3 ncol   A=1,B=n     -> C=K*n  (varies by n) (N -> output COLUMN mapping)
     *   4 mrow   A=m,B=1     -> C=K*m  (varies by m) (M -> output ROW mapping)
     *   5 aimp   A=delta@kimp,B=k+1 -> C=kimp (every cell) (A K-lane read: got v => A
     *            slot kimp paired with B value v => the K permutation A side)
     *   6 bimp   A=k+1,B=delta@kimp -> C=kimp (every cell) (B K-lane read, the other side)
     * The reference loop below computes the true C from A,B for any pattern. */
    srand(1234);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float v = (pat == 1) ? 1.0f : (pat == 2) ? (float)(k+1) :
                      (pat == 3) ? 1.0f : (pat == 4) ? (float)(m+1) :
                      (pat == 5) ? ((k+1 == kimp) ? 1.0f : 0.0f) : (pat == 6) ? (float)(k+1) :
                      ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * mag;
            A[m*K + k] = v; Atf[m*K + k] = round_to_tf32(v);
        }
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) {
            float v = (pat == 1) ? 1.0f : (pat == 2) ? (float)(k+1) :
                      (pat == 3) ? (float)(n+1) : (pat == 4) ? 1.0f :
                      (pat == 5) ? (float)(k+1) : (pat == 6) ? ((k+1 == kimp) ? 1.0f : 0.0f) :
                      ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * mag;
            B[n*K + k] = v; Btf[n*K + k] = round_to_tf32(v);
        }
    /* two fp32 references (double accumulate so the host sum order isn't the error).
     * Errors are normalized to the GLOBAL output SCALE = max|Cf|, NOT per-element
     * |ref|: random dot products that nearly cancel make per-element relative error
     * explode (a measurement artifact, not precision loss). Scale-normalized, the
     * matching reference sits at ~1e-6 (accumulation order) and the other at the
     * tf32-rounding gap (~2^-11*sqrt(K)) — a sharp characterization. */
    float scale = 0.0f;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double stf = 0.0, sf = 0.0;
            for (int l = 0; l < K; l++) {
                stf += (double)Atf[i*K + l] * (double)Btf[j*K + l];
                sf  += (double)A[i*K + l]  * (double)B[j*K + l];
            }
            Ctf[i*N + j] = (float)stf;
            Cf[i*N + j]  = (float)sf;
            if (fabsf(Cf[i*N + j]) > scale) scale = fabsf(Cf[i*N + j]);
        }
    if (scale <= 0.0f) scale = 1.0f;
    float ref_gap = 0.0f;
    for (int i = 0; i < M*N; i++) {
        float g = fabsf(Ctf[i] - Cf[i]) / scale;
        if (g > ref_gap) ref_gap = g;
    }
    printf("output scale (max|Cf|) = %.4g | ref gap (tf32-rounded vs full-fp32, /scale) = %.4g  "
           "(test characterizes iff this >> ~1e-6 noise; ~%.1g expected for K=%d)\n",
           scale, ref_gap, 5e-4 * sqrt((double)K), K);

    CK(rocket_bo_prep(fd, &regcmd, 1, 0));
    CK(rocket_bo_prep(fd, &input,  1, 0));
    CK(rocket_bo_prep(fd, &weights,1, 0));
    memcpy(regcmd.ptr, npu_regs, (size_t)p.task_count * sizeof(uint64_t));
    memset(input.ptr, 0, input.size);
    memset(weights.ptr, 0, weights.size);
    /* pack RAW fp32 into the NPU native 4-byte layouts: weight tile (N-group swept),
     * feature cube (C2 swept). Whole-element 4-byte stores. */
    {
        float *wdst = (float *)weights.ptr, *idst = (float *)input.ptr;
        for (int n = 1; n <= N; n++)
            for (int k = 1; k <= K; k++)
                wdst[wt_tf32_idx(K, n, k, wt_ng, wt_kg)] = B[(n-1)*K + (k-1)];
        for (int m = 1; m <= M; m++)
            for (int k = 1; k <= K; k++)
                idst[cube_index(K, M, c2, k, m)] = A[(m-1)*K + (k-1)];
    }
    CK(rocket_bo_fini(fd, &regcmd));
    CK(rocket_bo_fini(fd, &input));
    CK(rocket_bo_fini(fd, &weights));

    CK(rocket_bo_prep(fd, &output, 1, 0));
    memset(output.ptr, 0xAA, output.size);     /* sentinel 0xAAAAAAAA */
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
                "(precision/4-byte path likely rejected; sweep ROCKET_TF32_PREC default 7, "
                "and ROCKET_TF32_DENTRIES_DIV/ELEM_BYTES). rmmod/insmod the driver before retry.\n",
                prc, strerror(-prc));
        ret = prc; goto out_free;
    }

    /* STAGE 1: touched? (sentinel 0xAA byte) */
    size_t first = out_bytes; int touched = 0;
    for (size_t i = 0; i < out_bytes; i++)
        if (((uint8_t *)output.ptr)[i] != 0xAA) { touched = 1; first = i; break; }
    if (!touched) {
        printf("STAGE1: output still 0xAA — ran but wrote NOTHING (DPU enable / out_precision). "
               "Sweep ROCKET_TF32_OUT_PREC=5, SIZE_E/SURF_MULT=3/4.\n");
        rocket_bo_fini(fd, &output); ret = -1; goto out_free;
    }
    printf("STAGE1 OK: output TOUCHED (first non-0xAA byte @ %zu of %zu)\n", first, out_bytes);

    /* RAW DUMP (ROCKET_TF32_DUMP=1): print the output BO as fp32 in linear memory
     * order — independent of any cube interpretation. With ONES (expect C=K), written
     * cells read K and unwritten cells read the 0xAA sentinel (~-3.03e-13), so the
     * stride/positions of K reveal the true output cube directly. */
    if (dump) {
        const float *raw = output.ptr;
        size_t nfloats = out_bytes / sizeof(float);
        /* DUMP>1 dumps that many floats (probe beyond the cube region for stray
         * channels); DUMP=1 dumps just the expected cube region. */
        size_t cap = (dump > 1) ? (size_t)dump : (size_t)rup(M, 4) * rup(N, 32);
        if (nfloats > cap) nfloats = cap;
        if (nfloats > 640) nfloats = 640;
        printf("RAW OUTPUT (fp32, linear; ONES => K=%d=written, ~-3e-13=sentinel):\n", K);
        for (size_t i = 0; i < nfloats; i += 8) {
            printf("  [%3zu]", i);
            for (size_t j = i; j < i + 8 && j < nfloats; j++) printf(" %11.4g", raw[j]);
            printf("\n");
        }
        /* LOGICAL row m=1 read via the OUT_C2 cube — exposes the N permutation
         * directly (e.g. ncol: each entry should read 32*n; deviations show n_hw). */
        printf("LOGICAL m=1 [n:got] (cube OUT_C2=%d):", out_c2);
        for (int n = 1; n <= N && n <= 72; n++)
            printf(" %d:%.0f", n, raw[cube_index(N, M, out_c2, n, 1)]);
        printf("\n");
    }

    /* STAGE 2: values + precision characterization. */
    const float *od = output.ptr;
    int n_correct = 0, n_wrong = 0, n_sent = 0, n_nonfin = 0, shown = 0;
    int row0_prefix = 0, prefix_open = 1, row0_correct = 0;
    float max_abs = 0.0f, max_rel_tf = 0.0f, max_rel_f = 0.0f;
    for (int m = 1; m <= M; m++)
        for (int n = 1; n <= N; n++) {
            float act = od[cube_index(N, M, out_c2, n, m)];
            float etf = Ctf[(m-1)*N + (n-1)];
            float ef  = Cf[(m-1)*N + (n-1)];
            uint32_t raw; memcpy(&raw, &act, sizeof raw);
            int ok = 0;
            if (raw == 0xAAAAAAAAu) {
                n_sent++;
            } else if (!isfinite(act)) {
                n_nonfin++;
                if (shown < 8) { printf("  nonfinite m=%d n=%d ref_tf32=%g act=%g (raw=0x%08x)\n",
                                        m, n, etf, act, raw); shown++; }
            } else {
                /* errors normalized to the global output scale (see ref loop). */
                float rtf = fabsf(act - etf) / scale;
                float rf  = fabsf(act - ef)  / scale;
                float ad  = fabsf(act - etf);
                if (ad > max_abs)  max_abs  = ad;
                if (rtf > max_rel_tf) max_rel_tf = rtf;
                if (rf  > max_rel_f)  max_rel_f  = rf;
                /* correct = tracks EITHER plausible reference (tf32 OR full-fp32) within
                 * tol; the characterization line below says which one the HW matches. */
                if (rtf <= tol || rf <= tol) { ok = 1; n_correct++; }
                else {
                    n_wrong++;
                    if (shown < 8) { printf("  wrong m=%d n=%d ref_tf32=%g act=%g (rel_tf32=%.4g rel_fp32=%.4g)\n",
                                            m, n, etf, act, rtf, rf); shown++; }
                }
            }
            if (m == 1) {
                if (ok) { row0_correct++; if (prefix_open) row0_prefix++; }
                else prefix_open = 0;
            }
        }
    printf("STAGE2: correct=%d wrong=%d nonfinite=%d sentinel=%d (of %d) | max_abs=%.4g | "
           "max_err/scale vs tf32-ref=%.4g vs fp32-ref=%.4g | row0 %d/%d (prefix %d)\n",
           n_correct, n_wrong, n_nonfin, n_sent, M*N, max_abs,
           max_rel_tf, max_rel_f, row0_correct, N, row0_prefix);
    printf("sample got vs tf32-ref: [%.5g|%.5g] [%.5g|%.5g] [%.5g|%.5g]\n",
           od[cube_index(N,M,out_c2,1,1)], Ctf[0],
           N>=2 ? od[cube_index(N,M,out_c2,2,1)] : 0.0f, N>=2 ? Ctf[1] : 0.0f,
           N>=3 ? od[cube_index(N,M,out_c2,3,1)] : 0.0f, N>=3 ? Ctf[2] : 0.0f);

    if (n_correct == M*N) {
        printf("==> PASS (tf32xtf32->fp32 within tol). tf32 MATMUL IS REAL.\n");
        /* precision characterization: which reference does the HW track? */
        if (ref_gap < 1e-5) {
            printf("==> PRECISION INCONCLUSIVE: ref_gap=%.3g too small to distinguish "
                   "(raise ROCKET_TF32_MAG or K so full-mantissa inputs separate the refs).\n", ref_gap);
        } else if (max_rel_f < max_rel_tf * 0.5f) {
            printf("==> PRECISION: tracks FULL-FP32 ref distinctly better (vs-fp32 %.3g << vs-tf32 %.3g) "
                   "-> the MAC keeps MORE than 10 mantissa bits (precision=7 is NOT lossy tf32 here).\n",
                   max_rel_f, max_rel_tf);
        } else if (max_rel_tf < max_rel_f * 0.5f) {
            printf("==> PRECISION: tracks the TF32 (10-bit) ref distinctly better (vs-tf32 %.3g << vs-fp32 %.3g) "
                   "-> CONFIRMS ~10-bit tf32 mantissa rounding in the MAC.\n",
                   max_rel_tf, max_rel_f);
        } else {
            printf("==> PRECISION: tracks both refs comparably (vs-tf32 %.3g, vs-fp32 %.3g, ref_gap %.3g) "
                   "-> consistent with tf32 but tighten tol / raise mag to separate cleanly.\n",
                   max_rel_tf, max_rel_f, ref_gap);
        }
        if (big) printf("==> RANGE PROVEN: correct at |values|>fp16-max ⇒ tf32 carries fp32 range.\n");
        ret = 0;
    } else if (n_nonfin > 0 && n_sent == 0 && n_wrong == 0) {
        printf("==> NONFINITE: wrote inf/nan where ref is finite — precision/accum NOT tf32. "
               "Sweep ROCKET_TF32_PREC; this is the int16-style 'listed MAC op, no usable path' fail.\n");
        ret = -1;
    } else {
        printf("==> values wrong (stage1 cleared). Crack the 4-BYTE INPUT geometry: "
               "ROCKET_TF32_WT_NGROUP (try 16/4), ROCKET_TF32_C2 (try 8/2), "
               "ROCKET_TF32_DENTRIES_DIV (try 32/8), then ROCKET_TF32_OUT_C2. A nonzero row0-prefix "
               "that stops mid-row usually means the OUTPUT CUBE (OUT_C2) is wrong; an all-wrong "
               "result usually means the WEIGHT N-group or input cube.\n");
        ret = -1;
    }
    rocket_bo_fini(fd, &output);

out_free:
    free(A); free(B); free(Atf); free(Btf); free(Ctf); free(Cf);
out:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &input);
    rocket_bo_free(fd, &weights); rocket_bo_free(fd, &output);
    rocket_close(fd);
    return ret;
}
