// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_tf32_tiled_rocket.c — tiled tf32 x tf32 -> fp32 via
 * rocket_matmul_tf32 (M/N/K tiles + host fp32 K-accumulation). Validates the
 * TILING path the single-task test (matmul_tf32_rocket) can't reach — the big
 * Gemma shapes whose M·K·4 overflows the 384 KB CBUF (tf32 is 4-byte, so it tiles
 * sooner than the 2-byte bf16/fp16 paths).
 *
 * tf32 is the FIRST 4-byte-INPUT tiled path and the LAST datatype rung. It is the
 * float sibling of the bf16 tiled path, differing only in 4-byte geometry (feature
 * cube C2=4, weight (N/16,K/16,16,16)) and that the scatter feeds RAW fp32 (no
 * truncation) — the HW MAC rounds the mantissa to 10 bits and accumulates in fp32.
 * The reference rounds each input to tf32 (zero the low 13 mantissa bits) so the
 * only HW-vs-ref difference is summation order.
 *
 * VERIFICATION is SAMPLE-BASED (like the bf16 tiled test): for huge K a full
 * M·N·K host reference is enormous, so we sample up to ROCKET_TF32_SAMPLES output
 * positions and compute each one's exact double dot product on demand. Errors are
 * normalized to the GLOBAL output SCALE (max|ref| over the sample), NOT per-element
 * |ref| — random dot products that nearly cancel make per-element relative error
 * explode (a measurement artifact, mirrors the test). PASS = normalized error
 * < tol and no nonfinite.
 *
 * Usage: matmul_tf32_tiled_rocket <M> <K> <N>   (K%16, N%16, M%4||1)
 *   ladder: 256 384 256 | 512 3840 4096 | 256 4096 4096
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

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
/* Round an fp32 to NVIDIA-tf32: keep sign + 8-bit exp + top 10 mantissa bits, zero
 * the low 13. Matches what the HW MAC does before the fp32 multiply (mirrors the
 * test's round_to_tf32). */
static inline float tf32rt(float f) {
    uint32_t b; memcpy(&b, &f, sizeof b); b &= 0xFFFFE000u;
    float r; memcpy(&r, &b, sizeof r); return r;
}

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s <M> <K> <N>  (try 512 3840 4096)\n", argv[0]); return -1; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    const float tol     = (float)env_dbl("ROCKET_TF32_TOL", 2e-3);
    const float mag     = (float)env_dbl("ROCKET_TF32_MAG", 10.0);
    const int   samples = env_int("ROCKET_TF32_SAMPLES", 8192);

    int Mt = 0, Kt = 0, Nt = 0;
    int njobs = rocket_matmul_plan_tf32(M, K, N, &Mt, &Kt, &Nt);
    if (njobs < 0) { fprintf(stderr, "unsupported shape (need K%%16, N%%16, M%%4||1)\n"); return -1; }
    printf("tf32 TILED: C[%d,%d] = A[%d,%d] x B[%d,%d]^T  tile Mt=%d Kt=%d Nt=%d (%d jobs) tol=%g\n",
           M, N, M, K, N, K, Mt, Kt, Nt, njobs, tol);

    float *A = malloc((size_t)M * K * sizeof(float));
    float *B = malloc((size_t)N * K * sizeof(float));
    float *C = malloc((size_t)M * N * sizeof(float));
    if (!A || !B || !C) { fprintf(stderr, "host malloc failed\n"); return -1; }
    srand(1234);
    for (size_t i = 0; i < (size_t)M*K; i++) A[i] = ((float)rand()/(float)RAND_MAX*2.0f - 1.0f) * mag;
    for (size_t i = 0; i < (size_t)N*K; i++) B[i] = ((float)rand()/(float)RAND_MAX*2.0f - 1.0f) * mag;

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); free(A); free(B); free(C); return 2; }

    int64_t t0 = now_us();
    int ret = rocket_matmul_tf32(fd, M, K, N, A, B, C);
    double ms = (now_us() - t0) / 1000.0;
    rocket_close(fd);
    if (ret) { fprintf(stderr, "rocket_matmul_tf32 failed (%d)\n", ret); free(A); free(B); free(C); return ret; }
    double gops = 2.0 * (double)M * N * K / (ms / 1000.0) / 1e9;
    printf("rocket_matmul_tf32 = 0  (%.2f ms, %.1f GOP/s)\n", ms, gops);

    /* sample-based verification: exact double dot product (tf32-rounded inputs) on
     * demand. Errors normalized to the global output scale (max|ref| over sample). */
    size_t MN = (size_t)M * N;
    size_t stride = MN > (size_t)samples ? MN / (size_t)samples : 1;
    int checked = 0, nonfin = 0, shown = 0;
    double max_abs = 0.0, max_ref = 0.0, max_rel = 0.0;
    for (size_t idx = 0; idx < MN; idx += stride) {
        int m = (int)(idx / N), n = (int)(idx % N);
        double ref = 0.0;
        for (int k = 0; k < K; k++)
            ref += (double)tf32rt(A[(size_t)m*K + k]) * (double)tf32rt(B[(size_t)n*K + k]);
        float act = C[idx];
        if (!isfinite(act)) { nonfin++; if (shown < 8) { printf("  nonfinite m=%d n=%d ref=%g act=%g\n", m, n, ref, act); shown++; } continue; }
        double ad = fabs((double)act - ref);
        double rd = ad / (fabs(ref) + 1e-6);
        if (ad > max_abs) max_abs = ad;
        if (fabs(ref) > max_ref) max_ref = fabs(ref);
        if (rd > max_rel) max_rel = rd;
        checked++;
    }
    double norm = max_ref > 0 ? max_abs / max_ref : max_abs;
    /* second pass: flag the worst few absolute errors against the scale (post-scale). */
    if (max_ref > 0) {
        for (size_t idx = 0; idx < MN && shown < 8; idx += stride) {
            int m = (int)(idx / N), n = (int)(idx % N);
            double ref = 0.0;
            for (int k = 0; k < K; k++)
                ref += (double)tf32rt(A[(size_t)m*K + k]) * (double)tf32rt(B[(size_t)n*K + k]);
            double ad = fabs((double)C[idx] - ref);
            if (ad > tol * max_ref) {
                printf("  large err m=%d n=%d ref=%g act=%g (abs=%.4g /scale=%.4g)\n",
                       m, n, ref, C[idx], ad, ad / max_ref); shown++;
            }
        }
    }
    printf("checked %d sample outputs (stride %zu) | max_abs=%.4g max|ref|=%.4g "
           "norm_err=%.4g max_rel=%.4g nonfinite=%d\n",
           checked, stride, max_abs, max_ref, norm, max_rel, nonfin);
    printf("sample got vs ref: C[0]=%.5g  C[mid]=%.5g  C[last]=%.5g\n",
           C[0], C[MN/2], C[MN-1]);

    int pass = (nonfin == 0) && (norm < tol);
    printf("==> %s (tf32 tiled %s; norm_err=%.4g)\n",
           pass ? "PASS" : "FAIL",
           pass ? "tracks tf32 reference" : "OUT OF TOLERANCE — check tiling/accum",
           norm);

    free(A); free(B); free(C);
    return pass ? 0 : -1;
}
