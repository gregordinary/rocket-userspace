// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_bf16_tiled_rocket.c — tiled bf16 x bf16 -> fp32 via
 * rocket_matmul_bf16 (M/N/K tiles + host fp32 K-accumulation). Validates the
 * TILING path the single-task test (matmul_bf16_rocket) can't reach — the big
 * Gemma shapes whose M·K·2 overflows the 384 KB CBUF.
 *
 * bf16 is the float sibling of the int16 tiled path: same 2-byte-in / 4-byte-out
 * geometry, but fp32 output, host double K-accum, NO saturation. The driver takes
 * fp32 A/B and truncates to bf16 internally; the reference truncates identically
 * (bf16 round-trip) so the only HW-vs-ref difference is summation order.
 *
 * VERIFICATION is SAMPLE-BASED: for huge K (e.g. 15360) a full M·N·K host
 * reference is ~3e10 MACs. Instead we sample up to ROCKET_BF16_SAMPLES output
 * positions and compute each one's exact double dot product on demand — negligible
 * host cost, and enough to catch any tiling/layout/accum bug. PASS = normalized
 * error (max_abs / max|ref| over the sample) < tol and no nonfinite.
 *
 * Usage: matmul_bf16_tiled_rocket <M> <K> <N>   (K%32, N%16, M%4||1)
 *   ladder: 256 384 256 | 512 3840 4096 | 512 15360 3840
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
/* bf16 round-trip (truncate) — matches rocket_matmul_bf16's internal f32_to_bf16. */
static inline float bf16rt(float f) {
    uint32_t b; memcpy(&b, &f, sizeof b); b &= 0xFFFF0000u;
    float r; memcpy(&r, &b, sizeof r); return r;
}

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s <M> <K> <N>  (try 512 3840 4096)\n", argv[0]); return -1; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    const float tol     = (float)env_dbl("ROCKET_BF16_TOL", 0.01);
    const float mag     = (float)env_dbl("ROCKET_BF16_MAG", 10.0);
    const int   samples = env_int("ROCKET_BF16_SAMPLES", 8192);

    int Mt = 0, Kt = 0, Nt = 0;
    int njobs = rocket_matmul_plan_bf16(M, K, N, &Mt, &Kt, &Nt);
    if (njobs < 0) { fprintf(stderr, "unsupported shape (need K%%32, N%%16, M%%4||1)\n"); return -1; }
    printf("bf16 TILED: C[%d,%d] = A[%d,%d] x B[%d,%d]^T  tile Mt=%d Kt=%d Nt=%d (%d jobs) tol=%g\n",
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
    int ret = rocket_matmul_bf16(fd, M, K, N, A, B, C);
    double ms = (now_us() - t0) / 1000.0;
    rocket_close(fd);
    if (ret) { fprintf(stderr, "rocket_matmul_bf16 failed (%d)\n", ret); free(A); free(B); free(C); return ret; }
    double gops = 2.0 * (double)M * N * K / (ms / 1000.0) / 1e9;
    printf("rocket_matmul_bf16 = 0  (%.2f ms, %.1f GOP/s)\n", ms, gops);

    /* sample-based verification: exact double dot product on demand. */
    size_t MN = (size_t)M * N;
    size_t stride = MN > (size_t)samples ? MN / (size_t)samples : 1;
    int checked = 0, nonfin = 0, shown = 0;
    double max_abs = 0.0, max_ref = 0.0, max_rel = 0.0;
    for (size_t idx = 0; idx < MN; idx += stride) {
        int m = (int)(idx / N), n = (int)(idx % N);
        double ref = 0.0;
        for (int k = 0; k < K; k++)
            ref += (double)bf16rt(A[(size_t)m*K + k]) * (double)bf16rt(B[(size_t)n*K + k]);
        float act = C[idx];
        if (!isfinite(act)) { nonfin++; if (shown < 8) { printf("  nonfinite m=%d n=%d ref=%g act=%g\n", m, n, ref, act); shown++; } continue; }
        double ad = fabs((double)act - ref);
        double rd = ad / (fabs(ref) + 1e-6);
        if (ad > max_abs) max_abs = ad;
        if (fabs(ref) > max_ref) max_ref = fabs(ref);
        if (rd > max_rel) max_rel = rd;
        if (ad > tol * (fabs(ref) + 1.0) && shown < 8) {
            printf("  large err m=%d n=%d ref=%g act=%g (rel=%.4g)\n", m, n, ref, act, rd); shown++;
        }
        checked++;
    }
    double norm = max_ref > 0 ? max_abs / max_ref : max_abs;
    printf("checked %d sample outputs (stride %zu) | max_abs=%.4g max|ref|=%.4g "
           "norm_err=%.4g max_rel=%.4g nonfinite=%d\n",
           checked, stride, max_abs, max_ref, norm, max_rel, nonfin);
    printf("sample got vs ref: C[0]=%.5g  C[mid]=%.5g  C[last]=%.5g\n",
           C[0], C[MN/2], C[MN-1]);

    int pass = (nonfin == 0) && (norm < tol);
    printf("==> %s (bf16 tiled %s; norm_err=%.4g)\n",
           pass ? "PASS" : "FAIL",
           pass ? "tracks fp32 reference" : "OUT OF TOLERANCE — check tiling/accum",
           norm);

    free(A); free(B); free(C);
    return pass ? 0 : -1;
}
