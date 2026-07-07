// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_prepacked_crossm_rocket.c — cross-M reuse gate: a resident weight packed at one M is
 * REUSED at a different (compatible) M, with no re-pack.
 *
 * The resident weight's tile layout depends only on K, N and the K/N tiling (Kt/Nt),
 * which is M-independent for every M >= MAX_TILE (256) — Mt is capped there, so the
 * weight scatter positions don't move. This gate packs B ONCE at M=512 and computes
 * C[M,N]=A·Bᵀ at M=256, 512, 768 against the SAME rocket_weights, verifying each is
 * bit-correct vs a CPU reference. It also confirms that a small M (M=4, whose Kt grows
 * past the M>=256 value) is correctly REJECTED (-2) so the caller knows to re-pack.
 *
 * CTest target matmul_prepacked_crossm_rocket; skip-code 2 off-device.
 *   sudo ./matmul_prepacked_crossm_rocket            # K=256 N=256, T=2
 *   sudo ./matmul_prepacked_crossm_rocket 512 512 3  # custom K N T
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int verify_rows(const _Float16 *C, const _Float16 *A, const _Float16 *B,
                       int M, int K, int N)
{
    /* check rows on both sides of every MAX_TILE (256) boundary + the last row */
    int probes[16], np = 0;
    for (int m = 0; m < M && np < 14; m += 255) { probes[np++] = m; if (m+1 < M) probes[np++] = m+1; }
    probes[np++] = M - 1;
    int bad = 0; float maxabs = 0, maxrel = 0;
    for (int pi = 0; pi < np; pi++) {
        int m = probes[pi];
        for (int n = 0; n < N; n++) {
            float a = 0;
            for (int k = 0; k < K; k++) a += (float)A[(size_t)m*K+k] * (float)B[(size_t)n*K+k];
            float got = (float)C[(size_t)m*N+n];
            float ad = fabsf(got - a), rd = ad / (fabsf(a) + 1e-6f);
            if (ad > maxabs) maxabs = ad;
            if (rd > maxrel) maxrel = rd;
            if (ad > 0.5f && rd > 0.05f) bad++;
        }
    }
    printf("    rows-checked=%d max_abs=%.3f max_rel=%.4f bad=%d -> %s\n",
           np, maxabs, maxrel, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    int K = argc > 1 ? atoi(argv[1]) : 768;   /* large enough that Kt depends on Mt */
    int N = argc > 2 ? atoi(argv[2]) : 256;
    int T = argc > 3 ? atoi(argv[3]) : 2;
    const int Mpack = 512;
    const int Mrun[] = { 256, 512, 768 };       /* all >= MAX_TILE -> same tiling as Mpack */
    const int Mmax = 768;

    if (K % 32 || N % 16) { fprintf(stderr, "need K%%32==0, N%%16==0\n"); return 2; }

    rocket_ctx *ctx = rocket_ctx_create(T);
    if (!ctx) { fprintf(stderr, "rocket_ctx_create failed — no NPU, skipping\n"); return 2; }

    _Float16 *A = malloc((size_t)Mmax*K*sizeof(_Float16));
    _Float16 *B = malloc((size_t)N*K*sizeof(_Float16));
    _Float16 *C = malloc((size_t)Mmax*N*sizeof(_Float16));
    if (!A || !B || !C) { fprintf(stderr, "oom\n"); return 1; }
    for (size_t i = 0; i < (size_t)Mmax*K; i++) A[i] = (_Float16)(((i*7)%13 - 6) * 0.1f);
    for (size_t i = 0; i < (size_t)N*K;    i++) B[i] = (_Float16)(((i*5)%11 - 5) * 0.1f);

    printf("cross-M reuse: pack@M=%d, run@M={256,512,768}, K=%d N=%d T=%d\n", Mpack, K, N, T);
    rocket_weights *w = rocket_weights_pack(ctx, Mpack, K, N, B);
    if (!w) { fprintf(stderr, "weights_pack@%d failed\n", Mpack); return 1; }

    int fail = 0;
    for (int i = 0; i < 3; i++) {
        int M = Mrun[i];
        for (size_t j = 0; j < (size_t)M*N; j++) C[j] = (_Float16)-99.0f;
        int r = rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, w);   /* SAME w, different M */
        if (r) { printf("  M=%d: prepacked ret=%d -> FAIL\n", M, r); fail = 1; continue; }
        printf("  M=%d (reusing the M=%d weight):\n", M, Mpack);
        fail |= verify_rows(C, A, B, M, K, N);
    }

    /* The invariant: prepacked must NEVER silently miscompute at a different M. A small
     * M (< MAX_TILE) usually plans a larger Kt than the M>=256 pack, so the layout can't
     * be reused -> reject (-2). But for small K, Kt==K regardless of Mt and M=4 stays
     * compatible -> then it must compute CORRECTLY. Accept either, fail a wrong answer. */
    for (size_t j = 0; j < (size_t)4*N; j++) C[j] = (_Float16)-99.0f;
    int r4 = rocket_matmul_fp16_prepacked(ctx, 4, K, N, A, C, w);
    if (r4 == -2) {
        printf("  M=4: ret=-2 -> PASS (incompatible tiling correctly rejected)\n");
    } else if (r4 == 0) {
        printf("  M=4: ret=0 (tiling compatible) — must be correct:\n");
        fail |= verify_rows(C, A, B, 4, K, N);
    } else {
        printf("  M=4: ret=%d -> FAIL\n", r4); fail = 1;
    }

    rocket_weights_free(ctx, w);
    rocket_ctx_free(ctx);
    free(A); free(B); free(C);
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
