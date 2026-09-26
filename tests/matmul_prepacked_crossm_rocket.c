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
 * THE CHECK IS EXACT AND COVERS EVERY ROW. The operands are hashed integers (A in [-3,3],
 * B in [-2,2], tests/test_fill.h), so every sum is an exact fp16 integer and the result
 * must equal the integer reference bit for bit; the gate asserts max|ref| < 2048. The old
 * fills had periods 13 and 11 in the flattened index, which at K=768 left C only 143
 * distinct values, and the check probed ~9 rows under a bar of 0.5 absolute AND 5%
 * relative. The output starts as an fp16 NaN sentinel.
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
#include "test_fill.h"

/* Every row, exactly: the integer reference in fp16, then an element-by-element compare
 * that names the first mismatch. */
static int verify_all(const _Float16 *C, const _Float16 *A, const _Float16 *B,
                      int M, int K, int N)
{
    _Float16 *ref = malloc((size_t)M * N * sizeof(_Float16));
    if (!ref) { fprintf(stderr, "oom\n"); return 1; }
    double peak = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double a = 0;
            for (int k = 0; k < K; k++) a += (double)A[(size_t)m*K+k] * (double)B[(size_t)n*K+k];
            if (fabs(a) > peak) peak = fabs(a);
            ref[(size_t)m*N+n] = (_Float16)a;
        }
    const int dims[2] = { M, N };
    long bad = tf_cmp_f16("    resident", C, ref, dims, 2);
    int inexact = !(peak < 2048.0);
    printf("    %d rows, %ld of %zu elements differ, max|ref| %.0f%s -> %s\n", M, bad,
           (size_t)M * N, peak, inexact ? " (not below 2048: fp16 is not exact, the fill range is wrong)" : "",
           (bad || inexact) ? "FAIL" : "PASS");
    free(ref);
    return (bad || inexact) ? 1 : 0;
}

int main(int argc, char **argv)
{
    int K = argc > 1 ? atoi(argv[1]) : 768;   /* large enough that Kt depends on Mt */
    int N = argc > 2 ? atoi(argv[2]) : 256;
    int T = argc > 3 ? atoi(argv[3]) : 2;
    const int Mpack = 512;
    const int Mrun[] = { 256, 512, 768 };       /* all >= MAX_TILE -> same tiling as Mpack */
    const int Mmax = 768;

    if (K % 32 || N % 16) { fprintf(stderr, "need K%%32==0, N%%16==0\n"); return 1; }

    {   /* a missing device is the only skip: probe it before the context can fail */
        int fd = rocket_open();
        if (fd < 0) { fprintf(stderr, "no NPU (%d) -> SKIP\n", fd); return 2; }
        rocket_close(fd);
    }
    rocket_ctx *ctx = rocket_ctx_create(T);
    if (!ctx) { fprintf(stderr, "rocket_ctx_create failed\n"); return 1; }

    _Float16 *A = malloc((size_t)Mmax*K*sizeof(_Float16));
    _Float16 *B = malloc((size_t)N*K*sizeof(_Float16));
    _Float16 *C = malloc((size_t)Mmax*N*sizeof(_Float16));
    if (!A || !B || !C) { fprintf(stderr, "oom\n"); return 1; }
    tf_fill_f16_int(A, (size_t)Mmax*K, 0xC0055A, -3, 3);
    tf_fill_f16_int(B, (size_t)N*K,    0xC0055B, -2, 2);

    printf("cross-M reuse: pack@M=%d, run@M={256,512,768}, K=%d N=%d T=%d\n", Mpack, K, N, T);
    rocket_weights *w = rocket_weights_pack(ctx, Mpack, K, N, B);
    if (!w) { fprintf(stderr, "weights_pack@%d failed\n", Mpack); return 1; }

    int fail = 0;
    for (int i = 0; i < 3; i++) {
        int M = Mrun[i];
        tf_sentinel_f16(C, (size_t)M*N);
        int r = rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, w);   /* SAME w, different M */
        if (r) { printf("  M=%d: prepacked ret=%d -> FAIL\n", M, r); fail = 1; continue; }
        printf("  M=%d (reusing the M=%d weight):\n", M, Mpack);
        fail |= verify_all(C, A, B, M, K, N);
    }

    /* The invariant: prepacked must NEVER silently miscompute at a different M. A small
     * M (< MAX_TILE) usually plans a larger Kt than the M>=256 pack, so the layout can't
     * be reused -> reject (-2). But for small K, Kt==K regardless of Mt and M=4 stays
     * compatible -> then it must compute CORRECTLY. Accept either, fail a wrong answer. */
    tf_sentinel_f16(C, (size_t)4*N);
    int r4 = rocket_matmul_fp16_prepacked(ctx, 4, K, N, A, C, w);
    if (r4 == -2) {
        printf("  M=4: ret=-2 -> PASS (incompatible tiling correctly rejected)\n");
    } else if (r4 == 0) {
        printf("  M=4: ret=0 (tiling compatible) — must be correct:\n");
        fail |= verify_all(C, A, B, 4, K, N);
    } else {
        printf("  M=4: ret=%d -> FAIL\n", r4); fail = 1;
    }

    rocket_weights_free(ctx, w);
    rocket_ctx_free(ctx);
    free(A); free(B); free(C);
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
