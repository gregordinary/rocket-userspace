// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int16_exact_rocket.c — the BIT-EXACT int16 path: int16 x int16 -> int64
 * matmul via int8 BYTE DECOMPOSITION on the PROVEN rocket_matmul_int8 path.
 *
 * The native int16 NPU conv (matmul_int16_rocket) tops out at a 16-bit (saturating)
 * transposed output (tp_org_en is an 8/16-bit output writer, not int32). For a
 * FULL-PRECISION int16xint16 product we instead decompose each int16 into two
 * signed bytes and run four int8 matmuls, recombining in int64. This is emulation
 * (~4x int8 cost) but bit-exact with NO saturation, and reuses only HW-validated
 * int8 kernels.
 *
 * Balanced (round-to-nearest) signed split of an int16 x:
 *     xl = ((x + 128) & 0xFF) - 128;   // low  byte, signed, in [-128,127]
 *     xh = (x - xl) >> 8;              // high byte, signed, in [-128,127]
 *     x  == xh*256 + xl   (exact; both bytes are valid signed int8)
 * This avoids the row/col-sum correction an unsigned-low-byte split needs.
 *
 * DOMAIN: two signed int8s span xh*256+xl in [-32896, 32639], so the top 128
 * int16 codes (32640..32767) are NOT representable by a signed/signed split (xh
 * would need to be 128, out of int8 range). We therefore restrict inputs to
 * [-32768, 32639] (99.6% of int16; quantized weights never sit at the extreme).
 * Full-range would require the unsigned-low-byte split + sign-correction matmuls.
 *
 *     C = 65536*(Ah.Bh) + 256*(Ah.Bl + Al.Bh) + (Al.Bl)
 * Each term is an int8xint8->int32 matmul (rocket_matmul_int8). The recombine is
 * int64, so the result is the exact int16 dot product (well beyond int32 range).
 *
 * Usage: matmul_int16_exact_rocket <M> <K> <N>   (K%32, N%32, M%4||1)
 * Alignment follows the int8 path (N%32, not int16's 16) since it IS the int8 path.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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

/* balanced signed byte split: x == hi*256 + lo, hi,lo in [-128,127] */
static inline void split_i16(int16_t x, int8_t *hi, int8_t *lo) {
    int l = ((x + 128) & 0xFF) - 128;     /* low byte as signed [-128,127] */
    int h = (x - l) >> 8;                  /* exact, also in [-128,127] */
    *lo = (int8_t)l; *hi = (int8_t)h;
}

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s <M> <K> <N>  (K%%32, N%%32, M%%4||1)\n", argv[0]); return -1; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    if (M <= 0 || (M % 4 != 0 && M != 1)) { fprintf(stderr, "M must be %%4 or 1\n"); return -1; }
    if (K <= 0 || K % 32 != 0) { fprintf(stderr, "K must be %%32\n"); return -1; }
    if (N <= 0 || N % 32 != 0) { fprintf(stderr, "N must be %%32 (int8 N-group)\n"); return -1; }
    /* max |x| the signed/signed split represents exactly (see DOMAIN note above) */
    int range = env_int("ROCKET_INT16_RANGE", 32639);
    if (range < 1 || range > 32639) range = 32639;

    printf("int16 EXACT (byte-decomp): C[%d,%d] = A[%d,%d] x B[%d,%d]^T  range=+-%d\n",
           M, N, M, K, N, K, range);

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); return 2; }

    int16_t *A = malloc((size_t)M*K*sizeof(int16_t));
    int16_t *B = malloc((size_t)N*K*sizeof(int16_t));
    int64_t *C  = malloc((size_t)M*N*sizeof(int64_t));   /* HW result (library fn)  */
    int64_t *R  = malloc((size_t)M*N*sizeof(int64_t));   /* int64 reference          */
    if (!A||!B||!C||!R) { fprintf(stderr, "malloc failed\n"); return -1; }

    srand(1234);
    for (int i = 0; i < M*K; i++) A[i] = (int16_t)(rand() % (2*range+1) - range);
    for (int i = 0; i < N*K; i++) B[i] = (int16_t)(rand() % (2*range+1) - range);

    /* spot-check the split math matches the library's (catches a bad split early) */
    { int8_t h, l; split_i16(A[0], &h, &l);
      if (h*256 + l != A[0]) { fprintf(stderr, "BAD split A[0]=%d\n", A[0]); return -1; } }

    /* int64 reference */
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++) s += (int64_t)A[m*K+k] * (int64_t)B[n*K+k];
            R[m*N+n] = s;
        }

    int64_t t0 = now_us();
    int r = rocket_matmul_int16_exact(fd, M, K, N, A, B, C);  /* the library primitive */
    double ms = (now_us() - t0) / 1000.0;
    if (r) { fprintf(stderr, "rocket_matmul_int16_exact failed (%d)\n", r); return r; }

    int bad = 0, shown = 0; int64_t maxabs = 0;
    for (int i = 0; i < M*N; i++) {
        int64_t d = C[i] - R[i]; if (d < 0) d = -d;
        if (d > maxabs) maxabs = d;
        if (C[i] != R[i]) { bad++;
            if (shown++ < 8) printf("  mismatch i=%d (m=%d,n=%d) exp=%lld act=%lld\n",
                                    i, i/N, i%N, (long long)R[i], (long long)C[i]); }
    }
    printf("rocket_matmul_int16_exact (4x int8): %.3f ms | %d/%d wrong, max_abs_err=%lld\n",
           ms, bad, M*N, (long long)maxabs);
    printf(bad == 0 ? "==> PASS (bit-exact int16xint16->int64 via byte decomposition)\n"
                    : "==> FAIL\n");

    free(A);free(B);free(C);free(R);
    rocket_close(fd);
    return bad ? -1 : 0;
}
