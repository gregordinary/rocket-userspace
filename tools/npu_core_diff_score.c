// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * npu_core_diff_score.c — score tests/npu_core_diff_probe's raw-mode outputs against the
 * host model of the RK3588's fp16 x fp16 -> fp32 matmul.
 *
 * The model (the "base model"): each aligned 32-product K group of a dot product is summed
 * EXACTLY (every fp16 x fp16 product is exact, the sum is taken in a 2^-48 fixed point wide
 * enough for any fp16 pair) and rounded once to fp32, round to nearest even; the K/32 group
 * partials are then accumulated in K order in fp32, round to nearest even, starting from 0.
 * On the vendor RK1 (rknpu 0.9.8, 6.1.172, 300 MHz and 1000 MHz) it reproduces every output
 * element of every core bit for bit, except rare elements of output lane n % 16 == 0; which
 * lane-0 elements differ depends on the core. It is written for M 512, K 384, N 256, the
 * probe's raw shape.
 *
 * A lane-0 event lowers ONE group's exact sum by one unit of bit 6 of that group's coarse frame
 * before the fp32 rounding: 2^(4 Cmax - 44), where an fp16 word's coarse exponent is
 * max(e, 1) >> 2 and Cmax is the group's largest coarse exponent sum over its nonzero products
 * (the form NVDLA's CMAC multiplies in: significand << (e & 3) at exponent 4 (e >> 2)). Every
 * recorded event had Cmax 7, so it read as an absolute 2^-16; scaling every weight by 16 moves
 * it to 2^-12 on the same elements (runs s13, s14). NPU_CORE_DIFF_ATTRIB=1 appends to each
 * differing element the groups whose single event reproduces the device value bit for bit
 * ("ev g1 g5"), or "ev none"; with it set, the lines have six or more fields.
 *
 * Output: one line per element that differs, "tile m n device_bits model_bits" (hex fp32),
 * and a per-tile count on stderr. Build on the host or the board: cc -O2 -o score this.c -lm
 * usage: npu_core_diff_score INDIR OUTDIR NTILES [FIRST]
 *   INDIR holds raw_NNN.a16 / raw_NNN.b16 (the probe's raw inputs), OUTDIR raw_NNN.f32.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

enum { M = 512, K = 384, N = 256, G = 32 };
typedef __int128 i128;

/* An fp16 x fp16 product as an integer at scale 2^-48 (exact: |p| < 2^32, LSB >= 2^-48). */
static i128 prod(uint16_t a, uint16_t b)
{
    int ea = (a >> 10) & 31, eb = (b >> 10) & 31;
    int64_t ma = ea ? (a & 1023) | 1024 : a & 1023, mb = eb ? (b & 1023) | 1024 : b & 1023;
    int sh = (ea ? ea - 25 : -24) + (eb ? eb - 25 : -24) + 48;
    i128 v = (i128)(ma * mb) << sh;
    return ((a ^ b) & 0x8000) ? -v : v;
}

/* Coarse exponent sum of one product: max(e, 1) >> 2 per operand; -1 for a zero product. */
static int coarse(uint16_t a, uint16_t b)
{
    int ea = (a >> 10) & 31, eb = (b >> 10) & 31;
    if (!(a & 0x7fff) || !(b & 0x7fff)) return -1;
    return ((ea ? ea : 1) >> 2) + ((eb ? eb : 1) >> 2);
}

static int msb128(unsigned __int128 v)
{
    uint64_t hi = (uint64_t)(v >> 64), lo = (uint64_t)v;
    return hi ? 127 - __builtin_clzll(hi) : lo ? 63 - __builtin_clzll(lo) : -1;
}

/* Round an int128 at scale 2^-48 to fp32, nearest even. */
static float to_f32(i128 v)
{
    int neg = v < 0, ms;
    unsigned __int128 mag = neg ? (unsigned __int128)(-v) : (unsigned __int128)v;
    double r;
    if ((ms = msb128(mag)) < 0) return 0.0f;
    if (ms >= 24) {
        int sh = ms - 23;
        unsigned __int128 q = mag >> sh, rem = mag & (((unsigned __int128)1 << sh) - 1);
        unsigned __int128 half = (unsigned __int128)1 << (sh - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        r = ldexp((double)(uint64_t)q, sh - 48);
    } else {
        r = ldexp((double)(uint64_t)mag, -48);
    }
    return (float)(neg ? -r : r);
}

static int rd(const char *p, void *b, size_t n)
{
    FILE *f = fopen(p, "rb");
    size_t got;
    if (!f) return -1;
    got = fread(b, 1, n, f);
    fclose(f);
    return got == n ? 0 : -1;
}

int main(int argc, char **argv)
{
    uint16_t *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    float *C = malloc((size_t)M * N * 4);
    char p[4096];
    long tot = 0, bad = 0;
    int nt, t0, t, m, n, g, k, attrib;
    if (argc < 4) {
        fprintf(stderr, "usage: %s INDIR OUTDIR NTILES [FIRST]\n", argv[0]);
        return 1;
    }
    nt = atoi(argv[3]); t0 = argc > 4 ? atoi(argv[4]) : 0;
    attrib = getenv("NPU_CORE_DIFF_ATTRIB") && atoi(getenv("NPU_CORE_DIFF_ATTRIB"));
    if (!A || !B || !C) return 1;
    for (t = t0; t < t0 + nt; t++) {
        long tb = 0;
        snprintf(p, sizeof p, "%s/raw_%03d.a16", argv[1], t);
        if (rd(p, A, (size_t)M * K * 2)) { fprintf(stderr, "%s: read failed\n", p); return 1; }
        snprintf(p, sizeof p, "%s/raw_%03d.b16", argv[1], t);
        if (rd(p, B, (size_t)N * K * 2)) { fprintf(stderr, "%s: read failed\n", p); return 1; }
        snprintf(p, sizeof p, "%s/raw_%03d.f32", argv[2], t);
        if (rd(p, C, (size_t)M * N * 4)) { fprintf(stderr, "%s: read failed\n", p); return 1; }
        for (m = 0; m < M; m++)
            for (n = 0; n < N; n++) {
                const uint16_t *a = A + (size_t)m * K, *b = B + (size_t)n * K;
                float acc = 0.0f;
                uint32_t dv, mv;
                i128 gsum[K / G];
                int cmax[K / G];
                for (g = 0; g < K / G; g++) {
                    i128 s = 0;
                    cmax[g] = -1;
                    for (k = 0; k < G; k++) {
                        int c = coarse(a[g * G + k], b[g * G + k]);
                        s += prod(a[g * G + k], b[g * G + k]);
                        if (c > cmax[g]) cmax[g] = c;
                    }
                    gsum[g] = s;
                    acc = acc + to_f32(s);
                }
                memcpy(&dv, &C[(size_t)m * N + n], 4);
                memcpy(&mv, &acc, 4);
                if (dv != mv) {
                    printf("%d %d %d %08x %08x", t, m, n, dv, mv);
                    if (attrib) {         /* which single-group event reproduces the device value */
                        int hit = 0, h;
                        printf(" ev");
                        for (h = 0; h < K / G; h++) {
                            float alt = 0.0f;
                            uint32_t av;
                            if (cmax[h] < 0) continue;
                            for (g = 0; g < K / G; g++)
                                alt = alt + to_f32(g == h ? gsum[g] - ((i128)1 << (4 * cmax[g] + 4)) : gsum[g]);
                            memcpy(&av, &alt, 4);
                            if (av == dv) { printf(" g%d", h); hit++; }
                        }
                        if (!hit) printf(" none");
                    }
                    printf("\n");
                    tb++;
                }
            }
        tot += (long)M * N; bad += tb;
        fprintf(stderr, "tile %d: %ld of %d differ from the base model\n", t, tb, M * N);
    }
    fprintf(stderr, "total %ld of %ld differ\n", bad, tot);
    return 0;
}
