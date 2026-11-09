// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * reduce_mean_rocket.c — standalone test for on-NPU spatial reductions over [H,W]:
 * GlobalAveragePool / Mean / ReduceMean (rocket_global_avgpool_fp16) AND the extrema
 * GlobalMaxPool / GlobalMinPool / ReduceMax / ReduceMin (rocket_global_{max,min}pool_fp16),
 * all the multi-pass PPU telescoping reduction. Avg is tolerance-checked (per-pass fp16
 * reciprocal); max/min are idempotent so the decomposed NPU result is BIT-EXACT.
 *
 * Layers (mirroring the conv/pool bring-ups):
 *
 *  1. FACTOR-AXIS unit checks (anywhere): rocket_reduce_factor_axis produces factors
 *     in [2,8] whose product is the axis, and flags non-8-smooth axes (prime > 8). The
 *     plan's pass schedule keeps every pass inside the PPU's output-width allowance.
 *
 *  2. DECOMPOSITION / CUBE-LAYOUT self-check (anywhere, no NPU): simulate the runtime's
 *     exact pass schedule (rocket_global_pool_passes) in host cube buffers (scatter -> telescoped avg-pool through
 *     feature_index() cube indices, ping-ponged across passes -> de-scatter 1x1xC) and
 *     compare to the naive global mean. A PASS proves the factor schedule + per-pass
 *     dims + cube read/write layout chain express the global mean (the part verifiable
 *     off-hardware; recip quant is a HW-only concern checked in layer 3).
 *
 *  3. ON-HARDWARE end-to-end (only if /dev/accel/accel0 opens): rocket_global_avgpool_fp16
 *     on the NPU vs the fp64 host oracle, within tolerance (each pass divides via the PPU
 *     per-axis fp16(65536/k) reciprocal). Non-decomposable shapes (13x13, 17x17, 19x19,
 *     1024x1024) exercise the exact host fallback. Planes of 216 to 512, which ran a pass
 *     past the allowance when the windows went to 16, run on the NPU.
 *
 * Usage: reduce_mean_rocket               (sweep)
 *        reduce_mean_rocket C H W         (one shape)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_reduce.h"
#include "rocket_pool.h"     /* rocket_pool_fp16_max_ow */
#include "npu_pool.h"      /* ppu_recip_kernel_fp16 (not used directly; layout parity) */
#include "npu_matmul.h"    /* feature_index */
#include "test_fill.h"

/* ---- layer 1: factorisation ---- */
static int check_factor(int n, int expect_ok)
{
    int f[ROCKET_REDUCE_MAX_DIM > 64 ? 64 : ROCKET_REDUCE_MAX_DIM];
    int c = rocket_reduce_factor_axis(n, f, 64);
    if (!expect_ok) {
        int ok = (c < 0);
        printf("  factor(%d): %s (expected not-8-smooth) -> %s\n",
               n, c < 0 ? "rejected" : "accepted", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (c < 0) { printf("  factor(%d): rejected but should decompose -> FAIL\n", n); return 1; }
    long prod = 1; int bad = 0;
    for (int i = 0; i < c; i++) { prod *= f[i]; if (f[i] < 2 || f[i] > 8) bad = 1; }
    int ok = (prod == (n < 1 ? 1 : (n == 1 ? 1 : n))) && !bad;
    if (n == 1) ok = (c == 0);                 /* 1 -> no passes */
    printf("  factor(%d): %d factors, prod=%ld%s -> %s\n",
           n, c, prod, bad ? " (out-of-range!)" : "", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* ---- layer 2: schedule + cube-layout self-check (true division) ---- */
static int self_check(int C, int H, int W, const _Float16 *in)
{
    if (rocket_global_avgpool_plan(C, H, W) != 0) {
        printf("  self-check C=%d %dx%d: not decomposable (host fallback) -> SKIP\n", C, H, W);
        return 0;
    }
    int fh[64], fw[64];
    int npass = rocket_global_pool_passes(C, H, W, fh, fw, 64);
    if (npass < 0) { printf("  self-check C=%d %dx%d: no schedule (%d) -> FAIL\n", C, H, W, npass); return 1; }
    int nh = npass, nw = npass;
    const int C1 = (C + 7) / 8;
    /* every pass inside the PPU's output-width allowance for its window */
    for (int i = 0, cw = W; i < npass; i++) {
        rocket_pool_desc pd = { .c = C, .ih = 1, .iw = cw, .kh = fh[i], .kw = fw[i],
                                .stride_y = fh[i], .stride_x = fw[i], .method = POOL_METHOD_AVG };
        cw /= fw[i];
        if (fh[i] > 8 || fw[i] > 8 || (unsigned)cw > rocket_pool_fp16_max_ow(&pd)) {
            printf("  self-check C=%d %dx%d: pass %d (%dx%d) to width %d is past its "
                   "allowance -> FAIL\n", C, H, W, i, fh[i], fw[i], cw);
            return 1;
        }
    }

    _Float16 *A = calloc((size_t)C1 * H * W * 8, sizeof(_Float16));
    _Float16 *B = calloc((size_t)C1 * H * W * 8, sizeof(_Float16));
    _Float16 *got = calloc((size_t)C, sizeof(_Float16));
    _Float16 *ref = calloc((size_t)C, sizeof(_Float16));
    if (!A || !B || !got || !ref) { fprintf(stderr, "oom\n"); free(A);free(B);free(got);free(ref); return 1; }

    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                A[feature_index(C, H, W, 8, c+1, h+1, w+1)] = in[((size_t)c*H+h)*W+w];

    int ch = H, cw = W;
    _Float16 *finalc = A;
    for (int i = 0; i < npass; i++) {
        int kh = (i < nh) ? fh[i] : 1, kw = (i < nw) ? fw[i] : 1;
        if (kh == 1 && kw == 1) continue;
        int oh = ch / kh, ow = cw / kw;
        _Float16 *src = (i & 1) ? B : A, *dst = (i & 1) ? A : B;
        for (int c = 0; c < C; c++)
            for (int oy = 0; oy < oh; oy++)
                for (int ox = 0; ox < ow; ox++) {
                    float acc = 0.f;
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++)
                            acc += (float)src[feature_index(C, ch, cw, 8, c+1,
                                          oy*kh+ky+1, ox*kw+kx+1)];
                    dst[feature_index(C, oh, ow, 8, c+1, oy+1, ox+1)] =
                        (_Float16)(acc / (float)(kh*kw));   /* true division */
                }
        ch = oh; cw = ow; finalc = dst;
    }
    for (int c = 0; c < C; c++) got[c] = finalc[feature_index(C, ch, cw, 8, c+1, 1, 1)];
    rocket_global_avgpool_ref_fp16(C, H, W, in, ref);

    double max_abs = 0;
    for (int c = 0; c < C; c++) {
        double ad = fabs((float)got[c] - (float)ref[c]);
        if (ad > max_abs) max_abs = ad;
    }
    free(A); free(B); free(got); free(ref);
    /* true-division telescoping == direct mean up to fp16 rounding of the per-pass outputs */
    int ok = max_abs <= 0.02;
    printf("  self-check C=%d %dx%d (%d passes, first %dx%d): max_abs=%.5f -> %s\n",
           C, H, W, npass, npass ? fh[0] : 1, npass ? fw[0] : 1, max_abs, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int run_shape(int fd, int C, int H, int W)
{
    int plan = rocket_global_avgpool_plan(C, H, W);
    printf("C=%d %dx%d  (%s)\n", C, H, W,
           plan == 0 ? "NPU decomposable" : "host fallback");

    size_t n = (size_t)C * H * W;
    _Float16 *in = malloc(n * sizeof(_Float16));
    _Float16 *out = malloc((size_t)C * sizeof(_Float16));
    _Float16 *ref = malloc((size_t)C * sizeof(_Float16));
    if (!in || !out || !ref) { fprintf(stderr, "oom\n"); free(in);free(out);free(ref); return 1; }

    /* Each channel is its own hashed offset in [0.5,2.5] plus noise in [-1,1], so every
     * mean is O(1) and the channels' means differ. A zero-mean fill made each mean about
     * 0.04, where a divisor off by 0.75x moves it by 0.01 and passed; and identical means
     * across channels would hide a channel mixup. */
    uint32_t st = 0x1234u + (uint32_t)(C*131 + H*17 + W);
    for (size_t i = 0; i < n; i++) {
        st = st*1664525u + 1013904223u;
        float noise = ((float)((st >> 9) & 0x3ff) / 1023.f) * 2.f - 1.f;          /* [-1,1] */
        float off = 0.5f + 2.0f * (float)(tf_hash(0x3EA7u, i / ((size_t)H * W)) >> 11)
                                 * 0x1.0p-53f;                                     /* [0.5,2.5] */
        in[i] = (_Float16)(off + noise);
    }

    int fail = self_check(C, H, W, in);

    memset(out, 0, (size_t)C*sizeof(_Float16));
    int r = rocket_global_avgpool_fp16(fd, C, H, W, in, out);
    if (r) { printf("  rocket_global_avgpool_fp16 = %d -> FAIL\n", r); fail = 1; }
    else {
        rocket_global_avgpool_ref_fp16(C, H, W, in, ref);
        const char *tag = (fd >= 0 && plan == 0) ? "HW" : "host";
        double max_abs = 0, max_rel = 0; int bad = 0;
        for (int c = 0; c < C; c++) {
            double ad = fabs((float)out[c] - (float)ref[c]);
            double rd = ad / (fabs((float)ref[c]) + 1e-3);
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) max_rel = rd;
            /* 0.004 + 0.4% of |ref|: about 10x the multi-pass fp16 rounding at these
             * magnitudes, and a divisor off by 2% already misses it */
            if (!(ad <= 0.004 + 0.004 * fabs((float)ref[c]))) {
                if (bad < 6) printf("    [c=%d] ref=%.4f got=%.4f d=%.4f\n",
                                    c, (float)ref[c], (float)out[c], ad);
                bad++;
            }
        }
        printf("  %s: max_abs=%.5f max_rel=%.4f bad=%d -> %s\n",
               tag, max_abs, max_rel, bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) fail = 1;
    }

    /* GlobalMaxPool / GlobalMinPool over [H,W]: idempotent => EXACT (no reciprocal). On a
     * decomposable shape the NPU telescoped result must be BIT-EXACT vs the host reduction. */
    const char *etag = (fd >= 0 && plan == 0) ? "HW" : "host";
    struct { const char *name;
             int (*run)(int,int,int,int,const _Float16*,_Float16*);
             void (*ref)(int,int,int,const _Float16*,_Float16*); } ex[] = {
        { "max", rocket_global_maxpool_fp16, rocket_global_maxpool_ref_fp16 },
        { "min", rocket_global_minpool_fp16, rocket_global_minpool_ref_fp16 },
    };
    for (int e = 0; e < 2; e++) {
        memset(out, 0, (size_t)C*sizeof(_Float16));
        int r2 = ex[e].run(fd, C, H, W, in, out);
        if (r2) { printf("  global_%spool = %d -> FAIL\n", ex[e].name, r2); fail = 1; continue; }
        ex[e].ref(C, H, W, in, ref);
        int bad = 0;
        for (int c = 0; c < C; c++)
            if (out[c] != ref[c]) {
                if (bad < 6) printf("    [%s c=%d] ref=%.4f got=%.4f\n",
                                    ex[e].name, c, (float)ref[c], (float)out[c]);
                bad++;
            }
        printf("  %s %s: exact bad=%d -> %s\n", etag, ex[e].name, bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) fail = 1;
    }

    free(in); free(out); free(ref);
    return fail;
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0)
        printf("note: no /dev/accel/accel0 (%d) — factor + self-check + host path only\n\n", fd);

    int fail = 0;

    if (argc == 4) {
        fail = run_shape(fd, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
        if (fd >= 0) rocket_close(fd);
        if (fd < 0 && !fail) {
            printf("no NPU: the host checks passed and the device path never ran -> SKIP\n");
            return 2;
        }
        printf("==== %s ====\n", fail ? "FAIL" : "PASS");
        return fail ? 1 : 0;
    }

    printf("-- factorisation --\n");
    fail |= check_factor(7, 1);   fail |= check_factor(14, 1);  fail |= check_factor(16, 1);
    fail |= check_factor(28, 1);  fail |= check_factor(56, 1);  fail |= check_factor(32, 1);
    fail |= check_factor(64, 1);  fail |= check_factor(49, 1);  fail |= check_factor(1, 1);
    fail |= check_factor(17, 0);  fail |= check_factor(19, 0);  fail |= check_factor(289, 0); /* 17*17 */
    fail |= check_factor(11, 0);  fail |= check_factor(13, 0);  fail |= check_factor(216, 1);
    fail |= check_factor(512, 1);
    printf("\n");

    struct { int C, H, W; } shapes[] = {
        {  8,  7,  7 },   /* classifier-head global pool, single pass (<=8)          */
        { 16, 14, 14 },   /* two passes ([2,7]), 2 C-planes                          */
        { 64, 28, 28 },   /* two passes ([4,7]); SE-block scale                      */
        { 32, 56, 56 },   /* two passes ([7,8])                                      */
        { 24, 32, 32 },   /* power-of-two ([4,8])                                    */
        { 16, 64, 64 },   /* ([8,8])                                                 */
        {130,  7,  7 },   /* C not a multiple of 8 (padding-channel handling)        */
        { 16, 28, 20 },   /* equal-count rectangle: asymmetric kernels (7x5 pass)    */
        { 16, 56,  8 },   /* unequal factor count -> exact host fallback             */
        {512,  7,  7 },   /* realistic EfficientNet head width                        */
        {  8, 13, 13 },   /* not 8-smooth -> exact host fallback                      */
        {  8, 17, 17 },   /* not 8-smooth -> exact host fallback                      */
        { 12, 19, 19 },   /* not 8-smooth                                             */
        /* Planes whose passes went past the PPU's output-width allowance when the windows
         * went to 16: each was a plausible wrong mean on the RK1 (2026-11-02). */
        {  8,216,216 },   /* [3,3,3,8] puts 3x3 to 72: runs largest first, 8x8 to 27  */
        { 16,256,256 },   /* [4,8,8]: 4x4 to 64                                       */
        {  8,288,288 },   /* [6,6,8]: 6x6 to 48                                       */
        {  8,480,480 },   /* [2,5,6,8] puts 2x2 to 240: largest first, 8x8 to 60      */
        {  8,512,512 },   /* [8,8,8]: 8x8 to 64                                       */
        {  8,1024,1024 }, /* no order fits its first pass -> exact host fallback      */
    };
    for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++) {
        fail |= run_shape(fd, shapes[i].C, shapes[i].H, shapes[i].W);
        printf("\n");
    }

    if (fd >= 0) rocket_close(fd);
    if (fd < 0 && !fail) {
        /* The host checks passed, but the device path never ran, so this is not a pass. */
        printf("no NPU: the host checks passed and the device path never ran -> SKIP\n");
        return 2;
    }
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
