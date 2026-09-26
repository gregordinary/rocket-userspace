// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_mm_bf16_probe.c — does the RK3576 contract bf16 operands?
 *
 * A probe, not a gate. The RK3588 runs bf16 at precision 3 on the same IP. The only RK3576
 * evidence against it is CORE 0x3018 = 3 run on fp16 BYTES, which returns a wrong surface
 * whether or not the part can do bf16, so it could not have found the positive. This runs
 * the matmul-form fp16 program (gen_matmul_fp16_rk3576) with its precision words moved to
 * 3 and bf16 bytes in both operands, and scores the result against an exact reference.
 *
 * Three words carry the precision in that stream:
 *   CNA  0x100C  proc_precision [9:7] and in_precision [6:4]   (0x20200120 at fp16)
 *   CORE 0x3018  proc_precision [10:8]                         (0x10000200 at fp16)
 *   DPU  0x4010  proc_precision [2:0]; [31:29] = 5 stays the fp32 output width
 * The arms patch different subsets of them, so a positive says which words it needs. The
 * coefficient buffer holds a scale table of fp16 1.0 (0x3C00). Read as bf16 that is 2^-7,
 * so the arms run it both ways, and a surface that is the reference times one constant is
 * reported as a UNIFORM SCALE rather than as wrong.
 *
 * Two fills, both exact in fp32 whatever the accumulation order:
 *   int   A = ai/4, B = bi/2 with ai in [-6,6], bi in [-3,3]: the fp16 gate's value set.
 *         bf16 and fp16 hold these exactly, so this arm tests the LAYOUT alone.
 *   wide  the same integers at A*2^40 and B*2^-50. Neither operand is representable in fp16
 *         (2^38 is over 65504, 2^-51 is under fp16's smallest subnormal), so only a datapath
 *         carrying bf16's 8-bit exponent can return the reference, which is 2^-10 of int's.
 *
 * The fp16 control runs first and last. The output is fp32, the one wide writer measured not
 * to poison the next submit, but a bf16 arm is a different program; the closing control is
 * the canary for that, and every non-exact surface is followed by a power cycle.
 *
 *   rk3576_mm_bf16_probe [MxKxN ...]      default 16x256x64 64x1024x128
 *   rk3576_mm_bf16_probe round [M N]      default M 4, N 64; K runs 32 to 2048
 *
 * The round mode asks what the first cannot: how a bf16 sum that is NOT exact in fp32 is
 * rounded. Its fills carry full 8-bit bf16 mantissas, so every product is exact in fp32
 * (8 x 8 bits) and the running sum is not, and the device's output is scored against the
 * models rk3576_mm_fp16_round uses for the fp16 program: the whole dot product rounded once
 * ("exact"), fp32 accumulation one product at a time ("seq"), and blocks of B products summed
 * exactly and rounded to fp32 before an fp32 accumulation ("blkB", with "-rz" rounding that
 * accumulation toward zero). It runs the program with all three precision words at bf16,
 * which is what an entry would emit, and runs each shape twice.
 *
 * Exit 0 when both controls are exact (round mode: when every run completed), 1 otherwise,
 * 2 with no RK3576 device. The bf16 verdicts are reported, not scored.
 */
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"
#include "test_fill.h"

#define STAMP 0xA5u

enum { FILL_INT, FILL_WIDE };

struct arm {
    const char *name;
    int bf16_data;      /* operands are bf16 bytes, not fp16            */
    int cna, core, dpu; /* which precision words move to 3              */
    int bf16_scale;     /* the coefficient scale table holds bf16 1.0   */
};

static const struct arm ARMS[] = {
    { "fp16 control",          0, 0, 0, 0, 0 },
    { "bf16 all three",        1, 1, 1, 1, 0 },
    { "bf16 all three, bscale",1, 1, 1, 1, 1 },
    { "bf16 core+dpu",         1, 0, 1, 1, 0 },
    { "bf16 core+dpu, bscale", 1, 0, 1, 1, 1 },
    { "bf16 cna+core",         1, 1, 1, 0, 0 },
    { "bf16 core only",        1, 0, 1, 0, 0 },
    { "bf16 data, fp16 prog",  1, 0, 0, 0, 0 },
    { "fp16 control (canary)", 0, 0, 0, 0, 0 },
};
#define N_ARMS (sizeof ARMS / sizeof ARMS[0])

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint16_t to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);   /* every value here has a mantissa of 3 bits or fewer */
}

static uint16_t to_fp16(float f)
{
    _Float16 h = (_Float16)f;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}

/* Rewrite one register's value in an emitted stream; the op and register stay. */
static int patch(uint64_t *ops, unsigned n_ops, uint16_t reg, uint32_t clear, uint32_t set)
{
    for (unsigned i = 0; i < n_ops; i++) {
        if ((uint16_t)(ops[i] & 0xffff) != reg) continue;
        uint32_t v = (uint32_t)(ops[i] >> 16);
        v = (v & ~clear) | set;
        ops[i] = (ops[i] & ~(0xffffffffull << 16)) | ((uint64_t)v << 16);
        return 0;
    }
    return -1;
}

/* One arm, one fill, one shape. Returns 1 when the surface equals the reference. */
static int run_arm(int fd, const struct arm *a, int fill, unsigned m, unsigned k, unsigned n)
{
    size_t ib = rocket_rk3576_mm_fp16_in_bytes(m, k);
    size_t wb = rocket_rk3576_mm_fp16_weight_bytes(k, n);
    size_t cb = rocket_rk3576_mm_fp16_coef_bytes(n);
    size_t ob = rocket_rk3576_mm_fp16_out_bytes(m, n);
    size_t ne = (size_t)m * n, e;
    const double a_sc = fill == FILL_WIDE ? ldexp(0.25, 40) : 0.25;
    const double b_sc = fill == FILL_WIDE ? ldexp(0.5, -50) : 0.5;
    rocket_bo in = {0}, w = {0}, coef = {0}, out = {0}, rc = {0};
    uint64_t ops[RK3576_MM_FP16_OPS];
    int8_t *ai = malloc((size_t)m * k), *bi = malloc((size_t)n * k);
    uint16_t *B = malloc((size_t)n * k * 2);
    int32_t *ref = malloc(ne * 4);
    uint32_t in_h[4], out_h[1];
    int exact = 0;

    if (!ai || !bi || !B || !ref) { printf("  out of memory\n"); goto out; }
    for (e = 0; e < (size_t)m * k; e++) ai[e] = (int8_t)tf_int(0xA1, e, -6, 6);
    for (e = 0; e < (size_t)n * k; e++) {
        bi[e] = (int8_t)tf_int(0xB2, e, -3, 3);
        B[e] = a->bf16_data ? to_bf16((float)(bi[e] * b_sc)) : to_fp16((float)(bi[e] * b_sc));
    }
    for (unsigned r = 0; r < m; r++)
        for (unsigned c = 0; c < n; c++) {
            const int8_t *x = ai + (size_t)r * k, *y = bi + (size_t)c * k;
            int32_t acc = 0;
            for (unsigned i = 0; i < k; i++) acc += (int32_t)x[i] * y[i];
            ref[(size_t)r * n + c] = acc;
        }

    if (rocket_bo_alloc32(fd, ib, &in) || rocket_bo_alloc32(fd, wb, &w) ||
        rocket_bo_alloc32(fd, cb, &coef) || rocket_bo_alloc32(fd, ob, &out) ||
        rocket_bo_alloc32(fd, sizeof ops, &rc)) {
        printf("  BO allocation failed\n");
        goto out;
    }
    rocket_bo_prep(fd, &in, 1, 0);
    {
        uint16_t *A = (uint16_t *)in.ptr;
        for (e = 0; e < (size_t)m * k; e++)
            A[e] = a->bf16_data ? to_bf16((float)(ai[e] * a_sc))
                                : to_fp16((float)(ai[e] * a_sc));
        memset((unsigned char *)in.ptr + (size_t)m * k * 2, 0, ib - (size_t)m * k * 2);
    }
    rocket_bo_fini(fd, &in);
    rocket_bo_prep(fd, &w, 1, 0);
    rocket_rk3576_mm_fp16_pack_weights(w.ptr, w.size, B, k, n);
    rocket_bo_fini(fd, &w);
    rocket_bo_prep(fd, &coef, 1, 0);
    rocket_rk3576_mm_fp16_pack_coef(coef.ptr, coef.size, n);
    if (a->bf16_scale) {
        /* The scale table follows the 64-byte-a-group record table: see
         * r76_mmf16_table_bytes() / r76_mmf16_scale_bytes(). */
        size_t tb = (size_t)((n + 7) / 8) * 64, sb = (size_t)((n + 7) & ~7u) * 2;
        uint16_t one = 0x3F80;   /* bf16 1.0 */
        for (size_t i = 0; i < sb / 2; i++)
            memcpy((unsigned char *)coef.ptr + tb + i * 2, &one, 2);
    }
    rocket_bo_fini(fd, &coef);
    if (gen_matmul_fp16_rk3576(ops, m, k, n, (uint32_t)in.dma_address,
                               (uint32_t)w.dma_address, (uint32_t)out.dma_address,
                               (uint32_t)coef.dma_address) != RK3576_MM_FP16_OPS) {
        printf("  the generator refused\n");
        goto out;
    }
    if ((a->cna  && patch(ops, RK3576_MM_FP16_OPS, 0x100C, 0x3f0u, (3u << 7) | (3u << 4))) ||
        (a->core && patch(ops, RK3576_MM_FP16_OPS, 0x3018, 0x700u, 3u << 8)) ||
        (a->dpu  && patch(ops, RK3576_MM_FP16_OPS, 0x4010, 0x7u, 3u))) {
        printf("  a precision word is missing from the stream\n");
        goto out;
    }
    rocket_bo_prep(fd, &rc, 1, 0);
    memcpy(rc.ptr, ops, sizeof ops);
    rocket_bo_fini(fd, &rc);
    in_h[0] = in.handle; in_h[1] = w.handle; in_h[2] = coef.handle; in_h[3] = rc.handle;
    out_h[0] = out.handle;

    {
        size_t unwritten = 0, wrong = 0, first = 0, nz = 0, on_scale = 0;
        double scale = 0.0, max_rel = 0.0, dev, t0;
        const uint32_t *o;
        int src;

        rocket_bo_prep(fd, &out, 1, 0);
        memset(out.ptr, STAMP, ob);
        rocket_bo_fini(fd, &out);
        t0 = now_ms();
        src = rocket_submit_matmul(fd, &rc, RK3576_MM_FP16_OPS, in_h, 4, out_h, 1, 4000);
        if (src == 0 && rocket_bo_prep(fd, &out, 0, 4000000000ull) < 0) src = -1;
        dev = now_ms() - t0;
        if (src != 0) {
            printf("  submit or wait failed (%d)\n", src);
            rocket_rk3576_power_idle();
            goto out;
        }
        o = (const uint32_t *)out.ptr;
        for (e = 0; e < ne; e++) {
            double want = (double)ref[e] * 0.125 * (fill == FILL_WIDE ? ldexp(1.0, -10) : 1.0);
            float got;
            if (o[e] == STAMP * 0x01010101u) { unwritten++; continue; }
            memcpy(&got, &o[e], 4);
            if ((double)got != want) {
                if (!wrong++) first = e;
            }
            if (want != 0.0) {
                double r = (double)got / want;
                if (!nz++) scale = r;
                if (r == scale) on_scale++;
                if (fabs(r - 1.0) > max_rel) max_rel = fabs(r - 1.0);
            }
        }
        printf("  %-4s %6.3f ms  %-9s unwritten %zu wrong %zu of %zu", fill == FILL_WIDE ?
               "wide" : "int", dev, !unwritten && !wrong ? "EXACT" :
               unwritten ? "UNWRITTEN" : "WRONG", unwritten, wrong, ne);
        if (wrong) {
            float got; memcpy(&got, &o[first], 4);
            printf("  first m=%zu n=%zu got %.9g want %.9g", first / n, first % n,
                   (double)got, (double)ref[first] * 0.125 *
                   (fill == FILL_WIDE ? ldexp(1.0, -10) : 1.0));
            if (nz && on_scale == nz && scale != 1.0)
                printf("  UNIFORM SCALE %.9g (log2 %.3f)", scale, log2(fabs(scale)));
            else
                printf("  max |got/want-1| %.3g", max_rel);
        }
        printf("\n");
        rocket_bo_fini(fd, &out);
        fflush(stdout);
        exact = !unwritten && !wrong;
        if (!exact) rocket_rk3576_power_idle();
    }
out:
    if (rc.ptr) rocket_bo_free(fd, &rc);
    if (out.ptr) rocket_bo_free(fd, &out);
    if (coef.ptr) rocket_bo_free(fd, &coef);
    if (w.ptr) rocket_bo_free(fd, &w);
    if (in.ptr) rocket_bo_free(fd, &in);
    free(ai); free(bi); free(B); free(ref);
    return exact;
}

/* ---- round mode ------------------------------------------------------------------ */

static const unsigned KS[] = { 32, 64, 128, 256, 512, 1024, 2048 };
static const unsigned BLK[] = { 8, 16, 32, 64 };
#define NMODEL (2 + 2 * (int)(sizeof(BLK) / sizeof(BLK[0])))

static double bf16_val(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* Values with at most 8 significant bits, so each is exact in bf16. */
static void fill_bf16(uint16_t *p, size_t n, uint64_t seed, int wide)
{
    for (size_t i = 0; i < n; i++) {
        double v;
        if (!wide) {
            v = tf_int(seed, i, -255, 255) / 256.0;
        } else {
            const int m = tf_int(seed, 2 * i, 128, 255);
            const int e = tf_int(seed, 2 * i + 1, -20, -10);   /* m * 2^e: 2^-13 .. 2^-2 */
            v = ldexp((double)m, e) * (tf_int(seed ^ 0x55, i, 0, 1) ? 1.0 : -1.0);
        }
        p[i] = to_bf16((float)v);
    }
}

static uint32_t ulps(float a, float b)
{
    int32_t x, y;
    memcpy(&x, &a, 4);
    memcpy(&y, &b, 4);
    if (x < 0) x = (int32_t)0x80000000 - x;
    if (y < 0) y = (int32_t)0x80000000 - y;
    return x > y ? (uint32_t)(x - y) : (uint32_t)(y - x);
}

static void models(const uint16_t *a, const uint16_t *b, unsigned K, float *out)
{
    volatile float acc;
    double s = 0.0;
    int j = 0;

    for (unsigned k = 0; k < K; k++) s += bf16_val(a[k]) * bf16_val(b[k]);
    out[j++] = (float)s;
    acc = 0.0f;
    for (unsigned k = 0; k < K; k++) acc = acc + (float)(bf16_val(a[k]) * bf16_val(b[k]));
    out[j++] = acc;
    for (size_t bi = 0; bi < sizeof(BLK) / sizeof(BLK[0]); bi++) {
        for (int rz = 0; rz < 2; rz++) {
            acc = 0.0f;
            for (unsigned k0 = 0; k0 < K; k0 += BLK[bi]) {
                double bs = 0.0;
                for (unsigned k = k0; k < k0 + BLK[bi] && k < K; k++)
                    bs += bf16_val(a[k]) * bf16_val(b[k]);
                const float bf = (float)bs;
                if (rz) fesetround(FE_TOWARDZERO);
                acc = acc + bf;
                if (rz) fesetround(FE_TONEAREST);
            }
            out[j++] = acc;
        }
    }
}

static void model_name(int j, char *buf, size_t n)
{
    if (j == 0) snprintf(buf, n, "exact");
    else if (j == 1) snprintf(buf, n, "seq");
    else snprintf(buf, n, "blk%u%s", BLK[(j - 2) / 2], (j - 2) % 2 ? "-rz" : "");
}

/* One task of the matmul form at bf16, all three precision words moved. C gets m*n fp32. */
static int npu_bf16(int fd, unsigned m, unsigned k, unsigned n, const uint16_t *A,
                    const uint16_t *B, float *C)
{
    size_t ib = rocket_rk3576_mm_fp16_in_bytes(m, k), ob = rocket_rk3576_mm_fp16_out_bytes(m, n);
    rocket_bo in = {0}, w = {0}, coef = {0}, out = {0}, rc = {0};
    uint64_t ops[RK3576_MM_FP16_OPS];
    uint32_t in_h[4], out_h[1];
    size_t unwritten = 0;
    int ret = -1;

    if (rocket_bo_alloc32(fd, ib, &in) ||
        rocket_bo_alloc32(fd, rocket_rk3576_mm_fp16_weight_bytes(k, n), &w) ||
        rocket_bo_alloc32(fd, rocket_rk3576_mm_fp16_coef_bytes(n), &coef) ||
        rocket_bo_alloc32(fd, ob, &out) || rocket_bo_alloc32(fd, sizeof ops, &rc))
        goto out;
    rocket_bo_prep(fd, &in, 1, 0);
    memcpy(in.ptr, A, (size_t)m * k * 2);
    memset((unsigned char *)in.ptr + (size_t)m * k * 2, 0, ib - (size_t)m * k * 2);
    rocket_bo_fini(fd, &in);
    rocket_bo_prep(fd, &w, 1, 0);
    rocket_rk3576_mm_fp16_pack_weights(w.ptr, w.size, B, k, n);
    rocket_bo_fini(fd, &w);
    rocket_bo_prep(fd, &coef, 1, 0);
    rocket_rk3576_mm_fp16_pack_coef(coef.ptr, coef.size, n);
    rocket_bo_fini(fd, &coef);
    if (gen_matmul_fp16_rk3576(ops, m, k, n, (uint32_t)in.dma_address, (uint32_t)w.dma_address,
                               (uint32_t)out.dma_address, (uint32_t)coef.dma_address)
            != RK3576_MM_FP16_OPS ||
        patch(ops, RK3576_MM_FP16_OPS, 0x100C, 0x3f0u, (3u << 7) | (3u << 4)) ||
        patch(ops, RK3576_MM_FP16_OPS, 0x3018, 0x700u, 3u << 8) ||
        patch(ops, RK3576_MM_FP16_OPS, 0x4010, 0x7u, 3u))
        goto out;
    rocket_bo_prep(fd, &rc, 1, 0);
    memcpy(rc.ptr, ops, sizeof ops);
    rocket_bo_fini(fd, &rc);
    in_h[0] = in.handle; in_h[1] = w.handle; in_h[2] = coef.handle; in_h[3] = rc.handle;
    out_h[0] = out.handle;
    rocket_bo_prep(fd, &out, 1, 0);
    memset(out.ptr, STAMP, ob);
    rocket_bo_fini(fd, &out);
    if (rocket_submit_matmul(fd, &rc, RK3576_MM_FP16_OPS, in_h, 4, out_h, 1, 4000) ||
        rocket_bo_prep(fd, &out, 0, 4000000000ull) < 0)
        goto out;
    for (size_t e = 0; e < (size_t)m * n; e++)
        unwritten += ((const uint32_t *)out.ptr)[e] == STAMP * 0x01010101u;
    memcpy(C, out.ptr, (size_t)m * n * 4);
    rocket_bo_fini(fd, &out);
    ret = unwritten ? -2 : 0;
out:
    if (rc.ptr) rocket_bo_free(fd, &rc);
    if (out.ptr) rocket_bo_free(fd, &out);
    if (coef.ptr) rocket_bo_free(fd, &coef);
    if (w.ptr) rocket_bo_free(fd, &w);
    if (in.ptr) rocket_bo_free(fd, &in);
    return ret;
}

static int run_round(int fd, unsigned M, unsigned K, unsigned N, int wide)
{
    uint16_t *A = malloc((size_t)M * K * 2), *B = malloc((size_t)N * K * 2);
    float *C = malloc((size_t)M * N * 4), *C2 = malloc((size_t)M * N * 4);
    float mv[NMODEL];
    size_t match[NMODEL] = { 0 };
    uint32_t worst[NMODEL] = { 0 };
    int rc, rc2, same;

    if (!A || !B || !C || !C2) return -1;
    fill_bf16(A, (size_t)M * K, 0xF1 + K + 7u * (unsigned)wide, wide);
    fill_bf16(B, (size_t)N * K, 0xF2 + K + 7u * (unsigned)wide, wide);
    rc = npu_bf16(fd, M, K, N, A, B, C);
    rc2 = npu_bf16(fd, M, K, N, A, B, C2);
    if (rc || rc2) {
        printf("  %s %ux%ux%u: the runs returned %d and %d\n", wide ? "wide" : "unit", M, K, N,
               rc, rc2);
        rocket_rk3576_power_idle();
        free(A); free(B); free(C); free(C2);
        return -1;
    }
    same = memcmp(C, C2, (size_t)M * N * 4) == 0;
    for (unsigned m = 0; m < M; m++)
        for (unsigned n = 0; n < N; n++) {
            const float dev = C[(size_t)m * N + n];
            models(A + (size_t)m * K, B + (size_t)n * K, K, mv);
            for (int j = 0; j < NMODEL; j++) {
                const uint32_t u = ulps(dev, mv[j]);
                match[j] += u == 0;
                if (u > worst[j]) worst[j] = u;
            }
        }
    printf("  %s %ux%4ux%u, two runs %s:", wide ? "wide" : "unit", M, K, N,
           same ? "identical" : "DIFFER");
    for (int j = 0; j < NMODEL; j++) {
        char nm[16];
        model_name(j, nm, sizeof nm);
        printf(" %s %.1f%%/%u", nm, 100.0 * (double)match[j] / (double)(M * N), worst[j]);
    }
    printf("\n");
    free(A); free(B); free(C); free(C2);
    return 0;
}

int main(int argc, char **argv)
{
    static const char *def[] = { "16x256x64", "64x1024x128" };
    const char **shapes = argc > 1 ? (const char **)argv + 1 : def;
    int n_shapes = argc > 1 ? argc - 1 : 2;
    int fd, s, controls_ok = 1;

    fd = rocket_open();
    if (fd < 0 || strcmp(rocket_hw_current()->name, "rk3576") != 0) {
        printf("no RK3576 device\n");
        return 2;
    }
    if (argc > 1 && strcmp(argv[1], "round") == 0) {
        unsigned M = argc > 3 ? (unsigned)atoi(argv[2]) : 4, N = argc > 3 ? (unsigned)atoi(argv[3]) : 64;
        int bad = 0;
        printf("== bf16 matmul rounding, M %u N %u: per model, %% of outputs bit-exact / worst ulps ==\n",
               M, N);
        for (int wide = 0; wide < 2; wide++)
            for (size_t i = 0; i < sizeof(KS) / sizeof(KS[0]); i++)
                bad += run_round(fd, M, KS[i], N, wide) != 0;
        rocket_close(fd);
        return bad ? 1 : 0;
    }
    for (s = 0; s < n_shapes; s++) {
        unsigned m, k, n;
        if (sscanf(shapes[s], "%ux%ux%u", &m, &k, &n) != 3 ||
            !rocket_rk3576_mm_fp16_task_ok(m, k, n)) {
            fprintf(stderr, "bad or out-of-envelope shape %s\n", shapes[s]);
            return 1;
        }
        printf("== M=%u K=%u N=%u\n", m, k, n);
        for (unsigned i = 0; i < N_ARMS; i++) {
            const struct arm *a = &ARMS[i];
            int ok;
            printf("%s (cna %d core %d dpu %d)\n", a->name, a->cna, a->core, a->dpu);
            ok = run_arm(fd, a, FILL_INT, m, k, n);
            /* The wide fill is out of fp16's range by construction, so an fp16 program
             * on fp16 bytes cannot carry it; the controls run the int fill only. */
            if (a->bf16_data) run_arm(fd, a, FILL_WIDE, m, k, n);
            if (!a->bf16_data && !ok) controls_ok = 0;
        }
    }
    printf("controls %s\n", controls_ok ? "EXACT" : "FAILED");
    rocket_close(fd);
    return controls_ok ? 0 : 1;
}
