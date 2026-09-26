// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_mm_fp16_envelope.c — where one task of the RK3576 fp16 matmul program stops
 * being exact.
 *
 * A probe, not a gate. rocket_matmul_fp16_rk3576() refuses outside the envelope its gate
 * ran (M 16, K 2048, N 256), and those are bounds of the evidence, not of the part. This
 * runs the same register program as ONE task at any shape the closed forms can express,
 * through gen_matmul_fp16_rk3576_unchecked(), and says what came back.
 *
 * A wrong value here computes a full, correctly sized, plausible surface, so the verdict
 * is per element and the report says WHERE the damage is, because where it is names the
 * bound: rows past some index are the feature window, columns past some index a channel
 * bound, everything at once a word the closed forms got wrong.
 *
 * Each shape is run once; a surface that is not exact is run once more after a power
 * cycle, so a deterministic bound (the same both times) is told apart from a transient
 * one. The output is stamped with the sentinel before every run, so an unwritten element
 * is counted as unwritten rather than scored against whatever the buffer held.
 *
 * The fills are the gate's value sets, drawn as integers: A in 0.25 steps over
 * [-1.5, 1.5], B in 0.5 steps. Every product is a multiple of 1/8 and every partial sum
 * stays far under 2^21 in magnitude at K 8960, so fp32 holds each exactly and the order
 * of accumulation cannot move the answer. The reference is an integer dot product.
 *
 *   rk3576_mm_fp16_envelope MxKxN [MxKxN ...]
 *
 * One line per run. Exit 0 if every shape was exact on its first run, 1 otherwise, 2 with
 * no RK3576 device.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"
#include "test_fill.h"

#define STAMP 0xA5u

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* Rows or columns that hold a bad element: how many, the first and the last. */
struct span { unsigned count, lo, hi; };

static void span_add(struct span *s, unsigned char *seen, unsigned i)
{
    if (seen[i]) return;
    seen[i] = 1;
    if (!s->count || i < s->lo) s->lo = i;
    if (!s->count || i > s->hi) s->hi = i;
    s->count++;
}

static void span_print(const char *what, const struct span *s)
{
    if (!s->count) printf(" %s -", what);
    else printf(" %s %u in %u-%u", what, s->count, s->lo, s->hi);
}

static int run_shape(int fd, unsigned m, unsigned k, unsigned n)
{
    size_t ib = rocket_rk3576_mm_fp16_in_bytes(m, k);
    size_t wb = rocket_rk3576_mm_fp16_weight_bytes(k, n);
    size_t cb = rocket_rk3576_mm_fp16_coef_bytes(n);
    size_t ob = rocket_rk3576_mm_fp16_out_bytes(m, n);
    size_t ne = (size_t)m * n, e;
    rocket_bo in = {0}, w = {0}, coef = {0}, out = {0}, rc = {0};
    uint64_t ops[RK3576_MM_FP16_OPS];
    int8_t *ai = malloc((size_t)m * k), *bi = malloc((size_t)n * k);
    _Float16 *B = malloc((size_t)n * k * 2);
    int32_t *ref = malloc(ne * 4);
    unsigned char *rseen = malloc(2 * (size_t)m), *cseen = malloc(2 * (size_t)n);
    uint32_t in_h[4], out_h[1];
    unsigned attempt;
    int exact_first = 0, ret = 1;

    if (!ai || !bi || !B || !ref || !rseen || !cseen) {
        printf("%4u %5u %5u  out of memory\n", m, k, n);
        goto out;
    }
    for (e = 0; e < (size_t)m * k; e++) ai[e] = (int8_t)tf_int(0xA1, e, -6, 6);
    for (e = 0; e < (size_t)n * k; e++) {
        bi[e] = (int8_t)tf_int(0xB2, e, -3, 3);
        B[e] = (_Float16)(bi[e] * 0.5);
    }
    /* In units of 1/8: A is ai/4, B is bi/2. */
    for (unsigned r = 0; r < m; r++)
        for (unsigned c = 0; c < n; c++) {
            const int8_t *a = ai + (size_t)r * k, *b = bi + (size_t)c * k;
            int32_t acc = 0;
            for (unsigned i = 0; i < k; i++) acc += (int32_t)a[i] * b[i];
            ref[(size_t)r * n + c] = acc;
        }

    if (rocket_bo_alloc32(fd, ib, &in) || rocket_bo_alloc32(fd, wb, &w) ||
        rocket_bo_alloc32(fd, cb, &coef) || rocket_bo_alloc32(fd, ob, &out) ||
        rocket_bo_alloc32(fd, sizeof ops, &rc)) {
        printf("%4u %5u %5u  BO allocation failed (in %zu w %zu coef %zu out %zu)\n",
               m, k, n, ib, wb, cb, ob);
        goto out;
    }
    rocket_bo_prep(fd, &in, 1, 0);
    {
        _Float16 *A = (_Float16 *)in.ptr;
        for (e = 0; e < (size_t)m * k; e++) A[e] = (_Float16)(ai[e] * 0.25);
        memset((unsigned char *)in.ptr + (size_t)m * k * 2, 0, ib - (size_t)m * k * 2);
    }
    rocket_bo_fini(fd, &in);
    rocket_bo_prep(fd, &w, 1, 0);
    if (rocket_rk3576_mm_fp16_pack_weights(w.ptr, w.size, (const uint16_t *)B, k, n)) {
        rocket_bo_fini(fd, &w);
        printf("%4u %5u %5u  weight pack refused\n", m, k, n);
        goto out;
    }
    rocket_bo_fini(fd, &w);
    rocket_bo_prep(fd, &coef, 1, 0);
    rocket_rk3576_mm_fp16_pack_coef(coef.ptr, coef.size, n);
    rocket_bo_fini(fd, &coef);
    if (gen_matmul_fp16_rk3576_unchecked(ops, m, k, n, (uint32_t)in.dma_address,
                                         (uint32_t)w.dma_address, (uint32_t)out.dma_address,
                                         (uint32_t)coef.dma_address) != RK3576_MM_FP16_OPS) {
        printf("%4u %5u %5u  the generator refused\n", m, k, n);
        goto out;
    }
    rocket_bo_prep(fd, &rc, 1, 0);
    memcpy(rc.ptr, ops, sizeof ops);
    rocket_bo_fini(fd, &rc);
    in_h[0] = in.handle; in_h[1] = w.handle; in_h[2] = coef.handle; in_h[3] = rc.handle;
    out_h[0] = out.handle;

    for (attempt = 1; attempt <= 2; attempt++) {
        struct span ru = {0}, cu = {0}, rw = {0}, cw = {0};
        size_t unwritten = 0, wrong = 0, first = 0;
        const uint32_t *o;
        double t0, dev;
        int src;
        uint64_t slow0 = rocket_fence_wait_slow_count();

        rocket_bo_prep(fd, &out, 1, 0);
        memset(out.ptr, STAMP, ob);
        rocket_bo_fini(fd, &out);
        t0 = now_ms();
        src = rocket_submit_matmul(fd, &rc, RK3576_MM_FP16_OPS, in_h, 4, out_h, 1, 4000);
        if (src == 0 && rocket_bo_prep(fd, &out, 0, 4000000000ull) < 0) src = -1;
        dev = now_ms() - t0;
        if (src != 0) {
            printf("%4u %5u %5u  run %u: submit or wait failed (%d)\n", m, k, n, attempt, src);
            goto out;
        }
        o = (const uint32_t *)out.ptr;
        /* rseen/cseen hold two masks each: unwritten in the low half, wrong in the high. */
        memset(rseen, 0, 2 * (size_t)m); memset(cseen, 0, 2 * (size_t)n);
        for (e = 0; e < ne; e++) {
            unsigned r = (unsigned)(e / n), c = (unsigned)(e % n);
            float want = (float)ref[e] * 0.125f, got;
            if (o[e] == STAMP * 0x01010101u) {
                unwritten++;
                span_add(&ru, rseen, r);
                span_add(&cu, cseen, c);
                continue;
            }
            memcpy(&got, &o[e], 4);
            if (got != want) {
                if (!wrong++) first = e;
                span_add(&rw, rseen + m, r);
                span_add(&cw, cseen + n, c);
            }
        }
        printf("%4u %5u %5u  run %u %7.3f ms  %-9s unwritten %zu", m, k, n, attempt, dev,
               !unwritten && !wrong ? "EXACT" : unwritten ? "UNWRITTEN" : "WRONG",
               unwritten);
        span_print("rows", &ru); span_print("cols", &cu);
        printf("  wrong %zu", wrong);
        span_print("rows", &rw); span_print("cols", &cw);
        if (wrong) {
            float got; memcpy(&got, &o[first], 4);
            printf("  first m=%zu n=%zu got %.9g want %.9g", first / n, first % n,
                   (double)got, (double)ref[first] * 0.125);
        }
        if (rocket_fence_wait_slow_count() != slow0) printf("  SLOW-WAIT");
        printf("\n");
        rocket_bo_fini(fd, &out);
        fflush(stdout);
        if (!unwritten && !wrong) {
            if (attempt == 1) exact_first = 1;
            break;
        }
        rocket_rk3576_power_idle();
    }
    ret = !exact_first;
out:
    if (rc.ptr) rocket_bo_free(fd, &rc);
    if (out.ptr) rocket_bo_free(fd, &out);
    if (coef.ptr) rocket_bo_free(fd, &coef);
    if (w.ptr) rocket_bo_free(fd, &w);
    if (in.ptr) rocket_bo_free(fd, &in);
    free(ai); free(bi); free(B); free(ref); free(rseen); free(cseen);
    return ret;
}

int main(int argc, char **argv)
{
    int fd, a, fails = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s MxKxN [MxKxN ...]\n", argv[0]);
        return 1;
    }
    fd = rocket_open();
    if (fd < 0 || strcmp(rocket_hw_current()->name, "rk3576") != 0) {
        printf("no RK3576 device\n");
        return 2;
    }
    printf("   M     K     N  run     device  verdict\n");
    for (a = 1; a < argc; a++) {
        unsigned m, k, n;
        if (sscanf(argv[a], "%ux%ux%u", &m, &k, &n) != 3) {
            fprintf(stderr, "bad shape %s\n", argv[a]);
            return 1;
        }
        fails += run_shape(fd, m, k, n);
    }
    printf("%d of %d shapes exact on the first run\n", argc - 1 - fails, argc - 1);
    rocket_close(fd);
    return fails ? 1 : 0;
}
