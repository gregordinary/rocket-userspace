// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_batch_rocket.c — gate for the batched same-shape fp16 matmul
 * (rocket_matmul_fp16_batch): the attention dispatch-floor lever that runs nbatch
 * independent C[i]=A[i]·B[i]^T of one shared (M,K,N) as a SINGLE NPU job stream
 * (one submit + one fence wait for the whole group; one IRQ too under
 * ROCKET_BATCH_SUBMIT=1 + the kernel half).
 *
 * The contract is BIT-IDENTICAL output to calling rocket_matmul_fp16 per item — the
 * batch path is a submit-batching rearrangement, not a numeric change — so the oracle
 * is the per-item one-shot path itself and the bar is exact equality (max_abs == 0),
 * not a cosine band. The batched scratch and the per-item BOs differ only in how tiles
 * are grouped into jobs; the NPU computes each tile from identical regcmd + data.
 *
 * One catch makes that bit-identity CONDITIONAL on K-accum mode: the batch path is its
 * own CPU-fp32 K-accumulation sibling and never consults ROCKET_KACC, while the
 * default-on one-shot path accumulates the nKt K-partials on the NPU in fp16
 * (mm_compute_kacc). On nKt>1 shapes those two accumulation orders round differently —
 * ~1 fp16 ULP on a fraction of elements, cos still 1.0 — so the one-shot oracle is
 * bit-identical to the batch path only in the SAME fp32-accum mode. main() therefore
 * pins ROCKET_KACC=0 for the oracle; that selects mm_compute (host fp32 accum) and does
 * NOT touch the batch path, which ignores the knob (so the ROCKET_BATCH_SUBMIT chaining
 * variant is unaffected). Cosine is printed for context. Off-device: SKIP (exit 2).
 *
 * Shapes exercise: the QK case (single K-tile, many heads), an AV-at-longer-context
 * case (K>Kt so nKt>1 -> fp32 K-accum across job-batches), a case whose total tiles
 * exceed BATCH (multiple submits per group), and nbatch==1 (the one-shot fall-through).
 * Run ALSO with ROCKET_BATCH_SUBMIT=1 (+ kernel rocket_batch_submit=1) to gate the
 * chained-regcmd layout on HW.
 *
 * Usage: matmul_batch_rocket                       (sweep)
 *        matmul_batch_rocket M K N nbatch          (one shape)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int g_fail = 0;

static void fill(_Float16 *v, size_t n, float amp, uint32_t seed)
{
    uint32_t st = 0x9e3779b9u ^ seed;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)((st >> 8) & 0xffff) / 65535.f;
        v[i] = (_Float16)((u * 2.f - 1.f) * amp);
    }
}

/* One (M,K,N,nbatch) case: build nbatch random operand pairs, compute the per-item
 * one-shot result (the oracle) and the batched result, and require bit-identical. */
static int test_batch(int fd, int M, int K, int N, int nbatch)
{
    const size_t an = (size_t)M * K, bn = (size_t)N * K, cn = (size_t)M * N;
    _Float16 *Abuf = malloc((size_t)nbatch * an * sizeof(_Float16));
    _Float16 *Bbuf = malloc((size_t)nbatch * bn * sizeof(_Float16));
    _Float16 *Gbuf = malloc((size_t)nbatch * cn * sizeof(_Float16));   /* batched   */
    _Float16 *Rbuf = malloc((size_t)nbatch * cn * sizeof(_Float16));   /* per-item  */
    const _Float16 **A = malloc((size_t)nbatch * sizeof(*A));
    const _Float16 **B = malloc((size_t)nbatch * sizeof(*B));
    _Float16 **G = malloc((size_t)nbatch * sizeof(*G));
    if (!Abuf || !Bbuf || !Gbuf || !Rbuf || !A || !B || !G) {
        fprintf(stderr, "oom\n");
        free(Abuf); free(Bbuf); free(Gbuf); free(Rbuf); free(A); free(B); free(G);
        return 1;
    }
    for (int i = 0; i < nbatch; i++) {
        fill(Abuf + (size_t)i * an, an, 1.0f, (uint32_t)(M * 7 + K + i * 131));
        fill(Bbuf + (size_t)i * bn, bn, 1.0f, (uint32_t)(N * 5 + K + i * 977));
        A[i] = Abuf + (size_t)i * an;
        B[i] = Bbuf + (size_t)i * bn;
        G[i] = Gbuf + (size_t)i * cn;
    }

    /* oracle: each item through the validated one-shot path */
    int rc = 0;
    for (int i = 0; i < nbatch && !rc; i++)
        rc = rocket_matmul_fp16(fd, M, K, N, A[i], B[i], Rbuf + (size_t)i * cn);
    if (rc) { printf("  per-item ref M=%d K=%d N=%d n=%d -> FAIL (%d)\n", M, K, N, nbatch, rc);
              goto out_fail; }

    rc = rocket_matmul_fp16_batch(fd, M, K, N, nbatch, A, B, G);
    if (rc) { printf("  batch M=%d K=%d N=%d n=%d -> FAIL (%d)\n", M, K, N, nbatch, rc);
              goto out_fail; }

    /* bit-identical vs per-item is the contract; cosine is informational */
    double dot = 0, ng = 0, nr = 0, max_abs = 0;
    long nbad = 0;
    for (size_t i = 0; i < (size_t)nbatch * cn; i++) {
        double g = (double)Gbuf[i], r = (double)Rbuf[i];
        dot += g * r; ng += g * g; nr += r * r;
        double ad = fabs(g - r); if (ad > max_abs) max_abs = ad;
        if (Gbuf[i] != Rbuf[i]) nbad++;
    }
    double cos = dot / (sqrt(ng) * sqrt(nr) + 1e-30);
    int ok = (nbad == 0);   /* exact equality: the batch path must not perturb a bit */
    printf("  batch M=%d K=%d N=%d n=%d: cos=%.6f max_abs=%.4g nbad=%ld/%zu -> %s\n",
           M, K, N, nbatch, cos, max_abs, nbad, (size_t)nbatch * cn, ok ? "PASS" : "FAIL");
    free(Abuf); free(Bbuf); free(Gbuf); free(Rbuf); free(A); free(B); free(G);
    return ok ? 0 : 1;

out_fail:
    free(Abuf); free(Bbuf); free(Gbuf); free(Rbuf); free(A); free(B); free(G);
    return 1;
}

/* One (M,K,N,nbatch) case through a PERSISTENT context `b`: compute the per-item
 * one-shot oracle and the rocket_mm_batch_run result and require bit-identical. The
 * caller drives a SEQUENCE of these on one reused context to exercise the resident-BO
 * lifetime: a repeat shape (the skip-zero fast path), a larger shape (BO grow + re-zero),
 * and a shape change (layout re-zero) must all stay bit-exact. */
static int test_batch_ctx(int fd, rocket_mm_batch *b, int M, int K, int N, int nbatch)
{
    const size_t an = (size_t)M * K, bn = (size_t)N * K, cn = (size_t)M * N;
    _Float16 *Abuf = malloc((size_t)nbatch * an * sizeof(_Float16));
    _Float16 *Bbuf = malloc((size_t)nbatch * bn * sizeof(_Float16));
    _Float16 *Gbuf = malloc((size_t)nbatch * cn * sizeof(_Float16));   /* ctx batched */
    _Float16 *Rbuf = malloc((size_t)nbatch * cn * sizeof(_Float16));   /* per-item    */
    const _Float16 **A = malloc((size_t)nbatch * sizeof(*A));
    const _Float16 **B = malloc((size_t)nbatch * sizeof(*B));
    _Float16 **G = malloc((size_t)nbatch * sizeof(*G));
    if (!Abuf || !Bbuf || !Gbuf || !Rbuf || !A || !B || !G) {
        fprintf(stderr, "oom\n");
        free(Abuf); free(Bbuf); free(Gbuf); free(Rbuf); free(A); free(B); free(G);
        return 1;
    }
    for (int i = 0; i < nbatch; i++) {
        fill(Abuf + (size_t)i * an, an, 1.0f, (uint32_t)(M * 7 + K + i * 131 + N));
        fill(Bbuf + (size_t)i * bn, bn, 1.0f, (uint32_t)(N * 5 + K + i * 977 + M));
        A[i] = Abuf + (size_t)i * an;
        B[i] = Bbuf + (size_t)i * bn;
        G[i] = Gbuf + (size_t)i * cn;
    }
    int rc = 0;
    for (int i = 0; i < nbatch && !rc; i++)
        rc = rocket_matmul_fp16(fd, M, K, N, A[i], B[i], Rbuf + (size_t)i * cn);
    if (!rc) rc = rocket_mm_batch_run(b, M, K, N, nbatch, A, B, G);
    if (rc) { printf("  ctx M=%d K=%d N=%d n=%d -> FAIL (%d)\n", M, K, N, nbatch, rc);
              free(Abuf); free(Bbuf); free(Gbuf); free(Rbuf); free(A); free(B); free(G);
              return 1; }

    long nbad = 0; double max_abs = 0;
    for (size_t i = 0; i < (size_t)nbatch * cn; i++) {
        double ad = fabs((double)Gbuf[i] - (double)Rbuf[i]); if (ad > max_abs) max_abs = ad;
        if (Gbuf[i] != Rbuf[i]) nbad++;
    }
    int ok = (nbad == 0);
    printf("  ctx M=%d K=%d N=%d n=%d: max_abs=%.4g nbad=%ld/%zu -> %s\n",
           M, K, N, nbatch, max_abs, nbad, (size_t)nbatch * cn, ok ? "PASS" : "FAIL");
    free(Abuf); free(Bbuf); free(Gbuf); free(Rbuf); free(A); free(B); free(G);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    /* Pin the per-item oracle to host fp32 K-accum (mm_compute) so it matches the batch
     * path's fp32 accumulation bit-for-bit. The default-on fp16 NPU K-accum
     * (mm_compute_kacc) rounds each nKt partial in fp16 and legitimately diverges from a
     * single fp32->fp16 narrow by ~1 ULP on multi-K-tile shapes (cos=1.0) — a different
     * rounding order, not a corruption. The batch path never reads this knob, so pinning
     * it changes only the oracle, not the path under test. Set before the first matmul
     * (kacc_on() caches on first read). */
    setenv("ROCKET_KACC", "0", 1);

    int fd = rocket_open();
    if (fd < 0) { printf("note: no /dev/accel/accel0 (%d) — SKIP\n", fd); return 2; }

    if (argc == 5) {
        g_fail |= test_batch(fd, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
        rocket_close(fd);
        printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
        return g_fail ? 1 : 0;
    }

    /* QK-like: head_dim contraction (single K-tile), many heads in a worker's range. */
    g_fail |= test_batch(fd, 128, 256, 128, 8);   /* Tp=128, dh=256, n_kv=128, 8 heads */
    g_fail |= test_batch(fd, 512, 256, 512, 4);   /* multi M/N tile, 4 heads           */
    g_fail |= test_batch(fd, 512, 256, 512, 1);   /* nbatch==1 -> one-shot fall-through */
    /* AV-at-longer-context: K=n_kv > Kt forces nKt>1 -> fp32 K-accum across batches.  */
    g_fail |= test_batch(fd, 512, 1024, 256, 3);  /* AV-like: K=1024 (nKt>1), 3 heads   */
    g_fail |= test_batch(fd, 128, 2048, 256, 6);  /* deeper K-accum, 6 heads            */
    /* total tiles > BATCH(64): force multiple submits within one group.               */
    g_fail |= test_batch(fd, 512, 256, 2048, 8);  /* 16 tiles/item x 8 = 128 -> 2 jobs  */
    /* unaligned-ish edge tiles (N tail), GQA-ish odd head count.                       */
    g_fail |= test_batch(fd, 256, 256, 528, 5);   /* N=528 -> last N-tile is 16 wide    */

    /* persistent context: one reused rocket_mm_batch across a shape sequence that
     * exercises the resident-BO lifetime — the SAME shape twice (the skip-zero fast
     * path), a larger shape (BO grow + re-zero), back to the small shape (re-zero), an
     * AV-like shape (multi-K-tile), and that AV shape again (skip-zero). Every run must
     * stay bit-identical to the per-item oracle, so the skip-zero never leaks stale
     * padding. */
    rocket_mm_batch *b = rocket_mm_batch_create(fd);
    if (!b) { printf("  ctx: create FAILED\n"); g_fail = 1; }
    else {
        g_fail |= test_batch_ctx(fd, b, 128, 256, 128, 8);   /* first: grow + zero       */
        g_fail |= test_batch_ctx(fd, b, 128, 256, 128, 8);   /* repeat: SKIP-zero        */
        g_fail |= test_batch_ctx(fd, b, 512, 256, 512, 4);   /* larger: grow + re-zero   */
        g_fail |= test_batch_ctx(fd, b, 128, 256, 128, 8);   /* shrink-reuse: re-zero    */
        g_fail |= test_batch_ctx(fd, b, 512, 1024, 256, 3);  /* AV-like multi-K: re-zero */
        g_fail |= test_batch_ctx(fd, b, 512, 1024, 256, 3);  /* repeat AV: SKIP-zero     */
        rocket_mm_batch_free(b);
    }

    rocket_close(fd);
    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
