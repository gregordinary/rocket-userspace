// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * kacc_int32_probe.c -- can the RK3588 DPU add an int32 K-partial to a convolution's int32
 * result in integer arithmetic, with the conv's CACC result as the EW stage's main feed?
 *
 * ew_int_probe shows the EW ALU adds int32 exactly under the whole integer precision bundle,
 * but on a DPU-only program (flying MRDMA main feed). An integer K-accumulation needs the
 * other main feed: the CACC result of an int8 convolution, with the running int32 partial on
 * the ERDMA. gen_gemm_int8's accumulate branch emits that program with DPU 0x4010 at
 * out int32 / in int8 / proc int8 and the RDMA FEATURE_MODE_CFG at int32. This probe takes
 * that program and rewrites the precision triple in DPU 0x4010 (and optionally the EW input
 * converter bypass, DPU_EW_CFG bit 8) per variant, so the library itself is unchanged.
 *
 * Modes:
 *   kacc_int32_probe single [M K N]       one accumulate job over the whole K, the operand a
 *                                         CONSTANT int32 past 2^24 with odd low bits (layout
 *                                         invariant). Each variant is scored against:
 *                                           int   out == ref + C        (integer add)
 *                                           flt   out == (int)((float)ref + (float)C)
 *                                           ref   out == ref           (operand read as 0)
 *                                           cop   out == C             (conv dropped)
 *                                           sent  out == sentinel      (nothing written)
 *   kacc_int32_probe chain [M K N nKt]    the real case for the variants that pass `single`:
 *                                         K split in nKt tiles, ping-pong partial buffers,
 *                                         bit-exact against the int64 sum truncated to int32.
 *                                         Two data sets: full-range random int8, and a
 *                                         correlated one whose sums reach ~6e7 (past 2^24).
 *   kacc_int32_probe time [M K N reps]    device-inclusive time of one int32-out conv against
 *                                         the same conv with the EW add armed (best variant).
 * Variants (ROCKET_KI32_VARIANTS=a,b,... to choose; default all):
 *   g      the program as generated: 0x4010 out 4 / in 0 / proc 0, EW converter live
 *   gb     as g, EW converter bypassed
 *   p4     proc 4 (int32), in 0
 *   p4b    proc 4, EW converter bypassed
 *   a4     out/in/proc all 4
 *   a4b    all 4, EW converter bypassed (the DPU-only probe's integer bundle)
 * The EW surface stride is set to M (the int8 generator floors it at 12, erratum E12), and M
 * is kept >= 12 by default regardless. A failing variant may time out, and the driver retires
 * it at its deadline. Run each variant in its own process:
 * ROCKET_KI32_VARIANTS=a4b kacc_int32_probe single.
 * Caution: a timeout whose DMA is still running breaks the core's IOMMU, and the board then
 * needs a power cycle. ROCKET_KI32_FMC=0x27890 with ROCKET_KI32_SRC0=1 (the main feed's
 * RDMA disabled) did that on the RK3588 [HW 2026-11-07]: the reset logged "Enable stall
 * request timed out", later jobs on that core failed about half the time, and the core then
 * stuck resuming. Do not disable the main feed's RDMA on this conv-fed program, and stop
 * submitting at the first IOMMU stall warning in the kernel log.
 * Exit: 0 if any variant is exact in `single`/`chain`, 1 otherwise, 2 with no device.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "npu_matmul.h"
#include "rocket_npu.h"

static inline int in_idx(int K, int M, int c, int h)  { return feature_index(K, M, 1, 16, c, h, 1); }
static inline int out_idx(int N, int M, int n, int h) { return feature_index(N, M, 1, 4,  n, h, 1); }

typedef struct { const char *name; int out, in, proc, cvtbyp; } variant_t;
static const variant_t VARS[] = {
    {"g",   4, 0, 0, 0}, {"gb",  4, 0, 0, 1},
    {"p4",  4, 0, 4, 0}, {"p4b", 4, 0, 4, 1},
    {"a4",  4, 4, 4, 0}, {"a4b", 4, 4, 4, 1},
};
#define NVARS ((int)(sizeof(VARS) / sizeof(VARS[0])))

static int variant_on(const char *name)
{
    const char *e = getenv("ROCKET_KI32_VARIANTS");
    if (!e || !*e) return 1;
    char buf[256];
    snprintf(buf, sizeof buf, ",%s,", e);
    char key[32];
    snprintf(key, sizeof key, ",%s,", name);
    return strstr(buf, key) != NULL;
}

static void patch(uint64_t *ops, uint32_t n, uint16_t reg, uint32_t val)
{
    for (uint32_t z = 0; z < n; z++)
        if ((ops[z] & 0xFFFF) == reg)
            ops[z] = (ops[z] & ~(0xFFFFFFFFull << 16)) | ((uint64_t)val << 16);
}
static uint32_t get(const uint64_t *ops, uint32_t n, uint16_t reg)
{
    for (uint32_t z = 0; z < n; z++)
        if ((ops[z] & 0xFFFF) == reg) return (uint32_t)((ops[z] >> 16) & 0xFFFFFFFF);
    return 0xDEADBEEF;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* One int8 conv tile, out[M,N] (+)= A[M,Kt] B[N,Kt]^T, int32 out. v == NULL: the plain
 * program. add_bo must differ from out. */
static int run_tile(int fd, int M, int Kt, int N, rocket_bo *in, rocket_bo *wt, rocket_bo *out,
                    rocket_bo *rc, const variant_t *v, rocket_bo *add_bo, int verbose)
{
    uint64_t ops[256] = {0};
    gemm_params_t p = {
        .m = (uint16_t)M, .k = (uint16_t)Kt, .n = (uint16_t)N,
        .input_iova = (uint32_t)in->dma_address,
        .weights_iova = (uint32_t)wt->dma_address,
        .output_iova = (uint32_t)out->dma_address,
        .tasks = ops,
        .accumulate = v ? 1 : 0,
        .add_iova = v ? (uint32_t)add_bo->dma_address : 0,
    };
    if (gen_gemm_int8(&p)) { fprintf(stderr, "gen_gemm_int8 failed\n"); return -1; }
    if (v) {
        uint32_t fmt = get(ops, p.task_count, 0x4010);
        fmt &= ~((7u << 29) | (7u << 26) | 7u);
        fmt |= ((uint32_t)v->out << 29) | ((uint32_t)v->in << 26) | (uint32_t)v->proc;
        patch(ops, p.task_count, 0x4010, fmt);
        uint32_t ew = get(ops, p.task_count, 0x4070);
        ew = v->cvtbyp ? (ew | (1u << 8)) : (ew & ~(1u << 8));
        patch(ops, p.task_count, 0x4070, ew);
        /* Optional raw overrides for a sweep: ROCKET_KI32_FMC (DPU_RDMA FEATURE_MODE_CFG),
         * ROCKET_KI32_EWCFG (DPU_EW_CFG, replaces the variant's), ROCKET_KI32_ERDMA,
         * ROCKET_KI32_SRC0=1 (MRDMA SRC_BASE 0), ROCKET_KI32_EWSCALE (EW_CVT_SCALE_VALUE). */
        const char *e;
        if ((e = getenv("ROCKET_KI32_FMC")))   patch(ops, p.task_count, 0x5044, (uint32_t)strtoul(e, NULL, 0));
        if ((e = getenv("ROCKET_KI32_EWCFG"))) patch(ops, p.task_count, 0x4070, (uint32_t)strtoul(e, NULL, 0));
        if ((e = getenv("ROCKET_KI32_ERDMA"))) patch(ops, p.task_count, 0x5034, (uint32_t)strtoul(e, NULL, 0));
        if ((e = getenv("ROCKET_KI32_EWSCALE"))) patch(ops, p.task_count, 0x4078, (uint32_t)strtoul(e, NULL, 0));
        if ((e = getenv("ROCKET_KI32_SRC0")) && atoi(e)) patch(ops, p.task_count, 0x5018, 0);
        /* EW surface stride and notch at M (the generator floors at MAX(M,12)) */
        patch(ops, p.task_count, 0x5040, (uint32_t)M << 4);
        patch(ops, p.task_count, 0x506C, (uint32_t)M << 4);
        patch(ops, p.task_count, 0x504C, (uint32_t)M << 4);   /* RDMA SURF_NOTCH */
        if (verbose)
            fprintf(stderr, "  [%s] 0x4010=0x%08x EW_CFG=0x%08x ERDMA=0x%08x FMC=0x%08x "
                    "EW_STRIDE=0x%08x\n", v->name, get(ops, p.task_count, 0x4010),
                    get(ops, p.task_count, 0x4070), get(ops, p.task_count, 0x5034),
                    get(ops, p.task_count, 0x5044), get(ops, p.task_count, 0x5040));
    }
    rocket_bo_prep(fd, rc, 1, 0);
    memcpy(rc->ptr, ops, (size_t)p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, rc);
    rocket_task_desc t = { (uint32_t)rc->dma_address, p.task_count };
    uint32_t in_acc[] = { in->handle, wt->handle, rc->handle, v ? add_bo->handle : 0 };
    uint32_t out_h[] = { out->handle };
    int r = rocket_submit_tasks(fd, &t, 1, in_acc, v ? 4 : 3, out_h, 1);
    if (r) { fprintf(stderr, "submit failed (%d)\n", r); return -1; }
    r = rocket_bo_prep(fd, out, 0, 2000000000LL);
    if (r) { fprintf(stderr, "  job did not complete (%d)\n", r); return -2; }
    rocket_bo_fini(fd, out);
    return 0;
}

static void pack_in(int fd, rocket_bo *bo, const int8_t *A, int M, int K, int k0, int Kt)
{
    rocket_bo_prep(fd, bo, 1, 0);
    memset(bo->ptr, 0, bo->size);
    int8_t *ip = bo->ptr;
    for (int h = 1; h <= M; h++)
        for (int c = 1; c <= Kt; c++) ip[in_idx(Kt, M, c, h)] = A[(size_t)(h - 1) * K + k0 + c - 1];
    rocket_bo_fini(fd, bo);
}
static void pack_wt(int fd, rocket_bo *bo, const int8_t *B, int N, int K, int k0, int Kt)
{
    rocket_bo_prep(fd, bo, 1, 0);
    memset(bo->ptr, 0, bo->size);
    int8_t *wp = bo->ptr;
    for (int kk = 1; kk <= N; kk++)
        for (int c = 1; c <= Kt; c++) wp[gemm_weight_index_int8(Kt, kk, c)] = B[(size_t)(kk - 1) * K + k0 + c - 1];
    rocket_bo_fini(fd, bo);
}
static void fill_out(int fd, rocket_bo *bo, int32_t v)
{
    rocket_bo_prep(fd, bo, 1, 0);
    for (size_t z = 0; z < bo->size / 4; z++) ((int32_t *)bo->ptr)[z] = v;
    rocket_bo_fini(fd, bo);
}

#define IN_SZ(M, Kt) ((size_t)((M + 3) / 4 * 4) * ((Kt + 31) / 32 * 32) + 4096)
#define WT_SZ(N, Kt) ((size_t)((N + 31) / 32 * 32) * ((Kt + 31) / 32 * 32) + 4096)
#define OUT_SZ(M, N) ((size_t)((M + 3) / 4 * 4) * ((N + 31) / 32 * 32) * 4 + 4096)

static uint32_t rng = 12345;
static int8_t rnd8(void) { rng = rng * 1103515245u + 12345u; return (int8_t)(rng >> 16); }

static void ref_mm(const int8_t *A, const int8_t *B, int32_t *ref, int M, int K, int N)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64_t a = 0;
            for (int k = 0; k < K; k++) a += (int32_t)A[(size_t)m * K + k] * (int32_t)B[(size_t)n * K + k];
            ref[(size_t)m * N + n] = (int32_t)(uint32_t)(uint64_t)a;
        }
}

static int mode_single(int fd, int M, int K, int N)
{
    const int32_t COP = 0x01234567, SENT = 0x55AA55AA;
    const char *om = getenv("ROCKET_KI32_OPMODE");
    const int ramp = om && !strcmp(om, "ramp");
    int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K);
    int32_t *ref = malloc((size_t)M * N * 4);
    /* correlated data: sums ~4e7, past 2^24, low bits varied */
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)(112 + (rnd8() & 15));
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = (int8_t)(112 + (rnd8() & 15));
    ref_mm(A, B, ref, M, K, N);
    rocket_bo in = {0}, wt = {0}, out = {0}, op = {0}, rc = {0};
    if (rocket_bo_alloc(fd, IN_SZ(M, K), &in) || rocket_bo_alloc(fd, WT_SZ(N, K), &wt) ||
        rocket_bo_alloc(fd, OUT_SZ(M, N), &out) || rocket_bo_alloc(fd, OUT_SZ(M, N), &op) ||
        rocket_bo_alloc(fd, 256 * 8, &rc)) { fprintf(stderr, "alloc failed\n"); return 2; }
    pack_in(fd, &in, A, M, K, 0, K);
    pack_wt(fd, &wt, B, N, K, 0, K);
    printf("single: M=%d K=%d N=%d, operand const %d, ref[0]=%d\n", M, K, N, COP, ref[0]);
    /* control: the plain int32-out program */
    int exact_plain = 0;
    int creps = getenv("ROCKET_KI32_CTRLREPS") ? atoi(getenv("ROCKET_KI32_CTRLREPS")) : 1;
    for (int cr = 0; cr < creps; cr++) {
    fill_out(fd, &out, SENT);
    if (run_tile(fd, M, K, N, &in, &wt, &out, &rc, NULL, NULL, 0) == 0) {
        rocket_bo_prep(fd, &out, 0, 0);
        const int32_t *o = out.ptr;
        size_t ok = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) ok += o[out_idx(N, M, n + 1, m + 1)] == ref[(size_t)m * N + n];
        rocket_bo_fini(fd, &out);
        exact_plain = ok == (size_t)M * N;
        printf("  control (no EW): %zu/%d exact\n", ok, M * N);
    }
    }
    int any = 0;
    for (int vi = 0; vi < NVARS; vi++) {
        const variant_t *v = &VARS[vi];
        if (!variant_on(v->name)) continue;
        fill_out(fd, &op, COP);
        if (ramp) {   /* distinct 32-bit words, so each model below names one element */
            rocket_bo_prep(fd, &op, 1, 0);
            for (size_t z = 0; z < op.size / 4; z++)
                ((uint32_t *)op.ptr)[z] = (uint32_t)(z * 2654435761u) ^ 0x00010001u;
            rocket_bo_fini(fd, &op);
        }
        fill_out(fd, &out, SENT);
        int r = run_tile(fd, M, K, N, &in, &wt, &out, &rc, v, &op, 1);
        if (r) { printf("  %-4s no completion (%d)\n", v->name, r); continue; }
        rocket_bo_prep(fd, &out, 0, 0);
        const int32_t *o = out.ptr;
        size_t n_int = 0, n_flt = 0, n_ref = 0, n_cop = 0, n_sent = 0, n_h8 = 0, n_hx = 0, n_lo = 0;
        int shown = 0;
        rocket_bo_prep(fd, &op, 0, 0);
        const uint32_t *ow = op.ptr;
        const int16_t *oh = op.ptr;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int xi = out_idx(N, M, n + 1, m + 1);
                int32_t g = o[xi], rr = ref[(size_t)m * N + n];
                int32_t opw = ramp ? (int32_t)ow[xi] : COP;
                int32_t ei = (int32_t)((uint32_t)rr + (uint32_t)opw);
                int32_t ef = (int32_t)llrintf((float)rr + (float)opw);
                /* int16 readings of the operand: the element's own index in an int16 cube
                 * (C2 8), the int32 element's index used as a halfword index, its low half */
                n_h8 += g == rr + oh[feature_index(N, M, 1, 8, n + 1, m + 1, 1)];
                n_hx += g == rr + oh[xi];
                n_lo += g == rr + (int16_t)(opw & 0xFFFF);
                n_int += g == ei; n_flt += g == ef; n_ref += g == rr; n_cop += g == opw;
                n_sent += g == SENT;
                if (g != ei && shown < 3) {
                    printf("    [%d,%d] got %d (0x%08x) int %d flt %d ref %d\n", m, n, g,
                           (uint32_t)g, ei, ef, rr);
                    shown++;
                }
            }
        if (ramp && getenv("ROCKET_KI32_FIND")) {
            /* where did each sampled element's addend come from? Search the operand as int32
             * words and as int16 halfwords; print the index, or '-' if absent. */
            size_t nw = op.size / 4;
            for (int m = 0; m < M; m += (m < 3 ? 1 : 29))
                for (int n = 0; n < N; n += (n < 9 ? 1 : 13)) {
                    int xi = out_idx(N, M, n + 1, m + 1);
                    int32_t d = (int32_t)((uint32_t)o[xi] - (uint32_t)ref[(size_t)m * N + n]);
                    long wi = -1, hi = -1;
                    for (size_t z = 0; z < nw && wi < 0; z++) if ((int32_t)ow[z] == d) wi = (long)z;
                    for (size_t z = 0; z < 2 * nw && hi < 0; z++) if (oh[z] == d) hi = (long)z;
                    printf("    m %2d n %2d out_idx %5d  addend %11d  word %6ld  half %6ld  h8cube %6d\n",
                           m, n, xi, d, wi, hi, feature_index(N, M, 1, 8, n + 1, m + 1, 1));
                }
        }
        rocket_bo_fini(fd, &op);
        rocket_bo_fini(fd, &out);
        int tot = M * N;
        printf("  %-4s int %zu flt %zu ref %zu cop %zu sent %zu h16cube %zu h16idx %zu lo16 %zu of %d -> %s\n",
               v->name, n_int, n_flt, n_ref, n_cop, n_sent, n_h8, n_hx, n_lo, tot,
               n_int == (size_t)tot ? "INTEGER ADD EXACT" :
               n_flt == (size_t)tot ? "float add" : n_ref == (size_t)tot ? "operand read as 0" :
               n_cop == (size_t)tot ? "conv dropped" : n_sent == (size_t)tot ? "nothing written" :
               "other");
        any |= n_int == (size_t)tot;
    }
    rocket_bo_free(fd, &in); rocket_bo_free(fd, &wt); rocket_bo_free(fd, &out);
    rocket_bo_free(fd, &op); rocket_bo_free(fd, &rc);
    free(A); free(B); free(ref);
    return any && exact_plain ? 0 : 1;
}

static int chain_one(int fd, const variant_t *v, int M, int K, int N, int nKt, int correlated)
{
    int Kt = K / nKt;
    int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K);
    int32_t *ref = malloc((size_t)M * N * 4);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = correlated ? (int8_t)(112 + (rnd8() & 15)) : rnd8();
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = correlated ? (int8_t)(112 + (rnd8() & 15)) : rnd8();
    ref_mm(A, B, ref, M, K, N);
    rocket_bo *ins = calloc(nKt, sizeof(rocket_bo)), *wts = calloc(nKt, sizeof(rocket_bo));
    rocket_bo bA = {0}, bB = {0}, rc = {0};
    int e = 0;
    for (int t = 0; t < nKt; t++) {
        e |= rocket_bo_alloc(fd, IN_SZ(M, Kt), &ins[t]);
        e |= rocket_bo_alloc(fd, WT_SZ(N, Kt), &wts[t]);
    }
    e |= rocket_bo_alloc(fd, OUT_SZ(M, N), &bA) | rocket_bo_alloc(fd, OUT_SZ(M, N), &bB) |
         rocket_bo_alloc(fd, 256 * 8, &rc);
    if (e) { fprintf(stderr, "alloc failed\n"); return -1; }
    for (int t = 0; t < nKt; t++) {
        pack_in(fd, &ins[t], A, M, K, t * Kt, Kt);
        pack_wt(fd, &wts[t], B, N, K, t * Kt, Kt);
    }
    fill_out(fd, &bA, 0x55AA55AA);
    fill_out(fd, &bB, 0x55AA55AA);
    rocket_bo *bufs[2] = { &bA, &bB };
    int r = run_tile(fd, M, Kt, N, &ins[0], &wts[0], bufs[0], &rc, NULL, NULL, 0);
    for (int t = 1; t < nKt && !r; t++)
        r = run_tile(fd, M, Kt, N, &ins[t], &wts[t], bufs[t & 1], &rc, v, bufs[(t - 1) & 1], 0);
    size_t bad = 0;
    long long maxabs = 0, refmax = 0;
    if (!r) {
        rocket_bo *fin = bufs[(nKt - 1) & 1];
        rocket_bo_prep(fd, fin, 0, 0);
        const int32_t *o = fin->ptr;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int32_t g = o[out_idx(N, M, n + 1, m + 1)], rr = ref[(size_t)m * N + n];
                long long d = llabs((long long)g - rr);
                if (llabs(rr) > refmax) refmax = llabs(rr);
                if (d) { bad++; if (d > maxabs) maxabs = d; }
            }
        rocket_bo_fini(fd, fin);
        printf("  %-4s chain M=%d K=%d N=%d nKt=%d %s: %zu of %d wrong, max |err| %lld, max |ref| %lld"
               " -> %s\n", v->name, M, K, N, nKt, correlated ? "corr" : "rand", bad, M * N, maxabs,
               refmax, bad ? "FAIL" : "EXACT");
    } else {
        printf("  %-4s chain M=%d K=%d N=%d nKt=%d: no completion\n", v->name, M, K, N, nKt);
    }
    for (int t = 0; t < nKt; t++) { rocket_bo_free(fd, &ins[t]); rocket_bo_free(fd, &wts[t]); }
    rocket_bo_free(fd, &bA); rocket_bo_free(fd, &bB); rocket_bo_free(fd, &rc);
    free(ins); free(wts); free(A); free(B); free(ref);
    return r ? -1 : (int)(bad != 0);
}

static int mode_time(int fd, const variant_t *v, int M, int K, int N, int reps)
{
    int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rnd8();
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = rnd8();
    rocket_bo in = {0}, wt = {0}, out = {0}, op = {0}, rc = {0};
    if (rocket_bo_alloc(fd, IN_SZ(M, K), &in) || rocket_bo_alloc(fd, WT_SZ(N, K), &wt) ||
        rocket_bo_alloc(fd, OUT_SZ(M, N), &out) || rocket_bo_alloc(fd, OUT_SZ(M, N), &op) ||
        rocket_bo_alloc(fd, 256 * 8, &rc)) return 2;
    pack_in(fd, &in, A, M, K, 0, K);
    pack_wt(fd, &wt, B, N, K, 0, K);
    double best[2] = {1e9, 1e9};
    for (int rep = 0; rep < reps; rep++)
        for (int arm = 0; arm < 2; arm++) {
            double t0 = now_s();
            if (run_tile(fd, M, K, N, &in, &wt, &out, &rc, arm ? v : NULL, &op, 0)) return 1;
            double dt = now_s() - t0;
            if (dt < best[arm]) best[arm] = dt;
        }
    printf("time M=%d K=%d N=%d: plain %.1f us, with int32 EW add (%s) %.1f us, ratio %.3f "
           "(best of %d, submit to fence, one task)\n", M, K, N, best[0] * 1e6, v->name,
           best[1] * 1e6, best[1] / best[0], reps);
    rocket_bo_free(fd, &in); rocket_bo_free(fd, &wt); rocket_bo_free(fd, &out);
    rocket_bo_free(fd, &op); rocket_bo_free(fd, &rc);
    free(A); free(B);
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "single";
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no device (%d)\n", fd); return 2; }
    int rc = 1;
    if (!strcmp(mode, "single")) {
        int M = argc > 2 ? atoi(argv[2]) : 64, K = argc > 3 ? atoi(argv[3]) : 2048,
            N = argc > 4 ? atoi(argv[4]) : 64;
        rc = mode_single(fd, M, K, N);
    } else if (!strcmp(mode, "chain")) {
        int M = argc > 2 ? atoi(argv[2]) : 64, K = argc > 3 ? atoi(argv[3]) : 4096,
            N = argc > 4 ? atoi(argv[4]) : 64, nKt = argc > 5 ? atoi(argv[5]) : 2;
        if (K % nKt || (K / nKt) % 32 || N % 32 || M % 4) {
            fprintf(stderr, "chain: need K%%nKt==0, (K/nKt)%%32==0, N%%32==0, M%%4==0\n");
            return 2;
        }
        int any = 0;
        for (int vi = 0; vi < NVARS; vi++) {
            if (!variant_on(VARS[vi].name)) continue;
            int a = chain_one(fd, &VARS[vi], M, K, N, nKt, 0);
            int b = chain_one(fd, &VARS[vi], M, K, N, nKt, 1);
            any |= (a == 0 && b == 0);
        }
        rc = any ? 0 : 1;
    } else if (!strcmp(mode, "time")) {
        int M = argc > 2 ? atoi(argv[2]) : 256, K = argc > 3 ? atoi(argv[3]) : 2048,
            N = argc > 4 ? atoi(argv[4]) : 256, reps = argc > 5 ? atoi(argv[5]) : 50;
        const char *vn = getenv("ROCKET_KI32_VARIANTS");
        const variant_t *v = &VARS[NVARS - 1];
        for (int vi = 0; vn && vi < NVARS; vi++)
            if (!strcmp(vn, VARS[vi].name)) v = &VARS[vi];
        rc = mode_time(fd, v, M, K, N, reps);
    } else {
        fprintf(stderr, "usage: %s single|chain|time [M K N [nKt|reps]]\n", argv[0]);
        rc = 2;
    }
    rocket_close(fd);
    return rc;
}
