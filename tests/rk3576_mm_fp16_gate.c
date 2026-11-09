// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_mm_fp16_gate.c — the RK3576 matmul-form fp16 program, against its form and a
 * CPU reference.
 *
 * The program is Rockchip's compiled form (npu_regcmd_rk3576.c), and
 * rk3576_mm_fp16_vendor_gate holds it word for word to the compiler's programs in a
 * .rkllm. This gate has two halves that answer two other questions.
 *
 * THE HOST HALF asks whether every emitted stream has the compiled program's form. At
 * every shape in rk3576_mm_fp16_shapes.h it checks:
 *   - the stream: CNA, CORE and DPU words only, then the PC trailer that enables CNA,
 *     CORE and DPU; the eight words of the fp32 output stage; the fp16 operand words;
 *     the three buffer addresses where they were asked for;
 *   - the weight permutation, element by element, against the documented native layout,
 *     over a buffer whose every (n, k) holds a 16-bit value hashed from its own index.
 * It also holds the envelope: every shape is inside it, a shape one step past each bound
 * is refused, and one task is admitted at the last row inside the CBUF data window and
 * refused at the first past it, at three K. It needs no device, and it fails the run on
 * any part.
 *
 * THE DEVICE HALF asks whether the part computes it. Each shape runs through the public
 * entry, rocket_matmul_fp16_rk3576(), and every output is compared EXACTLY against the
 * CPU. The table's shapes are one task each; a second table runs shapes the entry cuts
 * into 2 to 25 tasks along M, and each row prints how many tasks it submitted. The fills are the first transcription run's value sets, drawn without a period: A in 0.25 steps over
 * [-1.5, 1.5] and B in 0.5 steps over [-1.5, 1.5]. Every product is a multiple of 1/8
 * and every partial sum stays under 2^15 of those units at K=2048, so fp32 holds each
 * one exactly and the order of accumulation cannot move the answer. A mismatch of any
 * size is a wrong result, and there is no tolerance.
 *
 * THE CONTROL is what says the comparison can fail. One shape runs the same program
 * with the weights packed row-major instead of in the part's layout, through a raw
 * submit, and must come back WRITTEN and WRONG. The first run's DENSE control was wrong on 64
 * of 64 outputs. A control that comes back exact fails the gate: the check could not
 * have seen a layout error.
 *
 * THE RESIDENT HANDLE is held to the per-call entry. At every shape the same weight goes
 * into rocket_rk3576_wbo_fp16_create() and the result of rocket_matmul_fp16_rk3576_wbo()
 * must equal the per-call surface bit for bit, as well as the CPU. One more row reuses a
 * single handle at K 256, N 64 at M from 1 to 1500, one to three tasks, since the handle
 * carries no M, and one checks the refusals: a K or N outside the envelope at create, a
 * call whose K or N differs from the handle's, and an M past the entry's bound.
 *
 * EACH ROW PRINTS ITS SUBMIT COUNT. The entry redoes a job that left outputs unwritten,
 * so a row can pass on its second submit; that is a poisoned job, and the count is the
 * only place it shows.
 *
 * Exit 0 on a pass, 1 on any failure, 2 when the host half passed and no RK3576 device
 * opened: a host pass is not a device pass.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"
#include "test_fill.h"

/* Placeholder addresses for the host half's streams. */
#define RK3576_MM_FP16_IN  0x11000000u
#define RK3576_MM_FP16_W   0x33000000u
#define RK3576_MM_FP16_OUT 0x22000000u

static const unsigned SHAPES[][3] = {
#include "rk3576_mm_fp16_shapes.h"
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof SHAPES[0]))

/* Past one task, through both entries, which cut M into tasks of at most the data
 * window's rows: two tasks at K 2048, uneven ones at N 80, two at N 1024, seven at an LLM's
 * K 1536, fifteen of 20 rows at K 6144, and the entry's M bound in 25. */
static const unsigned TILED[][3] = {
    { 65, 2048, 64 }, { 100, 2048, 80 }, { 513, 256, 1024 }, { 512, 1536, 256 },
    { 300, 6144, 64 }, { 2048, 1536, 256 },
};
#define N_TILED ((int)(sizeof TILED / sizeof TILED[0]))

static uint16_t wval(unsigned n, unsigned k)
{
    uint32_t z = n * 0x9E3779B1u ^ (k + 0x7F4A7C15u) * 0x85EBCA77u;
    z ^= z >> 15; z *= 0x2C1B3C6Du; z ^= z >> 12;
    return (uint16_t)z;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* The value the stream writes to one register of one block, or -1 when it writes none. */
static int64_t stream_val(const uint64_t *ops, unsigned n, uint16_t target, uint16_t reg)
{
    for (unsigned i = 0; i < n; i++)
        if ((uint16_t)(ops[i] >> 48) == target && (uint16_t)(ops[i] & 0xFFFF) == reg)
            return (int64_t)((ops[i] >> 16) & 0xFFFFFFFFu);
    return -1;
}

/* The form of one emitted stream, from what the compiled program is rather than from the
 * table: the PC trailer last, no DPU RDMA word, the fp32 output stage, the fp16 operand
 * words, and the three buffer addresses where they were asked for. Returns the number of
 * checks that failed and prints the first. */
static int stream_form(const uint64_t *ops, unsigned m, unsigned k, unsigned n)
{
    static const struct { uint16_t target, reg; uint32_t val, mask; const char *what; } W[] = {
        { TARGET_DPU, 0x4010, 0xA0000002u, 0xFFFFFFFFu, "output stage: fp32 out"    },
        { TARGET_DPU, 0x4030, 0x00000310u, 0x0000FFFFu, "output stage: WDMA mode"   },
        { TARGET_DPU, 0x4038, 0x00000053u, 0xFFFFFFFFu, "output stage: 0x4038"      },
        { TARGET_DPU, 0x4044, 0x00000002u, 0xFFFFFFFFu, "output stage: BS ALU"      },
        { TARGET_DPU, 0x4050, 0x00023333u, 0xFFFFFFFFu, "output stage: BS bypass"   },
        { TARGET_DPU, 0x40AC, 0x00000000u, 0xFFFFFFFFu, "output stage: CVT offset"  },
        { TARGET_DPU, 0x40B0, 0x00000001u, 0xFFFFFFFFu, "output stage: CVT scale"   },
        { TARGET_DPU, 0x40B4, 0x00000000u, 0xFFFFFFFFu, "output stage: CVT shift"   },
        { TARGET_CNA, 0x100C, 0x20200120u, 0xFFFFFFFFu, "fp16 operands, row-major A" },
        { TARGET_CORE, 0x3018, 0x10000200u, 0xFFFFFFFFu, "fp16 operands (CORE)"     },
        { TARGET_CNA, 0x1088, RK3576_MM_FP16_IN, 0xFFFFFFFFu, "activation address"  },
        { TARGET_CNA, 0x1110, RK3576_MM_FP16_W, 0xFFFFFFFFu, "weight address"       },
        { TARGET_DPU, 0x4018, RK3576_MM_FP16_OUT, 0xFFFFFFFFu, "output address"     },
    };
    static const uint64_t trailer[4] = {
        0,
        ((uint64_t)TARGET_PC << 48) | 0x0014u,            /* register amount 0: the last */
        (uint64_t)TARGET_PRE_ENABLE << 48,
        ((uint64_t)TARGET_ENABLE << 48) | (0x0Dull << 16) | 0x0008u,   /* CNA CORE DPU */
    };
    const unsigned N = RK3576_MM_FP16_OPS;
    int bad = 0;

    for (unsigned i = 0; i < N - 4; i++) {
        uint16_t t = (uint16_t)(ops[i] >> 48);
        if (t != TARGET_CNA && t != TARGET_CORE && t != TARGET_DPU) {
            if (!bad++)
                printf("  FAIL   %ux%ux%u: word %u writes block 0x%04x, outside CNA, CORE "
                       "and DPU\n", m, k, n, i, t);
        }
    }
    for (unsigned i = 0; i < 4; i++)
        if (ops[N - 4 + i] != trailer[i] && !bad++)
            printf("  FAIL   %ux%ux%u: trailer word %u is %016llx, want %016llx\n", m, k, n,
                   i, (unsigned long long)ops[N - 4 + i], (unsigned long long)trailer[i]);
    for (unsigned i = 0; i < sizeof W / sizeof W[0]; i++) {
        int64_t v = stream_val(ops, N - 4, W[i].target, W[i].reg);
        if ((v < 0 || ((uint32_t)v & W[i].mask) != W[i].val) && !bad++)
            printf("  FAIL   %ux%ux%u: %s, 0x%04x is %s0x%08x, want 0x%08x under 0x%08x\n",
                   m, k, n, W[i].what, W[i].reg, v < 0 ? "unwritten " : "",
                   (uint32_t)(v < 0 ? 0 : v), W[i].val, W[i].mask);
    }
    return bad;
}

/* The weight permutation, element by element, against the layout Rockchip's matmul API
 * documents for an fp16 B, (N/16, K/32, 16, 32). The packer copies rows of 32; this reads
 * every element through the documented index instead. */
static int weight_layout(const uint16_t *d, const uint16_t *B, unsigned k, unsigned n)
{
    for (unsigned nn = 0; nn < n; nn++)
        for (unsigned kk = 0; kk < k; kk++) {
            size_t at = (((size_t)(nn / 16) * (k / 32) + kk / 32) * 16 + nn % 16) * 32 +
                        kk % 32;
            if (d[at] != B[(size_t)nn * k + kk]) {
                printf("  FAIL   %ux%u weight (n %u, k %u) lands off the documented layout\n",
                       k, n, nn, kk);
                return 1;
            }
        }
    return 0;
}

static int host_half(void)
{
    int fails = 0, g;
    /* One step past each bound, and the two alignments. */
    static const unsigned outside[][3] = {
        { 2049, 256, 64 }, { 1, 6176, 64 }, { 1, 48, 64 }, { 1, 0, 64 }, { 0, 256, 64 },
        { 1, 256, 48 }, { 1, 256, 8208 }, { 1, 256, 72 },
    };
    /* One task: the last row inside the data window and the first past it, at three K. */
    static const unsigned task_in[][3] = { { 85, 1536, 64 }, { 1365, 96, 64 }, { 21, 6144, 64 } };
    static const unsigned task_out[][3] = { { 86, 1536, 64 }, { 1366, 96, 64 }, { 22, 6144, 64 } };

    if ((RK3576_MM_FP16_OPS * sizeof(uint64_t)) % 16) {
        printf("  FAIL   a task of %d words does not keep the next one 16-byte aligned\n",
               RK3576_MM_FP16_OPS);
        fails++;
    }
    for (g = 0; g < N_SHAPES; g++) {
        unsigned m = SHAPES[g][0], k = SHAPES[g][1], n = SHAPES[g][2];
        uint64_t ops[RK3576_MM_FP16_OPS];
        size_t wb = rocket_rk3576_mm_fp16_weight_bytes(k, n);
        uint16_t *w = malloc(wb), *B = malloc((size_t)n * k * 2);
        unsigned i;

        if (!w || !B) { printf("  FAIL   out of memory\n"); return fails + 1; }
        if (!rocket_rk3576_mm_fp16_shape_ok(m, k, n)) {
            printf("  FAIL   %ux%ux%u is a gated shape and the envelope refuses it\n", m, k, n);
            fails++;
        }
        if (gen_gemm_fp16_rk3576(ops, m, k, n, RK3576_MM_FP16_IN, RK3576_MM_FP16_W,
                                   RK3576_MM_FP16_OUT) != RK3576_MM_FP16_OPS) {
            printf("  FAIL   %ux%ux%u: the generator refused\n", m, k, n);
            fails++;
        } else {
            fails += stream_form(ops, m, k, n) != 0;
        }
        for (i = 0; i < n * k; i++) B[i] = wval(i / k, i % k);
        if (rocket_rk3576_mm_fp16_pack_weights(w, wb, B, k, n) != 0) {
            printf("  FAIL   %ux%ux%u: the weight pack refused\n", m, k, n);
            fails++;
        } else {
            fails += weight_layout(w, B, k, n);
        }
        free(w); free(B);
    }
    for (g = 0; g < (int)(sizeof outside / sizeof outside[0]); g++)
        if (rocket_rk3576_mm_fp16_shape_ok(outside[g][0], outside[g][1], outside[g][2])) {
            printf("  FAIL   %ux%ux%u is outside the gated grid and the envelope admits it\n",
                   outside[g][0], outside[g][1], outside[g][2]);
            fails++;
        }
    for (g = 0; g < 3; g++) {
        uint64_t ops[RK3576_MM_FP16_OPS];
        if (!rocket_rk3576_mm_fp16_task_ok(task_in[g][0], task_in[g][1], task_in[g][2])) {
            printf("  FAIL   one task of %ux%ux%u fits the data window and is refused\n",
                   task_in[g][0], task_in[g][1], task_in[g][2]);
            fails++;
        }
        if (rocket_rk3576_mm_fp16_task_ok(task_out[g][0], task_out[g][1], task_out[g][2]) ||
            gen_gemm_fp16_rk3576(ops, task_out[g][0], task_out[g][1], task_out[g][2],
                                   0, 0, 0) >= 0) {
            printf("  FAIL   one task of %ux%ux%u is past the data window and is admitted\n",
                   task_out[g][0], task_out[g][1], task_out[g][2]);
            fails++;
        }
    }
    printf("  host: %d shapes, the stream's form and the weight layout, %d refusals, the "
           "data window at 3 K: %s\n", N_SHAPES, (int)(sizeof outside / sizeof outside[0]),
           fails ? "FAIL" : "PASS");
    return fails;
}

/* The first transcription run's value sets. Seeds differ per tensor so A and B are unrelated. */
static void fill(_Float16 *A, _Float16 *B, unsigned m, unsigned k, unsigned n)
{
    for (size_t i = 0; i < (size_t)m * k; i++) A[i] = (_Float16)(tf_int(0xA1, i, -6, 6) * 0.25);
    for (size_t i = 0; i < (size_t)n * k; i++) B[i] = (_Float16)(tf_int(0xB2, i, -3, 3) * 0.5);
}

static void reference(const _Float16 *A, const _Float16 *B, unsigned m, unsigned k,
                      unsigned n, double *ref)
{
    for (unsigned r = 0; r < m; r++)
        for (unsigned c = 0; c < n; c++) {
            double acc = 0;
            for (unsigned i = 0; i < k; i++)
                acc += (double)A[(size_t)r * k + i] * (double)B[(size_t)c * k + i];
            ref[(size_t)r * n + c] = acc;
        }
}

static int run_shape(int fd, unsigned m, unsigned k, unsigned n)
{
    _Float16 *A = malloc((size_t)m * k * 2), *B = malloc((size_t)n * k * 2);
    float *C = malloc((size_t)m * n * 4), *Cr = malloc((size_t)m * n * 4);
    double *ref = malloc((size_t)m * n * 8), worst = 0, t0, ms;
    struct rocket_rk3576_wbo_fp16 *h = NULL;
    size_t i, mism = 0, first = 0, rdiff = 0;
    unsigned sub, rsub = 0;
    uint64_t tasks;
    int rc, rrc;

    if (!A || !B || !C || !Cr || !ref) { printf("  FAIL   out of memory\n"); return 1; }
    fill(A, B, m, k, n);
    reference(A, B, m, k, n, ref);
    rocket_submit_counters_reset();
    t0 = now_ms();
    rc = rocket_matmul_fp16_rk3576(fd, (int)m, (int)k, (int)n, A, B, C);
    ms = now_ms() - t0;
    tasks = rocket_submit_task_count();
    if (rc != ROCKET_OK) {
        printf("  FAIL   %2ux%4ux%-3u rocket_matmul_fp16_rk3576 returned %d\n", m, k, n, rc);
        free(A); free(B); free(C); free(Cr); free(ref);
        return 1;
    }
    sub = rocket_rk3576_mm_fp16_last_attempts();
    for (i = 0; i < (size_t)m * n; i++) {
        double d = (double)C[i] - ref[i];
        if (d != 0 || C[i] != C[i]) { if (!mism++) first = i; }
        if (d < 0) d = -d;
        if (d > worst) worst = d;
    }
    /* The resident handle, from the same B: bit for bit the per-call surface. The
     * output is stamped first so a call that returns without writing cannot pass on
     * the per-call result left in the buffer. */
    memset(Cr, 0xA5, (size_t)m * n * 4);
    rrc = rocket_rk3576_wbo_fp16_create(fd, (int)k, (int)n, B, &h);
    if (rrc == ROCKET_OK) {
        rrc = rocket_matmul_fp16_rk3576_wbo(fd, (int)m, (int)k, (int)n, A, h, Cr);
        rsub = rocket_rk3576_mm_fp16_last_attempts();
    }
    rocket_rk3576_wbo_fp16_free(fd, h);
    if (rrc == ROCKET_OK)
        for (i = 0; i < (size_t)m * n; i++)
            if (memcmp(&C[i], &Cr[i], 4) != 0) rdiff++;
    printf("  %-6s %2ux%4ux%-3u %zu of %u wrong, worst %g, %u submit(s) of %llu task(s), "
           "%.2f ms; resident ", mism || rrc != ROCKET_OK || rdiff ? "FAIL" : "PASS", m, k, n,
           mism, m * n, worst, sub, (unsigned long long)tasks, ms);
    if (rrc != ROCKET_OK) printf("returned %d", rrc);
    else printf("%zu of %u differ from it, %u submit(s)", rdiff, m * n, rsub);
    if (mism)
        printf(", first at m=%zu n=%zu: got %.9g want %.9g", first / n, first % n,
               (double)C[first], ref[first]);
    printf("\n");
    free(A); free(B); free(C); free(Cr); free(ref);
    return mism != 0 || rrc != ROCKET_OK || rdiff != 0;
}

/* One handle at K 256, N 64, used at every M the shape table runs there and at M that
 * need two and three tasks there (one task carries 512 rows at K 256). */
static int run_resident_across_m(int fd)
{
    static const unsigned ms[] = { 1, 2, 3, 4, 8, 16, 100, 512, 513, 1024, 1500 };
    const unsigned k = 256, n = 64, mmax = 1500;
    _Float16 *A = malloc((size_t)mmax * k * 2), *B = malloc((size_t)n * k * 2);
    float *C = malloc((size_t)mmax * n * 4);
    double *ref = malloc((size_t)mmax * n * 8);
    struct rocket_rk3576_wbo_fp16 *h = NULL;
    size_t i, mism = 0;
    unsigned j, calls = 0;
    int rc, fail = 1;

    if (!A || !B || !C || !ref) { printf("  FAIL   out of memory\n"); goto out; }
    fill(A, B, mmax, k, n);
    reference(A, B, mmax, k, n, ref);
    rc = rocket_rk3576_wbo_fp16_create(fd, (int)k, (int)n, B, &h);
    if (rc != ROCKET_OK) {
        printf("  FAIL   resident across M: create returned %d\n", rc);
        goto out;
    }
    /* Row r of the reference is row r of any M >= r+1, so one reference serves all. */
    for (j = 0; j < sizeof ms / sizeof ms[0]; j++) {
        memset(C, 0xA5, (size_t)ms[j] * n * 4);
        rc = rocket_matmul_fp16_rk3576_wbo(fd, (int)ms[j], (int)k, (int)n, A, h, C);
        if (rc != ROCKET_OK) {
            printf("  FAIL   resident across M: M=%u returned %d\n", ms[j], rc);
            goto out;
        }
        for (i = 0; i < (size_t)ms[j] * n; i++)
            if ((double)C[i] != ref[i]) mism++;
        calls++;
    }
    fail = mism != 0;
    printf("  %-6s resident across M: one handle at K %u N %u, M 1-1500 in %u calls, %zu "
           "wrong\n", fail ? "FAIL" : "PASS", k, n, calls, mism);
out:
    rocket_rk3576_wbo_fp16_free(fd, h);
    free(A); free(B); free(C); free(ref);
    return fail;
}

/* The handle's refusals: outside the envelope at create, and a K or N the handle does not
 * hold at the call. Each must refuse, and the refused call must not write C. */
static int run_resident_refusals(int fd)
{
    static const int bad_create[][2] = { { 48, 64 }, { 6176, 64 }, { 256, 48 }, { 256, 8208 } };
    static const int bad_call[][3] = { { 1, 512, 64 }, { 1, 256, 128 }, { 2049, 256, 64 } };
    const unsigned k = 256, n = 64;
    _Float16 *A = calloc((size_t)2049 * 512, 2), *B = calloc((size_t)8208 * 256, 2);
    float *C = malloc((size_t)2049 * 128 * 4);
    struct rocket_rk3576_wbo_fp16 *h = NULL, *hb;
    unsigned j, fails = 0, checks = 0;
    int rc;

    if (!A || !B || !C) { printf("  FAIL   out of memory\n"); fails++; goto out; }
    for (j = 0; j < sizeof bad_create / sizeof bad_create[0]; j++, checks++) {
        hb = (struct rocket_rk3576_wbo_fp16 *)1;
        rc = rocket_rk3576_wbo_fp16_create(fd, bad_create[j][0], bad_create[j][1], B, &hb);
        if (rc != ROCKET_E_UNSUPPORTED || hb != NULL) {
            printf("  FAIL   resident create K=%d N=%d returned %d, want %d and no handle\n",
                   bad_create[j][0], bad_create[j][1], rc, ROCKET_E_UNSUPPORTED);
            if (rc == ROCKET_OK) rocket_rk3576_wbo_fp16_free(fd, hb);
            fails++;
        }
    }
    if (rocket_rk3576_wbo_fp16_create(fd, (int)k, (int)n, B, &h) != ROCKET_OK) {
        printf("  FAIL   resident refusals: the K %u N %u handle would not create\n", k, n);
        fails++;
        goto out;
    }
    for (j = 0; j < sizeof bad_call / sizeof bad_call[0]; j++, checks++) {
        int want = bad_call[j][0] > 2048 ? ROCKET_E_UNSUPPORTED : ROCKET_E_SHAPE;
        size_t ne = (size_t)bad_call[j][0] * (size_t)bad_call[j][2], i, touched = 0;
        memset(C, 0x5A, ne * 4);
        rc = rocket_matmul_fp16_rk3576_wbo(fd, bad_call[j][0], bad_call[j][1],
                                           bad_call[j][2], A, h, C);
        for (i = 0; i < ne * 4; i++) if (((unsigned char *)C)[i] != 0x5A) touched++;
        if (rc != want || touched) {
            printf("  FAIL   resident call M=%d K=%d N=%d on a K %u N %u handle returned %d "
                   "(want %d), %zu output bytes written\n", bad_call[j][0], bad_call[j][1],
                   bad_call[j][2], k, n, rc, want, touched);
            fails++;
        }
    }
    checks++;
    if (rocket_matmul_fp16_rk3576_wbo(fd, 1, (int)k, (int)n, A, NULL, C) != ROCKET_E_SHAPE) {
        printf("  FAIL   a NULL handle did not refuse\n");
        fails++;
    }
    printf("  %-6s resident refusals: %u checks, %u wrong\n", fails ? "FAIL" : "PASS",
           checks, fails);
out:
    rocket_rk3576_wbo_fp16_free(fd, h);
    free(A); free(B); free(C);
    return fails != 0;
}

/* The same program with the weights row-major, B[n][k] at n*k + k: the first run's DENSE. */
static int run_control(int fd, unsigned m, unsigned k, unsigned n)
{
    rocket_bo in = {0}, w = {0}, out = {0}, rc = {0};
    uint64_t ops[RK3576_MM_FP16_OPS];
    uint32_t in_h[3], out_h[1];
    _Float16 *A = malloc((size_t)m * k * 2), *B = malloc((size_t)n * k * 2);
    double *ref = malloc((size_t)m * n * 8);
    size_t i, mism = 0, stale = 0, ob = rocket_rk3576_mm_fp16_out_bytes(m, n);
    int fail = 1;

    if (!A || !B || !ref) goto out;
    fill(A, B, m, k, n);
    reference(A, B, m, k, n, ref);
    if (rocket_bo_alloc32(fd, rocket_rk3576_mm_fp16_in_bytes(m, k), &in) ||
        rocket_bo_alloc32(fd, rocket_rk3576_mm_fp16_weight_bytes(k, n), &w) ||
        rocket_bo_alloc32(fd, ob, &out) || rocket_bo_alloc32(fd, sizeof ops, &rc)) {
        printf("  FAIL   control: a buffer would not allocate\n");
        goto out;
    }
    rocket_bo_prep(fd, &in, 1, 0);
    memset(in.ptr, 0, in.size); memcpy(in.ptr, A, (size_t)m * k * 2);
    rocket_bo_fini(fd, &in);
    rocket_bo_prep(fd, &w, 1, 0);
    memcpy(w.ptr, B, (size_t)n * k * 2);
    rocket_bo_fini(fd, &w);
    gen_gemm_fp16_rk3576(ops, m, k, n, (uint32_t)in.dma_address, (uint32_t)w.dma_address,
                           (uint32_t)out.dma_address);
    rocket_bo_prep(fd, &rc, 1, 0);
    memcpy(rc.ptr, ops, sizeof ops);
    rocket_bo_fini(fd, &rc);
    tf_sentinel_bo(fd, &out);
    in_h[0] = in.handle; in_h[1] = w.handle; in_h[2] = rc.handle;
    out_h[0] = out.handle;
    if (rocket_submit_matmul(fd, &rc, RK3576_MM_FP16_OPS, in_h, 3, out_h, 1, 4000) != 0 ||
        rocket_bo_prep(fd, &out, 0, 2000000000ull) < 0) {
        printf("  FAIL   control: the submit failed\n");
        goto out;
    }
    {
        const float *o = (const float *)out.ptr;
        const unsigned char *b = (const unsigned char *)out.ptr;
        for (i = 0; i < (size_t)m * n; i++) {
            if (b[4 * i] == TF_SENTINEL_BYTE && b[4 * i + 1] == TF_SENTINEL_BYTE &&
                b[4 * i + 2] == TF_SENTINEL_BYTE && b[4 * i + 3] == TF_SENTINEL_BYTE)
                stale++;
            else if ((double)o[i] != ref[i])
                mism++;
        }
    }
    rocket_bo_fini(fd, &out);
    /* Wrong where it wrote, and it wrote: a job that wrote nothing proves nothing. */
    fail = !(mism > 0 && stale < (size_t)m * n);
    printf("  %-6s control %ux%ux%u, weights row-major: %zu of %u wrong, %zu unwritten "
           "(must be written and wrong)\n", fail ? "FAIL" : "PASS", m, k, n, mism, m * n,
           stale);
out:
    if (rc.ptr) rocket_bo_free(fd, &rc);
    if (out.ptr) rocket_bo_free(fd, &out);
    if (w.ptr) rocket_bo_free(fd, &w);
    if (in.ptr) rocket_bo_free(fd, &in);
    free(A); free(B); free(ref);
    return fail;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw;
    int fd, i, fails, ran = 0;

    printf("== RK3576 fp16 matmul gate ==\n");
    fails = host_half();
    if (fails) { printf("== the host half failed: %d ==\n", fails); return 1; }

    hw = rocket_hw_current();
    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_mm_fp16_gate: profile is %s, not rk3576 — the host half passed, "
               "the device half is skipped\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) {
        printf("rk3576_mm_fp16_gate: no NPU device — the host half passed, the device "
               "half is skipped\n");
        return 2;
    }
    for (i = 0; i < N_SHAPES; i++) {
        char name[32];
        snprintf(name, sizeof name, "%ux%ux%u", SHAPES[i][0], SHAPES[i][1], SHAPES[i][2]);
        if (argc > 1 && !strstr(name, argv[1])) continue;
        fails += run_shape(fd, SHAPES[i][0], SHAPES[i][1], SHAPES[i][2]);
        ran++;
    }
    for (i = 0; i < N_TILED; i++) {
        char name[32];
        snprintf(name, sizeof name, "%ux%ux%u", TILED[i][0], TILED[i][1], TILED[i][2]);
        if (argc > 1 && !strstr(name, argv[1])) continue;
        fails += run_shape(fd, TILED[i][0], TILED[i][1], TILED[i][2]);
        ran++;
    }
    if (argc <= 1) {
        fails += run_resident_across_m(fd);
        fails += run_resident_refusals(fd);
        ran += 2;
    }
    fails += run_control(fd, 1, 64, 64);
    ran++;
    printf("== %d passed, %d failed ==\n", ran - fails, fails);
    rocket_close(fd);
    return fails ? 1 : 0;
}
