// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_matmul_gate.c — the RK3576 int8 matmul, against a CPU model.
 *
 * This gates the LIBRARY entries (rocket_matmul_int8_rk3576 and its int32-output
 * sibling), not the register program: the operand scatter, the N tiling, the K split,
 * the row window, the requant and the output de-scatter, all through the public API a
 * caller would use.
 *
 * The shape table is built around what hides defects rather than what looks tidy.
 * Round numbers hide layouts, so it carries M with no even divisor (7, 13, 49), K and
 * N off the round grid where the granularity still allows it, planes that are a single
 * row and a single column, and N wide enough to force several tiles. M=1 is in it
 * because M carries no constraint on this part and a gate that only ran M%4 shapes
 * would not say so.
 *
 * The `refuse` column is a result, not an omission: K is the contraction and is not
 * tiled here, so a K past what one task holds must come back ROCKET_E_SHAPE rather
 * than compute something quietly wrong.
 *
 * Run with `sudo -E` on the RK3576. Exits 2 (skip) on any other part.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

typedef struct {
    const char *name;
    int M, K, N;
    int refuse;      /* 1 = the planner MUST refuse; running it is the bug */
} mm_shape;

static const mm_shape SHAPES[] = {
    /* the plain envelope */
    {"tiny",            8,    64,   32, 0},
    {"m1",              1,   256,   32, 0},   /* no M constraint on this part */
    {"m2",              2,   256,   64, 0},
    {"m3",              3,   512,   32, 0},
    {"square",         64,   512,   64, 0},
    {"deep-k",         56,  4608,  128, 0},   /* K at what one task contracts */
    {"k4096",          64,  4096,  128, 0},
    {"wide-m",       1024,   256,  128, 0},   /* M*K at the feature-budget cap */
    {"tall-m",       2048,   128,   64, 0},   /* M past one task: the row window */
    {"wide-n",         64,   512, 1024, 0},
    {"n-tiled",        64,   512, 4096, 0},   /* several N tiles */
    {"n-tiled-deep",   32,  2048, 3072, 0},
    /* awkward numbers: M with no even divisor, N and K off the round grid */
    {"m7",              7,    64,   32, 0},
    {"m13",            13,    96,   32, 0},
    {"m49",            49,    32,   32, 0},
    {"m100",          100,   288,   96, 0},
    {"m33-n160",       33,   160,  160, 0},
    {"k1120",          40,  1120,  224, 0},
    /* past one task's contraction: K is split through the int32 writer and the requant
     * moves to the host. These used to be refusals. */
    {"k8192",          32,  8192,  128, 0},
    {"k16384",         16, 16384,   64, 0},
    /* the refusals that remain: neither is a K bound */
    {"k-unaligned",    32,   100,   32, 1},   /* K%32 */
    {"n-unaligned",    32,   128,   48, 1},   /* N%32 */
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof SHAPES[0]))

/* The int32-output entry, gated against an exact CPU model rather than a requantized
 * one: this path does no rounding at all, so anything but bit-exact is a defect. The
 * table crosses the K split (K past 4608 is several slices) with the awkward M and N
 * the int8 table uses, since the scatter into the delivered channel slots is what is
 * actually new and it lives on the N axis. */
static const mm_shape I32_SHAPES[] = {
    {"i32-tiny",        8,    64,   32, 0},
    {"i32-m1",          1,   256,   32, 0},
    {"i32-m7",          7,    96,   32, 0},
    {"i32-m13",        13,   160,   64, 0},
    {"i32-n-tiled",    32,   512,  256, 0},   /* several N tiles */
    {"i32-k4608",      32,  4608,   32, 0},   /* one int8 task's contraction */
    {"i32-k9216",      32,  9216,   32, 0},   /* K a whole number of slices */
    {"i32-k8192-n32",  32,  8192,   32, 0},   /* a short last slice, one N tile */
    {"i32-k8192",      32,  8192,  128, 0},   /* several K slices AND several N tiles */
    {"i32-k16384",     16, 16384,   64, 0},
    {"i32-tall-m",   1024,   128,   64, 0},   /* M past one task: the row window */
    /* The wide writer's map is a stream cut into runs of ow*oh_full, so the plane's
     * pixel count is one of its axes and a round one hides the cut. These are the
     * awkward ones: an odd surface that is not the degenerate A=1, a pixel count with
     * no even divisor, and a plane the row planner has to split into several tasks. */
    {"i32-m33",        33,   160,   96, 0},
    {"i32-m100",      100,   288,   96, 0},
    {"i32-m-rows",   4096,  1024,   64, 0},   /* several row tasks per tile */
    {"i32-n-wide",     64,  1024,  512, 0},   /* the N the wide writer buys */
    /* Surfaces in the region where a probe driving EVERY output channel stops fitting
     * the wide map. The matmul does not drive every channel — the scatter leaves half
     * of each sixteen at zero — so these say whether that boundary reaches the library
     * or stops at the probe. */
    {"i32-a48-oc192",  48,  1024,   96, 0},
    {"i32-a64-oc192",  64,  1024,   96, 0},
    {"i32-a128-oc192",128,  1024,   96, 0},
    {"i32-a33-oc192",  33,   512,   96, 0},
    /* Where the wide writer is actually WORTH something. The N tile is bounded by the
     * PROGRAMMED output channels, so halving the multiplier doubles the tile and halves
     * the submits — and a submit here costs an idle, not 1.4 ms. Below the tile cap the
     * two writers cost the same, so a table without these would show no gain at all. */
    {"i32-n2048",      64,  1024, 2048, 0},
    {"i32-n4096",      32,  1024, 4096, 0},
};
#define N_I32_SHAPES ((int)(sizeof I32_SHAPES / sizeof I32_SHAPES[0]))

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

/* The requant the emitter programs, in the caller's terms: scale/shift are derived
 * from the conv scale exactly as the vendor (QNNPACK) does and the DPU rounds to
 * nearest, so the model has to do the same arithmetic rather than a float multiply. */
static int model_requant(int64_t acc, float scale)
{
    union { float f; uint32_t u; } cv;
    uint32_t bits;
    unsigned shift_reg, mul;
    int64_t half, v;
    cv.f = scale;
    bits = cv.u;
    shift_reg = 127u + 31u - 32u - (bits >> 23) + 16u - 1u;
    mul = ((bits >> 9) & 0x7FFFu) + 1u;
    if (mul < (1u << 14)) mul |= (1u << 14);
    half = shift_reg ? ((int64_t)1 << (shift_reg - 1)) : 0;
    v = (acc * (int64_t)mul + half) >> shift_reg;
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int)v;
}

/* What the host does when K is split: the accumulator never reaches the DPU's OUT_CVT,
 * so the requant is a plain float multiply and round-half-away-from-zero. Bit-identical
 * to what rocket_matmul_int8_rk3576 applies on that path, and NOT to model_requant —
 * the two differ by a unit in the last place on some inputs, which is the visible edge
 * of the boundary. */
static int host_requant(int64_t acc, float scale)
{
    float v = (float)acc * scale;
    long r = (long)(v < 0.0f ? v - 0.5f : v + 0.5f);
    return (int)(r < -128 ? -128 : (r > 127 ? 127 : r));
}

static int run_shape(int fd, const mm_shape *s, int verbose)
{
    int M = s->M, K = s->K, N = s->N;
    int8_t *A = NULL, *B = NULL, *C = NULL;
    int32_t *bias = NULL;
    float scale;
    int rc = -1, exact = 0, total = M * N, maxdiff = 0, shown = 0;
    int m, n, k, jobs, Mt = 0, Kt = 0, Nt = 0, deep = 0;
    unsigned divisor, seed;
    double t0, ms;

    jobs = rocket_matmul_plan_int8_rk3576(M, K, N, &Mt, &Kt, &Nt);
    if (s->refuse) {
        printf("  %-6s %-14s %5dx%-5dx%-5d %s\n",
               jobs < 0 ? "PASS" : "RAN", s->name, M, K, N,
               jobs < 0 ? "refused, as the table says" : "PLANNED — it should not have");
        return jobs < 0 ? 0 : 1;
    }
    /* The single-task planner refuses a K past one contraction and the entry runs it
     * anyway, through the int32 writer. Which side of that boundary a shape lands on
     * decides which requant the model has to apply. */
    if (jobs < 0) deep = 1;

    A    = calloc((size_t)M * K, 1);
    B    = calloc((size_t)N * K, 1);
    C    = calloc((size_t)M * N, 1);
    bias = calloc(N, sizeof *bias);
    if (!A || !B || !C || !bias) { rc = 1; goto done; }

    /* Vary on every axis: an operand flat along an axis proves nothing about that
     * axis's stride. Keep the accumulator inside the int8 output range for most
     * elements, since a surface clamped everywhere hides the arithmetic under the
     * saturation — a dense random sum grows as sqrt(K). */
    seed = 0x9E3779B9u ^ (unsigned)(M * 31 + K * 17 + N * 7);
    for (m = 0; m < M; m++)
        for (k = 0; k < K; k++)
            A[(size_t)m * K + k] = (int8_t)(((m * 7 + k * 13) % 61) - 30);
    for (n = 0; n < N; n++)
        for (k = 0; k < K; k++) {
            seed = seed * 1103515245u + 12345u;
            B[(size_t)n * K + k] = (int8_t)((int)((seed >> 16) % 17u) - 8);
        }
    for (n = 0; n < N; n++)
        bias[n] = (int32_t)((n - N / 2) * 8);

    divisor = 1;
    while ((double)divisor < 2.0 * sqrt((double)K)) divisor *= 2;
    scale = 1.0f / (float)divisor;

    t0 = now_ms();
    rc = rocket_matmul_int8_rk3576(fd, M, K, N, A, B, bias, scale, C);
    ms = now_ms() - t0;
    if (rc != 0) {
        printf("  %-6s %-14s %5dx%-5dx%-5d rocket_matmul_int8_rk3576 returned %d\n",
               "FAIL", s->name, M, K, N, rc);
        rc = 1; goto done;
    }

    for (m = 0; m < M; m++)
        for (n = 0; n < N; n++) {
            int64_t acc = bias[n];
            int got, want, d;
            for (k = 0; k < K; k++)
                acc += (int64_t)A[(size_t)m * K + k] * B[(size_t)n * K + k];
            want = deep ? host_requant(acc, scale) : model_requant(acc, scale);
            got  = C[(size_t)m * N + n];
            d = got > want ? got - want : want - got;
            if (d > maxdiff) maxdiff = d;
            if (got == want) exact++;
            else if (verbose && shown < 8) {
                printf("      mism m=%d n=%d: want %d got %d (acc %lld)\n",
                       m, n, want, got, (long long)acc);
                shown++;
            }
        }
    rc = (exact == total) ? 0 : 1;
    if (deep)
        printf("  %-6s %-14s %5dx%-5dx%-5d %9d/%-9d K split, host requant  %7.1f ms\n",
               rc ? "FAIL" : "PASS", s->name, M, K, N, exact, total, ms);
    else
        printf("  %-6s %-14s %5dx%-5dx%-5d %9d/%-9d Nt=%-5d %2d job%s %7.1f ms\n",
               rc ? "FAIL" : "PASS", s->name, M, K, N, exact, total, Nt,
               jobs, jobs == 1 ? " " : "s", ms);
    if (rc && maxdiff) printf("         maxdiff %d\n", maxdiff);

done:
    free(A); free(B); free(C); free(bias);
    return rc;
}

/* The int32 entry, against an exact model. No requant anywhere, so the bar is every
 * element identical. */
static int run_i32_shape(int fd, const mm_shape *s, int verbose)
{
    int M = s->M, K = s->K, N = s->N;
    int8_t *A = NULL, *B = NULL;
    int32_t *C = NULL, *bias = NULL;
    int rc = -1, exact = 0, total = M * N, shown = 0;
    int m, n, k;
    unsigned seed;
    double t0, ms;

    A    = calloc((size_t)M * K, 1);
    B    = calloc((size_t)N * K, 1);
    C    = calloc((size_t)M * N, sizeof *C);
    bias = calloc(N, sizeof *bias);
    if (!A || !B || !C || !bias) { rc = 1; goto done; }

    /* Full int8 range on both operands: there is no output saturation to hide behind
     * here, so the accumulator is allowed to grow to whatever K gives it. */
    seed = 0x9E3779B9u ^ (unsigned)(M * 31 + K * 17 + N * 7);
    for (m = 0; m < M; m++)
        for (k = 0; k < K; k++)
            A[(size_t)m * K + k] = (int8_t)(((m * 7 + k * 13) % 255) - 127);
    for (n = 0; n < N; n++)
        for (k = 0; k < K; k++) {
            seed = seed * 1103515245u + 12345u;
            B[(size_t)n * K + k] = (int8_t)((int)((seed >> 16) % 255u) - 127);
        }
    for (n = 0; n < N; n++)
        bias[n] = (int32_t)((n - N / 2) * 1000);

    t0 = now_ms();
    rc = rocket_matmul_int8_rk3576_i32(fd, M, K, N, A, B, bias, C);
    ms = now_ms() - t0;
    if (rc != 0) {
        printf("  %-6s %-14s %5dx%-5dx%-5d rocket_matmul_int8_rk3576_i32 returned %d\n",
               "FAIL", s->name, M, K, N, rc);
        rc = 1; goto done;
    }

    for (m = 0; m < M; m++)
        for (n = 0; n < N; n++) {
            int32_t acc = bias[n], got = C[(size_t)m * N + n];
            for (k = 0; k < K; k++)
                acc += (int32_t)A[(size_t)m * K + k] * B[(size_t)n * K + k];
            if (acc == got) exact++;
            else if (verbose && shown < 8) {
                printf("      mism m=%d n=%d: want %d got %d\n", m, n, acc, got);
                shown++;
            }
        }
    rc = (exact == total) ? 0 : 1;
    printf("  %-6s %-14s %5dx%-5dx%-5d %9d/%-9d exact %7.1f ms\n",
           rc ? "FAIL" : "PASS", s->name, M, K, N, exact, total, ms);

done:
    free(A); free(B); free(C); free(bias);
    return rc;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, i, fails = 0, verbose = getenv("ROCKET_MG_VERBOSE") != NULL;
    /* The FIRST shape is paced too, not just the gaps between them. An int32 job
     * poisons the next submit of any kind across PROCESSES, so a gate run started right
     * after another one can find its first int8 shape already dead — which reads as a
     * broken matmul rather than as the previous run's tail. */
    int gap = getenv("ROCKET_MG_GAP_MS") ? atoi(getenv("ROCKET_MG_GAP_MS")) : 200;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_matmul_gate: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_matmul_gate: no NPU device — skipping\n"); return 2; }

    printf("== RK3576 int8 matmul gate ==\n");
    for (i = 0; i < N_SHAPES; i++) {
        if (argc > 1 && !strstr(SHAPES[i].name, argv[1])) continue;
        sleep_ms(gap);
        fails += run_shape(fd, &SHAPES[i], verbose);
    }
    printf("== the int32 output, exact ==\n");
    for (i = 0; i < N_I32_SHAPES; i++) {
        if (argc > 1 && !strstr(I32_SHAPES[i].name, argv[1])) continue;
        sleep_ms(gap);
        fails += run_i32_shape(fd, &I32_SHAPES[i], verbose);
    }
    printf("== %d passed, %d failed ==\n", N_SHAPES + N_I32_SHAPES - fails, fails);
    rocket_close(fd);
    return fails ? 1 : 0;
}
