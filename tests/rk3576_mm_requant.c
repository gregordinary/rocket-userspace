// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_mm_requant.c — is the int8 matmul entry's DEVICE requant the arithmetic the
 * host models say it is?
 *
 * The RK3576's int8 matmul writes an int8 surface through the DPU's output convertor,
 * so the entry's contract is a REQUANT and not an accumulation. Every accuracy statement
 * made about that entry so far — what one per-tensor output scale costs a real model,
 * and what a rotation plus a per-channel requant recover — was simulated on the host
 * with an EXACT float scale and round-to-even. The part does not implement that. It
 * implements a 15-bit integer multiplier and an arithmetic shift, and no gate anywhere
 * has compared the two. This one does, on the device, at the entry.
 *
 * THREE QUANTITIES, and each answers a different question:
 *
 *   dev vs int    the device's own surface against `requant_model.h`'s
 *                 `sat8(round_half_to_even((acc + bias) * MUL >> SHIFT))`, with
 *                 MUL/SHIFT from the same derivation the emitter programs. A non-zero
 *                 count here means the matmul path's epilogue is not the conv path's
 *                 arithmetic, and every int8 model prediction in the corpus rests on
 *                 it being so. This is what the gate's rc scores.
 *
 *   int vs float  the integer model against the exact-float-with-round-to-even model
 *                 the accuracy simulations used. This is not a defect — it is the
 *                 instrument error of those simulations, and it bounds how far their
 *                 perplexity numbers could move if they were re-run through the real
 *                 requant. Reported, never scored.
 *
 *   dev vs float  the two composed, for completeness.
 *
 * WHAT A GREEN RUN HERE WOULD NOT SHOW. It says the epilogue's ARITHMETIC is modelled,
 * at a per-tensor scale. It says nothing about the per-output-channel C ramp (a
 * different coefficient packing, a different gate), nothing about a real model's
 * activation distribution (these are pseudorandom operands), and nothing about the
 * drain deadline, whose loss is a DROPPED write rather than a wrong value — the library
 * refuses a task that wrote nothing, so a drop reaches this gate as a refusal.
 *
 * THE REFERENCE IS NOT DEGENERATE, deliberately. The operands come from a xorshift
 * rather than from a short periodic pattern: a periodic generator makes whole blocks of
 * the contraction sum to zero and hands back an identically-zero expected surface, which
 * cannot witness a wrong value at all (see rk3576_mm_corr's header). The span of the
 * expected surface is printed for that reason, and the gate refuses a cell whose
 * expected surface is within +-1 of zero everywhere.
 *
 * EACH CELL RUNS AT TWO SCALES: one that fills the int8 range with few saturations, and
 * one four times larger that saturates hard. The requant's rounding and its saturation
 * are separate clauses of the same expression and a cell that never reaches the clip
 * cannot score the second.
 *
 * Run with `sudo -E`, pinned:
 *   sudo -E taskset -c 4 ./build/rk3576_mm_requant
 *
 * A trailing argument list runs a single cell instead:
 *   sudo -E taskset -c 4 ./build/rk3576_mm_requant M K N
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "rocket_rk3576_internal.h"
#include "npu_regcmd_rk3576.h"
#include "requant_model.h"

/* The host buffer's own stamp: an element the entry's de-scatter never reached. It has
 * to be distinguishable from a legitimate result, and every legitimate result is an
 * int8, so there is no free value — the count of elements still holding it is reported
 * separately rather than folded into the wrong count. */
#define HOST_STAMP ((int8_t)0x5A)

static uint32_t xs32(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return (*s = x);
}

/* int8 operands with a spread, not a pattern: a byte from the generator mapped to
 * [-127, 127]. 0x80 is excluded so the CPU reference and the part agree on the operand
 * range the weight cube was validated over. */
static int8_t rnd_i8(uint32_t *s)
{
    int v = (int)(xs32(s) & 0xFFu) - 128;
    return (int8_t)(v == -128 ? -127 : v);
}

/* The exact-float model the accuracy simulations used: a real scale, round half to
 * even, saturate. Kept here rather than in requant_model.h because it is NOT a model of
 * this part — it is a model of the instrument that predicted this part. */
static int float_requant(int32_t acc, float scale)
{
    double v = (double)acc * (double)scale;
    double r = nearbyint(v);                  /* the default rounding mode is to-nearest-even */
    if (r >  127.0) r =  127.0;
    if (r < -128.0) r = -128.0;
    return (int)r;
}

struct cell { int M, K, N; };

/* ---- the per-output-column arm -------------------------------------------
 *
 * The per-column entry carries its scale on the coefficient group's int16 C ramp over a
 * shared (MUL, SHIFT), so the model here is the ramp and not an exact scale: it calls
 * the SAME planner the entry does and applies `(acc + bias) * C[n]` then the shared
 * shift. That is deliberate — the question is whether the DEVICE implements the
 * epilogue the planner assumes, not whether the planner is a good approximation. What
 * the ramp gives up against an exact per-column scale is a separate quantity and is
 * reported alongside, from the entry's own `worst_rel_err`.
 *
 * The tile boundary matters and is not visible in a total: the (MUL, SHIFT) is per N
 * tile, so the model has to plan tile by tile exactly as the entry does. The tile width
 * comes from the pure planner rather than from a constant.
 */
static int perc_arm(int fd, int M, int K, int N, const int8_t *A, const int8_t *B,
                    const int32_t *bias, const int32_t *acc, const float *scale_n,
                    double spread)
{
    int8_t *C = malloc((size_t)M * N);
    int8_t *model = malloc((size_t)M * N);
    int64_t *sum_abs_w = calloc((size_t)N, sizeof *sum_abs_w);
    int32_t *tile_bias = NULL;
    int16_t *cmul = NULL;
    double worst = -1.0, worst_sa = -1.0, model_worst = 0.0;
    long long wrong = 0, vs_exact = 0;
    int maxd = 0, maxd_exact = 0, nt = 0, rc, shown = 0, bad = 0;

    if (!C || !model || !sum_abs_w) { free(C); free(model); free(sum_abs_w); return -1; }

    for (int n = 0; n < N; n++) {
        const int8_t *w = B + (size_t)n * K;
        int64_t s = 0;
        for (int k = 0; k < K; k++) s += w[k] < 0 ? -(int64_t)w[k] : (int64_t)w[k];
        sum_abs_w[n] = s;
    }
    if (rocket_matmul_plan_int8_rk3576(M, K, N, NULL, NULL, &nt) < 0 || nt <= 0) {
        printf("  perc %4d x %5d x %4d  no plan — not scored\n", M, K, N);
        free(C); free(model); free(sum_abs_w);
        return 1;
    }

    /* Plan every tile the way the entry does, and requant this tile's columns. */
    for (int n0 = 0; n0 < N; n0 += nt) {
        unsigned tile_n = (unsigned)(N - n0 < nt ? N - n0 : nt);
        unsigned nreg = rocket_rk3576_pad_oc(tile_n);
        float base = 0.0f;
        double err = 0.0;
        unsigned mul, shift;

        free(tile_bias); free(cmul);
        tile_bias = calloc(nreg, sizeof *tile_bias);
        cmul = calloc(nreg, sizeof *cmul);
        if (!tile_bias || !cmul) { bad = 1; break; }
        if (bias) for (unsigned j = 0; j < tile_n; j++) tile_bias[j] = bias[n0 + j];

        if (rocket_rk3576_plan_perchannel("model", (unsigned)n0, tile_n, nreg, tile_bias,
                                          sum_abs_w, 1.0f, scale_n, 1.0f, NULL,
                                          cmul, &base, &err) != 0) { bad = 1; break; }
        if (err > model_worst) model_worst = err;
        requant_params(base, &mul, &shift);
        for (int m = 0; m < M; m++)
            for (unsigned j = 0; j < tile_n; j++) {
                int64_t v = (int64_t)acc[(size_t)m * N + n0 + j] * (int64_t)cmul[j];
                model[(size_t)m * N + n0 + j] =
                    (int8_t)requant_sat8(requant_round_shift(v * (int64_t)mul, shift));
            }
    }
    if (bad) { free(C); free(model); free(sum_abs_w); free(tile_bias); free(cmul); return -1; }

    memset(C, HOST_STAMP, (size_t)M * N);
    rc = rocket_matmul_int8_rk3576_perc(fd, M, K, N, A, B, bias, scale_n, C, &worst);
    if (rc != 0) {
        printf("  perc %4d x %5d x %4d  REFUSED rc=%d\n", M, K, N, rc);
        free(C); free(model); free(sum_abs_w); free(tile_bias); free(cmul);
        return 1;
    }

    for (size_t i = 0; i < (size_t)M * N; i++) {
        int d = C[i] - model[i];
        int e = C[i] - float_requant(acc[i], scale_n[i % (size_t)N]);
        if (d) {
            wrong++;
            if (abs(d) > maxd) maxd = abs(d);
            if (shown < 6) {
                printf("    [%zu,%zu] acc=%d  dev=%d  ramp_model=%d\n",
                       i / (size_t)N, i % (size_t)N, acc[i], C[i], model[i]);
                shown++;
            }
        }
        if (e) { vs_exact++; if (abs(e) > maxd_exact) maxd_exact = abs(e); }
    }

    printf("  perc %4d x %5d x %4d  Ntile %d  spread %.1fx  worst_rel_err %.4f%% "
           "(model %.4f%%)\n", M, K, N, nt, spread, 100.0 * worst, 100.0 * model_worst);
    printf("      dev-vs-ramp %lld of %lld (max |d| %d)   dev-vs-EXACT-per-column %lld "
           "(%.3f%%, max |d| %d)\n",
           wrong, (long long)M * N, maxd, vs_exact,
           100.0 * (double)vs_exact / ((double)M * N), maxd_exact);

    /* THE SUPPLIED-SUM ENTRY IS THE SAME ENTRY. `_perc_sa` skips the O(N*K) pass over B
     * and takes the caller's per-column sums instead; the sums handed over here are the
     * ones this gate computed for its own model, so the two surfaces must be identical
     * byte for byte. A difference is not a tolerance question — it means the supplied
     * array is not the quantity the entry's own pass produces, which is the one way this
     * parameter can be wrong without any caller noticing. */
    {
        int8_t *C_sa = malloc((size_t)M * N);
        long long diff = 0;
        if (!C_sa) { free(C); free(model); free(sum_abs_w); free(tile_bias); free(cmul);
                     return -1; }
        memset(C_sa, HOST_STAMP, (size_t)M * N);
        rc = rocket_matmul_int8_rk3576_perc_sa(fd, M, K, N, A, B, bias, scale_n,
                                               sum_abs_w, C_sa, &worst_sa);
        if (rc != 0) {
            printf("      supplied-sum arm REFUSED rc=%d\n", rc);
            wrong++;
        } else {
            for (size_t i = 0; i < (size_t)M * N; i++) if (C_sa[i] != C[i]) diff++;
            printf("      supplied-sum vs computed-sum %lld of %lld differ  "
                   "worst_rel_err %.4f%%\n", diff, (long long)M * N, 100.0 * worst_sa);
            if (diff) wrong++;
        }
        free(C_sa);
    }

    /* THE RESIDENT-WEIGHT ENTRY IS THE SAME ENTRY AGAIN. The cached whole-N cube must
     * be byte-for-byte the concatenation of the per-tile cubes the per-call path packs
     * — including across a forced multi-tile run — so the surface must match exactly.
     * And with ROCKET_RK3576_BO_POOL=1 the same call must still match: the pool
     * recycles this path's transient BOs, and a pooled BO not fully rewritten before
     * use shows up here as a differing surface. Run twice pooled — the first pooled
     * call only FILLS the pool on its frees; the second is the one that reuses. */
    {
        struct rocket_rk3576_wbo *wbo = NULL;
        int8_t *C_w = malloc((size_t)M * N);
        long long diffw = 0, diffp = 0;
        double worst_w = -1.0;
        if (!C_w) { free(C); free(model); free(sum_abs_w); free(tile_bias); free(cmul);
                    return -1; }
        rc = rocket_rk3576_wbo_create(fd, K, N, B, &wbo);
        if (rc != 0 || !wbo) {
            printf("      resident-weight create REFUSED rc=%d\n", rc);
            wrong++;
        } else {
            memset(C_w, HOST_STAMP, (size_t)M * N);
            rc = rocket_matmul_int8_rk3576_perc_wbo(fd, M, K, N, A, wbo, bias, scale_n,
                                                    sum_abs_w, C_w, &worst_w);
            if (rc != 0) {
                printf("      resident-weight arm REFUSED rc=%d\n", rc);
                wrong++;
            } else {
                for (size_t i = 0; i < (size_t)M * N; i++) if (C_w[i] != C[i]) diffw++;
                printf("      resident-weight vs per-call %lld of %lld differ\n",
                       diffw, (long long)M * N);
                if (diffw) wrong++;
            }
            setenv("ROCKET_RK3576_BO_POOL", "1", 1);
            memset(C_w, HOST_STAMP, (size_t)M * N);
            rc = rocket_matmul_int8_rk3576_perc_wbo(fd, M, K, N, A, wbo, bias, scale_n,
                                                    sum_abs_w, C_w, &worst_w);
            if (rc == 0) {
                memset(C_w, HOST_STAMP, (size_t)M * N);
                rc = rocket_matmul_int8_rk3576_perc_wbo(fd, M, K, N, A, wbo, bias,
                                                        scale_n, sum_abs_w, C_w,
                                                        &worst_w);
            }
            setenv("ROCKET_RK3576_BO_POOL", "0", 1);
            rocket_rk3576_bo_pool_drain(fd);
            if (rc != 0) {
                printf("      pooled arm REFUSED rc=%d\n", rc);
                wrong++;
            } else {
                for (size_t i = 0; i < (size_t)M * N; i++) if (C_w[i] != C[i]) diffp++;
                printf("      pooled(x2) vs per-call %lld of %lld differ\n",
                       diffp, (long long)M * N);
                if (diffp) wrong++;
            }
            /* Force the narrowest tile so the resident cube's per-tile dma offset is
             * exercised across MANY tiles, not the two the default tiling gives these
             * shapes. The ramp is planned per tile, so the reference is the per-call
             * path at the SAME tiling, not the default-tiling surface above. MM_NT is
             * read per call. */
            setenv("ROCKET_RK3576_MM_NT", "32", 1);
            memset(C_w, HOST_STAMP, (size_t)M * N);
            rc = rocket_matmul_int8_rk3576_perc_sa(fd, M, K, N, A, B, bias, scale_n,
                                                   sum_abs_w, C_w, &worst_w);
            if (rc != 0) {
                printf("      MM_NT=32 reference REFUSED rc=%d\n", rc);
                wrong++;
            } else {
                int8_t *C_nt = malloc((size_t)M * N);
                long long diffnt = 0;
                if (C_nt) {
                    memset(C_nt, HOST_STAMP, (size_t)M * N);
                    rc = rocket_matmul_int8_rk3576_perc_wbo(fd, M, K, N, A, wbo, bias,
                                                            scale_n, sum_abs_w, C_nt,
                                                            &worst_w);
                    if (rc != 0) {
                        printf("      MM_NT=32 resident arm REFUSED rc=%d\n", rc);
                        wrong++;
                    } else {
                        for (size_t i = 0; i < (size_t)M * N; i++)
                            if (C_nt[i] != C_w[i]) diffnt++;
                        printf("      resident vs per-call at MM_NT=32 %lld of %lld "
                               "differ\n", diffnt, (long long)M * N);
                        if (diffnt) wrong++;
                    }
                    free(C_nt);
                } else {
                    wrong++;
                }
            }
            unsetenv("ROCKET_RK3576_MM_NT");
            rocket_rk3576_wbo_free(fd, wbo);
        }
        free(C_w);
    }

    free(C); free(model); free(sum_abs_w); free(tile_bias); free(cmul);
    return wrong ? 1 : 0;
}

static int run_cell(int fd, int M, int K, int N, uint32_t seed, int *scored)
{
    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)N * K);
    int8_t *C = malloc((size_t)M * N);
    int32_t *acc = malloc((size_t)M * N * sizeof *acc);
    int32_t *bias = malloc((size_t)N * sizeof *bias);
    uint32_t s = seed;
    int arm, bad = 0;
    long absmax = 1;

    if (!A || !B || !C || !acc || !bias) {
        fprintf(stderr, "rk3576_mm_requant: out of memory at %dx%dx%d\n", M, K, N);
        free(A); free(B); free(C); free(acc); free(bias);
        return -1;
    }

    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rnd_i8(&s);
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = rnd_i8(&s);
    /* A bias with a real magnitude: the BS stage is part of the expression under test
     * and a zero bias would leave the A term of every coefficient group unexercised. */
    for (int n = 0; n < N; n++) bias[n] = (int32_t)(xs32(&s) % 65536u) - 32768;

    for (int m = 0; m < M; m++) {
        const int8_t *a = A + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const int8_t *b = B + (size_t)n * K;
            int32_t v = bias[n];
            for (int k = 0; k < K; k++) v += (int32_t)a[k] * (int32_t)b[k];
            acc[(size_t)m * N + n] = v;
            if (labs((long)v) > absmax) absmax = labs((long)v);
        }
    }

    /* Two arms of the same cell: a scale that fills the range, and one that clips. */
    for (arm = 0; arm < 2; arm++) {
        float scale = (float)((arm ? 508.0 : 127.0) / (double)absmax);
        unsigned mul, shift;
        long long dev_vs_int = 0, int_vs_flt = 0, dev_vs_flt = 0;
        long long unwritten = 0, clipped = 0, nonzero = 0;
        int maxd_dev_int = 0, maxd_int_flt = 0;
        int rmin = 127, rmax = -128;
        int rc, shown = 0;

        requant_params(scale, &mul, &shift);
        memset(C, HOST_STAMP, (size_t)M * N);
        rc = rocket_matmul_int8_rk3576(fd, M, K, N, A, B, bias, scale, C);
        if (rc != 0) {
            printf("  %4d x %5d x %4d  scale %.6g  REFUSED rc=%d\n", M, K, N, scale, rc);
            bad = 1;
            continue;
        }

        for (size_t i = 0; i < (size_t)M * N; i++) {
            int want_i = requant_apply((int64_t)acc[i], mul, shift);
            int want_f = float_requant(acc[i], scale);
            int got = C[i];
            int d;
            if (want_i < rmin) rmin = want_i;
            if (want_i > rmax) rmax = want_i;
            if (want_i == 127 || want_i == -128) clipped++;
            if (want_i != 0) nonzero++;
            if (got == HOST_STAMP && want_i != HOST_STAMP) unwritten++;
            d = got - want_i;
            if (d) {
                dev_vs_int++;
                if (abs(d) > maxd_dev_int) maxd_dev_int = abs(d);
                if (shown < 6) {
                    printf("    [%zu,%zu] acc=%d  dev=%d  int_model=%d  float_model=%d\n",
                           i / (size_t)N, i % (size_t)N, acc[i], got, want_i, want_f);
                    shown++;
                }
            }
            d = want_i - want_f;
            if (d) {
                int_vs_flt++;
                if (abs(d) > maxd_int_flt) maxd_int_flt = abs(d);
            }
            if (got != want_f) dev_vs_flt++;
        }

        /* A surface that is zero everywhere cannot witness a wrong value, whatever the
         * counts below say. Print the span and refuse to score a degenerate one. */
        if (rmin >= -1 && rmax <= 1) {
            printf("  %4d x %5d x %4d  scale %.6g  DEGENERATE reference (span %d..%d) "
                   "— not scored\n", M, K, N, scale, rmin, rmax);
            bad = 1;
            continue;
        }

        printf("  %4d x %5d x %4d  scale %.6g  mul=%u shift=%u  span %d..%d  "
               "clip %.2f%%  nonzero %.1f%%\n",
               M, K, N, scale, mul, shift, rmin, rmax,
               100.0 * (double)clipped / ((double)M * N),
               100.0 * (double)nonzero / ((double)M * N));
        printf("      dev-vs-int %lld of %lld (%.4f%%, max |d| %d)   "
               "int-vs-float %lld (%.4f%%, max |d| %d)   dev-vs-float %lld   "
               "unwritten %lld\n",
               dev_vs_int, (long long)M * N,
               100.0 * (double)dev_vs_int / ((double)M * N), maxd_dev_int,
               int_vs_flt, 100.0 * (double)int_vs_flt / ((double)M * N), maxd_int_flt,
               dev_vs_flt, unwritten);

        (*scored)++;
        if (dev_vs_int) bad = 1;
    }

    /* The per-column entry, at two column-scale spreads. FLAT is the control: it isolates
     * the ramp's ceiling, since every column then shares the largest C the accumulator
     * bound allows and the resolution is 0.5/C_max alone. The spread arm adds the only
     * other term. Both are scored against the ramp, not against an exact scale. */
    {
        float *scale_n = malloc((size_t)N * sizeof *scale_n);
        static const double spreads[] = { 1.0, 8.0 };
        if (scale_n) {
            for (unsigned si = 0; si < sizeof spreads / sizeof spreads[0]; si++) {
                double S = spreads[si];
                double base = 127.0 / (double)absmax;
                for (int n = 0; n < N; n++) {
                    /* log-spaced across the columns, so the ratio between the extremes
                     * is exactly S however many columns there are. */
                    double t = N > 1 ? (double)n / (double)(N - 1) : 0.5;
                    scale_n[n] = (float)(base * pow(S, t - 0.5));
                }
                if (perc_arm(fd, M, K, N, A, B, bias, acc, scale_n, S) == 0)
                    (*scored)++;
                else
                    bad = 1;
            }
        }
        free(scale_n);
    }

    free(A); free(B); free(C); free(acc); free(bias);
    return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    static const struct cell defaults[] = {
        {  32,  256,  64 },
        {  64,  512, 128 },
        { 128, 1024, 256 },
        { 256, 2048, 512 },
    };
    const struct cell *cells = defaults;
    struct cell one;
    unsigned ncell = sizeof defaults / sizeof defaults[0];
    int fd, fail = 0, scored = 0;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_mm_requant: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    if (argc > 3) {
        one.M = atoi(argv[1]); one.K = atoi(argv[2]); one.N = atoi(argv[3]);
        if (one.M <= 0 || one.K <= 0 || one.N <= 0) {
            fprintf(stderr, "usage: rk3576_mm_requant [M K N]\n");
            return 1;
        }
        cells = &one; ncell = 1;
    }

    fd = rocket_open();
    if (fd < 0) {
        fprintf(stderr, "rk3576_mm_requant: rocket_open failed (%d)\n", fd);
        return 1;
    }

    printf("rk3576_mm_requant: the int8 matmul entry's requant against two host models\n");
    printf("  int model:   sat8(round_half_to_even((acc + bias) * MUL >> SHIFT))\n");
    printf("  float model: sat8(round_half_to_even((acc + bias) * scale))  "
           "[what the accuracy simulations used]\n");

    for (unsigned c = 0; c < ncell; c++) {
        if (run_cell(fd, cells[c].M, cells[c].K, cells[c].N, 0x9E3779B9u + c, &scored))
            fail = 1;
    }

    if (!scored) {
        fprintf(stderr, "rk3576_mm_requant: no cell scored — nothing was measured\n");
        return 1;
    }
    printf("rk3576_mm_requant: %d arms scored, %s\n", scored, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
