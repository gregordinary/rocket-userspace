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
 * AND THE PER-COLUMN ENTRY, on both of its ramps (see the per-output-column arm below):
 * the shift ramp, where the DPU shift word carries the gain and `(acc + bias)*C` is held
 * wide past 2^31, and the shift-0 ramp ROCKET_RK3576_MM_PC_SHIFT=0 restores. Each is
 * scored against its own ramp's host model, and its plan against the float scale.
 *
 * WHAT A GREEN RUN HERE WOULD NOT SHOW. It says the epilogue's ARITHMETIC is modelled.
 * It says nothing about a real model's activation distribution or column-scale spread
 * (these are pseudorandom operands at three chosen spreads), nothing about a product
 * between 2^45.9 and 2^46 or an accumulator past int32 (neither is formed here), and
 * nothing about the drain deadline, whose loss is a DROPPED write rather than a wrong
 * value — the library refuses a task that wrote nothing, so a drop reaches this gate as
 * a refusal.
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
#include "perchannel_model.h"   /* plan_c: the ramp planner, a second implementation */

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

/* THE FLOAT THE EMITTER DERIVES ITS PAIR FROM, on the per-tensor entry and the shift-0
 * ramp: they hand it out_scale = 1/scale and it computes (in*w)/out = 1/(1/scale) in
 * float, which is not `scale` for about 17% of floats and moves the derived (MUL, SHIFT)
 * by one multiplier unit for 0.033% of them. The model takes the pair from this, so it
 * predicts what the part is programmed with. The shift ramp passes its gain as in_scale
 * with unit w and out, which is exact, and does not go through here. */
static float emitter_rt(float scale)
{
    volatile float one = 1.0f, inv = one / scale;
    return (one * one) / inv;
}
static long g_rt_moved;   /* tiles or scales whose pair the round trip moved */

/* ---- the per-output-column arm -------------------------------------------
 *
 * The per-column entry carries its scale on the coefficient group's int16 C ramp over a
 * shared (MUL, SHIFT), so the model here is the ramp and not an exact scale. It plans the
 * ramp with perchannel_model.h, a second implementation of the library's planner, and
 * applies the epilogue the part runs. The question is whether the DEVICE implements the
 * epilogue the planner's RULE assumes; a model that called the entry's own planner would
 * agree with a planner defect by construction.
 *
 * TWO RAMPS, one arm each, selected by ROCKET_RK3576_MM_PC_SHIFT (read per call):
 *
 *   shift  (default) plan_c_bs with every column live: the tile's largest scale at
 *          C = 32767, the DPU shift word `bs` the smallest at which every product fits
 *          int32 at its bound, and the epilogue sat32(rne(((acc + bias)*C) >> bs)) then
 *          the shared OUT_CVT. The product is held wide, so this arm's cells pass 2^31 on
 *          both signs, and the bias-heavy cell takes it to ~2^45.9, next to the int32 x
 *          int16 maximum of 2^46.
 *   shift0 (MM_PC_SHIFT=0) plan_c: C capped at INT32_MAX over the column's accumulator
 *          bound, no shift. At the wide spread its smallest columns clamp to C = 1.
 *
 * THE PLAN IS ALSO SCORED AGAINST THE FLOAT, not only the surface against the plan: every
 * unclamped column's delivered gain C*MUL/2^(SHIFT+bs) must sit within half a unit of C of
 * its own scale, which is a statement about the float and shares nothing with the planner
 * (a host model that calls the library's derivation agrees with the part even when both
 * are at the wrong value). And the entry's own `worst_rel_err` must equal the model's,
 * which is the plan compared rather than only the surface it happens to produce.
 *
 * The tile boundary matters and is not visible in a total: the (MUL, SHIFT) is per N
 * tile, so the model has to plan tile by tile exactly as the entry does. The tile width
 * comes from the pure planner rather than from a constant, and the pure planner reads
 * ROCKET_RK3576_MM_NT, so a forced tiling is modelled at the tiling it forces.
 */
enum { RAMP_SHIFT0 = 0, RAMP_SHIFT = 1 };

static void set_ramp(int ramp)
{
    if (ramp == RAMP_SHIFT0) setenv("ROCKET_RK3576_MM_PC_SHIFT", "0", 1);
    else                     unsetenv("ROCKET_RK3576_MM_PC_SHIFT");
}

/* What the gate must have exercised before it may pass. A coverage clause that never
 * fired is an arm that checked nothing, and it would read as green. */
static struct {
    int  shift_pos, shift_neg;       /* a shift-ramp arm with products past +-2^31  */
    int  wide_pos, wide_neg;         /* products past +-2^45.5 on the shift ramp     */
    int  old_clamp;                  /* a shift-0 arm with a clamped column          */
    int  multitile;                  /* a scored arm with >= 4 N tiles               */
} g_cov;

static int perc_arm(int fd, int M, int K, int N, const int8_t *A, const int8_t *B,
                    const int32_t *bias, const int32_t *acc, const float *scale_n,
                    double spread, int ramp, int full)
{
    int8_t *C = malloc((size_t)M * N);
    int8_t *model = malloc((size_t)M * N);
    int64_t *sum_abs_w = calloc((size_t)N, sizeof *sum_abs_w);
    int32_t *bias_n = calloc((size_t)N, sizeof *bias_n);   /* plan_c indexes the whole layer */
    int16_t *cmul = calloc((size_t)N, sizeof *cmul);
    int *cst = calloc((size_t)N, sizeof *cst);
    double worst = -1.0, worst_sa = -1.0, model_worst = 0.0, pmax = 0.0;
    long long wrong = 0, vs_exact = 0, ppos = 0, pneg = 0;
    int maxd = 0, maxd_exact = 0, nt = 0, rc, shown = 0, bad = 0, ntiles = 0;
    int nclamp = 0, nfloat_bad = 0, bsmin = 99, bsmax = -1, rmin = 127, rmax = -128;
    const char *rname = ramp == RAMP_SHIFT ? "shift " : "shift0";

    if (!C || !model || !sum_abs_w || !bias_n || !cmul || !cst) {
        free(C); free(model); free(sum_abs_w); free(bias_n); free(cmul); free(cst);
        return -1;
    }
    if (bias) memcpy(bias_n, bias, (size_t)N * sizeof *bias_n);
    set_ramp(ramp);

    for (int n = 0; n < N; n++) {
        const int8_t *w = B + (size_t)n * K;
        int64_t s = 0;
        for (int k = 0; k < K; k++) s += w[k] < 0 ? -(int64_t)w[k] : (int64_t)w[k];
        sum_abs_w[n] = s;
    }
    if (rocket_matmul_plan_int8_rk3576(M, K, N, NULL, NULL, &nt) < 0 || nt <= 0) {
        printf("  perc %s %4d x %5d x %4d  no plan — not scored\n", rname, M, K, N);
        free(C); free(model); free(sum_abs_w); free(bias_n); free(cmul); free(cst);
        return 1;
    }

    /* Plan every tile the way the entry does, and requant this tile's columns. The tile
     * width is the pure planner's; the ramp inside it is the model's. */
    for (int n0 = 0; n0 < N; n0 += nt) {
        unsigned tile_n = (unsigned)(N - n0 < nt ? N - n0 : nt);
        unsigned mul, shift, bs = 0;
        double gain, base;

        ntiles++;
        for (unsigned j = 0; j < tile_n; j++)
            if (!(scale_n[n0 + j] > 0.0f)) bad = 1;
        if (bad) break;
        if (ramp == RAMP_SHIFT)
            plan_c_bs((unsigned)n0, tile_n, NULL, NULL, sum_abs_w, bias_n, 1.0f, scale_n,
                      1.0f, 0, cmul, cst, &mul, &shift, &bs);
        else {
            /* plan_c builds C against requant_params(best); the device's OUT_CVT carries
             * the pair of 1/(1/best). `best` is spelled again here, from plan_c's rule. */
            double best = 0.0;
            unsigned mul_d, shift_d;
            plan_c((unsigned)n0, tile_n, NULL, sum_abs_w, bias_n, 1.0f, scale_n, 1.0f,
                   cmul, &mul, &shift);
            for (unsigned j = 0; j < tile_n; j++) {
                double bound = 128.0 * (double)sum_abs_w[n0 + j] +
                               fabs((double)bias_n[n0 + j]) + 1.0;
                double cmax = (double)INT32_MAX / bound;
                if (cmax > 32767.0) cmax = 32767.0;
                if (cmax < 1.0)     cmax = 1.0;
                if ((double)scale_n[n0 + j] / cmax > best)
                    best = (double)scale_n[n0 + j] / cmax;
            }
            requant_params(emitter_rt((float)best), &mul_d, &shift_d);
            if (mul_d != mul || shift_d != shift) {
                g_rt_moved++;
                printf("    note: tile n0=%d, the reciprocal round trip moves the OUT_CVT "
                       "pair %u/%u -> %u/%u (modelled)\n", n0, mul, shift, mul_d, shift_d);
                mul = mul_d; shift = shift_d;
            }
        }
        if ((int)bs < bsmin) bsmin = (int)bs;
        if ((int)bs > bsmax) bsmax = (int)bs;
        gain = (double)mul / (double)((uint64_t)1 << shift);
        base = gain / ldexp(1.0, (int)bs);          /* the gain one unit of C carries */
        for (unsigned j = 0; j < tile_n; j++) {
            double cs = (double)scale_n[n0 + j];
            double err = fabs((double)cmul[n0 + j] * base - cs) / cs;
            /* The entry's worst_rel_err counts a clamped column too, so this does. */
            if (err > model_worst) model_worst = err;
            if (cs / base < 0.5) { nclamp++; continue; }   /* off the ramp: C = 1 */
            /* Against the float: an unclamped C is the nearest integer to cs/base, so the
             * gain is within half a unit of it. */
            if (fabs((double)cmul[n0 + j] * base - cs) > 0.5 * base * (1.0 + 1e-9))
                nfloat_bad++;
        }
        for (int m = 0; m < M; m++)
            for (unsigned j = 0; j < tile_n; j++) {
                int64_t a = acc[(size_t)m * N + n0 + j];
                double p = (double)a * (double)cmul[n0 + j];
                int v = epilogue_bs(a, cmul[n0 + j], bs, mul, shift, 0);
                if (p >  2147483647.0) ppos++;
                if (p < -2147483648.0) pneg++;
                if (fabs(p) > pmax) pmax = fabs(p);
                if (ramp == RAMP_SHIFT && p >  3.2e13) g_cov.wide_pos = 1;   /* 2^44.9 */
                if (ramp == RAMP_SHIFT && p < -3.2e13) g_cov.wide_neg = 1;
                if (v < rmin) rmin = v;
                if (v > rmax) rmax = v;
                model[(size_t)m * N + n0 + j] = (int8_t)v;
            }
    }
    if (bad) {
        free(C); free(model); free(sum_abs_w); free(bias_n); free(cmul); free(cst);
        return -1;
    }

    memset(C, HOST_STAMP, (size_t)M * N);
    rc = rocket_matmul_int8_rk3576_perc(fd, M, K, N, A, B, bias, scale_n, C, &worst);
    if (rc != 0) {
        printf("  perc %s %4d x %5d x %4d  REFUSED rc=%d\n", rname, M, K, N, rc);
        free(C); free(model); free(sum_abs_w); free(bias_n); free(cmul); free(cst);
        set_ramp(RAMP_SHIFT);
        return 1;
    }

    for (size_t i = 0; i < (size_t)M * N; i++) {
        int d = C[i] - model[i];
        int e = C[i] - float_requant(acc[i], scale_n[i % (size_t)N]);
        if (d) {
            wrong++;
            if (abs(d) > maxd) maxd = abs(d);
            if (shown < 6) {
                printf("    [%zu,%zu] acc=%d  C=%d  dev=%d  ramp_model=%d\n",
                       i / (size_t)N, i % (size_t)N, acc[i], cmul[i % (size_t)N], C[i],
                       model[i]);
                shown++;
            }
        }
        if (e) { vs_exact++; if (abs(e) > maxd_exact) maxd_exact = abs(e); }
    }

    printf("  perc %s %4d x %5d x %4d  Ntile %d x%d  spread %.1fx  bs %d..%d  "
           "worst_rel_err %.5f%% (model %.5f%%)  clamped %d\n", rname, M, K, N, nt, ntiles,
           spread, bsmin, bsmax, 100.0 * worst, 100.0 * model_worst, nclamp);
    printf("      dev-vs-ramp %lld of %lld (max |d| %d)   dev-vs-EXACT-per-column %lld "
           "(%.3f%%, max |d| %d)   span %d..%d\n",
           wrong, (long long)M * N, maxd, vs_exact,
           100.0 * (double)vs_exact / ((double)M * N), maxd_exact, rmin, rmax);
    printf("      products past +2^31 %lld, past -2^31 %lld, max |(acc+bias)*C| 2^%.2f\n",
           ppos, pneg, pmax > 0.0 ? log2(pmax) : 0.0);

    /* The plan, compared: the entry reports its delivered worst gain error, the model
     * computes its own. They are the same number when the two planners agree. A clamped
     * column's error is included in both, so a shift-0 arm with a clamp reports one above
     * 100%. */
    if (fabs(worst - model_worst) > 1e-9 + 1e-6 * model_worst) {
        printf("      FAIL: the entry's worst_rel_err %.9g is not the model's %.9g — the "
               "plans differ\n", worst, model_worst);
        wrong++;
    }
    if (nfloat_bad) {
        printf("      FAIL: %d column(s) sit more than half a unit of C off their own "
               "float scale\n", nfloat_bad);
        wrong++;
    }
    if (rmin >= -1 && rmax <= 1) {
        printf("      FAIL: DEGENERATE reference (span %d..%d) — cannot witness a wrong "
               "value\n", rmin, rmax);
        wrong++;
    }
    if (ramp == RAMP_SHIFT && ppos && pneg) { g_cov.shift_pos = 1; g_cov.shift_neg = 1; }
    if (ramp == RAMP_SHIFT0 && nclamp) g_cov.old_clamp = 1;
    if (ntiles >= 4) g_cov.multitile = 1;

    if (!full) goto out;

    /* THE SUPPLIED-SUM ENTRY IS THE SAME ENTRY. `_perc_sa` skips the O(N*K) pass over B
     * and takes the caller's per-column sums instead; the sums handed over here are the
     * ones this gate computed for its own model, so the two surfaces must be identical
     * byte for byte. A difference is not a tolerance question — it means the supplied
     * array is not the quantity the entry's own pass produces, which is the one way this
     * parameter can be wrong without any caller noticing. */
    {
        int8_t *C_sa = malloc((size_t)M * N);
        long long diff = 0;
        if (!C_sa) { wrong++; goto out; }
        memset(C_sa, HOST_STAMP, (size_t)M * N);
        rc = rocket_matmul_int8_rk3576_perc_sa(fd, M, K, N, A, B, bias, scale_n,
                                               sum_abs_w, C_sa, &worst_sa);
        if (rc != 0) {
            printf("      supplied-sum arm REFUSED rc=%d\n", rc);
            wrong++;
        } else {
            for (size_t i = 0; i < (size_t)M * N; i++) if (C_sa[i] != C[i]) diff++;
            printf("      supplied-sum vs computed-sum %lld of %lld differ  "
                   "worst_rel_err %.5f%%\n", diff, (long long)M * N, 100.0 * worst_sa);
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
        const char *nt_was = getenv("ROCKET_RK3576_MM_NT");
        char nt_buf[32] = "";
        if (!C_w) { wrong++; goto out; }
        if (nt_was) snprintf(nt_buf, sizeof nt_buf, "%s", nt_was);
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
            if (nt_buf[0]) setenv("ROCKET_RK3576_MM_NT", nt_buf, 1);
            else           unsetenv("ROCKET_RK3576_MM_NT");
            rocket_rk3576_wbo_free(fd, wbo);
        }
        free(C_w);
    }

out:
    set_ramp(RAMP_SHIFT);
    free(C); free(model); free(sum_abs_w); free(bias_n); free(cmul); free(cst);
    return wrong ? 1 : 0;
}

/* BIAS_WIDE puts every column's bias near +-2^31 (the sign alternating by column), so the
 * shift ramp's C = 32767 takes (acc + bias)*C to ~2^45.9 on both signs: the widest product
 * an int32 accumulator and an int16 C can form is 2^46, and an earlier sweep read the product exact
 * only to 3.3e13 (2^44.9), positive side. The operands stay small enough that acc + bias
 * cannot leave int32. */
enum { BIAS_NORMAL = 0, BIAS_WIDE = 1 };

static int run_cell(int fd, int M, int K, int N, uint32_t seed, int bias_mode,
                    int *scored)
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
    for (int n = 0; n < N; n++) {
        if (bias_mode == BIAS_WIDE) {
            int32_t mag = 2080374784 - (int32_t)(xs32(&s) % 1048576u);   /* 2^31 - 2^26 - j */
            bias[n] = (n & 1) ? -mag : mag;
        } else {
            bias[n] = (int32_t)(xs32(&s) % 65536u) - 32768;
        }
    }

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

        requant_params(emitter_rt(scale), &mul, &shift);
        {
            unsigned m0, s0;
            requant_params(scale, &m0, &s0);
            if (m0 != mul || s0 != shift) {
                g_rt_moved++;
                printf("    note: scale %.9g, the reciprocal round trip moves the OUT_CVT "
                       "pair %u/%u -> %u/%u (modelled)\n", scale, m0, s0, mul, shift);
            }
        }
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

    /* The per-column entry, at three column-scale spreads and both ramps. FLAT is the
     * control: it isolates the ramp's ceiling, since every column then shares the largest
     * C the ramp allows and the resolution is 0.5/C_max alone. 8x is a real model's
     * within-tile median. 1000x is past what the shift-0 ramp can span at these depths
     * (its C_max is INT32_MAX / (128*sum|B|), 1000 at K=256 and 130 at K=2048 for these
     * full-range operands), so there its smallest columns clamp, where the shift ramp
     * still resolves them. Every arm is scored against its own ramp's model, not against
     * an exact scale. The bias-heavy cell runs the two narrow spreads, since a clamp is
     * not what it is for. */
    {
        float *scale_n = malloc((size_t)N * sizeof *scale_n);
        static const double spreads[] = { 1.0, 8.0, 1000.0 };
        unsigned nsp = bias_mode == BIAS_WIDE ? 2u : 3u;
        /* The bias-heavy cell's outputs sit near +-60 rather than on the rail, so the
         * bias term is witnessed by an unsaturated byte on both signs. */
        double fill = bias_mode == BIAS_WIDE ? 60.0 : 127.0;
        if (scale_n) {
            for (unsigned si = 0; si < nsp; si++) {
                double S = spreads[si];
                double base = fill / (double)absmax;
                for (int n = 0; n < N; n++) {
                    /* log-spaced across the columns, so the ratio between the extremes
                     * is exactly S however many columns there are. */
                    double t = N > 1 ? (double)n / (double)(N - 1) : 0.5;
                    scale_n[n] = (float)(base * pow(S, t - 0.5));
                }
                for (int ramp = RAMP_SHIFT; ramp >= RAMP_SHIFT0; ramp--) {
                    if (perc_arm(fd, M, K, N, A, B, bias, acc, scale_n, S, ramp, 1) == 0)
                        (*scored)++;
                    else
                        bad = 1;
                }
            }
            /* A FORCED MULTI-TILE RUN, scored against the model at the tiling it forces:
             * the (MUL, SHIFT) and the shift word are per tile, so a model that planned
             * the whole N as one tile would agree with the part only by luck. 64 columns
             * a tile gives these cells two to eight tiles. */
            if (bias_mode == BIAS_NORMAL && N >= 256) {
                double S = 8.0, base = fill / (double)absmax;
                for (int n = 0; n < N; n++) {
                    double t = N > 1 ? (double)n / (double)(N - 1) : 0.5;
                    scale_n[n] = (float)(base * pow(S, t - 0.5));
                }
                setenv("ROCKET_RK3576_MM_NT", "64", 1);
                for (int ramp = RAMP_SHIFT; ramp >= RAMP_SHIFT0; ramp--) {
                    if (perc_arm(fd, M, K, N, A, B, bias, acc, scale_n, S, ramp, 0) == 0)
                        (*scored)++;
                    else
                        bad = 1;
                }
                unsetenv("ROCKET_RK3576_MM_NT");
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
    printf("  per-column:  sat8(rne(sat32(rne((acc + bias)*C >> bs)) * MUL >> SHIFT)), "
           "the shift ramp (bs from the plan) and the shift-0 ramp (bs = 0)\n");

    for (unsigned c = 0; c < ncell; c++) {
        if (run_cell(fd, cells[c].M, cells[c].K, cells[c].N, 0x9E3779B9u + c, BIAS_NORMAL,
                     &scored))
            fail = 1;
    }
    if (ncell > 1) {
        printf("the bias-heavy cell: |bias| ~ 2^31 - 2^26, alternating sign by column\n");
        if (run_cell(fd, 32, 256, 64, 0x2545F491u, BIAS_WIDE, &scored))
            fail = 1;
    }

    if (!scored) {
        fprintf(stderr, "rk3576_mm_requant: no cell scored — nothing was measured\n");
        return 1;
    }
    /* THE GATE FAILS WHAT IT DID NOT EXERCISE. Every clause below is a regime the new
     * ramp depends on and a pass that never reached it checked nothing. A single-cell run
     * is a probe and is not held to them. */
    if (ncell > 1) {
        if (!g_cov.shift_pos || !g_cov.shift_neg) {
            printf("rk3576_mm_requant: no shift-ramp arm put products past 2^31 on both "
                   "signs\n");
            fail = 1;
        }
        if (!g_cov.wide_pos || !g_cov.wide_neg) {
            printf("rk3576_mm_requant: the bias-heavy cell did not reach 2^44.9 on both "
                   "signs (pos %d, neg %d)\n", g_cov.wide_pos, g_cov.wide_neg);
            fail = 1;
        }
        if (!g_cov.old_clamp) {
            printf("rk3576_mm_requant: no shift-0 arm clamped a column — the wide spread "
                   "did not reach the old cap\n");
            fail = 1;
        }
        if (!g_cov.multitile) {
            printf("rk3576_mm_requant: no scored arm ran four or more N tiles\n");
            fail = 1;
        }
    }
    printf("rk3576_mm_requant: %ld pair(s) moved by the reciprocal round trip\n",
           g_rt_moved);
    printf("rk3576_mm_requant: %d arms scored, %s\n", scored, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
