// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_w8a8_route.c — does the W8A8 calibration route COMPOSE on the part?
 *
 * The RK3576 W8A8 GEMM route ends in rocket_matmul_int8_rk3576_perc(), whose per-column
 * output scale the caller has to supply and cannot compute: the scale a column wants is
 * set by that column's accumulator, and the accumulator never leaves the part. The
 * frontend's answer is a two-pass BOOTSTRAP that reads the frozen colmax off the part's
 * own int8 output —
 *
 *   pass 1  run at 127/(128*sum_k|B[n][k]| + 1), the analytic accumulator bound. It
 *           overshoots a real column by ~60x, so it cannot saturate and every column
 *           comes back on scale, at a couple of codes.
 *   pass 2  run at pass one's readback times a margin (1.5), so the codes are spent.
 *   freeze  colmax = the running max over calibration windows, and the shipped scale is
 *           127/(colmax * CALSAFE).
 *   run     the frozen call, then de-quantize by the scale ASKED for.
 *
 * Every number behind that route is HOST arithmetic — a simulator over two language
 * models. Nothing in it had ever run on the part. This gate runs it.
 *
 * THREE QUANTITIES, and each answers a different question:
 *
 *   SHIFT       what the ramp planner actually programs on pass one. The concern the
 *               route was written under is that a ~6e-8 scale drives the OUT_CVT shift
 *               out of DPU 0x40B4's six-bit field. It does not, and it cannot: the
 *               planner divides the scale it is handed by a cmax carrying the SAME
 *               accumulator bound, so the two cancel and best_base is the constant
 *               127/INT32_MAX whatever the weights are. This gate reads the number back
 *               off rocket_rk3576_plan_perchannel() rather than asserting the algebra.
 *
 *   est/true    the bootstrap's per-column error against the exact int32 accumulator,
 *               computed here on the host because this test — unlike a frontend — has
 *               both. Reported as |est/true - 1|, ABSOLUTE and per column: pass one's
 *               signed median lands on the margin you put in and looks exact while its
 *               own per-column error is enormous, so the signed form cannot score it.
 *
 *   composed    the de-quantized surface against the exact product. This is the route's
 *               end-to-end arithmetic on the part, at the shipped safety factor.
 *
 * WHAT THE M=1 2x2 SETTLED, and it inverts the natural reading: freezing from the EXACT
 * per-column accumulator max is 13x WORSE at M=1 than freezing from the bootstrap
 * (composed RMS 201% against 15.3%, K=1536 N=2048). The bootstrap's readback floors at
 * one code, which bounds a column's colmax from BELOW and so bounds scale_n from above;
 * the exact max does not, so a column whose single accumulator draw is near zero is
 * handed an enormous scale, and best_base = max_j(cs_j/cmax_j) is taken over the WHOLE
 * tile — one such column costs every other column in the tile its C resolution. The
 * over-estimate that reads as the estimator's worst per-column error is the same thing
 * that keeps the shared ramp usable. Forcing a smaller output-channel tile moves it
 * monotonically and does not close it (15.3 -> 15.0 -> 10.7% at NT 2048/256/32), which
 * is the per-tile lever the entry's header describes, measured on this route.
 *
 * WHAT A GREEN RUN HERE WOULD NOT SHOW. The operands are pseudorandom, so this says
 * nothing about a real activation distribution — and nothing about CALIBRATION, since
 * the window frozen from is the window then measured. It holds the rotation constant by
 * omitting it: the Hadamard's job is to make a real activation Gaussian and these
 * already are, so the block-diagonal construction at K=1536 is NOT scored here (it lives
 * in the frontend, and only a ggml build reaches it). It drives one N tile order — the
 * entry does not sort — and one calibration window per weight.
 *
 * THE COLUMN SPREAD IS DELIBERATE. Row n's weights are drawn at a per-row sigma spanning
 * a decade and quantized against ONE shared scale, so sum|w| — and with it each column's
 * accumulator bound, its C ceiling and its share of the tile's ramp — varies by ~30x
 * across the columns of a single tile. Rows drawn at a common sigma give a flat spread
 * that the ramp cannot be scored on.
 *
 * Run with `sudo -E`, pinned:
 *   sudo -E taskset -c 4 ./build/rk3576_w8a8_route
 *
 * A trailing argument list runs a single cell instead:
 *   sudo -E taskset -c 4 ./build/rk3576_w8a8_route M K N
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

/* The frontend's shipped constants, named here so a divergence is visible rather than
 * buried in a literal. ggml-rocket's entry reads them from ROCKET_RK3576_CALSAFE /
 * ROCKET_RK3576_BOOTMARGIN with these defaults. */
#define ROUTE_CALSAFE     3.0
#define ROUTE_BOOTMARGIN  1.5

/* An element the entry's de-scatter never reached. Every legitimate result is an int8,
 * so there is no free value — the count still holding it is reported separately. */
#define HOST_STAMP ((int8_t)0x5A)

static uint32_t xs32(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return (*s = x);
}

/* Box-Muller off the same xorshift, so a cell is reproducible from its seed. */
static double xs_gauss(uint32_t *s)
{
    double u1 = (xs32(s) + 1.0) / 4294967297.0;
    double u2 = (xs32(s) + 1.0) / 4294967297.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

/* One column's largest |C8| read back as an estimate of that column's accumulator, and
 * whether it saturated. This is the frontend's rk76_bootstrap_pass(), which lives in
 * ggml-rocket.cpp as a static and cannot be linked from here — the arithmetic is three
 * lines and is restated rather than shared, so any drift is between THIS comment and
 * that function. */
static void bootstrap_read(const int8_t *C8, int M, int N, const float *scale_n,
                           double *est, char *satcol)
{
    int n, m;
    for (n = 0; n < N; n++) {
        int mx = 0, sat = 0;
        for (m = 0; m < M; m++) {
            int v = C8[(size_t)m * N + n];
            if (v < 0) v = -v;
            if (v > mx) mx = v;
            if (v >= 127) sat = 1;
        }
        if (mx < 1) mx = 1;   /* under half a code at this scale; credit it one */
        est[n] = (double)mx / (double)scale_n[n];
        if (satcol) satcol[n] = (char)sat;
    }
}

/* Freeze from the exact accumulator instead of from the bootstrap. Only the 2x2 below
 * sets it; a frontend has no such quantity, which is the whole reason for the route. */
static int g_use_oracle = 0;

struct cell_result {
    int    rc;
    double est_med, est_worst;      /* |est/true - 1| over the columns          */
    double p1_med_signed;           /* pass one's SIGNED median, the shape to distrust */
    double p1_med_abs;
    double rms_rel;                 /* composed: RMS(dequant - exact)/RMS(exact) */
    double worst_rel_err;           /* the entry's own integer-ramp resolution   */
    double sat_frac;                /* fraction of the frozen surface at +-127   */
    unsigned shift_p1, shift_frozen;
    double  base_p1;
    long    stamped;
};

static int run_cell(int fd, int M, int K, int N, uint32_t seed,
                    struct cell_result *out)
{
    int8_t  *A = NULL, *B = NULL, *C8 = NULL;
    int32_t *acc = NULL;
    double  *truemax = NULL, *est1 = NULL, *est2 = NULL, *colmax = NULL, *err = NULL;
    float   *sc1 = NULL, *sc2 = NULL, *scf = NULL;
    int64_t *sumw = NULL;
    char    *sat = NULL;
    int rc = -1, n, m, k;
    uint32_t s = seed;

    memset(out, 0, sizeof *out);
    out->rc = -1;

    A   = malloc((size_t)M * K);
    B   = malloc((size_t)N * K);
    C8  = malloc((size_t)M * N);
    acc = malloc((size_t)M * N * sizeof *acc);
    truemax = calloc((size_t)N, sizeof *truemax);
    est1 = calloc((size_t)N, sizeof *est1);
    est2 = calloc((size_t)N, sizeof *est2);
    colmax = calloc((size_t)N, sizeof *colmax);
    err  = calloc((size_t)N, sizeof *err);
    sc1  = calloc((size_t)N, sizeof *sc1);
    sc2  = calloc((size_t)N, sizeof *sc2);
    scf  = calloc((size_t)N, sizeof *scf);
    sumw = calloc((size_t)N, sizeof *sumw);
    sat  = calloc((size_t)N, 1);
    if (!A || !B || !C8 || !acc || !truemax || !est1 || !est2 || !colmax || !err ||
        !sc1 || !sc2 || !scf || !sumw || !sat) goto done;

    /* Activations: one per-tensor scale, which is the route's measured choice. */
    for (m = 0; m < M; m++)
        for (k = 0; k < K; k++) {
            double v = xs_gauss(&s) * 32.0;
            long q = lround(v);
            A[(size_t)m * K + k] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
        }
    /* Weights: a per-row sigma over a decade against ONE shared scale, so the columns of
     * a tile carry a real spread of sum|w|. */
    for (n = 0; n < N; n++) {
        double sigma = 3.0 * pow(30.0, (double)(xs32(&s) % 1024u) / 1023.0);
        for (k = 0; k < K; k++) {
            double v = xs_gauss(&s) * sigma;
            long q = lround(v);
            B[(size_t)n * K + k] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
        }
    }
    for (n = 0; n < N; n++) {
        const int8_t *w = B + (size_t)n * K;
        int64_t t = 0;
        for (k = 0; k < K; k++) t += w[k] < 0 ? -(int64_t)w[k] : (int64_t)w[k];
        sumw[n] = t;
        sc1[n] = (float)(127.0 / (128.0 * (double)t + 1.0));
    }

    /* The exact accumulator, so the bootstrap's estimate has something to be scored
     * against. A frontend has no such reference — that is the whole reason the route
     * bootstraps rather than reading it. */
    for (m = 0; m < M; m++) {
        const int8_t *a = A + (size_t)m * K;
        for (n = 0; n < N; n++) {
            const int8_t *w = B + (size_t)n * K;
            int32_t t = 0;
            for (k = 0; k < K; k++) t += (int32_t)a[k] * (int32_t)w[k];
            acc[(size_t)m * N + n] = t;
        }
    }
    for (n = 0; n < N; n++) {
        double mx = 0.0;
        for (m = 0; m < M; m++) {
            double v = fabs((double)acc[(size_t)m * N + n]);
            if (v > mx) mx = v;
        }
        truemax[n] = mx > 0.0 ? mx : 1.0;
    }

    /* What the emitter will actually program on pass one, read back off the planner
     * rather than re-derived: the DPU shift field is six bits and this is the number
     * the route was written in doubt of. */
    {
        int16_t *cm = calloc((size_t)N, sizeof *cm);
        int32_t *bz = calloc((size_t)N, sizeof *bz);
        float base = 0.0f;
        double e = 0.0;
        unsigned mul = 0, sh = 0;
        if (cm && bz &&
            rocket_rk3576_plan_perchannel("route probe", 0, (unsigned)N, (unsigned)N,
                                          bz, sumw, 1.0f, sc1, 1.0f, NULL,
                                          cm, &base, &e) == 0) {
            rocket_rk3576_requant_params(base, &mul, &sh);
            out->shift_p1 = sh;
            out->base_p1  = (double)base;
        }
        free(cm); free(bz);
    }

    memset(C8, HOST_STAMP, (size_t)M * N);
    rc = rocket_matmul_int8_rk3576_perc(fd, M, K, N, A, B, NULL, sc1, C8, NULL);
    if (rc != 0) { fprintf(stderr, "  pass 1 refused: rc=%d\n", rc); goto done; }
    bootstrap_read(C8, M, N, sc1, est1, NULL);

    for (n = 0; n < N; n++)
        sc2[n] = (float)(127.0 / (est1[n] * ROUTE_BOOTMARGIN));
    memset(C8, HOST_STAMP, (size_t)M * N);
    rc = rocket_matmul_int8_rk3576_perc(fd, M, K, N, A, B, NULL, sc2, C8, NULL);
    if (rc != 0) { fprintf(stderr, "  pass 2 refused: rc=%d\n", rc); goto done; }
    bootstrap_read(C8, M, N, sc2, est2, sat);

    for (n = 0; n < N; n++)
        colmax[n] = sat[n] ? est1[n] * ROUTE_BOOTMARGIN : est2[n];
    /* Scored below either way, so the oracle arm still reports the bootstrap's error. */
    for (n = 0; n < N; n++) est2[n] = colmax[n];
    if (g_use_oracle)
        for (n = 0; n < N; n++) colmax[n] = truemax[n];

    /* Pass one alone, scored the same way, because its SIGNED median is the statistic
     * that misreads it as exact. */
    for (n = 0; n < N; n++) err[n] = est1[n] / truemax[n];
    qsort(err, (size_t)N, sizeof *err, cmp_dbl);
    out->p1_med_signed = err[N / 2];
    for (n = 0; n < N; n++) err[n] = fabs(est1[n] / truemax[n] - 1.0);
    qsort(err, (size_t)N, sizeof *err, cmp_dbl);
    out->p1_med_abs = err[N / 2];

    for (n = 0; n < N; n++) err[n] = fabs(est2[n] / truemax[n] - 1.0);
    qsort(err, (size_t)N, sizeof *err, cmp_dbl);
    out->est_med   = err[N / 2];
    out->est_worst = err[N - 1];

    /* The frozen call, at the shipped safety factor. */
    for (n = 0; n < N; n++)
        scf[n] = (float)(127.0 / (colmax[n] * ROUTE_CALSAFE));
    {
        int16_t *cm = calloc((size_t)N, sizeof *cm);
        int32_t *bz = calloc((size_t)N, sizeof *bz);
        float base = 0.0f;
        double e = 0.0;
        unsigned mul = 0, sh = 0;
        if (cm && bz &&
            rocket_rk3576_plan_perchannel("route probe", 0, (unsigned)N, (unsigned)N,
                                          bz, sumw, 1.0f, scf, 1.0f, NULL,
                                          cm, &base, &e) == 0) {
            rocket_rk3576_requant_params(base, &mul, &sh);
            out->shift_frozen = sh;
        }
        free(cm); free(bz);
    }
    memset(C8, HOST_STAMP, (size_t)M * N);
    rc = rocket_matmul_int8_rk3576_perc(fd, M, K, N, A, B, NULL, scf, C8,
                                        &out->worst_rel_err);
    if (rc != 0) { fprintf(stderr, "  frozen call refused: rc=%d\n", rc); goto done; }

    /* Composed: de-quantize by the scale ASKED for, which is what the entry's caller
     * does, and score it against the exact product. */
    {
        double se = 0.0, ss = 0.0;
        long sats = 0, stamped = 0;
        for (m = 0; m < M; m++)
            for (n = 0; n < N; n++) {
                int v = C8[(size_t)m * N + n];
                double got = (double)v / (double)scf[n];
                double want = (double)acc[(size_t)m * N + n];
                if (v >= 127 || v <= -127) sats++;
                if ((int8_t)v == HOST_STAMP) stamped++;
                se += (got - want) * (got - want);
                ss += want * want;
            }
        out->rms_rel  = ss > 0.0 ? sqrt(se / ss) : 0.0;
        out->sat_frac = (double)sats / ((double)M * N);
        out->stamped  = stamped;
    }
    out->rc = 0;
    rc = 0;

done:
    free(A); free(B); free(C8); free(acc); free(truemax); free(est1); free(est2);
    free(colmax); free(err); free(sc1); free(sc2); free(scf); free(sumw); free(sat);
    return rc;
}

/* The gate's contract. Loose against the predicted magnitudes on purpose: what this row
 * scores is that the route COMPUTES on the part and that its estimator is on the order
 * the host arm measured, not a reproduction of that arm's third digit.
 *
 * THE COMPOSED BOUND IS A FUNCTION OF M, and that is a measurement rather than a
 * concession. A column's frozen colmax is a max over M accumulator rows, so at a prefill
 * M it is a tight statistic and at M=1 it is one Gaussian draw — and the entry plans ONE
 * (MUL, SHIFT) per output-channel tile, so the spread of scale_n across a tile is what
 * sets every column's C resolution. Measured on the part: 2.1-2.2% at M=256/512 against
 * 15.3-18.1% at M=1, same K, same weights. ggml-rocket does not route M below
 * rocket_min_m() (128), so the M=1 cells here are the library entry's envelope and not a
 * frontend path; they are held to a bound that a regression would still trip. */
#define EST_MED_MAX      0.05
#define RMS_REL_MAX      0.10    /* M >= 128 */
#define RMS_REL_MAX_M1   0.25    /* M < 128, where the tile's scale spread is the term */
#define RMS_M_PREFILL    128

int main(int argc, char **argv)
{
    static const struct { int M, K, N; uint32_t seed; } cells[] = {
        {   1, 1536, 2048, 0x51ce0001u },   /* Qwen2.5-1.5B's K, decode row count      */
        { 512, 1536, 2048, 0x51ce0002u },   /* the same K at a prefill M               */
        {   1, 2048, 2048, 0x51ce0003u },   /* a power-of-two K, M=1                   */
        { 256, 2048, 1024, 0x51ce0004u },   /* a second (M, N) at that K               */
    };
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, i, ncell = (int)(sizeof cells / sizeof cells[0]), fail = 0;
    int one_M = 0, one_K = 0, one_N = 0;

    if (!hw || strcmp(hw->name, "rk3576") != 0) {
        printf("SKIP: active profile is %s, not rk3576\n", hw ? hw->name : "(none)");
        return 2;
    }
    if (argc == 4) {
        one_M = atoi(argv[1]); one_K = atoi(argv[2]); one_N = atoi(argv[3]);
        if (one_M <= 0 || one_K <= 0 || one_N <= 0) {
            fprintf(stderr, "usage: %s [M K N]\n", argv[0]);
            return 1;
        }
    }
    fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open: %d\n", fd); return 1; }

    printf("rk3576 W8A8 route — the frontend's two-pass calibration, on the part\n");
    printf("  bootstrap margin %.2f, safety factor %.2f, bias NULL, no rotation\n\n",
           ROUTE_BOOTMARGIN, ROUTE_CALSAFE);
    printf("%5s %6s %6s | %5s %5s | %9s %9s | %9s %9s | %9s %9s %8s\n",
           "M", "K", "N", "sh_p1", "sh_fr", "p1|err|", "p1 signed", "est med",
           "est worst", "rms rel", "ramp err", "sat");
    printf("--------------------------------------------------------------"
           "----------------------------------------------------\n");

    for (i = 0; i < ncell; i++) {
        struct cell_result r;
        int M = one_M ? one_M : cells[i].M;
        int K = one_K ? one_K : cells[i].K;
        int N = one_N ? one_N : cells[i].N;
        if (run_cell(fd, M, K, N, cells[i].seed, &r) != 0) {
            printf("%5d %6d %6d | REFUSED\n", M, K, N);
            fail++;
            if (one_M) break;
            continue;
        }
        printf("%5d %6d %6d | %5u %5u | %8.3f%% %9.4f | %8.3f%% %8.2f%% | "
               "%8.3f%% %8.3f%% %7.4f%%\n",
               M, K, N, r.shift_p1, r.shift_frozen,
               100.0 * r.p1_med_abs, r.p1_med_signed,
               100.0 * r.est_med, 100.0 * r.est_worst,
               100.0 * r.rms_rel, 100.0 * r.worst_rel_err, 100.0 * r.sat_frac);
        if (r.stamped) {
            printf("      FAIL: %ld elements still hold the host stamp\n", r.stamped);
            fail++;
        }
        if (r.shift_p1 > 63u || r.shift_frozen > 63u) {
            printf("      FAIL: a programmed SHIFT is outside DPU 0x40B4's six-bit "
                   "field (p1 %u, frozen %u)\n", r.shift_p1, r.shift_frozen);
            fail++;
        }
        if (r.est_med > EST_MED_MAX) {
            printf("      FAIL: bootstrap median |est/true - 1| %.3f%% exceeds %.1f%%\n",
                   100.0 * r.est_med, 100.0 * EST_MED_MAX);
            fail++;
        }
        {
            double lim = M >= RMS_M_PREFILL ? RMS_REL_MAX : RMS_REL_MAX_M1;
            if (r.rms_rel > lim) {
                printf("      FAIL: composed RMS relative error %.3f%% exceeds %.1f%% "
                       "(the bound at M=%d)\n", 100.0 * r.rms_rel, 100.0 * lim, M);
                fail++;
            }
        }
        if (one_M) break;
    }

    /* ---- the 2x2: what an M=1 cell's composed error is actually made of.
     *
     * At M=1 a column's frozen colmax is ONE accumulator sample, so the spread of
     * scale_n across a tile is the raw Gaussian spread of a single draw rather than the
     * tight max-over-M-rows a prefill cell gives. Two candidates fall out of that and
     * they need separating, because a one-at-a-time arm cannot see a condition of two:
     *
     *   the ESTIMATOR   the bootstrap's readback floors at one code (`mx < 1` is
     *                   credited 1), so a column whose single accumulator is far below
     *                   one code is over-estimated without bound.
     *   the RAMP        base_scale is max_j(cs_j/cmax_j) over the WHOLE tile and this
     *                   entry does not sort, so one extreme column costs every other
     *                   column its C resolution.
     *
     * ORACLE replaces the bootstrap with the exact colmax this test already has, which
     * a frontend cannot; TILES forces a smaller output-channel tile. Both axes, crossed.
     */
    if (!one_M) {
        static const struct { int M, K, N; uint32_t seed; } sq[] = {
            { 1, 1536, 2048, 0x51ce0001u },
            { 1, 2048, 2048, 0x51ce0003u },
        };
        static const unsigned nts[] = { 0u, 256u, 32u };
        int c, o, t;
        printf("\nthe M=1 2x2 — bootstrap vs oracle colmax, crossed with the N tile\n");
        printf("%5s %6s %6s %8s %7s | %9s %9s %9s\n",
               "M", "K", "N", "colmax", "NT", "est med", "rms rel", "ramp err");
        printf("--------------------------------------------------"
               "-------------------------------\n");
        for (c = 0; c < 2; c++)
            for (o = 0; o < 2; o++)
                for (t = 0; t < 3; t++) {
                    struct cell_result r;
                    char buf[32];
                    if (nts[t]) {
                        snprintf(buf, sizeof buf, "%u", nts[t]);
                        setenv("ROCKET_RK3576_MM_NT", buf, 1);
                    } else {
                        unsetenv("ROCKET_RK3576_MM_NT");
                    }
                    g_use_oracle = o;
                    if (run_cell(fd, sq[c].M, sq[c].K, sq[c].N, sq[c].seed, &r) != 0) {
                        printf("%5d %6d %6d %8s %7s | REFUSED\n", sq[c].M, sq[c].K,
                               sq[c].N, o ? "oracle" : "boot", nts[t] ? buf : "default");
                        continue;
                    }
                    printf("%5d %6d %6d %8s %7s | %8.3f%% %8.3f%% %8.3f%%\n",
                           sq[c].M, sq[c].K, sq[c].N, o ? "oracle" : "boot",
                           nts[t] ? buf : "default",
                           100.0 * r.est_med, 100.0 * r.rms_rel,
                           100.0 * r.worst_rel_err);
                }
        unsetenv("ROCKET_RK3576_MM_NT");
        g_use_oracle = 0;
    }

    rocket_close(fd);
    printf("\n%s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
