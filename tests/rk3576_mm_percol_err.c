// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_mm_percol_err.c — what the per-column matmul's two ramps cost ONE real GEMM, and
 * what each costs per call.
 *
 * `err` reads operand dumps (tools/rk3576-w8a8-dump.py: a real model's rotated per-tensor
 * int8 activations and rotated per-channel int8 weights at one projection) and runs the
 * resident-weight entry, rocket_matmul_int8_rk3576_perc_wbo() — what ggml-rocket's W8A8
 * route calls — once on each ramp, at the route's own column scale: 127 / (colmax *
 * CALSAFE), with colmax the exact per-column accumulator maximum of the same window (the
 * oracle, which the route's bootstrap estimates to about 0.3%). For each ramp it reports:
 *
 *   gain     every column's delivered gain C*MUL/2^(SHIFT+bs) against its exact float
 *            scale, median / 90th percentile / worst, from the host model of that ramp;
 *   elements the device surface against the exact per-column float requant
 *            sat8(rne(acc*scale)), and against the host model of the ramp (which must be 0);
 *   composed RMS(C8/scale - acc) / RMS(acc), de-quantized by the scale ASKED for, as the
 *            route does, beside the same statistic for the exact float requant (the int8
 *            rounding floor no ramp can go under).
 *
 * and, once per GEMM, the fraction of columns whose shift-0 cap INT32_MAX/(128*sum|B|+1)
 * is below the int16 field (where that ramp's resolution is set by the product and not by
 * the field) and the within-tile column-scale spread.
 *
 *   sudo -E taskset -c 4-7 ./build/rk3576_mm_percol_err err 3.0 dump1.bin [dump2.bin ...]
 *
 * `time` is a paired per-call A/B of the same entry on the two ramps at one shape, the
 * ramps alternating call by call with the order rotated each pair (the knob is read per
 * call), the first pair discarded. Synthetic operands; the column scale is the route's form
 * over their exact accumulator.
 *
 *   sudo -E taskset -c 4-7 ./build/rk3576_mm_percol_err time M K N PAIRS [shift|shift0]
 *
 * A trailing ramp name runs that ramp alone for every call, for a profiled process
 * (ROCKET_MM_PROFILE) whose buckets must belong to one ramp.
 *
 * WHAT IT DOES NOT SHOW. One window's activations per projection, the fp32 model's rather
 * than the quantized model's own, and an oracle colmax rather than the frozen bootstrap one.
 * A per-GEMM error does not predict a model's perplexity (the header of
 * rocket_matmul_int8_rk3576_perc says why); llama-perplexity is the end-to-end arm.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "requant_model.h"
#include "perchannel_model.h"   /* the ramps, planned independently of the library */

struct gemm { int M, K, N; char name[65]; int8_t *A, *B; int32_t *acc; };

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

/* The exact int32 accumulator, four threads over N. */
struct acc_job { const struct gemm *g; int n0, n1; };
static void *acc_worker(void *p)
{
    const struct acc_job *j = p;
    const struct gemm *g = j->g;
    for (int m = 0; m < g->M; m++) {
        const int8_t *a = g->A + (size_t)m * g->K;
        for (int n = j->n0; n < j->n1; n++) {
            const int8_t *b = g->B + (size_t)n * g->K;
            int32_t s = 0;
            for (int k = 0; k < g->K; k++) s += (int32_t)a[k] * (int32_t)b[k];
            g->acc[(size_t)m * g->N + n] = s;
        }
    }
    return NULL;
}
static int compute_acc(struct gemm *g)
{
    pthread_t th[4];
    struct acc_job jb[4];
    g->acc = malloc((size_t)g->M * g->N * sizeof *g->acc);
    if (!g->acc) return -1;
    for (int t = 0; t < 4; t++) {
        jb[t].g = g;
        jb[t].n0 = g->N * t / 4;
        jb[t].n1 = g->N * (t + 1) / 4;
        pthread_create(&th[t], NULL, acc_worker, &jb[t]);
    }
    for (int t = 0; t < 4; t++) pthread_join(th[t], NULL);
    return 0;
}

static int read_dump(const char *path, struct gemm *g)
{
    FILE *f = fopen(path, "rb");
    char magic[4];
    int32_t hdr[3];
    memset(g, 0, sizeof *g);
    if (!f) return -1;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PCD1", 4) ||
        fread(hdr, sizeof hdr[0], 3, f) != 3 || fread(g->name, 1, 64, f) != 64) {
        fclose(f); return -1;
    }
    g->M = hdr[0]; g->K = hdr[1]; g->N = hdr[2];
    g->A = malloc((size_t)g->M * g->K);
    g->B = malloc((size_t)g->N * g->K);
    if (!g->A || !g->B ||
        fread(g->A, 1, (size_t)g->M * g->K, f) != (size_t)g->M * g->K ||
        fread(g->B, 1, (size_t)g->N * g->K, f) != (size_t)g->N * g->K) {
        fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

static void set_ramp(int shift) /* 1 = the shift ramp (default), 0 = shift-0 */
{
    if (shift) unsetenv("ROCKET_RK3576_MM_PC_SHIFT");
    else       setenv("ROCKET_RK3576_MM_PC_SHIFT", "0", 1);
}

/* The host model of one ramp over the whole N, tile by tile at the entry's tiling: the
 * surface it predicts and each column's delivered gain. */
static void model_ramp(const struct gemm *g, const float *cs, const int64_t *sa, int nt,
                       int shift, int8_t *out, double *gain, unsigned *bsmax)
{
    int16_t *C = calloc((size_t)g->N, sizeof *C);
    int32_t *bz = calloc((size_t)g->N, sizeof *bz);
    int *cst = calloc((size_t)g->N, sizeof *cst);
    *bsmax = 0;
    for (int n0 = 0; n0 < g->N; n0 += nt) {
        unsigned tn = (unsigned)(g->N - n0 < nt ? g->N - n0 : nt), mul, sh, bs = 0;
        double base;
        if (shift)
            plan_c_bs((unsigned)n0, tn, NULL, NULL, sa, bz, 1.0f, cs, 1.0f, 0, C, cst,
                      &mul, &sh, &bs);
        else
            plan_c((unsigned)n0, tn, NULL, sa, bz, 1.0f, cs, 1.0f, C, &mul, &sh);
        if (bs > *bsmax) *bsmax = bs;
        base = (double)mul / (double)((uint64_t)1 << sh) / ldexp(1.0, (int)bs);
        for (unsigned j = 0; j < tn; j++) gain[n0 + j] = (double)C[n0 + j] * base;
        for (int m = 0; m < g->M; m++)
            for (unsigned j = 0; j < tn; j++)
                out[(size_t)m * g->N + n0 + j] =
                    (int8_t)epilogue_bs(g->acc[(size_t)m * g->N + n0 + j], C[n0 + j], bs,
                                        mul, sh, 0);
    }
    free(C); free(bz); free(cst);
}

static int do_err(int fd, double calsafe, int ndump, char **dumps)
{
    int fail = 0;
    printf("%-26s %5s %5s %5s | %-6s %9s %9s %9s | %8s %7s %6s | %8s %8s | %4s\n",
           "gemm", "M", "K", "N", "ramp", "gain med", "gain p90", "gain max",
           "vs exact", "frac", "max|d|", "rms", "rms ex", "bs");
    for (int d = 0; d < ndump; d++) {
        struct gemm g;
        int nt = 0;
        size_t MN;
        float *cs;
        int64_t *sa;
        double *gain, *errs, rms_ex = 0.0, ss = 0.0;
        int8_t *C8, *mod, *ex;
        struct rocket_rk3576_wbo *wbo = NULL;
        long capbind = 0;
        double *spreads;
        int ntile = 0;

        if (read_dump(dumps[d], &g) != 0) {
            printf("%s: unreadable dump\n", dumps[d]);
            fail = 1; continue;
        }
        MN = (size_t)g.M * g.N;
        if (compute_acc(&g) != 0) { fail = 1; continue; }
        cs = malloc((size_t)g.N * sizeof *cs);
        sa = calloc((size_t)g.N, sizeof *sa);
        gain = malloc((size_t)g.N * sizeof *gain);
        errs = malloc((size_t)g.N * sizeof *errs);
        spreads = malloc((size_t)g.N * sizeof *spreads);
        C8 = malloc(MN); mod = malloc(MN); ex = malloc(MN);
        for (int n = 0; n < g.N; n++) {
            double mx = 0.0;
            const int8_t *b = g.B + (size_t)n * g.K;
            for (int k = 0; k < g.K; k++) sa[n] += b[k] < 0 ? -b[k] : b[k];
            for (int m = 0; m < g.M; m++) {
                double v = fabs((double)g.acc[(size_t)m * g.N + n]);
                if (v > mx) mx = v;
            }
            cs[n] = (float)(127.0 / ((mx < 1.0 ? 1.0 : mx) * calsafe));
            if ((double)INT32_MAX / (128.0 * (double)sa[n] + 1.0) < 32767.0) capbind++;
        }
        for (size_t i = 0; i < MN; i++) {
            double v = nearbyint((double)g.acc[i] * (double)cs[i % (size_t)g.N]);
            double r;
            ex[i] = (int8_t)(v > 127.0 ? 127 : (v < -128.0 ? -128 : v));
            r = (double)ex[i] / (double)cs[i % (size_t)g.N] - (double)g.acc[i];
            rms_ex += r * r;
            ss += (double)g.acc[i] * (double)g.acc[i];
        }
        rms_ex = sqrt(rms_ex / ss);
        if (rocket_matmul_plan_int8_rk3576(g.M, g.K, g.N, NULL, NULL, &nt) < 0 || nt <= 0 ||
            rocket_rk3576_wbo_create(fd, g.K, g.N, g.B, &wbo) != 0) {
            printf("%s: no plan or no resident weight\n", g.name);
            fail = 1; continue;
        }
        for (int n0 = 0; n0 < g.N; n0 += nt) {
            int n1 = n0 + nt < g.N ? n0 + nt : g.N;
            float lo = cs[n0], hi = cs[n0];
            for (int n = n0; n < n1; n++) {
                if (cs[n] < lo) lo = cs[n];
                if (cs[n] > hi) hi = cs[n];
            }
            spreads[ntile++] = (double)hi / (double)lo;
        }
        qsort(spreads, (size_t)ntile, sizeof *spreads, cmp_dbl);

        for (int shift = 1; shift >= 0; shift--) {
            double worst = 0.0, rms = 0.0, t0, t1;
            long vs_ex = 0, vs_mod = 0;
            int maxd = 0, rc;
            unsigned bsmax;
            model_ramp(&g, cs, sa, nt, shift, mod, gain, &bsmax);
            set_ramp(shift);
            memset(C8, 0x5A, MN);
            t0 = now_ms();
            rc = rocket_matmul_int8_rk3576_perc_wbo(fd, g.M, g.K, g.N, g.A, wbo, NULL, cs,
                                                    sa, C8, &worst);
            t1 = now_ms();
            if (rc != 0) {
                printf("%s: %s ramp REFUSED rc=%d\n", g.name, shift ? "shift" : "shift0",
                       rc);
                fail = 1; continue;
            }
            for (size_t i = 0; i < MN; i++) {
                int dd = C8[i] - ex[i];
                double r = (double)C8[i] / (double)cs[i % (size_t)g.N] - (double)g.acc[i];
                rms += r * r;
                if (dd) { vs_ex++; if (abs(dd) > maxd) maxd = abs(dd); }
                if (C8[i] != mod[i]) vs_mod++;
            }
            rms = sqrt(rms / ss);
            for (int n = 0; n < g.N; n++)
                errs[n] = fabs(gain[n] - (double)cs[n]) / (double)cs[n];
            qsort(errs, (size_t)g.N, sizeof *errs, cmp_dbl);
            printf("%-26s %5d %5d %5d | %-6s %8.4f%% %8.4f%% %8.4f%% | %8ld %6.3f%% %6d | "
                   "%7.4f%% %7.4f%% | %4u\n",
                   g.name, g.M, g.K, g.N, shift ? "shift" : "shift0",
                   100.0 * errs[g.N / 2], 100.0 * errs[(g.N * 9) / 10],
                   100.0 * errs[g.N - 1], vs_ex, 100.0 * (double)vs_ex / (double)MN, maxd,
                   100.0 * rms, 100.0 * rms_ex, bsmax);
            printf("      device vs model %ld of %zu, entry worst_rel_err %.4f%%, call "
                   "%.1f ms\n", vs_mod, MN, 100.0 * worst, t1 - t0);
            if (vs_mod) fail = 1;
        }
        printf("      N tile %d (%d tiles), within-tile spread median %.1fx max %.1fx; "
               "shift-0 cap below the int16 field on %.1f%% of columns\n",
               nt, ntile, spreads[ntile / 2], spreads[ntile - 1],
               100.0 * (double)capbind / (double)g.N);
        set_ramp(1);
        rocket_rk3576_wbo_free(fd, wbo);
        free(cs); free(sa); free(gain); free(errs); free(spreads);
        free(C8); free(mod); free(ex); free(g.A); free(g.B); free(g.acc);
    }
    return fail;
}

static uint32_t xs32(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return (*s = x);
}

/* `only` < 0 alternates the ramps; 0 or 1 runs that ramp alone for every call, so a
 * profiled process (ROCKET_MM_PROFILE) attributes its buckets to one ramp. */
static int do_time(int fd, int M, int K, int N, int pairs, int only)
{
    struct gemm g;
    uint32_t s = 0x13579BDFu;
    float *cs = malloc((size_t)N * sizeof *cs);
    int64_t *sa = calloc((size_t)N, sizeof *sa);
    int8_t *C8 = malloc((size_t)M * N), *ref[2] = { malloc((size_t)M * N),
                                                   malloc((size_t)M * N) };
    double *t[2] = { malloc((size_t)pairs * sizeof(double)),
                     malloc((size_t)pairs * sizeof(double)) };
    double *ratio = malloc((size_t)pairs * sizeof(double));
    struct rocket_rk3576_wbo *wbo = NULL;
    int bad = 0;

    memset(&g, 0, sizeof g);
    g.M = M; g.K = K; g.N = N;
    g.A = malloc((size_t)M * K); g.B = malloc((size_t)N * K);
    /* Gaussian-ish int8: the sum of four bytes, as rotated operands are. */
    for (size_t i = 0; i < (size_t)M * K; i++) {
        int v = 0;
        for (int q = 0; q < 4; q++) v += (int)(xs32(&s) & 0x3F) - 32;
        g.A[i] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
    }
    for (size_t i = 0; i < (size_t)N * K; i++) {
        int v = 0;
        for (int q = 0; q < 4; q++) v += (int)(xs32(&s) & 0x1F) - 16;
        g.B[i] = (int8_t)v;
    }
    compute_acc(&g);
    for (int n = 0; n < N; n++) {
        double mx = 1.0;
        for (int k = 0; k < K; k++) sa[n] += g.B[(size_t)n * K + k] < 0 ?
                                             -g.B[(size_t)n * K + k] : g.B[(size_t)n * K + k];
        for (int m = 0; m < M; m++) {
            double v = fabs((double)g.acc[(size_t)m * N + n]);
            if (v > mx) mx = v;
        }
        cs[n] = (float)(127.0 / (mx * 3.0));
    }
    if (rocket_rk3576_wbo_create(fd, K, N, g.B, &wbo) != 0) return 1;

    for (int p = -1; p < pairs; p++) {            /* p = -1 is the discarded pair */
        for (int o = 0; o < 2; o++) {
            int arm = ((p & 1) ? 1 - o : o);       /* arm 0 = shift-0, arm 1 = shift */
            if (only >= 0) arm = only;
            double t0, t1;
            set_ramp(arm);
            t0 = now_ms();
            if (rocket_matmul_int8_rk3576_perc_wbo(fd, M, K, N, g.A, wbo, NULL, cs, sa, C8,
                                                   NULL) != 0) { bad = 1; break; }
            t1 = now_ms();
            if (p < 0) memcpy(ref[arm], C8, (size_t)M * N);
            else {
                t[arm][p] = t1 - t0;
                if (memcmp(ref[arm], C8, (size_t)M * N)) bad = 1;   /* deterministic */
            }
        }
        if (bad) break;
        if (p >= 0) ratio[p] = t[1][p] / t[0][p];
    }
    set_ramp(1);
    if (!bad) {
        qsort(t[0], (size_t)pairs, sizeof(double), cmp_dbl);
        qsort(t[1], (size_t)pairs, sizeof(double), cmp_dbl);
        qsort(ratio, (size_t)pairs, sizeof(double), cmp_dbl);
        printf("time M=%d K=%d N=%d pairs=%d: shift0 median %.3f ms, shift median %.3f ms, "
               "paired shift/shift0 median %.4f (min %.4f, max %.4f)\n", M, K, N, pairs,
               t[0][pairs / 2], t[1][pairs / 2], ratio[pairs / 2], ratio[0],
               ratio[pairs - 1]);
    } else {
        printf("time M=%d K=%d N=%d: a call refused or a repeat differed\n", M, K, N);
    }
    rocket_rk3576_wbo_free(fd, wbo);
    return bad;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, rc;
    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_mm_percol_err: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    if (argc < 3 || (strcmp(argv[1], "err") && strcmp(argv[1], "time")) ||
        (!strcmp(argv[1], "time") && argc < 6)) {
        fprintf(stderr, "usage: %s err CALSAFE dump...\n"
                "       %s time M K N PAIRS [shift|shift0]\n", argv[0], argv[0]);
        return 1;
    }
    fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open: %d\n", fd); return 1; }
    if (!strcmp(argv[1], "err"))
        rc = do_err(fd, atof(argv[2]), argc - 3, argv + 3);
    else
        rc = do_time(fd, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
                     argc > 6 ? (strcmp(argv[6], "shift0") ? 1 : 0) : -1);
    rocket_close(fd);
    return rc ? 1 : 0;
}
