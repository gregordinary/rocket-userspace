// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * requant_round_probe.c — which way does the DPU's OUT_CVT round a TIE, and what does
 * OUT_CVT_SHIFT bit 30 change about it?
 *
 * The output convertor is `out = sat8( round(acc * MUL >> SHIFT) + offset )`. A tie is
 * an exact half after the shift, and the candidate rules differ only there: half up
 * (what `(x + half) >> s` gives), half away from zero (what the NVDLA documentation
 * specifies), half to even (QNNPACK's precise requantization), truncation toward zero,
 * and floor. They differ by one count on a sparse set of elements, which is exactly the
 * size of the standing noise under int8 measurements here.
 *
 * NO GATE CAN SEE IT. The scale a caller passes becomes MUL by the vendor's (QNNPACK)
 * derivation, `MUL = ((bits>>9) & 0x7fff) + 1` with bit 14 forced, and the trailing +1
 * makes MUL ODD for every round scale, including every power of two. An odd multiplier
 * moves an exact half off the tie, always outward, so a tie never reaches the rounder.
 * The probe therefore builds its ties on purpose.
 *
 * BIT 30. Mesa's registers.xml names OUT_CVT_SHIFT bit 30 CVT_ROUND, and open-rknpu
 * measures it on the RV1106, whose DPU map is the RK3588's: 0 rounds a tie half to even,
 * 1 half away from zero. Every entry in this library writes 0. ROCKET_OUT_CVT_ROUND=1
 * sets it on every integer OUT_CVT_SHIFT write, and each arm below runs both values.
 *
 * THE RK3576 ARM drives rocket_matmul_int8_rk3576 with zero operands, so the accumulator
 * IS the per-channel bias, and with a scale chosen so MUL lands on exactly 2^14 (a float
 * whose top 14 mantissa bits are all ones under an even exponent field). The requant is
 * then a plain `acc >> e`, and every odd multiple of 2^(e-1) is a tie.
 *
 * THE RK3588 ARM uses the int8 matmul's integer convert: ROCKET_INT8_DEQ=1 with
 * CVTTYPE 0, SCALE 1 and SHIFT e puts `acc >> e` through the OUT_CVT and writes the result
 * as fp32, so the rounded value is read back exactly. The accumulator is a rank-1
 * product chosen per element. The RK3588's public int8 matmul writes a raw int32 and
 * requantizes on the host, and its one int8-OUT entry, the int8 depthwise conv, takes a
 * scale the probe cannot pick freely, so it is not used here.
 *
 * THE SELF-CHECK is the load-bearing part: every element must match at least one of the
 * five rules. One that matches none means the accumulator is not what the probe built,
 * and then no tie on that surface is read. The nearest rules agree on every non-tie, so
 * the ties separate them, and the non-ties separate them from floor and truncation.
 *
 * It GATES the rule as well: bit 30 clear must leave half to even as the one surviving
 * rule, and bit 30 set half away from zero, on both parts [HW sweep, H96 MAX M9 and RK1].
 * Exits 2 (skip) where there is no NPU or no arm for the part; 1 when a run fails, the part
 * matches no rule, or the survivor is not the recorded one; 0 otherwise.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "npu_matmul.h"
#include "rocket_hw_profile.h"

/* The emitter's scale -> (MUL, SHIFT), copied from the encoders so the probe reports the
 * multiplier it is actually going to get rather than the one it assumes. */
static void derive(float conv_scale, unsigned *mul, unsigned *shift)
{
    union { float f; uint32_t u; } cv;
    uint32_t bits;
    unsigned s, m;
    cv.f = conv_scale;
    bits = cv.u;
    s = 127u + 31u - 32u - (bits >> 23) + 16u;
    m = ((bits >> 9) & 0x7FFFu) + 1u;
    if (m < (1u << 14)) m |= (1u << 14);
    *mul = m;
    *shift = s - 1u;
}

static int sat8(int64_t v)
{
    return (int)(v < -128 ? -128 : (v > 127 ? 127 : v));
}

/* The candidate rules, all on `acc * mul >> shift`, saturated to int8 when `narrow`. */
static int64_t r_half_up(int64_t acc, unsigned mul, unsigned shift)
{
    int64_t p = acc * (int64_t)mul;
    int64_t half = shift ? ((int64_t)1 << (shift - 1)) : 0;
    return (p + half) >> shift;
}
static int64_t r_half_away(int64_t acc, unsigned mul, unsigned shift)
{
    int64_t p = acc * (int64_t)mul;
    int64_t half = shift ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t mag = p < 0 ? -p : p;
    int64_t q = (mag + half) >> shift;
    return p < 0 ? -q : q;
}
static int64_t r_half_even(int64_t acc, unsigned mul, unsigned shift)
{
    int64_t p = acc * (int64_t)mul;
    int64_t half = shift ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q = (p + half) >> shift;
    if (shift && (p & (((int64_t)1 << shift) - 1)) == half && (q & 1)) q -= 1;
    return q;
}
static int64_t r_trunc(int64_t acc, unsigned mul, unsigned shift)
{
    int64_t p = acc * (int64_t)mul;
    int64_t mag = p < 0 ? -p : p;
    int64_t q = mag >> shift;
    return p < 0 ? -q : q;
}
static int64_t r_floor(int64_t acc, unsigned mul, unsigned shift)
{
    return (acc * (int64_t)mul) >> shift;
}

typedef struct { const char *name; int64_t (*f)(int64_t, unsigned, unsigned); } rule;
static const rule RULES[] = {
    { "half-up",          r_half_up },
    { "half-away-from-0", r_half_away },
    { "half-to-even",     r_half_even },
    { "truncate-toward-0", r_trunc },
    { "floor",            r_floor },
};
#define N_RULES ((int)(sizeof RULES / sizeof RULES[0]))

static int is_tie(int64_t acc, unsigned mul, unsigned shift)
{
    int64_t p = acc * (int64_t)mul;
    return shift && (p & (((int64_t)1 << shift) - 1)) == ((int64_t)1 << (shift - 1));
}

/* Scores one surface. got[i] is the part's value for acc[i]; `narrow` saturates the
 * model to int8. The self-check is rule-agnostic: an element that NO candidate rule
 * explains means the accumulator is not what the probe built, and then the surface says
 * nothing about rounding. Otherwise prints the rules every element agrees with. Returns
 * the consistent rules as a mask, bit r for RULES[r] (0 when none is), or -1 if the
 * self-check failed. */
static int score(const int64_t *acc, const int64_t *got, int n, unsigned mul, unsigned shift,
                 int narrow, int verbose)
{
    int live[N_RULES], r, i, ties = 0, bad = 0, survivors = 0, shown = 0, mask = 0;
    for (r = 0; r < N_RULES; r++) live[r] = 1;
    for (i = 0; i < n; i++) {
        int tie = is_tie(acc[i], mul, shift), any = 0;
        if (tie) ties++;
        for (r = 0; r < N_RULES; r++) {
            int64_t want = RULES[r].f(acc[i], mul, shift);
            if (narrow) want = sat8(want);
            if (want == got[i]) any = 1;
            else live[r] = 0;
        }
        if (!any) {
            if (bad < 6)
                printf("      setup: acc %+lld -> got %+lld, which no rule gives\n",
                       (long long)acc[i], (long long)got[i]);
            bad++;
        } else if (tie && verbose && shown < 8) {
            printf("      acc %+5lld TIE -> got %+4lld   up %+4lld away %+4lld even %+4lld "
                   "trunc %+4lld floor %+4lld\n", (long long)acc[i], (long long)got[i],
                   (long long)r_half_up(acc[i], mul, shift),
                   (long long)r_half_away(acc[i], mul, shift),
                   (long long)r_half_even(acc[i], mul, shift),
                   (long long)r_trunc(acc[i], mul, shift),
                   (long long)r_floor(acc[i], mul, shift));
            shown++;
        }
    }
    if (bad) {
        printf("      %d elements match no rule: the accumulator is not what the probe "
               "built, so the ties say nothing\n", bad);
        return -1;
    }
    printf("      %d of %d are exact ties.  consistent rules:", ties, n);
    for (r = 0; r < N_RULES; r++)
        if (live[r]) { printf("  %s", RULES[r].name); survivors++; mask |= 1 << r; }
    if (!survivors) printf("  NONE");
    printf("\n");
    return mask;
}

static void set_round(int round_bit)
{
    setenv("ROCKET_OUT_CVT_ROUND", round_bit ? "1" : "0", 1);
}

/* ---- the RK3576 arm ---------------------------------------------------------------- */

#define M_ROWS 4
#define K_DEP  32
#define N_CH   64

/* Each scale is chosen for MUL == 0x4000 exactly: significand 1.99993896484375 under an
 * even exponent field, so the top 14 mantissa bits are all ones and the +1 carries into
 * bit 14 rather than making the multiplier odd. */
typedef struct { const char *name; float scale; } probe_scale;
static const probe_scale SCALES[] = {
    { "acc>>1", 0.999969482421875f },      /* 0x3f7ffe00 */
    { "acc>>3", 0.24999237060546875f },    /* 0x3e7ffe00 */
};
#define N_SCALES ((int)(sizeof SCALES / sizeof SCALES[0]))

static int run_rk3576(int fd, const probe_scale *ps, int round_bit, int verbose)
{
    int8_t *A, *B, *C;
    int32_t *bias;
    int64_t acc[M_ROWS * N_CH], got[M_ROWS * N_CH];
    unsigned mul, shift;
    int n, m, rc, s;

    derive(ps->scale, &mul, &shift);
    printf("    %s  MUL=0x%04x SHIFT=%u  CVT_ROUND=%d\n", ps->name, mul, shift, round_bit);
    if (mul != (1u << 14)) {
        printf("      MUL is not 2^14, so this scale cannot make a tie\n");
        return -1;
    }

    A    = calloc((size_t)M_ROWS * K_DEP, 1);
    B    = calloc((size_t)N_CH * K_DEP, 1);
    C    = calloc((size_t)M_ROWS * N_CH, 1);
    bias = calloc(N_CH, sizeof *bias);
    if (!A || !B || !C || !bias) { free(A); free(B); free(C); free(bias); return -1; }

    /* Operands zero: the accumulator IS the bias, over both signs and both parities. */
    for (n = 0; n < N_CH; n++) bias[n] = n - N_CH / 2;

    set_round(round_bit);
    rc = rocket_matmul_int8_rk3576(fd, M_ROWS, K_DEP, N_CH, A, B, bias, ps->scale, C);
    unsetenv("ROCKET_OUT_CVT_ROUND");
    if (rc != 0) {
        printf("      rocket_matmul_int8_rk3576 returned %d\n", rc);
        free(A); free(B); free(C); free(bias);
        return -1;
    }
    for (m = 0; m < M_ROWS; m++)
        for (n = 0; n < N_CH; n++) {
            acc[m * N_CH + n] = bias[n];
            got[m * N_CH + n] = C[(size_t)m * N_CH + n];
        }
    s = score(acc, got, M_ROWS * N_CH, mul, shift, 1, verbose);
    free(A); free(B); free(C); free(bias);
    return s;
}

/* ---- the RK3588 arm ---------------------------------------------------------------- */

#define DQ_M 4
#define DQ_K 64
#define DQ_N 64

static unsigned rup(unsigned a, unsigned b) { return ((a + b - 1) / b) * b; }

/* One int8 matmul through gen_matmul_int8 with the integer convert armed, read back as
 * fp32. acc[m][n] = (m+1)*(n-32): a rank-1 product over both signs and both parities. */
static int run_rk3588(int fd, unsigned shift, int round_bit, int verbose)
{
    rocket_bo guard = {0}, regcmd = {0}, input = {0}, weights = {0}, output = {0};
    int8_t A[DQ_M * DQ_K], B[DQ_N * DQ_K];
    int64_t acc[DQ_M * DQ_N], got[DQ_M * DQ_N];
    uint64_t regs[256] = {0};
    size_t out_bytes = (size_t)rup(DQ_M, 16) * rup(DQ_N, 32) * 8 + 32768;
    char b[16];
    int m, n, kk, ret = -1, bad_float = 0;

    printf("    acc>>%u  SCALE=1 SHIFT=%u  CVT_ROUND=%d\n", shift, shift, round_bit);

    memset(A, 0, sizeof A);
    memset(B, 0, sizeof B);
    for (m = 0; m < DQ_M; m++) A[m * DQ_K] = (int8_t)(m + 1);
    for (n = 0; n < DQ_N; n++) B[n * DQ_K] = (int8_t)(n - 32);

    if (rocket_bo_alloc(fd, 4096, &guard) ||
        rocket_bo_alloc(fd, 4096, &regcmd) ||
        rocket_bo_alloc(fd, (size_t)DQ_M * DQ_K, &input) ||
        rocket_bo_alloc(fd, (size_t)DQ_N * DQ_K, &weights) ||
        rocket_bo_alloc(fd, out_bytes, &output)) {
        printf("      BO allocation failed\n");
        goto done;
    }

    setenv("ROCKET_INT8_DEQ", "1", 1);
    setenv("ROCKET_INT8_DEQ_PREC", "fp32", 1);
    setenv("ROCKET_INT8_DEQ_CVTTYPE", "0", 1);
    setenv("ROCKET_INT8_DEQ_SCALE", "1", 1);
    snprintf(b, sizeof b, "%u", shift);
    setenv("ROCKET_INT8_DEQ_SHIFT", b, 1);
    set_round(round_bit);
    {
        matmul_params_t p = { .m = DQ_M, .k = DQ_K, .n = DQ_N,
            .input_dma = (uint32_t)input.dma_address,
            .weights_dma = (uint32_t)weights.dma_address,
            .output_dma = (uint32_t)output.dma_address, .tasks = regs };
        int g = gen_matmul_int8(&p);
        unsetenv("ROCKET_INT8_DEQ");
        unsetenv("ROCKET_INT8_DEQ_PREC");
        unsetenv("ROCKET_INT8_DEQ_CVTTYPE");
        unsetenv("ROCKET_INT8_DEQ_SCALE");
        unsetenv("ROCKET_INT8_DEQ_SHIFT");
        unsetenv("ROCKET_OUT_CVT_ROUND");
        if (g != 0) { printf("      gen_matmul_int8 refused (%d)\n", g); goto done; }

        rocket_bo_prep(fd, &regcmd, 1, 0);
        memcpy(regcmd.ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &regcmd);

        rocket_bo_prep(fd, &input, 1, 0);
        memset(input.ptr, 0, input.size);
        for (m = 1; m <= DQ_M; m++)
            for (kk = 1; kk <= DQ_K; kk++)
                ((int8_t *)input.ptr)[feature_data(DQ_K, DQ_M, 1, 16, kk, m, 1)] =
                    A[(m - 1) * DQ_K + (kk - 1)];
        rocket_bo_fini(fd, &input);

        rocket_bo_prep(fd, &weights, 1, 0);
        memset(weights.ptr, 0, weights.size);
        for (n = 1; n <= DQ_N; n++)
            for (kk = 1; kk <= DQ_K; kk++)
                ((int8_t *)weights.ptr)[weight_int8(DQ_K, n, kk)] = B[(n - 1) * DQ_K + (kk - 1)];
        rocket_bo_fini(fd, &weights);

        /* Bracketed, never a bare memset: dirty CPU lines race the DPU's write DMA. */
        rocket_bo_prep(fd, &output, 1, 0);
        memset(output.ptr, 0xAA, output.size);
        rocket_bo_fini(fd, &output);

        {
            uint32_t in_h[]  = { input.handle, weights.handle, regcmd.handle };
            uint32_t out_h[] = { output.handle };
            if (rocket_submit_matmul(fd, &regcmd, p.task_count, in_h, 3, out_h, 1, 6000)) {
                printf("      submit failed\n");
                goto done;
            }
        }
    }
    if (rocket_bo_prep(fd, &output, 0, 2000000000LL)) {
        printf("      PREP_BO on the output timed out\n");
        goto done;
    }
    for (m = 1; m <= DQ_M; m++)
        for (n = 1; n <= DQ_N; n++) {
            size_t idx = (size_t)feature_data(DQ_N, DQ_M, 1, 4, n, m, 1);
            float f;
            memcpy(&f, (uint8_t *)output.ptr + idx * 4, 4);
            acc[(m - 1) * DQ_N + (n - 1)] = (int64_t)m * (n - 1 - 32);
            got[(m - 1) * DQ_N + (n - 1)] = (int64_t)f;
            if ((float)(int64_t)f != f) bad_float++;
        }
    rocket_bo_fini(fd, &output);
    if (bad_float) {
        printf("      %d outputs are not integers: the convert is not in integer mode\n",
               bad_float);
        goto done;
    }
    ret = score(acc, got, DQ_M * DQ_N, 1, shift, 0, verbose);
done:
    rocket_bo_free(fd, &guard); rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &input);
    rocket_bo_free(fd, &weights); rocket_bo_free(fd, &output);
    return ret;
}

int main(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    static const unsigned SHIFTS[] = { 1, 3 };
    int fd, i, rb, fails = 0, is76;
    int verbose = getenv("ROCKET_RR_VERBOSE") != NULL;

    is76 = strcmp(hw->name, "rk3576") == 0;
    if (!is76 && strcmp(hw->name, "rk3588") != 0) {
        printf("requant_round_probe: profile is %s, which has no arm here. Skipping\n",
               hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("requant_round_probe: no NPU device. Skipping\n"); return 2; }

    printf("== DPU OUT_CVT tie rounding (%s) ==\n", hw->name);
    for (rb = 0; rb <= 1; rb++) {
        /* The recorded rule: RULES[2] half to even with the bit clear, RULES[1] half
         * away from zero with it set. */
        const int want = rb ? 1 << 1 : 1 << 2;
        printf("  OUT_CVT_SHIFT bit 30 = %d\n", rb);
        for (i = 0; i < 2; i++) {
            int s = is76 ? run_rk3576(fd, &SCALES[i], rb, verbose)
                         : run_rk3588(fd, SHIFTS[i], rb, verbose);
            if (s <= 0) { fails++; continue; }
            if (s != want) {
                printf("      FAIL: the recorded rule at bit 30 = %d is %s alone\n", rb,
                       RULES[rb ? 1 : 2].name);
                fails++;
            }
        }
    }
    rocket_close(fd);
    return fails ? 1 : 0;
}
