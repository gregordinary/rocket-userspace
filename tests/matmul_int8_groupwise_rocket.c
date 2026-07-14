// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int8_groupwise_rocket.c — accuracy gate for rocket_matmul_int8_groupwise.
 *
 * The group-wise int8 matmul computes, in fp32:
 *   C_f[m,n] = sum_g a_scale[m,g] * b_scale[n,g] * (sum_{k in group g} A[m,k]*B[n,k])
 * with every K-tile kept inside one quant group. This is the primitive a NATIVELY
 * quantized weight needs (a GGUF MXFP4/Q8_0/Q4_K block carries one scale per K-block,
 * and the NPU cannot apply a K-blocked scale on-chip). The gate validates the per-K-tile
 * dequant against a host fp64 reference over IDENTICAL int8 inputs and per-group scales —
 * it isolates the matmul (packing / group indexing / per-group scaling / fp32 K-accum),
 * independent of quant quality.
 *
 * Each K-tile's NPU partial is an EXACT integer — unlike the int4 twin there is no
 * saturation bound, because int8's output accumulator is int32, not int16
 * (|partial| <= Kt*127*127 = 10.3 M at the CBUF's Kt <= 640, far inside int32). So the
 * ONLY inexactness is the host's fp32 accumulation of the nKt scaled terms.
 *
 * It also sweeps `group` to exercise the two tiling regimes, which is where the bugs
 * would be: a narrow group gives Kt == group (one K-tile per group), while a group too
 * wide for the CBUF is split into kt_per_group tiles that must all pick up the SAME
 * group's scale. The planner's chosen Kt is printed per group.
 *
 * Usage: matmul_int8_groupwise_rocket [M K N [group]]   (default 128 2880 256, sweep)
 *   Needs M%4, K%32, N%32, K%group, group%32.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"

static int8_t rand_i8(void)    { return (int8_t)(rand() % 255 - 127); }        /* [-127,127] */
static float  rand_scale(void) { return 0.5f + (rand() % 1000) / 1000.0f; }    /* [0.5,1.5)  */

/* host fp64 reference: per-group scaled accumulation (exact integer inner products) */
static void ref_gw(int M, int K, int N, int group, const int8_t *A, const int8_t *B,
                   const float *as, const float *bs, double *ref)
{
    const int nG = K / group;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int g = 0; g < nG; g++) {
                long ip = 0;
                for (int k = 0; k < group; k++)
                    ip += (long)A[(size_t)m*K + (size_t)g*group + k] *
                          (long)B[(size_t)n*K + (size_t)g*group + k];
                acc += (double)as[(size_t)m*nG + g] * (double)bs[(size_t)n*nG + g] * (double)ip;
            }
            ref[(size_t)m*N + n] = acc;
        }
}

/* Gate on the error NORMALIZED BY THE RESULT SCALE (rms of the reference), not on a
 * per-element relative error: A and B are random-signed, so a handful of outputs land
 * near zero by cancellation and their relative error carries no information. The fp32
 * accumulation of nKt terms of magnitude ~rms costs ~eps*sqrt(nKt)*rms ≈ 1e-7*rms, so
 * 1e-5*rms is a ~100x margin that still catches any real defect (a mis-indexed group or
 * a straddled K-tile moves an output by O(rms), not by O(eps*rms)). */
static int check(const float *got, const double *ref, size_t n, int nG)
{
    double sq = 0;
    for (size_t i = 0; i < n; i++) sq += ref[i] * ref[i];
    double rms = sqrt(sq / (double)n);
    double tol = 1e-5 * rms;

    double max_abs = 0; long nbad = 0, nnf = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) { nnf++; continue; }
        double ad = fabs((double)got[i] - ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad > tol) nbad++;
    }
    printf("    nG=%-2d rms=%.3e max_abs=%.4g (tol %.4g = 1e-5*rms) norm_err=%.2e "
           "nbad=%ld nonfinite=%ld -> %s\n",
           nG, rms, max_abs, tol, rms > 0 ? max_abs / rms : 0.0, nbad, nnf,
           (nbad || nnf) ? "FAIL" : "PASS");
    return (nbad || nnf) ? -1 : 0;
}

int main(int argc, char **argv)
{
    int M = 128, K = 2880, N = 256;
    /* Groups spanning both regimes at the RK3588 CBUF: 96/288/576 give Kt == group
     * (one K-tile per group), 1440/2880 are wider than the CBUF can hold and get split
     * into several K-tiles that must share one scale. 576 is the MoE operating point. */
    int sweep[] = { 96, 288, 576, 1440, 2880 };
    int ngroups = (int)(sizeof(sweep) / sizeof(sweep[0]));

    if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    if (argc == 5) { sweep[0] = atoi(argv[4]); ngroups = 1; }
    else if (argc != 1 && argc != 4) { printf("usage: %s [M K N [group]]\n", argv[0]); return -1; }
    if (M % 4 || K % 32 || N % 32) {
        fprintf(stderr, "bad shape: need M%%4, K%%32, N%%32\n");
        return -1;
    }
    printf("int8 GROUPWISE C_f[%d,%d] = A[%d,%d] x B[%d,%d]^T\n", M, N, M, K, N, K);

    size_t Asz = (size_t)M * K, Bsz = (size_t)N * K, Csz = (size_t)M * N;
    int8_t *A  = malloc(Asz), *B = malloc(Bsz);
    float  *Cf = malloc(Csz * sizeof(float));
    double *ref = malloc(Csz * sizeof(double));
    /* scales are sized for the FINEST group in the sweep and re-filled per group */
    float *as = malloc((size_t)M * (K / 32) * sizeof(float));
    float *bs = malloc((size_t)N * (K / 32) * sizeof(float));
    if (!A || !B || !as || !bs || !Cf || !ref) { fprintf(stderr, "host alloc failed\n"); return -1; }

    srand(20260714);
    for (size_t i = 0; i < Asz; i++) A[i] = rand_i8();
    for (size_t i = 0; i < Bsz; i++) B[i] = rand_i8();

    int fd = rocket_open();
    if (fd < 0) { printf("no NPU (%d) -> SKIP\n", fd); return 2; }

    int fails = 0, ran = 0;
    for (int gi = 0; gi < ngroups; gi++) {
        int group = sweep[gi];
        if (group < 32 || group % 32 || K % group) {
            printf("  group=%-5d SKIP (needs >=32, %%32, |K=%d)\n", group, K);
            continue;
        }
        int nG = K / group, Mt, Kt, Nt;
        int tiles = rocket_matmul_plan_int8_gw(M, K, N, group, &Mt, &Kt, &Nt);
        if (tiles < 0) { printf("  group=%-5d SKIP (planner rejects: %d)\n", group, tiles); continue; }
        printf("  group=%-5d Mt=%d Kt=%d Nt=%d  nKt=%d  kt_per_group=%d  tiles=%d\n",
               group, Mt, Kt, Nt, K / Kt, group / Kt, tiles);

        for (size_t i = 0; i < (size_t)M * nG; i++) as[i] = rand_scale();
        for (size_t i = 0; i < (size_t)N * nG; i++) bs[i] = rand_scale();
        ref_gw(M, K, N, group, A, B, as, bs, ref);

        memset(Cf, 0, Csz * sizeof(float));
        int rc = rocket_matmul_int8_groupwise(fd, M, K, N, A, B, as, bs, Cf, group);
        if (rc) { fprintf(stderr, "  rocket_matmul_int8_groupwise(group=%d) = %d\n", group, rc);
                  rocket_close(fd); return -1; }
        if (check(Cf, ref, Csz, nG)) fails++;
        ran++;
    }
    rocket_close(fd);

    if (!ran) { fprintf(stderr, "no group ran\n"); return -1; }
    printf("\n==> %s (%d/%d groups failed)\n", fails ? "FAIL" : "ALL PASS", fails, ran);
    free(A); free(B); free(as); free(bs); free(Cf); free(ref);
    return fails ? -1 : 0;
}
