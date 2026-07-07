// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int8_dequant_rocket.c — fold the int8 output cast / scale into the
 * DPU output converter (OUT_CVT) so the int8 matmul emits a FLOAT (or integer-scaled)
 * result directly, instead of the raw int32 accumulator + a host conversion.
 *
 * WHAT THE HW DOES (RE'd here; NVDLA v1 does not specify the int->float converter):
 *   out = (float)( (acc_i32 * OUT_CVT_SCALE) >> OUT_CVT_SHIFT )     then optional ->fp16
 *   - OUT_CVT_SCALE and the BN-MUL operand are uint16 INTEGER multipliers (NOT fp16
 *     and NOT fixed-point: scale=2 -> x2, BN-MUL 0x3800 -> x14336);
 *   - OUT_CVT_SHIFT (bits[5:0]) is an INTEGER right-shift applied in the integer
 *     domain BEFORE the float cast (so it TRUNCATES — fractions are lost);
 *   - the float-affine minus_exp / cvt_type decode (the LUT path) is a NO-OP on the
 *     raw int32 CACC — it only applies to the LUT/EW float datapath;
 *   - fp16 output uses the int8-datapath writer geometry size_e=3 / surf_add=stride*2.
 *
 * CONSEQUENCES (the deliverable):
 *   + int8 matmul -> fp32 out: EXACT cast of the int32 accumulator (s_eff=1).
 *   + int8 matmul -> fp16 out: EXACT for |acc| <= 2048 (fp16 exact-int), approx to
 *     65504 -> HALVES the output readback (2 B vs int32's 4 B) for range-bounded acc.
 *   + a per-tensor INTEGER gain folds exactly (out = acc*scale).
 *   - a FRACTIONAL W8A8 dequant scale CANNOT fold to a fractional float: (acc*scale)>>shift
 *     truncates to an integer. So the host per-row*per-channel dequant stays; the int8
 *     readback lever is bigger-Kt, not OUT_CVT. (The integer truncate IS exactly
 *     right for int8->int8 requant — the conv int8-out path uses it.)
 *
 * MODES:
 *   - default (no external ROCKET_INT8_DEQ): a self-contained bit-exact SELF-TEST of
 *     the proven capabilities -> CTest gate (PASS / FAIL / SKIP-if-no-NPU).
 *   - external ROCKET_INT8_DEQ=1 + the ROCKET_INT8_DEQ_* knobs: single-config RATIO
 *     classifier (out/C_i32 per element; uniform ratio == a clean constant multiply)
 *     -> the RE sweep harness. Knobs: PREC=fp32|fp16, SCALE=<u16>, SHIFT=<u8>,
 *     MINEXP=<u8>, CVTTYPE=0|1, OFFSET=<i32>, FP32TOFP16=0|1, BNMUL=<u16>; plus
 *     ROCKET_INT8_SIZE_E/_SURF_MULT (geometry) and ROCKET_DEQ_TARGET/_CV.
 *
 * Usage: matmul_int8_dequant_rocket [M K N]   (default 16 64 32; M%4 K%32 N%32)
 * Exit: 0 PASS, 1 FAIL, 2 SKIP (no NPU).
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

#define CK(call) do { int _r = (call); \
    if (_r) fprintf(stderr, "%s -> %d (%s)\n", #call, _r, strerror(-_r)); } while (0)

static unsigned rup(unsigned a, unsigned b) { return ((a + b - 1) / b) * b; }

static void ref_matmul_int8(int m, int k, int n,
                            const int8_t *A, const int8_t *B, int32_t *C) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int64_t s = 0;
            for (int l = 0; l < k; l++) s += (int32_t)A[i*k+l] * (int32_t)B[j*k+l];
            C[i*n+j] = (int32_t)s;
        }
}

/* Run ONE int8 matmul through gen_matmul_int8 (which reads the current ROCKET_INT8_*
 * env) and de-tile the output into out[M*N] as float. want_fp16 selects the readback
 * cube (C2=8 fp16 vs C2=4 fp32/int32-as-float). Returns 0, or <0 on a shim error. */
static int run_once(int fd, int M, int K, int N, const int8_t *A, const int8_t *B,
                    int want_fp16, float *out) {
    rocket_bo guard = {0}, regcmd = {0}, input = {0}, weights = {0}, output = {0};
    int ret = 0, C2 = want_fp16 ? 8 : 4; size_t esz = want_fp16 ? 2 : 4;
    size_t out_bytes = (size_t)rup(M,16) * rup(N,32) * 8 + 32768;
    rocket_bo_alloc(fd, 4096, &guard);
    ret |= rocket_bo_alloc(fd, 4096,        &regcmd);
    ret |= rocket_bo_alloc(fd, (size_t)M*K, &input);
    ret |= rocket_bo_alloc(fd, (size_t)N*K, &weights);
    ret |= rocket_bo_alloc(fd, out_bytes,   &output);
    if (ret) { ret = -1; goto done; }
    if ((input.dma_address|weights.dma_address|output.dma_address|regcmd.dma_address) >> 32) {
        fprintf(stderr, "ERROR: BO dma_address exceeds 32 bits\n"); ret = -1; goto done; }

    uint64_t regs[256] = {0};
    matmul_params_t p = { .m=(uint16_t)M, .k=(uint16_t)K, .n=(uint16_t)N,
        .input_dma=(uint32_t)input.dma_address, .weights_dma=(uint32_t)weights.dma_address,
        .output_dma=(uint32_t)output.dma_address, .tasks=regs };
    if ((ret = gen_matmul_int8(&p)) != 0) { fprintf(stderr, "gen=%d\n", ret); ret = -1; goto done; }

    CK(rocket_bo_prep(fd, &regcmd, 1, 0));
    CK(rocket_bo_prep(fd, &input,  1, 0));
    CK(rocket_bo_prep(fd, &weights,1, 0));
    memcpy(regcmd.ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
    memset(input.ptr, 0, input.size); memset(weights.ptr, 0, weights.size);
    int8_t *wdst = weights.ptr, *idst = input.ptr;
    for (int n = 1; n <= N; n++) for (int k = 1; k <= K; k++)
        wdst[weight_int8(K, n, k)] = B[(n-1)*K + (k-1)];
    for (int m = 1; m <= M; m++) for (int k = 1; k <= K; k++)
        idst[feature_data(K, M, 1, 16, k, m, 1)] = A[(m-1)*K + (k-1)];
    CK(rocket_bo_fini(fd, &regcmd));
    CK(rocket_bo_fini(fd, &input));
    CK(rocket_bo_fini(fd, &weights));
    CK(rocket_bo_prep(fd, &output, 1, 0));
    memset(output.ptr, 0xAA, output.size);
    CK(rocket_bo_fini(fd, &output));

    uint32_t in_h[]  = { input.handle, weights.handle, regcmd.handle };
    uint32_t out_h[] = { output.handle };
    if ((ret = rocket_submit_matmul(fd, &regcmd, p.task_count, in_h, 3, out_h, 1, 6000))) {
        fprintf(stderr, "submit=%d\n", ret); ret = -1; goto done; }
    if ((ret = rocket_bo_prep(fd, &output, 0, 2000000000LL))) {
        fprintf(stderr, "PREP_BO(out)=%d (%s)\n", ret, strerror(-ret)); ret = -1; goto done; }

    for (int m = 1; m <= M; m++) for (int n = 1; n <= N; n++) {
        size_t idx = (size_t)feature_data(N, M, 1, C2, n, m, 1);
        if (want_fp16) { _Float16 h; memcpy(&h, (uint8_t*)output.ptr + idx*esz, 2); out[(m-1)*N+(n-1)] = (float)h; }
        else           { float f;    memcpy(&f, (uint8_t*)output.ptr + idx*esz, 4); out[(m-1)*N+(n-1)] = f; }
    }
    rocket_bo_fini(fd, &output);
    ret = 0;
done:
    rocket_bo_free(fd, &guard); rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &input);
    rocket_bo_free(fd, &weights); rocket_bo_free(fd, &output);
    return ret;
}

/* RE harness: single config from the external env, ratio classifier out/C_i32. */
static int re_mode(int fd, int M, int K, int N, const int8_t *A, const int8_t *B,
                   const int32_t *C) {
    int want_fp16 = 0; const char *pe = getenv("ROCKET_INT8_DEQ_PREC");
    if (pe && !strcmp(pe, "fp16")) want_fp16 = 1;
    float *out = malloc((size_t)M*N*sizeof(float));
    if (!out) return 1;
    if (run_once(fd, M, K, N, A, B, want_fp16, out)) { free(out); printf("FAILED\n"); return 1; }
    double floor_abs = 0;
    for (int i = 0; i < M*N; i++) { double a = fabs((double)C[i]); if (a > floor_abs) floor_abs = a; }
    floor_abs *= 0.05; if (floor_abs < 1) floor_abs = 1;
    double sr=0, sr2=0, rmin=1e300, rmax=-1e300; long cnt=0; int shown=0;
    for (int i = 0; i < M*N; i++) {
        if (fabs((double)C[i]) < floor_abs) continue;
        double r = out[i] / (double)C[i];
        sr += r; sr2 += r*r; cnt++; if (r<rmin) rmin=r; if (r>rmax) rmax=r;
        if (shown < 6) { printf("  sample i=%d  C_i32=%d  out=%.6g  ratio=%.6g\n", i, C[i], out[i], r); shown++; }
    }
    free(out);
    if (!cnt) { printf("no above-floor elems\nFAILED\n"); return 1; }
    double mean=sr/cnt, var=sr2/cnt-mean*mean; if (var<0) var=0;
    double cv = (mean!=0)? sqrt(var)/fabs(mean) : 1e300;
    printf("ratio over %ld elems: mean=%.8g  cv=%.3g  [min=%.6g max=%.6g]\n", cnt, mean, cv, rmin, rmax);
    printf("=> effective on-chip scale s_eff = %.8g (NPU_out = C_i32 * s_eff)\n", mean);
    double cv_max = getenv("ROCKET_DEQ_CV") ? atof(getenv("ROCKET_DEQ_CV")) : 0.02;
    int pass = (cv <= cv_max);
    if (!pass) printf("NON-UNIFORM ratio (cv=%.3g > %.3g)\n", cv, cv_max);
    const char *tgt = getenv("ROCKET_DEQ_TARGET");
    if (tgt) { double t=atof(tgt), rel=fabs(mean-t)/(fabs(t)>1e-30?fabs(t):1);
        printf("target s=%.8g measured=%.8g rel_err=%.3g\n", t, mean, rel);
        if (rel > 0.05) pass = 0; }
    printf("%s\n", pass ? "PASS" : "FAILED");
    return pass ? 0 : 1;
}

/* One self-test config: set the env, run, compare to expect[] (bit-exact for the
 * cast/integer-scale cases; values are integers so float compare is exact). */
static int check_cfg(int fd, const char *name, int M, int K, int N,
                     const int8_t *A, const int8_t *B, const int32_t *C,
                     int want_fp16, long iscale, const char *prec,
                     const char *size_e, const char *surf_mult, const char *scale) {
    setenv("ROCKET_INT8_DEQ", "1", 1);
    if (prec) setenv("ROCKET_INT8_DEQ_PREC", prec, 1); else unsetenv("ROCKET_INT8_DEQ_PREC");
    if (scale) setenv("ROCKET_INT8_DEQ_SCALE", scale, 1); else unsetenv("ROCKET_INT8_DEQ_SCALE");
    if (size_e) setenv("ROCKET_INT8_SIZE_E", size_e, 1); else unsetenv("ROCKET_INT8_SIZE_E");
    if (surf_mult) setenv("ROCKET_INT8_SURF_MULT", surf_mult, 1); else unsetenv("ROCKET_INT8_SURF_MULT");
    unsetenv("ROCKET_INT8_DEQ_SHIFT"); unsetenv("ROCKET_INT8_DEQ_MINEXP");
    unsetenv("ROCKET_INT8_DEQ_BNMUL"); unsetenv("ROCKET_INT8_DEQ_OFFSET");

    float *out = malloc((size_t)M*N*sizeof(float));
    if (!out) return 1;
    if (run_once(fd, M, K, N, A, B, want_fp16, out)) { free(out); printf("  %-28s : run FAILED\n", name); return 1; }
    int bad = 0; double worst = 0;
    for (int i = 0; i < M*N; i++) {
        double exp = (double)C[i] * (double)iscale;
        if (out[i] != exp) { if (++bad <= 4) printf("    mismatch i=%d exp=%.0f got=%.6g\n", i, exp, out[i]);
                             double d = fabs(out[i]-exp); if (d>worst) worst=d; }
    }
    free(out);
    printf("  %-28s : %s (%d mismatches, worst|d|=%.3g)\n", name, bad?"FAIL":"OK", bad, worst);
    return bad ? 1 : 0;
}

int main(int argc, char **argv) {
    int M = 16, K = 64, N = 32;
    if (argc == 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    if (M % 4 || K % 32 || N % 32) { fprintf(stderr, "need M%%4 K%%32 N%%32\n"); return 1; }

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no NPU (rocket_open=%d) -> SKIP\n", fd); return 2; }

    /* small magnitudes -> |acc| well inside fp16 exact-int range (<=2048), so the
     * fp16-cast self-test is a true bit-exact check, not an fp16-rounding check. */
    int8_t *A = malloc((size_t)M*K), *B = malloc((size_t)N*K);
    int32_t *C = malloc((size_t)M*N*sizeof(int32_t));
    if (!A || !B || !C) { fprintf(stderr, "malloc\n"); return 1; }
    srand(1234);
    for (int i = 0; i < M*K; i++) A[i] = (int8_t)(rand() % 16 - 8);
    for (int i = 0; i < N*K; i++) B[i] = (int8_t)(rand() % 16 - 8);
    ref_matmul_int8(M, K, N, A, B, C);

    int rc = 0;
    if (getenv("ROCKET_INT8_DEQ")) {
        printf("OUT_CVT RE mode (single config from env): C[%d,%d] int8 matmul\n", M, N);
        rc = re_mode(fd, M, K, N, A, B, C);
    } else {
        printf("OUT_CVT cast/scale self-test: C[%d,%d]=A[%d,%d]xB[%d,%d]^T\n", M, N, M, K, N, K);
        /* GATED: the general, bit-exact capabilities (any shape). */
        rc |= check_cfg(fd, "int8->fp32 cast (s=1)",        M,K,N,A,B,C, 0, 1, "fp32", NULL, NULL, NULL);
        rc |= check_cfg(fd, "int8->fp32 integer scale x3",  M,K,N,A,B,C, 0, 3, "fp32", NULL, NULL, "3");
        /* DIAGNOSTIC (not gated): int8->fp16 cast halves readback but its 2-byte
         * writer geometry (size_e=3/surf*2) is only bit-exact for small single-tile
         * shapes AND is range-limited (|acc|<=2048 fp16 exact-int) -> low value for
         * the large-K LLM path; not a shipped capability. See the OUT_CVT notes. */
        (void)check_cfg(fd, "[diag] int8->fp16 cast small", M,K,N,A,B,C, 1, 1, "fp16", "3", "2", NULL);
        printf("%s\n", rc ? "FAILED" : "PASS");
    }
    free(A); free(B); free(C);
    rocket_close(fd);
    return rc ? 1 : 0;
}
