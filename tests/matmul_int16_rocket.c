// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * matmul_int16_rocket.c — standalone int16 x int16 -> int32 readiness
 * test + ENCODING SWEEP harness on the rocket driver. The int16 analogue of
 * matmul_int4_rocket: int16's register encodings are NOT documented anywhere (the
 * documented matmul modes are only fp16/int8/int4), so this CRACKS them on HW by sweep,
 * staged so each unknown is isolated:
 *
 *   STAGE 1 — does an int16 program RUN and WRITE int32 output? (precision value +
 *     output geometry, LAYOUT-INDEPENDENT). Watch for: WAIT TIMEOUT (bad/invalid
 *     precision value -> DPU never completes) vs "output still 0xAA" (ran but
 *     wrote nothing) vs "output touched" (precision + DPU enable correct). Sweep:
 *       ROCKET_INT16_PREC      in/proc precision (try 1=int16(NVDLA); rule out 3/7)
 *       ROCKET_INT16_OUT_PREC  output precision  (try 4=int32)
 *       ROCKET_INT16_SIZE_E / ROCKET_INT16_SURF_MULT  int32 output stride (try 7/8,
 *                              then 3/4 if needed — int8's int32-out broke the fp16
 *                              "size_e=bytes-1" rule, so 7/8 is the prime suspect)
 *     A "touched, int32-strided" result (even with WRONG values) clears stage 1.
 *
 *   STAGE 2 — are the VALUES correct? (weight/feature packing + cube C2s). Once
 *     stage 1 writes, fix the packing:
 *       ROCKET_INT16_C2        feature input cube (try 8 = fp16's 2-byte atom)
 *       ROCKET_INT16_OUT_C2    int32 output cube (try 4 = int8's int32 cube)
 *     The weight_int16 layout (npu_regcmd.c, prime suspect == weight_fp16) is
 *     the next suspect if feature/output cubes are right but values are still off.
 *
 * int32-output SATURATION (the int16-specific wrinkle): int16xint16 <= 32767^2
 * ~= 1.07e9 per product, so a sum of just TWO full-range products overflows int32.
 * The bit-exact test therefore uses SMALL int16 magnitudes (ROCKET_INT16_RANGE,
 * default 1024) so the single-K-pass sum stays within int32 and the int64 compare
 * is exact. SEPARATELY, ROCKET_INT16_BIG=1 drives near-full-range values to
 * CHARACTERIZE the overflow behaviour (saturate vs wrap vs wider-than-int32) — a
 * genuinely novel finding (int8/int4 never had an int32-output overflow regime).
 *
 * Small-shape constraint: the single-task test does ONE K-pass, so the output is
 * the full dot product over K. Keep K * RANGE^2 < 2^31 (e.g. RANGE=1024 -> K<2048)
 * or the "exact" mode itself overflows int32. Start at M=4 K=32 N=64 (N=64 spans
 * 16 int32 surfaces at C2=4 so the size_e quirk shows; K=32 keeps sums tiny).
 *
 * Usage: matmul_int16_rocket <M> <K> <N>   (M%4||1, K%32, N%16; N=64 recommended)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

#define CK(call) do { int _r = (call); \
    if (_r) fprintf(stderr, "%s -> %d (%s)\n", #call, _r, strerror(-_r)); } while (0)

static int64_t now_us(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

static int env_int(const char *name, int def) {
    const char *e = getenv(name); return e ? (int)strtol(e, NULL, 0) : def;
}

/* Output read index. ROCKET_INT16_TRANSPOSE=1 reads the TRANSPOSED cube
 * (M as channels / N as spatial) — for use with the gen's ROCKET_INT16_TP_ORG=1
 * "original transpose" output mode, which on HW writes the FULL M*N buffer (vs
 * the 1x16-tile truncation of the plain path). Default = the standard
 * (N-channels, M-spatial) cube. */
static int g_transpose = 0;
static size_t out_index(int N, int M, int out_c2, int n, int m) {
    return g_transpose ? (size_t)feature_data(M, N, 1, out_c2, m, n, 1)
                       : (size_t)feature_data(N, M, 1, out_c2, n, m, 1);
}

/* TP_ORG transposed int16-output element index (HW-CRACKED, strides
 * measured across M={4,8,16} x N={16,32,64}). The DPU "original transpose"
 * (tp_org_en) writes 8/16-bit elements, NOT int32; with tp_precision=1 the output
 * is int16 at this layout. 0-based m,n; na = n/4. ALL strides scale with M, NOT N:
 *   m            stride 4
 *   na%4         stride 1
 *   na/4         stride 4*M
 *   n%4 (lane)   stride 16*M
 * HW-VERIFIED bit-exact at N<=32 (8x32x32, 16x32x32 pass 100%). At N>=64 the
 * n/16 super term stops extrapolating linearly (n>=32 reads wrong), so the native
 * path is capped at per-task N<=32. The byte-decomp path (matmul_int16_exact) is
 * the general full-precision route. */
static size_t out_idx_i16_tp(int M, int N, int m, int n) {
    (void)N;
    int na = n / 4;
    return (size_t)4 * (size_t)m
         + (size_t)(na % 4)
         + (size_t)(na / 4) * (size_t)(4 * M)
         + (size_t)(n % 4) * (size_t)(16 * M);
}

/* int16 x int16 -> int64 reference (exact). Caller decides how the HW's int32
 * output should relate (== for the unsaturated/small-range test; saturate/wrap
 * classification for the BIG characterization). */
static void ref_matmul_int16(int m, int k, int n,
                             const int16_t *A, const int16_t *B, int64_t *C) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int64_t sum = 0;
            for (int l = 0; l < k; l++) sum += (int64_t)A[i*k + l] * (int64_t)B[j*k + l];
            C[i*n + j] = sum;
        }
}

static int32_t sat_i32(int64_t v) {
    if (v > INT32_MAX) return INT32_MAX;
    if (v < INT32_MIN) return INT32_MIN;
    return (int32_t)v;
}

int main(int argc, char **argv) {
    if (argc != 4) { printf("usage: %s <M> <K> <N>  (try 4 32 64)\n", argv[0]); return -1; }
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    if (M <= 0 || (M % 4 != 0 && M != 1)) { fprintf(stderr, "M must be %%4 or 1\n"); return -1; }
    if (K <= 0 || K % 32 != 0) { fprintf(stderr, "K must be %%32\n"); return -1; }
    if (N <= 0 || N % 16 != 0) { fprintf(stderr, "N must be %%16 (int16 N-group, like fp16)\n"); return -1; }

    const int c2     = env_int("ROCKET_INT16_C2", 8);      /* feature input cube  */
    const int out_c2 = env_int("ROCKET_INT16_OUT_C2", 4);  /* int32 output cube   */
    const int big    = env_int("ROCKET_INT16_BIG", 0);     /* overflow characterize */
    const int probe  = env_int("ROCKET_INT16_PROBE", 0);   /* layout-map probe (see below) */
    const int tp16   = env_int("ROCKET_INT16_TP16", 0);    /* read int16 TP_ORG layout + cmp */
    int range        = env_int("ROCKET_INT16_RANGE", 1024);
    if (range < 1) range = 1024;
    g_transpose      = env_int("ROCKET_INT16_TRANSPOSE", 0); /* read transposed cube (TP_ORG mode) */

    /* In BIG mode push near-full int16 range so sums exceed int32; warn if even
     * the small-range mode would overflow int32 at this K (compare would be moot). */
    if (big) range = 32767;
    else if ((double)K * range * range >= 2147483647.0)
        fprintf(stderr, "WARN: K*range^2 >= INT32_MAX (K=%d range=%d) — the exact "
                "compare may overflow int32; lower ROCKET_INT16_RANGE or K.\n", K, range);

    printf("int16 TEST%s: C[%d,%d] = A[%d,%d] x B[%d,%d]^T (int16xint16->int32)\n",
           big ? " [BIG/overflow]" : "", M, N, M, K, N, K);
    printf("knobs: PREC=%d OUT_PREC=%d QD_EN=%d SIZE_E=%d SURF_MULT=%d DENTRIES_DIV=%d C2=%d OUT_C2=%d range=%d\n",
           env_int("ROCKET_INT16_PREC", 1), env_int("ROCKET_INT16_OUT_PREC", 4),
           env_int("ROCKET_INT16_QD_EN", 1),
           env_int("ROCKET_INT16_SIZE_E", 7), env_int("ROCKET_INT16_SURF_MULT", 8),
           env_int("ROCKET_INT16_DENTRIES_DIV", 32), c2, out_c2, range);
    printf("write-out knobs: MC_SURF=%d TP_PREC=%d TP_ORG=%d SIZE_C=%d TRANSPOSE=%d PROBE=%d\n",
           env_int("ROCKET_INT16_MC_SURF", 0), env_int("ROCKET_INT16_TP_PREC", 0),
           env_int("ROCKET_INT16_TP_ORG", 0), env_int("ROCKET_INT16_SIZE_C", 0), g_transpose, probe);
    if (tp16) printf("  [TP16 mode: reading int16 output at cracked TP_ORG layout]\n");

    int fd = rocket_open();
    if (fd < 0) return fd;

    rocket_bo guard = {0}, regcmd = {0}, input = {0}, weights = {0}, output = {0};
    int ret = 0;
    size_t in_bytes  = (size_t)M * K * sizeof(int16_t);
    size_t wt_bytes  = (size_t)N * K * sizeof(int16_t);
    size_t out_bytes = (size_t)M * N * sizeof(int32_t);
    /* TP_ORG transposed int16 output spans a wider range than the int32 cube for
     * small-M/large-N (lane stride N*4 int16). Size the BO to cover it so tp16/
     * probe verification works across the ladder. Max element is at (M-1,N-1). */
    if (tp16 || probe) {
        size_t span_i16 = out_idx_i16_tp(M, N, M - 1, N - 1) + 1;
        size_t tp_bytes = span_i16 * sizeof(int16_t);
        if (tp_bytes > out_bytes) out_bytes = tp_bytes;
    }
    rocket_bo_alloc(fd, 4096, &guard);
    ret |= rocket_bo_alloc(fd, 4096,      &regcmd);
    ret |= rocket_bo_alloc(fd, in_bytes,  &input);
    ret |= rocket_bo_alloc(fd, wt_bytes,  &weights);
    ret |= rocket_bo_alloc(fd, out_bytes, &output);
    if (ret) { fprintf(stderr, "bo alloc failed\n"); goto out; }
    printf("dma  in=%lx w=%lx out=%lx regcmd=%lx  (in=%zuB wt=%zuB out=%zuB)\n",
           input.dma_address, weights.dma_address, output.dma_address, regcmd.dma_address,
           in_bytes, wt_bytes, out_bytes);
    if ((input.dma_address | weights.dma_address | output.dma_address | regcmd.dma_address) >> 32) {
        fprintf(stderr, "ERROR: a BO dma_address exceeds 32 bits\n"); ret = -1; goto out;
    }

    uint64_t npu_regs[256] = {0};
    matmul_params_t p = {
        .m = (uint16_t)M, .k = (uint16_t)K, .n = (uint16_t)N,
        .input_dma = (uint32_t)input.dma_address,
        .weights_dma = (uint32_t)weights.dma_address,
        .output_dma = (uint32_t)output.dma_address,
        .tasks = npu_regs,
    };
    if ((ret = gen_matmul_int16(&p)) != 0) {
        fprintf(stderr, "gen_matmul_int16 = %d (CBUF banks? -1 fd, -2 weight/kernel)\n", ret);
        goto out;
    }
    printf("regcmd ops = %u\n", p.task_count);

    int16_t *A = malloc((size_t)M * K * sizeof(int16_t));
    int16_t *B = malloc((size_t)N * K * sizeof(int16_t));
    int64_t *C = malloc((size_t)M * N * sizeof(int64_t));
    if (!A || !B || !C) { fprintf(stderr, "host malloc failed\n"); ret = -1; goto out_free; }
    srand(1234);
    if (probe) {
        /* LAYOUT-MAP PROBE: make every C[m,n] a unique, SMALL, decodable signature
         * so we can read the output buffer at ANY element size and see exactly
         * where each element lands. Signature: C[m,n] = m*N + (n+1), in [1, M*N]
         * — fits one byte if M*N<=255, always fits int16 if M*N<=65535. Built from
         * two K-lanes: A[m,0]=m · B[n,0]=N -> m*N, and A[m,1]=1 · B[n,1]=(n+1).
         * Decode of a value v: idx=v-1, m=idx/N, n=idx%N. Small values mean the
         * result survives whatever element size the TP_ORG transpose writes
         * (8/16-bit), so the dump below can identify the true output dtype. */
        if ((size_t)M*N > 65535)
            fprintf(stderr, "WARN: probe needs M*N <= 65535 (M=%d N=%d)\n", M, N);
        memset(A, 0, (size_t)M*K*sizeof(int16_t));
        memset(B, 0, (size_t)N*K*sizeof(int16_t));
        for (int m = 0; m < M; m++) { A[m*K + 0] = (int16_t)m; A[m*K + 1] = 1; }
        for (int n = 0; n < N; n++) { B[n*K + 0] = (int16_t)N; B[n*K + 1] = (int16_t)(n + 1); }
    } else {
        for (int i = 0; i < M*K; i++) A[i] = (int16_t)(rand() % (2*range) - range);
        for (int i = 0; i < N*K; i++) B[i] = (int16_t)(rand() % (2*range) - range);
    }

    CK(rocket_bo_prep(fd, &regcmd, 1, 0));
    CK(rocket_bo_prep(fd, &input,  1, 0));
    CK(rocket_bo_prep(fd, &weights,1, 0));
    memcpy(regcmd.ptr, npu_regs, (size_t)p.task_count * sizeof(uint64_t));
    memset(input.ptr, 0, input.size);
    memset(weights.ptr, 0, weights.size);
    /* pack into NPU native int16 layouts: weight_int16 (== weight_fp16), feature
     * cube C2 (default 8 = fp16's 2-byte atom). Whole-element stores (no nibbles). */
    int16_t *wdst = weights.ptr, *idst = input.ptr;
    for (int n = 1; n <= N; n++)
        for (int k = 1; k <= K; k++)
            wdst[weight_int16(K, n, k)] = B[(n-1)*K + (k-1)];
    for (int m = 1; m <= M; m++)
        for (int k = 1; k <= K; k++)
            idst[feature_data(K, M, 1, c2, k, m, 1)] = A[(m-1)*K + (k-1)];
    CK(rocket_bo_fini(fd, &regcmd));
    CK(rocket_bo_fini(fd, &input));
    CK(rocket_bo_fini(fd, &weights));

    CK(rocket_bo_prep(fd, &output, 1, 0));
    memset(output.ptr, 0xAA, output.size);     /* sentinel 0xAAAAAAAA */
    CK(rocket_bo_fini(fd, &output));

    ref_matmul_int16(M, K, N, A, B, C);

    uint32_t in_h[]  = { input.handle, weights.handle, regcmd.handle };
    uint32_t out_h[] = { output.handle };
    int64_t t0 = now_us();
    ret = rocket_submit_matmul(fd, &regcmd, p.task_count, in_h, 3, out_h, 1, 6000);
    printf("ROCKET_SUBMIT = %d  (%.3f ms)\n", ret, (now_us() - t0) / 1000.0);
    if (ret) goto out_free;

    int prc = rocket_bo_prep(fd, &output, 0, 2000000000LL);
    if (prc) {
        fprintf(stderr, "STAGE1 FAIL: PREP_BO(output) = %d (%s) — job never completed "
                "(precision value likely invalid; sweep ROCKET_INT16_PREC=1, rule out 3/7)\n",
                prc, strerror(-prc));
        ret = prc; goto out_free;
    }

    /* STAGE 1: touched? where? (0xAA sentinel == 0xAAAAAAAA as int32) */
    size_t first = out_bytes; int touched = 0;
    for (size_t i = 0; i < out_bytes; i++)
        if (((uint8_t *)output.ptr)[i] != 0xAA) { touched = 1; first = i; break; }
    if (!touched) {
        printf("STAGE1: output still 0xAA — ran but wrote NOTHING (DPU enable / out_precision). "
               "Sweep ROCKET_INT16_OUT_PREC=4, then SIZE_E/SURF_MULT=7/8.\n");
        rocket_bo_fini(fd, &output); ret = -1; goto out_free;
    }
    printf("STAGE1 OK: output TOUCHED (first non-0xAA byte @ %zu of %zu)\n", first, out_bytes);

    /* STAGE 2: values. read int32 at the guessed output cube C2 (default 4). */
    int32_t *od = output.ptr;
    if (probe) {
        /* ELEMENT-SIZE-AGNOSTIC LAYOUT MAP. Signature C[m,n]=m*N+(n+1) in [1,M*N].
         * We read the raw output BO as int8, int16, AND int32 and, for each size,
         * count how many slots hold an in-range signature value. The size whose
         * count is ~M*N (allowing replication) is the true output dtype; its
         * slot->(m,n) mapping IS the TP_ORG layout. This is robust to the HW
         * writing 8/16-bit transposed output (tp_precision) instead of int32. */
        const int8_t  *b8  = (const int8_t  *)output.ptr;
        const int16_t *b16 = (const int16_t *)output.ptr;
        const int32_t *b32 = (const int32_t *)output.ptr;
        size_t n8 = out_bytes, n16 = out_bytes/2, n32 = out_bytes/4;
        int lim = M * N;
        printf("PROBE: signature C[m,n]=m*N+(n+1) in [1,%d]; out_bytes=%zu\n", lim, out_bytes);

        /* per-size hit counts (a slot counts if its value is a valid signature) */
        #define CNT(arr, ne) ({ int _c=0; for (size_t _i=0;_i<(ne);_i++){ \
            long _v=(long)(arr)[_i]; if(_v>=1 && _v<=lim) _c++; } _c; })
        int c8 = CNT(b8, n8), c16 = CNT(b16, n16), c32 = CNT(b32, n32);
        printf("--- in-range signature hits by element size ---\n");
        printf("  int8 : %d / %zu slots\n", c8,  n8);
        printf("  int16: %d / %zu slots\n", c16, n16);
        printf("  int32: %d / %zu slots\n", c32, n32);
        #undef CNT

        /* For each candidate size, if its hit-count is plausible (>= M*N/2),
         * print the slot->(m,n) map so the layout is visible. Replicated bytes
         * (e.g. 0x01010101) show as 4 adjacent int8 slots with the same (m,n). */
        for (int pass = 0; pass < 3; pass++) {
            int esz = pass == 0 ? 1 : pass == 1 ? 2 : 4;
            size_t ne = pass == 0 ? n8 : pass == 1 ? n16 : n32;
            int cnt  = pass == 0 ? c8 : pass == 1 ? c16 : c32;
            if (cnt < lim / 2) continue;     /* not the output dtype */
            /* slot-of-(m,n): scan the buffer (this element size) for the signature */
            #define SLOT(mm,nn) ({ long _w=(long)(mm)*N+((nn)+1), _f=-1; \
                for (size_t _i=0;_i<ne;_i++){ long _v = esz==1?(long)b8[_i]: \
                    esz==2?(long)b16[_i]:(long)b32[_i]; if(_v==_w){_f=(long)_i;break;} } _f; })
            printf("=== STRIDE REPORT (int%d output) M=%d N=%d ===\n", esz*8, M, N);
            long base = SLOT(0,0);
            printf("base slot(0,0) = %ld\n", base);
            /* m-axis: slot(m,0) for m=0.. (delta reveals the m stride) */
            printf("m-axis slot(m,0): ");
            for (int m = 0; m < M && m < 12; m++) printf("%ld ", SLOT(m,0));
            printf("\n  (m-stride = %ld)\n", M>1 ? SLOT(1,0)-SLOT(0,0) : -1);
            /* n-axis: slot(0,n) for n=0..min(N,32) (reveals the multi-level n strides) */
            printf("n-axis slot(0,n): ");
            for (int n = 0; n < N && n < 32; n++) printf("%ld ", SLOT(0,n));
            printf("\n");
            /* key n strides at the boundaries that distinguish the levels */
            if (N > 1)  printf("  n%%4 lane stride  slot(0,1)-slot(0,0) = %ld\n", SLOT(0,1)-SLOT(0,0));
            if (N > 4)  printf("  n/4 step         slot(0,4)-slot(0,0) = %ld\n", SLOT(0,4)-SLOT(0,0));
            if (N > 16) printf("  n/16 super step  slot(0,16)-slot(0,0) = %ld\n", SLOT(0,16)-SLOT(0,0));
            #undef SLOT
        }
        ret = 0;
        rocket_bo_fini(fd, &output);
        goto out_free;
    }
    if (tp16) {
        /* NATIVE int16-OUTPUT verification via the cracked TP_ORG transposed
         * layout. The HW writes int16 (saturating) elements here, NOT int32, so
         * compare against the int16-saturated reference. Use a small range so
         * results fit int16 and the compare is meaningful (range<=16 -> |C|<=8k
         * at K=32). This PROVES the layout + the native int16->int16 primitive. */
        if (N > 32)
            printf("NOTE: native int16 layout is HW-verified only at N<=32; N=%d may "
                   "fail at n>=32 (use the byte-decomp path for large N)\n", N);
        const int16_t *o16 = (const int16_t *)output.ptr;
        int bad = 0, shown = 0, sat = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int64_t tru = C[m*N + n];
                int16_t want = tru > 32767 ? 32767 : tru < -32768 ? -32768 : (int16_t)tru;
                if (tru != want) sat++;
                int16_t act = o16[out_idx_i16_tp(M, N, m, n)];
                if (act != want) {
                    if (shown++ < 12)
                        printf("  mismatch m=%d n=%d true=%lld want(i16)=%d act=%d\n",
                               m, n, (long long)tru, want, act);
                    bad++;
                }
            }
        printf("STAGE2 [TP16]: %d/%d wrong (%d would saturate int16)\n", bad, M*N, sat);
        if (bad == 0) printf("==> PASS (native int16xint16->int16 via TP_ORG transpose, bit-exact vs int16-sat ref)\n");
        else printf("==> values wrong — layout out_idx_i16_tp(M,N,m,n) needs revisit\n");
        ret = bad ? -1 : 0;
        rocket_bo_fini(fd, &output);
        goto out_free;
    }
    if (big) {
        /* ---- OVERFLOW CHARACTERIZATION: classify each output element against the
         * true int64 dot product. If the output is plain int32 it must match either
         * the SATURATED (clamp to int32 range) or WRAPPED (low 32 bits) value; if
         * MOST land in "other" (neither), the output isn't plain int32 — that is the
         * signal it might be wider/encoded differently, to inspect by hand. */
        int n_sat = 0, n_wrap = 0, n_exact = 0, n_other = 0, n_sentinel = 0, shown = 0;
        for (int m = 1; m <= M; m++)
            for (int n = 1; n <= N; n++) {
                int64_t tru = C[(m-1)*N + (n-1)];
                int32_t act = od[out_index(N, M, out_c2, n, m)];
                int32_t sa  = sat_i32(tru);
                int32_t wr  = (int32_t)(uint32_t)(tru & 0xFFFFFFFFLL);
                if ((uint32_t)act == 0xAAAAAAAAu) n_sentinel++;
                else if ((int64_t)act == tru)     n_exact++;
                else if (act == sa)               n_sat++;
                else if (act == wr)               n_wrap++;
                else { n_other++;
                    if (shown < 8) { printf("  m=%d n=%d true=%lld sat=%d wrap=%d act=%d\n",
                                            m, n, (long long)tru, sa, wr, act); shown++; } }
            }
        printf("STAGE2 [BIG]: exact=%d saturate=%d wrap=%d other=%d sentinel=%d (of %d)\n",
               n_exact, n_sat, n_wrap, n_other, n_sentinel, M*N);
        printf("==> int16 int32-output overflow behaviour: %s\n",
               n_exact == M*N ? "no overflow at this shape (raise K/ROCKET_INT16_RANGE)" :
               n_sat >= n_wrap && n_sat > n_other ? "SATURATES (clamps to int32 range)" :
               n_wrap > n_sat && n_wrap > n_other ? "WRAPS (low 32 bits)" :
               "UNCLEAR — many 'other': output may be wider than int32, inspect above");
        ret = 0;  /* characterization, not pass/fail */
    } else {
        int bad = 0, shown = 0; long maxabs = 0; int any_match = 0;
        int row0_prefix = 0, prefix_open = 1;    /* contiguous correct cols in row m=1 */
        int row0_correct = 0;                    /* total correct cols in row m=1      */
        int sentinel = 0;                        /* outputs left as 0xAAAAAAAA          */
        for (int m = 1; m <= M; m++)
            for (int n = 1; n <= N; n++) {
                int32_t act = od[out_index(N, M, out_c2, n, m)];
                int64_t exp = C[(m-1)*N + (n-1)];
                if ((uint32_t)act == 0xAAAAAAAAu) sentinel++;
                if ((int64_t)act == exp) {
                    any_match = 1;
                    if (m == 1) { row0_correct++; if (prefix_open) row0_prefix++; }
                } else if (m == 1) {
                    prefix_open = 0;
                }
                long d = labs((long)((int64_t)act - exp));
                if (d > maxabs) maxabs = d;
                if ((int64_t)act != exp) {
                    if (shown < 8) { printf("  mismatch m=%d n=%d exp=%lld act=%d\n",
                                            m, n, (long long)exp, act); shown++; }
                    bad++;
                }
            }
        printf("STAGE2: %d/%d wrong, max_abs=%ld, any_match=%d | row0: %d/%d correct "
               "(contiguous prefix n=1..%d) | %d sentinel(0xAAAAAAAA)\n",
               bad, M*N, maxabs, any_match, row0_correct, N, row0_prefix, sentinel);
        if (bad == 0) { printf("==> PASS (bit-exact int16xint16->int32)\n"); ret = 0; }
        else {
            printf("==> values wrong (stage1 cleared). Crack packing: ROCKET_INT16_C2=8, "
                   "ROCKET_INT16_OUT_C2=4, and the weight_int16 layout (== weight_fp16?).\n");
            ret = -1;
        }
    }
    rocket_bo_fini(fd, &output);

out_free:
    free(A); free(B); free(C);
out:
    rocket_bo_free(fd, &guard);
    rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &input);
    rocket_bo_free(fd, &weights); rocket_bo_free(fd, &output);
    rocket_close(fd);
    return ret;
}
