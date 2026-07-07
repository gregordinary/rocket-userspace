// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * ffn_fused_rocket.c — HW gate for the CROSS-OP cube-resident FFN (rocket_ffn_fp16_fused):
 * the gated MLP whose [M,I] intermediates stay in NPU cube layout across
 * gate/up -> act(gate)⊙up -> down, so the host never de-tiles/re-tiles them (the projections
 * leave full output cubes, the activation runs element-wise over the cube bytes, and the down
 * matmul reads the product cube directly as its input).
 *
 * Two checks per shape, the right metrics for a multi-op fp16+LUT block whose intermediates
 * are aliased BO-to-BO:
 *   1. ORACLE   — cosine vs an fp64 reference (the pass criterion; a layout/aliasing
 *                 corruption collapses cosine, fp16+LUT rounding does not).
 *   2. NEAR-BIT — vs the validated host-handoff rocket_ffn_fp16 (same math, the only numeric
 *                 difference is the down matmul's K-tiling, so cosine must be ~1 and the
 *                 per-element gap small): proves the cube-resident handoff matches the
 *                 round-trip it replaces.
 *
 * The shapes exercise multi-M-tile, multi-N-tile and multi-K-tile producers, the SigLIP fc
 * shape (M=1024,H=768,I=3072 -> 48 output tiles), and a shape that exceeds one NPU batch so
 * the fused path transparently falls back to the host handoff (still correct).
 *
 * Inputs are scaled so the gate logits stay inside the SiLU/GELU LUT band.
 *
 *   sudo -E ./ffn_fused_rocket
 *   ROCKET_FFN_DEBUG=1 sudo -E ./ffn_fused_rocket   # log fused-vs-fallback routing per shape
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_ffn.h"
#include "rocket_activation.h"   /* ROCKET_ACTIVATION_SILU / _GELU */

static int g_fail = 0;

static void fill(_Float16 *v, size_t n, float amp, uint32_t seed)
{
    uint32_t st = 0x9e3779b9u ^ seed;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        float u = (float)((st >> 8) & 0xffff) / 65535.f;
        v[i] = (_Float16)((u * 2.f - 1.f) * amp);
    }
}

static double act_exact(int kind, double g)
{
    if (kind == ROCKET_ACTIVATION_GELU) return 0.5 * g * (1.0 + erf(g * M_SQRT1_2));
    return g / (1.0 + exp(-g));   /* SiLU */
}

/* fp64 oracle: out = (act(x·Wg^T) ⊙ (x·Wu^T)) · Wd^T */
static void oracle(int M, int H, int I, const _Float16 *x, const _Float16 *Wg,
                   const _Float16 *Wu, const _Float16 *Wd, int kind, double *out)
{
    double *prod = malloc((size_t)I * sizeof(double));
    for (int m = 0; m < M; m++) {
        const _Float16 *xr = x + (size_t)m * H;
        for (int i = 0; i < I; i++) {
            const _Float16 *wg = Wg + (size_t)i * H, *wu = Wu + (size_t)i * H;
            double g = 0, u = 0;
            for (int h = 0; h < H; h++) { g += (double)xr[h]*(double)wg[h];
                                          u += (double)xr[h]*(double)wu[h]; }
            prod[i] = act_exact(kind, g) * u;
        }
        for (int h = 0; h < H; h++) {
            const _Float16 *wd = Wd + (size_t)h * I;
            double a = 0;
            for (int i = 0; i < I; i++) a += prod[i] * (double)wd[i];
            out[(size_t)m * H + h] = a;
        }
    }
    free(prod);
}

static double cos_vs_dbl(const _Float16 *got, const double *ref, size_t n)
{
    double dot = 0, ng = 0, nr = 0;
    for (size_t i = 0; i < n; i++) {
        double a = (double)got[i], b = ref[i];
        if (!isfinite(a)) return -1.0;
        dot += a*b; ng += a*a; nr += b*b;
    }
    return (ng > 0 && nr > 0) ? dot / (sqrt(ng) * sqrt(nr)) : (ng == 0 && nr == 0 ? 1.0 : 0.0);
}

/* cosine + max-abs of two fp16 results (the fused path vs the host-handoff path) */
static void cos_vs_f16(const _Float16 *a, const _Float16 *b, size_t n,
                       double *cos, double *max_abs)
{
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (size_t i = 0; i < n; i++) {
        double x = (double)a[i], y = (double)b[i];
        dot += x*y; na += x*x; nb += y*y;
        double d = fabs(x - y); if (d > mx) mx = d;
    }
    *cos = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : (na == 0 && nb == 0 ? 1.0 : 0.0);
    *max_abs = mx;
}

static void test(int fd, int M, int H, int I, int kind)
{
    const char *kn = (kind == ROCKET_ACTIVATION_GELU) ? "gelu" : "silu";
    size_t nx = (size_t)M*H, nw = (size_t)I*H, ny = (size_t)M*H;
    _Float16 *x  = malloc(nx*sizeof(_Float16));
    _Float16 *Wg = malloc(nw*sizeof(_Float16));
    _Float16 *Wu = malloc(nw*sizeof(_Float16));
    _Float16 *Wd = malloc((size_t)H*I*sizeof(_Float16));
    _Float16 *fused = malloc(ny*sizeof(_Float16));
    _Float16 *host  = malloc(ny*sizeof(_Float16));
    double   *ref   = malloc(ny*sizeof(double));
    if (!x||!Wg||!Wu||!Wd||!fused||!host||!ref) { fprintf(stderr,"oom\n"); g_fail=1; goto done; }

    /* keep gate logits (sum over H) inside the LUT band: amps scaled by 1/sqrt(H) */
    float a = 3.0f / sqrtf((float)H);
    fill(x,  nx, 1.0f, 1u + kind);
    fill(Wg, nw, a,    2u + kind);
    fill(Wu, nw, a,    3u + kind);
    fill(Wd, (size_t)H*I, 1.5f/sqrtf((float)I), 4u + kind);

    oracle(M, H, I, x, Wg, Wu, Wd, kind, ref);

    /* sentinel so an unwritten output is distinct from a wrong one */
    for (size_t i = 0; i < ny; i++) ((uint16_t*)fused)[i] = 0xAAAA;
    int rf = rocket_ffn_fp16_fused(fd, M, H, I, x, Wg, Wu, Wd, kind, fused);
    int rh = rocket_ffn_fp16      (fd, M, H, I, x, Wg, Wu, Wd, kind, host);
    if (rf != 0 || rh != 0) {
        printf("  FFN %s M=%d H=%d I=%d: call rc fused=%d host=%d -> FAIL\n", kn,M,H,I,rf,rh);
        g_fail = 1; goto done;
    }
    double cos_o = cos_vs_dbl(fused, ref, ny);
    double cos_h, max_h;
    cos_vs_f16(fused, host, ny, &cos_h, &max_h);

    /* "unwritten" = an output the down matmul never touched (still the sentinel). A real
     * fp16 result can coincidentally equal the 0xAAAA bit pattern, so only flag a sentinel
     * lane whose host-handoff value also DISAGREES (a genuine gap differs from host by ~the
     * sentinel magnitude, not the ~1-ULP fused-vs-host noise). */
    size_t unwritten = 0;
    for (size_t i = 0; i < ny; i++)
        if (((uint16_t*)fused)[i] == 0xAAAA &&
            fabs((double)fused[i] - (double)host[i]) > 0.02) unwritten++;

    /* magnitude scale for a relative read on the host-handoff gap */
    double maxref = 0; for (size_t i=0;i<ny;i++){double v=fabs(ref[i]); if(v>maxref)maxref=v;}

    int ok = (unwritten == 0) && (cos_o >= 0.999) && (cos_h >= 0.9995);
    printf("  FFN %s M=%4d H=%4d I=%4d: cos(oracle)=%.6f  vs-host cos=%.6f max_abs=%.4g (|ref|max=%.3g) unwritten=%zu -> %s\n",
           kn, M, H, I, cos_o, cos_h, max_h, maxref, unwritten, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;

done:
    free(x); free(Wg); free(Wu); free(Wd); free(fused); free(host); free(ref);
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no NPU (rocket_open failed) -> SKIP\n"); return 2; }

    printf("cross-op cube-resident FFN gate (rocket_ffn_fp16_fused)\n");
    /* multi-N-tile / multi-M-tile / multi-K-tile producers + the SigLIP fc shape */
    test(fd, 128,  256,  512, ROCKET_ACTIVATION_SILU);   /* nMt=1 nNt=2  nKt=1            */
    test(fd, 512, 2048, 1024, ROCKET_ACTIVATION_SILU);   /* nMt=2 nNt=4, producer K multi */
    test(fd, 256,  768, 3072, ROCKET_ACTIVATION_GELU);   /* nNt=12 (SigLIP-ish I)         */
    test(fd, 1024, 768, 3072, ROCKET_ACTIVATION_SILU);   /* SigLIP fc: 4x12 = 48 tiles    */
    test(fd, 2048, 512, 8192, ROCKET_ACTIVATION_SILU);   /* 8x32 > batch -> host fallback */

    rocket_close(fd);
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 3 : 0;
}
