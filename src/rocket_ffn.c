// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_ffn.c — the gated-MLP transformer FFN block on the NPU: the gated-activation core
 * (GeGLU/SwiGLU) plus the block that wraps it between the three projection matmuls. See
 * rocket_ffn.h for the composition and the resident-fusion note.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "rocket_ffn.h"
#include "rocket_matmul.h"      /* rocket_matmul_fp16_mt (the projections)        */
#include "rocket_activation.h"  /* rocket_activation_fp16 + rocket_ew_mul_fp16    */
#include "rocket_npu.h"             /* rocket_bo_alloc/free/prep/fini (the cube BOs)  */
#include "rocket_matmul_internal.h" /* mm_plan/mm_bos + the full-cube KACC primitive  */

#define FFN_BATCH 64   /* mirrors rocket_matmul.c BATCH — the single-batch full-cube cap */

/* exact activation (matches the NPU LUT kinds: SiLU = x*sigmoid(x); GELU = exact erf) */
static double act_exact(int kind, double g)
{
    if (kind == ROCKET_ACTIVATION_GELU) return 0.5 * g * (1.0 + erf(g * M_SQRT1_2));
    return g / (1.0 + exp(-g));   /* SiLU / swish (default) */
}

/* ============================================================================
 * SECTION — gated activation (GeGLU / SwiGLU core)
 * ==========================================================================*/

void rocket_geglu_ref_fp16(const _Float16 *gate, const _Float16 *up,
                           int kind, _Float16 *prod, int n)
{
    for (int i = 0; i < n; i++)
        prod[i] = (_Float16)(act_exact(kind, (double)gate[i]) * (double)up[i]);
}

int rocket_geglu_fp16(int fd, const _Float16 *gate, const _Float16 *up,
                      int kind, _Float16 *prod, int n)
{
    if (n <= 0) return -1;
    if (fd < 0) { rocket_geglu_ref_fp16(gate, up, kind, prod, n); return 0; }

    _Float16 *act_g = malloc((size_t)n * sizeof(_Float16));
    if (!act_g) return -2;
    int r = rocket_activation_fp16(fd, kind, gate, act_g, n);   /* act(gate) on the NPU */
    if (r == 0) r = rocket_ew_mul_fp16(fd, act_g, up, prod, n); /* ⊙ up on the NPU      */
    free(act_g);
    return r;
}

/* ============================================================================
 * SECTION — full FFN block
 * ==========================================================================*/

void rocket_ffn_ref_fp16(int M, int H, int I,
                         const _Float16 *x, const _Float16 *Wg, const _Float16 *Wu,
                         const _Float16 *Wd, int kind, _Float16 *out)
{
    /* fp64 oracle: gate/up = x·W^T, prod = act(gate)⊙up, out = prod·Wd^T */
    for (int m = 0; m < M; m++) {
        const _Float16 *xr = x + (size_t)m * H;
        double *prod = malloc((size_t)I * sizeof(double));
        for (int i = 0; i < I; i++) {
            const _Float16 *wg = Wg + (size_t)i * H, *wu = Wu + (size_t)i * H;
            double g = 0, u = 0;
            for (int h = 0; h < H; h++) { g += (double)xr[h]*(double)wg[h];
                                          u += (double)xr[h]*(double)wu[h]; }
            prod[i] = act_exact(kind, g) * u;
        }
        _Float16 *orow = out + (size_t)m * H;
        for (int h = 0; h < H; h++) {
            const _Float16 *wd = Wd + (size_t)h * I;
            double acc = 0;
            for (int i = 0; i < I; i++) acc += prod[i] * (double)wd[i];
            orow[h] = (_Float16)acc;
        }
        free(prod);
    }
}

int rocket_ffn_fp16(int fd, int M, int H, int I,
                    const _Float16 *x, const _Float16 *Wg, const _Float16 *Wu,
                    const _Float16 *Wd, int kind, _Float16 *out)
{
    if (M < 1 || H < 1 || I < 1) return -1;
    /* The geglu element count M*I is passed as int below; reject a product that
     * would truncate (the buffers are sized in size_t, so the truncation would
     * under-process the op, not over-read). */
    if ((size_t)M * I > INT_MAX) return -1;
    if (fd < 0) { rocket_ffn_ref_fp16(M, H, I, x, Wg, Wu, Wd, kind, out); return 0; }

    const int NT = 3;                       /* fan the projections across the 3 NPU cores */
    int rc = -2;
    _Float16 *gate = malloc((size_t)M * I * sizeof(_Float16));
    _Float16 *up   = malloc((size_t)M * I * sizeof(_Float16));
    _Float16 *prod = malloc((size_t)M * I * sizeof(_Float16));
    if (!gate || !up || !prod) goto out;

    /* gate = x·Wg^T,  up = x·Wu^T  (Wg/Wu share x — fuse-able; correctness uses two mt calls) */
    if ((rc = rocket_matmul_fp16_mt(M, H, I, x, Wg, gate, NT)) != 0) goto out;
    if ((rc = rocket_matmul_fp16_mt(M, H, I, x, Wu, up,   NT)) != 0) goto out;

    /* prod = act(gate) ⊙ up  (the gated-activation core, on the NPU) */
    if ((rc = rocket_geglu_fp16(fd, gate, up, kind, prod, M * I)) != 0) goto out;

    /* out = prod·Wd^T */
    rc = rocket_matmul_fp16_mt(M, I, H, prod, Wd, out, NT);

out:
    free(prod); free(up); free(gate);
    return rc;
}

/* ---- cross-op cube-resident FFN ----------------------------------------------
 * Keep the [M,I] intermediates in NPU cube layout across gate/up -> act⊙up -> down.
 * See rocket_ffn.h for the contract and the cross-op-chaining mechanism. */
int rocket_ffn_fp16_fused(int fd, int M, int H, int I,
                          const _Float16 *x, const _Float16 *Wg, const _Float16 *Wu,
                          const _Float16 *Wd, int kind, _Float16 *out)
{
    if (M < 1 || H < 1 || I < 1) return -1;
    if ((size_t)M * I > INT_MAX) return -1;
    if (fd < 0) { rocket_ffn_ref_fp16(M, H, I, x, Wg, Wu, Wd, kind, out); return 0; }

    /* Shared cross-op tile: Nt(gate/up) == Kt(down), a multiple of 32, so the gate/up
     * output cubes alias the down matmul's input feature cube tile-for-tile
     * (out_slot == in_slot). MAX_TILE (256) is the natural choice. */
    const int T = 256;
    mm_plan plg, plu, pld;
    if (mm_plan_init_pin(&plg, M, H, I, T, 0) < 0 ||   /* gate: N=I tiled to T   */
        mm_plan_init_pin(&plu, M, H, I, T, 0) < 0 ||   /* up:   N=I tiled to T   */
        mm_plan_init_pin(&pld, M, I, H, 0, T) < 0)     /* down: K=I tiled to T   */
        return rocket_ffn_fp16(fd, M, H, I, x, Wg, Wu, Wd, kind, out);

    /* Matched-tiling + single-batch preconditions. If any fails, the cube cannot alias
     * the consumer's input transparently — fall back to the (correct) host-handoff FFN. */
    size_t cn = mm_cube_elems(&plg);
    if (!(plg.nNt == pld.nKt && plg.out_slot == pld.in_slot &&
          plg.nMt == pld.nMt && plg.Mt == pld.Mt &&
          plg.nMt * plg.nNt <= FFN_BATCH && cn <= (size_t)INT_MAX)) {
        if (getenv("ROCKET_FFN_DEBUG"))
            ROCKET_LOGI("ffn_fused: shape M=%d H=%d I=%d not cube-matchable -> host-handoff\n", M, H, I);
        return rocket_ffn_fp16(fd, M, H, I, x, Wg, Wu, Wd, kind, out);
    }
    if (getenv("ROCKET_FFN_DEBUG"))
        ROCKET_LOGI("ffn_fused: M=%d H=%d I=%d cube-resident (nMt=%d nNt=%d nKt[gate]=%d Mt=%d)\n",
                    M, H, I, plg.nMt, plg.nNt, plg.nKt, plg.Mt);

    int rc = -2, hg = 0, hu = 0, hd = 0;
    mm_bos bg, bu, bd;
    rocket_bo cube_g = {0}, cube_u = {0};
    size_t cube_bytes = cn * sizeof(_Float16) + 65536;   /* + a CBUF-bank of slack */

    if (mm_bos_alloc(fd, &plg, &bg) < 0) goto fb;
    hg = 1;
    if (mm_bos_alloc(fd, &plu, &bu) < 0) goto fb;
    hu = 1;
    if (mm_bos_alloc(fd, &pld, &bd) < 0) goto fb;
    hd = 1;
    if (rocket_bo_alloc(fd, cube_bytes, &cube_g) < 0) goto fb;
    if (rocket_bo_alloc(fd, cube_bytes, &cube_u) < 0) goto fb;

    /* gate = x·Wg^T, up = x·Wu^T — each left as a FULL output cube (no de-tile). */
    mm_pack_input(fd,   &plg, &bg, x);
    mm_pack_weights(fd, &plg, &bg, Wg);
    mm_pack_input(fd,   &plu, &bu, x);
    mm_pack_weights(fd, &plu, &bu, Wu);
    if (mm_compute_kacc_cube(fd, &plg, &bg, &cube_g, 0.0) != 0) goto fb;   /* incl. E_TILING -> fall back */
    if (mm_compute_kacc_cube(fd, &plu, &bu, &cube_u, 0.0) != 0) goto fb;

    /* prod = act(gate) ⊙ up, element-wise over the cube bytes (commutes with the cube
     * reindex; pad lanes stay 0). In-place into cube_g — both ops read each chunk fully
     * before writing it. */
    if ((rc = rocket_activation_fp16(fd, kind, (const _Float16 *)cube_g.ptr,
                                     (_Float16 *)cube_g.ptr, (int)cn)) != 0) goto fb_free;
    if ((rc = rocket_ew_mul_fp16(fd, (const _Float16 *)cube_g.ptr,
                                 (const _Float16 *)cube_u.ptr,
                                 (_Float16 *)cube_g.ptr, (int)cn)) != 0) goto fb_free;

    /* hand the product cube to the device (the activation/ew wrote it via the CPU map). */
    rocket_bo_prep(fd, &cube_g, 1, 0);
    rocket_bo_fini(fd, &cube_g);

    /* out = prod·Wd^T — the down matmul reads the product cube DIRECTLY as its input BO
     * (alias in_all, same IOVA, no host touch of the [M,I] intermediate). */
    mm_pack_weights(fd, &pld, &bd, Wd);
    {
        rocket_bo saved = bd.in_all;
        bd.in_all = cube_g;
        rc = mm_compute_kacc(fd, &pld, &bd, out, 0.0);
        if (rc == ROCKET_E_TILING) rc = mm_compute(fd, &pld, &bd, out, 0.0);  /* tiny-M */
        bd.in_all = saved;   /* restore so mm_bos_free owns the real in_all */
    }

fb_free:
    rocket_bo_free(fd, &cube_g); rocket_bo_free(fd, &cube_u);
    if (hd) mm_bos_free(fd, &bd);
    if (hu) mm_bos_free(fd, &bu);
    if (hg) mm_bos_free(fd, &bg);
    return rc;

fb:
    rocket_bo_free(fd, &cube_g); rocket_bo_free(fd, &cube_u);
    if (hd) mm_bos_free(fd, &bd);
    if (hu) mm_bos_free(fd, &bu);
    if (hg) mm_bos_free(fd, &bg);
    return rocket_ffn_fp16(fd, M, H, I, x, Wg, Wu, Wd, kind, out);
}

/* ============================================================================
 * SECTION — cross-op cube-resident non-gated MLP (encoder FFN)
 * ==========================================================================*/

/* add per-channel bias b[N] over the M rows of a row-major C[M,N] (host glue) */
static void mlp_add_bias(_Float16 *C, const _Float16 *b, int M, int N)
{
    if (!b) return;
    for (int m = 0; m < M; m++) {
        _Float16 *r = C + (size_t)m * N;
        for (int n = 0; n < N; n++) r[n] = (_Float16)((float)r[n] + (float)b[n]);
    }
}

/* host reference + host-handoff NPU fallback: out = act(x·W1^T + b1)·W2^T */
static int mlp_host_handoff(int fd, int M, int Din, int Dff, int Dout,
                            const _Float16 *x, const _Float16 *W1, const _Float16 *b1,
                            const _Float16 *W2, int kind, _Float16 *out)
{
    int rc = -2;
    _Float16 *f1  = malloc((size_t)M * Dff * sizeof(_Float16));
    _Float16 *act = malloc((size_t)M * Dff * sizeof(_Float16));
    if (!f1 || !act) { free(f1); free(act); return -2; }

    if (fd < 0) {
        /* fp16-rounded host reference */
        for (int m = 0; m < M; m++) {
            const _Float16 *xr = x + (size_t)m * Din;
            for (int o = 0; o < Dff; o++) {
                const _Float16 *w = W1 + (size_t)o * Din;
                double a = b1 ? (double)b1[o] : 0.0;
                for (int i = 0; i < Din; i++) a += (double)xr[i] * (double)w[i];
                act[(size_t)m*Dff+o] = (_Float16)act_exact(kind, a);
            }
            _Float16 *orow = out + (size_t)m * Dout;
            for (int o = 0; o < Dout; o++) {
                const _Float16 *w = W2 + (size_t)o * Dff;
                double a = 0;
                for (int i = 0; i < Dff; i++) a += (double)act[(size_t)m*Dff+i] * (double)w[i];
                orow[o] = (_Float16)a;
            }
        }
        rc = 0; goto out;
    }

    if ((rc = rocket_matmul_fp16(fd, M, Din, Dff, x, W1, f1)) != 0) goto out;
    mlp_add_bias(f1, b1, M, Dff);
    if ((rc = rocket_activation_fp16(fd, kind, f1, act, M * Dff)) != 0) goto out;   /* full 2-pass */
    rc = rocket_matmul_fp16(fd, M, Dff, Dout, act, W2, out);

out:
    free(f1); free(act);
    return rc;
}

int rocket_mlp_fp16_fused(int fd, int M, int Din, int Dff, int Dout,
                          const _Float16 *x, const _Float16 *W1, const _Float16 *b1,
                          const _Float16 *W2, int kind, _Float16 *out)
{
    if (M < 1 || Din < 1 || Dff < 1 || Dout < 1) return -1;
    if ((size_t)M * Dff > INT_MAX) return -1;
    if (fd < 0) return mlp_host_handoff(fd, M, Din, Dff, Dout, x, W1, b1, W2, kind, out);

    const int T = 256;   /* shared tile: Nt(fc1) == Kt(fc2), a multiple of 32 */
    mm_plan p1, p2;
    if (mm_plan_init_pin(&p1, M, Din, Dff, T, 0) < 0 ||   /* fc1: N=Dff tiled to T */
        mm_plan_init_pin(&p2, M, Dff, Dout, 0, T) < 0)    /* fc2: K=Dff tiled to T */
        return mlp_host_handoff(fd, M, Din, Dff, Dout, x, W1, b1, W2, kind, out);

    size_t cn = mm_cube_elems(&p1);
    if (!(p1.nNt == p2.nKt && p1.out_slot == p2.in_slot &&
          p1.nMt == p2.nMt && p1.Mt == p2.Mt &&
          p1.nMt * p1.nNt <= FFN_BATCH && cn <= (size_t)INT_MAX)) {
        if (getenv("ROCKET_FFN_DEBUG"))
            ROCKET_LOGI("mlp_fused: M=%d Din=%d Dff=%d not cube-matchable -> host-handoff\n", M, Din, Dff);
        return mlp_host_handoff(fd, M, Din, Dff, Dout, x, W1, b1, W2, kind, out);
    }
    if (getenv("ROCKET_FFN_DEBUG"))
        ROCKET_LOGI("mlp_fused: M=%d Din=%d Dff=%d Dout=%d cube-resident (nMt=%d nNt=%d nKt[fc1]=%d)\n",
                    M, Din, Dff, Dout, p1.nMt, p1.nNt, p1.nKt);

    int rc = -2, h1 = 0, h2 = 0;
    mm_bos b1m, b2m;
    rocket_bo cube = {0};
    _Float16 *bias_cube = NULL;
    size_t cube_bytes = cn * sizeof(_Float16) + 65536;

    if (mm_bos_alloc(fd, &p1, &b1m) < 0) goto fb;
    h1 = 1;
    if (mm_bos_alloc(fd, &p2, &b2m) < 0) goto fb;
    h2 = 1;
    if (rocket_bo_alloc(fd, cube_bytes, &cube) < 0) goto fb;

    /* fc1 = x·W1^T, left as a FULL output cube (no de-tile) */
    mm_pack_input(fd,   &p1, &b1m, x);
    mm_pack_weights(fd, &p1, &b1m, W1);
    if (mm_compute_kacc_cube(fd, &p1, &b1m, &cube, 0.0) != 0) goto fb;

    /* + b1 on the cube: scatter the per-channel bias into cube layout, flat ew-add */
    if (b1) {
        bias_cube = malloc(cn * sizeof(_Float16));
        if (!bias_cube) goto fb;
        mm_scatter_bias_cube(&p1, bias_cube, b1);
        if ((rc = rocket_ew_add_fp16(fd, (const _Float16 *)cube.ptr, bias_cube,
                                     (_Float16 *)cube.ptr, (int)cn)) != 0) goto fb_free;
    }
    /* act(.) on the cube, element-wise in place (the accurate 2-pass for GELU) */
    if ((rc = rocket_activation_fp16(fd, kind, (const _Float16 *)cube.ptr,
                                     (_Float16 *)cube.ptr, (int)cn)) != 0) goto fb_free;

    /* hand the activated cube to the device for fc2 */
    rocket_bo_prep(fd, &cube, 1, 0);
    rocket_bo_fini(fd, &cube);

    /* out = act·W2^T — fc2 reads the cube directly as its input BO */
    mm_pack_weights(fd, &p2, &b2m, W2);
    {
        rocket_bo saved = b2m.in_all;
        b2m.in_all = cube;
        rc = mm_compute_kacc(fd, &p2, &b2m, out, 0.0);
        if (rc == ROCKET_E_TILING) rc = mm_compute(fd, &p2, &b2m, out, 0.0);
        b2m.in_all = saved;
    }

fb_free:
    free(bias_cube);
    rocket_bo_free(fd, &cube);
    if (h2) mm_bos_free(fd, &b2m);
    if (h1) mm_bos_free(fd, &b1m);
    return rc;

fb:
    free(bias_cube);
    rocket_bo_free(fd, &cube);
    if (h2) mm_bos_free(fd, &b2m);
    if (h1) mm_bos_free(fd, &b1m);
    return mlp_host_handoff(fd, M, Din, Dff, Dout, x, W1, b1, W2, kind, out);
}
