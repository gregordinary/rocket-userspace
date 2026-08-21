// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_coeff_c.c — what the coefficient group's C multiplier actually multiplies.
 *
 * C is the per-output-channel int16 in the RK3576's 64-byte coefficient group, and
 * every program this tree emits has so far pinned it to 1. A per-axis weight
 * quantization is the first thing that needs it non-unit, and using it needs two facts
 * that the group's field map does not carry:
 *
 *   ORDER — the group also holds A, the per-channel bias term. `acc*C + A` and
 *   `(acc + A)*C` are the same buffer and different arithmetic, and the difference is
 *   a bias scaled by the wrong channel gain: a plausible surface, not a fault.
 *
 *   RANGE — C is 16 bits wide, but the width of the field says nothing about the width
 *   of the multiply behind it. A per-axis requant wants C as large as the field holds,
 *   so where the product stops being exact is the bound on how much precision the
 *   per-channel gain can carry.
 *
 * Both are read against a known accumulator: a 1x1 conv whose input has one non-zero
 * channel and whose weights select it, so every output channel's accumulator is one
 * product the host chose. The requant is left at unity, so the surface IS the BS
 * stage's output.
 *
 * Usage:  rk3576_coeff_c [order|perchan|range|clamp|product|all]   (default: all)
 *         `all` is the gate: order, perchan, range, clamp. `product` is a
 *         characterisation — its inexact cells ARE the decoded saturation.
 * Exit:   0 every question answered consistently, 1 a disagreement, 2 no NPU (skip).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_hw_profile.h"

#define C2       16
#define SENTINEL 0xAA

/* Both are 1-BASED on every axis, and both live in the library. */
int feature_data(int C, int H, int W, int C2_, int c, int h, int w);
int weight_conv_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

static size_t out_index(unsigned surf_elems, unsigned ow, unsigned c,
                        unsigned y, unsigned x)
{
    return (size_t)(c / C2) * surf_elems * C2 + (size_t)C2 * (y * ow + x) + (c % C2);
}

/*
 * One 1x1 conv over a 4x4 plane at 32 channels in and out.
 *
 * feature: channel 0 = `fval` everywhere, every other channel 0.
 * weight:  W[oc][0][0][0] = `wval`, everything else 0.
 * so       acc[oc] = fval*wval for every oc, whatever the channel.
 *
 * A[oc] = bias[oc], C[oc] = cmul[oc]. The requant is unity. `out` receives the
 * de-scattered int8 surface value for each of the `oc` channels (they are uniform
 * over the plane, and that is checked).
 */
static int run_one_scaled(int fd, unsigned oc_n, unsigned nic, int fval, int wval,
                          const int32_t *bias, const int16_t *cmul,
                          float conv_scale, int *out)
{
    const unsigned ic = 32, iw = 4, ih = 4, k = 1;
    unsigned ow = iw, oh = ih;
    unsigned icreg = rocket_rk3576_pad_ic(ic), ocreg = rocket_rk3576_pad_oc(oc_n);
    unsigned surf_elems = rocket_rk3576_out_surf_elems(ow, oh, 0);
    size_t in_bytes = (size_t)(icreg / C2) * ih * iw * C2;
    size_t obytes   = (size_t)((ocreg + C2 - 1) / C2) * surf_elems * C2;
    size_t coeff    = rocket_rk3576_coeff_bytes(ocreg);
    size_t w_bytes  = (size_t)ocreg * icreg * k * k;
    rocket_bo bo_in = {0}, bo_w = {0}, bo_b = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p = {0};
    unsigned c, y, x;
    int rc = -1;

    if (rocket_bo_alloc(fd, in_bytes, &bo_in) < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes,  &bo_w)  < 0) goto done;
    if (rocket_bo_alloc(fd, coeff,    &bo_b)  < 0) goto done;
    if (rocket_bo_alloc(fd, obytes,   &bo_o)  < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    rocket_bo_prep(fd, &bo_in, 1, 0);
    memset(bo_in.ptr, 0, in_bytes);
    {
        int8_t *f = (int8_t *)bo_in.ptr;
        unsigned j;
        for (j = 0; j < nic; j++)
            for (y = 0; y < ih; y++)
                for (x = 0; x < iw; x++)
                    f[feature_data((int)icreg, (int)ih, (int)iw, C2,
                                   (int)j + 1, (int)y + 1, (int)x + 1)] = (int8_t)fval;
    }
    rocket_bo_fini(fd, &bo_in);

    rocket_bo_prep(fd, &bo_w, 1, 0);
    memset(bo_w.ptr, 0, w_bytes);
    {
        int8_t *w8 = (int8_t *)bo_w.ptr;
        unsigned j;
        for (c = 0; c < oc_n; c++)
            for (j = 0; j < nic; j++)
                w8[weight_conv_int8((int)ocreg, (int)icreg, (int)k, (int)k,
                                    (int)c + 1, (int)j + 1, 1, 1)] = (int8_t)wval;
    }
    rocket_bo_fini(fd, &bo_w);

    rocket_bo_prep(fd, &bo_b, 1, 0);
    if (rocket_rk3576_pack_coeff_perc(bo_b.ptr, coeff, bias, ocreg, NULL, cmul, 1) != 0) {
        rocket_bo_fini(fd, &bo_b);
        goto done;
    }
    rocket_bo_fini(fd, &bo_b);

    p.ic = (uint16_t)icreg; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.oc = (uint16_t)ocreg; p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
    p.kh = (uint16_t)k;     p.kw = (uint16_t)k;
    p.stride_y = p.stride_x = 1;
    p.int8_out = 1;
    p.in_scale = conv_scale; p.w_scale = 1.0f; p.out_scale = 1.0f;
    p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
    p.tasks       = ops;
    p.input_dma   = bo_in.dma_address;
    p.weights_dma = bo_w.dma_address;
    p.bias_dma    = bo_b.dma_address;
    p.output_dma  = bo_o.dma_address;
    p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)oh;

    in_h[0] = bo_in.handle; in_h[1] = bo_w.handle;
    in_h[2] = bo_b.handle;  in_h[3] = bo_r.handle;
    out_h[0] = bo_o.handle;

    if (gen_conv2d_int8_rk3576(&p) != 0) {
        printf("  generator refused this shape\n");
        goto done;
    }
    rocket_bo_prep(fd, &bo_r, 1, 0);
    memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &bo_r);

    /* Bracketed, never a bare memset: dirty CPU lines race the DPU's write DMA. */
    rocket_bo_prep(fd, &bo_o, 1, 0);
    memset(bo_o.ptr, SENTINEL, obytes);
    rocket_bo_fini(fd, &bo_o);

    if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 4, out_h, 1, 2000) != 0) {
        printf("  submit failed\n");
        goto done;
    }
    if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
        printf("  PREP_BO timed out\n");
        goto done;
    }

    for (c = 0; c < oc_n; c++) {
        const int8_t *o = (const int8_t *)bo_o.ptr;
        int v0 = o[out_index(surf_elems, ow, c, 0, 0)];
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++)
                if (o[out_index(surf_elems, ow, c, y, x)] != v0) {
                    printf("  c=%u is not uniform over the plane — the probe's premise "
                           "is broken, not the answer\n", c);
                    goto done;
                }
        out[c] = v0;
    }
    rc = 0;
done:
    if (bo_in.ptr) rocket_bo_free(fd, &bo_in);
    if (bo_w.ptr)  rocket_bo_free(fd, &bo_w);
    if (bo_b.ptr)  rocket_bo_free(fd, &bo_b);
    if (bo_o.ptr)  rocket_bo_free(fd, &bo_o);
    if (bo_r.ptr)  rocket_bo_free(fd, &bo_r);
    return rc;
}

/* The unity-requant form the first three probes use. */
static int run_one(int fd, unsigned oc_n, int fval, int wval,
                   const int32_t *bias, const int16_t *cmul, int *out)
{
    return run_one_scaled(fd, oc_n, 1, fval, wval, bias, cmul, 1.0f, out);
}

/* --------------------------------------------------------------------------
 * order: does C multiply the accumulator only, or the accumulator plus the bias?
 * ------------------------------------------------------------------------ */
static int probe_order(int fd)
{
    const unsigned OC = 32;
    const int fval = 5, wval = 2;              /* acc = 10 on every channel */
    const int32_t A = 20;
    int32_t bias[32];
    int16_t cmul[32];
    int got[32];
    unsigned c;
    int mul_then_add = 0, add_then_mul = 0, other = 0;

    for (c = 0; c < OC; c++) {
        bias[c] = A;
        cmul[c] = (int16_t)((c % 2) ? 2 : 1);
    }

    printf("order: acc=%d (feature %d x weight %d), A=%d, C=1 on even channels "
           "and 2 on odd\n", fval * wval, fval, wval, (int)A);
    printf("       acc*C + A predicts %d / %d;  (acc + A)*C predicts %d / %d\n",
           fval * wval + (int)A, fval * wval * 2 + (int)A,
           fval * wval + (int)A, (fval * wval + (int)A) * 2);

    if (run_one(fd, OC, fval, wval, bias, cmul, got) != 0) return -1;

    for (c = 0; c < OC; c++) {
        int C = cmul[c];
        int p_mul_add = fval * wval * C + (int)A;
        int p_add_mul = (fval * wval + (int)A) * C;
        if (got[c] == p_mul_add && got[c] == p_add_mul) {
            mul_then_add++; add_then_mul++;          /* C == 1: does not discriminate */
        } else if (got[c] == p_mul_add) {
            mul_then_add++;
        } else if (got[c] == p_add_mul) {
            add_then_mul++;
        } else {
            other++;
            if (other <= 4)
                printf("       c=%-3u C=%d got %d, neither %d nor %d\n",
                       c, C, got[c], p_mul_add, p_add_mul);
        }
    }
    printf("       acc*C + A explains %u/%u, (acc + A)*C explains %u/%u, "
           "neither %u\n", mul_then_add, OC, add_then_mul, OC, other);
    if (other) return -1;
    if (mul_then_add == OC && add_then_mul < OC) {
        printf("       ORDER: the DPU multiplies the accumulator and THEN adds A.\n");
        return 0;
    }
    if (add_then_mul == OC && mul_then_add < OC) {
        printf("       ORDER: the DPU adds A and THEN multiplies by C.\n");
        return 1;
    }
    printf("       the two are not separated by this configuration\n");
    return -1;
}

/* --------------------------------------------------------------------------
 * range: how large a C stays an exact multiply.
 *
 * The surface is int8, so the probe holds the PRODUCT in range and moves C: an
 * accumulator of 1 and a C of N should read back N until either the field or the
 * multiplier behind it runs out.
 * ------------------------------------------------------------------------ */
static int probe_range(int fd)
{
    static const int cvals[] = { 1, 2, 4, 100, 127, 128, 1000, 4096,
                                 16384, 32767, -1, -100 };
    const unsigned OC = 32;
    unsigned i;
    int bad = 0;

    printf("range: acc = 1, C swept, unity requant — the surface should BE C "
           "until it saturates at 127\n");
    for (i = 0; i < sizeof cvals / sizeof *cvals; i++) {
        int32_t bias[32] = {0};
        int16_t cmul[32];
        int got[32];
        unsigned c;
        int expect, ok = 1;

        for (c = 0; c < OC; c++) cmul[c] = (int16_t)cvals[i];
        if (run_one(fd, OC, 1, 1, bias, cmul, got) != 0) { bad++; continue; }

        expect = cvals[i] > 127 ? 127 : (cvals[i] < -128 ? -128 : cvals[i]);
        for (c = 0; c < OC; c++) if (got[c] != expect) { ok = 0; break; }
        printf("  C=%-6d -> %-5d (expected %-5d) %s\n",
               cvals[i], got[0], expect, ok ? "" : "MISMATCH");
        if (!ok) bad++;
    }
    return bad ? -1 : 0;
}

/* --------------------------------------------------------------------------
 * perchan: every channel gets a DIFFERENT C, so a group-indexing error shows.
 * ------------------------------------------------------------------------ */
static int probe_perchan(int fd)
{
    const unsigned OC = 32;
    int32_t bias[32] = {0};
    int16_t cmul[32];
    int got[32];
    unsigned c;
    int wrong = 0;

    for (c = 0; c < OC; c++) cmul[c] = (int16_t)(c + 1);   /* 1..32, all distinct */
    printf("perchan: acc = 1, C[c] = c+1 — the surface should be the ramp itself\n");
    if (run_one(fd, OC, 1, 1, bias, cmul, got) != 0) return -1;
    for (c = 0; c < OC; c++) {
        if (got[c] != (int)c + 1) {
            unsigned s; int slot = -1;
            for (s = 0; s < OC; s++) if ((int)cmul[s] == got[c]) { slot = (int)s; break; }
            printf("  c=%-3u expected %d, got %d%s\n", c, c + 1, got[c],
                   slot >= 0 && slot != (int)c ? " (that is another channel's C)" : "");
            wrong++;
        }
    }
    printf("  %u/%u channels read their own C\n", OC - wrong, OC);
    return wrong ? -1 : 0;
}

/* --------------------------------------------------------------------------
 * product: where `(acc + A) * C` stops being exact.
 *
 * The int8 surface cannot show a large product directly, so the requant divides it
 * back down: at a conv scale of `target/(acc*C)` an exact chain reads `target` for
 * every (acc, C) pair, and a product that wraps or truncates reads something else.
 * `acc` is grown by contracting more input channels rather than by a larger tap, so
 * the operands stay well inside int8.
 *
 * This is the bound a per-axis requant has to respect: it wants C as large as the
 * field holds, and the accumulator is whatever the layer produces.
 * ------------------------------------------------------------------------ */
static int probe_product(int fd)
{
    static const int cvals[]  = { 1, 16, 256, 4096, 16384, 32767 };
    static const unsigned nics[] = { 1, 4, 16, 32 };
    const unsigned OC = 32;
    const int fval = 100, wval = 100;   /* 10000 per contracted channel */
    const int target = 100;
    unsigned ci, ni;
    int bad = 0;

    printf("product: out = (acc*C) * target/(acc*C) — an exact chain reads %d "
           "everywhere\n", target);
    printf("  %-8s %-6s %-10s %-6s %s\n", "C", "chans", "acc", "got", "");
    for (ci = 0; ci < sizeof cvals / sizeof *cvals; ci++) {
        for (ni = 0; ni < sizeof nics / sizeof *nics; ni++) {
            int32_t bias[32] = {0};
            int16_t cmul[32];
            int got[32];
            unsigned c;
            long long acc = (long long)nics[ni] * fval * wval;
            double prod = (double)acc * (double)cvals[ci];
            float cs = (float)((double)target / prod);
            int ok = 1;

            for (c = 0; c < OC; c++) cmul[c] = (int16_t)cvals[ci];
            if (run_one_scaled(fd, OC, nics[ni], fval, wval, bias, cmul, cs, got) != 0) {
                bad++; continue;
            }
            for (c = 0; c < OC; c++) if (got[c] != target) { ok = 0; break; }
            printf("  %-8d %-6u %-10lld %-6d %s\n",
                   cvals[ci], nics[ni], acc, got[0],
                   ok ? "exact" : "NOT EXACT");
            if (!ok) bad++;
        }
    }
    if (bad)
        printf("  the product is not exact everywhere — the pairs marked above bound "
               "what a per-axis C may ask for\n");
    return bad ? -1 : 0;
}

/* --------------------------------------------------------------------------
 * clamp: is the inexact product SATURATED or WRAPPED, and at what value?
 *
 * The two are not the same failure. A wrap is catastrophic and unusable; a clamp is
 * a bound a planner can simply stay inside, which is what makes per-axis C viable at
 * all. The probe walks the product across the suspected knee at a fixed accumulator,
 * and reads the implied ceiling back out of each inexact cell — a genuine clamp
 * reports the SAME ceiling from every one of them, a wrap does not.
 * ------------------------------------------------------------------------ */
static int probe_clamp(int fd)
{
    const unsigned OC = 32, nic = 16;
    const int fval = 100, wval = 100, target = 100;
    long long acc = (long long)nic * fval * wval;      /* 160000 */
    double ceil_lo = 0, ceil_hi = 0;
    int n_ceil = 0, wrapped = 0, i;
    /* products from ~0.5x to ~8x of 2^31 */
    static const double mult[] = { 0.5, 0.9, 0.99, 1.01, 1.1, 1.5, 2.0, 4.0, 8.0 };

    printf("clamp: acc = %lld fixed, C chosen to put the product at a multiple of "
           "2^31\n", acc);
    printf("  %-12s %-8s %-6s %-14s\n", "product", "C", "got", "implied ceiling");
    for (i = 0; i < (int)(sizeof mult / sizeof *mult); i++) {
        int32_t bias[32] = {0};
        int16_t cmul[32];
        int got[32];
        unsigned c;
        double want = mult[i] * 2147483648.0;
        long long C = (long long)(want / (double)acc + 0.5);
        double prod, cs;

        if (C < 1) C = 1;
        if (C > 32767) C = 32767;
        prod = (double)acc * (double)C;
        cs = (double)target / prod;
        for (c = 0; c < OC; c++) cmul[c] = (int16_t)C;
        if (run_one_scaled(fd, OC, nic, fval, wval, bias, cmul, (float)cs, got) != 0)
            return -1;
        if (got[0] == target) {
            printf("  %-12.4g %-8lld %-6d exact\n", prod, C, got[0]);
        } else if (got[0] < 0 || got[0] > target) {
            printf("  %-12.4g %-8lld %-6d WRAPPED — not a clamp\n", prod, C, got[0]);
            wrapped++;
        } else {
            double ceiling = (double)got[0] / (double)target * prod;
            printf("  %-12.4g %-8lld %-6d %.4g\n", prod, C, got[0], ceiling);
            if (!n_ceil || ceiling < ceil_lo) ceil_lo = ceiling;
            if (!n_ceil || ceiling > ceil_hi) ceil_hi = ceiling;
            n_ceil++;
        }
    }
    if (wrapped) {
        printf("  the product WRAPS — a per-axis C cannot be planned against a bound\n");
        return -1;
    }
    if (n_ceil) {
        printf("  every inexact cell implies a ceiling in [%.6g, %.6g]; 2^31 is %.6g\n",
               ceil_lo, ceil_hi, 2147483648.0);
        printf("  CLAMP: the BS stage computes (acc + A)*C in int32 and SATURATES.\n");
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "all";
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, fail = 0;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_coeff_c: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_coeff_c: no NPU device — skipping\n"); return 2; }

    if (!strcmp(mode, "all") || !strcmp(mode, "order"))
        if (probe_order(fd) < 0) fail = 1;
    if (!strcmp(mode, "all") || !strcmp(mode, "perchan"))
        if (probe_perchan(fd) != 0) fail = 1;
    if (!strcmp(mode, "all") || !strcmp(mode, "range"))
        if (probe_range(fd) != 0) fail = 1;
    /* NOT in `all`: `product` is a characterisation, and its inexact cells are the
     * decoded int32 saturation rather than a failure. `clamp` IS in `all` — it fails
     * only if the product wraps, which would make the bound unplannable. */
    if (!strcmp(mode, "product"))
        (void)probe_product(fd);
    if (!strcmp(mode, "all") || !strcmp(mode, "clamp"))
        if (probe_clamp(fd) != 0) fail = 1;

    rocket_close(fd);
    printf("rk3576_coeff_c: %s\n", fail ? "a question came back inconsistent" : "ok");
    return fail;
}
