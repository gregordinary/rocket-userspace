// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_residual_add.c — the int8 residual add, on the convolution datapath.
 *
 * A skip connection is the one primitive between this part's op library and a second
 * network: MobileNetV2 puts a skip on ten of its seventeen inverted residuals and every
 * ResNet block ends in one. The DPU's elementwise stage looked like where it would land
 * and it is not — that stage takes exactly ONE operand, and `tests/rk3576_add_probe.c`
 * closes the register interface exhaustively (every register the vendor's program leaves
 * at zero, at every output gain; every register it never writes; the operand buffer's
 * whole neighbourhood; destination accumulation).
 *
 * SO THE ADD IS LOWERED ONTO A CONVOLUTION, which this part already computes bit-exactly.
 * Concatenate the two operands along the channel axis and convolve with a 1x1 kernel of
 * two diagonal blocks:
 *
 *     out[o] = w1 * (a[o] - a_zp) + w2 * (b[o] - b_zp)   requantized on chip
 *
 * with `W[o][o] = w1`, `W[o][C+o] = w2` and zero everywhere else. Three properties fall
 * out of that, and two of them are better than what the vendor's own elementwise program
 * can do:
 *
 *   - THE TWO OPERANDS MAY CARRY DIFFERENT SCALES. The ratio rides in the weights —
 *     `w2/w1` — so the pair only has to be representable as two int8s, resolved to about
 *     one part in 127. The vendor's elementwise program has ONE operand converter and its
 *     compiler quantizes both operands to a COMMON scale to fit it (measured: two graph
 *     inputs calibrated 64x apart still compile to one converter, and the OUT_CVT gain
 *     comes out as the ratio of a single shared input scale to the output's).
 *   - THE TWO ZERO POINTS RIDE IN THE BIAS. The datapath has one input zero point, but
 *     `w2 * (a_zp - b_zp)` is a per-output-channel constant, which is exactly what the
 *     bias is, so the second operand's zero point is expressed EXACTLY rather than
 *     approximated.
 *   - IT FUSES. The concatenation is along channels, and a feature cube's channel-group
 *     stride is the plane while a direct conv's output surface stride is `ow*oh` — so two
 *     convolutions writing adjacent halves of one buffer ARE the concatenated operand,
 *     byte for byte, with no host work. A block's last convolution therefore absorbs its
 *     own skip by taking C more input channels and an identity block at the CENTRE TAP of
 *     its kernel, and the add costs no program at all against the vendor's one. `fused`
 *     measures it, and it is the hardware's own idiom: the vendor compiler folds
 *     `Add(Conv(x), x)` into exactly this shape.
 *
 * WHAT IT COSTS is the weight slice rule, `ic*kh*kw <= 4608`. A 1x1 project convolution
 * fusing its skip is nowhere near it; a 3x3 reaches it EXACTLY at C = 256 and is refused
 * at C = 512, so ResNet-18's widest stage keeps a standalone add — which the same rule
 * caps at C = 2304, on a `2C*C`-byte weight cube (512 KiB there, a resident-weights cost
 * paid once, not per inference).
 *
 * The gate scores the part against a CPU model of its OWN arithmetic (the conv gate's
 * `requant_apply_zp`), which is what "bit-exact" means here, and separately reports the
 * distance from an exact float residual add, which is what the lowering costs in
 * accuracy.
 *
 * Usage:  rk3576_residual_add
 * Exit:   0, 1 on a failure, 2 no NPU (skip).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "requant_model.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

typedef struct { const char *name; int c, h, w; } shape;

/* The residual shapes two real networks actually have. */
static const shape SHAPES[] = {
    { "mbv2 c24 56x56",   24, 56, 56 },
    { "mbv2 c32 28x28",   32, 28, 28 },
    { "mbv2 c64 14x14",   64, 14, 14 },
    { "mbv2 c96 14x14",   96, 14, 14 },
    { "mbv2 c160 7x7",   160,  7,  7 },
    { "res18 c64 56x56",  64, 56, 56 },
    { "res18 c128 28x28",128, 28, 28 },
    { "res18 c256 14x14",256, 14, 14 },
    { "res18 c512 7x7",  512,  7,  7 },
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof *SHAPES))

static void fill_ramp(int8_t *v, size_t n, int seed)
{
    size_t i;
    uint32_t s = 0x9E3779B9u ^ (uint32_t)seed;
    for (i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        v[i] = (int8_t)((int)((s >> 16) % 251u) - 125);
    }
}

/* One residual add. Returns 0 exact, 1 wrong, 2 refused. */
static int one(int fd, const shape *s, float sa, float sb, int za, int zb,
               int *out_worst_counts, double *out_ms)
{
    int C = s->c, H = s->h, W = s->w;
    size_t plane = (size_t)H * W, n = (size_t)C * plane;
    size_t nin = 2 * n, nw = (size_t)C * 2 * C;
    int8_t *a = malloc(n), *b = malloc(n), *in = malloc(nin), *wt = malloc(nw);
    int8_t *got = malloc(n);
    int32_t *bias = malloc((size_t)C * sizeof *bias);
    rocket_conv2d_desc d;
    float so, w_scale;
    int w1, w2, zo = 3, rc, i, exact = 0, maxdiff = 0, worst = 0;
    double t0;

    if (!a || !b || !in || !wt || !got || !bias) { rc = 1; goto done; }
    fill_ramp(a, n, C + 1);
    fill_ramp(b, n, C + 101);

    /* The concatenation, along channels: operand A then operand B. In CUBE layout this
     * copy does not exist — a feature cube's channel-group stride is the plane, so two
     * producers writing adjacent halves of one buffer already ARE this tensor. It is
     * here because the library's public entry takes row-major CHW. */
    memcpy(in, a, n);
    memcpy(in + n, b, n);

    /* The lowering's own arithmetic is the LIBRARY'S — the diagonal blocks, the ratio
     * quantization and the zero-point bias — so this gate drives that entry rather than
     * carrying a second copy of it to keep in agreement. */
    {
        int wp[2];
        rc = rocket_residual_add_weights_rk3576((unsigned)C, sa, sb, za, zb,
                                                wt, bias, &w_scale, wp);
        if (rc != ROCKET_OK) { rc = rc == ROCKET_E_UNSUPPORTED ? 2 : 1; goto done; }
        w1 = wp[0]; w2 = wp[1];
    }
    /* An output scale that keeps the sum inside int8 without clipping it away. */
    so = (sa * 127.0f + sb * 127.0f) / 127.0f;

    memset(&d, 0, sizeof d);
    d.ic = 2 * C; d.ih = H; d.iw = W; d.oc = C;
    d.kh = d.kw = 1; d.stride_y = d.stride_x = 1;
    d.dil_y = d.dil_x = 1;
    d.direct_datapath = 1;

    t0 = now_ms();
    rc = rocket_conv2d_int8_rk3576(fd, &d, in, wt, bias, sa, w_scale, so,
                                   za, 0, zo, got);
    *out_ms = now_ms() - t0;
    if (rc == ROCKET_E_UNSUPPORTED) { rc = 2; goto done; }
    if (rc != ROCKET_OK) { printf("    entry returned %d\n", rc); rc = 1; goto done; }

    /* The part's OWN arithmetic, which is what bit-exact means here. */
    {
        unsigned scale, shift_reg;
        requant_params(sa * w_scale / so, &scale, &shift_reg);
        for (i = 0; i < (int)n; i++) {
            int64_t acc = (int64_t)w1 * (a[i] - za) + (int64_t)w2 * (b[i] - za) +
                          bias[i / (int)plane];
            int want = requant_apply_zp(acc, scale, shift_reg, zo);
            int diff = got[i] > want ? got[i] - want : want - got[i];
            if (diff > maxdiff) maxdiff = diff;
            if (got[i] == want) exact++;
        }
    }
    /* And what the lowering costs against an exact float residual add. */
    for (i = 0; i < (int)n; i++) {
        double want = (sa * (a[i] - za) + sb * (b[i] - zb)) / so + zo;
        int q = (int)lround(want);
        int diff;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        diff = got[i] > q ? got[i] - q : q - got[i];
        if (diff > worst) worst = diff;
    }
    *out_worst_counts = worst;
    printf("  %-18s sa/sb %6.3f  w %3d/%-3d  %s (%d/%zu exact, max %d)"
           "  float dist %d  %.2f ms\n",
           s->name, (double)sb / sa, w1, w2,
           exact == (int)n ? "bit-exact" : "WRONG", exact, n, maxdiff, worst, *out_ms);
    rc = exact == (int)n ? 0 : 1;
done:
    free(a); free(b); free(in); free(wt); free(got); free(bias);
    return rc;
}

/* THE FUSED FORM: a convolution that absorbs its own skip.
 *
 * The standalone add above is a program. This is the claim that it need not be one — a
 * block's last convolution takes `C` more input channels, its kernel takes an identity
 * block at the CENTRE TAP, and the skip is then part of a convolution the network was
 * already paying for. The vendor's own compiler does exactly this when it can see the
 * skip: `Add(Conv(x), x)` compiles to a plain convolution with the identity folded into
 * the centre tap of its diagonal, which is what made that graph useless as a capture of
 * an add and is also the confirmation that the lowering is the hardware's own idiom.
 *
 * Two real shapes, and the bound between them. MobileNetV2's project convolution is 1x1
 * over `6C` channels, so fusing costs `C` more of them and the slice rule `ic*kh*kw <=
 * 4608` is nowhere near. ResNet-18's second convolution is 3x3 over `C`, so fusing makes
 * it `2C*9` — which reaches the rule exactly at C=256 and must be REFUSED at C=512. A
 * gate that only showed the cases that fit would be hiding where the fusion stops.
 *
 * Returns 0 exact, 1 wrong, 2 refused. */
static int fused(int fd, const char *name, int C, int E, int H, int W, int k,
                 int expect_refuse)
{
    int ic = E + C, o, i, kk, exact = 0, maxdiff = 0, rc;
    size_t plane = (size_t)H * W;
    size_t nin = (size_t)ic * plane, nout = (size_t)C * plane;
    size_t nw = (size_t)C * ic * k * k;
    int8_t *in = malloc(nin), *wt = malloc(nw), *got = malloc(nout);
    int32_t *bias = calloc((size_t)C, sizeof *bias);
    rocket_conv2d_desc d;
    float in_scale = 0.019f, w_scale = 1.0f / 96.0f, out_scale = 0.041f;
    int in_zp = -5, out_zp = 2, centre = (k * k) / 2;

    if (!in || !wt || !got || !bias) { rc = 1; goto done; }
    fill_ramp(in, nin, C + 7);
    fill_ramp(wt, nw, C + 13);
    /* The skip block: identity at the centre tap, zero at every other tap and every
     * other channel pair. The main half keeps the ramp, so this is a real convolution
     * with a skip in it rather than two identities. */
    for (o = 0; o < C; o++)
        for (i = 0; i < C; i++)
            for (kk = 0; kk < k * k; kk++)
                wt[((size_t)o * ic + E + i) * k * k + kk] =
                    (i == o && kk == centre) ? (int8_t)96 : 0;

    memset(&d, 0, sizeof d);
    d.ic = ic; d.ih = H; d.iw = W; d.oc = C;
    d.kh = d.kw = k; d.stride_y = d.stride_x = 1;
    d.pad_top = d.pad_left = k / 2;
    d.dil_y = d.dil_x = 1;
    d.direct_datapath = 1;

    rc = rocket_conv2d_int8_rk3576(fd, &d, in, wt, bias, in_scale, w_scale, out_scale,
                                   in_zp, 0, out_zp, got);
    if (rc == ROCKET_E_UNSUPPORTED) {
        printf("  %-26s ic=%-5d %s\n", name, ic,
               expect_refuse ? "refused, as the slice rule requires" : "REFUSED, unexpectedly");
        rc = expect_refuse ? 0 : 1;
        goto done;
    }
    if (expect_refuse) {
        printf("  %-26s ic=%-5d COMPUTED past the slice rule\n", name, ic);
        rc = 1; goto done;
    }
    if (rc != ROCKET_OK) { printf("    entry returned %d\n", rc); rc = 1; goto done; }

    {
        unsigned scale; unsigned shift_reg;
        int y, x, c;
        requant_params(in_scale * w_scale / out_scale, &scale, &shift_reg);
        for (c = 0; c < C; c++)
            for (y = 0; y < H; y++)
                for (x = 0; x < W; x++) {
                    int64_t acc = 0;
                    int ky, kx, want, diff, g;
                    for (ky = 0; ky < k; ky++)
                        for (kx = 0; kx < k; kx++) {
                            int iy = y + ky - k / 2, ix = x + kx - k / 2;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                            for (g = 0; g < ic; g++)
                                acc += (int64_t)(in[((size_t)g * H + iy) * W + ix] - in_zp) *
                                       wt[((size_t)c * ic + g) * k * k + ky * k + kx];
                        }
                    want = requant_apply_zp(acc, scale, shift_reg, out_zp);
                    diff = got[((size_t)c * H + y) * W + x] - want;
                    if (diff < 0) diff = -diff;
                    if (diff > maxdiff) maxdiff = diff;
                    if (!diff) exact++;
                }
    }
    printf("  %-26s ic=%-5d %s (%d/%zu exact, max %d)\n", name, ic,
           exact == (int)nout ? "bit-exact" : "WRONG", exact, nout, maxdiff);
    rc = exact == (int)nout ? 0 : 1;
done:
    free(in); free(wt); free(got); free(bias);
    return rc;
}

/* WHETHER A REAL BLOCK CAN FUSE AT ALL, which is a question about the QUANTIZATION rather
 * than about the datapath. `fused` above shows the part computes the shape; this shows
 * that on MobileNetV2 the identity weight it needs does not fit the field, and that where
 * it does fit the entry's arithmetic is right on silicon.
 *
 * The ten triples are the model's own: each project convolution's input and weight scales
 * with the scale of the tensor its block's skip comes from. */
struct fuse_case { const char *name; float in_s, w_s, skip_s; };
static const struct fuse_case MBV2[] = {
    { "add  9 c24",  0.023529f, 0.027410f, 0.275800f },
    { "add 16 c32",  0.023529f, 0.019060f, 0.218400f },
    { "add 20 c32",  0.023529f, 0.018290f, 0.259700f },
    { "add 27 c64",  0.023529f, 0.016780f, 0.185400f },
    { "add 31 c64",  0.023529f, 0.012900f, 0.189100f },
    { "add 35 c64",  0.023529f, 0.019560f, 0.199700f },
    { "add 42 c96",  0.023529f, 0.008382f, 0.170600f },
    { "add 46 c96",  0.023529f, 0.023980f, 0.176200f },
    { "add 53 c160", 0.023529f, 0.007899f, 0.132400f },
    { "add 57 c160", 0.023529f, 0.036970f, 0.150700f },
};
#define N_MBV2 ((int)(sizeof MBV2 / sizeof *MBV2))

/* One convolution whose contraction is short enough that the fusion IS expressible, run
 * end to end: the entry picks the identity weight and the bias correction, and the part's
 * output is scored against an exact float `main + skip`. Returns 0 exact enough, 1 wrong,
 * 2 the entry refused. */
static int fuse_runs(int fd)
{
    enum { E = 32, C = 32, H = 14, W = 14, ic = E + C };
    size_t plane = (size_t)H * W, nin = (size_t)ic * plane, nout = (size_t)C * plane;
    size_t nw = (size_t)C * ic;
    int8_t *in = malloc(nin), *wt = calloc(nw, 1), *got = malloc(nout);
    int32_t *bias = calloc((size_t)C, sizeof *bias);
    rocket_conv2d_desc d;
    /* Scales that put the identity weight inside int8: the accumulator unit is 2e-4 and
     * the skip's scale is a hundred of them. */
    float in_scale = 0.02f, w_scale = 0.01f, skip_scale = 0.02f, out_scale = 0.06f;
    int in_zp = -4, skip_zp = 7, out_zp = 3;
    int w_skip = 0, o, i, rc, worst = 0;
    int32_t bias_delta = 0;
    float rel = 0.0f;

    if (!in || !wt || !got || !bias) { rc = 1; goto done; }
    rc = rocket_residual_fuse_weight_rk3576(in_scale, w_scale, skip_scale, in_zp, skip_zp,
                                            &w_skip, &bias_delta, &rel);
    if (rc != ROCKET_OK) {
        printf("  an expressible shape was refused (rc %d)\n", rc);
        rc = 1; goto done;
    }
    fill_ramp(in, nin, 31);
    fill_ramp(wt, nw, 37);
    for (o = 0; o < C; o++) {
        for (i = 0; i < C; i++) wt[(size_t)o * ic + E + i] = (int8_t)(i == o ? w_skip : 0);
        bias[o] = bias_delta;
    }

    memset(&d, 0, sizeof d);
    d.ic = ic; d.ih = H; d.iw = W; d.oc = C;
    d.kh = d.kw = 1; d.stride_y = d.stride_x = 1;
    d.dil_y = d.dil_x = 1;
    d.direct_datapath = 1;
    rc = rocket_conv2d_int8_rk3576(fd, &d, in, wt, bias, in_scale, w_scale, out_scale,
                                   in_zp, 0, out_zp, got);
    if (rc != ROCKET_OK) { printf("  the fused entry returned %d\n", rc); rc = 1; goto done; }

    /* An exact float `main convolution + skip`, which is what the fusion claims to be. */
    for (o = 0; o < C; o++)
        for (i = 0; i < (int)plane; i++) {
            double acc = 0.0;
            int g, q, diff;
            for (g = 0; g < E; g++)
                acc += (double)in_scale * (in[(size_t)g * plane + i] - in_zp) *
                       (double)w_scale * wt[(size_t)o * ic + g];
            acc += (double)skip_scale *
                   (in[(size_t)(E + o) * plane + i] - skip_zp);
            q = (int)lround(acc / out_scale) + out_zp;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            diff = got[(size_t)o * plane + i] - q;
            if (diff < 0) diff = -diff;
            if (diff > worst) worst = diff;
        }
    printf("  an expressible block: w_skip %d (%.3f%% from the exact ratio), bias %+d — "
           "%d count(s) from an exact float main+skip\n",
           w_skip, (double)rel * 100.0, bias_delta, worst);
    rc = worst <= 2 ? 0 : 1;
done:
    free(in); free(wt); free(got); free(bias);
    return rc;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, i, fails = 0, ran = 0, refused = 0;
    (void)argc; (void)argv;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_residual_add: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_residual_add: no NPU device — skipping\n"); return 2; }

    printf("gate: the residual add lowered onto a 1x1 convolution over 2C channels\n");
    for (i = 0; i < N_SHAPES; i++) {
        /* Scales deliberately unequal and zero points deliberately different — the two
         * things the elementwise stage cannot express and this lowering can. */
        int worst = 0; double ms = 0;
        int rc = one(fd, &SHAPES[i], 0.021f, 0.013f, -7, 11, &worst, &ms);
        if (rc == 2) { printf("  %-18s refused\n", SHAPES[i].name); refused++; continue; }
        fails += rc;
        ran++;
    }
    /* Equal scales and equal zero points: the case the vendor's own program is limited
     * to, so the lowering must be at least as good there. */
    for (i = 0; i < N_SHAPES; i += 4) {
        int worst = 0; double ms = 0;
        int rc = one(fd, &SHAPES[i], 0.017f, 0.017f, 0, 0, &worst, &ms);
        if (rc == 2) { refused++; continue; }
        fails += rc;
        ran++;
    }

    /* The fused form: the skip inside a convolution the block already pays for. */
    printf("fused: a block's last convolution absorbing its own skip\n");
    fails += fused(fd, "mbv2 project C24 56x56",   24, 24 * 6, 56, 56, 1, 0); ran++;
    fails += fused(fd, "mbv2 project C96 14x14",   96, 96 * 6, 14, 14, 1, 0); ran++;
    fails += fused(fd, "mbv2 project C160 7x7",   160, 160 * 6, 7,  7, 1, 0); ran++;
    fails += fused(fd, "res18 3x3 C64 28x28",      64, 64,     28, 28, 3, 0); ran++;
    fails += fused(fd, "res18 3x3 C128 14x14",    128, 128,    14, 14, 3, 0); ran++;
    fails += fused(fd, "res18 3x3 C256 14x14",    256, 256,    14, 14, 3, 0); ran++;
    /* 2C*9 = 9216, past the 4608 weight-slice rule: the fusion must stop here. */
    fails += fused(fd, "res18 3x3 C512 7x7",      512, 512,     7,  7, 3, 1); ran++;

    /* AND WHETHER A REAL BLOCK CAN USE IT. The slice rule is not what stops the fusion on
     * a real model — the identity weight is, and it is fixed by the convolution's own
     * quantization. Every one of MobileNetV2's ten residual blocks must be refused. */
    printf("fuse: the identity weight a real block's quantization asks for\n");
    for (i = 0; i < N_MBV2; i++) {
        int w = 0; int32_t bd = 0; float rel = 0.0f;
        int rc = rocket_residual_fuse_weight_rk3576(MBV2[i].in_s, MBV2[i].w_s,
                                                    MBV2[i].skip_s, -5, 7, &w, &bd, &rel);
        double need = (double)MBV2[i].skip_s /
                      ((double)MBV2[i].in_s * (double)MBV2[i].w_s);
        printf("  %-14s needs %7.1f  %s\n", MBV2[i].name, need,
               rc == ROCKET_E_UNSUPPORTED ? "refused, as int8 requires"
                                          : "ACCEPTED past the int8 field");
        if (rc != ROCKET_E_UNSUPPORTED) fails++;
        ran++;
    }
    fails += fuse_runs(fd); ran++;

    printf("== %d shape(s), %d failed, %d refused ==\n", ran, fails, refused);
    rocket_close(fd);
    return fails ? 1 : 0;
}
