// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_dw_int8_runtime.c — end-to-end gate for the RK3588 int8 DEPTHWISE int8-out
 * runtime (rocket_conv2d_dw_int8): its host packing (the raw int8 cubes, the input
 * zero-point fold, the uint8-equivalent zero points the generator takes) and the
 * on-chip requant, against two oracles that share no code with it.
 *
 *   1. TFLite itself. tests/dw_litert_ref.py ran the capture's own model (dw_pt.tflite)
 *      under TFLite's reference kernels on a seeded random int8 input and wrote
 *      litert-in.bin / litert-out.bin beside the capture. The NPU requants through a
 *      15-bit multiplier and a shift where TFLite's multiplier is 31-bit fixed point, so
 *      an output may differ by one where it sits at a rounding boundary (TFLite's own
 *      XNNPACK kernel differs from its reference kernel on 13 of these 4096). The gate
 *      allows a difference of one on at most 2% of the elements and nothing larger.
 *   2. A host model of the same arithmetic, bit-exact: TFLite's int32 accumulator (a
 *      padded tap contributes nothing) followed by the OUT_CVT requant in
 *      tests/requant_model.h, at the captured shape and non-square ones, over full-range
 *      hashed data and zero points at both ends of the int8 range, including asymmetric
 *      weight zero points, which the DPU's CPEND operand carries, and pads past
 *      CNA_PAD_CON0's 4-bit field on planes that fit one pass (pads 16, 17 and a dilated
 *      "same" conv at 16 read 10416-17501 elements wrong on the library before 96d8e32).
 *
 * Teflon's captured output is NOT an oracle for this entry. Mesa's rocket driver is a
 * uint8 driver, and the capture ran an int8 model through it: it records Mesa's uint8
 * formulas applied to the int8 bytes, which is not the model's function (every one of its
 * 4096 outputs differs from TFLite's, by up to 133), and its input is constant zero.
 * replay_dw_mesa keeps the capture as a check of the register program.
 *
 * A fixture that is missing or short is a FAILURE: it is committed, so its absence is a
 * broken checkout rather than an environment to skip.
 *
 * Usage: conv_dw_int8_runtime [capture_dir]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "rocket_npu.h"
#include "rocket_conv.h"
#include "test_fill.h"
#include "requant_model.h"

#ifndef TEFLON_DW_CAPTURE_DIR
#define TEFLON_DW_CAPTURE_DIR "tests/data/teflon-dw-capture"
#endif

/* The capture model's quantization, as the .tflite carries it (float32). */
#define CAP_IN_S  0.036572348326444626f
#define CAP_W_S   0.0007963845855556428f
#define CAP_OUT_S 0.006911636330187321f
#define CAP_IN_ZP  (-2)
#define CAP_OUT_ZP 5

/* Exactly `want` bytes, or -1: a short file would leave the tail of the buffer holding
 * whatever malloc returned, and the gate would score that. */
static long load_file(const char *path, void *dst, long want) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    long n = (long)fread(dst, 1, want, f);
    fclose(f);
    if (n != want) { fprintf(stderr, "%s: read %ld bytes, want %ld\n", path, n, want); return -1; }
    return n;
}

/* TFLite's int8 depthwise accumulator, then the DPU's OUT_CVT requant. The scale is
 * formed in float exactly as the generator forms it. */
static void dw_model(const rocket_conv2d_desc *d, const int8_t *in, const int8_t *w,
                     const int32_t *bias, float in_s, float w_s, float out_s,
                     int in_zp, int w_zp, int out_zp, int8_t *out)
{
    unsigned mul, shift;
    requant_params(in_s * w_s / out_s, &mul, &shift);
    const int C = d->ic, IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t acc = bias ? bias[c] : 0;
                for (int kh = 0; kh < KH; kh++) {
                    int ih = oh * d->stride_y + kh * d->dil_y - d->pad_top;
                    if (ih < 0 || ih >= IH) continue;
                    for (int kw = 0; kw < KW; kw++) {
                        int iw = ow * d->stride_x + kw * d->dil_x - d->pad_left;
                        if (iw < 0 || iw >= IW) continue;
                        acc += (int64_t)(in[((size_t)c * IH + ih) * IW + iw] - in_zp) *
                               (w[((size_t)c * KH + kh) * KW + kw] - w_zp);
                    }
                }
                out[((size_t)c * OH + oh) * OW + ow] =
                    (int8_t)requant_apply_zp(acc, mul, shift, out_zp);
            }
}

/* Against a reference that may differ by one at a rounding boundary: returns the count
 * of differences larger than one, and writes the count of differences of exactly one. */
static long cmp_within_one(const char *tag, const int8_t *got, const int8_t *want,
                           const int dims[3], long *ones)
{
    const size_t n = (size_t)dims[0] * dims[1] * dims[2];
    long big = 0;
    *ones = 0;
    for (size_t i = 0; i < n; i++) {
        int dd = abs(got[i] - want[i]);
        if (dd == 1) { (*ones)++; continue; }
        if (dd == 0) continue;
        if (!big)
            printf("%s: first difference past one at (%zu,%zu,%zu): got %d, want %d\n", tag,
                   i / ((size_t)dims[1] * dims[2]), (i / dims[2]) % dims[1], i % dims[2],
                   got[i], want[i]);
        big++;
    }
    return big;
}

/* One shape over hashed full-range data against the host model, bit-exact, and against
 * the library's CPU oracle (TFLite's requant) within one. Returns 0/1. */
static int run_shape_dil(int fd, int C, int IH, int IW, int K, int sy, int sx, int pad,
                         int dil, float in_s, float w_s, float out_s, int in_zp, int w_zp,
                         int out_zp)
{
    rocket_conv2d_desc d = { .ic=C,.ih=IH,.iw=IW,.oc=C,.kh=K,.kw=K,
        .stride_y=sy,.stride_x=sx,.pad_top=pad,.pad_left=pad,.dil_y=dil,.dil_x=dil,
        .depthwise=1 };
    const int OH = rocket_conv2d_oh(&d), OW = rocket_conv2d_ow(&d);
    const size_t in_n = (size_t)C*IH*IW, out_n = (size_t)C*OH*OW;
    int8_t *in = malloc(in_n), *w = malloc((size_t)C*K*K);
    int8_t *got = malloc(out_n), *model = malloc(out_n), *oracle = malloc(out_n);
    int32_t *bias = malloc((size_t)C * sizeof(int32_t));
    int fail = 1;
    char tag[96];
    snprintf(tag, sizeof tag, "  C=%d %dx%d k%d s%dx%d p%d%s zp %d/%d/%d", C, IH, IW, K, sy,
             sx, pad, dil > 1 ? " dil" : "", in_zp, w_zp, out_zp);
    if (!in || !w || !got || !model || !oracle || !bias) { fprintf(stderr, "oom\n"); goto out; }
    uint64_t seed = tf_hash(0xD1A8, (uint64_t)(C * 131 + IH * 17 + IW * 5 + K * 3 + sy * 7 + sx +
                                               (pad > 2 ? pad * 1009 : 0) + (dil - 1) * 4001));
    tf_fill_i8(in, in_n, tf_hash(seed, 1), -128, 127);
    tf_fill_i8(w, (size_t)C*K*K, tf_hash(seed, 2), -128, 127);
    tf_fill_i32(bias, (size_t)C, tf_hash(seed, 3), -4000, 4000);
    tf_sentinel_bytes(got, out_n);
    int rp = rocket_conv2d_dw_int8_plan(&d);
    int r1 = rocket_conv2d_dw_int8(fd, &d, in, w, bias, in_s, w_s, out_s, in_zp, w_zp, out_zp, got);
    int r2 = rocket_conv2d_dw_int8(-1, &d, in, w, bias, in_s, w_s, out_s, in_zp, w_zp, out_zp, oracle);
    if (rp || r1 || r2) {
        printf("%s: plan=%d npu=%d oracle=%d -> FAIL\n", tag, rp, r1, r2);
        goto out;
    }
    dw_model(&d, in, w, bias, in_s, w_s, out_s, in_zp, w_zp, out_zp, model);
    const int dims[3] = { C, OH, OW };
    long bad = tf_cmp_i8(tag, got, model, dims, 3);
    long ones = 0, big = cmp_within_one(tag, got, oracle, dims, &ones);
    printf("%s -> %dx%d: %ld of %zu differ from the model; against the oracle %ld differ by "
           "one, %ld by more -> %s\n", tag, OH, OW, bad, out_n, ones, big,
           (bad || big) ? "FAIL" : "PASS");
    fail = (bad || big) ? 1 : 0;
out:
    free(in); free(w); free(got); free(model); free(oracle); free(bias);
    return fail;
}

static int run_shape(int fd, int C, int IH, int IW, int K, int sy, int sx, int pad,
                     float in_s, float w_s, float out_s, int in_zp, int w_zp, int out_zp)
{
    return run_shape_dil(fd, C, IH, IW, K, sy, sx, pad, 1, in_s, w_s, out_s, in_zp, w_zp, out_zp);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : TEFLON_DW_CAPTURE_DIR;
    const int C = 64, IH = 8, IW = 8, K = 3, OH = 8, OW = 8;
    const size_t in_n = (size_t)C * IH * IW, out_n = (size_t)C * OH * OW;
    int8_t *in = malloc(in_n), *w = malloc((size_t)C * K * K), *tfl = malloc(out_n);
    int8_t *got = malloc(out_n), *model = malloc(out_n);
    int32_t *bias = malloc((size_t)C * sizeof(int32_t));
    char path[1024];
    if (!in || !w || !tfl || !got || !model || !bias) { fprintf(stderr, "oom\n"); return 1; }

    snprintf(path, sizeof path, "%s/litert-in.bin", dir);
    if (load_file(path, in, (long)in_n) < 0) return 1;
    snprintf(path, sizeof path, "%s/litert-out.bin", dir);
    if (load_file(path, tfl, (long)out_n) < 0) return 1;
    snprintf(path, sizeof path, "%s/dw_w.bin", dir);
    if (load_file(path, w, (long)C * K * K) < 0) return 1;
    snprintf(path, sizeof path, "%s/dw_bias.bin", dir);
    if (load_file(path, bias, (long)C * (long)sizeof(int32_t)) < 0) return 1;

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no /dev/accel/accel0 (%d)\n", fd); return 2; }

    /* The capture model, as TFLite runs it. */
    rocket_conv2d_desc d = { .ic=C,.ih=IH,.iw=IW,.oc=C,.kh=K,.kw=K,
        .stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1,.depthwise=1 };
    tf_sentinel_bytes(got, out_n);
    int r = rocket_conv2d_dw_int8(fd, &d, in, w, bias, CAP_IN_S, CAP_W_S, CAP_OUT_S,
                                  CAP_IN_ZP, 0, CAP_OUT_ZP, got);
    if (r) { printf("rocket_conv2d_dw_int8 = %d\n==== FAIL ====\n", r); rocket_close(fd); return 1; }
    const int dims[3] = { C, OH, OW };
    long ones = 0, big = cmp_within_one("capture model vs TFLite", got, tfl, dims, &ones);
    const long ones_cap = (long)out_n / 50;
    dw_model(&d, in, w, bias, CAP_IN_S, CAP_W_S, CAP_OUT_S, CAP_IN_ZP, 0, CAP_OUT_ZP, model);
    long bad = tf_cmp_i8("capture model vs the host model", got, model, dims, 3);
    printf("capture model vs TFLite's reference kernels: %ld of %zu differ by one (cap %ld), "
           "%ld by more; vs the host model %ld differ -> %s\n", ones, out_n, ones_cap, big, bad,
           (big || ones > ones_cap || bad) ? "FAIL" : "PASS");
    int fail = big || ones > ones_cap || bad;

    printf("shapes against the host model and the library oracle:\n");
    fail |= run_shape(fd, 64,  8,  8, 3, 1, 1, 1, CAP_IN_S, CAP_W_S, CAP_OUT_S, -2, 0, 5);  /* the capture's */
    fail |= run_shape(fd, 64,  6, 10, 3, 1, 1, 1, CAP_IN_S, CAP_W_S, CAP_OUT_S, 37, 0, -20); /* H != W */
    fail |= run_shape(fd, 64,  9,  7, 3, 2, 1, 1, 0.0213f, 0.00347f, 0.0925f, -128, 0, 127); /* stride_y != stride_x */
    fail |= run_shape(fd, 128, 7,  5, 3, 1, 2, 1, CAP_IN_S, CAP_W_S, CAP_OUT_S, 127, 0, -128); /* two channel groups */
    fail |= run_shape(fd, 64,  8, 11, 5, 1, 1, 2, 0.0213f, 0.00347f, 0.0925f, 0, 0, 0);       /* 5x5, a wide plane */
    /* A PARTIAL last 64-channel group: the program leaves one unwritten unless the job is
     * padded to whole groups. C 96 and 144 are MobileDet's and SSD's own depthwise counts. */
    fail |= run_shape(fd, 96,  8,  8, 3, 1, 1, 1, CAP_IN_S, CAP_W_S, CAP_OUT_S, -2, 0, 5);   /* half a group */
    fail |= run_shape(fd, 80,  6, 10, 3, 2, 1, 1, 0.0213f, 0.00347f, 0.0925f, 37, 0, -20);  /* a quarter */
    fail |= run_shape(fd, 144, 7,  5, 3, 1, 2, 1, CAP_IN_S, CAP_W_S, CAP_OUT_S, 127, 0, -128); /* 16 past two groups */

    /* An ASYMMETRIC weight zero point, carried by the DPU's CPEND operand: at both ends of
     * the int8 range, on a partial group, and at the pad a frontend gets by materialising an
     * odd SAME pad itself (pad 0 on an odd plane at stride 2, where the uint8 detectors'
     * stride-2 layers land). The last is a MobileDet-sized layer, several channel chunks. */
    printf("asymmetric weight zero points:\n");
    fail |= run_shape(fd, 64,  8,  8, 3, 1, 1, 1, CAP_IN_S, CAP_W_S, CAP_OUT_S, -2, -62, 5);
    fail |= run_shape(fd, 96,  8,  8, 3, 1, 1, 1, CAP_IN_S, CAP_W_S, CAP_OUT_S, -2, 90, 5);
    fail |= run_shape(fd, 80,  6, 10, 3, 2, 1, 1, 0.0213f, 0.00347f, 0.0925f, 37, -128, -20);
    fail |= run_shape(fd, 144, 7,  5, 3, 1, 2, 1, CAP_IN_S, CAP_W_S, CAP_OUT_S, 127, 127, -128);
    fail |= run_shape(fd, 64,  9,  9, 3, 2, 2, 0, 0.0213f, 0.00347f, 0.0925f, -128, 17, 0);
    fail |= run_shape(fd, 128, 12, 12, 5, 1, 1, 2, 0.0213f, 0.00347f, 0.0925f, -128, -15, 127);
    fail |= run_shape(fd, 768, 24, 24, 5, 1, 1, 0, 0.0213f, 0.00347f, 0.0925f, -128, 4, -128);

    /* A PLANE PAST ONE CBUF PASS runs in row bands with the halo materialized as the input
     * zero point: SSD MobileNet v2's two 150x150 layers (the stride-2 one host-padded to an
     * odd plane, as the delegate feeds it), and an odd padded plane at 5x5 with an
     * asymmetric weight zero point, whose bands carry CPEND over the materialized halo. */
    printf("row bands:\n");
    fail |= run_shape(fd, 32, 150, 150, 3, 1, 1, 1, CAP_IN_S, CAP_W_S, CAP_OUT_S, -2, 0, 5);
    fail |= run_shape(fd, 96, 151, 151, 3, 2, 2, 0, 0.0213f, 0.00347f, 0.0925f, 37, 0, -20);
    fail |= run_shape(fd, 64, 75, 90, 5, 1, 1, 2, 0.0213f, 0.00347f, 0.0925f, -128, 17, 127);

    /* A PAD PAST 15. CNA_PAD_CON0 holds PAD_TOP and PAD_LEFT in 4 bits, and 16 behaves as 0
     * [HW sweep, tests/deconv_pad_probe.c]. Each of these planes fits one CBUF pass, so only
     * the pad routes it to the bands, which materialize the halo as the input zero point and
     * program 0. Pad 15 is the control, the field's largest value, and the last row is a
     * "same" dilated conv (k3 at dilation 16 pads 16), the shape that reaches the field. */
    printf("pads past CNA_PAD_CON0's 4-bit field:\n");
    fail |= run_shape(fd, 64,  8,  8, 3, 1, 1, 15, CAP_IN_S, CAP_W_S, CAP_OUT_S, -2, 0, 5);
    fail |= run_shape(fd, 64,  8,  8, 3, 1, 1, 16, CAP_IN_S, CAP_W_S, CAP_OUT_S, -2, 0, 5);
    fail |= run_shape(fd, 96,  6, 10, 3, 1, 1, 17, 0.0213f, 0.00347f, 0.0925f, 37, -62, -20);
    fail |= run_shape_dil(fd, 64, 20, 20, 3, 1, 1, 16, 16, 0.0213f, 0.00347f, 0.0925f, -128, 17,
                          127);

    /* The planner answers for the same program: OK above (checked per shape), and a refusal
     * along each axis it checks. */
    {
        rocket_conv2d_desc c72 = d, k25 = d, nodw = d;
        c72.ic = c72.oc = 72;
        k25.kh = k25.kw = 25; k25.pad_top = k25.pad_left = 12;
        nodw.depthwise = 0;
        int a = rocket_conv2d_dw_int8_plan(&c72), b = rocket_conv2d_dw_int8_plan(&k25),
            c = rocket_conv2d_dw_int8_plan(&nodw);
        printf("planner: C 72 -> %d, K 25 -> %d, not depthwise -> %d -> %s\n", a, b, c,
               (a == ROCKET_E_SHAPE && b == ROCKET_E_SHAPE && c == ROCKET_E_SHAPE)
                   ? "refused, PASS" : "FAIL");
        fail |= !(a == ROCKET_E_SHAPE && b == ROCKET_E_SHAPE && c == ROCKET_E_SHAPE);
    }
    /* A weight zero point outside int8 is not a TFLite int8 tensor and must refuse. */
    {
        int rr = rocket_conv2d_dw_int8(fd, &d, in, w, bias, CAP_IN_S, CAP_W_S, CAP_OUT_S,
                                       CAP_IN_ZP, 200, CAP_OUT_ZP, got);
        printf("w_zp = 200 on the NPU path: rc %d -> %s\n", rr,
               rr == ROCKET_E_UNSUPPORTED ? "refused, PASS" : "FAIL");
        fail |= rr != ROCKET_E_UNSUPPORTED;
    }
    rocket_close(fd);

    free(in); free(w); free(tfl); free(got); free(model); free(bias);
    printf("==== %s ====\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
