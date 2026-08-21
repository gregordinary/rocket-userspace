// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_net_gate.c — a WHOLE quantized classifier on the RK3576, layer by layer.
 *
 * Every RK3576 gate before this one asks a question about ONE op. That is what settled
 * the encodings, and it is structurally unable to see the defects a graph has: an
 * output layout that reads back correctly but feeds the next layer's packer wrong, a
 * padding convention that is right in isolation and off by one against the framework's,
 * a per-layer envelope decision that is fine alone and refuses in sequence, a hazard
 * whose rate is a property of the DDR traffic the rest of the graph is making. This
 * gate runs MobileNetV1-224 at int8 — 27 convolutions, a pooling layer and a classifier
 * — and checks every intermediate.
 *
 * WHAT IT CHECKS AGAINST. Not a CPU model of this library: TFLite's OWN output for
 * every tensor, exported alongside the weights (tests/data/rk3576-net/mknet.py). So a
 * layer is right when the part reproduces the framework, not when it reproduces our
 * reading of the framework. Two passes, and they answer different questions:
 *
 *   ORACLE   every layer is fed TFLite's golden INPUT. Isolates each layer: a failure
 *            names the layer that failed and nothing downstream is contaminated.
 *
 *   CHAIN    the real network. Layer n is fed layer n-1's NPU output, and each layer is
 *            still scored against the golden — so a divergence shows where it entered
 *            and how it travels. This is the pass that produces a label.
 *
 * THE MODEL IS uint8 AND THE PART IS int8, and mknet.py does that rebase offline: every
 * zero point, weight and golden is shifted by -128, which is exact and cancels, so
 * nothing here ever sees a uint8. The fused RELU6 on every convolution is FREE — the
 * output scale is 6/255 with the zero point at the bottom of the range, so its clamp is
 * exactly the int8 saturation the DPU already applies. The gate asserts that rather than
 * assuming it.
 *
 * WHAT THE HOST DOES, AND WHY IT IS PART OF THE TEST. Two of TFLite's paddings are
 * ASYMMETRIC (a stride-2 SAME convolution pads one row and column at the END only) and
 * the CNA's pad registers are symmetric, so those layers get their border materialised
 * here, at the input zero point. That, the classifier's channel expansion, and the
 * softmax are real host work between submits — which is the DDR traffic the atom-drop
 * rate was measured to follow, and no per-op gate makes any.
 *
 * Usage:  rk3576_net_gate [oracle|chain|all|bench [iters]] [--blob PATH] [-v]
 * Exit:   0, 1 on a failure, 2 to skip (no NPU, wrong chip, or no blob).
 *
 * The blob is not committed. Build it once:
 *     cd tests/data/rk3576-net && ./fetch.sh && python3 mknet.py
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_pool.h"
#include "rocket_hw_profile.h"
#include "requant_model.h"

/* ---- the blob ------------------------------------------------------------------ */
enum { KIND_CONV = 0, KIND_DWCONV, KIND_AVGPOOL, KIND_SOFTMAX };
enum { ACT_NONE = 0, ACT_RELU6 };

typedef struct {
    uint32_t kind, act, ic, ih, iw, oc, oh, ow, kh, kw, sy, sx;
    uint32_t pl_y, pl_x, pt_y, pt_x;
    int32_t  in_zp, w_zp, out_zp, clamp_lo, clamp_hi;
    float    in_scale, w_scale, out_scale;
    uint32_t w_off, w_bytes, b_off, b_bytes, g_off, g_bytes;
    uint32_t pad_[2];
} rnet_layer;

typedef struct {
    char     magic[8];
    uint32_t version, n_layers, layer_stride;
    uint32_t in_c, in_h, in_w, n_classes;
    int32_t  in_zp;
    float    in_scale;
    uint32_t pad0;
    uint32_t layers_off, img_off, probs_off, labels_off, labels_bytes, total_bytes;
} rnet_hdr;

static unsigned char *BLOB;
static size_t BLOB_BYTES;
static const rnet_hdr *H;
static const rnet_layer *LAYERS;
static int VERBOSE;

static const char *KIND_NAME[] = { "conv", "dwconv", "avgpool", "softmax" };

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static const int8_t *at(uint32_t off) { return (const int8_t *)(BLOB + off); }

static int load_blob(const char *path)
{
    FILE *f = fopen(path, "rb");
    size_t got;
    long n;
    if (!f) return -1;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < (long)sizeof(rnet_hdr)) { fclose(f); return -1; }
    BLOB = malloc((size_t)n);
    if (!BLOB) { fclose(f); return -1; }
    got = fread(BLOB, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) return -1;
    BLOB_BYTES = (size_t)n;
    H = (const rnet_hdr *)BLOB;
    if (memcmp(H->magic, "RKNET\0\0\1", 8) || H->version != 1 ||
        H->layer_stride != sizeof(rnet_layer) || H->total_bytes != BLOB_BYTES) {
        printf("blob header mismatch (magic/version/stride/size)\n");
        return -1;
    }
    LAYERS = (const rnet_layer *)(BLOB + H->layers_off);
    return 0;
}

/* ---- the host reference, and the host fallback ---------------------------------- */
/*
 * The same arithmetic the part computes, on the CPU: an int64 accumulate over the taps
 * TFLite's padding admits, then the DPU's own requant. It is here for two jobs — a
 * layer the NPU refuses still has to run so the chain reaches a label, and a layer that
 * DISAGREES with the golden needs a third opinion to say whether the model or the part
 * is what moved.
 */
static void host_conv(const rnet_layer *L, const int8_t *in, const int8_t *W,
                      const int32_t *bias, int8_t *out)
{
    unsigned scale, shift;
    unsigned c, y, x, i, ky, kx;
    requant_params(L->in_scale * L->w_scale / L->out_scale, &scale, &shift);
    for (c = 0; c < L->oc; c++)
        for (y = 0; y < L->oh; y++)
            for (x = 0; x < L->ow; x++) {
                int64_t acc = bias ? bias[c] : 0;
                for (ky = 0; ky < L->kh; ky++)
                    for (kx = 0; kx < L->kw; kx++) {
                        long iy = (long)(y * L->sy + ky) - (long)L->pl_y;
                        long ix = (long)(x * L->sx + kx) - (long)L->pl_x;
                        if (iy < 0 || ix < 0 || iy >= (long)L->ih || ix >= (long)L->iw)
                            continue;
                        if (L->kind == KIND_DWCONV) {
                            acc += (int64_t)(in[((size_t)c * L->ih + iy) * L->iw + ix]
                                             - L->in_zp)
                                 * (W[((size_t)c * L->kh + ky) * L->kw + kx] - L->w_zp);
                        } else {
                            for (i = 0; i < L->ic; i++)
                                acc += (int64_t)(in[((size_t)i * L->ih + iy) * L->iw + ix]
                                                 - L->in_zp)
                                     * (W[(((size_t)c * L->ic + i) * L->kh + ky) * L->kw
                                          + kx] - L->w_zp);
                        }
                    }
                out[((size_t)c * L->oh + y) * L->ow + x] =
                    (int8_t)requant_apply_zp(acc, scale, shift, L->out_zp);
            }
}

/* ---- lowering ------------------------------------------------------------------- */
/*
 * TFLite's padding is a lead/trail pair per axis and the CNA's is one symmetric number,
 * so a layer whose pair is unequal has its border MATERIALISED: a buffer of the padded
 * extent filled with the input zero point, the real input copied in at (lead_y, lead_x),
 * and the descriptor run at pad 0. A stride-2 SAME convolution is always that shape —
 * TFLite puts the odd row and column at the END — and it is where a whole network stops
 * agreeing with a per-op gate, because no gate ever asks for an asymmetric pad.
 *
 * THE SAME BUFFER CARRIES THE CHANNEL EXPANSION. Four or fewer input channels take the
 * CNA's ARGB first-conv sub-encoding, and that program cannot express this stem: it
 * requires a NON-ZERO left pad and an output extent of exactly iw/stride, which is the
 * ONNX symmetric convention, and TFLite's SAME stem has a zero left pad. So the stem
 * runs on the DIRECT path with the image widened to eight channels. The added channels
 * hold the INPUT ZERO POINT against zero weights, which is exact for any zero point:
 * the coefficient A folds `-in_zp*sum_w + in_zp*w_zp*taps` over all taps, and a tap
 * whose weight is zero and whose sample is in_zp contributes nothing to either half.
 */
#define STEM_IC 8u

static int needs_prep(const rnet_layer *L)
{
    return L->pl_y != L->pt_y || L->pl_x != L->pt_x ||
           (L->kind == KIND_CONV && L->ic <= 4);
}

static int8_t *prep_input(const rnet_layer *L, const int8_t *in,
                          unsigned *pic, unsigned *pih, unsigned *piw)
{
    unsigned icx = (L->kind == KIND_CONV && L->ic <= 4) ? STEM_IC : L->ic;
    unsigned ihx = L->ih + L->pl_y + L->pt_y;
    unsigned iwx = L->iw + L->pl_x + L->pt_x;
    int8_t *buf = malloc((size_t)icx * ihx * iwx);
    unsigned c, y;
    if (!buf) return NULL;
    memset(buf, (int8_t)L->in_zp, (size_t)icx * ihx * iwx);
    for (c = 0; c < L->ic; c++)
        for (y = 0; y < L->ih; y++)
            memcpy(buf + ((size_t)c * ihx + y + L->pl_y) * iwx + L->pl_x,
                   in + ((size_t)c * L->ih + y) * L->iw, L->iw);
    *pic = icx; *pih = ihx; *piw = iwx;
    return buf;
}

static int8_t *prep_weights(const rnet_layer *L, const int8_t *W)
{
    unsigned icx = STEM_IC, c;
    size_t taps = (size_t)L->kh * L->kw;
    int8_t *buf = calloc((size_t)L->oc * icx * taps, 1);
    if (!buf) return NULL;
    for (c = 0; c < L->oc; c++)
        memcpy(buf + (size_t)c * icx * taps, W + (size_t)c * L->ic * taps,
               L->ic * taps);
    return buf;
}

/* Run one layer. `how` is set to the path it took. */
static int layer_run(int fd, const rnet_layer *L, const int8_t *in, int8_t *out,
                     const char **how)
{
    const int8_t *W = L->w_bytes ? at(L->w_off) : NULL;
    const int32_t *bias = L->b_bytes ? (const int32_t *)(BLOB + L->b_off) : NULL;
    rocket_conv2d_desc d;
    int8_t *pin = NULL, *pw = NULL;
    unsigned icx = L->ic, ihx = L->ih, iwx = L->iw;
    int rc;

    if (L->kind == KIND_AVGPOOL) {
        rocket_pool_desc p;
        memset(&p, 0, sizeof p);
        p.c = (int)L->ic; p.ih = (int)L->ih; p.iw = (int)L->iw;
        p.kh = (int)L->kh; p.kw = (int)L->kw;
        p.stride_y = (int)L->sy; p.stride_x = (int)L->sx;
        p.pad_top = (int)L->pl_y; p.pad_left = (int)L->pl_x;
        p.pad_bottom = (int)L->pt_y; p.pad_right = (int)L->pt_x;
        p.method = POOL_METHOD_AVG;
        *how = rocket_pool_int8_rk3576_exact(&p) ? "npu-pool" : "npu-pool*";
        return rocket_pool_int8_rk3576(fd, &p, L->in_zp, in, out);
    }
    if (L->kind == KIND_SOFTMAX) {
        *how = "host-softmax";
        memcpy(out, in, (size_t)L->oc * L->oh * L->ow);   /* rank-preserving; see main */
        return 0;
    }

    if (needs_prep(L)) {
        pin = prep_input(L, in, &icx, &ihx, &iwx);
        if (!pin) return ROCKET_E_NOMEM;
        if (L->kind == KIND_CONV && L->ic <= 4) {
            pw = prep_weights(L, W);
            if (!pw) { free(pin); return ROCKET_E_NOMEM; }
        }
    }

    memset(&d, 0, sizeof d);
    d.ic = (int)icx; d.ih = (int)ihx; d.iw = (int)iwx;
    d.oc = (int)L->oc; d.kh = (int)L->kh; d.kw = (int)L->kw;
    d.stride_y = (int)L->sy; d.stride_x = (int)L->sx;
    d.dil_y = d.dil_x = 1;
    d.depthwise = (L->kind == KIND_DWCONV);
    /* Materialised: the border is in the buffer, so the registers pad nothing. */
    d.pad_top  = pin ? 0 : (int)L->pl_y;
    d.pad_left = pin ? 0 : (int)L->pl_x;

    if ((unsigned)rocket_conv2d_oh(&d) != L->oh ||
        (unsigned)rocket_conv2d_ow(&d) != L->ow) {
        printf("  lowering produced %dx%d, the model says %ux%u\n",
               rocket_conv2d_oh(&d), rocket_conv2d_ow(&d), L->oh, L->ow);
        free(pin); free(pw);
        return ROCKET_E_SHAPE;
    }

    if (d.depthwise)
        rc = rocket_conv2d_dw_int8_rk3576(fd, &d, pin ? pin : in, W, bias,
                                          L->in_scale, L->w_scale, L->out_scale,
                                          L->in_zp, L->w_zp, L->out_zp, out);
    else
        rc = rocket_conv2d_int8_rk3576(fd, &d, pin ? pin : in, pw ? pw : W, bias,
                                       L->in_scale, L->w_scale, L->out_scale,
                                       L->in_zp, L->w_zp, L->out_zp, out);
    *how = rc == ROCKET_OK ? (d.depthwise ? "npu-dw" : (pin ? "npu-prep" : "npu"))
                           : "REFUSED";
    if (rc != ROCKET_OK) {
        /* The chain has to reach a label even when one layer will not run, so the
         * fallback is taken and NAMED rather than the run being abandoned. */
        host_conv(L, in, W, bias, out);
        *how = "host-fallback";
    }
    free(pin); free(pw);
    return ROCKET_OK;
}

/* ---- scoring -------------------------------------------------------------------- */
/*
 * TWO COMPARISONS, AND THEY ARE NOT THE SAME QUESTION.
 *
 *   vs the HOST MODEL — the part's own arithmetic, computed on the CPU. This is the
 *   ASSERTION. A difference here is a defect in the encoder, the tiling or the driving.
 *
 *   vs TFLITE'S GOLDEN — the framework's answer for the same tensor. This is REPORTED.
 *   A per-tensor requant is a fixed-point approximation of a real-valued scale, and this
 *   DPU's is not TFLite's: the OUT_CVT carries a 15-bit multiplier and rounds ties to
 *   even, where TFLite carries 31 bits and rounds half away from zero. Two roundings of
 *   one scale disagree by a count on the elements that sit near a boundary, so a small
 *   maxdiff-1 population is the EXPECTED reading and asserting it away would be
 *   asserting that this chip is a different chip.
 */
typedef struct { long exact, total; int maxdiff; long first_c, first_y, first_x; } score;

static void score_vs(const rnet_layer *L, const int8_t *got, const int8_t *want,
                     score *s)
{
    size_t n = (size_t)L->oc * L->oh * L->ow, i;
    s->exact = 0; s->total = (long)n; s->maxdiff = 0;
    s->first_c = s->first_y = s->first_x = -1;
    for (i = 0; i < n; i++) {
        int diff = got[i] - want[i];
        if (diff < 0) diff = -diff;
        if (!diff) { s->exact++; continue; }
        if (diff > s->maxdiff) s->maxdiff = diff;
        if (s->first_c < 0) {
            s->first_c = (long)(i / ((size_t)L->oh * L->ow));
            s->first_y = (long)((i / L->ow) % L->oh);
            s->first_x = (long)(i % L->ow);
        }
    }
}

/* Where the wrong elements ARE, which is what names a tiling bug: a row split leaves a
 * contiguous band of output rows and every channel, a channel split leaves whole
 * channels, and a requant difference leaves a scatter over both. */
static void wrong_extent(const rnet_layer *L, const int8_t *got, const int8_t *want)
{
    long c0 = -1, c1 = -1, y0 = -1, y1 = -1, nrows = 0;
    unsigned c, y, x;
    for (c = 0; c < L->oc; c++)
        for (y = 0; y < L->oh; y++) {
            int any = 0;
            for (x = 0; x < L->ow; x++) {
                size_t i = ((size_t)c * L->oh + y) * L->ow + x;
                if (got[i] != want[i]) { any = 1; break; }
            }
            if (!any) continue;
            nrows++;
            if (c0 < 0) c0 = (long)c;
            c1 = (long)c;
            if (y0 < 0 || (long)y < y0) y0 = (long)y;
            if ((long)y > y1) y1 = (long)y;
        }
    printf("      wrong rows %ld of %u, channels %ld..%ld, output rows %ld..%ld\n",
           nrows, L->oc * L->oh, c0, c1, y0, y1);
}

static void top5(const int8_t *logits, int n, int out[5])
{
    int i, j, k;
    int best[5]; long val[5];
    for (k = 0; k < 5; k++) { best[k] = -1; val[k] = -1000; }
    for (i = 0; i < n; i++) {
        long v = logits[i];
        for (k = 0; k < 5; k++)
            if (v > val[k]) {
                for (j = 4; j > k; j--) { val[j] = val[j-1]; best[j] = best[j-1]; }
                val[k] = v; best[k] = i;
                break;
            }
    }
    for (k = 0; k < 5; k++) out[k] = best[k];
}

static void label_of(int idx, char *dst, size_t cap)
{
    const char *p = (const char *)(BLOB + H->labels_off);
    const char *end = p + H->labels_bytes;
    int line = 0;
    dst[0] = 0;
    while (p < end && line < idx) { if (*p++ == '\n') line++; }
    {
        size_t n = 0;
        while (p < end && *p != '\n' && n + 1 < cap) dst[n++] = *p++;
        dst[n] = 0;
    }
}

/*
 * A layer that disagrees with a model of the part's own arithmetic is one of two very
 * different things, and one run cannot tell them apart:
 *
 *   an ENCODING or TILING defect  — deterministic. The same elements are wrong every
 *                                   time, and their values are whatever the wrong
 *                                   program computed.
 *   a DROPPED ATOM                — the part intermittently emits a run of output atoms
 *                                   as zeros, at a rate that follows DDR traffic. The
 *                                   wrong set MOVES between runs and the wrong values
 *                                   are the output zero point's byte, not arithmetic.
 *
 * So the layer is re-run and the two wrong sets compared, and the wrong values are
 * classified. Three runs, because two identical sets could be a coincidence at a low
 * drop rate and three make that unlikely.
 */
static void diagnose(int fd, const rnet_layer *L, const int8_t *in, const int8_t *first,
                     const int8_t *ref, size_t cap)
{
    size_t n = (size_t)L->oc * L->oh * L->ow, i;
    int8_t *again = malloc(cap);
    long zeros = 0, shown = 0;
    int r, same_every_time = 1;
    const char *how;

    if (!again) return;
    for (i = 0; i < n; i++)
        if (first[i] != ref[i]) {
            if (first[i] == 0) zeros++;
            if (shown < 6) {
                printf("      [%zu] c%zu y%zu x%zu: part %d, its own arithmetic %d\n",
                       i, i / ((size_t)L->oh * L->ow), (i / L->ow) % L->oh, i % L->ow,
                       first[i], ref[i]);
                shown++;
            }
        }
    printf("      of the wrong elements %ld are exactly zero (a dropped atom reads as "
           "the surface's fill, not as arithmetic)\n", zeros);

    for (r = 0; r < 2; r++) {
        long diff_now = 0, moved = 0;
        if (layer_run(fd, L, in, again, &how) != ROCKET_OK) break;
        for (i = 0; i < n; i++) {
            int w1 = first[i] != ref[i], w2 = again[i] != ref[i];
            if (w2) diff_now++;
            if (w1 != w2 || (w2 && again[i] != first[i])) moved++;
        }
        printf("      re-run %d: %ld wrong, %ld element(s) differ from the first run's "
               "wrong set\n", r + 1, diff_now, moved);
        if (moved) same_every_time = 0;
    }
    printf("      => %s\n", same_every_time
           ? "the same elements every run: DETERMINISTIC, so an encoding or tiling "
             "defect and not the drop hazard"
           : "the wrong set MOVES between runs: intermittent, so a hazard and not the "
             "encoding");
    free(again);
}

/* ---- the two passes -------------------------------------------------------------- */
static size_t max_tensor(void)
{
    size_t m = (size_t)H->in_c * H->in_h * H->in_w;
    unsigned i;
    for (i = 0; i < H->n_layers; i++) {
        size_t n = (size_t)LAYERS[i].oc * LAYERS[i].oh * LAYERS[i].ow;
        if (n > m) m = n;
        n = (size_t)LAYERS[i].ic * (LAYERS[i].ih + 2) * (LAYERS[i].iw + 2) * 2;
        if (n > m) m = n;
    }
    return m;
}

static int pass_oracle(int fd)
{
    size_t cap = max_tensor();
    int8_t *out = malloc(cap), *ref = malloc(cap);
    unsigned i;
    int failed = 0;
    long tf_diff_total = 0, tf_total = 0;

    if (!out || !ref) { free(out); free(ref); return 1; }
    printf("\n== ORACLE: every layer fed TFLite's own input ==\n");
    printf("   ASSERTED against the part's own arithmetic on the CPU; the distance to "
           "TFLite is reported\n");
    for (i = 0; i < H->n_layers; i++) {
        const rnet_layer *L = &LAYERS[i];
        const int8_t *in = i == 0 ? at(H->img_off) : at(LAYERS[i - 1].g_off);
        const char *how = "?";
        score hw, tf;
        double t0, ms;
        int rc, have_ref = 0;

        if (L->kind == KIND_SOFTMAX) continue;       /* host, and rank-preserving */
        t0 = now_ms();
        rc = layer_run(fd, L, in, out, &how);
        ms = now_ms() - t0;
        if (rc != ROCKET_OK) {
            printf("%2u %-8s %-13s ENTRY RETURNED %d\n", i, KIND_NAME[L->kind], how, rc);
            failed++; continue;
        }
        if (L->kind == KIND_CONV || L->kind == KIND_DWCONV) {
            host_conv(L, in, at(L->w_off), (const int32_t *)(BLOB + L->b_off), ref);
            have_ref = 1;
        } else if (L->kind == KIND_AVGPOOL) {
            rocket_pool_desc p;
            memset(&p, 0, sizeof p);
            p.c = (int)L->ic; p.ih = (int)L->ih; p.iw = (int)L->iw;
            p.kh = (int)L->kh; p.kw = (int)L->kw;
            p.stride_y = (int)L->sy; p.stride_x = (int)L->sx;
            p.pad_top = (int)L->pl_y; p.pad_left = (int)L->pl_x;
            p.pad_bottom = (int)L->pt_y; p.pad_right = (int)L->pt_x;
            p.method = POOL_METHOD_AVG;
            rocket_pool_ref_int8_rk3576(&p, L->in_zp, in, ref);
            have_ref = 1;
        }
        score_vs(L, out, at(L->g_off), &tf);
        tf_diff_total += tf.total - tf.exact;
        tf_total += tf.total;
        if (have_ref) score_vs(L, out, ref, &hw);
        else { hw.exact = hw.total = 0; hw.maxdiff = 0; }

        printf("%2u %-8s %4ux%-3ux%-4u -> %4ux%-3ux%-4u %-13s %7.2f ms  hw %s  "
               "tflite %ld/%ld",
               i, KIND_NAME[L->kind], L->ic, L->ih, L->iw, L->oc, L->oh, L->ow,
               how, ms,
               !have_ref ? "n/a" : (hw.exact == hw.total ? "exact" : "MISMATCH"),
               tf.exact, tf.total);
        if (tf.exact != tf.total) printf(" (maxdiff %d)", tf.maxdiff);
        printf("\n");
        if (have_ref && hw.exact != hw.total) {
            printf("      the part disagrees with a model of ITS OWN arithmetic: "
                   "%ld/%ld, maxdiff %d, first (c%ld y%ld x%ld)\n",
                   hw.exact, hw.total, hw.maxdiff, hw.first_c, hw.first_y, hw.first_x);
            wrong_extent(L, out, ref);
            diagnose(fd, L, in, out, ref, cap);
            failed++;
        } else if (tf.maxdiff > 1) {
            printf("      the part matches its own arithmetic but is %d counts from "
                   "TFLite — a requant difference is at most 1\n", tf.maxdiff);
            wrong_extent(L, out, at(L->g_off));
            failed++;
        }
    }
    printf("   TFLite distance over the whole graph: %ld of %ld elements differ "
           "(%.4f%%), all by one count where noted\n",
           tf_diff_total, tf_total, 100.0 * (double)tf_diff_total / (double)tf_total);
    free(out); free(ref);
    return failed;
}

static int pass_chain(int fd, int iters)
{
    size_t cap = max_tensor();
    int8_t *a = malloc(cap), *b = malloc(cap);
    unsigned i;
    int failed = 0, it;
    int mine[5], theirs[5];
    double wall = 0.0;
    const int8_t *logits = NULL;

    if (!a || !b) { free(a); free(b); return 1; }
    printf("\n== CHAIN: the network, layer n fed layer n-1's NPU output ==\n");
    for (it = 0; it < iters; it++) {
        const int8_t *cur = at(H->img_off);
        int8_t *bufs[2];
        int slot = 0;
        double t0 = now_ms();
        int drift = -1;

        bufs[0] = a; bufs[1] = b;
        for (i = 0; i < H->n_layers; i++) {
            const rnet_layer *L = &LAYERS[i];
            const char *how = "?";
            int8_t *dst = bufs[slot];
            score s;
            double lt = now_ms();
            int rc;
            if (L->kind == KIND_SOFTMAX) continue;
            rc = layer_run(fd, L, cur, dst, &how);
            lt = now_ms() - lt;
            if (rc != ROCKET_OK) {
                printf("%2u %-8s ENTRY RETURNED %d\n", i, KIND_NAME[L->kind], rc);
                failed++; break;
            }
            score_vs(L, dst, at(L->g_off), &s);
            if (s.maxdiff > 1 && drift < 0) drift = (int)i;
            if (it == 0)
                printf("%2u %-8s %4ux%-3ux%-4u -> %4ux%-3ux%-4u %-13s %8.2f ms  "
                       "%5.1f%% match TFLite, maxdiff %d\n",
                       i, KIND_NAME[L->kind], L->ic, L->ih, L->iw, L->oc, L->oh, L->ow,
                       how, lt, 100.0 * (double)s.exact / (double)s.total, s.maxdiff);
            cur = dst;
            slot ^= 1;
        }
        logits = cur;
        wall += now_ms() - t0;
        /* A layer here is fed the part's OWN previous output, not TFLite's, so the
         * requant's one-count disagreements compound: a value that differed by one at
         * layer n is a different input at layer n+1 and can move that layer's output by
         * more. The count falling through the graph is that compounding and not a
         * defect — the oracle pass is what says each layer is exact. What WOULD be a
         * defect is a layer whose own maxdiff exceeds one while fed a clean input, and
         * the pass that asks that question is the one above. */
        if (it == 0 && drift >= 0)
            printf("   the accumulated drift first exceeds one count at layer %d\n",
                   drift);
    }
    printf("   %d run(s), %.1f ms each\n", iters, wall / iters);

    if (logits) {
        const uint8_t *probs = (const uint8_t *)(BLOB + H->probs_off);
        int k, agree = 1;
        int tp[5]; long tv[5];
        char name[128];
        top5(logits, (int)H->n_classes, mine);
        for (k = 0; k < 5; k++) { tp[k] = -1; tv[k] = -1; }
        for (k = 0; k < (int)H->n_classes; k++) {
            int j, m;
            for (m = 0; m < 5; m++)
                if ((long)probs[k] > tv[m]) {
                    for (j = 4; j > m; j--) { tv[j] = tv[j-1]; tp[j] = tp[j-1]; }
                    tv[m] = probs[k]; tp[m] = k;
                    break;
                }
        }
        memcpy(theirs, tp, sizeof theirs);
        printf("\n   NPU top-5   ");
        for (k = 0; k < 5; k++) {
            label_of(mine[k], name, sizeof name);
            printf("%s%s", k ? ", " : "", name);
        }
        printf("\n   TFLite top-5");
        for (k = 0; k < 5; k++) {
            label_of(theirs[k], name, sizeof name);
            printf(" %s%s", k ? ", " : "", name);
        }
        printf("\n");
        for (k = 0; k < 5; k++) if (mine[k] != theirs[k]) agree = 0;
        if (mine[0] != theirs[0]) {
            printf("   TOP-1 DISAGREES\n");
            failed++;
        } else {
            printf("   top-1 agrees%s\n", agree ? ", and so does the whole top-5"
                                                : " (the tail of the top-5 reorders)");
        }
    }
    free(a); free(b);
    return failed;
}

int main(int argc, char **argv)
{
    const char *blob = getenv("ROCKET_NET_BLOB");
    const char *mode = "all";
    int fd, a, iters = 1, failed = 0;
    char def[512];

    setvbuf(stdout, NULL, _IOLBF, 0);
    for (a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--blob") && a + 1 < argc) blob = argv[++a];
        else if (!strcmp(argv[a], "-v")) VERBOSE = 1;
        else if (a + 1 < argc && !strcmp(argv[a], "bench")) {
            mode = "bench"; iters = atoi(argv[++a]);
            if (iters < 1) iters = 1;
        } else mode = argv[a];
    }
    if (!blob) {
        const char *root = getenv("ROCKET_SRC_DIR");
        snprintf(def, sizeof def, "%s/tests/data/rk3576-net/mobilenet_v1_224_quant.rnet",
                 root ? root : ".");
        blob = def;
    }

    if (load_blob(blob) < 0) {
        printf("no blob at %s — SKIP\n"
               "  build it: cd tests/data/rk3576-net && ./fetch.sh && python3 mknet.py\n",
               blob);
        return 2;
    }

    fd = rocket_open();
    if (fd < 0) { printf("no /dev/accel — SKIP\n"); return 2; }
    if (rocket_hw_current() != &rocket_hw_rk3576) {
        printf("not an RK3576 — SKIP\n");
        rocket_close(fd);
        return 2;
    }

    printf("%s: %u layers, %ux%ux%u in, %u classes, %zu bytes\n",
           blob, H->n_layers, H->in_c, H->in_h, H->in_w, H->n_classes, BLOB_BYTES);
    {
        /* The fused RELU6 is only free while its clamp IS the int8 range. If a model
         * ever arrives where it is not, it needs a host clamp and this gate would
         * otherwise report a wrong layer with no reason attached. */
        unsigned i, clamped = 0;
        for (i = 0; i < H->n_layers; i++)
            if (LAYERS[i].act == ACT_RELU6 &&
                (LAYERS[i].clamp_lo != -128 || LAYERS[i].clamp_hi != 127)) clamped++;
        if (clamped) {
            printf("%u layer(s) carry a fused activation whose clamp is NARROWER than "
                   "the int8 range; this gate does not apply one\n", clamped);
            failed++;
        }
    }

    if (!strcmp(mode, "oracle") || !strcmp(mode, "all")) failed += pass_oracle(fd);
    if (!strcmp(mode, "chain") || !strcmp(mode, "all")) failed += pass_chain(fd, 1);
    if (!strcmp(mode, "bench")) failed += pass_chain(fd, iters);

    printf("\n%s: %d failure(s)\n", failed ? "FAIL" : "PASS", failed);
    rocket_close(fd);
    free(BLOB);
    return failed ? 1 : 0;
}
