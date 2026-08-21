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
 * gate runs a whole quantized classifier at int8 — MobileNetV1-224, MobileNetV2-224 or
 * ResNet-18-224 — and checks every intermediate.
 *
 * The three ask different questions of the part. V1 is the feed-forward chain. V2 adds
 * ten residual skips and channel counts that are not multiples of 32. ResNet-18 is
 * 3x3-heavy where both MobileNets are 1x1-heavy, and brings a 7x7 stem, a max pool, a
 * fused ReLU on its adds, and explicit torchvision padding — a LEAD pad larger than the
 * trailing one, which TFLite's SAME never produces.
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
 *   PER-AXIS the same network with PER-OUTPUT-CHANNEL weight scales, chained. A per-axis
 *            requant's error is a per-layer quantity that compounds, and no per-op gate
 *            can say what twenty-nine of them do to a label. Its reference is an EXACT
 *            per-axis requant rather than TFLite, because the model is per-tensor and
 *            this pass re-quantizes its weights.
 *
 * THE MOBILENETS ARE uint8 AND THE PART IS int8, and mknet.py does that rebase offline:
 * every zero point, weight and golden is shifted by -128, which is exact and cancels, so
 * nothing here ever sees a uint8. ResNet-18 is quantized locally and is already int8.
 * The fused activation on a convolution or an add is FREE — a quantizer that sees a ReLU
 * puts the zero point at the bottom of the range, and a RELU6's output scale is 6/255 —
 * so its clamp is exactly the int8 saturation the DPU already applies. The gate asserts
 * that rather than assuming it.
 *
 * WHAT THE HOST DOES, AND WHY IT IS PART OF THE TEST. Two of TFLite's paddings are
 * ASYMMETRIC (a stride-2 SAME convolution pads one row and column at the END only) and
 * the CNA's pad registers are symmetric, so those layers get their border materialised
 * here, at the input zero point. That, the classifier's channel expansion, and the
 * softmax are real host work between submits — which is the DDR traffic the atom-drop
 * rate was measured to follow, and no per-op gate makes any.
 *
 * Usage:  rk3576_net_gate [oracle|chain|peraxis|cube|hostchain|all|bench [iters]]
 *         [--net v1|v2|r18] [--blob PATH] [-v]
 * Exit:   0, 1 on a failure, 2 to skip (no NPU, wrong chip, or no blob).
 *
 * `hostchain` needs NO DEVICE — it runs the graph on the CPU three ways and is what
 * separates a per-axis accuracy result from an operand-wiring one.
 *
 * The blobs are not committed. Build them once:
 *     cd tests/data/rk3576-net && ./fetch.sh && python3 mknet.py
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_pool.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "requant_model.h"

/* ---- the blob ------------------------------------------------------------------ */
enum { KIND_CONV = 0, KIND_DWCONV, KIND_AVGPOOL, KIND_SOFTMAX, KIND_ADD, KIND_MAXPOOL };
enum { ACT_NONE = 0, ACT_RELU6, ACT_RELU };

/* Both pooling kinds are the same PPU program with a different reduction, so every site
 * that asks "is this layer the pool" means both. */
#define IS_POOL(k) ((k) == KIND_AVGPOOL || (k) == KIND_MAXPOOL)

/* A layer's operands are named by the LAYER that produced them rather than assumed to be
 * the one before. A feed-forward chain does not need that and a residual one does: a
 * skip's second operand is produced three to five layers back. NO_SRC is the network
 * input, and on `src2` it means the layer has no second operand. */
#define NO_SRC 0xFFFFFFFFu

typedef struct {
    uint32_t kind, act, ic, ih, iw, oc, oh, ow, kh, kw, sy, sx;
    uint32_t pl_y, pl_x, pt_y, pt_x;
    int32_t  in_zp, w_zp, out_zp, clamp_lo, clamp_hi;
    float    in_scale, w_scale, out_scale;
    uint32_t w_off, w_bytes, b_off, b_bytes, g_off, g_bytes;
    uint32_t src1, src2;
    int32_t  in2_zp;
    float    in2_scale;
    uint32_t pad_[6];
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

static const char *KIND_NAME[] = { "conv", "dwconv", "avgpool", "softmax", "add",
                                   "maxpool" };

/* A pooling layer's descriptor, in one place: the two kinds differ only in the
 * reduction, and the pad the blob carries is the pad the last window CONSUMES. */
static void pool_desc_of(const rnet_layer *L, rocket_pool_desc *p)
{
    memset(p, 0, sizeof *p);
    p->c = (int)L->ic; p->ih = (int)L->ih; p->iw = (int)L->iw;
    p->kh = (int)L->kh; p->kw = (int)L->kw;
    p->stride_y = (int)L->sy; p->stride_x = (int)L->sx;
    p->pad_top = (int)L->pl_y; p->pad_left = (int)L->pl_x;
    p->pad_bottom = (int)L->pt_y; p->pad_right = (int)L->pt_x;
    p->method = L->kind == KIND_MAXPOOL ? POOL_METHOD_MAX : POOL_METHOD_AVG;
}

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
    if (memcmp(H->magic, "RKNET\0\0\1", 8) || H->version != 2 ||
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
/* One rectangle of the output. The whole surface is the usual call; a sub-rectangle is
 * what recomputes the packed-image stem's border, so both come out of ONE arithmetic. */
static void host_conv_rect(const rnet_layer *L, const int8_t *in, const int8_t *W,
                           const int32_t *bias, int8_t *out,
                           unsigned y0, unsigned y1, unsigned x0, unsigned x1)
{
    /* The bounds live in the blob, which the compiler must assume the output aliases, so
     * every one of them is reloaded per tap unless it is hoisted. That costs about 20
     * cycles a MAC, and this is not only the reference any more — the packed-image stem's
     * border is COMPUTED here, per inference. Same arithmetic, in registers. */
    const unsigned IC = L->ic, IH = L->ih, IW = L->iw;
    const unsigned OC = L->oc, OH = L->oh, OW = L->ow;
    const unsigned KH = L->kh, KW = L->kw, SY = L->sy, SX = L->sx;
    const long PY = (long)L->pl_y, PX = (long)L->pl_x;
    const int IZ = L->in_zp, WZ = L->w_zp, OZ = L->out_zp;
    const int dw = (L->kind == KIND_DWCONV);
    const size_t iplane = (size_t)IH * IW, wtaps = (size_t)KH * KW;
    unsigned scale, shift;
    unsigned c, y, x, i, ky, kx;

    requant_params(L->in_scale * L->w_scale / L->out_scale, &scale, &shift);
    for (c = 0; c < OC; c++) {
        const int8_t *wc = W + (size_t)c * (dw ? wtaps : (size_t)IC * wtaps);
        const int64_t b0 = bias ? bias[c] : 0;
        for (y = y0; y < y1; y++)
            for (x = x0; x < x1; x++) {
                int64_t acc = b0;
                for (ky = 0; ky < KH; ky++) {
                    long iy = (long)(y * SY + ky) - PY;
                    if (iy < 0 || iy >= (long)IH) continue;
                    for (kx = 0; kx < KW; kx++) {
                        long ix = (long)(x * SX + kx) - PX;
                        const int8_t *ip, *wp;
                        if (ix < 0 || ix >= (long)IW) continue;
                        wp = wc + (size_t)ky * KW + kx;
                        if (dw) {
                            acc += (int64_t)(in[(size_t)c * iplane + (size_t)iy * IW + ix]
                                             - IZ) * (*wp - WZ);
                            continue;
                        }
                        ip = in + (size_t)iy * IW + ix;
                        for (i = 0; i < IC; i++)
                            acc += (int64_t)(ip[i * iplane] - IZ)
                                 * (wp[i * wtaps] - WZ);
                    }
                }
                out[((size_t)c * OH + y) * OW + x] =
                    (int8_t)requant_apply_zp(acc, scale, shift, OZ);
            }
    }
}

static void host_conv(const rnet_layer *L, const int8_t *in, const int8_t *W,
                      const int32_t *bias, int8_t *out)
{
    host_conv_rect(L, in, W, bias, out, 0, L->oh, 0, L->ow);
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

/*
 * ...AND THE WIDENING IS NOT NEEDED EITHER. The int8 direct cube is a 32-channel MAC group
 * at every count — rocket_rk3576_pad_ic(3) and _pad_ic(32) are both 32 — so ic 3 and ic 8
 * are the SAME register program, the same weight-cube size and the same feature-cube size,
 * and the channels past ic are the cube's own padding. All the widening bought was reaching
 * the direct path at all, which rocket_conv2d_desc.direct_datapath now says outright.
 *
 * So the stem scatters 3x224x224 = 150 KB instead of 8x224x224 = 401 KB, and the host
 * buffer holding the widened image goes away with it.
 *
 * ROCKET_RK3576_NET_WIDEN=1 goes back to widening, which is what the stem A/B is: the two
 * lowerings must agree byte for byte over the whole surface. They are exact for a different
 * reason each — the widened one because its added channels hold the input zero point
 * against zero weights, the narrow one because the library fills its cube's padding
 * channels with the same zero point and folds over the PROGRAMMED tap count.
 */
static int WIDEN_ON = -1;                   /* -1 = not yet read from the environment */

static int widen_on(void)
{
    if (WIDEN_ON < 0) {
        const char *e = getenv("ROCKET_RK3576_NET_WIDEN");
        WIDEN_ON = e && *e && *e != '0';
    }
    return WIDEN_ON;
}

/*
 * AN ASYMMETRIC PAD IS AN OUTPUT EXTENT, NOT A MATERIALISED BORDER.
 *
 * The CNA takes the pad its LAST WINDOW CONSUMES, derived from the output extent and the
 * leading pad — it has no configured trailing pad. So TFLite's SAME at an even input and
 * stride two (pad_before 0, pad_after 1) is a leading pad of zero against an output extent
 * one larger than the symmetric formula gives, which `rocket_conv2d_desc.oh/.ow` now say.
 * Nothing has to be copied.
 *
 * That geometry is already inside the int8 correctness envelope: six shapes of
 * rk3576_conv_shapes.h carry a zero leading pad against a consumed trailing one, direct and
 * depthwise. What kept it out of this graph was the descriptor, not the part.
 *
 * ROCKET_RK3576_NET_ASYMPAD=0 goes back to materialising the border, which is what the
 * A/B is: the two lowerings must agree byte for byte.
 */
static int asympad_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_NET_ASYMPAD");
        cached = !e || !*e || *e != '0';
    }
    return cached;
}

/* Whether this layer's border or channel count has to be copied into a row-major buffer.
 * Neither does by default now: the pad is an output extent and a narrow channel count runs
 * direct. Both clauses are the A/B controls. */
static int needs_prep(const rnet_layer *L)
{
    if (L->kind == KIND_CONV && L->ic <= 4 && widen_on()) return 1;
    if (!asympad_on()) return L->pl_y != L->pt_y || L->pl_x != L->pt_x;
    return 0;
}

/* Whether this layer reaches the direct datapath at a channel count that would otherwise
 * be routed to the packed-image first conv. */
static int narrow_direct(const rnet_layer *L)
{
    return L->kind == KIND_CONV && L->ic <= 4 && !widen_on();
}

static int8_t *prep_input(const rnet_layer *L, const int8_t *in,
                          unsigned *pic, unsigned *pih, unsigned *piw)
{
    unsigned icx = (L->kind == KIND_CONV && L->ic <= 4 && widen_on()) ? STEM_IC : L->ic;
    /* With the extent form on, the pad is the CNA's and only the channels are widened. */
    unsigned py = asympad_on() ? 0u : L->pl_y, px = asympad_on() ? 0u : L->pl_x;
    unsigned ihx = L->ih + (asympad_on() ? 0u : L->pl_y + L->pt_y);
    unsigned iwx = L->iw + (asympad_on() ? 0u : L->pl_x + L->pt_x);
    int8_t *buf = malloc((size_t)icx * ihx * iwx);
    unsigned c, y;
    if (!buf) return NULL;
    memset(buf, (int8_t)L->in_zp, (size_t)icx * ihx * iwx);
    for (c = 0; c < L->ic; c++)
        for (y = 0; y < L->ih; y++)
            memcpy(buf + ((size_t)c * ihx + y + py) * iwx + px,
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

/* ============================================================================
 * SECTION — the TFLite stem on the packed-image path
 *
 * The widening above is exact and costs a whole extra convolution's worth of feature
 * traffic: eight channels where the model has three, over a materialised 225x225 border.
 * The CNA's own packed-image sub-encoding runs the same stem in a fraction of that, and
 * the reason it was taken to be unreachable is a parity argument that is incomplete.
 *
 * That program needs a NON-ZERO left pad and an output extent of exactly iw/stride —
 * jointly the ONNX symmetric-SAME convention — while TFLite's SAME puts its odd row and
 * column at the END, so its lead pad is zero. Padding cannot fix that: at stride 2 a pad
 * of one moves the sample grid by half a step. But the grid also moves when the INPUT
 * moves, and an input shifted one pixel against a pad of one is a shift of TWO — a whole
 * stride.
 *
 * So: extend the image by TFLite's trailing border at the input zero point, and feed the
 * program that buffer from (dy, dx) at the ORIGINAL extent with a register pad of P on
 * each axis, where `d = P - lead`. Output (y,x) then taps exactly the samples TFLite's
 * grid taps, at every position except those whose outermost taps fall off the shifted
 * buffer while the extended image still holds a sample there. Those are the output's own
 * leading rows and columns — 1.8% of a 224 stem — and they are recomputed on the host by
 * the same host_conv_rect() the rest of this gate asserts against.
 *
 * The widened lowering stays as the fallback. It is exact at ANY input zero point and
 * this path is not: its border is wrong there and the entry refuses it.
 * [HW sweep, H96 MAX M9, tests/rk3576_argb_pad.c]
 *
 * IT IS OFF BY DEFAULT, BECAUSE AGAINST A RESIDENT GRAPH IT IS A LOSS. Measured on this
 * network, interleaved at `bench 100`: 5.5-5.8 ms against the widened direct path's
 * 3.5-3.7, taking the inference from ~30 ms to ~31.5 even though it drops the layer from
 * five submits to one. The earlier "worth ~4 ms" reading compared it to the widened path
 * TRANSIENT (10.6-12.2 ms), and resident weights had already collected that.
 *
 * Two measured costs stand between it and a win, and both are nameable:
 *
 *   THE LIBRARY'S PACKED-IMAGE PATH HAS NO RESIDENT HANDLE. Profiled
 *   (ROCKET_RK3576_INT8_PROF=1) the call is 3.73 ms, of which BO teardown alone is 1.29
 *   (36%) — it allocates and frees the image, regcmd, weight, coefficient and output
 *   buffers on every inference, which is precisely the cost rocket_conv2d_int8_pack_
 *   rk3576() removes from the direct path and which that entry refuses at ic <= 4.
 *
 *   THE BORDER FIX-UP IS 1.7-1.9 ms FOR 1.8% OF THE SURFACE. It is memory-bound rather
 *   than MAC-bound — 7136 outputs, nine scattered input cache lines each — so hoisting
 *   the loop's bounds into registers moved it by under 0.2 ms.
 *
 * Both fixed it lands near 2.9 ms against 3.5, so this is a sub-millisecond lever on a
 * 30 ms graph and not the several-millisecond one it was taken for. Kept, gated and
 * exact: ROCKET_RK3576_NET_TFLITE_STEM=1.
 * ==========================================================================*/
struct stem_plan {
    unsigned pad_y, pad_x;      /* the register pads the program is given          */
    unsigned dy, dx;            /* where its window starts inside the extended image */
    unsigned lo_y, hi_y;        /* output rows and columns the host must recompute */
    unsigned lo_x, hi_x;
};

/* Along one axis, count the leading and trailing output positions whose taps read the
 * hardware's pad constant where the EXTENDED image holds a sample of its own. Everywhere
 * else the pad the program supplies is the pad TFLite's grid wants, so nothing is owed. */
static void stem_border(unsigned o, unsigned s, unsigned k, unsigned pad, unsigned d,
                        unsigned n, unsigned trail, unsigned *lo, unsigned *hi)
{
    unsigned y, ky;
    *lo = *hi = 0;
    for (y = 0; y < o; y++) {
        int bad = 0;
        for (ky = 0; ky < k; ky++) {
            long t = (long)y * s + (long)ky - (long)pad;    /* in the shifted buffer */
            long e = t + (long)d;                           /* in the extended image */
            if ((t < 0 || t >= (long)n) && e >= 0 && e < (long)(n + trail)) bad = 1;
        }
        if (!bad) break;
        (*lo)++;
    }
    for (y = o; y-- > *lo; ) {
        int bad = 0;
        for (ky = 0; ky < k; ky++) {
            long t = (long)y * s + (long)ky - (long)pad;
            long e = t + (long)d;
            if ((t < 0 || t >= (long)n) && e >= 0 && e < (long)(n + trail)) bad = 1;
        }
        if (!bad) break;
        (*hi)++;
    }
}

/* Is this layer reachable on the packed-image path, and with what shift? The entry's own
 * bounds are re-stated here so the lowering is DECIDED rather than discovered from a
 * refusal — each of them is measured and silent if violated. */
static int STEM_OFF = -1;                   /* -1 = not yet read from the environment */

static int stem_plan_of(const rnet_layer *L, struct stem_plan *sp)
{
    const unsigned P = 1u;                  /* the pad a 3x3 SAME convolution wants */

    if (STEM_OFF < 0) {
        const char *e = getenv("ROCKET_RK3576_NET_TFLITE_STEM");
        STEM_OFF = !(e && *e && *e != '0');      /* opt-in; see the section comment */
    }
    if (STEM_OFF) return 0;
    if (L->kind != KIND_CONV || L->ic > 4) return 0;
    if (L->in_zp != 0) return 0;            /* a wrong border on this path          */
    if (L->kh != 3 || L->kw != 3) return 0;
    if (L->iw % 16u || L->ow % 16u) return 0;
    if (L->ow * L->sx != L->iw || L->oh * L->sy != L->ih) return 0;
    if (L->oc % 32u) return 0;
    if (L->pl_y > P || L->pl_x > P) return 0;
    sp->pad_y = P; sp->pad_x = P;
    sp->dy = P - L->pl_y; sp->dx = P - L->pl_x;
    /* The window has to stay inside the extended image, which is only `trail` longer. */
    if (sp->dy > L->pt_y || sp->dx > L->pt_x) return 0;
    stem_border(L->oh, L->sy, L->kh, sp->pad_y, sp->dy, L->ih, L->pt_y,
                &sp->lo_y, &sp->hi_y);
    stem_border(L->ow, L->sx, L->kw, sp->pad_x, sp->dx, L->iw, L->pt_x,
                &sp->lo_x, &sp->hi_x);
    /* A frame that swallows the surface is not a lowering. */
    if (sp->lo_y + sp->hi_y >= L->oh || sp->lo_x + sp->hi_x >= L->ow) return 0;
    return 1;
}

/* The extended image read from (dy, dx), at the original extent. The rows and columns
 * that fall past the real image are the trailing border, which is the input zero point. */
static int8_t *stem_shift_input(const rnet_layer *L, const struct stem_plan *sp,
                                const int8_t *in)
{
    size_t plane = (size_t)L->ih * L->iw;
    int8_t *buf = malloc((size_t)L->ic * plane);
    unsigned c, y;
    if (!buf) return NULL;
    memset(buf, (int8_t)L->in_zp, (size_t)L->ic * plane);
    for (c = 0; c < L->ic; c++)
        for (y = 0; y + sp->dy < L->ih; y++)
            memcpy(buf + (size_t)c * plane + (size_t)y * L->iw,
                   in + (size_t)c * plane + (size_t)(y + sp->dy) * L->iw + sp->dx,
                   L->iw - sp->dx);
    return buf;
}

/* The frame the program could not compute, over the ORIGINAL input at TFLite's own lead
 * pads — which is what host_conv_rect() already describes. */
static void stem_fixup(const rnet_layer *L, const struct stem_plan *sp, const int8_t *in,
                       const int8_t *W, const int32_t *bias, int8_t *out)
{
    unsigned y0 = sp->lo_y, y1 = L->oh - sp->hi_y;
    if (sp->lo_y) host_conv_rect(L, in, W, bias, out, 0, sp->lo_y, 0, L->ow);
    if (sp->hi_y) host_conv_rect(L, in, W, bias, out, y1, L->oh, 0, L->ow);
    if (sp->lo_x) host_conv_rect(L, in, W, bias, out, y0, y1, 0, sp->lo_x);
    if (sp->hi_x) host_conv_rect(L, in, W, bias, out, y0, y1, L->ow - sp->hi_x, L->ow);
}

/* ============================================================================
 * SECTION — the residual add, and the buffers a skip needs
 *
 * MobileNetV2 puts a skip on ten of its seventeen inverted residuals, and two things
 * follow that a feed-forward chain never asks for.
 *
 * THE ADD IS A CONVOLUTION. The DPU's elementwise stage takes exactly one operand, so
 * the add is lowered onto the datapath the part does have: the two operands concatenated
 * along channels, convolved with a 1x1 kernel of two diagonal blocks. The weights are
 * rocket_residual_add_weights_rk3576()'s — chip arithmetic, gated on its own in
 * tests/rk3576_residual_add.c — and everything else is an ordinary convolution, which is
 * why an add here gets a resident handle, a submit and a row-major output exactly like a
 * conv does.
 *
 * ITS SECOND OPERAND IS NOT THE PREVIOUS LAYER. It is produced three to five layers back,
 * and the ping-pong pair of buffers a feed-forward graph runs on has overwritten it by
 * then. A layer some later layer reads NON-ADJACENTLY therefore writes to a buffer of its
 * OWN, which nothing else reuses; the ping-pong is left to carry the main line. That costs
 * one allocation per skip and no copy — the layer writes there in the first place — and it
 * is why SKIP[] is indexed by the PRODUCING layer rather than by the consumer.
 *
 * THE TEST IS THE DISTANCE, NOT THE OPERAND. A residual add's second operand is one way to
 * read a distant layer and it is not the only one: a ResNet block that changes width reads
 * the block input twice, once as the 3x3 convolution's input and once THREE LAYERS LATER as
 * the 1x1 downsample's — a distant read on a convolution's FIRST operand. Keyed on "some
 * add names this as src2" that layer gets no buffer, the ping-pong overwrites it, and the
 * downsample convolution computes a full plausible surface from the wrong tensor. It shows
 * up only in the CHAIN pass, since the oracle hands every layer its golden input.
 *
 * TWO CONSEQUENCES FOR THE CUBE CHAIN, both asserted rather than assumed: a skip source
 * must MATERIALISE its output (a cube-out layer writes no row-major tensor, and the add
 * that reads it four layers later has nowhere to read from), and an add is never a cube
 * CONSUMER, because its input is a concatenation this gate builds on the host.
 * ==========================================================================*/
static int8_t **SKIP;            /* [n_layers], non-NULL where the layer feeds a skip */

/* A layer read by anything other than the layer immediately after it. The ping-pong pair
 * only ever holds the previous layer's output, so every other reader needs this one to
 * have written somewhere of its own. A SOFTMAX is skipped: the gate does not run it as a
 * layer, so its "read" is not one. */
static int skip_is_source(unsigned i)
{
    unsigned j;
    for (j = i + 1; j < H->n_layers; j++) {
        if (LAYERS[j].kind == KIND_SOFTMAX) continue;
        if (LAYERS[j].src2 == i && j != i + 1) return 1;
        if (LAYERS[j].src1 == i && j != i + 1) return 1;
    }
    return 0;
}

static int skip_alloc(void)
{
    unsigned i, n = 0;
    SKIP = calloc(H->n_layers, sizeof *SKIP);
    if (!SKIP) return -1;
    for (i = 0; i < H->n_layers; i++) {
        if (!skip_is_source(i)) continue;
        SKIP[i] = malloc((size_t)LAYERS[i].oc * LAYERS[i].oh * LAYERS[i].ow);
        if (!SKIP[i]) return -1;
        n++;
    }
    if (n) printf("   %u skip source(s) keep a buffer of their own — a residual's second "
                  "operand outlives the ping-pong\n", n);
    return 0;
}

static void skip_free(void)
{
    unsigned i;
    if (!SKIP) return;
    for (i = 0; i < H->n_layers; i++) free(SKIP[i]);
    free(SKIP);
    SKIP = NULL;
}

/* Operand B in the CHAIN passes, which is NOT always the distant one. MobileNetV2's adds
 * read a skip produced three to five layers back and the main line from the layer before;
 * a ResNet block that changes width reads them the OTHER WAY ROUND — operand B is the 1x1
 * downsample immediately before the add and operand A is the 3x3 leg past it. So neither
 * operand can be assumed to be the near one, and both resolve the same way: the producer's
 * own buffer when it has one, the ping-pong when it is the previous layer. */
static const int8_t *chain_operand_b(const rnet_layer *L, unsigned i, const int8_t *cur)
{
    if (L->kind != KIND_ADD || L->src2 == NO_SRC) return NULL;
    if (SKIP && SKIP[L->src2]) return SKIP[L->src2];
    return L->src2 + 1u == i ? cur : NULL;
}

/* Operand A, in the CHAIN passes. A feed-forward graph's is always the previous layer's
 * output and this one's is not: a ResNet block that changes width reads its input again
 * three layers later, once on the 3x3 convolution and once on the 1x1 downsample, and the
 * add after them reads the 3x3 leg past the downsample. Every producer that some later
 * layer reads non-adjacently holds a buffer of its own (skip_is_source), so operand A is a
 * LOOKUP by src1 rather than the threaded ping-pong. Threading it computes a full and
 * plausible surface from the wrong tensor, and only the chain pass can see that — the
 * oracle hands every layer its golden input. */
static const int8_t *chain_input(unsigned i, const int8_t *cur)
{
    unsigned s = LAYERS[i].src1;
    if (s != NO_SRC && s + 1u != i && SKIP && SKIP[s]) return SKIP[s];
    return cur;
}

/* Operand B, in whichever pass is asking: the golden tensor when a layer is being fed
 * TFLite's own input, the skip buffer when the network is being run for real. */
static const int8_t *skip_operand(const rnet_layer *L, int from_golden)
{
    if (L->kind != KIND_ADD || L->src2 == NO_SRC) return NULL;
    if (from_golden) return at(LAYERS[L->src2].g_off);
    return SKIP ? SKIP[L->src2] : NULL;
}

/* WHERE OPERAND B STARTS, and why it is not always channel `c`. In cube layout the two
 * operands are slices of one buffer and a slice starts every SIXTEEN channels — sixteen
 * of them share each atom, so channel 24 is not an address at all. Rounding operand B's
 * start up to a group makes the two layouts the same object: the same descriptor, the
 * same weight matrix and the same CPU reference whether the concatenation is built by the
 * host or by two producers writing their own slices. The channels in the gap carry zero
 * weights, so they contribute nothing at any content. */
static unsigned add_boff(const rnet_layer *L)
{
    return (L->oc + 15u) & ~15u;
}

/* The add's operands side by side, which is what the convolution contracts over. In cube
 * layout this copy does not exist — two producers writing their own slices of one buffer
 * already are this tensor — but the row-major path is the control it is measured against,
 * so here it is two memcpys and they are counted as this layer's own cost. */
static int8_t *add_concat(const rnet_layer *L, const int8_t *a, const int8_t *b)
{
    size_t n = (size_t)L->oc * L->oh * L->ow;
    size_t px = (size_t)L->oh * L->ow;
    size_t boff = (size_t)add_boff(L) * px;
    int8_t *buf = calloc(boff + n, 1);
    if (!buf) return NULL;
    memcpy(buf, a, n);
    memcpy(buf + boff, b, n);
    return buf;
}

/* The weights the lowering needs. Pure, and a function of the QUANTIZATION alone, so a
 * resident add builds them once at pack time. */
static int add_weights(const rnet_layer *L, int8_t **W, int32_t **bias, float *w_scale,
                       int *w_pair)
{
    unsigned c = L->oc, boff = add_boff(L), o;
    int8_t *narrow;
    int rc;
    *W = calloc((size_t)c * (boff + c), 1);
    *bias = malloc((size_t)c * sizeof **bias);
    narrow = malloc((size_t)c * 2u * c);
    if (!*W || !*bias || !narrow) {
        free(*W); free(*bias); free(narrow);
        *W = NULL; *bias = NULL; return -1;
    }
    rc = rocket_residual_add_weights_rk3576(c, L->in_scale, L->in2_scale,
                                            L->in_zp, L->in2_zp, narrow, *bias,
                                            w_scale, w_pair);
    if (rc != ROCKET_OK) {
        free(*W); free(*bias); free(narrow);
        *W = NULL; *bias = NULL; return rc;
    }
    /* The library's matrix is `c x 2c`; this one puts operand B's block at the group
     * boundary the cube layout can address. One arithmetic either way — the two diagonal
     * blocks are moved, not recomputed. */
    for (o = 0; o < c; o++) {
        memcpy(*W + (size_t)o * (boff + c), narrow + (size_t)o * 2u * c, c);
        memcpy(*W + (size_t)o * (boff + c) + boff, narrow + (size_t)o * 2u * c + c, c);
    }
    free(narrow);
    return 0;
}

/* The add's descriptor: a 1x1 convolution over the concatenation. `direct_datapath`
 * because a narrow residual (24 channels makes a 48-channel contraction) would otherwise
 * be routed to the packed-image first conv, which is a different program. */
static void add_desc(const rnet_layer *L, rocket_conv2d_desc *d)
{
    memset(d, 0, sizeof *d);
    d->ic = (int)(add_boff(L) + L->oc); d->ih = (int)L->oh; d->iw = (int)L->ow;
    d->oc = (int)L->oc;
    d->kh = d->kw = 1;
    d->stride_y = d->stride_x = 1;
    d->dil_y = d->dil_x = 1;
    d->direct_datapath = 1;
}

/* The same lowering as a LAYER, so the CPU reference for an add is the CPU reference for
 * a convolution — one arithmetic rather than two to keep in agreement. The weight zero
 * point is zero (the diagonal blocks are the weights, not a quantized tensor) and the
 * input zero point is operand A's; operand B's rides in the bias. */
static void add_ref_layer(const rnet_layer *L, float w_scale, rnet_layer *R)
{
    *R = *L;
    R->kind = KIND_CONV;
    R->ic = add_boff(L) + L->oc; R->ih = L->oh; R->iw = L->ow;
    R->kh = R->kw = 1; R->sy = R->sx = 1;
    R->pl_y = R->pl_x = R->pt_y = R->pt_x = 0;
    R->w_zp = 0;
    R->w_scale = w_scale;
}

/* How far from TFLite a correct layer may land, in counts, and WHY each number is what
 * it is rather than a tolerance someone widened until the gate passed.
 *
 * A CONVOLUTION owes ONE. Both sides apply the same real-valued scale through different
 * fixed-point roundings — this OUT_CVT has a 15-bit multiplier and rounds ties to even,
 * TFLite has 31 bits and rounds half away from zero — so the elements near a boundary
 * disagree by one and nothing disagrees by two.
 *
 * AN ADD OWES TWO, and the second one is nameable. The datapath applies a single requant
 * gain, so the operands' different scales ride in the weights as `w2/w1` with both terms
 * int8. The best rational approximation with a denominator under 128 is within about
 * 5e-4 relative on this model's ten skips, and the second operand contributes at most a
 * full int8 range, so the ratio can move an output by a small fraction of a count — which
 * lands on the far side of a rounding boundary for some elements and not others. One
 * count for the requant, one for that. A third would be a defect. */
static int tflite_slack(const rnet_layer *L)
{
    return L->kind == KIND_ADD ? 2 : 1;
}

/*
 * RESIDENT WEIGHTS, one handle per convolution layer.
 *
 * The three things a conv rebuilds from the weights alone — the filter sums, the
 * coefficient group and the weight cube — are ~25 ms of this graph's 115, and a graph runs
 * the same weights every inference. Packed lazily on a layer's first call and held for the
 * whole run.
 *
 * ROCKET_RK3576_NET_RESIDENT=1 turns it on, so the same binary measures both and the
 * per-layer assertion runs either way: a resident layer must be as bit-exact against the
 * part's own arithmetic as a transient one, which is the check that matters — a wall-clock
 * win on a wrong answer is not a win.
 */
static rocket_conv2d_int8_weights_rk3576 **RESIDENT;
static int RESIDENT_ON;

/* The pooling layer's own resident handle. Separate from RESIDENT[] because a pool is a
 * different program (PC_OPERATION_ENABLE 0x60 against a convolution's 0x1D) and a
 * different handle type — the graph has one pooling layer, so one slot. */
static rocket_pool_int8_rk3576_handle **RESIDENT_POOL;

/* The cube chain's state; see cube_link() below for what it is. */
static int CUBE_ON;
static int *CUBE_IN, *CUBE_OUT;   /* per layer */
static int CUBE_JOINS;

/* THE CONCATENATION BUFFERS. An add's two operands, side by side in one allocation that
 * its two producers write directly — see cat_link() for the whole of it. CAT[j] is the
 * buffer add `j` reads; CAT_OF[i] is non-zero where layer i writes a slice of one, which
 * is what says its output is placed rather than owned. */
static rocket_rk3576_cube *CAT;
static int *CAT_OF;
static int CAT_ON, CAT_ADDS;

/* Whether a cube-out producer's readers are all wired here rather than in the pair loop.
 * Indexed by the PRODUCER, like SKIP[] and CAT_OF[]. */
static int *MULTI_OF;
static int MULTI_ON, MULTI_SRCS;

/*
 * THE CROSS-LAYER KICK. A run of consecutive cube-linked layers has no host work between
 * its members at all, and a chained regcmd stream honours read-after-write between its
 * programs — so the whole run can go out as ONE hardware kick instead of one per layer.
 * rocket_conv2d_int8_chain_new_rk3576() owns that; here it is only a matter of finding the
 * maximal runs and letting the layer loop skip over one.
 *
 * KICK_OF[i] is the chain a run STARTING at layer i, and KICK_END[i] the layer after it, so
 * the loop advances past the layers the kick covered. It is built after cube_link(), because
 * the links are what make a run eligible and the constructor refuses a pair that has none.
 *
 * ROCKET_RK3576_NET_KICK=0 turns it off, which is the A/B: the same graph one submit per
 * layer must give byte-identical results.
 */
static rocket_conv2d_int8_chain_rk3576 **KICK_OF;
static unsigned *KICK_END;
static int KICK_RUNS, KICK_LAYERS;

static int kick_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_NET_KICK");
        cached = !e || !*e || *e != '0';
    }
    return cached;
}

/* Build ONLY the run starting at this layer, leaving every other layer one submit each.
 * The localizer: the byte comparison can only see MATERIALISED layers, so a wrong byte
 * anywhere in a chain first shows up at the next one — with a single run in play, a diff
 * there is a diff caused by THAT run. -1 (the default) builds them all.
 * ROCKET_RK3576_NET_KICK_AT=<i>. */
static int kick_at(void)
{
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("ROCKET_RK3576_NET_KICK_AT");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) : -1;
    }
    return cached;
}

/* The longest run a kick may cover. Not a hardware bound — it is the instrument for
 * bisecting one: a chain that computes at 2 layers and not at 24 is a length effect, and a
 * chain that fails at every length is not. ROCKET_RK3576_NET_KICK_MAX=<n>. */
static unsigned kick_max(void)
{
    static unsigned cached = 0;
    if (!cached) {
        const char *e = getenv("ROCKET_RK3576_NET_KICK_MAX");
        long v = (e && *e) ? strtol(e, NULL, 0) : 0;
        cached = (v >= 2 && v < 64) ? (unsigned)v : 64u;
    }
    return cached;
}

static void resident_init(void)
{
    const char *e = getenv("ROCKET_RK3576_NET_RESIDENT");
    RESIDENT_ON = e && *e && *e != '0';
    if (RESIDENT_ON) RESIDENT = calloc(H->n_layers, sizeof *RESIDENT);
    if (RESIDENT_ON) RESIDENT_POOL = calloc(H->n_layers, sizeof *RESIDENT_POOL);
    if (RESIDENT_ON && (!RESIDENT || !RESIDENT_POOL)) RESIDENT_ON = 0;

    e = getenv("ROCKET_RK3576_NET_CUBE");
    CUBE_ON = RESIDENT_ON && e && *e && *e != '0';
    if (CUBE_ON) {
        CUBE_IN = calloc(H->n_layers, sizeof *CUBE_IN);
        CUBE_OUT = calloc(H->n_layers, sizeof *CUBE_OUT);
        KICK_OF = calloc(H->n_layers, sizeof *KICK_OF);
        KICK_END = calloc(H->n_layers, sizeof *KICK_END);
        if (!CUBE_IN || !CUBE_OUT || !KICK_OF || !KICK_END) CUBE_ON = 0;
    }
    /* The concatenation buffers ride on the cube chain and are its own A/B:
     * ROCKET_RK3576_NET_CAT=0 leaves every add reading a host-built tensor, which is what
     * the joins they buy are measured against. */
    if (CUBE_ON) {
        e = getenv("ROCKET_RK3576_NET_CAT");
        CAT_ON = !e || !*e || *e != '0';
        CAT = calloc(H->n_layers, sizeof *CAT);
        CAT_OF = calloc(H->n_layers, sizeof *CAT_OF);
        if (!CAT || !CAT_OF) CAT_ON = 0;
        /* The shared surfaces are their own A/B for the same reason: they close a whole
         * refusal bucket and the joins they buy are measured against leaving it open. */
        e = getenv("ROCKET_RK3576_NET_MULTI");
        MULTI_ON = !e || !*e || *e != '0';
        MULTI_OF = calloc(H->n_layers, sizeof *MULTI_OF);
        if (!MULTI_OF) MULTI_ON = 0;
    }
}

static void resident_free(int fd)
{
    unsigned i;
    /* The chains BORROW the handles, so they go first — and before the RESIDENT check, since
     * they are allocated whether or not a handle was ever packed. */
    if (KICK_OF)
        for (i = 0; i < H->n_layers; i++)
            if (KICK_OF[i]) rocket_conv2d_int8_chain_free_rk3576(fd, KICK_OF[i]);
    free(KICK_OF); free(KICK_END);
    KICK_OF = NULL; KICK_END = NULL;
    KICK_RUNS = KICK_LAYERS = 0;
    if (RESIDENT_POOL) {
        for (i = 0; i < H->n_layers; i++)
            if (RESIDENT_POOL[i]) rocket_pool_int8_free_rk3576(fd, RESIDENT_POOL[i]);
        free(RESIDENT_POOL);
        RESIDENT_POOL = NULL;
    }
    if (!RESIDENT) return;
    for (i = 0; i < H->n_layers; i++)
        if (RESIDENT[i]) rocket_conv2d_int8_weights_free_rk3576(fd, RESIDENT[i]);
    free(RESIDENT);
    RESIDENT = NULL;
    free(CUBE_IN); free(CUBE_OUT);
    CUBE_IN = CUBE_OUT = NULL;
    CUBE_ON = 0;
    if (CAT) {
        for (i = 0; i < H->n_layers; i++) rocket_rk3576_cube_free(fd, &CAT[i]);
        free(CAT);
        CAT = NULL;
    }
    free(CAT_OF);
    CAT_OF = NULL;
    CAT_ON = 0;
    free(MULTI_OF);
    MULTI_OF = NULL;
    MULTI_ON = MULTI_SRCS = 0;
}

/*
 * THE CUBE CHAIN.
 *
 * The entries take and return row-major tensors, so a graph pays the CHW <-> NC1HWC2
 * transpose at both ends of every layer — the two largest host buckets left. It does not
 * have to pay them BETWEEN layers: a direct conv's output surface stride is `ow*oh`
 * exactly, so layer n's surface IS layer n+1's feature cube byte for byte whenever the
 * plane and the channel rounding agree. Both flags live on the resident handle.
 *
 * The linking is a PRE-PASS over already-packed handles rather than a decision taken as
 * the graph runs, because the producer's "leave it in the cube" and the consumer's "read
 * it from there" have to be set as a PAIR: a producer that skipped its de-scatter for a
 * consumer that then refused the cube would have left no row-major tensor at all.
 *
 * ROCKET_RK3576_NET_CUBE=1, and only with the resident weights the flags live on.
 */
/* Whether layer j can read its input from a producer's surface. Every clause here is a
 * property of the GATE's lowering rather than of the hardware, so it is asked here and the
 * hardware's own bounds are left to the library's refusal:
 *
 *   a materialised border — prep_input() writes the pad into a row-major buffer, so the
 *   tensor the entry sees is not the producer's output at all;
 *   the packed-image stem, which owns a different cube;
 *   a layer that is not a resident convolution (the pooling entry and the host softmax
 *   both take row-major). */
static int cube_consumer_ok(unsigned j)
{
    const rnet_layer *L = &LAYERS[j];
    struct stem_plan sp;
    /* The POOLING layer is a consumer too, and it is the same join: the PPU reads the
     * same NC1HWC2 cube the convolution path packs. It is not a chain member — a pool is
     * a different program in a BO of its own — so this only removes its host scatter. */
    if (IS_POOL(L->kind)) return RESIDENT_POOL && RESIDENT_POOL[j] != NULL;
    /* An ADD reads a CONCATENATION this gate builds on the host out of two tensors from
     * different places, so there is no single producer surface for it to read. It is a
     * cube producer like any other convolution; it is never a consumer. */
    if (L->kind == KIND_ADD) return 0;
    if (L->kind != KIND_CONV && L->kind != KIND_DWCONV) return 0;
    if (needs_prep(L)) return 0;
    if (stem_plan_of(L, &sp)) return 0;
    return RESIDENT[j] != NULL;
}

/* WHY A JOIN WAS REFUSED, counted. On a feed-forward chain almost every adjacent pair
 * links and the breakdown is not interesting; on a residual one it is the whole answer to
 * how much of the graph can go out as one kick, so the reasons are separated rather than
 * left as "32 of 63". Each bucket is a DIFFERENT thing to fix, and two of them are
 * properties of this network rather than of the part. */
enum { NJ_ADD_CONSUMER = 0, NJ_SKIP_SOURCE, NJ_IC_ALIGN, NJ_SURFACE, NJ_OTHER,
       NJ_FORCED, NJ_N };
static int NOJOIN[NJ_N];
static const char *NOJOIN_WHY[NJ_N] = {
    "the consumer is an ADD, whose input is a host-built concatenation",
    "the producer feeds a SKIP and so must leave a row-major tensor",
    "the consumer's input channel count is not a multiple of 32",
    "the producer's output surface is not a feature cube (padded stride, or tiled)",
    "the consumer is not a resident convolution",
    "forced off by ROCKET_RK3576_NET_NOJOIN (the per-pair A/B)",
};
static const char *NOJOIN_TAG[NJ_N] = {
    "add-consumer", "skip-source", "ic-align", "surface", "not-resident", "forced",
};

/* AND WHAT EACH BUCKET IS WORTH, which the COUNT does not say. What a join removes is the
 * two transposes at it, and those are BYTES — a 56x56 pair moves 64x more than a 7x7 one,
 * so twelve refusals at the deep end and twelve at the shallow end are different numbers.
 * A count prices a submit; only the bytes price the transposes, and on this graph the
 * transposes are the larger term by about 6x. */
static size_t NOJOIN_BYTES[NJ_N];
static size_t JOIN_BYTES;

/* The tensor that crosses one adjacent pair: the producer's whole output surface, which
 * is de-scattered at the producer and scattered again at the consumer. */
static size_t pair_bytes(unsigned i)
{
    return (size_t)LAYERS[i].oc * LAYERS[i].oh * LAYERS[i].ow;
}

/* THE PRODUCER SIDE, for either kind of handle. A POOL is a cube producer like any
 * convolution — the PPU writes the same 16-byte-atom surface, at round4(ow*oh) rather than
 * the plane, which is the consumer's DDR channel-group jump and a register. Keeping the
 * two behind one set of calls is what lets a pool sit on either side of a join instead of
 * only on the consumer side, and ResNet-18's max pool is the case: it feeds the next
 * convolution AND a residual add three layers on. */
static int prod_ok(unsigned i)
{
    if (IS_POOL(LAYERS[i].kind)) return RESIDENT_POOL && RESIDENT_POOL[i] != NULL;
    return RESIDENT && RESIDENT[i] != NULL;
}

static int prod_cube_of(unsigned i, rocket_rk3576_cube *c)
{
    return IS_POOL(LAYERS[i].kind)
             ? rocket_pool_int8_cube_of_rk3576(RESIDENT_POOL[i], c)
             : rocket_conv2d_int8_cube_of_rk3576(RESIDENT[i], c);
}

static int prod_cube_out(unsigned i, int on)
{
    return IS_POOL(LAYERS[i].kind)
             ? rocket_pool_int8_cube_out_rk3576(RESIDENT_POOL[i], on)
             : rocket_conv2d_int8_cube_out_rk3576(RESIDENT[i], on);
}

static int prod_cube_out_at(unsigned i, const rocket_rk3576_cube *dst)
{
    return IS_POOL(LAYERS[i].kind)
             ? rocket_pool_int8_cube_out_at_rk3576(RESIDENT_POOL[i], dst)
             : rocket_conv2d_int8_cube_out_at_rk3576(RESIDENT[i], dst);
}

/* A pool is never a chain member — a pooling program lives in a BO of its own — so nothing
 * re-stamps its surface behind its back and it has no declaration to make. */
static void prod_cube_shared(unsigned i, int on)
{
    if (!IS_POOL(LAYERS[i].kind))
        rocket_conv2d_int8_cube_shared_rk3576(RESIDENT[i], on);
}

static int cons_cube_in(unsigned j, const rocket_rk3576_cube *c)
{
    return IS_POOL(LAYERS[j].kind)
             ? rocket_pool_int8_cube_in_rk3576(RESIDENT_POOL[j], c)
             : rocket_conv2d_int8_cube_in_rk3576(RESIDENT[j], c);
}

/* The layer that follows i in execution order — a SOFTMAX is not run as a layer here, so
 * it is stepped over. H->n_layers when there is none. */
static unsigned next_layer(unsigned i)
{
    unsigned j = i + 1u;
    while (j < H->n_layers && LAYERS[j].kind == KIND_SOFTMAX) j++;
    return j;
}

/* ONE WIRED PRODUCER -> READER RELATION, counted. The headline is "joins of ADJACENT
 * pairs", because that is the unit the refusal buckets are counted in and the unit every
 * earlier measurement is quoted in — so a relation whose reader is not the producer's
 * immediate successor is real work removed and is counted apart rather than folded in.
 * ResNet-18's downsample adds are all of that kind: their distant operand is the 3x3 leg,
 * one layer further back than the 1x1 the add sits behind. */
static int CUBE_FAR;
static size_t CUBE_FAR_BYTES;

static void cube_joined(unsigned p, unsigned r)
{
    if (next_layer(p) == r) { CUBE_JOINS++; JOIN_BYTES += pair_bytes(p); }
    else { CUBE_FAR++; CUBE_FAR_BYTES += pair_bytes(p); }
}

static void cube_unjoined(unsigned p, unsigned r)
{
    if (next_layer(p) == r) { CUBE_JOINS--; JOIN_BYTES -= pair_bytes(p); }
    else { CUBE_FAR--; CUBE_FAR_BYTES -= pair_bytes(p); }
}

/* The refused pairs, NAMED. A bucket's count prices a submit and its bytes price the
 * transposes, but neither prices a PAIR: the two classes already closed measure 8x apart
 * per join, so the pairs a session proposes to close have to be listed one at a time with
 * the tensor each of them carries. Printed for the refusals only — a joined pair has
 * nothing left to decide, and its price is the A/B below. */
#define NJ_LIST_MAX 64
static struct { unsigned i, j; size_t bytes; int why; } NJ_LIST[NJ_LIST_MAX];
static int NJ_LIST_N;

static void nojoin_note(unsigned i, unsigned j, int why)
{
    size_t bytes = pair_bytes(i);
    NOJOIN[why]++;
    NOJOIN_BYTES[why] += bytes;
    if (NJ_LIST_N < NJ_LIST_MAX) {
        NJ_LIST[NJ_LIST_N].i = i; NJ_LIST[NJ_LIST_N].j = j;
        NJ_LIST[NJ_LIST_N].bytes = bytes; NJ_LIST[NJ_LIST_N].why = why;
        NJ_LIST_N++;
    }
}

/* THE PER-PAIR A/B. What a join is WORTH is measured, not derived from its bytes: the
 * measurement is the finite difference between the graph as it stands and the same graph
 * with exactly one join refused. ROCKET_RK3576_NET_NOJOIN=<i> refuses the join whose
 * PRODUCER is layer i and leaves every other join intact. A concatenation-wired add is
 * wired as a PAIR, so naming either of its producers takes both of that add's joins off —
 * which is what its price is, since neither half stands alone. */
static int nojoin_at(void)
{
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("ROCKET_RK3576_NET_NOJOIN");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) : -2;
    }
    return cached;
}

static int nojoin_forced(unsigned i) { return nojoin_at() == (int)i; }

/* THE LOCALIZER for a join whose consumer's channel count is not a multiple of 32 at a
 * non-zero weight zero point. That join is sound only because a direct producer's partial
 * output group carries its output zero point, so it is the one class of join whose
 * correctness rests on a property of the PRODUCER — and a graph that disagrees needs to be
 * able to enable them one at a time. ROCKET_RK3576_NET_PADJOIN=-1 refuses them all (the
 * A/B), =<i> allows only the pair whose producer is layer i, unset allows every one. */
static int padjoin_at(void)
{
    static int cached = -2;
    if (cached == -2) {
        const char *e = getenv("ROCKET_RK3576_NET_PADJOIN");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) : -2;
    }
    return cached;
}

static int padjoin_ok(unsigned i, unsigned j)
{
    if (LAYERS[j].ic % 32u == 0 || !LAYERS[j].w_zp) return 1;   /* not that class */
    if (padjoin_at() == -2) return 1;
    return padjoin_at() == (int)i;
}

/* Every layer that reads layer i's output. A SOFTMAX is not run as a layer here, so its
 * read is not one. Returns the count and fills `out` up to `max`, or `max + 1` when there
 * are more than that — a caller may not silently wire a subset of a producer's readers. */
static unsigned consumers_of(unsigned i, unsigned *out, unsigned max)
{
    unsigned j, n = 0;
    for (j = i + 1; j < H->n_layers; j++) {
        if (LAYERS[j].kind == KIND_SOFTMAX) continue;
        if (LAYERS[j].src1 != i && LAYERS[j].src2 != i) continue;
        if (n < max) out[n] = j;
        n++;
        if (n > max) return max + 1u;
    }
    return n;
}

/*
 * THE CONCATENATION BUFFERS — what the two big refusal buckets actually wanted.
 *
 * Twenty of MobileNetV2's thirty-one refused joins are one shape of problem: an add reads
 * a CONCATENATION of two tensors from different places, and the layer that produces its
 * second operand therefore has to leave a row-major tensor for it to be copied out of,
 * three to five layers later. Neither is a property of the part. Both are the host
 * building a tensor that the hardware could have been asked to write in the first place.
 *
 * A cube's base is a plain address on both sides of a convolution, so it can be: allocate
 * ONE buffer per add, give operand A the low slice and operand B the high one, and point
 * the two producers at them with rocket_conv2d_int8_cube_out_at_rk3576(). The add then
 * reads the whole buffer as its feature cube and no concatenation is built at all.
 *
 * WHAT MAKES IT A PAIR, and why it is refused as one. Operand B's producer is also the
 * layer BEFORE the block's expand convolution, so making it write a slice means that
 * layer has to read the same slice as its cube — a producer that leaves no row-major
 * tensor for a consumer that cannot take a cube would leave the graph with nothing at
 * all. So an add is wired only when BOTH producers can be placed, and every OTHER reader
 * of either producer can read the slice back.
 *
 * NEITHER OPERAND IS "THE LAYER BEFORE", AND NEITHER IS THE PLACEMENT RULE. MobileNetV2's
 * adds take operand A from the layer before and operand B from a skip three to five layers
 * back; a ResNet block that changes width takes them the OTHER WAY ROUND, operand B from
 * the 1x1 downsample immediately before the add and operand A from the 3x3 leg past it. A
 * pass keyed on "operand A's producer is the previous layer" wires the first family and
 * refuses the second — which is what left ResNet-18's three downsample adds reading a host
 * concatenation. The buffer is a dedicated allocation per add and each slice has exactly
 * one writer, so the ORDER the two producers run in does not matter at all; what has to
 * hold is that both precede the add, which they do by construction.
 *
 * The gap between operand A's channels and the group boundary operand B starts on carries
 * zero weights (add_boff), so it contributes nothing at any content.
 */
static void cat_link(int fd)
{
    unsigned j;
    if (!CUBE_ON || !CAT_ON) return;
    for (j = 0; j < H->n_layers; j++) {
        const rnet_layer *L = &LAYERS[j];
        rocket_rk3576_cube lo, hi, all;
        rocket_rk3576_cube pc[2];
        unsigned a, b, boff, chans, p, ok = 1;
        unsigned rd[2][4], nrd[2];
        if (L->kind != KIND_ADD || L->src2 == NO_SRC || L->src1 == NO_SRC) continue;
        a = L->src1; b = L->src2;
        /* ONE of the two producers is the layer immediately before the add — either one.
         * Without that the add has no adjacent producer at all and the layer between them
         * would be reading and writing the buffer this pass is placing. */
        if (a + 1u != j && b + 1u != j) continue;
        /* The per-pair A/B: naming either producer refuses the whole wiring, and the two
         * joins it would have made fall to the buckets below like any other refusal. */
        if (nojoin_forced(a) || nojoin_forced(b)) continue;
        if (!prod_ok(a) || !prod_ok(b) || !RESIDENT[j]) continue;
        /* EVERY OTHER READER OF EITHER PRODUCER has to be able to read the slice: a placed
         * producer leaves no row-major tensor, so a reader that cannot take a cube would be
         * left with nothing. MobileNetV2's operand B has one such reader (the next block's
         * expand convolution); ResNet-18's downsample operands have none. */
        for (p = 0; p < 2 && ok; p++) {
            unsigned q, src = p ? b : a, n;
            n = consumers_of(src, rd[p], 4u);
            if (n > 4u) { ok = 0; break; }
            nrd[p] = n;
            for (q = 0; q < n; q++)
                if (rd[p][q] != j && !cube_consumer_ok(rd[p][q])) ok = 0;
        }
        if (!ok) continue;
        /* Neither producer may be a layer this gate prepares on the host, and the add
         * itself has to be a plain resident convolution. */
        if (needs_prep(&LAYERS[a]) || needs_prep(&LAYERS[b])) continue;
        boff = add_boff(L);
        /* The buffer is as deep as the add's feature DMA WALKS, which is the register
         * count rounded to the 32-channel MAC group — not the live channels. */
        chans = (boff + L->oc + 31u) & ~31u;
        if (rocket_rk3576_cube_alloc(fd, chans, L->oh, L->ow, &CAT[j]) != ROCKET_OK)
            continue;
        /* The add reads `boff + oc` channels, which is what its descriptor says and not
         * the whole allocation — the groups past it are the 32-channel rounding the
         * feature DMA walks and the cube's own size check covers. */
        if (rocket_rk3576_cube_slice(&CAT[j], 0u, boff + L->oc, &all) != ROCKET_OK ||
            rocket_rk3576_cube_slice(&CAT[j], 0u, L->oc, &lo) != ROCKET_OK ||
            rocket_rk3576_cube_slice(&CAT[j], boff, L->oc, &hi) != ROCKET_OK ||
            prod_cube_out_at(a, &lo) != ROCKET_OK ||
            prod_cube_out_at(b, &hi) != ROCKET_OK ||
            rocket_conv2d_int8_cube_in_rk3576(RESIDENT[j], &all) != ROCKET_OK) {
            prod_cube_out_at(a, NULL);
            prod_cube_out_at(b, NULL);
            rocket_conv2d_int8_cube_in_rk3576(RESIDENT[j], NULL);
            rocket_rk3576_cube_free(fd, &CAT[j]);
            continue;
        }
        /* THE OTHER READERS ARE ASKED FOR THE PRODUCER'S CUBE, NOT FOR THE SLICE THIS PASS
         * BUILT. The two name the same bytes, but only the producer can say what its tail
         * holds — a partial output group lands on its output zero point — and a consumer
         * whose own channel count is not a multiple of 32 needs exactly that to be stated. */
        if (prod_cube_of(a, &pc[0]) != ROCKET_OK ||
            prod_cube_of(b, &pc[1]) != ROCKET_OK) ok = 0;
        for (p = 0; p < 2 && ok; p++) {
            unsigned q;
            for (q = 0; q < nrd[p]; q++) {
                unsigned r = rd[p][q];
                if (r == j) continue;
                if (cons_cube_in(r, &pc[p]) != ROCKET_OK) { ok = 0; break; }
                CUBE_IN[r] = 1;
                cube_joined(p ? b : a, r);
            }
        }
        if (!ok) {
            for (p = 0; p < 2; p++) {
                unsigned q;
                for (q = 0; q < nrd[p]; q++) {
                    unsigned r = rd[p][q];
                    if (r == j || !CUBE_IN[r]) continue;
                    cons_cube_in(r, NULL);
                    CUBE_IN[r] = 0;
                    cube_unjoined(p ? b : a, r);
                }
            }
            prod_cube_out_at(a, NULL);
            prod_cube_out_at(b, NULL);
            rocket_conv2d_int8_cube_in_rk3576(RESIDENT[j], NULL);
            rocket_rk3576_cube_free(fd, &CAT[j]);
            continue;
        }
        CUBE_OUT[a] = 1; CUBE_OUT[b] = 1;
        CUBE_IN[j] = 1;
        CAT_OF[a] = 1; CAT_OF[b] = 1;
        /* Counted HERE or not at all: an adjacent pair the loop below then skips would
         * otherwise leave the joined-bytes total understating the denominator every
         * refusal ratio is quoted against. */
        cube_joined(a, j); cube_joined(b, j);
        CAT_ADDS++;
    }
    if (CAT_ADDS)
        printf("   concatenation buffers: %d of the adds read a buffer their two producers "
               "wrote directly — no host concatenation, and the skip source no longer has "
               "to materialise\n", CAT_ADDS);
}

/*
 * A SKIP SOURCE MAY WRITE A CUBE AFTER ALL — when every layer that reads it can read one.
 *
 * "A producer some later layer reads must materialise" is a statement about the HOST
 * buffers, not about the part: it holds because the reader three layers on wants a
 * row-major tensor. A resident handle owns its output surface and nothing else writes
 * there until that handle runs again, so the surface outlives the whole inference — which
 * is exactly the lifetime a distant reader needs. Two readers of one surface is not a
 * second copy of anything; it is the same cube described twice.
 *
 * That closes the class this graph loses the most to. ResNet-18's identity blocks end in
 * an add whose output is read TWICE — by the next block's 3x3 convolution and, three
 * layers later, by that block's 1x1 downsample — and both are ordinary convolutions.
 *
 * WHAT IT COSTS THE CHAIN, and why the library has to be told. A cross-layer kick
 * re-stamps every interior layer's surface in its verify bracket, sound only because the
 * next kick rewrites it before anything reads it. A shared surface is read first, so the
 * stamp would replace the layer's output with 0xA5 and the outside reader would compute a
 * full and plausible surface from it — invisible until the next materialised layer.
 * rocket_conv2d_int8_cube_shared_rk3576() moves that stamp to the start of the producer's
 * next call, one PREP/FINI pair and nothing else.
 *
 * ROCKET_RK3576_NET_MULTI=0 refuses them all, which is the A/B.
 */
static void multi_link(void)
{
    unsigned i;
    if (!CUBE_ON || !MULTI_ON) return;
    for (i = 0; i < H->n_layers; i++) {
        rocket_rk3576_cube c;
        unsigned rd[4], n, q, ok = 1;
        if (!SKIP[i] || CAT_OF[i] || CUBE_OUT[i]) continue;
        if (!prod_ok(i) || nojoin_forced(i)) continue;
        n = consumers_of(i, rd, 4u);
        if (!n || n > 4u) continue;
        for (q = 0; q < n; q++)
            if (!cube_consumer_ok(rd[q]) || CUBE_IN[rd[q]] || !padjoin_ok(i, rd[q])) ok = 0;
        if (!ok) continue;
        if (prod_cube_of(i, &c) != ROCKET_OK) continue;
        for (q = 0; q < n && ok; q++)
            if (cons_cube_in(rd[q], &c) != ROCKET_OK) ok = 0;
        if (ok && prod_cube_out(i, 1) != ROCKET_OK) ok = 0;
        if (!ok) {
            for (q = 0; q < n; q++) cons_cube_in(rd[q], NULL);
            continue;
        }
        prod_cube_shared(i, 1);
        CUBE_OUT[i] = 1; MULTI_OF[i] = 1;
        for (q = 0; q < n; q++) { CUBE_IN[rd[q]] = 1; cube_joined(i, rd[q]); }
        MULTI_SRCS++;
    }
    if (MULTI_SRCS)
        printf("   shared surfaces: %d skip source(s) leave a cube that every one of their "
               "readers takes, so they no longer materialise\n", MULTI_SRCS);
}

/* Link every adjacent pair the library accepts. Called once, after a warm-up inference has
 * packed the handles. */
static void cube_link(int fd)
{
    unsigned i, pairs = 0;
    if (!CUBE_ON) return;
    /* The concatenation buffers first: they decide two of the refusal buckets below, and
     * a pair they wire is a join this pass must not undo. The shared surfaces then take
     * what is left of the skip sources — placement first, because a placed producer's
     * slice is cheaper than its own surface (one buffer for the add instead of two). */
    cat_link(fd);
    multi_link();
    for (i = 0; i + 1 < H->n_layers; i++) {
        rocket_rk3576_cube c;
        unsigned j = i + 1;
        while (j < H->n_layers && LAYERS[j].kind == KIND_SOFTMAX) j++;
        if (j >= H->n_layers) break;
        pairs++;
        /* Already wired as a concatenation: layer i writes a slice and layer j reads it.
         * Counted there, and nothing left to do here. */
        if (CUBE_OUT[i] && CUBE_IN[j]) continue;
        if (nojoin_forced(i)) { nojoin_note(i, j, NJ_FORCED); continue; }
        if (!prod_ok(i)) { nojoin_note(i, j, NJ_OTHER); continue; }
        if (LAYERS[j].kind == KIND_ADD) { nojoin_note(i, j, NJ_ADD_CONSUMER); continue; }
        if (!cube_consumer_ok(j)) { nojoin_note(i, j, NJ_OTHER); continue; }
        /* A SKIP SOURCE MUST MATERIALISE unless it was PLACED. A cube-out layer writes no
         * row-major tensor, and the add that names it as an operand runs three to five
         * layers later with host work in between — so unless that add reads the slice this
         * layer wrote, there would be nothing for it to read. */
        if (SKIP[i] && !CAT_OF[i] && !MULTI_OF[i]) {
            nojoin_note(i, j, NJ_SKIP_SOURCE); continue; }
        if (CUBE_OUT[i] || CUBE_IN[j]) { nojoin_note(i, j, NJ_OTHER); continue; }
        if (!padjoin_ok(i, j)) { nojoin_note(i, j, NJ_IC_ALIGN); continue; }
        if (prod_cube_of(i, &c) != ROCKET_OK) {
            nojoin_note(i, j, NJ_SURFACE); continue;
        }
        if (cons_cube_in(j, &c) != ROCKET_OK) {
            /* The library refuses a cube whose channel count is not a multiple of 32 —
             * the consumer's own cube relies on the zero it memset into the padding
             * channels, which a producer does not control. Every other reason it can
             * refuse (plane, stride, fd) is impossible for an adjacent pair here. */
            nojoin_note(i, j, LAYERS[j].ic % 32u ? NJ_IC_ALIGN : NJ_OTHER);
            continue;
        }
        if (prod_cube_out(i, 1) != ROCKET_OK) {
            /* The consumer took the cube and the producer will not leave one, so the pair
             * has to come apart again — the consumer's row-major input is the only thing
             * that still exists. */
            cons_cube_in(j, NULL);
            nojoin_note(i, j, NJ_SURFACE);
            continue;
        }
        CUBE_OUT[i] = 1; CUBE_IN[j] = 1;
        cube_joined(i, j);
    }
    printf("   cube chain: %d join(s) of %u adjacent pair(s) — layer n's output surface is "
           "layer n+1's feature cube, so neither transpose runs at those joins\n",
           CUBE_JOINS, pairs);
    if (CUBE_FAR)
        printf("      and %d NON-adjacent link(s), %.0f KiB: a reader further on than the "
               "next layer takes the same cube, which is real work removed but not an "
               "adjacent pair\n", CUBE_FAR, CUBE_FAR_BYTES / 1024.0);
    {
        size_t refused = 0;
        int k;
        for (k = 0; k < NJ_N; k++) refused += NOJOIN_BYTES[k];
        for (k = 0; k < NJ_N; k++)
            if (NOJOIN[k])
                printf("      %2d refused, %5.0f KiB: %s\n", NOJOIN[k],
                       NOJOIN_BYTES[k] / 1024.0, NOJOIN_WHY[k]);
        /* THE REFUSED PAIRS, ONE LINE EACH. The bytes ratio above is an ordering, NOT a
         * cap: scaling the existing joins' measured value by it puts more milliseconds on
         * the refusals than the whole wall holds. What prices a pair is the finite
         * difference with that one join forced off — ROCKET_RK3576_NET_NOJOIN=<producer>,
         * over the joins that EXIST — read against these tensor sizes. */
        if (refused) {
            printf("      the joins that exist carry %.0f KiB and the refused ones %.0f "
                   "KiB (%.2fx) — an ORDERING, not a cap: price a pair with "
                   "ROCKET_RK3576_NET_NOJOIN\n",
                   JOIN_BYTES / 1024.0, refused / 1024.0,
                   JOIN_BYTES ? (double)refused / (double)JOIN_BYTES : 0.0);
            for (k = 0; k < NJ_LIST_N; k++)
                printf("      pair %2u -> %2u  %6.1f KiB  %s\n",
                       NJ_LIST[k].i, NJ_LIST[k].j, NJ_LIST[k].bytes / 1024.0,
                       NOJOIN_TAG[NJ_LIST[k].why]);
        }
        /* The producers the A/B has to sweep, so the list does not have to be reconstructed
         * from the graph by hand. */
        if (CUBE_JOINS) {
            unsigned m;
            printf("      joined at producer(s):");
            for (m = 0; m < H->n_layers; m++) if (CUBE_OUT[m]) printf(" %u", m);
            printf("\n");
        }
    }
}

/* Build one chain per run of two or more cube-linked layers.
 *
 * FINDING the runs is the library's, not this gate's: it is the same thirty lines every
 * frontend with a graph would write, it needs no hardware knowledge, and duplicating it
 * was the smell. rocket_conv2d_int8_chain_plan_rk3576() does it, and this is what gates
 * it — including its program-count split, which is why the build-and-shorten loop that
 * used to be here is gone.
 *
 * What stays here is the part that is a property of THIS caller's lowering rather than of
 * the handles: a NULL entry is how the finder is told a layer cannot be in a run, and the
 * array below is built with a NULL wherever this gate would prepare the layer's input on
 * the host. A chain scatters the tensor it is handed and has no way to know a widened or
 * bordered copy was meant. */
static void kick_build(int fd)
{
    rocket_chain_node_rk3576 *cand;
    rocket_conv2d_int8_run_rk3576 runs[32];
    unsigned i, nruns, r;

    if (!CUBE_ON || !KICK_OF || !kick_on()) {
        if (CUBE_ON && !kick_on())
            printf("   cross-layer kick: OFF (ROCKET_RK3576_NET_KICK=0) — one submit per "
                   "layer, which is the A/B control\n");
        return;
    }
    cand = calloc(H->n_layers, sizeof *cand);
    if (!cand) return;
    for (i = 0; i < H->n_layers; i++) {
        if (needs_prep(&LAYERS[i])) continue;         /* a host-prepared input */
        /* An ADD reading a host-built concatenation, and a SKIP SOURCE whose output is
         * read on the host later, are both host work between two programs — so neither
         * can sit inside a stream, and a NULL here is how the finder is told so. A WIRED
         * one is neither: its operands and its consumer are slices of one buffer, and
         * there is nothing left between the two programs at all. */
        if ((LAYERS[i].kind == KIND_ADD && !(CUBE_IN && CUBE_IN[i])) ||
            (SKIP[i] && !(CAT_OF && CAT_OF[i]) && !(MULTI_OF && MULTI_OF[i]))) continue;
        /* THE LOCALIZER. The finder returns MAXIMAL runs, so naming a layer that sits
         * inside a longer one would name no run at all — cut the array before it instead,
         * and the run the finder then finds is the one that starts there. */
        if (kick_at() >= 0 && i < (unsigned)kick_at()) continue;
        /* A POOLING layer is a node of the run rather than a break in it: a pool program
         * runs inside a convolution stream on this part [HW sweep, rk3576_chain_pool].
         * The finder places it — it may only be interior — so this only has to offer it. */
        if (IS_POOL(LAYERS[i].kind)) {
            if (RESIDENT_POOL) cand[i].pool = RESIDENT_POOL[i];
            continue;
        }
        cand[i].conv = RESIDENT[i];
    }
    nruns = rocket_chain_plan_rk3576(cand, H->n_layers, runs,
                                     (unsigned)(sizeof runs / sizeof *runs));
    if (nruns > sizeof runs / sizeof *runs) nruns = sizeof runs / sizeof *runs;
    for (r = 0; r < nruns; r++) {
        unsigned first = runs[r].first, n = runs[r].count;
        if (kick_at() >= 0 && (unsigned)kick_at() != first) continue;
        if (n > kick_max()) n = kick_max();          /* the bisection instrument */
        if (n < 2u) continue;
        KICK_OF[first] = rocket_chain_new_rk3576(fd, cand + first, n);
        if (!KICK_OF[first]) {
            printf("   cross-layer kick: the run at layer %u (%u layers, %u programs) was "
                   "refused; those layers keep one submit each\n",
                   first, n, runs[r].programs);
            continue;
        }
        KICK_END[first] = first + n;
        KICK_RUNS++;
        KICK_LAYERS += (int)n;
    }
    free(cand);
    if (KICK_RUNS)
        printf("   cross-layer kick: %d run(s) covering %d layer(s) — one hardware kick "
               "each where the per-layer path takes one per layer\n",
               KICK_RUNS, KICK_LAYERS);
    else
        printf("   cross-layer kick: no run of two or more linked layers, so nothing to "
               "chain\n");
}

/* Run one layer. `how` is set to the path it took. `in2` is the second operand of an add
 * and NULL everywhere else. */
static int layer_run(int fd, const rnet_layer *L, const int8_t *in, const int8_t *in2,
                     int8_t *out, const char **how)
{
    const int8_t *W = L->w_bytes ? at(L->w_off) : NULL;
    const int32_t *bias = L->b_bytes ? (const int32_t *)(BLOB + L->b_off) : NULL;
    rocket_conv2d_desc d;
    int8_t *pin = NULL, *pw = NULL;
    unsigned icx = L->ic, ihx = L->ih, iwx = L->iw;
    int rc;

    if (L->kind == KIND_ADD) {
        unsigned li = (unsigned)(L - LAYERS);
        int8_t *cat, *aw = NULL;
        int32_t *ab = NULL;
        float ws = 1.0f;
        int wp[2];

        /* A WIRED ADD HAS NO HOST OPERANDS AT ALL: its two producers wrote their own
         * slices of the buffer its handle reads as a cube, so there is nothing to
         * concatenate and `in2` is meaningless. */
        int wired = CUBE_IN && CUBE_IN[li];
        if (!in2 && !wired) { *how = "NO-OPERAND"; return ROCKET_E_SHAPE; }
        rc = add_weights(L, &aw, &ab, &ws, wp);
        if (rc) return rc < 0 ? ROCKET_E_NOMEM : rc;
        cat = wired ? NULL : add_concat(L, in, in2);
        if (!cat && !wired) { free(aw); free(ab); return ROCKET_E_NOMEM; }
        add_desc(L, &d);
        {
            rocket_conv2d_int8_weights_rk3576 *rp = NULL;
            if (RESIDENT_ON) {
                if (!RESIDENT[li])
                    RESIDENT[li] = rocket_conv2d_int8_pack_rk3576(
                        fd, &d, aw, ab, L->in_scale, ws, NULL, L->out_scale,
                        L->in_zp, 0, L->out_zp);
                rp = RESIDENT[li];
            }
            if (rp) {
                rc = rocket_conv2d_int8_prepacked_rk3576(fd, rp, cat,
                                                         (CUBE_OUT && CUBE_OUT[li])
                                                             ? NULL : out);
                *how = wired ? ">npu-add<"
                             : (CUBE_OUT && CUBE_OUT[li]) ? "npu-add<" : "npu-add-res";
            } else {
                rc = rocket_conv2d_int8_rk3576(fd, &d, cat, aw, ab, L->in_scale, ws,
                                               L->out_scale, L->in_zp, 0, L->out_zp,
                                               out);
                *how = "npu-add";
            }
        }
        /* The host fallback needs the concatenated tensor, which a wired add does not
         * build — so there is nothing to fall back TO and the failure is reported. */
        if (rc != ROCKET_OK && cat) {
            rnet_layer R;
            add_ref_layer(L, ws, &R);
            host_conv(&R, cat, aw, ab, out);
            *how = "host-add";
            rc = ROCKET_OK;
        }
        free(cat); free(aw); free(ab);
        return rc;
    }

    if (IS_POOL(L->kind)) {
        unsigned li = (unsigned)(L - LAYERS);
        rocket_pool_desc p;
        pool_desc_of(L, &p);
        *how = rocket_pool_int8_rk3576_exact(&p) ? "npu-pool" : "npu-pool*";
        if (RESIDENT_ON) {
            if (!RESIDENT_POOL[li])
                RESIDENT_POOL[li] = rocket_pool_int8_pack_rk3576(fd, &p, L->in_zp);
            if (RESIDENT_POOL[li]) {
                int ci = CUBE_IN && CUBE_IN[li];
                int co = CUBE_OUT && CUBE_OUT[li];
                *how = ci ? (co ? ">npu-pool<" : "npu-pool>")
                          : (co ? "npu-pool<" : "npu-pool-res");
                return rocket_pool_int8_prepacked_rk3576(fd, RESIDENT_POOL[li],
                                                         ci ? NULL : in, co ? NULL : out);
            }
        }
        return rocket_pool_int8_rk3576(fd, &p, L->in_zp, in, out);
    }
    if (L->kind == KIND_SOFTMAX) {
        *how = "host-softmax";
        memcpy(out, in, (size_t)L->oc * L->oh * L->ow);   /* rank-preserving; see main */
        return 0;
    }

    /* THE PACKED-IMAGE STEM, before the widening it replaces. It owns its own weights
     * (a different cube, which the resident handle does not drive) so it returns rather
     * than falling through. */
    {
        struct stem_plan sp;
        if (stem_plan_of(L, &sp)) {
            int8_t *sh = stem_shift_input(L, &sp, in);
            if (!sh) return ROCKET_E_NOMEM;
            memset(&d, 0, sizeof d);
            d.ic = (int)L->ic; d.ih = (int)L->ih; d.iw = (int)L->iw;
            d.oc = (int)L->oc; d.kh = (int)L->kh; d.kw = (int)L->kw;
            d.stride_y = (int)L->sy; d.stride_x = (int)L->sx;
            d.dil_y = d.dil_x = 1;
            d.pad_top = (int)sp.pad_y; d.pad_left = (int)sp.pad_x;
            if ((unsigned)rocket_conv2d_oh(&d) == L->oh &&
                (unsigned)rocket_conv2d_ow(&d) == L->ow) {
                rc = rocket_conv2d_int8_rk3576(fd, &d, sh, W, bias, L->in_scale,
                                               L->w_scale, L->out_scale,
                                               L->in_zp, L->w_zp, L->out_zp, out);
                if (rc == ROCKET_OK) {
                    stem_fixup(L, &sp, in, W, bias, out);
                    free(sh);
                    *how = "npu-stem";
                    return ROCKET_OK;
                }
                /* A shape the plan admits and the entry does not: the widened lowering
                 * is still exact, so the run continues and says which path it took. */
                printf("   the packed-image stem refused (%d) — widening instead\n", rc);
            }
            free(sh);
        }
    }

    if (needs_prep(L)) {
        pin = prep_input(L, in, &icx, &ihx, &iwx);
        if (!pin) return ROCKET_E_NOMEM;
        if (L->kind == KIND_CONV && L->ic <= 4 && widen_on()) {
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
    d.direct_datapath = narrow_direct(L);
    if (asympad_on()) {
        /* The LEADING pad is the registers', the trailing one is derived from the extent,
         * and `pin` — when there is one — only widened the channels. */
        d.pad_top  = (int)L->pl_y;
        d.pad_left = (int)L->pl_x;
        d.oh = (int)L->oh; d.ow = (int)L->ow;
    } else {
        /* Materialised: the border is in the buffer, so the registers pad nothing. */
        d.pad_top  = pin ? 0 : (int)L->pl_y;
        d.pad_left = pin ? 0 : (int)L->pl_x;
    }

    if ((unsigned)rocket_conv2d_oh(&d) != L->oh ||
        (unsigned)rocket_conv2d_ow(&d) != L->ow) {
        printf("  lowering produced %dx%d, the model says %ux%u\n",
               rocket_conv2d_oh(&d), rocket_conv2d_ow(&d), L->oh, L->ow);
        free(pin); free(pw);
        return ROCKET_E_SHAPE;
    }

    {
        /* The handle is packed from the weights this call would have handed the transient
         * entry — `pw` for the widened stem, `W` otherwise — so residency changes nothing
         * about what is computed, only how often it is prepared. */
        rocket_conv2d_int8_weights_rk3576 *rp = NULL;
        if (RESIDENT_ON) {
            unsigned li = (unsigned)(L - LAYERS);
            if (!RESIDENT[li])
                RESIDENT[li] = rocket_conv2d_int8_pack_rk3576(
                    fd, &d, pw ? pw : W, bias, L->in_scale, L->w_scale, NULL,
                    L->out_scale, L->in_zp, L->w_zp, L->out_zp);
            rp = RESIDENT[li];
        }
        if (rp)
            rc = rocket_conv2d_int8_prepacked_rk3576(fd, rp, pin ? pin : in, out);
        else if (d.depthwise)
            rc = rocket_conv2d_dw_int8_rk3576(fd, &d, pin ? pin : in, W, bias,
                                              L->in_scale, L->w_scale, L->out_scale,
                                              L->in_zp, L->w_zp, L->out_zp, out);
        else
            rc = rocket_conv2d_int8_rk3576(fd, &d, pin ? pin : in, pw ? pw : W, bias,
                                           L->in_scale, L->w_scale, L->out_scale,
                                           L->in_zp, L->w_zp, L->out_zp, out);
        {
            unsigned li = (unsigned)(L - LAYERS);
            int ci = CUBE_IN && CUBE_IN[li], co = CUBE_OUT && CUBE_OUT[li];
            *how = rc != ROCKET_OK ? "REFUSED"
                 : !rp            ? (d.depthwise ? "npu-dw"
                                   : pin         ? "npu-prep" : "npu")
                 /* Which transposes this layer still pays, which is the whole point of
                  * the chain: ">" reads its input as a cube, "<" leaves its output as
                  * one, "<>" neither transpose runs. */
                 : ci && co       ? (d.depthwise ? "npu-dw<>" : "npu<>")
                 : ci             ? (d.depthwise ? "npu-dw>"  : "npu>")
                 : co             ? (d.depthwise ? "npu-dw<"  : "npu<")
                 :                  (d.depthwise ? "npu-dw-res" : "npu-res");
        }
    }
    if (rc != ROCKET_OK) {
        /* The chain has to reach a label even when one layer will not run, so the
         * fallback is taken and NAMED rather than the run being abandoned. The REASON is
         * printed, because a fallback computes the part's own arithmetic on the host and
         * so scores "exact" — silently, it reads as a layer that ran. */
        printf("  layer %u will not run on the part: %d\n",
               (unsigned)(L - LAYERS), rc);
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

/* TFLite's own top-5 for this image, out of the class probabilities the blob carries.
 * One arithmetic, because every pass that reports a label compares against it. */
static void tflite_top5(int out[5])
{
    const uint8_t *probs = (const uint8_t *)(BLOB + H->probs_off);
    long val[5];
    int i, j, k;
    for (k = 0; k < 5; k++) { out[k] = -1; val[k] = -1; }
    for (i = 0; i < (int)H->n_classes; i++)
        for (k = 0; k < 5; k++)
            if ((long)probs[i] > val[k]) {
                for (j = 4; j > k; j--) { val[j] = val[j-1]; out[j] = out[j-1]; }
                val[k] = probs[i]; out[k] = i;
                break;
            }
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

static void print_top5(const char *what, const int t[5])
{
    char name[128];
    int k;
    printf("   %-26s ", what);
    for (k = 0; k < 5; k++) {
        label_of(t[k], name, sizeof name);
        printf("%s%s", k ? ", " : "", name);
    }
    printf("\n");
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
static void diagnose(int fd, const rnet_layer *L, const int8_t *in, const int8_t *in2,
                     const int8_t *first, const int8_t *ref, size_t cap)
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
        if (layer_run(fd, L, in, in2, again, &how) != ROCKET_OK) break;
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

/*
 * THREE LOWERINGS OF ONE LAYER, AND THEY MUST AGREE BYTE FOR BYTE.
 *
 * The stem is the only layer this network can express three ways: the CNA's packed-image
 * sub-encoding, the direct datapath at the model's own three channels, and the direct
 * datapath with the image widened to eight. Each is exact for a DIFFERENT reason, which is
 * what makes the comparison worth running rather than circular —
 *
 *   packed-image  the shifted-input trick, with the output's leading rows and columns
 *                 recomputed on the host;
 *   narrow direct the library fills its cube's padding channels with the input zero point
 *                 and folds over the PROGRAMMED tap count;
 *   widened       the caller's own added channels hold that zero point against zero
 *                 weights, and the fold is over eight channels' worth of taps.
 *
 * THE BORDER FIX-UP NEEDS A NON-CIRCULAR CHECK, and this is it. Every other element of the
 * gate is asserted against host_conv(), and the packed-image stem's frame is COMPUTED by
 * host_conv_rect() — so the usual assertion says nothing about the 1.8% of the surface that
 * matters most there. The other two lowerings reach that border through the hardware. It
 * also catches a frame one row or column too narrow, which would leave hardware output in
 * place where the shift makes it wrong.
 *
 * It drives all three whatever the defaults are, so turning a lowering off for being slower
 * does not quietly stop gating it.
 *
 * THE RESIDENT HANDLE HAS TO BE DROPPED BETWEEN THE TWO DIRECT RUNS. layer_run() packs one
 * per layer index and reuses it, and the two direct lowerings differ in their DESCRIPTOR —
 * so sharing a handle would compare a lowering against itself and pass unconditionally.
 */
static int stem_ab(int fd, const rnet_layer *L, const int8_t *in, size_t cap)
{
    struct stem_plan sp;
    int8_t *buf[3] = {NULL, NULL, NULL};
    const char *how[3] = {"?", "?", "?"};
    size_t n = (size_t)L->oc * L->oh * L->ow, i;
    unsigned li = (unsigned)(L - LAYERS);
    int rc = 0, saved_stem = STEM_OFF, saved_widen = WIDEN_ON, v;
    long diff_nd = 0, diff_wd = 0;

    STEM_OFF = 0;
    if (!stem_plan_of(L, &sp)) { STEM_OFF = saved_stem; return 0; }
    for (v = 0; v < 3; v++)
        if (!(buf[v] = malloc(cap))) {
            STEM_OFF = saved_stem;
            while (v-- > 0) free(buf[v]);
            return 0;
        }

    /* 0: packed-image. 1: narrow direct. 2: widened direct. */
    for (v = 0; v < 3 && !rc; v++) {
        rocket_conv2d_int8_weights_rk3576 *held = NULL;
        STEM_OFF = (v != 0);
        WIDEN_ON = (v == 2);
        if (RESIDENT_ON && v != 0) { held = RESIDENT[li]; RESIDENT[li] = NULL; }
        if (layer_run(fd, L, in, NULL, buf[v], &how[v]) != ROCKET_OK) rc = 1;
        if (RESIDENT_ON && v != 0) {
            if (RESIDENT[li]) rocket_conv2d_int8_weights_free_rk3576(fd, RESIDENT[li]);
            RESIDENT[li] = held;
        }
    }
    STEM_OFF = saved_stem;
    WIDEN_ON = saved_widen;

    if (!rc) {
        size_t frame = (size_t)L->oc *
                       ((size_t)L->oh * L->ow -
                        (size_t)(L->oh - sp.lo_y - sp.hi_y) * (L->ow - sp.lo_x - sp.hi_x));
        for (i = 0; i < n; i++) {
            if (buf[1][i] != buf[2][i]) diff_nd++;
            if (buf[0][i] != buf[2][i]) diff_wd++;
        }
        printf("   stem A/B over %zu elements: %s vs %s %ld differ, %s vs %s %ld differ; "
               "the frame the host recomputes is %zu of them (%.1f%%)\n",
               n, how[1], how[2], diff_nd, how[0], how[2], diff_wd,
               frame, 100.0 * (double)frame / (double)n);
        if (diff_nd) {
            printf("      the NARROW direct path and the widened one DISAGREE — the tap "
                   "count the zero-point fold uses and the value the cube's padding "
                   "channels carry have to match, and one of them does not\n");
            rc = 1;
        }
        if (diff_wd) {
            printf("      the packed-image stem and the widened lowering DISAGREE — one "
                   "of them is wrong, and the border fix-up's extent is the first "
                   "suspect (rows %u/%u, columns %u/%u)\n",
                   sp.lo_y, sp.hi_y, sp.lo_x, sp.hi_x);
            rc = 1;
        }
    }
    for (v = 0; v < 3; v++) free(buf[v]);
    return rc;
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
        /* The layer's OWN operand edge, not "the one before". They coincide on a
         * feed-forward chain and a skip is exactly where they do not. */
        const int8_t *in = L->src1 == NO_SRC ? at(H->img_off) : at(LAYERS[L->src1].g_off);
        const int8_t *in2 = skip_operand(L, 1);
        const char *how = "?";
        score hw, tf;
        double t0, ms;
        int rc, have_ref = 0;

        if (L->kind == KIND_SOFTMAX) continue;       /* host, and rank-preserving */
        t0 = now_ms();
        rc = layer_run(fd, L, in, in2, out, &how);
        ms = now_ms() - t0;
        if (rc != ROCKET_OK) {
            printf("%2u %-8s %-13s ENTRY RETURNED %d\n", i, KIND_NAME[L->kind], how, rc);
            failed++; continue;
        }
        if (L->kind == KIND_ADD) {
            /* The add's reference is the CONVOLUTION's reference on the lowered problem,
             * built from the same library weights the part was handed. */
            int8_t *aw = NULL, *cat;
            int32_t *ab = NULL;
            float ws = 1.0f;
            int wp[2];
            if (!add_weights(L, &aw, &ab, &ws, wp) &&
                (cat = add_concat(L, in, in2)) != NULL) {
                rnet_layer R;
                add_ref_layer(L, ws, &R);
                host_conv(&R, cat, aw, ab, ref);
                have_ref = 1;
                free(cat);
            }
            free(aw); free(ab);
        } else if (L->kind == KIND_CONV || L->kind == KIND_DWCONV) {
            host_conv(L, in, at(L->w_off), (const int32_t *)(BLOB + L->b_off), ref);
            have_ref = 1;
        } else if (IS_POOL(L->kind)) {
            rocket_pool_desc p;
            pool_desc_of(L, &p);
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
            diagnose(fd, L, in, in2, out, ref, cap);
            failed++;
        } else if (tf.maxdiff > tflite_slack(L)) {
            printf("      the part matches its own arithmetic but is %d counts from "
                   "TFLite, past the %d this lowering can owe\n",
                   tf.maxdiff, tflite_slack(L));
            wrong_extent(L, out, at(L->g_off));
            failed++;
        }
        failed += stem_ab(fd, L, in, cap);
    }
    printf("   TFLite distance over the whole graph: %ld of %ld elements differ "
           "(%.4f%%), all by one count where noted\n",
           tf_diff_total, tf_total, 100.0 * (double)tf_diff_total / (double)tf_total);
    free(out); free(ref);
    return failed;
}

/* POISON THE KICK ON PURPOSE, so the write guard's coverage is a measurement.
 *
 * A job whose DPU output element is wider than one byte leaves the NEXT submit of any
 * kind — across calls and across processes — completing normally and writing nothing.
 * That is the hazard the sentinel guard exists for, and on a cross-layer kick it is what
 * decides whether checking ONE surface is enough: a poisoned kick is a poisoned SUBMIT,
 * so every surface in it should still hold its sentinel and any of them witnesses it.
 *
 * ROCKET_RK3576_NET_POISON=1 runs one int32-output matmul before each inference. The arm
 * that makes the others mean anything is the one with the guard OFF: if the graph still
 * passes there, the poisoning never reached the kick and no PASS above it is evidence. */
static int poison_once(int fd)
{
    enum { PM = 32, PK = 64, PN = 64 };
    static int8_t *pa, *pb;
    static int32_t *pc;
    if (!pa) {
        pa = calloc((size_t)PM * PK, 1);
        pb = calloc((size_t)PN * PK, 1);
        pc = calloc((size_t)PM * PN, sizeof *pc);
        if (!pa || !pb || !pc) return -1;
        pa[0] = 1; pb[0] = 1;
    }
    return rocket_matmul_int8_rk3576_i32(fd, PM, PK, PN, pa, pb, NULL, pc);
}

static int poison_on(void)
{
    const char *e = getenv("ROCKET_RK3576_NET_POISON");
    return (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 0;
}

/* A BENCH THAT FEEDS THE SAME IMAGE EVERY ITERATION CANNOT SEE A STALE SURFACE, and that
 * is not a detail of this test — it is what any repeated-identical-input measurement is
 * blind to. A layer whose program did not run leaves its output surface holding the
 * PREVIOUS inference's contents, which on a repeated input are exactly the contents the
 * layer would have written. The graph then returns the right answer for the wrong reason
 * and every check downstream of it passes.
 *
 * ROCKET_RK3576_NET_VARY=1 alternates two inputs and scores each inference byte for byte
 * against the logits that same input produced on a clean, unpoisoned iteration. Staleness
 * then shows up as one input returning the other's answer. The two references are taken
 * from iterations 0 and 1, which run without the poison injection, and the run refuses if
 * they are equal — identical references would make the whole check vacuous. */
static int vary_on(void)
{
    const char *e = getenv("ROCKET_RK3576_NET_VARY");
    return (e && *e) ? (int)strtol(e, NULL, 0) != 0 : 0;
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

    /* STEADY STATE, per layer. The table below is printed on the FIRST iteration, where a
     * resident layer is still paying its one-time pack — so it cannot be read as the cost
     * of anything. These accumulate every iteration and are what a lever is measured on. */
    double *acc = calloc(H->n_layers, sizeof *acc);
    const char **paths = calloc(H->n_layers, sizeof *paths);

    if (!a || !b || !acc || !paths) {
        free(a); free(b); free(acc); free(paths); return 1;
    }
    printf("\n== CHAIN: the network, layer n fed layer n-1's NPU output ==\n");
    uint64_t g_sub0 = rocket_submit_ioctl_count(), g_task0 = rocket_submit_task_count();
    /* The two inputs and their references, when the staleness check is on. */
    size_t img_bytes = (size_t)H->in_c * H->in_h * H->in_w;
    int8_t *alt = NULL, *ref[2] = { NULL, NULL };
    int stale = 0;
    if (vary_on()) {
        size_t k;
        alt    = malloc(img_bytes);
        ref[0] = malloc(H->n_classes);
        ref[1] = malloc(H->n_classes);
        if (!alt || !ref[0] || !ref[1]) {
            free(a); free(b); free(acc); free(paths);
            free(alt); free(ref[0]); free(ref[1]); return 1;
        }
        for (k = 0; k < img_bytes; k++) alt[k] = (int8_t)~at(H->img_off)[k];
        printf("   VARYING the input: two images alternating, each scored byte for byte "
               "against its own clean answer\n");
    }
    if (poison_on())
        printf("   POISONING each inference with an int32-output matmul first\n");
    for (it = 0; it < iters; it++) {
        const int8_t *cur = (alt && (it & 1)) ? alt : at(H->img_off);
        int8_t *bufs[2];
        int slot = 0;
        double t0 = now_ms();
        int drift = -1;

        /* Iterations 0 and 1 are the clean references; the injection starts after them. */
        if (poison_on() && (!alt || it >= 2) && poison_once(fd) != ROCKET_OK) {
            failed++; break;
        }
        bufs[0] = a; bufs[1] = b;
        for (i = 0; i < H->n_layers; i++) {
            const rnet_layer *L = &LAYERS[i];
            const char *how = "?";
            /* A skip source writes to a buffer of its own, which nothing else reuses, so
             * the add three layers later still has an operand to read. Everything else
             * runs on the ping-pong pair. */
            int8_t *dst = SKIP[i] ? SKIP[i] : bufs[slot];
            score s;
            double lt = now_ms();
            uint64_t s0 = rocket_submit_ioctl_count();
            uint64_t k0 = rocket_submit_task_count();
            int rc;
            if (L->kind == KIND_SOFTMAX) continue;
            /* A CROSS-LAYER KICK covers layers i..KICK_END[i]-1 in one submit, so its cost
             * is attributed to the run's FIRST layer and the ones it swallowed show 0.00.
             * That is what the table can honestly say: there is no per-layer wall inside a
             * kick to report. */
            if (KICK_OF && KICK_OF[i]) {
                unsigned last = KICK_END[i] - 1u;
                /* THE KICK'S OUTPUT IS ITS LAST LAYER'S, so the buffer it lands in is
                 * that layer's — a run starting at a skip source would otherwise write
                 * the last layer's tensor into a buffer sized for the first one. */
                int8_t *kdst = SKIP[last] ? SKIP[last] : bufs[slot];
                rc = rocket_conv2d_int8_chain_run_rk3576(fd, KICK_OF[i], chain_input(i, cur),
                        CUBE_OUT[last] ? NULL : kdst);
                lt = now_ms() - lt;
                acc[i] += lt;
                paths[i] = "npu-kick";
                if (rc != ROCKET_OK) {
                    printf("%2u %-8s THE KICK RETURNED %d\n", i, KIND_NAME[L->kind], rc);
                    failed++; break;
                }
                if (!CUBE_OUT[last]) { cur = kdst; if (!SKIP[last]) slot ^= 1; }
                i = last;
                continue;
            }
            rc = layer_run(fd, L, chain_input(i, cur), chain_operand_b(L, i, cur), dst, &how);
            lt = now_ms() - lt;
            if (rc != ROCKET_OK) {
                printf("%2u %-8s ENTRY RETURNED %d\n", i, KIND_NAME[L->kind], rc);
                failed++; break;
            }
            acc[i] += lt;
            paths[i] = how;
            /* A cube-out layer wrote no row-major tensor, so there is nothing here to
             * score and `dst` still holds an older layer's output — reading it would
             * report a fabricated distance. The join is asserted by pass_cube(), which
             * compares the whole chain against the row-major one. */
            if (CUBE_OUT && CUBE_OUT[i]) {
                s.exact = s.total = 0; s.maxdiff = 0;
            } else {
                score_vs(L, dst, at(L->g_off), &s);
                if (s.maxdiff > tflite_slack(L) && drift < 0) drift = (int)i;
            }
            if (it == 0)
                printf("%2u %-8s %4ux%-3ux%-4u -> %4ux%-3ux%-4u %-13s %8.2f ms  "
                       "%3llu sub/%3llu task  %5.1f%% match TFLite, maxdiff %d\n",
                       i, KIND_NAME[L->kind], L->ic, L->ih, L->iw, L->oc, L->oh, L->ow,
                       how, lt,
                       (unsigned long long)(rocket_submit_ioctl_count() - s0),
                       (unsigned long long)(rocket_submit_task_count() - k0),
                       s.total ? 100.0 * (double)s.exact / (double)s.total : 0.0,
                       s.maxdiff);
            /* Only a materialised output advances the ping-pong. The flags are set in
             * PAIRS, so a layer that is not a cube consumer always follows one that is not
             * a cube producer, and `cur` is that layer's input. */
            if (!(CUBE_OUT && CUBE_OUT[i])) {
                cur = dst;
                if (!SKIP[i]) slot ^= 1;    /* a skip source did not take a slot */
            }
        }
        logits = cur;
        wall += now_ms() - t0;
        if (alt) {
            if (it < 2) {
                memcpy(ref[it], logits, H->n_classes);
                if (it == 1 && !memcmp(ref[0], ref[1], H->n_classes)) {
                    printf("   REFUSING: the two inputs produce identical logits, so the "
                           "staleness check would pass on anything\n");
                    failed++; break;
                }
            } else if (memcmp(logits, ref[it & 1], H->n_classes)) {
                stale++;
            }
        }
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
    /* Submits per inference, NOT an estimate of what they cost. The part's ~439 us
     * submit floor is a floor on a round trip that waits for a full surface to drain,
     * and these jobs are far under it — measured, the submits are single-digit percent
     * of this graph's wall while the host operand packing is most of it. The split is
     * ROCKET_RK3576_INT8_PROF=1, per layer. */
    printf("   %d run(s), %.1f ms each, %.1f submits and %.1f tasks per inference\n",
           iters, wall / iters,
           (double)(rocket_submit_ioctl_count() - g_sub0) / iters,
           (double)(rocket_submit_task_count()  - g_task0) / iters);
    if (alt) {
        printf("   %d of %d scored inference(s) returned logits that are NOT this input's "
               "own clean answer\n", stale, iters > 2 ? iters - 2 : 0);
        if (stale) failed++;
        free(alt); free(ref[0]); free(ref[1]);
        /* The TFLite top-5 below scores the LAST iteration, which under VARY is whichever
         * of the two inputs the parity landed on — the second one is not the image TFLite
         * was run on, so that comparison is not this mode's question. */
        logits = NULL;
    }

    if (iters > 1) {
        double tot = 0.0;
        printf("   the ten most expensive layers, averaged over all %d runs:\n", iters);
        for (i = 0; i < H->n_layers; i++) tot += acc[i];
        for (i = 0; i < 10; i++) {
            unsigned j, best = 0;
            double bv = -1.0;
            for (j = 0; j < H->n_layers; j++)
                if (acc[j] > bv) { bv = acc[j]; best = j; }
            if (bv <= 0.0) break;
            printf("      %2u %-8s %4ux%-3ux%-4u -> %4ux%-3ux%-4u %-12s %6.2f ms "
                   "(%4.1f%%)\n",
                   best, KIND_NAME[LAYERS[best].kind], LAYERS[best].ic, LAYERS[best].ih,
                   LAYERS[best].iw, LAYERS[best].oc, LAYERS[best].oh, LAYERS[best].ow,
                   paths[best] ? paths[best] : "?", bv / iters, 100.0 * bv / tot);
            acc[best] = -1.0;
        }
    }
    free(acc); free(paths);

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

/* ============================================================================
 * SECTION — the PER-AXIS pass
 *
 * rocket_conv2d_int8_perchannel_rk3576() is gated bit-exactly per shape, but a per-axis
 * requant's error is a PER LAYER quantity that COMPOUNDS, and no per-op gate can answer
 * what twenty-seven of them do to a label. This pass is that question.
 *
 * WHERE THE PER-AXIS MODEL COMES FROM. Not from a second `.tflite` — this one is
 * per-tensor because it is the legacy uint8 quantization — but from RE-QUANTIZING its
 * own weights per output channel, which is exactly what a TFLite int8 converter does to
 * the same float tensor. Channel c's stored weights are `(w - w_zp)` integers on a
 * shared grid of `w_scale`; the per-axis form puts channel c on its own grid of
 * `max|w_c - w_zp| * w_scale / 127` and rescales its weights and its bias onto it. The
 * activations keep the model's own per-tensor scales and zero points, again as TFLite
 * does. The result is a genuinely per-axis layer with the scale spread this model's real
 * weights have, and the spread is printed per layer so the numbers are interpretable.
 *
 * WHAT IS ASSERTED, AND WHAT IS NOT. The bit-exact assertion against the chip's own
 * arithmetic lives in tests/rk3576_perchannel_gate.c, where the planner's C multipliers
 * and OUT_CVT shift are visible to a model. Here they are not — the planner picks the
 * output-channel tile from the weights and the accuracy target — so this pass asserts
 * what a graph is FOR: that no layer refuses, that no layer's distance from an EXACT
 * per-axis requant is large enough to be a defect rather than the datapath's single
 * shift, and that the compounded chain still LANDS IN TFLITE'S TOP-5. The per-layer
 * distances are reported.
 *
 * THE TOP-1 IS NOT ASSERTED, AND THE REASON IS A MEASUREMENT. Re-quantizing per axis
 * moves the logits, and on an image whose top-2 are close that is enough to reorder
 * them: ResNet-18 puts "suit" 7 counts ahead of "bulletproof vest" per-tensor and the
 * per-axis chain puts them 2 counts the other way, with every operand resolved by
 * producer, ON THE CPU (rk3576_net_gate hostchain). A rank swap inside the top-5 at a
 * margin the int8 logit grid can barely express is the requant's accuracy; landing
 * outside it is not, and the same instrument measured that regime too — threading
 * operand A puts TFLite's own top-1 120 counts behind. Two counts and 120 counts are
 * what this assertion is placed between, and a top-5 membership test separates them
 * without a threshold fitted to either.
 * ==========================================================================*/

/* Past this a layer is not "one shift's worth of gain resolution", it is broken. The
 * per-op gate's worst case at a 100x spread and a large fan-in is 26.6 counts in ONE
 * tile and 1.0 at a tile of 32, and the planner here is free to tile; anything past 32
 * counts on a real layer means something other than the requant. */
#define PERAXIS_DEFECT_COUNTS  32

struct peraxis_layer {
    int8_t  *w;        /* [OC][IC][KH][KW] or [C][KH][KW], symmetric int8   */
    float   *wsc;      /* [OC] the per-channel scale                        */
    int32_t *bias;     /* [OC] re-quantized to in_scale*wsc[c]              */
    double   spread;   /* max(wsc)/min(wsc) over the non-zero channels      */
    unsigned clamped;  /* bias entries that saturated int32 on the rescale  */
};

static void peraxis_free(struct peraxis_layer *p, unsigned n)
{
    unsigned i;
    if (!p) return;
    for (i = 0; i < n; i++) { free(p[i].w); free(p[i].wsc); free(p[i].bias); }
    free(p);
}

/* Re-quantize one layer's weights and bias per output channel. Integer throughout on
 * the weight side: `k = w - w_zp` is already an integer and the new value is
 * `round(k * 127 / max|k|)`, so no float grid is invented in between. */
static int peraxis_build(const rnet_layer *L, struct peraxis_layer *p)
{
    size_t per = (L->kind == KIND_DWCONV) ? (size_t)L->kh * L->kw
                                          : (size_t)L->ic * L->kh * L->kw;
    const int8_t *W = at(L->w_off);
    const int32_t *bias = L->b_bytes ? (const int32_t *)(BLOB + L->b_off) : NULL;
    double lo = 0.0, hi = 0.0;
    unsigned c;
    size_t j;

    p->w    = malloc((size_t)L->oc * per);
    p->wsc  = malloc((size_t)L->oc * sizeof *p->wsc);
    p->bias = calloc(L->oc, sizeof *p->bias);
    if (!p->w || !p->wsc || !p->bias) return -1;
    p->clamped = 0;

    for (c = 0; c < L->oc; c++) {
        const int8_t *src = W + (size_t)c * per;
        int8_t *dst = p->w + (size_t)c * per;
        long peak = 0;
        for (j = 0; j < per; j++) {
            long k = (long)src[j] - L->w_zp;
            if (k < 0) k = -k;
            if (k > peak) peak = k;
        }
        if (!peak) {
            /* An all-zero filter has no scale of its own. Any positive value is right
             * for the weights and wrong for nothing; the smallest one keeps it out of
             * the spread's numerator. */
            for (j = 0; j < per; j++) dst[j] = 0;
            p->wsc[c] = L->w_scale;
            p->bias[c] = bias ? bias[c] : 0;
            continue;
        }
        for (j = 0; j < per; j++) {
            double v = (double)((long)src[j] - L->w_zp) * 127.0 / (double)peak;
            long q = (long)(v < 0.0 ? ceil(v - 0.5) : floor(v + 0.5));
            dst[j] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
        }
        p->wsc[c] = (float)((double)peak * (double)L->w_scale / 127.0);
        if (bias) {
            /* The bias is quantized in the ACCUMULATOR domain, so its quantum follows
             * the weight scale: in_scale*w_scale before, in_scale*wsc[c] after. */
            double nb = (double)bias[c] * 127.0 / (double)peak;
            if (nb > 2147483647.0)  { nb = 2147483647.0;  p->clamped++; }
            if (nb < -2147483648.0) { nb = -2147483648.0; p->clamped++; }
            p->bias[c] = (int32_t)(nb < 0.0 ? ceil(nb - 0.5) : floor(nb + 0.5));
        }
    }
    for (c = 0; c < L->oc; c++) {
        double s = p->wsc[c];
        if (!(s > 0.0)) continue;
        if (lo == 0.0 || s < lo) lo = s;
        if (s > hi) hi = s;
    }
    p->spread = lo > 0.0 ? hi / lo : 1.0;
    return 0;
}

/* An EXACT per-axis requant on the CPU: a real-valued per-channel gain, rounded once.
 * Deliberately not the chip's fixed-point form — this is the reference the datapath's
 * single OUT_CVT shift is being measured AGAINST. */
static void peraxis_ref(const rnet_layer *L, const struct peraxis_layer *p,
                        const int8_t *in, int8_t *out)
{
    unsigned c, y, x, i, ky, kx;
    size_t per = (L->kind == KIND_DWCONV) ? (size_t)L->kh * L->kw
                                          : (size_t)L->ic * L->kh * L->kw;
    for (c = 0; c < L->oc; c++) {
        double gain = (double)L->in_scale * (double)p->wsc[c] / (double)L->out_scale;
        const int8_t *wc = p->w + (size_t)c * per;
        for (y = 0; y < L->oh; y++)
            for (x = 0; x < L->ow; x++) {
                int64_t acc = p->bias[c];
                double v;
                long q;
                for (ky = 0; ky < L->kh; ky++)
                    for (kx = 0; kx < L->kw; kx++) {
                        long iy = (long)(y * L->sy + ky) - (long)L->pl_y;
                        long ix = (long)(x * L->sx + kx) - (long)L->pl_x;
                        if (iy < 0 || ix < 0 || iy >= (long)L->ih || ix >= (long)L->iw)
                            continue;
                        if (L->kind == KIND_DWCONV) {
                            acc += (int64_t)(in[((size_t)c * L->ih + iy) * L->iw + ix]
                                             - L->in_zp)
                                 * wc[(size_t)ky * L->kw + kx];
                        } else {
                            for (i = 0; i < L->ic; i++)
                                acc += (int64_t)(in[((size_t)i * L->ih + iy) * L->iw + ix]
                                                 - L->in_zp)
                                     * wc[((size_t)i * L->kh + ky) * L->kw + kx];
                        }
                    }
                v = (double)acc * gain + (double)L->out_zp;
                q = (long)(v < 0.0 ? ceil(v - 0.5) : floor(v + 0.5));
                out[((size_t)c * L->oh + y) * L->ow + x] =
                    (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
            }
    }
}

/* The stem's channel widening, for a per-axis weight array. The added channels are
 * zero weights against samples at the input zero point, which contributes nothing. */
static int8_t *peraxis_prep_weights(const rnet_layer *L, const int8_t *W)
{
    size_t taps = (size_t)L->kh * L->kw;
    int8_t *buf = calloc((size_t)L->oc * STEM_IC * taps, 1);
    unsigned c;
    if (!buf) return NULL;
    for (c = 0; c < L->oc; c++)
        memcpy(buf + (size_t)c * STEM_IC * taps, W + (size_t)c * L->ic * taps,
               L->ic * taps);
    return buf;
}

static int pass_peraxis(int fd)
{
    size_t cap = max_tensor();
    int8_t *a = malloc(cap), *b = malloc(cap), *ref = malloc(cap);
    struct peraxis_layer *P = calloc(H->n_layers, sizeof *P);
    const int8_t *cur, *logits = NULL;
    int8_t *bufs[2];
    unsigned i;
    int slot = 0, failed = 0, worst_all = 0;

    if (!a || !b || !ref || !P) {
        free(a); free(b); free(ref); peraxis_free(P, H->n_layers);
        return 1;
    }
    printf("\n== PER-AXIS: the same network with per-output-channel weight scales ==\n");
    printf("   the weights are re-quantized per channel off this model's own float "
           "tensor; the activations keep its per-tensor scales\n");
    printf("   ASSERTED: no refusal, no layer past %d counts from an EXACT per-axis "
           "requant, and the chain lands inside TFLite's top-5\n", PERAXIS_DEFECT_COUNTS);

    bufs[0] = a; bufs[1] = b;
    cur = at(H->img_off);
    for (i = 0; i < H->n_layers; i++) {
        const rnet_layer *L = &LAYERS[i];
        int8_t *dst = SKIP[i] ? SKIP[i] : bufs[slot];
        /* OPERAND A IS A LOOKUP, on this branch as much as on the pooling and add one
         * below. A convolution whose input is read again three layers later is fed by
         * `src1`, not by the ping-pong: threading it here computed a full and plausible
         * surface for three of ResNet-18's layers, and the per-layer distances could not
         * see it — the reference below is fed the SAME tensor the part was, so a wrong
         * input scores exact. What it moved was the LABEL, by 120 logit counts against
         * the 2 the per-axis requant itself is worth (rk3576_net_gate hostchain). */
        const int8_t *in = chain_input(i, cur);
        const char *how = "?";
        int8_t *pin = NULL, *pw = NULL;
        unsigned icx = L->ic, ihx = L->ih, iwx = L->iw, tile = 0;
        rocket_conv2d_desc d;
        score s;
        int rc;

        if (L->kind == KIND_SOFTMAX) continue;
        if (L->kind != KIND_CONV && L->kind != KIND_DWCONV) {
            /* Pooling and the residual add carry no per-channel weights — the add's are
             * two diagonal blocks, not a quantized tensor — so both stay on the
             * per-tensor path either way. */
            rc = layer_run(fd, L, in, chain_operand_b(L, i, cur), dst, &how);
            if (rc != ROCKET_OK) { printf("%2u %-8s RETURNED %d\n", i,
                                          KIND_NAME[L->kind], rc); failed++; break; }
            printf("%2u %-8s %-11s (no per-channel weights)\n", i,
                   KIND_NAME[L->kind], how);
            cur = dst; if (!SKIP[i]) slot ^= 1;
            continue;
        }

        if (peraxis_build(L, &P[i]) < 0) { failed++; break; }

        if (needs_prep(L)) {
            pin = prep_input(L, in, &icx, &ihx, &iwx);
            if (!pin) { failed++; break; }
            if (L->kind == KIND_CONV && L->ic <= 4 && widen_on()) {
                pw = peraxis_prep_weights(L, P[i].w);
                if (!pw) { free(pin); failed++; break; }
            }
        }
        memset(&d, 0, sizeof d);
        d.ic = (int)icx; d.ih = (int)ihx; d.iw = (int)iwx;
        d.oc = (int)L->oc; d.kh = (int)L->kh; d.kw = (int)L->kw;
        d.stride_y = (int)L->sy; d.stride_x = (int)L->sx;
        d.dil_y = d.dil_x = 1;
        d.depthwise = (L->kind == KIND_DWCONV);
        d.direct_datapath = narrow_direct(L);
        /* THE SAME LOWERING THE PER-TENSOR PATH TAKES, and it has to be: `pin` here only
         * widens the channels when the extent form is on, so a pad_top of zero would put
         * the reference and the hardware on different grids and the layer would report a
         * maxdiff of 255 with no reason attached. */
        if (asympad_on()) {
            d.pad_top  = (int)L->pl_y;
            d.pad_left = (int)L->pl_x;
            d.oh = (int)L->oh; d.ow = (int)L->ow;
        } else {
            d.pad_top  = pin ? 0 : (int)L->pl_y;
            d.pad_left = pin ? 0 : (int)L->pl_x;
        }

        tile = rocket_conv2d_int8_perchannel_oc_tile_rk3576(
                   &d, pw ? pw : P[i].w, P[i].bias, L->in_scale, P[i].wsc,
                   L->out_scale, L->in_zp);
        rc = rocket_conv2d_int8_perchannel_rk3576(
                 fd, &d, pin ? pin : in, pw ? pw : P[i].w, P[i].bias,
                 L->in_scale, P[i].wsc, L->out_scale, L->in_zp, L->out_zp, dst);
        if (rc != ROCKET_OK) {
            printf("%2u %-8s THE PER-AXIS ENTRY RETURNED %d\n", i,
                   KIND_NAME[L->kind], rc);
            free(pin); free(pw);
            failed++;
            break;
        }
        /* The reference is fed the SAME input this layer got — the widened, padded one
         * where there is one — so the number below is this layer's own requant error
         * and not the chain's accumulated drift. */
        {
            rnet_layer R = *L;
            struct peraxis_layer Q = P[i];
            if (pin) {
                R.ic = icx; R.ih = ihx; R.iw = iwx;
                /* The pad moved into the buffer only on the materialising lowering. With
                 * the extent form the registers still pad, so the reference must too. */
                if (!asympad_on()) { R.pl_y = 0; R.pl_x = 0; }
            }
            /* THE WIDENED WEIGHTS, when there are any. peraxis_ref() takes the filter's
             * per-channel stride from R.ic, so handing it the three-channel array under
             * an eight-channel descriptor walks a different filter for every output
             * channel — a surface that is wrong everywhere and looks like a hardware
             * fault. Same trap the per-tensor path avoids by passing `pw` explicitly. */
            if (pw) Q.w = pw;
            peraxis_ref(&R, &Q, pin ? pin : in, ref);
        }
        score_vs(L, dst, ref, &s);
        if (s.maxdiff > worst_all) worst_all = s.maxdiff;
        printf("%2u %-8s %4ux%-3ux%-4u -> %4ux%-3ux%-4u  spread %7.1fx  tile %-4u  "
               "%5.1f%% exact vs an exact per-axis requant, maxdiff %d%s\n",
               i, KIND_NAME[L->kind], L->ic, L->ih, L->iw, L->oc, L->oh, L->ow,
               P[i].spread, tile, 100.0 * (double)s.exact / (double)s.total,
               s.maxdiff, P[i].clamped ? "  [bias saturated]" : "");
        if (s.maxdiff > PERAXIS_DEFECT_COUNTS) {
            printf("     past %d counts this is not the datapath's single shift\n",
                   PERAXIS_DEFECT_COUNTS);
            failed++;
        }
        free(pin); free(pw);
        cur = dst; if (!SKIP[i]) slot ^= 1;
    }
    logits = cur;

    if (!failed && logits) {
        int mine[5], tp[5];
        int k, rank = -1, agree = 1;
        top5(logits, (int)H->n_classes, mine);
        tflite_top5(tp);
        printf("\n   worst per-layer distance to an exact per-axis requant: %d count%s\n",
               worst_all, worst_all == 1 ? "" : "s");
        print_top5("per-axis top-5", mine);
        print_top5("TFLite top-5", tp);
        for (k = 0; k < 5; k++) if (mine[0] == tp[k]) { rank = k; break; }
        for (k = 0; k < 5; k++) if (mine[k] != tp[k]) agree = 0;
        if (rank < 0) {
            printf("   THE TOP-1 IS NOT IN TFLITE'S TOP-5 — the chain landed on a "
                   "different graph, not on a requant's worth of movement\n");
            failed++;
        } else if (rank == 0) {
            printf("   top-1 agrees%s\n", agree ? ", and so does the whole top-5"
                                                : " (the tail of the top-5 reorders)");
        } else {
            printf("   the top-1 is TFLite's own number %d, %d count%s ahead of TFLite's "
                   "top-1 in these logits — a rank swap inside the top-5, which is the "
                   "requant's accuracy and not an encoding defect (`hostchain` is the "
                   "instrument that separates the two)\n",
                   rank + 1, logits[mine[0]] - logits[tp[0]],
                   logits[mine[0]] - logits[tp[0]] == 1 ? "" : "s");
        }
    }

    free(a); free(b); free(ref);
    peraxis_free(P, H->n_layers);
    return failed;
}

/* ============================================================================
 * SECTION — the host chain, which is what separates an accuracy result from a
 * wiring defect
 *
 * A TABLE OF PER-LAYER DISTANCES CANNOT SAY WHY A GRAPH'S LABEL MOVED. A layer fed the
 * wrong tensor scores EXACT against a reference built from that same wrong tensor, so a
 * pass reporting one count at every layer beside a top-1 for an unrelated class is
 * equally consistent with "the requant costs the label" and with "some layer reads the
 * wrong producer" — and those want opposite responses.
 *
 * The discriminator runs the whole network on the CPU, where there is no part to blame
 * and the operand resolution is the only variable. Three arms:
 *
 *   PER-TENSOR, operands by producer   the POSITIVE CONTROL. This is the arithmetic every
 *                                      layer of the chain pass is bit-exact against, so it
 *                                      must return TFLite's own top-1. If it does not, the
 *                                      harness is wrong and neither other arm means
 *                                      anything.
 *   PER-AXIS,   operands by producer   the ANSWER. TFLite's top-1 here says the per-axis
 *                                      requant does not cost the label.
 *   PER-AXIS,   operand A threaded     the REPRODUCTION, and the arm that makes the
 *                                      conclusion a measurement. This is the ping-pong an
 *                                      NPU pass takes when it resolves operand A by
 *                                      position instead of by producer. Reproducing that
 *                                      pass's label here names the wiring; landing on
 *                                      TFLite's would say the wiring is not what moved it.
 *
 * It needs no device, so it also runs on a dev machine.
 * ==========================================================================*/

/* One layer, on the CPU. `P` non-NULL selects the per-axis weights, and is only ever
 * passed for a convolution — a pool has no weights and an add's are two diagonal blocks
 * rather than a quantized tensor, so both stay per-tensor in either arm, exactly as the
 * per-axis NPU pass leaves them. */
static int host_layer(const rnet_layer *L, const struct peraxis_layer *P,
                      const int8_t *in, const int8_t *in2, int8_t *out)
{
    if (L->kind == KIND_ADD) {
        int8_t *aw = NULL, *cat;
        int32_t *ab = NULL;
        float ws = 1.0f;
        rnet_layer R;
        int wp[2], rc;
        if (!in2) return ROCKET_E_SHAPE;
        rc = add_weights(L, &aw, &ab, &ws, wp);
        if (rc) { free(aw); free(ab); return rc < 0 ? ROCKET_E_NOMEM : rc; }
        cat = add_concat(L, in, in2);
        if (!cat) { free(aw); free(ab); return ROCKET_E_NOMEM; }
        add_ref_layer(L, ws, &R);
        host_conv(&R, cat, aw, ab, out);
        free(cat); free(aw); free(ab);
        return ROCKET_OK;
    }
    if (IS_POOL(L->kind)) {
        rocket_pool_desc p;
        pool_desc_of(L, &p);
        rocket_pool_ref_int8_rk3576(&p, L->in_zp, in, out);
        return ROCKET_OK;
    }
    if (P) peraxis_ref(L, P, in, out);
    else host_conv(L, in, at(L->w_off),
                   L->b_bytes ? (const int32_t *)(BLOB + L->b_off) : NULL, out);
    return ROCKET_OK;
}

/* The whole network on the CPU. `P` NULL is the per-tensor arm. `threaded` takes operand
 * A from the ping-pong rather than by a lookup on `src1`, and on the CONVOLUTION branch
 * alone — which is where an NPU pass takes it, its pooling and add branches going through
 * the same chain_input() the graph does. */
static const int8_t *host_chain(const struct peraxis_layer *P, int threaded,
                                int8_t *a, int8_t *b, int *failed)
{
    const int8_t *cur = at(H->img_off);
    int8_t *bufs[2];
    int slot = 0;
    unsigned i;

    bufs[0] = a; bufs[1] = b;
    for (i = 0; i < H->n_layers; i++) {
        const rnet_layer *L = &LAYERS[i];
        int weighted = (L->kind == KIND_CONV || L->kind == KIND_DWCONV);
        int8_t *dst = SKIP[i] ? SKIP[i] : bufs[slot];
        const int8_t *in;
        int rc;

        if (L->kind == KIND_SOFTMAX) continue;
        in = (threaded && weighted) ? cur : chain_input(i, cur);
        rc = host_layer(L, (P && weighted) ? &P[i] : NULL, in,
                        chain_operand_b(L, i, cur), dst);
        if (rc != ROCKET_OK) {
            printf("%2u %-8s THE HOST LAYER RETURNED %d\n", i, KIND_NAME[L->kind], rc);
            (*failed)++;
            return NULL;
        }
        cur = dst;
        if (!SKIP[i]) slot ^= 1;
    }
    return cur;
}

/* What one arm did to the label. `behind` is how far TFLite's own top-1 fell in THIS
 * arm's logits, and it is the number the arms are compared on — not the ranking.
 *
 * THE RANKING ALONE OVERSTATES THE ARITHMETIC, because these are int8 logits: an arm that
 * reorders a pair two counts apart has moved the label by about the smallest amount the
 * network can express, and an arm that puts the right class 120 counts down has landed on
 * a different graph. Both read as "the top-1 changed". */
struct arm_result {
    int moved;    /* top-1 differs from TFLite's                                     */
    int rank;     /* where this arm's top-1 sits in TFLite's top-5, -1 if outside     */
    int behind;   /* counts by which TFLite's top-1 trails this arm's, 0 if it IS it  */
};

static struct arm_result host_arm(const char *what, const struct peraxis_layer *P,
                                  int threaded, int8_t *a, int8_t *b, const int tf[5],
                                  int *failed)
{
    struct arm_result r = { 1, -1, 0 };
    const int8_t *lg = host_chain(P, threaded, a, b, failed);
    int mine[5], k;

    if (!lg) return r;
    top5(lg, (int)H->n_classes, mine);
    print_top5(what, mine);
    printf("   %-26s top-1 logit %d, runner-up %d, margin %d count%s\n", "",
           lg[mine[0]], lg[mine[1]], lg[mine[0]] - lg[mine[1]],
           lg[mine[0]] - lg[mine[1]] == 1 ? "" : "s");
    r.moved = mine[0] != tf[0];
    r.behind = lg[mine[0]] - lg[tf[0]];
    for (k = 0; k < 5; k++) if (mine[0] == tf[k]) { r.rank = k; break; }
    if (r.moved)
        printf("   %-26s TFLite's own top-1 sits at logit %d here, %d count(s) behind; "
               "this arm's top-1 is %s\n", "", lg[tf[0]], r.behind,
               r.rank >= 0 ? "inside TFLite's top-5" : "OUTSIDE TFLite's top-5");
    return r;
}

static int pass_hostchain(void)
{
    size_t cap = max_tensor();
    int8_t *a = malloc(cap), *b = malloc(cap);
    struct peraxis_layer *P = calloc(H->n_layers, sizeof *P);
    struct arm_result tensor, axis, thread;
    int tf[5], failed = 0;
    unsigned i;

    if (!a || !b || !P) {
        free(a); free(b); peraxis_free(P, H->n_layers);
        return 1;
    }
    printf("\n== HOST CHAIN: the same graph three ways on the CPU, with no part in it ==\n");
    printf("   ASSERTED: the per-tensor arm returns TFLite's top-1. The other two REPORT — "
           "they are what separates an accuracy result from an operand-wiring one\n");

    /* HOW MUCH OF THIS GRAPH THE THREADED ARM CAN EVEN SEE. On a feed-forward network the
     * two operand resolutions are the same function and the arm is a null control, which
     * is exactly why both MobileNets were blind to the per-axis pass reading by position.
     * The count says up front whether this network can distinguish them. */
    {
        unsigned far_conv = 0, far_other = 0;
        for (i = 0; i < H->n_layers; i++) {
            const rnet_layer *L = &LAYERS[i];
            if (L->kind == KIND_SOFTMAX || L->src1 == NO_SRC || L->src1 + 1u == i) continue;
            if (L->kind == KIND_CONV || L->kind == KIND_DWCONV) far_conv++;
            else far_other++;
        }
        printf("   %u convolution(s) and %u other layer(s) take operand A from a producer "
               "that is NOT the layer before%s\n", far_conv, far_other,
               far_conv ? "" : " — the threaded arm is a null control on this network");
    }

    for (i = 0; i < H->n_layers; i++) {
        const rnet_layer *L = &LAYERS[i];
        if (L->kind != KIND_CONV && L->kind != KIND_DWCONV) continue;
        if (peraxis_build(L, &P[i]) < 0) {
            free(a); free(b); peraxis_free(P, H->n_layers);
            return 1;
        }
    }

    tflite_top5(tf);
    print_top5("TFLite top-5", tf);
    tensor = host_arm("per-tensor, by producer", NULL, 0, a, b, tf, &failed);
    axis   = host_arm("per-axis, by producer",   P,    0, a, b, tf, &failed);
    thread = host_arm("per-axis, A threaded",    P,    1, a, b, tf, &failed);

    /* THE TWO CAUSES ARE NOT EXCLUSIVE and the magnitudes are what separate them, so both
     * are reported rather than one being chosen. */
    if (tensor.moved) {
        printf("   THE CONTROL ARM MOVED THE LABEL — the host chain itself is wrong and "
               "neither other arm can be read\n");
        failed++;
    } else {
        printf("   the requant is worth %d count%s of label movement (%s); threading "
               "operand A is worth %d (%s)\n",
               axis.behind, axis.behind == 1 ? "" : "s",
               !axis.moved ? "the top-1 does not move"
                           : axis.rank >= 0 ? "a rank swap inside TFLite's top-5"
                                            : "OUTSIDE TFLite's top-5",
               thread.behind,
               !thread.moved ? "the top-1 does not move"
                             : thread.rank >= 0 ? "a rank swap inside TFLite's top-5"
                                                : "OUTSIDE TFLite's top-5");
        if (thread.rank < 0 && axis.rank >= 0)
            printf("   so an NPU pass that reads operand A by POSITION rather than by "
                   "producer is what lands on an unrelated class here; the requant alone "
                   "does not leave the top-5\n");
    }

    free(a); free(b);
    peraxis_free(P, H->n_layers);
    return failed;
}

/*
 * THE CUBE CHAIN'S ASSERTION.
 *
 * Keeping a tensor in cube layout between two layers is a LAYOUT change and nothing else,
 * so what it owes is byte-identity: the same graph, the same 40 submits, the same surfaces.
 * The oracle pass already asserts each layer against a CPU model of the part's own
 * arithmetic, and it runs row-major — so the question left for the chain is only whether
 * skipping the two transposes at a join moved anything.
 *
 * Run the graph row-major, snapshot every layer, link the cubes, run it again, and compare
 * every layer the second run still materialises. A join's own layers produce no row-major
 * tensor to compare, but a wrong byte inside a join reaches the next materialisation point
 * and the final logits, both of which are compared — so nothing hides in the middle of a
 * chain.
 */
static int graph_once(int fd, int8_t **snap, const int8_t **logits, int8_t *a, int8_t *b)
{
    const int8_t *cur = at(H->img_off);
    int8_t *bufs[2];
    int slot = 0, failed = 0;
    unsigned i;

    bufs[0] = a; bufs[1] = b;
    for (i = 0; i < H->n_layers; i++) {
        const rnet_layer *L = &LAYERS[i];
        const char *how = "?";
        int8_t *dst = SKIP[i] ? SKIP[i] : bufs[slot];
        if (L->kind == KIND_SOFTMAX) continue;
        /* A CROSS-LAYER KICK covers layers i..KICK_END[i]-1 in one submit. Its last layer
         * leaves a cube by construction (that is what made the run), so there is no
         * row-major tensor at the far end and nothing to snapshot — pass_cube() skips those
         * layers for the same reason. */
        if (KICK_OF && KICK_OF[i]) {
            unsigned last = KICK_END[i] - 1u;
            /* The kick's output is its LAST layer's, so it lands in that layer's buffer
             * and not in the one the run's first layer would have used. */
            int8_t *kdst = SKIP[last] ? SKIP[last] : bufs[slot];
            if (rocket_conv2d_int8_chain_run_rk3576(fd, KICK_OF[i], chain_input(i, cur),
                    CUBE_OUT[last] ? NULL : kdst) != ROCKET_OK) { failed++; break; }
            if (!CUBE_OUT[last]) {
                if (snap && snap[last])
                    memcpy(snap[last], kdst, (size_t)LAYERS[last].oc * LAYERS[last].oh *
                           LAYERS[last].ow);
                cur = kdst;
                if (!SKIP[last]) slot ^= 1;
            }
            i = last;
            continue;
        }
        if (layer_run(fd, L, chain_input(i, cur), chain_operand_b(L, i, cur), dst, &how) != ROCKET_OK) {
            failed++; break;
        }
        if (CUBE_OUT && CUBE_OUT[i]) continue;      /* no row-major tensor to keep */
        if (snap && snap[i])
            memcpy(snap[i], dst, (size_t)L->oc * L->oh * L->ow);
        cur = dst;
        if (!SKIP[i]) slot ^= 1;
    }
    if (logits) *logits = cur;
    return failed;
}

static int pass_cube(int fd)
{
    size_t cap = max_tensor();
    int8_t *a = malloc(cap), *b = malloc(cap);
    int8_t **snap = calloc(H->n_layers, sizeof *snap);
    int8_t **snap2 = calloc(H->n_layers, sizeof *snap2);
    int8_t **snap3 = calloc(H->n_layers, sizeof *snap3);
    /* The logits are KEPT, not pointed at. graph_once() returns a pointer into the two
     * ping-pong buffers it was handed, and the next run rewrites both — so holding the
     * pointer compares whatever the LATER run happened to leave there. Which buffer the
     * last layer lands in is a function of how many layers materialised an output, so a
     * lever that changes the join count silently changes what is being compared: before
     * this was a copy, adding one cube join flipped the parity and turned a comparison
     * that had been a buffer against ITSELF into one against a mid-graph tensor. */
    int8_t *lb1 = NULL, *lb2 = NULL, *lb3 = NULL;
    const int8_t *l1 = NULL, *l2 = NULL;
    int failed = 0, compared = 0;
    unsigned i;

    printf("\n== CUBE CHAIN: the same graph with the transposes removed at every join ==\n");
    if (!CUBE_ON) {
        printf("   OFF — ROCKET_RK3576_NET_CUBE=1 with ROCKET_RK3576_NET_RESIDENT=1 "
               "turns it on; the flags live on the resident handle\n");
        free(a); free(b); free(snap); free(snap2); free(snap3);
        return 0;
    }
    if (!a || !b || !snap || !snap2 || !snap3) { failed = 1; goto out; }
    for (i = 0; i < H->n_layers; i++) {
        size_t n = (size_t)LAYERS[i].oc * LAYERS[i].oh * LAYERS[i].ow;
        if (LAYERS[i].kind == KIND_SOFTMAX) continue;
        snap[i] = malloc(n); snap2[i] = malloc(n);
        if (!snap[i] || !snap2[i]) { failed = 1; goto out; }
    }

    lb1 = malloc(H->n_classes); lb2 = malloc(H->n_classes); lb3 = malloc(H->n_classes);
    if (!lb1 || !lb2 || !lb3) { failed = 1; goto out; }

    /* Row-major first, which also packs every resident handle the link pass then reads. */
    failed += graph_once(fd, snap, &l1, a, b);
    if (failed) goto out;
    if (l1) { memcpy(lb1, l1, H->n_classes); l1 = lb1; }

    cube_link(fd);
    if (!CUBE_JOINS) {
        printf("   no join was accepted, so there is nothing to compare — this is a "
               "REFUSAL to explain, not a pass\n");
        failed++;
        goto out;
    }

    failed += graph_once(fd, snap2, &l2, a, b);
    if (failed) goto out;
    if (l2) { memcpy(lb2, l2, H->n_classes); l2 = lb2; }

    for (i = 0; i < H->n_layers; i++) {
        size_t n = (size_t)LAYERS[i].oc * LAYERS[i].oh * LAYERS[i].ow;
        if (LAYERS[i].kind == KIND_SOFTMAX) continue;
        if (CUBE_OUT[i]) continue;
        compared++;
        if (!memcmp(snap[i], snap2[i], n)) continue;
        {
            size_t k, diff = 0, first = 0;
            for (k = 0; k < n; k++)
                if (snap[i][k] != snap2[i][k]) { if (!diff) first = k; diff++; }
            printf("%2u %-8s the cube chain DISAGREES with the row-major one: %zu of %zu "
                   "elements, first at %zu (%d vs %d)\n",
                   i, KIND_NAME[LAYERS[i].kind], diff, n, first,
                   snap[i][first], snap2[i][first]);
            failed++;
        }
    }
    printf("   %d join(s); %d materialised layer(s) compared byte for byte, %s\n",
           CUBE_JOINS, compared, failed ? "and some DIFFER" : "all identical");
    if (l1 && l2 && memcmp(l1, l2, H->n_classes)) {
        printf("   the LOGITS differ\n");
        failed++;
    } else if (l1 && l2) {
        int m1[5], m2[5], k;
        top5(l1, (int)H->n_classes, m1);
        top5(l2, (int)H->n_classes, m2);
        for (k = 0; k < 5; k++) if (m1[k] != m2[k]) failed++;
        printf("   the logits are byte-identical and the top-5 is unchanged\n");
    }
    if (failed) goto out;

    /* ---- THE CROSS-LAYER KICK, against the cube-chained graph it changes the SUBMIT of ----
     *
     * Its own A/B and not folded into the one above, because the two levers change different
     * things: the cube chain changes what the host does between layers, the kick changes how
     * many jobs those layers are. Comparing the kick against the ROW-MAJOR run would prove
     * both at once and neither on its own, and a wrong byte inside a kick would read as a
     * cube-chain defect. */
    printf("\n== CROSS-LAYER KICK: the same cube-chained graph as fewer hardware kicks ==\n");
    kick_build(fd);
    if (!KICK_RUNS) {
        if (kick_on()) {
            printf("   no run of two or more linked layers was accepted — this is a "
                   "REFUSAL to explain, not a pass\n");
            failed++;
        }
        goto out;
    }
    {
        const int8_t *l3 = NULL;
        unsigned kicks = 0, r;
        for (i = 0; i < H->n_layers; i++) {
            size_t n = (size_t)LAYERS[i].oc * LAYERS[i].oh * LAYERS[i].ow;
            if (LAYERS[i].kind == KIND_SOFTMAX) continue;
            snap3[i] = malloc(n);
            if (!snap3[i]) { failed = 1; goto out; }
        }
        failed += graph_once(fd, snap3, &l3, a, b);
        if (failed) goto out;
        if (l3) { memcpy(lb3, l3, H->n_classes); l3 = lb3; }
        compared = 0;
        for (i = 0; i < H->n_layers; i++) {
            size_t n = (size_t)LAYERS[i].oc * LAYERS[i].oh * LAYERS[i].ow;
            if (LAYERS[i].kind == KIND_SOFTMAX) continue;
            if (CUBE_OUT[i]) continue;
            compared++;
            if (!memcmp(snap2[i], snap3[i], n)) continue;
            {
                size_t k, diff = 0, first = 0;
                for (k = 0; k < n; k++)
                    if (snap2[i][k] != snap3[i][k]) { if (!diff) first = k; diff++; }
                printf("%2u %-8s the kicked graph DISAGREES with the per-layer one: %zu of "
                       "%zu elements, first at %zu (%d vs %d)\n",
                       i, KIND_NAME[LAYERS[i].kind], diff, n, first,
                       snap2[i][first], snap3[i][first]);
                failed++;
            }
        }
        /* What the runs COST in kicks, read back rather than assumed: a chain that had to
         * redo itself took more than one, and a lever measured against an assumed count is
         * measured against nothing. */
        for (r = 0; r < H->n_layers; r++)
            if (KICK_OF[r]) kicks += rocket_conv2d_int8_chain_kicks_rk3576(KICK_OF[r]);
        printf("   %d run(s) over %d layer(s), %u kick(s) for them on the last inference; "
               "%d materialised layer(s) compared byte for byte, %s\n",
               KICK_RUNS, KICK_LAYERS, kicks, compared,
               failed ? "and some DIFFER" : "all identical");
        if (l2 && l3 && memcmp(l2, l3, H->n_classes)) {
            printf("   the LOGITS differ\n");
            failed++;
        } else if (l2 && l3) {
            printf("   the logits are byte-identical to the per-layer cube-chained graph\n");
        }
    }

out:
    for (i = 0; i < H->n_layers; i++) { free(snap[i]); free(snap2[i]); free(snap3[i]); }
    free(snap); free(snap2); free(snap3); free(a); free(b);
    free(lb1); free(lb2); free(lb3);
    return failed;
}

int main(int argc, char **argv)
{
    const char *blob = getenv("ROCKET_NET_BLOB");
    const char *net = getenv("ROCKET_NET") ? getenv("ROCKET_NET") : "v1";
    const char *mode = "all";
    int fd, a, iters = 1, failed = 0;
    char def[512];

    setvbuf(stdout, NULL, _IOLBF, 0);
    for (a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--blob") && a + 1 < argc) blob = argv[++a];
        else if (!strcmp(argv[a], "--net") && a + 1 < argc) net = argv[++a];
        else if (!strcmp(argv[a], "-v")) VERBOSE = 1;
        else if (a + 1 < argc && !strcmp(argv[a], "bench")) {
            mode = "bench"; iters = atoi(argv[++a]);
            if (iters < 1) iters = 1;
        } else mode = argv[a];
    }
    if (!blob) {
        const char *root = getenv("ROCKET_SRC_DIR");
        const char *stem = !strcmp(net, "r18") ? "resnet18" : NULL;
        if (stem)
            snprintf(def, sizeof def, "%s/tests/data/rk3576-net/%s_224_quant.rnet",
                     root ? root : ".", stem);
        else
            snprintf(def, sizeof def,
                     "%s/tests/data/rk3576-net/mobilenet_%s_224_quant.rnet",
                     root ? root : ".", net);
        blob = def;
    }

    if (load_blob(blob) < 0) {
        printf("no blob at %s — SKIP\n"
               "  build it: cd tests/data/rk3576-net && ./fetch.sh && python3 mknet.py\n",
               blob);
        return 2;
    }

    /* THE HOST CHAIN NEEDS NO DEVICE, so it is dispatched before one is opened: it runs
     * on a dev machine, and on a board it cannot be confounded by the part. */
    if (!strcmp(mode, "hostchain")) {
        printf("%s: %u layers, %ux%ux%u in, %u classes, %zu bytes\n",
               blob, H->n_layers, H->in_c, H->in_h, H->in_w, H->n_classes, BLOB_BYTES);
        if (skip_alloc() < 0) { printf("out of memory for the skip buffers\n"); return 1; }
        failed = pass_hostchain();
        printf("\n%s: %d failure(s)\n", failed ? "FAIL" : "PASS", failed);
        skip_free();
        free(BLOB);
        return failed ? 1 : 0;
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
        /* A fused activation is only free while its clamp IS the int8 range. If a model
         * ever arrives where it is not, it needs a host clamp and this gate would
         * otherwise report a wrong layer with no reason attached. */
        unsigned i, clamped = 0;
        for (i = 0; i < H->n_layers; i++)
            if (LAYERS[i].act != ACT_NONE &&
                (LAYERS[i].clamp_lo != -128 || LAYERS[i].clamp_hi != 127)) clamped++;
        if (clamped) {
            printf("%u layer(s) carry a fused activation whose clamp is NARROWER than "
                   "the int8 range; this gate does not apply one\n", clamped);
            failed++;
        }
    }

    /* The extent form's own contract, once: the pad the CNA DERIVES from the output extent
     * and the leading pad has to be the trailing pad the model asks for. If it is not, the
     * lowering is wrong before any hardware runs and every layer would report a distance
     * with no reason attached. */
    if (asympad_on()) {
        unsigned i, asym = 0, bad = 0;
        for (i = 0; i < H->n_layers; i++) {
            const rnet_layer *L = &LAYERS[i];
            int ty, tx;
            rocket_conv2d_desc d;
            if (L->kind != KIND_CONV && L->kind != KIND_DWCONV) continue;
            memset(&d, 0, sizeof d);
            d.ic = (int)L->ic; d.ih = (int)L->ih; d.iw = (int)L->iw;
            d.oc = (int)L->oc; d.kh = (int)L->kh; d.kw = (int)L->kw;
            d.stride_y = (int)L->sy; d.stride_x = (int)L->sx;
            d.dil_y = d.dil_x = 1;
            d.pad_top = (int)L->pl_y; d.pad_left = (int)L->pl_x;
            d.oh = (int)L->oh; d.ow = (int)L->ow;
            ty = rocket_conv2d_trail_y(&d); tx = rocket_conv2d_trail_x(&d);
            if (L->pl_y != L->pt_y || L->pl_x != L->pt_x) asym++;
            if (ty != (int)L->pt_y || tx != (int)L->pt_x) {
                printf("%2u %-8s the extent %ux%u over a %ux%u plane derives a trailing pad "
                       "of %d/%d and the model asks for %u/%u\n",
                       i, KIND_NAME[L->kind], L->oh, L->ow, L->ih, L->iw, ty, tx,
                       L->pt_y, L->pt_x);
                bad++; failed++;
            }
        }
        printf("asymmetric pad: as an OUTPUT EXTENT — %u layer(s) pad asymmetrically and "
               "none materialise a border; the derived trailing pad matches the model on "
               "%u of %u compute layer(s)\n",
               asym, H->n_layers - bad, H->n_layers);
    } else {
        printf("asymmetric pad: MATERIALISED into a row-major buffer (the A/B control)\n");
    }

    /* The skip buffers, before any pass: which layers need one is a property of the
     * GRAPH, and every pass that runs the network for real reads them. */
    if (skip_alloc() < 0) {
        printf("out of memory for the skip buffers\n");
        rocket_close(fd);
        return 1;
    }

    resident_init();
    if (RESIDENT_ON)
        printf("resident weights: ON — each layer's filter sums, coefficient group and "
               "weight cube are packed once and held\n");

    if (CUBE_ON)
        printf("cube chain: ON — the linking happens after the row-major passes, because "
               "the flags are frozen on the handle once set\n");

    if (!strcmp(mode, "all")) failed += pass_hostchain();
    if (!strcmp(mode, "oracle") || !strcmp(mode, "all")) failed += pass_oracle(fd);
    if (!strcmp(mode, "chain") || !strcmp(mode, "all")) failed += pass_chain(fd, 1);
    if (!strcmp(mode, "peraxis") || !strcmp(mode, "all")) failed += pass_peraxis(fd);
    /* LAST in `all`, and its own mode otherwise: the cube flags are frozen on the handles
     * the earlier passes drive row-major, so linking before them would leave those passes
     * with no tensor to score. */
    if (!strcmp(mode, "cube") || !strcmp(mode, "all")) failed += pass_cube(fd);
    if (!strcmp(mode, "bench")) {
        if (CUBE_ON) {
            /* One warm-up to pack the handles the link pass reads, then the joins, then
             * the measurement — a `bench` whose first iteration also did the linking would
             * put the pack inside the average. */
            const int8_t *lg = NULL;
            int8_t *wa = malloc(max_tensor()), *wb = malloc(max_tensor());
            if (!wa || !wb) failed++;
            else { failed += graph_once(fd, NULL, &lg, wa, wb); cube_link(fd);
                   kick_build(fd); }
            free(wa); free(wb);
        }
        failed += pass_chain(fd, iters);
    }

    printf("\n%s: %d failure(s)\n", failed ? "FAIL" : "PASS", failed);
    resident_free(fd);
    skip_free();
    rocket_close(fd);
    free(BLOB);
    return failed ? 1 : 0;
}
