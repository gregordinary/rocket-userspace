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
 *         [--net v1|v2|r18|iv1] [--blob PATH] [-v]
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
#include "rocket_graph_rk3576.h"

/* ---- the blob ------------------------------------------------------------------ */
enum { KIND_CONV = 0, KIND_DWCONV, KIND_AVGPOOL, KIND_SOFTMAX, KIND_ADD, KIND_MAXPOOL,
       KIND_CONCAT };
enum { ACT_NONE = 0, ACT_RELU6, ACT_RELU };

/* Both pooling kinds are the same PPU program with a different reduction, so every site
 * that asks "is this layer the pool" means both. */
#define IS_POOL(k) ((k) == KIND_AVGPOOL || (k) == KIND_MAXPOOL)

/* A layer's operands are named by the LAYER that produced them rather than assumed to be
 * the one before. A feed-forward chain does not need that and a residual one does: a
 * skip's second operand is produced three to five layers back. NO_SRC is the network
 * input, and past `src1` it means the layer has no operand there.
 *
 * FOUR of them, because a CONCATENATION has four: an Inception module is a 1x1, a 3x3, a
 * 5x5 and a pooled branch joined along the channel axis. Every kind reads the same array
 * and every pass resolves an operand the same way — a resolution written per BRANCH gets
 * applied to some of them, which has already cost this gate one wrong label. */
#define NO_SRC 0xFFFFFFFFu
#define MAX_SRC 4

typedef struct {
    uint32_t kind, act, ic, ih, iw, oc, oh, ow, kh, kw, sy, sx;
    uint32_t pl_y, pl_x, pt_y, pt_x;
    int32_t  in_zp, w_zp, out_zp, clamp_lo, clamp_hi;
    float    in_scale, w_scale, out_scale;
    uint32_t w_off, w_bytes, b_off, b_bytes, g_off, g_bytes;
    uint32_t src1, src2;
    int32_t  in2_zp;
    float    in2_scale;
    uint32_t src3, src4;
    uint32_t pad_[4];
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
                                   "maxpool", "concat" };

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
    /* TFLite's AVERAGE_POOL_2D divides a border window by the taps that fell inside the
     * plane, so this is the model's arithmetic and not a choice. It is set for every
     * average pool rather than only the padded ones: at pad 0 the two divisors are the
     * same function, which `rk3576_pool_probe`'s avg-nopad-p0 cell asserts on hardware,
     * so stating the lowering once is better than a conditional that hides the axis. */
    /* ROCKET_RK3576_NET_AVGNOPAD=0 puts the count-include-pad divisor back. It is the
     * control that says whether a defect seen on a padded average pool is the mode bit's
     * or the pooling path's — the model follows the descriptor either way, so both arms
     * are scored against the arithmetic they asked for. */
    p->avg_exclude_pad = L->kind == KIND_AVGPOOL &&
                         !(getenv("ROCKET_RK3576_NET_AVGNOPAD") &&
                           *getenv("ROCKET_RK3576_NET_AVGNOPAD") == '0');
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
    if (memcmp(H->magic, "RKNET\0\0\1", 8) || H->version != 3 ||
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

/*
 * THE PACKED-IMAGE FIRST CONV, ASKED RATHER THAN PREDICTED.
 *
 * A three-channel image on the direct datapath is a 32-channel MAC group, so the stem
 * programs about 10.6x the MACs it needs and a convolution's execution is its PROGRAMMED
 * MAC count. The packed-image sub-encoding folds the kernel's columns into the channel
 * axis instead — 8.0x fewer MACs, 1.24 ms against the direct lowering's 3.09 at
 * 224x224 k7 s2 as a whole resident call [HW sweep, H96 MAX M9, rk3576_stem_cost].
 *
 * WHICH GRAPHS IT REACHES IS A MEASUREMENT, NOT AN ARITHMETIC. Its bounds are the
 * library's — a non-zero left pad, an output extent of exactly iw/stride, ow a multiple
 * of 16, a zero input zero point — and this gate does not restate them, because a bound
 * restated in a harness reads exactly like the part's and goes stale the same way. The
 * decision is taken by ATTEMPTING the pack: a refusal is printed with the library's own
 * reason and the layer falls back to the direct lowering, which is exact everywhere.
 * ARGB[] caches that answer so every pass and every inference takes the same path.
 *
 * ROCKET_RK3576_NET_ARGB=0 forces the direct lowering, which is the A/B: the two must
 * agree byte for byte over the whole surface.
 *
 * A MATERIALISING STEM IS REACHABLE AND COSTS A JOIN, so this gate does not take it while
 * the graph is cube-linked. At a non-zero input zero point the entry materialises the pad
 * columns, which makes the encoding EXACT on ResNet-18's stem and 1.41 ms against the
 * direct lowering's 3.13 as a resident per-op call — and its surface is then wider than the
 * caller's plane, so it cannot be a consumer's cube. Measured on the graph at `bench 100`,
 * three repeats each: 9.3-9.5 ms in 3 submits against the direct stem's 9.2-9.3 in 1. The
 * program's 1.7 ms goes back out at the break, and the break ejects TWO layers rather than
 * one — the stem de-scatters 784 KiB, the max pool scatters the same 784 KiB back, and a
 * run cannot OPEN on a pool, so both fall outside the kick (stem 1.9, pool 1.4, kick 6.0
 * against one kick of 9.2).
 *
 * So the rule is the frontend's, not the part's: take the packed encoding where it does not
 * break a join, which is a stem whose input zero point is zero. ROCKET_RK3576_NET_ARGB=2
 * takes it anyway and is the arm those numbers come from.
 */
static signed char *ARGB;                 /* -1 not asked, 0 direct, 1 packed */
static int CUBE_ON;                       /* set by resident_init(); see cube_link()  */

static int argb_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_NET_ARGB");
        cached = !e || !*e ? 1 : (int)strtol(e, NULL, 0);
    }
    return cached;
}

/* Whether this layer is a candidate for the packed-image encoding at all — the shape
 * question only; whether the part takes it is the pack's answer. */
static int argb_cand(const rnet_layer *L)
{
    if (!argb_on() || L->kind != KIND_CONV || L->ic > 4 || widen_on()) return 0;
    /* THE MATERIALISING FORM, AND WHO CAN READ WHAT IT WRITES. It programs a wider output
     * extent than the caller asked for, so its surface is a plane sitting inside wider
     * rows. A POOL consumer reads that in place — the PPU carries the consumed extent and
     * the DDR line stride in different registers and honours a pitch above the extent — and
     * a CONVOLUTION consumer cannot, the CNA having one register for both. So the encoding
     * is taken where a pool reads it and left alone where a convolution would, which is a
     * property of the GRAPH rather than a bound of the part's; the library refuses the cube
     * either way, so a wrong answer here costs a join and not correctness. */
    if (L->in_zp != 0 && CUBE_ON && argb_on() < 2) {
        unsigned li = (unsigned)(L - LAYERS), j;
        if (!LAYERS || !H) return 0;
        for (j = li + 1u; j < H->n_layers && LAYERS[j].kind == KIND_SOFTMAX; j++) ;
        if (j >= H->n_layers || !IS_POOL(LAYERS[j].kind)) return 0;
    }
    return 1;
}

/* Whether this layer reaches the direct datapath at a channel count that would otherwise
 * be routed to the packed-image first conv. A candidate whose pack has not been attempted
 * yet reads as direct, which is the safe answer: ARGB[] is filled by the first pass to
 * build a handle and every later reader sees the decision it took. */
static int narrow_direct(const rnet_layer *L)
{
    unsigned li;
    if (L->kind != KIND_CONV || L->ic > 4 || widen_on()) return 0;
    if (!argb_cand(L) || !ARGB) return 1;
    li = (unsigned)(L - LAYERS);
    return ARGB[li] != 1;
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
/* A layer's operands, in order, as LAYER indices — one for a convolution or a pool, two
 * for an add, up to four for a concatenation. NO_SRC (the network input, or an unused
 * slot) is dropped, so the count is the number of LAYER operands. */
static unsigned srcs_of(const rnet_layer *L, unsigned *out)
{
    const uint32_t s[MAX_SRC] = { L->src1, L->src2, L->src3, L->src4 };
    unsigned k, n = 0;
    for (k = 0; k < MAX_SRC; k++)
        if (s[k] != NO_SRC) out[n++] = s[k];
    return n;
}

/* A layer read by anything other than the layer immediately after it — the planner's own
 * query, over this blob's operand graph, so the rule has one home. */
static int skip_is_source(unsigned i)
{
    unsigned j;
    for (j = i + 1; j < H->n_layers; j++) {
        unsigned s[MAX_SRC], n, k;
        if (LAYERS[j].kind == KIND_SOFTMAX) continue;
        n = srcs_of(&LAYERS[j], s);
        for (k = 0; k < n; k++)
            if (s[k] == i && j != i + 1) return 1;
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

/* WHERE ONE OPERAND LIVES, for any layer and any pass. The oracle hands a layer TFLite's
 * own tensor; every other pass reads what the graph actually wrote — the producer's own
 * buffer when it has one, the ping-pong when the producer is the layer immediately
 * before. NULL when the graph has not kept it, which is a harness bug rather than a
 * result and is reported as one. */
static const int8_t *operand_at(unsigned i, unsigned src, int from_golden,
                                const int8_t *cur)
{
    if (from_golden) return at(LAYERS[src].g_off);
    if (SKIP && SKIP[src]) return SKIP[src];
    return src + 1u == i ? cur : NULL;
}

/* Every operand of layer i, resolved the same way. Returns the count, or 0 if any of
 * them is missing. */
static unsigned gather_ops(unsigned i, int from_golden, const int8_t *cur,
                           const int8_t **ops)
{
    unsigned s[MAX_SRC], n, k;
    n = srcs_of(&LAYERS[i], s);
    for (k = 0; k < n; k++) {
        ops[k] = operand_at(i, s[k], from_golden, cur);
        if (!ops[k]) return 0;
    }
    return n;
}

/* A CONCATENATION IS PLACEMENT. Its operands are already in its own quantization — the
 * blob builder refuses a model where they are not — so there is no arithmetic here at
 * all: on the host one memcpy per operand into its channel offset, and on the part
 * nothing whatever, because the producers wrote those bytes themselves.
 *
 * The offsets come from the PRODUCERS' channel counts in operand order, which is the one
 * thing the blob does not spell out and the one thing a wrong answer here would be. */
static void concat_run(const rnet_layer *L, const int8_t *const *ops, int8_t *out)
{
    size_t px = (size_t)L->oh * L->ow;
    unsigned s[MAX_SRC], n, k, off = 0;
    n = srcs_of(L, s);
    for (k = 0; k < n; k++) {
        unsigned c = LAYERS[s[k]].oc;
        memcpy(out + (size_t)off * px, ops[k], (size_t)c * px);
        off += c;
    }
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
    return rocket_graph_add_boff(L->oc);
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
 * different handle type. */
static rocket_pool_int8_rk3576_handle **RESIDENT_POOL;

/*
 * THE PLANNER IS NOT THIS GATE'S.
 *
 * Which tensors stay in cube layout, which producers write slices of a shared buffer and
 * which runs of layers go out as one hardware kick are decided in graph/, over a
 * frontend-neutral layer description — because a delegate needs exactly the same rules and
 * a second copy of them would fork silently. What stays here is what a CALLER owns: the
 * weights, the tensors, the ping-pong, the skip lifetimes and the de-scatter points.
 *
 * GRAPH[] is this blob's layers in that description. It is built once, and its handle
 * fields are refreshed just before the plan is built — the handles are packed lazily on a
 * layer's first call, so a plan built before a warm-up inference would see none of them.
 */
static rocket_graph_layer *GRAPH;
static rocket_graph_plan *PLAN;
static int CUBE_ON;                       /* set by resident_init(); see plan_build() */

/* The plan's arrays under the names every pass of this gate already reads. NULL before the
 * plan is built, which is the state a row-major run stays in. */
#define CUBE_IN     (PLAN ? PLAN->cube_in  : NULL)
#define CUBE_OUT    (PLAN ? PLAN->cube_out : NULL)
#define KICK_OF     (PLAN ? PLAN->kick     : NULL)
#define KICK_END    (PLAN ? PLAN->kick_end : NULL)
#define CUBE_JOINS  (PLAN ? PLAN->joins       : 0)
#define KICK_RUNS   (PLAN ? PLAN->kick_runs   : 0)
#define KICK_LAYERS (PLAN ? PLAN->kick_layers : 0)

static rocket_graph_kind graph_kind_of(uint32_t k)
{
    switch (k) {
    case KIND_CONV:    return ROCKET_GRAPH_CONV;
    case KIND_DWCONV:  return ROCKET_GRAPH_DWCONV;
    case KIND_AVGPOOL: return ROCKET_GRAPH_AVGPOOL;
    case KIND_MAXPOOL: return ROCKET_GRAPH_MAXPOOL;
    case KIND_ADD:     return ROCKET_GRAPH_ADD;
    case KIND_CONCAT:  return ROCKET_GRAPH_CONCAT;
    default:           return ROCKET_GRAPH_HOST;   /* the softmax, run on the host */
    }
}

/* The blob's layers as the planner's description. The geometry and the operand graph are
 * the blob's; `host_input` is this gate's own lowering, and it is the one field no rule in
 * graph/ could derive — a layer whose input this gate materialises (an asymmetric border,
 * a widened or shifted stem) hands the entry a tensor that is not its producer's output. */
static int graph_desc_build(void)
{
    unsigned i, k;
    GRAPH = calloc(H->n_layers, sizeof *GRAPH);
    if (!GRAPH) return -1;
    for (i = 0; i < H->n_layers; i++) {
        const rnet_layer *L = &LAYERS[i];
        struct stem_plan sp;
        rocket_graph_layer *G = &GRAPH[i];
        const uint32_t s[MAX_SRC] = { L->src1, L->src2, L->src3, L->src4 };
        G->kind = graph_kind_of(L->kind);
        G->ic = L->ic; G->oc = L->oc; G->oh = L->oh; G->ow = L->ow;
        G->w_zp = L->w_zp; G->out_zp = L->out_zp;
        for (k = 0; k < MAX_SRC; k++)
            G->src[k] = s[k] == NO_SRC ? ROCKET_GRAPH_NO_SRC : s[k];
        G->host_input = needs_prep(L) || stem_plan_of(L, &sp);
    }
    return 0;
}

static void graph_desc_handles(void)
{
    unsigned i;
    if (!GRAPH) return;
    for (i = 0; i < H->n_layers; i++) {
        GRAPH[i].conv = RESIDENT ? RESIDENT[i] : NULL;
        GRAPH[i].pool = RESIDENT_POOL ? RESIDENT_POOL[i] : NULL;
    }
}

/* Build the plan: the links only. The runs are a second call, so the two are measurable
 * apart — pass_cube() runs the graph between them. */
static void plan_build(int fd)
{
    if (!CUBE_ON || PLAN) return;
    graph_desc_handles();
    PLAN = rocket_graph_plan_new(fd, GRAPH, H->n_layers, 1);
}

static void resident_init(void)
{
    const char *e = getenv("ROCKET_RK3576_NET_RESIDENT");
    /* Allocated whether or not the weights are resident: the decision is about the
     * ENCODING, and a transient run takes it too. */
    if (argb_on()) {
        ARGB = malloc(H->n_layers);
        if (ARGB) memset(ARGB, -1, H->n_layers);
    }
    RESIDENT_ON = e && *e && *e != '0';
    if (RESIDENT_ON) RESIDENT = calloc(H->n_layers, sizeof *RESIDENT);
    if (RESIDENT_ON) RESIDENT_POOL = calloc(H->n_layers, sizeof *RESIDENT_POOL);
    if (RESIDENT_ON && (!RESIDENT || !RESIDENT_POOL)) RESIDENT_ON = 0;

    e = getenv("ROCKET_RK3576_NET_CUBE");
    CUBE_ON = RESIDENT_ON && e && *e && *e != '0';
    if (CUBE_ON && graph_desc_build() < 0) CUBE_ON = 0;
}

static void resident_free(int fd)
{
    unsigned i;
    free(ARGB); ARGB = NULL;
    /* The plan BORROWS the handles for its chains, so it goes first. */
    rocket_graph_plan_free(PLAN);
    PLAN = NULL;
    free(GRAPH); GRAPH = NULL;
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
    CUBE_ON = 0;
}

/* Run one layer. `how` is set to the path it took. `in2` is the second operand of an add
 * and NULL everywhere else. */
static int layer_run(int fd, const rnet_layer *L, const int8_t *in, const int8_t *in2,
                     const int8_t *const *ops, int8_t *out, const char **how)
{
    const int8_t *W = L->w_bytes ? at(L->w_off) : NULL;
    const int32_t *bias = L->b_bytes ? (const int32_t *)(BLOB + L->b_off) : NULL;
    rocket_conv2d_desc d;
    int8_t *pin = NULL, *pw = NULL;
    unsigned icx = L->ic, ihx = L->ih, iwx = L->iw;
    int rc;

    if (L->kind == KIND_CONCAT) {
        unsigned li = (unsigned)(L - LAYERS);
        /* WIRED, there is nothing to do at all: the four producers wrote their own slices
         * of the buffer the next layer reads as its cube, so this layer is a name for an
         * address. Otherwise it is the host copy, which is the control that lever is
         * measured against. */
        if (CUBE_OUT && CUBE_OUT[li]) { *how = ">concat<"; return ROCKET_OK; }
        if (!ops || !ops[0]) { *how = "NO-OPERAND"; return ROCKET_E_SHAPE; }
        concat_run(L, ops, out);
        *how = "host-concat";
        return ROCKET_OK;
    }

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
        /* PAST THE PER-TASK OUTPUT-WIDTH ALLOWANCE THE PART IS SILENTLY WRONG, so the
         * plan function refuses and the layer runs on the CPU model of the part's own
         * arithmetic — which keeps the graph's numbers identical to what the part would
         * have computed, so a fallback shows up as a submit and never as a divergence.
         * Inception V3 is the first graph here to have one: its 147-wide max pool and
         * its three 35-wide branch averages. */
        if (rocket_pool_int8_rk3576_plan(&p) != ROCKET_OK) {
            *how = "host-pool";
            rocket_pool_ref_int8_rk3576(&p, L->in_zp, in, out);
            return ROCKET_OK;
        }
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

    /* THE PACKED-IMAGE STEM IS TRIED HERE, and once. `direct_datapath` clear is what routes
     * an ic <= 4 convolution to it; the pack answers with the part's own bounds, and a
     * refusal leaves this layer on the direct lowering for the rest of the run. */
    if (ARGB && argb_cand(L) && !pin) {
        unsigned li = (unsigned)(L - LAYERS);
        if (ARGB[li] < 0) {
            rocket_conv2d_desc t = d;
            rocket_conv2d_int8_weights_rk3576 *h;
            t.direct_datapath = 0;
            h = rocket_conv2d_int8_pack_rk3576(fd, &t, W, bias, L->in_scale, L->w_scale,
                                               NULL, L->out_scale, L->in_zp, L->w_zp,
                                               L->out_zp);
            ARGB[li] = h ? 1 : 0;
            printf("   layer %u stem (ic=%u %ux%u k%u s%u pad %u,%u): the packed-image "
                   "first conv %s\n", li, L->ic, L->ih, L->iw, L->kh, L->sy,
                   L->pl_y, L->pl_x,
                   h ? "TAKES this geometry" : "refused it — the direct lowering runs");
            if (h) {
                if (RESIDENT_ON && !RESIDENT[li]) RESIDENT[li] = h;
                else rocket_conv2d_int8_weights_free_rk3576(fd, h);
            }
        }
        d.direct_datapath = !ARGB[li];
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

/* ===========================================================================
 * WHICH DIVISOR A PADDED AVERAGE POOL USED WHEN IT USED THE WRONG ONE.
 *
 * A wrong-element count says a pooling layer disagreed; it cannot say what function the
 * part evaluated instead, and on this part the two candidates differ only in a per-window
 * DIVISOR. So a failing pool is scored against three rivals as well as its own model:
 *
 *   incl    count-include-pad, the other mode bit
 *   lag-x   the in-plane COLUMN count taken one output column late
 *   lag-y   the same one output ROW late
 *
 * The two `lag` models are the shape the failures have, not a guess about a mechanism: a
 * wrong set confined to output columns 1 and ow-1 is a column count that is right
 * everywhere except one column early and one column late.
 * ==========================================================================*/
static int pool_inplane(int o, int stride, int k, int pad, int extent)
{
    int lo, hi;
    if (o < 0) o = 0;
    lo = o * stride - pad;
    hi = lo + k;
    if (lo < 0) lo = 0;
    if (hi > extent) hi = extent;
    return hi > lo ? hi - lo : 0;
}

/* rocket_pool_ref_int8_rk3576()'s own rounding over a divisor pair this caller chooses,
 * including the wrap to int8 with no saturation — which is what makes a divisor of 6
 * where 9 was meant read as +79 rather than as a clamp. */
static int8_t pool_round(long sum, int dw, int dh)
{
    long n = (long)dw * dh, half, q, r;
    if (n <= 0) { n = 1; dw = dh = 1; }
    half = n / 2;
    q = sum >= 0 ? (sum + half) / n : -(((-sum) + half) / n);
    if ((n & 1) == 0) {
        int exact_recip = (0x10000 % dw) == 0 && (0x10000 % dh) == 0;
        r = sum - q * n;
        if ((r == half || r == -half) && (!exact_recip || (q & 1)))
            q += (sum >= 0) ? -1 : 1;
    }
    return (int8_t)q;
}

/* The divisor pair the part is asked for at (y,x), and the one the RASTER PREDECESSOR
 * carries. A one-window lag on the divisor is the model every failure fits: the taps are
 * summed for this window and divided by the count belonging to the one before it. */
static void pool_prev_window(const rocket_pool_desc *d, int ow, int y, int x,
                             int *dw, int *dh)
{
    int py = y, px = x - 1;
    if (px < 0) { px = ow - 1; py = y - 1; }
    if (py < 0) { py = y; px = x; }
    *dw = pool_inplane(px, d->stride_x, d->kw, d->pad_left, d->iw);
    *dh = pool_inplane(py, d->stride_y, d->kh, d->pad_top, d->ih);
}

static void pool_divisor_readout(const rnet_layer *L, const int8_t *in,
                                 const int8_t *got, const int8_t *ref)
{
    static const char *NAME[4] = { "incl", "lag-x", "lag-y", "prev-window" };
    rocket_pool_desc d;
    long wrong = 0, expl[4] = { 0, 0, 0, 0 };
    int c, y, x, kh, kw, m;

    if (L->kind != KIND_AVGPOOL) return;
    pool_desc_of(L, &d);
    for (c = 0; c < d.c; c++)
        for (y = 0; y < (int)L->oh; y++)
            for (x = 0; x < (int)L->ow; x++) {
                size_t i = ((size_t)c * L->oh + y) * L->ow + x;
                long sum = 0;
                int dw0, dh0;
                if (got[i] == ref[i]) continue;
                wrong++;
                for (kh = 0; kh < d.kh; kh++)
                    for (kw = 0; kw < d.kw; kw++) {
                        int iy = y * d.stride_y + kh - d.pad_top;
                        int ix = x * d.stride_x + kw - d.pad_left;
                        /* The pad value is zero on every arm here: the mode bit moves the
                         * divisor and not the sum, so an outside tap adds nothing whichever
                         * divisor is under test. The graph's averages all run it. */
                        if (iy < 0 || ix < 0 || iy >= d.ih || ix >= d.iw) continue;
                        sum += in[((size_t)c * d.ih + iy) * d.iw + ix];
                    }
                dw0 = pool_inplane(x, d.stride_x, d.kw, d.pad_left, d.iw);
                dh0 = pool_inplane(y, d.stride_y, d.kh, d.pad_top, d.ih);
                for (m = 0; m < 4; m++) {
                    int dw = dw0, dh = dh0;
                    if (m == 0) { dw = d.kw; dh = d.kh; }
                    else if (m == 1)
                        dw = pool_inplane(x - 1, d.stride_x, d.kw, d.pad_left, d.iw);
                    else if (m == 2)
                        dh = pool_inplane(y - 1, d.stride_y, d.kh, d.pad_top, d.ih);
                    else
                        pool_prev_window(&d, (int)L->ow, y, x, &dw, &dh);
                    if (got[i] == pool_round(sum, dw, dh)) expl[m]++;
                }
            }
    printf("      the divisor the part used, over the %ld wrong elements:", wrong);
    for (m = 0; m < 4; m++) printf("  %s %ld", NAME[m], expl[m]);
    printf("\n");

    /* AND THE MAP ITSELF, for the first channel that has one. A count says how well a
     * candidate fits; this says what the part actually divided by at each position, by
     * solving for the divisor pair rather than proposing one. `.` is exact, a digit is
     * the tap count that reproduces the part's value, `?` is none of the kh*kw pairs. */
    for (c = 0; c < d.c; c++) {
        int any = 0;
        for (y = 0; y < (int)L->oh && !any; y++)
            for (x = 0; x < (int)L->ow; x++)
                if (got[((size_t)c * L->oh + y) * L->ow + x] !=
                    ref[((size_t)c * L->oh + y) * L->ow + x]) { any = 1; break; }
        if (!any) continue;
        printf("      channel %d, `.` exact / digit = the tap count the part divided by "
               "/ `?` none of the %d:\n", c, d.kh * d.kw);
        for (y = 0; y < (int)L->oh && y < 24; y++) {
            printf("        ");
            for (x = 0; x < (int)L->ow && x < 40; x++) {
                size_t i = ((size_t)c * L->oh + y) * L->ow + x;
                long sum = 0;
                int dw, dh, hit = 0;
                if (got[i] == ref[i]) { printf(" ."); continue; }
                for (kh = 0; kh < d.kh; kh++)
                    for (kw = 0; kw < d.kw; kw++) {
                        int iy = y * d.stride_y + kh - d.pad_top;
                        int ix = x * d.stride_x + kw - d.pad_left;
                        if (iy < 0 || ix < 0 || iy >= d.ih || ix >= d.iw) continue;
                        sum += in[((size_t)c * d.ih + iy) * d.iw + ix];
                    }
                for (dh = 1; dh <= d.kh && !hit; dh++)
                    for (dw = 1; dw <= d.kw; dw++)
                        if (got[i] == pool_round(sum, dw, dh)) {
                            printf("%2d", dw * dh); hit = 1; break;
                        }
                if (!hit) printf(" ?");
            }
            printf("\n");
        }
        break;
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
        if (layer_run(fd, L, in, in2, NULL, again, &how) != ROCKET_OK) break;
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
        if (layer_run(fd, L, in, NULL, NULL, buf[v], &how[v]) != ROCKET_OK) rc = 1;
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
        {
            const int8_t *ops[MAX_SRC] = { NULL, NULL, NULL, NULL };
            if (L->kind == KIND_CONCAT) gather_ops(i, 1, NULL, ops);
            rc = layer_run(fd, L, in, in2, ops, out, &how);
        }
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
            pool_divisor_readout(L, in, out, ref);
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
            {
                const int8_t *ops[MAX_SRC] = { NULL, NULL, NULL, NULL };
                if (L->kind == KIND_CONCAT && !gather_ops(i, 0, cur, ops)) {
                    printf("%2u concat: an operand the graph did not keep\n", i);
                    failed++; break;
                }
                rc = layer_run(fd, L, chain_input(i, cur), chain_operand_b(L, i, cur),
                               ops, dst, &how);
            }
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
             * compares the whole chain against the row-major one.
             *
             * SCORED ON THE FIRST ITERATION ONLY, because `s` and `drift` are both read
             * only there and a per-element scan of every materialised layer is not free:
             * it is a full pass over the tensor, outside the per-layer `lt` but inside
             * `wall`, so it was 10.1 ms of Inception V1's 28.9 ms — 35% of the number the
             * bench reported as the graph's. The three older graphs are each one kick with
             * one materialised layer and could not show it, which is the shape of the trap:
             * an instrument's overhead is a function of the GRAPH, so a corpus where it is
             * zero says nothing about the one where it is not. [HW sweep, H96 MAX M9] */
            if (it == 0) {
                if (CUBE_OUT && CUBE_OUT[i]) {
                    s.exact = s.total = 0; s.maxdiff = 0;
                } else {
                    score_vs(L, dst, at(L->g_off), &s);
                    if (s.maxdiff > tflite_slack(L) && drift < 0) drift = (int)i;
                }
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
        /* WHERE THE DIVISOR LAG ACTUALLY FIRES. The rate is what the check's cost and
         * every redo are a function of, and a rate averaged over a GEOMETRY cannot say
         * whether every pool of that shape lags or one placement does. Read off the
         * handles, which count it themselves. */
        if (RESIDENT_POOL) {
            unsigned any = 0;
            for (i = 0; i < H->n_layers; i++) {
                unsigned fires = 0, calls = 0;
                if (!RESIDENT_POOL[i]) continue;
                rocket_pool_int8_rk3576_lag_counts(RESIDENT_POOL[i], &fires, &calls);
                if (!calls) continue;
                if (!any++)
                    printf("   the divisor lag, per pooling layer (fires / scored):\n");
                printf("      %2u %-8s %4ux%-3ux%-4u  %5u / %-5u  %5.1f%%   %lu discr, "
                       "src bo %u / %.0f KiB, path %s\n",
                       i, KIND_NAME[LAYERS[i].kind], LAYERS[i].ic, LAYERS[i].ih,
                       LAYERS[i].iw, fires, calls, 100.0 * fires / calls,
                       rocket_pool_int8_rk3576_lag_discr(RESIDENT_POOL[i]) / calls,
                       rocket_pool_int8_rk3576_lag_src_handle(RESIDENT_POOL[i]),
                       rocket_pool_int8_rk3576_lag_src_bytes(RESIDENT_POOL[i]) / 1024.0,
                       paths[i] ? paths[i] : "?");
            }
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
            {
                const int8_t *ops[MAX_SRC] = { NULL, NULL, NULL, NULL };
                if (L->kind == KIND_CONCAT && !gather_ops(i, 0, cur, ops)) {
                    printf("%2u concat: an operand the graph did not keep\n", i);
                    failed++; break;
                }
                rc = layer_run(fd, L, in, chain_operand_b(L, i, cur), ops, dst, &how);
            }
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
        /* A PER-AXIS REQUANT IS REFUSED ON THE PACKED-IMAGE ENCODING — its output-channel
         * tiling and scale sort are the direct path's — so this pass asks for the direct
         * lowering at a narrow channel count whatever the per-tensor path chose. It is
         * exact at any zero point and any spread, so nothing about the accuracy question
         * this pass asks depends on which encoding the other one took. */
        d.direct_datapath = L->kind == KIND_CONV && L->ic <= 4 && !widen_on();
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
                      const int8_t *in, const int8_t *in2, const int8_t *const *ops,
                      int8_t *out)
{
    if (L->kind == KIND_CONCAT) {
        if (!ops || !ops[0]) return ROCKET_E_SHAPE;
        concat_run(L, ops, out);
        return ROCKET_OK;
    }
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
        {
            const int8_t *ops[MAX_SRC] = { NULL, NULL, NULL, NULL };
            if (L->kind == KIND_CONCAT && !gather_ops(i, 0, cur, ops)) {
                printf("%2u concat: an operand the host chain did not keep\n", i);
                (*failed)++;
                return NULL;
            }
            rc = host_layer(L, (P && weighted) ? &P[i] : NULL, in,
                            chain_operand_b(L, i, cur), ops, dst);
        }
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
        const int8_t *ops[MAX_SRC] = { NULL, NULL, NULL, NULL };
        if (L->kind == KIND_CONCAT && !gather_ops(i, 0, cur, ops)) return 1;
        if (layer_run(fd, L, chain_input(i, cur), chain_operand_b(L, i, cur), ops,
                      dst, &how) != ROCKET_OK) {
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

    plan_build(fd);
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
    rocket_graph_plan_kicks(PLAN);
    if (!KICK_RUNS) {
        if (rocket_graph_kick_on()) {
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
        /* The blob's own name carries the input resolution, so a network at anything
         * other than 224 needs the table rather than a suffix. */
        static const struct { const char *net, *stem; } BLOBS[] = {
            { "v1",  "mobilenet_v1_224"   }, { "v2",  "mobilenet_v2_224"   },
            { "r18", "resnet18_224"       }, { "iv1", "inception_v1_224"   },
            { "iv3", "inception_v3_299"   },
        };
        const char *root = getenv("ROCKET_SRC_DIR");
        const char *stem = NULL;
        size_t k;
        for (k = 0; k < sizeof BLOBS / sizeof *BLOBS; k++)
            if (!strcmp(net, BLOBS[k].net)) { stem = BLOBS[k].stem; break; }
        if (!stem) {
            printf("unknown network %s (v1 v2 r18 iv1 iv3)\n", net);
            return 2;
        }
        snprintf(def, sizeof def, "%s/tests/data/rk3576-net/%s_quant.rnet",
                 root ? root : ".", stem);
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
            else { failed += graph_once(fd, NULL, &lg, wa, wb); plan_build(fd);
                   rocket_graph_plan_kicks(PLAN); }
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
