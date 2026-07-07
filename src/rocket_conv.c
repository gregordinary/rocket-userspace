// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_conv.c — general fp16 CONV_2D on the rocket NPU. See rocket_conv.h.
 *
 * The NPU CNA is a native convolution engine, so a KxK / stride / pad / dilation
 * conv runs directly (no im2col): the host scatters the input feature into the
 * NC1HWC2 cube (feature_data, C2=8) and the weights into the conv weight cube
 * (weight_conv_fp16, the Mesa-confirmed oc1/ic1/kh/kw/oc2/ic2 reorder), and the
 * generator (gen_conv2d_fp16) programs the sliding-window MAC. The matmul path is
 * the degenerate 1x1 case of exactly this.
 *
 * SCOPE: direct (non-depthwise) conv now tiles across multiple NPU jobs over OC and
 * the output spatial dims (OH/OW), and pads IC<32 first layers, so a feature map + OC
 * kernel set larger than one CBUF pass IS handled by host-side tiling. Each tile still
 * reduces its full KH*KW*IC contraction in one native pass (no host K-accumulation).
 * Depthwise has CHANNEL tiling but NOT spatial tiling: a DW layer whose single
 * channel's feature map exceeds the CBUF budget still falls back to the caller's CPU
 * path. A native int8/uint8 conv path also exists (gen_conv2d_*int8); see rocket_conv.h.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"   /* CBUF bank count from the active hardware profile */
#include "npu_matmul.h"
#include "rocket_conv.h"
#include "rocket_activation.h"   /* lut_epilogue_t builder + ref (conv->act fusion) */
#include "rocket_affinity.h"
#include "rocket_chain.h"        /* rkt_chain_pack — gapped multi-task batched submit */
#include "rocket_scratch.h"      /* per-ctx bump arena for the tilers' host scratch */
#include "rocket_log.h"     // centralized log channel

/* ############################################################################
 * PART 1 — Layout budgets, resident-BO context + multicore worker pool
 * ##########################################################################*/

/* CBUF bank SIZE: a compile-time constant pointed at the single npu_hw.h source (the
 * feature/weight budget macros below want a constant). The bank COUNT is read from the
 * active hardware profile per tiler (rocket_hw_current()->cbuf_banks) so the chip value
 * lives in ONE place rather than as a bare 12 duplicating NPU_CBUF_BANKS (the "edit one,
 * miss the others" mis-tile trap). */
#define CBUF_BANK NPU_CBUF_BANK_SIZE
/* Feature-tile budget: rows*IC*IW*2 must leave room for the weight tile in the
 * 12x32KB CBUF. Keep the feature within ~8 banks so the weight always has >=4. The
 * tilers SHRINK this further when a single OC-group's weight cube needs >4 banks (a
 * KxK conv at large IC) so feature_banks + weight_banks still fit the 12-bank CBUF. */
#define CONV_FEAT_BUDGET (8 * CBUF_BANK)
/* int8 feature budget = 8 banks - 1 SLACK bank. The int8 C2=16 feature-cube DMA
 * over-reads one CBUF bank past its ceil allocation at near-bank-full geometries
 * (gen_conv2d_int8_fill sets data_bank = fd_banks+1, HW-proven for the conv). The
 * tiler must keep feature_banks + 1 + weight_banks <= 12,
 * so the int8 feature ceiling drops to 7 banks (fp16 is immune -> CONV_FEAT_BUDGET). */
#define CONV_FEAT_BUDGET_I8 (7 * CBUF_BANK)

/* Batched-submit of a DIRECT int8/uint8 conv's independent tiles: lay each tile's
 * complete regcmd in its own slot of one regcmd BO and submit the slice as ONE
 * multi-task job (lever-1, gapped) instead of one ioctl per tile. The job runs the
 * tasks as separate HW kicks (the int32 CACC clears per kick, so int8 stays exact —
 * unlike fp16 chaining), but pays ONE submit syscall + ONE fence wait + ONE IOMMU
 * attach for the whole slice. Bit-identical to the per-tile path; opt-in while it
 * proves out. CONV_RC_STRIDE is the per-task regcmd slot (>= gen_conv2d_int8's 256-word
 * cap). */
#define CONV_RC_STRIDE 256
static int conv_batch_enabled(void)
{
    static _Atomic int c = -1;
    if (c < 0) { const char *e = getenv("ROCKET_CONV_BATCH"); c = (e && atoi(e) > 0) ? 1 : 0; }
    return c;
}
/* round `n` bytes up to a whole CBUF bank (tile regions are bank-aligned so each tile's
 * base matches a fresh single-job BO base and its feature-DMA +1-bank over-read lands in
 * the next zeroed bank — bit-identical to the standalone job). */
static inline size_t bank_round(size_t n) { return (n + CBUF_BANK - 1) & ~((size_t)CBUF_BANK - 1); }

/* Depthwise channel group G (the weight cube's innermost channel atom). Mesa's
 * int8 value is 64 (feature-atom 16 × 4); fp16 halves the feature atom to 8, so
 * the same 4× ratio gives **32** — HW-CONFIRMED 2026-06-20 (G=64 fails, G=32 is
 * bit-exact on every DW shape). ROCKET_CONV_DW_GROUP overrides. Both the
 * generator and the host scatter read this so they agree. */
static int conv_dw_group(void)
{
    const char *e = getenv("ROCKET_CONV_DW_GROUP");
    int g = e ? atoi(e) : 32;
    return g > 0 ? g : 32;
}

/* Depthwise feature budget (bytes). The single-job DW feature uses the SAME CNA input
 * DMA (surf_stride / height-blocked) as the direct path, so a DW job is only inside a
 * HW-validated envelope when its feature stays within the direct path's CONV_FEAT_BUDGET
 * (8 banks) — NOT the optimistic 12-bank "whatever the CBUF holds". Bounding the feature
 * to the direct path's 8 banks (and spatially tiling past that) keeps every DW job inside
 * the feature-DMA envelope the direct path proved on HW; an unbounded 12-bank DW feature
 * overflows and fails on HW (observed on a C=192 32x32 MobileNetV2 block, whereas the
 * validated DW shapes were all <1 bank). ROCKET_CONV_DW_FEAT_BANKS overrides the bank
 * count for an on-HW bisection of the true ceiling. */
static size_t dw_feat_budget(void)
{
    const char *e = getenv("ROCKET_CONV_DW_FEAT_BANKS");
    int banks = e ? atoi(e) : (CONV_FEAT_BUDGET / CBUF_BANK);   /* default 8 */
    if (banks < 1)  banks = 1;
    if (banks > 11) banks = 11;                                 /* leave >=1 bank for weight */
    return (size_t)banks * CBUF_BANK;
}

/* Does a depthwise channel chunk of Cc channels fit ONE CBUF pass at the FULL IHxIW
 * feature? (the whole KH*KW*Cc weight cube in one bank + the Cc-channel feature within
 * the validated feature budget). When this is false because the feature is too big, the
 * tiler reduces Cc (channel tiling) or, once even one weight group G overflows on
 * feature, spatially tiles the chunk (dw_spatial). */
static int dw_chunk_fits(int Cc, long IH, long IW, int KH, int KW)
{
    if (Cc <= 0) return 0;
    size_t wpk = (size_t)KW * KH * Cc * sizeof(_Float16);
    if (wpk > CBUF_BANK) return 0;                       /* weight chunk must fit one bank */
    size_t fd_bytes = (size_t)Cc * IH * IW * sizeof(_Float16);
    return fd_bytes <= dw_feat_budget();                 /* feature within validated budget */
}

/* The CNA feature DMA is height-blocked in 4-row blocks (surf_stride = IW*(IH-4)/4);
 * a sub-input with fewer than 4 rows (datain_height < 4) produces WRONG hardware
 * output regardless of surf_stride — HW-confirmed at every IC (IH=2,3 fail; IH>=4
 * pass; tests/conv_bisect.c). datain_height is the materialized sub-input height
 * (rh-1)*sy + (KH-1)*dy + 1, so this is the hard floor on each row band's OUTPUT rows. */
#define CONV_MIN_DATAIN_H 4
static inline long datain_h(int out_rows, int sy, int KH, int dy)
{
    return (long)(out_rows - 1) * sy + (long)(KH - 1) * dy + 1;
}

/* Partition `total` output positions into the FEWEST equal-ish bands no larger than
 * `maxb` (>=1), so every band is `total/n` or `total/n + 1` — never a tiny remainder
 * stub (the bug: a budget-derived `maxb` leaves an OH%maxb remainder that can fall
 * below CONV_MIN_DATAIN_H and corrupt that tile). Writes the band sizes into sizes[]
 * (caller pre-sizes to >= ceil(total/maxb)) and returns the band count n. The minimum
 * band is sizes[n-1] == total/n, maximal over all valid <=maxb partitions, so if even
 * this balanced split underflows the row floor the shape is genuinely untileable. */
static int balance_bands(int total, int maxb, int *sizes)
{
    if (maxb < 1) maxb = 1;
    int n = (total + maxb - 1) / maxb;
    if (n < 1) n = 1;
    int base = total / n, extra = total % n;
    for (int i = 0; i < n; i++) sizes[i] = base + (i < extra ? 1 : 0);
    return n;                                             /* sizes[] descending: extra first */
}

/* ---- resident-BO conv context (see rocket_conv.h) ------------------------------
 * A pool of the five BOs conv2d_one_job needs (IOVA guard / input / weight / regcmd /
 * output), cached on a borrowed fd and grown to the largest tile seen. conv2d_one_job
 * takes an optional ctx: NULL => the legacy alloc-per-call-and-free path (byte-for-byte
 * unchanged); non-NULL => borrow the pool (no per-call alloc/free). The fd is borrowed,
 * never opened/closed here. */
struct rocket_conv_ctx {
    int       fd;                              /* borrowed; not owned */
    rocket_bo guard, in, wt, rc, out;          /* ptr==NULL until first use; grown on demand */
    rocket_bo bias;                            /* int8 depthwise-out per-OC int32 bias cube */
    rocket_scratch_pool scratch;               /* grow-only host scratch for the spatial tilers */
};

rocket_conv_ctx *rocket_conv_ctx_create(int fd)
{
    rocket_conv_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->fd = fd;
    return c;                                   /* fd<0 is fine: BOs are never allocated */
}

void rocket_conv_ctx_free(rocket_conv_ctx *c)
{
    if (!c) return;
    if (c->fd >= 0) {                           /* rocket_bo_free is a no-op on a {0} BO */
        rocket_bo_free(c->fd, &c->bias);
        rocket_bo_free(c->fd, &c->out);
        rocket_bo_free(c->fd, &c->rc);
        rocket_bo_free(c->fd, &c->wt);
        rocket_bo_free(c->fd, &c->in);
        rocket_bo_free(c->fd, &c->guard);
    }
    rocket_scratch_pool_free(&c->scratch);      /* fd-independent host scratch */
    free(c);
}

/* --- multicore worker pool for the native int8/uint8 DIRECT conv ----------------
 * The rocket driver pins one fd to one NPU core while it has queued work, so the
 * single-fd conv2d_int8_run serializes its independent OC/OH/OW tiles onto one of
 * the 3 cores. The pool holds N persistent worker fds, each with its own resident
 * rocket_conv_ctx (BO pool), so the tiles fan out across all 3 cores while keeping
 * the resident-BO (no per-call alloc/free) win. Created once per delegate partition
 * and reused across ops/inferences. The pool OWNS its fds (opens + closes them). */
#define ROCKET_CONV_POOL_MAX 8
struct rocket_conv_pool {
    int n;
    int fd[ROCKET_CONV_POOL_MAX];
    rocket_conv_ctx *ctx[ROCKET_CONV_POOL_MAX];
};

rocket_conv_pool *rocket_conv_pool_create(int nthreads)
{
    if (nthreads < 1) nthreads = 1;
    if (nthreads > ROCKET_CONV_POOL_MAX) nthreads = ROCKET_CONV_POOL_MAX;
    rocket_conv_pool *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    for (int i = 0; i < nthreads; i++) {
        int fd = rocket_open();
        if (fd < 0) break;                       /* degrade to however many opened */
        rocket_conv_ctx *c = rocket_conv_ctx_create(fd);
        if (!c) { rocket_close(fd); break; }
        p->fd[p->n]  = fd;
        p->ctx[p->n] = c;
        p->n++;
    }
    if (p->n == 0) { free(p); return NULL; }
    return p;
}

void rocket_conv_pool_free(rocket_conv_pool *p)
{
    if (!p) return;
    for (int i = 0; i < p->n; i++) {
        rocket_conv_ctx_free(p->ctx[i]);         /* frees BOs on fd[i] (uses it) ... */
        rocket_close(p->fd[i]);                  /* ... then the pool closes the fd */
    }
    free(p);
}

/* Ensure *bo holds an allocation of >= need bytes: reuse if already big enough, else
 * (re)allocate (freeing the old, smaller BO first). Returns 0 / <0. */
static int conv_bo_ensure(int fd, rocket_bo *bo, size_t need)
{
    if (bo->ptr && bo->size >= need) return 0;  /* reuse */
    if (bo->ptr) rocket_bo_free(fd, bo);        /* grow: release the old one first */
    return rocket_bo_alloc(fd, need, bo);
}

/* ############################################################################
 * PART 2 — CPU reference oracles + descriptor validation
 * ##########################################################################*/

/* CPU fp32-accumulate reference — the golden oracle (also a host fallback). */
void rocket_conv2d_ref_fp16(const rocket_conv2d_desc *d,
                            const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    int IC = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;

    for (int oc = 0; oc < OC; oc++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                float s = 0.f;
                /* depthwise: each output channel reduces only its own input
                 * channel (W is [OC][1][KH][KW]); direct: full IC reduction. */
                int ic_lo = d->depthwise ? oc : 0;
                int ic_hi = d->depthwise ? oc + 1 : IC;
                for (int ic = ic_lo; ic < ic_hi; ic++) {
                    int wic = d->depthwise ? 0 : ic;       /* weight ic index   */
                    int wic_span = d->depthwise ? 1 : IC;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * d->stride_y + kh * d->dil_y - d->pad_top;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * d->stride_x + kw * d->dil_x - d->pad_left;
                            if (iw < 0 || iw >= IW) continue;
                            float a = (float)in[((size_t)ic * IH + ih) * IW + iw];
                            float w = (float)W[(((size_t)oc * wic_span + wic) * KH + kh) * KW + kw];
                            s += a * w;
                        }
                    }
                }
                out[((size_t)oc * OH + oh) * OW + ow] = (_Float16)s;
            }
        }
    }
}

/* CPU int64-accumulate -> int32 reference for the native int8 conv (golden oracle +
 * the fd<0 host fallback). int8 x int8 reduced over KH*KW*IC: a 7x7x512 conv sums to
 * ~406M, past int32, so accumulate in int64 and store int32 (saturating range warned). */
void rocket_conv2d_ref_int8(const rocket_conv2d_desc *d,
                            const int8_t *in, const int8_t *W, int32_t *out)
{
    int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    int IC = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;

    for (int oc = 0; oc < OC; oc++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int64_t s = 0;
                int ic_lo = d->depthwise ? oc : 0;
                int ic_hi = d->depthwise ? oc + 1 : IC;
                for (int ic = ic_lo; ic < ic_hi; ic++) {
                    int wic = d->depthwise ? 0 : ic;
                    int wic_span = d->depthwise ? 1 : IC;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * d->stride_y + kh * d->dil_y - d->pad_top;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * d->stride_x + kw * d->dil_x - d->pad_left;
                            if (iw < 0 || iw >= IW) continue;
                            int32_t a = in[((size_t)ic * IH + ih) * IW + iw];
                            int32_t w = W[(((size_t)oc * wic_span + wic) * KH + kh) * KW + kw];
                            s += (int64_t)a * w;
                        }
                    }
                }
                out[((size_t)oc * OH + oh) * OW + ow] = (int32_t)s;
            }
        }
    }
}

/* Validate a descriptor against the supported set. Returns 0 or <0. */
int rocket_conv2d_plan(const rocket_conv2d_desc *d)
{
    if (!d) return -1;
    if (d->ic <= 0 || d->ih <= 0 || d->iw <= 0 || d->oc <= 0 || d->kh <= 0 || d->kw <= 0)
        return -1;
    if (d->stride_x <= 0 || d->stride_y <= 0 || d->dil_x <= 0 || d->dil_y <= 0)
        return -1;
    /* direct: OC need NOT be a multiple of 16 — OC%16!=0 (e.g. an SSD box/class head,
     * OC=24) is zero-padded up to 16 by the driver (extra OC kernels contribute 0 and
     * the extra output channels are sliced off), mirroring the IC<32 first-layer pad.
     * depthwise: OC==IC==C and IC%G==0 (G=32), so OC%16==0 there automatically. */
    if (d->depthwise && d->oc != d->ic) return -2; /* depthwise: one ic per oc   */
    if (d->depthwise && (d->ic % conv_dw_group())) return -2;  /* ic % G          */
    /* direct: IC < 32 (e.g. the RGB first layer) is zero-padded to 32 by the
     * driver, so any IC is accepted. */
    int IC = d->depthwise ? d->ic : ((d->ic + 31) / 32) * 32;
    int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    if (OH <= 0 || OW <= 0) return -3;
    if (d->depthwise) {
        /* depthwise tiles over CHANNELS (each channel is independent) and, when even one
         * weight group G of channels still overflows the feature budget (very large IH*IW
         * for one channel, e.g. a high-res early layer), over SPACE as well (dw_spatial,
         * mirroring the direct path's band tiler). So the only hard requirements are that
         * one G-channel weight cube fits a bank and the MINIMAL spatial tile (G channels,
         * one output row x one output col) fits the feature budget — everything larger is
         * reachable by channel + spatial tiling. */
        const int G = conv_dw_group();
        size_t wpk = (size_t)d->kw * d->kh * G * sizeof(_Float16);
        if (wpk > CBUF_BANK) return -4;                       /* a G-chunk weight must fit a bank */
        long ih_min = (long)(d->kh - 1) * d->dil_y + 1;
        long iw_min = (long)(d->kw - 1) * d->dil_x + 1;
        if ((size_t)G * ih_min * iw_min * sizeof(_Float16) > dw_feat_budget()) return -4;
    } else {
        size_t wpk = (size_t)d->kw * d->kh * IC * sizeof(_Float16);    /* one OC kernel */
        if (wpk > CBUF_BANK) return -4;                                /* a kernel must fit a bank */
        /* direct tiles over OC + OH-rows + OW-cols; only the MINIMAL tile (one
         * output row x one output col) must fit the feature budget. */
        long ih_min = (long)(d->kh - 1) * d->dil_y + 1;
        long iw_min = (long)(d->kw - 1) * d->dil_x + 1;
        if ((size_t)IC * ih_min * iw_min * sizeof(_Float16) > CONV_FEAT_BUDGET) return -4;
    }
    return 0;
}

/* Run ONE conv as a single NPU job (must fit one CBUF pass). Explicit shapes +
 * row-major host buffers: in [IC][IH][IW], W direct [OC][IC][KH][KW] / depthwise
 * [OC][1][KH][KW], out [OC][OH][OW]. OH/OW are caller-computed. Returns 0, or <0
 * (incl. the generator's -1/-2 when the tile overflows the CBUF — the tiling
 * wrapper shrinks and retries on that). This is the HW-validated direct path; the
 * tiling wrapper composes it. */
/* ############################################################################
 * PART 3 — fp16 direct + depthwise conv (single job, spatial/channel tiling)
 * ##########################################################################*/
static int conv2d_one_job(int fd, rocket_conv_ctx *ctx,
                          int IC, int IH, int IW, int OC, int OH, int OW,
                          int KH, int KW, int sy, int sx, int pt, int pl,
                          int dy, int dx, int DW, int G, const lut_epilogue_t *act,
                          const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    const int Cpad = DW ? ((IC + G - 1) / G) * G : IC;
    /* conv->activation fusion is the DIRECT-conv path only (the smooth FFN/Whisper
     * gates); depthwise stays plain (its hardswish hits the NVDLA flat-tail quirk). */
    const lut_epilogue_t *jact = DW ? NULL : act;
    int ret = -1;   /* the early-error gotos (BO alloc / 32-bit IOVA) return via this */

    /* No device (fd<0): compute this sub-problem with the CPU oracle. This makes
     * rocket_conv2d_fp16 a pure-CPU tiled conv off-device, so the tiling
     * decomposition (band extraction, materialized pad, placement) is verifiable on
     * x86 against the whole-image oracle — the per-tile NPU path is the separately
     * HW-validated single job. (void OH/OW: the oracle recomputes them, == passed.)
     * Reached before any BO is touched, so a ctx wrapping fd<0 is inert. */
    if (fd < 0) {
        rocket_conv2d_desc sd = { .ic=IC,.ih=IH,.iw=IW,.oc=OC,.kh=KH,.kw=KW,
            .stride_y=sy,.stride_x=sx,.pad_top=pt,.pad_left=pl,.dil_y=dy,.dil_x=dx,
            .depthwise=DW };
        const int OHr = rocket_conv2d_oh(&sd), OWr = rocket_conv2d_ow(&sd);
        (void)OH; (void)OW;
        rocket_conv2d_ref_fp16(&sd, in, W, out);
        /* fused activation: apply f(x) on this tile's FINAL outputs (each output element
         * is fully reduced within its tile, so per-tile == once-per-element). Keeps the
         * off-device tiled path == CPU conv-then-f, so the tiling is x86-verifiable. */
        if (jact) rocket_activation_ref_fp16(jact->kind, out, out, OC * OHr * OWr);
        return 0;
    }

    /* Pad the job's input height up to the 4-row CNA DMA floor with zero rows below the
     * real data (see the int8 sibling): the cube gets IHj rows, datain_height is IHj, but
     * OH stays the real output count, so the padded rows never enter a computed output's
     * receptive field — bit-exact, and a short band / small-spatial conv just works. */
    const int IHj = IH < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : IH;
    size_t in_elems  = (size_t)IC * IHj * IW;
    size_t wt_elems  = DW ? (size_t)Cpad * KH * KW : (size_t)OC * IC * KH * KW;
    size_t out_elems = (size_t)OC * OH * OW;

    /* BOs: borrow the ctx's resident pool when given (no per-call alloc/free, grown to
     * the largest tile), else local transients freed at the end. Either way every job
     * memsets + refills all five, so the resident reuse is bit-identical to a fresh
     * alloc. The locals stay {0} when ctx!=NULL (rocket_bo_free on {0} is a no-op). */
    rocket_bo lguard = {0}, lin = {0}, lwt = {0}, lrc = {0}, lout = {0};
    rocket_bo *guard  = ctx ? &ctx->guard : &lguard;
    rocket_bo *in_bo  = ctx ? &ctx->in    : &lin;
    rocket_bo *wt_bo  = ctx ? &ctx->wt    : &lwt;
    rocket_bo *rc_bo  = ctx ? &ctx->rc    : &lrc;
    rocket_bo *out_bo = ctx ? &ctx->out   : &lout;
    /* The conv->activation fusion appends the 2*513-entry LE/LO table upload to the
     * regcmd (~1030 extra ops), so size the regcmd buffer/BO for it when fused. */
    const int NREGS = jact ? 2048 : 256;
    uint64_t regs[2048] = {0};
    int rc = -1;

    if (conv_bo_ensure(fd, guard, 4096) ||                                       /* off IOVA 0 */
        conv_bo_ensure(fd, in_bo,  in_elems  * sizeof(_Float16) + CBUF_BANK) ||
        conv_bo_ensure(fd, wt_bo,  wt_elems  * sizeof(_Float16) + CBUF_BANK) ||
        conv_bo_ensure(fd, rc_bo,  (size_t)NREGS * sizeof(uint64_t))        ||
        conv_bo_ensure(fd, out_bo, out_elems * sizeof(_Float16) + CBUF_BANK)) {
        ROCKET_LOGE("rocket_conv2d_fp16: BO alloc failed\n");
        goto out;
    }
    if (((in_bo->dma_address + in_bo->size) | (wt_bo->dma_address + wt_bo->size) |
         (out_bo->dma_address + out_bo->size) | (rc_bo->dma_address + rc_bo->size)) >> 32) {
        ROCKET_LOGE("rocket_conv2d_fp16: a BO dma_address exceeds 32 bits\n");
        goto out;
    }

    /* scatter input feature -> NC1HWC2 cube (C2=8) */
    rocket_bo_prep(fd, in_bo, 1, 0);
    memset(in_bo->ptr, 0, in_bo->size);
    {
        _Float16 *dst = in_bo->ptr;     /* cube laid out for IHj rows; rows >= IH stay 0 */
        for (int ic = 0; ic < IC; ic++)
            for (int ih = 0; ih < IH; ih++)
                for (int iw = 0; iw < IW; iw++)
                    dst[feature_data(IC, IHj, IW, 8, ic + 1, ih + 1, iw + 1)] =
                        in[((size_t)ic * IH + ih) * IW + iw];
    }
    rocket_bo_fini(fd, in_bo);

    /* scatter weights -> the conv weight cube. direct: oc1/ic1/kh/kw/oc2/ic2 from
     * W[OC][IC][KH][KW]. depthwise: ic1/kh/kw/ic2 (group G) from W[OC][1][KH][KW]. */
    rocket_bo_prep(fd, wt_bo, 1, 0);
    memset(wt_bo->ptr, 0, wt_bo->size);
    {
        _Float16 *dst = wt_bo->ptr;
        if (DW) {
            for (int c = 0; c < IC; c++)               /* OC==IC, one filter per channel */
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        dst[weight_conv_dw_fp16(IC, KH, KW, G, c + 1, kh + 1, kw + 1)] =
                            W[((size_t)c * KH + kh) * KW + kw];   /* W[c][0][kh][kw] */
        } else {
            for (int oc = 0; oc < OC; oc++)
                for (int ic = 0; ic < IC; ic++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                            dst[weight_conv_fp16(OC, IC, KH, KW, oc + 1, ic + 1, kh + 1, kw + 1)] =
                                W[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
        }
    }
    rocket_bo_fini(fd, wt_bo);

    /* generate the conv regcmd */
    conv_params_t p = {
        .ic = IC, .ih = IHj, .iw = IW, .oc = OC, .oh = OH, .ow = OW,
        .kh = KH, .kw = KW,
        .stride_y = sy, .stride_x = sx,
        .dil_y = dy, .dil_x = dx,
        .pad_top = pt, .pad_left = pl,
        .input_dma = (uint32_t)in_bo->dma_address,
        .weights_dma = (uint32_t)wt_bo->dma_address,
        .output_dma = (uint32_t)out_bo->dma_address,
        .tasks = regs, .fp32tofp16 = 1, .dw_group = (uint8_t)(DW ? G : 0),
        .act = jact,   /* fp16-out conv + LUT epilogue (direct only; NULL == plain conv) */
    };
    if ((ret = DW ? gen_conv2d_dw_fp16(&p) : gen_conv2d_fp16(&p)) != 0) {
        ROCKET_LOGE("rocket_conv2d_fp16: gen failed (%d)\n", ret);
        goto out;
    }
    if (p.task_count > sizeof(regs)/sizeof(regs[0])) {   /* unconditional: -DNDEBUG strips asserts */
        ROCKET_LOGE("rocket_conv2d_fp16: regcmd overflow (task_count %u > %zu words)\n",
                p.task_count, sizeof(regs)/sizeof(regs[0]));
        ret = -1; goto out;
    }
    rocket_bo_prep(fd, rc_bo, 1, 0);
    memcpy(rc_bo->ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, rc_bo);

    /* zero + hand the output BO to the device */
    rocket_bo_prep(fd, out_bo, 1, 0);
    memset(out_bo->ptr, 0, out_bo->size);
    rocket_bo_fini(fd, out_bo);

    {
        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo->dma_address,
                                  .regcmd_count = p.task_count };
        uint32_t in_h[]  = { in_bo->handle, wt_bo->handle, rc_bo->handle };
        uint32_t out_h[] = { out_bo->handle };
        ret = rocket_submit_tasks(fd, &task, 1, in_h, 3, out_h, 1);
        if (ret) { ROCKET_LOGE("rocket_conv2d_fp16: submit failed (%d)\n", ret); goto out; }
    }

    /* read back: de-scatter the output cube (OC/8, OH, OW, 8) */
    ret = rocket_bo_prep(fd, out_bo, 0, 2000000000ULL);   /* 2s wait */
    if (ret) { ROCKET_LOGE("rocket_conv2d_fp16: wait timeout (%d)\n", ret); goto out; }
    {
        _Float16 *src = out_bo->ptr;
        for (int oc = 0; oc < OC; oc++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                    out[((size_t)oc * OH + oh) * OW + ow] =
                        src[feature_data(OC, OH, OW, 8, oc + 1, oh + 1, ow + 1)];
    }
    rocket_bo_fini(fd, out_bo);
    rc = 0;

out:
    /* Resident BOs (ctx!=NULL) persist on the ctx for the next call; only the local
     * transients (ctx==NULL) are freed here. */
    if (!ctx) {
        rocket_bo_free(fd, out_bo);
        rocket_bo_free(fd, rc_bo);
        rocket_bo_free(fd, wt_bo);
        rocket_bo_free(fd, in_bo);
        rocket_bo_free(fd, guard);
    }
    return rc ? (ret ? ret : -1) : 0;
}

/* Spatially tile a single depthwise channel chunk of Cn channels (Cn % G == 0) whose
 * feature is too big for one CBUF pass. Mirrors the direct path's OH-row/OW-col band
 * tiler EXACTLY — materialized halo + explicit zero pad so each band runs with
 * pad_top=pad_left=0, the min_ihs floor the direct path's HW bug established, columns
 * narrowed only when a full-width band still overflows — but for the depthwise job
 * (each channel reduces only itself, so there is no OC loop and the weight cube [Cn]
 * [KH][KW] is shared by every band). Each emitted tile is therefore a depthwise job
 * that is simultaneously inside the direct path's HW-validated feature-DMA envelope
 * (<= budget feature, >= min_ihs rows) and the depthwise path's validated single-job
 * envelope (small C, native DW_EN) — the intersection both gates already proved.
 * fd<0 -> conv2d_one_job computes each tile on the CPU oracle, so the band/halo
 * decomposition is x86-verifiable against the whole-image DW oracle. Returns 0 / <0. */
static int dw_spatial(int fd, rocket_conv_ctx *ctx, int Cn, int IH, int IW, int OH, int OW,
                      int KH, int KW, int sy, int sx, int pt, int pl, int dy, int dx,
                      int G, const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    const size_t budget = dw_feat_budget();
    #define DW_FEAT(rh, cw) ((size_t)Cn * \
        ((size_t)((rh) - 1) * sy + (size_t)(KH - 1) * dy + 1) * \
        ((size_t)((cw) - 1) * sx + (size_t)(KW - 1) * dx + 1) * sizeof(_Float16))

    int min_ihs = 6;
    { const char *e = getenv("ROCKET_CONV_MIN_IH"); if (e && atoi(e) > 0) min_ihs = atoi(e); }
    if (min_ihs < CONV_MIN_DATAIN_H) min_ihs = CONV_MIN_DATAIN_H;   /* never below the HW row floor */
    int rht_lo = 1;
    while (rht_lo < OH && datain_h(rht_lo, sy, KH, dy) < min_ihs) rht_lo++;
    int rht = OH, cwt = OW;
    while (rht > rht_lo && DW_FEAT(rht, cwt) > budget) rht--;
    if (DW_FEAT(rht, cwt) > budget)                    /* full-width band still over -> narrow cols */
        while (cwt > 1 && DW_FEAT(rht, cwt) > budget) cwt--;
    if (DW_FEAT(rht, cwt) > budget) return -4;          /* minimal tile still over */

    const int IHs = (rht - 1) * sy + (KH - 1) * dy + 1;
    const int IWs = (cwt - 1) * sx + (KW - 1) * dx + 1;
    /* BALANCED row/col bands: no tiny remainder stub below CONV_MIN_DATAIN_H. */
    int nRmax = (OH + rht - 1) / rht, nCmax = (OW + cwt - 1) / cwt;
    const size_t szs[4] = { (size_t)nRmax * sizeof(int), (size_t)nCmax * sizeof(int),
                            (size_t)Cn * IHs * IWs * sizeof(_Float16),
                            (size_t)Cn * rht * cwt * sizeof(_Float16) };
    rocket_arena a = {0};
    if (rocket_arena_open(&a, ctx ? &ctx->scratch : NULL, rocket_arena_reserve(szs, 4)) < 0)
        return -1;
    int      *rband   = rocket_arena_push(&a, szs[0]);
    int      *cband   = rocket_arena_push(&a, szs[1]);
    _Float16 *sub_in  = rocket_arena_push(&a, szs[2]);
    _Float16 *sub_out = rocket_arena_push(&a, szs[3]);
    if (!rband || !cband || !sub_in || !sub_out) { rocket_arena_close(&a); return -1; }
    int nR = balance_bands(OH, rht, rband);
    int nC = balance_bands(OW, cwt, cband);
    /* short bands are padded to the 4-row floor inside the job (conv2d_one_job IHj). */

    int ret = 0;
    int r0 = 0;
    for (int ri = 0; ri < nR && !ret; ri++) {
        int rh = rband[ri];
        int ih_sub = (rh - 1) * sy + (KH - 1) * dy + 1;
        int gh0 = r0 * sy - pt;
        int c0 = 0;
        for (int ci = 0; ci < nC && !ret; ci++) {
            int cw = cband[ci];
            int iw_sub = (cw - 1) * sx + (KW - 1) * dx + 1;
            int gw0 = c0 * sx - pl;

            memset(sub_in, 0, (size_t)Cn * ih_sub * iw_sub * sizeof(_Float16));
            for (int c = 0; c < Cn; c++)
                for (int j = 0; j < ih_sub; j++) {
                    int gih = gh0 + j;
                    if (gih < 0 || gih >= IH) continue;
                    int cs = gw0 < 0 ? 0 : gw0;
                    int ce = gw0 + iw_sub; if (ce > IW) ce = IW;
                    if (cs >= ce) continue;
                    memcpy(&sub_in[((size_t)c * ih_sub + j) * iw_sub + (cs - gw0)],
                           &in[((size_t)c * IH + gih) * IW + cs],
                           (size_t)(ce - cs) * sizeof(_Float16));
                }

            ret = conv2d_one_job(fd, ctx, Cn, ih_sub, iw_sub, Cn, rh, cw, KH, KW,
                                 sy, sx, 0, 0, dy, dx, 1, G, NULL, sub_in, W, sub_out);
            if (ret) break;

            for (int c = 0; c < Cn; c++)
                for (int r = 0; r < rh; r++)
                    memcpy(&out[((size_t)c * OH + (r0 + r)) * OW + c0],
                           &sub_out[((size_t)c * rh + r) * cw],
                           (size_t)cw * sizeof(_Float16));
            c0 += cw;
        }
        r0 += rh;
    }
    #undef DW_FEAT
    rocket_arena_close(&a);
    return ret;
}

/* Run the conv on the NPU, tiling over output channels (OC) and output rows (OH)
 * when the whole problem won't fit one CBUF pass. Each tile is an independent
 * single-job direct conv (the HW-validated path); OH-band tiles MATERIALIZE their
 * top/bottom edge padding into the sub-input (the CNA only has symmetric pad_top,
 * so a per-band pad_top would also pad the band's interior bottom — instead we feed
 * real halo rows + explicit zero rows and run the sub-conv with pad_top=0). Width
 * (OW) is not tiled; pad_left is applied by the HW as usual. in / W / out are
 * row-major fp16. Returns 0 / <0.
 *
 * Native depthwise (DW_EN) is HW-VALIDATED (2026-06-20, bit-exact on every DW test
 * shape) with group G=32 and the size_e=3 / surf_add*2 / feature_grains=52 /
 * bs_ow_op=128 register fixes. It tiles over CHANNELS — each channel is independent,
 * so a wide DW layer is split into chunks of Cc channels (multiple of G) that each fit
 * one CBUF pass and run as an independent single DW job (bit-exact; concatenating the
 * chunks == the whole). A single channel whose own feature is too large for one pass
 * would need SPATIAL tiling (not yet implemented) — plan() rejects that case. The
 * whole-fits case is one chunk == the original single job. ROCKET_CONV_DW_NATIVE is
 * accepted as a diagnostic no-op; ROCKET_CONV_DW_GROUP overrides G for sweeps.
 *
 * ctx (optional, may be NULL): a resident-BO pool threaded down to conv2d_one_job so
 * repeat calls / tiles reuse BOs instead of alloc/freeing them per job. NULL keeps the
 * legacy per-call alloc/free. The public rocket_conv2d_fp16 / _ctx wrap this. */
static int conv2d_run(int fd, rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                      const lut_epilogue_t *act,
                      const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    /* Direct OC%16!=0: pad OC up to a multiple of 16 (the conv weight oc group), run the
     * padded conv, then slice the real OC channels off the front of the output. The extra
     * kernels are zero, and each output channel of a direct conv is independent, so the
     * first OC channels are bit-exact. This wraps the WHOLE validated path (tiling, IC
     * pad, ...) once — mirrors the IC<32 pad, just on the output side. (Depthwise has
     * OC==IC==C with C%32==0, so it never reaches here.) */
    if (!d->depthwise && (d->oc % 16)) {
        const int OC = d->oc, OCp = ((OC + 15) / 16) * 16;
        const int IC = d->ic, KH = d->kh, KW = d->kw;
        const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
        if (OH <= 0 || OW <= 0) return -3;
        rocket_conv2d_desc dp = *d; dp.oc = OCp;
        _Float16 *Wp = calloc((size_t)OCp * IC * KH * KW, sizeof(_Float16));
        _Float16 *op = malloc((size_t)OCp * OH * OW * sizeof(_Float16));
        if (!Wp || !op) { free(Wp); free(op); return -1; }
        memcpy(Wp, W, (size_t)OC * IC * KH * KW * sizeof(_Float16));   /* extra kernels stay 0 */
        int r = conv2d_run(fd, ctx, &dp, act, in, Wp, op);
        if (!r) memcpy(out, op, (size_t)OC * OH * OW * sizeof(_Float16)); /* first OC chans */
        free(Wp); free(op);
        return r;
    }

    int ret = rocket_conv2d_plan(d);
    if (ret) return ret;

    const int ICr = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int sy = d->stride_y, sx = d->stride_x, pt = d->pad_top, pl = d->pad_left;
    const int dy = d->dil_y, dx = d->dil_x;

    if (d->depthwise) {
        /* native depthwise (OC==IC==C, IC%G==0). Two nested, independent tilings:
         *   CHANNEL — each channel reduces only itself, so the layer splits into chunks
         *     of Cc channels (a multiple of G). Cc is the largest G-multiple that fits one
         *     CBUF pass (weight in one bank + feature within the validated budget); Cc==C
         *     when the whole fits (one job, == the pre-tiling validated path).
         *   SPATIAL — if even one weight group G of channels overflows the feature budget
         *     (a high-res single channel), each chunk is additionally banded over output
         *     rows/cols by dw_spatial (materialized halo, pad_top=pad_left=0). The input
         *     [C][IH][IW], weight [C][KH][KW] and output [C][OH][OW] are channel-major, so
         *     each channel chunk is a contiguous slice of all three. */
        const int G = conv_dw_group();
        int Cc = ICr;
        while (Cc > G && !dw_chunk_fits(Cc, IH, IW, KH, KW)) Cc -= G;
        const int spatial = !dw_chunk_fits(Cc, IH, IW, KH, KW);   /* even G overflows -> band */
        if (getenv("ROCKET_CONV_DW_DEBUG"))
            ROCKET_LOGD("[rocket_conv dw] C=%d IH=%d IW=%d -> chunk Cc=%d (%d chunk%s)%s\n",
                    ICr, IH, IW, Cc, (ICr + Cc - 1) / Cc, (ICr + Cc - 1) / Cc > 1 ? "s" : "",
                    spatial ? " +spatial-tile" : "");
        ret = 0;
        for (int c0 = 0; c0 < ICr && !ret; c0 += Cc) {
            int Cn = (ICr - c0 < Cc) ? (ICr - c0) : Cc;       /* C%G==0 & Cc%G==0 => Cn%G==0 */
            const _Float16 *in_c = in + (size_t)c0 * IH * IW;
            const _Float16 *W_c  = W  + (size_t)c0 * KH * KW;
            _Float16 *out_c      = out + (size_t)c0 * OH * OW;
            if (spatial)
                ret = dw_spatial(fd, ctx, Cn, IH, IW, OH, OW, KH, KW, sy, sx, pt, pl, dy, dx,
                                 G, in_c, W_c, out_c);
            else
                ret = conv2d_one_job(fd, ctx, Cn, IH, IW, Cn, OH, OW, KH, KW, sy, sx, pt, pl,
                                     dy, dx, 1, G, NULL, in_c, W_c, out_c);
        }
        return ret;
    }

    /* First-layer IC<32 (e.g. RGB): zero-pad input channels up to the weight ic
     * group of 32. The padded channels contribute 0. Build a padded weight once;
     * the input is padded lazily in the per-tile materialization (ic>=ICr -> 0). */
    const int IC = ((ICr + 31) / 32) * 32;
    const _Float16 *Wuse = W;
    _Float16 *Wpad = NULL;
    rocket_arena a = {0};                  /* closed until the tiling section opens it */
    if (IC != ICr) {
        Wpad = calloc((size_t)OC * IC * KH * KW, sizeof(_Float16));
        if (!Wpad) return -1;
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < ICr; ic++)
                memcpy(&Wpad[(((size_t)oc * IC + ic) * KH) * KW],
                       &W[(((size_t)oc * ICr + ic) * KH) * KW],
                       (size_t)KH * KW * sizeof(_Float16));
        Wuse = Wpad;
    }

    /* fast path: no channel pad AND the whole problem fits one CBUF pass. */
    {
        long ih_full = (long)(OH - 1) * sy + (long)(KH - 1) * dy + 1;
        long iw_full = (long)(OW - 1) * sx + (long)(KW - 1) * dx + 1;
        if (IC == ICr &&
            (size_t)OC * IC * KH * KW * sizeof(_Float16) <= (size_t)4 * CBUF_BANK &&
            (size_t)IC * ih_full * iw_full * sizeof(_Float16) <= CONV_FEAT_BUDGET)
            return conv2d_one_job(fd, ctx, IC, IH, IW, OC, OH, OW, KH, KW, sy, sx, pt, pl,
                                  dy, dx, 0, 0, act, in, W, out);
    }

    /* feature bytes for a (rh out-rows x cw out-cols) tile's materialized sub-input */
    #define CONV_FEAT(rh, cw) ((size_t)IC * \
        ((size_t)((rh) - 1) * sy + (size_t)(KH - 1) * dy + 1) * \
        ((size_t)((cw) - 1) * sx + (size_t)(KW - 1) * dx + 1) * sizeof(_Float16))

    /* OC tile (mult-16) keeps the weight tile's banks modest; row/col tiles keep
     * the materialized feature within budget — shrink rows first (full width),
     * then columns if a full-width band still overflows.
     *
     * The per-tile NPU job is HW-validated only for "tall enough" tiles: the
     * smallest validated single job is datain_height 6 / 3 output rows, and a
     * 1-output-row tile (sub-input 3 rows) produces WRONG hardware output even
     * with surf_stride clamped to 0 (the CNA's height-blocked feature DMA is not
     * correct below ~4 input rows). So never shrink rows past the point where the
     * materialized sub-input drops below MIN_IHs rows — narrow the COLUMNS to fit
     * budget instead (ROCKET_CONV_MIN_IH overrides the floor for HW sweeps). */
    int min_ihs = 6;
    { const char *e = getenv("ROCKET_CONV_MIN_IH"); if (e && atoi(e) > 0) min_ihs = atoi(e); }
    if (min_ihs < CONV_MIN_DATAIN_H) min_ihs = CONV_MIN_DATAIN_H;   /* never below the HW row floor */

    /* OC tile (mult-16) and the feature budget JOINTLY share the 12 CBUF banks (the gen
     * gives weight 12 - data_bank banks). Shrink OCt toward the 16 oc-group to keep
     * weight ~4 banks; when even one oc-group's weight cube needs more (a KxK conv at
     * large IC), accept the bigger weight and SHRINK the feature budget so the two still
     * sum to <=12 banks — else the weight overflows its CBUF allocation (the int8 bug,
     * latent here too: fp16 64x64->128 3x3 only passed because its remainder landed on
     * exactly datain_height=4). */
    /* CBUF bank count from the active hardware profile (chip-agnostic; RK3588 today). */
    const int CBUF_BANKS = rocket_hw_current()->cbuf_banks;
    #define WBANKS_F16(oct) (((size_t)(oct) * IC * KH * KW * sizeof(_Float16) + CBUF_BANK - 1) / CBUF_BANK)
    int OCt = OC;
    while (OCt > 16 && WBANKS_F16(OCt) > 4) OCt -= 16;
    size_t wbanks = WBANKS_F16(OCt);
    if (wbanks > (size_t)(CBUF_BANKS - 1)) { ret = -4; goto done; }   /* no bank left for feature */
    size_t feat_budget = (size_t)(CBUF_BANKS - wbanks) * CBUF_BANK;
    if (feat_budget > CONV_FEAT_BUDGET) feat_budget = CONV_FEAT_BUDGET;
    #undef WBANKS_F16

    /* smallest row-tile whose materialized sub-input is >= min_ihs rows (capped at OH). */
    int rht_lo = 1;
    while (rht_lo < OH && datain_h(rht_lo, sy, KH, dy) < min_ihs) rht_lo++;
    int rht = OH, cwt = OW;
    while (rht > rht_lo && CONV_FEAT(rht, cwt) > feat_budget) rht--;
    if (CONV_FEAT(rht, cwt) > feat_budget)             /* full-width band still over -> narrow cols */
        while (cwt > 1 && CONV_FEAT(rht, cwt) > feat_budget) cwt--;
    if (CONV_FEAT(rht, cwt) > feat_budget) { ret = -4; goto done; }   /* minimal tile still over */

    const int IHs = (rht - 1) * sy + (KH - 1) * dy + 1;   /* max sub-input dims */
    const int IWs = (cwt - 1) * sx + (KW - 1) * dx + 1;
    /* BALANCED row/col bands: no tiny remainder stub below CONV_MIN_DATAIN_H. */
    int nRmax = (OH + rht - 1) / rht, nCmax = (OW + cwt - 1) / cwt;
    const size_t szs[4] = { (size_t)nRmax * sizeof(int), (size_t)nCmax * sizeof(int),
                            (size_t)IC * IHs * IWs * sizeof(_Float16),
                            (size_t)OCt * rht * cwt * sizeof(_Float16) };
    if (rocket_arena_open(&a, ctx ? &ctx->scratch : NULL, rocket_arena_reserve(szs, 4)) < 0) {
        ret = -1; goto done; }
    int      *rband   = rocket_arena_push(&a, szs[0]);
    int      *cband   = rocket_arena_push(&a, szs[1]);
    _Float16 *sub_in  = rocket_arena_push(&a, szs[2]);
    _Float16 *sub_out = rocket_arena_push(&a, szs[3]);
    if (!rband || !cband || !sub_in || !sub_out) { ret = -1; goto done; }
    int nR = balance_bands(OH, rht, rband);
    int nC = balance_bands(OW, cwt, cband);
    /* short bands / small-spatial convs are padded to the 4-row floor inside the job. */

    ret = 0;
    for (int oc0 = 0; oc0 < OC && !ret; oc0 += OCt) {
        int OCn = (OC - oc0 < OCt) ? (OC - oc0) : OCt;
        const _Float16 *Wslice = Wuse + (size_t)oc0 * IC * KH * KW; /* [OCn][IC][KH][KW] */
        int r0 = 0;
        for (int ri = 0; ri < nR && !ret; ri++) {
            int rh = rband[ri];
            int ih_sub = (rh - 1) * sy + (KH - 1) * dy + 1;
            int gh0 = r0 * sy - pt;                                /* band top in input rows */
            int c0 = 0;
            for (int ci = 0; ci < nC && !ret; ci++) {
                int cw = cband[ci];
                int iw_sub = (cw - 1) * sx + (KW - 1) * dx + 1;
                int gw0 = c0 * sx - pl;                            /* band left in input cols */

                /* materialize the tile's sub-input: real halo + zero pad, with the
                 * channel pad (ic>=ICr stays zero from the memset). */
                memset(sub_in, 0, (size_t)IC * ih_sub * iw_sub * sizeof(_Float16));
                for (int ic = 0; ic < ICr; ic++)
                    for (int j = 0; j < ih_sub; j++) {
                        int gih = gh0 + j;
                        if (gih < 0 || gih >= IH) continue;
                        int cs = gw0 < 0 ? 0 : gw0;                /* first in-range input col */
                        int ce = gw0 + iw_sub; if (ce > IW) ce = IW;
                        if (cs >= ce) continue;
                        memcpy(&sub_in[((size_t)ic * ih_sub + j) * iw_sub + (cs - gw0)],
                               &in[((size_t)ic * IH + gih) * IW + cs],
                               (size_t)(ce - cs) * sizeof(_Float16));
                    }

                /* sub-conv: both pads materialized -> pad_top=pad_left=0; output
                 * is exactly rh x cw. */
                ret = conv2d_one_job(fd, ctx, IC, ih_sub, iw_sub, OCn, rh, cw, KH, KW,
                                     sy, sx, 0, 0, dy, dx, 0, 0, act, sub_in, Wslice, sub_out);
                if (ret) break;

                for (int oc = 0; oc < OCn; oc++)
                    for (int r = 0; r < rh; r++)
                        memcpy(&out[(((size_t)(oc0 + oc) * OH) + (r0 + r)) * OW + c0],
                               &sub_out[((size_t)oc * rh + r) * cw],
                               (size_t)cw * sizeof(_Float16));
                c0 += cw;
            }
            r0 += rh;
        }
    }

    #undef CONV_FEAT
 done:
    rocket_arena_close(&a);        /* frees an owned frame; resets a borrowed ctx pool */
    free(Wpad);
    return ret;
}

/* ############################################################################
 * PART 4 — fp16 public API: conv2d / conv1d / conv->activation fusion
 * ##########################################################################*/

/* Public entry points: the legacy per-call path (ctx=NULL) and the resident-BO path. */
int rocket_conv2d_fp16(int fd, const rocket_conv2d_desc *d,
                       const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    return conv2d_run(fd, NULL, d, NULL, in, W, out);
}

int rocket_conv2d_fp16_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                           const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    if (!ctx) return -1;
    return conv2d_run(ctx->fd, ctx, d, NULL, in, W, out);   /* ctx->fd may be <0 -> oracle */
}

/* conv1d = a conv2d with the TIME axis on the HEIGHT axis (IW=KW2=OW=1). The [IC][IT]/
 * [OC][IC][KW]/[OC][OT] layouts are byte-identical to the conv2d [IC][IT][1]/[OC][IC][KW][1]/
 * [OC][OT][1] cubes, so this is a pure descriptor build + dispatch (no repacking). Time-on-
 * HEIGHT (not width) is deliberate: the conv tiler tiles output ROWS (OH) first, so a long /
 * many-channel sequence shrinks the per-tile height until the feature fits one CBUF pass — the
 * width-on-time layout (IH=1) instead leaves OH=1 untileable and overflows the feature banks
 * for Whisper's IC=80/512. See rocket_conv.h. */
int rocket_conv1d_fp16(int fd, int ic, int it, int oc, int kw, int stride, int pad,
                       const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    if (ic < 1 || it < 1 || oc < 1 || kw < 1 || stride < 1 || pad < 0) return -1;
    rocket_conv2d_desc d = {
        .ic = ic, .ih = it, .iw = 1, .oc = oc,
        .kh = kw, .kw = 1, .stride_y = stride, .stride_x = 1,
        .pad_top = pad, .pad_left = 0, .dil_y = 1, .dil_x = 1, .depthwise = 0,
    };
    return conv2d_run(fd, NULL, &d, NULL, in, W, out);
}

/* ---- conv -> activation fusion (DIRECT fp16 conv; SiLU/tanh/GELU) --------------
 * Run a DIRECT fp16 conv and apply f(x) in the SAME NPU job: the conv result is
 * post-processed by the DPU LUT epilogue (BN-mul index scale -> EW LUT -> affine
 * OUT_CVT), so out[oc][oh][ow] = f(conv(...)) with NO second NPU round-trip and no
 * host activation pass. `kind` is a SMOOTH single-pass kind (ROCKET_ACTIVATION_SILU /
 * _TANH / _GELU); HardSwish is rejected (its flat x<=-3 tail trips the NVDLA LE/LO
 * mux — keep it on the host / 2-pass path). Depthwise is rejected (direct-only scope).
 * Same shapes/layouts/tiling as rocket_conv2d_fp16. Returns 0, or <0 on error. */
static int conv2d_act_run(int fd, rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                          int kind, const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    if (d->depthwise) return -10;          /* fusion is the direct-conv path only */
    if (kind != ROCKET_ACTIVATION_SILU && kind != ROCKET_ACTIVATION_TANH &&
        kind != ROCKET_ACTIVATION_GELU)
        return -11;                        /* smooth single-pass kinds only         */
    uint16_t lut[1026];                    /* table lives across every tile's gen    */
    lut_epilogue_t ep;
    int b = rocket_lut_epilogue_build(kind, lut, &ep);
    if (b) return b;
    return conv2d_run(fd, ctx, d, &ep, in, W, out);
}

int rocket_conv2d_act_fp16(int fd, const rocket_conv2d_desc *d, int kind,
                           const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    return conv2d_act_run(fd, NULL, d, kind, in, W, out);
}

int rocket_conv2d_act_fp16_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d, int kind,
                               const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    if (!ctx) return -1;
    return conv2d_act_run(ctx->fd, ctx, d, kind, in, W, out);
}

/* ############################################################################
 * PART 5 — Native int8/uint8 DIRECT conv (int32-out): single job, tiling,
 *          batched-submit + multicore fan-out, public API
 * ##########################################################################*/

/* =========================================================================
 * Native int8 DIRECT CONV_2D (int32-raw): the exact-W8A8 sibling of the fp16
 * path above. int8 in/weight -> int32 accumulate on the NPU, host requant. The
 * structure mirrors conv2d_one_job / conv2d_run verbatim with the int8 deltas:
 * feature cube C2=16 (vs fp16 C2=8), weight cube weight_conv_int8 (oc-group 32 vs
 * fp16's 16), int32 output cube C2=4, gen_conv2d_int8, and the int8/int32 BO byte
 * sizes. The fp16 path is byte-for-byte untouched.
 * ========================================================================= */

/* Run ONE int8 DIRECT conv as a single NPU job (must fit one CBUF pass). OC MUST be a
 * multiple of 32 (the int8 weight oc-group) — conv2d_int8_run pads it. in [IC][IH][IW]
 * int8, W [OC][IC][KH][KW] int8, out [OC][OH][OW] int32. Returns 0, or <0 (incl. the
 * generator's -1/-2 when the tile overflows the CBUF — the tiler shrinks and retries). */
static int conv2d_int8_one_job(int fd, rocket_conv_ctx *ctx,
                               int IC, int IH, int IW, int OC, int OH, int OW,
                               int KH, int KW, int sy, int sx, int pt, int pl,
                               int dy, int dx,
                               const int8_t *in, const int8_t *W, int32_t *out)
{
    int ret = -1;   /* the early-error gotos (BO alloc / 32-bit IOVA) return via this */

    /* No device (fd<0): compute on the int64->int32 oracle, so the tiling
     * decomposition is x86-verifiable against the whole-image oracle (the per-tile NPU
     * path is the separately HW-validated single job). Reached before any BO is touched. */
    if (fd < 0) {
        rocket_conv2d_desc sd = { .ic=IC,.ih=IH,.iw=IW,.oc=OC,.kh=KH,.kw=KW,
            .stride_y=sy,.stride_x=sx,.pad_top=pt,.pad_left=pl,.dil_y=dy,.dil_x=dx,
            .depthwise=0 };
        (void)OH; (void)OW;
        rocket_conv2d_ref_int8(&sd, in, W, out);
        return 0;
    }

    /* CNA feature DMA is height-blocked in 4-row blocks; datain_height < 4 reads the cube
     * wrong (HW-confirmed, any IC). Pad the job's input height up to the floor with zero
     * rows below the real data: the cube gets IHj rows (extra rows zeroed by the memset),
     * the regcmd's datain_height is IHj, but OH stays the real output count, so the padded
     * rows lie beyond every computed output row's receptive field — bit-exact. This makes
     * a small-spatial conv (a 3x3/2x2/1x1 SSD head) or any short row-band correct without
     * a separate code path. */
    const int IHj = IH < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : IH;
    size_t in_elems  = (size_t)IC * IHj * IW;
    size_t wt_elems  = (size_t)OC * IC * KH * KW;
    size_t out_elems = (size_t)OC * OH * OW;

    rocket_bo lguard = {0}, lin = {0}, lwt = {0}, lrc = {0}, lout = {0};
    rocket_bo *guard  = ctx ? &ctx->guard : &lguard;
    rocket_bo *in_bo  = ctx ? &ctx->in    : &lin;
    rocket_bo *wt_bo  = ctx ? &ctx->wt    : &lwt;
    rocket_bo *rc_bo  = ctx ? &ctx->rc    : &lrc;
    rocket_bo *out_bo = ctx ? &ctx->out   : &lout;
    uint64_t regs[256] = {0};
    int rc = -1;

    if (conv_bo_ensure(fd, guard, 4096) ||                                       /* off IOVA 0 */
        conv_bo_ensure(fd, in_bo,  in_elems  * sizeof(int8_t)  + CBUF_BANK) ||
        conv_bo_ensure(fd, wt_bo,  wt_elems  * sizeof(int8_t)  + CBUF_BANK) ||
        conv_bo_ensure(fd, rc_bo,  256 * sizeof(uint64_t))                  ||
        conv_bo_ensure(fd, out_bo, out_elems * sizeof(int32_t) + CBUF_BANK)) {
        ROCKET_LOGE("rocket_conv2d_int8: BO alloc failed\n");
        goto out;
    }
    if (((in_bo->dma_address + in_bo->size) | (wt_bo->dma_address + wt_bo->size) |
         (out_bo->dma_address + out_bo->size) | (rc_bo->dma_address + rc_bo->size)) >> 32) {
        ROCKET_LOGE("rocket_conv2d_int8: a BO dma_address exceeds 32 bits\n");
        goto out;
    }

    /* scatter input feature -> NC1HWC2 cube (int8, C2=16) */
    rocket_bo_prep(fd, in_bo, 1, 0);
    memset(in_bo->ptr, 0, in_bo->size);
    {
        int8_t *dst = in_bo->ptr;       /* cube laid out for IHj rows; rows >= IH stay 0 */
        for (int ic = 0; ic < IC; ic++)
            for (int ih = 0; ih < IH; ih++)
                for (int iw = 0; iw < IW; iw++)
                    dst[feature_data(IC, IHj, IW, 16, ic + 1, ih + 1, iw + 1)] =
                        in[((size_t)ic * IH + ih) * IW + iw];
    }
    rocket_bo_fini(fd, in_bo);

    /* scatter weights -> int8 conv weight cube (oc-group 32 / ic-group 32) */
    rocket_bo_prep(fd, wt_bo, 1, 0);
    memset(wt_bo->ptr, 0, wt_bo->size);
    {
        int8_t *dst = wt_bo->ptr;
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < IC; ic++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        dst[weight_conv_int8(OC, IC, KH, KW, oc + 1, ic + 1, kh + 1, kw + 1)] =
                            W[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
    }
    rocket_bo_fini(fd, wt_bo);

    /* generate the int8 conv regcmd */
    {
        conv_params_t p = {
            .ic = IC, .ih = IHj, .iw = IW, .oc = OC, .oh = OH, .ow = OW,
            .kh = KH, .kw = KW, .stride_y = sy, .stride_x = sx,
            .dil_y = dy, .dil_x = dx, .pad_top = pt, .pad_left = pl,
            .input_dma = (uint32_t)in_bo->dma_address,
            .weights_dma = (uint32_t)wt_bo->dma_address,
            .output_dma = (uint32_t)out_bo->dma_address,
            .tasks = regs,
        };
        if ((ret = gen_conv2d_int8(&p)) != 0) {
            ROCKET_LOGE("rocket_conv2d_int8: gen failed (%d)\n", ret);
            goto out;
        }
        if (p.task_count > sizeof(regs)/sizeof(regs[0])) {   /* unconditional: -DNDEBUG strips asserts */
            ROCKET_LOGE("rocket_conv2d_int8: regcmd overflow (task_count %u > %zu words)\n",
                    p.task_count, sizeof(regs)/sizeof(regs[0]));
            ret = -1; goto out;
        }
        rocket_bo_prep(fd, rc_bo, 1, 0);
        memcpy(rc_bo->ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, rc_bo);

        /* zero + hand the output BO to the device */
        rocket_bo_prep(fd, out_bo, 1, 0);
        memset(out_bo->ptr, 0, out_bo->size);
        rocket_bo_fini(fd, out_bo);

        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo->dma_address,
                                  .regcmd_count = p.task_count };
        uint32_t in_h[]  = { in_bo->handle, wt_bo->handle, rc_bo->handle };
        uint32_t out_h[] = { out_bo->handle };
        ret = rocket_submit_tasks(fd, &task, 1, in_h, 3, out_h, 1);
        if (ret) { ROCKET_LOGE("rocket_conv2d_int8: submit failed (%d)\n", ret); goto out; }
    }

    /* read back: de-scatter the int32 output cube (OC/4, OH, OW, 4) */
    ret = rocket_bo_prep(fd, out_bo, 0, 2000000000ULL);   /* 2s wait */
    if (ret) { ROCKET_LOGE("rocket_conv2d_int8: wait timeout (%d)\n", ret); goto out; }
    {
        int32_t *src = out_bo->ptr;
        for (int oc = 0; oc < OC; oc++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                    out[((size_t)oc * OH + oh) * OW + ow] =
                        src[feature_data(OC, OH, OW, 4, oc + 1, oh + 1, ow + 1)];
    }
    rocket_bo_fini(fd, out_bo);
    rc = 0;

out:
    if (!ctx) {
        rocket_bo_free(fd, out_bo);
        rocket_bo_free(fd, rc_bo);
        rocket_bo_free(fd, wt_bo);
        rocket_bo_free(fd, in_bo);
        rocket_bo_free(fd, guard);
    }
    return rc ? (ret ? ret : -1) : 0;
}

/* One independent OC/OH/OW tile of the int8 DIRECT conv: materialize the tile's
 * sub-input (real halo + zero pad + channel pad), run the HW-validated single job,
 * scatter the int32 result into out's disjoint region. Shared verbatim by the serial
 * loop and the multicore workers (each worker passes its own fd/ctx/scratch), so the
 * per-tile math is identical to the original in-loop body. sub_in/sub_out must be
 * sized to the largest tile (IC*IHs*IWs / OCt*rht*cwt). Returns 0 / <0. */
static int conv2d_int8_one_tile(int fd, rocket_conv_ctx *ctx,
                                int IC, int ICr, int IH, int IW, int OH, int OW,
                                int KH, int KW, int sy, int sx, int pt, int pl,
                                int dy, int dx, const int8_t *in, const int8_t *Wuse,
                                int32_t *out, int oc0, int OCn, int r0, int rh,
                                int c0, int cw, int8_t *sub_in, int32_t *sub_out)
{
    int ih_sub = (rh - 1) * sy + (KH - 1) * dy + 1;
    int gh0    = r0 * sy - pt;
    int iw_sub = (cw - 1) * sx + (KW - 1) * dx + 1;
    int gw0    = c0 * sx - pl;

    /* materialize the tile's sub-input: real halo + zero pad, with the channel pad
     * (ic>=ICr stays zero from the memset). */
    memset(sub_in, 0, (size_t)IC * ih_sub * iw_sub * sizeof(int8_t));
    for (int ic = 0; ic < ICr; ic++)
        for (int j = 0; j < ih_sub; j++) {
            int gih = gh0 + j;
            if (gih < 0 || gih >= IH) continue;
            int cs = gw0 < 0 ? 0 : gw0;
            int ce = gw0 + iw_sub; if (ce > IW) ce = IW;
            if (cs >= ce) continue;
            memcpy(&sub_in[((size_t)ic * ih_sub + j) * iw_sub + (cs - gw0)],
                   &in[((size_t)ic * IH + gih) * IW + cs],
                   (size_t)(ce - cs) * sizeof(int8_t));
        }

    const int8_t *Wslice = Wuse + (size_t)oc0 * IC * KH * KW;   /* [OCn][IC][KH][KW] */
    int ret = conv2d_int8_one_job(fd, ctx, IC, ih_sub, iw_sub, OCn, rh, cw, KH, KW,
                                  sy, sx, 0, 0, dy, dx, sub_in, Wslice, sub_out);
    if (ret) return ret;

    for (int oc = 0; oc < OCn; oc++)
        for (int r = 0; r < rh; r++)
            memcpy(&out[(((size_t)(oc0 + oc) * OH) + (r0 + r)) * OW + c0],
                   &sub_out[((size_t)oc * rh + r) * cw],
                   (size_t)cw * sizeof(int32_t));
    return 0;
}

/* an independent tile of the OC/OH/OW decomposition */
typedef struct { int oc0, OCn, r0, rh, c0, cw; } i8tile;

/* Batch tiles[t0..t0+tn) into ONE multi-task int8 conv job on fd/ctx (lever-1, gapped;
 * see CONV_RC_STRIDE). ctx holds the (grown) batched input/weight/regcmd/output BOs; each
 * tile lands at a bank-aligned, zeroed slot so its materialize/scatter/gen/de-scatter is
 * byte-for-byte conv2d_int8_one_tile + _one_job — only the per-tile submit+fence is
 * coalesced into one. The whole-BO zero (gaps + the per-tile slack bank) keeps the int8
 * feature-DMA +1-bank over-read reading zeros, exactly as the standalone job. The job's
 * tasks are separate HW kicks (CACC clears per kick), so int8 stays bit-exact. 0 / <0. */
static int conv2d_int8_batch_tiles(int fd, rocket_conv_ctx *ctx,
        int IC, int ICr, int IH, int IW, int OH, int OW, int KH, int KW,
        int sy, int sx, int pt, int pl, int dy, int dx,
        const int8_t *in, const int8_t *Wuse, int32_t *out,
        const i8tile *tiles, int t0, int tn)
{
    if (tn <= 0) return 0;

    size_t *off_in  = malloc((size_t)tn * sizeof(size_t));
    size_t *off_wt  = malloc((size_t)tn * sizeof(size_t));
    size_t *off_out = malloc((size_t)tn * sizeof(size_t));
    rocket_task_desc *tasks = malloc((size_t)tn * sizeof(rocket_task_desc));
    int8_t *sub_in = NULL; void *scratch = NULL;
    if (!off_in || !off_wt || !off_out || !tasks) { goto oom; }

    /* Pass 1: per-tile bank-aligned offsets + the largest row-major sub-input scratch. */
    size_t in_tot = 0, wt_tot = 0, out_tot = 0, max_sub_in = 0;
    for (int k = 0; k < tn; k++) {
        const i8tile *tl = &tiles[t0 + k];
        int ih_sub = (tl->rh - 1) * sy + (KH - 1) * dy + 1;
        int iw_sub = (tl->cw - 1) * sx + (KW - 1) * dx + 1;
        int IHj = ih_sub < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : ih_sub;
        size_t sin = (size_t)IC * ih_sub * iw_sub;
        if (sin > max_sub_in) max_sub_in = sin;
        off_in[k]  = in_tot;   in_tot  += bank_round((size_t)IC * IHj * iw_sub) + CBUF_BANK;
        off_wt[k]  = wt_tot;   wt_tot  += bank_round((size_t)tl->OCn * IC * KH * KW) + CBUF_BANK;
        off_out[k] = out_tot;  out_tot += bank_round((size_t)tl->OCn * tl->rh * tl->cw * sizeof(int32_t)) + CBUF_BANK;
    }
    sub_in = malloc(max_sub_in ? max_sub_in : 1);
    if (!sub_in) goto oom;

    int ret = -1;
    rocket_bo *guard = &ctx->guard, *in_bo = &ctx->in, *wt_bo = &ctx->wt,
              *rc_bo = &ctx->rc,    *out_bo = &ctx->out;
    if (conv_bo_ensure(fd, guard, 4096) ||
        conv_bo_ensure(fd, in_bo,  in_tot)  ||
        conv_bo_ensure(fd, wt_bo,  wt_tot)  ||
        conv_bo_ensure(fd, rc_bo,  (size_t)tn * CONV_RC_STRIDE * sizeof(uint64_t)) ||
        conv_bo_ensure(fd, out_bo, out_tot)) {
        ROCKET_LOGE("conv int8 batch: BO alloc failed\n"); goto out;
    }
    if (((in_bo->dma_address + in_bo->size) | (wt_bo->dma_address + wt_bo->size) |
         (out_bo->dma_address + out_bo->size) | (rc_bo->dma_address + rc_bo->size)) >> 32) {
        ROCKET_LOGE("conv int8 batch: a BO dma_address exceeds 32 bits\n"); goto out;
    }
    scratch = malloc(rocket_submit_scratch_size((uint32_t)tn));
    if (!scratch) { ROCKET_LOGE("conv int8 batch: scratch alloc failed\n"); goto out; }

    /* input cubes: materialize each tile's halo/pad sub-input, scatter to its NC1HWC2 slot */
    rocket_bo_prep(fd, in_bo, 1, 0);
    memset(in_bo->ptr, 0, in_bo->size);
    for (int k = 0; k < tn; k++) {
        const i8tile *tl = &tiles[t0 + k];
        int ih_sub = (tl->rh - 1) * sy + (KH - 1) * dy + 1;
        int iw_sub = (tl->cw - 1) * sx + (KW - 1) * dx + 1;
        int IHj = ih_sub < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : ih_sub;
        int gh0 = tl->r0 * sy - pt, gw0 = tl->c0 * sx - pl;
        memset(sub_in, 0, (size_t)IC * ih_sub * iw_sub);
        for (int ic = 0; ic < ICr; ic++)
            for (int j = 0; j < ih_sub; j++) {
                int gih = gh0 + j;
                if (gih < 0 || gih >= IH) continue;
                int cs = gw0 < 0 ? 0 : gw0, ce = gw0 + iw_sub; if (ce > IW) ce = IW;
                if (cs >= ce) continue;
                memcpy(&sub_in[((size_t)ic * ih_sub + j) * iw_sub + (cs - gw0)],
                       &in[((size_t)ic * IH + gih) * IW + cs], (size_t)(ce - cs));
            }
        int8_t *dst = (int8_t *)in_bo->ptr + off_in[k];
        for (int ic = 0; ic < IC; ic++)
            for (int ih = 0; ih < ih_sub; ih++)
                for (int iw = 0; iw < iw_sub; iw++)
                    dst[feature_data(IC, IHj, iw_sub, 16, ic + 1, ih + 1, iw + 1)] =
                        sub_in[((size_t)ic * ih_sub + ih) * iw_sub + iw];
    }
    rocket_bo_fini(fd, in_bo);

    /* weight cubes: each tile's OC-slice */
    rocket_bo_prep(fd, wt_bo, 1, 0);
    memset(wt_bo->ptr, 0, wt_bo->size);
    for (int k = 0; k < tn; k++) {
        const i8tile *tl = &tiles[t0 + k];
        const int8_t *Wslice = Wuse + (size_t)tl->oc0 * IC * KH * KW;
        int8_t *dst = (int8_t *)wt_bo->ptr + off_wt[k];
        for (int oc = 0; oc < tl->OCn; oc++)
            for (int ic = 0; ic < IC; ic++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        dst[weight_conv_int8(tl->OCn, IC, KH, KW, oc + 1, ic + 1, kh + 1, kw + 1)] =
                            Wslice[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
    }
    rocket_bo_fini(fd, wt_bo);

    /* regcmds: a complete gen_conv2d_int8 per tile in its gapped slot */
    rocket_bo_prep(fd, rc_bo, 1, 0);
    for (int k = 0; k < tn; k++) {
        const i8tile *tl = &tiles[t0 + k];
        int ih_sub = (tl->rh - 1) * sy + (KH - 1) * dy + 1;
        int iw_sub = (tl->cw - 1) * sx + (KW - 1) * dx + 1;
        int IHj = ih_sub < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : ih_sub;
        uint64_t regs[CONV_RC_STRIDE] = {0};
        conv_params_t p = {
            .ic = IC, .ih = IHj, .iw = iw_sub, .oc = tl->OCn, .oh = tl->rh, .ow = tl->cw,
            .kh = KH, .kw = KW, .stride_y = sy, .stride_x = sx, .dil_y = dy, .dil_x = dx,
            .pad_top = 0, .pad_left = 0,             /* halo materialized -> no HW pad */
            .input_dma   = (uint32_t)(in_bo->dma_address  + off_in[k]),
            .weights_dma = (uint32_t)(wt_bo->dma_address  + off_wt[k]),
            .output_dma  = (uint32_t)(out_bo->dma_address + off_out[k]),
            .tasks = regs,
        };
        if ((ret = gen_conv2d_int8(&p)) != 0) { ROCKET_LOGE("conv int8 batch: gen %d\n", ret); goto out; }
        if (p.task_count > CONV_RC_STRIDE) {
            ROCKET_LOGE("conv int8 batch: regcmd overflow (%u > %d)\n", p.task_count, CONV_RC_STRIDE);
            ret = -1; goto out;
        }
        rkt_chain_pack(0, rc_bo, tasks, k, regs, p.task_count, CONV_RC_STRIDE);
    }
    rocket_bo_fini(fd, rc_bo);

    rocket_bo_prep(fd, out_bo, 1, 0);
    memset(out_bo->ptr, 0, out_bo->size);
    rocket_bo_fini(fd, out_bo);

    {
        uint32_t in_h[]  = { in_bo->handle, wt_bo->handle, rc_bo->handle };
        uint32_t out_h[] = { out_bo->handle };
        ret = rocket_submit_tasks_pre(fd, scratch, tasks, (uint32_t)tn, in_h, 3, out_h, 1, 0);
        if (ret) { ROCKET_LOGE("conv int8 batch: submit %d\n", ret); goto out; }
    }
    ret = rocket_bo_prep(fd, out_bo, 0, 2000000000ULL);   /* ONE wait for the whole job */
    if (ret) { ROCKET_LOGE("conv int8 batch: wait %d\n", ret); goto out; }

    for (int k = 0; k < tn; k++) {                        /* de-scatter every tile's output */
        const i8tile *tl = &tiles[t0 + k];
        const int32_t *src = (const int32_t *)((const int8_t *)out_bo->ptr + off_out[k]);
        for (int oc = 0; oc < tl->OCn; oc++)
            for (int oh = 0; oh < tl->rh; oh++)
                for (int ow = 0; ow < tl->cw; ow++)
                    out[(((size_t)(tl->oc0 + oc) * OH) + (tl->r0 + oh)) * OW + tl->c0 + ow] =
                        src[feature_data(tl->OCn, tl->rh, tl->cw, 4, oc + 1, oh + 1, ow + 1)];
    }
    rocket_bo_fini(fd, out_bo);
    ret = 0;
out:
    free(scratch); free(sub_in); free(off_in); free(off_wt); free(off_out); free(tasks);
    return ret;
oom:
    free(scratch); free(sub_in); free(off_in); free(off_wt); free(off_out); free(tasks);
    return -1;
}

/* one multicore worker: strided slice of the tile list on its own fd/ctx + scratch */
typedef struct {
    int fd; rocket_conv_ctx *ctx;
    int IC, ICr, IH, IW, OH, OW, KH, KW, sy, sx, pt, pl, dy, dx;
    const int8_t *in; const int8_t *Wuse; int32_t *out;
    const i8tile *tiles; int ntiles; int wstart, wstride;
    size_t sub_in_sz, sub_out_elems;
    int idx; int ret;
    int batch, wcount;                           /* batch: one job over the [wstart,wstart+wcount) run */
    int core_base;                               /* big-core rotation base inherited from caller thread */
} i8w_arg;

static void *i8_worker(void *a)
{
    i8w_arg *w = (i8w_arg *)a;
    rocket_pin_worker_based(w->idx, w->core_base); /* keep the materialize/scatter off the A55s */
    if (w->batch) {                              /* coalesce this worker's contiguous run into one job */
        w->ret = conv2d_int8_batch_tiles(w->fd, w->ctx, w->IC, w->ICr, w->IH, w->IW, w->OH,
                                         w->OW, w->KH, w->KW, w->sy, w->sx, w->pt, w->pl,
                                         w->dy, w->dx, w->in, w->Wuse, w->out,
                                         w->tiles, w->wstart, w->wcount);
        return NULL;
    }
    int8_t  *sub_in  = malloc(w->sub_in_sz);
    int32_t *sub_out = malloc(w->sub_out_elems * sizeof(int32_t));
    if (!sub_in || !sub_out) { free(sub_in); free(sub_out); w->ret = -1; return NULL; }
    int ret = 0;
    for (int t = w->wstart; t < w->ntiles && !ret; t += w->wstride) {
        const i8tile *tl = &w->tiles[t];
        ret = conv2d_int8_one_tile(w->fd, w->ctx, w->IC, w->ICr, w->IH, w->IW, w->OH,
                                   w->OW, w->KH, w->KW, w->sy, w->sx, w->pt, w->pl,
                                   w->dy, w->dx, w->in, w->Wuse, w->out,
                                   tl->oc0, tl->OCn, tl->r0, tl->rh, tl->c0, tl->cw,
                                   sub_in, sub_out);
    }
    free(sub_in); free(sub_out);
    w->ret = ret;
    return NULL;
}

/* Run the int8 DIRECT conv on the NPU, tiling over OC (mult-32) + OH-rows + OW-cols
 * exactly like conv2d_run. Each tile is an independent HW-validated single job;
 * OH-band tiles MATERIALIZE their edge padding (the CNA only has symmetric pad), and
 * the min_ihs floor keeps every tile inside the HW-validated feature-DMA envelope (the
 * fp16 datain_height<4 lesson — the int8 path inherits the same geometry). in/W are
 * int8, out is int32. When pool!=NULL the independent tiles fan out across the pool's
 * worker fds/ctxs (multicore); pool==NULL keeps the serial single-fd path. Returns
 * 0 / <0. */
static int conv2d_int8_run(int fd, rocket_conv_ctx *ctx, rocket_conv_pool *pool,
                           const rocket_conv2d_desc *d,
                           const int8_t *in, const int8_t *W, int32_t *out)
{
    if (d->depthwise) return -2;   /* int8 depthwise is rocket_conv2d_dw_int8 (int8-out) */

    /* OC%32!=0: pad OC up to the int8 weight oc-group (32), run, slice the real OC off
     * the front. The extra kernels are zero and each direct output channel is
     * independent, so the first OC channels are bit-exact (mirrors the fp16 OC%16 pad,
     * just on the int8 oc-group). */
    if (d->oc % 32) {
        const int OC = d->oc, OCp = ((OC + 31) / 32) * 32;
        const int IC = d->ic, KH = d->kh, KW = d->kw;
        const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
        if (OH <= 0 || OW <= 0) return -3;
        rocket_conv2d_desc dp = *d; dp.oc = OCp;
        int8_t  *Wp = calloc((size_t)OCp * IC * KH * KW, 1);
        int32_t *op = malloc((size_t)OCp * OH * OW * sizeof(int32_t));
        if (!Wp || !op) { free(Wp); free(op); return -1; }
        memcpy(Wp, W, (size_t)OC * IC * KH * KW);                  /* extra kernels stay 0 */
        int r = conv2d_int8_run(fd, ctx, pool, &dp, in, Wp, op);
        if (!r) memcpy(out, op, (size_t)OC * OH * OW * sizeof(int32_t)); /* first OC chans */
        free(Wp); free(op);
        return r;
    }

    const int ICr = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int sy = d->stride_y, sx = d->stride_x, pt = d->pad_top, pl = d->pad_left;
    const int dy = d->dil_y, dx = d->dil_x;
    if (OH <= 0 || OW <= 0) return -3;
    if (OC <= 0 || KH <= 0 || KW <= 0 || IH <= 0 || IW <= 0 || ICr <= 0) return -1;
    if (sy <= 0 || sx <= 0 || dy <= 0 || dx <= 0) return -1;

    /* First-layer IC<32 (RGB): zero-pad input channels up to the ic group of 32 (the
     * padded channels contribute 0). Padded weight built once; input padded lazily in
     * the per-tile materialization (ic>=ICr -> 0). */
    const int IC = ((ICr + 31) / 32) * 32;
    /* a single OC kernel must fit one CBUF bank (int8: 1 B/elem) */
    if ((size_t)KW * KH * IC * sizeof(int8_t) > CBUF_BANK) return -4;
    const int8_t *Wuse = W;
    int8_t *Wpad = NULL;
    if (IC != ICr) {
        Wpad = calloc((size_t)OC * IC * KH * KW, 1);
        if (!Wpad) return -1;
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < ICr; ic++)
                memcpy(&Wpad[(((size_t)oc * IC + ic) * KH) * KW],
                       &W[(((size_t)oc * ICr + ic) * KH) * KW],
                       (size_t)KH * KW * sizeof(int8_t));
        Wuse = Wpad;
    }

    /* fast path: no channel pad AND the whole problem fits one CBUF pass. (int8 weight
     * tile <= 4 banks, feature <= the 8-bank feature budget — both in int8 bytes.) */
    {
        long ih_full = (long)(OH - 1) * sy + (long)(KH - 1) * dy + 1;
        long iw_full = (long)(OW - 1) * sx + (long)(KW - 1) * dx + 1;
        if (IC == ICr &&
            (size_t)OC * IC * KH * KW * sizeof(int8_t) <= (size_t)4 * CBUF_BANK &&
            (size_t)IC * ih_full * iw_full * sizeof(int8_t) <= CONV_FEAT_BUDGET_I8)
            return conv2d_int8_one_job(fd, ctx, IC, IH, IW, OC, OH, OW, KH, KW,
                                       sy, sx, pt, pl, dy, dx, in, W, out);
    }

    /* feature bytes for a (rh out-rows x cw out-cols) tile's materialized sub-input (int8) */
    #define CONV_FEAT_I8(rh, cw) ((size_t)IC * \
        ((size_t)((rh) - 1) * sy + (size_t)(KH - 1) * dy + 1) * \
        ((size_t)((cw) - 1) * sx + (size_t)(KW - 1) * dx + 1) * sizeof(int8_t))

    int min_ihs = 6;
    { const char *e = getenv("ROCKET_CONV_MIN_IH"); if (e && atoi(e) > 0) min_ihs = atoi(e); }
    if (min_ihs < CONV_MIN_DATAIN_H) min_ihs = CONV_MIN_DATAIN_H;   /* never below the HW row floor */

    /* OC tile (mult-32) and the feature budget JOINTLY share the 12 CBUF banks: the gen
     * gives weight (12 - data_bank) banks, so a tile is valid only when its weight cube
     * fits the banks the feature leaves. Prefer weight within ~4 banks (feature ~8) by
     * shrinking OCt toward the 32 oc-group; when even one oc-group's weight needs more (a
     * KxK conv at large IC, e.g. 3x3 IC=768 -> 7 weight banks), accept the bigger weight
     * and SHRINK the feature budget so the two still sum to <=12 banks. (The bug: the old
     * fixed 8/4 split left weight 4 banks but the 32-group needed 7 -> CBUF overflow.) */
    /* CBUF bank count from the active hardware profile (chip-agnostic; RK3588 today). */
    const int CBUF_BANKS = rocket_hw_current()->cbuf_banks;
    #define WBANKS_I8(oct) (((size_t)(oct) * IC * KH * KW + CBUF_BANK - 1) / CBUF_BANK)
    int OCt = OC;
    while (OCt > 32 && WBANKS_I8(OCt) > 4) OCt -= 32;
    size_t wbanks = WBANKS_I8(OCt);
    if (wbanks > (size_t)(CBUF_BANKS - 2)) { free(Wpad); return -4; }   /* leave >=1 feature bank + the slack */
    /* Reserve ONE feature slack bank (gen_conv2d_int8_fill sets data_bank = fd_banks+1):
     * feature_banks + 1 + weight_banks <= 12.  */
    size_t feat_budget = (size_t)(CBUF_BANKS - 1 - wbanks) * CBUF_BANK;
    if (feat_budget > CONV_FEAT_BUDGET_I8) feat_budget = CONV_FEAT_BUDGET_I8;

    int rht_lo = 1;
    while (rht_lo < OH && datain_h(rht_lo, sy, KH, dy) < min_ihs) rht_lo++;
    int rht = OH, cwt = OW;
    while (rht > rht_lo && CONV_FEAT_I8(rht, cwt) > feat_budget) rht--;
    if (CONV_FEAT_I8(rht, cwt) > feat_budget)          /* full-width band still over -> narrow cols */
        while (cwt > 1 && CONV_FEAT_I8(rht, cwt) > feat_budget) cwt--;
    if (CONV_FEAT_I8(rht, cwt) > feat_budget) { free(Wpad); return -4; }   /* minimal tile still over */

    const int IHs = (rht - 1) * sy + (KH - 1) * dy + 1;
    const int IWs = (cwt - 1) * sx + (KW - 1) * dx + 1;
    const size_t sub_in_sz     = (size_t)IC * IHs * IWs * sizeof(int8_t);
    const size_t sub_out_elems = (size_t)OCt * rht * cwt;
    #undef CONV_FEAT_I8
    #undef WBANKS_I8

    /* BALANCED row/col bands: split OH/OW into equal-ish bands (no tiny remainder stub
     * that could fall below CONV_MIN_DATAIN_H). The balanced minimum is the largest
     * achievable, so if it still underflows the row floor the shape is untileable. */
    int nOC = (OC + OCt - 1) / OCt;
    int nRmax = (OH + rht - 1) / rht, nCmax = (OW + cwt - 1) / cwt;
    int *rband = malloc((size_t)nRmax * sizeof(int));
    int *cband = malloc((size_t)nCmax * sizeof(int));
    if (!rband || !cband) { free(rband); free(cband); free(Wpad); return -1; }
    int nR = balance_bands(OH, rht, rband);
    int nC = balance_bands(OW, cwt, cband);
    /* a short band (or small-spatial conv) is padded to the 4-row floor inside the single
     * job (conv2d_int8_one_job IHj), so no band needs to clear it here. */

    /* enumerate the independent (oc-group x out-row-band x out-col-band) tiles. Each
     * writes a disjoint region of `out`, so they can run concurrently across cores. */
    int ntiles = nOC * nR * nC;
    i8tile *tiles = malloc((size_t)ntiles * sizeof(i8tile));
    if (!tiles) { free(rband); free(cband); free(Wpad); return -1; }
    int ti = 0;
    for (int oc0 = 0; oc0 < OC; oc0 += OCt) {
        int OCn = (OC - oc0 < OCt) ? (OC - oc0) : OCt;   /* OC%32==0 & OCt%32==0 => OCn%32==0 */
        int r0 = 0;
        for (int ri = 0; ri < nR; ri++) {
            int rh = rband[ri], c0 = 0;
            for (int ci = 0; ci < nC; ci++) {
                int cw = cband[ci];
                tiles[ti++] = (i8tile){ oc0, OCn, r0, rh, c0, cw };
                c0 += cw;
            }
            r0 += rh;
        }
    }
    ntiles = ti;
    free(rband); free(cband);

    int ret = 0;
    int batch = conv_batch_enabled();
    int nw = (pool && pool->n > 1 && ntiles > 1) ? pool->n : 0;
    if (nw > ntiles) nw = ntiles;
    if (nw > 1) {
        /* multicore: fan the disjoint tiles across the pool's worker fds/ctxs. When
         * batching, each worker takes a CONTIGUOUS run and coalesces it into one job
         * (per-tile strided submits otherwise). */
        pthread_t th[ROCKET_CONV_POOL_MAX];
        i8w_arg   args[ROCKET_CONV_POOL_MAX];
        int joinable[ROCKET_CONV_POOL_MAX] = {0};
        int base = ntiles / nw, extra = ntiles % nw, cstart = 0;
        int affbase = rocket_affinity_get_base();  /* spread in-process pools across the cluster */
        for (int w = 0; w < nw; w++) {
            int wcount = base + (w < extra ? 1 : 0);   /* balanced contiguous run */
            args[w] = (i8w_arg){ pool->fd[w], pool->ctx[w], IC, ICr, IH, IW, OH, OW,
                                 KH, KW, sy, sx, pt, pl, dy, dx, in, Wuse, out,
                                 tiles, ntiles, batch ? cstart : w, batch ? 1 : nw,
                                 sub_in_sz, sub_out_elems, w, 0, batch, wcount, affbase };
            cstart += wcount;
            if (pthread_create(&th[w], NULL, i8_worker, &args[w]) == 0) joinable[w] = 1;
            else i8_worker(&args[w]);            /* spawn failed -> run this slice inline */
        }
        for (int w = 0; w < nw; w++) {
            if (joinable[w]) pthread_join(th[w], NULL);
            if (args[w].ret) ret = args[w].ret;
        }
    } else if (batch) {
        /* serial single-fd: coalesce ALL tiles into one job. Needs a ctx for the batched
         * BOs; the NULL-ctx public entry gets a transient one (its BOs freed here). */
        rocket_conv_ctx local = {0}; local.fd = fd;
        rocket_conv_ctx *bctx = ctx ? ctx : &local;
        ret = conv2d_int8_batch_tiles(fd, bctx, IC, ICr, IH, IW, OH, OW, KH, KW,
                                      sy, sx, pt, pl, dy, dx, in, Wuse, out, tiles, 0, ntiles);
        if (!ctx) {
            rocket_bo_free(fd, &local.out); rocket_bo_free(fd, &local.rc);
            rocket_bo_free(fd, &local.wt);  rocket_bo_free(fd, &local.in);
            rocket_bo_free(fd, &local.guard);
        }
    } else {
        /* serial: one scratch pair on the borrowed ctx/fd (pool==NULL or a single tile) */
        int8_t  *sub_in  = malloc(sub_in_sz);
        int32_t *sub_out = malloc(sub_out_elems * sizeof(int32_t));
        if (!sub_in || !sub_out) { free(sub_in); free(sub_out); free(tiles); free(Wpad); return -1; }
        for (int t = 0; t < ntiles && !ret; t++) {
            const i8tile *tl = &tiles[t];
            ret = conv2d_int8_one_tile(fd, ctx, IC, ICr, IH, IW, OH, OW, KH, KW,
                                       sy, sx, pt, pl, dy, dx, in, Wuse, out,
                                       tl->oc0, tl->OCn, tl->r0, tl->rh, tl->c0, tl->cw,
                                       sub_in, sub_out);
        }
        free(sub_in); free(sub_out);
    }

    free(tiles); free(Wpad);
    return ret;
}

int rocket_conv2d_int8(int fd, const rocket_conv2d_desc *d,
                       const int8_t *in, const int8_t *W, int32_t *out)
{
    return conv2d_int8_run(fd, NULL, NULL, d, in, W, out);
}

/* Multicore: fan the DIRECT conv's independent OC/OH/OW tiles across the pool's
 * worker fds (one NPU core each). Falls back to serial (worker 0) for single-tile
 * convs. Bit-identical to rocket_conv2d_int8 — same tiles, same single jobs. */
int rocket_conv2d_int8_mt(rocket_conv_pool *pool, const rocket_conv2d_desc *d,
                          const int8_t *in, const int8_t *W, int32_t *out)
{
    if (!pool || pool->n < 1) return -1;
    return conv2d_int8_run(pool->fd[0], pool->ctx[0], pool, d, in, W, out);
}

int rocket_conv2d_int8_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                           const int8_t *in, const int8_t *W, int32_t *out)
{
    if (!ctx) return -1;
    return conv2d_int8_run(ctx->fd, ctx, NULL, d, in, W, out);  /* ctx->fd may be <0 -> oracle */
}

/* ############################################################################
 * PART 6 — Native int8 DEPTHWISE conv (int8-out, on-chip requant)
 * ##########################################################################*/

/* =========================================================================
 * Native int8 DEPTHWISE CONV_2D (int8-OUT, on-chip requant) — the Teflon-cracked
 * path (gen_conv2d_dw_int8 int8_out=1, HW-validated bit-exact vs Mesa/Teflon by
 * replay_dw_mesa). PER-TENSOR quant only. Everything runs in Mesa's uint8-centered
 * domain: the host scatters (in_byte - 0x80) and (w_byte - 0x80) cubes, folds Mesa's
 * zero-point correction into the per-OC int32 bias cube, the NPU requants on-chip, and
 * the host reads back int8 + 0x80 (the model domain). The domain constants are pinned
 * empirically against the capture (correction = Σ_kernel(w_u8 - w_zp)*(in_zp - 0x80),
 * output = npu + 0x80). G = 64 (Mesa's int8 DW group). The CPU oracle (fd<0) computes
 * the exact int8 conv + TFLite requant (the delegate's reference path).
 * ========================================================================= */
#define DW_INT8_G 64

/* CPU reference for the int8 DW int8-out path (fd<0 fallback + a self-check oracle):
 * the exact TFLite int8 depthwise — acc = Σ(in_q - in_zp)*(w_q - w_zp) + bias_q, then
 * requant real = in_scale*w_scale*acc; q = clamp(lrintf(real/out_scale)+out_zp). NOTE
 * this is the TFLite math; Teflon's NPU int8 DW can differ from it by a few ULP on
 * saturating inputs (its requant rounding), so the HW gate is capture-replay, not this. */
static void dw_int8_ref(const rocket_conv2d_desc *d, const int8_t *in, const int8_t *w,
                        const int32_t *bias, float in_scale, float w_scale, float out_scale,
                        int in_zp, int w_zp, int out_zp, int8_t *out)
{
    const int C = d->ic, IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int32_t acc = bias ? bias[c] : 0;
                for (int kh = 0; kh < KH; kh++) {
                    int ih = oh * d->stride_y + kh * d->dil_y - d->pad_top;
                    for (int kw = 0; kw < KW; kw++) {
                        int iw = ow * d->stride_x + kw * d->dil_x - d->pad_left;
                        int q = (ih >= 0 && ih < IH && iw >= 0 && iw < IW)
                              ? in[((size_t)c * IH + ih) * IW + iw] : in_zp;
                        int wq = w[((size_t)c * KH + kh) * KW + kw];
                        acc += (q - in_zp) * (wq - w_zp);
                    }
                }
                float v = (in_scale * w_scale) * (float)acc;
                long q = (long)lrintf(v / out_scale) + out_zp;
                if (q < -128) q = -128;
                if (q > 127)  q = 127;
                out[((size_t)c * OH + oh) * OW + ow] = (int8_t)q;
            }
}

/* Run ONE int8 DW int8-out job (single CBUF pass). C channels (padded to G in the cube),
 * one int8 filter per channel, per-tensor quant. in [C][IH][IW] int8, w [C][KH][KW] int8,
 * bias_q [C] int32 (may be NULL), out [C][OH][OW] int8 (model domain). Returns 0 / <0. */
static int conv2d_dw_int8_one_job(int fd, rocket_conv_ctx *ctx,
                                  int C, int IH, int IW, int OH, int OW, int KH, int KW,
                                  int sy, int sx, int pt, int pl, int dy, int dx,
                                  const int8_t *in, const int8_t *w, const int32_t *bias_q,
                                  float in_scale, float w_scale, float out_scale,
                                  int in_zp, int w_zp, int out_zp, int8_t *out)
{
    const int G = DW_INT8_G;
    const int Cpad = ((C + G - 1) / G) * G;
    int ret = -1;

    if (fd < 0) {
        rocket_conv2d_desc sd = { .ic=C,.ih=IH,.iw=IW,.oc=C,.kh=KH,.kw=KW,
            .stride_y=sy,.stride_x=sx,.pad_top=pt,.pad_left=pl,.dil_y=dy,.dil_x=dx,.depthwise=1 };
        (void)OH; (void)OW;
        dw_int8_ref(&sd, in, w, bias_q, in_scale, w_scale, out_scale, in_zp, w_zp, out_zp, out);
        return 0;
    }

    /* Pad input height to the 4-row CNA DMA floor (see conv2d_int8_one_job): padded rows
     * sit below the real data, beyond every computed output's receptive field, so a
     * small-spatial DW (a deep EfficientDet block) is correct. IHj==IH when IH>=4. */
    const int IHj = IH < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : IH;
    size_t in_elems   = (size_t)C * IHj * IW;
    size_t wt_elems   = (size_t)Cpad * KH * KW;
    size_t out_elems  = (size_t)C * OH * OW;

    rocket_bo lguard = {0}, lin = {0}, lwt = {0}, lrc = {0}, lout = {0}, lbias = {0};
    rocket_bo *guard  = ctx ? &ctx->guard : &lguard;
    rocket_bo *in_bo  = ctx ? &ctx->in    : &lin;
    rocket_bo *wt_bo  = ctx ? &ctx->wt    : &lwt;
    rocket_bo *rc_bo  = ctx ? &ctx->rc    : &lrc;
    rocket_bo *out_bo = ctx ? &ctx->out   : &lout;
    rocket_bo *bs_bo  = ctx ? &ctx->bias  : &lbias;
    uint64_t regs[256] = {0};
    int rc = -1;

    if (conv_bo_ensure(fd, guard, 4096) ||
        conv_bo_ensure(fd, in_bo,  in_elems  * sizeof(int8_t)  + CBUF_BANK) ||
        conv_bo_ensure(fd, wt_bo,  wt_elems  * sizeof(int8_t)  + CBUF_BANK) ||
        conv_bo_ensure(fd, rc_bo,  256 * sizeof(uint64_t))                  ||
        conv_bo_ensure(fd, bs_bo,  (size_t)Cpad * sizeof(int32_t) + CBUF_BANK) ||
        conv_bo_ensure(fd, out_bo, out_elems * sizeof(int8_t)  + CBUF_BANK)) {
        ROCKET_LOGE("rocket_conv2d_dw_int8: BO alloc failed\n");
        goto out;
    }
    if (((in_bo->dma_address + in_bo->size) | (wt_bo->dma_address + wt_bo->size) |
         (out_bo->dma_address + out_bo->size) | (rc_bo->dma_address + rc_bo->size) |
         (bs_bo->dma_address + bs_bo->size)) >> 32) {
        ROCKET_LOGE("rocket_conv2d_dw_int8: a BO dma_address exceeds 32 bits\n");
        goto out;
    }

    /* input feature cube (C2=16), centered: (in_byte - 0x80). */
    rocket_bo_prep(fd, in_bo, 1, 0);
    memset(in_bo->ptr, 0, in_bo->size);
    {
        int8_t *dst = in_bo->ptr;       /* cube laid out for IHj rows; rows >= IH stay 0 */
        for (int c = 0; c < C; c++)
            for (int ih = 0; ih < IH; ih++)
                for (int iw = 0; iw < IW; iw++)
                    dst[feature_data(C, IHj, IW, 16, c + 1, ih + 1, iw + 1)] =
                        (int8_t)((uint8_t)in[((size_t)c * IH + ih) * IW + iw] - 0x80u);
    }
    rocket_bo_fini(fd, in_bo);

    /* DW weight cube (group G), centered: (w_byte - 0x80). */
    rocket_bo_prep(fd, wt_bo, 1, 0);
    memset(wt_bo->ptr, 0, wt_bo->size);
    {
        int8_t *dst = wt_bo->ptr;
        for (int c = 0; c < C; c++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    dst[weight_conv_dw_int8(C, KH, KW, G, c + 1, kh + 1, kw + 1)] =
                        (int8_t)((uint8_t)w[((size_t)c * KH + kh) * KW + kw] - 0x80u);
    }
    rocket_bo_fini(fd, wt_bo);

    /* per-OC int32 bias cube: bias_q - Σ_kernel(w_u8 - w_zp)*(in_zp - 0x80) (Mesa fold). */
    rocket_bo_prep(fd, bs_bo, 1, 0);
    memset(bs_bo->ptr, 0, bs_bo->size);
    {
        int32_t *dst = bs_bo->ptr;
        for (int c = 0; c < C; c++) {
            int32_t corr = 0;
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++) {
                    int w_u8 = (uint8_t)w[((size_t)c * KH + kh) * KW + kw];
                    corr += (w_u8 - w_zp) * (in_zp - 0x80);
                }
            dst[c] = (bias_q ? bias_q[c] : 0) - corr;
        }
    }
    rocket_bo_fini(fd, bs_bo);

    {
        conv_params_t p = {
            .ic = C, .ih = IHj, .iw = IW, .oc = C, .oh = OH, .ow = OW,
            .kh = KH, .kw = KW, .stride_y = sy, .stride_x = sx,
            .dil_y = dy, .dil_x = dx, .pad_top = pt, .pad_left = pl,
            .input_dma = (uint32_t)in_bo->dma_address,
            .weights_dma = (uint32_t)wt_bo->dma_address,
            .output_dma = (uint32_t)out_bo->dma_address,
            .tasks = regs, .dw_group = (uint8_t)G, .int8_out = 1,
            .in_scale = in_scale, .w_scale = w_scale, .out_scale = out_scale,
            .input_zero_point = in_zp, .output_zero_point = out_zp, .weight_zero_point = w_zp,
            .bias_dma = (uint32_t)bs_bo->dma_address,
        };
        if ((ret = gen_conv2d_dw_int8(&p)) != 0) {
            ROCKET_LOGE("rocket_conv2d_dw_int8: gen failed (%d)\n", ret);
            goto out;
        }
        if (p.task_count > sizeof(regs)/sizeof(regs[0])) {   /* unconditional: -DNDEBUG strips asserts */
            ROCKET_LOGE("rocket_conv2d_dw_int8: regcmd overflow (task_count %u > %zu words)\n",
                    p.task_count, sizeof(regs)/sizeof(regs[0]));
            ret = -1; goto out;
        }
        rocket_bo_prep(fd, rc_bo, 1, 0);
        memcpy(rc_bo->ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, rc_bo);

        rocket_bo_prep(fd, out_bo, 1, 0);
        memset(out_bo->ptr, 0, out_bo->size);
        rocket_bo_fini(fd, out_bo);

        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo->dma_address,
                                  .regcmd_count = p.task_count };
        uint32_t in_h[]  = { in_bo->handle, wt_bo->handle, bs_bo->handle, rc_bo->handle };
        uint32_t out_h[] = { out_bo->handle };
        ret = rocket_submit_tasks(fd, &task, 1, in_h, 4, out_h, 1);
        if (ret) { ROCKET_LOGE("rocket_conv2d_dw_int8: submit failed (%d)\n", ret); goto out; }
    }

    ret = rocket_bo_prep(fd, out_bo, 0, 2000000000ULL);
    if (ret) { ROCKET_LOGE("rocket_conv2d_dw_int8: wait timeout (%d)\n", ret); goto out; }
    {
        /* int8 output cube (C2=16), de-center back to the model domain (+0x80). */
        uint8_t *src = out_bo->ptr;
        for (int c = 0; c < C; c++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++)
                    out[((size_t)c * OH + oh) * OW + ow] =
                        (int8_t)(src[feature_data(C, OH, OW, 16, c + 1, oh + 1, ow + 1)] + 0x80u);
    }
    rocket_bo_fini(fd, out_bo);
    rc = 0;

out:
    if (!ctx) {
        rocket_bo_free(fd, out_bo);
        rocket_bo_free(fd, bs_bo);
        rocket_bo_free(fd, rc_bo);
        rocket_bo_free(fd, wt_bo);
        rocket_bo_free(fd, in_bo);
        rocket_bo_free(fd, guard);
    }
    return rc ? (ret ? ret : -1) : 0;
}

/* DW int8-out runtime: validate + channel-tile (chunks of G=64, each an independent job
 * — channels are independent). No spatial tiling yet (a single channel whose feature
 * overflows one CBUF pass returns <0; the delegate falls back to fp16 for that). */
static int conv2d_dw_int8_run(int fd, rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *w, const int32_t *bias_q,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out)
{
    if (!d->depthwise || d->oc != d->ic) return -2;
    const int C = d->ic, IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw;
    /* The int8-DW feature cube is C2=16 (feature_data(C,...,16,...)); a channel
     * count not a multiple of 16 desyncs the cube padding from the regcmd
     * channel-grain count -> wrong output. The fp16 DW path gets this gate via
     * rocket_conv2d_plan's ic%group check; this int8 path has no plan(), so reject
     * up front. (Cc stays a G-multiple and G=64%16==0, so C%16==0 also keeps every
     * channel chunk aligned. MobileDet DW is C%64==0, so this never fires today.) */
    if (C % 16 != 0) {
        ROCKET_LOGE("rocket_conv2d_dw_int8: channel count C=%d must be a multiple "
                "of 16 (C2 feature-cube grain)\n", C);
        return -5;
    }
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int sy = d->stride_y, sx = d->stride_x, pt = d->pad_top, pl = d->pad_left;
    const int dy = d->dil_y, dx = d->dil_x;
    const int G = DW_INT8_G;
    if (OH <= 0 || OW <= 0) return -3;
    /* one G-channel weight cube must fit a bank; one channel's feature must fit the budget
     * (no spatial tiling here — overflow => caller's fp16 fallback). */
    if ((size_t)KW * KH * G * sizeof(int8_t) > CBUF_BANK) return -4;

    int Cc = C;                                  /* channel chunk: largest G-multiple that fits */
    while (Cc > G && (size_t)Cc * IH * IW * sizeof(int8_t) > CONV_FEAT_BUDGET) Cc -= G;
    if ((size_t)Cc * IH * IW * sizeof(int8_t) > CONV_FEAT_BUDGET) return -4;   /* even G overflows */

    int ret = 0;
    for (int c0 = 0; c0 < C && !ret; c0 += Cc) {
        int Cn = (C - c0 < Cc) ? (C - c0) : Cc;
        ret = conv2d_dw_int8_one_job(fd, ctx, Cn, IH, IW, OH, OW, KH, KW, sy, sx, pt, pl, dy, dx,
                                     in + (size_t)c0 * IH * IW, w + (size_t)c0 * KH * KW,
                                     bias_q ? bias_q + c0 : NULL,
                                     in_scale, w_scale, out_scale, in_zp, w_zp, out_zp,
                                     out + (size_t)c0 * OH * OW);
    }
    return ret;
}

int rocket_conv2d_dw_int8(int fd, const rocket_conv2d_desc *d,
                          const int8_t *in, const int8_t *w, const int32_t *bias,
                          float in_scale, float w_scale, float out_scale,
                          int in_zp, int w_zp, int out_zp, int8_t *out)
{
    return conv2d_dw_int8_run(fd, NULL, d, in, w, bias, in_scale, w_scale, out_scale,
                              in_zp, w_zp, out_zp, out);
}

int rocket_conv2d_dw_int8_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *w, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out)
{
    if (!ctx) return -1;
    return conv2d_dw_int8_run(ctx->fd, ctx, d, in, w, bias, in_scale, w_scale, out_scale,
                              in_zp, w_zp, out_zp, out);
}
