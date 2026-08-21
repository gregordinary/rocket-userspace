// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_pool_rk3576.c — POOLING for the RK3576, behind a library entry.
 *
 * gen_pool_rk3576() emits the PPU's own 31-write program and computes bit-exactly, and
 * for as long as it was reachable only from tests/rk3576_pool_probe.c a caller with a
 * row-major tensor could not use it: the NC1HWC2 cube, the sentinel, the submit and the
 * de-scatter all lived in the probe. This file is that layer, and it is the same shape
 * as rocket_conv2d_rk3576.c's — one entry, row-major in and out, the part's submit
 * discipline owned here rather than by the caller.
 *
 * WHAT THE PART COMPUTES, and so what this entry states:
 *
 *   int8 -> int8, a window reduction with NO requant in the path. The PPU is not a
 *   convolution epilogue on this part: it is 23 PPU writes and 8 PPU_RDMA, no CNA, no
 *   CORE, no DPU, reading and writing the same 16-byte-atom cube the convolution path
 *   already packs.
 *
 *   THE AVERAGE ROUNDS HALF TO EVEN, which is where this diverges from the RK3588's
 *   rocket_pool_int8(): that one truncates toward zero. Two different roundings are two
 *   different functions, so the RK3588 entry refuses here and names this one rather than
 *   dispatching — the same answer rocket_conv2d_int8() and rocket_matmul_int8() give.
 *
 *   THE DIVISOR IS THE WINDOW, NOT THE TAP COUNT. The PPU has no divider; it multiplies
 *   the window sum by a per-axis Q16 reciprocal, 0x10000/kw and 0x10000/kh. So
 *   POOL_METHOD_AVG divides by kh*kw whatever the padding excluded — TFLite's
 *   count-include-pad = TRUE — and ROCKET_RK3576_POOL_AVG_NOPAD is the mode bit that
 *   drops the pad taps instead. THAT RECIPROCAL IS TRUNCATED, and the error it leaves
 *   grows with the window sum: at k2/k3/k5 it is far under half a count over any int8
 *   window and the result is the exact rounded average, and at larger windows it is not
 *   guaranteed to be. rocket_pool_int8_rk3576_exact() answers that question for a
 *   descriptor without running it, from the reciprocal the emitter will program and the
 *   worst-case int8 window sum.
 *
 * THE PAD VALUE IS THE INPUT ZERO POINT on the average path, which is why this entry
 * takes one where the RK3588's does not: a quantized average pool whose input zero point
 * is not zero pads with that value, and the emitter carries the field.
 *
 * A HANDLE IS THE SHAPE, NOT AN OPTIMIZATION OF IT. Everything below runs through
 * rocket_pool_int8_pack_rk3576(): the transient entry packs, runs once and frees, so
 * there is one arithmetic and one submit discipline rather than two that have to be kept
 * in agreement. What a handle holds is what does not depend on the input — the three BOs,
 * the register program, the plan — and what it saves on a graph is the BO churn, which is
 * what the convolution path's residency lever turned out to be.
 *
 * ROCKET_RK3576_POOL_PROF=1 logs one line per call, at ROCKET_LOG_INFO.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_pool.h"
#include "rocket_conv.h"     /* rocket_rk3576_cube */
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"

#define C2 16u      /* the int8 feature/output channel atom */

static unsigned r76p_round4(unsigned v) { return (v + 3u) & ~3u; }

/* What a handle freezes and why each is not bookkeeping — the same three the convolution
 * path's handle does. The fd, because a BO belongs to its file and an IOVA is per-fd. The
 * geometry, because the register program is generated from it once. The input zero point,
 * because the average path's pad value is folded into that program.
 *
 * `src` is a BORROWED cube: when cube_in is set the handle reads a producer's output
 * surface directly and its own feature cube is dropped, so an inference does no scatter
 * and no cache maintenance on the input side at all. */
struct rocket_pool_int8_rk3576_handle {
    int              fd;
    rocket_pool_desc d;
    int              in_zp;
    unsigned         c, ih, iw, oh, ow, creg;
    unsigned         in_surf, out_surf;      /* elements per channel group */
    size_t           src_off;                /* the input cube's byte offset in `src` */
    size_t           in_bytes, out_bytes;
    rocket_bo        in, out, rc;            /* `in` unallocated when cube_in */
    rocket_bo        src;                    /* borrowed; valid when cube_in */
    int              cube_in;
    /* The output surface is left where the PPU wrote it and a consumer reads it as its
     * feature cube, so there is no de-scatter and `out` may be NULL. `out_ext` is a
     * caller's buffer this handle writes its own slice of — borrowed, never freed here,
     * and while it is in use this handle's own `out` is released. */
    int              cube_out;
    rocket_bo        out_ext;
    size_t           out_off;
    uint32_t         task_count;
};

struct r76p_prof {
    int      on;
    double   pack_us, scatter_us, stamp_us, submit_us, read_us, descat_us, free_us;
    unsigned attempts;
};

static int r76p_prof_on(void)
{
    const char *e = getenv("ROCKET_RK3576_POOL_PROF");
    return (e && *e && *e != '0');
}

static double r76p_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

#define R76P_T(p)          ((p).on ? r76p_now_us() : 0.0)
#define R76P_ACC(p, f, t0) do { if ((p).on) (p).f += r76p_now_us() - (t0); } while (0)

/* How many times a task that wrote nothing is redone. The guard a wide-output job leaves
 * behind is probabilistic — a confirmed power-domain collapse clears it about 87% of the
 * time — so the redo count is the lever. Eight matches the conv and matmul paths. */
static unsigned r76p_attempts(void)
{
    const char *e = getenv("ROCKET_RK3576_TASK_ATTEMPTS");
    long v = e && *e ? strtol(e, NULL, 0) : 8;
    if (v < 1) v = 1;
    if (v > 64) v = 64;
    return (unsigned)v;
}

/* Whether a handle that owns its own feature cube packs it at the plane's own element
 * count rather than the round4 the vendor's pooling programs carry. It exists to make
 * the alignment a GATED question rather than an assumption: a cube-in join hands
 * PPU_RDMA a producer's surface stride verbatim, and a direct convolution's is `ow*oh`
 * exactly, so whether the PPU takes a stride that is not a multiple of four decides
 * whether the join is available at all. ROCKET_RK3576_POOL_PACK_SRC=1. */
static int r76p_pack_src(void)
{
    const char *e = getenv("ROCKET_RK3576_POOL_PACK_SRC");
    return e && *e && *e != '0';
}

/* WHERE THIS HANDLE'S OUTPUT SURFACE IS: its own BO, or the slice of a caller's buffer it
 * was placed in. The stamp, the write check and the de-scatter all address it through
 * here, so a placed handle never touches the bytes outside its own slice — the rest of
 * that buffer belongs to another producer and has already been written this inference. */
static rocket_bo *r76p_surf(struct rocket_pool_int8_rk3576_handle *h, size_t *off)
{
    if (h->out_ext.ptr) { *off = h->out_off; return &h->out_ext; }
    *off = 0;
    return &h->out;
}

static int r76p_is_this_chip(const char *entry)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (hw == &rocket_hw_rk3576) return 1;
    ROCKET_LOGE("%s emits the RK3576 PPU program and the active profile is %s\n",
                entry, hw && hw->name ? hw->name : "unknown");
    return 0;
}

int rocket_pool_int8_rk3576_plan(const rocket_pool_desc *d)
{
    if (!d) return ROCKET_E_SHAPE;
    if (d->c <= 0 || d->ih <= 0 || d->iw <= 0 || d->kh <= 0 || d->kw <= 0 ||
        d->stride_y <= 0 || d->stride_x <= 0)
        return ROCKET_E_SHAPE;
    if (d->pad_top < 0 || d->pad_left < 0 || d->pad_bottom < 0 || d->pad_right < 0)
        return ROCKET_E_SHAPE;
    if (d->method != POOL_METHOD_MAX && d->method != POOL_METHOD_AVG)
        return ROCKET_E_SHAPE;
    /* The kernel and stride fields are four bits each; a larger window cascades. */
    if (d->kh > 16 || d->kw > 16 || d->stride_y > 16 || d->stride_x > 16)
        return ROCKET_E_UNSUPPORTED;
    if (d->pad_top > 255 || d->pad_left > 255 ||
        d->pad_bottom > 255 || d->pad_right > 255)
        return ROCKET_E_UNSUPPORTED;
    if (rocket_pool_oh(d) <= 0 || rocket_pool_ow(d) <= 0)
        return ROCKET_E_SHAPE;
    return ROCKET_OK;
}

/*
 * Whether the programmed reciprocal can reproduce the exactly-rounded average for this
 * window, over EVERY int8 input. The PPU multiplies the window sum by 0x10000/kw and
 * 0x10000/kh, both truncated, so the computed quotient is low by
 *
 *     |sum| * (1/(kh*kw) - (floor(2^16/kw)/2^16)*(floor(2^16/kh)/2^16))
 *
 * and the largest |sum| an int8 window can reach is 128*kh*kw. The rounded average moves
 * when that error can cross a half-count boundary; the closest a quotient of an integer
 * by n can come to a half without being one is 1/(2n), so the test is against that.
 * Exact for max, which divides nothing.
 */
int rocket_pool_int8_rk3576_exact(const rocket_pool_desc *d)
{
    unsigned n;
    double rw, rh, err;
    if (rocket_pool_int8_rk3576_plan(d) != ROCKET_OK) return 0;
    if (d->method == POOL_METHOD_MAX) return 1;
    n = (unsigned)d->kh * (unsigned)d->kw;
    rw = (double)(0x10000u / (unsigned)d->kw) / 65536.0;
    rh = (double)(0x10000u / (unsigned)d->kh) / 65536.0;
    err = (1.0 / (double)n - rw * rh) * (128.0 * (double)n);
    if (err < 0.0) err = -err;
    return err < 1.0 / (2.0 * (double)n);
}

/* Build the register program into `ops`, returning its word count in `*nops`. Split out of
 * r76p_gen so that a cross-layer chain, which lays every program of a run out contiguously
 * in one BO of its own, gets the same words from the same code rather than a second
 * transcription of the geometry. */
static int r76p_build(const struct rocket_pool_int8_rk3576_handle *h, const char *entry,
                      uint64_t *ops, uint32_t *nops)
{
    pool_params_rk3576_t p;

    memset(&p, 0, sizeof p);
    memset(ops, 0, RK3576_POOL_TASK_OPS * sizeof *ops);
    p.iw = (uint16_t)h->iw; p.ih = (uint16_t)h->ih; p.c = (uint16_t)h->c;
    p.ow = (uint16_t)h->ow; p.oh = (uint16_t)h->oh;
    p.kw = (uint8_t)h->d.kw; p.kh = (uint8_t)h->d.kh;
    p.stride_x = (uint8_t)h->d.stride_x; p.stride_y = (uint8_t)h->d.stride_y;
    p.pad_left   = (uint8_t)h->d.pad_left;   p.pad_right  = (uint8_t)h->d.pad_right;
    p.pad_top    = (uint8_t)h->d.pad_top;    p.pad_bottom = (uint8_t)h->d.pad_bottom;
    p.mode = (uint8_t)(h->d.method == POOL_METHOD_MAX ? ROCKET_RK3576_POOL_MAX
                                                      : ROCKET_RK3576_POOL_AVG);
    p.input_zero_point = h->in_zp;
    p.input_dma  = h->cube_in ? h->src.dma_address + h->src_off : h->in.dma_address;
    p.output_dma = (uint32_t)((h->out_ext.ptr ? h->out_ext.dma_address + h->out_off
                                             : h->out.dma_address));
    p.src_surf_elems = h->in_surf;
    p.tasks = ops;

    if (gen_pool_rk3576(&p) != 0) {
        ROCKET_LOGE("%s: the generator refused (%ux%u c%u k%ux%u s%ux%u -> %ux%u)\n",
                    entry, h->iw, h->ih, h->c, h->d.kw, h->d.kh,
                    h->d.stride_x, h->d.stride_y, h->ow, h->oh);
        return ROCKET_E_UNSUPPORTED;
    }
    *nops = p.task_count;
    return ROCKET_OK;
}

/* (Re)generate the register program into the handle's regcmd BO. Called at pack time and
 * again whenever the input BO changes, which is the only thing in the program that a
 * caller can move after packing. */
static int r76p_gen(struct rocket_pool_int8_rk3576_handle *h, const char *entry)
{
    uint64_t ops[RK3576_POOL_TASK_OPS];
    uint32_t nops = 0;
    int rc = r76p_build(h, entry, ops, &nops);

    if (rc != ROCKET_OK) return rc;
    h->task_count = nops;
    rocket_bo_prep(h->fd, &h->rc, 1, 0);
    memcpy(h->rc.ptr, ops, (size_t)nops * sizeof(uint64_t));
    rocket_bo_fini(h->fd, &h->rc);
    return ROCKET_OK;
}

/* THE CHAIN'S VIEW OF THIS HANDLE. See rocket_rk3576_internal.h for why a pool may sit in
 * a convolution stream and why only as an INTERIOR node. Everything here is read off the
 * frozen handle; nothing on it is written, and the program goes into the caller's array
 * rather than the handle's own regcmd BO, whose trailer the chain would otherwise rewrite
 * under the per-layer entry that shares it. */
int rocket_rk3576_pool_link(struct rocket_pool_int8_rk3576_handle *h,
                            struct rocket_rk3576_pool_link *out, uint64_t *ops)
{
    static const char entry[] = "rocket_rk3576_pool_link";
    uint32_t nops = 0;
    int rc;

    if (!h || !out || !ops) return ROCKET_E_SHAPE;
    /* Somewhere to write, which is its OWN surface or a caller's buffer it writes a slice
     * of — a placed handle releases `out`, so asking for that one alone refuses every
     * pool whose output was placed. */
    if (!h->rc.ptr || (!h->out.ptr && !h->out_ext.ptr)) {
        ROCKET_LOGE("%s: this handle was never packed\n", entry);
        return ROCKET_E_SHAPE;
    }
    rc = r76p_build(h, entry, ops, &nops);
    if (rc != ROCKET_OK) return rc;

    memset(out, 0, sizeof *out);
    out->fd         = h->fd;
    out->feat_dma   = h->cube_in ? h->src.dma_address + h->src_off : h->in.dma_address;
    out->surf_dma   = h->out_ext.ptr ? h->out_ext.dma_address + h->out_off
                                     : h->out.dma_address;
    out->cube_in    = h->cube_in;
    out->cube_out   = h->cube_out;
    out->surf       = h->out_ext.ptr ? &h->out_ext : &h->out;
    out->surf_off   = h->out_ext.ptr ? h->out_off : 0;
    out->groups     = h->creg / C2;
    out->surf_elems = h->out_surf;
    /* The PPU writes ow*oh elements per group into a stride of round4(ow*oh), so the
     * write guard must ask about the live prefix and not the padding — which the PPU
     * never touches and which would therefore read as "this task wrote nothing". */
    out->live_elems = h->ow * h->oh;
    out->nops       = nops;
    return ROCKET_OK;
}

rocket_pool_int8_rk3576_handle *
rocket_pool_int8_pack_rk3576(int fd, const rocket_pool_desc *d, int in_zp)
{
    static const char entry[] = "rocket_pool_int8_pack_rk3576";
    struct rocket_pool_int8_rk3576_handle *h;

    if (!d || fd < 0) return NULL;
    if (!r76p_is_this_chip(entry)) return NULL;
    if (rocket_pool_int8_rk3576_plan(d) != ROCKET_OK) return NULL;
    if (in_zp < -128 || in_zp > 127) return NULL;

    h = (struct rocket_pool_int8_rk3576_handle *)calloc(1, sizeof *h);
    if (!h) return NULL;
    h->fd = fd;
    h->d = *d;
    h->in_zp = in_zp;
    h->c  = (unsigned)d->c; h->ih = (unsigned)d->ih; h->iw = (unsigned)d->iw;
    h->oh = (unsigned)rocket_pool_oh(d);
    h->ow = (unsigned)rocket_pool_ow(d);
    h->creg = ((h->c + C2 - 1u) / C2) * C2;
    h->in_surf  = r76p_pack_src() ? h->iw * h->ih : r76p_round4(h->iw * h->ih);
    h->out_surf = r76p_round4(h->ow * h->oh);
    h->in_bytes  = (size_t)(h->creg / C2) * h->in_surf * C2;
    h->out_bytes = (size_t)(h->creg / C2) * h->out_surf * C2;

    if (rocket_bo_alloc(fd, h->in_bytes, &h->in) < 0 ||
        rocket_bo_alloc(fd, h->out_bytes, &h->out) < 0 ||
        rocket_bo_alloc(fd, RK3576_POOL_TASK_OPS * sizeof(uint64_t), &h->rc) < 0) {
        ROCKET_LOGE("%s: BO allocation failed\n", entry);
        rocket_pool_int8_free_rk3576(fd, h);
        return NULL;
    }
    /* The whole cube is zeroed once here rather than per call. Pooling reduces WITHIN a
     * channel, so what the padding channels of the last group hold never reaches a live
     * channel's output and is never de-scattered — but the PPU reads them, so they have
     * to hold something defined. */
    rocket_bo_prep(fd, &h->in, 1, 0);
    memset(h->in.ptr, 0, h->in_bytes);
    rocket_bo_fini(fd, &h->in);

    if (r76p_gen(h, entry) != ROCKET_OK) {
        rocket_pool_int8_free_rk3576(fd, h);
        return NULL;
    }
    return h;
}

int rocket_pool_int8_cube_in_rk3576(rocket_pool_int8_rk3576_handle *h,
                                    const rocket_rk3576_cube *src)
{
    static const char entry[] = "rocket_pool_int8_cube_in_rk3576";

    if (!h) return ROCKET_E_SHAPE;
    if (!src) {
        if (!h->cube_in) return ROCKET_OK;
        if (rocket_bo_alloc(h->fd, h->in_bytes, &h->in) < 0) return ROCKET_E_NOMEM;
        rocket_bo_prep(h->fd, &h->in, 1, 0);
        memset(h->in.ptr, 0, h->in_bytes);
        rocket_bo_fini(h->fd, &h->in);
        h->cube_in = 0;
        h->src_off = 0;
        memset(&h->src, 0, sizeof h->src);
        h->in_surf = r76p_pack_src() ? h->iw * h->ih : r76p_round4(h->iw * h->ih);
        return r76p_gen(h, entry);
    }
    if (src->fd != h->fd) {
        ROCKET_LOGE("%s: the cube belongs to fd %d and this handle to fd %d; an IOVA is "
                    "per-fd\n", entry, src->fd, h->fd);
        return ROCKET_E_SHAPE;
    }
    if (src->c != h->c || src->h != h->ih || src->w != h->iw) {
        ROCKET_LOGE("%s: the cube is %ux%ux%u and this handle reads %ux%ux%u\n",
                    entry, src->c, src->h, src->w, h->c, h->ih, h->iw);
        return ROCKET_E_SHAPE;
    }
    /* The channel count has to round the same way at both ends. Below a multiple of 16
     * the handle's own cube carries padding channels it fills itself and a producer does
     * not, and the PPU reads the whole programmed group — so the result of a live channel
     * is unaffected but the padding is a producer's business, and there is no producer
     * this library builds whose surface would not simply be a multiple of 16. */
    if (h->c % C2) {
        ROCKET_LOGE("%s: %u channels is not a multiple of %u, so this handle's own cube "
                    "carries padding channels a producer does not write\n",
                    entry, h->c, C2);
        return ROCKET_E_UNSUPPORTED;
    }
    if (src->groups * C2 < h->creg ||
        src->bo.size < src->off + (size_t)src->groups * src->surf_elems * C2) {
        ROCKET_LOGE("%s: the cube carries %u channel group(s) of %zu bytes and this "
                    "handle's PPU_RDMA walks %u channels\n",
                    entry, src->groups, src->surf_elems * C2, h->creg);
        return ROCKET_E_SHAPE;
    }
    if (src->surf_elems > 0xFFFFFFFFu / C2) return ROCKET_E_SHAPE;

    /* A cube may be a SLICE of a bigger buffer, and the base is a plain address the PPU
     * honours like every other. */
    if (src->off % ((size_t)src->surf_elems * C2)) {
        ROCKET_LOGE("%s: the cube starts at byte %zu, which is not a whole channel group "
                    "of %zu bytes into its buffer\n",
                    entry, src->off, (size_t)src->surf_elems * C2);
        return ROCKET_E_SHAPE;
    }
    h->src = src->bo;
    h->src_off = src->off;
    h->cube_in = 1;
    /* The producer's surface stride is what PPU_RDMA has to walk, and a direct
     * convolution's is `ow*oh` exactly rather than the round4 this handle would allocate
     * for itself. The register takes it verbatim. */
    h->in_surf = (unsigned)src->surf_elems;
    if (h->in.ptr) rocket_bo_free(h->fd, &h->in);
    memset(&h->in, 0, sizeof h->in);
    return r76p_gen(h, entry);
}

int rocket_pool_int8_cube_out_rk3576(rocket_pool_int8_rk3576_handle *h, int on)
{
    static const char entry[] = "rocket_pool_int8_cube_out_rk3576";

    if (!h) return ROCKET_E_SHAPE;
    if (!on) { h->cube_out = 0; return ROCKET_OK; }
    /* The same rounding rule the input side has, for the same reason: pooling reduces
     * WITHIN a channel, so a partial group's channels carry whatever the input cube's
     * padding held and this handle cannot say what a consumer would read there. */
    if (h->c % C2) {
        ROCKET_LOGE("%s: %u channels is not a multiple of %u, so the last group's tail is "
                    "the input cube's padding and not a value this handle can declare\n",
                    entry, h->c, C2);
        return ROCKET_E_UNSUPPORTED;
    }
    h->cube_out = 1;
    return ROCKET_OK;
}

int rocket_pool_int8_cube_out_at_rk3576(rocket_pool_int8_rk3576_handle *h,
                                        const struct rocket_rk3576_cube *dst)
{
    static const char entry[] = "rocket_pool_int8_cube_out_at_rk3576";
    unsigned need;
    size_t bytes;

    if (!h) return ROCKET_E_SHAPE;
    if (!dst) {
        if (!h->out_ext.ptr) { h->cube_out = 0; return ROCKET_OK; }
        if (rocket_bo_alloc(h->fd, h->out_bytes, &h->out) < 0) return ROCKET_E_NOMEM;
        memset(&h->out_ext, 0, sizeof h->out_ext);
        h->out_off = 0;
        h->cube_out = 0;
        return r76p_gen(h, entry);
    }
    if (rocket_pool_int8_cube_out_rk3576(h, 1) != ROCKET_OK) return ROCKET_E_UNSUPPORTED;
    if (dst->fd != h->fd) {
        ROCKET_LOGE("%s: the buffer belongs to fd %d and this handle to fd %d; an IOVA is "
                    "per-fd\n", entry, dst->fd, h->fd);
        return ROCKET_E_SHAPE;
    }
    if (dst->h != h->oh || dst->w != h->ow) {
        ROCKET_LOGE("%s: the slice is %ux%u and this handle writes %ux%u\n",
                    entry, dst->w, dst->h, h->ow, h->oh);
        return ROCKET_E_SHAPE;
    }
    /* THE DESTINATION SURFACE STRIDE IS NOT A PARAMETER OF THE PROGRAM. The emitter
     * derives it from the output plane as round4(ow*oh) — the shape every vendor pooling
     * program carries — and there is no decoded register to move it, so a slice at any
     * other stride is refused rather than written at the wrong offsets. A plane whose
     * element count is already a multiple of four (56x56 is) needs nothing. */
    if (dst->surf_elems != (size_t)h->out_surf) {
        ROCKET_LOGE("%s: the slice's channel-group stride is %zu elements and this handle's "
                    "program writes round4(%u*%u) = %u; the destination stride is not a "
                    "field of the pooling program\n",
                    entry, dst->surf_elems, h->ow, h->oh, h->out_surf);
        return ROCKET_E_UNSUPPORTED;
    }
    need = h->creg / C2;
    bytes = (size_t)need * h->out_surf * C2;
    if (dst->groups < need || dst->bo.size < dst->off + bytes) {
        ROCKET_LOGE("%s: this handle writes %u channel group(s) (%zu bytes) and the slice "
                    "carries %u at byte %zu of a %zu-byte buffer\n",
                    entry, need, bytes, dst->groups, dst->off, dst->bo.size);
        return ROCKET_E_SHAPE;
    }
    if (dst->off % ((size_t)h->out_surf * C2)) {
        ROCKET_LOGE("%s: the slice starts at byte %zu, which is not a whole channel group "
                    "of %zu bytes into its buffer\n",
                    entry, dst->off, (size_t)h->out_surf * C2);
        return ROCKET_E_SHAPE;
    }
    h->out_ext = dst->bo;
    h->out_off = dst->off;
    /* The handle's own surface is dead weight now — the point of the shared buffer is to
     * pay for one allocation and not two. */
    if (h->out.ptr) rocket_bo_free(h->fd, &h->out);
    memset(&h->out, 0, sizeof h->out);
    return r76p_gen(h, entry);
}

int rocket_pool_int8_cube_of_rk3576(const rocket_pool_int8_rk3576_handle *h,
                                    struct rocket_rk3576_cube *out)
{
    static const char entry[] = "rocket_pool_int8_cube_of_rk3576";

    if (!h || !out) return ROCKET_E_SHAPE;
    if (h->c % C2) {
        ROCKET_LOGE("%s: %u channels is not a multiple of %u\n", entry, h->c, C2);
        return ROCKET_E_UNSUPPORTED;
    }
    memset(out, 0, sizeof *out);
    out->fd = h->fd;
    out->c = h->c;
    out->h = h->oh;
    out->w = h->ow;
    out->groups = h->creg / C2;
    /* The PPU writes round4(ow*oh) per channel group where a direct convolution writes the
     * plane exactly. That is the consumer's DDR channel-group jump, which is a register and
     * is honoured at any value at or above the plane. */
    out->surf_elems = h->out_surf;
    if (h->out_ext.ptr) { out->bo = h->out_ext; out->off = h->out_off; }
    else                { out->bo = h->out;     out->off = 0; }
    return ROCKET_OK;
}

int rocket_pool_int8_prepacked_rk3576(int fd, rocket_pool_int8_rk3576_handle *h,
                                      const int8_t *in, int8_t *out)
{
    static const char entry[] = "rocket_pool_int8_prepacked_rk3576";
    uint32_t in_h[2], out_h[1];
    rocket_bo *osurf;
    size_t ooff;
    unsigned attempt, attempts, ci, y, x;
    unsigned char stamp;
    struct r76p_prof prof;
    double t0;
    int rc, cycled = 0, confirmed = 0;

    if (!h) return ROCKET_E_SHAPE;
    if (!out && !h->cube_out) return ROCKET_E_SHAPE;
    if (!in && !h->cube_in) return ROCKET_E_SHAPE;
    if (fd != h->fd) {
        ROCKET_LOGE("%s: this handle's buffers belong to fd %d, not %d — an IOVA is "
                    "per-fd, so a foreign fd would submit successfully against addresses "
                    "that mean nothing there\n", entry, h->fd, fd);
        return ROCKET_E_SHAPE;
    }

    memset(&prof, 0, sizeof prof);
    prof.on = r76p_prof_on();

    if (!h->cube_in) {
        /* The feature cube. The channel atom is the innermost axis, so a row-major CHW
         * source lands strided however this is written; the loop is ordered to keep the
         * source sequential and the destination's 16-byte stride predictable. */
        t0 = R76P_T(prof);
        rocket_bo_prep(fd, &h->in, 1, 0);
        {
            int8_t *cube = (int8_t *)h->in.ptr;
            for (ci = 0; ci < h->c; ci++) {
                const int8_t *src = in + (size_t)ci * h->ih * h->iw;
                int8_t *dst = cube + (size_t)(ci / C2) * h->in_surf * C2 + (ci % C2);
                for (y = 0; y < h->ih; y++)
                    for (x = 0; x < h->iw; x++)
                        dst[(size_t)C2 * (y * h->iw + x)] = *src++;
            }
        }
        rocket_bo_fini(fd, &h->in);
        R76P_ACC(prof, scatter_us, t0);
    }

    osurf = r76p_surf(h, &ooff);
    in_h[0] = h->cube_in ? h->src.handle : h->in.handle;
    in_h[1] = h->rc.handle;
    out_h[0] = osurf->handle;

    /* One PPU program, so the write check is over the whole surface. The stamp is what
     * makes "never written" a fact rather than a guess — a fresh BO's zeros cannot tell
     * an unwritten atom from a legitimately zero one — and it is BRACKETED, because a
     * bare memset leaves dirty lines that race the PPU's write DMA. */
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;
    attempts = r76p_attempts();
    rc = ROCKET_E_DEVICE;
    for (attempt = 0; attempt < attempts; attempt++) {
        size_t i;
        int wrote;

        prof.attempts++;
        if (stamp) {
            t0 = R76P_T(prof);
            rocket_bo_prep(fd, osurf, 1, 0);
            memset((char *)osurf->ptr + ooff, stamp, h->out_bytes);
            rocket_bo_fini(fd, osurf);
            R76P_ACC(prof, stamp_us, t0);
        }
        t0 = R76P_T(prof);
        /* THE COMPLETION IS THE PPU'S, NOT THE DPU'S. A pool program is 23 PPU writes
         * and 8 PPU_RDMA with no DPU stage at all, so the DPU bits the driver waits on
         * by default can never set and the job retires on the grace period in full —
         * measured at 640 us of a 1.48 ms call, and tracking dpu_grace_us count for
         * count. Naming the class makes it the PPU's own completion instead.
         *
         * GATED on the kernel, not sent blind: the submit ioctl REJECTS a flag word it
         * does not recognise, so an older kernel would fail the submit rather than fall
         * back to the grace period. [HW sweep, H96 MAX M9] */
        if (rocket_submit_matmul_flags(fd, &h->rc, h->task_count, in_h, 2, out_h, 1,
                                       rocket_ppu_done_supported()
                                           ? ROCKET_JOB_PPU_DONE : 0u) != 0) {
            ROCKET_LOGE("%s: submit failed\n", entry);
            return ROCKET_E_DEVICE;
        }
        R76P_ACC(prof, submit_us, t0);
        t0 = R76P_T(prof);
        if (rocket_bo_prep(fd, osurf, 0, 2000000000ull) < 0) {
            ROCKET_LOGE("%s: PREP_BO on the output timed out\n", entry);
            return ROCKET_E_DEVICE;
        }
        if (!stamp) { R76P_ACC(prof, read_us, t0); rc = ROCKET_OK; break; }
        wrote = 0;
        for (i = 0; i < h->out_bytes; i++)
            if (((const unsigned char *)osurf->ptr)[ooff + i] != stamp) { wrote = 1; break; }
        rocket_bo_fini(fd, osurf);
        R76P_ACC(prof, read_us, t0);
        if (wrote) { rc = ROCKET_OK; break; }
        ROCKET_LOGD("%s: the program wrote nothing on attempt %u; cycling the power "
                    "domain and redoing it\n", entry, attempt + 1u);
        cycled++;
        confirmed += rocket_rk3576_power_idle();
    }
    if (rc != ROCKET_OK) {
        ROCKET_LOGE("%s: the program wrote nothing over %u attempts (%d power cycles, "
                    "%d of them confirmed to reach suspended)\n",
                    entry, attempts, cycled, confirmed);
        return rc;
    }

    /* CUBE OUT: the surface stays where the PPU wrote it and the next layer's CNA reads
     * it, so there is nothing to de-scatter. The fence was already waited on above. */
    t0 = R76P_T(prof);
    if (!h->cube_out) {
        const int8_t *o = (const int8_t *)osurf->ptr + ooff;
        for (ci = 0; ci < h->c; ci++) {
            const int8_t *src = o + (size_t)(ci / C2) * h->out_surf * C2 + (ci % C2);
            int8_t *dst = out + (size_t)ci * h->oh * h->ow;
            for (y = 0; y < h->oh; y++)
                for (x = 0; x < h->ow; x++)
                    *dst++ = src[(size_t)C2 * (y * h->ow + x)];
        }
    }
    R76P_ACC(prof, descat_us, t0);

    if (prof.on)
        ROCKET_LOGI("pool %ux%ux%u k%ux%u -> %ux%u: scatter %.0f stamp %.0f submit %.0f "
                    "read %.0f de-scatter %.0f us (%u attempt%s%s)\n",
                    h->c, h->ih, h->iw, h->d.kh, h->d.kw, h->oh, h->ow,
                    prof.scatter_us, prof.stamp_us, prof.submit_us, prof.read_us,
                    prof.descat_us, prof.attempts, prof.attempts == 1u ? "" : "s",
                    h->cube_in ? ", cube in" : "");
    return ROCKET_OK;
}

void rocket_pool_int8_free_rk3576(int fd, rocket_pool_int8_rk3576_handle *h)
{
    if (!h) return;
    if (h->rc.ptr)  rocket_bo_free(fd, &h->rc);
    if (h->out.ptr) rocket_bo_free(fd, &h->out);
    if (h->in.ptr)  rocket_bo_free(fd, &h->in);
    free(h);
}

/*
 * The transient entry: pack, run once, free. It is deliberately not a second
 * implementation — a caller that runs this shape once pays the BO churn a handle exists
 * to amortize, and a caller that runs it repeatedly holds the handle instead.
 */
int rocket_pool_int8_rk3576(int fd, const rocket_pool_desc *d, int in_zp,
                            const int8_t *in, int8_t *out)
{
    static const char entry[] = "rocket_pool_int8_rk3576";
    struct rocket_pool_int8_rk3576_handle *h;
    struct r76p_prof prof;
    double t0;
    int rc;

    if (!d || !in || !out) return ROCKET_E_SHAPE;
    if (!r76p_is_this_chip(entry)) return ROCKET_E_UNSUPPORTED;
    rc = rocket_pool_int8_rk3576_plan(d);
    if (rc != ROCKET_OK) return rc;
    if (in_zp < -128 || in_zp > 127) return ROCKET_E_SHAPE;
    if (fd < 0) return ROCKET_E_SHAPE;

    memset(&prof, 0, sizeof prof);
    prof.on = r76p_prof_on();

    t0 = R76P_T(prof);
    h = rocket_pool_int8_pack_rk3576(fd, d, in_zp);
    R76P_ACC(prof, pack_us, t0);
    if (!h) return ROCKET_E_UNSUPPORTED;

    rc = rocket_pool_int8_prepacked_rk3576(fd, h, in, out);

    t0 = R76P_T(prof);
    rocket_pool_int8_free_rk3576(fd, h);
    R76P_ACC(prof, free_us, t0);

    if (prof.on)
        ROCKET_LOGI("pool transient: pack %.0f (3 BOs + the program) free %.0f us\n",
                    prof.pack_us, prof.free_us);
    return rc;
}

/*
 * The CPU model of the above: signed integer max, or an average over the WINDOW that
 * rounds half to even. Against round-half-away-from-zero a k2 average disagrees on one
 * output in eight, so this is the rule and not a detail. An odd window has no exact
 * half and so no tie to break.
 */
void rocket_pool_ref_int8_rk3576(const rocket_pool_desc *d, int in_zp,
                                 const int8_t *in, int8_t *out)
{
    int oh = rocket_pool_oh(d), ow = rocket_pool_ow(d);
    int c, y, x, kh, kw;

    for (c = 0; c < d->c; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                long best = -1000, sum = 0;
                long n = (long)d->kh * d->kw;
                for (kh = 0; kh < d->kh; kh++)
                    for (kw = 0; kw < d->kw; kw++) {
                        int iy = y * d->stride_y + kh - d->pad_top;
                        int ix = x * d->stride_x + kw - d->pad_left;
                        int v;
                        if (iy < 0 || ix < 0 || iy >= d->ih || ix >= d->iw)
                            v = (d->method == POOL_METHOD_MAX) ? -128 : in_zp;
                        else
                            v = in[((size_t)c * d->ih + iy) * d->iw + ix];
                        if (v > best) best = v;
                        sum += v;
                    }
                if (d->method == POOL_METHOD_MAX) {
                    out[((size_t)c * oh + y) * ow + x] = (int8_t)best;
                } else {
                    long half = n / 2, q, r;
                    q = sum >= 0 ? (sum + half) / n : -(((-sum) + half) / n);
                    if ((n & 1) == 0) {
                        r = sum - q * n;
                        if ((r == half || r == -half) && (q & 1))
                            q += (sum >= 0) ? -1 : 1;
                    }
                    out[((size_t)c * oh + y) * ow + x] = (int8_t)q;
                }
            }
}
