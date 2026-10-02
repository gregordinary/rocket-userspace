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
#include <time.h>

#include "rocket_cube.h"   /* the NPU cube index math + blocked moves */
#include "rocket_npu.h"
#include "rocket_hw_profile.h"   /* CBUF bank count from the active hardware profile */
#include "npu_matmul.h"
#include "rocket_conv.h"
#include "rocket_conv_internal.h"
#include "rocket_activation.h"   /* lut_epilogue_t builder + ref (conv->act fusion) */
#include "rocket_affinity.h"
#include "rocket_chain.h"        /* rkt_chain_pack — gapped multi-task batched submit */
#include "rocket_scratch.h"      /* per-ctx bump arena for the tilers' host scratch */
#include "rocket_log.h"     // centralized log channel
#include "npu_requant.h"    /* the one float scale -> OUT_CVT pair derivation */

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
 * tasks as separate HW kicks, but pays ONE submit syscall + ONE fence wait + ONE IOMMU
 * attach for the whole slice. Bit-identical to the per-tile path; opt-in while it
 * proves out. CONV_RC_STRIDE is the per-task regcmd slot (>= gen_conv2d_int8's 256-word
 * cap). */
#define CONV_RC_STRIDE 256

/* ROCKET_CONV_PROFILE=1: one line at exit per RK3588 conv program (int8 direct, int8
 * depthwise, fp16), summing its jobs' phases over the process. A bucket is a sum of
 * CLOCK_MONOTONIC intervals on the thread that ran the job, so under the conv pool's
 * workers it also counts time descheduled: an upper bound on its term, never a share of
 * the wall. in = the input BO's zero, scatter and flush, with a tile's sub-input
 * materialize; wt = the weight BO's zero, scatter and flush; outz = the output BO's zero
 * and flush; gen = the regcmd; submit; wait = submit return to fence, the output BO's
 * invalidate included (PREP_BO waits and syncs in one ioctl); read = the de-scatter and a
 * tile's copy into the caller's output; fini = the output BO's FINI_BO after it. */
enum { CP_IN, CP_WT, CP_OUTZ, CP_GEN, CP_SUBMIT, CP_WAIT, CP_READ, CP_FINI, CP_N };
enum { CPP_I8, CPP_DW8, CPP_F16, CPP_I8Q, CPP_N };
static int g_cprof = -1;
static pthread_mutex_t g_cprof_mu = PTHREAD_MUTEX_INITIALIZER;
static double g_cprof_ms[CPP_N][CP_N];
static long g_cprof_jobs[CPP_N];
/* Bytes of the input and output BOs each job synced (their whole size) against the bytes
 * the job needed: a context's BOs only grow, so the ratio is what a smaller call pays for
 * the largest one before it. */
static double g_cprof_bo[CPP_N][4];              /* in size, in need, out size, out need */
static int g_cprof_armed;
static int cprof_on(void)
{
    int v = __atomic_load_n(&g_cprof, __ATOMIC_RELAXED);
    if (v < 0) {
        v = getenv("ROCKET_CONV_PROFILE") != NULL;
        __atomic_store_n(&g_cprof, v, __ATOMIC_RELAXED);
    }
    return v;
}
static double cprof_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static void cprof_dump(void)
{
    static const char *prog[CPP_N] = { "int8-direct", "int8-dw", "fp16", "int8-direct-q" };
    for (int g = 0; g < CPP_N; g++) {
        const double *m = g_cprof_ms[g];
        if (!g_cprof_jobs[g]) continue;
        ROCKET_LOGI("ROCKET conv profile %s total(ms): in=%.0f wt=%.0f outz=%.0f gen=%.0f "
                    "submit=%.0f wait=%.0f read=%.0f fini=%.0f  over %ld jobs; BO MB "
                    "synced/needed: in %.0f/%.0f out %.0f/%.0f\n", prog[g], m[CP_IN],
                    m[CP_WT], m[CP_OUTZ], m[CP_GEN], m[CP_SUBMIT], m[CP_WAIT], m[CP_READ],
                    m[CP_FINI],
                    g_cprof_jobs[g], g_cprof_bo[g][0] / 1e6, g_cprof_bo[g][1] / 1e6,
                    g_cprof_bo[g][2] / 1e6, g_cprof_bo[g][3] / 1e6);
    }
}
static void cprof_bo(int prog, size_t in_size, size_t in_need, size_t out_size, size_t out_need)
{
    pthread_mutex_lock(&g_cprof_mu);
    g_cprof_bo[prog][0] += (double)in_size;  g_cprof_bo[prog][1] += (double)in_need;
    g_cprof_bo[prog][2] += (double)out_size; g_cprof_bo[prog][3] += (double)out_need;
    pthread_mutex_unlock(&g_cprof_mu);
}
static void cprof_add(int prog, const double *t, int jobs)
{
    pthread_mutex_lock(&g_cprof_mu);
    if (!g_cprof_armed) { atexit(cprof_dump); g_cprof_armed = 1; }
    for (int i = 0; i < CP_N; i++) g_cprof_ms[prog][i] += t[i];
    g_cprof_jobs[prog] += jobs;
    pthread_mutex_unlock(&g_cprof_mu);
}
/* Close the phase that ran since `t` into acc[ph] and restart the clock. */
#define CP_MARK(on, acc, ph, t) do { if (on) { double n_ = cprof_now();      \
        (acc)[ph] += n_ - (t); (t) = n_; } } while (0)

/* Pack C planes of IH rows into a feature cube of granule G (G*esz = 16 bytes) at IHj
 * rows: the rc_* transposes write every lane of every real pixel (0 past C), the rows
 * IH..IHj-1 are zeroed per group, and so are the two CBUF banks past the cube, which is
 * as far as the int8 feature DMA's one-bank over-read reaches. The rest of the BO is
 * never read, so it is not zeroed. `bo_bytes` bounds the tail. */
static void conv_pack_planes(void *cube, size_t bo_bytes, const void *in, int C, int IH,
                             int IW, int IHj, int G, int esz)
{
    const size_t hw = (size_t)IH * IW, line = (size_t)G * esz;
    const size_t plane = (size_t)IHj * IW * line;
    const int ng = (C + G - 1) / G;
    if ((size_t)ng * plane > bo_bytes) {
        /* Whole groups do not fit: the BO was sized at C, not at whole groups. Keep the
         * per-channel pass, which writes only the real lanes. */
        memset(cube, 0, bo_bytes);
        for (int c = 0; c < C; c++)
            for (size_t p = 0; p < hw; p++)
                memcpy((uint8_t *)cube + (size_t)(c / G) * plane + (p * G + c % G) * esz,
                       (const uint8_t *)in + ((size_t)c * hw + p) * esz, esz);
        return;
    }
    for (int g = 0; g < ng; g++) {
        uint8_t *d = (uint8_t *)cube + (size_t)g * plane;
        const int nc = C - g * G < G ? C - g * G : G;
        if (G == 16) {
            const uint8_t *pl[16];
            for (int j = 0; j < nc; j++) pl[j] = (const uint8_t *)in + ((size_t)(g * G + j) * hw);
            rc_planes_to_group16_u8(d, pl, nc, hw);
        } else {
            const uint16_t *pl[8];
            for (int j = 0; j < nc; j++) pl[j] = (const uint16_t *)in + ((size_t)(g * G + j) * hw);
            rc_planes_to_group8_u16((uint16_t *)d, pl, nc, hw);
        }
        memset(d + hw * line, 0, plane - hw * line);
    }
    const size_t used = (size_t)ng * plane;
    size_t tail = used + 2 * (size_t)CBUF_BANK;
    if (tail > bo_bytes) tail = bo_bytes;
    if (tail > used) memset((uint8_t *)cube + used, 0, tail - used);
}
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

static inline long datain_w(int out_cols, int sx, int KW, int dx)
{
    return (long)(out_cols - 1) * sx + (long)(KW - 1) * dx + 1;
}

/* The largest tile the CNA can be told about. DATA_SIZE0 holds the sub-input width and
 * height in 11 bits, and CONV_CON2's FEATURE_GRAINS, which the generators set to the job's
 * height plus one, holds 10. So a tile is at most 1022 rows by 2047 columns, and the
 * regcmd emitters refuse past either. The CBUF budget bounds a tile's AREA, which leaves
 * one axis free to pass these when the channels are thin: an int8 tile 2 rows by 2300
 * columns at 32 channels fits the budget and programs a width of 252. */
#define CNA_TILE_MAX_H 1022
#define CNA_TILE_MAX_W 2047

/* The largest pad one job can program. CNA_PAD_CON0 holds PAD_TOP and PAD_LEFT in 4 bits
 * each, and a pad of 16 behaves exactly as 0, 17 as 1 [HW sweep, tests/deconv_pad_probe.c]:
 * a full, plausible surface at the wrong offset. The tilers materialize the pad into each
 * tile's input and program 0, so only a one-job path can reach the field. Each forward one
 * sends a larger pad to its tiler instead, and the direct deconvolution route admits at most
 * 11 (rocket_conv_transpose2d_route). A "same" dilated conv reaches the limit: k7 at dilation
 * 12 pads 36. tests/conv_pad_field_gate scores both sides of it. */
#define CNA_PAD_FIELD_MAX 15
static inline int cna_pad_fits(int pt, int pl)
{
    return pt <= CNA_PAD_FIELD_MAX && pl <= CNA_PAD_FIELD_MAX;
}

/* The rows a tile costs the CBUF. A job pads a sub-input shorter than CONV_MIN_DATAIN_H up
 * to it, so a budget taken at the real rows under-counts a short tile, and the generator's
 * bank check then refuses a tile the budget admitted. */
static inline long job_rows(long rows)
{
    return rows < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : rows;
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
    if (IH > CNA_TILE_MAX_H || IW > CNA_TILE_MAX_W) return 0;   /* past the CNA's fields */
    size_t fd_bytes = (size_t)Cc * job_rows(IH) * IW * sizeof(_Float16);
    return fd_bytes <= dw_feat_budget();                 /* feature within validated budget */
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
 * A pool of the BOs a conv job needs (IOVA guard / input / weight / regcmd / output / the
 * depthwise bias), cached on a borrowed fd. A job takes an optional ctx: NULL => the
 * legacy alloc-per-call-and-free path (byte-for-byte unchanged); non-NULL => borrow the
 * pool (no per-call alloc/free). The fd is borrowed, never opened/closed here.
 *
 * EACH ROLE HOLDS ONE BO PER POWER-OF-TWO SIZE CLASS, from 64 KiB, and a job takes the class
 * that fits it. A job zeroes and syncs its BOs whole (PREP_BO/FINI_BO cover the object, and
 * this kernel has no ranged form), so one grow-only BO a role made every small conv pay for
 * the largest before it: MobileDet's int8 direct conv synced 2228 MB of output BO for 449 MB
 * of output over 21 invokes [HW, RK1, ROCKET_CONV_PROFILE]. A class BO is at most twice what
 * its job needs, and a role holds at most twice its largest job's bytes in all. */
#define CONV_BO_CLASSES 10                      /* 64 KiB << 0..9; the last takes anything larger */
struct rocket_conv_ctx {
    int       fd;                              /* borrowed; not owned */
    rocket_bo guard;                           /* ptr==NULL until first use */
    rocket_bo in[CONV_BO_CLASSES], wt[CONV_BO_CLASSES], rc[CONV_BO_CLASSES];
    rocket_bo out[CONV_BO_CLASSES];
    rocket_bo bias[CONV_BO_CLASSES];           /* int8 depthwise-out per-OC int32 bias cube */
    rocket_scratch_pool scratch;               /* grow-only host scratch for the spatial tilers */
};

/* The BO for one role at a job's `*need` bytes: the context's class BO, with *need raised to
 * the class size so the BO is allocated once at it, or `local` without a context. */
static rocket_bo *conv_bo(rocket_conv_ctx *ctx, rocket_bo *cls, rocket_bo *local, size_t *need)
{
    if (!ctx) return local;
    size_t cap = (size_t)64 << 10;
    int k = 0;
    while (cap < *need && k < CONV_BO_CLASSES - 1) { cap <<= 1; k++; }
    if (cap > *need) *need = cap;
    return &cls[k];
}

rocket_conv_ctx *rocket_conv_ctx_create(int fd)
{
    rocket_conv_ctx *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->fd = fd;
    return c;                                   /* fd<0 is fine: BOs are never allocated */
}

/* Free every BO a context holds (rocket_bo_free is a no-op on a {0} BO). */
static void conv_ctx_free_bos(rocket_conv_ctx *c, int fd)
{
    for (int k = 0; k < CONV_BO_CLASSES; k++) {
        rocket_bo_free(fd, &c->bias[k]);
        rocket_bo_free(fd, &c->out[k]);
        rocket_bo_free(fd, &c->rc[k]);
        rocket_bo_free(fd, &c->wt[k]);
        rocket_bo_free(fd, &c->in[k]);
    }
    rocket_bo_free(fd, &c->guard);
}

void rocket_conv_ctx_free(rocket_conv_ctx *c)
{
    if (!c) return;
    if (c->fd >= 0) conv_ctx_free_bos(c, c->fd);
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
/* The conv tilers' smallest validated sub-input height: a tile below it computes wrong
 * (see conv2d_fp16_tiled), and ROCKET_CONV_MIN_IH moves it for a sweep. */
static int conv_min_ihs(void)
{
    int min_ihs = 6;
    const char *e = getenv("ROCKET_CONV_MIN_IH");
    if (e && atoi(e) > 0) min_ihs = atoi(e);
    return min_ihs < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : min_ihs;
}

int rocket_conv2d_plan(const rocket_conv2d_desc *d)
{
    if (!d) return -1;
    if (d->ic <= 0 || d->ih <= 0 || d->iw <= 0 || d->oc <= 0 || d->kh <= 0 || d->kw <= 0)
        return -1;
    if (d->stride_x <= 0 || d->stride_y <= 0 || d->dil_x <= 0 || d->dil_y <= 0)
        return -1;
    /* An EXPLICIT output extent is the RK3576 int8 path's way of asking for a trailing
     * pad. This planner serves the RK3588 fp16 generator, whose pad registers this library
     * has only ever driven symmetrically, so an extent it did not derive is refused rather
     * than programmed on an untested encoding. */
    if ((d->oh && d->oh != rocket_conv_out_dim(d->ih, d->kh, d->stride_y, d->pad_top,
                                               d->dil_y)) ||
        (d->ow && d->ow != rocket_conv_out_dim(d->iw, d->kw, d->stride_x, d->pad_left,
                                               d->dil_x)))
        return -1;
    /* `direct_datapath` is satisfied by construction here rather than ignored: this
     * generator has one convolution encoding, and IC < 32 on it is zero-padded up to 32
     * exactly as the flag asks for. It is the RK3576 that has a second, packed-image
     * encoding for four or fewer channels and so needs to be told which one. */
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
        if (ih_min > CNA_TILE_MAX_H || iw_min > CNA_TILE_MAX_W) return -4;
        if ((size_t)G * job_rows(ih_min) * iw_min * sizeof(_Float16) > dw_feat_budget()) return -4;
    } else {
        size_t wpk = (size_t)d->kw * d->kh * IC * sizeof(_Float16);    /* one OC kernel */
        if (wpk > CBUF_BANK) return -4;                                /* a kernel must fit a bank */
        /* The tiler's smallest output-channel tile is 16 kernels, and its weight shares the
         * CBUF banks with the feature tile, at least one bank left for the feature. IC 1024
         * at k4 needs 16 banks there: this plan passed it and the run returned -4 [HW, RK1,
         * tests/ct_model_bench.c, pix2pix's 1024-channel layers]. */
        const int nbanks = rocket_hw_current()->cbuf_banks;
        const size_t wb16 = ((size_t)16 * wpk + CBUF_BANK - 1) / CBUF_BANK;
        if (wb16 > (size_t)(nbanks - 1)) return -4;
        size_t fbudget = (size_t)(nbanks - (int)wb16) * CBUF_BANK;
        if (fbudget > CONV_FEAT_BUDGET) fbudget = CONV_FEAT_BUDGET;
        /* direct tiles over OC + OH-rows + OW-cols; the MINIMAL tile must fit what the
         * weight leaves: one output column, and the fewest rows whose sub-input reaches the
         * tiler's validated floor (conv_min_ihs), as the run chooses them. */
        int rlo = 1;
        while (rlo < OH && datain_h(rlo, d->stride_y, d->kh, d->dil_y) < conv_min_ihs()) rlo++;
        long ih_min = datain_h(rlo, d->stride_y, d->kh, d->dil_y);
        long iw_min = datain_w(1, d->stride_x, d->kw, d->dil_x);
        if (ih_min > CNA_TILE_MAX_H || iw_min > CNA_TILE_MAX_W) return -4;
        if ((size_t)IC * job_rows(ih_min) * iw_min * sizeof(_Float16) > fbudget) return -4;
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
                          const _Float16 *in, const _Float16 *W, _Float16 *out,
                          int dsy, int dsx)
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
    /* The conv->activation fusion appends the 2*513-entry LE/LO table upload to the
     * regcmd (~1030 extra ops), so size the regcmd buffer/BO for it when fused. */
    const int NREGS = jact ? 2048 : 256;
    size_t in_need  = in_elems  * sizeof(_Float16) + CBUF_BANK;
    size_t wt_need  = wt_elems  * sizeof(_Float16) + CBUF_BANK;
    size_t rc_need  = (size_t)NREGS * sizeof(uint64_t);
    size_t out_need = out_elems * sizeof(_Float16) + CBUF_BANK;
    rocket_bo lguard = {0}, lin = {0}, lwt = {0}, lrc = {0}, lout = {0};
    rocket_bo *guard  = ctx ? &ctx->guard : &lguard;
    rocket_bo *in_bo  = conv_bo(ctx, ctx ? ctx->in : NULL,  &lin,  &in_need);
    rocket_bo *wt_bo  = conv_bo(ctx, ctx ? ctx->wt : NULL,  &lwt,  &wt_need);
    rocket_bo *rc_bo  = conv_bo(ctx, ctx ? ctx->rc : NULL,  &lrc,  &rc_need);
    rocket_bo *out_bo = conv_bo(ctx, ctx ? ctx->out : NULL, &lout, &out_need);
    uint64_t regs[2048] = {0};
    int rc = -1;

    if (rocket_bo_ensure32(fd, guard, 4096) < 0 ||                                       /* off IOVA 0 */
        rocket_bo_ensure32(fd, in_bo,  in_need) < 0 ||
        rocket_bo_ensure32(fd, wt_bo,  wt_need) < 0 ||
        rocket_bo_ensure32(fd, rc_bo,  rc_need) < 0 ||
        rocket_bo_ensure32(fd, out_bo, out_need) < 0) {
        ROCKET_LOGE("rocket_conv2d_fp16: BO alloc failed\n");
        goto out;
    }
    const int cp = cprof_on();
    double cpa[CP_N] = {0}, cpt = cp ? cprof_now() : 0.0;

    /* scatter input feature -> NC1HWC2 cube (C2=8) */
    rocket_bo_prep(fd, in_bo, 1, 0);
    /* feature_data()'s layout, (ic/8, h, w, ic%8) over IHj rows, eight channels a pass:
     * one call per element was ~80% of a depthwise call [HW, RK1,
     * tests/conv_pack_cost_probe.c]. Rows >= IH are 0. */
    conv_pack_planes(in_bo->ptr, in_bo->size, in, IC, IH, IW, IHj, 8, 2);
    rocket_bo_fini(fd, in_bo);
    CP_MARK(cp, cpa, CP_IN, cpt);

    /* scatter weights -> the conv weight cube. direct: oc1/ic1/kh/kw/oc2/ic2 from
     * W[OC][IC][KH][KW]. depthwise: ic1/kh/kw/ic2 (group G) from W[OC][1][KH][KW]. */
    rocket_bo_prep(fd, wt_bo, 1, 0);
    memset(wt_bo->ptr, 0, wt_bo->size);
    {
        _Float16 *dst = wt_bo->ptr;
        if (DW) {
            /* OC==IC, one filter per channel: weight_conv_dw_fp16()'s layout,
             * (c/G, kh, kw, c%G), from W[c][0][kh][kw]. */
            const size_t taps = (size_t)KH * KW;
            for (int c = 0; c < IC; c++) {
                _Float16 *d = dst + (size_t)(c / G) * taps * G + (c % G);
                const _Float16 *src = W + (size_t)c * taps;
                for (size_t t = 0; t < taps; t++) d[t * G] = src[t];
            }
        } else {
            ROCKET_CONV_WT_SCATTER(_Float16, 16, dst, W, OC, IC, KH, KW);
        }
    }
    rocket_bo_fini(fd, wt_bo);
    CP_MARK(cp, cpa, CP_WT, cpt);

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
        .deconv_sy = (uint8_t)dsy, .deconv_sx = (uint8_t)dsx,   /* 1 == off */
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
    CP_MARK(cp, cpa, CP_GEN, cpt);

    /* zero + hand the output BO to the device */
    rocket_bo_prep(fd, out_bo, 1, 0);
    memset(out_bo->ptr, 0, out_bo->size);
    rocket_bo_fini(fd, out_bo);
    CP_MARK(cp, cpa, CP_OUTZ, cpt);

    {
        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo->dma_address,
                                  .regcmd_count = p.task_count };
        uint32_t in_h[]  = { in_bo->handle, wt_bo->handle, rc_bo->handle };
        uint32_t out_h[] = { out_bo->handle };
        ret = rocket_submit_tasks(fd, &task, 1, in_h, 3, out_h, 1);
        if (ret) { ROCKET_LOGE("rocket_conv2d_fp16: submit failed (%d)\n", ret); goto out; }
    }
    CP_MARK(cp, cpa, CP_SUBMIT, cpt);

    /* read back: de-scatter the output cube (OC/8, OH, OW, 8) */
    ret = rocket_bo_prep(fd, out_bo, 0, 2000000000ULL);   /* 2s wait */
    if (ret) { ROCKET_LOGE("rocket_conv2d_fp16: wait timeout (%d)\n", ret); goto out; }
    CP_MARK(cp, cpa, CP_WAIT, cpt);
    {
        const _Float16 *src = out_bo->ptr;
        const size_t plane = (size_t)OH * OW * 8, hw = (size_t)OH * OW;
        for (int g = 0; g < (OC + 7) / 8; g++) {     /* eight channels a pass */
            const int nc = OC - g * 8 < 8 ? OC - g * 8 : 8;
            uint16_t *pl[8];
            for (int j = 0; j < nc; j++) pl[j] = (uint16_t *)(out + (size_t)(g * 8 + j) * hw);
            rc_group8_to_planes_u16(pl, nc, (const uint16_t *)(src + (size_t)g * plane), hw);
        }
    }
    CP_MARK(cp, cpa, CP_READ, cpt);
    rocket_bo_fini(fd, out_bo);
    CP_MARK(cp, cpa, CP_FINI, cpt);
    if (cp) {
        cprof_add(CPP_F16, cpa, 1);
        cprof_bo(CPP_F16, in_bo->size, in_elems * sizeof(_Float16), out_bo->size,
                 out_elems * sizeof(_Float16));
    }
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

/* One direct fp16 conv job with the output extent GIVEN rather than derived from the
 * input, stride and pad. The forward entries always derive it, and the CNA deconvolution
 * mode (ROCKET_CNA_DECONV and its stride knobs) needs the transposed extent, which the
 * forward arithmetic cannot produce. Internal: rocket_conv_internal.h. Stride and
 * dilation are 1, the only values the deconvolution mode is run with. */
int rocket_conv2d_fp16_job_extent(int fd, int IC, int IH, int IW, int OC, int OH, int OW,
                                  int KH, int KW, int pt, int pl,
                                  const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    if (fd < 0) return -1;   /* the one-job CPU fallback derives its own extent */
    return conv2d_one_job(fd, NULL, IC, IH, IW, OC, OH, OW, KH, KW, 1, 1, pt, pl, 1, 1,
                          0, 0, NULL, in, W, out, 1, 1);
}

/* ---- the CNA deconvolution mode as a route --------------------------------------------
 *
 * rocket_conv_transpose2d_fp16's hardware route: one direct fp16 job per output-channel
 * tile with the deconvolution mode on, the COMPACT input, the transposed output extent and
 * a pad of k-1-p. Against the lowering it skips the host's s^2-larger dilated input, and
 * it is 0.35-0.54x the lowering's wall per call at decoder shapes [HW sweep, Turing RK1,
 * 2026-09-23, tests/deconv_extent_probe.c bench]. Pure fit check, then the run. */
int rocket_conv2d_fp16_deconv_fits(int IC, int IH, int IW, int OC, int OH, int OW,
                                   int KH, int KW)
{
    const int CBUF_BANKS = rocket_hw_current()->cbuf_banks;
    const int IHj = IH < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : IH;
    const size_t feat = (size_t)IC * IHj * IW * sizeof(_Float16);
    const size_t fbanks = (feat + CBUF_BANK - 1) / CBUF_BANK;
    const size_t w16 = ((size_t)16 * IC * KH * KW * sizeof(_Float16) + CBUF_BANK - 1) / CBUF_BANK;

    if (IC <= 0 || IH <= 0 || IW <= 0 || OC <= 0 || OH <= 0 || OW <= 0) return 0;
    if (IC % 32) return 0;                                     /* whole K-groups only */
    if (feat > CONV_FEAT_BUDGET) return 0;                     /* one pass holds the input */
    if ((size_t)KH * KW * IC * sizeof(_Float16) > CBUF_BANK) return 0;  /* a kernel per bank */
    if (fbanks + w16 > (size_t)CBUF_BANKS) return 0;           /* room for 16 kernels */
    /* The geometry fields: DATA_SIZE2 is 11 bits of width, DATA_SIZE3 18 bits of pixels. */
    if (OW > 0x7FF || OH > 0x7FF || (long)OH * OW > 0x3FFFF) return 0;
    return 1;
}

int rocket_conv2d_fp16_deconv(int fd, rocket_conv_ctx *ctx, int IC, int IH, int IW, int OC,
                              int OH, int OW, int KH, int KW, int sy, int sx, int pt, int pl,
                              const _Float16 *in, const _Float16 *Wf, _Float16 *out)
{
    const int CBUF_BANKS = rocket_hw_current()->cbuf_banks;
    const int dfd = ctx ? ctx->fd : fd;
    const int OCp = ((OC + 15) / 16) * 16;       /* the fp16 weight oc group */
    const int IHj = IH < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : IH;
    const size_t fbanks = ((size_t)IC * IHj * IW * sizeof(_Float16) + CBUF_BANK - 1) / CBUF_BANK;
    const _Float16 *W = Wf;
    _Float16 *Wpad = NULL, *opad = NULL, *o = out;
    size_t wcap;
    int oc0, OCt, ret = 0;

    if (dfd < 0 || !rocket_conv2d_fp16_deconv_fits(IC, IH, IW, OC, OH, OW, KH, KW)) return -4;

    /* OC not a multiple of 16: zero kernels pad the tile and the real channels are sliced
     * off the front, as the forward path does. */
    if (OCp != OC) {
        Wpad = calloc((size_t)OCp * IC * KH * KW, sizeof *Wpad);
        opad = malloc((size_t)OCp * OH * OW * sizeof *opad);
        if (!Wpad || !opad) { free(Wpad); free(opad); return -1; }
        memcpy(Wpad, Wf, (size_t)OC * IC * KH * KW * sizeof *Wpad);
        W = Wpad;
        o = opad;
    }

    /* The weight tile takes what the compact input leaves, capped at 4 banks as the
     * forward tiler keeps it; one 16-channel tile always fits (the fit check). */
    #define DECONV_WBANKS(oct) \
        (((size_t)(oct) * IC * KH * KW * sizeof(_Float16) + CBUF_BANK - 1) / CBUF_BANK)
    wcap = (size_t)CBUF_BANKS - fbanks;
    if (wcap > 4) wcap = 4;
    OCt = OCp;
    while (OCt > 16 && DECONV_WBANKS(OCt) > wcap) OCt -= 16;
    #undef DECONV_WBANKS

    for (oc0 = 0; oc0 < OCp && !ret; oc0 += OCt) {
        const int n = OCp - oc0 < OCt ? OCp - oc0 : OCt;
        ret = conv2d_one_job(dfd, ctx, IC, IH, IW, n, OH, OW, KH, KW, 1, 1, pt, pl, 1, 1,
                             0, 0, NULL, in, W + (size_t)oc0 * IC * KH * KW,
                             o + (size_t)oc0 * OH * OW, sy, sx);
    }
    if (!ret && o != out) memcpy(out, o, (size_t)OC * OH * OW * sizeof *out);
    free(Wpad);
    free(opad);
    return ret;
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
    #define DW_FEAT(rh, cw) ((size_t)Cn * (size_t)job_rows(datain_h(rh, sy, KH, dy)) * \
        (size_t)datain_w(cw, sx, KW, dx) * sizeof(_Float16))

    const int min_ihs = conv_min_ihs();   /* never below the HW row floor */
    int rht_lo = 1;
    while (rht_lo < OH && datain_h(rht_lo, sy, KH, dy) < min_ihs) rht_lo++;
    int rht = OH, cwt = OW;
    while (rht > rht_lo && datain_h(rht, sy, KH, dy) > CNA_TILE_MAX_H) rht--;   /* CNA fields */
    while (cwt > 1 && datain_w(cwt, sx, KW, dx) > CNA_TILE_MAX_W) cwt--;
    while (rht > rht_lo && DW_FEAT(rht, cwt) > budget) rht--;
    if (DW_FEAT(rht, cwt) > budget)                    /* full-width band still over -> narrow cols */
        while (cwt > 1 && DW_FEAT(rht, cwt) > budget) cwt--;
    if (DW_FEAT(rht, cwt) > budget ||                  /* minimal tile still over */
        datain_h(rht, sy, KH, dy) > CNA_TILE_MAX_H || datain_w(cwt, sx, KW, dx) > CNA_TILE_MAX_W)
        return -4;

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
                                 sy, sx, 0, 0, dy, dx, 1, G, NULL, sub_in, W, sub_out, 1, 1);
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
        const int spatial = !dw_chunk_fits(Cc, IH, IW, KH, KW)    /* even G overflows -> band */
                            || !cna_pad_fits(pt, pl);                /* pad past PAD_CON0 -> band */
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
                                     dy, dx, 1, G, NULL, in_c, W_c, out_c, 1, 1);
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
        if (IC == ICr && IH <= CNA_TILE_MAX_H && IW <= CNA_TILE_MAX_W && cna_pad_fits(pt, pl) &&
            (size_t)OC * IC * KH * KW * sizeof(_Float16) <= (size_t)4 * CBUF_BANK &&
            (size_t)IC * job_rows(ih_full) * iw_full * sizeof(_Float16) <= CONV_FEAT_BUDGET &&
            (size_t)IC * job_rows(IH) * IW * sizeof(_Float16) <= CONV_FEAT_BUDGET)
            return conv2d_one_job(fd, ctx, IC, IH, IW, OC, OH, OW, KH, KW, sy, sx, pt, pl,
                                  dy, dx, 0, 0, act, in, W, out, 1, 1);
    }

    /* feature bytes for a (rh out-rows x cw out-cols) tile's materialized sub-input */
    #define CONV_FEAT(rh, cw) ((size_t)IC * (size_t)job_rows(datain_h(rh, sy, KH, dy)) * \
        (size_t)datain_w(cw, sx, KW, dx) * sizeof(_Float16))

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
    const int min_ihs = conv_min_ihs();   /* never below the HW row floor */

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
    while (rht > rht_lo && datain_h(rht, sy, KH, dy) > CNA_TILE_MAX_H) rht--;   /* CNA fields */
    while (cwt > 1 && datain_w(cwt, sx, KW, dx) > CNA_TILE_MAX_W) cwt--;
    while (rht > rht_lo && CONV_FEAT(rht, cwt) > feat_budget) rht--;
    if (CONV_FEAT(rht, cwt) > feat_budget)             /* full-width band still over -> narrow cols */
        while (cwt > 1 && CONV_FEAT(rht, cwt) > feat_budget) cwt--;
    if (CONV_FEAT(rht, cwt) > feat_budget ||           /* minimal tile still over */
        datain_h(rht, sy, KH, dy) > CNA_TILE_MAX_H || datain_w(cwt, sx, KW, dx) > CNA_TILE_MAX_W) {
        ret = -4; goto done; }

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
                                     sy, sx, 0, 0, dy, dx, 0, 0, act, sub_in, Wslice, sub_out, 1, 1);
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
/* ============================================================================
 * SECTION — the per-chip dispatch point
 *
 * Every generator this file drives emits the RK3588 geometry-register encoding, and
 * that encoding is IP-revision-specific: the RK3576 re-packs the CNA/CORE/DPU blocks
 * at the same block bases, so a program built here does not compute slowly or
 * approximately there — it submits, the job completes, and the DPU writes nothing.
 *
 * Where the RK3576's own entry has the SAME semantics, the public entry routes to it
 * and a caller writes nothing chip-specific. Where it does not, the entry refuses and
 * names the one that does; emulating an on-chip requant with a host one is a different
 * arithmetic, not a wrapper. This mirrors rocket_matmul.c's mm_wrong_encoding().
 */
static int conv_is_rk3576(void)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    return hw && hw->name && !strcmp(hw->name, "rk3576");
}

static int conv_wrong_encoding(const char *entry, const char *instead)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (hw && hw->name && !strcmp(hw->name, "rk3588")) return 0;
    ROCKET_LOGE("%s emits the RK3588 geometry-register encoding, which the %s does not "
                "run — the job would complete and write nothing. Use %s\n",
                entry, hw && hw->name ? hw->name : "?", instead);
    return 1;
}

int rocket_conv2d_fp16(int fd, const rocket_conv2d_desc *d,
                       const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    if (conv_is_rk3576())
        return rocket_conv2d_fp16_rk3576(fd, d, in, W, out);
    if (conv_wrong_encoding("rocket_conv2d_fp16", "the chip's own conv encoder"))
        return ROCKET_E_UNSUPPORTED;
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

/* The int8-OUT form of the direct conv (rocket_conv2d_int8_q). A job programs Mesa's direct
 * int8-output writer (gen_conv2d_int8 at int8_out=1: QD_EN 1, DATA_FORMAT 0, size_e 1,
 * SURF_ADD x2, the folded per-OC bias in the BS stage, CPEND at OW_OP = -w_zp, the OUT_CVT
 * requant) and writes int8, requantized on chip [HW sweep, RK1, tests/conv_i8out_probe].
 * NULL keeps the int32-raw writer. Every function in this part takes it, so the two forms
 * share one planner, one tiler and one job body and cannot disagree about a tile. A is the
 * bias of the job's own OC slice: a caller passing a tile offsets it.
 *
 * The per-channel form (rocket_conv2d_int8_q_perc) sets C: the BS stage then multiplies
 * channel c's biased accumulator by C[c] and shifts by bs_shift, both signs, before the
 * OUT_CVT applies the job's one gain (in_scale, with w_scale and out_scale 1). A and C then
 * travel in the BRDMA's 64-byte coefficient group per 8 channels [HW sweep, RK1,
 * tests/conv_i8out_probe]. C is offset with A. */
typedef struct {
    const int32_t *A;          /* [OC] folded bias, bias - in_zp * sum_{ic,kh,kw}(w - w_zp) */
    float in_scale, w_scale, out_scale;
    int in_zp, w_zp, out_zp;   /* int8 domain */
    const int16_t *C;          /* per-channel: [OC] BS multiplier, or NULL (per-tensor) */
    unsigned bs_shift;         /* per-channel: the BS shift for either sign of product */
} i8q_t;

/* The OUT_CVT pair the generator programs, and its rounding (ties to even), for the host
 * model of the int8-out job below. */
static void i8q_cvt(const i8q_t *q, unsigned *mul, unsigned *shift)
{
    npu_out_cvt_pair((q->in_scale * q->w_scale) / q->out_scale, mul, shift);
}
static int64_t i8q_rne_shift(int64_t p, unsigned shift)
{
    int64_t v = p;
    if (shift) {
        const int64_t half = (int64_t)1 << (shift - 1);
        v = (p + half) >> shift;
        if ((p & (((int64_t)1 << shift) - 1)) == half && (v & 1)) v -= 1;
    }
    return v;
}
static int8_t i8q_requant(int64_t acc, unsigned mul, unsigned shift, int out_zp)
{
    int64_t v = i8q_rne_shift(acc * (int64_t)mul, shift) + out_zp;
    return (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
}
/* The per-channel BS stage: (acc * C) held wide, shifted to nearest (ties to even), then
 * saturated to int32 before the OUT_CVT. */
static int64_t i8q_bs(int64_t acc, int16_t c, unsigned shift)
{
    int64_t v = i8q_rne_shift(acc * (int64_t)c, shift);
    return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : v;
}

/* The int8-out job on the host: what the program computes, for fd < 0. The CNA pads every
 * programmed channel with in_zp, CPEND adds -w_zp times the window sum of that input, and the
 * BS stage adds A: acc = A[oc] + sum xp*w - w_zp * sum xp, then the OUT_CVT. */
static void conv2d_int8_q_host(int IC, int IH, int IW, int OC, int OH, int OW, int KH, int KW,
                               int sy, int sx, int pt, int pl, int dy, int dx,
                               const int8_t *in, const int8_t *W, const i8q_t *q, int8_t *out)
{
    unsigned mul, shift;
    i8q_cvt(q, &mul, &shift);
    for (int oc = 0; oc < OC; oc++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int64_t p = 0, sm = 0;
                for (int ic = 0; ic < IC; ic++)
                    for (int kh = 0; kh < KH; kh++) {
                        const int ih = oh * sy + kh * dy - pt;
                        for (int kw = 0; kw < KW; kw++) {
                            const int iw = ow * sx + kw * dx - pl;
                            const int xp = (ih >= 0 && ih < IH && iw >= 0 && iw < IW)
                                         ? in[((size_t)ic * IH + ih) * IW + iw] : q->in_zp;
                            p  += (int64_t)xp * W[(((size_t)oc * IC + ic) * KH + kh) * KW + kw];
                            sm += xp;
                        }
                    }
                int64_t acc = q->A[oc] + p - (int64_t)q->w_zp * sm;
                if (q->C) acc = i8q_bs(acc, q->C[oc], q->bs_shift);
                out[((size_t)oc * OH + oh) * OW + ow] = i8q_requant(acc, mul, shift, q->out_zp);
            }
}

/* The coefficient cube BRDMA reads for one int8-out job over OC channels (a multiple of 8):
 * the int32 A per channel, or with C the 64-byte group per 8 channels, A[8] int32 at 0,
 * B[8] int16 at 32 (0: no per-channel weight zero point), C[8] int16 at 48. */
static size_t i8q_coeff_bytes(const i8q_t *q, int OC)
{
    return q->C ? (size_t)(OC + 7) / 8 * 64 : (size_t)OC * sizeof(int32_t);
}
static void i8q_coeff_fill(void *dst, const i8q_t *q, const int32_t *A, const int16_t *C, int OC)
{
    if (!q->C) {
        memcpy(dst, A, (size_t)OC * sizeof(int32_t));
        return;
    }
    memset(dst, 0, i8q_coeff_bytes(q, OC));
    for (int c = 0; c < OC; c++) {
        uint8_t *g = (uint8_t *)dst + (size_t)(c / 8) * 64;
        memcpy(g + 4 * (c % 8), &A[c], 4);
        memcpy(g + 48 + 2 * (c % 8), &C[c], 2);
    }
}

/* Run ONE int8 DIRECT conv as a single NPU job (must fit one CBUF pass). OC MUST be a
 * multiple of 32 (the int8 weight oc-group) — conv2d_int8_run pads it. in [IC][IH][IW]
 * int8, W [OC][IC][KH][KW] int8, out [OC][OH][OW] int32, or int8 with q (the int8-out
 * writer). Returns 0, or <0 (incl. the generator's -1/-2 when the tile overflows the CBUF —
 * the tiler shrinks and retries). */
static int conv2d_int8_one_job(int fd, rocket_conv_ctx *ctx,
                               int IC, int IH, int IW, int OC, int OH, int OW,
                               int KH, int KW, int sy, int sx, int pt, int pl,
                               int dy, int dx,
                               const int8_t *in, const int8_t *W, void *out, const i8q_t *q)
{
    int ret = -1;   /* the early-error gotos (BO alloc / 32-bit IOVA) return via this */

    /* No device (fd<0): compute on the int64->int32 oracle, so the tiling
     * decomposition is x86-verifiable against the whole-image oracle (the per-tile NPU
     * path is the separately HW-validated single job). Reached before any BO is touched. */
    if (fd < 0) {
        if (q) {
            conv2d_int8_q_host(IC, IH, IW, OC, OH, OW, KH, KW, sy, sx, pt, pl, dy, dx,
                               in, W, q, out);
            return 0;
        }
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

    /* the int8-out cube is (ceil(OC/16), OH, OW, 16) bytes; the int32 one (OC/4, OH, OW, 4) */
    const size_t out_bytes = q ? (size_t)((OC + 15) / 16) * 16 * OH * OW
                               : out_elems * sizeof(int32_t);
    size_t in_need  = in_elems  * sizeof(int8_t)  + CBUF_BANK;
    size_t wt_need  = wt_elems  * sizeof(int8_t)  + CBUF_BANK;
    size_t rc_need  = 256 * sizeof(uint64_t);
    size_t out_need = out_bytes + CBUF_BANK;
    size_t bs_need  = (q ? i8q_coeff_bytes(q, OC) : 0) + CBUF_BANK;
    rocket_bo lguard = {0}, lin = {0}, lwt = {0}, lrc = {0}, lout = {0}, lbias = {0};
    rocket_bo *guard  = ctx ? &ctx->guard : &lguard;
    rocket_bo *in_bo  = conv_bo(ctx, ctx ? ctx->in : NULL,  &lin,  &in_need);
    rocket_bo *wt_bo  = conv_bo(ctx, ctx ? ctx->wt : NULL,  &lwt,  &wt_need);
    rocket_bo *rc_bo  = conv_bo(ctx, ctx ? ctx->rc : NULL,  &lrc,  &rc_need);
    rocket_bo *out_bo = conv_bo(ctx, ctx ? ctx->out : NULL, &lout, &out_need);
    rocket_bo *bs_bo  = q ? conv_bo(ctx, ctx ? ctx->bias : NULL, &lbias, &bs_need) : &lbias;
    uint64_t regs[256] = {0};
    int rc = -1;

    if (rocket_bo_ensure32(fd, guard, 4096) < 0 ||                                       /* off IOVA 0 */
        rocket_bo_ensure32(fd, in_bo,  in_need) < 0 ||
        rocket_bo_ensure32(fd, wt_bo,  wt_need) < 0 ||
        rocket_bo_ensure32(fd, rc_bo,  rc_need) < 0 ||
        (q && rocket_bo_ensure32(fd, bs_bo, bs_need) < 0) ||
        rocket_bo_ensure32(fd, out_bo, out_need) < 0) {
        ROCKET_LOGE("rocket_conv2d_int8: BO alloc failed\n");
        goto out;
    }
    const int cp = cprof_on();
    const int cpp = q ? CPP_I8Q : CPP_I8;
    double cpa[CP_N] = {0}, cpt = cp ? cprof_now() : 0.0;

    /* scatter input feature -> NC1HWC2 cube (int8, C2=16): feature_data()'s layout,
     * (ic/16, h, w, ic%16) over IHj rows, sixteen channels a pass. Rows >= IH are 0. */
    rocket_bo_prep(fd, in_bo, 1, 0);
    conv_pack_planes(in_bo->ptr, in_bo->size, in, IC, IH, IW, IHj, 16, 1);
    rocket_bo_fini(fd, in_bo);
    CP_MARK(cp, cpa, CP_IN, cpt);

    /* scatter weights -> int8 conv weight cube (oc-group 32 / ic-group 32) */
    rocket_bo_prep(fd, wt_bo, 1, 0);
    memset(wt_bo->ptr, 0, wt_bo->size);
    {
        int8_t *dst = wt_bo->ptr;
        ROCKET_CONV_WT_SCATTER(int8_t, 32, dst, W, OC, IC, KH, KW);
    }
    rocket_bo_fini(fd, wt_bo);
    if (q) {                                   /* the folded per-OC bias, read by BRDMA */
        rocket_bo_prep(fd, bs_bo, 1, 0);
        i8q_coeff_fill(bs_bo->ptr, q, q->A, q->C, OC);
        rocket_bo_fini(fd, bs_bo);
    }
    CP_MARK(cp, cpa, CP_WT, cpt);

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
        if (q) {
            /* the generator takes Mesa's uint8-equivalent zero points (see the depthwise
             * job): the pad lands on in_zp, the offset on out_zp, CPEND on -w_zp */
            p.int8_out = 1;
            p.in_scale = q->in_scale; p.w_scale = q->w_scale; p.out_scale = q->out_scale;
            p.input_zero_point = q->in_zp + 0x80;
            p.output_zero_point = q->out_zp + 0x80;
            p.weight_zero_point = q->w_zp + 0x80;
            p.bias_dma = (uint32_t)bs_bo->dma_address;
            p.bs_mul_perc = q->C ? 1 : 0;
            p.bs_mul_shift = q->C ? (uint8_t)q->bs_shift : 0;
        }
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
        CP_MARK(cp, cpa, CP_GEN, cpt);

        /* zero + hand the output BO to the device */
        rocket_bo_prep(fd, out_bo, 1, 0);
        memset(out_bo->ptr, 0, out_bo->size);
        rocket_bo_fini(fd, out_bo);
        CP_MARK(cp, cpa, CP_OUTZ, cpt);

        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo->dma_address,
                                  .regcmd_count = p.task_count };
        uint32_t in_h[]  = { in_bo->handle, wt_bo->handle, rc_bo->handle, bs_bo->handle };
        uint32_t out_h[] = { out_bo->handle };
        ret = rocket_submit_tasks(fd, &task, 1, in_h, q ? 4 : 3, out_h, 1);
        if (ret) { ROCKET_LOGE("rocket_conv2d_int8: submit failed (%d)\n", ret); goto out; }
        CP_MARK(cp, cpa, CP_SUBMIT, cpt);
    }

    /* read back: de-scatter the int32 output cube (OC/4, OH, OW, 4) */
    ret = rocket_bo_prep(fd, out_bo, 0, 2000000000ULL);   /* 2s wait */
    if (ret) { ROCKET_LOGE("rocket_conv2d_int8: wait timeout (%d)\n", ret); goto out; }
    CP_MARK(cp, cpa, CP_WAIT, cpt);
    if (q) {
        /* the int8 output cube, (oc/16, h, w, oc%16), sixteen channels a pass */
        const uint8_t *src = out_bo->ptr;
        const size_t plane = (size_t)OH * OW * 16, hw = (size_t)OH * OW;
        for (int g = 0; g < (OC + 15) / 16; g++) {
            const int nc = OC - g * 16 < 16 ? OC - g * 16 : 16;
            uint8_t *pl8[16];
            for (int j = 0; j < nc; j++) pl8[j] = (uint8_t *)out + (size_t)(g * 16 + j) * hw;
            rc_group16_to_planes_u8(pl8, nc, src + (size_t)g * plane, hw);
        }
    } else {
        /* the int32 output cube, (oc/4, h, w, oc%4), four channels a pass */
        const int32_t *src = out_bo->ptr;
        const size_t plane = (size_t)OH * OW * 4, hw = (size_t)OH * OW;
        for (int g = 0; g < (OC + 3) / 4; g++) {
            const int nc = OC - g * 4 < 4 ? OC - g * 4 : 4;
            int32_t *pl[4];
            for (int j = 0; j < nc; j++) pl[j] = (int32_t *)out + (size_t)(g * 4 + j) * hw;
            rc_group4_to_planes_i32(pl, nc, src + (size_t)g * plane, hw);
        }
    }
    CP_MARK(cp, cpa, CP_READ, cpt);
    rocket_bo_fini(fd, out_bo);
    CP_MARK(cp, cpa, CP_FINI, cpt);
    if (cp) {
        cprof_add(cpp, cpa, 1);
        cprof_bo(cpp, in_bo->size, in_elems, out_bo->size, out_bytes);
    }
    rc = 0;

out:
    if (!ctx) {
        rocket_bo_free(fd, bs_bo);
        rocket_bo_free(fd, out_bo);
        rocket_bo_free(fd, rc_bo);
        rocket_bo_free(fd, wt_bo);
        rocket_bo_free(fd, in_bo);
        rocket_bo_free(fd, guard);
    }
    return rc ? (ret ? ret : -1) : 0;
}

/* A tile's sub-input before its real data is copied in: what a tap off the image reads.
 * The int32-raw form reads 0 (its caller folds the input zero point into its own
 * requant). The int8-out form reads in_zp on the REAL channels, as the CNA's own pad would,
 * and 0 on the channel padding: CPEND sums every programmed channel of the window, and a
 * padded channel must add nothing to that sum [HW sweep, RK1, tests/conv_i8out_probe: the
 * CNA pads the channel padding with in_zp as well, so a program with a CNA pad and a padded
 * IC carries -w_zp * in_zp per padded tap on its border outputs]. */
static void conv_i8_halo_fill(int8_t *sub_in, int IC, int ICr, size_t plane, const i8q_t *q)
{
    if (q && q->in_zp) {
        memset(sub_in, q->in_zp, (size_t)ICr * plane);
        memset(sub_in + (size_t)ICr * plane, 0, (size_t)(IC - ICr) * plane);
    } else {
        memset(sub_in, 0, (size_t)IC * plane);
    }
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
                                void *out, int oc0, int OCn, int r0, int rh,
                                int c0, int cw, int8_t *sub_in, void *sub_out, const i8q_t *q)
{
    int ih_sub = (rh - 1) * sy + (KH - 1) * dy + 1;
    int gh0    = r0 * sy - pt;
    int iw_sub = (cw - 1) * sx + (KW - 1) * dx + 1;
    int gw0    = c0 * sx - pl;

    const int cp = cprof_on();
    double cpa[CP_N] = {0}, cpt = cp ? cprof_now() : 0.0;
    /* materialize the tile's sub-input: real halo + zero pad, with the channel pad
     * (ic>=ICr stays zero from the memset). */
    conv_i8_halo_fill(sub_in, IC, ICr, (size_t)ih_sub * iw_sub, q);
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
    CP_MARK(cp, cpa, CP_IN, cpt);
    i8q_t qt;
    if (q) { qt = *q; qt.A = q->A + oc0; if (q->C) qt.C = q->C + oc0; }
    int ret = conv2d_int8_one_job(fd, ctx, IC, ih_sub, iw_sub, OCn, rh, cw, KH, KW,
                                  sy, sx, 0, 0, dy, dx, sub_in, Wslice, sub_out, q ? &qt : NULL);
    if (ret) return ret;
    if (cp) cpt = cprof_now();

    const size_t es = q ? 1 : sizeof(int32_t);
    for (int oc = 0; oc < OCn; oc++)
        for (int r = 0; r < rh; r++)
            memcpy((uint8_t *)out + ((((size_t)(oc0 + oc) * OH) + (r0 + r)) * OW + c0) * es,
                   (const uint8_t *)sub_out + (((size_t)oc * rh + r) * cw) * es,
                   (size_t)cw * es);
    CP_MARK(cp, cpa, CP_READ, cpt);
    if (cp) cprof_add(q ? CPP_I8Q : CPP_I8, cpa, 0);   /* the job counted itself */
    return 0;
}

/* an independent tile of the OC/OH/OW decomposition */
typedef struct {
    int oc0, OCn, r0, rh, c0, cw;
    const i8q_t *q;      /* the tile's own int8-out form (a per-channel job's), or NULL: the run's */
} i8tile;

/* Batch tiles[t0..t0+tn) into ONE multi-task int8 conv job on fd/ctx (lever-1, gapped;
 * see CONV_RC_STRIDE). ctx holds the (grown) batched input/weight/regcmd/output BOs; each
 * tile lands at a bank-aligned, zeroed slot so its materialize/scatter/gen/de-scatter is
 * byte-for-byte conv2d_int8_one_tile + _one_job — only the per-tile submit+fence is
 * coalesced into one. The whole-BO zero (gaps + the per-tile slack bank) keeps the int8
 * feature-DMA +1-bank over-read reading zeros, exactly as the standalone job. The job's
 * tasks are separate HW kicks, bit-identical to the per-tile path. 0 / <0. */
static int conv2d_int8_batch_tiles(int fd, rocket_conv_ctx *ctx,
        int IC, int ICr, int IH, int IW, int OH, int OW, int KH, int KW,
        int sy, int sx, int pt, int pl, int dy, int dx,
        const int8_t *in, const int8_t *Wuse, void *out,
        const i8tile *tiles, int t0, int tn, const i8q_t *q)
{
    if (tn <= 0) return 0;

    size_t *off_in  = malloc((size_t)tn * sizeof(size_t));
    size_t *off_wt  = malloc((size_t)tn * sizeof(size_t));
    size_t *off_out = malloc((size_t)tn * sizeof(size_t));
    size_t *off_bs  = malloc((size_t)tn * sizeof(size_t));
    rocket_task_desc *tasks = malloc((size_t)tn * sizeof(rocket_task_desc));
    int8_t *sub_in = NULL; void *scratch = NULL;
    if (!off_in || !off_wt || !off_out || !off_bs || !tasks) { goto oom; }

    /* Pass 1: per-tile bank-aligned offsets + the largest row-major sub-input scratch. */
    size_t in_tot = 0, wt_tot = 0, out_tot = 0, bs_tot = 0, max_sub_in = 0;
    for (int k = 0; k < tn; k++) {
        const i8tile *tl = &tiles[t0 + k];
        int ih_sub = (tl->rh - 1) * sy + (KH - 1) * dy + 1;
        int iw_sub = (tl->cw - 1) * sx + (KW - 1) * dx + 1;
        int IHj = ih_sub < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : ih_sub;
        size_t sin = (size_t)IC * ih_sub * iw_sub;
        if (sin > max_sub_in) max_sub_in = sin;
        off_in[k]  = in_tot;   in_tot  += bank_round((size_t)IC * IHj * iw_sub) + CBUF_BANK;
        off_wt[k]  = wt_tot;   wt_tot  += bank_round((size_t)tl->OCn * IC * KH * KW) + CBUF_BANK;
        off_out[k] = out_tot;  out_tot += bank_round(q ? (size_t)((tl->OCn + 15) / 16) * 16 * tl->rh * tl->cw
                                                        : (size_t)tl->OCn * tl->rh * tl->cw * sizeof(int32_t))
                                          + CBUF_BANK;
        off_bs[k]  = bs_tot;   bs_tot  += q ? ((i8q_coeff_bytes(tl->q ? tl->q : q, tl->OCn) + 63) & ~(size_t)63) : 0;
    }
    sub_in = malloc(max_sub_in ? max_sub_in : 1);
    if (!sub_in) goto oom;

    int ret = -1;
    size_t in_need = in_tot, wt_need = wt_tot, out_need = out_tot, bs_need = bs_tot + CBUF_BANK;
    size_t rc_need = (size_t)tn * CONV_RC_STRIDE * sizeof(uint64_t);
    rocket_bo *guard  = &ctx->guard;
    rocket_bo *in_bo  = conv_bo(ctx, ctx->in,  NULL, &in_need);
    rocket_bo *wt_bo  = conv_bo(ctx, ctx->wt,  NULL, &wt_need);
    rocket_bo *rc_bo  = conv_bo(ctx, ctx->rc,  NULL, &rc_need);
    rocket_bo *out_bo = conv_bo(ctx, ctx->out, NULL, &out_need);
    rocket_bo *bs_bo  = q ? conv_bo(ctx, ctx->bias, NULL, &bs_need) : NULL;
    if (rocket_bo_ensure32(fd, guard, 4096) < 0 ||
        rocket_bo_ensure32(fd, in_bo,  in_need) < 0  ||
        rocket_bo_ensure32(fd, wt_bo,  wt_need) < 0  ||
        rocket_bo_ensure32(fd, rc_bo,  rc_need) < 0 ||
        (q && rocket_bo_ensure32(fd, bs_bo, bs_need) < 0) ||
        rocket_bo_ensure32(fd, out_bo, out_need) < 0) {
        ROCKET_LOGE("conv int8 batch: BO alloc failed\n"); goto out;
    }
    scratch = malloc(rocket_submit_scratch_size((uint32_t)tn));
    if (!scratch) { ROCKET_LOGE("conv int8 batch: scratch alloc failed\n"); goto out; }
    const int cp = cprof_on();
    const int cpp = q ? CPP_I8Q : CPP_I8;
    double cpa[CP_N] = {0}, cpt = cp ? cprof_now() : 0.0;

    /* input cubes: materialize each tile's halo/pad sub-input, scatter to its NC1HWC2 slot.
     * Each slot is written whole, its slack included, so the BO is not zeroed first. */
    rocket_bo_prep(fd, in_bo, 1, 0);
    for (int k = 0; k < tn; k++) {
        const i8tile *tl = &tiles[t0 + k];
        int ih_sub = (tl->rh - 1) * sy + (KH - 1) * dy + 1;
        int iw_sub = (tl->cw - 1) * sx + (KW - 1) * dx + 1;
        int IHj = ih_sub < CONV_MIN_DATAIN_H ? CONV_MIN_DATAIN_H : ih_sub;
        int gh0 = tl->r0 * sy - pt, gw0 = tl->c0 * sx - pl;
        conv_i8_halo_fill(sub_in, IC, ICr, (size_t)ih_sub * iw_sub, q);
        for (int ic = 0; ic < ICr; ic++)
            for (int j = 0; j < ih_sub; j++) {
                int gih = gh0 + j;
                if (gih < 0 || gih >= IH) continue;
                int cs = gw0 < 0 ? 0 : gw0, ce = gw0 + iw_sub; if (ce > IW) ce = IW;
                if (cs >= ce) continue;
                memcpy(&sub_in[((size_t)ic * ih_sub + j) * iw_sub + (cs - gw0)],
                       &in[((size_t)ic * IH + gih) * IW + cs], (size_t)(ce - cs));
            }
        const size_t slot = (k + 1 < tn ? off_in[k + 1] : in_tot) - off_in[k];
        conv_pack_planes((int8_t *)in_bo->ptr + off_in[k], slot, sub_in, IC, ih_sub, iw_sub,
                         IHj, 16, 1);
    }
    rocket_bo_fini(fd, in_bo);
    CP_MARK(cp, cpa, CP_IN, cpt);

    /* weight cubes: each tile's OC-slice */
    rocket_bo_prep(fd, wt_bo, 1, 0);
    memset(wt_bo->ptr, 0, wt_bo->size);
    for (int k = 0; k < tn; k++) {
        const i8tile *tl = &tiles[t0 + k];
        const int8_t *Wslice = Wuse + (size_t)tl->oc0 * IC * KH * KW;
        int8_t *dst = (int8_t *)wt_bo->ptr + off_wt[k];
        ROCKET_CONV_WT_SCATTER(int8_t, 32, dst, Wslice, tl->OCn, IC, KH, KW);
    }
    rocket_bo_fini(fd, wt_bo);
    if (q) {                                   /* each tile's slice of the folded bias */
        rocket_bo_prep(fd, bs_bo, 1, 0);
        for (int k = 0; k < tn; k++) {
            const i8tile *tl = &tiles[t0 + k];
            const i8q_t *tq = tl->q ? tl->q : q;
            i8q_coeff_fill((uint8_t *)bs_bo->ptr + off_bs[k], tq, tq->A + tl->oc0,
                           tq->C ? tq->C + tl->oc0 : NULL, tl->OCn);
        }
        rocket_bo_fini(fd, bs_bo);
    }
    CP_MARK(cp, cpa, CP_WT, cpt);

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
        if (q) {
            const i8q_t *tq = tl->q ? tl->q : q;
            p.int8_out = 1;
            p.in_scale = tq->in_scale; p.w_scale = tq->w_scale; p.out_scale = tq->out_scale;
            p.input_zero_point = tq->in_zp + 0x80;
            p.output_zero_point = tq->out_zp + 0x80;
            p.weight_zero_point = tq->w_zp + 0x80;
            p.bias_dma = (uint32_t)(bs_bo->dma_address + off_bs[k]);
            p.bs_mul_perc = tq->C ? 1 : 0;
            p.bs_mul_shift = tq->C ? (uint8_t)tq->bs_shift : 0;
        }
        if ((ret = gen_conv2d_int8(&p)) != 0) { ROCKET_LOGE("conv int8 batch: gen %d\n", ret); goto out; }
        if (p.task_count > CONV_RC_STRIDE) {
            ROCKET_LOGE("conv int8 batch: regcmd overflow (%u > %d)\n", p.task_count, CONV_RC_STRIDE);
            ret = -1; goto out;
        }
        if (rkt_chain_pack(0, rc_bo, tasks, k, regs, p.task_count, CONV_RC_STRIDE) != 0) {
            ret = -1; goto out;
        }
    }
    rocket_bo_fini(fd, rc_bo);
    CP_MARK(cp, cpa, CP_GEN, cpt);

    rocket_bo_prep(fd, out_bo, 1, 0);
    memset(out_bo->ptr, 0, out_bo->size);
    rocket_bo_fini(fd, out_bo);
    CP_MARK(cp, cpa, CP_OUTZ, cpt);

    {
        uint32_t in_h[]  = { in_bo->handle, wt_bo->handle, rc_bo->handle, q ? bs_bo->handle : 0 };
        uint32_t out_h[] = { out_bo->handle };
        ret = rocket_submit_tasks_pre(fd, scratch, tasks, (uint32_t)tn, in_h, q ? 4 : 3, out_h, 1, 0);
        if (ret) { ROCKET_LOGE("conv int8 batch: submit %d\n", ret); goto out; }
    }
    CP_MARK(cp, cpa, CP_SUBMIT, cpt);
    ret = rocket_bo_prep(fd, out_bo, 0, 2000000000ULL);   /* ONE wait for the whole job */
    if (ret) { ROCKET_LOGE("conv int8 batch: wait %d\n", ret); goto out; }
    CP_MARK(cp, cpa, CP_WAIT, cpt);

    for (int k = 0; k < tn && q; k++) {                   /* de-scatter every tile's int8 output */
        const i8tile *tl = &tiles[t0 + k];
        const uint8_t *src = (const uint8_t *)out_bo->ptr + off_out[k];
        const size_t plane = (size_t)tl->rh * tl->cw * 16;
        for (int g = 0; g < (tl->OCn + 15) / 16; g++) { /* sixteen channels a pass, a row at a time */
            const int nc = tl->OCn - g * 16 < 16 ? tl->OCn - g * 16 : 16;
            for (int oh = 0; oh < tl->rh; oh++) {
                uint8_t *pl8[16];
                for (int j = 0; j < nc; j++)
                    pl8[j] = (uint8_t *)out + (((size_t)(tl->oc0 + g * 16 + j) * OH) + (tl->r0 + oh)) * OW + tl->c0;
                rc_group16_to_planes_u8(pl8, nc, src + (size_t)g * plane + (size_t)oh * tl->cw * 16,
                                        (size_t)tl->cw);
            }
        }
    }
    for (int k = 0; k < tn && !q; k++) {                  /* de-scatter every tile's output */
        const i8tile *tl = &tiles[t0 + k];
        const int32_t *src = (const int32_t *)((const int8_t *)out_bo->ptr + off_out[k]);
        const size_t plane = (size_t)tl->rh * tl->cw * 4;
        for (int g = 0; g < (tl->OCn + 3) / 4; g++) { /* four channels a pass, a row at a time */
            const int nc = tl->OCn - g * 4 < 4 ? tl->OCn - g * 4 : 4;
            for (int oh = 0; oh < tl->rh; oh++) {
                int32_t *pl[4];
                for (int j = 0; j < nc; j++)
                    pl[j] = (int32_t *)out + (((size_t)(tl->oc0 + g * 4 + j) * OH) + (tl->r0 + oh)) * OW + tl->c0;
                rc_group4_to_planes_i32(pl, nc, src + (size_t)g * plane + (size_t)oh * tl->cw * 4,
                                        (size_t)tl->cw);
            }
        }
    }
    CP_MARK(cp, cpa, CP_READ, cpt);
    rocket_bo_fini(fd, out_bo);
    CP_MARK(cp, cpa, CP_FINI, cpt);
    if (cp) {
        cprof_add(cpp, cpa, tn);
        cprof_bo(cpp, in_bo->size, in_tot, out_bo->size, out_tot);
    }
    ret = 0;
out:
    free(scratch); free(sub_in); free(off_in); free(off_wt); free(off_out); free(off_bs); free(tasks);
    return ret;
oom:
    free(scratch); free(sub_in); free(off_in); free(off_wt); free(off_out); free(off_bs); free(tasks);
    return -1;
}

/* one multicore worker: strided slice of the tile list on its own fd/ctx + scratch */
typedef struct {
    int fd; rocket_conv_ctx *ctx;
    int IC, ICr, IH, IW, OH, OW, KH, KW, sy, sx, pt, pl, dy, dx;
    const int8_t *in; const int8_t *Wuse; void *out;
    const i8tile *tiles; int ntiles; int wstart, wstride;
    size_t sub_in_sz, sub_out_elems;
    int idx; int ret;
    int batch, wcount;                           /* batch: one job over the [wstart,wstart+wcount) run */
    int core_base;                               /* big-core rotation base inherited from caller thread */
    const i8q_t *q;                              /* the int8-out form, or NULL */
} i8w_arg;

static void *i8_worker(void *a)
{
    i8w_arg *w = (i8w_arg *)a;
    rocket_pin_worker_based(w->idx, w->core_base); /* keep the materialize/scatter off the A55s */
    if (w->batch) {                              /* coalesce this worker's contiguous run into one job */
        w->ret = conv2d_int8_batch_tiles(w->fd, w->ctx, w->IC, w->ICr, w->IH, w->IW, w->OH,
                                         w->OW, w->KH, w->KW, w->sy, w->sx, w->pt, w->pl,
                                         w->dy, w->dx, w->in, w->Wuse, w->out,
                                         w->tiles, w->wstart, w->wcount, w->q);
        return NULL;
    }
    int8_t  *sub_in  = malloc(w->sub_in_sz);
    void    *sub_out = malloc(w->sub_out_elems * sizeof(int32_t));
    if (!sub_in || !sub_out) { free(sub_in); free(sub_out); w->ret = -1; return NULL; }
    int ret = 0;
    for (int t = w->wstart; t < w->ntiles && !ret; t += w->wstride) {
        const i8tile *tl = &w->tiles[t];
        ret = conv2d_int8_one_tile(w->fd, w->ctx, w->IC, w->ICr, w->IH, w->IW, w->OH,
                                   w->OW, w->KH, w->KW, w->sy, w->sx, w->pt, w->pl,
                                   w->dy, w->dx, w->in, w->Wuse, w->out,
                                   tl->oc0, tl->OCn, tl->r0, tl->rh, tl->c0, tl->cw,
                                   sub_in, sub_out, tl->q ? tl->q : w->q);
    }
    free(sub_in); free(sub_out);
    w->ret = ret;
    return NULL;
}

/* The direct int8 conv's plan, shared by the int32-raw and the int8-out forms so the two
 * cannot disagree about a tile: they run the same CNA program and differ only past the BS
 * stage. d->oc must be a multiple of 32 (the run pads it). Pure: opens nothing.
 *
 * Tiling over OC (mult-32) + OH-rows + OW-cols exactly like conv2d_run. Each tile is an
 * independent HW-validated single job; OH-band tiles MATERIALIZE their edge padding (the
 * CNA only has symmetric pad), and the min_ihs floor keeps every tile inside the
 * HW-validated feature-DMA envelope (the fp16 datain_height<4 lesson — the int8 path
 * inherits the same geometry). Returns 0, or -1 (bad parameters), -3 (empty output), -4
 * (no tile fits the CBUF or the CNA's fields). */
typedef struct {
    int IC;              /* input channels as programmed: the real count rounded up to 32 */
    int fast;            /* 1: the whole problem is ONE job at the caller's pad */
    int OCt, rht, cwt;   /* the tile: output channels, rows, columns (when !fast) */
} i8plan;

static int conv2d_int8_plan(const rocket_conv2d_desc *d, i8plan *pl)
{
    const int ICr = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int sy = d->stride_y, sx = d->stride_x, dy = d->dil_y, dx = d->dil_x;
    if (OH <= 0 || OW <= 0) return -3;
    if (OC <= 0 || KH <= 0 || KW <= 0 || IH <= 0 || IW <= 0 || ICr <= 0) return -1;
    if (sy <= 0 || sx <= 0 || dy <= 0 || dx <= 0) return -1;
    if (OC % 32) return -1;
    memset(pl, 0, sizeof(*pl));

    /* First-layer IC<32 (RGB): zero-pad input channels up to the ic group of 32 (the
     * padded channels contribute 0). */
    const int IC = ((ICr + 31) / 32) * 32;
    pl->IC = IC;
    /* a single OC kernel must fit one CBUF bank (int8: 1 B/elem) */
    if ((size_t)KW * KH * IC * sizeof(int8_t) > CBUF_BANK) return -4;

    /* fast path: no channel pad AND the whole problem fits one CBUF pass. (int8 weight
     * tile <= 4 banks, feature <= the 8-bank feature budget — both in int8 bytes.) */
    {
        long ih_full = (long)(OH - 1) * sy + (long)(KH - 1) * dy + 1;
        long iw_full = (long)(OW - 1) * sx + (long)(KW - 1) * dx + 1;
        if (IC == ICr && IH <= CNA_TILE_MAX_H && IW <= CNA_TILE_MAX_W &&
            cna_pad_fits(d->pad_top, d->pad_left) &&
            (size_t)OC * IC * KH * KW * sizeof(int8_t) <= (size_t)4 * CBUF_BANK &&
            (size_t)IC * job_rows(ih_full) * iw_full * sizeof(int8_t) <= CONV_FEAT_BUDGET_I8 &&
            (size_t)IC * job_rows(IH) * IW * sizeof(int8_t) <= CONV_FEAT_BUDGET_I8) {
            pl->fast = 1;
            return 0;
        }
    }

    /* feature bytes for a (rh out-rows x cw out-cols) tile's materialized sub-input (int8) */
    #define CONV_FEAT_I8(rh, cw) ((size_t)IC * (size_t)job_rows(datain_h(rh, sy, KH, dy)) * \
        (size_t)datain_w(cw, sx, KW, dx) * sizeof(int8_t))

    const int min_ihs = conv_min_ihs();   /* never below the HW row floor */

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
    if (wbanks > (size_t)(CBUF_BANKS - 2)) return -4;   /* leave >=1 feature bank + the slack */
    /* Reserve ONE feature slack bank (gen_conv2d_int8_fill sets data_bank = fd_banks+1):
     * feature_banks + 1 + weight_banks <= 12.  */
    size_t feat_budget = (size_t)(CBUF_BANKS - 1 - wbanks) * CBUF_BANK;
    if (feat_budget > CONV_FEAT_BUDGET_I8) feat_budget = CONV_FEAT_BUDGET_I8;

    int rht_lo = 1;
    while (rht_lo < OH && datain_h(rht_lo, sy, KH, dy) < min_ihs) rht_lo++;
    int rht = OH, cwt = OW;
    while (rht > rht_lo && datain_h(rht, sy, KH, dy) > CNA_TILE_MAX_H) rht--;   /* CNA fields */
    while (cwt > 1 && datain_w(cwt, sx, KW, dx) > CNA_TILE_MAX_W) cwt--;
    while (rht > rht_lo && CONV_FEAT_I8(rht, cwt) > feat_budget) rht--;
    if (CONV_FEAT_I8(rht, cwt) > feat_budget)          /* full-width band still over -> narrow cols */
        while (cwt > 1 && CONV_FEAT_I8(rht, cwt) > feat_budget) cwt--;
    if (CONV_FEAT_I8(rht, cwt) > feat_budget ||        /* minimal tile still over */
        datain_h(rht, sy, KH, dy) > CNA_TILE_MAX_H || datain_w(cwt, sx, KW, dx) > CNA_TILE_MAX_W)
        return -4;
    #undef CONV_FEAT_I8
    #undef WBANKS_I8
    pl->OCt = OCt; pl->rht = rht; pl->cwt = cwt;
    return 0;
}

/* Run a tile list of the int8 DIRECT conv: across the pool's worker fds (one NPU core
 * each), as one batched job, or one tile at a time. Each tile writes a disjoint region of
 * `out`. Shared by the per-tensor run and the per-channel one, whose tiles carry their own
 * int8-out form. Returns 0 / <0. */
static int conv2d_int8_dispatch(int fd, rocket_conv_ctx *ctx, rocket_conv_pool *pool,
        int IC, int ICr, int IH, int IW, int OH, int OW, int KH, int KW,
        int sy, int sx, int pt, int pl, int dy, int dx,
        const int8_t *in, const int8_t *Wuse, void *out, const i8tile *tiles, int ntiles,
        size_t sub_in_sz, size_t sub_out_elems, const i8q_t *q)
{
    const size_t es = q ? 1 : sizeof(int32_t);
    int ret = 0;
    /* the batched job allocates its BOs on fd, so without a device the tiles run one by
     * one on the host model */
    int batch = fd >= 0 && conv_batch_enabled();
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
                                 sub_in_sz, sub_out_elems, w, 0, batch, wcount, affbase, q };
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
                                      sy, sx, pt, pl, dy, dx, in, Wuse, out, tiles, 0, ntiles, q);
        if (!ctx) conv_ctx_free_bos(&local, fd);
    } else {
        /* serial: one scratch pair on the borrowed ctx/fd (pool==NULL or a single tile) */
        int8_t *sub_in  = malloc(sub_in_sz);
        void   *sub_out = malloc(sub_out_elems * es);
        if (!sub_in || !sub_out) { free(sub_in); free(sub_out); return -1; }
        for (int t = 0; t < ntiles && !ret; t++) {
            const i8tile *tl = &tiles[t];
            ret = conv2d_int8_one_tile(fd, ctx, IC, ICr, IH, IW, OH, OW, KH, KW,
                                       sy, sx, pt, pl, dy, dx, in, Wuse, out,
                                       tl->oc0, tl->OCn, tl->r0, tl->rh, tl->c0, tl->cw,
                                       sub_in, sub_out, tl->q ? tl->q : q);
        }
        free(sub_in); free(sub_out);
    }
    return ret;
}

/* Run the int8 DIRECT conv on the NPU on conv2d_int8_plan's tiles. in/W are int8, out is
 * int32, or int8 with q (the int8-out form; q->A covers d->oc). When pool!=NULL the
 * independent tiles fan out across the pool's worker fds/ctxs (multicore); pool==NULL keeps
 * the serial single-fd path. Returns 0 / <0. */
static int conv2d_int8_run(int fd, rocket_conv_ctx *ctx, rocket_conv_pool *pool,
                           const rocket_conv2d_desc *d,
                           const int8_t *in, const int8_t *W, void *out, const i8q_t *q)
{
    if (d->depthwise) return -2;   /* int8 depthwise is rocket_conv2d_dw_int8 (int8-out) */
    const size_t es = q ? 1 : sizeof(int32_t);

    /* OC%32!=0: pad OC up to the int8 weight oc-group (32), run, slice the real OC off
     * the front. The extra kernels are zero and each direct output channel is
     * independent, so the first OC channels are bit-exact (mirrors the fp16 OC%16 pad,
     * just on the int8 oc-group). The int8-out form pads the folded bias with zeros too.
     * A program whose OC ends part way through a 32-kernel group does not complete on the
     * int8-out writer: the job outlasts the driver's watchdog and that group is left
     * unwritten [HW sweep, RK1, tests/conv_i8out_probe witness]. */
    if (d->oc % 32) {
        const int OC = d->oc, OCp = ((OC + 31) / 32) * 32;
        const int IC = d->ic, KH = d->kh, KW = d->kw;
        const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
        if (OH <= 0 || OW <= 0) return -3;
        rocket_conv2d_desc dp = *d; dp.oc = OCp;
        int8_t  *Wp = calloc((size_t)OCp * IC * KH * KW, 1);
        void    *op = malloc((size_t)OCp * OH * OW * es);
        int32_t *Ap = q ? calloc((size_t)OCp, sizeof(int32_t)) : NULL;
        int16_t *Cp = q && q->C ? calloc((size_t)OCp, sizeof(int16_t)) : NULL;
        if (!Wp || !op || (q && !Ap) || (q && q->C && !Cp)) {
            free(Wp); free(op); free(Ap); free(Cp);
            return -1;
        }
        memcpy(Wp, W, (size_t)OC * IC * KH * KW);                  /* extra kernels stay 0 */
        i8q_t qp;
        if (q) { memcpy(Ap, q->A, (size_t)OC * sizeof(int32_t)); qp = *q; qp.A = Ap; }
        if (Cp) { memcpy(Cp, q->C, (size_t)OC * sizeof(int16_t)); qp.C = Cp; }   /* C 0 past OC */
        int r = conv2d_int8_run(fd, ctx, pool, &dp, in, Wp, op, q ? &qp : NULL);
        if (!r) memcpy(out, op, (size_t)OC * OH * OW * es);          /* first OC chans */
        free(Wp); free(op); free(Ap); free(Cp);
        return r;
    }

    i8plan pln;
    int prc = conv2d_int8_plan(d, &pln);
    if (prc) return prc;

    const int ICr = d->ic, IH = d->ih, IW = d->iw, OC = d->oc, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int sy = d->stride_y, sx = d->stride_x, pt = d->pad_top, pl = d->pad_left;
    const int dy = d->dil_y, dx = d->dil_x;
    const int IC = pln.IC;

    if (pln.fast)
        return conv2d_int8_one_job(fd, ctx, IC, IH, IW, OC, OH, OW, KH, KW,
                                   sy, sx, pt, pl, dy, dx, in, W, out, q);

    /* Padded weight built once; input padded lazily in the per-tile materialization
     * (ic>=ICr -> 0). */
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

    const int OCt = pln.OCt, rht = pln.rht, cwt = pln.cwt;
    const int IHs = (rht - 1) * sy + (KH - 1) * dy + 1;
    const int IWs = (cwt - 1) * sx + (KW - 1) * dx + 1;
    const size_t sub_in_sz     = (size_t)IC * IHs * IWs * sizeof(int8_t);
    const size_t sub_out_elems = (size_t)OCt * rht * cwt;

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
                tiles[ti++] = (i8tile){ oc0, OCn, r0, rh, c0, cw, NULL };
                c0 += cw;
            }
            r0 += rh;
        }
    }
    ntiles = ti;
    free(rband); free(cband);

    int ret = conv2d_int8_dispatch(fd, ctx, pool, IC, ICr, IH, IW, OH, OW, KH, KW, sy, sx,
                                   pt, pl, dy, dx, in, Wuse, out, tiles, ntiles, sub_in_sz,
                                   sub_out_elems, q);
    free(tiles); free(Wpad);
    return ret;
}

int rocket_conv2d_int8(int fd, const rocket_conv2d_desc *d,
                       const int8_t *in, const int8_t *W, int32_t *out)
{
    if (conv_wrong_encoding("rocket_conv2d_int8",
                            "rocket_conv2d_int8_rk3576, which takes the quant "
                            "parameters and writes int8 — this part's direct int8 "
                            "datapath requantizes on chip rather than writing a raw "
                            "int32 accumulator"))
        return ROCKET_E_UNSUPPORTED;
    return conv2d_int8_run(fd, NULL, NULL, d, in, W, out, NULL);
}

/* Multicore: fan the DIRECT conv's independent OC/OH/OW tiles across the pool's
 * worker fds (one NPU core each). Falls back to serial (worker 0) for single-tile
 * convs. Bit-identical to rocket_conv2d_int8 — same tiles, same single jobs. */
int rocket_conv2d_int8_mt(rocket_conv_pool *pool, const rocket_conv2d_desc *d,
                          const int8_t *in, const int8_t *W, int32_t *out)
{
    if (!pool || pool->n < 1) return -1;
    return conv2d_int8_run(pool->fd[0], pool->ctx[0], pool, d, in, W, out, NULL);
}

int rocket_conv2d_int8_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                           const int8_t *in, const int8_t *W, int32_t *out)
{
    if (!ctx) return -1;
    return conv2d_int8_run(ctx->fd, ctx, NULL, d, in, W, out, NULL);  /* ctx->fd may be <0 -> oracle */
}

/* ---- the int8-OUT direct conv (on-chip requant), see rocket_conv.h ----------------- */

/* The shape check behind rocket_conv2d_int8_q: the int32-raw entry's own planner at the
 * padded OC, so the two entries accept the same shapes. */
int rocket_conv2d_int8_q_plan(const rocket_conv2d_desc *d)
{
    if (!d || d->depthwise) return ROCKET_E_SHAPE;
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (!hw || !hw->name || strcmp(hw->name, "rk3588")) return ROCKET_E_UNSUPPORTED;
    rocket_conv2d_desc dp = *d;
    dp.oc = ((d->oc + 31) / 32) * 32;
    i8plan pl;
    return conv2d_int8_plan(&dp, &pl) ? ROCKET_E_SHAPE : ROCKET_OK;
}

static int conv2d_int8_q_entry(int fd, rocket_conv_ctx *ctx, rocket_conv_pool *pool,
                               const rocket_conv2d_desc *d, const int8_t *in, const int8_t *W,
                               const int32_t *bias, float in_scale, float w_scale,
                               float out_scale, int in_zp, int w_zp, int out_zp, int8_t *out)
{
    if (!d || d->depthwise || !in || !W || !out) return ROCKET_E_SHAPE;
    if (conv_wrong_encoding("rocket_conv2d_int8_q",
                            "rocket_conv2d_int8_rk3576, this part's int8-out direct conv"))
        return ROCKET_E_UNSUPPORTED;
    /* The tensors are int8, so a zero point outside that range is not a TFLite int8
     * tensor; a uint8 one maps on by subtracting 128 from its values and zero points. */
    if (in_zp < -128 || in_zp > 127 || w_zp < -128 || w_zp > 127 || out_zp < -128 || out_zp > 127) {
        ROCKET_LOGE("rocket_conv2d_int8_q: a zero point (%d, %d, %d) is outside int8\n",
                    in_zp, w_zp, out_zp);
        return ROCKET_E_UNSUPPORTED;
    }
    /* The OUT_CVT carries the scale as a 15-bit multiplier and a 6-bit shift, 141 minus the
     * scale's biased exponent (the generator's derivation): a scale whose shift leaves that
     * field is not programmable. */
    {
        const float cs = (in_scale * w_scale) / out_scale;
        union { float f; uint32_t u; } cv = { .f = cs };
        const int shift_reg = 141 - (int)((cv.u >> 23) & 0xFF);
        if (!(cs > 0.0f) || cs != cs || shift_reg < 1 || shift_reg > 63) {
            ROCKET_LOGE("rocket_conv2d_int8_q: requant scale %g is outside the OUT_CVT's range\n",
                        (double)cs);
            return ROCKET_E_UNSUPPORTED;
        }
    }
    const int OC = d->oc, IC = d->ic, KH = d->kh, KW = d->kw;
    if (OC <= 0 || IC <= 0 || KH <= 0 || KW <= 0) return ROCKET_E_SHAPE;
    int32_t *A = malloc((size_t)OC * sizeof(int32_t));
    if (!A) return ROCKET_E_NOMEM;
    /* The input zero point folds into the bias: bias - in_zp * sum(w - w_zp). A padded tap
     * reads in_zp (the CNA's pad, or a tile's materialized halo) and the fold takes it back
     * out, so it contributes nothing, as TFLite's padding does. The weight zero point adds
     * -w_zp * sum(x) per window, which depends on the input; CPEND adds it on chip. */
    const size_t K = (size_t)IC * KH * KW;
    for (int oc = 0; oc < OC; oc++) {
        int64_t sw = 0;
        const int8_t *wk = W + (size_t)oc * K;
        for (size_t k = 0; k < K; k++) sw += wk[k] - w_zp;
        const int64_t a = (int64_t)(bias ? bias[oc] : 0) - (int64_t)in_zp * sw;
        if (a > INT32_MAX || a < INT32_MIN) {
            ROCKET_LOGE("rocket_conv2d_int8_q: folded bias of channel %d leaves int32\n", oc);
            free(A);
            return ROCKET_E_UNSUPPORTED;
        }
        A[oc] = (int32_t)a;
    }
    const i8q_t q = { A, in_scale, w_scale, out_scale, in_zp, w_zp, out_zp, NULL, 0 };
    int ret = conv2d_int8_run(fd, ctx, pool, d, in, W, out, &q);
    free(A);
    return ret;
}

int rocket_conv2d_int8_q(int fd, const rocket_conv2d_desc *d,
                         const int8_t *in, const int8_t *W, const int32_t *bias,
                         float in_scale, float w_scale, float out_scale,
                         int in_zp, int w_zp, int out_zp, int8_t *out)
{
    return conv2d_int8_q_entry(fd, NULL, NULL, d, in, W, bias, in_scale, w_scale, out_scale,
                               in_zp, w_zp, out_zp, out);
}

int rocket_conv2d_int8_q_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                             const int8_t *in, const int8_t *W, const int32_t *bias,
                             float in_scale, float w_scale, float out_scale,
                             int in_zp, int w_zp, int out_zp, int8_t *out)
{
    if (!ctx) return ROCKET_E_SHAPE;
    return conv2d_int8_q_entry(ctx->fd, ctx, NULL, d, in, W, bias, in_scale, w_scale,
                               out_scale, in_zp, w_zp, out_zp, out);
}

int rocket_conv2d_int8_q_mt(rocket_conv_pool *pool, const rocket_conv2d_desc *d,
                            const int8_t *in, const int8_t *W, const int32_t *bias,
                            float in_scale, float w_scale, float out_scale,
                            int in_zp, int w_zp, int out_zp, int8_t *out)
{
    if (!pool || pool->n < 1) return ROCKET_E_SHAPE;
    return conv2d_int8_q_entry(pool->fd[0], pool->ctx[0], pool, d, in, W, bias, in_scale,
                               w_scale, out_scale, in_zp, w_zp, out_zp, out);
}

/* ############################################################################
 * PART 6 — Native int8 DEPTHWISE conv (int8-out, on-chip requant)
 * ##########################################################################*/

/* =========================================================================
 * Native int8 DEPTHWISE CONV_2D (int8-OUT, on-chip requant). The register program is
 * Mesa's int8-output depthwise writer (gen_conv2d_dw_int8 int8_out=1, which replay_dw_mesa
 * checks against a Teflon capture). PER-TENSOR quant, symmetric weights (w_zp == 0).
 *
 * Mesa's rocket driver is a uint8 driver: its formulas take a uint8 tensor u with zero
 * point Z and program the cube as u - 0x80, the pad as Z - 0x80, the output offset as
 * Z_out - 0x80, and fold (Z_in - 0x80) * sum(u_w - Z_w) into the bias. An int8 tensor is
 * the uint8 tensor u = x + 0x80, Z = zp + 0x80 at the same scale, so for int8 data the
 * cubes hold the raw int8 values, the generator takes zp + 0x80 for the pad and the output
 * offset, the fold is in_zp * sum(w), and the output reads back raw. Handing the int8
 * BYTES to the uint8 formulas instead (the byte flip x ^ 0x80) is not linear in x; it
 * computes a different function, and a Teflon capture of an int8 model records that
 * function, not the model's. G = 64 (Mesa's int8 DW group). The CPU oracle (fd<0)
 * computes the exact int8 conv and TFLite's requant.
 * ========================================================================= */
#define DW_INT8_G 64

/* CPU reference for the int8 DW int8-out path (fd<0 fallback + a self-check oracle):
 * the exact TFLite int8 depthwise — acc = Σ(in_q - in_zp)*(w_q - w_zp) + bias_q, then
 * requant real = in_scale*w_scale*acc; q = clamp(lrintf(real/out_scale)+out_zp). The NPU
 * requants through the OUT_CVT multiplier and shift instead, so the two can differ by one
 * where a value sits at a rounding boundary; the accumulator is the same. */
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

/* Run ONE int8 DW int8-out job (single CBUF pass). C channels, one int8 filter per channel,
 * per-tensor quant. in [C][IH][IW] int8, w [C][KH][KW] int8, bias_q [C] int32 (may be NULL),
 * out [C][OH][OW] int8 (model domain). Returns 0 / <0.
 *
 * THE JOB IS PROGRAMMED AT WHOLE G-CHANNEL GROUPS. A program whose channel count ends part
 * way through a 64-channel group never completes: the driver retires it at its 500 ms
 * watchdog, the fence signals as if it had, and the partial group's channels are left
 * UNWRITTEN. At C 96, 80 and 144 every element of the partial group read back as the output
 * BO's zero fill, one job timeout per shape, and the entry returned 0 [HW sweep, RK1,
 * 2026-09-27, tests/conv_dw_int8_runtime]. So the cubes and the program carry Cj, C rounded
 * up to G, with zero input, weight and bias in the padding, and only the C real channels
 * are read back. Programmed that way the jobs complete. */
/* The per-channel epilogue of one int8-DW job: the BS stage multiplies channel c's
 * accumulator by cmul[c] and shifts by `shift` (both sign fields), then the OUT_CVT applies
 * one gain for the job, programmed from `cvt` as a per-tensor scale. */
typedef struct {
    const int16_t *cmul;   /* [C], the job's channel order */
    unsigned shift;
    float cvt;
} dw8_perc;

/* conv_pack_planes at G=16, esz=1, with plane j of the cube read from in + chan[j]*IH*IW. */
static void conv_pack_planes_idx(void *cube, size_t bo_bytes, const int8_t *in, const int *chan,
                                 int C, int IH, int IW, int IHj)
{
    const size_t hw = (size_t)IH * IW, plane = (size_t)IHj * IW * 16;
    const int ng = (C + 15) / 16;
    if ((size_t)ng * plane > bo_bytes) {
        memset(cube, 0, bo_bytes);
        for (int c = 0; c < C; c++)
            for (size_t p = 0; p < hw; p++)
                ((uint8_t *)cube)[(size_t)(c / 16) * plane + p * 16 + c % 16] =
                    ((const uint8_t *)in)[(size_t)chan[c] * hw + p];
        return;
    }
    for (int g = 0; g < ng; g++) {
        uint8_t *d = (uint8_t *)cube + (size_t)g * plane;
        const int nc = C - g * 16 < 16 ? C - g * 16 : 16;
        const uint8_t *pl[16];
        for (int j = 0; j < nc; j++) pl[j] = (const uint8_t *)in + (size_t)chan[g * 16 + j] * hw;
        rc_planes_to_group16_u8(d, pl, nc, hw);
        memset(d + hw * 16, 0, plane - hw * 16);
    }
    const size_t used = (size_t)ng * plane;
    size_t tail = used + 2 * (size_t)CBUF_BANK;
    if (tail > bo_bytes) tail = bo_bytes;
    if (tail > used) memset((uint8_t *)cube + used, 0, tail - used);
}

/* `in_chan` / `out_chan`, when not NULL, name the plane of `in` / `out` each of the job's C
 * channels reads and writes; `w` and `bias_q` are always the job's own C channels in order.
 * `pc` switches the epilogue to the per-channel multiplier (w_scale is then unused). */
static int conv2d_dw_int8_one_job(int fd, rocket_conv_ctx *ctx,
                                  int C, int IH, int IW, int OH, int OW, int KH, int KW,
                                  int sy, int sx, int pt, int pl, int dy, int dx,
                                  const int8_t *in, const int8_t *w, const int32_t *bias_q,
                                  float in_scale, float w_scale, float out_scale,
                                  int in_zp, int w_zp, int out_zp, int8_t *out,
                                  const int *in_chan, const int *out_chan, const dw8_perc *pc)
{
    const int G = DW_INT8_G;
    const int Cpad = ((C + G - 1) / G) * G;
    const int Cj = Cpad;                 /* the channel count the job is programmed at */
    int ret = -1;

    if (fd < 0) {
        /* the per-tensor host fallback; the per-channel entry runs its own before this */
        if (pc || in_chan || out_chan) return -1;
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
    size_t in_elems   = (size_t)Cj * IHj * IW;
    size_t wt_elems   = (size_t)Cj * KH * KW;
    size_t out_elems  = (size_t)Cj * OH * OW;

    size_t in_need  = in_elems  * sizeof(int8_t)  + CBUF_BANK;
    size_t wt_need  = wt_elems  * sizeof(int8_t)  + CBUF_BANK;
    size_t rc_need  = 256 * sizeof(uint64_t);
    /* the per-channel cube is 64 bytes per 8 channels (A, B, C); the bias alone 4 a channel */
    size_t bs_need  = (pc ? (size_t)(Cpad / 8) * 64 : (size_t)Cpad * sizeof(int32_t)) + CBUF_BANK;
    size_t out_need = out_elems * sizeof(int8_t)  + CBUF_BANK;
    rocket_bo lguard = {0}, lin = {0}, lwt = {0}, lrc = {0}, lout = {0}, lbias = {0};
    rocket_bo *guard  = ctx ? &ctx->guard : &lguard;
    rocket_bo *in_bo  = conv_bo(ctx, ctx ? ctx->in : NULL,   &lin,   &in_need);
    rocket_bo *wt_bo  = conv_bo(ctx, ctx ? ctx->wt : NULL,   &lwt,   &wt_need);
    rocket_bo *rc_bo  = conv_bo(ctx, ctx ? ctx->rc : NULL,   &lrc,   &rc_need);
    rocket_bo *out_bo = conv_bo(ctx, ctx ? ctx->out : NULL,  &lout,  &out_need);
    rocket_bo *bs_bo  = conv_bo(ctx, ctx ? ctx->bias : NULL, &lbias, &bs_need);
    uint64_t regs[256] = {0};
    int rc = -1;

    if (rocket_bo_ensure32(fd, guard, 4096) < 0 ||
        rocket_bo_ensure32(fd, in_bo,  in_need) < 0 ||
        rocket_bo_ensure32(fd, wt_bo,  wt_need) < 0 ||
        rocket_bo_ensure32(fd, rc_bo,  rc_need) < 0 ||
        rocket_bo_ensure32(fd, bs_bo,  bs_need) < 0 ||
        rocket_bo_ensure32(fd, out_bo, out_need) < 0) {
        ROCKET_LOGE("rocket_conv2d_dw_int8: BO alloc failed\n");
        goto out;
    }
    const int cp = cprof_on();
    double cpa[CP_N] = {0}, cpt = cp ? cprof_now() : 0.0;

    /* input feature cube (C2=16): the raw int8 values, which the CNA reads signed. The
     * layout is feature_data()'s, (c/16, h, w, c%16) over IHj rows, with the index hoisted:
     * a per-element call was 83% of this entry's time at MobileDet's shapes [HW, RK1,
     * tests/dw_int8_cost_probe.c]. Rows >= IH stay 0. */
    rocket_bo_prep(fd, in_bo, 1, 0);
    if (in_chan) conv_pack_planes_idx(in_bo->ptr, in_bo->size, in, in_chan, C, IH, IW, IHj);
    else         conv_pack_planes(in_bo->ptr, in_bo->size, in, C, IH, IW, IHj, 16, 1);
    rocket_bo_fini(fd, in_bo);
    CP_MARK(cp, cpa, CP_IN, cpt);

    /* DW weight cube (group G): the raw int8 values, in weight_conv_dw_int8()'s layout,
     * (c/G, kh, kw, c%G). */
    rocket_bo_prep(fd, wt_bo, 1, 0);
    memset(wt_bo->ptr, 0, wt_bo->size);
    {
        int8_t *dst = wt_bo->ptr;
        const size_t taps = (size_t)KH * KW;
        for (int c = 0; c < C; c++) {
            int8_t *d = dst + (size_t)(c / G) * taps * G + (c % G);
            const int8_t *src = w + (size_t)c * taps;
            for (size_t t = 0; t < taps; t++) d[t * G] = src[t];
        }
    }
    rocket_bo_fini(fd, wt_bo);

    /* per-OC int32 bias cube: bias_q - in_zp * Σ_kernel(w - w_zp). The CNA pads with in_zp,
     * so a padded tap adds in_zp * w and the fold takes it back out: it contributes 0, as
     * TFLite's padding does. */
    rocket_bo_prep(fd, bs_bo, 1, 0);
    memset(bs_bo->ptr, 0, bs_bo->size);
    {
        int32_t *dst = bs_bo->ptr;
        for (int c = 0; c < C; c++) {
            int32_t sw = 0;
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    sw += w[((size_t)c * KH + kh) * KW + kw] - w_zp;
            const int32_t a = (bias_q ? bias_q[c] : 0) - in_zp * sw;
            if (pc) {
                /* the 64-byte group: A[8] int32 at 0, B[8] int16 at 32 (0: no per-channel
                 * weight zero point), C[8] int16 at 48 */
                uint8_t *g = (uint8_t *)bs_bo->ptr + (size_t)(c / 8) * 64;
                memcpy(g + 4 * (c % 8), &a, 4);
                memcpy(g + 48 + 2 * (c % 8), &pc->cmul[c], 2);
            } else {
                dst[c] = a;
            }
        }
    }
    rocket_bo_fini(fd, bs_bo);
    CP_MARK(cp, cpa, CP_WT, cpt);   /* the weight and the bias cube */

    {
        conv_params_t p = {
            .ic = Cj, .ih = IHj, .iw = IW, .oc = Cj, .oh = OH, .ow = OW,
            .kh = KH, .kw = KW, .stride_y = sy, .stride_x = sx,
            .dil_y = dy, .dil_x = dx, .pad_top = pt, .pad_left = pl,
            .input_dma = (uint32_t)in_bo->dma_address,
            .weights_dma = (uint32_t)wt_bo->dma_address,
            .output_dma = (uint32_t)out_bo->dma_address,
            .tasks = regs, .dw_group = (uint8_t)G, .int8_out = 1,
            /* per-channel: the OUT_CVT carries the job's base gain alone */
            .in_scale = pc ? pc->cvt : in_scale, .w_scale = pc ? 1.0f : w_scale,
            .out_scale = pc ? 1.0f : out_scale,
            .bs_mul_perc = pc ? 1 : 0, .bs_mul_shift = pc ? (uint8_t)pc->shift : 0,
            /* The generator carries Mesa's uint8 formulas (pad = Z - 0x80, offset =
             * Z_out - 0x80, DPU_BS_OW_OP = 0x80 - Z_w), so it takes the uint8-equivalent
             * zero points: the pad lands on in_zp, the offset on out_zp, and the CPEND
             * operand on 0. At the 0x80 a raw w_zp of 0 would give, nearly every output
             * is wrong [HW sweep, RK1]. */
            .input_zero_point = in_zp + 0x80, .output_zero_point = out_zp + 0x80,
            .weight_zero_point = w_zp + 0x80,
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
        CP_MARK(cp, cpa, CP_GEN, cpt);

        rocket_bo_prep(fd, out_bo, 1, 0);
        memset(out_bo->ptr, 0, out_bo->size);
        rocket_bo_fini(fd, out_bo);
        CP_MARK(cp, cpa, CP_OUTZ, cpt);

        rocket_task_desc task = { .regcmd = (uint32_t)rc_bo->dma_address,
                                  .regcmd_count = p.task_count };
        uint32_t in_h[]  = { in_bo->handle, wt_bo->handle, bs_bo->handle, rc_bo->handle };
        uint32_t out_h[] = { out_bo->handle };
        ret = rocket_submit_tasks(fd, &task, 1, in_h, 4, out_h, 1);
        if (ret) { ROCKET_LOGE("rocket_conv2d_dw_int8: submit failed (%d)\n", ret); goto out; }
        CP_MARK(cp, cpa, CP_SUBMIT, cpt);
    }

    ret = rocket_bo_prep(fd, out_bo, 0, 2000000000ULL);
    if (ret) { ROCKET_LOGE("rocket_conv2d_dw_int8: wait timeout (%d)\n", ret); goto out; }
    CP_MARK(cp, cpa, CP_WAIT, cpt);
    {
        /* int8 output cube (C2=16), already in the model domain: feature_data()'s layout
         * over OH rows, gathered with the index hoisted. */
        const int8_t *src = out_bo->ptr;
        const size_t plane = (size_t)OH * OW * 16, hw = (size_t)OH * OW;
        for (int g = 0; g < (C + 15) / 16; g++) {    /* sixteen channels a pass */
            const int nc = C - g * 16 < 16 ? C - g * 16 : 16;
            uint8_t *pl[16];
            for (int j = 0; j < nc; j++)
                pl[j] = (uint8_t *)out + (size_t)(out_chan ? out_chan[g * 16 + j] : g * 16 + j) * hw;
            rc_group16_to_planes_u8(pl, nc, (const uint8_t *)src + (size_t)g * plane, hw);
        }
    }
    CP_MARK(cp, cpa, CP_READ, cpt);
    rocket_bo_fini(fd, out_bo);
    CP_MARK(cp, cpa, CP_FINI, cpt);
    if (cp) {
        cprof_add(CPP_DW8, cpa, 1);
        cprof_bo(CPP_DW8, in_bo->size, in_elems, out_bo->size, out_elems);
    }
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

/* Row and column bands for one 64-channel group of the int8-DW program when its whole plane
 * is past one CBUF pass, chosen as the fp16 tiler chooses them: the tallest band at the full
 * width whose padded feature fits the budget, narrower columns only if one row band still
 * overflows, and no band under the 4-row DMA floor. 0 and the band extents, or -4. Pure. */
static int dw_int8_bands(int OH, int OW, int KH, int KW, int sy, int sx, int dy, int dx,
                         int *rht_out, int *cwt_out)
{
    const int G = DW_INT8_G;
    #define DW8_FEAT(rh, cw) ((size_t)G * (size_t)job_rows(datain_h(rh, sy, KH, dy)) * \
        (size_t)datain_w(cw, sx, KW, dx))
    int rht_lo = 1;
    while (rht_lo < OH && datain_h(rht_lo, sy, KH, dy) < CONV_MIN_DATAIN_H) rht_lo++;
    int rht = OH, cwt = OW;
    while (rht > rht_lo && datain_h(rht, sy, KH, dy) > CNA_TILE_MAX_H) rht--;
    while (cwt > 1 && datain_w(cwt, sx, KW, dx) > CNA_TILE_MAX_W) cwt--;
    while (rht > rht_lo && DW8_FEAT(rht, cwt) > CONV_FEAT_BUDGET) rht--;
    while (cwt > 1 && DW8_FEAT(rht, cwt) > CONV_FEAT_BUDGET) cwt--;
    const int ok = DW8_FEAT(rht, cwt) <= CONV_FEAT_BUDGET &&
                   datain_h(rht, sy, KH, dy) <= CNA_TILE_MAX_H &&
                   datain_w(cwt, sx, KW, dx) <= CNA_TILE_MAX_W;
    #undef DW8_FEAT
    if (!ok) return -4;
    if (rht_out) *rht_out = rht;
    if (cwt_out) *cwt_out = cwt;
    return 0;
}

/* The RK3588 int8-DW program's shape envelope, shared by the run and the public planner so
 * the two cannot disagree. Returns 0 and the channel chunk in *cc, or <0. A plane whose one
 * 64-channel group is past a CBUF pass takes row (and if needed column) bands: *cc is then
 * the group and *rht / *cwt the band extents; otherwise they are 0. Pure. */
static int dw_int8_shape_bands(const rocket_conv2d_desc *d, int *cc, int *rht, int *cwt)
{
    if (rht) *rht = 0;
    if (cwt) *cwt = 0;
    if (!d || !d->depthwise || d->oc != d->ic) return -2;
    if (d->ic <= 0 || d->ih <= 0 || d->iw <= 0 || d->kh <= 0 || d->kw <= 0) return -2;
    if (d->stride_x <= 0 || d->stride_y <= 0 || d->dil_x <= 0 || d->dil_y <= 0) return -2;
    const int C = d->ic, IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw;
    /* The int8-DW feature cube is C2=16 (feature_data(C,...,16,...)); a channel
     * count not a multiple of 16 desyncs the cube padding from the regcmd
     * channel-grain count -> wrong output. (Cc stays a G-multiple and G=64%16==0, so
     * C%16==0 also keeps every channel chunk aligned.) */
    if (C % 16 != 0) return -5;
    const int G = DW_INT8_G;
    if (rocket_conv2d_oh(d) <= 0 || rocket_conv2d_ow(d) <= 0) return -3;
    /* one G-channel weight cube must fit a bank */
    if ((size_t)KW * KH * G * sizeof(int8_t) > CBUF_BANK) return -4;
    /* channel chunk: the largest G-multiple that fits, budgeted at the job's PADDED width,
     * since conv2d_dw_int8_one_job programs a partial last group as a whole one */
    int Cc = ((C + G - 1) / G) * G;
    while (Cc > G && (size_t)Cc * job_rows(IH) * IW * sizeof(int8_t) > CONV_FEAT_BUDGET) Cc -= G;
    if ((size_t)Cc * job_rows(IH) * IW * sizeof(int8_t) > CONV_FEAT_BUDGET ||
        IH > CNA_TILE_MAX_H || IW > CNA_TILE_MAX_W || !cna_pad_fits(d->pad_top, d->pad_left)) {
        /* even one group's plane is past a pass, past the CNA's fields, or its pad past
         * PAD_CON0's: bands, which materialize the pad */
        if (dw_int8_bands(rocket_conv2d_oh(d), rocket_conv2d_ow(d), KH, KW, d->stride_y,
                          d->stride_x, d->dil_y, d->dil_x, rht, cwt) < 0) return -4;
        Cc = G;
    }
    if (cc) *cc = Cc;
    return 0;
}

static int dw_int8_shape(const rocket_conv2d_desc *d, int *cc)
{
    return dw_int8_shape_bands(d, cc, NULL, NULL);
}

/* One channel chunk of the int8-DW program in row and column bands: each band's sub-input
 * is materialized with its halo, rows and columns outside the plane filled with the INPUT
 * ZERO POINT, and runs as one job with no hardware pad. A materialized in_zp tap is what the
 * hardware pad writes: the bias fold removes in_zp times the kernel sum from every window and
 * CPEND sees the same window sum, so each band computes the whole plane's rows exactly. */
static int dw_int8_spatial(int fd, rocket_conv_ctx *ctx, int Cn, int IH, int IW, int OH, int OW,
                           int KH, int KW, int sy, int sx, int pt, int pl, int dy, int dx,
                           int rht, int cwt, const int8_t *in, const int8_t *w,
                           const int32_t *bias_q, float in_scale, float w_scale, float out_scale,
                           int in_zp, int w_zp, int out_zp, int8_t *out,
                           const int *chan, const dw8_perc *pc)
{
    const int IHs = (int)datain_h(rht, sy, KH, dy), IWs = (int)datain_w(cwt, sx, KW, dx);
    const int nRmax = (OH + rht - 1) / rht, nCmax = (OW + cwt - 1) / cwt;
    const size_t szs[4] = { (size_t)nRmax * sizeof(int), (size_t)nCmax * sizeof(int),
                            (size_t)Cn * IHs * IWs, (size_t)Cn * rht * cwt };
    rocket_arena a = {0};
    if (rocket_arena_open(&a, ctx ? &ctx->scratch : NULL, rocket_arena_reserve(szs, 4)) < 0)
        return -1;
    int    *rband   = rocket_arena_push(&a, szs[0]);
    int    *cband   = rocket_arena_push(&a, szs[1]);
    int8_t *sub_in  = rocket_arena_push(&a, szs[2]);
    int8_t *sub_out = rocket_arena_push(&a, szs[3]);
    if (!rband || !cband || !sub_in || !sub_out) { rocket_arena_close(&a); return -1; }
    const int nR = balance_bands(OH, rht, rband), nC = balance_bands(OW, cwt, cband);

    int ret = 0, r0 = 0;
    for (int ri = 0; ri < nR && !ret; ri++) {
        const int rh = rband[ri], ih_sub = (int)datain_h(rh, sy, KH, dy), gh0 = r0 * sy - pt;
        int c0 = 0;
        for (int ci = 0; ci < nC && !ret; ci++) {
            const int cw = cband[ci], iw_sub = (int)datain_w(cw, sx, KW, dx), gw0 = c0 * sx - pl;
            memset(sub_in, (unsigned char)(signed char)in_zp, (size_t)Cn * ih_sub * iw_sub);
            const int cs = gw0 < 0 ? 0 : gw0, ce = gw0 + iw_sub > IW ? IW : gw0 + iw_sub;
            for (int c = 0; c < Cn && cs < ce; c++)
                for (int j = 0; j < ih_sub; j++) {
                    const int gih = gh0 + j;
                    if (gih < 0 || gih >= IH) continue;
                    memcpy(&sub_in[((size_t)c * ih_sub + j) * iw_sub + (cs - gw0)],
                           &in[((size_t)(chan ? chan[c] : c) * IH + gih) * IW + cs],
                           (size_t)(ce - cs));
                }
            ret = conv2d_dw_int8_one_job(fd, ctx, Cn, ih_sub, iw_sub, rh, cw, KH, KW, sy, sx,
                                         0, 0, dy, dx, sub_in, w, bias_q, in_scale, w_scale,
                                         out_scale, in_zp, w_zp, out_zp, sub_out, NULL, NULL, pc);
            if (ret) break;
            for (int c = 0; c < Cn; c++)
                for (int r = 0; r < rh; r++)
                    memcpy(&out[((size_t)(chan ? chan[c] : c) * OH + (r0 + r)) * OW + c0],
                           &sub_out[((size_t)c * rh + r) * cw], (size_t)cw);
            c0 += cw;
        }
        r0 += rh;
    }
    rocket_arena_close(&a);
    return ret;
}

int rocket_conv2d_dw_int8_plan(const rocket_conv2d_desc *d)
{
    if (!d || !d->depthwise) return ROCKET_E_SHAPE;
    if (conv_is_rk3576()) return rocket_conv2d_int8_plan_rk3576(d);
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (!hw || !hw->name || strcmp(hw->name, "rk3588")) return ROCKET_E_UNSUPPORTED;
    return dw_int8_shape(d, NULL) ? ROCKET_E_SHAPE : ROCKET_OK;
}

/* DW int8-out runtime: validate + channel-tile (chunks of G=64, each an independent job
 * — channels are independent), and row/column bands when one group's plane is past a
 * CBUF pass (dw_int8_spatial). */
static int conv2d_dw_int8_run(int fd, rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *w, const int32_t *bias_q,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out)
{
    int Cc = 0, rht = 0, cwt = 0;
    int ret = dw_int8_shape_bands(d, &Cc, &rht, &cwt);
    if (ret == -5)
        ROCKET_LOGE("rocket_conv2d_dw_int8: channel count C=%d must be a multiple "
                "of 16 (C2 feature-cube grain)\n", d->ic);
    if (ret) return ret;
    /* A weight zero point adds -w_zp * sum(x) per window, which depends on the input and
     * cannot fold into the bias. The generator writes DPU_BS_OW_OP = 0x80 - (w_zp + 0x80),
     * and the DPU adds that operand times each window's input sum, border pad included, so
     * the term is computed on chip [HW sweep, RK1, tests/cpend_wzp_probe.c]. The weight
     * bytes are int8, so a zero point outside that range is not a TFLite int8 tensor. */
    if (w_zp < -128 || w_zp > 127) {
        ROCKET_LOGE("rocket_conv2d_dw_int8: weight zero point %d is outside int8\n", w_zp);
        return ROCKET_E_UNSUPPORTED;
    }
    const int C = d->ic, IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int sy = d->stride_y, sx = d->stride_x, pt = d->pad_top, pl = d->pad_left;
    const int dy = d->dil_y, dx = d->dil_x;

    for (int c0 = 0; c0 < C && !ret; c0 += Cc) {
        int Cn = (C - c0 < Cc) ? (C - c0) : Cc;
        if (rht)
            ret = dw_int8_spatial(fd, ctx, Cn, IH, IW, OH, OW, KH, KW, sy, sx, pt, pl, dy, dx,
                                  rht, cwt, in + (size_t)c0 * IH * IW, w + (size_t)c0 * KH * KW,
                                  bias_q ? bias_q + c0 : NULL,
                                  in_scale, w_scale, out_scale, in_zp, w_zp, out_zp,
                                  out + (size_t)c0 * OH * OW, NULL, NULL);
        else
            ret = conv2d_dw_int8_one_job(fd, ctx, Cn, IH, IW, OH, OW, KH, KW, sy, sx, pt, pl,
                                         dy, dx, in + (size_t)c0 * IH * IW,
                                         w + (size_t)c0 * KH * KW, bias_q ? bias_q + c0 : NULL,
                                         in_scale, w_scale, out_scale, in_zp, w_zp, out_zp,
                                         out + (size_t)c0 * OH * OW, NULL, NULL, NULL);
    }
    return ret;
}

int rocket_conv2d_dw_int8(int fd, const rocket_conv2d_desc *d,
                          const int8_t *in, const int8_t *w, const int32_t *bias,
                          float in_scale, float w_scale, float out_scale,
                          int in_zp, int w_zp, int out_zp, int8_t *out)
{
    if (conv_is_rk3576())
        return rocket_conv2d_dw_int8_rk3576(fd, d, in, w, bias, in_scale, w_scale,
                                            out_scale, in_zp, w_zp, out_zp, out);
    if (conv_wrong_encoding("rocket_conv2d_dw_int8", "the chip's own conv encoder"))
        return ROCKET_E_UNSUPPORTED;
    return conv2d_dw_int8_run(fd, NULL, d, in, w, bias, in_scale, w_scale, out_scale,
                              in_zp, w_zp, out_zp, out);
}

int rocket_conv2d_dw_int8_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *w, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out)
{
    if (!ctx) return -1;
    /* The same part dispatch as rocket_conv2d_dw_int8, so this entry, that one and
     * rocket_conv2d_dw_int8_plan answer for one program per part. */
    if (conv_is_rk3576())
        return rocket_conv2d_dw_int8_rk3576(ctx->fd, d, in, w, bias, in_scale, w_scale,
                                            out_scale, in_zp, w_zp, out_zp, out);
    return conv2d_dw_int8_run(ctx->fd, ctx, d, in, w, bias, in_scale, w_scale, out_scale,
                              in_zp, w_zp, out_zp, out);
}

/* ---- per-channel (per-axis) int8 depthwise on the BS multiplier --------------------
 *
 * One job carries one OUT_CVT, so a channel reaches its own scale through the int16 C in
 * the BS coefficient group: the stage computes sat32(rne(((acc + A) * C) >> s)) with the
 * product held wide, and the OUT_CVT applies the job's base gain G [HW sweep, RK1,
 * tests/dw_perc_probe.c]. Per job the planner puts the largest channel at C = 32767 and
 * the rest in proportion, with the smallest s whose product fits int32 at every channel's
 * accumulator bound, so G is as fine as the bound allows.
 *
 * A channel's gain then resolves to 0.5/C relative, which the job's scale spread sets. The
 * channels are sorted by scale and a job ends where the next channel's C would fall under
 * ROCKET_DW_PERC_MIN_C (2048, a gain within 2^-12, about the fp16 route's operand rounding),
 * so a wide spread costs jobs rather than resolution. Writing those channels on the host
 * costs more than the route saves: 14 host channels took EfficientDet-Lite0's 160x160x32
 * layer from 3.0 to 12.8 ms [HW, RK1, 2026-09-27]. A channel whose filter is all zero is
 * written by the host as its bias's constant. */

static int dw_perc_min_c(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *e = getenv("ROCKET_DW_PERC_MIN_C");
        int x = e ? atoi(e) : 2048;
        v = x < 1 ? 1 : (x > 32767 ? 32767 : x);
    }
    return v;
}

/* One channel of TFLite's per-channel depthwise on the host, dw_int8_ref's arithmetic. */
static void dw_int8_ref_chan(const rocket_conv2d_desc *d, const int8_t *in, const int8_t *w,
                             const int32_t *bias, float in_scale, float w_scale_c,
                             float out_scale, int in_zp, int out_zp, int c, int8_t *out)
{
    const int IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            int32_t acc = bias ? bias[c] : 0;
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh * d->stride_y + kh * d->dil_y - d->pad_top;
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow * d->stride_x + kw * d->dil_x - d->pad_left;
                    int q = (ih >= 0 && ih < IH && iw >= 0 && iw < IW)
                          ? in[((size_t)c * IH + ih) * IW + iw] : in_zp;
                    acc += (q - in_zp) * w[((size_t)c * KH + kh) * KW + kw];
                }
            }
            float v = (in_scale * w_scale_c) * (float)acc;
            long q = (long)lrintf(v / out_scale) + out_zp;
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            out[((size_t)c * OH + oh) * OW + ow] = (int8_t)q;
        }
}

/* The OUT_CVT pair the emitter derives from a float scale, and the gain it realizes. */
static double dw_perc_cvt_gain(float g)
{
    unsigned m, sh;
    npu_out_cvt_pair(g, &m, &sh);
    return ldexp((double)m, -(int)sh);
}

/* Plan one job over `n` channels in `idx` (sorted by scale, largest first): the shift, the
 * OUT_CVT scale and each channel's C. Marks host[] for a channel the ramp cannot resolve.
 * 0, or -1 when no shift fits (an accumulator bound past what 63 bits of shift absorb). */
static int dw_perc_plan_job(const int *idx, int n, const double *M, const double *bound,
                            int16_t *cmul, unsigned *shift, float *cvt, uint8_t *host)
{
    const double Mhi = M[idx[0]];
    double gmin = 0.0;
    for (int j = 0; j < n; j++) {
        const double g = bound[idx[j]] * M[idx[j]] / 2147483647.0;
        if (g > gmin) gmin = g;
    }
    gmin *= 1.001;
    int s = 0;
    if (32767.0 * gmin > Mhi) s = (int)ceil(log2(32767.0 * gmin / Mhi));
    if (s > 63) return -1;
    const float g = (float)(Mhi * ldexp(1.0, s) / 32767.0);
    const double greal = dw_perc_cvt_gain(g);
    const int minc = dw_perc_min_c();
    for (int j = 0; j < n; j++) {
        double cv = M[idx[j]] * ldexp(1.0, s) / greal;
        long cc = lround(cv);
        if (cc > 32767) cc = 32767;
        if (cc < 0) cc = 0;
        cmul[j] = (int16_t)cc;
        if (cc < minc) host[idx[j]] = 1;
    }
    *shift = (unsigned)s;
    *cvt = g;
    return 0;
}

/* qsort order for dw_perc_plan_layer: scale descending, then channel index, so the order is
 * total and the plan does not depend on the sort's stability. */
static const double *g_dw_perc_sort_M;
static pthread_mutex_t g_dw_perc_sort_mu = PTHREAD_MUTEX_INITIALIZER;
static int dw_perc_cmp(const void *a, const void *b)
{
    const int x = *(const int *)a, y = *(const int *)b;
    const double mx = g_dw_perc_sort_M[x], my = g_dw_perc_sort_M[y];
    if (mx != my) return mx > my ? -1 : 1;
    return (x > y) - (x < y);
}

/* The next job's channel count from order[j0]: at most Cc, and cut where a channel's scale
 * falls under ROCKET_DW_PERC_MIN_C/32767 of the job's first (largest), so every channel of a
 * job keeps C >= that floor. A scale class costs a job; a channel below the floor would cost a
 * host pass over its plane, which is dearer on every plane but the smallest. */
static int dw_perc_job_len(const int *order, int j0, int nlive, int Cc, const double *M)
{
    const double floor_m = M[order[j0]] * (double)dw_perc_min_c() / 32767.0;
    int n = 1;
    while (j0 + n < nlive && n < Cc && M[order[j0 + n]] >= floor_m) n++;
    return n;
}

/* The whole layer's plan: live channels sorted by scale (largest first) in `order`, cut into
 * jobs by dw_perc_job_len, each job planned by dw_perc_plan_job; per-channel results in the
 * [C] arrays. `nlive` is how many of `order` run on the device. */
static int perc_plan_layer(int C, size_t taps, const int8_t *w, const int32_t *bias,
                           float in_scale, const float *w_scale, float out_scale, int in_zp,
                           int Cc, int *order, int *nlive_out, int16_t *cmul, uint8_t *shift,
                           float *cvt, uint8_t *host)
{
    const int xm = (128 + in_zp) > (127 - in_zp) ? (128 + in_zp) : (127 - in_zp);
    double *M = malloc((size_t)C * sizeof(double)), *bound = malloc((size_t)C * sizeof(double));
    int16_t *cj = malloc((size_t)Cc * sizeof(int16_t));
    int ret = 0, nlive = 0;
    if (!M || !bound || !cj) { ret = ROCKET_E_NOMEM; goto out; }
    memset(host, 0, (size_t)C);
    memset(cmul, 0, (size_t)C * sizeof(int16_t));
    memset(shift, 0, (size_t)C);
    memset(cvt, 0, (size_t)C * sizeof(float));
    /* an all-zero filter, or a scale that is not a positive number, goes to the host */
    for (int c = 0; c < C; c++) {
        const int8_t *wc = w + (size_t)c * taps;
        int32_t si = 0;                      /* exact: taps * 128 stays far inside int32 */
        for (size_t t = 0; t < taps; t++) si += wc[t] < 0 ? -wc[t] : wc[t];
        const double sw = (double)si;
        M[c] = (double)in_scale * (double)w_scale[c] / (double)out_scale;
        bound[c] = fabs(bias ? (double)bias[c] : 0.0) + (double)xm * sw;
        if (sw == 0.0 || !(M[c] > 0.0)) host[c] = 1;
        else order[nlive++] = c;
    }
    pthread_mutex_lock(&g_dw_perc_sort_mu);
    g_dw_perc_sort_M = M;
    qsort(order, (size_t)nlive, sizeof(int), dw_perc_cmp);
    pthread_mutex_unlock(&g_dw_perc_sort_mu);
    for (int j0 = 0, n = 0; j0 < nlive; j0 += n) {
        n = dw_perc_job_len(order, j0, nlive, Cc, M);
        unsigned s = 0;
        float g = 0.0f;
        if (dw_perc_plan_job(order + j0, n, M, bound, cj, &s, &g, host)) {
            ret = ROCKET_E_UNSUPPORTED;
            goto out;
        }
        for (int j = 0; j < n; j++) {
            const int c = order[j0 + j];
            cmul[c] = cj[j];
            shift[c] = (uint8_t)s;
            cvt[c] = g;
        }
    }
    *nlive_out = nlive;
out:
    free(M); free(bound); free(cj);
    return ret;
}

/* The depthwise layer's plan: one filter of KH*KW taps per channel, jobs of at most Cc. */
static int dw_perc_plan_layer(const rocket_conv2d_desc *d, const int8_t *w, const int32_t *bias,
                              float in_scale, const float *w_scale, float out_scale, int in_zp,
                              int Cc, int *order, int *nlive_out, int16_t *cmul, uint8_t *shift,
                              float *cvt, uint8_t *host)
{
    return perc_plan_layer(d->ic, (size_t)d->kh * d->kw, w, bias, in_scale, w_scale, out_scale,
                           in_zp, Cc, order, nlive_out, cmul, shift, cvt, host);
}

int rocket_conv2d_dw_int8_perc_plan_channels(const rocket_conv2d_desc *d, const int8_t *w,
                                             const int32_t *bias, float in_scale,
                                             const float *w_scale, float out_scale, int in_zp,
                                             int16_t *cmul, uint8_t *shift, float *cvt,
                                             uint8_t *host)
{
    int Cc = 0, nlive = 0;
    if (!d || !w || !w_scale || !cmul || !shift || !cvt || !host) return ROCKET_E_SHAPE;
    if (dw_int8_shape_bands(d, &Cc, NULL, NULL)) return ROCKET_E_SHAPE;
    int *order = malloc((size_t)d->ic * sizeof(int));
    if (!order) return ROCKET_E_NOMEM;
    int ret = dw_perc_plan_layer(d, w, bias, in_scale, w_scale, out_scale, in_zp, Cc, order,
                                 &nlive, cmul, shift, cvt, host);
    free(order);
    return ret;
}

static int conv2d_dw_int8_perc_run(int fd, rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                                   const int8_t *in, const int8_t *w, const int32_t *bias,
                                   float in_scale, const float *w_scale, float out_scale,
                                   int in_zp, int out_zp, int8_t *out)
{
    if (!d || !in || !w || !w_scale || !out) return ROCKET_E_SHAPE;
    int Cc = 0, rht = 0, cwt = 0, nlive = 0;
    int ret = dw_int8_shape_bands(d, &Cc, &rht, &cwt);
    if (ret) return ROCKET_E_SHAPE;
    const int C = d->ic, IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const size_t taps = (size_t)KH * KW;

    int *order = malloc((size_t)C * sizeof(int));
    uint8_t *host = malloc((size_t)C), *shift = malloc((size_t)C);
    int16_t *cmul = malloc((size_t)C * sizeof(int16_t)), *cj = malloc((size_t)Cc * sizeof(int16_t));
    float *cvt = malloc((size_t)C * sizeof(float));
    int8_t *wj = malloc((size_t)Cc * taps);
    int32_t *bj = malloc((size_t)Cc * sizeof(int32_t));
    if (!order || !host || !shift || !cmul || !cj || !cvt || !wj || !bj) {
        ret = ROCKET_E_NOMEM;
        goto out;
    }
    ret = dw_perc_plan_layer(d, w, bias, in_scale, w_scale, out_scale, in_zp, Cc, order, &nlive,
                             cmul, shift, cvt, host);
    if (ret) goto out;

    if (fd >= 0) {
        /* the plan's per-channel scales, for the same job cut the planner made */
        double *Mo = malloc((size_t)C * sizeof(double));
        if (!Mo) { ret = ROCKET_E_NOMEM; goto out; }
        for (int c = 0; c < C; c++) Mo[c] = (double)in_scale * (double)w_scale[c] / (double)out_scale;
        for (int j0 = 0, n = 0; j0 < nlive && !ret; j0 += n) {
            n = dw_perc_job_len(order, j0, nlive, Cc, Mo);
            const int *idx = order + j0;
            dw8_perc pc = { .cmul = cj, .shift = shift[idx[0]], .cvt = cvt[idx[0]] };
            for (int j = 0; j < n; j++) {
                memcpy(wj + (size_t)j * taps, w + (size_t)idx[j] * taps, taps);
                bj[j] = bias ? bias[idx[j]] : 0;
                cj[j] = cmul[idx[j]];
            }
            if (rht)
                ret = dw_int8_spatial(fd, ctx, n, IH, IW, OH, OW, KH, KW, d->stride_y,
                                      d->stride_x, d->pad_top, d->pad_left, d->dil_y, d->dil_x,
                                      rht, cwt, in, wj, bj, in_scale, 1.0f, out_scale,
                                      in_zp, 0, out_zp, out, idx, &pc);
            else
                ret = conv2d_dw_int8_one_job(fd, ctx, n, IH, IW, OH, OW, KH, KW, d->stride_y,
                                             d->stride_x, d->pad_top, d->pad_left, d->dil_y,
                                             d->dil_x, in, wj, bj, in_scale, 1.0f, out_scale,
                                             in_zp, 0, out_zp, out, idx, idx, &pc);
        }
        free(Mo);
        if (ret) goto out;
    } else {
        memset(host, 1, (size_t)C);
    }
    {
        int nh = 0;
        for (int c = 0; c < C; c++) {
            if (!host[c]) continue;
            int zero = 1;
            for (size_t t = 0; t < taps && zero; t++) zero = w[(size_t)c * taps + t] == 0;
            if (zero) {
                /* an all-zero filter: every tap, padded ones included, adds (q - in_zp) * 0,
                 * so the accumulator is the bias and the plane one value */
                const float v = (in_scale * w_scale[c]) * (float)(bias ? bias[c] : 0);
                long q = (long)lrintf(v / out_scale) + out_zp;
                q = q < -128 ? -128 : q > 127 ? 127 : q;
                memset(out + (size_t)c * OH * OW, (int)(int8_t)q, (size_t)OH * OW);
            } else {
                dw_int8_ref_chan(d, in, w, bias, in_scale, w_scale[c], out_scale, in_zp, out_zp,
                                 c, out);
            }
            nh++;
        }
        ROCKET_LOGD("rocket_conv2d_dw_int8_perc: C=%d %dx%d k%dx%d, %d of %d channels on the "
                    "host\n", C, IH, IW, KH, KW, nh, C);
    }
out:
    free(order); free(host); free(shift); free(cmul); free(cj); free(cvt); free(wj); free(bj);
    return ret;
}

int rocket_conv2d_dw_int8_perc_plan(const rocket_conv2d_desc *d)
{
    if (!d || !d->depthwise) return ROCKET_E_SHAPE;
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (!hw || !hw->name || strcmp(hw->name, "rk3588")) return ROCKET_E_UNSUPPORTED;
    return dw_int8_shape(d, NULL) ? ROCKET_E_SHAPE : ROCKET_OK;
}

int rocket_conv2d_dw_int8_perc(int fd, const rocket_conv2d_desc *d,
                               const int8_t *in, const int8_t *w, const int32_t *bias,
                               float in_scale, const float *w_scale, float out_scale,
                               int in_zp, int out_zp, int8_t *out)
{
    if (conv_is_rk3576() ||
        conv_wrong_encoding("rocket_conv2d_dw_int8_perc", "the chip's own conv encoder"))
        return ROCKET_E_UNSUPPORTED;
    return conv2d_dw_int8_perc_run(fd, NULL, d, in, w, bias, in_scale, w_scale, out_scale,
                                   in_zp, out_zp, out);
}

int rocket_conv2d_dw_int8_perc_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                                   const int8_t *in, const int8_t *w, const int32_t *bias,
                                   float in_scale, const float *w_scale, float out_scale,
                                   int in_zp, int out_zp, int8_t *out)
{
    if (!ctx) return ROCKET_E_SHAPE;
    if (conv_is_rk3576() ||
        conv_wrong_encoding("rocket_conv2d_dw_int8_perc", "the chip's own conv encoder"))
        return ROCKET_E_UNSUPPORTED;
    return conv2d_dw_int8_perc_run(ctx->fd, ctx, d, in, w, bias, in_scale, w_scale, out_scale,
                                   in_zp, out_zp, out);
}

/* ---- per-channel (per-axis) int8 DIRECT conv on the BS multiplier ---------------------
 *
 * rocket_conv2d_int8_q_perc: the direct int8-out program of rocket_conv2d_int8_q with a
 * per-output-channel weight scale, carried as the per-channel depthwise entry carries it: the
 * BS stage multiplies channel c's biased accumulator by the int16 C[c] and shifts, and the
 * OUT_CVT applies one gain per job [HW sweep, RK1, tests/conv_i8out_probe: the multiplier
 * and the sign-split shift compute on the direct program as on the depthwise one].
 *
 * The channels are sorted by scale and cut into jobs at the depthwise entry's scale-class
 * floor (ROCKET_DW_PERC_MIN_C, default 2048, a gain within 2^-12); there is no count cut,
 * since the direct tiler splits a job's channels by the CBUF itself. Each job is programmed
 * at whole 32-kernel groups (zero kernels, zero bias and C = 0 in the padding: a program
 * whose OC ends part way through a group does not complete), with the weights permuted into
 * the sorted order. The tiles of every job go through one dispatch, so a layer's jobs fan
 * out across the pool together, and the output comes back in the sorted order and is put
 * back plane by plane. A layer that plans to one job (76 of EfficientDet-Lite0's 102 convs)
 * keeps its channel order and runs as rocket_conv2d_int8_q does, one-job fast path included,
 * with no permutation and no materialized halo. A channel the plan gives to the host (an
 * all-zero filter, a scale that is not a positive number, a C that rounds under the floor)
 * is TFLite's float arithmetic. */

/* One output channel of TFLite's per-channel direct conv on the host (a padded tap
 * contributes nothing), with its float requant. */
static void conv_i8_ref_chan(const rocket_conv2d_desc *d, const int8_t *in, const int8_t *W,
                             const int32_t *bias, float in_scale, float w_scale_c,
                             float out_scale, int in_zp, int out_zp, int c, int8_t *out)
{
    const int IC = d->ic, IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int8_t *wc = W + (size_t)c * IC * KH * KW;
    int any = 0;
    for (size_t k = 0; k < (size_t)IC * KH * KW && !any; k++) any = wc[k] != 0;
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            int64_t acc = bias ? bias[c] : 0;
            for (int ic = 0; ic < IC && any; ic++)
                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * d->stride_y + kh * d->dil_y - d->pad_top;
                    if (ih < 0 || ih >= IH) continue;
                    for (int kw = 0; kw < KW; kw++) {
                        const int iw = ow * d->stride_x + kw * d->dil_x - d->pad_left;
                        if (iw < 0 || iw >= IW) continue;
                        acc += (int64_t)(in[((size_t)ic * IH + ih) * IW + iw] - in_zp) *
                               wc[((size_t)ic * KH + kh) * KW + kw];
                    }
                }
            const float v = (in_scale * w_scale_c) * (float)acc;
            long q = (long)lrintf(v / out_scale) + out_zp;
            out[((size_t)c * OH + oh) * OW + ow] = (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q);
        }
}

/* A job's gain must sit inside the OUT_CVT's 6-bit shift field; nonzero (and logged) when not. */
static int perc_gain_check(float g)
{
    union { float f; uint32_t u; } cv = { .f = g };
    const int shift_reg = 141 - (int)((cv.u >> 23) & 0xFF);
    if (g > 0.0f && shift_reg >= 1 && shift_reg <= 63) return 0;
    ROCKET_LOGE("rocket_conv2d_int8_q_perc: job gain %g is outside the OUT_CVT's range\n",
                (double)g);
    return -1;
}

static int conv2d_int8_q_perc_run(int fd, rocket_conv_ctx *ctx, rocket_conv_pool *pool,
                                  const rocket_conv2d_desc *d, const int8_t *in, const int8_t *W,
                                  const int32_t *bias, float in_scale, const float *w_scale,
                                  float out_scale, int in_zp, int out_zp, int8_t *out)
{
    if (!d || d->depthwise || !in || !W || !w_scale || !out) return ROCKET_E_SHAPE;
    if (in_zp < -128 || in_zp > 127 || out_zp < -128 || out_zp > 127) {
        ROCKET_LOGE("rocket_conv2d_int8_q_perc: a zero point (%d, %d) is outside int8\n",
                    in_zp, out_zp);
        return ROCKET_E_UNSUPPORTED;
    }
    const int OC = d->oc, ICr = d->ic, IH = d->ih, IW = d->iw, KH = d->kh, KW = d->kw;
    const int OH = rocket_conv2d_oh(d), OW = rocket_conv2d_ow(d);
    const int sy = d->stride_y, sx = d->stride_x, pt = d->pad_top, pl = d->pad_left;
    const int dy = d->dil_y, dx = d->dil_x;
    if (OC <= 0 || ICr <= 0 || KH <= 0 || KW <= 0 || OH <= 0 || OW <= 0) return ROCKET_E_SHAPE;
    const size_t K = (size_t)ICr * KH * KW, hw = (size_t)OH * OW;

    int ret = 0, nlive = 0, njobs = 0;
    int *order = malloc((size_t)OC * sizeof(int));
    int *jb = malloc((size_t)OC * sizeof(int)), *jn = malloc((size_t)OC * sizeof(int));
    uint8_t *host = malloc((size_t)OC), *shift = malloc((size_t)OC);
    int16_t *cmul = malloc((size_t)OC * sizeof(int16_t));
    float *cvt = malloc((size_t)OC * sizeof(float));
    double *Mo = malloc((size_t)OC * sizeof(double));
    int8_t *Wp = NULL, *tmp = NULL;
    int32_t *Ap = NULL;
    int16_t *Cp = NULL;
    i8q_t *qs = NULL;
    i8tile *tiles = NULL;
    int *rband = NULL, *cband = NULL;
    if (!order || !jb || !jn || !host || !shift || !cmul || !cvt || !Mo) {
        ret = ROCKET_E_NOMEM;
        goto out;
    }
    ret = perc_plan_layer(OC, K, W, bias, in_scale, w_scale, out_scale, in_zp, OC, order,
                          &nlive, cmul, shift, cvt, host);
    if (ret) goto out;

    if (nlive > 0) {
        /* the job cut the planner made, and each job's place in the programmed order */
        for (int c = 0; c < OC; c++)
            Mo[c] = (double)in_scale * (double)w_scale[c] / (double)out_scale;
        int OCp = 0, maxj = 0;
        for (int j0 = 0, n = 0; j0 < nlive; j0 += n) {
            n = dw_perc_job_len(order, j0, nlive, OC, Mo);
            jb[njobs] = OCp;
            jn[njobs] = n;
            njobs++;
            const int np = (n + 31) / 32 * 32;
            OCp += np;
            if (np > maxj) maxj = np;
        }

        if (njobs == 1) {
            /* One job: the channels keep their order, and the run is rocket_conv2d_int8_q's
             * (its one-job fast path included). A host channel carries C = 0 and the host
             * writes its plane after. */
            Ap = malloc((size_t)OC * sizeof(int32_t));
            if (!Ap) { ret = ROCKET_E_NOMEM; goto out; }
            for (int c = 0; c < OC; c++) {
                const int8_t *wc = W + (size_t)c * K;
                int64_t sw = 0;
                for (size_t k = 0; k < K; k++) sw += wc[k];
                const int64_t a = (int64_t)(bias ? bias[c] : 0) - (int64_t)in_zp * sw;
                if (a > INT32_MAX || a < INT32_MIN) {
                    ROCKET_LOGE("rocket_conv2d_int8_q_perc: folded bias of channel %d leaves "
                                "int32\n", c);
                    ret = ROCKET_E_UNSUPPORTED;
                    goto out;
                }
                Ap[c] = (int32_t)a;
            }
            const int c0 = order[0];
            if (perc_gain_check(cvt[c0])) { ret = ROCKET_E_UNSUPPORTED; goto out; }
            const i8q_t q1 = { Ap, cvt[c0], 1.0f, 1.0f, in_zp, 0, out_zp, cmul, shift[c0] };
            ret = conv2d_int8_run(fd, ctx, pool, d, in, W, out, &q1);
            if (ret) goto out;
            goto host_chans;
        }

        rocket_conv2d_desc dp = *d;
        dp.oc = maxj;
        i8plan pln;
        if (conv2d_int8_plan(&dp, &pln)) { ret = ROCKET_E_SHAPE; goto out; }
        const int IC = pln.IC;
        int OCt = pln.OCt, rht = pln.rht, cwt = pln.cwt;
        if (pln.fast) { OCt = maxj; rht = OH; cwt = OW; }   /* one tile a job, halo materialized */

        /* the programmed operands: weights at the padded IC in the sorted order, and A, C */
        Wp = calloc((size_t)OCp * IC * KH * KW, 1);
        Ap = calloc((size_t)OCp, sizeof(int32_t));
        Cp = calloc((size_t)OCp, sizeof(int16_t));
        qs = calloc((size_t)njobs, sizeof(i8q_t));
        tmp = malloc((size_t)OCp * hw);
        if (!Wp || !Ap || !Cp || !qs || !tmp) { ret = ROCKET_E_NOMEM; goto out; }
        for (int j = 0, j0 = 0; j < njobs; j0 += jn[j], j++) {
            for (int i = 0; i < jn[j]; i++) {
                const int c = order[j0 + i], o = jb[j] + i;
                const int8_t *wc = W + (size_t)c * K;
                int64_t sw = 0;
                /* [ICr][KH][KW] leads the padded [IC][KH][KW] block, so one copy a channel */
                memcpy(Wp + (size_t)o * IC * KH * KW, wc, K);
                for (size_t k = 0; k < K; k++) sw += wc[k];
                const int64_t a = (int64_t)(bias ? bias[c] : 0) - (int64_t)in_zp * sw;
                if (a > INT32_MAX || a < INT32_MIN) {
                    ROCKET_LOGE("rocket_conv2d_int8_q_perc: folded bias of channel %d leaves "
                                "int32\n", c);
                    ret = ROCKET_E_UNSUPPORTED;
                    goto out;
                }
                Ap[o] = (int32_t)a;
                Cp[o] = cmul[c];
            }
            const int c0 = order[j0];
            if (perc_gain_check(cvt[c0])) { ret = ROCKET_E_UNSUPPORTED; goto out; }
            qs[j] = (i8q_t){ Ap, cvt[c0], 1.0f, 1.0f, in_zp, 0, out_zp, Cp, shift[c0] };
        }

        /* every job's (oc tile x row band x col band) tiles, each with the job's form */
        const int nRmax = (OH + rht - 1) / rht, nCmax = (OW + cwt - 1) / cwt;
        rband = malloc((size_t)nRmax * sizeof(int));
        cband = malloc((size_t)nCmax * sizeof(int));
        int ntiles = 0;
        for (int j = 0; j < njobs; j++) {
            const int np = (jn[j] + 31) / 32 * 32;
            ntiles += (np + OCt - 1) / OCt;
        }
        ntiles *= nRmax * nCmax;
        tiles = malloc((size_t)ntiles * sizeof(i8tile));
        if (!rband || !cband || !tiles) { ret = ROCKET_E_NOMEM; goto out; }
        const int nR = balance_bands(OH, rht, rband), nC = balance_bands(OW, cwt, cband);
        int ti = 0;
        for (int j = 0; j < njobs; j++) {
            const int np = (jn[j] + 31) / 32 * 32;
            for (int oc0 = jb[j]; oc0 < jb[j] + np; oc0 += OCt) {
                const int OCn = jb[j] + np - oc0 < OCt ? jb[j] + np - oc0 : OCt;
                for (int ri = 0, r0 = 0; ri < nR; r0 += rband[ri], ri++)
                    for (int ci = 0, c0 = 0; ci < nC; c0 += cband[ci], ci++)
                        tiles[ti++] = (i8tile){ oc0, OCn, r0, rband[ri], c0, cband[ci], &qs[j] };
            }
        }
        const int IHs = (rht - 1) * sy + (KH - 1) * dy + 1;
        const int IWs = (cwt - 1) * sx + (KW - 1) * dx + 1;
        ret = conv2d_int8_dispatch(fd, ctx, pool, IC, ICr, IH, IW, OH, OW, KH, KW, sy, sx, pt,
                                   pl, dy, dx, in, Wp, tmp, tiles, ti,
                                   (size_t)IC * IHs * IWs, (size_t)OCt * rht * cwt, &qs[0]);
        if (ret) goto out;
        for (int j = 0, j0 = 0; j < njobs; j0 += jn[j], j++)
            for (int i = 0; i < jn[j]; i++)
                memcpy(out + (size_t)order[j0 + i] * hw, tmp + (size_t)(jb[j] + i) * hw, hw);
    }
host_chans:
    {
        int nh = 0;
        for (int c = 0; c < OC; c++) {
            if (!host[c]) continue;
            conv_i8_ref_chan(d, in, W, bias, in_scale, w_scale[c], out_scale, in_zp, out_zp, c,
                             out);
            nh++;
        }
        ROCKET_LOGD("rocket_conv2d_int8_q_perc: OC=%d IC=%d %dx%d k%dx%d, %d jobs, %d of %d "
                    "channels on the host\n", OC, ICr, IH, IW, KH, KW, njobs, nh, OC);
    }
out:
    free(order); free(jb); free(jn); free(host); free(shift); free(cmul); free(cvt); free(Mo);
    free(Wp); free(Ap); free(Cp); free(qs); free(tmp); free(tiles); free(rband); free(cband);
    return ret;
}

int rocket_conv2d_int8_q_perc_plan(const rocket_conv2d_desc *d)
{
    return rocket_conv2d_int8_q_plan(d);
}

int rocket_conv2d_int8_q_perc_plan_channels(const rocket_conv2d_desc *d, const int8_t *W,
                                            const int32_t *bias, float in_scale,
                                            const float *w_scale, float out_scale, int in_zp,
                                            int16_t *cmul, uint8_t *shift, float *cvt,
                                            uint8_t *host)
{
    int nlive = 0;
    if (!d || !W || !w_scale || !cmul || !shift || !cvt || !host || d->oc <= 0)
        return ROCKET_E_SHAPE;
    int *order = malloc((size_t)d->oc * sizeof(int));
    if (!order) return ROCKET_E_NOMEM;
    int ret = perc_plan_layer(d->oc, (size_t)d->ic * d->kh * d->kw, W, bias, in_scale, w_scale,
                              out_scale, in_zp, d->oc, order, &nlive, cmul, shift, cvt, host);
    free(order);
    return ret;
}

static int conv2d_int8_q_perc_entry(int fd, rocket_conv_ctx *ctx, rocket_conv_pool *pool,
                                    const rocket_conv2d_desc *d, const int8_t *in,
                                    const int8_t *W, const int32_t *bias, float in_scale,
                                    const float *w_scale, float out_scale, int in_zp,
                                    int out_zp, int8_t *out)
{
    if (conv_wrong_encoding("rocket_conv2d_int8_q_perc", "the chip's own conv encoder"))
        return ROCKET_E_UNSUPPORTED;
    if (rocket_conv2d_int8_q_plan(d) != ROCKET_OK) return ROCKET_E_SHAPE;
    return conv2d_int8_q_perc_run(fd, ctx, pool, d, in, W, bias, in_scale, w_scale, out_scale,
                                  in_zp, out_zp, out);
}

int rocket_conv2d_int8_q_perc(int fd, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *W, const int32_t *bias,
                              float in_scale, const float *w_scale, float out_scale,
                              int in_zp, int out_zp, int8_t *out)
{
    return conv2d_int8_q_perc_entry(fd, NULL, NULL, d, in, W, bias, in_scale, w_scale,
                                    out_scale, in_zp, out_zp, out);
}

int rocket_conv2d_int8_q_perc_ctx(rocket_conv_ctx *ctx, const rocket_conv2d_desc *d,
                                  const int8_t *in, const int8_t *W, const int32_t *bias,
                                  float in_scale, const float *w_scale, float out_scale,
                                  int in_zp, int out_zp, int8_t *out)
{
    if (!ctx) return ROCKET_E_SHAPE;
    return conv2d_int8_q_perc_entry(ctx->fd, ctx, NULL, d, in, W, bias, in_scale, w_scale,
                                    out_scale, in_zp, out_zp, out);
}

int rocket_conv2d_int8_q_perc_mt(rocket_conv_pool *pool, const rocket_conv2d_desc *d,
                                 const int8_t *in, const int8_t *W, const int32_t *bias,
                                 float in_scale, const float *w_scale, float out_scale,
                                 int in_zp, int out_zp, int8_t *out)
{
    if (!pool || pool->n < 1) return ROCKET_E_SHAPE;
    return conv2d_int8_q_perc_entry(pool->fd[0], pool->ctx[0], pool, d, in, W, bias, in_scale,
                                    w_scale, out_scale, in_zp, out_zp, out);
}
