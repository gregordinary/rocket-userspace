// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_conv2d_rk3576.c — CONV_2D for the RK3576, behind the library's own entries.
 *
 * The register encoders in npu_regcmd_rk3576.c compute bit-exactly on this part, and
 * for a long time they were reachable only from a test harness: everything a caller
 * needs between a row-major tensor and a submit — the BO management, the operand
 * scatter, the tiling, the output de-scatter, and the submit-loop discipline the part
 * demands — lived in tests/rk3576_conv_gate.c. That made the reverse engineering done
 * and the chip unusable at the same time. This file is that layer.
 *
 * WHAT THE PART COMPUTES, and so what these entries expose:
 *
 *   DIRECT int8 -> int8, requantized ON CHIP by the DPU. That is the native shape of
 *   this datapath, not a convenience: the DPU's OUT_CVT applies the conv scale and
 *   writes a byte. rocket_conv2d_int8_rk3576() therefore takes the quant parameters
 *   and writes int8, where the RK3588's rocket_conv2d_int8() writes a raw int32
 *   accumulator. Same reasoning, and the same answer, as the matmul: a per-chip entry
 *   states the semantics the chip has rather than emulating another chip's.
 *
 *   DEPTHWISE int8 -> int8, likewise, and rocket_conv2d_dw_int8()'s public signature
 *   already IS that shape, so that entry dispatches here rather than refusing.
 *
 *   fp16 -> fp16, through the input-channel split. One fp16 task contracts exactly
 *   sixteen input channels, so an arbitrary channel count is ic/16 submits summed on
 *   the host.
 *
 * WHAT TILES, AND WHY IT MOVES THE ENVELOPE. Rows split through
 * rocket_rk3576_plan_rows() exactly as the harness drives them. Output channels split
 * here, and that is new: the resident weight slice is 32*ic*kh*kw bytes — one
 * output-channel GROUP, independent of oc — but the slice the part tolerates is a
 * function of how many groups the conv drives, 144 KiB at four or more and rising as
 * the count falls. So a conv the single-program emitter refuses for its slice computes
 * when its output channels are split into fewer groups per submit, at a cost of one
 * submit per tile. The planner below picks the largest tile whose group count still
 * admits the slice. tests/rk3576_conv_lib_gate.c is what says whether that holds: it
 * runs the emitter gate's own shape table through these entries, and the shapes the
 * emitter refuses for their weight slice are exactly the ones expected to compute here.
 *
 * WHAT IS REFUSED, and it is refused rather than approximated:
 *
 *   ic <= 4 takes the CNA's ARGB first-conv sub-encoding, whose register program is
 *   transcribed and gated but whose WEIGHT CUBE is not decoded. A capture carries a
 *   register program, not the memory it addresses, and this is the memory.
 *
 *   Dilation. conv_params_t carries the fields and no RK3576 shape has been run
 *   through them, so they are not claimed.
 *
 *   A weight zero point on the depthwise path. The depthwise coefficient group has no
 *   B field at all, and the correction is not a per-channel constant, so it cannot be
 *   folded into the bias either.
 *
 *   A shape whose weight slice does not fit even one output-channel group. The
 *   recourse there is an input-channel split, which the on-chip requant forecloses:
 *   int8 partials cannot be summed without quantizing each one. The int32-output
 *   writer is where that shape belongs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"
#include "rocket_chain.h"

#define C2     16u      /* int8 feature/output channel atom */
#define C2F     8u      /* fp16 feature/output channel atom */

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);
int weight_conv_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

/* ============================================================================
 * SECTION — shared plumbing
 * ==========================================================================*/
struct r76_conv_bos {
    rocket_bo in, w, coeff, out, rc;
};

static void r76_conv_free(int fd, struct r76_conv_bos *b)
{
    if (b->rc.ptr)    rocket_bo_free(fd, &b->rc);
    if (b->out.ptr)   rocket_bo_free(fd, &b->out);
    if (b->coeff.ptr) rocket_bo_free(fd, &b->coeff);
    if (b->w.ptr)     rocket_bo_free(fd, &b->w);
    if (b->in.ptr)    rocket_bo_free(fd, &b->in);
    memset(b, 0, sizeof *b);
}

static int r76_is_this_chip(const char *entry)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    if (hw && hw->name && !strcmp(hw->name, "rk3576")) return 1;
    ROCKET_LOGE("%s emits the RK3576 geometry-register encoding and the active profile "
                "is %s\n", entry, hw && hw->name ? hw->name : "?");
    return 0;
}

/* The shape checks that are common to every precision here. `dw` selects the
 * depthwise contract (oc == ic). Returns 0, or a negative rocket_status. */
static int r76_conv_check(const char *entry, int fd, const rocket_conv2d_desc *d,
                          int dw, int argb_ok, unsigned *ow_out, unsigned *oh_out)
{
    int ow, oh;

    if (fd < 0 || !d) return ROCKET_E_SHAPE;
    if (!r76_is_this_chip(entry)) return ROCKET_E_UNSUPPORTED;
    if (d->ic <= 0 || d->oc <= 0 || d->ih <= 0 || d->iw <= 0 ||
        d->kh <= 0 || d->kw <= 0 || d->stride_y <= 0 || d->stride_x <= 0 ||
        d->pad_top < 0 || d->pad_left < 0)
        return ROCKET_E_SHAPE;
    if ((d->dil_y && d->dil_y != 1) || (d->dil_x && d->dil_x != 1)) {
        ROCKET_LOGE("%s: dilation is not claimed on this part — no RK3576 shape has "
                    "been run through the CONV_CON3 rate fields\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    /* The packed-image first conv is a different program with a different feature
     * buffer and a different weight cube, so the entries that own those say so by
     * passing argb_ok; anything else reaching here with four or fewer channels would
     * be driving the normal path's buffers at a channel count it cannot contract. */
    if (d->ic <= 4 && !argb_ok) {
        ROCKET_LOGE("%s: %d input channels takes the CNA's ARGB first-conv "
                    "sub-encoding, which this entry does not own. Both precisions of it "
                    "run: rocket_conv2d_int8_rk3576() and rocket_conv2d_fp16_rk3576()\n",
                    entry, d->ic);
        return ROCKET_E_UNSUPPORTED;
    }
    if (d->ic <= 4 && dw) {
        ROCKET_LOGE("%s: the first conv has no depthwise form — the channel fold leaves "
                    "nothing to be depthwise over\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    if (dw && d->oc != d->ic) {
        ROCKET_LOGE("%s: a depthwise conv has oc == ic (got oc=%d ic=%d)\n",
                    entry, d->oc, d->ic);
        return ROCKET_E_SHAPE;
    }
    if (d->pad_top > 255 || d->pad_left > 255) return ROCKET_E_SHAPE;

    ow = rocket_conv2d_ow(d);
    oh = rocket_conv2d_oh(d);
    if (ow <= 0 || oh <= 0) return ROCKET_E_SHAPE;
    *ow_out = (unsigned)ow;
    *oh_out = (unsigned)oh;
    return ROCKET_OK;
}

/* Where one row task's own output lives, in bytes, so "did it write" can be asked of
 * exactly that task. The output cube is NC1HWC2, so a row run is one span per channel
 * group at that group's own surface offset — scanning only the first span would call a
 * task written on channel group 0 alone. */
struct r76_task_extent {
    unsigned groups;      /* output channel groups the surface carries  */
    size_t   group_bytes; /* one channel group's whole surface          */
    size_t   row_off;     /* this task's first row within a group       */
    size_t   span;        /* this task's rows within a group            */
};

/* A span still holding the stamp everywhere was never written.
 *
 * PER TASK, not per tile: one poisoned submit among several leaves its own rows stale
 * while its siblings are full, so a whole-tile check reads "something was written" and
 * passes the hole straight through to the caller. */
static int r76_task_wrote(const unsigned char *o, const struct r76_task_extent *e,
                          unsigned char stamp)
{
    unsigned g;
    for (g = 0; g < e->groups; g++) {
        const unsigned char *p = o + (size_t)g * e->group_bytes + e->row_off;
        size_t i;
        for (i = 0; i < e->span; i++)
            if (p[i] != stamp) return 1;
    }
    return 0;
}

/* Env-gated phase timing for the fp16 input-channel split. On-chip accumulation across
 * the slices would remove the READBACK and nothing else — the slice count, and so the
 * wide-output submits and the poisoning retries they carry, is set by the sixteen-channel
 * contraction either way. So the lever is only worth building if the readback is a real
 * share of the wall, and this is what says. ROCKET_RK3576_FP16_PROF=1 logs one line per
 * call, at ROCKET_LOG_INFO. */
struct r76_fp16_prof {
    int      on;
    unsigned slices;
    double   pack_us, stamp_us, submit_us, read_us;
};

static double r76_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static int r76_fp16_prof_on(void)
{
    const char *e = getenv("ROCKET_RK3576_FP16_PROF");
    return (e && *e && *e != '0');
}

/* How long to let a surface drain before calling it unwritten, in microseconds.
 * ROCKET_RK3576_DRAIN_US sets it; DEFAULT 0, because it is a MEASURED NEGATIVE — see
 * r76_task_wrote_late() below. */
static int r76_drain_us(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_DRAIN_US");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) : 0;
        if (cached < 0) cached = 0;
    }
    return cached;
}

/* How many times to redo a row task that wrote nothing. ROCKET_RK3576_TASK_ATTEMPTS.
 *
 * THE POWER CYCLE THE REDO WAITS FOR CLEARS THE POISONING ABOUT 87% OF THE TIME, so the
 * count is what makes the guard reliable, not a better cycle. Over twelve fp16 gate runs
 * the redo fired on 358 row tasks and the attempt that failed next was attempt 2 for 45
 * of them, 3 for 10, 4 for 2, 5 for 1 and 6 for none — every one of them recovering, and
 * each cycle CONFIRMED to have taken the domain to `suspended` first. At four attempts
 * that is about 0.6% of retried tasks returned as a device error, which is the rate the
 * conv gate saw. Eight matches the matmul path's R76_I32_TASK_ATTEMPTS, and costs
 * nothing on a task that succeeds. [HW sweep, H96 MAX M9] */
static unsigned r76_task_attempts(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_TASK_ATTEMPTS");
        cached = (e && *e) ? (int)strtol(e, NULL, 0) : 8;
        if (cached < 1) cached = 1;
    }
    return (unsigned)cached;
}

/* Ask the surface again after a settle. A task whose DPU output element is wider than
 * one byte raises no DPU completion on this part, so the driver retires it on PC_DONE
 * plus a blind grace — and PC_DONE means the program counter finished ISSUING, not that
 * the writes have landed. The fence can therefore signal while the surface is still
 * draining, and a surface that arrives late is a completion-visibility fact rather than
 * the poisoning. It is asked BEFORE the power cycle because the cycle cannot fix it and
 * costs four orders of magnitude more.
 *
 * IT RESCUES NOTHING, and that is the result: at a 2 ms settle, 0 of 67 row tasks that
 * read unwritten had arrived by the time it looked again, while the power cycle behind
 * it recovered all 67. So a task that reads unwritten on this path really is unwritten,
 * the fence is not signalling ahead of the writes, and the 14% wall this costs when it
 * is on buys nothing. Off by default; the knob is kept because it is the instrument
 * that settled it. [HW sweep, H96 MAX M9] */
/* 1 if every task's own rows landed. `first_missing` names the earliest that did not,
 * which is what separates "the stream was poisoned" (none of them) from "the program
 * counter ran one task and stopped" (all but the first). */
static int r76_all_wrote(const unsigned char *o, const struct r76_task_extent *e,
                         unsigned ne, unsigned char stamp, unsigned *first_missing)
{
    unsigned i, missing = 0;
    int ok = 1;
    for (i = 0; i < ne; i++)
        if (!r76_task_wrote(o, &e[i], stamp)) {
            if (ok) missing = i;
            ok = 0;
        }
    if (first_missing) *first_missing = missing;
    return ok;
}

static int r76_task_wrote_late(int fd, struct r76_conv_bos *b,
                               const struct r76_task_extent *e, unsigned ne,
                               unsigned char stamp)
{
    struct timespec ts;
    int us = r76_drain_us(), wrote;

    if (!us) return 0;
    ts.tv_sec = us / 1000000;
    ts.tv_nsec = (long)(us % 1000000) * 1000L;
    nanosleep(&ts, NULL);

    rocket_bo_prep(fd, &b->out, 0, 0);
    wrote = r76_all_wrote((const unsigned char *)b->out.ptr, e, ne, stamp, NULL);
    rocket_bo_fini(fd, &b->out);
    return wrote;
}

/* Whether the row tasks of one output-channel tile go out as ONE submit.
 *
 * The per-submit floor on this part is ~439 us and a row-windowed convolution is one
 * submit per window, so a plane that plans into n windows pays it n times for work the
 * program counter could issue back to back. The row tasks of one tile are independent by
 * construction — each writes its own rows of the same surface, reads its own window of
 * the same feature cube, and the weight and coefficient buffers do not change across
 * them — so concatenating their programs into one regcmd stream is arithmetically sound.
 *
 * A CONCATENATED STREAM IN ONE DRM TASK DOES NOT RUN, and the mechanism is decoded. A
 * drm task descriptor carries ONE PC program however many the stream holds: the driver
 * programs PC_TASK_CON with TASK_NUMBER = 1 per descriptor, so the program counter
 * executes the first task and stops with the rest of the stream unexecuted. Measured
 * with a per-task write check that names the earliest task missing: at 3 tasks and at 6,
 * task 0 lands and task 1 is the first missing, on every one of eight attempts with the
 * power domain confirmed cycled between them — deterministic, and not the poisoning.
 *
 * WHAT COLLECTS IT IS THE JOINT LAYOUT CONTRACT the RK3588 already uses: userspace lays
 * the n programs out contiguously at a fixed stride and rewrites each trailer to point
 * the PC at the next (rocket_chain.c), submits them as n drm task descriptors with
 * DRM_ROCKET_JOB_BATCHED, and the kernel programs TASK_NUMBER = n so the PC streams them
 * from ONE kick and raises ONE completion. That is what makes it a lever rather than an
 * ioctl saving — the completion poll IS the ~439 us floor, and n row tasks then pay it
 * once. The kernel half is patches/rk3576/npu/0015 (extensible submit descriptors) and
 * 0016 (the flag); rocket_batched_submit_supported() refuses to self-chain without it,
 * because a chained layout run down the per-task path stalls.
 *
 * ROCKET_RK3576_BATCH_TASKS=1 turns it on. [HW sweep, H96 MAX M9] */
static int r76_batch_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ROCKET_RK3576_BATCH_TASKS");
        int want = (e && *e) ? (*e != '0') : 0;
        if (want && !rocket_batched_submit_supported()) {
            ROCKET_LOGW("ROCKET_RK3576_BATCH_TASKS=1 but this kernel does not honor "
                        "DRM_ROCKET_JOB_BATCHED (needs patches/rk3576/npu/0015-0016; "
                        "the driver must report >= 1.1). One submit per row task.\n");
            want = 0;
        }
        cached = want;
    }
    return cached;
}

/* The row tasks one chained job may carry. Every plan on this part is bounded by the
 * output height, so this is far above anything the row planner emits; a shape that
 * somehow exceeded it runs one submit per task instead. */
#define R76_MAX_CHAIN_TASKS 512u

/* Lay `ne` programs of `task_ops` words each out in the regcmd BO as ONE chained stream
 * and describe them as ne drm tasks. The host buffer holds each program at the fixed
 * RK3576_CONV_TASK_OPS stride (the generator's upper bound); the BO gets them at the
 * chain's own even-word stride with each trailer rewritten to point the PC at the next,
 * and the last task's forward link cleared. The kernel then programs TASK_NUMBER = ne so
 * the PC streams all of them from one kick and raises one completion. */
static void r76_chain_stream(struct r76_conv_bos *b, rocket_task_desc *td,
                             const uint64_t *ops, uint32_t task_ops, unsigned ne)
{
    unsigned t;
    for (t = 0; t < ne; t++)
        rkt_chain_pack(1, &b->rc, td, (int)t,
                       ops + (size_t)t * RK3576_CONV_TASK_OPS, task_ops, 0);
    rkt_chain_seal(1, &b->rc, (int)ne, task_ops);
}

/* Submit one tile's row tasks and satisfy ourselves that each wrote. `ops` holds `ne`
 * programs of `task_ops` words; they go out as one chained job when batching is on and
 * as one submit each when it is not (ne is then 1). The retry is what covers the
 * poisoning an int32-output job leaves behind — it crosses calls and processes, so a
 * conv inherits it from whatever ran before — and the idle in front of the redo is the
 * NPU power domain cycling rather than a settling time. */
static int r76_submit_ops(int fd, struct r76_conv_bos *b, const uint64_t *ops,
                          uint32_t task_ops, const uint32_t *in_h, unsigned n_in,
                          const uint32_t *out_h, const struct r76_task_extent *e,
                          unsigned ne, unsigned char stamp, uint32_t job_flags,
                          const char *entry)
{
    unsigned attempt, attempts = r76_task_attempts(), missing = 0;
    int cycled = 0, cycles_confirmed = 0;
    int chained = ne > 1u;
    rocket_task_desc td[R76_MAX_CHAIN_TASKS];

    if (chained && ne > R76_MAX_CHAIN_TASKS) {
        ROCKET_LOGE("%s: %u chained tasks exceeds the %u this path lays out\n",
                    entry, ne, (unsigned)R76_MAX_CHAIN_TASKS);
        return ROCKET_E_SHAPE;
    }

    for (attempt = 0; attempt < attempts; attempt++) {
        int srv;

        rocket_bo_prep(fd, &b->rc, 1, 0);
        if (chained)
            r76_chain_stream(b, td, ops, task_ops, ne);
        else
            memcpy(b->rc.ptr, ops, (size_t)task_ops * sizeof(uint64_t));
        rocket_bo_fini(fd, &b->rc);

        srv = chained
            ? rocket_submit_tasks_flags(fd, td, ne, in_h, n_in, out_h, 1,
                                        job_flags | ROCKET_JOB_BATCHED)
            : rocket_submit_matmul_flags(fd, &b->rc, task_ops, in_h, n_in, out_h, 1,
                                         job_flags);
        if (srv != 0) {
            ROCKET_LOGE("%s: submit failed\n", entry);
            return ROCKET_E_DEVICE;
        }
        if (rocket_bo_prep(fd, &b->out, 0, 2000000000ull) < 0) {
            ROCKET_LOGE("%s: PREP_BO on the output timed out\n", entry);
            return ROCKET_E_DEVICE;
        }
        if (!stamp) { rocket_bo_fini(fd, &b->out); return ROCKET_OK; }
        {
            /* EVERY task in the stream, not the stream as a whole: one dead task among
             * several leaves its own rows stale while its siblings land, and a check
             * that asks "did anything change" passes that hole straight to the caller. */
            int wrote = r76_all_wrote((const unsigned char *)b->out.ptr, e, ne, stamp,
                                      &missing);
            rocket_bo_fini(fd, &b->out);
            if (wrote) return ROCKET_OK;
        }
        if (r76_task_wrote_late(fd, b, e, ne, stamp)) {
            ROCKET_LOGD("%s: the surface arrived after the fence, not with it — a "
                        "drain, not the poisoning (attempt %u)\n", entry, attempt + 1u);
            return ROCKET_OK;
        }
        ROCKET_LOGD("%s: of %u row task(s) in this submit the first that wrote nothing "
                    "is %u, on attempt %u; cycling the power domain and redoing it\n",
                    entry, ne, missing, attempt + 1u);
        cycled++;
        cycles_confirmed += rocket_rk3576_power_idle();
    }
    /* Which of the two failures this was, rather than only that it failed: a redo after
     * a CONFIRMED domain collapse that still wrote nothing is not the poisoning, and a
     * redo after an unconfirmed one never had the guard the retry assumes. */
    ROCKET_LOGE("%s: a row task wrote nothing over %u attempts (%d power cycles, "
                "%d of them confirmed to reach suspended)\n",
                entry, attempts, cycled, cycles_confirmed);
    return ROCKET_E_DEVICE;
}

static int r76_submit_task(int fd, struct r76_conv_bos *b, const conv_params_t *q,
                           const uint64_t *ops, const uint32_t *in_h, unsigned n_in,
                           const uint32_t *out_h, const struct r76_task_extent *e,
                           unsigned char stamp, uint32_t job_flags, const char *entry)
{
    return r76_submit_ops(fd, b, ops, q->task_count, in_h, n_in, out_h, e, 1, stamp,
                          job_flags, entry);
}

/* ============================================================================
 * SECTION — the output-channel tile
 *
 * r76_weight_slice_cap() in the emitter is the measured table: a resident weight slice
 * of 144 KiB leaves every output-channel group exact at four groups, 148 KiB at three,
 * 156 KiB at two, and a single group is governed by the CBUF pool alone. The loss is
 * GRADED — the leading groups stay bit-exact and the trailing ones come back wrong — so
 * driving fewer groups per submit is what buys the slice back.
 *
 * This is the inverse of that table: given the slice a shape needs, the most output
 * channels one submit may drive. Splitting there is exact by construction — each tile
 * is an independent convolution over its own channels — and costs one submit per tile.
 * ==========================================================================*/
static unsigned r76_conv_oc_tile(unsigned icreg, unsigned kh, unsigned kw, unsigned oc)
{
    size_t slice = (size_t)32u * icreg * kh * kw;
    unsigned groups;

    if (slice <= 144u * 1024u)      return oc;      /* no group-count constraint */
    else if (slice <= 148u * 1024u) groups = 3u;
    else if (slice <= 156u * 1024u) groups = 2u;
    else                            groups = 1u;    /* the pool check governs */

    return groups * 32u < oc ? groups * 32u : oc;
}

/* ============================================================================
 * SECTION — the int8 convolution, direct and depthwise
 * ==========================================================================*/

/* The zero-point algebra, once, so both paths agree on it.
 *
 * The hardware computes  acc = sum(x_s*w_s) + A + B*sum(x_s)  over ALL taps, with an
 * out-of-bounds tap substituting the border constant, which the emitter programs to the
 * input zero point — so a pad tap's TRUE value is zero and needs no special case.
 *
 * The caller's convolution is  y = sum((x_s - in_zp)*(w_s - w_zp)) + bias, so
 *
 *     B[oc] = -w_zp                       (the DPU ADDS the B term)
 *     A[oc] = bias[oc] - in_zp*sum_w[oc] + in_zp*w_zp*N
 *
 * with sum_w the sum of that output channel's whole filter and N its tap count. Both
 * corrections are pixel-independent, which is what makes them foldable at all. */
/* The output channel a tile's slot j actually carries. Identity unless a per-channel
 * requant has reordered them; see r76_sort_by_scale. */
static unsigned r76_oc_of(const unsigned *perm, unsigned i)
{
    return perm ? perm[i] : i;
}

/*
 * Sort the output channels by their weight scale, ascending.
 *
 * Every output-channel TILE is its own task and so carries its own OUT_CVT shift, and
 * the C ramp inside a tile only has to span THAT tile's range of scales. Left in model
 * order a tile sees the whole layer's spread; sorted, each tile sees roughly the
 * spread's n-th root. That is the difference between a usable per-axis requant and a
 * useless one on a layer with both a wide scale spread and a large fan-in, where the
 * int32 clamp already caps the largest C at a couple of hundred.
 *
 * The permutation is a relabelling of the output axis and nothing else: the weight
 * cube, the bias fold, the C ramp and the de-scatter all read the same slot, so the
 * caller's `out` comes back in the caller's channel order.
 *
 * An insertion sort, because oc is at most a few thousand and this runs once.
 */
static void r76_sort_by_scale(unsigned *perm, unsigned oc, const float *w_scale)
{
    unsigned i, j;
    for (i = 0; i < oc; i++) perm[i] = i;
    for (i = 1; i < oc; i++) {
        unsigned v = perm[i];
        float    s = w_scale[v];
        j = i;
        while (j > 0 && w_scale[perm[j - 1]] > s) { perm[j] = perm[j - 1]; j--; }
        perm[j] = v;
    }
}

static void r76_fold_coeff(int32_t *A, const int32_t *bias, unsigned oc0,
                           unsigned tile_oc, const int64_t *sum_w, int in_zp, int w_zp,
                           unsigned taps, const unsigned *perm)
{
    unsigned j;
    for (j = 0; j < tile_oc; j++) {
        unsigned c = r76_oc_of(perm, oc0 + j);
        int64_t a = bias ? (int64_t)bias[c] : 0;
        a -= (int64_t)in_zp * sum_w[c];
        a += (int64_t)in_zp * w_zp * taps;
        A[j] = (int32_t)a;
    }
}

/*
 * The per-output-channel requant plan.
 *
 * The DPU's epilogue is `(acc + A[oc]) * C[oc]` in saturating int32, then ONE
 * `(v*MUL)>>SHIFT` for the whole task. So a per-axis weight quantization rides on
 * C, and the job here is to pick the one (MUL, SHIFT) and the C ramp that together
 * approximate every channel's own `in_scale*w_scale[oc]/out_scale` as closely as the
 * two hardware bounds allow:
 *
 *   - C is an integer, so channel oc's gain resolution is 0.5/C[oc]. Bigger is better.
 *   - `(acc + A) * C` saturates at INT32_MAX, so C[oc] is capped by that channel's own
 *     worst-case accumulator. Bigger is not free.
 *
 * The cap is computed from the ACTUAL weights rather than from the int8 envelope,
 * because 127*sum|w| over a real filter is one to two orders of magnitude below
 * ic*kh*kw*127*127 and the difference is most of the available precision.
 *
 * Returns the worst-case relative gain error over the channels in `*max_rel_err`.
 */
static int r76_plan_perchannel(const char *entry, unsigned oc0, unsigned tile_oc,
                               unsigned ocreg, const int32_t *A,
                               const int64_t *sum_abs_w, float in_scale,
                               const float *w_scale, float out_scale,
                               const unsigned *perm,
                               int16_t *C, float *base_scale, double *max_rel_err)
{
    double best_base = 0.0;
    unsigned j;

    /* The tightest base gain: every channel must reach its own scale with a C that
     * neither exceeds the int16 field nor saturates the int32 product. */
    for (j = 0; j < tile_oc; j++) {
        unsigned oc_j = r76_oc_of(perm, oc0 + j);
        double cs = (double)in_scale * (double)w_scale[oc_j] / (double)out_scale;
        /* |acc| <= 128*sum|w| (the input is signed int8), and A rides with it. */
        double bound = 128.0 * (double)sum_abs_w[oc_j] + fabs((double)A[j]) + 1.0;
        double cmax = (double)INT32_MAX / bound;
        double need;
        if (!(cs > 0.0)) {
            ROCKET_LOGE("%s: w_scale[%u] is %g — every per-channel scale must be "
                        "positive\n", entry, oc_j, (double)w_scale[oc_j]);
            return -1;
        }
        if (cmax > 32767.0) cmax = 32767.0;
        if (cmax < 1.0)     cmax = 1.0;
        need = cs / cmax;
        if (need > best_base) best_base = need;
    }
    if (!(best_base > 0.0)) return -1;

    /* Quantize the base into the pair the emitter will actually program, then read it
     * back: the C ramp has to be built against the gain the hardware gets, not the one
     * the planner asked for. */
    {
        unsigned mul, shift;
        double base_actual;
        rocket_rk3576_requant_params((float)best_base, &mul, &shift);
        base_actual = (double)mul / (double)((uint64_t)1 << shift);
        *base_scale = (float)best_base;
        *max_rel_err = 0.0;
        for (j = 0; j < ocreg; j++) {
            double cs, want, err;
            long long c;
            if (j >= tile_oc) { C[j] = 1; continue; }   /* a padded channel computes nothing */
            cs   = (double)in_scale * (double)w_scale[r76_oc_of(perm, oc0 + j)] /
                   (double)out_scale;
            want = cs / base_actual;
            c = (long long)(want + 0.5);
            if (c < 1)     c = 1;
            if (c > 32767) c = 32767;
            C[j] = (int16_t)c;
            err = fabs((double)c * base_actual - cs) / cs;
            if (err > *max_rel_err) *max_rel_err = err;
        }
    }
    return 0;
}

/*
 * Pick the output-channel tile on the per-channel path.
 *
 * Everywhere else the tile is a CBUF-fit decision. Here it is also an ACCURACY one,
 * because a tile is one task and a task carries one OUT_CVT shift: the C ramp inside a
 * tile only spans that tile's range of scales, so halving the tile roughly halves the
 * spread the ramp has to cover and the worst channel's gain resolution improves with
 * it. Measured on the part at ic=128 oc=128 with a 100x spread: 26.6 counts of
 * deviation from an exact per-axis requant in one tile, 2.7 at 64 channels, 1.0 at 32.
 *
 * The cost is submits — one row-task set per tile at the part's ~439 us floor — so
 * this takes the LARGEST tile that meets the error target rather than the smallest
 * tile that would fit. The error is predicted from the plan alone, with no submits.
 *
 * ROCKET_RK3576_PC_MAX_ERR sets the target (default 1%); ROCKET_RK3576_PC_OC_TILE
 * forces a tile and skips the search.
 */
static double r76_pc_err_at(unsigned OC, unsigned tile, const unsigned *perm,
                            const int64_t *sum_abs_w, const int32_t *bias,
                            const int64_t *sum_w, int in_zp, int w_zp, unsigned taps,
                            float in_scale, const float *w_scale, float out_scale)
{
    double worst = 0.0;
    unsigned oc0;
    for (oc0 = 0; oc0 < OC; oc0 += tile) {
        unsigned n = OC - oc0 < tile ? OC - oc0 : tile, j;
        double best_base = 0.0, base_actual;
        unsigned mul, shift;
        for (j = 0; j < n; j++) {
            unsigned c = r76_oc_of(perm, oc0 + j);
            double cs = (double)in_scale * (double)w_scale[c] / (double)out_scale;
            double A  = (double)(bias ? bias[c] : 0)
                      - (double)in_zp * (double)sum_w[c]
                      + (double)in_zp * w_zp * taps;
            double bound = 128.0 * (double)sum_abs_w[c] + fabs(A) + 1.0;
            double cmax = (double)INT32_MAX / bound;
            double need;
            if (!(cs > 0.0)) return 1e30;
            if (cmax > 32767.0) cmax = 32767.0;
            if (cmax < 1.0)     cmax = 1.0;
            need = cs / cmax;
            if (need > best_base) best_base = need;
        }
        if (!(best_base > 0.0)) return 1e30;
        rocket_rk3576_requant_params((float)best_base, &mul, &shift);
        base_actual = (double)mul / (double)((uint64_t)1 << shift);
        for (j = 0; j < n; j++) {
            unsigned c = r76_oc_of(perm, oc0 + j);
            double cs = (double)in_scale * (double)w_scale[c] / (double)out_scale;
            long long v = (long long)(cs / base_actual + 0.5);
            double err;
            if (v < 1)     v = 1;
            if (v > 32767) v = 32767;
            err = fabs((double)v * base_actual - cs) / cs;
            if (err > worst) worst = err;
        }
    }
    return worst;
}

static unsigned r76_pc_oc_tile(unsigned OC, unsigned cbuf_tile, const unsigned *perm,
                               const int64_t *sum_abs_w, const int32_t *bias,
                               const int64_t *sum_w, int in_zp, int w_zp, unsigned taps,
                               float in_scale, const float *w_scale, float out_scale)
{
    const char *e = getenv("ROCKET_RK3576_PC_OC_TILE");
    double target = 0.01;
    unsigned tile;

    if (e && *e) {
        long v = strtol(e, NULL, 0);
        if (v > 0) return (unsigned)v < cbuf_tile ? (unsigned)v : cbuf_tile;
    }
    e = getenv("ROCKET_RK3576_PC_MAX_ERR");
    if (e && *e) {
        double v = strtod(e, NULL);
        if (v > 0.0) target = v;
    }
    /* Largest first, so the search stops at the fewest submits that will do. The floor
     * is the 32-channel MAC group; below it a tile is not a shape the emitter takes. */
    for (tile = cbuf_tile; ; tile = tile / 2u > 32u ? tile / 2u : 32u) {
        if (r76_pc_err_at(OC, tile, perm, sum_abs_w, bias, sum_w, in_zp, w_zp, taps,
                          in_scale, w_scale, out_scale) <= target)
            return tile;
        if (tile <= 32u) return 32u;   /* nothing narrower to try */
    }
}

static int r76_conv_int8_run(const char *entry, int fd, const rocket_conv2d_desc *d,
                             int dw, const int8_t *in, const int8_t *W,
                             const int32_t *bias, float in_scale, float w_scale,
                             const float *w_scale_oc,
                             float out_scale, int in_zp, int w_zp, int out_zp,
                             int8_t *out)
{
    struct r76_conv_bos b = {0};
    uint64_t *ops = NULL;
    struct r76_task_extent *ext = NULL;
    rocket_rk3576_row_task *plan = NULL;
    int32_t *A = NULL;
    int64_t *sum_w = NULL;
    int64_t *sum_abs_w = NULL;
    int16_t *B = NULL;
    int16_t *Cmul = NULL;
    unsigned *perm = NULL;
    double   worst_rel_err = 0.0;
    unsigned IC, OC, IH, IW, KH, KW, SY, SX, PT, PL;
    unsigned ow, oh, icreg, icpad, surf_elems, oc_tile, oc0, max_tasks, taps;
    size_t in_bytes;
    unsigned char stamp;
    int rc, batch;

    rc = r76_conv_check(entry, fd, d, dw, 0, &ow, &oh);
    if (rc != ROCKET_OK) return rc;
    if (!in || !W || !out) return ROCKET_E_SHAPE;
    if (w_scale_oc && dw) {
        ROCKET_LOGE("%s: the depthwise coefficient group's C is the only per-channel "
                    "field it has and the direct path's planner is not wired to it\n",
                    entry);
        return ROCKET_E_UNSUPPORTED;
    }
    if (w_scale_oc && w_zp) {
        ROCKET_LOGE("%s: a per-axis weight quantization is symmetric by construction; "
                    "a non-zero weight zero point with per-channel scales is not a "
                    "shape this path models\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    if (!(in_scale > 0.0f) || !(out_scale > 0.0f) ||
        (!w_scale_oc && !(w_scale > 0.0f))) {
        ROCKET_LOGE("%s: the quant scales must be positive — the DPU's OUT_CVT gates "
                    "the whole BS stage off at zero and writes an empty surface\n", entry);
        return ROCKET_E_SHAPE;
    }
    if (in_zp < -128 || in_zp > 127 || w_zp < -128 || w_zp > 127 ||
        out_zp < -128 || out_zp > 127)
        return ROCKET_E_SHAPE;
    /* A DEPTHWISE WEIGHT ZERO POINT RIDES IN THE CUBE, not in the coefficient group.
     * That group really has no B field and the correction really is not a per-channel
     * constant, so folding it into the bias is impossible — but it does not have to be
     * folded at all. This part's int8 depthwise cube gives every (channel, tap) TWO live
     * bytes and the datapath ADDS them, so the effective weight is a nine-bit value and
     * `w - w_zp` is carried exactly by splitting it across the pair. The coefficient A
     * already computes `-in_zp*(sum_w - w_zp*taps)`, which is the fold this pre-centred
     * cube wants, so nothing downstream changes. r76_dw_split() below is that split.
     *
     * What bounds it is the pair's own range: two signed bytes reach [-256, 254], and a
     * `w - w_zp` outside that is refused rather than clamped. It cannot arise from an
     * int8 weight and an int8 zero point without one of them being out of domain.
     * [HW sweep, H96 MAX M9, tests/rk3576_conv_lib_gate.c dwzp] */
    if (dw && w_zp) {
        int lo = -128 - w_zp, hi = 127 - w_zp;
        if (lo < -256 || hi > 254) {
            ROCKET_LOGE("%s: a depthwise weight zero point of %d puts w-w_zp outside "
                        "the [-256, 254] the cube's two live bytes reach\n", entry, w_zp);
            return ROCKET_E_UNSUPPORTED;
        }
    }

    IC = (unsigned)d->ic; OC = (unsigned)d->oc;
    IH = (unsigned)d->ih; IW = (unsigned)d->iw;
    KH = (unsigned)d->kh; KW = (unsigned)d->kw;
    SY = (unsigned)d->stride_y; SX = (unsigned)d->stride_x;
    PT = (unsigned)d->pad_top;  PL = (unsigned)d->pad_left;

    /* Channel counts as told to the REGISTERS. The direct path needs both rounded to the
     * 32-channel MAC group. The depthwise path takes the RAW count — its own two
     * granules (the weight cube rounds to 16, the CBUF allocation sometimes one 16-group
     * further) are the emitter's business, and rounding here would hide every count
     * where the two differ. */
    icreg = dw ? IC : rocket_rk3576_pad_ic(IC);
    icpad = (icreg + 31u) / 32u * 32u;
    surf_elems = rocket_rk3576_out_surf_elems(ow, oh, dw);
    taps = dw ? KH * KW : IC * KH * KW;
    in_bytes = (size_t)((icpad + C2 - 1u) / C2) * IH * IW * C2;
    max_tasks = oh + 2u;
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;

    oc_tile = dw ? OC : r76_conv_oc_tile(icreg, KH, KW, rocket_rk3576_pad_oc(OC));
    if (!oc_tile) return ROCKET_E_SHAPE;

    /* Room for every row task's program at once when they go out as one stream, and
     * one task's when they do not — the plan is bounded by max_tasks either way. */
    batch = r76_batch_on() && max_tasks <= R76_MAX_CHAIN_TASKS;
    ops   = calloc((size_t)(batch ? max_tasks : 1u) * RK3576_CONV_TASK_OPS, sizeof *ops);
    ext   = calloc(max_tasks, sizeof *ext);
    plan  = calloc(max_tasks, sizeof *plan);
    sum_w = calloc(OC, sizeof *sum_w);
    if (w_scale_oc) {
        sum_abs_w = calloc(OC, sizeof *sum_abs_w);
        perm      = calloc(OC, sizeof *perm);
    }
    if (!ops || !ext || !plan || !sum_w || (w_scale_oc && (!sum_abs_w || !perm))) {
        rc = ROCKET_E_NOMEM; goto done;
    }
    /* Sorted by scale, so each output-channel tile — its own task, its own OUT_CVT
     * shift — spans as little of the layer's scale range as the tiling allows. */
    if (perm) r76_sort_by_scale(perm, OC, w_scale_oc);

    /* Each output channel's whole filter, for the input zero point's fold — and its
     * absolute sum, which is what bounds that channel's accumulator and so how large a
     * per-channel C multiplier it can carry. */
    {
        unsigned c, i, y, x;
        for (c = 0; c < OC; c++) {
            int64_t s = 0, sa = 0;
            if (dw) {
                for (y = 0; y < KH; y++)
                    for (x = 0; x < KW; x++) {
                        int64_t v = W[((size_t)c * KH + y) * KW + x];
                        s += v; sa += v < 0 ? -v : v;
                    }
            } else {
                for (i = 0; i < IC; i++)
                    for (y = 0; y < KH; y++)
                        for (x = 0; x < KW; x++) {
                            int64_t v = W[(((size_t)c * IC + i) * KH + y) * KW + x];
                            s += v; sa += v < 0 ? -v : v;
                        }
            }
            sum_w[c] = s;
            if (sum_abs_w) sum_abs_w[c] = sa;
        }
    }

    /* AFTER the filter sums, which is what the accumulator bound is built from: on
     * this path the output-channel tile is an ACCURACY parameter as well as a CBUF
     * one, because a tile is one task and a task carries one OUT_CVT shift. */
    if (w_scale_oc)
        oc_tile = r76_pc_oc_tile(OC, oc_tile, perm, sum_abs_w, bias, sum_w,
                                 in_zp, w_zp, taps, in_scale, w_scale_oc, out_scale);

    /* The FEATURE cube is packed once and shared by every output-channel tile: the
     * tiling is on the output axis, which the feature side does not see. */
    if (rocket_bo_alloc(fd, in_bytes, &b.in) < 0) { rc = ROCKET_E_NOMEM; goto done; }
    rocket_bo_prep(fd, &b.in, 1, 0);
    memset(b.in.ptr, 0, in_bytes);
    /* The channel-group arithmetic is hoisted out of the pixel loop. A CHW tensor and an
     * NC1HWC2 cube are a transpose, not a copy, so this stays a strided store — but
     * calling the index function per element pays a divide and a modulo on every one of
     * IC*IH*IW, and that was most of what a conv on this part spent outside the NPU. */
    {
        int8_t *cube = (int8_t *)b.in.ptr;
        size_t px = (size_t)IH * IW, p;
        unsigned c;
        for (c = 0; c < IC; c++) {
            int8_t *dst = cube + (size_t)(c / C2) * px * C2 + (c % C2);
            const int8_t *src = in + (size_t)c * px;
            for (p = 0; p < px; p++) dst[p * C2] = src[p];
        }
    }
    rocket_bo_fini(fd, &b.in);

    /* One extra word per task: the chain lays each program out at an EVEN word stride
     * (rkt_chain_words), which rounds an odd program length up by one. */
    if (rocket_bo_alloc(fd, (size_t)(batch ? max_tasks : 1u) *
                        (RK3576_CONV_TASK_OPS + 1u) * sizeof(uint64_t), &b.rc) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }

    for (oc0 = 0; oc0 < OC; oc0 += oc_tile) {
        unsigned tile_oc = OC - oc0 < oc_tile ? OC - oc0 : oc_tile;
        unsigned ocreg = dw ? tile_oc : rocket_rk3576_pad_oc(tile_oc);
        size_t w_bytes = dw ? rocket_rk3576_weight_dw_bytes(ocreg, KH, KW)
                            : (size_t)((ocreg + 31u) / 32u) * ((icreg + 31u) / 32u) *
                              32u * 32u * KH * KW;
        size_t coeff_bytes = dw ? rocket_rk3576_coeff_bytes_dw(ocreg)
                                : rocket_rk3576_coeff_bytes(ocreg);
        size_t obytes = (size_t)((ocreg + C2 - 1u) / C2) * surf_elems * C2;
        conv_params_t p = {0};
        uint32_t in_h[4], out_h[1];
        unsigned ntask = 1u, t;
        uint32_t task_ops;
        float tile_base_scale = w_scale;

        if (b.w.ptr)     rocket_bo_free(fd, &b.w);
        if (b.coeff.ptr) rocket_bo_free(fd, &b.coeff);
        if (b.out.ptr)   rocket_bo_free(fd, &b.out);
        memset(&b.w, 0, sizeof b.w);
        memset(&b.coeff, 0, sizeof b.coeff);
        memset(&b.out, 0, sizeof b.out);

        if (rocket_bo_alloc(fd, w_bytes, &b.w) < 0 ||
            rocket_bo_alloc(fd, coeff_bytes, &b.coeff) < 0 ||
            rocket_bo_alloc(fd, obytes, &b.out) < 0) { rc = ROCKET_E_NOMEM; goto done; }

        /* The WEIGHT cube is per-tile and each tile is its own convolution, so its
         * group count follows the tile rather than the whole output-channel count. */
        rocket_bo_prep(fd, &b.w, 1, 0);
        memset(b.w.ptr, 0, w_bytes);
        {
            int8_t *cube = (int8_t *)b.w.ptr;
            unsigned c, i, y, x;
            if (dw) {
                /* This part's own depthwise cube: channels grouped by 64, tap-major
                 * inside a group, two byte slots per weight — and at int8 the byte a
                 * channel owns inside a tap block is 4*(c/2) + (c%2), with its SECOND
                 * two further on. Neither the direct path's cube nor the RK3588's
                 * single-byte one. Both bytes contribute, so the pair carries the
                 * zero-point-centred weight when one byte cannot. */
                for (c = 0; c < tile_oc; c++)
                    for (y = 0; y < KH; y++)
                        for (x = 0; x < KW; x++) {
                            int v = W[(((size_t)r76_oc_of(perm, oc0 + c) * KH) + y) * KW
                                      + x] - w_zp;
                            int lo = v > 127 ? 127 : (v < -128 ? -128 : v);
                            int at = rocket_rk3576_weight_dw_int8(ocreg, KH, KW, c, y, x);
                            cube[at]      = (int8_t)lo;
                            cube[at + 2]  = (int8_t)(v - lo);
                        }
            } else {
                /* The same hoist. weight_conv_int8() at these groups is
                 * (c/32)*nIC1*KH*KW*1024 + (i/32)*KH*KW*1024 + (y*KW + x)*1024 +
                 * (c%32)*32 + (i%32), so the taps of one (channel, input channel) pair
                 * walk a fixed 1024-byte stride from a contiguous source run. */
                unsigned nIC1 = (icreg + 31u) / 32u, t, taps = KH * KW;
                for (c = 0; c < tile_oc; c++) {
                    size_t cbase = (size_t)(c / 32u) * nIC1 * taps * 1024u
                                 + (size_t)(c % 32u) * 32u;
                    for (i = 0; i < IC; i++) {
                        int8_t *dst = cube + cbase
                                    + (size_t)(i / 32u) * taps * 1024u + (i % 32u);
                        const int8_t *src =
                            W + ((size_t)r76_oc_of(perm, oc0 + c) * IC + i) * taps;
                        for (t = 0; t < taps; t++) dst[(size_t)t * 1024u] = src[t];
                    }
                }
            }
        }
        rocket_bo_fini(fd, &b.w);

        /* The COEFFICIENT buffer is NOT a flat int32 bias array on this part, and a
         * zeroed one makes the DPU write a full but entirely empty surface whatever the
         * MAC did — the C term gates the BS stage. The tail channels of a partial group
         * get a zero A term so they carry a C term too. */
        free(A); free(B); free(Cmul);
        A = calloc(ocreg, sizeof *A);
        /* B is the DIRECT path's weight-zero-point term. The depthwise group has no such
         * field and does not need one: its cube is pre-centred above. */
        B = (w_zp && !dw) ? calloc(ocreg, sizeof *B) : NULL;
        Cmul = w_scale_oc ? calloc(ocreg, sizeof *Cmul) : NULL;
        if (!A || (w_zp && !dw && !B) || (w_scale_oc && !Cmul)) {
            rc = ROCKET_E_NOMEM; goto done;
        }
        r76_fold_coeff(A, bias, oc0, tile_oc, sum_w, in_zp, w_zp, taps, perm);
        if (B) {
            unsigned j;
            /* An asymmetric weight is w_true = w_stored - w_zp, whose correction is
             * -w_zp*sum(x), and the DPU ADDS the B term. */
            for (j = 0; j < ocreg; j++) B[j] = (int16_t)(-w_zp);
        }
        /* The per-channel gain is planned per TILE, because the one (MUL, SHIFT) the
         * OUT_CVT carries is per task and each output-channel tile is its own task. */
        if (w_scale_oc) {
            double err = 0.0;
            if (r76_plan_perchannel(entry, oc0, tile_oc, ocreg, A, sum_abs_w,
                                    in_scale, w_scale_oc, out_scale, perm,
                                    Cmul, &tile_base_scale, &err) != 0) {
                rc = ROCKET_E_SHAPE; goto done;
            }
            if (err > worst_rel_err) worst_rel_err = err;
        }
        rocket_bo_prep(fd, &b.coeff, 1, 0);
        if (dw)          rocket_rk3576_pack_coeff_dw(b.coeff.ptr, coeff_bytes, A, ocreg);
        else if (Cmul)   rocket_rk3576_pack_coeff_perc(b.coeff.ptr, coeff_bytes, A, ocreg,
                                                       B, Cmul, 1);
        else if (B)      rocket_rk3576_pack_coeff_asym(b.coeff.ptr, coeff_bytes, A, ocreg,
                                                       B, 1);
        else             rocket_rk3576_pack_coeff(b.coeff.ptr, coeff_bytes, A, ocreg);
        rocket_bo_fini(fd, &b.coeff);

        p.ic = (uint16_t)icreg; p.ih = (uint16_t)IH; p.iw = (uint16_t)IW;
        p.oc = (uint16_t)ocreg; p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
        p.kh = (uint16_t)KH;    p.kw = (uint16_t)KW;
        p.stride_y = (uint8_t)SY; p.stride_x = (uint8_t)SX;
        p.pad_top  = (uint8_t)PT; p.pad_left = (uint8_t)PL;
        p.ih_full = (uint16_t)IH; p.oh_full = (uint16_t)oh;
        p.int8_out = 1;
        /* Per-channel: the C multipliers carry every channel's gain RELATIVE to one
         * base, and the base is what the OUT_CVT programs — so it is handed over whole
         * rather than as the in/w/out triple the per-tensor path decomposes into. */
        if (w_scale_oc) {
            p.in_scale = tile_base_scale; p.w_scale = 1.0f; p.out_scale = 1.0f;
        } else {
            p.in_scale = in_scale; p.w_scale = w_scale; p.out_scale = out_scale;
        }
        /* Both zero points reach the registers uint8-centered: the emitter programs the
         * border constant as (input_zero_point & 0xff) - 0x80 and the output offset as
         * output_zero_point - 0x80, so a model-domain signed zero point is that value
         * plus 0x80. The weight zero point rides in the coefficient group's B term
         * instead and this field is inert on the RK3576 path. */
        p.input_zero_point  = in_zp  + 0x80;
        p.output_zero_point = out_zp + 0x80;
        p.weight_zero_point = 0x80;
        p.tasks       = ops;
        p.input_dma   = b.in.dma_address;
        p.weights_dma = b.w.dma_address;
        p.bias_dma    = b.coeff.dma_address;
        p.output_dma  = b.out.dma_address;

        {
            conv_params_t q = p;
            if (rocket_rk3576_plan_rows(&q, dw, plan, max_tasks, &ntask) < 0) {
                ROCKET_LOGE("%s: no row plan for ic=%u oc tile %u (%ux%u k%ux%u s%u) — "
                            "the recourse is an input-channel split, which this path's "
                            "on-chip requant forecloses\n",
                            entry, IC, tile_oc, IW, IH, KW, KH, SX);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }
        }

        in_h[0] = b.in.handle; in_h[1] = b.w.handle;
        in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
        out_h[0] = b.out.handle;

        if (stamp) {
            rocket_bo_prep(fd, &b.out, 1, 0);
            memset(b.out.ptr, stamp, obytes);
            rocket_bo_fini(fd, &b.out);
        }

        /* The row tasks of this tile, as one stream when batching is on and as one
         * submit each when it is not. Independent by construction: task t reads its own
         * window of the shared feature cube and writes its own rows of the shared
         * surface, and the weight and coefficient buffers do not change across them. */
        task_ops = 0;
        for (t = 0; t < ntask; t++) {
            conv_params_t q = p;
            q.ih = plan[t].ih; q.oh = plan[t].oh;
            q.pad_top = plan[t].pad_top;
            q.input_dma  = p.input_dma  + plan[t].feature_off;
            q.output_dma = p.output_dma + plan[t].output_off;
            /* ALWAYS, not only when the plan split. A single-task plan's window is still
             * shorter than the plane whenever the last output row does not reach the
             * bottom input row — ordinary stride-2 VALID geometry — and leaving these at
             * the window makes the emitter derive the DDR channel-group stride from the
             * WINDOW, so every group past the first reads at the wrong offset and the
             * surface comes back unrelated to the input, with nothing to fault on. */
            q.ih_full = (uint16_t)IH; q.oh_full = (uint16_t)oh;
            q.tasks = ops + (size_t)(batch ? t : 0u) * RK3576_CONV_TASK_OPS;
            if ((dw ? gen_conv2d_dw_int8_rk3576(&q) : gen_conv2d_int8_rk3576(&q)) != 0) {
                ROCKET_LOGE("%s: the generator refused task %u of %u\n", entry, t, ntask);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }
            /* The chain lays every program out at ONE stride, so a tile whose tasks
             * somehow differ in length cannot be chained. They do not differ — the row
             * tasks of a tile write the same registers with different values — but
             * ROCKET_RK3576_ADD can lengthen a program, so check rather than assume. */
            if (!t) task_ops = q.task_count;
            else if (batch && q.task_count != task_ops) {
                ROCKET_LOGE("%s: task %u is %u ops against task 0's %u, so this tile "
                            "cannot be chained\n", entry, t, q.task_count, task_ops);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }
            ext[t].groups      = (ocreg + C2 - 1u) / C2;
            ext[t].group_bytes = (size_t)surf_elems * C2;
            ext[t].row_off     = (size_t)plan[t].oy0 * ow * C2;
            ext[t].span        = (size_t)plan[t].oh * ow * C2;
            if (!batch) {
                rc = r76_submit_ops(fd, &b, ops, q.task_count, in_h, 4u, out_h,
                                    &ext[t], 1u, stamp, 0u, entry);
                if (rc != ROCKET_OK) goto done;
            }
        }
        if (batch) {
            rc = r76_submit_ops(fd, &b, ops, task_ops, in_h, 4u, out_h, ext, ntask,
                                stamp, 0u, entry);
            if (rc != ROCKET_OK) goto done;
        }

        /* De-scatter this tile's channels straight into the caller's row-major out. */
        rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
        /* The same hoist on the way back: y*ow + x is the pixel index, contiguous in the
         * CHW output and at a fixed C2 stride in the cube. */
        {
            const int8_t *o = (const int8_t *)b.out.ptr;
            size_t px = (size_t)oh * ow, p;
            unsigned c;
            for (c = 0; c < tile_oc; c++) {
                const int8_t *src = o + (size_t)(c / C2) * surf_elems * C2 + (c % C2);
                int8_t *dst = out + (size_t)r76_oc_of(perm, oc0 + c) * px;
                for (p = 0; p < px; p++) dst[p] = src[p * C2];
            }
        }
        rocket_bo_fini(fd, &b.out);
    }
    rc = ROCKET_OK;

done:
    free(ops); free(ext); free(plan); free(A); free(B); free(Cmul);
    free(sum_w); free(sum_abs_w); free(perm);
    if (w_scale_oc && rc == ROCKET_OK)
        ROCKET_LOGI("%s: per-channel requant, worst-case gain error %.3g%%\n",
                    entry, worst_rel_err * 100.0);
    r76_conv_free(fd, &b);
    return rc;
}

/* ============================================================================
 * SECTION — the INT8 first conv, on the packed-image datapath
 *
 * The quantized stem. Same CNA sub-encoding as the fp16 first conv and a different
 * object in three places, each of which had to be read off the part:
 *
 *   - THE WEIGHT CUBE is single bytes in output-channel groups of 32 with the tap row
 *     outside the group and the tap column folded in beside the four lanes, where the
 *     float cube is 16-bit slots in groups of sixteen with the tap axis outermost.
 *     rocket_rk3576_argb_int8_pack_weights() owns it.
 *   - THE LEFT PAD MUST BE NON-ZERO and THE OUTPUT WIDTH MUST BE iw/stride. Neither
 *     holds on the float path, and neither appears in any capture, because every
 *     captured first conv is a SAME convolution and satisfies both by construction. A
 *     zero left pad writes NOTHING; a wrong output width writes a full surface sheared
 *     by one column per row. Both are refused here rather than computed wrong.
 *   - ONE PROGRAM DELIVERS 64 OUTPUT CHANNELS, not the float path's 32.
 *
 * Everything else is the direct int8 path's: the zero-point fold, the coefficient
 * group, the row window, the NC1HWC2 output and its de-scatter.
 * ==========================================================================*/
/* Output channels one program drives. 32 and 64 are measured — every live weight byte
 * of a 64-channel cube lands on exactly one output position of one channel — and above
 * that the split is the caller's, exactly as on the float path. [HW sweep] */
#define R76_ARGB_INT8_OC_MAX 64u

static int r76_conv_int8_argb(const char *entry, int fd, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *W, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out,
                              unsigned ow, unsigned oh)
{
    struct r76_conv_bos b = {0};
    uint64_t *ops = NULL;
    rocket_rk3576_row_task *rows = NULL;
    int32_t *A = NULL;
    int16_t *B = NULL;
    int64_t *sum_w = NULL;
    int8_t *wtile = NULL;
    unsigned IC = (unsigned)d->ic, OC = (unsigned)d->oc;
    /* ONE image channel is programmed as TWO. The int8 first conv writes nothing at
     * ic=1 — an untouched surface, not a wrong one — and what gates it is the feature
     * DMA's row width: the emitted `line_stride - 1` is correct from ic=2 up, and at
     * ic=1 the program only writes when that field is raised to `line_stride`, at
     * which point the DMA reads past the packed row and nothing is exact. So the row
     * is widened rather than the register: a second interleaved channel of zero
     * samples against zero weights. The arithmetic is untouched by it — the MAC term
     * is zero because the weight is, `sum_w` is unchanged so the coefficient A is,
     * and the asymmetric B multiplies a sum of RAW samples that gains only zeros — so
     * `taps` below stays the caller's count. The cost is one byte per pixel of host
     * packing and a doubled feature read. */
    unsigned ICP = ((unsigned)d->ic == 1u) ? 2u : (unsigned)d->ic;
    unsigned IH = (unsigned)d->ih, IW = (unsigned)d->iw;
    unsigned KH = (unsigned)d->kh, KW = (unsigned)d->kw;
    unsigned SY = (unsigned)d->stride_y, SX = (unsigned)d->stride_x;
    unsigned PT = (unsigned)d->pad_top,  PL = (unsigned)d->pad_left;
    unsigned surf_elems = rocket_rk3576_out_surf_elems(ow, oh, 0);
    unsigned tile = OC < R76_ARGB_INT8_OC_MAX ? OC : R76_ARGB_INT8_OC_MAX;
    unsigned taps = IC * KH * KW;
    unsigned oc0, nrow = 0, r, max_tasks = oh + 2u;
    size_t in_bytes;
    conv_params_t plan = {0};
    unsigned char stamp;
    int rc;

    if (IW % 16u) {
        ROCKET_LOGE("%s: the first conv needs iw a multiple of 16 (iw=%u); its DDR row "
                    "stride and CBUF row are both counted in 16-byte granules\n",
                    entry, IW);
        return ROCKET_E_UNSUPPORTED;
    }
    /* The two bounds this path adds, both measured and both silent if violated. */
    if (PL == 0) {
        ROCKET_LOGE("%s: the int8 first conv needs a NON-ZERO left pad (pad_left=0). At "
                    "pad_left 0 the DPU writes nothing at all — an untouched surface, not "
                    "a wrong one — at every plane, stride and kernel. The fp16 form of "
                    "the same conv has no such bound: rocket_conv2d_fp16_rk3576()\n",
                    entry);
        return ROCKET_E_UNSUPPORTED;
    }
    /* ONE image channel writes nothing, where two, three and four are bit-exact. The
     * mode word's ARGB_IN nibble is 8 | (ic-1), so ic=1 is the one value that leaves
     * its low bits clear, and the fp16 form of the same conv computes at ic=1 — so
     * this is an int8-side gap rather than a property of the packed datapath. Refused
     * rather than left to write an untouched surface. */
    /* The OUTPUT width has its own granule, and it is not implied by the input's: at
     * ow 24 and 56 — both from an iw that is a multiple of 16 — output row 0 is exact
     * and every row after it is wrong, while ow 16, 32, 48, 64, 80, 112 are exact.
     * The direct path carries no such rule; this one comes with the channel fold. */
    if (ow % 16u || oh == 0u) {
        ROCKET_LOGE("%s: the int8 first conv needs ow a multiple of 16 (ow=%u from "
                    "iw=%u stride %u). At any other output width the first output row "
                    "is exact and every row after it is wrong\n", entry, ow, IW, SX);
        return ROCKET_E_UNSUPPORTED;
    }
    if (ow * SX != IW || oh * SY != IH) {
        ROCKET_LOGE("%s: the int8 first conv needs the SAME-padding output extent — "
                    "ow*stride_x == iw and oh*stride_y == ih (got %u*%u vs iw=%u, %u*%u "
                    "vs ih=%u). Any other output width writes a full surface SHEARED by "
                    "one column per row, with nothing to fault on\n",
                    entry, ow, SX, IW, oh, SY, IH);
        return ROCKET_E_UNSUPPORTED;
    }
    if (OC != rocket_rk3576_pad_oc(OC)) {
        ROCKET_LOGE("%s: oc=%u is a partial 32-channel group and writes nothing; size "
                    "the output and coefficient buffers for %u channels and pass that "
                    "count (rocket_rk3576_pad_oc)\n",
                    entry, OC, rocket_rk3576_pad_oc(OC));
        return ROCKET_E_UNSUPPORTED;
    }

    in_bytes = (size_t)IH * IW * ICP;
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;

    ops   = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    rows  = calloc(max_tasks, sizeof *rows);
    sum_w = calloc(OC, sizeof *sum_w);
    wtile = calloc((size_t)tile * ICP * KH * KW, 1);
    if (!ops || !rows || !sum_w || !wtile) { rc = ROCKET_E_NOMEM; goto done; }

    /* Each output channel's whole filter, for the input zero point's fold. */
    {
        unsigned c, i, y, x;
        for (c = 0; c < OC; c++) {
            int64_t s = 0;
            for (i = 0; i < IC; i++)
                for (y = 0; y < KH; y++)
                    for (x = 0; x < KW; x++)
                        s += W[(((size_t)c * IC + i) * KH + y) * KW + x];
            sum_w[c] = s;
        }
    }

    /* The row window, on the same planner as every other path. Told the precision
     * because the offsets come back in PACKED-IMAGE row units here — an int8 packed
     * image is `ic` interleaved bytes per pixel where a float one is halfwords. */
    plan.ic = (uint16_t)ICP; plan.ih = (uint16_t)IH; plan.iw = (uint16_t)IW;
    plan.oc = (uint16_t)tile; plan.oh = (uint16_t)oh; plan.ow = (uint16_t)ow;
    plan.kh = (uint16_t)KH; plan.kw = (uint16_t)KW;
    plan.stride_y = (uint8_t)SY; plan.stride_x = (uint8_t)SX;
    plan.pad_top = (uint8_t)PT; plan.pad_left = (uint8_t)PL;
    plan.ih_full = (uint16_t)IH; plan.oh_full = (uint16_t)oh;
    if (rocket_rk3576_plan_rows_prec(&plan, 0, precision_int8, rows, max_tasks,
                                     &nrow) < 0 || !nrow) {
        ROCKET_LOGE("%s: no row plan for the first conv (ic=%u %ux%u k%ux%u)\n",
                    entry, IC, IW, IH, KW, KH);
        rc = ROCKET_E_UNSUPPORTED; goto done;
    }

    if (rocket_bo_alloc(fd, in_bytes, &b.in) < 0 ||
        rocket_bo_alloc(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }

    /* CHW in, interleaved out — the packed image the CNA reads, shared by every tile.
     * The sample goes in as PLAIN TWO'S COMPLEMENT and the MAC reads it that way.
     *
     * THE CONVERTER'S OFFSET IS INERT, which is the opposite of what the datapath's
     * description says and is why the zero point is folded exactly as the direct path
     * folds it. The CVT registers are transcribed from captures that are all zero
     * point 0, so nothing ever exercised them; driven on the part, an image written at
     * raw = s + (zp + 0x80) comes back as the raw byte read as a signed int8 with no
     * subtraction at all — 0x80 reads -128, 0xC0 reads -64, 0xFF reads -1. So the
     * packed image is signed, and A carries the -in_zp*sum_w correction. */
    rocket_bo_prep(fd, &b.in, 1, 0);
    {
        int8_t *img = (int8_t *)b.in.ptr;
        unsigned c, y, x;
        for (y = 0; y < IH; y++)
            for (x = 0; x < IW; x++)
                for (c = 0; c < ICP; c++)
                    img[((size_t)y * IW + x) * ICP + c] =
                        c < IC ? in[((size_t)c * IH + y) * IW + x] : 0;
    }
    rocket_bo_fini(fd, &b.in);

    for (oc0 = 0; oc0 < OC; oc0 += tile) {
        unsigned n = OC - oc0 < tile ? OC - oc0 : tile;
        size_t w_bytes = rocket_rk3576_weight_argb_int8_bytes(n, KH, KW);
        size_t coeff_bytes = rocket_rk3576_coeff_bytes(n);
        size_t obytes = (size_t)((n + C2 - 1u) / C2) * surf_elems * C2;
        conv_params_t p = {0};
        struct r76_task_extent e;
        uint32_t in_h[4], out_h[1];

        if (b.w.ptr)     rocket_bo_free(fd, &b.w);
        if (b.coeff.ptr) rocket_bo_free(fd, &b.coeff);
        if (b.out.ptr)   rocket_bo_free(fd, &b.out);
        memset(&b.w, 0, sizeof b.w);
        memset(&b.coeff, 0, sizeof b.coeff);
        memset(&b.out, 0, sizeof b.out);
        if (rocket_bo_alloc(fd, w_bytes, &b.w) < 0 ||
            rocket_bo_alloc(fd, coeff_bytes, &b.coeff) < 0 ||
            rocket_bo_alloc(fd, obytes, &b.out) < 0) { rc = ROCKET_E_NOMEM; goto done; }

        /* This tile's channels renumbered from zero — its own whole convolution. */
        if (ICP == IC) {
            memcpy(wtile, W + (size_t)oc0 * IC * KH * KW, (size_t)n * IC * KH * KW);
        } else {
            unsigned j, t;
            memset(wtile, 0, (size_t)tile * ICP * KH * KW);
            for (j = 0; j < n; j++)
                for (t = 0; t < IC * KH * KW; t++)
                    wtile[(size_t)j * ICP * KH * KW + t] =
                        W[(size_t)(oc0 + j) * IC * KH * KW + t];
        }
        rocket_bo_prep(fd, &b.w, 1, 0);
        rc = rocket_rk3576_argb_int8_pack_weights(b.w.ptr, w_bytes, wtile, n, ICP, KH, KW);
        rocket_bo_fini(fd, &b.w);
        if (rc < 0) { rc = ROCKET_E_SHAPE; goto done; }

        free(A); free(B);
        A = calloc(n, sizeof *A);
        B = w_zp ? calloc(n, sizeof *B) : NULL;
        if (!A || (w_zp && !B)) { rc = ROCKET_E_NOMEM; goto done; }
        r76_fold_coeff(A, bias, oc0, n, sum_w, in_zp, w_zp, taps, NULL);
        if (B) { unsigned j; for (j = 0; j < n; j++) B[j] = (int16_t)(-w_zp); }
        rocket_bo_prep(fd, &b.coeff, 1, 0);
        if (B) rocket_rk3576_pack_coeff_asym(b.coeff.ptr, coeff_bytes, A, n, B, 1);
        else   rocket_rk3576_pack_coeff(b.coeff.ptr, coeff_bytes, A, n);
        rocket_bo_fini(fd, &b.coeff);

        p.ic = (uint16_t)ICP; p.iw = (uint16_t)IW; p.ih = (uint16_t)IH;
        p.oc = (uint16_t)n;  p.ow = (uint16_t)ow; p.oh = (uint16_t)oh;
        p.kh = (uint16_t)KH; p.kw = (uint16_t)KW;
        p.stride_y = (uint8_t)SY; p.stride_x = (uint8_t)SX;
        p.pad_top = (uint8_t)PT; p.pad_left = (uint8_t)PL;
        p.ih_full = (uint16_t)IH; p.oh_full = (uint16_t)oh;
        p.int8_out = 1;
        p.in_scale = in_scale; p.w_scale = w_scale; p.out_scale = out_scale;
        /* uint8-centered, as on the direct path: the emitter programs the border
         * constant as (input_zero_point + 0x80) & 0xFF, and that byte has to BE the
         * stored zero point, so that a pad tap's true value is zero. The converter
         * offset the same field feeds is inert (above), so it costs nothing. */
        p.input_zero_point  = in_zp + 0x80;
        p.output_zero_point = out_zp + 0x80;
        p.weight_zero_point = 0x80;
        p.tasks       = ops;
        p.input_dma   = b.in.dma_address;
        p.weights_dma = b.w.dma_address;
        p.bias_dma    = b.coeff.dma_address;
        p.output_dma  = b.out.dma_address;

        in_h[0] = b.in.handle; in_h[1] = b.w.handle;
        in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
        out_h[0] = b.out.handle;

        if (stamp) {
            rocket_bo_prep(fd, &b.out, 1, 0);
            memset(b.out.ptr, stamp, obytes);
            rocket_bo_fini(fd, &b.out);
        }

        for (r = 0; r < nrow; r++) {
            conv_params_t q = p;
            q.ih = rows[r].ih; q.oh = rows[r].oh;
            q.pad_top = rows[r].pad_top;
            q.input_dma  = p.input_dma  + rows[r].feature_off;
            q.output_dma = p.output_dma + rows[r].output_off;
            q.ih_full = (uint16_t)IH; q.oh_full = (uint16_t)oh;
            if (gen_conv2d_int8_rk3576(&q) != 0) {
                ROCKET_LOGE("%s: the generator refused the first-conv program (ic=%u "
                            "%ux%u k%ux%u oc %u..%u rows %u..%u)\n", entry, IC, IW, IH,
                            KW, KH, oc0, oc0 + n, rows[r].oy0, rows[r].oy0 + rows[r].oh);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }
            /* Per TASK, not per tile: one poisoned submit among several leaves its own
             * rows stale while its siblings are full. */
            e.groups      = (n + C2 - 1u) / C2;
            e.group_bytes = (size_t)surf_elems * C2;
            e.row_off     = (size_t)rows[r].oy0 * ow * C2;
            e.span        = (size_t)rows[r].oh * ow * C2;
            rc = r76_submit_task(fd, &b, &q, ops, in_h, 4u, out_h, &e, stamp, 0u, entry);
            if (rc != ROCKET_OK) goto done;
        }

        rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
        {
            const int8_t *o = (const int8_t *)b.out.ptr;
            size_t px = (size_t)oh * ow, p;
            unsigned c;
            for (c = 0; c < n; c++) {
                const int8_t *src = o + (size_t)(c / C2) * surf_elems * C2 + (c % C2);
                int8_t *dst = out + (size_t)(oc0 + c) * px;
                for (p = 0; p < px; p++) dst[p] = src[p * C2];
            }
        }
        rocket_bo_fini(fd, &b.out);
    }
    rc = ROCKET_OK;

done:
    free(ops); free(rows); free(A); free(B); free(sum_w); free(wtile);
    r76_conv_free(fd, &b);
    return rc;
}

int rocket_conv2d_int8_rk3576(int fd, const rocket_conv2d_desc *d,
                              const int8_t *in, const int8_t *W, const int32_t *bias,
                              float in_scale, float w_scale, float out_scale,
                              int in_zp, int w_zp, int out_zp, int8_t *out)
{
    const char *entry = "rocket_conv2d_int8_rk3576";

    if (d && d->depthwise)
        return rocket_conv2d_dw_int8_rk3576(fd, d, in, W, bias, in_scale, w_scale,
                                            out_scale, in_zp, w_zp, out_zp, out);
    /* Four or fewer channels is the packed-image first conv: a different CNA program,
     * a different feature buffer and a different weight cube. */
    if (d && d->ic <= 4) {
        unsigned ow, oh;
        int rc = r76_conv_check(entry, fd, d, 0, 1, &ow, &oh);
        if (rc != ROCKET_OK) return rc;
        if (!in || !W || !out) return ROCKET_E_SHAPE;
        if (!(in_scale > 0.0f) || !(w_scale > 0.0f) || !(out_scale > 0.0f))
            return ROCKET_E_SHAPE;
        if (in_zp < -128 || in_zp > 127 || w_zp < -128 || w_zp > 127 ||
            out_zp < -128 || out_zp > 127)
            return ROCKET_E_SHAPE;
        return r76_conv_int8_argb(entry, fd, d, in, W, bias, in_scale, w_scale,
                                  out_scale, in_zp, w_zp, out_zp, out, ow, oh);
    }
    return r76_conv_int8_run(entry, fd, d, 0, in, W, bias,
                             in_scale, w_scale, NULL, out_scale,
                             in_zp, w_zp, out_zp, out);
}

unsigned rocket_conv2d_int8_perchannel_oc_tile_rk3576(const rocket_conv2d_desc *d,
                                                      const int8_t *W,
                                                      const int32_t *bias,
                                                      float in_scale,
                                                      const float *w_scale,
                                                      float out_scale, int in_zp)
{
    unsigned IC, OC, KH, KW, icreg, taps, cbuf_tile, tile = 0;
    int64_t *sum_w = NULL, *sum_abs_w = NULL;
    unsigned *perm = NULL;
    unsigned c, i, y, x;

    if (!d || !W || !w_scale || d->depthwise || d->ic <= 4) return 0;
    IC = (unsigned)d->ic; OC = (unsigned)d->oc;
    KH = (unsigned)d->kh; KW = (unsigned)d->kw;
    icreg = rocket_rk3576_pad_ic(IC);
    taps  = IC * KH * KW;
    cbuf_tile = r76_conv_oc_tile(icreg, KH, KW, rocket_rk3576_pad_oc(OC));
    if (!cbuf_tile) return 0;

    sum_w = calloc(OC, sizeof *sum_w);
    sum_abs_w = calloc(OC, sizeof *sum_abs_w);
    perm = calloc(OC, sizeof *perm);
    if (!sum_w || !sum_abs_w || !perm) goto done;
    for (c = 0; c < OC; c++) {
        int64_t s = 0, sa = 0;
        for (i = 0; i < IC; i++)
            for (y = 0; y < KH; y++)
                for (x = 0; x < KW; x++) {
                    int64_t v = W[(((size_t)c * IC + i) * KH + y) * KW + x];
                    s += v; sa += v < 0 ? -v : v;
                }
        sum_w[c] = s; sum_abs_w[c] = sa;
    }
    r76_sort_by_scale(perm, OC, w_scale);
    tile = r76_pc_oc_tile(OC, cbuf_tile, perm, sum_abs_w, bias, sum_w,
                          in_zp, 0, taps, in_scale, w_scale, out_scale);
done:
    free(sum_w); free(sum_abs_w); free(perm);
    return tile;
}

int rocket_conv2d_int8_perchannel_rk3576(int fd, const rocket_conv2d_desc *d,
                                         const int8_t *in, const int8_t *W,
                                         const int32_t *bias, float in_scale,
                                         const float *w_scale, float out_scale,
                                         int in_zp, int out_zp, int8_t *out)
{
    const char *entry = "rocket_conv2d_int8_perchannel_rk3576";

    if (!w_scale) return ROCKET_E_SHAPE;
    if (d && d->depthwise) {
        ROCKET_LOGE("%s: the depthwise coefficient group carries a C multiplier too, "
                    "but this planner is wired to the direct path's 64-byte group\n",
                    entry);
        return ROCKET_E_UNSUPPORTED;
    }
    if (d && d->ic <= 4) {
        ROCKET_LOGE("%s: four or fewer input channels is the packed-image first conv, "
                    "a different program whose coefficient path this planner does not "
                    "drive\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }
    return r76_conv_int8_run(entry, fd, d, 0, in, W, bias,
                             in_scale, 1.0f, w_scale, out_scale,
                             in_zp, 0, out_zp, out);
}

int rocket_conv2d_dw_int8_rk3576(int fd, const rocket_conv2d_desc *d,
                                 const int8_t *in, const int8_t *w, const int32_t *bias,
                                 float in_scale, float w_scale, float out_scale,
                                 int in_zp, int w_zp, int out_zp, int8_t *out)
{
    return r76_conv_int8_run("rocket_conv2d_dw_int8_rk3576", fd, d, 1, in, w, bias,
                             in_scale, w_scale, NULL, out_scale,
                             in_zp, w_zp, out_zp, out);
}

/* ============================================================================
 * SECTION — the fp16 FIRST CONV, on the packed-image datapath
 *
 * A convolution whose input carries four or fewer channels is not the one below at a
 * small channel count: the CNA reads a PACKED IMAGE straight out of DDR and expands
 * every pixel to four lanes, so one task contracts the whole image and there is no
 * input-channel split to make. That is how a vision stem runs on this part at all —
 * the normal float path contracts sixteen channels at a time and an RGB image is three.
 *
 * Two things differ from every other entry here and both are the caller's:
 *
 *   - The FEATURE BUFFER is a packed image, `ic` interleaved fp16 per pixel, not the
 *     NC1HWC2 cube. This entry still takes the library's row-major CHW tensor and
 *     interleaves it itself, so a caller sees one convolution API either way.
 *   - The WEIGHT CUBE is neither the int8 cube nor the float one: four lanes per
 *     (output channel, tap), output channels interleaved by sixteen inside a tap.
 *     rocket_rk3576_argb_fp16_pack_weights() owns it.
 *
 * The output is the ordinary fp16 surface, so nothing downstream changes.
 * ==========================================================================*/
/* THE OUTPUT-CHANNEL BOUND ON THIS PATH, measured rather than transcribed: one
 * first-conv program delivers THIRTY-TWO output channels and no more. At oc 48, 64 and
 * 96 exactly 32 whole channels come back bit-exact and the rest of the surface is never
 * written — a contiguous prefix, not the interleave the int32 writer's byte budget
 * gives — while oc 24 and 32 are complete. It is not a register the emitter is getting
 * wrong: our program matches the vendor's own oc=48 and oc=64 captures register for
 * register, and the weight cube reproduces those captures too, so the vendor's compiler
 * emits single programs the part does not fully execute.
 *
 * So the split is the caller's, exactly as the direct path splits for its weight slice:
 * each tile is an independent convolution over its own output channels, at one submit
 * each. [HW sweep, H96 MAX M9] */
#define R76_ARGB_OC_MAX 32u

static int r76_conv_fp16_argb(const char *entry, int fd, const rocket_conv2d_desc *d,
                              const _Float16 *in, const _Float16 *W, _Float16 *out,
                              unsigned ow, unsigned oh)
{
    struct r76_conv_bos b = {0};
    uint64_t *ops = NULL;
    float *acc = NULL;
    _Float16 *wtile = NULL;
    unsigned IC = (unsigned)d->ic, OC = (unsigned)d->oc;
    unsigned IH = (unsigned)d->ih, IW = (unsigned)d->iw;
    unsigned KH = (unsigned)d->kh, KW = (unsigned)d->kw;
    unsigned tile = OC < R76_ARGB_OC_MAX ? OC : R76_ARGB_OC_MAX;
    unsigned tilepad = rocket_rk3576_fp16_pad_oc(tile);
    unsigned oc0, nrow = 0, r;
    rocket_rk3576_row_task *rows = NULL;
    size_t in_bytes, w_bytes, coeff_bytes, surf, tile_elems;
    conv_params_t plan = {0};
    struct r76_task_extent e;
    uint32_t in_h[4], out_h[1];
    unsigned char stamp;
    int rc;

    in_bytes    = (size_t)IH * IW * IC * sizeof(_Float16);
    w_bytes     = rocket_rk3576_weight_argb_fp16_bytes(tilepad, KH, KW);
    coeff_bytes = rocket_rk3576_coeff_bytes(tilepad);
    surf        = rocket_rk3576_fp16_out_bytes(OC, oh, ow);
    tile_elems  = (size_t)tilepad * IC * KH * KW;
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;

    ops   = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    acc   = calloc((size_t)OC * oh * ow, sizeof *acc);
    wtile = calloc(tile_elems, sizeof *wtile);
    rows  = calloc(oh ? oh : 1u, sizeof *rows);
    if (!ops || !acc || !wtile || !rows) { rc = ROCKET_E_NOMEM; goto done; }

    /* The row window, on the same axis and the same planner as every other path. A
     * stem-sized plane does not fit the CBUF in one task — 224x224 is 6272 granules
     * against a 6144 ceiling — so the plane is cut into windows, each reading the input
     * rows its output rows need. The offsets come back in PACKED-IMAGE row units here,
     * which is why the planner has to be told the precision: a float packed image is
     * `ic` interleaved halfwords per pixel where an int8 one is bytes. */
    plan.ic = (uint16_t)IC; plan.ih = (uint16_t)IH; plan.iw = (uint16_t)IW;
    plan.oc = (uint16_t)tilepad; plan.oh = (uint16_t)oh; plan.ow = (uint16_t)ow;
    plan.kh = (uint16_t)KH; plan.kw = (uint16_t)KW;
    plan.stride_y = (uint8_t)d->stride_y; plan.stride_x = (uint8_t)d->stride_x;
    plan.pad_top = (uint8_t)d->pad_top; plan.pad_left = (uint8_t)d->pad_left;
    plan.ih_full = (uint16_t)IH; plan.oh_full = (uint16_t)oh;
    if (rocket_rk3576_plan_rows_prec(&plan, 0, precision_float16, rows,
                                     oh ? oh : 1u, &nrow) < 0 || !nrow) {
        ROCKET_LOGE("%s: no row plan for the first conv (ic=%u %ux%u k%ux%u)\n",
                    entry, IC, IW, IH, KW, KH);
        rc = ROCKET_E_UNSUPPORTED; goto done;
    }

    if (rocket_bo_alloc(fd, in_bytes, &b.in) < 0 ||
        rocket_bo_alloc(fd, w_bytes, &b.w) < 0 ||
        rocket_bo_alloc(fd, coeff_bytes, &b.coeff) < 0 ||
        rocket_bo_alloc(fd, surf, &b.out) < 0 ||
        rocket_bo_alloc(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }

    /* CHW in, interleaved out — the packed image the CNA reads. Shared by every tile. */
    rocket_bo_prep(fd, &b.in, 1, 0);
    {
        _Float16 *img = (_Float16 *)b.in.ptr;
        unsigned c, y, x;
        for (y = 0; y < IH; y++)
            for (x = 0; x < IW; x++)
                for (c = 0; c < IC; c++)
                    img[((size_t)y * IW + x) * IC + c] =
                        in[((size_t)c * IH + y) * IW + x];
    }
    rocket_bo_fini(fd, &b.in);

    /* C gates the BS stage and a float program reads it as fp16, where the integer 1
     * is a denormal that empties the surface. One tile's worth, reused by every tile. */
    rocket_bo_prep(fd, &b.coeff, 1, 0);
    rc = rocket_rk3576_pack_coeff_prec(b.coeff.ptr, coeff_bytes, NULL, tilepad,
                                       precision_float16);
    rocket_bo_fini(fd, &b.coeff);
    if (rc < 0) { rc = ROCKET_E_SHAPE; goto done; }

    in_h[0] = b.in.handle; in_h[1] = b.w.handle;
    in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
    out_h[0] = b.out.handle;

    if (stamp) {
        rocket_bo_prep(fd, &b.out, 1, 0);
        memset(b.out.ptr, stamp, surf);
        rocket_bo_fini(fd, &b.out);
    }

    for (oc0 = 0; oc0 < OC; oc0 += tile) {
        unsigned n = OC - oc0 < tile ? OC - oc0 : tile;
        unsigned npad = rocket_rk3576_fp16_pad_oc(n);

        /* This tile's channels, renumbered from zero — its own whole convolution. The
         * cube is the same for every row window, so it is packed once per tile. */
        memset(wtile, 0, tile_elems * sizeof *wtile);
        memcpy(wtile, W + (size_t)oc0 * IC * KH * KW,
               (size_t)n * IC * KH * KW * sizeof *wtile);
        rocket_bo_prep(fd, &b.w, 1, 0);
        rc = rocket_rk3576_argb_fp16_pack_weights(b.w.ptr, w_bytes, wtile, npad, IC,
                                                  KH, KW);
        rocket_bo_fini(fd, &b.w);
        if (rc < 0) { rc = ROCKET_E_SHAPE; goto done; }

        for (r = 0; r < nrow; r++) {
            conv_params_t p = {0};

            p.ic = (uint16_t)IC; p.iw = (uint16_t)IW;
            p.ih = rows[r].ih;   p.oh = rows[r].oh;
            p.oc = (uint16_t)npad; p.ow = (uint16_t)ow;
            p.kh = (uint16_t)KH; p.kw = (uint16_t)KW;
            p.stride_y = (uint8_t)d->stride_y; p.stride_x = (uint8_t)d->stride_x;
            p.pad_top = rows[r].pad_top; p.pad_left = (uint8_t)d->pad_left;
            p.ih_full = (uint16_t)IH; p.oh_full = (uint16_t)oh;
            p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;
            p.input_zero_point = 0; p.output_zero_point = 0; p.weight_zero_point = 0;
            p.tasks       = ops;
            p.input_dma   = b.in.dma_address + rows[r].feature_off;
            p.weights_dma = b.w.dma_address;
            p.bias_dma    = b.coeff.dma_address;
            /* The output cube's channel groups are contiguous planes, so a tile
             * starting on a group boundary is the same BO at a plane offset, and a row
             * window is that plus the planner's own row offset. */
            p.output_dma  = b.out.dma_address +
                            (uint32_t)((size_t)oc0 * oh * ow * sizeof(_Float16)) +
                            rows[r].output_off;
            if (gen_conv2d_fp16_rk3576(&p) != 0) {
                ROCKET_LOGE("%s: the generator refused the first-conv program (ic=%u "
                            "%ux%u k%ux%u oc %u..%u rows %u..%u)\n", entry, IC, IW, IH,
                            KW, KH, oc0, oc0 + n, rows[r].oy0,
                            rows[r].oy0 + rows[r].oh);
                rc = ROCKET_E_UNSUPPORTED; goto done;
            }

            /* Per TASK, not per tile: one poisoned submit among several leaves its own
             * rows stale while its siblings are full. */
            e.groups      = (npad + C2F - 1u) / C2F;
            e.group_bytes = (size_t)ow * oh * C2F * sizeof(_Float16);
            e.row_off     = (size_t)oc0 * oh * ow * sizeof(_Float16)
                            + (size_t)rows[r].output_off;
            e.span        = (size_t)rows[r].oh * ow * C2F * sizeof(_Float16);
            rc = r76_submit_task(fd, &b, &p, ops, in_h, 4u, out_h, &e, stamp,
                             ROCKET_JOB_NO_DPU_DONE, entry);
            if (rc != ROCKET_OK) goto done;
        }
    }

    rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
    rc = rocket_rk3576_fp16_accumulate(acc, b.out.ptr, surf, OC, oh, ow);
    rocket_bo_fini(fd, &b.out);
    if (rc < 0) { rc = ROCKET_E_SHAPE; goto done; }

    {
        size_t i, n = (size_t)OC * oh * ow;
        for (i = 0; i < n; i++) out[i] = (_Float16)acc[i];
    }
    rc = ROCKET_OK;

done:
    free(ops); free(acc); free(wtile); free(rows);
    r76_conv_free(fd, &b);
    return rc;
}

/* ============================================================================
 * SECTION — the fp16 convolution, through the input-channel split
 *
 * One fp16 task contracts exactly SIXTEEN input channels — the DPU's output element
 * stride is 16/ic words, so that is the only count at which an element occupies its own
 * two bytes — and gen_conv2d_fp16_rk3576() refuses any other. rocket_rk3576_plan_ic()
 * turns an arbitrary count into that sequence; the feature cube is shared and addressed
 * by each slice's offset, the weight cube is per-slice, and the partial surfaces are
 * summed on the host.
 *
 * WHAT THIS DOES NOT DO: compose the split with the row window. A plane whose 16-channel
 * slice still overflows the CBUF is refused by the planner, and running the ic slices of
 * each row task would carry it further. On-chip accumulation across the slices — the
 * RK3588's ROCKET_KACC analog — would delete ic/16 readbacks and is no longer blocked by
 * anything in the writer. Both are open levers rather than defects.
 * ==========================================================================*/
int rocket_conv2d_fp16_rk3576(int fd, const rocket_conv2d_desc *d,
                              const _Float16 *in, const _Float16 *W, _Float16 *out)
{
    const char *entry = "rocket_conv2d_fp16_rk3576";
    struct r76_conv_bos b = {0};
    uint64_t *ops = NULL;
    float *acc = NULL;
    rocket_rk3576_ic_task *slices = NULL;
    unsigned IC, OC, IH, IW, KH, KW;
    unsigned ow, oh, icpad, ocpad, nslice = 0, s, max_slices;
    size_t in_bytes, w_bytes, coeff_bytes, surf;
    conv_params_t base = {0};
    struct r76_task_extent e;
    struct r76_fp16_prof prof = {0, 0, 0, 0, 0, 0};
    uint32_t in_h[4], out_h[1];
    unsigned char stamp;
    int rc;

    rc = r76_conv_check(entry, fd, d, d && d->depthwise, 1, &ow, &oh);
    if (rc != ROCKET_OK) return rc;
    if (!in || !W || !out) return ROCKET_E_SHAPE;
    if (d->depthwise) {
        ROCKET_LOGE("%s: the fp16 depthwise cube is not decoded on this part; the int8 "
                    "depthwise path is (rocket_conv2d_dw_int8_rk3576)\n", entry);
        return ROCKET_E_UNSUPPORTED;
    }

    IC = (unsigned)d->ic; OC = (unsigned)d->oc;
    IH = (unsigned)d->ih; IW = (unsigned)d->iw;
    KH = (unsigned)d->kh; KW = (unsigned)d->kw;

    /* Four or fewer channels is the packed-image first conv, a different CNA program
     * with a different feature buffer and a different weight cube. */
    if (IC <= 4u)
        return r76_conv_fp16_argb(entry, fd, d, in, W, out, ow, oh);

    icpad = rocket_rk3576_fp16_pad_ic(IC);
    ocpad = rocket_rk3576_fp16_pad_oc(OC);
    in_bytes    = (size_t)(icpad / C2F) * IH * IW * C2F * sizeof(_Float16);
    w_bytes     = rocket_rk3576_fp16_slice_weight_bytes(OC, IC, KH, KW);
    coeff_bytes = rocket_rk3576_coeff_bytes(ocpad);
    surf        = rocket_rk3576_fp16_out_bytes(OC, oh, ow);
    max_slices  = icpad / ROCKET_RK3576_FP16_IC_SLICE + 2u;
    stamp = rocket_rk3576_sentinel_on() ? (unsigned char)ROCKET_RK3576_SENTINEL_BYTE : 0;

    base.ic = (uint16_t)IC; base.ih = (uint16_t)IH; base.iw = (uint16_t)IW;
    base.oc = (uint16_t)OC; base.oh = (uint16_t)oh; base.ow = (uint16_t)ow;
    base.kh = (uint16_t)KH; base.kw = (uint16_t)KW;
    base.stride_y = (uint8_t)d->stride_y; base.stride_x = (uint8_t)d->stride_x;
    base.pad_top  = (uint8_t)d->pad_top;  base.pad_left = (uint8_t)d->pad_left;
    base.ih_full = (uint16_t)IH; base.oh_full = (uint16_t)oh;
    base.in_scale = 1.0f; base.w_scale = 1.0f; base.out_scale = 1.0f;
    base.input_zero_point = 0x80; base.output_zero_point = 0x80;
    base.weight_zero_point = 0x80;

    ops    = calloc(RK3576_CONV_TASK_OPS, sizeof *ops);
    slices = calloc(max_slices, sizeof *slices);
    acc    = calloc((size_t)OC * oh * ow, sizeof *acc);
    if (!ops || !slices || !acc) { rc = ROCKET_E_NOMEM; goto done; }

    if (rocket_rk3576_plan_ic(&base, slices, max_slices, &nslice) < 0) {
        ROCKET_LOGE("%s: no input-channel plan for ic=%u (%ux%u k%ux%u) — a 16-channel "
                    "slice of this plane still overflows the CBUF, and composing the "
                    "split with the row window is not wired\n", entry, IC, IW, IH, KW, KH);
        rc = ROCKET_E_UNSUPPORTED; goto done;
    }

    if (rocket_bo_alloc(fd, in_bytes, &b.in) < 0 ||
        rocket_bo_alloc(fd, w_bytes, &b.w) < 0 ||
        rocket_bo_alloc(fd, coeff_bytes, &b.coeff) < 0 ||
        rocket_bo_alloc(fd, surf, &b.out) < 0 ||
        rocket_bo_alloc(fd, RK3576_CONV_TASK_OPS * sizeof(uint64_t), &b.rc) < 0) {
        rc = ROCKET_E_NOMEM; goto done;
    }

    /* The feature cube is shared: at an 8-channel atom its channel groups are contiguous
     * planes, so slice s is the same BO at input_dma + feature_off. */
    rocket_bo_prep(fd, &b.in, 1, 0);
    memset(b.in.ptr, 0, in_bytes);
    {
        _Float16 *cube = (_Float16 *)b.in.ptr;
        size_t px = (size_t)IH * IW, p;
        unsigned c;
        for (c = 0; c < IC; c++) {
            _Float16 *dst = cube + (size_t)(c / C2F) * px * C2F + (c % C2F);
            const _Float16 *src = in + (size_t)c * px;
            for (p = 0; p < px; p++) dst[p * C2F] = src[p];
        }
    }
    rocket_bo_fini(fd, &b.in);

    /* The coefficient buffer is shared and carries no bias, but its C multiplier still
     * gates the BS stage — and a FLOAT program reads C as fp16, where the integer 1 is
     * the denormal 6e-8 and underflows the whole surface to empty. */
    rocket_bo_prep(fd, &b.coeff, 1, 0);
    if (rocket_rk3576_pack_coeff_prec(b.coeff.ptr, coeff_bytes, NULL, ocpad,
                                      precision_float16) < 0) {
        rocket_bo_fini(fd, &b.coeff);
        rc = ROCKET_E_SHAPE; goto done;
    }
    rocket_bo_fini(fd, &b.coeff);

    in_h[0] = b.in.handle; in_h[1] = b.w.handle;
    in_h[2] = b.coeff.handle; in_h[3] = b.rc.handle;
    out_h[0] = b.out.handle;

    prof.on = r76_fp16_prof_on();
    prof.slices = nslice;

    for (s = 0; s < nslice; s++) {
        conv_params_t p = base;
        double t0;

        t0 = prof.on ? r76_now_us() : 0;
        rocket_bo_prep(fd, &b.w, 1, 0);
        if (rocket_rk3576_fp16_pack_slice_weights(b.w.ptr, w_bytes, W, OC, IC, KH, KW,
                                                  &slices[s]) < 0) {
            rocket_bo_fini(fd, &b.w);
            rc = ROCKET_E_SHAPE; goto done;
        }
        rocket_bo_fini(fd, &b.w);
        if (prof.on) prof.pack_us += r76_now_us() - t0;

        p.ic          = slices[s].ic;
        p.tasks       = ops;
        p.input_dma   = b.in.dma_address + slices[s].feature_off;
        p.weights_dma = b.w.dma_address;
        p.bias_dma    = b.coeff.dma_address;
        p.output_dma  = b.out.dma_address;
        if (gen_conv2d_fp16_rk3576(&p) != 0) {
            ROCKET_LOGE("%s: the generator refused slice %u of %u\n", entry, s, nslice);
            rc = ROCKET_E_UNSUPPORTED; goto done;
        }

        /* Every slice rewrites the whole surface, so it is stamped per slice and read
         * back between submits. */
        t0 = prof.on ? r76_now_us() : 0;
        if (stamp) {
            rocket_bo_prep(fd, &b.out, 1, 0);
            memset(b.out.ptr, stamp, surf);
            rocket_bo_fini(fd, &b.out);
        }
        if (prof.on) prof.stamp_us += r76_now_us() - t0;

        e.groups      = (ocpad + C2F - 1u) / C2F;
        e.group_bytes = (size_t)ow * oh * C2F * sizeof(_Float16);
        e.row_off     = 0;
        e.span        = e.group_bytes;
        t0 = prof.on ? r76_now_us() : 0;
        rc = r76_submit_task(fd, &b, &p, ops, in_h, 4u, out_h, &e, stamp,
                             ROCKET_JOB_NO_DPU_DONE, entry);
        if (prof.on) prof.submit_us += r76_now_us() - t0;
        if (rc != ROCKET_OK) goto done;

        t0 = prof.on ? r76_now_us() : 0;
        rocket_bo_prep(fd, &b.out, 0, 2000000000ull);
        if (rocket_rk3576_fp16_accumulate(acc, b.out.ptr, surf, OC, oh, ow) < 0) {
            rocket_bo_fini(fd, &b.out);
            rc = ROCKET_E_SHAPE; goto done;
        }
        rocket_bo_fini(fd, &b.out);
        if (prof.on) prof.read_us += r76_now_us() - t0;
    }

    if (prof.on) {
        double tot = prof.pack_us + prof.stamp_us + prof.submit_us + prof.read_us;
        ROCKET_LOGI("rk3576 fp16 ic-split ic=%u oc=%u %ux%u k%ux%u: %u slices, %.2f ms"
                    " -- weights %.2f (%.0f%%)  stamp %.2f (%.0f%%)  submit %.2f (%.0f%%)"
                    "  readback %.2f (%.0f%%)\n",
                    IC, OC, IW, IH, KW, KH, nslice, tot / 1e3,
                    prof.pack_us / 1e3, 100.0 * prof.pack_us / (tot > 0 ? tot : 1),
                    prof.stamp_us / 1e3, 100.0 * prof.stamp_us / (tot > 0 ? tot : 1),
                    prof.submit_us / 1e3, 100.0 * prof.submit_us / (tot > 0 ? tot : 1),
                    prof.read_us / 1e3, 100.0 * prof.read_us / (tot > 0 ? tot : 1));
    }

    {
        size_t i, n = (size_t)OC * oh * ow;
        for (i = 0; i < n; i++) out[i] = (_Float16)acc[i];
    }
    rc = ROCKET_OK;

done:
    free(ops); free(slices); free(acc);
    r76_conv_free(fd, &b);
    return rc;
}
