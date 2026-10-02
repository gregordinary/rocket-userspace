// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_RK3576_INTERNAL_H
#define ROCKET_RK3576_INTERNAL_H

/*
 * rocket_rk3576_internal.h — the driving-side helpers every RK3576 entry point shares.
 *
 * The register encoders live in npu_regcmd_rk3576.c and the operand layouts in its
 * public header. What is here is the submit-loop discipline the part needs and no
 * generator can express: how long to idle when a submit came back having written
 * nothing, and whether an output surface is stamped before it runs. Both are properties
 * of the CHIP rather than of the op, so the matmul and the convolution share one copy.
 */
#include <stddef.h>
#include <stdint.h>

#include "rocket_npu.h"

/* The idle a poisoned submit needs before the next one will write.
 *
 * An int32-output job leaves the NEXT submit — of any kind, across calls and across
 * processes — completing normally in about 1.4 ms and writing nothing. What clears it
 * is the driver's runtime-PM autosuspend cycling the NPU power domain, not elapsed
 * time: with `power/control` = `on` no amount of idle clears it at all, and the working
 * gap tracks `power/autosuspend_delay_ms` one for one. So the delay is read from the
 * driver rather than fixed, and a system that lowers it gets a cheaper path for free.
 * ROCKET_RK3576_MM_GAP_MS overrides it outright. [HW sweep, H96 MAX M9]
 *
 * Returns 1 when the domain was OBSERVED to reach `suspended`, 0 when the call fell
 * back to a blind idle (no write access to the sysfs delay, or the domain did not
 * collapse inside the budget). A caller that retries on "wrote nothing" needs that
 * distinction: a redo after a confirmed cycle that still writes nothing is a
 * different fact from a redo after an idle that may never have cleared anything. */
int rocket_rk3576_power_idle(void);

/* Free everything the RK3576 transient-BO pool (ROCKET_RK3576_BO_POOL) holds for `fd`.
 *
 * Declared here as well as in rocket_matmul.h because rocket_close() must call it: the
 * pool is a PROCESS-global table keyed on the raw fd integer, and the kernel recycles
 * fd numbers. Without this, a process that closes NPU fd 7 and later opens anything that
 * lands on 7 is handed BOs whose GEM handle and dma_address belong to a dead file --
 * and the handles were already destroyed with it, so the addresses are stale rather
 * than merely wrong. Draining at close makes the pool's lifetime the fd's. */
void rocket_rk3576_bo_pool_drain(int fd);

/* Whether an output BO is stamped with a sentinel before the tasks that write it.
 *
 * A fresh BO arrives zeroed and zero is also a legitimate result, so a zeroed surface
 * cannot tell "never written" from "written and zero" — which is what made a poisoned
 * submit read as a wrong answer. Against a stamp the question is exact.
 *
 * Stamping is safe because the fill is BRACKETED by PREP_BO and FINI_BO, so the lines
 * are written back before the submit and none are left dirty to race the DPU's DMA. A
 * bare memset with no FINI_BO is the trap, and it is a different thing.
 *
 * ROCKET_RK3576_I32_SENTINEL=0 turns it off. */
int rocket_rk3576_sentinel_on(void);

#define ROCKET_RK3576_SENTINEL_BYTE 0xA5u

/* WHETHER A JOB COMPLETED, NOT ONLY WHETHER IT WROTE.
 *
 * The driver retires a job that runs past its backstop (125 ms from the kick here) and
 * signals its fence exactly as it signals a completed one, so PREP_BO returns 0 either
 * way. A retired job can leave a PARTIAL surface: depthwise at 8224 channels wrote 146 of
 * 131584 elements and the entry returned 0, because a guard that asks "did each task
 * write something" is satisfied by one byte [HW, H96, 2026-09-26]. So every guard site
 * also scores the job's completion, by its time from BEFORE the submit ioctl to its
 * fence. The kick follows the submit's start, so a retired job reads at least the
 * backstop; a healthy one reads under it, because the backstop bounds it too. A job
 * queued behind another process's reads long and is redone, which is conservative.
 *
 * What this cannot see: a job the kernel retires EARLY, which is `dpu_grace_us` shorter
 * than a drain and reads fast, and a partial write by a job that
 * completed. A caller that does host work between its submit and its wait still reads a
 * retirement, since it only adds to the time.
 *
 * rocket_rk3576_job_clock() is taken just before the submit; rocket_rk3576_job_ns()
 * reads the elapsed time just after the wait returns; rocket_rk3576_past_backstop() is
 * the pure test; rocket_rk3576_score_retired() counts and logs one retirement (with
 * whether the site's write guard saw the surface written) and returns the same test.
 * ROCKET_RK3576_BACKSTOP_US overrides the profile's backstop, which is how a probe makes
 * every job read as retired to prove each site consults it. */
uint64_t rocket_rk3576_job_clock(void);
uint64_t rocket_rk3576_job_ns(uint64_t t0);
int      rocket_rk3576_past_backstop(uint64_t elapsed_ns);
int      rocket_rk3576_score_retired(const char *entry, uint64_t elapsed_ns, int wrote);

/* The per-output-channel requant plan, shared by the convolution and the matmul.
 *
 * The DPU's epilogue is `(acc + A[oc]) * C[oc]` in saturating int32 followed by ONE
 * `(v*MUL)>>SHIFT` per task, so a per-channel scale is an integer C ramp riding on a
 * single (MUL, SHIFT). This picks both: the largest base gain every channel can reach
 * without its C exceeding the int16 field or its `(acc + A)*C` product overflowing
 * int32, then the ramp against the base the emitter will ACTUALLY program — the
 * quantized one, read back through the same derivation, not the one that was asked for.
 *
 * `w_scale[oc]` is per channel and `in_scale`/`out_scale` are per tensor; the target
 * gain is `in_scale*w_scale[oc]/out_scale`. A caller whose per-channel factor is
 * already the whole gain passes it as `w_scale` with the other two at 1.
 *
 * `sum_abs_w[oc]` is that channel's sum of |weight| over its whole filter, which is
 * what bounds the accumulator: taken from the ACTUAL weights, because the int8 envelope
 * is one to two orders of magnitude looser and the difference is most of the available
 * precision. `perm` may be NULL for the identity order. `C` is written for `ocreg`
 * channels — the padded tail gets 1, never 0, because a zero C gates the whole
 * eight-channel group's BS stage off and the DPU writes an empty surface with no fault.
 *
 * Returns the worst-case relative gain error over the tile's channels in
 * `*max_rel_err`, which is the resolution the integer ramp actually delivered and the
 * only thing that separates it from an exact per-column scale. */
int rocket_rk3576_plan_perchannel(const char *entry, unsigned oc0, unsigned tile_oc,
                                  unsigned ocreg, const int32_t *A,
                                  const int64_t *sum_abs_w, float in_scale,
                                  const float *w_scale, float out_scale,
                                  const unsigned *perm,
                                  int16_t *C, float *base_scale, double *max_rel_err);

/* THE SAME RAMP WITH THE BS SHIFT WORD CARRYING THE GAIN (the convolution's per-axis path
 * and the matmul's per-column entry; see the definition for the rule). Beside what the form
 * above returns it reports the shift the task's word must carry and the channels it
 * programmed as a CONSTANT, and it REWRITES those channels' `A` (in/out, per slot, the fold
 * on entry). `sum_w` is needed as well as `sum_abs_w` because a channel's reachable
 * accumulator is signed; NULL plans every channel live and leaves `A` alone, which is how
 * the matmul calls it. */
struct rocket_rk3576_pc_plan {
    unsigned bs_shift;      /* the shift word's value, both sign fields                  */
    float    base_scale;    /* the OUT_CVT gain the task programs                        */
    double   max_rel_err;   /* worst live channel's relative gain error                  */
    unsigned n_const;       /* channels programmed as their constant byte                */
    unsigned n_clamp;       /* live channels whose C fell below the field's floor        */
};
int rocket_rk3576_plan_perchannel_bs(const char *entry, unsigned oc0, unsigned tile_oc,
                                     unsigned ocreg, int32_t *A, const int64_t *sum_w,
                                     const int64_t *sum_abs_w, float in_scale,
                                     const float *w_scale, float out_scale, int out_zp,
                                     const unsigned *perm, int16_t *C,
                                     struct rocket_rk3576_pc_plan *pl);

/* The matmul's per-column plan of one N tile [n0, n0 + tile_n), padded to `nreg`, from the
 * ramp ROCKET_RK3576_MM_PC_SHIFT selects (read per call): the shift ramp by default, the
 * shift-0 ramp at 0. `scale_n` and `sum_abs_w` are indexed over the whole N, `tile_bias`
 * and `C` by slot. What rocket_matmul_int8_rk3576_perc*() programs, exported so a gate can
 * read the plan back rather than re-derive it; a model that must stay independent of the
 * planner (tests/perchannel_model.h) must not call it. Returns 0, or -1 as the planners do. */
struct rocket_rk3576_percol_plan {
    int      shift_ramp;    /* 1 when the shift word carries the gain                   */
    unsigned bs_shift;      /* the shift word, both sign fields; 0 on the shift-0 ramp  */
    float    gain;          /* the OUT_CVT gain the task programs                       */
    double   max_rel_err;   /* worst column's relative gain error                       */
    unsigned n_clamp;       /* columns whose C fell below the field's floor of one (the
                             * shift ramp only; the shift-0 ramp leaves it 0)            */
};
int rocket_rk3576_plan_percol(unsigned n0, unsigned tile_n, unsigned nreg,
                              int32_t *tile_bias, const int64_t *sum_abs_w,
                              const float *scale_n, int16_t *C,
                              struct rocket_rk3576_percol_plan *pl);

/* Where in the CBUF a task stages, as a granule offset added to the window base and
 * the fetch base together — a bring-up knob, zero for every shipped path.
 *
 * The two NPU cores share one CBUF and both stage from granule 0, which is why two
 * jobs executing at once compute wrong answers. This is the one field a userspace
 * encoder emits that looks like an address into that pool, so it is the one candidate
 * for expressing a partition. Set PER THREAD, because a concurrency probe has to give
 * two workers different bases inside one process and ROCKET_RK3576_CBUF_BIAS is
 * process-wide; the environment variable is the fallback when it is never called.
 *
 * See tests/rk3576_cbuf_base.c — a bias the hardware IGNORES is invisible on a solo
 * job, so "it still computes" is not evidence that the base moved. */
void rocket_rk3576_set_cbuf_bias(unsigned granules);

/* WHAT A CROSS-LAYER CHAIN NEEDS OF A POOLING HANDLE.
 *
 * A pool is its own register program here — PPU and PPU_RDMA only, PC_OPERATION_ENABLE
 * 0x60 against a convolution's 0x1D — and the two bitmaps being disjoint was the reason
 * to doubt that one could sit inside a convolution stream. It can: 20 of 20 iterations
 * over five geometries to a 110x110 plane gave the same pool output AND the same output
 * from the convolution after it as separate submits, with both intermediates read so a
 * failure would have named its boundary [HW sweep, H96 MAX M9, tests/rk3576_chain_pool.c].
 *
 * The chain lives with the convolution entries because that is where the row plan and the
 * write guard are, so this is the whole of what it reads off a pool handle. INTERIOR
 * nodes only: a pool that began a run would need the chain to scatter into its cube and
 * one that ended a run would need it de-scattered, and both are host work the stream
 * exists to remove — so the chain refuses either and the caller keeps that layer's own
 * submit.
 *
 * `ops` is filled by the caller (RK3576_POOL_TASK_OPS words) rather than borrowed from the
 * handle's own regcmd BO: the chain lays every program of a run out contiguously in a BO
 * of its own and rewrites each trailer to link to the next, which must not disturb the
 * handle — it stays callable one at a time through the per-layer entry, which is what the
 * chain falls back to when a program has to be redone. */
struct rocket_rk3576_pool_link {
    int        fd;
    uint64_t   feat_dma;      /* the cube it reads     */
    uint64_t   surf_dma;      /* the surface it writes */
    int        cube_in, cube_out;
    rocket_bo *surf;          /* the BO the surface lives in   */
    size_t     surf_off;      /* this handle's slice inside it */
    unsigned   groups;        /* channel groups written        */
    unsigned   surf_elems;    /* elements per channel group    */
    unsigned   live_elems;    /* elements of a group it fills  */
    uint32_t   nops;          /* words written into `ops`      */
};

struct rocket_pool_int8_rk3576_handle;

int rocket_rk3576_pool_link(struct rocket_pool_int8_rk3576_handle *h,
                            struct rocket_rk3576_pool_link *out, uint64_t *ops);

/* THE CHW <-> NC1HWC2 TRANSPOSE, for one channel group of a cube.
 *
 * A cube interleaves sixteen channels into every sixteen-byte atom, so a row-major
 * tensor and a cube are a transpose rather than a copy — and written an element at a
 * time it is one useful byte per destination cache line. The implementation is a 16x16
 * NEON block (rocket_rk3576_cube_pack.c) with the scalar loop as the pixel tail and the
 * non-NEON build.
 *
 * `sp`/`dp` are up to sixteen channel PLANES, as pointers rather than a base and a
 * stride, because a per-axis convolution sorts its output channels by scale so channel
 * c of a group can land anywhere in the caller's tensor. `px` is the pixel count and
 * `live` how many of the sixteen lanes the caller owns.
 *
 * On the PACK side the dead lanes of a partial group are still WRITTEN, because a whole
 * atom is stored either way — so `pad` is what they carry, and it is NOT free: it has to
 * be the value the consumer's datapath substitutes where those channels are read (the
 * CNA's border constant on a convolution's cube). A caller with no such contract keeps
 * the scalar loop for its partial group. */
void rocket_rk3576_c2_pack(int8_t *cube, const int8_t *const *sp, unsigned live,
                           size_t px, unsigned char pad);
void rocket_rk3576_c2_unpack(int8_t *const *dp, unsigned live, const int8_t *cube,
                             size_t px);

/* CHW -> the packed-image first conv's INTERLEAVED image: img[p*ic + c] = sp[c][p]. A
 * different transform from the cube transpose above — a 2/3/4-way interleave of whole
 * planes, which NEON stores with one vst2/vst3/vst4 — and it is what the packed encoding
 * spends against the MAC count it saves. `ic` is the PROGRAMMED channel count, so every
 * lane is a real plane and there is no dead-lane contract to keep. */
void rocket_rk3576_argb_pack(int8_t *img, const int8_t *const *sp, unsigned ic, size_t px);

#endif /* ROCKET_RK3576_INTERNAL_H */
