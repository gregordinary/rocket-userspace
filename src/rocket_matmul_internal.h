// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_matmul_internal.h — shared internals between rocket_matmul.c (the
 * one-shot tiled matmul) and rocket_prepacked.c (the pack-weights-once path).
 * NOT a public header; not installed.
 *
 * The tiled matmul splits into three reusable phases over a fixed tiling plan:
 *   pack weights (B) -> pack input (A) -> compute (gen+submit+wait+readback).
 * The one-shot path does all three per call and frees everything; the prepacked
 * path does the weight pack + BO alloc ONCE (into a handle) and repeats only
 * pack-input + compute, reusing resident weight/scratch BOs and a persistent fd.
 */
#ifndef ROCKET_MATMUL_INTERNAL_H
#define ROCKET_MATMUL_INTERNAL_H

#include <stddef.h>
#include <stdio.h>
#include "rocket_npu.h"
#include "rocket_log.h"     // centralized log channel

/* ============================================================================
 * SECTION — regcmd-slot overflow guard
 * ==========================================================================*/

/* Unconditional regcmd-slot overflow guard. After generating a tile's regcmd we
 * memcpy `count` u64 words into a fixed `stride`-word slot of the regcmd BO; a
 * generator emitting more than the slot holds would overrun it (a BO/heap
 * corruption). An assert() guarded this, but the default Release build compiles
 * asserts out (-DNDEBUG), leaving the shipping library with NO guard — so this
 * stays a hard runtime test. Evaluates to nonzero (after logging) on overflow;
 * the caller bails with its local cleanup idiom. */
#define MM_REGCMD_OVERFLOWS(count, stride) \
    ((size_t)(count) > (size_t)(stride) && \
     (ROCKET_LOGE("rocket_matmul: regcmd slot overflow: %zu > %d words (%s:%d)\n", \
              (size_t)(count), (int)(stride), __func__, __LINE__), 1))

/* ============================================================================
 * SECTION — Tiling plan & BO-set descriptors (mm_plan / mm_bos)
 * ==========================================================================*/

/* Resolved tiling plan + the per-tile NPU-layout slot sizes (in fp16 elems). */
typedef struct {
    int M, K, N;
    int Mt, Kt, Nt;
    int nMt, nNt, nKt;
    size_t in_slot, wt_slot, out_slot;
} mm_plan;

/* The BOs a tiled matmul drives: weights (resident across calls when prepacked)
 * plus per-problem scratch (input/output/regcmd + the low-IOVA guard). */
typedef struct {
    rocket_bo guard, regcmd, in_all, wt_all, out_all;
    rocket_bo okacc0, pong; /* the NPU K-accum output ping-pong, RIGHT-SIZED to
                      * nMt*nNt tiles (not BATCH) — the kacc path uses one job per
                      * K-tile with only nMt*nNt output tiles, and PREP_BO/FINI_BO
                      * cache-sync is proportional to BO size, so this cuts the
                      * per-ki sync. Resident (alloc'd only when ROCKET_KACC) so
                      * mm_compute_kacc needn't alloc per call; out_all stays
                      * BATCH-sized for the tiny-M mm_compute fallback.
                      * .handle==0 when unused. */
    int prezeroed;   /* in_all/wt_all were zeroed once at alloc: the
                      * per-call pack can then skip its full-BO memset. */
    int has_pong;    /* pong BO was allocated (ROCKET_KACC at alloc time). The compute
                      * dispatcher branches on THIS, not a fresh getenv, so the chosen
                      * path can never diverge from the allocated scratch (a KACC read
                      * after a non-KACC alloc would otherwise use a .handle==0 pong). */
    int has_pipe;    /* pipe double-buffer scratch was allocated (ROCKET_MM_PIPE at
                      * alloc time) — same alloc-time-capability rule as has_pong. */

    /* Resident host-side compute scratch, so the hot streaming/prepacked
     * path stops malloc/free-ing it every matmul (fixed per-job CPU floor). The
     * BATCH-sized per-job arrays are allocated once in mm_bos_alloc; `acc` (the
     * M*N fp32 accumulator for the CPU-accum mm_compute path) is sized lazily on
     * first use and grown only on shape change — under ROCKET_KACC it stays NULL
     * (mm_compute is only the tiny-M fallback), so no RAM is wasted. A worker
     * thread owns its mm_bos and calls compute serially, so no locking is needed. */
    rocket_task_desc *tasks;                  /* BATCH task descriptors            */
    int    *bm0, *bn0, *bMtile, *bNtile;      /* BATCH each (mm_compute)           */
    int    *bmi, *bni;                        /* BATCH each (mm_compute_kacc)      */
    size_t *boff;                             /* BATCH (mm_compute)                */
    float  *acc;                              /* M*N fp32 accumulator (mm_compute) */
    size_t  acc_cap;                          /* allocated float count in acc      */
    void   *submit_dt;                        /* BATCH drm_rocket_task[] scratch for
                                               * rocket_submit_tasks_pre;
                                               * opaque (uapi type) — sized via
                                               * rocket_submit_scratch_size(BATCH). */

    /* Double-buffer scratch for mm_compute_pipe (opt-in
     * ROCKET_MM_PIPE). A 2nd regcmd BO + 2nd output BO + a 2nd task array and
     * 2-slot tile metadata let the CPU build batch N+1's regcmd and de-tile batch
     * N-1 while the NPU computes batch N. Allocated only when ROCKET_MM_PIPE is set; the
     * non-pipe mm_compute path never touches these (stays the byte-identical
     * oracle). pm0/pn0/pMtile/pNtile/poff hold BOTH slots (2*BATCH), indexed
     * [slot*BATCH + i]; pnb[slot] is that slot's live tile count. */
    rocket_bo regcmd2, out2;                  /* 2nd regcmd + 2nd output BO        */
    rocket_task_desc *tasks2;                 /* BATCH (slot-1 task array)         */
    int    *pm0, *pn0, *pMtile, *pNtile;      /* 2*BATCH each (both slots)         */
    size_t *poff;                             /* 2*BATCH                           */
    int     pnb[2];                           /* live tiles per slot               */
} mm_bos;

/* ============================================================================
 * SECTION — Plan resolution, BO allocation & cube helpers
 * ==========================================================================*/

/* Resolve the tiling plan for M,K,N (honours the ROCKET_MM_* env knobs via
 * rocket_matmul_plan). Returns 0, or <0 if the shape is unsupported. */
int mm_plan_init(mm_plan *pl, int M, int K, int N);

/* Like mm_plan_init but PIN the output-tile Nt and/or the contraction-tile Kt to a
 * caller-given value (0 = leave free, the planner maximizes/caps it as usual). Used by
 * the cross-op fused FFN to force a producer matmul's Nt == a consumer matmul's Kt (a
 * multiple of 32), so the producer's fp16 output cube aliases the consumer's input
 * feature cube tile-for-tile (out_slot == in_slot; see cross-op-chaining). The pinned
 * dim is fixed; the free dims shrink to fit the CBUF. The ROCKET_MM_* env overrides are
 * NOT consulted (the pin is the contract). A pinned value must be a multiple of 32 and
 * not exceed its dimension. Returns 0, or <0 if the shape is unsupported / cannot fit. */
int mm_plan_init_pin(mm_plan *pl, int M, int K, int N, int pin_Nt, int pin_Kt);

/* Total fp16 element count of the full output cube (nMt*nNt canonical tile slots) —
 * the flat span an on-cube element-wise op (activation / ew-mul) covers. */
size_t mm_cube_elems(const mm_plan *pl);

/* Scatter a per-output-channel bias[N] (broadcast over all M rows) into the matmul
 * OUTPUT cube layout — the same canonical [nMt][nNt] tiling mm_compute_kacc_cube writes.
 * A flat element-wise add of an output cube + this bias cube then adds bias[n] to every
 * (m,n), staying cube-resident (the cross-op bias epilogue for a non-gated FFN). `dst`
 * spans mm_cube_elems(pl) fp16 elems; it is fully zeroed first, so pad lanes stay 0. */
void mm_scatter_bias_cube(const mm_plan *pl, _Float16 *dst, const _Float16 *bias);

/* Allocate all five BOs on `fd` for the plan (incl. the 32-bit IOVA check).
 * Returns 0 or <0; on failure any partially-allocated BOs are freed. */
int  mm_bos_alloc(int fd, const mm_plan *pl, mm_bos *b);
void mm_bos_free(int fd, mm_bos *b);

/* ============================================================================
 * SECTION — Weight / input packing & pack-time profiling
 * ==========================================================================*/

/* Scatter the weights B[N,K] / input A[M,K] into NPU tile layout in wt_all/in_all
 * (each does its own prep/fini). Return the pack time in ms (for profiling). */
double mm_pack_weights(int fd, const mm_plan *pl, mm_bos *b, const _Float16 *B);
double mm_pack_input  (int fd, const mm_plan *pl, mm_bos *b, const _Float16 *A);

/* A segment of a concatenated-along-N weight for the fused matmul: the
 * source Bseg[Nseg,K] occupies combined-N global output columns [g0, g0+Nseg). */
typedef struct { const _Float16 *Bseg; int g0; int Nseg; } mm_wt_seg;

/* Like mm_pack_weights, but the worker's column slice (pl->N columns starting at
 * combined-N global `worker_g0`) is scattered from a CONCATENATED weight: each
 * global column is resolved to its source segment. Same tile layout; writes only
 * live lanes. Lets several static weights share one fused matmul without
 * materializing a [sumN,K] concat buffer on the host. Returns pack ms. */
double mm_pack_weights_seg(int fd, const mm_plan *pl, mm_bos *b,
                           const mm_wt_seg *segs, int nseg, int worker_g0);

/* Dedup the per-worker A-pack: scatter A ONCE into a CPU buffer (mm_input_elems
 * fp16 elems), then each worker memcpy's it into its in_all BO (mm_load_input).
 * Used when all workers share the same input tiling plan.
 *   mm_pack_input_buf  — zero `dst` then scatter (for a fresh/unzeroed buffer).
 *   mm_scatter_input   — scatter ONLY (caller guarantees `dst` padding is already
 *                        zero, e.g. a persistent calloc'd buffer reused across
 *                        calls with a fixed plan). Both return pack ms. */
size_t mm_input_elems(const mm_plan *pl);
double mm_pack_input_buf(const mm_plan *pl, _Float16 *dst, const _Float16 *A);
double mm_scatter_input(const mm_plan *pl, _Float16 *dst, const _Float16 *A);
double mm_load_input(int fd, const mm_plan *pl, mm_bos *b, const _Float16 *packed);

/* Add externally-measured pack time to the ROCKET_MM_PROFILE aggregate. _pack
 * attributes to the input(A) scatter, _pack_b to the weight(B) scatter (the
 * streaming path re-packs B per call, so it reports it separately). */
void mm_prof_add_pack(double ms);
void mm_prof_add_pack_b(double ms);

/* ============================================================================
 * SECTION — Tile compute dispatchers (plain / kacc / cube / pipe / reuse)
 * ==========================================================================*/

/* Run the batched tile compute over already-packed in_all/wt_all, accumulate the
 * fp32 partials and write C[M,N]. `t_pack` is folded into the optional
 * ROCKET_MM_PROFILE line. Returns 0 or <0. Does NOT free any BO in `b`. */
int mm_compute(int fd, const mm_plan *pl, mm_bos *b, _Float16 *C, double t_pack);

/* NPU-side K-accumulation variant of mm_compute (ROCKET_KACC=1): accumulates
 * the nKt K-partials of each tile ON the NPU (DPU eltwise-add ping-pong) and reads
 * each output tile ONCE (read ∝ M·N, not M·N·nKt). Same signature/contract as
 * mm_compute. Returns -2 (NOT a hard error) for shapes whose last M-tile < 12,
 * where the EW MAX(M,12) stride floor would misread — caller falls back to
 * mm_compute for those. Accumulates in fp16 (~0.4% drift vs the fp32 host sum). */
int mm_compute_kacc(int fd, const mm_plan *pl, mm_bos *b, _Float16 *C, double t_pack);

/* CROSS-OP variant of mm_compute_kacc: leave the COMPLETE output cube in the caller's
 * `cube` BO instead of de-tiling it to row-major C. The result is the nMt*nNt tile
 * slots in CANONICAL order (slot (mi,ni) at offset (mi*nNt+ni)*out_slot — the same
 * [nMt][nKt] layout a consumer matmul's input cube wants when nNt==nKt), so a following
 * fp16 matmul can read `cube` directly as its input BO (alias in_all) with no host
 * de-tile/re-tile of the intermediate. On return `cube` is left CPU-visible (prepped for
 * read) so an on-cube element-wise op can run over cube->ptr; the caller flushes it back
 * to the device before the consumer matmul. K-accumulates on the NPU like mm_compute_kacc
 * (fp16 eltwise add). `cube` must be >= mm_cube_elems(pl)*sizeof(_Float16) bytes.
 *
 * Requires the whole result to be ONE NPU batch (nMt*nNt <= BATCH) so every tile co-exists
 * in `cube`; returns ROCKET_E_TILING (a routing sentinel, NOT a hard error) for shapes that
 * exceed it or that hit the same tiny-M EW-stride floor mm_compute_kacc guards — the caller
 * falls back to the host-handoff path for those. Returns 0 or <0 on a hard error. */
int mm_compute_kacc_cube(int fd, const mm_plan *pl, mm_bos *b, rocket_bo *cube, double t_pack);

/* Software-pipelined variant of mm_compute (ROCKET_MM_PIPE=1):
 * double-buffers the regcmd/output BOs so the CPU builds batch N+1 and de-tiles
 * batch N-1 while the NPU computes batch N — hiding the host readback behind NPU
 * compute. Same signature/contract as mm_compute, and because it processes the
 * SAME (mi,ni,ki) batches in the SAME order its fp32 result is BIT-IDENTICAL to
 * mm_compute. Requires the pipe scratch in `b` (mm_bos_alloc under ROCKET_MM_PIPE);
 * returns <0 on error. */
int mm_compute_pipe(int fd, const mm_plan *pl, mm_bos *b, _Float16 *C, double t_pack);

/* CBUF operand-reuse variant of mm_compute (ROCKET_REUSE=1 weight / 2 data):
 * reorders the tile loop so consecutive tasks in a job share the weight (mode 1) or
 * input (mode 2) tile, and sets the CNA WEIGHT_REUSE / DATA_REUSE bit so the CNA
 * reads it from CBUF instead of re-fetching from DRAM. CPU fp32-accum like
 * mm_compute and BIT-IDENTICAL to it when the CBUF persists correctly (the
 * load-bearing check). `mode` is 1 (weight) or 2 (data). Returns 0 or <0. */
int mm_compute_reuse(int fd, const mm_plan *pl, mm_bos *b, _Float16 *C,
                     double t_pack, int mode);

#endif /* ROCKET_MATMUL_INTERNAL_H */
