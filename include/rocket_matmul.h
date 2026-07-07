// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_MATMUL_H
#define ROCKET_MATMUL_H

#include <stdint.h>   /* int8_t / int32_t in the int8 matmul API */
#include <stddef.h>   /* size_t (resident-weight byte accounting) */

/* The fp16 matmul API takes _Float16 operands. _Float16 is a compiler extension
 * (GCC/Clang on aarch64, the RK3588 target). A consumer whose compiler lacks it can
 * predefine ROCKET_FLOAT16_T to a 16-bit storage type before including this header. */
#ifdef ROCKET_FLOAT16_T
typedef ROCKET_FLOAT16_T _Float16;
#endif

/*
 * rocket_matmul — high-level tiled fp16 matmul on the rocket NPU, built on the
 * validated single-task generator (npu_regcmd.c) + the rocket shim.
 *
 * Computes  C[M,N] = A[M,K] * B[N,K]^T   (the gen_matmul convention: each output
 * channel n is the dot product of input row m with weight row n). All operands
 * are _Float16, row-major: A is M*K, B is N*K, C is M*N.
 *
 * Tiles the problem so each sub-matmul fits the 12x32KB CBUF:
 *   - M (rows) and N (output channels) are split into INDEPENDENT output blocks.
 *   - K (contraction) is split when it is too large to fit a tile, with the
 *     partial products accumulated on the CPU in fp32 (so large-K stays precise).
 * This single-fd path issues one NPU job per (M,N,K) tile, sequentially; the
 * 3-core fan-out is rocket_matmul_fp16_mt below.
 *
 * Requires K%32==0, N%16==0, M%4==0. (M==1 is NOT a plan-supported shape: the
 * height-1 conv geometry is broken — see the cosine-sim correctness matrix. The
 * one-shot rocket_matmul_fp16 below accepts M==1 by padding to a height-4 tile
 * internally; the multicore/streaming/prepacked paths do not pad and reject it.
 * Single-vector callers on those paths: pad M up to 4.)
 */

/* Compute a tiling plan for M,K,N. Writes the per-tile sizes (Mt<=M, Kt<=K,
 * Nt<=N) and returns the total number of NPU jobs it will take (a POSITIVE count),
 * or a negative enum rocket_status (ROCKET_E_SHAPE) if the shape is unsupported
 * (need K%32==0, N%16==0, M%4==0). Pure/no hardware.
 * CONTRACT: test the result with `< 0` for failure — NOT `!= 0`. The matmul
 * *_plan() previews return a positive job count on success, whereas the single-op
 * previews (rocket_conv2d_plan / rocket_pool_fp16_plan / …) return ROCKET_OK (0);
 * `< 0` is the one error test that is correct for all of them. */

#ifdef __cplusplus
extern "C" {
#endif
int rocket_matmul_plan(int M, int K, int N, int *Mt, int *Kt, int *Nt);

/* Run the tiled matmul on the NPU. `fd` is an open rocket device (rocket_open()).
 * Returns 0 on success, negative on error. M==1 is padded to 4 internally (the
 * documented single-vector case); all other M must satisfy M%4==0.
 *
 * NUMERICS KNOB: K-accumulation runs on the NPU (fp16 DPU-eltwise accumulate) by
 * DEFAULT — the operating mode, +19% throughput. It is FASTER but CHANGES THE RESULT
 * (~0.4% drift vs a host fp64 sum). Opt out with ROCKET_KACC=0 (or ROCKET_NO_KACC) to
 * get the byte-exact fp64-accum reference path. (DATA_REUSE rides along automatically;
 * see ROCKET_REUSE.) */
int rocket_matmul_fp16(int fd, int M, int K, int N,
                       const _Float16 *A, const _Float16 *B, _Float16 *C);

/* Batched same-shape fp16 matmul: nbatch INDEPENDENT C[i] = A[i]·B[i]^T, all sharing
 * one (M,K,N), run as a SINGLE NPU job stream. The per-item output tiles flow through
 * one submit + one fence wait for the whole group (vs nbatch submit/wait round-trips),
 * and with ROCKET_BATCH_SUBMIT=1 (+ the kernel half) the per-item regcmds chain so the
 * group also fires one completion IRQ — the contiguous self-chaining dispatch-floor
 * lever, extended across same-shape matmuls. Use it to collapse a set of small,
 * dispatch-bound GEMMs that share a shape (e.g. flash attention's per-head QK, and
 * separately AV, matmuls within a worker's head range).
 *
 * A/B/C are nbatch-long arrays of row-major operand/result pointers (A[i] is [M,K],
 * B[i] is [N,K], C[i] is [M,N]). BIT-IDENTICAL to calling rocket_matmul_fp16 per item
 * (same plan, fp16 pack, CPU fp32 K-accum, de-tile/narrow); it does NOT consult
 * ROCKET_KACC/REUSE/PIPE (the single-matmul perf variants) — the batch lever is
 * orthogonal. Requires M%4, K%32, N%16; nbatch==1 falls through to rocket_matmul_fp16.
 * The caller bounds nbatch (the in/wt BO and the fp32 accumulators scale with it).
 * Returns 0 on success, negative on error. */
int rocket_matmul_fp16_batch(int fd, int M, int K, int N, int nbatch,
                             const _Float16 *const *A, const _Float16 *const *B,
                             _Float16 *const *C);

/* ---- persistent batched-matmul context -----------------------------------
 * The pack-once-keep-resident form of rocket_matmul_fp16_batch, for a caller that
 * runs the SAME-shape batch repeatedly (flash attention runs one per layer per
 * forward). rocket_matmul_fp16_batch allocates its in/wt/out NPU BOs AND full-zeroes
 * the in/wt BO on EVERY call (the pad lanes must read zero); held resident, the BOs
 * are allocated once (grown only on a larger shape) and the full-BO zero is skipped
 * whenever the (M,K,N,nbatch) layout repeats — for a fixed plan the scatter rewrites
 * the same live lanes and never touches padding, so padding zeroed once stays zero.
 * This reclaims the per-call `pack` cost (the dominant chained-FA overhead).
 *
 * The context is bound to ONE fd (create with the fd it will run on). A single
 * context tracks ONE last-layout, so a caller alternating two shapes (e.g. flash
 * attention's QK and AV) should use TWO contexts — one per shape — so neither sees a
 * layout switch that forces a re-zero. Not thread-safe: a context mutates its
 * resident BOs/scratch per run, so give each concurrent worker its OWN context.
 *
 *   b = rocket_mm_batch_create(fd);
 *   ... per call (same or a new shape):
 *       rocket_mm_batch_run(b, M, K, N, nbatch, A, B, C);
 *   rocket_mm_batch_free(b);
 *
 * rocket_mm_batch_run is BIT-IDENTICAL to rocket_matmul_fp16_batch (same plan, pack,
 * CPU fp32 K-accum, de-tile/narrow); the only difference is BO/scratch lifetime.
 * Same shape contract (M%4, K%32, N%16; nbatch==1 falls through to rocket_matmul_fp16).
 * Returns 0 on success, negative on error (the resident BOs are kept for the next run;
 * free with rocket_mm_batch_free). */
typedef struct rocket_mm_batch rocket_mm_batch;
rocket_mm_batch *rocket_mm_batch_create(int fd);
void             rocket_mm_batch_free(rocket_mm_batch *b);
int rocket_mm_batch_run(rocket_mm_batch *b, int M, int K, int N, int nbatch,
                        const _Float16 *const *A, const _Float16 *const *B,
                        _Float16 *const *C);

/* fp16 x fp16 -> FP32 output. Same C[M,N] = A[M,K] * B[N,K]^T convention and
 * the identical fp16 input packing as rocket_matmul_fp16, but the DPU writes the full
 * fp32 accumulator (fp32tofp16=0) instead of narrowing each K-partial to fp16, and the
 * K-partials are summed on the host in fp64. So this removes BOTH the per-K-tile fp16
 * rounding (which the fp16 path applies before host accumulation) AND the final
 * narrowing — the result is the genuine fp32 dot product (the inputs are still fp16).
 * Worth it after activation scaling / for large-K accuracy; costs 2x the output
 * readback bytes (fp32 vs fp16 cube), so it is a separate opt-in entry, not the default
 * (the prefill path is readback-bound). C is `float`. M==1 padded to 4 internally;
 * other M must satisfy M%4==0 (K%32, N%16). Returns 0 on success. */
int rocket_matmul_fp16_f32out(int fd, int M, int K, int N,
                              const _Float16 *A, const _Float16 *B, float *C);

/* ---- int8 tiled matmul -------------------------------------------------
 * Same C[M,N] = A[M,K] * B[N,K]^T convention, but int8 x int8 -> int32: A and B
 * are PRE-QUANTIZED int8 (row-major; the backend applies per-row activation /
 * per-channel weight scales), C is the RAW int32 accumulation (the backend
 * dequantizes: C[m,n] * a_scale[m] * b_scale[n]). HW does int8xint8->int32; K is
 * split with int32 partials summed on the HOST (no on-chip requant), matching the
 * fp16 path's host K-accum. Alignment is stricter than fp16: K%32, N%32 (the int8
 * weight k-group is 32, not 16), and M%4==0. (As with fp16, M==1 is broken on the
 * HW height-1 geometry — every dtype one-shot here pads M==1->4 internally and
 * returns row 0; the plans + resident-weight paths require M%4==0 and reject M==1,
 * so pad single vectors to 4 caller-side for those.)
 *
 * rocket_matmul_plan_int8 previews the tiling (pure, no HW); returns NPU job count
 * or <0 on unsupported shape. rocket_matmul_int8 runs it on `fd`. */
int rocket_matmul_plan_int8(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_int8(int fd, int M, int K, int N,
                       const int8_t *A, const int8_t *B, int32_t *C);

/* ---- int4 tiled matmul -------------------------------------------------
 * int4 x int4 -> int16 (the NPU output), host-accumulated to int32 C. A and B are
 * PRE-QUANTIZED int4 values stored one-per-int8_t in [-8,7] (the backend owns the
 * scales/Hadamard); C is the RAW int32 accumulation. int4 deltas vs int8: feature
 * cube C2=32, weight layout (N/64,K/32,64,32) [weight_int4], int16 output cube
 * C2=8; in/wt are NIBBLE-packed (2 int4/byte). Alignment: K%32, N%64 (int4's
 * N-group is 64, not int8's 32), (M%4||1).
 *
 * int16-output SATURATION: each K-tile partial is read back as int16, so a
 * Kt-pass whose |sum| exceeds 32767 saturates (lossy). The plan keeps Kt within
 * the CBUF limit; whether that also stays unsaturated depends on the data
 * magnitude (fine for Hadamard-rotated/scaled weights; for raw full-range int4
 * keep Kt small). rocket_matmul_plan_int4 previews tiling (pure). */
int rocket_matmul_plan_int4(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_int4(int fd, int M, int K, int N,
                       const int8_t *A, const int8_t *B, int32_t *C);

/* int4 matmul with an explicit int16-saturation Kt cap: kt_cap>0 caps the K-tile
 * (rounded to %32) so a [-8,7] partial (|.|<=64*Kt) cannot overflow the int16 output;
 * kt_cap=0 == rocket_matmul_int4 (no cap). In-model callers pass kt_cap=480. */
int rocket_matmul_int4_ex(int fd, int M, int K, int N,
                          const int8_t *A, const int8_t *B, int32_t *C, int kt_cap);

/* GROUP-WISE int4: C_f[M,N] = sum_g a_scale[m,g]*b_scale[n,g] * (int4 partial of
 * K-group g), fp32-accumulated. A,B are pre-quantized int4 in [-7,7]; a_scale is
 * [M*nG], b_scale is [N*nG] (nG = K/group), the per-group quant scales. The K-tile
 * is forced to `group` (one tile == one quant group), so a separate dequant scale
 * applies per K-slice -- the W4 quality lever toward int8 fidelity (GPTQ/AWQ regime).
 * group must divide K, be %32, M%4, and keep 49*group < 32767 (int16 saturation).
 * Returns 0, or <0 on unsupported shape. */
int rocket_matmul_int4_groupwise(int fd, int M, int K, int N,
                                 const int8_t *A, const int8_t *B,
                                 const float *a_scale, const float *b_scale,
                                 float *Cf, int group);

/* ---- int16 tiling plan (shared geometry) ------------------------------------
 * Previews the int16 tile shape (pure). int16 = fp16's INPUT geometry (feature
 * cube C2=8, weight (N/16,K/32,16,32) [weight_int16 == weight_fp16], 2 B elements)
 * with a 4 B output cube (C2=4); the bf16/tf32 paths reuse this plan. Alignment:
 * K%32, N%16, (M%4||1).
 *
 * There is no native int16 x int16 -> int32 matmul on RK3588 (the int32-output conv
 * writes a single 1x16 tile and never iterates; only the SATURATING int16-output
 * transposed primitive iterates, N<=32 — see tests/matmul_int16_rocket.c). Use
 * rocket_matmul_int16_exact (below) for a correct full-precision int16 matmul. */
int rocket_matmul_plan_int16(int M, int K, int N, int *Mt, int *Kt, int *Nt);

/* ---- bit-exact int16 x int16 -> int64 (the SUPPORTED full-precision route) -----
 * The RK3588 NPU has NO native int16 matmul output (only int8->int32 and fp16->fp32
 * have native output paths). The native int16 conv can only be driven as an
 * int16->int16 SATURATING transposed-output primitive (tp_org_en; see
 * tests/matmul_int16_rocket.c), which cannot represent a full int32/int64 product.
 * For full precision, decompose each int16 into two signed bytes and run four
 * PROVEN int8 matmuls, recombining in int64: C = 65536*(Ah.Bh) + 256*(Ah.Bl +
 * Al.Bh) + Al.Bl. Bit-exact, no saturation, ~4x int8 cost (completeness, not
 * speed). Inputs are clamped to [-32768, 32639] (the signed/signed split cannot
 * reach the top 128 int16 codes; bit-exact within that domain). C is int64
 * [M*N] row-major. Alignment: K%32, N%32, (M%4||1). Returns 0 / negative. */
int rocket_matmul_int16_exact(int fd, int M, int K, int N,
                              const int16_t *A, const int16_t *B, int64_t *C);

/* ---- bf16 tiled matmul --------------------------------------------------------
 * bf16 x bf16 -> fp32. bf16 has fp32's 8-bit exponent at fp16's 2-byte cost, so it
 * carries full activation range with NO per-row scaling (which the fp16 path needs
 * an amax scan for). The CNA/DPU MAC does bf16xbf16 at precision 3 and accumulates
 * to fp32 — exact products + fp32 accumulate (max rel ~1e-6).
 *
 * bf16 shares int16's NPU geometry (2-byte input cube C2=8, weight (N/16,K/32,
 * 16,32), 4-byte output cube C2=4) but is a FLOAT path: output fp32, host fp32
 * (double) K-accum, NO saturation. The interface takes fp32 A/B and TRUNCATES to
 * bf16 during the tile scatter (no separate pack / scale / unscale); C is fp32
 * [M*N] row-major. Alignment K%32, N%16, (M%4||1) — same as fp16/int16.
 * rocket_matmul_plan_bf16 previews the tiling (pure; == the int16 plan). */
int rocket_matmul_plan_bf16(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_bf16(int fd, int M, int K, int N,
                       const float *A, const float *B, float *C);

/* ---- fast bf16 paths (rocket_bf16_stream.c) -----------------------------------
 * The single-fd rocket_matmul_bf16 above opens nothing persistent and allocs+frees
 * every BO per call, so an in-model bf16 prefill ran on one core with a full BO
 * alloc/free per matmul. These hoist both costs out of the loop, the bf16 siblings
 * of rocket_matmul_fp16_mt / rocket_matmul_fp16_stream.
 *
 * rocket_matmul_bf16_mt: split the output columns N across `nthreads` worker fds
 * (one scheduling entity per fd => the 3 NPU cores run in parallel), each running the
 * unchanged single-fd rocket_matmul_bf16 on its slice. Opens/closes its own fds — do
 * NOT pass one in; nthreads clamped to [1,8]. fp32 A/B/C, output fp32 [M,N] row-major. */
int rocket_matmul_bf16_mt(int M, int K, int N,
                          const float *A, const float *B, float *C, int nthreads);

/* Streaming bf16: a persistent context keeps the worker fds AND the per-(M,K,N)
 * scratch BOs (input/weight/output/regcmd) resident, re-packing only A and B each
 * call. bf16 weights are used once per prefill token batch, so there is no resident-
 * weight (prepacked) variant — streaming is the right shape. The output is bit-
 * identical to rocket_matmul_bf16 at nthreads=1 and the same valid bf16 product within
 * float reassociation at nthreads>1. Returns 0, or <0 to fall back to the per-call mt
 * path (shape cache full or unsupported). Free the stream before exit.
 *
 * NOT thread-safe: the stream mutates its shared per-shape scratch (the A/B re-pack
 * buffers + per-worker BOs) every call and grows its shape cache on first use of a
 * new (M,K,N), so concurrent calls on ONE stream race. Give each concurrent host
 * thread its own stream. (The per-call fan-out across the 3 NPU cores is internal and
 * separate — one stream already drives all cores.) */
typedef struct rocket_bf16_stream rocket_bf16_stream;
rocket_bf16_stream *rocket_bf16_stream_create(int nthreads);
void                rocket_bf16_stream_free(rocket_bf16_stream *s);
int rocket_matmul_bf16_stream(rocket_bf16_stream *s, int M, int K, int N,
                              const float *A, const float *B, float *C);

/* ---- tf32 tiled matmul --------------------------------------------------------
 * tf32 x tf32 -> fp32. The only 4-byte-INPUT path. tf32 = fp32 RANGE (8-bit exp) +
 * fp16 PRECISION (10-bit mantissa) in a 4-byte fp32 container: feed RAW fp32, the
 * MAC rounds the mantissa to 10 bits and accumulates in fp32 (genuine 10-bit tf32,
 * max rel ~1.5e-7 vs a tf32-rounded reference).
 *
 * Like bf16 it needs NO activation scaling and is a FLOAT path (output fp32, host
 * double K-accum, NO saturation); the interface takes fp32 A/B straight through
 * with NO truncation on the scatter (the HW does the tf32 rounding). It differs
 * from bf16 only in the 4-byte geometry: feature cube C2=4, weight tile
 * (N/16,K/16,16,16) — K-group HALVES to 16 (N-group stays 16) — banks 4 B/elem,
 * data_entries K/16. Alignment K%16, N%16, (M%4||1). gen_matmul_tf32 sets the
 * per-stage precision (CNA/CORE=7 tf32, DPU=fp32). C is fp32 [M*N] row-major.
 *
 * LOWEST-value rung (half-rate MAC; bf16 already gives fp32 range at full speed) —
 * completeness, not a workload. No in-model backend. rocket_matmul_plan_tf32
 * previews the tiling (pure, no HW). */
int rocket_matmul_plan_tf32(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_tf32(int fd, int M, int K, int N,
                       const float *A, const float *B, float *C);

/* Multi-core: split N across `nthreads` worker threads, each with its OWN rocket
 * fd, each computing a contiguous column-slice C[:,n0:n1] = A*B[n0:n1]^T via the
 * (unchanged) single-fd rocket_matmul_fp16. The rocket driver runs one drm_sched
 * per NPU core and one scheduling entity per fd, so independent fds let the 3
 * cores run in parallel (one fd serializes onto one core). Opens/closes its own
 * fds — do NOT pass one in. nthreads is clamped to [1,8]; ~3 saturates the 3
 * cores (4 can edge higher by hiding CPU-side gaps). Returns 0 / negative. */
int rocket_matmul_fp16_mt(int M, int K, int N,
                          const _Float16 *A, const _Float16 *B, _Float16 *C,
                          int nthreads);

/* ---- pack-weights-once path-----------------------------
 * For repeated matmuls that reuse the SAME weights B (e.g. a model's static
 * weight matrices across forward passes). Packs B into resident NPU BOs once and
 * keeps the worker fds + scratch BOs alive, so each matmul only packs A.
 *
 *   ctx = rocket_ctx_create(nthreads);              // persistent worker fds
 *   w   = rocket_weights_pack(ctx, M, K, N, B);     // scatter B once
 *   ... per forward pass:
 *       rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, w);
 *   ... teardown (weights BEFORE ctx, they hold BOs on the ctx fds):
 *   rocket_weights_free(ctx, w);
 *   rocket_ctx_free(ctx);
 *
 * The _prepacked call's M MAY DIFFER from the _pack M: the resident weight
 * layout depends only on K, N, and the K/N tiling (Kt/Nt) — which is M-independent for
 * all M >= MAX_TILE (256), where Mt is capped. So a weight packed at warmup-M is reused
 * directly at prefill-M whenever the tiling matches (the common LLM/Whisper case), with
 * no re-pack. K and N must still match; an incompatible tiling (e.g. packed at M>=256,
 * called at a small M whose Kt grows) returns -2 so the caller re-packs for that M.
 *
 * NOT thread-safe: a ctx mutates its shared per-shape scratch (the A-pack buffer +
 * per-worker BOs that a compute borrows) every call and grows its shape/weight caches
 * on first use, so concurrent rocket_weights_pack / _prepacked calls on ONE ctx race.
 * Give each concurrent host thread its own ctx (the per-call fan-out across the 3 NPU
 * cores is internal — one ctx already drives all cores). A rocket_weights handle is
 * tied to the ctx it was packed on. */
typedef struct rocket_ctx     rocket_ctx;
typedef struct rocket_weights rocket_weights;

rocket_ctx     *rocket_ctx_create(int nthreads);
void            rocket_ctx_free(rocket_ctx *ctx);

rocket_weights *rocket_weights_pack(rocket_ctx *ctx, int M, int K, int N, const _Float16 *B);
void            rocket_weights_free(rocket_ctx *ctx, rocket_weights *w);

int rocket_matmul_fp16_prepacked(rocket_ctx *ctx, int M, int K, int N,
                                 const _Float16 *A, _Float16 *C, rocket_weights *w);

/* ---- streaming path-----------------------------------
 * For LLM prefill, where each weight is used ONCE per generation (so caching
 * resident weights like the prepacked path would waste ~20GB of NPU BOs for a
 * 12B model). The streaming context keeps the per-worker fds AND the scratch/
 * weight BOs resident, but RE-PACKS B every call. The scratch/weight BOs are
 * cached per distinct (M,K,N) shape, so an LLM's small, repeating shape set
 * (q/k/v/o/gate/up/down/lm_head) pays the BO alloc + fd open cost once total,
 * not once per matmul -- without holding the whole model resident. It also
 * shares the A-pack across workers (one scatter, not nthreads).
 *
 *   s = rocket_stream_create(nthreads);
 *   ... per matmul (any cached or new shape):
 *       rocket_matmul_fp16_stream(s, M, K, N, A, B, C);
 *   rocket_stream_free(s);
 *
 * Returns 0 on success; <0 to tell the caller to fall back to the per-call mt
 * path (e.g. the shape cache is full or the shape is unsupported).
 *
 * NOT thread-safe: a stream mutates its shared per-shape scratch (the A-pack buffer +
 * per-worker weight/IO BOs, re-packed each call) and grows its shape cache on first use
 * of a new (M,K,N), so concurrent calls on ONE stream race. Give each concurrent host
 * thread its own stream (the per-call fan-out across the 3 NPU cores is internal — one
 * stream already drives all cores). */
typedef struct rocket_stream rocket_stream;

rocket_stream *rocket_stream_create(int nthreads);
void           rocket_stream_free(rocket_stream *s);

int rocket_matmul_fp16_stream(rocket_stream *s, int M, int K, int N,
                              const _Float16 *A, const _Float16 *B, _Float16 *C);

/* ---- fused streaming matmul---------------------------
 * Several STATIC weights that share one activation A (e.g. a transformer's
 * Q/K/V or gate/up projections) concatenated along N and run as ONE matmul:
 *   C[M, sum(Ns[i])] = A[M,K] * [B0 ; B1 ; ...]^T
 * Each Bs[i] is [Ns[i], K] row-major; the driver scatters each weight directly
 * into its global-N slice of the combined weight BO (no host-side [sumN,K]
 * concat copy), runs one N-split multicore matmul over the combined N, and
 * writes C as [M, sum(Ns)] row-major (the caller splits the columns back into
 * each projection's own output). This collapses `nseg` NPU jobs + A-packs into
 * one and grows N for better core utilization; it does NOT reduce the
 * (hardware-mandatory) weight-scatter byte count -- the weights are distinct.
 *
 * Same contract as rocket_matmul_fp16_stream (0, or <0 to tell the caller to
 * fall back to running the members individually). nseg in [1, ROCKET_MAX_FUSE]. */
#define ROCKET_MAX_FUSE 8

int rocket_matmul_fp16_stream_fused(rocket_stream *s, int M, int K,
                                    const _Float16 *const *Bs, const int *Ns, int nseg,
                                    const _Float16 *A, _Float16 *C);

/* ---- resident int8 (W8A8) path-----------------------------------
 * The int8 analogue of the fp16 pack-weights-once path above: a pre-quantized
 * int8 weight B[N,K] is scattered into resident per-worker NPU BOs ONCE and
 * reused across forward passes, so each matmul only packs A (the int8 weight
 * scatter — "packB" — is the per-call cost this removes, as for fp16). N is
 * fanned across worker fds (each fd has its own 4GB IOVA window;
 * int8's ~11GB whole-model footprint fits across 5 fds with room to spare).
 *
 *   ctx = rocket_i8_ctx_create(nthreads);            // persistent worker fds
 *   w   = rocket_i8_weights_pack(ctx, M, K, N, qB);  // scatter int8 B once
 *   ... per forward pass (qA pre-quantized int8 [M,K]):
 *       rocket_matmul_int8_prepacked(ctx, M, K, N, qA, C32, w);   // C32 raw int32
 *   ... teardown (weights BEFORE ctx; they hold BOs on the ctx fds):
 *   rocket_i8_weights_free(ctx, w);
 *   rocket_i8_ctx_free(ctx);
 *
 * Same int8 convention as rocket_matmul_int8: A/B are PRE-QUANTIZED int8 (the
 * backend owns the per-row/per-channel scales + any Hadamard rotation), C is the
 * RAW int32 accumulation (host int64 K-accum — int8 NPU K-accum is HW-dead, the
 * EW operand DMA is <=16-bit). Alignment: K%32, N%32, M%4==0 (resident paths do
 * NOT pad M==1 — pad single vectors to 4 caller-side); the per-worker
 * N-slice is rounded to a multiple of 32. The (M,K,N) given to _prepacked MUST
 * match those given to _pack (the tile plan, hence the resident layout, depends
 * on all three). Result is BIT-EXACT to the one-shot rocket_matmul_int8.
 *
 * NOT thread-safe (as rocket_ctx): a ctx mutates its shared per-shape scratch each
 * call — give each concurrent host thread its own ctx. */
typedef struct rocket_i8_ctx     rocket_i8_ctx;
typedef struct rocket_i8_weights rocket_i8_weights;

rocket_i8_ctx     *rocket_i8_ctx_create(int nthreads);
void               rocket_i8_ctx_free(rocket_i8_ctx *ctx);

rocket_i8_weights *rocket_i8_weights_pack(rocket_i8_ctx *ctx, int M, int K, int N,
                                          const int8_t *B);
void               rocket_i8_weights_free(rocket_i8_ctx *ctx, rocket_i8_weights *w);
/* Resident NPU-BO footprint of a packed weight, in bytes (the per-worker weight
 * tiles; the shared scratch is cached across weights and not counted). */
size_t             rocket_i8_weights_bytes(const rocket_i8_weights *w);

int rocket_matmul_int8_prepacked(rocket_i8_ctx *ctx, int M, int K, int N,
                                 const int8_t *A, int32_t *C, rocket_i8_weights *w);

/* ---- resident int4 (W4A4) path — int4 sibling of the resident int8 path.
 * Same usage; A/B pre-quantized int4 one-per-int8_t in [-8,7], C raw int32. N
 * fanned across worker fds; weight scattered once into resident int4 NPU BOs.
 * Built to measure int4's multicore throughput vs the resident fp16/int8 bars.
 * NOT thread-safe (as rocket_ctx): one ctx per concurrent host thread. */
typedef struct rocket_i4_ctx     rocket_i4_ctx;
typedef struct rocket_i4_weights rocket_i4_weights;

rocket_i4_ctx     *rocket_i4_ctx_create(int nthreads);
void               rocket_i4_ctx_free(rocket_i4_ctx *ctx);
rocket_i4_weights *rocket_i4_weights_pack(rocket_i4_ctx *ctx, int M, int K, int N, const int8_t *B);
void               rocket_i4_weights_free(rocket_i4_ctx *ctx, rocket_i4_weights *w);
int rocket_matmul_int4_prepacked(rocket_i4_ctx *ctx, int M, int K, int N,
                                 const int8_t *A, int32_t *C, rocket_i4_weights *w);

/* GROUP-WISE resident int4 — the resident sibling of rocket_matmul_int4_groupwise and
 * the in-model W4A4 path. Pack the weight once with rocket_i4_weights_pack_gw (K-tile
 * forced to `group`, group%32, K%group, 49*group<32767); then each call quantizes only
 * A and returns the fp32 dequantized result Cf[M,N] = sum_g a_scale[m,g]*b_scale[n,g]*
 * (int4 partial of K-group g). a_scale is [M*nG], b_scale is [N*nG] (nG = K/group), the
 * per-row / per-channel per-group scales. Hadamard (when used) is baked into the packed
 * weight + applied to A by the caller — the rotation is product-preserving, so no driver
 * support is needed. C raw-int32 path stays rocket_matmul_int4_prepacked. */
rocket_i4_weights *rocket_i4_weights_pack_gw(rocket_i4_ctx *ctx, int M, int K, int N,
                                             const int8_t *B, int group);
int rocket_matmul_int4_prepacked_gw(rocket_i4_ctx *ctx, int M, int K, int N,
                                    const int8_t *A, const float *a_scale,
                                    const float *b_scale, float *Cf, rocket_i4_weights *w);
/* Resident NPU-BO footprint of a packed int4 weight (per-worker tiles; excludes the
 * shared scratch). The int4 sibling of rocket_i8_weights_bytes. */
size_t rocket_i4_weights_bytes(const rocket_i4_weights *w);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_MATMUL_H */
