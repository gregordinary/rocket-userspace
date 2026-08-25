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
/* ---- the RK3576's own int8 matmul --------------------------------------
 *
 * Same C[M,N] = A[M,K] * B[N,K]^T convention, and a DIFFERENT contract from
 * rocket_matmul_int8 above, which is why it is a separate entry rather than a
 * dispatch behind it: the RK3576's DPU writes an int8 surface through its requant,
 * where the RK3588 writes the raw int32. So:
 *
 *     C[m][n] = sat8( round( ( sum_k A[m][k]*B[n][k] + bias[n] ) * scale ) )
 *
 * `bias` may be NULL. `scale` is per-tensor and must be positive — the DPU's OUT_CVT
 * multiplier gates the whole bias stage, and a zero there writes a full, correctly
 * sized, entirely empty surface with nothing to fault on. Requires K%32 and N%32.
 *
 * WHAT IT WILL AND WILL NOT DO, both measured on the part:
 *
 *   - M carries NO constraint. M=1 computes, and so does every plane the M rows can
 *     be cut into. This is the opposite of the RK3588, where rows are the conv's
 *     spatial height, height < 4 mis-computes, and M==1 is padded to 4.
 *   - N is TILED, and the tile is what buys throughput: a submit costs ~1.4 ms
 *     whatever it carries, so MACs per submit is the only lever and N is the only
 *     free axis. ROCKET_RK3576_MM_NT overrides the default tile.
 *   - K IS NOT tiled here, and the entry REFUSES a K it cannot contract in one task.
 *     The boundary is K >= 6176, for every (M, N) enumerated: the planner shrinks the
 *     output-channel tile by halves to 32 channels before it gives up, so what decides
 *     it is the CBUF pool against a one-group weight cube, a condition on K alone. Past
 *     it the return is ROCKET_E_UNSUPPORTED and nothing is submitted. The K-split route
 *     that used to run those shapes — the int32 entry below — wedges the NPU across
 *     processes until the board is rebooted, so a caller that needs a larger K runs it
 *     on the host. ROCKET_RK3576_MM_KSPLIT=1 takes the route anyway, for measurement.
 *   - BELOW that boundary K is free, but it is not FLAT: the output tile collapses from
 *     1024 channels to 32 in one step at K > 4992, which multiplies the submit count by
 *     32 at M=512 N=2048 with nothing refusing and nothing computing wrong. A caller
 *     picking a K for speed should stay at or below 4608.
 *
 * WHAT ONE PER-TENSOR OUTPUT SCALE COSTS AN LLM. `scale` is the OUTPUT's alone: A and B
 * arrive quantized, so every input-side choice is the caller's. Simulated exactly on two
 * real models' own prefill activations at M=512, every offloadable projection replaced,
 * wikitext-2 perplexity against an fp16 floor of 1.000x:
 *
 *     per-tensor A and B          SmolLM2-1.7B 2.26x      Qwen2.5-1.5B 1.11x
 *     per-row A, per-channel B    SmolLM2-1.7B 86x        Qwen2.5-1.5B 1.03x
 *
 * TWO THINGS FOLLOW FOR A CALLER. The cost is model-dependent by more than an order of
 * magnitude, so measure a candidate model end to end — a per-GEMM norm does not predict
 * it, and prefers the arm that is 86x worse. And per-axis INPUT scales are a coin flip
 * here rather than an improvement: free at this interface, better on every per-GEMM
 * number, and on one model of two catastrophic. Mechanism open — the error-concentration
 * story does not separate the models, and SmolLM2's damage is calibration rather than
 * ranking (top-1 60.7% against the per-tensor arm's 62.0%, perplexity 25 -> 964).
 *
 * WHAT RECOVERS IT, on both models, and neither half alone: an orthonormal Hadamard
 * rotation along K before quantization (host-side, no library change, 1.31x / 1.02x on
 * its own) plus a per-output-channel output requant (a library change — this DPU's
 * epilogue is `(acc + A[oc]) * C[oc]` then one `(v*MUL)>>SHIFT`, and the per-axis
 * convolution path programs that C ramp today; 1.54x / 1.06x on its own). Together they
 * measure 1.004x and 1.002x at 97% top-1 — with an EXACT per-column scale, so that is
 * the ceiling of the route rather than what the integer ramp would deliver.
 * [host arithmetic over two real models, 2026-08-10]
 *
 * rocket_matmul_plan_int8_rk3576 previews the tiling (pure, no HW) and returns the
 * NPU job count, or <0 on a shape this part cannot run — which now agrees with what
 * the entry does, since both refuse the same K. [HW sweep, H96 MAX M9] */
int rocket_matmul_plan_int8_rk3576(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_int8_rk3576(int fd, int M, int K, int N,
                              const int8_t *A, const int8_t *B,
                              const int32_t *bias, float scale, int8_t *C);

/* ---- the same matmul with a PER-OUTPUT-COLUMN output scale -------------------
 *
 *     C[m][n] = sat8( round_half_to_even( (sum_k A[m][k]*B[n][k] + bias[n])
 *                                         * scale_n[n] ) )
 *
 * Every shape rule, refusal and hazard of rocket_matmul_int8_rk3576() applies unchanged
 * — this is that entry with the output scale given per column instead of per tensor.
 * `scale_n` has N entries and every one must be positive.
 *
 * WHY IT EXISTS. One per-tensor output scale is what blocks an LLM from using the
 * per-tensor entry: simulated on two real models' own prefill activations, it costs
 * 2.26x perplexity on SmolLM2-1.7B and 1.11x on Qwen2.5-1.5B, and a per-column output
 * requant recovers those to 1.54x / 1.06x on its own and to 1.004x / 1.002x composed
 * with a host-side Hadamard rotation along K. [host arithmetic, two models, 2026-08-10]
 *
 * HOW IT IS EXPRESSED, and where its accuracy therefore stops. The DPU has ONE (MUL,
 * SHIFT) per task, so the per-column part rides on the coefficient group's int16 C
 * term: the epilogue is `(acc + bias[n]) * C[n]` in saturating int32 and then the shared
 * `(v*MUL)>>SHIFT`. Column n's gain is C[n] * MUL/2^SHIFT, so its resolution is
 * 0.5/C[n] — and C is capped not by its int16 field but by that column's worst-case
 * accumulator, because `(acc + bias)*C` must not overflow int32. At a GEMM's contraction
 * depth that cap binds hard: the bound is 128*sum_k|B[n][k]|, so a K of a few thousand
 * leaves C in the low hundreds whatever the field allows. `worst_rel_err` (may be NULL)
 * reports the largest relative gain error the ramp actually delivered over all columns,
 * which is the only number that separates this from an exact per-column scale — READ IT
 * rather than assuming the 1.004x above, which was simulated with exact scales and is
 * the route's ceiling, not this entry's.
 *
 * The ramp is planned PER N TILE, since the (MUL, SHIFT) is per task. A tile whose
 * columns share a scale loses nothing; a wide spread inside one tile is where the
 * resolution goes — forcing eight tiles took an 8x-spread K=1024 cell from 1.53% to
 * 0.265%. That measurement is the SORTED case and this entry does NOT sort: the columns
 * are tiled in the caller's order, so a caller whose scales are interleaved gets the
 * whole spread in every tile. Handing the columns over already grouped by scale, and
 * un-permuting the result, is the caller's lever until the entry takes a permutation.
 *
 * WHERE `scale_n` COMES FROM, AND WHAT IT COSTS. This entry cannot compute it: the scale
 * a column wants is set by that column's accumulator, which is what the call produces.
 * The 1.004x / 1.002x above, and every other perplexity figure on this route, used an
 * ORACLE scale taken from the accumulator being quantized, so they are the route's best
 * case and not a caller's. What a caller can actually supply has been measured on the
 * same windows: a scale FROZEN from a calibration pass over disjoint text costs
 * Qwen2.5-1.5B **1.047x fp32 at 87.3% top-1** against the oracle's 1.0025x at 96.6% —
 * an excess eighteen times the oracle arm's, and the largest single term in the route.
 * The pure analytic bound `128*sum_k|B[n][k]|` — the same term that caps C below — is
 * NOT a usable substitute: it overshoots a real column's accumulator by ~60x, spending
 * six bits of the int8 output on headroom nothing reaches. **So a caller must calibrate,
 * and must budget for what calibration costs it.** [host arithmetic, 2026-08-11]
 *
 * Measured: 0 wrong of 698368 elements against a host model of this ramp, over six
 * shapes at two column-scale spreads, including a forced eight-tile run.
 * [HW gate, H96 MAX M9, tests/rk3576_mm_requant, 2026-08-10] */
int rocket_matmul_int8_rk3576_perc(int fd, int M, int K, int N,
                                   const int8_t *A, const int8_t *B,
                                   const int32_t *bias, const float *scale_n,
                                   int8_t *C, double *worst_rel_err);

/* The same entry with the per-column sum of |B| supplied by the caller.
 *
 * `sum_abs_w[n]` must be exactly `sum_k |B[n][k]|` over the SAME B and the SAME K this
 * call is given. NULL means compute it here, which is what the entry above does.
 *
 * WHY IT IS A PARAMETER. That sum caps the C ramp (see above), so the entry needs it on
 * every call; it is a pass over all N*K weight bytes, it is `M`-independent, and the
 * weight does not change between calls. A caller that holds a weight across calls has
 * usually computed the same quantity already — a frontend deriving the no-saturate scale
 * `127/(128*sum_k|B[n][k]|+1)` has it exactly — so handing it over removes the pass
 * rather than duplicating it.
 *
 * WHAT A WRONG VALUE DOES. It is not checked and it cannot be: verifying it is the pass
 * this entry exists to skip. A value that does not match B changes the per-column
 * multiplier the ramp plans, so the call returns a full, correctly sized, entirely
 * plausible surface computed at a gain the caller did not ask for — too small and the
 * output loses resolution, too large and `(acc + bias)*C` overflows int32 and the column
 * saturates. Recompute it whenever B changes, and do not derive it back from a float
 * scale: the round trip is not exact and the plan is a function of the integer.
 *
 * Measured: bit-identical to the computing entry — supplying the sum returns the same
 * surface, and `llama-perplexity --chunks 4` over Qwen2.5-1.5B returns the same
 * PPL 12.8615 to every printed digit either way.
 * [HW gate, H96 MAX M9, tests/rk3576_mm_requant, 2026-08-13] */
int rocket_matmul_int8_rk3576_perc_sa(int fd, int M, int K, int N,
                                      const int8_t *A, const int8_t *B,
                                      const int32_t *bias, const float *scale_n,
                                      const int64_t *sum_abs_w,
                                      int8_t *C, double *worst_rel_err);

/* ---- the same matmul with the weight resident in a device BO -----------------
 *
 * rocket_rk3576_wbo holds one weight, packed once into the NPU's int8 weight-cube
 * layout in a device BO that lives until freed. The entry below is
 * rocket_matmul_int8_rk3576_perc_sa() with that object in place of `B`: the same
 * arithmetic, the same tiling, the same refusals — minus the per-call cube memset,
 * blocked copy and cache maintenance, and minus the weight BO's per-tile allocate and
 * free. Those are per-call passes over all N*K weight bytes for a weight that does not
 * change between calls, which is the whole reason this object exists.
 *
 * WHAT IT HOLDS. N*ceil(K/32)*32 bytes of DEVICE memory per weight (N and K are
 * multiples of 32, so N*K bytes), allocated on `fd` and freed only by
 * rocket_rk3576_wbo_free() or the fd closing. A caller that creates one per model
 * weight is choosing to hold the model's packed int8 bytes on the device; the create
 * cost is one pass over B (the same pass one call used to pay).
 *
 * WHAT IS REQUIRED, AND WHY THE SUMS ARE NOT OPTIONAL HERE. `sum_abs_w` must be
 * non-NULL: the C-ramp planner needs each column's sum of |B|, the row-major B is not
 * an argument to this entry, and re-deriving the sums from the packed cube would be the
 * O(N*K) pass this route removes. NULL refuses. The array carries the _sa contract
 * unchanged: it is not validated, and a value that does not match the packed weight
 * plans a wrong per-column gain and returns a full, plausible surface.
 *
 * WHAT A STALE OBJECT DOES. The object is not validated against anything on each call
 * (checking it would be the pass being skipped): an object created from different bytes
 * than the caller thinks, or outliving a weight it was created from, computes a full,
 * correctly sized, entirely plausible surface from the OLD weight. K and N are the one
 * cheap consistency check and a mismatch refuses.
 *
 * A K this part has no single-task plan for (K >= 6176) refuses here outright — the
 * int32 K-split fallback needs the row-major B this entry does not take, and that route
 * wedges the device at prefill scale anyway (see above).
 *
 * Measured: bit-identical to rocket_matmul_int8_rk3576_perc_sa() over the requant
 * gate's shapes, including a forced eight-tile run — the cached cube is byte-for-byte
 * the concatenation of the per-tile cubes the per-call path packs.
 * [HW gate, H96 MAX M9, tests/rk3576_mm_requant, 2026-08-19] */
struct rocket_rk3576_wbo;
int rocket_rk3576_wbo_create(int fd, int K, int N, const int8_t *B,
                             struct rocket_rk3576_wbo **out);
void rocket_rk3576_wbo_free(int fd, struct rocket_rk3576_wbo *w);
int rocket_matmul_int8_rk3576_perc_wbo(int fd, int M, int K, int N,
                                       const int8_t *A,
                                       const struct rocket_rk3576_wbo *wbo,
                                       const int32_t *bias, const float *scale_n,
                                       const int64_t *sum_abs_w,
                                       int8_t *C, double *worst_rel_err);

/* ROCKET_RK3576_BO_POOL=1 keeps this path's transient BOs (feature, coefficient,
 * output, regcmd) on a small per-process free list instead of allocating and freeing
 * them per tile per call; ROCKET_RK3576_BO_POOL_MB caps the pooled bytes (default 64).
 * Default OFF: holding device memory across calls is the exposure the pool shares with
 * the weight object above — whether the per-call free is what keeps IOMMU mapping churn
 * inside what the BO-lifetime fixes were written for is unmeasured. The knob is read on
 * every allocation, so an A/B can flip it between arms in one process. The pool does
 * not change any surface: a pooled BO is fully rewritten before use by the same packs
 * that filled a fresh one, and the poisoning sentinel stamps the output BO per task
 * either way. rocket_rk3576_bo_pool_drain(fd) frees whatever the pool holds for `fd`
 * (call it before closing an fd whose allocations may have been pooled). */
void rocket_rk3576_bo_pool_drain(int fd);

/* ---- the RK3576's int32-output matmul, and the K split ------------------
 *
 *     C[m][n] = sum_k A[m][k]*B[n][k] + bias[n]          (raw int32, no requant)
 *
 * K is split internally and the partials are summed on the host, so K is bounded only
 * by memory. `bias` may be NULL. Requires K%32 and N%32.
 *
 * WHAT IT COSTS. The DPU's 32-bit writer keeps the INT8 surface's byte budget whatever
 * the output element width is, so it delivers only the first eight output channels of
 * every thirty-two. This entry gets around that by programming four times the output
 * channels and scattering the real ones into the delivered slots — correct, and a
 * quarter of the int8 path's MACs per submit. Use it for the K a single task cannot
 * contract, not as the default matmul. [HW sweep, H96 MAX M9]
 *
 * WHAT IT CAN DO TO THE BOARD. At M=512 K=8192 N=2048 — 256 submits, a 4.5 MiB weight
 * cube, the narrow writer — this entry's write guard spent its eight power cycles, the
 * call returned -4, and the part raised the driver's DMA-error WARN_ON. The NPU then
 * computed nothing in ANY process: shapes that had run clean minutes earlier came back
 * -4, and a module reload left core 0 failing to probe at -22. Only a reboot cleared
 * it. Small cells in the class do compute (M=32 K=8192 N=128 and M=16 K=16384 N=64 run
 * green in the gate list; M=64 K=8192 N=32 scored exact three times), and where the
 * boundary between them lies is NOT known — one shape reached the wedge and each
 * occurrence costs a reboot. rocket_matmul_int8_rk3576() therefore no longer falls onto
 * this entry at all. [HW, H96 MAX M9, one shape, 2026-08-07] */
int rocket_matmul_int8_rk3576_i32(int fd, int M, int K, int N,
                                  const int8_t *A, const int8_t *B,
                                  const int32_t *bias, int32_t *C);

/* What the wide writer's detector saw during the last int32 matmul on this thread.
 *
 * The entry repairs both of its failure modes silently, so a caller that wants to know
 * what it paid — or a probe asking which shapes provoke the wide writer's zero-emission
 * runs — has to read it out. Counted per ROW TASK, which is the unit both defects and
 * both repairs work in.
 *
 * `empty` and `zeroed` are different hazards with different repairs: an atom still
 * holding the sentinel was never emitted, which is the wide-output poisoning and needs a
 * power cycle, while a stretch of the emission stream coming back zero is the writer's
 * own defect and is repaired by an immediate redo.
 *
 * Not thread-safe and not per-fd: this is the last call on this thread, and it is
 * overwritten by the next one. Zeroed at the top of every call, so it reads as all-zero
 * after a call that ran the narrow writer or refused. */
typedef struct rocket_rk3576_i32_stats {
    unsigned submits;       /* submits the call issued, retries included */
    unsigned tasks;         /* row tasks it ran */
    unsigned redo_empty;    /* task attempts redone for atoms never emitted */
    unsigned redo_zeroed;   /* task attempts redone for a zero-emission run */
    unsigned atoms_empty;   /* atoms still holding the sentinel, summed over attempts */
    unsigned atoms_zeroed;  /* atoms inside a detected zero run, summed the same way */
    unsigned accepted_zero; /* tasks whose zero atoms reproduced and were taken as data */
    unsigned refused;       /* tasks that exhausted their attempts */
} rocket_rk3576_i32_stats;
void rocket_rk3576_i32_last_stats(rocket_rk3576_i32_stats *out);

int rocket_matmul_plan_int8(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_int8(int fd, int M, int K, int N,
                       const int8_t *A, const int8_t *B, int32_t *C);

/* GROUP-WISE int8: C_f[M,N] = sum_g a_scale[m,g]*b_scale[n,g] * (int32 partial of
 * K-group g), fp32-accumulated. A,B are pre-quantized int8; a_scale is [M*nG],
 * b_scale is [N*nG] (nG = K/group), the per-group quant scales; Cf[M,N] is
 * overwritten. Every K-tile is kept inside ONE quant group, so that group's scale
 * applies to the tile's partial as it is read back -- no extra NPU work over
 * rocket_matmul_int8, just a scaled multiply-add on readback instead of an integer
 * one. This is what a natively-quantized weight (a GGUF MXFP4/Q8_0/Q4_K block, one
 * scale per K-block) needs, since the NPU cannot apply a K-blocked scale on-chip.
 *
 * group must be %32 and divide K; alignment as rocket_matmul_int8 minus the M==1
 * pad (K%32, N%32, M%4 — pad single vectors to M=4 caller-side). Unlike the int4
 * twin there is no saturation bound: the NPU output is int32, not int16.
 * rocket_matmul_plan_int8_gw previews the tiling (pure, no HW) and reports the Kt
 * it actually chose, which may be a proper DIVISOR of `group` when the group is
 * too wide for the CBUF (a K-tile need only lie inside one group, not be one).
 * Like rocket_matmul_plan_int8 it returns the NPU tile count, or <0 (rocket_status)
 * on an unsupported shape; rocket_matmul_int8_groupwise returns 0 on success. */
int rocket_matmul_plan_int8_gw(int M, int K, int N, int group,
                               int *Mt, int *Kt, int *Nt);
int rocket_matmul_int8_groupwise(int fd, int M, int K, int N,
                                 const int8_t *A, const int8_t *B,
                                 const float *a_scale, const float *b_scale,
                                 float *Cf, int group);

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

/* The segmented spelling of rocket_weights_pack, for a resident FUSED weight: several
 * static weights concatenated along N and packed as ONE resident weight of combined N.
 * Each Bs[i] is [Ns[i], K] row-major and occupies output columns [sum(Ns[0..i-1]),
 * +Ns[i]); the scatter resolves each global column to its source segment, so the caller
 * does NOT materialize a [sum(Ns), K] concat buffer -- which at a transformer's gate|up
 * group is hundreds of MB of transient host memory per group. This is the resident
 * counterpart of rocket_matmul_fp16_stream_fused's segmented scatter.
 *
 * N is sum(Ns[0..nseg-1]) and the returned handle is an ordinary rocket_weights: compute
 * with rocket_matmul_fp16_prepacked at that combined N and split the [M, N] output's
 * columns back into each member's own destination. nseg in [1, ROCKET_MAX_FUSE].
 * Same M-independence, threading and teardown rules as rocket_weights_pack. */
rocket_weights *rocket_weights_pack_seg(rocket_ctx *ctx, int M, int K, int N,
                                        const _Float16 *const *Bs, const int *Ns, int nseg);

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
 * fanned across worker fds (on mainline `rocket` each fd has its own 4GB IOVA window;
 * int8's ~11GB whole-model footprint fits across 5 fds with room to spare). That sizing
 * argument is mainline-only: the vendor `rknpu` driver shares ONE domain across the whole
 * process and fragments it over a board's uptime, so fanning across fds buys no address
 * space there and a large contiguous request can be refused while smaller ones fit.
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
 * N-slice is rounded to a multiple of 32. The _prepacked call's M MAY DIFFER
 * from the _pack M: the weight is planned at the canonical tile M (MAX_TILE),
 * so the resident layout depends only on K, N, and the K/N tiling and is
 * M-independent — one pack serves every micro-batch size (a warmup-M pack is
 * reused directly at prefill-M). K and N must still match; a genuine tiling
 * mismatch returns -2 (re-pack for that M), never a wrong answer. Result is
 * BIT-EXACT to the one-shot rocket_matmul_int8.
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

/* GROUP-WISE resident int8 — the resident sibling of rocket_matmul_int8_groupwise, and
 * the path a NATIVELY quantized weight (a GGUF MXFP4 / Q8_0 / Q4_K block, one scale per
 * K-block) needs: the int8 codes live on the NPU permanently, so the per-forward-pass
 * host dequant-to-fp16 and weight scatter both disappear. Pack once with
 * rocket_i8_weights_pack_gw (group%32, K%group); then each call quantizes only A and
 * returns the fp32 dequantized Cf[M,N] = sum_g a_scale[m,g]*b_scale[n,g] * (int32
 * partial of K-group g). a_scale is [M*nG], b_scale is [N*nG] (nG = K/group), the
 * per-row / per-channel per-group scales.
 *
 * Deltas from the int4 twin, both from int8's int32 (not int16) output accumulator:
 * there is NO saturation bound on `group`, and the K-tile is not forced to equal the
 * group — it is the largest DIVISOR of the group the CBUF can hold, so a group wider
 * than the CBUF cap is legal (it just costs more K-tiles per group). The raw-int32
 * path stays rocket_matmul_int8_prepacked.
 *
 * The weight stays M-independent (canonical tiling), so one pack serves every
 * micro-batch size; a genuine tiling mismatch returns -2 (re-pack), never a wrong
 * answer. NOT thread-safe (as rocket_ctx): one ctx per concurrent host thread. */
rocket_i8_weights *rocket_i8_weights_pack_gw(rocket_i8_ctx *ctx, int M, int K, int N,
                                             const int8_t *B, int group);
int rocket_matmul_int8_prepacked_gw(rocket_i8_ctx *ctx, int M, int K, int N,
                                    const int8_t *A, const float *a_scale,
                                    const float *b_scale, float *Cf, rocket_i8_weights *w);

/* ---- resident int4 (W4A4) path — int4 sibling of the resident int8 path.
 * Same usage; A/B pre-quantized int4 one-per-int8_t in [-7,7], C raw int32. N
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
 * support is needed. C raw-int32 path stays rocket_matmul_int4_prepacked.
 *
 * The 49*group bound is 49 = 7*7: it assumes the [-7,7] value range every int4 quantizer
 * produces, so a single K-tile partial stays inside the int16 group readback. If a future
 * producer ever emits -8, this bound must tighten to 64*group (and kt_cap drop to match) —
 * so the [-7,7] range and this bound are one invariant; do not widen one without the other. */
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
