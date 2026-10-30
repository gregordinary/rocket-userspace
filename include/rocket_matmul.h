// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_MATMUL_H
#define ROCKET_MATMUL_H

#include <stdint.h>   /* int8_t and int32_t in the int8 matmul API */
#include <stddef.h>   /* size_t (resident-weight byte accounting) */

/* The fp16 entries take `_Float16` operands. `_Float16` is a compiler extension (GCC and
 * Clang on aarch64, the RK3588 target). A caller whose compiler lacks it can define
 * `ROCKET_FLOAT16_T` as a 16-bit storage type before including this header. */
#ifdef ROCKET_FLOAT16_T
typedef ROCKET_FLOAT16_T _Float16;
#endif

/*
 * Declares the matmul entries, `C[M,N] = A[M,K] * B[N,K]^T` on the NPU. The RK3588 has
 * entries in six datatypes, with their multicore and resident forms, and the RK3576 has
 * int8 and fp16 entries. Each output channel n is the dot product of input row m with
 * weight row n (the `gen_matmul` convention). Every operand is row-major: A is M*K, B is
 * N*K and C is M*N.
 *
 * The fp16 entries take `_Float16` operands and drive the validated single-task generator
 * (`npu_regcmd.c`) through the submit seam. They tile the problem so that each tile fits
 * the 12x32KB CBUF. M (rows) and N (output channels) split into independent output
 * blocks, and K (the contraction) splits when it is too large for one tile. On one `fd`
 * the tiles run one after another, one task per tile. `rocket_matmul_fp16_mt` below fans
 * the columns out over the 3 cores.
 *
 * `rocket_matmul_fp16` sums the K partials on the NPU, with the DPU's fp16 elementwise add,
 * while `ROCKET_KACC` is at its default (on). The host sums them in fp32 instead with
 * `ROCKET_KACC=0`. The fp32 host sum keeps a large `K` precise.
 *
 * The fp16 entries require `K%32==0`, `N%16==0` and `M%4==0`. The `M` rule is the
 * planner's, which tiles a matmul's rows in multiples of 4: the generators compute any `M`.
 * The one-shot `rocket_matmul_fp16` accepts `M` = 1 and pads it internally to a
 * height-4 tile. The multicore, streaming and prepacked entries do not pad, and refuse
 * `M` = 1. A single-vector caller of those entries must pad `M` to 4.
 */

/* Computes the fp16 tiling plan for `M`, `K` and `N` without touching the hardware. It
 * writes the tile sizes (`Mt` <= `M`, `Kt` <= `K`, `Nt` <= `N`) and returns the number of
 * tiles, one task each, as a positive count. A shape that breaks the fp16 requirements
 * (`K%32==0`, `N%16==0`, `M%4==0`) returns `ROCKET_E_SHAPE`, a negative value of
 * `enum rocket_status`.
 *
 * Note: test the result with `< 0` for failure, not with `!= 0`. The matmul `*_plan()`
 * previews return a positive tile count on success, and the single-op previews
 * (`rocket_conv2d_plan`, `rocket_pool_fp16_plan` and the others) return `ROCKET_OK` (0).
 * `< 0` is the one failure test that is correct for all of them. */

#ifdef __cplusplus
extern "C" {
#endif
int rocket_matmul_plan(int M, int K, int N, int *Mt, int *Kt, int *Nt);

/* Computes the tiled fp16 matmul on the NPU. `fd` is an open `rocket` device
 * (`rocket_open()`). `M` must satisfy `M%4==0`, except that `M` = 1, the documented
 * single-vector case, is padded to 4 internally. Returns 0 on success and a negative value
 * on error.
 *
 * K-accumulation on the NPU (an fp16 DPU elementwise add) is the default operating mode.
 * It raises Gemma-4-12B prefill throughput by 19% (RK3588 at ~816 MHz) and changes the result
 * by ~0.4% against a host fp64 sum.
 * `ROCKET_KACC=0` (or `ROCKET_NO_KACC`) selects the byte-exact reference path, which sums
 * the K partials on the host in fp32. CBUF operand reuse turns on with K-accumulation and
 * picks `DATA_REUSE` or `WEIGHT_REUSE` per shape, and `ROCKET_REUSE` forces a mode. */
int rocket_matmul_fp16(int fd, int M, int K, int N,
                       const _Float16 *A, const _Float16 *B, _Float16 *C);

/* Computes `nbatch` independent fp16 matmuls `C[i] = A[i]·B[i]^T` that share one
 * (M,K,N), as a single stream of NPU tasks. The output tiles of every item share one
 * submit and one fence wait, instead of `nbatch` submit and wait round trips. With
 * `ROCKET_BATCH_SUBMIT=1` and the kernel support it needs, the items' regcmds also chain,
 * so the group raises one completion IRQ. Use it for small GEMMs of one shape whose cost
 * is dispatch. Flash attention's per-head QK matmuls within one worker's range of heads
 * are one such set, and its AV matmuls are another.
 *
 * `A`, `B` and `C` are arrays of `nbatch` row-major pointers: `A[i]` is [M,K], `B[i]` is
 * [N,K] and `C[i]` is [M,N]. The entry requires `M%4`, `K%32` and `N%16`, and `nbatch`==1
 * goes to `rocket_matmul_fp16`. The caller must bound `nbatch`, because the input and
 * weight BOs and the fp32 accumulators grow with it.
 *
 * The result equals in value `rocket_matmul_fp16` per item with `ROCKET_KACC=0`: the same
 * plan, the same fp16 pack and a host fp32 K sum. At one K tile the entry de-tiles the NPU's
 * fp16 output straight into `C`, which keeps a -0 output -0. Past one K tile it sums the K
 * tiles on the host in fp32 and narrows, and a -0 output comes back +0. The entry does not
 * read `ROCKET_KACC`, `ROCKET_REUSE` or `ROCKET_MM_PIPE`.
 *
 * On the RK3588 the equality holds for jobs on the same core. Each core occasionally writes
 * output lane n%16 == 0 below the host model, on its own set of inputs. Items on different
 * cores can therefore differ there.
 *
 * Returns 0 on success and a negative value on error. */
int rocket_matmul_fp16_batch(int fd, int M, int K, int N, int nbatch,
                             const _Float16 *const *A, const _Float16 *const *B,
                             _Float16 *const *C);

/* Runs `rocket_matmul_fp16_batch` with its BOs kept resident. It suits a caller that runs
 * a batch of one shape repeatedly, as flash attention does once per layer per forward
 * pass. `rocket_matmul_fp16_batch` allocates its input, weight and output BOs on every
 * call. It also zeroes the whole input and weight BOs, because the pad lanes must read
 * zero. A context allocates the BOs once and grows them only for a larger shape. It skips
 * the zeroing whenever the (M,K,N,nbatch) layout repeats.
 *
 * For a fixed plan the scatter rewrites the same live lanes and never touches the padding,
 * so padding zeroed once stays zero. Keeping the BOs resident removes the per-call `pack`
 * cost, the largest overhead of a chained flash attention.
 *
 *     b = rocket_mm_batch_create(fd);
 *     ... per call (same or a new shape):
 *         rocket_mm_batch_run(b, M, K, N, nbatch, A, B, C);
 *     rocket_mm_batch_free(b);
 *
 * A context is bound to the one `fd` it is created with, and tracks one last layout. A
 * caller alternating two shapes, such as flash attention's QK and AV, should use one
 * context per shape. Neither context then re-zeroes on a layout switch. A context is
 * not thread-safe: it changes its resident BOs and scratch on every run, so each
 * concurrent worker must have its own context.
 *
 * `rocket_mm_batch_run` is bit-identical to `rocket_matmul_fp16_batch`, with the same plan,
 * pack and de-tile. It de-tiles straight into `C` at one K tile, and narrows a host fp32 K
 * sum past it. Only the BO and scratch lifetime differ. The shape requirements are the same
 * (`M%4`, `K%32`, `N%16`), and `nbatch`==1 goes to `rocket_matmul_fp16`. It returns 0 on
 * success and a negative value on error. The resident BOs are kept for later calls, and
 * `rocket_mm_batch_free` frees them. */
typedef struct rocket_mm_batch rocket_mm_batch;
rocket_mm_batch *rocket_mm_batch_create(int fd);
void             rocket_mm_batch_free(rocket_mm_batch *b);
int rocket_mm_batch_run(rocket_mm_batch *b, int M, int K, int N, int nbatch,
                        const _Float16 *const *A, const _Float16 *const *B,
                        _Float16 *const *C);

/* Computes the fp16 matmul with an fp32 output, fp16 x fp16 -> fp32:
 * `C[M,N] = A[M,K] * B[N,K]^T`, with the same fp16 input packing as `rocket_matmul_fp16`.
 * The DPU writes the full fp32 accumulator (`fp32tofp16=0`) instead of narrowing each K
 * partial to fp16, and the host sums the K partials in fp64. The entry therefore skips both
 * the fp16 rounding that `rocket_matmul_fp16` applies to each K partial and the final
 * narrowing. The result is the fp32 dot product of the fp16 inputs.
 *
 * `C` is `float`. `K` must be a multiple of 32 and `N` a multiple of 16. `M` must satisfy
 * `M%4==0`, except that `M` = 1 is padded to 4 internally. Returns 0 on success.
 *
 * Use it after activation scaling, or for accuracy at a large `K`. It reads back 2x the
 * output bytes (an fp32 cube against an fp16 one), so it is a separate opt-in entry rather
 * than the default. */
int rocket_matmul_fp16_f32out(int fd, int M, int K, int N,
                              const _Float16 *A, const _Float16 *B, float *C);

/* Computes the RK3576's own int8 matmul, `C[M,N] = A[M,K] * B[N,K]^T`, with an int8 output
 * that the DPU writes through its requant:
 *
 *     C[m][n] = sat8( round( ( sum_k A[m][k]*B[n][k] + bias[n] ) * scale ) )
 *
 * The contract differs from `rocket_matmul_int8`, which returns the RK3588's raw int32, so
 * this is a separate entry rather than a dispatch behind that one. `bias` can be NULL.
 * `scale` is per-tensor and must be positive. `K` and `N` must satisfy `K%32` and `N%32`.
 *
 * Caution: the DPU's `OUT_CVT` multiplier applies to the whole bias stage. A `scale` of
 * zero writes a complete output of the right size that is entirely empty, and nothing
 * faults.
 *
 * Along each axis the entry behaves as follows, measured on the RK3576:
 *
 *   - `M` has no constraint. `M`=1 is supported, as is every plane the `M` rows can be cut
 *     into. On the RK3588, by contrast, rows are the convolution's spatial height, a
 *     height < 4 produces wrong values, and `M`==1 is padded to 4.
 *   - `N` is tiled, and the tile sets the throughput. At the feature budget a submit's
 *     device time rises from about 0.12 ms at N 32 to 0.91 ms at N 2048. The device's
 *     rate therefore rises almost linearly with `N`, the only free axis [HW sweep, rocket
 *     1.6.0, 786 MHz, 2026-09-25]. `ROCKET_RK3576_MM_NT` overrides the default tile.
 *   - `K` is not tiled, and the entry refuses a `K` it cannot contract in one task. The
 *     boundary is `K` >= 6176 for every (M, N) enumerated. The planner halves the
 *     output-channel tile down to 32 channels before it gives up. The CBUF pool against a
 *     one-group weight cube therefore decides it, a condition on `K` alone. Past it the
 *     entry returns `ROCKET_E_UNSUPPORTED` and submits nothing.
 *   - Below that boundary `K` is free, but its cost is not flat. At `K` > 4992 the output
 *     tile drops from 1024 channels to 32 in one step. At M=512 N=2048 the drop multiplies
 *     the submit count by 32. Nothing refuses, and the values stay correct. A caller
 *     choosing a `K` for speed should stay at or below 4608.
 *   - A task can stall at a fixed output channel of its tile, and the plan keeps every
 *     task clear of it (`rocket_rk3576_weight_phase_groups()`). The planner either
 *     shortens the row window until its allowance holds the whole tile, or caps the tile at
 *     the groups the window holds. It takes whichever needs fewer tasks.
 *
 * The K-split path is the int32 entry below. A caller that needs a larger `K` calls
 * `rocket_matmul_int8_rk3576_i32()` and requantizes, or runs it on the host.
 * `ROCKET_RK3576_MM_KSPLIT=1` makes this entry take that path itself.
 *
 * The refusal is policy. On that entry's narrow writer, one shape of the class (M=512
 * K=8192 N=2048) wedged the NPU across processes until a reboot. The wedge is a single
 * observation, on the H96 MAX M9 with `rocket` 1.5.0 on 2026-08-05. That entry's dense
 * writer, the default, runs the same shape in 345.7 ms, 48 submits with no redo. The 4096
 * sampled elements are exact [HW sweep, H96 MAX M9, rocket 1.6.0, 2026-09-30].
 *
 * The stall plan is exact at M=512 N=1536 [HW sweep, rocket 1.6.0, 2026-09-27]. At K
 * 2144-2528 it runs in 23-30 ms on the first call with no retirement (K 2304 23.8-27.0).
 * At K 4192, 4480 and 4576 it runs at 672-, 160- and 128-channel tiles in 59-90 ms. K 2240
 * is exact at M 192, 1024 and 2048.
 *
 * If a stall the plan does not predict occurs, the entry re-plans the tile at half its
 * width, down to 32 channels. It keeps that width per `K`. `ROCKET_RK3576_MM_NARROW=0`
 * turns the re-plan off, and the entry then refuses there with `ROCKET_E_DEVICE`.
 * `ROCKET_RK3576_MM_PHASE=0` turns the stall plan off and leaves the re-plan to find the
 * width.
 *
 * Caution: a stalled task writes nothing past the stall, raises no completion and is
 * retired at the driver's backstop. A write check alone passes its partial output.
 *
 * One per-tensor output scale costs an LLM accuracy. `scale` applies to the output alone:
 * `A` and `B` arrive quantized, so every input-side choice is the caller's. The cost was
 * simulated exactly on two models' own prefill activations at M=512, with every
 * offloadable projection replaced. The figures are wikitext-2 perplexity against an fp16
 * baseline of 1.000x:
 *
 *     per-tensor A and B          SmolLM2-1.7B 2.26x      Qwen2.5-1.5B 1.11x
 *     per-row A, per-channel B    SmolLM2-1.7B 86x        Qwen2.5-1.5B 1.03x
 *
 * The cost varies by more than an order of magnitude between models, so a caller must
 * measure a candidate model end to end. A per-GEMM norm does not predict the cost, and
 * prefers the configuration that is 86x worse. Per-axis input scales are free at this
 * interface and better on every per-GEMM number, yet catastrophic on one model of two.
 * The cause is not characterized. SmolLM2's loss is in calibration rather than ranking:
 * top-1 is 60.7% against the per-tensor configuration's 62.0%, and perplexity rises from
 * 25 to 964.
 *
 * An orthonormal Hadamard rotation along K before quantization, plus a per-output-channel
 * output requant, recovers it on both models, and neither half alone does. The rotation
 * is host-side and needs no library change (1.31x and 1.02x on its own). The requant is
 * `rocket_matmul_int8_rk3576_perc()` (1.54x and 1.06x on its own). Together they measure
 * 1.004x and 1.002x at 97% top-1. Those figures use an exact per-column scale
 * [host-computed, two models, 2026-08-10]. They are the best this path gives, not what the
 * integer ramp delivers.
 *
 * `rocket_matmul_plan_int8_rk3576` previews the tiling without touching the hardware. It
 * returns the task count, or a negative value for a shape the RK3576 cannot run. It
 * refuses the same `K` as the entry [HW sweep, H96 MAX M9]. Both refuse with
 * `ROCKET_E_SHAPE` a dimension past 2^24 (the RK3588 planners' bound) and an `M` whose
 * plane is taller than the 16-bit height field it is programmed into. Any prime `M` past
 * 65535 is among those. Both refuse with `ROCKET_E_TILING` a plan of more than `INT_MAX`
 * tasks, which the return value cannot hold. */
int rocket_matmul_plan_int8_rk3576(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_int8_rk3576(int fd, int M, int K, int N,
                              const int8_t *A, const int8_t *B,
                              const int32_t *bias, float scale, int8_t *C);

/* Computes the same matmul as `rocket_matmul_int8_rk3576()` with the output scale given
 * per output column instead of per tensor:
 *
 *     C[m][n] = sat8( round_half_to_even( (sum_k A[m][k]*B[n][k] + bias[n])
 *                                         * scale_n[n] ) )
 *
 * Every shape rule, refusal and hazard of `rocket_matmul_int8_rk3576()` applies unchanged.
 * `scale_n` has `N` entries, and every one must be positive.
 *
 * The entry exists because one per-tensor output scale keeps an LLM off the per-tensor
 * entry. Simulated on two models' own prefill activations, that scale costs 2.26x
 * perplexity on SmolLM2-1.7B and 1.11x on Qwen2.5-1.5B. A per-column output requant
 * recovers those to 1.54x and 1.06x on its own. Combined with a host-side Hadamard
 * rotation along K, it recovers them to 1.004x and 1.002x [host-computed, two models,
 * 2026-08-10].
 *
 * The DPU has one (MUL, SHIFT) per task. The per-column part is therefore programmed in
 * the coefficient group's int16 C term and in the task's shift word `s`. The epilogue is
 * `sat32(rne(((acc + bias[n]) * C[n]) >> s))`, then the shared `(v*MUL)>>SHIFT`. The
 * product is held wide and rounded half to even at `s` [HW sweep, H96 MAX M9,
 * `tests/rk3576_coeff_c.c` shift]. It is exact to 2^45.95 on both signs, next to the int32
 * x int16 maximum of 2^46. A bias-heavy test case reaches that value in `(acc + bias)*C`.
 *
 * Column n's gain is C[n] * MUL/2^(SHIFT+s), so its resolution is 0.5/C[n]. The planner
 * puts the tile's largest scale near C = 32767. It takes the smallest `s` at which every
 * column's `(acc + bias)*C` fits int32 at its bound, 128*sum_k|B[n][k]| + |bias[n]|. C
 * therefore spans the int16 field in every tile, whatever the contraction depth.
 *
 * A column resolves its scale to 0.5/C = 0.5*spread/32767 relative, where `spread` is the
 * ratio of the tile's largest scale to that column's. A column below 1/65534 of its tile's
 * largest scale needs a C under one. It runs at the gain of C = 1 instead of its own.
 *
 * `worst_rel_err` (can be NULL) reports the largest relative gain error the ramp delivered
 * over all columns, a clamped one included. It is the only number that separates this
 * entry from an exact per-column scale, so read it rather than assume it.
 *
 * `ROCKET_RK3576_MM_PC_SHIFT=0`, read per call, selects the SHIFT-0 ramp, with the shift
 * word at zero. `(acc + bias)*C` then saturates at int32, and C is capped at `INT32_MAX`
 * over the column's bound. At a GEMM's depth that cap sits in the low hundreds, below the
 * int16 field on every column of the projections below. The resolution falls with it. The
 * shift-0 ramp is the configuration to compare against.
 *
 * The ramp is planned per N tile, since the (MUL, SHIFT) and the shift word are per task.
 * The entry does not sort, and tiles the columns in the caller's order. A caller whose
 * scales are interleaved therefore gets the whole spread in every tile. On the shift-0 ramp
 * that spread costs most of the resolution. Forcing eight tiles took an 8x-spread K=1024
 * test case from 1.53% to 0.265%, the sorted case. On the shift ramp, a model's within-tile
 * spread of 5-35x at the projections below costs 0.006-0.05% at the worst column.
 *
 * The two ramps cost a GEMM the following at Qwen2.5-1.5B's and SmolLM2-1.7B's own
 * projections, three layers each. The shapes are M 512, K 1536 at N 256, 1536 and 8960,
 * and K 2048 at N 2048 and 8192. The operands are the models' rotated int8 operands, and
 * the column scale is 127/(colmax*3) from the window's exact column maxima:
 *
 *                                   shift ramp            shift-0 ramp
 *     column gain error, median     0.0013-0.0035%        0.10-0.33%
 *     column gain error, worst      0.006-0.047%          0.40-3.8%
 *     bytes off the exact requant   0.012-0.086%          1.1-7.9%
 *     composed error over the       +0.0000-0.0001        +0.005-0.50
 *       exact requant's (points)
 *
 * The composed error is RMS(C/scale - acc)/RMS(acc), which the exact per-column requant
 * holds at 1.4-2.8% on these operands (the int8 rounding). The device output matches the
 * host model of each ramp on every element of the 18 GEMMs. [HW sweep, H96 MAX M9, rocket
 * 1.6.0, `tests/rk3576_mm_percol_err` with `tools/rk3576-w8a8-dump.py`, 2026-09-27]
 *
 * End to end, the difference between the ramps is below what perplexity resolves. Through
 * ggml-rocket's W8A8 path, `llama-perplexity --chunks 4` over Qwen2.5-1.5B measures 8.7848
 * on the shift ramp and 8.7433 on the shift-0 ramp. Each is identical to every printed
 * digit on repeat. A different N tile alone (`ROCKET_RK3576_MM_NT=1024`) moves the shift-0
 * ramp to 8.8576 and the shift ramp to 8.7625. The frontend's bootstrap reads each column
 * back at a couple of codes, and any perturbation resamples it. The ramp's cost on this
 * path was simulated at 0.06% of the ratio, 1/20 of that spread [HW sweep, H96 MAX M9,
 * 2026-09-27].
 *
 * Where the spread is wide, the simulation does resolve it, over 32 windows of 512 tokens
 * at oracle scales (`tools/rk3576-w8a8-ppl.py`). SmolLM2-1.7B's perplexity ratio to fp32 is
 * 1.0048 at an exact per-column scale, 1.0165 on the shift-0 ramp and 1.0055 on the shift
 * ramp. Qwen2.5-1.5B's three are 1.0028, 1.0030 and 1.0032, one noise band [host-computed,
 * 2026-09-27].
 *
 * The caller must supply `scale_n`, because this entry cannot compute it. The scale a
 * column needs is set by that column's accumulator, which is what the call produces. The
 * 1.004x and 1.002x above, and every other perplexity figure on this path, used an oracle
 * scale taken from the accumulator being quantized. They are the path's best case rather
 * than a caller's.
 *
 * What a caller can supply was measured on the same windows. A scale frozen from a
 * calibration pass over disjoint text costs Qwen2.5-1.5B 1.047x fp32 at 87.3% top-1,
 * against the oracle's 1.0025x at 96.6%. That excess is eighteen times the oracle scale's,
 * and the largest single term on the path. The pure analytic bound `128*sum_k|B[n][k]|`,
 * the term that caps C above, is not a usable substitute. It overshoots a real column's
 * accumulator by ~60x, which uses six bits of the int8 output on headroom nothing
 * reaches.
 *
 * A caller must therefore calibrate, and must budget for what calibration costs it
 * [host-computed, 2026-08-11].
 *
 * Evidence: 0 wrong of 1380352 elements against the host models of the two ramps, 690176
 * each [HW sweep, H96 MAX M9, rocket 1.6.0, `tests/rk3576_mm_requant`, 2026-09-27]. The
 * test covers four shapes (K 256-2048) at three column-scale spreads (1x, 8x and 1000x,
 * where the shift-0 ramp clamps columns and the shift ramp does not), with forced runs of
 * up to eight tiles. Every shift-ramp case puts products past 2^31 on both signs, and
 * every case's gains match the float scale to half a unit of C. */
int rocket_matmul_int8_rk3576_perc(int fd, int M, int K, int N,
                                   const int8_t *A, const int8_t *B,
                                   const int32_t *bias, const float *scale_n,
                                   int8_t *C, double *worst_rel_err);

/* Computes the same matmul as `rocket_matmul_int8_rk3576_perc()`, with the per-column sum
 * of |B| supplied by the caller.
 *
 * `sum_abs_w[n]` must be exactly `sum_k |B[n][k]|` over the same `B` and the same `K` this
 * call is given. NULL means the entry computes it, as `rocket_matmul_int8_rk3576_perc()`
 * does.
 *
 * The sum is a parameter because it caps the C ramp, so the entry needs it on every call.
 * Computing it is a pass over all N*K weight bytes, it is `M`-independent, and the weight
 * does not change between calls. A caller that holds a weight across calls has usually
 * computed the same quantity already. A frontend deriving the no-saturate scale
 * `127/(128*sum_k|B[n][k]|+1)` has it exactly, so handing it over removes the pass rather
 * than repeating it.
 *
 * Caution: the entry does not check `sum_abs_w`, and cannot, because checking it is the
 * pass this entry exists to skip. A value that does not match `B` changes the plan, and the
 * call returns a complete output of the right size with plausible wrong values. A sum below
 * the true one lets `(acc + bias)*C` (after the shift word, on the shift ramp) leave int32,
 * and the column saturates. A sum above it costs the shift-0 ramp resolution, and on the
 * shift ramp it can only raise the shift word.
 *
 * The caller must recompute the sum whenever `B` changes, and must not derive it back from
 * a float scale. That round trip is not exact, and the plan is a function of the integer.
 *
 * Evidence: bit-identical to the computing entry [HW sweep, H96 MAX M9,
 * `tests/rk3576_mm_requant`, 2026-08-13]. Supplying the sum returns the same output, and
 * `llama-perplexity --chunks 4` over Qwen2.5-1.5B returns the same PPL 12.8615 to every
 * printed digit either way. */
int rocket_matmul_int8_rk3576_perc_sa(int fd, int M, int K, int N,
                                      const int8_t *A, const int8_t *B,
                                      const int32_t *bias, const float *scale_n,
                                      const int64_t *sum_abs_w,
                                      int8_t *C, double *worst_rel_err);

/* Holds one int8 weight in a `rocket_rk3576_wbo`: a device BO, packed once into the NPU's
 * int8 weight-cube layout, that lives until freed. `rocket_matmul_int8_rk3576_perc_wbo()`
 * is `rocket_matmul_int8_rk3576_perc_sa()` with that object in place of `B`: the same
 * arithmetic, tiling and refusals. It skips the per-call cube memset, blocked copy and
 * cache maintenance, and the weight BO's allocation and free per tile. Those are per-call
 * passes over all N*K weight bytes for a weight that does not change between calls.
 *
 * An object holds N*ceil(K/32)*32 bytes of device memory per weight (N*K bytes, since `N`
 * and `K` are multiples of 32). It is allocated on `fd` and freed only by
 * `rocket_rk3576_wbo_free()` or by closing the fd. A caller that creates one per model
 * weight holds the model's packed int8 bytes on the device. Creating one costs one pass
 * over `B`, the same pass each call of the per-call entry makes.
 *
 * `sum_abs_w` must be non-NULL, and NULL refuses. The C-ramp planner needs each column's
 * sum of |B|, and the row-major `B` is not an argument to this entry. Deriving the sums
 * from the packed cube would be the O(N*K) pass this entry removes. The array has the
 * `_sa` contract unchanged, and is not validated. A value that does not match the packed
 * weight plans a wrong per-column gain, and the call returns a complete, plausible output.
 *
 * Caution: the object is not validated on each call, because checking it would be the pass
 * being skipped. An object can be created from different bytes than the caller assumes,
 * or outlive the weight it was created from. The call then returns a complete output of
 * the right size, computed from the old weight. `K` and `N` are the one cheap consistency
 * check, and a mismatch refuses.
 *
 * A `K` with no single-task plan on the RK3576 (`K` >= 6176) refuses here outright. The
 * int32 K-split fallback needs the row-major `B`, which this entry does not take.
 *
 * Evidence: bit-identical to `rocket_matmul_int8_rk3576_perc_sa()` over the requant test's
 * shapes, including a forced eight-tile run [HW sweep, H96 MAX M9,
 * `tests/rk3576_mm_requant`, 2026-08-19]. The cached cube is byte-for-byte the
 * concatenation of the per-tile cubes the per-call path packs. */
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

/* Frees whatever the RK3576 transient-BO pool holds for `fd`. Call
 * `rocket_rk3576_bo_pool_drain(fd)` before closing an fd whose allocations the pool can
 * hold.
 *
 * `ROCKET_RK3576_BO_POOL=1` keeps this path's transient BOs (feature, coefficient, output,
 * regcmd) on a small per-process free list, instead of allocating and freeing them per
 * tile per call. `ROCKET_RK3576_BO_POOL_MB` caps the pooled bytes (default 64). Both
 * environment variables are read once per process, so a comparison runs one process per
 * configuration.
 *
 * The pool is off by default. Holding device memory across calls is the exposure the pool
 * shares with the weight object above. Whether the per-call free is what keeps IOMMU
 * mapping churn inside what the BO-lifetime fixes were written for is unmeasured.
 *
 * A pooled BO keeps the size it was allocated at, and a request takes the smallest free BO
 * that covers it. After the mix of entries or shapes changes, a small request can hold a
 * far larger buffer. `PREP_BO` and `FINI_BO` sync the whole BO, so the call costs cache
 * maintenance at that size. A pooled cost therefore depends on what the process ran before.
 * Five entries sharing a pool put the fp16 resident call 0.09-0.16 ms above its device
 * time at 9 of 18 shapes. One entry alone, whose three buffers are all small, stayed
 * 0.014-0.043 ms above at all 18 [HW sweep, H96 MAX M9, `tests/rk3576_mm_fp16_cost`,
 * 2026-09-25].
 *
 * The pool does not change any output. The same packs that fill a fresh BO fully rewrite
 * a pooled one before use. The sentinel that detects an unwritten output stamps the output
 * BO per task either way. */
void rocket_rk3576_bo_pool_drain(int fd);

/* Computes the RK3576's int32-output matmul, with the K split:
 *
 *     C[m][n] = sum_k A[m][k]*B[n][k] + bias[n]          (raw int32, no requant)
 *
 * The entry splits K internally and sums the partials on the host. `K` is therefore bounded
 * by memory and by the 2^24 every matmul planner takes for each dimension. An `M` with no
 * plane short enough for the 16-bit height field at any K slice is refused, as on the int8
 * entry. `bias` can be NULL. `K` and `N` must satisfy `K%32` and `N%32`.
 *
 * It runs on the dense writer: the int8 program on the whole wide output stage
 * (`gen_conv2d_int8_rk3576_i32out_dense`). The dense writer writes every accumulator of a
 * task as a raw int32, and does not corrupt the next submit. The requested output channels
 * are programmed (rounded to 32), the tasks run back to back with no idle, and the call
 * leaves the NPU clean. The matmul test's 21 int32 shapes (M 1-4096, K 64-16384, N 32-4096)
 * are exact, in 0.26 s together where the older writers took 6.3 s. Per call that is 1.0 ms
 * against 106.9 at 8x64x32, and 48.1 against 1163.9 at 32x1024x4096 [HW sweep, H96 MAX M9,
 * rocket 1.6.0, `tests/rk3576_matmul_gate`, 2026-09-30].
 *
 * The write guard stays. It stamps a sentinel, scans each atom, redoes a retired task after
 * a power cycle, and refuses rather than return a partial output.
 * `ROCKET_RK3576_I32_DENSE=0`, or `ROCKET_RK3576_I32_OC_MULT=2` or `=4`, selects the older
 * writers. They deliver a quarter or half of the programmed channels, corrupt the next
 * submit and idle for it.
 *
 * Caution: on the older writers, M=512 K=8192 N=2048 (256 submits, a 4.5 MiB weight cube,
 * the narrow writer) used up the write guard's eight power cycles. The call returned -4,
 * and the NPU raised the driver's DMA-error `WARN_ON`. The NPU then produced no output in
 * any process. Shapes that had run clean minutes earlier came back -4, and a module reload
 * left core 0 failing to probe at -22. Only a reboot cleared it. The wedge is a single
 * observation, at one shape, on the H96 MAX M9 with `rocket` 1.5.0 on 2026-08-05.
 *
 * Small shapes in that class do complete (M=32 K=8192 N=128 and M=16 K=16384 N=64 pass in
 * the test list, and M=64 K=8192 N=32 was exact three times). Where the boundary between
 * them lies is not known, because one shape reached the wedge and each occurrence costs a
 * reboot. On the dense writer that shape runs in 345.7 ms, 48 submits with no redo, and
 * 4096 sampled elements are exact. [HW sweep, H96 MAX M9, rocket 1.6.0, 2026-09-30]
 *
 * As policy, `rocket_matmul_int8_rk3576()` falls back to this entry only under
 * `ROCKET_RK3576_MM_KSPLIT=1`. */
int rocket_matmul_int8_rk3576_i32(int fd, int M, int K, int N,
                                  const int8_t *A, const int8_t *B,
                                  const int32_t *bias, int32_t *C);

/* Reports what the wide writer's detector saw during the last int32 matmul on this thread.
 *
 * The entry repairs both of its failure modes silently. A caller that wants to know what
 * the repairs cost, or a probe asking which shapes provoke the wide writer's zero-emission
 * runs, reads them here. The counts are per row task, the unit both defects and both
 * repairs work in.
 *
 * `empty` and `zeroed` are different hazards with different repairs. An atom still holding
 * the sentinel was never emitted: that is the wide-output corruption of the next submit,
 * and it needs a power cycle. A stretch of the emission stream that comes back zero is the
 * writer's own defect, and an immediate redo repairs it.
 *
 * The statistics are not thread-safe and not per-fd: they describe the last call on this
 * thread, and the next call overwrites them. Every call zeroes them first, so they are all
 * zero after a call that ran the narrow writer or refused. */
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

/* Computes the RK3576's fp16 matmul, with the whole of K in one task:
 *
 *     C[m][n] = sum_k A[m][k] * B[n][k]      A: M*K fp16, B: N*K fp16, C: M*N fp32
 *
 * All three are row-major, and `B` has one row per output channel, as
 * `rocket_matmul_fp16()` takes it. The activation goes in as it arrives, and the weights
 * are permuted into the RK3576's layout. The output is the tasks' own fp32 output, so
 * nothing is scattered or gathered on the host.
 *
 * It is a separate program from the fp16 convolution program, which contracts sixteen
 * input channels a task. On its fp16 writer, which costs a power cycle per program, that
 * program measured 30-299x slower than int8 over ordinary shapes. Its fp32-stage default,
 * which costs none, has not been compared with int8. This entry is the matmul form of
 * gahingwoo/charsiu (`524e10a`), transcribed word for word.
 *
 * The limits are `1 <= M <= 2048`, `32 <= K <= 6144` with `K%32 == 0`, and
 * `64 <= N <= 8192` with `N%16 == 0`. Anything else returns `ROCKET_E_UNSUPPORTED` before
 * a buffer is allocated. `K` and `N` are one task's bounds, measured on the RK3576. `K`
 * 6176 never completes, and at `N` 8960 the task writes the first 768 columns and hangs.
 * The `M` bound is the test's.
 *
 * One task holds at most 4096/(K/32) rows, since its input must fit the CBUF data window.
 * A task past that bound produces wrong values on every row rather than failing. The entry
 * therefore cuts `M` into equal tasks under the bound (512 rows at K 256, 85 at K 1536, 21
 * at K 6144). It submits them together: one ioctl and one fence per call.
 *
 * The output is exact against a CPU reference whose fills make every sum exact in fp32.
 * The test covers 23 one-task shapes to the bounds (the window at three K, K 6144, N 8192,
 * an 8 MiB weight) and 6 shapes of 2 to 25 tasks, up to 2048x1536x256. A row-major weight
 * control was wrong on 64 of 64 [HW sweep, H96 MAX M9, rocket 1.6.0,
 * `tests/rk3576_mm_fp16_gate`, 2026-09-25].
 *
 * The NPU sums each group of 32 consecutive products along K with no intermediate
 * rounding, and rounds the group sum to fp32. It accumulates the group sums in K order in
 * fp32, and both roundings are to nearest even.
 *
 * That model matches every output bit for bit at M 4, N 64 and K 32 to 2048. The fills have
 * full mantissas, with products spread over 2^-28 to 2^-6. Blocks of 16 or 64, or rounding
 * each product, miss by up to hundreds of ulps at K 2048. Two runs are identical.
 * [HW sweep, H96 MAX M9, rocket 1.6.0, `tests/rk3576_mm_fp16_round`, 2026-09-25]
 *
 * The device's share at M 512 K 1536 N 1536 is 4.5 ms, about 2.1x the int8 resident
 * entry's at the same shape. Each task re-reads the whole weight, and this program
 * contracts at about half the int8 program's rate. Per call, the entry allocates, fills
 * and frees five buffers, and `ROCKET_MM_PROFILE` prints its time per phase. [HW sweep,
 * H96 MAX M9, 786 MHz, governor performance, `tests/rk3576_mm_fp16_cost`, 2026-09-25]
 *
 * Its fp32 output does not corrupt the next submit, unlike the older int32 writers and the
 * fp16 convolution's fp16 writer. In one process at a zero gap, the int8 direct program
 * wrote 20 of 20 after it, and so did this entry after itself. [HW sweep, H96 MAX M9,
 * `rk3576_poison_probe`, 2026-09-25]
 *
 * If the output still holds the sentinel stamp after the fence, the entry redoes the call
 * after a power cycle. Past eight attempts it refuses with `ROCKET_E_DEVICE`. `C` is
 * unspecified after that refusal. `rocket_rk3576_mm_fp16_last_attempts()` returns how many
 * submits the last call took, so a result that needed a redo is visible to the caller. A
 * value of 1 means the first submit wrote. */
int rocket_matmul_fp16_rk3576(int fd, int M, int K, int N,
                              const _Float16 *A, const _Float16 *B, float *C);
unsigned rocket_rk3576_mm_fp16_last_attempts(void);

/* Holds one fp16 weight in a `rocket_rk3576_wbo_fp16`, permuted once into the RK3576's
 * layout in a device BO. The coefficient buffer for its `N` sits beside it.
 * `rocket_matmul_fp16_rk3576_wbo()` is `rocket_matmul_fp16_rk3576()` with that object in
 * place of `B`: the same program, limits, write check and redo. It drops the two buffers
 * the per-call entry allocates, fills and frees every call. Neither changes between
 * calls: the weight permutation is a pass over all N*K elements, and the coefficients
 * depend on `N` alone.
 *
 * An object holds two device BOs allocated on `fd`, freed only by
 * `rocket_rk3576_wbo_fp16_free()` or by closing the fd. They are the weight, K*N*2 bytes,
 * and the coefficient buffer, `rocket_rk3576_mm_fp16_coef_bytes(N)`, about 257 KiB
 * whatever `N` is. Nearly all of the coefficient buffer is padding against RDMA reads past
 * its table. Whether the task needs the padding is unmeasured, so a handle per model
 * weight also holds about 257 KiB of padding per weight.
 *
 * Caution: the object is not checked against anything on each call. An object can be
 * created from different bytes than the caller assumes, or outlive the weight it was
 * created from. The call then returns a complete output of the right size, computed from
 * the old weight. `K` and `N` are the one cheap consistency check, and a mismatch refuses
 * with `ROCKET_E_SHAPE`. `M` is free per call, inside the limits.
 *
 * A `K` or `N` outside the limits refuses at create with `ROCKET_E_UNSUPPORTED`, before a
 * buffer is allocated.
 *
 * The entry is bit-identical to `rocket_matmul_fp16_rk3576()` at all 29 tested shapes. One
 * handle is exact at eleven `M` from 1 to 1500, one to three tasks. [HW sweep, H96 MAX M9,
 * rocket 1.6.0, `tests/rk3576_mm_fp16_gate`, 2026-09-25]
 *
 * Its cost against `rocket_matmul_int8_rk3576_perc_wbo()` at the same shape, both resident,
 * paired within a round:
 *
 *     M 1-16, K to 2048, N to 256    0.86-0.94x at 14 of 17 shapes, 1.07-1.50x at K 2048
 *     K 1536 N 1536, M 16 to 512     1.5-1.6x at M 16, 2.0x at 64, 2.0-2.2x from 128 to 512
 *     M 512 K 1536, N 256 to 8192    1.8-1.9x, 2.0-2.1x at 1536 and 4096, 2.6x at 8192
 *     M 512 N 1536, K 256 to 6144    2.7x at K 256, 1.9-2.0x at 4096, 1.0x at 6144
 *
 * The gap does not shrink with `M`. Past one task the fp16 program re-reads the weight once
 * a task and contracts at about half the int8 program's rate. At M 512 K 1536 N 1536 its
 * device share is 4.5 ms against the int8 entry's 2.1 ms. The gap closes at K 6144 because
 * the int8 entry slows there. At 512x1536x1536 a call is 17.5-17.8 ms, 4.5 of it on the
 * device and 7.2 allocating and freeing the call's buffers. With `ROCKET_RK3576_BO_POOL=1`
 * it is 6.9 ms against the int8 entry's pooled 4.8 [HW sweep, H96 MAX M9, 786 MHz,
 * governor performance, A72s, `tests/rk3576_mm_fp16_cost`, 2026-09-25]. */
struct rocket_rk3576_wbo_fp16;
int rocket_rk3576_wbo_fp16_create(int fd, int K, int N, const _Float16 *B,
                                  struct rocket_rk3576_wbo_fp16 **out);
void rocket_rk3576_wbo_fp16_free(int fd, struct rocket_rk3576_wbo_fp16 *w);
int rocket_matmul_fp16_rk3576_wbo(int fd, int M, int K, int N, const _Float16 *A,
                                  const struct rocket_rk3576_wbo_fp16 *wbo, float *C);

/* Computes the RK3588's tiled int8 matmul, int8 x int8 -> int32, with the
 * `C[M,N] = A[M,K] * B[N,K]^T` convention. `A` and `B` are row-major int8 that the caller
 * has already quantized, applying the per-row activation and per-channel weight scales.
 * `C` is the raw int32 accumulation, and the caller dequantizes it as
 * `C[m,n] * a_scale[m] * b_scale[n]`.
 *
 * The NPU computes int8xint8->int32. K splits into int32 partials that the host sums,
 * as the fp16 path does with `ROCKET_KACC=0`, and there is no on-chip requant. The
 * alignment is stricter than fp16's: `K%32` and `N%32` (the int8 weight k-group is 32, not
 * 16), and `M%4==0`.
 *
 * As with fp16, the plans take `M` in multiples of 4. Every one-shot
 * entry of each datatype pads `M`==1 to 4 internally and returns row 0. The plans and the
 * resident-weight entries require `M%4==0` and refuse `M`==1, so a caller of those must
 * pad a single vector to 4.
 *
 * `rocket_matmul_plan_int8` previews the tiling without touching the hardware. It returns
 * the tile count, or a negative value for an unsupported shape. `rocket_matmul_int8` runs
 * the matmul on `fd`. */
int rocket_matmul_plan_int8(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_int8(int fd, int M, int K, int N,
                       const int8_t *A, const int8_t *B, int32_t *C);

/* Computes the group-wise int8 matmul, accumulated in fp32:
 *
 *     C_f[M,N] = sum_g a_scale[m,g]*b_scale[n,g] * (int32 partial of K-group g)
 *
 * `A` and `B` are int8 that the caller has already quantized. `a_scale` is [M*nG] and
 * `b_scale` is [N*nG] (nG = K/group), the per-group quant scales, and `Cf[M,N]` is
 * overwritten.
 *
 * Every K tile stays inside one quant group, so that group's scale applies to the tile's
 * partial as it is read back. The NPU does no more work than for `rocket_matmul_int8`, and
 * the readback does a scaled multiply-add instead of an integer one. A natively quantized
 * weight (a GGUF MXFP4, Q8_0 or Q4_K block, one scale per K block) needs this, since the
 * NPU cannot apply a K-blocked scale on-chip.
 *
 * `group` must be a multiple of 32 and divide `K`. The alignment is that of
 * `rocket_matmul_int8` without the `M`==1 pad (`K%32`, `N%32`, `M%4`), so a caller must pad
 * a single vector to M=4. Unlike the int4 twin there is no saturation bound, because the
 * NPU output is int32, not int16.
 *
 * `rocket_matmul_plan_int8_gw` previews the tiling without touching the hardware, and
 * reports the `Kt` it chose. The chosen `Kt` can be a proper divisor of `group` when the
 * group is too wide for the CBUF. A K tile need only lie inside one group, not be one. Like
 * `rocket_matmul_plan_int8` it returns the NPU tile count, or a negative `rocket_status`
 * for an unsupported shape. `rocket_matmul_int8_groupwise` returns 0 on success. */
int rocket_matmul_plan_int8_gw(int M, int K, int N, int group,
                               int *Mt, int *Kt, int *Nt);
int rocket_matmul_int8_groupwise(int fd, int M, int K, int N,
                                 const int8_t *A, const int8_t *B,
                                 const float *a_scale, const float *b_scale,
                                 float *Cf, int group);

/* Computes the tiled int4 matmul: int4 x int4 -> int16 on the NPU, accumulated on the host
 * to an int32 `C`. `A` and `B` are int4 values that the caller has already quantized,
 * stored one per `int8_t` in [-8,7]. The caller is responsible for the scales and any
 * Hadamard rotation. `C` is the raw int32 accumulation.
 *
 * int4 differs from int8 in its feature cube (C2=32), its weight layout (N/64,K/32,64,32)
 * [weight_int4] and its int16 output cube (C2=8). The input and weight are nibble-packed,
 * 2 int4 values a byte. The alignment is `K%32`, `N%64` (int4's N-group is 64, not int8's
 * 32) and (M%4||1). `rocket_matmul_plan_int4` previews the tiling without touching the
 * hardware.
 *
 * Caution: each K tile's partial is read back as int16, so a K pass whose |sum| exceeds
 * 32767 saturates, and the result is lossy. The plan keeps `Kt` within the CBUF limit.
 * Whether that also stays unsaturated depends on the data's magnitude. It is fine for
 * Hadamard-rotated or scaled weights. For raw full-range int4, keep `Kt` small. */
int rocket_matmul_plan_int4(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_int4(int fd, int M, int K, int N,
                       const int8_t *A, const int8_t *B, int32_t *C);

/* Computes the int4 matmul with an explicit cap on `Kt` against int16 saturation. A
 * `kt_cap` > 0 caps the K tile (rounded to a multiple of 32) so that a [-8,7] partial
 * (|.|<=64*Kt) cannot overflow the int16 output. `kt_cap`=0 is `rocket_matmul_int4` (no
 * cap). In-model callers pass `kt_cap`=480. */
int rocket_matmul_int4_ex(int fd, int M, int K, int N,
                          const int8_t *A, const int8_t *B, int32_t *C, int kt_cap);

/* Computes the group-wise int4 matmul, accumulated in fp32:
 *
 *     C_f[M,N] = sum_g a_scale[m,g]*b_scale[n,g] * (int4 partial of K-group g)
 *
 * `A` and `B` are int4 in [-7,7] that the caller has already quantized. `a_scale` is
 * [M*nG] and `b_scale` is [N*nG] (nG = K/group), the per-group quant scales.
 *
 * The K tile is forced to `group` (one tile is one quant group), so a separate dequant
 * scale applies per K slice. This brings W4 quality toward int8 fidelity, the GPTQ and
 * AWQ regime. `group` must divide `K`, be a multiple of 32 and keep 49*group < 32767 (int16
 * saturation), and `M` must satisfy `M%4`. Returns 0, or a negative value for an
 * unsupported shape. */
int rocket_matmul_int4_groupwise(int fd, int M, int K, int N,
                                 const int8_t *A, const int8_t *B,
                                 const float *a_scale, const float *b_scale,
                                 float *Cf, int group);

/* Previews the int16 tile shape without touching the hardware. int16 has fp16's input
 * geometry (feature cube C2=8, weight (N/16,K/32,16,32) [weight_int16 == weight_fp16],
 * 2-byte elements) with a 4-byte output cube (C2=4), and the bf16 path reuses this plan.
 * The alignment is `K%32`, `N%16` and (M%4||1).
 *
 * The single-task generator writes int16 x int16 -> int32 natively and saturates to int32,
 * so no entry here exposes it. Use `rocket_matmul_int16_exact` (below) for an int64-exact
 * int16 matmul over full-range operands. */
int rocket_matmul_plan_int16(int M, int K, int N, int *Mt, int *Kt, int *Nt);

/* Computes a bit-exact int16 x int16 -> int64 matmul, the supported full-precision path.
 * The native int16 output saturates to int32 (bit-exact below that bound,
 * `tests/int16_native_probe`), so it cannot represent a full-range int64 product. This
 * entry decomposes each int16 into two signed bytes, runs four proven int8 matmuls and
 * recombines them in int64:
 *
 *     C = 65536*(Ah.Bh) + 256*(Ah.Bl + Al.Bh) + Al.Bl
 *
 * The result is bit-exact with no saturation, at ~4x the int8 cost, for completeness rather
 * than speed. Inputs are clamped to [-32768, 32639], because the signed/signed split cannot
 * reach the top 128 int16 codes, and the result is bit-exact within that domain. `C` is
 * int64 [M*N] row-major. The alignment is `K%32`, `N%32` and (M%4||1). Returns 0 on success
 * and a negative value on error. */
int rocket_matmul_int16_exact(int fd, int M, int K, int N,
                              const int16_t *A, const int16_t *B, int64_t *C);

/* Computes the tiled bf16 matmul, bf16 x bf16 -> fp32. bf16 has fp32's 8-bit exponent at
 * fp16's 2-byte cost, so it covers the full activation range with no per-row scaling
 * (which the fp16 path needs an amax scan for). The CNA/DPU MAC computes bf16xbf16 at
 * precision 3 and accumulates to fp32: exact products and an fp32 accumulation (max rel
 * ~1e-6).
 *
 * bf16 shares int16's NPU geometry (2-byte input cube C2=8, weight (N/16,K/32,16,32),
 * 4-byte output cube C2=4) but is a float path. The output is fp32, the host sums K in
 * double, and nothing saturates. The interface takes fp32 `A` and `B` and truncates them to
 * bf16 during the tile scatter, with no separate pack, scale or unscale. `C` is fp32
 * [M*N] row-major. The alignment is `K%32`, `N%16` and (M%4||1), as for fp16 and int16.
 * `rocket_matmul_plan_bf16` previews the tiling without touching the hardware, and is the
 * int16 plan. */
int rocket_matmul_plan_bf16(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_bf16(int fd, int M, int K, int N,
                       const float *A, const float *B, float *C);

/* Computes the bf16 matmul on several cores. The single-fd `rocket_matmul_bf16` opens
 * nothing persistent, and allocates and frees every BO per call. An in-model bf16 prefill
 * on it runs on one core, with a BO allocation and free per matmul. The entries here move
 * both costs out of the loop (`rocket_bf16_stream.c`). They are the bf16 siblings of
 * `rocket_matmul_fp16_mt` and `rocket_matmul_fp16_stream`.
 *
 * `rocket_matmul_bf16_mt` splits the output columns N across `nthreads` worker fds, each
 * running the unchanged single-fd `rocket_matmul_bf16` on its slice. Each fd is one
 * scheduling entity, so the 3 NPU cores run in parallel. The entry opens and closes its
 * own fds, so the caller must not pass one in. `nthreads` is clamped to [1,8]. `A`, `B`
 * and `C` are fp32, and the output is fp32 [M,N] row-major. */
int rocket_matmul_bf16_mt(int M, int K, int N,
                          const float *A, const float *B, float *C, int nthreads);

/* Computes the bf16 matmul on a persistent streaming context. The context keeps the worker
 * fds and the per-(M,K,N) scratch BOs (input, weight, output, regcmd) resident, and
 * re-packs only `A` and `B` each call. bf16 weights are used once per prefill token batch,
 * so there is no resident-weight (prepacked) variant, and streaming suits that use.
 *
 * The output is bit-identical to `rocket_matmul_bf16` at `nthreads`=1, and the same valid
 * bf16 product within float reassociation at `nthreads`>1. Returns 0, or a negative value
 * to tell the caller to fall back to the per-call mt path (shape cache full or shape
 * unsupported). Free the stream before exit.
 *
 * A stream is not thread-safe. It changes its shared per-shape scratch (the A and B re-pack
 * buffers and the per-worker BOs) every call, and grows its shape cache on first use of a
 * new (M,K,N). Concurrent calls on one stream therefore race. Each concurrent host thread
 * must have its own stream. The fan-out of one call across the 3 NPU cores is internal and
 * separate, and one stream already drives all cores. */
typedef struct rocket_bf16_stream rocket_bf16_stream;
rocket_bf16_stream *rocket_bf16_stream_create(int nthreads);
void                rocket_bf16_stream_free(rocket_bf16_stream *s);
int rocket_matmul_bf16_stream(rocket_bf16_stream *s, int M, int K, int N,
                              const float *A, const float *B, float *C);

/* Computes the tiled tf32 matmul, tf32 x tf32 -> fp32, the only path with 4-byte inputs.
 * tf32 is fp32's range (8-bit exponent) with fp16's precision (10-bit mantissa) in a
 * 4-byte fp32 container. The caller feeds raw fp32. The MAC truncates each operand's
 * mantissa to 10 bits, dropping the low 13 without rounding, and accumulates in fp32. `C`
 * is fp32 [M*N] row-major.
 *
 * The result is max rel ~1.5e-7 against a reference whose inputs are truncated the same
 * way. A round-to-nearest reference differs from that one on about half of its inputs.
 *
 * Like bf16, it needs no activation scaling and is a float path: the output is fp32, the
 * host sums K in double, and nothing saturates. The interface takes fp32 `A` and `B`
 * straight through with no truncation on the scatter, and the hardware truncates to
 * tf32 itself. A caller that wants round-to-nearest must round on the host first. It differs from bf16 only in the 4-byte geometry: feature cube C2=4, weight
 * tile (N/16,K/16,16,16), banks 4 B/elem and data_entries K/16. The K-group halves to 16,
 * and the N-group stays 16. The alignment is `K%16`, `N%16` and (M%4||1).
 *
 * `gen_matmul_tf32` sets the per-stage precision (CNA/CORE=7 tf32, DPU=fp32). tf32 has the
 * lowest value of the datatypes: its MAC runs at half rate, and bf16 already gives fp32
 * range at full speed. It exists for completeness, not for a workload, and no in-model
 * backend uses it. `rocket_matmul_plan_tf32` previews the tiling without touching the
 * hardware. */
int rocket_matmul_plan_tf32(int M, int K, int N, int *Mt, int *Kt, int *Nt);
int rocket_matmul_tf32(int fd, int M, int K, int N,
                       const float *A, const float *B, float *C);

/* Computes the fp16 matmul on several cores. It splits N across `nthreads` worker
 * threads, each with its own `rocket` fd. Each thread computes a contiguous column slice
 * C[:,n0:n1] = A*B[n0:n1]^T with the unchanged single-fd `rocket_matmul_fp16`. The
 * `rocket` driver runs one drm_sched per NPU core and one scheduling entity per fd.
 * Independent fds therefore let the 3 cores run in parallel, and one fd serializes onto one
 * core.
 *
 * The entry opens and closes its own fds, so the caller must not pass one in. `nthreads`
 * is clamped to [1,8]. A count of ~3 saturates the 3 cores, and 4 can run slightly faster
 * by hiding gaps on the CPU side. Returns 0 on success and a negative value on error. */
int rocket_matmul_fp16_mt(int M, int K, int N,
                          const _Float16 *A, const _Float16 *B, _Float16 *C,
                          int nthreads);

/* Packs a weight once and keeps it resident, for repeated matmuls that reuse the same
 * weights `B`. A model's static weight matrices across forward passes are the typical
 * case. The entry packs `B` into resident NPU BOs once and keeps the worker fds and
 * scratch BOs alive, so each matmul packs only `A`.
 *
 *     ctx = rocket_ctx_create(nthreads);              // persistent worker fds
 *     w   = rocket_weights_pack(ctx, M, K, N, B);     // scatter B once
 *     ... per forward pass:
 *         rocket_matmul_fp16_prepacked(ctx, M, K, N, A, C, w);
 *     ... teardown (weights BEFORE ctx, they hold BOs on the ctx fds):
 *     rocket_weights_free(ctx, w);
 *     rocket_ctx_free(ctx);
 *
 * The `M` of a `_prepacked` call can differ from the `M` of the `_pack`. The resident
 * weight layout depends only on `K`, `N` and the K and N tiling (`Kt`, `Nt`). That tiling
 * is M-independent for all M >= MAX_TILE (256), where `Mt` is capped. A weight packed at
 * the warmup `M` is therefore reused directly at the prefill `M` whenever the tiling
 * matches, with no re-pack. The tiling matches in the common LLM and Whisper case.
 *
 * `K` and `N` must still match. An incompatible tiling returns -2, so that the caller
 * re-packs for that `M`. One example is a pack at M>=256 called at a small `M` whose `Kt`
 * grows.
 *
 * A context is not thread-safe. It changes its shared per-shape scratch (the A-pack buffer
 * and the per-worker BOs that a compute borrows) every call, and grows its shape and weight
 * caches on first use. Concurrent `rocket_weights_pack` and `_prepacked` calls on one
 * context therefore race, and each concurrent host thread must have its own context. The
 * fan-out of one call across the 3 NPU cores is internal, and one context already drives
 * all cores. A `rocket_weights` handle is tied to the context it was packed on. */
typedef struct rocket_ctx     rocket_ctx;
typedef struct rocket_weights rocket_weights;

rocket_ctx     *rocket_ctx_create(int nthreads);
void            rocket_ctx_free(rocket_ctx *ctx);

/* Creates a context, as `rocket_ctx_create` does, with flags. `ROCKET_CTX_TILING_CANONICAL`
 * makes every resident weight on the context M-independent down to M = 4. At M < MAX_TILE
 * the plan keeps the K and N tiling the M = MAX_TILE plan chose, and sets `Mt` = `M`. It
 * does not grow `Kt` into the CBUF that the smaller input tile frees.
 *
 * The weight's layout depends only on `K`, `N` and that tiling, so one pack serves every
 * call's `M`. `_prepacked` then never returns -2 for a small `M`. The cost is the `Kt` a
 * small `M` gives up: at most the K-tile count the M = MAX_TILE plan already runs. The flag
 * suits a resident encoder whose sequence length varies per call. Flags 0 is
 * `rocket_ctx_create`. */
#define ROCKET_CTX_TILING_CANONICAL 0x1u
rocket_ctx     *rocket_ctx_create_ex(int nthreads, unsigned flags);

rocket_weights *rocket_weights_pack(rocket_ctx *ctx, int M, int K, int N, const _Float16 *B);
void            rocket_weights_free(rocket_ctx *ctx, rocket_weights *w);

/* Packs a resident fused weight: several static weights concatenated along N and packed as
 * one resident weight of combined `N`. It is the segmented form of `rocket_weights_pack`,
 * and the resident counterpart of the segmented scatter in
 * `rocket_matmul_fp16_stream_fused`. Each `Bs[i]` is [Ns[i], K] row-major and occupies
 * output columns [sum(Ns[0..i-1]), +Ns[i]).
 *
 * The scatter resolves each global column to its source segment, so the caller does not
 * build a [sum(Ns), K] concatenation buffer. At a transformer's gate|up group, that buffer
 * would be hundreds of MB of transient host memory per group.
 *
 * `N` is sum(Ns[0..nseg-1]), and the returned handle is an ordinary `rocket_weights`.
 * Compute with `rocket_matmul_fp16_prepacked` at that combined `N`, and split the [M, N]
 * output's columns back into each member's own destination. `nseg` must be in
 * [1, ROCKET_MAX_FUSE]. The M-independence, threading and teardown rules are those of
 * `rocket_weights_pack`. */
rocket_weights *rocket_weights_pack_seg(rocket_ctx *ctx, int M, int K, int N,
                                        const _Float16 *const *Bs, const int *Ns, int nseg);

int rocket_matmul_fp16_prepacked(rocket_ctx *ctx, int M, int K, int N,
                                 const _Float16 *A, _Float16 *C, rocket_weights *w);

/* Computes fp16 matmuls on a streaming context, for LLM prefill, where each weight is used
 * once per generation. Caching resident weights as the prepacked path does would waste
 * ~20GB of NPU BOs for a 12B model. The streaming context keeps the per-worker fds and the
 * scratch and weight BOs resident, but re-packs `B` every call.
 *
 * The context caches the scratch and weight BOs per distinct (M,K,N) shape. An LLM's
 * small, repeating set of shapes (q/k/v/o/gate/up/down/lm_head) then costs the BO
 * allocation and fd open once in total, not once per matmul. The whole model is not held
 * resident. The context also shares the A pack across workers, one scatter rather than
 * `nthreads`.
 *
 *     s = rocket_stream_create(nthreads);
 *     ... per matmul (any cached or new shape):
 *         rocket_matmul_fp16_stream(s, M, K, N, A, B, C);
 *     rocket_stream_free(s);
 *
 * Returns 0 on success, or a negative value to tell the caller to fall back to the
 * per-call mt path. The fallback applies, for example, when the shape cache is full or the
 * shape is unsupported. `M`==1 is such a shape, as on the resident paths. The one-shot
 * `rocket_matmul_fp16` pads a single row to a height-4 tile, and this entry does not
 * (`tests/correctness_matrix.sh` checks it).
 *
 * A stream is not thread-safe. It changes its shared per-shape scratch (the A-pack buffer
 * and the per-worker weight and IO BOs, re-packed each call) and grows its shape cache on
 * first use of a new (M,K,N). Concurrent calls on one stream therefore race, and each
 * concurrent host thread must have its own stream. The fan-out of one call across the 3
 * NPU cores is internal, and one stream already drives all cores. */
typedef struct rocket_stream rocket_stream;

rocket_stream *rocket_stream_create(int nthreads);
void           rocket_stream_free(rocket_stream *s);

int rocket_matmul_fp16_stream(rocket_stream *s, int M, int K, int N,
                              const _Float16 *A, const _Float16 *B, _Float16 *C);

/* Computes several static weights that share one activation `A` as one matmul. Examples
 * are a transformer's Q, K and V projections, or its gate and up projections. The weights
 * are concatenated along N:
 *
 *     C[M, sum(Ns[i])] = A[M,K] * [B0 ; B1 ; ...]^T
 *
 * Each `Bs[i]` is [Ns[i], K] row-major. The entry scatters each weight directly into its
 * global-N slice of the combined weight BO, with no host-side [sumN,K] concatenation copy.
 * It runs one N-split multicore matmul over the combined N, and writes `C` as
 * [M, sum(Ns)] row-major. The caller splits the columns back into each projection's own
 * output.
 *
 * The fusion turns `nseg` matmuls, each with its own submits and A pack, into one, and
 * grows N for better core utilization. The weight-scatter byte count stays the same: the
 * weights are distinct, and the hardware requires the scatter.
 *
 * The contract is that of `rocket_matmul_fp16_stream`: 0, or a negative value to tell the
 * caller to fall back to running the members individually. `nseg` must be in
 * [1, ROCKET_MAX_FUSE]. */
#define ROCKET_MAX_FUSE 8

int rocket_matmul_fp16_stream_fused(rocket_stream *s, int M, int K,
                                    const _Float16 *const *Bs, const int *Ns, int nseg,
                                    const _Float16 *A, _Float16 *C);

/* Packs a quantized int8 weight once and keeps it resident (W8A8), the int8 analogue of
 * the fp16 pack-weights-once path above. The int8 weight B[N,K] is scattered into resident
 * per-worker NPU BOs once and reused across forward passes, so each matmul packs only `A`.
 * The int8 weight scatter ("packB") is the per-call cost this removes, as for fp16. N fans
 * out across worker fds.
 *
 * On mainline `rocket` each fd has its own 4GB IOVA window, and int8's ~11GB whole-model
 * footprint fits across 5 fds with room to spare. That sizing holds on mainline only. The
 * vendor `rknpu` driver shares one domain across the whole process and fragments it over a
 * board's uptime. Fanning across fds then gives no more address space, and a large
 * contiguous request can be refused while smaller ones fit.
 *
 *     ctx = rocket_i8_ctx_create(nthreads);            // persistent worker fds
 *     w   = rocket_i8_weights_pack(ctx, M, K, N, qB);  // scatter int8 B once
 *     ... per forward pass (qA pre-quantized int8 [M,K]):
 *         rocket_matmul_int8_prepacked(ctx, M, K, N, qA, C32, w);   // C32 raw int32
 *     ... teardown (weights BEFORE ctx; they hold BOs on the ctx fds):
 *     rocket_i8_weights_free(ctx, w);
 *     rocket_i8_ctx_free(ctx);
 *
 * The int8 convention is that of `rocket_matmul_int8`. `A` and `B` are int8 that the caller
 * has already quantized. The caller is responsible for the per-row and per-channel scales
 * and any Hadamard rotation. `C` is the raw int32 accumulation. The host sums K in int64,
 * and no NPU integer K-accumulation is implemented.
 *
 * The alignment is `K%32`, `N%32` and `M%4==0`. The resident paths do not pad `M`==1, so a
 * caller must pad a single vector to 4. The per-worker N slice is rounded to a multiple of
 * 32.
 *
 * The `M` of a `_prepacked` call can differ from the `M` of the `_pack`. The weight is
 * planned at the canonical tile M (MAX_TILE). The resident layout therefore depends only
 * on `K`, `N` and the K and N tiling, and is M-independent. One pack serves every
 * micro-batch size, and a warmup-M pack is reused directly at the prefill `M`.
 *
 * `K` and `N` must still match. A tiling mismatch returns -2 (re-pack for that `M`), never
 * a wrong answer. The result is bit-exact to the one-shot `rocket_matmul_int8`.
 *
 * A context is not thread-safe, as for `rocket_ctx`, because it changes its shared
 * per-shape scratch each call. Each concurrent host thread must have its own context. */
typedef struct rocket_i8_ctx     rocket_i8_ctx;
typedef struct rocket_i8_weights rocket_i8_weights;

rocket_i8_ctx     *rocket_i8_ctx_create(int nthreads);
void               rocket_i8_ctx_free(rocket_i8_ctx *ctx);

rocket_i8_weights *rocket_i8_weights_pack(rocket_i8_ctx *ctx, int M, int K, int N,
                                          const int8_t *B);
void               rocket_i8_weights_free(rocket_i8_ctx *ctx, rocket_i8_weights *w);
/* Returns the resident NPU-BO footprint of a packed weight in bytes: the per-worker weight
 * tiles. The shared scratch is cached across weights and not counted. */
size_t             rocket_i8_weights_bytes(const rocket_i8_weights *w);

int rocket_matmul_int8_prepacked(rocket_i8_ctx *ctx, int M, int K, int N,
                                 const int8_t *A, int32_t *C, rocket_i8_weights *w);

/* Computes the group-wise resident int8 matmul, the resident sibling of
 * `rocket_matmul_int8_groupwise`. A natively quantized weight (a GGUF MXFP4, Q8_0 or Q4_K
 * block, one scale per K block) needs this path. The int8 codes stay on the NPU, so the
 * host dequantization to fp16 and the weight scatter of every forward pass both go away.
 *
 * Pack once with `rocket_i8_weights_pack_gw` (`group%32`, `K%group`). Each call then
 * quantizes only `A` and returns the fp32 dequantized result:
 *
 *     Cf[M,N] = sum_g a_scale[m,g]*b_scale[n,g] * (int32 partial of K-group g)
 *
 * `a_scale` is [M*nG] and `b_scale` is [N*nG] (nG = K/group), the per-row and per-channel
 * per-group scales.
 *
 * Two differences from the int4 twin both follow from int8's int32 (not int16) output
 * accumulator. There is no saturation bound on `group`. The K tile is not forced to equal
 * the group. It is the largest divisor of the group the CBUF can hold. A group wider than
 * the CBUF cap is therefore legal, and costs more K tiles per group. The raw-int32 path
 * stays `rocket_matmul_int8_prepacked`.
 *
 * The weight stays M-independent (canonical tiling), so one pack serves every micro-batch
 * size. A tiling mismatch returns -2 (re-pack), never a wrong answer. A context is not
 * thread-safe, as for `rocket_ctx`: each concurrent host thread must have its own. */
rocket_i8_weights *rocket_i8_weights_pack_gw(rocket_i8_ctx *ctx, int M, int K, int N,
                                             const int8_t *B, int group);
int rocket_matmul_int8_prepacked_gw(rocket_i8_ctx *ctx, int M, int K, int N,
                                    const int8_t *A, const float *a_scale,
                                    const float *b_scale, float *Cf, rocket_i8_weights *w);

/* Packs a quantized int4 weight once and keeps it resident (W4A4), the int4 sibling of
 * the resident int8 path, with the same usage. `A` and `B` are int4 that the caller has
 * already quantized, one per `int8_t` in [-7,7], and `C` is raw int32. N fans out across
 * worker fds, and the weight is scattered once into resident int4 NPU BOs. The path exists
 * to measure int4's multicore throughput against the resident fp16 and int8 paths. A
 * context is not thread-safe, as for `rocket_ctx`: each concurrent host thread must have
 * its own. */
typedef struct rocket_i4_ctx     rocket_i4_ctx;
typedef struct rocket_i4_weights rocket_i4_weights;

rocket_i4_ctx     *rocket_i4_ctx_create(int nthreads);
void               rocket_i4_ctx_free(rocket_i4_ctx *ctx);
rocket_i4_weights *rocket_i4_weights_pack(rocket_i4_ctx *ctx, int M, int K, int N, const int8_t *B);
void               rocket_i4_weights_free(rocket_i4_ctx *ctx, rocket_i4_weights *w);
int rocket_matmul_int4_prepacked(rocket_i4_ctx *ctx, int M, int K, int N,
                                 const int8_t *A, int32_t *C, rocket_i4_weights *w);

/* Computes the group-wise resident int4 matmul, the resident sibling of
 * `rocket_matmul_int4_groupwise` and the in-model W4A4 path. Pack the weight once with
 * `rocket_i4_weights_pack_gw` (K tile forced to `group`, `group%32`, `K%group`,
 * `49*group<32767`). Each call then quantizes only `A` and returns the fp32 dequantized
 * result `Cf[M,N] = sum_g a_scale[m,g]*b_scale[n,g]* (int4 partial of K-group g)`.
 * `a_scale` is [M*nG] and `b_scale` is [N*nG] (nG = K/group), the per-row and per-channel
 * per-group scales.
 *
 * A caller that uses a Hadamard rotation bakes it into the packed weight and applies it to
 * `A`. The rotation preserves the product, so the library needs no support for it. The
 * raw-int32 path stays `rocket_matmul_int4_prepacked`.
 *
 * The 49*group bound is 49 = 7*7. It assumes the [-7,7] value range every int4 quantizer
 * produces, so that a single K-tile partial stays inside the int16 group readback. If a
 * producer emits -8, this bound must tighten to 64*group, and `kt_cap` must drop to match.
 * The [-7,7] range and this bound are one invariant: do not widen one without the other. */
rocket_i4_weights *rocket_i4_weights_pack_gw(rocket_i4_ctx *ctx, int M, int K, int N,
                                             const int8_t *B, int group);
int rocket_matmul_int4_prepacked_gw(rocket_i4_ctx *ctx, int M, int K, int N,
                                    const int8_t *A, const float *a_scale,
                                    const float *b_scale, float *Cf, rocket_i4_weights *w);
/* Returns the resident NPU-BO footprint of a packed int4 weight: the per-worker tiles,
 * without the shared scratch. It is the int4 sibling of `rocket_i8_weights_bytes`. */
size_t rocket_i4_weights_bytes(const rocket_i4_weights *w);


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_MATMUL_H */
