# librocketnpu: API reference

The function reference, test catalog, and environment-variable list for `librocketnpu`. The
[README](README.md) is the guide, and this page is the reference. The README's
[Terms](README.md#terms) table defines submit, task, kick, regcmd and cube layout. Convention
throughout: `C[M,N] = A[M,K] · B[N,K]ᵀ`, row-major.

## Library layout

```
include/                       public API
  rocket_npu.h                 device shim: open / bo_alloc / bo_prep(fence) / submit
  rocket_hw_profile.h          the chip machine-parameter profile (CBUF / tile geometry / dtype mask) + capability query
  rocket_log.h                 the logging channel: level, callback, stderr tee
  rocket_chain.h               the contiguous self-chaining regcmd layout of a batched submit
  rocket_matmul.h              the matmul API (fp16 / int8 / int4 / int16 / bf16 / tf32; tiled / mt / resident / streaming), RK3588 and RK3576
  rocket_conv.h                2D conv API (fp16 CONV_2D + native depthwise + ConvTranspose2d + int8; per-call or resident-BO ctx), the RK3576 conv, cube-layout and chained-run entries
  rocket_resize.h              integer-factor nearest / bilinear upsample (a depthwise ConvTranspose)
  rocket_pool.h                on-NPU MaxPool / AveragePool via the PPU pooling engine (fp16 + int8/uint8), RK3588 and RK3576
  rocket_reduce.h              spatial GlobalAvgPool/Mean + GlobalMax/MinPool (ReduceMax/Min) over [H,W] (PPU) + feature-axis reduce + cumsum (prefix sum) over the hidden axis (ones-/triangular-matmul)
  rocket_norm.h                on-NPU RMSNorm + LayerNorm + the per-row broadcast scale primitive
  rocket_normvision.h          on-NPU vision normalization: BatchNorm / GroupNorm / InstanceNorm / L2-Normalize
  rocket_ffn.h                 on-NPU gated FFN block (GeGLU/SwiGLU core + the three projections)
  rocket_softmax.h             on-NPU row-wise softmax + LogSoftmax + stable cross-entropy (EXP LUT + row-sum reduce + per-row scale / host gather)
  rocket_attn.h                on-NPU multi-head self-attention (Whisper encoder) + masked GQA flash attention (decoder/LLM prefill)
  rocket_encoder.h             one Whisper/transformer encoder block (LN->MHA->residual->LN->FFN->residual)
  rocket_siglip.h              SigLIP-B/16 vision encoder end-to-end (patch-embed + pos + 12xblock + post-LN)
  rocket_sam.h                 SAM ViT-Det image encoder end-to-end (facebook/sam-vit-base)
  rocket_modernbert.h          ModernBERT text encoder end-to-end (RoPE, global/sliding attention, GeGLU) + a pre-norm post stack, variable-length batch
  rocket_activation.h          elementwise fp16 activation via the DPU LUT (sigmoid/hardsigmoid/tanh/silu/gelu/leaky/sqrt/rsqrt/recip/exp/log; auto-tiled for any n)
  npu_matmul.h                 low-level gemm_params_t + the regcmd generators
  npu_activation.h             low-level lut_act_params_t + gen_lut_activation_fp16
  npu_pool.h                   low-level PPU pooling regcmd generator
  npu_requant.h                npu_out_cvt_pair: a float requant scale as the DPU OUT_CVT multiplier and shift
  npu_regcmd_rk3576.h          low-level RK3576 regcmd encoder: the CONV_2D, matmul, pool, EW and LUT generators and their packers
  npu_hw.h, npu_cna.h, npu_dpu.h   regcmd word, precision codes, CBUF geometry, LUT epilogue
```

`src/` holds the implementation, one file or more per header. The RK3588 regcmd generators are
in `npu_regcmd.c`, written on Mesa's `rocket` register map: `third_party/mesa-rocket/` holds the
database and `src/rkt_registers.h` the header generated from it. The RK3576 encoder and entries
are in `npu_regcmd_rk3576.c` and the `*_rk3576.c` files. `tests/` holds the tests and benchmarks ([Tests](#tests)).

## The submit seam (`rocket_npu.h`)

Every kernel interaction goes through one set of C symbols, and that set has two
implementations. `src/rocket_npu.c` drives the mainline `rocket` DRM-accel driver, which is
the default (`-DROCKETNPU_PROVIDER=builtin`). An external provider drives a vendor BSP driver
(`-DROCKETNPU_PROVIDER=external -DROCKETNPU_PROVIDER_LIB=...`).

Selection is at link time rather than through a vtable. The seam is a set of C symbols
already, one build targets one driver, and an indirect call on every BO operation would gain
nothing.

The seam is exactly the externally-visible `rocket_*` functions defined in
`src/rocket_npu.c`. A conforming provider must define exactly that set: every one of them,
and no `rocket_*` name of its own that the library already defines.

`rocket_npu.h` declares them, and the header is not the definition of the seam. It also
declares host-side entry points that stay in the core whatever the driver is
(`rocket_affinity_*`, `rocket_pin_worker`, `rocket_num_big_cores`, `rocket_busy_poll_set_us`,
`rocket_fence_wait_*`). Do not enumerate the seam by reading the header, and do not enumerate
it by hand. The list grows, and a hand-kept copy goes stale silently:

```sh
tools/provider-seam.sh                          # print the seam
tools/provider-seam.sh path/to/provider.a       # check a provider (.a / .o / .so / .c)
```

The seam has these groups:

- Device open and close.
- BO allocation, and the 32-bit-IOVA variants.
- Cache maintenance, including the ranged form and its capability query.
- Submit in its matmul, task, pre-arranged and job forms, plus the scratch-size query.
- The capability probes.
- The submit counters.

Configure runs the check for an external provider that names a file, and fails with the
missing symbols listed before an object is compiled. Without the check, a seam that grows
breaks late. A provider *defines* these symbols rather than calling them, so it still
compiles clean. `librocketnpu.a` is a static archive, which is allowed to contain unresolved
symbols, so the library still builds too. The break appears only in whoever links the two
together, as an undefined reference in every executable at once.

The table lists what `rocket_npu.h` declares, which is not the seam: `tools/provider-seam.sh` prints the seam. Every seam function returns a negative errno on failure, except where a row names a `rocket_status`.

| function | what it is |
|---|---|
| `rocket_open()` / `rocket_close(fd)` | Opens `/dev/accel/accel0`, or the node `ROCKET_DEV` names, and checks that its driver is `rocket`. Returns the fd or a negative errno. On an RK3576 with more than one core bound, a second open while an fd is held returns -EBUSY, because two jobs in flight there produce incorrect output. `ROCKET_RK3576_ALLOW_MULTI_FD=1` lifts the refusal for a caller that serializes its own submits |
| `rocket_bo_alloc(fd, size, bo)` / `rocket_bo_free(fd, bo)` | Allocates a BO with `CREATE_BO` and maps it, or frees it. The kernel rounds `size` to the page size. `bo->ptr` is the CPU mapping, and `bo->dma_address` the NPU address to program |
| `rocket_bo_alloc32(fd, size, bo)` / `rocket_bo_ensure32(fd, bo, need)` | `_alloc32` allocates as `rocket_bo_alloc` does, and checks that the last byte lies in the low 4 GB, which every BO a regcmd addresses needs. Outside it, `_alloc32` returns `ROCKET_E_DEVICE` with the BO freed. `_ensure32` grows a BO to at least `need` bytes in that window, and returns 1 if it reallocated or 0 if the BO already fit. Caution: a BO past the window is read as a wrapped address, and the output has the right size and wrong values |
| `rocket_bo_prep(fd, bo, dir, timeout_ns)` / `rocket_bo_fini(fd, bo)` / `rocket_busy_poll_set_us(us)` | Bracket every CPU access to a BO, because BOs are cached. `_prep` invalidates the CPU cache so the CPU sees the NPU's writes, and `_fini` flushes CPU writes for the NPU. `timeout_ns` is relative, and 0 is a non-blocking poll. A wait on a job's output therefore passes a real value such as 2e9, or the call returns -EBUSY. The builtin provider ignores `dir`. `rocket_busy_poll_set_us` sets how long a wait spins before it sleeps, overriding `ROCKET_BUSY_POLL` |
| `rocket_bo_prep_ranges(fd, bo, r, n, timeout_ns)` / `rocket_bo_fini_ranges(fd, bo, r, n)` / `rocket_bo_ranges_supported()` | The same maintenance over only the named `rocket_bo_range`s, since a bracket costs what it walks. The ranges must be ascending, disjoint and inside the BO, or the kernel returns -EINVAL. `n` == 0 covers the whole BO. Below interface 1.5 the calls fall back to the whole-BO form, and `_supported` says whether the saving is available. Caution: a range left out of `_fini_ranges` is not written back, and the NPU reads stale memory |
| `rocket_submit_matmul(fd, regcmd_bo, regcmd_count, in_handles, n_in, out_handles, n_out, timeout_ms)` / `_matmul_flags(…, job_flags)` | Submits one job of one task. `regcmd_count` counts every word the PC executes, the trailer included: 126 for `gen_gemm_fp16` and `gen_gemm_int8`. `in_handles` are the BOs the NPU reads, `regcmd_bo` among them, and `out_handles` those it writes. The builtin provider returns once the job is queued and ignores `timeout_ms`, so wait with `rocket_bo_prep`. It returns -EINVAL for a `regcmd_bo` above 4 GB. Caution: a `regcmd_count` too small hangs the NPU until the job times out, with no error |
| `rocket_submit_tasks(fd, tasks, n_tasks, in_handles, n_in, out_handles, n_out)` / `_tasks_flags(…, job_flags)` / `_tasks_pre(fd, scratch, …, job_flags)` / `rocket_submit_scratch_size(max_tasks)` | Submits one job of many tasks, each a `rocket_task_desc`, with one fence for the job. The call is asynchronous, so wait with `rocket_bo_prep` on the outputs. `_flags` passes a `ROCKET_JOB_*` word. `_pre` allocates nothing, given a resident `scratch` of `rocket_submit_scratch_size(n_tasks)` bytes, and takes the flag word too. Each returns 0 or a negative errno |
| `rocket_submit_jobs(fd, jobs, n_jobs)` | Submits `n_jobs` `rocket_job_desc`s in one ioctl. A job's tasks run in sequence on one core, and the kernel schedules the jobs across cores in dependency order. Jobs run concurrently only when each writes its own output BO, because a shared one serializes them. The call is asynchronous, so wait on each job's output. Returns `ROCKET_E_SHAPE` for a job with no tasks |
| `ROCKET_JOB_BATCHED` / `ROCKET_JOB_NO_DPU_DONE` / `ROCKET_JOB_PPU_DONE` | Job flags, passed to the kernel unchanged. `BATCHED`: the tasks are contiguous and self-chained, and run as one kick. `NO_DPU_DONE`: the program raises no DPU completion, so the driver waits a blind settle. `PPU_DONE`: the last program is a pool, which retires on the PPU's completion. Caution: under `NO_DPU_DONE` a program still writing returns with its tail unwritten. Never set `PPU_DONE` on a job that mixes DPU and PPU programs |
| `rocket_batched_submit_supported()` / `rocket_batched_submit_native()` / `rocket_batch_completion_tracked()` / `rocket_submit_batch_atomic()` | What the kernel does with a multi-task job, each probed once and cached. `_supported`: it honors `ROCKET_JOB_BATCHED`, and a caller must check this rather than `ROCKET_BATCH_SUBMIT`. `_native`: chaining is the uAPI's default, 0 on mainline and 1 on the vendor `rknpu` path. `_completion_tracked`: a batched job's wait starts once the whole kick retires (RK3576, interface 1.4). `_atomic`: one submit's tasks run with no other job between them, which CBUF operand reuse requires |
| `rocket_ppu_done_supported()` / `rocket_no_dpu_done_supported()` | Whether the running kernel honors `ROCKET_JOB_PPU_DONE` and `ROCKET_JOB_NO_DPU_DONE`. A kernel with a flag word rejects an unknown bit with -EINVAL, so software must check these before setting either. Both drivers have `PPU_DONE` from interface 1.3. The RK3576 driver has `NO_DPU_DONE` from 1.2, and the RK3588 driver has it at no version, so the version number alone does not answer this. The versions in these rows are this library's kernel patch series', which creates the `rocket_batch_submit` module parameter: where sysfs lists the `rocket` module without it, every probe answers 0 whatever version the kernel reports |
| `rocket_submit_ioctl_count()` / `rocket_submit_task_count()` / `rocket_submit_counters_reset()` | Count this process's `ROCKET_SUBMIT` ioctls and the tasks they contained. A job of n tasks, chained or gapped, is one submit of n tasks. A path that submits each task alone is n submits. The counters do not count kicks. They are process-wide and atomic, so a per-call figure belongs to that call only when nothing else in the process submits between the reset and the read |
| `rocket_fence_wait_count()` / `rocket_fence_wait_slow_count()` / `rocket_fence_wait_max_us()` / `rocket_fence_wait_counters_reset()` | Count this process's fenced waits (`rocket_bo_prep` and `rocket_bo_prep_ranges` with `timeout_ns` > 0), those that reached the slow-wait mark, and the longest in microseconds. The mark is `ROCKET_SLOW_WAIT_MS`, 450 ms on the RK3588 and 110 ms on the RK3576 by default. Caution: the driver retires a job that never completes and signals its fence, so a hung job looks like a slow wait. Zero slow waits does not prove that no job timed out |
| `rocket_rk3576_retired_counts(unwritten, written)` / `rocket_rk3576_retired_counts_reset()` | Count the RK3576 jobs this process's write guards scored as retired at the driver's backstop rather than completed. `written` counts the ones a write check alone would have passed. The backstop is the profile's `backstop_ms`, 125 on the RK3576, and `ROCKET_RK3576_BACKSTOP_US` overrides it. The entry redoes a retired job after a power cycle, and refuses once its attempts run out. Either pointer can be NULL |
| `rocket_affinity_set_base(base)` / `rocket_affinity_get_base()` / `rocket_pin_worker(worker_idx)` / `rocket_num_big_cores()` | Place host threads on the big (A76) cores. The library pins its fan-out workers round-robin from a thread-local base, which `_set_base` sets and `_get_base` reads. Several context pools in one process each set a distinct base, usually `pool_index * nthreads`, before creating their contexts. `rocket_pin_worker` pins the caller to a big core. `rocket_num_big_cores` returns their count, or 0 with pinning disabled or no distinct big cluster. None of them changes numerics |
| `rocket_driver_name()` | The kernel driver this build reaches, as a short name for a log line: `"mainline rocket driver"` for the builtin provider, or the name an external provider's build gave itself. `-DROCKETNPU_DRIVER_NAME` sets it at build time. The call makes no ioctl, is valid before any device is open, and is not part of the seam |

## Public matmul API (`rocket_matmul.h`)

| function | what it is |
|---|---|
| `rocket_matmul_fp16` / `rocket_matmul_plan` | tiled single-fd fp16 matmul (the validated core). `rocket_matmul_plan` previews its tiling without hardware and returns the tile count |
| `rocket_matmul_fp16_f32out` | fp16xfp16 -> **fp32** output: the full fp32 accumulator, with no per-K-tile or final fp16 rounding. ~5000x more accurate than fp16 output, tracking the true result to the fp32 noise floor (~1e-7). Opt-in. The output readback is 2x, which is free for tall-skinny prefill and costs +35% only when the output bytes rival the weights |
| `rocket_matmul_fp16_mt` | multicore: split N across `nthreads` worker fds (~3 cores) |
| `rocket_matmul_fp16_batch` | runs `nbatch` independent same-shape matmuls as one submit, with one fence, and one IRQ under `ROCKET_BATCH_SUBMIT`. Bit-identical to per-item `rocket_matmul_fp16`. `rocket_mm_batch_create` / `_run` / `_free` keep a batch context across calls ([Batched matmul](#batched-matmul)) |
| `rocket_ctx_create(nthreads)` / `_free` / `rocket_weights_pack` / `_pack_seg` / `rocket_matmul_fp16_prepacked` | pack weights once, reuse across calls (resident fp16). The compute M can differ from the pack M, because the resident layout is M-independent for M>=256, so a weight packed at warmup-M is reused at prefill-M with no re-pack. An incompatible tiling returns -2. `_pack_seg` packs several weights, concatenated along N, as one resident weight with no host concatenation buffer |
| `rocket_ctx_create_ex(nthreads, ROCKET_CTX_TILING_CANONICAL)` | a ctx whose resident weights are M-independent down to M=4, so one pack serves every call M and `_prepacked` never returns -2. Every element exact over 9 shapes and M 4-640 (`tests/matmul_prepacked_canonical_rocket.c`). It suits a resident encoder whose length varies per call ([Canonical tiling](#canonical-tiling)) |
| `rocket_stream_create` / `_free` / `rocket_matmul_fp16_stream[_fused]` | streaming path for LLM prefill: re-packs B, caches scratch per shape, and fuses gate/up |
| `rocket_matmul_int8` / `rocket_matmul_plan_int8` | int8xint8->int32 tiled (pre-quantized in, raw int32 out) |
| `rocket_i8_ctx_create` / `_free` / `rocket_i8_weights_pack` / `_free` / `_bytes` / `rocket_matmul_int8_prepacked` | resident int8 (W8A8) weights, raw int32 out, bit-exact to `rocket_matmul_int8`. One pack serves any compute M, which must be a positive multiple of 4: an unaligned M returns -1. `_bytes` is a packed weight's resident BO footprint ([Resident int8 and group-wise matmuls](#resident-int8-and-group-wise-matmuls)) |
| `rocket_matmul_int8_groupwise` / `rocket_matmul_plan_int8_gw` | per-K-group int8 dequant scales, fp32-accumulated: `C_f[m,n] = Σ_g a_scale[m,g]·b_scale[n,g]·(int32 partial of K-group g)`, the primitive a natively quantized weight needs. No saturation bound on `group`, and `Kt` need only divide it. `rocket_matmul_plan_int8_gw` previews the tiling without hardware and reports the `Kt` it chose (`tests/matmul_int8_groupwise_rocket.c`) ([Resident int8 and group-wise matmuls](#resident-int8-and-group-wise-matmuls)) |
| `rocket_i8_weights_pack_gw` / `rocket_matmul_int8_prepacked_gw` | resident group-wise int8: the int8 codes stay in NPU BOs, and each call quantizes only A (per row, per K-group) and returns the fp32 per-group dequant. The M-independence and M%4 contract of the per-channel form. Bit-exact against the one-shot entry when the group fits the CBUF at the worst-case tile, and within fp32 reassociation for a wider group (`tests/matmul_int8_prepacked_gw_rocket.c`) |
| `rocket_matmul_int4` / `rocket_matmul_int4_ex` / `rocket_matmul_plan_int4` | int4xint4->int16 tiled, accumulated to int32 on the host. `_ex` adds the int16-saturation Kt cap (in-model `[-7,7]` needs `kt_cap=480`). `rocket_matmul_plan_int4` previews the tiling without hardware |
| `rocket_matmul_int4_groupwise` | per-K-group int4 dequant scales, fp32-accumulated, for W4A4 quality, with the group as the saturation-safe K-tile |
| `rocket_i4_ctx_create` / `_free` / `rocket_i4_weights_pack` / `_free` / `_bytes` / `rocket_matmul_int4_prepacked` | resident int4 (W4A4) weights, raw int32 out (per-channel). `_bytes` is a packed weight's resident BO footprint |
| `rocket_i4_weights_pack_gw` / `rocket_matmul_int4_prepacked_gw` | **resident group-wise int4** (the in-model W4A4 path): weight packed once with the K-tile forced to `group`, each call quantizes only A and returns the fp32 per-group dequant `Σ_g a_scale[m,g]·b_scale[n,g]·partial`. Hadamard (baked into the resident weight + applied to A by the caller) is product-preserving, so no driver support is needed. Like the int8 path, the resident layout is M-independent (canonical-tileM), so a weight packed at warmup-M is reused at any prefill-M (incl. a small short-prompt M) with no re-pack |
| `rocket_matmul_int16_exact` | bit-exact int16xint16->int64 via int8 byte-decomposition (4 int8 matmuls) |
| `rocket_matmul_bf16` / `rocket_matmul_plan_bf16` | bf16xbf16->fp32 tiled, single-fd (fp32 A/B in, truncated to bf16 on scatter, with fp32 range and no activation scaling) |
| `rocket_matmul_bf16_mt` | multicore bf16: split N across `nthreads` worker fds over the unchanged single-fd path (~3 cores) |
| `rocket_bf16_stream_create` / `_free` / `rocket_matmul_bf16_stream` | **streaming bf16** for LLM prefill: persistent worker fds + per-shape resident scratch BOs, re-pack A/B per call (the bf16 sibling of `rocket_matmul_fp16_stream`, bit-identical to single-fd at `nthreads=1`). The in-model bf16 path. ~3.2x single-fd warm |
| `rocket_matmul_tf32` / `rocket_matmul_plan_tf32` | tf32xtf32->fp32 tiled (raw fp32 in, HW truncates to a 10-bit mantissa, the first 4-byte-input path) |
| `rocket_matmul_int8_rk3576` / `rocket_matmul_plan_int8_rk3576` | the RK3576's own int8 matmul, a different contract from `rocket_matmul_int8`: `C = sat8(round((A·Bᵀ + bias)·scale))`, an int8 output through the DPU's requant rather than raw int32. Requires `K%32, N%32`, and any M, 1 included. A K of 6176 or more refuses with `ROCKET_E_UNSUPPORTED`. Checked against a CPU model (`tests/rk3576_matmul_gate.c`). [RK3576 matmuls](#rk3576-matmuls) has the tiling, the other refusals and the cost |
| `rocket_matmul_int8_rk3576_perc` | the same int8 matmul with a per-output-column scale: `C = sat8(round_half_to_even((A·Bᵀ + bias)·scale_n[n]))`, `scale_n` N positive entries that the caller calibrates. Every shape rule and refusal of `rocket_matmul_int8_rk3576` applies. `worst_rel_err` (can be NULL) reports the largest relative gain error the call delivered. Exact against a host model of the ramp (`tests/rk3576_mm_requant`) |
| `rocket_matmul_int8_rk3576_perc_sa` | `_perc` with each column's `sum_k \|B[n][k]\|` supplied as `sum_abs_w` (int64, N entries), skipping a pass over B. NULL computes it. The entry does not check the value: one that does not match B plans a wrong gain and returns an output of the right size with wrong values. Bit-identical to `_perc` |
| `rocket_matmul_int8_rk3576_perc_wbo` / `rocket_rk3576_wbo_create` / `rocket_rk3576_wbo_free` | `_perc_sa` with the weight resident: `create` packs B into an N·K-byte device BO once, and the entry takes that handle in place of B. `sum_abs_w` is required, and NULL refuses. A K or N that differs from the handle's refuses, and a K of 6176 or more refuses outright. A stale handle returns an output of the right size computed from the old weight. Bit-identical to `_perc_sa` |
| `rocket_matmul_fp16_rk3576` | the RK3576's fp16 matmul: `C = A·Bᵀ` with fp16 A and B and an fp32 C, all row-major, the whole of K in one task. Accepts `1 <= M <= 2048`, `32 <= K <= 6144` with `K%32`, `64 <= N <= 8192` with `N%16`, and returns `ROCKET_E_UNSUPPORTED` outside them. Exact against a CPU reference whose fills make every sum exact in fp32 (`tests/rk3576_mm_fp16_gate.c`) ([RK3576 matmuls](#rk3576-matmuls)) |
| `rocket_matmul_fp16_rk3576_wbo` / `rocket_rk3576_wbo_fp16_create` / `rocket_rk3576_wbo_fp16_free` | the same fp16 matmul with resident weights: `create` permutes B into a device BO once, and the entry takes that handle in place of B. Bit-identical to `rocket_matmul_fp16_rk3576`, with M free per call inside the limits. A call whose K or N differs from the handle's refuses with `ROCKET_E_SHAPE`. Caution: a stale handle returns an output computed from the old weight |
| `rocket_matmul_int8_rk3576_i32` / `rocket_rk3576_i32_last_stats` | the RK3576's int32-output matmul and its K split: `C = A·Bᵀ + bias` in raw int32, K split internally and the partials summed on the host, so K is bounded by memory and by the 2^24 dimension cap. Requires `K%32, N%32`. All 21 of the test's int32 shapes exact (`tests/rk3576_matmul_gate`). `_last_stats` reports the write guard's counts for the last call on this thread |

Alignment requirements differ by dtype, following the native tile atoms:

| dtype | Alignment |
|---|---|
| fp16 | `K%32, N%16` |
| int8 | `K%32, N%32` |
| int4 | `K%32, N%64` |
| int16-exact | `K%32, N%32`, following int8 |
| bf16 | `K%32, N%16`, the same as fp16 |
| tf32 | `K%16, N%16`, where the 4-byte input halves the K-group to 16 |

All require `M%4==0`. The rule is the planners', which tile a matmul's rows in multiples of four.
An fp16 or int8 task computes any M on the hardware. The one-shot entry points pad M==1->4
internally and return row 0. The pure planners and the resident and streaming paths reject it
instead, so the caller must pad single vectors to 4. The plan functions are pure (no hardware)
and preview the tiling.

Everything in that table except the RK3576 entries emits the **RK3588**
geometry-register encoding. On a SoC that does not run it the entries **refuse**
(`ROCKET_E_UNSUPPORTED`, naming the entry that does work) rather than submit a job that
completes and writes nothing. The M rule above does not apply to the RK3576, where every M,
including 1, is correct.

### Batched matmul

`rocket_matmul_fp16_batch` collapses a set of small, dispatch-bound GEMMs that share a shape.
Flash attention's per-worker QK and AV submit batching uses it.

`rocket_mm_batch_create`, `rocket_mm_batch_run` and `rocket_mm_batch_free` hold a batch
context across calls. The context keeps resident input, weight and output BOs, grown only on a
larger shape. A prezero-once guard skips the full-BO zero when the `(M,K,N,nbatch)` layout
repeats. For a repeated-shape caller, such as flash attention's per-layer QK and AV, the
context removes the per-call BO allocation and zero. The one-shot entry is a thin wrapper over
it.

### Canonical tiling

On a `ROCKET_CTX_TILING_CANONICAL` context, a plan at M < 256 keeps the M=256 plan's K and N
tiling (Mt = M). The resident weight's layout therefore never moves. The cost is the K tile a
small M gives up: 1.16-1.42x a pack made for that M at M 48-128, and parity from 196.
`ROCKET_CANON_BENCH=21` makes `tests/matmul_prepacked_canonical_rocket.c` print the cost.

### Resident int8 and group-wise matmuls

The compute M of `rocket_matmul_int8_prepacked` can differ from the pack M, because the
resident tile layout is M-independent: the tiling is planned at `MAX_TILE` (canonical-tileM).
A weight packed at warmup-M is reused at any prefill-M with no re-pack, and the int32
K-accumulation is exact for any tiling. An incompatible tiling returns -2. An unaligned M is
rejected rather than padded, because padding M would need a matching pad of `a_scale`, which
only the caller has. The caller pads ragged row counts with `rocket_pad_m`.

`rocket_matmul_int8_groupwise` serves a GGUF MXFP4, Q8_0 or Q4_K block, which has one scale per
K-block. The NPU cannot apply a K-blocked scale on chip, because at the output stage K is fully
contracted. Integer partials already leave the chip at every K-tile boundary, since no
on-device integer K-accumulation is implemented. The block scale therefore costs nothing extra
at a boundary that already exists: it fuses into the readback accumulate.

Unlike the int4 twin, the group-wise int8 matmul has no saturation bound on `group`, because
the output accumulator is int32, not int16. `Kt` need only divide the group, not equal it: a
K-tile must lie inside one group, not be one. A group wider than the CBUF cap therefore stays
legal. The resident group-wise call measures 0.97-0.99x the resident per-channel one at M 64
and 0.81-0.90x at M 256 (`tests/moe_gw_bench.c` at a gpt-oss-20b expert, RK3588 at ~816 MHz,
2026-10-09).

The resident group-wise int8 codes are scattered into NPU BOs once and never leave. A
quantized weight then costs no per-forward-pass dequant and no per-call weight scatter. A group
that fits the CBUF at the worst-case tile forces `Kt == group` in both the resident and the
one-shot paths, which are then bit-exact. A wider group can get a different `Kt` divisor, and
the two then agree to fp32 reassociation.

### RK3576 matmuls

`rocket_matmul_int8_rk3576`'s contract differs from `rocket_matmul_int8`'s, so it is a
separate entry rather than a path inside `rocket_matmul_int8`. N is tiled, and throughput
rises with it. A submit's device time rises from ~0.12 ms at N 32 to ~0.91 ms at N 2048
(`rocket` 1.6.0, 786 MHz, 2026-09-25). Sixty-four times the work therefore takes 7.6 times as
long. `ROCKET_RK3576_MM_NT` overrides the tile.

A K of 6176 or more is past one task's contraction. `ROCKET_RK3576_MM_KSPLIT=1` sends it through
`rocket_matmul_int8_rk3576_i32` with the requant on the host. For speed, keep K at or below
4608: past 4992 the output tile drops from 1024 channels to 32. The planner and the entry refuse
a dimension past 2^24 with `ROCKET_E_SHAPE`. They also refuse an M whose plane is taller than
the 65535 rows its 16-bit height field holds, such as any prime M past 65535. They refuse a plan
of more than `INT_MAX` jobs with `ROCKET_E_TILING`.

`rocket_matmul_int8_rk3576_perc` programs the per-column gain in the coefficient group's int16
multiplier and the task's shift word. A column therefore resolves its scale to 0.5/C relative,
coarser where a tile's scales spread. The entry does not sort columns, and it cannot compute
`scale_n`.

`rocket_matmul_fp16_rk3576` contracts the whole of K in one task, where the fp16 convolution
contracts sixteen channels a task. B is one row per output channel, and the K and N limits are
one task's measured bounds. A task contains at most `4096/(K/32)` rows, the CBUF data window.
The entry cuts M into equal tasks under that and issues them as one submit
(`rocket_rk3576_mm_fp16_task_rows()`).

A call whose output still holds the sentinel is redone after a power cycle.
`rocket_rk3576_mm_fp16_last_attempts()` reports how many submits the last call took.
`ROCKET_MM_PROFILE` prints its buckets. Its output leaves the next submit unaffected.

`rocket_rk3576_wbo_fp16_create` also builds the coefficient buffer for its N in a device BO,
and refuses a K or N outside the limits. The handle is not checked against the weight it came
from. A stale one returns an output of the right size computed from the old weight. Each handle
uses K·N·2 bytes of weight and about 257 KiB of coefficients.

`rocket_matmul_int8_rk3576_i32` runs on the dense writer, the int8 program on a whole wide
output stage. That stage writes every accumulator of a task as a raw int32. The entry therefore
programs the real output channels (rounded to 32), runs its submits back to back and leaves the
next submit unaffected. A call takes 1.0 ms at 8×64×32 and 48.1 ms at 32×1024×4096 (`rocket`
1.6.0, 2026-09-30). M 512, K 8192, N 2048 runs in 345.7 ms, with 4096 sampled elements exact.

A sentinel stamp and a per-atom scan remain the int32 entry's write guard, and a retired job is
redone after a power cycle. `rocket_rk3576_i32_last_stats` reports the guard's counts for the
last call on the calling thread: submits, row tasks, redos and the atoms behind them. The next
call overwrites them.

`ROCKET_RK3576_I32_DENSE=0`, or `ROCKET_RK3576_I32_OC_MULT=2` or `=4`, takes the older narrow
and wide writers. Those deliver a quarter or a half of the programmed channels, leave the next
submit writing nothing, and idle between submits for it (`ROCKET_RK3576_MM_GAP_MS`).

> [!CAUTION]
> On the narrow writer, M 512, K 8192, N 2048 leaves the RK3576 unresponsive until a reboot.

## Conv API (`rocket_conv.h`)

Beyond the matmul, the library runs a general 2D convolution, the basis of the
`tflite-rocket` detection delegate. A 1×1 pointwise conv is the matmul, and everything else
is the conv path. HW-validated **bit-exact**.

| function | what it is |
|---|---|
| `rocket_conv2d_fp16` | general fp16 `CONV_2D`: KxK / stride / dilation, symmetric pad, OC%16 pad, OC/OH/OW spatial tiling. Allocates and frees its 5 BOs per call |
| `rocket_conv1d_fp16(fd, ic, it, oc, kw, stride, pad, …)` | **conv1d** (the Whisper encoder front-end: width-only 1D conv over time), lowered onto `rocket_conv2d_fp16` with **time on the height axis** (IW=1, so the OH-row tiler fits CBUF, while the IH=1 width layout overflows the feature banks for Whisper IC=80/512). Whisper conv1/conv2 (KW=3, pad=1, stride 1/2) HW-validated (`tests/conv1d_rocket.c`) |
| `rocket_conv2d_fp16` (`desc.depthwise=1`) | native depthwise (`DW_EN`, group G=32 + the DPU depthwise register fixes), channel + spatial tiled |
| `rocket_conv2d_act_fp16(fd, d, kind, …)` / `_ctx` | **conv->activation fusion**: a direct fp16 conv post-processes its own CACC result with a smooth `f(x)` in the same task (the DPU LUT epilogue in the conv program): `out = f(conv(x))`, no 2nd round-trip. `kind` = `SILU`/`TANH`/`GELU`. HW-validated: the fused epilogue **bit-reproduces the standalone LUT** (<=0.0039), conv->tanh bit-accurate (`tests/conv_act_rocket.c`). Off by default, and byte-identical when off. HardSwish/depthwise rejected (flat-tail quirk / direct-only). Caution: a narrow x~0 LE/LO boundary glitch spikes all single-pass kinds, and the 2-pass `x·gate(x)` path avoids it |
| `rocket_conv2d_int8` | native **int8 direct** `CONV_2D`: int8xint8->int32 raw accumulate (caller requants), OC pad 32 / IC pad 32, OC/OH/OW tiling. Bit-exact vs an int64 oracle. It is also the substrate for the `tflite-rocket` delegate's native **uint8** convs: the caller recenters uint8->int8 `byte−128` and folds the centering into its requant and a box-sum, with no driver change |
| `rocket_conv2d_dw_int8` | native **int8 depthwise** (int8-out, on-chip requant): per-tensor quant, any int8 weight zero point (the DPU's CPEND operand adds `-w_zp` times each window's input sum), the input zero point folded into the bias. TFLite's accumulator exactly, within one of TFLite at a rounding boundary. A uint8 tensor maps onto it by subtracting 128 from its values and zero points |
| `rocket_conv2d_dw_int8_plan` | the pure shape check behind `rocket_conv2d_dw_int8`: `ROCKET_OK` when its NPU path takes the descriptor, `ROCKET_E_SHAPE` or `ROCKET_E_UNSUPPORTED` otherwise. Opens no device, so a frontend claims a node with it |
| `rocket_conv2d_int8_q` / `_ctx` / `_mt` / `_plan` | native **int8 direct conv with the int8-out writer** (RK3588): Mesa's direct program, requantized on chip through the OUT_CVT, any per-tensor weight zero point on CPEND, the input zero point folded into the bias. Shares the int32-raw entry's planner, tiler and job body. Programs whole 32-kernel groups and materializes the halo (the input zero point on real channels, 0 on the channel padding). Within one of TFLite at a rounding boundary |
| `rocket_conv2d_int8_q_perc` / `_perc_ctx` / `_perc_mt` / `_perc_plan` | native int8 direct conv with per-channel weight scales (TFLite's per-axis quantization), RK3588: the int8-out program of `rocket_conv2d_int8_q` with each output channel's scale on the BS stage's int16 multiplier and one OUT_CVT gain per job. Within one of TFLite at a rounding boundary (`tests/conv2d_int8_q_perc_rocket.c`). [Per-channel int8 convolution](#per-channel-int8-convolution) has the job split |
| `rocket_conv2d_dw_int8_perc` / `_perc_ctx` / `_perc_plan` | native **int8 depthwise with per-channel weight scales** (TFLite's per-axis quantization), RK3588: each channel's scale is programmed in an int16 multiplier the DPU's BS stage reads per channel, with one OUT_CVT gain per job. Channels are sorted by scale and cut into jobs so each keeps a multiplier of at least `ROCKET_DW_PERC_MIN_C` (2048). Within one of TFLite at a rounding boundary. The host writes an all-zero filter as its bias's constant |
| `rocket_conv_ctx_create` / `_free` / `rocket_conv2d_fp16_ctx` / `rocket_conv2d_int8_ctx` / `rocket_conv2d_dw_int8_ctx` | the same convs with a **resident** BO pool reused across calls/tiles (ctx=NULL ⇒ byte-identical legacy alloc-per-call) |
| `rocket_conv_pool_create` / `rocket_conv2d_int8_mt` / `rocket_conv_pool_free` | **multicore** int8/uint8 direct conv: a pool of N worker fds (each its own resident `rocket_conv_ctx`) fans the conv's independent OC/OH/OW tiles across the 3 NPU cores. **Bit-identical** to `rocket_conv2d_int8` (same tiles/jobs). It falls back to serial for single-tile convs. Mirrors `rocket_matmul_fp16_mt`'s one-fd-per-core design |
| `rocket_conv_transpose2d_fp16(fd, d, in, W, out)` / `_ctx` | ConvTranspose2d (deconvolution, learned upsampling). Weights `[IC][OC][KH][KW]` direct or `[C][1][KH][KW]` depthwise (`desc.depthwise=1`, OC==IC, C%32). Any stride, pad, output_padding, dilation, OC and IC. `_plan` returns `−2` for the unimplemented `pad > d·(K−1)` crop case. Two paths, both bit-exact against a scatter reference (`tests/conv_transpose_rocket.c`) ([ConvTranspose paths](#convtranspose-paths)) |
| `rocket_conv_transpose2d_route(d)` | which of the two paths a descriptor takes (`ROCKET_CONV_TRANSPOSE_DECONV` / `_LOWERED`), or the plan's negative refusal. Pure. `ROCKET_CONV_TRANSPOSE_HW=0` forces the lowering |
| `rocket_conv_transpose2d_weights_pack(ctx, d, W, bias)` / `rocket_conv_transpose2d_fp16_prepacked(ctx, w, in, out)` / `_weights_free` / `_prepacked_plan(d)` | the resident-weight ConvTranspose, for a model that runs the same layer repeatedly: one resident fp16 GEMM, `in^T . W`, on the caller's `rocket_ctx`, then a host scatter-add of each tap's plane into the output. Any stride, pad (a cropping pad included), dilation and output_padding. Within a few fp16 roundings of each element's magnitude sum, bit-exact on small-integer data (`tests/conv_transpose_resident.c`). The plan refuses depthwise and non-RK3588 SoCs (`ROCKET_E_UNSUPPORTED`) and a tap-plane buffer past 64 MiB (-4) ([Resident convolutions](#resident-convolutions)) |
| `rocket_conv2d_weights_pack(ctx, d, W, bias)` / `rocket_conv2d_fp16_prepacked(ctx, w, in, out)` / `_weights_free` / `_prepacked_plan(d)` / `_prepacked_estimate_ms(d)` | the resident-weight forward conv, for a model that runs the same layer repeatedly: a host im2col of the input, then one resident fp16 GEMM against the weight as stored (`[OC][IC·KH·KW]`) on the caller's `rocket_ctx`. Any stride, pad, dilation and kernel. Within a few fp16 roundings of each element's magnitude sum, bit-exact on small-integer data (`tests/conv_resident.c`). `_prepacked_estimate_ms` models a call's time. The plan refuses depthwise and non-RK3588 SoCs (`ROCKET_E_UNSUPPORTED`) and an unfold or staging buffer past 64 MiB (-4) ([Resident convolutions](#resident-convolutions)) |
| `rocket_upsample_nearest_fp16` / `rocket_upsample_bilinear_fp16` (`rocket_resize.h`) | **integer-factor resize** (the FPN/decoder neck: `RESIZE_NEAREST_NEIGHBOR` / `RESIZE_BILINEAR`). Realized as a **depthwise ConvTranspose** with a fixed box (nearest) / separable-triangle (bilinear) kernel, where `pad=(k−s)/2` gives exactly `IH·s x IW·s`. Nearest = bit-exact block replication, and bilinear = half-pixel 2-tap (the triangle's stride-subsample is a partition of unity), `align_corners=False` interior + zero boundary. `C%32` (depthwise group). HW-validated vs independent gather refs + partition-of-unity / linear-exactness properties (`tests/resize_rocket.c`) |
| `rocket_upsample_nearest_plan(C, IH, IW, scale_y, scale_x)` / `rocket_upsample_bilinear_plan` / `rocket_upsample_nearest_ref_fp16` / `rocket_upsample_bilinear_ref_fp16` (`rocket_resize.h`) | `_plan` returns 0 when an upsample can run, checking `C % G`, positive scales and the forward convolution's CBUF fit, and a negative value otherwise. The `_ref_fp16` functions compute each upsample on the CPU, independently of the NPU path, and `tests/resize_rocket.c` compares against them |

Pure planners (`rocket_conv2d_plan` / `_oh` / `_ow`, `rocket_conv_out_dim`,
`rocket_conv2d_trail_x` / `_trail_y`, plus `rocket_conv_transpose2d_plan` / `_oh` / `_ow`)
preview the dimensions and the CBUF-fit check without hardware. The trail functions return the
pad the last window consumes past the plane, which is zero for a symmetric descriptor.
`rocket_conv2d_ref_int8` / `rocket_conv_transpose2d_ref_fp16` are golden oracles.
`rocket_conv2d_ref_fp16` is the fp16 conv's CPU reference, accumulating in fp32 and storing
fp16, and also a host fallback.

Native **int8** `CONV_2D` runs end-to-end and is HW-validated, both the regcmd and
cube layers (`gen_conv2d_int8` / `gen_conv2d_dw_int8`) and the runtime wrappers.
The direct conv has two output forms. `rocket_conv2d_int8` writes the raw int32 accumulator for
the caller to requantize. `tflite-rocket`'s `rocket_out_nchw_to_nhwc_q_per_axis` does that per
axis, with the input zero-point correction folded into the bias, bit-identical to CPU TFLite.
`rocket_conv2d_int8_q` and `rocket_conv2d_int8_q_perc` requantize on chip and write int8, with
a per-tensor or a per-channel weight scale, within one of TFLite at a rounding boundary.

The depthwise conv writes int8 with on-chip requant
(`conv_params_t.int8_out=1`: QD_EN + per-OC int32 bias in the BS ALU + `OUT_CVT` requant),
per-tensor with any int8 weight zero point (`rocket_conv2d_dw_int8`), or with per-channel
weight scales (`rocket_conv2d_dw_int8_perc`). Its accumulator is TFLite's exactly and its
requant is the DPU's. An output can therefore differ from CPU TFLite by one at a rounding
boundary (`tests/conv_dw_int8_runtime.c`). `tests/replay_dw_mesa.c` checks the register program
against Mesa's. The `tflite-rocket` delegate sends signed-int8 convs to these under
`--option native_int8=1`.

### Per-channel int8 convolution

`rocket_conv2d_int8_q_perc` sorts the channels by scale and cuts them into jobs at
`ROCKET_DW_PERC_MIN_C` (2048). Each job has whole 32-kernel groups, and every job's tiles go in
one dispatch across the pool. A layer that plans to one job keeps its channel order and the
one-job fast path. The host writes an all-zero filter's channel.

### ConvTranspose paths

`rocket_conv_transpose2d_fp16` serves segmentation and decoder heads, super-resolution and an
FPN's learned upsampling. On the RK3588 one class of descriptor runs on the CNA's hardware
deconvolution mode, one task per output-channel tile. That class is a direct transpose at a
power-of-two stride (2, 4 or 8), dilation 1, `IC%32==0` and a compact input that fits one CBUF
pass. It takes 0.35-0.54x the lowering's wall-clock time at decoder shapes.

Everything else is lowered to an interior-dilated input, `rot180(Wᵀ)` and a stride-1 forward
`rocket_conv2d_fp16`, inheriting the conv's hardware-exact tiling. The lowering's cost scales
with the upsampled size. The direct weight is in-channels first. `_plan`'s −2 is a clean
decline.

### Resident convolutions

`rocket_conv_transpose2d_fp16_prepacked` reads the input as IC planes of IH·IW pixels and the
weight as stored (`[IC][OC·KH·KW]`). That product is every tap of every input pixel, with no
inserted zeros. The GEMM runs in channel-planes form, with no host transpose on either side.
`rocket_conv_transpose2d_weights_pack` transposes the weight and packs it once into the handle,
with an optional `[OC]` bias. The handle never re-reads W: a changed weight is a new pack.

The scatter-add is split by output channel across the context's worker threads
(`ROCKET_CT_SCATTER_THREADS`). The workers split the GEMM's M, each holding the whole weight,
once each has 64 rows and the weight is at most 4x the input. Otherwise they split N
(`ROCKET_CT_SPLIT=m|n`). The worst element error measured is 2^-11.1 of its magnitude sum.

At pix2pix's and SAM's ten layers a call takes 0.94-9.16 ms, with the scatter-add on one thread
[HW sweep, mainline RK1, five rotated passes, 2026-09-27]. The call takes 0.11-0.58x ONNX
Runtime's 2-thread per-node time on the same board, and 0.02-0.29x the one-shot entry's. The
threaded scatter runs at 0.45-0.57x its one-thread time at up6, up7 and SAM's upscalers and
0.6-0.8x at up8 [HW sweep, RK1 at ~816 MHz, 2026-10-08].

`rocket_conv2d_fp16_prepacked` unfolds the input into `IC·KH·KW` planes of the `OH·OW` output
pixels, and the GEMM's channel-planes output is the conv's `[OC][OH][OW]`.
`rocket_conv2d_weights_pack` packs the weight once into the handle, with an optional `[OC]`
bias. The workers split the GEMM by the ConvTranspose entry's rule (`ROCKET_CONV_SPLIT=m|n`).
The unfold runs on the context's worker count of threads (`ROCKET_IM2COL_THREADS`).

A 1x1 stride-1 unpadded conv reads its input in place. An explicit `oh` or `ow` sets a
trailing pad. The worst element error measured is 2^-11.0 of its magnitude sum.

A forward call takes 1.1-5.8 ms at pix2pix's eight encoder layers (c2-c8 sum 19.7 ms). At
Depth Anything v2-small's neck it takes 0.5-11.9 ms [HW sweep, mainline RK1 at ~816 MHz,
governor `ondemand`, two passes, 2026-10-08, `tests/conv_model_bench.c`].
`_prepacked_estimate_ms` is a linear model of a call, for a frontend deciding whether to claim
a node. It is fitted to 22 shapes at 928 MHz, and within 0.85-1.16x of the measured time there.

### RK3576 convolution, pooling and activation

These entries run the RK3576's own encoder. On an RK3576,
`rocket_conv2d_dw_int8` and `rocket_conv2d_fp16` dispatch to the entries below. `rocket_conv2d_int8`
writes raw int32 and refuses there, naming `rocket_conv2d_int8_rk3576`. Each entry refuses on
any other SoC.

| function | what it is |
|---|---|
| `rocket_conv2d_int8_rk3576` / `rocket_conv2d_int8_plan_rk3576` | **the RK3576's int8 direct conv**: in `[IC][IH][IW]`, W `[OC][IC][KH][KW]`, out `[OC][OH][OW]`, per-tensor quant with model-domain signed zero points, int8 out through the DPU's requant. Any IC and OC, with rows and output channels tiled. Four or fewer input channels take the packed-image first conv. Refuses dilation, a weight slice that fits no output-channel group, a width past 8192, a kernel past 32 and a stride past 7. Bit-exact against a CPU model (`tests/rk3576_conv_lib_gate.c`) |
| `rocket_conv2d_dw_int8_rk3576` | the depthwise form, W `[C][KH][KW]`, with a per-tensor weight zero point stored in the weight cube. Refuses four or fewer channels and a channel count past 8192, with the direct entry's other refusals |
| `rocket_conv2d_int8_perchannel_rk3576` | the direct or depthwise int8 conv with **per-output-channel weight scales** `w_scale[oc]`, TFLite's per-axis form. Each channel's gain is programmed in the coefficient group's int16 multiplier under one OUT_CVT and shift per task, so it is close to a per-axis requant rather than bit-exact, and the planner logs the worst gain error it accepted. Refuses four or fewer input channels on the direct path and a weight zero point. Bit-exact against a CPU model of that arithmetic (`tests/rk3576_perchannel_gate.c`) |
| `rocket_conv2d_int8_act_rk3576` | the direct or depthwise int8 conv with a **LUT activation fused into its epilogue**, one submit, `kind` a `ROCKET_RK3576_ACT_*`. It reads the accumulator before rounding, so the hardware interpolates between table entries and the distance to an exact f() is reported, not asserted. Per-tensor quant only. Refuses a value unit too fine for the table's int16 peak and a domain no step reaches (`tests/rk3576_act_gate.c`) |
| `rocket_conv2d_fp16_rk3576` | **the RK3576's fp16 conv**, fp16 in and out. One task contracts sixteen input channels, and the slices sum on the host in fp32 and round once. Each slice writes fp32 through a whole output stage, so the call leaves the next submit unaffected. `ROCKET_RK3576_FP16_F32OUT=0` restores the fp16 writer and its power cycle per program. Four or fewer input channels take the packed-image first conv, which cycles per program. Refuses depthwise and a plane whose sixteen-channel slice overflows the CBUF |
| `rocket_residual_add_weights_rk3576` | **a residual add as a 1×1 convolution's weights**: fills W (`c` by `2c` int8), the `c` int32 biases and the weight scale for `out = requant(w1·(a - a_zp) + w2·(b - b_zp))`. The caller concatenates the operands along channels and runs `rocket_conv2d_int8_rk3576`. The operands can have different scales, resolved to about one part in 127², and the bias applies both zero points exactly. Pure, no hardware. Refuses `2c > 4608` and operand scales more than 127x apart (`tests/rk3576_residual_add.c`) |
| `rocket_pool_int8_rk3576` / `rocket_pool_int8_rk3576_plan` / `rocket_pool_int8_rk3576_exact` / `rocket_pool_int8_pack_rk3576` / `rocket_pool_int8_prepacked_rk3576` / `rocket_pool_int8_free_rk3576` | the RK3576's int8 MaxPool and AveragePool on the PPU, `[C][IH][IW]` -> `[C][OH][OW]`. A different function from `rocket_pool_int8`: the average rounds half to even, divides by the whole window and pads with the input zero point. `_exact` says without a device whether a window's average is exactly rounded. Refuses rows or channels past 8192, and the pack returns NULL where `rocket_pool_npu_supported()` returns 0. Bit-exact against `rocket_pool_ref_int8_rk3576` (`tests/rk3576_pool_probe.c`) ([RK3576 pooling](#rk3576-pooling)) |
| `rocket_act_int8_rk3576` | **an int8 activation on the DPU LUT**: `[C][H][W]` int8 in and out, per-tensor quant. `kind` is a `ROCKET_RK3576_ACT_*`: sigmoid, tanh, SiLU, HardSwish, HardSigmoid or ELU. Each of the 256 inputs maps to its own table entry, so the only error is that entry's quantization. The table load and the op run as two tasks of one submit (`tests/rk3576_act_gate.c`) |
| `rocket_pool_int8_rk3576_max_ow(d)` / `rocket_pool_int8_rk3576_ow_slices(d)` | The output columns one RK3576 pooling task can produce, and the number of column slices a descriptor needs. Neither needs a device. The allowance starts at 128 and halves for each output row in flight over one input row, so it depends on the kernel height and vertical stride. A wider plane runs one submit per slice. The plan refuses more than `ROCKET_RK3576_POOL_MAX_SLICES` (8). Caution: a task past this width writes an output that is wrong from column 0 |
| `rocket_pool_int8_rk3576_lag_check(fd, h, surf)` / `_lag_observable(h)` / `_lag_can_show(h)` | Caution: a padded average pool on the RK3576 intermittently divides a window by the previous window's tap count, in an output a write check passes. The per-call entry checks and redoes its own submit. A caller that issues the submit, such as a cross-layer kick, calls `lag_check` on its CPU-synced output `surf`. It returns 0 when the kick must be redone, and 1 otherwise. `_lag_observable` says whether the check runs. A placement decision reads `_lag_can_show`, which ignores `ROCKET_RK3576_POOL_LAGCHECK=0` |
| `rocket_pool_int8_rk3576_lag_src_handle(h)` / `_lag_src_bytes(h)` / `_lag_counts(h, fires, calls)` / `_lag_attempts()` / `_lag_discr(h)` | Support for the lag check. `_lag_src_handle` names the BO the check reads, which the caller must not overwrite before the check runs. `_lag_src_bytes` is the bytes the check's bracket walks, 0 with no check. `_lag_counts` returns how often the divisor lagged and over how many checks, and `_lag_discr` how many scored elements could have shown it, counted under `ROCKET_RK3576_POOL_LAG_DISCR` only. Both sum a parent's column slices. `_lag_attempts` is the redo budget, 32 by default, set by `ROCKET_RK3576_POOL_LAG_ATTEMPTS` |
| `rocket_conv2d_int8_perchannel_plan_rk3576(d, W, bias, in_scale, w_scale, out_scale, in_zp, out_zp, oc_tile, max_rel_err)` | Whether every channel's gain fits the int16 `C` ramp of `rocket_conv2d_int8_perchannel_rk3576`, and the output-channel tile it uses. Pure. Returns `ROCKET_OK`, `ROCKET_E_UNSUPPORTED` when a live channel's gain is off the ramp, such as one below 1/65534 of its task's largest scale, or `ROCKET_E_SHAPE`. `oc_tile` and `max_rel_err` can be NULL, and are filled on a refusal too. `max_rel_err` is reported, not asserted (`tests/rk3576_perchannel_gate.c`). Caution: `ROCKET_RK3576_PC_ALLOW_CLAMP=1` runs such a channel at the wrong gain |
| `rocket_conv2d_int8_perchannel_oc_tile_rk3576(d, W, bias, in_scale, w_scale, out_scale, in_zp, out_zp)` | The output-channel tile the per-axis entry will use, or 0 for a descriptor it refuses. The entry's submit count is `ceil(oc/tile)` row-task sets. The tile is also an accuracy parameter: a task has one OUT_CVT shift, and a smaller tile narrows the scale spread its ramp covers. The planner takes the largest tile whose predicted worst gain error meets `ROCKET_RK3576_PC_MAX_ERR` (default 1%). `ROCKET_RK3576_PC_OC_TILE` forces a tile. A depthwise descriptor returns `oc` |
| `rocket_residual_fuse_weight_rk3576(in_scale, w_scale, skip_scale, in_zp, skip_zp, w_skip, bias_delta, rel_err)` | The identity weight that fuses a residual add into a block's last convolution, as `c` extra input channels with an identity block at the kernel's center tap. The weight is `skip_scale / (in_scale·w_scale)`, and it must fit int8. `bias_delta` receives the constant to add to every bias, and `rel_err` (can be NULL) the weight's rounding error. Pure. Returns `ROCKET_E_UNSUPPORTED` when the weight does not fit, which is the result for every real residual block measured |

#### RK3576 pooling

`rocket_pool_int8_rk3576` takes the input zero point it pads with. A plane wider than one
task's output allowance splits by columns, up to 8 slices. The pack handle
(`rocket_pool_int8_pack_rk3576`) holds the buffers and program across calls, bit-identical to
the one-shot entry.

### RK3576 resident weights, cube layout and chained runs

These entries keep an RK3576 int8 network on the device between layers. A handle packs a layer's weights once. A cube join passes one layer's output to the next with no host transpose. A cube slice places several producers in one buffer, and a chain runs a run of joined layers as one submit and one kick. The pack and chain constructors refuse on any other SoC.

| function | what it is |
|---|---|
| `rocket_conv2d_int8_pack_rk3576(fd, d, W, bias, in_scale, w_scale, w_scale_oc, out_scale, in_zp, w_zp, out_zp)` | Packs an RK3576 int8 conv's resident weights once: the filter sums, the coefficient group and the weight cube, and on the per-axis path the scale sort and output-channel tile. A NULL `w_scale_oc` selects the per-tensor contract with `w_scale`. A non-NULL one selects the per-axis contract and ignores `w_scale`. `desc.depthwise` selects direct or depthwise. Four or fewer input channels take the packed-image first conv unless `direct_datapath` is set. Returns NULL for anything the per-call entries refuse |
| `rocket_conv2d_int8_prepacked_rk3576(fd, h, in, out)` / `rocket_conv2d_int8_weights_free_rk3576(fd, h)` | Runs a packed handle, bit-identical to the per-call entry. The handle fixes the fd, the tile plan and the quantization, and refuses a foreign fd. It contains the feature cube, register program and each tile's output, so a call allocates no BOs. Every tile's weight cube stays resident, about the size of `W`. `in` can be NULL when the handle reads a cube, and `out` when it leaves one. Caution: a handle is not thread-safe |
| `rocket_conv2d_int8_cube_out_rk3576(h, on)` / `rocket_conv2d_int8_cube_of_rk3576(h, out)` | `cube_out` leaves a handle's output in cube layout instead of de-tiling it. `cube_of` describes that output as a `rocket_rk3576_cube`, valid from the pack until the producer is freed. The next layer reads it with no transpose, and the bytes are the ones the row-major entries de-tile. Both refuse a handle whose output channels span more than one tile ([Cube tails and row pitch](#cube-tails-and-row-pitch)) |
| `rocket_conv2d_int8_cube_in_rk3576(h, src)` | Reads a handle's feature input from a producer's cube instead of a row-major tensor, and skips the scatter. A NULL `src` restores row-major input. The cube is borrowed and must outlive the calls that read it. Refuses a cube whose plane, channel count or fd differs from the handle's, a channel-group stride shorter than the plane, and a cube with a row pitch. A packed-image handle refuses every cube, so it can only begin a run |
| `rocket_conv2d_int8_cube_shared_rk3576(h, on)` | Declares that a handle's output is also read outside its chain, later in the same inference, for example by a second consumer. The chain then stamps that output at the start of the handle's next call, instead of in its verify bracket. It costs one PREP/FINI pair per call. A handle that writes into a caller's buffer needs no declaration. Caution: without it, the outside consumer reads the sentinel and returns a complete output computed from 0xA5 |
| `rocket_rk3576_cube_alloc(fd, c, h, w, out)` / `rocket_rk3576_cube_slice(buf, c0, c, out)` / `rocket_rk3576_cube_free(fd, buf)` | Allocates a cube buffer for several producers, cuts views of it, and frees it. A slice is `c` channels from channel `c0`, borrowed and valid while `buf` is. Producers that write adjacent slices form a channel concatenation with no host copy, such as a residual add's two operands. `_slice` refuses a `c0` that is not a multiple of 16, and a range past the end. The caller is responsible for the buffer |
| `rocket_rk3576_cube_declare_tail(c, live, value)` | Narrows a cube to `live` channels and states that the channels past them, to the end of its groups, contain `value`. A consumer whose input count is not a multiple of 32 reads one group more. At a nonzero weight zero point those channels enter the result, so `cube_in` accepts the join only when `value` is the consumer's input zero point. The caller asserts that nothing else writes there. Refuses a `live` past the cube's channels |
| `rocket_conv2d_int8_cube_out_at_rk3576(h, dst)` | Writes a handle's output into `dst`, a slice of a caller's buffer, and releases the handle's own output BO. It implies cube output, and `cube_of` then describes the slice. A NULL `dst` restores the handle's own buffer and clears cube output. The buffer is borrowed. Refuses a foreign fd, more than one output-channel tile, a plane, channel-group stride or row pitch that is not the handle's, and a slice too short for its channels or not on a whole group |
| `rocket_pool_int8_cube_in_rk3576(h, src)` | The pool handle's input side of a cube join. It reads a producer's cube, skipping the scatter and its cache maintenance, so `in` can be NULL. A NULL `src` restores row-major input. The cube is borrowed. A column-split handle takes a cube too. Refuses a foreign fd, a mismatched plane or channel count, a buffer too short for the channel groups the PPU walks, and a channel count that is not a multiple of 16 |
| `rocket_pool_int8_cube_out_rk3576(h, on)` / `_cube_of_rk3576(h, out)` / `_cube_out_at_rk3576(h, dst)` | The pool handle's output side: leave the output in cube layout for the next convolution, describe it, or write it into a slice of a caller's buffer beside other producers. `out` can then be NULL. `_cube_of` refuses a channel count that is not a multiple of 16. `_cube_out_at` refuses a foreign fd, another plane, a slice too short, an offset that is not a whole group, and a channel-group stride below the plane. A column-split handle refuses cube output |
| `rocket_conv2d_int8_chain_new_rk3576(fd, h, n)` | Builds one batched job from a run of `n` resident handles, so the run takes one submit and one kick. Every handle but the last must leave a cube. Every handle but the first must read a cube that an earlier member wrote. The handles are borrowed and must outlive the chain. Refuses fewer than two handles, a multi-tile handle, a handle with a DPU LUT, a broken run, a foreign fd and a kernel without `DRM_ROCKET_JOB_BATCHED`, returning NULL |
| `rocket_conv2d_int8_chain_run_rk3576(fd, c, in, out)` / `_chain_kicks_rk3576(c)` / `_chain_free_rk3576(fd, c)` | Runs the chain. `in` is the first handle's row-major input, or NULL when it reads a cube from outside the chain. `out` is the last handle's row-major output, or NULL when it leaves a cube. The result is the one the handles give run one at a time. A redone program retries the whole chain, and `_chain_kicks` reports how many kicks the last run took |
| `rocket_conv2d_int8_chain_plan_rk3576(h, n, runs, max_runs)` / `rocket_conv2d_int8_programs_rk3576(h)` | The run finder: the maximal runs in an array of handles, for the constructor. A NULL entry breaks a run. A layer whose input the caller prepares on the host must not begin one. The finder splits a run past the program-count bound at a layer boundary. Pure. Returns the number of runs of two or more layers, which can exceed `max_runs`, and writes the first `max_runs`. `_programs` returns a handle's register-program count, or 0 if it cannot be chained |
| `rocket_conv2d_int8_chain_max_programs_rk3576()` | The longest program stream a chain accepts. The kernel's completion deadline sets it. A driver before interface 1.4 gives a batched job one task's deadline, and from 1.4 the deadline scales with the stream. `ROCKET_RK3576_CHAIN_MAX_TASKS` sets it |
| `rocket_chain_new_rk3576(fd, nodes, n)` | The node form of the constructor, so a pool or a placement layer can sit inside a run. A `rocket_chain_node_rk3576` sets exactly one of `conv`, `pool` and `placement`, and the constructor refuses a node with none or several. A pool or placement node can only be interior. A placement node emits no program: the caller states that nothing runs there, such as a concatenation of placed slices. The convolution-only entries are this form with every node a `conv` |
| `rocket_chain_plan_rk3576(nodes, n, runs, max_runs)` / `rocket_chain_node_programs_rk3576(node)` | The run finder and program count for nodes. A padded average pool whose divisor can lag (`rocket_pool_int8_rk3576_lag_can_show()`) is not a run node: the finder splits there, and the constructor refuses it. `ROCKET_RK3576_CHAIN_POOL_LAG=1` admits it. `_node_programs` returns 0 for a placement node, which emits none, and for a node that cannot be chained. The node's `placement` field tells the two apart |

#### Cube tails and row pitch

A direct handle's `rocket_conv2d_int8_cube_of_rk3576` declares its constant tail at the output
zero point. When a depthwise count is a multiple of 16 and not of 32, the pack adds a channel
group to the handle's own output. The depthwise handle then declares that tail. Otherwise a
depthwise handle declares one only when its output channels fill the last group exactly.

A packed-image stem that materializes its pad columns writes a plane wider than the caller's.
`rocket_conv2d_int8_cube_out_rk3576` and `_cube_of_rk3576` describe that output as a cube with a
row pitch, which only a pool reads. `rocket_conv2d_int8_cube_out_at_rk3576` refuses the handle.

## Pooling and spatial-reduction API (`rocket_pool.h`, `rocket_reduce.h`)

MaxPool, AveragePool and the global spatial reductions on the NPU's PPU pooling engine. The tensors are row-major, batch 1 and channels-major, as for `rocket_conv2d_fp16`: `[C][IH][IW]` in and `[C][OH][OW]` out. A pool is one job with no weights. The host scatters the input into cube layout and de-tiles the output. These entries emit the RK3588 encoding and refuse on the RK3576, whose int8 pool is in RK3576 convolution, pooling and activation.

| function | what it is |
|---|---|
| `rocket_pool_fp16(fd, d, in, out)` | MaxPool or AveragePool of fp16 `[C][IH][IW]` -> `[C][OH][OW]`, described by a `rocket_pool_desc`. A max pads out-of-range reads with -inf. An average multiplies each window sum by a per-axis `fp16(65536/k)` reciprocal, so it divides by kh·kw at every window. An output wider than one PPU task's allowance (`rocket_pool_fp16_max_ow`) runs as column bands, one task each in one job, each band scattered into its own cube. With `fd` < 0 the entry computes `rocket_pool_ref_fp16`. With `fd` >= 0 it returns `ROCKET_E_UNSUPPORTED` where `rocket_pool_npu_supported()` returns 0. Returns 0 or a negative value (`tests/pool_fp16_rocket.c`). Caution: a padded average differs from TFLite's AVERAGE_POOL_2D at the border windows, where TFLite divides by the valid count |
| `rocket_pool_fp16_plan(d)` / `rocket_pool_oh(d)` / `rocket_pool_ow(d)` / `rocket_pool_out_dim(in, k, stride, pad_lo, pad_hi)` | Pure checks and shape helpers. `_plan` checks a descriptor's field widths and its fit in one job, and returns 0, `ROCKET_E_SHAPE` for a malformed descriptor or `ROCKET_E_UNSUPPORTED` past the PPU's fields. It refuses `avg_exclude_pad` = 1, which only the RK3576 int8 entry accepts, an average kernel under 2, a kernel above 8 or a stride above the kernel on either axis, and an output needing more than `ROCKET_POOL_MAX_BANDS` (256) bands. `rocket_pool_out_dim` is `(in + pad_lo + pad_hi - k) / stride + 1`, rounded down. `_oh` and `_ow` apply it to a descriptor |
| `rocket_pool_fp16_max_ow(d)` / `rocket_pool_fp16_bands(d)` / `rocket_pool_fp16_band(d, s, &x0, &in_col, &band)` | The column bands of the RK3588 fp16 pool. A PPU task past its output-width allowance writes a complete, incorrect output and most windows also time out, so `_max_ow` returns the widest output one task computes for the window: 64, 32 or 16 as `ceil(kh/stride_y)` is 1-2, 3-4 or 5-8, 128 for a 2×2 window at stride 2, 256 for a one-row window to 1×3 at stride 1, and 0 for a kernel above 8 or a stride above the kernel. `_bands` is the band count `rocket_pool_fp16` runs (0 where it refuses), and `_band` describes band `s`: its first output and input columns and the descriptor it runs as. Pure |
| `rocket_pool_npu_supported()` | Whether the running kernel retires a job that ends in a pooling program. A pooling program raises no DPU completion, and a `rocket` driver that waits only for the DPU's retires such a job at its 500 ms job timeout with a core reset. Where it returns 0, `rocket_pool_fp16`, `rocket_pool_int8`, `rocket_pool_uint8` and `rocket_pool_int8_pack_rk3576` refuse, and the global pools compute on the host. The plans touch no hardware, so a frontend placing a pool asks both. Probed once and cached |
| `rocket_pool_int8(fd, d, in, out)` / `rocket_pool_uint8(fd, d, in, out)` | The int8 and uint8 forms, with the descriptor and layouts of `rocket_pool_fp16`, which they run through: every int8 and uint8 value is exact in fp16. `rocket_pool_uint8` subtracts 128 from each byte before the pool and adds it back after. A max is bit-exact. An average is the fp16 pool's, rounded to nearest and clamped, within one of the exactly rounded average, and needs a kernel of at least 2 (-9). With `fd` < 0 they compute `rocket_pool_ref_int8` (`tests/pool_int8_rocket.c`) |
| `rocket_pool_ref_fp16(d, in, out)` / `rocket_pool_ref_int8(d, in, out)` | The CPU references the tests compare against, and the `fd` < 0 path. The fp16 reference accumulates in fp32, divides an average by kh·kw and pads a max with -inf. The int8 reference is the fp16 reference of the same values, rounded to nearest and clamped. Host only |
| `rocket_global_avgpool_fp16(fd, C, H, W, in, out)` | GlobalAveragePool, Mean or ReduceMean over `[H,W]`: fp16 `[C][H][W]` -> fp16 `[C]`. One pass's window is at most 8×8, so the entry chains non-overlapping average pools whose divisors multiply to H·W, with the intermediates in cube layout. Each pass's output is inside its window's allowance (`rocket_pool_fp16_max_ow`). A shape that does not decompose, `fd` < 0, or a kernel where `rocket_pool_npu_supported()` returns 0 takes the exact host reduction instead. Each pass adds fp16 rounding and ~1e-3 of reciprocal quantization, so compare with a relative tolerance (`tests/reduce_mean_rocket.c`) |
| `rocket_global_maxpool_fp16(fd, C, H, W, in, out)` / `rocket_global_minpool_fp16(fd, C, H, W, in, out)` | GlobalMaxPool or GlobalMinPool, ReduceMax or ReduceMin over `[H,W]`. The decomposition, shape conditions, layout and return values are those of `rocket_global_avgpool_fp16`. No pass needs a reciprocal, so a decomposable shape is bit-exact against the host reduction (`tests/reduce_mean_rocket.c`) |
| `rocket_global_avgpool_plan(C, H, W)` / `rocket_global_pool_passes(C, H, W, kh, kw, cap)` / `rocket_reduce_factor_axis(n, f, cap)` | Pure shape checks for the three global pools. `_plan` returns 0 for a shape that runs on the NPU: both axes 8-smooth (every prime factor at most 7), an equal factor count, every pass inside its window's output-width allowance, and each axis at most `ROCKET_REDUCE_MAX_DIM` (8192). The passes run smallest window first, or largest first where that order puts a pass past its allowance (216: 8×8 first). A negative result, `ROCKET_E_TILING` for a shape the passes cannot decompose, means the entry computes on the host, not an error. `rocket_global_pool_passes` writes the pass windows in run order and returns their count. `rocket_reduce_factor_axis` writes up to `cap` factors in [2,8] whose product is `n`, in ascending order. It returns their count, 0 for `n`==1, or -1 when `n` is not 8-smooth |
| `rocket_global_avgpool_ref_fp16` / `rocket_global_maxpool_ref_fp16` / `rocket_global_minpool_ref_fp16` (each `(C, H, W, in, out)`) | The CPU references the tests compare against, and the `fd` < 0 and fallback path. The average accumulates in fp64. Host only |

## Activation API (`rocket_activation.h`)

The on-NPU nonlinear activations. Each is a standalone DPU **LUT** pass, NVDLA SDP with no
conv, that applies `f(x)` to a flat fp16 vector entirely on the NPU.

| function | what |
|---|---|
| `rocket_activation_fp16(fd, kind, in, out, n)` | elementwise fp16 activation on the NPU LUT. `SIGMOID`, `HARDSIGMOID` and `TANH` run in one pass, and `HARDSWISH`, `SILU` and `GELU` (exact erf) in two, `x·gate(x)`. Measured max_abs 0.00146 for sigmoid, 0.00049 for hardsigmoid and tanh, and 0.00098 for HardSwish, and GELU cos=1.000000 against the erf GELU over `[-12,12]`. `n` padded to a multiple of 8 (`tests/activation_lut_rocket.c`, `lut_tanh_rocket.c`, `gelu_rocket.c`) ([Vector length, GELU and the elementwise multiply](#vector-length-gelu-and-the-elementwise-multiply)) |
| `rocket_activation_fp16`, **positive-domain kinds** `SQRT` / `RSQRT` / `RECIPROCAL` | `f(x)` for `x>0` via the **shifted single-table** LUT (the whole domain maps onto the positive index half ⇒ no LE/LO glitch). The DPU LUT computes the reciprocal family. Uniform-grid ⇒ accuracy is domain-bounded: **<1% over a ~100-200x range** placed away from the steep near-0 region (HW: sqrt 0.85% / rsqrt 0.44% / recip 1.0%), set with `ROCKET_LUT_XLO/XHI`. `RSQRT` is the **RMSNorm/LayerNorm** core, and `RECIPROCAL` the softmax-denominator / Div core (`tests/recip_rsqrt_rocket.c`) |
| `rocket_activation_fp16`, **`EXP`** | `exp(x)` via the same shifted single-table, default domain `[-16,0]` (the **softmax** case: input <=0 after the row-max subtraction). Works on the standalone flying path (unlike GELU, with no LE/LO sign-mux glitch). exp's relative interp error is ~constant (`Δ²/8` ~1e-4). A table entry of exactly q=0 decodes to ~4.0, and exp's deep tail quantizes to q=0, so every shifted-table entry is clamped to q>=1 (`ROCKET_LUT_QFLOOR`, default 1, a no-op for sqrt/rsqrt/recip). softmax-sum end-to-end rel <=0.04% (`tests/exp_lut_rocket.c`) |
| `rocket_activation_fp16`, **`LOG`** | `ln(x)`, `x>0`, the natural inverse of EXP (log-probabilities / NLL / cross-entropy). Same shifted single-table, but the **first signed-output kind on the positive-domain path**: `log(x)<0` for `x<1`, so `out_lo=log(x_lo)` is negative and the OUT_CVT offset decodes the signed range (the tanh/ELU mechanism). Default domain `[0.25,32]`. Uniform-grid ⇒ **absolute** error is the metric (relative is ill-defined at the `x=1` zero crossing). HW **max_abs 0.0066 / mean 0.0007** over `[0.3,30]`, worst at the steep small-x end (`tests/recip_rsqrt_rocket.c`) |
| `rocket_leaky_relu_fp16(fd, alpha, in, out, n)` | **LeakyReLU** `x>=0?x:alpha*x` (YOLO, ONNX LeakyRelu/PRelu scalar slope), one DPU LUT pass on the natural LE/LO split (LE=`alpha*x`, LO=`x`). `alpha>0`, exact over [-R,R] (R via `ROCKET_LEAKY_R`, default 16), saturates beyond. The LUT's **sign-based LE/LO mux** spikes at exactly x~0 -> repaired on the host (the readback streams every element anyway). HW-validated alpha 0.01-0.5 (`tests/leaky_relu_rocket.c`) |
| `rocket_activation_fp16`, **`SOFTPLUS`** / **`MISH`** / **`ABS`** | **Softplus** `log(1+e^x)` (shifted single-table, `out_lo=0`, x~0-clean, max_rel 0.14%). **Mish** `x·tanh(softplus(x))`, the **YOLOv4/v7** backbone activation, 2-pass (a `[0,1]` gate LUT + EW-mul, max_rel 0.06%). **Abs** `\|x\|` (symmetric shifted single-table, the x=0 kink on the middle sample, max_abs 1e-3). All HW-validated (`tests/softplus_mish_rocket.c`) |
| `rocket_elu_fp16(fd, alpha, in, out, n)` / `rocket_selu_fp16(fd, in, out, n)` | **ELU** `x>=0?x:alpha*(e^x-1)` and **SELU** `λ·ELU_α` (fixed self-normalizing α=1.673, λ=1.0507), on the **symmetric shifted single-table**. Negative outputs ⇒ a signed OUT_CVT, which keeps the x~0 mux spike -> **host x~0 repair** (`ROCKET_ELU_NOREPAIR` disables). Exact over [-R,R] (`ROCKET_ELU_R`, default 8). HW-validated alpha 0.5-2.0 (`tests/elu_rocket.c`) |
| `rocket_ew_add_fp16` / `rocket_ew_sub_fp16` / `rocket_ew_mul_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise **add** (residuals) / **subtract** / **multiply** (gated activations), flat fp16 vectors. All use the conv-main EW path (identity-matmul main feed + ERDMA operand, `EW_OP_TYPE` add/mul). SUB reuses the ADD datapath with the operand negated (`a-b==a+(-b)`, exact fp16 sign flip). n is reshaped to [M,32] and M-tiled. **Bit-exact** vs host (`tests/ew_mul_rocket.c` runtime check sweeps add/sub/mul, n up to 40000 across the M-tile boundary) |
| `rocket_ew_div_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise **divide** `a/b` = `reciprocal(b)` (DPU LUT) then `a*recip` (EW mul). `b` positive, within the reciprocal LUT domain. fp16-LUT-approx (HW max_rel 0.35% for `b∈[0.5,8]`). Covers TFLite/ONNX `DIV` (`tests/recip_rsqrt_rocket.c`) |
| `rocket_ew_max_fp16` / `rocket_ew_min_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise two-tensor **max/min** `out=max/min(a,b)`. The DPU EW **ALU algo** field (`DPU_EW_CFG` bits[17:16]) reaches **MAX(0)/MIN(1)**, not just SUM(2=add): same conv-main EW datapath, one word changed. **Bit-exact**, because they select an operand (`tests/ew_minmax_rocket.c`, n to 40000). Covers TFLite/ONNX `Maximum`/`Minimum` and `ReLU=max(x,0)` |
| `rocket_clip_fp16(fd, lo, hi, in, out, n)` | **Clip** `min(max(x,lo),hi)` on the NPU (constant-operand MAX then MIN, two EW passes). Bit-exact. Covers TFLite/ONNX `Clip` and the bounded-ReLU family (ReLU6=`Clip(0,6)`) (`tests/ew_minmax_rocket.c`) |
| `rocket_prelu_fp16(fd, C, S, x, alpha, out)` | **PReLU** with a **per-channel** slope `alpha[C]` (YOLO/segmentation, ONNX PRelu), input `[C][S]`. **No LUT**, so no x~0 glitch. For the universal `alpha∈[0,1]` it is `max(x, alpha_c·x)` (per-channel scale via row-broadcast `ew_mul`, then `ew_max`), in 2 passes. Any `alpha` outside [0,1] falls back to `relu(x)+alpha_c·min(x,0)`. **Bit-exact** (`tests/prelu_rocket.c`) |
| `rocket_lut_epilogue_build(kind, lut, ep)` | builds the LE/LO tables + DPU epilogue constants for a smooth single-pass kind (SiLU/tanh/GELU). The **single source** of the single-pass LUT params, shared by the standalone op and the **conv->activation fusion** (`rocket_conv2d_act_fp16`), so the two cannot diverge |
| `rocket_ew_mul_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise fp16 multiply `out=a*b` (identity-conv main feed + `EW_OP_TYPE=1`, M-tiled). Bit-exact (`tests/ew_mul_rocket.c`). The building block of on-NPU HardSwish/SiLU |
| `rocket_activation_ref_fp16` / `rocket_leaky_relu_ref_fp16` / `rocket_elu_ref_fp16` / `rocket_selu_ref_fp16` / `rocket_prelu_ref_fp16` | The CPU references, fp16-rounded, for tests. `tests/activation_lut_rocket.c`, `leaky_relu_rocket.c` and `prelu_rocket.c` compare against the first, second and last, and `tests/elu_rocket.c` against its own double-precision model. Host only |
| `rocket_activation_name(kind)` | A kind's name, such as `"sigmoid"`, or `"?"` for an unknown kind |
| `gen_lut_activation_fp16` (`npu_activation.h`) | low-level: the flying-mode DPU LUT regcmd (LE/LO hybrid tables, BN-mul index, OUT_CVT Q0.15->fp16). The same epilogue is fused into `gen_conv2d_fp16` (`lut_epilogue_t`, `conv_params_t.act`) for conv->activation |

### Vector length, GELU and the elementwise multiply

A LUT op lays the vector out as a cube of `cols = n/8` width positions, and
`DPU_DATA_CUBE_WIDTH` is 13-bit, so one op is limited to `n <= 65528`. `run_dpu_lut` tiles
automatically, so every activation works at any `n`, including a transformer's `[M,I]` cube of
millions of elements. Running at the exact maximum width (`cols = 8191`) corrupts ~54 cube
positions, so the tile cap (32768) stays well under that maximum and is bit-clean.

The two-pass kinds run sigmoid (SiLU), hardsigmoid (HardSwish) or Φ (GELU) on the NPU LUT,
then multiply the result by x. The multiply runs on the host by default, and fully on the NPU
with `ROCKET_ACT_NPU_MUL=1`.

**GELU** runs the accurate **2-pass** `x·Φ(x)`, where Φ is the Gaussian CDF on the clean
unit-LUT geometry. The single-pass GELU spikes ~128 in the flat negative tail, which also makes
a *fused* single-pass matmul->GELU spike for wide FFN inputs. The 2-pass form is the on-NPU
GELU path, and `ROCKET_ACT_WIDE_LUT` selects the single pass, for reverse engineering only.

A fully-on-NPU elementwise multiply (`rocket_ew_mul_fp16`) is implemented and HW-validated
bit-exact (`tests/ew_mul_rocket.c`). It feeds the DPU's main operand from an **identity
conv** and sets the EW op to multiply (`EW_OP_TYPE=1`, `DPU_EW_CFG=0x108003C4`). The program
is the fp16 K-accum eltwise-add with one register field changed, `gen_gemm_fp16`'s `ew_mul`
flag. With it, **HardSwish/SiLU run fully on the NPU** under `ROCKET_ACT_NPU_MUL=1`, as a LUT
gate plus an EW mul with no host arithmetic. The host multiply stays the default, since a
standalone EW-mul costs a second NPU round-trip (fusing the mul into the producing conv is the
faster path).

A two-buffer op needs no conv main feed. A DPU-only program, both operands read from memory,
computes fp16 ADD, MAX, MIN, MINUS and MUL exactly on every element on the RK3588
(`tests/ew_int_probe`). Software must program three fields correctly:

- `DPU_DATA_CUBE_CHANNEL`'s `ORIG_CHANNEL` equal to its `CHANNEL`, or only lane 0 of each atom
  is written
- `FEATURE_MODE_CFG`'s `COMB_USE` with bit 0 clear, or the job never completes
- At fp16, the EW input converter live for ADD and bypassed for MUL

No library entry uses the DPU-only form.

## Transformer-block API (`rocket_reduce.h`, `rocket_norm.h`, `rocket_ffn.h`, `rocket_softmax.h`, `rocket_attn.h`, `rocket_encoder.h`)

On-NPU transformer primitives, built by composing the matmul / reduce / LUT / EW pieces:
the substrate for fused FFN / encoder blocks. Each has a CTest test against an fp64 oracle.
Together they run a full Whisper/transformer encoder block on the NPU (cos = 1.000000).

| function | what it is |
|---|---|
| `rocket_reduce_feature_fp16(fd, M, H, in, out, mean)` | **feature-axis reduce** `sum_h x[m,h]` (or mean), per row -> fp32 `[M]`. The contraction RMSNorm/LayerNorm/softmax need, and the one the PPU **cannot** give (it pools spatial `[H,W]` *within* a channel, never across). Realized as a **ones-vector matmul** reusing `rocket_matmul_fp16_f32out` (no new regcmd, genuine fp32 K-accum). Essentially **bit-exact** (max_rel <= 1.4e-7) |
| `rocket_cumsum_fp16(fd, M, N, in, out, exclusive, reverse)` | **cumsum / prefix sum** along the last axis (TFLite/ONNX `CumSum`). The feature reduce **widened from a single ones column to a full triangular ones matrix**: `out = in·Lᵀ`, `L[n][k]=1` iff column `k` is in prefix `n` (incl/excl x forward/reverse pick the triangle). Same `rocket_matmul_fp16_f32out` reuse (no new regcmd, fp32 K-accum for long prefixes). HW **bit-exact** (max_abs 0 across all variants incl. T=1500) (`tests/cumsum_rocket.c`) |
| `rocket_rmsnorm_fp16(fd, M, H, x, weight, eps, out)` | **RMSNorm** `x/sqrt(mean_h(x²)+eps)·weight[h]`. O(M·H) on the NPU (square->reduce->scale). The O(M) per-row rsqrt tail is **exact on the host** (sending M scalars to the DPU rsqrt LUT would add a round-trip *and* hit the LUT domain). fp16-square overflow (`\|x\|>256`) -> exact power-of-2 prescale. `weight` is the effective scale (Gemma: pass `1+w`). max_rel <= 3.5e-3 |
| `rocket_layernorm_fp16(fd, M, H, x, gamma, beta, eps, out)` | **LayerNorm** `(x−mean)/sqrt(var+eps)·gamma[h]+beta[h]` (the Whisper/encoder norm). **Both reductions share one feature reduce by stacking rows**: `A=[x ; x⊙x]` (2M rows) under the ones weight -> first M = `sum(x)`, next M = `sum(x²)`. Host O(M) mean/var/rsqrt. The affine folds to `x⊙A+B` (one ew_mul + one ew_add). Same fp16-square overflow prescale as RMSNorm. `beta` can be NULL (`tests/layernorm_rocket.c`) |
| `rocket_scale_rows_fp16(fd, M, N, in, r, out)` | **per-row broadcast multiply** `out[m,n]=in[m,n]·r[m]` (`r` fp32 `[M]`), the FFN/attention/softmax post-scale (the RMSNorm 1/rms folds here, and the weight folds into the next matmul). The reusable building block |
| `rocket_geglu_fp16(fd, gate, up, kind, prod, n)` | **gated activation** (GeGLU/SwiGLU core) `prod = act(gate)⊙up`, the only computation an FFN adds beyond matmul. `kind` = `SILU` (robust 2-pass) / `GELU`. Fully on the NPU (LUT + EW-mul) |
| `rocket_ffn_fp16(fd, M, H, I, x, Wg, Wu, Wd, kind, out)` | **gated FFN block** `gate=x·Wgᵀ -> act(gate)⊙(x·Wuᵀ) -> ·Wdᵀ`. Composes the three projections + the geglu core. **cos = 1.000000** vs fp64 (Gemma-ish 128×2048×1024). It hands each intermediate back to the host between ops. `rocket_ffn_fp16_fused` computes the same block with the intermediates kept in cube layout on the NPU |
| `rocket_ffn_fp16_fused(fd, M, H, I, x, Wg, Wu, Wd, kind, out)` | the gated FFN block, cube-resident: the signature and block of `rocket_ffn_fp16`, with the `[M,I]` intermediates kept in cube layout on the NPU. fp16 only, `H` and `I` multiples of 32, `M` of 4. Tested at cos >= 0.999 against fp64 and >= 0.9995 against `rocket_ffn_fp16`, every output written (`tests/ffn_fused_rocket.c`) ([Cube-resident FFNs](#cube-resident-ffns)) |
| `rocket_mlp_fp16_fused(fd, M, Din, Dff, Dout, x, W1, b1, W2, kind, out)` | the non-gated encoder FFN, cube-resident: `out = act(x·W1ᵀ + b1)·W2ᵀ`, with the `[M,Dff]` intermediate kept in cube layout on the NPU. fp16 only, `Din` and `Dff` multiples of 32, `Dout` of 16, `M` of 4. It is `rocket_encoder_block_fp16`'s default FFN and is tested through that block (`tests/encoder_block_rocket.c`) ([Cube-resident FFNs](#cube-resident-ffns)) |
| `rocket_softmax_fp16(fd, M, N, in, out)` | **row-wise softmax** over the last axis. Host row-max + subtract -> NPU `exp` (LUT) -> NPU row-sum (feature reduce) -> host `1/s` -> NPU per-row scale. The **row-max is on the host**, because matmul and reduce can only sum, and the only on-NPU max is the PPU max-pool, the resident-fusion path. Validated to T=1500 (Whisper seq), rows sum to 1±5e-4 (`tests/softmax_rocket.c`) |
| `rocket_logsoftmax_fp16(fd, M, N, in, out)` | **row-wise LogSoftmax** `x − logsumexp(x)` (the classification / NLL-loss head, LM log-probs). Shares softmax steps 1-3 (host row-max+subtract -> NPU `exp` -> NPU row-sum), then a per-row **subtract** instead of a divide: host `ls=log(s)` (exact, O(M), like softmax's `1/s`, and **not** the DPU LOG LUT, which is for large-tensor log) -> NPU per-row `ew_sub`. All-additive ⇒ better-conditioned than softmax (no tiny-prob blow-up). HW max_abs <= 0.031, `Σexp(out)=1` (`tests/softmax_rocket.c`) |
| `rocket_cross_entropy_fp16(fd, M, N, logits, target, loss)` | **stable per-row cross-entropy** `CE[m]=logsumexp(logits[m]) − logits[m][target[m]]` = `−logsoftmax[target]` (softmax-classifier / LM NLL loss). The on-NPU logsumexp reduction (host row-max -> NPU `exp` -> NPU fp32 row-sum -> host `log(s)`) + a **host gather** of the target logit (there is **no HW gather**, so M scalar lookups, like the host `1/s`). Never materializes softmax (no divide). CE >= 0. **fp32-grade** (max_abs <= 1.5e-4, since the loss never round-trips through fp16 output storage) (`tests/cross_entropy_rocket.c`) |
| `rocket_mha_self_fp16(fd, T, d, n_head, x, Wq,bq, Wk,bk, Wv,bv, Wo,bo, out)` | **multi-head self-attention** (the pure attention sublayer): QKV proj + per-head `scale·(q·kᵀ)` + softmax + `P·v` + out-proj, **all matmuls + every softmax on the NPU**. Weights row-major `[out,in]`. Biases optional (Whisper: bq,bv,bo, no bk). The **key count is padded to %32 and the pad score columns masked to −30000 before softmax** (the matmul rejects unaligned N/K, and Whisper T=1500 is unaligned). cos = 1.000000 at Whisper-base d=512/8-head incl. T%16≠0 (`tests/mha_rocket.c`) |
| `rocket_flash_attn_fp16(fd, n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads, scale, softcap, Q, K, V, mask, out)` / `_mt(…, nthreads)` / `rocket_fa_ctx_create(nthreads)` / `_free` / `rocket_flash_attn_fp16_ctx` | **masked grouped-query (flash) attention**, the decoder and LLM-prefill sublayer (llama.cpp's `FLASH_ATTN_EXT`). Takes projected Q, K and V and an **additive mask the caller supplies**, and computes per head `scale·(q·kᵀ)` -> optional `softcap·tanh` -> `+mask` -> softmax -> `P·v`, both matmuls on the NPU. GQA and MQA. cos = 1.000000 against an fp64 oracle at Gemma-4-12B shapes (`tests/flash_attn_rocket.c`). Returns `ROCKET_E_RANGE` when an fp16 score saturates. `_mt` and the context form are numerically identical. [Flash attention](#flash-attention) has the layouts and options |
| `rocket_encoder_block_fp16(fd, T, d, n_head, d_ff, x, ln1…, Wq…Wo…, ln2…, Wf1…Wf2…, eps, out)` | one full Whisper/transformer encoder block (pre-norm): `x += MHA(LN1(x)); x += MLP(LN2(x))`. On the NPU: both LayerNorms, the attention matmuls and softmax, both residual adds, the two FFN projection matmuls and the Φ-gate DPU LUT of the FFN's 2-pass GELU `x·Φ(x)`. cos = 1.000000 against an fp64 block oracle (`tests/encoder_block_rocket.c`). [Cube-resident FFNs](#cube-resident-ffns) has where its FFN runs |
| `rocket_siglip_load(path, m)` / `rocket_siglip_free(m)` (`rocket_siglip.h`) | Maps and validates a flat fp16 weight blob from `tools/siglip_extract.py` and points the `rocket_siglip_model` at each tensor, or unmaps it. Returns 0, or -1 for open, -2 for mmap or size, -3 for a bad magic or version and -4 for inconsistent dimensions |
| `rocket_siglip_encode(fd, m, pixels, out, hidden)` / `rocket_siglip_ctx_create(m, nthreads)` / `rocket_siglip_encode_ctx(c, …)` / `rocket_siglip_ctx_free(c)` (`rocket_siglip.h`) | the SigLIP-B/16 vision encoder end-to-end (SmolVLM-256M front-end): patch-embed (im2col -> matmul), position add, 12 `rocket_encoder_block_fp16` blocks `(L=1024, d=768, 12 heads, d_ff=3072)` and post-LayerNorm. A per-call form, and a resident form (`_ctx`) with the static GEMMs prepacked once across the cores, one caller per context. Per-layer cosine 0.999998 against an fp32 reference (`tests/siglip_rocket.c`) ([SigLIP encoder](#siglip-encoder)) |
| `rocket_sam_load(path, m)` / `rocket_sam_free(m)` (`rocket_sam.h`) | Maps and validates a flat fp16 weight blob from `tools/sam_extract.py` and points the `rocket_sam_model` at each tensor, or unmaps it. Returns 0, or -1 for open, -2 for mmap or size, -3 for a bad magic or version and -4 for inconsistent dimensions. A frontend can fill the struct itself and leave `map` a non-NULL sentinel it is responsible for. Such a frontend must never call `rocket_sam_free` |
| `rocket_sam_encode(fd, m, pixels_chw, out, hidden_opt)` (`rocket_sam.h`) | The SAM ViT-Det image encoder (facebook/sam-vit-base) end to end: patch embed, position add, blocks with windowed or global attention and a decomposed relative-position bias, then the conv neck. `pixels_chw` is fp16 `[ic][image_size][image_size]`, and `out` is fp16 `[neck_out][grid][grid]`, the ONNX `last_hidden_state`. The large GEMMs and each window's per-head QKᵀ and P·V run on the NPU. The LayerNorms, GELU, softmax and relative-position bias run on the host. The call packs the weights again each time. Returns 0 or a negative value |
| `rocket_sam_encode`, `hidden_opt` and `fd` < 0 | A non-NULL `hidden_opt` receives the `n_layers+1` token grids, `[(n_layers+1)][grid*grid][d]` fp16: the embeddings, then each layer's output. `tests/sam_rocket.c` scores each layer's cosine similarity from them. With `fd` < 0 the encoder runs the exact host reference, a datapath self-check that needs no device |
| `rocket_sam_ctx_create(m, nthreads)` / `rocket_sam_encode_ctx(c, pixels_chw, out, hidden_opt)` / `rocket_sam_ctx_free(c)` (`rocket_sam.h`) | The resident encoder. `_ctx_create` packs the static GEMM weights once: the patch projection, each layer's qkv, o, fc1 and fc2, and the neck's first conv. Each image then packs only its activations. The result matches `rocket_sam_encode` within fp16. `_ctx_create` returns NULL on failure |
| `rocket_modernbert_ctx_create(m, nthreads)` / `rocket_modernbert_encode(c, B, T, len, emb, post_bias, out, hidden)` / `rocket_modernbert_ctx_free(c)` (`rocket_modernbert.h`) | a ModernBERT text encoder end-to-end, plus an optional stack of torch pre-norm layers after it (a decision or classification head). B right-padded sequences each run at their own length. `nthreads` 0 is a host mode for off-device checks. Worst layer cosine 0.9999936 against ONNX Runtime's fp32 CPU provider on the Laya decision model, post-stack output 0.99999995 (`tests/modernbert_rocket.c`) ([ModernBERT encoder](#modernbert-encoder)) |
| `rocket_reduce_feature_ref_fp16` / `rocket_cumsum_ref_fp16` / `rocket_rmsnorm_ref_fp16` / `rocket_layernorm_ref_fp16` / `rocket_scale_rows_ref_fp16` / `rocket_geglu_ref_fp16` / `rocket_ffn_ref_fp16` | The host references of the reduce, norm and FFN entries, each the `fd` < 0 path of its entry, which the entries' tests compare against. The reduce and cumsum references accumulate in fp64. Host only |
| `rocket_softmax_ref_fp16` / `rocket_logsoftmax_ref_fp16` / `rocket_cross_entropy_ref_fp16` / `rocket_mha_self_ref_fp16` / `rocket_flash_attn_ref_fp16` / `rocket_encoder_block_ref_fp16` | The exact host references of the softmax, attention and encoder entries, each the `fd` < 0 path of its entry, which the entries' tests compare against. The cross-entropy reference is fp64. Host only |

### Cube-resident FFNs

`rocket_ffn_fp16_fused` packs only `x` on the host and reads back only `out`. The down
matmul's K tiling is pinned to the projections' N tiling. That grouping of the fp16
K-accumulation is the only numerical difference from `rocket_ffn_fp16`. A projection output
past one NPU batch (`nMt·nNt > 64`), or a tiling that cannot be matched, falls back to
`rocket_ffn_fp16`.

`rocket_mlp_fp16_fused` adds `b1` (or NULL) on the cube and runs the activation elementwise over
the cube bytes. The fc2 matmul reads the activated cube, so the host packs only `x` and reads
back only `out`. A fc1 output past one NPU batch, or a tiling that cannot be matched, falls back
to a host-handoff FFN.

By default `rocket_encoder_block_fp16` runs its FFN through `rocket_mlp_fp16_fused`, which adds
`bf1` on the NPU and multiplies `x·Φ(x)` on the host unless `ROCKET_ACT_NPU_MUL=1`.
`ROCKET_ENCODER_NOFUSE` (any value) runs separate calls instead, with `bf1` on the host and the
multiply on the DPU EW unit.

### Flash attention

`rocket_flash_attn_fp16` computes every head of one call. Head-major dense layouts make each
head a contiguous matmul operand. With `n_head > n_kv_heads`, query head `h` reads kv head
`h/(n_head/n_kv_heads)`. `n_tokens` is padded to a multiple of 4 and `n_kv` to a multiple of
32, and the pad keys score −∞. The mask is the caller's: the entry does not synthesize a causal
or sliding-window mask. The oracle test covers head_dim 256, 16 query heads, 8 kv heads local
and 1 global, a sliding window and a soft-cap.

The scale, soft-cap, mask and softmax run as one fp32 host pass per row by default.
`ROCKET_ATTN_HOST_SOFTMAX=0` runs them on the NPU. The variable reads the same on the encoder
MHA path, where the NPU is the default and `=1` selects the host.

> [!CAUTION]
> The scores are fp16 and saturate. A score `scale·q·k` past 65504 is written as ±65504, not
> inf. The keys past it tie, and the softmax returns finite, wrong values [HW sweep
> 2026-10-09, RK1, ~816 MHz]. Every path checks the raw scores in its host mask pass. It
> returns `ROCKET_E_RANGE` when a real key reaches the limit, and `out` is then not valid. A
> caller holding the operands at higher precision can recompute.

`rocket_flash_attn_fp16_mt(…, nthreads)` fans the heads across worker fds, and the cores run
head ranges in parallel. `rocket_fa_ctx` with `rocket_flash_attn_fp16_ctx` is the persistent
form for repeated calls: the worker fds stay open and each worker's score scratch stays
resident. All three forms are numerically identical.

`ROCKET_FA_CHAIN` (on by default) batches each worker's per-head QK matmuls into one NPU job
and its AV matmuls into a second. The mask and softmax run between the two, so the
host-softmax default holds. A head range then costs one submit and fence instead of one per
head. The jobs go through a per-worker resident batched-matmul context, `rocket_mm_batch`,
whose BOs and score scratch are held and prezeroed once. `ROCKET_FA_CHAIN=0` forces the
per-head path.

Chaining cuts the small-GEMM fixed dispatch overhead. It nearly doubles the flash-attention
throughput against a per-call batch. Against the per-head path it is ~2.0-3.8x faster in
head-range time (186->50 ms @512, 363->183 ms @2K). The crossover where the NPU beats the CPU
at prefill moves from ~6K to ~2K. Chained and per-head outputs are identical (cos = 1.000000).

`ROCKET_FA_CHAIN_ELEMS` (default 32M score elements) bounds the chained head group. A worker's
~3-head range then stays batched up to ~20K context, at ~150-200 MB/worker scratch. Collapsing
the per-head submits there gives **1.10x@4K / 1.47x@8K / 1.32x@16K / 1.16x@32K** at a 512-token
ubatch (`fabench`, T=512, [HW sweep]). End to end it gives +3% at pp8192 (Qwen3.5-0.8B-F16),
and nothing at short context, which is already batched to 2K. At a 2048-token ubatch each
head's score alone exceeds the budget, and the gain shrinks to ~1.05x. Raise
`ROCKET_FA_CHAIN_ELEMS` to chain further, at a higher scratch cost.

`ROCKET_FA_ROWMAJOR` (on by default) hands the scores and P between the chained QK and AV jobs
as rows in a resident BO. The scores are then not de-tiled out of the NPU's cube layout, and P
is not packed back into it. Plain rows serve to a 4 KiB row (`ROCKET_FA_RM_MAX_ROW`), with P
rewritten in place. Past that, QK writes one block of rows per output tile, because a wider
row-major write costs the device more. The handoff needs the persistent `rocket_fa_ctx` path,
`ROCKET_FA_CHAIN` and the fused host softmax (`ROCKET_FA_FUSED_SOFTMAX`). QK must also plan the
head dimension in one K tile, or the cube handoff runs.

The row handoff takes 0.64-0.69x the cube handoff's time at a 1370-token ViT layer. At a
512-query prefill shape from 2048 to 32768 keys it takes 0.73-0.82x, with the output
bit-identical. Both were measured on the RK1 at ~816 MHz [HW sweep 2026-10-11]. The one
exception is a head that runs alone in its chained group. It keeps fp32 host K-accumulation for AV, where the cube handoff's
one-shot matmul accumulates in fp16 on the device. The two differ by a few fp16 ulps.

`ROCKET_FA_TILE_KV` (default off) is the opt-in tiled long-context variant. It walks the key
axis in `ROCKET_FA_TILE_KV`-wide tiles that keep the FlashAttention-2 running softmax (fp32
max, denominator and output). The working score tile is then `[Tp,tile]` rather than the full
`[Tp,n_kv]` (32 MB/head at 32K). It engages above `ROCKET_FA_TILE_MIN_KV` (default 8192).

The tiled variant is bit-faithful (fp32 accumulation, cos = 1.000000) but slower on this
dispatch-bound NPU, with more and smaller per-tile submits. Its speed converges to the
materialized path's from below (0.58x@2K-tile -> 0.93x@16K-tile at n_kv 32K). It bounds the
flash-attention scratch at extreme context and does not add speed.

Standalone, these ops are submit-bound, and the host is faster for an isolated norm. Their
value is in composition: keeping the activation cube-resident between two NPU matmuls skips
the de-tile->host->re-pack round-trip.

### SigLIP encoder

The resident SigLIP path runs the 12 attention heads across the worker fds through
`rocket_flash_attn_fp16_ctx`, as unmasked MHA. The heads are independent, so one DRM scheduling
entity per fd dispatches head ranges across the NPU cores, with head-chaining and resident
scratch. The fan-out is 1.44-1.51x warm against a single-stream per-head loop, which
`ROCKET_SIGLIP_FA=0` selects.

The host GELU uses a bit-exact fp16->fp16 LUT: all 65536 fp16 outputs fit one 128 KB table,
identical to the scalar `tanhf`. It is a further 1.22x, for 1.78x warm combined, with the
cosine unchanged. `ROCKET_SIGLIP_GELU_SCALAR=1` selects the scalar form.

The context allocates the per-image host scratch, ~17 MiB at Depth Anything's ViT-S, so an
encode does no heap traffic. The bias adds and LayerNorm are vectorized, LayerNorm in float
with per-lane row sums. Depth Anything v2 Small through ort-rocket runs in 0.90-0.95x the
wall-clock time of the unvectorized bias adds and LayerNorm [HW sweep 2026-10-09, RK1,
`ondemand`]. `ROCKET_SIGLIP_PROF=1` prints the phases, the bias adds' own bucket and the call's
minor faults.

### ModernBERT encoder

`rocket_modernbert_encode` has RoPE at a per-layer theta, global and sliding
(`|i-j| <= window`) layers, GeGLU and bias-less LayerNorms. Every projection runs once over the
live rows of the B sequences stacked into one M, and attention runs per sequence, so padding
costs nothing. The residual stream stays fp32 on the host, because ModernBERT-large's grows
past 29000. Each FFN down-projection's input is scaled by a power of two into fp16's range and
back.
Sliding layers past `2T + 2 window` tokens run banded (`ROCKET_MB_BAND_TILE`). The test's
artifacts come from `tools/modernbert_extract.py`.

## Vision-normalization API (`rocket_normvision.h`)

The **vision** normalization family uses the same primitives as the transformer norms. The
feature-axis reduce computes the per-group mean and variance, and the DPU elementwise path
applies the affine.

The four ops differ only in **which axis is reduced** and **how the affine broadcasts**. All
take channels-major NCHW-style buffers with `P = H*W`, the spatial count per channel, where
`P=1` is a pure `[N,C]` tensor. Each has a CTest test (`tests/norm_vision_rocket.c`) against an
fp64 oracle.

| function | what it is |
|---|---|
| `rocket_batchnorm_fp16(fd, N, C, P, x, gamma, beta, mean, var, eps, out)` | **BatchNorm (inference)** `(x−mean[c])/sqrt(var[c]+eps)·gamma[c]+beta[c]`. **No reduction** (inference BN uses the stored running `mean`/`var` `[C]`) -> a per-channel affine folded to `x·s[c]+b[c]` (one NPU ew_mul + one ew_add over the broadcast tensors). `gamma`/`beta` `[C]` can be NULL (⇒ 1 / 0) |
| `rocket_groupnorm_fp16(fd, N, C, G, P, x, gamma, beta, eps, out)` | **GroupNorm**: normalize each `(n, group)` over its `C/G` channels x `P` spatial. A group's elements are contiguous in `[N,C,P]`, so the per-`(n,g)` reduce is a `[N·G, (C/G)·P]` **stacked** feature reduce (`[x ; x⊙x]` in one reduce, like LayerNorm). The affine is **per-channel** (varies within a group) -> full broadcast `x·A+B`. `C%G==0`. `gamma`/`beta` `[C]` (NULL ⇒ 1/0). `G=1` = LayerNorm-over-CHW |
| `rocket_instancenorm_fp16(fd, N, C, P, x, gamma, beta, eps, out)` | **InstanceNorm** = GroupNorm with `G=C` (normalize each `(n,c)` over its `P` spatial positions). Per-row affine |
| `rocket_l2norm_fp16(fd, M, H, x, eps, out)` | **L2-Normalize** each row over `H` (TFLite `L2_NORMALIZATION`, ONNX `LpNormalization` p=2): `x/sqrt(sum_h x²+eps)`. `sq=x⊙x` (NPU) -> `ss=sum_h sq` (NPU fp32 reduce) -> host `1/sqrt` -> per-row scale (NPU) |
| `rocket_batchnorm_ref_fp16` / `rocket_groupnorm_ref_fp16` / `rocket_instancenorm_ref_fp16` / `rocket_l2norm_ref_fp16` | The exact fp64 host references, the value each entry's `fd` < 0 path returns, which `tests/norm_vision_rocket.c` compares against. Host only |

All four use the RMSNorm/LayerNorm fp16-square **overflow prescale** (`|x|>~223` ⇒ x² overflows
fp16 -> exact power-of-2 prescale, recovered as `·4^k`), accumulate the reduce in fp32, and do the
O(rows) mean/var/rsqrt tail exact on the host. HW-validated bit-faithful (`max_abs` = fp16 affine
rounding) across the row-tile boundary, `C%32≠0`, `P=1`, every group count, and the large-magnitude
prescale path. Like the transformer norms they are submit-bound standalone. Their value is op
coverage (a delegate need not send the node to the CPU) and a base for cube-resident fusion.

## Datatype matrix

The native datatype matrix is **complete**: a working, validated matmul for every
native dtype (the per-dtype alignment atoms are listed in the matmul API above).

| dtype | support | use |
|---|---|---|
| fp16 -> fp32 | native, validated | the core path, coherent Gemma-4-12B prefill |
| int8 -> int32 | native, bit-exact | W8A8 (+Hadamard) = char-identical to fp16, for RAM rather than speed |
| int4 -> int16 | native, bit-exact | W4A4, for RAM, can reach single-pass K |
| bf16 -> fp32 | native, validated | fp32 range with no activation scaling, token-identical to fp16 |
| tf32 -> fp32 | native, validated | first 4-byte-input path, for coverage (half-rate) |
| int16 -> int32 | native single task, **saturating**, no tiled entry | int64-exact via int8 byte-decomposition |

**int16's int32 output saturates.** `gen_gemm_int16` writes the whole int32 output in one
task. It is bit-exact against the int64 dot product clamped to int32 (`int16_native_probe`,
`matmul_int16_rocket`). It is therefore exact only while the operand ranges keep each sum
inside int32. No tiled entry exposes it. `rocket_matmul_plan_int16` previews its tile shape
without hardware, and the bf16 path reuses that plan.

An int64-exact int16 matmul over full-range operands is the int8 byte-decomposition,
`rocket_matmul_int16_exact`: four int8 matmuls, recombined in int64. The transposed writer
(`tp_org_en`, N<=32) keeps the low 16 bits of each sum.

The two **float** datatypes, bf16 and tf32, share the int16/fp16 fp32-out writer, with host
`double` K-accum and no saturation.

**bf16** has fp32's 8-bit exponent at fp16's 2-byte cost, so it needs no per-row
activation scaling, and it ran Gemma-4-12B token-identical to fp16.

**tf32** is the first 4-byte-input path (feature cube C2=4, weight `(N/16,K/16,16,16)`), where
the K-group *halves* to 16 for a 4-byte element. It has a genuine 10-bit mantissa, and the
hardware truncates each fp32 operand to it, dropping the low 13 bits without rounding. It
tracks a reference whose inputs are truncated the same way to ~1e-7, and differs from full
fp32 by the 10-bit gap. A caller that wants round-to-nearest must round on the host first. tf32 has the least practical use of the datatypes: "256×3 MAC/cycle" is half bf16's
rate, and bf16 already gives fp32 range at full speed. tf32 provides coverage rather than
speed.

## Tests

Each test under `tests/` is a standalone executable that links the library, and most run
on the NPU. The tests double as the regression and bring-up checks.

The correctness tests, including the bit-exact `int8`/`int4`/`int16`/`bf16`/`tf32`
tiled-or-resident matmul paths and the int8 depthwise conv, are registered with CTest.
Run `ctest` from the build directory after `cmake --build`. Perf probes and
reverse-engineering sweeps are built but left unregistered.

Every registered test follows one exit-code contract:

| Exit | Means |
|---|---|
| 0 | A numeric check ran and passed |
| 1 | A wrong answer, a declined path, a usage error, a host allocation failure |
| 2 | The device, or the SoC the test needs, is absent. CTest reports a skip |

A path that declines is a failure, not a skip. A test built to assert a refusal says which
stage refuses, and passes only when that stage does. Off-device the NPU tests skip, and the
host-only ones (`bytes_moved_rocket`, `chain_layout_rocket`, `chain_verify_gate`,
`matmul_plan_gate`, `matmul_plan_gate_rk3576`, `cube_transpose_gate`, `regcmd_rk3576_gate`,
`pool_rk3576_bracket_gate`, `pool_stock_kernel_gate`) still run and must pass. A skip is visible in the summary, so a
board run that skips a device test has not checked it.

The microbenchmarks written in AArch64 NEON intrinsics (`softmax_exp_bench`) are built only
where those intrinsics compile, so a host without them still builds every correctness test.

Each NPU test is registered with `RESOURCE_LOCK npu`, so `ctest -j` never has two in flight.
Each has a `TIMEOUT` of 300 s, or 600 s for flash attention and 900 s for the ViT encoders.

A hung job fails no syscall. The `rocket` driver retires a job that runs 500 ms and signals
its fence as if it had completed. A test that scores the untouched output BO can then pass.

Two fixtures bracket the NPU tests to catch it. `npu_klog_mark` saves a kernel-journal cursor
before the first one. `npu_klog_check` then fails the run on any `NPU job timed out`,
`rk_iommu` or NPU-side `WARNING` line after it. Under the rknpu-submit provider it also fails
on the `rknpu` driver's `job timeout`, `job abort` and `failed to wait job` lines.

The journal is read directly, which the `adm` or `systemd-journal` group allows, or through
`sudo -n`. With neither, `npu_klog_check` reports a skip. Inside a test, a fenced wait past the
slow-wait mark logs a warning and counts in `rocket_fence_wait_slow_count()`. The registration
fails the test that made it.

On the RK3576 the entries' write guards also time each job from before its submit to its fence.
A job that took the driver's 125 ms backstop is redone, whatever its output contains.
`rocket_rk3576_retired_counts()` counts those, split by whether the write check alone passed
the output.

| test | purpose |
|---|---|
| `matmul_correctness_matrix_rocket` | **the layout/readback correctness test**: realistic random inputs + **cosine-similarity** validation (catches silent layout/scatter/readback corruption that the exact-integer tests miss), dtype-aware (fp16/int8/int4/int16/bf16/tf32), M%4≠0 via padding, K>8192, all entry points. A declined path fails unless `EXPECT_DECLINE=pack` or `=path` names the stage that must refuse. Driver: `tests/correctness_matrix.sh`, the full M/K/N x dtype x path matrix, which asserts each documented M=1 refusal (the fp16 streaming and resident entries, resident int8 and int4) at its stage |
| `matmul_tiled_rocket` | tiled fp16 matmul + profiling/sweep options (the workhorse) |
| `matmul_mt_rocket` | multicore correctness + scaling |
| `matmul_prepacked_rocket` | resident-weights path vs mt vs CPU ref |
| `matmul_prepacked_crossm_rocket` | one resident weight reused bit-exact across compatible M (pack@512 -> run@256/512/768). Incompatible small-M rejected |
| `matmul_prepacked_canonical_rocket` | `ROCKET_CTX_TILING_CANONICAL`: one resident weight per shape, packed once at M 60 or 512, computed at every M in 4-640, every element exact against an integer reference, over the Laya projection shapes |
| `matmul_stream_vs_prepacked_rocket` | per-call packB cost (stream vs resident) |
| `iova_ceiling_rocket` | the 32-bit IOVA window size, per-fd vs shared |
| `prototype_shared_scratch_rocket` | de-risk the shared-scratch ownership refactor |
| `matmul_int8_rocket` | int8xint8->int32 readiness (encoding sweep) |
| `matmul_int8_tiled_rocket` | tiled int8 (M/N tiles + host int64 K-accum), bit-exact |
| `matmul_int8_prepacked_rocket` | resident int8 weights, bit-exact vs one-shot |
| `int8_chain_probe` | independent int8 tasks self-chained into one kick with `ROCKET_JOB_BATCHED`, each scored against a CPU model, beside a gapped control and a chain whose first task reads zeros. Prints the NPU interrupts each job raised |
| `int16_native_probe` | int16xint16->int32 in one task under two output geometries (fp16's `size_e` 3 and int8's 7), four operand fills including full-range int16, six non-square shapes, every element scored against the int64 sum clamped to int32. Counts tasks the watchdog retired. A probe: its int8-geometry cases hang by design |
| `rowmajor_matmul_probe` | one fp16 -> fp32 matmul task in two forms, the cube program and the same program with six registers patched so it reads A and writes C as plain row-major buffers. 12 shapes to K 4096 and N 2048, every element scored, plus interleaved submit-to-fence medians. A probe, not a correctness test |
| `rowmajor_tile_probe` | the row-major matmul form at the shapes a tiled path issues, every element of a sentinel-filled output scored, beside cube-program controls. Then submit-to-fence timings at the library's tile shapes. A probe, not a correctness test ([`rowmajor_tile_probe`](#rowmajor_tile_probe)) |
| `rowmajor_kacc_chain_probe` | the row-major matmul form inside one chained fp16 K-accumulation kick with `DATA_REUSE`, in the library's own order: the cube program, own-width row-major tiles, and tiles written straight into C, each with reuse on and off, every element scored, then the whole kick's submit-to-fence over rotated reps. A probe, not a correctness test |
| `fa_replay_probe` | replays one flash-attention op from the tiles `ROCKET_FA_DUMP_OP` wrote, through `rocket_flash_attn_fp16_ctx`, many times in one context and in fresh ones, at a chosen worker count, and counts how many trials equal each of two dumped outputs and how many distinct outputs appear. With `FA_REPLAY_MARKER` naming a tracefs `trace_marker` it brackets each trial for `tools/fa_trace_join.py`, which joins the `gpu_scheduler` job events (run under `tools/fa_core_trace.sh`) to say which core ran each worker's QK and AV jobs. A probe, not a correctness test |
| `fa_core_matmul_probe` | one head's real AV operands (written by `tests/fa_av_dump_wrap.c`, a `-Wl,--wrap=rocket_mm_batch_run` link of the replay) from several worker fds at once, the handler's passes and pass 0 through `rocket_matmul_fp16_f32out`, each call compared with the first of its kind and with the exact sum. The core that ran a call comes from the trace. A probe, not a correctness test |
| `npu_core_diff_probe` | `run` writes fp16 -> fp32 tiles at 512×384×256 for three synthetic A distributions, an `int8` control through `rocket_matmul_int8`, and `perm` K permutations of real AV operands with `NPU_CORE_DIFF_OPS` (`NPU_CORE_DIFF_PERM` restricts them to inside or between aligned K blocks, `NPU_CORE_DIFF_XOR` remaps each 32-wide block's lanes), to files. `compare` diffs two runs element by element and says which side is the correctly rounded one. One core per process, chosen by `RKNPU_CORE_MASK` on the vendor provider. A probe, not a correctness test |
| `ppu_sub4_chain_probe` | two chained PPU pooling passes with the kernel order forced so the intermediate is 2x2 or 3x3 (MAX and AVG, C 64 and 130, a 4x4 control): the chained second pass scored per channel against the same pass over a CPU-scattered copy of the device-written intermediate and against CPU models. `--no-ppu-done` adds a reproduction case whose jobs retire by the watchdog. A probe, not a correctness test |
| `dw_int8_cost_probe` | where the int8 depthwise entry's per-call time goes at MobileDet's largest depthwise shapes: the whole warm call, its host scatter and gather as written, and a blocked form of each checked byte for byte against them. A probe: it asserts only that the blocked forms agree |
| `conv_pack_cost_probe` | the fp16 conv's and the int8 direct conv's per-element host pack against the whole warm call, at EfficientDet-Lite0's depthwise shapes and MobileDet's direct shapes, with a blocked form of each pack checked byte for byte. A probe: it asserts only that the blocked forms agree |
| `cpend_wzp_probe` | `DPU_BS_OW_OP`, the TRM's CPEND operand, as an asymmetric weight zero point, on the int8-out depthwise program and the direct int32-raw program, each case scored against models of the operand. `layer` mode runs one real depthwise layer for `tflite-rocket/tools/requant_chain.py`. A probe, not a correctness test ([`cpend_wzp_probe`](#cpend_wzp_probe)) |
| `softmax_exp_bench` | CPU only, no NPU: `host_softmax_rows`, the flash-attention handler's separate softmax (its path under `ROCKET_FA_FUSED_SOFTMAX=0`), copied verbatim, timed on one pinned core against the same loop without `expf`, with a NEON exp pass, and fully in NEON, at rows of 512-4096, each variant's outputs scored in fp16 ulps against the verbatim loop. Measures how much time a vectorized exp can remove from the softmax phase. A benchmark, not a correctness test |
| `matmul_int8_groupwise_rocket` | one-shot **group-wise** int8 vs an fp64 reference, swept over both tiling regimes: `Kt == group` (one K-tile per quant group) and `Kt < group` (several tiles sharing one group's scale) |
| `matmul_int8_prepacked_gw_rocket` | resident **group-wise** int8 (the native-quant path) vs the one-shot oracle + fp64, plus a distinct-weights-sharing-one-ctx aliasing guard, a wrong-entry-point guard, and the unaligned-call-M rejection. Registered twice: at `group=576` (bit-exact) and at a group wider than the CBUF cap (`Kt < group`) |
| `matmul_int8_crossm_gw_rocket` | resident group-wise int8 weight packed once and reused at M = 512/256/768/64/8 with no re-pack, bit-exact at every M |
| `matmul_int8_dequant_rocket` | folding the dequant/cast into the DPU `OUT_CVT`: bit-exact int8->fp32 cast + per-tensor integer scale (tested), the ratio-classifier reverse-engineering harness (`ROCKET_INT8_DEQ*`), + the fp16-cast diagnostic. Proves OUT_CVT is an *integer* converter (fractional dequant cannot fold) |
| `matmul_int4_rocket` | int4xint4->int16 readiness + the precision/size_e sweep |
| `matmul_int4_tiled_rocket` | tiled int4, bit-exact, single-pass-K proof |
| `matmul_int4_prepacked_rocket` | resident int4, bit-exact vs one-shot |
| `matmul_int4_prepacked_gw_rocket` | resident **group-wise** int4 (the in-model W4A4 path), bit-exact vs the one-shot `rocket_matmul_int4_groupwise` + fp64 (incl. the deep FFN K=15360 / 120 groups, N-fan across workers) |
| `matmul_int16_rocket` | int16xint16->int32 test at the library's geometry, plus the saturation and transposed-writer characterization modes |
| `matmul_int16_exact_rocket` | bit-exact int16->int64 via int8 byte-decomposition (the production int16 matmul) |
| `matmul_bf16_rocket` | bf16xbf16->fp32 feasibility + encoding sweep |
| `matmul_bf16_tiled_rocket` | tiled bf16 (M/N/K tiles + host `double` K-accum), sample-verified |
| `matmul_tf32_rocket` | tf32xtf32->fp32 feasibility + 4-byte-input geometry sweep + precision characterization |
| `matmul_tf32_tiled_rocket` | tiled tf32 (M/N/K tiles + host `double` K-accum), sample-verified (incl. K=48 %16-not-%32) |
| `matmul_dtype_perf_rocket` | the "not MAC-bound" check: fp16 vs int8 vs int4 resident throughput |
| `matmul_accum_rocket` | DPU eltwise K-accum classifier (fp16): constant-operand + sentinel |
| `matmul_accum_int8_rocket` | the int32/fp32 EW-add classifier. It varies the EW algorithm and operand size but not the precision triple, so its int32 failure does not rule out an integer EW add |
| `multicore_probe` / `multicore_threads` | scheduling probes: 1 fd serializes, N fds parallelize |
| `ctx_pool_throughput` | multi-instance context-pool throughput sweep (the "rknnpool" path): P independent contexts each running prepacked-matmul "inferences". Measures aggregate scaling vs pool depth. Spread the contexts across the big cores (`ROCKET_CPU_AFFINITY=off` + pin each ctx): up to ~3.9x at P=4 on submit-bound ops, vs ~2.1x if every 1-thread context collides on one core |
| `uapi_selftest_rocket` | rocket uAPI conformance test: `CREATE_BO` contract (IOVA bump-starts at 0, so `dma_address==0` is valid, page-aligned, low-4 GB regcmd window), `PREP_BO` absolute-`CLOCK_MONOTONIC` deadline semantics, `FINI_BO`. The cross-kernel canary |
| `open_guard_race_rocket` | the RK3576 **second-fd guard under concurrent opens**: two threads released from one barrier into `rocket_open()`, 500 rounds. On an RK3576 with two cores bound and `ROCKET_RK3576_ALLOW_MULTI_FD` unset, every round must give exactly one fd and one `-EBUSY`. Anywhere else, both opens must succeed. Opens and closes only, no submit |
| `prep_signal_robust_rocket` | **`PREP_BO` wait robustness** test: a real fp16 job driven in a loop while a `SIGUSR1` storm hits the waiting thread (handler without `SA_RESTART`): the interruptible kernel wait must not surface a spurious timeout (`EINTR` is retried to the same absolute deadline), and a `UINT64_MAX` "wait forever" timeout must saturate rather than wrap the signed deadline negative into an instant poll. Outputs byte-checked every iteration |
| `activation_lut_rocket` | DPU LUT elementwise activation: fp16 sigmoid + hardsigmoid vs fp16 CPU ref (tol 0.005), a regcmd smoke test off-device, and a test of the on-NPU EW multiply + fully-on-NPU HardSwish/SiLU (`ROCKET_ACT_NPU_MUL`) |
| `leaky_relu_rocket` | DPU LUT **LeakyReLU**: sweep across [-R,R] + the x~0 band (the sign-based mux spike) vs fp16 ref, alpha 0.01-0.5. The `scan` arg maps the raw glitch (repair off) |
| `activation_concurrent_rocket` | the parametric LUT activations under **concurrent callers**: two threads on two fds, released from one barrier per round, alternate LeakyReLU slopes 0.01 and 0.3, then ELU beside SELU, each result scored within 2 fp16 ulp of the same configuration run serially first. A table built in storage shared across calls computes the other caller's function. RK3588 only, skips elsewhere |
| `gelu_rocket` | the **2-pass on-NPU GELU** (`x·Φ(x)`, clean unit-LUT gate) vs true erf-GELU over a wide `[-12,12]` (incl. the flat tails that spike the single-pass), cos=1.000000. SiLU regression check |
| `ew_mul_rocket` | fully-on-NPU elementwise binary op (identity-conv main + `EW_OP_TYPE`): low-level gen `A+B`/`A*B` bit-exact (sweepable via `ROCKET_EW_CFG`) **+ a runtime check of `rocket_ew_add_fp16`/`rocket_ew_mul_fp16`** (flat vectors, n up to 40000 across the M-tile boundary) |
| `ew_minmax_rocket` | on-NPU elementwise two-tensor **max/min** (DPU EW ALU algo MAX/MIN) + **Clip**, bit-exact vs host (n to 40000) |
| `prelu_rocket` | on-NPU **PReLU** (per-channel slope): the `max(x,α_c·x)` (α∈[0,1]) and general (α outside) paths, bit-exact vs an fp16-faithful ref |
| `softplus_mish_rocket` | DPU-LUT **Softplus** / **Mish** (YOLOv4/v7) / **Abs** vs double-precision math (sweeps + large-n tile-boundary) |
| `elu_rocket` | DPU-LUT **ELU/SELU** (symmetric shifted table + host x~0 repair) vs the math, alpha 0.5-2.0 + the SELU constants |
| `matmul_plan_gate` | host-only, no NPU: the RK3588 tiling policy's invariants over every shape and `ROCKET_MM_*` override. `matmul_plan_gate rk3576` (registered as `matmul_plan_gate_rk3576`) holds `rocket_matmul_plan_int8_rk3576` to its limits instead: the job count is exactly `ceil(N/Nt)*ceil(M/Mt)`, a prime M plans in bounded time, a plan past `INT_MAX` jobs returns `ROCKET_E_TILING`, a dimension past 2^24 or a plane taller than 65535 returns `ROCKET_E_SHAPE` |
| `pool_rk3576_bracket_gate` | host-only, no NPU: the RK3576 pooling entry against a host-memory provider defined in the test, which replaces the whole submit seam and faults a CPU access to a BO outside `PREP_BO`/`FINI_BO`. Seven paths, each held to an exact PREP/FINI count on the output and scored against `rocket_pool_ref_int8_rk3576()` ([`pool_rk3576_bracket_gate`](#pool_rk3576_bracket_gate)) |
| `pool_stock_kernel_gate` | host-only, no NPU: the pooling entries against a mock provider whose `rocket_ppu_done_supported()` answers 0, as a driver without the PPU completion does. Every pool entry refuses and the three global pools return the host reference, with no BO allocated and no submit. A control with the answer at 1 reaches the allocator |
| `bytes_moved_rocket` | analytical DRAM-traffic model (pure, no NPU): per-phase bytes from shape+tiling+dtype+reuse, planner `njobs` cross-check, int8 readback lower-bound demo |
| `conv2d_fp16_rocket` | general fp16 `CONV_2D` + native depthwise, bit-exact vs an NHWC oracle (direct, OC%16 pad, DW G=32, tiling) |
| `pool_fp16_rocket` | on-NPU MaxPool / AveragePool (PPU): the output-width allowance against every measured cell, cube and band-split self-checks, HW oracle (max bit-exact, avg <= fp16-recip tol), including outputs run as column bands (YOLO's SPPF among them) |
| `pool_int8_rocket` | on-NPU int8 / uint8 MaxPool / AveragePool via the fp16 PPU path: int8 & uint8 MAX bit-exact, int8 AVG ±1 ULP vs an integer golden |
| `pool_int8_native_probe` | the PPU's native int8 pooling precision, as a map over the precision pair: `PPU_DATA_FORMAT[2:0]`=0 with `PPU_RDMA_DATA_FORMAT[1:0]`=1 over a C2=16 byte cube pools MAX, MIN and AVG bit-exactly, given an integer Q16 `0x10000/k` reciprocal and the integer pad fill. A probe, not a correctness test. The library's pooling entries use the fp16 path |
| `conv_transpose_resident` | the resident ConvTranspose against the scatter reference: bit-exact on integer fills, and `\|err\| <= 2^-9` of each element's magnitude sum on real fills with a bias, over 18 direct shapes and pix2pix's and SAM's ten layers. Registered twice, the second under `ROCKET_KACC=0` ([`conv_transpose_resident`](#conv_transpose_resident)) |
| `ct_model_bench` | per-call time of the old entry and the resident entry at pix2pix's and SAM's ten layers, warm medians, the order rotated by an argument for interleaved passes. A benchmark, not a correctness test |
| `ct_route_probe` | the candidate ConvTranspose paths, timed: col2im and sub-pixel as resident matmuls, sub-pixel as one forward conv, host phases split. A probe, not a correctness test |
| `deconv_extent_probe` | the CNA deconvolution mode with the output extent programmed rather than derived, through `rocket_conv2d_fp16_job_extent()`: one task per case, scored over the whole output against `rocket_conv_transpose2d_ref_fp16`. Bit-exact on the RK3588 up to 64×64 -> 128×128. A probe, not a correctness test. `rocket_conv_transpose2d_fp16` runs the mode on its direct path (`rocket_conv_transpose2d_route()`) |
| `requant_round_probe` | the OUT_CVT's tie rule at `OUT_CVT_SHIFT` bit 30 = 0 and 1 (`ROCKET_OUT_CVT_ROUND`), over ties it builds on purpose, since no round scale reaches one. It asserts the recorded rule: half to even with the bit clear and half away from zero with it set, each the one survivor of five candidate rules, on both SoCs (the RK3576 through `rocket_matmul_int8_rk3576`, the RK3588 through the int8 matmul's integer convert). A `ctest` and a `gates-rk3576.sh` row |
| `reduce_mean_rocket` | on-NPU spatial reductions over [H,W]: GlobalAvgPool / Mean (HW vs fp64, tolerance) **and** GlobalMax/MinPool / ReduceMax/Min (**bit-exact**, idempotent), plus a factor and schedule/cube self-check (single + multi-pass, square + equal-count rect, host fallback for non-8-smooth / unequal-count / a plane no pass order fits), planes of 216 to 512 |
| `cumsum_rocket` | on-NPU **cumsum / prefix sum** (triangular ones-matmul) vs an fp64 prefix-sum oracle, over all four variants (incl/excl x forward/reverse), M-tile boundary, N%32≠0, T=1500. **Bit-exact** (max_abs 0). Independent O(N²) fp64 recompute self-check |
| `conv2d_int8_rocket` | native int8 `CONV_2D`: cube self-check + single-job HW + the **tiled-runtime direct case** (big/wide/IC<32/OC-pad) bit-exact vs an int64 oracle |
| `conv1d_rocket` | **conv1d** front-end (Whisper conv1/conv2, KW=3, stride 1/2) lowered onto a width-1 conv2d with time on the height axis, bit-exact / fp16-tolerance vs the conv oracle (incl. IC=80/512) |
| `conv_width_gate` | int8 and fp16 convs through the public entries whose tiles would pass the CNA's 11-bit width or 10-bit grains fields (2300 and 2500 columns at one or two rows, 3000 rows at one column), beside controls under the limit. Every element scored, a refusal counts as a failure |
| `rk3576_conv_width_probe` | the same question on the RK3576: its int8 and fp16 conv entries and int8 max pool, driven with extents just under and past 11, 12 and 13 bits, kernels of 32 and 33, and strides past the 3-bit field, every element scored. `plan` prints each case's plan without submitting. A probe, not a correctness test ([`rk3576_conv_width_probe`](#rk3576_conv_width_probe)) |
| `rk3576_retire_probe` | the RK3576 write guard's completion half. `forced` sets the backstop to 1 us and asserts that eight small calls each refuse with the retirement counters moved. `real` asserts that they return 0 with none counted, and that an int8 matmul whose jobs retire either refuses or is exact. A probe, not a correctness test ([`rk3576_retire_probe`](#rk3576_retire_probe)) |
| `exp_lut_rocket` | DPU-LUT **EXP** (shifted single-table, the q>=1 clamp) + a **softmax-sum** end-to-end check (row-max subtracted) vs `exp` |
| `softmax_rocket` | on-NPU **row-wise softmax and LogSoftmax** vs an fp64 oracle (M-tile boundary, Whisper T=1500, wide-spread tail). Softmax rows sum to 1, logsoftmax `Σexp(out)=1` |
| `cross_entropy_rocket` | stable per-row **cross-entropy** (logsumexp reduce + host gather) vs an fp64 CE oracle (M-tile boundary, T=1500 vocab, random + argmax targets, wide spread). **fp32-grade** (max_abs <= 1.5e-4), CE >= 0. Self-check `CE == −logsoftmax[target]` (independent path) |
| `layernorm_rocket` | on-NPU **LayerNorm** (stacked-row reduce + affine fold) vs fp64 (M-tile boundary, no-beta, H%32≠0, the overflow prescale) |
| `norm_vision_rocket` | on-NPU **BatchNorm / GroupNorm / InstanceNorm / L2-Normalize** vs fp64 (group counts incl. G=1 / G=C, C%32≠0, P=1, row-tile boundary, the overflow prescale) |
| `mha_rocket` | on-NPU **multi-head self-attention** vs an fp64 attention oracle (cosine sim, Whisper-base d=512/8-head, T%16≠0 key-pad path) |
| `flash_attn_rocket` | masked **grouped-query (flash) attention** vs an fp64 oracle (cosine sim, Gemma-4-12B head_dim 256 / 16 q-heads / 8-kv GQA + 1-kv MQA, sliding window, soft-cap, n_kv>T, unaligned T/n_kv). The **chained long-context** path (`ROCKET_FA_CHAIN_ELEMS` high -> a worker's whole head range in one QK+AV job, to 16K). The **online/tiled** path (`ROCKET_FA_TILE_KV` -> per-row masked-tile skip, short last tile, 16-tile 8K, grow/reuse ctx). `bench` mode A/Bs materialized vs tiled `_ctx` wall-time |
| `encoder_block_rocket` | one **full Whisper encoder block** (LN+MHA+residual+LN+MLP) vs an fp64 block oracle (cosine sim, Whisper-base, T%16≠0) |
| `siglip_rocket` | the **full SigLIP-B/16 vision encoder** vs an fp32 reference (per-layer cosine, mean 0.999998, plus a resident-path bench, and the reference encoded again after another image on the same context, which must match the first encode). Skips without the weight blob + oracle artifacts on disk |
| `sam_rocket` | the **SAM ViT-Det image encoder** vs the Hugging Face fp32 reference from `tools/sam_extract.py`, per layer and at the neck (each cosine >= 0.99). It always runs the exact host reference, so it checks the datapath off-device too, then the NPU path. `ROCKET_SAM_BENCH=N` times the resident path. Skips without the artifacts (`-DROCKETNPU_SAM_ARTIFACTS`) |
| `modernbert_rocket` | the **resident ModernBERT encoder and its post stack** vs ONNX Runtime's fp32 taps (per-layer cosine over the live rows, a batch of 3 and single long sequences, and `ROCKET_MB_BENCH=N` adds a warm median). `--host` runs the host mode. Skips without `-DROCKETNPU_MODERNBERT_ARTIFACTS` |
| `replay_dw_mesa` | int8 DW int8-out **regcmd** replayed on Mesa/Teflon's captured BOs, bit-exact vs `mesa-output`: the library's register program is Mesa's, pad word aside. Not a check of the function: the capture ran an int8 model through Mesa's uint8 driver, on a constant input |
| `conv_dw_int8_runtime` | int8 DW int8-out **runtime** against TFLite's reference kernels on the capture's model, within one at a rounding boundary on at most 2% of outputs, and bit-exact against an exact host model of the requant at fifteen shapes. The planner refuses along each axis it checks, and a weight zero point outside int8 refuses ([`conv_dw_int8_runtime`](#conv_dw_int8_runtime)) |
| `conv_dw_int8_perc_runtime` | the **per-channel** int8 depthwise entry at six shapes (one job, 1152 deep with a 3e4 scale spread, all-zero and near-dead channels, 480 at 20x20, banded 160x160, a partial group, three jobs): every element equal to a host model of the BS-multiplier and OUT_CVT arithmetic built from the entry's own plan, and within one of TFLite's per-channel arithmetic. Every output zero point sits inside the range, since at -128 a wrong negative-product shift passes |
| `conv2d_int8_q_rocket` | the int8-out direct conv entry at 13 shapes through the fd, ctx, 3-fd pool and fd -1 entries, every element against a host model of TFLite's accumulator and the OUT_CVT. Five are 1×1s from 160×160 to 20×20 at a frontend's pointwise shapes. A `ROCKET_CONV_BATCH=1` twin |
| `conv2d_int8_q_perc_rocket` | the **per-channel** int8-out direct conv at 8 shapes (three scale-class jobs at 80×80, OC 672 and 1152, a 160×160 1×1, a padded IC under a CNA pad, the stem at stride 2, 5×5 stride 2, all-zero filters) through the fd, ctx, 3-fd pool and fd -1 entries: every element equal to the test's own accumulator taken through the split requant at the entry's plan, and within one of TFLite's per-channel arithmetic. A `ROCKET_CONV_BATCH=1` twin |
| `conv_i8out_probe` | a probe of Mesa's direct int8-out program on the RK3588: its geometry, CPEND on it, the per-channel BS multiplier and `B` on it. `witness` adds the cases that time out (a partial 32-kernel group, the depthwise writer geometry) |
| `dw_perc_probe` | a probe of the BS stage's per-channel multiplier on the raw int8-out depthwise program: the coefficient layout, the order of add and multiply, the held-wide product, the tie rule, the sign-split shift fields, `B` as a per-channel CPEND, and the BN stage's multiplier (which hung in the one configuration tried) |
| `dump_regcmd` | diff the generated regcmd on a host machine (no HW) |
| `replay_dump` | replay a dumped in-context failing matmul |
| `matmul_fp16_rocket` | standalone fp16xfp16->fp16/fp32 single-task smoke test (the `gen_gemm_fp16` path, no tiling) |
| `dump_dw_regcmd` | host-only: emit a depthwise-conv regcmd as a u64 stream for the Mesa decoder (no HW) |
| `membench` | DRAM bandwidth + NPU readback de-tile microbench (no deps) |

The `accum` classifiers and `dtype_perf` characterize K-accum feasibility and the
dtype-independent throughput limit.

### `rowmajor_tile_probe`

The cases are a K slice of a wider A, an N tile into a wider C, and both. Further cases are
int8 -> int32, fp16 out, and a two-task fp16 K-accumulation with a cube partial and a row-major
final write. The timings
split into the read, the write and the row pitch (`--pitch`, `--pitch-asym`, `--pitch2`). The
wait is on a small fence BO, so the output BO's sync is not timed.

### `cpend_wzp_probe`

The depthwise cases are five weight zero points and three shapes, with the operand swept over
`-zw`, 0, `+zw` and the low byte. Each is scored against TFLite's accumulator plus the OUT_CVT
model, and against `bias' + sum xp*w + v*sum xp` for four readings of the operand. The direct
int32-raw program has CPEND patched live, and is scored in int32 and for placement anywhere in
the BO. Two shapes have a partial last 64-channel group, which the program leaves unwritten.

### `pool_rk3576_bracket_gate`

The provider holds each BO `PROT_NONE` outside its bracket, so a CPU access outside
`PREP_BO`/`FINI_BO` faults. A double PREP, an unmatched FINI, a submit while the CPU holds a BO
and a bracket left open are counted. Each of the seven paths runs with
`ROCKET_RK3576_I32_SENTINEL` on and off:

- A clean call
- The divisor-lag check
- A retired job
- A lagged output
- An unwritten output
- A failed submit
- A failed wait

### `conv_transpose_resident`

On integer fills the host asserts every matmul partial prefix and output below 2048. The shapes
are the old test's 18 direct shapes, a cropping pad, a per-axis dilation and stride, and
pix2pix's and SAM's ten layers. Both worker splits run at four of those layers. The test also
checks a depthwise refusal, a handle that keeps its packed weight after the caller's buffer
changes, and one that refuses another context.

### `rk3576_conv_width_probe`

The entries are the int8 direct, depthwise and packed-image convolutions, the fp16 direct and
packed-image ones, and the int8 max pool. The ladders reach a width of 12288, a one-task height
of 6144, and depthwise and pool channels of 8224. Planes of four rows, 2048-8192 wide, have
tasks that read the line stride past 14 bits. Each case reports rc, wrong and never-written
elements, and the column, row and channel a wrap starts at. `plan` prints each case's row plan
and its first task's geometry words.

### `rk3576_retire_probe`

Under `forced` every job reads as retired. The eight calls are the direct, depthwise,
packed-image and fp16 convolutions, the int8, int32 and fp16 matmuls, and the max pool. `real`
runs the int8 matmul at M 512 and a chosen K and N, by default 2240 and 1536. At that shape
every job retires at the backstop. It asserts that the retirements are counted as the kernel
logged them, and that the call never returns 0 over a wrong output. The net test covers the
chained kick, run under `ROCKET_RK3576_BACKSTOP_US=1`.

### `conv_dw_int8_runtime`

`tests/dw_litert_ref.py` writes the TFLite fixture. Eight host-model shapes have symmetric
weights, and seven have an asymmetric weight zero point. Those seven cover zero points from -128
to 127 and a partial 64-channel group. They also cover pad 0 on an odd plane at stride 2, and a
768-channel 20×20 layer.

## Runtime environment variables

The library reads the `ROCKET_*` environment variables below, and the `ggml-rocket` backend
reads more. `ggml-rocket` and some tests also read `ROCKET_N_THREADS`, the worker count they
pass to a context, which the library does not read. `sudo` strips the environment, so use
`sudo -E`.

Values parse in one of three ways, and a row says which where it matters. A presence switch
acts on any value, `0` included. An integer switch reads its value as a number, so `on`, `off`
and `yes` all read as `0`. A leading-zero switch turns off only on a value that begins with
`0`, so `off` leaves it on.

About 150 more variables override one register field each, for the reverse-engineering probes,
and are not listed. A caller has no reason to set them.

### Device, submission and threads

| Variable | Default | What it does |
|---|---|---|
| `ROCKET_DEV` | `/dev/accel/accel0` | The device node `rocket_open()` opens, for a board with more than one NPU or a test rig. An empty value selects the default |
| `ROCKET_CHIP` | auto | Forces the hardware profile by name, `rk3588` or `rk3576`, instead of the device tree's `compatible`. Any other name warns and selects `rk3588`. Read once. Caution: forced to `rk3588` on RK3576 silicon, the RK3588 generators no longer refuse, and their jobs complete with the output unwritten. See Chip selection |
| `ROCKET_WAIT_MS` | `8000` | The fence deadline, in milliseconds, of the RK3588 matmul entries: one-shot, context, resident int8 and int4, and bf16. A value below 1 selects 8000. The convolution and RK3576 paths wait a fixed 2 s. Read once |
| `ROCKET_SLOW_WAIT_MS` | per SoC: 450 RK3588, 110 RK3576 | The slow-wait mark, in milliseconds. A fenced wait that reaches it logs a warning naming the BO and counts in `rocket_fence_wait_slow_count()`, because the `rocket` driver retires a job that runs past its watchdog (500 ms on the RK3588, a 125 ms backstop on the RK3576) and signals its fence as if it had completed. A value of 0 or below selects the SoC's. Read once |
| `ROCKET_BUSY_POLL` | `0` | Microseconds a fenced wait polls the fence before it sleeps. `0` never polls. It suits small, submit-bound jobs with a small output BO, and the poll occupies a CPU core. `rocket_busy_poll_set_us()` overrides it. Read once |
| `ROCKET_BATCH_SUBMIT` | per provider: off on the `rocket` driver, on with `rknpu-submit` | `1` self-chains a multi-task submit into one kick with one completion interrupt: a tiled matmul's tiles, `rocket_matmul_fp16_batch` and flash attention's chained heads. `0` forces the per-task path. An integer switch, so `on` selects the per-task path. A kernel without batched-submit support logs a warning and runs the per-task path. Read once. See Batched submit and chaining |
| `ROCKET_CPU_AFFINITY` | auto | The CPUs the pack and readback workers pin to, as a list such as `4-7` or `4,6`. Auto selects the CPUs with the highest `cpuinfo_max_freq`. `off`, `none`, an empty value and `0` turn pinning off, so CPU 0 alone is `0-0`. Read once. See CPU affinity for multi-pool processes |
| `ROCKET_PIN_AT_CREATE` | on | Each per-call worker is created on the big core it pins to. `0` creates it with its creator's CPU mask, so a creator pinned to one core starts every worker on that core. An integer switch, so `on` turns it off. Read once |

### Logging and profiling

| Variable | Default | What it does |
|---|---|---|
| `ROCKET_LOG_LEVEL` | `info` | The log threshold: `error`, `warn`, `info` or `debug`, or `0` to `3`. Any other value selects `info`. Read once, at the first message. See Logging |
| `ROCKET_DEBUG` | unset | Any value, `0` included, raises the log threshold to `debug` and adds per-call traces: the device open, the hardware profile and each worker's CPU |
| `ROCKET_LOG_STDERR` | off | Any nonempty value except `0`, `off` included, also writes each line to `stderr` while a host callback is installed (`rocket_log_set_callback`). With no callback the default sink already writes to `stderr`. Read once |
| `ROCKET_MM_PROFILE` | unset | Any value, `0` included, prints at exit the matmul entries' time by phase: pack, sync, submit, wait and read. It covers the RK3588 matmul entries and the RK3576 int8 and fp16 matmuls. Read once |
| `ROCKET_CONV_PROFILE` | unset | Any value, `0` included, prints one line at exit per RK3588 convolution program (int8 direct, int8 depthwise, fp16 and int8-out), its jobs' phases summed. It also prints the resident forward convolution's unfold, GEMM and output time, and the resident ConvTranspose's GEMM and scatter-add time. Read once |
| `ROCKET_FA_PROFILE` | unset | Any value, `0` included, splits a flash-attention head range into gather / QK / mask / softmax / AV / scatter. At exit it prints the bucket sums, then the largest single range's buckets. The sums add intervals over concurrent workers, so they are not a share of anything. The second line was measured on one thread while the dispatch thread waited. Apply a bucket's share of `max-range` to the caller's measured wall-clock time. Read once |
| `ROCKET_MB_PROF` | unset | Any value, `0` included: `rocket_modernbert` prints each encode's phase split (LayerNorm, GEMM, attention, activation, down-projection). Read per encode |
| `ROCKET_SIGLIP_PROF` | unset | Any value, `0` included: `rocket_siglip_encode_ctx` prints each encode's phase split, the bias adds' own bucket and the call's minor page faults. Read per encode |

### Matmul

| Variable | Default | What it does |
|---|---|---|
| `ROCKET_KACC` | on | The RK3588 fp16 matmul sums its K partials on the NPU with the DPU's fp16 add, which moves the result by about 0.4% against a host fp64 sum. `0` selects the byte-exact host fp32 sum. An integer switch, so `on` turns it off. Read once |
| `ROCKET_NO_KACC` | unset | Any value, `0` included, turns `ROCKET_KACC` off. It is read only while `ROCKET_KACC` is unset |
| `ROCKET_KACC_CHAIN` | off | `1` chains a tile's K-accumulation steps into one kick where at least three tiles share each step (`64/nKt >= 3`), and runs one submit per step elsewhere. `2` chains every K-step count up to 64. It needs `ROCKET_KACC` and the kernel's batched-submit support, and otherwise runs per step. Read per call. See Batched submit and chaining |
| `ROCKET_REUSE` | auto | CBUF operand reuse: `0` off, `1` weight, `2` data. Auto picks, per shape, the operand with the longer run of consecutive tiles, data on a tie, and is off while `ROCKET_KACC` is off. A value outside 0-2 turns reuse off. Reuse stays off where a batch of tasks does not run as one job. Read once |
| `ROCKET_MM_MT/NT/KT` | planned | Tile overrides, rounded down to the tile granularity and clamped to the shape and the CBUF. Setting any of them selects the whole-tile plan, even at a value the planner ignores. `ROCKET_MM_NT` also turns off `ROCKET_MM_ASYM`. Read per call |
| `ROCKET_MM_STREAM` | on | The fp16 planner streams the weights through the CBUF, with far fewer tasks. `0` selects the whole-tile plan. An integer switch, so `on` turns it off. Read once. See Matmul tiling plans |
| `ROCKET_MM_STREAM_INT8` | on | The int8 planner streams its weights too, while `ROCKET_MM_STREAM` is on. `0` selects its whole-tile plan. An integer switch, so `on` turns it off. Read once. See Matmul tiling plans |
| `ROCKET_MM_ASYM` | on | Asymmetric Mt>Nt tiling of the whole-tile plan, for a shape that tiles both N and K. `0` forces the symmetric plan. An integer switch, so `on` turns it off. Read per call. See Matmul tiling plans |
| `ROCKET_MM_PIPE` | unset | Any value double-buffers `rocket_matmul_fp16`'s host-sum path, so the host de-tiles one batch while the NPU computes the next. The result is bit-identical. It applies only with `ROCKET_KACC=0` and CBUF reuse off. Read per call |
| `ROCKET_I8_PACK_FANOUT` | on | `rocket_i8_weights_pack` and `_pack_gw` pack each worker's column slice into its BO on its own thread, for a weight of 1 MiB or more. `0` packs the slices one after another on the calling thread. Only a value that begins with `0` turns it off. Read once |

### Attention and encoders

| Variable | Default | What it does |
|---|---|---|
| `ROCKET_ATTN_HOST_SOFTMAX` | per path: NPU for `rocket_mha_self_fp16`, host for flash attention | A nonzero integer runs the softmax on the host, and `0` runs it on the NPU, on both paths. `rocket_encoder_block_fp16` follows `rocket_mha_self_fp16`. An integer switch, so `on` moves flash attention's softmax to the NPU. On the NPU, flash attention uses neither `ROCKET_FA_FUSED_SOFTMAX` nor `ROCKET_FA_ROWMAJOR`. Read once per path |
| `ROCKET_FA_CHAIN` | on | Batches a flash-attention worker's per-head QK/AV submits through a resident batched-matmul context. `0` forces the per-head path. An integer switch, so `on` turns it off. Read once |
| `ROCKET_FA_CHAIN_ELEMS` | 32 Mi elements | Bounds the score elements of one chained head group, so a worker's head range stays batched up to about 20K context. Raise it to chain at deeper context, for more scratch. A value below 1 selects 1. Read once |
| `ROCKET_FA_TILE_KV` | off | Opt-in online, tiled long-context flash attention. The value is the key-tile width, rounded up to a multiple of 32, so `1` selects a 32-key tile. It bounds memory use and is slower than the materialized path on this dispatch-bound NPU. It needs equal K and V head sizes. Read per call |
| `ROCKET_FA_TILE_MIN_KV` | 8192 | The padded key count `ROCKET_FA_TILE_KV` must exceed to engage. A tile as wide as the key count does not engage either. Read per call |
| `ROCKET_FA_FUSED_SOFTMAX` | on | The flash attention's host scale, soft-cap, mask and softmax as one fp32 pass per row with a vector exp and one fp16 rounding. `0` restores the separate mask pass and softmax. An integer switch, so `on` turns it off. Read once |
| `ROCKET_FA_ROWMAJOR` | on | On the persistent `rocket_fa_ctx` path, QK writes the scores and AV reads P as rows in a resident BO, so neither is de-tiled out of nor packed into the NPU's cube layout. It also needs `ROCKET_FA_CHAIN`, the fused host softmax and a QK plan of one K tile. `0` keeps the cube handoff. An integer switch, so `on` turns it off. Read once |
| `ROCKET_FA_RM_MAX_ROW` | 4096 | The widest row of scores, in bytes, that `ROCKET_FA_ROWMAJOR` writes as plain rows, with P rewritten in place. The default is 2048 keys. Past it QK writes one block of rows per output tile and P goes to a second BO. Read once |
| `ROCKET_ENCODER_NOFUSE` | unset | Any value runs `rocket_encoder_block_fp16`'s FFN as separate calls instead of `rocket_mlp_fp16_fused`: `bf1` added on the host, the GELU multiply on the DPU EW unit. Read per call |
| `ROCKET_SIGLIP_FA` | on | With more than one worker, `rocket_siglip_ctx_create` fans the attention heads across the worker fds through a flash-attention context. `0` keeps one resident single-stream context. An integer switch, so `on` turns it off. Read at context creation |
| `ROCKET_SIGLIP_PIN` | on | The SigLIP encoder pins its host worker threads, and the calling thread's serial work, to the big cores. `0` leaves them unpinned. An integer switch, so `on` turns it off. Read once |
| `ROCKET_MB_BAND_TILE` | 128 | `rocket_modernbert`'s banded sliding attention query tile, in rows. `0` computes every sliding layer dense, and a negative value selects 0. Read at context creation |
| `ROCKET_MB_ATTN` | npu | `host` runs `rocket_modernbert`'s attention in fp32 on the host, and any other value keeps the NPU. Against the 2026-09-27 library, host attention tied the NPU at 58 tokens and was slower at 639 [HW sweep, RK1, ~816 MHz]. The crossover in later builds is not characterized. Read at context creation |

### Convolution and activations

| Variable | Default | What it does |
|---|---|---|
| `ROCKET_CONV_BATCH` | off | `1` submits a tiled RK3588 int8 or uint8 direct convolution's tiles as one multi-task job: one submit, one fence and one IOMMU attach, each tile still its own kick. It is bit-identical to the per-tile path. Any value that is not a positive integer leaves it off. Read once. See Batched submit and chaining |
| `ROCKET_DW_PERC_MIN_C` | 2048 | The smallest BS multiplier a channel keeps in the RK3588 per-channel int8 convolutions (`rocket_conv2d_int8_q_perc`, `rocket_conv2d_dw_int8_perc`). Channels sorted by scale start a new job where the next would fall below it. At 2048 a channel's gain is within 2^-12. A lower value gives fewer jobs and a coarser gain. Clamped to 1-32767. Read once |
| `ROCKET_CONV_TRANSPOSE_HW` | on | On the RK3588, `rocket_conv_transpose2d_fp16` takes the CNA deconvolution mode for a descriptor it fits. `0` forces the lowering onto a forward convolution, and `rocket_conv_transpose2d_route()` reports the same choice. Only a value that begins with `0` turns it off. Read per call |
| `ROCKET_CT_SPLIT` | per shape | `m` or `n` forces how the resident ConvTranspose's workers split its GEMM. Only the first letter counts, in either case. Read at pack time |
| `ROCKET_CONV_SPLIT` | per shape | `m` or `n` forces how the resident forward convolution's workers split its GEMM (`rocket_conv2d_weights_pack`), by the ConvTranspose's rule. Read at pack time |
| `ROCKET_CT_SCATTER_THREADS` | the context's worker count | The thread count of the resident ConvTranspose's host scatter-add. It is capped at 8 and at OC, and one thread runs below 2^16 output elements. Read at pack time |
| `ROCKET_IM2COL_THREADS` | the context's worker count | The thread count of the resident forward convolution's host unfold. It is capped at 8 and at IC, and one thread runs below 2^17 unfolded elements. Read at pack time |
| `ROCKET_ACT_NPU_MUL` | unset | Any value, `0` included, runs the multiply of the two-pass HardSwish, SiLU, GELU and Mish on the DPU EW unit instead of the host. It reaches every caller of `rocket_activation_fp16`, the FFN and encoder entries included. Read per call |
| `ROCKET_LUT_XLO/XHI` | per kind | The input domain of the shifted-table LUT kinds: `SQRT`, `RSQRT`, `RECIPROCAL`, `LOG`, `EXP`, `SOFTPLUS` and `ABS`. An input outside it clamps to the edge value. The grid is uniform, so the accuracy follows the domain width. `XLO` must be below `XHI`. Caution: the setting is process-wide, so it also moves the on-NPU softmax's exp and `rocket_ew_div_fp16`'s reciprocal. Read per call |
| `ROCKET_LEAKY_R` | 16 | The half-width R of `rocket_leaky_relu_fp16`'s table, which covers [-R, R] and saturates beyond. A value below 1 is ignored. Read per call |
| `ROCKET_ELU_R` | 8 | The half-width R of the `rocket_elu_fp16` and `rocket_selu_fp16` table, which covers [-R, R] and saturates beyond. A value below 1 is ignored. Read per call |

### RK3576

| Variable | Default | What it does |
|---|---|---|
| `ROCKET_RK3576_ALLOW_MULTI_FD` | off | On an RK3576 with more than one core bound, `rocket_open()` refuses a second fd with `-EBUSY`, because two jobs in flight there produce wrong results. A value lifts the refusal, for a caller that serializes its own submits. Any nonempty value that does not begin with `0` lifts it, `off` included. Read per open |
| `ROCKET_RK3576_BACKSTOP_US` | 125000 (the profile's) | What the RK3576 write guards score a job's submit-to-fence time against, in microseconds. A job at or past it was retired and is redone. A probe sets it to 1 to make every job count as retired. A value of 0 or below selects the profile's. Read once |
| `ROCKET_RK3576_PM_KICK` | on | Where the cores' `power/autosuspend_delay_ms` is writable, a write guard sets it to 0, waits for the power domain to suspend, and restores it. Only `0` exactly turns this off, which leaves a timed idle. Without write access it falls back to the idle. Probed once |
| `ROCKET_RK3576_MM_NT` | 2048 (the profile's) | The widest output-channel tile of the RK3576 int8 matmul entries, rounded down to 32 and capped at N. The planner still halves a tile that does not fit. Read per call |
| `ROCKET_RK3576_MM_KSPLIT` | off | `rocket_matmul_int8_rk3576` and its `_perc` and `_perc_sa` forms refuse a K with no single-task plan (K >= 6176). A value not beginning with `0` makes them run it through `rocket_matmul_int8_rk3576_i32` and requantize on the host. The `_wbo` form still refuses. Read per call |
| `ROCKET_RK3576_MM_NARROW` | on | The RK3576 int8 matmul re-plans a tile whose job retires with its output partly written at half its width, down to 32 channels, and keeps that width per K for the process. With the re-plan, a K where a job stalls at a fixed output channel still produces a result. `0` makes those K refuse. An integer switch, so `on` turns it off. Read once |
| `ROCKET_RK3576_BO_POOL` | off | A nonzero integer keeps the RK3576 matmul entries' transient BOs on a per-process free list, instead of allocating them per call. A pooled BO keeps its size and is synced whole, so a call's cost depends on what ran before. Call `rocket_rk3576_bo_pool_drain(fd)` before closing the fd. `on` parses as `0`. Read once |
| `ROCKET_RK3576_BO_POOL_MB` | 64 | The cap on pooled bytes, in MiB, under `ROCKET_RK3576_BO_POOL`. A negative value selects 0. Read once |
| `ROCKET_RK3576_BO_POOL_MAXOVER` | 2 | Under `ROCKET_RK3576_BO_POOL`, a pooled BO serves a request only if it is at most this many times the request or at most 64 KiB larger. `0` lets any larger BO serve, which costs a small request the large BO's cache maintenance on every sync. A negative value selects the default. Read once |
| `ROCKET_RK3576_PC_MAX_ERR` | 0.01 | The worst-case relative gain error the RK3576 per-axis convolution's tile search aims for. It takes the widest output-channel tile that meets it, and a narrower tile costs submits. A value of 0 or below selects 0.01. Read per call |
| `ROCKET_RK3576_PC_OC_TILE` | planned | Forces the RK3576 per-axis convolution's output-channel tile, capped at the tile the CBUF allows. Read per call |
| `ROCKET_RK3576_PC_ALLOW_CLAMP` | off | The RK3576 per-axis convolution refuses a live channel below 1/65534 of its task's largest live scale. A value not beginning with `0` accepts it. Caution: that channel then runs at the wrong gain, in a complete output of the right size. Read per call |
| `ROCKET_RK3576_POOL_LAG_ATTEMPTS` | 32 | How many redos the RK3576 padded average pool's divisor-lag check gives a call, each one a resubmit. A value outside 1-4096 selects 32. Read per call |

### Diagnostic and reverse-engineering overrides

A caller does not set these. Each restores an older program for comparison, removes a
correctness check, or exposes a hardware behavior for a probe.

| Variable | Default | What it does |
|---|---|---|
| `ROCKET_MM_STREAM_KT` | unset | Restricts a streamed plan to this Kt, rounded down to the K granularity, so each candidate can be timed. A Kt that leaves no M tile falls back to the planner's choice. Read once |
| `ROCKET_OUT_CVT_ROUND` | unset | Overrides `OUT_CVT_SHIFT` bit 30 on every integer requant of both encoders: `0` rounds ties to even, `1` away from zero. It exists for `tests/requant_round_probe.c`. Read per emission |
| `ROCKET_CONV_DW_GROUP` | per program: 32 fp16, 64 int8 | Overrides the RK3588 depthwise channel group in the generators and the host scatter, for a sweep. Only the defaults are measured bit-exact. Read per call |
| `ROCKET_LUT_QFLOOR` | 1 | The smallest quantized entry of the shifted LUT tables. `0` reproduces the hardware's decode of a 0 entry to about 4.0. Read per call |
| `ROCKET_LEAKY_NOREPAIR` | unset | Any value skips `rocket_leaky_relu_fp16`'s host repair of the band around x = 0, exposing the LUT's sign-mux spike there. Read per call |
| `ROCKET_ELU_NOREPAIR` | unset | Any value skips the same repair in `rocket_elu_fp16` and `rocket_selu_fp16`. Read per call |
| `ROCKET_ACT_WIDE_LUT` | unset | Any value runs HardSwish, SiLU and GELU as the single-pass wide-output LUT instead of the two-pass path, and Mish then returns `ROCKET_E_UNSUPPORTED`. The single-pass GELU has an error spike in its flat negative tail. Read per call |
| `ROCKET_SIGLIP_GELU_SCALAR` | unset | Any value computes the SigLIP host GELU per element instead of through its 65536-entry table. The two are bit-identical. Read per call |
| `ROCKET_RK3576_MM_PC_SHIFT` | on | The RK3576 per-column int8 matmul (`rocket_matmul_int8_rk3576_perc` and its `_sa` and `_wbo` forms) programs each tile's gain in the DPU shift word, so its per-column multiplier spans the int16 field at any K. `0` restores the shift-0 ramp, whose multiplier the int32 product caps in the low hundreds at a GEMM's depth. Only a value that begins with `0` turns it off. Read per call |
| `ROCKET_RK3576_PC_SHIFT` | on | The RK3576 per-axis convolution programs each channel's gain in the BS shift word, and a channel whose output cannot vary as its byte. `0` restores the shift-0 ramp for comparison. It is independent of `ROCKET_RK3576_MM_PC_SHIFT`. Read per call |
| `ROCKET_RK3576_PC_INERT` | on | Under `ROCKET_RK3576_PC_SHIFT=0` only: accepts a clamped channel whose all-zero filter saturates under both gains. `0` refuses every clamp, and `dw` accepts on the depthwise path only. Read per call |
| `ROCKET_RK3576_MM_PHASE` | on | The RK3576 int8 matmul's stall plan, which keeps every task clear of the output channel where it would stall. `0` turns it off and leaves `ROCKET_RK3576_MM_NARROW` to find the width. An integer switch, so `on` turns it off. Read once |
| `ROCKET_RK3576_FP16_F32OUT` | on | `rocket_conv2d_fp16_rk3576` writes fp32 through a whole output stage. `0` restores the fp16 writer, which prevents the next submit from writing and so takes a power cycle per program. Read per call |
| `ROCKET_RK3576_I32_DENSE` | on | `rocket_matmul_int8_rk3576_i32` runs on the dense writer. `0` selects the older narrow and wide writers, which prevent the next submit from writing. Read per call |
| `ROCKET_RK3576_I32_OC_MULT` | unset | Any nonempty value forces an older int32 writer and turns off the dense one: `2` the wide writer, any other value the narrow one, `0` included. Read per call |
| `ROCKET_RK3576_I32_SENTINEL` | on | The RK3576 matmul and pool write guards stamp each output with a sentinel before a task, so an unwritten atom is found and redone. `0` stamps zeros, which finds only a task that wrote nothing. An integer switch, so `on` turns it off. Read per call |
| `ROCKET_RK3576_MM_GAP_MS` | from the autosuspend delay | The idle, in milliseconds, a write guard waits for the power domain to cycle when it cannot drive the cycle, and the bound on a driven cycle's wait. Unset, it is twice the driver's autosuspend delay plus 5, within 5-300, or 150 if the delay is unreadable. Read per call |
| `ROCKET_RK3576_CHAIN_MAX_TASKS` | 512 on interface 1.4 or later, else 34 | The longest program stream an RK3576 chain accepts (`rocket_conv2d_int8_chain_max_programs_rk3576()`). A value outside 2-512 selects the default. Read once |
| `ROCKET_RK3576_CHAIN_POOL_LAG` | off | A value not beginning with `0` admits a padded average pool whose divisor can lag as a chain run node. Read once |
| `ROCKET_RK3576_POOL_LAGCHECK` | on | The RK3576 int8 pool's divisor-lag check and redo. `0` turns it off, which returns a lagged output. An integer switch, so `on` turns it off. Read per call |
| `ROCKET_RK3576_POOL_LAG_DISCR` | off | A nonzero integer makes the divisor-lag check also count the scored elements whose average rounds differently under the right divisor and the previous window's, which `rocket_pool_int8_rk3576_lag_discr()` returns. It adds a second rounding per scored element. An integer switch, so `on` leaves it off. Read once |

### Chip selection

At device open the library resolves a hardware profile from the NPU's device-tree
`compatible`: the driver-bound per-core device under `/sys/bus/platform/drivers/rocket/`,
else `/proc/device-tree/compatible`. A profile is the machine parameters, meaning CBUF banks
and bank size, tile caps, weight tile groups, the datatype menu and the worker default.

Two profiles exist. `rk3588` is the fully validated target, and every operation this library
offers runs there. `rk3576` contains machine parameters measured on the RK3576, which has its
own geometry-register encoders for these paths:

- Int8 direct and depthwise convolution, and the packed-image first conv
- fp16 convolution
- The int8, int32 and fp16 matmuls
- Int8 pooling
- The int8 LUT activations

Selecting it logs a warning naming the paths it leaves unserved.

A profile selects machine parameters only. It does not switch the regcmd encoder, and the
CNA/CORE/DPU geometry-register **encoding** is IP-revision-specific. On RK3576 silicon an
RK3588 program completes with the output buffer untouched. Every RK3588 generator therefore
refuses on the RK3576 rather than submit, and the entry that called it returns an error. A
Rockchip NPU with no profile at all warns and runs with RK3588 parameters.
`ROCKET_CHIP=<name>` forces a profile, and a name with no profile warns and falls back.

The profile's own entry points, from `rocket_hw_profile.h`:

| function | what it is |
|---|---|
| `rocket_hw_current()` | Returns the active profile, resolved once on the first call in the order above. A chip that matches no profile falls back to the RK3588 profile with a warning, or with a debug-level log when no `compatible` can be read. Code holding a context reads `ctx->hw`, which is set to this at initialization |
| `rocket_hw_by_name(name)` / `rocket_hw_rk3588` / `rocket_hw_rk3576` | Returns the profile whose `name` matches, such as `"rk3588"`, or NULL. `ROCKET_CHIP` resolves through it, so a name it does not know cannot be forced. `rocket_hw_rk3588` covers every datatype and op path. `rocket_hw_rk3576` is measured on the hardware, and its own encoders cover int8 direct and depthwise conv, fp16 conv, the int8, int32 and fp16 matmuls, pooling and the LUT activations. Its `select_warning` names what they leave unserved |
| `rocket_hw_dtype_supported(p, precision)` / `ROCKET_DT_BIT(prec)` | Returns 1 if a `precision_*` encoding can run on profile `p`, from its `dtype_supported` mask, and 0 otherwise. `ROCKET_DT_BIT` is a precision's bit in that mask. On the RK3588 the mask is all ones, so the check never rejects. On a chip that lacks an encoding, the caller returns `ROCKET_E_UNSUPPORTED` |
| `struct rocket_hw_profile` | The CBUF bank count and bank size, the matmul tile cap, the weight tile groups (exposed, and not yet read by the tiling loops), the datatype mask and the selection warning. It also has a suggested worker count no library code reads, the default slow-wait mark, and `backstop_ms`: how long a job runs before the driver retires it, 500 on the RK3588 and 125 on the RK3576 |

### CPU affinity for multi-pool processes

`ROCKET_CPU_AFFINITY` takes a CPU list like `4-7`, or `off`. It sets the per-process
big-core set the pack and readback workers pin to, auto-detected as the top-cpufreq cores
otherwise.

Several context pools can run concurrently in one process, such as one detector instance
per camera, each on its own thread. For those, `rocket_affinity_set_base(int)`
(`rocket_npu.h`) sets a per-thread rotation base. Each pool's workers then start at a distinct core,
`big[(base+worker)%n_big]`, instead of every pool starting at `big[0]`. The convention is
`rocket_affinity_set_base(pool_index * nthreads)` before the thread's `*_ctx_create`, matmul
and conv calls.

The base is thread-local and honored by the fp16/int8 matmul and conv-pool worker paths,
the pool-relevant ones a thread spawns. It is a scheduling hint only and never changes
numerics, and the default 0 starts every pool's workers at `big[0]`. The spread is
deterministic, and `ROCKET_DEBUG` logs each `worker N (base B) -> cpu C`.

Each per-call worker is created on its core rather than moved there after it starts. A thread
inherits its creator's CPU mask, so a caller pinned to one core would otherwise start every worker
on that core. Each worker queues there before it can pin itself away, and a frontend that pins its
compute thread incurs that delay on every call. Depth Anything v2-small through an ONNX Runtime EP runs ten
per-node convolutions beside a pinned encoder. Created in place, those convolutions run 1.01-1.04x
their back-to-back time instead of 1.27-1.29x. The model's wall-clock time is 0.969-0.981x, and pix2pix's,
whose caller is not pinned, 0.926-0.958x (RK3588 at ~816 MHz, governor `performance`, three passes
against `ROCKET_PIN_AT_CREATE=0` in one binary).

The rotation base's throughput gain depends on the operating point. At submit-bound matmul
shapes an in-process pool is **NPU-wait-bound**, because workers block on the fence. The pool
then scales ~`n_core` even with every worker collided on one core. The base therefore gives
deterministic placement and robustness under contention rather than a large speedup on an
idle board (`tests/ctx_pool_throughput`).

### Batched submit and chaining

`ROCKET_BATCH_SUBMIT=1` runs a tiled matmul's output tiles as **one HW kick**. The per-tile
regcmds are laid contiguously and self-chain, each task's trailer linking to the next. The NPU
streams through them and raises a single completion interrupt instead of one submit and IRQ
per tile. It cuts the fixed dispatch overhead of jobs that decompose into independent tiles,
and is bit-exact with the default per-task path. The same chaining backs
`rocket_matmul_fp16_batch` (and so `ROCKET_FA_CHAIN`), where the contiguous self-chaining
spans a batch of same-shape matmuls rather than one matmul's tiles.

Chaining is a **joint layout contract with the kernel**. Userspace self-chains the regcmds
and the kernel sets `TASK_NUMBER = task_count`, so both halves must agree. A kernel whose job
struct has no flag word handles the flag one of two ways. It ignores the flag and runs a
self-chained layout down the per-task path, which stalls or corrupts the job. At interface 1.0
it refuses the submit with -E2BIG instead.

`rocket_batched_submit_supported()` (`rocket_npu.h`, probed once and cached) reports whether
the running kernel honors it, and the library conditions every chaining entry point on it.
Asking for `ROCKET_BATCH_SUBMIT=1` on a kernel without the `patches/rocket` batched-submit
patch warns and runs the stock per-task path rather than producing wrong output. Call it before
self-chaining anything yourself.

`ROCKET_KACC_CHAIN` (default off) extends that chaining **across the fp16 K-accumulation
ki-steps**. Instead of one fenced submit per K-tile, it chains a tile's whole nKt-step
accumulation into one self-chained kick. Each `ki>0` task EW-adds the partial that an earlier
task in the *same kick* just wrote. The NPU honors that in-kick read-after-write, so
it is **byte-exact** to the per-ki path, and its test forces it across nKt=2…43. It is fp16
only, because the K-accumulation it chains is the fp16 EW add.

Its gain is marginal and depends on the shape. Collapsing the fences trims submit and sync,
and the ki-steps are serially dependent, so a chained kick only pipelines the *independent*
tiles within each ki-block. Chaining is a net gain only when `gcap = 64/nKt >= 3`, about 5% at
nKt~12-21, and a wash to a loss below that (`wait` +18% at nKt=40/gcap=1).

`=1` is therefore **adaptive**: it engages only where chaining gains and otherwise falls back
to per-ki, never regressing. `=2` forces chaining for any fitting nKt, which is the
correctness test's strict gcap=1 path. On the whole-tile plan the common Gemma FFN-down has
nKt 30 (40 with `ROCKET_MM_ASYM=0`) and falls back at either. That is why its end-to-end LLM
gain measured ~nil.

The streamed plan,
the fp16 default, runs that shape at M 512 in Kt 1120, nKt 14 and gcap 4, where `=1` engages.
**Not characterized:** the gain of `=1` on the streamed plan. `tests/matmul_kacc_chain_rocket.c`
tests it on both plans, and `matmul_kacc_chain_bench` A/Bs it.

`ROCKET_CONV_BATCH=1` (default off) coalesces a tiled int8/uint8 direct conv's per-tile
submits into one multi-task job (`conv2d_int8_batch_tiles`). It is the **gapped** form,
distinct from the chaining above. The kernel runs the tiles as separate HW kicks. The whole
tile set costs one submit syscall, one fence and one IOMMU attach instead of one per tile.

Each tile is placed at a bank-aligned, zeroed slot of the batched BOs, so its feature-DMA
over-read reads zeros. It is bit-identical to the per-tile path across nt=1/2/4.

It does not change single-stream speed. A tiled conv's wall-clock time is set by the host cube
scatter and de-tile rather than the fixed submit overhead. It gains only under multi-process
contention on conv-tile-heavy work, where several pool contexts share the submit and IOMMU
path. The gain is +7.6% aggregate at P=4 on a conv-tile-heavy unit, and about 0 on conv-tile-light
MobileDet. It needs no kernel patch, being gapped rather than chained, and the matmul path
already batches its own tiles.

### Matmul tiling plans

Two plans tile an fp16 matmul, and the planner chooses one per call. The streamed plan is the
default. The whole-tile plan is what `ROCKET_MM_STREAM=0` or a tile override selects.
The int4 and bf16 planners use it, and so does the int8 planner under `ROCKET_MM_STREAM_INT8=0`.

`ROCKET_MM_STREAM` (**default on**, `=0` opts out) selects the fp16 planner's streamed plan.
The whole-tile plan keeps each task's input and weight tiles in the CBUF together, which caps
Nt at 256. The streamed plan keeps the input tile resident and streams the weights through the
CBUF's weight banks. One task then covers up to 4096 output columns. `512×3840×4096` takes 12
tasks instead of 512. Any of `ROCKET_MM_MT`, `ROCKET_MM_NT` or `ROCKET_MM_KT` also selects the
whole-tile plan.

The streamed plan costs one 256-row slice of M, so every M of 256 or more plans the same Kt and
Nt. A resident weight packed at one row count therefore serves every prefill row count.

The plan's cost model charges each task the larger of its weight stream and its MACs. It charges
each K pass its partial output, eight times over when the host sums the passes (int8, or fp16
with `ROCKET_KACC=0`). It was fitted to a sweep of every streamed Kt at twelve shapes. Against
a byte-count model at 44 shapes it is 1.06x faster in fp16 and 1.23x in int8 (geo-mean). It is
slower at one shape, fp16 3584×1024, at 0.87x [HW sweep 2026-10-21].

At M 512 on five workers, the resident entries' streamed plan is 1.10-1.51x faster than the
whole-tile plan at LLM projection shapes in fp16. In int8 it is 0.99-2.43x. The int8 gain
is largest where the streamed plan removes K passes. The host sums every pass's int32
partial. `tflite-rocket` routes its detectors' small 1x1 convolutions to the same resident int8
entry. There the streamed plan computes byte-identical outputs, and an inference takes
0.985-1.007x the whole-tile plan's time on three detectors [HW sweep 2026-11-06].

End-to-end prefill through `ggml-rocket` on three cores, pp512, against `ROCKET_MM_STREAM=0`:

| Model | Gain |
|---|---|
| Llama-3.2-3B-Instruct F16, resident weights | 1.134-1.143x |
| Llama-3.2-3B Q4_K_M, quantized weights held resident | 1.163-1.167x |
| Ministral-3-3B F16, resident weights | 1.120-1.153x |
| Qwen3.5-9B Q4_K_M, quantized weights held resident | 1.069-1.110x |

**Conditions:** RK3588, NPU clock 928 MHz (the 900 MHz request), kernel 7.2.9, `rocket` 1.3.0,
CPU governor `performance`, three passes with the order rotated, 2026-10-21 [HW sweep].

Every result is bit-exact against its reference. Llama-3.2-3B F16's wikitext perplexity is
10.9135 with the whole-tile plan and 10.9120 with the streamed one.

`ggml-rocket`'s resident int8 route (`ROCKET_INT8=1 ROCKET_INT8_RESIDENT=1`) prefills 1.49-1.57x
faster on the streamed int8 plan than on the whole-tile one [HW sweep 2026-11-06]. That is
Llama-3.2-3B-Instruct and Ministral-3-3B-Instruct F16 at pp512 and pp2048, under the conditions
above. Its wikitext perplexity is equal on both plans at every chunk.

`gen_gemm_fp16` and `gen_gemm_int8` refuse a streamed task whose weight banks number fewer
than `npu_stream_weight_banks()` returns, with -3. `gen_gemm_int8` also refuses one at exactly
that count. Such a task can fail to complete, and the `rocket` driver's watchdog retires it with
its fence signalled and its output unwritten. Both also refuse `weight_reuse` on a streamed
task.

`ROCKET_MM_ASYM` (**default on**, `=0` opts out) applies only to the whole-tile plan, which an
fp16 call takes under `ROCKET_MM_STREAM=0` or a tile override. Under the streamed default no
fp16 shape reaches it [host-computed plans, 210 shapes]. The whole-tile planner caps Mt and Nt
at 256 and maximizes Kt, which picks a symmetric Mt=Nt=256 tile (Kt=384). For a shape that
tiles both N and K, halving Nt to 128 frees CBUF so Kt grows to 512, with Mt staying at 256.

The asymmetric Mt>Nt tile runs the NPU datapath measurably faster. It is **+6-9% warm** on
the square and large-K prefill matmuls (1024²/2048²×4096, Gemma FFN-down), and about a wash on
FFN-up. End-to-end warm pp2048 through llama.cpp, measured while the whole-tile plan was the
fp16 default:

| Model | Gain |
|---|---|
| Qwen3.5-9B-F16 | +9.5% |
| Gemma-4-12B-F16 | +5.7% |
| Qwen3.5-9B-Q4_K | +1.3% |

Quantized prefill is dequant-bound, so the datapath gain dilutes to about noise there. The
option is a gain or a wash on every shape and model tested, never a regression [HW sweep].

The gain is a `wait`-term datapath effect rather than fewer fences. Submit time rises a little
as nNt doubles, while wait drops ~10% and dominates. It applies only when N is N-tiled (N>256,
no `ROCKET_MM_NT` override) **and** the symmetric plan K-tiles (nKt>1). A bigger Kt has no
effect at nKt=1, so small-K and small-N shapes are an exact no-op.

Where it fires, the option changes both Nt (256 to 128) and Kt (384 to 512). Kt sets where
the K partials are cut. Each partial is rounded to fp16 before it is summed, on the NPU at the
`ROCKET_KACC` default and on the host with `ROCKET_KACC=0`. The two settings therefore round
at different points, and their outputs can differ. Composes with KACC.

`matmul_correctness_matrix_asym` and `matmul_correctness_matrix_sym` test each setting on its
own under `ROCKET_MM_STREAM=0`, at 512×3840×4096 on random fp16 data, against an fp32
reference. Each must reach cosine similarity 0.9995 and keep every element within an fp16 bound
that grows with nKt. Neither test compares the two settings.

## Logging (`rocket_log.h`)

Every diagnostic the library emits flows through one channel: errors, warnings, the
`ROCKET_MM_PROFILE` breakdown and the `ROCKET_DEBUG` traces. A host application can therefore
intercept, redirect or silence it, instead of raw `stderr` writes polluting its own output.

With no hook installed the default sink writes `ERROR`/`WARN`/`INFO` to `stderr` and drops
`DEBUG` (errors always print, traces only under `ROCKET_DEBUG`). A host overrides this much like `ggml_log_set_callback`:

```c
#include "rocket_log.h"

static void my_sink(rocket_log_level level, const char *text, void *user) {
    /* route into the host's own logger, drop it, etc. */
}

rocket_log_set_callback(my_sink, /*user_data=*/NULL);  /* NULL restores the stderr default */
rocket_log_set_level(ROCKET_LOG_WARN);                 /* or via the environment, below */
```

The threshold is read once from the environment on first use. `ROCKET_LOG_LEVEL` takes `error`,
`warn`, `info` or `debug`, or `0`..`3`, and `ROCKET_DEBUG` raises it to at least `debug`.
`rocket_log_get_level()` returns the threshold in force, reading the environment first if
nothing has set it yet. Installing a callback is safe while other threads log: the callback and
its `user_data` are published together. A message already in flight can still reach a replaced
pair, so the host must keep it valid.

`ROCKET_LOG_STDERR` (set to any non-`0` value) additionally tees every emitted line to
`stderr`, but only when a host callback is installed, so the default sink never
double-prints. It serves a host that silences its own logger. `llama-bench`, for one,
installs a no-op `ggml` callback unless `-v` is passed. A host that adopts the channel
forwards it into `ggml`. Without `ROCKET_LOG_STDERR`, such a callback then swallows every
rocket diagnostic, including one-shot decisions like the resident-weight budget that change
the benchmarked number.
