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
  rocket_matmul.h              the matmul API (fp16 / int8 / int4 / int16 / bf16 / tf32; tiled / mt / resident / streaming)
  rocket_conv.h                general 2D conv API (fp16 CONV_2D + native depthwise + ConvTranspose2d; per-call or resident-BO ctx)
  rocket_pool.h                on-NPU MaxPool / AveragePool via the PPU pooling engine (fp16 + int8/uint8)
  rocket_reduce.h              spatial GlobalAvgPool/Mean + GlobalMax/MinPool (ReduceMax/Min) over [H,W] (PPU) + feature-axis reduce + cumsum (prefix sum) over the hidden axis (ones-/triangular-matmul)
  rocket_norm.h                on-NPU RMSNorm + LayerNorm + the per-row broadcast scale primitive
  rocket_normvision.h          on-NPU vision normalization: BatchNorm / GroupNorm / InstanceNorm / L2-Normalize
  rocket_ffn.h                 on-NPU gated FFN block (GeGLU/SwiGLU core + the three projections)
  rocket_softmax.h             on-NPU row-wise softmax + LogSoftmax + stable cross-entropy (EXP LUT + row-sum reduce + per-row scale / host gather)
  rocket_attn.h                on-NPU multi-head self-attention (Whisper encoder) + masked GQA flash attention (decoder/LLM prefill)
  rocket_encoder.h             one Whisper/transformer encoder block (LN->MHA->residual->LN->FFN->residual)
  rocket_siglip.h              SigLIP-B/16 vision encoder end-to-end (patch-embed + pos + 12xblock + post-LN)
  rocket_modernbert.h          ModernBERT text encoder end-to-end (RoPE, global/sliding attention, GeGLU) + a pre-norm post stack, variable-length batch
  rocket_activation.h          elementwise fp16 activation via the DPU LUT (sigmoid/hardsigmoid/tanh/silu/gelu/leaky/sqrt/rsqrt/recip/exp/log; auto-tiled for any n)
  npu_matmul.h                 low-level matmul_params_t + the regcmd generators
  npu_activation.h             low-level lut_act_params_t + gen_lut_activation_fp16 / gen_ew_mul_fp16
  npu_hw.h, npu_cna.h, npu_dpu.h   register/precision/layout constants

src/
  rocket_npu.c                 the shim over /dev/accel/accel0
  rocket_hw_profile.c          the RK3588 and RK3576 hw-profile instances + rocket_hw_current() accessor
  npu_regcmd.c                 the regcmd generators: gen_matmul_fp16 / _int8 / _int4
                               / _int16 / _bf16 / _tf32 (fp16 EW K-accum + int8/int4
                               + the float bf16/tf32 + int16 paths) + gen_conv2d_fp16 /
                               _dw_fp16 + gen_conv2d_int8 / _dw_int8 (native int8 CONV_2D:
                               direct int32-raw + depthwise int8-out on-chip-requant, both
                               HW-validated, with runtime wrappers)
  rocket_conv.c                general fp16 CONV_2D + native depthwise: SAME/VALID pad,
                               OC%16 pad, OC/OH/OW + DW channel/spatial tiling, resident-BO ctx
  rocket_conv_transpose.c      ConvTranspose2d (deconv): host dilate+flip lowering onto
                               the forward conv (direct + depthwise; stride/pad/opad/dilation),
                               the CNA deconvolution path, and the resident entry (col2im
                               over the resident fp16 matmul in channel-planes form)
  rocket_resize.c              nearest / bilinear upsample via depthwise ConvTranspose
                               (box / separable-triangle kernel; FPN/decoder neck)
  rocket_pool.c                on-NPU MaxPool / AveragePool (PPU): single self-contained job
  rocket_reduce.c              spatial GlobalAvgPool/Mean + GlobalMax/MinPool (ReduceMax/Min: idempotent,
                               bit-exact), telescoping multi-pass PPU reduction, kernel cap 16, plus the
                               feature-axis reduce (ones-vector matmul, fp32-accum) and cumsum/prefix sum
                               (the same matmul widened to a triangular ones matrix; incl/excl/reverse)
  rocket_norm.c                on-NPU RMSNorm + LayerNorm (stacked-row reduce + affine fold) + per-row scale
  rocket_normvision.c          on-NPU BatchNorm / GroupNorm / InstanceNorm / L2-Normalize (vision normalization)
  rocket_ffn.c                 on-NPU gated FFN: the GeGLU/SwiGLU core act(gate)⊙up + the projections
  rocket_softmax.c             on-NPU row-wise softmax: host row-max -> NPU exp -> row-sum -> host 1/s -> NPU scale
                               (+ LogSoftmax: host log(s) + per-row ew_sub; + stable cross-entropy: logsumexp + host gather)
  rocket_attn.c                on-NPU self-attention (QKV/score/P·V matmuls + per-head softmax); masked GQA flash attention (scale->softcap->mask->softmax->P·V)
  rocket_encoder.c             one Whisper encoder block: LN -> MHA -> residual -> LN -> FFN(GELU) -> residual
  rocket_siglip_encoder.c      SigLIP-B/16 vision encoder: im2col patch-embed + pos + 12xblock + post-LN (simple + resident/prepacked)
  rocket_modernbert_encoder.c  ModernBERT encoder: resident GEMMs over a stacked batch, fp32 residual, banded sliding attention
  rocket_activation.c          on-NPU elementwise activation: the full DPU LUT family (incl. EXP), tiled for any n
                               (HW-validated) + fully-on-NPU EW mul/add/sub/div
  rocket_matmul.c              tiled matmul (CBUF-fit M/N/K tiling, host + fp16-NPU K-accum,
                               CBUF operand reuse); fp16 + int8 + int4 + int16 + bf16 + tf32 primitives
  rocket_matmul_mt.c           multicore fan-out: N split across per-core worker fds
  rocket_prepacked.c           pack-weights-once (resident fp16 weights, shared scratch)
  rocket_prepacked_int8.c      resident int8 (W8A8) weights
  rocket_prepacked_int4.c      resident int4 (W4A4) weights (per-channel + group-wise W4A4)
  rocket_affinity.c            pin pack/readback workers to the A76 big cores

tests/                         standalone tests + benchmarks (see below)
```

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

## Public matmul API (`rocket_matmul.h`)

| function | what it is |
|---|---|
| `rocket_matmul_fp16` | tiled single-fd fp16 matmul (the validated core) |
| `rocket_matmul_fp16_f32out` | fp16xfp16 -> **fp32** output: the full fp32 accumulator, with no per-K-tile or final fp16 rounding. ~5000x more accurate than fp16 output, tracking the true result to the fp32 noise floor (~1e-7). Opt-in. The output readback is 2x, which is free for tall-skinny prefill and costs +35% only when the output bytes rival the weights |
| `rocket_matmul_fp16_mt` | multicore: split N across `nthreads` worker fds (~3 cores) |
| `rocket_matmul_fp16_batch` | runs `nbatch` independent same-shape matmuls as one submit, with one fence, and one IRQ under `ROCKET_BATCH_SUBMIT`. It collapses a set of small, dispatch-bound GEMMs that share a shape. Bit-identical to per-item `rocket_matmul_fp16`. Flash attention's per-worker QK/AV submit batching uses it. `rocket_mm_batch_create/_run/_free` is the persistent form: resident in/wt/out BOs grown only on a larger shape, and a prezero-once guard that skips the full-BO zero when the `(M,K,N,nbatch)` layout repeats. For a repeated-shape caller (flash attention's per-layer QK/AV) it removes the per-call BO alloc and zero. The one-shot entry is a thin wrapper over it |
| `rocket_ctx` / `rocket_weights_pack` / `rocket_matmul_fp16_prepacked` | pack weights once, reuse across calls (resident fp16). The compute M can differ from the pack M, because the resident layout is M-independent for M>=256, so a weight packed at warmup-M is reused at prefill-M with no re-pack. An incompatible tiling returns -2 |
| `rocket_ctx_create_ex(nthreads, ROCKET_CTX_TILING_CANONICAL)` | a ctx whose resident weights are M-independent down to M=4. At M < 256 the plan keeps the M=256 plan's K and N tiling (Mt = M), so one pack serves every call M and `_prepacked` never returns -2. The cost is the K tile a small M gives up: 1.16-1.42x a pack made for that M at M 48-128, parity from 196 (`tests/matmul_prepacked_canonical_rocket.c`, every element exact over 9 shapes and M 4-640). `ROCKET_CANON_BENCH=21` prints the cost. It suits a resident encoder whose length varies per call |
| `rocket_stream` / `rocket_matmul_fp16_stream[_fused]` | streaming path for LLM prefill: re-packs B, caches scratch per shape, and fuses gate/up |
| `rocket_matmul_int8` / `rocket_matmul_plan_int8` | int8xint8->int32 tiled (pre-quantized in, raw int32 out) |
| `rocket_i8_ctx` / `rocket_matmul_int8_prepacked` | resident int8 (W8A8) weights. The compute M can differ from the pack M, because the resident tile layout is M-independent (canonical-tileM: the tiling is planned at `MAX_TILE`), so a weight packed at warmup-M is reused at any prefill-M with no re-pack (int32 K-accum is exact for any tiling -> bit-exact). An incompatible tiling returns -2. The compute M must still be a **positive multiple of 4**: an unaligned M miscomputes on the HW height geometry, so it is rejected (-1), not padded (padding M would need a matching pad of `a_scale`, which only the caller has). The caller pads ragged row counts with `rocket_pad_m` |
| `rocket_matmul_int8_groupwise` / `rocket_matmul_plan_int8_gw` | **per-K-group int8 dequant scales**, fp32-accumulated: `C_f[m,n] = Σ_g a_scale[m,g]·b_scale[n,g]·(int32 partial of K-group g)`. The primitive a **natively quantized** weight needs, since a GGUF MXFP4/Q8_0/Q4_K block has one scale per K-block, and the NPU cannot apply a K-blocked scale on-chip (at the output stage K is fully contracted). Integer partials already leave the chip at every K-tile boundary (no on-device integer K-accum is implemented), so the block scale costs nothing extra at a boundary that already exists: it fuses into the readback accumulate. The resident group-wise call measures 0.97-0.99x the resident per-channel one at M 64 and 0.81-0.90x at M 256 (`tests/moe_gw_bench.c` at a gpt-oss-20b expert, RK3588 at ~816 MHz, 2026-10-09). Unlike the int4 twin there is **no saturation bound on `group`** (the output accumulator is int32, not int16), and `Kt` need only **divide** the group, not equal it (a K-tile must lie *inside* one group, not *be* one), so a group wider than the CBUF cap stays legal. `rocket_matmul_plan_int8_gw` previews the tiling (pure, no HW) and reports the `Kt` it chose |
| `rocket_i8_weights_pack_gw` / `rocket_matmul_int8_prepacked_gw` | **resident group-wise int8**: the int8 codes are scattered into NPU BOs once and never leave, so a quantized weight costs no per-forward-pass dequant and no per-call weight scatter. Each call quantizes only A (per row, per K-group) and returns the fp32 per-group dequant. Same M-independence and the same M%4 contract as the per-channel form above. Bit-exact vs the one-shot oracle when the group fits the CBUF at the worst-case tile (which forces `Kt == group` in both paths). A wider group can get a different `Kt` divisor, and the two then agree to fp32 reassociation |
| `rocket_matmul_int4` / `rocket_matmul_int4_ex` | int4xint4->int16 tiled, accumulated to int32 on the host. `_ex` adds the int16-saturation Kt cap (in-model `[-7,7]` needs `kt_cap=480`) |
| `rocket_matmul_int4_groupwise` | per-K-group int4 dequant scales, fp32-accumulated, for W4A4 quality, with the group as the saturation-safe K-tile |
| `rocket_i4_ctx` / `rocket_matmul_int4_prepacked` | resident int4 (W4A4) weights, raw int32 out (per-channel) |
| `rocket_i4_weights_pack_gw` / `rocket_matmul_int4_prepacked_gw` | **resident group-wise int4** (the in-model W4A4 path): weight packed once with the K-tile forced to `group`, each call quantizes only A and returns the fp32 per-group dequant `Σ_g a_scale[m,g]·b_scale[n,g]·partial`. Hadamard (baked into the resident weight + applied to A by the caller) is product-preserving, so no driver support is needed. Like the int8 path, the resident layout is M-independent (canonical-tileM), so a weight packed at warmup-M is reused at any prefill-M (incl. a small short-prompt M) with no re-pack |
| `rocket_matmul_int16_exact` | bit-exact int16xint16->int64 via int8 byte-decomposition (4 int8 matmuls) |
| `rocket_matmul_bf16` / `rocket_matmul_plan_bf16` | bf16xbf16->fp32 tiled, single-fd (fp32 A/B in, truncated to bf16 on scatter, with fp32 range and no activation scaling) |
| `rocket_matmul_bf16_mt` | multicore bf16: split N across `nthreads` worker fds over the unchanged single-fd path (~3 cores) |
| `rocket_bf16_stream` / `rocket_matmul_bf16_stream` | **streaming bf16** for LLM prefill: persistent worker fds + per-shape resident scratch BOs, re-pack A/B per call (the bf16 sibling of `rocket_matmul_fp16_stream`, bit-identical to single-fd at `nthreads=1`). The in-model bf16 path. ~3.2x single-fd warm |
| `rocket_matmul_tf32` / `rocket_matmul_plan_tf32` | tf32xtf32->fp32 tiled (raw fp32 in, HW truncates to a 10-bit mantissa, the first 4-byte-input path) |
| `rocket_matmul_int8_rk3576` / `rocket_matmul_plan_int8_rk3576` | **the RK3576's own int8 matmul**, and a different contract from `rocket_matmul_int8`: `C = sat8(round((A·Bᵀ + bias)·scale))`, an int8 output through the DPU's requant rather than raw int32. The contract differs, so it is a separate entry rather than a path inside `rocket_matmul_int8`. Requires `K%32, N%32`. **M has no alignment constraint** on the RK3576, so `M=1` is correct. N is tiled, and throughput scales with it: a submit's device time rises from ~0.12 ms at N 32 to ~0.91 ms at N 2048, so the rate rises almost linearly with N (`rocket` 1.6.0, 786 MHz, 2026-09-25). `ROCKET_RK3576_MM_NT` overrides the tile. A K past one task's contraction, 6176 or more, refuses with `ROCKET_E_UNSUPPORTED`. `ROCKET_RK3576_MM_KSPLIT=1` sends it through the int32 entry below with the requant on the host. For speed, keep K at or below 4608: past 4992 the output tile drops from 1024 channels to 32. The planner and the entry refuse a dimension past 2^24, or an M whose plane is taller than the 65535 rows its 16-bit height field holds (any prime M past 65535), with `ROCKET_E_SHAPE`, and a plan of more than `INT_MAX` jobs with `ROCKET_E_TILING` |
| `rocket_matmul_int8_rk3576_perc` | **the same int8 matmul with a per-output-column scale**: `C = sat8(round_half_to_even((A·Bᵀ + bias)·scale_n[n]))`, `scale_n` N positive entries. Every shape rule and refusal of `rocket_matmul_int8_rk3576` applies. The per-column gain is programmed in the coefficient group's int16 multiplier and the task's shift word, so a column resolves its scale to 0.5/C relative, coarser where a tile's scales spread. `worst_rel_err` (can be NULL) reports the largest relative gain error the call delivered. The entry does not sort columns, and it cannot compute `scale_n`: the caller must calibrate it. Exact against a host model of the ramp (`tests/rk3576_mm_requant`) |
| `rocket_matmul_int8_rk3576_perc_sa` | `_perc` with each column's `sum_k \|B[n][k]\|` supplied as `sum_abs_w` (int64, N entries), skipping a pass over B. NULL computes it. The entry does not check the value: one that does not match B plans a wrong gain and returns an output of the right size with wrong values. Bit-identical to `_perc` |
| `rocket_matmul_int8_rk3576_perc_wbo` / `rocket_rk3576_wbo_create` / `rocket_rk3576_wbo_free` | `_perc_sa` with the weight resident: `create` packs B into an N·K-byte device BO once, and the entry takes that handle in place of B. `sum_abs_w` is required, and NULL refuses. A K or N that differs from the handle's refuses, and a K of 6176 or more refuses outright. A stale handle returns an output of the right size computed from the old weight. Bit-identical to `_perc_sa` |
| `rocket_matmul_fp16_rk3576` | **the RK3576's fp16 matmul**: `C = A·Bᵀ` with fp16 A and B and an **fp32** C, all row-major, B one row per output channel. The whole of K in one task, where the fp16 convolution contracts sixteen channels a task. Accepts `1 <= M <= 2048`, `32 <= K <= 6144` with `K%32`, `64 <= N <= 8192` with `N%16`, and returns `ROCKET_E_UNSUPPORTED` outside them. K and N are one task's measured bounds. A task contains at most `4096/(K/32)` rows, the CBUF data window, so the entry cuts M into equal tasks under that and issues them as one submit (`rocket_rk3576_mm_fp16_task_rows()`). A call whose output still holds the sentinel is redone after a power cycle, and `rocket_rk3576_mm_fp16_last_attempts()` reports how many submits the last call took. `ROCKET_MM_PROFILE` prints its buckets. Its output leaves the next submit unaffected |
| `rocket_matmul_fp16_rk3576_wbo` / `rocket_rk3576_wbo_fp16_create` / `rocket_rk3576_wbo_fp16_free` | **the same fp16 matmul with resident weights**: `create` permutes B into the RK3576's layout once and builds the coefficient buffer for its N, both in device BOs, and the entry takes that handle in place of B. Bit-identical to `rocket_matmul_fp16_rk3576`, with M free per call inside the limits. A K or N outside the limits refuses at `create`, and a call whose K or N differs from the handle's refuses with `ROCKET_E_SHAPE`. The handle is not checked against the weight it came from, so a stale one returns an output of the right size computed from the old weight. Each handle uses K·N·2 bytes of weight and about 257 KiB of coefficients |
| `rocket_matmul_int8_rk3576_i32` | **the RK3576's int32-output matmul, and its K split**: `C = A·Bᵀ + bias` in raw int32, K split internally and the partials summed on the host, so K is bounded by memory and by the 2^24 dimension cap. Requires `K%32, N%32`. It runs on the dense writer: the int8 program on a whole wide output stage, which writes every accumulator of a task as a raw int32. The entry therefore programs the real output channels (rounded to 32), runs its submits back to back and leaves the next submit unaffected. Per call 1.0 ms at 8×64×32 and 48.1 ms at 32×1024×4096, with all 21 of the test's int32 shapes exact (`tests/rk3576_matmul_gate`, `rocket` 1.6.0, 2026-09-30). M 512, K 8192, N 2048 runs in 345.7 ms, 4096 sampled elements exact. A sentinel stamp and a per-atom scan remain its write guard, and a retired job is redone after a power cycle. `ROCKET_RK3576_I32_DENSE=0`, or `ROCKET_RK3576_I32_OC_MULT=2` or `=4`, takes the older narrow and wide writers. Those deliver a quarter or a half of the programmed channels, leave the next submit writing nothing, and idle between submits for it (`ROCKET_RK3576_MM_GAP_MS`). Caution: on the narrow writer, M 512, K 8192, N 2048 leaves the RK3576 unresponsive until a reboot |

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

## Conv API (`rocket_conv.h`)

Beyond the matmul, the library runs a general 2D convolution, the basis of the
`tflite-rocket` detection delegate. A 1×1 pointwise conv is the matmul, and everything else
is the conv path. HW-validated **bit-exact**.

| function | what it is |
|---|---|
| `rocket_conv2d_fp16` | general fp16 `CONV_2D`: KxK / stride / dilation, symmetric pad, OC%16 pad, OC/OH/OW spatial tiling. Allocates and frees its 5 BOs per call |
| `rocket_conv1d_fp16(fd, ic, it, oc, kw, stride, pad, …)` | **conv1d** (the Whisper encoder front-end: width-only 1D conv over time), lowered onto `rocket_conv2d_fp16` with **time on the height axis** (IW=1, so the OH-row tiler fits CBUF, while the IH=1 width layout overflows the feature banks for Whisper IC=80/512). Whisper conv1/conv2 (KW=3, pad=1, stride 1/2) HW-validated (`tests/conv1d_rocket.c`) |
| `rocket_conv2d_fp16` (`desc.depthwise=1`) | native depthwise (`DW_EN`, group G=32 + the DPU depthwise register fixes), channel + spatial tiled |
| `rocket_conv2d_act_fp16(fd, d, kind, …)` / `_ctx` | **conv->activation fusion**: a direct fp16 conv post-processes its own CACC result with a smooth `f(x)` in the same task (the DPU LUT epilogue ported into `gen_conv2d_task`): `out = f(conv(x))`, no 2nd round-trip. `kind` = `SILU`/`TANH`/`GELU`. HW-validated: the fused epilogue **bit-reproduces the standalone LUT** (<=0.0039), conv->tanh bit-accurate (`tests/conv_act_rocket.c`). Off by default, and byte-identical when off. HardSwish/depthwise rejected (flat-tail quirk / direct-only). Caution: a narrow x~0 LE/LO boundary glitch spikes all single-pass kinds, and the 2-pass `x·gate(x)` path avoids it |
| `rocket_conv2d_int8` | native **int8 direct** `CONV_2D`: int8xint8->int32 raw accumulate (caller requants), OC pad 32 / IC pad 32, OC/OH/OW tiling. Bit-exact vs an int64 oracle. It is also the substrate for the `tflite-rocket` delegate's native **uint8** convs: the caller recenters uint8->int8 `byte−128` and folds the centering into its requant and a box-sum, with no driver change |
| `rocket_conv2d_dw_int8` | native **int8 depthwise** (int8-out, on-chip requant): per-tensor quant, any int8 weight zero point (the DPU's CPEND operand adds `-w_zp` times each window's input sum), the input zero point folded into the bias. TFLite's accumulator exactly, within one of TFLite at a rounding boundary. A uint8 tensor maps onto it by subtracting 128 from its values and zero points |
| `rocket_conv2d_dw_int8_plan` | the pure shape check behind `rocket_conv2d_dw_int8`: `ROCKET_OK` when its NPU path takes the descriptor, `ROCKET_E_SHAPE` or `ROCKET_E_UNSUPPORTED` otherwise. Opens no device, so a frontend claims a node with it |
| `rocket_conv2d_int8_q` / `_ctx` / `_mt` / `_plan` | native **int8 direct conv with the int8-out writer** (RK3588): Mesa's direct program, requantized on chip through the OUT_CVT, any per-tensor weight zero point on CPEND, the input zero point folded into the bias. Shares the int32-raw entry's planner, tiler and job body. Programs whole 32-kernel groups and materializes the halo (the input zero point on real channels, 0 on the channel padding). Within one of TFLite at a rounding boundary |
| `rocket_conv2d_int8_q_perc` / `_perc_ctx` / `_perc_mt` / `_perc_plan` | native **int8 direct conv with per-channel weight scales** (TFLite's per-axis quantization), RK3588: the int8-out program of `rocket_conv2d_int8_q` with each output channel's scale on the BS stage's int16 multiplier and one OUT_CVT gain per job. Channels are sorted by scale and cut into jobs at `ROCKET_DW_PERC_MIN_C` (2048). Each job has whole 32-kernel groups, and every job's tiles go in one dispatch across the pool. A layer that plans to one job keeps its channel order and the one-job fast path. Within one of TFLite at a rounding boundary. The host writes an all-zero filter's channel |
| `rocket_conv2d_dw_int8_perc` / `_perc_ctx` / `_perc_plan` | native **int8 depthwise with per-channel weight scales** (TFLite's per-axis quantization), RK3588: each channel's scale is programmed in an int16 multiplier the DPU's BS stage reads per channel, with one OUT_CVT gain per job. Channels are sorted by scale and cut into jobs so each keeps a multiplier of at least `ROCKET_DW_PERC_MIN_C` (2048). Within one of TFLite at a rounding boundary. The host writes an all-zero filter as its bias's constant |
| `rocket_conv_ctx_create` / `rocket_conv2d_{fp16,int8}_ctx` / `_dw_int8_ctx` | the same convs with a **resident** BO pool reused across calls/tiles (ctx=NULL ⇒ byte-identical legacy alloc-per-call) |
| `rocket_conv_pool_create` / `rocket_conv2d_int8_mt` / `rocket_conv_pool_free` | **multicore** int8/uint8 direct conv: a pool of N worker fds (each its own resident `rocket_conv_ctx`) fans the conv's independent OC/OH/OW tiles across the 3 NPU cores. **Bit-identical** to `rocket_conv2d_int8` (same tiles/jobs). It falls back to serial for single-tile convs. Mirrors `rocket_matmul_fp16_mt`'s one-fd-per-core design |
| `rocket_conv_transpose2d_fp16(fd, d, in, W, out)` / `_ctx` | **ConvTranspose2d / deconv** (learned upsampling: segmentation/decoder/super-res, FPN learned-upsample). Two paths, both bit-exact vs a scatter reference (`tests/conv_transpose_rocket.c`). On the RK3588 a **direct** transpose at power-of-two strides (2, 4 or 8), dilation 1, `IC%32==0` and a compact input that fits one CBUF pass runs on the CNA's **hardware deconvolution mode**, one task per output-channel tile: 0.35-0.54x the lowering's wall-clock time at decoder shapes. Everything else is lowered to interior-dilate-input + `rot180(Wᵀ)` + a **stride-1 forward `rocket_conv2d_fp16`**, inheriting the HW-exact conv tiling. The lowering's cost scales with the *upsampled* size. Weights `[IC][OC][KH][KW]` (in-channels first) direct or `[C][1][KH][KW]` depthwise (`desc.depthwise=1`, OC==IC, C%32). Supports stride/pad/output_padding/dilation, any OC/IC. `_plan` returns `−2` for the unimplemented `pad > d·(K−1)` crop case (clean decline) |
| `rocket_conv_transpose2d_route(d)` | which of the two paths a descriptor takes (`ROCKET_CONV_TRANSPOSE_DECONV` / `_LOWERED`), or the plan's negative refusal. Pure. `ROCKET_CONV_TRANSPOSE_HW=0` forces the lowering |
| `rocket_conv_transpose2d_weights_pack(ctx, d, W, bias)` / `rocket_conv_transpose2d_fp16_prepacked(ctx, w, in, out)` / `_weights_free` / `_prepacked_plan(d)` | **the resident-weight ConvTranspose**, for a model that runs the same layer repeatedly: one GEMM, `in^T . W` with the input read as IC planes of IH·IW pixels and the weight as stored (`[IC][OC·KH·KW]`), which is every tap of every input pixel with no inserted zeros, then a host scatter-add of each tap's plane into the output, split by output channel across the context's worker threads (`ROCKET_CT_SCATTER_THREADS`). The GEMM is the resident fp16 matmul on the caller's `rocket_ctx`, in channel-planes form (no host transpose on either side), the weight transposed and packed **once** into the handle with an optional `[OC]` bias. The workers split M (each holding the whole weight) once each has 64 rows and the weight is at most 4x the input, N otherwise (`ROCKET_CT_SPLIT=m\|n`). Any stride, pad (a cropping pad included), dilation and output_padding. Per call at pix2pix's and SAM's ten layers 0.94-9.16 ms, 0.11-0.58x ONNX Runtime's 2-thread per-node time on the same board and 0.02-0.29x the one-shot entry's, with the scatter-add on one thread [HW sweep, mainline RK1, five rotated passes, 2026-09-27]. The threaded scatter runs at 0.45-0.57x its one-thread time at up6, up7 and SAM's upscalers and 0.6-0.8x at up8 [HW sweep, RK1 at ~816 MHz, 2026-10-08]. Within a few fp16 roundings of each element's magnitude sum (worst 2^-11.1 measured), bit-exact on small-integer data (`tests/conv_transpose_resident.c`). The plan refuses depthwise and non-RK3588 parts (`ROCKET_E_UNSUPPORTED`) and a tap-plane buffer past 64 MiB (-4). The handle never re-reads W: a changed weight is a new pack |
| `rocket_conv2d_weights_pack(ctx, d, W, bias)` / `rocket_conv2d_fp16_prepacked(ctx, w, in, out)` / `_weights_free` / `_prepacked_plan(d)` / `_prepacked_estimate_ms(d)` | **the resident-weight forward conv**, for a model that runs the same layer repeatedly: a host im2col of the input into `IC·KH·KW` planes of the `OH·OW` output pixels, then one GEMM against the weight as stored (`[OC][IC·KH·KW]`), whose channel-planes output is the conv's `[OC][OH][OW]`. The GEMM is the resident fp16 matmul on the caller's `rocket_ctx`, the weight packed **once** into the handle with an optional `[OC]` bias, split across the workers by the ConvTranspose entry's rule (`ROCKET_CONV_SPLIT=m\|n`). The unfold runs on the context's worker count of threads (`ROCKET_IM2COL_THREADS`). A 1x1 stride-1 unpadded conv reads its input in place. Any stride, pad (an explicit `oh`/`ow` sets a trailing pad), dilation and kernel. Per call at pix2pix's eight encoder layers 1.1-5.8 ms (c2-c8 sum 19.7 ms), and 0.5-11.9 ms at Depth Anything v2-small's neck [HW sweep, mainline RK1 at ~816 MHz, governor `ondemand`, two passes, 2026-10-08, `tests/conv_model_bench.c`]. Within a few fp16 roundings of each element's magnitude sum (worst 2^-11.0 measured), bit-exact on small-integer data (`tests/conv_resident.c`). `_prepacked_estimate_ms` is a linear model of a call fitted to 21 of those shapes (0.78-1.20x of measured), for a frontend deciding whether to claim a node. The plan refuses depthwise and non-RK3588 parts (`ROCKET_E_UNSUPPORTED`) and an unfold or staging buffer past 64 MiB (-4) |
| `rocket_upsample_nearest_fp16` / `rocket_upsample_bilinear_fp16` (`rocket_resize.h`) | **integer-factor resize** (the FPN/decoder neck: `RESIZE_NEAREST_NEIGHBOR` / `RESIZE_BILINEAR`). Realized as a **depthwise ConvTranspose** with a fixed box (nearest) / separable-triangle (bilinear) kernel, where `pad=(k−s)/2` gives exactly `IH·s x IW·s`. Nearest = bit-exact block replication, and bilinear = half-pixel 2-tap (the triangle's stride-subsample is a partition of unity), `align_corners=False` interior + zero boundary. `C%32` (depthwise group). HW-validated vs independent gather refs + partition-of-unity / linear-exactness properties (`tests/resize_rocket.c`) |

Pure planners (`rocket_conv2d_plan` / `_oh` / `_ow` / `rocket_total_pad`, plus
`rocket_conv_transpose2d_plan` / `_oh` / `_ow`) preview the dimensions and the CBUF-fit
check without hardware. `rocket_conv2d_ref_int8` / `rocket_conv_transpose2d_ref_fp16` are golden oracles.

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
| `rocket_pool_int8_rk3576` / `rocket_pool_int8_rk3576_plan` / `rocket_pool_int8_rk3576_exact` / `rocket_pool_int8_pack_rk3576` / `rocket_pool_int8_prepacked_rk3576` / `rocket_pool_int8_free_rk3576` | **the RK3576's int8 MaxPool and AveragePool** on the PPU, `[C][IH][IW]` -> `[C][OH][OW]`. A different function from `rocket_pool_int8`: the average rounds half to even, divides by the whole window and pads with the input zero point, which the entry takes. `_exact` says without a device whether a window's average is exactly rounded. A plane wider than one task's output allowance splits by columns, up to 8 slices. Refuses rows or channels past 8192. The pack handle holds the buffers and program across calls, bit-identical to the one-shot entry. Bit-exact against `rocket_pool_ref_int8_rk3576` (`tests/rk3576_pool_probe.c`) |
| `rocket_act_int8_rk3576` | **an int8 activation on the DPU LUT**: `[C][H][W]` int8 in and out, per-tensor quant. `kind` is a `ROCKET_RK3576_ACT_*`: sigmoid, tanh, SiLU, HardSwish, HardSigmoid or ELU. Each of the 256 inputs maps to its own table entry, so the only error is that entry's quantization. The table load and the op run as two tasks of one submit (`tests/rk3576_act_gate.c`) |

## Activation API (`rocket_activation.h`)

The on-NPU nonlinear activations. Each is a standalone DPU **LUT** pass, NVDLA SDP with no
conv, that applies `f(x)` to a flat fp16 vector entirely on the NPU.

| function | what |
|---|---|
| `rocket_activation_fp16(fd, kind, in, out, n)` | elementwise fp16 activation. **`SIGMOID`** (max_abs 0.00146) + **`HARDSIGMOID`** (0.00049) run fully on the NPU LUT. **`HARDSWISH`** (0.00098), **`SILU`**, and **`GELU`** (exact erf) run the **2-pass** path `x·gate(x)` (gate = sigmoid/hardsigmoid/Φ on the NPU LUT, the multiply on host by default or **fully on the NPU** with `ROCKET_ACT_NPU_MUL=1`). **`TANH`** is single-pass. **GELU is the accurate 2-pass `x·Φ(x)`** (Φ = the Gaussian CDF on the clean unit-LUT geometry): cos=1.000000 vs true erf-GELU over `[-12,12]` (`tests/gelu_rocket.c`). The single-pass GELU spikes ~128 in the flat tail (`ROCKET_ACT_WIDE_LUT`, for reverse engineering only). All HW-validated (`tests/activation_lut_rocket.c`, `lut_tanh_rocket.c`, `gelu_rocket.c`). `n` padded to a multiple of 8 |
| `rocket_activation_fp16`, **positive-domain kinds** `SQRT` / `RSQRT` / `RECIPROCAL` | `f(x)` for `x>0` via the **shifted single-table** LUT (the whole domain maps onto the positive index half ⇒ no LE/LO glitch). The DPU LUT computes the reciprocal family. Uniform-grid ⇒ accuracy is domain-bounded: **<1% over a ~100-200x range** placed away from the steep near-0 region (HW: sqrt 0.85% / rsqrt 0.44% / recip 1.0%), set with `ROCKET_LUT_XLO/XHI`. `RSQRT` is the **RMSNorm/LayerNorm** core, and `RECIPROCAL` the softmax-denominator / Div core (`tests/recip_rsqrt_rocket.c`) |
| `rocket_activation_fp16`, **`EXP`** | `exp(x)` via the same shifted single-table, default domain `[-16,0]` (the **softmax** case: input <=0 after the row-max subtraction). Works on the standalone flying path (unlike GELU, with no LE/LO sign-mux glitch). exp's relative interp error is ~constant (`Δ²/8` ~1e-4). A LUT table entry of exactly **q=0 mis-decodes to a wrong ~4.0**: exp's deep tail quantizes to q=0 and decodes to ~4, so every shifted-table entry is **clamped to q>=1** (`ROCKET_LUT_QFLOOR`, default 1, a no-op for sqrt/rsqrt/recip). softmax-sum end-to-end rel <=0.04% (`tests/exp_lut_rocket.c`) |
| `rocket_activation_fp16`, **`LOG`** | `ln(x)`, `x>0`, the natural inverse of EXP (log-probabilities / NLL / cross-entropy). Same shifted single-table, but the **first signed-output kind on the positive-domain path**: `log(x)<0` for `x<1`, so `out_lo=log(x_lo)` is negative and the OUT_CVT offset decodes the signed range (the tanh/ELU mechanism). Default domain `[0.25,32]`. Uniform-grid ⇒ **absolute** error is the metric (relative is ill-defined at the `x=1` zero crossing). HW **max_abs 0.0066 / mean 0.0007** over `[0.3,30]`, worst at the steep small-x end (`tests/recip_rsqrt_rocket.c`) |
| `rocket_leaky_relu_fp16(fd, alpha, in, out, n)` | **LeakyReLU** `x>=0?x:alpha*x` (YOLO, ONNX LeakyRelu/PRelu scalar slope), one DPU LUT pass on the natural LE/LO split (LE=`alpha*x`, LO=`x`). `alpha>0`, exact over [-R,R] (R via `ROCKET_LEAKY_R`, default 16), saturates beyond. The LUT's **sign-based LE/LO mux** spikes at exactly x~0 -> repaired on the host (the readback streams every element anyway). HW-validated alpha 0.01-0.5 (`tests/leaky_relu_rocket.c`) |
| `rocket_activation_fp16`, **`SOFTPLUS`** / **`MISH`** / **`ABS`** | **Softplus** `log(1+e^x)` (shifted single-table, `out_lo=0`, x~0-clean, max_rel 0.14%). **Mish** `x·tanh(softplus(x))`, the **YOLOv4/v7** backbone activation, 2-pass (a `[0,1]` gate LUT + EW-mul, max_rel 0.06%). **Abs** `\|x\|` (symmetric shifted single-table, the x=0 kink on the middle sample, max_abs 1e-3). All HW-validated (`tests/softplus_mish_rocket.c`) |
| `rocket_elu_fp16(fd, alpha, in, out, n)` / `rocket_selu_fp16(fd, in, out, n)` | **ELU** `x>=0?x:alpha*(e^x-1)` and **SELU** `λ·ELU_α` (fixed self-normalizing α=1.673, λ=1.051), on the **symmetric shifted single-table**. Negative outputs ⇒ a signed OUT_CVT, which keeps the x~0 mux spike -> **host x~0 repair** (`ROCKET_ELU_NOREPAIR` disables). Exact over [-R,R] (`ROCKET_ELU_R`, default 8). HW-validated alpha 0.5-2.0 (`tests/elu_rocket.c`) |
| `rocket_ew_add_fp16` / `rocket_ew_sub_fp16` / `rocket_ew_mul_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise **add** (residuals) / **subtract** / **multiply** (gated activations), flat fp16 vectors. All use the conv-main EW path (identity-matmul main feed + ERDMA operand, `EW_OP_TYPE` add/mul). SUB reuses the ADD datapath with the operand negated (`a-b==a+(-b)`, exact fp16 sign flip). n is reshaped to [M,32] and M-tiled. **Bit-exact** vs host (`tests/ew_mul_rocket.c` runtime check sweeps add/sub/mul, n up to 40000 across the M-tile boundary) |
| `rocket_ew_div_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise **divide** `a/b` = `reciprocal(b)` (DPU LUT) then `a*recip` (EW mul). `b` positive, within the reciprocal LUT domain. fp16-LUT-approx (HW max_rel 0.35% for `b∈[0.5,8]`). Covers TFLite/ONNX `DIV` (`tests/recip_rsqrt_rocket.c`) |
| `rocket_ew_max_fp16` / `rocket_ew_min_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise two-tensor **max/min** `out=max/min(a,b)`. The DPU EW **ALU algo** field (`DPU_EW_CFG` bits[17:16]) reaches **MAX(0)/MIN(1)**, not just SUM(2=add): same conv-main EW datapath, one word changed. **Bit-exact**, because they select an operand (`tests/ew_minmax_rocket.c`, n to 40000). Covers TFLite/ONNX `Maximum`/`Minimum` and `ReLU=max(x,0)` |
| `rocket_clip_fp16(fd, lo, hi, in, out, n)` | **Clip** `min(max(x,lo),hi)` on the NPU (constant-operand MAX then MIN, two EW passes). Bit-exact. Covers TFLite/ONNX `Clip` and the bounded-ReLU family (ReLU6=`Clip(0,6)`) (`tests/ew_minmax_rocket.c`) |
| `rocket_prelu_fp16(fd, C, S, x, alpha, out)` | **PReLU** with a **per-channel** slope `alpha[C]` (YOLO/segmentation, ONNX PRelu), input `[C][S]`. **No LUT**, so no x~0 glitch. For the universal `alpha∈[0,1]` it is `max(x, alpha_c·x)` (per-channel scale via row-broadcast `ew_mul`, then `ew_max`), in 2 passes. Any `alpha` outside [0,1] falls back to `relu(x)+alpha_c·min(x,0)`. **Bit-exact** (`tests/prelu_rocket.c`) |
| `rocket_lut_epilogue_build(kind, lut, ep)` | builds the LE/LO tables + DPU epilogue constants for a smooth single-pass kind (SiLU/tanh/GELU). The **single source** of the single-pass LUT params, shared by the standalone op and the **conv->activation fusion** (`rocket_conv2d_act_fp16`), so the two cannot diverge |
| `rocket_ew_mul_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise fp16 multiply `out=a*b` (identity-conv main feed + `EW_OP_TYPE=1`, M-tiled). Bit-exact (`tests/ew_mul_rocket.c`). The building block of on-NPU HardSwish/SiLU |
| `gen_lut_activation_fp16` (`npu_activation.h`) | low-level: the flying-mode DPU LUT regcmd (LE/LO hybrid tables, BN-mul index, OUT_CVT Q0.15->fp16). The same epilogue is fused into `gen_conv2d_task` (`lut_epilogue_t` / `npu_dpu_desc.lut_en`) for conv->activation |

### Vector length, GELU and the elementwise multiply

A LUT op lays the vector out as a cube of `cols = n/8` width positions, and
`DPU_DATA_CUBE_WIDTH` is 13-bit, so one op is limited to `n <= 65528`. `run_dpu_lut` tiles
automatically, so every activation works at any `n`, including a transformer's `[M,I]` cube of
millions of elements. Running at the exact maximum width (`cols = 8191`) corrupts ~54 cube
positions, so the tile cap (32768) stays well under that maximum and is bit-clean.

**GELU** runs the accurate **2-pass** `x·Φ(x)`. The single-pass GELU spikes ~128 in the flat
negative tail, which also makes a *fused* single-pass matmul->GELU spike for wide FFN inputs.
The 2-pass form is the on-NPU GELU path.

A fully-on-NPU elementwise multiply (`rocket_ew_mul_fp16`) is implemented and HW-validated
bit-exact (`tests/ew_mul_rocket.c`). It feeds the DPU's main operand from an **identity
conv** and sets the EW op to multiply (`EW_OP_TYPE=1`, `DPU_EW_CFG=0x108003C4`). The program
is the fp16 K-accum eltwise-add with one register field changed, `gen_matmul_fp16`'s `ew_mul`
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

> [!CAUTION]
> The flying generator `gen_ew_mul_fp16` programs the first two fields incorrectly, so its
> program never completes, or with `COMB_USE` fixed writes one lane in eight. No entry point
> calls it, and no environment variable enables it.

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
| `rocket_ffn_fp16_fused(fd, M, H, I, x, Wg, Wu, Wd, kind, out)` | **the gated FFN block, cube-resident**: the signature and block of `rocket_ffn_fp16`, with the `[M,I]` intermediates kept in cube layout on the NPU, so the host packs only `x` and reads back only `out`. fp16 only. The down matmul's K tiling is pinned to the projections' N tiling, and that grouping of the fp16 K-accumulation is the only numerical difference from `rocket_ffn_fp16`. `H` and `I` multiples of 32, `M` of 4. A projection output past one NPU batch (`nMt·nNt > 64`), or a tiling that cannot be matched, falls back to `rocket_ffn_fp16`. Gated at cos >= 0.999 against fp64 and >= 0.9995 against `rocket_ffn_fp16`, every output written (`tests/ffn_fused_rocket.c`) |
| `rocket_mlp_fp16_fused(fd, M, Din, Dff, Dout, x, W1, b1, W2, kind, out)` | **the non-gated encoder FFN, cube-resident** `out = act(x·W1ᵀ + b1)·W2ᵀ`. `b1` (or NULL) is added on the cube, the activation runs elementwise over the cube bytes, and fc2 reads the activated cube, so the host packs only `x` and reads back only `out`. fp16 only. `Din` and `Dff` multiples of 32, `Dout` of 16, `M` of 4. A fc1 output past one NPU batch, or a tiling that cannot be matched, falls back to a host-handoff FFN. It is `rocket_encoder_block_fp16`'s default FFN and is gated through that block (`tests/encoder_block_rocket.c`) |
| `rocket_softmax_fp16(fd, M, N, in, out)` | **row-wise softmax** over the last axis. Host row-max + subtract -> NPU `exp` (LUT) -> NPU row-sum (feature reduce) -> host `1/s` -> NPU per-row scale. The **row-max is on the host**, because matmul and reduce can only sum, and the only on-NPU max is the PPU max-pool, the resident-fusion path. Validated to T=1500 (Whisper seq), rows sum to 1±5e-4 (`tests/softmax_rocket.c`) |
| `rocket_logsoftmax_fp16(fd, M, N, in, out)` | **row-wise LogSoftmax** `x − logsumexp(x)` (the classification / NLL-loss head, LM log-probs). Shares softmax steps 1-3 (host row-max+subtract -> NPU `exp` -> NPU row-sum), then a per-row **subtract** instead of a divide: host `ls=log(s)` (exact, O(M), like softmax's `1/s`, and **not** the DPU LOG LUT, which is for large-tensor log) -> NPU per-row `ew_sub`. All-additive ⇒ better-conditioned than softmax (no tiny-prob blow-up). HW max_abs <= 0.031, `Σexp(out)=1` (`tests/softmax_rocket.c`) |
| `rocket_cross_entropy_fp16(fd, M, N, logits, target, loss)` | **stable per-row cross-entropy** `CE[m]=logsumexp(logits[m]) − logits[m][target[m]]` = `−logsoftmax[target]` (softmax-classifier / LM NLL loss). The on-NPU logsumexp reduction (host row-max -> NPU `exp` -> NPU fp32 row-sum -> host `log(s)`) + a **host gather** of the target logit (there is **no HW gather**, so M scalar lookups, like the host `1/s`). Never materializes softmax (no divide). CE >= 0. **fp32-grade** (max_abs <= 1.5e-4, since the loss never round-trips through fp16 output storage) (`tests/cross_entropy_rocket.c`) |
| `rocket_mha_self_fp16(fd, T, d, n_head, x, Wq,bq, Wk,bk, Wv,bv, Wo,bo, out)` | **multi-head self-attention** (the pure attention sublayer): QKV proj + per-head `scale·(q·kᵀ)` + softmax + `P·v` + out-proj, **all matmuls + every softmax on the NPU**. Weights row-major `[out,in]`. Biases optional (Whisper: bq,bv,bo, no bk). The **key count is padded to %32 and the pad score columns masked to −30000 before softmax** (the matmul rejects unaligned N/K, and Whisper T=1500 is unaligned). cos = 1.000000 at Whisper-base d=512/8-head incl. T%16≠0 (`tests/mha_rocket.c`) |
| `rocket_flash_attn_fp16(fd, n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads, scale, softcap, Q, K, V, mask, out)` | **masked grouped-query (flash) attention**, the decoder / LLM-prefill attention sublayer (the op an llama.cpp `FLASH_ATTN_EXT` lowers to). Already-projected Q/K/V + an **additive mask the caller supplies** (causal + sliding-window, which the entry does not synthesize): per head `scale·(q·kᵀ)` -> optional `softcap·tanh` -> `+mask` -> softmax -> `P·v`. **Both matmuls run on the NPU**, and the scale, soft-cap, mask and softmax run as one fp32 host pass per row by default (`ROCKET_ATTN_HOST_SOFTMAX=0` runs it on the NPU. The variable reads the same on the encoder MHA path, where the NPU is the default and `=1` selects the host). Handles **GQA/MQA** (`n_head>n_kv_heads`, kv head `h/(n_head/n_kv_heads)`). Head-major dense layouts make each head a contiguous matmul operand. `n_tokens` padded to %4, `n_kv` to %32 (pad keys scored −∞). cos = 1.000000 vs an fp64 oracle at Gemma-4-12B shapes (head_dim 256, 16 q-heads, 8 kv-heads local / 1 global, sliding window, soft-cap) (`tests/flash_attn_rocket.c`). Caution: the scores are fp16 and **saturate**. A score `scale·q·k` past 65504 is written as ±65504, not inf, so the keys past it tie and the softmax returns finite, wrong values [HW sweep 2026-10-09, RK1, ~816 MHz]. Every path checks the raw scores in its host mask pass and returns `ROCKET_E_RANGE` when a real key reaches the limit, and `out` is then not valid. A caller holding the operands at higher precision can recompute. `rocket_flash_attn_fp16_mt(…, nthreads)` fans the heads across worker fds (the cores run head ranges in parallel). `rocket_fa_ctx` + `rocket_flash_attn_fp16_ctx` is the persistent form for repeated calls (worker fds held open + per-worker score scratch kept resident), all numerically identical. `ROCKET_FA_CHAIN` (on by default) batches each worker's per-head QK matmuls into one NPU job and its AV matmuls into a second (the mask + softmax sit between, preserving the host-softmax default): one submit + fence per head range instead of per head, through a per-worker resident batched-matmul context (`rocket_mm_batch`, BOs + score scratch held + prezeroed once). Chaining cuts the small-GEMM fixed dispatch overhead: it nearly **doubles** the FA-op throughput vs a per-call batch (~2.0-3.8x the per-head path: 186->50 ms @512, 363->183 ms @2K head-range time), moving the FA-NPU-vs-CPU prefill crossover from ~6K to ~2K. The chaining gain persists at depth: with the head group bounded by `ROCKET_FA_CHAIN_ELEMS` (default 32M score elems = a worker's ~3-head range batched up to ~20K context, ~150-200 MB/worker scratch), collapsing the per-head submits gives **1.10x@4K / 1.47x@8K / 1.32x@16K / 1.16x@32K** at a 512-token ubatch (`fabench`, T=512, [HW sweep]), and +3% end-to-end pp8192 (Qwen3.5-0.8B-F16), with no short-context change (already batched <=2K). At a 2048-token ubatch each head's score alone exceeds the budget and the gain shrinks to ~1.05x. Raise `ROCKET_FA_CHAIN_ELEMS` to chain further, at a higher scratch cost. Numerically identical (cos = 1.000000). `ROCKET_FA_CHAIN=0` forces the per-head path. **`ROCKET_FA_ROWMAJOR`** (on by default) hands the scores and P between the chained QK and AV jobs as rows in a resident BO, through the row-major matmul form, so the scores are not de-tiled out of the NPU's cube layout and P is not packed back into it. Plain rows serve to a 4 KiB row (`ROCKET_FA_RM_MAX_ROW`), with P rewritten in place. Past that, QK writes one block of rows per output tile, because a wider row-major write costs the device more. The call is 0.64-0.69x the cube handoff's at a 1370-token ViT layer and 0.73-0.82x at a 512-query prefill shape from 2048 to 32768 keys, with the output bit-identical [HW sweep 2026-10-11, RK1, ~816 MHz]. The one exception is a head that runs alone in its chained group: it keeps fp32 host K-accumulation for AV, where the cube handoff's one-shot matmul accumulates in fp16 on the device (a few fp16 ulps). **`ROCKET_FA_TILE_KV`** (default off) is the opt-in online/tiled long-context variant: walk the key axis in `ROCKET_FA_TILE_KV`-wide tiles that keep the FlashAttention-2 running softmax (fp32 max/denom/output), so the working score tile is `[Tp,tile]` not the full `[Tp,n_kv]` (32 MB/head at 32K). Bit-faithful (fp32 accumulation, cos = 1.000000) but **slower** on this dispatch-bound NPU, with more and smaller per-tile submits, converging to the materialized path from below (0.58x@2K-tile -> 0.93x@16K-tile at n_kv 32K). It bounds the FA scratch at extreme context and does not add speed. It engages above `ROCKET_FA_TILE_MIN_KV` (default 8192) |
| `rocket_encoder_block_fp16(fd, T, d, n_head, d_ff, x, ln1…, Wq…Wo…, ln2…, Wf1…Wf2…, eps, out)` | **one full Whisper/transformer encoder block** (pre-norm): `x += MHA(LN1(x)); x += MLP(LN2(x))`. On the NPU: both LayerNorms, all attention matmuls + softmax, both residual adds, the two FFN projection matmuls, and the Φ-gate DPU LUT of the FFN's 2-pass GELU `x·Φ(x)`. By default the FFN runs through `rocket_mlp_fp16_fused`, which adds `bf1` on the NPU and multiplies `x·Φ(x)` on the host unless `ROCKET_ACT_NPU_MUL=1`. `ROCKET_ENCODER_NOFUSE` (any value) runs separate calls instead, with `bf1` on the host and the multiply on the DPU EW unit. cos = 1.000000 vs an fp64 block oracle (`tests/encoder_block_rocket.c`) |
| `rocket_siglip_encode(fd, m, pixels, out, hidden)` / `rocket_siglip_encode_ctx(c, …)` (`rocket_siglip.h`) | **the SigLIP-B/16 vision encoder end-to-end** (SmolVLM-256M front-end): patch-embed (im2col -> matmul) + position add + 12x `rocket_encoder_block_fp16` `(L=1024, d=768, 12 heads, d_ff=3072)` + post-LayerNorm. Weights mmap'd from a flat fp16 blob (`tools/siglip_extract.py`). Two paths: the simple per-call form and a **resident** form (`_ctx`: static GEMMs prepacked once + multicore). The resident path runs the 12 attention heads **across the worker fds** (`rocket_flash_attn_fp16_ctx`, unmasked MHA, since heads are independent, so one drm scheduling entity per fd dispatches head ranges across the NPU cores, with head-chaining and resident scratch), **1.44-1.51x warm** against a single-stream per-head loop (`ROCKET_SIGLIP_FA=0` selects the loop). The host GELU uses a **bit-exact fp16->fp16 LUT** (all 65536 fp16 outputs fit one 128 KB table, identical to the scalar `tanhf`), a further **1.22x** (`ROCKET_SIGLIP_GELU_SCALAR=1` selects the scalar form), for **1.78x warm combined**, cosine unchanged. The context allocates the per-image host scratch (~17 MiB at Depth Anything's ViT-S), so an encode does no heap traffic, and one context serves one caller at a time. The bias adds and LayerNorm are vectorized, LayerNorm in float with per-lane row sums: Depth Anything v2 Small through ort-rocket runs in 0.90-0.95x the wall-clock time of the unvectorized bias adds and LayerNorm [HW sweep 2026-10-09, RK1, `ondemand`]. `ROCKET_SIGLIP_PROF=1` prints the phases, the bias adds' own bucket and the call's minor faults. **Per-layer cosine 0.999998 vs an fp32 reference** (`tests/siglip_rocket.c`) |
| `rocket_modernbert_ctx_create(m, nthreads)` / `rocket_modernbert_encode(c, B, T, len, emb, post_bias, out, hidden)` (`rocket_modernbert.h`) | **a ModernBERT text encoder end-to-end**, plus an optional stack of torch pre-norm layers after it (a decision or classification head). RoPE at a per-layer theta, global and sliding (`\|i-j\| <= window`) layers, GeGLU, bias-less LayerNorms. B right-padded sequences run each at its own length: every projection runs once over their live rows stacked into one M, attention per sequence, so padding costs nothing. The residual stream stays fp32 on the host, because ModernBERT-large's grows past 29000. Each FFN down-projection's input is scaled by a power of two into fp16's range and back. Sliding layers past `2T + 2 window` tokens run **banded** (`ROCKET_MB_BAND_TILE`). `nthreads` 0 is a host mode for off-device checks. Worst layer cosine 0.9999936 against ONNX Runtime's fp32 CPU provider on the Laya decision model, post-stack output 0.99999995 (`tests/modernbert_rocket.c`, artifacts from `tools/modernbert_extract.py`) |

Standalone, these ops are submit-bound, and the host is faster for an isolated norm. Their
value is in composition: keeping the activation cube-resident between two NPU matmuls skips
the de-tile->host->re-pack round-trip.

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

**int16's int32 output saturates.** `gen_matmul_int16` writes the whole int32 output in one
task. It is bit-exact against the int64 dot product clamped to int32 (`int16_native_probe`,
`matmul_int16_rocket`). It is therefore exact only while the operand ranges keep each sum
inside int32. No tiled entry exposes it.

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
`pool_rk3576_bracket_gate`) still run and must pass. A skip is visible in the summary, so a
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
| `rowmajor_tile_probe` | the row-major form at the shapes a tiled path issues: a K slice of a wider A, an N tile into a wider C, both, int8 -> int32, fp16 out, and a two-task fp16 K-accumulation with a cube partial and a row-major final write, every element of a sentinel-filled output scored, cube-program controls. Then submit-to-fence timings at the library's tile shapes split into the read, the write and the row pitch (`--pitch`, `--pitch-asym`, `--pitch2`), with the wait on a small fence BO so the output BO's sync is not timed. A probe, not a correctness test |
| `rowmajor_kacc_chain_probe` | the row-major matmul form inside one chained fp16 K-accumulation kick with `DATA_REUSE`, in the library's own order: the cube program, own-width row-major tiles, and tiles written straight into C, each with reuse on and off, every element scored, then the whole kick's submit-to-fence over rotated reps. A probe, not a correctness test |
| `fa_replay_probe` | replays one flash-attention op from the tiles `ROCKET_FA_DUMP_OP` wrote, through `rocket_flash_attn_fp16_ctx`, many times in one context and in fresh ones, at a chosen worker count, and counts how many trials equal each of two dumped outputs and how many distinct outputs appear. With `FA_REPLAY_MARKER` naming a tracefs `trace_marker` it brackets each trial for `tools/fa_trace_join.py`, which joins the `gpu_scheduler` job events (run under `tools/fa_core_trace.sh`) to say which core ran each worker's QK and AV jobs. A probe, not a correctness test |
| `fa_core_matmul_probe` | one head's real AV operands (written by `tests/fa_av_dump_wrap.c`, a `-Wl,--wrap=rocket_mm_batch_run` link of the replay) from several worker fds at once, the handler's passes and pass 0 through `rocket_matmul_fp16_f32out`, each call compared with the first of its kind and with the exact sum. The core that ran a call comes from the trace. A probe, not a correctness test |
| `npu_core_diff_probe` | `run` writes fp16 -> fp32 tiles at 512×384×256 for three synthetic A distributions, an `int8` control through `rocket_matmul_int8`, and `perm` K permutations of real AV operands with `NPU_CORE_DIFF_OPS` (`NPU_CORE_DIFF_PERM` restricts them to inside or between aligned K blocks, `NPU_CORE_DIFF_XOR` remaps each 32-wide block's lanes), to files. `compare` diffs two runs element by element and says which side is the correctly rounded one. One core per process, chosen by `RKNPU_CORE_MASK` on the vendor provider. A probe, not a correctness test |
| `ppu_sub4_chain_probe` | two chained PPU pooling passes with the kernel order forced so the intermediate is 2x2 or 3x3 (MAX and AVG, C 64 and 130, a 4x4 control): the chained second pass scored per channel against the same pass over a CPU-scattered copy of the device-written intermediate and against CPU models. `--no-ppu-done` adds a reproduction case whose jobs retire by the watchdog. A probe, not a correctness test |
| `dw_int8_cost_probe` | where the int8 depthwise entry's per-call time goes at MobileDet's largest depthwise shapes: the whole warm call, its host scatter and gather as written, and a blocked form of each checked byte for byte against them. A probe: it asserts only that the blocked forms agree |
| `conv_pack_cost_probe` | the fp16 conv's and the int8 direct conv's per-element host pack against the whole warm call, at EfficientDet-Lite0's depthwise shapes and MobileDet's direct shapes, with a blocked form of each pack checked byte for byte. A probe: it asserts only that the blocked forms agree |
| `cpend_wzp_probe` | `DPU_BS_OW_OP`, the TRM's CPEND operand, as an asymmetric weight zero point: the int8-out depthwise program at five weight zero points and three shapes with the operand swept (`-zw`, 0, `+zw`, the low byte), each case scored against TFLite's accumulator plus the OUT_CVT model and against `bias' + sum xp*w + v*sum xp` for four readings of the operand. The direct int32-raw program with CPEND patched live, scored in int32 and for placement anywhere in the BO. Two shapes with a partial last 64-channel group, which the program leaves unwritten. `layer` mode runs one real depthwise layer for `tflite-rocket/tools/requant_chain.py`. A probe, not a correctness test |
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
| `pool_rk3576_bracket_gate` | host-only, no NPU: the RK3576 pooling entry against a host-memory provider defined in the test, which replaces the whole submit seam. BOs are held `PROT_NONE` outside their bracket, so a CPU access outside `PREP_BO`/`FINI_BO` faults. A double PREP, an unmatched FINI, a submit while the CPU holds a BO and a bracket left open are counted. Seven paths (clean, the divisor-lag check, a retired job, a lagged output, an unwritten one, a failed submit, a failed wait) with `ROCKET_RK3576_I32_SENTINEL` on and off, each held to an exact PREP/FINI count on the output and the result scored against `rocket_pool_ref_int8_rk3576()` |
| `bytes_moved_rocket` | analytical DRAM-traffic model (pure, no NPU): per-phase bytes from shape+tiling+dtype+reuse, planner `njobs` cross-check, int8 readback lower-bound demo |
| `conv2d_fp16_rocket` | general fp16 `CONV_2D` + native depthwise, bit-exact vs an NHWC oracle (direct, OC%16 pad, DW G=32, tiling) |
| `pool_fp16_rocket` | on-NPU MaxPool / AveragePool (PPU): cube self-check + HW oracle (max bit-exact, avg <= fp16-recip tol) |
| `pool_int8_rocket` | on-NPU int8 / uint8 MaxPool / AveragePool via the fp16 PPU path: int8 & uint8 MAX bit-exact, int8 AVG ±1 ULP vs an integer golden |
| `pool_int8_native_probe` | the PPU's native int8 pooling precision, as a map over the precision pair: `PPU_DATA_FORMAT[2:0]`=0 with `PPU_RDMA_DATA_FORMAT[1:0]`=1 over a C2=16 byte cube pools MAX, MIN and AVG bit-exactly, given an integer Q16 `0x10000/k` reciprocal and the integer pad fill. A probe, not a correctness test. The library's pooling entries use the fp16 path |
| `conv_transpose_resident` | the resident ConvTranspose against the scatter reference: bit-exact on integer fills (the host asserts every matmul partial prefix and output below 2048) at the old test's 18 direct shapes, a cropping pad, a per-axis dilation and stride and pix2pix's and SAM's ten layers, both worker splits at four of them. `\|err\| <= 2^-9` of each element's magnitude sum on real fills with a bias. A depthwise refusal, a handle that keeps its packed weight after the caller's buffer changes, and one that refuses another context. Registered twice, the second under `ROCKET_KACC=0` |
| `ct_model_bench` | per-call time of the old entry and the resident entry at pix2pix's and SAM's ten layers, warm medians, the order rotated by an argument for interleaved passes. A benchmark, not a correctness test |
| `ct_route_probe` | the candidate ConvTranspose paths, timed: col2im and sub-pixel as resident matmuls, sub-pixel as one forward conv, host phases split. A probe, not a correctness test |
| `deconv_extent_probe` | the CNA deconvolution mode with the output extent programmed rather than derived, through `rocket_conv2d_fp16_job_extent()`: one task per case, scored over the whole output against `rocket_conv_transpose2d_ref_fp16`. Bit-exact on the RK3588 up to 64×64 -> 128×128. A probe, not a correctness test. `rocket_conv_transpose2d_fp16` runs the mode on its direct path (`rocket_conv_transpose2d_route()`) |
| `requant_round_probe` | the OUT_CVT's tie rule at `OUT_CVT_SHIFT` bit 30 = 0 and 1 (`ROCKET_OUT_CVT_ROUND`), over ties it builds on purpose, since no round scale reaches one. It asserts the recorded rule: half to even with the bit clear and half away from zero with it set, each the one survivor of five candidate rules, on both SoCs (the RK3576 through `rocket_matmul_int8_rk3576`, the RK3588 through the int8 matmul's integer convert). A `ctest` and a `gates-rk3576.sh` row |
| `reduce_mean_rocket` | on-NPU spatial reductions over [H,W]: GlobalAvgPool / Mean (HW vs fp64, tolerance) **and** GlobalMax/MinPool / ReduceMax/Min (**bit-exact**, idempotent), plus a factor and schedule/cube self-check (single + multi-pass, square + equal-count rect, host fallback for non-16-smooth / unequal-count) |
| `cumsum_rocket` | on-NPU **cumsum / prefix sum** (triangular ones-matmul) vs an fp64 prefix-sum oracle, over all four variants (incl/excl x forward/reverse), M-tile boundary, N%32≠0, T=1500. **Bit-exact** (max_abs 0). Independent O(N²) fp64 recompute self-check |
| `conv2d_int8_rocket` | native int8 `CONV_2D`: cube self-check + single-job HW + the **tiled-runtime direct case** (big/wide/IC<32/OC-pad) bit-exact vs an int64 oracle |
| `conv1d_rocket` | **conv1d** front-end (Whisper conv1/conv2, KW=3, stride 1/2) lowered onto a width-1 conv2d with time on the height axis, bit-exact / fp16-tolerance vs the conv oracle (incl. IC=80/512) |
| `conv_width_gate` | int8 and fp16 convs through the public entries whose tiles would pass the CNA's 11-bit width or 10-bit grains fields (2300 and 2500 columns at one or two rows, 3000 rows at one column), beside controls under the limit. Every element scored, a refusal counts as a failure |
| `rk3576_conv_width_probe` | the RK3576 question: the int8 direct, depthwise and packed-image conv entries, the fp16 direct and packed-image ones and the int8 max pool, driven with extents on ladders just under and past 11, 12 and 13 bits (width to 12288, one-task height to 6144, depthwise and pool channels to 8224), 4-row planes 2048-8192 wide whose tasks read the line stride past 14 bits, kernels of 32 and 33, and strides past the 3-bit field. Every element scored, reporting rc, wrong, never-written and the column, row and channel a wrap starts at. `plan` prints each case's row plan and its first task's geometry words without submitting. A probe, not a correctness test |
| `rk3576_retire_probe` | the RK3576 write guard's completion half. `forced` sets the backstop to 1 us, so every job reads as retired, and asserts that each of eight small calls (the direct, depthwise, packed-image and fp16 convolutions, the int8, int32 and fp16 matmuls, the max pool) refuses with the retirement counters moved. `real` asserts the same calls return 0 with none counted, then runs the int8 matmul at M 512 and a chosen K and N (default 2240 and 1536, where every job retires at the backstop) and asserts the retirements are counted as the kernel logged them and that the call refuses or is exact, never rc 0 over a wrong output. The net test covers the chained kick, run under `ROCKET_RK3576_BACKSTOP_US=1`. A probe, not a correctness test |
| `exp_lut_rocket` | DPU-LUT **EXP** (shifted single-table, the q>=1 clamp) + a **softmax-sum** end-to-end check (row-max subtracted) vs `exp` |
| `softmax_rocket` | on-NPU **row-wise softmax and LogSoftmax** vs an fp64 oracle (M-tile boundary, Whisper T=1500, wide-spread tail). Softmax rows sum to 1, logsoftmax `Σexp(out)=1` |
| `cross_entropy_rocket` | stable per-row **cross-entropy** (logsumexp reduce + host gather) vs an fp64 CE oracle (M-tile boundary, T=1500 vocab, random + argmax targets, wide spread). **fp32-grade** (max_abs <= 1.5e-4), CE >= 0. Self-check `CE == −logsoftmax[target]` (independent path) |
| `layernorm_rocket` | on-NPU **LayerNorm** (stacked-row reduce + affine fold) vs fp64 (M-tile boundary, no-beta, H%32≠0, the overflow prescale) |
| `norm_vision_rocket` | on-NPU **BatchNorm / GroupNorm / InstanceNorm / L2-Normalize** vs fp64 (group counts incl. G=1 / G=C, C%32≠0, P=1, row-tile boundary, the overflow prescale) |
| `mha_rocket` | on-NPU **multi-head self-attention** vs an fp64 attention oracle (cosine sim, Whisper-base d=512/8-head, T%16≠0 key-pad path) |
| `flash_attn_rocket` | masked **grouped-query (flash) attention** vs an fp64 oracle (cosine sim, Gemma-4-12B head_dim 256 / 16 q-heads / 8-kv GQA + 1-kv MQA, sliding window, soft-cap, n_kv>T, unaligned T/n_kv). The **chained long-context** path (`ROCKET_FA_CHAIN_ELEMS` high -> a worker's whole head range in one QK+AV job, to 16K). The **online/tiled** path (`ROCKET_FA_TILE_KV` -> per-row masked-tile skip, short last tile, 16-tile 8K, grow/reuse ctx). `bench` mode A/Bs materialized vs tiled `_ctx` wall-time |
| `encoder_block_rocket` | one **full Whisper encoder block** (LN+MHA+residual+LN+MLP) vs an fp64 block oracle (cosine sim, Whisper-base, T%16≠0) |
| `siglip_rocket` | the **full SigLIP-B/16 vision encoder** vs an fp32 reference (per-layer cosine, mean 0.999998, plus a resident-path bench, and the reference encoded again after another image on the same context, which must match the first encode). Skips without the weight blob + oracle artifacts on disk |
| `modernbert_rocket` | the **resident ModernBERT encoder and its post stack** vs ONNX Runtime's fp32 taps (per-layer cosine over the live rows, a batch of 3 and single long sequences, and `ROCKET_MB_BENCH=N` adds a warm median). `--host` runs the host mode. Skips without `-DROCKETNPU_MODERNBERT_ARTIFACTS` |
| `replay_dw_mesa` | int8 DW int8-out **regcmd** replayed on Mesa/Teflon's captured BOs, bit-exact vs `mesa-output`: the library's register program is Mesa's, pad word aside. Not a check of the function: the capture ran an int8 model through Mesa's uint8 driver, on a constant input |
| `conv_dw_int8_runtime` | int8 DW int8-out **runtime** vs TFLite's reference kernels on the capture's model (`tests/dw_litert_ref.py` writes the fixture, within one at a rounding boundary, at most 2% of outputs), and vs an exact host model of the requant, bit-exact at eight shapes with symmetric weights and seven with an asymmetric weight zero point (-128 to 127, a partial 64-channel group, pad 0 on an odd plane at stride 2, and a 768-channel 20×20 layer). The planner refuses along each axis it checks, and a weight zero point outside int8 refuses |
| `conv_dw_int8_perc_runtime` | the **per-channel** int8 depthwise entry at six shapes (one job, 1152 deep with a 3e4 scale spread, all-zero and near-dead channels, 480 at 20x20, banded 160x160, a partial group, three jobs): every element equal to a host model of the BS-multiplier and OUT_CVT arithmetic built from the entry's own plan, and within one of TFLite's per-channel arithmetic. Every output zero point sits inside the range, since at -128 a wrong negative-product shift passes |
| `conv2d_int8_q_rocket` | the int8-out direct conv entry at 13 shapes through the fd, ctx, 3-fd pool and fd -1 entries, every element against a host model of TFLite's accumulator and the OUT_CVT. Five are 1×1s from 160×160 to 20×20 at a frontend's pointwise shapes. A `ROCKET_CONV_BATCH=1` twin |
| `conv2d_int8_q_perc_rocket` | the **per-channel** int8-out direct conv at 8 shapes (three scale-class jobs at 80×80, OC 672 and 1152, a 160×160 1×1, a padded IC under a CNA pad, the stem at stride 2, 5×5 stride 2, all-zero filters) through the fd, ctx, 3-fd pool and fd -1 entries: every element equal to the test's own accumulator taken through the split requant at the entry's plan, and within one of TFLite's per-channel arithmetic. A `ROCKET_CONV_BATCH=1` twin |
| `conv_i8out_probe` | a probe of Mesa's direct int8-out program on the RK3588: its geometry, CPEND on it, the per-channel BS multiplier and `B` on it. `witness` adds the cases that time out (a partial 32-kernel group, the depthwise writer geometry) |
| `dw_perc_probe` | a probe of the BS stage's per-channel multiplier on the raw int8-out depthwise program: the coefficient layout, the order of add and multiply, the held-wide product, the tie rule, the sign-split shift fields, `B` as a per-channel CPEND, and the BN stage's multiplier (which hung in the one configuration tried) |
| `dump_regcmd` | diff the generated regcmd on a host machine (no HW) |
| `replay_dump` | replay a dumped in-context failing matmul |
| `matmul_fp16_rocket` | standalone fp16xfp16->fp16/fp32 single-task smoke test (the `gen_matmul_fp16` path, no tiling) |
| `dump_dw_regcmd` | host-only: emit a depthwise-conv regcmd as a u64 stream for the Mesa decoder (no HW) |
| `membench` | DRAM bandwidth + NPU readback de-tile microbench (no deps) |

The `accum` classifiers and `dtype_perf` characterize K-accum feasibility and the
dtype-independent throughput limit.

## Runtime environment variables

The library reads a few `ROCKET_*` environment variables, and the `ggml-rocket` backend
exposes more. `sudo` strips the environment, so use `sudo -E`.

| Variable | Default | What it does |
|---|---|---|
| `ROCKET_DEV` | `/dev/accel/accel0` | The device node `rocket_open()` opens, for a box with more than one NPU or a test rig |
| `ROCKET_RK3576_ALLOW_MULTI_FD` | off | On an RK3576 with more than one core bound, `rocket_open()` refuses a second fd with `-EBUSY`, because two jobs in flight there compute wrong answers. `=1` lifts the refusal for a caller that serializes its own submits |
| `ROCKET_KACC` | on | fp16 NPU K-accum. `=0`, or `ROCKET_NO_KACC`, opts out to the byte-exact host fp32-accum path |
| `ROCKET_REUSE` | auto under KACC, else 0 | CBUF operand reuse: 0 off, 1 weight, 2 data. Auto picks the longer reuse run per shape, data on a tie |
| `ROCKET_MM_MT/NT/KT` | planned | Tile overrides. Any one of them selects the whole-tile plan |
| `ROCKET_MM_STREAM` | on | The fp16 planner streams the weights through the CBUF, with far fewer tasks. `=0` selects the whole-tile plan. See below |
| `ROCKET_MM_STREAM_INT8` | off | `=1` makes the int8 planner stream its weights too, under `ROCKET_MM_STREAM`. See below |
| `ROCKET_MM_STREAM_KT` | | Diagnostic. Restricts a streamed plan to this Kt, so each candidate can be timed. A Kt that leaves no M tile falls back to the planner's choice |
| `ROCKET_MM_ASYM` | on | Asymmetric Mt>Nt tiling of the whole-tile plan. See below |
| `ROCKET_N_THREADS` | | Read by `ggml-rocket` and some tests, not by the library: the worker count they pass to a context |
| `ROCKET_PIN_AT_CREATE` | on | Each per-call worker is created on the big core it pins to. `=0` creates it with its creator's CPU mask, so a creator pinned to one core starts every worker on that core |
| `ROCKET_I8_PACK_FANOUT` | on | `rocket_i8_weights_pack` and `_pack_gw` pack each worker's column slice into its BO on its own thread, for a weight of 1 MiB or more. `=0` packs the slices one after another on the calling thread |
| `ROCKET_WAIT_MS` | | Fence deadline |
| `ROCKET_SLOW_WAIT_MS` | per SoC: 450 RK3588, 110 RK3576 | The slow-wait mark. A fenced wait that reaches it logs a warning naming the BO and counts in `rocket_fence_wait_slow_count()`, because the `rocket` driver retires a job that runs past its watchdog (500 ms on the RK3588, a 125 ms backstop on the RK3576) and signals its fence as if it had completed |
| `ROCKET_RK3576_BACKSTOP_US` | the profile's, 125000 | What the RK3576 write guards score a job's submit-to-fence time against. A job at or past it was retired and is redone. A probe sets it to 1 to make every job count as retired |
| `ROCKET_RK3576_MM_NARROW` | on | The RK3576 int8 matmul re-plans a tile whose job retires with its output partly written at half its width, down to 32 channels, and keeps that width per K for the process. With the re-plan, a K where a job stalls at a fixed output channel still produces a result. `=0` makes those K refuse |
| `ROCKET_RK3576_MM_PC_SHIFT` | on | The RK3576 per-column int8 matmul (`rocket_matmul_int8_rk3576_perc` and its `_sa` and `_wbo` forms) programs each tile's gain in the DPU shift word, so its per-column multiplier spans the int16 field at any K. `=0` restores the shift-0 ramp, whose multiplier the int32 product caps in the low hundreds at a GEMM's depth. Read per call |
| `ROCKET_MM_PROFILE` | off | Time breakdown by phase (buckets) |
| `ROCKET_CONV_PROFILE` | off | One line at exit per RK3588 conv program (phases summed over its jobs), and for the resident ConvTranspose its GEMM and scatter-add time |
| `ROCKET_CT_SPLIT` | per shape | `m` or `n` forces how the resident ConvTranspose's workers split its GEMM, at pack time |
| `ROCKET_FA_PROFILE` | off | Splits one flash-attention head range into gather / QK / mask / softmax / AV / scatter and prints two lines at exit: the bucket sums, then the largest single range's own buckets. The sums are worker-thread intervals added over concurrent workers, so they are not a share of anything. The second line is the one to divide, because its terms were measured on one thread over one interval and the dispatch thread waited for that range. The usable quantity is a bucket's share of `max-range` applied to the caller's measured wall-clock time |
| `ROCKET_FA_CHAIN` | on | Batches a flash-attention worker's per-head QK/AV submits through a resident batched-matmul context. `=0` forces the per-head path |
| `ROCKET_FA_CHAIN_ELEMS` | 32M score elems | Bounds the chained head group, so a worker's head range stays batched up to ~20K context. Raise it to chain at deeper context, for more scratch |
| `ROCKET_FA_TILE_KV` | off | Opt-in online and tiled long-context flash attention. It bounds memory use and is slower than the materialized path on this dispatch-bound NPU |
| `ROCKET_FA_TILE_MIN_KV` | 8192 | Where `ROCKET_FA_TILE_KV` engages |
| `ROCKET_FA_FUSED_SOFTMAX` | on | The flash attention's host scale, soft-cap, mask and softmax as one fp32 pass per row with a vector exp and one fp16 rounding. `=0` restores the separate mask pass and softmax |
| `ROCKET_FA_ROWMAJOR` | on | On the persistent `rocket_fa_ctx` path, QK writes the scores and AV reads P as rows in a resident BO, so neither is de-tiled out of nor packed into the NPU's cube layout. `=0` keeps the cube handoff |
| `ROCKET_FA_RM_MAX_ROW` | 4096 bytes | The widest row of scores `ROCKET_FA_ROWMAJOR` writes as plain rows, with P rewritten in place. Past it QK writes one block of rows per output tile and P goes to a second BO |
| `ROCKET_MB_BAND_TILE` | 128 | `rocket_modernbert`'s banded sliding attention query tile. `0` computes every sliding layer dense |
| `ROCKET_MB_ATTN` | npu | `host` runs `rocket_modernbert`'s attention in fp32 on the host. Against the 2026-09-27 library it tied the NPU at 58 tokens and lost at 639 (RK1, ~816 MHz) |
| `ROCKET_MB_PROF` | off | `rocket_modernbert` prints each encode's phase split (LayerNorm, GEMM, attention, activation, down-projection) |
| `ROCKET_ENCODER_NOFUSE` | unset | Any value runs `rocket_encoder_block_fp16`'s FFN as separate calls instead of `rocket_mlp_fp16_fused`: `bf1` added on the host, the GELU multiply on the DPU EW unit |

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
correctness test's strict gcap=1 path. The common Gemma FFN-down (nKt=40) falls back, so the
end-to-end LLM gain is ~nil. The variable is a narrow option rather than a default.
`tests/matmul_kacc_chain_rocket.c` tests it, and `matmul_kacc_chain_bench` A/Bs it.

`ROCKET_MM_ASYM` (**default on**, `=0` opts out) is a tiling option rather than a submit
one. The planner caps Mt and Nt at 256 and maximizes Kt, which picks a symmetric Mt=Nt=256
tile (Kt=384). For a shape that tiles both N and K, halving Nt to 128 frees CBUF so Kt grows
to 512, with Mt staying at 256.

The asymmetric Mt>Nt tile runs the NPU datapath measurably faster. It is **+6-9% warm** on
the square and large-K prefill matmuls (1024²/2048²×4096, Gemma FFN-down), and about a wash on
FFN-up. End-to-end warm pp2048 through llama.cpp:

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
own, at 512×3840×4096 on random fp16 data, against an fp32 reference. Each must reach cosine
similarity 0.9995 and keep every element within an fp16 bound that grows with nKt. Neither
test compares the two settings.

`ROCKET_MM_STREAM` (**default on**, `=0` opts out) selects the fp16 planner's streamed plan.
The whole-tile plan keeps each task's input and weight tiles in the CBUF together, which caps
Nt at 256. The streamed plan keeps the input tile resident and streams the weights through the
CBUF's weight banks. One task then covers up to 4096 output columns. `512×3840×4096` takes 12
tasks instead of 512. Any of `ROCKET_MM_MT`, `ROCKET_MM_NT` or `ROCKET_MM_KT` also selects the
whole-tile plan, and `ROCKET_MM_ASYM` applies only to the whole-tile plan.

The streamed plan costs one 256-row slice of M, so every M of 256 or more plans the same Kt and
Nt. A resident weight packed at one row count therefore serves every prefill row count.

The plan's cost model charges each task the larger of its weight stream and its MACs. It charges
each K pass its partial output, eight times over when the host sums the passes (int8, or fp16
with `ROCKET_KACC=0`). It was fitted to a sweep of every streamed Kt at twelve shapes. Against
a byte-count model at 44 shapes it is 1.06x faster in fp16 and 1.23x in int8 (geo-mean). It is
slower at one shape, fp16 3584×1024, at 0.87x [HW sweep 2026-10-21].

The int8 planner streams under `ROCKET_MM_STREAM_INT8=1`, and the
int4 and bf16 planners keep the whole-tile plan.

At M 512 on five workers, the resident entries' streamed plan is 1.10-1.51x faster than the
whole-tile plan at LLM projection shapes in fp16. In int8 it is 0.99-2.43x. The int8 gain
is largest where the streamed plan removes K passes. The host sums every pass's int32
partial. int8 streaming stays opt-in until the routed 1x1 convolutions of `tflite-rocket`'s
detectors are measured, since they reach the same resident int8 entry.

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

`gen_matmul_fp16` and `gen_matmul_int8` refuse a streamed task whose weight banks number fewer
than `npu_stream_weight_banks()` returns, with -3. `gen_matmul_int8` also refuses one at exactly
that count. Such a task can fail to complete, and the `rocket` driver's watchdog retires it with
its fence signalled and its output unwritten. Both also refuse `weight_reuse` on a streamed
task.

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

The threshold is read once from the environment on first use. `ROCKET_LOG_LEVEL` takes
`error`, `warn`, `info` or `debug`, or `0`..`3`, and `ROCKET_DEBUG` raises it to at least
`debug`. Installing a callback is safe while other threads log: the callback and its
`user_data` are published together. A message already in flight can still reach a replaced
pair, so the host must keep it valid.

`ROCKET_LOG_STDERR` (set to any non-`0` value) additionally tees every emitted line to
`stderr`, but only when a host callback is installed, so the default sink never
double-prints. It serves a host that silences its own logger. `llama-bench`, for one,
installs a no-op `ggml` callback unless `-v` is passed. A host that adopts the channel
forwards it into `ggml`. Without `ROCKET_LOG_STDERR`, such a callback then swallows every
rocket diagnostic, including one-shot decisions like the resident-weight budget that change
the benchmarked number.
