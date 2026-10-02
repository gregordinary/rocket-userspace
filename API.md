# librocketnpu: API reference

The complete function reference, test catalog, and runtime-knob list for `librocketnpu`. The
[README](README.md) is the guide, and this is the reference. Convention throughout:
`C[M,N] = A[M,K] · B[N,K]ᵀ`, row-major.

## Library layout

```
include/                       public API
  rocket_npu.h                 device shim: open / bo_alloc / bo_prep(fence) / submit
  rocket_hw_profile.h          the chip machine-parameter profile (CBUF / tile geometry / dtype mask) + capability query
  rocket_matmul.h              the matmul API (fp16 / int8 / int4 / int16 / bf16 / tf32; tiled / mt / resident / streaming)
  rocket_conv.h                general 2D conv API (fp16 CONV_2D + native depthwise + ConvTranspose2d; per-call or resident-BO ctx)
  rocket_pool.h                on-NPU MaxPool / AveragePool via the PPU pooling engine (fp16 + int8/uint8)
  rocket_reduce.h              spatial GlobalAvgPool/Mean + GlobalMax/MinPool (ReduceMax/Min) over [H,W] (PPU) + FEATURE-axis reduce + CUMSUM (prefix sum) over the hidden axis (ones-/triangular-matmul)
  rocket_norm.h                on-NPU RMSNorm + LayerNorm + the per-row broadcast scale primitive
  rocket_normvision.h          on-NPU vision normalization: BatchNorm / GroupNorm / InstanceNorm / L2-Normalize
  rocket_ffn.h                 on-NPU gated-MLP FFN block (GeGLU/SwiGLU core + the three projections)
  rocket_softmax.h             on-NPU row-wise softmax + LogSoftmax + stable cross-entropy (EXP LUT + row-sum reduce + per-row scale / host gather)
  rocket_attn.h                on-NPU multi-head self-attention (Whisper encoder) + masked GQA flash attention (decoder/LLM prefill)
  rocket_encoder.h             one Whisper/transformer encoder block (LN->MHA->residual->LN->MLP->residual)
  rocket_siglip.h              SigLIP-B/16 vision encoder end-to-end (patch-embed + pos + 12xblock + post-LN)
  rocket_modernbert.h          ModernBERT text encoder end-to-end (RoPE, global/sliding attention, GeGLU) + a pre-norm post stack, variable-length batch
  rocket_activation.h          elementwise fp16 activation via the DPU LUT (sigmoid/hardsigmoid/tanh/silu/gelu/leaky/sqrt/rsqrt/recip/exp/log; auto-tiled for any n)
  npu_matmul.h                 low-level matmul_params_t + the regcmd generators
  npu_activation.h             low-level lut_act_params_t + gen_lut_activation_fp16 / gen_ew_mul_fp16
  npu_hw.h, npu_cna.h, npu_dpu.h   register/precision/layout constants

src/
  rocket_npu.c                 the shim over /dev/accel/accel0
  rocket_hw_profile.c          the one RK3588 hw-profile instance + rocket_hw_current() accessor
  npu_regcmd.c                 the regcmd generators: gen_matmul_fp16 / _int8 / _int4
                               / _int16 / _bf16 / _tf32 (fp16 EW K-accum + int8/int4
                               + the float bf16/tf32 + int16 paths) + gen_conv2d_fp16 /
                               _dw_fp16 + gen_conv2d_int8 / _dw_int8 (native int8 CONV_2D:
                               DIRECT int32-raw + DEPTHWISE int8-OUT on-chip-requant, both
                               HW-validated, with runtime wrappers)
  rocket_conv.c                general fp16 CONV_2D + native depthwise: SAME/VALID pad,
                               OC%16 pad, OC/OH/OW + DW channel/spatial tiling, resident-BO ctx
  rocket_conv_transpose.c      ConvTranspose2d (deconv): host dilate+flip lowering onto
                               the forward conv (direct + depthwise; stride/pad/opad/dilation),
                               the CNA deconvolution route, and the resident entry (col2im
                               over the resident fp16 matmul in channel-planes form)
  rocket_resize.c              nearest / bilinear upsample via depthwise ConvTranspose
                               (box / separable-triangle kernel; FPN/decoder neck)
  rocket_pool.c                on-NPU MaxPool / AveragePool (PPU): single self-contained job
  rocket_reduce.c              spatial GlobalAvgPool/Mean + GlobalMax/MinPool (ReduceMax/Min: idempotent,
                               bit-exact), telescoping multi-pass PPU reduction, kernel cap 16, plus the
                               FEATURE-axis reduce (ones-vector matmul, fp32-accum) and CUMSUM/prefix sum
                               (the same matmul widened to a triangular ones matrix; incl/excl/reverse)
  rocket_norm.c                on-NPU RMSNorm + LayerNorm (stacked-row reduce + affine fold) + per-row scale
  rocket_normvision.c          on-NPU BatchNorm / GroupNorm / InstanceNorm / L2-Normalize (vision normalization)
  rocket_ffn.c                 on-NPU gated-MLP FFN: the GeGLU/SwiGLU core act(gate)⊙up + the projections
  rocket_softmax.c             on-NPU row-wise softmax: host row-max -> NPU exp -> row-sum -> host 1/s -> NPU scale
                               (+ LogSoftmax: host log(s) + per-row ew_sub; + stable cross-entropy: logsumexp + host gather)
  rocket_attn.c                on-NPU self-attention (QKV/score/P·V matmuls + per-head softmax); masked GQA flash attention (scale->softcap->mask->softmax->P·V)
  rocket_encoder.c             one Whisper encoder block: LN -> MHA -> residual -> LN -> MLP(GELU) -> residual
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
already, one build targets one driver, and an indirect call on every BO operation would buy
nothing.

**The seam is exactly the externally-visible `rocket_*` functions defined in
`src/rocket_npu.c`, and a conforming provider defines exactly that set.** Nothing is missing,
and it adds no `rocket_*` name of its own that the library already defines.

`rocket_npu.h` is where they are declared, and the header is not the definition of the
seam. It also declares host-side entry points that stay in the core whatever the driver is
(`rocket_affinity_*`, `rocket_pin_worker`, `rocket_num_big_cores`, `rocket_busy_poll_set_us`,
`rocket_fence_wait_*`).
Do not enumerate the seam by reading the header, and do not enumerate it by hand. The list
grows, and a hand-kept copy goes stale silently:

```sh
tools/provider-seam.sh                          # print the seam
tools/provider-seam.sh path/to/provider.a       # check a provider (.a / .o / .so / .c)
```

Groups, as they stand:

- Device open and close.
- BO allocation, and the 32-bit-IOVA variants.
- Cache maintenance, including the ranged form and its capability query.
- Submit in its matmul, task, pre-arranged and job forms, plus the scratch-size query.
- The capability probes.
- The submit counters.

**Why this is checked rather than trusted.** A provider *defines* these symbols rather than
calling them, so when the seam grows the provider still compiles clean. `librocketnpu.a` is a
static archive, which is allowed to carry unresolved symbols, so the library still builds too.
The break surfaces only in whoever links the two together, as an undefined reference in every
executable at once. Configure runs the check for an external provider that names a file, and
fails with the missing symbols listed before an object is compiled.

## Public matmul API (`rocket_matmul.h`)

| function | what it is |
|---|---|
| `rocket_matmul_fp16` | tiled single-fd fp16 matmul (the validated core) |
| `rocket_matmul_fp16_f32out` | fp16xfp16 -> **fp32** output: full fp32 accumulator out (no per-K-tile/final fp16 rounding); ~5000x more accurate than fp16-out (tracks the true result to the fp32 noise floor ~1e-7); opt-in (2x output readback, free for tall-skinny prefill, +35% only when output bytes rival the weights) |
| `rocket_matmul_fp16_mt` | multicore: split N across `nthreads` worker fds (~3 cores) |
| `rocket_matmul_fp16_batch` | run `nbatch` independent same-shape matmuls as ONE NPU job (one submit + one fence; one IRQ under `ROCKET_BATCH_SUBMIT`); collapses a set of small, dispatch-bound GEMMs that share a shape. Bit-identical to per-item `rocket_matmul_fp16`. Backs flash attention's per-worker QK/AV submit batching. `rocket_mm_batch_create/_run/_free` is the persistent form: resident in/wt/out BOs grown only on a larger shape + a prezero-once guard that skips the full-BO zero when the `(M,K,N,nbatch)` layout repeats; for a repeated-shape caller (flash attention's per-layer QK/AV) it removes the per-call BO alloc + zero (the one-shot is a thin wrapper over it) |
| `rocket_ctx` / `rocket_weights_pack` / `rocket_matmul_fp16_prepacked` | pack weights once, reuse across calls (resident fp16). The compute M may differ from the pack M, because the resident layout is M-independent for M>=256, so a weight packed at warmup-M is reused at prefill-M with no re-pack; an incompatible tiling returns -2 |
| `rocket_ctx_create_ex(nthreads, ROCKET_CTX_TILING_CANONICAL)` | a ctx whose resident weights are M-independent down to M=4. At M < 256 the plan keeps the M=256 plan's K and N tiling (Mt = M), so one pack serves every call M and `_prepacked` never returns -2. The price is the K tile a small M gives up: 1.16-1.42x a pack made for that M at M 48-128, parity from 196 (`tests/matmul_prepacked_canonical_rocket.c`, every element exact over 9 shapes and M 4-640; `ROCKET_CANON_BENCH=21` prints the price). For a resident encoder whose length varies per call |
| `rocket_stream` / `rocket_matmul_fp16_stream[_fused]` | streaming path for LLM prefill (re-pack B, but cache scratch per shape; fuse gate/up) |
| `rocket_matmul_int8` / `rocket_matmul_plan_int8` | int8xint8->int32 tiled (pre-quantized in, raw int32 out) |
| `rocket_i8_ctx` / `rocket_matmul_int8_prepacked` | resident int8 (W8A8) weights. The compute M may differ from the pack M, because the resident tile layout is M-independent (canonical-tileM: the tiling is planned at `MAX_TILE`), so a weight packed at warmup-M is reused at any prefill-M with no re-pack (int32 K-accum is exact for any tiling -> bit-exact); an incompatible tiling returns -2. The compute M must still be a **positive multiple of 4**: an unaligned M miscomputes on the HW height geometry, so it is rejected (-1), not padded (padding M would need a matching pad of `a_scale`, which only the caller has); pad ragged row counts with `rocket_pad_m` |
| `rocket_matmul_int8_groupwise` / `rocket_matmul_plan_int8_gw` | **per-K-group int8 dequant scales**, fp32-accumulated: `C_f[m,n] = Σ_g a_scale[m,g]·b_scale[n,g]·(int32 partial of K-group g)`. The primitive a **natively quantized** weight needs, since a GGUF MXFP4/Q8_0/Q4_K block carries one scale per K-block, and the NPU cannot apply a K-blocked scale on-chip (at the output stage K is fully contracted). Integer partials already leave the chip at every K-tile boundary (no on-device integer K-accum is implemented), so the block scale is free at a boundary already being paid for: it fuses into the readback accumulate and costs **+0.6%** over per-channel int8. Unlike the int4 twin there is **no saturation bound on `group`** (the output accumulator is int32, not int16), and `Kt` need only **divide** the group, not equal it (a K-tile must lie *inside* one group, not *be* one), so a group wider than the CBUF cap stays legal. `rocket_matmul_plan_int8_gw` previews the tiling (pure, no HW) and reports the `Kt` it chose |
| `rocket_i8_weights_pack_gw` / `rocket_matmul_int8_prepacked_gw` | **resident group-wise int8**: the int8 codes are scattered into NPU BOs once and never leave, so a quantized weight costs **no per-forward-pass dequant and no per-call weight scatter**, the point of the path. Each call quantizes only A (per row, per K-group) and returns the fp32 per-group dequant. Same M-independence and the same M%4 contract as the per-channel form above. Bit-exact vs the one-shot oracle when the group fits the CBUF at the worst-case tile (which forces `Kt == group` in both paths); a wider group can land on a different `Kt` divisor, and the two then agree to fp32 reassociation |
| `rocket_matmul_int4` / `rocket_matmul_int4_ex` | int4xint4->int16 tiled (host-accum to int32); `_ex` adds the int16-saturation Kt cap (in-model `[-7,7]` needs `kt_cap=480`) |
| `rocket_matmul_int4_groupwise` | per-K-group int4 dequant scales, fp32-accumulated (the W4A4 quality lever; group = the saturation-safe K-tile) |
| `rocket_i4_ctx` / `rocket_matmul_int4_prepacked` | resident int4 (W4A4) weights, raw int32 out (per-channel) |
| `rocket_i4_weights_pack_gw` / `rocket_matmul_int4_prepacked_gw` | **resident group-wise int4** (the in-model W4A4 path): weight packed once with the K-tile forced to `group`, each call quantizes only A and returns the fp32 per-group dequant `Σ_g a_scale[m,g]·b_scale[n,g]·partial`. Hadamard (baked into the resident weight + applied to A by the caller) is product-preserving, so no driver support is needed. Like the int8 path, the resident layout is M-independent (canonical-tileM), so a weight packed at warmup-M is reused at any prefill-M (incl. a small short-prompt M) with no re-pack |
| `rocket_matmul_int16_exact` | bit-exact int16xint16->int64 via int8 byte-decomposition (4 int8 matmuls) |
| `rocket_matmul_bf16` / `rocket_matmul_plan_bf16` | bf16xbf16->fp32 tiled, single-fd (fp32 A/B in, truncated to bf16 on scatter; fp32 range, no activation scaling) |
| `rocket_matmul_bf16_mt` | multicore bf16: split N across `nthreads` worker fds over the unchanged single-fd path (~3 cores) |
| `rocket_bf16_stream` / `rocket_matmul_bf16_stream` | **streaming bf16** for LLM prefill: persistent worker fds + per-shape resident scratch BOs, re-pack A/B per call (the bf16 sibling of `rocket_matmul_fp16_stream`; bit-identical to single-fd at `nthreads=1`). The in-model bf16 path. ~3.2x single-fd warm |
| `rocket_matmul_tf32` / `rocket_matmul_plan_tf32` | tf32xtf32->fp32 tiled (raw fp32 in, HW rounds to 10-bit mantissa; the first 4-byte-input path) |
| `rocket_matmul_int8_rk3576` / `rocket_matmul_plan_int8_rk3576` | **the RK3576's own int8 matmul**, and a different contract from `rocket_matmul_int8`: `C = sat8(round((A·Bᵀ + bias)·scale))`, an int8 surface through the DPU's requant rather than raw int32. That is why it is a separate entry and not a routing. Requires `K%32, N%32`; **M carries no constraint at all** on that part, so `M=1` is simply correct. N is tiled and is what buys throughput (a submit costs ~1.4 ms whatever it carries; `ROCKET_RK3576_MM_NT` overrides the tile). K past one task's contraction is split through the int32 entry below, with the requant done on the host |
| `rocket_matmul_fp16_rk3576` | **the RK3576's fp16 matmul**: `C = A·Bᵀ` with fp16 A and B and an **fp32** C, all row-major, B one row per output channel. The whole of K in one task, where the fp16 convolution contracts sixteen channels a task. Accepts `1 <= M <= 2048`, `32 <= K <= 6144` with `K%32`, `64 <= N <= 8192` with `N%16`, and returns `ROCKET_E_UNSUPPORTED` outside them. K and N are one task's measured bounds. A task carries at most `4096/(K/32)` rows, the CBUF data window, so the entry cuts M into equal tasks under that and submits them as one job (`rocket_rk3576_mm_fp16_task_rows()`). A call whose output still holds the sentinel is redone after a power cycle, and `rocket_rk3576_mm_fp16_last_attempts()` reports how many submits the last call took. `ROCKET_MM_PROFILE` prints its buckets. Its output does not poison the next submit |
| `rocket_matmul_fp16_rk3576_wbo` / `rocket_rk3576_wbo_fp16_create` / `rocket_rk3576_wbo_fp16_free` | **the same fp16 matmul with resident weights**: `create` permutes B into the part's layout once and builds the coefficient buffer for its N, both in device BOs, and the entry takes that handle in place of B. Bit-identical to `rocket_matmul_fp16_rk3576`, with M free per call inside the envelope. A K or N outside the envelope refuses at `create`, and a call whose K or N differs from the handle's refuses with `ROCKET_E_SHAPE`. The handle is not checked against the weight it came from, so a stale one computes a plausible surface from the old weight. Each handle holds K·N·2 bytes of weight and about 257 KiB of coefficients |
| `rocket_matmul_int8_rk3576_i32` | **the RK3576's int32-output matmul, and its K split**: `C = A·Bᵀ + bias` in raw int32, K split internally and the partials summed on the host, so K is bounded only by memory. Requires `K%32, N%32`. The DPU's 32-bit writer keeps the INT8 surface's byte budget whatever the element width is, so it delivers only the first eight output channels of every thirty-two; this entry programs four times the output channels and scatters the real ones into the delivered slots, which is correct and costs a quarter of the int8 path's MACs per submit. It also idles ~150 ms between its submits and once on the way out, because an int32 job leaves the next submit of ANY kind writing nothing until 50-100 ms have passed (`ROCKET_RK3576_MM_GAP_MS` overrides). Use it for the K a single task cannot contract, not as a default matmul |

Alignment requirements differ by dtype, following the native tile atoms:

| dtype | Alignment |
|---|---|
| fp16 | `K%32, N%16` |
| int8 | `K%32, N%32` |
| int4 | `K%32, N%64` |
| int16-exact | `K%32, N%32`, following int8 |
| bf16 | `K%32, N%16`, the same as fp16 |
| tf32 | `K%16, N%16`, where the 4-byte input halves the K-group to 16 |

All require `M%4==0`. **`M==1` (single-vector GEMV) is broken on the hardware** at every
dtype, because the conv feature-height-1 geometry mis-computes. The one-shot entry points
therefore **pad M==1->4 internally** and return row 0. The pure planners and the resident and
streaming paths reject it instead, so pad single vectors to 4 caller-side. The plan functions are pure (no
hardware) and preview the tiling.

Everything in that table except the RK3576 entries emits the **RK3588**
geometry-register encoding. On a part that does not run it the entries **refuse**
(`ROCKET_E_UNSUPPORTED`, naming the entry that does work) rather than submit a job that
completes and writes nothing. The M rule above is one of the things that does not carry:
on the RK3576 every M computes, including 1.

## Conv API (`rocket_conv.h`)

Beyond the matmul, the library runs a general 2D convolution, the basis of the
`tflite-rocket` detection delegate. A 1×1 pointwise conv is the matmul, and everything else
is the conv path. HW-validated **bit-exact**.

| function | what it is |
|---|---|
| `rocket_conv2d_fp16` | general fp16 `CONV_2D`: KxK / stride / dilation, symmetric pad, OC%16 pad, OC/OH/OW spatial tiling; allocs+frees its 5 BOs per call |
| `rocket_conv1d_fp16(fd, ic, it, oc, kw, stride, pad, …)` | **conv1d** (the Whisper encoder front-end: width-only 1D conv over time), lowered onto `rocket_conv2d_fp16` with **time on the height axis** (IW=1, so the OH-row tiler fits CBUF; the IH=1 width layout overflows the feature banks for Whisper IC=80/512). Whisper conv1/conv2 (KW=3, pad=1, stride 1/2) HW-validated (`tests/conv1d_rocket.c`) |
| `rocket_conv2d_fp16` (`desc.depthwise=1`) | native depthwise (`DW_EN`, group G=32 + the DPU depthwise register fixes), channel + spatial tiled |
| `rocket_conv2d_act_fp16(fd, d, kind, …)` / `_ctx` | **conv->activation FUSION**: a DIRECT fp16 conv post-processes its own CACC result with a smooth `f(x)` in the SAME NPU job (the DPU LUT epilogue ported into `gen_conv2d_task`): `out = f(conv(x))`, no 2nd round-trip. `kind` = `SILU`/`TANH`/`GELU`. HW-validated: the fused epilogue **bit-reproduces the standalone LUT** (<=0.0039), conv->tanh bit-accurate (`tests/conv_act_rocket.c`). Default-off byte-identical. HardSwish/depthwise rejected (flat-tail quirk / direct-only). *Caveat: a narrow x~0 LE/LO boundary glitch spikes all single-pass kinds; the 2-pass `x·gate(x)` route avoids it.* |
| `rocket_conv2d_int8` | native **int8 DIRECT** `CONV_2D`: int8xint8->int32 raw accumulate (caller requants); OC pad 32 / IC pad 32, OC/OH/OW tiling. Bit-exact vs an int64 oracle. *(Also the substrate for the `tflite-rocket` delegate's native **uint8** convs, Option D: the caller recenters uint8->int8 `byte−128` and folds the centering into its requant + a box-sum; no driver change.)* |
| `rocket_conv2d_dw_int8` | native **int8 DEPTHWISE** (int8-out, on-chip requant): per-tensor quant, any int8 weight zero point (the DPU's CPEND operand adds `-w_zp` times each window's input sum), the input zero point folded into the bias; TFLite's accumulator exactly, within one of TFLite at a rounding boundary. A uint8 tensor maps onto it by subtracting 128 from its values and zero points |
| `rocket_conv2d_dw_int8_plan` | the pure shape check behind `rocket_conv2d_dw_int8`: `ROCKET_OK` when its NPU path takes the descriptor, `ROCKET_E_SHAPE` or `ROCKET_E_UNSUPPORTED` otherwise. Opens no device, so a frontend claims a node with it |
| `rocket_conv2d_int8_q` / `_ctx` / `_mt` / `_plan` | native **int8 DIRECT conv with the int8-out writer** (RK3588): Mesa's direct program, requantized on chip through the OUT_CVT, any per-tensor weight zero point on CPEND, the input zero point folded into the bias. Shares the int32-raw entry's planner, tiler and job body; programs whole 32-kernel groups and materializes the halo (the input zero point on real channels, 0 on the channel padding). Within one of TFLite at a rounding boundary |
| `rocket_conv2d_int8_q_perc` / `_perc_ctx` / `_perc_mt` / `_perc_plan` | native **int8 DIRECT conv with per-channel weight scales** (TFLite's per-axis quantization), RK3588: the int8-out program of `rocket_conv2d_int8_q` with each output channel's scale on the BS stage's int16 multiplier and one OUT_CVT gain per job. Channels sorted by scale and cut into jobs at `ROCKET_DW_PERC_MIN_C` (2048); each job whole 32-kernel groups, every job's tiles in one dispatch across the pool; a layer that plans to one job keeps its channel order and the one-job fast path. Within one of TFLite at a rounding boundary; an all-zero filter's channel is written by the host |
| `rocket_conv2d_dw_int8_perc` / `_perc_ctx` / `_perc_plan` | native **int8 DEPTHWISE with per-channel weight scales** (TFLite's per-axis quantization), RK3588: each channel's scale rides on an int16 multiplier the DPU's BS stage reads per channel, and one OUT_CVT gain per job; channels sorted by scale and cut into jobs so each keeps a multiplier of at least `ROCKET_DW_PERC_MIN_C` (2048). Within one of TFLite at a rounding boundary; an all-zero filter is written by the host as its bias's constant |
| `rocket_conv_ctx_create` / `rocket_conv2d_{fp16,int8}_ctx` / `_dw_int8_ctx` | the same convs with a **resident** BO pool reused across calls/tiles (ctx=NULL ⇒ byte-identical legacy alloc-per-call) |
| `rocket_conv_pool_create` / `rocket_conv2d_int8_mt` / `rocket_conv_pool_free` | **multicore** int8/uint8 DIRECT conv: a pool of N worker fds (each its own resident `rocket_conv_ctx`) fans the conv's independent OC/OH/OW tiles across the 3 NPU cores. **Bit-identical** to `rocket_conv2d_int8` (same tiles/jobs); falls back to serial for single-tile convs. Mirrors `rocket_matmul_fp16_mt`'s one-fd-per-core design |
| `rocket_conv_transpose2d_fp16(fd, d, in, W, out)` / `_ctx` | **ConvTranspose2d / deconv** (learned upsampling: segmentation/decoder/super-res, FPN learned-upsample). Two routes, both bit-exact vs a scatter reference (`tests/conv_transpose_rocket.c`). On the RK3588 a **direct** transpose at power-of-two strides (2, 4 or 8), dilation 1, `IC%32==0` and a compact input that fits one CBUF pass runs on the CNA's **hardware deconvolution mode**, one task per output-channel tile: 0.35-0.54x the lowering's wall at decoder shapes. Everything else is lowered to interior-dilate-input + `rot180(Wᵀ)` + a **stride-1 forward `rocket_conv2d_fp16`**, inheriting the HW-exact conv tiling; its cost scales with the *upsampled* size. Weights `[IC][OC][KH][KW]` (in-channels first) direct or `[C][1][KH][KW]` depthwise (`desc.depthwise=1`, OC==IC, C%32). Supports stride/pad/output_padding/dilation, any OC/IC. `_plan` returns `−2` for the unimplemented `pad > d·(K−1)` crop case (clean decline) |
| `rocket_conv_transpose2d_route(d)` | which of the two routes a descriptor takes (`ROCKET_CONV_TRANSPOSE_DECONV` / `_LOWERED`), or the plan's negative refusal. Pure. `ROCKET_CONV_TRANSPOSE_HW=0` forces the lowering |
| `rocket_conv_transpose2d_weights_pack(ctx, d, W, bias)` / `rocket_conv_transpose2d_fp16_prepacked(ctx, w, in, out)` / `_weights_free` / `_prepacked_plan(d)` | **the resident-weight ConvTranspose**, for a model that runs the same layer repeatedly: one GEMM, `in^T . W` with the input read as IC planes of IH·IW pixels and the weight as stored (`[IC][OC·KH·KW]`), which is every tap of every input pixel with no inserted zeros, then a host scatter-add of each tap's plane into the output. The GEMM is the resident fp16 matmul on the caller's `rocket_ctx`, in channel-planes form (no host transpose on either side), the weight transposed and packed **once** into the handle with an optional `[OC]` bias; the workers split M (each holding the whole weight) once each has a full `max_tile` of rows, N otherwise (`ROCKET_CT_SPLIT=m\|n`). Any stride, pad (a cropping pad included), dilation and output_padding. Per call at pix2pix's and SAM's ten layers **0.94-9.16 ms, 0.11-0.58x ONNX Runtime's 2-thread per-node time on the same board** and 0.02-0.29x the one-shot entry's [HW sweep, mainline RK1, five rotated passes, 2026-09-27]. Within a few fp16 roundings of each element's magnitude sum (worst 2^-11.1 measured), bit-exact on small-integer data (`tests/conv_transpose_resident.c`). The plan refuses depthwise and non-RK3588 parts (`ROCKET_E_UNSUPPORTED`) and a tap-plane buffer past 64 MiB (-4). The handle never re-reads W: a changed weight is a new pack |
| `rocket_upsample_nearest_fp16` / `rocket_upsample_bilinear_fp16` (`rocket_resize.h`) | **integer-factor resize** (the FPN/decoder neck: `RESIZE_NEAREST_NEIGHBOR` / `RESIZE_BILINEAR`). Realised as a **depthwise ConvTranspose** with a fixed box (nearest) / separable-triangle (bilinear) kernel, where `pad=(k−s)/2` gives exactly `IH·s x IW·s`. Nearest = bit-exact block replication; bilinear = half-pixel 2-tap (the triangle's stride-subsample is a partition of unity), `align_corners=False` interior + zero boundary. `C%32` (depthwise group). HW-validated vs independent gather refs + partition-of-unity / linear-exactness properties (`tests/resize_rocket.c`) |

Pure planners (`rocket_conv2d_plan` / `_oh` / `_ow` / `rocket_total_pad`, plus
`rocket_conv_transpose2d_plan` / `_oh` / `_ow`) preview dims + the CBUF-fit gate without
hardware. `rocket_conv2d_ref_int8` / `rocket_conv_transpose2d_ref_fp16` are golden oracles.

Native **int8** `CONV_2D` runs end-to-end and is HW-validated, both the regcmd and
cube layers (`gen_conv2d_int8` / `gen_conv2d_dw_int8`) and the runtime wrappers
(`rocket_conv2d_int8` / `rocket_conv2d_dw_int8`). **DIRECT** = int8xint8->int32 raw + host
per-axis requant, with the input zero-point correction folded into the bias, in
`tflite-rocket`'s `rocket_out_nchw_to_nhwc_q_per_axis`. That is a real int8 accumulate,
bit-identical to CPU TFLite.

**DEPTHWISE** = int8-OUT with on-chip requant
(`conv_params_t.int8_out=1`: QD_EN + per-OC int32 bias in the BS ALU + `OUT_CVT` requant),
per-tensor with symmetric weights. Its accumulator is TFLite's exactly and its requant is
the DPU's. So an output can differ from CPU TFLite by one at a rounding boundary
(`tests/conv_dw_int8_runtime.c`). `tests/replay_dw_mesa.c` checks the register program
against Mesa's. The `tflite-rocket` delegate routes signed-int8 convs to these under
`--option native_int8=1`.

## Activation API (`rocket_activation.h`)

The first **on-NPU nonlinear activation** on this stack. It is a standalone DPU **LUT**
pass, NVDLA SDP with no conv, that applies `f(x)` to a flat fp16 vector entirely on the
NPU.

| function | what |
|---|---|
| `rocket_activation_fp16(fd, kind, in, out, n)` | elementwise fp16 activation. **`SIGMOID`** (max_abs 0.00146) + **`HARDSIGMOID`** (0.00049) run fully on the NPU LUT; **`HARDSWISH`** (0.00098), **`SILU`**, and **`GELU`** (exact erf) run the **2-pass** route `x·gate(x)` (gate = sigmoid/hardsigmoid/Φ on the NPU LUT, the multiply on host by default or **fully on the NPU** with `ROCKET_ACT_NPU_MUL=1`); **`TANH`** is single-pass. **GELU is the accurate 2-pass `x·Φ(x)`** (Φ = the Gaussian CDF on the clean unit-LUT geometry): cos=1.000000 vs true erf-GELU over `[-12,12]` (`tests/gelu_rocket.c`); the SINGLE-pass GELU spikes ~128 in the flat tail (QUIRK 1, `ROCKET_ACT_WIDE_LUT` for RE only). All HW-validated (`tests/activation_lut_rocket.c`, `lut_tanh_rocket.c`, `gelu_rocket.c`). `n` padded to a multiple of 8 |
| `rocket_activation_fp16`, **positive-domain kinds** `SQRT` / `RSQRT` / `RECIPROCAL` | `f(x)` for `x>0` via the **shifted single-table** LUT (the whole domain maps onto the positive index half ⇒ no LE/LO glitch). The DPU LUT *does* compute the reciprocal family. Uniform-grid ⇒ accuracy is domain-bounded: **<1% over a ~100-200x range** placed away from the steep near-0 region (HW: sqrt 0.85% / rsqrt 0.44% / recip 1.0%), tune with `ROCKET_LUT_XLO/XHI`. `RSQRT` is the **RMSNorm/LayerNorm** core; `RECIPROCAL` the softmax-denominator / Div core (`tests/recip_rsqrt_rocket.c`) |
| `rocket_activation_fp16`, **`EXP`** | `exp(x)` via the same shifted single-table, default domain `[-16,0]` (the **softmax** case: input <=0 after the row-max subtraction). Works on the standalone flying path (unlike GELU, with no LE/LO sign-mux glitch). exp's relative interp error is ~constant (`Δ²/8` ~1e-4). A LUT table entry of exactly **q=0 mis-decodes to a garbage ~4.0**: exp's deep tail quantizes to q=0 and reads ~4, so every shifted-table entry is **floored to q>=1** (`ROCKET_LUT_QFLOOR`, default 1; a no-op for sqrt/rsqrt/recip). softmax-sum end-to-end rel <=0.04% (`tests/exp_lut_rocket.c`) |
| `rocket_activation_fp16`, **`LOG`** | `ln(x)`, `x>0`, the natural inverse of EXP (log-probabilities / NLL / cross-entropy). Same shifted single-table, but the **first signed-output kind on the positive-domain path**: `log(x)<0` for `x<1`, so `out_lo=log(x_lo)` is negative and the OUT_CVT offset decodes the signed range (the tanh/ELU machinery, now exercised here). Default domain `[0.25,32]`; uniform-grid ⇒ **absolute** error is the metric (relative is ill-defined at the `x=1` zero crossing). HW **max_abs 0.0066 / mean 0.0007** over `[0.3,30]`, worst at the steep small-x end (`tests/recip_rsqrt_rocket.c`) |
| `rocket_leaky_relu_fp16(fd, alpha, in, out, n)` | **LeakyReLU** `x>=0?x:alpha*x` (YOLO; ONNX LeakyRelu/PRelu scalar slope), one DPU LUT pass on the natural LE/LO split (LE=`alpha*x`, LO=`x`). `alpha>0`; exact over [-R,R] (R via `ROCKET_LEAKY_R`, default 16), saturates beyond. The LUT's **sign-based LE/LO mux** spikes at exactly x~0 -> repaired on the host (the readback streams every element anyway). HW-validated alpha 0.01-0.5 (`tests/leaky_relu_rocket.c`) |
| `rocket_activation_fp16`, **`SOFTPLUS`** / **`MISH`** / **`ABS`** | **Softplus** `log(1+e^x)` (shifted single-table, `out_lo=0`, x~0-clean; max_rel 0.14%); **Mish** `x·tanh(softplus(x))`, the **YOLOv4/v7** backbone activation, 2-pass (a `[0,1]` gate LUT + EW-mul, max_rel 0.06%); **Abs** `\|x\|` (symmetric shifted single-table, the x=0 kink on the middle sample, max_abs 1e-3). All HW-validated (`tests/softplus_mish_rocket.c`) |
| `rocket_elu_fp16(fd, alpha, in, out, n)` / `rocket_selu_fp16(fd, in, out, n)` | **ELU** `x>=0?x:alpha*(e^x-1)` and **SELU** `λ·ELU_α` (fixed self-normalizing α=1.673, λ=1.051), on the **symmetric shifted single-table**. Negative outputs ⇒ a signed OUT_CVT, which keeps the x~0 mux spike -> **host x~0 repair** (`ROCKET_ELU_NOREPAIR` disables). Exact over [-R,R] (`ROCKET_ELU_R`, default 8). HW-validated alpha 0.5-2.0 (`tests/elu_rocket.c`) |
| `rocket_ew_add_fp16` / `rocket_ew_sub_fp16` / `rocket_ew_mul_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise **add** (residuals) / **subtract** / **multiply** (gated activations), flat fp16 vectors. All use the conv-main EW path (identity-matmul main feed + ERDMA operand, `EW_OP_TYPE` add/mul); SUB reuses the ADD datapath with the operand negated (`a-b==a+(-b)`, exact fp16 sign flip); n reshaped to [M,32] + M-tiled. **Bit-exact** vs host (`tests/ew_mul_rocket.c` runtime check sweeps add/sub/mul, n up to 40000 across the M-tile boundary) |
| `rocket_ew_div_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise **divide** `a/b` = `reciprocal(b)` (DPU LUT) then `a*recip` (EW mul). `b` positive, within the reciprocal LUT domain. fp16-LUT-approx (HW max_rel 0.35% for `b∈[0.5,8]`); covers TFLite/ONNX `DIV` (`tests/recip_rsqrt_rocket.c`) |
| `rocket_ew_max_fp16` / `rocket_ew_min_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise two-tensor **max/min** `out=max/min(a,b)`. The DPU EW **ALU algo** field (`DPU_EW_CFG` bits[17:16]) reaches **MAX(0)/MIN(1)**, not just SUM(2=add): same conv-main EW datapath, one word changed. **Bit-exact** (they select an operand; `tests/ew_minmax_rocket.c`, n to 40000). Covers TFLite/ONNX `Maximum`/`Minimum` and `ReLU=max(x,0)` |
| `rocket_clip_fp16(fd, lo, hi, in, out, n)` | **Clip** `min(max(x,lo),hi)` on the NPU (constant-operand MAX then MIN, two EW passes). Bit-exact; covers TFLite/ONNX `Clip` and the bounded-ReLU family (ReLU6=`Clip(0,6)`) (`tests/ew_minmax_rocket.c`) |
| `rocket_prelu_fp16(fd, C, S, x, alpha, out)` | **PReLU** with a **per-channel** slope `alpha[C]` (YOLO/segmentation; ONNX PRelu), input `[C][S]`. **No LUT** (so no x~0 glitch): for the universal `alpha∈[0,1]` it is `max(x, alpha_c·x)` (per-channel scale via row-broadcast `ew_mul`, then `ew_max`): 2 passes; any `alpha` outside [0,1] falls back to `relu(x)+alpha_c·min(x,0)`. **Bit-exact** (`tests/prelu_rocket.c`) |
| `rocket_lut_epilogue_build(kind, lut, ep)` | builds the LE/LO tables + DPU epilogue constants for a SMOOTH single-pass kind (SiLU/tanh/GELU). The **single source** of the single-pass LUT params, shared by the standalone op and the **conv->activation fusion** (`rocket_conv2d_act_fp16`), so they can't drift |
| `rocket_ew_mul_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise fp16 multiply `out=a*b` (identity-conv main feed + `EW_OP_TYPE=1`; M-tiled). Bit-exact (`tests/ew_mul_rocket.c`). The building block of on-NPU HardSwish/SiLU |
| `gen_lut_activation_fp16` (`npu_activation.h`) | low-level: the flying-mode DPU LUT regcmd (LE/LO hybrid tables, BN-mul index, OUT_CVT Q0.15->fp16). The same epilogue is fused into `gen_conv2d_task` (`lut_epilogue_t` / `npu_dpu_desc.lut_en`) for conv->activation |

### Any `n`, and the max-width quirk

A LUT op carries the vector as a cube of `cols = n/8` width positions, and
`DPU_DATA_CUBE_WIDTH` is 13-bit, so one op caps at `n <= 65528`. `run_dpu_lut` **tiles**
automatically, so every activation works at any `n`, and a transformer's `[M,I]` cube is
millions of elements. Riding the *exact* max width (`cols = 8191`) corrupts ~54 cube
positions, so the tile cap stays **well under** the ceiling (32768), bit-clean.

**GELU** runs the accurate **2-pass** `x·Φ(x)`. The single-pass GELU spikes ~128 in the flat
negative tail, quirk 1, which also makes a *fused* single-pass matmul->GELU spike for wide FFN
inputs. The 2-pass is the on-NPU GELU route.

A **fully-on-NPU** elementwise multiply (`rocket_ew_mul_fp16`) is **implemented and
HW-validated bit-exact** (`tests/ew_mul_rocket.c`). rocket's DPU EW reads its 2nd operand
only combined with a conv/CACC **main** feed. That is ERDMA+MRDMA `COMB_USE(5)`, the Teflon
`add_tensor` RE.

So `rocket_ew_mul_fp16` uses an **identity conv** as the main feed and sets the EW op to
multiply (`EW_OP_TYPE=1`, `DPU_EW_CFG=0x108003C4`). That is the same machinery as the fp16
K-accum eltwise-add with one register field changed, `gen_matmul_fp16`'s new `ew_mul` flag.
With it, **HardSwish/SiLU run fully on the NPU** under `ROCKET_ACT_NPU_MUL=1`, as a LUT
gate plus an EW mul with no host arithmetic. The host multiply stays the default, since a
standalone EW-mul costs a second NPU round-trip (the perf path is fusing the mul into the
producing conv). The flying-main `gen_ew_mul_fp16` (which reads 0, since there is no main feed)
stays behind `ROCKET_ACT_EXPERIMENTAL=1`, disabled by default.

## Transformer-block API (`rocket_reduce.h`, `rocket_norm.h`, `rocket_ffn.h`, `rocket_softmax.h`, `rocket_attn.h`, `rocket_encoder.h`)

On-NPU transformer primitives, built by composing the matmul / reduce / LUT / EW pieces:
the substrate for fused FFN / encoder blocks. Each is a CTest gate vs an fp64 oracle. Together they
**run a full Whisper/transformer encoder block on the NPU (cos = 1.000000)**.

| function | what it is |
|---|---|
| `rocket_reduce_feature_fp16(fd, M, H, in, out, mean)` | **feature-axis reduce** `sum_h x[m,h]` (or mean), per row -> fp32 `[M]`. The contraction RMSNorm/LayerNorm/softmax need, and the one the PPU **cannot** give (it pools spatial `[H,W]` *within* a channel, never across). Realised as a **ones-vector matmul** reusing `rocket_matmul_fp16_f32out` (no new regcmd, genuine fp32 K-accum). Essentially **bit-exact** (max_rel <= 1.4e-7) |
| `rocket_cumsum_fp16(fd, M, N, in, out, exclusive, reverse)` | **cumsum / prefix sum** along the last axis (TFLite/ONNX `CumSum`). The feature reduce **widened from a single ones-COLUMN to a full triangular ones MATRIX**: `out = in·Lᵀ`, `L[n][k]=1` iff column `k` is in prefix `n` (incl/excl x forward/reverse pick the triangle). Same `rocket_matmul_fp16_f32out` reuse (no new regcmd, fp32 K-accum for long prefixes). HW **bit-exact** (max_abs 0 across all variants incl. T=1500) (`tests/cumsum_rocket.c`) |
| `rocket_rmsnorm_fp16(fd, M, H, x, weight, eps, out)` | **RMSNorm** `x/sqrt(mean_h(x²)+eps)·weight[h]`. O(M·H) on the NPU (square->reduce->scale); the O(M) per-row rsqrt tail is **exact on the host** (sending M scalars to the DPU rsqrt LUT would add a round-trip *and* hit the LUT domain). fp16-square overflow (`\|x\|>256`) -> exact power-of-2 prescale. `weight` is the effective scale (Gemma: pass `1+w`). max_rel <= 3.5e-3 |
| `rocket_layernorm_fp16(fd, M, H, x, gamma, beta, eps, out)` | **LayerNorm** `(x−mean)/sqrt(var+eps)·gamma[h]+beta[h]` (the Whisper/encoder norm). **Both reductions share ONE feature-reduce job by STACKING rows**: `A=[x ; x⊙x]` (2M rows) under the ones weight -> first M = `sum(x)`, next M = `sum(x²)`. Host O(M) mean/var/rsqrt; the affine folds to `x⊙A+B` (one ew_mul + one ew_add). Same fp16-square overflow prescale as RMSNorm; `beta` may be NULL (`tests/layernorm_rocket.c`) |
| `rocket_scale_rows_fp16(fd, M, N, in, r, out)` | **per-row broadcast multiply** `out[m,n]=in[m,n]·r[m]` (`r` fp32 `[M]`), the FFN/attention/softmax post-scale (the RMSNorm 1/rms folds here; the weight folds into the next matmul). The reusable building block |
| `rocket_geglu_fp16(fd, gate, up, kind, prod, n)` | **gated activation** (GeGLU/SwiGLU core) `prod = act(gate)⊙up`, the only computation an FFN adds beyond matmul. `kind` = `SILU` (robust 2-pass) / `GELU`. Fully on the NPU (LUT + EW-mul) |
| `rocket_ffn_fp16(fd, M, H, I, x, Wg, Wu, Wd, kind, out)` | **gated-MLP FFN block** `gate=x·Wgᵀ -> act(gate)⊙(x·Wuᵀ) -> ·Wdᵀ`. Composes the three projections + the geglu core. **cos = 1.000000** vs fp64 (Gemma-ish 128×2048×1024). *Host handoff today; the cube-resident fusion (fewer round-trips) is the perf follow-on* |
| `rocket_softmax_fp16(fd, M, N, in, out)` | **row-wise softmax** over the last axis. Host row-max + subtract -> NPU `exp` (LUT) -> NPU row-sum (feature reduce) -> host `1/s` -> NPU per-row scale. The **row-max is on the host** (matmul/reduce can only SUM; the only on-NPU max is the PPU max-pool = the resident-fusion path). Validated to T=1500 (Whisper seq), rows sum to 1±5e-4 (`tests/softmax_rocket.c`) |
| `rocket_logsoftmax_fp16(fd, M, N, in, out)` | **row-wise LogSoftmax** `x − logsumexp(x)` (the classification / NLL-loss head, LM log-probs). Shares softmax steps 1-3 (host row-max+subtract -> NPU `exp` -> NPU row-sum), then a per-row **subtract** instead of a divide: host `ls=log(s)` (exact, O(M), like softmax's `1/s`, and **not** the DPU LOG LUT, which is for large-tensor log) -> NPU per-row `ew_sub`. All-additive ⇒ better-conditioned than softmax (no tiny-prob blow-up); HW max_abs <= 0.031, `Σexp(out)=1` (`tests/softmax_rocket.c`) |
| `rocket_cross_entropy_fp16(fd, M, N, logits, target, loss)` | **stable per-row cross-entropy** `CE[m]=logsumexp(logits[m]) − logits[m][target[m]]` = `−logsoftmax[target]` (softmax-classifier / LM NLL loss). The on-NPU logsumexp reduction (host row-max -> NPU `exp` -> NPU fp32 row-sum -> host `log(s)`) + a **host gather** of the target logit (there is **no HW gather**, so M scalar lookups, like the host `1/s`). Never materializes softmax (no divide); CE >= 0. **fp32-grade** (max_abs <= 1.5e-4, since the loss never round-trips through fp16 output storage) (`tests/cross_entropy_rocket.c`) |
| `rocket_mha_self_fp16(fd, T, d, n_head, x, Wq,bq, Wk,bk, Wv,bv, Wo,bo, out)` | **multi-head self-attention** (the pure attention sublayer): QKV proj + per-head `scale·(q·kᵀ)` + softmax + `P·v` + out-proj, **all matmuls + every softmax on the NPU**. Weights row-major `[out,in]`; biases optional (Whisper: bq,bv,bo, no bk). The **key count is padded to %32 and the pad score columns masked to −30000 before softmax** (the matmul rejects unaligned N/K, and Whisper T=1500 is unaligned). cos = 1.000000 at Whisper-base d=512/8-head incl. T%16≠0 (`tests/mha_rocket.c`) |
| `rocket_flash_attn_fp16(fd, n_tokens, n_kv, head_dim, n_head, n_kv_heads, scale, softcap, Q, K, V, mask, out)` | **masked grouped-query (flash) attention**, the decoder / LLM-prefill attention sublayer (the op an llama.cpp `FLASH_ATTN_EXT` lowers to). Already-projected Q/K/V + an **additive mask the caller supplies** (causal + sliding-window; this code does not synthesize it): per head `scale·(q·kᵀ)` -> optional `softcap·tanh` -> `+mask` -> softmax -> `P·v`, **all matmuls + softmax on the NPU**. Handles **GQA/MQA** (`n_head>n_kv_heads`, kv head `h/(n_head/n_kv_heads)`); head-major dense layouts so each head is a contiguous matmul operand; `n_tokens` padded to %4, `n_kv` to %32 (pad keys scored −∞). cos = 1.000000 vs an fp64 oracle at Gemma-4-12B shapes (head_dim 256, 16 q-heads, 8 kv-heads local / 1 global, sliding window, soft-cap) (`tests/flash_attn_rocket.c`). `rocket_flash_attn_fp16_mt(…, nthreads)` fans the heads across worker fds (the cores run head ranges in parallel); `rocket_fa_ctx` + `rocket_flash_attn_fp16_ctx` is the persistent form for repeated calls (worker fds held open + per-worker score scratch kept resident), all numerically identical. `ROCKET_FA_CHAIN` (on by default) batches each worker's per-head QK matmuls into one NPU job and its AV matmuls into a second (the mask + softmax sit between, preserving the host-softmax default): one submit + fence per head range instead of per head, through a per-worker resident batched-matmul context (`rocket_mm_batch`, BOs + score scratch held + prezeroed once). That attacks the small-GEMM dispatch floor: it nearly **doubles** the FA-op throughput vs a per-call batch (~2.0-3.8x the per-head path: 186->50 ms @512, 363->183 ms @2K head-range time), moving the FA-NPU-vs-CPU prefill crossover from ~6K to ~2K. The chaining win does **not** decay to nothing at depth: with the head group bounded by `ROCKET_FA_CHAIN_ELEMS` (default 32M score elems = a worker's ~3-head range batched up to ~20K context, ~150-200 MB/worker scratch), collapsing the per-head submits is **1.10x@4K / 1.47x@8K / 1.32x@16K / 1.16x@32K** at a 512-token ubatch (`fabench`, T=512, [HW sweep]), and +3% end-to-end pp8192 (Qwen3.5-0.8B-F16), with no short-context change (already batched <=2K). At a 2048-token ubatch each head's score alone exceeds the budget and the win shrinks to ~1.05x; raise the knob to chain further at a higher scratch cost. Numerically identical (cos = 1.000000); `ROCKET_FA_CHAIN=0` forces the per-head path. **`ROCKET_FA_TILE_KV`** (default off) is the opt-in online/tiled long-context variant: walk the key axis in `ROCKET_FA_TILE_KV`-wide tiles carrying the FlashAttention-2 running softmax (fp32 max/denom/output), so the working score tile is `[Tp,tile]` not the full `[Tp,n_kv]` (32 MB/head at 32K). Bit-faithful (fp32 accumulation; cos = 1.000000) but **slower** on this dispatch-bound NPU, with more and smaller per-tile submits, converging to the materialized path from below (0.58x@2K-tile -> 0.93x@16K-tile at n_kv 32K). It is a **memory escape hatch** that bounds the FA scratch at extreme context, not a speed lever; engages above `ROCKET_FA_TILE_MIN_KV` (default 8192) |
| `rocket_encoder_block_fp16(fd, T, d, n_head, d_ff, x, ln1…, Wq…Wo…, ln2…, Wf1…Wf2…, eps, out)` | **one full Whisper/transformer encoder block** (pre-norm): `x += MHA(LN1(x)); x += MLP(LN2(x))`. **Fully on the NPU**: both LayerNorms, all attention matmuls + softmax, both residual adds, the two MLP projection matmuls, AND the MLP's GELU (the 2-pass `x·Φ(x)`: Φ-gate DPU LUT + DPU EW-mul). cos = 1.000000 vs an fp64 block oracle (`tests/encoder_block_rocket.c`) |
| `rocket_siglip_encode(fd, m, pixels, out, hidden)` / `rocket_siglip_encode_ctx(c, …)` (`rocket_siglip.h`) | **the SigLIP-B/16 vision encoder end-to-end** (SmolVLM-256M front-end): patch-embed (im2col -> matmul) + position add + 12x `rocket_encoder_block_fp16` `(L=1024, d=768, 12 heads, d_ff=3072)` + post-LayerNorm. Weights mmap'd from a flat fp16 blob (`tools/siglip_extract.py`). Two paths: the simple per-call form and a **resident** form (`_ctx`: static GEMMs prepacked once + multicore). The resident path runs the 12 attention heads **across the worker fds** (`rocket_flash_attn_fp16_ctx`, unmasked MHA, since heads are independent, so one drm scheduling entity per fd dispatches head ranges across the NPU cores; head-chaining + resident scratch), **1.44-1.51x warm** vs the prior single-stream per-head loop (`ROCKET_SIGLIP_FA=0` reverts); the host GELU uses a **bit-exact fp16->fp16 LUT** (all 65536 fp16 outputs fit one 128 KB table, identical to the scalar `tanhf`), a further **1.22x** (`ROCKET_SIGLIP_GELU_SCALAR=1` reverts), for **1.78x warm combined**, cosine unchanged. **Per-layer cosine 0.999998 vs an fp32 reference** (`tests/siglip_rocket.c`) |
| `rocket_modernbert_ctx_create(m, nthreads)` / `rocket_modernbert_encode(c, B, T, len, emb, post_bias, out, hidden)` (`rocket_modernbert.h`) | **a ModernBERT text encoder end-to-end**, plus an optional stack of torch pre-norm layers after it (a decision or classification head). RoPE at a per-layer theta, global and sliding (`\|i-j\| <= window`) layers, GeGLU, bias-less LayerNorms. B right-padded sequences run each at its own length: every projection runs once over their live rows stacked into one M, attention per sequence, so padding costs nothing. The residual stream stays fp32 on the host, because ModernBERT-large's grows past 29000; each MLP down-projection's input is scaled by a power of two into fp16's range and back. Sliding layers past `2T + 2 window` tokens run **banded** (`ROCKET_MB_BAND_TILE`). `nthreads` 0 is a host mode for off-device checks. Worst layer cosine 0.9999936 against ONNX Runtime's fp32 CPU provider on the Laya decision model, post-stack output 0.99999995 (`tests/modernbert_rocket.c`, artifacts from `tools/modernbert_extract.py`) |

These are submit-bound standalone, and the host wins for an isolated norm. The value is
**compositional**: keeping the activation cube-resident between two NPU matmuls skips the
de-tile->host->re-pack round-trip.

## Vision-normalization API (`rocket_normvision.h`)

The **vision** normalization family, built on the SAME primitives as the transformer norms.
Those are the feature-axis reduce for the per-group mean and variance, and the DPU elementwise
path for the affine.

The four ops differ only in **which axis is reduced** and **how the affine broadcasts**. All
take channels-major NCHW-style buffers with `P = H*W`, the spatial count per channel, where
`P=1` is a pure `[N,C]` tensor. Each is a CTest gate (`tests/norm_vision_rocket.c`) against an
fp64 oracle.

| function | what it is |
|---|---|
| `rocket_batchnorm_fp16(fd, N, C, P, x, gamma, beta, mean, var, eps, out)` | **BatchNorm (inference)** `(x−mean[c])/sqrt(var[c]+eps)·gamma[c]+beta[c]`. **No reduction** (inference BN uses the stored running `mean`/`var` `[C]`) -> a per-channel affine folded to `x·s[c]+b[c]` (one NPU ew_mul + one ew_add over the broadcast tensors). `gamma`/`beta` `[C]` may be NULL (⇒ 1 / 0) |
| `rocket_groupnorm_fp16(fd, N, C, G, P, x, gamma, beta, eps, out)` | **GroupNorm**: normalize each `(n, group)` over its `C/G` channels x `P` spatial. A group's elements are contiguous in `[N,C,P]`, so the per-`(n,g)` reduce is a `[N·G, (C/G)·P]` **stacked** feature reduce (`[x ; x⊙x]` in one job, like LayerNorm). The affine is **per-channel** (varies within a group) -> full broadcast `x·A+B`. `C%G==0`; `gamma`/`beta` `[C]` (NULL ⇒ 1/0). `G=1` = LayerNorm-over-CHW |
| `rocket_instancenorm_fp16(fd, N, C, P, x, gamma, beta, eps, out)` | **InstanceNorm** = GroupNorm with `G=C` (normalize each `(n,c)` over its `P` spatial positions). Per-row affine |
| `rocket_l2norm_fp16(fd, M, H, x, eps, out)` | **L2-Normalize** each row over `H` (TFLite `L2_NORMALIZATION`, ONNX `LpNormalization` p=2): `x/sqrt(sum_h x²+eps)`. `sq=x⊙x` (NPU) -> `ss=sum_h sq` (NPU fp32 reduce) -> host `1/sqrt` -> per-row scale (NPU) |

All four use the RMSNorm/LayerNorm fp16-square **overflow prescale** (`|x|>~223` ⇒ x² overflows
fp16 -> exact power-of-2 prescale, recovered as `·4^k`), accumulate the reduce in fp32, and do the
O(rows) mean/var/rsqrt tail exact on the host. HW-validated bit-faithful (`max_abs` = fp16 affine
rounding) across the row-tile boundary, `C%32≠0`, `P=1`, every group count, and the large-magnitude
prescale path. Like the transformer norms they are submit-bound standalone, and the value is op
coverage (a delegate need not spill the node to CPU) + the cube-resident fusion substrate.

## Datatype matrix

The native datatype matrix is **complete**: a working, validated matmul for every
native dtype (the per-dtype alignment atoms are listed in the matmul API above).

| dtype | support | use |
|---|---|---|
| fp16 -> fp32 | native, validated | the core path; coherent Gemma-4-12B prefill |
| int8 -> int32 | native, bit-exact | W8A8 (+Hadamard) = char-identical to fp16; RAM, not speed |
| int4 -> int16 | native, bit-exact | W4A4; RAM; can reach single-pass K |
| bf16 -> fp32 | native, validated | fp32 range with no activation scaling; token-identical to fp16 |
| tf32 -> fp32 | native, validated | first 4-byte-input path; completeness rung (half-rate) |
| int16 -> int32 | native single task, **saturating**; no tiled entry | int64-exact via int8 byte-decomposition |

**int16's int32 output saturates.** `gen_matmul_int16` writes the whole int32 surface in one
task. It is bit-exact against the int64 dot product clamped to int32 (`int16_native_probe`,
`matmul_int16_rocket`). So it is exact only while the operand ranges keep each sum inside
int32. No tiled entry exposes it.

An int64-exact int16 matmul over full-range operands is the int8 byte-decomposition,
`rocket_matmul_int16_exact`: four int8 matmuls, recombined in int64. The transposed writer
(`tp_org_en`, N<=32) keeps the low 16 bits of each sum.

The two **float** rungs share the int16/fp16 fp32-out writer, with host `double` K-accum and
no saturation.

**bf16** carries fp32's 8-bit exponent at fp16's 2-byte cost, so it needs no per-row
activation scaling, and it ran Gemma-4-12B token-identical to fp16.

**tf32** is the first 4-byte-input path (feature cube C2=4, weight `(N/16,K/16,16,16)`), where
the K-group *halves* to 16 for a 4-byte element. It is a genuine 10-bit-mantissa
NVIDIA-style tf32. It tracks a tf32-rounded reference to ~1e-7, the 10-bit gap from full
fp32. tf32 is the lowest-value rung: "256×3 MAC/cycle" is half bf16's rate, and bf16 already
gives fp32 range at full speed. It is a completeness deliverable rather than a perf path.

## Tests

Each test under `tests/` is a standalone executable that links the library and runs
on the NPU. They double as the regression/bring-up checks.

The correctness gates, including the bit-exact `int8`/`int4`/`int16`/`bf16`/`tf32`
tiled-or-resident matmul paths and the int8 depthwise conv, are registered with CTest.
Run `ctest` from the build directory after `cmake --build`. Perf probes and RE sweeps are
built but left unregistered.

Every registered test holds one exit-code contract:

| Exit | Means |
|---|---|
| 0 | A numeric check ran and passed |
| 1 | A wrong answer, a declined path, a usage error, a host allocation failure |
| 2 | The device, or the part the test needs, is absent. CTest reports a skip |

A path that declines is a failure, not a skip. A test built to assert a refusal says which
stage refuses, and passes only when that stage does. Off-device the NPU tests skip, and the
host-only ones (`bytes_moved_rocket`, `chain_layout_rocket`, `chain_verify_gate`,
`matmul_plan_gate`, `regcmd_rk3576_gate`) still run and must pass. A skip is visible in the
summary, so a board run that skips a device test has not checked it.

Each NPU test is registered with `RESOURCE_LOCK npu`, so `ctest -j` never has two in flight.
Each has a `TIMEOUT` of 300 s, or 600 s for flash attention and 900 s for the ViT encoders.

A hung job fails no syscall. The `rocket` driver retires a job that runs 500 ms and signals
its fence as if it had completed. A test that scores the untouched output BO can then pass.
Two fixtures bracket the NPU tests to catch it. `npu_klog_mark` saves a kernel-journal cursor
before the first one. `npu_klog_check` then fails the run on any `NPU job timed out`,
`rk_iommu` or NPU-side `WARNING` line after it.

The journal is read directly, which the `adm` or `systemd-journal` group allows, or through
`sudo -n`. With neither, `npu_klog_check` reports a skip. Inside a test, a fenced wait past the
slow-wait mark logs a warning and counts in `rocket_fence_wait_slow_count()`. The registration
fails the test that made it.

On the RK3576 the entries' write guards also time each job from before its submit to its fence.
A job that took the driver's 125 ms backstop is redone, whatever its surface reads.
`rocket_rk3576_retired_counts()` counts those, split by whether the write check alone passed
the surface.

| test | purpose |
|---|---|
| `matmul_correctness_matrix_rocket` | **the layout/readback correctness gate**: realistic random inputs + **cosine-similarity** validation (catches silent layout/scatter/readback corruption that the exact-integer tests miss), dtype-aware (fp16/int8/int4/int16/bf16/tf32), M%4≠0 via padding, K>8192, all entry points. A declined path fails unless `EXPECT_DECLINE=pack` or `=path` names the stage that must refuse. Driver: `tests/correctness_matrix.sh`, the full M/K/N x dtype x path matrix, which asserts each documented M=1 refusal (the fp16 streaming and resident entries, resident int8 and int4) at its stage |
| `matmul_tiled_rocket` | tiled fp16 matmul + profiling/sweep knobs (the workhorse) |
| `matmul_mt_rocket` | multicore correctness + scaling |
| `matmul_prepacked_rocket` | resident-weights path vs mt vs CPU ref |
| `matmul_prepacked_crossm_rocket` | one resident weight reused bit-exact across compatible M (pack@512 -> run@256/512/768); incompatible small-M rejected |
| `matmul_prepacked_canonical_rocket` | `ROCKET_CTX_TILING_CANONICAL`: one resident weight per shape, packed once at M 60 or 512, computed at every M in 4-640, every element exact against an integer reference, over the Laya projection shapes |
| `matmul_stream_vs_prepacked_rocket` | per-call packB cost (stream vs resident) |
| `iova_ceiling_rocket` | the 32-bit IOVA window size, per-fd vs shared |
| `prototype_shared_scratch_rocket` | de-risk the shared-scratch ownership refactor |
| `matmul_int8_rocket` | int8xint8->int32 readiness (encoding sweep) |
| `matmul_int8_tiled_rocket` | tiled int8 (M/N tiles + host int64 K-accum), bit-exact |
| `matmul_int8_prepacked_rocket` | resident int8 weights, bit-exact vs one-shot |
| `int8_chain_probe` | independent int8 tasks self-chained into one kick with `ROCKET_JOB_BATCHED`, each scored against a CPU model, beside a gapped control and a chain whose first task reads zeros; prints the NPU interrupts each job raised |
| `int16_native_probe` | int16xint16->int32 in one task under two output geometries (fp16's `size_e` 3 and int8's 7), four operand fills including full-range int16, six non-square shapes, every element scored against the int64 sum clamped to int32; counts tasks the watchdog retired. A probe: its int8-geometry arms hang by design |
| `rowmajor_matmul_probe` | one fp16 -> fp32 matmul task in two forms, the cube program and the same program with six registers patched so it reads A and writes C as plain row-major buffers; 12 shapes to K 4096 and N 2048, every element scored, plus interleaved submit-to-fence medians. A probe, not a gate |
| `rowmajor_tile_probe` | the row-major form at the shapes a tiled path issues: a K slice of a wider A, an N tile into a wider C, both, int8 -> int32, fp16 out, and a two-task fp16 K-accumulation with a cube partial and a row-major final write, every element of a sentinel-filled output scored, cube-program controls. Then submit-to-fence timings at the library's tile shapes split into the read, the write and the row pitch (`--pitch`, `--pitch-asym`, `--pitch2`), with the wait on a small fence BO so the output BO's sync is not timed. A probe, not a gate |
| `rowmajor_kacc_chain_probe` | the row-major matmul form inside one chained fp16 K-accumulation kick with `DATA_REUSE`, in the library's own order: the cube program, own-width row-major tiles, and tiles written straight into C, each with reuse on and off, every element scored, then the whole kick's submit-to-fence over rotated reps. A probe, not a gate |
| `fa_replay_probe` | replays one flash-attention op from the tiles `ROCKET_FA_DUMP_OP` wrote, through `rocket_flash_attn_fp16_ctx`, many times in one context and in fresh ones, at a chosen worker count, and counts how many trials equal each of two dumped outputs and how many distinct outputs appear. With `FA_REPLAY_MARKER` naming a tracefs `trace_marker` it brackets each trial for `tools/fa_trace_join.py`, which joins the `gpu_scheduler` job events (run under `tools/fa_core_trace.sh`) to say which core ran each worker's QK and AV jobs. A probe, not a gate |
| `fa_core_matmul_probe` | one head's real AV operands (written by `tests/fa_av_dump_wrap.c`, a `-Wl,--wrap=rocket_mm_batch_run` link of the replay) from several worker fds at once, the handler's passes and pass 0 through `rocket_matmul_fp16_f32out`, each call compared with the first of its kind and with the exact sum; the core that ran a call comes from the trace. Probe, not a gate |
| `npu_core_diff_probe` | `run` writes fp16 -> fp32 tiles at 512×384×256 for three synthetic A distributions, an `int8` control through `rocket_matmul_int8`, and `perm` K permutations of real AV operands with `NPU_CORE_DIFF_OPS` (`NPU_CORE_DIFF_PERM` restricts them to inside or between aligned K blocks, `NPU_CORE_DIFF_XOR` remaps each 32-wide block's lanes), to files; `compare` diffs two runs element by element and says which side is the correctly rounded one. One core per process, chosen by `RKNPU_CORE_MASK` on the vendor provider. Probe, not a gate |
| `ppu_sub4_chain_probe` | two chained PPU pooling passes with the kernel order forced so the intermediate is 2x2 or 3x3 (MAX and AVG, C 64 and 130, a 4x4 control): the chained second pass scored per channel against the same pass over a CPU-scattered copy of the device-written intermediate and against CPU models; `--no-ppu-done` adds a reproduction arm whose jobs retire by the watchdog. A probe, not a gate |
| `dw_int8_cost_probe` | where the int8 depthwise entry's per-call time goes at MobileDet's largest depthwise shapes: the whole warm call, its host scatter and gather as written, and a blocked form of each checked byte for byte against them. A probe: it asserts only that the blocked forms agree |
| `conv_pack_cost_probe` | the fp16 conv's and the int8 direct conv's per-element host pack against the whole warm call, at EfficientDet-Lite0's depthwise shapes and MobileDet's direct shapes, with a blocked form of each pack checked byte for byte. A probe: it asserts only that the blocked forms agree |
| `cpend_wzp_probe` | `DPU_BS_OW_OP`, the TRM's CPEND operand, as an asymmetric weight zero point: the int8-out depthwise program at five weight zero points and three shapes with the operand swept (`-zw`, 0, `+zw`, the low byte), each arm scored against TFLite's accumulator plus the OUT_CVT model and against `bias' + sum xp*w + v*sum xp` for four readings of the operand; the direct int32-raw program with CPEND patched live, scored in int32 and for placement anywhere in the BO; two shapes with a partial last 64-channel group, which the program leaves unwritten. `layer` mode runs one real depthwise layer for `tflite-rocket/tools/requant_chain.py`. A probe, not a gate |
| `softmax_exp_bench` | CPU only, no NPU: the flash-attention handler's host softmax loop copied verbatim, timed on one pinned core against the same loop without `expf`, with a NEON exp pass, and fully in NEON, at rows of 512-4096, each variant's outputs scored in fp16 ulps against the verbatim loop. Prices what a vectorised exp can take out of the softmax bucket. A bench, not a gate |
| `matmul_int8_groupwise_rocket` | one-shot **group-wise** int8 vs an fp64 reference, swept over both tiling regimes: `Kt == group` (one K-tile per quant group) and `Kt < group` (several tiles sharing one group's scale) |
| `matmul_int8_prepacked_gw_rocket` | resident **group-wise** int8 (the native-quant path) vs the one-shot oracle + fp64, plus a distinct-weights-sharing-one-ctx aliasing guard, a wrong-entry-point guard, and the unaligned-call-M rejection. Registered twice: at `group=576` (bit-exact) and at a group wider than the CBUF cap (`Kt < group`) |
| `matmul_int8_crossm_gw_rocket` | resident group-wise int8 weight packed once and reused at M = 512/256/768/64/8 with no re-pack, bit-exact at every M |
| `matmul_int8_dequant_rocket` | folding the dequant/cast into the DPU `OUT_CVT`: bit-exact int8->fp32 cast + per-tensor integer scale (gated), the ratio-classifier RE harness (`ROCKET_INT8_DEQ*`), + the fp16-cast diagnostic. Proves OUT_CVT is an *integer* converter (fractional dequant can't fold) |
| `matmul_int4_rocket` | int4xint4->int16 readiness + the precision/size_e sweep |
| `matmul_int4_tiled_rocket` | tiled int4, bit-exact, single-pass-K proof |
| `matmul_int4_prepacked_rocket` | resident int4, bit-exact vs one-shot |
| `matmul_int4_prepacked_gw_rocket` | resident **group-wise** int4 (the in-model W4A4 path), bit-exact vs the one-shot `rocket_matmul_int4_groupwise` + fp64 (incl. the deep FFN K=15360 / 120 groups, N-fan across workers) |
| `matmul_int16_rocket` | int16xint16->int32 gate at the shipped geometry, plus the saturation and transposed-writer characterization modes |
| `matmul_int16_exact_rocket` | bit-exact int16->int64 via int8 byte-decomposition (the production int16 matmul) |
| `matmul_bf16_rocket` | bf16xbf16->fp32 feasibility + encoding sweep |
| `matmul_bf16_tiled_rocket` | tiled bf16 (M/N/K tiles + host fp32 K-accum), sample-verified |
| `matmul_tf32_rocket` | tf32xtf32->fp32 feasibility + 4-byte-input geometry sweep + precision characterization |
| `matmul_tf32_tiled_rocket` | tiled tf32 (M/N/K tiles + host fp32 K-accum), sample-verified (incl. K=48 %16-not-%32) |
| `matmul_dtype_perf_rocket` | the "not MAC-bound" check: fp16 vs int8 vs int4 resident throughput |
| `matmul_accum_rocket` | DPU eltwise K-accum classifier (fp16): constant-operand + sentinel |
| `matmul_accum_int8_rocket` | the int32/fp32 EW-add classifier. It varies the EW algorithm and operand size but not the precision triple, so its int32 failure does not rule out an integer EW add |
| `multicore_probe` / `multicore_threads` | scheduling probes: 1 fd serializes, N fds parallelize |
| `ctx_pool_throughput` | multi-instance context-pool throughput sweep (the "rknnpool" path): P independent contexts each running prepacked-matmul "inferences"; measures aggregate scaling vs pool depth. Spread the contexts across the big cores (`ROCKET_CPU_AFFINITY=off` + pin each ctx): up to ~3.9x at P=4 on submit-bound ops, vs ~2.1x if every 1-thread context collides on one core |
| `uapi_selftest_rocket` | rocket uAPI conformance gate: `CREATE_BO` contract (IOVA bump-starts at 0, so `dma_address==0` is valid, page-aligned, low-4 GB regcmd window), `PREP_BO` absolute-`CLOCK_MONOTONIC` deadline semantics, `FINI_BO`; the cross-kernel canary |
| `prep_signal_robust_rocket` | **`PREP_BO` wait robustness** gate: a real fp16 job driven in a loop while a `SIGUSR1` storm hits the waiting thread (handler without `SA_RESTART`): the interruptible kernel wait must not surface a spurious timeout (`EINTR` is retried to the same absolute deadline), and a `UINT64_MAX` "wait forever" timeout must saturate rather than wrap the signed deadline negative into an instant poll. Outputs byte-checked every iteration |
| `activation_lut_rocket` | DPU LUT elementwise activation: fp16 sigmoid + hardsigmoid vs fp16 CPU ref (tol 0.005); regcmd smoke off-device; also gates the on-NPU EW multiply + fully-on-NPU HardSwish/SiLU (`ROCKET_ACT_NPU_MUL`) |
| `leaky_relu_rocket` | DPU LUT **LeakyReLU**: sweep across [-R,R] + the x~0 band (the sign-based mux spike) vs fp16 ref, alpha 0.01-0.5; `scan` arg maps the raw glitch (repair off) |
| `gelu_rocket` | the **2-pass on-NPU GELU** (`x·Φ(x)`, clean unit-LUT gate) vs true erf-GELU over a WIDE `[-12,12]` (incl. the flat tails that spike the single-pass), cos=1.000000; SiLU regression check |
| `ew_mul_rocket` | fully-on-NPU elementwise binary op (identity-conv main + `EW_OP_TYPE`): low-level gen `A+B`/`A*B` bit-exact (sweepable via `ROCKET_EW_CFG`) **+ a runtime check of `rocket_ew_add_fp16`/`rocket_ew_mul_fp16`** (flat vectors, n up to 40000 across the M-tile boundary) |
| `ew_minmax_rocket` | on-NPU elementwise two-tensor **max/min** (DPU EW ALU algo MAX/MIN) + **Clip**, bit-exact vs host (n to 40000) |
| `prelu_rocket` | on-NPU **PReLU** (per-channel slope): the `max(x,α_c·x)` (α∈[0,1]) and general (α outside) paths, bit-exact vs an fp16-faithful ref |
| `softplus_mish_rocket` | DPU-LUT **Softplus** / **Mish** (YOLOv4/v7) / **Abs** vs double-precision math (sweeps + large-n tile-boundary) |
| `elu_rocket` | DPU-LUT **ELU/SELU** (symmetric shifted table + host x~0 repair) vs the math, alpha 0.5-2.0 + the SELU constants |
| `bytes_moved_rocket` | analytical DRAM-traffic model (pure, no NPU): per-phase bytes from shape+tiling+dtype+reuse, planner `njobs` cross-check, int8 readback-floor demo |
| `conv2d_fp16_rocket` | general fp16 `CONV_2D` + native depthwise, bit-exact vs an NHWC oracle (direct, OC%16 pad, DW G=32, tiling) |
| `pool_fp16_rocket` | on-NPU MaxPool / AveragePool (PPU): cube self-check + HW oracle (max bit-exact; avg <= fp16-recip tol) |
| `pool_int8_rocket` | on-NPU int8 / uint8 MaxPool / AveragePool via the fp16 PPU route: int8 & uint8 MAX bit-exact, int8 AVG ±1 ULP vs an integer golden |
| `pool_int8_native_probe` | the PPU's NATIVE int8 pooling precision, as a map over the precision pair: `PPU_DATA_FORMAT[2:0]`=0 with `PPU_RDMA_DATA_FORMAT[1:0]`=1 over a C2=16 byte cube pools MAX, MIN and AVG bit-exactly, given an integer Q16 `0x10000/k` reciprocal and the integer pad fill. Probe, not a gate — the shipping entries stay fp16-routed |
| `conv_transpose_resident` | the resident ConvTranspose against the scatter reference: bit-exact on integer fills (the host asserts every matmul partial prefix and output below 2048) at the old gate's 18 direct shapes, a cropping pad, a per-axis dilation and stride and pix2pix's and SAM's ten layers, both worker splits at four of them; `\|err\| <= 2^-9` of each element's magnitude sum on real fills with a bias; a depthwise refusal, a handle that keeps its packed weight after the caller's buffer changes, and one that refuses another context. Registered twice, the second under `ROCKET_KACC=0` |
| `ct_model_bench` | per-call time of the old entry and the resident entry at pix2pix's and SAM's ten layers, warm medians, the arm order rotated by an argument for interleaved passes. A bench, not a gate |
| `ct_route_probe` | the candidate ConvTranspose routes priced before the resident entry was built: col2im and sub-pixel as resident matmuls, sub-pixel as one forward conv, host phases split. A probe, not a gate |
| `deconv_extent_probe` | the CNA deconvolution mode with the output extent PROGRAMMED rather than derived, through `rocket_conv2d_fp16_job_extent()`: one task per cell, scored over the whole surface against `rocket_conv_transpose2d_ref_fp16`. Bit-exact on the RK3588 up to 64×64 -> 128×128. Probe, not a gate — `rocket_conv_transpose2d_fp16` does not use the mode |
| `requant_round_probe` | the OUT_CVT's tie rule at `OUT_CVT_SHIFT` bit 30 = 0 and 1 (`ROCKET_OUT_CVT_ROUND`), over ties it builds on purpose, since no round scale reaches one. It asserts the recorded rule: half to even with the bit clear and half away from zero with it set, each the one survivor of five candidate rules, on both parts (the RK3576 through `rocket_matmul_int8_rk3576`, the RK3588 through the int8 matmul's integer convert). A `ctest` and a `gates-rk3576.sh` row |
| `reduce_mean_rocket` | on-NPU spatial reductions over [H,W]: GlobalAvgPool / Mean (HW vs fp64, tolerance) **and** GlobalMax/MinPool / ReduceMax/Min (**bit-exact**, idempotent), plus a factor and schedule/cube self-check (single + multi-pass, square + equal-count rect; host fallback for non-16-smooth / unequal-count) |
| `cumsum_rocket` | on-NPU **cumsum / prefix sum** (triangular ones-matmul) vs an fp64 prefix-sum oracle, over all four variants (incl/excl x forward/reverse), M-tile boundary, N%32≠0, T=1500; **bit-exact** (max_abs 0). Independent O(N²) fp64 recompute self-check |
| `conv2d_int8_rocket` | native int8 `CONV_2D`: cube self-check + single-job HW + the **tiled-runtime DIRECT arm** (big/wide/IC<32/OC-pad) bit-exact vs an int64 oracle |
| `conv1d_rocket` | **conv1d** front-end (Whisper conv1/conv2, KW=3, stride 1/2) lowered onto a width-1 conv2d with time on the height axis, bit-exact / fp16-tolerance vs the conv oracle (incl. IC=80/512) |
| `conv_width_gate` | int8 and fp16 convs through the public entries whose tiles would pass the CNA's 11-bit width or 10-bit grains fields (2300 and 2500 columns at one or two rows, 3000 rows at one column), beside controls under the limit; every element scored, a refusal counts as a failure |
| `rk3576_conv_width_probe` | the RK3576 question: the int8 direct, depthwise and packed-image conv entries, the fp16 direct and packed-image ones and the int8 max pool, driven with extents on ladders just under and past 11, 12 and 13 bits (width to 12288, one-task height to 6144, depthwise and pool channels to 8224), 4-row planes 2048-8192 wide whose tasks read the line stride past 14 bits, kernels of 32 and 33, and strides past the 3-bit field; every element scored, reporting rc, wrong, never-written and the column, row and channel a wrap starts at. `plan` prints each arm's row plan and its first task's geometry words without submitting. Probe, not a gate |
| `rk3576_retire_probe` | the RK3576 write guard's completion half. `forced` sets the backstop to 1 us, so every job reads as retired, and asserts that each of eight small calls (the direct, depthwise, packed-image and fp16 convolutions, the int8, int32 and fp16 matmuls, the max pool) refuses with the retirement counters moved. `real` asserts the same calls return 0 with none counted, then runs the int8 matmul at M 512 and a chosen K and N (default 2240 and 1536, where every job retires at the backstop) and asserts the retirements are counted as the kernel logged them and that the call refuses or is exact, never rc 0 over a wrong surface. The chained kick is the net gate's, run under `ROCKET_RK3576_BACKSTOP_US=1`. Probe, not a gate |
| `exp_lut_rocket` | DPU-LUT **EXP** (shifted single-table, the q>=1 floor) + a **softmax-sum** end-to-end check (row-max subtracted) vs `exp` |
| `softmax_rocket` | on-NPU **row-wise softmax AND LogSoftmax** vs an fp64 oracle (M-tile boundary, Whisper T=1500, wide-spread tail); softmax rows sum to 1, logsoftmax `Σexp(out)=1` |
| `cross_entropy_rocket` | stable per-row **cross-entropy** (logsumexp reduce + host gather) vs an fp64 CE oracle (M-tile boundary, T=1500 vocab, random + argmax targets, wide spread); **fp32-grade** (max_abs <= 1.5e-4), CE >= 0; self-check `CE == −logsoftmax[target]` (independent path) |
| `layernorm_rocket` | on-NPU **LayerNorm** (stacked-row reduce + affine fold) vs fp64 (M-tile boundary, no-beta, H%32≠0, the overflow prescale) |
| `norm_vision_rocket` | on-NPU **BatchNorm / GroupNorm / InstanceNorm / L2-Normalize** vs fp64 (group counts incl. G=1 / G=C, C%32≠0, P=1, row-tile boundary, the overflow prescale) |
| `mha_rocket` | on-NPU **multi-head self-attention** vs an fp64 attention oracle (cosine sim; Whisper-base d=512/8-head; T%16≠0 key-pad path) |
| `flash_attn_rocket` | masked **grouped-query (flash) attention** vs an fp64 oracle (cosine sim; Gemma-4-12B head_dim 256 / 16 q-heads / 8-kv GQA + 1-kv MQA, sliding window, soft-cap, n_kv>T, unaligned T/n_kv); the **chained long-context** path (`ROCKET_FA_CHAIN_ELEMS` high -> a worker's whole head range in one QK+AV job, to 16K); the **online/tiled** path (`ROCKET_FA_TILE_KV` -> per-row masked-tile skip, short last tile, 16-tile 8K, grow/reuse ctx); `bench` mode A/Bs materialized vs tiled `_ctx` wall-time |
| `encoder_block_rocket` | one **full Whisper encoder block** (LN+MHA+residual+LN+MLP) vs an fp64 block oracle (cosine sim; Whisper-base; T%16≠0) |
| `siglip_rocket` | the **full SigLIP-B/16 vision encoder** vs an fp32 reference (per-layer cosine, mean 0.999998; + resident-path bench). SKIPs without the weight blob + oracle artifacts on disk |
| `modernbert_rocket` | the **resident ModernBERT encoder and its post stack** vs ONNX Runtime's fp32 taps (per-layer cosine over the live rows, a batch of 3 and single long sequences; `ROCKET_MB_BENCH=N` adds a warm median). `--host` runs the host mode. SKIPs without `-DROCKETNPU_MODERNBERT_ARTIFACTS` |
| `replay_dw_mesa` | int8 DW int8-out **regcmd** replayed on Mesa/Teflon's captured BOs, bit-exact vs `mesa-output`: our register program is Mesa's, pad word aside. Not a check of the function: the capture ran an int8 model through Mesa's uint8 driver, on a constant input |
| `conv_dw_int8_runtime` | int8 DW int8-out **runtime** vs TFLite's reference kernels on the capture's model (`tests/dw_litert_ref.py` writes the fixture; within one at a rounding boundary, at most 2% of outputs) and vs an exact host model of the requant, bit-exact at eight shapes with symmetric weights and seven with an asymmetric weight zero point (-128 to 127, a partial 64-channel group, pad 0 on an odd plane at stride 2, and a 768-channel 20×20 layer); the planner refuses along each axis it checks, and a weight zero point outside int8 refuses |
| `conv_dw_int8_perc_runtime` | the **per-channel** int8 depthwise entry at six shapes (one job, 1152 deep with a 3e4 scale spread, all-zero and near-dead channels, 480 at 20x20, banded 160x160, a partial group, three jobs): every element equal to a host model of the BS-multiplier and OUT_CVT arithmetic built from the entry's own plan, and within one of TFLite's per-channel arithmetic. Every output zero point sits inside the range, since at -128 a wrong negative-product shift passes |
| `conv2d_int8_q_rocket` | the int8-out direct conv entry at 13 shapes through the fd, ctx, 3-fd pool and fd -1 entries, every element against a host model of TFLite's accumulator and the OUT_CVT; five are 1×1s from 160×160 to 20×20 at a frontend's pointwise shapes; a `ROCKET_CONV_BATCH=1` twin |
| `conv2d_int8_q_perc_rocket` | the **per-channel** int8-out direct conv at 8 shapes (three scale-class jobs at 80×80, OC 672 and 1152, a 160×160 1×1, a padded IC under a CNA pad, the stem at stride 2, 5×5 stride 2, all-zero filters) through the fd, ctx, 3-fd pool and fd -1 entries: every element equal to the gate's own accumulator taken through the split requant at the entry's plan, and within one of TFLite's per-channel arithmetic; a `ROCKET_CONV_BATCH=1` twin |
| `conv_i8out_probe` | a probe of Mesa's direct int8-out program on the RK3588: its geometry, CPEND on it, the per-channel BS multiplier and `B` on it; `witness` adds the arms that time out (a partial 32-kernel group, the depthwise writer geometry) |
| `dw_perc_probe` | a probe of the BS stage's per-channel multiplier on the raw int8-out depthwise program: the coefficient layout, the order of add and multiply, the held-wide product, the tie rule, the sign-split shift fields, `B` as a per-channel CPEND, and the BN stage's multiplier (which hung in the one configuration tried) |
| `dump_regcmd` | diff the generated regcmd on the laptop (no HW) |
| `replay_dump` | replay a dumped in-context failing matmul |
| `matmul_fp16_rocket` | standalone fp16xfp16->fp16/fp32 single-task smoke test (the `gen_matmul_fp16` path, no tiling) |
| `dump_dw_regcmd` | host-only: emit a depthwise-conv regcmd as a u64 stream for the Mesa decoder (no HW) |
| `membench` | DRAM bandwidth + NPU readback de-tile microbench (no deps) |

The `accum` classifiers and `dtype_perf` characterize K-accum feasibility and the
dtype-independent throughput ceiling.

## Runtime knobs

The library reads a few `ROCKET_*` env vars, and the `ggml-rocket` backend exposes more.
`sudo` strips the environment, so use `sudo -E`.

| Knob | Default | What it does |
|---|---|---|
| `ROCKET_KACC` | on | fp16 NPU K-accum. `=0`, or `ROCKET_NO_KACC`, opts out to the byte-exact host fp64-accum path |
| `ROCKET_REUSE` | DATA_REUSE under KACC | CBUF reuse, 0/1/2 |
| `ROCKET_MM_MT/NT/KT` | planned | Tile overrides |
| `ROCKET_MM_ASYM` | on | Asymmetric Mt>Nt tiling. See below |
| `ROCKET_N_THREADS` | profile | Worker count |
| `ROCKET_WAIT_MS` | | Fence deadline |
| `ROCKET_SLOW_WAIT_MS` | per part: 450 RK3588, 110 RK3576 | The slow-wait mark. A fenced wait that reaches it logs a warning naming the BO and counts in `rocket_fence_wait_slow_count()`, because the `rocket` driver retires a job that runs past its watchdog (500 ms on the RK3588, a 125 ms backstop on the RK3576) and signals its fence as if it had completed |
| `ROCKET_RK3576_BACKSTOP_US` | the profile's, 125000 | What the RK3576 write guards score a job's submit-to-fence time against. A job at or past it was retired and is redone. A probe sets it to 1 to make every job read as retired |
| `ROCKET_RK3576_MM_NARROW` | on | The RK3576 int8 matmul re-plans a tile whose job retires with its surface partly written at half its width, down to 32 channels, and keeps that width per K for the process. That is what computes the K where a job stalls at a fixed output channel. `=0` makes those K refuse |
| `ROCKET_RK3576_MM_PC_SHIFT` | on | The RK3576 per-column int8 matmul (`rocket_matmul_int8_rk3576_perc` and its `_sa` and `_wbo` forms) carries each tile's gain on the DPU shift word, so its per-column multiplier spans the int16 field at any K. `=0` restores the shift-0 ramp, whose multiplier the int32 product caps in the low hundreds at a GEMM's depth. Read per call |
| `ROCKET_MM_PROFILE` | off | Bucket breakdown |
| `ROCKET_CONV_PROFILE` | off | One line at exit per RK3588 conv program (phases summed over its jobs), and for the resident ConvTranspose its GEMM and scatter-add time |
| `ROCKET_CT_SPLIT` | per shape | `m` or `n` forces how the resident ConvTranspose's workers split its GEMM, at pack time |
| `ROCKET_FA_PROFILE` | off | Splits one flash-attention head range into gather / QK / mask / softmax / AV / scatter and prints two lines at exit: the bucket sums, then the largest single range's own buckets. The sums are worker-thread intervals added over concurrent workers, so they are not a share of anything; the second line is the one to divide, because its terms were measured on one thread over one interval and the dispatch thread waited for that range. The usable quantity is a bucket's share of `max-range` applied to the caller's measured wall |
| `ROCKET_FA_CHAIN` | on | Batches a flash-attention worker's per-head QK/AV submits through a resident batched-matmul context. `=0` forces the per-head path |
| `ROCKET_FA_CHAIN_ELEMS` | 32M score elems | Bounds the chained head group, so a worker's head range stays batched up to ~20K context. Raise it to chain at deeper context, for more scratch |
| `ROCKET_FA_TILE_KV` | off | Opt-in online and tiled long-context flash attention. A memory escape hatch, slower than the materialized path on this dispatch-bound NPU |
| `ROCKET_FA_TILE_MIN_KV` | 8192 | Where `ROCKET_FA_TILE_KV` engages |
| `ROCKET_FA_FUSED_SOFTMAX` | on | The flash attention's host scale, soft-cap, mask and softmax as one fp32 pass per row with a vector exp and one fp16 rounding. `=0` restores the separate mask pass and softmax |
| `ROCKET_MB_BAND_TILE` | 128 | `rocket_modernbert`'s banded sliding attention query tile. `0` computes every sliding layer dense |
| `ROCKET_MB_ATTN` | npu | `host` runs `rocket_modernbert`'s attention in fp32 on the host. It ties the NPU at 58 tokens and loses at 639 |
| `ROCKET_MB_PROF` | off | `rocket_modernbert` prints each encode's phase split (LayerNorm, GEMM, attention, activation, down-projection) |

### Chip selection

At device open the library resolves a hardware profile from the NPU's device-tree
`compatible`: the driver-bound per-core device under `/sys/bus/platform/drivers/rocket/`,
else `/proc/device-tree/compatible`. A profile is the machine parameters, meaning CBUF banks
and bank size, tile caps, weight tile groups, the datatype menu and the worker default.

Two profiles exist. `rk3588` is the fully validated target, and every operation this library
offers runs there. `rk3576` carries machine parameters measured on the part, and only its
int8 **direct** `CONV_2D` has this chip's own geometry-register encoder. Selecting it warns
exactly which paths that leaves unserved.

That warning is load-bearing, because a profile selects machine parameters only. It does not
switch the regcmd encoder, and the CNA/CORE/DPU geometry-register **encoding** is
IP-revision-specific. An operation with no encoder for the running chip does not degrade. On
RK3576 silicon an RK3588 program submits, and the job completes with the output buffer
untouched. A Rockchip NPU with no profile at all warns and runs with RK3588 parameters.
`ROCKET_CHIP=<name>` forces a profile, and a name with no profile warns and falls back.

### CPU affinity for multi-pool processes

`ROCKET_CPU_AFFINITY` takes a CPU list like `4-7`, or `off`. It sets the per-process
big-core set the pack and readback workers pin to, auto-detected as the top-cpufreq cores
otherwise.

Several context pools can run concurrently in one process, such as one detector instance
per camera, each on its own thread. For those, `rocket_affinity_set_base(int)`
(`rocket_npu.h`) sets a per-thread rotation base. Each pool's workers then start at a distinct core,
`big[(base+worker)%n_big]`, instead of every pool stacking on `big[0]`. The convention is
`rocket_affinity_set_base(pool_index * nthreads)` before the thread's `*_ctx_create`, matmul
and conv calls.

The base is thread-local and honoured by the **fp16/int8 matmul and conv-pool** worker paths,
the pool-relevant ones a thread spawns. It is a **scheduling hint only** and never changes
numerics, and the default 0 starts every pool's workers at `big[0]`. The spread is
deterministic, and `ROCKET_DEBUG` logs each `worker N (base B) -> cpu C`.

Its throughput payoff is operating-point-dependent. At submit-bound matmul shapes an
in-process pool is **NPU-wait-bound**, because workers block on the fence. The pool then
scales ~`n_core` even with every worker collided on one core. The base is therefore a
deterministic-placement and contention-robustness lever rather than a large idle-box speedup
(`tests/ctx_pool_throughput`).

### Batched submit and chaining

`ROCKET_BATCH_SUBMIT=1` runs a tiled matmul's output tiles as **one HW kick**. The per-tile
regcmds are laid contiguously and self-chain, each task's trailer linking to the next. The NPU
streams through them and raises a single completion interrupt instead of one submit and IRQ
per tile. It cuts the dispatch floor on jobs that decompose into independent tiles, and is
bit-exact with the default per-task path. The same chaining backs
`rocket_matmul_fp16_batch` (and so `ROCKET_FA_CHAIN`), where the contiguous self-chaining
spans a batch of same-shape matmuls rather than one matmul's tiles.

Chaining is a **joint layout contract with the kernel**. Userspace self-chains the regcmds
and the kernel sets `TASK_NUMBER = task_count`, so both halves must agree. A kernel that does
not know `DRM_ROCKET_JOB_BATCHED` ignores the flag and runs a self-chained layout down the
per-task path, which stalls or corrupts the job.

`rocket_batched_submit_supported()` (`rocket_npu.h`, probed once and cached) reports whether
the running kernel honors it, and the driver gates every chaining entry point on it. Asking
for `ROCKET_BATCH_SUBMIT=1` on a kernel without the `patches/rocket` batched-submit patch
warns and runs the stock per-task path rather than producing garbage. Call it before
self-chaining anything yourself.

`ROCKET_KACC_CHAIN` (default off) extends that chaining **across the fp16 K-accumulation
ki-steps**. Instead of one fenced submit per K-tile, it chains a tile's whole nKt-step
accumulation into one self-chained kick. Each `ki>0` task EW-adds the partial that an earlier
task in the *same kick* just wrote. The NPU honors that in-kick read-after-write, so
it is **byte-exact** to the per-ki path, and the gate forces it across nKt=2…43. It is fp16
only, because integer chaining garbles as above.

It is **marginal and shape-gated**. Collapsing the fences trims submit and sync, and the
ki-steps are serially dependent, so a chained kick only pipelines the *independent* tiles
within each ki-block. That is a net win only when `gcap = 64/nKt >= 3`, about 5% at nKt~12-21,
and a wash to a loss below that (`wait` +18% at nKt=40/gcap=1).

So `=1` is **adaptive**: it engages only in that winning regime and otherwise falls back to
per-ki, never regressing. `=2` forces chaining for any fitting nKt, which is the correctness
gate's strict gcap=1 path. The common Gemma FFN-down (nKt=40) falls back, so the end-to-end
LLM gain is ~nil. The knob is a narrow lever rather than a default.
`tests/matmul_kacc_chain_rocket.c` gates it, and `matmul_kacc_chain_bench` A/Bs it.

`ROCKET_MM_ASYM` (**default on**, `=0` opts out) is a **tiling** lever rather than a submit
one. The planner caps Mt and Nt at 256 and maximizes Kt, which picks a **symmetric**
Mt=Nt=256 tile (Kt=384). For a shape that tiles both N and K, **halving Nt to 128 frees CBUF
so Kt grows to 512**, with Mt staying at 256.

The asymmetric Mt>Nt tile runs the NPU datapath measurably faster. That is **+6-9% warm** on
the square and large-K prefill matmuls (1024²/2048²×4096, Gemma FFN-down), and about a wash
on FFN-up. End-to-end warm pp2048 through llama.cpp:

| Model | Gain |
|---|---|
| Qwen3.5-9B-F16 | +9.5% |
| Gemma-4-12B-F16 | +5.7% |
| Qwen3.5-9B-Q4_K | +1.3% |

Quant prefill is dequant-bound, so the datapath win dilutes to about noise there. It is
**win-or-wash on every shape and model tested, never a regression** [HW sweep].

The win is a `wait`-term datapath effect rather than fewer fences. Submit actually rises a
little as nNt doubles, and wait drops ~10% and dominates. It fires only when N is actually
N-tiled (N>256, no `ROCKET_MM_NT` override) **and** the symmetric plan K-tiles (nKt>1). A
bigger Kt is moot at nKt=1, so small-K and small-N shapes are an exact no-op. Bit-exact,
because tiling never changes the result, and gated bit-exact under both settings by
`matmul_correctness_matrix_asym` and `matmul_correctness_matrix_sym`. Composes with KACC.

`ROCKET_CONV_BATCH=1` (default off) coalesces a tiled int8/uint8 DIRECT conv's per-tile
submits into one multi-task job (`conv2d_int8_batch_tiles`). This is the **gapped** "lever 1"
form, and it is distinct from the chaining above. The kernel runs the tiles as separate HW
kicks. The whole tile set pays one submit syscall, one fence and one IOMMU attach instead
of one per tile.

Each tile lands at a bank-aligned, zeroed slot of the batched BOs, so its feature-DMA
over-read reads zeros. It is bit-identical to the per-tile path across nt=1/2/4.

It is **single-stream-neutral**. A tiled conv's wall is the host cube scatter and de-scatter
rather than the submit floor. So it pays only under multi-process contention on
conv-tile-heavy work, where several pool contexts share the submit and IOMMU path. That is
+7.6% aggregate at P=4 on a conv-tile-heavy unit, and about 0 on conv-tile-light MobileDet.
It needs no kernel patch, being gapped rather than chained, and the matmul path already
batches its own tiles.

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
`debug`. The contract is set-callback-once-before-first-use, with no internal locking.

`ROCKET_LOG_STDERR` (set to any non-`0` value) additionally tees every emitted line to
`stderr`, but only when a host callback is installed, so the default sink never
double-prints. It is the escape hatch for a host that silences its own logger. `llama-bench`,
for one, installs a no-op `ggml` callback unless `-v` is passed. A host that adopts the
channel forwards it into `ggml`. That would otherwise swallow every rocket diagnostic,
including one-shot decisions like the resident-weight budget that change the benchmarked
number.
