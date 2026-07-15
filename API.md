# librocketnpu — API reference

The complete function reference, test catalog, and runtime-knob list for `librocketnpu`. The
[README](README.md) is the guide; this is the reference. Convention throughout:
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
  rocket_encoder.h             one Whisper/transformer encoder block (LN→MHA→residual→LN→MLP→residual)
  rocket_siglip.h              SigLIP-B/16 vision encoder end-to-end (patch-embed + pos + 12×block + post-LN)
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
                               the forward conv (direct + depthwise; stride/pad/opad/dilation)
  rocket_resize.c              nearest / bilinear upsample via depthwise ConvTranspose
                               (box / separable-triangle kernel; FPN/decoder neck)
  rocket_pool.c                on-NPU MaxPool / AveragePool (PPU): single self-contained job
  rocket_reduce.c              spatial GlobalAvgPool/Mean + GlobalMax/MinPool (ReduceMax/Min: idempotent,
                               bit-exact) — telescoping multi-pass PPU reduction, kernel cap 16 — plus the
                               FEATURE-axis reduce (ones-vector matmul, fp32-accum) and CUMSUM/prefix sum
                               (the same matmul widened to a triangular ones matrix; incl/excl/reverse)
  rocket_norm.c                on-NPU RMSNorm + LayerNorm (stacked-row reduce + affine fold) + per-row scale
  rocket_normvision.c          on-NPU BatchNorm / GroupNorm / InstanceNorm / L2-Normalize (vision normalization)
  rocket_ffn.c                 on-NPU gated-MLP FFN: the GeGLU/SwiGLU core act(gate)⊙up + the projections
  rocket_softmax.c             on-NPU row-wise softmax: host row-max → NPU exp → row-sum → host 1/s → NPU scale
                               (+ LogSoftmax: host log(s) + per-row ew_sub; + stable cross-entropy: logsumexp + host gather)
  rocket_attn.c                on-NPU self-attention (QKV/score/P·V matmuls + per-head softmax); masked GQA flash attention (scale→softcap→mask→softmax→P·V)
  rocket_encoder.c             one Whisper encoder block: LN → MHA → residual → LN → MLP(GELU) → residual
  rocket_siglip_encoder.c      SigLIP-B/16 vision encoder: im2col patch-embed + pos + 12×block + post-LN (simple + resident/prepacked)
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

## Public matmul API (`rocket_matmul.h`)

| function | what it is |
|---|---|
| `rocket_matmul_fp16` | tiled single-fd fp16 matmul (the validated core) |
| `rocket_matmul_fp16_f32out` | fp16×fp16 → **fp32** output: full fp32 accumulator out (no per-K-tile/final fp16 rounding) — ~5000× more accurate than fp16-out (tracks the true result to the fp32 noise floor ~1e-7); opt-in (2× output readback, free for tall-skinny prefill, +35% only when output bytes rival the weights) |
| `rocket_matmul_fp16_mt` | multicore: split N across `nthreads` worker fds (~3 cores) |
| `rocket_matmul_fp16_batch` | run `nbatch` independent same-shape matmuls as ONE NPU job (one submit + one fence; one IRQ under `ROCKET_BATCH_SUBMIT`) — collapses a set of small, dispatch-bound GEMMs that share a shape. Bit-identical to per-item `rocket_matmul_fp16`. Backs flash attention's per-worker QK/AV submit batching. `rocket_mm_batch_create/_run/_free` is the persistent form: resident in/wt/out BOs grown only on a larger shape + a prezero-once guard that skips the full-BO zero when the `(M,K,N,nbatch)` layout repeats — for a repeated-shape caller (flash attention's per-layer QK/AV) it removes the per-call BO alloc + zero (the one-shot is a thin wrapper over it) |
| `rocket_ctx` / `rocket_weights_pack` / `rocket_matmul_fp16_prepacked` | pack weights once, reuse across calls (resident fp16). The compute M MAY differ from the pack M — the resident layout is M-independent for M≥256, so a weight packed at warmup-M is reused at prefill-M with no re-pack; an incompatible tiling returns -2 |
| `rocket_stream` / `rocket_matmul_fp16_stream[_fused]` | streaming path for LLM prefill (re-pack B, but cache scratch per shape; fuse gate/up) |
| `rocket_matmul_int8` / `rocket_matmul_plan_int8` | int8×int8→int32 tiled (pre-quantized in, raw int32 out) |
| `rocket_i8_ctx` / `rocket_matmul_int8_prepacked` | resident int8 (W8A8) weights. The compute M MAY differ from the pack M — the resident tile layout is M-independent (canonical-tileM: the tiling is planned at `MAX_TILE`), so a weight packed at warmup-M is reused at any prefill-M with no re-pack (int32 K-accum is exact for any tiling → bit-exact); an incompatible tiling returns -2. The compute M must still be a **positive multiple of 4** — an unaligned M miscomputes on the HW height geometry, so it is rejected (-1), not padded (padding M would need a matching pad of `a_scale`, which only the caller has); pad ragged row counts with `rocket_pad_m` |
| `rocket_matmul_int8_groupwise` / `rocket_matmul_plan_int8_gw` | **per-K-group int8 dequant scales**, fp32-accumulated: `C_f[m,n] = Σ_g a_scale[m,g]·b_scale[n,g]·(int32 partial of K-group g)`. The primitive a **natively quantized** weight needs — a GGUF MXFP4/Q8_0/Q4_K block carries one scale per K-block, and the NPU cannot apply a K-blocked scale on-chip (at the output stage K is fully contracted). Integer partials already leave the chip at every K-tile boundary (on-device integer K-accum is HW-dead), so the block scale is free at a boundary already being paid for: it fuses into the readback accumulate and costs **+0.6%** over per-channel int8. Unlike the int4 twin there is **no saturation bound on `group`** (the output accumulator is int32, not int16), and `Kt` need only **divide** the group, not equal it — a K-tile must lie *inside* one group, not *be* one — so a group wider than the CBUF cap stays legal. `rocket_matmul_plan_int8_gw` previews the tiling (pure, no HW) and reports the `Kt` it chose |
| `rocket_i8_weights_pack_gw` / `rocket_matmul_int8_prepacked_gw` | **resident group-wise int8**: the int8 codes are scattered into NPU BOs once and never leave, so a quantized weight costs **no per-forward-pass dequant and no per-call weight scatter** — the point of the path. Each call quantizes only A (per row, per K-group) and returns the fp32 per-group dequant. Same M-independence and the same M%4 contract as the per-channel form above. Bit-exact vs the one-shot oracle when the group fits the CBUF at the worst-case tile (which forces `Kt == group` in both paths); a wider group can land on a different `Kt` divisor, and the two then agree to fp32 reassociation |
| `rocket_matmul_int4` / `rocket_matmul_int4_ex` | int4×int4→int16 tiled (host-accum to int32); `_ex` adds the int16-saturation Kt cap (in-model `[-7,7]` needs `kt_cap=480`) |
| `rocket_matmul_int4_groupwise` | per-K-group int4 dequant scales, fp32-accumulated (the W4A4 quality lever; group = the saturation-safe K-tile) |
| `rocket_i4_ctx` / `rocket_matmul_int4_prepacked` | resident int4 (W4A4) weights, raw int32 out (per-channel) |
| `rocket_i4_weights_pack_gw` / `rocket_matmul_int4_prepacked_gw` | **resident group-wise int4** (the in-model W4A4 path): weight packed once with the K-tile forced to `group`, each call quantizes only A and returns the fp32 per-group dequant `Σ_g a_scale[m,g]·b_scale[n,g]·partial`. Hadamard (baked into the resident weight + applied to A by the caller) is product-preserving, so no driver support is needed. Like the int8 path, the resident layout is M-independent (canonical-tileM), so a weight packed at warmup-M is reused at any prefill-M (incl. a small short-prompt M) with no re-pack |
| `rocket_matmul_int16_exact` | bit-exact int16×int16→int64 via int8 byte-decomposition (4 int8 matmuls) |
| `rocket_matmul_bf16` / `rocket_matmul_plan_bf16` | bf16×bf16→fp32 tiled, single-fd (fp32 A/B in, truncated to bf16 on scatter; fp32 range, no activation scaling) |
| `rocket_matmul_bf16_mt` | multicore bf16: split N across `nthreads` worker fds over the unchanged single-fd path (~3 cores) |
| `rocket_bf16_stream` / `rocket_matmul_bf16_stream` | **streaming bf16** for LLM prefill: persistent worker fds + per-shape resident scratch BOs, re-pack A/B per call (the bf16 sibling of `rocket_matmul_fp16_stream`; bit-identical to single-fd at `nthreads=1`). The in-model bf16 path. ~3.2× single-fd warm |
| `rocket_matmul_tf32` / `rocket_matmul_plan_tf32` | tf32×tf32→fp32 tiled (raw fp32 in, HW rounds to 10-bit mantissa; the first 4-byte-input path) |

Alignment requirements differ by dtype (the native tile atoms): fp16 `K%32, N%16`;
int8 `K%32, N%32`; int4 `K%32, N%64`; int16-exact follows int8 `K%32, N%32`; bf16
`K%32, N%16` (== fp16); tf32 `K%16, N%16` (4-byte halves the K-group to 16); all
require `M%4==0`. **`M==1` (single-vector GEMV) is broken on the hardware** at every
dtype — the conv feature-height-1 geometry mis-computes — so the one-shot entry points **pad M==1→4
internally** and return row 0, while the pure planners and resident/streaming paths
reject it (pad single vectors to 4 caller-side). The plan functions are pure (no
hardware) and preview the tiling.

## Conv API (`rocket_conv.h`)

Beyond the matmul, the library runs a general 2D convolution — the basis of the
`tflite-rocket` detection delegate (a 1×1 pointwise conv is just the matmul; everything
else is the conv path). HW-validated **bit-exact**.

| function | what it is |
|---|---|
| `rocket_conv2d_fp16` | general fp16 `CONV_2D`: KxK / stride / dilation, symmetric pad, OC%16 pad, OC/OH/OW spatial tiling; allocs+frees its 5 BOs per call |
| `rocket_conv1d_fp16(fd, ic, it, oc, kw, stride, pad, …)` | **conv1d** (the Whisper encoder front-end: width-only 1D conv over time), lowered onto `rocket_conv2d_fp16` with **time on the HEIGHT axis** (IW=1 — the OH-row tiler fits CBUF; the IH=1 width layout overflows the feature banks for Whisper IC=80/512). Whisper conv1/conv2 (KW=3, pad=1, stride 1/2) HW-validated (`tests/conv1d_rocket.c`) |
| `rocket_conv2d_fp16` (`desc.depthwise=1`) | native depthwise (`DW_EN`, group G=32 + the DPU depthwise register fixes), channel + spatial tiled |
| `rocket_conv2d_act_fp16(fd, d, kind, …)` / `_ctx` | **conv→activation FUSION**: a DIRECT fp16 conv post-processes its own CACC result with a smooth `f(x)` in the SAME NPU job (the DPU LUT epilogue ported into `gen_conv2d_task`) — `out = f(conv(x))`, no 2nd round-trip. `kind` = `SILU`/`TANH`/`GELU`. HW-validated: the fused epilogue **bit-reproduces the standalone LUT** (≤0.0039), conv→tanh bit-accurate (`tests/conv_act_rocket.c`). Default-off byte-identical. HardSwish/depthwise rejected (flat-tail quirk / direct-only). *Caveat: a narrow x≈0 LE/LO boundary glitch spikes all single-pass kinds; the 2-pass `x·gate(x)` route avoids it.* |
| `rocket_conv2d_int8` | native **int8 DIRECT** `CONV_2D`: int8×int8→int32 raw accumulate (caller requants); OC pad 32 / IC pad 32, OC/OH/OW tiling. Bit-exact vs an int64 oracle. *(Also the substrate for the `tflite-rocket` delegate's native **uint8** convs — Option D: the caller recenters uint8→int8 `byte−128` and folds the centering into its requant + a box-sum; no driver change.)* |
| `rocket_conv2d_dw_int8` | native **int8 DEPTHWISE** (int8-out, on-chip requant): per-tensor quant, the Mesa zero-point bias fold; bit-exact vs Teflon ground truth |
| `rocket_conv_ctx_create` / `rocket_conv2d_{fp16,int8}_ctx` / `_dw_int8_ctx` | the same convs with a **resident** BO pool reused across calls/tiles (ctx=NULL ⇒ byte-identical legacy alloc-per-call) |
| `rocket_conv_pool_create` / `rocket_conv2d_int8_mt` / `rocket_conv_pool_free` | **multicore** int8/uint8 DIRECT conv: a pool of N worker fds (each its own resident `rocket_conv_ctx`) fans the conv's independent OC/OH/OW tiles across the 3 NPU cores. **Bit-identical** to `rocket_conv2d_int8` (same tiles/jobs); falls back to serial for single-tile convs. Mirrors `rocket_matmul_fp16_mt`'s one-fd-per-core design |
| `rocket_conv_transpose2d_fp16(fd, d, in, W, out)` / `_ctx` | **ConvTranspose2d / deconv** (learned upsampling: segmentation/decoder/super-res, FPN learned-upsample). NO hardware deconv mode — lowered to interior-dilate-input + `rot180(Wᵀ)` + a **stride-1 forward `rocket_conv2d_fp16`**, so it inherits the HW-exact conv tiling bit-for-bit. Weights `[IC][OC][KH][KW]` (in-channels first) direct or `[C][1][KH][KW]` depthwise (`desc.depthwise=1`, OC==IC, C%32). Supports stride/pad/output_padding/dilation, any OC/IC. `_plan` returns `−2` for the unimplemented `pad > d·(K−1)` crop case (clean decline). HW-validated bit-exact vs a scatter reference (`tests/conv_transpose_rocket.c`); cost scales with the *upsampled* size (sub-pixel stride² decomposition is the perf follow-on) |
| `rocket_upsample_nearest_fp16` / `rocket_upsample_bilinear_fp16` (`rocket_resize.h`) | **integer-factor resize** (the FPN/decoder neck: `RESIZE_NEAREST_NEIGHBOR` / `RESIZE_BILINEAR`). Realised as a **depthwise ConvTranspose** with a fixed box (nearest) / separable-triangle (bilinear) kernel — `pad=(k−s)/2` gives exactly `IH·s × IW·s`. Nearest = bit-exact block replication; bilinear = half-pixel 2-tap (the triangle's stride-subsample is a partition of unity), `align_corners=False` interior + zero boundary. `C%32` (depthwise group). HW-validated vs independent gather refs + partition-of-unity / linear-exactness properties (`tests/resize_rocket.c`) |

Pure planners (`rocket_conv2d_plan` / `_oh` / `_ow` / `rocket_total_pad`, plus
`rocket_conv_transpose2d_plan` / `_oh` / `_ow`) preview dims + the CBUF-fit gate without
hardware. `rocket_conv2d_ref_int8` / `rocket_conv_transpose2d_ref_fp16` are golden oracles.

Native **int8** `CONV_2D` runs end-to-end and is HW-validated: the regcmd +
cube layers (`gen_conv2d_int8` / `gen_conv2d_dw_int8`) and the runtime wrappers
(`rocket_conv2d_int8` / `rocket_conv2d_dw_int8`). **DIRECT** = int8×int8→int32 raw + host
per-axis requant (the input zero-point correction folded into the bias; in `tflite-rocket`'s
`rocket_out_nchw_to_nhwc_q_per_axis`) — real int8 accumulate, bit-identical to CPU TFLite.
**DEPTHWISE** = int8-OUT with on-chip requant (`conv_params_t.int8_out=1`: QD_EN + per-OC int32
bias in the BS ALU + `OUT_CVT` requant) — per-tensor, bit-exact vs Teflon (`tests/replay_dw_mesa.c`
for the regcmd, `tests/conv_dw_int8_runtime.c` for the runtime). The `tflite-rocket` delegate
routes signed-int8 convs to these under `--option native_int8=1`.

## Activation API (`rocket_activation.h`)

The first **on-NPU nonlinear activation** on this stack: a standalone DPU **LUT**
pass (NVDLA SDP, no conv) that applies `f(x)` to a flat fp16 vector entirely on the
NPU.

| function | what |
|---|---|
| `rocket_activation_fp16(fd, kind, in, out, n)` | elementwise fp16 activation. **`SIGMOID`** (max_abs 0.00146) + **`HARDSIGMOID`** (0.00049) run fully on the NPU LUT; **`HARDSWISH`** (0.00098), **`SILU`**, and **`GELU`** (exact erf) run the **2-pass** route `x·gate(x)` (gate = sigmoid/hardsigmoid/Φ on the NPU LUT, the multiply on host by default or **fully on the NPU** with `ROCKET_ACT_NPU_MUL=1`); **`TANH`** is single-pass. **GELU is the accurate 2-pass `x·Φ(x)`** (Φ = the Gaussian CDF on the clean unit-LUT geometry) — cos=1.000000 vs true erf-GELU over `[-12,12]` (`tests/gelu_rocket.c`); the SINGLE-pass GELU spikes ~128 in the flat tail (QUIRK 1, `ROCKET_ACT_WIDE_LUT` for RE only). All HW-validated (`tests/activation_lut_rocket.c`, `lut_tanh_rocket.c`, `gelu_rocket.c`). `n` padded to a multiple of 8 |
| `rocket_activation_fp16` — **positive-domain kinds** `SQRT` / `RSQRT` / `RECIPROCAL` | `f(x)` for `x>0` via the **shifted single-table** LUT (the whole domain maps onto the positive index half ⇒ no LE/LO glitch). The DPU LUT *does* compute the reciprocal family. Uniform-grid ⇒ accuracy is domain-bounded: **<1% over a ~100–200× range** placed away from the steep near-0 region (HW: sqrt 0.85% / rsqrt 0.44% / recip 1.0%), tune with `ROCKET_LUT_XLO/XHI`. `RSQRT` is the **RMSNorm/LayerNorm** core; `RECIPROCAL` the softmax-denominator / Div core (`tests/recip_rsqrt_rocket.c`) |
| `rocket_activation_fp16` — **`EXP`** | `exp(x)` via the same shifted single-table, default domain `[-16,0]` (the **softmax** case: input ≤0 after the row-max subtraction). Works on the STANDALONE flying path (unlike GELU — no LE/LO sign-mux glitch). exp's relative interp error is ~constant (`Δ²/8` ≈ 1e-4). A LUT table entry of exactly **q=0 mis-decodes to a garbage ~4.0** — exp's deep tail quantizes to q=0 and reads ~4, so every shifted-table entry is **floored to q≥1** (`ROCKET_LUT_QFLOOR`, default 1; a no-op for sqrt/rsqrt/recip). softmax-sum end-to-end rel ≤0.04% (`tests/exp_lut_rocket.c`) |
| `rocket_activation_fp16` — **`LOG`** | `ln(x)`, `x>0`, the natural inverse of EXP (log-probabilities / NLL / cross-entropy). Same shifted single-table, but the **first SIGNED-output kind on the positive-domain path** — `log(x)<0` for `x<1`, so `out_lo=log(x_lo)` is negative and the OUT_CVT offset decodes the signed range (the tanh/ELU machinery, now exercised here). Default domain `[0.25,32]`; uniform-grid ⇒ **absolute** error is the metric (relative is ill-defined at the `x=1` zero crossing) — HW **max_abs 0.0066 / mean 0.0007** over `[0.3,30]`, worst at the steep small-x end (`tests/recip_rsqrt_rocket.c`) |
| `rocket_leaky_relu_fp16(fd, alpha, in, out, n)` | **LeakyReLU** `x>=0?x:alpha*x` (YOLO; ONNX LeakyRelu/PRelu scalar slope), one DPU LUT pass on the natural LE/LO split (LE=`alpha*x`, LO=`x`). `alpha>0`; exact over [-R,R] (R via `ROCKET_LEAKY_R`, default 16), saturates beyond. The LUT's **sign-based LE/LO mux** spikes at exactly x≈0 → repaired on the host (the readback streams every element anyway). HW-validated alpha 0.01–0.5 (`tests/leaky_relu_rocket.c`) |
| `rocket_activation_fp16` — **`SOFTPLUS`** / **`MISH`** / **`ABS`** | **Softplus** `log(1+e^x)` (shifted single-table, `out_lo=0`, x≈0-clean; max_rel 0.14%); **Mish** `x·tanh(softplus(x))` — the **YOLOv4/v7** backbone activation — 2-pass (a `[0,1]` gate LUT + EW-mul, max_rel 0.06%); **Abs** `\|x\|` (symmetric shifted single-table, the x=0 kink on the middle sample, max_abs 1e-3). All HW-validated (`tests/softplus_mish_rocket.c`) |
| `rocket_elu_fp16(fd, alpha, in, out, n)` / `rocket_selu_fp16(fd, in, out, n)` | **ELU** `x>=0?x:alpha*(e^x-1)` and **SELU** `λ·ELU_α` (fixed self-normalizing α=1.673, λ=1.051), on the **symmetric shifted single-table**. Negative outputs ⇒ a signed OUT_CVT, which keeps the x≈0 mux spike → **host x≈0 repair** (`ROCKET_ELU_NOREPAIR` disables). Exact over [-R,R] (`ROCKET_ELU_R`, default 8). HW-validated alpha 0.5–2.0 (`tests/elu_rocket.c`) |
| `rocket_ew_add_fp16` / `rocket_ew_sub_fp16` / `rocket_ew_mul_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise **add** (residuals) / **subtract** / **multiply** (gated activations), flat fp16 vectors. All use the conv-main EW path (identity-matmul main feed + ERDMA operand, `EW_OP_TYPE` add/mul); SUB reuses the ADD datapath with the operand negated (`a-b==a+(-b)`, exact fp16 sign flip); n reshaped to [M,32] + M-tiled. **Bit-exact** vs host (`tests/ew_mul_rocket.c` runtime check sweeps add/sub/mul, n up to 40000 across the M-tile boundary) |
| `rocket_ew_div_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise **divide** `a/b` = `reciprocal(b)` (DPU LUT) then `a*recip` (EW mul). `b` positive, within the reciprocal LUT domain. fp16-LUT-approx (HW max_rel 0.35% for `b∈[0.5,8]`); covers TFLite/ONNX `DIV` (`tests/recip_rsqrt_rocket.c`) |
| `rocket_ew_max_fp16` / `rocket_ew_min_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise two-tensor **max/min** `out=max/min(a,b)`. The DPU EW **ALU algo** field (`DPU_EW_CFG` bits[17:16]) reaches **MAX(0)/MIN(1)**, not just SUM(2=add) — same conv-main EW datapath, one word changed. **Bit-exact** (they select an operand; `tests/ew_minmax_rocket.c`, n to 40000). Covers TFLite/ONNX `Maximum`/`Minimum` and `ReLU=max(x,0)` |
| `rocket_clip_fp16(fd, lo, hi, in, out, n)` | **Clip** `min(max(x,lo),hi)` on the NPU (constant-operand MAX then MIN, two EW passes). Bit-exact; covers TFLite/ONNX `Clip` and the bounded-ReLU family (ReLU6=`Clip(0,6)`) (`tests/ew_minmax_rocket.c`) |
| `rocket_prelu_fp16(fd, C, S, x, alpha, out)` | **PReLU** with a **per-channel** slope `alpha[C]` (YOLO/segmentation; ONNX PRelu), input `[C][S]`. **No LUT** (so no x≈0 glitch): for the universal `alpha∈[0,1]` it is `max(x, alpha_c·x)` (per-channel scale via row-broadcast `ew_mul`, then `ew_max`) — 2 passes; any `alpha` outside [0,1] falls back to `relu(x)+alpha_c·min(x,0)`. **Bit-exact** (`tests/prelu_rocket.c`) |
| `rocket_lut_epilogue_build(kind, lut, ep)` | builds the LE/LO tables + DPU epilogue constants for a SMOOTH single-pass kind (SiLU/tanh/GELU). The **single source** of the single-pass LUT params — shared by the standalone op and the **conv→activation fusion** (`rocket_conv2d_act_fp16`), so they can't drift |
| `rocket_ew_mul_fp16(fd, a, b, out, n)` | fully-on-NPU elementwise fp16 multiply `out=a*b` (identity-conv main feed + `EW_OP_TYPE=1`; M-tiled). Bit-exact (`tests/ew_mul_rocket.c`). The building block of on-NPU HardSwish/SiLU |
| `gen_lut_activation_fp16` (`npu_activation.h`) | low-level: the flying-mode DPU LUT regcmd (LE/LO hybrid tables, BN-mul index, OUT_CVT Q0.15→fp16). The same epilogue is fused into `gen_conv2d_task` (`lut_epilogue_t` / `npu_dpu_desc.lut_en`) for conv→activation |

**Any `n`, and the max-width quirk.** A LUT op carries the vector as a cube of `cols = n/8` width
positions, and `DPU_DATA_CUBE_WIDTH` is 13-bit, so one op caps at `n ≤ 65528`. `run_dpu_lut`
**tiles** automatically, so every activation works at any `n` (a transformer's `[M,I]` cube is
millions of elements). Riding the *exact* max width (`cols = 8191`) corrupts ~54 cube
positions, so the tile cap stays **well under** the ceiling (32768) — bit-clean. **GELU** runs the
accurate **2-pass** `x·Φ(x)` (the single-pass GELU spikes ~128 in the flat negative tail — QUIRK 1,
which also makes a *fused* single-pass matmul→GELU spike for wide FFN inputs; the 2-pass is the
on-NPU GELU route).

A **fully-on-NPU** elementwise multiply (`rocket_ew_mul_fp16`) is **implemented and
HW-validated bit-exact** (`tests/ew_mul_rocket.c`). rocket's DPU EW reads its 2nd operand
only combined with a conv/CACC **main** feed (ERDMA+MRDMA `COMB_USE(5)`, the Teflon
`add_tensor` RE), so `rocket_ew_mul_fp16` uses an **identity conv** as the main feed and
sets the EW op to multiply (`EW_OP_TYPE=1`, `DPU_EW_CFG=0x108003C4`) — the same machinery
as the fp16 K-accum eltwise-add with one register field changed (`gen_matmul_fp16`'s new
`ew_mul` flag). With it, **HardSwish/SiLU run fully on the NPU** (LUT gate + EW mul, no
host arithmetic) under `ROCKET_ACT_NPU_MUL=1`; the host multiply stays the default since a
standalone EW-mul costs a second NPU round-trip (the perf path is fusing the mul into the
producing conv). The flying-main `gen_ew_mul_fp16` (which reads 0 — no main feed)
stays behind `ROCKET_ACT_EXPERIMENTAL=1`, disabled by default.

## Transformer-block API (`rocket_reduce.h`, `rocket_norm.h`, `rocket_ffn.h`, `rocket_softmax.h`, `rocket_attn.h`, `rocket_encoder.h`)

On-NPU transformer primitives, built by composing the matmul / reduce / LUT / EW pieces —
the substrate for fused FFN / encoder blocks. Each is a CTest gate vs an fp64 oracle. Together they
**run a full Whisper/transformer encoder block on the NPU (cos = 1.000000)**.

| function | what it is |
|---|---|
| `rocket_reduce_feature_fp16(fd, M, H, in, out, mean)` | **feature-axis reduce** `sum_h x[m,h]` (or mean), per row → fp32 `[M]`. The contraction RMSNorm/LayerNorm/softmax need — and the one the PPU **cannot** give (it pools spatial `[H,W]` *within* a channel, never across). Realised as a **ones-vector matmul** reusing `rocket_matmul_fp16_f32out` (no new regcmd, genuine fp32 K-accum). Essentially **bit-exact** (max_rel ≤ 1.4e-7) |
| `rocket_cumsum_fp16(fd, M, N, in, out, exclusive, reverse)` | **cumsum / prefix sum** along the last axis (TFLite/ONNX `CumSum`). The feature reduce **widened from a single ones-COLUMN to a full triangular ones MATRIX**: `out = in·Lᵀ`, `L[n][k]=1` iff column `k` is in prefix `n` (incl/excl × forward/reverse pick the triangle). Same `rocket_matmul_fp16_f32out` reuse (no new regcmd, fp32 K-accum for long prefixes). HW **bit-exact** (max_abs 0 across all variants incl. T=1500) (`tests/cumsum_rocket.c`) |
| `rocket_rmsnorm_fp16(fd, M, H, x, weight, eps, out)` | **RMSNorm** `x/sqrt(mean_h(x²)+eps)·weight[h]`. O(M·H) on the NPU (square→reduce→scale); the O(M) per-row rsqrt tail is **exact on the host** (sending M scalars to the DPU rsqrt LUT would add a round-trip *and* hit the LUT domain). fp16-square overflow (`|x|>256`) → exact power-of-2 prescale. `weight` is the effective scale (Gemma: pass `1+w`). max_rel ≤ 3.5e-3 |
| `rocket_layernorm_fp16(fd, M, H, x, gamma, beta, eps, out)` | **LayerNorm** `(x−mean)/sqrt(var+eps)·gamma[h]+beta[h]` (the Whisper/encoder norm). **Both reductions share ONE feature-reduce job by STACKING rows**: `A=[x ; x⊙x]` (2M rows) under the ones weight → first M = `sum(x)`, next M = `sum(x²)`. Host O(M) mean/var/rsqrt; the affine folds to `x⊙A+B` (one ew_mul + one ew_add). Same fp16-square overflow prescale as RMSNorm; `beta` may be NULL (`tests/layernorm_rocket.c`) |
| `rocket_scale_rows_fp16(fd, M, N, in, r, out)` | **per-row broadcast multiply** `out[m,n]=in[m,n]·r[m]` (`r` fp32 `[M]`) — the FFN/attention/softmax post-scale (the RMSNorm 1/rms folds here; the weight folds into the next matmul). The reusable building block |
| `rocket_geglu_fp16(fd, gate, up, kind, prod, n)` | **gated activation** (GeGLU/SwiGLU core) `prod = act(gate)⊙up` — the only computation an FFN adds beyond matmul. `kind` = `SILU` (robust 2-pass) / `GELU`. Fully on the NPU (LUT + EW-mul) |
| `rocket_ffn_fp16(fd, M, H, I, x, Wg, Wu, Wd, kind, out)` | **gated-MLP FFN block** `gate=x·Wgᵀ → act(gate)⊙(x·Wuᵀ) → ·Wdᵀ`. Composes the three projections + the geglu core. **cos = 1.000000** vs fp64 (Gemma-ish 128×2048×1024). *Host handoff today — the cube-resident fusion (fewer round-trips) is the perf follow-on* |
| `rocket_softmax_fp16(fd, M, N, in, out)` | **row-wise softmax** over the last axis. Host row-max + subtract → NPU `exp` (LUT) → NPU row-sum (feature reduce) → host `1/s` → NPU per-row scale. The **row-max is on the host** (matmul/reduce can only SUM; the only on-NPU max is the PPU max-pool = the resident-fusion path). Validated to T=1500 (Whisper seq), rows sum to 1±5e-4 (`tests/softmax_rocket.c`) |
| `rocket_logsoftmax_fp16(fd, M, N, in, out)` | **row-wise LogSoftmax** `x − logsumexp(x)` (the classification / NLL-loss head, LM log-probs). Shares softmax steps 1–3 (host row-max+subtract → NPU `exp` → NPU row-sum), then a per-row **subtract** instead of a divide: host `ls=log(s)` (exact, O(M) — like softmax's `1/s`, **not** the DPU LOG LUT, which is for large-tensor log) → NPU per-row `ew_sub`. All-additive ⇒ better-conditioned than softmax (no tiny-prob blow-up); HW max_abs ≤ 0.031, `Σexp(out)=1` (`tests/softmax_rocket.c`) |
| `rocket_cross_entropy_fp16(fd, M, N, logits, target, loss)` | **stable per-row cross-entropy** `CE[m]=logsumexp(logits[m]) − logits[m][target[m]]` = `−logsoftmax[target]` (softmax-classifier / LM NLL loss). The on-NPU logsumexp reduction (host row-max → NPU `exp` → NPU fp32 row-sum → host `log(s)`) + a **HOST GATHER** of the target logit (there is **no HW gather** — M scalar lookups, like the host `1/s`). Never materializes softmax (no divide); CE ≥ 0. **fp32-grade** (max_abs ≤ 1.5e-4 — the loss never round-trips through fp16 output storage) (`tests/cross_entropy_rocket.c`) |
| `rocket_mha_self_fp16(fd, T, d, n_head, x, Wq,bq, Wk,bk, Wv,bv, Wo,bo, out)` | **multi-head self-attention** (the pure attention sublayer): QKV proj + per-head `scale·(q·kᵀ)` + softmax + `P·v` + out-proj, **all matmuls + every softmax on the NPU**. Weights row-major `[out,in]`; biases optional (Whisper: bq,bv,bo, no bk). The **key count is padded to %32 and the pad score columns masked to −30000 before softmax** (the matmul rejects unaligned N/K, and Whisper T=1500 is unaligned). cos = 1.000000 at Whisper-base d=512/8-head incl. T%16≠0 (`tests/mha_rocket.c`) |
| `rocket_flash_attn_fp16(fd, n_tokens, n_kv, head_dim, n_head, n_kv_heads, scale, softcap, Q, K, V, mask, out)` | **masked grouped-query (flash) attention** — the decoder / LLM-prefill attention sublayer (the op an llama.cpp `FLASH_ATTN_EXT` lowers to). Already-projected Q/K/V + an **additive mask the caller supplies** (causal + sliding-window — this code does not synthesise it): per head `scale·(q·kᵀ)` → optional `softcap·tanh` → `+mask` → softmax → `P·v`, **all matmuls + softmax on the NPU**. Handles **GQA/MQA** (`n_head>n_kv_heads`, kv head `h/(n_head/n_kv_heads)`); head-major dense layouts so each head is a contiguous matmul operand; `n_tokens` padded to %4, `n_kv` to %32 (pad keys scored −∞). cos = 1.000000 vs an fp64 oracle at Gemma-4-12B shapes (head_dim 256, 16 q-heads, 8 kv-heads local / 1 global, sliding window, soft-cap) (`tests/flash_attn_rocket.c`). `rocket_flash_attn_fp16_mt(…, nthreads)` fans the heads across worker fds (the cores run head ranges in parallel); `rocket_fa_ctx` + `rocket_flash_attn_fp16_ctx` is the persistent form for repeated calls (worker fds held open + per-worker score scratch kept resident), all numerically identical. `ROCKET_FA_CHAIN` (on by default) batches each worker's per-head QK matmuls into one NPU job and its AV matmuls into a second (the mask + softmax sit between, preserving the host-softmax default) — one submit + fence per head range instead of per head, through a per-worker resident batched-matmul context (`rocket_mm_batch`, BOs + score scratch held + prezeroed once). That attacks the small-GEMM dispatch floor: it nearly **doubles** the FA-op throughput vs a per-call batch (≈2.0–3.8× the per-head path — 186→50 ms @512, 363→183 ms @2K head-range time), moving the FA-NPU-vs-CPU prefill crossover from ~6K to ~2K. The chaining win does **not** decay to nothing at depth: with the head group bounded by `ROCKET_FA_CHAIN_ELEMS` (default 32M score elems = a worker's ~3-head range batched up to ~20K context, ~150–200 MB/worker scratch), collapsing the per-head submits is **1.10×@4K / 1.47×@8K / 1.32×@16K / 1.16×@32K** at a 512-token ubatch (`fabench`, T=512, [HW sweep]) — +3% end-to-end pp8192 (Qwen3.5-0.8B-F16) — with no short-context change (already batched ≤2K). At a 2048-token ubatch each head's score alone exceeds the budget and the win shrinks to ~1.05×; raise the knob to chain further at a higher scratch cost. Numerically identical (cos = 1.000000); `ROCKET_FA_CHAIN=0` forces the per-head path. **`ROCKET_FA_TILE_KV`** (default off) is the opt-in online/tiled long-context variant — walk the key axis in `ROCKET_FA_TILE_KV`-wide tiles carrying the FlashAttention-2 running softmax (fp32 max/denom/output), so the working score tile is `[Tp,tile]` not the full `[Tp,n_kv]` (32 MB/head at 32K). Bit-faithful (fp32 accumulation; cos = 1.000000) but **slower** on this dispatch-bound NPU — more, smaller per-tile submits, converging to the materialized path from below (0.58×@2K-tile → 0.93×@16K-tile at n_kv 32K) — so it is a **memory escape hatch** that bounds the FA scratch at extreme context, not a speed lever; engages above `ROCKET_FA_TILE_MIN_KV` (default 8192) |
| `rocket_encoder_block_fp16(fd, T, d, n_head, d_ff, x, ln1…, Wq…Wo…, ln2…, Wf1…Wf2…, eps, out)` | **one full Whisper/transformer encoder block** (pre-norm): `x += MHA(LN1(x)); x += MLP(LN2(x))`. **FULLY on the NPU** — both LayerNorms, all attention matmuls + softmax, both residual adds, the two MLP projection matmuls, AND the MLP's GELU (the 2-pass `x·Φ(x)`: Φ-gate DPU LUT + DPU EW-mul). cos = 1.000000 vs an fp64 block oracle (`tests/encoder_block_rocket.c`) |
| `rocket_siglip_encode(fd, m, pixels, out, hidden)` / `rocket_siglip_encode_ctx(c, …)` (`rocket_siglip.h`) | **the SigLIP-B/16 vision encoder end-to-end** (SmolVLM-256M front-end): patch-embed (im2col → matmul) + position add + 12× `rocket_encoder_block_fp16` `(L=1024, d=768, 12 heads, d_ff=3072)` + post-LayerNorm. Weights mmap'd from a flat fp16 blob (`tools/siglip_extract.py`). Two paths: the simple per-call form and a **resident** form (`_ctx`: static GEMMs prepacked once + multicore). The resident path runs the 12 attention heads **across the worker fds** (`rocket_flash_attn_fp16_ctx`, unmasked MHA — heads are independent, so one drm scheduling entity per fd dispatches head ranges across the NPU cores; head-chaining + resident scratch), **1.44–1.51× warm** vs the prior single-stream per-head loop (`ROCKET_SIGLIP_FA=0` reverts); the host GELU uses a **bit-exact fp16→fp16 LUT** (all 65536 fp16 outputs fit one 128 KB table, identical to the scalar `tanhf`), a further **1.22×** (`ROCKET_SIGLIP_GELU_SCALAR=1` reverts) — **1.78× warm combined**, cosine unchanged. **Per-layer cosine 0.999998 vs an fp32 reference** (`tests/siglip_rocket.c`) |

These standalone are submit-bound (the host wins for an isolated norm); the value is
**compositional** — keeping the activation cube-resident between two NPU matmuls skips the
de-tile→host→re-pack round-trip.

## Vision-normalization API (`rocket_normvision.h`)

The **vision** normalization family, built on the SAME primitives as the transformer norms (the
feature-axis reduce for the per-group mean/variance + the DPU elementwise path for the affine).
The four ops differ only in **which axis is reduced** and **how the affine broadcasts**; all take
channels-major NCHW-style buffers with `P = H*W` (the spatial count per channel, `P=1` for a pure
`[N,C]` tensor). Each is a CTest gate (`tests/norm_vision_rocket.c`) vs an fp64 oracle.

| function | what it is |
|---|---|
| `rocket_batchnorm_fp16(fd, N, C, P, x, gamma, beta, mean, var, eps, out)` | **BatchNorm (inference)** `(x−mean[c])/sqrt(var[c]+eps)·gamma[c]+beta[c]`. **No reduction** (inference BN uses the stored running `mean`/`var` `[C]`) → a per-channel affine folded to `x·s[c]+b[c]` (one NPU ew_mul + one ew_add over the broadcast tensors). `gamma`/`beta` `[C]` may be NULL (⇒ 1 / 0) |
| `rocket_groupnorm_fp16(fd, N, C, G, P, x, gamma, beta, eps, out)` | **GroupNorm** — normalize each `(n, group)` over its `C/G` channels × `P` spatial. A group's elements are contiguous in `[N,C,P]`, so the per-`(n,g)` reduce is a `[N·G, (C/G)·P]` **stacked** feature reduce (`[x ; x⊙x]` in one job, like LayerNorm). The affine is **per-channel** (varies within a group) → full broadcast `x·A+B`. `C%G==0`; `gamma`/`beta` `[C]` (NULL ⇒ 1/0). `G=1` = LayerNorm-over-CHW |
| `rocket_instancenorm_fp16(fd, N, C, P, x, gamma, beta, eps, out)` | **InstanceNorm** = GroupNorm with `G=C` (normalize each `(n,c)` over its `P` spatial positions). Per-row affine |
| `rocket_l2norm_fp16(fd, M, H, x, eps, out)` | **L2-Normalize** each row over `H` (TFLite `L2_NORMALIZATION`, ONNX `LpNormalization` p=2): `x/sqrt(sum_h x²+eps)`. `sq=x⊙x` (NPU) → `ss=sum_h sq` (NPU fp32 reduce) → host `1/sqrt` → per-row scale (NPU) |

All four use the RMSNorm/LayerNorm fp16-square **overflow prescale** (`|x|>~223` ⇒ x² overflows
fp16 → exact power-of-2 prescale, recovered as `·4^k`), accumulate the reduce in fp32, and do the
O(rows) mean/var/rsqrt tail exact on the host. HW-validated bit-faithful (`max_abs` = fp16 affine
rounding) across the row-tile boundary, `C%32≠0`, `P=1`, every group count, and the large-magnitude
prescale path. Like the transformer norms they are submit-bound standalone — the value is op
coverage (a delegate need not spill the node to CPU) + the cube-resident fusion substrate.

## Datatype matrix

The native datatype matrix is **complete** — a working, validated matmul for every
native dtype (the per-dtype alignment atoms are listed in the matmul API above).

| dtype | support | use |
|---|---|---|
| fp16 → fp32 | native, validated | the core path; coherent Gemma-4-12B prefill |
| int8 → int32 | native, bit-exact | W8A8 (+Hadamard) = char-identical to fp16; RAM, not speed |
| int4 → int16 | native, bit-exact | W4A4; RAM; can reach single-pass K |
| bf16 → fp32 | native, validated | fp32 range with no activation scaling; token-identical to fp16 |
| tf32 → fp32 | native, validated | first 4-byte-input path; completeness rung (half-rate) |
| int16 → int32 | **no native output** | bit-exact via int8 byte-decomposition |

**int16 is the only exception:** the int16 conv computes correctly but its DPU writer
only emits a single int32 tile (broken iteration) or a full int16-saturating
*transposed* buffer via `tp_org_en` (cracked, N≤32) — never full-iteration int32. So
full-precision int16 is done by int8 byte-decomposition (`rocket_matmul_int16_exact`).

The two **float** rungs share the int16/fp16 fp32-out writer (host `double` K-accum, no
saturation). **bf16** carries fp32's 8-bit exponent at fp16's 2-byte cost, so it needs
no per-row activation scaling — it ran Gemma-4-12B token-identical to fp16. **tf32** is
the first 4-byte-input path (feature cube C2=4, weight `(N/16,K/16,16,16)` — the K-group
*halves* to 16 for a 4-byte element) and a genuine 10-bit-mantissa NVIDIA-style tf32
(tracks a tf32-rounded reference to ~1e-7, the 10-bit gap from full fp32). tf32 is the
lowest-value rung — "256×3 MAC/cycle" is half bf16's rate, and bf16 already gives fp32
range at full speed — so it is a completeness deliverable, not a perf path.

## Tests

Each test under `tests/` is a standalone executable that links the library and runs
on the NPU. They double as the regression/bring-up checks.

The correctness gates — including the bit-exact `int8`/`int4`/`int16`/`bf16`/`tf32`
tiled-or-resident matmul paths and the int8 depthwise conv — are registered with CTest.
Run `ctest` from the build directory after `cmake --build`; each gate exits with the
CTest skip code when no NPU is present (or, for `conv_dw_int8_runtime`, when its
host-packing fixtures are absent), so the suite is green off-device and fails only on a
real on-device regression. Perf probes and RE sweeps are built but left unregistered.

| test | purpose |
|---|---|
| `matmul_correctness_matrix_rocket` | **the layout/readback correctness gate**: realistic random inputs + **cosine-similarity** validation (catches silent layout/scatter/readback corruption that the exact-integer tests miss), dtype-aware (fp16/int8/int4/int16/bf16/tf32), M%4≠0 via padding, K>8192, all entry points. Driver: `tests/correctness_matrix.sh` (the full M/K/N × dtype × path matrix; 150 PASS / 6 SKIP / 0 FAIL) |
| `matmul_tiled_rocket` | tiled fp16 matmul + profiling/sweep knobs (the workhorse) |
| `matmul_mt_rocket` | multicore correctness + scaling |
| `matmul_prepacked_rocket` | resident-weights path vs mt vs CPU ref |
| `matmul_prepacked_crossm_rocket` | one resident weight reused bit-exact across compatible M (pack@512 → run@256/512/768); incompatible small-M rejected |
| `matmul_stream_vs_prepacked_rocket` | per-call packB cost (stream vs resident) |
| `iova_ceiling_rocket` | the 32-bit IOVA window size, per-fd vs shared |
| `prototype_shared_scratch_rocket` | de-risk the shared-scratch ownership refactor |
| `matmul_int8_rocket` | int8×int8→int32 readiness (encoding sweep) |
| `matmul_int8_tiled_rocket` | tiled int8 (M/N tiles + host int64 K-accum), bit-exact |
| `matmul_int8_prepacked_rocket` | resident int8 weights, bit-exact vs one-shot |
| `matmul_int8_groupwise_rocket` | one-shot **group-wise** int8 vs an fp64 reference, swept over both tiling regimes: `Kt == group` (one K-tile per quant group) and `Kt < group` (several tiles sharing one group's scale) |
| `matmul_int8_prepacked_gw_rocket` | resident **group-wise** int8 (the native-quant path) vs the one-shot oracle + fp64, plus a distinct-weights-sharing-one-ctx aliasing guard, a wrong-entry-point guard, and the unaligned-call-M rejection. Registered twice: at `group=576` (bit-exact) and at a group wider than the CBUF cap (`Kt < group`) |
| `matmul_int8_crossm_gw_rocket` | resident group-wise int8 weight packed once and reused at M = 512/256/768/64/8 with no re-pack, bit-exact at every M |
| `matmul_int8_dequant_rocket` | folding the dequant/cast into the DPU `OUT_CVT`: bit-exact int8→fp32 cast + per-tensor integer scale (gated), the ratio-classifier RE harness (`ROCKET_INT8_DEQ*`), + the fp16-cast diagnostic. Proves OUT_CVT is an *integer* converter (fractional dequant can't fold) |
| `matmul_int4_rocket` | int4×int4→int16 readiness + the precision/size_e sweep |
| `matmul_int4_tiled_rocket` | tiled int4, bit-exact, single-pass-K proof |
| `matmul_int4_prepacked_rocket` | resident int4, bit-exact vs one-shot |
| `matmul_int4_prepacked_gw_rocket` | resident **group-wise** int4 (the in-model W4A4 path), bit-exact vs the one-shot `rocket_matmul_int4_groupwise` + fp64 (incl. the deep FFN K=15360 / 120 groups, N-fan across workers) |
| `matmul_int16_rocket` | int16×int16 readiness + the saturate/transpose characterization |
| `matmul_int16_exact_rocket` | bit-exact int16→int64 via int8 byte-decomposition (the production int16 matmul) |
| `matmul_bf16_rocket` | bf16×bf16→fp32 feasibility + encoding sweep |
| `matmul_bf16_tiled_rocket` | tiled bf16 (M/N/K tiles + host fp32 K-accum), sample-verified |
| `matmul_tf32_rocket` | tf32×tf32→fp32 feasibility + 4-byte-input geometry sweep + precision characterization |
| `matmul_tf32_tiled_rocket` | tiled tf32 (M/N/K tiles + host fp32 K-accum), sample-verified (incl. K=48 %16-not-%32) |
| `matmul_dtype_perf_rocket` | the "not MAC-bound" check: fp16 vs int8 vs int4 resident throughput |
| `matmul_accum_rocket` | DPU eltwise K-accum classifier (fp16) — constant-operand + sentinel |
| `matmul_accum_int8_rocket` | the int32/fp32 EW-add classifier establishing that integer K-accum is not possible on this hardware |
| `multicore_probe` / `multicore_threads` | scheduling probes: 1 fd serializes, N fds parallelize |
| `ctx_pool_throughput` | multi-instance context-pool throughput sweep (the "rknnpool" path): P independent contexts each running prepacked-matmul "inferences"; measures aggregate scaling vs pool depth. Spread the contexts across the big cores (`ROCKET_CPU_AFFINITY=off` + pin each ctx) — up to ~3.9× at P=4 on submit-bound ops, vs ~2.1× if every 1-thread context collides on one core |
| `uapi_selftest_rocket` | rocket uAPI conformance gate: `CREATE_BO` contract (IOVA bump-starts at 0 — `dma_address==0` is valid, page-aligned, low-4 GB regcmd window), `PREP_BO` absolute-`CLOCK_MONOTONIC` deadline semantics, `FINI_BO`; the cross-kernel canary |
| `prep_signal_robust_rocket` | **`PREP_BO` wait robustness** gate: a real fp16 job driven in a loop while a `SIGUSR1` storm hits the waiting thread (handler without `SA_RESTART`) — the interruptible kernel wait must not surface a spurious timeout (`EINTR` is retried to the same absolute deadline), and a `UINT64_MAX` "wait forever" timeout must saturate rather than wrap the signed deadline negative into an instant poll. Outputs byte-checked every iteration |
| `activation_lut_rocket` | DPU LUT elementwise activation: fp16 sigmoid + hardsigmoid vs fp16 CPU ref (tol 0.005); regcmd smoke off-device; also gates the on-NPU EW multiply + fully-on-NPU HardSwish/SiLU (`ROCKET_ACT_NPU_MUL`) |
| `leaky_relu_rocket` | DPU LUT **LeakyReLU**: sweep across [-R,R] + the x≈0 band (the sign-based mux spike) vs fp16 ref, alpha 0.01–0.5; `scan` arg maps the raw glitch (repair off) |
| `gelu_rocket` | the **2-pass on-NPU GELU** (`x·Φ(x)`, clean unit-LUT gate) vs true erf-GELU over a WIDE `[-12,12]` (incl. the flat tails that spike the single-pass), cos=1.000000; SiLU regression check |
| `ew_mul_rocket` | fully-on-NPU elementwise binary op (identity-conv main + `EW_OP_TYPE`): low-level gen `A+B`/`A*B` bit-exact (sweepable via `ROCKET_EW_CFG`) **+ a runtime check of `rocket_ew_add_fp16`/`rocket_ew_mul_fp16`** (flat vectors, n up to 40000 across the M-tile boundary) |
| `ew_minmax_rocket` | on-NPU elementwise two-tensor **max/min** (DPU EW ALU algo MAX/MIN) + **Clip**, bit-exact vs host (n to 40000) |
| `prelu_rocket` | on-NPU **PReLU** (per-channel slope): the `max(x,α_c·x)` (α∈[0,1]) and general (α outside) paths, bit-exact vs an fp16-faithful ref |
| `softplus_mish_rocket` | DPU-LUT **Softplus** / **Mish** (YOLOv4/v7) / **Abs** vs double-precision math (sweeps + large-n tile-boundary) |
| `elu_rocket` | DPU-LUT **ELU/SELU** (symmetric shifted table + host x≈0 repair) vs the math, alpha 0.5–2.0 + the SELU constants |
| `bytes_moved_rocket` | analytical DRAM-traffic model (pure, no NPU): per-phase bytes from shape+tiling+dtype+reuse, planner `njobs` cross-check, int8 readback-floor demo |
| `conv2d_fp16_rocket` | general fp16 `CONV_2D` + native depthwise, bit-exact vs an NHWC oracle (direct, OC%16 pad, DW G=32, tiling) |
| `pool_fp16_rocket` | on-NPU MaxPool / AveragePool (PPU): cube self-check + HW oracle (max bit-exact; avg ≤ fp16-recip tol) |
| `pool_int8_rocket` | on-NPU int8 / uint8 MaxPool / AveragePool via the fp16 PPU route (no native int8 PPU precision): int8 & uint8 MAX bit-exact, int8 AVG ±1 ULP vs an integer golden |
| `reduce_mean_rocket` | on-NPU spatial reductions over [H,W] — GlobalAvgPool / Mean (HW vs fp64, tolerance) **and** GlobalMax/MinPool / ReduceMax/Min (**bit-exact**, idempotent) — factor + schedule/cube self-check (single + multi-pass, square + equal-count rect; host fallback for non-16-smooth / unequal-count) |
| `cumsum_rocket` | on-NPU **cumsum / prefix sum** (triangular ones-matmul) vs an fp64 prefix-sum oracle — all four variants (incl/excl × forward/reverse), M-tile boundary, N%32≠0, T=1500; **bit-exact** (max_abs 0). Independent O(N²) fp64 recompute self-check |
| `conv2d_int8_rocket` | native int8 `CONV_2D`: cube self-check + single-job HW + the **tiled-runtime DIRECT arm** (big/wide/IC<32/OC-pad) bit-exact vs an int64 oracle |
| `conv1d_rocket` | **conv1d** front-end (Whisper conv1/conv2, KW=3, stride 1/2) lowered onto the height-1-time conv2d, bit-exact / fp16-tolerance vs the conv oracle (incl. IC=80/512) |
| `exp_lut_rocket` | DPU-LUT **EXP** (shifted single-table, the q≥1 floor) + a **softmax-sum** end-to-end check (row-max subtracted) vs `exp` |
| `softmax_rocket` | on-NPU **row-wise softmax AND LogSoftmax** vs an fp64 oracle (M-tile boundary, Whisper T=1500, wide-spread tail); softmax rows sum to 1, logsoftmax `Σexp(out)=1` |
| `cross_entropy_rocket` | stable per-row **cross-entropy** (logsumexp reduce + host gather) vs an fp64 CE oracle (M-tile boundary, T=1500 vocab, random + argmax targets, wide spread); **fp32-grade** (max_abs ≤ 1.5e-4), CE ≥ 0; self-check `CE == −logsoftmax[target]` (independent path) |
| `layernorm_rocket` | on-NPU **LayerNorm** (stacked-row reduce + affine fold) vs fp64 (M-tile boundary, no-beta, H%32≠0, the overflow prescale) |
| `norm_vision_rocket` | on-NPU **BatchNorm / GroupNorm / InstanceNorm / L2-Normalize** vs fp64 (group counts incl. G=1 / G=C, C%32≠0, P=1, row-tile boundary, the overflow prescale) |
| `mha_rocket` | on-NPU **multi-head self-attention** vs an fp64 attention oracle (cosine sim; Whisper-base d=512/8-head; T%16≠0 key-pad path) |
| `flash_attn_rocket` | masked **grouped-query (flash) attention** vs an fp64 oracle (cosine sim; Gemma-4-12B head_dim 256 / 16 q-heads / 8-kv GQA + 1-kv MQA, sliding window, soft-cap, n_kv>T, unaligned T/n_kv); the **chained long-context** path (`ROCKET_FA_CHAIN_ELEMS` high → a worker's whole head range in one QK+AV job, to 16K); the **online/tiled** path (`ROCKET_FA_TILE_KV` → per-row masked-tile skip, short last tile, 16-tile 8K, grow/reuse ctx); `bench` mode A/Bs materialized vs tiled `_ctx` wall-time |
| `encoder_block_rocket` | one **full Whisper encoder block** (LN+MHA+residual+LN+MLP) vs an fp64 block oracle (cosine sim; Whisper-base; T%16≠0) |
| `siglip_rocket` | the **full SigLIP-B/16 vision encoder** vs an fp32 reference (per-layer cosine, mean 0.999998; + resident-path bench). SKIPs without the weight blob + oracle artifacts on disk |
| `replay_dw_mesa` | int8 DW int8-out **regcmd** replayed on Mesa/Teflon's captured BOs, bit-exact vs `mesa-output` (ground truth) |
| `conv_dw_int8_runtime` | int8 DW int8-out **runtime** (host packing) vs `mesa-output`, bit-exact (raw filter+bias from the tflite model; `tests/dw_dump_capture.py` pins the domain constants) |
| `dump_regcmd` | diff the generated regcmd on the laptop (no HW) |
| `replay_dump` | replay a dumped in-context failing matmul |
| `matmul_fp16_rocket` | standalone fp16×fp16→fp16/fp32 single-task smoke test (the `gen_matmul_fp16` path, no tiling) |
| `dump_dw_regcmd` | host-only: emit a depthwise-conv regcmd as a u64 stream for the Mesa decoder (no HW) |
| `membench` | DRAM bandwidth + NPU readback de-tile microbench (no deps) |

The `accum` classifiers and `dtype_perf` characterize K-accum feasibility and the
dtype-independent throughput ceiling.

## Runtime knobs

The library reads a few `ROCKET_*` env vars (the `ggml-rocket` backend exposes more):
`ROCKET_KACC` (fp16 NPU K-accum — **default on**; `=0` or `ROCKET_NO_KACC` opts out to
the byte-exact host fp64-accum path), `ROCKET_REUSE` (0/1/2 CBUF reuse — defaults to
DATA_REUSE under KACC), `ROCKET_MM_MT/NT/KT` (tile overrides), `ROCKET_MM_ASYM` (asymmetric
Mt>Nt tiling — see below), `ROCKET_N_THREADS`
(worker count), `ROCKET_WAIT_MS` (fence deadline), `ROCKET_MM_PROFILE=1` (bucket
breakdown), `ROCKET_FA_CHAIN` (default on — batch a flash-attention worker's per-head QK/AV
submits through a resident batched-matmul context; `=0` forces the per-head path,
`ROCKET_FA_CHAIN_ELEMS` — default **32M** score elems — bounds the chained head group, so a
worker's head range stays batched up to ~20K context for the long-context win; raise it to
chain at deeper context for more scratch), `ROCKET_FA_TILE_KV` (default off — opt-in
online/tiled long-context flash attention; a memory escape hatch, slower than the materialized
path on this dispatch-bound NPU, engages above `ROCKET_FA_TILE_MIN_KV`, default 8192). Note:
`sudo` strips env — use `sudo -E`.

**CPU-affinity for multi-pool processes.** `ROCKET_CPU_AFFINITY` (a CPU list like `4-7`, or
`off`) sets the per-process big-core SET the pack/readback workers pin to (auto-detected as the
top-cpufreq cores otherwise). For running **several context pools concurrently in one process**
(e.g. one detector instance per camera, each on its own thread), `rocket_affinity_set_base(int)`
(`rocket_npu.h`) sets a per-thread rotation BASE so each pool's workers start at a distinct core
(`big[(base+worker)%n_big]`) instead of every pool stacking on `big[0]` — convention
`rocket_affinity_set_base(pool_index * nthreads)` before the thread's `*_ctx_create`/matmul/conv
calls. The base is thread-local and honoured by the **fp16/int8 matmul and conv-pool** worker
paths (the pool-relevant ones) a thread spawns; it is a **scheduling hint only** (never changes
numerics; default 0 = every pool's workers start at `big[0]`). The spread is deterministic (`ROCKET_DEBUG` logs each `worker N (base B) -> cpu C`);
its throughput payoff is operating-point-dependent — at submit-bound matmul shapes an in-process
pool is **NPU-wait-bound** (workers block on the fence, so the pool scales ~`n_core` even with
every worker collided on one core), so the base is a deterministic-placement + contention-
robustness lever rather than a large idle-box speedup (`tests/ctx_pool_throughput`).

`ROCKET_BATCH_SUBMIT=1` runs a tiled matmul's output tiles as **one HW kick** — the
per-tile regcmds are laid contiguously and self-chain (each task's trailer links to the
next), so the NPU streams through them and raises a single completion interrupt instead
of one submit + IRQ per tile. It cuts the dispatch floor on jobs that decompose into
independent tiles; bit-exact with the default per-task path. The same chaining backs
`rocket_matmul_fp16_batch` (and so `ROCKET_FA_CHAIN`), where the contiguous self-chaining
spans a batch of same-shape matmuls rather than one matmul's tiles.

Chaining is a **joint layout contract with the kernel** — userspace self-chains the
regcmds, the kernel sets `TASK_NUMBER = task_count` — so both halves must agree. A kernel
that does not know `DRM_ROCKET_JOB_BATCHED` ignores the flag and runs a self-chained
layout down the per-task path, which stalls or corrupts the job. `rocket_batched_submit_supported()`
(`rocket_npu.h`; probed once, cached) reports whether the running kernel honors it, and
the driver gates every chaining entry point on it: asking for `ROCKET_BATCH_SUBMIT=1` on a
kernel without the `patches/rocket` batched-submit patch warns and runs the stock per-task
path rather than producing garbage. Call it before self-chaining anything yourself.

`ROCKET_KACC_CHAIN` (default off) extends that chaining **across the fp16 K-accumulation
ki-steps**: instead of one fenced submit per K-tile, it chains a tile's whole nKt-step
accumulation into one self-chained kick, where each `ki>0` task EW-adds the partial an earlier
task in the *same kick* just wrote. The NPU honors that in-kick read-after-write, so it is
**byte-exact** to the per-ki path (the gate forces it across nKt=2…43); fp16 only (integer
chaining garbles, as above). It is **marginal and shape-gated**: collapsing the fences trims
submit/sync, but the ki-steps are serially dependent, so a chained kick only pipelines the
*independent* tiles within each ki-block — net win only when `gcap = 64/nKt ≥ 3` (~5% at
nKt≈12–21), a wash-to-loss below that (`wait` +18% at nKt=40/gcap=1). So `=1` is **adaptive**:
it engages only in that winning regime and otherwise falls back to per-ki (never regresses);
`=2` forces chaining for any fitting nKt (the correctness gate's strict gcap=1 path). The
common Gemma FFN-down (nKt=40) falls back, so the end-to-end LLM gain is ~nil — the knob is a
narrow lever, not a default. `tests/matmul_kacc_chain_rocket.c` gates it; `matmul_kacc_chain_bench`
A/Bs it.

`ROCKET_MM_ASYM` (**default on**; `=0` opts out) is a **tiling** lever, not a submit one. The planner
caps Mt and Nt at 256 and maximizes Kt, which picks a **symmetric** Mt=Nt=256 tile (Kt=384). For a
shape that tiles both N and K, **halving Nt to 128 (Mt stays 256) frees CBUF so Kt grows to 512**, and
the asymmetric Mt>Nt tile runs the NPU datapath measurably faster: **+6–9% warm** on the square /
large-K prefill matmuls (1024²/2048²×4096, Gemma FFN-down), ~wash on FFN-up. End-to-end warm pp2048
through llama.cpp: **Qwen3.5-9B-F16 +9.5%, Gemma-4-12B-F16 +5.7%, Qwen3.5-9B-Q4_K +1.3%** (quant
prefill is dequant-bound so the datapath win dilutes to ~noise) — **win-or-wash on every shape/model
tested, never a regression** [HW sweep]. The win is a `wait`-term (datapath) effect, not fewer fences
(submit actually rises a little as nNt doubles; wait drops ~10% and dominates). It fires only when N is
actually N-tiled (N>256, no `ROCKET_MM_NT` override) **and** the symmetric plan K-tiles (nKt>1) — a
bigger Kt is moot at nKt=1, so small-K / small-N shapes are an exact no-op. Bit-exact (tiling never
changes the result; gated bit-exact under both settings by `matmul_correctness_matrix_asym` /
`matmul_correctness_matrix_sym`). Composes with KACC.

`ROCKET_CONV_BATCH=1` (default off) coalesces a tiled int8/uint8 DIRECT conv's per-tile submits
into ONE multi-task job (`conv2d_int8_batch_tiles`) — the **gapped** "lever 1" form, distinct from
the chaining above: the kernel runs the tiles as separate HW kicks, so the int32 CACC clears per
kick and **int8 stays bit-exact** (chaining is fp16-only), while the whole tile set pays one submit
syscall + one fence + one IOMMU attach instead of one per tile. Each tile lands at a bank-aligned,
zeroed slot of the batched BOs so its feature-DMA over-read reads zeros; bit-identical to the
per-tile path across nt=1/2/4. It is **single-stream-neutral** — a
tiled conv's wall is the host cube scatter/descatter, not the submit floor — and pays only under
multi-process contention on conv-tile-heavy work (several pool contexts sharing the submit/IOMMU
path; +7.6% aggregate at P=4 on a conv-tile-heavy unit, ~0 on conv-tile-light MobileDet). Needs no
kernel patch (gapped, not chained); the matmul path already batches its own tiles.

## Logging (`rocket_log.h`)

Every diagnostic the library emits — errors, warnings, the `ROCKET_MM_PROFILE` breakdown,
the `ROCKET_DEBUG` traces — flows through one channel so a host application can intercept,
redirect, or silence it instead of having raw `stderr` writes pollute its own output.

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

The threshold is read once from the environment on first use: `ROCKET_LOG_LEVEL` =
`error`|`warn`|`info`|`debug` (or `0`..`3`); `ROCKET_DEBUG` raises it
to at least `debug`. The contract is set-callback-once-before-first-use (no internal locking).

`ROCKET_LOG_STDERR` (set to any non-`0` value) additionally tees every emitted line to
`stderr` — but only when a host callback is installed, so the default sink never
double-prints. It is the escape hatch for a host that silences its own logger: `llama-bench`,
for one, installs a no-op `ggml` callback unless `-v` is passed, which (because a host that
adopts the channel forwards it into `ggml`) would otherwise swallow every rocket diagnostic —
including one-shot decisions like the resident-weight budget that change the benchmarked number.
