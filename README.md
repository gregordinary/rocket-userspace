# rocket-userspace — `librocketnpu`

## AI Disclosure

With the exception of prior work this may build on, rocket-userspace was developed by AI, primarily Claude Code (Opus 4.8). Human involvement was mostly limited to setting project goals and providing hardware access. This is a side project for curiosity's sake and comes with no guarantee of quality, accuracy, or update frequency.

## About rocket-userspace

A userspace driver + tiled/multicore matmul and on-NPU op library for Rockchip NPUs, built on the
mainline `rocket` DRM-accel driver. Validated on the
RK3588; designed to extend to other Rockchip NPUs (RK3576, RK3566) as their machine parameters are
confirmed on-device.

It is a self-contained C library (`librocketnpu`); a thin device
shim over `/dev/accel/accel0`, a tiled/multicore/resident matmul, and an on-NPU op library on top.
It builds and is usable on its own, as the substrate for custom NPU compute; the `ggml-rocket` ggml
backend and the `tflite-rocket` delegate both link it.

The `rocket` driver is a generic register-command submitter (`CREATE_BO` / `SUBMIT` / `PREP_BO` /
`FINI_BO`) — not locked to any op set, so op coverage is a userspace matter. The library emits its
own matmul register programs (a matmul is a 1×1 convolution over the NVDLA-style CNA→CORE→DPU blocks)
and the kernel runs them: Gemma-4-12B prefill runs on the NPU through this path (decode stays on the
CPU — GEMV-bound, ~82× slower at M=1).

## What it provides

- **Matmul** — tiled, K-accumulating, fanned across the 3 NPU cores, weights resident, in fp16, int8,
  int4, bf16, and tf32 (plus bit-exact int16 via int8 byte-decomposition).
- **Convolution** — general CONV_2D with depthwise, transpose, and resize.
- **Pooling & reductions** — Max / Average pool, spatial and feature-axis reductions, cumsum.
- **Activations** — the DPU LUT family (sigmoid, tanh, SiLU, GELU, sqrt, rsqrt, reciprocal, exp,
  softplus, mish, abs, ELU, SELU, LeakyReLU, PReLU) plus elementwise add/sub/mul/div/max/min/clip.
- **Transformer / Whisper primitives** — RMSNorm, LayerNorm, softmax, gated FFN, multi-head
  self-attention, and a full encoder block — enough to run a Whisper/transformer encoder block
  end-to-end on the NPU.

The complete function reference — every entry point, per dtype and op — is in [API.md](API.md).

## Requirements

- An RK3588 board (the only validated target — see Hardware support).
- A mainline kernel carrying the `rocket` DRM-accel driver (developed against ~v7.1), with the device
  node `/dev/accel/accel0` present. Confirm with `ls /dev/accel/accel0` and `lsmod | grep rocket`
  (the module is named `rocket`).
- The `drm/rocket_accel.h` uAPI header (ships in `/usr/include/drm` on a kernel with `rocket`); CMake
  checks for it at configure time.
- libdrm and pthreads. No ML-framework dependency.
- Privilege to open the accel node — run as a user in the node's group, or with `sudo -E` (the `-E`
  preserves the `ROCKET_*` env knobs that plain `sudo` strips).

**On the RK3576, install `udev/99-rocket-npu-pm.rules`.** A job on that part whose output element is
wider than one byte — any fp16 convolution, either int32 matmul writer — leaves the next submit
completing normally and writing nothing, and what clears it is the runtime-PM autosuspend cycling the
NPU power domain. The library drives that cycle rather than waiting out the driver's 50 ms timer,
which needs write access to one sysfs file; the rule hands it to the same group that already owns
`/dev/accel/accel0`, so it grants nothing to anyone who could not already submit NPU work. Without it
the library falls back to waiting and an fp16 224x224 stem costs 226 ms instead of 13.8. The RK3588
has no such hazard and needs no rule.

**Clock.** The NPU boots at 200 MHz and the library is correct there, but every performance figure
below is at 600 MHz: apply the `patches/rocket` clock patch and load the module with
`rocket_npu_clk_hz=600000000`. The patches only raise the clock (~1.43×) and trim dispatch latency.

## Quickstart

Build the static library and run the correctness gates (green off-device — the gates that need the
NPU skip):

```sh
cmake -S . -B build && cmake --build build -j
ctest --test-dir build
```

Call it from C — open the device, run one fp16 matmul, close it:

```c
#include "rocket_npu.h"
#include "rocket_matmul.h"

int fd = rocket_open();                       /* opens /dev/accel/accel0 (needs privilege) */
/* C[M,N] = A[M,K] . B[N,K]^T, row-major fp16; requires M%4, K%32, N%16 */
rocket_matmul_fp16(fd, M, K, N, A, B, C);
rocket_close(fd);
```

Link against the `rocketnpu::rocketnpu` CMake target. The multicore, resident-weight, and
int8/int4/bf16/tf32 entry points — and the conv / activation / transformer op library — are in
[API.md](API.md).

## Hardware support

**RK3588** (rknpu v2, 3 NPU cores) is the full target: every datatype, the whole op library, and
every `tests/` gate bit-exact.

**RK3576** (2 cores) runs a growing subset through its own encoder. `CONV_2D` computes bit-exactly —
direct at int8 and fp16, and depthwise at int8 — and so does an int8 **matmul**
(`rocket_matmul_int8_rk3576()`). Three of that matmul's properties are the RK3588's inverted, and a
caller porting between the two should expect all three: int8 is the matmul precision here (one int8
task contracts up to 4608 input channels where one fp16 task contracts sixteen); the M axis carries
no constraint at all, so `M=1` is simply correct; and the output is int8 through the DPU's requant
rather than raw int32. K past one task's contraction is split through
`rocket_matmul_int8_rk3576_i32()`, which reads the DPU's raw 32-bit accumulator and sums the partials
on the host — correct at any K, and a quarter of the int8 path's MACs per submit, because that writer
delivers only the first eight output channels of every thirty-two and the way round it is to program
four times as many.

For a layer run repeatedly — a graph's forward pass, a video stream — `rocket_conv2d_int8_pack_rk3576()`
packs the weights once and `rocket_conv2d_int8_prepacked_rk3576()` runs the packed handle per
inference. Everything a per-call entry rebuilds from the weights alone (the per-output-channel filter
sums, the coefficient group, the weight cube) and every buffer whose size follows the shape (the
feature cube, the register program, each tile's output surface) is prepared once, so an inference
allocates no buffers and does no weight arithmetic. On MobileNetV1-224 that is **98.8 ms to 33.6 ms
per inference on the same 40 submits**, with the same result to the byte. The handle is bound to the
`fd` it was packed on — a buffer belongs to the file that created it — and it freezes the
output-channel tiling and the quantization contract it was packed for; it is not thread-safe, so pack
per thread or serialize.

A **residual add** is a convolution on this part, not an elementwise op: the DPU's elementwise stage
takes exactly one operand, so a skip is lowered by concatenating the two operands along channels and
convolving with a 1x1 kernel of two diagonal blocks. `rocket_residual_add_weights_rk3576()` fills that
weight matrix, its bias and its weight scale and stops — pure, no hardware — and the caller runs the
ordinary conv entries on the result. Two properties of the lowering beat a dedicated elementwise
program: the operands may carry **different scales** (the ratio rides in the weights as a pair of
int8s, and searching every denominator resolves it to about one part in 127², where anchoring the
larger term at 127 gives one part in 127), and **both zero points ride exactly**, because
`w2*(a_zp - b_zp)` is a per-output-channel constant and that is what the bias is. It also fuses: a
block's last convolution can absorb its own skip by taking `C` more input channels and an identity
block at the centre tap of its kernel, so the add costs no program at all — bounded by the weight
slice rule `ic*kh*kw <= 4608`, which a 1x1 project convolution is nowhere near and a 3x3 reaches
exactly at `C = 256`.

A tensor can stay in CUBE LAYOUT between two layers, so neither host transpose runs at the join: a
direct convolution's output surface is the next layer's feature cube byte for byte
(`rocket_conv2d_int8_cube_of_rk3576()` / `_cube_in_` / `_cube_out_`), and a run of cube-linked layers
goes out as ONE hardware kick (`rocket_chain_new_rk3576()`, with `rocket_chain_plan_rk3576()`
grouping already-linked nodes into the longest legal runs). A run's members are NODES rather than
convolution handles, because three kinds may sit in one stream: a convolution, a POOL (its own
register program, interior only), and a PLACEMENT layer — one the caller states emits no program and
leaves no host work, such as a concatenation whose operands are already slices of one buffer. Without
that third kind a layer with no program breaks every run it sits in.

**A cube may be a SLICE of a larger buffer.** Its base is a plain address on both sides of a
convolution, so a producer can be told to write its surface at a channel-group offset inside a
caller's allocation (`rocket_rk3576_cube_alloc()`, `_cube_slice()`, `_cube_declare_tail()`,
`rocket_conv2d_int8_cube_out_at_rk3576()`). That is what makes a channel CONCATENATION free:
producers writing their own slices of one buffer already ARE the concatenated tensor, so a residual
add's operands and an Inception module's four branches need no host copy, and the layer that feeds
the skip does not have to leave a row-major tensor behind for it. A slice starts every sixteen
channels, which is the width one atom interleaves; a slice is sized for what its producer WRITES
(the round-32 register channel count) and a consumer's view for what its feature DMA WALKS, which is
what `_cube_declare_tail()` states along with the constant the groups past the live channels hold.

Four whole networks run end to end, each layer bit-exact against a CPU model of the part's own
arithmetic and each returning TFLite's own top-1: **MobileNetV1-224** (29 compute layers, 4.9 ms),
**MobileNetV2-224** (64, with ten residual skips, 6.7), **ResNet-18-224** (31, 9.2) and
**Inception V1-224** (81, with nine four-operand concatenations and thirteen pooling layers, 10.8).
**Each is ONE hardware kick in ONE submit**, against 112.9 / 144.9 / 159.7 ms and tens of submits
with none of resident weights, cube layout, the placed slices and the cross-layer kick — the same
answer to the byte at every step.

The rest of the
op library still emits the RK3588 encoding, and on this part the matmul entries **refuse** rather than
submit a program the hardware will not run.

**RK3566/RK3568** are named but unprofiled: they are recognized from the device tree and warned about,
not supported.

Portability is by construction, but it is two layers and not one — the RK3576 is what settled that:

1. **Machine parameters** — CBUF banks + size, tile caps, tile-group sizes, datatype mask, worker
   default — live in one `rocket_hw_profile` (`include/rocket_hw_profile.h`, read via
   `rocket_hw_current()`), which the tiling planners consult instead of bare literals. These must be
   *measured* on the part: the RK3576's CBUF, matmul tile cap (2048, not 256) and weight N-group (32,
   not 16) all differ from the RK3588's.
2. **A regcmd encoder for the CNA/CORE/DPU geometry registers**, which is *not* a shared offset table.
   The RK3576 re-packs those blocks at the same block bases — registers move, the bit-packing differs
   at shared offsets, and it drives offsets the RK3588 leaves at reset — so the emission itself is
   per-chip (`src/npu_regcmd_rk3576.c`). What stays shared is the datapath *semantics*: the precision
   encodings, the CNA→CORE→DPU sequence, the BS/BN/EW/LUT field meanings, the block bases.

Chip selection reads the NPU's device-tree `compatible` (with a `ROCKET_CHIP` override for bring-up)
and logs which datapaths the selected chip actually has an encoder for. A part with no profile falls
back to the RK3588's and says so, rather than silently applying the wrong parameters.

## Performance

The matmul is a prefill / batched-GEMM engine. At 600 MHz its resident throughput is ~460 GOP/s
across precisions (fp16 461 / int8 386 / int4 413 GOP/s on `512×3840×4096`), and as an LLM prefill it
runs Gemma-4-12B at ~15 t/s pp2048 — ~3.2× the 8-thread CPU at M≥512 (3.6× at pp512). It is
DMA/dispatch-bound rather than MAC-bound, so quantization buys RAM and model-fit, not prefill speed.
Decode (M=1 GEMV) stays on the CPU.

**The MRDMA trap.** A matmul regcmd that configures CNA/CORE/DPU but omits the DPU-RDMA block
(`0x5xxx`) hangs — the DPU read-DMA waits forever. The matmul path emits the DPU-RDMA block and the
correct enable mask; tiling past the 12×32 KB CBUF (M/N independent, K split with host fp32
accumulation) covers all Gemma FFN shapes.

**The fp16 ladder (Gemma-4-12B pp2048).** The ~15 t/s figure is a stack of operating-point wins over
the 200 MHz boot clock:

| step | result |
|---|---|
| baseline (200 MHz, tiled multicore fp16) | 7.98 t/s |
| clock 600 MHz (`patches/rocket`) | ×1.43 → 11.40 |
| fp16 NPU K-accum (DPU eltwise-add, read each tile once) | +19% → 13.38 |
| CBUF DATA_REUSE | +7% → ~14.5 |
| resident weights (prefill-only) | +6% → ~15.1 |

(A76 affinity, NEON fp16 converts, and NEON readback de-tile trim the host buckets around these.) The
CPU baseline is ~4.7 t/s flat; the win is ~3.2–3.6× at M≥512 (3.6× at pp512, 3.2× at pp2048), ≈tie at
pp128. With the clock raised and K-accum on, the matmul is NPU-`wait`-bound (~60–68%) with packB
~22%; the remaining host work is memory/gather-bound, not instruction-bound.

**Quantization does not speed prefill at this operating point.** Resident int8/int4 tie the
~460 GOP/s floor (the NPU runs at ~15% of fp16 MAC peak), and in-model resident int8 prefill is
0.60× fp16 (its int32 readback can't be K-accumulated — the DPU eltwise operand DMA is ≤16-bit). So
int8/int4 buy RAM and model-fit, not throughput — treat it as bottleneck-conditional, not a permanent
property. The full per-dtype detail is in the [datatype matrix](API.md#datatype-matrix).

## Capabilities and limitations

Everything above is HW-validated on the RK3588 — the matmul dtypes bit-exact (int8/int4) or
fp16-tolerance, the conv and op library each a CTest gate vs an fp64/oracle reference (the full
catalog is in [API.md](API.md#tests)). The envelope:

- **Integer on-NPU K-accumulation is impossible** on this hardware. The conv accumulator reduces K
  only within one CBUF-resident tile; the only cross-tile adder (the DPU eltwise) has a ≤16-bit
  operand DMA that int32 partials can't fit. Integer K-partials accumulate on the host. (fp16 partials
  *do* fit — that is the on-NPU K-accum win.)
- **Quantization doesn't accelerate prefill** — int8/int4 are for RAM, not speed.
- **Decode (M=1 GEMV) stays on the CPU** — ~82× slower on the NPU.
- **Host layout packing is irreducible** — the NPU has no on-chip row-major→tiled conversion (the
  datapath has no transpose/RUBIK engine); you can only move/vectorize/amortize it.
- **The clock boots throttled at 200 MHz** — 600 MHz needs the `patches/rocket` clock patch, and
  900 MHz is unstable here.
- **bf16 / int16 / tf32 are completeness rungs, not speedups** — bf16 and int16 tie the ~460 GOP/s
  floor and tf32 is half-rate; only bf16 has an in-model use (fp32 range, no activation scaling).

## Build and test

No ML dependencies — just libdrm and pthreads.

```sh
cmake -S . -B build && cmake --build build -j        # -> build/librocketnpu.a
cmake -S . -B build -DBUILD_SHARED_LIBS=ON            # also emit librocketnpu.so
cmake --install build --prefix /usr/local            # export the rocketnpu::rocketnpu package
ctest --test-dir build                               # correctness gates (skip off-device)
```

The default is `-O2`; `-DROCKETNPU_OPT_FLAGS="-O3;-mcpu=native;-DNDEBUG"` overrides it, though on the
(memory/gather-bound) matmul path these are flat within ±3%. Off-device, the full build compiles on
x86 (portable `_Float16`; the NEON readback intrinsics are `__aarch64__`-gated) and `ctest` is green
with the NPU gates skipped. `drm/rocket_accel.h` must be present (CMake checks at configure time).

Each file under `tests/` is a standalone executable that links the library and runs on the NPU,
doubling as a CTest correctness gate; the full catalog is in [API.md](API.md#tests).

**Key runtime knobs** (`sudo` strips the env — use `sudo -E`):

| knob | default | effect |
|---|---|---|
| `ROCKET_KACC` | on | fp16 NPU K-accumulation (+19%); the operating mode |
| `ROCKET_REUSE` | 2 | CBUF operand reuse (DATA_REUSE under KACC, +7%) |
| `ROCKET_N_THREADS` | 5 | worker count (~one above the core count) |
| `ROCKET_CPU_AFFINITY` | auto | big-core set the pack/readback workers pin to |

The full `ROCKET_*` reference (flash-attention chaining, tiling overrides, batched submit) and the
diagnostic log channel are in [API.md](API.md#runtime-knobs).

## The rocket NPU stack

This is the foundation of an open source stack for Rockchip NPUs — three userspace projects plus a
set of optional kernel patches:

- **`rocket-userspace`** (this project) — the userspace driver, matmul, and on-NPU op library.
  Self-contained; the two frontends below link it.
- **[`ggml-rocket`](https://github.com/gregordinary/ggml-rocket)** — a ggml backend `.so`, a drop-in NPU device for stock `llama.cpp` /
  `whisper.cpp`. Links `librocketnpu`.
- **[`tflite-rocket`](https://github.com/gregordinary/tflite-rocket)** — a TFLite external delegate for detection models. Links `librocketnpu`.
- **[`patches`](https://github.com/gregordinary/patches)** (`rocket/` scope) — optional out-of-tree kernel-module patches
  (clock / voltage / IOMMU). They raise the NPU clock from its 200 MHz boot default to 600 MHz and
  trim dispatch latency; the performance figures above assume them.

## License & credits

`librocketnpu` is GPL-3.0-or-later.

It builds on prior work:

- The RK3588 NPU register interface reverse-engineered by Jasbir Matharu (`mtx512/rk3588-npu`), whose
  copyright is retained verbatim in the hardware headers (`npu_cna.h`, `npu_dpu.h`, `npu_hw.h`,
  `npu_matmul.h`).
- The `rocket` regcmd format established by the Mesa Teflon "rocket" gallium driver by Tomeu Vizoso
  (MIT). The CNA→CORE→DPU register sequence this library emits derives from that work; no Mesa source
  is vendored here.
- johanvdb/librocket, a FOSS userspace fp16 matmul on mainline `rocket` that combined the two above,
  which served as the starting point for the kernel-access layer.

This project uses that prior work as a starting point: it adopts the RK3588 register headers, rewrites
the kernel-access layer for the mainline `rocket` DRM-accel driver, and builds the tiled / multicore
/ resident / multi-dtype matmul and the on-NPU op library on top. Those additions are validated
bit-exact on real RK3588 hardware.
