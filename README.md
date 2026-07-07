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

Validated bit-exact on the RK3588 (rknpu v2 — 3 NPU cores), the only supported target today. The
regcmd approach generalizes to the wider Rockchip NPU family, and RK3576 and RK3566 are the intended
next targets — but each differs in CBUF size, core count, and possibly the datatype set, so its
values must be confirmed on-device (re-running the `tests/` bit-exact gates), not inferred.

Portability is by construction: the chip-specific machine parameters (CBUF banks + size, tile caps,
tile-group sizes, datatype mask, worker default) live in one `rocket_hw_profile`
(`include/rocket_hw_profile.h` — the `rocket_hw_rk3588` instance read via `rocket_hw_current()`),
which the tiling planners read instead of bare literals. Adding a chip is a second profile with
HW-validated values plus a `compatible`-string autodetect (and a `ROCKET_CHIP` override for
bring-up); the regcmd datapath is shared across the IP family, so only the profile changes.

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
