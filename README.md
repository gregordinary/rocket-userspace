# rocket-userspace: `librocketnpu`

## AI disclosure

Except for the prior work it builds on, rocket-userspace was developed by AI, primarily
Claude Code (Opus 4.8). Human involvement was mostly limited to setting project goals and
providing hardware access. This is a side project for curiosity's sake, and it comes with no
guarantee of quality, accuracy, or update frequency.

## About rocket-userspace

A userspace driver, tiled/multicore matmul and on-NPU op library for Rockchip NPUs, built on
the mainline `rocket` DRM-accel driver. It is validated on the RK3588 and runs a growing
subset on the RK3576. Other Rockchip NPUs follow as their machine parameters are confirmed
on-device.

`librocketnpu` is a self-contained C library in three layers: a thin device shim over
`/dev/accel/accel0`, a tiled/multicore/resident matmul, and an on-NPU op library. It builds
and runs on its own, as the substrate for custom NPU compute. The `ggml-rocket` ggml backend
and the `tflite-rocket` delegate both link it.

The `rocket` driver is a generic register-command submitter (`CREATE_BO` / `SUBMIT` /
`PREP_BO` / `FINI_BO`). It runs whatever register program it is handed, so op coverage is a
userspace matter. The library emits its own matmul register programs and the kernel runs
them, where a matmul is a 1×1 convolution over the NVDLA-style CNA→CORE→DPU blocks.
Gemma-4-12B prefill runs on the NPU through this path. Decode stays on the CPU, because it
is GEMV-bound and ~82x slower at M=1.

## Operations

- **Matmul**: tiled, K-accumulating, fanned across the 3 NPU cores, weights resident. Runs
  in fp16, int8, int4, bf16 and tf32, plus bit-exact int16 via int8 byte-decomposition.
- **Convolution**: general `CONV_2D` with depthwise, transpose and resize.
- **Pooling and reductions**: Max and Average pool, spatial and feature-axis reductions,
  cumsum.
- **Activations**: the DPU LUT family (sigmoid, tanh, SiLU, GELU, sqrt, rsqrt, reciprocal,
  exp, softplus, mish, abs, ELU, SELU, LeakyReLU, PReLU), plus elementwise
  add/sub/mul/div/max/min/clip.
- **Transformer and Whisper primitives**: RMSNorm, LayerNorm, softmax, gated FFN,
  multi-head self-attention, and a full encoder block. That is enough to run a
  Whisper/transformer encoder block end to end on the NPU.

The complete function reference, every entry point per dtype and op, is in [API.md](API.md).

## Terms

The register-level vocabulary the rest of this document uses. The first three count
different things, and conflating them is the common mistake:

| Term | What it means here |
|---|---|
| Submit | One `ROCKET_SUBMIT` ioctl. It costs one syscall, one fence and one IOMMU attach. |
| Task | One register program inside a submit. A submit carries one task or many. |
| Kick | One hardware start of the datapath, and one completion interrupt. On-chip state lives inside a kick and clears between kicks. |
| regcmd | The register program itself: the CNA, CORE and DPU register writes the kernel replays. |
| CNA, CORE, DPU | The three datapath blocks a program drives in sequence. Feature and weight fetch, the MAC array, then post-processing and write-out. |
| CBUF | The on-chip buffer the CNA fills. Tile sizes are chosen to fit it. |
| Cube layout | The NPU's native tensor format. The host scatters row-major data into it and de-scatters the result. |
| Resident weights | Weights left in device memory across calls, so a repeated layer pays the packing once. |
| Prefill, decode | The two LLM phases. Prefill is a batched GEMM and runs here, and decode is `M=1` and stays on the CPU. |
| The submit seam | The `rocket_*` symbols every kernel interaction goes through, and the swap point for a non-mainline driver. |

Submits, tasks and kicks vary independently, which is what makes them worth separating:

| Path | Submits | Tasks | Kicks |
|---|---:|---:|---:|
| Default, one job per tile | n | n | n |
| `ROCKET_BATCH_SUBMIT=1`, self-chained, fp16 | 1 | n | 1 |
| `ROCKET_CONV_BATCH=1`, gapped, int8 | 1 | n | n |

The last two rows cost the same syscalls and run the same programs. They differ only in
kicks, and that difference decides the arithmetic. The int32 accumulator clears between
kicks, so the gapped form keeps int8 bit-exact. The self-chained form instead lets a task
read what an earlier task in the same kick just wrote.

## Requirements

- An RK3588 board. It is the only fully validated target, and
  [Hardware support](#hardware-support) has the rest.
- A mainline kernel carrying the `rocket` DRM-accel driver, developed against ~v7.1, with
  the device node `/dev/accel/accel0` present. Confirm with `ls /dev/accel/accel0` and
  `lsmod | grep rocket`, where the module is named `rocket`.
- The `drm/rocket_accel.h` uAPI header, which ships in `/usr/include/drm` on a kernel with
  `rocket`. CMake checks for it at configure time. Only the built-in submit provider needs
  it, because a BSP-kernel board has no mainline `rocket` uAPI and supplies its own provider
  instead. See [Submit provider](#submit-provider).
- libdrm and pthreads. No ML-framework dependency.
- Privilege to open the accel node. Run as a user in the node's group, or with `sudo -E`,
  where the `-E` preserves the `ROCKET_*` env knobs that plain `sudo` strips.

### The RK3576 power-management rule

Install `udev/99-rocket-npu-pm.rules` on the RK3576.

A submit on that part whose output element is wider than one byte leaves the next one
completing normally and writing nothing. Any fp16 convolution and either int32 matmul writer
does it. What clears it is the runtime-PM autosuspend cycling the NPU power domain.

The library drives that cycle rather than waiting out the driver's 50 ms timer, which needs
write access to one sysfs file. The rule hands that file to the group that already owns
`/dev/accel/accel0`. It therefore grants nothing to anyone who could not already submit NPU
work. Without the rule the library falls back to waiting, and an fp16 224×224 stem costs
226 ms instead of 13.8 ms. The RK3588 has no such hazard and needs no rule.

### Clock

The NPU boots at 200 MHz and the library is correct there. Every performance figure below is
at 600 MHz. Apply the `patches/rocket` clock patch and load the module with
`rocket_npu_clk_hz=600000000`. The patches only raise the clock (~1.43x) and trim dispatch
latency.

## Quickstart

Build the static library and run the correctness gates. They are green off-device, because
the gates that need the NPU skip:

```sh
cmake -S . -B build && cmake --build build -j
ctest --test-dir build
```

Call it from C. Open the device, run one fp16 matmul, close it:

```c
#include "rocket_npu.h"
#include "rocket_matmul.h"

int fd = rocket_open();                       /* opens /dev/accel/accel0 (needs privilege) */
/* C[M,N] = A[M,K] . B[N,K]^T, row-major fp16; requires M%4, K%32, N%16 */
rocket_matmul_fp16(fd, M, K, N, A, B, C);
rocket_close(fd);
```

Link against the `rocketnpu::rocketnpu` CMake target. [API.md](API.md) has the multicore,
resident-weight and int8/int4/bf16/tf32 entry points, and the conv, activation and
transformer op library.

## Hardware support

### RK3588

The RK3588 (rknpu v2, 3 NPU cores) is the full target: every datatype, the whole op library,
and every `tests/` gate bit-exact.

### RK3576

The RK3576 (2 cores) runs a growing subset through its own encoder. `CONV_2D` computes
bit-exactly, direct at int8 and fp16 and depthwise at int8. So does an int8 matmul,
`rocket_matmul_int8_rk3576()`.

Three of that matmul's properties are the RK3588's inverted, and a caller porting between
the two must expect all three:

- int8 is the matmul precision here. One int8 task contracts up to 4608 input channels,
  where one fp16 task contracts sixteen.
- The M axis carries no constraint at all, so `M=1` is simply correct.
- The output is int8 through the DPU's requant rather than raw int32.

K past one task's contraction is split through `rocket_matmul_int8_rk3576_i32()`, which
reads the DPU's raw 32-bit accumulator and sums the partials on the host. It is correct at
any K, and costs a quarter of the int8 path's MACs per submit. That writer delivers only the
first eight output channels of every thirty-two, and the way round it is to program four
times as many.

The rest of the op library still emits the RK3588 encoding. On this part the matmul entries
refuse rather than submit a program the hardware will not run.

#### Prepacked convolution

Some layers run repeatedly, in a graph's forward pass or over a video stream. For those,
`rocket_conv2d_int8_pack_rk3576()` packs the weights once, and
`rocket_conv2d_int8_prepacked_rk3576()` runs the packed handle per inference.

Everything a per-call entry rebuilds from the weights alone is prepared once: the
per-output-channel filter sums, the coefficient group and the weight cube. So is every
buffer whose size follows the shape: the feature cube, the register program and each tile's
output surface. An inference then allocates no buffers and does no weight arithmetic. On
MobileNetV1-224 that is **98.8 ms to 33.6 ms per inference on the same 40 submits**, with
the same result to the byte.

The handle is bound to the `fd` it was packed on, because a buffer belongs to the file that
created it. It freezes the output-channel tiling and the quantization contract it was packed
for. It is not thread-safe, so pack per thread or serialize.

#### Residual add

A residual add is a convolution on this part rather than an elementwise op. The DPU's
elementwise stage takes exactly one operand. A skip is therefore lowered by concatenating the
two operands along channels, then convolving with a 1×1 kernel of two diagonal blocks.
`rocket_residual_add_weights_rk3576()` fills that weight matrix, its bias and its weight
scale and stops. It is pure and touches no hardware, and the caller runs the ordinary conv
entries on the result.

Two properties of the lowering beat a dedicated elementwise program:

- **The operands can carry different scales.** The ratio rides in the weights as a pair of
  int8s. Searching every denominator resolves it to about one part in 127², where anchoring
  the larger term at 127 gives one part in 127.
- **Both zero points ride exactly.** `w2*(a_zp - b_zp)` is a per-output-channel constant,
  and that is what the bias is.

It also fuses. A block's last convolution can absorb its own skip. It takes `C` more input
channels and an identity block at the center tap of its kernel. The add then costs no program
at all. The weight slice rule `ic*kh*kw <= 4608` bounds it, which a 1×1 project convolution
is nowhere near and a 3×3 reaches exactly at `C = 256`.

#### Cube layout and chained runs

A tensor can stay in cube layout between two layers, so neither host transpose runs at the
join. A direct convolution's output surface is the next layer's feature cube byte for byte
(`rocket_conv2d_int8_cube_of_rk3576()` / `_cube_in_` / `_cube_out_`). A run of cube-linked
layers goes out as one hardware kick through `rocket_chain_new_rk3576()`, with
`rocket_chain_plan_rk3576()` grouping already-linked nodes into the longest legal runs.

A run's members are *nodes* rather than convolution handles, because three kinds can sit in
one stream:

- A convolution.
- A pool. It carries its own register program and covers the interior only.
- A placement layer. The caller states that it emits no program and leaves no host work. A
  concatenation whose operands are already slices of one buffer is one.

Without that third kind, a layer with no program breaks every run it sits in.

#### Cube slices

A cube can be a slice of a larger buffer. Its base is a plain address on both sides of a
convolution. A producer can therefore be told to write its surface at a channel-group offset
inside a caller's allocation (`rocket_rk3576_cube_alloc()`, `_cube_slice()`,
`_cube_declare_tail()`, `rocket_conv2d_int8_cube_out_at_rk3576()`).

That is what makes a channel concatenation free. Producers writing their own slices of one
buffer already *are* the concatenated tensor. A residual add's operands and an Inception
module's four branches therefore need no host copy. The layer that feeds the skip does not
have to leave a row-major tensor behind for it.

A slice starts every sixteen channels, which is the width one atom interleaves. A slice is
sized for what its producer *writes*, the round-32 register channel count, and a consumer's
view for what its feature DMA *walks*. `_cube_declare_tail()` states that walk along with
the constant the groups past the live channels hold.

#### Whole networks

Four whole networks run end to end. Each layer is bit-exact against a CPU model of the
part's own arithmetic, and each network returns TFLite's own top-1. Each runs as one hardware
kick in one submit:

| Network | Compute layers | Topology | This library | Per-op entries, transient weights |
|---|---:|---|---:|---:|
| MobileNetV1-224 | 29 | | 4.9 ms | ~115 ms |
| MobileNetV2-224 | 64 | Ten residual skips | 6.7 ms | 112.9 ms |
| ResNet-18-224 | 31 | | 9.2 ms | 144.9 ms |
| Inception V1-224 | 81 | Nine four-operand concatenations, thirteen pooling layers | 10.8 ms | 159.7 ms |

The last column is the same graph with none of resident weights, cube layout, the placed
slices and the cross-layer kick. It costs tens of submits. The answer is the same to the byte
at every step.

### RK3566 and RK3568

The RK3566 and RK3568 are recognized from the device tree and warned about. They are named
but unprofiled, and not supported.

### Portability

Portability is by construction, and it is two layers rather than one. The RK3576 settled
that:

1. **Machine parameters.** CBUF banks and size, tile caps, tile-group sizes, datatype mask
   and worker default. These live in one `rocket_hw_profile` (`include/rocket_hw_profile.h`,
   read via `rocket_hw_current()`), which the tiling planners consult instead of bare
   literals. These must be *measured* on the part: the RK3576's CBUF, matmul tile cap (2048,
   not 256) and weight N-group (32, not 16) all differ from the RK3588's.
2. **A regcmd encoder for the CNA/CORE/DPU geometry registers**, which is *not* a shared
   offset table. The RK3576 re-packs those blocks at the same block bases. Registers move,
   the bit-packing differs at shared offsets, and it drives offsets the RK3588 leaves at
   reset. The emission is therefore per-chip (`src/npu_regcmd_rk3576.c`). What stays shared
   is the datapath *semantics*: the precision encodings, the CNA→CORE→DPU sequence, the
   BS/BN/EW/LUT field meanings, and the block bases.

Chip selection reads the NPU's device-tree `compatible`, with a `ROCKET_CHIP` override for
bring-up, and logs which datapaths the selected chip has an encoder for. A part with no
profile falls back to the RK3588's and says so, rather than silently applying the wrong
parameters.

## Performance

The matmul is a prefill and batched-GEMM engine. At 600 MHz its resident throughput is
~460 GOP/s across precisions: fp16 461, int8 386 and int4 413 GOP/s on `512×3840×4096`. As
an LLM prefill it runs Gemma-4-12B at ~15 t/s pp2048, which is ~3.2x the 8-thread CPU at
M>=512 and 3.6x at pp512. It is DMA/dispatch-bound rather than MAC-bound, so quantization
buys RAM and model-fit rather than prefill speed. Decode (M=1 GEMV) stays on the CPU.

### The MRDMA trap

A matmul regcmd that configures CNA/CORE/DPU but omits the DPU-RDMA block (`0x5xxx`) hangs,
because the DPU read-DMA waits forever. The matmul path emits the DPU-RDMA block and the
correct enable mask. Tiling past the 12×32 KB CBUF covers all Gemma FFN shapes, with M and N
independent and K split with host fp32 accumulation.

### The fp16 ladder

The ~15 t/s figure on Gemma-4-12B pp2048 is a stack of operating-point wins over the 200 MHz
boot clock:

| Step | Result |
|---|---|
| Baseline (200 MHz, tiled multicore fp16) | 7.98 t/s |
| Clock 600 MHz (`patches/rocket`) | 1.43x -> 11.40 |
| fp16 NPU K-accum (DPU eltwise-add, read each tile once) | +19% -> 13.38 |
| CBUF DATA_REUSE | +7% -> ~14.5 |
| Resident weights (prefill-only) | +6% -> ~15.1 |

A76 affinity, NEON fp16 converts and NEON readback de-tile trim the host buckets around
these. The CPU baseline is ~4.7 t/s flat, so the win is ~3.2-3.6x at M>=512 and about a tie
at pp128. With the clock raised and K-accum on, the matmul is NPU-`wait`-bound at ~60-68%
with packB ~22%. The remaining host work is memory-bound and gather-bound rather than
instruction-bound.

### Quantization and prefill speed

Quantization does not speed prefill at this operating point. Resident int8 and int4 tie the
~460 GOP/s floor, where the NPU runs at ~15% of fp16 MAC peak. In-model resident int8
prefill is 0.60x fp16, because its int32 readback cannot be K-accumulated: the DPU eltwise
operand DMA is <=16-bit. So int8 and int4 buy RAM and model-fit rather than throughput.
Treat that as bottleneck-conditional rather than a permanent property. The full per-dtype
detail is in the [datatype matrix](API.md#datatype-matrix).

## Capabilities and limitations

Everything above is HW-validated on the RK3588. The matmul dtypes are bit-exact at int8 and
int4, and within fp16 tolerance otherwise. The conv and op library each carry a CTest gate
against an fp64 oracle reference, and [API.md](API.md#tests) holds the full catalog. The
envelope:

- **Integer on-NPU K-accumulation is impossible** on this hardware. The conv accumulator
  reduces K only within one CBUF-resident tile. The only cross-tile adder, the DPU
  eltwise, has a <=16-bit operand DMA that int32 partials cannot fit, so integer K-partials
  accumulate on the host. fp16 partials do fit, and that is the on-NPU K-accum win.
- **Quantization does not accelerate prefill.** int8 and int4 are for RAM rather than speed.
- **Decode (M=1 GEMV) stays on the CPU**, ~82x slower on the NPU.
- **Host layout packing is irreducible.** The NPU has no on-chip row-major-to-tiled
  conversion, because the datapath has no transpose or RUBIK engine. You can only move,
  vectorize or amortize it.
- **The clock boots throttled at 200 MHz.** 600 MHz needs the `patches/rocket` clock patch,
  and 900 MHz is unstable here.
- **bf16, int16 and tf32 are completeness rungs rather than speedups.** bf16 and int16 tie
  the ~460 GOP/s floor and tf32 is half-rate. Only bf16 has an in-model use: fp32 range with
  no activation scaling.

## Build and test

No ML dependencies: libdrm and pthreads only.

```sh
cmake -S . -B build && cmake --build build -j        # -> build/librocketnpu.a
cmake -S . -B build -DBUILD_SHARED_LIBS=ON           # also emit librocketnpu.so
cmake --install build --prefix /usr/local            # export the rocketnpu::rocketnpu package
ctest --test-dir build                               # correctness gates (skip off-device)
```

The default is `-O2`. `-DROCKETNPU_OPT_FLAGS="-O3;-mcpu=native;-DNDEBUG"` overrides it,
though on the matmul path, which is memory-bound and gather-bound, these are flat within
±3%. Off-device the full build compiles on x86, with a portable `_Float16` and the NEON
readback intrinsics `__aarch64__`-gated. `ctest` is green there with the NPU gates skipped.
`drm/rocket_accel.h` must be present, and CMake checks at configure time.

Each file under `tests/` is a standalone executable that links the library and runs on the
NPU, doubling as a CTest correctness gate. [API.md](API.md#tests) holds the full catalog.

### Key runtime knobs

`sudo` strips the env, so use `sudo -E`:

| Knob | Default | Effect |
|---|---|---|
| `ROCKET_KACC` | on | fp16 NPU K-accumulation (+19%), the operating mode |
| `ROCKET_REUSE` | 2 | CBUF operand reuse (DATA_REUSE under KACC, +7%) |
| `ROCKET_N_THREADS` | 5 | Worker count, about one above the core count |
| `ROCKET_CPU_AFFINITY` | auto | Big-core set the pack and readback workers pin to |

[API.md](API.md#runtime-knobs) holds the full `ROCKET_*` reference, covering
flash-attention chaining, tiling overrides and batched submit, and the diagnostic log
channel.

### Submit provider

Every kernel interaction the library makes goes through one set of C symbols, the submit
seam. It covers device lifetime, buffer management, cache maintenance, submission and the
capability queries. The rest of the library targets the silicon, so it compiles unchanged whichever
kernel is underneath.

The seam has two implementations, selected at link time:

| `-DROCKETNPU_PROVIDER=` | Drives | Needs |
|---|---|---|
| `builtin` (default) | The mainline `rocket` DRM-accel driver, `/dev/accel/accel0` | `drm/rocket_accel.h` |
| `external` | Whatever you link, via `-DROCKETNPU_PROVIDER_LIB=` | Nothing from mainline |

`external` is how the library runs on a BSP kernel, where the NPU is an `rknpu` node rather
than an `accel/rocket` one. It is also the only configuration that builds at all on such a
board. The built-in provider needs a uAPI header the BSP kernel does not ship.

```sh
cmake -S . -B build -DROCKETNPU_PROVIDER=external \
      -DROCKETNPU_PROVIDER_LIB=/path/to/libyour-provider.a
```

The seam is exactly the externally-visible `rocket_*` functions defined in
`src/rocket_npu.c`, and **a provider must define all of them.** `tools/provider-seam.sh`
prints that list, and checks a provider against it:

```sh
tools/provider-seam.sh                          # the symbols a provider owes
tools/provider-seam.sh path/to/provider.a       # or a .o, a .so, or the .c source
```

Configure runs the same check whenever the provider is external and names a file, and stops
with the missing symbols listed. The failure is otherwise late and misleading. A provider
*defines* these symbols rather than calling them, so a seam that grows breaks nothing at the
provider's own compile. `librocketnpu.a` is a static archive, and it links happily with the
symbol unresolved. Without the check, the first sign is every executable in the tree failing
at the end of the build.

## The rocket NPU stack

This is the foundation of an open source stack for Rockchip NPUs, three userspace projects
plus a set of optional kernel patches:

- **`rocket-userspace`** (this project): the userspace driver, matmul and on-NPU op library.
  It is self-contained, and the two frontends below link it.
- **[`ggml-rocket`](https://github.com/gregordinary/ggml-rocket)**: a ggml backend `.so`, a
  drop-in NPU device for stock `llama.cpp` and `whisper.cpp`. Links `librocketnpu`.
- **[`tflite-rocket`](https://github.com/gregordinary/tflite-rocket)**: a TFLite external
  delegate for detection models. Links `librocketnpu`.
- **[`patches`](https://github.com/gregordinary/patches)** (`rocket/` scope): optional
  out-of-tree kernel-module patches for clock, voltage and IOMMU. They raise the NPU clock
  from its 200 MHz boot default to 600 MHz and trim dispatch latency, and the performance
  figures above assume them.

## License and credits

`librocketnpu` is GPL-3.0-or-later.

It builds on prior work:

- The RK3588 NPU register interface reverse-engineered by Jasbir Matharu
  (`mtx512/rk3588-npu`), whose copyright is retained verbatim in the hardware headers
  (`npu_cna.h`, `npu_dpu.h`, `npu_hw.h`, `npu_matmul.h`).
- The `rocket` regcmd format established by the Mesa Teflon "rocket" gallium driver by Tomeu
  Vizoso (MIT). The CNA→CORE→DPU register sequence this library emits derives from that
  work, and no Mesa source is vendored here.
- johanvdb/librocket, a FOSS userspace fp16 matmul on mainline `rocket` that combined the
  two above, which served as the starting point for the kernel-access layer.

This project uses that prior work as a starting point. It adopts the RK3588 register headers
and rewrites the kernel-access layer for the mainline `rocket` DRM-accel driver. It then adds
the tiled/multicore/resident/multi-dtype matmul and the on-NPU op library. Those additions
are validated bit-exact on real RK3588 hardware.
