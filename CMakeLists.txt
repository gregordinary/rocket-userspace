# rocket-userspace — FOSS userspace driver/library for the RK3588 NPU via the mainline
# "rocket" DRM-accel driver. This builds the DRIVER project: a standalone library
# (librocketnpu) exposing the shim + fp16 matmul (tiled + multicore). It has NO
# dependency on any ML framework — the ggml-rocket backend is a SEPARATE project
# that links this library, keeping driver and backend independently buildable.
#
#   cmake -S . -B build && cmake --build build -j
#   cmake --install build --prefix /usr/local   # exports rocketnpu::rocketnpu
#   cmake -S . -B build -DBUILD_SHARED_LIBS=ON   # also build the .so
#   cmake -S . -B build -DROCKETNPU_BUILD_TESTS=OFF
cmake_minimum_required(VERSION 3.16)
project(rocketnpu VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)                # rocket_matmul_mt.c

# --- which submit provider is compiled in -------------------------------------
#
# The submit seam is include/rocket_npu.h: device open/close, BO alloc/free, cache
# maintenance, submit, and the capability + counter queries. `builtin` compiles
# src/rocket_npu.c, the shim over the mainline `rocket` DRM-accel driver, and is what
# every consumer gets by default. `external` omits it, leaving those symbols for a
# provider the consumer links alongside this library -- the rknpu-submit provider for
# the vendor BSP driver, or a host-memory mock for off-device gates.
#
# Selection is at LINK time rather than through a vtable: the seam is a set of C
# symbols already, one build targets one driver, and an indirect call on every BO
# operation would buy nothing here. Everything ABOVE the seam is silicon-shaped and
# identical either way.
set(ROCKETNPU_PROVIDER "builtin" CACHE STRING
    "Submit provider: builtin (mainline rocket DRM-accel) or external (linked separately)")
set_property(CACHE ROCKETNPU_PROVIDER PROPERTY STRINGS builtin external)
if(NOT ROCKETNPU_PROVIDER MATCHES "^(builtin|external)$")
    message(FATAL_ERROR "ROCKETNPU_PROVIDER must be 'builtin' or 'external', got '${ROCKETNPU_PROVIDER}'")
endif()
# A CMake target or a plain library path. Carried as an INTERFACE dependency of
# rocketnpu, so tests and frontends pick it up with no per-target change.
set(ROCKETNPU_PROVIDER_LIB "" CACHE STRING
    "Provider library every consumer links alongside rocketnpu (external provider only)")
message(STATUS "rocketnpu: submit provider = ${ROCKETNPU_PROVIDER}")

# An external provider must define the whole seam, and a missing symbol is not a compile
# error anywhere near it: src/rocket_npu.c is compiled only for the builtin provider, so a
# symbol added there and called from above the seam is an undefined reference in EVERY
# external build. librocketnpu.a is a static archive and links anyway, which is why the
# first sign of it is dozens of executables failing at the very end of the build.
#
# Check it here instead, before an object is compiled, and name the symbols. Only possible
# when the provider is a file this can read -- ROCKETNPU_PROVIDER_LIB is also allowed to be
# a CMake target, whose archive does not exist yet at configure time.
if(ROCKETNPU_PROVIDER STREQUAL "external" AND EXISTS "${ROCKETNPU_PROVIDER_LIB}")
    execute_process(
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tools/provider-seam.sh" "${ROCKETNPU_PROVIDER_LIB}"
        RESULT_VARIABLE _seam_rc OUTPUT_VARIABLE _seam_out ERROR_VARIABLE _seam_err)
    if(_seam_rc EQUAL 1)
        message(FATAL_ERROR
            "the external provider does not implement librocketnpu's submit seam.\n"
            "${_seam_err}${_seam_out}"
            "Update the provider, or run tools/provider-seam.sh with no arguments to list "
            "the seam it has to define.")
    elseif(NOT _seam_rc EQUAL 0)
        message(STATUS "rocketnpu: seam check skipped (${_seam_err})")
    else()
        message(STATUS "rocketnpu: provider implements the submit seam")
    endif()
endif()

# libdrm only provides <libdrm/drm.h>; the actual build dependency is the mainline
# rocket DRM-accel uAPI header <drm/rocket_accel.h> (the DRM_IOCTL_ROCKET_* / struct
# drm_rocket_* definitions), which ships as include/uapi/drm/rocket_accel.h and is
# usually installed to /usr/include/drm/rocket_accel.h. Check it at configure time so a
# missing uAPI header is a clear message, not a confusing mid-compile error.
#
# Only the builtin provider includes that header. A BSP-kernel box has no mainline
# rocket uAPI at all, so demanding it there would stop the library configuring for a
# provider that does not want it.
if(ROCKETNPU_PROVIDER STREQUAL "builtin")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DRM REQUIRED libdrm)    # <libdrm/drm.h>
    include(CheckIncludeFile)
    set(CMAKE_REQUIRED_INCLUDES ${DRM_INCLUDE_DIRS})
    check_include_file("drm/rocket_accel.h" HAVE_DRM_ROCKET_ACCEL_H)
    unset(CMAKE_REQUIRED_INCLUDES)
    if(NOT HAVE_DRM_ROCKET_ACCEL_H)
        message(FATAL_ERROR
            "drm/rocket_accel.h not found. Install the mainline rocket DRM-accel uAPI header "
            "(usually /usr/include/drm/rocket_accel.h, from include/uapi/drm/rocket_accel.h in "
            "the kernel source) before building rocket-userspace, or configure with "
            "-DROCKETNPU_PROVIDER=external to supply a provider of your own.")
    endif()
endif()

# --- the library: shim + validated fp16 generator + tiled + multicore matmul ---
set(ROCKETNPU_SOURCES
    src/rocket_log.c          # centralized log channel (levels + host-settable callback)
    src/rocket_sysfs.c        # the one enumeration of the bound NPU cores (core side of the seam)
    src/rocket_busy_poll.c    # the completion-wait spin budget (core side of the seam)
    src/rocket_hw_profile.c   # the chip machine-parameter profile (RK3588 today)
    src/npu_regcmd.c     # regcmd generators (fp16/int8/int4/int16/bf16/tf32)
    src/npu_regcmd_rk3576.c   # RK3576 int8 conv regcmd (its own geometry-register encoding)
    src/rocket_chain.c        # contiguous self-chaining regcmd layout (batched submit, shared)
    src/rocket_fanout.c       # worker pool + per-shape scratch cache, shared by every resident path
    src/rocket_op.c           # the one-shot op scaffolding (regcmd bound, submit, fence wait)
    src/rocket_matmul.c       # tiled single-fd matmul (CBUF-fit, CPU K-accum)
    src/rocket_matmul_rk3576.c # the RK3576 int8 matmul (1x1 conv, its own cubes + envelope)
    src/rocket_rk3576_cube_pack.c # the CHW <-> NC1HWC2 transpose (16x16 NEON block), shared
    src/rocket_conv2d_rk3576.c # the RK3576 CONV_2D entries (int8 direct/depthwise, fp16)
    src/rocket_conv.c         # general fp16 CONV_2D (KxK/stride/pad/dilation)
    src/rocket_conv_transpose.c # ConvTranspose2d (deconv) lowered onto the forward conv
    src/rocket_resize.c       # nearest/bilinear upsample via depthwise ConvTranspose
    src/rocket_pool.c         # on-NPU MaxPool / AveragePool via the PPU
    src/rocket_pool_rk3576.c  # the RK3576 PPU pooling entry (its own program, half-to-even avg)
    src/rocket_lut_rk3576.c   # the RK3576 DPU LUT table builder (the activation grid)
    src/rocket_reduce.c       # on-NPU GlobalAvgPool / Mean (multi-pass PPU reduction) + feature-axis reduce
    src/rocket_norm.c         # on-NPU RMSNorm + per-row broadcast scale
    src/rocket_normvision.c   # on-NPU BatchNorm/GroupNorm/InstanceNorm/L2Norm (vision normalization)
    src/rocket_ffn.c          # on-NPU gated-MLP FFN block (GeGLU/SwiGLU core + projections)
    src/rocket_softmax.c      # on-NPU row-wise softmax (EXP LUT + row-sum reduce + per-row scale)
    src/rocket_attn.c         # on-NPU multi-head self-attention (QKV + scores + softmax + P·V + out-proj)
    src/rocket_encoder.c      # one Whisper encoder block (LN -> MHA -> residual -> LN -> MLP -> residual)
    src/rocket_siglip_encoder.c # SigLIP-B/16 vision encoder (patch-embed + pos + 12x block + post-LN)
    src/rocket_sam_encoder.c  # SAM ViT-Det image encoder (windowed + global attn + decomposed rel-pos + conv neck)
    src/rocket_activation.c   # elementwise fp16 activation via the DPU LUT block
    src/rocket_matmul_mt.c    # multicore fan-out (per-core worker fds)
    src/rocket_bf16_stream.c  # fast bf16 matmul: multicore fan-out + streaming resident scratch
    src/rocket_prepacked.c    # pack-weights-once path (persistent fds + resident BOs)
    src/rocket_prepacked_int8.c # resident int8 (W8A8) weight path
    src/rocket_prepacked_int4.c # resident int4 (W4A4) weight path
    src/rocket_affinity.c)    # pin pack/readback workers to the big (A76) cores

# The submit provider itself. `external` leaves the seam's symbols undefined in this
# library; the consumer links a provider that defines them.
if(ROCKETNPU_PROVIDER STREQUAL "builtin")
    list(APPEND ROCKETNPU_SOURCES src/rocket_npu.c)   # DRM-accel shim over /dev/accel/accel0
endif()

# Optimization flags for the hot library (pack/scatter/readback/CPU-K-accum).
# Defaults to -O2. Override per-build to benchmark or to
# adopt a faster target, e.g.:
#   -DROCKETNPU_OPT_FLAGS="-O3;-mcpu=native;-DNDEBUG"
#   plus -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON for LTO.
# Appended AFTER -O2 so a later -O3 wins. NOT a place for -ffast-math (the layout
# index math + fp16 converts care about NaN/Inf/rounding); -fno-math-errno /
# -fno-trapping-math are the safe subset, gated by the cosine-sim correctness
# matrix (tests/correctness_matrix.sh).
set(ROCKETNPU_OPT_FLAGS "" CACHE STRING "extra optimization flags for the hot library")

# OBJECT lib so we can emit both static and (optionally) shared from one compile.
add_library(rocketnpu_obj OBJECT ${ROCKETNPU_SOURCES})
set_target_properties(rocketnpu_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(rocketnpu_obj
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${DRM_INCLUDE_DIRS})
target_compile_options(rocketnpu_obj PRIVATE -O2 -Wall -Wextra ${DRM_CFLAGS_OTHER} ${ROCKETNPU_OPT_FLAGS})

add_library(rocketnpu STATIC $<TARGET_OBJECTS:rocketnpu_obj>)
add_library(rocketnpu::rocketnpu ALIAS rocketnpu)
target_include_directories(rocketnpu
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include/rocketnpu>)
target_link_libraries(rocketnpu PUBLIC Threads::Threads m PRIVATE ${DRM_LIBRARIES})
# With an external provider the seam's symbols are undefined in this archive. Carrying
# the provider as an INTERFACE dependency here is what makes every consumer — the test
# executables, rocketgraph, the frontends — link it without a per-target change.
#
# The two archives reference each other, so the link needs a RESCAN: the provider
# defines the seam this library calls, and calls back into rocket_hw_current() to
# resolve the chip profile at open. Whichever archive a one-pass linker sees first has
# its unresolved references dropped, so the group is not optional — without it the
# error is an undefined rocket_hw_current at every consumer's link.
#
# The archive is named through the NAMESPACED alias, not the bare target. This
# property is exported verbatim, and the genex is evaluated in the consumer's
# project, where the plain name `rocketnpu` does not exist - a find_package()
# consumer of an INSTALLED external-provider package then fails at generate with
# `No target "rocketnpu"`. The imported `rocketnpu::rocketnpu` is defined in both
# trees, so the alias form is the one that works from a build tree and an install.
if(ROCKETNPU_PROVIDER STREQUAL "external" AND ROCKETNPU_PROVIDER_LIB)
    target_link_libraries(rocketnpu INTERFACE
        -Wl,--start-group ${ROCKETNPU_PROVIDER_LIB} $<TARGET_FILE:rocketnpu::rocketnpu> -Wl,--end-group)
endif()

# --- the RK3576 graph planner: a component, NOT part of the library ---------------
#
# No graph-execution object lives in librocketnpu — a frontend owns scheduling and its own
# tensor model, and the one thing it cannot own (which streams the part will run) is the
# run FINDER, which is pure and is in the library already. What is between those two is the
# placement and linking rules: generic to the part, not the part, and expensive to get
# right — twelve refusal classes, every one closed so far a host-buffer rule rather than
# the hardware's. So they are a separate target the frontends and the gate both link,
# because a second copy of them would fork silently.
add_library(rocketgraph STATIC graph/rocket_graph_rk3576.c)
add_library(rocketnpu::rocketgraph ALIAS rocketgraph)
target_include_directories(rocketgraph
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/graph>
           $<INSTALL_INTERFACE:include/rocketnpu>)
target_link_libraries(rocketgraph PUBLIC rocketnpu)
target_compile_options(rocketgraph PRIVATE -O2 -Wall -Wextra ${DRM_CFLAGS_OTHER})

if(BUILD_SHARED_LIBS)
    add_library(rocketnpu_shared SHARED $<TARGET_OBJECTS:rocketnpu_obj>)
    set_target_properties(rocketnpu_shared PROPERTIES
        OUTPUT_NAME rocketnpu VERSION ${PROJECT_VERSION} SOVERSION 0)
    target_include_directories(rocketnpu_shared
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:include/rocketnpu>)
    target_link_libraries(rocketnpu_shared PUBLIC Threads::Threads m PRIVATE ${DRM_LIBRARIES})
endif()

# --- public API (consumed by the ggml-rocket backend) ---
set(ROCKETNPU_PUBLIC_HEADERS
    include/rocket_log.h       # log levels + host-settable log callback
    include/rocket_npu.h       # device shim: open/bo/submit
    include/rocket_hw_profile.h # the chip machine-parameter profile + capability query
    include/rocket_matmul.h    # rocket_matmul_fp16 / _mt / _plan
    include/rocket_conv.h      # rocket_conv2d_fp16 / _plan / _ref (general conv) + ConvTranspose2d
    include/rocket_resize.h    # rocket_upsample_{nearest,bilinear}_fp16 (depthwise-transpose resize)
    include/rocket_pool.h      # rocket_pool_fp16 / _plan / _ref (on-NPU MaxPool/AvgPool)
    include/rocket_reduce.h    # rocket_global_avgpool_fp16 / _plan / _ref (GlobalAvgPool/Mean) + rocket_reduce_feature_fp16
    include/rocket_norm.h      # rocket_rmsnorm_fp16 + rocket_scale_rows_fp16 (on-NPU normalization)
    include/rocket_normvision.h # rocket_batchnorm/groupnorm/instancenorm/l2norm_fp16 (vision normalization)
    include/rocket_ffn.h       # rocket_ffn_fp16 + rocket_geglu_fp16 (gated-MLP FFN block)
    include/rocket_softmax.h   # rocket_softmax_fp16 (row-wise softmax)
    include/rocket_attn.h      # rocket_mha_self_fp16 (multi-head self-attention)
    include/rocket_encoder.h   # rocket_encoder_block_fp16 (one Whisper encoder block)
    include/rocket_siglip.h    # rocket_siglip_encode (SigLIP-B/16 vision encoder end to end)
    include/rocket_sam.h       # rocket_sam_encode (SAM ViT-Det image encoder end to end)
    include/rocket_activation.h # rocket_activation_fp16 (DPU LUT elementwise)
    include/npu_matmul.h       # matmul_params_t + gen_matmul_fp16 / gen_conv2d_* (low-level)
    include/npu_pool.h         # pool_params_t + gen_pool_fp16 (low-level PPU pooling)
    include/npu_activation.h   # lut_act_params_t + gen_lut_activation_fp16 (low-level)
    # internal headers, but #included (transitively) by the public ones above — must be
    # installed or a consumer of the installed pkg (e.g. the tflite-rocket delegate .so)
    # fails to find them. npu_matmul.h / rocket_activation.h -> npu_dpu.h; keep npu_cna.h /
    # npu_hw.h alongside (register/descriptor defs the low-level API leans on).
    include/npu_dpu.h
    include/npu_cna.h
    include/npu_hw.h)

include(GNUInstallDirs)
install(TARGETS rocketnpu EXPORT rocketnpu-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
# The planner ships with the package: a frontend that links the INSTALLED rocketnpu (the
# tflite-rocket delegate does) has to reach it, and a second copy of its rules would fork
# silently. Its header goes in the same include directory as the library's.
install(TARGETS rocketgraph EXPORT rocketnpu-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES graph/rocket_graph_rk3576.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocketnpu)
if(BUILD_SHARED_LIBS)
    install(TARGETS rocketnpu_shared EXPORT rocketnpu-targets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
install(FILES ${ROCKETNPU_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocketnpu)
install(EXPORT rocketnpu-targets NAMESPACE rocketnpu::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocketnpu)
export(EXPORT rocketnpu-targets NAMESPACE rocketnpu::
       FILE ${CMAKE_CURRENT_BINARY_DIR}/rocketnpu-targets.cmake)

# Package config so consumers can `find_package(rocketnpu CONFIG)` (the EXPORT above
# only ships the targets file; CMake needs a Config wrapper to discover it). Generated
# into the build tree (so an uninstalled build is also find_package-able via
# -Drocketnpu_DIR=<build>) and installed alongside the targets file.
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocketnpu-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rocketnpu-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocketnpu)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rocketnpu-config-version.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rocketnpu-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rocketnpu-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocketnpu)

# --- tests / benches (link the library) ---
option(ROCKETNPU_BUILD_TESTS "Build rocket-userspace test/bench executables" ON)
if(ROCKETNPU_BUILD_TESTS)
    foreach(t
            matmul_int8_rocket     # standalone int8 x int8 -> int32
            matmul_int4_rocket     # int4 x int4 -> int16 + encoding sweep
            matmul_int4_tiled_rocket # tiled int4 (M/N tiles + host K-accum), bit-exact
            matmul_int4_groupwise_rocket # group-wise int4 (per-K-group scales, fp32 K-accum)
            matmul_int4_prepacked_rocket # resident int4 bit-exact vs one-shot
            matmul_int4_prepacked_gw_rocket # resident GROUP-WISE int4 vs one-shot + fp64 (in-model W4A4)
            matmul_int4_crossm_rocket # resident int4 weight reused across M, no re-pack (warmup-M serves prefill-M)
            matmul_bf16_rocket     # bf16 x bf16 -> fp32 + encoding sweep
            matmul_bf16_tiled_rocket # tiled bf16 (M/N/K tiles + host fp32 K-accum)
            matmul_bf16_stream_rocket # fast bf16: multicore fan-out + streaming resident scratch
            matmul_fp32out_rocket  # fp16->fp32-output: beats fp16-out vs fp64 ref
            matmul_tf32_rocket     # tf32 x tf32 -> fp32 + 4-byte-input geometry + precision sweep
            matmul_tf32_tiled_rocket # tiled tf32 (M/N/K tiles + host fp32 K-accum)
            matmul_int16_rocket    # int16 x int16 encoding sweep (RE: native output modes)
            matmul_int16_exact_rocket # bit-exact int16->int64 via int8 byte-decomposition
            matmul_dtype_perf_rocket # fp16 vs int8 vs int4 resident throughput
            matmul_int8_tiled_rocket # tiled int8 (M/N tiles + host K-accum)
            mixed_chain_coexist_rocket # GATE: chained-fp16 + gapped-int8 in one process (per-job batched flag)
            matmul_int8_prepacked_rocket # resident int8 weights (bit-exact vs one-shot)
            matmul_int8_crossm_rocket # resident int8 weight reused across M, no re-pack (warmup-M serves prefill-M)
            matmul_int8_groupwise_rocket # group-wise int8 (per-K-group scales, fp32 K-accum) — the native-quant primitive
            matmul_int8_prepacked_gw_rocket # resident GROUP-WISE int8 vs one-shot + fp64 (native-quant MoE experts)
            matmul_int8_crossm_gw_rocket # resident group-wise int8 weight reused across M, no re-pack
            matmul_int8_dequant_rocket # fold int8 dequant scale into DPU OUT_CVT (int8->float out)
            activation_lut_rocket  # DPU LUT elementwise activation (sigmoid/hardsigmoid) HW gate
            leaky_relu_rocket      # parametric LeakyReLU on the DPU LUT (shifted single-table, x≈0 glitch-free)
            recip_rsqrt_rocket     # positive-domain LUT (sqrt/rsqrt/reciprocal) + on-NPU Div accuracy gate
            lut_tanh_rocket        # SIGNED-output LUT (tanh bias-trick) — OUT_CVT affine HW gate/RE
            exp_lut_rocket         # EXP DPU-LUT (shifted single-table) + softmax-sum end to end
            ew_mul_rocket          # fully-on-NPU elementwise multiply (conv-main EW op) bit-exact gate
            ew_minmax_rocket       # on-NPU elementwise two-tensor MAX/MIN (EW ALU algo) + Clip, bit-exact
            prelu_rocket           # on-NPU PReLU (per-channel slope; max(x,alpha*x) / general), bit-exact
            conv_act_rocket        # conv -> activation FUSION (single job): SiLU/tanh/GELU LUT epilogue
            gelu_rocket            # 2-pass on-NPU GELU (x·Φ(x), clean unit-LUT gate) vs true erf-GELU over wide inputs
            softplus_mish_rocket   # Softplus/Mish (YOLOv4/v7)/Abs DPU-LUT activations vs double-precision math
            elu_rocket             # ELU/SELU on the symmetric shifted single-table (kink on the middle sample)
            bytes_moved_rocket     # analytical DRAM-traffic model (pure; planner cross-check + readback floor)
            conv2d_fp16_rocket     # general fp16 CONV_2D: cube-layout self-check + HW oracle
            conv1d_rocket          # Whisper conv1d front-end (height-1 conv2d lowering) bit-exact
            conv_transpose_rocket  # ConvTranspose2d: lowering self-check + HW oracle (vs scatter ref)
            deconv_mode_probe      # RK3588 board: is CNA_CONV_CON1 bit 16 (DECONV) a live hardware mode?
            resize_rocket          # nearest/bilinear upsample (depthwise transpose) vs independent gather refs
            pool_fp16_rocket       # on-NPU MaxPool/AvgPool (PPU): cube self-check + HW oracle
            pool_int8_rocket       # on-NPU int8/uint8 MaxPool/AvgPool (PPU C2=16): cube self-check + HW oracle
            reduce_mean_rocket     # on-NPU GlobalAvgPool/Mean: multi-pass decomposition + HW oracle
            reduce_feature_rocket  # feature-axis reduce (ones-matmul): sum/mean over H per row vs fp64 oracle
            cumsum_rocket          # on-NPU prefix sum (triangular ones-matmul): incl/excl/reverse vs fp64 oracle
            rmsnorm_rocket         # on-NPU RMSNorm + per-row broadcast scale vs fp64 oracle
            norm_vision_rocket     # on-NPU BatchNorm/GroupNorm/InstanceNorm/L2Norm vs fp64 oracle
            ffn_rocket             # on-NPU gated-MLP FFN block (GeGLU/SwiGLU core + projections) vs fp64 oracle
            ffn_fused_rocket       # cross-op cube-resident FFN (gate/up->act⊙up->down, intermediates never de-tiled)
            softmax_rocket         # on-NPU row-wise softmax (EXP + row-sum + per-row scale) vs fp64 oracle
            cross_entropy_rocket   # stable per-row cross-entropy (logsumexp reduce + host gather) vs fp64 oracle
            layernorm_rocket       # on-NPU LayerNorm (stacked-row reduce + affine fold) vs fp64 oracle
            mha_rocket             # on-NPU multi-head self-attention (scores+softmax+P·V) vs fp64 oracle (cos-sim)
            flash_attn_rocket      # masked GQA flash attention (decoder/LLM prefill, FLASH_ATTN_EXT) vs fp64 oracle (cos-sim)
            encoder_block_rocket   # one full Whisper encoder block (LN+MHA+residual+LN+MLP) vs fp64 oracle (cos-sim)
            siglip_rocket          # full SigLIP-B/16 vision encoder vs fp32 HF oracle (per-layer cos-sim) + bench
            sam_rocket             # full SAM ViT-Det image encoder vs fp32 HF ref (per-layer cos-sim) + host-ref self-check + bench
            conv2d_int8_rocket     # native int8 CONV_2D: cube self-check + int32 single-job HW gate
            replay_dw_mesa         # replay captured Teflon int8 DW BOs vs mesa-output (ground truth)
            conv_dw_int8_runtime   # int8 DW int8-out RUNTIME (host packing) vs mesa-output
            matmul_correctness_matrix_rocket # cosine-sim layout/readback gate (realistic inputs)
            matmul_tiled_rocket    # tiled matmul + profiling/sweep knobs
            fc_data_bank_sweep_rocket # RE: CNA_CBUF_CON0 FC_DATA_BANK[10:8] sweep + large-K probe
            matmul_mt_rocket       # multicore matmul: correctness + scaling
            matmul_batch_rocket    # batched same-shape matmul (one job/group) bit-exact vs per-item (attn QK/AV chaining)
            matmul_prepacked_rocket # pack-weights-once path vs mt vs CPU ref
            matmul_prepacked_crossm_rocket # resident weight reused across compatible M (no re-pack)
            scratch_cache_gate     # GATE: the per-shape scratch cache recycles instead of refusing
            matmul_stream_vs_prepacked_rocket # per-call packB cost (stream vs resident)
            iova_ceiling_rocket    # NPU 32-bit IOVA window size + per-fd vs shared
            prototype_shared_scratch_rocket # shared-scratch ownership prototype
            crossop_chain_rocket   # cross-op chaining: feed one matmul's output BO straight into the next's input
            replay_dump            # replay a dumped in-context failing matmul
            matmul_kacc_chain_rocket # cross-ki KACC chaining: chained == unchained byte-exact + CPU ref
            matmul_kacc_chain_bench # cross-ki KACC chaining warm-loop perf probe (hand-run, not a gate)
            matmul_accum_rocket    # DPU eltwise K-accum experiment (fp16)
            matmul_accum_int8_rocket # isolate int32 DPU eltwise-add for int8 K-accum
            multicore_probe        # 1 fd, N jobs -> serial (scheduling probe)
            multicore_threads      # N fds/threads -> parallel cores
            ctx_pool_throughput    # multi-instance context-pool throughput sweep (detection)
            moe_gw_bench           # resident group-wise int8 at the MoE expert shape (hand-run, not a gate)
            uapi_bo_lifetime_rocket # rocket uAPI GATE: a BO referenced by an in-flight job survives its file's close
            prep_signal_robust_rocket # PREP_BO wait: EINTR-retry + UINT64_MAX-deadline saturation gate
            submit_overhead_rocket # per-job dispatch floor (us/submit) A/B probe for IOMMU keep-attach
            x0_glitch_probe        # RE: does the x≈0 LE/LO mux glitch fire for a shifted table straddling 0?
            regcmd_persist_rocket  # does register state persist across tasks in a job (delta regcmd)?
            # host-only regcmd dumpers + a standalone fp16 test + a DRAM microbench: tools, not CTest gates
            matmul_fp16_rocket     # standalone fp16 x fp16 single-task test (gen_matmul_fp16 path)
            dump_regcmd            # host-only: diff the matmul regcmd accumulate=0 vs 1 (no HW)
            dump_dw_regcmd         # host-only: emit depthwise-conv regcmd for the Mesa decoder
            chain_layout_rocket    # host-only GATE: batched-submit contiguous layout + trailer link (no HW)
            chain_verify_gate      # host-only GATE: rkt_chain_verify catches a wrong next_count (no HW)
            matmul_plan_gate       # host-only GATE: the shared matmul tiling policy's invariants (no HW)
            regcmd_rk3576_gate     # host-only GATE: RK3576 conv regcmd vs the vendor capture (no HW)
            rk3576_first_light     # RK3576 board: does the RK3576 emitter make the NPU compute?
            rk3576_conv_gate       # RK3576 board GATE: the int8 conv envelope + row windows + depthwise
            rk3576_conv_lib_gate   # RK3576 board GATE: the same envelope through the LIBRARY entries
            rk3576_fp16_sweep      # RK3576 board: settle the fp16 / asymmetric encodings against a CPU model
            rk3576_matmul_probe    # RK3576 board: the matmul-as-1x1-conv envelope (M axis, cost, precision)
            rk3576_matmul_gate     # RK3576 board GATE: the int8 matmul library entry vs a CPU model
            rk3576_refusal_gate    # RK3576 board GATE: ops with no encoder refuse, not write nothing
            rk3576_submit_floor    # RK3576 board: the per-submit dispatch floor (us/submit) probe
            rk3576_mm_corr         # RK3576 board: does an arbitrary matmul shape lose surface, and how often?
            rk3576_mm_requant      # RK3576 board GATE: the matmul entry's requant vs the host models of it
            rk3576_w8a8_route      # RK3576 board GATE: the W8A8 two-pass calibration route, composed on the part
            rk3576_fc_vendor       # RK3576 board: submit a vendor capture VERBATIM — does it write?
            rk3576_cost_table      # RK3576 board: what the library entries cost, against a CPU reference
            rk3576_stem_cost       # RK3576 board: is a conv's execution its MACs or its feature DMA?
            rk3576_poison_probe    # RK3576 board: which programs poison the NEXT submit, and what carries it
            rk3576_argb_ic1        # RK3576 board: why the int8 first conv writes nothing at one image channel
            rk3576_fp16_split_cost # RK3576 board: where the fp16 ic split spends its wall
            rk3576_multicore       # RK3576 board: what a second NPU core buys, across W worker fds
            rk3576_i32_height_bound # RK3576 board: is the plane width in the wide int32 writer's bound?
            rk3576_i32_zero_run    # RK3576 board: what selects the wide int32 writer's zero-emission runs
            rk3576_core_pair       # RK3576 board: what makes two cores in flight compute wrong answers
            rk3576_core_asym       # RK3576 board: is the shared resource a CAPACITY or a control path?
            rk3576_cbuf_base       # RK3576 board: is the CBUF base an address, and does biasing it partition the cores?
            rk3576_coeff_c         # RK3576 board: what the coefficient group's C multiplier multiplies (per-axis quant)
            rk3576_perchannel_gate # RK3576 board GATE: per-output-channel int8 conv vs the chip's own arithmetic
            rk3576_pool_probe      # RK3576 board: make the PPU write (the destination base no capture carries)
            rk3576_add_probe       # RK3576 board: the DPU's elementwise stage — does it compute, and to what model
            rk3576_residual_add    # RK3576 board: the residual add, lowered onto the convolution datapath
            rk3576_lut_probe       # RK3576 board: the DPU LUT's input map, off our own table-load program
            rk3576_act_gate        # RK3576 board: the int8 LUT activation entry, against a model of the chip
            rk3576_argb_pad        # RK3576 board: is a TFLite stem reachable on the int8 first conv
            rk3576_argb_extent     # RK3576 board: does the packed-image first conv honour an asymmetric extent?
            rk3576_argb_padcon     # RK3576 board: what value does a PAD TAP feed the MAC on the packed path?
            rk3576_argb_padmap     # RK3576 board: WHERE does a pad tap's value come from (position, not word)?
            rk3576_argb_border_cost # RK3576 host: what the two routes past the packed stem's column bound cost
            rk3576_argb_extend     # RK3576 board: can the packed stem's pad columns be MATERIALISED?
            rk3576_net_gate        # RK3576 board GATE: a whole quantized classifier, layer by layer
            rk3576_chain_raw       # RK3576 board: does a chained stream honour RAW between its programs?
            rk3576_chain_pool      # RK3576 board: may a POOL program sit inside a chained convolution stream?
            rk3576_chain_argb      # RK3576 board: may a PACKED-IMAGE first conv head a chained stream?
            rk3576_chain_len       # RK3576 board: how long a chained stream may be, and what the bound follows
            rk3576_offset_cube     # RK3576 board: is a cube base a plain address (a channel-group offset)?
            rk3576_pad_channels    # RK3576 board: what a producer leaves past its live output channels
            rk3576_surf_stride     # RK3576 board: will the CNA read a cube whose groups sit further apart than the plane?
            rk3576_row_pitch       # RK3576 board: will the PPU read a plane whose ROWS sit further apart than its windows consume?
            rk3576_conv_pitch      # RK3576 board: and will the CNA? a readout of what 0x1090 does
            rk3576_conv_sym        # RK3576 board: the symmetry audit on the CNA — non-square strides, and a rung fitted at kh == kw
            rk3576_rowplan         # RK3576 host: the row window the planner chooses for one layer, and what it stages
            requant_round_probe    # board: which way does the DPU's OUT_CVT round an exact TIE?
            hostload               # host memory-system load generator: run any gate under DDR / L1 / spin pressure
            membench               # DRAM bandwidth + NPU readback de-tile microbench
            fabench)               # flash-attention QK/AV submit-chaining A/B (ROCKET_FA_CHAIN 0 vs 1)
        add_executable(${t} tests/${t}.c)
        target_link_libraries(${t} PRIVATE rocketnpu)
        target_compile_options(${t} PRIVATE -Wall -Wextra)
    endforeach()
    # Gates ON the mainline rocket uAPI rather than on the library: each includes
    # <drm/rocket_accel.h> and drives DRM_IOCTL_ROCKET_* itself, so what it asserts is a
    # contract of THAT driver and it cannot be built, let alone run, against another
    # provider. Excluded from an external-provider build rather than made to skip — a
    # test that cannot compile is not a test that skipped.
    if(ROCKETNPU_PROVIDER STREQUAL "builtin")
        foreach(t
                uapi_selftest_rocket   # rocket uAPI conformance gate (BO/deadline/IOVA contracts)
                uapi_submit_errpath_rocket # rocket uAPI GATE: a rejected submit returns an errno, it does not oops
                uapi_bo_ranges_rocket  # rocket uAPI GATE: the ranged cache-maintenance ioctls accept and refuse correctly
                rk3576_prep_floor      # RK3576 board: what a PREP_BO/FINI_BO bracket costs, floor vs bytes
                uapi_regcmd_fault_rocket) # probe, NOT a gate: a client-requested NPU DMA fault (faults the NPU on purpose)
            add_executable(${t} tests/${t}.c)
            target_link_libraries(${t} PRIVATE rocketnpu)
            target_include_directories(${t} PRIVATE ${DRM_INCLUDE_DIRS})
            target_compile_options(${t} PRIVATE -Wall -Wextra)
        endforeach()
    endif()

    # the network gate is the graph planner's first caller: it owns the tensors and the
    # weights, graph/ owns which of them stay in cube layout and which runs are one kick
    if(TARGET rk3576_net_gate)
        target_link_libraries(rk3576_net_gate PRIVATE rocketgraph)
    endif()

    # the shared-scratch prototype + the cross-op chain probe drive the internal
    # tiled-matmul primitives directly; the LUT and elementwise probes run the library's
    # own wide-output poison guard (rocket_rk3576_power_idle) around their submits, both
    # being DPU-only programs and so victims of it
    foreach(t prototype_shared_scratch_rocket crossop_chain_rocket rk3576_cbuf_base
              rk3576_lut_probe rk3576_add_probe rk3576_mm_requant rk3576_w8a8_route)
        if(TARGET ${t})
            target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        endif()
    endforeach()

    # the RK3576 regcmd gate compiles the decoded vendor register programs in
    # (tests/data/rk3576-vendor-capture/rk3576_vendor_golden.h)
    foreach(t regcmd_rk3576_gate rk3576_fc_vendor)
        if(TARGET ${t})
            target_include_directories(${t} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/rk3576-vendor-capture)
        endif()
    endforeach()

    # the int8 depthwise ground-truth gates load the vendored Teflon capture fixtures
    # (tests/data/teflon-dw-capture); bake the absolute path at build time so the gate
    # runs from any working directory. Override at runtime with argv[1].
    foreach(t replay_dw_mesa conv_dw_int8_runtime)
        if(TARGET ${t})
            target_compile_definitions(${t} PRIVATE
                TEFLON_DW_CAPTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/teflon-dw-capture")
        endif()
    endforeach()

    # Register the correctness gates with CTest (`ctest` after a build). The conv2d
    # tests do real work even on x86 (cube-layout self-check) and additionally gate HW
    # when an NPU is present; the matmul gates need an NPU and exit with their skip code
    # when absent. The other built binaries are perf probes / sweeps run by hand, not
    # gates, so they are intentionally left unregistered.
    enable_testing()
    add_test(NAME activation_lut_rocket COMMAND activation_lut_rocket)
    set_tests_properties(activation_lut_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME leaky_relu_rocket COMMAND leaky_relu_rocket)
    set_tests_properties(leaky_relu_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME recip_rsqrt_rocket COMMAND recip_rsqrt_rocket)
    set_tests_properties(recip_rsqrt_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME lut_tanh_rocket COMMAND lut_tanh_rocket)
    set_tests_properties(lut_tanh_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME exp_lut_rocket COMMAND exp_lut_rocket)
    set_tests_properties(exp_lut_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME softplus_mish_rocket COMMAND softplus_mish_rocket)
    set_tests_properties(softplus_mish_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME elu_rocket COMMAND elu_rocket)
    set_tests_properties(elu_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME ew_minmax_rocket COMMAND ew_minmax_rocket)
    set_tests_properties(ew_minmax_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME prelu_rocket COMMAND prelu_rocket)
    set_tests_properties(prelu_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME ew_mul_rocket COMMAND ew_mul_rocket)
    set_tests_properties(ew_mul_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME conv_act_rocket COMMAND conv_act_rocket)
    set_tests_properties(conv_act_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME gelu_rocket COMMAND gelu_rocket)
    set_tests_properties(gelu_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME bytes_moved_rocket COMMAND bytes_moved_rocket 512 3840 4096)
    set_tests_properties(bytes_moved_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME conv2d_fp16_rocket COMMAND conv2d_fp16_rocket)
    add_test(NAME chain_layout_rocket COMMAND chain_layout_rocket)  # host-only; no NPU, never skips
    add_test(NAME chain_verify_gate COMMAND chain_verify_gate)      # host-only; no NPU, never skips
    add_test(NAME matmul_plan_gate COMMAND matmul_plan_gate)        # host-only; no NPU, skips off-RK3588
    set_tests_properties(matmul_plan_gate PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME scratch_cache_gate COMMAND scratch_cache_gate)
    set_tests_properties(scratch_cache_gate PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME regcmd_rk3576_gate COMMAND regcmd_rk3576_gate)    # host-only; no NPU, never skips
    # the RK3576 conv + matmul correctness gates. Both skip on any other chip (and with
    # no NPU), so they are inert in an RK3588 ctest run. The conv gate runs `all` — every
    # group including depthwise, which computes bit-exactly on this part.
    add_test(NAME rk3576_conv_gate COMMAND rk3576_conv_gate all)
    set_tests_properties(rk3576_conv_gate PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME rk3576_conv_lib_gate COMMAND rk3576_conv_lib_gate)
    set_tests_properties(rk3576_conv_lib_gate PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME rk3576_argb_ic1 COMMAND rk3576_argb_ic1)
    set_tests_properties(rk3576_argb_ic1 PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME rk3576_matmul_gate COMMAND rk3576_matmul_gate)
    set_tests_properties(rk3576_matmul_gate PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME rk3576_perchannel_gate COMMAND rk3576_perchannel_gate)
    set_tests_properties(rk3576_perchannel_gate PROPERTIES SKIP_RETURN_CODE 2)
    # The PPU. `gate` and not `sweep`: the destination base is settled, and the sweep
    # is the probe that settled it rather than a check on anything.
    add_test(NAME rk3576_pool_probe COMMAND rk3576_pool_probe gate)
    set_tests_properties(rk3576_pool_probe PROPERTIES SKIP_RETURN_CODE 2)
    # A pool writing a SLICE of a shared concatenation buffer, whose channel-group stride is
    # the plane where the pool's own surface is round4(ow*oh). Asserts the arithmetic, that
    # nothing outside the slice moved, and that a stride below the plane is still refused.
    add_test(NAME rk3576_pool_place COMMAND rk3576_pool_probe place)
    set_tests_properties(rk3576_pool_place PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME rk3576_add_probe COMMAND rk3576_add_probe gate)
    set_tests_properties(rk3576_add_probe PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME rk3576_residual_add COMMAND rk3576_residual_add gate)
    set_tests_properties(rk3576_residual_add PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME rk3576_lut_probe COMMAND rk3576_lut_probe gate)
    set_tests_properties(rk3576_lut_probe PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME rk3576_act_gate COMMAND rk3576_act_gate gate)
    set_tests_properties(rk3576_act_gate PROPERTIES SKIP_RETURN_CODE 2)
    # Two producers writing slices of one buffer, against the row-major path they replace.
    add_test(NAME rk3576_offset_cube COMMAND rk3576_offset_cube gate)
    set_tests_properties(rk3576_offset_cube PROPERTIES SKIP_RETURN_CODE 2)
    # What a producer leaves past its live output channels, which is what an unaligned join
    # reads as its border constant.
    add_test(NAME rk3576_pad_channels COMMAND rk3576_pad_channels)
    set_tests_properties(rk3576_pad_channels PROPERTIES SKIP_RETURN_CODE 2)
    # A feature cube whose channel groups sit further apart than the plane, which is what a
    # depthwise producer's surface is.
    add_test(NAME rk3576_surf_stride COMMAND rk3576_surf_stride)
    set_tests_properties(rk3576_surf_stride PROPERTIES SKIP_RETURN_CODE 2)
    # And a feature plane whose ROWS sit further apart than the plane is wide — the same
    # question one register down, asked of the CNA rather than of the PPU.
    add_test(NAME rk3576_conv_pitch COMMAND rk3576_conv_pitch)
    set_tests_properties(rk3576_conv_pitch PROPERTIES SKIP_RETURN_CODE 2)
    # A whole quantized classifier, layer by layer against TFLite's own intermediates.
    # It skips without the model blob, which is not committed — build it once with
    # tests/data/rk3576-net/fetch.sh + mknet.py. ROCKET_SRC_DIR is how it finds the blob
    # from a build directory.
    add_test(NAME rk3576_net_gate COMMAND rk3576_net_gate all)
    set_tests_properties(rk3576_net_gate PROPERTIES SKIP_RETURN_CODE 2
        ENVIRONMENT "ROCKET_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME rk3576_coeff_c COMMAND rk3576_coeff_c all)
    set_tests_properties(rk3576_coeff_c PROPERTIES SKIP_RETURN_CODE 2)
    # A datapath with no encoder for the running part must fail loudly rather than emit
    # the RK3588 program, which that part completes without writing anything.
    add_test(NAME rk3576_refusal_gate COMMAND rk3576_refusal_gate)
    set_tests_properties(rk3576_refusal_gate PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME conv1d_rocket COMMAND conv1d_rocket)
    set_tests_properties(conv1d_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME conv_transpose_rocket COMMAND conv_transpose_rocket)
    set_tests_properties(conv_transpose_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME resize_rocket COMMAND resize_rocket)
    set_tests_properties(resize_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME pool_fp16_rocket COMMAND pool_fp16_rocket)
    add_test(NAME pool_int8_rocket COMMAND pool_int8_rocket)
    add_test(NAME reduce_mean_rocket COMMAND reduce_mean_rocket)
    add_test(NAME reduce_feature_rocket COMMAND reduce_feature_rocket)
    set_tests_properties(reduce_feature_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME cumsum_rocket COMMAND cumsum_rocket)
    set_tests_properties(cumsum_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME rmsnorm_rocket COMMAND rmsnorm_rocket)
    set_tests_properties(rmsnorm_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME norm_vision_rocket COMMAND norm_vision_rocket)
    set_tests_properties(norm_vision_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME ffn_rocket COMMAND ffn_rocket)
    set_tests_properties(ffn_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME ffn_fused_rocket COMMAND ffn_fused_rocket)
    set_tests_properties(ffn_fused_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME softmax_rocket COMMAND softmax_rocket)
    set_tests_properties(softmax_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME cross_entropy_rocket COMMAND cross_entropy_rocket)
    set_tests_properties(cross_entropy_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME layernorm_rocket COMMAND layernorm_rocket)
    set_tests_properties(layernorm_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME mha_rocket COMMAND mha_rocket)
    set_tests_properties(mha_rocket PROPERTIES SKIP_RETURN_CODE 2)
    # Default invocation: per-worker QK/AV submit chaining is ON by default (the persistent
    # batched-context path), so this gates the chained path across every FA shape (cos=1.0).
    # Uses the gapped one-job layout (no kernel param); the fully-chained one-IRQ layout adds
    # ROCKET_BATCH_SUBMIT=1 + the kernel half, exercised on-device by hand.
    add_test(NAME flash_attn_rocket COMMAND flash_attn_rocket)
    set_tests_properties(flash_attn_rocket PROPERTIES SKIP_RETURN_CODE 2)
    # The same gate with chaining OFF, to keep the per-head (one-submit-per-head) path covered.
    add_test(NAME flash_attn_rocket_nochain COMMAND flash_attn_rocket)
    set_tests_properties(flash_attn_rocket_nochain PROPERTIES
        SKIP_RETURN_CODE 2 ENVIRONMENT "ROCKET_FA_CHAIN=0")
    add_test(NAME encoder_block_rocket COMMAND encoder_block_rocket)
    set_tests_properties(encoder_block_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME siglip_rocket COMMAND siglip_rocket)
    set_tests_properties(siglip_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME sam_rocket COMMAND sam_rocket)
    set_tests_properties(sam_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME conv2d_int8_rocket COMMAND conv2d_int8_rocket)
    add_test(NAME matmul_tiled_rocket COMMAND matmul_tiled_rocket)
    add_test(NAME fc_data_bank_sweep_rocket COMMAND fc_data_bank_sweep_rocket)
    set_tests_properties(fc_data_bank_sweep_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME matmul_kacc_chain_rocket COMMAND matmul_kacc_chain_rocket)
    set_tests_properties(matmul_kacc_chain_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME matmul_mt_rocket    COMMAND matmul_mt_rocket)
    add_test(NAME matmul_batch_rocket COMMAND matmul_batch_rocket)
    set_tests_properties(matmul_batch_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME matmul_fp32out_rocket COMMAND matmul_fp32out_rocket 64 4096 512)
    set_tests_properties(matmul_fp32out_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME matmul_int8_dequant_rocket COMMAND matmul_int8_dequant_rocket)
    set_tests_properties(matmul_int8_dequant_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME matmul_prepacked_crossm_rocket COMMAND matmul_prepacked_crossm_rocket)
    set_tests_properties(matmul_prepacked_crossm_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME matmul_correctness_matrix_rocket COMMAND matmul_correctness_matrix_rocket 512 1024 1024)
    set_tests_properties(matmul_tiled_rocket matmul_mt_rocket PROPERTIES SKIP_RETURN_CODE 2)
    set_tests_properties(matmul_correctness_matrix_rocket PROPERTIES SKIP_RETURN_CODE 3)
    # asymmetric-Nt tiling (ROCKET_MM_ASYM, default ON): gate its bit-exactness on a shape where
    # it fires (N=4096>cap N-tiled, K=4096 K-tiled => Nt 256->128, Kt 384->512). The base
    # correctness_matrix above (1024x1024) also fires it now that the default is ON; this pins the
    # firing shape explicitly, and the _sym gate below keeps the symmetric plan (=0) covered too.
    add_test(NAME matmul_correctness_matrix_asym COMMAND matmul_correctness_matrix_rocket 512 4096 4096)
    set_tests_properties(matmul_correctness_matrix_asym PROPERTIES SKIP_RETURN_CODE 3 ENVIRONMENT "ROCKET_MM_ASYM=1")
    add_test(NAME matmul_correctness_matrix_sym COMMAND matmul_correctness_matrix_rocket 512 4096 4096)
    set_tests_properties(matmul_correctness_matrix_sym PROPERTIES SKIP_RETURN_CODE 3 ENVIRONMENT "ROCKET_MM_ASYM=0")
    add_test(NAME crossop_chain_rocket COMMAND crossop_chain_rocket)
    set_tests_properties(crossop_chain_rocket PROPERTIES SKIP_RETURN_CODE 2)

    # Bit-exact dtype + depthwise gates — the int8/int4/int16/bf16/tf32 tiled-or-resident
    # matmul paths and the int8 depthwise conv the READMEs market as bit-exact. Each exits
    # with code 2 when no NPU is present (and conv_dw_int8_runtime also when its host-packing
    # fixtures are absent), so `ctest` reports SKIP off-device and FAIL only on a real
    # on-device regression. The bf16/tf32 tiled and int16-exact gates require an explicit
    # <M K N> (their mains have no default shape); the rest use built-in defaults.
    add_test(NAME matmul_int8_tiled_rocket     COMMAND matmul_int8_tiled_rocket)
    add_test(NAME mixed_chain_coexist_rocket   COMMAND mixed_chain_coexist_rocket)
    add_test(NAME matmul_int4_tiled_rocket     COMMAND matmul_int4_tiled_rocket)
    add_test(NAME matmul_int4_groupwise_rocket COMMAND matmul_int4_groupwise_rocket)
    add_test(NAME matmul_bf16_tiled_rocket     COMMAND matmul_bf16_tiled_rocket 512 3840 4096)
    add_test(NAME matmul_bf16_stream_rocket    COMMAND matmul_bf16_stream_rocket 512 3840 4096 4)
    add_test(NAME matmul_tf32_tiled_rocket     COMMAND matmul_tf32_tiled_rocket 512 3840 4096)
    add_test(NAME matmul_int8_prepacked_rocket COMMAND matmul_int8_prepacked_rocket)
    add_test(NAME matmul_int8_crossm_rocket    COMMAND matmul_int8_crossm_rocket)
    add_test(NAME matmul_int8_groupwise_rocket COMMAND matmul_int8_groupwise_rocket)
    add_test(NAME matmul_int8_prepacked_gw_rocket COMMAND matmul_int8_prepacked_gw_rocket)
    # group=576 (the default above) has Kt == group, so it never exercises the resident
    # path's kt_per_group > 1 branch — the one that maps several K-tiles onto ONE quant
    # group's scale. group=1440 forces Kt=480 (3 tiles per group) and gates that mapping.
    add_test(NAME matmul_int8_prepacked_gw_wide_rocket
             COMMAND matmul_int8_prepacked_gw_rocket 256 2880 512 1440 2)
    # The default N=512 splits into 128-wide slices — BELOW the 256 tile cap, so every N-tile
    # already fits its slice exactly and the tile-padding path is never taken. The real MoE
    # expert shape (N=2880 over 5 worker fds -> a 576-wide slice, the first width that exceeds
    # the cap) is what exercises i8_pick_nt, and it is also the shape whose bit-exactness the
    # fix must not disturb: shrinking Nt frees CBUF banks, and if that let Kt grow the resident
    # path would re-tile K and stop matching the one-shot oracle. Both are gated here.
    add_test(NAME matmul_int8_prepacked_gw_moe_rocket
             COMMAND matmul_int8_prepacked_gw_rocket 256 2880 2880 576 2)
    add_test(NAME matmul_int8_crossm_gw_rocket COMMAND matmul_int8_crossm_gw_rocket)
    add_test(NAME matmul_int4_prepacked_rocket COMMAND matmul_int4_prepacked_rocket)
    add_test(NAME matmul_int4_prepacked_gw_rocket COMMAND matmul_int4_prepacked_gw_rocket)
    add_test(NAME matmul_int4_crossm_rocket    COMMAND matmul_int4_crossm_rocket)
    add_test(NAME matmul_int16_exact_rocket    COMMAND matmul_int16_exact_rocket 512 3840 4096)
    add_test(NAME replay_dw_mesa               COMMAND replay_dw_mesa)
    add_test(NAME conv_dw_int8_runtime         COMMAND conv_dw_int8_runtime)
    set_tests_properties(
        matmul_int8_tiled_rocket matmul_int4_tiled_rocket matmul_int4_groupwise_rocket
        matmul_bf16_tiled_rocket matmul_bf16_stream_rocket matmul_tf32_tiled_rocket
        matmul_int8_prepacked_rocket matmul_int8_crossm_rocket
        matmul_int8_groupwise_rocket matmul_int8_prepacked_gw_rocket
        matmul_int8_prepacked_gw_wide_rocket matmul_int8_prepacked_gw_moe_rocket
        matmul_int8_crossm_gw_rocket
        matmul_int4_prepacked_rocket matmul_int4_prepacked_gw_rocket
        matmul_int4_crossm_rocket
        matmul_int16_exact_rocket replay_dw_mesa conv_dw_int8_runtime
        mixed_chain_coexist_rocket
        PROPERTIES SKIP_RETURN_CODE 2)

    # The mainline-uAPI gates only exist in a builtin-provider build (see the foreach
    # above); registering them otherwise would name targets that were never created.
    if(ROCKETNPU_PROVIDER STREQUAL "builtin")
    # uAPI conformance gate: runtime-verifies the drm_rocket_* contracts the
    # shim depends on (BO/regcmd-IOVA window, PREP_BO absolute-deadline semantics).
    # The cross-kernel canary — a kernel that drifts fails here, not deep in matmul.
    add_test(NAME uapi_selftest_rocket COMMAND uapi_selftest_rocket)
    set_tests_properties(uapi_selftest_rocket PROPERTIES SKIP_RETURN_CODE 2)
    # Every rejection site in rocket_ioctl_submit_job() is reached BEFORE the job's
    # IOMMU domain reference is taken, and rocket_job_cleanup() puts that reference
    # unconditionally — so a malformed submit from any client holding /dev/accel/accel0
    # (group `render`) oopses the kernel. Each case runs in a forked child, so the
    # gate reports which site did it instead of dying with it, and stops at the first.
    add_test(NAME uapi_submit_errpath_rocket COMMAND uapi_submit_errpath_rocket)
    set_tests_properties(uapi_submit_errpath_rocket PROPERTIES SKIP_RETURN_CODE 2)
    add_test(NAME uapi_bo_ranges_rocket COMMAND uapi_bo_ranges_rocket)
    set_tests_properties(uapi_bo_ranges_rocket PROPERTIES SKIP_RETURN_CODE 2)
    endif()
    # The per-context IOVA allocator is freed in rocket_postclose(), but a BO's IOVA
    # node is removed in its GEM free path — which the drm_sched free worker can run
    # after that close. Submitting and closing without waiting races the two on
    # purpose; the verdict is the kernel log, since the defect fails no syscall.
    add_test(NAME uapi_bo_lifetime_rocket COMMAND uapi_bo_lifetime_rocket 64)
    set_tests_properties(uapi_bo_lifetime_rocket PROPERTIES SKIP_RETURN_CODE 2)
    # PREP_BO wait robustness: a SIGUSR1 storm (handler without SA_RESTART) must not turn the
    # interruptible kernel wait into a spurious timeout (EINTR retry), and a UINT64_MAX
    # "wait forever" timeout must saturate rather than wrap the signed deadline negative into an
    # instant poll. 60 iters/phase keeps the gate ~1s; exit 2 (SKIP) when no NPU is present.
    add_test(NAME prep_signal_robust_rocket COMMAND prep_signal_robust_rocket)
    set_tests_properties(prep_signal_robust_rocket PROPERTIES SKIP_RETURN_CODE 2 ENVIRONMENT "ROCKET_SIG_ITERS=60")
endif()
