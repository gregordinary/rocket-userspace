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

# What rocket_driver_name() reports, for a device listing or a log line. The library can
# name its own builtin driver and cannot name someone else's, so an external build says
# only what is known unless its packager names it -- rknpu-submit's README passes
# "vendor rknpu driver" here. A frontend prints this instead of a literal, so a BSP board
# is never told the mainline path is live.
set(ROCKETNPU_DRIVER_NAME "" CACHE STRING
    "Kernel driver this build reaches, as a short human name (default: per provider)")
if(NOT ROCKETNPU_DRIVER_NAME)
    if(ROCKETNPU_PROVIDER STREQUAL "builtin")
        set(ROCKETNPU_DRIVER_NAME "mainline rocket driver")
    else()
        set(ROCKETNPU_DRIVER_NAME "external submit provider")
    endif()
endif()
message(STATUS "rocketnpu: driver name = ${ROCKETNPU_DRIVER_NAME}")

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
    src/rocket_fence_watch.c  # the fenced-wait counters (core side of the seam)
    src/rocket_hw_profile.c   # the chip machine-parameter profile (RK3588 today)
    src/rocket_driver_name.c  # the build's kernel-driver name (not seam: no provider defines it)
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
target_compile_definitions(rocketnpu_obj PRIVATE ROCKETNPU_DRIVER_NAME="${ROCKETNPU_DRIVER_NAME}")

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
            int8_chain_probe       # probe: does a chained batch of int8 matmul tasks compute? (GA-5.3)
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
            deconv_geometry_probe  # RK3588 board: WHERE the DECONV mode scatters, via an impulse response
            deconv_arith_probe     # RK3588 board: does DECONV compute a real transposed conv (dense, vs oracle)?
            deconv_pad_probe       # RK3588 board: does DECONV take a pad, and where does its result land?
            deconv_extent_probe    # RK3588 board: does DECONV compute the WHOLE transposed conv given its extent?
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
            replay_dw_mesa         # our int8 DW register program over Teflon's captured BOs vs mesa-output
            conv_dw_int8_runtime   # int8 DW int8-out RUNTIME vs TFLite's reference kernels + an exact host model
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
            iova_rcache_rocket     # does freeing a BO return its IOVA? (rcache withholding; hand-run, not a gate)
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
            regcmd_persist_rocket  # the one-task-after delta arm (lands in an unwritten group); see regcmd_delta_probe
            regcmd_delta_probe     # probe: a delta task in its own register group, three pointer modes (GA-5.5)
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
            rk3576_mm_fp16_gate    # RK3576 GATE: the matmul-form fp16 program vs charsiu (host) and a CPU reference (board)
            rk3576_mm_fp16_cost    # RK3576 board: what the fp16 matmul costs per call, against the int8 entry
            rk3576_mm_fp16_envelope # RK3576 board: where one task of the fp16 matmul program stops being exact
            rk3576_mm_fp16_round   # RK3576 board: how the fp16 matmul rounds an fp32 sum that is not exact
            rk3576_mm_bf16_probe   # RK3576 board: does the matmul form contract bf16 operands?
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
            pool_int8_native_probe # board: does the PPU pool natively in int8? (the precision PAIR, as a map)
            hostload               # host memory-system load generator: run any gate under DDR / L1 / spin pressure
            membench               # DRAM bandwidth + NPU readback de-tile microbench
            ddr_pmu_cal            # positive control for the rockchip_ddr DDR PMU: known bytes in, counted bytes out
            ddr_mm_bytes           # the matmul arm the DDR PMU measures against bytes_moved_rocket
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
    # being DPU-only programs and so victims of it; the bf16 stream gate reads the fan-out's
    # N split (rocket_fanout_nstep) to decide when its multicore paths must be bit-exact;
    # the fp16 envelope probe power-cycles before its second run of a shape that was not exact
    foreach(t prototype_shared_scratch_rocket crossop_chain_rocket rk3576_cbuf_base
              rk3576_lut_probe rk3576_add_probe rk3576_mm_requant rk3576_w8a8_route
              pool_int8_native_probe deconv_extent_probe matmul_bf16_stream_rocket
              replay_dump rk3576_mm_fp16_envelope rk3576_mm_bf16_probe)
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

    # Register the correctness gates with CTest (`ctest` after a build). The other built
    # binaries are perf probes / sweeps run by hand, not gates, so they are intentionally
    # left unregistered.
    #
    # EVERY TEST HOLDS ONE EXIT-CODE CONTRACT: 0 when a numeric check ran and passed, 2
    # when the device (or the part the test needs) is absent, and 1 for everything else --
    # a wrong answer, a declined path, a usage error. A declined path that exits 0 reads
    # as PASS here, which is how a test once passed three registrations over no check.
    enable_testing()

    # The kernel-log bracket. The `rocket` driver retires a job that runs 500 ms and
    # signals its fence as if it had completed, so a hung job fails no syscall: the only
    # traces are its duration in this process and an "NPU job timed out" line in the
    # kernel log. The mark saves a journal cursor before the first NPU test and the check
    # fails the run on any timeout, rk_iommu line or NPU-side WARNING logged after it.
    # With an unreadable journal the mark still succeeds (so the NPU tests run) and the
    # check reports SKIP.
    add_test(NAME npu_klog_mark
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/npu_klog_fixture.sh
                     mark ${CMAKE_CURRENT_BINARY_DIR}/npu_klog.cursor)
    set_tests_properties(npu_klog_mark PROPERTIES FIXTURES_SETUP npu_klog TIMEOUT 60)
    add_test(NAME npu_klog_check
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/npu_klog_fixture.sh
                     check ${CMAKE_CURRENT_BINARY_DIR}/npu_klog.cursor)
    set_tests_properties(npu_klog_check PROPERTIES
        FIXTURES_CLEANUP npu_klog SKIP_RETURN_CODE 2 TIMEOUT 60)

    # A test that submits to the NPU.
    #   RESOURCE_LOCK npu   one at a time under `ctest -j`: two `rocket` jobs in flight
    #                       compute wrong answers on the RK3576, across processes.
    #   TIMEOUT             a child stuck in D state otherwise holds the suite for 1500 s.
    #   FAIL_REGULAR_EXPRESSION
    #                       a fenced wait past the slow-wait mark (src/rocket_fence_watch.c)
    #                       fails the test that made it, whatever its own verdict was.
    function(rocket_npu_test name)
        cmake_parse_arguments(T "" "TIMEOUT" "COMMAND;ENVIRONMENT" ${ARGN})
        if(NOT T_COMMAND)
            set(T_COMMAND ${name})
        endif()
        if(NOT T_TIMEOUT)
            set(T_TIMEOUT 300)
        endif()
        add_test(NAME ${name} COMMAND ${T_COMMAND})
        set_tests_properties(${name} PROPERTIES
            SKIP_RETURN_CODE 2
            RESOURCE_LOCK npu
            TIMEOUT ${T_TIMEOUT}
            FIXTURES_REQUIRED npu_klog
            FAIL_REGULAR_EXPRESSION "past the [0-9]+ ms slow-wait mark")
        if(T_ENVIRONMENT)
            set_tests_properties(${name} PROPERTIES ENVIRONMENT "${T_ENVIRONMENT}")
        endif()
    endfunction()

    # A test that never opens the device. Pure arithmetic cannot skip, so it has no skip
    # code unless SKIP_OFF_PART says it checks one part's planner and skips on another.
    function(rocket_host_test name)
        cmake_parse_arguments(T "SKIP_OFF_PART" "" "COMMAND" ${ARGN})
        if(NOT T_COMMAND)
            set(T_COMMAND ${name})
        endif()
        add_test(NAME ${name} COMMAND ${T_COMMAND})
        set_tests_properties(${name} PROPERTIES TIMEOUT 300)
        if(T_SKIP_OFF_PART)
            set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 2)
        endif()
    endfunction()

    # --- host-only ---------------------------------------------------------------
    rocket_host_test(bytes_moved_rocket COMMAND bytes_moved_rocket 512 3840 4096)
    rocket_host_test(chain_layout_rocket)
    rocket_host_test(chain_verify_gate)
    rocket_host_test(matmul_plan_gate SKIP_OFF_PART)    # the RK3588 planners; skips elsewhere
    rocket_host_test(regcmd_rk3576_gate)                # forces ROCKET_CHIP=rk3576 itself

    # --- the DPU LUT and elementwise ops -----------------------------------------
    rocket_npu_test(activation_lut_rocket)
    rocket_npu_test(leaky_relu_rocket)
    rocket_npu_test(recip_rsqrt_rocket)
    rocket_npu_test(lut_tanh_rocket)
    rocket_npu_test(exp_lut_rocket)
    rocket_npu_test(softplus_mish_rocket)
    rocket_npu_test(elu_rocket)
    rocket_npu_test(ew_minmax_rocket)
    rocket_npu_test(prelu_rocket)
    rocket_npu_test(ew_mul_rocket)
    rocket_npu_test(conv_act_rocket)
    rocket_npu_test(gelu_rocket)

    # The conv and pooling gates do real work even off-device (a cube-layout self-check),
    # but exit 2 when no device opened: the host checks passing is not the NPU passing.
    rocket_npu_test(conv2d_fp16_rocket)
    rocket_npu_test(scratch_cache_gate)

    # The RK3576 gates. Each skips on any other part (and with no NPU), so they are inert
    # in an RK3588 ctest run. The conv gate runs `all` -- every group including
    # depthwise, which computes bit-exactly on this part.
    rocket_npu_test(rk3576_conv_gate COMMAND rk3576_conv_gate all)
    rocket_npu_test(rk3576_conv_lib_gate)
    rocket_npu_test(rk3576_argb_ic1)
    rocket_npu_test(rk3576_matmul_gate)
    rocket_npu_test(rk3576_mm_fp16_gate)
    rocket_npu_test(rk3576_perchannel_gate)
    # The PPU. `gate` and not `sweep`: the destination base is settled, and the sweep
    # is the probe that settled it rather than a check on anything.
    rocket_npu_test(rk3576_pool_probe COMMAND rk3576_pool_probe gate)
    # A pool writing a SLICE of a shared concatenation buffer, whose channel-group stride is
    # the plane where the pool's own surface is round4(ow*oh). Asserts the arithmetic, that
    # nothing outside the slice moved, and that a stride below the plane is still refused.
    rocket_npu_test(rk3576_pool_place COMMAND rk3576_pool_probe place)
    rocket_npu_test(rk3576_add_probe COMMAND rk3576_add_probe gate)
    rocket_npu_test(rk3576_residual_add COMMAND rk3576_residual_add gate)
    rocket_npu_test(rk3576_lut_probe COMMAND rk3576_lut_probe gate)
    rocket_npu_test(rk3576_act_gate COMMAND rk3576_act_gate gate)
    # Two producers writing slices of one buffer, against the row-major path they replace.
    rocket_npu_test(rk3576_offset_cube COMMAND rk3576_offset_cube gate)
    # What a producer leaves past its live output channels, which is what an unaligned join
    # reads as its border constant.
    rocket_npu_test(rk3576_pad_channels)
    # A feature cube whose channel groups sit further apart than the plane, which is what a
    # depthwise producer's surface is.
    rocket_npu_test(rk3576_surf_stride)
    # And a feature plane whose ROWS sit further apart than the plane is wide -- the same
    # question one register down, asked of the CNA rather than of the PPU.
    rocket_npu_test(rk3576_conv_pitch)
    # A whole quantized classifier, layer by layer against TFLite's own intermediates.
    # It skips without the model blob, which is not committed -- build it once with
    # tests/data/rk3576-net/fetch.sh + mknet.py. ROCKET_SRC_DIR is how it finds the blob
    # from a build directory.
    rocket_npu_test(rk3576_net_gate COMMAND rk3576_net_gate all
                    ENVIRONMENT "ROCKET_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
    rocket_npu_test(rk3576_coeff_c COMMAND rk3576_coeff_c all)
    # A datapath with no encoder for the running part must fail loudly rather than emit
    # the RK3588 program, which that part completes without writing anything.
    rocket_npu_test(rk3576_refusal_gate)

    # --- conv, pooling, reduce, norms, attention ---------------------------------
    rocket_npu_test(conv1d_rocket)
    rocket_npu_test(conv_transpose_rocket)
    rocket_npu_test(resize_rocket)
    rocket_npu_test(pool_fp16_rocket)
    rocket_npu_test(pool_int8_rocket)
    rocket_npu_test(reduce_mean_rocket)
    rocket_npu_test(reduce_feature_rocket)
    rocket_npu_test(cumsum_rocket)
    rocket_npu_test(rmsnorm_rocket)
    rocket_npu_test(norm_vision_rocket)
    rocket_npu_test(ffn_rocket)
    rocket_npu_test(ffn_fused_rocket)
    rocket_npu_test(softmax_rocket)
    rocket_npu_test(cross_entropy_rocket)
    rocket_npu_test(layernorm_rocket)
    rocket_npu_test(mha_rocket)
    # Default invocation: per-worker QK/AV submit chaining is ON by default (the persistent
    # batched-context path), so this gates the chained path across every FA shape (cos=1.0).
    # Uses the gapped one-job layout (no kernel param); the fully-chained one-IRQ layout adds
    # ROCKET_BATCH_SUBMIT=1 + the kernel half, exercised on-device by hand. About 100 s.
    rocket_npu_test(flash_attn_rocket TIMEOUT 600)
    # The same gate with chaining OFF, to keep the per-head (one-submit-per-head) path covered.
    rocket_npu_test(flash_attn_rocket_nochain COMMAND flash_attn_rocket TIMEOUT 600
                    ENVIRONMENT "ROCKET_FA_CHAIN=0")
    rocket_npu_test(encoder_block_rocket)
    # The ViT encoders score against artifacts extracted on a host with the models
    # (tools/siglip_extract.py, tools/sam_extract.py), which are not committed. Name their
    # directories at configure time and the gates run, the resident path included, and
    # fail if the directory is empty. Leave them unset and the gates skip.
    set(ROCKETNPU_SIGLIP_ARTIFACTS "" CACHE PATH "SigLIP gate artifacts (siglip_weights.f16, pixels/hidden/postln.f32)")
    set(ROCKETNPU_SAM_ARTIFACTS "" CACHE PATH "SAM gate artifacts (sam_weights.f16, pixels/hidden/neck.f32)")
    if(ROCKETNPU_SIGLIP_ARTIFACTS)
        rocket_npu_test(siglip_rocket COMMAND siglip_rocket ${ROCKETNPU_SIGLIP_ARTIFACTS}
                        TIMEOUT 900 ENVIRONMENT "ROCKET_SIGLIP_BENCH=2")
    else()
        rocket_npu_test(siglip_rocket TIMEOUT 900)
    endif()
    if(ROCKETNPU_SAM_ARTIFACTS)
        rocket_npu_test(sam_rocket COMMAND sam_rocket ${ROCKETNPU_SAM_ARTIFACTS} TIMEOUT 900)
    else()
        rocket_npu_test(sam_rocket TIMEOUT 900)
    endif()
    rocket_npu_test(conv2d_int8_rocket)

    # --- the matmul family --------------------------------------------------------
    rocket_npu_test(matmul_tiled_rocket)
    rocket_npu_test(fc_data_bank_sweep_rocket)
    rocket_npu_test(matmul_kacc_chain_rocket)
    rocket_npu_test(matmul_mt_rocket)
    rocket_npu_test(matmul_batch_rocket)
    rocket_npu_test(matmul_fp32out_rocket COMMAND matmul_fp32out_rocket 64 4096 512)
    rocket_npu_test(matmul_int8_dequant_rocket)
    rocket_npu_test(matmul_prepacked_crossm_rocket)
    # Every row is verified (FULL_ROWS=1), and K != N in every registration: with K == N a
    # field that swapped the two would compute a plausible surface.
    rocket_npu_test(matmul_correctness_matrix_rocket
                    COMMAND matmul_correctness_matrix_rocket 512 1024 768
                    ENVIRONMENT "FULL_ROWS=1")
    # asymmetric-Nt tiling (ROCKET_MM_ASYM, default ON): gate its bit-exactness on a shape where
    # it fires (N=4096>cap N-tiled, K=3840 K-tiled => Nt 256->128, Kt 384->512, the last K-tile
    # partial). The base correctness_matrix above also fires it now that the default is ON; this
    # pins the firing shape explicitly, and the _sym gate below keeps the symmetric plan (=0)
    # covered too.
    rocket_npu_test(matmul_correctness_matrix_asym
                    COMMAND matmul_correctness_matrix_rocket 512 3840 4096
                    ENVIRONMENT "ROCKET_MM_ASYM=1;FULL_ROWS=1")
    rocket_npu_test(matmul_correctness_matrix_sym
                    COMMAND matmul_correctness_matrix_rocket 512 3840 4096
                    ENVIRONMENT "ROCKET_MM_ASYM=0;FULL_ROWS=1")
    rocket_npu_test(crossop_chain_rocket)

    # Bit-exact dtype + depthwise gates -- the int8/int4/int16/bf16/tf32 tiled-or-resident
    # matmul paths and the int8 depthwise conv. The bf16/tf32 tiled and int16-exact gates
    # require an explicit <M K N> (their mains have no default shape); the rest use built-in
    # defaults. conv_dw_int8_runtime fails when its committed fixtures are absent.
    rocket_npu_test(matmul_int8_tiled_rocket)
    rocket_npu_test(mixed_chain_coexist_rocket)
    rocket_npu_test(matmul_int4_tiled_rocket)
    rocket_npu_test(matmul_int4_groupwise_rocket)
    rocket_npu_test(matmul_bf16_tiled_rocket COMMAND matmul_bf16_tiled_rocket 512 3840 4096)
    rocket_npu_test(matmul_bf16_stream_rocket COMMAND matmul_bf16_stream_rocket 512 3840 4096 4)
    rocket_npu_test(matmul_tf32_tiled_rocket COMMAND matmul_tf32_tiled_rocket 512 3840 4096)
    rocket_npu_test(matmul_int8_prepacked_rocket)
    rocket_npu_test(matmul_int8_crossm_rocket)
    rocket_npu_test(matmul_int8_groupwise_rocket)
    rocket_npu_test(matmul_int8_prepacked_gw_rocket)
    # group=576 (the default above) has Kt == group, so it never exercises the resident
    # path's kt_per_group > 1 branch -- the one that maps several K-tiles onto ONE quant
    # group's scale. group=1440 forces Kt=480 (3 tiles per group) and gates that mapping.
    rocket_npu_test(matmul_int8_prepacked_gw_wide_rocket
                    COMMAND matmul_int8_prepacked_gw_rocket 256 2880 512 1440 2)
    # The default N=512 splits into 128-wide slices -- BELOW the 256 tile cap, so every N-tile
    # already fits its slice exactly and the tile-padding path is never taken. The real MoE
    # expert shape (N=2880 over 5 worker fds -> a 576-wide slice, the first width that exceeds
    # the cap) is what exercises i8_pick_nt, and it is also the shape whose bit-exactness the
    # fix must not disturb: shrinking Nt frees CBUF banks, and if that let Kt grow the resident
    # path would re-tile K and stop matching the one-shot oracle. Both are gated here.
    rocket_npu_test(matmul_int8_prepacked_gw_moe_rocket
                    COMMAND matmul_int8_prepacked_gw_rocket 256 2880 2880 576 2)
    rocket_npu_test(matmul_int8_crossm_gw_rocket)
    rocket_npu_test(matmul_int4_prepacked_rocket)
    rocket_npu_test(matmul_int4_prepacked_gw_rocket)
    rocket_npu_test(matmul_int4_crossm_rocket)
    rocket_npu_test(matmul_int16_exact_rocket COMMAND matmul_int16_exact_rocket 512 3840 4096)
    rocket_npu_test(replay_dw_mesa)
    rocket_npu_test(conv_dw_int8_runtime)
    # The OUT_CVT tie rule, which no scale a caller passes can reach: it builds its ties
    # on purpose and asserts half to even at bit 30 clear, half away from zero set.
    rocket_npu_test(requant_round_probe)

    # --- the uAPI -------------------------------------------------------------------
    # The mainline-uAPI gates only exist in a builtin-provider build (see the foreach
    # above); registering them otherwise would name targets that were never created.
    if(ROCKETNPU_PROVIDER STREQUAL "builtin")
        # uAPI conformance gate: runtime-verifies the drm_rocket_* contracts the
        # shim depends on (BO/regcmd-IOVA window, PREP_BO absolute-deadline semantics).
        # The cross-kernel canary -- a kernel that drifts fails here, not deep in matmul.
        rocket_npu_test(uapi_selftest_rocket)
        # Every rejection site in rocket_ioctl_submit_job() is reached BEFORE the job's
        # IOMMU domain reference is taken, and rocket_job_cleanup() puts that reference
        # unconditionally -- so a malformed submit from any client holding /dev/accel/accel0
        # (group `render`) oopses the kernel. Each case runs in a forked child, so the
        # gate reports which site did it instead of dying with it, and stops at the first.
        rocket_npu_test(uapi_submit_errpath_rocket)
        rocket_npu_test(uapi_bo_ranges_rocket)
    endif()
    # The per-context IOVA allocator is freed in rocket_postclose(), but a BO's IOVA
    # node is removed in its GEM free path -- which the drm_sched free worker can run
    # after that close. Submitting and closing without waiting races the two on
    # purpose; the verdict is the kernel log, since the defect fails no syscall.
    rocket_npu_test(uapi_bo_lifetime_rocket COMMAND uapi_bo_lifetime_rocket 64)
    # PREP_BO wait robustness: a SIGUSR1 storm (handler without SA_RESTART) must not turn the
    # interruptible kernel wait into a spurious timeout (EINTR retry), and a UINT64_MAX
    # "wait forever" timeout must saturate rather than wrap the signed deadline negative into an
    # instant poll. 60 iters/phase keeps the gate ~1s; exit 2 (SKIP) when no NPU is present.
    rocket_npu_test(prep_signal_robust_rocket ENVIRONMENT "ROCKET_SIG_ITERS=60")
endif()
