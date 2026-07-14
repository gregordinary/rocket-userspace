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

find_package(PkgConfig REQUIRED)
pkg_check_modules(DRM REQUIRED libdrm)        # <libdrm/drm.h>
find_package(Threads REQUIRED)                # rocket_matmul_mt.c

# libdrm only provides <libdrm/drm.h>; the actual build dependency is the mainline
# rocket DRM-accel uAPI header <drm/rocket_accel.h> (the DRM_IOCTL_ROCKET_* / struct
# drm_rocket_* definitions), which ships as include/uapi/drm/rocket_accel.h and is
# usually installed to /usr/include/drm/rocket_accel.h. Check it at configure time so a
# missing uAPI header is a clear message, not a confusing mid-compile error.
include(CheckIncludeFile)
set(CMAKE_REQUIRED_INCLUDES ${DRM_INCLUDE_DIRS})
check_include_file("drm/rocket_accel.h" HAVE_DRM_ROCKET_ACCEL_H)
unset(CMAKE_REQUIRED_INCLUDES)
if(NOT HAVE_DRM_ROCKET_ACCEL_H)
    message(FATAL_ERROR
        "drm/rocket_accel.h not found. Install the mainline rocket DRM-accel uAPI header "
        "(usually /usr/include/drm/rocket_accel.h, from include/uapi/drm/rocket_accel.h in "
        "the kernel source) before building rocket-userspace.")
endif()

# --- the library: shim + validated fp16 generator + tiled + multicore matmul ---
set(ROCKETNPU_SOURCES
    src/rocket_log.c          # centralized log channel (levels + host-settable callback)
    src/rocket_npu.c          # DRM-accel shim over /dev/accel/accel0
    src/rocket_hw_profile.c   # the chip machine-parameter profile (RK3588 today)
    src/npu_regcmd.c     # regcmd generators (fp16/int8/int4/int16/bf16/tf32)
    src/rocket_chain.c        # contiguous self-chaining regcmd layout (batched submit, shared)
    src/rocket_matmul.c       # tiled single-fd matmul (CBUF-fit, CPU K-accum)
    src/rocket_conv.c         # general fp16 CONV_2D (KxK/stride/pad/dilation)
    src/rocket_conv_transpose.c # ConvTranspose2d (deconv) lowered onto the forward conv
    src/rocket_resize.c       # nearest/bilinear upsample via depthwise ConvTranspose
    src/rocket_pool.c         # on-NPU MaxPool / AveragePool via the PPU
    src/rocket_reduce.c       # on-NPU GlobalAvgPool / Mean (multi-pass PPU reduction) + feature-axis reduce
    src/rocket_norm.c         # on-NPU RMSNorm + per-row broadcast scale
    src/rocket_normvision.c   # on-NPU BatchNorm/GroupNorm/InstanceNorm/L2Norm (vision normalization)
    src/rocket_ffn.c          # on-NPU gated-MLP FFN block (GeGLU/SwiGLU core + projections)
    src/rocket_softmax.c      # on-NPU row-wise softmax (EXP LUT + row-sum reduce + per-row scale)
    src/rocket_attn.c         # on-NPU multi-head self-attention (QKV + scores + softmax + P·V + out-proj)
    src/rocket_encoder.c      # one Whisper encoder block (LN -> MHA -> residual -> LN -> MLP -> residual)
    src/rocket_siglip_encoder.c # SigLIP-B/16 vision encoder (patch-embed + pos + 12x block + post-LN)
    src/rocket_activation.c   # elementwise fp16 activation via the DPU LUT block
    src/rocket_matmul_mt.c    # multicore fan-out (per-core worker fds)
    src/rocket_bf16_stream.c  # fast bf16 matmul: multicore fan-out + streaming resident scratch
    src/rocket_prepacked.c    # pack-weights-once path (persistent fds + resident BOs)
    src/rocket_prepacked_int8.c # resident int8 (W8A8) weight path
    src/rocket_prepacked_int4.c # resident int4 (W4A4) weight path
    src/rocket_affinity.c)    # pin pack/readback workers to the big (A76) cores

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
            uapi_selftest_rocket   # rocket uAPI conformance gate (BO/deadline/IOVA contracts)
            submit_overhead_rocket # per-job dispatch floor (us/submit) A/B probe for IOMMU keep-attach
            x0_glitch_probe        # RE: does the x≈0 LE/LO mux glitch fire for a shifted table straddling 0?
            regcmd_persist_rocket  # does register state persist across tasks in a job (delta regcmd)?
            # host-only regcmd dumpers + a standalone fp16 test + a DRAM microbench: tools, not CTest gates
            matmul_fp16_rocket     # standalone fp16 x fp16 single-task test (gen_matmul_fp16 path)
            dump_regcmd            # host-only: diff the matmul regcmd accumulate=0 vs 1 (no HW)
            dump_dw_regcmd         # host-only: emit depthwise-conv regcmd for the Mesa decoder
            chain_layout_rocket    # host-only GATE: batched-submit contiguous layout + trailer link (no HW)
            membench               # DRAM bandwidth + NPU readback de-tile microbench
            fabench)               # flash-attention QK/AV submit-chaining A/B (ROCKET_FA_CHAIN 0 vs 1)
        add_executable(${t} tests/${t}.c)
        target_link_libraries(${t} PRIVATE rocketnpu)
        target_compile_options(${t} PRIVATE -Wall -Wextra)
    endforeach()
    # the shared-scratch prototype + the cross-op chain probe drive the internal
    # tiled-matmul primitives directly
    foreach(t prototype_shared_scratch_rocket crossop_chain_rocket)
        if(TARGET ${t})
            target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
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
        matmul_int8_prepacked_gw_wide_rocket matmul_int8_crossm_gw_rocket
        matmul_int4_prepacked_rocket matmul_int4_prepacked_gw_rocket
        matmul_int4_crossm_rocket
        matmul_int16_exact_rocket replay_dw_mesa conv_dw_int8_runtime
        mixed_chain_coexist_rocket
        PROPERTIES SKIP_RETURN_CODE 2)

    # uAPI conformance gate: runtime-verifies the drm_rocket_* contracts the
    # shim depends on (BO/regcmd-IOVA window, PREP_BO absolute-deadline semantics).
    # The cross-kernel canary — a kernel that drifts fails here, not deep in matmul.
    add_test(NAME uapi_selftest_rocket COMMAND uapi_selftest_rocket)
    set_tests_properties(uapi_selftest_rocket PROPERTIES SKIP_RETURN_CODE 2)
endif()
