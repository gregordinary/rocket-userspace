// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * Declares the hardware profile: the NPU's machine parameters, which are fixed for a given
 * SoC and can differ on another RKNPU or NVDLA-class chip. They are the CBUF size, the
 * tile geometry, the usable datatype menu and the worker default. Collecting them in one
 * profile makes `librocketnpu` chip-agnostic by construction. The RK3588 is the fully
 * validated target, and the RK3576 is validated on the paths its own encoders cover.
 *
 * The profile contains these machine parameters:
 *
 * - The CBUF banks and bank size
 * - The matmul tile cap
 * - The tile-group sizes
 * - The datatype-availability mask
 * - The worker default
 *
 * The profile does not contain the datapath common to the IP family, which `npu_hw.h` and
 * `npu_regcmd.c` define. That datapath is the `precision_*` encoding values, the
 * CNA->CORE->DPU sequence, the BS, BN and EW field meanings, the LUT machinery and the
 * tile-layout index algebra. Those come from the IP rather than the SoC, so they are not
 * parameterized.
 *
 * The CNA, CORE and DPU geometry-register encoding is the one datapath element that is not
 * fully shared. The offsets and field packing in `npu_hw.h` are the RK3588's, and they are
 * specific to the IP revision. The RK3576 NPU re-packs the CNA block at the same block
 * base, and the difference is more than an offset shift. Registers move, the bit-packing
 * differs at shared offsets, and the RK3576 uses offsets that are gaps on the RK3588. Only
 * 2 CNA geometry registers coincide.
 *
 * The datapath semantics still transfer: the precision-field encodings, the CNA->CORE->DPU
 * block sequence, and the BS, BN, EW and LUT meanings. Multi-chip support therefore needs
 * a per-chip regcmd encoder for the geometry registers, not only an offset table. The
 * RK3576's encoders, in `npu_regcmd_rk3576.c`, cover int8 direct and depthwise
 * convolution, fp16 convolution and the packed-image first convolution. They also cover
 * the int8, int32 and fp16 matmuls, pooling and the LUT activations. The int8 convolution encoder is tested
 * against the vendor's own register programs (`tests/regcmd_rk3576_gate.c`).
 *
 * Two profiles exist. A chip has a profile only once its machine parameters are measured
 * on that silicon. On the RK3576, its encoders are bit-exact for int8 and fp16
 * convolutions, the int8, int32 and fp16 matmuls, pooling and the LUT activations. Its
 * CBUF capacity was swept against the int8 convolution. A profile does not claim full
 * support. The RK3576's covers only what has been measured, and its `select_warning` names
 * the datapaths that still have no encoder for it.
 *
 * `rocket_hw_current()` selects the profile. It reads the NPU's device-tree `compatible`,
 * which `ROCKET_CHIP` overrides. A `compatible` that matches no profile falls back to the
 * RK3588's with a warning, rather than silently applying RK3588 parameters to other
 * silicon. With no `compatible` to read, as off the device, the fallback logs only at debug
 * level.
 *
 * On a profiled SoC other than the RK3588, every RK3588 generator refuses
 * (`rk3588_encoding_only` in `npu_regcmd.c`), because its program would complete and write
 * nothing there. A SoC with no profile runs on the RK3588's, and the RK3588 generators do
 * not refuse it.
 */
#ifndef ROCKET_HW_PROFILE_H
#define ROCKET_HW_PROFILE_H

#include <stdint.h>
#include "npu_hw.h"   /* NPU_CBUF_BANKS and NPU_CBUF_BANK_SIZE, which the RK3588
                       * profile mirrors so that each compile-time literal is defined
                       * once, and the precision_* enum that keys the datatype mask. */

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the bit in `rocket_hw_profile.dtype_supported` for a `precision_*` value. The
 * mask is keyed on the encoding enum in `npu_hw.h`, which is common to every RKNPU. A
 * chip declares which of the shared encodings it can run, without redefining the
 * encodings. */
#define ROCKET_DT_BIT(prec)   (1u << (prec))

struct rocket_hw_profile {
    const char *name;          /* "rk3588", for logs and the ROCKET_CHIP override */

    /* The CBUF is the CNA's on-chip convolution buffer, and its size is the tiling budget.
     * The bank count is the highest-risk machine parameter, because every tile-budget
     * loop depends on it. On the RK3588 both fields mirror `npu_hw.h`, so each literal is
     * defined once. */
    int cbuf_banks;            /* number of CBUF banks      (RK3588: NPU_CBUF_BANKS)     */
    int cbuf_bank_size;        /* bytes per CBUF bank       (RK3588: NPU_CBUF_BANK_SIZE) */

    /* Matmul tile geometry. */
    int max_tile;              /* Mt and Nt cap, the M-independent resident-weight pivot */

    /* Weight tile-group sizes. They are exposed for a chip that narrows them, and they
     * document multi-chip support, but no code reads them yet. The tiling loops use the
     * validated per-datatype group literals (`Kt -= 32` and so on). The values come from
     * the IP, and software must confirm them on the device before a new chip's tiling uses
     * them. */
    int kgroup_2b;             /* weight K-group, <=2-byte dtypes (int8/int16/fp16/bf16) = 32 */
    int kgroup_4b;             /* weight K-group, 4-byte dtypes (tf32) = 16 (4-byte halves it) */
    int ngroup;                /* weight N-group = 16 */

    /* The `precision_*` encodings this SoC can run, as a capability mask rather than
     * encoding values. Query it with `rocket_hw_dtype_supported()`. */
    uint32_t dtype_supported;

    /* A warning logged once when this profile is selected, or NULL. A profile means the
     * machine parameters are measured. It says nothing about how much of the datapath has
     * a per-chip geometry encoder. A SoC where the two do not yet agree says so here,
     * rather than appearing fully supported. */
    const char *select_warning;

    int default_workers;       /* a suggested worker-thread count per fd, 8 on both
                                * profiles. It is a tuning value rather than the hardware
                                * core count, because the kernel schedules across the
                                * physical cores. No library code reads it. An entry
                                * takes its worker count from the caller's `nthreads` and
                                * caps it at a compile-time *_MAX_WORKERS bound. The field
                                * exists so that a chip can state a different value. */

    int slow_wait_ms;          /* the default slow-wait mark (ROCKET_SLOW_WAIT_MS). A
                                * fenced wait this long is counted, with a warning, as a
                                * probable retirement. The mark must sit under the
                                * driver's retirement of a job that never completes, or
                                * the counter never moves on that SoC. The retirement is
                                * the RK3588's 500 ms watchdog and the RK3576 series'
                                * 125 ms backstop. It is a property of each SoC's driver,
                                * kept here because the two differ. */

    int backstop_ms;           /* the driver's retirement: how long a job can run before
                                * the kernel gives up and signals its fence as if the job
                                * had completed. It is 500 on the RK3588 (the job
                                * watchdog) and 125 on the RK3576 (the poll backstop of
                                * patch rk3576/npu/0025, counted from the kick). The
                                * RK3576 write guard scores a job's submit-to-fence time
                                * against it, and ROCKET_RK3576_BACKSTOP_US overrides it.
                                * See rocket_rk3576_retired_counts(). */
};

/* Returns the active hardware profile, detected once on the first call. It uses
 * `ROCKET_CHIP` if set, then the device-tree `compatible` of the platform device bound to
 * the `rocket` driver, then the SoC-level `/proc/device-tree/compatible`. A chip that
 * matches no profile falls back to the RK3588 profile, with a warning. When neither
 * `compatible` can be read, it falls back with only a debug-level log. Code holding a
 * context reads `ctx->hw` instead, which is set to this at context initialization. */
const struct rocket_hw_profile *rocket_hw_current(void);

/* Returns the profile whose `name` matches, such as "rk3588", or NULL if there is none.
 * `ROCKET_CHIP` resolves through this function, so a name absent here cannot be forced. */
const struct rocket_hw_profile *rocket_hw_by_name(const char *name);

/* The RK3588 profile, the full target: every datatype and every op path. */
extern const struct rocket_hw_profile rocket_hw_rk3588;

/* The RK3576 profile. Its machine parameters are measured on the hardware. Its own
 * encoders cover int8 direct and depthwise convolution, fp16 convolution, the int8, int32
 * and fp16 matmuls, pooling and the LUT activations. The profile's `select_warning` names
 * what they leave unserved. */
extern const struct rocket_hw_profile rocket_hw_rk3576;

/* Returns true if `precision`, a `precision_*` value, can run on this profile. On the
 * RK3588 the mask is all-ones, because the datatype matrix is complete, so this never
 * rejects. On a chip that lacks an encoding it returns 0, and the caller returns
 * `ROCKET_E_UNSUPPORTED`. */
static inline int rocket_hw_dtype_supported(const struct rocket_hw_profile *p, int precision)
{
    return (p->dtype_supported & ROCKET_DT_BIT(precision)) != 0;
}


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_HW_PROFILE_H */
