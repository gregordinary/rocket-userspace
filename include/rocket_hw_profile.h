// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_hw_profile.h — the single source of truth for the NPU's MACHINE
 * PARAMETERS: the values that are fixed for a given SoC but may differ on another
 * rknpu/NVDLA-class chip (CBUF size, tile geometry, the usable datatype menu, the
 * worker default). Collecting them in one profile makes librocketnpu agnostic by
 * construction, with the RK3588 as the first — and currently only HARDWARE-
 * VALIDATED — target.
 *
 * The stack divides cleanly, and the two halves belong in different places:
 *   - HERE (machine parameters): CBUF banks + bank size, the matmul tile cap, the
 *     tile-group sizes, the dtype-availability mask, the worker default.
 *   - NOT here (inherent datapath — shared across the IP family; lives in npu_hw.h
 *     / npu_regcmd.c): register offsets, the precision_* ENCODING values, the
 *     CNA->CORE->DPU sequence, the BS/BN/EW field meanings, the LUT machinery, and
 *     the tile-layout index algebra. Those come from the IP, not the SoC, so they
 *     are NOT parameterized.
 *
 * This header is a value-preserving inventory: one struct, one instance
 * (rocket_hw_rk3588) populated with today's RK3588 constants, read through
 * rocket_hw_current(). It adds NO autodetect and NO second profile — a second
 * chip's profile must be HW-validated on that silicon (re-run the bit-exact gates
 * under tests/), which is deferred until such hardware is in hand.
 */
#ifndef ROCKET_HW_PROFILE_H
#define ROCKET_HW_PROFILE_H

#include <stdint.h>
#include "npu_hw.h"   /* NPU_CBUF_BANKS / NPU_CBUF_BANK_SIZE (the compile-time
                       * constants the profile MIRRORS, so the literal lives once)
                       * + the precision_* enum the dtype mask is keyed on. */

#ifdef __cplusplus
extern "C" {
#endif

/* Bit in rocket_hw_profile.dtype_supported for a precision_* value. The mask is
 * keyed on the IP-inherent encoding enum (npu_hw.h): a chip declares WHICH of the
 * shared encodings it can actually run, without redefining the encodings. */
#define ROCKET_DT_BIT(prec)   (1u << (prec))

struct rocket_hw_profile {
    const char *name;          /* "rk3588" — for logs / the future ROCKET_CHIP override */

    /* CBUF — the CNA's on-chip conv buffer; the tiling-budget surface. The bank
     * COUNT is the highest-risk machine parameter (every tile-budget loop keys off
     * it). Both mirror npu_hw.h so the literal is defined exactly once. */
    int cbuf_banks;            /* number of CBUF banks      (== NPU_CBUF_BANKS)     */
    int cbuf_bank_size;        /* bytes per CBUF bank       (== NPU_CBUF_BANK_SIZE) */

    /* Matmul tile geometry. */
    int max_tile;              /* Mt/Nt cap; the M-independent resident-weight pivot */

    /* Weight tile-group sizes. SURFACED here for a chip that narrows them, but the
     * tiling loops still use the validated per-dtype group literals (Kt -= 32 etc.):
     * these are IP-derived and must be CONFIRMED on-device, not assumed, before a
     * new chip routes through them. Documentation for multi-chip support, not yet consumed. */
    int kgroup_2b;             /* weight K-group, <=2-byte dtypes (int8/int16/fp16/bf16) = 32 */
    int kgroup_4b;             /* weight K-group, 4-byte dtypes (tf32) = 16 (4-byte halves it) */
    int ngroup;                /* weight N-group = 16 */

    /* Which precision_* encodings this SoC can actually run — capability, not the
     * encoding values. Query with rocket_hw_dtype_supported(). */
    uint32_t dtype_supported;

    int default_workers;       /* default per-fd worker-thread cap. A TUNING default,
                                * NOT the HW core count (the kernel schedules across
                                * the physical cores); the array-sizing *_MAX_WORKERS
                                * bound stays a compile-time constant in the prepacked
                                * files. Surfaced only so a chip can prefer a different
                                * default. */
};

/* The active hardware profile. Single-profile today, so this is a trivial accessor
 * returning &rocket_hw_rk3588; it is the seam a future per-device autodetect (Phase
 * 2: read the accel node's DT `compatible`) would hook, plus a ROCKET_CHIP override.
 * Code holding a context reads ctx->hw instead (set to this at context init). */
const struct rocket_hw_profile *rocket_hw_current(void);

/* The RK3588 profile — the one validated target. */
extern const struct rocket_hw_profile rocket_hw_rk3588;

/* True if `precision` (a precision_* value) is runnable on this profile. On the
 * RK3588 the mask is all-ones (the datatype matrix is complete), so this never
 * rejects; on a chip that lacks an encoding it returns 0 and the caller surfaces
 * ROCKET_E_UNSUPPORTED. */
static inline int rocket_hw_dtype_supported(const struct rocket_hw_profile *p, int precision)
{
    return (p->dtype_supported & ROCKET_DT_BIT(precision)) != 0;
}


#ifdef __cplusplus
}
#endif
#endif /* ROCKET_HW_PROFILE_H */
