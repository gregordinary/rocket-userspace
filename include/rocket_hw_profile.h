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
 *     / npu_regcmd.c): the precision_* ENCODING values, the CNA->CORE->DPU sequence,
 *     the BS/BN/EW field meanings, the LUT machinery, and the tile-layout index
 *     algebra. Those come from the IP, not the SoC, so they are NOT parameterized.
 *
 * One datapath element is NOT fully shared: the CNA/CORE/DPU GEOMETRY-register
 * encoding. The offsets and field packing in npu_hw.h are the RK3588's, and they are
 * IP-REVISION-specific. The RK3576 NPU re-packs the CNA block at the same block base,
 * and it is more than an offset shift: registers move, the bit-packing differs at shared
 * offsets, and the RK3576 uses offsets that are gaps here (only 2 CNA geometry regs
 * coincide). The datapath SEMANTICS (the precision-field encodings, the CNA->CORE->DPU
 * block sequence, the BS/BN/EW/LUT meanings) still transfer. Multi-chip support
 * therefore needs a per-chip regcmd ENCODER for the geometry registers, not just an
 * offset table: npu_regcmd_rk3576.c is that encoder for the RK3576 int8 conv path, gated
 * off the vendor's own register programs (tests/regcmd_rk3576_gate.c). See
 * rockchip-npu-notes/chips/rk3576-regcmd.md.
 *
 * Two profiles exist. A chip earns one by having its machine parameters MEASURED on
 * that silicon, which the RK3576 now has: its encoder computes bit-exact int8
 * convolutions there, and the CBUF capacity was swept against them. A profile is not
 * a claim of full support — the RK3576's covers only what has been measured, and it
 * carries a select_warning saying which datapaths still have no encoder for it.
 *
 * What rocket_hw_current() does add is SELECTION: it reads the NPU's DT `compatible`
 * (overridable with ROCKET_CHIP) and warns when the part is not an RK3588, instead of
 * silently applying RK3588 parameters to foreign silicon. On such a chip the warning
 * is the whole value — the geometry encoder is wrong for it too, so the run is
 * expected to produce no output or wrong output until its per-chip encoder lands.
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

    /* Warned once when this profile is selected, or NULL. A profile means the
     * MACHINE PARAMETERS are measured; it says nothing about how much of the
     * datapath has a per-chip geometry encoder. A part where those two do not yet
     * agree says so here rather than looking fully supported. */
    const char *select_warning;

    int default_workers;       /* default per-fd worker-thread cap. A TUNING default,
                                * NOT the HW core count (the kernel schedules across
                                * the physical cores); the array-sizing *_MAX_WORKERS
                                * bound stays a compile-time constant in the prepacked
                                * files. Surfaced only so a chip can prefer a different
                                * default. */
};

/* The active hardware profile, detected once on first call: ROCKET_CHIP if set, else
 * the rocket-bound platform device's DT `compatible`, else the SoC-level
 * /proc/device-tree/compatible. Anything that is not a profile we have falls back to
 * the RK3588 profile and warns. Code holding a context reads ctx->hw instead (set to
 * this at context init). */
const struct rocket_hw_profile *rocket_hw_current(void);

/* Look a profile up by its `name` ("rk3588"); NULL if there is none. This is what
 * ROCKET_CHIP resolves through, so a name that is absent here is not forceable. */
const struct rocket_hw_profile *rocket_hw_by_name(const char *name);

/* The RK3588 profile — the full target: every datatype and every op path. */
extern const struct rocket_hw_profile rocket_hw_rk3588;

/* The RK3576 profile. Its machine parameters are HW-measured, but only the int8
 * CONV_2D path has a geometry encoder for this chip, so the profile carries a
 * select_warning saying so. */
extern const struct rocket_hw_profile rocket_hw_rk3576;

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
