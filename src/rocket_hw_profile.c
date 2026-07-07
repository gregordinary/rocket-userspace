// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_hw_profile.c — the one RK3588 hardware-profile instance + the active-
 * profile accessor. See rocket_hw_profile.h for the machine-parameter vs
 * inherent-datapath split and why a second profile is deferred.
 */
#include "rocket_hw_profile.h"

const struct rocket_hw_profile rocket_hw_rk3588 = {
    .name           = "rk3588",
    .cbuf_banks     = NPU_CBUF_BANKS,      /* 12 — mirrors npu_hw.h so the literal lives once */
    .cbuf_bank_size = NPU_CBUF_BANK_SIZE,  /* 32768 */
    .max_tile       = 256,                 /* Mt/Nt cap. 256 (not 384) lets Kt grow to 384 in
                                            * the CBUF (nKt 15->10), cutting readback: measured
                                            * 48.3->56.4 GFLOP/s on 512x3840x4096. Below 256,
                                            * DRAM reload (more N-tiles) outweighs the gain.
                                            * Override per-run with ROCKET_MM_MT/NT. */
    .kgroup_2b      = 32,                  /* int8/int16/fp16/bf16 weight K-group */
    .kgroup_4b      = 16,                  /* tf32 weight K-group (4-byte halves it) */
    .ngroup         = 16,                  /* weight N-group */

    /* All eight native encodings are HW-validated on the RK3588 (int4/int8/int16/
     * fp16/bf16/int32/fp32/tf32 — the datatype matrix is complete), so the mask is
     * all-ones over the precision_* range. A chip lacking one clears its bit. */
    .dtype_supported = ROCKET_DT_BIT(precision_int8)    | ROCKET_DT_BIT(precision_int16)
                     | ROCKET_DT_BIT(precision_float16) | ROCKET_DT_BIT(precision_bfloat16)
                     | ROCKET_DT_BIT(precision_int32)   | ROCKET_DT_BIT(precision_float32)
                     | ROCKET_DT_BIT(precision_int4)    | ROCKET_DT_BIT(precision_tf32),

    .default_workers = 8,
};

const struct rocket_hw_profile *rocket_hw_current(void)
{
    /* Single validated profile today. Multi-chip support (deferred until a second SoC is in
     * hand) replaces this with a per-device selection keyed on the accel node's DT
     * `compatible` string, plus a ROCKET_CHIP override for bring-up. */
    return &rocket_hw_rk3588;
}
