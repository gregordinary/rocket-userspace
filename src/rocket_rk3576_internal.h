// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef ROCKET_RK3576_INTERNAL_H
#define ROCKET_RK3576_INTERNAL_H

/*
 * rocket_rk3576_internal.h — the driving-side helpers every RK3576 entry point shares.
 *
 * The register encoders live in npu_regcmd_rk3576.c and the operand layouts in its
 * public header. What is here is the submit-loop discipline the part needs and no
 * generator can express: how long to idle when a submit came back having written
 * nothing, and whether an output surface is stamped before it runs. Both are properties
 * of the CHIP rather than of the op, so the matmul and the convolution share one copy.
 */

/* The idle a poisoned submit needs before the next one will write.
 *
 * An int32-output job leaves the NEXT submit — of any kind, across calls and across
 * processes — completing normally in about 1.4 ms and writing nothing. What clears it
 * is the driver's runtime-PM autosuspend cycling the NPU power domain, not elapsed
 * time: with `power/control` = `on` no amount of idle clears it at all, and the working
 * gap tracks `power/autosuspend_delay_ms` one for one. So the delay is read from the
 * driver rather than fixed, and a system that lowers it gets a cheaper path for free.
 * ROCKET_RK3576_MM_GAP_MS overrides it outright. [HW sweep, H96 MAX M9] */
void rocket_rk3576_power_idle(void);

/* Whether an output BO is stamped with a sentinel before the tasks that write it.
 *
 * A fresh BO arrives zeroed and zero is also a legitimate result, so a zeroed surface
 * cannot tell "never written" from "written and zero" — which is what made a poisoned
 * submit read as a wrong answer. Against a stamp the question is exact.
 *
 * Stamping is safe because the fill is BRACKETED by PREP_BO and FINI_BO, so the lines
 * are written back before the submit and none are left dirty to race the DPU's DMA. A
 * bare memset with no FINI_BO is the trap, and it is a different thing.
 *
 * ROCKET_RK3576_I32_SENTINEL=0 turns it off. */
int rocket_rk3576_sentinel_on(void);

#define ROCKET_RK3576_SENTINEL_BYTE 0xA5u

#endif /* ROCKET_RK3576_INTERNAL_H */
