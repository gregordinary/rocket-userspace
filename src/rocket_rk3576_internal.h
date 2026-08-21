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
#include <stddef.h>
#include <stdint.h>

#include "rocket_npu.h"

/* The idle a poisoned submit needs before the next one will write.
 *
 * An int32-output job leaves the NEXT submit — of any kind, across calls and across
 * processes — completing normally in about 1.4 ms and writing nothing. What clears it
 * is the driver's runtime-PM autosuspend cycling the NPU power domain, not elapsed
 * time: with `power/control` = `on` no amount of idle clears it at all, and the working
 * gap tracks `power/autosuspend_delay_ms` one for one. So the delay is read from the
 * driver rather than fixed, and a system that lowers it gets a cheaper path for free.
 * ROCKET_RK3576_MM_GAP_MS overrides it outright. [HW sweep, H96 MAX M9]
 *
 * Returns 1 when the domain was OBSERVED to reach `suspended`, 0 when the call fell
 * back to a blind idle (no write access to the sysfs delay, or the domain did not
 * collapse inside the budget). A caller that retries on "wrote nothing" needs that
 * distinction: a redo after a confirmed cycle that still writes nothing is a
 * different fact from a redo after an idle that may never have cleared anything. */
int rocket_rk3576_power_idle(void);

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

/* Where in the CBUF a task stages, as a granule offset added to the window base and
 * the fetch base together — a bring-up knob, zero for every shipped path.
 *
 * The two NPU cores share one CBUF and both stage from granule 0, which is why two
 * jobs executing at once compute wrong answers. This is the one field a userspace
 * encoder emits that looks like an address into that pool, so it is the one candidate
 * for expressing a partition. Set PER THREAD, because a concurrency probe has to give
 * two workers different bases inside one process and ROCKET_RK3576_CBUF_BIAS is
 * process-wide; the environment variable is the fallback when it is never called.
 *
 * See tests/rk3576_cbuf_base.c — a bias the hardware IGNORES is invisible on a solo
 * job, so "it still computes" is not evidence that the base moved. */
void rocket_rk3576_set_cbuf_bias(unsigned granules);

/* WHAT A CROSS-LAYER CHAIN NEEDS OF A POOLING HANDLE.
 *
 * A pool is its own register program here — PPU and PPU_RDMA only, PC_OPERATION_ENABLE
 * 0x60 against a convolution's 0x1D — and the two bitmaps being disjoint was the reason
 * to doubt that one could sit inside a convolution stream. It can: 20 of 20 iterations
 * over five geometries to a 110x110 plane gave the same pool output AND the same output
 * from the convolution after it as separate submits, with both intermediates read so a
 * failure would have named its boundary [HW sweep, H96 MAX M9, tests/rk3576_chain_pool.c].
 *
 * The chain lives with the convolution entries because that is where the row plan and the
 * write guard are, so this is the whole of what it reads off a pool handle. INTERIOR
 * nodes only: a pool that began a run would need the chain to scatter into its cube and
 * one that ended a run would need it de-scattered, and both are host work the stream
 * exists to remove — so the chain refuses either and the caller keeps that layer's own
 * submit.
 *
 * `ops` is filled by the caller (RK3576_POOL_TASK_OPS words) rather than borrowed from the
 * handle's own regcmd BO: the chain lays every program of a run out contiguously in a BO
 * of its own and rewrites each trailer to link to the next, which must not disturb the
 * handle — it stays callable one at a time through the per-layer entry, which is what the
 * chain falls back to when a program has to be redone. */
struct rocket_rk3576_pool_link {
    int        fd;
    uint64_t   feat_dma;      /* the cube it reads     */
    uint64_t   surf_dma;      /* the surface it writes */
    int        cube_in, cube_out;
    rocket_bo *surf;          /* the BO the surface lives in   */
    size_t     surf_off;      /* this handle's slice inside it */
    unsigned   groups;        /* channel groups written        */
    unsigned   surf_elems;    /* elements per channel group    */
    unsigned   live_elems;    /* elements of a group it fills  */
    uint32_t   nops;          /* words written into `ops`      */
};

struct rocket_pool_int8_rk3576_handle;

int rocket_rk3576_pool_link(struct rocket_pool_int8_rk3576_handle *h,
                            struct rocket_rk3576_pool_link *out, uint64_t *ops);

#endif /* ROCKET_RK3576_INTERNAL_H */
