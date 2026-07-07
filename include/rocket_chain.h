// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_chain.h — contiguous self-chaining regcmd layout for batched NPU submit.
 *
 * A multi-task NPU job can run as ONE hardware kick over CONTIGUOUS regcmds when
 * each task's trailer redirects the PC to the next task (an embedded
 * PC_BASE_ADDRESS write, plus the next segment's PC_REGISTER_AMOUNTS length) and
 * the kernel sets PC_TASK_CON.TASK_NUMBER = N to gate a single completion IRQ.
 * The PC_BASE_ADDRESS redirect is LOAD-BEARING: with the address left pointing at
 * task 0 the PC runs task 0 and stops, regardless of the amount.
 *
 * This is the SINGLE point where the gapped-vs-contiguous regcmd layout is
 * decided, shared by every submit path (matmul fp16/int8/int4, conv, ...). Each
 * path keeps its own gapped slot stride (the matmul/int8/int4 paths use 128 u64
 * words); the chained stride is always the regcmd word count rounded up to a
 * 128-bit (2-u64) boundary. The chain rewrites only the inert OP_NONE filler in
 * the trailer [OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE] that every gen_*
 * emits, so the op count — and so the stride — is unchanged.
 *
 * NOT a public/installed header.
 */
#ifndef ROCKET_CHAIN_H
#define ROCKET_CHAIN_H

#include <stddef.h>
#include <stdint.h>
#include "rocket_npu.h"   /* rocket_bo, rocket_task_desc */

/* Whether contiguous chaining is requested (ROCKET_BATCH_SUBMIT=1). Read once.
 * A chained batch must be uniform-length (the gen op count is data-independent
 * per dtype); a caller mixing regcmd lengths in one job must keep it off and let
 * each task keep its own gapped slot. The userspace half pairs with the kernel
 * `rocket_batch_submit=1` param — enable BOTH together or a chained stream runs
 * task 0 into the gap and times out. */
int rkt_chain_enabled(void);

/* The chained per-task stride in u64 words = the PC's per-task advance: the
 * regcmd word count rounded up to a 128-bit (2-u64) boundary. For the matmul's
 * even op counts this equals the count exactly (no padding). */
size_t rkt_chain_words(uint32_t regcmd_count);

/* Pack tile `nb`'s freshly generated regcmd (`src`, `count` u64 ops) into the
 * regcmd BO `rcbo` and fill its task descriptor `tasks[nb]`. When `chained`, lay
 * it contiguously at rkt_chain_words(count) stride and link its trailer to the
 * next task (next address + next stream length); otherwise use the caller's
 * `gapped_stride`-word slot. Mixed-length batches must pass chained=0. */
void rkt_chain_pack(int chained, rocket_bo *rcbo, rocket_task_desc *tasks,
                    int nb, const uint64_t *src, uint32_t count,
                    size_t gapped_stride);

/* Close a chained batch of `nb` tasks: the last task has no successor, so point
 * its embedded PC_BASE_ADDRESS link at 0 (TASK_NUMBER halts the PC after the last
 * OP_ENABLE, so this link is never followed — but a live address there would have
 * the PC prefetch past the batch). No-op when not chained or for a single task.
 * `count` is the uniform per-task op count (== tasks[0].regcmd_count). */
void rkt_chain_seal(int chained, rocket_bo *rcbo, int nb, uint32_t count);

#endif /* ROCKET_CHAIN_H */
