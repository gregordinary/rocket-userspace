// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * Declares the contiguous, self-chaining regcmd layout for a batched submit.
 *
 * A multi-task job can run as one kick over contiguous regcmds. Each task's trailer must
 * redirect the PC to the next task, with an embedded `PC_BASE_ADDRESS` write and the next
 * segment's `PC_REGISTER_AMOUNTS` length. The kernel sets `PC_TASK_CON.TASK_NUMBER` = N,
 * so that one completion IRQ ends the kick.
 *
 * Caution: the `PC_BASE_ADDRESS` redirect is required. With the address left pointing at
 * task 0, the PC runs task 0 and stops, whatever the amount.
 *
 * These functions decide the gapped or contiguous regcmd layout for every submit path
 * (matmul fp16, int8 and int4, convolution and others). Each path keeps its own gapped
 * slot stride, and the matmul, int8 and int4 paths use 128 u64 words. The chained stride
 * is always the regcmd word count rounded up to a 128-bit (2-u64) boundary. The chain
 * rewrites only the inert `OP_NONE` filler in the trailer that every `gen_*` emits,
 * [OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]. The op count, and therefore the
 * stride, is unchanged.
 *
 * This header is internal and is not installed.
 */
#ifndef ROCKET_CHAIN_H
#define ROCKET_CHAIN_H

#include <stddef.h>
#include <stdint.h>
#include "rocket_npu.h"   /* rocket_bo, rocket_task_desc */

/* Returns whether contiguous chaining is on. `ROCKET_BATCH_SUBMIT=1` requests it, and when
 * the variable is unset the default is `rocket_batched_submit_native()`. The value is read
 * once.
 *
 * A batch that `rkt_chain_pack` and `rkt_chain_seal` chain must be uniform in length. The
 * generators' op count does not depend on the data within one datatype. A caller mixing
 * regcmd lengths in one job has two choices. It can keep chaining off and give each task
 * its own gapped slot. It can instead lay the chain out with `rkt_chain_pack_at` and check
 * it with `rkt_chain_verify`.
 *
 * The kernel half is the `rocket_batch_submit=1` module parameter, and both halves must be
 * enabled together. Otherwise a chained stream runs task 0 into the gap and times out.
 * Where `rocket_batched_submit_supported()` reports that the kernel does not honor the
 * flag, this function returns 0. It logs a warning only when chaining was wanted, through
 * `ROCKET_BATCH_SUBMIT` or the provider's default. */
int rkt_chain_enabled(void);

/* Returns the chained per-task stride in u64 words, which is the PC's per-task advance:
 * the regcmd word count rounded up to a 128-bit (2-u64) boundary. For the matmul's even
 * op counts it equals the count, with no padding. */
size_t rkt_chain_words(uint32_t regcmd_count);

/* Packs tile `nb`'s generated regcmd (`src`, `count` u64 ops) into the regcmd BO `rcbo`,
 * and fills its task descriptor `tasks[nb]`. With `chained` set, it lays the regcmd out
 * contiguously at a stride of `rkt_chain_words(count)`. It then links the trailer to the
 * next task's address and stream length. Otherwise it uses the caller's slot of
 * `gapped_stride` words. A mixed-length batch must pass `chained` = 0 here, and chains
 * through `rkt_chain_pack_at` instead.
 *
 * Returns 0 on success, and -1 on any of these failures:
 *
 * - The slot does not fit in the BO.
 * - Its address does not fit the regcmd's 32-bit address field.
 * - The trailer shape is not the one the chain expects.
 *
 * On a failure the caller must not submit the batch. As with `rkt_chain_pack_at`, a
 * failure means the chain link was not written, and the job would run task 0 and stall
 * into a fence timeout. */
int rkt_chain_pack(int chained, rocket_bo *rcbo, rocket_task_desc *tasks,
                   int nb, const uint64_t *src, uint32_t count,
                   size_t gapped_stride);

/* Packs the same link at an explicit offset, for a chain whose programs differ in length.
 * `rkt_chain_pack` derives task `nb`'s slot from `nb` and this task's own count, which is
 * a position only when every program is the same size. The sizes match for a tile's row
 * tasks, which write the same registers with different values. They differ across layers,
 * where a depthwise program and a direct one are different lengths.
 *
 * Both trailer fields describe the next segment, which a uniform-length chain cannot
 * distinguish. The kernel programs `PC_BASE_ADDRESS` and `PC_REGISTER_AMOUNTS` from task 0
 * alone. For every later program, the previous program's trailer supplies both the
 * address to fetch from and how much to fetch. `next_count` is therefore a parameter
 * separate from `count`.
 *
 * Caution: with a wrong `next_count`, the chain fetches a short segment as a long one. The
 * affected program executes partially or not at all, and nothing faults.
 *
 * `word_off` is where this program goes in the BO, and `next_word_off` is where the PC
 * continues. To mark the last program, pass `next_word_off` == `word_off`. Its forward
 * link is then cleared, and its `next_count` is ignored. The hardware stops there on
 * `TASK_NUMBER`, but a live address would make the PC prefetch past the chain. The
 * function fills `tasks[idx]`. Returns 0 on success, and -1 on the failures listed for
 * `rkt_chain_pack`, in which case the caller must not submit the batch. */
int rkt_chain_pack_at(rocket_bo *rcbo, rocket_task_desc *tasks, int idx,
                      size_t word_off, size_t next_word_off,
                      const uint64_t *src, uint32_t count, uint32_t next_count);

/* Closes a chained batch of `nb` tasks. The last task has no successor, so the function
 * clears its forward link, restoring the trailer's inert `OP_NONE` filler. `TASK_NUMBER`
 * halts the PC after the last `OP_ENABLE`, so this link is never followed. A live address
 * there would still make the PC prefetch past the batch. It does nothing when `chained`
 * is 0 or `nb` < 1. `count` is the uniform per-task op count (`tasks[0].regcmd_count`). */
void rkt_chain_seal(int chained, rocket_bo *rcbo, int nb, uint32_t count);

/* Reads back what was packed and checks that the chain is self-consistent. For every task,
 * its trailer's `PC_BASE_ADDRESS` must point at the next task's regcmd address, and its
 * `PC_REGISTER_AMOUNTS` must encode the next task's word count. The last task's forward
 * link must be cleared.
 *
 * The check is for `rkt_chain_pack_at`, which takes `next_count` from the caller and
 * cannot check it. A wrong value makes the chain fetch a short segment as a long one, with
 * nothing to fault on. Here the next program is in the BO, so the claimed length can be
 * checked. On the uniform-length `rkt_chain_pack` path, both fields derive from one count
 * and cannot disagree, so the check always passes there and costs little.
 *
 * The function applies to chained batches only. A gapped batch has no forward links to
 * check, and every task would read as one whose link was never written. `tasks[0..n)` must
 * be the descriptors the pack calls filled, in chain order. It reads the regcmd BO, so the
 * caller must call it after packing and after `rkt_chain_seal`, and before the submit.
 * Returns 0 on success. On a mismatch it returns -1 and logs a diagnostic naming the first
 * task whose link disagrees, and the caller must not submit. */
int rkt_chain_verify(const rocket_bo *rcbo, const rocket_task_desc *tasks, int n);

#endif /* ROCKET_CHAIN_H */
