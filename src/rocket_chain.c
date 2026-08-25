// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_chain.c — contiguous self-chaining regcmd layout (see rocket_chain.h).
 *
 * Extracted from rocket_matmul.c so the matmul (fp16/int8/int4/...) and conv
 * submit paths share one chain implementation. The mode-3 layout (an embedded
 * PC_BASE_ADDRESS redirect repurposing the trailer's OP_NONE filler) was
 * established by on-HW A/B on the RK1: contiguous + trailer=0 OR
 * encoded-amount-only BOTH stall (task 0 runs, counter stuck, 500 ms timeout) —
 * the PC must be told WHERE the next task is, not just how long.
 */
#include <stdlib.h>   /* getenv, atoi */
#include <string.h>
#include <stdatomic.h>

#include "rocket_chain.h"
#include "npu_hw.h"   /* OP_NONE, OP_REG_PC, PC_BASE_ADDRESS, PC_REGISTER_AMOUNTS */
#include "rocket_npu.h"  /* rocket_batched_submit_supported */
#include "rocket_log.h"
#include "rocket_op.h"   /* rocket_op_iova_overflow — one 32-bit IOVA rule */

/* A trailer rewrite below can only fail if a gen_* generator stops emitting the
 * [OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE] trailer this chain layout claims
 * (the chain_layout_rocket host gate guards that). If it ever does, the chain link is
 * not written and the batch runs task 0 and stalls into a multi-second fence timeout —
 * a mystery hang. Convert that into one loud, actionable diagnostic instead. Logged
 * once (atomic guard) so a batch of N tiles does not emit N copies. */
static void chain_trailer_fail(const char *where)
{
    static atomic_flag warned = ATOMIC_FLAG_INIT;
    if (!atomic_flag_test_and_set_explicit(&warned, memory_order_relaxed))
        ROCKET_LOGE("rocket_chain: %s found no [OP_NONE, PC_REGISTER_AMOUNTS] trailer to "
                    "rewrite — the regcmd trailer shape drifted, so the chained batch link "
                    "was NOT written and this submit will stall. Disable ROCKET_BATCH_SUBMIT "
                    "(and file a bug); run the chain_layout_rocket gate.\n", where);
}

int rkt_chain_enabled(void) {
    static _Atomic int c = -1;
    if (c < 0) {
        /* The DEFAULT is the provider's, not 0: where the chained layout is the uAPI's
         * only multi-program shape, the gapped one is the odd choice and costs both the
         * extra ioctls and CBUF reuse. The env var still forces either way. */
        const char *e = getenv("ROCKET_BATCH_SUBMIT");
        int want = e ? (atoi(e) > 0 ? 1 : 0) : rocket_batched_submit_native();
        /*
         * Asking for chaining is not enough — the kernel has to hold up its half.
         * Chaining is a joint layout contract (we self-chain the regcmds, the
         * kernel sets TASK_NUMBER = task_count); a kernel that does not know
         * DRM_ROCKET_JOB_BATCHED ignores the flag and runs our chained layout
         * down the per-task path, which corrupts or stalls the job. So an
         * unsupported kernel must DISABLE chaining, not silently proceed.
         */
        if (want && !rocket_batched_submit_supported()) {
            ROCKET_LOGW("ROCKET_BATCH_SUBMIT=1 but this kernel does not honor "
                        "DRM_ROCKET_JOB_BATCHED (needs the rocket batched-submit "
                        "patch; driver must report >= 1.1). Running the stock "
                        "per-task path instead.\n");
            want = 0;
        }
        c = want;
    }
    return c;
}

size_t rkt_chain_words(uint32_t regcmd_count) {
    return (size_t)((regcmd_count + 1u) & ~1u);
}

/* PC_DATA_AMOUNT encoding for a regcmd of `count` u64 ops — mirrors the kernel's
 * PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT((count+1)/2 - 1). */
static inline uint32_t rkt_amount_encode(uint32_t count) {
    return (count + 1u) / 2u - 1u;
}

/* Rewrite a copied regcmd's trailer PC_REGISTER_AMOUNTS op — the single
 * NPUOP(OP_REG_PC, *, PC_REGISTER_AMOUNTS) near the end — to carry `amount` in
 * its value field. Scans from the end so it finds the trailer, not any earlier
 * PC write. -1 if the regcmd has no such op. */
static int rkt_set_trailer_amount(uint64_t *rc, size_t words, uint32_t amount) {
    for (size_t j = words; j-- > 0; ) {
        if ((uint16_t)(rc[j] >> 48) == OP_REG_PC &&
            (uint16_t)(rc[j] & 0xffff) == PC_REGISTER_AMOUNTS) {
            rc[j] = ((uint64_t)OP_REG_PC << 48) |
                    ((uint64_t)amount << 16) | PC_REGISTER_AMOUNTS;
            return (int)j;
        }
    }
    return -1;
}

/* Mode 3: give the regcmd an embedded PC_BASE_ADDRESS op pointing at the next
 * task so the PC advances to it after this task's OP_ENABLE. The trailer's inert
 * OP_NONE filler (the op immediately preceding the PC_REGISTER_AMOUNTS op) is
 * repurposed into that write, keeping the op count — and so the chained stride —
 * unchanged. The value is the raw 32-bit IOVA, exactly as the kernel programs
 * PC_BASE_ADDRESS for task 0. Returns 0 on success, -1 if the expected
 * OP_NONE/amount trailer shape is not found (caller then must not use mode 3). */
static int rkt_set_trailer_base(uint64_t *rc, size_t words, uint32_t next_addr) {
    int amt = -1;
    for (size_t j = words; j-- > 0; )
        if ((uint16_t)(rc[j] >> 48) == OP_REG_PC &&
            (uint16_t)(rc[j] & 0xffff) == PC_REGISTER_AMOUNTS) { amt = (int)j; break; }
    if (amt < 1)
        return -1;
    /* The PC trailer is [OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]; the
     * OP_NONE just before the amount op is the slot we claim. Refuse if it is not
     * the expected inert filler, rather than clobber a real op. */
    if ((uint16_t)(rc[amt - 1] >> 48) != OP_NONE)
        return -1;
    rc[amt - 1] = ((uint64_t)OP_REG_PC << 48) |
                  ((uint64_t)next_addr << 16) | PC_BASE_ADDRESS;
    return 0;
}

/*
 * THE MULTI-TASK IOVA CHOKE POINT.
 *
 * rocket_submit_matmul_flags validates the regcmd BO because it still has one:
 * it takes a `const rocket_bo *`. rocket_submit_tasks_pre takes
 * rocket_task_desc { uint32_t regcmd; }, so by the time a chained batch reaches
 * submit the address has ALREADY been truncated and there is nothing left to
 * check. Every multi-task path therefore depended on a per-call-site convention.
 *
 * This file is the single funnel all of them pass through, and it still has the
 * rocket_bo, so the check belongs here. Two separate questions: does the slot fit
 * inside the BO at all, and is the BO addressable from a 32-bit register field.
 */
static int rkt_chain_slot_ok(const char *where, rocket_bo *rcbo,
                             size_t word_off, uint32_t count)
{
    rocket_bo *const chk[] = { rcbo };
    size_t end = (word_off + count) * sizeof(uint64_t);

    if (end > rcbo->size) {
        ROCKET_LOGE("rocket_chain: %s wants bytes [%zu,%zu) of a %zu-byte regcmd BO\n",
                    where, word_off * sizeof(uint64_t), end, rcbo->size);
        return 0;
    }
    /* The whole BO, not this slot: if its last byte is encodable then so is every
     * slot inside it, and the 32-bit rule stays written down in exactly one place. */
    return rocket_op_iova_overflow(where, chk, 1) ? 0 : 1;
}

int rkt_chain_pack(int chained, rocket_bo *rcbo, rocket_task_desc *tasks,
                   int nb, const uint64_t *src, uint32_t count,
                   size_t gapped_stride) {
    size_t stride = chained ? rkt_chain_words(count) : gapped_stride;
    size_t word_off = (size_t)nb * stride;
    uint64_t *slot = (uint64_t *)rcbo->ptr + word_off;

    if (!rkt_chain_slot_ok("rkt_chain_pack", rcbo, word_off, count))
        return -1;
    memcpy(slot, src, (size_t)count * sizeof(uint64_t));
    if (chained) {
        /* Link to the next contiguous slot: the PC_BASE_ADDRESS redirect points the
         * PC there after this task's OP_ENABLE, and PC_REGISTER_AMOUNTS gives that
         * segment's length. rkt_chain_seal zeroes the final task's link (no next). */
        if (rkt_set_trailer_amount(slot, count, rkt_amount_encode(count)) < 0) {
            chain_trailer_fail("rkt_chain_pack (amount)");
            return -1;
        }
        uint32_t next_addr = (uint32_t)(rcbo->dma_address +
                                        (size_t)(nb + 1) * stride * sizeof(uint64_t));
        if (rkt_set_trailer_base(slot, count, next_addr) < 0) {
            chain_trailer_fail("rkt_chain_pack (base)");
            return -1;
        }
    }
    tasks[nb].regcmd = (uint32_t)(rcbo->dma_address + word_off * sizeof(uint64_t));
    tasks[nb].regcmd_count = count;
    return 0;
}

/* Drop the forward link from a task's trailer: restore the PC_BASE_ADDRESS redirect
 * that rkt_set_trailer_base installed (in the OP_NONE filler slot before
 * PC_REGISTER_AMOUNTS) back to the inert OP_NONE filler. Idempotent — a slot still
 * holding OP_NONE is left as-is. -1 if the regcmd has no PC_REGISTER_AMOUNTS trailer. */
static int rkt_clear_trailer_base(uint64_t *rc, size_t words) {
    for (size_t j = words; j-- > 1; ) {
        if ((uint16_t)(rc[j] >> 48) == OP_REG_PC &&
            (uint16_t)(rc[j] & 0xffff) == PC_REGISTER_AMOUNTS) {
            rc[j - 1] = NPUOP(OP_NONE, 0x0, 0x0);   /* the inert filler gen emits */
            return 0;
        }
    }
    return -1;
}

int rkt_chain_pack_at(rocket_bo *rcbo, rocket_task_desc *tasks, int idx,
                      size_t word_off, size_t next_word_off,
                      const uint64_t *src, uint32_t count, uint32_t next_count) {
    uint64_t *slot = (uint64_t *)rcbo->ptr + word_off;
    int last = (next_word_off == word_off);

    if (!rkt_chain_slot_ok("rkt_chain_pack_at", rcbo, word_off, count))
        return -1;
    memcpy(slot, src, (size_t)count * sizeof(uint64_t));
    /* The NEXT segment's length, because that is what this write is for — see the header.
     * The last program has no next, so its own count goes in and is never read. */
    if (rkt_set_trailer_amount(slot, count,
                               rkt_amount_encode(last ? count : next_count)) < 0) {
        chain_trailer_fail("rkt_chain_pack_at (amount)");
        return -1;
    }
    if (!last) {
        uint32_t next_addr = (uint32_t)(rcbo->dma_address +
                                       next_word_off * sizeof(uint64_t));
        if (rkt_set_trailer_base(slot, count, next_addr) < 0) {
            chain_trailer_fail("rkt_chain_pack_at (base)");
            return -1;
        }
    } else if (rkt_clear_trailer_base(slot, count) < 0) {
        chain_trailer_fail("rkt_chain_pack_at (seal)");
        return -1;
    }
    tasks[idx].regcmd = (uint32_t)(rcbo->dma_address + word_off * sizeof(uint64_t));
    tasks[idx].regcmd_count = count;
    return 0;
}

/* Locate the trailer's PC_REGISTER_AMOUNTS op and hand back its encoded amount.
 * Scans from the end so it finds the trailer, not an earlier PC write. Returns the
 * op's index, or -1. */
static int rkt_find_trailer_amount(const uint64_t *rc, size_t words, uint32_t *amount) {
    for (size_t j = words; j-- > 0; ) {
        if ((uint16_t)(rc[j] >> 48) == OP_REG_PC &&
            (uint16_t)(rc[j] & 0xffff) == PC_REGISTER_AMOUNTS) {
            *amount = (uint32_t)((rc[j] >> 16) & 0xffffffffu);
            return (int)j;
        }
    }
    return -1;
}

int rkt_chain_verify(const rocket_bo *rcbo, const rocket_task_desc *tasks, int n) {
    if (n < 1)
        return 0;

    for (int i = 0; i < n; i++) {
        uint32_t base = (uint32_t)rcbo->dma_address;
        if (tasks[i].regcmd < base) {
            ROCKET_LOGE("rocket_chain: verify task %d regcmd 0x%x is below the regcmd BO "
                        "base 0x%x\n", i, tasks[i].regcmd, base);
            return -1;
        }
        size_t word_off = (size_t)(tasks[i].regcmd - base) / sizeof(uint64_t);
        uint32_t count = tasks[i].regcmd_count;
        if ((word_off + count) * sizeof(uint64_t) > rcbo->size) {
            ROCKET_LOGE("rocket_chain: verify task %d runs past the regcmd BO\n", i);
            return -1;
        }

        const uint64_t *slot = (const uint64_t *)rcbo->ptr + word_off;
        uint32_t amount = 0;
        int amt = rkt_find_trailer_amount(slot, count, &amount);
        if (amt < 1) {
            ROCKET_LOGE("rocket_chain: verify task %d has no [OP_NONE, PC_REGISTER_AMOUNTS] "
                        "trailer\n", i);
            return -1;
        }

        const uint64_t link = slot[amt - 1];
        int is_link = ((uint16_t)(link >> 48) == OP_REG_PC &&
                       (uint16_t)(link & 0xffff) == PC_BASE_ADDRESS);

        if (i == n - 1) {
            /* No successor: the forward link must have been cleared, or the PC
             * prefetches past the end of the chain. */
            if (is_link) {
                ROCKET_LOGE("rocket_chain: verify last task %d still carries a forward link "
                            "to 0x%x — rkt_chain_seal did not run\n",
                            i, (uint32_t)((link >> 16) & 0xffffffffu));
                return -1;
            }
            continue;
        }

        if (!is_link) {
            ROCKET_LOGE("rocket_chain: verify task %d has no forward link; the chain would "
                        "run task %d and stall\n", i, i);
            return -1;
        }
        uint32_t next_addr = (uint32_t)((link >> 16) & 0xffffffffu);
        if (next_addr != tasks[i + 1].regcmd) {
            ROCKET_LOGE("rocket_chain: verify task %d links to 0x%x but task %d is at 0x%x\n",
                        i, next_addr, i + 1, tasks[i + 1].regcmd);
            return -1;
        }
        /* The claim that cannot be checked at pack time: this task's trailer says
         * how many words to fetch for the NEXT program, and only here is the next
         * program's real length in hand. A short segment fetched as a long one
         * executes partially with nothing to fault on. */
        uint32_t want = rkt_amount_encode(tasks[i + 1].regcmd_count);
        if (amount != want) {
            ROCKET_LOGE("rocket_chain: verify task %d encodes next length %u (%u words) but "
                        "task %d is %u words (%u)\n", i, amount, (amount + 1u) * 2u,
                        i + 1, tasks[i + 1].regcmd_count, want);
            return -1;
        }
    }
    return 0;
}

void rkt_chain_seal(int chained, rocket_bo *rcbo, int nb, uint32_t count) {
    if (!chained || nb < 1)
        return;
    size_t stride = rkt_chain_words(count);
    uint64_t *last = (uint64_t *)rcbo->ptr + (size_t)(nb - 1) * stride;
    /* rkt_chain_pack links EVERY task to the next contiguous slot (it cannot know which
     * is last); the final task has no successor, so clear its forward link. Earlier this
     * called rkt_set_trailer_base(...,0), which requires the slot to still be OP_NONE and
     * thus always failed after pack had converted it to PC_BASE_ADDRESS — a dangling link
     * into the slot past the chain survived (benign only because the HW stops at
     * TASK_NUMBER). Clearing it removes that hazard and the spurious "will stall" warning. */
    if (rkt_clear_trailer_base(last, count) < 0)
        chain_trailer_fail("rkt_chain_seal");
}
