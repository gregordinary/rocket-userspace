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
#include "rocket_log.h"

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
        const char *e = getenv("ROCKET_BATCH_SUBMIT");
        c = (e && atoi(e) > 0) ? 1 : 0;
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

void rkt_chain_pack(int chained, rocket_bo *rcbo, rocket_task_desc *tasks,
                    int nb, const uint64_t *src, uint32_t count,
                    size_t gapped_stride) {
    size_t stride = chained ? rkt_chain_words(count) : gapped_stride;
    uint64_t *slot = (uint64_t *)rcbo->ptr + (size_t)nb * stride;
    memcpy(slot, src, (size_t)count * sizeof(uint64_t));
    if (chained) {
        /* Link to the next contiguous slot: the PC_BASE_ADDRESS redirect points the
         * PC there after this task's OP_ENABLE, and PC_REGISTER_AMOUNTS gives that
         * segment's length. rkt_chain_seal zeroes the final task's link (no next). */
        if (rkt_set_trailer_amount(slot, count, rkt_amount_encode(count)) < 0)
            chain_trailer_fail("rkt_chain_pack (amount)");
        uint32_t next_addr = (uint32_t)(rcbo->dma_address +
                                        (size_t)(nb + 1) * stride * sizeof(uint64_t));
        if (rkt_set_trailer_base(slot, count, next_addr) < 0)
            chain_trailer_fail("rkt_chain_pack (base)");
    }
    tasks[nb].regcmd = (uint32_t)(rcbo->dma_address + (size_t)nb * stride * sizeof(uint64_t));
    tasks[nb].regcmd_count = count;
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
