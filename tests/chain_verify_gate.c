// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * chain_verify_gate.c — host-only GATE for rkt_chain_verify(). No NPU.
 *
 * rkt_chain_pack_at() takes `next_count` as a parameter: the caller asserts how
 * long the NEXT program is, and that number goes into THIS program's trailer as
 * the length the PC will fetch. Nothing at pack time can check the claim, and
 * rocket_chain.h says getting it wrong makes the chain fetch a short segment as a
 * long one — the affected program executes partially or not at all, with nothing
 * to fault on. rkt_chain_verify() closes that by reading back what was packed.
 *
 * This gate exists because the verifier's one production caller is on the RK3576
 * conv path, and every RK3576 gate SKIPS on an RK3588 board — so on the hardware
 * this project runs its suite on, the verifier would otherwise never execute. An
 * unexercised checker is not evidence that anything is checked.
 *
 * Host-only on purpose: the pack/verify pair is pure memory work over a rocket_bo,
 * so a synthetic BO reaches all of it and no device is needed. That also means
 * this gate cannot skip, and a green here is never a green about nothing.
 *
 * The negative cases are the point. A verifier that only ever sees correct input
 * is indistinguishable from `return 0`.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_chain.h"
#include "npu_hw.h"
#include "rocket_log.h"

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

#define BO_BYTES  (64u * 1024u)
#define BO_IOVA   0x10000000u    /* arbitrary, low enough to encode in 32 bits */

/* A synthetic program with the trailer shape the chain rewrite claims:
 *   [... , OP_NONE, PC_REGISTER_AMOUNTS, OP_40, OP_ENABLE]
 * Only the trailer matters to pack/verify; the body is filler that is never read. */
static void synth(uint64_t *ops, uint32_t count)
{
    for (uint32_t i = 0; i < count - 4; i++)
        ops[i] = NPUOP(OP_REG_DPU, i, DPU_S_POINTER);
    ops[count - 4] = NPUOP(OP_NONE, 0x0, 0x0);
    ops[count - 3] = NPUOP(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
    ops[count - 2] = NPUOP(OP_40, 0x0, 0x0);
    ops[count - 1] = NPUOP(OP_ENABLE, 0x0, 0x0);
}

/* A rocket_bo backed by plain host memory. handle must be nonzero or
 * rocket_op_iova_overflow treats the BO as unallocated and skips it. */
static void fake_bo(rocket_bo *bo, void *mem)
{
    memset(bo, 0, sizeof *bo);
    bo->handle      = 1;
    bo->size        = BO_BYTES;
    bo->dma_address = BO_IOVA;
    bo->ptr         = mem;
}

/*
 * Three programs of DIFFERENT lengths — the heterogeneous case rkt_chain_pack_at
 * exists for, and the only one where next_count can disagree with reality.
 * `bad_at` >= 0 corrupts that task's declared next_count, which is the defect
 * this whole mechanism is aimed at.
 */
static const uint32_t COUNTS[3] = { 12, 20, 16 };
static const size_t   OFFS[3]   = { 0, 32, 64 };

static int pack_chain(rocket_bo *bo, rocket_task_desc *td, int bad_at, int leave_link)
{
    uint64_t prog[64];
    for (int i = 0; i < 3; i++) {
        int last  = (i == 2);
        size_t nx = last ? OFFS[i] : OFFS[i + 1];
        uint32_t nc = last ? COUNTS[i] : COUNTS[i + 1];
        if (i == bad_at) nc = COUNTS[i + 1] + 2;   /* the lie */
        if (last && leave_link) nx = OFFS[0];      /* a live forward link on the last task */
        synth(prog, COUNTS[i]);
        if (rkt_chain_pack_at(bo, td, i, OFFS[i], nx, prog, COUNTS[i], nc) != 0)
            return -1;
    }
    return 0;
}

int main(void)
{
    void *mem = malloc(BO_BYTES);
    if (!mem) { printf("FAIL: out of memory\n"); return 1; }
    rocket_bo bo;
    rocket_task_desc td[3];

    /* Keep the library's own error prints out of the expected-failure cases, so a
     * green run's output is not full of alarming lines that are the point. */
    printf("chain_verify_gate: 3 heterogeneous programs, counts %u/%u/%u\n\n",
           COUNTS[0], COUNTS[1], COUNTS[2]);

    /* 1. A correctly packed chain verifies. */
    memset(mem, 0, BO_BYTES);
    fake_bo(&bo, mem);
    CHECK(pack_chain(&bo, td, -1, 0) == 0, "packing a well-formed chain failed");
    CHECK(rkt_chain_verify(&bo, td, 3) == 0, "verify rejected a correctly packed chain");
    printf("[ok] a correct chain verifies\n");

    /* 2. THE DEFECT: task 0 declares the wrong length for task 1. Every address is
     *    right, every program is present, and the only thing wrong is a number that
     *    pack_at had no way to check. */
    memset(mem, 0, BO_BYTES);
    fake_bo(&bo, mem);
    CHECK(pack_chain(&bo, td, 0, 0) == 0, "packing with a wrong next_count failed early");
    printf("  (the next two lines are the expected diagnostics)\n");
    CHECK(rkt_chain_verify(&bo, td, 3) != 0,
          "verify ACCEPTED task 0 declaring the wrong length for task 1 — "
          "the chain would fetch a %u-word segment as %u words",
          COUNTS[1], COUNTS[1] + 2);
    printf("[ok] a wrong next_count is caught\n");

    /* 3. The same lie one link further along, so the walk is exercised past the head. */
    memset(mem, 0, BO_BYTES);
    fake_bo(&bo, mem);
    CHECK(pack_chain(&bo, td, 1, 0) == 0, "packing with a wrong next_count failed early");
    CHECK(rkt_chain_verify(&bo, td, 3) != 0,
          "verify ACCEPTED task 1 declaring the wrong length for task 2");
    printf("[ok] a wrong next_count is caught at the second link too\n");

    /* 4. A last task that still carries a forward link. The hardware halts on
     *    TASK_NUMBER so this does not corrupt, but it leaves the PC prefetching
     *    past the chain — rkt_chain_seal's whole job. */
    memset(mem, 0, BO_BYTES);
    fake_bo(&bo, mem);
    CHECK(pack_chain(&bo, td, -1, 1) == 0, "packing with a live last link failed early");
    CHECK(rkt_chain_verify(&bo, td, 3) != 0,
          "verify ACCEPTED a last task still carrying a forward link");
    printf("[ok] an unsealed last task is caught\n");

    /* 5. A single-task chain is a valid chain: one program, no links, sealed. */
    memset(mem, 0, BO_BYTES);
    fake_bo(&bo, mem);
    {
        uint64_t prog[64];
        synth(prog, COUNTS[0]);
        CHECK(rkt_chain_pack_at(&bo, td, 0, 0, 0, prog, COUNTS[0], COUNTS[0]) == 0,
              "packing a single-task chain failed");
        CHECK(rkt_chain_verify(&bo, td, 1) == 0, "verify rejected a single-task chain");
    }
    printf("[ok] a single-task chain verifies\n");

    /* 6. A task pointing outside the BO is refused rather than read. Deliberately a
     *    ONE-task chain: in a longer one the previous task's link mismatch fires
     *    first, and the case would pass without the bounds check ever running. */
    memset(mem, 0, BO_BYTES);
    fake_bo(&bo, mem);
    {
        uint64_t prog[64];
        synth(prog, COUNTS[0]);
        CHECK(rkt_chain_pack_at(&bo, td, 0, 0, 0, prog, COUNTS[0], COUNTS[0]) == 0,
              "packing a single-task chain failed");
        td[0].regcmd = BO_IOVA + BO_BYTES - 8;   /* count words would run past the end */
        CHECK(rkt_chain_verify(&bo, td, 1) != 0, "verify ACCEPTED a task running past the BO");
        td[0].regcmd = BO_IOVA - 8;              /* and one below the BO base */
        CHECK(rkt_chain_verify(&bo, td, 1) != 0, "verify ACCEPTED a task below the BO base");
    }
    printf("[ok] a task outside the BO is caught\n");

    free(mem);
    printf(fails ? "\n%d CHECK(s) FAILED\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
