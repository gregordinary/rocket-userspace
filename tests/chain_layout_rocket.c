// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/* chain_layout_rocket.c — host-only (no NPU): verify the assumptions the batched
 * (chained) submit layout rests on, so a packing bug is caught before it reaches
 * the hardware (where a bad stream length walks the PC through live memory).
 *
 * It generates real matmul regcmds and checks, independently of rocket_matmul.c:
 *   1. the regcmd word count is EVEN (so the chained per-task stride == the count
 *      and the PC's per-task advance lands exactly on the next segment),
 *   2. the count is UNIFORM across tile shapes (a batch shares one stride),
 *   3. the trailer PC_REGISTER_AMOUNTS op exists, is the LAST PC write, and holds
 *      0 by default (a single-task regcmd's end-of-task marker),
 *   4. a contiguous 3-task buffer places each task's first op (DPU_S_POINTER) at
 *      offset i*stride — i.e. the layout is self-consistent,
 *   5. the chained trailer rewrites land the encoded next-length and the next
 *      task's address in the value fields without disturbing op/reg or the magic.
 *
 * Build (host, no hardware):
 *   gcc -O2 -Iinclude -D__fp16=_Float16 tests/chain_layout_rocket.c \
 *       src/npu_regcmd.c -o chain_layout_rocket
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "npu_hw.h"
#include "npu_matmul.h"

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

/* Mirror of the library's chain math (independent reimplementation on purpose: a
 * test that calls the impl cannot catch an impl bug). */
static size_t chain_words(uint32_t n)   { return (size_t)((n + 1u) & ~1u); }
static uint32_t amount_encode(uint32_t n){ return (n + 1u) / 2u - 1u; }

/* Index of the trailer PC_REGISTER_AMOUNTS op (the single OP_REG_PC write to
 * 0x14), scanning from the end. -1 if absent. */
static int find_trailer(const uint64_t *rc, int n) {
    for (int j = n - 1; j >= 0; j--)
        if ((uint16_t)(rc[j] >> 48) == OP_REG_PC &&
            (uint16_t)(rc[j] & 0xffff) == PC_REGISTER_AMOUNTS)
            return j;
    return -1;
}

/* Generate a single TILE's regcmd (the granularity the tiler hands to gen — never
 * the whole matrix) for the requested dtype. Returns the op count, or 0 if gen
 * rejected the shape. dt: 0=fp16, 1=int8, 2=int4 — all three route through the
 * shared gen_matmul_task trailer-emitter, so the chained layout must hold for each. */
static uint32_t gen_dt(uint64_t *ops, int dt, int M, int K, int N) {
    matmul_params_t p = { .m = M, .k = K, .n = N, .input_dma = 0x1000,
        .weights_dma = 0x2000, .output_dma = 0x3000, .tasks = ops, .fp32tofp16 = 1 };
    int r = (dt == 1) ? gen_matmul_int8(&p)
          : (dt == 2) ? gen_matmul_int4(&p)
                      : gen_matmul_fp16(&p);
    return r ? 0 : p.task_count;
}

/* fp16 alias for the original (thorough) fp16 body below. */
static uint32_t gen(uint64_t *ops, int M, int K, int N) { return gen_dt(ops, 0, M, K, N); }

/* Validate the chained layout assumptions for one dtype against three real tile
 * shapes. Mirrors the matmul (fp16) checks but parameterized so int8/int4 — whose
 * gen op count differs — are gated off-device before they reach the hardware. */
static void check_dtype(const char *name, int dt, int M0, int K0, int N0,
                        int M1, int K1, int N1, int M2, int K2, int N2) {
    uint64_t a[256] = {0}, b[256] = {0}, c[256] = {0};
    uint32_t na = gen_dt(a, dt, M0, K0, N0);
    uint32_t nb = gen_dt(b, dt, M1, K1, N1);
    uint32_t nc = gen_dt(c, dt, M2, K2, N2);
    printf("[%s] regcmd word counts: %u, %u, %u\n", name, na, nb, nc);
    CHECK(na && nb && nc, "[%s] a tile shape was rejected by gen (%u/%u/%u)", name, na, nb, nc);
    if (!(na && nb && nc)) return;

    CHECK(na % 2 == 0, "[%s] regcmd count %u is odd; chained stride != advance", name, na);
    CHECK(na == nb && nb == nc, "[%s] regcmd count not uniform across shapes (%u/%u/%u)",
          name, na, nb, nc);
    CHECK(chain_words(na) == na, "[%s] chain stride %zu != count %u for even count",
          name, chain_words(na), na);

    int t = find_trailer(a, na);
    CHECK(t >= 0, "[%s] no trailer PC_REGISTER_AMOUNTS op found", name);
    if (t < 0) return;
    CHECK(t == (int)na - 3, "[%s] trailer at idx %d, expected count-3 (%d)", name, t, na - 3);
    uint32_t val = (uint32_t)((a[t] >> 16) & 0xffffffff);
    CHECK(val == 0, "[%s] default trailer amount is %u, expected 0 (end marker)", name, val);
    uint16_t lastop = (uint16_t)(a[na - 1] >> 48);
    CHECK(lastop == OP_ENABLE, "[%s] last op is 0x%04x, expected OP_ENABLE 0x%04x",
          name, lastop, OP_ENABLE);

    /* Contiguous 3-task layout at the chained stride: each task's first op lands at
     * i*stride (gaps poisoned, so a short copy or wrong stride is caught). */
    size_t stride = chain_words(na);
    static uint64_t buf[3 * 256];
    memset(buf, 0xAB, sizeof(buf));
    const uint64_t *srcs[3] = { a, b, c };
    for (int i = 0; i < 3; i++)
        memcpy(buf + (size_t)i * stride, srcs[i], (size_t)na * sizeof(uint64_t));
    for (int i = 0; i < 3; i++) {
        uint64_t first = buf[(size_t)i * stride];
        CHECK((uint16_t)(first >> 48) == OP_REG_DPU &&
              (uint16_t)(first & 0xffff) == DPU_S_POINTER,
              "[%s] task %d first op = (0x%04x,0x%04x), expected DPU_S_POINTER",
              name, i, (uint16_t)(first >> 48), (uint16_t)(first & 0xffff));
    }

    /* The OP_NONE filler at count-4 is the slot the PC_BASE_ADDRESS redirect claims;
     * confirm it is inert and that OP_40 follows the amount op (the trailer shape the
     * chain rewrite depends on, identical across dtypes via gen_matmul_task). */
    CHECK((uint16_t)(a[t - 1] >> 48) == OP_NONE,
          "[%s] trailer slot count-4 is 0x%04x, expected OP_NONE filler",
          name, (uint16_t)(a[t - 1] >> 48));
    CHECK((uint16_t)(a[t + 1] >> 48) == OP_40, "[%s] op after amount is not OP_40 magic", name);
    printf("[%s] contiguous stride=%zu words, trailer idx=%d, layout OK\n", name, stride, t);
}

int main(void) {
    /* Tile shapes the planner actually emits: a full interior tile and the two
     * smaller boundary tiles (M%4==0, K%32, N%16). Op count is data-independent,
     * so all must agree. */
    uint64_t a[256] = {0}, b[256] = {0}, c[256] = {0};
    uint32_t na = gen(a, 64, 512, 64);     /* interior tile  */
    uint32_t nb = gen(b, 4, 64, 16);       /* tiny boundary  */
    uint32_t nc = gen(c, 128, 256, 32);    /* another shape  */
    printf("regcmd word counts: %u, %u, %u\n", na, nb, nc);
    CHECK(na && nb && nc, "a tile shape was rejected by gen (%u/%u/%u)", na, nb, nc);

    CHECK(na % 2 == 0, "fp16 regcmd count %u is odd; chained stride != advance", na);
    CHECK(na == nb && nb == nc, "regcmd count not uniform across shapes (%u/%u/%u)", na, nb, nc);
    CHECK(chain_words(na) == na, "chain stride %zu != count %u for even count", chain_words(na), na);

    int t = find_trailer(a, na);
    CHECK(t >= 0, "no trailer PC_REGISTER_AMOUNTS op found");
    if (t >= 0) {
        CHECK(t == (int)na - 3, "trailer at idx %d, expected count-3 (%d)", t, na - 3);
        uint32_t val = (uint32_t)((a[t] >> 16) & 0xffffffff);
        CHECK(val == 0, "default trailer amount is %u, expected 0 (end marker)", val);
        /* the OP_ENABLE must be the very last op */
        uint16_t lastop = (uint16_t)(a[na - 1] >> 48);
        CHECK(lastop == OP_ENABLE, "last op is 0x%04x, expected OP_ENABLE 0x%04x", lastop, OP_ENABLE);
    }

    /* Build a 3-task contiguous buffer at the chained stride and confirm each
     * task's first op (DPU_S_POINTER, value 0xE @ 0x4004) is at i*stride. */
    size_t stride = chain_words(na);
    static uint64_t buf[3 * 256];
    memset(buf, 0xAB, sizeof(buf));     /* poison the gaps; chained packing must overwrite */
    const uint64_t *srcs[3] = { a, b, c };
    for (int i = 0; i < 3; i++)
        memcpy(buf + (size_t)i * stride, srcs[i], (size_t)na * sizeof(uint64_t));
    for (int i = 0; i < 3; i++) {
        uint64_t first = buf[(size_t)i * stride];
        uint16_t op  = (uint16_t)(first >> 48);
        uint16_t reg = (uint16_t)(first & 0xffff);
        CHECK(op == OP_REG_DPU && reg == DPU_S_POINTER,
              "task %d first op = (0x%04x,0x%04x), expected DPU_S_POINTER", i, op, reg);
    }
    printf("contiguous layout: stride=%zu words (%zu bytes), 3 tasks span %zu bytes\n",
           stride, stride * 8, 3 * stride * 8);

    /* Chained trailer rewrite: encode(na) into the value field, op/reg intact. */
    if (t >= 0) {
        uint64_t before = a[t];
        uint32_t amt = amount_encode(na);
        a[t] = ((uint64_t)OP_REG_PC << 48) | ((uint64_t)amt << 16) | PC_REGISTER_AMOUNTS;
        CHECK((uint16_t)(a[t] >> 48) == OP_REG_PC, "rewrite clobbered op field");
        CHECK((uint16_t)(a[t] & 0xffff) == PC_REGISTER_AMOUNTS, "rewrite clobbered reg field");
        CHECK((uint32_t)((a[t] >> 16) & 0xffffffff) == amt, "rewrite value wrong");
        printf("chained encoded amount = %u (0x%04x); trailer 0x%016llx -> 0x%016llx\n",
               amt, amt, (unsigned long long)before, (unsigned long long)a[t]);
    }

    /* Chained base-address op: the OP_NONE just before the amount op (count-4) is
     * the inert filler we repurpose into a PC_BASE_ADDRESS write. Confirm it is
     * OP_NONE, then that the rewrite lands a raw next-address in the value field
     * with op=OP_REG_PC reg=PC_BASE_ADDRESS — and does NOT disturb the amount op
     * or the OP_40 magic / OP_ENABLE that follow. */
    if (t >= 1) {
        CHECK((uint16_t)(b[t - 1] >> 48) == OP_NONE,
              "trailer slot count-4 is 0x%04x, expected OP_NONE filler", (uint16_t)(b[t - 1] >> 48));
        CHECK((uint16_t)(b[t + 1] >> 48) == OP_40, "op after amount is not OP_40 magic");
        uint32_t next = 0x2000 + (uint32_t)(chain_words(nb) * 8);   /* task-1 IOVA in a 2-task chain */
        b[t - 1] = ((uint64_t)OP_REG_PC << 48) | ((uint64_t)next << 16) | PC_BASE_ADDRESS;
        CHECK((uint16_t)(b[t - 1] >> 48) == OP_REG_PC, "base rewrite clobbered op field");
        CHECK((uint16_t)(b[t - 1] & 0xffff) == PC_BASE_ADDRESS, "base rewrite clobbered reg field");
        CHECK((uint32_t)((b[t - 1] >> 16) & 0xffffffff) == next, "base rewrite value wrong");
        printf("chained base op -> next IOVA 0x%08x; slot count-4 = 0x%016llx\n",
               next, (unsigned long long)b[t - 1]);
    }

    /* int8 / int4 share gen_matmul_task's trailer, so the chained LAYOUT holds for
     * them too (validated below). NOTE: layout-valid does NOT mean execution-safe —
     * the integer datapath is HW-blocked from chaining (the int32 CACC clears
     * per-kick not per-task, so chained integer tasks garble; the resident int8/int4
     * paths force gapped, see rocket_prepacked_int8.c). This check guards the layout
     * math only, used by any FUTURE integer-chain attempt that first cracks the CACC
     * clear. fp16 is the only dtype that chains end-to-end today. */
    printf("\n");
    check_dtype("int8", 1, 64, 256, 64,  4, 64, 64,  128, 256, 32);
    check_dtype("int4", 2, 64, 256, 64,  4, 64, 64,  128, 256, 32);

    printf(fails ? "\n%d CHECK(s) FAILED\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
