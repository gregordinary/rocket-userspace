// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_mm_fp16_vendor_gate.c — the RK3576 matmul-form fp16 program, word for word
 * against the programs Rockchip's compiler emits.
 *
 * RKLLM-Toolkit compiles an LLM's fp16 attention matmuls for the RK3576 to the program
 * gen_gemm_fp16_rk3576() emits, and a .rkllm file stores every compiled program verbatim
 * as 64-bit register commands. This finds each one in a .rkllm, reads its shape out of
 * its own words, emits the library's program at that shape and at the program's own
 * buffer addresses, and compares the two.
 *
 *   - A program is an 8-byte-aligned run of words whose blocks and registers, in order,
 *     are those of the library's program before its trailer, and which writes the fp16
 *     operand type (CORE 0x3018) and the fp32 output stage (DPU 0x4010). A run with the
 *     same registers and other values in those two is another program, and is counted.
 *   - A program that reads the activation in the other mode (CNA 0x100C bit 29 clear) is
 *     counted and not compared.
 *   - In the CBUF arrangement the library emits (CNA 0x1018, 0x1038, 0x103C, 0x1040),
 *     every word must match. In another arrangement, every word but those must match.
 *   - The four words after a program are its PC trailer. The last task of a chain
 *     carries the library's trailer exactly. A chained task carries its successor's
 *     address and register amount in the first two words, which the driver writes for
 *     the library, so only its last two are compared.
 *
 * No .rkllm is in this tree. Name one to run the gate:
 *
 *   rk3576_mm_fp16_vendor_gate FILE.rkllm
 *
 * Two files pass, with no mismatch:
 *
 *   - SmolLM2-1.7B-Instruct_RK3576_w4a16_g128.rkllm, compiled for this check from
 *     huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct (revision 31b70e2e, Apache-2.0)
 *     with RKLLM-Toolkit 1.3.0, following the toolkit's export example: load_huggingface on
 *     the CPU at float32, then build(quantized_dtype='w4a16_g128', quantized_algorithm=
 *     'normal', target_platform='rk3576', num_npu_core=2, optimization_level=1,
 *     dataset=None, max_context=4096). sha256
 *     fd03e5e015ce236b01e6d5b5ab12edd7c18f3b903a304168fdb29653fcb96189. 2116 programs at
 *     867 shapes in the library's arrangement, 2840 in others.
 *   - Llama-3.2-1B-Instruct_RK3576_w4a16_g128.rkllm in
 *     huggingface.co/HanzoHuang/Llama-3.2-1B-Instruct-RKLLM (RKLLM-Toolkit 1.3.0, sha256
 *     bd6cbbf84e90b63d451e672a542858a5a6aa15ad2b45f83e1b5dbcd944ac3018), the file the
 *     table was first read from. 3104 programs at 1267 shapes, 4340 in others.
 *
 * Exit 0 when every compared program matches and at least one is in the library's
 * arrangement, 1 otherwise, 2 with no file named.
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"

#define NP     (RK3576_MM_FP16_OPS - 4)    /* words before the trailer */
#define KEY(w) ((w) & 0xFFFF00000000FFFFull)
#define VAL(w) ((uint32_t)((w) >> 16))

struct shape { unsigned m, k, n; };
struct shapes { struct shape *s; size_t n, cap; };

static int push(struct shapes *v, unsigned m, unsigned k, unsigned n)
{
    if (v->n == v->cap) {
        size_t cap = v->cap ? 2 * v->cap : 1024;
        struct shape *s = realloc(v->s, cap * sizeof *s);
        if (!s) return -1;
        v->s = s; v->cap = cap;
    }
    v->s[v->n++] = (struct shape){ m, k, n };
    return 0;
}

static int cmp_shape(const void *a, const void *b)
{
    const struct shape *x = a, *y = b;
    if (x->m != y->m) return x->m < y->m ? -1 : 1;
    if (x->k != y->k) return x->k < y->k ? -1 : 1;
    return x->n == y->n ? 0 : x->n < y->n ? -1 : 1;
}

/* Sorts the shapes and prints how many distinct ones there are, with each axis's range. */
static void report_shapes(const char *what, struct shape *s, size_t n)
{
    size_t i, distinct = 0;
    unsigned lo[3] = { ~0u, ~0u, ~0u }, hi[3] = { 0, 0, 0 };
    if (!n) { printf("  %s: none\n", what); return; }
    qsort(s, n, sizeof *s, cmp_shape);
    for (i = 0; i < n; i++) {
        unsigned v[3] = { s[i].m, s[i].k, s[i].n };
        if (!i || cmp_shape(&s[i], &s[i - 1])) distinct++;
        for (int a = 0; a < 3; a++) {
            if (v[a] < lo[a]) lo[a] = v[a];
            if (v[a] > hi[a]) hi[a] = v[a];
        }
    }
    printf("  %s: %zu programs at %zu shapes, M %u-%u, K %u-%u, N %u-%u\n", what, n,
           distinct, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
}

/* The index of a register of a block in the library's program, or -1. */
static int at(const uint64_t *ref, uint16_t target, uint16_t reg)
{
    for (int i = 0; i < NP; i++)
        if ((uint16_t)(ref[i] >> 48) == target && (uint16_t)(ref[i] & 0xFFFF) == reg)
            return i;
    return -1;
}

int main(int argc, char **argv)
{
    uint64_t ref[RK3576_MM_FP16_OPS], ours[RK3576_MM_FP16_OPS];
    int i3018, i4010, i100c, i1020, i1024, i102c, i1088, i1110, i4018;
    int cfg[NP] = { 0 };
    const uint64_t *w;
    size_t nw, i, n_otherprog = 0, n_mode = 0, n_last = 0, n_chain = 0, bad = 0;
    struct shapes same = { 0 }, other = { 0 };
    struct stat st;
    int fd, j;

    printf("== RK3576 fp16 matmul against Rockchip's compiled programs ==\n");
    if (argc < 2) {
        printf("rk3576_mm_fp16_vendor_gate: no .rkllm named (the tree carries none) — "
               "SKIP\n");
        return 2;
    }
    if (gen_gemm_fp16_rk3576_unchecked(ref, 4, 64, 32, 0, 0, 0) != RK3576_MM_FP16_OPS) {
        printf("  FAIL   the generator refused its own reference shape\n");
        return 1;
    }
    i3018 = at(ref, TARGET_CORE, 0x3018); i4010 = at(ref, TARGET_DPU, 0x4010);
    i100c = at(ref, TARGET_CNA, 0x100C);  i1020 = at(ref, TARGET_CNA, 0x1020);
    i1024 = at(ref, TARGET_CNA, 0x1024);  i102c = at(ref, TARGET_CNA, 0x102C);
    i1088 = at(ref, TARGET_CNA, 0x1088);  i1110 = at(ref, TARGET_CNA, 0x1110);
    i4018 = at(ref, TARGET_DPU, 0x4018);
    if (i3018 < 0 || i4010 < 0 || i100c < 0 || i1020 < 0 || i1024 < 0 || i102c < 0 ||
        i1088 < 0 || i1110 < 0 || i4018 < 0) {
        printf("  FAIL   the library's program lacks a register this gate reads\n");
        return 1;
    }
    /* The CBUF arrangement: every CNA write to these, 0x1038 included both times. */
    for (j = 0; j < NP; j++) {
        uint16_t reg = (uint16_t)(ref[j] & 0xFFFF);
        cfg[j] = (uint16_t)(ref[j] >> 48) == TARGET_CNA &&
                 (reg == 0x1018 || reg == 0x1038 || reg == 0x103C || reg == 0x1040);
    }

    fd = open(argv[1], O_RDONLY);
    if (fd < 0 || fstat(fd, &st) != 0) { perror(argv[1]); return 1; }
    nw = (size_t)st.st_size / 8;
    w = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (w == MAP_FAILED) { perror("mmap"); return 1; }

    for (i = 0; i + RK3576_MM_FP16_OPS <= nw; i++) {
        const uint64_t *p = w + i, *t = w + i + NP;
        unsigned m, k, n;
        int same_cfg = 1, diff = -1;

        if (KEY(p[0]) != KEY(ref[0])) continue;
        for (j = 1; j < NP && KEY(p[j]) == KEY(ref[j]); j++) ;
        if (j < NP) continue;
        if (VAL(p[i3018]) != VAL(ref[i3018]) || VAL(p[i4010]) != VAL(ref[i4010])) {
            n_otherprog++;
            i += NP - 1;
            continue;
        }
        if (!(VAL(p[i100c]) & (1u << 29))) { n_mode++; i += NP - 1; continue; }
        m = VAL(p[i102c]) + 1; k = VAL(p[i1020]) / 2; n = VAL(p[i1024]) + 1;
        if (gen_gemm_fp16_rk3576_unchecked(ours, m, k, n, VAL(p[i1088]), VAL(p[i1110]),
                                           VAL(p[i4018])) != RK3576_MM_FP16_OPS) {
            if (bad++ < 8)
                printf("  FAIL   at word %zu: the library refuses the program's shape "
                       "%ux%ux%u\n", i, m, k, n);
            i += NP - 1;
            continue;
        }
        for (j = 0; j < NP; j++)
            if (cfg[j] && ours[j] != p[j]) same_cfg = 0;
        for (j = 0; j < NP && diff < 0; j++)
            if ((same_cfg || !cfg[j]) && ours[j] != p[j]) diff = j;
        if (t[2] != ours[NP + 2] || t[3] != ours[NP + 3]) {
            diff = NP + (t[2] != ours[NP + 2] ? 2 : 3);
        } else if (t[0] == ours[NP] && t[1] == ours[NP + 1]) {
            n_last++;
        } else if ((uint16_t)(t[0] >> 48) == TARGET_PC && (uint16_t)(t[0] & 0xFFFF) == 0x0010 &&
                   (uint16_t)(t[1] >> 48) == TARGET_PC && (uint16_t)(t[1] & 0xFFFF) == 0x0014) {
            n_chain++;
        } else {
            diff = NP;
        }
        if (diff >= 0) {
            if (bad++ < 8)
                printf("  FAIL   at word %zu, %ux%ux%u: word %d is %016llx, the library's "
                       "%016llx\n", i, m, k, n, diff,
                       (unsigned long long)(diff < NP ? p[diff] : t[diff - NP]),
                       (unsigned long long)ours[diff]);
        } else if (push(same_cfg ? &same : &other, m, k, n) != 0) {
            printf("  FAIL   out of memory\n");
            return 1;
        }
        i += NP - 1;
    }

    report_shapes("match in every word, the library's CBUF arrangement", same.s, same.n);
    report_shapes("match outside the CBUF words, another arrangement", other.s, other.n);
    printf("  trailers: %zu the last task's (equal to the library's), %zu chained\n",
           n_last, n_chain);
    printf("  not compared: %zu with the same registers and another operand type or "
           "output stage, %zu in the other activation mode\n", n_otherprog, n_mode);
    printf("  mismatches: %zu\n", bad);
    munmap((void *)w, (size_t)st.st_size);
    close(fd);
    if (!same.n) printf("  FAIL   no program in the library's arrangement was found\n");
    printf("== %s ==\n", bad || !same.n ? "FAIL" : "PASS");
    free(same.s); free(other.s);
    return bad || !same.n ? 1 : 0;
}
