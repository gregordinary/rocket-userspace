// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * fa_av_dump_wrap.c — a link-time wrapper that writes one head's flash-attention AV operands
 * (the softmax P tile and the gathered V tile) and its AV result, the first time the FA handler
 * makes an AV call of the given shape. For tests/fa_replay_probe, so tests/fa_core_matmul_probe
 * can run exactly those operands on each NPU core.
 *
 * Link with the replay against the STATIC library and -Wl,--wrap=rocket_mm_batch_run: the FA
 * handler's call (rocket_attn.o -> rocket_matmul.o) then lands here. Only a call from another
 * object is redirected, so the library's internal calls are untouched.
 *
 * Env: FA_AV_DUMP_DIR  directory to write fa_av_{a,b,c}.f16 (A [M][K] = P, B [N][K] = V
 *                      gathered, C [M][N] the call's own result); unset -> a pass-through
 *      FA_AV_DUMP_ITEM the batch item to write (default 12: head 12 at one worker)
 *      FA_AV_DUMP_K    the AV contraction K to match (default 1024); N must equal the call's dv
 * It writes once per process, on the first matching call, and prints what it wrote.
 * What it does NOT capture: the QK scores, or which core ran the call.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_matmul.h"

int __real_rocket_mm_batch_run(rocket_mm_batch *b, int M, int K, int N, int nbatch,
                               const _Float16 *const *A, const _Float16 *const *B,
                               _Float16 *const *C);

static int g_done;

static void put(const char *dir, const char *tag, const void *p, size_t n)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/fa_av_%s.f16", dir, tag);
    FILE *f = fopen(path, "wb");
    size_t w = f ? fwrite(p, sizeof(_Float16), n, f) : 0;
    if (f) fclose(f);
    fprintf(stderr, "fa_av_dump_wrap: %s %zu of %zu elements\n", path, w, n);
}

int __wrap_rocket_mm_batch_run(rocket_mm_batch *b, int M, int K, int N, int nbatch,
                               const _Float16 *const *A, const _Float16 *const *B,
                               _Float16 *const *C)
{
    int rc = __real_rocket_mm_batch_run(b, M, K, N, nbatch, A, B, C);
    const char *dir = getenv("FA_AV_DUMP_DIR");
    const char *ei = getenv("FA_AV_DUMP_ITEM"), *ek = getenv("FA_AV_DUMP_K");
    const int item = ei ? atoi(ei) : 12, kk = ek ? atoi(ek) : 1024;
    if (rc == 0 && dir && *dir && !g_done && K == kk && item < nbatch) {
        g_done = 1;
        fprintf(stderr, "fa_av_dump_wrap: AV call M %d K %d N %d nbatch %d, writing item %d\n",
                M, K, N, nbatch, item);
        put(dir, "a", A[item], (size_t)M * K);
        put(dir, "b", B[item], (size_t)N * K);
        put(dir, "c", C[item], (size_t)M * N);
    }
    return rc;
}
