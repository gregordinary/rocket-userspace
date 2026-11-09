// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * fp16_ew_overflow_probe.c -- what does the RK3588 DPU write when an fp16 EW result passes
 * the fp16 range: +-inf (IEEE round-to-nearest-even, as the matmul's FP32TOFP16 narrowing
 * does) or +-65504 (saturation)?
 *
 * Drives the library's validated EW entries, rocket_ew_add_fp16 and rocket_ew_mul_fp16 (an
 * identity matmul main feed with the operand on the ERDMA), with operand pairs whose exact
 * result walks across the top of the fp16 range in steps finer than its last ulp:
 *   add  65472 + b, b = 0, 8, ..., 1024, and the same negated
 *   mul  explicit pairs whose products are 65472, 65504, the 65520 tie, 65536 and in-range
 *        checks, and the same negated
 * The add walk has both ties of the top binade: 65488 (between 65472 and 65504) and 65520
 * (between 65504 and the first value past the range). Each element is scored against:
 *   ieee  the exact result narrowed to fp16 with round-to-nearest-even, inf past 65519.99
 *   sat   the same narrowing clamped to +-65504
 * They agree below 65504 + 16 and differ from 65520 up. The walk also includes exact
 * in-range results, which check the operands reached the ALU.
 * Exit 0 when one model explains every element of each op, 1 otherwise, 2 without a device.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocket_activation.h"
#include "rocket_npu.h"

#define N 256

static uint16_t bits(_Float16 h) { uint16_t u; memcpy(&u, &h, 2); return u; }

static int score(const char *op, const _Float16 *a, const _Float16 *b, const _Float16 *got,
                 int n, int is_mul)
{
    int n_ieee = 0, n_sat = 0, n_none = 0, shown = 0;
    for (int i = 0; i < n; i++) {
        double exact = is_mul ? (double)a[i] * (double)b[i] : (double)a[i] + (double)b[i];
        _Float16 ieee = (_Float16)(float)exact;        /* exact in fp32 here; RNE to fp16 */
        double cl = exact > 65504.0 ? 65504.0 : exact < -65504.0 ? -65504.0 : exact;
        _Float16 sat = (_Float16)(float)cl;
        if (fabs(exact) >= 65504.0 && fabs(exact) < 65520.0) sat = ieee;   /* both 65504 */
        int mi = bits(got[i]) == bits(ieee), ms = bits(got[i]) == bits(sat);
        n_ieee += mi; n_sat += ms; n_none += !mi && !ms;
        if (fabs(exact) >= 65480.0 && fabs(exact) <= 65536.0 && shown < 12) {
            printf("    %s %9.3f -> 0x%04x (ieee 0x%04x, sat 0x%04x)\n", op, exact,
                   bits(got[i]), bits(ieee), bits(sat));
            shown++;
        }
    }
    printf("  %s: ieee %d, sat %d, neither %d, of %d\n", op, n_ieee, n_sat, n_none, n);
    return n_ieee == n || n_sat == n;
}

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no device (%d)\n", fd); return 2; }
    static _Float16 a[N], b[N], out[N];
    int ok = 1;

    for (int i = 0; i < N; i++) {
        int k = i % 128;
        double s = i < 128 ? 1.0 : -1.0;
        a[i] = (_Float16)(s * 65472.0);
        b[i] = (_Float16)(s * 8.0 * k);
    }
    memset(out, 0, sizeof out);
    if (rocket_ew_add_fp16(fd, a, b, out, N)) { printf("add failed\n"); ok = 0; }
    else ok &= score("add", a, b, out, N, 0);

    /* explicit pairs: no product of two fp16 values is the 65488 tie (16 x 4093, a prime),
     * but 45 x 1456 and 91 x 720 are the 65520 tie */
    static const double MP[][2] = {
        {256, 255.75}, {256, 255.875}, {45, 1456}, {91, 720}, {256, 256}, {2, 32752},
        {128, 511.75}, {128, 512}, {16, 4094}, {16, 4096}, {100, 100}, {3, 0.5},
    };
    const int NP = (int)(sizeof MP / sizeof MP[0]);
    for (int i = 0; i < N; i++) {
        int k = i % 128;
        double s = i < 128 ? 1.0 : -1.0;
        a[i] = (_Float16)MP[k % NP][0];
        b[i] = (_Float16)(s * MP[k % NP][1]);
    }
    memset(out, 0, sizeof out);
    if (rocket_ew_mul_fp16(fd, a, b, out, N)) { printf("mul failed\n"); ok = 0; }
    else ok &= score("mul", a, b, out, N, 1);

    rocket_close(fd);
    return ok ? 0 : 1;
}
