// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * cube_transpose_gate.c — the plane <-> cube-group moves in rocket_cube.h against the
 * per-channel loop they replace. Host only, no device.
 *
 * Every helper runs at every channel count 1..G and at pixel counts that exercise the
 * vector body, its scalar tail and both together (0, 1, one short of a block, a block,
 * one past, and a large odd count). The fill has no period that divides a block. The
 * destination is prefilled with a sentinel, so a lane or pixel the helper should write
 * and did not fails, and so does one it should leave alone (the planes past nc and past
 * npix on the way out).
 *
 * Exit 0 when every case matches, 1 otherwise.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_cube.h"

#define SENT8  0xA5u
#define SENT16 0xA5A5u
#define SENT32 ((int32_t)0xA5A5A5A5)

static uint32_t rng = 20260927u;
static uint32_t next(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static const size_t NPIX[] = { 0, 1, 7, 15, 16, 17, 31, 33, 1037 };
#define NN (sizeof NPIX / sizeof NPIX[0])

static int check_u8_in(void)
{
    int bad = 0;
    for (int nc = 1; nc <= 16; nc++)
        for (size_t t = 0; t < NN; t++) {
            size_t n = NPIX[t];
            uint8_t *pl[16], *got = malloc(n * 16 + 32), *ref = malloc(n * 16 + 32);
            for (int j = 0; j < 16; j++) {
                pl[j] = malloc(n + 1);
                for (size_t p = 0; p < n; p++) pl[j][p] = (uint8_t)next();
            }
            memset(got, SENT8, n * 16 + 32);
            memset(ref, SENT8, n * 16 + 32);
            for (size_t p = 0; p < n; p++)
                for (int j = 0; j < 16; j++) ref[p * 16 + j] = j < nc ? pl[j][p] : 0;
            rc_planes_to_group16_u8(got, (const uint8_t *const *)pl, nc, n);
            if (memcmp(got, ref, n * 16 + 32)) { printf("u8 in  nc=%d npix=%zu FAIL\n", nc, n); bad++; }
            for (int j = 0; j < 16; j++) free(pl[j]);
            free(got); free(ref);
        }
    return bad;
}

static int check_u8_out(void)
{
    int bad = 0;
    for (int nc = 1; nc <= 16; nc++)
        for (size_t t = 0; t < NN; t++) {
            size_t n = NPIX[t];
            uint8_t *src = malloc(n * 16 + 1), *got[16], *ref[16];
            for (size_t i = 0; i < n * 16; i++) src[i] = (uint8_t)next();
            for (int j = 0; j < 16; j++) {
                got[j] = malloc(n + 8); ref[j] = malloc(n + 8);
                memset(got[j], SENT8, n + 8); memset(ref[j], SENT8, n + 8);
                if (j < nc) for (size_t p = 0; p < n; p++) ref[j][p] = src[p * 16 + j];
            }
            rc_group16_to_planes_u8(got, nc, src, n);
            for (int j = 0; j < 16; j++)
                if (memcmp(got[j], ref[j], n + 8)) { printf("u8 out nc=%d npix=%zu plane %d FAIL\n", nc, n, j); bad++; break; }
            for (int j = 0; j < 16; j++) { free(got[j]); free(ref[j]); }
            free(src);
        }
    return bad;
}

static int check_u16_in(void)
{
    int bad = 0;
    for (int nc = 1; nc <= 8; nc++)
        for (size_t t = 0; t < NN; t++) {
            size_t n = NPIX[t];
            uint16_t *pl[8], *got = malloc((n * 8 + 16) * 2), *ref = malloc((n * 8 + 16) * 2);
            for (int j = 0; j < 8; j++) {
                pl[j] = malloc((n + 1) * 2);
                for (size_t p = 0; p < n; p++) pl[j][p] = (uint16_t)next();
            }
            for (size_t i = 0; i < n * 8 + 16; i++) got[i] = ref[i] = SENT16;
            for (size_t p = 0; p < n; p++)
                for (int j = 0; j < 8; j++) ref[p * 8 + j] = j < nc ? pl[j][p] : 0;
            rc_planes_to_group8_u16(got, (const uint16_t *const *)pl, nc, n);
            if (memcmp(got, ref, (n * 8 + 16) * 2)) { printf("u16 in  nc=%d npix=%zu FAIL\n", nc, n); bad++; }
            for (int j = 0; j < 8; j++) free(pl[j]);
            free(got); free(ref);
        }
    return bad;
}

static int check_u16_out(void)
{
    int bad = 0;
    for (int nc = 1; nc <= 8; nc++)
        for (size_t t = 0; t < NN; t++) {
            size_t n = NPIX[t];
            uint16_t *src = malloc((n * 8 + 1) * 2), *got[8], *ref[8];
            for (size_t i = 0; i < n * 8; i++) src[i] = (uint16_t)next();
            for (int j = 0; j < 8; j++) {
                got[j] = malloc((n + 8) * 2); ref[j] = malloc((n + 8) * 2);
                for (size_t p = 0; p < n + 8; p++) got[j][p] = ref[j][p] = SENT16;
                if (j < nc) for (size_t p = 0; p < n; p++) ref[j][p] = src[p * 8 + j];
            }
            rc_group8_to_planes_u16(got, nc, src, n);
            for (int j = 0; j < 8; j++)
                if (memcmp(got[j], ref[j], (n + 8) * 2)) { printf("u16 out nc=%d npix=%zu plane %d FAIL\n", nc, n, j); bad++; break; }
            for (int j = 0; j < 8; j++) { free(got[j]); free(ref[j]); }
            free(src);
        }
    return bad;
}

static int check_i32_out(void)
{
    int bad = 0;
    for (int nc = 1; nc <= 4; nc++)
        for (size_t t = 0; t < NN; t++) {
            size_t n = NPIX[t];
            int32_t *src = malloc((n * 4 + 1) * 4), *got[4], *ref[4];
            for (size_t i = 0; i < n * 4; i++) src[i] = (int32_t)(next() * 257u);
            for (int j = 0; j < 4; j++) {
                got[j] = malloc((n + 8) * 4); ref[j] = malloc((n + 8) * 4);
                for (size_t p = 0; p < n + 8; p++) got[j][p] = ref[j][p] = SENT32;
                if (j < nc) for (size_t p = 0; p < n; p++) ref[j][p] = src[p * 4 + j];
            }
            rc_group4_to_planes_i32(got, nc, src, n);
            for (int j = 0; j < 4; j++)
                if (memcmp(got[j], ref[j], (n + 8) * 4)) { printf("i32 out nc=%d npix=%zu plane %d FAIL\n", nc, n, j); bad++; break; }
            for (int j = 0; j < 4; j++) { free(got[j]); free(ref[j]); }
            free(src);
        }
    return bad;
}

int main(void)
{
    int bad = check_u8_in() + check_u8_out() + check_u16_in() + check_u16_out() + check_i32_out();
    const int cases = (16 + 16 + 8 + 8 + 4) * (int)NN;
    printf("cube_transpose_gate: %d of %d cases match the per-channel loop -> %s\n",
           cases - bad, cases, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
