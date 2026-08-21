// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_argb_padcon.c — what value does a PAD TAP feed the MAC on the packed-image
 * first conv, and what does `CNA_PAD_CON1` have to hold to make it a chosen one?
 *
 * THE OPEN ITEM THIS ANSWERS. The packed-image (ARGB) int8 first conv is bit-exact in
 * its interior at any input zero point and wrong on its padded taps the moment that zero
 * point leaves zero, so the entry refuses a non-zero one. That refusal is what keeps
 * ResNet-18's 7x7 stem — `pad_left` 3, `in_zp` -128 — on the direct lowering, where the
 * packed program is worth about 2.1 ms of a 9.3 ms graph. The earlier probe established
 * that `CNA_PAD_CON1` is LIVE (driving it moves the count) and that no single centring of
 * it explained the readings, and stopped there.
 *
 * A NEGATIVE ABOUT REACHABILITY IS THE ONE TO RE-OPEN, and a pass/fail sweep is the weak
 * form of the question: it asks "is there a value that works" and returns nothing when the
 * answer is no. This file DECODES instead — it reads the pad tap's value off the surface,
 * one number per programmed word, so the map from the register to the datapath comes out
 * whatever it is, including "the three image channels do not read three bytes".
 *
 * HOW A PAD TAP IS MADE READABLE. One output channel, one non-zero weight: channel `c`,
 * kernel tap (0,0), value 1, everything else zero. At k3 s1 with a leading pad of 1,
 * output (y,x) tap (0,0) reads input (y-1, x-1), so
 *
 *     output (0,0)  reads the CORNER pad     output (0,1)  reads the TOP pad
 *     output (1,0)  reads the LEFT pad       output (1,1)  reads image (0,0)
 *
 * With `in_zp` 0, `w_zp` 0, no bias and unit scales the coefficient fold is zero and the
 * requant gain is one, so each of those outputs IS the sample the MAC saw, clamped to
 * int8. Output (1,1) is the positive control: it must come back as the image pixel, which
 * is what says the weight is where this file thinks it is and the program ran at all.
 *
 * `ROCKET_RK3576_SET=0x1084=<word>` rewrites the emitted register before submit, so the
 * sweep never enters the transcribed program. The word the emitter would have written for
 * a given `in_zp` is printed beside the sweep, so the "what is programmed today" row and
 * the "what would have to be programmed" row are read off the same table.
 *
 * Usage: rk3576_argb_padcon [channel]
 * Exit:  0 the control held and the map was read, 1 the control failed, 2 no NPU or the
 *        wrong chip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"

#define STAMP  0x5A
#define IC     3u
#define N      64u          /* plane; ow == iw at k3 s1 pad 1, and 64 is a multiple of 16 */
#define OC     32u          /* a whole 32-channel group, which this path requires */
#define K      3u

/* The words to try. Each is a hypothesis about the domain, and the first three are the
 * ones the two sub-encodings actually emit. */
struct cand { const char *why; uint32_t word; };

static int8_t *IMG;
static int8_t *W;

/* One run: program `word` into CNA_PAD_CON1, return the three pad samples and the
 * interior control. Values are what the MAC saw, since the fold and the gain are unity. */
static int probe_on(int fd, uint32_t word, unsigned c, int direct,
                    int *corner, int *top, int *left, int *interior);

static int probe(int fd, uint32_t word, unsigned c,
                 int *corner, int *top, int *left, int *interior)
{
    return probe_on(fd, word, c, 0, corner, top, left, interior);
}

static int probe_on(int fd, uint32_t word, unsigned c, int direct,
                    int *corner, int *top, int *left, int *interior)
{
    rocket_conv2d_desc d;
    char spec[64];
    int8_t *out;
    int rc;

    memset(&d, 0, sizeof d);
    d.direct_datapath = direct;
    d.ic = (int)IC; d.oc = (int)OC; d.ih = (int)N; d.iw = (int)N;
    d.kh = (int)K; d.kw = (int)K;
    d.stride_y = 1; d.stride_x = 1;
    d.dil_y = d.dil_x = 1;
    d.pad_top = 1; d.pad_left = 1;

    out = malloc((size_t)OC * N * N);
    if (!out) return ROCKET_E_NOMEM;
    memset(out, STAMP, (size_t)OC * N * N);

    memset(W, 0, (size_t)OC * IC * K * K);
    W[((size_t)0 * IC + c) * K * K + 0] = 1;      /* oc 0, channel c, tap (0,0) */

    snprintf(spec, sizeof spec, "0x1084=0x%08X", word);
    setenv("ROCKET_RK3576_SET", spec, 1);
    rc = rocket_conv2d_int8_rk3576(fd, &d, IMG, W, NULL, 1.0f, 1.0f, 1.0f,
                                   0, 0, 0, out);
    unsetenv("ROCKET_RK3576_SET");
    if (rc != ROCKET_OK) { free(out); return rc; }

    *corner   = out[0 * N * N + 0 * N + 0];
    *top      = out[0 * N * N + 0 * N + 1];
    *left     = out[0 * N * N + 1 * N + 0];
    *interior = out[0 * N * N + 1 * N + 1];
    free(out);
    return ROCKET_OK;
}

int main(int argc, char **argv)
{
    unsigned c = argc > 1 ? (unsigned)atoi(argv[1]) : 0u;
    struct cand cands[40];
    unsigned n = 0, i;
    int fd, failed = 0, ctl_want;
    static const int BYTES[] = { 0x00, 0x01, 0x40, 0x7F, 0x80, 0x81, 0xC0, 0xFF };

    if (c >= IC) { fprintf(stderr, "channel must be 0..%u\n", IC - 1u); return 2; }

    if (strcmp(rocket_hw_current()->name, "rk3576") != 0) {
        printf("rk3576_argb_padcon: profile is %s, not rk3576 — skipping\n",
               rocket_hw_current()->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_argb_padcon: no NPU device — skipping\n"); return 2; }

    IMG = malloc((size_t)IC * N * N);
    W   = malloc((size_t)OC * IC * K * K);
    if (!IMG || !W) { free(IMG); free(W); rocket_close(fd); return 2; }
    /* A distinct sample per channel at (0,0), so the interior control also says WHICH
     * channel's lane the weight landed on. */
    for (i = 0; i < IC * N * N; i++) IMG[i] = (int8_t)((i * 7u + 3u) & 0x3F);
    IMG[(size_t)0 * N * N] = 11; IMG[(size_t)1 * N * N] = 22; IMG[(size_t)2 * N * N] = 33;
    ctl_want = IMG[(size_t)c * N * N];

    /* Replicated over the three live lanes — what the emitter writes today. */
    for (i = 0; i < sizeof BYTES / sizeof *BYTES; i++) {
        uint32_t b = (uint32_t)BYTES[i];
        cands[n].why = "replicated x3";
        cands[n++].word = b | (b << 8) | (b << 16);
    }
    /* The same byte in ONE lane, to see whether channel `c` reads lane `c` at all. */
    for (i = 0; i < 4; i++) {
        cands[n].why = "0x7F in lane 0/1/2/3 only";
        cands[n++].word = 0x7Fu << (8 * i);
    }
    /* ALL FOUR LANES. A packed image of three channels leaves the fourth ARGB lane
     * unused, and every candidate above leaves it ZERO — so a column that reads it would
     * be indistinguishable from one that reads nothing. */
    cands[n].why = "replicated x4";
    cands[n++].word = 0x7F7F7F7Fu;
    cands[n].why = "x3 live, 0x40 in lane 3";
    cands[n++].word = 0x407F7F7Fu;

    printf("== packed-image first conv: what a PAD TAP feeds the MAC ==\n");
    printf("   %ux%u k%u s1 pad 1, ic=%u oc=%u, in_zp 0, unit scales — so an output IS "
           "the sample\n", N, N, K, IC, OC);
    printf("   weight: output channel 0, image channel %u, tap (0,0) = 1; all else 0\n", c);
    printf("   the emitter writes 0x%08X at in_zp 0 and 0x%08X at in_zp -128\n\n",
           0x00808080u, (uint32_t)((0x00u) | (0x00u << 8) | (0x00u << 16)));
    printf("   %-24s %-12s   corner    top   left | interior (want %d)\n",
           "hypothesis", "word", ctl_want);

    for (i = 0; i < n; i++) {
        int corner, top, left, interior;
        int rc = probe(fd, cands[i].word, c, &corner, &top, &left, &interior);
        if (rc != ROCKET_OK) {
            printf("   %-24s 0x%08X   REFUSED (%d)\n", cands[i].why, cands[i].word, rc);
            failed++;
            continue;
        }
        printf("   %-24s 0x%08X   %6d %6d %6d | %6d%s\n",
               cands[i].why, cands[i].word, corner, top, left, interior,
               interior == ctl_want ? "" : "   <- CONTROL FAILED");
        if (interior != ctl_want) failed++;
        /* THE SAME QUESTION ON THE DIRECT DATAPATH, at the same geometry and the same
         * weights. Its pad constant is measured correct at every zero point, so this arm
         * is what says whether a map read here is a property of the packed sub-encoding
         * or of this instrument. */
        rc = probe_on(fd, cands[i].word, c, 1, &corner, &top, &left, &interior);
        if (rc == ROCKET_OK)
            printf("   %-24s 0x%08X   %6d %6d %6d | %6d%s\n",
                   "   ^ same on DIRECT", cands[i].word, corner, top, left, interior,
                   interior == ctl_want ? "" : "   <- CONTROL FAILED");
        else
            printf("   %-24s 0x%08X   REFUSED (%d)\n", "   ^ same on DIRECT",
                   cands[i].word, rc);
    }

    printf("\n   The map is read off the `corner`/`top`/`left` columns: the byte the\n"
           "   datapath feeds a pad tap, against the byte programmed. A path whose pad\n"
           "   constant is usable has a value there for every int8 the caller can ask\n"
           "   for; one whose columns do not move, or move in fewer than 256 steps, does\n"
           "   not, and the entry's refusal stands.\n");

    free(IMG); free(W);
    rocket_close(fd);
    return failed ? 1 : 0;
}
