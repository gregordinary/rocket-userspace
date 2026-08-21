// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_argb_padmap.c — where does a pad tap's value COME FROM on the packed-image
 * first conv, as a function of the tap's POSITION rather than of the programmed word?
 *
 * THE OPEN READING THIS ANSWERS. `rk3576_argb_padcon` read the map from `CNA_PAD_CON1`
 * to the datapath — `(int8_t)byte`, per image channel — and found the top pad ROWS
 * track it while the leading COLUMNS feed 0. Against that, a dense-weight run past the
 * entry's zero-point bound is only 55-58% exact on the leading pad region AND on the
 * TRAILING one, and no leading-column story covers a wrong trailing region at all.
 *
 * WHAT THE EARLIER READOUT HELD FIXED. It swept the programmed word and the byte lane
 * and read the left pad at ONE position: output (1,0), whose tap sees input (0,-1) —
 * the row-0 left pad, which in a contiguous packed image lies BEFORE THE START OF THE
 * BUFFER. So "the columns feed a constant 0" and "the columns feed whatever is linearly
 * adjacent in the buffer" are the same observation there, and only the second covers a
 * trailing region. The packed image is `ih*iw*ic` interleaved bytes with no per-row
 * padding (rocket_rk3576_argb_pack), so the two separate the moment a column pad is read
 * at a row other than the first:
 *
 *     input (r, -1)    constant: 0        linear: the LAST pixel of row r-1
 *     input (r, iw)    constant: 0        linear: the FIRST pixel of row r+1
 *
 * HOW A PAD TAP IS MADE READABLE. One output channel, one non-zero weight, so the
 * accumulator IS the sample the MAC saw. With `w_zp` 0, no bias and unit scales the
 * coefficient fold is `-in_zp * sum_w = -in_zp`, so
 *
 *     fed value = output + in_zp
 *
 * at every position and at every zero point — which is what lets the zero-point axis be
 * crossed with the position axis instead of confounded with it, and what the interior
 * control checks. Two weight placements reach all four edges:
 *
 *     tap (0,0)      output (y,x) reads input (y-1, x-1)   — the TOP row, the LEFT column
 *     tap (k-1,k-1)  output (y,x) reads input (y+1, x+1)   — the BOTTOM row, the RIGHT column
 *
 * NOTHING HERE READS THE EMITTER FOR WHAT IS PROGRAMMED. Two branches of
 * npu_regcmd_rk3576.c compute `pad_con1` and the one that looks like the packed path's is
 * not what a packed program carries. Since the map is `(int8_t)byte` [HW sweep,
 * rk3576_argb_padcon, 8 values on both datapaths], the TOP ROW SAMPLE IS the effective
 * constant, so this file reports it and scores the other three edges against it.
 *
 * The same geometry and weights on the DIRECT datapath is the control arm: its border is
 * measured exact at every zero point, so it is what says a map read here belongs to the
 * packed sub-encoding and not to this instrument.
 *
 * THE DELTA SWEEP. If the columns are dead the entry's zero-point refusal stands, and the
 * next question is whether any register reaches them: the direct path proves SOME
 * configuration does. Section 2 flips the packed program one register at a time to the
 * direct program's value for that register and re-reads the map. It is a one-at-a-time
 * search and cannot see a condition of two registers — but every arm is a READOUT, so an
 * arm that half-works is visible where a pass/fail sweep would report nothing.
 *
 * Usage: rk3576_argb_padmap [channel]
 * Env:   ROCKET_RK3576_PADMAP_ZP=<n>  zero point for the crossed cells (default 20)
 *        ROCKET_RK3576_PADMAP_DELTA=0 skip section 2
 * Exit:  0 the controls held, 1 a control failed, 2 no NPU or the wrong chip.
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
#define N      64u          /* plane; ow == iw at k3 s1 pad 1, so the trailing pad is 1 */
#define OC     32u
#define K      3u

static int8_t *IMG;         /* planar CHW, what the caller hands the entry   */
static int8_t *PACKED;      /* the interleave the entry builds, for `linear` */
static int8_t *W;

/* Where along an edge to sample. The corners are read separately — they mix two edges.
 * Row 0 of the left column is the position the earlier readout was taken at, and is the
 * one position where `constant 0` and `the buffer's neighbour` agree. */
static const unsigned POS[] = { 1, 2, 3, 17, 32, 61, 62 };
#define NPOS (sizeof POS / sizeof *POS)

/* The registers the packed and the direct program disagree on at this geometry, read off
 * ROCKET_RK3576_DUMP=1 on both arms. The four address registers they also differ on
 * (0x4018, 0x5020, 0x5024, 0x1110) are excluded — pointing a program at another
 * program's buffers says nothing about the border. */
struct delta { uint16_t reg; uint32_t packed, direct; };
static const struct delta DELTA[] = {
    { 0x100c, 0x2000a006, 0x00000000 },
    { 0x101c, 0x00000600, 0x00002400 },
    { 0x1020, 0x00000030, 0x00000120 },
    { 0x1028, 0x0140000b, 0x0800001f },
    { 0x1030, 0x0060003f, 0x0240003f },
    { 0x103c, 0x00050000, 0x00200000 },
    { 0x1044, 0x00400005, 0x00400020 },
    { 0x1048, 0x000e38e0, 0x0000000b },
    { 0x104c, 0x40004000, 0x00010001 },
    { 0x1050, 0x00014000, 0x00010001 },
    { 0x1078, 0x000b003f, 0x003f003f },
    { 0x107c, 0x00000002, 0x0000001f },
    { 0x1090, 0x0000000c, 0x00000100 },
    { 0x1094, 0x00000300, 0x00001000 },
    { 0x1098, 0x00000300, 0x00001000 },
    { 0x118c, 0x00040003, 0x003f003f },
    { 0x3018, 0x10000081, 0x10000001 },
};
#define NDELTA (sizeof DELTA / sizeof *DELTA)

/* What one cell read: the four edges, the two corners, and the interior control. */
struct readout {
    int top, bottom, left, right;      /* the sample each edge fed, at POS[0]         */
    int lead_corner, trail_corner;
    int ctl_lead, ctl_trail;           /* interior taps — must be the image pixel     */
    int col_varies;                    /* a column sample that changed down the edge  */
    int left_at[NPOS], right_at[NPOS];
};

/* One run. `tap` selects the weight placement: 0 = (0,0), 1 = (K-1,K-1). */
static int probe(int fd, unsigned c, int direct, int tap, int in_zp,
                 const char *setword, int8_t *plane)
{
    rocket_conv2d_desc d;
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
    W[((size_t)0 * IC + c) * K * K + (tap ? (K * K - 1u) : 0u)] = 1;

    if (setword) setenv("ROCKET_RK3576_SET", setword, 1);
    rc = rocket_conv2d_int8_rk3576(fd, &d, IMG, W, NULL, 1.0f, 1.0f, 1.0f,
                                   in_zp, 0, 0, out);
    if (setword) unsetenv("ROCKET_RK3576_SET");
    if (rc == ROCKET_OK) memcpy(plane, out, (size_t)N * N);
    free(out);
    return rc;
}

static int read_cell(int fd, unsigned c, int direct, int in_zp, const char *setword,
                     struct readout *r)
{
    int8_t *lead = malloc((size_t)N * N), *trail = malloc((size_t)N * N);
    int rc, i;

    if (!lead || !trail) { free(lead); free(trail); return ROCKET_E_NOMEM; }
    memset(r, 0, sizeof *r);

    rc = probe(fd, c, direct, 0, in_zp, setword, lead);
    if (rc == ROCKET_OK) rc = probe(fd, c, direct, 1, in_zp, setword, trail);
    if (rc != ROCKET_OK) { free(lead); free(trail); return rc; }

    r->ctl_lead     = lead[1 * N + 1] + in_zp;      /* reads image (0,0) */
    r->ctl_trail    = trail[0] + in_zp;             /* reads image (1,1) */
    r->lead_corner  = lead[0] + in_zp;
    r->trail_corner = trail[(N - 1) * N + (N - 1)] + in_zp;
    r->top          = lead[0 * N + POS[0]] + in_zp;
    r->left         = lead[POS[0] * N + 0] + in_zp;
    r->bottom       = trail[(N - 1) * N + POS[0]] + in_zp;
    r->right        = trail[POS[0] * N + (N - 1)] + in_zp;
    for (i = 0; i < (int)NPOS; i++) {
        r->left_at[i]  = lead[POS[i] * N + 0] + in_zp;
        r->right_at[i] = trail[POS[i] * N + (N - 1)] + in_zp;
        if (r->left_at[i] != r->left || r->right_at[i] != r->right) r->col_varies = 1;
    }
    free(lead); free(trail);
    return ROCKET_OK;
}

/* One line per cell: what each edge fed, and the verdict. `want` is the value a correct
 * border feeds — the input zero point, since that is the raw byte standing for a real
 * zero once the coefficient fold has subtracted it. */
static void report(const char *label, int in_zp, const struct readout *r,
                   int ctl_lead_want, int ctl_trail_want)
{
    const char *v;
    int rows_ok = (r->top == in_zp && r->bottom == in_zp &&
                   r->lead_corner == in_zp && r->trail_corner == in_zp);
    int cols_ok = (r->left == in_zp && r->right == in_zp);
    int ctl_ok  = (r->ctl_lead == ctl_lead_want && r->ctl_trail == ctl_trail_want);

    if (!ctl_ok) v = "PROGRAM DID NOT COMPUTE (interior control failed)";
    else if (rows_ok && cols_ok) v = "border exact on both axes";
    else if (rows_ok && !cols_ok) v = "ROWS exact, COLUMNS do not";
    else if (!rows_ok && cols_ok) v = "COLUMNS exact, ROWS do not";
    else v = "neither axis exact";

    printf("      %-30s zp %4d | rows %5d %5d  cols %5d %5d  corners %5d %5d | %s%s\n",
           label, in_zp, r->top, r->bottom, r->left, r->right,
           r->lead_corner, r->trail_corner, v,
           r->col_varies ? "  (a column sample VARIES down the edge)" : "");
}

int main(int argc, char **argv)
{
    unsigned c = argc > 1 ? (unsigned)atoi(argv[1]) : 0u;
    const char *e = getenv("ROCKET_RK3576_PADMAP_ZP");
    const char *nod = getenv("ROCKET_RK3576_PADMAP_DELTA");
    int zp2 = e ? atoi(e) : 20;
    int fd, failed = 0, i, rc;
    unsigned ch, y, x;
    struct readout r;
    int ctl0, ctl1;
    char set[64];

    if (c >= IC) { fprintf(stderr, "channel must be 0..%u\n", IC - 1u); return 2; }
    if (strcmp(rocket_hw_current()->name, "rk3576") != 0) {
        printf("rk3576_argb_padmap: profile is %s, not rk3576 — skipping\n",
               rocket_hw_current()->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_argb_padmap: no NPU device — skipping\n"); return 2; }

    IMG    = malloc((size_t)IC * N * N);
    PACKED = malloc((size_t)IC * N * N);
    W      = malloc((size_t)OC * IC * K * K);
    if (!IMG || !PACKED || !W) { rocket_close(fd); return 2; }

    /* Every pixel distinct enough that a buffer-neighbour read cannot be confused with a
     * constant: the row's LAST pixel and its FIRST — what a linear left/right pad would
     * read — get spreads of their own. Values stay in 5..100 so `out + in_zp` never
     * clamps at the zero points here. */
    for (ch = 0; ch < IC; ch++)
        for (y = 0; y < N; y++)
            for (x = 0; x < N; x++) {
                unsigned v = 5u + ((y * 7u + x * 3u + ch * 29u) % 90u);
                if (x == N - 1u) v = 5u + ((y * 11u + ch * 31u) % 90u);
                if (x == 0u)     v = 7u + ((y * 13u + ch * 37u) % 90u);
                IMG[((size_t)ch * N + y) * N + x] = (int8_t)v;
            }
    for (y = 0; y < N; y++)
        for (x = 0; x < N; x++)
            for (ch = 0; ch < IC; ch++)
                PACKED[((size_t)y * N + x) * IC + ch] = IMG[((size_t)ch * N + y) * N + x];
    ctl0 = IMG[(size_t)c * N * N + 0];              /* image (0,0), the tap-(0,0) control */
    ctl1 = IMG[(size_t)c * N * N + 1 * N + 1];      /* image (1,1), the trailing control  */

    printf("== packed-image first conv: WHERE does a pad tap's value come from ==\n");
    printf("   %ux%u k%u s1 pad 1 (trailing pad 1), ic=%u oc=%u, unit scales — so the\n"
           "   numbers below are the SAMPLES the MAC saw, at any zero point.\n", N, N, K,
           IC, OC);
    printf("   weight: output channel 0, image channel %u, one tap = 1; all else 0.\n", c);
    printf("   A correct border feeds the INPUT ZERO POINT at every pad tap, so `rows`,\n"
           "   `cols` and `corners` should all read zp. The map from the register is\n"
           "   `(int8_t)byte` [rk3576_argb_padcon], so a row sample IS the constant the\n"
           "   program carries — nothing here reads the emitter for it.\n\n");

    printf("   SECTION 1 — the map, by position and by zero point\n");
    printf("      %-30s %4s | %-25s %-13s %-13s |\n", "arm", "zp",
           "rows  top  bottom", "cols  L    R", "corners");

    rc = read_cell(fd, c, 0, 0, NULL, &r);
    if (rc) { printf("      packed zp 0: REFUSED (%d)\n", rc); failed++; }
    else { report("packed", 0, &r, ctl0, ctl1); if (r.ctl_lead != ctl0) failed++; }

    rc = read_cell(fd, c, 0, zp2, NULL, &r);
    if (rc != ROCKET_OK) {
        /* The entry refuses a non-zero zero point on this path — which is the bound this
         * file is about. ROCKET_RK3576_ARGB_INZP is the RE escape past it. */
        setenv("ROCKET_RK3576_ARGB_INZP", "1", 1);
        rc = read_cell(fd, c, 0, zp2, NULL, &r);
        unsetenv("ROCKET_RK3576_ARGB_INZP");
    }
    if (rc) { printf("      packed zp %d: REFUSED (%d)\n", zp2, rc); failed++; }
    else { report("packed (ARGB_INZP escape)", zp2, &r, ctl0, ctl1); if (r.ctl_lead != ctl0) failed++; }

    rc = read_cell(fd, c, 1, 0, NULL, &r);
    if (rc) { printf("      direct zp 0: REFUSED (%d)\n", rc); failed++; }
    else { report("direct (control)", 0, &r, ctl0, ctl1); if (r.ctl_lead != ctl0) failed++; }

    rc = read_cell(fd, c, 1, zp2, NULL, &r);
    if (rc) { printf("      direct zp %d: REFUSED (%d)\n", zp2, rc); failed++; }
    else { report("direct (control)", zp2, &r, ctl0, ctl1); if (r.ctl_lead != ctl0) failed++; }

    printf("\n      The `cols` pair is the whole question. If it reads 0 at EVERY zero\n"
           "      point while `rows` tracks zp, the column pad is a dead constant and no\n"
           "      value of CNA_PAD_CON1 reaches it. If it varies down the edge, it is the\n"
           "      packed buffer's own neighbouring pixel instead.\n");

    if (nod && atoi(nod) == 0) goto done;

    /* --- SECTION 2 ---------------------------------------------------------------
     * Which register, if any, turns column padding on. The direct program is the
     * existence proof that some configuration reaches those columns; this walks the
     * packed program one register at a time toward it. Run at the NON-ZERO zero point,
     * because at zp 0 a dead column feeds the right answer by accident. */
    printf("\n   SECTION 2 — one register at a time from the packed program to the direct\n"
           "   one, at zp %d. An arm whose interior control fails has broken the program\n"
           "   and says nothing about the border.\n", zp2);
    printf("      %-30s %4s | %-25s %-13s %-13s |\n", "arm", "zp",
           "rows  top  bottom", "cols  L    R", "corners");

    setenv("ROCKET_RK3576_ARGB_INZP", "1", 1);
    for (i = 0; i < (int)NDELTA; i++) {
        char label[40];
        snprintf(set, sizeof set, "0x%04X=0x%08X", DELTA[i].reg, DELTA[i].direct);
        snprintf(label, sizeof label, "0x%04X -> 0x%08X", DELTA[i].reg, DELTA[i].direct);
        rc = read_cell(fd, c, 0, zp2, set, &r);
        if (rc) { printf("      %-30s zp %4d | REFUSED (%d)\n", label, zp2, rc); continue; }
        report(label, zp2, &r, ctl0, ctl1);
    }
    unsetenv("ROCKET_RK3576_ARGB_INZP");

    printf("\n      A one-at-a-time walk cannot see a condition of two registers. What it\n"
           "      CAN see, because every arm is a readout rather than a pass/fail, is an\n"
           "      arm that moves the columns at all — including one that moves them to the\n"
           "      wrong value while the interior still computes.\n");

done:
    free(IMG); free(PACKED); free(W);
    rocket_close(fd);
    return failed ? 1 : 0;
}
