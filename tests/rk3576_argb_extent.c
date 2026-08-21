// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_argb_extent.c — does the packed-image first conv honour an ASYMMETRIC OUTPUT
 * EXTENT?
 *
 * The CNA takes no configured trailing pad: it DERIVES the pad its last window consumes
 * from the output extent and the leading pad, and the emitter writes that derivation into
 * CNA_PAD_CON0's high bytes for both sub-encodings from one shared
 * `r76_trail_pad(OW, stride, KW, pad_left, IW)`. The DIRECT path is measured bit-exact
 * that way — it is how five MobileNet layers run TFLite's SAME. The packed-image
 * (ARGB) first conv has never been ASKED: its entry refuses an extent it did not derive
 * itself, because that path has never claimed one.
 *
 * It DOES honour one, which is what this file measures and now asserts. The refusal it
 * replaced was a LIBRARY refusal with no hardware bound behind it, and it stood between
 * Inception V1's stem and a program with 8.0x fewer MACs. That graph's stem is the one of
 * four whose geometry meets every measured bound of this path — `pad_left` 2 (non-zero,
 * which this path REQUIRES), `in_zp` 0 (a non-zero one computes a wrong border here), and
 * an output width of 112 that is both `iw/stride` and a multiple of 16 — and TFLite's SAME
 * over a 224 plane at k7 s2 puts `pad_before` 2 against `pad_after` 3, which is an extent
 * of 112 where the symmetric formula gives 111.
 *
 * Every other bound of the path still applies, so a cell here is a question about the
 * trailing pad alone.
 *
 * WHAT EACH CELL RUNS, and the first two arms are what make the third mean anything:
 *
 *   direct    the same convolution on the DIRECT datapath, whose asymmetric extent is
 *             measured exact. THE ARM THAT MUST SUCCEED: if it disagrees, the reference
 *             is what is wrong and nothing below is a statement about the part.
 *   symmetric the packed path at a SYMMETRIC extent of the same kernel and stride — the
 *             shape it is known to compute. The positive control for the packed
 *             machinery: its weight cube, its coefficient group, its row plan and its
 *             de-scatter are all shared with the candidate.
 *   packed    the candidate: the packed path with the asymmetric extent programmed.
 *
 * THE SCORE IS BY PAD REGION, because the answer's SIGNATURE is what distinguishes the
 * two ways this can fail. Every output position is classified by which pads its window
 * reaches — interior (none), lead (top/left only), trail (bottom/right only), both — so a
 * part that ignores the derived trailing pad comes back exact on interior and lead and
 * wrong on trail, while a part that mis-programs the whole geometry comes back wrong
 * everywhere. THREE OUTCOMES ARE KEPT APART: NOTHING (the surface is still the stamp),
 * WROTE (it changed and disagrees), EXACT.
 *
 * The cells vary the kernel, the stride, the leading pad and the plane, and two of them
 * make ONE axis asymmetric while the other stays derived, because an encoding that
 * honoured only the row pad would otherwise read as a flat negative.
 *
 * This is a GATE: it asserts that every cell is bit-exact and that both controls are.
 * Run with `sudo -E` on the RK3576.
 *
 * Usage: rk3576_argb_extent [cell-name-substring] [in_zp]
 * Exit:  0 every cell exact, 1 a cell or a control disagreed, 2 no NPU or the wrong chip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "requant_model.h"

#define STAMP 0x5A
#define IC     3

static const float SCALE = 1.0f / 512.0f;

static void fill_i8(int8_t *p, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        p[i] = (int8_t)((int)((seed >> 16) % 61u) - 30);
    }
}

/* A convolution over a CHW tensor with an explicit LEADING pad per axis and an explicit
 * output extent. The trailing pad is whatever the extent reaches past the plane, which is
 * exactly what the part is being asked to derive.
 *
 * THE PAD IS THE INPUT ZERO POINT, so in the model domain an out-of-range sample
 * contributes (in_zp - in_zp)*w = nothing. Every cell here carries in_zp 0 — the packed
 * path refuses anything else — but the reference is written the general way so a cell that
 * moves it is scored against arithmetic and not against a convenience. */
struct geom {
    const char *name;
    int ih, iw, kh, kw, sy, sx, pt, pl, oc;
    int oh, ow;          /* 0 = whatever the symmetric formula derives */
};

static int ref_at(const int8_t *img, int ih, int iw, const int8_t *W,
                  const struct geom *g, int o, int y, int x, int in_zp)
{
    int64_t acc = 0;
    int c, ky, kx;
    for (c = 0; c < IC; c++)
        for (ky = 0; ky < g->kh; ky++)
            for (kx = 0; kx < g->kw; kx++) {
                int sy = y * g->sy + ky - g->pt;
                int sx = x * g->sx + kx - g->pl;
                int v = (sy < 0 || sy >= ih || sx < 0 || sx >= iw)
                        ? 0 : img[((size_t)c * ih + sy) * iw + sx] - in_zp;
                acc += (int64_t)v * W[(((size_t)o * IC + c) * g->kh + ky) * g->kw + kx];
            }
    return requant_scale(acc, SCALE);
}

/* Which pads output position (y, x) reaches. Bit 0 = a leading pad tap, bit 1 = a
 * trailing one, on either axis. The trailing bit is the region the extent creates and is
 * the whole signal this probe is after. */
static int pad_class(const struct geom *g, int y, int x)
{
    int cls = 0;
    if (y * g->sy - g->pt < 0 || x * g->sx - g->pl < 0) cls |= 1;
    if (y * g->sy - g->pt + g->kh - 1 >= g->ih ||
        x * g->sx - g->pl + g->kw - 1 >= g->iw) cls |= 2;
    return cls;
}

static const char *CLASS_NAME[4] = { "interior", "lead", "trail", "both" };

struct outcome { int touched, refused; int exact[4], total[4]; };

/* Run one arm and score it. `direct` picks the datapath; everything else is the cell. */
static void run_arm(int fd, const char *label, const struct geom *g, int direct,
                    const int8_t *img, const int8_t *W, int in_zp, struct outcome *res)
{
    rocket_conv2d_desc d = {0};
    int32_t *bias = NULL;
    int8_t *in = NULL, *out = NULL;
    unsigned ow, oh, n_out, i;
    int rc, o, y, x, c, k;

    memset(res, 0, sizeof *res);
    d.ic = IC; d.oc = g->oc; d.ih = g->ih; d.iw = g->iw;
    d.kh = g->kh; d.kw = g->kw;
    d.stride_y = g->sy; d.stride_x = g->sx;
    d.pad_top = g->pt; d.pad_left = g->pl;
    d.dil_y = 1; d.dil_x = 1;
    d.oh = g->oh; d.ow = g->ow;
    d.direct_datapath = direct;
    ow = (unsigned)rocket_conv2d_ow(&d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    n_out = (unsigned)g->oc * ow * oh;

    /* The direct arm gets the SAME image in the same CHW layout — the direct int8 cube is
     * a 32-channel MAC group at every count, so three channels needs no widening and the
     * two arms differ in the register program alone. */
    in = malloc((size_t)IC * g->ih * g->iw);
    out = malloc(n_out);
    bias = calloc(g->oc, sizeof *bias);
    if (!in || !out || !bias) { free(in); free(out); free(bias); res->refused = 2; return; }
    memcpy(in, img, (size_t)IC * g->ih * g->iw);
    memset(out, STAMP, n_out);

    rc = rocket_conv2d_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f, 1.0f / SCALE,
                                   in_zp, 0, 0, out);
    if (rc != ROCKET_OK) {
        printf("    %-10s REFUSED (%d)\n", label, rc);
        res->refused = 1;
        free(in); free(out); free(bias);
        return;
    }
    for (i = 0; i < n_out; i++)
        if ((unsigned char)out[i] != STAMP) { res->touched = 1; break; }

    for (o = 0; o < g->oc; o++)
        for (y = 0; y < (int)oh; y++)
            for (x = 0; x < (int)ow; x++) {
                int cls = pad_class(g, y, x);
                int want = ref_at(img, g->ih, g->iw, W, g, o, y, x, in_zp);
                int got = out[((size_t)o * oh + y) * ow + x];
                res->total[cls]++;
                if (want == got) res->exact[cls]++;
            }

    for (c = 0, k = 0; c < 4; c++) k += res->exact[c] == res->total[c];
    printf("    %-10s %s  %ux%u out", label,
           res->touched ? (k == 4 ? "EXACT  " : "WROTE  ") : "NOTHING", ow, oh);
    for (c = 0; c < 4; c++)
        if (res->total[c])
            printf("  %s %d/%d", CLASS_NAME[c], res->exact[c], res->total[c]);
    printf("\n");
    free(in); free(out); free(bias);
}

/* The cells. Every one satisfies the packed path's three measured bounds — a non-zero
 * left pad, ow*stride == iw and oh*stride == ih, and ow a multiple of 16 — so the only
 * thing under test is the trailing pad the extent implies.
 *
 * `sym` cells set no extent and are the positive control at that kernel and stride;
 * `asym` ones set the extent TFLite's SAME asks for. The last two make ONE axis
 * asymmetric and leave the other derived. */
static const struct geom CELLS[] = {
    /* name            ih   iw  kh kw sy sx pt pl  oc   oh   ow */
    { "sym-k7",       224, 224,  7, 7, 2, 2, 3, 3, 64,   0,   0 },
    { "asym-k7-iv1",  224, 224,  7, 7, 2, 2, 2, 2, 64, 112, 112 },
    { "asym-k7-pl1",  224, 224,  7, 7, 2, 2, 1, 1, 64, 112, 112 },
    { "sym-k5",       224, 224,  5, 5, 2, 2, 2, 2, 32,   0,   0 },
    { "asym-k5",      224, 224,  5, 5, 2, 2, 1, 1, 32, 112, 112 },
    { "asym-k5-s1",   224, 224,  5, 5, 1, 1, 1, 1, 32, 224, 224 },
    { "asym-k7-64",    64,  64,  7, 7, 2, 2, 2, 2, 32,  32,  32 },
    { "asym-k7-128",  128, 128,  7, 7, 2, 2, 2, 2, 32,  64,  64 },
    { "asym-x-only",  224, 224,  7, 7, 2, 2, 3, 2, 32,   0, 112 },
    { "asym-y-only",  224, 224,  7, 7, 2, 2, 2, 3, 32, 112,   0 },
    /* k3, which every cell above skips. A stem is 3, 5 or 7 and the two larger ones are
     * here; a cell whose kernel reaches only ONE pad row is the shape where a pad
     * constant that is wrong by a constant is hardest to hide, since a k7 window at the
     * border still sums six real rows against one padded one. */
    { "sym-k3-s1",     64,  64,  3, 3, 1, 1, 1, 1, 32,   0,   0 },
    { "sym-k3-s2",    224, 224,  3, 3, 2, 2, 1, 1, 32,   0,   0 },
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *only = argc > 1 ? argv[1] : NULL;
    int in_zp = argc > 2 ? atoi(argv[2]) : 0;
    int fd, i, ran = 0, honoured = 0, denied = 0, bad_control = 0;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_argb_extent: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_argb_extent: no NPU device — skipping\n"); return 2; }

    printf("== RK3576 packed-image first conv: does it honour an asymmetric output "
           "extent? (in_zp=%d) ==\n", in_zp);

    for (i = 0; i < NCELLS; i++) {
        const struct geom *g = &CELLS[i];
        int8_t *img = NULL, *W = NULL;
        struct outcome direct, sym, packed;
        int asym = g->oh || g->ow;
        int ty, tx;
        rocket_conv2d_desc d = {0};

        if (only && !strstr(g->name, only)) continue;
        img = malloc((size_t)IC * g->ih * g->iw);
        W   = malloc((size_t)g->oc * IC * g->kh * g->kw);
        if (!img || !W) { free(img); free(W); break; }
        fill_i8(img, (size_t)IC * g->ih * g->iw, 3u + (unsigned)i);
        fill_i8(W, (size_t)g->oc * IC * g->kh * g->kw, 11u + (unsigned)i);

        d.ic = IC; d.oc = g->oc; d.ih = g->ih; d.iw = g->iw;
        d.kh = g->kh; d.kw = g->kw;
        d.stride_y = g->sy; d.stride_x = g->sx;
        d.pad_top = g->pt; d.pad_left = g->pl;
        d.dil_y = 1; d.dil_x = 1; d.oh = g->oh; d.ow = g->ow;
        ty = rocket_conv2d_trail_y(&d); tx = rocket_conv2d_trail_x(&d);

        printf("  %-13s %dx%d k%dx%d s%d pad %d/%d oc=%u -> %dx%d, trailing pad %d/%d\n",
               g->name, g->ih, g->iw, g->kh, g->kw, g->sy, g->pt, g->pl,
               (unsigned)g->oc, rocket_conv2d_oh(&d), rocket_conv2d_ow(&d), ty, tx);

        /* THE ARM THAT MUST SUCCEED. */
        run_arm(fd, "direct", g, 1, img, W, in_zp, &direct);
        if (direct.refused ||
            direct.exact[0] != direct.total[0] || direct.exact[1] != direct.total[1] ||
            direct.exact[2] != direct.total[2] || direct.exact[3] != direct.total[3]) {
            printf("      the DIRECT control disagrees, so the reference is what is "
                   "wrong and this cell says nothing about the packed path\n");
            bad_control++;
            free(img); free(W);
            continue;
        }

        /* The packed path's own positive control, at the SAME kernel and stride. A
         * symmetric cell IS its own control, so it is only run for an asymmetric one. */
        if (asym) {
            /* The symmetric SAME lead at this kernel, which is what makes the DERIVED
             * extent iw/stride again — the same output shape with no extent asked for. */
            struct geom s = *g;
            s.oh = 0; s.ow = 0;
            s.pt = s.pl = (g->kh - 1) / 2;
            run_arm(fd, "symmetric", &s, 0, img, W, in_zp, &sym);
            if (sym.refused || sym.exact[0] != sym.total[0]) {
                printf("      the packed control is not exact at the symmetric lead, so "
                       "nothing below is about the extent\n");
                bad_control++;
                free(img); free(W);
                continue;
            }
        }

        run_arm(fd, "packed", g, 0, img, W, in_zp, &packed);
        if (asym && !packed.refused) {
            int all = packed.exact[0] == packed.total[0] &&
                      packed.exact[1] == packed.total[1] &&
                      packed.exact[2] == packed.total[2] &&
                      packed.exact[3] == packed.total[3];
            ran++;
            if (all && packed.touched) {
                honoured++;
                printf("      THE EXTENT IS HONOURED. Every output is exact, the trailing "
                       "region included, so the packed converter reads CNA_PAD_CON0's "
                       "derived pad the way the direct path does\n");
            } else if (packed.touched && packed.exact[0] == packed.total[0] &&
                       packed.exact[2] != packed.total[2]) {
                denied++;
                printf("      the interior is exact and the TRAILING region is not — the "
                       "signature of a derived pad this sub-encoding does not read\n");
            } else if (packed.touched) {
                denied++;
                printf("      it wrote and disagrees beyond the trailing region, so the "
                       "extent is not the only thing that moved\n");
            } else {
                denied++;
                printf("      nothing written\n");
            }
        }
        free(img); free(W);
    }

    printf("== %d asymmetric cells ran: %d honoured, %d not, %d controls failed ==\n",
           ran, honoured, denied, bad_control);
    rocket_close(fd);
    return (denied || bad_control || !ran) ? 1 : 0;
}
