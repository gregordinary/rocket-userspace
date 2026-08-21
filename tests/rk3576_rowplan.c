/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * What row window does the shipped planner choose for a given layer, and how many
 * granules does that window stage? Pure planner arithmetic — no device, no submit —
 * so a model's own geometry can be read off on the dev machine.
 *
 *   rk3576_rowplan <iw> <ih> <ic> <oc> <kh> <kw> <sy> <sx> <same> <dw> [name]
 *
 * Reads ROCKET_RK3576_MAX_ROWS like every other caller, so an arm can be priced here
 * before it is run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"

int main(int argc, char **argv)
{
    conv_params_t p;
    rocket_rk3576_row_task t[512];
    unsigned n = 0, i, iw, ih, ic, oc, kh, kw, sy, sx, same, dw, entries, cap, oh, ow;
    unsigned pad_t, pad_l;

    if (argc < 11) {
        fprintf(stderr, "usage: %s iw ih ic oc kh kw sy sx same dw [name]\n", argv[0]);
        return 2;
    }
    iw = atoi(argv[1]); ih = atoi(argv[2]); ic = atoi(argv[3]); oc = atoi(argv[4]);
    kh = atoi(argv[5]); kw = atoi(argv[6]); sy = atoi(argv[7]); sx = atoi(argv[8]);
    same = atoi(argv[9]); dw = atoi(argv[10]);

    pad_t = same ? (kh - 1u) / 2u : 0u;
    pad_l = same ? (kw - 1u) / 2u : 0u;
    oh = same ? (ih + sy - 1u) / sy : (ih - kh) / sy + 1u;
    ow = same ? (iw + sx - 1u) / sx : (iw - kw) / sx + 1u;

    memset(&p, 0, sizeof p);
    p.ic = ic; p.ih = ih; p.iw = iw;
    p.oc = oc; p.oh = oh; p.ow = ow;
    p.kh = kh; p.kw = kw;
    p.stride_y = sy; p.stride_x = sx;
    p.pad_top = pad_t; p.pad_left = pad_l;
    p.ih_full = ih; p.oh_full = oh;

    /* The packed-image first conv reads its rows a different way — four lanes of the
     * image rather than a cube's channel atom — so this row-granule arithmetic does not
     * describe it and is not printed there. The CBUF rung below comes from the library
     * either way, and it is the quantity that decides whether the window is honoured:
     * the allowance is a LADDER, and a rung the part does not deliver leaves the task
     * with the F=0 budget and a wrong tail past 4096/entries rows. */
    entries = ic <= 4u ? 0u : (iw * ic + 63u) / 64u;
    cap = rocket_rk3576_max_task_rows(iw, ic, oc, kh, kw, (int)dw);
    printf("%-18s iw=%-4u ih=%-4u ic=%-4u oc=%-4u k%ux%u s%u %s %s  ->  oh=%u ow=%u\n",
           argc > 11 ? argv[11] : "-", iw, ih, ic, oc, kh, kw, sy,
           same ? "SAME" : "VALID", dw ? "dw" : "direct", oh, ow);
    if (entries)
        printf("   entries/row %u   planner cap %u input row(s) = %u granule(s)\n",
               entries, cap, cap * entries);
    else
        printf("   packed-image rows (entries not this formula)   planner cap %u input "
               "row(s)\n", cap);
    if (rocket_rk3576_plan_rows(&p, (int)dw, t, 512u, &n) < 0 || !n) {
        printf("   NO ROW PLAN\n");
        return 1;
    }
    printf("   %u task(s):\n", n);
    for (i = 0; i < n; i++) {
        unsigned f = 0;
        int rc = rocket_rk3576_cbuf_f(iw, ic, t[i].ih, oc, kh, kw, (int)dw, &f);
        printf("     [%2u] in rows %4u..%-4u (%3u)  out rows %4u..%-4u (%3u)  pad_top %u"
               "   CBUF F=%u",
               i, t[i].iy0, t[i].iy0 + t[i].ih - 1u, t[i].ih,
               t[i].oy0, t[i].oy0 + t[i].oh - 1u, t[i].oh, t[i].pad_top,
               rc < 0 ? 0u : f);
        if (rc < 0)  printf(" (REFUSED)");
        if (entries) printf("   staged %u granule(s)", t[i].ih * entries);
        printf("\n");
    }
    return 0;
}
