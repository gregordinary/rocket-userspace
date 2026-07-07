// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * resize_rocket.c — HW gate for integer-factor spatial upsample (nearest / bilinear) on
 * the rocket NPU (rocket_upsample_{nearest,bilinear}_fp16), realised as a depthwise
 * ConvTranspose2d with a fixed box / triangle kernel.
 *
 * Validation (all against INDEPENDENT references — the NPU runs a transposed-conv scatter,
 * the references run a gather, so a kernel/coordinate bug can't hide):
 *   - nearest : NPU vs block-replication ref. Bit-exact (max_abs == 0).
 *   - bilinear: NPU vs 2-tap half-pixel gather ref. Within fp16 interpolation tolerance.
 *   - bilinear PARTITION OF UNITY: a constant input must upsample to that constant in the
 *     interior (proves the triangle kernel's stride-subsample sums to 1).
 *   - bilinear LINEAR EXACTNESS: a y/x ramp must upsample to the half-pixel-mapped ramp in
 *     the interior (bilinear is exact on linear functions).
 *
 * Off-device (no NPU): the references run and self-validate the partition-of-unity + linear
 * properties (the math), then SKIP (exit 2). On RK3588 hardware the NPU paths are exercised too.
 *
 * Usage: resize_rocket                       (built-in sweep)
 *        resize_rocket MODE C IH IW SY SX     (MODE: 0 nearest / 1 bilinear)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rocket_npu.h"
#include "rocket_resize.h"

static int g_fail = 0;

/* one upsample case: NPU (or CPU ref when fd<0) vs the independent reference */
static void run_case(int fd, int mode, int C, int IH, int IW, int sy, int sx)
{
    const int OH = IH * sy, OW = IW * sx;
    const char *name = mode ? "bilinear" : "nearest ";
    int plan = mode ? rocket_upsample_bilinear_plan(C, IH, IW, sy, sx)
                    : rocket_upsample_nearest_plan(C, IH, IW, sy, sx);
    if (plan) { printf("  %s C=%d %dx%d x%d,%d: plan unsupported (%d) — skip\n",
                       name, C, IH, IW, sy, sx, plan); return; }

    size_t in_n = (size_t)C * IH * IW, out_n = (size_t)C * OH * OW;
    _Float16 *in = malloc(in_n*sizeof(_Float16)), *out = malloc(out_n*sizeof(_Float16));
    _Float16 *ref = malloc(out_n*sizeof(_Float16));
    if (!in || !out || !ref) { fprintf(stderr,"oom\n"); g_fail=1; goto done; }

    for (size_t i = 0; i < in_n; i++) in[i] = (_Float16)((int)(i % 7) - 3);

    int r = mode ? rocket_upsample_bilinear_fp16(fd, in, out, C, IH, IW, sy, sx)
                 : rocket_upsample_nearest_fp16 (fd, in, out, C, IH, IW, sy, sx);
    if (r) { printf("  %s C=%d %dx%d x%d,%d: call=%d FAIL\n", name,C,IH,IW,sy,sx,r); g_fail=1; goto done; }

    if (mode) rocket_upsample_bilinear_ref_fp16(in, ref, C, IH, IW, sy, sx);
    else      rocket_upsample_nearest_ref_fp16 (in, ref, C, IH, IW, sy, sx);

    const double tol = mode ? 0.05 : 0.0;     /* nearest bit-exact; bilinear fp16-weight slack */
    double max_abs = 0; int bad = 0;
    for (size_t i = 0; i < out_n; i++) {
        double ad = fabs((float)out[i] - (float)ref[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad > tol && bad < 4) { printf("    [%zu] ref=%.3f got=%.3f\n", i,(float)ref[i],(float)out[i]); bad++; }
    }
    int ok = (mode ? max_abs <= tol : max_abs == 0.0);
    printf("  %s C=%d %dx%d -> %dx%d  x%d,%d: max_abs=%.4f -> %s\n",
           name, C, IH, IW, OH, OW, sy, sx, max_abs, ok ? "PASS" : "FAIL");
    if (!ok) g_fail = 1;
done:
    free(in); free(out); free(ref);
}

/* bilinear math properties (run on the NPU output if fd>=0, else the reference):
 * a constant must stay constant in the interior; a y/x ramp must become the
 * half-pixel-mapped ramp. */
static void run_properties(int fd, int C, int IH, int IW, int sy, int sx)
{
    if (rocket_upsample_bilinear_plan(C, IH, IW, sy, sx)) return;
    const int OH = IH*sy, OW = IW*sx;
    size_t in_n=(size_t)C*IH*IW, out_n=(size_t)C*OH*OW;
    _Float16 *in=malloc(in_n*sizeof(_Float16)), *out=malloc(out_n*sizeof(_Float16));
    if (!in||!out){fprintf(stderr,"oom\n");g_fail=1;free(in);free(out);return;}

    /* (1) constant -> constant interior */
    for (size_t i=0;i<in_n;i++) in[i]=(_Float16)2.0f;
    rocket_upsample_bilinear_fp16(fd,in,out,C,IH,IW,sy,sx);
    double cmax=0;
    for (int oh=sy;oh<OH-sy;oh++) for (int ow=sx;ow<OW-sx;ow++) {
        double ad=fabs((float)out[(size_t)oh*OW+ow]-2.0); if(ad>cmax)cmax=ad;   /* channel 0 */
    }
    printf("  bilinear partition-of-unity (const interior) x%d,%d: max|Δ|=%.4f -> %s\n",
           sy,sx,cmax, cmax<=0.02?"PASS":"FAIL");
    if (cmax>0.02) g_fail=1;

    /* (2) y-ramp -> half-pixel ramp interior:  in[y][x]=y  =>  out[oh] ≈ (oh+0.5)/sy-0.5 */
    for (int y=0;y<IH;y++) for (int x=0;x<IW;x++) in[(size_t)y*IW+x]=(_Float16)(float)y;
    rocket_upsample_bilinear_fp16(fd,in,out,C,IH,IW,sy,sx);
    double rmax=0;
    for (int oh=sy;oh<OH-sy;oh++){
        double want=(oh+0.5)/sy-0.5;
        for (int ow=sx;ow<OW-sx;ow++){
            double ad=fabs((float)out[(size_t)oh*OW+ow]-want); if(ad>rmax)rmax=ad;
        }
    }
    printf("  bilinear linear-exactness (y-ramp interior) x%d,%d: max|Δ|=%.4f -> %s\n",
           sy,sx,rmax, rmax<=0.05?"PASS":"FAIL");
    if (rmax>0.05) g_fail=1;
    free(in); free(out);
}

int main(int argc, char **argv)
{
    int fd = rocket_open();
    if (fd < 0) printf("note: no /dev/accel/accel0 (%d) — reference + math-property checks only (no HW)\n\n", fd);

    if (argc == 7) {
        run_case(fd, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
    } else {
        /* nearest: common FPN factors, C across the 32-group boundary */
        run_case(fd, 0, 32, 8, 8, 2, 2);
        run_case(fd, 0, 64, 10, 7, 2, 2);
        run_case(fd, 0, 32, 5, 5, 4, 4);
        run_case(fd, 0, 64, 6, 9, 3, 3);
        run_case(fd, 0, 32, 8, 8, 2, 1);     /* asymmetric scale */
        printf("\n");
        /* bilinear: factor 2 (exact fp16 weights), factor 3/4, asymmetric */
        run_case(fd, 1, 32, 8, 8, 2, 2);
        run_case(fd, 1, 64, 8, 8, 2, 2);
        run_case(fd, 1, 32, 6, 6, 3, 3);
        run_case(fd, 1, 32, 6, 9, 4, 2);
        printf("\n");
        run_properties(fd, 32, 8, 8, 2, 2);
        run_properties(fd, 32, 6, 6, 3, 3);
    }

    if (fd >= 0) rocket_close(fd);
    printf("==== %s ====\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : (fd < 0 ? 2 : 0);
}
