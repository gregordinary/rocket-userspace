// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * conv_dw_int8_runtime.c — end-to-end ground-truth gate for the int8 DEPTHWISE
 * int8-out RUNTIME (rocket_conv2d_dw_int8). Where replay_dw_mesa validates the REGCMD
 * by feeding Mesa's captured cubes verbatim, this validates the RUNTIME's HOST PACKING
 * (the -0x80 input/weight centering, the zero-point bias fold, the +0x80 output) by
 * driving rocket_conv2d_dw_int8 with the RAW model tensors and comparing to Teflon's
 * captured mesa-output (the ground-truth oracle for this op).
 *
 * Inputs (dumped by tests/dw_dump_capture.py from the SAME capture + dw_pt.tflite):
 *   /tmp/dw_raw_in.bin  raw int8 input  [C][IH][IW]  (descattered from mesa-input)
 *   /tmp/dw_w.bin       raw int8 filter [C][KH][KW]  (from the tflite model, INDEPENDENT)
 *   /tmp/dw_bias.bin    int32 bias      [C]          (from the tflite model)
 * The filter+bias come from the tflite model (not descattered), so a PASS independently
 * confirms the runtime's DW weight-cube + bias-fold reproduce Teflon. Captured shape +
 * quant: IC=64 8x8 K3x3 s1 p1; in 0.03657235/-2, w 0.00079638/0, out 0.00691164/5.
 *
 * Usage: conv_dw_int8_runtime [capture_dir] [in.bin] [w.bin] [bias.bin]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "rocket_npu.h"
#include "rocket_conv.h"
#include "npu_matmul.h"   /* feature_data */

#ifndef TEFLON_DW_CAPTURE_DIR
#define TEFLON_DW_CAPTURE_DIR "tests/data/teflon-dw-capture"
#endif

static long load_file(const char *path, void *dst, long cap) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    long n = (long)fread(dst, 1, cap, f);
    fclose(f);
    return n;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : TEFLON_DW_CAPTURE_DIR;
    const char *inp  = argc > 2 ? argv[2] : "/tmp/dw_raw_in.bin";
    const char *wp   = argc > 3 ? argv[3] : "/tmp/dw_w.bin";
    const char *bp   = argc > 4 ? argv[4] : "/tmp/dw_bias.bin";

    const int C = 64, IH = 8, IW = 8, KH = 3, KW = 3;
    const int OH = 8, OW = 8;
    const float in_s = 0.03657235f, w_s = 0.00079638f, out_s = 0.00691164f;
    const int in_zp = -2, w_zp = 0, out_zp = 5;

    int8_t  *in   = malloc((size_t)C * IH * IW);
    int8_t  *w    = malloc((size_t)C * KH * KW);
    int32_t *bias = malloc((size_t)C * sizeof(int32_t));
    int8_t  *got  = malloc((size_t)C * OH * OW);
    uint8_t *mesa = malloc((size_t)C * OH * OW * 16);   /* int8 cube C2=16 */
    if (!in || !w || !bias || !got || !mesa) { fprintf(stderr, "oom\n"); return 2; }

    if (load_file(inp, in, (size_t)C*IH*IW) < 0) return 2;
    if (load_file(wp,  w,  (size_t)C*KH*KW) < 0) return 2;
    if (load_file(bp,  bias, (size_t)C*sizeof(int32_t)) < 0) return 2;

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no /dev/accel/accel0 (%d)\n", fd); return 2; }

    rocket_conv2d_desc d = { .ic=C,.ih=IH,.iw=IW,.oc=C,.kh=KH,.kw=KW,
        .stride_y=1,.stride_x=1,.pad_top=1,.pad_left=1,.dil_y=1,.dil_x=1,.depthwise=1 };
    int r = rocket_conv2d_dw_int8(fd, &d, in, w, bias, in_s, w_s, out_s,
                                  in_zp, w_zp, out_zp, got);
    rocket_close(fd);
    if (r) { printf("rocket_conv2d_dw_int8 = %d\n==== FAIL ====\n", r); return 1; }

    /* mesa-output is the raw NPU cube (uint8-centered); model domain = byte + 0x80. */
    char path[1024]; snprintf(path, sizeof(path), "%s/mesa-output-000-000.bin", dir);
    if (load_file(path, mesa, (size_t)C*OH*OW*16) < 0) return 2;

    long mism = 0, first = -1;
    for (int c = 0; c < C; c++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                long idx = feature_data(C, OH, OW, 16, c + 1, oh + 1, ow + 1);
                int ref = (int8_t)(mesa[idx] + 0x80u);
                int mine = got[((size_t)c * OH + oh) * OW + ow];
                if (mine != ref) {
                    if (first < 0) { first = idx;
                        printf("  first mismatch c=%d oh=%d ow=%d mine=%d mesa(model)=%d\n",
                               c, oh, ow, mine, ref); }
                    mism++;
                }
            }
    long total = (long)C * OH * OW;
    printf("DW int8 runtime vs Teflon mesa-output: %ld/%ld mismatch -> %s\n",
           mism, total, mism == 0 ? "BIT-EXACT PASS" : "MISMATCH");
    free(in); free(w); free(bias); free(got); free(mesa);
    printf("==== %s ====\n", mism == 0 ? "PASS" : "FAIL");
    return mism == 0 ? 0 : 1;
}
