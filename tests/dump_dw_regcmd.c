// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * dump_dw_regcmd.c — host-only: emit gen_conv2d_dw_fp16's regcmd for a depthwise
 * shape as a raw little-endian u64 stream, so it can be decoded with the Mesa
 * rocket decoder (decode.py --xml registers.xml) and diffed against a captured
 * Teflon depthwise regcmd. No hardware. Our NPUOP word encoding
 * (target<<48 | value<<16 | reg) is identical to Mesa's, so decode.py decodes
 * either stream. Build:
 *   gcc -O2 -Iinclude -D__fp16=_Float16 tests/dump_dw_regcmd.c src/npu_regcmd.c -o dump_dw_regcmd
 * Run (defaults to the C=64 8x8 K3x3 s1 shape used for the Teflon ground-truth
 * capture; writes ./dw-regcmd.bin unless an output path is given):
 *   ./dump_dw_regcmd [out.bin]
 *   python3 .../rocket/decode.py --xml .../rocket/registers.xml --dump dw-regcmd.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "npu_matmul.h"

int main(int argc, char **argv)
{
    const char *out = argc > 1 ? argv[1] : "dw-regcmd.bin";
    uint64_t tasks[4096] = {0};
    conv_params_t p;
    memset(&p, 0, sizeof(p));
    p.ic = 64; p.ih = 8; p.iw = 8;
    p.oc = 64; p.oh = 8; p.ow = 8;
    p.kh = 3; p.kw = 3;
    p.stride_y = 1; p.stride_x = 1;
    p.dil_y = 1; p.dil_x = 1;
    p.pad_top = 1; p.pad_left = 1;
    p.input_dma = 0x2000; p.weights_dma = 0x3000; p.output_dma = 0x4000;
    p.tasks = tasks; p.fp32tofp16 = 1; p.dw_group = 32;  /* fp16 DW group=32 (HW-confirmed) */

    int r = gen_conv2d_dw_fp16(&p);
    fprintf(stderr, "gen_conv2d_dw_fp16 ret=%d task_count=%u -> %s\n", r, p.task_count, out);
    if (r) return 1;

    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return 1; }
    fwrite(tasks, sizeof(uint64_t), p.task_count, f);
    fclose(f);
    return 0;
}
