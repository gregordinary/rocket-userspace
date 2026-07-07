// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/* dump_regcmd.c — host-only: generate the matmul regcmd for accumulate=0 vs 1
 * and diff them. No hardware. Build:
 *   gcc -O2 -Iinclude -D__fp16=_Float16 tests/dump_regcmd.c src/npu_regcmd.c -o dump_regcmd
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "npu_matmul.h"

static const char *blk(uint16_t op) {
    switch (op & 0xFF00) {
        case 0x0100: return "PC  ";
        case 0x0200: return "CNA ";
        case 0x0800: return "CORE";
        case 0x1000: return "DPU ";
        case 0x2000: return "RDMA";
        default:     return (op==0)?"NONE":"??  ";
    }
}

int main(void) {
    int M=4, K=64, N=16;
    uint64_t a[256]={0}, b[256]={0};
    matmul_params_t p0 = { .m=M,.k=K,.n=N,.input_dma=0x1000,.weights_dma=0x2000,
        .output_dma=0x3000,.tasks=a,.fp32tofp16=1,.accumulate=0,.add_dma=0 };
    matmul_params_t p1 = { .m=M,.k=K,.n=N,.input_dma=0x1000,.weights_dma=0x2000,
        .output_dma=0x3000,.tasks=b,.fp32tofp16=1,.accumulate=1,.add_dma=0x3000 };
    gen_matmul_fp16(&p0);
    gen_matmul_fp16(&p1);
    printf("accumulate=0: %u ops   accumulate=1: %u ops\n\n", p0.task_count, p1.task_count);

    int n = p0.task_count > p1.task_count ? p0.task_count : p1.task_count;
    printf("%-4s  %-6s  %-10s  %-10s  %s\n","idx","blk","reg","val(=0)","val(=1)");
    for (int i=0;i<n;i++){
        uint16_t op0=a[i]>>48, rg0=a[i]&0xFFFF; uint32_t v0=(a[i]>>16)&0xFFFFFFFF;
        uint16_t op1=b[i]>>48, rg1=b[i]&0xFFFF; uint32_t v1=(b[i]>>16)&0xFFFFFFFF;
        int diff = (a[i]!=b[i]);
        /* only print RDMA/DPU/PC rows + any diff, to keep it readable */
        int interesting = diff || (op1&0xFF00)==0x2000 || (op1&0xFF00)==0x1000 || (op1&0xFF00)==0x0100;
        if (!interesting) continue;
        printf("%-4d  %s   0x%04x      0x%08x  0x%08x  %s\n",
               i, blk(op1?op1:op0), rg1?rg1:rg0, v0, v1, diff?"  <-- DIFF":"");
    }
    return 0;
}
