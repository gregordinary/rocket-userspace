// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Ground-truth replay of the captured Teflon int8 DEPTHWISE conv.
 *
 * Loads Mesa's exact captured BOs (input feature cube, weight cube, int32 bias
 * cube) from the teflon-dw-capture dir, generates OUR int8-OUT on-chip-requant DW
 * regcmd (gen_conv2d_dw_int8 with int8_out=1) for the captured shape/scales, runs
 * it on /dev/accel/accel0, and compares the int8 output byte-for-byte to Mesa's
 * captured mesa-output. This validates the int8-out requant datapath end-to-end
 * against ground truth with ZERO host-packing logic of our own (Mesa's BOs are
 * reused verbatim) — the decisive test that the regcmd is HW-correct.
 *
 * Captured shape: IC=64 8x8 K3x3 s1 p1, per-tensor int8
 *   in  scale 0.03657235 zp -2 ; w scale 0.00079638 zp 0 ; out scale 0.00691164 zp 5
 *
 * Usage: replay_dw_mesa [capture_dir]
 *   default dir: tests/data/teflon-dw-capture (baked at build time via TEFLON_DW_CAPTURE_DIR)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rocket_npu.h"
#include "npu_matmul.h"

#define CBUF_BANK 32768

#ifndef TEFLON_DW_CAPTURE_DIR
#define TEFLON_DW_CAPTURE_DIR "tests/data/teflon-dw-capture"
#endif

static long load_file(const char *dir, const char *name, void *dst, long cap) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    long n = (long)fread(dst, 1, cap, f);
    fclose(f);
    fprintf(stderr, "loaded %ld bytes from %s\n", n, name);
    return n;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : TEFLON_DW_CAPTURE_DIR;

    const int IC = 64, IH = 8, IW = 8, OC = 64, KH = 3, KW = 3;
    const int OH = 8, OW = 8;
    const long IN_SZ = 8192, WT_SZ = 1152, BIAS_SZ = 256, OUT_SZ = 8192;

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no /dev/accel/accel0 (%d)\n", fd); return 2; }

    rocket_bo guard = {0}, regcmd = {0}, input = {0}, weights = {0}, biases = {0}, output = {0};
    uint64_t regs[256] = {0};
    int rc = 1, ret = 0;

    rocket_bo_alloc(fd, 4096, &guard);   /* reserve IOVA 0 */
    ret |= rocket_bo_alloc(fd, 256 * sizeof(uint64_t), &regcmd);
    ret |= rocket_bo_alloc(fd, IN_SZ   + CBUF_BANK, &input);
    ret |= rocket_bo_alloc(fd, WT_SZ   + CBUF_BANK, &weights);
    ret |= rocket_bo_alloc(fd, BIAS_SZ + CBUF_BANK, &biases);
    ret |= rocket_bo_alloc(fd, OUT_SZ  + CBUF_BANK, &output);
    if (ret) { fprintf(stderr, "BO alloc failed\n"); goto out; }

    /* load Mesa's captured cubes verbatim */
    rocket_bo_prep(fd, &input, 1, 0);
    memset(input.ptr, 0, input.size);
    if (load_file(dir, "mesa-input-000-000.bin", input.ptr, IN_SZ) < 0) goto out;
    rocket_bo_fini(fd, &input);

    rocket_bo_prep(fd, &weights, 1, 0);
    memset(weights.ptr, 0, weights.size);
    if (load_file(dir, "mesa-weights-000-000.bin", weights.ptr, WT_SZ) < 0) goto out;
    rocket_bo_fini(fd, &weights);

    rocket_bo_prep(fd, &biases, 1, 0);
    memset(biases.ptr, 0, biases.size);
    if (load_file(dir, "mesa-biases-000-000.bin", biases.ptr, BIAS_SZ) < 0) goto out;
    rocket_bo_fini(fd, &biases);

    /* OUR int8-out DW regcmd, addresses = actual BO IOVAs */
    conv_params_t p = {
        .ic = IC, .ih = IH, .iw = IW, .oc = OC, .oh = OH, .ow = OW,
        .kh = KH, .kw = KW, .stride_y = 1, .stride_x = 1,
        .dil_y = 1, .dil_x = 1, .pad_top = 1, .pad_left = 1,
        .input_dma = (uint32_t)input.dma_address,
        .weights_dma = (uint32_t)weights.dma_address,
        .output_dma = (uint32_t)output.dma_address,
        .tasks = regs, .dw_group = 64,
        .int8_out = 1,
        .in_scale = 0.03657235f, .w_scale = 0.00079638f, .out_scale = 0.00691164f,
        .input_zero_point = -2, .output_zero_point = 5, .weight_zero_point = 0,
        .bias_dma = (uint32_t)biases.dma_address,
    };
    int g = gen_conv2d_dw_int8(&p);
    if (g != 0) { fprintf(stderr, "gen failed %d\n", g); goto out; }
    rocket_bo_prep(fd, &regcmd, 1, 0);
    memcpy(regcmd.ptr, regs, (size_t)p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &regcmd);

    rocket_bo_prep(fd, &output, 1, 0);
    memset(output.ptr, 0xAA, output.size);
    rocket_bo_fini(fd, &output);

    rocket_task_desc task = { .regcmd = (uint32_t)regcmd.dma_address, .regcmd_count = p.task_count };
    uint32_t in_h[]  = { input.handle, weights.handle, biases.handle, regcmd.handle };
    uint32_t out_h[] = { output.handle };
    if ((ret = rocket_submit_tasks(fd, &task, 1, in_h, 4, out_h, 1)) != 0) {
        fprintf(stderr, "submit failed %d\n", ret); goto out;
    }
    if ((ret = rocket_bo_prep(fd, &output, 0, 2000000000ULL)) != 0) {
        fprintf(stderr, "wait timeout %d\n", ret); goto out;
    }

    /* compare to mesa-output */
    {
        uint8_t *mesa = malloc(OUT_SZ);
        if (load_file(dir, "mesa-output-000-000.bin", mesa, OUT_SZ) < 0) { free(mesa); goto out; }
        uint8_t *mine = output.ptr;
        int touched = 0; for (long i = 0; i < OUT_SZ; i++) if (mine[i] != 0xAA) { touched = 1; break; }
        if (!touched) { printf("DIAGNOSTIC: output still 0xAA — NPU did not write\n"); free(mesa); goto out; }

        /* logical NCHW compare via the int8 NHWC cube (C2=16, DST_SURF_STRIDE=OH*OW) */
        long mism = 0, nonsat_mism = 0, first = -1;
        for (int c = 0; c < OC; c++)
          for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                long idx = feature_data(OC, OH, OW, 16, c + 1, oh + 1, ow + 1);
                int mv = (int8_t)mine[idx], ev = (int8_t)mesa[idx];
                if (mv != ev) {
                    mism++;
                    if (ev != 127 && ev != -128) nonsat_mism++;
                    if (first < 0) { first = idx;
                        printf("  first mismatch: cube[%ld] (c=%d oh=%d ow=%d) mine=%d mesa=%d\n",
                               idx, c, oh, ow, mv, ev); }
                }
            }
        long total = (long)OC * OH * OW;
        printf("logical int8 compare: %ld/%ld mismatch (%ld non-saturated) -> %s\n",
               mism, total, nonsat_mism, mism == 0 ? "BIT-EXACT PASS" : "MISMATCH");
        /* raw byte compare too */
        long raw = 0; for (long i = 0; i < OUT_SZ; i++) if (mine[i] != mesa[i]) raw++;
        printf("raw byte compare: %ld/%ld bytes differ\n", raw, OUT_SZ);
        rc = (mism == 0) ? 0 : 1;
        free(mesa);
    }
    rocket_bo_fini(fd, &output);

out:
    rocket_bo_free(fd, &output); rocket_bo_free(fd, &biases); rocket_bo_free(fd, &weights);
    rocket_bo_free(fd, &input); rocket_bo_free(fd, &regcmd); rocket_bo_free(fd, &guard);
    rocket_close(fd);
    printf("==== %s ====\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
