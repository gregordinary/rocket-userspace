// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_fc_vendor.c — submit a VENDOR CAPTURE VERBATIM and ask whether it writes.
 *
 * The int8 packed-image first conv is the one RK3576 datapath whose register program
 * is reproduced byte for byte from vendor captures and which nonetheless writes
 * nothing at all on the part, at every geometry — while the fp16 first conv at the
 * identical geometry, in the identical harness, writes its whole surface. A
 * leave-one-out over the 19-register delta between the two revives nothing, so it is
 * a mode rather than a register, and the emitter is no longer the thing to question.
 *
 * This removes the emitter from the question entirely. It takes the captured op
 * stream out of the golden table, patches ONLY the five address registers (which the
 * capture carries as the vendor model's own IOVAs and which the register-fidelity
 * gate excludes for exactly that reason), appends the PC trailer that `rocket` needs
 * and the vendor kernel adds outside the stream, and submits it.
 *
 * The result separates two possibilities that no host-side comparison can:
 *
 *   the surface changes   — the vendor's program computes, and what is wrong is our
 *                           EXTRAPOLATION away from the captured geometry
 *   nothing is written    — a register-identical vendor program writes nothing, so
 *                           the gap is outside the register stream: a buffer layout,
 *                           or state the vendor runtime establishes elsewhere
 *
 * A control runs the same way against a captured DIRECT int8 program, which is the
 * path known bit-exact — if that one writes and the first conv does not, the harness
 * is not what is being measured.
 *
 * Usage: rk3576_fc_vendor [case-name-substring]   (default: the ARGB int8 captures)
 * Exit: 0 the run completed, 1 a submit failed, 2 no NPU (skip).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_hw_profile.h"

#include "rk3576_vendor_golden.h"

#define SENTINEL 0xAA

static void sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

/* Feature bytes the capture's own DMA describes. The ARGB path reads a packed image,
 * `ic_img` interleaved bytes per pixel; every other path reads the NC1HWC2 cube. */
static size_t feature_bytes(const struct rk3576_golden_case *c, unsigned ic_img)
{
    if (c->argb) return (size_t)c->iw * c->ih_task * ic_img;
    return (size_t)((c->ic + 15u) / 16u) * c->iw * c->ih_task * 16u;
}

/* Our emitter's program for the same case, so "the vendor's stream writes" and "ours
 * does not" can be read at one geometry rather than across two. Returns the op count,
 * or 0 if the generator refused. */
static unsigned emit_ours(const struct rk3576_golden_case *c, unsigned ic_img,
                          uint64_t *ops, unsigned max_ops,
                          uint64_t in_dma, uint64_t w_dma, uint64_t o_dma,
                          uint64_t b_dma)
{
    conv_params_t p = {0};
    char fbuf[32];

    (void)max_ops;
    p.ic = (uint16_t)(c->argb ? ic_img : c->ic);
    p.oc = (uint16_t)c->oc;
    p.iw = (uint16_t)c->iw;   p.ih = (uint16_t)c->ih_task;
    p.ow = (uint16_t)c->ow;   p.oh = (uint16_t)c->oh_task;
    p.kh = (uint16_t)c->kh;   p.kw = (uint16_t)c->kw;
    p.stride_y = (uint8_t)c->sy; p.stride_x = (uint8_t)c->sx;
    p.pad_top  = (uint8_t)c->pad_top;  p.pad_left = (uint8_t)c->pad_left;
    /* The ARGB program carries no full-plane height; the gate reads ih_full as the
     * task's own rows on that path. */
    p.ih_full = (uint16_t)(c->argb ? c->ih_task : c->ih_full);
    p.oh_full = (uint16_t)c->oh_full;
    p.int8_out = 1;
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;
    p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
    p.input_dma = in_dma; p.weights_dma = w_dma;
    p.output_dma = o_dma; p.bias_dma = b_dma;
    p.tasks = ops;

    /* The vendor allocator's own CBUF allowance, so the comparison is of the program
     * and not of two allocators — the same thing the fidelity gate forces. */
    snprintf(fbuf, sizeof fbuf, "%u", c->cbuf_f);
    setenv("ROCKET_RK3576_CBUF_F", fbuf, 1);
    if (c->dw ? gen_conv2d_dw_int8_rk3576(&p) : gen_conv2d_int8_rk3576(&p)) {
        unsetenv("ROCKET_RK3576_CBUF_F");
        return 0;
    }
    unsetenv("ROCKET_RK3576_CBUF_F");
    return p.task_count;
}

static int run_case(int fd, const struct rk3576_golden_case *c, int ours)
{
    /* The image channel count the ARGB mode word carries, which is not c->ic — that
     * is the FOLDED count the MAC sees (4 lanes x kw columns). */
    unsigned ic_img = c->argb ? (c->ic / 4u) : c->ic;
    unsigned groups = (c->oc + 15u) / 16u;
    size_t in_bytes = feature_bytes(c, ic_img);
    /* Four times the declared cube, so a layout that reads past its extent lands
     * inside the BO rather than faulting. */
    size_t w_bytes  = (size_t)c->weight_bytes * 4;
    size_t coeff    = c->dw ? rocket_rk3576_coeff_bytes_dw(c->oc)
                            : rocket_rk3576_coeff_bytes(c->oc);
    size_t obytes   = (size_t)groups * c->surface_add * 16u;
    unsigned n = c->n_ops + 4u;
    unsigned n_alloc = n > RK3576_CONV_TASK_OPS ? n : RK3576_CONV_TASK_OPS;
    rocket_bo in_bo = {0}, w_bo = {0}, b_bo = {0}, o_bo = {0}, r_bo = {0};
    uint64_t *ops = NULL;
    uint32_t in_h[4], out_h[1];
    int32_t *bias = NULL;
    unsigned i, touched = 0;
    int rc = 1;

    ops  = calloc(n_alloc, sizeof *ops);
    bias = calloc(c->oc, sizeof *bias);
    if (!ops || !bias) goto done;

    if (rocket_bo_alloc(fd, in_bytes, &in_bo) < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes,  &w_bo)  < 0) goto done;
    if (rocket_bo_alloc(fd, coeff,    &b_bo)  < 0) goto done;
    if (rocket_bo_alloc(fd, obytes,   &o_bo)  < 0) goto done;
    if (rocket_bo_alloc(fd, (size_t)n_alloc * sizeof(uint64_t), &r_bo) < 0) goto done;

    /* Operands that cannot make the answer be "nothing to write": a feature of all
     * ones and a weight cube of all ones, whatever the cube's real order turns out to
     * be, give every output a large non-zero accumulator under ANY reading of the
     * layout. What is being measured is whether the DPU writes, not what it writes. */
    rocket_bo_prep(fd, &in_bo, 1, 0);
    memset(in_bo.ptr, 1, in_bytes);
    rocket_bo_fini(fd, &in_bo);
    rocket_bo_prep(fd, &w_bo, 1, 0);
    memset(w_bo.ptr, 1, w_bytes);
    rocket_bo_fini(fd, &w_bo);
    rocket_bo_prep(fd, &b_bo, 1, 0);
    if (c->dw) rocket_rk3576_pack_coeff_dw(b_bo.ptr, coeff, bias, c->oc);
    else       rocket_rk3576_pack_coeff(b_bo.ptr, coeff, bias, c->oc);
    rocket_bo_fini(fd, &b_bo);

    if (ours) {
        n = emit_ours(c, ic_img, ops, n, in_bo.dma_address, w_bo.dma_address,
                      o_bo.dma_address, b_bo.dma_address);
        if (!n) { printf("  %-40s OURS: the generator refused\n", c->name); goto done; }
    } else {
        /* The captured stream, verbatim but for the five addresses — which the capture
         * carries as the vendor model's IOVAs and which the fidelity gate excludes. */
        for (i = 0; i < c->n_ops; i++) {
            uint32_t v = c->ops[i].val;
            switch (c->ops[i].reg) {
                case 0x1088: v = (uint32_t)in_bo.dma_address; break;  /* feature base */
                case 0x1110: v = (uint32_t)w_bo.dma_address;  break;  /* weight base  */
                case 0x4018: v = (uint32_t)o_bo.dma_address;  break;  /* output base  */
                case 0x5020: v = (uint32_t)b_bo.dma_address;  break;  /* coeff base   */
                case 0x5024: v = (uint32_t)(b_bo.dma_address + coeff - 64u); break;
                default: break;
            }
            ops[i] = NPUOP(c->ops[i].target, v, c->ops[i].reg);
        }
        /* The PC trailer the vendor stream does not carry: its kernel starts the PC
         * outside the regcmd, and `rocket` needs the start inside it. */
        ops[c->n_ops + 0] = NPUOP(OP_NONE,   0x0,  0x0);
        ops[c->n_ops + 1] = NPUOP(OP_REG_PC, 0x0,  PC_REGISTER_AMOUNTS);
        ops[c->n_ops + 2] = NPUOP(OP_40,     0x0,  0x0);
        ops[c->n_ops + 3] = NPUOP(OP_ENABLE, 0x1D, PC_OPERATION_ENABLE);
    }

    rocket_bo_prep(fd, &r_bo, 1, 0);
    memcpy(r_bo.ptr, ops, (size_t)n * sizeof(uint64_t));
    rocket_bo_fini(fd, &r_bo);

    /* A sentinel, bracketed — a bare fill races the DPU's own DMA and the writeback
     * lands on top of the result, which reads as a job that wrote nothing. */
    rocket_bo_prep(fd, &o_bo, 1, 0);
    memset(o_bo.ptr, SENTINEL, obytes);
    rocket_bo_fini(fd, &o_bo);

    in_h[0] = in_bo.handle; in_h[1] = w_bo.handle;
    in_h[2] = b_bo.handle;  in_h[3] = r_bo.handle;
    out_h[0] = o_bo.handle;

    if (rocket_submit_matmul(fd, &r_bo, n, in_h, 4, out_h, 1, 4000) != 0) {
        printf("  %-40s SUBMIT FAILED\n", c->name);
        goto done;
    }
    if (rocket_bo_prep(fd, &o_bo, 0, 2000000000ull) < 0) {
        printf("  %-40s PREP_BO TIMED OUT\n", c->name);
        goto done;
    }
    {
        const unsigned char *o = o_bo.ptr;
        size_t j;
        for (j = 0; j < obytes; j++) if (o[j] != SENTINEL) touched++;
    }
    rocket_bo_fini(fd, &o_bo);

    printf("  %-40s %-10s %-6s %6u of %zu output bytes written%s\n", c->name,
           c->dw ? "depthwise" : (c->argb ? "first-conv" : "direct"),
           ours ? "OURS" : "vendor",
           touched, obytes, touched ? "" : "   — NOTHING");
    rc = 0;
done:
    if (r_bo.ptr)  rocket_bo_free(fd, &r_bo);
    if (o_bo.ptr)  rocket_bo_free(fd, &o_bo);
    if (b_bo.ptr)  rocket_bo_free(fd, &b_bo);
    if (w_bo.ptr)  rocket_bo_free(fd, &w_bo);
    if (in_bo.ptr) rocket_bo_free(fd, &in_bo);
    free(ops); free(bias);
    return rc;
}

int main(int argc, char **argv)
{
    const char *filter = argc > 1 ? argv[1] : NULL;
    size_t n = sizeof rk3576_golden_cases / sizeof rk3576_golden_cases[0];
    size_t i;
    int fd, ran = 0, bad = 0;

    fd = rocket_open();
    if (fd < 0) { printf("no /dev/accel/accel0 (%d) — SKIP\n", fd); return 2; }
    {
        const struct rocket_hw_profile *hw = rocket_hw_current();
        if (!hw || !hw->name || strcmp(hw->name, "rk3576")) {
            printf("chip is '%s', not rk3576 — SKIP\n", hw && hw->name ? hw->name : "?");
            rocket_close(fd);
            return 2;
        }
    }

    printf("RK3576 vendor-capture submit — the captured stream verbatim, addresses "
           "patched\n");
    for (i = 0; i < n; i++) {
        const struct rk3576_golden_case *c = &rk3576_golden_cases[i];
        if (c->is_float) continue;                 /* the float captures compute already */
        if (filter) { if (!strstr(c->name, filter)) continue; }
        else if (!c->argb && ran >= 2) continue;   /* two direct/dw controls is enough */
        /* Back-to-back submits of a first-conv program come back unwritten whatever
         * the program says; the idle is what makes the answer about the program. */
        sleep_ms(5000);
        if (run_case(fd, c, 0)) bad++;
        sleep_ms(5000);
        if (run_case(fd, c, 1)) bad++;
        ran++;
        if (!filter && ran >= 6) break;
    }
    printf("%d case%s submitted, %d could not be run\n", ran, ran == 1 ? "" : "s", bad);
    rocket_close(fd);
    return bad ? 1 : 0;
}
