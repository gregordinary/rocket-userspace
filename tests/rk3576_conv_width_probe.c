// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_conv_width_probe.c — can an RK3576 convolution entry reach its encoder with an
 * extent past the field the encoder writes it into, and return 0 over a wrong surface?
 *
 * The RK3588 question, asked of the other part. There the CNA's width, height and output
 * width fields are 11 bits [TRM], the tilers bounded only a tile's area, and an int8 conv
 * 2300 columns wide returned 0 over a wrapped surface (tests/conv_width_gate.c). The fix
 * was a refusal in the generators past each field. The RK3576 encoder,
 * src/npu_regcmd_rk3576.c, has no such refusal: it writes every extent unmasked into a
 * 16-bit register half or a whole register, and checks only that the value fits the
 * HALF. How many of those bits the part implements is not measured for any geometry
 * field — there is no TRM chapter for the re-packed CNA, and no gate or vendor capture
 * has put a width past 2048, a task height past 2048 or a depthwise channel count past
 * 256 through the conv program.
 *
 * What bounds an extent on this part is the CBUF, not a field. One row of an int8 direct
 * conv at 32 channels is iw/2 granules against a 6144-granule data side, so a one-row
 * task reaches 12288 columns and a two-column plane reaches 6144 rows in ONE task. The
 * field widths are unknown, so each axis is a LADDER: controls just under 11, 12 and 13
 * bits (2048, 4096, 8192) and arms just past each, up to the largest extent the planner
 * admits. A wrap then shows where it starts: `onset x` against W & 0x7FF / 0xFFF / 0x1FFF.
 *
 * The groups, each through a PUBLIC entry so the tilers and planners are under test:
 *
 *   w8    int8 direct, one row, the width ladder       CNA 0x102C hi, 0x1030 lo, 0x1044 hi,
 *                                                       0x1078 hi, 0x118C; CORE 0x301C lo;
 *                                                       DPU 0x4020, 0x4034 lo; RDMA 0x500C
 *   h8    int8 direct, two columns, the height ladder  CNA 0x102C lo, 0x1078 lo;
 *                                                       CORE 0x301C hi; DPU 0x4024,
 *                                                       0x4034 hi; RDMA 0x5010
 *   kw    int8 direct, the kernel's two axes           CNA 0x1024 [31:24] and [23:16]
 *   st    int8 direct, stride past the 3-bit mask      CNA 0x1014 (the code masks & 0x7)
 *   surf  int8 direct, planes past 2^16 elements       CNA 0x1094; DPU 0x401C, 0x40B8
 *   dww   int8 depthwise, the width ladder
 *   dwh   int8 depthwise, the height ladder
 *   fcw   int8 packed-image first conv, width          (its own 0x1078/0x118C forms)
 *   fch   int8 packed-image first conv, height
 *   poolh int8 max pool, height                        PPU 0x6010/0x601C, PPU_RDMA 0x7010
 *   f16w  fp16 direct, width      (wide output: runs after the int8 groups)
 *   f16h  fp16 direct, height
 *   fc16w fp16 packed-image first conv, width
 *   dwc   int8 depthwise, the channel count            CNA 0x1028 lo, 0x107C; CORE 0x3020;
 *                                                       DPU 0x402C, 0x4030 hi; RDMA 0x5014
 *   poolc int8 max pool, the channel count             PPU 0x6014/0x6020, PPU_RDMA 0x7014
 *
 * The channel groups run last: the matmul form at N 8960 wrote 768 columns and HUNG, so a
 * channel past a 13-bit field is expected to cost a retirement, not a wrong surface.
 *
 * Scoring. Every element, against the CPU model the RK3576 gates use: an int64
 * accumulate and requant_model.h's requant for int8 (in/w/out scale 1/1/divisor, zero
 * points 0), an fp32 accumulate on small integers narrowed to fp16 for fp16, and
 * rocket_pool_ref_int8_rk3576 for the pool. Fills are splitmix64 hashes, so no period
 * hides a column or row alias. Per arm: the entry's rc, wrong elements, how many of them
 * hold the library's own output stamp (0xA5: the device never wrote them), how many hold
 * THIS caller's sentinel (the de-scatter never wrote them), the first wrong element and
 * the smallest wrong column, row and channel — the onset of a wrap on each axis — and
 * whether a fence wait went slow (a watchdog or backstop retirement).
 *
 * Usage:
 *   rk3576_conv_width_probe [plan] [group|arm ...]   default: every group, in the order above;
 *           an arm's name (w8-12288) runs that arm alone, in isolation from its ladder, and
 *           when EVERY argument names an arm they run in the order given, in one process —
 *           which is how a wrapped job's effect on the NEXT submit is asked (w8-8300 w8-2048)
 *   plan    host-only: print each arm's row plan and the geometry words its FIRST task
 *           emits, and submit nothing (off the part: ROCKET_CHIP=rk3576)
 *
 * Exit: 0 when every control is exact and no arm returned 0 over a wrong surface; 1
 * otherwise; 2 no RK3576 device. A probe, not a gate: the arms past a field EXPECT a
 * wrong answer somewhere, and a refusal is a result.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_pool.h"
#include "rocket_hw_profile.h"
#include "npu_hw.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"
#include "requant_model.h"
#include "test_fill.h"

enum kind { K_INT8, K_DW, K_FP16, K_POOL };

struct arm {
    const char *group;
    const char *name;
    enum kind kind;
    unsigned ic, oc, ih, iw, kh, kw, sy, sx, pad;   /* pad: symmetric, both axes */
    int ctrl;               /* 1 = a control: must be exact                       */
    const char *field;      /* what the arm crosses (or sits under)               */
};

/* Wide single rows at 32 channels: a row is iw/2 granules, so 12288 is the most one task
 * holds (F=2048). Two-column planes: one granule a row, so 6144 rows is the most. */
static const struct arm ARMS[] = {
    /* ---- int8 direct: width ---- */
    {"w8", "w8-2048",   K_INT8, 32, 32,    1,  2048, 1, 1, 1, 1, 0, 1, "width < 2^11"},
    {"w8", "w8-2300",   K_INT8, 32, 32,    1,  2300, 1, 1, 1, 1, 0, 0, "width > 2^11"},
    {"w8", "w8-4096",   K_INT8, 32, 32,    1,  4096, 1, 1, 1, 1, 0, 0, "width = 2^12"},
    {"w8", "w8-4200",   K_INT8, 32, 32,    1,  4200, 1, 1, 1, 1, 0, 0, "width > 2^12"},
    {"w8", "w8-8192",   K_INT8, 32, 32,    1,  8192, 1, 1, 1, 1, 0, 0, "width = 2^13"},
    {"w8", "w8-8300",   K_INT8, 32, 32,    1,  8300, 1, 1, 1, 1, 0, 0, "width > 2^13"},
    {"w8", "w8-12288",  K_INT8, 32, 32,    1, 12288, 1, 1, 1, 1, 0, 0, "width, the most one task holds"},
    /* ---- int8 direct: a MULTI-ROW plane at those widths. The line stride 0x1090 is iw*4,
     *      and the vendor compiler holds that field at 14 bits (limit 0x3FFF), which iw*4
     *      passes from iw 4096. A one-row plane never reads it. The planner gives 2-row
     *      tasks at 4096 and 6000, and those compute exactly on the part, so the compiler's
     *      width is not the silicon's here [HW, H96, 2026-09-27]. ---- */
    {"r4", "r4-2048",   K_INT8, 32, 32,    4,  2048, 1, 1, 1, 1, 0, 1, "line stride 0x2000"},
    {"r4", "r4-4096",   K_INT8, 32, 32,    4,  4096, 1, 1, 1, 1, 0, 0, "line stride 0x4000 > 14 bits"},
    {"r4", "r4-6000",   K_INT8, 32, 32,    4,  6000, 1, 1, 1, 1, 0, 0, "line stride 0x5DC0"},
    {"r4", "r4-8192",   K_INT8, 32, 32,    4,  8192, 1, 1, 1, 1, 0, 0, "line stride 0x8000"},
    {"r4", "r4k3-6000", K_INT8, 32, 32,    4,  6000, 3, 3, 1, 1, 1, 0, "a 3-row window at 0x5DC0"},
    {"r4", "r4c16-6000",K_INT8, 16, 32,    4,  6000, 1, 1, 1, 1, 0, 0, "16 channels, the same stride"},
    /* ---- int8 direct: height (one task: the whole plane fits the CBUF) ---- */
    {"h8", "h8-2048",   K_INT8, 32, 32, 2048,     2, 1, 1, 1, 1, 0, 1, "height < 2^11"},
    {"h8", "h8-2300",   K_INT8, 32, 32, 2300,     2, 1, 1, 1, 1, 0, 0, "height > 2^11"},
    {"h8", "h8-4096",   K_INT8, 32, 32, 4096,     2, 1, 1, 1, 1, 0, 0, "height = 2^12"},
    {"h8", "h8-4200",   K_INT8, 32, 32, 4200,     2, 1, 1, 1, 1, 0, 0, "height > 2^12"},
    {"h8", "h8-6144",   K_INT8, 32, 32, 6144,     2, 1, 1, 1, 1, 0, 0, "height, the most one task holds"},
    /* ---- int8 direct: the kernel word's two bytes. Every kernel run on this part is
     * 7 or smaller; the RK3588's kernel fields are 5 bits. ---- */
    {"kw", "kw-32",     K_INT8, 32, 32,    1,    63, 1, 32, 1, 1, 0, 1, "kernel width 32 (kw-1 = 0x1F)"},
    {"kw", "kw-33",     K_INT8, 32, 32,    1,    64, 1, 33, 1, 1, 0, 0, "kernel width 33 (kw-1 = 0x20)"},
    {"kw", "kh-32",     K_INT8, 32, 32,   63,     4, 32, 1, 1, 1, 0, 1, "kernel height 32"},
    {"kw", "kh-33",     K_INT8, 32, 32,   64,     4, 33, 1, 1, 1, 0, 0, "kernel height 33"},
    /* ---- int8 direct: stride. 0x1014 is (sy&7)<<3 | (sx&7) and nothing refuses a stride
     * past 7, so 9 programs 1 and 14 programs 6. Strides 8 and 16 program ZERO and are not
     * run: a stride of zero is a program nobody has seen complete. ---- */
    {"st", "st-4",      K_INT8, 32, 32,   35,    35, 3, 3, 4, 4, 0, 1, "stride 4"},
    {"st", "st-7",      K_INT8, 32, 32,   59,    59, 3, 3, 7, 7, 0, 1, "stride 7, the top of 3 bits"},
    {"st", "st-9",      K_INT8, 32, 32,   75,    75, 3, 3, 9, 9, 0, 0, "stride 9 -> field 1"},
    {"st", "st-14",     K_INT8, 32, 32,  115,   115, 3, 3, 14, 14, 0, 0, "stride 14 -> field 6"},
    /* ---- int8 direct: the plane-sized strides. 224x224 is the largest gated plane. ---- */
    {"surf", "surf-224", K_INT8, 32, 32,  224,   224, 1, 1, 1, 1, 0, 1, "plane 50176 (gated)"},
    {"surf", "surf-256", K_INT8, 32, 32,  256,   256, 1, 1, 1, 1, 0, 0, "plane 2^16"},
    {"surf", "surf-640", K_INT8, 32, 32,  640,   640, 1, 1, 1, 1, 0, 0, "plane 409600 (19 bits)"},
    /* ---- int8 depthwise ---- */
    {"dww", "dww-2048", K_DW,   32, 32,    4,  2048, 3, 3, 1, 1, 1, 1, "width < 2^11"},
    {"dww", "dww-2300", K_DW,   32, 32,    4,  2300, 3, 3, 1, 1, 1, 0, "width > 2^11"},
    {"dww", "dww-4200", K_DW,   32, 32,    2,  4200, 3, 3, 1, 1, 1, 0, "width > 2^12"},
    {"dww", "dww-8300", K_DW,   32, 32,    1,  8300, 3, 3, 1, 1, 1, 0, "width > 2^13"},
    {"dwh", "dwh-2048", K_DW,   32, 32, 2048,     2, 3, 3, 1, 1, 1, 1, "height < 2^11"},
    {"dwh", "dwh-2300", K_DW,   32, 32, 2300,     2, 3, 3, 1, 1, 1, 0, "height > 2^11"},
    {"dwh", "dwh-4200", K_DW,   32, 32, 4200,     2, 3, 3, 1, 1, 1, 0, "height > 2^12"},
    /* ---- int8 packed-image first conv: iw and ow multiples of 16, ow*s == iw, a
     * non-zero left pad, in_zp 0 so no materialised extension widens it. ---- */
    {"fcw", "fcw-2048", K_INT8,  3, 32,    4,  2048, 3, 3, 1, 1, 1, 1, "width < 2^11"},
    {"fcw", "fcw-2304", K_INT8,  3, 32,    4,  2304, 3, 3, 1, 1, 1, 0, "width > 2^11"},
    {"fcw", "fcw-4096", K_INT8,  3, 32,    4,  4096, 3, 3, 1, 1, 1, 0, "width = 2^12"},
    {"fcw", "fcw-4352", K_INT8,  3, 32,    4,  4352, 3, 3, 1, 1, 1, 0, "width > 2^12"},
    {"fcw", "fcw-8192", K_INT8,  3, 32,    4,  8192, 3, 3, 1, 1, 1, 0, "width = 2^13"},
    {"fcw", "fcw-8448", K_INT8,  3, 32,    4,  8448, 3, 3, 1, 1, 1, 0, "width > 2^13"},
    {"fch", "fch-2048", K_INT8,  3, 32, 2048,    16, 3, 3, 1, 1, 1, 1, "height < 2^11"},
    {"fch", "fch-2304", K_INT8,  3, 32, 2304,    16, 3, 3, 1, 1, 1, 0, "height > 2^11"},
    /* ---- int8 max pool: the PPU's own extent registers, no CBUF, no row window ---- */
    {"poolh", "poolh-2048", K_POOL, 16, 16, 2048, 16, 2, 2, 1, 1, 0, 1, "height < 2^11"},
    {"poolh", "poolh-2300", K_POOL, 16, 16, 2300, 16, 2, 2, 1, 1, 0, 0, "height > 2^11"},
    {"poolh", "poolh-8192", K_POOL, 16, 16, 8192, 16, 2, 2, 1, 1, 0, 0, "height = 2^13"},
    {"poolh", "poolh-8300", K_POOL, 16, 16, 8300, 16, 2, 2, 1, 1, 0, 0, "height > 2^13"},
    /* ---- fp16 direct: one 16-channel slice, no row window (the whole plane is one task) */
    {"f16w", "f16w-2048",  K_FP16, 16, 16,    1,  2048, 1, 1, 1, 1, 0, 1, "width < 2^11"},
    {"f16w", "f16w-2300",  K_FP16, 16, 16,    1,  2300, 1, 1, 1, 1, 0, 0, "width > 2^11"},
    {"f16w", "f16w-4200",  K_FP16, 16, 16,    1,  4200, 1, 1, 1, 1, 0, 0, "width > 2^12"},
    {"f16w", "f16w-8300",  K_FP16, 16, 16,    1,  8300, 1, 1, 1, 1, 0, 0, "width > 2^13"},
    {"f16w", "f16w-12288", K_FP16, 16, 16,    1, 12288, 1, 1, 1, 1, 0, 0, "width, the most one task holds"},
    {"f16h", "f16h-2048",  K_FP16, 16, 16, 2048,     2, 1, 1, 1, 1, 0, 1, "height < 2^11"},
    {"f16h", "f16h-2300",  K_FP16, 16, 16, 2300,     2, 1, 1, 1, 1, 0, 0, "height > 2^11"},
    {"f16h", "f16h-4200",  K_FP16, 16, 16, 4200,     2, 1, 1, 1, 1, 0, 0, "height > 2^12"},
    {"fc16w", "fc16w-2048", K_FP16, 3, 16,    4,  2048, 3, 3, 1, 1, 1, 1, "width < 2^11"},
    {"fc16w", "fc16w-2304", K_FP16, 3, 16,    4,  2304, 3, 3, 1, 1, 1, 0, "width > 2^11"},
    {"fc16w", "fc16w-4352", K_FP16, 3, 16,    4,  4352, 3, 3, 1, 1, 1, 0, "width > 2^12"},
    {"fc16w", "fc16w-8448", K_FP16, 3, 16,    4,  8448, 3, 3, 1, 1, 1, 0, "width > 2^13"},
    /* ---- channel counts: last, see the header ---- */
    {"dwc", "dwc-1024", K_DW, 1024, 1024,  4,     4, 3, 3, 1, 1, 1, 1, "channels 1024"},
    {"dwc", "dwc-8192", K_DW, 8192, 8192,  4,     4, 3, 3, 1, 1, 1, 0, "channels = 2^13"},
    {"dwc", "dwc-8224", K_DW, 8224, 8224,  4,     4, 3, 3, 1, 1, 1, 0, "channels > 2^13"},
    {"poolc", "poolc-1024", K_POOL, 1024, 1024, 4, 4, 2, 2, 2, 2, 0, 1, "channels 1024"},
    {"poolc", "poolc-8192", K_POOL, 8192, 8192, 4, 4, 2, 2, 2, 2, 0, 0, "channels = 2^13"},
    {"poolc", "poolc-8208", K_POOL, 8208, 8208, 4, 4, 2, 2, 2, 2, 0, 0, "channels > 2^13"},
};
#define N_ARMS ((int)(sizeof ARMS / sizeof ARMS[0]))

static const char *GROUP_ORDER[] = {
    "w8", "r4", "h8", "kw", "st", "surf", "dww", "dwh", "fcw", "fch", "poolh",
    "f16w", "f16h", "fc16w", "dwc", "poolc",
};
#define N_GROUPS ((int)(sizeof GROUP_ORDER / sizeof GROUP_ORDER[0]))

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void fill_desc(const struct arm *a, rocket_conv2d_desc *d)
{
    memset(d, 0, sizeof *d);
    d->ic = (int)a->ic; d->oc = (int)a->oc; d->ih = (int)a->ih; d->iw = (int)a->iw;
    d->kh = (int)a->kh; d->kw = (int)a->kw;
    d->stride_y = (int)a->sy; d->stride_x = (int)a->sx;
    d->pad_top = (int)a->pad; d->pad_left = (int)a->pad;
    d->dil_y = 1; d->dil_x = 1;
    d->depthwise = a->kind == K_DW;
}

static void fill_pool(const struct arm *a, rocket_pool_desc *d)
{
    memset(d, 0, sizeof *d);
    d->c = (int)a->ic; d->ih = (int)a->ih; d->iw = (int)a->iw;
    d->kh = (int)a->kh; d->kw = (int)a->kw;
    d->stride_y = (int)a->sy; d->stride_x = (int)a->sx;
    d->method = POOL_METHOD_MAX;
}

/* ---- per-arm tally ---- */
struct tally {
    size_t total, wrong, stamp, untouched;
    long first;                       /* flat index of the first wrong element, -1 none */
    unsigned min_x, min_y, min_c;     /* smallest wrong column, row, channel            */
};

static void tally_init(struct tally *t, size_t total)
{
    memset(t, 0, sizeof *t);
    t->total = total;
    t->first = -1;
    t->min_x = t->min_y = t->min_c = ~0u;
}

static void tally_wrong(struct tally *t, size_t flat, unsigned c, unsigned y, unsigned x,
                        int is_stamp, int is_untouched)
{
    if (t->first < 0) t->first = (long)flat;
    t->wrong++;
    t->stamp += is_stamp != 0;
    t->untouched += is_untouched != 0;
    if (x < t->min_x) t->min_x = x;
    if (y < t->min_y) t->min_y = y;
    if (c < t->min_c) t->min_c = c;
}

/* ---- the host-only plan: what the planner cuts, and what task 0 writes ---- */

static uint32_t reg_of(const uint64_t *ops, int n, uint16_t reg, int *found)
{
    uint32_t v = 0;
    int i;
    *found = 0;
    for (i = 0; i < n; i++)
        if ((uint16_t)(ops[i] & 0xFFFFu) == reg &&
            (uint16_t)(ops[i] >> 48) != 0 /* skip OP_NONE padding */) {
            v = (uint32_t)(ops[i] >> 16);
            *found = 1;
        }
    return v;
}

static void print_words(const uint64_t *ops, int n)
{
    static const uint16_t REGS[] = {
        0x1014, 0x1024, 0x1028, 0x102C, 0x1030, 0x1034, 0x1044, 0x1078, 0x107C, 0x1094,
        0x118C, 0x301C, 0x3020, 0x401C, 0x4020, 0x4024, 0x402C, 0x4030, 0x4034, 0x40B8,
        0x500C, 0x5010, 0x5014,
    };
    size_t r;
    int col = 0;
    printf("      task 0:");
    for (r = 0; r < sizeof REGS / sizeof REGS[0]; r++) {
        int found;
        uint32_t v = reg_of(ops, n, REGS[r], &found);
        if (!found) continue;
        printf(" %04x=%08x", REGS[r], v);
        if (++col % 6 == 0 && r + 1 < sizeof REGS / sizeof REGS[0]) printf("\n             ");
    }
    printf("\n");
}

/* The conv_params_t the library's int8 path hands its row planner for the first
 * output-channel tile (r76_int8_exec), and the generator it calls. */
static int plan_int8(const struct arm *a, int emit)
{
    rocket_conv2d_desc d;
    conv_params_t q = {0};
    rocket_rk3576_row_task *plan;
    unsigned n = 0, t, max_ih = 0, max_oh = 0, oh, ow, icreg, ocreg, maxt;
    int dw = a->kind == K_DW, argb = !dw && a->ic <= 4;

    fill_desc(a, &d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    ow = (unsigned)rocket_conv2d_ow(&d);
    icreg = argb ? (a->ic == 1 ? 2u : a->ic) : (dw ? a->ic : rocket_rk3576_pad_ic(a->ic));
    ocreg = dw ? a->oc : rocket_rk3576_pad_oc(argb && a->oc > 64 ? 64 : a->oc);
    maxt = oh + 2u;
    plan = calloc(maxt, sizeof *plan);
    if (!plan) return -1;
    q.ic = (uint16_t)icreg; q.ih = (uint16_t)a->ih; q.iw = (uint16_t)a->iw;
    q.oc = (uint16_t)ocreg; q.oh = (uint16_t)oh; q.ow = (uint16_t)ow;
    q.kh = (uint16_t)a->kh; q.kw = (uint16_t)a->kw;
    q.stride_y = (uint8_t)a->sy; q.stride_x = (uint8_t)a->sx;
    q.pad_top = (uint8_t)a->pad; q.pad_left = (uint8_t)a->pad;
    q.ih_full = (uint16_t)a->ih; q.oh_full = (uint16_t)oh;
    if (rocket_rk3576_plan_rows(&q, dw, plan, maxt, &n) < 0) {
        printf("      plan: the row planner REFUSES\n");
        free(plan);
        return 0;
    }
    for (t = 0; t < n; t++) {
        if (plan[t].ih > max_ih) max_ih = plan[t].ih;
        if (plan[t].oh > max_oh) max_oh = plan[t].oh;
    }
    printf("      plan: %u row task(s), widest window %u input / %u output rows, "
           "ow %u, ic %u oc %u programmed\n", n, max_ih, max_oh, ow, icreg, ocreg);
    if (emit) {
        uint64_t ops[RK3576_CONV_TASK_OPS];
        conv_params_t p = q;
        int rc;
        memset(ops, 0, sizeof ops);
        p.ih = plan[0].ih; p.oh = plan[0].oh; p.pad_top = plan[0].pad_top;
        p.int8_out = 1;
        p.tasks = ops;
        rc = dw ? gen_conv2d_dw_int8_rk3576(&p) : gen_conv2d_int8_rk3576(&p);
        if (rc != 0) printf("      task 0: the generator REFUSES (%d)\n", rc);
        else print_words(ops, (int)p.task_count);
    }
    free(plan);
    return 0;
}

static int plan_fp16(const struct arm *a, int emit)
{
    rocket_conv2d_desc d;
    conv_params_t q = {0};
    unsigned oh, ow;
    uint64_t ops[RK3576_CONV_TASK_OPS];
    int rc;

    fill_desc(a, &d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    ow = (unsigned)rocket_conv2d_ow(&d);
    q.ih = (uint16_t)a->ih; q.iw = (uint16_t)a->iw;
    q.oh = (uint16_t)oh; q.ow = (uint16_t)ow;
    q.kh = (uint16_t)a->kh; q.kw = (uint16_t)a->kw;
    q.stride_y = (uint8_t)a->sy; q.stride_x = (uint8_t)a->sx;
    q.pad_top = (uint8_t)a->pad; q.pad_left = (uint8_t)a->pad;
    q.ih_full = (uint16_t)a->ih; q.oh_full = (uint16_t)oh;
    q.in_scale = q.w_scale = q.out_scale = 1.0f;
    if (a->ic <= 4) {
        rocket_rk3576_row_task *plan = calloc(oh + 2u, sizeof *plan);
        unsigned n = 0, t, max_ih = 0;
        if (!plan) return -1;
        q.ic = (uint16_t)a->ic;
        q.oc = (uint16_t)rocket_rk3576_fp16_pad_oc(a->oc > 64 ? 64 : a->oc);
        if (rocket_rk3576_plan_rows_prec(&q, 0, precision_float16, plan, oh + 2u, &n) < 0) {
            printf("      plan: the row planner REFUSES\n");
            free(plan);
            return 0;
        }
        for (t = 0; t < n; t++) if (plan[t].ih > max_ih) max_ih = plan[t].ih;
        printf("      plan: %u row task(s), widest window %u input rows, ow %u\n",
               n, max_ih, ow);
        q.ih = plan[0].ih; q.oh = plan[0].oh; q.pad_top = plan[0].pad_top;
        free(plan);
    } else {
        rocket_rk3576_ic_task sl[8];
        unsigned n = 0;
        q.ic = (uint16_t)rocket_rk3576_fp16_pad_ic(a->ic);
        q.oc = (uint16_t)rocket_rk3576_fp16_pad_oc(a->oc);
        if (rocket_rk3576_plan_ic(&q, sl, 8, &n) < 0) {
            printf("      plan: the input-channel planner REFUSES\n");
            return 0;
        }
        printf("      plan: %u ic slice(s), each ONE task of the whole %ux%u plane\n",
               n, a->iw, a->ih);
        q.ic = 16;
    }
    if (!emit) return 0;
    memset(ops, 0, sizeof ops);
    q.tasks = ops;
    rc = gen_conv2d_fp16_rk3576(&q);
    if (rc != 0) printf("      task 0: the generator REFUSES (%d)\n", rc);
    else print_words(ops, (int)q.task_count);
    return 0;
}

static void plan_arm(const struct arm *a)
{
    printf("  %-11s %-24s", a->name, a->field);
    if (a->kind == K_POOL) {
        rocket_pool_desc pd;
        fill_pool(a, &pd);
        printf("\n      plan: rocket_pool_int8_rk3576_plan %d, %u column slice(s), "
               "out %dx%d\n", rocket_pool_int8_rk3576_plan(&pd),
               rocket_pool_int8_rk3576_ow_slices(&pd), rocket_pool_oh(&pd),
               rocket_pool_ow(&pd));
        return;
    }
    printf("\n");
    if (a->kind == K_FP16) plan_fp16(a, 1);
    else plan_int8(a, 1);
}

/* ---- the runs ---- */

#define CALLER_SENTINEL_I8 ((int8_t)0xAA)

static int run_int8(int fd, const struct arm *a, struct tally *t, int *rc_out)
{
    rocket_conv2d_desc d;
    const int dw = a->kind == K_DW;
    unsigned oh, ow, c, y, x, i, kh, kw, terms, divisor, scale, shift;
    size_t n_in, n_w, n_out, e;
    int8_t *in, *W, *out;
    int32_t *bias;
    const int8_t lib_stamp = (int8_t)ROCKET_RK3576_SENTINEL_BYTE;
    const uint64_t seed = 0x57D7Bu ^ ((uint64_t)a->iw << 20) ^ ((uint64_t)a->ih << 4) ^ a->ic;

    fill_desc(a, &d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    ow = (unsigned)rocket_conv2d_ow(&d);
    n_in = (size_t)a->ic * a->ih * a->iw;
    n_w = dw ? (size_t)a->ic * a->kh * a->kw : (size_t)a->oc * a->ic * a->kh * a->kw;
    n_out = (size_t)a->oc * oh * ow;
    in = malloc(n_in); W = malloc(n_w); out = malloc(n_out);
    bias = malloc((size_t)a->oc * sizeof *bias);
    if (!in || !W || !out || !bias) { free(in); free(W); free(out); free(bias); return -1; }

    /* No period anywhere: a wrapped extent reads a column or row that holds an unrelated
     * value. The packed-image path reads its input as the raw pixel (in_zp 0). */
    for (e = 0; e < n_in; e++) in[e] = (int8_t)tf_int(seed, e, -30, 30);
    for (e = 0; e < n_w; e++) W[e] = (int8_t)tf_int(seed ^ 0x77, e, -8, 8);
    for (c = 0; c < a->oc; c++) bias[c] = ((int)c - (int)a->oc / 2) * 8;
    terms = dw ? a->kh * a->kw : a->ic * a->kh * a->kw;
    divisor = 1;
    while ((double)divisor < 2.0 * sqrt((double)terms)) divisor *= 2;
    requant_params(1.0f / (float)divisor, &scale, &shift);

    memset(out, (unsigned char)CALLER_SENTINEL_I8, n_out);
    *rc_out = dw ? rocket_conv2d_dw_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f,
                                                (float)divisor, 0, 0, 0, out)
                 : rocket_conv2d_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f,
                                             (float)divisor, 0, 0, 0, out);
    tally_init(t, n_out);
    if (*rc_out == 0) {
        for (c = 0; c < a->oc; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    int64_t acc = bias[c];
                    size_t flat = ((size_t)c * oh + y) * ow + x;
                    int want, got;
                    for (kh = 0; kh < a->kh; kh++) {
                        int iy = (int)(y * a->sy + kh) - (int)a->pad;
                        if (iy < 0 || iy >= (int)a->ih) continue;
                        for (kw = 0; kw < a->kw; kw++) {
                            int ix = (int)(x * a->sx + kw) - (int)a->pad;
                            if (ix < 0 || ix >= (int)a->iw) continue;
                            if (dw) {
                                acc += (int64_t)in[((size_t)c * a->ih + iy) * a->iw + ix] *
                                       W[((size_t)c * a->kh + kh) * a->kw + kw];
                            } else {
                                for (i = 0; i < a->ic; i++)
                                    acc += (int64_t)in[((size_t)i * a->ih + iy) * a->iw + ix] *
                                           W[(((size_t)c * a->ic + i) * a->kh + kh) * a->kw + kw];
                            }
                        }
                    }
                    want = requant_apply_zp(acc, scale, shift, 0);
                    got = out[flat];
                    if (got != want)
                        tally_wrong(t, flat, c, y, x, got == lib_stamp,
                                    got == CALLER_SENTINEL_I8);
                }
    }
    free(in); free(W); free(out); free(bias);
    return 0;
}

static int run_fp16(int fd, const struct arm *a, struct tally *t, int *rc_out)
{
    rocket_conv2d_desc d;
    unsigned oh, ow, c, y, x, i, kh, kw;
    size_t n_in, n_w, n_out, e;
    _Float16 *in, *W, *out;
    const uint64_t seed = 0xF16u ^ ((uint64_t)a->iw << 20) ^ ((uint64_t)a->ih << 4) ^ a->ic;
    const uint16_t lib_stamp = (uint16_t)(ROCKET_RK3576_SENTINEL_BYTE * 0x101u);

    fill_desc(a, &d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    ow = (unsigned)rocket_conv2d_ow(&d);
    n_in = (size_t)a->ic * a->ih * a->iw;
    n_w = (size_t)a->oc * a->ic * a->kh * a->kw;
    n_out = (size_t)a->oc * oh * ow;
    in = malloc(n_in * 2); W = malloc(n_w * 2); out = malloc(n_out * 2);
    if (!in || !W || !out) { free(in); free(W); free(out); return -1; }
    /* Small integers: every partial sum is exact in fp32 and in fp16, so the only
     * rounding is none and the comparison is bit-exact whatever order the part sums in. */
    for (e = 0; e < n_in; e++) in[e] = (_Float16)tf_int(seed, e, -2, 2);
    for (e = 0; e < n_w; e++) W[e] = (_Float16)tf_int(seed ^ 0x77, e, -2, 2);
    tf_sentinel_f16(out, n_out);
    *rc_out = rocket_conv2d_fp16_rk3576(fd, &d, in, W, out);
    tally_init(t, n_out);
    if (*rc_out == 0) {
        for (c = 0; c < a->oc; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    float acc = 0.0f;
                    size_t flat = ((size_t)c * oh + y) * ow + x;
                    _Float16 want, got;
                    uint16_t gbits;
                    for (kh = 0; kh < a->kh; kh++) {
                        int iy = (int)(y * a->sy + kh) - (int)a->pad;
                        if (iy < 0 || iy >= (int)a->ih) continue;
                        for (kw = 0; kw < a->kw; kw++) {
                            int ix = (int)(x * a->sx + kw) - (int)a->pad;
                            if (ix < 0 || ix >= (int)a->iw) continue;
                            for (i = 0; i < a->ic; i++)
                                acc += (float)in[((size_t)i * a->ih + iy) * a->iw + ix] *
                                       (float)W[(((size_t)c * a->ic + i) * a->kh + kh) *
                                                a->kw + kw];
                        }
                    }
                    want = (_Float16)acc;
                    got = out[flat];
                    memcpy(&gbits, &got, 2);
                    if (!((float)got == (float)want))
                        tally_wrong(t, flat, c, y, x, gbits == lib_stamp,
                                    tf_is_sentinel_f16(got));
                }
    }
    free(in); free(W); free(out);
    return 0;
}

static int run_pool(int fd, const struct arm *a, struct tally *t, int *rc_out)
{
    rocket_pool_desc d;
    unsigned oh, ow, c, y, x;
    size_t n_in, n_out, e;
    int8_t *in, *out, *want;
    const int8_t lib_stamp = (int8_t)ROCKET_RK3576_SENTINEL_BYTE;
    const uint64_t seed = 0x9001u ^ ((uint64_t)a->ih << 8) ^ a->ic;

    fill_pool(a, &d);
    oh = (unsigned)rocket_pool_oh(&d);
    ow = (unsigned)rocket_pool_ow(&d);
    n_in = (size_t)a->ic * a->ih * a->iw;
    n_out = (size_t)a->ic * oh * ow;
    in = malloc(n_in); out = malloc(n_out); want = malloc(n_out);
    if (!in || !out || !want) { free(in); free(out); free(want); return -1; }
    for (e = 0; e < n_in; e++) in[e] = (int8_t)tf_int(seed, e, -128, 127);
    rocket_pool_ref_int8_rk3576(&d, 0, in, want);
    memset(out, (unsigned char)CALLER_SENTINEL_I8, n_out);
    *rc_out = rocket_pool_int8_rk3576(fd, &d, 0, in, out);
    tally_init(t, n_out);
    if (*rc_out == 0)
        for (c = 0; c < a->ic; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    size_t flat = ((size_t)c * oh + y) * ow + x;
                    if (out[flat] != want[flat])
                        tally_wrong(t, flat, c, y, x, out[flat] == lib_stamp,
                                    out[flat] == CALLER_SENTINEL_I8);
                }
    free(in); free(out); free(want);
    return 0;
}

static void out_dims(const struct arm *a, unsigned *oh, unsigned *ow)
{
    if (a->kind == K_POOL) {
        rocket_pool_desc pd;
        fill_pool(a, &pd);
        *oh = (unsigned)rocket_pool_oh(&pd);
        *ow = (unsigned)rocket_pool_ow(&pd);
    } else {
        rocket_conv2d_desc d;
        fill_desc(a, &d);
        *oh = (unsigned)rocket_conv2d_oh(&d);
        *ow = (unsigned)rocket_conv2d_ow(&d);
    }
}

static const char *entry_name(const struct arm *a)
{
    switch (a->kind) {
    case K_INT8: return a->ic <= 4 ? "int8 packed" : "int8 direct";
    case K_DW:   return "int8 dw";
    case K_FP16: return a->ic <= 4 ? "fp16 packed" : "fp16 direct";
    case K_POOL: return "int8 maxpool";
    }
    return "?";
}

/* Returns 1 when the arm is a finding the exit code should carry: a control that was not
 * exact, or any arm that returned 0 over a wrong surface. */
static int run_arm(int fd, const struct arm *a)
{
    struct tally t;
    int rc = 0, silent;
    unsigned oh, ow;
    uint64_t slow0 = rocket_fence_wait_slow_count();
    double t0 = now_ms(), ms;

    out_dims(a, &oh, &ow);
    switch (a->kind) {
    case K_INT8: case K_DW: if (run_int8(fd, a, &t, &rc) < 0) return 1; break;
    case K_FP16:            if (run_fp16(fd, a, &t, &rc) < 0) return 1; break;
    case K_POOL:            if (run_pool(fd, a, &t, &rc) < 0) return 1; break;
    }
    ms = now_ms() - t0;

    printf("  %-11s %-12s c%-5u %5ux%-5u k%ux%u s%u -> %5ux%-5u  rc %3d",
           a->name, entry_name(a), a->ic, a->ih, a->iw, a->kh, a->kw, a->sx, oh, ow, rc);
    if (rc == 0) {
        printf("  wrong %zu/%zu", t.wrong, t.total);
        if (t.wrong) {
            unsigned fc, fy, fx;
            size_t f = (size_t)t.first;
            fx = (unsigned)(f % ow); fy = (unsigned)((f / ow) % oh);
            fc = (unsigned)(f / ((size_t)ow * oh));
            printf(" (stamp %zu, untouched %zu) first c%u y%u x%u, onset x%u y%u c%u",
                   t.stamp, t.untouched, fc, fy, fx, t.min_x, t.min_y, t.min_c);
        }
    }
    printf("  %.0f ms%s", ms, rocket_fence_wait_slow_count() != slow0 ? "  [SLOW FENCE]" : "");
    silent = rc == 0 && t.wrong;
    if (a->ctrl) printf("  %s\n", (rc == 0 && !t.wrong) ? "CONTROL EXACT" : "CONTROL FAILED");
    else if (silent) printf("  SILENT WRONG\n");
    else if (rc) printf("  REFUSED\n");
    else printf("  EXACT\n");
    fflush(stdout);
    return (a->ctrl && (rc != 0 || t.wrong)) || silent;
}

/* An arm runs when nothing is named, when its group is named, or when it is. */
static int arm_selected(const struct arm *a, int argc, char **argv, int first)
{
    int i;
    if (first >= argc) return 1;
    for (i = first; i < argc; i++)
        if (!strcmp(argv[i], a->group) || !strcmp(argv[i], a->name)) return 1;
    return 0;
}

static int group_has_selected(const char *g, int argc, char **argv, int first)
{
    int i;
    for (i = 0; i < N_ARMS; i++)
        if (!strcmp(ARMS[i].group, g) && arm_selected(&ARMS[i], argc, argv, first)) return 1;
    return 0;
}

int main(int argc, char **argv)
{
    int plan = 0, first = 1, fd = -1, findings = 0, ran = 0, gi, i;
    const struct rocket_hw_profile *hw;

    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc > 1 && !strcmp(argv[1], "plan")) { plan = 1; first = 2; }
    for (i = first; i < argc; i++) {
        int k;
        for (gi = 0; gi < N_GROUPS; gi++) if (!strcmp(argv[i], GROUP_ORDER[gi])) break;
        for (k = 0; k < N_ARMS; k++) if (!strcmp(argv[i], ARMS[k].name)) break;
        if (gi == N_GROUPS && k == N_ARMS) {
            fprintf(stderr, "unknown group or arm '%s'; groups:", argv[i]);
            for (gi = 0; gi < N_GROUPS; gi++) fprintf(stderr, " %s", GROUP_ORDER[gi]);
            fprintf(stderr, "\n");
            return 1;
        }
    }

    if (!plan) {
        fd = rocket_open();
        if (fd < 0) { printf("no NPU device\n"); return 2; }
    }
    hw = rocket_hw_current();
    if (!hw || !hw->name || strcmp(hw->name, "rk3576")) {
        printf("not an RK3576 (profile %s)%s\n", hw && hw->name ? hw->name : "?",
               plan ? "; set ROCKET_CHIP=rk3576 for the host-only plan" : "");
        if (fd >= 0) rocket_close(fd);
        return 2;
    }

    printf("== rk3576 conv width probe: %s ==\n",
           plan ? "host-only plan, nothing submitted" : "public entries, every element scored");
    /* Every argument an arm: run them in the order given, back to back in this process. */
    {
        int all_arms = first < argc, k;
        for (i = first; i < argc && all_arms; i++) {
            for (k = 0; k < N_ARMS; k++) if (!strcmp(argv[i], ARMS[k].name)) break;
            if (k == N_ARMS) all_arms = 0;
        }
        if (all_arms) {
            printf("-- in the order given --\n");
            for (i = first; i < argc; i++) {
                for (k = 0; k < N_ARMS; k++) if (!strcmp(argv[i], ARMS[k].name)) break;
                if (plan) plan_arm(&ARMS[k]);
                else { findings += run_arm(fd, &ARMS[k]); ran++; }
            }
            gi = N_GROUPS;     /* the group walk below has nothing left to do */
        } else {
            gi = 0;
        }
    }
    for (; gi < N_GROUPS; gi++) {
        if (!group_has_selected(GROUP_ORDER[gi], argc, argv, first)) continue;
        printf("-- %s --\n", GROUP_ORDER[gi]);
        for (i = 0; i < N_ARMS; i++) {
            if (strcmp(ARMS[i].group, GROUP_ORDER[gi])) continue;
            if (!arm_selected(&ARMS[i], argc, argv, first)) continue;
            if (plan) plan_arm(&ARMS[i]);
            else { findings += run_arm(fd, &ARMS[i]); ran++; }
        }
    }
    if (!plan) {
        printf("== %d arm(s) run, %d finding(s): a control not exact or a silent wrong ==\n",
               ran, findings);
        rocket_close(fd);
    }
    return findings ? 1 : 0;
}
