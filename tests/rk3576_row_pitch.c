// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_row_pitch.c — will the PPU read a plane whose ROWS sit further apart in DDR
 * than the windows consume?
 *
 * What wants it: the packed-image first conv can only run at a non-zero input zero point
 * by MATERIALISING the pad columns, which forces a wider programmed output extent — at
 * ResNet-18's stem, a 128-wide surface for a 112-wide output. That surface is not the
 * caller's tensor, so the handle refuses to be anybody's input cube, the stem de-scatters
 * 784 KiB to row-major and the max pool after it scatters the same 784 KiB back. Two
 * transposes of one tensor, measured at 1.10 ms and 0.85 ms, and the run finder turns the
 * pool loose as well.
 *
 * The standing reading is that a plane's ROW stride is implied by the width with no
 * register to move it, where the channel-GROUP stride is one the emitter fills. That is
 * true of the CNA's feature DMA, whose width field and line stride are one quantity in
 * two units. It is NOT true of the PPU: `0x600C` carries what the windows consume,
 * `(ow-1)*sx + kw` less the leading pad and clamped to the plane, while `0x7024` carries
 * the DDR line stride and `0x7028` the channel-group jump. The plane appears ONLY in the
 * strides, so the geometry is expressible — but `iw` stood for BOTH quantities, and the
 * clamp is what separates them: a pitch passed as the plane lifts it, and a window that
 * should read a synthesised pad reads the producer's surplus columns instead. Hence
 * `src_line_elems`, and hence the two geometries below whose windows overhang.
 *
 * Whether the part HONOURS it is the hardware question, and the vendor captures answer it
 * only at a gap of one column (a 19-wide plane pooled k2 s2 programs an extent of 18
 * against a 19-wide line stride). One column is the smallest gap there is, and a part that
 * honoured a one-column overhang need not honour sixteen.
 *
 * The cases, per geometry:
 *
 *   PITCH    the same logical plane laid out at a row pitch P > iw, the surplus columns
 *            holding a poison byte, the pitch declared in `src_line_elems` so the two
 *            stride registers describe the wider buffer while the CONSUMED extent stays
 *            clamped against the real plane. The output must be bit-identical to a
 *            reference over a tight plane.
 *            The poison is what makes that mean something: a pool reading at the tight
 *            pitch picks it up and cannot come out equal by luck.
 *   CONTROL  the same padded buffer with the GROUP stride told correctly and the row pitch
 *            left at `iw`. This must DIFFER — it is what isolates `0x7024` from `0x7028`.
 *            Without it a pass says only that some stride register carried the layout.
 *
 * And, because it is the fallback route and its cost is the one term of that item's cap
 * that was a model rather than a measurement:
 *
 *   COST     the host COMPACTION the cube link would otherwise need — 4 channel groups x
 *            112 rows of 1792 contiguous bytes out of a 128-wide surface into a 112-wide
 *            cube, over two BOs, against a flat memcpy of the same bytes for the ceiling.
 *
 * Usage: rk3576_row_pitch [pitch|cost|all]
 * Exit:  0 the part honours a padded row pitch, 1 it does not, 2 no NPU or wrong chip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_hw_profile.h"
#include "rocket_pool.h"

#define C2       16u
#define SENTINEL 0xAA
#define POISON   0x5A

static unsigned round4(unsigned v) { return (v + 3u) & ~3u; }

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

/* The logical feature byte at channel c, row y, column x. No symmetry in any axis, so a
 * surface built from the wrong bytes cannot come out equal to one built from the right
 * ones. */
static int8_t feat_byte(unsigned c, unsigned y, unsigned x)
{
    unsigned v = c * 37u + y * 11u + x * 5u;
    return (int8_t)((int)(v % 251u) - 125);
}

/* Lay the logical c x ih x iw plane into an NC1HWC2 cube at row pitch `pitch` and group
 * stride `gstride` (both in elements), poisoning everything the plane does not cover. */
static void cube_fill(int fd, rocket_bo *bo, size_t bytes, unsigned c, unsigned ih,
                      unsigned iw, unsigned pitch, unsigned gstride)
{
    unsigned creg = ((c + 15u) / 16u) * 16u;
    int8_t *cube;
    unsigned ci, y, x;

    rocket_bo_prep(fd, bo, 1, 0);
    cube = (int8_t *)bo->ptr;
    memset(cube, POISON, bytes);
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                cube[(size_t)(ci / C2) * gstride * C2 +
                     (size_t)C2 * ((size_t)y * pitch + x) + (ci % C2)] =
                    feat_byte(ci, y, x);
    (void)creg;
    rocket_bo_fini(fd, bo);
}

struct geom {
    const char *name;
    unsigned c, iw, ih, k, stride, pad, mode;
};

/* One pool run. The plane stays the tensor's; `pitch` is the DDR row stride to declare
 * (0 leaves it derived from the plane) and `surf` the group stride to force (0 likewise).
 * Returns 0 on a submit that completed, -1 otherwise; the surface is left in `out`. */
static int pool_run(int fd, const struct geom *g, unsigned ow, unsigned oh,
                    rocket_bo *in, rocket_bo *out, size_t out_bytes,
                    unsigned pitch, unsigned surf, size_t base_off)
{
    uint64_t ops[RK3576_POOL_TASK_OPS] = {0};
    pool_params_rk3576_t p = {0};
    rocket_bo bo_r = {0};
    uint32_t in_h[2], out_h[1];
    int rc = -1;

    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) return -1;

    p.iw = (uint16_t)g->iw; p.ih = (uint16_t)g->ih; p.c = (uint16_t)g->c;
    p.ow = (uint16_t)ow;      p.oh = (uint16_t)oh;
    p.kw = (uint8_t)g->k;     p.kh = (uint8_t)g->k;
    p.stride_x = (uint8_t)g->stride; p.stride_y = (uint8_t)g->stride;
    p.pad_left = p.pad_right = p.pad_top = p.pad_bottom = (uint8_t)g->pad;
    p.mode = (uint8_t)g->mode;
    p.input_zero_point = 0;
    p.input_dma  = in->dma_address + (uint32_t)base_off;
    p.output_dma = out->dma_address;
    p.src_surf_elems = surf;
    p.src_line_elems = pitch;
    p.tasks = ops;

    if (gen_pool_rk3576(&p) != 0) { printf("      generator refused\n"); goto done; }

    rocket_bo_prep(fd, &bo_r, 1, 0);
    memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &bo_r);

    /* Bracketed, so "never written" is a property of the surface and not a guess. */
    rocket_bo_prep(fd, out, 1, 0);
    memset(out->ptr, SENTINEL, out_bytes);
    rocket_bo_fini(fd, out);

    in_h[0] = in->handle; in_h[1] = bo_r.handle;
    out_h[0] = out->handle;
    if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 2, out_h, 1, 2000) != 0) {
        printf("      submit failed\n"); goto done;
    }
    if (rocket_bo_prep(fd, out, 0, 2000000000ull) < 0) {
        printf("      PREP_BO timed out\n"); goto done;
    }
    rc = 0;
done:
    rocket_bo_free(fd, &bo_r);
    return rc;
}

/* Lay the logical plane at row pitch `pitch` starting at COLUMN `c0`, which is what a
 * materialising stem's surface holds: its extension plan discards the first D output
 * columns, so the caller's tensor is a sub-rectangle rather than a prefix. */
static void cube_fill_at(int fd, rocket_bo *bo, size_t bytes, unsigned c, unsigned ih,
                         unsigned iw, unsigned pitch, unsigned gstride, unsigned c0)
{
    int8_t *cube;
    unsigned ci, y, x;

    rocket_bo_prep(fd, bo, 1, 0);
    cube = (int8_t *)bo->ptr;
    memset(cube, POISON, bytes);
    for (ci = 0; ci < c; ci++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                cube[(size_t)(ci / C2) * gstride * C2 +
                     (size_t)C2 * ((size_t)y * pitch + x + c0) + (ci % C2)] =
                    feat_byte(ci, y, x);
    rocket_bo_fini(fd, bo);
}

/* One geometry: a reference over a tight plane, a ladder of row pitches, and the control
 * that isolates the line-stride register from the group-stride one. */
static int pitch_case(int fd, const struct geom *g)
{
    static const unsigned GAPS[] = { 1u, 4u, 16u, 64u };
    const unsigned NGAP = (unsigned)(sizeof GAPS / sizeof GAPS[0]);
    unsigned creg = ((g->c + 15u) / 16u) * 16u, groups = creg / C2;
    unsigned ow = (g->iw + 2u * g->pad - g->k) / g->stride + 1u;
    unsigned oh = (g->ih + 2u * g->pad - g->k) / g->stride + 1u;
    unsigned out_surf = round4(ow * oh);
    unsigned maxpitch = g->iw + GAPS[NGAP - 1u];
    size_t out_bytes = (size_t)groups * out_surf * C2;
    size_t tight_bytes = (size_t)groups * round4(g->iw * g->ih) * C2;
    size_t wide_bytes  = (size_t)groups * round4(maxpitch * g->ih) * C2;
    rocket_bo tight = {0}, wide = {0}, out = {0};
    unsigned char *ref = NULL;
    unsigned i;
    int bad = 0;

    printf("  %s: c=%u %ux%u k%u s%u pad%u %s -> %ux%u, %u group(s)\n",
           g->name, g->c, g->iw, g->ih, g->k, g->stride, g->pad,
           g->mode == ROCKET_RK3576_POOL_MAX ? "max" : "avg", ow, oh, groups);

    if (rocket_bo_alloc(fd, tight_bytes, &tight) < 0 ||
        rocket_bo_alloc(fd, wide_bytes, &wide) < 0 ||
        rocket_bo_alloc(fd, out_bytes, &out) < 0) {
        printf("      allocation failed\n"); bad = -1; goto done;
    }

    /* ---- the reference: a tight plane, every stride derived as always ---- */
    cube_fill(fd, &tight, tight_bytes, g->c, g->ih, g->iw, g->iw,
              round4(g->iw * g->ih));
    if (pool_run(fd, g, ow, oh, &tight, &out, out_bytes, 0, 0, 0) != 0) {
        printf("      the reference run failed\n"); bad = -1; goto done;
    }
    ref = malloc(out_bytes);
    if (!ref) { bad = -1; goto done; }
    memcpy(ref, out.ptr, out_bytes);
    rocket_bo_fini(fd, &out);
    for (i = 0; i < out_bytes; i++) if (ref[i] != SENTINEL) break;
    if (i == out_bytes) {
        printf("      the reference surface is entirely sentinel — nothing wrote, so no "
               "comparison below means anything\n");
        bad = -1; goto done;
    }

    /* ---- the ladder ---- */
    for (i = 0; i < NGAP; i++) {
        unsigned P = g->iw + GAPS[i], S = round4(P * g->ih);
        int same;

        cube_fill(fd, &wide, wide_bytes, g->c, g->ih, g->iw, P, S);
        if (pool_run(fd, g, ow, oh, &wide, &out, out_bytes, P, 0, 0) != 0) {
            printf("      PITCH   %-5u (+%-3u) the run failed\n", P, GAPS[i]);
            bad = 1; continue;
        }
        same = memcmp(out.ptr, ref, out_bytes) == 0;
        rocket_bo_fini(fd, &out);
        printf("      PITCH   %-5u (+%-3u) output %s\n", P, GAPS[i],
               same ? "== reference" : "DIFFERS");
        if (!same) bad = 1;
    }

    /* ---- THE COLUMN OFFSET. A materialising stem discards its first D output columns,
     * so the caller's tensor starts at column D of the surface rather than at column 0 —
     * a base that is NOT a whole channel group. The PPU's source base is a plain address,
     * which this asks at three offsets. ---- */
    for (i = 0; i < 3u; i++) {
        static const unsigned C0[] = { 1u, 3u, 8u };
        unsigned P = g->iw + 16u, S = round4(P * g->ih), c0 = C0[i];
        int same;

        cube_fill_at(fd, &wide, wide_bytes, g->c, g->ih, g->iw, P, S, c0);
        if (pool_run(fd, g, ow, oh, &wide, &out, out_bytes, P, S,
                     (size_t)c0 * C2) != 0) {
            printf("      COLOFF  %-5u (pitch %u) the run failed\n", c0, P);
            bad = 1; continue;
        }
        same = memcmp(out.ptr, ref, out_bytes) == 0;
        rocket_bo_fini(fd, &out);
        printf("      COLOFF  %-5u (pitch %u) output %s\n", c0, P,
               same ? "== reference" : "DIFFERS");
        if (!same) bad = 1;
    }

    /* ---- THE CONTROL. The group stride told, the row pitch left at the tensor's.
     * This must DIFFER, or the pass above says nothing about 0x7024. ---- */
    {
        unsigned P = g->iw + 16u, S = round4(P * g->ih);
        int same;

        cube_fill(fd, &wide, wide_bytes, g->c, g->ih, g->iw, P, S);
        if (pool_run(fd, g, ow, oh, &wide, &out, out_bytes, 0, S, 0) != 0) {
            printf("      CONTROL pitch not told: the run failed\n");
            bad = 1;
        } else {
            same = memcmp(out.ptr, ref, out_bytes) == 0;
            rocket_bo_fini(fd, &out);
            printf("      CONTROL pitch not told (group stride %u told): output %s%s\n",
                   S, same ? "== reference" : "DIFFERS",
                   same ? "  <-- the ladder above is VACUOUS" : "");
            if (same) bad = 1;
        }
    }

done:
    rocket_bo_free(fd, &out);
    rocket_bo_free(fd, &wide);
    rocket_bo_free(fd, &tight);
    free(ref);
    return bad;
}

static int pitch_probe(int fd)
{
    static const struct geom G[] = {
        /* ResNet-18's own max pool, and the surface its materialising stem writes. */
        { "resnet18 pool", 64u, 112u, 112u, 3u, 2u, 1u, ROCKET_RK3576_POOL_MAX },
        /* A different method, kernel, stride and pad. */
        { "avg k2 s2",     32u,  56u,  56u, 2u, 2u, 0u, ROCKET_RK3576_POOL_AVG },
        /* An ODD plane, where the consumed extent and the plane already differ. */
        { "odd k3 s2",     16u,  35u,  35u, 3u, 2u, 1u, ROCKET_RK3576_POOL_MAX },
        /* A NON-SQUARE plane, which is what separates the two axes' strides. */
        { "nonsquare k3",  32u,  64u,  32u, 3u, 1u, 1u, ROCKET_RK3576_POOL_MAX },
    };
    const int N = (int)(sizeof G / sizeof G[0]);
    int i, bad = 0, ran = 0;

    printf("== PITCH: does the PPU honour a DDR row pitch wider than the windows consume? ==\n");
    for (i = 0; i < N; i++) {
        int rc = pitch_case(fd, &G[i]);
        if (rc < 0) continue;
        ran++;
        if (rc) bad = 1;
    }
    if (!ran) { printf("== no case ran ==\n"); return -1; }
    printf("== %d geometr%s asked; the part %s a padded row pitch ==\n",
           ran, ran == 1 ? "y" : "ies", bad ? "does NOT honour" : "HONOURS");
    return bad;
}

/* ---------------------------------------------------------------------------
 * COST — the fallback route's price, so the item's cap has a measurement where it
 * had a model. This is the host compaction a cube link would need if the part did
 * NOT honour a padded pitch: the producer's 128-wide surface copied into a 112-wide
 * cube, which is 4 channel groups x 112 rows of 1792 contiguous bytes.
 * ------------------------------------------------------------------------ */
static int cost_probe(int fd)
{
    const unsigned groups = 4u, rows = 112u, wide = 128u, narrow = 112u;
    const size_t row_bytes = (size_t)narrow * C2;
    const size_t src_bytes = (size_t)groups * wide * rows * C2;
    const size_t dst_bytes = (size_t)groups * narrow * rows * C2;
    const int iters = 200;
    rocket_bo src = {0}, dst = {0};
    double t0, tc = 0, tm = 0, tb = 0;
    int i;
    unsigned g, y;

    printf("== COST: the host compaction, %u groups x %u rows x %u bytes "
           "(%.0f KiB read, %.0f KiB written) ==\n",
           groups, rows, (unsigned)row_bytes, src_bytes / 1024.0, dst_bytes / 1024.0);

    if (rocket_bo_alloc(fd, src_bytes, &src) < 0 ||
        rocket_bo_alloc(fd, dst_bytes, &dst) < 0) {
        printf("   allocation failed\n");
        rocket_bo_free(fd, &dst); rocket_bo_free(fd, &src);
        return -1;
    }
    rocket_bo_prep(fd, &src, 1, 0);
    memset(src.ptr, 0x11, src_bytes);
    rocket_bo_fini(fd, &src);
    rocket_bo_prep(fd, &dst, 1, 0);
    memset(dst.ptr, 0x22, dst_bytes);
    rocket_bo_fini(fd, &dst);

    /* Warm, then the strided compaction itself — no cache maintenance inside the loop,
     * because the bracket is a separate term already measured at 2.24 us + 0.194 us/KiB. */
    for (i = 0; i < 8; i++)
        for (g = 0; g < groups; g++)
            for (y = 0; y < rows; y++)
                memcpy((char *)dst.ptr + ((size_t)g * rows + y) * row_bytes,
                       (char *)src.ptr + ((size_t)g * rows + y) * wide * C2, row_bytes);

    t0 = now_us();
    for (i = 0; i < iters; i++)
        for (g = 0; g < groups; g++)
            for (y = 0; y < rows; y++)
                memcpy((char *)dst.ptr + ((size_t)g * rows + y) * row_bytes,
                       (char *)src.ptr + ((size_t)g * rows + y) * wide * C2, row_bytes);
    tc = (now_us() - t0) / iters;

    /* The ceiling: one flat memcpy of the same bytes, no stride. */
    t0 = now_us();
    for (i = 0; i < iters; i++) memcpy(dst.ptr, src.ptr, dst_bytes);
    tm = (now_us() - t0) / iters;

    /* And the same compaction IN PLACE, which needs no second BO and no second name in
     * the job's BO list — rows move down, so a forward per-row copy is safe. */
    t0 = now_us();
    for (i = 0; i < iters; i++)
        for (g = 0; g < groups; g++)
            for (y = 0; y < rows; y++)
                memmove((char *)src.ptr + ((size_t)g * rows + y) * row_bytes,
                        (char *)src.ptr + ((size_t)g * rows + y) * wide * C2, row_bytes);
    tb = (now_us() - t0) / iters;

    printf("   compaction (two BOs)  %7.3f ms   %.2f GB/s\n",
           tc / 1000.0, dst_bytes / (tc * 1e-6) / 1e9);
    printf("   flat memcpy, same bytes %5.3f ms   %.2f GB/s   (the ceiling)\n",
           tm / 1000.0, dst_bytes / (tm * 1e-6) / 1e9);
    printf("   compaction in place   %7.3f ms   %.2f GB/s\n",
           tb / 1000.0, dst_bytes / (tb * 1e-6) / 1e9);
    printf("   plus one PREP/FINI bracket over %.0f KiB: %.3f ms "
           "(2.24 us + 0.194 us/KiB, rk3576_prep_floor)\n",
           dst_bytes / 1024.0, (2.24 + 0.194 * (dst_bytes / 1024.0)) / 1000.0);

    rocket_bo_free(fd, &dst);
    rocket_bo_free(fd, &src);
    return 0;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *mode = argc > 1 ? argv[1] : "all";
    int fd, rc = 0;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_row_pitch: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_row_pitch: no NPU device — skipping\n"); return 2; }

    if (!strcmp(mode, "cost")) {
        rc = cost_probe(fd);
    } else if (!strcmp(mode, "pitch")) {
        rc = pitch_probe(fd);
    } else {
        rc = pitch_probe(fd);
        printf("\n");
        if (cost_probe(fd) < 0 && rc == 0) rc = -1;
    }
    rocket_close(fd);
    return rc > 0 ? 1 : (rc < 0 ? 1 : 0);
}
