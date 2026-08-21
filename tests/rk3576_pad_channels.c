// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_pad_channels.c — what does a producer leave in the channels between its live
 * output count and the group boundary its CONSUMER walks?
 *
 * A consumer whose input channel count is not a multiple of 32 has its feature DMA walk
 * the round-32 count, and at a NON-ZERO weight zero point those extra channels are
 * load-bearing: their weights are the zeros the cube was memset with, so they never reach
 * the MAC, but the coefficient group's B term sums every channel of the programmed group.
 * Reading a row-major tensor the handle fills them itself, with the border constant — its
 * own input zero point — and folds them over the register tap count. Reading a PRODUCER'S
 * SURFACE it gets whatever the DPU wrote there, and whether that is the same constant is
 * what decides whether such a join can exist at all. MobileNetV2 has seven of them.
 *
 * What is settled here, and both halves are needed:
 *
 *   A DIRECT producer's partial output group carries its OUTPUT ZERO POINT, which at a
 *   graph join is the consumer's input zero point and so exactly the border constant the
 *   consumer's own cube would have used. Those channels are packed with zero weights and a
 *   zero A term, so their accumulator is zero and the OUT_CVT sends them there.
 *
 *   ONLY IF THE TAIL IS GIVEN NO B TERM. B is per output channel and the packer's obvious
 *   loop gives every channel of the programmed count the same one — and then a padding
 *   channel carries `requant(B*sum(x))`, which is data-dependent and defeats the whole
 *   arrangement. The asymmetric cases below are the ones that say so: before the packer
 *   was fixed, 5776 of 12544 padding elements at oc=16 w_zp=12 were away from the output
 *   zero point. A gate whose producers are all symmetric cannot see this axis, and the one
 *   that missed it took MobileNetV2's second layer to 0.5% agreement.
 *
 *   A DEPTHWISE producer is programmed with the RAW output count and never writes past it,
 *   so it declares nothing and the same join is refused. Its surface is also short: 9
 *   groups at oc=144 against the 10 a direct consumer of 144 channels walks.
 *
 * Two things are asked, because they are different facts:
 *
 *   CONTENT   a producer writes into a caller's cube that was stamped first, and the
 *             channels past its live count are read back. A stamp still standing means
 *             the DPU never wrote them; the output zero point means it did and the value
 *             is the one a consumer needs.
 *   DEPTH     how many channel groups the producer actually covers, against the
 *             round-32 count a direct consumer's feature DMA walks.
 *
 * Usage:  rk3576_pad_channels
 * Exit:   0, 1 on a failure, 2 no NPU (skip).
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "npu_regcmd_rk3576.h"

#define C2        16u
#define SENTINEL  0xA5
#define PLANE     28u

/* One value out of a cube: NC1HWC2, so channel `c` at pixel `p` sits in group `c/16`. */
static int cube_at(const rocket_rk3576_cube *cu, unsigned c, unsigned p)
{
    const int8_t *b = (const int8_t *)cu->bo.ptr + cu->off;
    return b[(size_t)(c / C2) * cu->surf_elems * C2 + (size_t)p * C2 + (c % C2)];
}

/* Stamp every byte of a cube. Bracketed, never a bare memset: dirty lines race the DPU's
 * own DMA and the writeback lands on top of the result. */
static void cube_stamp(int fd, rocket_rk3576_cube *cu)
{
    rocket_bo_prep(fd, &cu->bo, 1, 0);
    memset((char *)cu->bo.ptr + cu->off, SENTINEL,
           (size_t)cu->groups * cu->surf_elems * C2);
    rocket_bo_fini(fd, &cu->bo);
}

struct layer {
    rocket_conv2d_int8_weights_rk3576 *h;
    rocket_conv2d_desc d;
    int8_t  *W;
    int32_t *bias;
    int      out_zp;
};

static int layer_pack(int fd, struct layer *L, unsigned ic, unsigned oc, unsigned k,
                      int dw, int out_zp, int w_zp)
{
    size_t n = dw ? (size_t)oc * k * k : (size_t)oc * ic * k * k;
    size_t i;
    unsigned st = 0x51A7C0DEu ^ (oc * 7919u);

    memset(L, 0, sizeof *L);
    L->d.ic = (int)ic; L->d.oc = (int)oc;
    L->d.ih = L->d.iw = (int)PLANE;
    L->d.kh = L->d.kw = (int)k;
    L->d.stride_y = L->d.stride_x = 1;
    L->d.pad_top = L->d.pad_left = (int)(k / 2u);
    L->d.dil_y = L->d.dil_x = 1;
    L->d.depthwise = dw;
    L->d.direct_datapath = !dw;
    L->out_zp = out_zp;

    L->W = malloc(n);
    L->bias = calloc((size_t)oc, sizeof *L->bias);
    if (!L->W || !L->bias) return -1;
    for (i = 0; i < n; i++) {
        st = st * 1103515245u + 12345u;
        L->W[i] = (int8_t)((int)((st >> 16) % 11u) - 5);
    }
    for (i = 0; i < oc; i++) L->bias[i] = (int32_t)(i % 13u) - 6;
    L->h = rocket_conv2d_int8_pack_rk3576(fd, &L->d, L->W, L->bias,
                                          0.02f, 0.015f, NULL, 0.35f, 3, w_zp, out_zp);
    return L->h ? 0 : -1;
}

static void layer_drop(int fd, struct layer *L)
{
    if (L->h) rocket_conv2d_int8_weights_free_rk3576(fd, L->h);
    free(L->W); free(L->bias);
    memset(L, 0, sizeof *L);
}

/* One producer, into a cube deep enough to hold both its own groups and the round-32 count
 * a direct consumer of the same channel count would walk. Returns 0 when the channels past
 * `oc` all carry the output zero point, 1 when they carry anything else. */
static int one(int fd, unsigned ic, unsigned oc, unsigned k, int dw, int w_zp,
               int expect_written)
{
    struct layer L;
    rocket_rk3576_cube cu;
    unsigned deep = rocket_rk3576_pad_ic(oc);   /* what a direct consumer walks */
    unsigned c, p, npx = PLANE * PLANE, written = 0, stamped = 0, other = 0, zp = 0;
    int8_t *in = NULL;
    int rc, bad = 0, first_other = 0;

    if (layer_pack(fd, &L, ic, oc, k, dw, -5, w_zp)) {
        printf("   %s ic=%u oc=%u k%u: would not pack\n", dw ? "dw    " : "direct",
               ic, oc, k);
        return 1;
    }
    if (rocket_rk3576_cube_alloc(fd, deep, PLANE, PLANE, &cu) != ROCKET_OK) {
        layer_drop(fd, &L); return 1;
    }
    cube_stamp(fd, &cu);
    in = malloc((size_t)ic * npx);
    if (!in) { rocket_rk3576_cube_free(fd, &cu); layer_drop(fd, &L); return 1; }
    for (c = 0; c < (unsigned)((size_t)ic * npx); c++)
        in[c] = (int8_t)((int)((c * 13u + 7u) % 61u) - 30);

    rc = rocket_conv2d_int8_cube_out_at_rk3576(L.h, &cu);
    if (rc != ROCKET_OK) {
        printf("   %s ic=%-4u oc=%-4u k%u: a %u-channel cube was refused as the output "
               "(rc %d) — the handle writes more groups than it holds\n",
               dw ? "dw    " : "direct", ic, oc, k, deep, rc);
        free(in); rocket_rk3576_cube_free(fd, &cu); layer_drop(fd, &L);
        return 2;
    }
    if (rocket_conv2d_int8_prepacked_rk3576(fd, L.h, in, NULL) != ROCKET_OK) {
        printf("   the producer failed\n");
        free(in); rocket_rk3576_cube_free(fd, &cu); layer_drop(fd, &L);
        return 1;
    }

    /* The live channels are the layer's own business; what this asks about is the rest. */
    for (c = oc; c < deep; c++)
        for (p = 0; p < npx; p++) {
            int v = cube_at(&cu, c, p);
            if (v == (int8_t)SENTINEL) stamped++;
            else if (v == L.out_zp) zp++;
            else { if (!other) first_other = v; other++; }
        }
    written = zp + other;
    if (deep > oc) {
        printf("   %s ic=%-4u oc=%-4u k%u w_zp=%-4d: %2u padding channel(s) — "
               "%6u at the output zero point, %6u never written, %6u other\n",
               dw ? "dw    " : "direct", ic, oc, k, w_zp, deep - oc, zp, stamped, other);
        if (other)
            printf("      the first value that is neither is %d\n", first_other);
        bad = expect_written ? (zp != (deep - oc) * npx)
                             : (stamped != (deep - oc) * npx);
    } else {
        printf("   %s ic=%-4u oc=%-4u k%u w_zp=%-4d: no padding channels (round32 of %u "
               "is %u)\n", dw ? "dw    " : "direct", ic, oc, k, w_zp, oc, deep);
    }
    (void)written;

    free(in);
    rocket_conv2d_int8_cube_out_at_rk3576(L.h, NULL);
    rocket_rk3576_cube_free(fd, &cu);
    layer_drop(fd, &L);
    return bad;
}

/* What a producer's OWN surface carries, which is the cube an adjacent join is handed:
 * how many groups it has, against the round-32 count a direct consumer walks — and, when
 * it carries a tail the host filled rather than the DPU, what is actually in it after a
 * run. `want_deep` is 1 for a producer whose surface has to reach the round-32 count.
 *
 * A DEPTHWISE tile is the case this asks about: the program is told the raw channel count
 * and writes nothing past it, so the surface is a group short unless the library allocates
 * the deeper one and fills the extra group with the output zero point at pack time. Reading
 * it back AFTER an inference is what says that fill survives the program and the sentinel.
 */
static int depth(int fd, unsigned ic, unsigned oc, unsigned k, int dw, int want_deep)
{
    struct layer L;
    rocket_rk3576_cube cu;
    unsigned need = rocket_rk3576_pad_ic(oc);
    int8_t *in = NULL, *out = NULL;
    int bad = 0;

    if (layer_pack(fd, &L, ic, oc, k, dw, -5, 0)) return 1;
    in = malloc((size_t)ic * PLANE * PLANE);
    out = malloc((size_t)oc * PLANE * PLANE);
    if (!in || !out) { layer_drop(fd, &L); free(in); free(out); return 1; }
    memset(in, 1, (size_t)ic * PLANE * PLANE);
    if (rocket_conv2d_int8_prepacked_rk3576(fd, L.h, in, out) != ROCKET_OK) {
        printf("   the producer failed\n"); bad = 1; goto done;
    }
    if (rocket_conv2d_int8_cube_of_rk3576(L.h, &cu) != ROCKET_OK) {
        printf("   %s oc=%-4u: no cube (a padded surface stride or several tiles)\n",
               dw ? "dw    " : "direct", oc);
        bad = want_deep;
        goto done;
    }
    printf("   %s oc=%-4u: its own surface carries %u group(s) = %u channel(s); a direct "
           "consumer of %u channels walks %u — %s",
           dw ? "dw    " : "direct", oc, cu.groups, cu.groups * C2, oc, need,
           cu.groups * C2 >= need ? "deep enough" : "SHORT");
    if (want_deep && cu.groups * C2 < need) { printf(" — EXPECTED DEEP\n"); bad = 1; goto done; }
    /* The tail's declaration, and — on the depthwise path — the bytes it names.
     *
     * A DIRECT tail lies inside the groups the program writes, so the sentinel covers it
     * and a resident surface carries 0xA5 there once the call has re-stamped it. Its
     * content is measured by one() above, which reads a caller's cube the stamp does not
     * ride. What is asked here is only that the cube DECLARES the tail.
     *
     * A DEPTHWISE tail is past the written groups, which is exactly why the sentinel leaves
     * it alone — so the pack-time fill is still standing after an inference, and that is a
     * fact this pass can check directly. */
    if (cu.groups * C2 > oc) {
        unsigned c, p, npx = PLANE * PLANE, off = 0, zp = 0;
        for (c = oc; c < cu.groups * C2; c++)
            for (p = 0; p < npx; p++) {
                if (cube_at(&cu, c, p) == L.out_zp) zp++; else off++;
            }
        printf("; %u tail channel(s): %u at the output zero point, %u not",
               cu.groups * C2 - oc, zp, off);
        if (dw && want_deep && off) bad = 1;
    }
    if (want_deep && (!cu.pad_from || cu.pad_from > oc || cu.pad_value != L.out_zp)) {
        printf(" — the cube does not declare that tail");
        bad = 1;
    }
    if (!want_deep && cu.pad_from) {
        printf(" — the cube declares a tail nobody filled");
        bad = 1;
    }
    printf("\n");
done:
    free(in); free(out);
    layer_drop(fd, &L);
    return bad;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    rocket_bo guard;
    int fd, bad = 0;
    (void)argc; (void)argv;

    if (!hw || !hw->name || strcmp(hw->name, "rk3576")) {
        printf("rk3576_pad_channels: not an RK3576 — skipping\n");
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_pad_channels: no NPU device — skipping\n"); return 2; }
    /* IOVA 0 is a real buffer on this part, so nothing else may land there. */
    if (rocket_bo_alloc(fd, 4096, &guard) < 0) { rocket_close(fd); return 2; }

    printf("== CONTENT: what a producer writes past its live output channels ==\n");
    printf("-- a SYMMETRIC producer --\n");
    bad |= one(fd, 32u, 16u, 1u, 0, 0, 1);
    bad |= one(fd, 32u, 24u, 1u, 0, 0, 1);
    bad |= one(fd, 32u, 144u, 1u, 0, 0, 1);
    bad |= one(fd, 64u, 48u, 3u, 0, 0, 1);
    /* The DEPTHWISE producer, which writes nothing there at all. */
    bad |= one(fd, 144u, 144u, 3u, 1, 0, 0);
    /* AN ASYMMETRIC one. The B term is per output channel and the packer gives every
     * channel of the programmed count the same one, so a padding channel's value is
     * `requant(B*sum(x))` rather than the output zero point — data-dependent, and the whole
     * reason a producer's own weight zero point decides whether a join is sound. */
    printf("-- an ASYMMETRIC producer (a non-zero weight zero point of its own) --\n");
    bad |= one(fd, 32u, 16u, 1u, 0, 12, 1);
    bad |= one(fd, 32u, 24u, 1u, 0, -24, 1);
    bad |= one(fd, 32u, 144u, 1u, 0, 16, 1);
    bad |= one(fd, 64u, 48u, 3u, 0, -7, 1);

    printf("\n== DEPTH: how many groups a producer's own surface carries ==\n");
    bad |= depth(fd, 32u, 16u, 1u, 0, 1);
    bad |= depth(fd, 32u, 24u, 1u, 0, 1);
    bad |= depth(fd, 32u, 144u, 1u, 0, 1);
    /* A DEPTHWISE producer whose channel count fills its last written group: the surface
     * is allocated a group deeper and that group is filled with the output zero point at
     * pack time, so the join a direct consumer of the same count needs is available. */
    bad |= depth(fd, 144u, 144u, 3u, 1, 1);
    bad |= depth(fd, 48u, 48u, 3u, 1, 1);
    bad |= depth(fd, 16u, 16u, 3u, 1, 1);
    /* And one whose count does NOT fill it. The tail lies inside a group the program
     * writes, whether the DPU leaves those channels alone has not been measured, and the
     * surface stays short — so this one is still expected to refuse. */
    bad |= depth(fd, 24u, 24u, 3u, 1, 0);

    rocket_bo_free(fd, &guard);
    rocket_close(fd);
    printf("\n%s\n", bad
        ? "FAIL: a producer's tail is not what a consumer's fold assumes"
        : "PASS: a direct producer's partial output group carries its output zero point at "
          "any weight zero point, and a depthwise one writes nothing there");
    return bad ? 1 : 0;
}
