// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_chain_len.c — how long a chained stream may be, and what the bound is a
 * function of.
 *
 * A cross-layer kick lays every layer's register program out contiguously in one BO and
 * submits the lot as ONE hardware job. On MobileNetV1-224 a 34-program stream is exact
 * and a 35-program one intermittently writes nothing at all, with the layer that reads
 * unwritten varying between the first and the last — so the whole kick fails rather than
 * one program being dropped, and eight redos with confirmed power cycles do not recover
 * it. That is a REFUSAL with a magic number in a shipping path, which is a liability: a
 * frontend with a deeper graph meets it immediately.
 *
 * What is already known is that it is NOT the program count on its own. A single layer
 * forced into 38 row windows — 38 programs, identical in length and in every register but
 * the row offsets — is bit-exact. MobileNet's 35 are a mix of direct and depthwise
 * programs at a dozen geometries. So the candidate is the TRANSITION MIX, and this is the
 * instrument that separates the two: it builds a chain of N layers whose geometry it
 * chooses, so the count and the mix move independently.
 *
 *   uniform N   N identical 1x1 direct convs over one plane. Every program is the same
 *               length and differs only in its addresses. If this fails at the same N a
 *               graph does, the count is the bound after all and the mix is a red
 *               herring; if it does not, the count is excluded at that length.
 *
 *   mixed N     the same N layers alternating DIRECT and 3x3 DEPTHWISE at the same
 *               plane and channel count. Two program shapes, one geometry — so a failure
 *               here is the transition and not the geometry.
 *
 *   geom N      N layers alternating direct and depthwise AND stepping the plane down,
 *               which is what a real graph looks like. The control for the two above.
 *
 *   wide N      N uniform DIRECT layers, the mode that exists to be driven past the bound.
 *
 * EVERY mode takes the FOOTPRINT knobs — ROCKET_CHAINLEN_C channels, ROCKET_CHAINLEN_P
 * plane, ROCKET_CHAINLEN_K the direct leg's kernel — and their defaults are per-mode and
 * unchanged (32 channels over 32x32 with a 1x1 direct leg for the three mixes, 128 over
 * 28x28 with a 3x3 leg for `wide`). MobileNet's own failing stream is neither of the
 * shapes a single mode used to build: its k>1 programs are DEPTHWISE with cubes under 18
 * KiB, the size a 88-program chain is exact at, and its k=1 programs reach the 1 MiB cube
 * that chains cleanly eight deep — so what is untested is the COMBINATION at a large
 * footprint, which is `mixed` at MobileNet's own channels and plane.
 *
 * The KERNEL knob is what separates a budget counted in weight BYTES from one counted in
 * 32x32 weight GROUP PAIRS — a channel sweep at a fixed kernel moves both together and
 * cannot tell them apart.
 *
 *   sweep       the three small-footprint mixes above, each from 8 up, reporting the
 *               first length that is not bit-exact and STOPPING there. `wide` is not
 *               swept: it exists to drive the part past the bound, and past the bound
 *               the kernel faults (see the traps).
 *
 * WHAT IS ASSERTED is the chained answer against the PER-LAYER one, layer by layer,
 * bytes: the same handles run one submit each through
 * rocket_conv2d_int8_prepacked_rk3576() are the reference, because the arithmetic is not
 * what is in question and a CPU model would only add a second thing that can be wrong.
 * The last layer materialises a row-major tensor either way, and every interior surface
 * is compared through the cube it hands on.
 *
 * TRAPS this instrument had to be built around:
 *
 *   A chain BORROWS its handles and freezes their IOVAs, so the handles must have run
 *   once before it is built. The constructor allocates the first layer's feature cube for
 *   exactly this reason, but the SURFACES are allocated at pack time and the links are
 *   asserted against those addresses, so the order here is pack -> link -> chain.
 *
 *   A comparison must not hold a POINTER into a buffer a later run rewrites.
 *
 *   A failing chain leaves the sentinel in a surface, and a surface that reads unwritten
 *   is evidence about THAT surface only — a faulted job retires cleanly on this part.
 *
 *   PAST THE BOUND THE KERNEL FAULTS, so a length that is not bit-exact is not a free
 *   observation. The NPU raises a DMA write error, rocket_job_irq_handler() takes it as
 *   WARN_ON (kernel taint +512), and the IOMMU is left reporting "Enable stall request
 *   timed out" / "FORCE_RESET command timed out" — after which every job on every path
 *   fails, including the per-layer reference at a geometry that passed a minute earlier.
 *   NOTHING measured after the taint moves is a hardware answer about the chain. Read
 *   /proc/sys/kernel/tainted after every length and reboot when it changes.
 *
 *   A STALE BINARY ANSWERS A QUESTION YOU DID NOT ASK. This file's own `wide` mode was
 *   once believed to hang the SoC on the strength of a run in which the shipped binary
 *   predated the mode and exited on its usage line. If a mode prints nothing, check that
 *   the board is running the source you think it is before reading anything into it.
 *
 * Usage: rk3576_chain_len [sweep | uniform N | mixed N | geom N | wide N]  (default: sweep)
 * Exit:  0 the question is answered, 1 the instrument could not run it, 2 no NPU / wrong
 *        chip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"

#define MAXN 96u

enum { MIX_UNIFORM = 0, MIX_ALTERNATE, MIX_GEOM, MIX_WIDE };

/* The FOOTPRINT axis, separate from the length one, and EVERY mode takes it. The three
 * small mixes default to 32 channels over a 32x32 plane, which is a small CBUF allocation
 * and a small surface; MobileNet's own failing stream carries 112x112 and 56x56 planes at
 * up to 512 channels, and `wide` defaults to 128 channels over 28x28. So the defaults are
 * per-mode and unchanged, while the knobs move any mode to any footprint — which is what a
 * MobileNet-shaped mix (large k=1 direct alternating with k=3 depthwise over a 112x112
 * plane) needs, since `mixed` alone was hardcoded small and `wide` alone is uniform.
 * ROCKET_CHAINLEN_C channels, ROCKET_CHAINLEN_P plane, ROCKET_CHAINLEN_K the DIRECT leg's
 * kernel (odd, symmetrically padded so the plane survives it). A depthwise leg stays at
 * k=3, which is what a real graph carries and what keeps the two legs distinguishable.
 *
 * The KERNEL is what separates the two quantities a channel sweep alone cannot tell
 * apart. A direct weight cube is ceil(oc/32)*ceil(ic/32)*1024*kh*kw bytes, so at a fixed
 * kernel its BYTE size and its count of 32x32 GROUP PAIRS move together and no value of C
 * distinguishes them. Moving k holds the pair count and scales the bytes by k*k. */
static unsigned chain_c(int mode)
{
    const char *e = getenv("ROCKET_CHAINLEN_C");
    long def = (mode == MIX_WIDE) ? 128 : 32;
    long v = (e && *e) ? strtol(e, NULL, 0) : def;
    if (v < 32 || v > 2048 || (v % 32)) v = def;
    return (unsigned)v;
}

static unsigned chain_p(int mode)
{
    const char *e = getenv("ROCKET_CHAINLEN_P");
    long def = (mode == MIX_WIDE) ? 28 : 32;
    long v = (e && *e) ? strtol(e, NULL, 0) : def;
    if (v < 4 || v > 224) v = def;
    return (unsigned)v;
}

static unsigned chain_k(int mode)
{
    const char *e = getenv("ROCKET_CHAINLEN_K");
    long def = (mode == MIX_WIDE) ? 3 : 1;
    long v = (e && *e) ? strtol(e, NULL, 0) : def;
    if (v < 1 || v > 7 || !(v & 1)) v = def;
    return (unsigned)v;
}

static const char *MODE_NAME[] = { "uniform", "mixed", "geom", "wide" };

/* One layer of the chain. The plane is carried per layer so `geom` can step it down. */
struct layer {
    int ic, oc, ih, iw, kh, kw, sy, sx, pt, pl, dw;
};

/* Build the layer table for a mode. Every consecutive pair must be cube-linkable, which
 * constrains it: the consumer's input plane is the producer's output plane, the channel
 * count stays a multiple of 32 (below that a handle's own cube carries padding channels a
 * producer does not write), and a depthwise layer's surface stride is round4(ow*oh), so
 * its plane has to make that the plane itself. 32x32, 16x16 and 8x8 all do. */
static unsigned build_layers(int mode, unsigned n, struct layer *L)
{
    unsigned i, plane = chain_p(mode);
    unsigned chan  = chain_c(mode);
    unsigned dk    = chain_k(mode);
    for (i = 0; i < n; i++) {
        int dw = (mode == MIX_ALTERNATE || mode == MIX_GEOM) && (i & 1u);
        L[i].ic = (int)chan; L[i].oc = (int)chan;
        L[i].ih = L[i].iw = (int)plane;
        L[i].sy = L[i].sx = 1;
        L[i].dw = dw;
        /* A depthwise leg stays at 3x3, which is the shape a real graph carries. The
         * direct leg is odd and symmetrically padded, so the plane is the same whatever k
         * is and every consecutive pair stays cube-linkable. */
        if (dw) { L[i].kh = L[i].kw = 3; L[i].pt = L[i].pl = 1; }
        else    { L[i].kh = L[i].kw = (int)dk; L[i].pt = L[i].pl = (int)(dk / 2u); }
        /* `geom` steps the plane down every fourth layer, the way a real graph does. A
         * stride-2 layer would do it too, but the plane is what the next layer's cube is
         * a function of, so halving it here keeps the link trivially checkable. */
        if (mode == MIX_GEOM && (i % 4u) == 3u && plane > 8u) {
            plane /= 2u;
            L[i].sy = L[i].sx = 2;
            L[i].kh = L[i].kw = 2; L[i].pt = L[i].pl = 0;
            L[i].dw = 0;
        }
    }
    return n;
}

static void fill(int8_t *p, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        p[i] = (int8_t)((int)((seed >> 16) % 251u) - 125);
    }
}

/* The weight cube one layer carries, which is one of the two budgets the bound was
 * suspected of being counted in. A direct cube is a 32x32 group pair per (oc, ic) group
 * per tap; a depthwise one is two live bytes per channel per tap over a round16 channel
 * count, which is why MobileNet's k>1 programs are small however wide they are. */
static unsigned long layer_cube_bytes(const struct layer *l)
{
    if (l->dw)
        return (unsigned long)(((unsigned)l->oc + 15u) & ~15u) *
               (unsigned)l->kh * (unsigned)l->kw * 2u;
    return (unsigned long)(((unsigned)l->oc + 31u) / 32u) *
           (((unsigned)l->ic + 31u) / 32u) * 1024u *
           (unsigned)l->kh * (unsigned)l->kw;
}

static size_t layer_out_elems(const struct layer *l)
{
    rocket_conv2d_desc d;
    memset(&d, 0, sizeof d);
    d.ic = l->ic; d.oc = l->oc; d.ih = l->ih; d.iw = l->iw;
    d.kh = l->kh; d.kw = l->kw; d.stride_y = l->sy; d.stride_x = l->sx;
    d.pad_top = l->pt; d.pad_left = l->pl; d.dil_y = d.dil_x = 1;
    return (size_t)l->oc * rocket_conv2d_oh(&d) * rocket_conv2d_ow(&d);
}

/*
 * Run one length. Returns 0 bit-exact, 1 a difference, -1 the instrument could not build
 * it (which is NOT a hardware answer and is reported as such).
 */
static int run_len(int fd, int mode, unsigned n, int verbose)
{
    struct layer L[MAXN];
    rocket_conv2d_int8_weights_rk3576 *h[MAXN];
    int8_t *W[MAXN], *bias_unused = NULL;
    int32_t *B[MAXN];
    rocket_conv2d_int8_chain_rk3576 *c = NULL;
    int8_t *in = NULL, *ref = NULL, *got = NULL;
    size_t in_n, out_n;
    unsigned i, programs = 0;
    int rc = -1, bad = 0;

    (void)bias_unused;
    memset(h, 0, sizeof h); memset(W, 0, sizeof W); memset(B, 0, sizeof B);
    build_layers(mode, n, L);

    /* Say what was actually built, so a log line stands on its own. The direct weight cube
     * is what the two candidate budgets are counted in, so report it and its running sum
     * rather than leaving the reader to recompute it from the environment. */
    {
        unsigned long total = 0, dcube = 0, wcube = 0;
        unsigned ndw = 0;
        for (i = 0; i < n; i++) {
            unsigned long b = layer_cube_bytes(&L[i]);
            total += b;
            if (L[i].dw) { ndw++; if (b > wcube) wcube = b; }
            else if (b > dcube) dcube = b;
        }
        printf("  %s c=%d p=%d kdirect=%d: %u direct (cube <=%lu B) + %u depthwise "
               "(cube <=%lu B), %lu B of cube in the stream\n",
               MODE_NAME[mode], L[0].ic, L[0].ih, L[0].dw ? L[1].kh : L[0].kh,
               n - ndw, dcube, ndw, wcube, total);
    }

    in_n  = (size_t)L[0].ic * L[0].ih * L[0].iw;
    out_n = layer_out_elems(&L[n - 1]);
    in  = malloc(in_n);
    ref = malloc(out_n);
    got = malloc(out_n);
    if (!in || !ref || !got) goto done;
    fill(in, in_n, 0x1234u);

    for (i = 0; i < n; i++) {
        rocket_conv2d_desc d;
        size_t wn = L[i].dw ? (size_t)L[i].oc * L[i].kh * L[i].kw
                            : (size_t)L[i].oc * L[i].ic * L[i].kh * L[i].kw;
        memset(&d, 0, sizeof d);
        d.ic = L[i].ic; d.oc = L[i].oc; d.ih = L[i].ih; d.iw = L[i].iw;
        d.kh = L[i].kh; d.kw = L[i].kw;
        d.stride_y = L[i].sy; d.stride_x = L[i].sx;
        d.pad_top = L[i].pt; d.pad_left = L[i].pl;
        d.dil_y = d.dil_x = 1;
        d.depthwise = L[i].dw;
        W[i] = malloc(wn);
        B[i] = calloc(L[i].oc, sizeof **B);
        if (!W[i] || !B[i]) goto done;
        fill(W[i], wn, 0x9E37u + i * 7u);
        /* Scales chosen so the requant neither saturates nor collapses: the question is
         * whether the stream ran, and a layer whose output is all one value cannot say. */
        h[i] = rocket_conv2d_int8_pack_rk3576(fd, &d, W[i], B[i],
                                              0.02f, 0.02f, NULL, 0.25f, 0, 0, 0);
        if (!h[i]) {
            printf("  n=%-3u %s: layer %u would not pack\n", n, MODE_NAME[mode], i);
            goto done;
        }
    }

    /* One run per layer, row-major, which is both the reference and what allocates
     * everything the links are then asserted against. */
    {
        const int8_t *cur = in;
        int8_t *a = NULL, *b = NULL;
        size_t cap = 0;
        for (i = 0; i < n; i++) {
            size_t e = layer_out_elems(&L[i]);
            if (e > cap) cap = e;
        }
        a = malloc(cap); b = malloc(cap);
        if (!a || !b) { free(a); free(b); goto done; }
        for (i = 0; i < n; i++) {
            int8_t *dst = (i & 1u) ? b : a;
            if (rocket_conv2d_int8_prepacked_rk3576(fd, h[i], cur, dst) != ROCKET_OK) {
                printf("  n=%-3u the per-layer reference failed at layer %u\n", n, i);
                free(a); free(b); goto done;
            }
            cur = dst;
        }
        memcpy(ref, cur, out_n);
        free(a); free(b);
    }

    /* Link, then chain. */
    for (i = 0; i + 1u < n; i++) {
        rocket_rk3576_cube cube;
        if (rocket_conv2d_int8_cube_of_rk3576(h[i], &cube) != ROCKET_OK ||
            rocket_conv2d_int8_cube_in_rk3576(h[i + 1], &cube) != ROCKET_OK ||
            rocket_conv2d_int8_cube_out_rk3576(h[i], 1) != ROCKET_OK) {
            printf("  n=%-3u layers %u and %u would not link — the instrument cannot ask "
                   "the question at this geometry\n", n, i, i + 1u);
            goto done;
        }
    }
    for (i = 0; i < n; i++) programs += rocket_conv2d_int8_programs_rk3576(h[i]);

    c = rocket_conv2d_int8_chain_new_rk3576(fd, h, n);
    if (!c) {
        printf("  n=%-3u %u program(s): the chain was REFUSED (raise "
               "ROCKET_RK3576_CHAIN_MAX_TASKS)\n", n, programs);
        goto done;
    }
    memset(got, 0x5A, out_n);
    if (rocket_conv2d_int8_chain_run_rk3576(fd, c, in, got) != ROCKET_OK) {
        printf("  n=%-3u %u program(s): the KICK RETURNED AN ERROR\n", n, programs);
        rc = 1; goto done;
    }
    {
        size_t k, diff = 0, first = 0;
        for (k = 0; k < out_n; k++)
            if (got[k] != ref[k]) { if (!diff) first = k; diff++; }
        bad = diff != 0;
        printf("  %-4s n=%-3u %2u program(s), %u kick(s)  %s",
               bad ? "FAIL" : "PASS", n, programs,
               rocket_conv2d_int8_chain_kicks_rk3576(c),
               bad ? "" : "bit-exact against the per-layer run\n");
        if (bad)
            printf("%zu of %zu elements differ, first at %zu (%d vs %d)\n",
                   diff, out_n, first, ref[first], got[first]);
        else if (verbose)
            (void)0;
    }
    rc = bad;

done:
    if (c) rocket_conv2d_int8_chain_free_rk3576(fd, c);
    for (i = 0; i < n; i++) {
        if (h[i]) rocket_conv2d_int8_weights_free_rk3576(fd, h[i]);
        free(W[i]); free(B[i]);
    }
    free(in); free(ref); free(got);
    return rc;
}


/*
 * The routine sweep, and it deliberately stops short of the hardware fault.
 *
 * PAST ITS BOUND A CHAIN DOES NOT MERELY COMPUTE THE WRONG ANSWER — the NPU raises a DMA
 * error, rocket_job_irq_handler() takes it as WARN_ON, and the IOMMU is left wedged for
 * every later job on every path. So this sweep does two things a bisection would not:
 *
 *   it STOPS a mix at its first non-bit-exact length, because a longer one says nothing
 *   the first already said and each attempt is another chance to take the kernel down; and
 *
 *   it leaves `wide` out. `wide` exists to drive the part PAST the bound, which is a
 *   probe and not a gate — run it deliberately, one length at a time, checking
 *   /proc/sys/kernel/tainted after each.
 */
static int sweep(int fd)
{
    static const unsigned LEN[] = { 8, 16, 24, 30, 33, 34, 35, 36, 38, 40, 48, 56, 64, 80 };
    int mode, failed = 0;

    printf("SWEEP: the first length that is not bit-exact, per mix.\n");
    printf("  Raise ROCKET_RK3576_CHAIN_MAX_TASKS past the length being asked for, or the "
           "library refuses before the hardware is reached.\n");
    printf("  `wide` is NOT swept here — it drives the part past a bound that faults the "
           "kernel. Run it on purpose.\n\n");
    for (mode = MIX_UNIFORM; mode <= MIX_GEOM; mode++) {
        unsigned k;
        int first_bad = -1;
        printf("-- %s --\n", MODE_NAME[mode]);
        for (k = 0; k < sizeof LEN / sizeof *LEN; k++) {
            int r;
            if (LEN[k] > MAXN) break;
            r = run_len(fd, mode, LEN[k], 0);
            if (r < 0) continue;               /* not a hardware answer */
            if (r > 0) { first_bad = (int)LEN[k]; break; }   /* stop; see above */
        }
        if (first_bad < 0)
            printf("   every length asked for is bit-exact in this mix\n\n");
        else {
            printf("   the first length that is not bit-exact is %d layers\n\n", first_bad);
            failed++;
        }
    }
    return failed ? 0 : 0;      /* a failing length is the RESULT, not a test failure */
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *mode = argc > 1 ? argv[1] : "sweep";
    int fd, rc;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_chain_len: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_chain_len: no NPU device — skipping\n"); return 2; }

    if (!strcmp(mode, "sweep")) {
        rc = sweep(fd);
    } else {
        int m = !strcmp(mode, "uniform") ? MIX_UNIFORM
              : !strcmp(mode, "mixed")   ? MIX_ALTERNATE
              : !strcmp(mode, "geom")    ? MIX_GEOM
              : !strcmp(mode, "wide")    ? MIX_WIDE : -1;
        unsigned n = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 0) : 34u;
        if (m < 0 || n < 2u || n > MAXN) {
            printf("usage: rk3576_chain_len [sweep | uniform N | mixed N | geom N | wide N]\n");
            rocket_close(fd);
            return 1;
        }
        rc = run_len(fd, m, n, 1) < 0 ? 1 : 0;
    }
    rocket_close(fd);
    return rc;
}
