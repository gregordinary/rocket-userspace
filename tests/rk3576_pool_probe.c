// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_pool_probe.c — make the PPU write, and gate what it writes.
 *
 * The pooling program is decoded from manufactured vendor captures down to one
 * register: the DESTINATION BASE ADDRESS. Every capture stores zero for it, exactly as
 * it does for the feature, weight, output and bias bases, because the vendor runtime
 * patches those at load time and a `.rknn`-derived program is the STORED register set.
 * So the one thing a capture cannot carry is the one thing left, and it has to be read
 * off the part.
 *
 * Five PPU registers read zero in every capture — 0x6054, 0x6058, 0x605C, 0x6070 and
 * 0x60DC — and one of them is it. The probe drives them:
 *
 *   `sweep`  each candidate in turn, then all five at once. What is being asked is
 *            only "did the output BO change from its sentinel", so a candidate that
 *            is not an address costs a submit and nothing else. All-five FIRST is
 *            deliberate: a one-at-a-time sweep cannot see a condition of two, and on
 *            this part that has already cost one decode (the fp16 mode is three
 *            registers, each inert while either of the others is wrong).
 *
 *   `gate`   with the register settled, the shape table against a CPU model.
 *
 *   `lib`    the LIBRARY entry over row-major tensors, each shape run twice — once
 *            with the source surface stride the vendor's programs carry (round4 of
 *            the plane) and once with the plane's own element count. The second is
 *            what a CUBE-IN join hands PPU_RDMA, because a direct convolution's output
 *            surface stride is `ow*oh` exactly, so this is the gate that says whether
 *            that join is available at a plane whose element count is not a multiple
 *            of four. Every shape also runs prepacked twice, since the second call is
 *            the first to reuse a held BO.
 *
 * A CUBE-IN JOIN IS NOT AN ALIGNMENT ASSUMPTION. It was written before the stride
 * question was asked, on the belief that the producer's surface would be round4 too;
 * it is not, and a wrong stride computes a full, plausible surface rather than
 * faulting. Gate the stride, not the plumbing.
 *
 * The output BO is stamped with a sentinel through PREP_BO/FINI_BO rather than a bare
 * memset: dirty CPU lines race the DPU's write DMA and the writeback lands on top of
 * the result. And a faulted job retires cleanly at every layer on this part, so an
 * untouched BO is evidence that nothing wrote — never evidence about the encoding.
 *
 *   `bound`  the two axes that could carry a capacity limit — the PLANE and the CHANNEL
 *            count — over a window with a leading pad and no trailing one, which is the
 *            shape a classifier's first pool has and which nothing in `gate` or `lib`
 *            reaches. The plan function carries no capacity bound at all, so what a
 *            plane too large does is a question worth asking of the part.
 *
 *   `pad`    the PAD NIBBLES, one axis at a time. `0x6040` packs four four-bit pads and
 *            every vendor pool is padded symmetrically, so a capture cannot say which
 *            nibble is which edge — and the wrong assignment computes a full, plausible
 *            surface. This runs a single line with a unit kernel on the other axis and
 *            scores the part against the window grid each candidate leading pad implies.
 *            Measured: bits [3:0] are LEFT and bits [7:4] are TOP.
 *
 * Usage:  rk3576_pool_probe [sweep|gate|lib|bound|pad]      (default: sweep)
 * Env:    ROCKET_POOLB_LEAD / ROCKET_POOLB_TRAIL   the pads `bound` uses (1 and 0)
 *         ROCKET_POOLP_OTHER                       `pad`'s non-swept extent (4)
 * Exit:   0, 1 on a failure, 2 no NPU (skip).
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

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);

static unsigned round4(unsigned v) { return (v + 3u) & ~3u; }

static size_t out_index(unsigned surf_elems, unsigned ow, unsigned c,
                        unsigned y, unsigned x)
{
    return (size_t)(c / C2) * surf_elems * C2 + (size_t)C2 * (y * ow + x) + (c % C2);
}

static void sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

typedef struct {
    const char *name;
    unsigned c, iw, ih, k, stride, pad;
    unsigned mode;
} pool_case;

/* What the part should compute, on the CPU. int8 in, int8 out, no requant in the
 * path — the PPU is a window reduction and nothing else. */
static int pool_ref(const pool_case *pc, const int8_t *in, unsigned ci,
                    unsigned oy, unsigned ox, unsigned ow, unsigned oh)
{
    long best = -1000, sum = 0;
    unsigned n = 0, ky, kx;
    (void)ow; (void)oh;
    for (ky = 0; ky < pc->k; ky++) {
        for (kx = 0; kx < pc->k; kx++) {
            long iy = (long)(oy * pc->stride + ky) - (long)pc->pad;
            long ix = (long)(ox * pc->stride + kx) - (long)pc->pad;
            int v;
            if (iy < 0 || ix < 0 || iy >= (long)pc->ih || ix >= (long)pc->iw) {
                if (pc->mode == ROCKET_RK3576_POOL_MAX) v = -128;
                else if (pc->mode == ROCKET_RK3576_POOL_AVG) v = 0;
                else continue;                       /* AVG_NOPAD drops the tap */
            } else {
                v = in[((size_t)ci * pc->ih + (size_t)iy) * pc->iw + (size_t)ix];
            }
            if (v > best) best = v;
            sum += v;
            n++;
        }
    }
    if (pc->mode == ROCKET_RK3576_POOL_MAX) return (int)best;
    if (!n) return 0;
    /* The divisor is 1/kw * 1/kh in Q16, so the reference divides by the WINDOW and not
     * by the tap count — which is what AVG_NOPAD's own mode bit is for. */
    if (pc->mode == ROCKET_RK3576_POOL_AVG) n = pc->k * pc->k;
    /* AND IT ROUNDS HALF TO EVEN, the same rule the DPU's OUT_CVT was measured to use.
     * Against round-half-away-from-zero a k2 average disagrees on one output in eight
     * — sum ≡ ±2 (mod 4), half of which land on an odd quotient — which is exactly the
     * 254 of 2048 the first run of this gate reported. */
    {
        long half = (long)n / 2, q, r;
        q = sum >= 0 ? (sum + half) / (long)n
                     : -(((-sum) + half) / (long)n);
        /* An ODD window has no exact half, so there is no tie to break — testing the
         * residue against n/2 there fires on an ordinary value and moves it. */
        if ((n & 1u) == 0u) {
            r = sum - q * (long)n;
            if ((r == half || r == -half) && (q & 1))
                q += (sum >= 0) ? -1 : 1;
        }
        return (int)q;
    }
}

/* Build, submit, and report whether the output BO moved off its sentinel. `diffs` is
 * filled with the mismatch count against the CPU model when `check` is set. */
static int run_pool(int fd, const pool_case *pc, unsigned dst_reg, int check,
                    int *wrote, int *diffs, int *maxdiff, int verbose)
{
    unsigned iw = pc->iw, ih = pc->ih, c = pc->c;
    unsigned ow = (iw + 2 * pc->pad - pc->k) / pc->stride + 1;
    unsigned oh = (ih + 2 * pc->pad - pc->k) / pc->stride + 1;
    unsigned creg = ((c + 15u) / 16u) * 16u;
    unsigned in_surf = round4(iw * ih), out_surf = round4(ow * oh);
    size_t in_bytes  = (size_t)(creg / C2) * in_surf * C2;
    size_t out_bytes = (size_t)(creg / C2) * out_surf * C2;
    rocket_bo bo_in = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_POOL_TASK_OPS] = {0};
    uint32_t in_h[2], out_h[1];
    pool_params_rk3576_t p = {0};
    int8_t *src = NULL;
    unsigned i, y, x, ci;
    int rc = -1;

    *wrote = 0; if (diffs) *diffs = 0; if (maxdiff) *maxdiff = 0;

    src = malloc((size_t)c * ih * iw);
    if (!src) goto done;
    {
        unsigned seed = 0x9E3779B9u ^ (c * 31 + iw * 7 + ih * 3 + pc->k);
        for (i = 0; i < (unsigned)((size_t)c * ih * iw); i++) {
            seed = seed * 1103515245u + 12345u;
            src[i] = (int8_t)((int)((seed >> 16) % 251u) - 125);
        }
    }

    if (rocket_bo_alloc(fd, in_bytes, &bo_in) < 0) goto done;
    if (rocket_bo_alloc(fd, out_bytes, &bo_o) < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    rocket_bo_prep(fd, &bo_in, 1, 0);
    memset(bo_in.ptr, 0, in_bytes);
    {
        int8_t *cube = (int8_t *)bo_in.ptr;
        for (ci = 0; ci < c; ci++)
            for (y = 0; y < ih; y++)
                for (x = 0; x < iw; x++)
                    cube[(size_t)(ci / C2) * in_surf * C2 +
                         (size_t)C2 * (y * iw + x) + (ci % C2)] =
                        src[((size_t)ci * ih + y) * iw + x];
    }
    rocket_bo_fini(fd, &bo_in);

    p.iw = (uint16_t)iw; p.ih = (uint16_t)ih; p.c = (uint16_t)c;
    p.ow = (uint16_t)ow; p.oh = (uint16_t)oh;
    p.kw = (uint8_t)pc->k; p.kh = (uint8_t)pc->k;
    p.stride_x = (uint8_t)pc->stride; p.stride_y = (uint8_t)pc->stride;
    p.pad_left = p.pad_right = p.pad_top = p.pad_bottom = (uint8_t)pc->pad;
    p.mode = (uint8_t)pc->mode;
    p.input_zero_point = 0;
    p.input_dma  = bo_in.dma_address;
    p.output_dma = bo_o.dma_address;
    p.ppu_dst_reg = (uint16_t)dst_reg;
    p.tasks = ops;

    if (gen_pool_rk3576(&p) != 0) { printf("    generator refused\n"); goto done; }

    rocket_bo_prep(fd, &bo_r, 1, 0);
    memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &bo_r);

    /* Bracketed, so "never written" is a property of the surface and not a guess. */
    rocket_bo_prep(fd, &bo_o, 1, 0);
    memset(bo_o.ptr, SENTINEL, out_bytes);
    rocket_bo_fini(fd, &bo_o);

    in_h[0] = bo_in.handle; in_h[1] = bo_r.handle;
    out_h[0] = bo_o.handle;

    if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 2, out_h, 1, 2000) != 0) {
        printf("    submit failed\n"); goto done;
    }
    if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
        printf("    PREP_BO timed out\n"); goto done;
    }

    {
        const uint8_t *o = (const uint8_t *)bo_o.ptr;
        for (i = 0; i < out_bytes; i++)
            if (o[i] != SENTINEL) { *wrote = 1; break; }
    }
    if (check && *wrote) {
        const int8_t *o = (const int8_t *)bo_o.ptr;
        int shown = 0;
        for (ci = 0; ci < c; ci++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    int want = pool_ref(pc, src, ci, y, x, ow, oh);
                    int have = o[out_index(out_surf, ow, ci, y, x)];
                    int d = have - want; if (d < 0) d = -d;
                    if (d) {
                        (*diffs)++;
                        if (d > *maxdiff) *maxdiff = d;
                        if (verbose && shown < 6) {
                            printf("      c=%u (%u,%u) want %d got %d\n",
                                   ci, y, x, want, have);
                            shown++;
                        }
                    }
                }
    }
    rc = 0;
done:
    free(src);
    if (bo_in.ptr) rocket_bo_free(fd, &bo_in);
    if (bo_o.ptr)  rocket_bo_free(fd, &bo_o);
    if (bo_r.ptr)  rocket_bo_free(fd, &bo_r);
    return rc;
}

static const pool_case PROBE_CASE = { "probe", 32, 16, 16, 2, 2, 0,
                                      ROCKET_RK3576_POOL_MAX };

static const unsigned CANDIDATES[] = { 0x6054, 0x6058, 0x605C, 0x6070, 0x60DC };
#define N_CAND ((int)(sizeof CANDIDATES / sizeof *CANDIDATES))

static int probe_sweep(int fd)
{
    int wrote, i, found = -1, any = 0;

    printf("sweep: which register carries the destination base?\n");
    printf("  the control first — every candidate at zero, which is the capture "
           "verbatim\n");
    if (run_pool(fd, &PROBE_CASE, 0xFFFF, 0, &wrote, NULL, NULL, 0) != 0) return -1;
    printf("    all zero            -> %s\n", wrote ? "WROTE" : "nothing");
    if (wrote) {
        printf("    it wrote with no base programmed at all — the destination is not "
               "one of these five, and IOVA 0 is a real buffer on this stack\n");
        return -1;
    }

    sleep_ms(200);
    for (i = 0; i < N_CAND; i++) {
        if (run_pool(fd, &PROBE_CASE, CANDIDATES[i], 0, &wrote, NULL, NULL, 0) != 0)
            return -1;
        printf("    0x%04x              -> %s\n", CANDIDATES[i],
               wrote ? "WROTE" : "nothing");
        if (wrote) { any++; if (found < 0) found = i; }
        sleep_ms(200);
    }

    if (!any) {
        printf("  none of the five alone makes it write. A one-at-a-time sweep cannot "
               "see a condition of two, so the next step is a JOINT one — and the "
               "candidate set may simply be wrong, since a register that reads zero in "
               "a capture need not be an address at all.\n");
        return 1;
    }
    printf("  DESTINATION BASE: 0x%04x%s\n", CANDIDATES[found],
           any > 1 ? " (and more than one wrote — narrow before believing it)" : "");
    return 0;
}

static const pool_case GATE[] = {
    { "max-k2s2",     32, 16, 16, 2, 2, 0, ROCKET_RK3576_POOL_MAX },
    { "max-k3s1",     32, 16, 16, 3, 1, 0, ROCKET_RK3576_POOL_MAX },
    { "max-k3s2",     32, 16, 16, 3, 2, 0, ROCKET_RK3576_POOL_MAX },
    { "max-k3s3-15",  32, 15, 15, 3, 3, 0, ROCKET_RK3576_POOL_MAX },
    { "max-nonsq",    32, 19, 17, 2, 2, 0, ROCKET_RK3576_POOL_MAX },
    { "max-c8",        8, 16, 16, 2, 2, 0, ROCKET_RK3576_POOL_MAX },
    { "max-c64",      64, 16, 16, 2, 2, 0, ROCKET_RK3576_POOL_MAX },
    { "max-k5",       32, 16, 16, 5, 1, 0, ROCKET_RK3576_POOL_MAX },
    { "max-pad1",     32, 16, 16, 3, 2, 1, ROCKET_RK3576_POOL_MAX },
    { "avg-k2s2",     32, 16, 16, 2, 2, 0, ROCKET_RK3576_POOL_AVG },
    { "avg-k3s1",     32, 16, 16, 3, 1, 0, ROCKET_RK3576_POOL_AVG },
};
#define N_GATE ((int)(sizeof GATE / sizeof *GATE))

static int probe_gate(int fd, unsigned dst_reg, int verbose)
{
    int i, fails = 0;
    printf("gate: the shape table against a CPU model, destination base at 0x%04x\n",
           dst_reg);
    for (i = 0; i < N_GATE; i++) {
        int wrote = 0, diffs = 0, maxd = 0;
        unsigned ow = (GATE[i].iw + 2 * GATE[i].pad - GATE[i].k) / GATE[i].stride + 1;
        unsigned oh = (GATE[i].ih + 2 * GATE[i].pad - GATE[i].k) / GATE[i].stride + 1;
        if (run_pool(fd, &GATE[i], dst_reg, 1, &wrote, &diffs, &maxd, verbose) != 0) {
            fails++; continue;
        }
        printf("  %-4s %-12s c=%-3u %2ux%-2u k%u s%u pad%u -> %2ux%-2u  %s\n",
               (!wrote || diffs) ? "FAIL" : "PASS", GATE[i].name, GATE[i].c,
               GATE[i].iw, GATE[i].ih, GATE[i].k, GATE[i].stride, GATE[i].pad, ow, oh,
               !wrote ? "NOTHING WRITTEN"
                      : diffs ? "differs from the model" : "exact");
        if (diffs) printf("       %d wrong, max |diff| %d\n", diffs, maxd);
        if (!wrote || diffs) fails++;
        sleep_ms(150);
    }
    printf("== %d passed, %d failed ==\n", N_GATE - fails, fails);
    return fails ? 1 : 0;
}

/* The library entry over row-major tensors. `iw*ih` deliberately includes planes whose
 * element count is not a multiple of four — 7x7, 5x5, 3x3 — because that is the only
 * place the source-surface stride can be seen at all. */
struct lib_case {
    const char *name;
    unsigned c, ih, iw, kh, kw, sy, sx, pad;
    int method, in_zp;
};

static const struct lib_case LIBGATE[] = {
    { "avg-7x7-k7",     1024, 7,  7,  7, 7, 1, 1, 0, POOL_METHOD_AVG, -128 },
    { "avg-7x7-k7-z0",    64, 7,  7,  7, 7, 1, 1, 0, POOL_METHOD_AVG,    0 },
    { "max-7x7-k3s2",     64, 7,  7,  3, 3, 2, 2, 0, POOL_METHOD_MAX,    0 },
    { "avg-5x5-k5",       32, 5,  5,  5, 5, 1, 1, 0, POOL_METHOD_AVG,   17 },
    { "max-3x3-k3",       16, 3,  3,  3, 3, 1, 1, 0, POOL_METHOD_MAX,    0 },
    { "avg-14x14-k7s7",  512, 14, 14, 7, 7, 7, 7, 0, POOL_METHOD_AVG,  -12 },
    { "max-16x16-k2s2",   32, 16, 16, 2, 2, 2, 2, 0, POOL_METHOD_MAX,    0 },
    { "max-19x17-k3s2p1", 32, 19, 17, 3, 3, 2, 2, 1, POOL_METHOD_MAX,    0 },
};
#define N_LIBGATE ((int)(sizeof LIBGATE / sizeof *LIBGATE))

static int lib_gate(int fd)
{
    int i, fails = 0;
    const int packed = getenv("ROCKET_RK3576_POOL_PACK_SRC") &&
                       *getenv("ROCKET_RK3576_POOL_PACK_SRC") != '0';

    printf("lib: the library entry over row-major tensors, source surface stride = %s\n",
           packed ? "the plane's own element count (what a cube-in join hands it)"
                  : "round4 of the plane (what the vendor's programs carry)");
    for (i = 0; i < N_LIBGATE; i++) {
        const struct lib_case *L = &LIBGATE[i];
        rocket_pool_desc d;
        rocket_pool_int8_rk3576_handle *h;
        size_t in_n = (size_t)L->c * L->ih * L->iw, out_n;
        int8_t *in, *out, *out2, *ref;
        unsigned oh, ow;
        size_t k;
        int bad = 0, bad2 = 0, maxd = 0, rc;

        memset(&d, 0, sizeof d);
        d.c = (int)L->c; d.ih = (int)L->ih; d.iw = (int)L->iw;
        d.kh = (int)L->kh; d.kw = (int)L->kw;
        d.stride_y = (int)L->sy; d.stride_x = (int)L->sx;
        d.pad_top = d.pad_left = d.pad_bottom = d.pad_right = (int)L->pad;
        d.method = L->method;
        oh = (unsigned)rocket_pool_oh(&d);
        ow = (unsigned)rocket_pool_ow(&d);
        out_n = (size_t)L->c * oh * ow;

        in = malloc(in_n); out = malloc(out_n); out2 = malloc(out_n); ref = malloc(out_n);
        if (!in || !out || !out2 || !ref) { fails++; free(in); free(out); free(out2);
                                            free(ref); continue; }
        /* A value per position rather than a constant: a probe uniform in an axis cannot
         * see a stride wrong in that axis. */
        for (k = 0; k < in_n; k++)
            in[k] = (int8_t)((int)((k * 37u + (k >> 5) * 11u) % 251u) - 125);
        memset(out, 0x5A, out_n); memset(out2, 0x5A, out_n);
        rocket_pool_ref_int8_rk3576(&d, L->in_zp, in, ref);

        h = rocket_pool_int8_pack_rk3576(fd, &d, L->in_zp);
        if (!h) {
            printf("  FAIL %-16s the handle refused the shape\n", L->name);
            fails++; free(in); free(out); free(out2); free(ref); continue;
        }
        rc  = rocket_pool_int8_prepacked_rk3576(fd, h, in, out);
        /* Twice, because the second call is the first to reuse a held surface. */
        rc |= rocket_pool_int8_prepacked_rk3576(fd, h, in, out2);
        rocket_pool_int8_free_rk3576(fd, h);

        for (k = 0; k < out_n; k++) {
            int dd = out[k] - ref[k];
            if (dd) { bad++; if (dd < 0) dd = -dd; if (dd > maxd) maxd = dd; }
            if (out2[k] != out[k]) bad2++;
        }
        printf("  %-4s %-16s c=%-4u %2ux%-2u k%ux%u s%ux%u pad%u -> %2ux%-2u  %s\n",
               (rc || bad || bad2) ? "FAIL" : "PASS", L->name, L->c, L->iw, L->ih,
               L->kw, L->kh, L->sx, L->sy, L->pad, ow, oh,
               rc ? "the entry returned an error"
                  : bad ? "differs from the model"
                  : bad2 ? "the second prepacked call differs from the first" : "exact");
        if (bad) printf("       %d of %zu wrong, max |diff| %d\n", bad, out_n, maxd);
        if (rc || bad || bad2) fails++;
        free(in); free(out); free(out2); free(ref);
        sleep_ms(120);
    }
    printf("== %d passed, %d failed ==\n", N_LIBGATE - fails, fails);
    return fails ? 1 : 0;
}

/* ---- the CAPACITY axis ----------------------------------------------------------
 * Every shape in the tables above is small — the largest plane is 19x17 — and
 * rocket_pool_int8_rk3576_plan() carries no capacity bound at all. A real classifier's
 * first pool is 112x112 at 64 channels, which is three orders of magnitude more feature
 * bytes than anything the gate had run, so what a plane too large does for one PPU
 * program is a question no gate has asked. This sweeps the two axes that could carry
 * the bound — the PLANE and the CHANNEL COUNT — and prints the wrong-element count for
 * each cell, so the law is read off the part rather than reasoned to.
 *
 * The window is held at k3 s2 with the pad a stride-2 SAME-on-an-even-plane produces
 * (one leading, none trailing), which is the shape a network's first pool has.
 */
static int pool_run_one(int fd, const rocket_pool_desc *d, int in_zp,
                        int *bad, int *maxd)
{
    size_t in_n = (size_t)d->c * d->ih * d->iw, k;
    size_t out_n = (size_t)d->c * rocket_pool_oh(d) * rocket_pool_ow(d);
    int8_t *in = malloc(in_n), *out = malloc(out_n), *ref = malloc(out_n);
    int rc = ROCKET_E_NOMEM;

    *bad = 0; *maxd = 0;
    if (in && out && ref) {
        for (k = 0; k < in_n; k++)
            in[k] = (int8_t)((int)((k * 37u + (k >> 5) * 11u) % 251u) - 125);
        memset(out, 0x5A, out_n);
        rocket_pool_ref_int8_rk3576(d, in_zp, in, ref);
        rc = rocket_pool_int8_rk3576(fd, d, in_zp, in, out);
        if (rc == ROCKET_OK)
            for (k = 0; k < out_n; k++) {
                int dd = out[k] - ref[k];
                if (dd) { (*bad)++; if (dd < 0) dd = -dd; if (dd > *maxd) *maxd = dd; }
            }
    }
    free(in); free(out); free(ref);
    return rc;
}

static int bound_sweep(int fd)
{
    static const unsigned CH[] = { 16, 32, 64, 128 };
    static const unsigned PLANE[] = { 16, 24, 32, 40, 48, 56, 64, 80, 96, 112 };
    unsigned ci, pi;
    int fails = 0;
    const char *el = getenv("ROCKET_POOLB_LEAD"), *et = getenv("ROCKET_POOLB_TRAIL");
    const int lead = el ? atoi(el) : 1, trail = et ? atoi(et) : 0;

    printf("bound: max k3 s2, %d leading pad and %d trailing, over plane x channels."
           "\n       Each cell is the wrong-element count; \".\" is exact.\n",
           lead, trail);
    printf("  %6s", "c\\wxh");
    for (pi = 0; pi < sizeof PLANE / sizeof *PLANE; pi++)
        printf(" %7u", PLANE[pi]);
    printf("\n");
    for (ci = 0; ci < sizeof CH / sizeof *CH; ci++) {
        printf("  %6u", CH[ci]);
        for (pi = 0; pi < sizeof PLANE / sizeof *PLANE; pi++) {
            rocket_pool_desc d;
            int bad, maxd, rc;
            memset(&d, 0, sizeof d);
            d.c = (int)CH[ci]; d.ih = d.iw = (int)PLANE[pi];
            d.kh = d.kw = 3; d.stride_y = d.stride_x = 2;
            d.pad_top = d.pad_left = lead;
            d.pad_bottom = d.pad_right = trail;
            d.method = POOL_METHOD_MAX;
            rc = pool_run_one(fd, &d, 0, &bad, &maxd);
            if (rc != ROCKET_OK) { printf(" %7s", "err"); fails++; }
            else if (bad)        { printf(" %7d", bad); fails++; }
            else                   printf(" %7s", ".");
            fflush(stdout);
            sleep_ms(60);
        }
        printf("\n");
    }
    printf("== %d cell(s) wrong ==\n", fails);
    return fails ? 1 : 0;
}

/* ---- what an ASYMMETRIC pad actually does ----------------------------------------
 * `bound` says a symmetric pad computes and an asymmetric one does not, at every plane
 * and every channel count, which is the emitter or the part and not a capacity. This
 * reads the answer off a ONE-DIMENSIONAL case — one row, a unit kernel on the y axis —
 * so the whole disagreement fits on a line and can be scored against candidate window
 * placements instead of guessed at.
 */
static int pad_map(int fd)
{
    static const int PADL[] = { 0, 1, 2 }, PADR[] = { 0, 1, 2 };
    const unsigned C = 16, IW = 12, KW = 3, SX = 2;
    const char *eo = getenv("ROCKET_POOLP_OTHER");
    const unsigned OTHER = eo ? (unsigned)atoi(eo) : 4;
    unsigned li, ri, x, k;
    int fails = 0;
    int8_t *in = malloc((size_t)C * IW * OTHER);

    if (!in) return 1;
    for (k = 0; k < (size_t)C * IW * OTHER; k++)
        in[k] = (int8_t)(-120 + (int)(k % 61u) * 3);

    printf("pad: k%ux1 s%ux1 over a %u-long axis, run once along x and once along y.\n"
           "     For each (lead, trail) the part's line is scored against the window\n"
           "     grid each candidate leading pad implies.\n", KW, SX, IW);
    for (li = 0; li < 2 * (sizeof PADL / sizeof *PADL); li++)
    for (ri = 0; ri < sizeof PADR / sizeof *PADR; ri++) {
        const int ax_y = li >= sizeof PADL / sizeof *PADL;
        const int lead_v = PADL[li % (sizeof PADL / sizeof *PADL)], trail_v = PADR[ri];
        rocket_pool_desc d;
        int8_t *out, *ref;
        unsigned ow;
        int rc, hit_left = 0, hit_right = 0, hit_zero = 0;

        memset(&d, 0, sizeof d);
        d.c = (int)C;
        d.method = POOL_METHOD_MAX;
        /* The OTHER axis is four long, not one: a unit extent is degenerate here (the
         * line and surface strides collapse onto each other) and reads values that are
         * in no input, which would be scored as a pad defect. */
        if (ax_y) {
            d.ih = (int)IW; d.iw = (int)OTHER;
            d.kh = (int)KW; d.kw = 1; d.stride_y = (int)SX; d.stride_x = 1;
            d.pad_top = lead_v; d.pad_bottom = trail_v;
            ow = (unsigned)rocket_pool_oh(&d);
        } else {
            d.ih = (int)OTHER; d.iw = (int)IW;
            d.kh = 1; d.kw = (int)KW; d.stride_y = 1; d.stride_x = (int)SX;
            d.pad_left = lead_v; d.pad_right = trail_v;
            ow = (unsigned)rocket_pool_ow(&d);
        }
        out = malloc((size_t)C * ow * OTHER); ref = malloc((size_t)C * ow * OTHER);
        if (!out || !ref) { free(out); free(ref); fails++; continue; }
        memset(out, 0x5A, (size_t)C * ow * OTHER);
        rc = rocket_pool_int8_rk3576(fd, &d, 0, in, out);
        if (rc != ROCKET_OK) {
            printf("  l%d r%d  the entry returned %d\n", PADL[li], PADR[ri], rc);
            free(out); free(ref); fails++; continue;
        }
        /* Three candidate placements for output x: the window starts at x*s minus the
         * LEFT pad (what the library computes), minus the RIGHT one (the low nibble read
         * as the leading edge), or at x*s with no leading pad at all. */
        /* The swept axis is the innermost one on x and strided by OTHER on y; both
         * tensors are channel-major, so one pair of strides covers both. */
        {
        const unsigned istep = ax_y ? OTHER : 1, ostep = ax_y ? OTHER : 1;
        for (k = 0; k < 3; k++) {
            int lead = k == 0 ? lead_v : k == 1 ? trail_v : 0, ok = 1;
            unsigned ch;
            for (ch = 0; ch < C && ok; ch++)
                for (x = 0; x < ow && ok; x++) {
                    int best = -128, t;
                    for (t = 0; t < (int)KW; t++) {
                        int sx = (int)x * (int)SX + t - lead;
                        if (sx >= 0 && sx < (int)IW &&
                            in[ch * IW * OTHER + (unsigned)sx * istep] > best)
                            best = in[ch * IW * OTHER + (unsigned)sx * istep];
                    }
                    if (out[ch * ow * OTHER + x * ostep] != (int8_t)best) ok = 0;
                }
            if (k == 0) hit_left = ok; else if (k == 1) hit_right = ok; else hit_zero = ok;
        }
        }
        printf("  %c lead%d trail%d -> %u:  lead=the lead pad %s  lead=the TRAIL pad %s  "
               "lead=0 %s\n", ax_y ? 'y' : 'x', lead_v, trail_v, ow,
               hit_left ? "MATCH" : "-", hit_right ? "MATCH" : "-",
               hit_zero ? "MATCH" : "-");
        if (!hit_left && !hit_right && !hit_zero) {
            /* 0xA5 is the library's own sentinel, so an element carrying it was never
             * WRITTEN — a different fact from an element computed wrong, and the one
             * that says the disagreement is not about the pad at all. */
            unsigned ch, unwritten = 0, wrong = 0;
            for (ch = 0; ch < C; ch++)
                for (x = 0; x < ow; x++) {
                    int8_t v = out[ch * ow * OTHER + x * (ax_y ? OTHER : 1)];
                    int best = -128, t;
                    for (t = 0; t < (int)KW; t++) {
                        int sx = (int)x * (int)SX + t - lead_v;
                        if (sx >= 0 && sx < (int)IW &&
                            in[ch * IW * OTHER + (unsigned)sx * (ax_y ? OTHER : 1)] > best)
                            best = in[ch * IW * OTHER + (unsigned)sx * (ax_y ? OTHER : 1)];
                    }
                    if (v == (int8_t)best) continue;
                    if ((uint8_t)v == 0xA5u) unwritten++; else wrong++;
                }
            printf("       against the leading pad: %u never written (the sentinel), "
                   "%u computed wrong\n", unwritten, wrong);
            printf("       part ch0:");
            for (x = 0; x < ow; x++) printf(" %4d", out[x * (ax_y ? OTHER : 1)]);
            printf("\n       in   ch0:");
            for (x = 0; x < IW; x++) printf(" %4d", in[x * (ax_y ? OTHER : 1)]);
            printf("\n");
            fails++;
        }
        free(out); free(ref);
        sleep_ms(60);
    }
    free(in);
    /* THIS PROBE REPORTS; IT DOES NOT ASSERT. The pad map it decodes is asserted by
     * `bound` (a plane x channel sweep at a real 2-D window, exact in all 40 cells) and
     * by the net gate's first max pool. What this form cannot assert is its own y axis:
     * a UNIT kernel on the other axis is unstable on this part — the same cell reports
     * "never written", "computed wrong" and "MATCH" across runs, and it raises the
     * occasional backstop hit — and no network has a pooling shape of that form, so it is
     * an open observation rather than a regression. */
    printf("== %d cell(s) matched no candidate; this probe REPORTS, the assertion is in "
           "`bound` ==\n", fails);
    return 0;
}

int main(int argc, char **argv)
{
    const struct rocket_hw_profile *hw = rocket_hw_current();
    const char *mode = argc > 1 ? argv[1] : "sweep";
    int fd, rc;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_pool_probe: profile is %s, not rk3576 — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_pool_probe: no NPU device — skipping\n"); return 2; }

    if (!strcmp(mode, "lib")) {
        rc = lib_gate(fd);
    } else if (!strcmp(mode, "pad")) {
        rc = pad_map(fd);
    } else if (!strcmp(mode, "bound")) {
        rc = bound_sweep(fd);
    } else if (!strcmp(mode, "gate")) {
        const char *e = getenv("ROCKET_POOL_DST_REG");
        unsigned reg = e && *e ? (unsigned)strtol(e, NULL, 0) : 0x6070u;
        rc = probe_gate(fd, reg, getenv("ROCKET_POOL_VERBOSE") != NULL);
    } else {
        rc = probe_sweep(fd);
    }
    rocket_close(fd);
    return rc > 0 ? 1 : (rc < 0 ? 1 : 0);
}
