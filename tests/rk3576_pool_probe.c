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
 * The output BO is stamped with a sentinel through PREP_BO/FINI_BO rather than a bare
 * memset: dirty CPU lines race the DPU's write DMA and the writeback lands on top of
 * the result. And a faulted job retires cleanly at every layer on this part, so an
 * untouched BO is evidence that nothing wrote — never evidence about the encoding.
 *
 * Usage:  rk3576_pool_probe [sweep|gate]      (default: sweep)
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

    if (!strcmp(mode, "gate")) {
        const char *e = getenv("ROCKET_POOL_DST_REG");
        unsigned reg = e && *e ? (unsigned)strtol(e, NULL, 0) : 0x6070u;
        rc = probe_gate(fd, reg, getenv("ROCKET_POOL_VERBOSE") != NULL);
    } else {
        rc = probe_sweep(fd);
    }
    rocket_close(fd);
    return rc > 0 ? 1 : (rc < 0 ? 1 : 0);
}
