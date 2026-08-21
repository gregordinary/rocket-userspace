// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_first_light.c — does the RK3576 conv emitter make the NPU COMPUTE?
 *
 * Runs on an RK3576 board. Everything above the geometry encoder was already known
 * to work there — jobs submit, the fence signals, no IOMMU faults — while the
 * RK3588 register program made the DPU write nothing at all. With the RK3576
 * emitter, and the coefficient buffer packed the way this part wants it, the
 * emitter computes a k=3 SAME 32x32 int8 conv (IC=OC=32) over a dense random weight
 * set bit-exactly over the whole surface, border included, over the full signed
 * int8 feature range. The feature domain needs no centering and the DPU epilogue is
 * exact for negative accumulators.
 *
 * The envelope is every geometry whose INPUT channel count is a multiple of 32 and
 * whose plane fits the CBUF: IC=OC of 32 and 64, planes 8x8 to 80x80, k1/k3/k5,
 * stride 1 and 2, VALID and SAME, feature tensors varying on all three axes. OC is
 * unconstrained. Outside it, a partial input-channel group (IC 8/16/17/48) computes
 * wrong at every geometry, and a plane too large for the CBUF comes back saturated
 * across the whole surface.
 *
 * ROCKET_FL_FEAT=11 is the single-source feature: every channel flat at
 * ROCKET_FL_FEATVAL except ROCKET_FL_FEATCHAN, which carries ROCKET_FL_FEATCHANVAL,
 * optionally at one position only (ROCKET_FL_FEATCHANPOS). Sweeping the perturbed
 * channel against the "wrong output channels" line maps which source byte reaches
 * which output — which is what identified the DPU shift word: with BS_BASE_ADDR1
 * (0x5024) left at zero the DPU reads its two shift fields out of whichever BO
 * lands at IOVA 0, and byte 0 / byte 1 of the feature cube silently attenuate the
 * result by a power of two per sign. ROCKET_FL_GUARD claims IOVA 0 with a
 * zero-filled BO, which is the negative control for that whole class of fault.
 *
 * The test separates the ways "it computed" can fail, one probe per invocation so
 * each probe can be given its own NPU power session:
 *
 *   bias      features and weights zero, a distinctive per-channel bias. Exercises
 *             the DPU epilogue alone — BS bias add, OUT_CVT requant, the writer.
 *             A correct result here means everything downstream of the MAC works,
 *             so a zero result from the conv probes is a CNA/CSC/CMAC problem.
 *   identity  conv with an identity weight matrix on the centre tap: the output
 *             cube must be the input cube, elementwise, wherever the window is
 *             wholly in bounds. A spatial or channel shuffle shows up as
 *             structure, so a layout bug is readable straight off the dump.
 *   random    conv with dense random weights against a CPU model of conv +
 *             requant. Dense weights are what tell a LAYOUT fault (garbage out)
 *             apart from an OPERAND fault (exactly zero out): a permutation of a
 *             dense weight set still produces non-zero results. ROCKET_FL_WMAX
 *             bounds their magnitude and ROCKET_FL_WIC confines them to the first N
 *             input channels.
 *   wcurve    one weight per output channel, stepping across the int8 range, against
 *             a uniform feature — so the output plane IS the weight transfer curve.
 *             ROCKET_FL_WBASE/_WSTEP window it, _WDENSE spreads the same weight over
 *             every input channel, _WICSEL picks which input channel carries it.
 *   pad       features zero, weights the identity, and a NON-ZERO border pad
 *             constant (ROCKET_FL_IZP=0 -> CNA_PAD_CON1 = -128). The pad taps are
 *             injected by the CNA at window-formation time, NOT fetched by the
 *             feature DMA, so this probe drives the MAC with an operand the
 *             feature path never touches. A non-zero border ring therefore means
 *             the MAC and the weight path are live and the FEATURE path is the
 *             empty one; an all-zero result means the weights or the MAC are.
 *             Needs k > 1 and padding, so it is skipped at k=1.
 *
 * GEOMETRY. The default is deliberately minimal — 1x1 kernel, no padding, no CBUF
 * row window, IC and OC exact multiples of the cube groups — but every dimension
 * is settable, because "the emitter is right for the captured shapes and wrong
 * where it extrapolates" and "the register program is not the whole story" are
 * different diagnoses and only a capture-shaped run separates them:
 *
 *   ROCKET_FL_IC / _OC / _IW / _IH   feature and channel dims   (default 32/32/8/8)
 *   ROCKET_FL_K                      square kernel              (default 1)
 *   ROCKET_FL_STRIDE                 stride                     (default 1)
 *   ROCKET_FL_SAME                   1 = SAME padding, TFLite's asymmetric split
 *   ROCKET_FL_IZP                    input zero point -> pad constant (default 0x80,
 *                                    i.e. a zero pad tap)
 *   ROCKET_FL_ICREG / _OCREG         channel counts as told to the REGISTERS only,
 *                                    defaulting to what the hardware needs
 *                                    (rocket_rk3576_pad_ic / _pad_oc); set either
 *                                    back to the logical count to reproduce the
 *                                    defect it corrects.
 *                                    The data stays ROCKET_FL_IC / _OC. The weight
 *                                    cube's index depends on either count only
 *                                    through ceil(n/32), and the feature cube is
 *                                    calloc'd and padded to the same group, so
 *                                    rounding up to 32 changes nothing but the
 *                                    register program (plus the output and
 *                                    coefficient allocations, which follow OCREG).
 *                                    That is what separates a channel-rounding rule
 *                                    in the registers from a cube-packing one.
 *
 * Two settings reproduce geometries the vendor itself captured, so that no
 * register value in the emitted program is an extrapolation:
 *   IC=32 OC=32 IW=IH=32 K=1 STRIDE=1            (the vendor "pwB" pointwise)
 *   IC=32 OC=32 IW=IH=80 K=5 STRIDE=2 SAME=1     (the "conv2d" capture's shape)
 *
 * Usage: rk3576_first_light [bias|identity|random|pad] [repeat]
 *   `repeat` submits the same job N times in one process without an intervening
 *   power cycle — the cold-start discriminator. On a kernel whose `rocket` still
 *   programs PC_TASK_CON with the RK3588 task-number width, only run 1 of each NPU
 *   power session computes, and the rest come back empty unless ROCKET_FL_GAP_MS
 *   puts a gap past the 50 ms runtime-PM autosuspend between runs.
 *
 *   ROCKET_FL_TIME=1 reports the split's wall time: total, submit-and-wait, host
 *   regcmd emission, and per task.
 *
 * Exit: 0 pass, 1 fail, 2 skip (no NPU).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_hw.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"

#define C2   16                 /* int8 cube channel atom */
#define SENTINEL 0xAA

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);
int weight_conv_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

/* Runtime geometry. Set once in main from the environment; every buffer, the CPU
 * model and the register program all read these, so a shape change is one place. */
static int IC = 32, OC = 32, IW = 8, IH = 8, K = 1, STRIDE = 1, SAME = 0;
/* Channel counts as told to the REGISTERS. Default to the logical counts; the two
 * *REG knobs separate an encoding rule from a buffer-layout one (see the header). */
static int ICREG = 32, OCREG = 32;
static int OW, OH, PAD_LEAD, IZP = 0x80;
static int OBYTES;

static int env_int(const char *name, int dflt)
{
    const char *e = getenv(name);
    return (e && *e) ? (int)strtol(e, NULL, 0) : dflt;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* The requant the emitter programs. The register holds `shift`; the DPU computes
 * out = sat8( (acc * scale) >> shift ) + offset. The emitter derives scale and the
 * pre-decrement shift from the fp32 conv scale exactly as the vendor (QNNPACK)
 * does and writes shift-1, so the CPU model shifts by that same register value. */
static void requant_params(float conv_scale, unsigned *scale, unsigned *shift_reg)
{
    union { float f; uint32_t u; } cv;
    uint32_t bits;
    cv.f = conv_scale;
    bits = cv.u;
    *shift_reg = 127u + 31u - 32u - (bits >> 23) + 16u - 1u;
    *scale = ((bits >> 9) & 0x7FFFu) + 1u;
    if (*scale < (1u << 14)) *scale |= (1u << 14);
}

/* The DPU rounds to nearest rather than truncating: a pad-probe border whose exact
 * value is -10.0006 comes back as -10, not the -11 an arithmetic shift gives. Adding
 * half an LSB before the shift reproduces every border pixel of that probe exactly. */
static int64_t requant_raw(int64_t acc, unsigned scale, unsigned shift_reg, int offset)
{
    int64_t half = shift_reg ? ((int64_t)1 << (shift_reg - 1)) : 0;
    return ((acc * (int64_t)scale + half) >> shift_reg) + offset;
}

static int requant_apply(int64_t acc, unsigned scale, unsigned shift_reg, int offset)
{
    int64_t v = requant_raw(acc, scale, shift_reg, offset);
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int)v;
}

/* The two candidate saturation domains for the writer, as the byte the probe reads
 * back through an int8 pointer. A signed writer clamps to [-128,127]; an unsigned
 * one clamps to [0,255] and the same pointer then reads 200 as -56. Which one the
 * RK3576 uses is exactly what the mismatch table below is for. */
static int sat_signed(int64_t v)
{
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int)v;
}

static int sat_unsigned(int64_t v)
{
    if (v > 255) v = 255;
    if (v < 0)   v = 0;
    return (int)(int8_t)(uint8_t)v;
}

/* What the output surface looks like, in the three terms that tell probes apart:
 * untouched (still the sentinel), written-but-empty (all zero), or real data. */
struct surface_stat { int sentinel, zero, other, distinct; };

static struct surface_stat classify(const int8_t *buf, int n)
{
    struct surface_stat s = {0, 0, 0, 0};
    unsigned char seen[256] = {0};
    int i;
    for (i = 0; i < n; i++) {
        unsigned char b = (unsigned char)buf[i];
        if (!seen[b]) { seen[b] = 1; s.distinct++; }
        if (b == SENTINEL)   s.sentinel++;
        else if (b == 0)     s.zero++;
        else                 s.other++;
    }
    return s;
}

static void print_stat(const char *tag, struct surface_stat s, int n)
{
    printf("  %s: %d/%d untouched, %d zero, %d nonzero, %d distinct byte values\n",
           tag, s.sentinel, n, s.zero, s.other, s.distinct);
    if (s.sentinel == n)      printf("        -> the DPU did not write at all\n");
    else if (s.other == 0)    printf("        -> the DPU wrote a full but EMPTY surface\n");
}

struct job {
    conv_params_t p;
    const int8_t *in_cube, *w_cube;
    const int32_t *bias;
};

/* The coefficient buffer at BS_BASE_ADDR. The library packs the layout this part
 * actually wants (rocket_rk3576_pack_coeff — 64-byte groups of eight channels,
 * A/B/C); the probe keeps the WRONG layout reachable as a negative control, because
 * the two failure modes are worth being able to reproduce on demand:
 *
 *   ROCKET_FL_ABC=0   a flat per-OC int32 array, which is what the RK3588 takes.
 *                     Opt-in, because it is a negative control: every probe here
 *                     fails under it, and the failures read like encoder bugs.
 *                     The part reads that array AS the group structure, so the bias
 *                     lands on the first 16 channels only, alternating, at 1024x —
 *                     and every C term reads zero.
 *   ROCKET_FL_ABC_C   overrides the C multiplier. C=0 does not merely drop the bias:
 *                     it gates the whole BS stage, so the DPU writes a full,
 *                     correctly sized, entirely EMPTY surface. That is what an
 *                     all-zero coefficient buffer produces, and it is indis-
 *                     tinguishable by inspection from a broken geometry encoder.
 *
 * The buffer is always sized for the group layout so both paths submit the same
 * allocation.
 */
static int bias_bytes(void) { return (int)rocket_rk3576_coeff_bytes((unsigned)OCREG); }

/* The int8 weight cube pads BOTH channel axes to groups of 32, so a shape whose IC
 * or OC is not a multiple of 32 needs more room than OC*IC*K*K — sizing it by the
 * logical count overruns the buffer and reads garbage weights. */
static size_t w_cube_bytes(void)
{
    return (size_t)((OCREG + 31) / 32) * ((ICREG + 31) / 32) * 32 * 32 * K * K;
}

/* The feature cube, sized to the same 32-channel group the WEIGHT cube pads to.
 * A partial ic group leaves the weight cube carrying zeros out to 32 channels, so
 * the products past ic are zero — but only if those feature reads land inside the
 * buffer. Sizing by ceil(IC/16) surfaces instead leaves them off the end. */
static size_t in_cube_bytes(void)
{
    int icp = env_int("ROCKET_FL_ICPAD", 1) ? ((ICREG + 31) / 32) * 32 : IC;
    return (size_t)((icp + C2 - 1) / C2) * IH * IW * C2;
}

static void r76_pack_bias(void *dst, const int32_t *bias)
{
    const char *abc = getenv("ROCKET_FL_ABC");
    const char *cs  = getenv("ROCKET_FL_ABC_C");
    uint8_t *b = (uint8_t *)dst;
    int oc;

    if (abc && *abc == '0') {                    /* negative control: the flat layout */
        memset(b, 0, (size_t)bias_bytes());
        memcpy(b, bias, (size_t)OC * 4);
        return;
    }
    /* Packed for OCREG channels: any channel past the logical OC takes a zero bias
     * and the same C=1, so a padded group is a real channel computing zero rather
     * than a group the BS stage sees gated off. */
    rocket_rk3576_pack_coeff(b, (size_t)bias_bytes(), bias, (unsigned)OCREG);
    if (cs && *cs) {                             /* C sweep, on top of the packed buffer */
        int16_t c = (int16_t)strtol(cs, NULL, 0);
        for (oc = 0; oc < OCREG; oc++)
            memcpy(b + (oc / 8) * 64 + 48 + (oc % 8) * 2, &c, sizeof c);
    }
}

/* Did one row window's own output rows come back written? The output cube is
 * NC1HWC2, so a row run is NOT contiguous — it is one span per channel group, each
 * at the group's own surface offset. Scanning only the first span would call a task
 * written on the strength of channel group 0 alone. */
static int task_rows_written(const int8_t *o, unsigned oy0, unsigned oh)
{
    unsigned groups = (unsigned)((OCREG + C2 - 1) / C2), g, i;
    size_t surf = (size_t)OW * OH * C2;
    for (g = 0; g < groups; g++) {
        const int8_t *p = o + g * surf + (size_t)oy0 * OW * C2;
        for (i = 0; i < oh * (unsigned)OW * C2; i++)
            if ((uint8_t)p[i] != SENTINEL) return 1;
    }
    return 0;
}

static int submit_once(int fd, const struct job *j, int8_t *out)
{
    rocket_bo in_bo = {0}, w_bo = {0}, b_bo = {0}, o_bo = {0}, r_bo = {0};
    rocket_bo g_bo = {0};
    conv_params_t p = j->p;
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    size_t in_bytes = in_cube_bytes();
    size_t w_bytes  = w_cube_bytes();
    uint32_t in_h[4], out_h[1];
    int rc = -1;

    /* IOVA 0 is a REAL buffer on this stack — the per-fd address space bump-starts
     * at 0, so whichever BO is allocated first is addressable as 0. Any operand DMA
     * whose base register the program leaves at 0 therefore reads that buffer
     * instead of reading nothing. ROCKET_FL_GUARD claims the low addresses with a
     * zero-filled BO first, which moves the feature cube off 0 and is the direct
     * test for whether a stray read at 0 is what perturbs the result. */
    if (env_int("ROCKET_FL_GUARD", 0)) {
        if (rocket_bo_alloc(fd, (size_t)env_int("ROCKET_FL_GUARD", 0), &g_bo) < 0) goto done;
        rocket_bo_prep(fd, &g_bo, 1, 0);
        memset(g_bo.ptr, env_int("ROCKET_FL_GUARDVAL", 0) & 0xFF, g_bo.size);
        rocket_bo_fini(fd, &g_bo);
    }
    if (rocket_bo_alloc(fd, in_bytes, &in_bo)  < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes,  &w_bo)   < 0) goto done;
    if (rocket_bo_alloc(fd, (size_t)bias_bytes(), &b_bo) < 0) goto done;
    if (rocket_bo_alloc(fd, (size_t)OBYTES, &o_bo) < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &r_bo) < 0) goto done;

    rocket_bo_prep(fd, &in_bo, 1, 0); memcpy(in_bo.ptr, j->in_cube, in_bytes); rocket_bo_fini(fd, &in_bo);
    rocket_bo_prep(fd, &w_bo,  1, 0); memcpy(w_bo.ptr,  j->w_cube,  w_bytes);  rocket_bo_fini(fd, &w_bo);
    rocket_bo_prep(fd, &b_bo,  1, 0); r76_pack_bias(b_bo.ptr, j->bias);        rocket_bo_fini(fd, &b_bo);
    rocket_bo_prep(fd, &o_bo,  1, 0); memset(o_bo.ptr,  SENTINEL, (size_t)OBYTES); rocket_bo_fini(fd, &o_bo);

    p.tasks       = ops;
    p.input_dma   = in_bo.dma_address;
    p.weights_dma = w_bo.dma_address;
    p.bias_dma    = b_bo.dma_address;
    p.output_dma  = o_bo.dma_address;
    if (env_int("ROCKET_FL_ADDRS", 0))
        printf("  IOVA: in %#llx  w %#llx  bias %#llx  out %#llx\n",
               (unsigned long long)in_bo.dma_address, (unsigned long long)w_bo.dma_address,
               (unsigned long long)b_bo.dma_address, (unsigned long long)o_bo.dma_address);

    in_h[0] = in_bo.handle; in_h[1] = w_bo.handle;
    in_h[2] = b_bo.handle;  in_h[3] = r_bo.handle;
    out_h[0] = o_bo.handle;

    /* One task per row window. ROCKET_FL_ROWS turns the split on; without it the
     * plan is a single task covering the whole plane, so the emitted program is
     * byte-identical to the un-windowed one and this loop runs once.
     *
     * ROCKET_FL_TASK_GAP_MS defaults to a gap between tasks because a `rocket` that
     * still programs PC_TASK_CON with the RK3588 task-number width computes only the
     * first job of each NPU power session, and back-to-back tasks there leave every
     * window after the first unwritten. Set it to 0 on a driver that carries the
     * per-SoC width; the per-task wall check below covers either case. */
    {
        rocket_rk3576_row_task plan[512];
        unsigned n = 1, t, r, walled = 0;
        int wrote = 1, gap = env_int("ROCKET_FL_TASK_GAP_MS", 400);
        /* Wall time of the split, in the two parts that scale differently: the
         * host regcmd emission per window, and the submit-plus-wait per window. */
        double t_loop0, t_emit = 0.0, t_dev = 0.0, t_mark;

        if (env_int("ROCKET_FL_ROWS", 0)) {
            conv_params_t q = p;
            q.ih_full = (uint16_t)IH; q.oh_full = (uint16_t)OH;
            if (rocket_rk3576_plan_rows(&q, 0, plan, 512, &n) < 0) {
                printf("  row planning failed\n"); rc = -1; goto done;
            }
        } else {
            plan[0].iy0 = 0; plan[0].ih = (uint16_t)IH;
            plan[0].oy0 = 0; plan[0].oh = (uint16_t)OH;
            plan[0].pad_top = (uint8_t)PAD_LEAD;
            plan[0].feature_off = plan[0].output_off = 0;
        }

        t_loop0 = now_ms();
        for (t = 0; t < n; t++) {
            conv_params_t q = p;
            q.ih = plan[t].ih;  q.oh = plan[t].oh;
            q.pad_top = plan[t].pad_top;
            q.input_dma  = p.input_dma  + plan[t].feature_off;
            q.output_dma = p.output_dma + plan[t].output_off;
            /* Always, not only for a split: a single-task PLAN still carries a window
             * shorter than the plane whenever the last output row does not reach the
             * last input row (any stride > 1 that leaves trailing rows unread). The
             * emitter would then take the DDR channel-group stride from the window and
             * every group past the first would read at the wrong offset. */
            if (env_int("ROCKET_FL_ROWS", 0)) { q.ih_full = (uint16_t)IH; q.oh_full = (uint16_t)OH; }
            if (n > 1 && env_int("ROCKET_FL_ROWS_V", 0))
                printf("  task %u/%u: in rows [%u,%u) pad_top %u -> out rows [%u,%u)\n",
                       t + 1, n, plan[t].iy0, plan[t].iy0 + plan[t].ih, plan[t].pad_top,
                       plan[t].oy0, plan[t].oy0 + plan[t].oh);

            t_mark = now_ms();
            rc = gen_conv2d_int8_rk3576(&q);
            if (rc != 0) { printf("  generator failed: %d\n", rc); goto done; }

            rocket_bo_prep(fd, &r_bo, 1, 0);
            memcpy(r_bo.ptr, ops, q.task_count * sizeof(uint64_t));
            rocket_bo_fini(fd, &r_bo);
            t_emit += now_ms() - t_mark;

            /* Per-task wall check. A multi-task program spends most of its life
             * inside one NPU power session, which is exactly where this part stops
             * computing, and a walled task leaves ITS OWN rows untouched while every
             * other task's land correctly — so the run comes back a few rows wrong
             * and reads like a window-arithmetic bug. Retry the task until its rows
             * move, rather than trusting a gap to be long enough. */
            for (r = 0; r <= (unsigned)env_int("ROCKET_FL_TASK_RETRIES", 8); r++) {
                if ((t || r) && gap > 0) {
                    struct timespec ts = { gap / 1000, (long)(gap % 1000) * 1000000L };
                    nanosleep(&ts, NULL);
                }
                t_mark = now_ms();
                rc = rocket_submit_matmul(fd, &r_bo, q.task_count, in_h, 4, out_h, 1, 2000);
                if (rc != 0) { printf("  submit failed: %d\n", rc); goto done; }
                if (rocket_bo_prep(fd, &o_bo, 0, 2000000000ull) < 0) {
                    printf("  PREP_BO on the output timed out\n"); rc = -1; goto done;
                }
                t_dev += now_ms() - t_mark;
                wrote = task_rows_written((const int8_t *)o_bo.ptr, plan[t].oy0, plan[t].oh);
                rocket_bo_fini(fd, &o_bo);
                if (wrote || n == 1) break;
                walled++;
            }
            if (!wrote && n > 1) {
                printf("  task %u never wrote its rows (cold-start wall)\n", t + 1);
                rc = -1; goto done;
            }
        }
        if (walled)
            printf("  (cold-start wall: %u task resubmit%s)\n",
                   walled, walled == 1 ? "" : "s");
        if (env_int("ROCKET_FL_TIME", 0))
            printf("  time: %u task(s) in %.2f ms  (submit+wait %.2f, emit %.2f, "
                   "%.2f ms/task)\n", n, now_ms() - t_loop0, t_dev, t_emit,
                   (now_ms() - t_loop0) / (double)n);
    }

    rocket_bo_prep(fd, &o_bo, 0, 2000000000ull);
    memcpy(out, o_bo.ptr, (size_t)OBYTES);
    rocket_bo_fini(fd, &o_bo);
    rc = 0;

done:
    if (g_bo.ptr)  rocket_bo_free(fd, &g_bo);
    if (r_bo.ptr)  rocket_bo_free(fd, &r_bo);
    if (o_bo.ptr)  rocket_bo_free(fd, &o_bo);
    if (b_bo.ptr)  rocket_bo_free(fd, &b_bo);
    if (w_bo.ptr)  rocket_bo_free(fd, &w_bo);
    if (in_bo.ptr) rocket_bo_free(fd, &in_bo);
    return rc;
}

/* Plain-layout accessors: features [ic][y][x], weights [oc][ic][kh][kw]. */
#define INP(ic, y, x)      in_plain[(((size_t)(ic) * IH) + (y)) * IW + (x)]
#define WP(oc, ic, kh, kw) w_plain[((((size_t)(oc) * IC + (ic)) * K + (kh)) * K + (kw))]

/* The CPU model's accumulator at one output position. Kept separate so the retry
 * check below can ask what the surface should look like before deciding whether an
 * all-zero surface is a result or the cold-start wall. */
static int64_t model_acc(const int8_t *in_plain, const int8_t *w_plain,
                         const int32_t *bias, int pad_val, int oc, int y, int x)
{
    int64_t acc = bias[oc];
    int ic, kh, kw;
    for (ic = 0; ic < IC; ic++)
        for (kh = 0; kh < K; kh++)
            for (kw = 0; kw < K; kw++) {
                int iy = y * STRIDE - PAD_LEAD + kh;
                int ix = x * STRIDE - PAD_LEAD + kw;
                int tap = (iy >= 0 && iy < IH && ix >= 0 && ix < IW)
                          ? INP(ic, iy, ix) : pad_val;
                acc += (int64_t)tap * WP(oc, ic, kh, kw);
            }
    return acc;
}

int main(int argc, char **argv)
{
    const char *probe = argc > 1 ? argv[1] : "identity";
    int repeat = argc > 2 ? atoi(argv[2]) : 1;
    int8_t *in_plain = NULL, *w_plain = NULL, *in_cube = NULL, *w_cube = NULL;
    int8_t *out[8] = {0};
    int32_t *bias = NULL;
    struct job j;
    unsigned scale, shift_reg, seed = 12345u;
    int fd, rc, ic, oc, y, x, kh, kw, run, fail = 0, gap_ms = 0, pad_val;
    /* The requant divisor. Sweeping it separates a fixed arithmetic defect in the
     * OUT_CVT from one that tracks the programmed scale and shift. */
    const float conv_scale = 1.0f / (float)env_int("ROCKET_FL_OUTDIV", 64);

    IC     = env_int("ROCKET_FL_IC", 32);
    OC     = env_int("ROCKET_FL_OC", 32);
    IW     = env_int("ROCKET_FL_IW", 8);
    IH     = env_int("ROCKET_FL_IH", 8);
    K      = env_int("ROCKET_FL_K", 1);
    STRIDE = env_int("ROCKET_FL_STRIDE", 1);
    SAME   = env_int("ROCKET_FL_SAME", 0);
    IZP    = env_int("ROCKET_FL_IZP", 0x80);
    /* Default to the counts the hardware needs, so a bare ROCKET_FL_IC=16 run
     * exercises the supported path; setting either knob back to the logical count
     * reproduces the defect it corrects, which is the negative control. */
    ICREG  = env_int("ROCKET_FL_ICREG", (int)rocket_rk3576_pad_ic((unsigned)IC));
    OCREG  = env_int("ROCKET_FL_OCREG", (int)rocket_rk3576_pad_oc((unsigned)OC));

    /* Output geometry. SAME follows TFLite: the output covers ceil(in/stride) and
     * the total pad is split with the smaller half leading — which is the
     * asymmetric (l1,r2) the conv2d capture carries at k=5 s=2. */
    if (SAME) {
        OW = (IW + STRIDE - 1) / STRIDE;
        OH = (IH + STRIDE - 1) / STRIDE;
        PAD_LEAD = ((OW - 1) * STRIDE + K - IW) / 2;
        if (PAD_LEAD < 0) PAD_LEAD = 0;
    } else {
        OW = (IW - K) / STRIDE + 1;
        OH = (IH - K) / STRIDE + 1;
        PAD_LEAD = 0;
    }
    if (OW <= 0 || OH <= 0) { printf("  geometry gives a %dx%d output — nothing to do\n", OW, OH); return 1; }
    OBYTES  = ((OCREG + C2 - 1) / C2) * OH * OW * C2;
    pad_val = IZP - 0x80;

    if (repeat < 1) repeat = 1;
    if (repeat > 8) repeat = 8;

    printf("RK3576 first light — probe '%s', int8 conv IC=%d OC=%d %dx%d k%d s%d pad%d "
           "-> %dx%d, pad tap %d, %d run(s)\n",
           probe, IC, OC, IW, IH, K, STRIDE, PAD_LEAD, OW, OH, pad_val, repeat);
    /* The counts the register program actually carries. Printed unconditionally
     * because "which channel count did this run program?" is the first question
     * every channel-granularity result raises. */
    printf("  registers: ic=%d oc=%d (logical ic=%d oc=%d), feature cube %zu B, "
           "weight cube %zu B, output %d B\n",
           ICREG, OCREG, IC, OC, in_cube_bytes(), w_cube_bytes(), OBYTES);

    if (!strcmp(probe, "pad") && (K == 1 || PAD_LEAD == 0)) {
        printf("  the pad probe needs k>1 and padding (set ROCKET_FL_K=3 ROCKET_FL_SAME=1) — SKIP\n");
        return 2;
    }
    if (!strcmp(probe, "pad") && pad_val == 0)
        printf("  NOTE: pad tap is 0 — set ROCKET_FL_IZP=0 for a -128 pad tap\n");

    in_plain = calloc((size_t)IC * IH * IW, 1);
    w_plain  = calloc((size_t)OC * IC * K * K, 1);
    in_cube  = calloc(in_cube_bytes(), 1);
    w_cube   = calloc(w_cube_bytes(), 1);
    bias     = calloc((size_t)(OCREG > OC ? OCREG : OC), sizeof *bias);
    for (run = 0; run < repeat; run++) out[run] = calloc((size_t)OBYTES, 1);
    if (!in_plain || !w_plain || !in_cube || !w_cube || !bias || !out[0]) {
        printf("  out of memory\n"); return 1;
    }

    fd = rocket_open();
    if (fd < 0) { printf("  no /dev/accel/accel0 (%d) — SKIP\n", fd); return 2; }

    requant_params(conv_scale, &scale, &shift_reg);
    printf("  requant: conv_scale=%g -> scale=0x%04x shift_reg=%u offset=0\n",
           (double)conv_scale, scale, shift_reg);

    memset(&j, 0, sizeof j);
    /* ICREG/OCREG tell the REGISTERS a different channel count than the data
     * carries. Both cubes already hold exactly what the padded count wants — the
     * feature cube is calloc'd and sized to the 32-channel group, and
     * weight_conv_int8's index depends on the counts only through ceil(n/32), so an
     * IC=16 cube is byte-identical to an IC=32 cube with the upper channels zero.
     * The only thing this changes is the register program (and, for OCREG, the
     * output and coefficient allocations), which separates a channel-rounding rule
     * in the registers from a cube-packing one. */
    j.p.ic = (unsigned)ICREG;
    j.p.ih = IH; j.p.iw = IW;
    j.p.oc = (unsigned)OCREG; j.p.oh = OH; j.p.ow = OW;
    j.p.kh = K;  j.p.kw = K;
    j.p.stride_y = (uint8_t)STRIDE; j.p.stride_x = (uint8_t)STRIDE;
    j.p.pad_top  = (uint8_t)PAD_LEAD; j.p.pad_left = (uint8_t)PAD_LEAD;
    j.p.int8_out = 1;
    j.p.in_scale = 1.0f; j.p.w_scale = 1.0f; j.p.out_scale = 1.0f / conv_scale;
    /* Symmetric int8: the datapath is uint8-centered, so a symmetric int8 tensor's
     * uint8 zero point is 0x80 and every zero-point term cancels. ROCKET_FL_IZP
     * moves it, which is how the pad probe gets a non-zero border tap. */
    j.p.input_zero_point = IZP;
    j.p.output_zero_point = 0x80;
    j.p.weight_zero_point = 0x80;

    if (!strcmp(probe, "bias")) {
        /* Nothing for the MAC to do; the whole answer comes from the bias. A negative
         * ROCKET_FL_BIASMUL drives the DPU epilogue with a negative accumulator that
         * the MAC never touched, which places any sign-dependent arithmetic either
         * downstream of the bias add (it shows up here) or upstream in the CSC/CMAC
         * (it does not). */
        for (oc = 0; oc < OC; oc++)
            bias[oc] = env_int("ROCKET_FL_BIASMUL", 64) * (oc + 1);
    } else if (!strcmp(probe, "pad")) {
        /* Features stay zero by default. The only operand with any magnitude is
         * then the border pad the CNA injects, so a non-zero result cannot have
         * come through the feature DMA. */
        for (oc = 0; oc < OC && oc < IC; oc++)
            for (kh = 0; kh < K; kh++)
                for (kw = 0; kw < K; kw++)
                    WP(oc, oc, kh, kw) = 1;
    }
    /* ROCKET_FL_FEAT forces the feature tensor either way, independently of the
     * probe's weights. That crossing is what separates "the feature operand never
     * arrives" from "a non-zero feature operand breaks the MAC": with it, a sparse
     * and a dense weight set can each be run against zero and against real
     * features, and only one cell of the 2x2 changes at a time.
     * Default: zero for bias/pad, the standard pattern for identity/random. */
    {
        int feat = env_int("ROCKET_FL_FEAT",
                           (!strcmp(probe, "bias") || !strcmp(probe, "pad")) ? 0 : 1);
        memset(in_plain, 0, (size_t)IC * IH * IW);
        switch (feat) {
        case 0: break;                                    /* all zero            */
        case 2:                                 /* uniform ROCKET_FL_FEATVAL      */
            memset(in_plain, env_int("ROCKET_FL_FEATVAL", 1) & 0xFF,
                   (size_t)IC * IH * IW);
            break;
        case 3:                                           /* one channel of ones */
            for (y = 0; y < IH; y++) for (x = 0; x < IW; x++) INP(0, y, x) = 1;
            break;
        case 4:                                           /* ONE non-zero byte   */
            INP(0, 0, 0) = 1;
            break;
        /* One axis at a time. A uniform fill hides every addressing error — any
         * read that lands inside the buffer returns the same byte — so these three
         * vary along exactly one axis and hold the other two flat. Whichever one
         * fails names the stride that is wrong: channel -> the CBUF surface
         * stride, row -> the line stride, column -> the C2 atom layout. The base
         * of 64 keeps every tap well above the requant LSB so a wrong read shows
         * up as a wrong value rather than a rounding tie. */
        case 5:  /* every channel distinct, across groups as well as within one */
            for (ic = 0; ic < IC; ic++) for (y = 0; y < IH; y++) for (x = 0; x < IW; x++)
                INP(ic, y, x) = (int8_t)(64 + 2 * (ic % 32));
            break;
        case 9:  /* the standard ramp shifted into the large-positive band. Same
                  * variation on all three axes, different magnitude — which is what
                  * separates "the feature path cannot handle varying data" from
                  * "it cannot handle small or signed data". */
            for (ic = 0; ic < IC; ic++) for (y = 0; y < IH; y++) for (x = 0; x < IW; x++)
                INP(ic, y, x) = (int8_t)(env_int("ROCKET_FL_FEATBASE", 64) +
                                         ((ic * 7 + y * 13 + x * 3) % 61));
            break;
        case 10: /* every int8 value exactly once per channel, in spatial order. With
                  * the identity weights and a unity requant this makes the output
                  * plane the datapath's transfer curve, read off directly: at
                  * IW=IH=16 the 256 positions carry -128..127 in order. */
            for (ic = 0; ic < IC; ic++) for (y = 0; y < IH; y++) for (x = 0; x < IW; x++)
                INP(ic, y, x) = (int8_t)(-128 + ((y * IW + x) & 0xFF));
            break;
        case 8:  /* channel GROUP only: flat inside a C2 atom, so this isolates the
                  * CBUF surface (group) stride and nothing else */
            for (ic = 0; ic < IC; ic++) for (y = 0; y < IH; y++) for (x = 0; x < IW; x++)
                INP(ic, y, x) = (int8_t)(64 + 32 * (ic / C2));
            break;
        case 11: /* ONE channel perturbed, every other channel flat. The single-source
                  * form of the channel-varying tensor: with the weights confined to
                  * input channel 0 (ROCKET_FL_WIC=1), every output is a function of
                  * channel 0 alone, so any output that MOVES as ROCKET_FL_FEATCHAN is
                  * swept names the source channel that leaks into it. A leak confined
                  * to one C2 atom or one 64-byte CBUF granule names the staging unit;
                  * one indexed by output channel is a CSC/MAC pairing fault instead. */
            {
                int fc  = env_int("ROCKET_FL_FEATCHAN", -1);
                int fv  = env_int("ROCKET_FL_FEATVAL", 64) & 0xFF;
                int fcv = env_int("ROCKET_FL_FEATCHANVAL", 126) & 0xFF;
                /* ROCKET_FL_FEATCHANPOS narrows the perturbation to ONE spatial
                 * position of that channel, which separates a per-element operand
                 * from a single scalar word read once at a fixed buffer offset: the
                 * first moves one output pixel, the second moves the whole surface. */
                int fp  = env_int("ROCKET_FL_FEATCHANPOS", -1);
                memset(in_plain, fv, (size_t)IC * IH * IW);
                if (fc >= 0 && fc < IC) {
                    if (fp >= 0 && fp < IH * IW)
                        in_plain[(size_t)fc * IH * IW + fp] = (int8_t)fcv;
                    else
                        memset(in_plain + (size_t)fc * IH * IW, fcv, (size_t)IH * IW);
                }
            }
            break;
        case 6:
            for (ic = 0; ic < IC; ic++) for (y = 0; y < IH; y++) for (x = 0; x < IW; x++)
                INP(ic, y, x) = (int8_t)(64 + 4 * (y % 16));
            break;
        case 7:
            for (ic = 0; ic < IC; ic++) for (y = 0; y < IH; y++) for (x = 0; x < IW; x++)
                INP(ic, y, x) = (int8_t)(64 + 4 * (x % 16));
            break;
        default:                                          /* the standard ramp   */
            for (ic = 0; ic < IC; ic++)
                for (y = 0; y < IH; y++)
                    for (x = 0; x < IW; x++)
                        INP(ic, y, x) = (int8_t)(((ic * 7 + y * 13 + x * 3) % 61) - 30);
            break;
        }
    }
    if (strcmp(probe, "bias") && strcmp(probe, "pad")) {
        if (!strcmp(probe, "wcurve")) {
            /* The WEIGHT transfer curve, read off in one run. Output channel oc gets a
             * single weight on input channel 0, stepping across the int8 range; with a
             * uniform feature of 64 and a 1/64 requant the output for channel oc is
             * that weight itself. Whatever the datapath does to a weight value is then
             * the difference between the printed channel column and a straight line.
             * The feature is held positive and constant so nothing else varies. */
            /* The step defaults to spanning the whole int8 range across the output
             * channels; ROCKET_FL_WBASE/_WSTEP narrow it to a window, which is what
             * resolves a curve whose interesting structure sits near zero. */
            int step = env_int("ROCKET_FL_WSTEP", 256 / (OC > 0 ? OC : 1));
            int base = env_int("ROCKET_FL_WBASE", -128);
            int wsel = env_int("ROCKET_FL_WICSEL", 0);   /* which input channel carries it */
            /* ROCKET_FL_WDENSE puts the same weight on EVERY input channel instead of
             * one, so the curve can be read with a dense weight cube. Sparsity and
             * operand value are then separable: the sparse form isolates a single
             * product, the dense form the full channel accumulation. */
            if (env_int("ROCKET_FL_WDENSE", 0)) {
                for (oc = 0; oc < OC; oc++)
                    for (ic = 0; ic < IC; ic++)
                        WP(oc, ic, 0, 0) = (int8_t)(base + oc * step);
            } else {
                for (oc = 0; oc < OC; oc++)
                    WP(oc, wsel % IC, 0, 0) = (int8_t)(base + oc * step);
            }
        } else if (!strcmp(probe, "identity")) {
            /* Centre-tap identity: at k=1 this is the plain identity matrix, and at
             * k>1 the output still equals the input wherever the window is in bounds. */
            /* ROCKET_FL_IDW moves the identity tap off 64. Pair it with a matching
             * ROCKET_FL_OUTDIV to keep the requant at unity, so the only thing that
             * changes between runs is the weight VALUE the multiplier is handed.
             *
             * ROCKET_FL_IDTAPS replaces the diagonal with a unit tap on the FIRST N
             * input channels for every output channel, so the output is the sum of N
             * feature channels. With a channel-varying feature tensor, sweeping N from
             * 1 upwards is the minimal reproducer for an accumulation that only holds
             * while the summed channels carry equal values: N=1 is a single product,
             * N=2 the shortest sum that can disagree. */
            int idw = env_int("ROCKET_FL_IDW", 64);
            int taps = env_int("ROCKET_FL_IDTAPS", 0);
            if (taps > 0) {
                for (oc = 0; oc < OC; oc++)
                    for (ic = 0; ic < taps && ic < IC; ic++)
                        WP(oc, ic, K / 2, K / 2) = (int8_t)idw;
            } else {
                for (oc = 0; oc < OC && oc < IC; oc++)
                    WP(oc, oc, K / 2, K / 2) = (int8_t)idw;
            }
        } else {
            /* ROCKET_FL_WMAX bounds the random weight magnitude. The default 20 makes
             * a 32-channel k3 accumulation saturate the int8 output almost everywhere,
             * which hides the arithmetic under the clamp; a small bound keeps every
             * output in range so each one carries information. */
            int wmax = env_int("ROCKET_FL_WMAX", 20);
            /* ROCKET_FL_WIC limits the non-zero weights to the first N input
             * channels, holding the geometry fixed. Sweeping it varies only the
             * number of terms the accumulator sums, which is what separates a
             * defect in one product from one in the accumulation. */
            int wic = env_int("ROCKET_FL_WIC", IC);
            for (oc = 0; oc < OC; oc++)
                for (ic = 0; ic < (wic < IC ? wic : IC); ic++)
                    for (kh = 0; kh < K; kh++)
                        for (kw = 0; kw < K; kw++) {
                            seed = seed * 1103515245u + 12345u;
                            WP(oc, ic, kh, kw) =
                                (int8_t)((int)((seed >> 16) % (2 * wmax + 1)) - wmax);
                        }
        }
    }

    for (ic = 0; ic < IC; ic++)
        for (y = 0; y < IH; y++)
            for (x = 0; x < IW; x++)
                in_cube[feature_data(IC, IH, IW, C2, ic + 1, y + 1, x + 1)] = INP(ic, y, x);
    for (oc = 0; oc < OC; oc++)
        for (ic = 0; ic < IC; ic++)
            for (kh = 0; kh < K; kh++)
                for (kw = 0; kw < K; kw++)
                    w_cube[weight_conv_int8(OC, IC, K, K, oc + 1, ic + 1, kh + 1, kw + 1)] =
                        WP(oc, ic, kh, kw);

    j.in_cube = in_cube; j.w_cube = w_cube; j.bias = bias;

    /* Gap between runs. The NPU runtime-suspends 50 ms after the last job, so a gap
     * longer than that powers the domains down while this fd stays open — which is
     * what separates "the arm is per NPU power session" from "the arm is per fd or
     * per context". */
    gap_ms = env_int("ROCKET_FL_GAP_MS", 0);

    for (run = 0; run < repeat; run++) {
        char tag[32];
        if (run && gap_ms > 0) {
            struct timespec ts = { gap_ms / 1000, (long)(gap_ms % 1000) * 1000000L };
            nanosleep(&ts, NULL);
        }
        rc = submit_once(fd, &j, out[run]);
        if (rc != 0) { rocket_close(fd); return 1; }
        /* The cold-start wall, not a register result: this part arms its weight loader
         * once per NPU power session, so a job that lands inside a session another job
         * already consumed does not MAC. It has TWO signatures — the output BO comes
         * back wholly untouched, or the DPU writes a full surface carrying only the
         * bias, so the MAC contribution is missing and the surface is all zero. Both
         * are indistinguishable from a real negative by inspection, so retry past the
         * runtime-PM autosuspend. The all-zero case is only treated as the wall when
         * the CPU model says the surface should have been non-zero; a genuinely zero
         * expected surface is left alone. */
        {
            int tries = env_int("ROCKET_FL_RETRIES", 6);
            int wait_ms = env_int("ROCKET_FL_RETRY_MS", 1200), t;
            int expect_nonzero = 0;
            for (oc = 0; oc < OC && !expect_nonzero; oc++)
                for (y = 0; y < OH && !expect_nonzero; y++)
                    for (x = 0; x < OW && !expect_nonzero; x++)
                        if (requant_apply(model_acc(in_plain, w_plain, bias, pad_val,
                                                    oc, y, x), scale, shift_reg, 0))
                            expect_nonzero = 1;
            for (t = 0; t < tries; t++) {
                struct surface_stat s = classify(out[run], OBYTES);
                if (s.sentinel != OBYTES && !(expect_nonzero && s.other == 0)) break;
                struct timespec ts = { wait_ms / 1000, (long)(wait_ms % 1000) * 1000000L };
                nanosleep(&ts, NULL);
                rc = submit_once(fd, &j, out[run]);
                if (rc != 0) { rocket_close(fd); return 1; }
            }
            if (t) printf("  (cold-start wall: %d retr%s before the DPU wrote)\n",
                          t, t == 1 ? "y" : "ies");
        }
        snprintf(tag, sizeof tag, "run %d", run + 1);
        print_stat(tag, classify(out[run], OBYTES), OBYTES);
    }

    /* Per-channel value at one spatial position. Whether a channel carries data at
     * all, and which channels, is what distinguishes a wrong element size or
     * channel stride on an operand DMA from a wrong value. */
    {
        printf("  out[c][0][0] by channel:");
        for (oc = 0; oc < OC; oc++) {
            if (oc % 16 == 0) printf("\n    c%2d:", oc);
            printf("%5d", out[0][feature_data(OC, OH, OW, C2, oc + 1, 1, 1)]);
        }
        printf("\n");
    }

    /* ---- correctness of run 1 ---- */
    {
        int mism = 0, first = 1, maxdiff = 0;
        int n_signed = 0, n_unsigned = 0, n_relu = 0, shown = 0, in_range = 0;
        /* Which OUTPUT channels are wrong, as one line. A sweep that perturbs one
         * input channel per run (ROCKET_FL_FEAT=11) reads its whole answer off this:
         * the set of output channels the perturbation reached. */
        int *mism_c = calloc((size_t)OC, sizeof *mism_c);
        for (oc = 0; oc < OC; oc++)
            for (y = 0; y < OH; y++)
                for (x = 0; x < OW; x++) {
                    int64_t acc = bias[oc], raw;
                    int got, want, d;
                    for (ic = 0; ic < IC; ic++)
                        for (kh = 0; kh < K; kh++)
                            for (kw = 0; kw < K; kw++) {
                                int iy = y * STRIDE - PAD_LEAD + kh;
                                int ix = x * STRIDE - PAD_LEAD + kw;
                                int tap = (iy >= 0 && iy < IH && ix >= 0 && ix < IW)
                                          ? INP(ic, iy, ix) : pad_val;
                                acc += (int64_t)tap * WP(oc, ic, kh, kw);
                            }
                    raw  = requant_raw(acc, scale, shift_reg, 0);
                    want = requant_apply(acc, scale, shift_reg, 0);
                    got  = out[0][feature_data(OC, OH, OW, C2, oc + 1, y + 1, x + 1)];
                    /* Which saturation domain reproduces this pixel. Counted over
                     * every pixel, not just the mismatches, because the two rules
                     * agree on everything already inside [0,127] and only their
                     * disagreement carries information. */
                    if (raw >= 0 && raw <= 127) in_range++;
                    if (got == sat_signed(raw))          n_signed++;
                    if (got == sat_unsigned(raw))        n_unsigned++;
                    if (got == sat_signed(raw < 0 ? 0 : raw)) n_relu++;
                    d = got > want ? got - want : want - got;
                    if (d > maxdiff) maxdiff = d;
                    if (got != want) {
                        mism++;
                        if (mism_c) mism_c[oc]++;
                        if (first) {
                            printf("  first mismatch at c=%d y=%d x=%d: want %d got %d\n",
                                   oc, y, x, want, got);
                            first = 0;
                        }
                        if (shown < 12 && got != sat_signed(raw < 0 ? 0 : raw)) {
                            printf("    mism c=%2d y=%2d x=%2d: raw %6lld  signed %4d  "
                                   "unsigned %4d  relu %4d  got %4d\n",
                                   oc, y, x, (long long)raw, sat_signed(raw),
                                   sat_unsigned(raw), sat_signed(raw < 0 ? 0 : raw), got);
                            shown++;
                        }
                    }
                }
        printf("  vs CPU: %d / %d exact, max |diff| = %d\n",
               OC * OH * OW - mism, OC * OH * OW, maxdiff);
        if (mism_c) {
            int nc = 0;
            printf("  wrong output channels:");
            for (oc = 0; oc < OC; oc++)
                if (mism_c[oc]) { printf(" %d", oc); nc++; }
            printf("%s  (%d of %d)\n", nc ? "" : " none", nc, OC);
            free(mism_c);
        }
        printf("  saturation rule: signed %d, unsigned %d, relu %d, of %d "
               "(%d already inside [0,127])\n",
               n_signed, n_unsigned, n_relu, OC * OH * OW, in_range);
        /* One row per output channel at a single spatial position, carrying the
         * accumulator that produced it. When the feature tensor is constant over
         * (y,x) — ROCKET_FL_FEAT=5 or 2 — every position in a channel holds the same
         * value, so these 32 rows are the whole result and a wrong channel sum is
         * readable against the accumulator that should have produced it. */
        if (env_int("ROCKET_FL_CHANMAP", 0)) {
            printf("  per-output-channel at (0,0): acc / raw / want / got\n");
            for (oc = 0; oc < OC; oc++) {
                int64_t acc = bias[oc];
                int got;
                for (ic = 0; ic < IC; ic++)
                    for (kh = 0; kh < K; kh++)
                        for (kw = 0; kw < K; kw++) {
                            int iy = kh - PAD_LEAD, ix = kw - PAD_LEAD;
                            int tap = (iy >= 0 && iy < IH && ix >= 0 && ix < IW)
                                      ? INP(ic, iy, ix) : pad_val;
                            acc += (int64_t)tap * WP(oc, ic, kh, kw);
                        }
                got = out[0][feature_data(OC, OH, OW, C2, oc + 1, 1, 1)];
                printf("    c%2d: acc %8lld  raw %6lld  want %4d  got %4d%s\n", oc,
                       (long long)acc,
                       (long long)requant_raw(acc, scale, shift_reg, 0),
                       requant_apply(acc, scale, shift_reg, 0), got,
                       got == requant_apply(acc, scale, shift_reg, 0) ? "" : "   <-");
            }
        }
        /* The transfer curve, printed as runs of (raw -> got). With ROCKET_FL_FEAT=10
         * and the identity weights every raw value appears exactly once, so this is
         * the datapath's input-to-output map with nothing else in the way. Runs are
         * collapsed because a correct datapath prints one run per saturation region
         * and a broken one prints its breakpoints. */
        if (env_int("ROCKET_FL_CURVE", 0)) {
            int64_t prev_raw = 0; int prev_got = 0, run = 0, n = 0;
            printf("  transfer curve, channel 0 (raw -> got), runs of got-raw:\n   ");
            for (y = 0; y < OH; y++)
                for (x = 0; x < OW; x++) {
                    int64_t acc = bias[0], raw;
                    int got, delta;
                    for (ic = 0; ic < IC; ic++)
                        for (kh = 0; kh < K; kh++)
                            for (kw = 0; kw < K; kw++) {
                                int iy = y * STRIDE - PAD_LEAD + kh;
                                int ix = x * STRIDE - PAD_LEAD + kw;
                                int tap = (iy >= 0 && iy < IH && ix >= 0 && ix < IW)
                                          ? INP(ic, iy, ix) : pad_val;
                                acc += (int64_t)tap * WP(0, ic, kh, kw);
                            }
                    raw = requant_raw(acc, scale, shift_reg, 0);
                    got = out[0][feature_data(OC, OH, OW, C2, 1, y + 1, x + 1)];
                    delta = (int)(got - raw);
                    if (run && delta == prev_got) { run++; prev_raw = raw; continue; }
                    if (run) {
                        printf(" [%lld..%lld]%+d", (long long)(prev_raw - run + 1),
                               (long long)prev_raw, prev_got);
                        if (++n % 4 == 0) printf("\n   ");
                    }
                    prev_got = delta; prev_raw = raw; run = 1;
                }
            if (run) printf(" [%lld..%lld]%+d", (long long)(prev_raw - run + 1),
                            (long long)prev_raw, prev_got);
            printf("\n  (a correct signed datapath is one +0 run over [-128..127])\n");
        }
        /* What the writer does with an out-of-range value, resolved per channel and
         * per direction. A single global clamp and a per-channel one look identical
         * in a mismatch list that happens to be sorted by channel, so bracket the raw
         * range that maps to each observed byte, channel by channel. */
        {
            int c;
            printf("  out-of-range map (raw -> got), per channel:\n");
            for (c = 0; c < OC && c < 6; c++) {
                int nlo = 0, nhi = 0, lo_got = 0, hi_got = 0, lo_mixed = 0, hi_mixed = 0;
                int64_t lo_min = 0, lo_max = 0, hi_min = 0, hi_max = 0;
                for (y = 0; y < OH; y++)
                    for (x = 0; x < OW; x++) {
                        int64_t acc = bias[c], raw;
                        int got;
                        for (ic = 0; ic < IC; ic++)
                            for (kh = 0; kh < K; kh++)
                                for (kw = 0; kw < K; kw++) {
                                    int iy = y * STRIDE - PAD_LEAD + kh;
                                    int ix = x * STRIDE - PAD_LEAD + kw;
                                    int tap = (iy >= 0 && iy < IH && ix >= 0 && ix < IW)
                                              ? INP(ic, iy, ix) : pad_val;
                                    acc += (int64_t)tap * WP(c, ic, kh, kw);
                                }
                        raw = requant_raw(acc, scale, shift_reg, 0);
                        got = out[0][feature_data(OC, OH, OW, C2, c + 1, y + 1, x + 1)];
                        if (raw < 0) {
                            if (!nlo) { lo_got = got; lo_min = lo_max = raw; }
                            else { if (raw < lo_min) lo_min = raw; if (raw > lo_max) lo_max = raw;
                                   if (got != lo_got) lo_mixed = 1; }
                            nlo++;
                        } else if (raw > 127) {
                            if (!nhi) { hi_got = got; hi_min = hi_max = raw; }
                            else { if (raw < hi_min) hi_min = raw; if (raw > hi_max) hi_max = raw;
                                   if (got != hi_got) hi_mixed = 1; }
                            nhi++;
                        }
                    }
                printf("    c%2d: %5d raw<0 in [%lld,%lld] -> %s%d%s ; "
                       "%5d raw>127 in [%lld,%lld] -> %s%d%s\n",
                       c, nlo, (long long)lo_min, (long long)lo_max,
                       lo_mixed ? "MIXED, first " : "", lo_got, lo_mixed ? "" : " (uniform)",
                       nhi, (long long)hi_min, (long long)hi_max,
                       hi_mixed ? "MIXED, first " : "", hi_got, hi_mixed ? "" : " (uniform)");
            }
        }
        if (mism) {
            fail++;
            printf("  channel 0 row 0 got : ");
            for (x = 0; x < OW && x < 16; x++)
                printf("%5d", out[0][feature_data(OC, OH, OW, C2, 1, 1, x + 1)]);
            printf("\n  channel 0 row 0 want: ");
            for (x = 0; x < OW && x < 16; x++) {
                int64_t acc = bias[0];
                for (ic = 0; ic < IC; ic++)
                    for (kh = 0; kh < K; kh++)
                        for (kw = 0; kw < K; kw++) {
                            int iy = 0 * STRIDE - PAD_LEAD + kh;
                            int ix = x * STRIDE - PAD_LEAD + kw;
                            int tap = (iy >= 0 && iy < IH && ix >= 0 && ix < IW)
                                      ? INP(ic, iy, ix) : pad_val;
                            acc += (int64_t)tap * WP(0, ic, kh, kw);
                        }
                printf("%5d", requant_apply(acc, scale, shift_reg, 0));
            }
            printf("\n");
        }
    }

    /* ---- the cold-start discriminator ---- */
    if (repeat > 1) {
        int differs = 0, i;
        for (run = 1; run < repeat; run++) {
            differs = 0;
            for (i = 0; i < OBYTES; i++) if (out[run][i] != out[0][i]) differs++;
            printf("  run %d vs run 1: %d bytes differ\n", run + 1, differs);
            if (differs) fail++;
        }
        if (!differs)
            printf("  every run in this power session matched: back-to-back jobs compute,\n"
                   "  so the reported per-power-session weight-loader arm does not hold here.\n");
    }

    rocket_close(fd);
    printf("RK3576 first light (%s): %s\n", probe, fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
