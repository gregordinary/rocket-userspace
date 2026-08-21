// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_conv_gate.c — the RK3576 int8 conv correctness envelope, as one gate.
 *
 * Runs on an RK3576 board. Where rk3576_first_light is one shape per invocation
 * with a probe per question, this is a shape TABLE swept in a single process: each
 * entry builds its own operands, submits, and is compared against a CPU model
 * bit-exactly over the whole surface. It exists for four reasons.
 *
 * It runs GAP-FREE. Every earlier envelope result on this part was gathered while
 * the cold-start wall was live, with a multi-second gap between shapes and retry
 * logic compensating — correct by construction, but it means each of those results
 * was taken through the compensation rather than past it. With a `rocket` carrying
 * the per-SoC PC_TASK_CON width the wall is gone, so the whole envelope fits in one
 * NPU power session with no gap at all. The retry path below is kept, and reports
 * every retry it takes: a run that prints no retries is a run the wall did not
 * touch. On an unpatched module the count is the measurement.
 *
 * It covers the ROW WINDOW as a matrix rather than as a one-off. A plane over the
 * CBUF allowance computes WRONG with the DPU still writing a full surface, so the
 * split is load-bearing correctness, not a performance path, and the register it
 * pins (DPU 0x40B8, the channel-group jump) is the one whose form was fitted at a
 * single output-channel count. The `window` and `surface` groups drive it across
 * kernel size, stride, channel count and window count.
 *
 * It asserts the BOUNDARIES, not only the interior. The emitter refuses a shape past
 * the CBUF allowance or past the weight-slice limit, because both compute wrong with a
 * full surface written and nothing to fault on. A refusal is therefore a result the
 * gate checks in both directions: the table says which shapes must be refused, and a
 * shape that computes when the measured boundary says it should not is as much a
 * failure as one that comes back wrong.
 *
 * It carries DEPTHWISE, whose registers came from manufactured vendor captures and
 * whose two BUFFER layouts did not: the coefficient group and the int8 weight cube
 * are each different from the direct path's, and a capture carries register programs
 * rather than buffers, so both had to be read off the part. The `dw` group is what
 * says whether it agrees; the probe modes below are what read the layouts, and their
 * `direct` controls are the same reading on the path that is known bit-exact.
 *
 * The dw group spans the axes those layouts are indexed on rather than a few round
 * numbers: channel counts that are not multiples of 32 (where the weight granule, the
 * CBUF granule and 0x4050's group field stop agreeing), and planes whose ow*oh is not
 * a multiple of four (where the depthwise output surface stride is padded and the
 * direct one is not). Both were invisible at every earlier shape.
 *
 * Usage:
 *   rk3576_conv_gate [group ...]   groups: envelope window surface weight dw
 *                                  (default: envelope window; "all" runs every group)
 *   rk3576_conv_gate -l            list the table without running it
 *   rk3576_conv_gate dwmap         read the depthwise weight-cube layout off the part
 *   rk3576_conv_gate dwbias        which coefficient slot each channel read
 *   rk3576_conv_gate dwcoeff       the inverse: which output byte each slot reaches
 *   rk3576_conv_gate dwout         dump the raw output cube, undescattered
 *
 * Env knobs (bring-up; the defaults are the gate):
 *   ROCKET_G_FILTER=<substr>   run only shapes whose name contains this
 *   ROCKET_G_GAP_MS=<n>        gap between shapes (default 0 — the point of the gate)
 *   ROCKET_G_TASK_GAP_MS=<n>   gap between the tasks of one windowed shape (default 0)
 *   ROCKET_G_RETRIES=<n>       wall retries per task (default 4; 0 disables the
 *                              compensation entirely, which is how a wall is measured)
 *   ROCKET_G_RETRY_MS=<n>      wait before a retry (default 1200, past the 50 ms
 *                              runtime-PM autosuspend that re-arms the part)
 *   ROCKET_G_CREG=<n>          override the channel count PROGRAMMED for a dw shape
 *   ROCKET_G_VERBOSE=1         per-task window lines and the first mismatches
 *   ROCKET_G_DWMAP_C / _K / _IW / _SPAN   dwmap geometry and how much cube to sweep
 *   ROCKET_G_DWOUT_OUTSCALE=<n>  dwout: divide the requant so a saturated lane names
 *                                its magnitude instead of reading as the clip
 *   ROCKET_G_DWOUT_BASE / _STEP  dwout: the bias ramp that names the channels
 *   ROCKET_G_COEFF_BASE / _PROBE dwcoeff: the baseline bias and the bump
 *   ROCKET_G_COEFF_OUTSCALE=<n>  dwcoeff: same requant divisor, for probing a lane
 *                                that sits at the clip in the baseline
 *   ROCKET_G_COEFF_LO / _HI      dwcoeff: restrict the swept slot range
 *   ROCKET_G_COEFF_GAP_MS=<n>    dwcoeff: pace the sweep
 *   ROCKET_RK3576_SET=<r>=<v>,…  patch emitted registers (driver-side; hex is fine)
 *
 * Exit: 0 all pass, 1 a shape failed, 2 no NPU (skip).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_hw_profile.h"

#define C2       16
#define SENTINEL 0xAA

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);
int weight_conv_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

/* ============================================================================
 * SECTION — the shape table
 *
 * `max_rows` forces the per-task row cap below what the CBUF allows, which is how
 * a shape that fits one task is made to exercise the split against the same shape's
 * single-task result. 0 means "let the CBUF planner decide", which for most of these
 * is one task.
 * ==========================================================================*/
#include "rk3576_conv_shapes.h"

/* ============================================================================
 * SECTION — helpers shared with rk3576_first_light
 * ==========================================================================*/
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

static void sleep_ms(int ms)
{
    struct timespec ts;
    if (ms <= 0) return;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

/* The requant the emitter programs: out = sat8( (acc*scale + half) >> shift ). The
 * emitter derives scale and the pre-decrement shift from the fp32 conv scale exactly
 * as the vendor (QNNPACK) does and writes shift-1, so the model shifts by the same
 * register value. The DPU rounds to nearest, which is the `half` term. */
/* The output cube index. NOT feature_data(): that assumes surfaces sit exactly
 * ow*oh apart, and the depthwise writer advances by the plane rounded up to four. */
static size_t out_index(unsigned surf_elems, unsigned ow, unsigned c,
                        unsigned y, unsigned x)
{
    return (size_t)(c / C2) * surf_elems * C2 + (size_t)C2 * (y * ow + x) + (c % C2);
}

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

static int requant_apply(int64_t acc, unsigned scale, unsigned shift_reg)
{
    int64_t half = shift_reg ? ((int64_t)1 << (shift_reg - 1)) : 0;
    int64_t v = (acc * (int64_t)scale + half) >> shift_reg;
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int)v;
}

/* ============================================================================
 * SECTION — one shape
 * ==========================================================================*/
struct run_stat {
    int      tasks;
    int      retries;     /* cold-start-wall resubmits; 0 on a patched driver */
    int      exact, total;
    int      maxdiff;
    int      untouched;   /* sentinel bytes left; only meaningful when it is ALL of them,
                           * since a legitimate output byte can BE the sentinel value */
    int      obytes;      /* output cube size, to read `untouched` against */
    /* ROCKET_G_WZP only: how many outputs each candidate sign convention explains. */
    int      asym_minus, asym_plus, asym_ignored;
    double   ms;
};

/* Did this task's own output rows come back written? The output cube is NC1HWC2, so
 * a row run is one span per channel group at that group's own surface offset —
 * scanning only the first span would call a task written on channel group 0 alone. */
static int rows_written(const int8_t *o, unsigned oc_reg, unsigned ow,
                        unsigned surf_elems, unsigned oy0, unsigned oh_task)
{
    unsigned groups = (oc_reg + C2 - 1) / C2, g, i;
    size_t surf = (size_t)surf_elems * C2;
    for (g = 0; g < groups; g++) {
        const int8_t *p = o + g * surf + (size_t)oy0 * ow * C2;
        for (i = 0; i < oh_task * ow * C2; i++)
            if ((uint8_t)p[i] != SENTINEL) return 1;
    }
    return 0;
}

static int run_shape(int fd, const shape_t *s, struct run_stat *st)
{
    unsigned ic = s->ic, oc = s->oc, iw = s->iw, ih = s->ih, k = s->k, stride = s->stride;
    unsigned icreg, ocreg, icpad, ow, oh, terms, divisor;
    int pad_lead;
    int8_t *in_plain = NULL, *w_plain = NULL, *in_cube = NULL, *w_cube = NULL, *out = NULL;
    int32_t *bias = NULL;
    size_t in_bytes, w_bytes, obytes, coeff_bytes;
    unsigned surf_elems;
    rocket_bo in_bo = {0}, w_bo = {0}, b_bo = {0}, o_bo = {0}, r_bo = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p = {0};
    unsigned scale, shift_reg, seed;
    int verbose = env_int("ROCKET_G_VERBOSE", 0);
    int wzp = env_int("ROCKET_G_WZP", 0);
    int rc = -1, shown = 0;
    unsigned c, y, x, kh, kw, i;

    memset(st, 0, sizeof *st);

    /* Output geometry. SAME follows TFLite: the output covers ceil(in/stride) and the
     * total pad splits with the smaller half leading. */
    if (s->same) {
        ow = (iw + stride - 1) / stride;
        oh = (ih + stride - 1) / stride;
        pad_lead = (int)(((ow - 1) * stride + k - iw) / 2);
        if (pad_lead < 0) pad_lead = 0;
    } else {
        if (iw < k || ih < k) return 2;
        ow = (iw - k) / stride + 1;
        oh = (ih - k) / stride + 1;
        pad_lead = 0;
    }
    if (!ow || !oh) return 2;

    /* Channel counts as told to the REGISTERS. The direct path needs both rounded to
     * the 32-channel MAC group. The depthwise path takes the RAW count — its own two
     * granules are the emitter's business (the weight cube rounds to 16, the CBUF
     * allocation sometimes one 16-group further), and rounding here to 32 would hide
     * every count where the two differ. ROCKET_G_CREG still sweeps it. */
    if (s->dw) {
        icreg = ocreg = (unsigned)env_int("ROCKET_G_CREG", (int)ic);
    } else {
        icreg = rocket_rk3576_pad_ic(ic);
        ocreg = rocket_rk3576_pad_oc(oc);
    }
    icpad = (icreg + 31) / 32 * 32;

    /* Keep the accumulator inside the int8 output range for most pixels: a surface
     * clamped everywhere hides the arithmetic under the saturation. A dense random
     * sum grows as sqrt(terms), so scale the requant divisor with it. */
    terms   = s->dw ? (k * k) : (ic * k * k);
    divisor = 1;
    while ((double)divisor < 2.0 * sqrt((double)terms)) divisor *= 2;
    requant_params(1.0f / (float)divisor, &scale, &shift_reg);

    in_bytes    = (size_t)((icpad + C2 - 1) / C2) * ih * iw * C2;
    w_bytes     = s->dw ? rocket_rk3576_weight_dw_bytes(icreg, k, k)
                        : (size_t)((ocreg + 31) / 32) * ((icreg + 31) / 32) * 32 * 32 * k * k;
    surf_elems  = rocket_rk3576_out_surf_elems(ow, oh, s->dw);
    obytes      = (size_t)((ocreg + C2 - 1) / C2) * surf_elems * C2;
    coeff_bytes = s->dw ? rocket_rk3576_coeff_bytes_dw(ocreg)
                        : rocket_rk3576_coeff_bytes(ocreg);

    in_plain = calloc((size_t)ic * ih * iw, 1);
    w_plain  = calloc(s->dw ? (size_t)ic * k * k : (size_t)oc * ic * k * k, 1);
    in_cube  = calloc(in_bytes, 1);
    w_cube   = calloc(w_bytes, 1);
    out      = calloc(obytes, 1);
    bias     = calloc(ocreg > oc ? ocreg : oc, sizeof *bias);
    if (!in_plain || !w_plain || !in_cube || !w_cube || !out || !bias) goto done;

#define INP(c_, y_, x_)     in_plain[(((size_t)(c_) * ih) + (y_)) * iw + (x_)]
#define WD(oc_, ic_, h_, w_) w_plain[((((size_t)(oc_) * ic + (ic_)) * k + (h_)) * k + (w_))]
#define WW(c_, h_, w_)      w_plain[(((size_t)(c_) * k + (h_)) * k + (w_))]

    /* Deterministic operands, varying on every axis: a feature that is flat along an
     * axis proves nothing about that axis's stride. */
    seed = 0x9E3779B9u ^ (unsigned)(ic * 31 + oc * 17 + iw * 7 + ih * 3 + k);
    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                INP(c, y, x) = (int8_t)(((int)((c * 7 + y * 13 + x * 3) % 61)) - 30);
    if (s->dw) {
        for (c = 0; c < ic; c++)
            for (kh = 0; kh < k; kh++)
                for (kw = 0; kw < k; kw++) {
                    seed = seed * 1103515245u + 12345u;
                    WW(c, kh, kw) = (int8_t)((int)((seed >> 16) % 17u) - 8);
                }
    } else {
        for (c = 0; c < oc; c++)
            for (i = 0; i < ic; i++)
                for (kh = 0; kh < k; kh++)
                    for (kw = 0; kw < k; kw++) {
                        seed = seed * 1103515245u + 12345u;
                        WD(c, i, kh, kw) = (int8_t)((int)((seed >> 16) % 17u) - 8);
                    }
    }
    /* A per-channel bias, so the BS stage carries a value the MAC cannot produce. */
    for (c = 0; c < oc; c++)
        bias[c] = (int32_t)((int)c - (int)oc / 2) * 8;

    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                in_cube[feature_data((int)ic, (int)ih, (int)iw, C2,
                                     (int)c + 1, (int)y + 1, (int)x + 1)] = INP(c, y, x);
    if (s->dw) {
        /* The RK3576's own cube: channels grouped by 32, tap-major inside a group,
         * two bytes per weight — and at int8 the byte a channel owns inside a tap
         * block is 4*(c/2) + (c%2), not the low half of the float slot. NOT the
         * RK3588's 64-channel single-byte cube either. */
        int8_t *w8 = (int8_t *)w_cube;
        for (c = 0; c < ic; c++)
            for (kh = 0; kh < k; kh++)
                for (kw = 0; kw < k; kw++)
                    w8[rocket_rk3576_weight_dw_int8(icreg, k, k, c, kh, kw)] =
                        WW(c, kh, kw);
    } else {
        for (c = 0; c < oc; c++)
            for (i = 0; i < ic; i++)
                for (kh = 0; kh < k; kh++)
                    for (kw = 0; kw < k; kw++)
                        w_cube[weight_conv_int8((int)oc, (int)ic, (int)k, (int)k,
                                                (int)c + 1, (int)i + 1,
                                                (int)kh + 1, (int)kw + 1)] =
                            WD(c, i, kh, kw);
    }

    if (rocket_bo_alloc(fd, in_bytes, &in_bo)  < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes,  &w_bo)   < 0) goto done;
    if (rocket_bo_alloc(fd, coeff_bytes, &b_bo) < 0) goto done;
    if (rocket_bo_alloc(fd, obytes, &o_bo)     < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &r_bo) < 0) goto done;

    rocket_bo_prep(fd, &in_bo, 1, 0); memcpy(in_bo.ptr, in_cube, in_bytes); rocket_bo_fini(fd, &in_bo);
    rocket_bo_prep(fd, &w_bo,  1, 0); memcpy(w_bo.ptr,  w_cube,  w_bytes);  rocket_bo_fini(fd, &w_bo);
    /* The coefficient buffer is NOT a flat int32 bias array on this part, and an
     * all-zero one makes the DPU write a full but empty surface whatever the MAC did
     * — the C term gates the whole BS stage. */
    rocket_bo_prep(fd, &b_bo, 1, 0);
    /* ROCKET_G_WZP drives the coefficient group's B field, whose SIGN CONVENTION this
     * settled: THE DPU ADDS IT, `acc + B*sum(x)`. Every vendor capture carries B = 0, so
     * that the field exists at all was read off its position and width rather than off a
     * program that uses it. The compare below still scores both signs and an inert B,
     * because a wrong sign gives a plausible surface with a bias-shaped error and
     * nothing that faults — so this stays a regression on the convention rather than a
     * one-off probe. Direct path only: the depthwise group has no B field. */
    if (wzp && !s->dw) {
        int16_t *zp = calloc(ocreg, sizeof *zp);
        if (!zp) goto done;
        for (c = 0; c < ocreg; c++) zp[c] = (int16_t)wzp;
        rocket_rk3576_pack_coeff_asym(b_bo.ptr, coeff_bytes, bias, ocreg, zp, 1);
        free(zp);
    }
    else if (s->dw) rocket_rk3576_pack_coeff_dw(b_bo.ptr, coeff_bytes, bias, ocreg);
    else       rocket_rk3576_pack_coeff(b_bo.ptr, coeff_bytes, bias, ocreg);
    rocket_bo_fini(fd, &b_bo);
    rocket_bo_prep(fd, &o_bo, 1, 0); memset(o_bo.ptr, SENTINEL, obytes); rocket_bo_fini(fd, &o_bo);

    p.ic = (uint16_t)icreg; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.oc = (uint16_t)ocreg; p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
    p.kh = (uint16_t)k;     p.kw = (uint16_t)k;
    p.stride_y = (uint8_t)stride; p.stride_x = (uint8_t)stride;
    p.pad_top  = (uint8_t)pad_lead; p.pad_left = (uint8_t)pad_lead;
    p.int8_out = 1;
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = (float)divisor;
    p.input_zero_point = 0x80;   /* symmetric int8: every zero-point term cancels */
    p.output_zero_point = 0x80;
    p.weight_zero_point = 0x80;
    p.tasks       = ops;
    p.input_dma   = in_bo.dma_address;
    p.weights_dma = w_bo.dma_address;
    p.bias_dma    = b_bo.dma_address;
    p.output_dma  = o_bo.dma_address;

    in_h[0] = in_bo.handle; in_h[1] = w_bo.handle;
    in_h[2] = b_bo.handle;  in_h[3] = r_bo.handle;
    out_h[0] = o_bo.handle;

    /* ---- the task sequence ---- */
    {
        rocket_rk3576_row_task plan[1024];
        unsigned n = 1, t, r;
        int gap = env_int("ROCKET_G_TASK_GAP_MS", 0);
        int retries = env_int("ROCKET_G_RETRIES", 4);
        int retry_ms = env_int("ROCKET_G_RETRY_MS", 1200);
        double t0 = now_ms();

        if (s->max_rows) {
            char buf[16];
            snprintf(buf, sizeof buf, "%u", s->max_rows);
            setenv("ROCKET_RK3576_MAX_ROWS", buf, 1);
        } else {
            unsetenv("ROCKET_RK3576_MAX_ROWS");
        }
        {
            conv_params_t q = p;
            q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)oh;
            if (rocket_rk3576_plan_rows(&q, s->dw, plan, 1024, &n) < 0) {
                /* A refusal is a RESULT, not a breakdown: the CBUF and weight-slice
                 * guards exist to turn a shape that would compute wrong into one that
                 * does not run. The table says which shapes must land here. */
                rc = 3;
                goto done;
            }
        }
        st->tasks = (int)n;

        for (t = 0; t < n; t++) {
            conv_params_t q = p;
            int wrote = 0;
            q.ih = plan[t].ih; q.oh = plan[t].oh;
            q.pad_top = plan[t].pad_top;
            q.input_dma  = p.input_dma  + plan[t].feature_off;
            q.output_dma = p.output_dma + plan[t].output_off;
            /* ALWAYS, not just when the plan split. A single-task plan's window is
             * still shorter than the plane whenever the last output row does not
             * reach the bottom input row — any stride > 1 whose output does not
             * consume the plane, which is ordinary VALID s2 geometry. Setting these
             * only for n > 1 leaves the emitter deriving the DDR channel-group
             * stride from the WINDOW, so every channel group past the first reads at
             * the wrong offset and the surface comes back unrelated to the input. */
            q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)oh;
            if (verbose && n > 1)
                printf("    task %u/%u: in rows [%u,%u) pad_top %u -> out rows [%u,%u)\n",
                       t + 1, n, plan[t].iy0, plan[t].iy0 + plan[t].ih, plan[t].pad_top,
                       plan[t].oy0, plan[t].oy0 + plan[t].oh);

            rc = s->dw ? gen_conv2d_dw_int8_rk3576(&q) : gen_conv2d_int8_rk3576(&q);
            if (rc != 0) { printf("  generator failed: %d\n", rc); goto done; }

            rocket_bo_prep(fd, &r_bo, 1, 0);
            memcpy(r_bo.ptr, ops, q.task_count * sizeof(uint64_t));
            rocket_bo_fini(fd, &r_bo);

            /* Per-task wall check. A walled task leaves ITS OWN rows untouched while
             * every other task's land, so the run comes back a few rows wrong and
             * reads like window arithmetic. Every retry is counted and reported: on a
             * driver carrying the per-SoC PC_TASK_CON width this stays at zero, and a
             * non-zero count is the measurement rather than an inconvenience. */
            /* Count a retry only once a LATER attempt succeeds. A task that never
             * writes at all was not walled — it is a shape this emitter does not
             * drive — and counting it as a wall retry would make the summary accuse
             * a correctly patched driver. */
            for (r = 0; (int)r <= retries; r++) {
                if (t + r > 0) sleep_ms(r ? retry_ms : gap);
                rc = rocket_submit_matmul(fd, &r_bo, q.task_count, in_h, 4, out_h, 1, 2000);
                if (rc != 0) { printf("  submit failed: %d\n", rc); goto done; }
                if (rocket_bo_prep(fd, &o_bo, 0, 2000000000ull) < 0) {
                    printf("  PREP_BO on the output timed out\n"); rc = -1; goto done;
                }
                wrote = rows_written((const int8_t *)o_bo.ptr, ocreg, ow,
                                     surf_elems, plan[t].oy0, plan[t].oh);
                rocket_bo_fini(fd, &o_bo);
                if (wrote) { st->retries += (int)r; break; }
            }
            if (!wrote) {
                printf("  task %u never wrote its rows (cold-start wall, %d retries)\n",
                       t + 1, retries);
                rc = -1; goto done;
            }
        }
        st->ms = now_ms() - t0;
    }

    rocket_bo_prep(fd, &o_bo, 0, 2000000000ull);
    memcpy(out, o_bo.ptr, obytes);
    rocket_bo_fini(fd, &o_bo);

    /* ---- the CPU model ---- */
    for (i = 0; i < obytes; i++)
        if ((uint8_t)out[i] == SENTINEL) st->untouched++;

    st->obytes = (int)obytes;
    st->total  = (int)(oc * oh * ow);
    for (c = 0; c < oc; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                int64_t acc = bias[c], xsum = 0;
                int got, want, d;
                for (kh = 0; kh < k; kh++)
                    for (kw = 0; kw < k; kw++) {
                        int iy = (int)(y * stride + kh) - pad_lead;
                        int ix = (int)(x * stride + kw) - pad_lead;
                        int in_bounds = iy >= 0 && iy < (int)ih && ix >= 0 && ix < (int)iw;
                        if (s->dw) {
                            /* The pad tap is the symmetric zero point, so an out-of-
                             * bounds window contributes nothing. */
                            if (in_bounds)
                                acc += (int64_t)INP(c, iy, ix) * WW(c, kh, kw);
                        } else {
                            if (!in_bounds) continue;
                            for (i = 0; i < ic; i++) {
                                acc  += (int64_t)INP(i, iy, ix) * WD(c, i, kh, kw);
                                xsum += (int64_t)INP(i, iy, ix);
                            }
                        }
                    }
                want = requant_apply(acc, scale, shift_reg);
                got  = out[out_index(surf_elems, ow, c, y, x)];
                if (wzp && !s->dw) {
                    /* Score both signs, and the symmetric model as the control: if the
                     * part ignored B entirely, that is the one that would match.
                     * `+B*sum(x)` is the settled answer and explains every output. */
                    if (got == requant_apply(acc - (int64_t)wzp * xsum, scale, shift_reg))
                        st->asym_minus++;
                    if (got == requant_apply(acc + (int64_t)wzp * xsum, scale, shift_reg))
                        st->asym_plus++;
                    if (got == want) st->asym_ignored++;
                }
                d = got > want ? got - want : want - got;
                if (d > st->maxdiff) st->maxdiff = d;
                if (got == want) st->exact++;
                else if (verbose && shown < 8) {
                    printf("    mism c=%u y=%u x=%u: want %d got %d (acc %lld)\n",
                           c, y, x, want, got, (long long)acc);
                    shown++;
                }
            }
    rc = (st->exact == st->total) ? 0 : 1;

#undef INP
#undef WD
#undef WW
done:
    if (r_bo.ptr)  rocket_bo_free(fd, &r_bo);
    if (o_bo.ptr)  rocket_bo_free(fd, &o_bo);
    if (b_bo.ptr)  rocket_bo_free(fd, &b_bo);
    if (w_bo.ptr)  rocket_bo_free(fd, &w_bo);
    if (in_bo.ptr) rocket_bo_free(fd, &in_bo);
    free(in_plain); free(w_plain); free(in_cube); free(w_cube); free(out); free(bias);
    return rc < 0 ? 1 : rc;   /* 0 exact, 1 wrong, 2 skip, 3 refused by the planner */
}

/* ============================================================================
 * SECTION — dwmap: read the depthwise weight-cube layout off the hardware
 *
 * The depthwise weight cube is the one operand whose layout no capture pins: the
 * vendor's `.rknn` carries the register program, not the weight bytes, and the
 * RK3588's group-of-64 packing is itself an unvalidated Mesa reading. Guessing
 * costs a run per guess and a wrong guess looks exactly like a wrong register.
 *
 * So ask the part instead. Drive an IMPULSE feature — one non-zero pixel at
 * (y0,x0), the same in every channel — and a weight cube that is entirely zero
 * except for ONE byte. Whatever that byte means, the surface that comes back
 * carries it at exactly one position of exactly one channel, and the position
 * names the kernel tap:
 *
 *     out channel  = the channel that byte belongs to
 *     out y        = y0 - kh   (stride 1, no pad), so kh = y0 - y
 *     out x        = x0 - kw
 *
 * Sweeping the byte offset across the cube therefore reads the whole layout off
 * in one pass, at one submit per byte. It also settles the element WIDTH without a
 * separate experiment: the probe writes 64, so a 1-byte layout answers 64 at every
 * live offset, while a 2-byte layout answers 64 on the low byte of each element and
 * saturates (127) on the high byte, where the same write means 64<<8.
 * ==========================================================================*/
static int dw_map(int fd)
{
    unsigned c_n  = (unsigned)env_int("ROCKET_G_DWMAP_C", 32);
    unsigned k    = (unsigned)env_int("ROCKET_G_DWMAP_K", 3);
    unsigned iw   = (unsigned)env_int("ROCKET_G_DWMAP_IW", 8);
    unsigned span = (unsigned)env_int("ROCKET_G_DWMAP_SPAN", 0);   /* 0 = the whole cube */
    unsigned creg = (unsigned)env_int("ROCKET_G_CREG", (int)((c_n + 31) / 32 * 32));
    unsigned ih = iw, ow = iw - k + 1, oh = ow;
    unsigned y0 = k - 1, x0 = k - 1;      /* every tap of the impulse lands in range */
    size_t in_bytes  = (size_t)((creg + 31) / 32 * 32 / C2) * ih * iw * C2;
    unsigned surf_elems = rocket_rk3576_out_surf_elems(ow, oh, 1);
    size_t obytes    = (size_t)((creg + C2 - 1) / C2) * surf_elems * C2;
    size_t coeff     = rocket_rk3576_coeff_bytes_dw(creg);
    /* Cover every candidate packing at once: the register the emitter programs as
     * total weight bytes is the largest of them, so a cube that big cannot be
     * under-read whichever layout turns out to be live. */
    size_t w_bytes   = (size_t)creg * k * k * 2;
    rocket_bo in_bo = {0}, w_bo = {0}, b_bo = {0}, o_bo = {0}, r_bo = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p = {0};
    int32_t *bias = NULL;
    unsigned off, live = 0;
    int rc = 1;

    if (!span || span > w_bytes) span = (unsigned)w_bytes;
    bias = calloc(creg, sizeof *bias);
    if (!bias) return 1;

    if (rocket_bo_alloc(fd, in_bytes, &in_bo) < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes,  &w_bo)  < 0) goto done;
    if (rocket_bo_alloc(fd, coeff,    &b_bo)  < 0) goto done;
    if (rocket_bo_alloc(fd, obytes,   &o_bo)  < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &r_bo) < 0) goto done;

    /* The impulse, in every channel. */
    rocket_bo_prep(fd, &in_bo, 1, 0);
    memset(in_bo.ptr, 0, in_bytes);
    {
        int8_t *f = in_bo.ptr;
        unsigned c;
        for (c = 0; c < c_n; c++)
            f[feature_data((int)creg, (int)ih, (int)iw, C2,
                           (int)c + 1, (int)y0 + 1, (int)x0 + 1)] = 64;
    }
    rocket_bo_fini(fd, &in_bo);

    rocket_bo_prep(fd, &b_bo, 1, 0);
    rocket_rk3576_pack_coeff_dw(b_bo.ptr, coeff, bias, creg);
    rocket_bo_fini(fd, &b_bo);

    p.ic = (uint16_t)creg; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.oc = (uint16_t)creg; p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
    p.kh = (uint16_t)k;    p.kw = (uint16_t)k;
    p.stride_y = p.stride_x = 1;
    p.int8_out = 1;
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 64.0f;  /* out == the weight byte */
    p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
    p.tasks       = ops;
    p.input_dma   = in_bo.dma_address;
    p.weights_dma = w_bo.dma_address;
    p.bias_dma    = b_bo.dma_address;
    p.output_dma  = o_bo.dma_address;
    p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)oh;

    in_h[0] = in_bo.handle; in_h[1] = w_bo.handle;
    in_h[2] = b_bo.handle;  in_h[3] = r_bo.handle;
    out_h[0] = o_bo.handle;

    if (gen_conv2d_dw_int8_rk3576(&p) != 0) { printf("dwmap: generator failed\n"); goto done; }
    rocket_bo_prep(fd, &r_bo, 1, 0);
    memcpy(r_bo.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &r_bo);

    printf("dwmap: C=%u (programmed %u) k=%u plane %ux%u -> %ux%u, impulse at (%u,%u), "
           "cube %zu bytes\n", c_n, creg, k, iw, ih, ow, oh, y0, x0, w_bytes);

    for (off = 0; off < span; off++) {
        unsigned c, y, x, hits = 0, hc = 0, hy = 0, hx = 0;
        int hv = 0;
        const int8_t *o;

        rocket_bo_prep(fd, &w_bo, 1, 0);
        memset(w_bo.ptr, 0, w_bytes);
        ((uint8_t *)w_bo.ptr)[off] = 64;
        rocket_bo_fini(fd, &w_bo);

        rocket_bo_prep(fd, &o_bo, 1, 0);
        memset(o_bo.ptr, SENTINEL, obytes);
        rocket_bo_fini(fd, &o_bo);

        if (rocket_submit_matmul(fd, &r_bo, p.task_count, in_h, 4, out_h, 1, 2000) != 0) {
            printf("dwmap: submit failed at off %u\n", off); goto done;
        }
        if (rocket_bo_prep(fd, &o_bo, 0, 2000000000ull) < 0) {
            printf("dwmap: PREP_BO timed out at off %u\n", off); goto done;
        }
        o = (const int8_t *)o_bo.ptr;
        for (c = 0; c < creg; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    int v = o[out_index(surf_elems, ow, c, y, x)];
                    if (v && (uint8_t)v != SENTINEL) {
                        if (!hits) { hc = c; hy = y; hx = x; hv = v; }
                        hits++;
                    }
                }
        rocket_bo_fini(fd, &o_bo);

        if (!hits) {
            printf("  off %4u -> dead\n", off);
        } else {
            live++;
            printf("  off %4u -> c=%-3u kh=%-2d kw=%-2d v=%-4d%s\n", off, hc,
                   (int)y0 - (int)hy, (int)x0 - (int)hx, hv,
                   hits > 1 ? "  (MULTIPLE positions)" : "");
        }
    }
    printf("dwmap: %u of %u cube bytes live\n", live, span);
    rc = 0;
done:
    if (r_bo.ptr)  rocket_bo_free(fd, &r_bo);
    if (o_bo.ptr)  rocket_bo_free(fd, &o_bo);
    if (b_bo.ptr)  rocket_bo_free(fd, &b_bo);
    if (w_bo.ptr)  rocket_bo_free(fd, &w_bo);
    if (in_bo.ptr) rocket_bo_free(fd, &in_bo);
    free(bias);
    return rc;
}

/* ============================================================================
 * SECTION — dwbias: read the depthwise COEFFICIENT indexing off the hardware
 *
 * Depthwise drives exactly EIGHT live output channels at C=32 and none at all above
 * it, and eight is the coefficient buffer's own group: the A/B/C layout is 64-byte
 * groups of 8 output channels, and the C multiplier in a group gates the whole BS
 * stage for it — at C=0 the DPU writes a full, correctly sized, entirely empty
 * surface. Both signatures are what a depthwise-specific indexing of that buffer
 * would produce, one that lines up for a single group at C=32 and lands past the end
 * above it.
 *
 * Asking that costs one submit, not a sweep. Drive ZERO weights and a ZERO feature,
 * so the accumulator is zero and every output value is its channel's bias alone, and
 * give channel c the bias c+1 so the value names the slot it was read from. Then:
 *
 *     channel c reads back c+1   the buffer is indexed as the direct path indexes it
 *     channel c reads back v!=0  it read channel v-1's slot: v-1-c is the shift
 *     channel c reads back 0     that channel's group is gated off (its C is not 1)
 *
 * The direct path is the control: the same probe there must return c+1 everywhere,
 * which is what the bias probe in rk3576_first_light already establishes.
 * ==========================================================================*/
static int dw_bias(int fd, int dw)
{
    unsigned c_n  = (unsigned)env_int("ROCKET_G_DWMAP_C", 32);
    unsigned k    = (unsigned)env_int("ROCKET_G_DWMAP_K", 3);
    unsigned iw   = (unsigned)env_int("ROCKET_G_DWMAP_IW", 8);
    unsigned creg = (unsigned)env_int("ROCKET_G_CREG", (int)((c_n + 31) / 32 * 32));
    unsigned ih = iw, ow = iw - k + 1, oh = ow;
    size_t in_bytes = (size_t)((creg + 31) / 32 * 32 / C2) * ih * iw * C2;
    unsigned surf_elems = rocket_rk3576_out_surf_elems(ow, oh, dw);
    size_t obytes   = (size_t)((creg + C2 - 1) / C2) * surf_elems * C2;
    size_t coeff    = dw ? rocket_rk3576_coeff_bytes_dw(creg)
                         : rocket_rk3576_coeff_bytes(creg);
    size_t w_bytes  = dw ? (size_t)creg * k * k * 2
                         : (size_t)creg * creg * k * k;
    rocket_bo bo_in = {0}, bo_w = {0}, bo_b = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p = {0};
    int32_t *bias = NULL;
    unsigned c, y, x, gated = 0, shifted = 0, exact = 0;
    int rc = 1;

    /* The slot name is the VALUE, so it has to survive the int8 epilogue: a channel
     * whose bias clips reads back 127 whatever slot it came from, and 127 is a
     * perfectly plausible slot index. ROCKET_G_DWBIAS_BASE/_STEP move the naming off
     * the clip — run the probe twice at different bases and a genuine slot read moves
     * with them while a saturated one does not. */
    {
        int base = env_int("ROCKET_G_DWBIAS_BASE", 1);
        int step = env_int("ROCKET_G_DWBIAS_STEP", 1);
        bias = calloc(creg, sizeof *bias);
        if (!bias) return 1;
        for (c = 0; c < creg; c++) bias[c] = base + (int32_t)c * step;
    }

    if (rocket_bo_alloc(fd, in_bytes, &bo_in) < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes,  &bo_w)  < 0) goto done;
    if (rocket_bo_alloc(fd, coeff,    &bo_b)  < 0) goto done;
    if (rocket_bo_alloc(fd, obytes,   &bo_o)  < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    rocket_bo_prep(fd, &bo_in, 1, 0);
    memset(bo_in.ptr, 0, in_bytes);
    rocket_bo_fini(fd, &bo_in);
    rocket_bo_prep(fd, &bo_w, 1, 0);
    memset(bo_w.ptr, 0, w_bytes);
    rocket_bo_fini(fd, &bo_w);
    rocket_bo_prep(fd, &bo_b, 1, 0);
    if (dw) rocket_rk3576_pack_coeff_dw(bo_b.ptr, coeff, bias, creg);
    else    rocket_rk3576_pack_coeff(bo_b.ptr, coeff, bias, creg);
    rocket_bo_fini(fd, &bo_b);

    p.ic = (uint16_t)creg; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.oc = (uint16_t)creg; p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
    p.kh = (uint16_t)k;    p.kw = (uint16_t)k;
    p.stride_y = p.stride_x = 1;
    p.int8_out = 1;
    /* Unit requant: the surface carries the bias itself. */
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;
    p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
    p.tasks       = ops;
    p.input_dma   = bo_in.dma_address;
    p.weights_dma = bo_w.dma_address;
    p.bias_dma    = bo_b.dma_address;
    p.output_dma  = bo_o.dma_address;
    p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)oh;

    in_h[0] = bo_in.handle; in_h[1] = bo_w.handle;
    in_h[2] = bo_b.handle;  in_h[3] = bo_r.handle;
    out_h[0] = bo_o.handle;

    if ((dw ? gen_conv2d_dw_int8_rk3576(&p) : gen_conv2d_int8_rk3576(&p)) != 0) {
        printf("dwbias: generator failed\n");
        goto done;
    }
    rocket_bo_prep(fd, &bo_r, 1, 0);
    memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &bo_r);

    rocket_bo_prep(fd, &bo_o, 1, 0);
    memset(bo_o.ptr, SENTINEL, obytes);
    rocket_bo_fini(fd, &bo_o);

    printf("dwbias: %s C=%u (programmed %u) k=%u plane %ux%u -> %ux%u, "
           "zero weights, bias[c]=c+1\n",
           dw ? "depthwise" : "direct", c_n, creg, k, iw, ih, ow, oh);

    if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 4, out_h, 1, 2000) != 0) {
        printf("dwbias: submit failed\n"); goto done;
    }
    if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
        printf("dwbias: PREP_BO timed out\n"); goto done;
    }

    for (c = 0; c < creg; c++) {
        const int8_t *o = (const int8_t *)bo_o.ptr;
        int v0 = o[out_index(surf_elems, ow, c, 0, 0)];
        int uniform = 1;
        for (y = 0; y < oh && uniform; y++)
            for (x = 0; x < ow; x++)
                if (o[out_index(surf_elems, ow, c, y, x)] != v0) {
                    uniform = 0; break;
                }
        if ((uint8_t)v0 == SENTINEL) {
            printf("  c=%-3u UNTOUCHED\n", c);
            gated++;
        } else if (v0 == 0) {
            printf("  c=%-3u gated off (its coefficient group's C is not live)\n", c);
            gated++;
        } else if (v0 == 127 || v0 == -128) {
            printf("  c=%-3u SATURATED at %d — not a slot read; lower "
                   "ROCKET_G_DWBIAS_STEP and re-run\n", c, v0);
            shifted++;
        } else if (v0 == bias[c]) {
            exact++;
            if (!uniform) printf("  c=%-3u bias correct but NOT uniform over the surface\n", c);
        } else {
            unsigned slot; int found = -1;
            for (slot = 0; slot < creg; slot++)
                if (bias[slot] == v0) { found = (int)slot; break; }
            if (found >= 0)
                printf("  c=%-3u read slot %d (shift %+d)%s\n", c, found,
                       found - (int)c, uniform ? "" : "  (not uniform)");
            else
                printf("  c=%-3u read %d, which is no channel's bias%s\n", c, v0,
                       uniform ? "" : "  (not uniform)");
            shifted++;
        }
    }
    rocket_bo_fini(fd, &bo_o);
    printf("dwbias: %u of %u channels carry their own bias, %u read another slot, "
           "%u gated/untouched\n", exact, creg, shifted, gated);
    rc = (exact == creg) ? 0 : 1;
done:
    if (bo_r.ptr)  rocket_bo_free(fd, &bo_r);
    if (bo_o.ptr)  rocket_bo_free(fd, &bo_o);
    if (bo_b.ptr)  rocket_bo_free(fd, &bo_b);
    if (bo_w.ptr)  rocket_bo_free(fd, &bo_w);
    if (bo_in.ptr) rocket_bo_free(fd, &bo_in);
    free(bias);
    return rc;
}

/* ============================================================================
 * SECTION — dwcoeff: read the COEFFICIENT BUFFER's indexing off the hardware
 *
 * dwbias asks which slot a channel read; this asks the inverse, which is the one
 * that decodes rather than scores: bump one int16 of the coefficient buffer and see
 * which OUTPUT BYTES move. A permutation read the first way can be misread as
 * saturation or as a gated group; read this way it is a map or it is not.
 *
 * Two things make it readable on a path whose surface is not flat.
 *
 * The baseline is MEASURED, not assumed. An earlier form of this probe compared every
 * position against the constant it had packed, which is only a delta where the part
 * agrees that the surface is flat — on the depthwise path it is not, so every position
 * read as "changed" and the probe said nothing. One unprobed submit up front, kept as
 * the reference image, makes a per-position delta meaningful whatever the baseline
 * looks like.
 *
 * And it reads the RAW BO. feature_data() is the direct path's cube, so scoring a
 * depthwise surface through it hides exactly the permutation this mode is looking
 * for. A changed byte is reported at its own offset, folded to (surface, lane) with
 * the pixel count, so a map falls out of the report rather than being fitted to it.
 *
 * The A and C terms answer with different arithmetic, so a moved position names not
 * only its channel but which term it is:
 *
 *     position reads BIAS + PROBE            that position is the channel's A (bias)
 *     position reads BIAS * (1 + PROBE)      that position is its C (multiplier)
 *     nothing moves                          the position is B, padding, or dead
 *
 * A position already at the int8 clip in the baseline cannot show an increase.
 * ROCKET_G_COEFF_OUTSCALE divides the requant down so a saturated accumulator lands
 * in range and its arithmetic becomes visible; use it with a probe scaled to match.
 *
 * One submit per int16, which for 32 channels is 160 of them. Run it on the direct
 * path first (`dwcoeff direct`): that path is bit-exact, so its map is the control
 * this encoder already believes, and a depthwise map that differs from it is the
 * depthwise indexing.
 * ==========================================================================*/
static int dw_coeff(int fd, int dw)
{
    unsigned c_n  = (unsigned)env_int("ROCKET_G_DWMAP_C", 32);
    unsigned k    = (unsigned)env_int("ROCKET_G_DWMAP_K", 3);
    unsigned iw   = (unsigned)env_int("ROCKET_G_DWMAP_IW", 8);
    unsigned creg = (unsigned)env_int("ROCKET_G_CREG", (int)((c_n + 31) / 32 * 32));
    int      base = env_int("ROCKET_G_COEFF_BASE", 2);
    int      probe = env_int("ROCKET_G_COEFF_PROBE", 5);
    int      oscale = env_int("ROCKET_G_COEFF_OUTSCALE", 1);
    int      gap = env_int("ROCKET_G_COEFF_GAP_MS", 0);
    unsigned ih = iw, ow = iw - k + 1, oh = ow;
    size_t in_bytes = (size_t)((creg + 31) / 32 * 32 / C2) * ih * iw * C2;
    /* Four times the direct surface, as dwout allocates: a writer that walks past the
     * direct sizing has to land inside the BO to be seen moving at all. */
    unsigned surf = ((creg + C2 - 1) / C2)
                    * rocket_rk3576_out_surf_elems(ow, oh, dw) * C2;
    size_t obytes   = (size_t)4 * surf;
    size_t coeff    = dw ? rocket_rk3576_coeff_bytes_dw(creg)
                         : rocket_rk3576_coeff_bytes(creg);
    size_t w_bytes  = dw ? rocket_rk3576_weight_dw_bytes(creg, k, k)
                         : (size_t)creg * creg * k * k;
    rocket_bo bo_in = {0}, bo_w = {0}, bo_b = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p = {0};
    int32_t *bias = NULL;
    uint8_t *ref = NULL;
    unsigned c, slot, nslots = (unsigned)(coeff / 2), live = 0;
    unsigned slot_lo = (unsigned)env_int("ROCKET_G_COEFF_LO", 0);
    unsigned slot_hi = (unsigned)env_int("ROCKET_G_COEFF_HI", (int)nslots);
    unsigned plane = rocket_rk3576_out_surf_elems(ow, oh, dw) * C2;
    int rc = 1;

    if (slot_hi > nslots) slot_hi = nslots;
    bias = calloc(creg, sizeof *bias);
    ref  = malloc(obytes);
    if (!bias || !ref) goto done;
    for (c = 0; c < creg; c++) bias[c] = base;

    if (rocket_bo_alloc(fd, in_bytes, &bo_in) < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes,  &bo_w)  < 0) goto done;
    if (rocket_bo_alloc(fd, coeff,    &bo_b)  < 0) goto done;
    if (rocket_bo_alloc(fd, obytes,   &bo_o)  < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    rocket_bo_prep(fd, &bo_in, 1, 0);
    memset(bo_in.ptr, 0, in_bytes);
    rocket_bo_fini(fd, &bo_in);
    rocket_bo_prep(fd, &bo_w, 1, 0);
    memset(bo_w.ptr, 0, w_bytes);
    rocket_bo_fini(fd, &bo_w);

    p.ic = (uint16_t)creg; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.oc = (uint16_t)creg; p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
    p.kh = (uint16_t)k;    p.kw = (uint16_t)k;
    p.stride_y = p.stride_x = 1;
    p.int8_out = 1;
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = (float)oscale;
    p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
    p.tasks       = ops;
    p.input_dma   = bo_in.dma_address;
    p.weights_dma = bo_w.dma_address;
    p.bias_dma    = bo_b.dma_address;
    p.output_dma  = bo_o.dma_address;
    p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)oh;

    in_h[0] = bo_in.handle; in_h[1] = bo_w.handle;
    in_h[2] = bo_b.handle;  in_h[3] = bo_r.handle;
    out_h[0] = bo_o.handle;

    if ((dw ? gen_conv2d_dw_int8_rk3576(&p) : gen_conv2d_int8_rk3576(&p)) != 0) {
        printf("dwcoeff: generator failed\n"); goto done;
    }
    rocket_bo_prep(fd, &bo_r, 1, 0);
    memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &bo_r);

    printf("dwcoeff: %s C=%u (programmed %u) k=%u plane %ux%u -> %ux%u, "
           "%u bytes of coefficient buffer, bias %d everywhere, probe +%d, "
           "out_scale %d, slots [%u,%u)\n",
           dw ? "depthwise" : "direct", c_n, creg, k, iw, ih, ow, oh,
           (unsigned)coeff, base, probe, oscale, slot_lo, slot_hi);

    /* ---- the measured baseline: one unprobed submit, kept as the reference ---- */
    rocket_bo_prep(fd, &bo_b, 1, 0);
    if (dw) rocket_rk3576_pack_coeff_dw(bo_b.ptr, coeff, bias, creg);
    else    rocket_rk3576_pack_coeff(bo_b.ptr, coeff, bias, creg);
    rocket_bo_fini(fd, &bo_b);
    rocket_bo_prep(fd, &bo_o, 1, 0);
    memset(bo_o.ptr, SENTINEL, obytes);
    rocket_bo_fini(fd, &bo_o);
    if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 4, out_h, 1, 2000) != 0) {
        printf("dwcoeff: baseline submit failed\n"); goto done;
    }
    if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
        printf("dwcoeff: baseline PREP_BO timed out\n"); goto done;
    }
    memcpy(ref, bo_o.ptr, obytes);
    rocket_bo_fini(fd, &bo_o);
    {
        unsigned w = 0;
        for (c = 0; c < obytes; c++) if (ref[c] != SENTINEL) w++;
        printf("  baseline: %u of %u bytes written\n", w, (unsigned)obytes);
    }

    for (slot = slot_lo; slot < slot_hi; slot++) {
        int16_t *cb;
        const uint8_t *o;
        unsigned i;
        int shown = 0;

        rocket_bo_prep(fd, &bo_b, 1, 0);
        if (dw) rocket_rk3576_pack_coeff_dw(bo_b.ptr, coeff, bias, creg);
    else    rocket_rk3576_pack_coeff(bo_b.ptr, coeff, bias, creg);
        cb = (int16_t *)bo_b.ptr;
        cb[slot] = (int16_t)(cb[slot] + probe);
        rocket_bo_fini(fd, &bo_b);

        rocket_bo_prep(fd, &bo_o, 1, 0);
        memset(bo_o.ptr, SENTINEL, obytes);
        rocket_bo_fini(fd, &bo_o);

        if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 4, out_h, 1, 2000) != 0) {
            printf("dwcoeff: submit failed at slot %u\n", slot); goto done;
        }
        if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
            printf("dwcoeff: PREP_BO timed out at slot %u\n", slot); goto done;
        }
        o = (const uint8_t *)bo_o.ptr;

        /* Fold the changed bytes to (surface, lane): the pattern repeats per pixel,
         * so one line per lane with a pixel count is the whole of what moved. */
        for (i = 0; i < obytes; i++) {
            unsigned lane, s, npix = 0, pix;
            if (o[i] == ref[i]) continue;
            s = i / plane; lane = i % C2;
            if (i % plane / C2 != 0) continue;         /* count from pixel 0 only */
            for (pix = 0; pix < plane / C2; pix++) {
                unsigned off = s * plane + pix * C2 + lane;
                if (o[off] != ref[off]) npix++;
            }
            if (!shown) printf("  slot %3u (byte %3u):\n", slot, slot * 2), shown = 1;
            printf("      surface %u lane %2u : %4d -> %-4d  (%u of %u pixels)\n",
                   s, lane, (int)ref[i], (int)o[i], npix, plane / C2);
            live++;
        }
        /* Anything that moved at a pixel other than 0 would be missed by the fold
         * above; say so rather than let it pass silently. */
        {
            unsigned odd = 0;
            for (i = 0; i < obytes; i++) {
                unsigned s = i / plane, lane = i % C2, pix = i % plane / C2;
                unsigned ref0 = s * plane + lane;
                if (!pix) continue;
                if ((o[i] != ref[i]) != (o[ref0] != ref[ref0])) odd++;
            }
            if (odd) printf("      (%u bytes move at some pixels and not at pixel 0)\n", odd);
        }
        rocket_bo_fini(fd, &bo_o);
        if (gap) sleep_ms(gap);
    }
    printf("dwcoeff: %u live (slot, lane) pairs over %u slots\n", live,
           slot_hi - slot_lo);
    rc = 0;
done:
    if (bo_r.ptr)  rocket_bo_free(fd, &bo_r);
    if (bo_o.ptr)  rocket_bo_free(fd, &bo_o);
    if (bo_b.ptr)  rocket_bo_free(fd, &bo_b);
    if (bo_w.ptr)  rocket_bo_free(fd, &bo_w);
    if (bo_in.ptr) rocket_bo_free(fd, &bo_in);
    free(bias);
    free(ref);
    return rc;
}

/* ============================================================================
 * SECTION — dwout: read the depthwise OUTPUT CUBE off the hardware
 *
 * Every other depthwise probe here reads the surface through feature_data(), the
 * direct path's NC1HWC2 cube — so every one of them is scored through a layout it
 * cannot itself check, and a channel permutation and a wrong descatter are
 * indistinguishable in their output. This one does not descatter at all.
 *
 * Zero weights, zero feature, bias[c] = c+1: under any layout every written byte is
 * some channel's own bias, so a byte holding v NAMES channel v-1 and the byte's
 * position is where that channel landed. Dumping the raw BO therefore decodes the
 * cube instead of scoring it. The plane is chosen so the channel names (1..C) and the
 * output pixel count stay far apart, and the surface is filled with a sentinel first
 * so an unwritten byte is visible as unwritten rather than as channel -87.
 *
 * The direct path is the control: there the dump must be C/16 planes of oh*ow*16
 * bytes with channel c at lane c%16 of plane c/16.
 * ==========================================================================*/
/* The coefficient group, written with A and C under INDEPENDENT control.
 *
 * rocket_rk3576_pack_coeff() drives one scalar multiplier across every channel, which
 * cannot separate "this lane read the wrong A" from "this lane read its C out of the A
 * region". The BS stage computes (acc + A) * C, so a lane whose C comes from its own A
 * reads A*A and one that reads a real C reads A*C — two ramps tell them apart in one
 * submit. The group layout is the documented one in npu_regcmd_rk3576.h:
 *   A[oc] int32 at (oc%8)*4, B[oc] int16 at 32+(oc%8)*2, C[oc] int16 at 48+(oc%8)*2,
 * with oc in group oc/8, groups of 64 bytes. */
static void pack_abc(void *dst, size_t bytes, unsigned oc,
                     int abase, int astep, int cbase, int cstep,
                     unsigned stride, unsigned coff)
{
    uint8_t *b = (uint8_t *)dst;
    unsigned c;
    memset(b, 0, bytes);
    for (c = 0; c < oc; c++) {
        uint8_t *g = b + (size_t)(c / 8) * stride;
        int32_t a = (int32_t)(abase + (int)c * astep);
        int16_t m = (int16_t)(cbase + (int)c * cstep);
        memcpy(g + (c % 8) * 4, &a, sizeof a);
        memcpy(g + coff + (c % 8) * 2, &m, sizeof m);
    }
}

static int dw_out(int fd, int dw)
{
    unsigned c_n  = (unsigned)env_int("ROCKET_G_DWMAP_C", 32);
    unsigned k    = (unsigned)env_int("ROCKET_G_DWMAP_K", 3);
    unsigned iw   = (unsigned)env_int("ROCKET_G_DWMAP_IW", 8);
    unsigned creg = (unsigned)env_int("ROCKET_G_CREG", (int)((c_n + 31) / 32 * 32));
    /* A lane at the int8 clip says only "large". ROCKET_G_DWOUT_OUTSCALE divides the
     * requant down so a saturated accumulator lands in range and NAMES its magnitude,
     * which is what separates "read a bias" from "read a pair of C multipliers as one
     * 32-bit word". The bias naming survives it: at scale S a channel's own bias
     * rounds to zero and a large read does not. */
    int      oscale = env_int("ROCKET_G_DWOUT_OUTSCALE", 1);
    int      bbase = env_int("ROCKET_G_DWOUT_BASE", 1);
    int      bstep = env_int("ROCKET_G_DWOUT_STEP", 1);
    int      cbase = env_int("ROCKET_G_DWOUT_CBASE", 1);
    int      cstep = env_int("ROCKET_G_DWOUT_CSTEP", 0);
    /* The coefficient group geometry, so the depthwise candidate can be driven
     * against the direct one in the same binary: stride 64 with C at +48 is the
     * direct group, stride 48 with C at +32 the depthwise candidate. */
    unsigned gstride = (unsigned)env_int("ROCKET_G_DWOUT_GSTRIDE", dw ? 48 : 64);
    unsigned gcoff   = (unsigned)env_int("ROCKET_G_DWOUT_GCOFF", dw ? 32 : 48);
    unsigned ih = iw, ow = iw - k + 1, oh = ow;
    /* Allocate FOUR times the direct path's surface: the depthwise channel-group jump
     * is four surfaces where the direct one is two, so a writer that walks further
     * than the direct sizing must land inside this BO to be seen at all. */
    size_t obytes   = (size_t)4 * ((creg + C2 - 1) / C2)
                      * rocket_rk3576_out_surf_elems(ow, oh, dw) * C2;
    size_t in_bytes = (size_t)((creg + 31) / 32 * 32 / C2) * ih * iw * C2;
    size_t coeff    = dw ? rocket_rk3576_coeff_bytes_dw(creg)
                         : rocket_rk3576_coeff_bytes(creg);
    size_t w_bytes  = dw ? rocket_rk3576_weight_dw_bytes(creg, k, k)
                         : (size_t)creg * creg * k * k;
    rocket_bo bo_in = {0}, bo_w = {0}, bo_b = {0}, bo_o = {0}, bo_r = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p = {0};
    unsigned i, written = 0, named = 0, other = 0;
    int rc = 1;

    if (rocket_bo_alloc(fd, in_bytes, &bo_in) < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes,  &bo_w)  < 0) goto done;
    if (rocket_bo_alloc(fd, coeff,    &bo_b)  < 0) goto done;
    if (rocket_bo_alloc(fd, obytes,   &bo_o)  < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &bo_r) < 0) goto done;

    rocket_bo_prep(fd, &bo_in, 1, 0);  memset(bo_in.ptr, 0, in_bytes); rocket_bo_fini(fd, &bo_in);
    rocket_bo_prep(fd, &bo_w, 1, 0);   memset(bo_w.ptr, 0, w_bytes);   rocket_bo_fini(fd, &bo_w);
    rocket_bo_prep(fd, &bo_b, 1, 0);
    pack_abc(bo_b.ptr, coeff, creg, bbase, bstep, cbase, cstep, gstride, gcoff);
    rocket_bo_fini(fd, &bo_b);

    p.ic = (uint16_t)creg; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.oc = (uint16_t)creg; p.oh = (uint16_t)oh; p.ow = (uint16_t)ow;
    p.kh = (uint16_t)k;    p.kw = (uint16_t)k;
    p.stride_y = p.stride_x = 1;
    p.int8_out = 1;
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = (float)oscale;
    p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
    p.tasks       = ops;
    p.input_dma   = bo_in.dma_address;
    p.weights_dma = bo_w.dma_address;
    p.bias_dma    = bo_b.dma_address;
    p.output_dma  = bo_o.dma_address;
    p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)oh;

    in_h[0] = bo_in.handle; in_h[1] = bo_w.handle;
    in_h[2] = bo_b.handle;  in_h[3] = bo_r.handle;
    out_h[0] = bo_o.handle;

    if ((dw ? gen_conv2d_dw_int8_rk3576(&p) : gen_conv2d_int8_rk3576(&p)) != 0) {
        printf("dwout: generator failed\n"); goto done;
    }
    rocket_bo_prep(fd, &bo_r, 1, 0);
    memcpy(bo_r.ptr, ops, p.task_count * sizeof(uint64_t));
    rocket_bo_fini(fd, &bo_r);
    rocket_bo_prep(fd, &bo_o, 1, 0);
    memset(bo_o.ptr, SENTINEL, obytes);
    rocket_bo_fini(fd, &bo_o);

    printf("dwout: %s C=%u (programmed %u) k=%u plane %ux%u -> %ux%u, "
           "A[c]=%d+%d*c, C[c]=%d+%d*c, group stride %u C at +%u, out_scale %d, "
           "%u output bytes allocated (4x the direct surface), %u pixels\n",
           dw ? "depthwise" : "direct", c_n, creg, k, iw, ih, ow, oh, bbase, bstep,
           cbase, cstep, gstride, gcoff, oscale, (unsigned)obytes, ow * oh);

    if (rocket_submit_matmul(fd, &bo_r, p.task_count, in_h, 4, out_h, 1, 2000) != 0) {
        printf("dwout: submit failed\n"); goto done;
    }
    if (rocket_bo_prep(fd, &bo_o, 0, 2000000000ull) < 0) {
        printf("dwout: PREP_BO timed out\n"); goto done;
    }
    {
        const uint8_t *o = (const uint8_t *)bo_o.ptr;
        unsigned run_start = 0;
        int run_val = -1;
        /* Print the surface as RUNS. A byte's value names its channel, so a run is
         * "channel v-1 occupied bytes [a,b)", and the run structure is the cube. */
        for (i = 0; i <= obytes; i++) {
            int v = (i < obytes) ? (int)o[i] : -2;
            if (v != run_val) {
                if (run_val >= 0 && run_val != SENTINEL) {
                    /* A byte NAMES the channel whose bias it carries — which is only
                     * a name where the ramp actually hits that value exactly. */
                    int d = run_val - bbase;
                    int ch = (bstep && d >= 0 && d % bstep == 0) ? d / bstep : -1;
                    int is_ch = (ch >= 0 && (unsigned)ch < creg);
                    printf("  bytes %6u..%-6u  value %3d  (%s %d, %u bytes)\n",
                           run_start, i - 1, run_val,
                           is_ch ? "channel" : "NOT A CHANNEL", ch, i - run_start);
                    if (is_ch) named += i - run_start;
                    else other += i - run_start;
                }
                if (run_val != -1 && run_val != SENTINEL) written += 0;
                run_start = i; run_val = v;
            }
        }
        for (i = 0; i < obytes; i++) if (o[i] != SENTINEL) written++;
    }
    printf("dwout: %u of %u bytes written, %u carry a channel's bias, %u do not\n",
           written, (unsigned)obytes, named, other);
    rocket_bo_fini(fd, &bo_o);
    rc = 0;
done:
    if (bo_r.ptr)  rocket_bo_free(fd, &bo_r);
    if (bo_o.ptr)  rocket_bo_free(fd, &bo_o);
    if (bo_b.ptr)  rocket_bo_free(fd, &bo_b);
    if (bo_w.ptr)  rocket_bo_free(fd, &bo_w);
    if (bo_in.ptr) rocket_bo_free(fd, &bo_in);
    return rc;
}

/* ============================================================================
 * SECTION — main
 * ==========================================================================*/
static int wanted(const shape_t *s, int argc, char **argv)
{
    const char *filter = getenv("ROCKET_G_FILTER");
    int i, any = 0;
    if (filter && *filter && !strstr(s->name, filter)) return 0;
    for (i = 1; i < argc; i++) {
        if (argv[i][0] == '-') continue;
        any = 1;
        if (!strcmp(argv[i], "all") || !strcmp(argv[i], s->group)) return 1;
    }
    if (any) return 0;
    return !strcmp(s->group, "envelope") || !strcmp(s->group, "window");
}

int main(int argc, char **argv)
{
    int fd, i, pass = 0, fail = 0, skip = 0, retries = 0;
    int gap = env_int("ROCKET_G_GAP_MS", 0);
    double t0;

    if (argc > 1 && !strcmp(argv[1], "-l")) {
        for (i = 0; i < N_SHAPES; i++)
            printf("%-9s %-18s ic%-4u oc%-4u %ux%u k%u s%u %-5s%s%s\n",
                   SHAPES[i].group, SHAPES[i].name, SHAPES[i].ic, SHAPES[i].oc,
                   SHAPES[i].iw, SHAPES[i].ih, SHAPES[i].k, SHAPES[i].stride,
                   SHAPES[i].same ? "SAME" : "VALID", SHAPES[i].dw ? " dw" : "",
                   SHAPES[i].max_rows ? " windowed" : "");
        return 0;
    }

    fd = rocket_open();
    if (fd < 0) { printf("no /dev/accel/accel0 (%d) — SKIP\n", fd); return 2; }

    /* This emitter's geometry registers are RK3576-only: driving them at an RK3588
     * programs a different part's CNA and the results would be noise rather than a
     * failure. Skip rather than report. */
    {
        const struct rocket_hw_profile *hw = rocket_hw_current();
        if (!hw || !hw->name || strcmp(hw->name, "rk3576")) {
            printf("chip is '%s', not rk3576 — SKIP\n", hw && hw->name ? hw->name : "?");
            rocket_close(fd);
            return 2;
        }
    }

    if (argc > 1 && !strcmp(argv[1], "dwmap")) {
        int r = dw_map(fd);
        rocket_close(fd);
        return r;
    }
    /* `dwbias direct` runs the same probe on the working path as the control. */
    if (argc > 1 && !strcmp(argv[1], "dwbias")) {
        int r = dw_bias(fd, !(argc > 2 && !strcmp(argv[2], "direct")));
        rocket_close(fd);
        return r;
    }
    /* `dwcoeff direct` likewise: the direct map is the control the depthwise one is
     * read against. */
    if (argc > 1 && !strcmp(argv[1], "dwcoeff")) {
        int r = dw_coeff(fd, !(argc > 2 && !strcmp(argv[2], "direct")));
        rocket_close(fd);
        return r;
    }
    /* `dwout direct` is the control for the raw output-cube dump. */
    if (argc > 1 && !strcmp(argv[1], "dwout")) {
        int r = dw_out(fd, !(argc > 2 && !strcmp(argv[2], "direct")));
        rocket_close(fd);
        return r;
    }

    printf("RK3576 conv gate — %d shapes, gap %d ms between shapes, %d wall retries\n",
           N_SHAPES, gap, env_int("ROCKET_G_RETRIES", 4));
    t0 = now_ms();

    for (i = 0; i < N_SHAPES; i++) {
        const shape_t *s = &SHAPES[i];
        struct run_stat st;
        int rc;
        if (!wanted(s, argc, argv)) continue;
        if (pass + fail + skip) sleep_ms(gap);
        rc = run_shape(fd, s, &st);
        retries += st.retries;
        if (rc == 2) { skip++; printf("SKIP %-9s %-18s\n", s->group, s->name); continue; }
        if (rc == 3 || s->refuse) {
            /* The guards are as load-bearing as the arithmetic: a shape past the
             * measured weight-slice or CBUF boundary must be refused, and one inside
             * it must not be. Both directions are a failure of the gate. */
            int ok = (rc == 3) == (s->refuse != 0);
            printf("%s %-9s %-18s %s\n", ok ? "PASS" : "FAIL", s->group, s->name,
                   rc == 3 ? (s->refuse ? "refused, as the measured boundary requires"
                                        : "REFUSED but the boundary says it computes")
                           : "COMPUTED but the measured boundary says it is wrong");
            if (ok) pass++; else fail++;
            continue;
        }
        printf("%s %-9s %-18s %6d/%-6d exact  maxdiff %-4d %2d task%s %7.1f ms%s%s\n",
               rc ? "FAIL" : "PASS", s->group, s->name, st.exact, st.total, st.maxdiff,
               st.tasks, st.tasks == 1 ? " " : "s", st.ms,
               st.retries ? "  WALL-RETRIES" : "",
               (st.untouched && st.untouched == st.obytes) ? "  NO-WRITE" : "");
        if (env_int("ROCKET_G_WZP", 0) && !s->dw)
            printf("         weight zero point %d: -B*sum(x) explains %d/%d, "
                   "+B*sum(x) %d/%d, B ignored %d/%d\n",
                   env_int("ROCKET_G_WZP", 0),
                   st.asym_minus, st.total, st.asym_plus, st.total,
                   st.asym_ignored, st.total);
        if (rc) fail++; else pass++;
    }

    printf("\n%d passed, %d failed, %d skipped in %.1f s — %d cold-start-wall retr%s\n",
           pass, fail, skip, (now_ms() - t0) / 1000.0, retries, retries == 1 ? "y" : "ies");
    if (retries)
        printf("  NOTE: results carrying retries were taken through the wall\n"
               "        compensation, not past it. On a driver WITHOUT the per-SoC\n"
               "        PC_TASK_CON width every group retries; if only a failing group\n"
               "        does, the retries are its blank surfaces tripping the same\n"
               "        heuristic — depthwise writes one — and say nothing about the\n"
               "        driver. Re-run the passing groups alone to tell the two apart.\n");
    rocket_close(fd);
    return fail ? 1 : 0;
}
