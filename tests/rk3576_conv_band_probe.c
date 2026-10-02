// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_conv_band_probe.c — does a direct int8 convolution stall where the weight-slice
 * phase rule says it will?
 *
 * A probe, not a gate. An int8 matmul job's output-channel group g completes only if its
 * weight slice S = 32*ic*kh*kw bytes, at its 64 KiB phase from the weight base, ends inside
 * the CBUF weight area the task's feature allowance F leaves, 64 KiB * floor((3072-F)/1024)
 * (rocket_rk3576_weight_phase_groups()). The convolution planner bounds the slice by the
 * measured r76_weight_slice_cap() table instead, and two bands pass that table and break
 * the rule: a slice in (64, 80] KiB at F=1024 and one in (128, 144] KiB at F=0, with more
 * groups than the rule allows. This runs named shapes through rocket_conv2d_int8_rk3576()
 * and says, per shape:
 *
 *   plan   the row window the planner allows, the allowance it programs, the groups and
 *          the slice
 *   rule   the groups the rule allows at F=0 and F=1024
 *   rc     what the entry returned, in how long
 *   ret    retirements the library scored (unwritten / written) and the kernel's
 *          'retiring it' count over the call
 *   exact  the output against the host int64 reference, and the first wrong channel
 *
 * A stall reaches the caller as a REFUSAL: the entry redoes a retired job whatever it
 * wrote and refuses when its attempts run out. So a stalled shape reads rc -4 with its
 * retirements scored written; the boundary is read by bracketing oc.
 *
 * THE OTHER THREE PATHS. `f:` runs the same question through rocket_conv2d_fp16_rk3576(),
 * whose ic > 4 form is one task per sixteen input channels carrying EVERY output channel
 * (no output-channel tile) at whatever allowance the whole plane needs, and whose float
 * weight cube groups output channels by sixteen: a group's slice there is
 * 16*16*kh*kw*2 = 512*kh*kw bytes, which is also what the planner charges (32*ic*kh*kw at
 * ic 16). `d:` runs rocket_conv2d_dw_int8_rk3576(), whose planner charges the WHOLE
 * depthwise cube against the pool, at the caller's raw channel count while the cube is
 * sized at the count rounded up to 16. The fp16 arm scores against an exact host sum
 * (small-integer operands, so every partial is an exact fp32 and the output an exact fp16
 * where |sum| <= 2048); the depthwise one against the int64 reference and the requant
 * model. Each prints the allowance its planner reaches; set ROCKET_RK3576_DUMP=1 with
 * ROCKET_LOG_LEVEL=info to read CNA 0x1040 off the program itself.
 *
 * Usage: rk3576_conv_band_probe ARM...
 *   ARM = ic:oc:ih:iw:k     int8 direct   (stride 1, pad k/2)
 *       = f:ic:oc:ih:iw:k   fp16          (stride 1, pad k/2; ic > 4)
 *       = d:c:ih:iw:k       int8 depthwise (stride 1, pad k/2)
 * Exit: 0 ran, 2 no NPU or not an RK3576.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_hw_profile.h"
#include "npu_regcmd_rk3576.h"
#include "npu_hw.h"
#include "requant_model.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static long klog_retired(void)
{
    struct timespec ts = { 0, 200000000L };
    FILE *f;
    long n = -1;
    nanosleep(&ts, NULL);
    f = popen("journalctl -b -k -q --no-pager 2>/dev/null | grep -c 'retiring it'", "r");
    if (!f) return -1;
    if (fscanf(f, "%ld", &n) != 1) n = -1;
    if (pclose(f) == -1) n = -1;
    return n;
}

static int run(int fd, unsigned ic, unsigned oc, unsigned ih, unsigned iw, unsigned k)
{
    const unsigned pad = k / 2, oh = ih, ow = iw;   /* stride 1, SAME */
    const unsigned icreg = rocket_rk3576_pad_ic(ic);
    const unsigned slice = 32u * icreg * k * k;
    const unsigned terms = ic * k * k;
    unsigned divisor = 1, mul, shift, c, i, y, x, kh, kw, first = oc;
    int8_t *in = malloc((size_t)ic * ih * iw), *W = malloc((size_t)oc * ic * k * k);
    int8_t *out = malloc((size_t)oc * oh * ow);
    int32_t *bias = malloc((size_t)oc * sizeof(int32_t));
    rocket_conv2d_desc d;
    uint64_t u0, w0, u1, w1;
    long k0, k1;
    size_t wrong = 0;
    double t0, ms;
    int rc, tasks;
    unsigned seed = 0x9E3779B9u ^ (ic * 31u + oc * 17u + iw * 7u + ih * 3u + k);
    unsigned g0 = rocket_rk3576_weight_phase_groups(slice, 0);
    unsigned g1 = rocket_rk3576_weight_phase_groups(slice, 1024);

    if (!in || !W || !out || !bias) { fprintf(stderr, "oom\n"); return 1; }
    while ((double)divisor < 2.0 * sqrt((double)terms)) divisor *= 2;
    requant_params(1.0f / (float)divisor, &mul, &shift);
    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                in[((size_t)c * ih + y) * iw + x] = (int8_t)((int)((c * 7 + y * 13 + x * 3) % 61) - 30);
    for (c = 0; c < oc * ic * k * k; c++) {
        seed = seed * 1103515245u + 12345u;
        W[c] = (int8_t)((int)((seed >> 16) % 17u) - 8);
    }
    for (c = 0; c < oc; c++) bias[c] = (int32_t)((int)c - (int)oc / 2) * 8;

    memset(&d, 0, sizeof d);
    d.ic = ic; d.ih = ih; d.iw = iw; d.oc = oc; d.kh = k; d.kw = k;
    d.stride_y = 1; d.stride_x = 1; d.pad_top = pad; d.pad_left = pad;
    d.dil_y = 1; d.dil_x = 1;
    tasks = (int)rocket_rk3576_max_task_rows(iw, icreg, rocket_rk3576_pad_oc(oc), k, k, 0);
    {
        unsigned w = (unsigned)tasks < ih ? (unsigned)tasks : ih, f = ~0u;
        if (rocket_rk3576_cbuf_f(iw, icreg, w, rocket_rk3576_pad_oc(oc), k, k, 0, &f) == 0)
            printf("   planner: %d rows a task, a %u-row window programs F %u\n", tasks, w, f);
    }

    printf("ic %u (icreg %u) oc %u plane %ux%u k %u: slice %u B (%.1f KiB), %u groups; "
           "rule allows %d groups at F0 and %d at F1024 (-1 = all)\n",
           ic, icreg, oc, ih, iw, k, slice, slice / 1024.0, (oc + 31u) / 32u,
           g0 == UINT32_MAX ? -1 : (int)g0, g1 == UINT32_MAX ? -1 : (int)g1);

    rocket_rk3576_retired_counts(&u0, &w0);
    k0 = klog_retired();
    t0 = now_ms();
    rc = rocket_conv2d_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f, (float)divisor,
                                   0, 0, 0, out);
    ms = now_ms() - t0;
    rocket_rk3576_retired_counts(&u1, &w1);
    k1 = klog_retired();

    if (rc == 0) {
        for (c = 0; c < oc; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    int64_t acc = bias[c];
                    for (kh = 0; kh < k; kh++)
                        for (kw = 0; kw < k; kw++) {
                            int iy = (int)(y + kh) - (int)pad, ix = (int)(x + kw) - (int)pad;
                            if (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw) continue;
                            for (i = 0; i < ic; i++)
                                acc += (int64_t)in[((size_t)i * ih + iy) * iw + ix] *
                                       W[(((size_t)c * ic + i) * k + kh) * k + kw];
                        }
                    if (out[((size_t)c * oh + y) * ow + x] != requant_apply_zp(acc, mul, shift, 0)) {
                        wrong++;
                        if (c < first) first = c;
                    }
                }
    }
    printf("   rc %d  %.1f ms  retired: %llu unwritten, %llu written, kernel %ld  %s",
           rc, ms, (unsigned long long)(u1 - u0), (unsigned long long)(w1 - w0),
           (k0 >= 0 && k1 >= 0) ? k1 - k0 : -1L,
           rc ? "REFUSED" : wrong ? "WRONG" : "exact");
    if (rc == 0 && wrong) printf(" (%zu of %u, first channel %u)", wrong, oc * oh * ow, first);
    printf("\n");
    fflush(stdout);
    free(in); free(W); free(out); free(bias);
    return 0;
}

/* The phase rule's groups at F, printed as channels at a group of `gch` channels, or "all". */
static void print_rule(const char *tag, unsigned slice, unsigned f, unsigned gch)
{
    unsigned g = rocket_rk3576_weight_phase_groups(slice, f);
    if (g == UINT32_MAX) printf("%s all", tag);
    else printf("%s %u ch", tag, g * gch);
}

static int run_fp16(int fd, unsigned ic, unsigned oc, unsigned ih, unsigned iw, unsigned k)
{
    const unsigned pad = k / 2, oh = ih, ow = iw;   /* stride 1, SAME */
    const unsigned s16 = 512u * k * k;              /* one 16-channel float group, ic 16 */
    unsigned c, i, y, x, kh, kw, first = oc, f = ~0u, big = 0;
    _Float16 *in = malloc((size_t)ic * ih * iw * sizeof(_Float16));
    _Float16 *W = malloc((size_t)oc * ic * k * k * sizeof(_Float16));
    _Float16 *out = malloc((size_t)oc * oh * ow * sizeof(_Float16));
    rocket_conv2d_desc d;
    uint64_t u0, w0, u1, w1;
    long k0, k1;
    size_t wrong = 0, n = (size_t)oc * oh * ow;
    double t0, ms;
    int rc;
    unsigned seed = 0x85EBCA6Bu ^ (ic * 31u + oc * 17u + iw * 7u + ih * 3u + k);

    if (!in || !W || !out) { fprintf(stderr, "oom\n"); return 1; }
    for (c = 0; c < ic; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                in[((size_t)c * ih + y) * iw + x] =
                    (_Float16)((int)((c * 7 + y * 13 + x * 3) % 5) - 2);
    for (c = 0; c < oc * ic * k * k; c++) {
        seed = seed * 1103515245u + 12345u;
        W[c] = (_Float16)((int)((seed >> 16) % 3u) - 1);
    }
    /* Poison the caller's buffer so an entry that returned 0 without writing reads wrong. */
    for (i = 0; i < n; i++) out[i] = (_Float16)12345.0f;

    memset(&d, 0, sizeof d);
    d.ic = ic; d.ih = ih; d.iw = iw; d.oc = oc; d.kh = k; d.kw = k;
    d.stride_y = 1; d.stride_x = 1; d.pad_top = pad; d.pad_left = pad;
    d.dil_y = 1; d.dil_x = 1;
    if (rocket_rk3576_cbuf_f_prec(iw, 16u, ih, (oc + 15u) & ~15u, k, k, 0,
                                  precision_float16, &f) != 0)
        f = ~0u;
    printf("fp16 ic %u oc %u plane %ux%u k %u: 16-ch group slice %u B (%.1f KiB), planner F %d,"
           " %u tasks (one per 16 ic);", ic, oc, ih, iw, k, s16, s16 / 1024.0,
           f == ~0u ? -1 : (int)f, (ic + 15u) / 16u);
    if (f != ~0u) {
        print_rule(" rule@F, 16-ch groups:", s16, f, 16u);
        print_rule(", 32-ch groups:", 2u * s16, f, 32u);
    }
    printf("\n");

    rocket_rk3576_retired_counts(&u0, &w0);
    k0 = klog_retired();
    t0 = now_ms();
    rc = rocket_conv2d_fp16_rk3576(fd, &d, in, W, out);
    ms = now_ms() - t0;
    rocket_rk3576_retired_counts(&u1, &w1);
    k1 = klog_retired();

    if (rc == 0) {
        for (c = 0; c < oc; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    long acc = 0;
                    for (kh = 0; kh < k; kh++)
                        for (kw = 0; kw < k; kw++) {
                            int iy = (int)(y + kh) - (int)pad, ix = (int)(x + kw) - (int)pad;
                            if (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw) continue;
                            for (i = 0; i < ic; i++)
                                acc += (long)in[((size_t)i * ih + iy) * iw + ix] *
                                       (long)W[(((size_t)c * ic + i) * k + kh) * k + kw];
                        }
                    if (acc > 2048 || acc < -2048) big++;
                    if (out[((size_t)c * oh + y) * ow + x] != (_Float16)(float)acc) {
                        wrong++;
                        if (c < first) first = c;
                    }
                }
    }
    printf("   rc %d  %.1f ms  retired: %llu unwritten, %llu written, kernel %ld  %s",
           rc, ms, (unsigned long long)(u1 - u0), (unsigned long long)(w1 - w0),
           (k0 >= 0 && k1 >= 0) ? k1 - k0 : -1L,
           rc ? "REFUSED" : wrong ? "WRONG" : "exact");
    if (rc == 0 && wrong) printf(" (%zu of %zu, first channel %u)", wrong, n, first);
    if (big) printf(" [%u sums past 2048]", big);
    printf("\n");
    /* ROCKET_BAND_MAP=1: for each output channel, which reference channel's WHOLE plane it
     * holds (-1 none; P the caller's poison; Z all zero), so a wrong surface is read as a
     * function rather than a count. */
    if (rc == 0 && wrong && getenv("ROCKET_BAND_MAP")) {
        size_t px = (size_t)oh * ow, p2;
        float *ref = malloc((size_t)oc * px * sizeof(float));
        if (ref) {
            for (c = 0; c < oc; c++)
                for (y = 0; y < oh; y++)
                    for (x = 0; x < ow; x++) {
                        long acc = 0;
                        for (kh = 0; kh < k; kh++)
                            for (kw = 0; kw < k; kw++) {
                                int iy = (int)(y + kh) - (int)pad, ix = (int)(x + kw) - (int)pad;
                                if (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw) continue;
                                for (i = 0; i < ic; i++)
                                    acc += (long)in[((size_t)i * ih + iy) * iw + ix] *
                                           (long)W[(((size_t)c * ic + i) * k + kh) * k + kw];
                            }
                        ref[(size_t)c * px + (size_t)y * ow + x] = (float)(_Float16)(float)acc;
                    }
            float cval = 0.0f;
            unsigned nconst = 0;
            printf("   map (out channel -> reference channel whose plane it holds; "
                   "C a constant plane):");
            for (c = 0; c < oc; c++) {
                const _Float16 *o = out + (size_t)c * px;
                int hit = -1, poison = 1, zero = 1, cst = 1;
                unsigned e2;
                for (p2 = 0; p2 < px; p2++) {
                    if ((float)o[p2] != 12345.0f) poison = 0;
                    if ((float)o[p2] != 0.0f) zero = 0;
                    if ((float)o[p2] != (float)o[0]) cst = 0;
                }
                if (cst && !poison && !zero) {
                    if (!nconst || (float)o[0] != cval) { cval = (float)o[0]; }
                    nconst++;
                    if (c % 16 == 0) printf("\n     %4u:", c);
                    printf("    C");
                    continue;
                }
                for (e2 = 0; e2 < oc && hit < 0; e2++) {
                    const float *r = ref + (size_t)e2 * px;
                    for (p2 = 0; p2 < px; p2++) if ((float)o[p2] != r[p2]) break;
                    if (p2 == px) hit = (int)e2;
                }
                if (c % 16 == 0) printf("\n     %4u:", c);
                if (poison) printf("    P");
                else if (zero) printf("    Z");
                else printf(" %4d", hit);
            }
            /* One code per channel: '.' exact, 'C' a whole constant (unwritten) plane, 't'
             * wrong only where it still holds the unwritten value (a tail the DPU had not
             * landed), 'x' wrong otherwise. The unwritten value is the sentinel summed over
             * the input-channel slices, as the entry's host accumulate leaves it. */
            {
                const _Float16 sent = ((union { uint16_t u; _Float16 h; }){ .u = 0xA5A5u }).h;
                const float unw = (float)(_Float16)((float)sent * (float)((ic + 15u) / 16u));
                printf("\n   chan:");
                for (c = 0; c < oc; c++) {
                    const _Float16 *o = out + (size_t)c * px;
                    const float *r = ref + (size_t)c * px;
                    int bad = 0, other = 0, cst = 1;
                    for (p2 = 0; p2 < px; p2++) {
                        if ((float)o[p2] != (float)o[0]) cst = 0;
                        if ((float)o[p2] != r[p2]) {
                            bad = 1;
                            if ((float)o[p2] != unw) other = 1;
                        }
                    }
                    if (c % 16 == 0) printf(" ");
                    putchar(!bad ? '.' : cst ? 'C' : other ? 'x' : 't');
                }
            }
            if (nconst) printf("\n   %u constant planes, last value %g (the sentinel 0xA5A5 is %g)",
                               nconst, (double)cval,
                               (double)((union { uint16_t u; _Float16 h; }){ .u = 0xA5A5u }).h);
            /* Where inside the first wrong channel: wrong pixels per output row, and the
             * first few wrong positions with both values. */
            if (first < oc) {
                const _Float16 *o = out + (size_t)first * px;
                const float *r = ref + (size_t)first * px;
                unsigned shown = 0, rows_bad = 0;
                printf("\n   ch %u wrong per row:", first);
                for (y = 0; y < oh; y++) {
                    unsigned nb = 0;
                    for (x = 0; x < ow; x++) if ((float)o[y * ow + x] != r[y * ow + x]) nb++;
                    if (nb) { rows_bad++; if (rows_bad <= 24) printf(" %u:%u", y, nb); }
                }
                printf(" (%u rows)\n   first wrong:", rows_bad);
                for (p2 = 0; p2 < px && shown < 6; p2++)
                    if ((float)o[p2] != r[p2]) {
                        printf(" (%zu,%zu) got %g ref %g;", p2 / ow, p2 % ow, (double)o[p2],
                               (double)r[p2]);
                        shown++;
                    }
            }
            printf("\n   sample ch %u px 0..3: got %g %g %g %g ref %g %g %g %g\n",
                   first < oc ? first : 0,
                   (double)out[(size_t)first * px + 0], (double)out[(size_t)first * px + 1],
                   (double)out[(size_t)first * px + 2], (double)out[(size_t)first * px + 3],
                   (double)ref[(size_t)first * px + 0], (double)ref[(size_t)first * px + 1],
                   (double)ref[(size_t)first * px + 2], (double)ref[(size_t)first * px + 3]);
            free(ref);
        }
    }
    fflush(stdout);
    free(in); free(W); free(out);
    return 0;
}

static int run_dw(int fd, unsigned ch, unsigned ih, unsigned iw, unsigned k)
{
    const unsigned pad = k / 2, oh = ih, ow = iw;   /* stride 1, SAME */
    const unsigned cw = (ch + 15u) & ~15u;
    const unsigned charged = ch * k * k * 2u, cube = cw * k * k * 2u;
    unsigned divisor = 1, mul, shift, c, y, x, kh, kw, first = ch, f = ~0u, rows;
    int8_t *in = malloc((size_t)ch * ih * iw), *W = malloc((size_t)ch * k * k);
    int8_t *out = malloc((size_t)ch * oh * ow);
    int32_t *bias = malloc((size_t)ch * sizeof(int32_t));
    rocket_conv2d_desc d;
    uint64_t u0, w0, u1, w1;
    long k0, k1;
    size_t wrong = 0;
    double t0, ms;
    int rc;
    unsigned seed = 0xC2B2AE35u ^ (ch * 17u + iw * 7u + ih * 3u + k);

    if (!in || !W || !out || !bias) { fprintf(stderr, "oom\n"); return 1; }
    while ((double)divisor < 2.0 * sqrt((double)(k * k))) divisor *= 2;
    requant_params(1.0f / (float)divisor, &mul, &shift);
    for (c = 0; c < ch; c++)
        for (y = 0; y < ih; y++)
            for (x = 0; x < iw; x++)
                in[((size_t)c * ih + y) * iw + x] = (int8_t)((int)((c * 7 + y * 13 + x * 3) % 61) - 30);
    for (c = 0; c < ch * k * k; c++) {
        seed = seed * 1103515245u + 12345u;
        W[c] = (int8_t)((int)((seed >> 16) % 17u) - 8);
    }
    for (c = 0; c < ch; c++) bias[c] = (int32_t)((int)(c % 97u) - 48) * 8;
    memset(out, 0x5A, (size_t)ch * oh * ow);

    memset(&d, 0, sizeof d);
    d.ic = ch; d.ih = ih; d.iw = iw; d.oc = ch; d.kh = k; d.kw = k; d.depthwise = 1;
    d.stride_y = 1; d.stride_x = 1; d.pad_top = pad; d.pad_left = pad;
    d.dil_y = 1; d.dil_x = 1;
    rows = rocket_rk3576_max_task_rows(iw, ch, ch, k, k, 1);
    {
        unsigned w = rows < ih ? rows : ih;
        if (!w || rocket_rk3576_cbuf_f(iw, ch, w, ch, k, k, 1, &f) != 0) f = ~0u;
    }
    printf("dw c %u (cube c %u) plane %ux%u k %u: planner charges %u B, cube %u B, "
           "planner %u rows a task, F %d; 64 KiB area units at F: %d\n",
           ch, cw, ih, iw, k, charged, cube, rows, f == ~0u ? -1 : (int)f,
           f == ~0u ? -1 : (int)((3072u - (f < 3072u ? f : 3072u)) / 1024u));

    rocket_rk3576_retired_counts(&u0, &w0);
    k0 = klog_retired();
    t0 = now_ms();
    rc = rocket_conv2d_dw_int8_rk3576(fd, &d, in, W, bias, 1.0f, 1.0f, (float)divisor,
                                      0, 0, 0, out);
    ms = now_ms() - t0;
    rocket_rk3576_retired_counts(&u1, &w1);
    k1 = klog_retired();

    if (rc == 0) {
        for (c = 0; c < ch; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    int64_t acc = bias[c];
                    for (kh = 0; kh < k; kh++)
                        for (kw = 0; kw < k; kw++) {
                            int iy = (int)(y + kh) - (int)pad, ix = (int)(x + kw) - (int)pad;
                            if (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw) continue;
                            acc += (int64_t)in[((size_t)c * ih + iy) * iw + ix] *
                                   W[((size_t)c * k + kh) * k + kw];
                        }
                    if (out[((size_t)c * oh + y) * ow + x] != requant_apply_zp(acc, mul, shift, 0)) {
                        wrong++;
                        if (c < first) first = c;
                    }
                }
    }
    printf("   rc %d  %.1f ms  retired: %llu unwritten, %llu written, kernel %ld  %s",
           rc, ms, (unsigned long long)(u1 - u0), (unsigned long long)(w1 - w0),
           (k0 >= 0 && k1 >= 0) ? k1 - k0 : -1L,
           rc ? "REFUSED" : wrong ? "WRONG" : "exact");
    if (rc == 0 && wrong) printf(" (%zu of %u, first channel %u)", wrong, ch * oh * ow, first);
    printf("\n");
    /* ROCKET_BAND_MAP=1: where the wrong elements sit (channel range, rows, and the first
     * few with both values). 0x5A is the value the caller's buffer was filled with. */
    if (rc == 0 && wrong && getenv("ROCKET_BAND_MAP")) {
        unsigned cmin = ch, cmax = 0, shown = 0, rowbad[64] = {0}, nfill = 0;
        size_t e2 = 0;
        for (c = 0; c < ch; c++)
            for (y = 0; y < oh; y++)
                for (x = 0; x < ow; x++) {
                    int64_t acc = bias[c];
                    int8_t got = out[((size_t)c * oh + y) * ow + x], want;
                    for (kh = 0; kh < k; kh++)
                        for (kw = 0; kw < k; kw++) {
                            int iy = (int)(y + kh) - (int)pad, ix = (int)(x + kw) - (int)pad;
                            if (iy < 0 || iy >= (int)ih || ix < 0 || ix >= (int)iw) continue;
                            acc += (int64_t)in[((size_t)c * ih + iy) * iw + ix] *
                                   W[((size_t)c * k + kh) * k + kw];
                        }
                    want = requant_apply_zp(acc, mul, shift, 0);
                    if (got == want) continue;
                    e2++;
                    if (c < cmin) cmin = c;
                    if (c > cmax) cmax = c;
                    if (y < 64) rowbad[y]++;
                    if ((uint8_t)got == 0x5Au || (uint8_t)got == 0xA5u) nfill++;
                    if (shown < 6) {
                        printf("   (c %u y %u x %u) got %d want %d\n", c, y, x, got, want);
                        shown++;
                    }
                }
        printf("   %zu wrong in channels %u..%u, %u holding a fill byte; per row:", e2,
               cmin, cmax, nfill);
        for (y = 0; y < oh && y < 64; y++) if (rowbad[y]) printf(" %u:%u", y, rowbad[y]);
        printf("\n");
    }
    fflush(stdout);
    free(in); free(W); free(out); free(bias);
    return 0;
}

int main(int argc, char **argv)
{
    int fd, a;
    if (strcmp(rocket_hw_current()->name, "rk3576") != 0) {
        fprintf(stderr, "not an RK3576\n");
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no NPU\n"); return 2; }
    for (a = 1; a < argc; a++) {
        unsigned v[5];
        if (argv[a][0] == 'f' && argv[a][1] == ':') {
            if (sscanf(argv[a] + 2, "%u:%u:%u:%u:%u", &v[0], &v[1], &v[2], &v[3], &v[4]) != 5)
                fprintf(stderr, "bad arm %s\n", argv[a]);
            else
                run_fp16(fd, v[0], v[1], v[2], v[3], v[4]);
            continue;
        }
        if (argv[a][0] == 'd' && argv[a][1] == ':') {
            if (sscanf(argv[a] + 2, "%u:%u:%u:%u", &v[0], &v[1], &v[2], &v[3]) != 4)
                fprintf(stderr, "bad arm %s\n", argv[a]);
            else
                run_dw(fd, v[0], v[1], v[2], v[3]);
            continue;
        }
        if (sscanf(argv[a], "%u:%u:%u:%u:%u", &v[0], &v[1], &v[2], &v[3], &v[4]) != 5) {
            fprintf(stderr, "bad arm %s\n", argv[a]);
            continue;
        }
        run(fd, v[0], v[1], v[2], v[3], v[4]);
    }
    rocket_close(fd);
    return 0;
}
