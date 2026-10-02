// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_stallk_map.c — where does a stalled int8 matmul job stop, and what moves it?
 *
 * A probe, not a gate. At some K an
 * int8 matmul job raises no completion: `rocket` retires it at the 125 ms backstop and the
 * surface is exact up to a fixed output channel and unwritten past it. The library entry
 * re-plans such a tile narrower, which hides the boundary; this drives the same emitter one
 * row task at a time, with no retry and no narrowing, and makes the boundary an OUTPUT.
 *
 * Each ARM is one matmul laid out as a 1x1 convolution, exactly as the library lays it out
 * (the same feature cube, weight cube, coefficient packing and per-task offsets), with the
 * plane and the rows per task chosen by the arm rather than by the planner. Every row task
 * is one job. Per job it prints:
 *
 *   ms        submit to fence; a job at or past the backstop was retired
 *   F         the feature allowance the emitter PROGRAMMED (read back from CNA 0x1040)
 *   exact     leading 32-channel groups that are exact on every pixel of the task
 *   first     the first output channel with any wrong pixel (N if none)
 *   in-group  for the first wrong group: pixels exact / pixels unwritten in each 16-lane half
 *   past      of the channels after that group: elements unwritten (still the sentinel)
 *             against elements written wrong
 *   rel/abs   the first stalled channel two models predict (see below)
 *
 * THE MODELS. S = 32*K bytes is one output-channel group's weight slice, and
 * R_w = 64 KiB * floor((3072 - F) / 1024) is the CBUF weight area the feature allowance
 * leaves, in whole 64 KiB units. Group g completes iff (phase_g mod 64 KiB) + S <= R_w, with
 *   rel  phase_g = g*S, counted from the weight base register
 *   abs  phase_g = IOVA of group g, the absolute address
 * The arm's weight base can be offset into its BO (`woff`), which moves `abs` and not `rel`.
 * `rel` is the rule the library plans by (rocket_rk3576_weight_phase_groups()).
 *
 * Scored against a host int64 reference through tests/requant_model.h, the model the gate
 * list uses. After a retired or wrong job the NPU is power-cycled before the next one.
 *
 * Usage: rk3576_stallk_map ARM...
 *   ARM = K:N:iw:ih:rows[:woff[:tasks]]
 *     K     contraction depth (multiple of 32)       N  output channels (multiple of 32)
 *     iw    plane width; M = iw*ih                   rows  output rows per task (Mt = iw*rows)
 *     woff  weight base offset into its BO, bytes (default 0)
 *     tasks row tasks to run, from the top (default all)
 *   ROCKET_RK3576_CBUF_RUNGS (the library's knob) restricts the rungs the emitter may pick,
 *   which is how an arm forces F at a fixed Mt. It is read once per process.
 * Exit: 0 ran, 2 no NPU / not an RK3576, 1 a setup failure.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "npu_matmul.h"
#include "npu_regcmd_rk3576.h"
#include "rocket_rk3576_internal.h"
#include "test_fill.h"
#include "requant_model.h"

#define C2 16u
#define SENT ROCKET_RK3576_SENTINEL_BYTE
#define BACKSTOP_NS 125000000ull

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

/* The first group g < groups for which (phase0 + g*S) mod 64 KiB + S > R_w, or -1. */
static int model_stall(uint64_t phase0, unsigned S, unsigned rw, unsigned groups)
{
    unsigned g;
    for (g = 0; g < groups; g++) {
        uint64_t start = (phase0 + (uint64_t)g * S) % 65536u;
        if (start + S > rw) return (int)g;
    }
    return -1;
}

struct arm {
    unsigned K, N, iw, ih, rows, woff, tasks;
};

static int parse_arm(const char *s, struct arm *a)
{
    unsigned v[7] = { 0, 0, 0, 0, 0, 0, 0 };
    int n = 0;
    char *end;
    memset(a, 0, sizeof *a);
    while (*s && n < 7) {
        v[n++] = (unsigned)strtoul(s, &end, 0);
        if (end == s) return -1;
        s = end;
        if (*s == ':') s++;
    }
    if (n < 5) return -1;
    a->K = v[0]; a->N = v[1]; a->iw = v[2]; a->ih = v[3]; a->rows = v[4];
    a->woff = n > 5 ? v[5] : 0u;
    a->tasks = n > 6 ? v[6] : 0u;
    if (!a->K || a->K % 32u || !a->N || a->N % 32u || !a->iw || !a->ih || !a->rows)
        return -1;
    return 0;
}

static int run_arm(int fd, const struct arm *a)
{
    const unsigned K = a->K, N = a->N, iw = a->iw, ih = a->ih;
    const unsigned M = iw * ih, nreg = rocket_rk3576_pad_oc(N);
    const unsigned ntask_all = (ih + a->rows - 1u) / a->rows;
    const unsigned ntask = a->tasks && a->tasks < ntask_all ? a->tasks : ntask_all;
    const unsigned nK1 = (K + 31u) / 32u, S = 32u * K;
    const float scale = 1.0f / 2048.0f;
    const unsigned surf = rocket_rk3576_out_surf_elems(iw, ih, 0);
    const size_t in_bytes = (size_t)(K / C2) * ih * iw * C2;
    const size_t w_bytes = (size_t)(nreg / 32u) * nK1 * 1024u;
    const size_t coeff_bytes = rocket_rk3576_coeff_bytes(nreg);
    const size_t obytes = (size_t)(nreg / C2) * surf * C2;
    const unsigned mrows = ntask == ntask_all ? M : (ntask * a->rows < ih ? ntask * a->rows : ih) * iw;
    int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K);
    int32_t *bias = calloc(nreg, sizeof *bias);
    int8_t *want = malloc((size_t)mrows * N);
    uint64_t ops[RK3576_CONV_TASK_OPS];
    rocket_bo in = {0}, w = {0}, co = {0}, out = {0}, rc = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p;
    unsigned t, m, n, k;
    int ret = 1, retired_jobs = 0, unwritten_jobs = 0;
    long k0, k1;

    if (!A || !B || !bias || !want) { fprintf(stderr, "oom\n"); goto done; }
    tf_fill_i8(A, (size_t)M * K, 21, -16, 16);
    tf_fill_i8(B, (size_t)N * K, 22, -16, 16);
    tf_fill_i32(bias, N, 23, -4096, 4096);
    /* The host reference, for the rows this arm runs. int32 is wide enough: |acc| is at
     * most K*256 + 4096. */
    for (m = 0; m < mrows; m++) {
        const int8_t *ar = A + (size_t)m * K;
        for (n = 0; n < N; n++) {
            const int8_t *br = B + (size_t)n * K;
            int32_t acc = 0;
            for (k = 0; k < K; k++) acc += (int32_t)ar[k] * br[k];
            want[(size_t)m * N + n] = (int8_t)requant_scale((int64_t)acc + bias[n], scale);
        }
    }

    if (rocket_bo_alloc32(fd, in_bytes, &in) || rocket_bo_alloc32(fd, w_bytes + a->woff, &w) ||
        rocket_bo_alloc32(fd, coeff_bytes, &co) || rocket_bo_alloc32(fd, obytes, &out) ||
        rocket_bo_alloc32(fd, sizeof ops, &rc)) {
        fprintf(stderr, "bo alloc failed\n");
        goto done;
    }
    rocket_bo_prep(fd, &in, 1, 0);
    memset(in.ptr, 0, in_bytes);
    {
        size_t plane = (size_t)ih * iw * C2;
        unsigned k1;
        for (m = 0; m < M; m++)
            for (k1 = 0; k1 < K / C2; k1++)
                memcpy((int8_t *)in.ptr + (size_t)k1 * plane + (size_t)C2 * m,
                       A + (size_t)m * K + (size_t)k1 * C2, C2);
    }
    rocket_bo_fini(fd, &in);
    rocket_bo_prep(fd, &w, 1, 0);
    memset(w.ptr, 0, w_bytes + a->woff);
    {
        int8_t *cube = (int8_t *)w.ptr + a->woff;
        unsigned k1;
        for (n = 0; n < N; n++)
            for (k1 = 0; k1 < nK1; k1++)
                memcpy(cube + (size_t)(n / 32u) * nK1 * 1024u + (size_t)(n % 32u) * 32u
                            + (size_t)k1 * 1024u,
                       B + (size_t)n * K + (size_t)k1 * 32u, 32);
    }
    rocket_bo_fini(fd, &w);
    rocket_bo_prep(fd, &co, 1, 0);
    rocket_rk3576_pack_coeff(co.ptr, coeff_bytes, bias, nreg);
    rocket_bo_fini(fd, &co);

    memset(&p, 0, sizeof p);
    p.ic = (uint16_t)K; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.oc = (uint16_t)nreg; p.oh = (uint16_t)ih; p.ow = (uint16_t)iw;
    p.kh = 1; p.kw = 1; p.stride_y = 1; p.stride_x = 1;
    p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)ih;
    p.int8_out = 1;
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f / scale;
    p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
    p.tasks = ops;
    p.input_dma = in.dma_address;
    p.weights_dma = w.dma_address + a->woff;
    p.bias_dma = co.dma_address;
    p.output_dma = out.dma_address;
    in_h[0] = in.handle; in_h[1] = w.handle; in_h[2] = co.handle; in_h[3] = rc.handle;
    out_h[0] = out.handle;

    printf("arm K %u N %u plane %ux%u (M %u) rows %u (Mt %u) woff %u  S %u B  w iova 0x%llx "
           "(base phase %llu B)  %u of %u tasks\n",
           K, N, iw, ih, M, a->rows, iw * a->rows, a->woff, S,
           (unsigned long long)p.weights_dma,
           (unsigned long long)(p.weights_dma % 65536u), ntask, ntask_all);
    fflush(stdout);

    rocket_rk3576_power_idle();
    k0 = klog_retired();
    for (t = 0; t < ntask; t++) {
      unsigned attempt;
      for (attempt = 0; attempt < 3; attempt++) {
        const unsigned oy0 = t * a->rows;
        const unsigned oh = ih - oy0 < a->rows ? ih - oy0 : a->rows;
        conv_params_t q = p;
        unsigned f = ~0u, i, g, first = N, gexact = 0, grp_bad = ~0u;
        uint64_t tj, ns;
        size_t unw_past = 0, wrong_past = 0, wrong_all = 0;
        unsigned half_exact[2] = { 0, 0 }, half_unw[2] = { 0, 0 };
        int rel, abs_;
        size_t wrote = 0;
        const unsigned char *o;

        q.ih = (uint16_t)oh; q.oh = (uint16_t)oh;
        q.input_dma = p.input_dma + (uint64_t)oy0 * iw * C2;
        q.output_dma = p.output_dma + (uint64_t)oy0 * iw * C2;
        q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
        if (gen_conv2d_int8_rk3576(&q) != 0) {
            printf("  task %u: the emitter refused this task\n", t);
            ret = 0;
            goto done;
        }
        for (i = 0; i < q.task_count; i++)
            if ((ops[i] & 0xFFFFu) == 0x1040u) f = (unsigned)((ops[i] >> 16) >> 16) & 0xFFFu;

        rocket_bo_prep(fd, &out, 1, 0);
        memset(out.ptr, SENT, obytes);
        rocket_bo_fini(fd, &out);
        rocket_bo_prep(fd, &rc, 1, 0);
        memcpy(rc.ptr, ops, q.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc);
        tj = rocket_rk3576_job_clock();
        if (rocket_submit_matmul(fd, &rc, q.task_count, in_h, 4, out_h, 1, 4000) != 0) {
            printf("  task %u: submit failed\n", t);
            goto done;
        }
        if (rocket_bo_prep(fd, &out, 0, 2000000000ull) < 0) {
            printf("  task %u: fence wait failed\n", t);
            goto done;
        }
        ns = rocket_rk3576_job_ns(tj);
        o = (const unsigned char *)out.ptr;
        for (n = 0; n < nreg / C2; n++)
            for (m = oy0 * iw; m < (oy0 + oh) * iw; m++) {
                const unsigned char *atom = o + (size_t)n * surf * C2 + (size_t)C2 * m;
                for (i = 0; i < C2; i++) wrote += atom[i] != SENT;
            }

        /* Score by group, in channel order. */
        for (g = 0; g < nreg / 32u && g * 32u < N; g++) {
            size_t bad = 0;
            for (n = g * 32u; n < g * 32u + 32u && n < N; n++)
                for (m = oy0 * iw; m < (oy0 + oh) * iw; m++) {
                    unsigned char got = o[(size_t)(n / C2) * surf * C2 + (size_t)C2 * m + n % C2];
                    if ((int8_t)got != want[(size_t)m * N + n]) {
                        bad++;
                        if (n < first) first = n;
                        if (grp_bad != ~0u && g > grp_bad) {
                            if (got == SENT) unw_past++; else wrong_past++;
                        }
                    }
                }
            wrong_all += bad;
            if (bad && grp_bad == ~0u) grp_bad = g;
            if (!bad && grp_bad == ~0u) gexact++;
        }
        if (grp_bad != ~0u) {
            unsigned hh;
            for (hh = 0; hh < 2; hh++)
                for (m = oy0 * iw; m < (oy0 + oh) * iw; m++) {
                    int ex = 1, un = 1;
                    for (n = grp_bad * 32u + hh * 16u; n < grp_bad * 32u + hh * 16u + 16u && n < N; n++) {
                        unsigned char got = o[(size_t)(n / C2) * surf * C2 + (size_t)C2 * m + n % C2];
                        if ((int8_t)got != want[(size_t)m * N + n]) ex = 0;
                        if (got != SENT) un = 0;
                    }
                    half_exact[hh] += ex;
                    half_unw[hh] += un;
                }
        }
        rocket_bo_fini(fd, &out);
        rel = f == ~0u ? -2 : model_stall(0, S, (3072u - f) / 1024u * 65536u, nreg / 32u);
        abs_ = f == ~0u ? -2 : model_stall(p.weights_dma, S, (3072u - f) / 1024u * 65536u,
                                           nreg / 32u);
        printf("  task %2u rows %u-%u  %7.1f ms %s  F %4u  exact %2u grp  first %4u",
               t, oy0, oy0 + oh - 1, ns / 1e6, ns >= BACKSTOP_NS ? "RETIRED" : "done   ",
               f, gexact, first);
        if (grp_bad != ~0u)
            printf("  in-group px exact %u/%u unw %u/%u of %u  past unw %zu wrong %zu",
                   half_exact[0], half_exact[1], half_unw[0], half_unw[1], oh * iw,
                   unw_past, wrong_past);
        printf("  wrong %zu  rel %d abs %d%s\n", wrong_all,
               rel < 0 ? -1 : rel * 32, abs_ < 0 ? -1 : abs_ * 32,
               !wrote ? "  UNWRITTEN"
               : (rel < 0 ? (first == N) : (first == (unsigned)rel * 32u)) ? "  REL-OK"
                                                                          : "  REL-MISS");
        fflush(stdout);
        if (ns >= BACKSTOP_NS) retired_jobs++;
        if (!wrote) unwritten_jobs++;
        if (ns >= BACKSTOP_NS || wrong_all) rocket_rk3576_power_idle();
        /* A job that wrote NOTHING is not a stall at channel 0 — the rule has no such
         * case below R_w — it is the carried-over unwritten retirement, so it is
         * redone after the power cycle and counted apart. */
        if (wrote) break;
      }
    }
    k1 = klog_retired();
    printf("  arm: %d job(s) timed as retired (%d wrote nothing), kernel 'retiring it' %ld\n",
           retired_jobs, unwritten_jobs, (k0 >= 0 && k1 >= 0) ? k1 - k0 : -1L);
    ret = 0;

done:
    if (rc.ptr) rocket_bo_free(fd, &rc);
    if (out.ptr) rocket_bo_free(fd, &out);
    if (co.ptr) rocket_bo_free(fd, &co);
    if (w.ptr) rocket_bo_free(fd, &w);
    if (in.ptr) rocket_bo_free(fd, &in);
    free(A); free(B); free(bias); free(want);
    return ret;
}

int main(int argc, char **argv)
{
    int fd, i, rc = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s K:N:iw:ih:rows[:woff[:tasks]]...\n", argv[0]);
        return 1;
    }
    if (strcmp(rocket_hw_current()->name, "rk3576") != 0) {
        fprintf(stderr, "not an RK3576 (profile %s)\n", rocket_hw_current()->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no NPU\n"); return 2; }
    for (i = 1; i < argc; i++) {
        struct arm a;
        if (parse_arm(argv[i], &a)) {
            fprintf(stderr, "bad arm '%s'\n", argv[i]);
            rc = 1;
            continue;
        }
        if (run_arm(fd, &a)) rc = 1;
    }
    rocket_close(fd);
    return rc;
}
