// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_matmul_probe.c — is there a matmul on this part, and what shape is it?
 *
 * A matmul is a 1x1 convolution: the A rows are the conv's spatial pixels, K is the
 * input-channel axis, N the output-channel axis. The RK3576 already computes 1x1
 * convolutions bit-exactly at int8 and fp16, so nothing about the register program is
 * in question here. What this probe reads off the part is the ENVELOPE the userspace
 * layer has to plan inside:
 *
 *   `shape`  the mapping itself, against a CPU model, over a (M,K,N) table
 *   `m`      the M axis: does a plane one or two pixels tall compute, and does the
 *            factorization of M into (iw,ih) change the answer
 *   `cost`   what one submit costs at int8 and at fp16, at the same (M,K,N) — the
 *            precision asymmetry that decides which precision a matmul is built on
 *   `out32`  what the DPU's 32-bit writer delivers on an INTEGER program, and where
 *
 * Everything here drives the same emitter the conv gate drives; the only thing that
 * is new is the operand mapping. Run with `sudo -E` on the RK3576.
 *
 * `out32` is the register instrument as much as the measurement. `ROCKET_MP_VARIANTS`
 * replaces its built-in table with a `width:regspec;width:regspec;...` list, so a MODE —
 * several registers that are each inert until the others are right — can be driven
 * without a rebuild, and `ROCKET_MP_K/_M/_N` move the shape under it. Two things it
 * reports matter: the writer's EXTENT in words, and the decoded map, which says whether
 * the arithmetic survived whatever was driven.
 *
 * KEEP THE PACING. `ROCKET_MP_GAP_MS` defaults to 200 for a reason: an int32 job leaves
 * the next submit of any kind writing nothing until 50-100 ms have passed, and at a gap
 * of 0 even the first job in a run can come back empty. An unpaced sweep reads as a
 * datapath that does not work.
 */
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
#include "requant_model.h"

#define C2       16
#define SENTINEL 0xAA

int feature_data(int C, int H, int W, int C2_, int c, int h, int w);
int weight_conv_int8(int OCn, int ICn, int KH, int KW, int oc, int ic, int kh, int kw);

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


/* ============================================================================
 * SECTION — the mapping
 *
 * A[M,K] * B[N,K]^T = C[M,N], the gen_matmul convention: output channel n is the dot
 * product of input row m with weight row n. As a conv:
 *
 *   feature cube   channel k, pixel m at (m/iw, m%iw), C2 = 16 int8 lanes per atom
 *   weight cube    weight_conv_int8(N, K, 1, 1, n+1, k+1, 1, 1)
 *   output cube    channel n, pixel m, the same C2 = 16 surface the conv path writes
 *
 * The plane is the free parameter: any (iw, ih) with iw*ih == M carries the same M
 * rows, and the choice is a TILING one, not a correctness one — the feature budget is
 * ceil(iw*K/64) granules per row times the task's rows, so the pixels one task can
 * hold is 262144/K however the plane is cut, as long as iw*K is a whole number of
 * 64-byte granules.
 * ==========================================================================*/

/* Pick a plane for M rows at contraction K: the widest even divisor of M whose row
 * cost still leaves the row window somewhere to go. iw*K must be a multiple of 64 or
 * every row wastes a granule, and K is a multiple of 32, so an even iw is free. */
static void pick_plane(unsigned M, unsigned K, unsigned *iw_out, unsigned *ih_out)
{
    unsigned forced = (unsigned)env_int("ROCKET_MP_IW", 0);
    unsigned iw, best = 1;
    if (forced) { *iw_out = forced; *ih_out = M / forced; return; }
    /* The most pixels one task can hold, in granule terms. */
    unsigned budget_px = (unsigned)(4096ull * 64ull / K);
    if (!budget_px) budget_px = 1;
    for (iw = 1; iw <= M; iw++) {
        if (M % iw) continue;
        if (iw > budget_px) break;
        if ((iw * K) % 64) continue;      /* a partial granule per row wastes budget */
        best = iw;
    }
    *iw_out = best;
    *ih_out = M / best;
}

struct mm_result {
    int      tasks;
    int      exact, total;
    int      maxdiff;
    int      untouched, obytes;
    double   ms;
};

/* One int8 matmul, emitted as a 1x1 conv, checked against a CPU model.
 * Returns 0 exact, 1 wrong, 3 refused by the planner, <0 broken. */
static int mm_int8_ex(int fd, unsigned M, unsigned K, unsigned N,
                      unsigned iw, unsigned ih, int i32out, struct mm_result *st)
{
    unsigned kreg = rocket_rk3576_pad_ic(K), nreg = rocket_rk3576_pad_oc(N);
    unsigned kpad = (kreg + 31) / 32 * 32;
    int8_t *A = NULL, *B = NULL, *in_cube = NULL, *w_cube = NULL, *out = NULL;
    int32_t *bias = NULL;
    size_t in_bytes, w_bytes, obytes, coeff_bytes;
    unsigned surf_elems;
    rocket_bo in_bo = {0}, w_bo = {0}, b_bo = {0}, o_bo = {0}, r_bo = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p = {0};
    unsigned scale, shift_reg, seed, divisor;
    int verbose = env_int("ROCKET_MP_VERBOSE", 0);
    int gap = env_int("ROCKET_MP_GAP_MS", 200);
    int rc = -1, shown = 0;
    unsigned m, n, k;

    memset(st, 0, sizeof *st);
    if (iw * ih != M) return -1;

    divisor = 1;
    while ((double)divisor < 2.0 * sqrt((double)K)) divisor *= 2;
    requant_params(1.0f / (float)divisor, &scale, &shift_reg);

    in_bytes    = (size_t)((kpad + C2 - 1) / C2) * ih * iw * C2;
    w_bytes     = (size_t)((nreg + 31) / 32) * ((kreg + 31) / 32) * 32 * 32;
    surf_elems  = rocket_rk3576_out_surf_elems(iw, ih, 0);
    /* The int32 writer puts FOUR lanes in the same 16-byte atom the int8 writer puts
     * sixteen in, so a channel group costs the same bytes and there are four times as
     * many groups: the BO is 4x, every stride register is unchanged. */
    obytes      = (size_t)((nreg + C2 - 1) / C2) * surf_elems * C2 * (i32out ? 4u : 1u);
    obytes     *= (size_t)env_int("ROCKET_MP_OSLACK", 1);
    coeff_bytes = rocket_rk3576_coeff_bytes(nreg);

    A       = calloc((size_t)M * K, 1);
    B       = calloc((size_t)N * K, 1);
    in_cube = calloc(in_bytes, 1);
    w_cube  = calloc(w_bytes, 1);
    out     = calloc(obytes, 1);
    bias    = calloc(nreg > N ? nreg : N, sizeof *bias);
    if (!A || !B || !in_cube || !w_cube || !out || !bias) goto done;

    seed = 0x9E3779B9u ^ (M * 31u + K * 17u + N * 7u);
    for (m = 0; m < M; m++)
        for (k = 0; k < K; k++)
            A[(size_t)m * K + k] = (int8_t)((int)((m * 7 + k * 13) % 61) - 30);
    for (n = 0; n < N; n++)
        for (k = 0; k < K; k++) {
            seed = seed * 1103515245u + 12345u;
            B[(size_t)n * K + k] = (int8_t)((int)((seed >> 16) % 17u) - 8);
        }
    for (n = 0; n < N; n++)
        bias[n] = (int32_t)((int)n - (int)N / 2) * 8;

    for (m = 0; m < M; m++)
        for (k = 0; k < K; k++)
            in_cube[feature_data((int)K, (int)ih, (int)iw, C2,
                                 (int)k + 1, (int)(m / iw) + 1, (int)(m % iw) + 1)] =
                A[(size_t)m * K + k];
    for (n = 0; n < N; n++)
        for (k = 0; k < K; k++)
            w_cube[weight_conv_int8((int)N, (int)K, 1, 1,
                                    (int)n + 1, (int)k + 1, 1, 1)] =
                B[(size_t)n * K + k];

    if (rocket_bo_alloc(fd, in_bytes, &in_bo)   < 0) goto done;
    if (rocket_bo_alloc(fd, w_bytes,  &w_bo)    < 0) goto done;
    if (rocket_bo_alloc(fd, coeff_bytes, &b_bo) < 0) goto done;
    if (rocket_bo_alloc(fd, obytes,   &o_bo)    < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &r_bo)  < 0) goto done;

    rocket_bo_prep(fd, &in_bo, 1, 0); memcpy(in_bo.ptr, in_cube, in_bytes); rocket_bo_fini(fd, &in_bo);
    rocket_bo_prep(fd, &w_bo,  1, 0); memcpy(w_bo.ptr,  w_cube,  w_bytes);  rocket_bo_fini(fd, &w_bo);
    rocket_bo_prep(fd, &b_bo,  1, 0);
    rocket_rk3576_pack_coeff(b_bo.ptr, coeff_bytes, bias, nreg);
    rocket_bo_fini(fd, &b_bo);
    rocket_bo_prep(fd, &o_bo,  1, 0); memset(o_bo.ptr, SENTINEL, obytes); rocket_bo_fini(fd, &o_bo);

    p.ic = (uint16_t)kreg; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.oc = (uint16_t)nreg; p.oh = (uint16_t)ih; p.ow = (uint16_t)iw;
    p.kh = 1; p.kw = 1;
    p.stride_y = 1; p.stride_x = 1;
    p.int8_out = 1;
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = (float)divisor;
    p.input_zero_point = 0x80;
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

    {
        rocket_rk3576_row_task plan[4096];
        unsigned nt = 1, t;
        double t0;
        conv_params_t q = p;
        q.ih_full = (uint16_t)ih; q.oh_full = (uint16_t)ih;
        if (rocket_rk3576_plan_rows(&q, 0, plan, 4096, &nt) < 0) { rc = 3; goto done; }
        st->tasks = (int)nt;
        st->ms = 0.0;
        for (t = 0; t < nt; t++) {
            conv_params_t r = p;
            r.ih = plan[t].ih; r.oh = plan[t].oh;
            r.pad_top = plan[t].pad_top;
            r.input_dma  = p.input_dma  + plan[t].feature_off;
            r.output_dma = p.output_dma + plan[t].output_off;
            r.ih_full = (uint16_t)ih; r.oh_full = (uint16_t)ih;
            rc = i32out ? gen_conv2d_int8_rk3576_i32out(&r)
                        : gen_conv2d_int8_rk3576(&r);
            if (rc != 0) { printf("  generator failed: %d\n", rc); goto done; }
            rocket_bo_prep(fd, &r_bo, 1, 0);
            memcpy(r_bo.ptr, ops, r.task_count * sizeof(uint64_t));
            rocket_bo_fini(fd, &r_bo);
            /* Pace OUTSIDE the measurement. The part needs a gap between jobs or a
             * later submit completes without writing at all, and timing through the
             * sleep would measure the sleep. */
            if (t) sleep_ms(gap);
            t0 = now_ms();
            rc = rocket_submit_matmul(fd, &r_bo, r.task_count, in_h, 4, out_h, 1, 4000);
            if (rc != 0) { printf("  submit failed: %d\n", rc); goto done; }
            /* The submit ioctl returns before the job does. Fence on the output BO
             * inside the loop, or the timing measures the ioctl and reports 0.0 ms. */
            if (rocket_bo_prep(fd, &o_bo, 0, 2000000000ull) < 0) {
                printf("  PREP_BO on the output timed out\n"); rc = -1; goto done;
            }
            st->ms += now_ms() - t0;
            rocket_bo_fini(fd, &o_bo);
        }
    }

    rocket_bo_prep(fd, &o_bo, 0, 2000000000ull);
    memcpy(out, o_bo.ptr, obytes);
    rocket_bo_fini(fd, &o_bo);

    {
        size_t i;
        for (i = 0; i < obytes; i++)
            if ((uint8_t)out[i] == SENTINEL) st->untouched++;
    }
    st->obytes = (int)obytes;
    st->total  = (int)(M * N);
    for (m = 0; m < M; m++)
        for (n = 0; n < N; n++) {
            int64_t acc = bias[n];
            int got, want, d;
            unsigned y = m / iw, x = m % iw;
            for (k = 0; k < K; k++)
                acc += (int64_t)A[(size_t)m * K + k] * B[(size_t)n * K + k];
            if (i32out) {
                /* The raw accumulator, and the int32 map: four 32-bit lanes to the
                 * atom, one atom per pixel, channel groups contiguous. */
                const int32_t *o32 = (const int32_t *)out;
                want = (int)acc;
                got  = o32[(size_t)(n / 4) * surf_elems * 4 + (size_t)4 * (y * iw + x) + (n % 4)];
            } else {
                want = requant_apply(acc, scale, shift_reg);
                got  = out[(size_t)(n / C2) * surf_elems * C2 + (size_t)C2 * (y * iw + x) + (n % C2)];
            }
            d = got > want ? got - want : want - got;
            if (d > st->maxdiff) st->maxdiff = d;
            if (got == want) st->exact++;
            else if (verbose && shown < 8) {
                printf("    mism m=%u n=%u: want %d got %d (acc %lld)\n",
                       m, n, want, got, (long long)acc);
                shown++;
            }
        }
    rc = (st->exact == st->total) ? 0 : 1;

done:
    if (r_bo.ptr)  rocket_bo_free(fd, &r_bo);
    if (o_bo.ptr)  rocket_bo_free(fd, &o_bo);
    if (b_bo.ptr)  rocket_bo_free(fd, &b_bo);
    if (w_bo.ptr)  rocket_bo_free(fd, &w_bo);
    if (in_bo.ptr) rocket_bo_free(fd, &in_bo);
    free(A); free(B); free(in_cube); free(w_cube); free(out); free(bias);
    return rc;
}

static int mm_int8(int fd, unsigned M, unsigned K, unsigned N,
                   unsigned iw, unsigned ih, struct mm_result *st)
{
    return mm_int8_ex(fd, M, K, N, iw, ih, env_int("ROCKET_MP_I32", 0), st);
}

/* ============================================================================
 * SECTION — the fp16 matmul, through the ic split
 *
 * One fp16 task contracts SIXTEEN input channels, so a K of any size costs K/16
 * submits and a host accumulation. This runs the same matmul that way, against the
 * same CPU model, so the two precisions are measured on one shape rather than two.
 * ==========================================================================*/
static int mm_fp16(int fd, unsigned M, unsigned K, unsigned N,
                   unsigned iw, unsigned ih, struct mm_result *st)
{
    unsigned kreg = rocket_rk3576_fp16_pad_ic(K), nreg = rocket_rk3576_fp16_pad_oc(N);
    _Float16 *A = NULL, *B = NULL, *in_cube = NULL, *w_slice = NULL;
    uint8_t *surface = NULL;
    float *acc = NULL;
    int32_t *bias = NULL;
    size_t in_bytes, obytes, coeff_bytes, wmax = 0;
    rocket_bo in_bo = {0}, w_bo = {0}, b_bo = {0}, o_bo = {0}, r_bo = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t in_h[4], out_h[1];
    conv_params_t p = {0};
    rocket_rk3576_ic_task tasks[512];
    unsigned ntask = 0, s, m, n, k;
    int gap = env_int("ROCKET_MP_GAP_MS", 200);
    int verbose = env_int("ROCKET_MP_VERBOSE", 0);
    int rc = -1, shown = 0;
    double t0;

    memset(st, 0, sizeof *st);
    if (iw * ih != M) return -1;

    in_bytes    = (size_t)((kreg + 7) / 8) * ih * iw * 8 * sizeof(_Float16);
    obytes      = rocket_rk3576_fp16_out_bytes(nreg, ih, iw);
    coeff_bytes = rocket_rk3576_coeff_bytes(nreg);

    p.ic = (uint16_t)kreg; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
    p.oc = (uint16_t)nreg; p.oh = (uint16_t)ih; p.ow = (uint16_t)iw;
    p.kh = 1; p.kw = 1;
    p.stride_y = 1; p.stride_x = 1;
    p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)ih;
    if (rocket_rk3576_plan_ic(&p, tasks, 512, &ntask) < 0) { rc = 3; goto done; }
    st->tasks = (int)ntask;

    for (s = 0; s < ntask; s++) {
        size_t b = rocket_rk3576_fp16_slice_weight_bytes(nreg, tasks[s].ic, 1, 1);
        if (b > wmax) wmax = b;
    }

    /* A and B carry the PADDED channel counts, zero beyond the logical ones: the
     * slice weight packer indexes w_oihw at the full ic stride, so a row-major B at
     * the logical K would put the next output channel where the padding belongs. */
    A       = calloc((size_t)M * kreg, sizeof *A);
    B       = calloc((size_t)nreg * kreg, sizeof *B);
    in_cube = calloc(in_bytes, 1);
    w_slice = calloc(wmax, 1);
    surface = calloc(obytes, 1);
    acc     = calloc((size_t)rocket_rk3576_fp16_out_channels(nreg) * ih * iw, sizeof *acc);
    bias    = calloc(nreg > N ? nreg : N, sizeof *bias);
    if (!A || !B || !in_cube || !w_slice || !surface || !acc || !bias) goto done;

    {
        unsigned seed = 0x9E3779B9u ^ (M * 31u + K * 17u + N * 7u);
        for (m = 0; m < M; m++)
            for (k = 0; k < K; k++)
                A[(size_t)m * kreg + k] = (_Float16)(((int)((m * 7 + k * 13) % 61) - 30) / 32.0f);
        for (n = 0; n < N; n++)
            for (k = 0; k < K; k++) {
                seed = seed * 1103515245u + 12345u;
                B[(size_t)n * kreg + k] = (_Float16)(((int)((seed >> 16) % 17u) - 8) / 16.0f);
            }
    }

    /* The fp16 feature cube: 8 lanes to a 16-byte atom, channel groups contiguous. */
    for (m = 0; m < M; m++)
        for (k = 0; k < kreg; k++)
            in_cube[(size_t)(k / 8) * ih * iw * 8 + (size_t)8 * m + (k % 8)] =
                A[(size_t)m * kreg + k];

    if (rocket_bo_alloc(fd, in_bytes, &in_bo)   < 0) goto done;
    if (rocket_bo_alloc(fd, wmax,     &w_bo)    < 0) goto done;
    if (rocket_bo_alloc(fd, coeff_bytes, &b_bo) < 0) goto done;
    if (rocket_bo_alloc(fd, obytes,   &o_bo)    < 0) goto done;
    if (rocket_bo_alloc(fd, sizeof ops, &r_bo)  < 0) goto done;

    rocket_bo_prep(fd, &in_bo, 1, 0); memcpy(in_bo.ptr, in_cube, in_bytes); rocket_bo_fini(fd, &in_bo);
    rocket_bo_prep(fd, &b_bo, 1, 0);
    rocket_rk3576_pack_coeff_prec(b_bo.ptr, coeff_bytes, NULL, nreg, precision_float16);
    rocket_bo_fini(fd, &b_bo);

    p.input_dma   = in_bo.dma_address;
    p.weights_dma = w_bo.dma_address;
    p.bias_dma    = b_bo.dma_address;
    p.output_dma  = o_bo.dma_address;
    p.tasks       = ops;
    p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;

    in_h[0] = in_bo.handle; in_h[1] = w_bo.handle;
    in_h[2] = b_bo.handle;  in_h[3] = r_bo.handle;
    out_h[0] = o_bo.handle;

    st->ms = 0.0;
    for (s = 0; s < ntask; s++) {
        conv_params_t q = p;
        size_t wb = rocket_rk3576_fp16_slice_weight_bytes(nreg, tasks[s].ic, 1, 1);
        memset(w_slice, 0, wmax);
        /* The slice weight packer wants OIHW at the FULL contraction; it takes the
         * slice's channel range from the task. B is [N,K] which is exactly OIHW at
         * kh=kw=1. */
        if (rocket_rk3576_fp16_pack_slice_weights(w_slice, wb, B, nreg, kreg, 1, 1,
                                                  &tasks[s]) < 0) {
            printf("  slice weight pack failed at s=%u\n", s); goto done;
        }
        rocket_bo_prep(fd, &w_bo, 1, 0); memcpy(w_bo.ptr, w_slice, wb); rocket_bo_fini(fd, &w_bo);
        rocket_bo_prep(fd, &o_bo, 1, 0); memset(o_bo.ptr, 0, obytes); rocket_bo_fini(fd, &o_bo);

        q.ic = (uint16_t)tasks[s].ic;
        q.input_dma = p.input_dma + tasks[s].feature_off;
        rc = gen_conv2d_fp16_rk3576(&q);
        if (rc != 0) { printf("  fp16 generator failed: %d\n", rc); goto done; }
        rocket_bo_prep(fd, &r_bo, 1, 0);
        memcpy(r_bo.ptr, ops, q.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &r_bo);
        if (s) sleep_ms(gap);
        t0 = now_ms();
        rc = rocket_submit_matmul(fd, &r_bo, q.task_count, in_h, 4, out_h, 1, 4000);
        if (rc != 0) { printf("  submit failed: %d\n", rc); goto done; }
        rocket_bo_prep(fd, &o_bo, 0, 2000000000ull);
        st->ms += now_ms() - t0;
        memcpy(surface, o_bo.ptr, obytes);
        rocket_bo_fini(fd, &o_bo);
        if (rocket_rk3576_fp16_accumulate(acc, surface, obytes, nreg, ih, iw) < 0) {
            printf("  accumulate failed at s=%u\n", s); goto done;
        }
    }

    st->total = (int)(M * N);
    for (m = 0; m < M; m++)
        for (n = 0; n < N; n++) {
            double want = 0.0, got, d, tol;
            unsigned y = m / iw, x = m % iw;
            for (k = 0; k < K; k++)
                want += (double)(float)A[(size_t)m * kreg + k] *
                        (double)(float)B[(size_t)n * kreg + k];
            got = acc[((size_t)n * ih + y) * iw + x];
            d = fabs(got - want);
            tol = 0.02 * fabs(want) + 0.05;
            if (d > (double)st->maxdiff) st->maxdiff = (int)d;
            if (d <= tol) st->exact++;
            else if (verbose && shown < 8) {
                printf("    mism m=%u n=%u: want %.4f got %.4f\n", m, n, want, got);
                shown++;
            }
        }
    rc = (st->exact == st->total) ? 0 : 1;

done:
    if (r_bo.ptr)  rocket_bo_free(fd, &r_bo);
    if (o_bo.ptr)  rocket_bo_free(fd, &o_bo);
    if (b_bo.ptr)  rocket_bo_free(fd, &b_bo);
    if (w_bo.ptr)  rocket_bo_free(fd, &w_bo);
    if (in_bo.ptr) rocket_bo_free(fd, &in_bo);
    free(A); free(B); free(in_cube); free(w_slice); free(surface); free(acc); free(bias);
    return rc;
}

/* ============================================================================
 * SECTION — the modes
 * ==========================================================================*/
static const char *verdict(int rc)
{
    switch (rc) {
    case 0:  return "PASS";
    case 1:  return "WRONG";
    case 3:  return "refused";
    default: return "BROKEN";
    }
}

struct mkn { unsigned M, K, N; };

static int mode_shape(int fd)
{
    static const struct mkn T[] = {
        {   8,   64,  32}, {   8,  128,  32}, {   8,  256,  32},
        {  16,  512,  32}, {  16,  512,  64}, {  16,  512, 128},
        {  32, 1024,  64}, {  32, 2048,  64}, {  32, 4096,  64},
        {  56, 4608, 128}, {  64, 4096, 128}, { 128, 2048, 128},
        { 256, 1024, 128}, { 512,  512, 128}, {1024,  256, 128},
        {  64, 1024, 256}, {  64, 1024, 512}, {  64,  512,1024},
        /* awkward numbers: M with no even divisor, K and N off the round grid */
        {   7,   64,  32}, {  13,   96,  32}, {  33,  160,  96},
        { 100,  288,  32}, {  49,   32,  32},
    };
    int i, fails = 0;
    printf("  %-22s %-9s %8s %6s  %s\n", "shape", "plane", "exact/total", "tasks", "ms");
    for (i = 0; i < (int)(sizeof T / sizeof T[0]); i++) {
        struct mm_result st;
        unsigned iw, ih;
        char shape[32], plane[16];
        int rc;
        pick_plane(T[i].M, T[i].K, &iw, &ih);
        rc = mm_int8(fd, T[i].M, T[i].K, T[i].N, iw, ih, &st);
        snprintf(shape, sizeof shape, "%ux%ux%u", T[i].M, T[i].K, T[i].N);
        snprintf(plane, sizeof plane, "%ux%u", iw, ih);
        printf("  %-6s %-15s %-9s %6d/%-6d %4d  %6.1f%s\n",
               verdict(rc), shape, plane, st.exact, st.total, st.tasks, st.ms,
               st.untouched == st.obytes ? "  (surface untouched)" : "");
        if (rc) fails++;
    }
    return fails;
}

/* The M axis. On the RK3588 rows are the conv's spatial height and a height under 4
 * mis-computes, so M%4 is the real bound there. This asks the RK3576 the same
 * question directly, and separately from the plane's factorization. */
static int mode_m(int fd)
{
    static const unsigned MS[] = {1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 17, 31, 32, 64};
    unsigned K = (unsigned)env_int("ROCKET_MP_K", 64);
    unsigned N = (unsigned)env_int("ROCKET_MP_N", 32);
    int i, fails = 0;
    printf("  the M axis at K=%u N=%u, one row of pixels per plane shape\n", K, N);
    printf("  %-6s %5s %-9s %s\n", "", "M", "plane", "exact/total");
    for (i = 0; i < (int)(sizeof MS / sizeof MS[0]); i++) {
        unsigned M = MS[i], iw, ih, f;
        /* Every factorization of M, so "M is fine" and "this plane is fine" cannot be
         * confused. A tall plane (iw=1) and a flat one (ih=1) are the two extremes. */
        for (f = 1; f <= M; f++) {
            struct mm_result st;
            char plane[16];
            int rc;
            if (M % f) continue;
            iw = f; ih = M / f;
            rc = mm_int8(fd, M, K, N, iw, ih, &st);
            snprintf(plane, sizeof plane, "%ux%u", iw, ih);
            printf("  %-6s %5u %-9s %6d/%-6d%s\n", verdict(rc), M, plane,
                   st.exact, st.total,
                   st.untouched == st.obytes ? "  (surface untouched)" : "");
            if (rc) fails++;
        }
    }
    return fails;
}

/* What one submit costs, at both precisions, on the same shape. */
static int mode_cost(int fd)
{
    static const struct mkn T[] = {
        {  16,  128,  64}, {  16,  256,  64}, {  16,  512,  64},
        {  32,  512,  64}, {  32, 1024,  64}, {  32, 1024, 128},
        {  56, 4608, 128}, { 128, 2048, 128}, { 256, 1024, 128},
    };
    int i;
    printf("  %-16s %6s %8s %9s | %6s %8s %9s | %s\n",
           "shape", "i8 tk", "i8 ms", "i8 GOP/s", "f16 tk", "f16 ms", "f16 GOP/s", "i8/f16");
    for (i = 0; i < (int)(sizeof T / sizeof T[0]); i++) {
        struct mm_result a, b;
        unsigned iw, ih;
        char shape[24];
        double ops = 2.0 * T[i].M * T[i].K * T[i].N / 1e9;
        int ra, rb;
        pick_plane(T[i].M, T[i].K, &iw, &ih);
        /* Pace between runs as well as between tasks: an unpaced job completes without
         * writing, which reads as an arithmetic failure. Both are measured with the
         * pacing outside the accumulator, so what is reported is submit + fence. */
        sleep_ms(env_int("ROCKET_MP_GAP_MS", 200));
        ra = mm_int8(fd, T[i].M, T[i].K, T[i].N, iw, ih, &a);
        sleep_ms(env_int("ROCKET_MP_GAP_MS", 200));
        rb = mm_fp16(fd, T[i].M, T[i].K, T[i].N, iw, ih, &b);
        snprintf(shape, sizeof shape, "%ux%ux%u", T[i].M, T[i].K, T[i].N);
        printf("  %-16s %6d %8.1f %9.1f | %6d %8.1f %9.1f | %5.2fx  %s/%s\n",
               shape, a.tasks, a.ms, a.ms > 0 ? ops / (a.ms / 1000.0) : 0.0,
               b.tasks, b.ms, b.ms > 0 ? ops / (b.ms / 1000.0) : 0.0,
               (a.ms > 0 && b.ms > 0) ? b.ms / a.ms : 0.0,
               verdict(ra), verdict(rb));
    }
    return 0;
}

/* Where the best operating point is. One submit costs ~1.3 ms whatever it carries, so
 * throughput is MACs per submit, and M*K is capped by the feature budget — which
 * leaves N, the output-channel axis, as the only free one. It is not free without
 * limit: the weight path holds one output-channel GROUP resident at a time, and a
 * conv driving several loses the trailing ones once the slice grows. This asks the
 * part where that boundary is on a k=1 conv, which is the matmul's own kernel. */
static int mode_peak(int fd)
{
    static const unsigned NS[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    static const unsigned KS[] = {256, 512, 1024, 2048, 4096, 4608};
    unsigned ki, ni;
    printf("  MACs per submit at the feature-budget cap (M*K = 262144), by N\n");
    printf("  %-6s %-16s %-9s %6s %8s %10s\n",
           "", "shape", "plane", "tasks", "ms", "GOP/s");
    for (ki = 0; ki < sizeof KS / sizeof KS[0]; ki++) {
        unsigned K = KS[ki];
        unsigned M = 262144u / K;
        unsigned iw, ih;
        pick_plane(M, K, &iw, &ih);
        for (ni = 0; ni < sizeof NS / sizeof NS[0]; ni++) {
            struct mm_result st;
            char shape[24], plane[16];
            double ops = 2.0 * M * K * NS[ni] / 1e9;
            int rc;
            sleep_ms(env_int("ROCKET_MP_GAP_MS", 200));
            rc = mm_int8(fd, M, K, NS[ni], iw, ih, &st);
            snprintf(shape, sizeof shape, "%ux%ux%u", M, K, NS[ni]);
            snprintf(plane, sizeof plane, "%ux%u", iw, ih);
            printf("  %-6s %-16s %-9s %6d %8.2f %10.1f%s\n",
                   verdict(rc), shape, plane, st.tasks, st.ms,
                   st.ms > 0 ? ops / (st.ms / 1000.0) : 0.0,
                   st.untouched == st.obytes ? "  (surface untouched)" : "");
        }
    }
    return 0;
}

/* Bracket the N boundary the peak sweep found. Throughput on this part is MACs per
 * submit and N is the axis that buys them, so where N stops computing is the single
 * number a tiling planner most needs — and it fails SILENTLY, with a full correctly
 * sized surface, so it has to be a refusal in the planner rather than a hope.
 *
 * Two candidate bounds are separated here: an N cap that holds at every K, and a
 * weight-cube size cap K*N that only bites at large K. The sweep runs each point at
 * the feature-budget cap so the feature side is constant and out of the way. */
static unsigned env_list(const char *name, unsigned *out, unsigned max,
                         const unsigned *dflt, unsigned ndflt)
{
    const char *e = getenv(name);
    unsigned n = 0;
    if (!e || !*e) {
        for (n = 0; n < ndflt && n < max; n++) out[n] = dflt[n];
        return n;
    }
    while (*e && n < max) {
        char *end;
        unsigned long v = strtoul(e, &end, 0);
        if (end == e) break;
        out[n++] = (unsigned)v;
        e = end;
        while (*e == ',' || *e == ' ') e++;
    }
    return n;
}

static int mode_ncap(int fd)
{
    static const unsigned KD[] = {256, 1024, 2048, 4096, 4608};
    static const unsigned ND[] = {1024, 1536, 2048, 2560, 3072, 4096};
    unsigned KS[32], NS[32];
    unsigned nk = env_list("ROCKET_MP_KS", KS, 32, KD, sizeof KD / sizeof KD[0]);
    unsigned nn = env_list("ROCKET_MP_NS", NS, 32, ND, sizeof ND / sizeof ND[0]);
    unsigned ki, ni;
    unsigned reps = (unsigned)env_int("ROCKET_MP_REPS", 3);
    printf("  the N boundary, at M*K = 262144 throughout, %u repeats each\n", reps);
    printf("  %-16s %10s %6s %8s %10s  %s\n",
           "shape", "wcube KiB", "pass", "ms", "GOP/s", "worst run");
    for (ki = 0; ki < nk; ki++) {
        unsigned K = KS[ki], M = 262144u / K, iw, ih;
        pick_plane(M, K, &iw, &ih);
        for (ni = 0; ni < nn; ni++) {
            char shape[24], worst[48] = "";
            double ops = 2.0 * M * K * NS[ni] / 1e9, ms = 0.0;
            unsigned r, passes = 0;
            for (r = 0; r < reps; r++) {
                struct mm_result st;
                int rc;
                sleep_ms(env_int("ROCKET_MP_GAP_MS", 200));
                rc = mm_int8(fd, M, K, NS[ni], iw, ih, &st);
                ms += st.ms;
                if (rc == 0) passes++;
                else if (!worst[0])
                    /* Say WHICH failure it is. A dead submit leaves the sentinel
                     * everywhere and is a pacing/wall symptom; a partial surface is a
                     * capacity boundary; wrong values with the surface fully written
                     * is arithmetic. The three want different responses. */
                    snprintf(worst, sizeof worst, "%s %d/%d exact, %d/%d untouched",
                             verdict(rc), st.exact, st.total, st.untouched, st.obytes);
            }
            snprintf(shape, sizeof shape, "%ux%ux%u", M, K, NS[ni]);
            printf("  %-16s %10.0f %4u/%-2u %8.2f %10.1f  %s\n",
                   shape, (double)K * NS[ni] / 1024.0, passes, reps, ms / reps,
                   ms > 0 ? ops / (ms / reps / 1000.0) : 0.0, worst);
        }
    }
    return 0;
}

/* ============================================================================
 * SECTION — out32: does an INTEGER program reach the DPU's 32-bit epilogue word?
 *
 * The int8 conv path writes an int8 surface through the OUT_CVT requant, which caps a
 * matmul's contraction at what one task holds: a K larger than that would have to be
 * split, and int8 partials cannot be summed without quantizing each one. The RK3588's
 * int8 matmul avoids that by writing int32 and summing partials on the host, and the
 * DPU field that would do the same here is 0x4010 [31:29], a WIDTH selector — 5 takes
 * the whole 32-bit word on the float path.
 *
 * This asks the part directly, and DECODES rather than scores: the operands make every
 * accumulator distinct, so each 32-bit word in the raw output BO either equals exactly
 * one expected accumulator or it does not. A map read that way is a bijection or it is
 * nothing; comparing against an assumed layout would hide the very permutation the
 * probe is looking for.
 * ==========================================================================*/
static int mode_out32(int fd)
{
    unsigned M = (unsigned)env_int("ROCKET_MP_M", 8);
    unsigned K = (unsigned)env_int("ROCKET_MP_K", 32);
    unsigned N = (unsigned)env_int("ROCKET_MP_N", 32);
    /* The plane is part of the map, not a presentation choice: the writer's blocks are
     * sized in ow*oh_full and the pixel index is y*ow + x, so a tall plane and a flat
     * one of the same M address the surface differently. ROCKET_MP_IW picks it. */
    unsigned iw = (unsigned)env_int("ROCKET_MP_IW", (int)M), ih;
    if (!iw || M % iw) { printf("  ROCKET_MP_IW must divide M\n"); return -1; }
    ih = M / iw;
    /* The fourth variant is the interesting one. The first three showed the writer
     * emits the SAME byte budget at every output width — ceil(oc/16) groups of one
     * 16-byte atom per pixel — so at 4 bytes an element only a quarter of the channels
     * reach DDR. If the budget is simply the DPU's own channel count, telling it there
     * are FOUR TIMES as many channels should buy the other three quarters. */
    typedef struct { unsigned w; int chan4; const char *set; const char *what; } variant_t;
    static const variant_t BUILTIN[] = {
        {0, 0, NULL, "int8, the control"},
        {4, 0, NULL, "int32"},
        {5, 0, NULL, "fp32"},
        {4, 1, NULL, "int32, DPU channel count x4"},
        /* The RK3588's int32-raw writer differs from its int8 one in exactly two
         * quantities beyond DATA_FORMAT: size_e 7 rather than 3, and surf_add
         * dst_surf_stride*8 rather than *4 (npu_regcmd.c). If the RK3576 carries the
         * same pair, its counterparts are 0x401C (the destination surface stride) and
         * 0x40B8 (the channel-group jump), and each wants DOUBLING at a 4-byte element.
         * These drive them one at a time and together, which is what separates a stride
         * that needs scaling from a write EXTENT that is set somewhere else. */
        {4, 0, "0x401C=0x00000010",                   "int32, 0x401C doubled"},
        {4, 0, "0x40B8=0x00000010",                   "int32, 0x40B8 doubled"},
        {4, 0, "0x401C=0x00000010,0x40B8=0x00000010", "int32, both doubled"},
        {4, 0, "0x401C=0x00000020,0x40B8=0x00000020", "int32, both x4"},
        /* 0x4030's low half is 0x710 on the direct path and 0x310 on depthwise — the
         * same 7-versus-3 the RK3588 calls size_e. Direct already carries the 7, so if
         * that field is the write extent it is already at its int32 value and something
         * else caps it; driving it to 3 and to 0xF says which. */
        {4, 0, "0x4030=0x001F0310",                   "int32, 0x4030 low half -> 0x310"},
        {4, 0, "0x4030=0x001F0F10",                   "int32, 0x4030 low half -> 0xF10"},
    };
    /* The table is the default, not the instrument. `ROCKET_MP_VARIANTS` replaces it
     * with a semicolon-separated list of `width:regspec`, where regspec is the same
     * `<reg>=<val>,...` ROCKET_RK3576_SET takes and may be empty:
     *
     *   ROCKET_MP_VARIANTS='0:;4:;4:0x3020=0x7F,0x1024=0x7F'
     *
     * A register combination is one string, so a MODE — several registers that are
     * each inert until the others are right — can be driven without a rebuild. */
    const variant_t *VARIANTS = BUILTIN;
    variant_t *parsed = NULL;
    unsigned nvar = sizeof BUILTIN / sizeof BUILTIN[0];
    char *vspec = NULL;
    unsigned wi;
    unsigned oslack = (unsigned)env_int("ROCKET_MP_OSLACK", 8);
    int scatter = env_int("ROCKET_MP_SCATTER", 0);

    {
        const char *e = getenv("ROCKET_MP_VARIANTS");
        if (e && *e) {
            char *p;
            unsigned n = 1;
            for (p = (char *)e; *p; p++) if (*p == ';') n++;
            vspec = strdup(e);
            parsed = calloc(n, sizeof *parsed);
            if (!vspec || !parsed) return -1;
            VARIANTS = parsed;
            nvar = 0;
            for (p = vspec; p && *p; ) {
                char *semi = strchr(p, ';');
                char *colon = strchr(p, ':');
                if (semi) *semi = '\0';
                if (!colon || (semi && colon > semi)) { p = semi ? semi + 1 : NULL; continue; }
                *colon = '\0';
                parsed[nvar].w    = (unsigned)strtoul(p, NULL, 0);
                parsed[nvar].set  = *(colon + 1) ? colon + 1 : NULL;
                parsed[nvar].what = *(colon + 1) ? colon + 1 : "no override";
                nvar++;
                p = semi ? semi + 1 : NULL;
            }
        }
    }

    printf("  M=%u K=%u N=%u, unity requant, output BO %ux the int8 size\n", M, K, N, oslack);
    printf("  every accumulator distinct; a word is decoded only if it matches ONE\n\n");

    for (wi = 0; wi < nvar; wi++) {
        unsigned w = VARIANTS[wi].w;
        int chan4 = VARIANTS[wi].chan4;
        int8_t *A = calloc((size_t)M * K, 1), *B = calloc((size_t)N * K, 1);
        int8_t *in_cube = NULL, *w_cube = NULL;
        int32_t *want = calloc((size_t)M * N, sizeof(int32_t));
        size_t in_bytes = (size_t)((K + C2 - 1) / C2) * ih * iw * C2;
        unsigned surf_px = rocket_rk3576_out_surf_elems(iw, ih, 0);
        size_t w_bytes  = (size_t)((N + 31) / 32) * ((K + 31) / 32) * 32 * 32;
        size_t obytes   = (size_t)((N + C2 - 1) / C2) * surf_px * C2 * oslack;
        size_t coeff_bytes = rocket_rk3576_coeff_bytes(N);
        rocket_bo in_bo = {0}, w_bo = {0}, b_bo = {0}, o_bo = {0}, r_bo = {0};
        uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
        uint32_t in_h[4], out_h[1];
        conv_params_t p = {0};
        char setspec[160];
        unsigned m, n, k, found = 0, collide = 0, fits = 0, fits_wide = 0, off_set = 0;
        unsigned words = (unsigned)(obytes / 4);
        const int32_t *o32;
        unsigned first_idx = 0, first_stride = 0, nshown = 0;

        in_cube = calloc(in_bytes, 1);
        w_cube  = calloc(w_bytes, 1);
        if (!A || !B || !in_cube || !w_cube || !want) return -1;

        /* The decode is only a decode if every accumulator NAMES exactly one output,
         * so the operands are drawn until all M*N of them are distinct. A small ramp
         * looks tidy and produces collisions (two different (m,n) sharing a value),
         * and a collision reads as "the map is ambiguous here" when what is ambiguous
         * is the probe. */
        if (!env_int("ROCKET_MP_RANDOM", 0)) {
            /* Distinct BY CONSTRUCTION, which a random draw stops managing well past a
             * few hundred accumulators: two live contraction lanes give
             * C[m][n] = (n+1) + m*(N+1), which names every position for N<=126 and
             * M<=127 and never lands on zero or on the sentinel. The arithmetic is not
             * what this mode measures — the conv gate covers that — so a sparse operand
             * costs nothing here and buys the whole oc axis. */
            unsigned hi = (N + 1u) / 2u, lo = (N + 1u) - hi;
            if (N > 252 || M > 127 || K < 4) {
                printf("  constructed operands need N<=252, M<=127, K>=4\n"); return -1;
            }
            /* FOUR live lanes, not two: an oc past 126 does not fit one int8 weight
             * byte, and both the channel term and the row stride have to be split
             * across a pair of lanes to reach it. */
            for (m = 0; m < M; m++) {
                A[(size_t)m * K + 0] = 1;
                A[(size_t)m * K + 1] = 1;
                A[(size_t)m * K + 2] = (int8_t)m;
                A[(size_t)m * K + 3] = (int8_t)m;
            }
            for (n = 0; n < N; n++) {
                unsigned c0 = (n + 1u) > 127u ? 127u : (n + 1u);
                B[(size_t)n * K + 0] = (int8_t)c0;
                B[(size_t)n * K + 1] = (int8_t)((n + 1u) - c0);
                B[(size_t)n * K + 2] = (int8_t)hi;
                B[(size_t)n * K + 3] = (int8_t)lo;
            }
            for (m = 0; m < M; m++)
                for (n = 0; n < N; n++)
                    want[(size_t)m * N + n] = (int32_t)(n + 1) + (int32_t)m * (int32_t)(N + 1);
        } else {
            unsigned seed = 0x1234567u, tries;
            for (tries = 0; tries < 64; tries++) {
                unsigned i, j;
                int dup = 0;
                for (m = 0; m < M; m++)
                    for (k = 0; k < K; k++) {
                        seed = seed * 1103515245u + 12345u;
                        A[(size_t)m * K + k] = (int8_t)((int)((seed >> 16) % 101u) - 50);
                    }
                for (n = 0; n < N; n++)
                    for (k = 0; k < K; k++) {
                        seed = seed * 1103515245u + 12345u;
                        B[(size_t)n * K + k] = (int8_t)((int)((seed >> 16) % 101u) - 50);
                    }
                for (m = 0; m < M; m++)
                    for (n = 0; n < N; n++) {
                        int32_t acc = 0;
                        for (k = 0; k < K; k++)
                            acc += (int32_t)A[(size_t)m * K + k] * B[(size_t)n * K + k];
                        want[(size_t)m * N + n] = acc;
                    }
                for (i = 0; i < M * N && !dup; i++)
                    for (j = i + 1; j < M * N; j++)
                        if (want[i] == want[j]) { dup = 1; break; }
                if (!dup) break;
            }
            if (tries == 64) { printf("  could not draw distinct accumulators\n"); return -1; }
        }

        for (m = 0; m < M; m++)
            for (k = 0; k < K; k++)
                in_cube[feature_data((int)K, (int)ih, (int)iw, C2, (int)k + 1,
                                     (int)(m / iw) + 1, (int)(m % iw) + 1)] =
                    A[(size_t)m * K + k];
        /* ROCKET_MP_SCATTER leaves the UNDELIVERED channels of every sixteen at zero,
         * which is how the library drives this writer — it scatters real channels into
         * the delivered slots and never programs the rest. Driving every channel live
         * is a different job for the part, and the two do not agree: a surface the
         * all-live probe reads as a broken map can be exactly right under the scatter.
         * So a boundary found without this is a boundary of the probe. */
        for (n = 0; n < N; n++) {
            if (scatter && (n % 16u) >= 8u) {
                unsigned mm;
                for (mm = 0; mm < M; mm++) want[(size_t)mm * N + n] = 0;
                continue;
            }
            for (k = 0; k < K; k++)
                w_cube[weight_conv_int8((int)N, (int)K, 1, 1, (int)n + 1, (int)k + 1, 1, 1)] =
                    B[(size_t)n * K + k];
        }

        if (rocket_bo_alloc(fd, in_bytes, &in_bo) < 0) return -1;
        if (rocket_bo_alloc(fd, w_bytes, &w_bo) < 0) return -1;
        if (rocket_bo_alloc(fd, coeff_bytes, &b_bo) < 0) return -1;
        if (rocket_bo_alloc(fd, obytes, &o_bo) < 0) return -1;
        if (rocket_bo_alloc(fd, sizeof ops, &r_bo) < 0) return -1;

        rocket_bo_prep(fd, &in_bo, 1, 0); memcpy(in_bo.ptr, in_cube, in_bytes); rocket_bo_fini(fd, &in_bo);
        rocket_bo_prep(fd, &w_bo, 1, 0);  memcpy(w_bo.ptr, w_cube, w_bytes);    rocket_bo_fini(fd, &w_bo);
        rocket_bo_prep(fd, &b_bo, 1, 0);
        rocket_rk3576_pack_coeff(b_bo.ptr, coeff_bytes, NULL, N);
        rocket_bo_fini(fd, &b_bo);
        rocket_bo_prep(fd, &o_bo, 1, 0);
        memset(o_bo.ptr, 0xEE, obytes);   /* 0xEEEEEEEE matches no accumulator here */
        rocket_bo_fini(fd, &o_bo);

        p.ic = (uint16_t)K; p.ih = (uint16_t)ih; p.iw = (uint16_t)iw;
        p.oc = (uint16_t)N; p.oh = (uint16_t)ih; p.ow = (uint16_t)iw;
        p.kh = 1; p.kw = 1; p.stride_y = 1; p.stride_x = 1;
        p.ih_full = (uint16_t)ih; p.oh_full = (uint16_t)ih;
        p.int8_out = 1;
        p.in_scale = 1.0f; p.w_scale = 1.0f; p.out_scale = 1.0f;   /* unity requant */
        p.input_zero_point = 0x80; p.output_zero_point = 0x80; p.weight_zero_point = 0x80;
        p.tasks = ops;
        p.input_dma = in_bo.dma_address; p.weights_dma = w_bo.dma_address;
        p.bias_dma = b_bo.dma_address;   p.output_dma = o_bo.dma_address;

        /* Patch the width AFTER emission, so the transcribed program is untouched when
         * the knob is not driven and the capture gate still holds it. */
        if (chan4)
            snprintf(setspec, sizeof setspec,
                     "0x4010=0x%08x,0x402C=0x%08x,0x4030=0x%08x",
                     w << 29, 4 * N - 1, ((4 * N - 1) << 16) | 0x710u);
        else if (VARIANTS[wi].set)
            snprintf(setspec, sizeof setspec, "0x4010=0x%08x,%s", w << 29,
                     VARIANTS[wi].set);
        else
            snprintf(setspec, sizeof setspec, "0x4010=0x%08x", w << 29);
        setenv("ROCKET_RK3576_SET", setspec, 1);
        if (gen_conv2d_int8_rk3576(&p) != 0) { printf("  generator failed\n"); return -1; }
        unsetenv("ROCKET_RK3576_SET");

        in_h[0] = in_bo.handle; in_h[1] = w_bo.handle;
        in_h[2] = b_bo.handle;  in_h[3] = r_bo.handle;
        out_h[0] = o_bo.handle;
        rocket_bo_prep(fd, &r_bo, 1, 0);
        memcpy(r_bo.ptr, ops, p.task_count * sizeof(uint64_t));
        rocket_bo_fini(fd, &r_bo);
        sleep_ms(env_int("ROCKET_MP_GAP_MS", 200));
        if (rocket_submit_matmul(fd, &r_bo, p.task_count, in_h, 4, out_h, 1, 4000) != 0) {
            printf("  submit failed\n"); return -1;
        }
        rocket_bo_prep(fd, &o_bo, 0, 2000000000ull);
        o32 = (const int32_t *)o_bo.ptr;

        printf("  0x4010 out width %u (%s):\n", w, VARIANTS[wi].what);
        /* The candidate map, from the int32 output cube the RK3588 writes: four 32-bit
         * lanes to a 16-byte atom, one atom per pixel, channel groups as contiguous
         * planes. Every decoded position is CHECKED against it rather than assumed, so
         * a different packing shows up as a miss instead of being read through. */
        int *at_of = calloc((size_t)M * N, sizeof(int));
        if (!at_of) return -1;
        for (m = 0; m < M * N; m++) at_of[m] = -1;
        for (m = 0; m < M; m++)
            for (n = 0; n < N; n++) {
                unsigned i, hits = 0, at = 0;
                /* Two candidate maps, scored separately: the NARROW writer's, which
                 * is the plain RK3588 int32 cube, and the WIDE one the doubled byte
                 * budget produces. Which of them a variant lands on is the finding,
                 * so neither is assumed. */
                int wide = rocket_rk3576_i32_wide_word(iw, ih, n, m);
                size_t cand = (size_t)(n / 4) * surf_px * 4 + (size_t)4 * m + (n % 4);
                if (scatter && (n % 16u) >= 8u) continue;
                for (i = 0; i < words; i++)
                    if (o32[i] == want[(size_t)m * N + n]) { hits++; at = i; }
                if (hits == 1) at_of[(size_t)m * N + n] = (int)at;
                if (hits == 1) {
                    found++;
                    /* A value delivered at a channel the wide map calls UNDELIVERED is
                     * not a misplaced word — it means the delivered SET is not the one
                     * the map describes, which is a different failure and wants saying
                     * separately. */
                    if (wide < 0) off_set++;
                    else if (at == (unsigned)wide) fits_wide++;
                    if (at == cand) fits++;
                    else if (nshown < 6) {
                        printf("      C[%u][%u] at word %u, candidate map says %zu\n",
                               m, n, at, cand);
                        nshown++;
                    }
                    if (found == 1) first_idx = at;
                    else if (found == 2) first_stride = at - first_idx;
                } else if (hits > 1) collide++;
            }
        /* How far the writer actually REACHES. This is not a curiosity: the int8
         * surface's byte extent under-allocates it, and the DPU then writes past the
         * BO, which faults the IOMMU and wedges the part until a reboot. The BO here
         * is deliberately oversized so the extent can be read instead of hit. */
        {
            unsigned i, last = 0, touched = 0;
            for (i = 0; i < words; i++)
                if ((uint32_t)o32[i] != 0xEEEEEEEEu) { last = i; touched++; }
            printf("    writer extent: %u words touched, last at %u of %u"
                   " (the int8 surface would be %zu words)\n",
                   touched, last, words,
                   (size_t)((N + C2 - 1) / C2) * surf_px * C2 / 4);
        }
        printf("    %u/%u accumulators decoded uniquely, %u ambiguous; "
               "%u/%u fit the narrow map, %u/%u the wide one, %u outside its "
               "delivered set (first at word %u, next step %u)\n",
               found, M * N, collide, fits, found, fits_wide, found, off_set,
               first_idx, first_stride);
        /* The two axes of the map, printed raw. The lane slot within an atom and the
         * pixel stride are what name the writer's element width, and reading them off
         * beats inferring them from a fit count against one candidate map. */
        if (found && (fits != found || fits_wide != found ||
                      env_int("ROCKET_MP_VERBOSE", 0))) {
            unsigned c;
            unsigned lim = (unsigned)env_int("ROCKET_MP_AXIS", 40);
            printf("      channel axis (m=0):");
            for (c = 0; c < N && c < lim; c++) printf(" %d", at_of[c]);
            printf("\n      pixel axis (n=0):  ");
            for (c = 0; c < M && c < lim; c++) printf(" %d", at_of[(size_t)c * N]);
            printf("\n");
            /* Where the wide map and the part disagree, in ATOMS, which is the unit
             * the writer's runs are counted in. A drift that grows with the channel
             * super-group is a stride; one that grows with the pixel is a run length. */
            if (env_int("ROCKET_MP_DIFF", 0)) {
                unsigned mm, shown2 = 0;
                printf("      wide-map misses (c, p): got_atom want_atom\n");
                for (c = 0; c < N; c++)
                    for (mm = 0; mm < M; mm++) {
                        int g = at_of[(size_t)mm * N + c];
                        int wt = rocket_rk3576_i32_wide_word(iw, ih, c, mm);
                        if (g < 0 || wt < 0 || g == wt) continue;
                        if (shown2++ >= (unsigned)env_int("ROCKET_MP_DIFF", 0)) break;
                        printf("        c=%3u p=%3u: %5d %5d\n", c, mm, g / 4, wt / 4);
                    }
            }
        }
        /* The INVERSE map: what each written word holds. Two axes read at m=0 and n=0
         * name a separable layout and nothing else — a block base that shifts with the
         * pixel half, or a word the writer touches with no accumulator in it, only
         * shows up when every touched word is accounted for. `-` is a word the writer
         * wrote that no accumulator claims. */
        if (env_int("ROCKET_MP_DUMP", 0)) {
            unsigned i, lim = (unsigned)env_int("ROCKET_MP_DUMP", 0);
            int *inv = calloc(words, sizeof(int));
            if (inv) {
                for (i = 0; i < words; i++) inv[i] = -1;
                for (m = 0; m < M * N; m++)
                    if (at_of[m] >= 0) inv[at_of[m]] = (int)m;
                printf("      word -> C[m][n], by 16-byte atom:\n");
                for (i = 0; i < words && i / 4 < lim; i += 4) {
                    unsigned l;
                    printf("        atom %3u:", i / 4);
                    for (l = 0; l < 4; l++) {
                        if (inv[i + l] >= 0)
                            printf("  C[%u][%u]", (unsigned)inv[i + l] / N,
                                   (unsigned)inv[i + l] % N);
                        else
                            printf("  %s", (uint32_t)o32[i + l] == 0xEEEEEEEEu ? "  .   " : "  -   ");
                    }
                    printf("\n");
                }
                free(inv);
            }
        }
        printf("\n");
        free(at_of);
        rocket_bo_fini(fd, &o_bo);

        rocket_bo_free(fd, &r_bo); rocket_bo_free(fd, &o_bo); rocket_bo_free(fd, &b_bo);
        rocket_bo_free(fd, &w_bo); rocket_bo_free(fd, &in_bo);
        free(A); free(B); free(in_cube); free(w_cube); free(want);
    }
    free(parsed); free(vspec);
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "shape";
    const struct rocket_hw_profile *hw = rocket_hw_current();
    int fd, fails;

    if (strcmp(hw->name, "rk3576") != 0) {
        printf("rk3576_matmul_probe: not an RK3576 (profile %s) — skipping\n", hw->name);
        return 2;
    }
    fd = rocket_open();
    if (fd < 0) { printf("rk3576_matmul_probe: no /dev/accel device — skipping\n"); return 2; }

    printf("== rk3576 matmul probe: %s ==\n", mode);
    if      (!strcmp(mode, "shape")) fails = mode_shape(fd);
    else if (!strcmp(mode, "m"))     fails = mode_m(fd);
    else if (!strcmp(mode, "cost"))  fails = mode_cost(fd);
    else if (!strcmp(mode, "peak"))  fails = mode_peak(fd);
    else if (!strcmp(mode, "ncap"))  fails = mode_ncap(fd);
    else if (!strcmp(mode, "out32")) fails = mode_out32(fd);
    else { printf("modes: shape m cost peak ncap out32\n"); rocket_close(fd); return 2; }

    printf("== %s: %d failure%s ==\n", mode, fails, fails == 1 ? "" : "s");
    rocket_close(fd);
    return fails ? 1 : 0;
}
