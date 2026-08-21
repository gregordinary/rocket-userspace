// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rk3576_cost_table.c — what the RK3576 NPU costs, at shapes a model actually uses.
 *
 * There are otherwise almost no numbers on this part beyond the dispatch floor. This
 * puts the library entries beside a CPU reference at vision-stem and backbone shapes,
 * warm, and reports both wall times and the ratio.
 *
 * WHAT THE CPU COLUMN IS, so the ratio is read for what it is: a straightforward
 * single-threaded direct loop over the same tensors, built at the library's own
 * optimisation level and auto-vectorised, with no im2col, no packing and no threading.
 * It is a reference, not a tuned kernel — a good NEON GEMM with four threads would
 * close a large part of every ratio here. Read the NPU column as the cost of the
 * offload and the ratio as an upper bound against this baseline.
 *
 * WHAT THE NPU COLUMN IS: the whole public entry — host packing into the native cubes,
 * the submits, the de-scatter back to row-major — because that is what a frontend
 * pays. It is not a kernel time. Run 0 is discarded: the NPU clock parks at idle and a
 * cold call reads low, and on one path (below) a cold call is also the only one that
 * does not pay the retry.
 *
 * THE ONE THING TO KNOW BEFORE READING THE FP16 ROWS: the fp16 first conv poisons its
 * own next submit, and what clears it is the runtime-PM autosuspend cycling the NPU
 * power domain rather than elapsed time — with power/control=on it writes once and
 * never again, at any gap. So every fp16 first-conv call after the first pays the
 * library's retry-and-power-idle guard, about 122 ms, and that is most of its wall.
 * The int8 first conv has no such hazard: 20 of 20 submits write at a zero gap.
 *
 * Usage: rk3576_cost_table [substring]     (default: every row)
 * Env:   ROCKET_CT_REPS=<n>   timed repeats after the discarded first (default 3)
 * Exit:  0 ran, 2 no NPU or the wrong chip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "rocket_npu.h"
#include "rocket_conv.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"

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

/* ---- the CPU reference ---------------------------------------------------- */

static void cpu_conv_int8(const int8_t *in, const int8_t *W, int32_t *acc,
                          unsigned ic, unsigned oc, unsigned ih, unsigned iw,
                          unsigned k, unsigned stride, unsigned pad,
                          unsigned oh, unsigned ow, int dw)
{
    unsigned c, y, x, kh, kw, i;
    for (c = 0; c < oc; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                int32_t s = 0;
                for (kh = 0; kh < k; kh++) {
                    int iy = (int)(y * stride + kh) - (int)pad;
                    if (iy < 0 || iy >= (int)ih) continue;
                    for (kw = 0; kw < k; kw++) {
                        int ix = (int)(x * stride + kw) - (int)pad;
                        if (ix < 0 || ix >= (int)iw) continue;
                        if (dw) {
                            s += (int32_t)in[((size_t)c * ih + iy) * iw + ix] *
                                 W[((size_t)c * k + kh) * k + kw];
                        } else {
                            for (i = 0; i < ic; i++)
                                s += (int32_t)in[((size_t)i * ih + iy) * iw + ix] *
                                     W[(((size_t)c * ic + i) * k + kh) * k + kw];
                        }
                    }
                }
                acc[((size_t)c * oh + y) * ow + x] = s;
            }
}

/* C[M,N] = A[M,K] * B[N,K]^T — the library entry's own convention, so both columns
 * walk the same operands with the same locality. */
static void cpu_matmul_int8(const int8_t *A, const int8_t *B, int32_t *C,
                            unsigned M, unsigned K, unsigned N)
{
    unsigned m, n, kk;
    for (m = 0; m < M; m++)
        for (n = 0; n < N; n++) {
            int32_t s = 0;
            for (kk = 0; kk < K; kk++)
                s += (int32_t)A[(size_t)m * K + kk] * B[(size_t)n * K + kk];
            C[(size_t)m * N + n] = s;
        }
}

/* ---- rows ------------------------------------------------------------------ */

enum row_kind { ROW_FC_INT8, ROW_FC_FP16, ROW_CONV, ROW_DW, ROW_MM };

typedef struct {
    const char   *name;
    enum row_kind kind;
    unsigned      ic, oc, iw, ih, k, stride, pad;   /* conv    */
    unsigned      M, K, N;                          /* matmul  */
} row_t;

static const row_t ROWS[] = {
    /* The vision stem, both precisions, at the shape a 224x224 backbone opens with. */
    {"stem 3x224x224 k3s2 oc32",  ROW_FC_INT8, 3, 32, 224, 224, 3, 2, 1, 0,0,0},
    {"stem 3x224x224 k3s2 oc64",  ROW_FC_INT8, 3, 64, 224, 224, 3, 2, 1, 0,0,0},
    {"stem 3x224x224 k7s2 oc32",  ROW_FC_INT8, 3, 32, 224, 224, 7, 2, 3, 0,0,0},
    {"stem 3x224x224 k3s2 fp16",  ROW_FC_FP16, 3, 32, 224, 224, 3, 2, 1, 0,0,0},
    /* Backbone convolutions: the 1x1 and 3x3 a MobileNet/ResNet body is made of. */
    {"conv 32->64 k1 56x56",      ROW_CONV,  32,  64,  56, 56, 1, 1, 0, 0,0,0},
    {"conv 64->64 k3 56x56",      ROW_CONV,  64,  64,  56, 56, 3, 1, 1, 0,0,0},
    {"conv 128->128 k3 28x28",    ROW_CONV, 128, 128,  28, 28, 3, 1, 1, 0,0,0},
    {"conv 256->256 k3 14x14",    ROW_CONV, 256, 256,  14, 14, 3, 1, 1, 0,0,0},
    {"conv 256->512 k1 14x14",    ROW_CONV, 256, 512,  14, 14, 1, 1, 0, 0,0,0},
    /* Depthwise, the other half of a separable block. */
    {"dw 64 k3 56x56",            ROW_DW,    64,  64,  56, 56, 3, 1, 1, 0,0,0},
    {"dw 128 k3 28x28",           ROW_DW,   128, 128,  28, 28, 3, 1, 1, 0,0,0},
    {"dw 256 k3 14x14",           ROW_DW,   256, 256,  14, 14, 3, 1, 1, 0,0,0},
    /* The matmul, on the axis worth spending: N. */
    {"matmul 1x1024x1024",        ROW_MM, 0,0,0,0,0,0,0,    1, 1024, 1024},
    {"matmul 1x2048x1024",        ROW_MM, 0,0,0,0,0,0,0,    1, 2048, 1024},
    {"matmul 1x1536x1536",        ROW_MM, 0,0,0,0,0,0,0,    1, 1536, 1536},
    {"matmul 32x2048x1024",       ROW_MM, 0,0,0,0,0,0,0,   32, 2048, 1024},
    {"matmul 32x1024x1024",       ROW_MM, 0,0,0,0,0,0,0,   32, 1024, 1024},
    {"matmul 32x1024x2560",       ROW_MM, 0,0,0,0,0,0,0,   32, 1024, 2560},
    {"matmul 128x1024x1024",      ROW_MM, 0,0,0,0,0,0,0,  128, 1024, 1024},
    {"matmul 256x512x2048",       ROW_MM, 0,0,0,0,0,0,0,  256,  512, 2048},
};
#define N_ROWS ((int)(sizeof ROWS / sizeof ROWS[0]))

static void fill_i8(int8_t *p, size_t n, unsigned seed)
{
    size_t i;
    for (i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        p[i] = (int8_t)((int)((seed >> 16) & 0x1Fu) - 16);
    }
}

/* One conv row, both columns. Returns 0 ran, 1 the entry refused, 2 out of memory. */
static int run_conv(int fd, const row_t *r, int reps, double *npu, double *cpu)
{
    rocket_conv2d_desc d = {0};
    unsigned ow, oh, i;
    int8_t *in = NULL, *W = NULL, *out = NULL;
    int32_t *bias = NULL, *acc = NULL;
    _Float16 *fin = NULL, *fW = NULL, *fout = NULL;
    int fp16 = (r->kind == ROW_FC_FP16), dw = (r->kind == ROW_DW);
    int rc = 2;
    size_t nw;

    d.ic = (int)r->ic; d.oc = (int)r->oc; d.ih = (int)r->ih; d.iw = (int)r->iw;
    d.kh = (int)r->k;  d.kw = (int)r->k;
    d.stride_y = (int)r->stride; d.stride_x = (int)r->stride;
    d.pad_top = (int)r->pad; d.pad_left = (int)r->pad;
    d.dil_y = 1; d.dil_x = 1;
    d.depthwise = dw;
    ow = (unsigned)rocket_conv2d_ow(&d);
    oh = (unsigned)rocket_conv2d_oh(&d);
    nw = dw ? (size_t)r->ic * r->k * r->k : (size_t)r->oc * r->ic * r->k * r->k;

    if (fp16) {
        fin  = calloc((size_t)r->ic * r->ih * r->iw, sizeof *fin);
        fW   = calloc(nw, sizeof *fW);
        fout = calloc((size_t)r->oc * oh * ow, sizeof *fout);
        if (!fin || !fW || !fout) goto done;
        for (i = 0; i < r->ic * r->ih * r->iw; i++) fin[i] = (_Float16)((i % 17) - 8);
        for (i = 0; i < (unsigned)nw; i++) fW[i] = (_Float16)((int)(i % 9) - 4);
    } else {
        in   = calloc((size_t)r->ic * r->ih * r->iw, 1);
        W    = calloc(nw, 1);
        out  = calloc((size_t)r->oc * oh * ow, 1);
        bias = calloc(r->oc, sizeof *bias);
        acc  = calloc((size_t)r->oc * oh * ow, sizeof *acc);
        if (!in || !W || !out || !bias || !acc) goto done;
        fill_i8(in, (size_t)r->ic * r->ih * r->iw, 1u);
        fill_i8(W, nw, 7u);
    }

    /* The NPU column, warm: run 0 is discarded. */
    for (i = 0; i < (unsigned)reps + 1u; i++) {
        double t0;
        int e;
        if (i == 1) *npu = 0.0;
        t0 = now_ms();
        if (fp16)     e = rocket_conv2d_fp16_rk3576(fd, &d, fin, fW, fout);
        else if (dw)  e = rocket_conv2d_dw_int8_rk3576(fd, &d, in, W, bias,
                                                       1.0f, 1.0f, 256.0f, 0, 0, 0, out);
        else          e = rocket_conv2d_int8_rk3576(fd, &d, in, W, bias,
                                                    1.0f, 1.0f, 256.0f, 0, 0, 0, out);
        if (e != ROCKET_OK) { rc = 1; goto done; }
        if (i) *npu += now_ms() - t0;
    }
    *npu /= reps;

    /* The CPU column. fp16 rows are timed against the int8 reference at the same
     * shape — the arithmetic differs, the work does not, and running an fp16 loop
     * here would compare two references rather than one. */
    if (!fp16) {
        double t0 = now_ms();
        for (i = 0; i < (unsigned)reps; i++)
            cpu_conv_int8(in, W, acc, r->ic, r->oc, r->ih, r->iw, r->k, r->stride,
                          r->pad, oh, ow, dw);
        *cpu = (now_ms() - t0) / reps;
    } else {
        *cpu = 0.0;
    }
    rc = 0;
done:
    free(in); free(W); free(out); free(bias); free(acc);
    free(fin); free(fW); free(fout);
    return rc;
}

static int run_mm(int fd, const row_t *r, int reps, double *npu, double *cpu)
{
    int8_t *A = calloc((size_t)r->M * r->K, 1);
    int8_t *B = calloc((size_t)r->K * r->N, 1);
    int8_t *C = calloc((size_t)r->M * r->N, 1);
    int32_t *ref = calloc((size_t)r->M * r->N, sizeof *ref);
    unsigned i;
    int rc = 2;

    if (!A || !B || !C || !ref) goto done;
    fill_i8(A, (size_t)r->M * r->K, 3u);
    fill_i8(B, (size_t)r->K * r->N, 11u);

    for (i = 0; i < (unsigned)reps + 1u; i++) {
        double t0;
        int e;
        if (i == 1) *npu = 0.0;
        t0 = now_ms();
        e = rocket_matmul_int8_rk3576(fd, (int)r->M, (int)r->K, (int)r->N, A, B,
                                      NULL, 1.0f / 4096.0f, C);
        if (e != ROCKET_OK) { rc = 1; goto done; }
        if (i) *npu += now_ms() - t0;
    }
    *npu /= reps;

    {
        double t0 = now_ms();
        for (i = 0; i < (unsigned)reps; i++)
            cpu_matmul_int8(A, B, ref, r->M, r->K, r->N);
        *cpu = (now_ms() - t0) / reps;
    }
    rc = 0;
done:
    free(A); free(B); free(C); free(ref);
    return rc;
}

int main(int argc, char **argv)
{
    const char *filter = argc > 1 ? argv[1] : NULL;
    int reps = env_int("ROCKET_CT_REPS", 3);
    int fd, i;

    fd = rocket_open();
    if (fd < 0) { printf("no /dev/accel/accel0 (%d) — SKIP\n", fd); return 2; }
    {
        const struct rocket_hw_profile *hw = rocket_hw_current();
        if (!hw || !hw->name || strcmp(hw->name, "rk3576")) {
            printf("chip is '%s', not rk3576 — SKIP\n", hw && hw->name ? hw->name : "?");
            rocket_close(fd);
            return 2;
        }
    }

    printf("RK3576 cost table — %d timed repeats after a discarded first, whole public\n"
           "entries (host packing + submits + de-scatter) against a single-threaded\n"
           "scalar CPU reference. The CPU column is a reference, not a tuned kernel.\n\n",
           reps);
    printf("  %-28s %10s %10s %8s\n", "shape", "NPU ms", "CPU ms", "ratio");
    printf("  %-28s %10s %10s %8s\n", "----------------------------", "----------",
           "----------", "--------");

    for (i = 0; i < N_ROWS; i++) {
        const row_t *r = &ROWS[i];
        double npu = 0.0, cpu = 0.0;
        int rc;
        if (filter && !strstr(r->name, filter)) continue;
        rc = (r->kind == ROW_MM) ? run_mm(fd, r, reps, &npu, &cpu)
                                 : run_conv(fd, r, reps, &npu, &cpu);
        if (rc == 1) { printf("  %-28s %10s\n", r->name, "refused"); continue; }
        if (rc == 2) { printf("  %-28s %10s\n", r->name, "no memory"); continue; }
        if (cpu > 0.0)
            printf("  %-28s %10.2f %10.2f %7.2fx\n", r->name, npu, cpu, cpu / npu);
        else
            printf("  %-28s %10.2f %10s %8s\n", r->name, npu, "-", "-");
    }

    rocket_close(fd);
    return 0;
}
