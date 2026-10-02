// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rowmajor_tile_probe.c — the row-major matmul form at the shapes a tiled path needs.
 *
 * A probe, not a gate. rowmajor_matmul_probe showed
 * one fp16 -> fp32 task reading A as plain [M][K] rows and writing C as plain [M][N] rows. A
 * tiled path needs more than that: a task reads a K slice out of a wider A, writes an N tile
 * into a wider C, and runs int8 and the fp16-out K-accumulation program too. Each arm below
 * changes one of those axes, is one single-task job (two for the K-accumulation), and is
 * scored per element against a CPU reference. Every byte of the output BO is prefilled with a
 * sentinel, so an element the task did not write, or wrote outside its tile, is counted.
 *
 * The register model the arms test. On the input side, CNA CONV_CON1 bit 29 (GROUP_LINE_OFF),
 * DMA_CON1 = one row of the FULL A in 16-byte units, DMA_CON2 = 0, and the feature address
 * offset to (m0, k0). On the output side, DPU DST_SURF_STRIDE = one 16-byte atom, both halves
 * of DATA_CUBE_NOTCH = one row of the FULL C in atoms, minus one, SURFACE_ADD = one output
 * group's bytes in atoms (a group is 16 channels for fp16 in and 32 for int8 in, so 4 atoms
 * at fp32 out, 2 at fp16 out, 8 at int32 out), and the destination offset to (m0, n0).
 *
 * Arms (every one prints wrong / unwritten-in-tile / written-outside-tile):
 *   cube_*     the shipped program, A scattered and C de-scattered: the control that must pass
 *   kslice     a K slice read out of a wider A, C plain
 *   ntile      an N tile written into a wider C, at a row offset, A plain
 *   tile       both at once, the task a tiled path issues
 *   i8_*       int8 in, int32 out, plain and tiled
 *   f16o_*     fp16 out, plain and tiled
 *   f16o_ab    fp16 out under allbilly's geometry (notch at 4 bytes a channel, SURFACE_ADD 4),
 *              scored under the gapped layout their generator reads back: a reproduction arm
 *              for the SURFACE_ADD model, expected exact there and nowhere else
 *   kacc_*     two tasks, the second adding the first's fp16 partial through the DPU_RDMA:
 *              the partial is a CUBE (device-internal, never read by the host), the final
 *              write is row-major into a wider C. kacc_cube is the shipped program.
 *
 * Operands are integers: fp16 x fp16 -> fp32 in [-3, 3], every fp16-out and K-accumulation
 * arm in [-1, 1] with K <= 2048 so every partial sum is exact in fp16, int8 over its full
 * range. So equality is the test at every element.
 *
 * Then a timing pass at the library's own tile shapes (256x384x256 symmetric, 256x512x128
 * asymmetric, and int8 at 256x512x256): the cube program against the row-major form on the input
 * side only, the output side only, both at the tile's own width, and both inside a 4096-wide A
 * and C, interleaved with the order rotated, medians of submit-to-fence. Both arms get BOs of the same sizes, and the timed wait is on a 4 KiB BO
 * the job lists as an output, so the CPU sync PREP_BO runs over the BO it waits on is the same
 * small one in both. Device time plus a fixed submit cost; not a price of the host work removed.
 *
 * What a green result does NOT show: a multi-task or chained job, a multi-core job, resident
 * weights, the row-major K-accumulation partial (not tested: the partial never leaves the
 * device), int4, int16, bf16 or tf32, or what the form is worth to a model.
 *
 * --wait-out times the wait on the output BO, so its CPU sync, and --unequal gives each timed arm
 * BOs at its own layout's sizes (the cube arm's output is then 16x smaller than the widest):
 * together they reproduce a timing that reads a BO sync as device time.
 *
 * --pitch runs only a sweep of the row pitch on each side alone, at 256x384x256 fp16 -> fp32
 * (--pitch-asym the same at 256x512x128);
 * --pitch2 H the output side at pitches on and 16 bytes off a power of two, plus the 16 KiB
 * pitch with only NOTCH_ADDR_0 (H bit 0) or NOTCH_ADDR_1 (H bit 1) programmed.
 *
 * Usage: rowmajor_tile_probe [--no-timing|--timing-only|--pitch|--pitch-asym|--pitch2 H] [--wait-out]
 *        [--unequal] [--reps N]
 * Exit: 0 every arm exact and every control exact, 1 otherwise, 2 no NPU.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_hw.h"
#include "test_fill.h"

enum dtype { F16F32, F16F16, I8I32 };

static int in_size(int dt)  { return dt == I8I32 ? 1 : 2; }
static int out_size(int dt) { return dt == F16F16 ? 2 : 4; }
static int out_group(int dt) { return dt == I8I32 ? 32 : 16; }   /* output channels a CACC group */

struct arm {
    const char *name;
    int dt;
    int M, Kt, N;          /* the task                                           */
    int Mf, Kf, Nf;        /* A is Mf x Kf, C is Mf x Nf (row-major arms)        */
    int m0, k0, n0;        /* the task's origin in them                          */
    int rm_in, rm_out;     /* 1 row-major, 0 the cube program's layout           */
    int kacc;              /* 1: two K slices, the second adds the first's partial */
    int notch, sadd;       /* 0: the model's value; else an override (atoms)     */
    int abmap;             /* score under allbilly's gapped fp16 layout          */
    int nhalf;             /* 0 both notch halves; 1 NOTCH_ADDR_0 only; 2 NOTCH_ADDR_1 only */
};

static const struct arm arms[] = {
    /*  name        dt      M   Kt    N    Mf    Kf    Nf   m0   k0   n0 rin rout kacc */
    { "cube_f32",  F16F32,  64, 384,  96,   64,  384,   96,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0 },
    { "kslice",    F16F32,  64, 384,  96,   64, 1024,   96,   0, 640,   0, 1, 1, 0, 0, 0, 0, 0 },
    { "kslice2",   F16F32,  36,  96, 160,   36,  320,  160,   0, 224,   0, 1, 1, 0, 0, 0, 0, 0 },
    { "ntile",     F16F32,  64, 256,  96,  100,  256,  512,   8,   0, 224, 1, 1, 0, 0, 0, 0, 0 },
    { "ntile2",    F16F32,  48, 128,  64,   48,  128, 2048,   0,   0,1984, 1, 1, 0, 0, 0, 0, 0 },
    { "tile",      F16F32, 100, 192, 128,  128,  768,  640,  20, 384, 320, 1, 1, 0, 0, 0, 0, 0 },
    { "kslice_cubeout", F16F32, 64, 384, 96, 64, 1024, 96,   0, 640,   0, 1, 0, 0, 0, 0, 0, 0 },
    { "cube_i8",   I8I32,   64, 256,  96,   64,  256,   96,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0 },
    { "i8_plain",  I8I32,   64, 256,  96,   64,  256,   96,   0,   0,   0, 1, 1, 0, 0, 0, 0, 0 },
    { "i8_tile",   I8I32,  100, 192, 128,  128,  768,  640,  20, 384, 320, 1, 1, 0, 0, 0, 0, 0 },
    { "cube_f16o", F16F16,  64, 256,  96,   64,  256,   96,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0 },
    { "f16o_plain",F16F16,  64, 256,  96,   64,  256,   96,   0,   0,   0, 1, 1, 0, 0, 0, 0, 0 },
    { "f16o_tile", F16F16, 100, 192, 128,  128,  768,  640,  20, 384, 320, 1, 1, 0, 0, 0, 0, 0 },
    { "f16o_ab",   F16F16,  64, 256,  96,   64,  256,   96,   0,   0,   0, 1, 1, 0,
      .notch = 96 / 4 - 1, .sadd = 4, .abmap = 1 },
    { "kacc_cube", F16F16,  64, 192,  96,   64,  384,   96,   0,   0,   0, 0, 0, 1, 0, 0, 0, 0 },
    { "kacc_rm",   F16F16,  64, 192,  96,   64,  384,  256,   0,   0, 128, 1, 1, 1, 0, 0, 0, 0 },
    { "kacc_rm_big",F16F16,256, 384, 256,  256,  768, 1024,   0,   0, 512, 1, 1, 1, 0, 0, 0, 0 },
    { "ntile_lo",  F16F32,  64, 256,  96,  100,  256,  512,   8,   0, 224, 1, 1, 0, 0, 0, 0, 1 },
    { "ntile_hi",  F16F32,  64, 256,  96,  100,  256,  512,   8,   0, 224, 1, 1, 0, 0, 0, 0, 2 },
    { "ntile_odd", F16F32,  64, 256,  96,  100,  256,  516,   8,   0, 224, 1, 1, 0, 0, 0, 0, 0 },
};

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* Rewrite one register's value in a generated program; OR it in when `orbits` is set.
 * Returns 0, or -1 when the program never writes that register. */
static int patch(uint64_t *ops, int n, uint16_t target, uint16_t reg, uint32_t value, int orbits)
{
    for (int i = 0; i < n; i++) {
        if ((uint16_t)(ops[i] >> 48) != target || (uint16_t)(ops[i] & 0xFFFF) != reg) continue;
        uint32_t v = orbits ? (uint32_t)(ops[i] >> 16) | value : value;
        ops[i] = NPUOP(target, v, reg);
        return 0;
    }
    return -1;
}

/* Host operands and the per-slice references, for one arm. */
struct host {
    void *A, *B;                 /* A Mf x Kf, B Nf x Kf (B[n][k]), fp16 or int8 */
    double *want, *want0, *want1; /* M x N: every slice, the first only, the last only */
};

static double a_at(const struct arm *a, const struct host *h, int m, int k)
{
    return a->dt == I8I32 ? (double)((int8_t *)h->A)[(size_t)m * a->Kf + k]
                          : (double)((_Float16 *)h->A)[(size_t)m * a->Kf + k];
}

static double b_at(const struct arm *a, const struct host *h, int n, int k)
{
    return a->dt == I8I32 ? (double)((int8_t *)h->B)[(size_t)n * a->Kf + k]
                          : (double)((_Float16 *)h->B)[(size_t)n * a->Kf + k];
}

static int host_init(const struct arm *a, struct host *h, uint64_t seed)
{
    const size_t na = (size_t)a->Mf * a->Kf, nb = (size_t)a->Nf * a->Kf;
    const int nsl = a->kacc ? 2 : 1;
    h->A = malloc(na * in_size(a->dt));
    h->B = malloc(nb * in_size(a->dt));
    h->want = malloc((size_t)a->M * a->N * sizeof(double));
    h->want0 = malloc((size_t)a->M * a->N * sizeof(double));
    h->want1 = malloc((size_t)a->M * a->N * sizeof(double));
    if (!h->A || !h->B || !h->want || !h->want0 || !h->want1) return -1;
    if (a->dt == I8I32) {
        tf_fill_i8(h->A, na, seed, -128, 127);
        tf_fill_i8(h->B, nb, seed + 1, -128, 127);
    } else {
        const int r = (a->dt == F16F16 || a->kacc) ? 1 : 3;
        tf_fill_f16_int(h->A, na, seed, -r, r);
        tf_fill_f16_int(h->B, nb, seed + 1, -r, r);
    }
    for (int m = 0; m < a->M; m++)
        for (int n = 0; n < a->N; n++) {
            double s[2] = { 0, 0 };
            for (int sl = 0; sl < nsl; sl++)
                for (int k = 0; k < a->Kt; k++) {
                    const int kk = a->k0 + sl * a->Kt + k;
                    s[sl] += a_at(a, h, a->m0 + m, kk) * b_at(a, h, a->n0 + n, kk);
                }
            h->want[(size_t)m * a->N + n] = s[0] + s[1];
            h->want0[(size_t)m * a->N + n] = s[0];
            h->want1[(size_t)m * a->N + n] = s[1];
        }
    return 0;
}

static void host_free(struct host *h)
{
    free(h->A); free(h->B); free(h->want); free(h->want0); free(h->want1);
}

/* `fence` is a 4 KiB BO listed as a second output of every job and never written. A timed
 * wait goes on it: PREP_BO syncs the whole BO it waits on for the CPU, so waiting on a
 * multi-megabyte output times a cache clean and invalidate as well as the job. */
struct bufs { rocket_bo rc, in, w, part, out, fence; };

static int g_wait_out;   /* --wait-out: time the wait on the output BO instead, sync included */
static int g_unequal;    /* --unequal: each timed arm's BOs at its own layout's sizes */

static size_t out_bytes(const struct arm *a)
{
    if (!a->rm_out) return (size_t)a->M * a->N * out_size(a->dt);
    return (size_t)a->Mf * a->Nf * (a->abmap ? 4 : out_size(a->dt));
}

static size_t in_bytes(const struct arm *a)
{
    const int nsl = a->kacc ? 2 : 1, isz = in_size(a->dt);
    return a->rm_in ? (size_t)a->Mf * a->Kf * isz : (size_t)nsl * a->M * a->Kt * isz;
}

/* Allocate an arm's BOs, the input and output at least min_in and min_out bytes, so timed arms
 * pay the same per-BO costs whatever their layouts need. */
static int bufs_alloc(int fd, const struct arm *a, struct bufs *b, size_t min_in, size_t min_out)
{
    const int nsl = a->kacc ? 2 : 1, isz = in_size(a->dt);
    size_t in_b = in_bytes(a), out_b = out_bytes(a);
    if (min_in > in_b) in_b = min_in;
    if (min_out > out_b) out_b = min_out;
    return rocket_bo_alloc32(fd, 4096, &b->rc) ||
           rocket_bo_alloc32(fd, in_b, &b->in) ||
           rocket_bo_alloc32(fd, (size_t)nsl * a->N * a->Kt * isz, &b->w) ||
           rocket_bo_alloc32(fd, (size_t)a->M * a->N * 2 + 64, &b->part) ||
           rocket_bo_alloc32(fd, out_b, &b->out) ||
           rocket_bo_alloc32(fd, 4096, &b->fence);
}

static void bufs_free(int fd, struct bufs *b)
{
    rocket_bo_free(fd, &b->rc); rocket_bo_free(fd, &b->in); rocket_bo_free(fd, &b->w);
    rocket_bo_free(fd, &b->part); rocket_bo_free(fd, &b->out); rocket_bo_free(fd, &b->fence);
}

/* Stage A and the weights. Row-major A goes in whole; a cube arm scatters each K slice of its
 * tile into its own slot. The weights are always the packed tile of each slice. */
static void stage_inputs(int fd, const struct arm *a, const struct host *h, struct bufs *b)
{
    const int nsl = a->kacc ? 2 : 1, isz = in_size(a->dt), c2 = a->dt == I8I32 ? 16 : 8;
    rocket_bo_prep(fd, &b->in, 1, 0);
    rocket_bo_prep(fd, &b->w, 1, 0);
    memset(b->in.ptr, 0, b->in.size);
    memset(b->w.ptr, 0, b->w.size);
    if (a->rm_in) {
        memcpy(b->in.ptr, h->A, (size_t)a->Mf * a->Kf * isz);
    } else {
        for (int sl = 0; sl < nsl; sl++) {
            const size_t slot = (size_t)sl * a->M * a->Kt;
            for (int m = 1; m <= a->M; m++)
                for (int k = 1; k <= a->Kt; k++) {
                    const size_t src = (size_t)(a->m0 + m - 1) * a->Kf + a->k0 + sl * a->Kt + k - 1;
                    const size_t dst = slot + (size_t)feature_data(a->Kt, a->M, 1, c2, k, m, 1);
                    if (a->dt == I8I32) ((int8_t *)b->in.ptr)[dst] = ((int8_t *)h->A)[src];
                    else ((_Float16 *)b->in.ptr)[dst] = ((_Float16 *)h->A)[src];
                }
        }
    }
    for (int sl = 0; sl < nsl; sl++) {
        const size_t slot = (size_t)sl * a->N * a->Kt;
        for (int n = 1; n <= a->N; n++)
            for (int k = 1; k <= a->Kt; k++) {
                const size_t src = (size_t)(a->n0 + n - 1) * a->Kf + a->k0 + sl * a->Kt + k - 1;
                if (a->dt == I8I32)
                    ((int8_t *)b->w.ptr)[slot + (size_t)weight_int8(a->Kt, n, k)] = ((int8_t *)h->B)[src];
                else
                    ((_Float16 *)b->w.ptr)[slot + (size_t)weight_fp16(a->Kt, n, k)] = ((_Float16 *)h->B)[src];
            }
    }
    rocket_bo_fini(fd, &b->in);
    rocket_bo_fini(fd, &b->w);
}

/* Emit and submit slice `sl` of the arm, then wait for its fence. Returns 0, or -1. */
static int run_task(int fd, const struct arm *a, struct bufs *b, int sl, double *ms, int *slow)
{
    const int nsl = a->kacc ? 2 : 1, isz = in_size(a->dt), osz = out_size(a->dt);
    const int final = (sl == nsl - 1);
    uint64_t ops[256] = { 0 };
    uint32_t in_dma = (uint32_t)b->in.dma_address +
        (uint32_t)(a->rm_in ? ((size_t)a->m0 * a->Kf + a->k0 + (size_t)sl * a->Kt) * isz
                            : (size_t)sl * a->M * a->Kt * isz);
    uint32_t out_dma = final
        ? (uint32_t)b->out.dma_address + (uint32_t)(a->rm_out ? ((size_t)a->m0 * a->Nf + a->n0) * osz : 0)
        : (uint32_t)b->part.dma_address;
    matmul_params_t p = {
        .m = (uint16_t)a->M, .k = (uint16_t)a->Kt, .n = (uint16_t)a->N,
        .input_dma = in_dma,
        .weights_dma = (uint32_t)b->w.dma_address + (uint32_t)((size_t)sl * a->N * a->Kt * isz),
        .output_dma = out_dma, .tasks = ops,
        .fp32tofp16 = (a->dt == F16F16) ? 1 : 0,
        .accumulate = (uint8_t)(sl > 0), .add_dma = sl > 0 ? (uint32_t)b->part.dma_address : 0,
    };
    int rc = a->dt == I8I32 ? gen_matmul_int8(&p) : gen_matmul_fp16(&p);
    if (rc != 0) { printf("  %s: generator refused (%d)\n", a->name, rc); return -1; }
    const int n = (int)p.task_count;
    if (a->rm_in &&
        (patch(ops, n, OP_REG_CNA, CNA_CONV_CON1, 1u << 29, 1) ||
         patch(ops, n, OP_REG_CNA, CNA_DMA_CON1, (uint32_t)(a->Kf * isz / 16), 0) ||
         patch(ops, n, OP_REG_CNA, CNA_DMA_CON2, 0, 0))) {
        printf("  %s: an input register to patch is missing\n", a->name); return -1;
    }
    if (final && a->rm_out) {
        const uint32_t notch = a->notch ? (uint32_t)a->notch : (uint32_t)(a->Nf * osz / 16 - 1);
        const uint32_t sadd = a->sadd ? (uint32_t)a->sadd : (uint32_t)(out_group(a->dt) * osz / 16);
        if (patch(ops, n, OP_REG_DPU, DPU_DST_SURF_STRIDE, 1u << 4, 0) ||
            patch(ops, n, OP_REG_DPU, DPU_DATA_CUBE_NOTCH_ADDR,
                  (a->nhalf == 1 ? 0 : notch << 16) | (a->nhalf == 2 ? 0 : notch), 0) ||
            patch(ops, n, OP_REG_DPU, DPU_SURFACE_ADD, sadd << 4, 0)) {
            printf("  %s: an output register to patch is missing\n", a->name); return -1;
        }
    }
    rocket_bo_prep(fd, &b->rc, 1, 0);
    memcpy(b->rc.ptr, ops, (size_t)n * sizeof(uint64_t));
    rocket_bo_fini(fd, &b->rc);

    rocket_bo *dst = final ? &b->out : &b->part;
    uint32_t in_h[4] = { b->in.handle, b->w.handle, b->rc.handle, b->part.handle };
    uint32_t out_h[2] = { dst->handle, b->fence.handle };
    const uint32_t n_in = (sl > 0) ? 4 : 3;   /* a partial read is an input, never both lists */
    uint64_t slow0 = rocket_fence_wait_slow_count();
    double t0 = now_ms();
    rc = rocket_submit_matmul(fd, &b->rc, (uint32_t)n, in_h, n_in, out_h, 2, 6000);
    if (rc == 0) rc = rocket_bo_prep(fd, g_wait_out ? dst : &b->fence, 0, 2000000000ull);
    *ms = now_ms() - t0;
    if (rc == 0) rocket_bo_fini(fd, &b->fence);
    if (rc == 0) rc = rocket_bo_prep(fd, dst, 0, 2000000000ull);
    if (rc == 0) rocket_bo_fini(fd, dst);
    *slow |= rocket_fence_wait_slow_count() != slow0;
    if (rc != 0) { printf("  %s: slice %d submit or wait failed (%d)\n", a->name, sl, rc); return -1; }
    return 0;
}

/* The element (m, n) of the tile, as an offset in output elements from the BO's start. */
static size_t out_index(const struct arm *a, int m, int n)
{
    if (!a->rm_out) return (size_t)feature_data(a->N, a->M, 1, 16 / out_size(a->dt), n + 1, m + 1, 1);
    if (a->abmap) return (size_t)(a->m0 + m) * (2 * a->Nf) + (size_t)((a->n0 + n) / 16) * 32 + (a->n0 + n) % 16;
    return (size_t)(a->m0 + m) * a->Nf + a->n0 + n;
}

static double out_value(const struct arm *a, const void *o, size_t at)
{
    switch (a->dt) {
    case F16F32: return ((const float *)o)[at];
    case F16F16: return ((const _Float16 *)o)[at];
    default:     return ((const int32_t *)o)[at];
    }
}

static int is_sentinel(const void *o, size_t at, int osz)
{
    const uint8_t *p = (const uint8_t *)o + at * osz;
    for (int i = 0; i < osz; i++) if (p[i] != TF_SENTINEL_BYTE) return 0;
    return 1;
}

/* Run an arm once; score it unless `quiet`. Returns 0 exact, 1 wrong, -1 not run. */
static int run_arm(int fd, const struct arm *a, const struct host *h, struct bufs *b, int quiet,
                   double *ms_last)
{
    const int nsl = a->kacc ? 2 : 1, osz = out_size(a->dt);
    int slow = 0;
    double ms = 0;
    stage_inputs(fd, a, h, b);
    if (tf_sentinel_bo(fd, &b->out) || tf_sentinel_bo(fd, &b->part)) return -1;
    for (int sl = 0; sl < nsl; sl++)
        if (run_task(fd, a, b, sl, &ms, &slow) != 0) return -1;
    if (ms_last) *ms_last = ms;
    if (quiet) return 0;

    rocket_bo_prep(fd, &b->out, 0, 0);
    const void *o = b->out.ptr;
    const size_t total = out_bytes(a) / osz;
    uint8_t *in_tile = calloc(total, 1);
    long wrong = 0, unwritten = 0, outside = 0, first = -1, eq0 = 0, eq1 = 0;
    for (int m = 0; m < a->M; m++)
        for (int n = 0; n < a->N; n++) {
            const size_t at = out_index(a, m, n), w = (size_t)m * a->N + n;
            if (at >= total) { wrong++; continue; }
            in_tile[at] = 1;
            const double got = out_value(a, o, at);
            if (is_sentinel(o, at, osz)) unwritten++;
            if (got != h->want[w]) { if (first < 0) first = (long)w; wrong++; }
            if (got == h->want0[w]) eq0++;
            if (got == h->want1[w]) eq1++;
        }
    for (size_t at = 0; at < total; at++)
        if (!in_tile[at] && !is_sentinel(o, at, osz)) outside++;
    rocket_bo_fini(fd, &b->out);
    free(in_tile);

    printf("  %-14s %s %3dx%4dx%4d (A %dx%d @%d,%d  C %dx%d @%d,%d): %6ld of %6d wrong, "
           "%6ld unwritten, %6ld outside%s", a->name,
           a->dt == I8I32 ? "i8>i32" : a->dt == F16F16 ? "f16>f16" : "f16>f32",
           a->M, a->Kt * nsl, a->N, a->Mf, a->Kf, a->m0, a->k0, a->Mf, a->Nf, a->m0, a->n0,
           wrong, a->M * a->N, unwritten, outside, slow ? " [SLOW FENCE]" : "");
    if (a->kacc) printf(", =first slice %ld, =last slice %ld", eq0, eq1);
    if (wrong) printf(", first (m %ld, n %ld)", first / a->N, first % a->N);
    printf("\n");
    return (wrong || outside || slow) ? 1 : 0;
}

static int cmp_d(const void *x, const void *y)
{
    const double p = *(const double *)x, q = *(const double *)y;
    return p < q ? -1 : p > q;
}

/* Time a set of programs at one tile shape, each scored once first, interleaved with the order
 * rotated every rep. set[0] is the reference (the cube program). Prints each arm's median
 * submit-to-fence and the median of its per-rep ratio to set[0], with the quartiles. */
#define MAX_TIMED 16
static int time_arms(int fd, const struct arm *set, int na, int reps, uint64_t seed)
{
    struct host h[MAX_TIMED];
    struct bufs b[MAX_TIMED];
    double *t[MAX_TIMED], *r[MAX_TIMED];
    size_t min_in = 0, min_out = 0;
    int bad = 0;
    if (na > MAX_TIMED) return 1;
    memset(h, 0, sizeof h);
    memset(b, 0, sizeof b);
    for (int i = 0; i < na; i++) {
        t[i] = calloc((size_t)reps, sizeof(double));
        r[i] = calloc((size_t)reps, sizeof(double));
        if (!g_unequal && in_bytes(&set[i]) > min_in) min_in = in_bytes(&set[i]);
        if (!g_unequal && out_bytes(&set[i]) > min_out) min_out = out_bytes(&set[i]);
    }
    for (int i = 0; i < na && !bad; i++) {
        if (!t[i] || !r[i] || host_init(&set[i], &h[i], seed + (uint64_t)i) ||
            bufs_alloc(fd, &set[i], &b[i], min_in, min_out)) {
            printf("  timing %s: allocation failed\n", set[i].name); bad = 1; break;
        }
        if (run_arm(fd, &set[i], &h[i], &b[i], 0, NULL) != 0) bad = 1;
    }
    for (int rep = 0; rep < reps && !bad; rep++)
        for (int j = 0; j < na; j++) {
            const int i = (j + rep) % na;            /* rotate which arm goes first */
            if (run_arm(fd, &set[i], &h[i], &b[i], 1, &t[i][rep]) != 0) { bad = 1; break; }
        }
    if (!bad) {
        const struct arm *c = &set[0];
        for (int rep = 0; rep < reps; rep++)
            for (int i = 0; i < na; i++) r[i][rep] = t[i][rep] / t[0][rep];
        printf("  timing %s %3dx%4dx%4d, %d rotated reps, submit to fence, median ms (ratio to %s, "
               "median of per-rep ratios, quartiles):\n",
               c->dt == I8I32 ? "i8>i32" : c->dt == F16F16 ? "f16>f16" : "f16>f32",
               c->M, c->Kt, c->N, reps, c->name);
        for (int i = 0; i < na; i++) {
            qsort(t[i], (size_t)reps, sizeof(double), cmp_d);
            qsort(r[i], (size_t)reps, sizeof(double), cmp_d);
            printf("    %-14s A pitch %5d  C pitch %5d  %.4f ms (%.4f-%.4f)  x%.3f (%.3f-%.3f)\n",
                   set[i].name, set[i].rm_in ? set[i].Kf : 0, set[i].rm_out ? set[i].Nf : 0,
                   t[i][reps / 2], t[i][0], t[i][reps - 1], r[i][reps / 2], r[i][reps / 4],
                   r[i][(3 * reps) / 4]);
        }
    }
    for (int i = 0; i < na; i++) { bufs_free(fd, &b[i]); host_free(&h[i]); free(t[i]); free(r[i]); }
    return bad;
}

/* Five programs at one of the library's tile shapes: the cube program, then the row-major form
 * on the input side only, on the output side only, on both at the tile's own width (A Mt x Kt,
 * C Mt x Nt), and on both inside a 4096-wide A and C. So the added time splits into the read,
 * the write, and the row pitch. Pitches are in elements. */
static int time_set(int fd, int dt, int M, int Kt, int N, int reps, uint64_t seed)
{
    const int W = 4096;
    const struct arm set[] = {
        { "t_cube",      dt, M, Kt, N, M, Kt, N, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0 },
        { "t_in_wide",   dt, M, Kt, N, M, W,  N, 0, Kt, 0, 1, 0, 0, 0, 0, 0, 0 },
        { "t_out_wide",  dt, M, Kt, N, M, Kt, W, 0, 0,  N, 0, 1, 0, 0, 0, 0, 0 },
        { "t_both_own",  dt, M, Kt, N, M, Kt, N, 0, 0,  0, 1, 1, 0, 0, 0, 0, 0 },
        { "t_both_wide", dt, M, Kt, N, M, W,  W, 0, Kt, N, 1, 1, 0, 0, 0, 0, 0 },
    };
    return time_arms(fd, set, (int)(sizeof(set) / sizeof(set[0])), reps, seed);
}

/* The pitch on its own, fp16 -> fp32 at 256x384x256: the output side alone at C pitches from the
 * tile's own 256 to 8192 elements (1 KiB to 32 KiB a row), then the input side alone at A pitches
 * from its own 384 to 8192 (768 B to 16 KiB). Says whether the added time grows with the pitch
 * or steps once each row sits in its own page. */
static int time_pitch(int fd, int reps, uint64_t seed, int M, int Kt, int N)
{
    const int dt = F16F32;
    struct arm set[MAX_TIMED];
    int na = 0;
    set[na++] = (struct arm){ "t_cube", dt, M, Kt, N, M, Kt, N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    static const int cp[] = { 256, 512, 1024, 2048, 4096, 8192 };
    const int ap[] = { Kt, 1024, 2048, 4096, 8192 };
    for (unsigned i = 0; i < sizeof(cp) / sizeof(cp[0]); i++) {
        if (cp[i] < N) continue;
        set[na++] = (struct arm){ "t_out_pitch", dt, M, Kt, N, M, Kt, cp[i], 0, 0,
                                  cp[i] > N ? N : 0, 0, 1, 0, 0, 0, 0, 0 };
    }
    for (unsigned i = 0; i < sizeof(ap) / sizeof(ap[0]); i++)
        set[na++] = (struct arm){ "t_in_pitch", dt, M, Kt, N, M, ap[i], N, 0,
                                  ap[i] > Kt ? Kt : 0, 0, 1, 0, 0, 0, 0, 0, 0 };
    return time_arms(fd, set, na, reps, seed);
}

/* Two candidates for the output pitch's cost, fp16 -> fp32 at 256x384x256, output side alone.
 * A power-of-two pitch aliasing DRAM banks predicts that a pitch 16 bytes off a power of two
 * costs what the tile's own width does; a notch whose cost grows with its size predicts the
 * off-by-16 pitch costs what its power-of-two neighbour does. Then the same wide pitch with only
 * one notch half programmed, when that arm is exact in the arm table. */
static int time_pitch2(int fd, int reps, uint64_t seed, int halves)
{
    const int M = 256, Kt = 384, N = 256, dt = F16F32;
    struct arm set[MAX_TIMED];
    int na = 0;
    set[na++] = (struct arm){ "t_cube", dt, M, Kt, N, M, Kt, N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    static const int cp[] = { 1024, 2048, 2052, 4096, 4100, 4160, 8192, 8196 };
    for (unsigned i = 0; i < sizeof(cp) / sizeof(cp[0]); i++)
        set[na++] = (struct arm){ "t_out_pitch", dt, M, Kt, N, M, Kt, cp[i], 0, 0, N, 0, 1,
                                  0, 0, 0, 0, 0 };
    if (halves & 1)
        set[na++] = (struct arm){ "t_out_lo_only", dt, M, Kt, N, M, Kt, 4096, 0, 0, N, 0, 1,
                                  0, 0, 0, 0, 1 };
    if (halves & 2)
        set[na++] = (struct arm){ "t_out_hi_only", dt, M, Kt, N, M, Kt, 4096, 0, 0, N, 0, 1,
                                  0, 0, 0, 0, 2 };
    return time_arms(fd, set, na, reps, seed);
}

int main(int argc, char **argv)
{
    int timing = 1, arms_on = 1, pitch = 0, halves = 0, reps = 30, bad = 0, skipped = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-timing")) timing = 0;
        else if (!strcmp(argv[i], "--timing-only")) arms_on = 0;
        else if (!strcmp(argv[i], "--pitch")) { arms_on = 0; pitch = 1; }
        else if (!strcmp(argv[i], "--pitch-asym")) { arms_on = 0; pitch = 3; }
        else if (!strcmp(argv[i], "--pitch2") && i + 1 < argc) { arms_on = 0; pitch = 2; halves = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--wait-out")) g_wait_out = 1;
        else if (!strcmp(argv[i], "--unequal")) g_unequal = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--no-timing|--timing-only|--pitch|--pitch-asym|--pitch2 H] [--wait-out] "
                    "[--unequal] [--reps N]\n", argv[0]);
            return 1;
        }
    }
    if (reps < 3) reps = 3;
    int fd = rocket_open();
    if (fd < 0) { printf("no NPU device\n"); return 2; }
    const int narms = (int)(sizeof(arms) / sizeof(arms[0]));
    printf("== row-major tile probe: %d arms, one single-task job each (two for kacc) ==\n", narms);
    for (int i = 0; i < narms && arms_on; i++) {
        const struct arm *a = &arms[i];
        struct host h = { 0 };
        struct bufs b;
        memset(&b, 0, sizeof b);
        if (host_init(a, &h, 0x5100 + (uint64_t)i * 16) || bufs_alloc(fd, a, &b, 0, 0)) {
            printf("  %s: allocation failed\n", a->name); bad++;
        } else {
            int rc = run_arm(fd, a, &h, &b, 0, NULL);
            if (rc < 0) skipped++;
            if (rc != 0) bad++;
        }
        bufs_free(fd, &b);
        host_free(&h);
    }
    if (pitch) {
        printf("== the row pitch alone, fp16 -> fp32 %s; BOs %s ==\n",
               pitch == 3 ? "256x512x128" : "256x384x256",
               g_unequal ? "each at its own size" : "equal in every arm");
        bad += pitch == 2 ? time_pitch2(fd, reps, 0x5500, halves)
             : pitch == 3 ? time_pitch(fd, reps, 0x5600, 256, 512, 128)
                          : time_pitch(fd, reps, 0x5400, 256, 384, 256);
    } else if (timing) {
        printf("== timing at the library's tile shapes, row-major arms in a 4096-wide A and C; "
               "wait on %s, BOs %s ==\n", g_wait_out ? "the OUTPUT BO (its sync timed)" : "a 4 KiB fence BO",
               g_unequal ? "each at its own size" : "equal in both arms");
        bad += time_set(fd, F16F32, 256, 384, 256, reps, 0x5300);
        bad += time_set(fd, F16F32, 256, 512, 128, reps, 0x5310);
        bad += time_set(fd, F16F16, 256, 384, 256, reps, 0x5320);
        bad += time_set(fd, I8I32, 256, 512, 256, reps, 0x5330);
    }
    printf("== %d of %d arms not exact (%d not run); fence-slow count %llu ==\n", bad, narms, skipped,
           (unsigned long long)rocket_fence_wait_slow_count());
    rocket_close(fd);
    return bad ? 1 : 0;
}
