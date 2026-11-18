// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
/*
 * prepacked_cost_gate.c — host-only GATE: a resident matmul's scratch grows with the call's
 * rows, and its pack scatters each weight element once.
 *
 * The resident paths plan their tiling at the canonical tile (MAX_TILE rows), so one packed
 * weight serves every M. Two costs follow from that plan, and neither changes a result:
 *
 *   scratch  The input and output slots must be sized from the call's rows, min(M, Mt).
 *            Sized at Mt, a call at M = 4 allocates, zeroes and cache-syncs the buffers of
 *            a MAX_TILE-row call and computes the same output.
 *   pack     Each weight tile is scattered once. A tile scattered again writes the same
 *            bytes, so the packed weight, and every output computed from it, is unchanged.
 *
 * The bit-exact gates compare outputs, so they cannot see either cost. This gate measures
 * the costs instead, against a host-memory provider defined in this file.
 *
 * THE MOCK IS THE WHOLE SUBMIT SEAM, as in pool_rk3576_bracket_gate: with every seam symbol
 * defined here, the builtin provider never links. Its BOs are anonymous mappings, its
 * submits complete at once and write nothing, and each submit records the size of every BO
 * it names as read or as written. The outputs are therefore meaningless and are not checked.
 *
 * Each path (int8, int8 group-wise, int4, int4 group-wise, and fp16 on a
 * ROCKET_CTX_TILING_CANONICAL context) packs one weight and is then called at M = 4, 8, 32
 * and 64, all below its canonical Mt. Each path is held to two checks:
 *
 *   scratch  The bytes of the BOs a call reads, and of the BOs it writes, grow with M: more
 *            at M = 8 than at M = 4, and at M = 32 and 64 by exactly (M/4 - 1) times that
 *            step. Every M-dependent slot is rup(M, 4) times a constant, so the growth is
 *            exact. A slot sized at Mt makes the step zero.
 *   pack     The pack's process CPU time is at most PACK_RATIO times that of one pass of
 *            the same rocket_cube.h scatter over the whole weight, plus PACK_SLACK_MS. A
 *            tile scattered once per row of the tile costs about Nt passes, which the
 *            report prints beside the bound.
 *
 * Forces ROCKET_CHIP=rk3588, the part these paths plan for, so it runs the same on any
 * host. No NPU and no device file. Exit 0 or 1, never skips.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_hw_profile.h"
#include "rocket_cube.h"

/* ============================================================================
 * SECTION — the mock submit seam
 * ==========================================================================*/

#define MOCK_FD     77
#define MOCK_MAX_BO 8192
#define LOG_MAX     256

struct mock_bo { void *map; size_t len, size; int live; };
static struct mock_bo g_bo[MOCK_MAX_BO];   /* handle h is g_bo[h - 1]; never reused */
static uint32_t g_nbo;
static uint64_t g_next_iova = 0x10000000ull;
static uint64_t g_submits, g_tasks;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

/* The BOs the submits since log_reset() named, once each: [0] read, [1] written. */
static struct { uint32_t h; size_t size; } g_log[2][LOG_MAX];
static unsigned g_nlog[2];
static int g_log_overflow;

static void log_reset(void)
{
    pthread_mutex_lock(&g_mu);
    g_nlog[0] = g_nlog[1] = 0;
    g_log_overflow = 0;
    pthread_mutex_unlock(&g_mu);
}

static size_t log_bytes(int dir)
{
    size_t s = 0;
    for (unsigned i = 0; i < g_nlog[dir]; i++) s += g_log[dir][i].size;
    return s;
}

static void log_handles(int dir, const uint32_t *h, uint32_t n)
{
    pthread_mutex_lock(&g_mu);
    for (uint32_t i = 0; i < n; i++) {
        if (h[i] == 0 || h[i] > g_nbo) continue;
        unsigned j = 0;
        while (j < g_nlog[dir] && g_log[dir][j].h != h[i]) j++;
        if (j < g_nlog[dir]) continue;
        if (j == LOG_MAX) { g_log_overflow = 1; continue; }
        g_log[dir][j].h = h[i];
        g_log[dir][j].size = g_bo[h[i] - 1].size;
        g_nlog[dir]++;
    }
    pthread_mutex_unlock(&g_mu);
}

int rocket_open(void) { return MOCK_FD; }
void rocket_close(int fd) { (void)fd; }

int rocket_bo_alloc(int fd, size_t size, rocket_bo *bo)
{
    const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    memset(bo, 0, sizeof *bo);
    if (fd != MOCK_FD || size == 0) return -EINVAL;
    const size_t len = (size + pg - 1) / pg * pg;
    void *m = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return -ENOMEM;
    pthread_mutex_lock(&g_mu);
    if (g_nbo == MOCK_MAX_BO) {
        pthread_mutex_unlock(&g_mu);
        munmap(m, len);
        return -ENOMEM;
    }
    g_bo[g_nbo] = (struct mock_bo){ .map = m, .len = len, .size = size, .live = 1 };
    bo->handle = ++g_nbo;
    /* The IOVA is a label nothing dereferences. It wraps below 2 GiB so that a long run
     * never trips the library's 32-bit IOVA check. */
    if (g_next_iova + len + pg >= 0x80000000ull) g_next_iova = 0x10000000ull;
    bo->dma_address = g_next_iova;
    g_next_iova += len + pg;
    pthread_mutex_unlock(&g_mu);
    bo->size = size;
    bo->ptr = m;
    return 0;
}

int rocket_bo_alloc32(int fd, size_t size, rocket_bo *bo)
{
    int rc = rocket_bo_alloc(fd, size, bo);
    if (rc) return ROCKET_E_NOMEM;
    if ((bo->dma_address + size - 1) >> 32) { rocket_bo_free(fd, bo); return ROCKET_E_DEVICE; }
    return 0;
}

int rocket_bo_ensure32(int fd, rocket_bo *bo, size_t need)
{
    if (bo->handle && bo->size >= need) return 0;
    rocket_bo_free(fd, bo);
    int rc = rocket_bo_alloc32(fd, need, bo);
    return rc == 0 ? 1 : rc;
}

void rocket_bo_free(int fd, rocket_bo *bo)
{
    (void)fd;
    if (!bo || !bo->handle) return;
    pthread_mutex_lock(&g_mu);
    if (bo->handle <= g_nbo && g_bo[bo->handle - 1].live) {
        munmap(g_bo[bo->handle - 1].map, g_bo[bo->handle - 1].len);
        g_bo[bo->handle - 1].live = 0;
    }
    pthread_mutex_unlock(&g_mu);
    memset(bo, 0, sizeof *bo);
}

int rocket_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t timeout_ns)
{ (void)fd; (void)bo; (void)dir; (void)timeout_ns; return 0; }
int rocket_bo_fini(int fd, rocket_bo *bo) { (void)fd; (void)bo; return 0; }
int rocket_bo_prep_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r, unsigned n,
                          uint64_t timeout_ns)
{ (void)fd; (void)bo; (void)r; (void)n; (void)timeout_ns; return 0; }
int rocket_bo_fini_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r, unsigned n)
{ (void)fd; (void)bo; (void)r; (void)n; return 0; }

int rocket_bo_ranges_supported(void)       { return 0; }
int rocket_batched_submit_supported(void)  { return 0; }
int rocket_batched_submit_native(void)     { return 0; }
int rocket_ppu_done_supported(void)        { return 0; }
int rocket_no_dpu_done_supported(void)     { return 0; }
int rocket_batch_completion_tracked(void)  { return 0; }
int rocket_submit_batch_atomic(void)       { return 0; }

static int mock_submit(uint32_t n_tasks, const uint32_t *in_handles, uint32_t n_in,
                       const uint32_t *out_handles, uint32_t n_out)
{
    log_handles(0, in_handles, n_in);
    log_handles(1, out_handles, n_out);
    pthread_mutex_lock(&g_mu);
    g_submits++;
    g_tasks += n_tasks;
    pthread_mutex_unlock(&g_mu);
    return 0;
}

int rocket_submit_matmul_flags(int fd, const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                               const uint32_t *in_handles, uint32_t n_in,
                               const uint32_t *out_handles, uint32_t n_out,
                               uint32_t job_flags)
{
    (void)fd; (void)regcmd_bo; (void)regcmd_count; (void)job_flags;
    return mock_submit(1, in_handles, n_in, out_handles, n_out);
}
int rocket_submit_matmul(int fd, const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                         const uint32_t *in_handles, uint32_t n_in,
                         const uint32_t *out_handles, uint32_t n_out, uint32_t timeout_ms)
{
    (void)timeout_ms;
    return rocket_submit_matmul_flags(fd, regcmd_bo, regcmd_count, in_handles, n_in,
                                      out_handles, n_out, 0u);
}
int rocket_submit_tasks_flags(int fd, const rocket_task_desc *tasks, uint32_t n_tasks,
                              const uint32_t *in_handles, uint32_t n_in,
                              const uint32_t *out_handles, uint32_t n_out,
                              uint32_t job_flags)
{
    (void)fd; (void)tasks; (void)job_flags;
    return mock_submit(n_tasks, in_handles, n_in, out_handles, n_out);
}
int rocket_submit_tasks(int fd, const rocket_task_desc *tasks, uint32_t n_tasks,
                        const uint32_t *in_handles, uint32_t n_in,
                        const uint32_t *out_handles, uint32_t n_out)
{
    return rocket_submit_tasks_flags(fd, tasks, n_tasks, in_handles, n_in, out_handles,
                                     n_out, 0u);
}
size_t rocket_submit_scratch_size(uint32_t max_tasks) { return (size_t)max_tasks * 64u + 64u; }
int rocket_submit_tasks_pre(int fd, void *scratch, const rocket_task_desc *tasks,
                            uint32_t n_tasks, const uint32_t *in_handles, uint32_t n_in,
                            const uint32_t *out_handles, uint32_t n_out, uint32_t job_flags)
{
    (void)scratch;
    return rocket_submit_tasks_flags(fd, tasks, n_tasks, in_handles, n_in, out_handles,
                                     n_out, job_flags);
}
int rocket_submit_jobs(int fd, const rocket_job_desc *jobs, uint32_t n_jobs)
{
    (void)fd;
    for (uint32_t j = 0; j < n_jobs; j++)
        mock_submit(jobs[j].n_tasks, jobs[j].in_handles, jobs[j].n_in,
                    jobs[j].out_handles, jobs[j].n_out);
    return 0;
}
uint64_t rocket_submit_ioctl_count(void) { return g_submits; }
uint64_t rocket_submit_task_count(void)  { return g_tasks; }
void     rocket_submit_counters_reset(void)
{
    pthread_mutex_lock(&g_mu);
    g_submits = g_tasks = 0;
    pthread_mutex_unlock(&g_mu);
}

/* ============================================================================
 * SECTION — the resident paths
 * ==========================================================================*/

/* One N slice of 512 columns per worker, aligned for every path's N granule; K a multiple
 * of the group; the pack at M = 4, so its time is the weight's and not a large scratch's. */
enum { K = 1024, N = 1536, GROUP = 128, NG = K / GROUP, NTHREADS = 3, PACK_M = 4, MMAX = 64 };
static const int MS[] = { 4, 8, 32, 64 };
#define NMS ((int)(sizeof MS / sizeof MS[0]))

#define PACK_REPS     3
#define PACK_RATIO    4.0
#define PACK_SLACK_MS 2.0

enum path { P_I8, P_I8GW, P_I4, P_I4GW, P_F16, NPATH };
static const char *const PNAME[NPATH] = {
    "int8", "int8 group-wise", "int4", "int4 group-wise", "fp16 canonical",
};

typedef struct {
    rocket_i8_ctx *c8; rocket_i8_weights *w8;
    rocket_i4_ctx *c4; rocket_i4_weights *w4;
    rocket_ctx    *cf; rocket_weights    *wf;
} pstate;

static int8_t   B8[(size_t)N * K], A8[(size_t)MMAX * K];
static _Float16 Bh[(size_t)N * K], Ah[(size_t)MMAX * K], Ch[(size_t)MMAX * N];
static int32_t  C32[(size_t)MMAX * N];
static float    Cf[(size_t)MMAX * N], a_scale[(size_t)MMAX * NG], b_scale[(size_t)N * NG];

static int p_open(enum path p, pstate *s)
{
    memset(s, 0, sizeof *s);
    switch (p) {
    case P_I8: case P_I8GW: s->c8 = rocket_i8_ctx_create(NTHREADS); return s->c8 ? 0 : -1;
    case P_I4: case P_I4GW: s->c4 = rocket_i4_ctx_create(NTHREADS); return s->c4 ? 0 : -1;
    default:
        s->cf = rocket_ctx_create_ex(NTHREADS, ROCKET_CTX_TILING_CANONICAL);
        return s->cf ? 0 : -1;
    }
}

static int p_pack(enum path p, pstate *s)
{
    switch (p) {
    case P_I8:   s->w8 = rocket_i8_weights_pack(s->c8, PACK_M, K, N, B8);           break;
    case P_I8GW: s->w8 = rocket_i8_weights_pack_gw(s->c8, PACK_M, K, N, B8, GROUP); break;
    case P_I4:   s->w4 = rocket_i4_weights_pack(s->c4, PACK_M, K, N, B8);           break;
    case P_I4GW: s->w4 = rocket_i4_weights_pack_gw(s->c4, PACK_M, K, N, B8, GROUP); break;
    default:     s->wf = rocket_weights_pack(s->cf, PACK_M, K, N, Bh);              break;
    }
    return (s->w8 || s->w4 || s->wf) ? 0 : -1;
}

static void p_unpack(pstate *s)
{
    if (s->w8) rocket_i8_weights_free(s->c8, s->w8);
    if (s->w4) rocket_i4_weights_free(s->c4, s->w4);
    if (s->wf) rocket_weights_free(s->cf, s->wf);
    s->w8 = NULL; s->w4 = NULL; s->wf = NULL;
}

static void p_close(pstate *s)
{
    p_unpack(s);
    if (s->c8) rocket_i8_ctx_free(s->c8);
    if (s->c4) rocket_i4_ctx_free(s->c4);
    if (s->cf) rocket_ctx_free(s->cf);
    memset(s, 0, sizeof *s);
}

static int p_call(enum path p, pstate *s, int M)
{
    switch (p) {
    case P_I8:   return rocket_matmul_int8_prepacked(s->c8, M, K, N, A8, C32, s->w8);
    case P_I8GW: return rocket_matmul_int8_prepacked_gw(s->c8, M, K, N, A8, a_scale, b_scale,
                                                        Cf, s->w8);
    case P_I4:   return rocket_matmul_int4_prepacked(s->c4, M, K, N, A8, C32, s->w4);
    case P_I4GW: return rocket_matmul_int4_prepacked_gw(s->c4, M, K, N, A8, a_scale, b_scale,
                                                        Cf, s->w4);
    default:     return rocket_matmul_fp16_prepacked(s->cf, M, K, N, Ah, Ch, s->wf);
    }
}

/* The canonical plan of one worker's N slice: the tile every resident call shares. */
static int canonical_plan(enum path p, int *Mt, int *Nt)
{
    const int tile = rocket_hw_current()->max_tile, nsub = N / NTHREADS;
    int Kt;
    switch (p) {
    case P_I8:   return rocket_matmul_plan_int8(tile, K, nsub, Mt, &Kt, Nt);
    case P_I8GW: return rocket_matmul_plan_int8_gw(tile, K, nsub, GROUP, Mt, &Kt, Nt);
    case P_I4: case P_I4GW: return rocket_matmul_plan_int4(tile, K, nsub, Mt, &Kt, Nt);
    default:     return rocket_matmul_plan(tile, K, nsub, Mt, &Kt, Nt);
    }
}

/* ============================================================================
 * SECTION — the measurements
 * ==========================================================================*/

static double cpu_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static size_t rup(size_t x, size_t a) { return (x + a - 1) / a * a; }

/* One pass of the path's own scatter over the whole weight, into a fresh zeroed mapping:
 * the work a pack must do, done once, with the mapping and the zeroing a pack's BO costs. */
static double ref_pass_ms(enum path p)
{
    size_t bytes;
    switch (p) {
    case P_I8: case P_I8GW: bytes = rup(N, 32) * rup(K, 32);                    break;
    case P_I4: case P_I4GW: bytes = rup(N, 64) * rup(K, 32) / 2;                break;
    default:                bytes = rup(N, 16) * rup(K, 32) * sizeof(_Float16); break;
    }
    const double t0 = cpu_ms();
    void *buf = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) return -1.0;
    memset(buf, 0, bytes);
    switch (p) {
    case P_I8: case P_I8GW:
        MM_WT_SCATTER(int8_t, int8_t, 32, 32, wt_idx_i8, buf, B8, K, 0, 0, N, K, MM_ID);
        break;
    case P_I4: case P_I4GW:
        i4_wt_scatter((uint8_t *)buf, B8, (size_t)K, 0, 0, N, K);
        break;
    default:
        MM_WT_SCATTER(_Float16, _Float16, 32, 16, wt_idx, buf, Bh, K, 0, 0, N, K, MM_ID);
        break;
    }
    const double t = cpu_ms() - t0;
    munmap(buf, bytes);
    return t;
}

static int fails;

#define CHECK(cond, ...) do {                                                   \
        if (cond) { printf("  ok     "); printf(__VA_ARGS__); printf("\n"); }   \
        else { fails++; printf("  FAIL   "); printf(__VA_ARGS__); printf("\n"); } \
    } while (0)

static void run_path(enum path p)
{
    pstate s;
    int Mt = 0, Nt = 0;

    printf("%s\n", PNAME[p]);
    if (canonical_plan(p, &Mt, &Nt) < 0 || p_open(p, &s) != 0) {
        CHECK(0, "the canonical plan and the context exist");
        return;
    }
    CHECK(Mt > MS[NMS - 1], "the canonical tile (Mt %d, Nt %d) lies above every M called (to %d)",
          Mt, Nt, MS[NMS - 1]);

    /* pack: the best of PACK_REPS packs against the best of PACK_REPS reference passes */
    double t_pack = -1.0, t_ref = -1.0;
    int packed = 1;
    for (int r = 0; r < PACK_REPS && packed; r++) {
        p_unpack(&s);
        const double t0 = cpu_ms();
        packed = p_pack(p, &s) == 0;
        const double t = cpu_ms() - t0;
        if (packed && (t_pack < 0 || t < t_pack)) t_pack = t;
    }
    for (int r = 0; r < PACK_REPS; r++) {
        const double t = ref_pass_ms(p);
        if (t >= 0 && (t_ref < 0 || t < t_ref)) t_ref = t;
    }
    if (!packed || t_ref < 0) {
        CHECK(0, "the weight packs and the reference pass runs");
        p_close(&s);
        return;
    }
    const double bound = PACK_RATIO * t_ref + PACK_SLACK_MS;
    CHECK(t_pack <= bound,
          "pack %.2f ms CPU <= %.1f x one scatter pass %.2f ms + %.1f ms = %.2f ms "
          "(a tile scattered once per row would cost ~%.0f ms)",
          t_pack, PACK_RATIO, t_ref, PACK_SLACK_MS, bound, t_ref * Nt);

    /* scratch: what each call's submits read and write */
    size_t in[NMS], out[NMS];
    int called = 1;
    for (int i = 0; i < NMS; i++) {
        log_reset();
        const int r = p_call(p, &s, MS[i]);
        in[i] = log_bytes(0);
        out[i] = log_bytes(1);
        if (r != 0 || g_log_overflow || !in[i] || !out[i]) {
            CHECK(0, "the call at M=%d ran and its BOs were logged (rc %d)", MS[i], r);
            called = 0;
        }
    }
    if (called) {
        for (int i = 0; i < NMS; i++)
            printf("         M=%-3d reads %9zu B   writes %9zu B\n", MS[i], in[i], out[i]);
        const size_t din = in[1] - in[0], dout = out[1] - out[0];
        CHECK(in[1] > in[0] && out[1] > out[0],
              "the BOs read and written grow from M=4 to M=8 (+%zu B read, +%zu B written)",
              in[1] > in[0] ? din : 0, out[1] > out[0] ? dout : 0);
        for (int i = 2; i < NMS; i++) {
            const size_t steps = (size_t)(MS[i] / 4 - 1);
            CHECK(in[i] - in[0] == steps * din && out[i] - out[0] == steps * dout,
                  "at M=%d they grow by exactly %zu steps of M=8's", MS[i], steps);
        }
    }
    p_close(&s);
}

int main(void)
{
    setenv("ROCKET_CHIP", "rk3588", 1);    /* the part these paths plan for */

    for (size_t i = 0; i < (size_t)N * K; i++) {
        B8[i] = (int8_t)((int)((i * 2654435761u) >> 7 & 15) - 7);   /* int4's [-7, 7] */
        Bh[i] = (_Float16)B8[i];
    }
    for (size_t i = 0; i < (size_t)MMAX * K; i++) {
        A8[i] = (int8_t)((int)((i * 40503u) >> 5 & 15) - 7);
        Ah[i] = (_Float16)A8[i];
    }
    for (size_t i = 0; i < (size_t)MMAX * NG; i++) a_scale[i] = 1.0f / 64;
    for (size_t i = 0; i < (size_t)N * NG; i++) b_scale[i] = 1.0f / 64;

    for (int p = 0; p < NPATH; p++) run_path((enum path)p);

    printf(fails ? "\n%d CHECK(s) FAILED\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
