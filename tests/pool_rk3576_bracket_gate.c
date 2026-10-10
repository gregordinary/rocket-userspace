// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * pool_rk3576_bracket_gate.c — host-only GATE: the RK3576 pooling entry keeps every CPU
 * access to a BO inside a PREP_BO/FINI_BO bracket, on every path.
 *
 * rocket_npu.h makes the CPU's access to a BO a bracket: PREP before reading or writing,
 * FINI after, and nothing touching the mapping outside one. A path that reads after its
 * FINI reads whatever the cache holds, a path that never FINIs leaves the provider's
 * ownership state wrong for the next call, and a FINI with no PREP open is a second close.
 * None of the three is visible on the device: the mainline provider happens to keep the
 * bytes current, and the result is right. So this gate replaces the provider instead.
 *
 * THE MOCK IS THE WHOLE SUBMIT SEAM, defined in this file. The seam is exactly the
 * externally visible rocket_* functions of src/rocket_npu.c (tools/provider-seam.sh), and
 * a static archive only pulls a member in for a symbol nothing has defined, so with all of
 * them defined here the builtin provider never links and the library runs against host
 * memory. Each BO is an anonymous mapping held PROT_NONE outside its bracket, so a CPU
 * access outside one faults and the gate names the scenario; PREP on an open bracket, FINI
 * on a closed one, a submit naming a BO the CPU still holds and a bracket left open when
 * the call returns are each counted. The "device" writes the reference pool's output cube
 * into the output BO, so the result is also scored against rocket_pool_ref_int8_rk3576().
 *
 * Every scenario runs with ROCKET_RK3576_I32_SENTINEL on and off:
 *
 *   clean     max pool, one attempt
 *   lagcheck  padded average excluding the pad: the divisor-lag check reads the surface
 *   retired   the first job runs past the backstop, so the entry redoes it
 *   lagged    the first surface fails the divisor-lag check, so the entry redoes it
 *   unwritten the first job writes nothing (sentinel on only: off, it cannot be seen)
 *   submit    the submit fails
 *   wait      the fence wait fails
 *
 * and is held to an exact PREP/FINI count on the output BO, which is printed.
 *
 * No NPU and no device file; forces ROCKET_CHIP=rk3576 itself. Exit 0 or 1, never skips.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "rocket_npu.h"
#include "rocket_hw_profile.h"
#include "rocket_log.h"
#include "rocket_pool.h"

/* ============================================================================
 * SECTION — the mock provider
 * ==========================================================================*/

#define MOCK_FD      77
#define MOCK_MAX_BO  64
#define MOCK_C2      16

struct mock_bo {
    void    *map;
    size_t   len;
    int      live, open;
    long     preps, finis;          /* brackets opened and closed */
};
static struct mock_bo g_bo[MOCK_MAX_BO];
static uint64_t g_next_iova = 0x10000000ull;

/* What the next submits do, set per scenario. */
static struct {
    long submits, waits;            /* counted per scenario */
    int  fail_submit;               /* this submit (1-based) fails */
    int  fail_wait;                 /* this output wait (1-based) fails */
    int  sleep_submit;              /* this submit runs past the backstop */
    int  blank_submit;              /* this submit writes nothing */
    int  lag_submit;                /* this submit writes a lagged corner */
    const int8_t *cube;             /* the output cube the device writes */
    size_t cube_bytes;
    size_t lag_byte;                /* the corner byte a lagged surface moves */
} g_dev;

static long g_violations;
static volatile const char *g_scenario = "setup";

static void violation(const char *what, uint32_t handle)
{
    if (g_violations < 20)
        printf("  FAIL   [%s] %s (BO %u)\n", (const char *)g_scenario, what, handle);
    g_violations++;
}

static struct mock_bo *bo_of(uint32_t handle)
{
    if (handle == 0 || handle > MOCK_MAX_BO || !g_bo[handle - 1].live) return NULL;
    return &g_bo[handle - 1];
}

/* A fault on a BO mapping is a CPU access outside its bracket. Name it and stop. */
static void on_segv(int sig)
{
    static const char a[] = "  FAIL   [";
    static const char b[] = "] CPU access to a BO outside its PREP/FINI bracket\n";
    const char *s = (const char *)g_scenario;
    (void)sig;
    if (write(1, a, sizeof a - 1) < 0 || write(1, s, strlen(s)) < 0 ||
        write(1, b, sizeof b - 1) < 0) {}
    _exit(1);
}

int rocket_open(void) { return MOCK_FD; }
void rocket_close(int fd) { (void)fd; }

int rocket_bo_alloc(int fd, size_t size, rocket_bo *bo)
{
    size_t pg = (size_t)sysconf(_SC_PAGESIZE), len;
    unsigned i;
    memset(bo, 0, sizeof *bo);
    if (fd != MOCK_FD || size == 0) return -EINVAL;
    len = (size + pg - 1) / pg * pg;
    for (i = 0; i < MOCK_MAX_BO && g_bo[i].live; i++) ;
    if (i == MOCK_MAX_BO) return -ENOMEM;
    void *m = mmap(NULL, len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return -ENOMEM;
    g_bo[i] = (struct mock_bo){ .map = m, .len = len, .live = 1 };
    bo->handle = i + 1;
    bo->dma_address = g_next_iova;
    bo->size = size;
    bo->ptr = m;
    g_next_iova += len + pg;
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
    struct mock_bo *m;
    (void)fd;
    if (!bo || !bo->handle) return;
    m = bo_of(bo->handle);
    if (m) {
        if (m->open) violation("freed with its bracket open", bo->handle);
        munmap(m->map, m->len);
        memset(m, 0, sizeof *m);
    }
    memset(bo, 0, sizeof *bo);
}

static int is_output(uint32_t handle);

int rocket_bo_prep(int fd, rocket_bo *bo, int dir, uint64_t timeout_ns)
{
    struct mock_bo *m = bo_of(bo->handle);
    (void)dir;
    if (fd != MOCK_FD || !m) { violation("PREP on an unknown BO", bo->handle); return -ENOENT; }
    /* The injected wait failure opens no bracket, as a fence that never signalled does
     * not — the convention every entry's error path is written to. */
    if (timeout_ns && is_output(bo->handle) && ++g_dev.waits == g_dev.fail_wait)
        return -ETIMEDOUT;
    if (m->open) violation("PREP on a bracket that is already open", bo->handle);
    m->open = 1;
    m->preps++;
    mprotect(m->map, m->len, PROT_READ | PROT_WRITE);
    return 0;
}

int rocket_bo_fini(int fd, rocket_bo *bo)
{
    struct mock_bo *m = bo_of(bo->handle);
    if (fd != MOCK_FD || !m) { violation("FINI on an unknown BO", bo->handle); return -ENOENT; }
    if (!m->open) violation("FINI with no bracket open", bo->handle);
    m->open = 0;
    m->finis++;
    mprotect(m->map, m->len, PROT_NONE);
    return 0;
}

int rocket_bo_prep_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r, unsigned n,
                          uint64_t timeout_ns)
{
    (void)r; (void)n;
    return rocket_bo_prep(fd, bo, 0, timeout_ns);
}

int rocket_bo_fini_ranges(int fd, rocket_bo *bo, const rocket_bo_range *r, unsigned n)
{
    (void)r; (void)n;
    return rocket_bo_fini(fd, bo);
}

int rocket_bo_ranges_supported(void)       { return 0; }
int rocket_batched_submit_supported(void)  { return 0; }
int rocket_batched_submit_native(void)     { return 0; }
int rocket_ppu_done_supported(void)        { return 1; }
int rocket_no_dpu_done_supported(void)     { return 0; }
int rocket_batch_completion_tracked(void)  { return 0; }
int rocket_submit_batch_atomic(void)       { return 0; }

static uint32_t g_out_handle;
static int is_output(uint32_t handle) { return handle && handle == g_out_handle; }

/* The device: write the output cube into the output BO through its own access, the way
 * DMA does, without opening a CPU bracket. */
static void device_write(uint32_t handle, int blank, int lag)
{
    struct mock_bo *m = bo_of(handle);
    if (!m || blank || !g_dev.cube) return;
    mprotect(m->map, m->len, PROT_READ | PROT_WRITE);
    memcpy(m->map, g_dev.cube, g_dev.cube_bytes);
    if (lag) ((int8_t *)m->map)[g_dev.lag_byte] ^= 0x01;
    mprotect(m->map, m->len, m->open ? PROT_READ | PROT_WRITE : PROT_NONE);
}

static int mock_submit(const uint32_t *in_h, uint32_t n_in, const uint32_t *out_h,
                       uint32_t n_out)
{
    uint32_t i;
    long n = ++g_dev.submits;
    if (n_out) g_out_handle = out_h[0];
    for (i = 0; i < n_in; i++) {
        struct mock_bo *m = bo_of(in_h[i]);
        if (!m) violation("submit names an unknown BO", in_h[i]);
        else if (m->open) violation("submit while the CPU holds an input's bracket", in_h[i]);
    }
    for (i = 0; i < n_out; i++) {
        struct mock_bo *m = bo_of(out_h[i]);
        if (!m) violation("submit names an unknown BO", out_h[i]);
        else if (m->open) violation("submit while the CPU holds the output's bracket", out_h[i]);
    }
    if (n == g_dev.fail_submit) return -EIO;
    if (n_out) device_write(out_h[0], n == g_dev.blank_submit, n == g_dev.lag_submit);
    if (n == g_dev.sleep_submit) {
        struct timespec ts = { 0, 60 * 1000000L };   /* past the 40 ms backstop set below */
        nanosleep(&ts, NULL);
    }
    return 0;
}

int rocket_submit_matmul_flags(int fd, const rocket_bo *regcmd_bo, uint32_t regcmd_count,
                               const uint32_t *in_handles, uint32_t n_in,
                               const uint32_t *out_handles, uint32_t n_out,
                               uint32_t job_flags)
{
    (void)regcmd_bo; (void)regcmd_count; (void)job_flags;
    if (fd != MOCK_FD) return -EBADF;
    return mock_submit(in_handles, n_in, out_handles, n_out);
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
    (void)tasks; (void)n_tasks; (void)job_flags;
    if (fd != MOCK_FD) return -EBADF;
    return mock_submit(in_handles, n_in, out_handles, n_out);
}

int rocket_submit_tasks(int fd, const rocket_task_desc *tasks, uint32_t n_tasks,
                        const uint32_t *in_handles, uint32_t n_in,
                        const uint32_t *out_handles, uint32_t n_out)
{
    return rocket_submit_tasks_flags(fd, tasks, n_tasks, in_handles, n_in, out_handles,
                                     n_out, 0u);
}

size_t rocket_submit_scratch_size(uint32_t max_tasks) { return (size_t)max_tasks * 16u + 64u; }

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
    (void)fd; (void)jobs; (void)n_jobs;
    violation("rocket_submit_jobs is not a pooling submit", 0);
    return -ENOSYS;
}

uint64_t rocket_submit_ioctl_count(void) { return (uint64_t)g_dev.submits; }
uint64_t rocket_submit_task_count(void)  { return (uint64_t)g_dev.submits; }
void     rocket_submit_counters_reset(void) {}

/* ============================================================================
 * SECTION — the scenarios
 * ==========================================================================*/

struct shape {
    const char *name;
    rocket_pool_desc d;
    int in_zp;
};

struct scenario {
    const char *name;
    int fail_submit, fail_wait, sleep_submit, blank_submit, lag_submit;
    int lagcheck;                   /* needs the padded-average shape */
    int sentinel_only;              /* invisible without the sentinel */
    int expect_ok;
    long prep_on, fini_on, prep_off, fini_off;   /* output-BO bracket counts */
};

static const struct scenario k_scen[] = {
    /* name        fsub fwait sleep blank lag  lagc  sent  ok   on      off  */
    { "clean",      0,   0,    0,    0,    0,   0,    0,    1,   2, 2,   1, 1 },
    { "lagcheck",   0,   0,    0,    0,    0,   1,    0,    1,   2, 2,   1, 1 },
    { "retired",    0,   0,    1,    0,    0,   0,    0,    1,   4, 4,   2, 2 },
    { "lagged",     0,   0,    0,    0,    1,   1,    0,    1,   4, 4,   2, 2 },
    { "unwritten",  0,   0,    0,    1,    0,   0,    1,    1,   4, 4,   0, 0 },
    { "submit",     1,   0,    0,    0,    0,   0,    0,    0,   1, 1,   0, 0 },
    { "wait",       0,   1,    0,    0,    0,   0,    0,    0,   1, 1,   0, 0 },
};

static int run(const struct shape *sh, const struct scenario *sc, int sentinel)
{
    const rocket_pool_desc *d = &sh->d;
    const int oh = rocket_pool_oh(d), ow = rocket_pool_ow(d);
    const size_t n_in = (size_t)d->c * d->ih * d->iw, n_out = (size_t)d->c * oh * ow;
    const unsigned groups = ((unsigned)d->c + MOCK_C2 - 1u) / MOCK_C2;
    const size_t surf = ((size_t)ow * oh + 3u) & ~(size_t)3u;
    int8_t *in = malloc(n_in), *ref = malloc(n_out), *out = malloc(n_out);
    int8_t *cube = calloc((size_t)groups * surf * MOCK_C2, 1);
    rocket_pool_int8_rk3576_handle *h;
    struct mock_bo *ob;
    char tag[96];
    int rc, bad = 0;
    size_t i;
    unsigned c, y, x;

    snprintf(tag, sizeof tag, "%s, %s, sentinel %s", sh->name, sc->name, sentinel ? "on" : "off");
    g_scenario = tag;
    if (!in || !ref || !out || !cube) { printf("  FAIL   [%s] out of memory\n", tag); return 1; }
    for (i = 0; i < n_in; i++) in[i] = (int8_t)((i * 2654435761u >> 13) & 0xFF);
    rocket_pool_ref_int8_rk3576(d, sh->in_zp, in, ref);
    for (c = 0; c < (unsigned)d->c; c++)
        for (y = 0; y < (unsigned)oh; y++)
            for (x = 0; x < (unsigned)ow; x++)
                cube[(size_t)(c / MOCK_C2) * surf * MOCK_C2 + ((size_t)y * ow + x) * MOCK_C2 +
                     c % MOCK_C2] = ref[((size_t)c * oh + y) * ow + x];

    setenv("ROCKET_RK3576_I32_SENTINEL", sentinel ? "1" : "0", 1);
    memset(&g_dev, 0, sizeof g_dev);
    h = rocket_pool_int8_pack_rk3576(MOCK_FD, d, sh->in_zp);
    if (!h) { printf("  FAIL   [%s] the pack refused\n", tag); return 1; }

    g_dev.cube = cube;
    g_dev.cube_bytes = (size_t)groups * surf * MOCK_C2;
    /* Output (0, 1) of the last channel: its window has one more column in the plane than
     * its raster predecessor's, so it is a position the lag check scores, and it is in the
     * last channel group, which is the one the shipped check reads. */
    g_dev.lag_byte = (size_t)((d->c - 1) / MOCK_C2) * surf * MOCK_C2 + MOCK_C2 +
                     (size_t)(d->c - 1) % MOCK_C2;
    g_dev.fail_submit = sc->fail_submit;
    g_dev.fail_wait = sc->fail_wait;
    g_dev.sleep_submit = sc->sleep_submit;
    g_dev.blank_submit = sc->blank_submit;
    g_dev.lag_submit = sc->lag_submit;
    g_dev.submits = g_dev.waits = 0;

    /* The output BO is the one the first submit names; count from the handle's own. */
    for (i = 0; i < MOCK_MAX_BO; i++) { g_bo[i].preps = 0; g_bo[i].finis = 0; }
    g_out_handle = 0;
    memset(out, 0x5A, n_out);

    rc = rocket_pool_int8_prepacked_rk3576(MOCK_FD, h, in, out);

    for (i = 0; i < MOCK_MAX_BO; i++)
        if (g_bo[i].live && g_bo[i].open) violation("bracket left open when the call returned", (uint32_t)i + 1u);
    ob = bo_of(g_out_handle);
    {
        long ep = sentinel ? sc->prep_on : sc->prep_off, ef = sentinel ? sc->fini_on : sc->fini_off;
        long gp = ob ? ob->preps : -1, gf = ob ? ob->finis : -1;
        printf("  %-4s   [%s] rc %d, output BO: %ld PREP / %ld FINI (expected %ld / %ld), "
               "%ld submit(s)\n", (gp == ep && gf == ef) ? "ok" : "FAIL", tag, rc, gp, gf,
               ep, ef, g_dev.submits);
        if (gp != ep || gf != ef) bad = 1;
    }
    if (sc->expect_ok) {
        if (rc != ROCKET_OK) { printf("  FAIL   [%s] returned %d\n", tag, rc); bad = 1; }
        else if (memcmp(out, ref, n_out) != 0) {
            printf("  FAIL   [%s] the de-scattered output is not the reference pool\n", tag);
            bad = 1;
        }
    } else if (rc == ROCKET_OK) {
        printf("  FAIL   [%s] an injected failure returned success\n", tag);
        bad = 1;
    }

    g_scenario = tag;
    rocket_pool_int8_free_rk3576(MOCK_FD, h);
    free(in); free(ref); free(out); free(cube);
    return bad;
}

int main(void)
{
    static const struct shape k_max = {
        "max 3x3 s2", { .c = 32, .ih = 9, .iw = 11, .kh = 3, .kw = 3, .stride_y = 2,
                        .stride_x = 2, .method = POOL_METHOD_MAX }, 0 };
    static const struct shape k_avg = {
        "avg 3x3 s1 pad 1, pad excluded",
        { .c = 32, .ih = 8, .iw = 8, .kh = 3, .kw = 3, .stride_y = 1, .stride_x = 1,
          .pad_top = 1, .pad_left = 1, .pad_bottom = 1, .pad_right = 1,
          .method = POOL_METHOD_AVG, .avg_exclude_pad = 1 }, 0 };
    struct sigaction sa;
    unsigned s;
    int fail = 0, ran = 0;

    /* Before anything resolves them: the part, a backstop the "retired" scenario can pass
     * in 60 ms, and no power-domain idle or sysfs write on a redo. */
    setenv("ROCKET_CHIP", "rk3576", 1);
    setenv("ROCKET_RK3576_BACKSTOP_US", "40000", 1);
    setenv("ROCKET_RK3576_MM_GAP_MS", "0", 1);
    setenv("ROCKET_RK3576_PM_KICK", "0", 1);
    unsetenv("ROCKET_RK3576_POOL_LAGCHECK");
    unsetenv("ROCKET_RK3576_TASK_ATTEMPTS");
    rocket_log_set_level(ROCKET_LOG_ERROR);

    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_segv;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);

    if (strcmp(rocket_hw_current()->name, "rk3576") != 0) {
        printf("FAIL: ROCKET_CHIP=rk3576 did not select the RK3576 profile\n");
        return 1;
    }
    printf("pool_rk3576_bracket_gate: the RK3576 pool entry against a host-memory provider\n");
    for (s = 0; s < sizeof k_scen / sizeof k_scen[0]; s++) {
        const struct scenario *sc = &k_scen[s];
        for (int sentinel = 1; sentinel >= 0; sentinel--) {
            if (sc->sentinel_only && !sentinel) continue;
            fail |= run(sc->lagcheck ? &k_avg : &k_max, sc, sentinel);
            ran++;
        }
    }
    printf("pool_rk3576_bracket_gate: %d scenario run(s), %ld bracket violation(s)\n",
           ran, g_violations);
    return (fail || g_violations) ? 1 : 0;
}
