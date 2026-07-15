// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * prep_signal_robust_rocket.c — robustness gate for the blocking NPU wait in
 * rocket_bo_prep(). Two independent failure modes, one real fp16 job driving a
 * genuine PREP_BO wait each iteration:
 *
 *   Phase A (EINTR retry). The kernel wait is dma_resv_wait_timeout(..., true, ..)
 *     — INTERRUPTIBLE (rocket_gem.c). A signal delivered mid-wait to a handler
 *     WITHOUT SA_RESTART makes the ioctl return -1/EINTR. A robust wait must
 *     retry to the same absolute deadline, not surface a spurious "WAIT TIMEOUT".
 *     A helper thread spams SIGUSR1 at the waiting (main) thread while the loop
 *     submits and waits. Without the retry loop, the first interrupted wait
 *     returns nonzero; with it, every iteration completes clean.
 *
 *   Phase B (deadline saturation). timeout_ns is unsigned; the idiomatic
 *     "wait forever" value UINT64_MAX must not, through the signed-int64 add,
 *     wrap the absolute deadline NEGATIVE (a past deadline degrades PREP_BO to a
 *     non-blocking poll -> instant -EBUSY while the job is still running). With
 *     saturation the wait blocks correctly and returns 0.
 *
 * Byte-identical outputs are checked every iteration so a "wait returned 0" that
 * actually read an un-synced BO is still caught. Exit 0 = PASS, 1 = FAIL.
 *
 * This is an A/B gate: run it against the pre-fix library and it FAILS (both
 * phases), demonstrating the bugs are real; against the fixed library it PASSES.
 *
 * Usage: prep_signal_robust_rocket [M K N]   (default 64 1024 64)
 *   env  ROCKET_SIG_ITERS=<n>   iterations per phase (default 200)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include <stdatomic.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

/* ---- signal plumbing ---------------------------------------------------- */
static volatile sig_atomic_t g_sig_count = 0;
static void on_sigusr1(int s) { (void)s; g_sig_count++; }

static pthread_t        g_main_tid;
static atomic_int       g_spam_on   = 0;   /* 1 => helper delivers SIGUSR1 to main */
static atomic_int       g_helper_stop = 0;

static void *spam_thread(void *arg)
{
    (void)arg;
    /* Target the main thread specifically (it owns the PREP_BO wait); a
     * process-directed signal could be taken by this helper instead. ~10 kHz is
     * far denser than a multi-ms job's wait window, so each wait is interrupted
     * many times over. */
    while (!atomic_load(&g_helper_stop)) {
        if (atomic_load(&g_spam_on))
            pthread_kill(g_main_tid, SIGUSR1);
        struct timespec ns = { 0, 100000 };   /* 100 us */
        nanosleep(&ns, NULL);
    }
    return NULL;
}

/* ---- the job ------------------------------------------------------------ */
static int    g_M, g_K, g_N;
static int    g_fd;
static rocket_bo g_guard, g_regcmd, g_input, g_weights, g_output;
static uint32_t  g_task_count;
static _Float16 *g_A, *g_B, *g_Cref;

static int64_t now_ns(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void ref_matmul(void)
{
    for (int i = 0; i < g_M; i++)
        for (int j = 0; j < g_N; j++) {
            float s = 0;
            for (int l = 0; l < g_K; l++) s += (float)g_A[i*g_K+l] * (float)g_B[j*g_K+l];
            g_Cref[i*g_N+j] = (_Float16)s;
        }
}

/* One-time device + host setup. Returns 0 on success. */
static int setup(void)
{
    g_fd = rocket_open();
    if (g_fd < 0) return g_fd;

    int ret = 0;
    rocket_bo_alloc(g_fd, 4096, &g_guard);                        /* reserve IOVA 0 */
    ret |= rocket_bo_alloc(g_fd, 4096, &g_regcmd);
    ret |= rocket_bo_alloc(g_fd, (size_t)g_M*g_K*sizeof(_Float16), &g_input);
    ret |= rocket_bo_alloc(g_fd, (size_t)g_N*g_K*sizeof(_Float16), &g_weights);
    ret |= rocket_bo_alloc(g_fd, (size_t)g_M*g_N*sizeof(_Float16), &g_output);
    if (ret) { fprintf(stderr, "bo alloc failed\n"); return -1; }

    uint64_t npu_regs[256] = {0};
    matmul_params_t p = {
        .m = g_M, .k = g_K, .n = g_N,
        .input_dma = g_input.dma_address, .weights_dma = g_weights.dma_address,
        .output_dma = g_output.dma_address, .tasks = npu_regs, .fp32tofp16 = 1,
    };
    if ((ret = gen_matmul_fp16(&p)) != 0) {
        fprintf(stderr, "gen_matmul_fp16 = %d (shape exceeds one CBUF task; pick a "
                        "smaller M*K + N*K)\n", ret);
        return ret;
    }
    g_task_count = p.task_count;

    rocket_bo_prep(g_fd, &g_regcmd, 1, 0);
    rocket_bo_prep(g_fd, &g_input,  1, 0);
    rocket_bo_prep(g_fd, &g_weights, 1, 0);
    memcpy(g_regcmd.ptr, npu_regs, (size_t)g_task_count * sizeof(uint64_t));

    g_A = malloc((size_t)g_M*g_K*2);
    g_B = malloc((size_t)g_N*g_K*2);
    g_Cref = malloc((size_t)g_M*g_N*2);
    srand(1234);
    for (int i = 0; i < g_M*g_K; i++) g_A[i] = (_Float16)(int)(10.0 * rand()/(float)RAND_MAX);
    for (int i = 0; i < g_N*g_K; i++) g_B[i] = (_Float16)(int)(10.0 * rand()/(float)RAND_MAX);

    _Float16 *wdst = g_weights.ptr, *idst = g_input.ptr;
    memset(g_input.ptr, 0, g_input.size);
    memset(g_weights.ptr, 0, g_weights.size);
    for (int n = 1; n <= g_N; n++)
        for (int k = 1; k <= g_K; k++) wdst[weight_fp16(g_K, n, k)] = g_B[(n-1)*g_K + (k-1)];
    for (int m = 1; m <= g_M; m++)
        for (int k = 1; k <= g_K; k++) idst[feature_data(g_K, g_M, 1, 8, k, m, 1)] = g_A[(m-1)*g_K + (k-1)];

    rocket_bo_fini(g_fd, &g_regcmd);
    rocket_bo_fini(g_fd, &g_input);
    rocket_bo_fini(g_fd, &g_weights);
    ref_matmul();
    return 0;
}

/* Submit one job (NPU reads input+weights+regcmd, writes output). The output BO
 * is synced for the device (fini) first so a re-submit in the loop is coherent. */
static int submit_once(void)
{
    rocket_bo_fini(g_fd, &g_output);
    uint32_t in_h[]  = { g_input.handle, g_weights.handle, g_regcmd.handle };
    uint32_t out_h[] = { g_output.handle };
    return rocket_submit_matmul(g_fd, &g_regcmd, g_task_count, in_h, 3, out_h, 1, 6000);
}

/* Count byte-mismatches of the NPU output against the host reference (bounded). */
static int check_output(void)
{
    _Float16 *od = g_output.ptr;
    int bad = 0;
    for (int m = 1; m <= g_M && bad < 8; m++)
        for (int n = 1; n <= g_N; n++) {
            _Float16 act = od[feature_data(g_N, g_M, 1, 8, n, m, 1)];
            _Float16 exp = g_Cref[(m-1)*g_N + (n-1)];
            if (act != exp && ++bad >= 8) break;
        }
    return bad;
}

/* Drain any in-flight job with a clean blocking wait (no signal spam). Used to
 * keep the BO coherent after a *pre-fix* spurious failure so the loop can go on. */
static void drain(void)
{
    int was = atomic_exchange(&g_spam_on, 0);
    for (int i = 0; i < 100; i++)
        if (rocket_bo_prep(g_fd, &g_output, 0, 2000000000LL) == 0) break;
    atomic_store(&g_spam_on, was);
}

int main(int argc, char **argv)
{
    g_M = argc > 3 ? atoi(argv[1]) : 64;
    g_K = argc > 3 ? atoi(argv[2]) : 1024;
    g_N = argc > 3 ? atoi(argv[3]) : 64;
    const char *it_env = getenv("ROCKET_SIG_ITERS");
    int iters = it_env && *it_env ? atoi(it_env) : 200;
    if (iters < 1) iters = 1;

    /* Match the suite convention: exit 2 (ctest SKIP) when there is no NPU to
     * open; a real setup error on a present device is a genuine failure (1). */
    if (setup() != 0) return (g_fd < 0) ? 2 : 1;
    printf("prep_signal_robust: M=%d K=%d N=%d, %d iters/phase, task_count=%u\n",
           g_M, g_K, g_N, iters, g_task_count);

    /* SIGUSR1 handler WITHOUT SA_RESTART — the whole point: a restart-free
     * handler is what surfaces EINTR to userspace from the interruptible wait. */
    struct sigaction sa = {0};
    sa.sa_handler = on_sigusr1;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;                       /* NO SA_RESTART */
    if (sigaction(SIGUSR1, &sa, NULL) != 0) { perror("sigaction"); return 1; }

    g_main_tid = pthread_self();
    pthread_t helper;
    if (pthread_create(&helper, NULL, spam_thread, NULL) != 0) { perror("pthread_create"); return 1; }

    /* ---- Phase A: EINTR retry under signal storm ------------------------ */
    int a_waitfail = 0, a_corrupt = 0;
    atomic_store(&g_spam_on, 1);
    for (int i = 0; i < iters; i++) {
        int s = submit_once();
        if (s != 0) { fprintf(stderr, "submit failed: %d (%s)\n", s, strerror(-s)); a_waitfail++; continue; }
        int rc = rocket_bo_prep(g_fd, &g_output, 0, 2000000000LL);   /* 2 s */
        if (rc != 0) { a_waitfail++; drain(); continue; }
        if (check_output() != 0) a_corrupt++;
    }
    atomic_store(&g_spam_on, 0);
    long sigs = (long)g_sig_count;
    printf("Phase A (EINTR): %d/%d clean, wait-failures=%d, corrupt=%d, signals_delivered=%ld\n",
           iters - a_waitfail, iters, a_waitfail, a_corrupt, sigs);

    /* ---- Phase B: UINT64_MAX must wait, not instant-timeout ------------- */
    int b_instant = 0, b_corrupt = 0;
    int64_t dt_min = INT64_MAX, dt_max = 0, dt_sum = 0;
    for (int i = 0; i < iters; i++) {
        int s = submit_once();
        if (s != 0) { fprintf(stderr, "submit failed: %d (%s)\n", s, strerror(-s)); b_instant++; continue; }
        int64_t t0 = now_ns();
        int rc = rocket_bo_prep(g_fd, &g_output, 0, UINT64_MAX);     /* "wait forever" */
        int64_t dt = now_ns() - t0;
        if (rc != 0) { b_instant++; drain(); continue; }
        if (dt < dt_min) dt_min = dt;
        if (dt > dt_max) dt_max = dt;
        dt_sum += dt;
        if (check_output() != 0) b_corrupt++;
    }
    printf("Phase B (UINT64_MAX): %d/%d waited, instant-timeouts=%d, corrupt=%d, "
           "wait ns min/avg/max = %lld/%lld/%lld\n",
           iters - b_instant, iters, b_instant, b_corrupt,
           (long long)(b_instant < iters ? dt_min : 0),
           (long long)(b_instant < iters ? dt_sum / (iters - b_instant) : 0),
           (long long)dt_max);

    atomic_store(&g_helper_stop, 1);
    pthread_join(helper, NULL);

    /* ---- verdict -------------------------------------------------------- */
    int pass = (a_waitfail == 0 && a_corrupt == 0 && b_instant == 0 && b_corrupt == 0);
    if (sigs == 0)
        printf("WARNING: no signals were delivered — Phase A did not exercise the "
               "EINTR path (inconclusive, not a pass).\n");
    printf("%s\n", (pass && sigs > 0) ? "OK: PREP_BO wait is signal- and UINT64_MAX-robust"
                                      : "FAILED");

    free(g_A); free(g_B); free(g_Cref);
    rocket_bo_free(g_fd, &g_guard);
    rocket_bo_free(g_fd, &g_regcmd); rocket_bo_free(g_fd, &g_input);
    rocket_bo_free(g_fd, &g_weights); rocket_bo_free(g_fd, &g_output);
    rocket_close(g_fd);
    return (pass && sigs > 0) ? 0 : 1;
}
