// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_matmul_mt.c — multi-core fan-out over the validated single-fd tiled
 * matmul. The rocket driver makes one drm_sched per NPU core and one scheduling
 * ENTITY per open fd; an entity pins to one core while it has queued work, so a
 * single fd serializes onto one core. Driving N independent fds (one per worker
 * thread) lets the kernel dispatch across all 3 cores in parallel — measured
 * ~2.94x at 3 threads, ~3.06x at 4 (tests/multicore_threads.c).
 *
 * Strategy: split the output columns N into `nthreads` contiguous slices; each
 * worker opens its own fd and runs rocket_matmul_fp16() UNCHANGED on its slice
 * C[:,n0:n1] = A * B[n0:n1,:]^T, then scatters its dense [M,nsub] result into the
 * strided columns of C. A is re-packed per worker (each needs all of A); that
 * work runs concurrently on separate CPU cores, and B is naturally partitioned.
 */
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_affinity.h"

typedef struct {
    int M, K, N, n0, nsub;
    const _Float16 *A, *B;
    _Float16 *C;
    int ret;
    int idx;            /* worker index, for big-core pinning */
    int core_base;      /* big-core rotation base inherited from the caller thread */
} mt_arg;

static void *mt_worker(void *a)
{
    mt_arg *w = (mt_arg *)a;
    rocket_pin_worker_based(w->idx, w->core_base);   /* keep the pack/readback off the A55s */
    int fd = rocket_open();
    if (fd < 0) { w->ret = fd; return NULL; }

    /* dense scratch for this worker's slice, then scatter into C's columns */
    _Float16 *Csub = malloc((size_t)w->M * w->nsub * sizeof(_Float16));
    if (!Csub) { rocket_close(fd); w->ret = -1; return NULL; }

    w->ret = rocket_matmul_fp16(fd, w->M, w->K, w->nsub,
                                w->A, w->B + (size_t)w->n0 * w->K, Csub);
    if (w->ret == 0)
        for (int m = 0; m < w->M; m++)
            memcpy(w->C + (size_t)m * w->N + w->n0,
                   Csub + (size_t)m * w->nsub,
                   (size_t)w->nsub * sizeof(_Float16));

    free(Csub);
    rocket_close(fd);
    return NULL;
}

int rocket_matmul_fp16_mt(int M, int K, int N,
                          const _Float16 *A, const _Float16 *B, _Float16 *C,
                          int nthreads)
{
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 8) nthreads = 8;

    /* per-worker column slice, rounded up to a multiple of 16 (rocket_matmul_fp16
     * requires N%16==0; since N is a multiple of 16, the last slice is too). */
    int Nstep = ((N + nthreads - 1) / nthreads + 15) / 16 * 16;
    if (Nstep < 16) Nstep = 16;

    int base = rocket_affinity_get_base();  /* spread N in-process pools across the cluster */
    pthread_t th[8];
    mt_arg args[8];
    int joinable[8] = {0};
    int nt = 0;
    for (int t = 0; t < nthreads; t++) {
        int n0 = t * Nstep;
        if (n0 >= N) break;                 /* fewer slices than threads if N small */
        int n1 = n0 + Nstep; if (n1 > N) n1 = N;
        args[nt] = (mt_arg){ M, K, N, n0, n1 - n0, A, B, C, 0, nt, base };
        if (pthread_create(&th[nt], NULL, mt_worker, &args[nt]) == 0)
            joinable[nt] = 1;
        /* else: leave joinable[nt]=0 and run it inline AFTER the spawn loop — running
         * it inline HERE would block the loop on its ~8s NPU wait and serialize every
         * not-yet-spawned worker behind it (correct result, wrong concurrency). */
        nt++;
    }

    /* Now that all creatable workers are running concurrently, run any whose
     * pthread_create failed inline (they overlap the already-spawned threads). */
    for (int t = 0; t < nt; t++)
        if (!joinable[t]) mt_worker(&args[t]);

    int ret = 0;
    for (int t = 0; t < nt; t++) {
        if (joinable[t]) pthread_join(th[t], NULL);
        if (args[t].ret) ret = args[t].ret;
    }
    return ret;
}
