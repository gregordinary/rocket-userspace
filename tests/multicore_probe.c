// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * multicore_probe.c — does the rocket driver run concurrent jobs on the 3 NPU
 * cores, or serialize them?
 *
 * The uapi says tasks in ONE job run on ONE core (SRAM residency), and a submit
 * carries an ARRAY of jobs "scheduled in dependency order". So N independent jobs
 * (each writing its OWN output BO, to avoid a write-after-write dependency) should
 * spread across the 3 cores. This probe submits the SAME batch of matmul tasks as
 * J=1,2,3 parallel jobs and times the wait:
 *   wall(J=3) ≈ wall(J=1)  -> parallel across cores (multicore works, ~3x headroom)
 *   wall(J=3) ≈ 3*wall(J=1) -> serialized (one core; multicore won't help here)
 *
 * Each job is TASKS_PER_JOB copies of one fp16 tile (M=256,K=384,N=256 — the tile
 * the throughput sweep picked), enough work to dwarf timing noise. Correctness is
 * not checked here (all tasks reuse one regcmd); this measures scheduling only.
 *
 * Build:
 *   gcc -O2 -Iinclude tests/multicore_probe.c src/rocket_npu.c \
 *       src/npu_regcmd.c -o multicore_probe -lm
 * Run:  sudo ./multicore_probe
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

#define MAXC          3
#define TASKS_PER_JOB 64
#define REPS          10

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void)
{
    const int M = 256, K = 384, N = 256;
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open failed (%d)\n", fd); return 1; }

    size_t in_slot  = (size_t)M * K, wt_slot = (size_t)N * K, out_slot = (size_t)M * N;
    /* FULLY INDEPENDENT per-core resources: own in, wt, regcmd, out. This rules
     * out the kernel serializing on shared read BOs — if even independent jobs
     * serialize, the driver is genuinely single-core / in-order. */
    rocket_bo in[MAXC]={{0}}, wt[MAXC]={{0}}, regcmd[MAXC]={{0}}, out[MAXC]={{0}};
    for (int c = 0; c < MAXC; c++)
        if (rocket_bo_alloc(fd, in_slot*sizeof(_Float16)+4096, &in[c]) ||
            rocket_bo_alloc(fd, wt_slot*sizeof(_Float16)+4096, &wt[c]) ||
            rocket_bo_alloc(fd, 256*sizeof(uint64_t), &regcmd[c]) ||
            rocket_bo_alloc(fd, out_slot*sizeof(_Float16)+4096, &out[c])) {
            fprintf(stderr, "alloc failed\n"); return 1; }

    uint32_t rc_count = 0;
    rocket_task_desc *tasks[MAXC];
    for (int c = 0; c < MAXC; c++) {
        /* pack arbitrary fp16 data (values irrelevant — we time scheduling) */
        rocket_bo_prep(fd, &in[c], 1, 0); memset(in[c].ptr, 0, in[c].size);
        for (int h = 1; h <= M; h++) for (int x = 1; x <= K; x++)
            ((_Float16*)in[c].ptr)[feature_data(K, M, 1, 8, x, h, 1)] = (_Float16)(0.01f*((h+x)%7));
        rocket_bo_fini(fd, &in[c]);
        rocket_bo_prep(fd, &wt[c], 1, 0); memset(wt[c].ptr, 0, wt[c].size);
        for (int k = 1; k <= N; k++) for (int x = 1; x <= K; x++)
            ((_Float16*)wt[c].ptr)[weight_fp16(K, k, x)] = (_Float16)(0.01f*((k*x)%5));
        rocket_bo_fini(fd, &wt[c]);

        uint64_t ops[256] = {0};
        matmul_params_t p = { .m=M,.k=K,.n=N, .input_dma=(uint32_t)in[c].dma_address,
            .weights_dma=(uint32_t)wt[c].dma_address, .output_dma=(uint32_t)out[c].dma_address,
            .tasks=ops, .fp32tofp16=1 };
        if (gen_matmul_fp16(&p)) { fprintf(stderr, "gen failed\n"); return 1; }
        rocket_bo_prep(fd, &regcmd[c], 1, 0);
        memcpy(regcmd[c].ptr, ops, p.task_count*sizeof(uint64_t));
        rocket_bo_fini(fd, &regcmd[c]);
        rc_count = p.task_count;

        tasks[c] = malloc(TASKS_PER_JOB*sizeof(rocket_task_desc));
        for (int i = 0; i < TASKS_PER_JOB; i++)
            tasks[c][i] = (rocket_task_desc){ (uint32_t)regcmd[c].dma_address, rc_count };
    }

    printf("probe: tile %dx%dx%d, %d tasks/job, %d reps, INDEPENDENT BOs/core\n",
           M,K,N, TASKS_PER_JOB, REPS);
    double base = 0;
    for (int J = 1; J <= MAXC; J++) {
        rocket_job_desc jobs[MAXC];
        for (int j = 0; j < J; j++) {
            static uint32_t inh[MAXC][3];
            inh[j][0]=in[j].handle; inh[j][1]=wt[j].handle; inh[j][2]=regcmd[j].handle;
            jobs[j] = (rocket_job_desc){ tasks[j], TASKS_PER_JOB, inh[j], 3, &out[j].handle, 1 };
        }

        double best = 1e9;
        for (int r = 0; r < REPS; r++) {
            double t0 = now_ms();
            if (rocket_submit_jobs(fd, jobs, J)) { fprintf(stderr,"submit failed\n"); return 1; }
            for (int j = 0; j < J; j++)
                if (rocket_bo_prep(fd, &out[j], 0, 2000000000LL)) { fprintf(stderr,"timeout\n"); return 1; }
            for (int j = 0; j < J; j++) rocket_bo_fini(fd, &out[j]);
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        if (J == 1) base = best;
        printf("J=%d jobs (%d tasks total): %6.2f ms  (%.2fx vs J=1, ideal-parallel=1.00x)\n",
               J, J*TASKS_PER_JOB, best, best/base);
    }
    printf("=> %s\n",
           "interpret: ~1x across J means cores run in parallel; ~Jx means serialized");

    for (int c = 0; c < MAXC; c++) {
        free(tasks[c]);
        rocket_bo_free(fd, &in[c]); rocket_bo_free(fd, &wt[c]);
        rocket_bo_free(fd, &regcmd[c]); rocket_bo_free(fd, &out[c]);
    }
    rocket_close(fd);
    return 0;
}
