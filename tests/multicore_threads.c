// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * multicore_threads.c — does the rocket driver run the 3 NPU cores in parallel
 * when driven by MULTIPLE fds/entities (not multiple jobs on one fd)?
 *
 * Why this differs from multicore_probe.c: the driver makes one drm_sched per
 * core and one scheduling ENTITY per open fd spanning all cores. A DRM entity
 * binds to ONE scheduler (core) while it has queued work (in-order guarantee),
 * so many jobs on ONE fd serialize onto ONE core (what multicore_probe showed).
 * Tomeu's blog: dispatch happens across cores when driven by multiple threads.
 * So: N threads, each with its OWN fd/BOs/entity, each looping submit+wait. If
 * the cores run in parallel, aggregate throughput scales with N up to 3.
 *
 * Build:
 *   gcc -O2 -Iinclude tests/multicore_threads.c src/rocket_npu.c \
 *       src/npu_regcmd.c -o multicore_threads -lm -lpthread
 * Run:
 *   sudo ./multicore_threads 1    # baseline (1 thread/core)
 *   sudo ./multicore_threads 3    # 3 threads -> 3 cores?
 *   for N in 1 2 3 4; do sudo ./multicore_threads $N; done
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#include "rocket_npu.h"
#include "npu_matmul.h"

#define TASKS_PER_JOB 64
#define REPS          40
#define M 256
#define K 384
#define N 256

static pthread_barrier_t barrier;
static double g_elapsed[8];

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6; }

static void *worker(void *arg)
{
    long id = (long)arg;
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr,"[t%ld] open failed\n",id); return NULL; }

    size_t in_sz=(size_t)M*K*sizeof(_Float16)+4096, wt_sz=(size_t)N*K*sizeof(_Float16)+4096,
           out_sz=(size_t)M*N*sizeof(_Float16)+4096;
    rocket_bo in={0},wt={0},rc={0},out={0};
    if (rocket_bo_alloc(fd,in_sz,&in)||rocket_bo_alloc(fd,wt_sz,&wt)||
        rocket_bo_alloc(fd,256*sizeof(uint64_t),&rc)||rocket_bo_alloc(fd,out_sz,&out)){
        fprintf(stderr,"[t%ld] alloc failed\n",id); return NULL; }

    rocket_bo_prep(fd,&in,1,0); memset(in.ptr,0,in.size);
    for(int h=1;h<=M;h++)for(int c=1;c<=K;c++)
        ((_Float16*)in.ptr)[feature_data(K,M,1,8,c,h,1)]=(_Float16)(0.01f*((h+c)%7));
    rocket_bo_fini(fd,&in);
    rocket_bo_prep(fd,&wt,1,0); memset(wt.ptr,0,wt.size);
    for(int k=1;k<=N;k++)for(int c=1;c<=K;c++)
        ((_Float16*)wt.ptr)[weight_fp16(K,k,c)]=(_Float16)(0.01f*((k*c)%5));
    rocket_bo_fini(fd,&wt);

    uint64_t ops[256]={0};
    matmul_params_t p={.m=M,.k=K,.n=N,.input_dma=(uint32_t)in.dma_address,
        .weights_dma=(uint32_t)wt.dma_address,.output_dma=(uint32_t)out.dma_address,
        .tasks=ops,.fp32tofp16=1};
    if(gen_matmul_fp16(&p)){fprintf(stderr,"[t%ld] gen failed\n",id);return NULL;}
    rocket_bo_prep(fd,&rc,1,0); memcpy(rc.ptr,ops,p.task_count*sizeof(uint64_t)); rocket_bo_fini(fd,&rc);

    rocket_task_desc *tasks=malloc(TASKS_PER_JOB*sizeof(*tasks));
    for(int i=0;i<TASKS_PER_JOB;i++) tasks[i]=(rocket_task_desc){(uint32_t)rc.dma_address,p.task_count};
    uint32_t inh[]={in.handle,wt.handle,rc.handle}, outh[]={out.handle};

    pthread_barrier_wait(&barrier);
    double t0=now_ms();
    for(int r=0;r<REPS;r++){
        if(rocket_submit_tasks(fd,tasks,TASKS_PER_JOB,inh,3,outh,1)){fprintf(stderr,"[t%ld] submit\n",id);break;}
        if(rocket_bo_prep(fd,&out,0,2000000000LL)){fprintf(stderr,"[t%ld] timeout\n",id);break;}
        rocket_bo_fini(fd,&out);
    }
    g_elapsed[id]=now_ms()-t0;

    free(tasks);
    rocket_bo_free(fd,&in);rocket_bo_free(fd,&wt);rocket_bo_free(fd,&rc);rocket_bo_free(fd,&out);
    rocket_close(fd);
    return NULL;
}

int main(int argc,char**argv)
{
    int nthreads = argc>=2 ? atoi(argv[1]) : 3;
    if(nthreads<1) nthreads=1;
    if(nthreads>8) nthreads=8;
    pthread_barrier_init(&barrier,NULL,nthreads);
    pthread_t th[8];
    for(long i=0;i<nthreads;i++) pthread_create(&th[i],NULL,worker,(void*)i);
    for(int i=0;i<nthreads;i++) pthread_join(th[i],NULL);

    int total_jobs=nthreads*REPS;
    double max_e=0; for(int i=0;i<nthreads;i++) if(g_elapsed[i]>max_e) max_e=g_elapsed[i];
    printf("threads=%d  jobs/thread=%d  total_jobs=%d  wall(post-barrier max)=%.1f ms\n",
           nthreads,REPS,total_jobs,max_e);
    printf("  throughput=%.1f jobs/s  (%.2f ms/job aggregate)\n",
           total_jobs/(max_e/1000.0), max_e/total_jobs);
    printf("  => compare jobs/s across N: scales with N (up to 3) => cores parallel; flat => serialized\n");
    pthread_barrier_destroy(&barrier);
    return 0;
}
