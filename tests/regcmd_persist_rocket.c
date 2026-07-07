// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * regcmd_persist_rocket.c — does the NPU's register state PERSIST across the
 * tasks of one job, so later tasks can emit only the changed registers (a "delta"
 * regcmd) + the enable, instead of the full ~126-op program every task?
 *
 * If yes, the tiled/multi-tile hot path (and the dispatch-bound detection 1x1s) could
 * ship one full task then tiny delta tasks — fewer regcmd words + fewer PC register
 * writes per task. Sibling of the "CBUF persists across tasks" finding.
 *
 * CONCLUSION (2026-06-22, RK3588, DETERMINISTIC): delta-regcmd is NOT usable.
 *   - A 6-8 op delta task leaves the output UNTOUCHED (compute does not fire) — every
 *     run, both S_POINTER=0 and 0xE.
 *   - It only "passes" when a FULL job ran immediately before (even in a separate
 *     process): the NPU register file is NOT cleared between jobs/processes, so a delta
 *     inherits a prior FULL job's leftover config. That is unsafe to exploit (depends on
 *     unpredictable global state across 3 cores / multiple fds), not within-job chaining.
 *   - So each task must carry its FULL self-contained regcmd. The safe regcmd
 *     optimization is to cache + patch the full regcmd per tile; no gen cost).
 *   NPU fact learned: the register file persists globally (not reset on job/process
 *   boundaries).
 *
 * Ping-pong register groups: every task starts with `DPU_S_POINTER = 0xE`
 * (POINTER_PP_MODE | EXECUTER_PP_EN | POINTER_PP_EN) = NVDLA dual-register-group
 * ping-pong. We probe both S_POINTER = 0x0 (single group) and 0xE (PP).
 *
 * PROBE: a 2-task job. task0 = full regcmd, A0xB0 -> outA. task1 -> outB:
 *   MODE=full     task1 = full regcmd (A0xB0, output=outB)  (CONTROL: must pass)
 *   MODE=minimal  task1 = {S_POINTER, DPU_DST_BASE_ADD=outB, PC trailer} (6 ops),
 *                 recomputes A0xB0 to outB -> tests pure register persistence.
 *   MODE=tile     task1 = {S_POINTER, CNA_FEATURE_DATA_ADDR=A1, CNA_DCOMP_ADDR0=B1,
 *                 DPU_DST_BASE_ADD=outB, PC trailer} (7 ops), computes a DIFFERENT
 *                 tile A1xB1 -> the real tiled-matmul / detection-1x1 use case.
 *   ROCKET_PERSIST_SPTR=0|0xE  S_POINTER for BOTH tasks (default 0xE).
 *
 * Usage: regcmd_persist_rocket [M K N]   (default 4 64 32; M%4 K%32 N%32, single tile)
 * Exit: 0 PASS, 1 FAIL, 2 SKIP (no NPU). A delta-mode PERSIST=NO is a documented
 * outcome (exit 0); only a wrong CONTROL run or a corrupt outA is a hard FAIL.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "npu_hw.h"          /* NPUOP, OP_*, register addresses */

#define CK(c) do { int _r=(c); if(_r) fprintf(stderr,"%s -> %d (%s)\n",#c,_r,strerror(-_r)); } while(0)

static void ref_i8(int m,int k,int n,const int8_t*A,const int8_t*B,int32_t*C){
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){int64_t s=0;
        for(int l=0;l<k;l++)s+=(int32_t)A[i*k+l]*(int32_t)B[j*k+l];
        C[i*n+j]=(int32_t)s;}
}
static void patch_val(uint64_t*ops,int n,uint16_t reg,uint32_t val){
    for(int i=0;i<n;i++) if((ops[i]&0xffff)==reg){
        uint16_t op=(uint16_t)(ops[i]>>48); ops[i]=NPUOP(op,val,reg); return; }
}
static int count_bad(const int32_t*od,const int32_t*C,int M,int N){
    int bad=0; for(int m=1;m<=M;m++)for(int n=1;n<=N;n++)
        if(od[feature_data(N,M,1,4,n,m,1)]!=C[(m-1)*N+(n-1)]) bad++;
    return bad;
}
static void pack_i8(rocket_bo*in,rocket_bo*wt,int M,int K,int N,const int8_t*A,const int8_t*B){
    int8_t*id=in->ptr,*wd=wt->ptr; memset(id,0,in->size); memset(wd,0,wt->size);
    for(int n=1;n<=N;n++)for(int k=1;k<=K;k++)wd[weight_int8(K,n,k)]=B[(n-1)*K+(k-1)];
    for(int m=1;m<=M;m++)for(int k=1;k<=K;k++)id[feature_data(K,M,1,16,k,m,1)]=A[(m-1)*K+(k-1)];
}

int main(int argc,char**argv){
    int M=4,K=64,N=32; if(argc==4){M=atoi(argv[1]);K=atoi(argv[2]);N=atoi(argv[3]);}
    if(M%4||K%32||N%32){fprintf(stderr,"need M%%4 K%%32 N%%32\n");return 1;}
    const char*mode=getenv("ROCKET_PERSIST_MODE"); if(!mode)mode="minimal";
    uint32_t sptr=getenv("ROCKET_PERSIST_SPTR")?(uint32_t)strtoul(getenv("ROCKET_PERSIST_SPTR"),0,0):0xE;
    int is_full=!strcmp(mode,"full"), is_tile=!strcmp(mode,"tile");

    int fd=rocket_open(); if(fd<0){fprintf(stderr,"no NPU -> SKIP\n");return 2;}
    rocket_bo guard={0},regcmd={0},in0={0},wt0={0},in1={0},wt1={0},outA={0},outB={0};
    int ret=0; size_t obytes=(size_t)M*N*sizeof(int32_t);
    rocket_bo_alloc(fd,4096,&guard);
    ret|=rocket_bo_alloc(fd,8192,&regcmd);
    ret|=rocket_bo_alloc(fd,(size_t)M*K,&in0); ret|=rocket_bo_alloc(fd,(size_t)N*K,&wt0);
    ret|=rocket_bo_alloc(fd,(size_t)M*K,&in1); ret|=rocket_bo_alloc(fd,(size_t)N*K,&wt1);
    ret|=rocket_bo_alloc(fd,obytes,&outA); ret|=rocket_bo_alloc(fd,obytes,&outB);
    if(ret){fprintf(stderr,"alloc\n");ret=1;goto out;}
    if((in0.dma_address|wt0.dma_address|in1.dma_address|wt1.dma_address|outA.dma_address|outB.dma_address|regcmd.dma_address)>>32){
        fprintf(stderr,"BO >32bit\n");ret=1;goto out;}

    /* task0: full regcmd, A0xB0 -> outA */
    uint64_t r0[256]={0};
    matmul_params_t p={.m=M,.k=K,.n=N,.input_dma=(uint32_t)in0.dma_address,
        .weights_dma=(uint32_t)wt0.dma_address,.output_dma=(uint32_t)outA.dma_address,.tasks=r0};
    if((ret=gen_matmul_int8(&p))){fprintf(stderr,"gen0=%d\n",ret);ret=1;goto out;}
    uint32_t n0=p.task_count; patch_val(r0,n0,DPU_S_POINTER,sptr);

    uint64_t r1[256]={0}; uint32_t n1;
    if(is_full){
        matmul_params_t q=p; q.output_dma=(uint32_t)outB.dma_address; q.tasks=r1;
        if((ret=gen_matmul_int8(&q))){fprintf(stderr,"gen1=%d\n",ret);ret=1;goto out;}
        n1=q.task_count; patch_val(r1,n1,DPU_S_POINTER,sptr);
    } else {
        int i=0;
        r1[i++]=NPUOP(OP_REG_DPU, sptr, DPU_S_POINTER);
        if(is_tile){
            r1[i++]=NPUOP(OP_REG_CNA,(uint32_t)in1.dma_address, CNA_FEATURE_DATA_ADDR);
            r1[i++]=NPUOP(OP_REG_CNA,(uint32_t)wt1.dma_address, CNA_DCOMP_ADDR0);
        }
        r1[i++]=NPUOP(OP_REG_DPU,(uint32_t)outB.dma_address, DPU_DST_BASE_ADD);
        r1[i++]=NPUOP(OP_NONE,0x0,0x0);
        r1[i++]=NPUOP(OP_REG_PC,0x0,PC_REGISTER_AMOUNTS);
        r1[i++]=NPUOP(OP_40,0x0,0x0);
        r1[i++]=NPUOP(OP_ENABLE,0x1D,PC_OPERATION_ENABLE);
        n1=i;
    }
    printf("persist probe: mode=%s S_POINTER=0x%x  task0_ops=%u task1_ops=%u  C[%d,%d]\n",
           mode,sptr,n0,n1,M,N);

    int8_t*A0=malloc((size_t)M*K),*B0=malloc((size_t)N*K);
    int8_t*A1=malloc((size_t)M*K),*B1=malloc((size_t)N*K);
    int32_t*C0=malloc(obytes),*C1=malloc(obytes);
    if(!A0||!B0||!A1||!B1||!C0||!C1){ret=1;goto outf;}
    srand(7);  for(int i=0;i<M*K;i++)A0[i]=(int8_t)(rand()%16-8); for(int i=0;i<N*K;i++)B0[i]=(int8_t)(rand()%16-8);
    srand(99); for(int i=0;i<M*K;i++)A1[i]=(int8_t)(rand()%16-8); for(int i=0;i<N*K;i++)B1[i]=(int8_t)(rand()%16-8);
    ref_i8(M,K,N,A0,B0,C0); ref_i8(M,K,N,A1,B1,C1);
    const int32_t*Cexp = is_tile ? C1 : C0;   /* what task1 should produce */

    CK(rocket_bo_prep(fd,&regcmd,1,0));
    CK(rocket_bo_prep(fd,&in0,1,0)); CK(rocket_bo_prep(fd,&wt0,1,0));
    CK(rocket_bo_prep(fd,&in1,1,0)); CK(rocket_bo_prep(fd,&wt1,1,0));
    memcpy((uint8_t*)regcmd.ptr,      r0,(size_t)n0*8);
    memcpy((uint8_t*)regcmd.ptr+4096, r1,(size_t)n1*8);
    pack_i8(&in0,&wt0,M,K,N,A0,B0); pack_i8(&in1,&wt1,M,K,N,A1,B1);
    CK(rocket_bo_fini(fd,&regcmd));
    CK(rocket_bo_fini(fd,&in0)); CK(rocket_bo_fini(fd,&wt0));
    CK(rocket_bo_fini(fd,&in1)); CK(rocket_bo_fini(fd,&wt1));
    CK(rocket_bo_prep(fd,&outA,1,0)); memset(outA.ptr,0xAA,outA.size); CK(rocket_bo_fini(fd,&outA));
    CK(rocket_bo_prep(fd,&outB,1,0)); memset(outB.ptr,0xAA,outB.size); CK(rocket_bo_fini(fd,&outB));

    rocket_task_desc td[2]={
        {.regcmd=(uint32_t)regcmd.dma_address,       .regcmd_count=n0},
        {.regcmd=(uint32_t)(regcmd.dma_address+4096), .regcmd_count=n1},
    };
    uint32_t inh[]={in0.handle,wt0.handle,in1.handle,wt1.handle,regcmd.handle};
    uint32_t outh[]={outA.handle,outB.handle};
    ret=rocket_submit_tasks(fd,td,2,inh,5,outh,2);
    if(ret){fprintf(stderr,"submit=%d\n",ret);ret=1;goto outf;}
    int prc=rocket_bo_prep(fd,&outB,0,2000000000LL);
    if(prc){fprintf(stderr,"PREP(outB)=%d (%s): job never completed\n",prc,strerror(-prc));ret=1;goto outf;}
    rocket_bo_prep(fd,&outA,0,2000000000LL);

    int touchedB=0; for(size_t i=0;i<outB.size;i++) if(((uint8_t*)outB.ptr)[i]!=0xAA){touchedB=1;break;}
    int badA=count_bad(outA.ptr,C0,M,N), badB=count_bad(outB.ptr,Cexp,M,N);
    printf("  task0->outA: %s (%d bad)\n",badA?"WRONG":"OK",badA);
    printf("  task1->outB: %s (%d bad)%s\n",badB?"WRONG":"OK",badB,touchedB?"":" [UNTOUCHED]");
    rocket_bo_fini(fd,&outA); rocket_bo_fini(fd,&outB);

    if(badA){ printf("FAILED (job did not run correctly)\n"); ret=1; }
    else if(!badB){ printf("PERSIST=YES (mode=%s sptr=0x%x): 2nd task computed correctly with %u ops\n",mode,sptr,n1); ret=0; }
    else { printf("PERSIST=NO (mode=%s sptr=0x%x): 2nd task needs the omitted registers%s\n",
                  mode,sptr,touchedB?"":" (output untouched)"); ret=is_full?1:0; }
outf:
    free(A0);free(B0);free(A1);free(B1);free(C0);free(C1);
out:
    rocket_bo_free(fd,&guard);rocket_bo_free(fd,&regcmd);
    rocket_bo_free(fd,&in0);rocket_bo_free(fd,&wt0);rocket_bo_free(fd,&in1);rocket_bo_free(fd,&wt1);
    rocket_bo_free(fd,&outA);rocket_bo_free(fd,&outB);
    rocket_close(fd); return ret;
}
