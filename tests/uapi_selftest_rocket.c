// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * uapi_selftest_rocket.c — runtime conformance probe for the mainline rocket
 * DRM-accel uAPI. It checks the drm_rocket_* contracts the whole library
 * leans on, so a kernel that drifts (a new SoC port, a uAPI revision, a distro
 * kernel) fails HERE with a named diagnostic instead of silently mis-waiting or
 * mis-addressing deep in the matmul path.
 *
 * What it pins, and why each matters:
 *   1. Driver identity + version           — the shim only speaks "rocket".
 *   2. CREATE_BO contract                   — handle/ptr/dma_address valid, BO
 *      + 32-bit regcmd IOVA window            page-aligned, and the FIRST BO sits
 *                                             in the low 4 GB (the NPU PC's
 *                                             BASE_ADDRESS reg is 32-bit; a regcmd
 *                                             BO above 4 GB silently hangs).
 *   3. PREP/FINI cache round-trip           — rocket BOs are cached; a write that
 *                                             survives fini->prep proves the sync.
 *   4. PREP_BO DEADLINE SEMANTICS           — the load-bearing footgun: the kernel
 *      (the cross-kernel canary)              runs timeout_ns through
 *                                             drm_timeout_abs_to_jiffies() => it is
 *                                             an ABSOLUTE CLOCK_MONOTONIC deadline,
 *                                             NOT a duration. The shim converts a
 *                                             relative timeout to absolute; if a
 *                                             kernel changed this, every job wait
 *                                             would degrade to an immediate -EBUSY
 *                                             poll. We verify both directions on a
 *                                             real in-flight job.
 *   5. IOVA growth report                   — where successive BOs land (per-fd
 *                                             window); informational.
 *
 * This is a GATE (registered in CTest): exit 0 = all required checks pass, 1 =
 * a required check failed, 2 = no NPU device (skip). The deadline canary's
 * "job already finished" branch is informational, never a failure.
 *
 * Build: linked against rocketnpu by CMake. Run: ./uapi_selftest_rocket
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>

#include <drm/rocket_accel.h>   /* raw DRM_IOCTL_ROCKET_* for the deadline canary */

#include "rocket_npu.h"
#include "npu_matmul.h"

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (cond) { printf("  ok   : %s\n", msg); } \
    else { printf("  FAIL : %s\n", msg); fails++; } } while (0)
#define INFO(...) do { printf("  info : "); printf(__VA_ARGS__); printf("\n"); } while (0)

static int64_t mono_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (int64_t)ts.tv_sec*1000000000LL + ts.tv_nsec; }

/* DRM_IOCTL_VERSION lives in <drm/drm.h>, pulled in by rocket_accel.h. */
#include <drm/drm.h>

int main(void)
{
    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no rocket device (skip)\n"); return 2; }

    printf("== rocket uAPI self-test ==\n");

    /* 1. Driver identity + version --------------------------------------- */
    {
        char name[32] = {0};
        struct drm_version dv = { .name = name, .name_len = sizeof(name)-1 };
        int rc = ioctl(fd, DRM_IOCTL_VERSION, &dv);
        CHECK(rc == 0, "DRM_IOCTL_VERSION");
        CHECK(rc == 0 && strcmp(name, "rocket") == 0, "driver name == \"rocket\"");
        INFO("driver \"%s\" v%d.%d.%d", name, dv.version_major, dv.version_minor,
             dv.version_patchlevel);
    }

    /* 2. CREATE_BO contract + 32-bit regcmd window ----------------------- */
    rocket_bo b0 = {0};
    {
        int rc = rocket_bo_alloc(fd, 1000, &b0);   /* deliberately non-page-multiple */
        CHECK(rc == 0, "CREATE_BO(1000) succeeds");
        CHECK(rc == 0 && b0.handle != 0,       "BO handle non-zero");
        CHECK(rc == 0 && b0.ptr != NULL,       "BO mmap pointer valid");
        /* NPU FACT: the per-fd IOVA allocator BUMP-STARTS AT 0, so the FIRST BO on a
         * fresh fd legitimately gets dma_address == 0x0. dma_address is therefore NOT
         * a validity sentinel — handle/ptr are. The real invariants are page-alignment
         * and the 32-bit regcmd window. */
        CHECK(rc == 0 && (b0.dma_address & 0xFFF) == 0,
              "BO dma_address page-aligned (kernel rounds size to a page)");
        /* The regcmd BO must live below 4 GB (32-bit NPU PC BASE_ADDRESS). */
        CHECK(rc == 0 && (b0.dma_address >> 32) == 0,
              "first BO in low 4 GB (regcmd-addressable IOVA window)");
        if (b0.dma_address == 0)
            INFO("dma_address = 0x0 (allocator bump-starts at 0 — a VALID IOVA, not an error)");
        else
            INFO("dma_address = 0x%llx", (unsigned long long)b0.dma_address);
    }

    /* 3. PREP/FINI cache round-trip -------------------------------------- */
    if (b0.ptr) {
        rocket_bo_prep(fd, &b0, 1, 0);            /* own for CPU write */
        for (int i = 0; i < 250; i++) ((uint8_t*)b0.ptr)[i] = (uint8_t)(i*7+3);
        rocket_bo_fini(fd, &b0);                  /* flush for device   */
        rocket_bo_prep(fd, &b0, 0, 0);            /* invalidate for CPU read */
        int intact = 1;
        for (int i = 0; i < 250; i++) if (((uint8_t*)b0.ptr)[i] != (uint8_t)(i*7+3)) intact = 0;
        CHECK(intact, "PREP/FINI cache-sync preserves a CPU-written pattern");
        rocket_bo_fini(fd, &b0);
    }

    /* 4. PREP_BO deadline semantics on a REAL in-flight job --------------- */
    {
        const int M = 64, K = 64, N = 64;
        const int NTASK = 512;   /* enough tasks that the job is briefly in-flight */
        rocket_bo in={0}, wt={0}, rc_bo={0}, out={0};
        size_t in_sz=(size_t)M*K*sizeof(_Float16)+4096,
               wt_sz=(size_t)N*K*sizeof(_Float16)+4096,
               out_sz=(size_t)M*N*sizeof(_Float16)+4096;
        int ok = !rocket_bo_alloc(fd,in_sz,&in) && !rocket_bo_alloc(fd,wt_sz,&wt) &&
                 !rocket_bo_alloc(fd,512*sizeof(uint64_t),&rc_bo) &&
                 !rocket_bo_alloc(fd,out_sz,&out);
        CHECK(ok, "alloc job BOs (in/wt/regcmd/out)");
        if (ok) {
            rocket_bo_prep(fd,&in,1,0);  memset(in.ptr,0,in.size);
            for(int h=1;h<=M;h++) for(int c=1;c<=K;c++)
                ((_Float16*)in.ptr)[feature_data(K,M,1,8,c,h,1)]=(_Float16)(0.01f*((h+c)%7));
            rocket_bo_fini(fd,&in);
            rocket_bo_prep(fd,&wt,1,0);  memset(wt.ptr,0,wt.size);
            for(int k=1;k<=N;k++) for(int c=1;c<=K;c++)
                ((_Float16*)wt.ptr)[weight_fp16(K,k,c)]=(_Float16)(0.01f*((k*c)%5));
            rocket_bo_fini(fd,&wt);

            uint64_t ops[512]={0};
            matmul_params_t p={.m=M,.k=K,.n=N,.input_dma=(uint32_t)in.dma_address,
                .weights_dma=(uint32_t)wt.dma_address,.output_dma=(uint32_t)out.dma_address,
                .tasks=ops,.fp32tofp16=1};
            int genrc = gen_matmul_fp16(&p);
            CHECK(genrc == 0, "gen_matmul_fp16 (build a real regcmd)");
            rocket_bo_prep(fd,&rc_bo,1,0); memcpy(rc_bo.ptr,ops,p.task_count*sizeof(uint64_t));
            rocket_bo_fini(fd,&rc_bo);

            rocket_task_desc tasks[512];
            for(int i=0;i<NTASK;i++) tasks[i]=(rocket_task_desc){(uint32_t)rc_bo.dma_address,p.task_count};
            uint32_t inh[]={in.handle,wt.handle,rc_bo.handle}, outh[]={out.handle};

            /* (a) ABSOLUTE-deadline canary: submit async, then a RAW PREP_BO with a
             * deadline 1 ms IN THE PAST. Absolute semantics => return promptly
             * (not hang); a non-zero errno means the fence is unsignalled (job in
             * flight) — the expected, healthy reading. A relative-duration kernel
             * would instead WAIT ~0 and is indistinguishable here, so this branch
             * is informational; check (b) is the hard regression guard. */
            if (!rocket_submit_tasks(fd,tasks,NTASK,inh,3,outh,1)) {
                struct drm_rocket_prep_bo past = { .handle = out.handle,
                    .timeout_ns = mono_ns() - 1000000LL /* 1 ms ago */ };
                int64_t t0 = mono_ns();
                int prc = ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, &past);
                int64_t dt = mono_ns() - t0;
                int e = prc<0 ? errno : 0;
                CHECK(dt < 500000000LL, "raw PREP_BO with a PAST absolute deadline returns promptly (no hang)");
                if (prc < 0) INFO("  past-deadline poll -> errno %d (%s) = fence unsignalled, job in-flight",
                                  e, strerror(e));
                else         INFO("  past-deadline poll -> 0 (job already completed; canary inconclusive)");

                /* (b) HARD GUARD: the shim's relative->absolute conversion must let
                 * a generous relative wait actually complete the job. If the
                 * conversion regressed (passing a raw small value as absolute), the
                 * kernel would see a past deadline and return -EBUSY even though the
                 * job finishes fine. */
                int wrc = rocket_bo_prep(fd, &out, 0, 3000000000LL /* 3 s relative */);
                CHECK(wrc == 0, "shim PREP_BO (relative 3s) completes the in-flight job (abs-deadline conversion OK)");
                rocket_bo_fini(fd, &out);

                /* FINI_BO must always succeed. */
                CHECK(rocket_bo_fini(fd, &in) == 0, "FINI_BO succeeds");
            } else {
                CHECK(0, "rocket_submit_tasks (async job)");
            }
        }
        rocket_bo_free(fd,&in); rocket_bo_free(fd,&wt);
        rocket_bo_free(fd,&rc_bo); rocket_bo_free(fd,&out);
    }

    /* 5. IOVA growth report (informational) ------------------------------ */
    {
        rocket_bo bos[8]; int n=0; int all_low=1; int monotonic=1;
        uint64_t lo=~0ULL, hi=0, prev=0;
        for (; n<8; n++) {
            if (rocket_bo_alloc(fd, 1<<20, &bos[n])) break;
            uint64_t a = bos[n].dma_address;
            if (a < lo) lo = a;
            if (a + (1<<20) > hi) hi = a + (1<<20);
            if ((a>>32) != 0) all_low = 0;
            if (n > 0 && a <= prev) monotonic = 0;
            prev = a;
        }
        INFO("allocated %d x 1 MB; IOVA span [0x%llx .. 0x%llx], all<4GB=%d",
             n, (unsigned long long)lo, (unsigned long long)hi, all_low);
        CHECK(n > 0 && monotonic, "successive BOs get distinct ascending IOVAs (bump allocator)");
        for (int i=0;i<n;i++) rocket_bo_free(fd,&bos[i]);
    }

    rocket_bo_free(fd, &b0);
    rocket_close(fd);

    printf("== %d checks, %d failed ==\n", checks, fails);
    return fails ? 1 : 0;
}
