// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * uapi_regcmd_fault_rocket.c — an NPU DMA fault a client can ask for, and what
 * the driver does with it.
 *
 * drm_rocket_task.regcmd is a raw 32-bit NPU IOVA. rocket_job_hw_submit() writes
 * it straight into the PC block's BASE_ADDRESS with no check that it addresses
 * anything the job's BOs mapped, so a client can point the program counter's
 * instruction fetch at unmapped IOVA. The fetch faults, and the NPU raises
 * DMA_READ_ERROR in PC_INTERRUPT_RAW_STATUS.
 *
 * rocket_job_irq_handler() reports both DMA error bits with bare WARN_ON()
 * (mainline v7.1 rocket_job.c:424-425). WARN_ON on a condition a client can
 * request is a taint, a backtrace and a dmesg flood per submit -- and on a kernel
 * booted with panic_on_warn, a panic. /dev/accel/accel0 is group `render`.
 *
 * WHAT IT FOUND ON THE RK3576 (H96 MAX M9, 7.1.3, 0001-0013): neither mode
 * reaches the WARN_ON. Both surface as `rk_iommu: Enable stall request timed
 * out` and nothing else -- 0 WARNINGs, taint unchanged, and the matmul gate is
 * 46/46 straight afterwards. The RK3576's IOMMU absorbs a bad address as a stall
 * rather than the NPU's PC raising a DMA-error bit, so on this part the WARN_ON
 * is NOT shown to be client-triggerable. The one WARNING actually seen at that
 * line came from a two-core concurrent run, not from a malformed address.
 *
 * It did establish the neighbouring fact: a job whose program faults still
 * RETIRES CLEANLY. The submit returns 0, PREP_BO returns 0, the output BO is
 * untouched, and userspace is told nothing -- which is the "the encoder writes
 * nothing" false trail, arriving from the driver rather than the encoder.
 *
 * WHAT IT FOUND ON THE RK3588 (Turing RK1, 7.1.1, rocket 081-086): the DMA-error
 * WARN_ON is not reached there either -- but a client-triggerable WARNING IS, and
 * it is somewhere else entirely. Both modes end in the job timeout, and the reset
 * that follows walks into the IOMMU core:
 *
 *   rk_iommu fdaca000.iommu: Enable stall request timed out, status: 0x2b
 *   WARNING: drivers/iommu/iommu.c:157 at __iommu_group_set_core_domain
 *    iommu_detach_group / rocket_reset.part.0 [rocket] / rocket_job_timedout
 *
 * So the reachable taint on this part is in rocket_reset()'s detach, not in the
 * IRQ handler: the NPU is still faulting, the IOMMU's stall and disable-paging
 * requests time out, and detaching the group then WARNs. Note that 083
 * (keep-the-domain-attached-across-jobs) is applied here and does not cover this
 * path -- the reset detaches explicitly. On the RK3576 the same reset produces the
 * rk_iommu stall lines and no WARNING, so the two parts differ in the consequence
 * rather than in the trigger.
 *
 * THAT IS WHAT THIS PROBE IS FOR NOW, and it is the A/B for patches/rocket/088
 * (reset the core before detaching its IOMMU group). Count the stall timeouts,
 * not the WARNINGs: the stall failure is deterministic -- exactly two lines per
 * faulting job, one per MMU bank -- while the WARN needs the SECOND handshake,
 * the one behind the default domain, to fail as well, and it is intermittent
 * enough to sit out 30 consecutive faults. On the RK1 with 081-087, 30 faults
 * (both modes, 15 rounds each) give 30 "NPU job timed out" and 60 stall
 * timeouts; with 088 on top, the same 30 timeouts and ZERO stall lines.
 * A run that reports 0 WARNINGs has therefore measured nothing on its own.
 *
 * This is a PROBE, not a gate: it deliberately faults the NPU, which costs a job
 * timeout and a core reset. Run it by hand on a board you are willing to reboot.
 * It prints the dmesg delta itself so the WARN, if any, is attributed.
 *
 * Usage: sudo -E ./uapi_regcmd_fault_rocket [read|write]
 *   read  (default) — unmapped regcmd IOVA: the PC's instruction fetch faults
 *   write           — a REAL program whose DPU output address is an unmapped IOVA,
 *                     so the CNA and CORE run to completion and only the write DMA
 *                     faults. This is the one that targets the DMA_WRITE_ERROR bit,
 *                     which is the WARN that has actually been seen (on core 1's
 *                     IRQ during a two-core run). Per part, because the geometry
 *                     encodings differ and neither runs on the other: an int8
 *                     convolution on the RK3576, an fp16 matmul on the RK3588.
 *
 * Exit: 0 = ran (read the report), 2 = no NPU device or no encoder for this part.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include "rocket_npu.h"
#include "npu_matmul.h"
#include "rocket_hw_profile.h"
#include "npu_regcmd_rk3576.h"

/* An IOVA no BO on this fd can hold. The per-fd allocator bump-starts at 0 and
 * this probe maps a few pages, so anything up in the 3 GB region is unmapped --
 * and it stays inside the 32-bit window the PC's BASE_ADDRESS register can
 * express, so the fault is the IOMMU's and not a truncation. */
#define UNMAPPED_IOVA 0xC0DE0000u

static long dmesg_lines(void)
{
    FILE *f = popen("dmesg 2>/dev/null | wc -l", "r");
    if (!f) return -1;
    long n = -1;
    if (fscanf(f, "%ld", &n) != 1) n = -1;
    pclose(f);
    return n;
}

static void dmesg_since(long from)
{
    if (from < 0) { printf("  info : dmesg unreadable (need root?)\n"); return; }
    char cmd[256];
    snprintf(cmd, sizeof cmd,
             "dmesg 2>/dev/null | tail -n +%ld | grep -iE "
             "'rocket|iommu|WARNING|Call trace|DMA' | head -30", from + 1);
    printf("  ---- : dmesg delta\n");
    fflush(stdout);
    if (system(cmd) != 0) { /* nothing matched, or no dmesg: both are results */ }
}

int main(int argc, char **argv)
{
    const char *mode = (argc > 1) ? argv[1] : "read";

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "no rocket device (skip)\n"); return 2; }

    printf("== rocket client-requested DMA fault probe (%s) ==\n", mode);
    printf("  info : drm_rocket_task.regcmd goes to PC BASE_ADDRESS unvalidated\n");

    /* A real in BO and a real out BO, so the job itself is well-formed and the
     * only thing wrong is where the program counter is told to fetch from.
     * Sized for the largest program either `write` branch below builds. */
    rocket_bo in = {0}, out = {0};
    if (rocket_bo_alloc(fd, 64 * 1024, &in) != 0 ||
        rocket_bo_alloc(fd, 64 * 1024, &out) != 0) {
        fprintf(stderr, "  FAIL : CREATE_BO\n");
        rocket_close(fd);
        return 2;
    }

    /* `write` needs a program the part actually executes, so the CNA and CORE run
     * and the DPU is the only block that faults. Everything but the output address
     * is real; the cubes are left zeroed because the arithmetic is irrelevant here. */
    rocket_bo wt = {0}, bias = {0}, rc_bo = {0};
    uint64_t ops[RK3576_CONV_TASK_OPS] = {0};
    uint32_t regcmd_iova, regcmd_count = 16;
    rocket_bo *bos[] = { &in, &out, &wt, &bias, &rc_bo };
    unsigned nbo = 2;
    const int is_rk3576 = !strcmp(rocket_hw_current()->name, "rk3576");

    if (strcmp(mode, "write") == 0) {
        uint32_t count = 0;

        if (rocket_bo_alloc(fd, 64 * 1024, &wt) != 0 ||
            rocket_bo_alloc(fd, 4096, &bias) != 0 ||
            rocket_bo_alloc(fd, sizeof ops, &rc_bo) != 0) {
            fprintf(stderr, "  FAIL : CREATE_BO for the program\n");
            goto out;
        }
        nbo = 5;

        /* The program has to be one THIS part executes, or the CNA and CORE never
         * run and the DPU's write DMA is never the thing that faults. Each part's
         * own generator, then: they emit different geometry-register encodings and
         * neither runs on the other. */
        if (is_rk3576) {
            conv_params_t p = {
                .ic = 32, .ih = 8, .iw = 8, .oc = 32, .oh = 8, .ow = 8,
                .kh = 1, .kw = 1, .stride_y = 1, .stride_x = 1,
                .dil_y = 1, .dil_x = 1, .ih_full = 8, .oh_full = 8,
                .int8_out = 1, .tasks = ops,
                .in_scale = 1.0f, .w_scale = 1.0f, .out_scale = 1.0f,
                .input_dma   = (uint32_t)in.dma_address,
                .weights_dma = (uint32_t)wt.dma_address,
                .bias_dma    = (uint32_t)bias.dma_address,
                .output_dma  = UNMAPPED_IOVA,   /* the ONLY thing wrong with it */
            };
            if (gen_conv2d_int8_rk3576(&p) != 0 || p.task_count == 0) {
                printf("  info : this part has no int8 conv encoder -- `write` needs "
                       "one (skip)\n");
                goto out;
            }
            count = p.task_count;
            printf("  info : a real 32x8x8 -> 32x8x8 int8 conv, %u ops, output aimed "
                   "at unmapped 0x%x\n", count, UNMAPPED_IOVA);
        } else {
            /* The RK3588 side: an fp16 matmul, whose operands may be zero because
             * only the destination of its writes is under test. */
            matmul_params_t p = {
                .m = 64, .k = 64, .n = 64, .tasks = ops, .fp32tofp16 = 1,
                .input_dma   = (uint32_t)in.dma_address,
                .weights_dma = (uint32_t)wt.dma_address,
                .output_dma  = UNMAPPED_IOVA,   /* the ONLY thing wrong with it */
            };
            if (gen_matmul_fp16(&p) != 0 || p.task_count == 0) {
                printf("  info : gen_matmul_fp16 refused on this part -- `write` needs "
                       "a program it runs (skip)\n");
                goto out;
            }
            count = p.task_count;
            printf("  info : a real 64x64x64 fp16 matmul, %u ops, output aimed at "
                   "unmapped 0x%x\n", count, UNMAPPED_IOVA);
        }

        rocket_bo_prep(fd, &rc_bo, 1, 0);
        memcpy(rc_bo.ptr, ops, count * sizeof(uint64_t));
        rocket_bo_fini(fd, &rc_bo);
        regcmd_iova  = (uint32_t)rc_bo.dma_address;
        regcmd_count = count;
    } else {
        regcmd_iova = UNMAPPED_IOVA;
    }

    printf("  info : in BO IOVA 0x%llx, out BO IOVA 0x%llx, regcmd IOVA 0x%x\n",
           (unsigned long long)in.dma_address, (unsigned long long)out.dma_address,
           regcmd_iova);

    long mark = dmesg_lines();

    rocket_task_desc t = { .regcmd = regcmd_iova, .regcmd_count = regcmd_count };
    uint32_t inh[] = { in.handle, wt.handle, bias.handle, rc_bo.handle };
    uint32_t outh[] = { out.handle };
    uint32_t nin = (nbo == 5) ? 4u : 1u;

    printf("  ---- : submitting\n");
    fflush(stdout);
    int rc = rocket_submit_tasks(fd, &t, 1, inh, nin, outh, 1);
    printf("  info : submit rc=%d (%s)\n", rc, rc ? strerror(-rc) : "accepted");
    fflush(stdout);

    /* Wait the job out. A faulted program never completes, so this is the job
     * timeout and the core reset that follows it -- both expected. */
    int prc = rocket_bo_prep(fd, &out, 0, 3ULL * 1000000000ULL);
    printf("  info : PREP_BO on the output rc=%d (%s)\n", prc,
           prc ? strerror(prc < 0 ? -prc : prc) : "job retired");
    rocket_bo_fini(fd, &out);

    sleep(1);
    dmesg_since(mark);

out:
    for (unsigned i = 0; i < nbo; i++)
        if (bos[i]->handle) rocket_bo_free(fd, bos[i]);
    rocket_close(fd);

    printf("\nprobe ran; read the dmesg delta above for WARNING / rocket_job_irq_handler\n");
    return 0;
}
