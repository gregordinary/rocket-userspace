// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * iova_ceiling_rocket.c — measure the NPU's 32-bit IOVA window.
 *
 * The NPU regcmd addresses every BO with a 32-bit field (npu_regcmd.c:
 * weights_dma & 0xFFFFFFFF), so all BOs the NPU touches must live in the low 4 GB.
 * That caps RESIDENT pre-tiled weights and resident int8 weights
 * far below a ~22 GB model: ROCKET_CACHE_MB=12000 trips "a BO dma_address exceeds 32
 * bits". This probe pins the ceiling AND whether the window is PER-fd or SHARED:
 * it allocates fixed BOs round-robin across N fds, printing each BO's device VA,
 * until every fd's VA top crosses 4 GB (or an alloc fails).
 *
 *   PER-fd 4 GB : each fd climbs its OWN 0..4 GB (VAs repeat across fds) => ~4*N GB
 *                 total resident possible -> spread weights over more fds to fit more.
 *   SHARED 4 GB : all fds draw ONE 0..4 GB (fd1's VAs continue after fd0's) => ~4 GB
 *                 hard cap no matter how many fds -> resident weights are stuck < 4 GB.
 *
 *   sudo ./iova_ceiling_rocket [n_fds=2] [chunk_mb=256]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rocket_npu.h"

#define MAXFD 8
#define MAXBO 4096

int main(int argc, char **argv)
{
    int    nfd   = argc >= 2 ? atoi(argv[1]) : 2;
    size_t chunk = (size_t)(argc >= 3 ? atoi(argv[2]) : 256) << 20;
    if (nfd < 1) nfd = 1;
    if (nfd > MAXFD) nfd = MAXFD;

    int fd[MAXFD];
    for (int i = 0; i < nfd; i++) {
        fd[i] = rocket_open();
        if (fd[i] < 0) { fprintf(stderr, "rocket_open(%d) failed\n", i); return 1; }
    }

    rocket_bo *bos = calloc(MAXBO, sizeof(rocket_bo));
    int    held[MAXFD]  = {0};
    size_t bytes[MAXFD] = {0};
    int    full[MAXFD]  = {0};   /* fd's VA top has crossed 4 GB */
    int    nbo = 0;

    printf("probing IOVA: %d fd(s), %zuMB chunks (32-bit window = 4096MB)\n\n",
           nfd, chunk >> 20);

    while (nbo < MAXBO) {
        int all_full = 1;
        for (int s = 0; s < nfd; s++) if (!full[s]) { all_full = 0; break; }
        if (all_full) { printf("\nall %d fd(s) crossed 4 GB.\n", nfd); break; }

        int f = nbo % nfd;
        if (full[f]) { nbo++; continue; }           /* skip a full fd, keep cycling */

        rocket_bo *bo = &bos[nbo];
        if (rocket_bo_alloc(fd[f], chunk, bo) != 0) {
            printf("fd[%d]: CREATE_BO FAILED after %zuMB held (%d BOs) -> kernel alloc ceiling\n",
                   f, bytes[f] >> 20, held[f]);
            full[f] = 1;                            /* treat as full */
            bo->handle = 0;                         /* nothing to free */
            nbo++;
            continue;
        }
        int over = ((bo->dma_address + chunk) >> 32) != 0;
        printf("fd[%d] BO#%-3d va=0x%010llx  top=0x%010llx  held=%4zuMB%s\n",
               f, nbo, (unsigned long long)bo->dma_address,
               (unsigned long long)(bo->dma_address + chunk),
               (bytes[f] + chunk) >> 20,
               over ? "   <-- top crosses 4 GB (NPU guard trips)" : "");
        held[f]++;
        bytes[f] += chunk;
        if (over) full[f] = 1;
        nbo++;
    }

    printf("\nSUMMARY (MB resident per fd): ");
    size_t total = 0;
    for (int i = 0; i < nfd; i++) { printf("fd[%d]=%zu ", i, bytes[i] >> 20); total += bytes[i]; }
    printf("| total=%zuMB\n", total >> 20);
    printf("READ: each fd ~4 GB (VAs repeat) => PER-FD window (more fds fit more resident).\n");
    printf("      fds share one ~4 GB climb (VAs continuous) => SHARED window (hard 4 GB cap).\n");

    for (int i = 0; i < nbo && i < MAXBO; i++)
        if (bos[i].handle) rocket_bo_free(fd[i % nfd], &bos[i]);
    for (int i = 0; i < nfd; i++) rocket_close(fd[i]);
    free(bos);
    return 0;
}
