// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * iova_rcache_rocket.c — does freeing a buffer give its address space back?
 *
 * On a driver that maps through the kernel's generic dma_map_sg() path, a freed IOVA
 * range does not go back to the rbtree: free_iova_fast() parks it in a per-CPU magazine
 * or the global depot of the IOVA rcache. alloc_iova() never consults that cache, so any
 * allocator using it sees the space as gone — permanently, and across processes, because
 * the rcache belongs to the iova_domain rather than to a file or a process.
 *
 * iova_rcache_insert() accepts only sizes up to 32 pages (128 KiB on a 4 KiB-page host),
 * so the effect is SIZE-SELECTIVE at that boundary. This is what makes it easy to miss:
 * a probe that churns megabyte buffers cannot reproduce it at any setting, because none
 * of its frees are small enough to be cached.
 *
 * The asymmetry is the whole defect. alloc_iova_fast() flushes every online CPU's
 * magazines and the whole depot when it cannot satisfy a request, and retries — so the
 * route that fills the cache can always empty it again. alloc_iova() has no such path,
 * and free_cpu_cached_iovas()/free_global_cached_iovas() are static to iova.c, so a
 * driver allocating that way has no in-kernel way to ask for the parked space back. It
 * simply returns -ENOMEM with hundreds of megabytes sitting in a cache it cannot see.
 *
 *   iova_rcache_rocket cap   <bytes> [map]              how many BOs of this size fit
 *                                                      (and where they landed)
 *   iova_rcache_rocket churn <bytes> [batch] [rounds]   allocate and free that many
 *
 * The experiment, one arm per boot (the domain is shared process-wide, so only a reboot
 * resets it) — `cap` before, `churn`, `cap` after:
 *
 *   churn size   route      expected
 *   128 KiB      generic    capacity DROPS and does not come back
 *   256 KiB      generic    no change (above the rcache bound)
 *   132 KiB      generic    no change (ONE page above the bound)
 *   128 KiB      tight      no change (never touches the rcache)
 *
 * Measured on RK3588 / vendor rknpu 0.9.8, 4400 buffers churned per arm, capacity in
 * buffers at 16/64/128/192 MiB: fresh 255/63/31/21, after a 128 KiB generic-route churn
 * 221/55/27/18, and unchanged in all three control arms. The loss saturates — a second
 * identical churn costs 3 more and a third costs 1 — because the cache fills. Spreading
 * the churn over 8 CPUs and every cached order costs 10 of the 31 128 MiB buffers, which
 * is what one real 2048-token prefill costs.
 *
 * A `cap` on the GENERIC route REPAIRS the domain, because allocating to refusal is what
 * triggers the flush. Measured on a domain a 128 KiB generic churn had taken from 31 to
 * 27 128 MiB buffers: a generic `cap` reads 31 and reaches the whole space again, and a
 * tight `cap` run after it also reads 31 — the repair is real and permanent, not a
 * different view. The same holds for a churn spread over 8 CPUs and all six orders
 * (255/63/31/21 -> 213/53/26/17 -> tight 255/63/31/21 after one generic cap).
 *
 * Which is why "route A cannot see the loss" and "route A just repaired it" are the same
 * reading, and separating them takes a THIRD measurement: re-read on the route that
 * showed the loss, AFTER the other route has run. Without that step a repair is recorded
 * as blindness.
 *
 * Two route properties that are NOT degradation, and read as it if the routes are mixed
 * within one comparison: the generic route allocates size-ALIGNED, so on a fresh domain
 * it serves 15 buffers of 192 MiB where the tight route serves 21 (192 MiB rounds its
 * alignment to 256 MiB); and the tight route packs below the generic route's first
 * address, so their `lo` differs. Both baselines are stable — take before and after on
 * the same route.
 *
 * On the mainline rocket driver there is nothing to find here: it does not mix the two
 * allocators, and every arm reads flat. That difference is the point of running it.
 *
 * Hand-run probe, not a gate. Needs the NPU device and privilege.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <rocket_npu.h>

static int addr_cmp(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Allocate this size until the driver refuses, then report both the COUNT and where the
 * addresses landed. The count alone cannot tell "the space is gone" from "the space is
 * there and this allocator cannot reach it": a route that reclaims parked ranges hands
 * back addresses inside the churned region, and one that does not never touches it. The
 * gap list is that readout — a hole the size of the churn, still unused at refusal, is
 * address space the domain holds and did not offer. */
static int do_cap(int fd, size_t sz, int map)
{
    enum { MAX = 4096 };
    rocket_bo *bo = calloc(MAX, sizeof(*bo));
    uint64_t *a = calloc(MAX, sizeof(*a));
    if (!bo || !a) { free(bo); free(a); return -1; }
    int n = 0;
    while (n < MAX && rocket_bo_alloc(fd, sz, &bo[n]) == 0) {
        a[n] = bo[n].dma_address;
        n++;
    }
    printf("cap: size=%zu fits=%d total=%.3f GiB\n",
           sz, n, (double)n * (double)sz / 1073741824.0);

    if (n > 0) {
        qsort(a, (size_t)n, sizeof(*a), addr_cmp);
        printf("cap: lo=0x%llx hi=0x%llx span=%.3f GiB\n",
               (unsigned long long)a[0],
               (unsigned long long)(a[n - 1] + sz - 1),
               (double)(a[n - 1] + sz - a[0]) / 1073741824.0);
        /* Every hole between consecutive allocations, largest first would need a sort;
         * printing them in address order is enough to see one big one. */
        uint64_t held = 0;
        int holes = 0;
        for (int i = 0; i + 1 < n; i++) {
            uint64_t gap = a[i + 1] - (a[i] + sz);
            if (gap) {
                held += gap;
                holes++;
                printf("cap: gap after 0x%llx: %llu B (%.1f MiB)\n",
                       (unsigned long long)a[i], (unsigned long long)gap,
                       (double)gap / 1048576.0);
            }
        }
        printf("cap: holes=%d held_between=%llu B (%.1f MiB)\n",
               holes, (unsigned long long)held, (double)held / 1048576.0);
        if (map)
            for (int i = 0; i < n; i++)
                printf("cap: [%d] 0x%llx\n", i, (unsigned long long)a[i]);
    }

    for (int i = 0; i < n; i++) rocket_bo_free(fd, &bo[i]);
    free(bo);
    free(a);
    return n;
}

/* Allocate `batch` buffers and free every one, `rounds` times. Nothing is retained, so
 * any capacity this costs the domain is withheld by the kernel and not held by us. */
static int do_churn(int fd, size_t sz, int batch, int rounds)
{
    rocket_bo *bo = calloc((size_t)batch, sizeof(*bo));
    if (!bo) return -1;
    long long cycled = 0, failed = 0;
    for (int r = 0; r < rounds; r++) {
        int n = 0;
        for (int i = 0; i < batch; i++) {
            if (rocket_bo_alloc(fd, sz, &bo[i]) != 0) { failed++; break; }
            n++;
        }
        for (int i = 0; i < n; i++) rocket_bo_free(fd, &bo[i]);
        cycled += n;
    }
    printf("churn: size=%zu batch=%d rounds=%d cycled=%lld alloc_failures=%lld\n",
           sz, batch, rounds, cycled, failed);
    free(bo);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s cap <bytes> [map]\n"
                "       %s churn <bytes> [batch=4400] [rounds=1]\n", argv[0], argv[0]);
        return 2;
    }
    size_t sz = (size_t)strtoull(argv[2], NULL, 0);
    if (sz == 0) { fprintf(stderr, "size must be non-zero\n"); return 2; }

    int fd = rocket_open();
    if (fd < 0) { fprintf(stderr, "rocket_open failed: %d\n", fd); return 1; }

    int rc = 0;
    if (!strcmp(argv[1], "cap")) {
        rc = do_cap(fd, sz, argc > 3 && !strcmp(argv[3], "map")) < 0;
    } else if (!strcmp(argv[1], "churn")) {
        int batch  = (argc > 3) ? atoi(argv[3]) : 4400;
        int rounds = (argc > 4) ? atoi(argv[4]) : 1;
        if (batch <= 0 || rounds <= 0) { fprintf(stderr, "batch/rounds must be positive\n"); rc = 2; }
        else rc = do_churn(fd, sz, batch, rounds) < 0;
    } else {
        fprintf(stderr, "unknown mode '%s'\n", argv[1]);
        rc = 2;
    }
    rocket_close(fd);
    return rc;
}
