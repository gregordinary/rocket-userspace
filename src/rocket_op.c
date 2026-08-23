// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/* rocket_op.c — see rocket_op.h. */
#include <string.h>

#include "rocket_op.h"
#include "rocket_log.h"

int rocket_op_iova_overflow(const char *who, rocket_bo *const *bos, int n)
{
    for (int i = 0; i < n; i++) {
        const rocket_bo *b = bos[i];
        if (!b || !b->handle) continue;
        if (((b->dma_address + b->size) >> 32) != 0) {
            ROCKET_LOGE("%s: a BO's last byte (0x%llx) leaves the low 4 GB the regcmd's "
                        "32-bit address fields can encode\n", who,
                        (unsigned long long)(b->dma_address + b->size));
            return 1;
        }
    }
    return 0;
}

int rocket_op_submit_one(int fd, const char *who,
                         rocket_bo *rc, const uint64_t *regs, uint32_t count,
                         rocket_bo *const *in, int n_in, rocket_bo *out)
{
    int ret;

    if (count == 0 || (size_t)count * sizeof(uint64_t) > rc->size) {
        ROCKET_LOGE("%s: regcmd of %u words does not fit its %zu-byte BO\n",
                    who, count, rc->size);
        return ROCKET_E_TILING;
    }
    if (n_in < 1 || n_in > ROCKET_OP_MAX_IN - 1) {
        ROCKET_LOGE("%s: %d input BOs, expected 1..%d\n", who, n_in, ROCKET_OP_MAX_IN - 1);
        return ROCKET_E_SHAPE;
    }

    if ((ret = rocket_bo_prep(fd, rc, 1, 0)) != 0) return ret;
    memcpy(rc->ptr, regs, (size_t)count * sizeof(uint64_t));
    if ((ret = rocket_bo_fini(fd, rc)) != 0) return ret;

    rocket_task_desc task = { .regcmd = (uint32_t)rc->dma_address, .regcmd_count = count };
    uint32_t in_h[ROCKET_OP_MAX_IN];
    uint32_t nh = 0;
    for (int i = 0; i < n_in; i++) in_h[nh++] = in[i]->handle;
    in_h[nh++] = rc->handle;                    /* the regcmd is always the last input */
    uint32_t out_h[] = { out->handle };

    if ((ret = rocket_submit_tasks(fd, &task, 1, in_h, nh, out_h, 1)) != 0) {
        ROCKET_LOGE("%s: submit failed (%d)\n", who, ret);
        return ret;
    }
    if ((ret = rocket_bo_prep(fd, out, 0, ROCKET_OP_WAIT_NS)) != 0) {
        ROCKET_LOGE("%s: the output fence did not signal within %llu ms (%d)\n",
                    who, (unsigned long long)(ROCKET_OP_WAIT_NS / 1000000ULL), ret);
        return ret;
    }
    return 0;
}
