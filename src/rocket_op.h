// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_op.h — the scaffolding every one-shot NPU op wrote out for itself.
 *
 * Eight entry points (fp16 pooling, both global pools, the DPU LUT, the element-wise
 * binaries, and the three conv one-jobs) run the same sequence around their own
 * generator: allocate a guard plus in/regcmd/out, check that no BO's last byte leaves
 * the low 4 GB, bracket a scatter into the input, bound the generated regcmd against a
 * stack array, stage it, prezero the output, submit ONE task, wait, de-scatter, free in
 * reverse. Roughly sixty lines, eight times, with the two-second wait spelled as a bare
 * 2000000000ULL in each of them.
 *
 * What is here is the part that is genuinely identical. What is NOT here is the
 * allocation lifetime: making these BOs resident (so the op library stops paying an
 * ioctl + mmap + munmap + GEM_CLOSE per call, which compounds badly through the composed
 * ops -- one rocket_softmax_fp16 is three NPU round trips and about thirteen such
 * cycles) needs a context to hang them on, and every one of these entry points is
 * `(int fd, ...)` with nowhere to put one. rocket_conv_ctx is the shape that fits; giving
 * the rest of the family the same thing is an API change, not a refactor, so it is left
 * as one. This header is where it would land.
 */
#ifndef ROCKET_OP_H
#define ROCKET_OP_H

#include <stdint.h>

#include "rocket_npu.h"

/* How long a one-shot op waits for its output fence. Two seconds is far above any
 * single-task op's runtime and far below a hang; it was written out as a literal at
 * every one of these sites. */
#define ROCKET_OP_WAIT_NS 2000000000ULL

/* Maximum input BOs one task can name here: feature, weight, a second operand, and the
 * regcmd. The element-wise binaries use all four. */
#define ROCKET_OP_MAX_IN 6

/* Stage `count` regcmd words into `rc`, submit them as ONE task reading `in[0..n_in)`
 * plus `rc`, and writing `out`; then wait for the output.
 *
 * The input list is a RESIDENCY set, not a program input: the kernel resolves it with
 * drm_gem_objects_lookup and the addresses the hardware actually reads come from the
 * regcmd, so the order is free and `rc` is appended last regardless of where a caller
 * used to place it.
 *
 * `who` names the caller in every diagnostic. Returns 0, or a negative rocket_status --
 * and on return the output BO is CPU-visible (prepped for read), ready to de-scatter;
 * the caller closes that bracket with rocket_bo_fini when it is done reading.
 *
 * Does NOT prezero the output: whether the padding lanes of a partial channel group must
 * read zero is the op's own contract, not this helper's, and a memset of a whole output
 * BO is not free. */
int rocket_op_submit_one(int fd, const char *who,
                         rocket_bo *rc, const uint64_t *regs, uint32_t count,
                         rocket_bo *const *in, int n_in, rocket_bo *out);

/* Does any of these BOs' last byte leave the low 4 GB the regcmd's 32-bit address fields
 * can encode? Logs and returns non-zero if so. NULL entries and unallocated BOs are
 * skipped, so a caller can pass its whole set. */
int rocket_op_iova_overflow(const char *who, rocket_bo *const *bos, int n);

#endif /* ROCKET_OP_H */
