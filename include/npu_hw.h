// SPDX-License-Identifier: MIT
// Copyright (C) 2026 The rocket-userspace authors
#ifndef NPU_HW_H
#define NPU_HW_H

/*
 * Declares the regcmd word format, the precision codes and the RK3588 CBUF geometry.
 *
 * A register program is an array of 64-bit regcmd words, and each word writes one register.
 * Bits 63:48 contain the target, bits 47:16 the value and bits 15:0 the register offset.
 * `REGCMD()` builds one word. Every task ends with a four-word PC trailer, and its last word
 * starts the task.
 *
 * The register offsets and fields are not declared here. The library's encoders program them
 * from Mesa's `rocket` register database, which the library compiles in privately.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * The regcmd word
 * ==========================================================================*/

/* The target half of a word: one select bit per register block, and the write bit. The select
 * bits are the values of Mesa's `target` enum in its `rocket` register database. A write to a
 * block's register takes that block's bit and the write bit, as Mesa's emitter adds 0x1. */
#define NPU_TARGET_PC       0x0100
#define NPU_TARGET_CNA      0x0200
#define NPU_TARGET_CORE     0x0800
#define NPU_TARGET_DPU      0x1000
#define NPU_TARGET_DPU_RDMA 0x2000
#define NPU_TARGET_PPU      0x4000
#define NPU_TARGET_PPU_RDMA 0x8000
#define NPU_REG_WRITE       0x0001

#define TARGET_PC       (NPU_TARGET_PC   | NPU_REG_WRITE)
#define TARGET_CNA      (NPU_TARGET_CNA  | NPU_REG_WRITE)
#define TARGET_CORE     (NPU_TARGET_CORE | NPU_REG_WRITE)
#define TARGET_DPU      (NPU_TARGET_DPU  | NPU_REG_WRITE)

/* The PC trailer's control words. The trailer is four words:
 *
 *     [TARGET_NONE] [PC_REGISTER_AMOUNTS write] [TARGET_PRE_ENABLE] [TARGET_ENABLE write]
 *
 * `TARGET_NONE` is an empty word, which a chained submit replaces with a `PC_BASE_ADDRESS`
 * write to the next task. `TARGET_PRE_ENABLE` is the target of an otherwise empty word that
 * must precede the enable [TRM]. `TARGET_ENABLE` is the target of the `PC_OPERATION_ENABLE`
 * write that starts the
 * task.
 *
 * The enable write's value lists the blocks the task runs, one bit each, in the layout of the
 * global enable register. The bits are CNA 0, CORE 2, DPU 3, DPU_RDMA 4, PPU 5 and
 * PPU_RDMA 6. A convolution or matmul writes 0x1D, a DPU-only program 0x18 and a pool 0x60. */
#define TARGET_NONE       0x0000
#define TARGET_PRE_ENABLE 0x0041
#define TARGET_ENABLE     0x0081

/* Builds one regcmd word from a target, a register offset and a 32-bit value, the order
 * Mesa's emit_raw() takes them in. Each argument is masked to its field as a whole, so a
 * compound value such as `(2u << 16) | (1u << 6)` is masked after the `|`. */
#define REGCMD(target, reg, value)                                                        \
    ((((uint64_t)(target) & 0xFFFFu) << 48) | (((uint64_t)(value) & 0xFFFFFFFFu) << 16) |   \
     ((uint64_t)(reg) & 0xFFFFu))

/* ============================================================================
 * Precision codes and the convolution mode
 * ==========================================================================*/

/* The code the CNA, CORE, DPU and DPU_RDMA precision fields take, 3 bits each. int8, fp16,
 * int32 and fp32 are the TRM's codes, and int16 1 is NVDLA's. int4 6, bf16 3 and tf32 7 are
 * measured [HW sweep]:
 *
 * - 6 is the only code that gives a bit-exact int4 product. 3 saturates and 7 writes nothing.
 * - bf16 runs at fp16's MAC rate.
 * - The CNA and the CORE take tf32's 7, and the DPU, whose precision field has no 7, does not.
 *
 * Each of the six input codes has a matmul generator (`npu_matmul.h`). int32 and fp32 are
 * output and DPU codes only. */
enum npu_precision {
    PREC_INT8  = 0,
    PREC_INT16 = 1,
    PREC_FP16  = 2,
    PREC_BF16  = 3,
    PREC_INT32 = 4,
    PREC_FP32  = 5,
    PREC_INT4  = 6,
    PREC_TF32  = 7,
};

/* The CNA's convolution mode. */
enum npu_conv_mode {
    CONV_MODE_DIRECT    = 0,
    CONV_MODE_DEPTHWISE = 3,
};

/* ============================================================================
 * The RK3588 CBUF
 * ==========================================================================*/

/* The RK3588's convolution buffer: 12 banks of 32 KiB. `rocket_hw_rk3588` takes both values
 * from here. Code that can run on another part reads the active profile, through
 * `rocket_hw_current()`, instead. */
#define RK3588_CBUF_BANK_BYTES 32768
#define RK3588_CBUF_BANKS      12

#ifdef __cplusplus
}
#endif
#endif /* NPU_HW_H */
