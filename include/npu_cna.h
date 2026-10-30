// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef NPU_CNA_H
#define NPU_CNA_H

/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * prose-lint: off
 *
 * This file is part of rocket-userspace.
 *
 * rocket-userspace is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rocket-userspace is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rocket-userspace.  If not, see <https://www.gnu.org/licenses/>.
 *
 * prose-lint: on
 */

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif
typedef struct npu_cna_desc {
  uint8_t enable;
  uint8_t conv_mode;          // 0x100C
  uint8_t in_precision;       // 0x100C
  uint8_t proc_precision;     // 0x100C
  uint8_t kernel_groups;      // 0x1010
  uint16_t feature_grains;    // 0x1010
  uint8_t conv_y_stride;      // 0x1014
  uint8_t conv_x_stride;      // 0x1014
  uint16_t datain_width;      // 0x1020
  uint16_t datain_height;     // 0x1020
  uint16_t datain_channel;    // 0x1024
  uint16_t dataout_width;     // 0x1028
  uint32_t dataout_atomics;   // 0x102C
  uint32_t weight_bytes;      // 0x1030
  uint32_t weight_bytes_per_kernel; // 0x1034
  uint8_t weight_width;       // 0x1038
  uint8_t weight_height;      // 0x1038
  uint16_t weight_kernels;    // 0x1038
  uint8_t weight_bank;        // 0x1040
  uint8_t data_bank;          // 0x1040
  uint8_t weight_reuse;       // 0x1040 bit13 (CBUF reuse, 0 = re-fetch)
  uint8_t data_reuse;         // 0x1040 bit12
  uint16_t data_entries;      // 0x1044
  uint8_t data_sign;          // 0x104c
  uint8_t cvt_type;           // 0x104c
  uint8_t cvt_bypass;         // 0x104c
  uint16_t cvt_scale0;        // 0x1050
  uint16_t cvt_scale1;        // 0x1054
  uint16_t cvt_scale2;        // 0x1058
  uint16_t cvt_scale3;        // 0x105C
  uint8_t fc_skip_en;         // 0x1060
  uint16_t data_offset;       // 0x1064
  uint8_t pad_left;           // 0x1068
  uint8_t pad_top;            // 0x1068
  uint32_t feature_base_addr; // 0x1070
  uint16_t weight_offset;     // 0x1074
  uint8_t weight_burst_len;   // 0x1078
  uint8_t data_burst_len;     // 0x1078
  uint32_t line_stride;       // 0x107C
  int32_t surf_stride;        // 0x1080
  uint16_t dma_width;         // 0x1084
  uint16_t dma_height;        // 0x1084
  uint16_t dma_channel;       // 0x1088
  uint32_t decompress_addr0;  // 0x1110

  uint16_t dataout_height;

  /* General CONV_2D fields. Only `gen_conv2d_task` (`npu_regcmd.c`) reads them. The matmul
   * generators leave them unset, and the shared matmul emitter never reads them. A 1x1-conv
   * matmul is therefore byte-identical whatever their values. `atrous_*` are the `CONV_CON3`
   * dilation fields [21:25]/[16:20]. Each contains the raw register value,
   * `dilation_rate - 1`, and 0 means no dilation. */
  uint8_t atrous_x_dilation;  // 0x1014 [20:16]  (rate-1)
  uint8_t atrous_y_dilation;  // 0x1014 [25:21]  (rate-1)

  /* Selects the deconvolution mode. `deconv` sets `CONV_CON1` bit 16, and the two strides
   * are the `DECONV` stride fields of `CONV_CON3`, as the raw field value (`stride - 1`). The
   * stride must be a power of two. The CNA then interior-dilates its input by the stride.
   * 0, the default, is off. Only `gen_conv2d_task` reads these fields. */
  uint8_t deconv;             // 0x100C bit16
  uint8_t deconv_x_stride;    // 0x1014 [10:8]   (stride-1)
  uint8_t deconv_y_stride;    // 0x1014 [13:11]  (stride-1)

  /* Sets `CNA_PAD_CON1` (0x10A4), the constant the CNA pads borders with. For the int8-out
   * (uint8-centered) datapath, it is the input zero-point in the shifted domain
   * (`input_zp_uint8 - 0x80`). It is 0 for the float and int32-raw paths. Only the conv
   * emitter reads it, and the matmul emitter always pads with 0. */
  uint32_t pad_con1;          // 0x10A4

  /* Sets `CONV_CON1` bit 29 (`GROUP_LINE_OFF`). If it is set with `line_stride` at one row of
   * `A` and `surf_stride` 0, the CNA reads a matmul's `A` as plain [M][K] rows.
   * `gen_matmul_fp16` sets this row-major form from `matmul_params_t.a_row_atoms`. 0 is the
   * cube program's value. The shared matmul emitter reads it, so every matmul generator must
   * zero-initialize its descriptor. */
  uint8_t group_line_off;     // 0x100C bit29
} npu_cna_desc;

typedef struct npu_core_desc {
  uint8_t proc_precision;   // 0x3010
  uint8_t qd_en;            // 0x3010
  uint16_t dataout_height;  // 0x3014
  uint16_t dataout_width;   // 0x3014
  uint16_t dataout_channel; // 0x3018
  /* Enables depthwise convolution (`CORE_MISC_CFG` `DW_EN`, bit1). Only `gen_conv2d_task`
   * reads it. The matmul emitter never reads it, so a matmul is byte-identical whatever its
   * value. The default is 0. */
  uint8_t dw_en;            // 0x3010 bit1
} npu_core_desc;

typedef struct nup_pc_desc {
 uint32_t pc_source_addr;  // 0x0010
 uint32_t pc_data_amount;  // 0x0014
} npu_pc_desc;

typedef struct npu_cna_core_task {

  // A task that uses the CNA and CORE, bypasses the DPU operations and writes its output
  // to memory. It fits within 112 operations.
  uint64_t ops[112];

} npu_cna_core_task;


#ifdef __cplusplus
}
#endif
#endif //NPU_CNA_H
