// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
/*
 * rocket_hw_profile.c — the RK3588 hardware-profile instance, the chip-detection
 * front end, and the active-profile accessor. See rocket_hw_profile.h for the
 * machine-parameter vs inherent-datapath split.
 *
 * Detection order (rocket_hw_detect, run once):
 *   1. ROCKET_CHIP=<name>  — force a profile (bring-up / off-device testing).
 *   2. the rocket-bound platform device's DT `compatible`
 *      (/sys/bus/platform/drivers/rocket/<dev>/of_node/compatible), e.g.
 *      "rockchip,rk3588-rknn-core" or "rockchip,rk3576-rknn-core".
 *   3. the SoC-level /proc/device-tree/compatible ("rockchip,rk3576"), for a
 *      kernel whose accel device exposes no of_node.
 * An unrecognized or absent string falls back to the RK3588 profile with a warning.
 *
 * A chip we can NAME but have no validated profile for (today: the RK3568/RK3566) is
 * not silently treated as an RK3588. It warns that both halves are wrong for it —
 * the machine parameters are unmeasured AND the CNA/CORE/DPU geometry-register
 * encoding is IP-revision-specific — so a run that computes nothing, or computes
 * wrongly, is expected rather than mysterious. That was confirmed on RK3576 silicon
 * before its encoder existed: the RK3588 program submits and the job completes, but
 * the DPU writes no output.
 *
 * A profile whose datapath coverage is narrower than its machine parameters carries a
 * select_warning, logged whenever it is chosen. The RK3576 is that case today.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocket_hw_profile.h"
#include "rocket_log.h"

const struct rocket_hw_profile rocket_hw_rk3588 = {
    .name           = "rk3588",
    .cbuf_banks     = NPU_CBUF_BANKS,      /* 12 — mirrors npu_hw.h so the literal lives once */
    .cbuf_bank_size = NPU_CBUF_BANK_SIZE,  /* 32768 */
    .max_tile       = 256,                 /* Mt/Nt cap. 256 (not 384) lets Kt grow to 384 in
                                            * the CBUF (nKt 15->10), cutting readback: measured
                                            * 48.3->56.4 GFLOP/s on 512x3840x4096. Below 256,
                                            * DRAM reload (more N-tiles) outweighs the gain.
                                            * Override per-run with ROCKET_MM_MT/NT. */
    .kgroup_2b      = 32,                  /* int8/int16/fp16/bf16 weight K-group */
    .kgroup_4b      = 16,                  /* tf32 weight K-group (4-byte halves it) */
    .ngroup         = 16,                  /* weight N-group */

    /* All eight native encodings are HW-validated on the RK3588 (int4/int8/int16/
     * fp16/bf16/int32/fp32/tf32 — the datatype matrix is complete), so the mask is
     * all-ones over the precision_* range. A chip lacking one clears its bit. */
    .dtype_supported = ROCKET_DT_BIT(precision_int8)    | ROCKET_DT_BIT(precision_int16)
                     | ROCKET_DT_BIT(precision_float16) | ROCKET_DT_BIT(precision_bfloat16)
                     | ROCKET_DT_BIT(precision_int32)   | ROCKET_DT_BIT(precision_float32)
                     | ROCKET_DT_BIT(precision_int4)    | ROCKET_DT_BIT(precision_tf32),

    .default_workers = 8,
};

/* The RK3576. Measured on an H96 MAX M9 (mainline 7.1.3) with the part's own
 * geometry encoder (npu_regcmd_rk3576.c) running bit-exact int8 convolutions.
 *
 * The CBUF number is the one measurement that matters and it is sharp. A conv whose
 * feature plane does not fit computes wrong with the DPU still writing a full surface
 * and no IOMMU fault — one row over, about half the surface is still bit-exact, and
 * it degrades from there — so the capacity is found by growing the plane until a
 * known-good conv breaks. The budget is a single scalar in the CNA's own granule
 * unit: ceil(iw*ic/64) 64-byte granules per feature row, times the task's input
 * rows. Under the register program the vendor captures carry it is 4096, and the
 * boundary is exact — 4096 computes, 4096+1 does not — holding across ic 32/64/96,
 * k1/k3/k5, stride 1 and 2, and aspect ratios from 16x512 through 32x256 and 90x91
 * to 128x64, plus widths 16/32/64 breaking at the same granule total rather than the
 * same row count. 4096 granules is 256 KiB. [HW sweep, H96 MAX M9]
 *
 * 256 KiB is the DEFAULT data allocation, not the physical CBUF. The feature budget
 * is 4096 + F granules, where F is bits[16:27] of CNA_CBUF_CON0 (0x1040) — the word
 * the vendor captures carry as the constant 0x10000000 (F=0) or 0x14000000 (F=1024,
 * on its windowed programs). Data caps at 6144 granules (384 KiB) and is zero-sum
 * against the weight path, whose per-pass slice ceiling falls by what data gains;
 * the two sum to roughly 448 KiB. Raising F is a real tiling lever and needs the
 * weight slice sized to match, so the emitter keeps the vendor's F=0 and this
 * profile records what that buys. See rockchip-npu-notes/chips/rk3576-regcmd.md.
 * [HW sweep, H96 MAX M9] */
const struct rocket_hw_profile rocket_hw_rk3576 = {
    .name           = "rk3576",
    .cbuf_banks     = 8,                   /* default data allocation, not the pool:
                                            * 8 x 32768 = 262144 B = 4096 granules  */
    .cbuf_bank_size = NPU_CBUF_BANK_SIZE,  /* 32768, assumed equal to the RK3588's  */

    /* Matmul parameters, carried over from the RK3588 and NOT validated here: there
     * is no RK3576 matmul at all yet, only the int8 conv encoder, so nothing on this
     * part reads them. They are placeholders to be measured when a matmul path
     * lands, not findings. */
    .max_tile       = 256,
    .kgroup_2b      = 32,
    .kgroup_4b      = 16,
    .ngroup         = 16,

    /* int8 only. The vendor captures the encoder was transcribed from are int8, so
     * the fp16 and other precision fields have never been exercised on this part —
     * an unset bit here means unvalidated, which for a bring-up chip is the same
     * thing as unsupported. */
    .dtype_supported = ROCKET_DT_BIT(precision_int8),

    .select_warning =
        "rk3576: machine parameters are measured, but only the int8 DIRECT CONV_2D "
        "path computes on this part (gen_conv2d_int8_rk3576). Its depthwise sibling "
        "is transcribed but does NOT compute, and every other operation still emits "
        "the RK3588 encoding, which this part does not run — expect no output from "
        "matmul, fp16 and the op library.",

    .default_workers = 8,                  /* not swept on this part */
};

/* Profiles that exist. A chip gets an entry only once its machine parameters are
 * measured on that silicon. */
static const struct rocket_hw_profile *const profiles[] = {
    &rocket_hw_rk3588,
    &rocket_hw_rk3576,
};

/* Chips we can recognize from a DT compatible but have no profile for. Naming them
 * turns a silent wrong-hardware run into a specific warning. */
static const struct {
    const char *dt_token;   /* substring of the compatible string  */
    const char *chip;       /* short name for the message          */
} known_unprofiled[] = {
    { "rk3568", "rk3568" },
    { "rk3566", "rk3566" },
};

const struct rocket_hw_profile *rocket_hw_by_name(const char *name)
{
    if (!name) return NULL;
    for (size_t i = 0; i < sizeof(profiles) / sizeof(profiles[0]); i++)
        if (!strcmp(profiles[i]->name, name)) return profiles[i];
    return NULL;
}

/* Read a NUL-separated DT string list into `buf`, turning the NULs into spaces so a
 * plain strstr() covers every entry. Returns 0 on success. */
static int read_dt_strings(const char *path, char *buf, size_t len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(buf, 1, len - 1, f);
    fclose(f);
    if (n == 0) return -1;
    for (size_t i = 0; i < n; i++) if (buf[i] == '\0') buf[i] = ' ';
    buf[n] = '\0';
    return 0;
}

/* The compatible of whatever device the rocket driver is bound to. This is the
 * per-core NPU node (e.g. 27700000.npu), not the aggregate accel device — the accel
 * device rocket registers has no of_node of its own. */
static int read_bound_device_compatible(char *buf, size_t len)
{
    static const char *dir = "/sys/bus/platform/drivers/rocket";
    DIR *d = opendir(dir);
    if (!d) return -1;
    int rc = -1;
    for (struct dirent *e; (e = readdir(d)) != NULL; ) {
        if (e->d_name[0] == '.' || !strcmp(e->d_name, "bind") ||
            !strcmp(e->d_name, "unbind") || !strcmp(e->d_name, "uevent") ||
            !strcmp(e->d_name, "module")) continue;
        char path[512];
        snprintf(path, sizeof(path), "%s/%s/of_node/compatible", dir, e->d_name);
        if (read_dt_strings(path, buf, len) == 0) { rc = 0; break; }
    }
    closedir(d);
    return rc;
}

static const struct rocket_hw_profile *rocket_hw_detect(void)
{
    const char *forced = getenv("ROCKET_CHIP");
    if (forced && *forced) {
        const struct rocket_hw_profile *p = rocket_hw_by_name(forced);
        if (p) {
            ROCKET_LOGD("hw profile: %s (forced by ROCKET_CHIP)\n", p->name);
            if (p->select_warning) ROCKET_LOGW("%s\n", p->select_warning);
            return p;
        }
        ROCKET_LOGW("ROCKET_CHIP=\"%s\" names no profile; falling back to %s\n",
                    forced, rocket_hw_rk3588.name);
        return &rocket_hw_rk3588;
    }

    char compat[512];
    const char *src = "bound rocket device";
    if (read_bound_device_compatible(compat, sizeof(compat)) != 0) {
        src = "/proc/device-tree/compatible";
        if (read_dt_strings("/proc/device-tree/compatible", compat, sizeof(compat)) != 0) {
            /* No device tree at all (off-device build/test). Not a warning: the
             * fallback is the only profile there is. */
            ROCKET_LOGD("hw profile: %s (no device tree to read)\n", rocket_hw_rk3588.name);
            return &rocket_hw_rk3588;
        }
    }

    for (size_t i = 0; i < sizeof(profiles) / sizeof(profiles[0]); i++) {
        if (strstr(compat, profiles[i]->name)) {
            ROCKET_LOGD("hw profile: %s (from %s)\n", profiles[i]->name, src);
            if (profiles[i]->select_warning)
                ROCKET_LOGW("%s\n", profiles[i]->select_warning);
            return profiles[i];
        }
    }
    for (size_t i = 0; i < sizeof(known_unprofiled) / sizeof(known_unprofiled[0]); i++) {
        if (strstr(compat, known_unprofiled[i].dt_token)) {
            ROCKET_LOGW("%s NPU detected (%s), but librocketnpu has no validated profile "
                        "for it: running with %s machine parameters AND the %s "
                        "geometry-register encoding, both of which are wrong for this chip. "
                        "Expect no output or wrong results.\n",
                        known_unprofiled[i].chip, src,
                        rocket_hw_rk3588.name, rocket_hw_rk3588.name);
            return &rocket_hw_rk3588;
        }
    }
    ROCKET_LOGW("unrecognized NPU compatible \"%s\" (%s); falling back to %s\n",
                compat, src, rocket_hw_rk3588.name);
    return &rocket_hw_rk3588;
}

const struct rocket_hw_profile *rocket_hw_current(void)
{
    /* Detect once. Benign race: two threads may both detect, and they compute the
     * same pointer, so the store is idempotent. */
    static const struct rocket_hw_profile *cached;
    const struct rocket_hw_profile *p = cached;
    if (!p) { p = rocket_hw_detect(); cached = p; }
    return p;
}
