// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The rocket-userspace authors
#ifndef RK3576_CONV_SHAPES_H
#define RK3576_CONV_SHAPES_H
/*
 * rk3576_conv_shapes.h — the RK3576 convolution correctness envelope, as one table.
 *
 * Two gates read it and they ask different questions of the same shapes.
 * rk3576_conv_gate.c drives the register EMITTER directly, one program per shape, and
 * `refuse` is what that program must decline. rk3576_conv_lib_gate.c drives the LIBRARY
 * entries, which tile, and `lib_refuse` is what those must decline.
 *
 * The two columns differ, and the difference is the point. The resident weight slice is
 * 32*ic*kh*kw bytes whatever oc is, but the slice the part tolerates is a function of
 * how many output-channel GROUPS the conv drives — 144 KiB at four or more, 148 at
 * three, 156 at two, and the CBUF pool alone at one. A single program past its cap loses
 * its trailing groups and has to be refused; the same convolution split into fewer
 * groups per submit computes. So every shape the emitter declines for its weight slice
 * is expected to COMPUTE through the library, except where the slice passes the pool
 * check at a single group too.
 */

typedef struct {
    const char *group;
    const char *name;
    unsigned ic, oc, iw, ih, k, stride;
    int      same;      /* 1 = SAME padding, 0 = VALID */
    int      dw;
    unsigned max_rows;  /* 0 = natural plan */
    int      refuse;    /* 1 = the single-program EMITTER must refuse it (rk3576_conv_gate) */
    int      lib_refuse;/* 1 = the tiling LIBRARY entry must refuse it too (rk3576_conv_lib_gate) */
} shape_t;

static const shape_t SHAPES[] = {
/* ---- envelope: the correctness envelope, single task, gap-free ------------ */
{"envelope", "base-k1",          32,  32,   8,   8, 1, 1, 0, 0, 0, 0, 0},
{"envelope", "base-k3-same",     32,  32,  32,  32, 3, 1, 1, 0, 0, 0, 0},
{"envelope", "base-k3-valid",    32,  32,  32,  32, 3, 1, 0, 0, 0, 0, 0},
{"envelope", "k5-same",          32,  32,  32,  32, 5, 1, 1, 0, 0, 0, 0},
{"envelope", "k5-valid-s2",      32,  32,  40,  40, 5, 2, 0, 0, 0, 0, 0},
{"envelope", "k3-s2-same",       32,  32,  32,  32, 3, 2, 1, 0, 0, 0, 0},
{"envelope", "ic64",             64,  32,  32,  32, 3, 1, 1, 0, 0, 0, 0},
{"envelope", "ic96",             96,  32,  16,  16, 3, 1, 1, 0, 0, 0, 0},
{"envelope", "ic128",           128,  32,  16,  16, 3, 1, 1, 0, 0, 0, 0},
{"envelope", "oc64",             32,  64,  32,  32, 3, 1, 1, 0, 0, 0, 0},
{"envelope", "oc128",            32, 128,  16,  16, 3, 1, 1, 0, 0, 0, 0},
{"envelope", "oc256",            32, 256,  16,  16, 1, 1, 0, 0, 0, 0, 0},
{"envelope", "ic64-oc64",        64,  64,  32,  32, 3, 1, 1, 0, 0, 0, 0},
{"envelope", "wide-plane",       32,  32, 128,  32, 3, 1, 1, 0, 0, 0, 0},
{"envelope", "tall-plane",       32,  32,  16, 128, 3, 1, 1, 0, 0, 0, 0},
{"envelope", "narrow-deep",      64,  64,   8,   8, 1, 1, 0, 0, 0, 0, 0},
{"envelope", "k1-s2",            32,  32,  32,  32, 1, 2, 0, 0, 0, 0, 0},
{"envelope", "unpadded-ic32",    32,  32,  20,  20, 3, 1, 0, 0, 0, 0, 0},

/* ---- window: the row split, including the two planes with no single-task plan */
{"window",   "split-forced-2",   32,  32,  32,  32, 3, 1, 1, 0, 16, 0, 0},
{"window",   "split-forced-4",   32,  32,  32,  32, 3, 1, 1, 0,  8, 0, 0},
{"window",   "split-forced-8",   32,  32,  32,  32, 3, 1, 1, 0,  4, 0, 0},
{"window",   "split-k5",         32,  32,  40,  40, 5, 1, 1, 0, 10, 0, 0},
{"window",   "split-s2",         32,  32,  40,  40, 3, 2, 1, 0, 10, 0, 0},
{"window",   "split-valid",      32,  32,  32,  32, 3, 1, 0, 0,  8, 0, 0},
{"window",   "split-112",        32,  32, 112, 112, 3, 1, 1, 0,  0, 0, 0},
{"window",   "split-224",        32,  32, 224, 224, 3, 1, 1, 0,  0, 0, 0},

/* ---- surface: 0x40B8, the channel-group jump, across shape ----------------
 * The form ow*(2*oh_full - oh_task) was pinned at oc=64, k=3, one plane. It is
 * only exercised at all when a WINDOWED task drives more than one output-channel
 * group, so every entry here is windowed with oc >= 32 and varies exactly the
 * axes the pinning did not: kernel size, stride, padding, plane aspect, and the
 * number of channel groups (oc/16). */
{"surface",  "grp2-k1",          32,  32,  16,  32, 1, 1, 0, 0,  8, 0, 0},
{"surface",  "grp4-k1",          32,  64,  16,  32, 1, 1, 0, 0,  8, 0, 0},
{"surface",  "grp8-k1",          32, 128,  16,  32, 1, 1, 0, 0,  8, 0, 0},
{"surface",  "grp16-k1",         32, 256,  16,  32, 1, 1, 0, 0,  8, 0, 0},
{"surface",  "grp8-k3-same",     32, 128,  16,  32, 3, 1, 1, 0,  8, 0, 0},
{"surface",  "grp8-k3-valid",    32, 128,  16,  32, 3, 1, 0, 0,  8, 0, 0},
{"surface",  "grp8-k5-same",     32, 128,  16,  32, 5, 1, 1, 0, 12, 0, 0},
{"surface",  "grp8-s2-same",     32, 128,  16,  32, 3, 2, 1, 0, 12, 0, 0},
{"surface",  "grp8-s2-valid",    32, 128,  16,  32, 3, 2, 0, 0, 12, 0, 0},
{"surface",  "grp4-wide",        32,  64,  64,  32, 3, 1, 1, 0,  8, 0, 0},
{"surface",  "grp4-tall",        32,  64,  16,  96, 3, 1, 1, 0,  8, 0, 0},
{"surface",  "grp4-ragged",      32,  64,  16,  30, 3, 1, 1, 0,  7, 0, 0},
{"surface",  "grp4-ic64",        64,  64,  16,  32, 3, 1, 1, 0,  8, 0, 0},
{"surface",  "grp4-ic128",      128,  64,  16,  32, 3, 1, 1, 0,  8, 0, 0},

/* ---- weight: the weight-path limit that is not slice size or group count --
 * Some large-ic, wide-kernel shapes compute only their first TWO output-channel
 * groups, at every CBUF allowance, so it is independent of the data budget. The
 * reported signature is ic=192 k=5: bit-exact through oc=63 and wrong past it,
 * whatever oc is programmed. These entries bracket it — the two known-exact points
 * either side (ic=448 k=3 oc=128 with a bigger cube; ic=64 oc=256 with eight
 * groups) against the reported failures — so the gate says where the boundary is
 * rather than only that one exists. Planes are kept small: the limit is a weight-
 * path property, so the feature side only has to stay out of the way. */
{"weight",   "w-ic192-k5-oc32",  192,  32,   8,   8, 5, 1, 1, 0, 0, 0, 0},
{"weight",   "w-ic192-k5-oc64",  192,  64,   8,   8, 5, 1, 1, 0, 0, 0, 0},
{"weight",   "w-ic192-k5-oc96",  192,  96,   8,   8, 5, 1, 1, 0, 0, 1, 0},
{"weight",   "w-ic192-k5-oc128", 192, 128,   8,   8, 5, 1, 1, 0, 0, 1, 0},
/* The one weight-slice refusal that survives the library's output-channel split: at
 * 200 KiB the slice is past the CBUF pool at a SINGLE group, so there is no group count
 * left to fall back to and the recourse is an input-channel split the on-chip requant
 * forecloses. Every other refusal below is a group-count bound the split lifts. */
{"weight",   "w-ic256-k5-oc32",  256,  32,   8,   8, 5, 1, 1, 0, 0, 1, 1},
{"weight",   "w-ic128-k5-oc128", 128, 128,   8,   8, 5, 1, 1, 0, 0, 0, 0},
{"weight",   "w-ic192-k3-oc128", 192, 128,   8,   8, 3, 1, 1, 0, 0, 0, 0},
{"weight",   "w-ic448-k3-oc128", 448, 128,   8,   8, 3, 1, 1, 0, 0, 0, 0},
{"weight",   "w-ic64-oc256-k3",   64, 256,   8,   8, 3, 1, 1, 0, 0, 0, 0},
{"weight",   "w-ic96-k5-oc128",   96, 128,   8,   8, 5, 1, 1, 0, 0, 0, 0},
/* The bracket. The resident weight slice is 32*ic*kh*kw bytes — one output-channel
 * group — so k=3 steps it in 9 KiB per 32 channels of ic where k=5 steps it in 25,
 * which is what makes a fine bracket possible at all. Every entry drives four
 * output-channel groups, since two are known to compute at any slice tried. */
{"weight",   "w-slice100-k5",    128, 128,   8,   8, 5, 1, 1, 0, 0, 0, 0},   /* 100 KiB */
{"weight",   "w-slice125-k5",    160, 128,   8,   8, 5, 1, 1, 0, 0, 0, 0},   /* 125 KiB */
{"weight",   "w-slice126-k3",    448, 128,   8,   8, 3, 1, 1, 0, 0, 0, 0},   /* 126 KiB */
{"weight",   "w-slice135-k3",    480, 128,   8,   8, 3, 1, 1, 0, 0, 0, 0},   /* 135 KiB */
{"weight",   "w-slice144-k3",    512, 128,   8,   8, 3, 1, 1, 0, 0, 0, 0},   /* 144 KiB */
{"weight",   "w-slice153-k3",    544, 128,   8,   8, 3, 1, 1, 0, 0, 1, 0},   /* 153 KiB */
{"weight",   "w-slice162-k3",    576, 128,   8,   8, 3, 1, 1, 0, 0, 1, 0},   /* 162 KiB */
{"weight",   "w-slice150-k5",    192, 128,   8,   8, 5, 1, 1, 0, 0, 1, 0},   /* 150 KiB */
/* The same bracket at k=1, on a 4x2 plane. The slice is 32*ic*kh*kw, so k=1 steps it
 * in 32 bytes per channel-group-multiple of ic where k=3 steps it in 288 — the only
 * way to resolve the boundary to better than a 9 KiB stride, and the control for
 * whether the governing quantity is the slice or the kernel. */
{"weight",   "w-e4608-k1",      4608, 128,   4,   2, 1, 1, 0, 0, 0, 0, 0},
{"weight",   "w-e4736-k1",      4736, 128,   4,   2, 1, 1, 0, 0, 0, 1, 0},
{"weight",   "w-e4800-k1",      4800, 128,   4,   2, 1, 1, 0, 0, 0, 1, 0},
{"weight",   "w-e5184-k1",      5184, 128,   4,   2, 1, 1, 0, 0, 0, 1, 0},
{"weight",   "w-e4640-k1",      4640, 128,   4,   2, 1, 1, 0, 0, 0, 1, 0},
{"weight",   "w-e4672-k1",      4672, 128,   4,   2, 1, 1, 0, 0, 0, 1, 0},
{"weight",   "w-e4704-k1",      4704, 128,   4,   2, 1, 1, 0, 0, 0, 1, 0},
{"weight",   "w-e4864-k1",      4864, 128,   4,   2, 1, 1, 0, 0, 0, 1, 0},
{"weight",   "w-e4992-k1",      4992, 128,   4,   2, 1, 1, 0, 0, 0, 1, 0},

/* ---- dw: depthwise, never run on this part -------------------------------
 * Un-windowed first, so a depthwise result is not read through the row split; the
 * windowed entries then separate "depthwise" from "windowed" for 0x40B8, which the
 * single vendor capture confounds. The ic sweep separates the weight-bytes term:
 * k*k*ic/8 and k*k*4 agree at ic=32 and diverge everywhere else. */
{"dw",       "dw-c32-k3",        32,  32,  16,  16, 3, 1, 1, 1, 0, 0, 0},
{"dw",       "dw-c32-k3-valid",  32,  32,  16,  16, 3, 1, 0, 1, 0, 0, 0},
{"dw",       "dw-c32-k1",        32,  32,  16,  16, 1, 1, 0, 1, 0, 0, 0},
{"dw",       "dw-c32-k5",        32,  32,  16,  16, 5, 1, 1, 1, 0, 0, 0},
{"dw",       "dw-c32-s2",        32,  32,  16,  16, 3, 2, 1, 1, 0, 0, 0},
{"dw",       "dw-c64-k3",        64,  64,  16,  16, 3, 1, 1, 1, 0, 0, 0},
{"dw",       "dw-c96-k3",        96,  96,  16,  16, 3, 1, 1, 1, 0, 0, 0},
{"dw",       "dw-c128-k3",      128, 128,  16,  16, 3, 1, 1, 1, 0, 0, 0},
{"dw",       "dw-c64-k5",        64,  64,  16,  16, 5, 1, 1, 1, 0, 0, 0},
{"dw",       "dw-c64-k1",        64,  64,  16,  16, 1, 1, 0, 1, 0, 0, 0},
{"dw",       "dw-c32-win2",      32,  32,  16,  32, 3, 1, 1, 1, 16, 0, 0},
{"dw",       "dw-c32-win4",      32,  32,  16,  32, 3, 1, 1, 1,  8, 0, 0},
{"dw",       "dw-c64-win4",      64,  64,  16,  32, 3, 1, 1, 1,  8, 0, 0},
{"dw",       "dw-c128-win4",    128, 128,  16,  32, 3, 1, 1, 1,  8, 0, 0},
{"dw",       "dw-c32-112",       32,  32, 112, 112, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c32-k7",        32,  32,  16,  16, 7, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c128-s2",      128, 128,  16,  16, 3, 2, 1, 1,  0, 0, 0},
{"dw",       "dw-c128-k3-valid",128, 128,  16,  16, 3, 1, 0, 1,  0, 0, 0},
{"dw",       "dw-c256-k3",      256, 256,  16,  16, 3, 1, 1, 1,  0, 0, 0},
/* The 0x401C round-up to four elements is DEPTHWISE-ONLY and invisible at every
 * plane whose ow*oh_full is already a multiple of four — which is every row above and
 * every direct shape in this table. These three are the vendor's own separating
 * planes: 15x18, 17x19 and 19x19 give 270, 323 and 361 output elements, one of each
 * non-zero residue mod 4. */
{"dw",       "dw-c32-15x18",     32,  32,  15,  18, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c32-17x19",     32,  32,  17,  19, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c32-19x19",     32,  32,  19,  19, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c64-19x19-win", 64,  64,  19,  19, 3, 1, 1, 1,  8, 0, 0},
/* Channel counts that are NOT multiples of 32, which is where the depthwise path's
 * two granules stop agreeing: the weight cube rounds C up to 16, the CBUF allocation
 * takes a 16-group count of 3 mod 4 one group further (48 -> 64, 112 -> 128, while 80
 * and 144 stay put), and 0x4050's 2-bit group field is that count minus one modulo 4,
 * so it wraps at 80 and 144. A partial weight group strides by round16 of what it
 * holds, which 24 and 72 separate from the raw count. */
{"dw",       "dw-c8-k3",          8,   8,  16,  16, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c16-k3",        16,  16,  16,  16, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c24-k3",        24,  24,  16,  16, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c48-k3",        48,  48,  16,  16, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c72-k3",        72,  72,  16,  16, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c80-k3",        80,  80,  16,  16, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c112-k3",      112, 112,  16,  16, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c144-k3",      144, 144,  16,  16, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c176-k3",      176, 176,  16,  16, 3, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c48-k5",        48,  48,  16,  16, 5, 1, 1, 1,  0, 0, 0},
{"dw",       "dw-c112-s2",      112, 112,  16,  16, 3, 2, 1, 1,  0, 0, 0},
{"dw",       "dw-c48-win4",      48,  48,  16,  32, 3, 1, 1, 1,  8, 0, 0},
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof SHAPES[0]))

#endif /* RK3576_CONV_SHAPES_H */
