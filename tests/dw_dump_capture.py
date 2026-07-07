#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Companion to tests/conv_dw_int8_runtime.c: from the Teflon DW capture + its dw_pt.tflite
# model, (1) verify our int8 DW cube + bias-fold formulas reproduce Mesa's captured BOs
# byte-for-byte (independent numpy), and (2) dump the raw tensors the runtime gate needs:
#   /tmp/dw_raw_in.bin  raw int8 input  [C][IH][IW]  (descattered from mesa-input)
#   /tmp/dw_w.bin       raw int8 filter [C][KH][KW]  (from the tflite model, INDEPENDENT)
#   /tmp/dw_bias.bin    int32 bias      [C]          (from the tflite model)
# The filter+bias come from the model (not descattered), so conv_dw_int8_runtime's PASS
# independently confirms the runtime's host packing reproduces Teflon ground truth.
#
# Domain constants (empirically pinned against the capture):
#   input/weight cube value = (byte - 0x80)         (Mesa's uint8-centered domain)
#   bias[oc] = bias_q[oc] - Σ_kernel(w_u8 - w_zp)*(in_zp - 0x80)
#   output (model)          = npu_byte + 0x80
#
# Usage: dw_dump_capture.py [capture_dir]   (default: tests/data/teflon-dw-capture by this script)
import sys, os, numpy as np
try:
    import tensorflow as tf; Interp = tf.lite.Interpreter
except Exception:
    from tflite_runtime.interpreter import Interpreter as Interp

D = sys.argv[1] if len(sys.argv) > 1 \
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "teflon-dw-capture")
it = Interp(model_path=D + "/dw_pt.tflite"); it.allocate_tensors()
flt = it.get_tensor(2)[0].astype(np.int32)        # [KH,KW,C] int8 filter
bias = it.get_tensor(1).astype(np.int64)          # [C] int32
in_zp, w_zp = -2, 0
C, KH, KW = flt.shape[2], flt.shape[0], flt.shape[1]; G = 64; IH = IW = 8
mesa_in = np.fromfile(D + "/mesa-input-000-000.bin", dtype=np.uint8)
mesa_wt = np.fromfile(D + "/mesa-weights-000-000.bin", dtype=np.uint8)
mesa_bs = np.fromfile(D + "/mesa-biases-000-000.bin", dtype=np.int32).astype(np.int64)

def fdat(Cc, H, W, C2, c, h, w):
    plane = (c - 1) // C2; return plane * H * W * C2 + C2 * ((h - 1) * W + (w - 1)) + (c - 1) % C2
def w_dw(kh_, kw_, Gg, c, kh, kw):
    ic1 = (c - 1) // Gg; ic2 = (c - 1) % Gg; return ((ic1 * kh_ + (kh - 1)) * kw_ + (kw - 1)) * Gg + ic2

# (1a) weight cube
wt = np.zeros(((C + G - 1) // G) * G * KH * KW, dtype=np.uint8)
for c in range(C):
    for kh in range(KH):
        for kw in range(KW):
            wt[w_dw(KH, KW, G, c + 1, kh + 1, kw + 1)] = ((int(flt[kh, kw, c]) & 0xff) - 0x80) & 0xff
nwt = min(len(wt), len(mesa_wt))
print("WEIGHT cube match:", np.array_equal(wt[:nwt], mesa_wt[:nwt]))

# (1b) bias fold
corr = np.array([sum((int(flt[kh, kw, c]) & 0xff) - w_zp for kh in range(KH) for kw in range(KW))
                 * (in_zp - 0x80) for c in range(C)], dtype=np.int64)
print("BIAS cube match:", np.array_equal((bias - corr).astype(np.int32), mesa_bs[:C].astype(np.int32)))

# (2) dump raw tensors for the C gate
raw = np.zeros((C, IH, IW), dtype=np.int64)
for c in range(C):
    for h in range(IH):
        for w in range(IW):
            raw[c, h, w] = (int(mesa_in[fdat(C, IH, IW, 16, c + 1, h + 1, w + 1)]) + 0x80) & 0xff
np.where(raw >= 128, raw - 256, raw).astype(np.int8).tofile("/tmp/dw_raw_in.bin")
np.transpose(flt, (2, 0, 1)).astype(np.int8).tofile("/tmp/dw_w.bin")
bias.astype(np.int32).tofile("/tmp/dw_bias.bin")
print("dumped /tmp/dw_raw_in.bin /tmp/dw_w.bin /tmp/dw_bias.bin  (C=%d KH=%d KW=%d)" % (C, KH, KW))
