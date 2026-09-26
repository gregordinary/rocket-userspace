#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Companion to tests/conv_dw_int8_runtime.c and tests/replay_dw_mesa.c. From the Teflon
# DW capture and its dw_pt.tflite model, (1) check that Mesa's uint8 formulas, applied to
# the model's int8 bytes, reproduce Mesa's captured weight and bias cubes byte for byte,
# which is what says the capture ran an int8 model through a uint8 driver, and (2) write
# the model's filter and bias beside the capture, for conv_dw_int8_runtime:
#   dw_w.bin       raw int8 filter [C][KH][KW]
#   dw_bias.bin    int32 bias      [C]
#
# Mesa's formulas (rkt_coefs.c, rkt_ml.c), for a uint8 tensor u with zero point Z:
#   input/weight cube value = u - 0x80
#   bias[oc] = bias_q[oc] - sum_kernel(u_w - Z_w)*(Z_in - 0x80)
#   output                  = npu_byte + 0x80
# The capture fed them the int8 bytes and zero points as if they were uint8, which is not
# the model's function. rocket_conv2d_dw_int8 passes the uint8-equivalent values instead
# (u = x + 0x80, Z = zp + 0x80), and conv_dw_int8_runtime gates it against TFLite.
#
# Usage: dw_dump_capture.py [capture_dir [out_dir]]
#   capture_dir defaults to tests/data/teflon-dw-capture beside this script, out_dir to it.
import sys, os, numpy as np
try:
    from ai_edge_litert.interpreter import Interpreter as Interp
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as Interp
    except ImportError:
        import tensorflow as tf; Interp = tf.lite.Interpreter

D = sys.argv[1] if len(sys.argv) > 1 \
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "teflon-dw-capture")
OUT = sys.argv[2] if len(sys.argv) > 2 else D
it = Interp(model_path=D + "/dw_pt.tflite"); it.allocate_tensors()
flt = it.get_tensor(2)[0].astype(np.int32)        # [KH,KW,C] int8 filter
bias = it.get_tensor(1).astype(np.int64)          # [C] int32
in_zp, w_zp = -2, 0
C, KH, KW = flt.shape[2], flt.shape[0], flt.shape[1]; G = 64
mesa_wt = np.fromfile(D + "/mesa-weights-000-000.bin", dtype=np.uint8)
mesa_bs = np.fromfile(D + "/mesa-biases-000-000.bin", dtype=np.int32).astype(np.int64)

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

# (2) the model's filter and bias for the C gate
np.transpose(flt, (2, 0, 1)).astype(np.int8).tofile(os.path.join(OUT, "dw_w.bin"))
bias.astype(np.int32).tofile(os.path.join(OUT, "dw_bias.bin"))
print("wrote dw_w.bin dw_bias.bin into %s  (C=%d KH=%d KW=%d)" % (OUT, C, KH, KW))
