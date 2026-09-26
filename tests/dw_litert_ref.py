#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Companion to tests/conv_dw_int8_runtime.c: run the Teflon capture's own model
# (dw_pt.tflite, one per-tensor int8 DEPTHWISE_CONV_2D) under TFLite's REFERENCE kernels on
# a seeded random int8 input, and write the input and TFLite's output beside the capture,
# both [C][H][W] int8:
#   litert-in.bin   the input the gate feeds the NPU
#   litert-out.bin  TFLite's output for it, the gate's oracle
# The reference kernels are the ones TFLite's semantics are defined by: XNNPACK, the
# interpreter's default, requantizes in fp32 and rounds differently at a boundary.
#
# Usage: dw_litert_ref.py [capture_dir [out_dir]]
#   capture_dir defaults to tests/data/teflon-dw-capture beside this script, out_dir to it.
# Needs ai_edge_litert, tflite_runtime or tensorflow.
import sys, os, numpy as np
try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter, OpResolverType
    except ImportError:
        import tensorflow as tf
        Interpreter, OpResolverType = tf.lite.Interpreter, tf.lite.experimental.OpResolverType

D = sys.argv[1] if len(sys.argv) > 1 \
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "teflon-dw-capture")
OUT = sys.argv[2] if len(sys.argv) > 2 else D
it = Interpreter(model_path=os.path.join(D, "dw_pt.tflite"),
                 experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
it.allocate_tensors()
ind, outd = it.get_input_details()[0], it.get_output_details()[0]
ops = [op["op_name"] for op in it._get_ops_details()]
if ops != ["DEPTHWISE_CONV_2D"]:
    sys.exit("expected one DEPTHWISE_CONV_2D and no delegate, got %s" % ops)

x = np.random.default_rng(20260923).integers(-128, 128, ind["shape"]).astype(np.int8)
it.set_tensor(ind["index"], x)
it.invoke()
y = it.get_tensor(outd["index"])
np.transpose(x[0], (2, 0, 1)).astype(np.int8).tofile(os.path.join(OUT, "litert-in.bin"))
np.transpose(y[0], (2, 0, 1)).astype(np.int8).tofile(os.path.join(OUT, "litert-out.bin"))
print("input %s %s, output %s %s; output %d..%d, %d of %d saturated" % (
    ind["shape"], ind["quantization"], outd["shape"], outd["quantization"],
    y.min(), y.max(), int(np.sum((y == 127) | (y == -128))), y.size))
print("wrote litert-in.bin litert-out.bin into %s" % OUT)
