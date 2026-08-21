#!/usr/bin/env python3
"""Rebuild the RK3576 vendor FLOAT captures beside this script.

Every .rknn in the parent directory is int8, so the fp16 precision fields had no
oracle and were inferred from the RK3588's field packing. They can be transcribed
instead: do_quantization=False on a FLOAT onnx leaves the weights float, and the
vendor compiler then picks the float datapath and emits a genuine fp16 conv program.
Several shapes, because the registers worth separating are the ones that MOVE with
ic — of the whole program only seven do, and six of those our emitter already
matched.

Needs an x86_64 host and a toolkit that is particular about its dependencies:

    uv venv --python 3.10 venv
    uv pip install --python ./venv/bin/python rknn-toolkit2 'onnx==1.14.1' \\
        'setuptools<81'
    ./venv/bin/python mkfloat.py .

onnx 1.16 removed onnx.mapping and setuptools 81 removed pkg_resources; the toolkit
imports both, and neither failure names itself.
"""
import sys, os
import numpy as np
import onnx
from onnx import helper, TensorProto

OUT = sys.argv[1] if len(sys.argv) > 1 else "."


def build_onnx(path, ic, oc, k, hw):
    pad = k // 2
    w = (np.arange(oc * ic * k * k, dtype=np.float32) % 7 - 3) / 8.0
    b = (np.arange(oc, dtype=np.float32) + 1) / 4.0
    node = helper.make_node(
        "Conv", ["x", "w", "b"], ["y"],
        kernel_shape=[k, k], pads=[pad, pad, pad, pad], strides=[1, 1], group=1)
    graph = helper.make_graph(
        [node], "conv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, ic, hw, hw])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, oc, hw, hw])],
        [helper.make_tensor("w", TensorProto.FLOAT, [oc, ic, k, k], w.tobytes(), raw=True),
         helper.make_tensor("b", TensorProto.FLOAT, [oc], b.tobytes(), raw=True)])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


from rknn.api import RKNN

CASES = [
    # (ic, oc, k, hw)
    (8,  32, 3, 16),
    (16, 32, 3, 16),
    (32, 32, 3, 16),
    (64, 32, 3, 16),
    (64, 32, 1, 16),
    (8,  64, 3, 32),
]

for ic, oc, k, hw in CASES:
    tag = "f16_ic%d_oc%d_k%d_%d" % (ic, oc, k, hw)
    onnx_path = os.path.join(OUT, tag + ".onnx")
    rknn_path = os.path.join(OUT, tag + ".rknn")
    build_onnx(onnx_path, ic, oc, k, hw)
    r = RKNN(verbose=False)
    r.config(target_platform="rk3576")
    if r.load_onnx(model=onnx_path) != 0:
        print("FAIL load", tag); r.release(); continue
    if r.build(do_quantization=False) != 0:
        print("FAIL build", tag); r.release(); continue
    if r.export_rknn(rknn_path) != 0:
        print("FAIL export", tag); r.release(); continue
    print("OK", tag, os.path.getsize(rknn_path), "bytes")
    r.release()
