#!/usr/bin/env python3
"""Manufacture RK3576 vendor captures that DRIVE THE DPU's LUT.

No capture anyone has of this part programs the LUT bank: every convolution capture
leaves 0x4100-0x4194 zero, which is why npu_regcmd_rk3576.c writes that whole bank as
zeros. The bank is what a nonlinear activation runs on, and activations are what
softmax, the norms, the gated FFN and attention all sit on — so it is the widest
piece of op coverage still closed on this part.

A capture can be MANUFACTURED rather than found: an ONNX compiled for rk3576 emits the
register program for whatever the graph asks for. A convolution followed by a nonlinear
activation is the shape a real network has and the shape the compiler fuses, so each
case here is Conv -> act at one fixed geometry, with a bare Conv as the CONTROL. The
control is the point: every register the act case writes and the control does not is
the activation's, and no sweep is needed to find them.

Two families, because they may not use the same machinery:

  piecewise  Relu, Relu6 (Clip), LeakyRelu, PRelu.  The DPU has explicit ReLU and a
             two-segment linear stage; these may never touch the LUT.
  nonlinear  Sigmoid, Tanh, HardSigmoid, HardSwish, Elu, Swish (x*Sigmoid), Softplus,
             Mish, Gelu, Exp, Sqrt.  These have no closed form in the datapath, so if
             the LUT is reachable at all it is these that reach it.

Also emitted standalone (no conv in front) so a fused program can be told apart from a
free-standing one, and at BOTH quantized int8 and float, since the RK3588's LUT has
separate LE (linear-exponent) and LO (linear-only) tables whose selection follows the
datatype.

Needs the same x86_64 toolkit as ../dw/mkdw.py; reuse that venv:

    ../dw/venv/bin/python mklut.py .

onnx 1.16 removed onnx.mapping and setuptools 81 removed pkg_resources; the toolkit
imports both, and neither failure names itself.
"""
import sys, os
import numpy as np
import onnx
from onnx import helper, TensorProto
from rknn.api import RKNN

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

C, K, HW = 32, 3, 16          # one geometry, so a diff against the control is clean.
OPSET = 13


def _conv_nodes(name_in, name_out):
    """A plain 3x3 conv over C channels: the carrier every fused case shares."""
    n = C * C * K * K
    w = ((np.arange(n, dtype=np.float32) % 11 - 5) / 16.0).reshape(C, C, K, K)
    b = ((np.arange(C, dtype=np.float32) + 1) / 8.0)
    node = helper.make_node("Conv", [name_in, "w", "b"], [name_out],
                            kernel_shape=[K, K], pads=[1, 1, 1, 1], strides=[1, 1])
    inits = [helper.make_tensor("w", TensorProto.FLOAT, [C, C, K, K],
                                w.astype(np.float32).tobytes(), raw=True),
             helper.make_tensor("b", TensorProto.FLOAT, [C], b.tobytes(), raw=True)]
    return [node], inits


def _act_nodes(kind, x, y):
    """The activation, as one or more ONNX nodes. Returns (nodes, initializers)."""
    n, i = [], []
    if kind == "none":
        return [helper.make_node("Identity", [x], [y])], []
    if kind == "relu":
        n = [helper.make_node("Relu", [x], [y])]
    elif kind == "relu6":
        i = [helper.make_tensor("clo", TensorProto.FLOAT, [], b"\x00\x00\x00\x00", raw=True),
             helper.make_tensor("chi", TensorProto.FLOAT, [], np.float32(6.0).tobytes(), raw=True)]
        n = [helper.make_node("Clip", [x, "clo", "chi"], [y])]
    elif kind == "leakyrelu":
        n = [helper.make_node("LeakyRelu", [x], [y], alpha=0.1)]
    elif kind == "sigmoid":
        n = [helper.make_node("Sigmoid", [x], [y])]
    elif kind == "tanh":
        n = [helper.make_node("Tanh", [x], [y])]
    elif kind == "hardsigmoid":
        n = [helper.make_node("HardSigmoid", [x], [y], alpha=1.0 / 6, beta=0.5)]
    elif kind == "hardswish":
        n = [helper.make_node("HardSigmoid", [x], ["hs"], alpha=1.0 / 6, beta=0.5),
             helper.make_node("Mul", [x, "hs"], [y])]
    elif kind == "swish":
        n = [helper.make_node("Sigmoid", [x], ["sg"]),
             helper.make_node("Mul", [x, "sg"], [y])]
    elif kind == "elu":
        n = [helper.make_node("Elu", [x], [y], alpha=1.0)]
    elif kind == "softplus":
        n = [helper.make_node("Softplus", [x], [y])]
    elif kind == "mish":
        n = [helper.make_node("Softplus", [x], ["sp"]),
             helper.make_node("Tanh", ["sp"], ["th"]),
             helper.make_node("Mul", [x, "th"], [y])]
    elif kind == "gelu":
        n = [helper.make_node("Erf", [x], [y])]
    elif kind == "exp":
        n = [helper.make_node("Exp", [x], [y])]
    elif kind == "sqrt":
        n = [helper.make_node("Abs", [x], ["ab"]),
             helper.make_node("Sqrt", ["ab"], [y])]
    else:
        raise ValueError(kind)
    return n, i


def build_onnx(path, kind, with_conv):
    nodes, inits = [], []
    src = "x"
    if with_conv:
        cn, ci = _conv_nodes("x", "c")
        nodes += cn
        inits += ci
        src = "c"
    an, ai = _act_nodes(kind, src, "y")
    nodes += an
    inits += ai
    graph = helper.make_graph(
        nodes, "lut",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, HW, HW])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, C, HW, HW])],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 8
    onnx.save(m, path)


def make_dataset(tag):
    """A one-entry calibration set spanning a wide range, so a saturating activation
    is not calibrated into its linear region and the table is exercised end to end."""
    npy = os.path.join(OUT, "cal_%s.npy" % tag)
    rng = np.random.RandomState(0)
    np.save(npy, rng.uniform(-8, 8, (1, C, HW, HW)).astype(np.float32))
    txt = os.path.join(OUT, "cal_%s.txt" % tag)
    open(txt, "w").write(npy + "\n")
    return txt


def emit(tag, kind, with_conv, quant):
    onnx_path = os.path.join(OUT, tag + ".onnx")
    rknn_path = os.path.join(OUT, tag + ".rknn")
    try:
        build_onnx(onnx_path, kind, with_conv)
    except Exception as e:
        print("SKIP", tag, "(onnx:", e, ")")
        return
    r = RKNN(verbose=False)
    r.config(target_platform="rk3576")
    if r.load_onnx(model=onnx_path) != 0:
        print("FAIL load", tag); r.release(); return
    ds = make_dataset(tag) if quant else None
    if r.build(do_quantization=quant, dataset=ds) != 0:
        print("FAIL build", tag); r.release(); return
    if r.export_rknn(rknn_path) != 0:
        print("FAIL export", tag); r.release(); return
    print("OK", tag, os.path.getsize(rknn_path), "bytes")
    r.release()


PIECEWISE = ["relu", "relu6", "leakyrelu"]
NONLINEAR = ["sigmoid", "tanh", "hardsigmoid", "hardswish", "swish", "elu",
             "softplus", "mish", "gelu", "exp", "sqrt"]

if __name__ == "__main__":
    # The controls first: a bare conv at each precision. Every register an activation
    # case writes and its control does not is the activation's.
    emit("ctl_conv_i8", "none", True, True)
    emit("ctl_conv_f16", "none", True, False)
    for kind in PIECEWISE + NONLINEAR:
        emit("conv_%s_i8" % kind, kind, True, True)
        emit("conv_%s_f16" % kind, kind, True, False)
        emit("bare_%s_i8" % kind, kind, False, True)
