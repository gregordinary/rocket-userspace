#!/usr/bin/env python3
"""Manufacture RK3576 vendor captures that DRIVE THE POOLING ENGINE (PPU).

Nothing in any capture of this part drives the PPU, so its register page is undecoded
here and the op library has no pooling. Pooling is the one thing a detection or
classification graph needs beyond convolution, and it is cheaper to reach than the LUT
because the geometry is the conv's geometry — the pooled cube stays in the same NC1HWC2
layout, so no new packing is involved.

A capture can be MANUFACTURED: an ONNX compiled for rk3576 emits whatever the graph
asks for. Each case here is Conv -> pool, because a bare pool may be routed to the CPU
while one behind a convolution is what a real network has and what the compiler places
on the NPU; a bare form is emitted too so the two can be told apart.

The axes, each separating something:

  method    MaxPool vs AveragePool vs GlobalAveragePool vs ReduceMean — which of
            these reaches the PPU at all, and which is lowered onto something else
            (a 1x1 conv by a ones kernel, say, which is how the RK3588 does reduce).
  kernel    2, 3, 5, 7 at stride 1 and at stride = kernel.
  plane     square and non-square, odd and even, so a round-up in the output extent
            is separable from the plane itself.
  channels  16 / 32 / 48, to catch the same 16-channel granule the conv path has.
  padding   SAME and VALID: the conv path's pad_left is load-bearing and silent, so
            the pool's is worth pinning before trusting it.

Needs the same x86_64 toolkit as ../dw/mkdw.py; reuse that venv:

    ../dw/venv/bin/python mkpool.py .
"""
import sys, os
import numpy as np
import onnx
from onnx import helper, TensorProto
from rknn.api import RKNN

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
OPSET = 13


def _conv_nodes(C, name_in, name_out):
    n = C * C * 3 * 3
    w = ((np.arange(n, dtype=np.float32) % 11 - 5) / 16.0).reshape(C, C, 3, 3)
    b = ((np.arange(C, dtype=np.float32) + 1) / 8.0)
    node = helper.make_node("Conv", [name_in, "w", "b"], [name_out],
                            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    inits = [helper.make_tensor("w", TensorProto.FLOAT, [C, C, 3, 3],
                                w.astype(np.float32).tobytes(), raw=True),
             helper.make_tensor("b", TensorProto.FLOAT, [C], b.tobytes(), raw=True)]
    return [node], inits


def _pool_node(method, x, y, k, stride, same):
    pad = (k // 2) if same else 0
    if method == "max":
        return helper.make_node("MaxPool", [x], [y], kernel_shape=[k, k],
                                strides=[stride, stride], pads=[pad, pad, pad, pad])
    if method == "avg":
        return helper.make_node("AveragePool", [x], [y], kernel_shape=[k, k],
                                strides=[stride, stride], pads=[pad, pad, pad, pad],
                                count_include_pad=0)
    if method == "gap":
        return helper.make_node("GlobalAveragePool", [x], [y])
    if method == "gmp":
        return helper.make_node("GlobalMaxPool", [x], [y])
    raise ValueError(method)


def out_hw(method, ih, iw, k, stride, same):
    if method in ("gap", "gmp"):
        return 1, 1
    pad = (k // 2) if same else 0
    return (ih + 2 * pad - k) // stride + 1, (iw + 2 * pad - k) // stride + 1


def build_onnx(path, C, hw, method, k, stride, same, with_conv):
    """method None is the CONTROL: the conv carrier with no pool at all."""
    ih, iw = (hw, hw) if isinstance(hw, int) else hw
    nodes, inits = [], []
    src = "x"
    if with_conv:
        cn, ci = _conv_nodes(C, "x", "c")
        nodes += cn
        inits += ci
        src = "c"
    if method is None:
        nodes.append(helper.make_node("Identity", [src], ["y"]))
        oh, ow = ih, iw
    else:
        nodes.append(_pool_node(method, src, "y", k, stride, same))
        oh, ow = out_hw(method, ih, iw, k, stride, same)
    graph = helper.make_graph(
        nodes, "pool",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, ih, iw])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, C, oh, ow])],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 8
    onnx.save(m, path)


def make_dataset(C, hw, tag):
    npy = os.path.join(OUT, "cal_%s.npy" % tag)
    ih, iw = (hw, hw) if isinstance(hw, int) else hw
    rng = np.random.RandomState(0)
    np.save(npy, rng.uniform(-1, 1, (1, C, ih, iw)).astype(np.float32))
    txt = os.path.join(OUT, "cal_%s.txt" % tag)
    open(txt, "w").write(npy + "\n")
    return txt


def emit(tag, C, hw, method, k, stride, same, with_conv=True, quant=True):
    onnx_path = os.path.join(OUT, tag + ".onnx")
    rknn_path = os.path.join(OUT, tag + ".rknn")
    try:
        build_onnx(onnx_path, C, hw, method, k, stride, same, with_conv)
    except Exception as e:
        print("SKIP", tag, "(onnx:", e, ")")
        return
    r = RKNN(verbose=False)
    r.config(target_platform="rk3576")
    if r.load_onnx(model=onnx_path) != 0:
        print("FAIL load", tag); r.release(); return
    ds = make_dataset(C, hw, tag) if quant else None
    if r.build(do_quantization=quant, dataset=ds) != 0:
        print("FAIL build", tag); r.release(); return
    if r.export_rknn(rknn_path) != 0:
        print("FAIL export", tag); r.release(); return
    print("OK", tag, os.path.getsize(rknn_path), "bytes")
    r.release()


# tag, C, plane, method, k, stride, SAME
CASES = [
    # The control: conv alone, so a diff names the pool's registers.
    ("ctl_conv_c32_16",     32, 16, None, 0, 0, False),

    # Method, at one geometry.
    ("max_c32_k2s2_16",     32, 16, "max", 2, 2, False),
    ("avg_c32_k2s2_16",     32, 16, "avg", 2, 2, False),
    ("gap_c32_16",          32, 16, "gap", 0, 0, False),
    ("gmp_c32_16",          32, 16, "gmp", 0, 0, False),

    # Kernel, stride 1 SAME: separates the kernel from the output extent.
    ("max_c32_k3s1_16",     32, 16, "max", 3, 1, True),
    ("max_c32_k5s1_16",     32, 16, "max", 5, 1, True),
    ("max_c32_k7s1_16",     32, 16, "max", 7, 1, True),
    ("avg_c32_k3s1_16",     32, 16, "avg", 3, 1, True),

    # Stride == kernel, VALID: the ordinary downsampling pool.
    ("max_c32_k3s3_15",     32, 15, "max", 3, 3, False),
    ("avg_c32_k3s3_15",     32, 15, "avg", 3, 3, False),

    # Plane: odd, and non-square, so a round-up in the output extent is separable.
    ("max_c32_k2s2_15x18",  32, (15, 18), "max", 2, 2, False),
    ("max_c32_k2s2_17x19",  32, (17, 19), "max", 2, 2, False),
    ("max_c32_k3s2_19",     32, 19, "max", 3, 2, True),

    # Channels: the 16 granule.
    ("max_c16_k2s2_16",     16, 16, "max", 2, 2, False),
    ("max_c48_k2s2_16",     48, 16, "max", 2, 2, False),
    ("max_c64_k2s2_16",     64, 16, "max", 2, 2, False),
    ("max_c8_k2s2_16",       8, 16, "max", 2, 2, False),

    # Padding: SAME against VALID at the same kernel and stride.
    ("max_c32_k3s2_16_same", 32, 16, "max", 3, 2, True),
    ("max_c32_k3s2_16_vld",  32, 16, "max", 3, 2, False),
]

# Bare forms, to tell a fused pool apart from a free-standing one.
BARE = [("bare_max_c32_k2s2_16", 32, 16, "max", 2, 2, False),
        ("bare_avg_c32_k2s2_16", 32, 16, "avg", 2, 2, False),
        ("bare_gap_c32_16",      32, 16, "gap", 0, 0, False)]

if __name__ == "__main__":
    for tag, C, hw, method, k, s, same in CASES:
        emit(tag, C, hw, method, k, s, same)
    for tag, C, hw, method, k, s, same in BARE:
        emit(tag, C, hw, method, k, s, same, with_conv=False)
