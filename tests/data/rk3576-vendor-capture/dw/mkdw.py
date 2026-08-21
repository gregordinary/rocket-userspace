#!/usr/bin/env python3
"""Manufacture RK3576 vendor DEPTHWISE captures.

The eight .rknn captures in the parent directory carry twenty depthwise programs
between them, and they confound almost everything worth separating: every C=32
program in them is stride 1 and every C=64 one is stride 2, every one is k=3, and
every one is a multiple of 32 channels. A vendor capture can be manufactured
instead — `group=C` in the ONNX makes the compiler emit the depthwise path at
whatever (C, k, stride, plane) is asked for — and that separates all of it in one
pass, with no hardware at all.

What the captures beside this script settle, at 22 channel counts from 8 to 256,
strides 1 and 2, kernels 1/3/5/7 and planes 14 to 32:

  0x4050  the depthwise BS config tracks the CHANNEL COUNT and is flat in the
          stride. Bits[9:8] are the 16-channel group count minus one, modulo 4.
  0x1028 / 0x103C / 0x1044
          the CBUF entry counts are taken against a channel count rounded up to 16
          and then, if that lands on a group index of 3 mod 4, up one group more:
          48 rounds to 64 and 112 to 128, while 80 and 144 stay put.
  0x101C / 0x1020 / 0x1030
          the weight fields round the channel count up to 16 only, and a weight
          occupies a 16-bit slot.
  0x401C  the destination surface is ow*oh_full ROUNDED UP TO FOUR — invisible at
          every power-of-two plane, and 272/324/364 at planes of 15x18, 17x19 and 19x19.
  0x40B8  the surface add is four of those surfaces minus the task's own rows, and
          the 4 is flat in the kernel size (which the all-k=3 captures could not
          separate from kh+1).

`named/` carries FLOAT captures whose weights are unique per (channel, tap), which
is what decoded the depthwise weight cube rather than inferring it. mknamed.py
rebuilds them.

Needs an x86_64 host and a toolkit particular about its dependencies:

    uv venv --python 3.10 venv
    uv pip install --python ./venv/bin/python rknn-toolkit2 'onnx==1.14.1' \\
        'setuptools<81'
    ./venv/bin/python mkdw.py .

onnx 1.16 removed onnx.mapping and setuptools 81 removed pkg_resources; the toolkit
imports both, and neither failure names itself.

An even kernel is NOT worth capturing: k=2 makes the compiler emit an extra 1x1
helper program alongside the conv, which is not a program this emitter produces and
shows up as a spurious gate failure.
"""
import sys, os
import numpy as np
import onnx
from onnx import helper, TensorProto
from rknn.api import RKNN

OUT = sys.argv[1] if len(sys.argv) > 1 else "."


def build_onnx(path, C, k, stride, hw, wfn=None):
    """hw is the plane: an int for a square one, or (h, w)."""
    ih, iw = (hw, hw) if isinstance(hw, int) else hw
    pad = k // 2
    n = C * k * k
    w = wfn(C, k) if wfn is not None else ((np.arange(n, dtype=np.float32) % 7 - 3) / 8.0)
    w = w.reshape(C, 1, k, k).astype(np.float32)
    b = ((np.arange(C, dtype=np.float32) + 1) / 4.0)
    oh = (ih + 2 * pad - k) // stride + 1
    ow = (iw + 2 * pad - k) // stride + 1
    node = helper.make_node(
        "Conv", ["x", "w", "b"], ["y"],
        kernel_shape=[k, k], pads=[pad, pad, pad, pad],
        strides=[stride, stride], group=C)
    graph = helper.make_graph(
        [node], "dwconv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, ih, iw])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, C, oh, ow])],
        [helper.make_tensor("w", TensorProto.FLOAT, [C, 1, k, k], w.tobytes(), raw=True),
         helper.make_tensor("b", TensorProto.FLOAT, [C], b.tobytes(), raw=True)])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def make_dataset(C, hw, tag):
    """A one-entry calibration set. Quantization needs one; its content only has to
    exercise the range, and the register program does not depend on it — only the
    pad constant (0x1084), which carries the input zero point, follows from it."""
    npy = os.path.join(OUT, "cal_%s.npy" % tag)
    ih, iw = (hw, hw) if isinstance(hw, int) else hw
    rng = np.random.RandomState(0)
    np.save(npy, rng.uniform(-1, 1, (1, C, ih, iw)).astype(np.float32))
    txt = os.path.join(OUT, "cal_%s.txt" % tag)
    open(txt, "w").write(npy + "\n")
    return txt


def emit(tag, C, k, stride, hw, quant=True, wfn=None, qmethod="channel"):
    onnx_path = os.path.join(OUT, tag + ".onnx")
    rknn_path = os.path.join(OUT, tag + ".rknn")
    build_onnx(onnx_path, C, k, stride, hw, wfn)
    r = RKNN(verbose=False)
    r.config(target_platform="rk3576", quantized_method=qmethod)
    if r.load_onnx(model=onnx_path) != 0:
        print("FAIL load", tag); r.release(); return
    ds = make_dataset(C, hw, tag) if quant else None
    if r.build(do_quantization=quant, dataset=ds) != 0:
        print("FAIL build", tag); r.release(); return
    if r.export_rknn(rknn_path) != 0:
        print("FAIL export", tag); r.release(); return
    print("OK", tag, os.path.getsize(rknn_path), "bytes")
    r.release()


# The set committed beside this script. Each line is here because it separates
# something no other line does; the comment says what.
CASES = [
    # tag                  C    k  s  hw
    ("dw_c8_k3_s1_16",     8,  3, 1, 16),   # below the 16 granule
    ("dw_c16_k3_s1_16",   16,  3, 1, 16),   # 0x4050 field 0
    ("dw_c24_k3_s1_16",   24,  3, 1, 16),   # between granules
    ("dw_c32_k3_s1_16",   32,  3, 1, 16),   # field 1; the corner the old captures had
    ("dw_c32_k3_s2_16",   32,  3, 2, 16),   # same C, other stride: breaks the confound
    ("dw_c32_k1_s1_16",   32,  1, 1, 16),
    ("dw_c32_k5_s1_16",   32,  5, 1, 16),   # 0x40B8's plane term vs kh+1
    ("dw_c32_k7_s1_16",   32,  7, 1, 16),
    ("dw_c32_k3_s1_19",   32,  3, 1, 19),        # ow*oh=361: the 0x401C round-up
    ("dw_c32_k3_s1_17x19", 32, 3, 1, (17, 19)),  # 323, the other odd residue
    ("dw_c32_k3_s1_15x18", 32, 3, 1, (15, 18)),  # 270, the even one
    ("dw_c48_k3_s1_16",   48,  3, 1, 16),   # field 2, and the feature granule bumps
    ("dw_c64_k3_s1_16",   64,  3, 1, 16),   # field 3, at stride 1
    ("dw_c80_k3_s1_16",   80,  3, 1, 16),   # the field wraps back to 0
    ("dw_c112_k3_s1_16", 112,  3, 1, 16),   # the granule bumps a second time
    ("dw_c176_k3_s1_16", 176,  3, 1, 16),   # and a third, past the first wrap
    ("dw_c256_k3_s1_16", 256,  3, 1, 16),
]

if __name__ == "__main__":
    for tag, C, k, s, hw in CASES:
        emit(tag, C, k, s, hw)
