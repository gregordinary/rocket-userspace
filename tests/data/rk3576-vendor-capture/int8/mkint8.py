#!/usr/bin/env python3
"""Manufacture the RK3576 vendor INT8 convolution captures beside this script.

These are the quantized programs the regcmd gate is anchored on: the int8 packed-image
(ARGB) first convolution, the int8 direct convolution and the int8 depthwise one, at
planes large enough that the compiler splits each into several row-window tasks. Every
model is written here as ONNX from seeded weights and compiled against seeded
calibration data, so the set rebuilds on any host that runs the toolkit:

  chain_*   a MobileNet-shaped stem compiled as ONE graph: a 3->32 k3 s2 first conv on
            a 224x224 image, ReLU6, a 32-channel k3 depthwise, ReLU6, a 32->64 1x1,
            ReLU6, a 64-channel k3 s2 depthwise, ReLU6. In the graph the later layers
            are middle layers fed by the first one's output, and at 112x112 each is
            split into windowed tasks: the continuation tasks that keep the kernel
            overlap resident in the CBUF come from here.
  conv0_*   the first conv alone, on the same 224x224 image.
  conv2d_*  a 16->128 k5 s2 direct conv on 80x80 with the SAME pad's uneven border,
            1 row/column before and 2 after.
  conv64_*  the first conv alone on a 64x64 image.
  dw_*      the 32-channel k3 depthwise alone on 112x112.
  iso_*     the conv2d geometry three more times with PER-TENSOR weight quantization
            (quantized_method='layer'; every other model here is per-channel). Their
            weights are built to move one quantity per output channel: the bias alone
            (one weight tensor shared by every channel), the per-channel magnitude
            alone (every tap of a channel the same positive value), and the sign mix
            at a fixed magnitude (a per-channel count of positive taps). With
            per-tensor weights the compiler programs a different BS-stage
            arrangement from conv2d's: 0x4044 = 0, 0x4050 = 0x80021111,
            0x501C = 0x114, against 1, 0x80011111 and 0x710. Compiling conv2d itself
            per-tensor gives the same three words, so they follow the quantization
            method and not these weights. mkgolden.py carries the iso_* programs as
            references rather than as conv oracles.

The toolkit compiles every graph three ways, so each file holds three compiles'
programs: the plane's row windows at CBUF origin 0, the same windows at origin 7168, and
the plane cut into two halves, the second at origin 6144. mkgolden.py names cases in
the order it reads files, which is sorted by file name, and the gate's CBUF row-reuse
table refers to the stem's programs by their unsuffixed names; the tags below sort the
stem first for that reason, and the iso_* tags keep the substrings mkgolden.py
identifies them by.

The calibration data is uniform over [0, 255], an 8-bit image range, so the input
quantizes with an int8 zero point of -128: the direct and depthwise programs pad with
0xffffff80 at 0x1084 and the ARGB ones with 0x808080, which is the zero point the gate
drives the ARGB emitter at.

Needs the toolkit environment ../dw/mkdw.py documents (x86_64, CPython 3.10,
onnx==1.14.1, setuptools<81; built with rknn-toolkit2 2.3.0):

    ../dw/venv/bin/python mkint8.py .

The intermediates (ONNX and calibration arrays) go to a temporary directory, and only
the .rknn files are written to the output directory.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
from onnx import helper, TensorProto

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def tensor(name, a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return helper.make_tensor(name, TensorProto.FLOAT, list(a.shape), a.tobytes(), raw=True)


def rand_weights(rng, oc, icg, k):
    """Weights and bias uniform in +-1/sqrt(fan_in)."""
    bound = 1.0 / np.sqrt(icg * k * k)
    w = rng.uniform(-bound, bound, (oc, icg, k, k)).astype(np.float32)
    b = rng.uniform(-bound, bound, (oc,)).astype(np.float32)
    return w, b


class Graph:
    """A chain of Conv layers, each optionally followed by ReLU6, on one NCHW input."""

    def __init__(self, ic, ih, iw):
        self.shape = [1, ic, ih, iw]
        self.cur, self.c, self.h, self.w = "x", ic, ih, iw
        self.nodes, self.inits, self.n = [], [], 0

    def conv(self, oc, k, stride, pads, w, b, group=1, relu6=False):
        """pads is (top, left, bottom, right)."""
        i = self.n
        self.n += 1
        wn, bn, yn = "w%d" % i, "b%d" % i, "y%d" % i
        self.inits += [tensor(wn, w), tensor(bn, b)]
        pt, pl, pb, pr = pads
        self.nodes.append(helper.make_node(
            "Conv", [self.cur, wn, bn], [yn], kernel_shape=[k, k],
            pads=[pt, pl, pb, pr], strides=[stride, stride], group=group))
        self.h = (self.h + pt + pb - k) // stride + 1
        self.w = (self.w + pl + pr - k) // stride + 1
        self.c = oc
        self.cur = yn
        if relu6:
            if not any(t.name == "relu6_lo" for t in self.inits):
                self.inits += [tensor("relu6_lo", np.float32(0.0)),
                               tensor("relu6_hi", np.float32(6.0))]
            rn = "r%d" % i
            self.nodes.append(helper.make_node("Clip", [yn, "relu6_lo", "relu6_hi"], [rn]))
            self.cur = rn
        return self

    def save(self, path):
        self.nodes[-1].output[0] = "y"
        graph = helper.make_graph(
            self.nodes, "int8conv",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, self.shape)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT,
                                           [1, self.c, self.h, self.w])],
            self.inits)
        m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        onnx.save(m, path)


SAME_K3 = (1, 1, 1, 1)
NO_PAD = (0, 0, 0, 0)
SAME_K5_S2 = (1, 1, 2, 2)     # k5 s2 on an even plane: SAME splits its 3 pad rows 1 + 2


def stem(seed):
    rng = np.random.RandomState(seed)
    g = Graph(3, 224, 224)
    g.conv(32, 3, 2, SAME_K3, *rand_weights(rng, 32, 3, 3), relu6=True)
    g.conv(32, 3, 1, SAME_K3, *rand_weights(rng, 32, 1, 3), group=32, relu6=True)
    g.conv(64, 1, 1, NO_PAD, *rand_weights(rng, 64, 32, 1), relu6=True)
    g.conv(64, 3, 2, SAME_K3, *rand_weights(rng, 64, 1, 3), group=64, relu6=True)
    return g


def first_conv(hw):
    def build(seed):
        rng = np.random.RandomState(seed)
        return Graph(3, hw, hw).conv(32, 3, 2, SAME_K3, *rand_weights(rng, 32, 3, 3))
    return build


def depthwise(seed):
    rng = np.random.RandomState(seed)
    return Graph(32, 112, 112).conv(32, 3, 1, SAME_K3, *rand_weights(rng, 32, 1, 3),
                                    group=32)


IC, OC, K = 16, 128, 5


def k5(w, b):
    return Graph(IC, 80, 80).conv(OC, K, 2, SAME_K5_S2, w, b)


def conv2d(seed):
    rng = np.random.RandomState(seed)
    return k5(*rand_weights(rng, OC, IC, K))


def iso_bias(seed):
    """One weight tensor shared by every output channel; the bias ramps over oc."""
    rng = np.random.RandomState(seed)
    w1 = rng.uniform(-1.0, 1.0, (IC, K, K)).astype(np.float32)
    w = np.broadcast_to(w1, (OC, IC, K, K)).copy()
    b = np.linspace(-2.0, 2.0, OC).astype(np.float32)
    return k5(w, b)


def iso_scale(seed):
    """Every tap of channel o is (o+1)/OC: the magnitude ramps over oc, every weight is
    positive, and the bias is zero."""
    w = np.empty((OC, IC, K, K), np.float32)
    for o in range(OC):
        w[o] = (o + 1) / OC
    return k5(w, np.zeros(OC, np.float32))


def iso_sum(seed):
    """Every tap is +0.5 or -0.5, and channel o has o*n/(OC-1) of them positive, at
    seeded positions: the magnitude is fixed, the weight sum ramps over oc, and the
    bias is zero."""
    rng = np.random.RandomState(seed)
    n = IC * K * K
    w = np.empty((OC, n), np.float32)
    for o in range(OC):
        row = np.full(n, -0.5, np.float32)
        row[rng.permutation(n)[:(o * n) // (OC - 1)]] = 0.5
        w[o] = row
    return k5(w.reshape(OC, IC, K, K), np.zeros(OC, np.float32))


# tag                     builder           seed  weight quantization
CASES = [
    ("chain_stem_224",       stem,             1, "channel"),
    ("conv0_argb_224",       first_conv(224),  2, "channel"),
    ("conv2d_c16_128_k5_s2", conv2d,           3, "channel"),
    ("conv64_argb_64",       first_conv(64),   4, "channel"),
    ("dw_c32_k3_s1_112",     depthwise,        5, "channel"),
    ("iso_bias_k5_s2",       iso_bias,         6, "layer"),
    ("iso_scale_k5_s2",      iso_scale,        7, "layer"),
    ("iso_sum_k5_s2",        iso_sum,          8, "layer"),
]


def calibration(tmp, tag, shape, seed):
    """A one-entry calibration set: seeded integers uniform over [0, 255]."""
    rng = np.random.RandomState(1000 + seed)
    npy = os.path.join(tmp, "cal_%s.npy" % tag)
    np.save(npy, rng.randint(0, 256, shape).astype(np.float32))
    txt = os.path.join(tmp, "cal_%s.txt" % tag)
    with open(txt, "w") as f:
        f.write(npy + "\n")
    return txt


def emit(tmp, tag, build, seed, qmethod):
    from rknn.api import RKNN
    g = build(seed)
    onnx_path = os.path.join(tmp, tag + ".onnx")
    rknn_path = os.path.join(OUT, tag + ".rknn")
    g.save(onnx_path)
    r = RKNN(verbose=False)
    r.config(target_platform="rk3576", quantized_method=qmethod)
    try:
        if r.load_onnx(model=onnx_path) != 0:
            print("FAIL load", tag)
            return 1
        if r.build(do_quantization=True,
                   dataset=calibration(tmp, tag, g.shape, seed)) != 0:
            print("FAIL build", tag)
            return 1
        if r.export_rknn(rknn_path) != 0:
            print("FAIL export", tag)
            return 1
    finally:
        r.release()
    print("OK", tag, os.path.getsize(rknn_path), "bytes")
    return 0


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="mkint8-")
    try:
        bad = sum(emit(tmp, *case) for case in CASES)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if bad else 0)
