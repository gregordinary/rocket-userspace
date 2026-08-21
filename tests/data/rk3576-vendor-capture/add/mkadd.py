#!/usr/bin/env python3
"""Manufacture RK3576 vendor captures that DRIVE AN ELEMENTWISE ADD.

The op library computes convolution, depthwise convolution, pooling, a LUT activation
and an int8 matmul on this part. A residual ADD is the one primitive between that set
and the second network worth running here: every inverted-residual and every ResNet
block is conv -> conv -> add, and MobileNetV1 -- the only graph gated so far -- is a
straight chain with no add anywhere in it.

The DPU's EW stage is where an add would land. Its field layout is known (the RK3588's,
transferred unchanged to the RK3576's moved offsets: EW_CFG at 0x407C, EW_BYPASS bit 0,
EW_LUT_BYPASS bit 7, EW_ALU_ALGO bits 16-19, EW_OP_SRC bit 6). What is NOT known is
which of those the vendor sets for an add, where the second operand's ADDRESS goes, and
whether the operand converter (EW_CVT_OFFSET/SCALE/SHIFT) is what rescales an int8
operand into the accumulator domain -- an int8 add needs that, because the two inputs
carry different quantization scales.

A capture can be MANUFACTURED: an ONNX compiled for rk3576 emits the register program
for whatever graph is asked for. Each case here separates something:

  none      conv alone -- the CONTROL, so a diff names the add's registers and nothing
            else. Every other case is this conv plus one thing.
  res       conv -> Add(conv_out, x): the residual as a graph writes it, one operand
            computed and one not.
  add2      two convs -> Add: BOTH operands computed. Separates "the compiler treats a
            graph input specially" from the add itself.
  addconst  conv -> Add(conv_out, initializer): a constant operand of the same shape.
            If this and `res` differ, the operand source is what differs.
  resmul    Mul in place of Add, same wiring: makes EW_ALU_ALGO move while every other
            field is held, which is the one field a single capture cannot pin.
  bareadd   Add(x, y) with two graph inputs and no conv at all. Says whether the add is
            a program in its own right or only ever an epilogue -- and a bare op may be
            routed to the CPU, which is itself the answer.
  btl       the MobileNetV2 inverted residual entire: 1x1 expand -> 3x3 depthwise ->
            1x1 project -> Add. The ground truth for whether the add FUSES into the
            project convolution's epilogue or costs its own program.

Channels cover the 16 granule and MobileNetV2's own awkward counts (16, 24, 96 are not
multiples of 32). Planes cover square/non-square and odd, so an output-extent round-up
is separable from the plane.

Needs the same x86_64 toolkit as ../dw/mkdw.py; reuse that venv:

    ../dw/venv/bin/python mkadd.py .
"""
import sys, os
import numpy as np
import onnx
from onnx import helper, TensorProto
from rknn.api import RKNN

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
OPSET = 13


def _w(name, oc, ic, k, seed):
    n = oc * ic * k * k
    v = ((np.arange(n, dtype=np.float32) + seed) % 11 - 5) / 16.0
    return helper.make_tensor(name, TensorProto.FLOAT, [oc, ic, k, k],
                              v.astype(np.float32).tobytes(), raw=True)


def _b(name, oc, seed):
    v = ((np.arange(oc, dtype=np.float32) + seed) % 7 + 1) / 8.0
    return helper.make_tensor(name, TensorProto.FLOAT, [oc], v.tobytes(), raw=True)


def _conv(x, y, tag, oc, ic, k=3, group=1, stride=1):
    """A plain Conv with SAME-ish padding, its weights named after the node."""
    wn, bn = "w_" + tag, "b_" + tag
    node = helper.make_node("Conv", [x, wn, bn], [y], kernel_shape=[k, k],
                            pads=[k // 2] * 4, strides=[stride, stride], group=group)
    return [node], [_w(wn, oc, ic // group, k, len(tag)), _b(bn, oc, len(tag))]


def build_onnx(path, C, hw, mode):
    ih, iw = (hw, hw) if isinstance(hw, int) else hw
    nodes, inits, ins = [], [], ["x"]

    if mode == "none":
        n, i = _conv("x", "y", "a", C, C); nodes += n; inits += i
    elif mode in ("res", "resmul"):
        n, i = _conv("x", "c", "a", C, C); nodes += n; inits += i
        op = "Mul" if mode == "resmul" else "Add"
        nodes.append(helper.make_node(op, ["c", "x"], ["y"]))
    elif mode == "add2":
        n, i = _conv("x", "c0", "a", C, C); nodes += n; inits += i
        n, i = _conv("x", "c1", "bb", C, C); nodes += n; inits += i
        nodes.append(helper.make_node("Add", ["c0", "c1"], ["y"]))
    elif mode == "addconst":
        n, i = _conv("x", "c", "a", C, C); nodes += n; inits += i
        k = np.linspace(-0.5, 0.5, C * ih * iw, dtype=np.float32)
        inits.append(helper.make_tensor("k", TensorProto.FLOAT, [1, C, ih, iw],
                                        k.tobytes(), raw=True))
        nodes.append(helper.make_node("Add", ["c", "k"], ["y"]))
    elif mode in ("bareadd", "bareasym"):
        ins.append("x2")
        nodes.append(helper.make_node("Add", ["x", "x2"], ["y"]))
    elif mode in ("baresub", "baresubr"):
        # Sub is the one ALU opcode that is NOT symmetric in its operands, so the two
        # orders are the same graph with the roles swapped. Whatever separates them in
        # the register program names an operand ROLE, which is the thing a capture
        # cannot carry as an address.
        ins.append("x2")
        a, b = ("x", "x2") if mode == "baresub" else ("x2", "x")
        nodes.append(helper.make_node("Sub", [a, b], ["y"]))
    elif mode == "addbcast":
        # A per-channel operand: same channel count, plane 1x1. If the second operand
        # rides a DMA with its own geometry registers, a broadcast operand has to move
        # them; if it rides a per-element feed, the compiler must materialise the plane.
        n, i = _conv("x", "c", "a", C, C); nodes += n; inits += i
        k = np.linspace(-0.5, 0.5, C, dtype=np.float32)
        inits.append(helper.make_tensor("k", TensorProto.FLOAT, [1, C, 1, 1],
                                        k.tobytes(), raw=True))
        nodes.append(helper.make_node("Add", ["c", "k"], ["y"]))
    elif mode == "btl":
        # MobileNetV2's inverted residual, expansion 6, stride 1, in_c == out_c.
        E = C * 6
        n, i = _conv("x", "e", "exp", E, C, k=1); nodes += n; inits += i
        nodes.append(helper.make_node("Clip", ["e", "lo", "hi"], ["er"]))
        n, i = _conv("er", "d", "dw", E, E, k=3, group=E); nodes += n; inits += i
        nodes.append(helper.make_node("Clip", ["d", "lo", "hi"], ["dr"]))
        n, i = _conv("dr", "p", "prj", C, E, k=1); nodes += n; inits += i
        nodes.append(helper.make_node("Add", ["p", "x"], ["y"]))
        inits += [helper.make_tensor("lo", TensorProto.FLOAT, [], b"\0\0\0\0", raw=True),
                  helper.make_tensor("hi", TensorProto.FLOAT, [],
                                     np.float32(6.0).tobytes(), raw=True)]
    else:
        raise ValueError(mode)

    graph = helper.make_graph(
        nodes, "add",
        [helper.make_tensor_value_info(v, TensorProto.FLOAT, [1, C, ih, iw])
         for v in ins],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, C, ih, iw])],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 8
    onnx.save(m, path)


def make_dataset(C, hw, tag, n_inputs, ranges=None):
    """`ranges` gives each input its own calibration amplitude. Two inputs at the SAME
    amplitude quantize to the SAME scale, which makes their two converters identical and
    so unattributable — asymmetric amplitudes are what let a register be assigned to an
    operand."""
    ih, iw = (hw, hw) if isinstance(hw, int) else hw
    rng = np.random.RandomState(0)
    paths = []
    for j in range(n_inputs):
        amp = 1.0 if ranges is None else ranges[j]
        npy = os.path.join(OUT, "cal_%s_%d.npy" % (tag, j))
        np.save(npy, rng.uniform(-amp, amp, (1, C, ih, iw)).astype(np.float32))
        paths.append(npy)
    txt = os.path.join(OUT, "cal_%s.txt" % tag)
    open(txt, "w").write(" ".join(paths) + "\n")
    return txt


def emit(tag, C, hw, mode, quant=True):
    onnx_path = os.path.join(OUT, tag + ".onnx")
    rknn_path = os.path.join(OUT, tag + ".rknn")
    try:
        build_onnx(onnx_path, C, hw, mode)
    except Exception as e:
        print("SKIP", tag, "(onnx:", e, ")")
        return
    r = RKNN(verbose=False)
    r.config(target_platform="rk3576")
    if r.load_onnx(model=onnx_path) != 0:
        print("FAIL load", tag); r.release(); return
    two_in = mode in ("bareadd", "bareasym", "baresub", "baresubr")
    ranges = (1.0, 64.0) if mode == "bareasym" else None
    ds = make_dataset(C, hw, tag, 2 if two_in else 1, ranges) if quant else None
    if r.build(do_quantization=quant, dataset=ds) != 0:
        print("FAIL build", tag); r.release(); return
    if r.export_rknn(rknn_path) != 0:
        print("FAIL export", tag); r.release(); return
    print("OK", tag, os.path.getsize(rknn_path), "bytes")
    r.release()


# tag, channels, plane, mode
CASES = [
    # The control, and the four wirings of one add at one geometry.
    ("ctl_conv_c32_16",   32, 16, "none"),
    ("res_c32_16",        32, 16, "res"),
    ("add2_c32_16",       32, 16, "add2"),
    ("addconst_c32_16",   32, 16, "addconst"),
    ("mul_c32_16",        32, 16, "resmul"),
    ("bare_add_c32_16",   32, 16, "bareadd"),

    # Channels: the 16 granule, and MobileNetV2's own non-multiples of 32.
    ("res_c16_16",        16, 16, "res"),
    ("res_c24_16",        24, 16, "res"),
    ("res_c64_16",        64, 16, "res"),
    ("res_c96_16",        96, 16, "res"),

    # Plane: larger, and non-square/odd so an extent round-up is separable.
    ("res_c32_28",        32, 28, "res"),
    ("res_c32_15x18",     32, (15, 18), "res"),

    # WHICH REGISTER BELONGS TO WHICH OPERAND. No capture can carry a base address, so
    # an operand is named indirectly: give the two of them properties that MUST differ
    # in the program, and read off what moves.
    ("asym_add_c32_16",   32, 16, "bareasym"),   # scales 64x apart
    ("sub_c32_16",        32, 16, "baresub"),    # an ALU op that is not symmetric
    ("subrev_c32_16",     32, 16, "baresubr"),   # the same, operands swapped
    ("bcast_add_c32_16",  32, 16, "addbcast"),   # an operand with no plane

    # The real block: does the add fuse into the project conv's epilogue?
    ("btl_c24_28",        24, 28, "btl"),
    ("btl_c32_16",        32, 16, "btl"),
]

if __name__ == "__main__":
    want = set(sys.argv[2:])
    for tag, C, hw, mode in CASES:
        if want and tag not in want: continue
        emit(tag, C, hw, mode)
