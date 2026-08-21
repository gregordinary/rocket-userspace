#!/usr/bin/env python3
"""First-conv ARGB captures whose weights NAME themselves, and the decoder that reads
the ARGB weight cube's layout out of them.

The twelve found ARGB programs pin the first-conv register program completely, and
none of them carries the weight BO those registers point at -- so all that is settled
about the cube is its SIZE and stride, `oc * kh * round16(4*kw)` bytes. A cube is the
one thing a register program cannot tell you, and on this part the depthwise path
already proved that guessing it by analogy with the RK3588 produces a program that is
register-for-register right and computes nothing.

A capture settles it directly, the same way `../dw/named/mknamed.py` settled the
depthwise cube: give every weight its own value and the packed cube in the .rknn names
the (output channel, kernel row, kernel column, input channel) of each slot.

The weights must be FLOAT (do_quantization=False). Per-channel weight quantization
normalizes each channel by its own maximum, which erases any naming carried in
magnitude; fp16 represents integers exactly to 2048, so a unique-integer ramp survives
the float path untouched and is findable in the file by its exact value set.

WHAT MAKES IT AN ARGB CAPTURE. Three input channels. The normal datapath needs `ic` a
multiple of 32, so the compiler has no choice: at `ic <= 4` it takes the CNA's packed-
image sub-encoding. That is checked rather than assumed -- `--check` reads the emitted
register program back and requires CNA_CONV_CON1 (0x100C) to carry GROUP_LINE_OFF
(bit 29) and an ARGB_IN nibble of `8 | (ic-1)`, which is what identifies the path.

Except at TWO channels, where the compiler has a preference rather than a constraint:
it takes the packed path at 1, 3 and 4 image channels and compiles a 2-channel conv
down the DIRECT one instead. That is a toolkit choice and not a hardware bound -- the
packed path computes bit-exactly at two channels on the part (rk3576_conv_lib_gate
`fc-2ch-k3`) -- so this set holds no 2-channel capture, and the library keeps taking
the packed path for every count at or below four.

The low bits of that word are NOT the same on the two precisions, and this is the one
place these captures correct the transcription: the found int8 ARGB programs carry
0x2000a006 and these float ones carry 0x2020a122, so the low field is 6 at int8 and 2
at fp16 -- bit 2 clears on the float path while GROUP_LINE_OFF and ARGB_IN stay put.

WHAT THIS DECODES, AND WHAT IT DOES NOT. The cube below is the FLOAT one. The
depthwise path already showed that this part's int8 cube and its float cube are not
the same object, and the int8 ARGB cube is NOT settled here: a quantized capture built
the same way takes the ARGB path (0x100C reads 0x2000a006, exactly the found value)
but its weights do not survive as a findable value set, so `rocket_conv2d_int8_rk3576()`
keeps its `ic <= 4` refusal.

Dependency pins are ../dw/mkdw.py's:

    ./venv/bin/python mkargb.py .
"""
import sys, os, struct, glob
import numpy as np
import onnx
from onnx import helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE

# tag                     ic  oc  k  ih  iw  stride
#
# The first five are square, stride 1 and 32x32, which is what the weight cube needed.
# A register formula needs more than that: every one of them holds iw == ih == 32 and
# stride 1, so `iw`, `ih`, the plane and the entry count are one column. The rest vary
# ONE axis at a time against them, which is what makes a fitted term separable from a
# coincidence -- the found int8 ARGB set is all k3 stride 2 224/64-wide, and four
# formulas fitted through that set's confounds were wrong.
CASES = [
    ("argbf_ic3_oc16_k3_32",  3, 16, 3, 32, 32, 1),  # the shape a vision stem uses
    ("argbf_ic3_oc32_k3_32",  3, 32, 3, 32, 32, 1),  # a second oc: separates the oc stride
    ("argbf_ic3_oc16_k1_32",  3, 16, 1, 32, 32, 1),  # kw=1: the 4*kw fold, which no found
    ("argbf_ic3_oc16_k5_32",  3, 16, 5, 32, 32, 1),  # kw=5: capture could separate
    ("argbf_ic4_oc16_k3_32",  4, 16, 3, 32, 32, 1),  # 4 channels: is the 4th lane the image's
    # width alone: does the CBUF entry count carry the int8 path's +1 lookahead?
    ("argbf_ic3_oc16_k3_16w", 3, 16, 3, 32, 16, 1),
    ("argbf_ic3_oc16_k3_64w", 3, 16, 3, 32, 64, 1),
    # height alone: separates iw from ih in every stride and count
    ("argbf_ic3_oc16_k3_r48", 3, 16, 3, 48, 32, 1),
    ("argbf_ic3_oc16_k3_r16", 3, 16, 3, 16, 32, 1),
    # stride 2, which every float capture lacked and every found int8 one had
    ("argbf_ic3_oc16_k3_s2",  3, 16, 3, 32, 32, 2),
    ("argbf_ic3_oc16_k1_s2",  3, 16, 1, 32, 32, 2),
    # oc past one 16-channel interleave group, and past 32
    ("argbf_ic3_oc64_k3_32",  3, 64, 3, 32, 32, 1),
    ("argbf_ic3_oc48_k3_32",  3, 48, 3, 32, 32, 1),
    # k7, and a rectangular kernel would need a second kernel axis the compiler folds
    ("argbf_ic3_oc16_k7_32",  3, 16, 7, 32, 32, 1),
    # fewer than three image channels: is ARGB_IN 8|(ic-1) at the bottom, and does the
    # lane count stay 4 when the image carries one? (TWO is deliberately absent -- see
    # the note on the compiler's channel-count preference below.)
    ("argbf_ic1_oc16_k3_32",  1, 16, 3, 32, 32, 1),
]

R76_ARGB_CONV_MODE = 6


def names(oc, ic, k):
    """Unique per (oc, ic, kh, kw). Zero is replaced -- a zero slot is not a name."""
    n = oc * ic * k * k
    v = np.arange(n, dtype=np.float32) - n // 2
    v[v == 0] = n
    return v


def build_onnx(path, ic, oc, k, ih, iw, stride):
    pad = k // 2
    w = names(oc, ic, k).reshape(oc, ic, k, k)
    b = ((np.arange(oc, dtype=np.float32) + 1) / 4.0)
    oh = (ih + 2 * pad - k) // stride + 1
    ow = (iw + 2 * pad - k) // stride + 1
    node = helper.make_node(
        "Conv", ["x", "w", "b"], ["y"],
        kernel_shape=[k, k], pads=[pad, pad, pad, pad],
        strides=[stride, stride], group=1)
    graph = helper.make_graph(
        [node], "argbconv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, ic, ih, iw])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, oc, oh, ow])],
        [helper.make_tensor("w", TensorProto.FLOAT, [oc, ic, k, k], w.tobytes(), raw=True),
         helper.make_tensor("b", TensorProto.FLOAT, [oc], b.tobytes(), raw=True)])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def emit(tag, ic, oc, k, ih, iw, stride):
    from rknn.api import RKNN
    onnx_path = os.path.join(OUT, tag + ".onnx")
    rknn_path = os.path.join(OUT, tag + ".rknn")
    build_onnx(onnx_path, ic, oc, k, ih, iw, stride)
    r = RKNN(verbose=False)
    r.config(target_platform="rk3576")
    if r.load_onnx(model=onnx_path) != 0:
        print("FAIL load", tag); r.release(); return
    if r.build(do_quantization=False, dataset=None) != 0:
        print("FAIL build", tag); r.release(); return
    if r.export_rknn(rknn_path) != 0:
        print("FAIL export", tag); r.release(); return
    print("OK", tag, os.path.getsize(rknn_path), "bytes")
    r.release()


# ---- reading the capture back -------------------------------------------------

TARGETS = {0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU", 0x2001: "RDMA",
           0x0041: "SYNC", 0x0081: "BCAST"}


def runs(path, min_run=20):
    data = open(path, "rb").read()
    n = len(data) // 8
    words = struct.unpack("<%dQ" % n, data[:n * 8])
    out, i = [], 0
    while i < n:
        if ((words[i] >> 48) & 0xffff) in TARGETS:
            j = i
            while j < n and ((words[j] >> 48) & 0xffff) in TARGETS:
                j += 1
            if j - i >= min_run:
                out.append([(int((w >> 48) & 0xffff), int(w & 0xffff),
                             int((w >> 16) & 0xffffffff)) for w in words[i:j]])
            i = j
        else:
            i += 1
    return out


GROUP_LINE_OFF = 1 << 29
ARGB_IN_SHIFT = 12


def check():
    """Did the compiler take the ARGB path? GROUP_LINE_OFF plus ARGB_IN = 8|(ic-1)."""
    bad = 0
    for tag, ic, oc, k, ih, iw, stride in CASES:
        path = os.path.join(OUT, tag + ".rknn")
        if not os.path.exists(path):
            print("%-24s NO CAPTURE" % tag); bad += 1; continue
        modes = sorted({v for run in runs(path) for t, r, v in run
                        if t == 0x0201 and r == 0x100c})
        want_in = 8 | (ic - 1)
        argb = [m for m in modes
                if (m & GROUP_LINE_OFF) and ((m >> ARGB_IN_SHIFT) & 0xf) == want_in]
        print("%-24s ic=%d oc=%2d k=%d  0x100C %s  ARGB=%s"
              % (tag, ic, oc, k, [hex(m) for m in modes], "YES" if argb else "NO"))
        bad += 0 if argb else 1
    print("\n%s" % ("every capture took the ARGB path" if not bad
                    else "%d capture(s) did NOT" % bad))
    return bad


OC_GROUP = 16          # output channels interleaved per tap
LANES = 4              # the CVT expands every pixel to four lanes


def slot(o, c, kh, kw, KH, KW):
    """Halfword index of weight (output channel o, lane c, kernel row kh, column kw).

    Output channels are interleaved in groups of SIXTEEN inside one tap, four lanes
    each, and the tap axis is kh-outer. `4*kw` is NOT rounded up to 16 here, which is
    what separates this from the size the register program declares."""
    return ((o // OC_GROUP) * (KH * KW * OC_GROUP * LANES)
            + kh * (KW * OC_GROUP * LANES)
            + kw * (OC_GROUP * LANES)
            + (o % OC_GROUP) * LANES
            + c)


def cube_positions(path, ic, oc, k):
    """Every name's halfword offset, relative to the first slot of the cube."""
    n = oc * ic * k * k
    h = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    h = h[:len(h) // 2 * 2].view(np.float16)
    v2i = {}
    for i in range(n):
        v = float(np.float16(i - n // 2 if i - n // 2 != 0 else n))
        v2i.setdefault(v, i)
    pos = {}
    for p, x in enumerate(h):
        li = v2i.get(float(x))
        if li is not None:
            pos.setdefault(li, []).append(p)
    if len(pos) < n:
        return None, None
    base = min(pos[i][-1] for i in range(n))
    return base, pos


def decode():
    """Re-derive the layout from the committed captures. The regression check."""
    bad = 0
    for tag, ic, oc, k, ih, iw, stride in CASES:
        path = os.path.join(OUT, tag + ".rknn")
        if not os.path.exists(path):
            print("%-24s NO CAPTURE" % tag); bad += 1; continue
        n = oc * ic * k * k
        base, pos = cube_positions(path, ic, oc, k)
        if base is None:
            print("%-24s cube NOT FOUND" % tag); bad += 1; continue
        wrong = 0
        for o in range(oc):
            for c in range(ic):
                for kh in range(k):
                    for kw in range(k):
                        li = ((o * ic + c) * k + kh) * k + kw
                        if base + slot(o, c, kh, kw, k, k) not in pos[li]:
                            wrong += 1
        print("%-24s ic=%d oc=%2d k=%d  n=%4d  cube at halfword %d  layout %s"
              % (tag, ic, oc, k, n, base, "REPRODUCED" if not wrong else "MISMATCH"))
        bad += 0 if not wrong else 1
    print("\n%s" % ("all cubes reproduce the layout" if not bad
                    else "%d cube(s) do not" % bad))
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(1 if check() else 0)
    if "--decode" in sys.argv:
        sys.exit(1 if decode() else 0)
    else:
        for c in CASES:
            emit(*c)
