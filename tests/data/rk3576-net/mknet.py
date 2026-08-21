# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""
mknet.py — a per-tensor-quantized TFLite classifier, flattened into the `.rnet` blob
tests/rk3576_net_gate.c runs on the RK3576.

WHAT IT CARRIES, AND WHY EACH PIECE IS HERE:

  the layer table       shapes, strides, TFLite's SAME/VALID padding resolved into an
                        explicit lead/trail pair per axis, and the per-tensor quant
                        triple. The gate reads geometry, never re-derives it.

  the weights           transposed ONCE, offline, into the layouts the library entries
                        take: [OC][IC][KH][KW] direct and [C][KH][KW] depthwise, where
                        TFLite stores [OC][KH][KW][IC] and [1][KH][KW][C].

  the goldens           EVERY intermediate tensor, straight out of the TFLite
                        interpreter, transposed to the CHW the library writes. This is
                        what makes a whole-network run diagnosable: a wrong layer is
                        named by its own index rather than by a wrong label 27 layers
                        later.

  THE INT8 REBASE       this model is uint8 and the part is int8, so every zero point,
                        every weight and every golden is shifted by -128 here. The
                        subtraction is exact and cancels: `w_stored - w_zp` is the same
                        integer in either domain, so the arithmetic is untouched and the
                        gate never sees a uint8.

The blob is not committed — `./fetch.sh && python3 mknet.py` rebuilds it byte for byte.
"""
import struct, sys, os
import numpy as np
import tflite
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.Pool2DOptions import Pool2DOptions
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "mobilenet_v1_1.0_224_quant.tflite")
IMAGE = os.path.join(HERE, "grace_hopper.bmp")
LABELS = os.path.join(HERE, "labels.txt")
OUT = os.path.join(HERE, "mobilenet_v1_224_quant.rnet")

MAGIC = b"RKNET\0\0\1"
VERSION = 1
LAYER_STRIDE = 128

KIND_CONV, KIND_DWCONV, KIND_AVGPOOL, KIND_SOFTMAX = 0, 1, 2, 3
ACT_NONE, ACT_RELU6 = 0, 1

# TFLite's Padding enum: SAME = 0, VALID = 1.
PAD_SAME, PAD_VALID = 0, 1


def pads(in_dim, k, stride, padding):
    """TFLite's own padding resolution, as an explicit (out, lead, trail)."""
    if padding == PAD_SAME:
        out = -(-in_dim // stride)                       # ceil
        total = max((out - 1) * stride + k - in_dim, 0)
        return out, total // 2, total - total // 2
    out = -(-(in_dim - k + 1) // stride)
    return out, 0, 0


def main():
    buf = open(MODEL, "rb").read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    g = m.Subgraphs(0)
    codes = []
    for i in range(m.OperatorCodesLength()):
        bc = m.OperatorCodes(i).BuiltinCode()
        codes.append([k for k, v in vars(tflite.BuiltinOperator).items() if v == bc][0])

    def raw(ix):
        t = g.Tensors(ix)
        b = m.Buffers(t.Buffer())
        if b.DataLength() == 0:
            return None
        shape = [int(t.Shape(k)) for k in range(t.ShapeLength())]
        dt = {tflite.TensorType.UINT8: np.uint8, tflite.TensorType.INT32: np.int32}
        return b.DataAsNumpy().view(dt[t.Type()]).reshape(shape)

    def quant(ix):
        q = g.Tensors(ix).Quantization()
        return float(q.Scale(0)), int(q.ZeroPoint(0))

    def shape(ix):
        t = g.Tensors(ix)
        return [int(t.Shape(k)) for k in range(t.ShapeLength())]

    # ---- the reference run: TFLite's own answer, every tensor preserved ----------
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=MODEL, experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    _, IH, IW, IC = [int(v) for v in inp["shape"]]

    img = Image.open(IMAGE).convert("RGB").resize((IW, IH), Image.BILINEAR)
    img = np.asarray(img, dtype=np.uint8)                      # HWC uint8
    interp.set_tensor(inp["index"], img[None, ...])
    interp.invoke()

    in_scale, in_zp = quant(int(g.Inputs(0)))

    layers, data = [], bytearray()

    def put(arr):
        off = len(data)
        data.extend(arr.tobytes())
        return off, len(data) - off

    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        name = codes[op.OpcodeIndex()]
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        outs = [op.Outputs(j) for j in range(op.OutputsLength())]
        bo = op.BuiltinOptions()
        if name == "RESHAPE":
            continue                                    # a view; the gate is already CHW

        L = dict(kind=None, act=ACT_NONE, kh=1, kw=1, sy=1, sx=1,
                 pl_y=0, pl_x=0, pt_y=0, pt_x=0,
                 w_zp=0, w_scale=1.0, w_off=0, w_bytes=0, b_off=0, b_bytes=0)

        isc, izp = quant(ins[0])
        osc, ozp = quant(outs[0])
        ishape, oshape = shape(ins[0]), shape(outs[0])
        ih, iw, ic = (ishape[1], ishape[2], ishape[3]) if len(ishape) == 4 \
                     else (1, 1, ishape[-1])
        oh, ow, oc = (oshape[1], oshape[2], oshape[3]) if len(oshape) == 4 \
                     else (1, 1, oshape[-1])

        if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            dw = name == "DEPTHWISE_CONV_2D"
            o = DepthwiseConv2DOptions() if dw else Conv2DOptions()
            o.Init(bo.Bytes, bo.Pos)
            if dw and o.DepthMultiplier() != 1:
                sys.exit("depth_multiplier != 1 is not lowered here")
            L["kind"] = KIND_DWCONV if dw else KIND_CONV
            L["sy"], L["sx"] = o.StrideH(), o.StrideW()
            fa = o.FusedActivationFunction()
            if fa == tflite.ActivationFunctionType.RELU6:
                L["act"] = ACT_RELU6
            elif fa != tflite.ActivationFunctionType.NONE:
                sys.exit("fused activation %d is not lowered here" % fa)
            f = raw(ins[1])                                    # OC,KH,KW,IC / 1,KH,KW,C
            L["kh"], L["kw"] = int(f.shape[1]), int(f.shape[2])
            oh_c, L["pl_y"], L["pt_y"] = pads(ih, L["kh"], L["sy"], o.Padding())
            ow_c, L["pl_x"], L["pt_x"] = pads(iw, L["kw"], L["sx"], o.Padding())
            assert (oh_c, ow_c) == (oh, ow), (oh_c, ow_c, oh, ow)
            L["w_scale"], wzp = quant(ins[1])
            L["w_zp"] = wzp - 128
            w = (f.astype(np.int16) - 128).astype(np.int8)
            w = w[0].transpose(2, 0, 1) if dw else w.transpose(0, 3, 1, 2)
            L["w_off"], L["w_bytes"] = put(np.ascontiguousarray(w))
            b = raw(ins[2]).astype(np.int32)
            L["b_off"], L["b_bytes"] = put(np.ascontiguousarray(b))
        elif name == "AVERAGE_POOL_2D":
            o = Pool2DOptions(); o.Init(bo.Bytes, bo.Pos)
            if o.FusedActivationFunction() != tflite.ActivationFunctionType.NONE:
                sys.exit("a fused activation on a pool is not lowered here")
            L["kind"] = KIND_AVGPOOL
            L["kh"], L["kw"] = o.FilterHeight(), o.FilterWidth()
            L["sy"], L["sx"] = o.StrideH(), o.StrideW()
            oh_c, L["pl_y"], L["pt_y"] = pads(ih, L["kh"], L["sy"], o.Padding())
            ow_c, L["pl_x"], L["pt_x"] = pads(iw, L["kw"], L["sx"], o.Padding())
            assert (oh_c, ow_c) == (oh, ow)
        elif name == "SOFTMAX":
            L["kind"] = KIND_SOFTMAX
        else:
            sys.exit("op %s is not lowered here" % name)

        # The fused ReLU6's clamp, in the int8 domain. On this model it always comes
        # out as the whole int8 range -- the output scale IS 6/255 with a zero point at
        # the bottom of the range -- so the clamp costs nothing and the gate asserts
        # that rather than assuming it.
        if L["act"] == ACT_RELU6:
            lo = max(0, ozp + int(round(0.0 / osc)))
            hi = min(255, ozp + int(round(6.0 / osc)))
        else:
            lo, hi = 0, 255
        L["clamp_lo"], L["clamp_hi"] = lo - 128, hi - 128

        golden = interp.get_tensor(outs[0])                    # NHWC or NC uint8
        gi = (golden.astype(np.int16) - 128).astype(np.int8)
        gi = gi[0].transpose(2, 0, 1) if gi.ndim == 4 else gi.reshape(-1)
        L["g_off"], L["g_bytes"] = put(np.ascontiguousarray(gi))

        L.update(ic=ic, ih=ih, iw=iw, oc=oc, oh=oh, ow=ow,
                 in_zp=izp - 128, out_zp=ozp - 128, in_scale=isc, out_scale=osc)
        layers.append(L)

    # ---- the network input, CHW int8, and the two answers to score against -------
    img_off, _ = put(np.ascontiguousarray(
        (img.astype(np.int16) - 128).astype(np.int8).transpose(2, 0, 1)))
    logits_t = int(g.Outputs(0))                                # SOFTMAX output
    pre = layers[-2]                                            # the classifier CONV
    probs = interp.get_tensor(logits_t).reshape(-1)
    probs_off, _ = put(np.ascontiguousarray(probs.astype(np.uint8)))
    labels = open(LABELS, "rb").read()
    labels_off = len(data); data.extend(labels)

    n_classes = int(pre["oc"])
    hdr = bytearray(128)
    struct.pack_into("<8sIIIIIIIif", hdr, 0, MAGIC, VERSION, len(layers),
                     LAYER_STRIDE, IC, IH, IW, n_classes, in_zp - 128, in_scale)
    struct.pack_into("<IIIIII", hdr, 48, 128, img_off, probs_off,
                     labels_off, len(labels), len(data))

    tbl = bytearray(LAYER_STRIDE * len(layers))
    for i, L in enumerate(layers):
        struct.pack_into("<12I", tbl, i * LAYER_STRIDE,
                         L["kind"], L["act"], L["ic"], L["ih"], L["iw"],
                         L["oc"], L["oh"], L["ow"], L["kh"], L["kw"], L["sy"], L["sx"])
        struct.pack_into("<4I", tbl, i * LAYER_STRIDE + 48,
                         L["pl_y"], L["pl_x"], L["pt_y"], L["pt_x"])
        struct.pack_into("<5i", tbl, i * LAYER_STRIDE + 64,
                         L["in_zp"], L["w_zp"], L["out_zp"],
                         L["clamp_lo"], L["clamp_hi"])
        struct.pack_into("<3f", tbl, i * LAYER_STRIDE + 84,
                         L["in_scale"], L["w_scale"], L["out_scale"])
        struct.pack_into("<6I", tbl, i * LAYER_STRIDE + 96,
                         L["w_off"], L["w_bytes"], L["b_off"], L["b_bytes"],
                         L["g_off"], L["g_bytes"])

    base = 128 + len(tbl)
    for i in range(len(layers)):                                # data offsets are absolute
        for fld in (96, 104, 112):
            v, = struct.unpack_from("<I", tbl, i * LAYER_STRIDE + fld)
            struct.pack_into("<I", tbl, i * LAYER_STRIDE + fld, v + base)
    for fld in (52, 56, 60):
        v, = struct.unpack_from("<I", hdr, fld)
        struct.pack_into("<I", hdr, fld, v + base)
    struct.pack_into("<I", hdr, 68, base + len(data))

    with open(OUT, "wb") as f:
        f.write(hdr); f.write(tbl); f.write(data)

    kinds = ("conv", "dwconv", "avgpool", "softmax")
    for i, L in enumerate(layers):
        print("%2d %-8s %4dx%-3dx%-4d -> %4dx%-3dx%-4d k%dx%d s%d pad %d/%d %d/%d "
              "act%d zp %d/%d/%d" %
              (i, kinds[L["kind"]], L["ic"], L["ih"], L["iw"], L["oc"], L["oh"],
               L["ow"], L["kh"], L["kw"], L["sy"], L["pl_y"], L["pt_y"],
               L["pl_x"], L["pt_x"], L["act"], L["in_zp"], L["w_zp"], L["out_zp"]))
    top = np.argsort(-probs.astype(np.int32))[:5]
    names = labels.decode().splitlines()
    print("\nTFLite top-5: " + ", ".join("%s (%d)" % (names[t], probs[t]) for t in top))
    print("wrote %s, %d bytes, %d layers" % (OUT, os.path.getsize(OUT), len(layers)))


if __name__ == "__main__":
    main()
