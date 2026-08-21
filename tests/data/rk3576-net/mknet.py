# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""
mknet.py — a per-tensor-quantized TFLite classifier, flattened into the `.rnet` blob
tests/rk3576_net_gate.c runs on the RK3576.

    python3 mknet.py [v1|v2|r18]    # default: every model that is present

WHAT IT CARRIES, AND WHY EACH PIECE IS HERE:

  the layer table       shapes, strides, TFLite's SAME/VALID padding resolved into an
                        explicit lead/trail pair per axis, and the per-tensor quant
                        triple. The gate reads geometry, never re-derives it.

  the graph edges       every layer names the LAYERS its operands come from, rather than
                        the gate assuming "the one before". A residual network needs
                        that: a skip's second operand is produced three to five layers
                        back, and a ping-pong pair of buffers cannot express it.

  the weights           transposed ONCE, offline, into the layouts the library entries
                        take: [OC][IC][KH][KW] direct and [C][KH][KW] depthwise, where
                        TFLite stores [OC][KH][KW][IC] and [1][KH][KW][C].

  the goldens           EVERY intermediate tensor, straight out of the TFLite
                        interpreter, transposed to the CHW the library writes. This is
                        what makes a whole-network run diagnosable: a wrong layer is
                        named by its own index rather than by a wrong label 27 layers
                        later.

  THE INT8 REBASE       the MobileNets are uint8 and the part is int8, so every zero
                        point, every weight and every golden is shifted by -128 here. The
                        subtraction is exact and cancels: `w_stored - w_zp` is the same
                        integer in either domain, so the arithmetic is untouched and the
                        gate never sees a uint8. A model that is already int8 (ResNet-18,
                        which mkresnet18.py quantizes locally) is passed through.

                        THE NETWORK INPUT IS THE IMAGE BYTES, reinterpreted in whatever
                        the model's storage type is — `x` for a uint8 model and `x - 128`
                        for an int8 one — which is a convention about the model, not a
                        derivation from its input quantization. The uint8 MobileNets
                        carry it as scale 1/128 at zero point 128 and the int8 ResNet-18
                        as scale 1 at zero point -128, and mkresnet18.py ASSERTS the
                        latter, because a model whose input quantization drifted off it
                        would be fed a wrong image rather than refused.

  EXPLICIT PADDING      a PAD op ahead of a VALID convolution or pool is folded into that
                        op's lead/trail pair. It is how a graph converted from an explicit
                        symmetric pad arrives — TFLite's Conv2D carries only SAME and
                        VALID, so a pad that is neither survives as its own op — and it
                        reaches geometries SAME does not: ResNet-18's 7x7 stem pads 3
                        rows before and consumes 2 after, where SAME would be 2 and 3.
                        The trailing pad recorded is the one the last window CONSUMES,
                        which is what the CNA derives from the output extent, and it can
                        be smaller than the pad the model declares.

WHAT IS NOT HERE: the residual add's WEIGHTS. An add is lowered onto a convolution over
the two operands concatenated along channels, and the diagonal blocks that expresses are
chip arithmetic — rocket_residual_add_weights_rk3576() owns it. This carries only the
second operand's quantization, which is what that entry needs and what TFLite has.

The blobs are not committed — `./fetch.sh && python3 mknet.py` rebuilds them byte for
byte.
"""
import struct, sys, os
import numpy as np
import tflite
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.Pool2DOptions import Pool2DOptions
from tflite.AddOptions import AddOptions
from tflite.ConcatenationOptions import ConcatenationOptions
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = os.path.join(HERE, "grace_hopper.bmp")
LABELS = os.path.join(HERE, "labels.txt")

NETS = {
    "v1": ("mobilenet_v1_1.0_224_quant.tflite", "mobilenet_v1_224_quant.rnet"),
    "v2": ("mobilenet_v2_1.0_224_quant.tflite", "mobilenet_v2_224_quant.rnet"),
    "r18": ("resnet18_224_quant.tflite", "resnet18_224_quant.rnet"),
    "iv1": ("inception_v1_224_quant.tflite", "inception_v1_224_quant.rnet"),
    "iv3": ("inception_v3_299_quant.tflite", "inception_v3_299_quant.rnet"),
}

MAGIC = b"RKNET\0\0\1"
VERSION = 3
LAYER_STRIDE = 160

KIND_CONV, KIND_DWCONV, KIND_AVGPOOL, KIND_SOFTMAX, KIND_ADD = 0, 1, 2, 3, 4
KIND_MAXPOOL, KIND_CONCAT = 5, 6
MAX_SRC = 4                  # a concatenation's operands; every other kind uses one or two
ACT_NONE, ACT_RELU6, ACT_RELU = 0, 1, 2

NO_SRC = 0xFFFFFFFF          # the network input, or "this layer has no second operand"

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


def fused_act(fa, where):
    if fa == tflite.ActivationFunctionType.NONE:
        return ACT_NONE
    if fa == tflite.ActivationFunctionType.RELU6:
        return ACT_RELU6
    if fa == tflite.ActivationFunctionType.RELU:
        return ACT_RELU
    sys.exit("%s carries fused activation %d, which is not lowered here" % (where, fa))


def resolve(in_dim, k, stride, padding, ex):
    """One axis as (out, lead, trail), from a TFLite padding mode or an explicit pad.

    The trailing value is the pad the LAST WINDOW CONSUMES, which is what the CNA derives
    from the output extent and the leading pad, and it can be smaller than the pad the
    model declares: torchvision's stride-2 3x3 pads one row at each end of an even plane
    and the last window never reaches the trailing one."""
    if ex is None:
        return pads(in_dim, k, stride, padding)
    if padding != PAD_VALID:
        sys.exit("an explicit pad ahead of an op that also pads itself")
    lead, trail = ex
    out = (in_dim + lead + trail - k) // stride + 1
    return out, lead, max((out - 1) * stride + k - (in_dim + lead), 0)


def build(model_path, out_path):
    buf = open(model_path, "rb").read()
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
        dt = {tflite.TensorType.UINT8: np.uint8, tflite.TensorType.INT8: np.int8,
              tflite.TensorType.INT32: np.int32}
        return b.DataAsNumpy().view(dt[t.Type()]).reshape(shape)

    def quant(ix):
        q = g.Tensors(ix).Quantization()
        if q.ScaleLength() != 1:
            sys.exit("tensor %d carries %d scales; this blob format is per-tensor"
                     % (ix, q.ScaleLength()))
        return float(q.Scale(0)), int(q.ZeroPoint(0))

    # The storage domain. The part is int8; a uint8 model is rebased by -128, which is
    # exact and cancels, and an int8 one passes through.
    STORED_INT8 = g.Tensors(int(g.Inputs(0))).Type() == tflite.TensorType.INT8
    SH = 0 if STORED_INT8 else 128
    QMIN, QMAX = (-128, 127) if STORED_INT8 else (0, 255)

    def shape(ix):
        t = g.Tensors(ix)
        return [int(t.Shape(k)) for k in range(t.ShapeLength())]

    # ---- the reference run: TFLite's own answer, every tensor preserved ----------
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=model_path, experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    _, IH, IW, IC = [int(v) for v in inp["shape"]]

    img = Image.open(IMAGE).convert("RGB").resize((IW, IH), Image.BILINEAR)
    img = np.asarray(img, dtype=np.uint8)                      # HWC uint8
    qin = (img.astype(np.int16) - (128 - SH))
    interp.set_tensor(inp["index"],
                      qin.astype(np.int8 if STORED_INT8 else np.uint8)[None, ...])
    interp.invoke()

    in_scale, in_zp = quant(int(g.Inputs(0)))

    layers, data = [], bytearray()
    # Which LAYER produced a tensor. A RESHAPE is a view the gate never runs, so its
    # output resolves to whatever produced its input — the edge must survive the op
    # being dropped, or a skip that crosses one would name a layer that is not there.
    src_of = {}
    net_in = int(g.Inputs(0))
    # A PAD op's output tensor -> (the tensor it pads, (top, bottom, left, right)). The
    # op is not a layer; it is folded into whichever convolution or pool consumes it.
    pad_of = {}

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
            src_of[outs[0]] = src_of.get(ins[0], NO_SRC)   # a view; the gate is CHW
            continue
        if name == "PAD":
            pv = raw(ins[1]).reshape(-1, 2)
            if pv[0].any() or pv[3].any():
                sys.exit("pad %d pads the batch or the channels" % i)
            src = pad_of[ins[0]][0] if ins[0] in pad_of else ins[0]
            pad_of[outs[0]] = (src, (int(pv[1][0]), int(pv[1][1]),
                                     int(pv[2][0]), int(pv[2][1])))
            src_of[outs[0]] = src_of.get(src, NO_SRC)
            continue

        L = dict(kind=None, act=ACT_NONE, kh=1, kw=1, sy=1, sx=1,
                 pl_y=0, pl_x=0, pt_y=0, pt_x=0,
                 w_zp=0, w_scale=1.0, w_off=0, w_bytes=0, b_off=0, b_bytes=0,
                 src2=NO_SRC, src3=NO_SRC, src4=NO_SRC, in2_zp=0, in2_scale=0.0)

        # An explicit pad ahead of this op is this op's own lead/trail, and the plane it
        # reads is the one BEFORE the pad -- which is what the previous layer wrote.
        ins[0], explicit = (pad_of[ins[0]][0], pad_of[ins[0]][1]) \
                           if ins[0] in pad_of else (ins[0], None)

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
            L["act"] = fused_act(o.FusedActivationFunction(), "convolution %d" % i)
            f = raw(ins[1])                                    # OC,KH,KW,IC / 1,KH,KW,C
            L["kh"], L["kw"] = int(f.shape[1]), int(f.shape[2])
            oh_c, L["pl_y"], L["pt_y"] = resolve(ih, L["kh"], L["sy"], o.Padding(),
                                                 explicit[:2] if explicit else None)
            ow_c, L["pl_x"], L["pt_x"] = resolve(iw, L["kw"], L["sx"], o.Padding(),
                                                 explicit[2:] if explicit else None)
            assert (oh_c, ow_c) == (oh, ow), (oh_c, ow_c, oh, ow)
            L["w_scale"], wzp = quant(ins[1])
            L["w_zp"] = wzp - SH
            w = (f.astype(np.int16) - SH).astype(np.int8)
            w = w[0].transpose(2, 0, 1) if dw else w.transpose(0, 3, 1, 2)
            L["w_off"], L["w_bytes"] = put(np.ascontiguousarray(w))
            b = raw(ins[2]).astype(np.int32)
            L["b_off"], L["b_bytes"] = put(np.ascontiguousarray(b))
        elif name == "ADD":
            # A skip connection. Both operands are activations of the same shape; the
            # gate lowers this onto a 1x1 convolution over the two concatenated along
            # channels, so all that is carried is the SECOND operand's quantization and
            # the layer it comes from.
            o = AddOptions(); o.Init(bo.Bytes, bo.Pos)
            L["act"] = fused_act(o.FusedActivationFunction(), "add %d" % i)
            if len(ins) != 2 or shape(ins[0]) != shape(ins[1]):
                sys.exit("an add with a broadcast or constant operand is not lowered here")
            L["kind"] = KIND_ADD
            b_sc, b_zp = quant(ins[1])
            L["src2"] = src_of.get(ins[1], NO_SRC)
            if L["src2"] == NO_SRC:
                sys.exit("add %d: the second operand is not produced by a layer" % i)
            L["in2_scale"], L["in2_zp"] = b_sc, b_zp - SH
        elif name in ("AVERAGE_POOL_2D", "MAX_POOL_2D"):
            o = Pool2DOptions(); o.Init(bo.Bytes, bo.Pos)
            if o.FusedActivationFunction() != tflite.ActivationFunctionType.NONE:
                sys.exit("a fused activation on a pool is not lowered here")
            L["kind"] = KIND_AVGPOOL if name == "AVERAGE_POOL_2D" else KIND_MAXPOOL
            L["kh"], L["kw"] = o.FilterHeight(), o.FilterWidth()
            L["sy"], L["sx"] = o.StrideH(), o.StrideW()
            oh_c, L["pl_y"], L["pt_y"] = resolve(ih, L["kh"], L["sy"], o.Padding(),
                                                 explicit[:2] if explicit else None)
            ow_c, L["pl_x"], L["pt_x"] = resolve(iw, L["kw"], L["sx"], o.Padding(),
                                                 explicit[2:] if explicit else None)
            assert (oh_c, ow_c) == (oh, ow)
        elif name == "CONCATENATION":
            # A CONCATENATION IS PLACEMENT, NOT ARITHMETIC — but only when every operand
            # is already in the output's quantization. TFLite's own kernel requantizes an
            # operand that is not, and this format cannot carry that, so it is refused
            # rather than copied through. Every operand of these models is already there,
            # which is what makes the whole op free on the part: each producer writes its
            # own slice of one buffer and the consumer reads the buffer as its cube.
            o = ConcatenationOptions(); o.Init(bo.Bytes, bo.Pos)
            if o.FusedActivationFunction() != tflite.ActivationFunctionType.NONE:
                sys.exit("a fused activation on concat %d is not lowered here" % i)
            if o.Axis() not in (3, -1):
                sys.exit("concat %d joins axis %d; only the channel axis is lowered here"
                         % (i, o.Axis()))
            if len(ins) > MAX_SRC:
                sys.exit("concat %d has %d operands; the blob carries %d"
                         % (i, len(ins), MAX_SRC))
            L["kind"] = KIND_CONCAT
            off = 0
            for k, t in enumerate(ins):
                sc, zp = quant(t)
                if abs(sc - osc) > 1e-9 * max(abs(osc), 1e-30) or zp != ozp:
                    sys.exit("concat %d operand %d: (%g, %d) is not the output's (%g, %d)"
                             % (i, k, sc, zp, osc, ozp))
                src = src_of.get(t, NO_SRC)
                if src == NO_SRC:
                    sys.exit("concat %d: operand %d is not produced by a layer" % (i, k))
                if shape(t)[-1] != layers[src]["oc"]:
                    sys.exit("concat %d: operand %d is %d channels, its producer writes %d"
                             % (i, k, shape(t)[-1], layers[src]["oc"]))
                # A PLACED SLICE STARTS ON A 16-CHANNEL ATOM, which is what one cube
                # interleaves. An operand order that lands one anywhere else can still be
                # run by copying on the host, but it cannot be wired, so it is worth
                # knowing at build time rather than as a silent fallback.
                if off % 16:
                    sys.exit("concat %d: operand %d starts at channel %d, which is not a "
                             "16-channel atom" % (i, k, off))
                L["src%d" % (k + 1)] = src
                off += int(shape(t)[-1])
            if off != oc:
                sys.exit("concat %d: operands sum to %d channels, output is %d"
                         % (i, off, oc))
            ic, ih, iw = oc, oh, ow          # the layer IS its output; operands are slices
        elif name == "QUANTIZE":
            # A REQUANTIZATION EDGE, and the first op here that is neither arithmetic nor
            # placement. TFLite emits one where a tensor is consumed at a different scale
            # from the one its producer wrote: on Inception V3, on the output of a
            # concatenation whose operands were calibrated somewhere the consumer is not.
            # The values change, so it cannot be dropped the way a RESHAPE is.
            #
            # IT LOWERS ONTO A DEPTHWISE 1x1 IDENTITY — the same idiom the DPU LUT uses to
            # put one input value at one table index. A unit weight at unit scale leaves the
            # convolution's own epilogue computing
            #     out = requant((in - in_zp) * in_scale / out_scale) + out_zp,
            # which is exactly TFLite's kernel. So a requantization needs no new layer kind,
            # no new blob field and no new library entry: the part already gates a depthwise
            # 1x1, and the layer can write straight into a concatenation's slice like any
            # other producer.
            if shape(ins[0]) != shape(outs[0]) or \
               g.Tensors(ins[0]).Type() != g.Tensors(outs[0]).Type():
                sys.exit("quantize %d changes shape or storage type; only a rescale is "
                         "lowered here" % i)
            L["w_scale"], L["w_zp"] = 1.0, 0
            L["b_off"], L["b_bytes"] = put(np.zeros(ic, dtype=np.int32))
            if os.environ.get("MKNET_REQUANT_DIRECT"):
                # The same arithmetic as a DIRECT 1x1 convolution over an identity
                # matrix. Quadratic in the channel count where the depthwise form is
                # linear, so it is the control rather than the lowering.
                L["kind"] = KIND_CONV
                L["w_off"], L["w_bytes"] = put(
                    np.eye(ic, dtype=np.int8).reshape(ic, ic, 1, 1))
            else:
                L["kind"] = KIND_DWCONV
                L["w_off"], L["w_bytes"] = put(np.ones((ic, 1, 1), dtype=np.int8))
        elif name == "SOFTMAX":
            L["kind"] = KIND_SOFTMAX
        else:
            sys.exit("op %s is not lowered here" % name)

        # The first operand's producer, likewise as a LAYER index. NO_SRC is the network
        # input, which only the stem has.
        L["src1"] = NO_SRC if ins[0] == net_in else src_of.get(ins[0], NO_SRC)
        if L["src1"] == NO_SRC and ins[0] != net_in:
            sys.exit("layer %d: its input is not produced by a layer" % i)

        # The fused activation's clamp, in the int8 domain. On these models it always
        # comes out as the whole storage range -- a quantizer that sees a ReLU puts the
        # zero point at the bottom of the range, and a ReLU6's output scale IS 6/255 --
        # so the clamp costs nothing and the gate asserts that rather than assuming it.
        lo, hi = QMIN, QMAX
        if L["act"] in (ACT_RELU6, ACT_RELU):
            lo = max(QMIN, ozp)
            if L["act"] == ACT_RELU6:
                hi = min(QMAX, ozp + int(round(6.0 / osc)))
        L["clamp_lo"], L["clamp_hi"] = lo - SH, hi - SH

        golden = interp.get_tensor(outs[0])                    # NHWC or NC, stored type
        gi = (golden.astype(np.int16) - SH).astype(np.int8)
        gi = gi[0].transpose(2, 0, 1) if gi.ndim == 4 else gi.reshape(-1)
        L["g_off"], L["g_bytes"] = put(np.ascontiguousarray(gi))

        L.update(ic=ic, ih=ih, iw=iw, oc=oc, oh=oh, ow=ow,
                 in_zp=izp - SH, out_zp=ozp - SH, in_scale=isc, out_scale=osc)
        src_of[outs[0]] = len(layers)
        layers.append(L)

    # ---- the network input, CHW int8, and the two answers to score against -------
    img_off, _ = put(np.ascontiguousarray(
        (qin - SH).astype(np.int8).transpose(2, 0, 1)))
    logits_t = int(g.Outputs(0))                                # SOFTMAX output
    pre = layers[-2]                                            # the classifier CONV
    probs = interp.get_tensor(logits_t).reshape(-1)
    # The blob's copy is unsigned whatever the model stores, because the gate only ever
    # compares these against each other and +128 is order-preserving.
    probs_off, _ = put(np.ascontiguousarray(
        (probs.astype(np.int16) - SH + 128).astype(np.uint8)))
    n_classes = int(pre["oc"])
    labels = open(LABELS, "rb").read()
    # The coral label file carries TFLite's 1001-class list, whose entry 0 is the
    # background class the TF-Slim models have and a torchvision-derived one does not.
    lines = labels.splitlines()
    if n_classes == len(lines) - 1:
        labels = b"\n".join(lines[1:]) + b"\n"
    labels_off = len(data); data.extend(labels)
    hdr = bytearray(128)
    struct.pack_into("<8sIIIIIIIif", hdr, 0, MAGIC, VERSION, len(layers),
                     LAYER_STRIDE, IC, IH, IW, n_classes, in_zp - SH, in_scale)
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
        struct.pack_into("<2I", tbl, i * LAYER_STRIDE + 120, L["src1"], L["src2"])
        struct.pack_into("<i", tbl, i * LAYER_STRIDE + 128, L["in2_zp"])
        struct.pack_into("<f", tbl, i * LAYER_STRIDE + 132, L["in2_scale"])
        struct.pack_into("<2I", tbl, i * LAYER_STRIDE + 136, L["src3"], L["src4"])

    base = 128 + len(tbl)
    for i in range(len(layers)):                                # data offsets are absolute
        for fld in (96, 104, 112):
            v, = struct.unpack_from("<I", tbl, i * LAYER_STRIDE + fld)
            struct.pack_into("<I", tbl, i * LAYER_STRIDE + fld, v + base)
    for fld in (52, 56, 60):
        v, = struct.unpack_from("<I", hdr, fld)
        struct.pack_into("<I", hdr, fld, v + base)
    struct.pack_into("<I", hdr, 68, base + len(data))

    with open(out_path, "wb") as f:
        f.write(hdr); f.write(tbl); f.write(data)

    kinds = ("conv", "dwconv", "avgpool", "softmax", "add", "maxpool", "concat")
    for i, L in enumerate(layers):
        edge = ""
        if L["kind"] == KIND_ADD:
            edge = "  + layer %d (s=%.6g zp=%d)" % (L["src2"], L["in2_scale"],
                                                    L["in2_zp"])
        if L["kind"] == KIND_CONCAT:
            edge = "  <- " + " ".join(str(L[k]) for k in ("src1", "src2", "src3", "src4")
                                      if L[k] != NO_SRC)
        print("%2d %-8s %4dx%-3dx%-4d -> %4dx%-3dx%-4d k%dx%d s%d pad %d/%d %d/%d "
              "act%d zp %d/%d/%d%s" %
              (i, kinds[L["kind"]], L["ic"], L["ih"], L["iw"], L["oc"], L["oh"],
               L["ow"], L["kh"], L["kw"], L["sy"], L["pl_y"], L["pt_y"],
               L["pl_x"], L["pt_x"], L["act"], L["in_zp"], L["w_zp"], L["out_zp"],
               edge))
    top = np.argsort(-probs.astype(np.int32))[:5]
    names = labels.decode().splitlines()
    print("\nTFLite top-5: " + ", ".join("%s (%d)" % (names[t], probs[t]) for t in top))
    print("wrote %s, %d bytes, %d layers" % (out_path, os.path.getsize(out_path),
                                             len(layers)))


def main():
    which = sys.argv[1:] or list(NETS)
    for w in which:
        if w not in NETS:
            sys.exit("unknown network %r; expected one of %s" % (w, ", ".join(NETS)))
        model, out = NETS[w]
        model = os.path.join(HERE, model)
        if not os.path.exists(model):
            print("skipping %s: %s is not fetched" % (w, os.path.basename(model)))
            continue
        print("== %s ==" % w)
        build(model, os.path.join(HERE, out))
        print("")


if __name__ == "__main__":
    main()
