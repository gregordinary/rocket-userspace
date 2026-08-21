# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""
mkresnet18.py — a per-tensor int8 ResNet-18, built rather than fetched.

    .venv-build/bin/python mkresnet18.py             # writes resnet18_224_quant.tflite
    .venv-build/bin/python mkresnet18.py --per-axis  # writes resnet18_224_peraxis.tflite

WHY THIS EXISTS. MobileNetV1 and V2 arrive from google-coral/test_data already
quantized per tensor. No such ResNet-18 is published: the well-known checkpoints are
float, and every quantized ResNet in circulation is per-axis. So the model this gate
needs is assembled here from a public float checkpoint and quantized locally, which
also makes the quantization contract something this file states rather than inherits.

WHAT IT ASSEMBLES.

  the weights      timm/resnet18.a1_in1k, as safetensors — the torchvision ResNet-18
                   parameter layout under torchvision's own names. Read with a 30-line
                   parser rather than a framework: safetensors is a JSON header and a
                   raw tensor blob.

  the fold         every convolution is followed by a batch norm in the checkpoint and
                   by nothing in the graph built here: gamma/sqrt(var+eps) is folded
                   into the kernel and the rest into a bias. The arithmetic is exact.

  the input        the model takes RAW PIXELS, 0..255, and the ImageNet normalization
                   is folded into conv1 the same way. That is what makes the quantized
                   input tensor the identity on bytes (scale 1.0, zero point -128), so
                   mknet.py feeds this network the same image bytes it feeds the
                   MobileNets and the blob format does not fork.

                   THE COST, AND IT IS THE ONE DEVIATION FROM torchvision: conv1's
                   explicit 3-pixel border is padded in the RAW domain, so it is black
                   where torchvision's is the channel mean. It is a 3-pixel frame on a
                   224-pixel plane and the gate scores against this model's own TFLite
                   output, so nothing downstream depends on the difference.

  the padding      EXPLICIT, matching torchvision, not TFLite's SAME. They differ at
                   five of this graph's ops — the 7x7 stem (lead 3 trail 2 against
                   SAME's 2/3), the max pool and the three stride-2 3x3 convolutions
                   (lead 1 trail 0 against SAME's 0/1) — and a SAME graph would be a
                   one-pixel-shifted network with the checkpoint's accuracy thrown
                   away. It arrives as a PAD op that mknet.py folds into the following
                   op's lead/trail pair.

  the tail         AveragePooling2D 7x7 -> a 1x1 CONV_2D classifier -> softmax, which is
                   MobileNetV1's own tail. The fully-connected layer is expressible as a
                   1x1 convolution and this keeps FULLY_CONNECTED and MEAN out of the
                   op set the gate has to lower.

WHAT IT ASSERTS, because a quantizer flag that silently stops working would produce a
model this gate reads WRONG rather than one it refuses: that every weight carries
exactly one scale (per-tensor, not per-axis), and that the input quantization is the
byte identity. `--per-axis` inverts the first of those — it asserts that every
convolution came out per-axis — and writes a separate file.

WHY THE --per-axis VARIANT EXISTS. It is the same architecture, the same weights and the
same calibration set, so the quantization GRANULARITY is the only variable, and the top-1
label is an assertion a classifier carries end to end. A per-axis defect in the delegate
or the library would otherwise show only as a detector's mAP loss, which has no other
symptom to read. It is not a gate blob — mknet.py reads the first scale only, so this file
is for a frontend that consumes the .tflite directly.

The .tflite is not committed. `./fetch.sh && .venv-build/bin/python mkresnet18.py`
rebuilds it; `mknet.py r18` then turns it into the .rnet blob the gate runs.
"""
import json, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "resnet18_a1_in1k.safetensors")
OUT = os.path.join(HERE, "resnet18_224_quant.tflite")
OUT_PER_AXIS = os.path.join(HERE, "resnet18_224_peraxis.tflite")

# timm/resnet18.a1_in1k is trained with the standard ImageNet statistics.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)
BN_EPS = 1e-5

# The calibration set. Real photographs from the same public corpus the gate's test
# image comes from — min/max calibration takes the extremes over the set, so synthetic
# extremes would widen every activation range and cost accuracy everywhere.
CALIB = ["bird.bmp", "cat.bmp", "dragonfly.bmp", "grace_hopper.bmp", "hot_dog.jpg",
         "owl.jpg", "parrot.jpg", "pets.jpg", "sunflower.bmp", "squat.bmp",
         "face.jpg", "kite_and_cold.jpg", "cat_720p.jpg", "bird_segmentation.bmp",
         "dog_segmentation.bmp", "missvickie_potato_chips.bmp"]


def read_safetensors(path):
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        head = json.loads(f.read(n))
        blob = f.read()
    dt = {"F32": np.float32, "I64": np.int64}
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        lo, hi = v["data_offsets"]
        out[k] = np.frombuffer(blob[lo:hi], dtype=dt[v["dtype"]]).reshape(v["shape"])
    return out


def fold_bn(W, g, b, mu, var):
    """Conv-then-BatchNorm as one convolution with a bias. W is [OC][IC][KH][KW]."""
    s = g / np.sqrt(var + BN_EPS)
    return W * s[:, None, None, None], b - mu * s


def build_keras(P):
    import tensorflow as tf
    from tensorflow import keras
    L = keras.layers

    def conv(x, W, b, stride, pad, name):
        """torchvision's Conv2d: an explicit symmetric pad and a VALID convolution."""
        if pad:
            x = L.ZeroPadding2D(pad, name=name + "_pad")(x)
        oc, ic, kh, kw = W.shape
        c = L.Conv2D(oc, (kh, kw), strides=stride, padding="valid", use_bias=True,
                     name=name)
        y = c(x)
        c.set_weights([np.ascontiguousarray(W.transpose(2, 3, 1, 0)),
                       np.ascontiguousarray(b)])
        return y

    def cbr(x, prefix, stride, pad, name, relu=True):
        W, b = fold_bn(P[prefix + ".weight"].astype(np.float64),
                       P[prefix.replace("conv", "bn") + ".weight"].astype(np.float64),
                       np.zeros(P[prefix + ".weight"].shape[0]),
                       P[prefix.replace("conv", "bn") + ".running_mean"].astype(np.float64),
                       P[prefix.replace("conv", "bn") + ".running_var"].astype(np.float64))
        bnb = P[prefix.replace("conv", "bn") + ".bias"].astype(np.float64)
        y = conv(x, W.astype(np.float32), (b + bnb).astype(np.float32), stride, pad, name)
        return L.ReLU(name=name + "_relu")(y) if relu else y

    # batch_size=1, so the tail's reshape is a constant. Left dynamic, Keras emits the
    # shape arithmetic as SHAPE/STRIDED_SLICE/PACK ops in the converted graph.
    inp = keras.Input(shape=(224, 224, 3), batch_size=1, dtype="float32", name="image")

    # conv1, with the ImageNet normalization folded in on top of the batch norm, so the
    # network's input is the raw pixel value.
    W1, b1 = fold_bn(P["conv1.weight"].astype(np.float64),
                     P["bn1.weight"].astype(np.float64),
                     np.zeros(64), P["bn1.running_mean"].astype(np.float64),
                     P["bn1.running_var"].astype(np.float64))
    b1 = b1 + P["bn1.bias"].astype(np.float64)
    a = 1.0 / (255.0 * STD)                       # x_norm[c] = a[c]*x[c] - t[c]
    t = MEAN / STD
    b1 = b1 - (W1 * t[None, :, None, None]).sum(axis=(1, 2, 3))
    W1 = W1 * a[None, :, None, None]
    x = conv(inp, W1.astype(np.float32), b1.astype(np.float32), 2, 3, "conv1")
    x = L.ReLU(name="conv1_relu")(x)
    x = L.ZeroPadding2D(1, name="pool1_pad")(x)
    x = L.MaxPooling2D((3, 3), strides=2, padding="valid", name="pool1")(x)

    for gi, (chan, stride) in enumerate([(64, 1), (128, 2), (256, 2), (512, 2)]):
        for bi in range(2):
            pre = "layer%d.%d." % (gi + 1, bi)
            s = stride if bi == 0 else 1
            y = cbr(x, pre + "conv1", s, 1, pre.replace(".", "_") + "conv1")
            y = cbr(y, pre + "conv2", 1, 1, pre.replace(".", "_") + "conv2", relu=False)
            if s != 1 or x.shape[-1] != chan:
                W, b = fold_bn(P[pre + "downsample.0.weight"].astype(np.float64),
                               P[pre + "downsample.1.weight"].astype(np.float64),
                               np.zeros(chan),
                               P[pre + "downsample.1.running_mean"].astype(np.float64),
                               P[pre + "downsample.1.running_var"].astype(np.float64))
                b = b + P[pre + "downsample.1.bias"].astype(np.float64)
                sc = conv(x, W.astype(np.float32), b.astype(np.float32), s, 0,
                          pre.replace(".", "_") + "down")
            else:
                sc = x
            x = L.Add(name=pre.replace(".", "_") + "add")([y, sc])
            x = L.ReLU(name=pre.replace(".", "_") + "relu")(x)

    x = L.AveragePooling2D((7, 7), name="avgpool")(x)
    fcw = P["fc.weight"].astype(np.float64)[:, :, None, None]     # [1000][512][1][1]
    x = conv(x, fcw.astype(np.float32), P["fc.bias"].astype(np.float32), 1, 0, "fc")
    x = L.Reshape((1000,), name="flatten")(x)
    x = L.Softmax(name="probs")(x)
    return keras.Model(inp, x, name="resnet18")


def calib_images():
    from PIL import Image
    out = []
    for n in CALIB:
        p = os.path.join(HERE, "calib", n)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32)
        out.append(a)
        out.append(a[:, ::-1, :].copy())
    if not out:
        sys.exit("no calibration images under %s/calib — run ./fetch.sh" % HERE)
    return out


def main():
    import tensorflow as tf
    per_axis = "--per-axis" in sys.argv[1:]
    out = OUT_PER_AXIS if per_axis else OUT
    if not os.path.exists(WEIGHTS):
        sys.exit("%s is not fetched — run ./fetch.sh" % os.path.basename(WEIGHTS))
    P = read_safetensors(WEIGHTS)
    model = build_keras(P)
    imgs = calib_images()
    print("%d calibration samples, pixel range [%g, %g]"
          % (len(imgs), min(a.min() for a in imgs), max(a.max() for a in imgs)))

    def rep():
        for a in imgs:
            yield [a[None, ...]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    conv._experimental_disable_per_channel = not per_axis
    buf = conv.convert()
    with open(out, "wb") as f:
        f.write(buf)

    # ---- the two assertions this file exists to make -----------------------------
    import tflite
    m = tflite.Model.GetRootAsModel(buf, 0)
    g = m.Subgraphs(0)
    codes = [m.OperatorCodes(i).BuiltinCode() for i in range(m.OperatorCodesLength())]
    names = {v: k for k, v in vars(tflite.BuiltinOperator).items()
             if isinstance(v, int)}
    seen, peraxis, pertensor, spread = {}, 0, 0, 1.0
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        n = names[codes[op.OpcodeIndex()]]
        seen[n] = seen.get(n, 0) + 1
        if n in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            q = g.Tensors(op.Inputs(1)).Quantization()
            if q.ScaleLength() != 1:
                peraxis += 1
                s = np.array([q.Scale(j) for j in range(q.ScaleLength())])
                spread = max(spread, float(s.max() / s.min()))
            else:
                pertensor += 1
    print("ops: " + ", ".join("%s x%d" % kv for kv in sorted(seen.items())))
    if per_axis:
        if pertensor:
            sys.exit("%d convolution(s) came out PER-TENSOR under --per-axis; the "
                     "quantizer no longer honours the flag" % pertensor)
        print("%d convolutions per-axis, widest per-layer scale spread %.1fx"
              % (peraxis, spread))
    elif peraxis:
        sys.exit("%d convolution(s) came out PER-AXIS; the per-tensor quantizer flag "
                 "no longer works and mknet.py would read only the first scale" % peraxis)

    q = g.Tensors(int(g.Inputs(0))).Quantization()
    isc, izp = float(q.Scale(0)), int(q.ZeroPoint(0))
    print("input quantization: scale %.9g, zero point %d" % (isc, izp))
    if abs(isc - 1.0) > 1e-6 or izp != -128:
        sys.exit("the input quantization is not the byte identity (scale 1.0, zp -128); "
                 "the calibration set does not span [0, 255]")
    print("wrote %s, %d bytes" % (out, len(buf)))


if __name__ == "__main__":
    main()
