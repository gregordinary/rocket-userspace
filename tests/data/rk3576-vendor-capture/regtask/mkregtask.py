#!/usr/bin/env python3
"""Map the vendor compiler's geometry-field widths by overflowing them on purpose.

RKNN-Toolkit2 checks every register field it emits against the field's width and prints
the ones that overflow:

    REGTASK: The bit width of field value exceeds the limit, target: v2, offset: 0x500c,
    shift = 0, limit: 0x1fff, value: 0x20cf

so a graph sized just past a field's limit makes the compiler name that field's offset,
low bit and maximum, as Rockchip holds them. This builds one-op ONNX graphs sized along
one axis at a time, compiles each for the target in a subprocess, and collects every
REGTASK line with the case that printed it.

Run rk3588 first: Mesa's registers.xml names every RK3588 field, so a printed field that
disagrees with it says the method is wrong before anything is read off the RK3576, whose
geometry registers are re-packed and undocumented.

What it cannot see: a field the compiler never overflows, because it tiles the axis,
splits the layer, or falls back to the CPU first; and what a field means. A case that
prints nothing is one of those, not a proof that no field is narrow.

Needs x86_64, CPython 3.10 and the toolkit's pins (see ../dw/mkdw.py):

    uv venv --python 3.10 venv
    uv pip install --python ./venv/bin/python rknn-toolkit2 'onnx==1.14.1' 'setuptools<81'
    ./venv/bin/python mkregtask.py rk3588 OUTDIR [case ...]

Prints one line per REGTASK hit, `case offset shift limit value`, and a summary per case.
"""
import os
import re
import subprocess
import sys

import numpy as np
import onnx
from onnx import helper, TensorProto

REGTASK = re.compile(r"REGTASK: The bit width of field value exceeds the limit, target: "
                     r"(\S+), offset: (0x[0-9a-fA-F]+), shift = (\d+), limit: "
                     r"(0x[0-9a-fA-F]+), value: (0x[0-9a-fA-F]+)")


def conv(path, C, H, W, OC, k=1, stride=1, group=1):
    """One Conv over [1, C, H, W], pad k//2, weights with no period."""
    pad = k // 2
    oh = (H + 2 * pad - k) // stride + 1
    ow = (W + 2 * pad - k) // stride + 1
    cin = C // group
    rng = np.random.RandomState(1)
    w = rng.uniform(-0.5, 0.5, (OC, cin, k, k)).astype(np.float32)
    b = rng.uniform(-0.5, 0.5, (OC,)).astype(np.float32)
    node = helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[k, k],
                            pads=[pad] * 4, strides=[stride, stride], group=group)
    g = helper.make_graph(
        [node], "conv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, H, W])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, OC, oh, ow])],
        [helper.make_tensor("w", TensorProto.FLOAT, list(w.shape), w.tobytes(), raw=True),
         helper.make_tensor("b", TensorProto.FLOAT, list(b.shape), b.tobytes(), raw=True)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)
    return (1, C, H, W)


def matmul(path, M, K, N):
    """[M, K] x [K, N] with a constant B."""
    rng = np.random.RandomState(2)
    b = rng.uniform(-0.5, 0.5, (K, N)).astype(np.float32)
    node = helper.make_node("MatMul", ["x", "b"], ["y"])
    g = helper.make_graph(
        [node], "mm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [M, N])],
        [helper.make_tensor("b", TensorProto.FLOAT, [K, N], b.tobytes(), raw=True)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)
    return (M, K)


# name: (builder, args). Each separates one axis; the comment says which field it aims at.
CASES = {
    # issue #163's layer: the DFL head's 1x1 conv on a 16-deep, 4-high, 8400-wide cube
    "issue163":    (conv, dict(C=16, H=4, W=8400, OC=16)),
    "w8191":       (conv, dict(C=16, H=4, W=8191, OC=16)),     # the control just under
    "w2100":       (conv, dict(C=16, H=4, W=2100, OC=16)),     # an 11-bit field, if not tiled
    "w4200":       (conv, dict(C=16, H=4, W=4200, OC=16)),     # a 12-bit one
    "h8400":       (conv, dict(C=16, H=8400, W=4, OC=16)),
    "h2100":       (conv, dict(C=16, H=2100, W=4, OC=16)),
    "ic8400":      (conv, dict(C=8400, H=4, W=4, OC=16)),
    "oc8400":      (conv, dict(C=16, H=4, W=4, OC=8400)),
    "k3w8400":     (conv, dict(C=16, H=4, W=8400, OC=16, k=3)),
    "s2w16400":    (conv, dict(C=16, H=4, W=16400, OC=16, stride=2)),
    "dw8400":      (conv, dict(C=8400, H=4, W=4, OC=8400, group=8400)),
    "mm_n8400":    (matmul, dict(M=4, K=64, N=8400)),
    "mm_k8400":    (matmul, dict(M=4, K=8400, N=64)),
    "mm_m8400":    (matmul, dict(M=8400, K=64, N=64)),
}


def compile_one(target, out, name):
    """Build and compile one case in THIS process; the parent reads our stdout."""
    from rknn.api import RKNN
    builder, kw = CASES[name]
    onnx_path = os.path.join(out, "%s_%s.onnx" % (target, name))
    shape = builder(onnx_path, **kw)
    npy = os.path.join(out, "%s_%s_cal.npy" % (target, name))
    np.save(npy, np.random.RandomState(0).uniform(-1, 1, shape).astype(np.float32))
    ds = os.path.join(out, "%s_%s_cal.txt" % (target, name))
    open(ds, "w").write(npy + "\n")
    r = RKNN(verbose=True)
    r.config(target_platform=target)
    if r.load_onnx(model=onnx_path) != 0:
        print("CASE-RESULT load-failed")
        return
    rc = r.build(do_quantization=True, dataset=ds)
    print("CASE-RESULT build-rc %d" % rc)
    r.release()


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--one":
        compile_one(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    target, out = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or list(CASES)
    os.makedirs(out, exist_ok=True)
    for name in names:
        p = subprocess.run([sys.executable, __file__, "--one", target, out, name],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           timeout=1800)
        log = os.path.join(out, "%s_%s.log" % (target, name))
        open(log, "w").write(p.stdout)
        hits = REGTASK.findall(p.stdout)
        seen = []
        for t, off, sh, lim, val in hits:
            key = (off.lower(), int(sh), lim.lower())
            if key not in seen:
                seen.append(key)
                print("%-10s %-10s target %s offset %s shift %2d limit %s value %s"
                      % (name, target, t, off.lower(), int(sh), lim.lower(), val.lower()))
        res = re.findall(r"CASE-RESULT (.*)", p.stdout)
        print("%-10s %-10s %d REGTASK line(s), %d distinct field(s), %s"
              % (name, target, len(hits), len(seen), res[-1] if res else "no result"))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
