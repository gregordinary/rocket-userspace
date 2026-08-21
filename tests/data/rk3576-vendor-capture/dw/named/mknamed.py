#!/usr/bin/env python3
"""Depthwise captures whose weights NAME themselves, and the decoder that reads the
weight cube's layout out of them.

A register program says how big the weight cube is and nothing about how it is laid
out, so every depthwise cube this driver ever packed was laid out by analogy with the
RK3588's — which is why depthwise never computed on this part. A capture settles it
directly: give every weight its own value and the packed cube in the .rknn names the
(channel, tap) of each slot.

The weights must be FLOAT (do_quantization=False). Per-channel weight quantization
normalizes each channel by its own maximum, which erases any naming carried in
magnitude; fp16 represents integers exactly to 2048, so a unique-integer ramp
survives the float path untouched and is findable in the file by its exact value set.

What it decodes, at C = 24, 32, 48, 64 and 128 and k = 3 and 5:

    slot(c, kh, kw) = (c/32)*32*KH*KW + (kh*KW + kw)*held + (c%32)
    held            = min(32, C - (c/32)*32)

with a weight in a SIXTEEN-BIT slot, channels grouped by 32, tap-major inside a
group in kh-outer order, and a trailing partial group packed DENSELY — its tap
stride is the channels it holds, not 32. The buffer is still sized round16(C)*KH*KW*2
bytes; the slots past the last channel are don't-care and the vendor leaves whatever
was in memory there.

That is rocket_rk3576_weight_dw() in the driver. Run this script with no argument to
rebuild the captures, or `--decode` to re-derive the layout from the ones committed
here.

Dependency pins are mkdw.py's.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

CASES = [
    # tag             C    k  hw
    ("dwf_c24_k3",   24,  3, 16),   # a trailing partial group, packed dense
    ("dwf_c48_k3",   48,  3, 16),   # a full group plus a partial one
    ("dwf_c64_k3",   64,  3, 16),   # two full groups
    ("dwf_c32_k5",   32,  5, 16),   # the tap axis at k=5
]

W_GROUP = 32


def names(C, k):
    """Unique per (channel, tap). Zero is replaced — a zero slot is not a name."""
    n = C * k * k
    v = np.arange(n, dtype=np.float32) - n // 2
    v[v == 0] = n
    return v


def slot(C, KH, KW, c, kh, kw):
    g = c // W_GROUP
    held = min(W_GROUP, C - g * W_GROUP)
    return g * W_GROUP * KH * KW + (kh * KW + kw) * held + (c % W_GROUP)


def find_cube(path, C, k):
    """The cube is the one window whose fp16 value SET is exactly the name set."""
    n = C * k * k
    want = set(float(v) if v != 0 else float(n) for v in (np.arange(n) - n // 2))
    d = open(path, "rb").read()
    h = np.frombuffer(d[:len(d) // 2 * 2], dtype=np.float16)
    for off in range(0, len(h) - n):
        seg = h[off:off + n]
        if not np.all(np.isfinite(seg)):
            continue
        if set(float(x) for x in seg) == want:
            return off, seg
    return None, None


def decode():
    bad = 0
    for tag, C, k, hw in CASES:
        path = os.path.join(HERE, tag + ".rknn")
        n = C * k * k
        off, seg = find_cube(path, C, k)
        if seg is None:
            print("%-14s cube NOT FOUND" % tag); bad += 1; continue
        lin = [(n // 2 if float(x) == float(n) else int(float(x)) + n // 2) for x in seg]
        ok = all(lin[slot(C, k, k, c, kh, kw)] == c * k * k + (kh * k + kw)
                 for c in range(C) for kh in range(k) for kw in range(k))
        print("%-14s C=%3d k=%d  cube at halfword %d  layout %s"
              % (tag, C, k, off, "REPRODUCED" if ok else "MISMATCH"))
        bad += 0 if ok else 1
    print("\n%s" % ("all cubes reproduce the layout" if not bad
                    else "%d cube(s) do not" % bad))
    return bad


def build():
    import mkdw
    for tag, C, k, hw in CASES:
        mkdw.OUT = HERE
        mkdw.emit(tag, C, k, 1, hw, quant=False, wfn=names)


if __name__ == "__main__":
    sys.exit(decode() and 1 or 0) if "--decode" in sys.argv else build()
