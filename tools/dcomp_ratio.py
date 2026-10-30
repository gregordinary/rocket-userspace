#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""
dcomp_ratio.py — how far DCOMP's dense lossless codec would shrink a GGUF's matmul weights.

Host-only. It sizes the codec's stream for int8 weights without encoding it: 16 lanes per
N32xK32 int8 block (lane = (n%2)*8 + (k%32)/4 in the (N/32,K/32,32,32) tile layout), each lane
a 128-bit header and then one symbol a weight, a prefix code ranked by width frequency
(lengths 2,2,3,3,3,4,5,5), a sign bit and w-1 magnitude bits of the residual from the lane's
modal value. A lane occupies max(8, ceil(bits/128)) 16-byte blocks, padded to 64 bytes. The
sizes match rknpu2_matmul_open's encoder byte for byte (five shapes, 2026-10-21). The format
is described in rockchip-npu-notes/encodings/cna-dcomp-weight-decompression.md.

The weights are each tensor dequantized and quantized again per output channel, symmetric
round-to-nearest with the scale at the channel's absmax over `limit` levels a side (127 is
plain int8, smaller limits are lossy). A Q8_0 tensor is also sized on its own codes.

usage: PYTHONPATH=<llama.cpp>/gguf-py dcomp_ratio.py MODEL.gguf [layers] [limits]
       e.g. dcomp_ratio.py Llama-3.2-3B-Instruct-F16.gguf 3 127,63,31,15
"""
import sys
import types
import numpy as np
try:
    import yaml  # noqa: F401  (gguf's metadata module wants it; the reader does not)
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")
from gguf import GGUFReader, GGMLQuantizationType as QT
from gguf.quants import dequantize

PREFIX = np.array([2, 2, 3, 3, 3, 4, 5, 5])


def lane_bits(hist):
    """hist: (16,256) counts of byte values per lane -> total stream bytes."""
    total = 0
    v = np.arange(256)
    for lane in range(16):
        h = hist[lane]
        if h.sum() == 0:
            continue
        modal = int(np.argmax(h))
        r = (v - modal) & 0xFF
        mag = np.where(r < 128, r, 255 - r)
        w = np.zeros(256, dtype=np.int64)
        nz = mag > 0
        w[nz] = np.floor(np.log2(mag[nz])).astype(np.int64) + 1
        whist = np.bincount(w, weights=h, minlength=8)[:8]
        rank = np.argsort(-whist, kind="stable")  # most frequent width first
        plen = np.empty(8, dtype=np.int64)
        plen[rank] = PREFIX
        bits = 128 + sum(whist[i] * (plen[i] + 1 + max(i - 1, 0)) for i in range(8))
        blocks = max(8, int(-(-bits // 128)))
        total += -(-blocks * 16 // 64) * 64
    return total


def lanes_of(n, k):
    return ((np.arange(n)[:, None] % 2) * 8 + (np.arange(k)[None, :] % 32) // 4)


def quant_rows(wf, limit):
    amax = np.abs(wf).max(axis=1, keepdims=True)
    s = np.where(amax == 0, 1.0, amax / limit)
    return np.clip(np.rint(wf / s), -limit, limit).astype(np.int8)


def main():
    path = sys.argv[1]
    max_t = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    limits = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [127]
    rd = GGUFReader(path)
    picks = []
    for t in rd.tensors:
        if len(t.shape) != 2 or not t.name.endswith(".weight") or "blk." not in t.name:
            continue
        k, n = int(t.shape[0]), int(t.shape[1])
        if k % 32 or n % 32:
            continue
        picks.append(t)
    # every matmul tensor of max_t layers spread over the depth
    layers = sorted({int(t.name.split(".")[1]) for t in picks})
    keep = set(layers[:: max(1, len(layers) // max_t)][:max_t])
    picks = [t for t in picks if int(t.name.split(".")[1]) in keep]
    agg = {l: [0, 0] for l in limits}
    aggq8 = [0, 0]
    for t in picks:
        k, n = int(t.shape[0]), int(t.shape[1])
        lanes = lanes_of(n, k)
        row = [t.name, f"{n}x{k}", t.tensor_type.name]
        if t.tensor_type == QT.Q8_0:
            # the file's own int8 codes, scales per 32 along K
            raw = np.asarray(t.data).reshape(-1, 34)
            q = raw[:, 2:].view(np.int8).reshape(n, k)
            hist = np.bincount((lanes * 256 + q.view(np.uint8)).ravel(),
                               minlength=4096).reshape(16, 256)
            c = lane_bits(hist)
            aggq8[0] += n * k
            aggq8[1] += c
            row.append(f"q8_0codes {n*k/c:.3f}x")
        wf = dequantize(t.data, t.tensor_type).reshape(n, k).astype(np.float32)
        for lim in limits:
            q = quant_rows(wf, lim)
            hist = np.bincount((lanes * 256 + q.view(np.uint8)).ravel(),
                               minlength=4096).reshape(16, 256)
            c = lane_bits(hist)
            agg[lim][0] += n * k
            agg[lim][1] += c
            row.append(f"L{lim} {n*k/c:.3f}x {8*c/(n*k):.2f}bpw")
        print("  ".join(row), flush=True)
    print("TOTAL", path.rsplit("/", 1)[-1])
    if aggq8[0]:
        print(f"  Q8_0 codes: {aggq8[0]/aggq8[1]:.3f}x  {8*aggq8[1]/aggq8[0]:.2f} bpw")
    for lim in limits:
        r, c = agg[lim]
        print(f"  per-channel RTN limit {lim}: {r/c:.3f}x  {8*c/r:.2f} bpw")


if __name__ == "__main__":
    main()
