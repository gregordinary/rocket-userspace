#!/usr/bin/env python3
"""Decode the DPU LUT programming protocol out of the manufactured activation captures.

The captures beside this script (mklut.py) are the first programs anyone has of this
part that drive the LUT at all. What they show:

  - a nonlinear activation is NOT fused into the convolution. It is its own program,
    DPU + DPU_RDMA only, with no CNA and no CORE — the DPU reads a cube through its
    own RDMA, runs it through the table and writes it back.
  - the table is written through a two-register window: LUT_ACCESS_CFG (0x4100) selects
    which table and where to start, and then every write to LUT_ACCESS_DATA (0x4104)
    stores one entry and auto-increments.

This prints, per capture, the LUT_CFG word, each (table select -> entry count) burst,
and the entries, so the encoding can be read rather than guessed. --csv dumps the
entries for fitting against the mathematical function.

    ../dw/venv/bin/python decode_lut.py *.rknn
"""
import struct, sys, collections

DPU = 0x1001
LUT_ACCESS_CFG, LUT_ACCESS_DATA, LUT_CFG, LUT_INFO = 0x4100, 0x4104, 0x4108, 0x410c
LUT_LE_START, LUT_LE_END = 0x4110, 0x4114
LUT_LO_START, LUT_LO_END = 0x4118, 0x411c
LUT_LE_SLOPE_SCALE, LUT_LE_SLOPE_SHIFT = 0x4120, 0x4124
LUT_LO_SLOPE_SCALE, LUT_LO_SLOPE_SHIFT = 0x4128, 0x412c

BLOCKS = {0x0101: "PC", 0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU",
          0x2001: "DPU_RDMA", 0x4001: "PPU", 0x8001: "PPU_RDMA",
          0x0041: "SYNC", 0x0081: "BCAST"}


def ops(path):
    data = open(path, "rb").read()
    n = len(data) // 8
    w = struct.unpack("<%dQ" % n, data[:n * 8])
    out, i = [], 0
    while i < n:
        if ((w[i] >> 48) & 0xffff) in BLOCKS:
            j = i
            while j < n and ((w[j] >> 48) & 0xffff) in BLOCKS:
                j += 1
            if j - i >= 12:
                out.append([(int((x >> 48) & 0xffff), int(x & 0xffff),
                             int((x >> 16) & 0xffffffff)) for x in w[i:j]])
            i = j
        else:
            i += 1
    return out


def bursts(run):
    """[(cfg_word, [entries...])] in program order, plus the surrounding LUT config."""
    out, cur, cfg = [], None, None
    other = {}
    for t, r, v in run:
        if t != DPU:
            continue
        if r == LUT_ACCESS_CFG:
            if cur is not None:
                out.append((cfg, cur))
            cfg, cur = v, []
        elif r == LUT_ACCESS_DATA:
            if cur is not None:
                cur.append(v)
        elif r in (LUT_CFG, LUT_INFO, LUT_LE_START, LUT_LE_END, LUT_LO_START,
                   LUT_LO_END, LUT_LE_SLOPE_SCALE, LUT_LE_SLOPE_SHIFT,
                   LUT_LO_SLOPE_SCALE, LUT_LO_SLOPE_SHIFT):
            other.setdefault(r, []).append(v)
    if cur:
        out.append((cfg, cur))
    # A burst with no data is the zeroing preamble's single write, not a table.
    return [b for b in out if b[1]], other


def summarize(path, csv=False):
    lines = []
    for k, run in enumerate(ops(path)):
        bs, other = bursts(run)
        if not bs:
            continue
        hist = collections.Counter(BLOCKS[t] for t, _, _ in run)
        lines.append("  run %d (%s): %d table burst(s)" % (
            k, " ".join("%s=%d" % kv for kv in sorted(hist.items())), len(bs)))
        for r in sorted(other):
            vals = [v for v in other[r] if v]
            if vals:
                lines.append("    %04x = %s" % (r, " ".join("%08x" % v for v in vals)))
        for cfg, ent in bs:
            lo, hi = min(ent), max(ent)
            lines.append("    cfg=%08x  %d entries  [%#x .. %#x]  first=%s  last=%s"
                         % (cfg, len(ent), lo, hi,
                            " ".join("%#x" % e for e in ent[:4]),
                            " ".join("%#x" % e for e in ent[-4:])))
            if csv:
                lines.append("    CSV %08x %s" % (cfg, ",".join(str(e) for e in ent)))
    if lines:
        print("== %s" % path)
        print("\n".join(lines))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv = "--csv" in sys.argv
    if not args:
        sys.exit(__doc__)
    for p in args:
        summarize(p, csv)
