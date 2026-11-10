#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 The rocket-userspace authors
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
import os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regcmd_scan import ALL_TARGETS, BLOCK_NAMES as BLOCKS, regcmd_programs  # noqa: E402

DPU = 0x1001
LUT_ACCESS_CFG, LUT_ACCESS_DATA, LUT_CFG, LUT_INFO = 0x4100, 0x4104, 0x4108, 0x410c
LUT_LE_START, LUT_LE_END = 0x4110, 0x4114
LUT_LO_START, LUT_LO_END = 0x4118, 0x411c
LUT_LE_SLOPE_SCALE, LUT_LE_SLOPE_SHIFT = 0x4120, 0x4124
LUT_LO_SLOPE_SCALE, LUT_LO_SLOPE_SHIFT = 0x4128, 0x412c


def ops(path):
    return regcmd_programs(path, ALL_TARGETS, 12)


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
