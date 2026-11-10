#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 The rocket-userspace authors
"""What register BLOCKS does an .rknn capture drive, and which registers in each?

mkgolden.py only scans for runs of the six block targets a convolution uses, so a
program that drives a block it does not name is invisible — it breaks the run and is
dropped without a word. That is exactly the wrong instrument for asking whether a
capture reached a block nobody has driven yet (the DPU's LUT bank, the PPU).

This scans for ANY 16-bit target in the known set, prints every run with its block
histogram, and lists the distinct registers each block received. Usage:

    dumpblocks.py <file.rknn> [more ...]
"""
import sys, collections

from regcmd_scan import ALL_TARGETS, BLOCK_NAMES as BLOCKS, regcmd_programs


def runs(path, min_run=12):
    return regcmd_programs(path, ALL_TARGETS, min_run)


def main(paths):
    for p in paths:
        rs = runs(p)
        print("== %s: %d run(s)" % (p, len(rs)))
        for k, run in enumerate(rs):
            hist = collections.Counter(BLOCKS[t] for t, _, _ in run)
            print("  run %d: %d writes  %s" % (
                k, len(run), " ".join("%s=%d" % kv for kv in sorted(hist.items()))))
            per = collections.defaultdict(list)
            for t, r, v in run:
                per[BLOCKS[t]].append((r, v))
            for b in sorted(per):
                regs = per[b]
                print("    %-9s %s" % (b, " ".join(
                    "%04x=%08x" % rv for rv in regs)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
