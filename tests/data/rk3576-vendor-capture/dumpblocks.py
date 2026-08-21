#!/usr/bin/env python3
"""What register BLOCKS does an .rknn capture drive, and which registers in each?

mkgolden.py only scans for runs of the six block targets a convolution uses, so a
program that drives a block it does not name is invisible — it breaks the run and is
dropped without a word. That is exactly the wrong instrument for asking whether a
capture reached a block nobody has driven yet (the DPU's LUT bank, the PPU).

This scans for ANY 16-bit target in the known set, prints every run with its block
histogram, and lists the distinct registers each block received. Usage:

    dumpblocks.py <file.rknn> [more ...]
"""
import struct, sys, collections

# Block ids as they appear in the top 16 bits of an NPUOP word: BLOCK | 0x1.
BLOCKS = {
    0x0101: "PC",
    0x0201: "CNA",
    0x0801: "CORE",
    0x1001: "DPU",
    0x2001: "DPU_RDMA",
    0x4001: "PPU",
    0x8001: "PPU_RDMA",
    0x0041: "SYNC",
    0x0081: "BCAST",
}


def words(path):
    data = open(path, "rb").read()
    n = len(data) // 8
    return struct.unpack("<%dQ" % n, data[:n * 8])


def runs(path, min_run=12):
    w = words(path)
    out, i, n = [], 0, len(w)
    while i < n:
        if ((w[i] >> 48) & 0xffff) in BLOCKS:
            j = i
            while j < n and ((w[j] >> 48) & 0xffff) in BLOCKS:
                j += 1
            if j - i >= min_run:
                out.append([(int((x >> 48) & 0xffff), int(x & 0xffff),
                             int((x >> 16) & 0xffffffff)) for x in w[i:j]])
            i = j
        else:
            i += 1
    return out


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
