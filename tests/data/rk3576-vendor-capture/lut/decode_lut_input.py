#!/usr/bin/env python3
"""Find the LUT's INPUT map: what carries an input value onto the 512 intervals.

The output side reads (decode_lut.py): two 513-entry tables are the lower and upper
halves of one monotone curve, entries Q15 of the output, uniform in the input with a
per-function span. What is missing is the map from a value onto the table index, and the
four registers named for it -- LE_START/LE_END/LO_START/LO_END -- are 0xffffc000 in every
capture, sigmoid and tanh alike, so they are not it.

The instrument is the control, not a sweep. Every case in mklut.py is the SAME 3x3
convolution over the same weights with only the activation changed, so any register the
sigmoid case writes differently from the tanh case is a function of the activation and of
nothing else. This diffs the CONSUMING convolution program (the CNA+CORE+DPU one, not the
table-load program) across activations and against the bare-conv control, and prints every
register that is not constant.

    ../dw/venv/bin/python decode_lut_input.py
"""
import struct, sys, collections, os

BLOCKS = {0x0101: "PC", 0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU",
          0x2001: "DPU_RDMA", 0x4001: "PPU", 0x8001: "PPU_RDMA",
          0x0041: "SYNC", 0x0081: "BCAST"}
LUT_ACCESS_DATA = 0x4104


def runs(path):
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


def pick_run(path, want_cna):
    """The consuming CONVOLUTION program (want_cna) or the table-LOAD program (not).
    Returns {(block, reg): value}, last write wins, with the table-data window dropped --
    the entries are the function, and what is being asked here is what surrounds them."""
    for r in runs(path):
        kinds = set(t for t, _, _ in r)
        if (0x0201 in kinds) == want_cna:
            m = {}
            for t, reg, v in r:
                if reg == LUT_ACCESS_DATA:
                    continue
                m[(BLOCKS[t], reg)] = v
            return m
    return None


def conv_run(path):
    return pick_run(path, True)


def main(argv):
    acts = argv[1:] or ["none", "sigmoid", "tanh", "hardsigmoid", "hardswish",
                        "swish", "elu", "softplus", "mish"]
    for prec, which in (("i8", True), ("i8", False), ("f16", True), ("f16", False)):
        maps = {}
        for a in acts:
            p = "conv_%s_%s.rknn" % (a, prec)
            if not os.path.exists(p):
                continue
            m = pick_run(p, which)
            if m:
                maps[a] = m
        if len(maps) < 2:
            continue
        print("== %s %s: %s" % (prec, "CONV program" if which else "table-LOAD program",
                                ", ".join(maps)))
        keys = set()
        for m in maps.values():
            keys |= set(m)
        varying = []
        for k in sorted(keys):
            vals = [maps[a].get(k) for a in maps]
            if len(set(vals)) > 1:
                varying.append(k)
        for blk, reg in varying:
            row = "  %-8s %04x " % (blk, reg)
            row += " ".join("%s=%08x" % (a, maps[a].get((blk, reg), 0)) for a in maps)
            print(row)
        if not varying:
            print("  (no register differs across these activations)")
        print()


if __name__ == "__main__":
    main(sys.argv)
