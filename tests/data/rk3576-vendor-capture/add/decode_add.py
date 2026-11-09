#!/usr/bin/env python3
"""Decode the RK3576's elementwise-ADD program out of the manufactured captures.

`dumpblocks.py ../add/*.rknn` says the shape of it: an elementwise add is a program of
89 writes driving DPU and DPU_RDMA only -- no CNA, no CORE -- the same shape as the LUT
table load. This lines those programs up against a convolution's DPU/DPU_RDMA half and
prints only the registers that MOVE, which is what names the fields.

    python3 decode_add.py            # every comparison
    python3 decode_add.py <a> <b>    # two tags
"""
import sys, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from regcmd_scan import ALL_TARGETS, BLOCK_NAMES, regcmd_programs  # noqa: E402


def runs(path, min_run=12):
    """The programs in `path`, each command's target given by its block name."""
    return [[(BLOCK_NAMES[t], r, v) for t, r, v in prog]
            for prog in regcmd_programs(path, ALL_TARGETS, min_run)]


def regs(run, blocks=("DPU", "DPU_RDMA")):
    """Last write wins, as the hardware sees it."""
    d = {}
    for b, r, v in run:
        if b in blocks:
            d[(b, r)] = v
    return d


def kind(run):
    h = collections.Counter(b for b, _, _ in run)
    if h.get("CNA"):
        return "conv"
    if h.get("PPU"):
        return "pool"
    if h.get("DPU") and not h.get("CNA"):
        return "dpu-only"
    return "other"


def load(tag):
    return runs(os.path.join(HERE, tag + ".rknn"))


def first(tag, want):
    for r in load(tag):
        if kind(r) == want:
            return r
    return None


def diff(name_a, a, name_b, b, skip=()):
    ka = regs(a)
    kb = regs(b)
    keys = sorted(set(ka) | set(kb))
    print("== %s  vs  %s" % (name_a, name_b))
    n = 0
    for k in keys:
        if k[1] in skip:
            continue
        va, vb = ka.get(k), kb.get(k)
        if va != vb:
            n += 1
            f = lambda v: "--------" if v is None else "%08x" % v
            print("   %-8s %04x   %s -> %s" % (k[0], k[1], f(va), f(vb)))
    if not n:
        print("   (identical)")
    print()


def main(argv):
    if len(argv) == 2:
        diff(argv[0], first(argv[0], "dpu-only") or first(argv[0], "conv"),
             argv[1], first(argv[1], "dpu-only") or first(argv[1], "conv"))
        return

    print("### what each capture emitted\n")
    for tag in sorted(t[:-5] for t in os.listdir(HERE) if t.endswith(".rknn")):
        h = collections.Counter(kind(r) for r in load(tag))
        print("  %-18s %s" % (tag, dict(h)))
    print()

    # 1. The add program against a convolution's DPU half: what an add sets that a
    #    conv does not, and what it leaves alone.
    diff("ctl_conv (conv program)", first("ctl_conv_c32_16", "conv"),
         "bare_add (dpu-only)", first("bare_add_c32_16", "dpu-only"))

    # 2. Add against Mul at identical wiring: isolates the ALU opcode.
    diff("addconst add", first("addconst_c32_16", "dpu-only"),
         "mul", first("mul_c32_16", "dpu-only"))

    # 3. A constant operand against two live tensors: isolates the operand source.
    diff("addconst (const operand)", first("addconst_c32_16", "dpu-only"),
         "bare_add (two live)", first("bare_add_c32_16", "dpu-only"))

    # 4. The MobileNetV2 bottleneck's own add, at two channel counts and planes.
    diff("btl_c32_16 add", first("btl_c32_16", "dpu-only"),
         "btl_c24_28 add", first("btl_c24_28", "dpu-only"))


if __name__ == "__main__":
    main(sys.argv[1:])
