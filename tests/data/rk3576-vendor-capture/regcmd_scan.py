#!/usr/bin/env python3
"""Find the register programs embedded in a compiled .rknn model.

RKNN-Toolkit2 stores each task's register program verbatim, as an array of 64-bit
little-endian register commands: the block target in bits 63:48, the value in 47:16 and
the register offset in 15:0. Nothing in the file marks where an array starts, so a
program is found by what it looks like: a stretch of 8-byte-aligned words whose top
halfword is a block target. Weights and other data rarely produce many such words in a
row, so a length floor separates programs from chance matches.

The scripts in this directory import it; on its own it lists the programs in a file:

    regcmd_scan.py <file.rknn> [min_len]
"""
import itertools
import struct
import sys

# Block targets as they appear in a command's top halfword.
BLOCK_NAMES = {
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

# The targets a convolution program writes, and nothing else.
CONV_TARGETS = frozenset((0x0201, 0x0801, 0x1001, 0x2001, 0x0041, 0x0081))

# Every block target named above.
ALL_TARGETS = frozenset(BLOCK_NAMES)


def regcmd_programs(path, targets, min_len):
    """Every maximal stretch of at least `min_len` commands whose target is in
    `targets`, in file order, each a list of (target, register, value) tuples. A word
    with any other top halfword ends the stretch; a trailing partial word is ignored."""
    with open(path, "rb") as f:
        blob = f.read()
    blob = blob[:len(blob) - len(blob) % 8]
    words = (w for (w,) in struct.iter_unpack("<Q", blob))
    found = []
    for is_cmd, group in itertools.groupby(words, key=lambda w: (w >> 48) in targets):
        if not is_cmd:
            continue
        prog = [(w >> 48, w & 0xFFFF, (w >> 16) & 0xFFFFFFFF) for w in group]
        if len(prog) >= min_len:
            found.append(prog)
    return found


def main(argv):
    if not argv:
        sys.exit(__doc__)
    min_len = int(argv[1]) if len(argv) > 1 else 12
    for i, prog in enumerate(regcmd_programs(argv[0], ALL_TARGETS, min_len)):
        print("program %d: %d commands" % (i, len(prog)))
        for t, r, v in prog:
            print("  %-8s %04x = %08x" % (BLOCK_NAMES[t], r, v))


if __name__ == "__main__":
    main(sys.argv[1:])
