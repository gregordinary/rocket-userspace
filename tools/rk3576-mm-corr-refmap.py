#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""What can tests/rk3576_mm_corr.c's reference surface actually score?

A correctness probe reports "wrong elements". That count is not the set of elements the
device lost: it is that set intersected with the set of positions where the value a lost
element reads back differs from the reference by more than the probe's tolerance. Call
that second set the SCORING MASK. If the mask's density sits near one half, a plain
rectangular loss reads back as a half-filled bounding box and invites a geometric story
it does not support.

This computes the mask with no device, replicating the probe's generators and reference
arithmetic exactly, and reports the mask's density per candidate value a lost element
might read back: a zeroed BO, the library's 0xA5 device stamp, the probe's own 0x5A host
sentinel, or the previous repetition's correct value.

  python3 tools/rk3576-mm-corr-refmap.py 2048 4096
"""
import sys

# Exactly tests/rk3576_mm_corr.c.
gen_a = lambda i: (i * 7 + 3) % 17 - 8
gen_b = lambda i: (i * 5 + 1) % 13 - 6


def ref_at(m, n, K):
    acc = sum(gen_a(m * K + k) * gen_b(n * K + k) for k in range(K))
    f = acc / 512.0
    want = int(f - 0.5) if f < 0 else int(f + 0.5)
    return max(-128, min(127, want))


def scores_wrong(got, want):     # the probe's own test: |got - want| <= 1 passes
    return not (want - 1 <= got <= want + 1)


def report(K):
    tab = [[ref_at(i, j, K) for j in range(13)] for i in range(17)]

    # Verify the claimed period rather than assume it.
    viol = sum(1 for m in range(17 * 4) for n in range(13 * 4)
               if ref_at(m, n, K) != tab[m % 17][n % 13])
    cells = [v for row in tab for v in row]

    print(f"K {K}: K mod 221 = {K % 221} surviving contraction terms")
    print(f"  period check: {viol} violations over {17*4} x {13*4} positions (17 x 13)")
    print(f"  reference: {len(set(cells))} distinct values over 221 residue cells, "
          f"range {min(cells)}..{max(cells)}, "
          f"{sum(1 for v in cells if -1 <= v <= 1)} of 221 within +-1 of zero, "
          f"{sum(1 for v in cells if v in (127, -128))} saturated")
    for name, got in (("zeroed BO (0)", 0),
                      ("device stamp 0xA5 (-91)", -91),
                      ("host sentinel 0x5A (90)", 90)):
        w = sum(1 for v in cells if scores_wrong(got, v))
        print(f"  mask density, lost element reads {name:<24}: {w:3d} of 221 = {w/221:.4f}")
    print(f"  mask density, lost element reads {'stale (previous rep)':<24}:   0 of 221 = 0.0000")
    hist = {}
    for v in cells:
        b = min(abs(v), 7)
        hist[b] = hist.get(b, 0) + 1
    print("  abs(ref) histogram: " +
          "  ".join(f"{'>=' if b == 7 else ''}{b}:{hist.get(b, 0)}" for b in range(8)))


if __name__ == "__main__":
    for K in ([int(a) for a in sys.argv[1:]] or [2048, 4096]):
        report(K)
