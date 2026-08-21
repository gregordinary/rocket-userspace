#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""At which K, if any, is the shared int8 matmul reference non-degenerate?

`tests/rk3576_mm_corr.c`, `tests/rk3576_submit_floor.c` and `tests/rk3576_multicore.c`
share one generator pair, A[i]=(i*7+3)%17-8 and B[i]=(i*5+1)%13-6, both periodic in the
FLATTENED index. Two consequences follow and neither is K-specific:

  - the reference depends only on (m mod 17, n mod 13), so the whole output takes at most
    221 distinct values however large M and N are;
  - each 221 consecutive k covers every residue pair exactly once (CRT) and both
    generators are zero-mean over a full period, so every whole 221-term block of the
    contraction sums to exactly zero. Only K mod 221 terms survive, and the reference is
    therefore a function of K mod 221 alone.

That makes the whole K axis enumerable: 221 values covers every K there is. This prints
the map — for each residue, the reference's range over all 221 cells and how many cells
fall outside the probes' +-1 tolerance band around zero.
"""
gen_a = lambda i: (i * 7 + 3) % 17 - 8
gen_b = lambda i: (i * 5 + 1) % 13 - 6


def table(K):
    """The 17 x 13 reference table at contraction depth K."""
    out = []
    for m in range(17):
        a = [gen_a(m * K + k) for k in range(K)]
        for n in range(13):
            acc = sum(a[k] * gen_b(n * K + k) for k in range(K))
            f = acc / 512.0
            want = int(f - 0.5) if f < 0 else int(f + 0.5)
            out.append(max(-128, min(127, want)))
    return out


# The claim that acc depends only on K mod 221, checked rather than assumed.
for K, r in ((2048, 2048 % 221), (4096, 4096 % 221), (1024, 1024 % 221)):
    print(f"K {K} vs K {r}: tables {'identical' if table(K) == table(r) else 'DIFFER'}")

worst, hist = 0, {}
rows = []
for r in range(221):
    t = table(r)
    span = max(abs(v) for v in t)
    outside = sum(1 for v in t if not -1 <= v <= 1)
    worst = max(worst, span)
    hist[span] = hist.get(span, 0) + 1
    if outside:
        rows.append((r, span, outside))

print(f"\nover all 221 residues of K: max abs(ref) anywhere = {worst}")
print(f"residues whose reference leaves the +-1 band at any cell: {len(rows)} of 221")
for r, span, outside in rows[:20]:
    print(f"  K mod 221 = {r:3d}: max abs(ref) {span}, {outside} of 221 cells outside +-1")
print("max abs(ref) histogram over residues: " +
      "  ".join(f"{k}:{v}" for k, v in sorted(hist.items())))
