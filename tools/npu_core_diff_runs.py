#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""Build and read the lane-0 experiments run through tests/npu_core_diff_probe's raw mode.

Every run starts from head 12's flash-attention AV operands (tests/fa_av_dump_wrap.c's
fa_av_a.f16 / fa_av_b.f16, P 512x1024 and V 256x1024), K pass 0, and the dot product the FA
replay flips: a0 = P[256][0:384], b0 = V[240][0:384]. Its K group 1 (k 32-63) is the group that
decides the lane-0 event.

Layouts (M 512, K 384, N 256, the probe's raw shape):
  full   the whole A and B given (controls: a0/b0 everywhere; the original operands)
  diag   variant v at (v, v): its a row in A row v, its b row in B row v (s1)
  lane   16 variants a call: variant j's b row on all 16 output lanes of column block j
         (n = 16j + c), its a row on every row m with m % 16 == j (s2, s11)
  arow   B = one row on every column, each A row one variant, so every variant is on every
         lane 16 times (512 variants a call; only changes to a are expressible) (s3-s6)
  grid   group 1 only (A's other groups 0x0000): a call's 512 A rows are a-bases and its 16
         lane-0 columns (n = 16 j, repeated over 16j..16j+15) b-bases, 8192 lane-0 groups a call,
         every one under the call's permutation tau of group 1's 32 positions (t1, t2, t4 S)
Each built run gets an index.json naming the variant at every position (t1-t4: meta.json and
.npy side files instead).

The t runs map the positions (the transposition map):
  t1  499 calls: the identity (calls 0, 250, 498) and the 496 transpositions of group 1's
      positions, over 512 a0-family a-bases x 16 b-bases (t1_bases())
  t2  2048 calls: 4 replicates x 16 blocks x (the identity, 31 swaps); each block's 16 columns a
      t1 b-base under a random permutation pi, rows 32j..32j+31 a-bases that fired with it on one
      core in t1 (FIRE, [3, 512, 16], from the 'fire' command); a replicate covers the 496 swaps once
  t3  512 calls: lane 49 (position 17) swapped into position q, a[q] swept over the 8192 words
      with exponent field 0-3, for every q
  t4  44 calls: a0 with n0 lanes' features set to -0 (arow, 12 calls); sparse groups, the frame
      lane plus k = 1-4 random lanes and -0 elsewhere (grid, 32 calls)

usage:
  npu_core_diff_runs.py build RUN OPSDIR OUTDIR [FIRE]
      RUN in s1c s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 t1 t2 t3 t4 (t2 needs FIRE)
  npu_core_diff_runs.py labels INDIR OUTDIR NCALLS DEST.npz
      Labels every lane-0 element of a group-1-only run (grid, arow, t3) against the base model:
      lab [NCALLS, 512, 16] is 1 for one event (the group lowered by 2^(4 Cmax - 44)), 0 for the
      model, -1 for neither; also counts columns 16j+c (c > 0) off the model.
  npu_core_diff_runs.py fire LAB0 LAB1 LAB2 DEST.npy
      t1's identity labels (call 0) of cores 0, 1, 2 as t2's FIRE table.
  npu_core_diff_runs.py outcome OUTDIR MIS1 MIS2 MIS4
      MISk: tools/npu_core_diff_score.c's output for the core pinned by RKNPU_CORE_MASK=k.
      Prints, per variant, 'E' (all of its lane-0 elements are an event), '.' (none) or a
      count, per core, for lane and arow runs.
An event is an element exactly 2^-16 below the base model; the scorer lists every element
that differs, and this reader counts only lane-0 ones and reports anything else.
"""
import itertools, json, os, re, struct, sys, collections
import numpy as np

M, K, N = 512, 384, 256


def load_ops(d):
    P = np.fromfile(os.path.join(d, 'fa_av_a.f16'), np.uint16).reshape(512, 1024)
    V = np.fromfile(os.path.join(d, 'fa_av_b.f16'), np.uint16).reshape(256, 1024)
    return P, V


def put(root, t, A, B):
    os.makedirs(root, exist_ok=True)
    A.astype(np.uint16).tofile(os.path.join(root, 'raw_%03d.a16' % t))
    B.astype(np.uint16).tofile(os.path.join(root, 'raw_%03d.b16' % t))


def diag_layout(root, var, a0, b0, per_tile=240):
    idx, nt = [], (len(var) + per_tile - 1) // per_tile
    for t in range(nt):
        A = np.tile(a0, (M, 1)); B = np.zeros((N, K), np.uint16)
        chunk = var[t * per_tile:(t + 1) * per_tile]
        for v, (lab, a, b) in enumerate(chunk):
            A[v] = a; B[v] = b; idx.append((t, v, lab))
        for v in range(len(chunk), N):
            B[v] = b0; idx.append((t, v, 'ctrl'))
        put(root, t, A, B)
    return {'layout': 'diag', 'ntiles': nt, 'idx': idx}


def lane_layout(root, var, a0, b0, P, V):
    idx, nt = [], (len(var) + 15) // 16
    for t in range(nt):
        chunk = var[t * 16:(t + 1) * 16]
        chunk = chunk + [('ctrl', a0, b0)] * (16 - len(chunk))
        A = np.zeros((M, K), np.uint16); B = np.zeros((N, K), np.uint16)
        for j, (lab, a, b) in enumerate(chunk):
            B[16 * j:16 * j + 16] = b; A[j::16] = a; idx.append((t, j, lab))
        put(root, t, A, B)
    put(root, nt, np.tile(a0, (M, 1)), np.tile(b0, (N, 1)))
    put(root, nt + 1, P, V)
    return {'layout': 'lane', 'ntiles': nt, 'ncalls': nt + 2, 'idx': idx}


def arow_layout(root, var, b, pad):
    idx, nt = [], (len(var) + M - 1) // M
    for t in range(nt):
        A = np.tile(pad, (M, 1))
        for r, (lab, a) in enumerate(var[t * M:(t + 1) * M]):
            A[r] = a; idx.append((t, r, lab))
        put(root, t, A, np.tile(b, (N, 1)))
    return {'layout': 'arow', 'ntiles': nt, 'idx': idx}


def f16_scale(h, j):
    f = np.float32(np.array(h, np.uint16).view(np.float16)) * np.float32(2.0 ** j)
    y = np.float16(f)
    if np.float32(y) != f or not np.isfinite(y) or (y == 0 and f != 0):
        return None
    return int(np.array(y).view(np.uint16))


def coarse_form(h):
    """(E, M) of an fp16 word in the coarse form NVDLA's CMAC uses and the lane-0 events
    follow [expected]: E = e >> 2, M = significand << (e & 3); a subnormal is e = 1 without
    the hidden bit. value = M * 2^(4E - 25)."""
    h = int(h); e, m = (h >> 10) & 31, h & 1023
    if e == 0: return (0, m << 1)
    return (e >> 2, (1024 | m) << (e & 3))


def sigmas():
    out = [('rot%d' % r, [(i + r) % 32 for i in range(32)]) for r in range(1, 32)]
    out += [('hrot%d' % r, [(i & 16) | ((i + r) & 15) for i in range(32)]) for r in range(1, 16)]
    out += [('qrot%d' % r, [(i & 24) | ((i + r) & 7) for i in range(32)]) for r in range(1, 8)]
    for perm in itertools.permutations(range(5)):
        out.append(('bits' + ''.join(map(str, perm)),
                    [sum(((i >> b) & 1) << perm[b] for b in range(5)) for i in range(32)]))
    return out


def s6_values():
    return np.random.default_rng(20260927).choice(np.arange(1, 4096), size=256, replace=False)


def rnd_sig(rng, w):
    """w with its 10-bit significand random, sign and exponent field kept, never +-0."""
    w = int(w)
    while True:
        m = int(rng.integers(0, 1024))
        if ((w >> 10) & 31) or m:
            return (w & 0xfc00) | m


def t1_bases(a0, b0):
    """512 a-bases (a0 with one or two random significands, base2 = a0 with a[47] = 0x0228 with one,
    the core-1 trigger a[36] = 0x0ec6, a[47] = 0x0228 with one; each family's first row its trigger)
    and 16 b-bases (b0, five with b[60]'s significand random, ten with one other lane's)."""
    rng = np.random.default_rng(20260927)
    A0, B0 = a0[32:64].copy(), b0[32:64].copy()
    base2 = A0.copy(); base2[47 - 32] = 0x0228
    k1 = A0.copy(); k1[36 - 32] = 0x0ec6; k1[47 - 32] = 0x0228
    ab, fam = [], []
    for r in range(512):
        if r < 256:
            a = A0.copy(); nl = 1 if r < 128 else 2; f = 'K2-%d' % nl
        elif r < 384:
            a = base2.copy(); nl = 1; f = 'K0'
        else:
            a = k1.copy(); nl = 1; f = 'K1'
        if r % 128 == 0:
            nl = 0
        for p in rng.choice(32, size=nl, replace=False):
            a[p] = rnd_sig(rng, a[p])
        ab.append(a); fam.append(f)
    bb, bl = [B0.copy()], ['b0']
    for c in range(1, 16):
        b = B0.copy()
        p = 60 - 32 if c <= 5 else int(rng.choice([q for q in range(32) if q != 60 - 32]))
        b[p] = rnd_sig(rng, b[p]); bb.append(b); bl.append('b sig %d' % (p + 32))
    return np.array(ab, np.uint16), np.array(bb, np.uint16), fam, bl


def grid_put(root, t, arows, bcols, b0):
    """arows 512 x 32 and bcols 16 x 32 (group 1, already permuted); A's other groups 0x0000."""
    A = np.zeros((M, K), np.uint16); A[:, 32:64] = arows
    B = np.tile(b0, (N, 1))
    for j in range(16):
        B[16 * j:16 * j + 16, 32:64] = bcols[j]
    put(root, t, A, B)


def build_t(run, out, a0, b0, fire=None):
    os.makedirs(out, exist_ok=True)
    pairs = [(i, j) for i in range(32) for j in range(i + 1, 32)]
    if run == 't1':
        ab, bb, fam, bl = t1_bases(a0, b0)
        taus, labels, it = [], [], iter(pairs)
        for t in range(499):
            s = list(range(32))
            if t in (0, 250, 498):
                labels.append('ident')
            else:
                i, j = next(it); s[i], s[j] = j, i; labels.append('swap %d %d' % (i, j))
            taus.append(s)
            grid_put(out, t, ab[:, s], bb[:, s], b0)
        np.save(os.path.join(out, 'abase.npy'), ab); np.save(os.path.join(out, 'bbase.npy'), bb)
        np.save(os.path.join(out, 'taus.npy'), np.array(taus, np.int16))
        info = {'run': run, 'layout': 'grid', 'ncalls': 499, 'labels': labels, 'afam': fam, 'blab': bl}
    elif run == 't2':
        ab, bb, _, _ = t1_bases(a0, b0)
        rng = np.random.default_rng(20260928)
        cores = [2, 0, 1]; P_, JB, CO, RW, calls = [], [], [], [], []
        t = 0
        for rep in range(4):
            order = rng.permutation(len(pairs))
            for blk in range(16):
                k = len(P_)
                pis = np.array([rng.permutation(32) for _ in range(16)])
                jbs, cs, rows = [], [], np.zeros(512, np.int64)
                for j in range(16):
                    c = cores[j % 3]
                    jb = int(rng.choice(np.where(fire[c].sum(0) >= 32)[0]))
                    rows[32 * j:32 * j + 32] = rng.choice(np.where(fire[c][:, jb] == 1)[0], size=32, replace=False)
                    jbs.append(jb); cs.append(c)
                P_.append(pis); JB.append(jbs); CO.append(cs); RW.append(rows)
                for sw in [None] + [pairs[q] for q in order[31 * blk:31 * blk + 31]]:
                    tau = np.arange(32)
                    if sw:
                        tau[sw[0]], tau[sw[1]] = sw[1], sw[0]
                    arows = np.zeros((512, 32), np.uint16)
                    for j in range(16):
                        arows[32 * j:32 * j + 32] = ab[rows[32 * j:32 * j + 32]][:, pis[j][tau]]
                    grid_put(out, t, arows, np.array([bb[jbs[j]][pis[j][tau]] for j in range(16)]), b0)
                    calls.append((k, -1, -1) if sw is None else (k, sw[0], sw[1])); t += 1
        np.savez(os.path.join(out, 't2meta.npz'), pi=np.array(P_), jb=np.array(JB), core=np.array(CO),
                 rows=np.array(RW), calls=np.array(calls))
        info = {'run': run, 'layout': 'blocks', 'ncalls': t}
    elif run == 't3':
        vals = np.array([(s << 15) | (e << 10) | f for s in (0, 1) for e in range(4) for f in range(1024)], np.uint16)
        t = 0
        for q in range(32):
            tau = np.arange(32); tau[17], tau[q] = q, 17
            ag = a0[32:64][tau]
            for r in range(16):
                arows = np.tile(ag, (512, 1)); arows[:, q] = vals[512 * r:512 * r + 512]
                grid_put(out, t, arows, np.tile(b0[32:64][tau], (16, 1)), b0); t += 1
        np.save(os.path.join(out, 'vals.npy'), vals)
        info = {'run': run, 'layout': 'sweep', 'ncalls': t}
    elif run == 't4':
        rng = np.random.default_rng(20260929)
        ag, bg = a0[32:64].copy(), b0[32:64].copy()
        t = 0; info = {'run': run, 'D': [], 'S': []}
        for n0 in (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 30):
            arows, sets = np.zeros((512, 32), np.uint16), []
            for r in range(512):
                a = ag.copy(); z = rng.choice([p for p in range(32) if p != 28], size=n0, replace=False)
                a[z] = 0x8000; arows[r] = a; sets.append(sorted(int(x) for x in z))
            grid_put(out, t, arows, np.tile(bg, (16, 1)), b0); info['D'].append({'call': t, 'n0': n0, 'zero': sets}); t += 1
        def word(lo, hi):
            while True:
                w = (int(rng.integers(0, 2)) << 15) | (int(rng.integers(lo, hi + 1)) << 10) | int(rng.integers(0, 1024))
                if w & 0x7fff:
                    return w
        for k in (1, 2, 3, 4):
            for rep in range(8):
                pos = sorted(int(x) for x in rng.choice([p for p in range(32) if p != 28], size=k, replace=False))
                base = np.full(32, 0x8000, np.uint16); base[28] = ag[28]
                arows = np.tile(base, (512, 1)); bcols = np.tile(bg, (16, 1))
                for r in range(512):
                    for p in pos:
                        arows[r, p] = word(0, 7)
                for j in range(16):
                    for p in pos:
                        bcols[j, p] = word(7, 16)
                grid_put(out, t, arows, bcols, b0)
                info['S'].append({'call': t, 'k': k, 'pos': pos}); t += 1
        info['ncalls'] = t
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(info, f)
    print(run, info['ncalls'], 'calls')


def lane_vals(a, b):
    """Exact coarse-form products at scale 2^-48 (int64) and each product's coarse exponent sum C
    (-1 when either word is +-0), for uint16 arrays a, b broadcast over (..., 32)."""
    a = a.astype(np.int64); b = b.astype(np.int64)
    ea, eb = (a >> 10) & 31, (b >> 10) & 31
    Ma = np.where(ea == 0, (a & 1023) << 1, ((a & 1023) | 1024) << (ea & 3))
    Mb = np.where(eb == 0, (b & 1023) << 1, ((b & 1023) | 1024) << (eb & 3))
    C = (np.maximum(ea, 1) >> 2) + (np.maximum(eb, 1) >> 2)
    sh = 4 * C - 2
    V = np.where(sh >= 0, (Ma * Mb) << np.maximum(sh, 0), (Ma * Mb) >> np.maximum(-sh, 0))
    V = np.where(((a ^ b) >> 15) & 1, -V, V)
    nz = ((a & 0x7fff) != 0) & ((b & 0x7fff) != 0)
    return V, np.where(nz, C, -1)


def f32bits(v):
    assert (np.abs(v) < (1 << 53)).all()
    return (v.astype(np.float64) * 2.0 ** -48).astype(np.float32).view(np.uint32)


def labels(ind, outd, nc, dest):
    lab = np.zeros((nc, 512, 16), np.int8); off = 0
    for t in range(nc):
        A = np.fromfile(os.path.join(ind, 'raw_%03d.a16' % t), np.uint16).reshape(M, K)
        B = np.fromfile(os.path.join(ind, 'raw_%03d.b16' % t), np.uint16).reshape(N, K)
        if (A[:, :32] != 0).any() or (A[:, 64:] != 0).any():
            sys.exit('call %d: A is not group 1 only' % t)
        C = np.fromfile(os.path.join(outd, 'raw_%03d.f32' % t), np.uint32).reshape(M, 16, 16)
        V, Cs = lane_vals(A[:, None, 32:64], B[None, 0::16, 32:64])
        S = V.sum(2); cm = Cs.max(2)
        mod = f32bits(S); ev = f32bits(S - (np.int64(1) << (4 * np.maximum(cm, 0) + 4)))
        L = np.full((M, 16), -1, np.int8)
        L[C[:, :, 0] == mod] = 0; L[(C[:, :, 0] == ev) & (ev != mod)] = 1
        lab[t] = L
        off += int((C[:, :, 1:] != mod[:, :, None]).sum())
    np.savez_compressed(dest, lab=lab)
    print(dest, 'events', int((lab == 1).sum()), 'neither', int((lab == -1).sum()), 'off-lane-0 off the model', off)


def build(run, ops, out, extra=None):
    P, V = load_ops(ops)
    P, V = P[:, :K].copy(), V[:, :K].copy()
    a0, b0 = P[256].copy(), V[240].copy()
    if run == 's1c':        # identity everywhere; the original operands
        put(out, 0, np.tile(a0, (M, 1)), np.tile(b0, (N, 1))); put(out, 1, P, V)
        info = {'layout': 'full', 'ncalls': 2}
    elif run in ('s1', 's2'):
        var = []
        for g in (range(12) if run == 's1' else (1,)):
            for i in range(32):
                for j in range(i + 1, 32):
                    a = a0.copy(); b = b0.copy(); ki, kj = g * 32 + i, g * 32 + j
                    a[ki], a[kj] = a0[kj], a0[ki]; b[ki], b[kj] = b0[kj], b0[ki]
                    var.append(('swap %d %d %d' % (g, i, j), a, b))
        info = diag_layout(out, var, a0, b0) if run == 's1' else lane_layout(out, var, a0, b0, P, V)
    elif run in ('s3', 's4', 's5', 's6'):
        var = [('ident', a0.copy()) for _ in range(8)]
        if run == 's3':
            for k in range(K):
                a = a0.copy(); a[k] = 0; var.append(('ko %d' % k, a))
            for k in range(K):
                a = a0.copy(); a[k] ^= 0x8000; var.append(('neg %d' % k, a))
            for g in range(12):
                a = np.zeros_like(a0); a[g * 32:(g + 1) * 32] = a0[g * 32:(g + 1) * 32]
                var.append(('only %d' % g, a))
                a = a0.copy(); a[g * 32:(g + 1) * 32] = 0; var.append(('zero %d' % g, a))
        elif run == 's4':
            for k in range(32, 64):
                for nm, v in (('z', 0x0000), ('nz', 0x8000), ('t', 0x0001), ('nt', 0x8001)):
                    a = a0.copy(); a[k] = v; var.append(('%s %d' % (nm, k), a))
            for s in (0, 1):
                for e in range(4):
                    for f in range(1024):
                        a = a0.copy(); a[49] = (s << 15) | (e << 10) | f
                        var.append(('sw49 %04x' % a[49], a))
        elif run == 's5':
            for tag, k, v in (('p34', 34, 0x0000), ('p60', 60, 0x0000), ('m34', 34, 0x8000)):
                for s in (0, 1):
                    for e in range(2):
                        for f in range(1024):
                            a = a0.copy(); a[k] = v; a[49] = (s << 15) | (e << 10) | f
                            var.append(('%s %04x' % (tag, a[49]), a))
        else:
            for k in range(32, 64):
                for v in s6_values():
                    a = a0.copy(); a[k] = int(v); var.append(('sw%d %04x' % (k, v), a))
        info = arow_layout(out, var, b0, a0)
    elif run == 's7':       # core 2 only: s6's lane-40/47 variants under each lane permutation
        var = [('sw%d %04x' % (k, v), k, int(v)) for k in (40, 47) for v in s6_values()]
        sg = sigmas()
        for t, (name, s) in enumerate(sg):
            s = np.array(s)
            A = np.tile(a0, (M, 1))
            for r, (_, k, v) in enumerate(var): A[r, k] = v
            B = np.tile(b0, (N, 1))
            A[:, 32:64] = A[:, 32:64][:, s]; B[:, 32:64] = B[:, 32:64][:, s]
            put(out, t, A, B)
        info = {'layout': 's7', 'sigmas': sg, 'vars': [l for l, _, _ in var]}
    elif run in ('s8', 's9', 's10'):
        rng = np.random.default_rng(8)
        base2 = a0.copy(); base2[47] = 0x0228
        rows = []
        for _ in range(512):
            a = a0.copy()
            for k in rng.choice(np.arange(32, 64), size=2, replace=False):
                a[61 if k == 60 else k] = int(rng.integers(1, 4096))
            rows.append(a)
        for _ in range(512):
            a = base2.copy(); k = int(rng.choice([x for x in range(32, 64) if x not in (47, 60)]))
            a[k] = int(rng.integers(1, 4096)); rows.append(a)
        A = np.stack(rows); Bt = np.tile(b0, (N, 1))
        if run == 's8':      # tiles 0-1 unpermuted; then (core 2) the sigmas s7 tied with core 0
            put(out, 0, A[:512], Bt); put(out, 1, A[512:], Bt)
            info = {'layout': 's8', 'note': 'sigma tiles are built from s7 results; see the session'}
        elif run == 's9':    # B scaled by 2^j
            t = 0
            for j in (-2, -1, 1, 2):
                e = (Bt >> 10) & 31
                B2 = ((Bt.astype(np.int32) & ~(31 << 10)) | ((e.astype(np.int32) + j) << 10)).astype(np.uint16)
                put(out, t, A[:512], B2); put(out, t + 1, A[512:], B2); t += 2
            info = {'layout': 'full', 'arms': ['B*2^-2', 'B*2^-1', 'B*2', 'B*4'], 'ncalls': 8}
        else:                # a x 2 with b / 2 (products unchanged); a x 2 alone
            A2 = np.vectorize(lambda h: f16_scale(h, 1))(A).astype(np.uint16)
            Bh = np.vectorize(lambda h: f16_scale(h, -1))(Bt).astype(np.uint16)
            put(out, 0, A2[:512], Bh); put(out, 1, A2[512:], Bh)
            put(out, 2, A2[:512], Bt); put(out, 3, A2[512:], Bt)
            info = {'layout': 'full', 'arms': ['a*2,b/2', 'a*2,b/2', 'a*2', 'a*2'], 'ncalls': 4}
    elif run == 's11':      # one lane re-encoded with its product unchanged
        var = []
        for k in range(32, 64):
            for j in (1, -1):
                na, nb = f16_scale(a0[k], j), f16_scale(b0[k], -j)
                if na is None or nb is None: continue
                a = a0.copy(); b = b0.copy(); a[k] = na; b[k] = nb
                var.append(('re%+d %d' % (j, k), a, b))
        for j in (-3, -2, 2, 3):
            na, nb = f16_scale(a0[60], j), f16_scale(b0[60], -j)
            if na is None or nb is None: continue
            a = a0.copy(); b = b0.copy(); a[60] = na; b[60] = nb
            var.append(('re%+d %d' % (j, 60), a, b))
        info = lane_layout(out, var, a0, b0, P, V)
    elif run == 's13':      # the coarse exponent frame: s8's variants with every weight x16, /16
        rng = np.random.default_rng(8)
        base2 = a0.copy(); base2[47] = 0x0228
        rows = []
        for _ in range(512):
            a = a0.copy()
            for k in rng.choice(np.arange(32, 64), size=2, replace=False):
                a[61 if k == 60 else k] = int(rng.integers(1, 4096))
            rows.append(a)
        for _ in range(512):
            a = base2.copy(); k = int(rng.choice([x for x in range(32, 64) if x not in (47, 60)]))
            a[k] = int(rng.integers(1, 4096)); rows.append(a)
        A = np.stack(rows); Bt = np.tile(b0, (N, 1))
        t = 0
        for j in (0, 4, -4):      # exponent field + j: x1, x16, /16 (b0's fields are 7-16)
            B2 = (Bt.astype(np.int32) + (j << 10)).astype(np.uint16)
            put(out, t, A[:512], B2); put(out, t + 1, A[512:], B2); t += 2
        info = {'layout': 'full', 'arms': ['x1', 'x1', 'B*16', 'B*16', 'B/16', 'B/16'], 'ncalls': 6}
    elif run == 's14':      # one lane, or all of group 1, re-encoded (a x 16^j, b x 16^-j)
        var = []
        for k in range(32, 64):
            for j in (1, -1):
                na, nb = f16_scale(a0[k], 4 * j), f16_scale(b0[k], -4 * j)
                if na is None or nb is None: continue
                a = a0.copy(); b = b0.copy(); a[k] = na; b[k] = nb
                cl = (coarse_form(na) == (coarse_form(a0[k])[0] + j, coarse_form(a0[k])[1]) and
                      coarse_form(nb) == (coarse_form(b0[k])[0] - j, coarse_form(b0[k])[1]))
                var.append(('x16%+d %d%s' % (j, k, '' if cl else ' nc'), a, b))
        for j in (1, -1):
            for clean in (0, 1):  # clean: the coarse form (e >> 2, mantissa << (e & 3)) moves
                a = a0.copy(); b = b0.copy(); n = 0   # only its exponents (coarse_form())
                for k in range(32, 64):
                    na, nb = f16_scale(a0[k], 4 * j), f16_scale(b0[k], -4 * j)
                    if na is None or nb is None: continue
                    if clean and (coarse_form(na) != (coarse_form(a0[k])[0] + j, coarse_form(a0[k])[1])
                                  or coarse_form(nb) != (coarse_form(b0[k])[0] - j, coarse_form(b0[k])[1])):
                        continue
                    a[k] = na; b[k] = nb; n += 1
                var.append(('grp x16%+d%s (%d lanes)' % (j, ' clean' if clean else '', n), a, b))
        for j in (4, -4):         # group 1's weights alone x16 or /16 (group 1 Cmax 8 or 6)
            b = b0.copy(); b[32:64] = (b0[32:64].astype(np.int32) + (j << 10)).astype(np.uint16)
            var.append(('g1 b exp%+d' % j, a0.copy(), b))
        info = lane_layout(out, var, a0, b0, P, V)
    elif run == 's15':      # group 1 alone (every other feature 0x0000): families of random operands
        rng = np.random.default_rng(15)
        def word(sign, e, m):
            return ((sign & 1) << 15) | ((e & 31) << 10) | (m & 1023)
        def rows(n, efn, sfn):
            out = np.zeros((n, K), np.uint16)
            for r in range(n):
                for k in range(32, 64):
                    e = efn(k); m = int(rng.integers(0, 1024))
                    if e == 0 and m == 0: m = 1            # never the word 0x0000
                    out[r, k] = word(sfn(k), e, m)
            return out
        rsign = lambda k: int(rng.integers(0, 2))
        fams = [  # (name, a exponent, a sign, b exponent, b sign)
            ('fracfree', lambda k: int(rng.integers(12, 16)), rsign, lambda k: int(rng.integers(16, 20)), rsign),
            ('a0exp a0sign', lambda k: int((a0[k] >> 10) & 31), lambda k: int(a0[k] >> 15),
             lambda k: int((b0[k] >> 10) & 31), lambda k: int(b0[k] >> 15)),
            ('a0exp rsign', lambda k: int((a0[k] >> 10) & 31), rsign, lambda k: int((b0[k] >> 10) & 31), rsign),
            ('wide', lambda k: int(rng.integers(0, 16)), rsign, lambda k: int(rng.integers(9, 17)), rsign),
        ]
        t = 0
        for name, ea, sa, eb, sb in fams:
            A = rows(M, ea, sa); B1 = rows(16, eb, sb)
            B = np.repeat(B1, 16, axis=0); B[:, :32] = b0[:32]; B[:, 64:] = b0[64:]
            put(out, t, A, B); t += 1
        # softmax-like features, weights to +-3 (some coarse weight exponents 4)
        A = np.zeros((M, K), np.uint16)
        A[:, 32:64] = np.exp(-rng.uniform(0, 12, size=(M, 32))).astype(np.float16).view(np.uint16)
        A[A == 0] = 1
        A[:, :32] = 0; A[:, 64:] = 0
        B1 = rng.uniform(-3, 3, size=(16, K)).astype(np.float16).view(np.uint16); B1[B1 == 0] = 0x3c00
        put(out, t, A, np.repeat(B1, 16, axis=0)); t += 1
        put(out, t, np.tile(a0, (M, 1)), np.tile(b0, (N, 1))); t += 1   # control: fires on core 2
        info = {'layout': 'full', 'arms': [f[0] for f in fams] + ['softmax-like', 'identity'], 'ncalls': t}
    elif run == 's16':      # a0's significands randomized at a0's sign and exponent field, per lane set
        rng = np.random.default_rng(16)
        cs = [coarse_form(a0[k])[0] + coarse_form(b0[k])[0] for k in range(32, 64)]
        sh = {k: max(cs) - cs[k - 32] for k in range(32, 64)}   # coarse shift in group 1
        def rnd(lanes):
            a = a0.copy()
            for k in lanes:
                m = int(rng.integers(0, 1024))
                if ((a0[k] >> 10) & 31) == 0 and m == 0: m = 1
                a[k] = (int(a0[k]) & 0xfc00) | m
            return a
        var = [('ident', a0.copy()) for _ in range(16)]
        for k in range(32, 64):
            var += [('lane %d' % k, rnd([k])) for _ in range(40)]
        for name, sel in (('shift<=1', lambda s: s <= 1), ('shift2', lambda s: s == 2),
                          ('shift>=3', lambda s: s >= 3), ('shift>=2', lambda s: s >= 2)):
            lanes = [k for k in range(32, 64) if sel(sh[k])]
            var += [(name, rnd(lanes)) for _ in range(128)]
        info = arow_layout(out, var, b0, a0)
        info['shifts'] = sh
    elif run in ('t1', 't2', 't3', 't4'):
        return build_t(run, out, a0, b0, np.load(extra) if extra else None)
    elif run == 's12':      # real K pass 1; zero-free softmax-like synthetic
        P1, V1 = load_ops(ops)
        put(out, 0, P1[:, 384:768].copy(), V1[:, 384:768].copy())
        rng = np.random.default_rng(12)
        for t in range(1, 9):
            A = np.exp(-rng.uniform(0, 15, size=(M, K))).astype(np.float16)
            B = rng.uniform(-2, 2, size=(N, K)).astype(np.float16)
            put(out, t, A.view(np.uint16), B.view(np.uint16))
        info = {'layout': 'full', 'ncalls': 9}
    else:
        sys.exit('unknown run ' + run)
    with open(os.path.join(out, 'index.json'), 'w') as f:
        json.dump(info, f)
    print(run, json.dumps({k: v for k, v in info.items() if k in ('layout', 'ntiles', 'ncalls')}))


def fbits(h):
    return struct.unpack('<f', struct.pack('<I', int(h, 16)))[0]


def outcome(out, mis):
    info = json.load(open(os.path.join(out, 'index.json')))
    lay = info['layout']
    if lay not in ('lane', 'arow'):
        sys.exit('outcome reads lane and arow runs')
    per = []
    for path in mis:
        ev = collections.defaultdict(int)
        for line in open(path):
            t, m, n, d, mo = line.split(); t, m, n = int(t), int(m), int(n)
            if n % 16 or fbits(d) - fbits(mo) != -2.0 ** -16:
                print('NOT A LANE-0 EVENT:', path, line.strip())
                continue
            if lay == 'lane':
                if t < info['ntiles'] and m % 16 == n // 16: ev[(t, n // 16)] += 1
            else:
                ev[(t, m)] += 1
        per.append(ev)
    full = 32 if lay == 'lane' else 16
    for t, j, lab in info['idx']:
        if lab == 'ctrl': continue
        cells = []
        for ev in per:
            c = ev.get((t, j), 0)
            cells.append('E' if c == full else ('.' if c == 0 else str(c)))
        print(lab, ' '.join(cells))


if __name__ == '__main__':
    if len(sys.argv) in (5, 6) and sys.argv[1] == 'build':
        build(*sys.argv[2:])
    elif len(sys.argv) == 6 and sys.argv[1] == 'labels':
        labels(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif len(sys.argv) == 6 and sys.argv[1] == 'fire':
        np.save(sys.argv[5], np.stack([np.load(p)['lab'][0] == 1 for p in sys.argv[2:5]]).astype(np.int8))
    elif len(sys.argv) >= 4 and sys.argv[1] == 'outcome':
        outcome(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
