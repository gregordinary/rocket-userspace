#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 The rocket-userspace authors
"""Extract the RK3576 vendor register programs from the .rknn captures into a C
header the host-only gate can compile against.

Four capture sets, all RKNN-Toolkit2 output for rk3576 with the register program
embedded verbatim, and all MANUFACTURED here from ONNX that each directory's own
mk*.py writes:

  - int8/*.rknn (int8/mkint8.py) — the quantized convolutions at planes the compiler
    splits into row windows: the packed-image first conv, a 16->128 k5 s2 direct conv,
    a k3 depthwise, and a MobileNet-shaped stem compiled as one graph.
  - dw/*.rknn (dw/mkdw.py) — depthwise at channel counts from 8 to 256, both strides
    and kernels 1/3/5/7, which the int8/ set alone confounds (every C=32 program in it
    is stride 1, every C=64 one stride 2, and every one k=3).
  - float/*.rknn (float/mkfloat.py) — the fp16 direct path.
  - argb/*.rknn (argb/mkargb.py) — the fp16 packed-image first conv.

The header is generated from them in that order:

    mkgolden.py int8 dw float argb rk3576_vendor_golden.h

Every register program in every capture is emitted, not one per file: a windowed
convolution is split into several tasks and each task is its own program, so the
later tasks are where a row-window or channel-count term is separable. Each case
carries the geometry decoded back out of its own registers, so the gate drives the
emitter from the capture instead of from a hand-written table.

Usage: mkgolden.py <dir with *.rknn> [more dirs ...] <output header>
"""
import sys, os, glob

from regcmd_scan import BLOCK_NAMES, CONV_TARGETS, regcmd_programs

# A full conv task programs ~139 registers; the vendor also emits short delta
# programs (~32 writes) that re-drive only what changed between two tasks. Only
# full programs are an emitter oracle.
FULL_RUN_MIN = 100


def runs(path, min_run=20):
    return regcmd_programs(path, CONV_TARGETS, min_run)


def label(t):
    """The block name in the header's comments, which call the DPU's read DMA RDMA."""
    return "RDMA" if t == 0x2001 else BLOCK_NAMES[t]


def decode(run):
    """Recover the conv geometry from the register program itself."""
    m = {}
    for t, r, v in run:
        m[(t, r)] = v
    g = lambda t, r: m.get((t, r))

    din, dfull, sc, ko = g(0x0201, 0x102c), g(0x0201, 0x118c), g(0x0201, 0x1028), g(0x0201, 0x1024)
    st, wo, ds0, ds1 = g(0x0201, 0x1014), g(0x0201, 0x1030), g(0x0801, 0x301c), g(0x0801, 0x3020)
    pad, cm, dsurf, cbuf = g(0x0201, 0x1080), g(0x0201, 0x100c), g(0x1001, 0x401c), g(0x0201, 0x1040)
    if None in (din, dfull, sc, ko, st, wo, ds0, ds1, pad, cm, dsurf, cbuf):
        return None

    d = {
        'dw': cm & 1,
        'iw': (din >> 16) + 1,
        'ih_task': (din & 0xffff) + 1,
        'ih_full': (dfull & 0xffff) + 1,
        'ic': (sc & 0xffff) + 1,
        'kw': ((ko >> 16) & 0xff) + 1,
        'kh': ((ko >> 24) & 0xff) + 1,
        'sx': st & 0x7,
        'sy': (st >> 3) & 0x7,
        'ow': (wo & 0xffff) + 1,
        'oh_task': (ds0 >> 16) + 1,
        'oc': ds1 + 1,
        'pad_top': pad & 0xff,
        'pad_left': (pad >> 8) & 0xff,
        'pad_bottom': (pad >> 16) & 0xff,
        'pad_right': (pad >> 24) & 0xff,
        'cbuf_f': (cbuf >> 16) & 0xfff,
        'cbuf_low': cbuf & 0xffff,
        'weight_bytes': g(0x0201, 0x101c),
        'weight_elems': g(0x0201, 0x1020),
        'wbpk': wo >> 16,
        'surface_add': g(0x1001, 0x40b8),
        'argb_pre': 1 if (cm >> 12) & 0xf else 0,
        # CNA_CONV_CON1 bit 21 is the float enable: set in every vendor float program
        # and in no integer one. The two precision fields beside it (proc at bits[9:7],
        # in at bits[6:4]) both read 2 = fp16 in the float captures and 0 at int8, so
        # the flag is what the gate needs to pick the emitter's precision.
        'is_float': (cm >> 21) & 1,
    }
    d['oh_full'] = dsurf // d['ow'] if d['ow'] else 0
    # 0x118C is NOT the full plane height. Both its halves carry iw-1, which is
    # indistinguishable from ih_full-1 on a square plane and every capture used to be
    # square. The plane height is in the DDR surface stride instead: 0x1094 is
    # iw*ih_full. (On the ARGB path 0x1094 is line_stride*ih and 0x118C is the CBUF
    # row's granule count, so that path keeps the 0x118C reading; the gate drives ARGB
    # programs with the task's own height regardless.)
    surf_full = g(0x0201, 0x1094)
    if not d['argb_pre'] and d['iw'] and surf_full:
        d['ih_full'] = surf_full // d['iw']
    # The first-conv ARGB sub-encoding, read off CNA_CONV_CON1's ARGB_IN field
    # (bits[15:12], at the RK3588's own position) rather than guessed from the
    # channel count. Two fields above are NOT the caller's geometry on this path:
    #   'ic' is 0x1028's folded count (4 lanes x kw), not the image's channels —
    #        those are in the feature DMA at 0x107C.
    #   'ih_full' is 0x118C, which on this path carries the CBUF row's granule
    #        count, not a plane height. An ARGB program carries no plane height at
    #        all: its packed image is a single surface, so every stride is the
    #        task's own.
    d['argb'] = d.pop('argb_pre')
    # A model on a plane whose size is not a power of two makes the compiler append a
    # degenerate 1x1x1 program beside the convolution — one pixel, one kernel tap, and
    # a DPU_RDMA config no conv program carries. It is byte-identical across every
    # capture that has one, it is not a convolution, and there is nothing in this
    # emitter that produces it.
    if d['iw'] == 1 and d['ih_task'] == 1 and d['ow'] == 1 and d['oh_task'] == 1:
        return None
    return d


# The iso_* models are conv2d's geometry compiled with PER-TENSOR weight
# quantization, which programs a different BS-stage arrangement (0x4044, 0x4050,
# 0x501C) from the per-channel compile. Their geometry duplicates conv2d's, so they
# are reference material for that stage rather than an oracle for a plain conv
# emitter.
EPILOGUE_SOURCES = ("iso_bias", "iso_scale", "iso_sum")


def name_for(d, seen):
    base = "%s%s_c%u_%u_k%u_s%u_%ux%u_o%ux%u" % (
        "f16" if d['is_float'] else "",
        "dw" if d['dw'] else ("argb" if d['argb'] else
                              ("epi" if d.get('epilogue') else "conv")),
        d['ic'], d['oc'], d['kh'], d['sy'], d['iw'], d['ih_task'], d['ow'], d['oh_task'])
    n, nm = 1, base
    while nm in seen:
        n += 1
        nm = "%s_v%u" % (base, n)
    seen.add(nm)
    return nm


def main():
    srcs, hdr = sys.argv[1:-1], sys.argv[-1]
    cases, seen, programs = [], set(), {}

    paths = []
    for src in srcs:
        paths += sorted(glob.glob(os.path.join(src, "*.rknn")))
    for path in paths:
        for idx, run in enumerate(runs(path)):
            if len(run) < FULL_RUN_MIN:
                continue
            d = decode(run)
            if d is None:
                continue
            key = tuple(run)
            if key in programs:
                continue                      # byte-identical program already carried
            programs[key] = True
            d['epilogue'] = 1 if any(k in os.path.basename(path) for k in EPILOGUE_SOURCES) else 0
            d['name'] = name_for(d, seen)
            d['source'] = "%s run %d" % (os.path.basename(path), idx)
            d['ops'] = run
            cases.append(d)

    L = []
    L.append("// SPDX-License-Identifier: MIT")
    L.append("// Copyright (C) 2026 The rocket-userspace authors")
    L.append("/*")
    L.append(" * rk3576_vendor_golden.h - GENERATED, do not edit by hand.")
    L.append(" *")
    L.append(" * RK3576 vendor register programs, decoded from RKNN-Toolkit2 .rknn captures")
    L.append(" * manufactured here and rebuilt by their own mk*.py: int8/ for the quantized")
    L.append(" * first, direct and depthwise convs, dw/ for depthwise, float/ for the fp16")
    L.append(" * direct path, argb/ for the fp16 first conv. Each table is one")
    L.append(" * task's complete CNA+CORE+DPU+DPU_RDMA register program, in stream order, as")
    L.append(" * (target, reg, value) triples. These are the acceptance oracle for the RK3576")
    L.append(" * regcmd emitter: what the vendor toolkit programs for a known conv geometry.")
    L.append(" *")
    L.append(" * Every task of every capture is here, deduplicated by program. A windowed")
    L.append(" * convolution is several tasks, and its later tasks carry a different output")
    L.append(" * row count against the same full plane -- which is what separates a term in")
    L.append(" * the row window from a term in the plane. Each case carries the geometry")
    L.append(" * decoded from its own registers, so the gate drives the emitter from it.")
    L.append(" *")
    L.append(" * The vendor stream carries no PC trailer - the vendor kernel fires the job")
    L.append(" * through its own task descriptor. Our emitter appends the PC_OPERATION_ENABLE")
    L.append(" * trailer `rocket` needs, so the gate compares the register program only.")
    L.append(" *")
    L.append(" * Regenerate with tests/data/rk3576-vendor-capture/mkgolden.py.")
    L.append(" */")
    L.append("#ifndef RK3576_VENDOR_GOLDEN_H")
    L.append("#define RK3576_VENDOR_GOLDEN_H")
    L.append("")
    L.append("#include <stdint.h>")
    L.append("")
    L.append("struct rk3576_golden_op { uint16_t target; uint16_t reg; uint32_t val; };")
    L.append("")
    L.append("struct rk3576_golden_case {")
    L.append("    const char *name;")
    L.append("    const char *source;")
    L.append("    const struct rk3576_golden_op *ops;")
    L.append("    unsigned n_ops;")
    L.append("    unsigned dw, argb, epilogue, is_float;")
    L.append("    unsigned ic, oc, iw, ih_task, ih_full, ow, oh_task, oh_full;")
    L.append("    unsigned kh, kw, sy, sx, pad_top, pad_left, pad_bottom, pad_right;")
    L.append("    unsigned cbuf_f, cbuf_low;      /* the vendor allocator's own choice */")
    L.append("    unsigned weight_bytes, weight_elems, wbpk, surface_add;")
    L.append("};")
    L.append("")

    for d in cases:
        L.append("/* %s: %s%s%s ic=%u oc=%u k=%ux%u s=%u in %ux%u of %u rows -> out %ux%u of %u"
                 % (d['name'], "fp16 " if d['is_float'] else "",
                    "depthwise " if d['dw'] else "", "ARGB " if d['argb'] else "",
                    d['ic'], d['oc'], d['kh'], d['kw'], d['sy'],
                    d['iw'], d['ih_task'], d['ih_full'], d['ow'], d['oh_task'], d['oh_full']))
        L.append(" * source: %s, %d entries */" % (d['source'], len(d['ops'])))
        L.append("static const struct rk3576_golden_op rk3576_golden_%s[] = {" % d['name'])
        for t, r, v in d['ops']:
            L.append("    { 0x%04x, 0x%04x, 0x%08xu },  /* %s */" % (t, r, v, label(t)))
        L.append("};")
        L.append("")

    L.append("static const struct rk3576_golden_case rk3576_golden_cases[] = {")
    for d in cases:
        L.append("    { \"%s\", \"%s\", rk3576_golden_%s, %d," % (d['name'], d['source'], d['name'], len(d['ops'])))
        L.append("      %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u }," % (
            d['dw'], d['argb'], d['epilogue'], d['is_float'],
            d['ic'], d['oc'], d['iw'], d['ih_task'], d['ih_full'],
            d['ow'], d['oh_task'], d['oh_full'], d['kh'], d['kw'], d['sy'], d['sx'],
            d['pad_top'], d['pad_left'], d['pad_bottom'], d['pad_right'],
            d['cbuf_f'], d['cbuf_low'], d['weight_bytes'], d['weight_elems'], d['wbpk'],
            d['surface_add']))
    L.append("};")
    L.append("")
    L.append("#endif /* RK3576_VENDOR_GOLDEN_H */")

    open(hdr, "w").write("\n".join(L) + "\n")
    print("wrote %s: %d cases (%d depthwise, %d ARGB, %d conv+epilogue)"
          % (hdr, len(cases), sum(c['dw'] for c in cases), sum(c['argb'] for c in cases),
             sum(c['epilogue'] for c in cases)))


if __name__ == "__main__":
    main()
