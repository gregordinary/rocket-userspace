#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""
sweep_rocket.py — per-shape autotune sweep for the rocket NPU matmul.

Finds the per-shape NPU tiling optimum with NO code change. The load-bearing knob
is Kt (ROCKET_MM_KT): a bigger K-tile means fewer K-passes (nKt) and proportionally
less CPU readback (read ∝ M·N·nKt), but shrinking Mt/Nt to grow Kt adds N-tiles and
DRAM reload — the trade is shape-dependent, so we measure it.

Two modes:

  DRIVER (default) — drives the standalone bench `matmul_mt_rocket M K N`, which
    itself sweeps T=1..4 and prints "T=n:  <ms> ms  <gflops> GFLOP/s" per thread
    count. We run it under a grid of ROCKET_MM_KT (and optionally MT/NT) for each
    Gemma projection shape and report the best (Kt, T) per shape. Cheap, no model
    load — run this EARLY and rerun after each tuning change. The NPU GFLOP/s here
    also feeds the NPU-vs-CPU crossover question.

  LLAMA (--llama MODEL) — the end-to-end -b/-ub sweep: runs llama-bench through the
    DL backend across n_ubatch values and ROCKET_N_THREADS, reporting real prefill
    t/s. Slow (loads the ~22GB F16 model) but it's the user-facing number.

USAGE (on the RK3588 device, as ROOT — the NPU device needs it):
    cd /path/to/rocket-userspace
    cmake --build build -j                         # ensure benches are built
    sudo python3 tests/sweep_rocket.py             # driver Kt×T sweep, all shapes
    sudo python3 tests/sweep_rocket.py --shapes ffn --verbose
    sudo python3 tests/sweep_rocket.py --kt 128,256,384,512,768,1024 --mt-nt
    sudo python3 tests/sweep_rocket.py --llama /path/to/gemma-4-12b-it-F16.gguf \
         --backend /path/to/ggml-rocket/build-dl/libggml-rocket.so

MODEL DIMS: the projection shapes are computed from DIMS below. n_embd and n_ff are
confirmed for Gemma-4-12B (3840 / 15360); the attention/vocab dims are best-guess
(n_head 16 × head_dim 256 reproduces the 4096 q-proj seen in the profile) — VERIFY
against YOUR model (llama prints n_embd/n_ff/n_head/n_head_kv/n_vocab at load) and
edit DIMS if they differ. A wrong dim just sweeps a wrong N; nothing is corrupted.
"""
import argparse, os, re, subprocess, sys

# --- model dims (EDIT to match your GGUF; n_embd/n_ff are confirmed) -------------
DIMS = dict(
    n_embd    = 3840,     # CONFIRMED: hidden size
    n_ff      = 15360,    # CONFIRMED: FFN intermediate
    n_head    = 16,       # VERIFY
    n_head_kv = 8,        # VERIFY (GQA)
    head_dim  = 256,      # VERIFY  (n_head*head_dim = 4096 == the profile's q-proj N)
    n_vocab   = 256000,   # VERIFY (Gemma 256k)
)

def gemma_shapes(d):
    """Return [(label, group, K, N)] for C[M,N] = A[M,K]·B[N,K]^T (weights [N,K])."""
    qN  = d["n_head"]    * d["head_dim"]
    kvN = d["n_head_kv"] * d["head_dim"]
    return [
        ("attn_q",   "attn", d["n_embd"], qN),
        ("attn_k",   "attn", d["n_embd"], kvN),
        ("attn_v",   "attn", d["n_embd"], kvN),
        ("attn_o",   "attn", qN,          d["n_embd"]),
        ("ffn_gate", "ffn",  d["n_embd"], d["n_ff"]),
        ("ffn_up",   "ffn",  d["n_embd"], d["n_ff"]),
        ("ffn_down", "ffn",  d["n_ff"],   d["n_embd"]),
        ("lm_head",  "head", d["n_embd"], d["n_vocab"]),
    ]

# matmul_mt_rocket prints e.g. "T=4:   125.0 ms  128.82 GFLOP/s  2.39x  verify ... PASS"
T_RE = re.compile(r"T=(\d+):\s+([\d.]+)\s+ms\s+([\d.]+)\s+GFLOP/s")

def run_driver(bench, M, K, N, env_overrides, timeout):
    """Run the bench once; return {T: (gflops, ms)}, ok, raw_output."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("ROCKET_DEBUG", "ROCKET_MM_PROFILE")}
    env.update(env_overrides)
    try:
        p = subprocess.run([bench, str(M), str(K), str(N)], env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, False, "(timeout)"
    out = p.stdout + p.stderr
    res = {int(m.group(1)): (float(m.group(3)), float(m.group(2)))
           for m in T_RE.finditer(out)}
    return res, ("FAIL" not in out and p.returncode == 0), out

def driver_sweep(args):
    bench = os.path.join(args.bench_dir, "matmul_mt_rocket")
    if not os.path.exists(bench):
        sys.exit(f"bench not found: {bench}  (build it: cmake --build {args.bench_dir} -j)")

    shapes = [s for s in gemma_shapes(DIMS)
              if args.shapes == "all" or s[1] == args.shapes]
    kt_grid  = [int(x) for x in args.kt.split(",")] if args.kt else None
    mt_grid  = [int(x) for x in args.mt.split(",")] if args.mt_nt else [0]   # 0 = default
    nt_grid  = [int(x) for x in args.nt.split(",")] if args.mt_nt else [0]
    Ms       = [int(x) for x in args.m.split(",")]

    summary = []  # (label, M, def_gf, best_gf, best_cfg)
    for label, group, K, N in shapes:
        # Kt grid defaults to a spread clamped to K (the plan re-clamps to CBUF fit).
        kts = kt_grid or [k for k in (128, 256, 384, 512, 768, 1024, 2048) if k <= K]
        for M in Ms:
            print(f"\n=== {label:8s}  C[M,{N}] = A[M,{K}] x B[{N},{K}]^T   (M={M}) ===")
            # baseline: no ROCKET_MM_* (the plan's own max-Kt choice)
            base, ok, _ = run_driver(bench, M, K, N, {}, args.timeout)
            def_gf = max((g for g, _ in base.values()), default=0.0)
            best = (def_gf, "default", best_t(base))
            hdr = "  cfg".ljust(22) + "".join(f"  T{t}".rjust(9) for t in (1,2,3,4))
            print(hdr); print("  " + "-"*(len(hdr)-2))
            print(row("default", base))
            for mt in mt_grid:
                for nt in nt_grid:
                    for kt in kts:
                        ov = {"ROCKET_MM_KT": str(kt)}
                        cfgname = f"Kt={kt}"
                        if mt: ov["ROCKET_MM_MT"] = str(mt); cfgname += f",Mt={mt}"
                        if nt: ov["ROCKET_MM_NT"] = str(nt); cfgname += f",Nt={nt}"
                        res, ok, _ = run_driver(bench, M, K, N, ov, args.timeout)
                        print(row(cfgname, res, flag="" if ok else " !FAIL"))
                        gf = max((g for g, _ in res.values()), default=0.0)
                        if gf > best[0]:
                            best = (gf, cfgname, best_t(res))
            gain = 100.0 * (best[0] - def_gf) / def_gf if def_gf else 0.0
            print(f"  -> best {best[1]} @ T{best[2]}: {best[0]:.1f} GFLOP/s "
                  f"(default {def_gf:.1f}, {gain:+.1f}%)")
            summary.append((label, M, def_gf, best[0], f"{best[1]} T{best[2]}", gain))

    print("\n" + "="*78 + "\nSUMMARY — best config per shape (driver GFLOP/s)\n" + "="*78)
    print(f"  {'shape':9s} {'M':>4s} {'default':>9s} {'best':>9s} {'gain':>7s}  config")
    for label, M, dg, bg, cfg, gain in summary:
        print(f"  {label:9s} {M:>4d} {dg:>9.1f} {bg:>9.1f} {gain:>6.1f}%  {cfg}")
    print("\nApply a per-shape Kt in the backend via ROCKET_MM_KT for a uniform run, or\n"
          "feed these into a shape-aware plan later. Re-run after each tuning change.")

def best_t(res):
    return max(res, key=lambda t: res[t][0]) if res else 0

def row(name, res, flag=""):
    cells = "".join((f"{res[t][0]:9.1f}" if t in res else f"{'-':>9}") for t in (1,2,3,4))
    return f"  {name:20s}{cells}{flag}"

# --- end-to-end llama -b/-ub sweep ----------------------------------------------
PP_RE = re.compile(r"\|\s*pp(\d+)\s*\|\s*([\d.]+)\s*±")

def llama_sweep(args):
    if not args.backend or not os.path.exists(args.backend):
        sys.exit("--llama needs --backend /path/to/libggml-rocket.so")
    bench = args.llama_bench or os.path.expanduser(
        "~/llama.cpp/build/bin/llama-bench")
    ubs   = [int(x) for x in args.ub.split(",")]
    nths  = [int(x) for x in args.nthreads.split(",")]
    print(f"llama -b/-ub sweep: model={args.llama}\n  ub={ubs} threads={nths} prompts={args.prompts}\n")
    rows = []
    for nt in nths:
        env = dict(os.environ, GGML_BACKEND_PATH=args.backend, ROCKET_N_THREADS=str(nt))
        # llama-bench sweeps -ub and -p internally in one model load.
        cmd = [bench, "-m", args.llama, "-ngl", "0", "-n", "0",
               "-p", args.prompts, "-ub", ",".join(map(str, ubs)),
               "-b", ",".join(map(str, ubs))]
        print(f"--- ROCKET_N_THREADS={nt}: {' '.join(cmd)}")
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout*20)
        except subprocess.TimeoutExpired:
            print("  (timeout)"); continue
        print(p.stdout)
        for m in PP_RE.finditer(p.stdout):
            rows.append((nt, int(m.group(1)), float(m.group(2))))
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-800:] + "\n")
    if rows:
        print("="*60 + "\nSUMMARY — prefill t/s by (threads, prompt)\n" + "="*60)
        for nt, pp, ts in rows:
            print(f"  threads={nt}  pp{pp:<5d}  {ts:.2f} t/s")

def main():
    ap = argparse.ArgumentParser(description="rocket NPU autotune sweep")
    ap.add_argument("--bench-dir", default="build", help="dir with matmul_mt_rocket (default: build)")
    ap.add_argument("--shapes", default="all", choices=["all", "attn", "ffn", "head"])
    ap.add_argument("--m", default="128,256,512", help="comma list of M (token counts)")
    ap.add_argument("--kt", default="", help="comma list of Kt to try (default: auto spread)")
    ap.add_argument("--mt-nt", action="store_true", help="also sweep a small Mt/Nt grid")
    ap.add_argument("--mt", default="128,256", help="Mt grid when --mt-nt")
    ap.add_argument("--nt", default="256", help="Nt grid when --mt-nt")
    ap.add_argument("--verbose", action="store_true", help="(reserved) full per-config tables")
    ap.add_argument("--timeout", type=int, default=300, help="per-bench timeout (s)")
    # llama mode
    ap.add_argument("--llama", default="", help="MODEL.gguf -> end-to-end -b/-ub sweep")
    ap.add_argument("--backend", default="", help="libggml-rocket.so for --llama")
    ap.add_argument("--llama-bench", default="", help="path to llama-bench (auto if unset)")
    ap.add_argument("--ub", default="128,256,512", help="n_ubatch grid for --llama")
    ap.add_argument("--nthreads", default="3,4", help="ROCKET_N_THREADS grid for --llama")
    ap.add_argument("--prompts", default="512", help="-p prompt lengths for --llama")
    args = ap.parse_args()

    if os.geteuid() != 0:
        sys.stderr.write("warning: not root — the NPU device usually needs sudo.\n")
    if args.llama:
        llama_sweep(args)
    else:
        driver_sweep(args)

if __name__ == "__main__":
    main()
