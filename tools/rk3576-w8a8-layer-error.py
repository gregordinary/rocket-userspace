#!/usr/bin/env python3
"""
w8a8_layer_error.py -- what one per-tensor int8 scale does to a REAL model's own
prefill activations, measured against the fp32 reference. Host only, no device.

The entry under test is rocket_matmul_int8_rk3576(), whose whole contract is

    C[m][n] = sat8( round( ( sum_k A[m][k]*B[n][k] + bias[n] ) * scale ) )

with A and B pre-quantized int8 by the caller and ONE per-tensor positive `scale`.
Every arm below is that expression with one thing changed, so the columns are
magnitudes in one unit rather than verdicts:

  fp16            the floor -- what the CPU path this would replace already loses
  q8in_i32out     per-tensor int8 A and B, EXACT int32 out            (input term)
  exactin_q8out   exact float GEMM, per-tensor int8 output            (output term)
  SHIPPED         both -- the entry as specified
  rowA_i32out     per-row A scale (not expressible in one call; M-blocking reaches it)
  chanB_i32out    per-channel B scale (the DPU's per-oc C ramp reaches this)
  rowA_chanB      both, exact int32 out
  chanB_chanQout  per-channel B scale AND a per-output-column int8 output scale
                  -- the requant this part's conv path already programs
  hadamard        Kronecker Hadamard rotation along K, then per-tensor int8, i32 out

Every scale is an ORACLE: computed from the tensor it quantizes, at its own max.
That is deliberately the entry's best case -- a real port must estimate them.

Rounding is round-half-to-even (numpy rint), which is what this DPU's requant does.
The reference is float64; the int32 accumulation is exact in float64 (|acc| <=
K*127*127 = 33e6 < 2^53).
"""
import os, sys, json, math
import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("THREADS", "16")))

MODEL = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-1.7B")
NTOK = int(os.environ.get("NTOK", "512"))
KREFUSE = 6176          # the entry declines K >= 6176
OUT = os.environ.get("OUT", "layer_error.json")


# ---- the quantizer the entry's caller has to be -------------------------------
def q8(x, axis=None):
    """Symmetric per-tensor (axis=None) or per-axis int8 quantization, oracle scale."""
    amax = np.abs(x).max() if axis is None else np.abs(x).max(axis=axis, keepdims=True)
    amax = np.maximum(amax, 1e-30)
    s = amax / 127.0
    q = np.clip(np.rint(x / s), -128, 127)
    return q, s


def relerr(hat, ref):
    return float(np.linalg.norm(hat - ref) / np.linalg.norm(ref))


def sqnr_db(hat, ref):
    n = np.linalg.norm(hat - ref)
    return float(20 * math.log10(np.linalg.norm(ref) / n)) if n > 0 else float("inf")


def cos(hat, ref):
    a, b = hat.ravel(), ref.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def sat8_requant(acc, scale):
    """The DPU epilogue: one float multiply, round half to even, saturate to int8."""
    return np.clip(np.rint(acc * scale), -128, 127)


# ---- the Hadamard rotation, K = 2^a * 3^b ... ---------------------------------
def hadamard_or_none(K):
    """Orthonormal H for K a power of two times 1, else None (the arm skips)."""
    if K & (K - 1):
        return None
    H = np.array([[1.0]])
    while H.shape[0] < K:
        H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(K)


# ---- one GEMM, every arm ------------------------------------------------------
def score(A, W, name, layer, tag, rows):
    """A is [M,K] float64 activations, W is [N,K] float64 weights."""
    M, K = A.shape
    N = W.shape[0]
    ref = A @ W.T

    r = {"model": tag, "layer": layer, "proj": name, "M": M, "K": K, "N": N,
         "offloadable": bool(K % 32 == 0 and N % 32 == 0 and K < KREFUSE),
         "A_outlier": float(np.abs(A).max() / np.sqrt((A * A).mean())),
         "W_outlier": float(np.abs(W).max() / np.sqrt((W * W).mean())),
         "C_outlier": float(np.abs(ref).max() / np.sqrt((ref * ref).mean()))}

    def put(arm, hat):
        r[arm] = {"rel": relerr(hat, ref), "sqnr": sqnr_db(hat, ref), "cos": cos(hat, ref)}

    put("fp16", (torch.from_numpy(A).half() @ torch.from_numpy(W).half().T)
        .float().numpy().astype(np.float64))

    Aq, sa = q8(A)
    Wq, sb = q8(W)
    acc = Aq @ Wq.T                       # exact int32, held in float64
    put("q8in_i32out", acc * sa * sb)

    step = np.abs(ref).max() / 127.0      # exact in, int8 out, per-tensor
    put("exactin_q8out", np.clip(np.rint(ref / step), -128, 127) * step)

    sc = 127.0 / max(np.abs(acc).max(), 1e-30)          # the shipped entry
    put("SHIPPED", sat8_requant(acc, sc) * (sa * sb) / sc)

    Ar, sar = q8(A, axis=1)
    put("rowA_i32out", (Ar @ Wq.T) * sar * sb)

    Wc, sbc = q8(W, axis=1)
    put("chanB_i32out", (Aq @ Wc.T) * sa * sbc.T)

    put("rowA_chanB", (Ar @ Wc.T) * sar * sbc.T)

    # What the SHIPPED entry actually reaches. Only the OUTPUT scale is per-tensor:
    # A and B are quantized by the caller, so per-row A and per-channel B cost the
    # host nothing and are dequantized on the way out. One call, one output scale.
    acc_c = Ar @ Wc.T
    scr = 127.0 / max(np.abs(acc_c).max(), 1e-30)
    put("REACHABLE", sat8_requant(acc_c, scr) / scr * sar * sbc.T)

    # The same, with the call split so each block carries its own output scale.
    # M has no constraint on this part, so a row block is just another call; an N
    # block is a call over a column range. Both cost submits, not accuracy machinery.
    for blk, axis, nm2 in ((64, 0, "REACHABLE_m64"), (512, 1, "REACHABLE_n512")):
        hat = np.empty_like(acc_c)
        lim = acc_c.shape[axis]
        for i in range(0, lim, blk):
            sl = (slice(i, min(i + blk, lim)), slice(None)) if axis == 0 else \
                 (slice(None), slice(i, min(i + blk, lim)))
            b = acc_c[sl]
            s = 127.0 / max(np.abs(b).max(), 1e-30)
            hat[sl] = sat8_requant(b, s) / s
        put(nm2, hat * sar * sbc.T)

    # per-channel B scale plus a per-output-COLUMN output scale: the requant this
    # part's conv epilogue already programs (per-oc C ramp, one MUL/SHIFT per task).
    scn = 127.0 / np.maximum(np.abs(acc_c).max(axis=0, keepdims=True), 1e-30)
    put("rowAchanB_chanQout", sat8_requant(acc_c, scn) / scn * sar * sbc.T)

    H = hadamard_or_none(K)
    if H is not None:
        Ah, sah = q8(A @ H)
        Wh, sbh = q8(W @ H)
        put("hadamard", (Ah @ Wh.T) * sah * sbh)

    rows.append(r)
    print(f"  {tag} L{layer:02d} {name:5s} M{M} K{K} N{N} "
          f"Aout={r['A_outlier']:6.1f} Cout={r['C_outlier']:6.1f} "
          f"fp16={r['fp16']['rel']:.2e} in={r['q8in_i32out']['rel']:.4f} "
          f"out={r['exactin_q8out']['rel']:.4f} SHIPPED={r['SHIPPED']['rel']:.4f} "
          f"rowA={r['rowA_i32out']['rel']:.4f} chanB={r['chanB_i32out']['rel']:.4f} "
          f"both={r['rowA_chanB']['rel']:.4f} REACH={r['REACHABLE']['rel']:.4f} "
          f"m64={r['REACHABLE_m64']['rel']:.4f} n512={r['REACHABLE_n512']['rel']:.4f} "
          f"route={r['rowAchanB_chanQout']['rel']:.4f}"
          + (f" had={r['hadamard']['rel']:.4f}" if H is not None else ""), flush=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pyarrow.parquet as pq

    tag = MODEL.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    text = "\n\n".join(pq.read_table("wikitext2-test.parquet")["text"].to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[:, :NTOK]
    print(f"{tag}: {ids.shape[1]} real tokens of wikitext-2", flush=True)

    rows = []
    hooks = []
    layers = model.model.layers
    only = os.environ.get("LAYERS")
    want = set(int(x) for x in only.split(",")) if only else set(range(len(layers)))

    def mk(li, nm):
        def hook(mod, inp, _out):
            if li not in want:
                return
            A = inp[0].detach().reshape(-1, inp[0].shape[-1]).double().numpy()
            W = mod.weight.detach().double().numpy()
            score(A, W, nm, li, tag, rows)
        return hook

    for li, L in enumerate(layers):
        for nm, mod in (("q", L.self_attn.q_proj), ("k", L.self_attn.k_proj),
                        ("v", L.self_attn.v_proj), ("o", L.self_attn.o_proj),
                        ("gate", L.mlp.gate_proj), ("up", L.mlp.up_proj),
                        ("down", L.mlp.down_proj)):
            hooks.append(mod.register_forward_hook(mk(li, nm)))

    model(ids)
    for h in hooks:
        h.remove()

    with open(OUT, "w") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} rows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
