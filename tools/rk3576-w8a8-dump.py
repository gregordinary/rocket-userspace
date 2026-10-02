#!/usr/bin/env python3
"""
rk3576-w8a8-dump.py -- a real model's W8A8 GEMM operands, as the RK3576 route hands them to
the per-column entry, for tests/rk3576_mm_percol_err.c. Host only.

For each selected layer and projection it captures the projection's input on the first
WIN-token window of wikitext-2 (the fp32 model's own activations), rotates activation and
weight by the same orthonormal Hadamard along K that rk3576-w8a8-ppl.py uses (block-diagonal
over the largest power of two dividing K), quantizes the activation PER TENSOR and the weight
PER OUTPUT CHANNEL to int8, and writes one file per GEMM:

    b"PCD1", int32 M, K, N, 64-byte name, int8 A[M][K], int8 B[N][K]

MODEL, LAYERS (comma list), PROJ (comma list of q_proj,k_proj,v_proj,o_proj,gate_proj,
up_proj), WIN (512) and OUTDIR select what is written. Reads wikitext2-test.parquet from
the working directory, as the simulator does.
"""
import os, sys, math, runpy, struct
import numpy as np
import torch

torch.set_grad_enabled(False)

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = runpy.run_path(os.path.join(HERE, "rk3576-w8a8-ppl.py"), run_name="not_main")
hadamard_or_none, q8_np = SIM["hadamard_or_none"], SIM["q8_np"]

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "0,13,27").split(",")]
PROJ = os.environ.get("PROJ", "q_proj,k_proj,o_proj,gate_proj").split(",")
WIN = int(os.environ.get("WIN", "512"))
OUTDIR = os.environ.get("OUTDIR", "dumps")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pyarrow.parquet as pq

    os.makedirs(OUTDIR, exist_ok=True)
    tag = MODEL.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "\n\n".join(pq.read_table("wikitext2-test.parquet")["text"].to_pylist())
    w = tok(text, return_tensors="pt").input_ids[0][:WIN]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()

    caught = {}
    hooks = []
    for li in LAYERS:
        L = model.model.layers[li]
        for nm in PROJ:
            parent = L.self_attn if nm in ("q_proj", "k_proj", "v_proj", "o_proj") else L.mlp
            lin = getattr(parent, nm)

            def hook(mod, inp, key=(li, nm)):
                caught[key] = (inp[0].detach().float().reshape(-1, inp[0].shape[-1]).numpy(),
                               mod.weight.detach().float().numpy())
            hooks.append(lin.register_forward_pre_hook(hook))
    model(w.unsqueeze(0))
    for h in hooks:
        h.remove()

    for (li, nm), (A, W) in sorted(caught.items()):
        M, K = A.shape
        N = W.shape[0]
        H = hadamard_or_none(K)
        if H is None:
            print(f"L{li} {nm}: K={K} has no rotation; skipped", file=sys.stderr)
            continue
        Aq, _ = q8_np(A @ H)                      # per tensor
        Wq, _ = q8_np(W @ H, axis=1)              # per output channel
        Aq = np.clip(Aq, -127, 127).astype(np.int8)
        Wq = np.clip(Wq, -127, 127).astype(np.int8)
        name = f"{tag}.L{li}.{nm}"[:64]
        path = os.path.join(OUTDIR, f"{tag}_L{li}_{nm}.bin")
        with open(path, "wb") as f:
            f.write(b"PCD1")
            f.write(struct.pack("<3i", M, K, N))
            f.write(name.encode().ljust(64, b"\0"))
            f.write(Aq.tobytes())
            f.write(Wq.tobytes())
        print(f"{path}: M={M} K={K} N={N}")


if __name__ == "__main__":
    main()
