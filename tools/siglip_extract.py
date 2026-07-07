#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""
siglip_extract.py — pack the SigLIP-B/16 vision encoder weights (SmolVLM-256M
front-end) into one flat fp16 blob the C driver (rocket_siglip_encoder.c) mmaps.

Blob layout (little-endian).  HEADER: 16 x int32 = 64 bytes
    [0]  magic       0x53474C50
    [1]  version     1
    [2]  d           hidden size                (768)
    [3]  n_layers                               (12)
    [4]  n_head                                 (12)
    [5]  d_ff        intermediate size          (3072)
    [6]  L           num patches = side*side    (1024)
    [7]  patch_dim   ic*kh*kw                    (768)
    [8]  ic                                      (3)
    [9]  kh                                      (16)
    [10] kw                                      (16)
    [11] stride                                  (16)
    [12] image_size                              (512)
    [13] eps_bits     float32 bits of layer_norm_eps
    [14] reserved 0
    [15] reserved 0
then fp16 weights, in THIS order:
    patch_W [d][patch_dim]   (= patch_embedding.weight reshaped, [oc][ic*kh*kw])
    patch_b [d]
    pos     [L][d]           (position embedding, already gathered into patch order)
    per layer 0..n_layers-1:
        ln1_g[d] ln1_b[d]
        Wq[d][d] bq[d]  Wk[d][d] bk[d]  Wv[d][d] bv[d]  Wo[d][d] bo[d]
        ln2_g[d] ln2_b[d]
        Wf1[d_ff][d] bf1[d_ff]  Wf2[d][d_ff] bf2[d]
    post_g[d] post_b[d]

All linear weights stay row-major [out,in] (PyTorch nn.Linear == rocket_matmul_fp16's
B=[N,K] for C=A.B^T), so no transpose. patch_W flattens [oc,ic,kh,kw] row-major so it
matches the host im2col patch ordering [ic*kh*kw].
"""
import argparse, json, os, struct, sys
import numpy as np

MAGIC = 0x53474C50
VERSION = 1


def resolve_vision_prefix(model):
    for path in ("model.vision_model", "vision_model",
                 "model.model.vision_model", "model.vision_tower"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj, path
    raise RuntimeError("could not locate the vision transformer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--out", default="./siglip-artifacts/siglip_weights.f16")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText

    print(f"loading {args.model} ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    vis, prefix = resolve_vision_prefix(model)
    vcfg = vis.config
    d, n_layers, n_head = vcfg.hidden_size, vcfg.num_hidden_layers, vcfg.num_attention_heads
    d_ff, patch, image_size = vcfg.intermediate_size, vcfg.patch_size, vcfg.image_size
    eps = float(getattr(vcfg, "layer_norm_eps", 1e-6))
    ic = getattr(vcfg, "num_channels", 3)
    side = image_size // patch
    L = side * side
    patch_dim = ic * patch * patch

    sd = model.state_dict()

    def get(name):
        full = f"{prefix}.{name}"
        if full not in sd:
            raise KeyError(f"missing weight {full}")
        return sd[full].detach().float().cpu().numpy()

    # ---- position embedding gathered into patch (raster) order for a full image ----
    pos_w = get("embeddings.position_embedding.weight")        # [num_pos, d]
    num_pos = pos_w.shape[0]
    # replicate Idefics3/SigLIP bucketize position-id mapping (identity for a full grid)
    import torch as T
    boundaries = T.arange(1.0 / side, 1.0, 1.0 / side)
    frac = T.arange(0, 1 - 1e-6, 1.0 / side)
    bh = T.bucketize(frac, boundaries, right=True)
    bw = T.bucketize(frac, boundaries, right=True)
    pos_ids = (bh[:, None] * side + bw).flatten().numpy()
    assert pos_ids.shape[0] == L, (pos_ids.shape, L)
    assert pos_ids.max() < num_pos
    pos = pos_w[pos_ids]                                        # [L][d]

    patch_W = get("embeddings.patch_embedding.weight").reshape(d, -1)   # [d][patch_dim]
    assert patch_W.shape == (d, patch_dim), (patch_W.shape, (d, patch_dim))
    patch_b = get("embeddings.patch_embedding.bias")

    def f16(a):
        return np.ascontiguousarray(a, dtype=np.float32).astype(np.float16)

    parts = [f16(patch_W), f16(patch_b), f16(pos)]

    for i in range(n_layers):
        p = f"encoder.layers.{i}."
        parts += [f16(get(p + "layer_norm1.weight")), f16(get(p + "layer_norm1.bias"))]
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            parts += [f16(get(p + f"self_attn.{proj}.weight")),
                      f16(get(p + f"self_attn.{proj}.bias"))]
        parts += [f16(get(p + "layer_norm2.weight")), f16(get(p + "layer_norm2.bias"))]
        parts += [f16(get(p + "mlp.fc1.weight")), f16(get(p + "mlp.fc1.bias")),
                  f16(get(p + "mlp.fc2.weight")), f16(get(p + "mlp.fc2.bias"))]

    parts += [f16(get("post_layernorm.weight")), f16(get("post_layernorm.bias"))]

    # sanity: total element count must equal the analytic size
    expect = (d * patch_dim + d + L * d
              + n_layers * (2 * d + 4 * (d * d + d) + 2 * d + (d_ff * d + d_ff) + (d * d_ff + d))
              + 2 * d)
    total = sum(a.size for a in parts)
    assert total == expect, f"element count {total} != {expect}"

    eps_bits = struct.unpack("<i", struct.pack("<f", eps))[0]
    header = np.array([MAGIC, VERSION, d, n_layers, n_head, d_ff, L, patch_dim,
                       ic, patch, patch, patch, image_size, eps_bits, 0, 0], dtype=np.int32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(header.tobytes())
        for a in parts:
            f.write(np.ascontiguousarray(a).tobytes())

    nbytes = os.path.getsize(args.out)
    print(f"  prefix='{prefix}' d={d} layers={n_layers} heads={n_head} d_ff={d_ff} "
          f"L={L} patch_dim={patch_dim} eps={eps}", flush=True)
    print(f"  wrote {args.out}  ({nbytes/1e6:.1f} MB, {total} fp16 weights)", flush=True)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
