#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""Build the modernbert_rocket gate's artifacts from a checkpoint and one reference case.

The checkpoint is a directory holding `model.safetensors` and `encoder/config.json` (a
ModernBERT config), the layout the Laya decision model ships. Tensors under `encoder.` are
the ModernBERT encoder. A `head.layers.N.` stack of torch nn.TransformerEncoderLayer weights,
when present, becomes the post stack, and `type_emb` supplies each sequence's post bias.

The case is an .npz with the model inputs and the reference taps:
    in_input_ids [B, T] int64, in_attention_mask [B, T] int64, in_qtype [B] int64
    tap_emb_norm, tap_enc0 .. tap_enc{L-1}, tap_final_norm, tap_head0 .. : [B, T, d] fp32

    python3 tools/modernbert_extract.py CKPT_DIR CASE.npz OUT_DIR

Writes OUT_DIR/mb_weights.f16 and OUT_DIR/<case>.case (formats in tests/modernbert_rocket.c).
"""
import json, os, struct, sys
import numpy as np
from safetensors.numpy import load_file

MAGIC, VERSION = 0x5452424D, 1


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ck, case, out = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    W = load_file(os.path.join(ck, "model.safetensors"))
    cfg = json.load(open(os.path.join(ck, "encoder", "config.json")))
    d, nL, H = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
    dff, dh = cfg["intermediate_size"], d // H
    types = cfg["layer_types"]
    rp = cfg["rope_parameters"]
    ropes = {}
    for t in ("full_attention", "sliding_attention"):
        inv = 1.0 / (np.float64(rp[t]["rope_theta"]) ** (np.arange(0, dh, 2, dtype=np.float64) / dh))
        ropes[t] = inv.astype(np.float32)
    names = list(ropes)
    n_post = 0
    while f"head.layers.{n_post}.norm1.weight" in W:
        n_post += 1
    post_H = max(1, d // 64) if n_post else 0
    post_ff = W["head.layers.0.linear1.weight"].shape[0] if n_post else 0

    f16 = lambda k: W[k].astype(np.float16).tobytes()
    with open(os.path.join(out, "mb_weights.f16"), "wb") as f:
        f.write(struct.pack("<12i", MAGIC, VERSION, d, nL, H, dff, cfg["local_attention"] // 2,
                            n_post, post_H, post_ff, 0, len(names)))
        f.write(struct.pack("<2f", cfg["norm_eps"], 1e-5))
        for l in range(nL):
            has = f"encoder.layers.{l}.attn_norm.weight" in W
            f.write(struct.pack("<3i", int(types[l] == "sliding_attention"), int(has), names.index(types[l])))
        for n in names:
            f.write(ropes[n].tobytes())
        f.write(f16("encoder.embeddings.norm.weight"))
        f.write(f16("encoder.final_norm.weight"))
        for l in range(nL):
            p = f"encoder.layers.{l}."
            if p + "attn_norm.weight" in W:
                f.write(f16(p + "attn_norm.weight"))
            for k in ("attn.Wqkv.weight", "attn.Wo.weight", "mlp_norm.weight", "mlp.Wi.weight", "mlp.Wo.weight"):
                f.write(f16(p + k))
        for j in range(n_post):
            p = f"head.layers.{j}."
            for k in ("norm1.weight", "norm1.bias", "self_attn.in_proj_weight", "self_attn.in_proj_bias",
                      "self_attn.out_proj.weight", "self_attn.out_proj.bias", "norm2.weight", "norm2.bias",
                      "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"):
                f.write(f16(p + k))

    z = np.load(case)
    ids, att = z["in_input_ids"], z["in_attention_mask"]
    B, T = ids.shape
    lens = att.sum(1).astype(np.int32)
    for b in range(B):
        assert att[b, :lens[b]].all() and not att[b, lens[b]:].any(), "mask is not a prefix of ones"
    emb = W["encoder.embeddings.tok_embeddings.weight"][ids].astype(np.float32)
    pb = (W["type_emb.weight"][z["in_qtype"]].astype(np.float32) if n_post
          else np.zeros((B, d), np.float32))
    labels = ["emb_norm"] + [f"enc{l}" for l in range(nL)] + ["final_norm"] + [f"head{j}" for j in range(n_post)]
    taps = np.stack([z["tap_" + k].astype(np.float32) for k in labels])
    name = os.path.splitext(os.path.basename(case))[0]
    with open(os.path.join(out, name + ".case"), "wb") as f:
        f.write(struct.pack("<4i", B, T, len(labels), d))
        f.write(lens.tobytes())
        f.write(emb.tobytes())
        f.write(pb.tobytes())
        f.write(taps.tobytes())
    print(f"{out}: d={d} layers={nL} heads={H} d_ff={dff} post={n_post}x(H={post_H}, ff={post_ff}); "
          f"case {name}: B={B} T={T} len={lens.tolist()}")


if __name__ == "__main__":
    main()
