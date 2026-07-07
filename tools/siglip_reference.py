#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""
siglip_reference.py — the fp32 fidelity oracle for the SigLIP-B/16 vision encoder
(the SmolVLM-256M front-end).

Loads SmolVLM-256M with HF transformers, builds ONE fixed 512x512 image, runs the
vision encoder in fp32 with output_hidden_states=True, and dumps to --out:

  pixels.f32     [3*512*512]      the preprocessed input (CHW), reused verbatim by the NPU
  hidden.f32     [(L+1)*Lpatch*d] the (n_layers+1) per-layer hidden states, fp32:
                                  index 0 = embeddings (patch+pos, the layer-0 input),
                                  index 1..n_layers = output of each encoder layer
  postln.f32     [Lpatch*d]       the post_layernorm output (what SmolVLM consumes)
  meta.json      the resolved config (d, n_layers, n_head, d_ff, L, patch geometry, eps)
                 + the resolved HF weight-name prefixes (so siglip_extract.py agrees)

The SAME pixel tensor is fed to the model and saved for the NPU, so the cosine
comparison is on identical inputs. The image content is deterministic (seeded) — for
a fidelity test only the consistency between oracle and NPU matters.
"""
import argparse, json, os, sys
import numpy as np


def make_image(h, w, seed):
    """A deterministic structured RGB image (smooth gradients + mild noise) so the
    encoder sees realistic dynamic range rather than pathological pure noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= h
    xx /= w
    img = np.empty((h, w, 3), np.float32)
    for c in range(3):
        img[..., c] = 0.5 + 0.5 * np.sin(2 * np.pi * ((c + 1) * 3 * xx + (c + 2) * 2 * yy))
    # a couple of blocks to add edges
    img[h // 4:h // 2, w // 4:w // 2, :] *= 0.4
    img[3 * h // 5:4 * h // 5, w // 8:3 * w // 8, 0] = 0.9
    img = 0.85 * img + 0.15 * rng.random((h, w, 3), dtype=np.float32)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def resolve_vision(model):
    """Return (vision_transformer_module, prefix) for SmolVLM / Idefics3 nestings."""
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
    raise RuntimeError("could not locate the vision transformer on the model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--out", default="./siglip-artifacts")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    print(f"loading {args.model} (fp32) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    vis, prefix = resolve_vision(model)
    vcfg = vis.config
    d = vcfg.hidden_size
    n_layers = vcfg.num_hidden_layers
    n_head = vcfg.num_attention_heads
    d_ff = vcfg.intermediate_size
    patch = vcfg.patch_size
    image_size = vcfg.image_size
    eps = float(getattr(vcfg, "layer_norm_eps", 1e-6))
    hidden_act = getattr(vcfg, "hidden_act", "gelu_pytorch_tanh")
    side = image_size // patch
    L = side * side
    print(f"  vision prefix='{prefix}'  d={d} layers={n_layers} heads={n_head} "
          f"d_ff={d_ff} patch={patch} image={image_size} L={L} eps={eps} act={hidden_act}",
          flush=True)

    # ---- one fixed image -> normalized pixel tensor [1,3,image,image] ----
    img = make_image(image_size, image_size, args.seed)            # HWC uint8
    mean = np.array([0.5, 0.5, 0.5], np.float32)
    std = np.array([0.5, 0.5, 0.5], np.float32)
    pix = ((img.astype(np.float32) / 255.0) - mean) / std         # HWC
    pix = np.transpose(pix, (2, 0, 1)).copy()                     # CHW
    pix.astype(np.float32).tofile(os.path.join(args.out, "pixels.f32"))
    pixel_values = torch.from_numpy(pix)[None].to(torch.float32)   # [1,3,H,W]

    # ---- vision encoder forward, all hidden states ----
    with torch.no_grad():
        try:
            out = vis(pixel_values=pixel_values, output_hidden_states=True)
        except TypeError:
            out = vis(pixel_values, output_hidden_states=True)

    hs = out.hidden_states                 # tuple length n_layers+1
    last = out.last_hidden_state            # [1,L,d] = post_layernorm output
    assert len(hs) == n_layers + 1, f"got {len(hs)} hidden states, expected {n_layers+1}"
    assert tuple(last.shape) == (1, L, d), f"last_hidden_state {tuple(last.shape)} != (1,{L},{d})"

    hid = np.stack([h[0].float().numpy() for h in hs], axis=0).astype(np.float32)  # [L+1,L,d]
    hid.tofile(os.path.join(args.out, "hidden.f32"))
    last[0].float().numpy().astype(np.float32).tofile(os.path.join(args.out, "postln.f32"))

    # report drift between consecutive layers (sanity: the encoder should transform x)
    e = hid[0].reshape(-1)
    p = last[0].float().numpy().reshape(-1)
    cos_emb_post = float(np.dot(e, p) / (np.linalg.norm(e) * np.linalg.norm(p) + 1e-30))

    meta = dict(model=args.model, vision_prefix=prefix, d=d, n_layers=n_layers,
                n_head=n_head, d_ff=d_ff, patch=patch, image_size=image_size,
                side=side, L=L, eps=eps, hidden_act=hidden_act, seed=args.seed,
                head_dim=d // n_head, cos_emb_vs_post=cos_emb_post)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  wrote pixels.f32 hidden.f32 postln.f32 meta.json to {args.out}", flush=True)
    print(f"  sanity cos(embeddings, post_ln) = {cos_emb_post:.4f} "
          f"(should be well below 1 — the encoder does real work)", flush=True)
    print("ORACLE_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
