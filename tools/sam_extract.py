#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""
sam_extract.py — pack the SAM ViT-Det image encoder (facebook/sam-vit-base) into one flat
fp16 blob the C driver (rocket_sam_encoder.c) mmaps, plus the reference tensors the gate
(tests/sam_rocket.c) compares against (a deterministic input + the HF per-layer hidden
states + the neck output). HF eager == the ONNX export (Stage-1/2 validated), so the gate
scores the NPU datapath against ground truth without an ONNX dependency on device.

Blob layout (little-endian). HEADER: 16 x int32 = 64 bytes
    [0] magic 0x53414D42 ("SAMB")   [1] version 1
    [2] d          [3] n_layers     [4] n_head      [5] d_ff
    [6] grid       [7] win          [8] patch       [9] ic
    [10] neck_out  [11] eps_bits(f) [12] image_size [13] dhead
    [14] patch_dim [15] reserved
then n_layers x int32 windowed flags (1 windowed, 0 global),
then fp16 weights in declaration order (see rocket_sam.h). All Linear weights stay row-major
[out,in] (torch nn.Linear == rocket B=[N,K]); the FUSED qkv is [3d,d] (split into q/k/v by
row-thirds downstream). The rel-pos tables are the RESOLVED get_rel_pos output [side,side,dhead]
(side = win windowed / grid global). neck conv1 is the 1x1 weight squeezed to [neck_out,d].
"""
import argparse, os, struct, sys
import numpy as np

MAGIC = 0x53414D42
VERSION = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/sam-vit-base")
    ap.add_argument("--out", default="./sam-artifacts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import SamVisionModel

    print(f"loading {args.model} ...", flush=True)
    model = SamVisionModel.from_pretrained(args.model, attn_implementation="eager",
                                           torch_dtype=torch.float32).eval()
    enc = model.vision_encoder
    cfg = model.config
    d, nh = cfg.hidden_size, cfg.num_attention_heads
    dhead = d // nh
    grid = cfg.image_size // cfg.patch_size
    win = cfg.window_size
    glob = set(cfg.global_attn_indexes)
    n_layers = len(enc.layers)
    d_ff = cfg.mlp_dim
    eps = float(cfg.layer_norm_eps)
    ic = getattr(cfg, "num_channels", 3)
    neck_out = cfg.output_channels
    patch = cfg.patch_size
    image_size = cfg.image_size
    patch_dim = ic * patch * patch

    def npy(t):
        return t.detach().cpu().numpy()

    def f16(a):
        return np.ascontiguousarray(a, dtype=np.float32).astype(np.float16)

    def resolve_rel(rel_pos, S):
        return npy(enc.layers[0].attn.get_rel_pos(S, S, rel_pos))   # [S,S,dhead]

    windowed = [0 if i in glob else 1 for i in range(n_layers)]

    parts = [f16(npy(enc.patch_embed.projection.weight).reshape(d, -1)),   # [d][patch_dim]
             f16(npy(enc.patch_embed.projection.bias)),                    # [d]
             f16(npy(enc.pos_embed).reshape(grid * grid, d))]              # [grid*grid][d]
    for i, layer in enumerate(enc.layers):
        S = win if windowed[i] else grid
        parts += [f16(npy(layer.layer_norm1.weight)), f16(npy(layer.layer_norm1.bias)),
                  f16(npy(layer.attn.qkv.weight)), f16(npy(layer.attn.qkv.bias)),   # [3d,d],[3d]
                  f16(npy(layer.attn.proj.weight)), f16(npy(layer.attn.proj.bias)),
                  f16(npy(layer.layer_norm2.weight)), f16(npy(layer.layer_norm2.bias)),
                  f16(npy(layer.mlp.lin1.weight)), f16(npy(layer.mlp.lin1.bias)),
                  f16(npy(layer.mlp.lin2.weight)), f16(npy(layer.mlp.lin2.bias)),
                  f16(resolve_rel(layer.attn.rel_pos_h, S)),
                  f16(resolve_rel(layer.attn.rel_pos_w, S))]
    parts += [f16(npy(enc.neck.conv1.weight).reshape(neck_out, d)),        # [neck_out][d]
              f16(npy(enc.neck.layer_norm1.weight)), f16(npy(enc.neck.layer_norm1.bias)),
              f16(npy(enc.neck.conv2.weight)),                             # [no][no][3][3]
              f16(npy(enc.neck.layer_norm2.weight)), f16(npy(enc.neck.layer_norm2.bias))]

    eps_bits = struct.unpack("<i", struct.pack("<f", eps))[0]
    header = np.array([MAGIC, VERSION, d, n_layers, nh, d_ff, grid, win, patch, ic,
                       neck_out, eps_bits, image_size, dhead, patch_dim, 0], dtype=np.int32)
    flags = np.array(windowed, dtype=np.int32)

    os.makedirs(args.out, exist_ok=True)
    wpath = os.path.join(args.out, "sam_weights.f16")
    with open(wpath, "wb") as f:
        f.write(header.tobytes()); f.write(flags.tobytes())
        for a in parts:
            f.write(np.ascontiguousarray(a).tobytes())
    print(f"  d={d} layers={n_layers} heads={nh} d_ff={d_ff} grid={grid} win={win} "
          f"neck_out={neck_out} eps={eps}", flush=True)
    print(f"  windowed flags: {windowed}", flush=True)
    print(f"  wrote {wpath}  ({os.path.getsize(wpath)/1e6:.1f} MB)", flush=True)

    # ---- reference: deterministic input -> HF forward -> per-layer hidden + neck ----
    rng = np.random.RandomState(args.seed)
    pix = rng.randn(1, ic, image_size, image_size).astype(np.float32) * 0.5
    with torch.no_grad():
        outp = model(torch.from_numpy(pix), output_hidden_states=True)
    neck = npy(outp.last_hidden_state)[0]                 # [neck_out, grid, grid]
    hid = np.stack([npy(h)[0].reshape(grid * grid, d) for h in outp.hidden_states])  # [nL+1][gg][d]
    pix[0].astype(np.float32).tofile(os.path.join(args.out, "pixels.f32"))
    hid.astype(np.float32).tofile(os.path.join(args.out, "hidden.f32"))
    neck.astype(np.float32).tofile(os.path.join(args.out, "neck.f32"))
    print(f"  refs: pixels[{ic},{image_size},{image_size}] hidden[{hid.shape[0]},{grid*grid},{d}] "
          f"neck[{neck_out},{grid},{grid}]", flush=True)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
