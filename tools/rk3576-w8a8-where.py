#!/usr/bin/env python3
"""
w8a8_where.py -- WHERE the per-tensor output requant does its damage, per call and
per region, under each arm's own inputs. One window, no perplexity.

The question this answers: per-row A and per-channel B scales lower the per-GEMM
Frobenius error and RAISE perplexity 38x. A norm cannot say why. So score each call
by region -- which rows carry the error, and how the error is distributed -- rather
than by its total.
"""
import os, math
import numpy as np
import torch
import torch.nn as nn

torch.set_grad_enabled(False)
torch.set_num_threads(16)
MODEL = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-1.7B")
WIN = 512
KREFUSE = 6176
ROWS = []


def q8_np(x, axis=None):
    amax = np.abs(x).max() if axis is None else np.abs(x).max(axis=axis, keepdims=True)
    amax = np.maximum(amax, 1e-30)
    s = (amax / 127.0).astype(np.float32)
    return np.clip(np.rint(x / s), -128, 127).astype(np.float32), s


class Sim(nn.Module):
    def __init__(self, lin, arm, li, nm):
        super().__init__()
        self.arm, self.li, self.nm, self.bias = arm, li, nm, lin.bias
        self.Wf = lin.weight.detach().float().numpy()
        self.N, self.K = self.Wf.shape
        self.off = (self.K % 32 == 0 and self.N % 32 == 0 and self.K < KREFUSE)
        self.W = lin.weight
        if not self.off:
            return
        self.Wq, self.sb = (q8_np(self.Wf, axis=1) if arm == "reachable"
                            else q8_np(self.Wf))
        self.Wq = self.Wq.T.copy()
        self.sb = self.sb.reshape(1, -1)

    def forward(self, x):
        if not self.off:
            return nn.functional.linear(x, self.W, self.bias)
        shp = x.shape
        A = x.reshape(-1, shp[-1]).float().numpy()
        ex = A @ self.Wf.T                                  # this call's exact answer
        Aq, sa = (q8_np(A, axis=1) if self.arm == "reachable" else q8_np(A))
        acc = Aq @ self.Wq
        sc = 127.0 / max(np.abs(acc).max(), 1e-30)
        out = np.clip(np.rint(acc * sc), -128, 127) / sc * sa * self.sb
        e = out - ex
        # per-row error energy against per-row signal energy: does one token carry it?
        er = (e * e).sum(1)
        sr = (ex * ex).sum(1) + 1e-30
        ROWS.append(dict(arm=self.arm, li=self.li, nm=self.nm,
                         rel=float(np.linalg.norm(e) / np.linalg.norm(ex)),
                         # share of the TOTAL error energy in the single worst row
                         top_row_share=float(er.max() / er.sum()),
                         # the worst row's OWN relative error -- the token that is lost
                         worst_row_rel=float(np.sqrt((er / sr).max())),
                         # how many rows are worse than 50% relative
                         rows_over_50pc=int((er / sr > 0.25).sum()),
                         sa_spread=float(sa.max() / max(sa.min(), 1e-30))
                         if sa.size > 1 else 1.0))
        t = torch.from_numpy(out).reshape(*shp[:-1], self.N)
        return t + self.bias if self.bias is not None else t


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pyarrow.parquet as pq
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "\n\n".join(pq.read_table("wikitext2-test.parquet")["text"].to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0][:WIN]

    for arm in ("naive", "reachable"):
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
        for li, L in enumerate(m.model.layers):
            for parent, nm in ((L.self_attn, "q_proj"), (L.self_attn, "k_proj"),
                               (L.self_attn, "v_proj"), (L.self_attn, "o_proj"),
                               (L.mlp, "gate_proj"), (L.mlp, "up_proj"),
                               (L.mlp, "down_proj")):
                setattr(parent, nm, Sim(getattr(parent, nm), arm, li, nm))
        m(ids.unsqueeze(0))
        del m

    for arm in ("naive", "reachable"):
        r = [x for x in ROWS if x["arm"] == arm]
        rel = np.array([x["rel"] for x in r])
        tos = np.array([x["top_row_share"] for x in r])
        wrr = np.array([x["worst_row_rel"] for x in r])
        ov = np.array([x["rows_over_50pc"] for x in r])
        print(f"\n== {arm}: {len(r)} calls ==")
        print(f"  per-call rel err        median {np.median(rel):.4f}  max {rel.max():.4f}")
        print(f"  worst row's OWN rel err median {np.median(wrr):.4f}  max {wrr.max():.4f}")
        print(f"  error energy in 1 row   median {np.median(tos)*100:5.2f}%  "
              f"max {tos.max()*100:.2f}%   (1/512 = 0.20% if spread evenly)")
        print(f"  rows past 50% rel err   median {np.median(ov):.0f}  max {ov.max()}  "
              f"of 512;  calls with any: {(ov>0).sum()} of {len(r)}")
        if arm == "reachable":
            sp = np.array([x["sa_spread"] for x in r])
            print(f"  per-row A scale spread  median {np.median(sp):.1f}x  max {sp.max():.0f}x")
        w = sorted(r, key=lambda x: -x["worst_row_rel"])[:5]
        for x in w:
            print(f"    worst: L{x['li']:02d} {x['nm']:10s} rel {x['rel']:.4f} "
                  f"worst-row {x['worst_row_rel']:.3f} rows>50% {x['rows_over_50pc']}")


main()
