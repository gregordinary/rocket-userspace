#!/usr/bin/env python3
"""
w8a8_ppl.py -- what rocket_matmul_int8_rk3576()'s contract does to a real model's
OUTPUT, not to one GEMM. Host only, no device.

Every offloadable projection in every layer is replaced by an exact simulation of

    C[m][n] = sat8( round( ( sum_k A[m][k]*B[n][k] + bias[n] ) * scale ) )

and the model is scored on wikitext-2 perplexity against its own fp32 and fp16 runs.
A projection is offloadable exactly where the entry accepts it: K % 32 == 0,
N % 32 == 0 and K < 6176 -- which puts every d_model-input projection on the NPU and
leaves the FFN down-projection (K = ffn) on the CPU, the same placement the sgemm
A/B's 2.39x was scoped against.

Arms:
  fp32        the reference
  fp16        the floor -- roughly what the CPU path being replaced delivers
  naive       per-tensor A, per-tensor B, per-tensor int8 out
  reachable   per-ROW A, per-CHANNEL B (the caller quantizes, so both are free),
              per-tensor int8 out -- the shipped entry driven competently
  reach_i32   the same inputs with NO output requant: the accuracy ceiling a
              per-output-channel requant route would approach
  hadamard    Hadamard-rotated inputs, per-tensor scales, per-tensor int8 out
  chanout     per-tensor A, per-channel B, EXACT per-column output scale -- the
              route's ceiling, not what any hardware delivers
  hadamard_chanout        the rotation composed with that exact per-column scale

and the arms that model what rocket_matmul_int8_rk3576_perc() ACTUALLY writes. The
DPU has one (MUL, SHIFT) per task and carries the per-column part on the coefficient
group's int16 C, so a column's gain is the INTEGER product C[n] * MUL/2^SHIFT and C is
capped by that column's worst-case accumulator rather than by its field:

  chanout_ramp            the shipped ramp, de-quantized by the scale the caller ASKED
                          for -- which is what the header's equation promises
  chanout_ramp_ach        the same surface de-quantized by the gain the ramp ACHIEVED
                          (C[n] * MUL/2^SHIFT), which the library computes and does not
                          currently return. The difference between these two arms is
                          the part of the ramp's cost a signature change could cancel.
  hadamard_chanout_ramp / hadamard_chanout_ramp_ach   the same two, composed with the
                          rotation -- the pair that prices the port's real accuracy story

Those arms all take the per-column output scale from the accumulator they are about to
quantize, which is an ORACLE: the entry takes `scale_n` from a caller that has A and B
and no accumulator. Two suffixes model what a caller can actually supply, and either
may be composed with any `chanout` base:

  _cal   the per-column scale is FROZEN from a calibration pass over NCAL windows
         disjoint from the scored ones, then divided by CALSAFE (>1 trades resolution
         for clipping headroom). The calibration pass runs the arm's own quantized
         forward, so the statistics are the ones a real calibration would see, and the
         scored pass CLIPS wherever a window exceeds what calibration saw -- which the
         device does too, in sat8.
  _calshuf  the information-free control for `_cal` (rule 96). It applies exactly the
         multiset of per-column perturbations `_cal` applies, to a fixed random
         permutation of the columns -- same magnitude, same distribution, nothing about
         WHICH column needed the headroom. If it costs as much as `_cal`, `_cal`'s
         number is not a measurement of a calibrated scale.
  _bnd   the per-column scale comes from the analytic accumulator bound
         128*sum_k|Bq[n][k]| -- the same term the shipped planner already uses to cap
         C. Pure, needs no calibration set, and prices what a caller that refuses to
         calibrate would get.

`CAL=1` prints, per arm, the ratio of the frozen scale's implied colmax to the window's
own and the fraction of the output surface that saturates -- the two quantities that
separate "the frozen scale is too tight" from "too loose".

`SPREAD=1` prints, per arm, the per-tile column-scale spread and the delivered
worst-case relative gain error over every offloaded projection. The ramp's resolution
is linear in BOTH the contraction depth and that spread, so the spread is the model
property the ramp table cannot be read without.

Decode is not modelled: llama.cpp offloads prefill only, and a perplexity window IS
a prefill pass, so this scores exactly the pass the port would change.

Accumulation is float32 -- exact for the products and off by at most a few units on
the largest sums, against an output requant step of order a thousand.
"""
import os, math, json
import numpy as np
import torch
import torch.nn as nn

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("THREADS", "16")))

MODEL = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-1.7B")
NWIN = int(os.environ.get("NWIN", "8"))
WIN = int(os.environ.get("WIN", "512"))
KREFUSE = 6176
ARMS = os.environ.get("ARMS", "fp32,fp16,naive,reachable,reach_i32,hadamard").split(",")
# When set, every per-tensor output requant records (arm, K, N, fraction of the
# surface that quantized to zero, the accumulator surface's own outlier ratio) --
# which is what separates "the requant is coarse" from "the requant is misaimed".
CRUSH = [] if os.environ.get("CRUSH") else None
# When set, every per-column ramp records (K, N, within-tile scale spread, C_max,
# worst relative gain error) -- the model property the ramp's resolution is linear in.
SPREAD = [] if os.environ.get("SPREAD") else None
# The output-channel tile the entry runs. rocket_hw_rk3576.max_tile is 2048 and
# r76_mm_fit_nt() only ever SHRINKS it, so this is the pessimistic case: a narrower
# tile spans less of the layer's spread and the ramp gets finer, never coarser.
NT = int(os.environ.get("NT", "2048"))
# The `_cal` arms' calibration set: NCAL windows taken AFTER the scored ones, so the
# frozen scale never saw a token it is scored on. CALSAFE divides the frozen scale, so
# >1 buys clipping headroom at the cost of resolution.
NCAL = int(os.environ.get("NCAL", "2"))
CALSAFE = float(os.environ.get("CALSAFE", "1.0"))
# 0 = no calibration arm; 1 = the calibration pass is running (record, use the oracle);
# 2 = frozen (use what was recorded). Module-level because main() drives the phases and
# every SimLinear has to switch together.
CAL_PHASE = 0
# When set, every `_cal` forward records (K, N, median frozen/actual colmax ratio,
# saturating fraction of the output surface).
CALSTAT = [] if os.environ.get("CAL") else None
# `_calboot`'s estimator error against the exact colmax it is standing in for, appended
# per calibration forward as (K, N, median |est/true - 1|, max, the fraction of columns
# whose second bootstrap pass saturated and fell back). Always collected: the arm's whole
# question is how far the estimate is from the exact freeze, and it costs nothing.
BOOTSTAT = []
# The margin the second bootstrap pass applies to the first pass's estimate. Sized so the
# second pass lands near 85 of 127 codes rather than saturating: pass 1 runs at the
# analytic bound, which overshoots by ~60x median, so est1 is coarse (its surface is a
# couple of codes) and can be LOW.
BOOTMARGIN = float(os.environ.get("BOOTMARGIN", "1.5"))

INT32_MAX = 2147483647.0


def requant_params(conv_scale):
    """The emitter's own QNNPACK derivation, bit for bit.

    src/npu_regcmd_rk3576.c:rocket_rk3576_requant_params. `shift` is the REGISTER
    value, already pre-decremented, so the model shifts by exactly what the DPU has.
    """
    bits = int(np.float32(conv_scale).view(np.uint32))
    shift = 127 + 31 - 32 - (bits >> 23) + 16 - 1
    m = ((bits >> 9) & 0x7FFF) + 1
    if m < (1 << 14):
        m |= (1 << 14)
    if not 0 <= shift <= 62:
        # The emitter writes this into a register field. Refuse rather than compute a
        # full, plausible, wrongly scaled surface -- which is this datapath's signature
        # failure and would be invisible in a perplexity number.
        raise ValueError(f"conv_scale {conv_scale:g} gives SHIFT {shift}, "
                         "outside what the emitter can express")
    return m, shift


def ramp_plan(cs, sum_abs_w):
    """One N tile's (MUL, SHIFT) and C ramp, from the shipped planner.

    src/rocket_conv2d_rk3576.c:rocket_rk3576_plan_perchannel, called from the matmul
    entry with in_scale = out_scale = 1 and w_scale = the caller's scale_n. Returns
    the multiplier, the shift and the gain each column ACTUALLY gets.
    """
    bound = 128.0 * sum_abs_w.astype(np.float64) + 1.0     # |acc| <= 128*sum|w|, bias 0
    cmax = np.clip(INT32_MAX / bound, 1.0, 32767.0)
    best_base = float((cs / cmax).max())
    mul, shift = requant_params(best_base)
    base_actual = mul / float(1 << shift)
    # The C source truncates `want + 0.5` toward zero -- round half UP, not half even.
    C = np.clip(np.floor(cs / base_actual + 0.5), 1.0, 32767.0)
    return mul, shift, C, C * base_actual, float(cmax.max())


def ramp_apply(acc, cs, sum_abs_w, nt, record=None):
    """The device epilogue, per N tile: sat_i32((acc + 0) * C[n]) then the shared
    round-half-to-even (v*MUL) >> SHIFT, then sat8. Returns the int8 surface and the
    per-column gain the ramp delivered."""
    N = acc.shape[1]
    out = np.empty_like(acc)
    gain = np.empty(N, dtype=np.float64)
    for n0 in range(0, N, nt):
        sl = slice(n0, min(n0 + nt, N))
        c = cs[sl].astype(np.float64)
        mul, shift, C, g, cmaxmax = ramp_plan(c, sum_abs_w[sl])
        v = np.clip(acc[:, sl].astype(np.float64) * C, -2147483648.0, INT32_MAX)
        out[:, sl] = np.clip(np.round(v * mul / float(1 << shift)), -128.0, 127.0)
        gain[sl] = g
        if record is not None:
            record.append((float(c.max() / c.min()), cmaxmax,
                           float((np.abs(g - c) / c).max())))
    return out, gain


def q8_np(x, axis=None):
    amax = np.abs(x).max() if axis is None else np.abs(x).max(axis=axis, keepdims=True)
    amax = np.maximum(amax, 1e-30)
    s = (amax / 127.0).astype(np.float32)
    return np.clip(np.rint(x / s), -128, 127).astype(np.float32), s


def hadamard_or_none(K):
    """Orthonormal rotation along K.

    A power-of-two K takes a plain Walsh-Hadamard. Anything else takes it
    BLOCK-DIAGONALLY over the largest power of two dividing K -- still orthonormal, and
    it smears an outlier over its block instead of over the whole contraction, which is
    the weaker of the two but is always available. (The Kronecker construction in
    ggml-rocket/test-hadamard-quant.cpp is the stronger one where K = 15*2^k.)
    """
    b = K & -K                      # largest power of two dividing K
    if b < 2:
        return None
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < b:
        H = np.block([[H, H], [H, -H]])
    H = (H / math.sqrt(b)).astype(np.float32)
    if b == K:
        return H
    out = np.zeros((K, K), dtype=np.float32)
    for i in range(0, K, b):
        out[i:i + b, i:i + b] = H
    return out


class SimLinear(nn.Module):
    """One nn.Linear replaced by the entry, exactly as the entry is specified."""

    def __init__(self, lin, arm):
        super().__init__()
        self.arm = arm
        # `_ramp` swaps the exact per-column output scale for the integer one the part
        # programs; `_ach` additionally de-quantizes by the gain that ramp achieved;
        # `_cal`/`_bnd` change where the per-column scale COMES FROM, which is the only
        # part of the route a caller of the entry has to supply itself. None of them
        # touches how the INPUTS are quantized, so strip them first, outermost first.
        base = arm
        self.scalesrc = "oracle"
        for sfx, nm in (("_calboot", "calboot"), ("_calshuf", "calshuf"),
                        ("_cal", "cal"), ("_bnd", "bnd")):
            if base.endswith(sfx):
                base, self.scalesrc = base[:-len(sfx)], nm
        self.ach = base.endswith("_ach")
        base = base[:-4] if self.ach else base
        self.ramp = base.endswith("_ramp")
        base = base[:-5] if self.ramp else base
        self.base = base
        self.bias = lin.bias
        W = lin.weight.detach().float().numpy()
        self.N, self.K = W.shape
        self.off = (self.K % 32 == 0 and self.N % 32 == 0 and self.K < KREFUSE)
        self.W = lin.weight
        if not self.off:
            return
        if base == "exact":
            return                                         # the control: no quantizer
        if base in ("hadamard", "hadamard_chanout"):
            self.H = hadamard_or_none(self.K)
            if self.H is None:
                self.off = False
                return
            Wr = W @ self.H
            self.Wq, self.sb = (q8_np(Wr, axis=1) if base.endswith("chanout")
                                else q8_np(Wr))
        elif base == "naive":
            self.Wq, self.sb = q8_np(W)                    # per-tensor
        else:
            self.Wq, self.sb = q8_np(W, axis=1)            # per-channel
        self.Wq = self.Wq.T.copy()                         # [K,N] for the GEMM
        self.sb = self.sb.reshape(1, -1)                   # broadcast over rows
        # sum_k |B[n][k]| is what caps C[n]; it is a property of the WEIGHTS alone, so
        # it is computed once here rather than per forward.
        self.sum_abs_w = np.abs(self.Wq).sum(axis=0)
        self.nt = min(NT, self.N) // 32 * 32 or 32
        # Accumulated over the calibration pass by `_cal` arms; never read otherwise.
        self.cal_colmax = np.zeros(self.N, dtype=np.float64)
        # Fixed per layer and seeded off its shape, so the control is reproducible and
        # is the same permutation in every window of a run.
        self.perm = np.random.default_rng(
            (self.K * 1000003 + self.N) & 0x7FFFFFFF).permutation(self.N)

    def _bootstrap_colmax(self, acc):
        """The frozen colmax a REAL caller can get, without a host int32 GEMM.

        `_cal` freezes the exact per-column max of the calibration accumulator, which a
        caller of rocket_matmul_int8_rk3576_perc() cannot see: the entry writes int8 and
        the accumulator never leaves the part. What a caller CAN do is ask the entry
        itself, twice. Pass one runs at the analytic bound 127/(128*sum|w|), which
        overshoots a real column by ~60x and therefore cannot saturate, so max|C8|/scale
        is a valid — if coarse, a couple of codes — lower bound on the column's max. Pass
        two runs at that estimate times BOOTMARGIN and reads it again, now with most of
        the int8 range in use.

        Both passes go through the SHIPPED integer ramp and are de-quantized by the scale
        the caller ASKED for, not the gain the ramp delivered, because that is what the
        entry's caller has (the `_ach` question is settled and dividing by the request is
        what ships). So the ramp's own gain error is inside this estimate, deliberately.

        Returns the estimate. A column whose second pass saturated anyway keeps the first
        pass's margined value, which is an over-estimate and therefore safe in the
        direction the tail arm says matters.
        """
        true = np.abs(acc).max(axis=0).astype(np.float64)
        bnd = 128.0 * self.sum_abs_w.astype(np.float64) + 1.0
        sc1 = 127.0 / bnd
        q1, _ = ramp_apply(acc, sc1, self.sum_abs_w, self.nt)
        # A column that reads back all-zero at the bound is one whose accumulator is below
        # half a code THERE; floor it at the code it would have taken, so the second pass
        # still has something to refine and no column is handed a zero scale.
        est1 = np.maximum(np.abs(q1).max(axis=0), 1.0) / sc1
        sc2 = 127.0 / (est1 * BOOTMARGIN)
        q2, _ = ramp_apply(acc, sc2, self.sum_abs_w, self.nt)
        satcol = (np.abs(q2) >= 127.0).any(axis=0)
        est2 = np.maximum(np.abs(q2).max(axis=0), 1.0) / sc2
        est = np.where(satcol, est1 * BOOTMARGIN, est2)
        rel = np.abs(est / np.maximum(true, 1e-30) - 1.0)
        # What pass ONE alone would have frozen, as its own error. The two are far apart
        # and the median of the SIGNED ratio hides it: at the analytic bound a column's
        # readback is a couple of codes, so pass one is coarse per column and unbiased
        # only in aggregate. This is the number that says whether pass two is load-bearing.
        rel1 = np.abs(est1 * BOOTMARGIN / np.maximum(true, 1e-30) - 1.0)
        BOOTSTAT.append((self.K, self.N, float(np.median(rel)), float(rel.max()),
                         float(satcol.mean()),
                         float(np.median(est1 * BOOTMARGIN / np.maximum(true, 1e-30))),
                         float(np.median(rel1)), float(rel1.max())))
        return est

    def _colscale(self, acc):
        """The per-column output scale the CALLER hands the entry, and this call's own
        column maxima for scoring it against.

        `oracle` is what every published number on this route used -- 127/max|acc| over
        the very accumulator being quantized, which no caller of
        rocket_matmul_int8_rk3576_perc() is in a position to compute. The other two are
        what a caller can actually supply: `bnd` is the pure analytic accumulator bound
        128*sum_k|Bq[n][k]| (the same term the shipped planner already uses to cap C),
        and `cal` is frozen from a disjoint calibration set. During the calibration pass
        `cal` RECORDS and still runs the oracle, so the statistics it freezes are the
        ones a real calibration of this quantized model would see.
        """
        actual = np.abs(acc).max(axis=0).astype(np.float64)
        actual = np.maximum(actual, 1e-6 * max(actual.max(), 1e-30))
        if self.scalesrc == "bnd":
            return 127.0 / (128.0 * self.sum_abs_w.astype(np.float64) + 1.0), actual
        if self.scalesrc in ("cal", "calshuf", "calboot"):
            if CAL_PHASE == 1:
                # `calboot` freezes what two device calls could have told it; the other
                # two freeze the exact colmax, which is the thing it is standing in for.
                np.maximum(self.cal_colmax,
                           self._bootstrap_colmax(acc)
                           if self.scalesrc == "calboot" else actual,
                           out=self.cal_colmax)
                return 127.0 / actual, actual
            frozen = np.maximum(self.cal_colmax, 1e-30) * CALSAFE
            if self.scalesrc in ("cal", "calboot"):
                return 127.0 / frozen, actual
            # The information-free control (rule 96). `_cal` multiplies the oracle scale
            # by d[n] = actual[n]/frozen[n]; this applies exactly that multiset of
            # per-column perturbations to the WRONG columns, so it has the same
            # magnitude and the same distribution and carries nothing about which
            # column needed the headroom. If the metric moves as far as `_cal` did,
            # `_cal`'s number is not a measurement of a calibrated scale.
            return (127.0 / actual) * (actual / frozen)[self.perm], actual
        return 127.0 / actual, actual

    def _calstat(self, sc, actual, q):
        """What separates "the supplied scale is too tight" from "too loose": the
        implied colmax against the call's own, and how much of the surface saturated.
        A ratio below 1 means the entry is clipping; well above 1 means it is spending
        int8 codes on headroom the accumulator never reached."""
        if CALSTAT is None or CAL_PHASE == 1:
            return
        a = np.maximum(np.asarray(actual, dtype=np.float64),
                       1e-6 * max(float(np.max(actual)), 1e-30))
        ratio = (127.0 / np.asarray(sc, dtype=np.float64).ravel()) / a
        CALSTAT.append((self.arm, self.K, self.N,
                        float(np.median(ratio)), float(np.min(ratio)),
                        float((np.abs(q) >= 127.0).mean())))

    def forward(self, x):
        if not self.off:
            return nn.functional.linear(x, self.W, self.bias)
        shp = x.shape
        if self.base == "exact":
            # The control that must succeed: the same reshape, the same numpy GEMM,
            # the same dequantize-shaped epilogue, and no quantizer anywhere. If this
            # does not reproduce fp32 exactly, every arm below is measuring the plumbing.
            A = x.reshape(-1, shp[-1]).float().numpy()
            out = torch.from_numpy(A @ self.W.detach().float().numpy().T)
            out = out.reshape(*shp[:-1], self.N)
            return out + self.bias if self.bias is not None else out
        A = x.reshape(-1, shp[-1]).float().numpy()
        if self.base in ("hadamard", "hadamard_chanout"):
            Aq, sa = q8_np(A @ self.H)
        elif self.base in ("naive", "chanout"):
            Aq, sa = q8_np(A)                              # per tensor
        else:
            Aq, sa = q8_np(A, axis=1)                      # per row
        acc = Aq @ self.Wq                                 # int32, held in float32
        if self.base.endswith("chanout") and self.ramp:
            # What the part WRITES: one (MUL, SHIFT) per N tile with the per-column
            # part on the coefficient group's integer C. The caller asks for `sc` and
            # gets `gain`; which of the two it de-quantizes by is the whole difference
            # between the `_ach` arm and this one.
            # A column whose accumulator is identically zero would otherwise ask for an
            # unbounded gain; floor it against the surface rather than against an
            # absolute epsilon, which is what any real caller would do.
            sc, actual = self._colscale(acc)
            rec = [] if SPREAD is not None else None
            q, gain = ramp_apply(acc, sc, self.sum_abs_w, self.nt, rec)
            self._calstat(sc, actual, q)
            # float32 out: the plan arithmetic is float64 but the model's activations
            # are float32, and torch will not matmul a double against a float weight.
            rec_acc = acc
            acc = (q / (gain if self.ach else sc).reshape(1, -1)).astype(np.float32)
            if rec is not None:
                # The RECONSTRUCTED surface's RMS relative to the true accumulator's
                # rides along because the spread and the gain error are both
                # SCALE-INVARIANT: a uniform magnitude blow-up leaves them healthy.
                r0 = float(np.sqrt((rec_acc.astype(np.float64) ** 2).mean()))
                r1 = float(np.sqrt((acc.astype(np.float64) ** 2).mean()))
                for spread, cmax, err in rec:
                    SPREAD.append((self.arm, self.K, self.N, spread, cmax, err,
                                   r1 / max(r0, 1e-30)))
        elif self.base.endswith("chanout"):
            # The route's CEILING: an exact per-column output scale, which no register
            # on this part can express. Kept as the arm the 1.004x was quoted from.
            # float32, as this arm has always been: the model's activations are float32
            # and `_colscale` works in float64 for the ramp's integer planner.
            sc = sc32 = self._colscale(acc)[0].astype(np.float32)
            q = np.clip(np.rint(acc * sc32), -128, 127)
            self._calstat(sc, np.abs(acc).max(axis=0).astype(np.float64), q)
            acc = q / sc32
        elif self.base != "reach_i32":
            sc = 127.0 / max(np.abs(acc).max(), 1e-30)     # ONE per-tensor scale
            q = np.rint(acc * sc)
            if CRUSH is not None:
                CRUSH.append((self.arm, self.K, self.N,
                              float((q == 0).mean()),
                              float(np.abs(acc).max() /
                                    np.sqrt((acc * acc).mean()))))
            acc = np.clip(q, -128, 127) / sc
        out = torch.from_numpy(acc * sa * self.sb).reshape(*shp[:-1], self.N)
        return out + self.bias if self.bias is not None else out


def patch(model, arm):
    n_off = n_cpu = 0
    for L in model.model.layers:
        for parent, nm in ((L.self_attn, "q_proj"), (L.self_attn, "k_proj"),
                           (L.self_attn, "v_proj"), (L.self_attn, "o_proj"),
                           (L.mlp, "gate_proj"), (L.mlp, "up_proj"),
                           (L.mlp, "down_proj")):
            lin = getattr(parent, nm)
            s = SimLinear(lin, arm)
            n_off += s.off
            n_cpu += (not s.off)
            setattr(parent, nm, s)
    return n_off, n_cpu


def main():
    global CAL_PHASE
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pyarrow.parquet as pq

    tag = MODEL.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "\n\n".join(pq.read_table("wikitext2-test.parquet")["text"].to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    wins = [ids[i * WIN:(i + 1) * WIN] for i in range(NWIN)]
    # Disjoint from the scored windows by construction, so a `_cal` arm's frozen scale
    # never saw a token it is scored on. A calibration set that overlaps the scoring set
    # would reproduce the oracle arm and read as though the oracle were free.
    cwins = [ids[(NWIN + i) * WIN:(NWIN + i + 1) * WIN] for i in range(NCAL)]
    print(f"{tag}: {NWIN} windows of {WIN} real wikitext-2 tokens "
          f"(+{NCAL} disjoint calibration windows for any _cal arm, "
          f"CALSAFE={CALSAFE})", flush=True)

    ref_top1 = None
    res = {}
    out_path = os.environ.get("OUT", f"ppl_{tag}.json")
    # An eight-arm run is hours, and writing the file once at the end loses every arm
    # if the run is killed. Results are appended per arm instead. A stored file is only
    # carried forward when it was produced at the SAME window count: merging a one-window
    # smoke into an eight-window table gives a complete, plausible, incomparable one.
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("_nwin") == NWIN and prev.get("_model") == MODEL:
            res = prev
            print(f"  carrying forward {len(res) - 3} arm(s) from {out_path} "
                  f"at NWIN={NWIN}", flush=True)
        else:
            print(f"  {out_path} is NWIN={prev.get('_nwin')} "
                  f"model={prev.get('_model')}; not comparable, overwriting", flush=True)
    res["_nwin"], res["_win"], res["_model"] = NWIN, WIN, MODEL

    def flush_res():
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)

    for arm in ARMS:
        # fp32 always re-runs even when stored: it is the top-1 reference every later
        # arm is scored against, and that reference only exists in this process.
        if arm in res and arm != "fp32":
            print(f"  {arm:26s} already stored: ppl {res[arm]['ppl']:9.4f}", flush=True)
            continue
        CAL_PHASE = 0
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
        model.eval()
        if arm == "fp16":
            model = model.half()
        elif arm != "fp32":
            n_off, n_cpu = patch(model, arm)
            print(f"  [{arm}] {n_off} projections on the NPU, {n_cpu} left on the CPU",
                  flush=True)
        if arm.endswith(("_cal", "_calshuf", "_calboot")):
            # The calibration pass. It runs the oracle so the activations reaching each
            # layer are the quantized model's own, records each column's max, and is
            # thrown away -- nothing it computes is scored.
            CAL_PHASE = 1
            for w in cwins:
                model(w.unsqueeze(0))
            CAL_PHASE = 2
            if SPREAD is not None:
                SPREAD.clear()
            print(f"  [{arm}] scales frozen over {NCAL} disjoint windows", flush=True)
        nll, ntok, agree, nagree = 0.0, 0, 0, 0
        top1 = []
        for w in wins:
            out = model(w.unsqueeze(0)).logits[0].float()
            lp = torch.log_softmax(out[:-1], -1)
            tgt = w[1:]
            nll += float(-lp[torch.arange(len(tgt)), tgt].sum())
            ntok += len(tgt)
            top1.append(out.argmax(-1))
        top1 = torch.cat(top1)
        if ref_top1 is None:
            ref_top1 = top1
        agree = float((top1 == ref_top1).float().mean())
        ppl = math.exp(nll / ntok)
        res[arm] = {"ppl": ppl, "top1_agree_vs_fp32": agree}
        base = res.get("fp32", {}).get("ppl", ppl)
        print(f"  {arm:26s} ppl {ppl:9.4f}   x fp32 {ppl/base:7.3f}   "
              f"top-1 agreement with fp32 {agree*100:6.2f}%", flush=True)
        if SPREAD:
            sp = np.array([s[3] for s in SPREAD])
            cm = np.array([s[4] for s in SPREAD])
            er = np.array([s[5] for s in SPREAD]) * 100.0
            print(f"             ramp over {len(SPREAD)} tiles: column-scale spread "
                  f"median {np.median(sp):.1f}x (min {sp.min():.1f}, max {sp.max():.1f}), "
                  f"C_max median {np.median(cm):.0f}, worst_rel_err median "
                  f"{np.median(er):.2f}% (max {er.max():.2f}%), "
                  f"reconstructed/true RMS median {np.median([s[6] for s in SPREAD]):.4f} "
                  f"(min {min(s[6] for s in SPREAD):.4f}, "
                  f"max {max(s[6] for s in SPREAD):.4f})", flush=True)
            res[arm]["spread_median"] = float(np.median(sp))
            res[arm]["spread_max"] = float(sp.max())
            res[arm]["cmax_median"] = float(np.median(cm))
            res[arm]["rel_err_median_pct"] = float(np.median(er))
            res[arm]["rel_err_max_pct"] = float(er.max())
            # Per SHAPE, because the ramp is linear in K and the layer families differ.
            for kn in sorted({(s[1], s[2]) for s in SPREAD}):
                e = np.array([s[5] for s in SPREAD
                              if (s[1], s[2]) == kn]) * 100.0
                s2 = np.array([s[3] for s in SPREAD if (s[1], s[2]) == kn])
                print(f"               K={kn[0]:5d} N={kn[1]:5d}  spread median "
                      f"{np.median(s2):6.1f}x  worst_rel_err median {np.median(e):6.2f}% "
                      f"max {e.max():6.2f}%", flush=True)
            SPREAD.clear()
        if CALSTAT:
            r = np.array([c[3] for c in CALSTAT])
            rmin = np.array([c[4] for c in CALSTAT])
            sat = np.array([c[5] for c in CALSTAT]) * 100.0
            print(f"             supplied scale over {len(CALSTAT)} forwards: implied "
                  f"colmax / this call's own, median-over-columns "
                  f"{np.median(r):8.3f}x (min {r.min():.3f}, max {r.max():.1f}); "
                  f"tightest single column {rmin.min():.3f}x; "
                  f"saturating fraction of the surface mean {sat.mean():.3f}% "
                  f"(max {sat.max():.2f}%)", flush=True)
            res[arm]["scale_ratio_median"] = float(np.median(r))
            res[arm]["scale_ratio_min"] = float(rmin.min())
            res[arm]["saturating_pct_mean"] = float(sat.mean())
            res[arm]["saturating_pct_max"] = float(sat.max())
            for kn in sorted({(c[1], c[2]) for c in CALSTAT}):
                e = np.array([c[3] for c in CALSTAT if (c[1], c[2]) == kn])
                s2 = np.array([c[5] for c in CALSTAT if (c[1], c[2]) == kn]) * 100.0
                print(f"               K={kn[0]:5d} N={kn[1]:5d}  ratio median "
                      f"{np.median(e):8.3f}x  saturating mean {s2.mean():6.3f}%",
                      flush=True)
            CALSTAT.clear()
        if BOOTSTAT:
            med = np.array([b[2] for b in BOOTSTAT]) * 100.0
            mx = np.array([b[3] for b in BOOTSTAT]) * 100.0
            sat = np.array([b[4] for b in BOOTSTAT]) * 100.0
            p1 = np.array([b[5] for b in BOOTSTAT])
            print(f"             bootstrap estimator over {len(BOOTSTAT)} calibration "
                  f"forwards, margin {BOOTMARGIN}: |est/true - 1| median-over-columns "
                  f"{np.median(med):.2f}% (worst forward's median {med.max():.2f}%, worst "
                  f"single column {mx.max():.1f}%); pass-2 saturating columns mean "
                  f"{sat.mean():.2f}% (max {sat.max():.2f}%); pass-1 margined estimate / "
                  f"true median {np.median(p1):.3f}x, and ITS |est/true - 1| median "
                  f"{np.median([b[6] for b in BOOTSTAT])*100:.1f}% "
                  f"(worst column {max(b[7] for b in BOOTSTAT)*100:.1f}%)", flush=True)
            res[arm]["boot_relerr_median_pct"] = float(np.median(med))
            res[arm]["boot_relerr_max_pct"] = float(mx.max())
            res[arm]["boot_sat_col_pct_mean"] = float(sat.mean())
            res[arm]["boot_pass1_ratio_median"] = float(np.median(p1))
            res[arm]["boot_pass1_relerr_median_pct"] = float(
                np.median([b[6] for b in BOOTSTAT]) * 100.0)
            for kn in sorted({(b[0], b[1]) for b in BOOTSTAT}):
                e = np.array([b[2] for b in BOOTSTAT if (b[0], b[1]) == kn]) * 100.0
                s2 = np.array([b[4] for b in BOOTSTAT if (b[0], b[1]) == kn]) * 100.0
                print(f"               K={kn[0]:5d} N={kn[1]:5d}  |est/true - 1| median "
                      f"{np.median(e):6.2f}%  pass-2 saturating {s2.mean():6.2f}%",
                      flush=True)
            BOOTSTAT.clear()
        if CRUSH:
            z = np.array([c[3] for c in CRUSH])
            o = np.array([c[4] for c in CRUSH])
            print(f"             per-tensor requant: {z.mean()*100:5.1f}% of the "
                  f"surface quantizes to ZERO (max {z.max()*100:.1f}%), "
                  f"accumulator outlier ratio median {np.median(o):.1f} "
                  f"max {o.max():.0f}", flush=True)
            res[arm]["crushed_to_zero"] = float(z.mean())
            res[arm]["acc_outlier_median"] = float(np.median(o))
            CRUSH.clear()
        del model
        flush_res()


if __name__ == "__main__":
    main()
