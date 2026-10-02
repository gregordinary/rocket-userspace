#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""Join tests/fa_replay_probe's per-trial markers to the gpu_scheduler trace: which NPU core ran
each worker's QK and AV jobs, against which output state the trial produced.

Input: the text of a tracefs instance's `trace` file, recorded with drm_sched_job_queue,
drm_sched_job_run and drm_sched_job_done enabled and the probe run with FA_REPLAY_MARKER set to
that instance's trace_marker (see tools/fa_core_trace.sh). The trace clock must be global
(mono), since the markers come from the main thread and the jobs from the workers.

How a job is attributed, and what it rests on:
  * A trial is the window between its "FAR B" and "FAR E" markers; a job belongs to it if its
    queue event (emitted in the submitting worker's ioctl) falls inside.
  * A worker is its fd's DRM client id. The context opens its fds in worker order on one thread,
    and client ids are allocated in open order, so the k-th smallest client id seen in a trial is
    worker k. The script refuses a trial whose client count is not the worker count.
  * A worker's jobs come in submit order: its QK run's jobs, then its AV run's. With the
    chained path's BATCH of 64 tasks a job, a run of G heads is ceil(G*tiles/64) jobs; the
    script checks every worker's job count against that and refuses a trial that disagrees.
  * The element's owning worker, head item and tiles follow fa_fan_heads' balanced contiguous
    split and mm_batch_run's (item, mi, ni, ki) task order.
It cannot see what a core held when a job started, or anything inside a job.

Usage: fa_trace_join.py TRACE --nthreads N --n-head 16 --head 12 --token 256 --chan 240
                        --qk-plan M,K,N,Mt,Kt,Nt --av-plan M,K,N,Mt,Kt,Nt [--csv OUT]
"""
import argparse
import collections
import re
import sys

BATCH = 64
CORE = {"fdab0000.npu": "c0", "fdac0000.npu": "c1", "fdad0000.npu": "c2"}

LINE = re.compile(r"^\s*(?P<comm>.*?)-(?P<tid>\d+)\s+\[(?P<cpu>\d+)\]\s+\S+\s+(?P<ts>\d+\.\d+):\s+"
                  r"(?P<ev>[\w]+):\s+(?P<msg>.*)$")
QR = re.compile(r"dev=(?P<dev>\S+), fence=(?P<fc>\d+):(?P<fs>\d+), ring=(?P<ring>\S+), job count:(?P<jc>\d+), "
                r"hw job count:(?P<hc>-?\d+), client_id:(?P<cid>\d+)")
DN = re.compile(r"fence=(?P<fc>\d+):(?P<fs>\d+) signaled")
MK = re.compile(r"FAR (?P<k>[BEM]) m=(?P<m>\d+)(?: t=(?P<t>\d+))?(?: s=(?P<s>\w))?")


def ceil_div(a, b):
    return (a + b - 1) // b


def plan(s):
    v = [int(x) for x in s.split(",")]
    if len(v) != 6:
        raise SystemExit("a plan is M,K,N,Mt,Kt,Nt")
    M, K, N, Mt, Kt, Nt = v
    return dict(M=M, K=K, N=N, Mt=Mt, Kt=Kt, Nt=Nt, nM=ceil_div(M, Mt), nN=ceil_div(N, Nt),
                nK=ceil_div(K, Kt))


def ranges(n_head, nt):
    base, rem = divmod(n_head, nt)
    out, h0 = [], 0
    for t in range(nt):
        cnt = base + (1 if t < rem else 0)
        if cnt <= 0:
            break
        out.append((h0, h0 + cnt))
        h0 += cnt
    return out


def task_index(p, item, mi, ni, ki):
    return ((item * p["nM"] + mi) * p["nN"] + ni) * p["nK"] + ki


def parse(path):
    ev = []
    with open(path, errors="replace") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            m = LINE.match(ln)
            if not m:
                continue
            ev.append((float(m["ts"]), int(m["tid"]), m["ev"], m["msg"].strip()))
    ev.sort(key=lambda e: e[0])
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--nthreads", type=int, required=True)
    ap.add_argument("--n-head", type=int, default=16)
    ap.add_argument("--head", type=int, default=12)
    ap.add_argument("--token", type=int, default=256)
    ap.add_argument("--chan", type=int, default=240)
    ap.add_argument("--qk-plan", required=True)
    ap.add_argument("--av-plan", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    qk, av = plan(a.qk_plan), plan(a.av_plan)
    rg = ranges(a.n_head, min(a.nthreads, a.n_head))
    nw = len(rg)
    tw = next(w for w, (h0, h1) in enumerate(rg) if h0 <= a.head < h1)
    item = a.head - rg[tw][0]
    mi = a.token // qk["Mt"]
    # The element's QK row spans every N tile and K tile of row tile mi; its AV output tile is
    # (mi, chan // Nt) over every K pass.
    qk_tasks = [task_index(qk, item, mi, ni, ki) for ni in range(qk["nN"]) for ki in range(qk["nK"])]
    av_tasks = [task_index(av, item, a.token // av["Mt"], a.chan // av["Nt"], ki) for ki in range(av["nK"])]
    qk_jobs = sorted({t // BATCH for t in qk_tasks})
    av_jobs = sorted({t // BATCH for t in av_tasks})
    per_head_q = qk["nM"] * qk["nN"] * qk["nK"]
    per_head_a = av["nM"] * av["nN"] * av["nK"]
    njobs = [(ceil_div((h1 - h0) * per_head_q, BATCH), ceil_div((h1 - h0) * per_head_a, BATCH)) for h0, h1 in rg]
    print(f"{a.label} workers {nw}, ranges {rg}; target head {a.head} is worker {tw} item {item}; "
          f"its QK job(s) {qk_jobs} of {njobs[tw][0]}, AV job(s) {av_jobs} of {njobs[tw][1]}")

    ev = parse(a.trace)
    fences = {}   # fence -> dict(cid, core, q, run, done, tid)
    trials = []   # dicts
    cur = None
    mode_workers = {}
    for ts, tid, name, msg in ev:
        if name == "tracing_mark_write":
            m = MK.search(msg)
            if not m:
                continue
            if m["k"] == "B":
                cur = dict(m=int(m["m"]), t=int(m["t"]), b=ts, jobs=[])
            elif m["k"] == "E" and cur is not None and cur["m"] == int(m["m"]) and cur["t"] == int(m["t"]):
                cur["e"] = ts
                cur["s"] = m["s"]
                trials.append(cur)
                cur = None
            continue
        if name in ("drm_sched_job_queue", "drm_sched_job_run"):
            m = QR.search(msg)
            if not m:
                continue
            f = (int(m["fc"]), int(m["fs"]))
            d = fences.setdefault(f, dict(cid=int(m["cid"]), core=CORE.get(m["ring"], m["ring"])))
            if name == "drm_sched_job_queue":
                d["q"] = ts
                d["tid"] = tid
                if cur is not None:
                    cur["jobs"].append(f)
            else:
                d["run"] = ts
                d["run_core"] = CORE.get(m["ring"], m["ring"])
        elif name == "drm_sched_job_done":
            m = DN.search(msg)
            if m:
                f = (int(m["fc"]), int(m["fs"]))
                if f in fences:
                    fences[f]["done"] = ts

    # Per-core run order over the whole trace, for "what ran on this core before".
    by_core = collections.defaultdict(list)
    for f, d in fences.items():
        if "run" in d:
            by_core[d.get("run_core", d["core"])].append((d["run"], f))
    prev_on_core = {}
    for c, lst in by_core.items():
        lst.sort()
        for i, (_, f) in enumerate(lst):
            prev_on_core[f] = lst[i - 1][1] if i else None

    # Label every job: (worker, kind, index within kind), per trial.
    label = {}
    bad = collections.Counter()
    rows = []
    for tr in trials:
        cids = sorted({fences[f]["cid"] for f in tr["jobs"]})
        if len(cids) != nw:
            bad["client count %d" % len(cids)] += 1
            continue
        wof = {c: i for i, c in enumerate(cids)}
        perw = collections.defaultdict(list)
        for f in tr["jobs"]:
            perw[wof[fences[f]["cid"]]].append(f)
        ok = True
        for w in range(nw):
            js = sorted(perw[w], key=lambda f: fences[f]["q"])
            if len(js) != sum(njobs[w]):
                bad["worker %d jobs %d != %d" % (w, len(js), sum(njobs[w]))] += 1
                ok = False
                break
            for i, f in enumerate(js):
                kind = "QK" if i < njobs[w][0] else "AV"
                idx = i if kind == "QK" else i - njobs[w][0]
                label[f] = (tr["m"], tr["t"], w, kind, idx)
            perw[w] = js
        if not ok:
            continue
        if any("run" not in fences[f] or fences[f].get("run_core") != fences[f]["core"] for f in tr["jobs"]):
            bad["run event missing or ring moved"] += 1
            continue
        tq = [perw[tw][j] for j in qk_jobs]
        ta = [perw[tw][njobs[tw][0] + j] for j in av_jobs]
        # What ran on the target AV job's core just before it, and how many other cores were
        # busy while it ran.
        f_av = ta[0]
        p = prev_on_core.get(f_av)
        pl = label.get(p)
        if p is None:
            prev = "none"
        elif pl is None:
            prev = "unlabelled"
        else:
            same = "same" if (pl[0], pl[1]) == (tr["m"], tr["t"]) else "prev-trial"
            prev = f"w{pl[2]}{pl[3]}/{same}"
        d = fences[f_av]
        conc = 0
        for g in tr["jobs"]:
            if g == f_av:
                continue
            e = fences[g]
            if e.get("run") is not None and e.get("done") is not None and d.get("done") is not None:
                if e["run"] < d["done"] and e["done"] > d["run"]:
                    conc += 1
        pq = prev_on_core.get(tq[0])
        pql = label.get(pq)
        prevq = "none" if pq is None else ("unlabelled" if pql is None else
                                           f"w{pql[2]}{pql[3]}/" + ("same" if (pql[0], pql[1]) == (tr["m"], tr["t"]) else "prev-trial"))
        rows.append(dict(
            m=tr["m"], t=tr["t"], s=tr["s"],
            qk="+".join(fences[f]["core"] for f in tq), av="+".join(fences[f]["core"] for f in ta),
            allq="".join(fences[perw[w][0]]["core"][1] for w in range(nw)),
            alla="".join(fences[perw[w][njobs[w][0]]]["core"][1] for w in range(nw)),
            prev_av=prev, prev_qk=prevq, conc_av=conc,
            av_ms=(d["done"] - d["run"]) * 1e3 if d.get("done") else -1.0))

    if bad:
        print("refused trials:", dict(bad))
    print(f"joined {len(rows)} of {len(trials)} marked trials")

    def table(key, title):
        c = collections.Counter((r["m"], r[key], r["s"]) for r in rows)
        keys = sorted({k for (_, k, _) in c})
        print(f"\n{title}  (cells: R modal / A odd / O neither)")
        for m in (0, 1):
            tot = collections.Counter()
            parts = []
            for k in keys:
                r_, a_, o_ = c[(m, k, "R")], c[(m, k, "A")], c[(m, k, "O")]
                if r_ + a_ + o_:
                    parts.append(f"{k}: {r_}/{a_}" + (f"/{o_}" if o_ else ""))
                tot["R"] += r_; tot["A"] += a_
            print(f"  mode {m} ({'one ctx' if m == 0 else 'fresh ctx'}), R {tot['R']} A {tot['A']}: " + ";  ".join(parts))

    table("qk", f"target QK job core (worker {tw})")
    table("av", f"target AV job core (worker {tw})")
    for r in rows:
        r["pair"] = f"{r['qk']}|{r['av']}"
    table("pair", "target (QK core | AV core)")
    table("prev_av", "job that ran on the target AV job's core before it")
    table("prev_qk", "job that ran on the target QK job's core before it")
    table("conc_av", "other jobs of the trial overlapping the target AV job")
    for w in range(nw):
        for r in rows:
            r[f"w{w}q"] = "c" + r["allq"][w]
            r[f"w{w}a"] = "c" + r["alla"][w]
    for w in range(nw):
        table(f"w{w}a", f"worker {w} AV core (control: {'TARGET' if w == tw else 'not the target'})")
    for w in range(nw):
        if w != tw:
            table(f"w{w}q", f"worker {w} QK core (control)")

    if a.csv:
        with open(a.csv, "w") as f:
            cols = ["m", "t", "s", "qk", "av", "allq", "alla", "prev_qk", "prev_av", "conc_av", "av_ms"]
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r[c]) if not isinstance(r[c], float) else f"{r[c]:.3f}" for c in cols) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
