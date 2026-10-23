"""H45 (rejected): A line adapter learned from the same line's screen of OTHER targets.

Context A looks like Jurkat (vcell.qc.identity: all 17 Jurkat fingerprint genes), whose only public screen covers essential genes, none
of them panel targets. That screen still shows how the line's genes respond compared with the pooled lines. Test on the four
lines that have both an essential-axis table and panel targets (K562, CD4_Rest, HCT116, HEK293T):

  1. essential axis, 600 random targets that are not on the main axis: pooled z from the other six essential-axis lines,
     against line L's own effects (beta, s);
  2. per gene, the precision-weighted slope of L's effect on the pooled z (through the origin), shrunk to the overall slope
     with k = the median information; multiplier m_g = slope_g / overall slope, clipped to [0, 4];
  3. main axis: L held out, the uploaded prediction (z x1.5 from the other main lines) times m_g ** alpha; proxy.est on L.

    python experiments/failed/h45_line_adapter.py

Verdict (EXPERIMENT_LOG H45): strength 0.5 / 1: K562 +0.002 / +0.002, HCT116 +0.001 / +0.002, CD4 -0.003 / -0.002,
HEK293T -0.003 / -0.010: how a line's genes respond to essential knockdowns does not carry over to the panel targets.
Was eda/53_line_adapter.py. Snapshot abc (the A/B/C validation tables). 5 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.etl.stage1.table import load
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel
from vcell.model import pool

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
ESS = ["K562", "CD4_Rest", "HCT116", "HEK293T", "jurkat", "hepg2", "rpe1"]
rng = np.random.default_rng(0)
first = load("ess_K562")
assert list(first["genes"]) == list(X.genes), "essential and main axes differ in genes"
main_t = set(X.targets)
cand = np.flatnonzero([t not in main_t for t in first["targets"]])
pick = np.sort(rng.choice(cand, 600, replace=False))
EB, ES = [], []
for n in ESS:
    t = first if n == "K562" else load(f"ess_{n}")
    EB.append(t["beta"][pick].astype(np.float32))
    ES.append(t["s"][pick].astype(np.float32))
    del t
EB, ES = np.stack(EB).astype(float), np.stack(ES).astype(float)
cpm = X.max_cpm


def adapter(L):
    """Per-gene multiplier for line L from its essential-axis effects against the pooled other lines."""
    li = ESS.index(L)
    tr = [i for i in range(len(ESS)) if i != li]
    fit = pool.predict(EB[tr], ES[tr], cpm, ash=False)
    z = pool.z_effect(fit["v3"]["pred"], fit["v3"]["sd"], fit["v3"]["K"] >= 1)
    b, s = EB[li], ES[li]
    ok = np.isfinite(b) & np.isfinite(s) & (s > 0) & np.isfinite(z)
    w = np.where(ok, 1.0 / np.where(ok, s, 1.0) ** 2, 0.0)
    zz, bb = np.where(ok, z, 0.0), np.where(ok, b, 0.0)
    num, den = (w * zz * bb).sum(0), (w * zz * zz).sum(0)
    overall = num.sum() / den.sum()
    k = np.median(den[den > 0])
    slope = (num + k * overall) / (den + k)
    m = np.clip(slope / overall, 0.0, 4.0)
    return m, overall, float(np.corrcoef(slope[den > 0], den[den > 0])[0, 1])


recs = []
for L in ("K562", "CD4_Rest", "HCT116", "HEK293T"):
    m, overall, _ = adapter(L)
    hi = X.lines.index(L)
    rows = X.rows_for(hi)
    R = X.rungs([i for i in range(5) if i != hi])
    ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
    base = 1.5 * ref * R["v3"] / R["sd"]
    q = np.quantile(m, [0.1, 0.5, 0.9])
    for alpha in (0.0, 0.5, 1.0):
        eff = base * (m ** alpha)[None, :]
        d = X.de(eff, hi, rows, per_target=True)
        recs.append({"line": L, "alpha": alpha, "est": proxy.est(X.pds(eff, hi, rows), d), "pds": X.pds(eff, hi, rows),
                     "fid": np.mean(d["fid"]), "jac": np.mean(d["jac"])})
    print(f"{L}: overall slope {overall:.3f}; multiplier 10/50/90% {np.round(q, 2)}", flush=True)
R = pd.DataFrame(recs)
R.to_csv(config.results("h45_line_adapter.csv"), index=False)
t = R.pivot(index="alpha", columns="line", values="est")
print("\nproxy.est by adapter strength alpha (0 = uploaded)")
print((t - t.loc[0.0]).round(4).to_string())
print("\npds")
print(R.pivot(index="alpha", columns="line", values="pds").round(4).to_string())
