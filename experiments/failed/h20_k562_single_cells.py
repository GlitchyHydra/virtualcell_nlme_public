"""H20 (rejected) and the H5 calibration check: K562 from single cells (exact delta-method s, all guides pooled) against the pseudobulk table, as a source line.

1. Calibration: robust sd of beta / s over all pairs (mostly null effects, so near 1 if s is right), and s and the number
   of cells per target in both tables.
2. As a source: for each other held-out line, z x1.5 from the four sources with K562 = pseudobulk or K562 = single cells.

    python experiments/failed/h20_k562_single_cells.py

Verdict (EXPERIMENT_LOG H20, H5): the single-cell K562 table (stage1_K562sc, exact s) scores 0.145 vs 0.147 for the
pseudobulk; the pseudobulk s was already calibrated (robust sd of beta/s 1.05; single cells 1.09; betas r 0.90).
Was eda/26_k562sc.py. Snapshot abc (the A/B/C validation tables). 3 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.etl.stage1.table import load
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
sc = load("K562sc")
assert list(sc["genes"]) == X.genes and list(sc["targets"]) == X.targets
k = X.lines.index("K562")
for nm, b, s, n in (("pseudobulk", X.B[k], X.S[k], X.NC[k]), ("single cells", sc["beta"], sc["s"], sc["n"])):
    z = (b / s)[np.isfinite(b) & np.isfinite(s)]
    print(f"{nm:13s}: robust sd(beta/s) {np.median(np.abs(z)) / 0.6745:.3f}; median s {np.nanmedian(s):.3f}; "
          f"targets {int(np.isfinite(n).sum())}, median cells {np.nanmedian(n):.0f}; finite pairs {np.isfinite(b).mean():.3f}")
both = np.isfinite(X.B[k]) & np.isfinite(sc["beta"])
print(f"pairs in both: {both.sum()}, correlation of beta {np.corrcoef(X.B[k][both], sc['beta'][both])[0, 1]:.3f}, "
      f"median s ratio sc/bulk {np.median(sc['s'][both] / X.S[k][both]):.3f}")

B0, S0 = X.B.copy(), X.S.copy()
recs = []
for hi, name in enumerate(X.lines):
    if hi == k:
        continue
    tr = [i for i in range(len(X.lines)) if i != hi]
    rows = X.rows_for(hi)
    for src in ("pseudobulk", "single cells"):
        X.B, X.S = B0.copy(), S0.copy()
        if src == "single cells":
            X.B[k], X.S[k] = sc["beta"], sc["s"]
        R = X.rungs(tr)
        X.B, X.S = B0, S0                                  # the held-out truth is always the original table
        eff = 1.5 * np.nanmedian(np.where(R["has"], R["sd"], np.nan)) * R["v3"] / R["sd"]
        d = X.de(eff, hi, rows, per_target=True)
        p = X.pds(eff, hi, rows)
        recs.append({"held_out": name, "K562 from": src, "pds": p, "de_mix": proxy.mixed(d), "est": proxy.est(p, d)})
    print(f"{name} done", flush=True)
df = pd.DataFrame(recs)
df.to_csv(config.results("h20_k562_single_cells.csv"), index=False)
for col in ["est", "pds", "de_mix"]:
    pv = df.pivot_table(index="K562 from", columns="held_out", values=col)
    pv["mean"] = pv.mean(1)
    print(f"\n{col}\n" + pv.to_string(float_format=lambda x: f"{x:.3f}"))
