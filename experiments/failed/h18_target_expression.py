"""H18 (rejected): Does a target's own expression in a line predict how strongly its knockdown changes that line?

CRISPRi cannot lower a gene the cell does not express, so a target silent in the new line should have almost no effect.
This uses only the new line's control cells, which we have for A/B/C and will have for D/E/F. Per held-out line: the real
DE count (proxy, 400 cells) and the correlation of predicted with real effects, by the target's control CPM in that line.

    python experiments/failed/h18_target_expression.py

Verdict (EXPERIMENT_LOG H18): a target's own expression in the new line does not predict its knockdown strength
(Spearman -0.02 to 0.18 with the real DE count; only 1-9 panel targets below 5 CPM). Side finding: the per-target
correlation of predicted with real effects is only 0.01-0.07.
Was eda/24_target_expr.py. Snapshot abc (the A/B/C validation tables). 2 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vcell import config, io
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
gi = {g: i for i, g in enumerate(X.genes)}
CUTS = [-1, 1, 5, 20, 100, 1e9]
LAB = ["<1", "1-5", "5-20", "20-100", ">100"]
recs = []
for hi, name in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    rows = X.rows_for(hi)
    R = X.rungs(tr)
    eff = 1.5 * np.nanmedian(np.where(R["has"], R["sd"], np.nan)) * R["v3"] / R["sd"]
    d = X.de(eff, hi, rows, per_target=True)
    cpm = X.ctrl_cpm(hi)
    testable = cpm > 5
    for k, i in enumerate(rows):
        t = X.targets[i]
        m = testable & np.isfinite(X.B[hi, i]) & (np.arange(len(X.genes)) != gi.get(t, -1))
        r = np.corrcoef(eff[i, m], X.B[hi, i, m])[0, 1]
        recs.append({"line": name, "target": t, "target_cpm": cpm[gi[t]] if t in gi else np.nan,
                     "n_real": d["n_real"][k], "r_pred_real": r})
df = pd.DataFrame(recs)
df["bin"] = pd.cut(df.target_cpm, CUTS, labels=LAB)
df.to_csv(config.results("h18_target_expression.csv"), index=False)
print("targets per line and target-CPM bin (missing CPM = target not in the gene universe):")
print(df.pivot_table(index="line", columns="bin", values="target", aggfunc="count", observed=False).to_string())
print("\nmedian real DE count:")
print(df.pivot_table(index="line", columns="bin", values="n_real", aggfunc="median", observed=False).to_string())
print("\nmedian correlation of predicted (z x1.5) and real effect:")
print(df.pivot_table(index="line", columns="bin", values="r_pred_real", aggfunc="median", observed=False).round(3).to_string())
print("\nSpearman(target CPM, real DE count) per line:")
print(df.groupby("line")[["target_cpm", "n_real"]].apply(lambda g: g.corr("spearman").iloc[0, 1]).round(3).to_string())
for ctx in "ABC":
    X0, _, gn = io.load_context(ctx)
    lib = np.asarray(X0.sum(1)).ravel()
    cp = np.asarray((X0.multiply(1e6 / lib[:, None])).mean(0)).ravel()
    s = pd.Series(cp, index=list(gn)).reindex(config.panel_targets())
    print(f"context {ctx}: panel targets by own CPM: " + ", ".join(f"{l} {n}" for l, n in
          zip(LAB, pd.cut(s, CUTS, labels=LAB).value_counts(sort=False).to_numpy())))
