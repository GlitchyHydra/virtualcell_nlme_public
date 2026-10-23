"""H12 (the best scale depends on target strength): Does the best effect scale depend on how many genes a target really moves?

The A/B/C panel is a weak regime (cell_eval2 notes: about half the val targets have fewer than 40 reference-significant
genes, 12 to 30% fewer than 10), while the held-out lines here have medians of 368 to 950. So the proxy results are split by
each target's real significant count (N_conf), and a per-target scale is tried that follows the strength the SOURCE lines
predict for the target (its count of pooled |z| > 3 genes), which needs no information from the held-out line.

    python experiments/h12_effect_regime.py

Verdict (EXPERIMENT_LOG H12): supported: best scale about 1 for 11-150 real DE genes, 1.5 for 151-500, 2-3 above 500; a
single 1.5 is the best compromise for an A/B/C-like mix. The hard cut-offs (z x2, |z| > c) of H8 lose here too, and the
source-predicted strength (count of |z| > 3) tracks the real count only loosely (the basis of H15, rejected).
Was eda/21_regime.py. Snapshot abc (the A/B/C validation tables). 4 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
BINS = [0, 10, 40, 150, 500, 1e9]
LAB = ["<=10", "11-40", "41-150", "151-500", ">500"]
DE_KEYS = ["fid", "jac", "reach", "nmae"]

recs = []
for hi, name in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    R = X.rungs(tr)
    rows = X.rows_for(hi)
    v3, sd = R["v3"], R["sd"]
    ref = np.nanmedian(sd)
    z = v3 / sd
    zs = z * ref
    strength = (np.abs(z) > 3).sum(1)                                  # source-predicted strength per target
    variants = {f"z x{k:g}": k * zs for k in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)}
    variants["v3 x1"] = v3
    for zc in (2.0, 3.0, 4.0):
        variants[f"z x2, |z|>{zc:g}"] = np.where(np.abs(z) > zc, 2 * zs, 0.0)
    variants["zero"] = np.zeros_like(v3)
    for vname, eff in variants.items():
        d = X.de(eff, hi, rows, per_target=True)
        for j, i in enumerate(rows):
            recs.append({"held_out": name, "variant": vname, "target": X.targets[i], "strength": strength[i],
                         **{k: d[k][j] for k in DE_KEYS}, "n_real": d["n_real"][j], "n_ours": d["n_ours"][j]})
    print(f"{name} done", flush=True)

df = pd.DataFrame(recs)
df.to_csv(config.results("h12_effect_regime.csv"), index=False)
real = df[df.variant == "zero"].set_index(["held_out", "target"]).n_real
df["n_conf"] = df.set_index(["held_out", "target"]).index.map(real)
df["bin"] = pd.cut(df.n_conf, BINS, labels=LAB, right=True, include_lowest=True)

print("\ntargets per N_conf bin and line")
print(df[df.variant == "zero"].groupby(["held_out", "bin"], observed=False).size().unstack().reindex(X.lines).to_string())

span = proxy.SPAN
df["de_score"] = sum((df[k] - (1.0 if k == "nmae" else 0.0)) / span[k] for k in DE_KEYS if k != "nmae") \
    + (df.nmae.fillna(1.0) - 1.0) / span["nmae"]
print("\nDE part of the score by N_conf bin (sum of fid, jac, reach, nmae on the leaderboard scale, before / 6; all lines pooled)")
pv = df.groupby(["variant", "bin"], observed=False).de_score.mean().unstack()
print(pv.sort_values("<=10", ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))
for k in DE_KEYS:
    print(f"\n{k} raw by N_conf bin")
    print(df.groupby(["variant", "bin"], observed=False)[k].mean().unstack().to_string(float_format=lambda x: f"{x:.3f}"))
print("\nmedian n_ours by bin")
print(df.groupby(["variant", "bin"], observed=False).n_ours.median().unstack().to_string(float_format=lambda x: f"{x:.0f}"))

print("\ndoes the source-predicted strength track the real count? Spearman per line")
z0 = df[df.variant == "zero"]
for name, d in z0.groupby("held_out"):
    print(f"   {name:9s} {d.strength.corr(d.n_conf, method='spearman'):.2f}")
