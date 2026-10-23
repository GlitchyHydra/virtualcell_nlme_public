"""H22 (rejected): Nearest-neighbour prior (docs/archive/IDEAS.md idea 4, the zero-training version).

For a held-out line h, training lines tr: pooled standardised profile zs[t] (z x1, as uploaded) of every target t on the main
axis (575 targets). Target similarity = cosine of the profiles after removing the mean over targets (the generic response).
Prior for t = similarity-weighted average of the profiles of its k most similar OTHER targets. Blend:
    eff = 1.5 * (zs[t] + alpha * prior[t])
alpha = 0 is the current model. Everything uses training lines only; the held-out line enters only as truth.

    python experiments/failed/h22_neighbour_prior.py

Verdict (EXPERIMENT_LOG H22): rejected in the cheap form: 0.146 for z x1.5 vs 0.138 / 0.131 / 0.115 (k 5, alpha 0.25 / 0.5 /
1) and 0.141 / 0.136 / 0.127 (k 60); neighbours dilute the target-specific signal (pds falls).
Was eda/31_neighbors.py. Snapshot abc (the A/B/C validation tables). 3 minutes.
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
recs = []
for hi, name in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    rows = X.rows_for(hi)
    R = X.rungs(tr)
    ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
    zs = ref * R["v3"] / R["sd"]
    testable = X.ctrl_cpm(hi) > 5
    A = np.nan_to_num(zs[:, testable])
    A = A - A.mean(0, keepdims=True)                              # remove the generic response
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    sim = A @ A.T
    np.fill_diagonal(sim, -np.inf)
    for k in (5, 20, 60):
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        w = np.take_along_axis(sim, idx, 1)
        w = np.clip(w, 0, None)
        w = w / (w.sum(1, keepdims=True) + 1e-9)
        prior = (w[:, :, None] * zs[idx]).sum(1)
        cos_top = np.take_along_axis(sim, idx, 1).mean()
        for alpha in (0.0, 0.25, 0.5, 1.0):
            if alpha == 0 and k != 5:
                continue
            eff = 1.5 * (zs + alpha * prior)
            d = X.de(eff, hi, rows, per_target=True)
            p = X.pds(eff, hi, rows)
            recs.append({"held_out": name, "k": k if alpha else 0, "alpha": alpha, "pds": p, "est": proxy.est(p, d),
                         "fid": d["fid"].mean(), "jac": d["jac"].mean(), "reach": np.nanmean(d["reach"]),
                         "nmae": np.nanmean(d["nmae"]), "mean_top_cos": cos_top})
    print(f"{name} done", flush=True)

df = pd.DataFrame(recs)
df.to_csv(config.results("h22_neighbour_prior.csv"), index=False)
for col in ["est", "pds", "fid", "jac", "reach", "nmae", "mean_top_cos"]:
    print(f"\n{col}")
    pv = df.pivot_table(index=["k", "alpha"], columns="held_out", values=col)[X.lines]
    pv["mean"] = pv.mean(1)
    print(pv.to_string(float_format=lambda x: f"{x:.3f}"))
