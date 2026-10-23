"""H13 / H15 (rejected): Line weighting by similarity group, and a per-target scale from the source-predicted strength.

Groups: blood (K562, CD4_Rest) and adherent (HCT116, HEK293T, H1). For a held-out line, source lines of the other group get
precision weight w. Context A looks like a T cell line (blood); B and C look epithelial or stem-like (adherent).
Scores: pds proxy, and the DE part weighted to an A/B/C-like mix of target strengths (N_conf bins 1-10: 0.2, 11-40: 0.3,
41-150: 0.25, 151-500: 0.15, >500: 0.1; targets with N_conf = 0 are left out, as the val panel has none).

    python experiments/failed/h13_h15_line_groups_adaptive_scale.py

Verdict (EXPERIMENT_LOG H13, H15): equal line weights win in four of five held-out lines (H13: blood vs adherent groups
help each other least); a per-target scale from the source-predicted strength is not better than one scale of 1.5 (H15).
Was eda/22_weights.py. Snapshot abc (the A/B/C validation tables). 4 minutes.
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
GROUP = {"K562": "blood", "CD4_Rest": "blood", "HCT116": "adherent", "HEK293T": "adherent", "H1": "adherent"}
BINS = [0.5, 10, 40, 150, 500, 1e9]
LAB = ["1-10", "11-40", "41-150", "151-500", ">500"]
MIX = dict(zip(LAB, [0.2, 0.3, 0.25, 0.15, 0.1]))


def de_score(d):
    return ((d["fid"] - 0.5) / proxy.SPAN["fid"] + d["jac"] / proxy.SPAN["jac"] + d["reach"] / proxy.SPAN["reach"]
            + (np.nan_to_num(d["nmae"], nan=1.0) - 1.0) / proxy.SPAN["nmae"])


def mixed(d):
    """DE score (fid against a coin-flip 0.5, the others against 0 / 1) averaged over bins with the A/B/C-like mix."""
    s = pd.Series(de_score(d))
    b = pd.cut(pd.Series(d["n_real"]), BINS, labels=LAB)
    per = s.groupby(b, observed=False).mean()
    return float(sum(MIX[k] * per[k] for k in LAB if np.isfinite(per[k])))


recs = []
for hi, name in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    rows = X.rows_for(hi)
    for w in (1.0, 0.5, 0.25, 0.1):
        wt = [1.0 if GROUP[X.lines[i]] == GROUP[name] else w for i in tr]
        R = X.rungs(tr, weights=wt)
        sd, v3 = R["sd"], R["v3"]
        ref = np.nanmedian(sd)
        z = v3 / sd
        zs = z * ref
        strength = (np.abs(z) > 3).sum(1).astype(float)
        q = np.nanquantile(strength[rows], [0.5, 0.8])
        k_t = np.where(strength >= q[1], 2.0, np.where(strength >= q[0], 1.5, 1.0))[:, None]
        for vname, eff in {"v3 x1": v3, "z x1": zs, "z x1.5": 1.5 * zs, "z x2": 2 * zs, "z adaptive 1/1.5/2": k_t * zs}.items():
            d = X.de(eff, hi, rows, per_target=True)
            recs.append({"held_out": name, "w_other_group": w, "variant": vname, "pds": X.pds(eff, hi, rows),
                         "de_mix": mixed(d)})
    print(f"{name} done", flush=True)

df = pd.DataFrame(recs)
df.to_csv(config.results("h13_h15_line_groups_adaptive_scale.csv"), index=False)
df["est"] = (df.pds - 0.5) / proxy.SPAN["pds"] / 6 + df.de_mix / 6           # contribution to the overall (mse 0)
for col in ["pds", "de_mix", "est"]:
    print(f"\n{col}")
    pv = df.pivot_table(index=["variant", "w_other_group"], columns="held_out", values=col)[X.lines]
    pv["mean"] = pv.mean(1)
    pv["mean w/o CD4"] = pv.drop(columns="CD4_Rest").iloc[:, :4].mean(1)
    print(pv.sort_values("mean", ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))
