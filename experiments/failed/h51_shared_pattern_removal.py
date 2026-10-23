"""H51 (rejected on the leaderboard, L6): Remove the component all target predictions share, so each target's profile stands out (pds).

pds ranks a target's own real profile among all panel targets by cosine; a component common to every prediction makes the
predicted profiles alike. Leaderboard #306 has mse 0 like us but pds 0.767 against our 0.603, most of its lead. Raw v3 minus
the generic response raised pds by 0.02-0.08 on three lines (h11_standardised_effect.py); here on the uploaded z x1.5, leave one line out:
  mean   z - alpha * (mean of z over the scored targets, per gene)
  svd k  z minus alpha * its top k components across targets (uncentred SVD over the scored targets)
proxy.est and the five metrics on the leaderboard scale (h11_standardised_effect.py machinery), five main lines.

    python experiments/failed/h51_shared_pattern_removal.py

Verdict (EXPERIMENT_LOG H51, L6): supported in the proxy (removing the top 3 components +0.010, every line up) and on the
official H1 scorer (+0.010 / +0.017), but the upload scored 0.1683 against 0.1705 for L5: the shared part is context
dependent. Variants: h51b_shared_pattern_harness.py, h51d_shared_pattern_variants.py.
Was eda/58_centering.py. Snapshot abc (the A/B/C validation tables). 3 minutes.
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
X.min_effect = 0.05
recs = []
for hi, name in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    R = X.rungs(tr)
    rows = X.rows_for(hi)
    base_raw = X.evaluate(X.baseline_eff(hi, rows), hi, rows)
    base_raw["pds"] = 0.5
    ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
    z = 1.5 * ref * R["v3"] / R["sd"]
    Zr = np.nan_to_num(z[rows])
    mean = Zr.mean(0)
    U, S, Vt = np.linalg.svd(Zr, full_matrices=False)
    variants = {"uploaded z x1.5": z}
    for alpha in (0.5, 1.0):
        e = z.copy()
        e[rows] = Zr - alpha * mean[None, :]
        variants[f"mean, alpha {alpha:g}"] = e
        for k in (1, 3):
            e = z.copy()
            e[rows] = Zr - alpha * (U[:, :k] * S[:k]) @ Vt[:k]
            variants[f"svd {k}, alpha {alpha:g}"] = e
    share = float((S[:1] ** 2).sum() / (S ** 2).sum())
    for v, eff in variants.items():
        raw = X.evaluate(eff, hi, rows)
        sc = proxy.score(raw, base_raw)
        d = X.de(eff, hi, rows, per_target=True)
        recs.append({"line": name, "variant": v, "est": proxy.est(X.pds(eff, hi, rows), d), **{m: sc[m] for m in proxy.SPAN}})
    print(f"{name}: top component carries {share:.1%} of the prediction's energy", flush=True)
R_ = pd.DataFrame(recs)
R_.to_csv(config.results("h51_shared_pattern_removal.csv"), index=False)
for m in ("est", "pds", "fid", "reach", "jac", "nmae"):
    t = R_.pivot(index="variant", columns="line", values=m)[X.lines]
    t["mean"] = t.mean(1)
    t["mean w/o CD4"] = t.drop(columns="CD4_Rest").iloc[:, :4].mean(1)
    print(f"\n{m}" + (" (proxy.est, change vs uploaded)" if m == "est" else " (leaderboard scale, change vs uploaded)"))
    print((t - t.loc["uploaded z x1.5"]).round(4).to_string())
