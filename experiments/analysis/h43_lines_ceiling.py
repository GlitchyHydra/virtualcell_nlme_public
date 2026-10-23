"""H43 (ceiling of the shared effect): How far can a better estimate of the SHARED effect go? Correlation of the pooled prediction with a held-out
line as a function of the number of source lines k, extrapolated to k -> infinity.

For each held-out main line and random subsets of k of the other lines (K562, CD4_Rest, HCT116, HEK293T, H1, KOLF, Pan_H1),
v3 is pooled from the subset and compared with the held-out effects on the scorer's genes and weights:
    rho_k = sum w p b / sqrt(sum w p^2 * sum w (b^2 - s^2))      (noise-corrected on the held-out side only)
The pooled p is itself noisy, so rho_k rises with k. Spearman-Brown: rho_k = rho_inf * sqrt(R_k), R_k = k r / (1 + (k - 1) r),
fitted per held-out line over (k, mean rho_k). rho_inf is the correlation a perfect estimate of the shared effect (from
infinitely many lines like these) would reach; any model of the shared part, however rich, is bounded by it.

    python experiments/analysis/h43_lines_ceiling.py

Result (EXPERIMENT_LOG H43): no for cell types we do not have: Spearman-Brown ceilings HCT116 0.126, HEK293T 0.060,
K562 0.150; most of each line's response is specific to it. The same cell type is different (Pan_H1 alone 0.310 for
H1), which led to the same-line rule (H44, H47).
Was eda/51_lines_ceiling.py. Snapshot abc (the A/B/C validation tables). 5 minutes.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from vcell import config
from vcell.etl.stage1.table import load
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
for n in ("KOLF", "Pan_H1"):
    t = load(n)
    X = X.add(n, t)
nt = X.not_target
rng = np.random.default_rng(0)
MAX_SUBSETS = 6


def rho(p, hi, rows):
    b, s = X.B[hi][rows][:, nt], X.S[hi][rows][:, nt]
    p = np.nan_to_num(p[rows][:, nt])
    ok = np.isfinite(b) & np.isfinite(s)
    w = np.broadcast_to(X.wts(hi)[nt], b.shape)[ok]
    p, b, s = p[ok], b[ok], s[ok]
    return float((w * p * b).sum() / np.sqrt((w * p * p).sum() * (w * (b * b - s * s)).sum()))


recs = []
for hi in range(5):
    rows = X.rows_for(hi)
    others = [i for i in range(len(X.lines)) if i != hi]
    for k in range(1, len(others) + 1):
        subs = list(itertools.combinations(others, k))
        if len(subs) > MAX_SUBSETS:
            subs = [subs[i] for i in rng.choice(len(subs), MAX_SUBSETS, replace=False)]
        for sub in subs:
            R = X.rungs(list(sub))
            recs.append({"line": X.lines[hi], "k": k, "sources": "+".join(X.lines[i] for i in sub),
                         "rho": rho(R["v3"], hi, rows)})
    print(f"{X.lines[hi]} done", flush=True)
D = pd.DataFrame(recs)
D.to_csv(config.results("h43_lines_ceiling.csv"), index=False)


def sb(k, rho_inf, r):
    return rho_inf * np.sqrt(k * r / (1 + (k - 1) * r))


print("\nmean rho by number of source lines k, and the Spearman-Brown extrapolation to infinitely many lines")
M = D.groupby(["line", "k"]).rho.mean().unstack()
out = []
for line, row in M.iterrows():
    ks, ys = row.index.to_numpy(float), row.to_numpy()
    try:
        (ri, r), _ = curve_fit(sb, ks, ys, p0=[0.3, 0.3], bounds=([0, 1e-4], [1, 1]))
    except RuntimeError:
        ri, r = np.nan, np.nan
    out.append({"line": line, **{f"k={int(k)}": y for k, y in zip(ks, ys)}, "rho_inf": ri, "r_single": r,
                "best single source": D[(D.line == line) & (D.k == 1)].sort_values("rho").iloc[-1].sources})
O = pd.DataFrame(out).set_index("line")
pd.set_option("display.width", 220)
print(O.round(3).to_string())
print("\nraw mse at the best scale for rho is 1 - rho^2; with rho_inf: "
      + ", ".join(f"{l} {1 - v ** 2:.3f}" for l, v in O.rho_inf.items()))
