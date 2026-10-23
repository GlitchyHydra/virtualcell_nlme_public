"""H19 (rejected): Weight each response gene by how well its effects transfer to a line the pool has not seen.

For a held-out line h, inside the training lines only: for each training line j, predict j from the others (standardised
effect, as uploaded) and measure per gene, across all targets, how well the prediction matches j's real effect:
    r_g      correlation (centred per gene within each line j), pooled over the j folds
    slope_g  least-squares slope of real on predicted, shrunk towards the global slope
The final prediction for h is z x1.5 times a gene weight built from these, resized to the median |effect| of z x1.5, so
only the pattern changes, not the overall size. The diagnostic `inner vs outer` is the Spearman correlation of r_g learned
inside with r_g measured on the held-out line itself: is transferability a stable property of a gene?

    python experiments/failed/h19_gene_transfer.py

Verdict (EXPERIMENT_LOG H19): transferability is a weak but real gene property (inner vs held-out r 0.16-0.27), but every
weighting raises pds and loses more on the DE metrics; best variants 0.141-0.143 vs 0.146.
Was eda/25_gene_transfer.py. Snapshot abc (the A/B/C validation tables). 5 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")


def zs_pred(tr):
    R = X.rungs(tr)
    ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
    return ref * R["v3"] / R["sd"]


def gene_stats(pairs):
    """pairs: list of (pred, real) arrays (T, G). Per gene: centred correlation and slope, pooled over the list."""
    sxy = sxx = syy = n = 0.0
    for p, y in pairs:
        ok = np.isfinite(p) & np.isfinite(y)
        p0 = np.where(ok, p, np.nan)
        y0 = np.where(ok, y, np.nan)
        p0 = np.nan_to_num(p0 - np.nanmean(p0, 0))
        y0 = np.nan_to_num(y0 - np.nanmean(y0, 0))
        sxy = sxy + (p0 * y0).sum(0)
        sxx = sxx + (p0 ** 2).sum(0)
        syy = syy + (y0 ** 2).sum(0)
        n = n + ok.sum(0)
    r = sxy / np.sqrt(sxx * syy + 1e-12)
    glob = sxy.sum() / sxx.sum()
    k = np.median(sxx[sxx > 0])
    slope = (sxy + k * glob) / (sxx + k)
    return r, slope, n


recs = []
for hi, name in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    rows = X.rows_for(hi)
    inner = [(zs_pred([i for i in tr if i != j]), X.B[j]) for j in tr]
    r_in, slope_in, n_in = gene_stats(inner)
    zs = zs_pred(tr)
    r_out, _, n_out = gene_stats([(zs[rows], X.B[hi][rows])])
    ok = (n_in > 50) & (n_out > 50) & np.isfinite(r_in) & np.isfinite(r_out)
    rho = spearmanr(r_in[ok], r_out[ok]).correlation
    print(f"{name}: genes {ok.sum()}, median r inner {np.nanmedian(r_in[ok]):.3f}, outer {np.nanmedian(r_out[ok]):.3f}; "
          f"inner vs outer Spearman {rho:.3f}", flush=True)
    base = 1.5 * zs
    size = np.nanmedian(np.abs(base[rows]))
    rp = np.clip(np.nan_to_num(r_in), 0, None)
    sp_ = np.clip(np.nan_to_num(slope_in), 0, None)
    q = np.quantile(rp[rp > 0], 0.75)
    cap = np.clip(rp / q, 0, 1)                    # 1 for the top quarter of transferring genes, less below, 0 if r <= 0
    weights = {"z x1.5 (reference)": np.ones_like(rp), "w = r+": rp, "w = sqrt(r+)": np.sqrt(rp),
               "w = slope+": sp_, "w = 0.5 + 0.5 r+/median": 0.5 + 0.5 * rp / np.median(rp[rp > 0]),
               "w = 1(r > 0)": (r_in > 0).astype(float),
               "down only: min(1, r+/q75)": cap, "down only: sqrt": np.sqrt(cap), "down only: 0.5 + 0.5 cap": 0.5 + 0.5 * cap,
               "up only: 1 + 0.5 cap": 1 + 0.5 * cap, "up only: 1 + cap": 1 + cap, "up only: 1 + 2 cap": 1 + 2 * cap}
    for wname, w in weights.items():
        eff = base * w[None, :]
        if not wname.startswith(("down only", "up only")):     # resized to the median |effect| of z x1.5; these keep sizes
            eff *= size / max(np.nanmedian(np.abs(eff[rows])), 1e-9)
        d = X.de(eff, hi, rows, per_target=True)
        p = X.pds(eff, hi, rows)
        recs.append({"held_out": name, "variant": wname, "pds": p, "de_mix": proxy.mixed(d), "est": proxy.est(p, d),
                     "fid": np.mean(d["fid"]), "jac": np.mean(d["jac"]), "reach": np.nanmean(d["reach"]),
                     "nmae": np.nanmean(d["nmae"]), "inner_vs_outer": rho})

df = pd.DataFrame(recs)
df.to_csv(config.results("h19_gene_transfer.csv"), index=False)
for col in ["est", "pds", "de_mix", "fid", "jac", "reach", "nmae"]:
    print(f"\n{col}")
    pv = df.pivot_table(index="variant", columns="held_out", values=col, sort=False)[X.lines]
    pv["mean"] = pv.mean(1)
    print(pv.to_string(float_format=lambda x: f"{x:.3f}"))
