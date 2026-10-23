"""H32 (rejected): Non-linear meta-regression (docs/archive/IDEAS.md): let the between-line variance and a calibration slope depend on
covariates through a 5 x 5 grid (a piecewise-constant surface, the moment-based stand-in for splines).

Covariates of a pair (target t, gene g) when predicting line h from sources L:
    e = mean log1p(control CPM) of g over the sources that report it        (how well the gene is measured)
    r = log1p(CPM of g in h's controls) - e                                  (how differently h expresses it; 0 if unknown)
Fitted inside the training lines (leave one out, so h is never used): for each training line j with sources tr \\ {j},
per grid cell
    tau2_cell   = mean[(beta_j - theta)^2 - s_j^2 - var(theta)]   (clipped at 0): between-line variance
    slope_cell  = sum(theta * beta_j) / sum(theta^2)                   : how much of theta carries over
Prediction for h: sd^2 = var + tau2_cell, eff proportional to slope_cell * theta / sd; every variant is resized to the median
|effect| of the uploaded z x1.5, so only the pattern is compared. Grid edges are quantiles of the training pairs.

    python experiments/failed/h32_meta_regression.py

Verdict (EXPERIMENT_LOG H32): the variance surface is neutral (0.145 vs 0.146), the slope surface worse (0.116; both
0.104): re-weighting genes by a least-squares slope costs fold-change accuracy, as v3b and the boosted model did.
Was eda/40_metareg.py. Snapshot abc (the A/B/C validation tables). 4 minutes.
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
NB = 5
CPM = np.stack([X.tabs[n]["ctrl_cpm"] for n in X.lines]).astype(float)


def covariates(L, h):
    lc = np.log1p(CPM[L])
    e = np.nanmean(lc, 0)
    e = np.where(np.isfinite(e), e, np.nanmean(np.log1p(X.max_cpm)))
    ch = np.log1p(CPM[h])
    r = ch - e                                                   # NaN where h reports no control expression (CD4+ T)
    return e, r


def pool_parts(L):
    R1 = X.rungs(L)
    R0 = X.rungs(L, kappa=0.0)                                   # sd without tau^2 = sqrt(var(theta))
    return R1["v3"], R0["sd"] ** 2, R1["sd"], R1["has"]


def cell_index(e, r, eq, rq):
    return np.clip(np.digitize(e, eq), 0, NB - 1) * NB + np.clip(np.digitize(r, rq), 0, NB - 1)


recs = []
for hi, name in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    # --- fit the grid inside the training lines
    E, Rr, TH, VAR, BJ, SJ = [], [], [], [], [], []
    for j in tr:
        inner = [i for i in tr if i != j]
        th, var, _, has = pool_parts(inner)
        e, r = covariates(inner, j)
        rows = np.flatnonzero(np.isfinite(X.B[j]).sum(1) > 100)
        m = has[rows] & np.isfinite(X.B[j][rows]) & np.isfinite(X.S[j][rows]) & (X.ctrl_cpm(j) > 5)[None, :]
        tt, gg = np.nonzero(m)
        tt = rows[tt]
        E.append(e[gg]); Rr.append(r[gg]); TH.append(th[tt, gg]); VAR.append(var[tt, gg])
        BJ.append(X.B[j][tt, gg]); SJ.append(X.S[j][tt, gg])
    E, Rr, TH, VAR, BJ, SJ = map(np.concatenate, (E, Rr, TH, VAR, BJ, SJ))
    eq = np.quantile(E, np.linspace(0, 1, NB + 1)[1:-1])
    known = np.isfinite(Rr)                                      # the r axis is fitted only where r is known
    rq = np.quantile(Rr[known], np.linspace(0, 1, NB + 1)[1:-1])
    erow = np.clip(np.digitize(E, eq), 0, NB - 1)

    def moments(k):
        return (max(0.0, float(np.mean((BJ[k] - TH[k]) ** 2 - SJ[k] ** 2 - VAR[k]))),
                float(np.sum(TH[k] * BJ[k]) / np.sum(TH[k] ** 2)))

    row_t2, row_sl = np.zeros(NB), np.ones(NB)                   # marginal over r (used when r is unknown)
    for a in range(NB):
        row_t2[a], row_sl[a] = moments(erow == a)
    t2 = np.repeat(row_t2, NB)
    sl = np.repeat(row_sl, NB)
    cell = np.where(known, cell_index(E, np.nan_to_num(Rr), eq, rq), -1)
    for c in range(NB * NB):
        k = cell == c
        if k.sum() > 1000:
            t2[c], sl[c] = moments(k)
    # --- predict h
    th, var, sd_ref, has = pool_parts(tr)
    e, r = covariates(tr, hi)
    if np.isfinite(r).any():
        cg = cell_index(e, np.nan_to_num(r), eq, rq)              # (G,) cell of each gene for this new line
    else:                                                         # no control expression: the r-marginal of each row
        cg = np.clip(np.digitize(e, eq), 0, NB - 1) * NB
        t2 = np.repeat(row_t2, NB)
        sl = np.repeat(row_sl, NB)
    rows = X.rows_for(hi)
    ref = np.nanmedian(np.where(has, sd_ref, np.nan))
    base = 1.5 * ref * th / sd_ref
    size = np.nanmedian(np.abs(base[rows]))
    sd_new = np.sqrt(var + t2[cg][None, :])
    variants = {"z x1.5 (uploaded)": base,
                "grid tau2": th / sd_new,
                "grid slope": sl[cg][None, :] * th / sd_ref,
                "grid tau2 + slope": sl[cg][None, :] * th / sd_new}
    for vn, eff in variants.items():
        if vn != "z x1.5 (uploaded)":
            eff = eff * size / np.nanmedian(np.abs(eff[rows]))
        d = X.de(eff, hi, rows, per_target=True)
        p = X.pds(eff, hi, rows)
        recs.append({"held_out": name, "variant": vn, "pds": p, "de_mix": proxy.mixed(d), "est": proxy.est(p, d),
                     "nmae": np.nanmean(d["nmae"])})
    grid = pd.DataFrame({"tau2": t2, "slope": sl}).round(4)
    print(f"{name}: slope range {sl.min():.3f} to {sl.max():.3f}; tau2 range {t2.min():.4f} to {t2.max():.4f}", flush=True)
    print("   slope grid (rows: source expression low to high; columns: new-line expression lower to higher)\n" +
          pd.DataFrame(sl.reshape(NB, NB)).round(3).to_string(), flush=True)

df = pd.DataFrame(recs)
df.to_csv(config.results("h32_meta_regression.csv"), index=False)
for col in ["est", "pds", "de_mix", "nmae"]:
    pv = df.pivot_table(index="variant", columns="held_out", values=col, sort=False)[X.lines]
    pv["mean"] = pv.mean(1)
    print(f"\n{col}\n" + pv.to_string(float_format=lambda x: f"{x:.3f}"))
