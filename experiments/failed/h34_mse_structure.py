"""H34 (not reachable): Which structure of prediction could bring the scorer's raw mse under 1, leave one line out.

The scorer's mse baseline is the held-out context's own mean effect given to every target (raw about 0.985 on A/B/C), yet
teams at rank ~245 reach 0.82 to 0.89. mse weights targets by their effect energy, so a target-strength x shared-direction
prediction might beat a flat mean without any target-specific detail. Each variant is fitted to the held-out beta by weighted
least squares (one coefficient per column), on the genes the scorer keeps, with the noise correction of analysis/mse_proxy_validation.py:

    ratio = sum w ((pred - beta)^2 - s^2) / sum w (beta^2 - s^2)        1 = predicting no effect

"fit" lets the held-out line choose its coefficients (an upper bound); "transfer" uses the median coefficients of the other
lines (what an upload would get). Oracle variants use the held-out line's own data and are bounds only.

    python experiments/failed/h34_mse_structure.py

Verdict (EXPERIMENT_LOG H34): no transferable structure brings raw mse under the baseline: z alone 0.978-1.000, plus
source strength x source mean 0.948-1.001, every variant above each line's own oracle baseline; only target strength
transfers (r up to 0.37). mse stays clamped at 0 for honest transfer.
Was eda/42_mse_structure.py. Snapshot abc (the A/B/C validation tables). 3 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
nt = X.not_target


def loading(Bl, wl, d):
    """Per-target weighted projection of one line's effects on direction d, over the genes the scorer keeps."""
    M = np.isfinite(Bl) & nt[None, :] & np.isfinite(d)[None, :]
    Bz, dz = np.where(M, Bl, 0.0), np.where(M, np.nan_to_num(d)[None, :], 0.0)
    den = (wl[None, :] * dz ** 2).sum(1)
    return np.where(den > 0, (wl[None, :] * dz * Bz).sum(1) / np.maximum(den, 1e-30), np.nan)


def design(preds, hi, rows):
    b, s = X.B[hi][rows][:, nt], X.S[hi][rows][:, nt]
    ok = np.isfinite(b) & np.isfinite(s)
    w = np.broadcast_to(X.wts(hi)[nt], b.shape)[ok]
    Xm = np.stack([np.nan_to_num(p[rows][:, nt])[ok] for p in preds], 1)
    return Xm, b[ok], s[ok], w


def ratio(Xm, y, s, w, coef):
    D = (w * (y ** 2 - s ** 2)).sum()
    return float((w * ((Xm @ coef - y) ** 2 - s ** 2)).sum() / D)


def fit(Xm, y, w):
    A = (Xm * w[:, None]).T @ Xm
    return np.linalg.lstsq(A, (Xm * w[:, None]).T @ y, rcond=None)[0]


VARIANTS = ["v3", "z", "generic g", "strength x g", "strength x g + z",
            "ORACLE mean (baseline)", "ORACLE strength x g", "ORACLE strength x mean",
            "src strength x ORACLE mean", "ORACLE strength x src mean", "src strength x src mean",
            "src strength x src mean + z"]
MEANS = {i: np.nanmean(X.B[i][X.rows_for(i)], 0) for i in range(len(X.lines))}   # each line's own shared response
OWN = {i: loading(X.B[i], X.wts(i), MEANS[i]) for i in range(len(X.lines))}       # target strength along it

print("weighted cosine between the lines' own mean responses (genes the scorer keeps), and r of target strength")
for i, a in enumerate(X.lines):
    cs, rs = [], []
    for j in range(len(X.lines)):
        m = nt & np.isfinite(MEANS[i]) & np.isfinite(MEANS[j])
        w = np.sqrt(X.wts(i) * X.wts(j))[m]
        u, v = MEANS[i][m] * np.sqrt(w), MEANS[j][m] * np.sqrt(w)
        cs.append(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
        k = np.isfinite(OWN[i]) & np.isfinite(OWN[j])
        rs.append(np.corrcoef(OWN[i][k], OWN[j][k])[0, 1] if k.sum() > 10 else np.nan)
    print(f"   {a:9s} cos " + " ".join(f"{c:5.2f}" for c in cs) + "    r " + " ".join(f"{r:5.2f}" for r in rs))
res, coefs = {}, {}
for hi, line in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    R = X.rungs(tr)
    rows = X.rows_for(hi)
    g = np.nan_to_num(R["g"])
    T = len(X.targets)
    z = np.where(R["has"], R["v3"] / R["sd"] * np.nanmedian(R["sd"][R["has"]]), 0.0)
    s_src = np.nanmean(np.stack([loading(X.B[l], X.wts(l), g) for l in tr]), 0)       # target strength along g, sources
    sg = np.nan_to_num(s_src)[:, None] * g[None, :]
    m_real = np.nan_to_num(np.nanmean(X.B[hi][rows], 0))
    s_or_g = np.nan_to_num(loading(X.B[hi], X.wts(hi), g))[:, None] * g[None, :]
    s_or_m = np.nan_to_num(loading(X.B[hi], X.wts(hi), m_real))[:, None] * m_real[None, :]
    flat = lambda v: np.broadcast_to(v, (T, len(v)))
    # source mean in the scorer's pseudobulk space (sqrt(w) * log2), brought back to log2 with the held-out line's weight,
    # so genes nearly silent in a source line cannot dominate with huge log2 values
    sw_hi = np.sqrt(X.wts(hi))
    d_src = np.nanmean(np.stack([np.sqrt(X.wts(l)) * MEANS[l] for l in tr]), 0)
    m_src = np.nan_to_num(np.where(sw_hi > 0, d_src / np.maximum(sw_hi, 1e-12), 0.0))
    st_src = np.nan_to_num(np.nanmean(np.stack([OWN[l] for l in tr]), 0))
    st_real = np.nan_to_num(OWN[hi])
    cols_extra = {"src strength x ORACLE mean": [st_src[:, None] * m_real[None, :]],
                  "ORACLE strength x src mean": [st_real[:, None] * m_src[None, :]],
                  "src strength x src mean": [st_src[:, None] * m_src[None, :]],
                  "src strength x src mean + z": [st_src[:, None] * m_src[None, :], z]}
    cols = {**cols_extra,"v3": [R["v3"]], "z": [z], "generic g": [flat(g)], "strength x g": [sg], "strength x g + z": [sg, z],
            "ORACLE mean (baseline)": [flat(m_real)], "ORACLE strength x g": [s_or_g],
            "ORACLE strength x mean": [s_or_m]}
    r = np.corrcoef(np.nan_to_num(s_src)[rows], np.nan_to_num(loading(X.B[hi], X.wts(hi), g))[rows])[0, 1]
    for v in VARIANTS:
        Xm, y, s, w = design(cols[v], hi, rows)
        c = fit(Xm, y, w)
        coefs[(line, v)] = c
        res[(line, v)] = {"fit": ratio(Xm, y, s, w, c), "data": (Xm, y, s, w)}
        if v == "ORACLE mean (baseline)":
            res[(line, v)]["at 1"] = ratio(Xm, y, s, w, np.ones(1))
    print(f"{line}: {len(rows)} targets; strength along g, sources vs real: r = {r:.2f}", flush=True)

rows_out = []
for v in VARIANTS:
    row = {"variant": v}
    for line in X.lines:
        other = [coefs[(l, v)] for l in X.lines if l != line]
        Xm, y, s, w = res[(line, v)]["data"]
        row[f"{line} fit"] = res[(line, v)]["fit"]
        row[f"{line} transfer"] = ratio(Xm, y, s, w, np.median(np.stack(other), 0))
    row["coef median"] = np.round(np.median(np.stack([coefs[(l, v)] for l in X.lines]), 0), 3).tolist()
    rows_out.append(row)
out = pd.DataFrame(rows_out).set_index("variant")
pd.set_option("display.width", 250, "display.max_columns", 30)
print("\nmse ratio (1 = no effect; scorer baseline on A/B/C ~0.985; rank ~245 teams 0.82-0.89)")
print(out.round(3).to_string())
print("\nscorer-style baseline (held-out mean, coefficient 1):",
      {l: round(res[(l, 'ORACLE mean (baseline)')]["at 1"], 3) for l in X.lines})
out.to_csv(X.config.results("h34_mse_structure.csv"))
