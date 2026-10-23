"""The mse proxy (evaluation.mse_proxy) and H6: A fast proxy for the scorer's expression-error metric (expr_mse_unbiased_capped_norm), to tune shrinkage without
generating cells.

The scorer compares pseudobulk profiles b = log(1 + 50000 * P / sum(P)); the effect is b_pert - b_ctrl. To first order a log2 fold
change beta on a gene with control expression x (per 50,000 counts, = CPM / 20) moves b by ln2 * x / (1 + x) * beta, so

    N = sum_pairs w_g * ((pred - beta)^2 - s^2)        D = sum_pairs w_g * (beta^2 - s^2)        ratio = N / D
    w_g = (ln2 * x_g / (1 + x_g))^2

where beta, s are the held-out line's Stage 1 estimate and its standard error (the s^2 term is the noise correction). Ratio 1 is the
same as predicting no effect.

    python experiments/analysis/mse_proxy_validation.py            # validate on H1 (scorer: zero 1.00, v1 1.85, v3 3.23, v3b 1.73), then analyse

Result (EXPERIMENT_LOG 'Proxy toolkit', H6): proxy 1.00 / 1.52 / 2.86 vs scorer 1.00 / 1.85 / 3.23 for zero / v1 / v3
on H1. H6 (scale effects below 1 to bring raw mse under 1) deprioritised: the best global scale is about 0.05 and the
best reachable ratio 0.98-1.00; mse is clamped at 0, so the metric is worth under 0.01 overall.
Was eda/17_mse_proxy.py. Snapshot abc (the A/B/C validation tables). 1 minute.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.etl.stage1.table import load, benchmark_targets
from vcell.model import pool

config.use_snapshot("abc")
warnings.filterwarnings("ignore")
LINES = ["K562", "CD4_Rest", "HCT116", "HEK293T", "H1"]
tabs = {n: load(n) for n in LINES}
genes = list(tabs["K562"]["genes"])
targets = list(tabs["K562"]["targets"])
for n, t in tabs.items():
    assert list(t["genes"]) == genes and list(t["targets"]) == targets
B = np.stack([tabs[n]["beta"] for n in LINES]).astype(float)
S = np.stack([tabs[n]["s"] for n in LINES]).astype(float)
gl = config.gene_lines().reindex(genes)
max_cpm = gl.max_cpm.to_numpy()
print(f"{len(LINES)} lines, {len(targets)} targets, {len(genes)} genes", flush=True)


def rungs(Bt, St):
    """v1 and v3 (no prior) from the training lines, as in stage2.predict, plus the pooled variance of v3."""
    g, gsd = pool.generic(Bt, St)
    T, G = Bt.shape[1:]
    v1 = np.broadcast_to(g, (T, G)).copy()
    P0 = pool.pool(Bt, St, tau2=0.0)
    t2 = pool.tau2_by_bin(P0["Q"], P0["K"], P0["C"], pool.gene_bins(max_cpm))
    P = pool.pool(Bt, St, tau2=t2[None, :])
    has = P0["K"] >= 1
    v3 = np.where(has, P["theta"], v1)
    var3 = np.where(has, P["var"], np.nan)
    return {"g": g, "v1": v1, "v3": v3, "var3": var3, "tau2": t2, "K": P["K"], "has": has}


def weights(line_ix):
    cpm = tabs[LINES[line_ix]]["ctrl_cpm"].astype(float)
    cpm = np.where(np.isfinite(cpm), cpm, max_cpm)
    x = cpm / 20.0
    return (np.log(2) * x / (1 + x)) ** 2


def parts(pred, beta, s, w, rows):
    """(sum of w * pred * beta, sum of w * pred^2, D) over the measured pairs of the chosen target rows."""
    b, ss = beta[rows], s[rows]
    ok = np.isfinite(b) & np.isfinite(ss) & np.isfinite(pred[rows])
    W = np.broadcast_to(w, b.shape)
    D = (W * (b ** 2 - ss ** 2))[ok].sum()
    return ok, W, D


def ratio(pred, beta, s, w, rows):
    ok, W, D = parts(pred, beta, s, w, rows)
    p, b, ss = pred[rows], beta[rows], s[rows]
    N = (W * ((p - b) ** 2 - ss ** 2))[ok].sum()
    return N / D


def best_scale(pred, beta, s, w, rows):
    """Optimal c in c * pred (weighted least squares; the noise term does not depend on c) and the ratio it reaches."""
    ok, W, D = parts(pred, beta, s, w, rows)
    p, b = pred[rows], beta[rows]
    cross = (W * p * b)[ok].sum()
    sq = (W * p ** 2)[ok].sum()
    c = cross / sq
    return c, 1.0 - c * cross / D


# ------------------------------------------------------------------ A. validation on H1 (held out)
h = LINES.index("H1")
train = [i for i in range(len(LINES)) if i != h]
R = rungs(B[train], S[train])
bench = set(benchmark_targets())
rows = np.array([i for i, t in enumerate(targets) if t in bench and np.isfinite(B[h, i]).any()])
w = weights(h)
print(f"\nA. H1 held out, {len(rows)} benchmark targets  (scorer: control 1.00, v1 1.85, v3 3.23, v3b 1.73)")
for name in ["v1", "v3"]:
    print(f"   proxy {name}: {ratio(R[name], B[h], S[h], w, rows):.2f}")
print(f"   proxy zero: {ratio(np.zeros_like(R['v3']), B[h], S[h], w, rows):.2f}")

# ------------------------------------------------------------------ B. how large is each part, and where
print("\nB. H1: where the excess of v3 comes from")
g = R["g"]
for lab, pred in [("v1 (generic only)", R["v1"]), ("v3 minus generic", R["v3"] - g[None, :]), ("v3", R["v3"])]:
    print(f"   {lab:22s} ratio {ratio(pred, B[h], S[h], w, rows):.2f}   best scale c={best_scale(pred, B[h], S[h], w, rows)[0]:.2f}"
          f"  -> ratio {best_scale(pred, B[h], S[h], w, rows)[1]:.3f}")
ok, W, D = parts(R["v3"], B[h], S[h], w, rows)
Nij = (W * ((R["v3"][rows] - B[h][rows]) ** 2 - S[h][rows] ** 2))
Nij = np.where(ok, Nij, 0.0)
cpm_h = np.where(np.isfinite(tabs["H1"]["ctrl_cpm"]), tabs["H1"]["ctrl_cpm"], max_cpm)
bins = np.digitize(cpm_h, [5, 20, 100, 500, 2000])
lab = ["5-", "5-20", "20-100", "100-500", "500-2000", "2000+"]
Dij = np.where(ok, W * (B[h][rows] ** 2 - S[h][rows] ** 2), 0.0)
print("   by control CPM bin   share of D   N/D within bin")
for k in range(6):
    m = bins == k
    print(f"   {lab[k]:>10s}   {Dij[:, m].sum() / Dij.sum():8.3f}   {Nij[:, m].sum() / max(Dij[:, m].sum(), 1e-12):8.2f}")
flat = np.sort(Nij.ravel())[::-1]
print(f"   top 1% of pairs carry {flat[:len(flat) // 100].sum() / Nij.sum():.2f} of N; top 10% {flat[:len(flat) // 10].sum() / Nij.sum():.2f}")

# ------------------------------------------------------------------ C. leave-one-line-out: how stable is the best scale
print("\nC. leave one line out: best global scale c and the ratio it reaches (v3), and the ratio at c = 1")
print("   held-out line   targets   ratio c=1   best c   ratio at best c   generic-only ratio")
res = {}
for hi, name in enumerate(LINES):
    tr = [i for i in range(len(LINES)) if i != hi]
    Rl = rungs(B[tr], S[tr])
    rws = np.array([i for i in range(len(targets)) if np.isfinite(B[hi, i]).sum() > 100])
    if name == "H1":
        rws = rows
    wl = weights(hi)
    c, r_best = best_scale(Rl["v3"], B[hi], S[hi], wl, rws)
    res[name] = (Rl, rws, wl)
    print(f"   {name:13s} {len(rws):8d}   {ratio(Rl['v3'], B[hi], S[hi], wl, rws):9.2f}   {c:6.2f}   {r_best:15.3f}"
          f"   {ratio(Rl['v1'], B[hi], S[hi], wl, rws):10.2f}")

# ------------------------------------------------------------------ D. shrink toward zero or toward the generic response
print("\nD. H1: two-term fit  beta ~ a * generic + b * (v3 - generic)  (weighted least squares)")
for name in LINES:
    hi = LINES.index(name)
    Rl, rws, wl = res[name]
    ok, W, Dd = parts(Rl["v3"], B[hi], S[hi], wl, rws)
    gg = np.broadcast_to(Rl["g"], B[hi].shape)[rws][ok]
    sp_ = (Rl["v3"] - Rl["g"][None, :])[rws][ok]
    b = B[hi][rws][ok]
    ww = W[ok]
    X = np.stack([gg, sp_], 1)
    A = (X * ww[:, None]).T @ X
    coef = np.linalg.solve(A, (X * ww[:, None]).T @ b)
    pred = X @ coef
    Nn = (ww * ((pred - b) ** 2)).sum() - (ww * (S[hi][rws][ok]) ** 2).sum()
    print(f"   {name:9s} a={coef[0]:5.2f} (generic)  b={coef[1]:5.2f} (target-specific)  ratio {Nn / Dd:.3f}")

# ------------------------------------------------------------------ E. per-pair reliability: slope by predictive-variance bin
print("\nE. slope of beta on v3 by predictive variance of the pooled estimate (pooled over all five held-out lines)")
rowsE = []
for name in LINES:
    hi = LINES.index(name)
    Rl, rws, wl = res[name]
    v = Rl["var3"][rws] + Rl["tau2"][None, :]
    ok = np.isfinite(B[hi][rws]) & np.isfinite(v) & np.isfinite(Rl["v3"][rws]) & Rl["has"][rws]
    W = np.broadcast_to(wl, ok.shape)
    rowsE.append(pd.DataFrame({"line": name, "p": Rl["v3"][rws][ok], "b": B[hi][rws][ok], "v": v[ok], "w": W[ok],
                               "K": Rl["K"][rws][ok]}))
E = pd.concat(rowsE)
E["bin"] = pd.qcut(E.v, 6, duplicates="drop")
print("   bin of predictive variance                slope   share of pairs")
for bn, d in E.groupby("bin", observed=True):
    sl = (d.w * d.p * d.b).sum() / (d.w * d.p ** 2).sum()
    print(f"   {str(bn):40s} {sl:6.2f}   {len(d) / len(E):.3f}")
print("   by number of training lines K")
for k, d in E.groupby("K"):
    print(f"   K={int(k)}  slope {(d.w * d.p * d.b).sum() / (d.w * d.p ** 2).sum():5.2f}   share {len(d) / len(E):.3f}")
