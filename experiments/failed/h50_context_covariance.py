"""H50 (rejected): Does a context's own control-cell co-variation predict its knockdown responses beyond the pool?

Linear response: lowering target t shifts gene g by about slope(g on t) x the drop of t, the slope measured across the
context's unperturbed cells. A new context's control cells are the only line-specific information we have, so this is the
one honest signal the cross-line ceiling (H43) does not cover. Test on H1, the one line with both control cells
(data/h1_benchmark/h1_controls.h5ad) and real effects (stage1_H1, the 126 benchmark targets):
  metacells   log1p(1e4) expression, PCA 30 on the 3,000 most variable genes, k-means into about n / k groups, summed counts
  cov         -slope of each gene's log2 CPM on the target's across metacells (knock-down lowers positively co-varying genes)
  pool        z from the four other lines (proxy.rungs, H1 held out), the uploaded model's direction
Score per target: cosine with the real H1 log2FC over response genes, weighted as the scorer's pseudobulk space (proxy.wts).
Null: the cov predictor of another target (target labels shuffled), which keeps any generic cell-state axis.

    python experiments/failed/h50_context_covariance.py [--k 30]

Verdict (EXPERIMENT_LOG H50): natural co-variation in H1 control cells says nothing about these knock-downs: weighted
cosine -0.012 / -0.012 / -0.003 vs a shuffled null -0.012 / -0.007 / +0.006 (pool 0.096); adding it only lowers the pool.
Was eda/57_context_covariance.py. Snapshot abc (the A/B/C validation tables). 5 minutes.
"""
from __future__ import annotations

import argparse
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from vcell import config
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, nargs="+", default=[15, 30, 60], help="cells per metacell")
ap.add_argument("--resid", type=int, default=0, help="remove this many leading components of the metacell matrix first "
                                                     "(cell size, cycle and other shared cell-state axes)")
ap.add_argument("--top", type=int, default=0, help="keep only each target's top genes by |slope| (0 = all)")
args = ap.parse_args()
HI = X.lines.index("H1")
rows = X.rows_for(HI)
tnames = [X.targets[i] for i in rows]
nt = X.not_target
w = X.wts(HI)

ctrl = ad.read_h5ad(config.REPO / "data" / "h1_benchmark" / "h1_controls.h5ad")
Xc = ctrl.X.tocsr().astype(np.float32)
cg = {g: j for j, g in enumerate(ctrl.var_names.astype(str))}
col = np.array([cg.get(g, -1) for g in X.genes])
lib = np.asarray(Xc.sum(1)).ravel()
L = sp.diags(1e4 / np.maximum(lib, 1)) @ Xc
L.data = np.log1p(L.data)
mu = np.asarray(L.mean(0)).ravel()
var = np.asarray(L.multiply(L).mean(0)).ravel() - mu ** 2
hv = np.argsort(var)[::-1][:3000]
pcs = PCA(30, random_state=0).fit_transform(L[:, hv].toarray() - mu[hv])
print(f"H1 controls {Xc.shape}; {len(rows)} targets; {int((col >= 0).sum())} of {len(X.genes)} axis genes in the file", flush=True)

R = X.rungs([i for i in range(len(X.lines)) if i != HI])
ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
pool = (ref * R["v3"] / R["sd"])[rows]
Y = X.B[HI][rows]


def cosine(p, y):
    ok = np.isfinite(p) & np.isfinite(y) & nt & (w > 0)
    a, b, ww = p[ok], y[ok], w[ok]
    return float((ww * a * b).sum() / np.sqrt((ww * a * a).sum() * (ww * b * b).sum() + 1e-30))


def scale_to(p, ref_):
    """p rescaled to the weighted norm of ref_ (so lambda in pool + lambda * cov means the same for every target)."""
    ok = np.isfinite(p) & np.isfinite(ref_) & nt
    return p * np.sqrt((w[ok] * ref_[ok] ** 2).sum() / max((w[ok] * p[ok] ** 2).sum(), 1e-30))


recs = []
for k in args.k:
    lab = MiniBatchKMeans(n_clusters=max(50, Xc.shape[0] // k), batch_size=4096, n_init=3, random_state=0).fit_predict(pcs)
    G = sp.csr_matrix((np.ones(len(lab)), (lab, np.arange(len(lab)))))
    M = (G @ Xc).toarray()                                              # metacells x genes, summed counts
    M = M[:, np.where(col >= 0, col, 0)]
    M[:, col < 0] = np.nan
    tot = (G @ lib[:, None]).ravel()
    E = np.log2(M / tot[:, None] * 1e6 + 1.0)                           # log2 CPM per metacell, axis genes
    Ec = E - np.nanmean(E, 0)
    if args.resid:
        Z0 = np.nan_to_num(Ec)
        U, Sv, Vt = np.linalg.svd(Z0, full_matrices=False)
        Ec = np.where(np.isfinite(Ec), Z0 - (U[:, :args.resid] * Sv[:args.resid]) @ Vt[:args.resid], np.nan)
    covp = np.full((len(rows), len(X.genes)), np.nan)
    tix = {g: j for j, g in enumerate(X.genes)}
    texpr = np.full(len(rows), np.nan)
    for a, t in enumerate(tnames):
        j = tix.get(t, -1)
        if j < 0 or col[j] < 0:
            continue
        x = Ec[:, j]
        vx = float((x * x).mean())
        texpr[a] = float(np.nanmean(M[:, j] / tot * 1e6))
        if vx > 0:
            covp[a] = -(Ec * x[:, None]).mean(0) / vx
            covp[a, j] = np.nan
            if args.top:
                keep = np.argsort(-np.abs(np.nan_to_num(covp[a])))[:args.top]
                sparse_ = np.zeros(len(X.genes))
                sparse_[keep] = covp[a, keep]
                covp[a] = np.where(np.isfinite(covp[a]), sparse_, np.nan)
    ok_t = np.isfinite(covp).any(1)
    rng = np.random.default_rng(0)
    perm = rng.permutation(np.flatnonzero(ok_t))
    shuf = dict(zip(np.flatnonzero(ok_t), perm))
    for a in np.flatnonzero(ok_t):
        c = scale_to(covp[a], pool[a])
        rec = {"k": k, "target": tnames[a], "target_cpm": texpr[a], "pool": cosine(pool[a], Y[a]), "cov": cosine(covp[a], Y[a]),
               "cov_shuffled": cosine(covp[shuf[a]], Y[a]), "pool_vs_cov": cosine(pool[a], covp[a])}
        for lam in (0.25, 0.5, 1.0):
            rec[f"pool+{lam}cov"] = cosine(pool[a] + lam * c, Y[a])
        recs.append(rec)
    d = pd.DataFrame([r for r in recs if r["k"] == k])
    print(f"\nk {k} ({M.shape[0]} metacells), {len(d)} targets; mean cosine with real H1 effects:", flush=True)
    print(d.drop(columns=["k", "target", "target_cpm"]).mean().round(4).to_string(), flush=True)
    q = pd.qcut(d.target_cpm, 3, labels=["low", "mid", "high"])
    print("by the target's own expression (tertiles of CPM in H1 controls):")
    print(d.groupby(q)[["pool", "cov", "cov_shuffled", "pool+0.5cov"]].mean().round(4).to_string(), flush=True)
pd.DataFrame(recs).to_csv(config.results("h50_context_covariance.csv"), index=False)
