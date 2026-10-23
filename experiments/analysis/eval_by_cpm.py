"""Scoring only well-expressed genes: The Stage 2 proxy metrics again, but scoring only genes that are expressed in the SOURCE lines too.

    python experiments/analysis/eval_by_cpm.py [--sources K562 CD4_Rest HCT116 HEK293T]

Predictions are the same as in analysis/h03_ladder_first_proxy.py (nothing is refitted); only the set of genes the metrics see changes.
A gene is scored when its control CPM is above `t` in every source line that reports one (CD4 tables carry no control CPM)
and above 5 CPM in the H1 controls. Output: OUT/experiments/stage2_eval_by_cpm.csv (median over the benchmark targets, per t, rung).

Result (EXPERIMENT_LOG 'What was tried' 6): higher CPM raises every score (rank correlation 0.065 pooled vs 0.065-0.08
for v1 above 20 CPM); a diagnostic, not a model change.
Was eda/12_eval_by_cpm.py. Snapshot abc (the A/B/C validation tables). 1 minute.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vcell import config
from vcell.etl.stage1.table import load, benchmark_targets
from vcell.model import pool

config.use_snapshot("abc")
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--sources", nargs="+", default=["K562", "CD4_Rest", "HCT116", "HEK293T"])
ap.add_argument("--z", type=float, default=3.0)
args = ap.parse_args()
THRESH = [0, 5, 10, 20, 50]

tabs = {n: load(n) for n in args.sources}
H = load("H1")
targets, genes = list(H["targets"]), list(H["genes"])
bench = [t for t in benchmark_targets() if t in set(targets) and np.isfinite(H["n"][targets.index(t)])]
bi = np.array([targets.index(t) for t in bench])
gl = config.gene_lines().reindex(genes)
tr_beta, tr_s = H["beta"][bi].astype(float), H["s"][bi].astype(float)
h1_ok = np.isfinite(H["ctrl_cpm"]) & (H["ctrl_cpm"] > config.CPM_FILTER)

src_cpm = np.stack([np.asarray(t["ctrl_cpm"], float) for t in tabs.values()])           # (L, G), NaN where not reported
min_src = np.where(np.isfinite(src_cpm).any(0), np.nanmin(np.where(np.isfinite(src_cpm), src_cpm, np.inf), 0), np.inf)
min_src = np.where(np.isinf(min_src), np.inf, min_src)

B = np.stack([t["beta"] for t in tabs.values()]).astype(float)
S = np.stack([t["s"] for t in tabs.values()]).astype(float)
preds = pool.predict(B, S, gl.max_cpm.to_numpy())


def metrics(p, d, y, s):
    sig = np.abs(y) / s > args.z
    k = int(sig.sum())
    r = {"n_genes": len(y), "n_resp": k}
    r["spearman"] = spearmanr(p, y).correlation if np.ptp(p) > 0 else np.nan
    nz = sig & (p != 0)
    r["sign_acc"] = float((np.sign(p[nz]) == np.sign(y[nz])).mean()) if nz.sum() >= 5 else np.nan
    r["prec_at_k"] = float(sig[np.argsort(-np.abs(p) / np.maximum(d, 1e-9))[:k]].mean()) if k >= 5 else np.nan
    r["chance_prec"] = k / len(y) if k >= 5 else np.nan
    r["z_sd"] = float(np.std((y - p) / np.sqrt(d ** 2 + s ** 2)))
    return r


rows = []
for t in THRESH:
    gmask = h1_ok & (min_src > t)
    for rung in ["v1", "v2", "v3", "v3b"]:
        P, D = preds[rung]["pred"], preds[rung]["sd"]
        for j, name in enumerate(bench):
            ok = gmask & np.isfinite(tr_beta[j]) & np.isfinite(tr_s[j]) & (tr_s[j] > 0)
            m = metrics(np.nan_to_num(P[bi[j]][ok]), np.nan_to_num(D[bi[j]][ok], nan=1.0), tr_beta[j][ok], tr_s[j][ok])
            rows.append({"min_src_cpm": t, "rung": rung, "target": name, **m})
res = pd.DataFrame(rows)
out = res.groupby(["min_src_cpm", "rung"]).median(numeric_only=True).drop(columns="n_resp").reset_index()
out.to_csv(config.results("eval_by_cpm.csv"), index=False)
print("sources:", args.sources)
print(out.round(3).to_string(index=False))
