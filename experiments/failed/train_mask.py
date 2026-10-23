"""Dropping low-expressed genes from training (rejected): Does leaving low-expressed genes OUT OF TRAINING help? (docs/archive/EDA_PLAN.md 5b side experiment)

    python experiments/failed/train_mask.py [--sources K562 CD4_Rest HCT116 HEK293T]

For each threshold t, source estimates for genes whose control CPM in that source line is <= t are set to NaN (CD4 tables
carry no control CPM and stay as they are), all rungs are refitted, and the SAME fixed gene sets are scored:
  all    genes above 5 CPM in H1 controls (what 09 scores)
  high   those that also have > 20 CPM in every reporting source line (the stable, well-measured genes)
Output: OUT/experiments/stage2_train_mask.csv (median over the benchmark targets).

Verdict (EXPERIMENT_LOG 'What was tried' 6): changes nothing; every gene stays in the fit.
Was eda/13_train_mask.py. Snapshot abc (the A/B/C validation tables). 2 minutes.
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
TRAIN_T = [0, 1, 5, 10, 20]

tabs = {n: load(n) for n in args.sources}
H = load("H1")
targets, genes = list(H["targets"]), list(H["genes"])
bench = [t for t in benchmark_targets() if t in set(targets) and np.isfinite(H["n"][targets.index(t)])]
bi = np.array([targets.index(t) for t in bench])
gl = config.gene_lines().reindex(genes)
tr_beta, tr_s = H["beta"][bi].astype(float), H["s"][bi].astype(float)
h1_ok = np.isfinite(H["ctrl_cpm"]) & (H["ctrl_cpm"] > config.CPM_FILTER)
cpm = {n: np.asarray(t["ctrl_cpm"], float) for n, t in tabs.items()}
has = {n: np.isfinite(c).any() for n, c in cpm.items()}
min_src = np.min(np.stack([c for n, c in cpm.items() if has[n]]), 0)
EVAL = {"all": h1_ok, "high": h1_ok & (min_src > 20)}


def metrics(p, d, y, s):
    sig = np.abs(y) / s > args.z
    k = int(sig.sum())
    r = {"spearman": spearmanr(p, y).correlation if np.ptp(p) > 0 else np.nan}
    nz = sig & (p != 0)
    r["sign_acc"] = float((np.sign(p[nz]) == np.sign(y[nz])).mean()) if nz.sum() >= 5 else np.nan
    r["z_sd"] = float(np.std((y - p) / np.sqrt(d ** 2 + s ** 2)))
    return r


rows = []
for t in TRAIN_T:
    B = np.stack([np.where((cpm[n] > t)[None, :] | (not has[n]), tabs[n]["beta"], np.nan) for n in args.sources]).astype(float)
    S = np.stack([np.where((cpm[n] > t)[None, :] | (not has[n]), tabs[n]["s"], np.nan) for n in args.sources]).astype(float)
    kept = np.isfinite(B).mean()
    preds = pool.predict(B, S, gl.max_cpm.to_numpy())
    for rung in ["v1", "v2", "v3", "v3b"]:
        P, D = preds[rung]["pred"], preds[rung]["sd"]
        for ev, gm in EVAL.items():
            for j, name in enumerate(bench):
                ok = gm & np.isfinite(tr_beta[j]) & np.isfinite(tr_s[j]) & (tr_s[j] > 0)
                m = metrics(np.nan_to_num(P[bi[j]][ok]), np.nan_to_num(D[bi[j]][ok], nan=1.0), tr_beta[j][ok], tr_s[j][ok])
                rows.append({"train_min_cpm": t, "eval_set": ev, "rung": rung, "target": name, **m})
    print(f"train mask {t}: kept {kept:.3f} of estimates", flush=True)
res = pd.DataFrame(rows)
out = res.groupby(["eval_set", "train_min_cpm", "rung"]).median(numeric_only=True).reset_index()
out.to_csv(config.results("train_mask.csv"), index=False)
for ev in EVAL:
    print(f"\n=== scored on '{ev}' genes ({int(EVAL[ev].sum())})")
    print(out[out.eval_set == ev].pivot(index="train_min_cpm", columns="rung", values=["spearman", "sign_acc"]).round(3).to_string())
