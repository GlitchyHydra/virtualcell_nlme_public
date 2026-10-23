"""H3 / H4 (add a line): Add-a-line test (docs/archive/EDA_PLAN.md 5b). Hold one line out as the truth, fit the others with and without H1.

    python experiments/analysis/h03_h04_add_line.py [--truth HCT116 HEK293T]

For each held-out line the sources are K562, CD4_Rest and the other X-Atlas line, optionally plus H1 (H1 carries only the
150 targets of its benchmark panel, so it can help only there). Scored on genes above 5 CPM in the held-out line's controls,
for (a) every target the truth line has and (b) the targets H1 covers. Output: OUT/experiments/stage2_add_line.csv
(median over targets). Rungs as in analysis/h03_ladder_first_proxy.py; responders are |beta| / s > 3 in the truth.

Result (EXPERIMENT_LOG H3, H4): each line adds a little (H3 weakly supported); H1 is neutral (differences at most
0.005), so it stays in the final fit.
Was eda/14_add_line.py. Snapshot abc (the A/B/C validation tables). 2 minutes.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vcell import config
from vcell.etl.stage1.table import load
from vcell.model import pool

config.use_snapshot("abc")
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--truth", nargs="+", default=["HCT116", "HEK293T"])
ap.add_argument("--z", type=float, default=3.0)
args = ap.parse_args()

ALL = ["K562", "CD4_Rest", "HCT116", "HEK293T"]
tabs = {n: load(n) for n in ALL + ["H1"]}
genes = list(tabs["H1"]["genes"])
gene_cpm = config.gene_lines().reindex(genes).max_cpm.to_numpy()
h1_cov = np.isfinite(tabs["H1"]["beta"]).any(1)


def metrics(p, d, y, s):
    sig = np.abs(y) / s > args.z
    k = int(sig.sum())
    r = {"n_resp": k, "spearman": spearmanr(p, y).correlation if np.ptp(p) > 0 else np.nan}
    nz = sig & (p != 0)
    r["sign_acc"] = float((np.sign(p[nz]) == np.sign(y[nz])).mean()) if nz.sum() >= 5 else np.nan
    r["prec_lift"] = float(sig[np.argsort(-np.abs(p) / np.maximum(d, 1e-9))[:k]].mean() - k / len(y)) if k >= 5 else np.nan
    r["z_sd"] = float(np.std((y - p) / np.sqrt(d ** 2 + s ** 2)))
    return r


rows = []
for truth in args.truth:
    T = tabs[truth]
    gm = np.isfinite(T["ctrl_cpm"]) & (T["ctrl_cpm"] > config.CPM_FILTER)
    tb, ts = T["beta"].astype(float), T["s"].astype(float)
    has_truth = np.isfinite(tb).sum(1) > 1000
    others = [n for n in ALL if n != truth]
    for label, srcs in [("without H1", others), ("with H1", others + ["H1"])]:
        B = np.stack([tabs[n]["beta"] for n in srcs]).astype(float)
        S = np.stack([tabs[n]["s"] for n in srcs]).astype(float)
        preds = pool.predict(B, S, gene_cpm)
        for rung in ["v1", "v2", "v3", "v3b"]:
            P, D = preds[rung]["pred"], preds[rung]["sd"]
            for i in np.flatnonzero(has_truth):
                ok = gm & np.isfinite(tb[i]) & np.isfinite(ts[i]) & (ts[i] > 0)
                m = metrics(np.nan_to_num(P[i][ok]), np.nan_to_num(D[i][ok], nan=1.0), tb[i][ok], ts[i][ok])
                rows.append({"truth": truth, "sources": label, "rung": rung, "h1_target": bool(h1_cov[i]), "target": tabs[truth]["targets"][i], **m})
        print(f"{truth}: {label} done ({len(srcs)} lines)", flush=True)
res = pd.DataFrame(rows)
res = res[res.n_resp >= 5]
out = res.groupby(["truth", "h1_target", "sources", "rung"]).agg(
    targets=("target", "size"), spearman=("spearman", "median"), sign_acc=("sign_acc", "median"),
    prec_lift=("prec_lift", "median"), z_sd=("z_sd", "median")).reset_index()
out.to_csv(config.results("h03_h04_add_line.csv"), index=False)
for (tr, h), d in out.groupby(["truth", "h1_target"]):
    print(f"\n=== truth {tr}; targets {'covered by H1' if h else 'not covered by H1'} ({int(d.targets.iloc[0])})")
    print(d.pivot(index="rung", columns="sources", values=["spearman", "sign_acc", "prec_lift"]).round(3).to_string())
