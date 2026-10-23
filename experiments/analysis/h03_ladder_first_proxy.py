"""H3 (first Stage 2 proxy): Score the Stage 2 ladder against held-out H1 effects (a fast proxy for the cell-eval2 benchmark).

    python experiments/analysis/h03_ladder_first_proxy.py                       # sources K562 + CD4+ T (Rest), truth H1 benchmark targets
    python experiments/analysis/h03_ladder_first_proxy.py --sources K562 CD4_Rest CD4_Stim8hr

Needs OUT/experiments/stage1_{K562,CD4_Rest,H1}.npz (vcell.etl.stage1). H1 is never an input of the pooling.

Truth: H1 Stage 1 (ratio of means over all cells of the training file) for the 126 vcc-h1 benchmark targets, on genes
above 5 CPM in H1 controls (the scorer's filter), excluding the target. A gene is a true responder if |beta| / s > 3.
Metrics per target, then the median over targets (and over the targets with at least 20 responders, "strong"):
    spearman   rank correlation of prediction and truth over genes
    skill      1 - SSE(pred) / SSE(0), precision-weighted by 1 / s_truth^2 (above 0 beats predicting no change)
    sign_acc   among true responders with a nonzero prediction: share with the right sign
    sign_yield right-signed true responders over all true responders (the scorer's "yield")
    prec_at_k  of the k = #responders genes ranked top by |pred| / sd, the share that are true responders
    z_sd       sd of (truth - pred) / sqrt(sd^2 + s_truth^2); about 1 if the uncertainty is honest
This is not the contest score: the real thing needs generated cells and `vcc-h1 score` (docs/archive/EDA_PLAN.md).

Result (docs/archive/EDA_PLAN.md section 3): K562 + CD4+ T give rank correlation 0.02-0.04 and sign accuracy 0.52-0.54;
four lines 0.0365 / 0.551 (v1 0.020 / 0.509). Shrinkage (v3b) ranked first in this proxy but lost under the real
scorer (H2, E3): the proxy metrics did not match the scored ones, which led to evaluation.panel.
Was eda/09_eval_stage2.py. Snapshot abc (the A/B/C validation tables). 1 minute.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vcell import config
from vcell.etl.stage1.table import load, benchmark_targets
from vcell.model import pool

config.use_snapshot("abc")
ap = argparse.ArgumentParser()
ap.add_argument("--sources", nargs="+", default=["K562", "CD4_Rest"])
ap.add_argument("--z", type=float, default=3.0, help="responder threshold on |beta| / s in the truth")
ap.add_argument("--big", type=float, default=0.2, help="a 'large' true effect: responder with |beta| above this (log2)")
ap.add_argument("--tag", default="", help="suffix for the output files")
args = ap.parse_args()

tabs = {n: load(n) for n in args.sources}
H = load("H1")
targets, genes = list(H["targets"]), list(H["genes"])
assert all(list(t["targets"]) == targets and list(t["genes"]) == genes for t in tabs.values())
bench = [t for t in benchmark_targets() if t in set(targets) and np.isfinite(H["n"][targets.index(t)])]
bi = np.array([targets.index(t) for t in bench])
print(f"{len(bench)} benchmark targets with H1 truth; sources {args.sources}")

gl = config.gene_lines().reindex(genes)
gene_cpm = gl.max_cpm.to_numpy()
tr_beta, tr_s = H["beta"][bi].astype(float), H["s"][bi].astype(float)
gmask = np.isfinite(H["ctrl_cpm"]) & (H["ctrl_cpm"] > config.CPM_FILTER)
print(f"genes above 5 CPM in H1 controls: {int(gmask.sum())} of {len(genes)}")


def metrics(pred, sd, truth, st):
    ok = gmask & np.isfinite(truth) & np.isfinite(st) & (st > 0)
    p, d, y, s = np.nan_to_num(pred[ok]), np.nan_to_num(sd[ok], nan=1.0), truth[ok], st[ok]
    sig = np.abs(y) / s > args.z
    k = int(sig.sum())
    w = 1.0 / s ** 2
    r = {"n_resp": k}
    r["spearman"] = spearmanr(p, y).correlation if np.ptp(p) > 0 else np.nan
    r["skill"] = 1 - (w * (y - p) ** 2).sum() / (w * y ** 2).sum()
    nz = sig & (p != 0)
    r["sign_acc"] = float((np.sign(p[nz]) == np.sign(y[nz])).mean()) if nz.sum() >= 5 else np.nan
    r["sign_yield"] = float((np.sign(p[sig]) == np.sign(y[sig])).sum() / k) if k >= 5 else np.nan
    if k >= 5:
        top = np.argsort(-np.abs(p) / np.maximum(d, 1e-9))[:k]
        r["prec_at_k"] = float(sig[top].mean())
    else:
        r["prec_at_k"] = np.nan
    r["z_sd"] = float(np.std((y - p) / np.sqrt(d ** 2 + s ** 2)))
    # large effects only: what the fold-change and sign metrics of the scorer care about most
    big = sig & (np.abs(y) > args.big)
    kb = int(big.sum())
    r["n_big"] = kb
    r["pearson"] = float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 else np.nan
    nzb = big & (p != 0)
    r["sign_acc_big"] = float((np.sign(p[nzb]) == np.sign(y[nzb])).mean()) if nzb.sum() >= 5 else np.nan
    if kb >= 5:
        top = np.argsort(-np.abs(p) / np.maximum(d, 1e-9))[:kb]
        r["prec_at_k_big"] = float(big[top].mean())
        r["chance_big"] = kb / len(p)
    else:
        r["prec_at_k_big"] = r["chance_big"] = np.nan
    return r


def evaluate(name, line_names):
    B = np.stack([tabs[n]["beta"] for n in line_names]).astype(float)
    S = np.stack([tabs[n]["s"] for n in line_names]).astype(float)
    preds = pool.predict(B, S, gene_cpm)
    preds = {"zero": {"pred": np.zeros_like(preds["v1"]["pred"]), "sd": np.ones_like(preds["v1"]["pred"])}, **preds}
    rows = []
    # how strongly do the source lines themselves see this target? genes with |pooled effect| / sd > 3 (fixed effect)
    zs = np.abs(preds["v2"]["pred"][bi] / preds["v2"]["sd"][bi])
    strength = np.where(gmask[None, :] & (preds["v2"]["K"][bi] >= 1), zs > 3, False).sum(1)
    for rung, p in preds.items():
        for j, t in enumerate(bench):
            m = metrics(p["pred"][bi[j]], p["sd"][bi[j]], tr_beta[j], tr_s[j])
            rows.append({"lines": name, "rung": rung, "target": t, "src_strength": int(strength[j]), **m})
    return pd.DataFrame(rows), preds


frames = []
combos = [(n, [n]) for n in args.sources] + ([("+".join(args.sources), args.sources)] if len(args.sources) > 1 else [])
for name, lines in combos:
    df, preds = evaluate(name, lines)
    frames.append(df)
    if len(lines) == len(args.sources):
        K = preds["v2"]["K"][bi]
        cover = (K >= 1).any(1)
        print(f"\n[{name}] targets with at least one source row: {int(cover.sum())} of {len(bench)}; "
              f"tau2 by expression bin {np.round(preds['v3']['tau2_bin'][[np.flatnonzero(pool.gene_bins(gene_cpm) == b)[0] for b in np.unique(pool.gene_bins(gene_cpm))]], 4).tolist()}; "
              f"prior null weight {preds['v3b']['pi'][0]:.3f}")
res = pd.concat(frames, ignore_index=True)
config.OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(config.results(f"h03_ladder_first_proxy_stage2_eval_by_target{args.tag}.csv"), index=False)

order = ["zero", "v1", "v2", "v3", "v3b"]
pd.set_option("display.width", 170)
def table(sub, cols):
    t = sub.groupby(["lines", "rung"])[cols].median().round(3)
    return t.reindex(pd.MultiIndex.from_product([sub.lines.unique(), order])).to_string()

allc = ["spearman", "pearson", "skill", "sign_acc", "sign_yield", "prec_at_k", "z_sd"]
bigc = ["n_big", "sign_acc_big", "prec_at_k_big", "chance_big"]
print(f"\n=== all {len(bench)} benchmark targets (medians; chance: spearman 0, sign_acc 0.5, skill 0) ===")
print(table(res, allc))
print(f"\n=== large true effects only (|beta| > {args.big} and |z| > {args.z}); chance_big = share of genes that are large ===")
print(table(res, bigc))

full = res[res.lines == "+".join(args.sources)] if len(args.sources) > 1 else res
t0 = full[full.rung == "zero"].set_index("target")
q = pd.qcut(t0.src_strength.rank(method="first"), 4, labels=["weakest", "q2", "q3", "strongest"])
res = res.merge(q.rename("stratum"), left_on="target", right_index=True)
lim = t0.groupby(q, observed=True).src_strength.agg(["min", "max"])
print("\n=== by how strongly the source lines themselves see the target (quartiles of genes with pooled |z| > 3) ===")
print(lim.to_string())
cmb = res[res.lines == ("+".join(args.sources) if len(args.sources) > 1 else args.sources[0])]
for c in ["spearman", "sign_acc", "prec_at_k", "skill"]:
    t = cmb.groupby(["rung", "stratum"], observed=True)[c].median().unstack().reindex(order).round(3)
    print(f"\n{c}:\n" + t.to_string())
res.groupby(["lines", "rung"])[allc + bigc].median().round(4).to_csv(config.results(f"h03_ladder_first_proxy_stage2_eval_summary{args.tag}.csv"))
print("\nsaved stage2_eval_by_target%s.csv and stage2_eval_summary%s.csv" % (args.tag, args.tag))
