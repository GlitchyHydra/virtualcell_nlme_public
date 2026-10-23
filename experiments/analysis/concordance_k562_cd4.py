"""Phase 2 of the EDA: Do the same knockdowns have the same effects in K562 and in CD4+ T cells?

This is the first look at between-line agreement (the quantity tau^2 describes, docs/archive/MODEL_DESIGN.md 3.3) and needs
only small data: the K562 pseudobulk plus the CD4+ T DE statistics for the panel targets, read remotely
(about 0.15 GB of range requests from a 17 GB file; takes a few minutes).

For each panel target present in both:
  K562  beta = log2((CPM_target + 1) / (CPM_nontargeting_pool + 1)), s from the spread of non-targeting pseudobulks
  CD4 T DESeq2 log2 fold change and its standard error (per condition)
and compares them over the genes both measure. A shuffled pairing (K562 target i against CD4 target j) is the null.
Writes OUT/experiments/concordance_k562_cd4.csv.

Result (docs/archive/EDA_PLAN.md section 3): median Spearman 0.01 (shuffled 0.00), only 9 of 259 targets above 0.1;
where both lines call a gene, signs agree 86-100% for the strongest quartile of targets. The effect comes from a few
genes, not the whole profile: the starting point of the pooling model.
Was eda/04_concordance_k562_cd4.py. Snapshot abc (the A/B/C validation tables). Needs network (remote CD4+ T rows). 8 minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import fsspec
import h5py
from scipy.stats import spearmanr

from vcell import config, io

config.use_snapshot("abc")
URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad"
rng = np.random.default_rng(0)
panel = config.panel_targets()

# ---------- K562 effects ----------
b = io.read_replogle_bulk(config.SOURCES / "replogle" / "K562_gwps_raw_bulk_01.h5ad")
X, meta, kg = b["X"], b["meta"], np.array(b["genes"])
cpm = X / X.sum(1, keepdims=True) * 1e6
nt = meta.is_nt.to_numpy() & (meta.cells.to_numpy() >= 20)
w = meta.cells.to_numpy()[nt]
pool = (cpm[nt] * w[:, None]).sum(0) / w.sum()
beta_nt = np.log2((cpm[nt] + 1) / (pool + 1))
c_gene = np.nanstd(beta_nt * np.sqrt(w)[:, None], axis=0)         # sd of beta times sqrt(cells), per gene
tgt = meta[~meta.is_nt].sort_values("cells", ascending=False).drop_duplicates("symbol")
tgt = tgt[tgt.symbol.isin(panel)]
beta_k = np.log2((cpm[tgt.index] + 1) / (pool + 1))
se_k = c_gene[None, :] / np.sqrt(tgt.cells.to_numpy())[:, None]
expressed = pool > config.CPM_FILTER
print(f"K562: {len(tgt)} panel targets, {int(expressed.sum())} genes above 5 CPM in the pool, "
      f"{len(w)} non-targeting pseudobulks used for s")
gidx_k = {g: i for i, g in enumerate(kg)}

# ---------- CD4+ T DE rows (remote) ----------
fs = fsspec.filesystem("https")
with fs.open(URL, "rb", cache_type="blockcache", block_size=256 * 1024) as fo, h5py.File(fo, "r") as f:
    def cat(name):
        x = f["obs"][name]
        cats = np.array(io.decode(x["categories"][...]), dtype=object)
        return cats[x["codes"][...]]
    obs = pd.DataFrame({"symbol": cat("target_contrast_gene_name"), "cond": cat("culture_condition"),
                        "cells": f["obs"]["n_cells_target"][...]})
    v = f["var"]["gene_name"]
    if isinstance(v, h5py.Group) and "categories" in v:
        cd_genes = np.array(io.decode(v["categories"][...]), dtype=object)[v["codes"][...]]
    else:
        cd_genes = np.array(io.decode(v["values"][...] if isinstance(v, h5py.Group) else v[...]), dtype=object)
    sel = obs[obs.symbol.isin(panel)]
    rows = np.sort(sel.index.to_numpy())
    print(f"CD4 T: {len(rows)} perturbation-condition rows for panel targets; reading log_fc and lfcSE ...")
    lfc = np.vstack([f["layers"]["log_fc"][int(r), :] for r in rows])
    lse = np.vstack([f["layers"]["lfcSE"][int(r), :] for r in rows])
sel = obs.loc[rows].reset_index(drop=True)
col_in_k = np.array([gidx_k.get(g, -1) for g in cd_genes])
shared = (col_in_k >= 0)
shared[shared] &= expressed[col_in_k[shared]]
print(f"genes on both axes and above 5 CPM in K562: {int(shared.sum())}")
kcol = col_in_k[shared]


def stats(bk, sk, lf, se, drop_cols):
    """bk, sk over the shared genes (K562) and lf, se (CD4 T) for one target pair."""
    ok = np.isfinite(bk) & np.isfinite(lf) & np.isfinite(se) & (se > 0)
    ok[drop_cols] = False
    if ok.sum() < 200:
        return None
    zc = lf[ok] / se[ok]
    zk = bk[ok] / sk[ok]
    rho = spearmanr(bk[ok], lf[ok]).correlation
    hit_c = np.abs(zc) > 3
    both = hit_c & (np.abs(zk) > 3)
    return {"n_genes": int(ok.sum()), "rho": rho, "n_hits_cd4": int(hit_c.sum()), "n_hits_both": int(both.sum()),
            "sign_agree_cd4hits": float((np.sign(bk[ok][hit_c]) == np.sign(lf[ok][hit_c])).mean()) if hit_c.sum() >= 5 else np.nan,
            "sign_agree_both": float((np.sign(bk[ok][both]) == np.sign(lf[ok][both])).mean()) if both.sum() >= 5 else np.nan}


tpos = {s: i for i, s in enumerate(tgt.symbol)}
out = []
for cond, sub in sel.groupby("cond"):
    syms = [s for s in sub.symbol if s in tpos]
    for i, s in sub[sub.symbol.isin(tpos)].iterrows():
        ti = tpos[s.symbol]
        drop = np.flatnonzero(cd_genes[shared] == s.symbol)          # exclude the target itself
        r = stats(beta_k[ti, kcol], se_k[ti, kcol], lfc[i, shared], lse[i, shared], drop)
        if r is None:
            continue
        # null: the same CD4 row against a different K562 target
        tj = rng.choice([j for j in range(len(tgt)) if j != ti])
        rn = stats(beta_k[tj, kcol], se_k[tj, kcol], lfc[i, shared], lse[i, shared], drop)
        out.append({"condition": cond, "target": s.symbol, "cells_cd4": int(s.cells), "cells_k562": float(tgt.cells.iloc[ti]),
                    "k562_kd": float(tgt.kd.iloc[ti]), **r, "rho_null": rn["rho"] if rn else np.nan})
res = pd.DataFrame(out)
config.OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(config.results("concordance_k562_cd4.csv"), index=False)

print("\n=== agreement of K562 and CD4+ T effects over shared genes (panel targets) ===")
for cond, g in res.groupby("condition"):
    print(f"{cond:9s} targets {len(g):3d}  Spearman rho median {g.rho.median():.3f} (null {g.rho_null.median():.3f}), "
          f"share rho > 0.1: {(g.rho > 0.1).mean():.2f} (null {(g.rho_null > 0.1).mean():.2f})")
print("\nstrong CD4+ T hits (|z| > 3): sign agreement with K562, over targets with at least 5 such genes")
for cond, g in res.groupby("condition"):
    print(f"{cond:9s} targets {g.sign_agree_cd4hits.notna().sum():3d}  median hits {g.n_hits_cd4.median():.0f}  "
          f"sign agreement median {g.sign_agree_cd4hits.median():.2f}  (chance 0.50)")
print("\nrho by the K562 target's knockdown (Rest):")
r = res[res.condition == "Rest"].copy()
r["kd_bin"] = pd.cut(r.k562_kd, [-1, 0.5, 0.8, 1.01])
print(r.groupby("kd_bin", observed=True).agg(targets=("rho", "size"), rho=("rho", "median"), hits=("n_hits_cd4", "median")).round(3).to_string())
print("\nsaved", config.results("concordance_k562_cd4.csv"))
