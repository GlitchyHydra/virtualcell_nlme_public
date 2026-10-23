"""QC of the organisers' control cells (was eda/01_context_summary.py, plus the per-cell audit of EXPERIMENT_LOG F2).

    summary()     per context: depth, genes per cell, balance of the 46 ntc_id groups, genes above the scorer's 5 CPM filter,
                  expression of the panel targets, Spearman between contexts, and the null calibration of the Stage 1
                  standard error (each ntc_id, 400 cells, is a fake perturbation against the other 45; z = beta / s should
                  be N(0, 1)). Writes OUT/context_summary.npz and OUT/null_calibration.csv (for A/B/C; other contexts get
                  a _<contexts> suffix, so the files etl.universe reads are not replaced)
    cell_audit()  per cell: depth, genes, mitochondrial share and 5-MAD outliers (qc.sources.cell_mask rules, one batch).
                  Reported only: the real knockdown cells come through the same pipeline, so the controls the generator
                  resamples are not filtered (F2)

    python -m vcell qc controls [--contexts D E F]     (VCC_CONTROLS or --controls picks the folder)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vcell import config, io, stats
from vcell.qc.sources import REASONS, cell_mask


def summary(contexts=None) -> None:
    config.OUT.mkdir(parents=True, exist_ok=True)
    man = config.manifest()
    contexts = list(contexts or man["contexts"])
    suffix = "" if contexts == ["A", "B", "C"] else "_" + "".join(contexts)   # A/B/C keep the names etl.universe reads
    panel = config.panel_targets()
    genes = config.gene_names()
    gidx = {g: i for i, g in enumerate(genes)}
    bins = [(5, 10), (10, 50), (50, 200), (200, 1000), (1000, np.inf)]
    res, rows_nt = {}, []

    for ctx in contexts:
        counts, obs, var = io.load_context(ctx)
        assert var == genes, "gene order differs from gene_names.csv"
        assert np.all(counts.data == np.round(counts.data)), "X is not integer counts"
        n = counts.shape[0]
        tot = np.asarray(counts.sum(1)).ravel()
        ngenes = np.diff(counts.indptr)
        cpm = stats.cpm_matrix(counts)
        mean_cpm = np.asarray(cpm.mean(0)).ravel()
        detect = np.asarray((counts > 0).sum(0)).ravel() / n
        ntc = obs["ntc_id"].astype(str).to_numpy()
        codes, uniq = pd.factorize(ntc)
        print(f"\n=== context {ctx}: {n} cells x {counts.shape[1]} genes, {len(uniq)} ntc_id, "
              f"{counts.nnz / n:.0f} stored entries per cell (cap about 13,200)")
        print(f"  total UMI per cell  q05/50/95: {np.quantile(tot, [.05, .5, .95]).round(0).tolist()}   "
              f"genes per cell q05/50/95: {np.quantile(ngenes, [.05, .5, .95]).round(0).tolist()}")
        sizes = pd.Series(codes).value_counts()
        print(f"  cells per ntc_id min/median/max: {sizes.min()}/{int(sizes.median())}/{sizes.max()}"
              f"   target_gene values: {sorted(obs['target_gene'].astype(str).unique())}")
        ntc_umi = pd.Series(tot).groupby(codes).median()
        print(f"  median UMI per ntc_id, min/max: {ntc_umi.min():.0f}/{ntc_umi.max():.0f}")
        testable = mean_cpm > config.CPM_FILTER
        pt = np.array([gidx[t] for t in panel if t in gidx])
        print(f"  genes above 5 CPM: {int(testable.sum())}; panel targets found on gene axis: {len(pt)}/300; "
              f"targets at or below 5 CPM in the controls: {int((mean_cpm[pt] <= config.CPM_FILTER).sum())}; "
              f"never detected: {int((detect[pt] == 0).sum())}")

        # Stage 1 null: ntc_id k vs the other 45
        ng, sums, sumsq = stats.group_stats(cpm, codes, len(uniq))
        tot_n, tot_s, tot_q = ng.sum(), sums.sum(0), sumsq.sum(0)
        beta, se = stats.log2fc_and_se(ng[:, None], sums, sumsq,
                                   (tot_n - ng)[:, None], tot_s[None, :] - sums, tot_q[None, :] - sumsq)
        z = beta / se
        for lo, hi in bins:
            m = (mean_cpm > lo) & (mean_cpm <= hi)
            zz = z[:, m][np.isfinite(z[:, m])]
            rows_nt.append({"context": ctx, "cpm_bin": f"{lo}-{hi}", "genes": int(m.sum()),
                            "median_se": float(np.nanmedian(se[:, m])), "sd_z": float(zz.std()),
                            "share_abs_z_gt_1.96": float((np.abs(zz) > 1.96).mean()),
                            "zero_in_group_share": float(np.isnan(beta[:, m]).mean())})
        res[f"{ctx}_mean_cpm"] = mean_cpm
        res[f"{ctx}_detect"] = detect
        res[f"{ctx}_tot"] = tot
        res[f"{ctx}_ngenes"] = ngenes
        res[f"{ctx}_ntc_codes"] = codes
        res[f"{ctx}_null_z"] = z.astype(np.float32)
        res[f"{ctx}_null_se"] = se.astype(np.float32)
        res[f"{ctx}_ntc_levels"] = np.array(uniq)

    nt = pd.DataFrame(rows_nt)
    print("\n=== null calibration of the standard error (fake perturbations = single ntc_id vs the rest) ===")
    print("sd_z near 1 and share near 0.05 mean s is calibrated; larger values mean guide-level or batch structure that s misses")
    print(nt.round(3).to_string(index=False))
    nt.to_csv(config.OUT / f"null_calibration{suffix}.csv", index=False)

    cp = np.array([res[f"{c}_mean_cpm"] for c in contexts])
    print("\n=== testable genes ===")
    print("in each:", (cp > config.CPM_FILTER).sum(1).tolist(), "| in all:", int((cp > config.CPM_FILTER).all(0).sum()),
          "| in at least one:", int((cp > config.CPM_FILTER).any(0).sum()))
    lc = np.log2(cp + 1)
    for a in range(len(contexts)):
        for b in range(a + 1, len(contexts)):
            keep = (cp[a] > 5) | (cp[b] > 5)
            r = pd.Series(lc[a][keep]).corr(pd.Series(lc[b][keep]), method="spearman")
            print(f"  Spearman of log2(CPM+1), contexts {contexts[a]}-{contexts[b]} (genes above 5 CPM in either): {r:.3f}")
    np.savez_compressed(config.OUT / f"context_summary{suffix}.npz", genes=np.array(genes), **res)
    print("\nsaved", config.OUT / f"context_summary{suffix}.npz")


def cell_audit(contexts=None) -> pd.DataFrame:
    """One row per context: cells, depth quantiles, genes per cell, mito share, outliers by reason, cells per ntc_id."""
    contexts = list(contexts or config.manifest()["contexts"])
    genes = config.gene_names()
    mt = np.array([g.startswith("MT-") for g in genes])
    rows = []
    for ctx in contexts:
        X, obs, _ = io.load_context(ctx)
        tot = np.asarray(X.sum(1)).ravel().astype(float)
        ngen = np.diff(X.indptr)
        pct_mt = 100 * np.asarray(X[:, mt].sum(1)).ravel() / np.maximum(tot, 1)
        keep, reason = cell_mask(tot, ngen, pct_mt)
        per_ntc = obs["ntc_id"].astype(str).value_counts()
        r = {"context": ctx, "cells": len(tot), "mt_genes": int(mt.sum()),
             "umi_min": tot.min(), "umi_q05": np.quantile(tot, 0.05), "umi_median": np.median(tot),
             "umi_q95": np.quantile(tot, 0.95), "umi_max": tot.max(), "genes_median": float(np.median(ngen)),
             "mito_median_pct": float(np.median(pct_mt)), "mito_max_pct": float(pct_mt.max()), "flagged": int((~keep).sum()),
             **{f"flag_{n}": int((reason == k).sum()) for k, n in enumerate(REASONS) if k},
             "ntc_ids": len(per_ntc), "cells_per_ntc_min": int(per_ntc.min()), "cells_per_ntc_max": int(per_ntc.max())}
        rows.append(r)
        flags = ", ".join(f"{n} {r['flag_' + n]}" for n in REASONS[1:] if r.get("flag_" + n))
        print(f"{ctx}: {r['cells']} cells, UMI min/median/q95/max {r['umi_min']:.0f}/{r['umi_median']:.0f}/{r['umi_q95']:.0f}/"
              f"{r['umi_max']:.0f}, mito median {r['mito_median_pct']:.2f}% (max {r['mito_max_pct']:.1f}%), flagged "
              f"{r['flagged']} ({flags}), {r['ntc_ids']} ntc_id with {r['cells_per_ntc_min']}-{r['cells_per_ntc_max']} cells",
              flush=True)
    df = pd.DataFrame(rows)
    out = config.OUT / "qc" / f"controls_{''.join(contexts)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("saved", out)
    return df
