"""CD4+ T cells (Zhu et al., genome-scale CRISPRi in primary T cells): DESeq2 log_fc and lfcSE, read remotely row by row
(one table per culture condition), and a control profile from the DESeq2 baseMean."""
from __future__ import annotations

import fsspec
import h5py
import numpy as np
import pandas as pd

from vcell import config, io
from vcell.etl.stage1.table import save

CD4_URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad"


def cd4(targets, genes, conditions=("Rest", "Stim8hr", "Stim48hr"), prefix: str = "CD4") -> None:
    tset = set(targets)
    fs = fsspec.filesystem("https")
    with fs.open(CD4_URL, "rb", cache_type="blockcache", block_size=256 * 1024) as fo, h5py.File(fo, "r") as f:
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
        cix = {g: i for i, g in enumerate(cd_genes)}
        col = np.array([cix.get(g, -1) for g in genes])
        have = col >= 0
        for cond in conditions:
            sub = obs[(obs.cond == cond) & obs.symbol.isin(tset)].drop_duplicates("symbol")
            print(f"CD4 {cond}: {len(sub)} target rows to read", flush=True)
            T, G = len(targets), len(genes)
            beta = np.full((T, G), np.nan)
            s = np.full((T, G), np.nan)
            n = np.full(T, np.nan)
            tix = {t: i for i, t in enumerate(targets)}
            for k, (r, row) in enumerate(sub.iterrows()):
                lf = f["layers"]["log_fc"][int(r), :]
                se = f["layers"]["lfcSE"][int(r), :]
                i = tix[row.symbol]
                beta[i, have] = lf[col[have]]
                s[i, have] = se[col[have]]
                n[i] = row.cells
                if (k + 1) % 100 == 0:
                    print(f"   {k + 1}/{len(sub)}", flush=True)
            save(f"{prefix}_{cond}", targets, genes, beta, s, n, np.full(len(genes), np.nan))


def cd4_ctrl_cpm(genes, cond: str = "Rest", n_rows: int = 30) -> np.ndarray:
    """Control profile for CD4+ T, whose DE file reports none: median DESeq2 baseMean over n_rows target rows of one
    condition (it varies under 1% between rows), as CPM on `genes`, NaN where absent. Saved to eda/out/cd4_<cond>_ctrl_cpm.npy."""
    fs = fsspec.filesystem("https")
    with fs.open(CD4_URL, "rb", cache_type="blockcache", block_size=256 * 1024) as fo, h5py.File(fo, "r") as f:
        x = f["obs"]["culture_condition"]
        conds = np.array(io.decode(x["categories"][...]), dtype=object)[x["codes"][...]]
        v = f["var"]["gene_name"]
        if isinstance(v, h5py.Group) and "categories" in v:
            cd_genes = np.array(io.decode(v["categories"][...]), dtype=object)[v["codes"][...]]
        else:
            cd_genes = np.array(io.decode(v["values"][...] if isinstance(v, h5py.Group) else v[...]), dtype=object)
        rows = np.flatnonzero(conds == cond)
        rows = rows[np.linspace(0, len(rows) - 1, n_rows).astype(int)]
        bm = np.stack([f["layers"]["baseMean"][int(r), :] for r in rows]).astype(float)
    prof = np.nanmedian(bm, 0)
    cpm = prof / np.nansum(prof) * 1e6
    cix = {g: i for i, g in enumerate(cd_genes)}
    out = np.array([cpm[cix[g]] if g in cix else np.nan for g in genes], dtype=np.float32)
    np.save(config.TABLES / f"cd4_{cond.lower()}_ctrl_cpm.npy", out)
    return out

