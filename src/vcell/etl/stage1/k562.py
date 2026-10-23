"""K562 sources (Replogle 2022): the genome-wide pseudobulk table (the source line used) and the single-cell builder,
which also serves every dense Replogle/Nadig single-cell file (RPE1, HepG2, Jurkat; etl.stage1.essential)."""
from __future__ import annotations

import h5py
import numpy as np
import scipy.sparse as sp

from vcell import config, io, stats
from vcell.etl.stage1.table import save


def k562(targets, genes, min_cpm: float = 1.0, name: str = "K562") -> None:
    b = io.read_replogle_bulk(config.SOURCES / "replogle" / "K562_gwps_raw_bulk_01.h5ad")
    X, meta, kg = b["X"], b["meta"], np.array(b["genes"])
    cpm = X / X.sum(1, keepdims=True) * 1e6
    nt = meta.is_nt.to_numpy() & (meta.cells.to_numpy() >= 20)
    w = meta.cells.to_numpy()[nt]
    pool = (cpm[nt] * w[:, None]).sum(0) / w.sum()
    beta_nt = np.log2((cpm[nt] + 1) / (pool + 1))
    c_gene = np.nanstd(beta_nt * np.sqrt(w)[:, None], axis=0)          # sd of beta times sqrt(cells), per gene
    sel = meta[~meta.is_nt].sort_values("cells", ascending=False).drop_duplicates("symbol")
    row_of = dict(zip(sel.symbol, sel.index))                          # largest pseudobulk per target
    kix = {g: i for i, g in enumerate(kg)}
    col = np.array([kix.get(g, -1) for g in genes])
    have = col >= 0
    detected = np.zeros(len(genes), bool)
    detected[have] = pool[col[have]] >= min_cpm
    T, G = len(targets), len(genes)
    beta = np.full((T, G), np.nan)
    s = np.full((T, G), np.nan)
    n = np.full(T, np.nan)
    kd = np.full(T, np.nan)
    for i, t in enumerate(targets):
        if t not in row_of:
            continue
        r = row_of[t]
        bt = np.log2((cpm[r] + 1) / (pool + 1))
        st = c_gene / np.sqrt(meta.cells[r])
        beta[i, have] = bt[col[have]]
        s[i, have] = st[col[have]]
        n[i], kd[i] = meta.cells[r], meta.kd[r]
    beta[:, ~detected] = np.nan
    s[:, ~detected] = np.nan
    ctrl = np.full(len(genes), np.nan)
    ctrl[have] = pool[col[have]]
    save(name, targets, genes, beta, s, n, ctrl, kd)


def k562_sc(targets, genes, block: int = 10000, name: str = "K562sc",
            path=None) -> None:
    """Single cells from a Replogle/Nadig dense float32 h5ad (K562 genome-wide, RPE1, HepG2, Jurkat; obs gene = target symbol):
    ratio of means and delta-method s, as h1(). CPM uses each cell's full UMI_count, so it matches lines normalised over the
    whole transcriptome. All guides of a target are pooled (the pseudobulk builder takes the largest single guide row).
    Defaults to the K562 genome-wide file."""
    path = path or config.SOURCES / "replogle" / "K562_gwps_raw_singlecell_01.h5ad"
    tix = {t: i for i, t in enumerate(targets)}
    with h5py.File(path, "r") as f:
        cats = np.array(io.decode(f["obs"]["__categories"]["gene"][...]), dtype=object)
        tg = cats[f["obs"]["gene"][...]]
        umi = f["obs"]["UMI_count"][...].astype(float)
        vc = np.array(io.decode(f["var"]["__categories"]["gene_name"][...]), dtype=object)
        kg = vc[f["var"]["gene_name"][...]]
        nt_code = len(targets)
        codes = np.array([tix.get(t, nt_code if "non-targeting" in t.lower() else -1) for t in tg])
        X = f["X"]
        n_cells, n_genes = X.shape
        n = np.zeros(nt_code + 1)
        S = np.zeros((nt_code + 1, n_genes))
        SS = np.zeros((nt_code + 1, n_genes))
        print(f"K562 single cells: {n_cells} cells x {n_genes} genes; {int((codes >= 0).sum())} cells kept "
              f"({int((codes == nt_code).sum())} non-targeting)", flush=True)
        for a in range(0, n_cells, block):
            b = min(a + block, n_cells)
            keep = np.flatnonzero(codes[a:b] >= 0)
            if keep.size:
                m = sp.csr_matrix(X[a:b][keep])
                cpm = (sp.diags(1e6 / np.maximum(umi[a:b][keep], 1)) @ m).tocsr()
                nn, s1, s2 = stats.group_stats(cpm, codes[a:b][keep], nt_code + 1)
                n += nn
                S += s1
                SS += s2
            if (a // block) % 20 == 0:
                print(f"   cells {b}/{n_cells}", flush=True)
    beta_k, se_k = stats.log2fc_and_se(n[:nt_code, None], S[:nt_code], SS[:nt_code], n[nt_code], S[nt_code], SS[nt_code])
    ctrl_k = S[nt_code] / n[nt_code]
    kix = {g: i for i, g in enumerate(kg)}
    col = np.array([kix.get(g, -1) for g in genes])
    have = col >= 0
    T, G = len(targets), len(genes)
    beta = np.full((T, G), np.nan)
    s = np.full((T, G), np.nan)
    beta[:, have] = beta_k[:, col[have]]
    s[:, have] = se_k[:, col[have]]
    nn_t = np.where(n[:nt_code] > 0, n[:nt_code], np.nan)
    beta[~np.isfinite(nn_t)] = np.nan
    ctrl = np.full(G, np.nan)
    ctrl[have] = ctrl_k[col[have]]
    kd = np.full(T, np.nan)
    for t, i in tix.items():
        if t in kix and n[i] > 0 and ctrl_k[kix[t]] > 0:
            kd[i] = 1 - (S[i, kix[t]] / n[i]) / ctrl_k[kix[t]]
    save(name, targets, genes, beta, s, nn_t, ctrl, kd)

