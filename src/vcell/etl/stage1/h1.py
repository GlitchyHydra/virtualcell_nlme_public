"""Arc H1 hESC (VCC 2025 training file): single cells, ratio of means and delta-method s, one streamed pass.
H1 is the held-out benchmark line of the official H1 scorer (evaluation.h1_benchmark) and a source line otherwise."""
from __future__ import annotations

import h5py
import numpy as np

from vcell import io, stats
from vcell.etl.stage1.table import H1_DIR, qc_keep, save, stream_group_stats


def h1(targets, genes, qc: bool = False, name: str | None = None) -> None:
    """H1 from the training file (150 targets, all cells; benchmark truth for docs/archive/EDA_PLAN.md).
    qc=True drops the cells eda/qc.py flags (saved as H1_qc by default)."""
    path = H1_DIR / "train" / "adata_Training.h5ad"
    with h5py.File(path, "r") as f:
        obs = io.read_df_group(f["obs"])
        hg = np.array(io.decode(f["var"]["_index"][...]), dtype=object)
    tg = obs["target_gene"].astype(str).to_numpy()
    present = sorted(set(tg) - {"non-targeting"})
    gcode = {t: i for i, t in enumerate(present)}
    nt_code = len(present)
    codes = np.array([gcode.get(t, nt_code if t == "non-targeting" else -1) for t in tg])
    if qc:
        keep, _ = qc_keep("H1")
        codes[~keep] = -1
        print(f"QC: {int((~keep).sum())} of {len(keep)} cells dropped", flush=True)
    n, S, SS = stream_group_stats(path, codes, nt_code + 1)
    npg, nc = n[:nt_code, None], n[nt_code]
    beta_h, se_h = stats.log2fc_and_se(npg, S[:nt_code], SS[:nt_code], nc, S[nt_code], SS[nt_code])
    ctrl_h = S[nt_code] / nc
    hix = {g: i for i, g in enumerate(hg)}
    col = np.array([hix.get(g, -1) for g in genes])
    have = col >= 0
    T, G = len(targets), len(genes)
    beta = np.full((T, G), np.nan)
    s = np.full((T, G), np.nan)
    nn = np.full(T, np.nan)
    tix = {t: i for i, t in enumerate(targets)}
    for t, k in gcode.items():
        if t in tix:
            beta[tix[t], have] = beta_h[k, col[have]]
            s[tix[t], have] = se_h[k, col[have]]
            nn[tix[t]] = n[k]
    ctrl = np.full(G, np.nan)
    ctrl[have] = ctrl_h[col[have]]
    save(name or ("H1_qc" if qc else "H1"), targets, genes, beta, s, nn, ctrl)

