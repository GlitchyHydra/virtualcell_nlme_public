"""The Stage 1 table format and the helpers every builder shares.

One table per source line, written as <TABLES>/stage1_<line>.npz:
    targets   (T,)    target gene symbols
    genes     (G,)    response genes: the universe of etl.universe (genes above 5 CPM in a context)
    beta      (T, G)  log2 fold change, NaN where missing (never zero)
    s         (T, G)  its standard error, NaN where missing
    n         (T,)    cells carrying the target in that line (NaN if the target was not measured)
    ctrl_cpm  (G,)    mean CPM of the gene in the line's control cells (NaN if the line does not report it)
    kd        (T,)    knockdown of the target itself where the source reports it, else NaN
The target gene itself is set to NaN in its own row.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vcell import config, io, stats

H1_DIR = config.SOURCES / "h1"


def universe_genes(path=None) -> list[str]:
    gl = config.gene_lines(path)
    return list(gl.index[gl.testable_any.astype(bool)])


def all_targets(controls=None) -> list[str]:
    """Panel targets plus every H1 target (train, validation, test)."""
    t = set(config.panel_targets(controls))
    for d, s in (("train", "Training"), ("validation", "Validation"), ("test", "Test")):
        t |= set(pd.read_csv(H1_DIR / d / f"pert_counts_{s}.csv").target_gene)
    return sorted(t)


def benchmark_targets() -> list[str]:
    """The 126 H1 targets of the vcc-h1 benchmark."""
    ref = pd.read_csv(config.H1_BENCHMARK / "benchmark" / "reference_cells.csv")
    return list(dict.fromkeys(ref.target_gene))


def table_path(name: str, tables=None) -> Path:
    return Path(tables or config.TABLES) / f"stage1_{name}.npz"


def save(name: str, targets, genes, beta, s, n, ctrl_cpm, kd=None, tables=None) -> None:
    targets, genes = list(targets), list(genes)
    tix = {t: i for i, t in enumerate(targets)}
    gix = {g: j for j, g in enumerate(genes)}
    for t, i in tix.items():                              # a target is not its own response gene
        if t in gix:
            beta[i, gix[t]] = np.nan
            s[i, gix[t]] = np.nan
    s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
    beta = np.where(np.isfinite(s), beta, np.nan)
    path = table_path(name, tables)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, targets=np.array(targets), genes=np.array(genes),
                        beta=beta.astype(np.float32), s=s.astype(np.float32), n=np.asarray(n, dtype=np.float32),
                        ctrl_cpm=np.asarray(ctrl_cpm, dtype=np.float32),
                        kd=np.full(len(targets), np.nan, np.float32) if kd is None else np.asarray(kd, np.float32))
    ok = np.isfinite(beta)
    print(f"saved stage1_{name}.npz: {len(targets)} targets x {len(genes)} genes; targets with data {int(ok.any(1).sum())}; "
          f"finite share {ok.mean():.3f}")


def load(name: str, tables=None) -> dict:
    z = np.load(table_path(name, tables), allow_pickle=True)
    return {k: z[k] for k in z.files}


def from_accum(line: str, targets, genes, name: str, path=None, tables=None) -> None:
    """Stage 1 table for any target list from accumulators (n, S, SS per target, last row the controls): the X-Atlas ones
    written by etl.stage1.xatlas (data/sources/xatlas/<line>_accum.npz) or any file in the same format (`path`, e.g. U87)."""
    z = np.load(path or config.SOURCES / "xatlas" / f"{line}_accum.npz", allow_pickle=True)
    all_t, ag, n, S, SS = list(z["targets"]), list(z["genes"]), z["n"], z["S"], z["SS"]
    assert ag == list(genes), "accumulator gene axis differs"
    code, nt = {t: i for i, t in enumerate(all_t)}, len(all_t)
    have = [t for t in targets if t in code]
    rows = np.array([code[t] for t in have])
    ctrl = S[nt] / n[nt]
    bs, ss = stats.log2fc_and_se(n[rows, None], S[rows], SS[rows], n[nt], S[nt], SS[nt])
    tix = {t: i for i, t in enumerate(targets)}
    gix = {g: j for j, g in enumerate(genes)}
    T, G = len(targets), len(genes)
    beta, s, nn, kd = np.full((T, G), np.nan), np.full((T, G), np.nan), np.full(T, np.nan), np.full(T, np.nan)
    for j, t in enumerate(have):
        i = tix[t]
        beta[i], s[i], nn[i] = bs[j], ss[j], n[rows[j]]
        if t in gix and ctrl[gix[t]] > 0:
            kd[i] = 1 - (S[rows[j], gix[t]] / n[rows[j]]) / ctrl[gix[t]]
    save(name, targets, genes, beta, s, nn, ctrl, kd, tables=tables)


def stream_group_stats(path, codes: np.ndarray, n_groups: int, block: int = 6000):
    """One pass over a CSR h5ad. codes[i] is the group of cell i (or -1 to skip).
    Returns n (groups,), sum and sum of squares of CPM, each (groups, genes of the file)."""
    with h5py.File(path, "r") as f:
        X = f["X"]
        indptr = X["indptr"][...]
        n_cells, n_genes = len(indptr) - 1, int(X.attrs["shape"][1])
        assert len(codes) == n_cells
        n = np.zeros(n_groups)
        S = np.zeros((n_groups, n_genes))
        SS = np.zeros((n_groups, n_genes))
        for a in range(0, n_cells, block):
            b = min(a + block, n_cells)
            c = codes[a:b]
            keep = np.flatnonzero(c >= 0)
            if keep.size == 0:
                continue
            lo, hi = int(indptr[a]), int(indptr[b])
            m = sp.csr_matrix((X["data"][lo:hi].astype(np.float32), X["indices"][lo:hi], indptr[a:b + 1] - lo),
                              shape=(b - a, n_genes))
            nn, s1, s2 = stats.group_stats(stats.cpm_matrix(m[keep]), c[keep], n_groups)
            n += nn
            S += s1
            SS += s2
            if (a // block) % 10 == 0:
                print(f"   cells {b}/{n_cells}", flush=True)
    return n, S, SS


def qc_keep(source: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Cell keep-mask written by qc.sources (rules in qc.sources.cell_mask) for a single-cell source."""
    z = np.load(config.OUT / "qc" / f"{source}.npz")
    return z["keep"], (z["offsets"] if "offsets" in z.files else None)


def dense_sc_targets(path) -> list[str]:
    """Target symbols of a Replogle/Nadig dense single-cell h5ad (non-targeting excluded)."""
    with h5py.File(path, "r") as f:
        cats = io.decode(f["obs"]["__categories"]["gene"][...])
    return sorted(c for c in cats if "non-targeting" not in c.lower())
