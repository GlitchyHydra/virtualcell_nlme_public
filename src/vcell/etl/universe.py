"""The gene universe (was eda/06_gene_coverage.py and eda/66_widen_axis.py): OUT/gene_lines.csv, one row per gene of the
organisers' axis, with cpm_<context> (mean per-cell CPM in the controls), testable_any (above 5 CPM in any context: the
universe of every Stage 1 table) and max_cpm (the expression bin of the between-line variance tau^2, model.pool.gene_bins).
It also records which source lines measure each gene and, per (target, gene) pair, how many lines can contribute.

    build()    from the validation contexts (A/B/C; needs qc.controls and etl.coverage first): gene_lines.csv
    widen()    add the test contexts (D/E/F, F6): 11,961 -> 12,816 genes
"""
from __future__ import annotations

import shutil
from pathlib import Path

import anndata as ad
import fsspec
import h5py
import numpy as np
import pandas as pd

from vcell import config, io, stats

CD4_URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad"
SOURCES = ["K562", "HCT116", "HEK293T", "CD4 T"]          # same names as OUT/coverage_targets.csv
BIN_EDGES = [5, 10, 50, 200, 1000, np.inf]
BIN_NAMES = ["5-10", "10-50", "50-200", "200-1000", ">1000"]


def k562_pool_cpm() -> pd.Series:
    b = io.read_replogle_bulk(config.SOURCES / "replogle" / "K562_gwps_raw_bulk_01.h5ad")
    X, meta = b["X"], b["meta"]
    cpm = X / X.sum(1, keepdims=True) * 1e6
    nt = meta.is_nt.to_numpy() & (meta.cells.to_numpy() >= 20)
    w = meta.cells.to_numpy()[nt]
    pool = (cpm[nt] * w[:, None]).sum(0) / w.sum()
    return pd.Series(pool, index=np.array(b["genes"])).groupby(level=0).max()


def cd4_axis() -> pd.Index:
    cache = config.OUT / "cd4_gene_axis.csv"
    if cache.exists():
        return pd.Index(pd.read_csv(cache).gene)
    print("reading the CD4+ T gene axis remotely (var only) ...")
    fs = fsspec.filesystem("https")
    with fs.open(CD4_URL, "rb", cache_type="blockcache", block_size=256 * 1024) as fo, h5py.File(fo, "r") as f:
        v = f["var"]["gene_name"]
        if isinstance(v, h5py.Group) and "categories" in v:
            g = np.array(io.decode(v["categories"][...]), dtype=object)[v["codes"][...]]
        else:
            g = np.array(io.decode(v["values"][...] if isinstance(v, h5py.Group) else v[...]), dtype=object)
    pd.DataFrame({"gene": g}).to_csv(cache, index=False)
    return pd.Index(g)


def build(min_cpm: float = 1.0, min_cells: int = 50) -> None:
    """gene_lines.csv and pair_lines_summary.csv for the contexts of CONTROLS (A/B/C when it was run, eda/06).
    Needs OUT/context_summary.npz (qc.controls) and OUT/coverage_targets.csv (etl.coverage)."""

    S = np.load(config.OUT / "context_summary.npz", allow_pickle=True)
    genes = config.gene_names()
    ctx = config.manifest()["contexts"]
    cpm_ctx = pd.DataFrame({c: S[f"{c}_mean_cpm"] for c in ctx}, index=genes)

    # ---- gene presence per source line: True / False / NaN (unknown)
    pres = pd.DataFrame(index=genes, columns=SOURCES, dtype=object)
    k = k562_pool_cpm()
    pres["K562"] = (k.reindex(genes).fillna(0) >= min_cpm).to_numpy()
    pres["CD4 T"] = genes.isin(cd4_axis()) if hasattr(genes, "isin") else np.isin(genes, cd4_axis())
    for line in ("HCT116", "HEK293T"):
        p = config.OUT / f"gene_cpm_{line}.csv"
        if p.exists():
            d = pd.read_csv(p).drop_duplicates("gene").set_index("gene").mean_cpm
            pres[line] = (d.reindex(genes).fillna(0) >= min_cpm).to_numpy()
        else:
            pres[line] = np.nan
    h1p = config.SOURCES / "h1" / "gene_names.csv"
    h1 = None
    if h1p.exists():
        h1 = np.isin(genes, pd.read_csv(h1p, header=None).iloc[:, 0].astype(str).to_numpy())   # no header row
    known = [c for c in SOURCES if pres[c].notna().all()]
    unknown = [c for c in SOURCES if c not in known]
    print(f"detection threshold {min_cpm} CPM (K562); target threshold {min_cells} cells")
    print(f"lines with a gene axis in hand: {known}; unknown for now: {unknown}")

    lo = pres.fillna(False).astype(bool)
    up = pres.fillna(True).astype(bool)

    # ---- target coverage per line, from 03_coverage.py
    cov = pd.read_csv(config.OUT / "coverage_targets.csv", index_col=0)
    T = (cov[SOURCES].fillna(0) >= min_cells).to_numpy()                 # 300 x lines

    out = pd.DataFrame(index=genes)
    for c in ctx:
        out[f"cpm_{c}"] = cpm_ctx[c]
    for c in SOURCES:
        out[f"in_{c}"] = pres[c]
    if h1 is not None:
        out["in_H1"] = h1
    out["gene_lines_lower"] = lo.sum(axis=1)
    out["gene_lines_upper"] = up.sum(axis=1)
    out["testable_any"] = (cpm_ctx > config.CPM_FILTER).any(axis=1)
    out["max_cpm"] = cpm_ctx.max(axis=1)

    # ---- pair level: lines that have the target AND the gene
    P_lo = T.astype(np.int16) @ lo.to_numpy().astype(np.int16).T              # 300 targets x genes
    P_up = T.astype(np.int16) @ up.to_numpy().astype(np.int16).T
    out["pair_median_lower"] = np.median(P_lo, axis=0)
    out["pair_ge3_share_lower"] = (P_lo >= 3).mean(0)
    out["pair_ge3_share_upper"] = (P_up >= 3).mean(0)

    rows = []
    for c in ctx:
        test = cpm_ctx[c].to_numpy() > config.CPM_FILTER
        b = pd.cut(cpm_ctx[c], BIN_EDGES, labels=BIN_NAMES, right=False)
        for name in BIN_NAMES + ["all testable"]:
            m = test & (b.to_numpy() == name) if name != "all testable" else test
            if not m.any():
                continue
            r = {"context": c, "cpm_bin": name, "genes": int(m.sum())}
            r["share_in_K562"] = float(lo["K562"].to_numpy()[m].mean())
            r["share_in_CD4T"] = float(lo["CD4 T"].to_numpy()[m].mean())
            for k_ in (0, 1, 2):
                r[f"genes_lines_{k_}_upper"] = int((out.gene_lines_upper.to_numpy()[m] == k_).sum())
            r["pairs_ge3_lower"] = float((P_lo[:, m] >= 3).mean())
            r["pairs_ge3_upper"] = float((P_up[:, m] >= 3).mean())
            rows.append(r)
    summ = pd.DataFrame(rows)

    config.OUT.mkdir(parents=True, exist_ok=True)
    out.rename_axis("gene").to_csv(config.OUT / "gene_lines.csv")
    summ.to_csv(config.OUT / "pair_lines_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== testable genes (> 5 CPM in the context) by expression bin: how well are they covered? ===")
    print(summ.round(2).to_string(index=False))
    t = out[out.testable_any]
    never = t[t.gene_lines_upper < 3].sort_values("max_cpm", ascending=False)
    print(f"\ntestable in at least one context: {len(t)}")
    print(f"  not on the K562 axis: {int((~t.in_K562.astype(bool)).sum())}; not on the CD4+ T axis: {int((~t['in_CD4 T'].astype(bool)).sum())}")
    print(f"  cannot reach 3 lines even if HCT116 and HEK293T had them: {len(never)}")
    print("  highest-expressed of those:", ", ".join(f"{g} ({v:.0f})" for g, v in never.max_cpm.head(15).items()))
    if h1 is not None:
        print(f"  on the H1 gene axis: {int(t.in_H1.sum())} of {len(t)}")
    print("\nsaved", config.OUT / "gene_lines.csv", "and", config.OUT / "pair_lines_summary.csv")


def widen(controls_dir="data/controls_final", contexts="DEF") -> None:
    """Add the test contexts to the universe (eda/66, F6): cpm_<context> columns, testable_any over all contexts; max_cpm
    (the tau^2 bin) is kept for genes already in the universe, so the model is unchanged on them, and a new gene gets its
    max over the new contexts. The previous file is kept once as gene_lines_abc.csv."""
    controls_dir = config._repo_path(controls_dir)
    contexts = list(contexts)
    path, backup = config.OUT / "gene_lines.csv", config.OUT / "gene_lines_abc.csv"
    if not backup.exists():
        shutil.copy(path, backup)
    gl = pd.read_csv(backup, index_col=0)
    old_ctx = [c[4:] for c in gl.columns if c.startswith("cpm_") and c[4:] not in contexts]   # A, B, C
    old = gl.testable_any.astype(bool)
    for c in contexts:
        a = ad.read_h5ad(Path(controls_dir) / f"context_{c}.h5ad")
        cpm = stats.cpm_matrix(a.X.tocsr().astype(np.float64))
        gl[f"cpm_{c}"] = pd.Series(np.asarray(cpm.mean(axis=0)).ravel(), index=a.var_names).reindex(gl.index)
    six = gl[[f"cpm_{c}" for c in [*old_ctx, *contexts]]]
    gl["testable_any"] = (six > config.CPM_FILTER).any(axis=1)
    new = gl.testable_any & ~old
    gl.loc[new, "max_cpm"] = gl.loc[new, [f"cpm_{c}" for c in contexts]].max(axis=1)
    gl.to_csv(path)
    print(f"universe {int(old.sum())} -> {int(gl.testable_any.sum())} genes (+{int(new.sum())}); "
          f"new per context: " + ", ".join(f"{c} {int((new & (gl[f'cpm_{c}'] > config.CPM_FILTER)).sum())}" for c in contexts))
    print("max_cpm bins of the new genes:", np.bincount(np.digitize(gl.loc[new, "max_cpm"], [5, 10, 50, 200, 1000]), minlength=6).tolist())
