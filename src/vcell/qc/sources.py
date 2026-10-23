"""Cell-level QC of the public source screens (was eda/qc.py and eda/62_qc_audit.py). Never applied to the organisers'
contexts or the vcc-h1 benchmark truth. The masks are written but not used by the final tables: cleaning was neutral to
slightly negative (H52, experiments/failed/h52_source_qc.py).

The steps, in the order they are applied (docs/DATASET_FORMATS.md section 8 says which source gets which):
    1. Low quality   per batch: log1p(total UMI) or log1p(genes detected) more than 5 MADs below the batch median, or a
                     mitochondrial share more than 3 MADs above the median AND above MT_FLOOR % (or above MT_HARD %).
                     The floor keeps the rule from firing on probe-based (10x Flex) data, where mito is near zero.
    2. Multiplets    per batch: log1p(total UMI) or log1p(genes) more than 5 MADs above the median (doublets), and cells
                     with guides from more than one construct (the source's own guide call decides, see each builder).
    3. Ambient RNA   not corrected: no raw droplets for most sources, and it is shared by perturbed and control cells of a
                     batch, so it cancels in the log fold change up to a damping that the tuned scale absorbs.
    4. Genes         unchanged: the Stage 1 universe (5 CPM in a context) and NaN for zero means already do this.
    5. Targets       a target row needs MIN_CELLS cells after QC, else it is left missing (NaN) for that line.
                     Escaper cells (guide present, no knockdown) are NOT removed: the scored truth cells are defined by
                     their guide, so they contain escapers too; whole-target knockdown weighting was neutral (H23).
Thresholds are computed per batch over all cells of the batch (perturbed and control alike), so a knockdown that
really changes depth or mito share is judged against its batch, not against the controls only.

Audit (one keep-mask per source, OUT/qc/<source>.npz: keep, reason; X-Atlas also the per-file row offsets in the
builder's file order):  python -m vcell qc sources [--sources HCT116 HEK293T H1 K562sc RPE1 HepG2 Jurkat KOLF]
Sources without cells (K562 pseudobulk, CD4+ T DE statistics, HKU moments) cannot be cleaned and are not listed.
"""
from __future__ import annotations

import fsspec
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vcell import config, io

N_MAD_LOW = 5.0
N_MAD_HIGH = 5.0
N_MAD_MT = 3.0
MT_FLOOR = 5.0      # percent
MT_HARD = 20.0      # percent
MIN_CELLS = 20

REASONS = ("ok", "low_depth", "low_genes", "high_mito", "high_depth", "high_genes", "multi_guide")


def _mad_bounds(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    return med, float(np.median(np.abs(x - med)) * 1.4826)


def cell_mask(total, genes=None, pct_mt=None, batch=None, multi_guide=None) -> tuple[np.ndarray, np.ndarray]:
    """Keep mask and a reason code per cell (index into REASONS; the first failing rule wins).
    total: UMI per cell; genes: genes detected per cell (optional); pct_mt: mitochondrial percent (optional);
    batch: batch label per cell (optional, one batch if None); multi_guide: bool per cell (optional)."""
    total = np.asarray(total, float)
    n = len(total)
    reason = np.zeros(n, np.int8)
    batch = np.zeros(n, int) if batch is None else pd.factorize(np.asarray(batch))[0]
    lt = np.log1p(total)
    lg = None if genes is None else np.log1p(np.asarray(genes, float))
    mt = None if pct_mt is None else np.asarray(pct_mt, float)
    for b in np.unique(batch):
        ix = np.flatnonzero(batch == b)
        checks = []
        med, mad = _mad_bounds(lt[ix])
        checks += [("low_depth", lt[ix] < med - N_MAD_LOW * mad), ("high_depth", lt[ix] > med + N_MAD_HIGH * mad)]
        if lg is not None:
            med, mad = _mad_bounds(lg[ix])
            checks += [("low_genes", lg[ix] < med - N_MAD_LOW * mad), ("high_genes", lg[ix] > med + N_MAD_HIGH * mad)]
        if mt is not None:
            med, mad = _mad_bounds(mt[ix])
            checks.append(("high_mito", ((mt[ix] > med + N_MAD_MT * mad) & (mt[ix] > MT_FLOOR)) | (mt[ix] > MT_HARD)))
        for name in REASONS[1:-1]:                                    # fixed order, so reasons are comparable
            for nm, bad in checks:
                if nm == name:
                    sel = ix[bad & (reason[ix] == 0)]
                    reason[sel] = REASONS.index(name)
    if multi_guide is not None:
        mg = np.asarray(multi_guide, bool) & (reason == 0)
        reason[mg] = REASONS.index("multi_guide")
    return reason == 0, reason


def report(name: str, keep: np.ndarray, reason: np.ndarray, target: np.ndarray, control_label: str) -> str:
    """One text block: removal overall, by reason, controls vs perturbed, and per-target removal quantiles."""
    target = np.asarray(target, dtype=object)
    is_c = target == control_label
    lines = [f"== {name}: {len(keep)} cells, removed {int((~keep).sum())} ({100 * (~keep).mean():.2f}%)"]
    lines.append("   by reason: " + ", ".join(f"{r} {int((reason == k).sum())}" for k, r in enumerate(REASONS) if k and (reason == k).any()))
    lines.append(f"   controls removed {100 * (~keep[is_c]).mean():.2f}% of {int(is_c.sum())}; "
                 f"perturbed {100 * (~keep[~is_c]).mean():.2f}% of {int((~is_c).sum())}")
    df = pd.DataFrame({"t": target[~is_c], "rm": ~keep[~is_c]})
    g = df.groupby("t").rm.agg(["mean", "size"])
    g = g[g["size"] >= MIN_CELLS]
    q = np.percentile(100 * g["mean"], [50, 90, 99, 100])
    lines.append(f"   per target ({len(g)} with >= {MIN_CELLS} cells) removed %: median {q[0]:.1f}, 90th {q[1]:.1f}, "
                 f"99th {q[2]:.1f}, max {q[3]:.1f}")
    worst = g.sort_values("mean", ascending=False).head(5)
    lines.append("   most affected targets: " + ", ".join(f"{t} {100 * m:.0f}% of {s}" for t, (m, s) in worst.iterrows()))
    after = (df[~df.rm].groupby("t").size() >= MIN_CELLS).sum()
    lines.append(f"   targets with >= {MIN_CELLS} cells: before {int((df.groupby('t').size() >= MIN_CELLS).sum())}, after {int(after)}")
    return "\n".join(lines)

# ------------------------------------------------------------------------------------------------ audit per source
QC_DIR = config.OUT / "qc"
DENSE = {"K562sc": config.SOURCES / "replogle" / "K562_gwps_raw_singlecell_01.h5ad",
         "RPE1": config.SOURCES / "replogle" / "rpe1_raw_singlecell_01.h5ad",
         "HepG2": config.SOURCES / "nadig" / "GSE264667_hepg2_raw_singlecell_01.h5ad",
         "Jurkat": config.SOURCES / "nadig" / "GSE264667_jurkat_raw_singlecell_01.h5ad"}
KOLF_URL = "https://ndownloader.figshare.com/files/64650261"
SOURCES = ["HCT116", "HEK293T", "H1", "K562sc", "RPE1", "HepG2", "Jurkat", "KOLF"]


def xatlas_files(line):
    d = config.SOURCES / "xatlas" / line
    return sorted(d.glob(f"{line}_Batch*.parquet"), key=lambda f: int(f.stem.split("Batch")[1]))


def audit_xatlas(line):
    import pyarrow.parquet as pq
    cols = ["sample", "num_features", "gene_target", "pass_guide_filter", "n_genes_by_counts", "total_counts", "pct_counts_mt"]
    parts, offsets = [], [0]
    for f in xatlas_files(line):
        df = pq.read_table(f, columns=cols).to_pandas()
        parts.append(df)
        offsets.append(offsets[-1] + len(df))
    df = pd.concat(parts, ignore_index=True)
    used = df.pass_guide_filter.to_numpy() == 1                      # the builder's guide filter: two guides of one pair
    print(f"{line}: {len(df)} cells, {int(used.sum())} pass the guide filter; num_features among them: "
          f"{df.num_features[used].value_counts().sort_index().head(6).to_dict()}", flush=True)
    keep = np.zeros(len(df), bool)
    reason = np.full(len(df), -1, np.int8)
    u = df[used]
    k, r = cell_mask(u.total_counts, u.n_genes_by_counts, u.pct_counts_mt, u["sample"], u.num_features > 2)
    keep[used], reason[used] = k, r
    print(report(line, k, r, u.gene_target.to_numpy(), "Non-Targeting"), flush=True)
    np.savez_compressed(QC_DIR / f"{line}.npz", keep=keep, reason=reason, offsets=np.array(offsets))


def audit_h1():
    path = config.SOURCES / "h1" / "train" / "adata_Training.h5ad"
    with h5py.File(path, "r") as f:
        obs = io.read_df_group(f["obs"])
        hg = np.array(io.decode(f["var"]["_index"][...]), dtype=object)
        mt = np.array([g.startswith("MT-") for g in hg])
        X = f["X"]
        indptr = X["indptr"][...]
        n_cells = len(indptr) - 1
        tot, ngen, mtc = np.zeros(n_cells), np.diff(indptr), np.zeros(n_cells)
        for a in range(0, n_cells, 20000):
            b = min(a + 20000, n_cells)
            lo, hi = int(indptr[a]), int(indptr[b])
            m = sp.csr_matrix((X["data"][lo:hi], X["indices"][lo:hi], indptr[a:b + 1] - lo), shape=(b - a, len(hg)))
            tot[a:b] = np.asarray(m.sum(1)).ravel()
            mtc[a:b] = np.asarray(m[:, mt].sum(1)).ravel()
    print(f"H1: {n_cells} cells, {int(mt.sum())} MT genes; guides per cell are single calls (guide_id)", flush=True)
    k, r = cell_mask(tot, ngen, 100 * mtc / np.maximum(tot, 1), obs["batch"].to_numpy())
    print(report("H1", k, r, obs["target_gene"].astype(str).to_numpy(), "non-targeting"), flush=True)
    np.savez_compressed(QC_DIR / "H1.npz", keep=k, reason=r)


def audit_dense(name):
    with h5py.File(DENSE[name], "r") as f:
        o = f["obs"]
        cats = np.array(io.decode(o["__categories"]["gene"][...]), dtype=object)
        tg = cats[o["gene"][...]]
        umi, mito, gem = o["UMI_count"][...].astype(float), o["mitopercent"][...].astype(float), o["gem_group"][...]
    pct = mito * 100 if np.nanmax(mito) <= 1 else mito                 # Replogle stores a fraction
    print(f"{name}: {len(tg)} cells, {len(np.unique(gem))} gem groups; genes per cell not stored (dense matrix), "
          f"so depth and mito only", flush=True)
    k, r = cell_mask(umi, None, pct, gem)
    ctrl = np.array(["non-targeting" if "non-targeting" in t.lower() else t for t in tg], dtype=object)
    print(report(name, k, r, ctrl, "non-targeting"), flush=True)
    np.savez_compressed(QC_DIR / f"{name}.npz", keep=k, reason=r)


def audit_kolf():
    fs = fsspec.filesystem("https")
    with fs.open(KOLF_URL, "rb", cache_type="blockcache", block_size=4 << 20) as fo, h5py.File(fo, "r") as f:
        o = f["obs"]
        keys = list(o.keys())
        print("KOLF obs columns:", keys, flush=True)

        def col(name):
            x = o[name]
            if isinstance(x, h5py.Group):
                return np.array(io.decode(x["categories"][...]), dtype=object)[x["codes"][...]]
            return x[...]
        pick = lambda *names: next((n for n in names if n in keys), None)
        c_tot, c_gen, c_mt = pick("total_counts", "UMI_count"), pick("n_genes_by_counts", "n_genes"), pick("pct_counts_mt", "percent_mito", "mitopercent")
        c_b = pick("batch", "gem_group", "sample", "library", "lane")
        c_ng = pick("num_features", "n_guides", "guides_per_cell")
        tot, gen = col(c_tot).astype(float), (col(c_gen).astype(float) if c_gen else None)
        pmt = col(c_mt).astype(float) if c_mt else None
        batch, ng = (col(c_b) if c_b else None), (col(c_ng) if c_ng else None)
        tg = col("gene_target")
    print(f"KOLF: using total {c_tot}, genes {c_gen}, mito {c_mt}, batch {c_b}, guides {c_ng}", flush=True)
    if pmt is not None and np.nanmax(pmt) <= 1:
        pmt = pmt * 100
    k, r = cell_mask(tot, gen, pmt, batch, None if ng is None else np.asarray(ng, float) > 1)
    ctrl = np.array(["NTC" if (t == "NTC" or ("non" in str(t).lower() and "target" in str(t).lower())) else t for t in tg],
                    dtype=object)
    print(report("KOLF", k, r, ctrl, "NTC"), flush=True)
    np.savez_compressed(QC_DIR / "KOLF.npz", keep=k, reason=r)


def audit(sources=SOURCES) -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    for s in sources:
        if s in ("HCT116", "HEK293T"):
            audit_xatlas(s)
        elif s == "H1":
            audit_h1()
        elif s in DENSE:
            audit_dense(s)
        elif s == "KOLF":
            audit_kolf()
