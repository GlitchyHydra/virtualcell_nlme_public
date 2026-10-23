"""R1 (the scorer's top anchor): Positive control for the H1 scorer.

06 of the benchmark (vcc-h1 score-control-baseline) showed the bottom of the scale: unchanged controls score -0.3026.
This builds the top: a "prediction" made of REAL H1 perturbed cells of the same targets that the scorer's fixed
reference sample does not contain (a fresh draw of 400 real cells per target). If the scorer works, it must score
close to the replicate ceiling (1.0 on the rescaled scale), well above the control baseline.

    python experiments/analysis/r1_positive_control.py                 # fresh real cells (default)
    python experiments/analysis/r1_positive_control.py --mode same     # the reference cells themselves
    python experiments/analysis/r1_positive_control.py --mode guide    # one guide design per target where two exist (see below)

Then:
    vcc-h1 validate  <out>.h5ad --data-dir data/h1_benchmark
    vcc-h1 score     <out>.h5ad --data-dir data/h1_benchmark --output OUT/experiments/h1_positive_control_<mode>

Output contract (benchmark README): 400 cells per target for the 126 benchmark targets, exact H1 gene axis, one obs
column target_gene, no controls, raw counts in sparse X.

Result (EXPERIMENT_LOG R1): real fresh H1 cells as a prediction score 1.067 overall vs -0.303 for unchanged control
resamples (R0): the official scorer works at both ends.
Was eda/07_h1_positive_control.py. Snapshot abc (the A/B/C validation tables). 10 minutes plus vcc-h1 score (~25 minutes).
"""
from __future__ import annotations

import argparse

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vcell import config, io

config.use_snapshot("abc")
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["fresh", "same", "guide"], default="fresh")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

H1 = config.SOURCES / "h1"
BM = config.REPO / "data" / "h1_benchmark"
ref = pd.read_csv(BM / "benchmark" / "reference_cells.csv")
targets = list(dict.fromkeys(ref.target_gene))               # canonical order of the benchmark file
print(f"{len(targets)} benchmark targets, {len(ref)} reference cells")

with h5py.File(H1 / "train" / "adata_Training.h5ad", "r") as f:
    obs = io.read_df_group(f["obs"])
    genes = np.array(io.decode(f["var"]["_index"][...]), dtype=object)
    indptr = f["X"]["indptr"][...]
    tg = obs["target_gene"].astype(str).to_numpy()
    gid = obs["guide_id"].astype(str).to_numpy()
    ok = (tg[ref.source_row.to_numpy()] == ref.target_gene.to_numpy()).mean()
    print(f"reference rows point at the right target in the train file: {ok:.4f}")
    assert ok > 0.999, "source_row does not index the train file; check the benchmark manifest"

    rows, labels, short = [], [], []
    for t in targets:
        in_ref = ref.source_row[ref.target_gene == t].to_numpy()
        allr = np.flatnonzero(tg == t)
        if args.mode == "same":
            pick = in_ref
        else:
            pool = np.setdiff1d(allr, in_ref)
            if args.mode == "guide":                          # guide-design split: the design NOT used most in the reference
                d = pd.Series(gid[in_ref]).value_counts()
                pool = pool[gid[pool] != d.index[0]] if len(d) > 1 and (gid[pool] != d.index[0]).sum() >= 100 else pool
            if len(pool) >= 400:
                pick = rng.choice(pool, 400, replace=False)
            else:                                             # not enough fresh cells: top up from the reference cells
                short.append((t, len(pool)))
                pick = np.concatenate([pool, rng.choice(in_ref, 400 - len(pool), replace=False)])
        rows.append(np.sort(pick)); labels += [t] * len(pick)
    rows = np.concatenate(rows)
    print(f"{len(rows)} cells; targets with fewer than 400 fresh cells (topped up from reference): {short}")

    X = f["X"]
    data, idx, ptr = [], [], [0]
    for r in rows:
        s, e = int(indptr[r]), int(indptr[r + 1])
        data.append(X["data"][s:e]); idx.append(X["indices"][s:e]); ptr.append(ptr[-1] + (e - s))
    M = sp.csr_matrix((np.concatenate(data).astype(np.float32), np.concatenate(idx).astype(np.int32),
                       np.array(ptr, dtype=np.int64)), shape=(len(rows), len(genes)))

# rows were sorted within a target, labels must follow the same order
lab = np.array(labels, dtype=object)
a = ad.AnnData(X=M, obs=pd.DataFrame({"target_gene": pd.Categorical(lab)}, index=[f"pred{i}" for i in range(len(lab))]),
               var=pd.DataFrame(index=pd.Index(genes.astype(str))))
out = BM / f"pred_positive_control_{args.mode}.h5ad"
a.write_h5ad(out)
print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB): {a.n_obs} cells x {a.n_vars} genes, nnz/cell {M.nnz / a.n_obs:.0f}")
