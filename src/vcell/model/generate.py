"""Cells from effects: 400 resampled control cells per target, counts changed by 2^effect (was the generator of
eda/16_generate_abc.py; the H1-benchmark variant of eda/15 is evaluation.h1_benchmark).

For every context and every panel target:
  1. draw the cells with replacement from that context's own control cells (keeps library size, sparsity and the gene-gene
     structure of real cells);
  2. per gene with a non-zero effect: f = 2^effect; f < 1 thins the counts binomially with p = f; f > 1 adds Poisson counts
     with mean (f - 1) x mu_gene x the cell's size factor (also where the gene was zero);
  3. integer counts, no renormalisation. Genes outside the Stage 1 universe are left unchanged.
One numpy Generator (cfg.seed) drives every draw in this order, so a file is reproducible bit for bit, and the first targets
of a --limit run equal those of the full run. The file holds about 2 billion stored values, so X is streamed to disk target
by target (CSR, int64 indptr) instead of held in memory.
"""
from __future__ import annotations

import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vcell import config, io
from vcell.model.effects import Effects


def emit(M: sp.csr_matrix, eff: np.ndarray, mu: np.ndarray, size_pick: np.ndarray, rng: np.random.Generator) -> sp.csr_matrix:
    """Apply one target's log2 effects (0 = unchanged) to its resampled cells M (cells x genes, integer counts)."""
    fct = 2.0 ** eff
    down, up = np.flatnonzero(fct < 1), np.flatnonzero(fct > 1)
    if len(down):                                          # binomial thinning of existing counts
        M = M.tocoo()
        m = np.isin(M.col, down)
        d = M.data.copy()
        d[m] = rng.binomial(d[m], fct[M.col[m]])
        M = sp.csr_matrix((d, (M.row, M.col)), shape=M.shape)
        M.eliminate_zeros()
    if len(up):                                            # extra Poisson counts, also where the gene was zero
        lam = (fct[up] - 1.0)[None, :] * mu[up][None, :] * size_pick[:, None]
        add = rng.poisson(lam).astype(np.int32)
        r_, c_ = np.nonzero(add)
        M = (M + sp.csr_matrix((add[r_, c_], (r_, up[c_])), shape=M.shape)).tocsr()
    return M


def control_stats(X0: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Mean count per gene and the size factor (library size over its mean) of each control cell."""
    mu = np.asarray(X0.mean(0)).ravel()
    size_f = np.asarray(X0.sum(1)).ravel()
    return mu, size_f / size_f.mean()


def iter_cells(effects: Effects, seed: int = 0, emitter=emit, cells: int | None = None):
    """Yield (context, target index, cells x genes CSR with sorted indices) in file order, drawing from one Generator."""
    cells = cells or config.manifest()["cells_per_pert"]
    axis = list(effects.axis)
    rng = np.random.default_rng(seed)
    for ctx in effects.contexts:
        X0, _, genes = io.load_context(ctx)
        assert list(genes) == axis, f"context {ctx}: gene axis differs from gene_names.csv"
        X0 = X0.tocsr().astype(np.int32)
        mu, size_f = control_stats(X0)
        print(f"context {ctx}: controls {X0.shape}, nnz/cell {X0.nnz / X0.shape[0]:.0f}", flush=True)
        for k in range(len(effects.targets)):
            pick = rng.integers(0, X0.shape[0], cells)
            M = X0[pick].tocsr()
            if effects.has_row[k]:
                M = emitter(M, effects.vector(ctx, k), mu, size_f[pick], rng)
            M.sort_indices()
            yield ctx, k, M
        del X0


def write_prediction(effects: Effects, out: Path, seed: int = 0, emitter=emit) -> Path:
    """The contest file for every context and target of `effects` (obs target_gene/context as in the manifest)."""
    man = config.manifest()
    pert_col, ctx_col, cells = man["pert_col"], man["context_col"], man["cells_per_pert"]
    ctxs, targets, axis = effects.contexts, effects.targets, np.array(effects.axis)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_obs = len(ctxs) * len(targets) * cells
    obs = pd.DataFrame({pert_col: pd.Categorical(np.repeat(targets * len(ctxs), cells)),
                        ctx_col: pd.Categorical(np.repeat(ctxs, len(targets) * cells))},
                       index=[f"pred{i}" for i in range(n_obs)])
    ad.AnnData(obs=obs, var=pd.DataFrame(index=pd.Index(axis))).write_h5ad(out)   # obs and var; X is streamed below

    indptr = np.zeros(n_obs + 1, dtype=np.int64)
    t0, row = time.time(), 0
    with h5py.File(out, "r+") as f:
        g = f.create_group("X")
        g.attrs["encoding-type"], g.attrs["encoding-version"] = "csr_matrix", "0.1.0"
        g.attrs["shape"] = np.array([n_obs, len(axis)], dtype=np.int64)
        dd = g.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(1 << 20,))
        di = g.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1 << 20,))
        nnz = 0
        for ctx, k, M in iter_cells(effects, seed, emitter, cells):
            n = M.nnz
            dd.resize(nnz + n, axis=0)
            di.resize(nnz + n, axis=0)
            dd[nnz:nnz + n] = M.data.astype(np.float32)
            di[nnz:nnz + n] = M.indices.astype(np.int32)
            indptr[row + 1:row + cells + 1] = nnz + M.indptr[1:]
            nnz += n
            row += cells
            if (k + 1) % 25 == 0 or k + 1 == len(targets):
                print(f"   {ctx} {k + 1}/{len(targets)} targets, {nnz / 1e9:.2f}G stored, {(time.time() - t0) / 60:.1f} min",
                      flush=True)
        g.create_dataset("indptr", data=indptr)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.1f} GB): {n_obs} cells x {len(axis)} genes, nnz/cell {nnz / n_obs:.0f}")
    return out
