"""The Stage 1 estimator and its helpers, shared by QC (null calibration) and every table builder (etl.stage1)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def cpm_matrix(counts: sp.csr_matrix) -> sp.csr_matrix:
    """Counts per million per cell, same sparsity pattern."""
    tot = np.asarray(counts.sum(1)).ravel()
    inv = sp.diags(1e6 / np.maximum(tot, 1))
    return (inv @ counts).tocsr()


def group_stats(cpm: sp.csr_matrix, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-group count, sum and sum of squares of CPM, as dense (groups, genes) arrays."""
    n = cpm.shape[0]
    G = sp.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(n_groups, n))
    sums = (G @ cpm).toarray()
    sq = cpm.copy()
    sq.data = sq.data ** 2
    sumsq = (G @ sq).toarray()
    return np.asarray(G.sum(1)).ravel(), sums, sumsq


def log2fc_and_se(n_p, sum_p, sumsq_p, n_c, sum_c, sumsq_c):
    """Stage 1 estimator (docs/archive/MODEL_DESIGN.md 3.2): log2 ratio of mean CPM, delta-method standard error.
    Inputs are broadcastable arrays. Genes with a zero mean in either group give NaN (missing, never zero)."""
    mp, mc = sum_p / n_p, sum_c / n_c
    vp = (sumsq_p / n_p - mp ** 2) * n_p / (n_p - 1)
    vc = (sumsq_c / n_c - mc ** 2) * n_c / (n_c - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log2(mp / mc)
        se = np.sqrt(vp / (n_p * mp ** 2) + vc / (n_c * mc ** 2)) / np.log(2)
    bad = ~((mp > 0) & (mc > 0))
    beta = np.where(bad, np.nan, beta)
    se = np.where(bad, np.nan, se)
    return beta, se


def context_gene_cpm(X: sp.csr_matrix) -> np.ndarray:
    """Mean per-cell CPM of each gene over the cells of X (the context's expression, used for the bin multipliers and
    for identity). Written without X.multiply, which fails on int64 indptr."""
    lib = np.asarray(X.sum(1)).ravel().astype(float)
    r_ = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    return np.bincount(X.indices, weights=X.data / np.maximum(lib, 1.0)[r_], minlength=X.shape[1]) / X.shape[0] * 1e6
