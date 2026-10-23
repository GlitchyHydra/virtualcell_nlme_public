"""H16 (rejected): a generator with a visibility switch. pds, mse and the DE log2FC see only the mean of each gene; the
Wilcoxon calls also see how the shift is spread over cells. So only the n genes of largest |z| get a rank-visible shift
(binomial thinning / Poisson additions in every cell, as generator v0); every other gene keeps its mean shift, placed where
a rank test barely sees it: f > 1 adds all extra counts to a few random cells (SPIKE_FRAC), f < 1 lowers only the highest
counts (water-filling from the top). Was eda/generator.py, used by eda/15_generate_h1.py --visible c (n = c x the count of
|z| > 3, at least 5).

Verdict (EXPERIMENT_LOG H16): rejected in the proxy before any scorer run (experiments/failed/h16_h17_visibility_and_stack.py):
fid punishes under-calling and our top genes rarely match the real DE genes, so hiding shifts loses for every n.

    from visibility_generator import visible_emitter
    h1_benchmark.run(cfg, emitter=visible_emitter(cfg, c=1.0))    # the H1 benchmark with the switch (never scored)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

SPIKE_FRAC = 0.02


def visible_set(eff: np.ndarray, z: np.ndarray, n: int) -> np.ndarray:
    """The n genes with the largest |z| among those with a non-zero effect."""
    c = np.where(eff != 0, np.abs(np.nan_to_num(z)), 0.0)
    n = int(min(n, (c > 0).sum()))
    out = np.zeros(len(eff), bool)
    out[np.argsort(-c)[:n]] = True
    return out


def apply(M: sp.csr_matrix, eff: np.ndarray, mu: np.ndarray, size_pick: np.ndarray, vis: np.ndarray,
          rng: np.random.Generator) -> sp.csr_matrix:
    """M: resampled control cells (cells x genes, integer counts); eff: log2 effect per gene, 0 = unchanged;
    vis: True where the shift should be visible to a rank test."""
    D = M.toarray()
    n = D.shape[0]
    f = 2.0 ** eff
    vd, vu = np.flatnonzero((f < 1) & vis), np.flatnonzero((f > 1) & vis)
    hd, hu = np.flatnonzero((f < 1) & ~vis), np.flatnonzero((f > 1) & ~vis)
    if len(vd):
        D[:, vd] = rng.binomial(D[:, vd], f[vd][None, :])
    if len(vu):
        D[:, vu] += rng.poisson((f[vu] - 1.0)[None, :] * mu[vu][None, :] * size_pick[:, None])
    if len(hu):                                     # all extra counts into a few random cells per gene
        k = max(1, int(round(SPIKE_FRAC * n)))
        total = (f[hu] - 1.0) * mu[hu] * size_pick.sum()
        cells = rng.integers(0, n, (k, len(hu)))
        np.add.at(D, (cells, np.broadcast_to(hu, cells.shape)), rng.poisson(total / k, (k, len(hu))))
    if len(hd):                                     # cut the highest counts down to a common level L per gene
        V = D[:, hd].astype(float)
        srt = -np.sort(-V, axis=0)
        cs = np.cumsum(srt, axis=0)
        nxt = np.vstack([srt[1:], np.zeros((1, len(hd)))])
        kk = np.arange(1, n + 1)[:, None]
        removed = cs - kk * nxt                     # mass removed when the top k counts are cut to the (k+1)-th value
        want = (1.0 - f[hd]) * cs[-1]
        k = np.argmax(removed >= want[None, :] - 1e-9, axis=0)          # first k (0-based) that removes enough
        L = (cs[k, np.arange(len(hd))] - want) / (k + 1)
        lo = np.floor(L)
        cut = np.where(V > L[None, :], lo[None, :] + (rng.random(V.shape) < (L - lo)[None, :]), V)
        D[:, hd] = cut.astype(D.dtype)
    out = sp.csr_matrix(D)
    out.eliminate_zeros()
    return out


def visible_emitter(cfg, c: float):
    """An emitter for evaluation.h1_benchmark.generate: per target (called in benchmark order for the targets with a Stage 1
    row), the c x strength genes of largest |z| are visible. z = pooled v3 / its predictive sd, as eda/15 computed it."""
    from vcell.evaluation import h1_benchmark as H
    from vcell.model import pool
    from vcell.model.effects import load_tables
    from vcell import config

    W = {n: w for n, w in cfg.model.weights().items() if w > 0}
    tabs = load_tables(list(W))
    first = tabs[cfg.model.sources[0]]
    uni_genes, uni_targets = list(first["genes"]), list(first["targets"])
    max_cpm = config.gene_lines().reindex(uni_genes).max_cpm.to_numpy()
    lam = list(W.values())
    fit = pool.predict(np.stack([tabs[n]["beta"] for n in W]).astype(float), np.stack([tabs[n]["s"] for n in W]).astype(float),
                       max_cpm, lam=None if all(w == 1.0 for w in lam) else lam, ash=False)
    Z = fit["v3"]["pred"] / fit["v3"]["sd"]
    import anndata as ad
    axis = np.array(ad.read_h5ad(H.BM / "h1_controls.h5ad", backed="r").var_names.astype(str))
    pos = {g: i for i, g in enumerate(axis)}
    col = np.array([pos.get(g, -1) for g in uni_genes])
    keep = col >= 0
    tix = {t: i for i, t in enumerate(uni_targets)}
    order = iter([t for t in H.benchmark_targets() if t in tix])

    def emit(M, eff, mu, size_pick, rng):
        z = np.zeros(len(axis))
        z[col[keep]] = np.nan_to_num(Z[tix[next(order)]][keep])
        n_vis = max(5, c * (np.abs(z) > 3).sum())
        return apply(M, eff, mu, size_pick, visible_set(eff, z, n_vis), rng)
    return emit
