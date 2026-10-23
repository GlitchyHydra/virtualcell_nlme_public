"""The official scorer offline: generate the public H1 benchmark (126 targets x 400 cells) with H1 held out of the fit, then
`vcc-h1 score` (cell-eval2 0.16.0, about 25 min per file). Was eda/15_generate_h1.py and the scorer runs of the E-series.

Effects: Stage 2 on the config's source lines (H1 never among them), in the config's line order (not sorted: eda/15 stacked
the lines in --sources order), the rung, transform and scale, effects below min_effect left out; no bins or neighbour cap
(the E-series never used them). Cells: resampled H1 control cells (data/h1_benchmark/h1_controls.h5ad) with the emission of
eda/15, which differs from model.generate.emit only in float order and one extra uniform draw per target (the share of
escaping cells, always 0 here); kept so the E-series files reproduce bit for bit.

    python -m vcell evaluate h1 --config configs/e4_h1.toml [--seed 1] [--line-weight Pan_H1=4] [--no-score]
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vcell import config
from vcell.config import ModelConfig, RunConfig
from vcell.etl.stage1.table import load
from vcell.model import effects as EF

BM = config.H1_BENCHMARK
VCC_H1 = Path(r"C:\Users\glitc\.local\bin\vcc-h1.exe")
METRICS = {"pds": "pds_cosine", "mse": "expr_mse_unbiased_capped_norm", "nmae": "de_wilcoxon_lfc_nmae",
           "fid": "de_wilcoxon_direction_fidelity_yield_raw", "reach": "de_wilcoxon_direction_reach_raw",
           "jac": "de_wilcoxon_sig_jaccard"}


def benchmark_targets() -> list[str]:
    ref = pd.read_csv(BM / "benchmark" / "reference_cells.csv")
    return list(dict.fromkeys(ref.target_gene))


def pooled_effects(m: ModelConfig, post=None) -> tuple[np.ndarray, list, list]:
    """(effect matrix, Stage 1 targets, universe genes) for the H1 benchmark, H1 held out."""
    W = {n: w for n, w in m.weights().items() if w > 0}
    assert "H1" not in W, "H1 is the held-out line of the H1 benchmark"
    tabs = EF.load_tables(list(W))
    first = tabs[m.sources[0]]
    uni_genes, uni_targets = list(first["genes"]), list(first["targets"])
    max_cpm = config.gene_lines().reindex(uni_genes).max_cpm.to_numpy()
    print(f"lines {W}", flush=True)
    return EF.fit(tabs, tuple(W.items()), max_cpm, m, post), uni_targets, uni_genes


def oracle_effects(name: str, scale: float = 1.0) -> tuple[np.ndarray, list, list]:
    """A table's own beta as the effect, no pooling (H42: what the generator loses when the biology is right). Converted
    tables hold a few huge values for barely expressed genes, so the effect is clipped to +-10 (2**x would overflow)."""
    o = load(name)
    return np.clip(scale * o["beta"].astype(float), -10.0, 10.0), list(o["targets"]), list(o["genes"])


def emit_h1(M, eff, mu, size_pick, rng, escape: float = 0.0):
    """eda/15's emission with no escapers, no cell noise and additive up-regulation (generator v0)."""
    f = 2.0 ** eff
    down = np.flatnonzero(f < 1)
    up = np.flatnonzero(f > 1)
    hit = rng.random(M.shape[0]) >= escape                 # cells that respond (all when escape = 0)
    q = escape
    if len(down):
        M = M.tocoo()
        m = np.isin(M.col, down) & hit[M.row]
        d = M.data.copy()
        p = np.maximum(1 - (1 - f[M.col[m]]) / (1 - q), 0.0)
        d[m] = rng.binomial(d[m], p)
        M = sp.csr_matrix((d, (M.row, M.col)), shape=M.shape)
        M.eliminate_zeros()
    if len(up):
        base = mu[up][None, :] * size_pick[:, None]
        lam = ((f[up] - 1.0) / (1 - q))[None, :] * base * hit[:, None]
        add = rng.poisson(lam).astype(np.int32)
        r_, c_ = np.nonzero(add)
        M = (M + sp.csr_matrix((add[r_, c_], (r_, up[c_])), shape=M.shape)).tocsr()
    return M


def generate(cfg: RunConfig, out: Path | None = None, P=None, emitter=emit_h1, write: bool = True):
    """The H1 benchmark prediction (AnnData; written to data/h1_benchmark/pred_<tag>.h5ad unless write=False).
    P: (effects, targets, genes) from pooled_effects or oracle_effects (default: pooled_effects(cfg.model))."""
    m = cfg.model
    rng = np.random.default_rng(cfg.seed)
    ctrl = ad.read_h5ad(BM / "h1_controls.h5ad")
    X0 = ctrl.X.tocsr().astype(np.int32)
    axis = np.array(ctrl.var_names.astype(str))
    print(f"controls {X0.shape}, nnz/cell {X0.nnz / X0.shape[0]:.0f}", flush=True)
    mu = np.asarray(X0.mean(0)).ravel()
    size = np.asarray(X0.sum(1)).ravel()
    size_f = size / size.mean()
    bench = benchmark_targets()
    if m.rung != "zero":
        Pm, uni_targets, uni_genes = P if P is not None else pooled_effects(m)
        print(f"effect |log2| quantiles 50/99/99.99%: {np.round(np.nanquantile(np.abs(Pm), [0.5, 0.99, 0.9999]), 3)}", flush=True)
        pos = {g: i for i, g in enumerate(axis)}
        col = np.array([pos.get(g, -1) for g in uni_genes])
        keep = col >= 0
        tix = {t: i for i, t in enumerate(uni_targets)}
        ones = np.ones(len(axis))
        print(f"{int(keep.sum())} of {len(uni_genes)} universe genes are on the H1 axis; rung {m.rung}", flush=True)
    blocks, labels = [], []
    for k, t in enumerate(bench):
        pick = rng.integers(0, X0.shape[0], 400)
        M = X0[pick].tocsr()
        if m.rung != "zero" and t in tix:
            eff = EF.target_effect(Pm[tix[t]], col, keep, ones, None, m.min_effect, len(axis))
            M = emitter(M, eff, mu, size_f[pick], rng)
        blocks.append(M.astype(np.float32))
        labels += [t] * 400
        if (k + 1) % 20 == 0:
            print(f"   {k + 1}/{len(bench)} targets", flush=True)
    X = sp.vstack(blocks).tocsr()
    a = ad.AnnData(X=X, obs=pd.DataFrame({"target_gene": pd.Categorical(labels)}, index=[f"pred{i}" for i in range(len(labels))]),
                   var=pd.DataFrame(index=pd.Index(axis)))
    if write:
        out = Path(out or BM / f"pred_{cfg.tag}.h5ad")
        a.write_h5ad(out)
        print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB): {a.n_obs} cells x {a.n_vars} genes, nnz/cell {X.nnz / a.n_obs:.0f}")
    return a


def run(cfg: RunConfig, P=None, emitter=emit_h1, keep: bool = False) -> dict:
    """Generate, score with vcc-h1 (about 25 min), delete the 3.5 GB prediction unless keep; returns the scores.
    The scorer output stays in OUT/h1_score_<tag> (read again with read_scores)."""
    cfg.activate()
    pred = BM / f"pred_{cfg.tag}.h5ad"
    generate(cfg, pred, P=P, emitter=emitter)
    try:
        return score(pred, cfg.tag)
    finally:
        if not keep:
            pred.unlink(missing_ok=True)


def score(pred: Path, tag: str, validate: bool = True) -> dict:
    """Run vcc-h1 (validate, score) on a prediction; returns {short metric name: from_replicate, 'overall': mean of six}."""
    out = config.OUT / f"h1_score_{tag}"
    if validate:
        subprocess.run([str(VCC_H1), "validate", str(pred), "--data-dir", str(BM)], check=True)
    subprocess.run([str(VCC_H1), "score", str(pred), "--data-dir", str(BM), "--output", str(out)], check=True)
    return read_scores(out)


def read_scores(folder: Path) -> dict:
    s = pd.read_csv(Path(folder) / "scores.csv").set_index("metric").from_replicate
    res = {k: float(s[v]) for k, v in METRICS.items()}
    res["overall"] = float(np.mean([res[k] for k in METRICS]))
    return res


def main(cfg: RunConfig, argv) -> int:
    import argparse
    from dataclasses import replace
    ap = argparse.ArgumentParser(prog="vcell evaluate h1")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--line-weight", action="append", default=[], metavar="NAME=W", help="e.g. Pan_H1=4 (H47)")
    ap.add_argument("--no-score", action="store_true", help="only write the prediction")
    a = ap.parse_args(argv)
    if a.seed is not None:
        cfg = replace(cfg, seed=a.seed, tag=f"{cfg.tag}_s{a.seed}")
    if a.line_weight:
        lw = {**cfg.model.line_weights, **{kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.line_weight}}
        cfg = replace(cfg, model=replace(cfg.model, line_weights=lw))
    pred = BM / f"pred_{cfg.tag}.h5ad"
    generate(cfg, pred)
    if not a.no_score:
        res = score(pred, cfg.tag)
        print(json.dumps({k: round(v, 4) for k, v in res.items()}))
    return 0
