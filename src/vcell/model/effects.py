"""The effect recipe: Stage 1 tables -> one log2 effect per (context, panel target, gene). This is the trained model.

Was the effects() function and the per-target loop of eda/16_generate_abc.py. For a run config (configs/*.toml):
  1. fit       Stage 2 on the source lines with weight > 0, in sorted name order (model.pool.predict; lam = the weights on
               the combination only, tau^2 unweighted), then the rung, the transform (z: model.pool.z_effect) and the scale;
               one fit per distinct weight set (the defaults, plus each context with its own weights)
  2. post      optional steps on the whole effect matrix (experiments only, e.g. the shared-pattern removal of H51)
  3. context   per context and target: the effect on the panel gene axis, times the bin multiplier of the gene's control
               CPM in that context (H36), capped for promoter neighbours (H37), effects below min_effect set to 0
The result (Effects) holds step 3 for every panel target, saved as OUT/effects_<tag>.npz; model.generate turns it into cells.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vcell import config, io, stats
from vcell.config import ModelConfig, RunConfig
from vcell.etl.coverage import missing_rows
from vcell.etl.stage1.table import load
from vcell.model import pool, promoter

BIN_EDGES = [15, 40, 120, 400]                         # control CPM bins of bin_mult: <15, 15-40, 40-120, 120-400, >400


def load_tables(names, tables=None) -> dict:
    tabs = {n: load(n, tables) for n in names}
    first = next(iter(tabs.values()))
    for n, t in tabs.items():
        assert list(t["genes"]) == list(first["genes"]) and list(t["targets"]) == list(first["targets"]), \
            f"{n} is on a different axis"
    return tabs


def weight_key(weights: dict) -> tuple:
    """The lines with weight > 0 in sorted name order: the order of the stacked arrays, which fixes the float sums."""
    return tuple(sorted((n, w) for n, w in weights.items() if w > 0))


def fit(tabs: dict, key: tuple, max_cpm: np.ndarray, m: ModelConfig, post=None) -> np.ndarray:
    """Effect matrix (Stage 1 targets x universe genes) from the lines of `key` ((name, weight) pairs, in that order)."""
    use = [n for n, _ in key]
    lam = [w for _, w in key]
    out = pool.predict(np.stack([tabs[n]["beta"] for n in use]).astype(float),
                       np.stack([tabs[n]["s"] for n in use]).astype(float), max_cpm,
                       lam=None if all(w == 1.0 for w in lam) else lam, kappa=m.kappa, ash=m.rung == "v3b")
    P = out[m.rung]["pred"]
    if m.transform == "z":
        P = pool.z_effect(out["v3"]["pred"] if m.rung == "v3" else P, out["v3"]["sd"], out["v3"]["K"] >= 1, m.scale)
    elif m.scale != 1.0:
        P = m.scale * P
    for step in post or []:
        P = step(P, uni_targets=list(tabs[use[0]]["targets"]))
    return P


def bin_multipliers(ctx_cpm: np.ndarray, bin_mult) -> np.ndarray:
    return np.asarray(bin_mult)[np.digitize(ctx_cpm, BIN_EDGES)]


def target_effect(P_row, col, keep, mult, nb, min_effect: float, n_axis: int) -> np.ndarray:
    """One target's emitted effect on the panel gene axis (eda/16, in its order: bins, then the neighbour cap, then the floor)."""
    eff = np.zeros(n_axis)
    eff[col[keep]] = np.nan_to_num(P_row[keep])
    eff *= mult
    if nb is not None:
        cols, cap = nb
        eff[cols] = np.minimum(eff[cols], cap)
    eff[np.abs(eff) < min_effect] = 0.0
    return eff


@dataclass
class Effects:
    """Per context, the emitted effect of every panel target on the universe genes that are on the panel axis."""
    tag: str
    contexts: list
    targets: list                 # panel targets, in pert_counts.csv order
    has_row: np.ndarray           # (T,) the target has a Stage 1 row (else its cells are unchanged resamples)
    axis: list                    # the panel gene axis (gene_names.csv)
    cols: np.ndarray              # axis positions of the stored columns
    eff: dict                     # context -> (T, len(cols)) float64
    config: dict

    def vector(self, ctx: str, k: int) -> np.ndarray:
        e = np.zeros(len(self.axis))
        e[self.cols] = self.eff[ctx][k]
        return e

    def save(self, path=None) -> Path:
        path = Path(path or config.OUT / f"effects_{self.tag}.npz")
        np.savez_compressed(path, tag=self.tag, contexts=np.array(self.contexts), targets=np.array(self.targets),
                            has_row=self.has_row, axis=np.array(self.axis), cols=self.cols,
                            config=json.dumps(self.config), **{f"eff_{c}": v for c, v in self.eff.items()})
        print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
        return path

    @classmethod
    def load(cls, path) -> "Effects":
        z = np.load(path, allow_pickle=False)
        ctx = [str(c) for c in z["contexts"]]
        return cls(str(z["tag"]), ctx, [str(t) for t in z["targets"]], z["has_row"], [str(g) for g in z["axis"]], z["cols"],
                   {c: z[f"eff_{c}"] for c in ctx}, json.loads(str(z["config"])))


def train(cfg: RunConfig, post=None, limit=None, contexts=None) -> Effects:
    """The effect of every panel target in every context of cfg.controls (limit: the first n targets only)."""
    cfg.activate()
    m = cfg.model
    man = config.manifest()
    ctxs = list(contexts or man["contexts"])
    axis = list(config.gene_names())
    targets = config.panel_targets()[:limit]
    T = len(targets)
    if m.rung == "zero":                                   # unchanged resamples: the scorer's lower anchor
        return Effects(cfg.tag, ctxs, targets, np.zeros(T, bool), axis, np.zeros(0, np.int64),
                       {c: np.zeros((T, 0)) for c in ctxs}, _cfg_dict(cfg))
    W0 = m.weights()
    WCTX = m.context_weights
    names = sorted({n for n in W0} | {n for w in WCTX.values() for n in w})
    tabs = load_tables(names)
    uni_genes = list(tabs[names[0]]["genes"])
    uni_targets = list(tabs[names[0]]["targets"])
    gl = config.gene_lines().reindex(uni_genes)
    max_cpm = gl.max_cpm.to_numpy()
    fits = {}

    def effects(weights: dict) -> np.ndarray:
        key = weight_key(weights)
        if key not in fits:
            fits[key] = fit(tabs, key, max_cpm, m, post)
            print(f"effects for lines {dict(key)}", flush=True)
        return fits[key]

    P = effects(W0)
    pctx = {c: effects({**W0, **{n: float(v) for n, v in WCTX[c].items()}}) for c in WCTX}
    print(f"effect |log2| quantiles 50/99/99.99%: {np.round(np.nanquantile(np.abs(P), [0.5, 0.99, 0.9999]), 3)}", flush=True)
    pos = {g: i for i, g in enumerate(axis)}
    col = np.array([pos.get(g, -1) for g in uni_genes])   # universe gene -> panel axis column
    keep = col >= 0
    missing = missing_rows(uni_targets, targets)
    print(f"{int(keep.sum())} of {len(uni_genes)} universe genes on the panel axis; rung {m.rung}; lines {W0}; "
          f"per context {WCTX}; {len(missing)} panel targets without a Stage 1 row", flush=True)
    nb = {}                                                # target -> (panel axis columns, log2 caps)
    if m.neighbour is not None:
        pairs = promoter.neighbour_pairs(uni_targets, uni_genes)
        cap = promoter.neighbour_cap(pairs, m.neighbour)
        print(f"neighbour rule f {m.neighbour}: {len(pairs)} target-gene pairs within 5 kb", flush=True)
        for ti, rows in pairs.groupby("ti").indices.items():
            c = col[pairs.gj.to_numpy()[rows]]
            ok = c >= 0
            nb[uni_targets[ti]] = (c[ok], cap[rows][ok])
    uni_set = set(uni_targets)
    has_row = np.array([t in uni_set for t in targets], dtype=bool)
    tix = {t: i for i, t in enumerate(uni_targets)}
    cols = col[keep]
    eff = {}
    for ctx in ctxs:
        Pc = pctx.get(ctx, P)
        mult = np.ones(len(axis))
        if m.bin_mult is not None:                         # bins of the gene's CPM in this context
            X0, _, genes = io.load_context(ctx)
            assert list(genes) == axis, f"context {ctx}: gene axis differs from gene_names.csv"
            cpm = stats.context_gene_cpm(X0.tocsr().astype(np.int32))
            del X0
            mult = bin_multipliers(cpm, m.bin_mult)
            print(f"   {ctx} bin multipliers: genes per bin {np.bincount(np.digitize(cpm, BIN_EDGES), minlength=5)}", flush=True)
        E = np.zeros((T, len(cols)))
        for k, t in enumerate(targets):
            if has_row[k]:
                E[k] = target_effect(Pc[tix[t]], col, keep, mult, nb.get(t), m.min_effect, len(axis))[cols]
        eff[ctx] = E
    return Effects(cfg.tag, ctxs, targets, has_row, axis, cols, eff, _cfg_dict(cfg))


def _cfg_dict(cfg: RunConfig) -> dict:
    d = asdict(cfg)
    d["model"]["bin_mult"] = list(d["model"]["bin_mult"]) if d["model"]["bin_mult"] else None
    d["model"]["sources"] = list(d["model"]["sources"])
    return d
