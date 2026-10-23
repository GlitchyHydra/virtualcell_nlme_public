"""Stage 1 tables from the uniformly processed CRISPRi screens of HKU DataHub (doi 10.25442/hku.34002315, CC BY-NC-SA 4.0;
data/sources/hku/effect_<ds>_moments.h5ad; was eda/48_hku_tables.py): Pan-GW-hESC (H1, another lab) and Feng-gwsf +
Feng-gwsnf + Feng-ts (iPSC, one line). Nourreddine-GW-ipsc is the KOLF2.1J atlas built from the raw file (etl.stage1.kolf).

Their effect E is a difference of mean log1p(1e4-normalised) expression; ours is log2 of a ratio of mean CPM. Conversion:
beta = k(bin) * E / (ln2 * mu / (1 + mu)), mu = expm1(control mean log1p), with k per control-expression bin calibrated on
VCC-H1, which exists in both forms (their effect_VCC vs our stage1_H1). s = effect_se scaled the same way. The target and
gene axis is read from stage1_K562, so build K562 first.
"""
from __future__ import annotations

import h5py
import numpy as np

from vcell import config, io
from vcell.etl.stage1.table import load, save

HKU = config.SOURCES / "hku"
TABLES = {"Pan_H1": ["Pan-GW-hESC"], "Feng_iPSC": ["Feng-gwsf", "Feng-gwsnf", "Feng-ts"]}
BIN_EDGES = [0.02, 0.05, 0.1, 0.3, 1.0, 3.0]             # control mean log1p (1e4-normalised)


def idx(g):
    a = g["_index"] if "_index" in g else g[g.attrs["_index"]]
    return np.array(io.decode(a[...]), dtype=object)


def read(ds, targets, genes):
    """E, SE, control mean log1p and n on our (targets, genes) axis; NaN where absent."""
    with h5py.File(HKU / f"effect_{ds}_moments.h5ad", "r") as f:
        p, g = idx(f["obs"]), idx(f["var"])
        pix, gix = {t: i for i, t in enumerate(p)}, {x: j for j, x in enumerate(g)}
        rows = [(i, pix[t]) for i, t in enumerate(targets) if t in pix]
        col = np.array([gix.get(x, -1) for x in genes])
        have = col >= 0
        E = np.full((len(targets), len(genes)), np.nan)
        SE = np.full_like(E, np.nan)
        src_rows = np.array([r for _, r in rows])
        order = np.argsort(src_rows)
        srt = src_rows[order]
        Xe = f["X"][srt, :][:, col[have]]
        Xs = f["layers"]["effect_se"][srt, :][:, col[have]]
        dst = np.array([i for i, _ in rows])[order]
        E[np.ix_(dst, np.flatnonzero(have))] = Xe
        SE[np.ix_(dst, np.flatnonzero(have))] = Xs
        cp = np.full(len(genes), np.nan)
        cp[have] = np.asarray(f["uns"]["control_profile"][...]).ravel()[col[have]]
        obs_cols = list(f["obs"].keys())
        n = np.full(len(targets), np.nan)
        for key in ("n_cells", "n_pert_cells", "n"):
            if key in obs_cols:
                nn = f["obs"][key][...]
                n[dst] = nn[srt]
                break
    for i, t in enumerate(targets):                          # the target's own column, as in our tables
        if t in genes:
            E[i, genes.index(t)] = np.nan
    print(f"{ds}: {len(rows)} axis targets; obs columns {obs_cols[:12]}", flush=True)
    return E, SE, cp, n


def to_log2(E, SE, cp, k):
    mu = np.expm1(cp)
    d = np.log(2) * mu / (1 + mu)
    kb = np.asarray(k)[np.digitize(cp, BIN_EDGES)]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(d > 1e-4, kb / d, np.nan)
    return E * f[None, :], SE * f[None, :]


def build(tables=tuple(TABLES)) -> None:
    ref = load("K562")
    TARGETS, GENES = list(ref["targets"]), list(ref["genes"])

    E, SE, cp, n = read("VCC", TARGETS, GENES)
    h1 = load("H1")
    ones = np.ones(len(BIN_EDGES) + 1)
    b0, s0 = to_log2(E, SE, cp, ones)
    bins = np.digitize(cp, BIN_EDGES)
    k = np.ones(len(BIN_EDGES) + 1)
    print("calibration against our H1 table (pairs with |beta/s| > 3 in ours): slope of ours on theirs, r, ratio of s")
    for b in range(len(k)):
        m = (bins == b)[None, :] & np.isfinite(b0) & np.isfinite(h1["beta"]) & (np.abs(h1["beta"] / h1["s"]) > 3)
        if m.sum() < 50:
            print(f"   bin {b}: {int(m.sum())} pairs, k = 1")
            continue
        x, y = b0[m], h1["beta"][m].astype(float)
        k[b] = float((x * y).sum() / (x * x).sum())
        r = np.corrcoef(x, y)[0, 1]
        sr = np.nanmedian(h1["s"][(bins == b)[None, :] & np.isfinite(s0)] / s0[(bins == b)[None, :] & np.isfinite(s0)])
        print(f"   bin {b} (cp {([0] + BIN_EDGES)[b]}-): {int(m.sum()):6d} pairs  slope k {k[b]:.3f}  r {r:.3f}  "
              f"median s ours/theirs {sr:.3f}")
    np.save(config.OUT / "hku_k.npy", k)
    b1, _ = to_log2(E, SE, cp, k)
    m = np.isfinite(b1) & np.isfinite(h1["beta"])
    print(f"all pairs after calibration: r {np.corrcoef(b1[m], h1['beta'][m])[0, 1]:.3f}")

    # ------------------------------------------------------------------ tables
    for name in tables:
        parts = TABLES[name]
        acc_b = np.zeros((len(TARGETS), len(GENES)))
        acc_w = np.zeros_like(acc_b)
        cps, ns = [], np.zeros(len(TARGETS))
        for ds in parts:
            E, SE, cp, n = read(ds, TARGETS, GENES)
            b, s = to_log2(E, SE, cp, k)
            ok = np.isfinite(b) & np.isfinite(s) & (s > 0)
            w = np.where(ok, 1.0 / np.where(ok, s, 1.0) ** 2, 0.0)
            acc_b += np.where(ok, b, 0.0) * w
            acc_w += w
            cps.append(cp)
            ns += np.nan_to_num(n)
        beta = np.where(acc_w > 0, acc_b / np.maximum(acc_w, 1e-30), np.nan)
        s = np.where(acc_w > 0, 1.0 / np.sqrt(np.maximum(acc_w, 1e-30)), np.nan)
        ctrl_cpm = np.expm1(np.nanmean(np.stack(cps), 0)) * 100.0     # per 1e4 -> per 1e6
        n_t = np.where(ns > 0, ns, np.where(np.isfinite(beta).any(1), 100.0, np.nan))
        save(name, TARGETS, GENES, beta, s, n_t, ctrl_cpm)
