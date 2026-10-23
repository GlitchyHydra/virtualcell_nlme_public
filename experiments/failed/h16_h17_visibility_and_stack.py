"""H16 / H17 (rejected): Two directions in the proxy, leave-one-line-out, A/B/C-like mix of target strengths (as failed/h13_h15_line_groups_adaptive_scale.py).

A. Visibility. pds, mse and the DE log2FC use only means; the Wilcoxon calls (fid, jac, reach) also depend on how the shift
   is spread over cells. So the generator could make a chosen set V of genes clearly significant (uniform shift) and hide
   the mean shift of all other genes from the test (counts moved in a few cells only). Idealised here: V = the n genes with
   the largest |z| = |theta| / sd, called significant with sign(effect) in |z| order; all other genes get uniform p-values.
   Magnitudes (pds, nmae) stay those of z x1.5.
B. Meta-predictor. Stacked linear regression of a line's real effect on features built from the other lines (v3, v1,
   per-line median, standardised z, sign agreement), fitted leave-one-line-out inside the training lines, then applied.

    python experiments/failed/h16_h17_visibility_and_stack.py

Verdict (EXPERIMENT_LOG H16, H17): visibility (only the top-n genes get a rank-visible shift) loses for every n and rule
(0.003-0.128 vs 0.146; an oracle n gains only +0.035): the bottleneck is per-gene accuracy, not the call count. The
stacked meta-predictor shrinks to about 0.1x and makes no DE calls (-0.125; resized 0.138, blend 0.144 vs 0.146).
Was eda/23_visibility.py. Snapshot abc (the A/B/C validation tables). 5 minutes. The cell-level visibility generator is experiments/failed/visibility_generator.py.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtr

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
BINS = [0.5, 10, 40, 150, 500, 1e9]
LAB = ["1-10", "11-40", "41-150", "151-500", ">500"]
MIX = dict(zip(LAB, [0.2, 0.3, 0.25, 0.15, 0.1]))


def de_score(d):
    return ((d["fid"] - 0.5) / proxy.SPAN["fid"] + d["jac"] / proxy.SPAN["jac"] + d["reach"] / proxy.SPAN["reach"]
            + (np.nan_to_num(d["nmae"], nan=1.0) - 1.0) / proxy.SPAN["nmae"])


def mixed(d):
    s = pd.Series(de_score(d))
    b = pd.cut(pd.Series(d["n_real"]), BINS, labels=LAB)
    per = s.groupby(b, observed=False).mean()
    return float(sum(MIX[k] * per[k] for k in LAB if np.isfinite(per[k])))


def mixed_parts(d):
    b = pd.cut(pd.Series(d["n_real"]), BINS, labels=LAB)
    out = {}
    for k in ["fid", "jac", "reach"]:
        per = pd.Series(d[k]).groupby(b, observed=False).mean()
        out[k] = float(sum(MIX[c] * per[c] for c in LAB if np.isfinite(per[c])))
    return out


def de_vis(eff, conf, n_sel, hi, rows, seed=0):
    """X.de with our calls replaced: the n_sel[k] genes of largest |conf| are significant (sign of eff), the rest null."""
    rng = np.random.default_rng(seed)
    testable = X.ctrl_cpm(hi) > 5
    F, J, R, M, nr, npred = [], [], [], [], [], []
    for k, i in enumerate(rows):
        m = testable & np.isfinite(X.B[hi, i]) & np.isfinite(X.S[hi, i]) & np.isfinite(eff[i])
        b, s = X.B[hi, i, m], X.S[hi, i, m]
        n_t = X.NC[hi, i] if np.isfinite(X.NC[hi, i]) else 400.0
        s4 = s * np.sqrt(max(n_t, 1.0) / 400.0)
        if n_t > 400:
            b = b + rng.normal(0, np.sqrt(np.maximum(s4 ** 2 - s ** 2, 0)))
        sig_r = proxy.bh(2 * ndtr(-np.abs(b / s4)))
        e = eff[i, m].copy()
        e[np.abs(e) < X.min_effect] = 0.0
        obs = e + rng.normal(0, s4)
        c = np.abs(np.nan_to_num(conf[i, m]))
        c[e == 0] = 0.0
        n = int(min(n_sel[k], (c > 0).sum()))
        top = np.argsort(-c)[:n]
        p_o = rng.uniform(0, 1, len(b))
        p_o[top] = 1e-12 * (1 + np.arange(n))
        sgn = np.sign(obs)
        sgn[top] = np.sign(e[top])
        sig_o = proxy.bh(p_o)
        n_r, n_o = sig_r.sum(), sig_o.sum()
        hit = (sig_o & (sgn == np.sign(b))).sum()
        F.append(hit / max(n_r, n_o) if max(n_r, n_o) else 1.0)
        u = (sig_r | sig_o).sum()
        J.append((sig_r & sig_o).sum() / u if u else 1.0)
        r_, m_ = np.nan, np.nan
        if n_r:
            idx = np.flatnonzero(sig_r)
            idx = idx[np.lexsort((p_o[idx], ~sig_o[idx]))]
            good = np.cumsum(sgn[idx] == np.sign(b[idx])) / np.arange(1, n_r + 1)
            r_ = (np.flatnonzero(good >= 0.9).max() + 1) / n_r if (good >= 0.9).any() else 0.0
            if n_r >= 10:
                m_ = np.abs(obs[idx] - b[idx]).sum() / np.abs(b[idx]).sum()
        R.append(r_)
        M.append(m_)
        nr.append(n_r)
        npred.append(n_o)
    return {"fid": np.array(F), "jac": np.array(J), "reach": np.array(R), "nmae": np.array(M),
            "n_real": np.array(nr), "n_ours": np.array(npred)}


def features(tr):
    R = X.rungs(tr)
    v3, v1, sd = R["v3"], R["v1"], R["sd"]
    ref = np.nanmedian(np.where(R["has"], sd, np.nan))
    Bt = X.B[tr]
    med = np.nanmedian(Bt, axis=0)
    med = np.where(np.isfinite(med), med, v3)
    agree = np.nanmean(np.sign(Bt), axis=0)
    agree = np.where(np.isfinite(agree), agree, 0.0) * np.abs(v3)
    zs = ref * v3 / sd
    return R, {"v3": v3, "v1": v1, "med": med, "zs": zs, "agree": agree}


FEATS = ["v3", "v1", "med", "zs", "agree"]


def stack_fit(tr):
    """Weights of the stack: for each line j in tr, features from tr minus j predict line j's real effect."""
    Xs, ys = [], []
    for j in tr:
        inner = [i for i in tr if i != j]
        _, F = features(inner)
        rows = X.rows_for(j)
        m = (X.ctrl_cpm(j) > 5)[None, :] & np.isfinite(X.B[j][rows])
        Xs.append(np.stack([F[f][rows][m] for f in FEATS], 1).astype(np.float32))
        ys.append(X.B[j][rows][m].astype(np.float32))
    A, y = np.concatenate(Xs), np.concatenate(ys)
    A = np.nan_to_num(A)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w


recs = []
for hi, name in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    rows = X.rows_for(hi)
    R, F = features(tr)
    z = R["v3"] / R["sd"]
    eff = 1.5 * F["zs"]
    strength = (np.abs(z) > 3).sum(1).astype(float)[rows]

    def rec(variant, d, pds_val):
        recs.append({"held_out": name, "variant": variant, "pds": pds_val, "de_mix": mixed(d), **mixed_parts(d),
                     "n_ours_med": float(np.median(d["n_ours"])), "n_real_med": float(np.median(d["n_real"]))})

    p_ref = X.pds(eff, hi, rows)
    d0 = X.de(eff, hi, rows, per_target=True)
    rec("A z x1.5 natural", d0, p_ref)
    n_real = d0["n_real"].astype(float)
    for n in (5, 10, 20, 40, 80, 160, 320):
        rec(f"A vis top {n}", de_vis(eff, z, np.full(len(rows), n), hi, rows), p_ref)
    for c in (0.25, 0.5, 1.0, 2.0):
        rec(f"A vis {c} x strength", de_vis(eff, z, np.clip(c * strength, 5, None), hi, rows), p_ref)
    for t in (3, 5, 8):
        rec(f"A vis |z|>{t}", de_vis(eff, z, np.maximum((np.abs(z[rows]) > t).sum(1), 5), hi, rows), p_ref)
    rec("A vis oracle n_real", de_vis(eff, z, np.maximum(n_real, 1), hi, rows), p_ref)
    rec("A vis oracle n_real, |eff| order", de_vis(eff, eff, np.maximum(n_real, 1), hi, rows), p_ref)

    w = stack_fit(tr)
    st = sum(wk * np.nan_to_num(F[f]) for wk, f in zip(w, FEATS))
    print(f"{name}: stack weights " + ", ".join(f"{f} {wk:+.3f}" for f, wk in zip(FEATS, w)), flush=True)
    k = np.nanmedian(np.abs(eff[rows])) / max(np.nanmedian(np.abs(st[rows])), 1e-9)
    for vname, e2 in {"B stack x1": st, "B stack, size of z x1.5": k * st,
                      "B 0.5 z x1.5 + 0.5 stack-sized": 0.5 * eff + 0.5 * k * st}.items():
        rec(vname, X.de(e2, hi, rows, per_target=True), X.pds(e2, hi, rows))
    print(f"{name} done", flush=True)

df = pd.DataFrame(recs)
df.to_csv(config.results("h16_h17_visibility_and_stack.csv"), index=False)
df["est"] = (df.pds - 0.5) / proxy.SPAN["pds"] / 6 + df.de_mix / 6
for col in ["est", "fid", "jac", "reach", "n_ours_med"]:
    print(f"\n{col}")
    pv = df.pivot_table(index="variant", columns="held_out", values=col, sort=False)[X.lines]
    pv["mean"] = pv.mean(1)
    print(pv.to_string(float_format=lambda x: f"{x:.3f}"))
