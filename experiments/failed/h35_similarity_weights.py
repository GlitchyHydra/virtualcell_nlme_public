"""H35 (rejected): weight the source lines by how similar their control profiles are to the new context, continuously;
with KOLF in the pool this doubles as a per-context selector. w_l = exp(beta (r_l - max r)), r_l the Spearman correlation
of line l's control CPM with the held-out line's (CD4+ T: the DESeq2 baseMean profile). beta = 0 is equal weights. The
weights act on the combination only, after an unweighted tau^2 (Panel.rungs(lam=...)).

Verdict (EXPERIMENT_LOG H35): rejected. Uploaded 0.1676; beta 0 with KOLF 0.1657, beta 10 0.1571, 20 0.1426, 40 0.1320,
80 0.1296; without KOLF beta 20 0.1560, 40 0.1500. The sharper the weights, the worse, in 7 of 8 lines; averaging more
lines beats choosing similar ones (as H13). A first run inflated s instead, which biases tau^2 low; the fix is in the
pipeline (pool.pool(lam=...)).
Was the H35 part of eda/43_similarity_bins.py (the H36 part is experiments/h36_bin_scale.py). Snapshot abc. 10 minutes.

    python experiments/failed/h35_similarity_weights.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vcell import config
from vcell.evaluation.harness import KI, Harness
from vcell.evaluation.panel import standardised
from vcell.qc.identity import cd4_ctrl_profile

config.use_snapshot("abc")
H = Harness()
N_MAIN = KI + 1                                     # the five main lines and KOLF, as in h36_bin_scale.py (no Pan_H1, Feng_iPSC)
CD4 = dict(zip(H.main.genes, cd4_ctrl_profile()))
STATES = {"main": (H.main, [0, 1, 2, 3, 4], H.main.lines[:N_MAIN]), "ess": (H.ess, [4, 5, 6], H.ess.lines)}


def profile(P, name):
    if name.startswith("CD4"):
        return np.array([CD4.get(g, np.nan) for g in P.genes], dtype=float)
    return P.tabs[name]["ctrl_cpm"].astype(float)


SIM = {}
for axis, (P, held, lines) in STATES.items():
    prof = [profile(P, n) for n in lines]
    L = len(prof)
    r = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            m = np.isfinite(prof[i]) & np.isfinite(prof[j]) & ((prof[i] > 0) | (prof[j] > 0))
            r[i, j] = r[j, i] = spearmanr(prof[i][m], prof[j][m])[0]
    SIM[axis] = r


def w_uploaded(axis, hi):
    w = np.ones(len(STATES[axis][2]))
    w[hi] = 0
    if axis == "main":
        w[KI] = 0
    return tuple(w)


def w_sim(beta, kolf=True):
    def f(axis, hi):
        r = SIM[axis][hi].copy()
        src = np.ones(len(r), bool)
        src[hi] = False
        if axis == "main" and not kolf:
            src[KI] = False
        w = np.where(src, np.maximum(np.exp(beta * (r - r[src].max())), 1e-3), 0.0)
        return tuple(np.round(w, 4))
    return f


CACHE = {}


def pooled(axis, hi, w):
    key = (axis, hi, w)
    if key not in CACHE:
        tr = [i for i, x in enumerate(w) if x > 0]
        lam = None if all(w[i] == 1.0 for i in tr) else [max(w[i], 1e-3) for i in tr]
        CACHE[key] = standardised(STATES[axis][0].rungs(tr, lam=lam))
    return CACHE[key]


recs, REF = [], None


def run(label, c=1.5, wfun=w_uploaded):
    global REF
    v = {P.lines[hi]: H.score(P, c * pooled(axis, hi, wfun(axis, hi)), hi, rows) for axis, P, hi, rows, _ in H.cases()}
    m = float(np.mean(list(v.values())))
    REF = REF or v
    dl = {k: v[k] - REF[k] for k in v}
    up = sum(x > 0.0005 for x in dl.values())
    down = sum(x < -0.0005 for x in dl.values())
    recs.append({"label": label, "c": c, "mult": None, **v, "mean_all8": m})
    pd.DataFrame(recs).to_csv(config.results("h35_similarity_weights.csv"), index=False)
    print(f"{label:38s} all8 {m:.4f}  ({m - np.mean(list(REF.values())):+.4f}; lines up {up}, down {down})   "
          + " ".join(f"{k} {dl[k]:+.4f}" for k in v), flush=True)
    return m


if __name__ == "__main__":
    for axis, (P, held, lines) in STATES.items():
        print(f"Spearman of control profiles, {axis} axis")
        print(pd.DataFrame(SIM[axis], index=lines, columns=lines).round(3).to_string(), flush=True)
    print("\nweights per held-out line at beta 40 (KOLF in the pool):")
    for axis, (P, held, lines) in STATES.items():
        for hi in held:
            print(f"   {lines[hi]:9s} " + " ".join(f"{n}:{w:.2f}" for n, w in zip(lines, w_sim(40)(axis, hi))))
    print("\nH35 similarity weights (objective = mean proxy.est over 8 held-out lines; deltas vs the uploaded model)")
    base = run("uploaded (equal, no KOLF)")
    best_sim, best_s = None, base
    for beta in (0, 10, 20, 40, 80):
        s = run(f"beta {beta}, KOLF in pool", wfun=w_sim(beta))
        if s > best_s:
            best_sim, best_s = beta, s
    for beta in (20, 40):
        run(f"beta {beta}, no KOLF", wfun=w_sim(beta, kolf=False))
    print(f"\nbest similarity beta: {best_sim} ({best_s:.4f}); uploaded {base:.4f}")
