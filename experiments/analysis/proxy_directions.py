"""DE proxy development: Fast checks of candidate directions with proxies for pds and the four DE metrics (leave one line out).

DE proxy, per held-out line L and target t (genes testable in L: control CPM > 5, finite estimate):
  real:  the Stage 1 estimate brought to 400 cells (extra noise added when L had more cells), z = beta / s400, Benjamini-Hochberg 0.05
  ours:  the generator's effect plus the noise of 400 cells, observed = eff + N(0, s400^2), same test
  fid = our significant genes whose sign matches the real log2FC (significant there or not) / max(n_ours, n_real), as in
  cell_eval2/metrics/direction.py (k / max(n_pred, N_conf)); jac = |both| / |either| (empty union 1); reach = deepest prefix of the real
  significant genes, ordered by our p-value, with sign purity >= 0.9, over n_real; nmae = sum |obs - real| / sum |real| on the real
  significant genes (targets with at least 10).
Validation against the real scorer on H1 (raw values): zero fid 0.000 jac 0.000; v1 0.258 / 0.019; v3 0.484 / 0.091; v3b 0.178 / 0.010.

    python experiments/analysis/proxy_directions.py

Result (EXPERIMENT_LOG 'Proxy toolkit'): the DE proxies land within about 0.01-0.02 of the scorer on H1 (raw fid / jac /
reach / nmae). This is the first version of the DE proxy (reach ordered by p-value only); evaluation.panel.Panel.de
orders our significant genes first, as cell-eval2 does. Also: single-line transfer, transforms, drop one line.
Was eda/19_directions.py. Snapshot abc (the A/B/C validation tables). 5 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtr

from vcell import config
from vcell.etl.stage1.table import load, benchmark_targets
from vcell.model import pool

config.use_snapshot("abc")
warnings.filterwarnings("ignore")
LINES = ["K562", "CD4_Rest", "HCT116", "HEK293T", "H1"]
tabs = {n: load(n) for n in LINES}
genes = list(tabs["K562"]["genes"])
targets = list(tabs["K562"]["targets"])
B = np.stack([tabs[n]["beta"] for n in LINES]).astype(float)
S = np.stack([tabs[n]["s"] for n in LINES]).astype(float)
NC = np.stack([tabs[n]["n"] for n in LINES]).astype(float)
gl = config.gene_lines().reindex(genes)
max_cpm = gl.max_cpm.to_numpy()
panel = set(config.panel_targets())
bench = set(benchmark_targets())
tset = set(targets)
not_target = np.array([g not in tset for g in genes])
MIN_EFFECT = 0.05


def rungs(tr, weights=None):
    """v1, v3 and the predictive sd from lines `tr`; `weights` multiplies each line's precision (line weighting)."""
    Bt, St = B[tr], S[tr].copy()
    if weights is not None:
        St = St / np.sqrt(np.asarray(weights, float))[:, None, None]
    g, _ = pool.generic(Bt, St)
    T, G = Bt.shape[1:]
    v1 = np.broadcast_to(g, (T, G)).copy()
    P0 = pool.pool(Bt, St, tau2=0.0)
    t2 = pool.tau2_by_bin(P0["Q"], P0["K"], P0["C"], pool.gene_bins(max_cpm))
    P = pool.pool(Bt, St, tau2=t2[None, :])
    has = P0["K"] >= 1
    v3 = np.where(has, P["theta"], v1)
    sd = np.where(has, np.sqrt(P["var"] + t2[None, :]), np.nan)
    sd = np.where(np.isfinite(sd), sd, np.nanmax(sd))
    return {"g": g, "v1": v1, "v3": v3, "sd": sd, "has": has}


def ctrl_cpm(hi):
    c = tabs[LINES[hi]]["ctrl_cpm"].astype(float)
    return np.where(np.isfinite(c), c, max_cpm)


def wts(hi):
    x = ctrl_cpm(hi) / 20.0
    return (np.log(2) * x / (1 + x)) ** 2


def rows_for(hi, which="panel"):
    name = LINES[hi]
    ok = [i for i in range(len(targets)) if np.isfinite(B[hi, i]).sum() > 100]
    pick = bench if name == "H1" else panel
    return np.array([i for i in ok if targets[i] in pick])


def pds(pred, hi, rows):
    sw = np.sqrt(wts(hi))[not_target]
    P = np.nan_to_num(pred[rows][:, not_target]) * sw
    Y = np.nan_to_num(B[hi][rows][:, not_target]) * sw
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    sim = P @ Y.T
    d = np.diag(sim)[:, None]
    rank = (sim > d).sum(1) + 0.5 * ((sim == d).sum(1) - 1)
    return float(np.mean(1 - rank / (len(rows) - 1)))


def bh(p, alpha=0.05):
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, n + 1) / n
    out = np.zeros(n, bool)
    if ok.any():
        out[o[:np.flatnonzero(ok).max() + 1]] = True
    return out


def de(eff, hi, rows, seed=0):
    rng = np.random.default_rng(seed)
    testable = ctrl_cpm(hi) > 5
    F, J, R, M, nr, npred = [], [], [], [], [], []
    for i in rows:
        m = testable & np.isfinite(B[hi, i]) & np.isfinite(S[hi, i]) & np.isfinite(eff[i])
        b, s = B[hi, i, m], S[hi, i, m]
        n_t = NC[hi, i] if np.isfinite(NC[hi, i]) else 400.0
        s4 = s * np.sqrt(max(n_t, 1.0) / 400.0)
        if n_t > 400:                                              # bring the truth to 400 cells
            b = b + rng.normal(0, np.sqrt(np.maximum(s4 ** 2 - s ** 2, 0)))
        sig_r = bh(2 * ndtr(-np.abs(b / s4)))
        e = eff[i, m].copy()
        e[np.abs(e) < MIN_EFFECT] = 0.0
        obs = e + rng.normal(0, s4)
        p_o = 2 * ndtr(-np.abs(obs / s4))
        sig_o = bh(p_o)
        n_r, n_o = sig_r.sum(), sig_o.sum()
        hit = (sig_o & (np.sign(obs) == np.sign(b))).sum()                # cell_eval2: any real sign counts, not only real hits
        F.append(hit / max(n_r, n_o) if max(n_r, n_o) else 1.0)
        u = (sig_r | sig_o).sum()
        J.append((sig_r & sig_o).sum() / u if u else 1.0)
        if n_r:
            idx = np.flatnonzero(sig_r)
            idx = idx[np.argsort(p_o[idx])]
            good = np.cumsum(np.sign(obs[idx]) == np.sign(b[idx])) / np.arange(1, n_r + 1)
            R.append((np.flatnonzero(good >= 0.9).max() + 1) / n_r if (good >= 0.9).any() else 0.0)
            if n_r >= 10:
                M.append(np.abs(obs[idx] - b[idx]).sum() / np.abs(b[idx]).sum())
        else:
            R.append(0.0)
        nr.append(n_r)
        npred.append(n_o)
    return {"fid": np.mean(F), "jac": np.mean(J), "reach": np.mean(R), "nmae": np.mean(M) if M else np.nan,
            "n_real": np.median(nr), "n_ours": np.median(npred)}


def transforms(R):
    v3, sd, g = R["v3"], R["sd"], R["g"]
    z = v3 / sd
    ref = np.nanmedian(sd)
    js = np.clip(1 - 1 / np.maximum(z ** 2, 1e-9), 0, 1)
    out = {"zero": np.zeros_like(v3), "v1": R["v1"], "v3": v3, "v3 x0.5": 0.5 * v3, "v3 x2": 2 * v3,
           "JS": v3 * js, "z>1": np.where(np.abs(z) > 1, v3, 0.0), "z>2": np.where(np.abs(z) > 2, v3, 0.0),
           "z>3": np.where(np.abs(z) > 3, v3, 0.0), "prec": v3 * np.minimum(1.0, ref / sd),
           "z-scaled": z * ref, "v3-generic": v3 - g[None, :]}
    return out


def table(rows_out, cols):
    df = pd.DataFrame(rows_out)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}", columns=cols))
    return df


# ------------------------------------------------------------------ 0. validate the DE proxy on H1 with the four scored rungs
h = LINES.index("H1")
tr = [i for i in range(len(LINES)) if i != h]
full = pool.predict(B[tr], S[tr], max_cpm)
rows_h = rows_for(h)
print(f"0. DE proxy on H1 ({len(rows_h)} targets) against the scorer (raw fid / reach / jac / nmae):")
print("   scorer: zero 0.000/0.027/0.000/1.005  v1 0.258/0.064/0.019/1.027  v3 0.484/0.122/0.091/1.036  v3b 0.178/0.086/0.010/1.006")
out = []
for name, eff in [("zero", np.zeros_like(full["v3"]["pred"])), ("v1", full["v1"]["pred"]), ("v3", full["v3"]["pred"]),
                  ("v3b", full["v3b"]["pred"])]:
    d = de(eff, h, rows_h)
    out.append({"rung": name, **d})
table(out, ["rung", "fid", "reach", "jac", "nmae", "n_real", "n_ours"])

# ------------------------------------------------------------------ 1. which source line looks like each context
print("\n1. Spearman correlation of control log CPM (universe genes): contexts against source lines")
ctx = gl[["cpm_A", "cpm_B", "cpm_C"]]
for n in LINES:
    c = tabs[n]["ctrl_cpm"].astype(float)
    if not np.isfinite(c).any():
        print(f"   {n:9s} no control CPM stored")
        continue
    m = np.isfinite(c)
    r = [pd.Series(c[m]).corr(pd.Series(ctx[k].to_numpy()[m]), method="spearman") for k in ctx]
    print(f"   {n:9s} A {r[0]:.2f}  B {r[1]:.2f}  C {r[2]:.2f}")

# ------------------------------------------------------------------ 2. single-line transfer (pds proxy, panel targets)
print("\n2. pds proxy when ONE source line predicts the held-out line (rows: held out, columns: source)")
print("   " + " ".join(f"{n:>9s}" for n in LINES))
for hi, name in enumerate(LINES):
    rws = rows_for(hi)
    vals = []
    for si in range(len(LINES)):
        if si == hi:
            vals.append("        -")
            continue
        vals.append(f"{pds(np.nan_to_num(B[si]), hi, rws):9.3f}")
    print(f"{name:9s} " + " ".join(vals))

# ------------------------------------------------------------------ 3. transforms of the v3 effect: pds and DE proxies
print("\n3. transforms of the v3 effect, leave one line out (panel targets; H1 benchmark targets for H1)")
allrows = []
for hi, name in enumerate(LINES):
    tr = [i for i in range(len(LINES)) if i != hi]
    R = rungs(tr)
    rws = rows_for(hi)
    for tname, eff in transforms(R).items():
        d = de(eff, hi, rws)
        allrows.append({"held_out": name, "transform": tname, "pds": pds(eff, hi, rws), **d})
    print(f"   {name} done", flush=True)
df = pd.DataFrame(allrows)
df.to_csv(config.results("proxy_directions.csv"), index=False)
for metric in ["pds", "fid", "reach", "jac", "nmae", "n_ours"]:
    pv = df.pivot(index="transform", columns="held_out", values=metric)[LINES]
    pv["mean"] = pv.mean(1)
    print(f"\n   {metric}" + ("" if metric != "n_ours" else f"   (median real significant genes: "
                                                          f"{df.groupby('held_out').n_real.first().reindex(LINES).to_dict()})"))
    print(pv.sort_values("mean", ascending=metric in ("nmae",)).to_string(float_format=lambda x: f"{x:.3f}"))

# ------------------------------------------------------------------ 4. which source lines help: drop one source at a time
print("\n4. drop one source line (pds proxy / fid proxy of v3); rows: held out, columns: the dropped source")
for hi, name in enumerate(LINES):
    rws = rows_for(hi)
    base_tr = [i for i in range(len(LINES)) if i != hi]
    R = rungs(base_tr)
    line = [f"all {pds(R['v3'], hi, rws):.3f}/{de(R['v3'], hi, rws)['fid']:.3f}"]
    for si in base_tr:
        R2 = rungs([i for i in base_tr if i != si])
        line.append(f"-{LINES[si]} {pds(R2['v3'], hi, rws):.3f}/{de(R2['v3'], hi, rws)['fid']:.3f}")
    print(f"   {name:9s} " + "  ".join(line), flush=True)
