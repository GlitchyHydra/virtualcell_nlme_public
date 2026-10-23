"""H39b (rejected): add target strength x a covariance-adjusted global direction to the pooled prediction, and score it.
H39 (experiments/analysis/h39_control_covariance.py) found that a line's leading perturbation direction lies mostly in the
span of its own control cells' top PCs, and that the covariance transform S g_src brings the cross-line direction closer to
the line's own. Essential axis of the harness, held out in turn: Jurkat, HepG2, RPE1 (sources: the other six lines):

    glob(t, g) = a_t * d(g) / sqrt(w_L(g))       (pseudobulk space back to log2 in line L)
    a_t        = mean over source lines of the projection of their effect for t on g_src (target strength)
    d          g_src (cross-line mean of leading directions), S g_src (line L's control covariance), or g_L (oracle)

Scored: the mse ratio (coefficients fitted on L = upper bound, or transferred = median of the other two lines), proxy.est at
c = 1.5 with glob added at the transferred ratio, and (C) the full six-metric total over a grid of scales.

Verdict (EXPERIMENT_LOG H39b): no. The covariance direction helps mse (ratio 0.882 -> 0.782 with fitted weights) but the
weight does not transfer and pds falls (0.837 -> 0.72); over the grid the best total is the current setting (c_z 1.5,
c_g 0). With 400 i.i.d. cells per target, mse needs effects ~10x smaller than pds and the DE metrics need: mse stays at 0.
Was eda/47_global_direction.py (control PCs cached in OUT/ctrlpcs_<line>.npz). Essential axis. About 6 minutes.

    python experiments/failed/h39b_global_direction.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config, io
from vcell.evaluation import proxy
from vcell.evaluation.harness import essential_panel
from vcell.evaluation.mse_proxy import fitted_ratio
from vcell.evaluation.panel import standardised
from vcell.qc.identity import cd4_ctrl_profile

warnings.filterwarnings("ignore")
X = essential_panel()
LIN = X.lines
HELD = {"jurkat": 4, "hepg2": 5, "rpe1": 6}
FILES = {"jurkat": config.SOURCES / "nadig" / "GSE264667_jurkat_raw_singlecell_01.h5ad",
         "hepg2": config.SOURCES / "nadig" / "GSE264667_hepg2_raw_singlecell_01.h5ad",
         "rpe1": config.SOURCES / "replogle" / "rpe1_raw_singlecell_01.h5ad"}
nt = X.not_target


def sqrt_w(i):
    c = X.tabs[LIN[i]]["ctrl_cpm"].astype(float)
    if LIN[i].startswith("CD4"):
        c = cd4_ctrl_profile()
    x = c / 20.0
    return np.log(2) * x / (1 + x)                     # NaN where the line reports no control CPM


def delta(i):
    """Effects of line i in pseudobulk space, genes the scorer keeps; NaN -> 0."""
    return np.nan_to_num(X.B[i] * sqrt_w(i)[None, :])[:, nt]


def leading(i):
    E = delta(i)
    E = E[np.abs(E).sum(1) > 0]
    v = np.linalg.svd(E, full_matrices=False)[2][0]
    return v * np.sign((E @ v).mean())


def control_cov(line, k=50):
    """Top-k PCs and variances of the line's control cells (cached), on the genes the scorer keeps."""
    path = config.OUT / f"ctrlpcs_{line}.npz"
    if not path.exists():
        import h5py
        rng = np.random.default_rng(0)
        with h5py.File(FILES[line], "r") as f:
            cats = np.array(io.decode(f["obs"]["__categories"]["gene"][...]), dtype=object)
            tg = cats[f["obs"]["gene"][...]]
            umi = f["obs"]["UMI_count"][...].astype(float)
            vc = np.array(io.decode(f["var"]["__categories"]["gene_name"][...]), dtype=object)
            fg = vc[f["var"]["gene_name"][...]]
            is_c = np.array(["non-targeting" in t.lower() for t in tg])
            sel = np.flatnonzero(is_c)
            sel = np.sort(rng.choice(sel, min(8000, len(sel)), replace=False))
            is_c[:] = False
            is_c[sel] = True
            rows = []
            for a in range(0, len(tg), 5000):
                p = np.flatnonzero(is_c[a:a + 5000])
                if p.size:
                    rows.append(f["X"][a:a + 5000][p].astype(np.float32))
        M = np.log1p(5e4 * np.vstack(rows).astype(np.float64) / umi[sel][:, None])
        fix = {g: i for i, g in enumerate(fg)}
        col = np.array([fix.get(g, -1) for g in np.array(X.genes)[nt]])
        Y = np.zeros((M.shape[0], len(col)))
        Y[:, col >= 0] = M[:, col[col >= 0]]
        Y -= Y.mean(0)
        _, s, vt = np.linalg.svd(Y, full_matrices=False)
        np.savez(path, V=vt[:k].astype(np.float32), var=(s[:k] ** 2 / (Y.shape[0] - 1)), n=Y.shape[0])
    d = np.load(path)
    return d["V"].astype(np.float64), d["var"]


LEAD = {i: leading(i) for i in range(len(LIN))}
VARIANTS = ["z only", "z + strength x g_src", "z + strength x S g_src", "z + strength x ORACLE g_L",
            "z + ORACLE strength x ORACLE g_L"]
out, coefs, store = [], {}, {}
for line, hi in HELD.items():
    tr = [i for i in range(len(LIN)) if i != hi]
    rows = X.rows_for(hi)
    z = standardised(X.rungs(tr)) / 1.0                            # standardised effect at c = 1
    g_src = np.mean([LEAD[i] for i in tr], 0)
    g_src /= np.linalg.norm(g_src)
    V, var = control_cov(line)
    Sg = (V.T * var) @ (V @ g_src)
    Sg /= np.linalg.norm(Sg)
    gL = LEAD[hi]
    a_src = np.nanmean(np.stack([np.where(np.abs(delta(i)).sum(1) > 0, delta(i) @ g_src, np.nan) for i in tr]), 0)
    a_src = np.nan_to_num(a_src)
    a_or = delta(hi) @ gL
    swL = sqrt_w(hi)[nt]
    back = np.where(swL > 1e-3, 1.0 / np.maximum(swL, 1e-3), 0.0)

    def glob(a, d):
        full = np.zeros_like(z)
        full[:, nt] = a[:, None] * (d * back)[None, :]
        return full

    cols = {"z only": [z], "z + strength x g_src": [z, glob(a_src, g_src)],
            "z + strength x S g_src": [z, glob(a_src, Sg)], "z + strength x ORACLE g_L": [z, glob(a_src, gL)],
            "z + ORACLE strength x ORACLE g_L": [z, glob(a_or, gL)]}
    for v in VARIANTS:
        r, c = fitted_ratio(X, cols[v], hi, rows)
        coefs[(line, v)] = c
        store[(line, v)] = (cols[v], rows)
        out.append({"line": line, "variant": v, "mse fit": r, "coef": np.round(c, 3).tolist()})
    print(f"{line}: cos(g_src, g_L) {g_src @ gL:.3f}, cos(S g_src, g_L) {Sg @ gL:.3f}, "
          f"r(strength src, oracle) {np.corrcoef(a_src[rows], a_or[rows])[0, 1]:.3f}", flush=True)

for rec in out:
    line, v = rec["line"], rec["variant"]
    other = np.median(np.stack([coefs[(l, v)] for l in HELD if l != line]), 0)
    cols, rows = store[(line, v)]
    hi = HELD[line]
    rec["mse transfer"] = fitted_ratio(X, cols, hi, rows, other)[0]
    # five-metric proxy at c = 1.5 on z, glob added at the transferred coefficient ratio
    eff = 1.5 * cols[0] + (1.5 * other[1] / other[0] * cols[1] if len(cols) > 1 else 0.0)
    d = X.de(eff, hi, rows, per_target=True)
    rec["est"] = proxy.est(X.pds(eff, hi, rows), d)
    rec["pds"] = X.pds(eff, hi, rows)
R = pd.DataFrame(out)
R.to_csv(config.results("h39b_global_direction.csv"), index=False)

# ------------------------------------------------------------------ C. full six-metric total over a grid of scales
# total = proxy.est (five metrics, mse counted as 0) + mse_scaled / 6, mse_scaled = max(0, (b - raw) / b) with b the line's
# own baseline (its real mean effect given to every target, as the scorer) and the replicate end taken as 0.
print("\nC. six-metric total: eff = c_z * z + c_g * (strength x S g_src); mse scaled against each line's own baseline")
grid = []
for line, hi in HELD.items():
    cols, rows = store[(line, "z + strength x S g_src")]
    base_eff = np.broadcast_to(np.nan_to_num(np.nanmean(X.B[hi][rows], 0)), cols[0].shape)
    b = fitted_ratio(X, [base_eff], hi, rows, np.ones(1))[0]
    for c_z in (0.1, 0.25, 0.5, 1.0, 1.5):
        for c_g in (0.0, 0.25, 0.5, 1.0):
            g_coef = c_g * coefs[(line, "z + strength x S g_src")][1] / max(coefs[(line, "z + strength x S g_src")][0], 1e-9)
            eff = c_z * cols[0] + g_coef * cols[1]
            raw = fitted_ratio(X, [eff], hi, rows, np.ones(1))[0]
            d = X.de(eff, hi, rows, per_target=True)
            e5 = proxy.est(X.pds(eff, hi, rows), d)
            grid.append({"line": line, "c_z": c_z, "c_g": c_g, "mse_raw": raw, "baseline": b,
                         "est5": e5, "total": e5 + max(0.0, (b - raw) / b) / 6})
G = pd.DataFrame(grid)
G.to_csv(config.results("h39b_global_direction_grid.csv"), index=False)
t = G.groupby(["c_z", "c_g"])[["mse_raw", "est5", "total"]].mean().round(4)
print(t.to_string())
print("baselines:", G.groupby("line").baseline.first().round(3).to_dict())
pd.set_option("display.width", 220)
for col in ("mse fit", "mse transfer", "est", "pds"):
    t = R.pivot(index="variant", columns="line", values=col).loc[VARIANTS]
    t["mean"] = t.mean(1)
    print(f"\n{col}")
    print(t.round(4).to_string())
