"""H39 (control covariance and the global direction): Does a line's own control-cell covariance reveal the direction of its global perturbation response?

H34 found that target strength x the held-out line's own shared direction would beat the scorer's mse baseline easily, but the
direction did not transfer between lines. Preprint 2026.10.05.756839 calls the leading component of perturbation effects
fitness-related (cell cycle, stress); unperturbed cells vary along such programs too, and every context's control cells are
given. Lines with single cells and ~2,400 essential targets (strong global response): Jurkat, HepG2, RPE1.

In the scorer's pseudobulk space (effect = sqrt(w) * log2FC, w as proxy.wts from the line's control CPM; target genes out):
    g_L       leading right singular vector of line L's effect matrix (uncentred, as the preprint), sign: mean loading > 0
    g_src     mean of the other essential-axis lines' g (K562, CD4_Rest, HCT116, HEK293T and the other two)
    control   control cells of L, per cell log1p(5e4 * counts / UMI) on the axis genes, centred; top PCs V_k, covariance S
Reported: cos(g_src, g_L); share of g_L in span(V_k); cos(PC1, g_L); cos(S g_src, g_L); cos(V_k V_k' g_src, g_L).

    python experiments/analysis/h39_control_covariance.py

Result (EXPERIMENT_LOG H39): supported: the leading component carries 8-17% of the effect energy (RPE1 36%); a line's own
direction lies 72-86% inside the span of its top 10 control PCs (random 0.5%); the covariance transform brings the
cross-line direction closer (cos 0.58 -> 0.67, 0.61 -> 0.74, 0.54 -> 0.67). Scored in failed/h39b_global_direction.py.
Was eda/46_control_covariance.py. Snapshot abc (the A/B/C validation tables). 5 minutes (control PCs cached in OUT/ctrlpcs_<line>.npz).
"""
from __future__ import annotations

import warnings

import h5py
import numpy as np

from vcell import config
from vcell.qc.identity import cd4_ctrl_profile, io
from vcell.etl.stage1.table import load

config.use_snapshot("abc")
warnings.filterwarnings("ignore")
FILES = {"jurkat": config.SOURCES / "nadig" / "GSE264667_jurkat_raw_singlecell_01.h5ad",
         "hepg2": config.SOURCES / "nadig" / "GSE264667_hepg2_raw_singlecell_01.h5ad",
         "rpe1": config.SOURCES / "replogle" / "rpe1_raw_singlecell_01.h5ad"}
LINES = ["K562", "CD4_Rest", "HCT116", "HEK293T", "jurkat", "hepg2", "rpe1"]
MAX_CTRL = 8000
rng = np.random.default_rng(0)
tabs = {n: load(f"ess_{n}") for n in LINES}
genes = list(tabs["K562"]["genes"])
targets = list(tabs["K562"]["targets"])
keep_g = np.array([g not in set(targets) for g in genes])


def sqrt_w(name):
    c = tabs[name]["ctrl_cpm"].astype(float)
    if name.startswith("CD4"):                                   # no control CPM in the DE file; baseMean profile (h36_bin_scale.py)
        c = cd4_ctrl_profile()
    x = np.where(np.isfinite(c), c, np.nan) / 20.0
    return np.log(2) * x / (1 + x)


def leading(name):
    """Leading right singular vector of the line's effect matrix in pseudobulk space (genes the scorer keeps)."""
    E = tabs[name]["beta"].astype(np.float64) * sqrt_w(name)[None, :]
    E = np.nan_to_num(E)[:, keep_g]
    E = E[np.abs(E).sum(1) > 0]
    _, s, vt = np.linalg.svd(E, full_matrices=False)
    v = vt[0]
    v *= np.sign((E @ v).mean())
    share = s[0] ** 2 / (s ** 2).sum()
    return v, share


def cos(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(a[m] @ b[m] / np.linalg.norm(a[m]) / np.linalg.norm(b[m]))


def control_pcs(path, k=50):
    """Top k PCs (and their variances) of the line's non-targeting cells on the axis genes the scorer keeps."""
    with h5py.File(path, "r") as f:
        cats = np.array(io.decode(f["obs"]["__categories"]["gene"][...]), dtype=object)
        tg = cats[f["obs"]["gene"][...]]
        umi = f["obs"]["UMI_count"][...].astype(float)
        vc = np.array(io.decode(f["var"]["__categories"]["gene_name"][...]), dtype=object)
        fg = vc[f["var"]["gene_name"][...]]
        ctrl = np.flatnonzero(np.array(["non-targeting" in t.lower() for t in tg]))
        ctrl = np.sort(rng.choice(ctrl, min(MAX_CTRL, len(ctrl)), replace=False))
        fix = {g: i for i, g in enumerate(fg)}
        col = np.array([fix.get(g, -1) for g in np.array(genes)[keep_g]])
        X = f["X"]
        rows, is_c = [], np.zeros(len(tg), bool)
        is_c[ctrl] = True
        for a in range(0, len(tg), 5000):                         # one sequential pass; controls are scattered
            pick = np.flatnonzero(is_c[a:a + 5000])
            if pick.size:
                rows.append(X[a:a + 5000][pick].astype(np.float32))
        M = np.vstack(rows).astype(np.float64)
    M = np.log1p(5e4 * M / umi[ctrl][:, None])
    Y = np.zeros((M.shape[0], len(col)))
    Y[:, col >= 0] = M[:, col[col >= 0]]
    Y -= Y.mean(0)
    _, s, vt = np.linalg.svd(Y, full_matrices=False)
    var = s ** 2 / (Y.shape[0] - 1)
    return vt[:k], var[:k], Y.shape[0], vt, var


G = {n: leading(n) for n in LINES}
print("leading component: share of effect energy  " + "  ".join(f"{n} {G[n][1]:.2f}" for n in LINES))
print("cosine between lines' leading directions")
for a in LINES:
    print(f"   {a:9s} " + " ".join(f"{cos(G[a][0], G[b][0]):5.2f}" for b in LINES))

for L in ("jurkat", "hepg2", "rpe1"):
    gL = G[L][0]
    src = [n for n in LINES if n != L]
    g_src = np.mean([G[n][0] for n in src], 0)
    V, var, n_cells, Vall, var_all = control_pcs(FILES[L])
    pc1 = V[0] * np.sign(V[0] @ g_src)
    print(f"\n{L}: {n_cells} control cells; cos(g_src, g_L) = {cos(g_src, gL):.3f}")
    print("   share of g_L inside the top-k control PCs: " +
          "  ".join(f"k={k}: {float(((V[:k] @ gL) ** 2).sum()):.3f}" for k in (1, 3, 10, 30, 50)) +
          f"   (random direction, k=50: {50 / keep_g.sum():.4f})")
    print(f"   cos(control PC1, g_L) = {cos(pc1, gL):.3f}")
    Sg = (Vall.T * var_all) @ (Vall @ g_src)                         # covariance times g_src
    print(f"   cos(S g_src, g_L) = {cos(Sg, gL):.3f}")
    for k in (3, 10, 30, 50):
        proj = V[:k].T @ (V[:k] @ g_src)
        print(f"   cos(projection of g_src on top-{k} control PCs, g_L) = {cos(proj, gL):.3f}")
    for a in (0.25, 0.5, 1.0):
        mix = g_src / np.linalg.norm(g_src) + a * Sg / np.linalg.norm(Sg)
        print(f"   cos(g_src + {a} S g_src, g_L) = {cos(mix, gL):.3f}")
