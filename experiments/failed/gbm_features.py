"""Pair features for the learners on top of the pool (H25 h25_gbm.py, H27 h27_rank_gbm.py and h27b_rank_effects_upload.py;
was experiments/failed/gbm_features.py), from source lines only plus the new line's control expression. See h25_gbm.py for the list.
Rejected learners, so this lives with them in experiments/failed/, built on the pipeline's Panel.

    GF = Features(panel)
    f, zs = GF.features(source_line_indices, new_line_index)
    M, names = GF.to_matrix(f, rows, cols)
"""
from __future__ import annotations

import numpy as np

from vcell.evaluation.panel import Panel


class Features:
    def __init__(self, X: Panel):
        self.X = X
        self.G = len(X.genes)
        self.KD = np.stack([X.tabs[n]["kd"] for n in X.lines]).astype(float)
        self.CPM = np.stack([X.tabs[n]["ctrl_cpm"] for n in X.lines]).astype(float)   # NaN where a line does not report it
        gi = {g: i for i, g in enumerate(X.genes)}
        self.TGT_COL = np.array([gi.get(t, -1) for t in X.targets])

    @staticmethod
    def gene_neighbours(Z, k=20, chunk=1500):
        """Mean of Z over the k genes whose columns correlate most with each gene (across targets)."""
        A = Z - Z.mean(0, keepdims=True)
        A = (A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-6)).astype(np.float32)
        G = Z.shape[1]
        idx = np.empty((G, k), np.int32)
        for a in range(0, G, chunk):
            Cm = A[:, a:a + chunk].T @ A                                            # (chunk, G)
            Cm[np.arange(Cm.shape[0]), np.arange(a, a + Cm.shape[0])] = -np.inf
            idx[a:a + chunk] = np.argpartition(-Cm, k, axis=1)[:, :k]
        out = np.zeros_like(Z)
        for j in range(k):
            out += Z[:, idx[:, j]]
        return out / k


    @staticmethod
    def target_neighbours(Z, k=10):
        """Mean of Z over the k targets whose rows correlate most with each target."""
        A = Z - Z.mean(0, keepdims=True)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-6
        S = A @ A.T
        np.fill_diagonal(S, -np.inf)
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        return Z[idx].mean(1)


    def features(self, L, ctx, drop=(), ctx_cpm=None):
        """Feature arrays (T, G) from source lines L for a new line ctx (only its control CPM is used), and zs = ref * z.
        ctx_cpm: the new line's control CPM on the universe genes, for a context that is not a source line (A to F)."""
        X, G, CPM, KD, TGT_COL = self.X, self.G, self.CPM, self.KD, self.TGT_COL
        R = X.rungs(L)
        th, sd = R["v3"], R["sd"]
        ref = np.nanmedian(np.where(R["has"], sd, np.nan))
        z = th / sd
        zs = np.nan_to_num(ref * z)
        Bl, Sl = X.B[L], X.S[L]
        with np.errstate(invalid="ignore"):
            f = {"theta": th, "sd": sd, "z": z, "K": R["K"].astype(float), "v1": R["v1"],
                 "l_mean": np.nanmean(Bl, 0), "l_min": np.nanmin(Bl, 0), "l_max": np.nanmax(Bl, 0),
                 "l_sign": np.nanmean(np.sign(Bl), 0), "l_maxz": np.nanmax(np.abs(Bl / Sl), 0)}
            lc = np.log1p(CPM[ctx] if ctx_cpm is None else ctx_cpm)
            lm = np.nanmean(np.log1p(CPM[L]), 0)
            T = th.shape[0]
            f["g_cpm_new"] = np.broadcast_to(lc, (T, G))
            f["g_cpm_src"] = np.broadcast_to(lm, (T, G))
            f["g_cpm_diff"] = np.broadcast_to(lc - lm, (T, G))
            strength = (np.abs(z) > 3).sum(1).astype(float)
            f["t_strength"] = np.broadcast_to(strength[:, None], (T, G))
            tc = np.where(TGT_COL >= 0, lc[np.maximum(TGT_COL, 0)], np.nan)
            f["t_cpm_new"] = np.broadcast_to(tc[:, None], (T, G))
            f["t_kd"] = np.broadcast_to(np.nanmean(KD[L], 0)[:, None], (T, G))
        f["nb_gene"] = self.gene_neighbours(zs)
        f["nb_target"] = self.target_neighbours(zs)
        return {k: v for k, v in f.items() if k not in drop}, zs


    @staticmethod
    def to_matrix(f, rows, cols):
        """(len(rows) * len(cols), n_features) float32, row-major over (target, gene); and the feature names."""
        names = list(f)
        return np.stack([np.asarray(f[k], np.float32)[np.ix_(rows, cols)].ravel() for k in names], 1), names
