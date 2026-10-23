"""A set of source lines on one axis, for leave-one-line-out evaluation (was the module state of eda/proxy.py).

A Panel stacks the Stage 1 tables of its lines: B, S, NC are (lines, targets, genes) / (lines, targets). Hold line hi out:
fit the others (rungs), build an effect matrix, and score it against line hi's real effects (pds, de). The metric code is the
old eda/proxy.py unchanged; what changed is that the arrays are attributes of an object instead of module globals, so an
experiment changes a copy (with_arrays, add) instead of overwriting another module's state.

    P = Panel.load()                                  # K562, CD4_Rest, HCT116, HEK293T, H1 from the current tables
    P = P.add("KOLF").add("Pan_H1")                   # more source lines (indices 5, 6)
    R = P.rungs([0, 1, 2, 3], lam=None)               # v1, v3 and the predictive sd from lines 0-3
    eff = 1.5 * standardised(R)                       # any effect matrix on the panel's axis
    d = P.de(eff, hi=4, rows=P.rows_for(4), per_target=True); proxy.est(P.pds(eff, 4, P.rows_for(4)), d)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import ndtr

from vcell import config
from vcell.etl.stage1.table import benchmark_targets, load
from vcell.evaluation.proxy import bh
from vcell.model import pool

LINES = ["K562", "CD4_Rest", "HCT116", "HEK293T", "H1"]
MIN_EFFECT = 0.05                       # the generator's default floor; the final model uses 0.02 (set Panel.min_effect)


@dataclass
class Panel:
    lines: list
    tabs: dict
    genes: list
    targets: list
    B: np.ndarray
    S: np.ndarray
    NC: np.ndarray
    max_cpm: np.ndarray
    panel: set                          # targets scored for every held-out line but H1
    bench: set                          # targets scored for H1 (the vcc-h1 benchmark)
    not_target: np.ndarray              # genes that are not a target on the axis (pds compares only those)
    min_effect: float = MIN_EFFECT
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, lines=LINES, tables=None, gene_lines=None, controls=None) -> "Panel":
        tabs = {n: load(n, tables) for n in lines}
        genes = list(tabs[lines[0]]["genes"])
        targets = list(tabs[lines[0]]["targets"])
        for n, t in tabs.items():
            assert list(t["genes"]) == genes and list(t["targets"]) == targets, f"{n} is on a different axis"
        gl = config.gene_lines(gene_lines).reindex(genes)
        tset = set(targets)
        return cls(lines=list(lines), tabs=tabs, genes=genes, targets=targets,
                   B=np.stack([tabs[n]["beta"] for n in lines]).astype(float),
                   S=np.stack([tabs[n]["s"] for n in lines]).astype(float),
                   NC=np.stack([tabs[n]["n"] for n in lines]).astype(float),
                   max_cpm=gl.max_cpm.to_numpy(), panel=set(config.panel_targets(controls)), bench=set(benchmark_targets()),
                   not_target=np.array([g not in tset for g in genes]))

    def add(self, name: str, table: dict | None = None, tables=None) -> "Panel":
        """A copy with one more line appended (a Stage 1 table on the same axis; loaded by name if not given)."""
        t = table if table is not None else load(name, tables)
        assert list(t["genes"]) == self.genes and list(t["targets"]) == self.targets, f"{name} is on a different axis"
        return replace(self, lines=self.lines + [name], tabs={**self.tabs, name: t},
                       B=np.concatenate([self.B, t["beta"][None].astype(float)]),
                       S=np.concatenate([self.S, t["s"][None].astype(float)]),
                       NC=np.concatenate([self.NC, t["n"][None].astype(float)]))

    def with_arrays(self, B=None, S=None) -> "Panel":
        """A copy whose sources use other beta / s arrays (the truth of a held-out line is read from the copy too, so score
        on the original panel when only the sources should change)."""
        return replace(self, B=self.B if B is None else B, S=self.S if S is None else S)

    def index(self, name: str) -> int:
        return self.lines.index(name)

    # ------------------------------------------------------------------------------------------------ fitting
    def rungs(self, tr, weights=None, kappa: float = 1.0, lam=None) -> dict:
        """v1, v3 and the predictive sd from lines `tr`; `weights` multiplies each line's precision (this also biases tau^2
        low: a down-weighted line still counts in K); `lam` weights the lines in the random-effects combination only, after
        an unweighted tau^2 (pool.pool); `kappa` scales tau^2 inside the predictive sd (1 = the uploaded model)."""
        Bt, St = self.B[tr], self.S[tr].copy()
        if weights is not None:
            St = St / np.sqrt(np.asarray(weights, float))[:, None, None]
        g, _ = pool.generic(Bt, St)
        T, G = Bt.shape[1:]
        v1 = np.broadcast_to(g, (T, G)).copy()
        P0 = pool.pool(Bt, St, tau2=0.0)
        t2 = pool.tau2_by_bin(P0["Q"], P0["K"], P0["C"], pool.gene_bins(self.max_cpm))
        P = pool.pool(Bt, St, tau2=t2[None, :], lam=lam)
        has = P0["K"] >= 1
        v3 = np.where(has, P["theta"], v1)
        sd = np.where(has, np.sqrt(P["var"] + kappa * t2[None, :]), np.nan)
        sd = np.where(np.isfinite(sd), sd, np.nanmax(sd))
        return {"g": g, "v1": v1, "v3": v3, "sd": sd, "has": has, "K": P0["K"]}

    # ------------------------------------------------------------------------------------------- held-out line
    def ctrl_cpm(self, hi: int) -> np.ndarray:
        c = self.tabs[self.lines[hi]]["ctrl_cpm"].astype(float)
        return np.where(np.isfinite(c), c, self.max_cpm)

    def wts(self, hi: int) -> np.ndarray:
        """Weight of a log2 effect in the scorer's log1p(50,000-normalised) pseudobulk space, squared."""
        x = self.ctrl_cpm(hi) / 20.0
        return (np.log(2) * x / (1 + x)) ** 2

    def rows_for(self, hi: int) -> np.ndarray:
        """Targets scored for a held-out line: panel targets (H1: the 126 benchmark targets) with a real estimate."""
        pick = self.bench if self.lines[hi] == "H1" else self.panel
        return np.array([i for i in range(len(self.targets)) if self.targets[i] in pick and np.isfinite(self.B[hi, i]).sum() > 100])

    # ------------------------------------------------------------------------------------------------- metrics
    def pds(self, pred, hi, rows) -> float:
        """Cosine rank of each predicted effect among the held-out line's real effects (pds_cosine, SCORING 3.1)."""
        sw = np.sqrt(self.wts(hi))[self.not_target]
        P = np.nan_to_num(pred[rows][:, self.not_target]) * sw
        Y = np.nan_to_num(self.B[hi][rows][:, self.not_target]) * sw
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
        sim = P @ Y.T
        d = np.diag(sim)[:, None]
        rank = (sim > d).sum(1) + 0.5 * ((sim == d).sum(1) - 1)
        return float(np.mean(1 - rank / (len(rows) - 1)))

    def de(self, eff, hi, rows, seed=0, per_target=False, min_effect=None) -> dict:
        """DE proxies. Real side: the Stage 1 estimate brought to 400 cells. Ours: eff (after the generator's floor) plus the
        noise of 400 cells. fid follows cell_eval2/metrics/direction.py: our significant genes whose sign matches the real
        log2FC (significant there or not) / max(n_ours, n_real). per_target=True returns one array per metric (NaN where a
        metric is undefined for that target) plus n_real and n_ours."""
        floor = self.min_effect if min_effect is None else min_effect
        rng = np.random.default_rng(seed)
        testable = self.ctrl_cpm(hi) > 5
        B, S, NC = self.B, self.S, self.NC
        F, J, R, M, nr, npred = [], [], [], [], [], []
        for i in rows:
            m = testable & np.isfinite(B[hi, i]) & np.isfinite(S[hi, i]) & np.isfinite(eff[i])
            b, s = B[hi, i, m], S[hi, i, m]
            n_t = NC[hi, i] if np.isfinite(NC[hi, i]) else 400.0
            s4 = s * np.sqrt(max(n_t, 1.0) / 400.0)
            if n_t > 400:
                b = b + rng.normal(0, np.sqrt(np.maximum(s4 ** 2 - s ** 2, 0)))
            sig_r = bh(2 * ndtr(-np.abs(b / s4)))
            e = eff[i, m].copy()
            e[np.abs(e) < floor] = 0.0
            obs = e + rng.normal(0, s4)
            p_o = 2 * ndtr(-np.abs(obs / s4))
            sig_o = bh(p_o)
            n_r, n_o = sig_r.sum(), sig_o.sum()
            hit = (sig_o & (np.sign(obs) == np.sign(b))).sum()
            F.append(hit / max(n_r, n_o) if max(n_r, n_o) else 1.0)
            u = (sig_r | sig_o).sum()
            J.append((sig_r & sig_o).sum() / u if u else 1.0)
            r_, m_ = np.nan, np.nan
            if n_r:
                idx = np.flatnonzero(sig_r)
                idx = idx[np.lexsort((p_o[idx], ~sig_o[idx]))]
                good = np.cumsum(np.sign(obs[idx]) == np.sign(b[idx])) / np.arange(1, n_r + 1)
                r_ = (np.flatnonzero(good >= 0.9).max() + 1) / n_r if (good >= 0.9).any() else 0.0
                if n_r >= 10:
                    m_ = np.abs(obs[idx] - b[idx]).sum() / np.abs(b[idx]).sum()
            R.append(r_)
            M.append(m_)
            nr.append(n_r)
            npred.append(n_o)
        if per_target:
            return {"fid": np.array(F), "jac": np.array(J), "reach": np.array(R), "nmae": np.array(M),
                    "n_real": np.array(nr), "n_ours": np.array(npred)}
        return {"fid": float(np.mean(F)), "jac": float(np.mean(J)), "reach": float(np.nanmean(R)),
                "nmae": float(np.nanmean(M)), "n_real": float(np.median(nr)), "n_ours": float(np.median(npred))}

    def baseline_eff(self, hi, rows) -> np.ndarray:
        """The scorer's baseline: the mean real effect over the held-out targets, given to every target."""
        eff = np.zeros_like(self.B[hi])
        eff[:] = np.nanmean(self.B[hi][rows], axis=0)
        return np.nan_to_num(eff)

    def evaluate(self, eff, hi, rows, seed=0) -> dict:
        return {"pds": self.pds(eff, hi, rows), **self.de(eff, hi, rows, seed)}


def standardised(R: dict, scale: float = 1.0) -> np.ndarray:
    """The uploaded transform on proxy rungs: scale x median sd x v3 / sd (as model.pool.z_effect; float32 as the harness)."""
    ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
    return (scale * ref * R["v3"] / R["sd"]).astype(np.float32)
