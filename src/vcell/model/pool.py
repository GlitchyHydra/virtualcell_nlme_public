"""Stage 2 (docs/archive/MODEL_DESIGN.md 3.3 to 3.6; was eda/stage2.py): pool Stage 1 tables across lines and shrink with a sparse prior.

Arrays are (L, T, G): lines, targets, response genes; NaN marks a missing estimate (never zero).

    pool()          DerSimonian-Laird random effects across lines for every (target, gene) pair
    tau2_by_bin()   between-line variance pooled over pairs inside expression bins (stable with few lines)
    generic()       rung v1: the mean effect on each gene over all targets and lines
    ash_fit/post()  sparse prior: point mass at 0 plus zero-centred normals (ashr-like), posterior mean, P(null), lfsr
    predict()       the ladder v1, v2, v3, v3b for a new line, as (prediction, predictive sd) pairs
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

SIGMAS = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])        # log2 units
BIN_EDGES = np.array([5, 10, 50, 200, 1000])                    # expression bins for tau^2, CPM


def gene_bins(cpm: np.ndarray) -> np.ndarray:
    return np.digitize(cpm, BIN_EDGES)


def pool(B: np.ndarray, S: np.ndarray, tau2=None, lam=None) -> dict:
    """tau2=None: per-pair DL estimate. Otherwise an array broadcastable to (T, G) (for example a per-gene vector).
    lam: optional per-line weights (L,) on the random-effects combination only (Q, C and tau2 stay unweighted); theta and
    var are then the weighted mean and its variance, sum a^2 (s^2 + tau^2) / (sum a)^2 with a = lam / (s^2 + tau^2)."""
    ok = np.isfinite(B) & np.isfinite(S) & (S > 0)
    K = ok.sum(0)
    b = np.where(ok, B, 0.0)
    s2 = np.where(ok, S, 1.0) ** 2
    w = np.where(ok, 1.0 / s2, 0.0)
    sw = w.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_fe = (w * b).sum(0) / sw
        Q = (w * (b - theta_fe) ** 2).sum(0)
        Cc = sw - (w ** 2).sum(0) / sw
        tau_dl = np.where(K >= 2, np.maximum(0.0, (Q - (K - 1)) / Cc), 0.0)
        t2 = tau_dl if tau2 is None else np.broadcast_to(tau2, tau_dl.shape)
        w2 = np.where(ok, 1.0 / (s2 + t2[None]), 0.0)
        sw2 = w2.sum(0)
        theta = (w2 * b).sum(0) / sw2
        var = 1.0 / sw2
        if lam is not None:
            a = w2 * np.asarray(lam, float).reshape((-1,) + (1,) * (w2.ndim - 1))
            sa = a.sum(0)
            theta = (a * b).sum(0) / sa
            var = (a ** 2 / np.where(ok, w2, 1.0)).sum(0) / sa ** 2
    return {"K": K, "theta_fe": theta_fe, "var_fe": 1.0 / sw, "Q": Q, "C": Cc, "tau2": np.asarray(t2),
            "theta": theta, "var": var}


def tau2_by_bin(Q, K, Cc, bins: np.ndarray) -> np.ndarray:
    """Method-of-moments tau^2 pooled over every pair with K >= 2 inside each expression bin. Returns a (G,) vector."""
    out = np.zeros(bins.shape[0])
    for b in np.unique(bins):
        m = (K >= 2) & (bins[None, :] == b)
        if m.any():
            out[bins == b] = max(0.0, (Q[m].sum() - (K[m] - 1).sum()) / Cc[m].sum())
    return out


MIN_OBS = 30        # a gene needs this many (target, line) estimates before its own mean and spread are trusted
SD_FLOOR = 0.05     # log2; no gene is declared exactly constant


def generic(B: np.ndarray, S: np.ndarray, min_obs: int = MIN_OBS):
    """Rung v1: the mean effect on each gene over all targets and lines, and its spread.
    A gene seen in fewer than `min_obs` estimates (for example, one target in HCT116 where a silent gene flickered on)
    gets effect 0 and the typical spread of the other genes, not its own one-sample mean with spread 0."""
    ok = np.isfinite(B) & np.isfinite(S)
    flat = np.where(ok, B, np.nan).reshape(-1, B.shape[-1])
    n = np.isfinite(flat).sum(0)
    with np.errstate(invalid="ignore"):
        g = np.nanmean(flat, axis=0)
        sd = np.nanstd(flat, axis=0)
    good = n >= min_obs
    typical = np.nanmedian(sd[good]) if good.any() else 1.0
    g = np.where(good, g, 0.0)
    sd = np.where(good, np.maximum(sd, SD_FLOOR), typical)
    return np.nan_to_num(g), sd


# ---------------------------------------------------------------- sparse prior (adaptive shrinkage)
def _lik(x: np.ndarray, sd: np.ndarray) -> np.ndarray:
    var = sd[:, None] ** 2 + np.r_[0.0, SIGMAS ** 2][None, :]
    return np.exp(-0.5 * x[:, None] ** 2 / var) / np.sqrt(2 * np.pi * var)


def ash_fit(x: np.ndarray, sd: np.ndarray, n_iter: int = 300, max_n: int = 1_000_000, seed: int = 0) -> np.ndarray:
    """Mixture weights (null first) by EM on observed estimates x with known sd."""
    ok = np.isfinite(x) & np.isfinite(sd) & (sd > 0)
    x, sd = x[ok], sd[ok]
    if len(x) > max_n:
        i = np.random.default_rng(seed).choice(len(x), max_n, replace=False)
        x, sd = x[i], sd[i]
    L = _lik(x, sd)
    pi = np.full(L.shape[1], 1.0 / L.shape[1])
    for _ in range(n_iter):
        R = L * pi
        R /= R.sum(1, keepdims=True)
        new = R.mean(0)
        if np.abs(new - pi).max() < 1e-7:
            pi = new
            break
        pi = new
    return pi


def ash_post(x: np.ndarray, sd: np.ndarray, pi: np.ndarray, chunk: int = 400_000) -> dict:
    """Posterior mean, P(null) and local false sign rate for every entry (NaN where x or sd is missing)."""
    shape = x.shape
    x, sd = x.ravel(), sd.ravel()
    mean = np.full(x.shape, np.nan)
    pnull = np.full(x.shape, np.nan)
    lfsr = np.full(x.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(x) & np.isfinite(sd) & (sd > 0))
    s2 = np.r_[0.0, SIGMAS ** 2]
    for a in range(0, len(ok), chunk):
        i = ok[a:a + chunk]
        xi, si = x[i], sd[i]
        W = _lik(xi, si) * pi
        W /= W.sum(1, keepdims=True)
        shrink = s2[None] / (s2[None] + si[:, None] ** 2)
        mk = xi[:, None] * shrink
        sk = np.sqrt(s2[None] * si[:, None] ** 2 / (s2[None] + si[:, None] ** 2))
        mean[i] = (W * mk).sum(1)
        pnull[i] = W[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            ppos = np.where(sk > 0, ndtr(mk / sk), 0.5)                 # P(beta > 0 | component); the null adds to neither side
        pos = (W[:, 1:] * ppos[:, 1:]).sum(1)
        neg = (W[:, 1:] * (1 - ppos[:, 1:])).sum(1)
        lfsr[i] = 1 - np.maximum(pos, neg)
    return {"mean": mean.reshape(shape), "p_null": pnull.reshape(shape), "lfsr": lfsr.reshape(shape)}


def z_effect(pred: np.ndarray, sd: np.ndarray, has: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """The standardised effect pred / sd, multiplied by the median sd so a typical pair keeps its size, then by `scale`.
    Pairs no line measured get the largest sd (so almost no effect). Evidence: experiments/h11_standardised_effect.py (H11)."""
    sd = np.where(has, sd, np.nan)
    ref = np.nanmedian(sd)
    sd = np.where(np.isfinite(sd), sd, np.nanmax(sd))
    return scale * ref * pred / sd


# ---------------------------------------------------------------- the ladder
def predict(B: np.ndarray, S: np.ndarray, gene_cpm: np.ndarray, lam=None, kappa: float = 1.0, ash: bool = True) -> dict:
    """Predict a new line from the lines in B, S. Returns {rung: {"pred", "sd", ...}}, each (T, G).
    Where no line measured the pair, v2 to v3b fall back to v1 (the generic response).
    lam: per-line weights on the v3 combination only (tau^2 unweighted, see pool); kappa: weight of tau^2 in v3's sd;
    ash=False skips v3b (slow)."""
    T, G = B.shape[1:]
    g, gsd = generic(B, S)
    v1 = np.broadcast_to(g, (T, G)).copy()
    v1sd = np.broadcast_to(gsd, (T, G)).copy()
    out = {"v1": {"pred": v1, "sd": v1sd}}
    P0 = pool(B, S, tau2=0.0)
    has = P0["K"] >= 1
    v2 = np.where(has, P0["theta_fe"], v1)
    v2sd = np.where(has, np.sqrt(P0["var_fe"]), v1sd)
    out["v2"] = {"pred": v2, "sd": v2sd, "K": P0["K"]}

    bins = gene_bins(gene_cpm)
    t2 = tau2_by_bin(P0["Q"], P0["K"], P0["C"], bins)
    P = pool(B, S, tau2=t2[None, :], lam=lam)
    v3 = np.where(has, P["theta"], v1)
    v3sd = np.where(has, np.sqrt(P["var"] + kappa * t2[None, :]), v1sd)
    out["v3"] = {"pred": v3, "sd": v3sd, "tau2_bin": t2, "K": P["K"]}
    if not ash:
        return out

    x = np.where(has, P["theta"], np.nan)
    sdp = np.where(has, np.sqrt(P["var"] + t2[None, :]), np.nan)
    pi = ash_fit(x.ravel(), sdp.ravel())
    post = ash_post(x, sdp, pi)
    v3b = np.where(has, post["mean"], v1)
    out["v3b"] = {"pred": v3b, "sd": v3sd, "lfsr": np.where(has, post["lfsr"], 1.0), "p_null": post["p_null"], "pi": pi}
    return out
