"""A fast proxy for the scorer's expression-error metric, expr_mse_unbiased_capped_norm (was the functions of eda/17).

The scorer compares pseudobulk profiles b = log(1 + 50000 * P / sum(P)); the effect is b_pert - b_ctrl. To first order a log2
fold change beta on a gene with control expression x (per 50,000 counts, = CPM / 20) moves b by ln2 * x / (1 + x) * beta, so

    N = sum_pairs w_g * ((pred - beta)^2 - s^2)        D = sum_pairs w_g * (beta^2 - s^2)        ratio = N / D
    w_g = (ln2 * x_g / (1 + x_g))^2   (Panel.wts)

where beta, s are the held-out line's Stage 1 estimate and its standard error (the s^2 term is the noise correction). Ratio 1
is the same as predicting no effect; the scorer clamps the metric at its baseline (about 0.985 raw on A/B/C), so honest
transfer scores 0 there (H34, H39b). Validation on H1 (A/B/C-era tables): proxy zero 1.00, v1 1.52, v3 2.86; scorer 1.00,
1.85, 3.23 (EXPERIMENT_LOG "Proxy toolkit").
"""
from __future__ import annotations

import numpy as np

from vcell.evaluation.panel import Panel


def parts(pred, beta, s, w, rows):
    """(mask of measured pairs, broadcast weights, D) over the chosen target rows."""
    b, ss = beta[rows], s[rows]
    ok = np.isfinite(b) & np.isfinite(ss) & np.isfinite(pred[rows])
    W = np.broadcast_to(w, b.shape)
    D = (W * (b ** 2 - ss ** 2))[ok].sum()
    return ok, W, D


def ratio(pred, beta, s, w, rows) -> float:
    ok, W, D = parts(pred, beta, s, w, rows)
    p, b, ss = pred[rows], beta[rows], s[rows]
    N = (W * ((p - b) ** 2 - ss ** 2))[ok].sum()
    return N / D


def best_scale(pred, beta, s, w, rows):
    """Optimal c in c * pred (weighted least squares; the noise term does not depend on c) and the ratio it reaches."""
    ok, W, D = parts(pred, beta, s, w, rows)
    p, b = pred[rows], beta[rows]
    cross = (W * p * b)[ok].sum()
    sq = (W * p ** 2)[ok].sum()
    c = cross / sq
    return c, 1.0 - c * cross / D


def h1_rows(P: Panel) -> np.ndarray:
    """The H1 benchmark targets with any H1 estimate (eda/17's row set, wider than Panel.rows_for)."""
    h = P.index("H1")
    return np.array([i for i, t in enumerate(P.targets) if t in P.bench and np.isfinite(P.B[h, i]).any()])


def h1_validation(P: Panel) -> dict:
    """Raw mse ratio on H1 held out for v1, v3 and no effect (the scorer read 1.85, 3.23 and 1.00 on its tables)."""
    h = P.index("H1")
    R = P.rungs([i for i in range(len(P.lines)) if i != h])
    rows, w = h1_rows(P), P.wts(h)
    out = {n: float(ratio(R[n], P.B[h], P.S[h], w, rows)) for n in ("v1", "v3")}
    out["zero"] = float(ratio(np.zeros_like(R["v3"]), P.B[h], P.S[h], w, rows))
    return out


def fitted_ratio(P: Panel, cols, hi: int, rows, coef=None):
    """The noise-corrected mse ratio of a linear combination of effect matrices `cols` for held-out line hi, on the genes
    that are no target: at the weighted least-squares coefficients (an upper bound) or at given ones (what transfers).
    Returns (ratio, coefficients). Used by H34 (experiments/failed/h34_mse_structure.py) and H37."""
    nt = P.not_target
    b, s = P.B[hi][rows][:, nt], P.S[hi][rows][:, nt]
    ok = np.isfinite(b) & np.isfinite(s)
    w = np.broadcast_to(P.wts(hi)[nt], b.shape)[ok]
    Xm = np.stack([np.nan_to_num(c[rows][:, nt])[ok] for c in cols], 1)
    y = b[ok]
    if coef is None:
        A = (Xm * w[:, None]).T @ Xm
        coef = np.linalg.lstsq(A, (Xm * w[:, None]).T @ y, rcond=None)[0]
    D = (w * (y ** 2 - s[ok] ** 2)).sum()
    return float((w * ((Xm @ coef - y) ** 2 - s[ok] ** 2)).sum() / D), coef
