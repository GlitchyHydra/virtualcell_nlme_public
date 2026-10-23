"""Leave-one-line-out proxies of the scored metrics (was eda/proxy.py): the parts that do not depend on the data.

The metrics themselves live on evaluation.panel.Panel (pds, de, ...), because they read a held-out line's real effects.
This module holds the rescaling to the leaderboard scale and the A/B/C-like mix of target strengths.

Validated on H1 against the real scorer (raw values, proxy vs scorer; EXPERIMENT_LOG "Proxy toolkit"):
    v3 fid / jac / reach / nmae 0.492 / 0.092 / 0.144 / 1.051 vs 0.484 / 0.091 / 0.122 / 1.036;
    v1 0.267 / 0.019 / 0.045 / 1.026 vs 0.258 / 0.019 / 0.064 / 1.027; v3b 0.190 / 0.009 / 0.069 / 1.009 vs 0.178 / 0.010 / 0.086 / 1.006.
Only differences between variants are meaningful: the proxy's own mean-response baseline does not reproduce the scorer's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Spans (replicate minus baseline) of each metric on H1, solved from two scored runs (unchanged controls and v3) of vcc-h1:
# from_replicate = (u - b) / (r - b). Baselines there: pds 0.500, fid 0.498, reach 0.034, jac 0.128, nmae 1.003.
SPAN = {"pds": 0.4780, "fid": 0.3692, "reach": 0.9504, "jac": 0.3008, "nmae": -0.6689}
H1_BASE = {"pds": 0.5000, "fid": 0.4983, "reach": 0.0342, "jac": 0.1275, "nmae": 1.0025}

MIX_BINS = [0.5, 10, 40, 150, 500, 1e9]
MIX_LAB = ["1-10", "11-40", "41-150", "151-500", ">500"]
MIX = dict(zip(MIX_LAB, [0.2, 0.3, 0.25, 0.15, 0.1]))     # A/B/C-like share of targets by real DE count


def bh(p, alpha=0.05):
    """Benjamini-Hochberg: True where the adjusted p-value is below alpha."""
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, n + 1) / n
    out = np.zeros(n, bool)
    if ok.any():
        out[o[:np.flatnonzero(ok).max() + 1]] = True
    return out


def mixed(d: dict) -> float:
    """Panel.de(per_target=True) summed over fid, jac, reach, nmae on the leaderboard scale (fid against a coin-flip 0.5),
    averaged over target-strength bins with the A/B/C-like mix."""
    s = pd.Series((d["fid"] - 0.5) / SPAN["fid"] + d["jac"] / SPAN["jac"] + d["reach"] / SPAN["reach"]
                  + (np.nan_to_num(d["nmae"], nan=1.0) - 1.0) / SPAN["nmae"])
    per = s.groupby(pd.cut(pd.Series(d["n_real"]), MIX_BINS, labels=MIX_LAB), observed=False).mean()
    return float(sum(MIX[k] * per[k] for k in MIX_LAB if np.isfinite(per[k])))


def est(pds_val: float, d: dict) -> float:
    """Estimated contribution to the overall: pds and the mixed DE part, each over six metrics (mse counted as 0)."""
    return (pds_val - 0.5) / SPAN["pds"] / 6 + mixed(d) / 6


def score(raw: dict, base: dict) -> dict:
    """Leaderboard-scale value of each metric, (u - b) / span, and the overall (mse counted as 0, as it is clamped)."""
    out = {k: (raw[k] - base[k]) / SPAN[k] for k in SPAN}
    out["nmae"] = max(out["nmae"], -6.0)
    out["overall"] = sum(out[k] for k in SPAN) / 6.0
    return out
