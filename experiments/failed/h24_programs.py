"""H24 (rejected): Shrinkage through gene programs (docs/archive/IDEAS.md idea 3), the linear way to borrow from correlated genes.

For a held-out line h, the standardised pooled profiles Z (targets x genes, z = theta / sd, so noise is about 1 per entry)
from the training lines. After removing the mean over targets (the generic response, kept aside and added back):
    Z = U S V'  (SVD over the training targets)
    component k has variance lambda_k = S_k^2 / T in noise units; a Wiener factor w_k = max(0, 1 - 1 / lambda_k) keeps the
    part above the noise floor
    Z_hat = sum_{k <= r} w_k * (projection on k)  +  rho * (residual outside the first r components)
rho = 1 keeps every gene's own estimate and only damps noise inside the programs; rho < 1 also shrinks what no program
explains. Scored as 1.5 x ref x Z_hat (the current scale), leave one line out.
Programs are learned from the 575 targets of the training lines; with --xatlas also from all ~18,000 targets of HCT116 and
HEK293T when they are training lines (their accumulators), which is where data without panel overlap enters.

    python experiments/failed/h24_programs.py [--xatlas]

Verdict (EXPERIMENT_LOG H24): rho = 1 leaves the score unchanged (0.145-0.146), shrinking what the programs do not explain
loses 0.005-0.08; programs learned from all X-Atlas targets (--xatlas) change nothing.
Was eda/34_programs.py. Snapshot abc (the A/B/C validation tables). 5 minutes (--xatlas: 30).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from vcell import config, stats
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--xatlas", action="store_true")
args = ap.parse_args()


def xatlas_cov(line, chunk=1000):
    """Gene x gene cross-product of the centred standardised effects of all targets of one X-Atlas line (>= 50 cells),
    accumulated in chunks of targets so only one chunk of effects is expanded at a time (peak about 4.5 GB: the two
    float64 sum arrays plus one chunk). Returns (Z'Z centred, n_targets)."""
    z = np.load(config.SOURCES / "xatlas" / f"{line}_accum.npz", allow_pickle=True)
    n = np.asarray(z["n"])
    nt = len(z["targets"])
    S, SS = z["S"], z["SS"]
    Sc, SSc = np.asarray(S[nt]), np.asarray(SS[nt])
    keep = np.flatnonzero(n[:nt] >= 50)
    G_ = S.shape[1]
    total, ssum = np.zeros((G_, G_), np.float32), np.zeros(G_, np.float64)
    for a in range(0, len(keep), chunk):
        k = keep[a:a + chunk]
        b, s = stats.log2fc_and_se(n[k, None], np.asarray(S[k]), np.asarray(SS[k]), n[nt], Sc, SSc)
        Zk = np.nan_to_num(np.clip(b / s, -20, 20)).astype(np.float32)
        total += Zk.T @ Zk
        ssum += Zk.sum(0)
    m = (ssum / len(keep)).astype(np.float32)
    return total - len(keep) * np.outer(m, m), len(keep)


EXTRA = {}
if args.xatlas:
    for ln in ("HCT116", "HEK293T"):
        EXTRA[ln] = xatlas_cov(ln)
        print(f"{ln}: {EXTRA[ln][1]} extra targets", flush=True)

recs = []
for hi, name in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    rows = X.rows_for(hi)
    R = X.rungs(tr)
    ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
    Z = np.nan_to_num(R["v3"] / R["sd"])
    mu = Z.mean(0, keepdims=True)
    Zc = Z - mu
    Zc32 = Zc.astype(np.float32)
    cov, nfit = Zc32.T @ Zc32, Zc.shape[0]
    for ln in [X.lines[i] for i in tr if X.lines[i] in EXTRA]:  # programs also from all targets of the training X-Atlas lines
        cov += EXTRA[ln][0]
        nfit += EXTRA[ln][1]
    ev, evec = np.linalg.eigh(cov)                              # gene x gene, 12k x 12k float32
    order = np.argsort(ev)[::-1][:300]
    lam = np.clip(ev[order], 0, None) / nfit
    Vt = evec[:, order].T
    del cov, evec
    for r in (10, 30, 100, 300):
        V = Vt[:r]
        w = np.clip(1 - 1 / lam[:r], 0, None)
        P = Zc @ V.T                                            # (T, r) coefficients of the panel profiles
        inside = (P * w) @ V
        resid = Zc - P @ V
        for rho in (1.0, 0.7, 0.4):
            Zh = mu + inside + rho * resid
            eff = 1.5 * ref * Zh
            d = X.de(eff, hi, rows, per_target=True)
            p = X.pds(eff, hi, rows)
            recs.append({"held_out": name, "r": r, "rho": rho, "pds": p, "de_mix": proxy.mixed(d), "est": proxy.est(p, d),
                         "fid": d["fid"].mean(), "jac": d["jac"].mean(), "nmae": np.nanmean(d["nmae"]),
                         "w_mean": float(w.mean())})
    eff = 1.5 * ref * Z
    d = X.de(eff, hi, rows, per_target=True)
    p = X.pds(eff, hi, rows)
    recs.append({"held_out": name, "r": 0, "rho": 1.0, "pds": p, "de_mix": proxy.mixed(d), "est": proxy.est(p, d),
                 "fid": d["fid"].mean(), "jac": d["jac"].mean(), "nmae": np.nanmean(d["nmae"]), "w_mean": np.nan})
    print(f"{name} done (top lambda {np.round(lam[:5], 1)})", flush=True)

df = pd.DataFrame(recs)
tag = "_xatlas" if args.xatlas else ""
df.to_csv(config.results(f"h24_programs_programs{tag}.csv"), index=False)
for col in ["est", "pds", "de_mix", "fid", "jac", "nmae"]:
    pv = df.pivot_table(index=["r", "rho"], columns="held_out", values=col)[X.lines]
    pv["mean"] = pv.mean(1)
    print(f"\n{col} (r = 0: z x1.5 reference)\n" + pv.to_string(float_format=lambda x: f"{x:.3f}"))
