"""H11 / H7 / H8 / H14 (the pds proxy): Proxy for pds_cosine (the largest part of the leaderboard score) and a look at what transfers between lines.

For each held-out line: train v3 on the other lines, then for every held-out target t compare the predicted vector with the
measured beta of all held-out targets by cosine (target genes removed, genes weighted by sqrt(w) as in analysis/mse_proxy_validation.py), and
score 1 - rank / (n - 1) for the correct one. Missing held-out estimates count as 0.
Validation: the real scorer gave raw pds 0.774 for v3 and 0.470 for v1 on H1 (0.485 for unchanged controls).

    python experiments/analysis/proxy_pds.py

Result: the standardised effect z beats raw v3 in all five lines (+0.01 to +0.07, H11, adopted); hard |z| cut-offs
(H8), the James-Stein factor (H7) and low-rank denoising (H14: pds 0.50-0.60 vs 0.71) lose, so they were rejected.
The rejected variants are columns of this one table, so they stay here rather than in failed/. The generic
(target-independent) response transfers poorly (line means r 0.0-0.1 for K562 and CD4+ T).
Was eda/18_pds_proxy.py. Snapshot abc (the A/B/C validation tables). 2 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np

from vcell import config
from vcell.etl.stage1.table import benchmark_targets
from vcell.evaluation.panel import Panel

warnings.filterwarnings("ignore")
config.use_snapshot("abc")
X = Panel.load()
B, S, LINES, genes, targets = X.B, X.S, X.lines, X.genes, X.targets
tset = set(targets)
not_target = np.array([g not in tset for g in genes])


def pds(pred, beta, w, rows):
    cols = not_target
    sw = np.sqrt(w)[cols]
    P = np.nan_to_num(pred[rows][:, cols]) * sw
    Y = np.nan_to_num(beta[rows][:, cols]) * sw
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    sim = P @ Y.T
    n = len(rows)
    rank = (sim > np.diag(sim)[:, None]).sum(1) + 0.5 * ((sim == np.diag(sim)[:, None]).sum(1) - 1)
    return float(np.mean(1 - rank / (n - 1)))


def lowrank(M, w, k):
    sw = np.sqrt(w)[None, :]
    U, s, Vt = np.linalg.svd(np.nan_to_num(M) * sw, full_matrices=False)
    return ((U[:, :k] * s[:k]) @ Vt[:k]) / sw


print("pds proxy (1 - rank / (n - 1)), leave one line out; 0.5 means no information")
hdr = ["v3", "v3-generic", "z=th/sd", "hard z>1", "hard z>2", "hard z>3", "JS", "rank1", "rank3", "rank10"]
print(f"{'held out':10s} {'n':>4s} " + " ".join(f"{h:>10s}" for h in hdr))
for hi, name in enumerate(LINES):
    tr = [i for i in range(len(LINES)) if i != hi]
    R = X.rungs(tr)
    rows = np.array([i for i in range(len(targets)) if np.isfinite(B[hi, i]).sum() > 100])
    if name == "H1":
        rows = np.array([i for i in rows if targets[i] in set(benchmark_targets())])
    w = X.wts(hi)
    v3, g = R["v3"], R["g"]
    sd = R["sd"]                                     # sqrt(var + tau^2), unmeasured pairs at the largest sd
    z = v3 / sd
    out = [pds(v3, B[hi], w, rows), pds(v3 - g[None, :], B[hi], w, rows), pds(z, B[hi], w, rows)]
    for z0 in (1, 2, 3):
        out.append(pds(np.where(np.abs(z) > z0, v3, 0.0), B[hi], w, rows))
    out.append(pds(v3 * np.clip(1 - 1 / np.maximum(z ** 2, 1e-9), 0, 1), B[hi], w, rows))
    for k in (1, 3, 10):
        out.append(pds(lowrank(np.where(R["has"], v3, 0.0), w, k), B[hi], w, rows))
    print(f"{name:10s} {len(rows):4d} " + " ".join(f"{o:10.3f}" for o in out), flush=True)

print("\nweighted correlation between the line-mean responses (generic vectors, each line's own mean over its targets)")
means = []
for hi, name in enumerate(LINES):
    rows = np.array([i for i in range(len(targets)) if np.isfinite(B[hi, i]).sum() > 100])
    means.append(np.nanmean(B[hi][rows], axis=0))
M = np.nan_to_num(np.stack(means)) * np.sqrt(X.wts(0))[None, :]
Cm = np.corrcoef(M)
print("          " + " ".join(f"{n:>9s}" for n in LINES))
for i, n in enumerate(LINES):
    print(f"{n:9s} " + " ".join(f"{Cm[i, j]:9.2f}" for j in range(len(LINES))))
