"""H51c (what the shared patterns are): What the top 3 shared patterns removed by --center-k 3 are made of (A/B/C candidate pool, z x1.5).

For each pattern (uncentred SVD over the 300 panel targets): its share of the effect energy, the genes with the largest
loadings, which source lines carry it (share of the line's own panel-target energy along it, and the mean projection: +
means the same direction as the typical target), and the length of its genes (GENCODE v47).

    python experiments/analysis/h51c_shared_patterns.py

Result (EXPERIMENT_LOG H51c): a gene-length trend (4.9% of the energy; long genes down, Spearman with log length -0.70),
an interferon / p53 stress response (2.6%) and a line-dependent amino-acid stress response (1.7%); not ribosomes.
Was eda/60_shared_patterns.py. Snapshot abc (the A/B/C validation tables). 2 minutes.
"""
from __future__ import annotations

import gzip
import re
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.etl.stage1.table import load
from vcell.model import pool

config.use_snapshot("abc")
warnings.filterwarnings("ignore")
W = {"K562": 1, "CD4_Rest": 1, "HCT116": 1, "HEK293T": 1, "H1": 1, "KOLF": 0.25, "Pan_H1": 0.25}
tabs = {n: load(n) for n in W}
genes = list(tabs["K562"]["genes"])
targets = list(tabs["K562"]["targets"])
B = np.stack([tabs[n]["beta"] for n in W]).astype(float)
S = np.stack([tabs[n]["s"] for n in W]).astype(float)
gl = config.gene_lines().reindex(genes)
fit = pool.predict(B, S, gl.max_cpm.to_numpy(), lam=list(W.values()), ash=False)
P = pool.z_effect(fit["v3"]["pred"], fit["v3"]["sd"], fit["v3"]["K"] >= 1, 1.5)
pi = [targets.index(t) for t in config.panel_targets() if t in targets]
Z = np.nan_to_num(P[pi])
U, Sv, Vt = np.linalg.svd(Z, full_matrices=False)
e = Sv ** 2 / (Sv ** 2).sum()
sign = np.sign(np.median(U[:, :3], axis=0))                       # so the typical target has a positive share
V = Vt[:3] * sign[:, None]
g = np.array(genes)
print(f"energy share of patterns 1-5: {np.round(e[:5], 3)}; top 3 together {e[:3].sum():.3f}")
share3 = ((U[:, :3] * Sv[:3]) ** 2).sum(1) / np.maximum((Z ** 2).sum(1), 1e-12)
print(f"share of a target's own energy in the top 3, quartiles: {np.round(np.quantile(share3, [0.25, 0.5, 0.75]), 3)}")

for k in range(3):
    o = np.argsort(-np.abs(V[k]))[:40]
    same = np.mean(np.sign(U[:, k]) == sign[k])
    print(f"\npattern {k + 1}: {e[k]:.1%} of the energy; {same:.0%} of targets carry it with the typical sign")
    print("  down:", " ".join(g[o][V[k][o] < 0][:20]))
    print("  up:  ", " ".join(g[o][V[k][o] > 0][:20]))

print(f"\nline: share of its panel-target energy along pattern 1 / 2 / 3 (a random direction: {1 / len(genes):.5f})")
for li, n in enumerate(W):
    b = B[li][pi]
    ok = np.isfinite(b).any(1)
    bb = np.nan_to_num(b[ok])
    pr = bb @ V.T
    cells = [f"{(pr[:, k] ** 2).sum() / (bb ** 2).sum():.4f} (mean {pr[:, k].mean():+.2f})" for k in range(3)]
    print(f"  {n:9s} {ok.sum():3d} targets: " + "  ".join(cells))

length = {}
with gzip.open(config.REPO / "data" / "sources" / "gencode" / "gencode.v47.annotation.gtf.gz", "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t", 9)
        if f[2] != "gene":
            continue
        m = re.search(r'gene_name "([^"]+)"', f[8])
        if m:
            length.setdefault(m.group(1), int(f[4]) - int(f[3]))
L = np.array([length.get(x, np.nan) for x in genes], float)
print()
for k in range(3):
    o = np.argsort(-np.abs(V[k]))[:100]
    rho = pd.Series(V[k]).corr(pd.Series(np.log(L)), method="spearman")
    print(f"pattern {k + 1}: median length of its top 100 genes {np.nanmedian(L[o]) / 1e3:.0f} kb (all genes "
          f"{np.nanmedian(L) / 1e3:.0f} kb); Spearman of loading with log length over all genes {rho:+.2f}")
