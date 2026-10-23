"""H33 (not adopted): generator realism. A per-(cell, gene) gamma(k, 1/k) multiplier on the effect (mean 1) gives the
response cell-to-cell variation; smaller k = more variation. E4 settings on the official H1 scorer, seeds 0 and 1.

Verdict (EXPERIMENT_LOG H33): within noise. k = 2: 0.127 vs 0.121 (seed 0), 0.1312 vs 0.1307 (seed 1): +0.003 on average;
k = 0.5: 0.119. The unchanged model's seed-to-seed spread on H1 is 0.010, so single draws cannot separate changes below
about 0.01. The generator itself is nearly lossless (H42), so realism is not a lever. The emitter below is evaluation.
h1_benchmark.emit_h1 with the multiplier; it consumes extra random numbers, so it is not part of the pipeline.
Was eda/15_generate_h1.py --cell-noise k (eda/out/run_cellnoise.sh). Snapshot abc. About 30 minutes per run.

    python experiments/failed/h33_cell_noise.py [--k 2 0.5] [--seeds 0 1]
"""
import argparse
import json
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from vcell import config
from vcell.evaluation import h1_benchmark as H


def cell_noise_emitter(k: float):
    """emit_h1 with a gamma(k, 1/k) efficiency per (cell, gene) on the knock-down and on the added counts."""
    def emit(M, eff, mu, size_pick, rng):
        f = 2.0 ** eff
        down, up = np.flatnonzero(f < 1), np.flatnonzero(f > 1)
        hit = rng.random(M.shape[0]) >= 0.0
        if len(down):
            M = M.tocoo()
            m = np.isin(M.col, down) & hit[M.row]
            d = M.data.copy()
            p = np.maximum(1 - (1 - f[M.col[m]]) / (1 - 0.0), 0.0)
            p = np.clip(1 - (1 - p) * rng.gamma(k, 1 / k, p.shape), 0, 1)
            d[m] = rng.binomial(d[m], p)
            M = sp.csr_matrix((d, (M.row, M.col)), shape=M.shape)
            M.eliminate_zeros()
        if len(up):
            base = mu[up][None, :] * size_pick[:, None]
            lam = ((f[up] - 1.0) / (1 - 0.0))[None, :] * base * hit[:, None]
            lam = lam * rng.gamma(k, 1 / k, lam.shape)
            add = rng.poisson(lam).astype(np.int32)
            r_, c_ = np.nonzero(add)
            M = (M + sp.csr_matrix((add[r_, c_], (r_, up[c_])), shape=M.shape)).tocsr()
        return M
    return emit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, nargs="+", default=[2.0, 0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--snapshot", default="abc", choices=["abc", "def"])
    a = ap.parse_args()
    base = config.load_config("configs/e4_h1.toml").on(a.snapshot)
    for k in a.k:
        for s in a.seeds:
            cfg = replace(base, tag=f"e4_cn{k:g}_{a.snapshot}_s{s}", seed=s)
            res = H.run(cfg, emitter=cell_noise_emitter(k))
            print(f"k {k:g} seed {s}", json.dumps({kk: round(v, 4) for kk, v in res.items()}), flush=True)
