"""A miniature project on synthetic data: two contexts of control cells, three Stage 1 source tables and a gene universe,
so the pipeline (train -> generate -> check) runs end to end in seconds without data/."""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from vcell.config import ModelConfig, RunConfig
from vcell.etl.stage1 import table

N_GENES, N_CELLS, CELLS_PER = 60, 300, 20
GENES = [f"G{i}" for i in range(N_GENES)]
PANEL = ["G1", "G2", "G3", "G4", "G5"]                           # targets are genes of the axis, as in the real panel
CONTEXTS = ["X", "Y"]
LINES = ["L1", "L2", "L3"]


def _controls(folder, rng):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps({"contexts": CONTEXTS, "pert_col": "target_gene", "context_col": "context",
                                                      "control_label": "non-targeting", "cells_per_pert": CELLS_PER}))
    pd.DataFrame({"gene_name": GENES}).to_csv(folder / "gene_names.csv", index=False)
    pd.DataFrame({"target_gene": PANEL}).to_csv(folder / "pert_counts.csv", index=False)
    for k, c in enumerate(CONTEXTS):
        mu = rng.gamma(0.6, 8.0, N_GENES) * (1 + k)               # the contexts differ in expression
        X = sp.csr_matrix(rng.poisson(mu[None, :] * rng.uniform(0.5, 1.5, (N_CELLS, 1))).astype(np.float32))
        obs = pd.DataFrame({"target_gene": "non-targeting", "context": c, "ntc_id": [f"ntc{i % 5}" for i in range(N_CELLS)]},
                           index=[f"{c}{i}" for i in range(N_CELLS)])
        ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES)).write_h5ad(folder / f"context_{c}.h5ad")


def _tables(folder, rng):
    uni = GENES[:50]                                             # the universe: genes above 5 CPM somewhere
    targets = PANEL[:4] + ["G40", "Z9"]                          # G5 has no Stage 1 row; two non-panel targets
    truth = rng.normal(0, 0.4, (len(targets), len(uni)))
    for k, n in enumerate(LINES):
        s = np.full(truth.shape, 0.1 + 0.05 * k)
        beta = truth + rng.normal(0, 0.15, truth.shape) + rng.normal(0, s)
        beta[:, k] = np.nan                                      # each line misses one gene
        table.save(n, targets, uni, beta, s, n=np.full(len(targets), 100.0), ctrl_cpm=np.full(len(uni), 50.0), tables=folder)
    gl = pd.DataFrame({"testable_any": [g in uni for g in GENES], "max_cpm": np.linspace(3, 900, N_GENES)},
                      index=pd.Index(GENES, name="gene"))
    gl.to_csv(folder / "gene_lines.csv")


@pytest.fixture
def mini(tmp_path, paths_restored):
    rng = np.random.default_rng(7)
    _controls(tmp_path / "controls", rng)
    _tables(tmp_path / "tables", rng)
    m = ModelConfig(rung="v3", transform="z", scale=1.5, min_effect=0.02, bin_mult=(1.5, 1.25, 1.0, 1.0, 0.8),
                    sources=("L1", "L2"), line_weights={"L3": 0.25}, context_weights={"Y": {"L3": 4.0}})
    cfg = RunConfig(tag="mini", controls=str(tmp_path / "controls"), tables=str(tmp_path / "tables"),
                    gene_lines=str(tmp_path / "tables" / "gene_lines.csv"), seed=3, model=m)
    cfg.activate()
    return cfg, tmp_path
