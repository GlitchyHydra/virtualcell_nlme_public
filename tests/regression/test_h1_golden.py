"""R5: the H1-benchmark prediction with the E4 settings (seed 0) equals the old eda/15_generate_h1.py output bit for bit.
R8 (release): its official score is 0.130 (EXPERIMENT_LOG F11, seed 0)."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import golden, sha
from vcell import config
from vcell.evaluation import h1_benchmark as H


@pytest.mark.data
def test_e4_h1_file_matches_old_code(paths_restored):
    g = golden("h1_e4")["X"]
    cfg = config.load_config("configs/e4_h1.toml")
    cfg.activate()
    a = H.generate(cfg, write=False)
    X = a.X.tocsr()
    assert list(X.shape) == g["shape"] and X.nnz == g["nnz"]
    assert sha(X.data) == g["data"] and sha(X.indices) == g["indices"] and sha(X.indptr.astype(np.int64)) == g["indptr"]
    assert sha(np.array(a.obs["target_gene"].astype(str), dtype="U")) == g["obs_target_gene"]
    assert sha(np.array(a.var_names.astype(str), dtype="U")) == g["var"]


@pytest.mark.release
def test_e4_h1_official_score(paths_restored):
    cfg = config.load_config("configs/e4_h1.toml", tag="e4_release_check")
    cfg.activate()
    pred = H.BM / "pred_e4_release_check.h5ad"
    H.generate(cfg, pred)
    try:
        res = H.score(pred, cfg.tag)
    finally:
        pred.unlink(missing_ok=True)
    # F11, seed 0 (OUT/h1_score_axis12816_e4_s0): the same file, so the same scores
    assert res["overall"] == pytest.approx(0.129881, abs=1e-6)
    assert res["pds"] == pytest.approx(0.696297, abs=1e-6) and res["reach"] == pytest.approx(0.102094, abs=1e-6)
