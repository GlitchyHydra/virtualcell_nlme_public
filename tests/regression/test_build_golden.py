"""R3 + R4: the pipeline's train + generate reproduce the old eda/16_generate_abc.py bit for bit (goldens from
tests/golden/capture.py, --limit 3, seed 0), and the first targets of the uploaded files."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import GOLD, golden, h5_csr_parts, h5_obs, sha
from vcell import config
from vcell.model import effects, generate

pytestmark = pytest.mark.data
CONFIGS = {"final_def": "configs/final_def.toml", "final_def_u87x": "configs/final_def_u87x.toml", "l5": "configs/l5.toml"}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_limit3_build_matches_old_code(name, tmp_path, paths_restored):
    g = golden(f"build_{name}")
    cfg = config.load_config(CONFIGS[name])
    fits = {}
    post = [lambda P, **_: fits.setdefault(len(fits), P)]  # records each Stage 2 fit (R3) without changing it
    e = effects.train(cfg, post=post, limit=3)
    arr = np.load(GOLD / f"build_{name}.npz")
    assert len(fits) == len(g["fit_order"])
    for i, key in enumerate(g["fit_order"]):
        assert sha(fits[i]) == g["fits"][key]["sha"], f"Stage 2 fit {key} differs"
        np.testing.assert_array_equal(fits[i][g["fits"][key]["sample_rows"]], arr[f"fit{i}"])
    out = generate.write_prediction(e, tmp_path / "pred.h5ad", seed=cfg.seed)
    parts = h5_csr_parts(out)
    obs = h5_obs(out)
    for k in ("data", "indices", "indptr"):
        assert sha(parts[k]) == g["X"][k], f"X/{k} differs from the old generator"
    assert sha(np.array(obs["target_gene"], dtype="U")) == g["obs_target_gene"]
    assert sha(np.array(obs["context"], dtype="U")) == g["obs_context"]
    assert g["matches_reference_first_rows"]["equal"], "the golden itself did not match the uploaded file"
