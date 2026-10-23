"""R7 (release): a full rebuild of the uploaded D/E/F files, compared block by block with them (no 16 GB file is written).
pred_final.h5ad is entry 5wVGrsnP45u0wFCEeQun (F13-F15); pred_final_u87x.h5ad is the variant F18. About 10 minutes each."""
from __future__ import annotations

import numpy as np
import pytest

from vcell import config, io
from vcell.model import effects, generate

pytestmark = pytest.mark.release


@pytest.mark.parametrize("name,ref", [("final_def", "pred_final.h5ad"), ("final_def_u87x", "pred_final_u87x.h5ad")])
def test_full_rebuild_equals_upload(name, ref, paths_restored):
    cfg = config.load_config(f"configs/{name}.toml")
    e = effects.train(cfg)
    path = config.SUBMISSION / ref
    cells = config.manifest()["cells_per_pert"]
    row = 0
    for ctx, k, M in generate.iter_cells(e, seed=cfg.seed):
        R = io.csr_rows(path, row, row + cells)
        assert np.array_equal(M.indptr, R.indptr) and np.array_equal(M.indices, R.indices), (ctx, e.targets[k])
        assert np.array_equal(M.data.astype(np.float32), R.data), (ctx, e.targets[k])
        row += cells
    assert row == 360_000
