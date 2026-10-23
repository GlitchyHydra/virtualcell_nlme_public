"""train -> generate -> check end to end on the miniature project (tests/unit/conftest.py)."""
from dataclasses import replace

import anndata as ad
import numpy as np

from vcell.model import effects, generate, package


def test_train_shapes_and_context_weights(mini):
    cfg, _ = mini
    e = effects.train(cfg)
    assert e.contexts == ["X", "Y"] and e.targets == ["G1", "G2", "G3", "G4", "G5"]
    assert list(e.has_row) == [True, True, True, True, False]             # G5 has no Stage 1 row
    assert e.eff["X"].shape == (5, 50) and not e.eff["X"][4].any()
    assert not np.array_equal(e.eff["X"], e.eff["Y"])                     # Y pools L3 at weight 4
    v = e.vector("X", 0)
    assert v.shape == (60,) and not v[50:].any()                          # genes outside the universe stay unchanged
    nz = v[v != 0]
    assert (np.abs(nz) >= cfg.model.min_effect).all()


def test_effects_round_trip(mini):
    cfg, tmp = mini
    e = effects.train(cfg)
    f = effects.Effects.load(e.save(tmp / "eff.npz"))
    assert f.targets == e.targets and f.config == e.config
    for c in e.contexts:
        assert np.array_equal(f.eff[c], e.eff[c])


def test_generate_passes_the_submission_checks(mini):
    cfg, tmp = mini
    out = generate.write_prediction(effects.train(cfg), tmp / "pred.h5ad", seed=cfg.seed)
    assert package.check(out) == []
    a = ad.read_h5ad(out)
    assert a.shape == (2 * 5 * 20, 60) and list(a.obs.context.unique()) == ["X", "Y"]
    assert a.X.indptr.dtype == np.int64 and (a.X.data == np.round(a.X.data)).all() and (a.X.data > 0).all()


def test_generation_is_deterministic_and_limit_is_a_prefix(mini):
    cfg, tmp = mini
    full = generate.write_prediction(effects.train(cfg), tmp / "a.h5ad", seed=cfg.seed)
    again = generate.write_prediction(effects.train(cfg), tmp / "b.h5ad", seed=cfg.seed)
    part = generate.write_prediction(effects.train(cfg, limit=2), tmp / "c.h5ad", seed=cfg.seed)
    A, B, C = (ad.read_h5ad(p).X for p in (full, again, part))
    assert (A != B).nnz == 0
    assert (A[:40] != C[:40]).nnz == 0                                    # first context, first 2 targets x 20 cells


def test_zero_rung_resamples_controls(mini):
    cfg, tmp = mini
    cfg0 = replace(cfg, model=replace(cfg.model, rung="zero"))
    e = effects.train(cfg0)
    assert not e.has_row.any()
    assert package.check(generate.write_prediction(e, tmp / "z.h5ad")) == []
