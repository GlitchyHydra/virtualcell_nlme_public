"""Cell emission (model.generate.emit, evaluation.h1_benchmark.emit_h1) on synthetic controls."""
import numpy as np
import pytest
import scipy.sparse as sp

from vcell.evaluation.h1_benchmark import emit_h1
from vcell.model.generate import control_stats, emit


@pytest.fixture
def controls():
    rng = np.random.default_rng(0)
    mu = np.array([0.0, 0.5, 5.0, 20.0, 50.0])
    X0 = sp.csr_matrix(rng.poisson(mu[None, :] * rng.uniform(0.5, 1.5, (4000, 1))).astype(np.int32))
    return X0, *control_stats(X0)


def _run(fn, X0, mu, size_f, eff, seed=1):
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, X0.shape[0], X0.shape[0])
    M = X0[pick].tocsr()
    return M, fn(M, eff, mu, size_f[pick], rng)


def test_zero_effect_leaves_cells_unchanged(controls):
    X0, mu, size_f = controls
    M, out = _run(emit, X0, mu, size_f, np.zeros(5))
    assert (M != out).nnz == 0


def test_thinning_and_additions_hit_the_target_means(controls):
    X0, mu, size_f = controls
    eff = np.array([1.0, -1.0, -1.0, 1.0, 0.0])                 # halve genes 1-2, double genes 0 and 3
    M, out = _run(emit, X0, mu, size_f, eff)
    before, after = np.asarray(M.mean(0)).ravel(), np.asarray(out.mean(0)).ravel()
    assert after[1] == pytest.approx(before[1] / 2, rel=0.08) and after[2] == pytest.approx(before[2] / 2, rel=0.03)
    assert after[3] == pytest.approx(before[3] + mu[3], rel=0.03)   # Poisson additions with mean (f - 1) x mu x size
    assert after[0] == 0.0                                          # nothing to add to a silent gene (mu = 0)
    assert after[4] == before[4]
    D, O = M.toarray(), out.toarray()
    assert (O[:, 1:3] <= D[:, 1:3]).all() and (O[:, 3] >= D[:, 3]).all()   # thinning never adds, additions never remove
    assert out.dtype.kind in "iu" and (out.data > 0).all()


def test_same_seed_same_cells(controls):
    X0, mu, size_f = controls
    eff = np.array([0.0, -0.5, 0.7, 0.3, -0.2])
    _, a = _run(emit, X0, mu, size_f, eff, seed=5)
    _, b = _run(emit, X0, mu, size_f, eff, seed=5)
    assert (a != b).nnz == 0


def test_h1_emitter_agrees_in_distribution(controls):
    X0, mu, size_f = controls
    eff = np.array([0.0, -1.0, -0.5, 0.8, 0.4])
    _, a = _run(emit, X0, mu, size_f, eff)
    _, b = _run(emit_h1, X0, mu, size_f, eff)
    assert np.allclose(np.asarray(a.mean(0)).ravel(), np.asarray(b.mean(0)).ravel(), rtol=0.05, atol=0.02)
