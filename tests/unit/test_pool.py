"""Stage 2 pooling (model.pool) against the worked examples of docs/archive/MODEL_DESIGN.md 3.3 and 3.4 (was eda/test_stage2.py)."""
import numpy as np
import pytest

from vcell.model import pool


@pytest.fixture
def worked_example():
    s = np.array([0.10, 0.12, 0.15, 0.20, 0.10])
    A = np.array([-0.80, -0.65, -0.20, -0.95, -0.55])            # five lines that disagree
    Bc = np.array([-0.62, -0.58, -0.66, -0.55, -0.60])           # five lines that agree
    B = np.stack([A, Bc], axis=1)[:, :, None]                    # (lines, 2 pairs, 1 gene)
    S = np.broadcast_to(s[:, None, None], B.shape).copy()
    return B, S


def test_pooling_matches_worked_example(worked_example):
    B, S = worked_example
    P = pool.pool(B, S)                                          # per-pair DerSimonian-Laird
    assert np.allclose(P["theta_fe"][:, 0], [-0.628, -0.606], atol=1e-3)
    assert np.allclose(P["Q"][:, 0], [14.33, 0.28], atol=0.02)
    assert np.allclose(P["tau2"][:, 0], [0.0401, 0.0], atol=1e-3)
    assert np.allclose(P["theta"][:, 0], [-0.622, -0.606], atol=2e-3)
    assert np.allclose(np.sqrt(P["var"][:, 0]), [0.107, 0.054], atol=2e-3)
    assert np.allclose(np.sqrt(P["var"] + P["tau2"])[:, 0], [0.227, 0.054], atol=3e-3)


def test_missing_lines_are_skipped(worked_example):
    B, S = worked_example
    B = B.copy()
    B[3:] = np.nan
    P = pool.pool(B, S)
    assert (P["K"] == 3).all() and np.isfinite(P["theta"]).all()


def test_line_weights_move_theta_not_tau2(worked_example):
    B, S = worked_example
    t2 = np.array([[0.04], [0.0]])
    base = pool.pool(B, S, tau2=t2)
    ones = pool.pool(B, S, tau2=t2, lam=np.ones(5))
    assert np.allclose(ones["theta"], base["theta"]) and np.allclose(ones["var"], base["var"])
    heavy = pool.pool(B, S, tau2=t2, lam=[1, 1, 1, 1, 8])        # line 4 (-0.55 in pair A) dominates
    assert heavy["theta"][0, 0] > base["theta"][0, 0]
    assert np.array_equal(heavy["tau2"], base["tau2"]) and np.array_equal(heavy["Q"], base["Q"])


def test_tau2_by_bin_pools_moments():
    rng = np.random.default_rng(1)
    L, T, G = 4, 300, 6
    truth = rng.normal(0, 0.3, (T, G))
    tau = np.array([0.0, 0.0, 0.0, 0.2, 0.2, 0.2])               # bins 0 and 1 differ in between-line spread
    B = truth[None] + rng.normal(0, 1, (L, T, G)) * tau + rng.normal(0, 0.05, (L, T, G))
    S = np.full_like(B, 0.05)
    P0 = pool.pool(B, S, tau2=0.0)
    bins = np.array([0, 0, 0, 1, 1, 1])
    t2 = pool.tau2_by_bin(P0["Q"], P0["K"], P0["C"], bins)
    assert t2[:3].max() < 0.002 and np.allclose(t2[3:], 0.04, rtol=0.15)
    assert np.array_equal(pool.gene_bins(np.array([4.9, 5, 49.9, 50, 1000])), [0, 1, 2, 3, 5])


def test_generic_needs_30_estimates():
    B = np.full((2, 20, 3), np.nan)
    B[:, :, 0] = 0.5 + np.random.default_rng(0).normal(0, 0.2, (2, 20))   # 40 estimates
    B[0, 0, 1] = 3.0                                             # one estimate: a flicker, not a generic response
    S = np.full_like(B, 0.1)
    g, sd = pool.generic(B, S)
    assert g[0] == pytest.approx(np.nanmean(B[:, :, 0])) and g[1] == 0.0 and g[2] == 0.0
    assert sd[0] == pytest.approx(np.nanstd(B[:, :, 0])) and sd[1] == sd[0] == sd[2]   # rare genes: the typical spread


def test_z_effect_keeps_typical_size_and_mutes_unmeasured():
    pred = np.array([[1.0, 1.0, 1.0]])
    sd = np.array([[0.5, 1.0, 2.0]])
    has = np.array([[True, True, False]])
    z = pool.z_effect(pred, sd, has, scale=1.5)
    ref = 0.75                                                   # median sd of the measured pairs
    assert np.allclose(z, [[1.5 * ref * 2.0, 1.5 * ref * 1.0, 1.5 * ref * 1.0]])   # unmeasured: the largest sd (1.0)


def test_sparse_prior_shrinks_noise_keeps_responders():
    rng = np.random.default_rng(0)
    sd = np.full(200_000, 0.1)
    x = rng.normal(0, 0.1, sd.size)
    x[:2000] += rng.choice([-1.0, 1.0], 2000)                    # 1% true responders of size 1
    pi = pool.ash_fit(x, sd)
    post = pool.ash_post(x, sd, pi)
    assert pi[0] > 0.9
    assert np.abs(post["mean"][2000:]).mean() < 0.01
    assert (np.abs(post["mean"][:2000]) > 0.7).mean() > 0.95
    assert post["lfsr"][:2000].max() < 0.05 and np.median(post["lfsr"][2000:]) > 0.3


def test_ladder_runs_end_to_end():
    rng = np.random.default_rng(0)
    B = rng.normal(0, 0.2, (2, 5, 50))
    S = np.full_like(B, 0.1)
    B[:, 0, :5] += 1.0
    out = pool.predict(B, S, np.full(50, 100.0))
    for k in ("v1", "v2", "v3", "v3b"):
        assert out[k]["pred"].shape == (5, 50) and np.isfinite(out[k]["pred"]).all(), k
    assert out["v3b"]["pred"][0, :5].mean() > 0.7
    assert "v3b" not in pool.predict(B, S, np.full(50, 100.0), ash=False)
