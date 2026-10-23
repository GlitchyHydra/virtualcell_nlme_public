"""The proxy metrics (evaluation.panel, evaluation.proxy) on a synthetic panel whose answers are known."""
import numpy as np
import pytest

from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel


def _panel(T=40, G=400, n=400.0, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 0.05, (T, G))
    truth[:, :40] += rng.choice([-0.8, 0.8], (T, 40)) * (rng.random((T, 40)) < 0.5)   # about 20 real responders per target
    s = np.full((T, G), 0.05)
    B = (truth + rng.normal(0, 0.05, (T, G)))[None]
    targets = [f"t{i}" for i in range(T)]
    tab = {"ctrl_cpm": np.full(G, 100.0)}
    return Panel(lines=["L"], tabs={"L": tab}, genes=[f"g{j}" for j in range(G)], targets=targets, B=B, S=s[None],
                 NC=np.full((1, T), n), max_cpm=np.full(G, 100.0), panel=set(targets), bench=set(),
                 not_target=np.ones(G, bool)), truth


def test_bh_matches_brute_force():
    p = np.random.default_rng(0).uniform(0, 0.1, 50) ** 2
    n = len(p)
    adj = np.minimum.accumulate((np.sort(p) * n / np.arange(1, n + 1))[::-1])[::-1]
    want = np.zeros(n, bool)
    want[np.argsort(p)] = adj <= 0.05
    assert np.array_equal(proxy.bh(p), want)
    assert proxy.bh(np.array([])).size == 0


def test_pds_perfect_and_uninformative():
    P, truth = _panel()
    rows = P.rows_for(0)
    assert len(rows) == 40
    assert P.pds(P.B[0], 0, rows) == pytest.approx(1.0)
    same = np.broadcast_to(truth.mean(0), truth.shape)          # every target gets the same prediction
    assert P.pds(same, 0, rows) == pytest.approx(0.5)


def test_de_rewards_the_truth_and_punishes_silence():
    P, truth = _panel()
    rows = P.rows_for(0)
    good = P.de(truth, 0, rows)
    silent = P.de(np.zeros_like(truth), 0, rows)
    assert good["fid"] > 0.8 and good["reach"] > 0.8 and good["jac"] > silent["jac"] + 0.2
    assert silent["fid"] < 0.1 and silent["n_ours"] == 0 and good["reach"] > silent["reach"] + 0.3
    assert good["nmae"] < 0.5 < silent["nmae"]                    # fold-change error on the real significant genes
    assert P.de(truth, 0, rows, min_effect=10.0)["fid"] < 0.1    # a floor above every effect leaves nothing to call


def test_fid_counts_right_signed_calls_outside_the_real_set():
    """fid = k / max(n_pred, n_real) with k = our calls whose sign matches the real log2FC, real-significant or not."""
    P, truth = _panel(T=12)
    rows = P.rows_for(0)
    loud = truth * 3                                             # calls more genes than the real side, signs right
    d = P.de(loud, 0, rows, per_target=True)
    assert (d["n_ours"] >= d["n_real"]).all() and d["fid"].mean() > 0.5


def test_score_rescaling():
    raw = {"pds": 0.5 + proxy.SPAN["pds"], "fid": proxy.H1_BASE["fid"], "reach": proxy.H1_BASE["reach"],
           "jac": proxy.H1_BASE["jac"], "nmae": 9.0}
    s = proxy.score(raw, proxy.H1_BASE)
    assert s["pds"] == pytest.approx(1.0) and s["fid"] == 0 and s["nmae"] == -6.0
    assert s["overall"] == pytest.approx((1.0 - 6.0) / 6)
    assert sum(proxy.MIX.values()) == pytest.approx(1.0)
