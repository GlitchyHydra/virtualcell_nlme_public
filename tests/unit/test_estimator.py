"""The Stage 1 estimator (vcell.stats) and the table format (etl.stage1.table)."""
import numpy as np
import scipy.sparse as sp

from vcell import stats
from vcell.etl.stage1 import table


def test_log2fc_and_se_by_hand():
    xp = np.array([10.0, 12.0, 14.0, 16.0])                     # perturbed cells, CPM of one gene
    xc = np.array([20.0, 22.0, 18.0, 20.0, 20.0])               # controls
    b, s = stats.log2fc_and_se(len(xp), xp.sum(), (xp ** 2).sum(), len(xc), xc.sum(), (xc ** 2).sum())
    mp, mc = xp.mean(), xc.mean()
    want_s = np.sqrt(xp.var(ddof=1) / (len(xp) * mp ** 2) + xc.var(ddof=1) / (len(xc) * mc ** 2)) / np.log(2)
    assert np.isclose(b, np.log2(mp / mc)) and np.isclose(s, want_s)


def test_zero_mean_is_missing_not_zero():
    b, s = stats.log2fc_and_se(np.array([3.0, 3.0]), np.array([0.0, 6.0]), np.array([0.0, 14.0]), 4.0, 8.0, 18.0)
    assert np.isnan(b[0]) and np.isnan(s[0]) and np.isfinite(b[1])


def test_cpm_and_group_stats():
    X = sp.csr_matrix(np.array([[1, 3, 0], [2, 2, 4], [0, 5, 5]], float))
    cpm = stats.cpm_matrix(X)
    assert np.allclose(np.asarray(cpm.sum(1)).ravel(), 1e6)
    n, S, SS = stats.group_stats(cpm, np.array([0, 1, 0]), 2)
    dense = cpm.toarray()
    assert list(n) == [2, 1] and np.allclose(S[0], dense[0] + dense[2]) and np.allclose(SS[1], dense[1] ** 2)
    assert np.allclose(stats.context_gene_cpm(X), dense.mean(0))


def test_table_save_rules(tmp_path):
    targets, genes = ["A", "B"], ["A", "C", "D"]
    beta = np.array([[0.5, 0.2, -0.1], [0.3, 0.4, 0.9]])
    s = np.array([[0.1, 0.1, 0.0], [0.1, np.nan, 0.2]])        # s <= 0 and NaN mark missing estimates
    table.save("T", targets, genes, beta, s, n=[10, 20], ctrl_cpm=[1, 2, 3], tables=tmp_path)
    t = table.load("T", tmp_path)
    assert np.isnan(t["beta"][0, 0]) and np.isnan(t["s"][0, 0])     # target A on its own gene A
    assert np.isnan(t["beta"][0, 2]) and np.isnan(t["beta"][1, 1])   # missing together with s
    assert t["beta"].dtype == np.float32 and np.isnan(t["kd"]).all()
    assert t["beta"][1, 0] == np.float32(0.3)


def test_from_accum(tmp_path):
    rng = np.random.default_rng(0)
    genes = ["g0", "g1", "g2"]
    n = np.array([50.0, 60.0, 200.0])                           # targets t0, t1, then the controls
    S = rng.uniform(100, 200, (3, 3)) * n[:, None]
    SS = S ** 2 / n[:, None] * 1.1
    np.savez(tmp_path / "acc.npz", targets=np.array(["t0", "t1"]), genes=np.array(genes), n=n, S=S, SS=SS)
    table.from_accum("x", ["t1", "zz", "t0"], genes, "X", path=tmp_path / "acc.npz", tables=tmp_path)
    t = table.load("X", tmp_path)
    b, _ = stats.log2fc_and_se(n[1], S[1], SS[1], n[2], S[2], SS[2])
    assert np.allclose(t["beta"][0], b, atol=1e-6) and np.isnan(t["beta"][1]).all() and t["n"][2] == 50
