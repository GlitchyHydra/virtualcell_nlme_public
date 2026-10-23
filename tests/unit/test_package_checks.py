"""The local submission checks (model.package.check, docs/SCORING.md section 1) catch each rule break."""
import anndata as ad
import pytest

from vcell.model import effects, generate, package


@pytest.fixture
def good(mini):
    cfg, tmp = mini
    return generate.write_prediction(effects.train(cfg), tmp / "good.h5ad"), tmp


def _variant(path, tmp, name, X=None, obs=None, var=None):
    a = ad.read_h5ad(path)
    b = ad.AnnData(X=a.X if X is None else X(a.X.copy()), obs=a.obs if obs is None else obs(a.obs.copy()),
                   var=a.var if var is None else var(a.var.copy()))
    out = tmp / f"{name}.h5ad"
    b.write_h5ad(out)
    return out


def test_good_file_passes(good):
    assert package.check(good[0]) == []


def test_each_rule_is_checked(good):
    path, tmp = good

    def noninteger(X):
        X.data[0] = 1.5
        return X

    def huge(X):
        X = X.tolil()
        X[0, 0] = 2_000_000
        return X.tocsr()

    def control_rows(obs):
        obs["target_gene"] = obs["target_gene"].astype(str)
        obs.iloc[0, obs.columns.get_loc("target_gene")] = "non-targeting"
        return obs

    cases = {"noninteger": dict(X=noninteger), "huge": dict(X=huge), "control": dict(obs=control_rows),
             "genes": dict(var=lambda v: v.iloc[::-1])}
    for name, kw in cases.items():
        assert package.check(_variant(path, tmp, name, **kw)), name


def test_wrong_row_count(good):
    path, tmp = good
    a = ad.read_h5ad(path)[:-20]
    out = tmp / "short.h5ad"
    a.write_h5ad(out)
    assert any("rows" in p for p in package.check(out))
