"""R1: the pipeline artifacts in OUT are the ones the final build used (sha256 manifest taken at tag final-def-build), and
the Stage 1 tables obey the format. R2: the ETL builders, rebuilt into a temporary folder, reproduce the tables exactly."""
from __future__ import annotations

import hashlib
import shutil

import numpy as np
import pytest

from conftest import golden
from vcell import config
from vcell.etl.stage1 import kolf, table, u87
from vcell.etl.stage1.k562 import k562

pytestmark = pytest.mark.data
FINAL_LINES = ["K562", "CD4_Rest", "HCT116", "HEK293T", "H1", "KOLF", "Pan_H1", "U87", "U87x"]


def _path(rel: str):
    """Manifest paths were taken under eda/out; map them to the current OUT."""
    return config.OUT / rel[len("eda/out/"):] if rel.startswith("eda/out/") else config.REPO / rel


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def test_artifacts_unchanged():
    changed = [rel for rel, want in golden("manifest")["files"].items() if _sha(_path(rel)) != want]
    assert not changed, f"rebuilt or edited since the final build: {changed}"


@pytest.mark.parametrize("name", FINAL_LINES)
def test_table_format(name):
    t = table.load(name, config.OUT)
    ref = table.load("K562", config.OUT)
    assert list(t["genes"]) == list(ref["genes"]) and list(t["targets"]) == list(ref["targets"]), "not on the shared axis"
    s, b = t["s"], t["beta"]
    assert (s[np.isfinite(s)] > 0).all()
    assert (np.isfinite(b) == np.isfinite(s)).all(), "beta and s must be missing together"
    gix = {g: j for j, g in enumerate(t["genes"])}
    own = [(i, gix[x]) for i, x in enumerate(t["targets"]) if x in gix]
    assert all(np.isnan(b[i, j]) for i, j in own), "a target must be NaN on its own gene"


def _same(name, tmp):
    new, old = table.load(name, tmp), table.load(name, config.OUT)
    assert set(new) == set(old)
    for k in old:
        np.testing.assert_array_equal(new[k], old[k], err_msg=f"{name}.{k}")


@pytest.fixture
def def_tmp(tmp_path, paths_restored):
    config.use_snapshot("def")
    config.use(tables=tmp_path)
    return tmp_path


def test_k562_rebuild(def_tmp):
    k562(table.all_targets(), table.universe_genes())
    _same("K562", def_tmp)


@pytest.mark.parametrize("line", ["HCT116", "HEK293T"])
def test_xatlas_from_accumulators(line, def_tmp):
    table.from_accum(line, table.all_targets(), table.universe_genes(), line)
    _same(line, def_tmp)


def test_kolf_from_accumulator(def_tmp):
    kolf.table_from_accum()
    _same("KOLF", def_tmp)


def test_u87_tables(def_tmp):
    table.from_accum("U87", table.all_targets(), table.universe_genes(), "U87", path=u87.D / "U87_accum.npz")
    _same("U87", def_tmp)
    u87.table_drug_adjusted()                                  # reads U87 from the same folder for its comparison
    _same("U87x", def_tmp)
    u87.table_drug_only()
    _same("U87d", def_tmp)


def test_hku_tables(def_tmp, monkeypatch):
    from vcell.etl.stage1 import hku
    for n in ("K562", "H1"):                                   # hku reads the axis and the calibration line from TABLES
        shutil.copy2(table.table_path(n, config.OUT), def_tmp)
    monkeypatch.setattr(config, "OUT", def_tmp)                # it also writes hku_k.npy to OUT
    hku.build()
    monkeypatch.undo()
    _same("Pan_H1", def_tmp)
    _same("Feng_iPSC", def_tmp)
