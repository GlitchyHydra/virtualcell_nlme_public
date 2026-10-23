"""R6: the refactored evaluation (Panel, harness, mse proxy) reproduces the old eda/proxy.py, eda/tune_harness.py,
eda/68_proxy_panels.py and eda/17_mse_proxy.py exactly on the current (D/E/F) tables, and the logged F10 values."""
from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import golden
from vcell import config
from vcell.evaluation import cli, mse_proxy
from vcell.evaluation.harness import Harness, final_settings
from vcell.evaluation.panel import Panel

pytestmark = pytest.mark.data
F10 = {"K562": 0.134, "CD4_Rest": 0.120, "HCT116": 0.156, "HEK293T": 0.135, "H1": 0.211}   # EXPERIMENT_LOG F10


@pytest.fixture
def def_snapshot(paths_restored):
    config.use_snapshot("def")


def test_proxy_panel_matches_old_code_and_log(def_snapshot):
    g = golden("proxy_panel")["rows"]
    m = config.load_config("configs/final_def.toml").model
    m = replace(m, line_weights={"Pan_H1": 0.25}, context_weights={})      # eda/68 left KOLF out (still streaming then)
    df = cli.proxy_lines(m)
    for r, want in zip(df.to_dict("records"), g):
        assert r["held out"] == want["held out"] and r["targets"] == want["targets"]
        for k in ("pds raw", "pds part", "DE part", "est"):
            assert r[k] == want[k], (r["held out"], k)
        assert round(r["est"], 3) == F10[r["held out"]]


def test_harness_matches_old_code(def_snapshot):
    g = golden("harness")
    H = Harness()
    assert H.objective(**final_settings()) == g["final"]
    assert H.objective() == g["defaults"]


def test_mse_proxy_matches_old_code(def_snapshot):
    assert mse_proxy.h1_validation(Panel.load()) == golden("mse")["h1_ratio"]
