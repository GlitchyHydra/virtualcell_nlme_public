"""Paths and run configs (vcell.config)."""
from vcell import config


def test_paths_resolve_from_the_repo():
    assert config._repo_path("data/controls") == config.REPO / "data" / "controls"
    assert config._repo_path("OUT/val_tables") == config.OUT / "val_tables"
    assert config._repo_path("OUT") == config.OUT
    assert config._repo_path(config.REPO / "x") == config.REPO / "x"


def test_snapshots(paths_restored):
    config.use_snapshot("abc")
    assert config.CONTROLS.name == "controls" and config.TABLES == config.OUT / "val_tables"
    assert config.GENE_LINES.name == "gene_lines_abc.csv"
    config.use_snapshot("def")
    assert config.CONTROLS.name == "controls_final" and config.TABLES == config.OUT


def test_load_config_overrides_and_activate(paths_restored):
    cfg = config.load_config("configs/l5.toml", tag="x")
    assert cfg.tag == "x" and cfg.model.bin_mult == (1.5, 1.25, 1.0, 1.0, 0.8)
    cfg.activate()
    assert config.TABLES == config.OUT / "val_tables" and config.CONTROLS == config.REPO / "data" / "controls"


def test_every_config_parses():
    for p in sorted((config.REPO / "configs").glob("*.toml")):
        cfg = config.load_config(p)
        assert cfg.tag and cfg.model.rung in ("zero", "v1", "v2", "v3", "v3b"), p.name
