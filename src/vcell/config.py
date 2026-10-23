"""Paths, the panel manifest and the run settings (configs/*.toml).

Paths resolve from the repository root, so everything works from any working directory:
    CONTROLS     the organisers' control folder; VCC_CONTROLS (relative to the repo or absolute), default data/controls
    OUT          cached results: Stage 1 tables, logs, reports
    TABLES       where Stage 1 tables are read (default OUT)
    GENE_LINES   the gene universe and its expression bins (etl.universe)
They are module attributes, read at call time, so `use_snapshot` or `use` switches them for the whole process.

Snapshots name the two states the experiments ran on:
    "def"  D/E/F test panel, 12,816-gene axis: the tables of the final build (F13), in OUT
    "abc"  A/B/C validation panel, 11,961-gene axis: the tables of the leaderboard uploads (L1-L6), kept in OUT/val_tables
"""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / "data" / "sources"          # downloaded public datasets
SUBMISSION = REPO / "data" / "submission"    # generated prediction files and .vcc packages
H1_BENCHMARK = REPO / "data" / "h1_benchmark"
OUT = REPO / "out"                           # cached results (Stage 1 tables, logs); experiments in OUT / "experiments"
CPM_FILTER = 5.0                             # the scorer's expression filter, docs/SCORING.md 2.2


def _repo_path(p) -> Path:
    """An absolute path, a path relative to the repository, or "OUT/..." relative to OUT (so configs survive a move of OUT)."""
    s = str(p).replace("\\", "/")
    if s == "OUT" or s.startswith("OUT/"):
        return OUT / s[4:] if len(s) > 3 else OUT
    p = Path(p)
    return p if p.is_absolute() else REPO / p


CONTROLS = _repo_path(os.environ.get("VCC_CONTROLS", "data/controls"))
TABLES = OUT
GENE_LINES = OUT / "gene_lines.csv"

SNAPSHOTS = {
    "def": {"controls": "data/controls_final", "tables": OUT, "gene_lines": OUT / "gene_lines.csv"},
    "abc": {"controls": "data/controls", "tables": OUT / "val_tables", "gene_lines": OUT / "gene_lines_abc.csv"},
}


def use(controls=None, tables=None, gene_lines=None) -> None:
    """Point the process at another control folder, table folder or gene universe file (None keeps the current one)."""
    global CONTROLS, TABLES, GENE_LINES
    if controls is not None:
        CONTROLS = _repo_path(controls)
    if tables is not None:
        TABLES = _repo_path(tables)
    if gene_lines is not None:
        GENE_LINES = _repo_path(gene_lines)


def use_snapshot(name: str) -> None:
    use(**SNAPSHOTS[name])


def manifest(controls=None) -> dict:
    return json.loads((_repo_path(controls or CONTROLS) / "manifest.json").read_text())


def panel_targets(controls=None) -> list[str]:
    c = _repo_path(controls or CONTROLS)
    return pd.read_csv(c / "pert_counts.csv")[manifest(c)["pert_col"]].tolist()


def gene_names(controls=None) -> list[str]:
    return pd.read_csv(_repo_path(controls or CONTROLS) / "gene_names.csv")["gene_name"].tolist()


def gene_lines(path=None) -> pd.DataFrame:
    """The gene universe table (one row per gene of the organisers' axis): testable_any, max_cpm, cpm_<context>, ..."""
    return pd.read_csv(_repo_path(path or GENE_LINES), index_col=0)


# ------------------------------------------------------------------------------------------------------ run settings
@dataclass(frozen=True)
class ModelConfig:
    """The effect recipe (model.effects). Defaults are the old eda/16_generate_abc.py defaults, not the final model:
    configs/*.toml state every setting explicitly."""
    rung: str = "v3"                                   # zero | v1 | v2 | v3 | v3b (stage 2 ladder)
    transform: str = "none"                            # none | z (standardised effect, model.pool.z_effect)
    scale: float = 1.0                                 # multiplies the effect after the transform
    kappa: float = 1.0                                 # weight of tau^2 in the standardising sd
    min_effect: float = 0.05                           # log2; smaller effects are left out
    bin_mult: tuple[float, ...] | None = None          # multipliers by the gene's control CPM bin in the context (H36)
    neighbour: float | None = None                     # promoter-neighbour cap f (H37)
    sources: tuple[str, ...] = ("K562", "CD4_Rest", "HCT116", "HEK293T", "H1")
    line_weights: dict = field(default_factory=dict)   # line -> weight in the combination (adds or reweights a line)
    context_weights: dict = field(default_factory=dict)  # context -> {line: weight}, overriding the above per context

    def weights(self) -> dict:
        """Base line weights: every source at 1, then line_weights (a weight of 0 drops the line later)."""
        w = {n: 1.0 for n in self.sources}
        w.update({n: float(v) for n, v in self.line_weights.items()})
        return w

    def context_weight_sets(self) -> dict:
        """context -> its full weight set ({**base, **override}); contexts without an override use weights()."""
        base = self.weights()
        return {c: {**base, **{n: float(v) for n, v in w.items()}} for c, w in self.context_weights.items()}


@dataclass(frozen=True)
class RunConfig:
    tag: str
    controls: str = "data/controls"
    tables: str | None = None                          # None = OUT
    gene_lines: str | None = None                      # None = OUT / gene_lines.csv
    seed: int = 0
    model: ModelConfig = ModelConfig()
    note: str = ""

    def activate(self) -> None:
        """Point the process paths at this run's controls, tables and gene universe."""
        use(controls=self.controls, tables=self.tables or OUT, gene_lines=self.gene_lines or OUT / "gene_lines.csv")

    def on(self, snapshot: str) -> "RunConfig":
        """The same run on another snapshot's controls, tables and gene universe (e.g. the E-series on "abc")."""
        s = SNAPSHOTS[snapshot]
        return replace(self, controls=s["controls"], tables=str(s["tables"]), gene_lines=str(s["gene_lines"]))


def load_config(path, **overrides) -> RunConfig:
    """Read configs/<name>.toml into a RunConfig; keyword overrides replace top-level fields (e.g. tag=...)."""
    raw = tomllib.loads(_repo_path(path).read_text(encoding="utf-8"))
    m = dict(raw.pop("model", {}))
    if "bin_mult" in m:
        m["bin_mult"] = tuple(float(x) for x in m["bin_mult"])
    if "sources" in m:
        m["sources"] = tuple(m["sources"])
    for k in ("scale", "kappa", "min_effect", "neighbour"):
        if k in m:
            m[k] = float(m[k])
    cfg = RunConfig(model=ModelConfig(**m), **raw)
    return replace(cfg, **overrides) if overrides else cfg


def results(name: str) -> Path:
    """Where an experiment writes its tables and figures: OUT/experiments/<name>.*"""
    d = OUT / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d / name
