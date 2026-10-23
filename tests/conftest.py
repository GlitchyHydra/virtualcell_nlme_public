"""Shared fixtures. Unit tests use synthetic data only; tests marked `data` need data/ and the Stage 1 tables in OUT and
compare the pipeline with goldens captured from the old eda/ code (tests/golden/capture.py, tag final-def-build)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

GOLD = Path(__file__).resolve().parent / "golden"
sys.path.insert(0, str(GOLD))

from capture import h5_csr_parts, h5_obs, sha  # noqa: E402  (re-exported for the regression tests)

__all__ = ["GOLD", "golden", "h5_csr_parts", "h5_obs", "sha"]


def golden(name: str) -> dict:
    return json.loads((GOLD / f"{name}.json").read_text())


@pytest.fixture
def paths_restored():
    """Restore vcell.config's process paths after a test that switches them."""
    from vcell import config
    saved = (config.CONTROLS, config.TABLES, config.GENE_LINES)
    yield
    config.CONTROLS, config.TABLES, config.GENE_LINES = saved
