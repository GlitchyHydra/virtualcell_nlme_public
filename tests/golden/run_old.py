"""Run an OLD eda/ script (as at tag final-def-build) against a scratch copy of its inputs, so it never writes into eda/out.

    .venv/Scripts/python.exe tests/golden/run_old.py setup                 # build eda/out/_verify/{def,abc} once
    .venv/Scripts/python.exe tests/golden/run_old.py def 72_u87_as_source.py [args]
    .venv/Scripts/python.exe tests/golden/run_old.py abc 45_combined.py

The script's stdout goes to eda/out/_verify/<snapshot>/logs/<script>.log. Used to check that each ported experiment in
experiments/ prints the same numbers as the script it replaces (tests/golden/compare_logs.py).
"""
from __future__ import annotations

import io
import os
import runpy
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "eda" / "out"
VERIFY = OUT / "_verify"
SNAP = {"def": "data/controls_final", "abc": "data/controls"}


class Tee(io.TextIOBase):
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for s in self.s:
            s.write(x)
            s.flush()
        return len(x)


def setup() -> None:
    skip = {".log", ".err", ".sh", ".ps1", ".png"}
    for snap in SNAP:
        d = VERIFY / snap
        d.mkdir(parents=True, exist_ok=True)
        for p in OUT.iterdir():
            if p.is_file() and p.suffix not in skip and not (d / p.name).exists():
                shutil.copy2(p, d / p.name)
        for sub in ["qc"] + [p.name for p in OUT.glob("h1_score_*") if p.is_dir()]:
            if not (d / sub).exists():
                shutil.copytree(OUT / sub, d / sub)
    abc = VERIFY / "abc"
    for p in (OUT / "val_tables").iterdir():                       # the A/B/C tables over the D/E/F ones
        shutil.copy2(p, abc / p.name)
    shutil.copy2(OUT / "gene_lines_abc.csv", abc / "gene_lines.csv")
    print("scratch ready:", [str(VERIFY / s) for s in SNAP])


def run(snap: str, script: str, args: list[str]) -> None:
    os.environ["VCC_CONTROLS"] = SNAP[snap]
    sys.path.insert(0, str(REPO / "eda"))
    import common as C
    C.OUT = VERIFY / snap
    logs = C.OUT / "logs"
    logs.mkdir(exist_ok=True)
    sys.argv = [script] + args
    tag = "_".join([Path(script).stem] + [a.strip("-") for a in args]) if args else Path(script).stem
    with open(logs / f"{tag}.log", "w", encoding="utf-8") as fh, redirect_stdout(Tee(sys.__stdout__, fh)):
        runpy.run_path(str(REPO / "eda" / script), run_name="__main__")


if __name__ == "__main__":
    if sys.argv[1] == "setup":
        setup()
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3:])
