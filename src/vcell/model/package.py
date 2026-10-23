"""Package a prediction for upload: a local check of the submission rules (docs/SCORING.md section 1), then `vcc prep`.

`vcc prep` must run alone (it commits about 31 GB), with PYTHONIOENCODING=utf-8 (it crashes on cp1252 otherwise) and TEMP/TMP
on D:. Submitting is the user's step: `PYTHONIOENCODING=utf-8 vcc submit data/submission/<file>.vcc`; nothing here submits.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import h5py
import numpy as np

from vcell import config, io

VCC = Path(r"C:\Users\glitc\.local\bin\vcc.exe")
TMP = Path("D:/tmp_vcc")
MAX_CELL_TOTAL = 1_000_000
MAX_STORED = 4_750_000_000


def check(pred, controls=None, block: int = 20_000) -> list[str]:
    """Problems with a prediction file (empty list = passes): rows, gene order, obs labels, integer non-negative finite
    counts, cell totals, stored entries. Streams X in blocks, so it works on the 16 GB files."""
    controls = config._repo_path(controls) if controls else config.CONTROLS
    man = config.manifest(controls)
    genes = config.gene_names(controls)
    panel = set(config.panel_targets(controls))
    problems = []
    with h5py.File(pred, "r") as f:
        obs = io.read_df_group(f["obs"])
        var = io.read_df_group(f["var"])
        X = f["X"]
        n_obs, n_var = (int(v) for v in X.attrs["shape"])
        if list(var.index) != genes:
            problems.append("var is not gene_names.csv in order")
        want = len(man["contexts"]) * len(panel) * man["cells_per_pert"]
        if n_obs != want:
            problems.append(f"{n_obs} rows, expected {want}")
        tg = obs[man["pert_col"]].astype(str)
        if (tg == man["control_label"]).any():
            problems.append("control rows present")
        if not set(tg) <= panel:
            problems.append(f"targets outside the panel: {sorted(set(tg) - panel)[:5]}")
        if set(obs[man["context_col"]].astype(str)) != set(man["contexts"]):
            problems.append("context labels differ from the manifest")
        indptr = X["indptr"][...]
        if int(indptr[-1]) > MAX_STORED:
            problems.append(f"{int(indptr[-1])} stored entries, cap {MAX_STORED}")
        for a in range(0, n_obs, block):
            b = min(a + block, n_obs)
            d = X["data"][int(indptr[a]):int(indptr[b])]
            if not np.isfinite(d).all() or (d < 0).any() or (d != np.round(d)).any():
                problems.append(f"rows {a}-{b}: values not finite, negative or not integer")
                break
            tot = np.add.reduceat(d.astype(np.float64), (indptr[a:b] - indptr[a]).astype(np.int64)) if len(d) else np.zeros(0)
            empty = np.diff(indptr[a:b + 1]) == 0                  # reduceat repeats a value for empty rows
            if (np.where(empty, 0, tot) > MAX_CELL_TOTAL).any():
                problems.append(f"rows {a}-{b}: a cell total above {MAX_CELL_TOTAL}")
                break
    return problems


def prep(pred, out, controls=None, dry_run: bool = False, log=None) -> int:
    """Run `vcc prep` on a prediction file (alone: it needs about 31 GB). Returns the exit code."""
    controls = config._repo_path(controls) if controls else config.CONTROLS
    TMP.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TEMP": str(TMP), "TMP": str(TMP)}
    cmd = [str(VCC), "prep", str(pred), "--perts", str(controls / "pert_counts.csv"), "-g", str(controls / "gene_names.csv"),
           "-o", str(out)] + (["--dry-run"] if dry_run else [])
    print(" ".join(cmd), flush=True)
    with open(log, "w", encoding="utf-8") if log else open(os.devnull, "w") as fh:
        p = subprocess.run(cmd, env=env, stdout=None if not log else fh, stderr=subprocess.STDOUT if log else None)
    return p.returncode
