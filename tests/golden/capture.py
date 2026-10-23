"""Capture golden outputs from the OLD eda/ code (tag final-def-build), for the regression tests in tests/regression/.

Run once, before the pipeline code changes, one step per process (each loads several GB):

    .venv/Scripts/python.exe tests/golden/capture.py manifest       # R1: sha256 of the pipeline artifacts in eda/out
    .venv/Scripts/python.exe tests/golden/capture.py build final_def      # R3 + R4: 16_generate_abc.py --limit 3, seed 0
    .venv/Scripts/python.exe tests/golden/capture.py build final_def_u87x
    .venv/Scripts/python.exe tests/golden/capture.py build l5             # A/B/C snapshot (eda/out/val_tables)
    .venv/Scripts/python.exe tests/golden/capture.py h1_e4          # R5: 15_generate_h1.py with the E4 settings, seed 0
    .venv/Scripts/python.exe tests/golden/capture.py proxy_panel    # R6: 68_proxy_panels.py values, unrounded
    .venv/Scripts/python.exe tests/golden/capture.py harness        # R6: tune_harness.objective at the final settings
    .venv/Scripts/python.exe tests/golden/capture.py mse            # R6: 17_mse_proxy.py H1 validation

The old scripts run in this process through runpy. Every output goes to a scratch folder, never into eda/out or
data/submission. Results land in tests/golden/<step>.json (+ .npz with small float samples).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EDA = REPO / "eda"
GOLD = Path(__file__).resolve().parent
SCRATCH = REPO / "eda" / "out" / "_golden_scratch"          # git-ignored; deleted after each step

L5_FLAGS = ["--rung", "v3", "--transform", "z", "--scale", "1.5", "--min-effect", "0.02",
            "--bin-mult", "1.5", "1.25", "1", "1", "0.8", "--neighbour", "0.15",
            "--line-weight", "KOLF=0.25", "--line-weight", "Pan_H1=0.25"]
BUILDS = {   # name: (controls folder, ctx weights file or None, reference submission whose first rows must match)
    "final_def": ("data/controls_final", "eda/out/ctx_weights_def.json", "data/submission/pred_final.h5ad"),
    "final_def_u87x": ("data/controls_final", "eda/out/ctx_weights_def_u87x.json", "data/submission/pred_final_u87x.h5ad"),
    "l5": ("data/controls", None, "data/submission/pred_cand.h5ad"),
}
LIMIT = 3


def sha(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def save(step: str, obj: dict, arrays: dict | None = None) -> None:
    (GOLD / f"{step}.json").write_text(json.dumps(obj, indent=1, sort_keys=True))
    if arrays:
        np.savez_compressed(GOLD / f"{step}.npz", **arrays)
    print(f"wrote tests/golden/{step}.json" + (" + .npz" if arrays else ""))


def old_modules(controls: str, out_dir: Path | None = None):
    """Import the old eda/common.py with VCC_CONTROLS set, optionally pointing its OUT at another folder."""
    os.environ["VCC_CONTROLS"] = controls
    sys.path.insert(0, str(EDA))
    import common as C
    if out_dir is not None:
        C.OUT = out_dir
    return C


def h5_csr_parts(path: Path, rows: int | None = None) -> dict:
    """data, indices, indptr of the first `rows` cells (all if None) of a CSR h5ad, indptr rebased to 0."""
    with h5py.File(path, "r") as f:
        X = f["X"]
        n = int(X.attrs["shape"][0]) if rows is None else rows
        ip = X["indptr"][: n + 1].astype(np.int64)
        lo, hi = int(ip[0]), int(ip[-1])
        return {"data": X["data"][lo:hi], "indices": X["indices"][lo:hi], "indptr": ip - lo}


def h5_obs(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        out = {}
        for k in ("target_gene", "context"):
            g = f["obs"][k]
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][...]]
            out[k] = [cats[c] for c in g["codes"][...]]
    return out


# ------------------------------------------------------------------------------------------------------------ steps
def step_manifest() -> None:
    out = REPO / "eda" / "out"
    files = sorted(out.glob("stage1_*.npz")) + [out / n for n in (
        "gene_lines.csv", "gene_lines_abc.csv", "gene_tss_gencode47.csv", "cd4_rest_ctrl_cpm.npy", "ctx_weights.json",
        "ctx_weights_abc.json", "ctx_weights_def.json", "ctx_weights_def_u87x.json")]
    files += sorted((out / "val_tables").glob("stage1_*.npz"))
    man = {p.relative_to(REPO).as_posix(): file_sha(p) for p in files if p.exists()}
    save("manifest", {"files": man})


def step_build(name: str) -> None:
    controls, ctxw, ref = BUILDS[name]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp_repo = Path(tempfile.mkdtemp(dir=SCRATCH))
    out_dir = None
    if name == "l5":                                           # A/B/C snapshot: the tables of the L5 upload
        out_dir = Path(tempfile.mkdtemp(dir=SCRATCH))
        for p in (REPO / "eda" / "out" / "val_tables").glob("stage1_*.npz"):
            shutil.copy2(p, out_dir / p.name)
        shutil.copy2(REPO / "eda" / "out" / "gene_lines_abc.csv", out_dir / "gene_lines.csv")
        shutil.copy2(REPO / "eda" / "out" / "gene_tss_gencode47.csv", out_dir / "gene_tss_gencode47.csv")
    C = old_modules(controls, out_dir)
    C.REPO = tmp_repo                                          # 16 writes to C.REPO / data / submission
    argv = L5_FLAGS + (["--ctx-weights", str(REPO / ctxw)] if ctxw else []) + ["--limit", str(LIMIT), "--seed", "0",
                                                                               "--tag", "golden"]
    sys.argv = ["16_generate_abc.py"] + argv
    buf = io.StringIO()
    with redirect_stdout(buf):
        g = runpy.run_path(str(EDA / "16_generate_abc.py"), run_name="__main__")
    log = buf.getvalue()
    pred = tmp_repo / "data" / "submission" / "pred_golden.h5ad"
    parts = h5_csr_parts(pred)
    obs = h5_obs(pred)
    res = {"argv": argv, "controls": controls, "log": log.splitlines(),
           "X": {k: sha(v) for k, v in parts.items()}, "nnz": int(len(parts["data"])),
           "obs_target_gene": sha(np.array(obs["target_gene"], dtype="U")), "obs_context": sha(np.array(obs["context"], dtype="U")),
           "n_obs": len(obs["target_gene"])}
    fits, arrays = {}, {}
    rng = np.random.default_rng(0)
    for key, P in g["FITS"].items():
        k = ",".join(f"{n}={w}" for n, w in key)
        rows = np.sort(rng.choice(P.shape[0], 12, replace=False))
        fits[k] = {"sha": sha(P), "dtype": str(P.dtype), "shape": list(P.shape), "sample_rows": rows.tolist()}
        arrays[f"fit{len(fits) - 1}"] = P[rows]
    res["fits"] = fits
    res["fit_order"] = list(fits)
    res["uni_targets"] = sha(np.array(g["uni_targets"], dtype="U"))
    res["uni_genes"] = sha(np.array(g["uni_genes"], dtype="U"))
    # first context, first LIMIT targets: identical rng stream to the full run, so they must equal the uploaded file
    n0 = LIMIT * 400
    mine = h5_csr_parts(pred, n0)
    theirs = h5_csr_parts(REPO / ref, n0)
    res["matches_reference_first_rows"] = {
        "reference": ref, "rows": n0,
        "equal": bool(all(np.array_equal(mine[k], theirs[k]) for k in ("data", "indices", "indptr")))}
    res["reference_first_rows"] = {k: sha(v) for k, v in theirs.items()}
    print(f"{name}: {res['n_obs']} cells, nnz {res['nnz']}; first {n0} rows equal {ref}: "
          f"{res['matches_reference_first_rows']['equal']}")
    save(f"build_{name}", res, arrays)
    shutil.rmtree(SCRATCH, ignore_errors=True)


def step_h1_e4() -> None:
    import anndata as ad
    old_modules("data/controls")
    captured = {}

    def fake_write(self, path, *a, **k):                       # hash instead of writing 3.5 GB
        X = self.X.tocsr()
        captured.update({"data": sha(X.data), "indices": sha(X.indices), "indptr": sha(X.indptr.astype(np.int64)),
                         "nnz": int(X.nnz), "shape": list(X.shape),
                         "obs_target_gene": sha(np.array(self.obs["target_gene"].astype(str), dtype="U")),
                         "var": sha(np.array(self.var_names.astype(str), dtype="U"))})
        Path(path).write_bytes(b"")

    ad.AnnData.write_h5ad = fake_write
    argv = ["--rung", "v3", "--transform", "z", "--scale", "1.5", "--seed", "0", "--tag", "golden_e4"]
    sys.argv = ["15_generate_h1.py"] + argv
    buf = io.StringIO()
    with redirect_stdout(buf):
        runpy.run_path(str(EDA / "15_generate_h1.py"), run_name="__main__")
    (REPO / "data" / "h1_benchmark" / "pred_golden_e4.h5ad").unlink(missing_ok=True)
    save("h1_e4", {"argv": argv, "X": captured, "log": buf.getvalue().splitlines()})


def step_proxy_panel() -> None:
    old_modules("data/controls_final")
    sys.argv = ["68_proxy_panels.py"]
    buf = io.StringIO()
    with redirect_stdout(buf):                                 # round() made a no-op, so the values are unrounded
        g = runpy.run_path(str(EDA / "68_proxy_panels.py"), run_name="__main__", init_globals={"round": lambda x, n=None: x})
    rows = [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in g["rows_out"]]
    save("proxy_panel", {"controls": "data/controls_final", "rows": rows, "log": buf.getvalue().splitlines()})


def step_harness() -> None:
    old_modules("data/controls_final")
    import tune_harness as T
    settings = {"c": 1.5, "mult": [1.5, 1.25, 1, 1, 0.8], "neighbour": 0.15, "floor": 0.02, "w_kolf": 0.25, "w_pan": 0.25}
    vals = T.objective(**settings)
    ref = T.objective()
    save("harness", {"controls": "data/controls_final", "settings": settings,
                     "final": {k: float(v) for k, v in vals.items()}, "defaults": {k: float(v) for k, v in ref.items()}})


def step_mse() -> None:
    old_modules("data/controls_final")
    buf = io.StringIO()
    with redirect_stdout(buf):
        g = runpy.run_path(str(EDA / "17_mse_proxy.py"), run_name="__main__")
    h, R, rows, w, B, S = g["h"], g["R"], g["rows"], g["w"], g["B"], g["S"]
    vals = {n: float(g["ratio"](R[n], B[h], S[h], w, rows)) for n in ("v1", "v3")}
    vals["zero"] = float(g["ratio"](np.zeros_like(R["v3"]), B[h], S[h], w, rows))
    save("mse", {"controls": "data/controls_final", "h1_ratio": vals, "log": buf.getvalue().splitlines()})


if __name__ == "__main__":
    step = sys.argv[1]
    rest = sys.argv[2:]
    {"manifest": step_manifest, "build": lambda: step_build(rest[0]), "h1_e4": step_h1_e4,
     "proxy_panel": step_proxy_panel, "harness": step_harness, "mse": step_mse}[step]()
