# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Virtual Cell Challenge 2026 (final phase, contexts D/E/F; validation panel `vcc2026-val-1` for A/B/C). `README.md` is the overview. Layout since 2026-10-23 (branch `restructure`; the code that built the uploaded entries is at git tag `final-def-build`, the old `eda/` scripts):

- `src/vcell/`: the pipeline package, four stages: `qc/` (controls, source cell QC, context identity), `etl/` (downloads, gene universe, Stage 1 tables per source line), `model/` (pooling, effect recipe, generation, packaging), `evaluation/` (leave-one-line-out proxies on a `Panel`, the 8-line `harness`, the official H1 scorer). CLI: `python -m vcell <stage> ...` (see `src/vcell/__main__.py`).
- `configs/*.toml`: one run config per uploaded entry (`final_def.toml` = the final entry, `final_def_u87x.toml`, `l1`-`l5`, `e4_h1.toml`). Paths starting `OUT/` are relative to `vcell.config.OUT`.
- `experiments/`: one script per tested idea named after its log ID; `analysis/` diagnostics, `failed/` rejected ideas; index `experiments/README.md`. Experiments call pipeline functions and set a snapshot (`config.use_snapshot("abc"|"def")`).
- `tests/`: `pytest` (unit, synthetic, ~5 s), `pytest -m data` (golden regression against the old code: the uploaded files' first cells, proxy/harness numbers, table rebuilds; ~15 min), `pytest -m release` (full rebuild vs the uploaded files, official H1 scorer; ~1 h). Goldens in `tests/golden/` (capture.py ran the old code). Lint: `python -m pyflakes src tests experiments`.
- `notebooks/EDA.ipynb`: the early EDA notebook (reads cached results in `out/`).

The environment is a uv project (`pyproject.toml`, package `vcell` installed editable, venv in `.venv/`, Python 3.12). Run as `.venv\Scripts\python.exe -m vcell ...` or `.venv\Scripts\python.exe experiments/<file>.py`; add packages with `uv add <pkg>` (`--dev` for tools); `uv sync` rebuilds the venv. uv is installed via `py -3.9 -m pip install --user uv`, so call it as `py -3.9 -m uv ...` if `uv` is not on PATH. The Jupyter kernel is named `virtual-cell`. `cell-eval2` (0.18.0) is installed in the venv. The public H1 stand-in benchmark is a separate uv tool pinned to its own `cell-eval2==0.16.0`: run `C:\Users\glitc\.local\bin\vcc-h1.exe` (`setup`, `score`, `score-control-baseline`), data in `data/h1_benchmark/`. The organisers' submission CLI `vcc` 0.3.0 (required for final submissions; `prep`, `sample`, `submit`, `status`, `datasets`; package `vcc-cli`, installed with `py -3.9 -m uv tool install vcc-cli`) lives at `C:\Users\glitc\.local\bin\vcc.exe`; `vcc login` needs the user's API token, so never run `login` or `submit` without being asked. Do not use `-I` (it hides the script directory, so `experiments/failed/` scripts cannot import their helper modules). Downloaded public datasets go in `data/sources/`, cached results (Stage 1 tables, logs) in `out/`, experiment results in `out/experiments/`, the A/B/C-era tables in `out/val_tables/` (the "abc" snapshot); organiser downloads (zips) in `data/downloads/`.

## Data layout

All data lives under `data/controls/`:

- `context_A.h5ad`, `context_B.h5ad`, `context_C.h5ad` — AnnData (HDF5) files, one per biological context (~210–226 MB each).
- `gene_names.csv` — 18,533 gene symbols (column `gene_name`), matching `n_genes` in the manifest.
- `pert_counts.csv` — 300 perturbation target genes (column `target_gene`), matching `n_constructs`.
- `manifest.json` — schema for the panel. Key fields:
  - `pert_col: "target_gene"` and `context_col: "context"` — the obs column names to use.
  - `control_label: "non-targeting"` — the control perturbation label.
  - Per context: 300 perturbations, 18,400 control cells, 138,400 ground-truth cells, 46 non-targeting IDs.
  - `cells_per_pert: 400`.

Per the organisers' dataset format description (not redistributed here): `X` holds raw counts; `obs` has `target_gene` (always `non-targeting`), `context`, and `ntc_id` (which of the 46 non-targeting guides, 400 cells each); `var` is indexed by gene symbol. The `.h5ad` files themselves have not been opened yet, so confirm this on first load.

## Task and plan

Zero-shot prediction of CRISPRi knockdown responses: 300 perturbations x 400 cells x 3 contexts, raw integer sparse counts, packaged with `vcc prep`. There is no training set; public Perturb-seq data is the training source. Final test inputs (contexts D, E, F) arrive 2026-10-22 and final submissions close 2026-11-05 23:59 UTC.

`docs/FINAL_RUNBOOK.md` is the step-by-step for D/E/F: their files go in `data/controls_final/`; run configs name their controls folder, other commands read `VCC_CONTROLS` (default `data/controls`), so A/B/C are never overwritten. **Start a new session with the "Current state" section of `docs/EXPERIMENT_LOG.md`**: the final model, what worked, what was rejected, and the D/E/F to-do list.

## Where things are written (enforced by hooks in `.claude/settings.json`)

| Content | File |
|---|---|
| Every experiment, run, upload, data check and the current state / to-do | `docs/EXPERIMENT_LOG.md` (H.. hypotheses, E.. H1 scorer runs, L.. leaderboard, F.. final phase) |
| The model, only when a better algorithm is adopted | `docs/ALGORITHM_WALKTHROUGH.md` (step by step, worked example, written for the user) |
| Build and packaging procedure for D/E/F | `docs/FINAL_RUNBOOK.md` |
| Reference, edit only when the facts change | `docs/SCORING.md` (metrics), `docs/DATASET_INVENTORY.md` (sources), `docs/DATASET_FORMATS.md` (file layouts) |
| Project overview (mission, method, results); each stage in depth; the experiment index | `README.md`; `src/vcell/{qc,etl,model,evaluation}/README.md`; `experiments/README.md` |

`docs/archive/` holds superseded documents (old plan, model design, model survey, EDA plan, ideas, novelty check, dataset preparation, the 2026-10-22 handoff). It is read-only: do not edit it or create new docs; put new content in the files above. A PreToolUse hook blocks writes to any other Markdown file, and a Stop hook asks for a log entry when code in `src/` or `experiments/`, or results in `out/`, changed after `docs/EXPERIMENT_LOG.md`.

A new experiment is a new file in `experiments/` (or `analysis/`, `failed/` once its verdict is known) named `<log id>_<slug>.py`, with the header of the others (question, method, verdict, snapshot, runtime), a row in `experiments/README.md` and an entry in the log. Pipeline changes keep `pytest` and `pytest -m data` green. The goldens record what the uploaded code produced; a change meant to alter the model's output replaces them on purpose and says so in the log.

## Constraints

- Never run `vcc login` or `submit`, and never read `~/.config/vcc/state.json`. The user submits: `PYTHONIOENCODING=utf-8 vcc submit data/submission/<file>.vcc`. Final phase: 2 uploads a day (reset 00:00 UTC), the last one counts.
- Always set `PYTHONIOENCODING=utf-8` for `vcc` (it crashes on cp1252 otherwise), and TEMP/TMP on D: for `vcc prep`.
- A low-memory reaper stops jobs: do not restart a stopped job without asking. Run heavy jobs (Stage 1 builds, the generator, `vcc prep`) one at a time.
- Honest i.i.d. cells only: no pinned pseudobulks, no use of the mse noise allowance, no tuning on A/B/C only.
- Hardware: 32 GB RAM, 8 GB GPU; D: had about 145 GB free on 2026-10-22.

## Working conventions

- The manifest is the source of truth for column names and expected counts. Check new code against it rather than hardcoding numbers.
- Files are large. Use `backed="r"` or load a single context at a time rather than reading all three into memory.
- Windows host: the PowerShell tool is the native shell; the Bash tool runs POSIX bash, so use forward slashes and quoted paths there.
