# Predicting CRISPRi knockdowns in cells nobody has perturbed

A solo entry to the Arc Institute Virtual Cell Challenge 2026: for a cell line we
have never seen perturbed, predict what 300 gene knockdowns do to its transcriptome, cell by cell, given only its
unperturbed control cells. The answer here is not a neural network but a statistical **effect atlas**: the knockdown
effects measured in public Perturb-seq screens of other cell lines, pooled by a random-effects meta-analysis and written
back into the new line's own control cells.

## Why it matters

A "virtual cell" that predicts the effect of a perturbation in any cell type would let biologists screen gene targets in
silico before running an experiment, which matters most for cell types that are hard to grow, rare, or patient-derived.
Perturb-seq screens cost weeks and tens of thousands of dollars per cell line, and exist for a handful of lines only. The
challenge asks the realistic version of the question: no training data from the new cell type at all, only its controls.

## The task

| | |
|---|---|
| Input | per context (a cell line): 18,400 control cells, 18,533 genes, raw counts |
| Output | 400 cells of raw counts for each of 300 knockdown targets, per context (360,000 cells for three contexts) |
| Validation / final | contexts A, B, C (live leaderboard); D, E, F (MOLM-13, U-87 MG, an hTERT fibroblast; hidden until 2026-11-19) |
| Scoring | six metrics (cell-eval2): profile discrimination among the 300 targets, pseudobulk error, and four differential-expression metrics; each rescaled so 0 = predicting the average response, 1 = repeating the experiment. See `docs/SCORING.md` |

## The idea in five steps

```
public screens of other cell lines (K562, CD4+ T, HCT116, HEK293T, H1, KOLF2.1J iPSC, ...)
   │  1. Stage 1: per line, the log2 fold change of every gene for every knockdown, with its standard error
   ▼
   │  2. Stage 2: random-effects pooling across lines (between-line variance per expression bin)
   │             -> a shared effect and how unsure we are about it in a NEW line
   ▼
   │  3. standardise: effects the lines agree on are kept, inconsistent ones shrink (z x 1.5)
   │  4. context: scale by the gene's expression in the new line, cap promoter neighbours, and give a
   │             same-cell-type screen, when one exists, four times the weight of the others
   ▼
   │  5. generate: resample the new line's own control cells and thin / add counts gene by gene
   ▼
prediction.h5ad  ->  vcc prep  ->  upload
```

The full method with a worked example is `docs/ALGORITHM_WALKTHROUGH.md`.

## What is different here

- **A meta-analysis instead of a deep model.** Each public screen is treated as a "study" of the same knockdown, and the
  new cell line as a study not yet run. The random-effects pool gives an effect and a predictive uncertainty for that
  unseen line, on a desktop PC (no GPU needed, 32 GB of RAM). (A limited literature check found the parts, meta-analysis and
  cross-context transfer, but not this combination for Perturb-seq; `docs/archive/NOVELTY_CHECK.md`.)
- **Shrink by consistency, not by size.** The textbook estimator (an adaptive-shrinkage prior) was more accurate and
  scored far worse: it pushes weak but real effects to zero, and the DE metrics need them. Dividing by the predictive sd
  keeps what the lines agree on. Every method that was optimal for squared error lost on the scorer (H2, H17, H25).
- **A proxy of the scorer itself.** Leave-one-line-out estimates of five of the six metrics in seconds, validated against
  the official scorer, so about 50 ideas could be tested honestly; the index of all of them is `experiments/README.md`.
- **Where the score really is.** Measurements, not opinion: the cell generator is nearly lossless (true effects score
  0.97), the shared effect across cell types has a low ceiling, and one screen of the same cell type is worth more than
  any modelling change (+0.06-0.07 on the official H1 scorer). The pipeline therefore identifies each new context from its
  control cells (on/off marker features and the Human Protein Atlas) and up-weights a same-line source when one exists.
- **Honest cells.** Every predicted cell is an i.i.d. draw from a model of the new line; no pinned pseudobulks and no use
  of the scorer's noise allowance.

## Results

| Step | H1 benchmark (official scorer) | Leaderboard A/B/C |
|---|---|---|
| Unchanged control cells | -0.30 | |
| Generic average response | -0.18 | |
| Pooled effects | 0.076 | 0.092 |
| Standardised x1.5 | 0.121 | 0.160 |
| + expression bins, promoter cap, KOLF and Pan_H1 (final recipe, L5) | | **0.1705** |
| + a same-cell-type source at weight 4 | 0.190 | |

The final D/E/F entry uses the L5 recipe plus a U-87 MG screen for context E. Every number is in `docs/EXPERIMENT_LOG.md`.

## The pipeline

Four stages, each a subpackage of `src/vcell` with its own README:

| Stage | What it does | README |
|---|---|---|
| **qc** | checks the inputs: the organisers' control cells, cell quality of the public screens, and which cell line each context is | [`src/vcell/qc/README.md`](src/vcell/qc/README.md) |
| **etl** | downloads the public screens, defines the gene universe, builds one Stage 1 table (beta, s) per source line | [`src/vcell/etl/README.md`](src/vcell/etl/README.md) |
| **model** | Stage 2 pooling, the effect recipe, cell generation, packaging for upload | [`src/vcell/model/README.md`](src/vcell/model/README.md) |
| **evaluation** | the leave-one-line-out proxies, the 8-line tuning harness, the official H1 scorer | [`src/vcell/evaluation/README.md`](src/vcell/evaluation/README.md) |

Run from the repository root with the uv environment (`.venv\Scripts\python.exe` or `uv run`):

```
python -m vcell qc controls --contexts D E F                 # 1. check the new contexts
python -m vcell qc identity --controls data/controls_final --contexts D E F --hpa
python -m vcell etl stage1 k562 hct116acc hek293tacc h1 cd4 hku kolf u87   # 2. tables for the panel targets
python -m vcell build   --config configs/final_def.toml       # 3. effects + cells: data/submission/pred_final.h5ad
python -m vcell package --config configs/final_def.toml       #    local checks, then vcc prep -> final.vcc
python -m vcell evaluate harness --config configs/final_def.toml   # 4. offline score estimate
```

`configs/` holds one TOML per uploaded entry (`final_def.toml` is the final entry). The D/E/F procedure step by step is
`docs/FINAL_RUNBOOK.md`.

## Repository map

| Path | Contents |
|---|---|
| `src/vcell/` | the pipeline package (`python -m vcell`) |
| `configs/` | run settings of every upload |
| `experiments/` | one script per tested idea: adopted, `analysis/`, `failed/`; index `experiments/README.md` |
| `tests/` | unit tests, golden regression tests (`tests/golden/`) |
| `docs/` | experiment log, algorithm walkthrough, scoring, datasets, runbook, article notes |
| `notebooks/EDA.ipynb` | the first exploration of the data |
| `data/`, `out/` | data and cached results (not in git: hundreds of GB) |

## Tests

```
python -m pytest                  # 48 unit tests on synthetic data, ~5 s
python -m pytest -m data          # 23 golden regression tests on the real files, ~6 min
python -m pytest -m release       # full rebuild compared with the uploaded files, and the official H1 scorer
```

The goldens were captured from the code that built the uploads (git tag `final-def-build`): the pipeline rebuilds the
final submission bit for bit.

## Author and licence

Valerii Kvan, 2026 (solo entry). Cite this work with GitHub's "Cite this repository" button (`CITATION.cff`).

- Code (`src/`, `experiments/`, `tests/`, `configs/`): MIT licence, see `LICENSE`.
- Documentation (`docs/` and the README files): CC BY-NC-ND 4.0, see `docs/LICENSE`: share it unchanged with credit,
  no commercial use, no modified versions.
- Third-party data and papers are not included; `docs/DATASET_INVENTORY.md` lists each source with its own licence.
