# etl: from public screens to one effect table per cell line

The model's only training data is other people's Perturb-seq screens. They arrive in five different forms (pseudobulk
means, DESeq2 statistics, single cells in h5ad, Parquet, a 189 GB column-stored h5ad on figshare, a sci-Plex GEO dump), so
this stage turns each of them into one common table: for every knockdown target and every response gene, a log2 fold
change and its standard error, on the same target and gene axis.

```
download.py ──► data/sources/<screen>/                      (resumable, checksum-verified)
universe.py ──► out/gene_lines.csv                          (which genes, and their expression bin)
stage1/     ──► out/stage1_<line>.npz                       (beta, s per target x gene)
coverage.py ──► out/coverage_targets.csv                    (which line measures which panel target)
```

## 1. Downloads (`download.py`)

`python -m vcell etl download small | h1 | xatlas --line HCT116 | more`

One resumable HTTP fetch (Range requests from a partial file) verified against what each server publishes: MD5 from the
Figshare API, MD5 or crc32c from Google Cloud Storage, sha256 from Hugging Face LFS, or the size plus an h5py open. The
sources, their sizes and licences are in `docs/DATASET_INVENTORY.md`, their layouts in `docs/DATASET_FORMATS.md`.

## 2. The gene universe (`universe.py`)

`python -m vcell etl universe` (validation contexts), `--widen --controls data/controls_final --contexts DEF` (test)

The scorer tests only genes above 5 CPM in the controls, so effects on other genes never count. The universe is every gene
above 5 CPM in any context: 11,961 genes for A/B/C, 12,816 after adding D/E/F (F6; D lost 7% of its counts otherwise).
`max_cpm` per gene sets the expression bin of the between-line variance in Stage 2; it is kept fixed for genes already in
the universe so a widening does not change the model on them.

## 3. Stage 1 tables (`stage1/`)

`python -m vcell etl stage1 k562 hct116acc hek293tacc h1 cd4 hku kolf u87 ...`

For a target t and gene g in line l, from the mean CPM of the knockdown cells and of the line's own controls:

```
beta = log2(mean_pert / mean_ctrl)                       ratio of means
s    = sqrt(var_pert / (n_pert mean_pert^2) + var_ctrl / (n_ctrl mean_ctrl^2)) / ln 2      delta method
```

A gene with a zero mean in either group is missing (NaN), never zero; the target's own gene is NaN in its row (the scorer
drops it). Robust sd(beta / s) of 1.04-1.09 in every line shows s is calibrated (H5), so lines can be pooled by precision.

| Builder | Source | How |
|---|---|---|
| `k562.py` | Replogle 2022 K562 genome-wide pseudobulk | 1 CPM pseudocount; s from the spread of non-targeting pseudobulks. Also the dense single-cell builder (RPE1, HepG2, Jurkat) |
| `xatlas.py` | X-Atlas/Orion HCT116, HEK293T (Parquet, 126 GB) | one streamed pass into per-target sums for all ~18,000 targets, checkpointed; `from_accum` then builds a table for any target list in minutes |
| `cd4.py` | Zhu et al. CD4+ T, DESeq2 log_fc and lfcSE | remote row reads from a 17 GB file; a control profile from baseMean |
| `h1.py` | Arc VCC 2025 H1 hESC | one pass over the training file; the H1 benchmark's held-out line |
| `kolf.py` | KOLF2.1J iPSC atlas (189 GB, figshare) | never downloaded: HTTP range requests fetch only the columns of universe genes (76 GB in 3.8 h), resumable |
| `hku.py` | HKU reprocessing: Pan_H1 (another H1 lab), Feng iPSC | their log1p-difference effects converted to log2 fold changes, calibrated per expression bin on H1, which exists in both forms |
| `u87.py` | McFaline-Figueroa sci-Plex CRISPRi kinome screen, U-87 MG (= context E) | hash, guide and cell filters as in the authors' notebooks; vehicle table (F8); all drug arms with the average drug effect divided out, pooled only if the drug-by-guide interaction is within 1.2x the null (U87x, H54) |
| `essential.py` | the essential-gene screens (Jurkat, HepG2, RPE1) and the panel lines on their 2,394 targets | extra held-out lines for the evaluation harness |

Every table is a `.npz` with `targets, genes, beta, s, n, ctrl_cpm, kd` (`stage1/table.py`). Cleaned variants (`--qc`)
use the masks of `vcell.qc.sources`; they are not used (H52).

## Snapshots

Tables are read from `vcell.config.TABLES`. Two states are kept: `def` (the D/E/F panel and the 12,816-gene axis, in
`out/`, used by the final build) and `abc` (the A/B/C panel, in `out/val_tables/`, used by most experiments and the
leaderboard uploads). `config.use_snapshot("abc")` switches.

## Checks

`tests/regression/test_tables.py` rebuilds K562, HCT116, HEK293T, KOLF, U87, U87x, U87d, Pan_H1 and Feng from their
sources or accumulators and requires the result to equal the tables of the final build exactly; `tests/unit/test_estimator.py`
checks the estimator and the table rules on synthetic data.
