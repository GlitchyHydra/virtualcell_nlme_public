# Dataset formats

Started 2026-10-17. How each file we use is laid out, where it lives, and which code reads it. Which datasets to use and why: `archive/DATASET_PREPARATION.md`. Order of use: `archive/EDA_PLAN.md`. Readers: `vcell.config / vcell.io`, `vcell.etl.stage1`.

Terms. **Cell-level** data: one row per cell. **Pseudobulk**: one row per perturbation, each number the average of that perturbation's cells. **Label**: which target a cell received (never an effect; effects are computed from labelled groups, `archive/MODEL_DESIGN.md` 3.2). **Guide pair**: two guide RNAs aimed at the same gene, delivered together in one construct; a cell carries both.

## 1. Overview

| Dataset | Path | Level | Genes | Label columns | Reader |
|---|---|---|---|---|---|
| Organisers' contexts A, B, C | `data/controls/context_{A,B,C}.h5ad` | cell (controls only) | 18,533 symbols | `target_gene` (always `non-targeting`), `ntc_id`, `context` | `common.load_context` |
| H1 (Arc) train, validation, test | `data/sources/h1/<split>/adata_*.h5ad` | cell | 18,080 symbols | `target_gene`, `guide_id`, `batch` | `stage1.h1`, `stage1.stream_group_stats` |
| X-Atlas/Orion HCT116 (and HEK293T) | `data/sources/xatlas/<line>/<line>_Batch<n>.parquet` | cell, sparse lists | 38,606 tokens (about 18,000 expressed) | `gene_target`, `guide_target`, `pass_guide_filter`, `num_features` | `stage1.xatlas` |
| K562 genome-wide (Replogle) | `data/sources/replogle/K562_gwps_raw_bulk_01.h5ad` | pseudobulk | about 8,250 symbols | row name `gene_transcript` | `common.read_replogle_bulk`, `stage1.k562` |
| CD4+ T (Zhu) DE statistics | remote `s3://genome-scale-tcell-perturb-seq/marson2025_data/GWCD4i.DE_stats.h5ad` | DE result per target and condition | 10,282 symbols | `target_contrast_gene_name`, `culture_condition` | `stage1.cd4` |
| Stage 1 tables | `out/stage1_<line>.npz` | one row per target | 11,961 universe genes | `targets` | `stage1.load` |

## 2. Organisers' contexts (A, B, C)

- AnnData (HDF5), CSR matrix `X` of **raw integer counts**, 18,400 cells by 18,533 genes per context.
- `obs`: `target_gene` (always `non-targeting`), `ntc_id` (which of 46 non-targeting guide pairs, 400 cells each), `context`. `var` is indexed by gene symbol (stored as a nullable string group, `values` and `mask`).
- Only control cells exist. The perturbed cells are what we predict (300 targets by 400 cells per context).
- Expected submission: raw counts, 400 cells per target, the context's gene axis, `target_gene` label (`SCORING.md`, `vcc prep`).

## 3. H1 (Arc, 2025 challenge)

- Public bucket `gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/`, fetched by `python -m vcell etl download h1`. Big files have only a crc32c checksum.
- Three h5ad files (CSR `X`, float32 holding integer raw counts, 18,080 genes):

| Split | Cells | Perturbed cells | Targets |
|---|---|---|---|
| train | 221,273 | 183,097 | 150 |
| validation | 98,927 | 60,751 | 50 |
| test | 170,846 | 132,670 | 100 |

- Every file repeats the **same 38,176 non-targeting cells** (identical barcodes). Count them once when combining splits: 376,518 perturbed plus 38,176 control cells.
- `obs`: `target_gene` (`non-targeting` for controls), `guide_id` (for example `CHMP3_P1P2_A|CHMP3_P1P2_B`; designs P1P2 about 160,700 perturbed cells, P1 11,200, P2 8,400; 31 control guide pairs), `batch` (48 10x Flex batches; controls are 16 to 18% of every batch). `var` index holds the symbols.
- `pert_counts_<Split>.csv`: `target_gene, n_cells, median_umi_per_cell`. `gene_names.csv` has **no header row**, 18,080 symbols.
- Median depth about 54,000 UMI per cell, more than twice the contest's 20,000.
- Benchmark side: `data/h1_benchmark/` (the public `vcc-h1` tool): 126 training targets with at least 400 cells, a fixed 400-cell reference sample per target in `benchmark/reference_cells.csv` (`source_row` is the row in the train file), 38,176 controls.
- `vcc-h1` prediction file: 50,400 cells (400 per target), the exact 18,080-gene axis, one obs column `target_gene`, no control cells, raw non-negative integer counts in a sparse matrix.

## 4. X-Atlas/Orion (Xaira, Hugging Face)

- `Xaira-Therapeutics/X-Atlas-Orion`, CC BY-NC-SA 4.0. HCT116: 109 batch files plus metadata, 46.6 GB; HEK293T: 223 files, 79.7 GB. Fetched by `python -m vcell etl download xatlas` (sha256 checked).
- **One row per cell**, one Parquet row group per file (about 18,500 cells, about 10,000 targets per file, a handful of cells per target per file):

| Column | Type | Meaning |
|---|---|---|
| `gene_token_id` | list of int64 | genes with non-zero counts in this cell |
| `gene_expression` | list of double | their raw counts, same order and length |
| `cell_barcode`, `sample` | string | barcode (suffix replaced by the sample) and GEM batch |
| `num_features` | int64 | number of guides detected |
| `guide_target`, `gene_target` | string | guide identity and the gene it targets (`Non-Targeting` for controls) |
| `n_genes_by_counts`, `total_counts`, `total_counts_mt`, `pct_counts_mt` | numeric | per-cell QC |
| `pass_guide_filter` | int64 | 1 if the cell holds two guides of the same pair |

- This is a sparse matrix written as nested columns: a cell with 4,883 non-zero genes has two lists of length 4,883. Zeros are not stored.
- `metadata/gene_metadata.parquet` maps `gene_token_id` (0 to 38,605) to `ensembl_id` and `gene_name`. Saved as `data/sources/xatlas/gene_metadata.parquet`.
- Cell filter used: `pass_guide_filter == 1`. Control label `Non-Targeting`. Depth: median about 16,000 UMI.
- Small companion files (Figshare): `guide_library.csv`, and `<line>_filtered_guide_calls_per_cell.csv.gz` (one row per cell; used for the target list and counts).

## 5. K562 genome-wide pseudobulk (Replogle 2022)

- AnnData, **dense** `X` of mean raw counts per cell (rows are perturbations, columns genes), 0.37 GB. Rows: about 9,900 targets plus 585 non-targeting pseudobulks.
- `obs`: `gene_transcript` (for example `3612_SMARCA5_P1P2_ENSG...`; the symbol and Ensembl ID are parsed from it), `num_cells_filtered`, `UMI_count_unfiltered`, `anderson_darling_counts` (number of DE genes), `pct_expr` (the target's own expression change; knockdown is taken as its negative). Gene names are stored as a categorical in `var`.
- Non-targeting rows have `non-targeting` in the name. The 585 of them define `s` (spread against their pool, scaled by 1/sqrt(cells)).
- Costs of using a pseudobulk instead of single cells: no per-cell variance, normalisation after averaging, no guide or batch information. The single-cell file (65.8 GB) is optional (`archive/DATASET_PREPARATION.md` 3.6).

## 6. CD4+ T cells (Zhu 2025)

- DE statistics (16.8 GB, read remotely row by row): 33,983 rows by 10,282 genes. Layers `log_fc` and `lfcSE` (uncompressed float64) hold the DESeq2 result for each row.
- `obs`: `target_contrast_gene_name`, `culture_condition` (Rest, Stim8hr, Stim48hr), `n_cells_target`. `var/gene_name` holds symbols.
- One row is one target in one condition: the **effect is already computed**, there are no cells here.
- Knockdown table `data/sources/zhu/guide_kd_efficiency.suppl_table.csv`: index is the guide (`GENE-1`), columns include `guide_n`, `guide_mean_expr`, `ntc_mean_expr`, `culture_condition`.

## 7. Stage 1 common format (`out/stage1_<line>.npz`)

Same axes for every line: 575 targets (the 300 panel targets plus every H1 target) by the 11,961 genes above 5 CPM in at least one of A, B, C.

| Array | Shape | Content |
|---|---|---|
| `targets`, `genes` | (T,), (G,) | symbols |
| `beta` | (T, G) | log2 fold change, NaN where missing, never zero |
| `s` | (T, G) | standard error, NaN where missing |
| `n` | (T,) | cells with the target (NaN if not measured) |
| `ctrl_cpm` | (G,) | mean CPM in the line's controls |
| `kd` | (T,) | knockdown of the target itself where known |

The target gene is NaN in its own row. Files: `K562`, `CD4_Rest`, `CD4_Stim8hr`, `CD4_Stim48hr`, `H1` (held-out benchmark truth), later `HCT116`, `HEK293T`. For X-Atlas, the sums for all ~18,300 targets are kept in `data/sources/xatlas/<line>_accum.npz` (`n`, `S`, `SS` of CPM). Cleaned versions (section 8) are saved as `stage1_<line>_qc.npz` and `<line>_qc_accum.npz`.

## 8. Cell-level QC of the source screens (`vcell.qc.sources`)

Applies only to public screens with single cells. Never to the organisers' contexts (the scored truth cells come from the same pipeline as the controls we resample, F2) and never to the `vcc-h1` benchmark truth.

**The steps** (code and thresholds in `vcell.qc.sources`; thresholds are per batch, over all cells of the batch, so a knockdown that really changes depth or mito is judged against its batch):

| # | Step | Rule | Why this way |
|---|---|---|---|
| 1 | Low quality | log1p(total UMI) or log1p(genes detected) more than 5 MADs below the batch median; mito share more than 3 MADs above the median **and** above 5%, or above 20% | MAD rules adapt to each dataset (sc-best-practices). The 5% floor stops the rule firing on probe-based (10x Flex) data, where mito is near zero |
| 2 | Multiplets | depth or genes more than 5 MADs above the median; cells with guides from more than one construct | In Perturb-seq the guide call is the main doublet detector; computational doublet scoring over millions of streamed cells is not affordable here |
| 3 | Ambient RNA | not corrected | No raw droplets for most sources; ambient RNA is shared by perturbed and control cells of a batch and largely cancels in the log fold change |
| 4 | Genes | unchanged | The 5 CPM universe and NaN for zero means already handle it |
| 5 | Targets | a target row needs 20 cells after QC | Escaper cells (guide present, no knockdown) are **kept**: the truth cells are defined by their guide too, and whole-target knockdown weighting was neutral (H23) |

**Audit** (`python -m vcell qc sources`, log `out/qc_audit.log`, masks `out/qc/<source>.npz`):

| Source | Cells | Removed | Mostly | Controls / perturbed removed | Per-target removal (median / max) | Decision |
|---|---|---|---|---|---|---|
| X-Atlas HCT116 | 3,409,169 | 4.39% | high mito | 4.33% / 4.39% | 4.2% / 23% | cleaned (`HCT116_qc`), not adopted (H52) |
| X-Atlas HEK293T | 4,534,299 | 2.50% | high mito | 2.38% / 2.51% | 2.3% / 24% | cleaned (`HEK293T_qc`), not adopted (H52) |
| KOLF2.1J | 2,659,209 | 0.84% | high mito | 0.60% / 0.86% | 0.7% / 14% | not needed after H52 (`vcell.etl.stage1.kolf` can still write `KOLF_qc` if the mask exists) |
| Arc H1 (train) | 221,273 | 0.01% | none | 0.01% / 0.01% | 0% | already clean; no rebuild |
| Replogle K562 single cells | 1,989,578 | 1.61% | mito, low depth | 1.48% / 1.61% | 1.4% / 39% | not used in the pool (H20); left as published |
| Replogle RPE1, Nadig HepG2, Jurkat | 145,000-263,000 | 1.8-2.1% | high mito | 0.8-1.1% / 1.8-2.2% | about 1% / 46-50% | evaluation truth only; left as published |

The Replogle and Nadig files show the risk of QC on perturbation data: essential knockdowns (TAF, MED, SUPT5H, mitochondrial ribosome MRPL genes) lose 30 to 50% of their cells, because the knockdown itself raises mito or lowers depth. Removing them would remove the phenotype. In X-Atlas and KOLF the removal is flat across controls and perturbed cells, so it removes damaged cells, not biology.

**Result (H52):** using the cleaned X-Atlas tables in the pool changes the leave-one-line-out proxy by -0.0013 on average (K562, CD4, H1 up 0.0002 to 0.0019; HCT116 and HEK293T down 0.002 and 0.007, also against cleaned truth). The published tables stay in the final build.

**Not cleanable (no cells):** K562 genome-wide pseudobulk (the authors' filtered cells), CD4+ T DESeq2 statistics, HKU moments (Pan_H1, Feng_iPSC). Their QC is the authors'.
