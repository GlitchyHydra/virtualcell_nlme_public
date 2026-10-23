# Dataset preparation (living document)

Started 2026-10-10, updated 2026-10-13. This file covers which source datasets to use, how many, how to check them, and how to turn them into the Stage 1 table that the model needs. The model itself is in `MODEL_DESIGN.md`, the schedule and download sizes in `APPROACH.md` (5.4), the scorer in `SCORING.md`.

Status markers: **decided**, **proposed**, **open**. No source dataset is in the repository yet; the numbers in section 3 were measured on 2026-10-13 from small files held in a temporary folder and from remote reads.

---

## 1. Why this is needed

The two-stage model (`MODEL_DESIGN.md` section 3) pools effects across cell lines. Stage 1 (3.2) produces one row per (line, target g, response gene j) with `beta_hat` and `s`; Stage 2 (3.3) needs the same pair in at least two lines to say anything about `tau^2`. Contexts A, B and C have no perturbed cells, so every effect must come from public Perturb-seq data. This file is about getting that data into one consistent table.

## 2. How many lines

**decided.** Aim for 5 independent cell lines, with 3 as the minimum for a pilot.

Reasoning (full discussion in `MODEL_DESIGN.md` 3.3 and 3.5):
- `tau^2` is learned as a shared trend across thousands of pairs, so 2 or 3 lines can support it. More lines make the predictive sd more trustworthy for unlike lines such as D, E, F.
- Leave-one-line-out trains on K-1 lines. With 3 lines that is 2, the weakest regime; with 5 it is 4.
- Count independent **lines**, not datasets. Two K562 datasets are replication (they lower `s` and expose batch effects), not extra information on how effects differ between cell types.
- Target coverage matters as much as line count. A line only helps for the targets it perturbed.

**proposed (2026-10-13).** Which lines these are has changed. Only genome-wide screens cover the panel (section 3), so the usable lines are K562, HCT116, HEK293T and primary CD4+ T cells, with KOLF2.1J iPSC as an optional fifth. H1 is a benchmark line, not a source. RPE1, HepG2 and Jurkat drop out as sources.

## 3. Candidate sources

**proposed.** Analysed 2026-10-13. This replaces the 2026-10-10 version, whose ranking was an inference. The numbers below were measured against `pert_counts.csv` and the three control files, using each dataset's own metadata or pseudobulk files (about 0.7 GB downloaded, the rest read remotely by HTTP range request). Figures quoted from papers are marked as such. The scripts that produced the numbers are not in the repository yet.

### 3.1 What the model needs from a source

Derived from `MODEL_DESIGN.md`. A dataset is judged on these, in this order.

| # | Priority | From | What to look for |
|---|---|---|---|
| 1 | Panel targets measured in 3 or more independent lines | 3.3, 3.5 (tier 1) | Share of the 300 targets a line perturbed |
| 2 | Breadth for the unknown test targets | section 8 question 4; test list arrives 2026-10-22 | Genome-wide screens, not gene-set screens |
| 3 | Usable Stage 1 input | 3.2 | CRISPRi knockdown; perturbed and non-targeting cells from one pool; raw counts; guide labels; single cells, or a mean plus a noise estimate |
| 4 | Precision of `beta_hat` | 3.2 worked example | Cells per perturbation, depth, knockdown of the target itself |
| 5 | Same effect regime as the panel | 3.3 (shared `tau^2` trend), 3.4 (prior), 4.2 | Hyperparameters learned from pairs that look like the panel's |
| 6 | Response-gene coverage | section 2 item 4; 3.5 missingness 2 and 3 | Share of each context's testable genes (above 5 CPM) on the source gene axis |
| 7 | A held-out line the scorer can run on | section 7 | Single cells, 400 or more per perturbation |
| 8 | Control profile of the line | 3.6 rung v4 | Control cells or their mean, for similarity weighting |
| 9 | Cost | `APPROACH.md` 1.4 | Download size, RAM, format, licence, time before 2026-10-22 |

### 3.2 Findings

1. **None of the 300 validation targets is in any essential-gene screen.** Replogle K562 essential, Replogle RPE1, Nadig HepG2 and Nadig Jurkat cover 0 of 300, by gene symbol and by Ensembl ID. These four screens share one library of about 2,393 genes. The 2025 H1 panel did overlap it (50 of 150 training targets), so the 2026 panel differs in this respect. Whether Arc excluded these genes on purpose is not stated.
2. **Genome-wide screens cover almost all of it.** K562 genome-wide 272 of 300, HCT116 and HEK293T 300 of 300, CD4+ T cells 291 to 297, KOLF2.1J iPSC 282. This confirms the organisers' note that K562 genome-wide overlaps most challenge genes.
3. **The earlier five-line plan gives no tier 1 target.** H1, K562, RPE1, HepG2 and Jurkat together measure a panel target in at most 2 lines, and 28 targets in none.
4. **The panel targets are weak perturbations.** In K562 their median is 5 differentially expressed genes (quartiles 1 and 18; Replogle's Anderson-Darling count), against 156 for the essential-screen genes in the same experiment. In CD4+ T cells the median is 5 at 10% FDR. So `tau^2` and the sparse prior should be learned from non-essential targets, not from the essential screens.
5. **Small files exist for the two cheapest sources.** K562 genome-wide has a 0.37 GB raw pseudobulk (the single-cell file is 65.8 GB). The CD4+ T cell DE statistics (log2 fold change and its standard error) can be read row by row from a remote 16.8 GB file, about 0.15 GB for the panel.
6. **K562 covers only about 71% of the testable response genes.** Its gene axis holds 8,248 genes. The rest are missing rows for that line (`MODEL_DESIGN.md` 3.5, missingness type 2), never zeros. X-Atlas/Orion covers 98%, CD4+ T cells 82 to 85%.
7. **Context A looks like Jurkat. B and C match no source line.** This is an inference from the control cells, not a statement by the organisers.
   - A: T-cell genes are high (CD3D 939, CD3E 760, LCK 431 CPM), with TAL1 and NKX3-1 expressed. Its control profile correlates 0.71 (Spearman) with Nadig Jurkat, against 0.31 and 0.35 for B and C. The three contexts correlate 0.45 to 0.61 with each other.
   - B: keratins 8 and 18 with very high VIM, no EPCAM or CDH1, no Y-chromosome gene.
   - C: EPCAM, KRT5 and SOX2 high, VIM absent, Y-chromosome gene present.
   - Among the genome-wide sources, A is closest to CD4+ T cells (0.52) and K562 (0.50), B and C to HCT116 (0.44 and 0.49). HEK293T was not measured. The source profiles come from different chemistries and differently filtered gene lists, so this ranking is rough.

### 3.3 Coverage of the 300 targets

Cells are per covered panel target. "Testable genes" is the share of genes above 5 CPM in A, B, C that are on the source gene axis.

| Dataset | Line | Targets in screen | Panel covered | Cells, median (quartiles) | At least 50 cells | Knockdown, median (share at 80% or more) | Testable genes, A / B / C | Controls |
|---|---|---|---|---|---|---|---|---|
| [Replogle K562 genome-wide](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387) | K562 | 9,866 | **272** | 172 (126, 225) | 95% | 87% (73%) | 71 / 72 / 70% | 75,328 cells, 585 guides |
| [X-Atlas/Orion](https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion) | HCT116 | 18,293 | **300** | 149 (87, 223) | 88% | 75% over all targets (paper); per target not measured | 98% | 165,777 cells |
| X-Atlas/Orion | HEK293T | 18,311 | **300** | 200 (113, 306) | 92% | 52% over all targets (paper) | 98% | 218,838 cells |
| [Zhu et al. CD4+ T](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq) | CD4+ T, 4 donors, 3 conditions | 12,733 | **297** perturbed, **291** with DE results | about 500 per condition (273, 722 at rest) | 96 to 97% | 83 to 88% (58 to 68%) | 85 / 82 / 82% | about 300,000 per condition |
| [KOLF2.1J atlas](https://plus.figshare.com/articles/dataset/_b_A_genome-scale_CRISPRi_perturbation_atlas_of_human_induced_pluripotent_stem_cells_b_/27261219) | KOLF2.1J iPSC | about 11,700 | **282** | 236 | 99% | not measured | not measured (axis has 37,567 genes) | 146,747 cells |
| VCC 2025 H1, training set | H1 hESC | 150 | 13 | 938 (589, 1,795) | 100% | not measured | 99% | 38,176 cells |
| VCC 2025 H1, all three splits (train, validation, test; 2026-10-14) | H1 hESC | 300 | 25 | median about 1,090 per target | 93% K562, 99.7% HCT116 and HEK293T, 95% CD4+ T | not measured | gene axis 18,080 | 38,176 shared controls; 376,518 perturbed cells |
| Replogle K562 essential | K562 | 2,057 | **0** | | | | 74 / 74 / 72% | 10,691 cells |
| Replogle RPE1 | RPE1 | 2,393 | **0** | | | | 74 / 78 / 75% | 11,485 cells |
| Nadig HepG2 | HepG2 | 2,393 | **0** | | | | 78 / 80 / 79% | 4,976 cells |
| Nadig Jurkat | Jurkat | 2,393 | **0** | | | | 81 / 75 / 74% | 12,013 cells |
| [Jiang et al.](https://zenodo.org/records/10520190) | A549, MCF7, HT29, HAP1, BxPC3, K562 | 218 | 9 | not measured | | | not measured | |
| [McFaline-Figueroa et al.](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225775), kinome screen | A172, T98G, U87MG | 523 | 23 | not measured | | | not measured | |
| Srivatsan et al. (sci-Plex) | A549, MCF7, K562 | chemical, no gene targets | not applicable | | | | | |

Notes:
- The 28 panel targets missing from K562 genome-wide: ABCD1, CAPRIN2, EPHB2, HEATR6, KIF21B, MAPK7, MLKL, NICN1, PARP3, PBLD, PDK3, PHF11, PSMB9, RAB11FIP5, SEMA4F, SLC44A1, SNN, SOCS5, SPATA20, TAF4, TBC1D19, TDRD7, TESK2, TMEM104, TMEM79, TTBK2, TXNDC16, ZC2HC1A.
- The CD4+ T screen did not perturb EEF1A2, EPHB2 or FZD2.
- Checked on 2026-10-14 over all three 2025 H1 splits: 25 of 300 targets are in the 2026 panel (train 13, validation 4, test 8).
- Jiang panel targets: ELK1, FOXO4, IFNAR2, IFNGR2, MED15, MTF1, SMARCA5, STAT6, ZNF22.
- 8,999 targets are perturbed in all four of K562, HCT116, HEK293T and CD4+ T. That is the pool for learning `tau^2` and the prior.

### 3.4 Tiers for the 300 targets

Tiers as in `MODEL_DESIGN.md` 3.5. The second set of numbers counts a line only if it has at least 50 cells for the target.

| Source lines | Tier 1 (3 or more) | Tier 2 (1 or 2) | Tier 3 (none) | With at least 50 cells |
|---|---|---|---|---|
| Organisers' Replogle and Nadig sets (K562 genome-wide, RPE1, HepG2, Jurkat) | 0 | 272 | 28 | 0 / 258 / 42 |
| K562 + CD4+ T | 0 | 296 | 4 | 0 / 295 / 5 |
| K562 + HCT116 + CD4+ T | 267 | 33 | 0 | 221 / 79 / 0 |
| K562 + HCT116 + HEK293T + CD4+ T | 296 | 4 | 0 | 272 / 28 / 0 |
| The same plus KOLF2.1J | 299 | 1 | 0 | 291 / 9 / 0 |

### 3.5 Verdict per dataset

| Dataset | Role | Reason |
|---|---|---|
| Replogle K562 genome-wide | **Source, first** | 272 targets. Raw pseudobulk is 0.37 GB, CC BY 4.0. Means only, so `s` comes from the 585 non-targeting pseudobulks (section 3.6). Weakness: 71% response-gene coverage |
| Zhu CD4+ T | **Source, second** | 291 targets, about 500 cells each, ready `beta_hat` and `s` (DESeq2 `log_fc` and `lfcSE`; whether `log_fc` is shrunk is not stated **[verify]**). MIT licence. 10x Flex on Ultima, the challenge's chemistry. With K562, the genome-wide source closest to context A. One line with three conditions, not three lines |
| X-Atlas/Orion HCT116 | **Source, third** | 300 targets, 98% response-gene coverage, closest genome-wide source to B and C. 46.6 GB of Parquet, read as a stream. CC BY-NC-SA 4.0, preprint |
| X-Atlas/Orion HEK293T | **Source, fourth** | 300 targets, but 79.7 GB and a median knockdown of 52%, so effects are attenuated. Filter or weight by each target's knockdown |
| VCC 2025 H1 | **Benchmark line, not a source** | 25 panel targets only. But 271 of its 300 targets (train, validation, test) are in all four source lines (293 in three or more), 298 have at least 50 cells (median about 1,090), and the chemistry matches 2026. It is the line to hold out and score with `cell-eval2`. The three files share the same 38,176 control cells (use them once); depth is about 54,000 UMI per cell and there are 48 batches |
| KOLF2.1J iPSC | Later | A fifth genome-wide line (282 targets) and the closest public analogue of a stem-cell line. One 189 GB file, stored column-wise, so too costly before the deadline |
| Nadig Jurkat | Optional benchmark line | No panel target, but it resembles context A. It can test transfer into a Jurkat-like line on its 2,393 essential genes, which the source lines also perturbed. Strong-effect regime, so a secondary check |
| Replogle K562 essential, RPE1; Nadig HepG2 | Skip | No panel target; wrong effect regime for the hyperparameters |
| Jiang et al. | Skip | 9 panel targets; cells were stimulated with cytokines for 24 hours; data are Seurat objects (R). Its per-line DE tables (0.32 GB) could later check `tau^2` across six lines |
| McFaline-Figueroa et al. | Skip | 23 panel targets (kinases only); combined with drug exposure; low-depth combinatorial indexing; R objects |
| Srivatsan et al., Tahoe-100M | Not usable | Small-molecule screens, no gene target for Stage 1 |

### 3.6 What each source provides for Stage 1

| Source | `beta_hat` from | `s` from | Differs from `MODEL_DESIGN.md` 3.2 by |
|---|---|---|---|
| K562 pseudobulk | Ratio of mean counts, rescaled to CPM over the pseudobulk | Spread of the 585 non-targeting pseudobulks against their pool, per gene, scaled by 1/sqrt(cells) | Pseudobulk CPM, not the mean of per-cell CPM; no per-cell variance |
| CD4+ T DE statistics | DESeq2 log2 fold change on pseudobulks | DESeq2 standard error, which includes donor and guide variation | A negative-binomial estimate, the alternative already listed in 3.2 |
| X-Atlas/Orion, H1 | Mean of per-cell CPM, as specified | Delta method, as specified | Nothing, but this `s` reflects cell sampling only |

The K562 route was checked on the downloaded file. Across the 509 non-targeting guides with at least 20 cells, the spread of `beta_hat` times sqrt(cells) is the same for guides with few, medium and many cells, so the 1/sqrt(cells) scaling holds. Implied `s` for one target:

| Control CPM (on the K562 gene axis) | Genes | `s` at 50 cells | `s` at 170 cells | `s` at 400 cells |
|---|---|---|---|---|
| 5 to 10 | 383 | 0.673 | 0.365 | 0.238 |
| 10 to 50 | 4,893 | 0.422 | 0.229 | 0.149 |
| 50 to 200 | 2,301 | 0.223 | 0.121 | 0.079 |
| 200 to 1000 | 536 | 0.126 | 0.068 | 0.045 |
| above 1000 | 135 | 0.064 | 0.035 | 0.023 |

At 400 cells this is close to the context A table in `MODEL_DESIGN.md` 3.2 (0.226, 0.119, 0.067, 0.041, 0.030), mostly a little above it. At the typical 170 cells, a K562 estimate is about 1.6 to 1.9 times noisier than a 400-cell estimate in the challenge data for genes below 1000 CPM. The bins are not identical, because CPM here is taken over K562's 8,248 genes.

**open.** The three kinds of `s` are not on the same footing. A cell-sampling `s` is smaller than one that includes guide and donor variation, so it would give X-Atlas/Orion too much weight in Stage 2. Proposed fix: run the null check of section 5 on each line's own non-targeting guides and rescale `s` per line.

### 3.7 Proposed order

Today is 2026-10-13 and the test targets arrive on 2026-10-22.

1. K562 genome-wide pseudobulk (0.37 GB). Stage 1 for all 9,866 targets.
2. CD4+ T DE statistics. Read the rows for the panel targets remotely; download the 16.8 GB file only if all targets are wanted.
3. H1 (train, validation, test, 34.4 GB; downloaded 2026-10-14) as the held-out benchmark line.
4. X-Atlas/Orion HCT116 (46.6 GB), streamed once, accumulating sums and sums of squares for **all** targets. This gives tier 1 for 267 targets.
5. X-Atlas/Orion HEK293T (79.7 GB), the same way. Tier 1 for 296 targets.
6. After the deadline, or if time allows: K562 single cells (65.8 GB) for the exact `s` and the negative-binomial ablation; KOLF2.1J.

Build Stage 1 for every target of a source, not only the 300. The test list is unknown until 2026-10-22, and a second pass over 126 GB of X-Atlas/Orion would cost a day. All-target tables also feed the gene-feature fallback (rung v5) and the hyperparameters. This changes `MODEL_DESIGN.md` 4.1, which says only the 300 targets are needed.

### 3.8 File notes

- Sizes on the organisers' dataset page are in GiB. Their "K562 Genome-wide (61.3 GB)" is the 65.8 GB Figshare file, "K562 (9.9 GB)" the 10.7 GB one, and so on. `APPROACH.md` 5.4 quotes the GiB figures.
- In the Replogle Figshare record, `raw` means UMI counts and `normalized` means z-scored per gem group. Stage 1 needs `raw`. Licence CC BY 4.0.
- The Arc copy on Hugging Face (`arcinstitute/Replogle-Nadig-Preprint`) differs: its `K562_essential_normalized_singlecell_01.h5ad` holds integer counts despite the name, and `replogle.h5ad` (22.3 GB) is a log-normalised merge of four lines on 6,642 genes, which Stage 1 cannot use. Both were checked on the first 200 cells only.
- Nadig files are dense float32 matrices of integer counts (HepG2 145,473 cells by 9,624 genes; Jurkat 262,956 by 8,882; integers checked on the first 200 cells), GEO GSE264667, no pseudobulk.
- X-Atlas/Orion on Hugging Face: 109 Parquet files for HCT116 and 223 for HEK293T, one row per cell with sparse raw counts. The Figshare h5ad files are 209 GB and 350 GB. `guide_library.csv` (4 MB) and the per-cell guide calls (70 and 91 MB) are on Figshare.
- Zhu CD4+ T on `s3://genome-scale-tcell-perturb-seq/marson2025_data/` (public): cell-level files 119 to 173 GB each (1.8 TB in all), pseudobulk by guide, donor and condition 44.6 GB, DE statistics 16.8 GB as uncompressed float64 layers of 33,983 rows by 10,282 genes.
- H1 is on Arc's public bucket `gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/` (`train/adata_Training.h5ad` 15.5 GB, `validation/adata_Validation.h5ad` 6.9 GB, `test/adata_Test.h5ad` 12.0 GB, plus `pert_counts_*.csv` and `gene_names.csv`, which has no header row). Fetched by `eda/05_download_h1.py`; the big files have only a crc32c checksum. Each file holds the 38,176 control cells again (same barcodes). The Hugging Face mirror `cyrilzakka/arc-institute-virtual-cell-dataset` is not needed.

## 4. Pipeline

**proposed.** One step per stage, each producing a file that the next step reads.

| Step | Action | Output |
|---|---|---|
| 1 | Download or stream one dataset at a time into its own directory | Raw files, untouched |
| 2 | **Coverage report**: for each dataset, how many of the 300 panel targets (`data/controls/pert_counts.csv`) were perturbed, and how many response genes are in the 18,533 panel (`gene_names.csv`) | Table per dataset, then the tier 1, 2, 3 split from `MODEL_DESIGN.md` 3.5. Done for the validation panel (section 3); repeat on 2026-10-22 for the test panel |
| 3 | **Harmonise genes** to the 18,533 symbols; record genes missing from each dataset as missing, never zero | Gene map per dataset |
| 4 | **QC**: check the target's own knockdown, drop weak perturbations or flag them, count cells per target and per guide | QC table, quality flag column |
| 5 | **Stage 1 builder**: `beta_hat` and `s` by the route in section 3.6, for all targets of the source | Arrays of targets by genes per line |
| 6 | **Assemble the atlas**: stack the lines; select the panel targets at prediction time | Effect atlas (`MODEL_DESIGN.md` 4.1) |

### 4.1 Stage 1 table

Columns, from `MODEL_DESIGN.md` 3.2: `dataset, line, target g, response gene j, beta_hat, s, n_p, n_c, m_control`, plus the target's knockdown flag. For the 300 targets this is about 3 million (target, gene) values per line. For all targets of a genome-wide line it is up to 18,300 by 18,533, so store it as arrays (about 1.4 GB per quantity per line in float32), not as a long table.

### 4.2 Decisions that change the table

These are the open Stage 1 questions (`MODEL_DESIGN.md` 3.2, open list) that affect what is built here:
- Which control cells form the control group: all non-targeting cells of the dataset, or only those matched by batch or lane.
- Zero perturbed mean: pseudocount or drop the row.
- Several guides per gene: pool or estimate separately.
- Compositional shifts from CPM when a knockdown changes many genes.
- CD4+ T conditions (rest, 8 hours, 48 hours after stimulation): use one, or combine the three into one line-level estimate.
- HEK293T: whether to rescale effects by the target's knockdown or only down-weight them.

Record the choice made for each dataset in this file.

## 5. Checks before trusting a dataset

**proposed.**
- Overlap with the 300 targets, reported by tier.
- Share of 18,533 genes present, and share passing the scorer's 5 CPM filter in that line's own controls (`SCORING.md`).
- Cells per perturbation. `s` scales roughly with 1/sqrt(cells), and public data has fewer than the 400 per perturbation in the 2026 panel (150 to 240 in the cell lines, about 500 in CD4+ T).
- Chemistry and normalisation differences between datasets; flag anything that makes `beta_hat` non-comparable.
- Null check as in `MODEL_DESIGN.md` 3.2, on each line's own non-targeting guides: confirm about 5% exceed 1.96 `s`, and rescale `s` if not (section 3.6).
- Agreement of the estimators: on one line with single cells, compare the pseudobulk ratio with the mean of per-cell CPM.

## 6. How the lines are used for evaluation

**decided** (see `MODEL_DESIGN.md` section 7). Lines take part in leave-one-line-out: each is held out once and the model trains on the others. No line is held out permanently. The final model, the one that predicts D, E and F, trains on all source lines.

**proposed (2026-10-13).** With the lines of section 3, "held out" means different things:

| Held-out line | Truth available | Scored with |
|---|---|---|
| H1 (2025) | Single cells, 300 targets (298 with at least 50 cells; subsample to 400 cells where available for the scorer) | `cell-eval2`, the real metrics. Primary benchmark |
| HCT116, HEK293T | Single cells, about 150 to 200 per target, if kept during the streaming pass | `cell-eval2` on fewer cells, or effect-level metrics |
| K562, CD4+ T | Effects only (pseudobulk, DE statistics) | Effect-level metrics: sign agreement and correlation on the genes the held-out line calls significant |
| Nadig Jurkat (optional) | Single cells, essential genes | A check of transfer into a line like context A |

Caveat for H1: its targets are stronger than the 2026 ones (a mean of 1,519 DE genes of 10,780, from the public benchmark's notes) and a third of them are essential-screen genes.

**decided.** Contexts A, B and C are an external check. We have no perturbed cells for them, but the organisers score submissions on them through the validation leaderboard (`APPROACH.md` 1.2, 1.3 and 5.3). A submission predicted from the atlas for A, B and C is therefore a test on three lines that no training or tuning ever saw. This replaces the idea of keeping one source line back: every source line can stay in training.

How to use it (`APPROACH.md` 5.7 ranks the live leaderboard last, as a sanity check):
- Submit only finished rungs of the ladder (`MODEL_DESIGN.md` 3.6), not every experiment. Check the submission limits **[verify]**.
- Record each pair (local leave-one-line-out score, leaderboard score) in one table. The aim is the plot listed in `APPROACH.md` section 7 (Portfolio): does the local benchmark predict the leaderboard?
- Do not tune to it. The leaderboard shows a subset of the six metrics **[verify which]**, and the final round uses other lines (D, E, F), other genes, and no score feedback.
- If local and leaderboard orderings disagree, that is a finding about transfer to new lines, not a reason to chase the leaderboard.

**proposed.** Learning curve: in leave-one-line-out, train on subsets of the other lines (K = 1, 2, 3) and plot the held-out score against K. This decides whether further downloads are worth it and shows how sensitive `tau^2` is to K.

## 7. Open questions

Answered on 2026-10-13:
- Target overlap of each candidate with the 300 panel targets: section 3.3.
- A published pseudobulk of Replogle genome-wide exists (0.37 GB, Figshare). This also answers `APPROACH.md` open question 5.
- A summary-statistics version of Zhu CD4+ T exists (DE statistics, 16.8 GB).
- Licences: Replogle CC BY 4.0, Zhu MIT, KOLF2.1J CC BY 4.0, Jiang CC BY 4.0, X-Atlas/Orion CC BY-NC-SA 4.0. Nadig (GEO) and H1 not stated.

Still open:
1. Submission limits on the validation leaderboard, and which metrics it shows.
2. Whether the test targets follow the same pattern as the validation ones (none in the essential screens, about 90% in K562 genome-wide). Check on 2026-10-22.
3. Whether a non-commercial licence (X-Atlas/Orion) is compatible with the challenge terms, given that it awards prizes.
4. Whether H1 can stand in as a held-out line, given its stronger targets (`MODEL_DESIGN.md` section 7, open).
5. How to put the three kinds of `s` on one footing (section 3.6).
6. Which CD4+ T condition to use, and how to treat HEK293T's weak knockdown (section 4.2).
7. What lines D, E and F are. Context A looks like Jurkat; B and C match no source (section 3.2). Whether the challenge rules restrict using a line's identity has not been checked.
8. The X-Atlas/Orion knockdown for each panel target (needs the expression data).

## 8. Checklist

- [x] Coverage report for the validation panel from gene lists (section 3.3)
- [ ] Move the K562 pseudobulk and the coverage scripts into the repository
- [ ] Stage 1 from the K562 pseudobulk, all targets, with `s` from the non-targeting pseudobulks
- [ ] Read the CD4+ T DE rows for the panel targets
- [x] Download H1 (2026-10-14, checksums verified)
- [x] Set up the H1 benchmark (2026-10-15): `vcc-h1` v0.4.0 (pins `cell-eval2==0.16.0`), 126 targets; unchanged-control baseline scored -0.302589, matching the published -0.302589 (`eda/out/h1_control_baseline/`). Top of the scale: 400 fresh real H1 cells per target (30 targets topped up from the reference sample) scored +1.067 (`eda/out/h1_positive_control_fresh/`, `eda/07_h1_positive_control.py`)
- [ ] Stream X-Atlas/Orion HCT116, then HEK293T; Stage 1 for all targets
- [ ] Gene harmonisation and QC (steps 3, 4)
- [ ] Null check and `s` rescaling per line
- [ ] Assemble atlas (step 6)
- [ ] 2026-10-22: coverage report for the test panel
