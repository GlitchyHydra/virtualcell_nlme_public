# Virtual Cell Challenge 2026: working plan

Written 2026-10-08 for a solo, part-time entry that started today. The goal is skill growth, a portfolio piece, and a head start on 2027, not a prize.

Sources are listed at the end. Items marked **[verify]** come from secondary sources or my inference and should be checked against the official pages (the evaluation page needs a browser; it would not load for me).

---

## 1. The situation in one page

### 1.1 The task

- Zero-shot. You get unperturbed (non-targeting) cells for three anonymous cell lines and a list of 300 genes. You predict the single-cell expression after CRISPRi knockdown of each gene in each line.
- There is no challenge training set. Any public or proprietary data is allowed.
- Output: 360,000 cells (300 perturbations x 400 cells x 3 contexts) x 18,533 genes, raw integer counts, sparse, packaged with `vcc prep`.
- The final test uses three different cell lines (D, E, F) and a different list of 300 genes. Validation and final scores are not comparable.

### 1.2 Dates

| Date | Event | Days from today |
|---|---|---|
| 2026-08-20 | Validation data and leaderboard opened | started 7 weeks ago |
| 2026-10-08 | You joined | 0 |
| 2026-10-22 | Final test inputs (D, E, F) released | 14 |
| 2026-11-05 23:59 UTC | Final submission deadline | 28 |
| mid/late November | Winners announced | |

The final phase gives no score feedback. Whatever you submit for D/E/F is judged blind, so a local benchmark matters more than the leaderboard.

### 1.3 How it is scored

Six metrics, computed by `cell-eval2` with the `vcc2026` preset. Each is rescaled so that 0 is the mean-response baseline for that context and 1 is a real replicate experiment.

| Metric id | Family | What it asks |
|---|---|---|
| `pds_cosine` | discrimination | Is each predicted effect closer to its own measured effect than to the other 299? It is a rank, so it rewards perturbation-specific signal, not magnitude. |
| `expr_mse_unbiased_capped_norm` | expression | How large is the expression error? |
| `de_wilcoxon_direction_fidelity_yield_raw` | DE | Are the predicted directions of change right? |
| `de_wilcoxon_direction_reach_raw` | DE | How far down the gene ranking do directions stay right? |
| `de_wilcoxon_sig_jaccard` | DE | Does the set of significant genes match? |
| `de_wilcoxon_lfc_nmae` | DE | Are the fold changes accurate? |

Consequences for modelling:

1. **The knocked-down gene itself is not scored.** PDS excludes all 300 panel target genes; the other five exclude each perturbation's own target. Setting the target gene to near zero earns nothing. All the score is in downstream effects.
2. **Four of six metrics run a Wilcoxon test on your 400 cells against controls.** The per-cell distribution matters, not only the mean. Too little noise makes everything significant; too much makes nothing significant.
3. **Shrinking effects is punished.** In a public benchmark, the same transfer method scored 0.083 on the leaderboard at full effect size and -0.183 at 0.13x.
4. **Copying controls is worse than zero.** Resampled control cells score about -0.30.
5. The live leaderboard uses a subset of the metrics; the final ranking uses all six **[verify which subset]**.

Reference points on the validation leaderboard: control resampling -0.30, a plain effect transfer about 0.08 to 0.10, a solo "atlas transfer" entry 0.155 at rank 82 on 2026-09-10. A score of 1.0 would mean matching a real replicate.

### 1.4 Your constraints

| Resource | What you have | What it rules in or out |
|---|---|---|
| Time | 28 days, part-time. I assumed 10 to 12 hours a week, so 40 to 50 hours total | One solid method plus one or two experiments. Not a foundation-model fine-tune. |
| GPU | RTX 3060 Ti, 8 GB | Inference with small models. No training of STATE-size models, no Evo 2 7B/40B. |
| RAM | 32 GB | Cannot load Replogle genome-wide (61 GB) in memory. Use backed mode and pseudobulk. |
| CPU | i5-7600K, 4 cores | Wilcoxon scoring will be slow. Use the GPU DE backend or score a subset of perturbations. |
| Disk | 268 GB free on D: | Enough for about 110 GB of source data, but packaging one submission reportedly needs 30 to 40 GB of temp space. |
| Software | Python 3.11 and 3.13, WSL2 Ubuntu | `cell-eval2` needs Python 3.11+. Windows support is not documented; fall back to WSL2 if installs fail. |

---

## 2. Strategy

### 2.1 What to optimise for

Winning is not realistic from a week-7 start. These are reachable and worth more to you:

1. A complete, reproducible pipeline from public data to a valid final submission.
2. A local benchmark you trust, with an ablation table showing what each idea added.
3. A written account, including negative results, that shows you understand why the simple method works.
4. Reusable assets for 2027: a harmonised effect atlas, the benchmark, and the count generator.

A realistic target is the top third of the leaderboard with a transparent method. That is a stronger portfolio item than an unexplained fine-tune.

### 2.2 The core idea

Perturbation effects of a gene are partly shared across cell lines. Measure them in public Perturb-seq data, transfer them to the new context, and generate cells around the context's own controls.

```
public Perturb-seq  ->  per (cell line, target gene) effect vector (log fold change)
                    ->  combine across cell lines, weighted by similarity to the context
context controls    ->  baseline profile and per-cell noise
effect + baseline   ->  400 integer-count cells per perturbation
```

The 2025 winners combined deep learning with classical statistics, and Arc reported that models did not consistently beat naive baselines. Independent benchmarks agree that a linear baseline captures most of the achievable gain at negligible cost.

---

## 3. Paths compared

| Path | What it is | Fits your hardware | Hours to first result | Expected score | Learning value | Verdict |
|---|---|---|---|---|---|---|
| A. Effect atlas transfer | Average measured effects from public cell lines, apply to context controls | Yes, CPU only | 15 to 20 | Competitive (0.08 to 0.15 from public evidence) | High: data handling, metrics, statistics | **Do this first** |
| B. Linear / low-rank model on gene features | Ridge or low-rank regression from gene embeddings to effect vectors; covers genes with no measured effect | Yes | +8 to 12 | Small gain over A, mainly on uncovered genes | High: this is the baseline every paper compares against | **Do second** |
| C. Context-aware weighting | Weight source lines by expression similarity; gate effects by whether the target and its partners are expressed in the context | Yes | +6 to 10 | Plausible gain; untested | High: this is the actual scientific question of 2026 | **Do third** |
| D. Frozen foundation-model embeddings as features | scGPT, Geneformer, ESM-2, GenePT embeddings as inputs to path B | Mostly (inference only) | +6 to 10 each | Uncertain; published gains are small | Medium: you learn the tooling | One or two as ablations |
| E. Fine-tune STATE or scGPT | Train a transformer on perturbation data | No (needs a rented GPU, 24 GB+) | 30+ | Uncertain; State transfers better than linear across contexts in one study, at about 100,000x the cost | High, but slow | After the deadline |
| F. Flow matching / generative models | The approach Arc recommends reading from a 2025 winning team | No | 40+ | Unknown for you | High | Read now, build in 2027 |
| G. Evo 2 | DNA language model | No for 7B/40B; 1B is marginal | 15+ | Likely none | Low for this task | **Skip** (see 4.2) |

Recommended order: A, then B, then C, with one embedding from D as an ablation if time remains. E and F are your 2027 preparation.

---

## 4. Your three questions

### 4.1 Should you read the papers listed with the data?

Yes, but not all, and not before you have a baseline running. Read in this order.

**Must read (about 6 hours total)**

| # | What | Why | How deep |
|---|---|---|---|
| 1 | `cell-eval2` metric specification, `docs/vcc2026_metrics/` in the repo | It defines what you are optimising. The most valuable document in the whole challenge. | Fully, twice |
| 2 | 2026 challenge paper in Cell, "Benchmarking zero-shot generalization across cellular contexts" | Task design and rules | Fully |
| 3 | Arc's 2025 wrap-up post | What won and what failed | Fully, 20 minutes |
| 4 | Replogle et al. 2022 | Your main training data. You need the experimental design, QC, and how effects were normalised. | Methods and figures 1 to 2 |
| 5 | A benchmark of deep models against linear baselines (Ahlmann-Eltze et al., Nature Methods 2025 **[verify citation]**) | Calibrates expectations and gives you the linear model for path B | Fully |

**Should read (about 4 hours)**

| # | What | Why |
|---|---|---|
| 6 | Nadig et al. 2025 (HepG2, Jurkat) | Second and third source cell lines; it also studies how much effects are shared between lines, which is your transfer assumption |
| 7 | Jiang et al. 2025 (six cancer lines) | The only listed dataset with one perturbation set across six tissues. Check target overlap with your 300 genes before investing; it may be small. |
| 8 | The flow-matching preprint from a 2025 winning team, recommended by Arc | Read for ideas on generating cell distributions, not to implement |
| 9 | STATE preprint (Arc) | The reference deep model; needed context for any write-up |

**Skip for now**

- Srivatsan 2020 (sci-Plex) and McFaline-Figueroa 2024: chemical perturbations and different chemistry. Low relevance to CRISPRi transfer.
- Tahoe-100M and scBaseCount: drug and observational data at a scale you cannot process in four weeks.

### 4.2 Should you use Evo 2?

No, not as a main component.

- Evo 2 models DNA sequence. This task asks how a cell's transcriptome responds when a gene is silenced. That depends on the gene's function and the cell's regulatory state, which a sequence model does not expose directly.
- The useful sizes (7B, 40B) do not fit in 8 GB. The 1B model might run, but its setup is Linux-oriented and would cost you days.
- The only sensible use is embedding each target gene's promoter or coding sequence as a gene feature for path B. Cheaper features carry more relevant information (see 4.3).

If you want Evo 2 on your CV, do it after the deadline as one row in the ablation table: "sequence embeddings added X".

### 4.3 What about other foundation models?

Treat them as feature sources, not as the model.

| Model | Type | Runs on 8 GB | Use here | Priority |
|---|---|---|---|---|
| ESM-2 (650M) | Protein language model | Yes | Embedding of each target gene's protein, as a gene feature | Good first choice |
| GenePT | Text embeddings of gene descriptions | Yes (precomputed) | Gene feature; no GPU needed | Good first choice |
| scGPT | scRNA transformer | Inference yes | Gene embeddings; cell embeddings to compare contexts to source lines | Optional |
| Geneformer | scRNA transformer | Inference yes | Same as scGPT | Optional |
| STATE / SE-600M | Arc's perturbation model and cell embedder | Inference maybe, training no | Reference model; try on a rented GPU after the deadline | Later |
| scFoundation, UCE | Larger scRNA models | Marginal | The 2025 winner built on scFoundation, with a full team and compute | Later |
| Evo 2 | DNA model | No | See 4.2 | Skip |

Non-neural features often beat all of these and cost nothing: DepMap co-essentiality profiles, STRING network neighbours, Gene Ontology and pathway membership. Include them in the same ablation.

Each feature source is one ablation row. Add one at a time and keep it only if the local benchmark improves.

---

## 5. Technical plan

### 5.1 Environment (2 to 3 hours)

1. Initialise git and add a `.gitignore` covering `data/` and any `.h5ad` or `.vcc` file.
2. Create a Python 3.11 environment. Try native Windows first; if `cell-eval2` or the `vcc` CLI fails to install, move the whole project into WSL2 Ubuntu and stop fighting it.
3. Install `anndata`, `scanpy`, `scipy`, `numpy`, `pandas`, `scikit-learn`, `pyarrow`, `cell-eval2` with its GPU DE extra, and the `vcc` CLI.
4. Pin versions. `cell-eval2` changed scored values between releases, so record the version next to every score.
5. Use this layout:

```
src/vcc/        data loading, atlas building, models, count generation
scripts/        one script per pipeline step
configs/        one YAML per experiment
notebooks/      EDA only, never pipeline logic
results/        scores as CSV, one row per experiment
docs/           plan, notes, write-up
```

### 5.2 Understand the inputs (3 to 4 hours)

For each context, from the control file:

- library size distribution, genes detected per cell, sparsity (check against the 13,200 stored entries per cell cap);
- mean and variance per gene, and the dispersion trend;
- variation between the 46 non-targeting guides (`ntc_id`), which is your noise floor and a free estimate of what a "replicate" looks like;
- expression of the 300 target genes in each context. A gene that is not expressed cannot be knocked down, though Arc selected targets with over 80% knockdown;
- which known cell lines each context resembles, by correlating control pseudobulk with your source lines. Use this only for weighting sources. **[verify the rules before attempting to identify the lines from outside expression databases.]**

### 5.3 Submission pipeline before any modelling (3 to 4 hours)

Make a deliberately bad submission first.

1. Generate 400 resampled control cells per perturbation and context.
2. Write one sparse H5AD with `target_gene` and `context` in `obs` and genes in `gene_names.csv` order.
3. Run `vcc prep ... --dry-run`, then package and upload.
4. Confirm the leaderboard returns about -0.30.

This proves the plumbing, the disk budget, and the upload. Do it in week 1. Check the submission limits while you are there **[verify]**.

### 5.4 Source data and the effect atlas (8 to 10 hours)

Download in this order, and check target overlap after each step before downloading the next.

| Order | Dataset | Size | Why |
|---|---|---|---|
| 1 | 2025 challenge H1 (training, validation, test) | about 15 GB | Same chemistry (10x Flex) and same lab as 2026; also your local benchmark |
| 2 | Replogle K562 essential, RPE1 | 9.9 + 8.1 GB | Two lines, good quality |
| 3 | Nadig HepG2, Jurkat | 5.2 + 8.7 GB | Two more tissues |
| 4 | Replogle K562 genome-wide | 61.3 GB | Arc states it covers most challenge target genes. Look for a published pseudobulk version first **[verify]**; it would save the download. |
| 5 | Jiang six lines; X-Atlas (HCT116, HEK293T) | varies | Only if overlap with the 300 targets justifies it |

For every (cell line, target gene) pair, compute and store:

- pseudobulk mean of perturbed cells and of matched controls, in counts per million;
- log fold change per gene, with a pseudocount;
- number of cells, and a standard error or a simple signal-to-noise score;
- achieved knockdown of the target.

Process files in backed mode, one perturbation at a time, and save the result as a small array (cell lines x targets x genes) with a metadata table. That atlas fits in memory and is the asset you keep for 2027.

Harmonise genes to the 18,533 symbols. Sources use Ensembl IDs and older symbols; build one mapping table, log what is unmapped, and treat missing genes as "no information", not as zero effect.

### 5.5 Model versions (ablation ladder)

Each version is one row in the results table. Do not skip rungs; the table is the portfolio piece.

| Version | Change | Tests |
|---|---|---|
| v0 | Resampled controls | Pipeline; expect about -0.30 |
| v1 | Add the mean effect over all perturbations and lines to every perturbation | Whether you can reach about 0 without any perturbation-specific signal |
| v2 | Per-target effect, averaged across source lines | Core transfer |
| v3 | Weight source lines by reliability (cell count, signal-to-noise) | Noise handling |
| v4 | Weight source lines by similarity to the context's controls | Context awareness |
| v5 | Fallback for uncovered targets: ridge regression from gene features to effect vectors | Coverage of the hidden test list |
| v6 | Effect-size calibration: a global or per-gene scale factor tuned on the local benchmark | The public benchmark shows this alone moves the score by 0.25 |
| v7 | One foundation-model embedding added to v5 | Whether it helps at all |

The test list of 300 genes is unknown until October 22, so v5 is not optional. Report coverage (share of targets with a measured effect in at least one source line) for validation and test.

### 5.6 Turning effects into cells (6 to 8 hours)

This step decides four of the six metrics and is easy to get wrong.

1. Start from real control cells of the same context, sampled with replacement. This keeps realistic library sizes, sparsity, and gene-gene correlation.
2. Apply the predicted fold change per gene on the count scale:
   - for decreases, thin counts binomially with keep-probability equal to the fold change;
   - for increases, add Poisson or negative-binomial counts with the missing mean.
3. Keep integers, no negatives, totals under 1,000,000 per cell, and stay under the sparsity cap. Increases on unexpressed genes add stored entries; monitor the count.
4. Decide what to do with small effects. Applying noisy tiny fold changes to all 18,533 genes produces false significant genes and hurts the Jaccard metric. Thresholding by signal-to-noise or keeping the top-k genes per perturbation is a tunable choice. The public benchmark variant with "scattered sparse rises" scored better than the plain one, which suggests the handling of sparse genes matters.
5. Consider per-cell heterogeneity: real knockdown is incomplete and varies by cell. A mixture of responding and non-responding cells is a later experiment, not a first version.

### 5.7 Local benchmark (6 to 8 hours, start in week 1)

The final round is blind, so this is the most important engineering piece.

- **Primary: leave-one-cell-line-out.** Hold out one source line entirely, predict it from its controls only, and score with `cell-eval2 --preset vcc2026`. Rotate over H1, K562, RPE1, HepG2, Jurkat. This mirrors the real task.
- **Secondary:** the public `vcc2026-h1-benchmark` repo, which packages H1 scoring and reports that its ranking of methods matched the leaderboard for four submissions.
- **Tertiary:** the live leaderboard, for a sanity check only. Do not tune to it; the final uses other cell lines, other genes, and more metrics.

Rules that keep the benchmark honest:

- The held-out line contributes nothing to the atlas, the weights, the scale factors, or the feature regression.
- Tune hyperparameters across several held-out lines, never one.
- Score 50 to 100 perturbations for quick iteration and the full set before any decision. Wilcoxon scoring on a 4-core CPU is slow.
- Record every run: config, code commit, `cell-eval2` version, all six metrics, and the average.
- Expect absolute values to differ from the leaderboard. Trust the ordering of methods, not the numbers.

---

## 6. Schedule

Assumes 10 to 12 hours a week. If you have less, cut from the bottom of each week, never the submission steps.

### Week 1 (Oct 8 to 14): plumbing and understanding

- [ ] Git, environment, project layout (5.1)
- [ ] Read the metric specification and the 2026 challenge paper
- [ ] EDA of the three control files (5.2)
- [ ] v0 submission end to end, confirm about -0.30 (5.3)
- [ ] Start the H1 download; set up the H1 benchmark

Exit criterion: a valid submission is on the leaderboard and you can score a prediction locally.

### Week 2 (Oct 15 to 21): the core method

- [ ] Download Replogle essential, RPE1, HepG2, Jurkat; check target overlap
- [ ] Build the effect atlas (5.4)
- [ ] v1 and v2, count generator (5.6)
- [ ] Leave-one-cell-line-out benchmark running (5.7)
- [ ] Submit v2 to the validation leaderboard
- [ ] Read Replogle 2022 methods and the linear-baseline benchmark paper

Exit criterion: a positive score locally and on the leaderboard.

### Week 3 (Oct 22 to 28): test data arrives

- [ ] **Oct 22: download D/E/F and submit the current best pipeline the same day as a safety entry** **[verify that multiple final submissions are allowed and which one counts]**
- [ ] Report coverage of the new 300 targets
- [ ] v3, v4, v5
- [ ] Decide on the K562 genome-wide download based on test coverage

Exit criterion: a final-round submission exists and every test target has a prediction.

### Week 4 (Oct 29 to Nov 5): calibrate, freeze, write

- [ ] v6 calibration; v7 if time allows
- [ ] **Nov 2: code freeze.** Only choose between existing versions after this.
- [ ] **Nov 3: final submission**, two days early. Packaging and upload are multi-gigabyte and can fail.
- [ ] Nov 4 to 5: buffer and write-up draft

### After the deadline (November to December)

- [ ] Clean the repo, write the README with the ablation table
- [ ] Publish a write-up
- [ ] When ground truth or winning solutions are released, score your versions against them and add a section on what you missed
- [ ] Start the 2027 list in section 8

---

## 7. Portfolio

What makes this credible to a reader who evaluates ML or computational biology candidates:

1. **A public repository** that reproduces the submission from raw downloads with documented commands.
2. **The ablation table** from 5.5, with all six metrics per version and per held-out cell line.
3. **A plot of local benchmark against leaderboard scores**, showing whether your validation predicted reality.
4. **An honest limits section**: coverage, what did not help, what you could not test on 8 GB.
5. **A write-up of 1,500 to 2,500 words**: the task, why simple transfer is strong, what context weighting added, what you would do with more compute.
6. **Final rank and score stated plainly**, with the start date. "Joined in week 7 of 11, solo, part-time, finished in the top X%" is a good line.

Keep a dated lab notebook in `docs/notes/` from today. The write-up is then mostly editing.

Before publishing code or data derivatives, check the challenge rules on sharing solutions during the competition and the licences of each source dataset **[verify]**.

---

## 8. Preparing for 2027

In priority order:

1. **Keep the assets current.** Effect atlas, benchmark, count generator. Add new public Perturb-seq data as it appears; Arc may release the 2026 six-line data after the challenge, which would be the best training set available.
2. **Study the 2026 winners** as soon as methods are published and reimplement the smallest winning idea on your benchmark.
3. **Learn one deep approach properly** on rented compute: fine-tune STATE on the atlas and compare it with your linear model under leave-one-cell-line-out.
4. **Learn the generative side**: flow matching and optimal transport for cell populations, starting from the preprint Arc recommends.
5. **Fill biology gaps**: CRISPRi mechanics, Perturb-seq QC, differential expression statistics, why effects differ by cell type.
6. **Start on day one next year.** Seven extra weeks of leaderboard feedback is the largest single advantage you gave up this time.

---

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `cell-eval2` or `vcc` does not install on Windows | Medium | Move to WSL2 on day one if the first attempt fails |
| Disk fills during packaging | Medium | Keep 80 GB free; delete raw downloads once the atlas is built |
| Local scoring too slow on 4 cores | High | GPU DE backend; score subsets; cache DE on the reference side |
| Many test targets absent from source data | Medium | v5 fallback built before Oct 22 |
| Overfitting to the validation leaderboard | Medium | Tune on leave-one-cell-line-out only |
| Gene symbol mismatches silently zero out effects | High | Mapping table with logged coverage; unit test on known genes |
| Upload fails near the deadline | Medium | Final submission by Nov 3 |
| Scope creep into foundation models | High | Nothing from path D or E before v5 is submitted |

---

## 10. Open questions to resolve this week

1. Submission limits per day, and whether the final round accepts more than one submission.
2. Which metrics the live leaderboard uses.
3. The exact definition of the zero baseline ("context mean") in the metric specification.
4. Whether the `vcc` CLI runs on native Windows.
5. Whether pseudobulk versions of the Replogle datasets are published.
6. Rules on sharing code before the deadline and on identifying the anonymised cell lines.

---

## Sources

- [The 2026 Virtual Cell Challenge announcement, Arc Institute](https://arcinstitute.org/news/virtual-cell-challenge-2026)
- [Generating high-quality Perturb-seq data for the 2026 challenge, Arc Institute](https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge-2026)
- [Virtual Cell Challenge 2026 paper, Cell](https://www.cell.com/cell/fulltext/S0092-8674(26)00931-1)
- [Evaluation page](https://virtualcellchallenge.org/evaluation) and [leaderboard](https://virtualcellchallenge.org/leaderboard)
- [cell-eval2 repository](https://github.com/ArcInstitute/cell-eval2) and [changelog](https://github.com/ArcInstitute/cell-eval2/blob/main/CHANGELOG.md)
- [vcc2026-h1-benchmark, a public local benchmark](https://github.com/forrestsheldon/vcc2026-h1-benchmark)
- [AtlasShift, a solo atlas-transfer entry at rank 82](https://github.com/kaipengm2/Virtual-Cell-Challenge-2026)
- [Virtual Cell Challenge 2025 wrap-up, Arc Institute](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up)
- [Arc Virtual Cell Atlas, challenge data README](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/virtual-cell-challenge/README.md)
- the organisers' dataset format description (not redistributed in this repository)
