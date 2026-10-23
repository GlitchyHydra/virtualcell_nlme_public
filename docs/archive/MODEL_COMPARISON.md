# Model comparison and decision

Written 2026-10-08. Companion to `APPROACH.md` (plan and schedule). This file records which model families exist, what datasets and metrics their authors used, and what we chose.

Evidence labels:

- **Confirmed**: stated by a source I retrieved.
- **Partial**: the source gave some of it; the rest needs the paper.
- **Not found**: my searches did not return it. Do not treat blanks as zero.

Most of the evidence below comes from search summaries and secondary pages, not the full papers. Before quoting any number in a write-up, check it against the paper (links at the end).

---

## 1. Decision

| Layer | Chosen | Why |
|---|---|---|
| Effect prediction | Effect-atlas transfer from public Perturb-seq, with hierarchical shrinkage (empirical Bayes, later NumPyro/PyMC) on pseudobulk log-fold-changes | CPU only; works per dataset against its own controls, which removes most batch and chemistry mismatch; independent benchmarks find simple baselines hard to beat |
| Uncovered genes | Ridge / low-rank regression from gene features (GenePT, ESM-2, DepMap, STRING) | The final target list is unknown until 2026-10-22 |
| Context weighting | Weight source lines by similarity of control profiles | The scientific question of the 2026 task |
| Cell generation | Resample the target context's own control cells, apply effects on the count scale | Preserves the variation we cannot get from a regressor |
| Foundation-model embeddings | Frozen features only, one at a time, as ablation rows | Evidence for fine-tuning gains is weak (section 3) |
| Deferred to after 2026-11-05 | STATE, flow matching (CellFlow-style), VAEs (scVI, SAMS-VAE, CPA) | Need a rented GPU or more time |
| Skipped | Evo 2 | DNA sequence model; wrong signal for this task |

Success is judged by a leave-one-cell-line-out benchmark scored with `cell-eval2 --preset vcc2026`, not by the validation leaderboard.

---

## 2. Model families

| Family | Examples | Idea | Fits 8 GB GPU | Role in our plan |
|---|---|---|---|---|
| Effect transfer / linear | atlas averaging, ridge, low-rank | Predict effect vectors from gene features and other lines | Yes | **Main** |
| Hierarchical Bayesian regression | NumPyro, PyMC | Partial pooling across lines; explicit uncertainty | Yes (pseudobulk) | **Main** (shrinkage step) |
| Latent-variable VAEs | scGen, CPA, SAMS-VAE, scVI | Perturbation shifts a latent state; count likelihood | Small data yes | Later ablation |
| Graph-based | GEARS | Gene-gene knowledge graph for unseen perturbations | Yes | Optional comparison |
| Single-cell foundation models | scGPT, scFoundation, Geneformer, UCE | Pretrain on millions of cells | Inference only | Embeddings as features |
| Perturbation-specific deep models | STATE (Arc) | Transformer over sets of cells, across contexts | Training no | After deadline |
| Generative transport | CellFlow (theislab), flow matching, optimal transport | Learn control-to-perturbed distribution | No | After deadline; read Arc's recommended preprint now |
| DNA language model | Evo 2 | Sequence modelling | No | Skip |

---

### 2.1 What each model is

Plain-language descriptions from general knowledge of the methods, not from the searches in section 3. "Effect vector" means the change in every gene's expression caused by one knockdown, usually a log fold-change against controls.

**Effect transfer (atlas averaging)**
- *Idea:* measure each target gene's effect vector in several public cell lines, then average them (weighted) to predict the effect in a new line.
- *Input to output:* gene name plus its measured effects elsewhere gives a predicted effect vector.
- *Strength:* no training loop, fast, hard to overfit, easy to explain.
- *Weakness:* only works for genes measured somewhere; assumes effects are mostly shared across cell lines.
- *Here:* the core of the plan.

**Ridge / low-rank linear regression on gene features**
- *Idea:* learn a linear map from a numeric description of the target gene (embedding, network neighbours, co-essentiality) to its effect vector. Ridge adds a penalty so the fit does not chase noise. Low-rank forces effects to be built from a few shared patterns, which also denoises.
- *Strength:* works for genes with no measurement; cheap; the standard baseline in published comparisons.
- *Weakness:* effects of a gene are not linear in its features, so it misses sharp, gene-specific responses.
- *Here:* fallback for uncovered test genes.

**Hierarchical Bayesian model (partial pooling) and empirical Bayes shrinkage**
- *Idea:* each (gene, cell line) effect is treated as a noisy measurement of a gene-level effect shared across lines, plus a line-specific deviation. The model estimates both and pulls noisy estimates toward the shared value, more strongly when the measurement is noisy.
- *Empirical Bayes* sets that pull strength from the data itself with simple formulas. *Full Bayesian* (NumPyro, PyMC) samples a posterior and gives uncertainty.
- *Strength:* principled combination of unequal-quality sources; honest uncertainty.
- *Weakness:* more code; on 18,533 genes you need low-rank structure or a gene subset to stay fast.
- *Here:* the weighting step (v3 in the plan).

**scGen**
- *Idea:* a variational autoencoder (VAE) compresses each cell into a small latent code. The perturbation is the average shift in latent space between control and perturbed cells, applied to unperturbed cells of another type by vector arithmetic.
- *Strength:* simple, early, easy to understand.
- *Weakness:* one shift per condition, so it handles one perturbation at a time and does not generalise to unseen ones.

**CPA (Compositional Perturbation Autoencoder)**
- *Idea:* a VAE-like model that splits a cell into separate parts: cell type, perturbation, dose, other covariates. An adversarial loss stops the cell-type part from carrying perturbation information, so parts can be recombined (new perturbation in an old cell type).
- *Strength:* handles combinations and doses.
- *Weakness:* needs the perturbation to be seen in training; the original design is for chemicals and known combinations.

**SAMS-VAE (Sparse Additive Mechanism Shift VAE)**
- *Idea:* the latent state of a perturbed cell is the control state plus a sparse perturbation offset; each perturbation switches on a few latent dimensions, and combinations add up.
- *Strength:* interpretable; designed for combinations and for testing with posterior predictive checks.
- *Weakness:* still needs training data for each perturbation.

**scVI (family)**
- *Idea:* a VAE whose output is a negative binomial count distribution, with library size and batch as explicit inputs. It is a probabilistic model of raw counts.
- *Strength:* the right likelihood for sparse counts; models batch; usable as a noise model.
- *Weakness:* not a perturbation predictor by itself.
- *Here:* possible later as the cell generator instead of control resampling.

**GEARS**
- *Idea:* each gene is a node in a graph built from gene-gene relationships (co-expression and Gene Ontology). A graph neural network propagates information so an unseen perturbation can borrow from its neighbours. Separate embeddings are combined to predict the post-perturbation expression.
- *Strength:* can generalise to genes never perturbed, including multi-gene combinations.
- *Weakness:* trained within a single cell line; not built for transfer to a new cell context.

**Single-cell foundation models (scGPT, scFoundation, Geneformer, UCE)**
- *Idea:* a transformer pretrained on millions of unperturbed cells, usually by masking genes and predicting them. The result is a learned embedding for each gene and each cell. For perturbation prediction, either fine-tune it with a perturbation token or feed its embeddings into another model (such as GEARS or a linear head).
- *Differences:* scGPT and scFoundation read expression values; Geneformer ranks genes within a cell and reads the ranks; UCE (Universal Cell Embeddings) targets embeddings that work across species.
- *Strength:* broad prior knowledge of gene relationships.
- *Weakness:* expensive; independent tests found little gain over simple baselines for perturbation prediction.
- *Here:* frozen embeddings as features only.

**STATE (Arc Institute)**
- *Idea:* two parts. State Embedding turns each cell into a vector; State Transition is a transformer that takes a set of control cells plus a perturbation and predicts the perturbed set, so it reasons about populations, not single cells. It is trained on large observational and perturbation collections.
- *Strength:* designed for the same cross-context question as this challenge.
- *Weakness:* heavy to train; non-commercial weights.
- *Here:* try after the deadline on rented compute.

**CellFlow, flow matching and optimal transport**
- *Idea:* learn a smooth transformation that moves a distribution of control cells into the perturbed distribution. Flow matching trains a network to give the direction and speed of movement for each point; optimal transport pairs cells so the total movement is minimal.
- *Strength:* models the whole distribution, so it keeps cell-to-cell variation.
- *Weakness:* needs paired training for each condition; training is costly and tuning is delicate.
- *Here:* what Arc recommends reading; a 2027 target.

**Evo 2**
- *Idea:* a language model trained on DNA sequence from many organisms that predicts the next base; it can score how disruptive a sequence change is.
- *Weakness for this task:* it sees sequence, not the cell's regulatory state or the downstream response.
- *Here:* skipped.

**Gene feature sources (used as inputs, not as predictors)**
- *ESM-2:* protein language model; gives a vector per protein from its amino-acid sequence.
- *GenePT:* embeddings of the text descriptions of genes from a language model.
- *DepMap co-essentiality:* how similarly two genes affect cell survival across hundreds of cancer lines; genes in the same pathway tend to match.
- *STRING:* a protein-interaction network; neighbours of the target are likely to respond together.

---

## 3. Datasets and metrics each model reported

### 3.1 Linear baselines vs deep models (Ahlmann-Eltze, Huber, Anders; Nature Methods 2025)

| | |
|---|---|
| Claim | Five foundation models and two other deep models did not outperform deliberately simple baselines on single and double perturbation prediction (Confirmed) |
| Datasets | Norman (K562; described elsewhere as 131 two-gene and 105 single-gene perturbations) and Replogle K562 genome-wide CRISPRi. Partial: the full list needs the paper. |
| Metrics | Pearson delta (correlation after subtracting control); prediction error across readout genes; gene rankings by expression and by DE. Partial |
| Caveats | A later talk argues systematic perturbation-vs-control differences inflate Pearson delta. A rival paper (Miller et al.) argues deep models do beat uninformative baselines on better-calibrated metrics. Metric choice is disputed. |

### 3.2 GEARS (Roohani, Huang, Leskovec; Nature Biotechnology)

| | |
|---|---|
| Idea | Deep model plus gene-gene knowledge graph; aims at combinations containing never-perturbed genes (Confirmed) |
| Datasets | Norman, Adamson, Dixit (named by the code package); Replogle RPE1 and K562 essential added later as dataloaders (Confirmed) |
| Reported result | Journal version: 40% higher precision than existing approaches and twice as good at identifying the strongest interactions, on four genetic-interaction subtypes. The preprint said five subtypes. (Confirmed, version discrepancy) |
| Metrics | Not found |

### 3.3 scGPT (Cui et al.; Nature Methods 2024)

| | |
|---|---|
| Pretraining | Over 33 million cells (Confirmed) |
| Perturbation datasets | Adamson (68,603 cells, CRISPRi), Norman (91,205 cells, single or dual CRISPRa), Replogle K562 (162,751) and RPE1 (162,733) subsets (Confirmed, from a secondary description) |
| Metrics | Pearson correlation on differential expression (PCC delta) is the main one; global MSE ranks models differently (Confirmed) |
| Independent results | A BMC Genomics study (2025) found even the training-mean baseline outperformed scGPT and scFoundation, and that current Perturb-seq benchmarks have low perturbation-specific variance. Ahlmann-Eltze et al. found no gain over baselines. (Confirmed) |

### 3.4 scFoundation and Geneformer

| | |
|---|---|
| scFoundation | Used mainly as a source of gene embeddings fed into GEARS. One study reports baseline PCC delta about 0.381 for scGPT and 0.326 for scFoundation, with scFoundation mostly frozen for GPU memory reasons, so not like for like. (Confirmed) |
| Geneformer | Not built for perturbation prediction; used with a linear decoder, or as a comparator. No clean head-to-head numbers found. |
| Pretraining corpora, datasets in the original papers | Not found |

### 3.5 CPA and scGen

| | |
|---|---|
| CPA | Factorised embeddings for cell type, perturbation and covariates, kept independent by an adversarial loss; extends scGen to combinations (Confirmed) |
| Datasets in the CPA paper | Six public datasets plus a new non-small cell lung cancer dataset; 5,329 missing genetic combinations imputed in a Perturb-seq experiment. Which public sets: Not confirmed. |
| Third-party sci-Plex numbers (PyTDC benchmark, R-squared, sciPlex2 drugs) | scGen seen: 0.903 BMS, 0.944 Dex, 0.891 Nutlin, 0.948 SAHA. ChemCPA seen/unseen: BMS 0.943/0.906, Dex 0.882/0.540, Nutlin 0.925/0.835, SAHA 0.825/0.690. Chemical perturbations, and ChemCPA is a variant, not original CPA. Not comparable to CRISPRi results. |

### 3.6 SAMS-VAE (Bereket and Karaletsos; NeurIPS 2023)

| | |
|---|---|
| Idea | Latent state = basal part + sparse additive perturbation part (Confirmed) |
| Evaluation | Average-treatment-effect framework linked to posterior predictive checks; generalisation in and out of distribution, including a combinatorial task (Confirmed) |
| Datasets | Two single-cell datasets; names Not found |

### 3.7 STATE (Arc Institute, 2025)

| | |
|---|---|
| Architecture | State Embedding (SE) model plus State Transition (ST) transformer over sets of cells (Confirmed) |
| Training data | Arc reports observational data from 167 million cells and perturbational data from over 100 million cells across 70 contexts; a secondary list names Tahoe-100M, Parse-PBMC, Replogle-Nadig and the Arc Virtual Cell Atlas (Partial; verify against the preprint, bioRxiv 2025.06.26.661135) |
| Reported result | On Tahoe-100M, 50% better perturbation discrimination and twice the accuracy at identifying true DE genes versus prior models (press account, not the paper's table) |
| Cross-context transfer, Replogle (independent study) | Every method loses 0.17 to 0.25 pΔ moving from within-context to cross-context. State retains 69% (0.55 to 0.38), scGPT 67% (0.46 to 0.31), a linear baseline 49% (0.49 to 0.24). State is best but about 100,000 times more expensive; scGPT is dominated. |
| License | Weights are non-commercial |

### 3.8 CellFlow and flow matching

| | |
|---|---|
| CellFlow (theislab, bioRxiv 2025) | Flow matching learns a conditional vector field from a control distribution to the perturbed one; conditions include drugs, genetic modifications, cytokines, morphogens; applied to cytokine stimulation, drug treatments and gene knockouts (Confirmed). Not to be confused with the image-based CellFlux/CellFlow (arXiv 2502.09775). |
| Metrics used in expression benchmarks | PerturBench: cosine similarity of log fold change, RMSE, MMD in expression or PCA space, DEG recall. A 2026 arXiv table on Norman: L2, MSE, MAE, DE-Spearman, Pearson delta. |
| Indicative result | In that table CellFlow scored below some competitors, for example DE-Spearman 0.2860 versus GEARS 0.3569 and scDFM 0.6957 on single perturbations (not verified against the paper) |
| Relevance | Arc recommends a 2025 winning team's flow-matching preprint before building this year |

### 3.9 Challenge-specific entries (the closest evidence to our task)

| Entry | Data | Result |
|---|---|---|
| 2025 winners | H1 hESC; BioMap xTrimoSCPerturb (improved scFoundation), XLearning Lab, Outlier (TransPert); Altos Labs generalist prize with flow matching | Hybrids of deep learning and classical statistics won. Arc: models did not consistently beat naive baselines; nearly all were worse than baseline on MAE. |
| AtlasShift, solo 2026 entry | K562 GWPS, HCT116, HEK293T, H1 (2025), CD4 DE statistics; weighted ensemble of source effects, target-control adjustment, promoter-neighbour correction | 0.1546, rank 82 on 2026-09-10 (self-reported) |
| `vcc2026-h1-benchmark` transfer baselines | H1 only | Validation leaderboard: control resampling -0.304, transfer at full effect 0.083, transfer at 0.13x effect -0.183, transfer plus scattered sparse rises 0.098 |

---

## 4. Metrics inventory

### 4.1 Our scoring (`cell-eval2`, `vcc2026` preset)

Each is rescaled so 0 is the mean-response baseline and 1 is a replicate of the real measurement.

| Metric id | Measures |
|---|---|
| `pds_cosine` | Perturbation discrimination: is a prediction closer to its own measured effect than to other perturbations' |
| `expr_mse_unbiased_capped_norm` | Expression error |
| `de_wilcoxon_direction_fidelity_yield_raw` | Correctness of predicted change directions |
| `de_wilcoxon_direction_reach_raw` | How far down the ranking directions stay right |
| `de_wilcoxon_sig_jaccard` | Overlap of significant gene sets |
| `de_wilcoxon_lfc_nmae` | Fold-change accuracy |

How the six are aggregated into one score is not on the pages I read; check `docs/vcc2026_metrics/` in the repo.

### 4.2 Metrics used elsewhere

| Metric | Used by | Caution |
|---|---|---|
| Pearson delta (PCC delta) | scGPT, GEARS-style benchmarks, Ahlmann-Eltze | Can be inflated by systematic perturbation-vs-control differences |
| R-squared | PyTDC sci-Plex benchmark | Rewards getting the mean right; weak for perturbation-specific signal |
| MSE, L2, MAE | Many | Dominated by highly variable genes; rankings change with metric |
| DE-Spearman, DEG recall | PerturBench, 2026 Norman table | Needs ground truth DE; matches our DE metrics in spirit |
| MMD, FID | Distributional and image benchmarks | Checks population shape, closest to our need for realistic cells |

The 2025 challenge showed that a metric can stop influencing optimisation (MAE) and that scores can be gamed by transformations. The 2026 reference-scaled, six-metric design responds to that, so tune against all six, not one.

---

## 5. What this means for the choice

1. Independent benchmarks repeatedly find simple baselines competitive on within-context tasks. Cross-context is where deep models retain more (State 0.38 against linear 0.24 in one study), but at about 100,000 times the cost. That justifies starting simple and revisiting STATE as a rented-GPU experiment later.
2. Datasets in the literature overlap heavily: Norman, Adamson, Replogle K562/RPE1, Dixit. Our source atlas should favour Replogle, Nadig, the 2025 H1 data, and any lines that overlap the 300 targets.
3. Most published metrics (Pearson delta, R-squared) differ from our six. Do not expect published scores to transfer; compare methods on our own benchmark.
4. Metric choice is disputed, so report several metrics in the write-up, not only the average.

---

## 6. Gaps to fill

- Full dataset list and metric definitions from Ahlmann-Eltze et al. (PMC12328236)
- GEARS metrics (Methods of the Nature Biotechnology paper)
- Datasets used by SAMS-VAE, scGen, original CPA, Geneformer, scFoundation
- STATE per-dataset numbers on Replogle-Nadig and Parse-PBMC (preprint tables)
- Aggregation rule for the six 2026 metrics
- Which models, if any, publish zero-shot cross-line numbers on CRISPRi data comparable to our benchmark

---

## Sources

- [Ahlmann-Eltze, Huber, Anders, Nature Methods 2025 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/)
- [Same paper, bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.09.16.613342.full.pdf)
- [GEARS, Nature Biotechnology](https://link.springer.com/10.1038/s41587-023-01905-6)
- [GEARS code package](https://pypi.org/project/cell-gears)
- [scGPT preprint](https://www.biorxiv.org/content/10.1101/2023.04.30.538439.full.pdf)
- [BMC Genomics 2025 benchmark of scGPT and scFoundation](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-025-11600-2)
- [CPA, Molecular Systems Biology](https://link.springer.com/article/10.15252/msb.202211517)
- [PyTDC benchmark](https://arxiv.org/pdf/2505.05577)
- [SAMS-VAE, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0001ca33ba34ce0351e4612b744b3936-Abstract.html)
- [CellFlow, bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.04.11.648220.full.pdf)
- [Arc State announcement](https://arcinstitute.org/news/virtual-cell-model-state)
- [Virtual Cell Challenge 2025 wrap-up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up)
- [cell-eval2](https://github.com/ArcInstitute/cell-eval2)
- [vcc2026-h1-benchmark](https://github.com/forrestsheldon/vcc2026-h1-benchmark)
- [AtlasShift entry](https://github.com/kaipengm2/Virtual-Cell-Challenge-2026)
