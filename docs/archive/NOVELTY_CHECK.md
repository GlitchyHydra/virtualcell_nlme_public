# Novelty check: two-stage random-effects pooling with adaptive shrinkage for knockdown transfer

Done 2026-10-09. Question: has anyone combined (1) random-effects pooling of perturbation effects across cell lines, (2) a sparse empirical-Bayes prior with local false sign rate (lfsr), and (3) prediction for an unseen cell line, for Perturb-seq knockdown data?

**Result: no single source found that combines all three.** The components exist separately. This is a limited check (two extended web searches, results read as summaries, no full texts), so it supports "not found", not "does not exist". Do not claim novelty in a write-up before the follow-up searches in section 5.

Evidence labels: **Confirmed** = stated in a retrieved source summary; **Not checked** = I did not open the paper's methods.

---

## 1. Searches run

| # | Query | Result |
|---|---|---|
| 1 | random-effects meta-analysis across cell lines Perturb-seq perturbation effect transfer empirical Bayes shrinkage unseen cell type prediction | Meta-analysis and empirical Bayes tools found; no Perturb-seq transfer method using them |
| 2 | adaptive shrinkage ashr local false sign rate CRISPR Perturb-seq differential expression hierarchical Bayesian cross-context transfer of knockdown effects | ashr and Perturb-seq DE found separately; no combined method |

---

## 2. Component 1: random-effects meta-analysis with prediction for a new study

| Source | What it offers | Status |
|---|---|---|
| [bayesmeta vignette](https://cran.r-project.org/web/packages/bayesmeta/vignettes/bayesmeta.html) | Posterior for the effect and heterogeneity (tau); predictive distributions, shrinkage estimates | Confirmed |
| [bayesmeta package manual](https://cran.r-project.org/web/packages/bayesmeta/bayesmeta.pdf) | Same package, reference | Not checked |
| [Using the bayesmeta R package for Bayesian random-effects meta-analysis (arXiv)](https://arxiv.org/pdf/2209.06004) | Tutorial paper | Not checked |
| [Bayesian random-effects meta-analysis using the ... (arXiv)](https://arxiv.org/pdf/1711.08683) | Methods paper | Not checked |
| [Study-specific prediction intervals for random-effects meta-analysis: a tutorial](https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1490) | Recommends empirical Bayes estimates (BLUPs) with study-specific prediction intervals; closest match to "estimate plus interval for one context" | Confirmed |

Take-away: the statistical machinery for pooling and for predicting a new study exists and is mature. It is from meta-analysis, not Perturb-seq.

---

## 3. Component 2: adaptive shrinkage and lfsr

| Source | What it offers | Status |
|---|---|---|
| [ashr: ash function documentation](https://rdrr.io/cran/ashr/man/ash.html) | Adaptive shrinkage using empirical Bayes; unimodal prior with mode at zero; posterior probabilities of positive and negative effects; local false sign rate | Confirmed |
| [ashr adaptive shrinkage vignette](https://rdrr.io/cran/ashr/f/vignettes/adaptive_shrinkage.Rmd) | Worked usage | Not checked |
| [DESeq2 lfcShrink](https://rdrr.io/bioc/DESeq2/man/lfcShrink.html) | Offers ashr as one shrinkage option, next to apeglm and a normal prior | Confirmed |
| [Heavy-tailed prior distributions for sequence count data (apeglm)](https://academic.oup.com/bioinformatics/article/35/12/2084/5159452) | Alternative shrinkage; s-values for "false sign or small" | Confirmed |
| [Empirical Bayes shrinkage and FDR estimation allowing for unwanted variation (arXiv)](https://arxiv.org/pdf/1709.10066) | Related empirical Bayes work | Not checked |
| [Empirical Bayes shrinkage of functional effects, dynamic eQTLs (arXiv)](https://arxiv.org/pdf/2602.03077) | Shrinkage applied to effects measured across a condition axis | Not checked |
| [ShrinkCRISPR: differential fitness analysis of CRISPR-Cas9 screens](https://pmc.ncbi.nlm.nih.gov/articles/PMC9896759) | Shrinkage in bulk CRISPR screens | Not checked |
| [Causal gene regulatory network inference from Perturb-seq (bioRxiv 2026)](https://www.biorxiv.org/content/biorxiv/early/2026/02/19/2026.02.18.706642.full.pdf) | Uses adaptive shrinkage; reports that it attenuates the trend of estimated out-degree decreasing with stronger CRISPRi effect | Confirmed (from summary) |

Take-away: adaptive shrinkage is established in expression analysis and has been used on Perturb-seq data for network inference. Not found: use for transferring knockdown effects to a new cell line.

---

## 4. Component 3: cross-context transfer of perturbation effects

| Source | What it offers | Status |
|---|---|---|
| [PerturbMap: cross-context transfer of single-cell perturbation responses](https://arxiv.org/html/2607.28090v1) | Handles perturbations measured in source contexts but missing in the recipient; authors warn that copying or weakly calibrating responses across contexts risks transferring the wrong signal | Confirmed (from summary) |
| [C3TL: causal cellular context transfer learning](https://arxiv.org/pdf/2603.13051) | Prediction of unseen perturbation effects across contexts | Not checked |
| [Mechanisms Matter: transportability of cellular perturbation effects (bioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.05.08.723625v1.full) | Transportability framing | Not checked |
| [CFM-GP: conditional flow matching across cell types](https://arxiv.org/pdf/2508.08312) | Learned generative transfer across cell types | Not checked |
| [CRISP: unseen cell types with foundation models](https://www.nature.com/articles/s43588-025-00887-6) | Drug responses in unseen cell types via transfer learning | Not checked |
| [PerturbNet](https://www.biorxiv.org/content/10.1101/2022.07.20.500854.full.pdf) | Deep generative model; cell type as covariate | Not checked |
| [A systematic comparison of single-cell perturbation response prediction models (Science Advances)](https://www.science.org/doi/10.1126/sciadv.aed3414) | Benchmark; treats cross-cell-type transfer as a separate task | Not checked |
| [Genome-scale Perturb-seq in primary human CD4+ T cells](https://www.sciencedirect.com/science/article/pii/S0092867426009293) | Tests whether effects for a gene replicate across datasets | Confirmed (from summary) |
| [Single-cell DE analysis between ... (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.08.01.606200v1.full.pdf) | Appeared in search 1; relevance not assessed | Not checked |

A 2026 preprint cited in the summaries reports a cross-context generalisation gap: performance under cross-context splits drops substantially, often to simple-baseline levels. Source not identified in the results.

Take-away: cross-context transfer is active, mostly with learned models. Not found: a classical hierarchical meta-analytic route.

---

## 5. Gaps and follow-up searches

Before claiming novelty in a write-up, search for:
1. `multi-context Perturb-seq meta-analysis` and `hierarchical Bayesian perturbation effect cell type borrowing`
2. `ashr` or `lfsr` together with `Perturb-seq` and `cell line`
3. `random effects` together with `single-cell CRISPR screen` and `heterogeneity across cell lines`
4. Nadig et al., "Transcriptome-wide analysis of differential expression in perturbation atlases" (Nature Genetics, 2025): appeared in a reference list; methods not checked, may use shrinkage or cross-line pooling
5. Forward citations of the 2024 to 2026 cross-context transfer papers above
6. The 2025 and 2026 Virtual Cell Challenge winner write-ups, for any pooling or shrinkage step
7. Google Scholar and bioRxiv full-text search, which the extended web search does not replace

Also read the methods of every "Not checked" entry that looks close (the dynamic eQTL shrinkage paper, C3TL, Mechanisms Matter, the Perturb-seq network preprint) to confirm what they actually do.

---

## 6. How this is used

- `MODEL_DESIGN.md` section 5 states novelty as a secondary, unverified motivation and points here.
- Whether the approach is good is decided by the leave-one-cell-line-out benchmark, not by novelty.
- If follow-up searches find prior work, cite it and reframe the contribution as an application to the 2026 challenge, which is still a valid portfolio piece.
