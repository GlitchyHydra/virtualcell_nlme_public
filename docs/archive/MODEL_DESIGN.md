# Model design (living document)

Started 2026-10-08, updated 2026-10-09. This is the place to expand the modelling plan. It records the framing decisions, the baseline model specification, the alternatives considered, and the open design questions. Scoring rules are in `SCORING.md`, the schedule in `APPROACH.md`, the model survey in `MODEL_COMPARISON.md`.

Status markers: **decided**, **proposed**, **open**.

---

## 1. Problem framing

**decided.** The input is a context's control cells plus a gene name. The output is 400 integer-count cells per gene, whose differential-expression profile against the controls matches the real knockdown.

The task does not have a separate "prediction of direction". Direction, magnitude and which genes respond are all properties of the generated cells and are measured only through the DE tests on them (`SCORING.md`).

### 1.1 Predict the change, not the absolute expression

**decided.** Model the effect (log-fold-change between perturbed and control populations), then add it back to the context's own controls to get absolute counts.

| | Change (delta) | Absolute |
|---|---|---|
| Baseline and batch | Cancel against each dataset's controls | Learned along with the signal |
| Transfer to a new line | Effect can carry over; baseline comes from the new controls | Baseline is wrong for the new line |
| Failure mode | Predicting no change | Predicting the average cell, which lowers error and teaches nothing |
| Fit to scoring | Metrics are scaled to the mean-response baseline and measure the effect | Measures the effect indirectly |

Two limits to keep in mind:
- The delta exists at population level only. Single-cell sequencing destroys the cell, so there is no per-cell before and after.
- Fold changes on low-count genes are noisy. Use a pseudocount, or a difference in normalised expression, and ignore genes below the scorer's 5 CPM filter (`SCORING.md`).

### 1.2 Three sub-problems

| Step | Question | Mainly judged by |
|---|---|---|
| A. Responder selection | Which genes change at all? | `sig_jaccard`, `direction_fidelity` |
| B. Direction and magnitude | Up or down, by how much, and how confident? | `direction_fidelity`, `direction_reach`, `lfc_nmae` |
| C. Cell generation | Turn the effect into 400 realistic count cells | All DE metrics, `pds_cosine`, `expr_mse` |

These can be developed and benchmarked separately even though they are scored as one.

---

## 2. Design constraints taken from the scoring

**decided.** Derived from `SCORING.md`; each is a requirement on the model.

1. **Calibrate the number of responders.** Yield and Jaccard punish both over- and under-calling.
2. **Rank genes by confidence.** Reach credits a long run of correctly signed genes in the order of your own confidence.
3. **Make predictions differ between perturbations.** PDS is a rank over all 300.
4. **Ignore lowly expressed genes.** Genes at or below 5 CPM in the controls are never tested. In our controls, 9,929 (A), 9,626 (B) and 10,124 (C) genes pass; 7,991 pass in all three and 11,961 in at least one.
5. **Do not shrink effects uniformly.** A public benchmark scored -0.183 at 0.13x effect versus 0.083 at 1x.
6. **Noise decides significance.** Wilcoxon runs on your 400 cells, so the cell-to-cell distribution is part of the model.
7. **Integers, sparse, within caps.** Raw counts, at most about 13,200 stored entries per cell on average.

---

## 3. Baseline model: two-stage empirical-Bayes pooling

**proposed.** The same logic as the standard two-stage approach in population PK, applied to knockdown effects. Stage 1 summarises each source dataset. Stage 2 pools across cell lines with a random-effects model, adds a sparse prior, and yields a prediction plus a confidence for a new line.

```
Source dataset d (cells with known target, plus non-targeting controls)
        |
   STAGE 1   per (line c, target g, response gene j):  effect estimate  beta_hat   +  its standard error  s
        |
   STAGE 2   across lines, for each (g, j):            pooled effect  theta_hat,  between-line variance  tau^2
        |
   SPARSE PRIOR (adaptive shrinkage)                   posterior mean, probability of null, sign error rate (lfsr)
        |
   PREDICTION for a new line (A, B, C):                effect + confidence   ->  responder set, direction, ranking
        |
   CELL GENERATOR                                      400 integer-count cells per target
```

Stage 1 can only run on source datasets, where perturbed cells exist. In contexts A, B and C there are no perturbed cells, so they enter only through their control cells (covariates and the generator).

### 3.1 Glossary

| Symbol or word | Meaning |
|---|---|
| Source dataset (d), source line (c) | A public Perturb-seq dataset and the cell line it measures. One line can appear in several datasets (K562 does), so dataset and line are distinct. |
| Target gene (g) | The gene silenced by the guide. |
| Response gene (j) | Any other gene whose expression is examined. The target gene itself is excluded, as in the scoring. |
| Perturbed group | All cells in dataset d that carry a guide against target g. This is the state **after** the alteration. n_p cells. |
| Control group | The non-targeting cells of the same dataset d. This is the reference state **before** the alteration. n_c cells. |
| CPM | Counts per million: a cell's counts divided by its total counts, times 1,000,000. Removes sequencing-depth differences between cells. |
| m_perturbed | The mean CPM of gene j over the perturbed group. |
| m_control | The mean CPM of gene j over the control group. |
| $\hat{\beta}$ | The Stage 1 effect estimate: $log2(m_{perturbed} / m_{control})$. |
| s | Standard error of $\hat{\beta}$. |
| $\theta$ | The true typical effect of target g on gene j across cell lines (Stage 2). |
| $\tau^2$ | Between-line variance of the true effect (section 3.3). |
| K | Number of source lines that measured the pair (g, j). |

On "before and after": the perturbed and control groups are different cells, not the same cells measured twice. A cell cannot be sequenced before and after, because sequencing destroys it. The comparison is between two populations that were grown in the same pool, so the control group stands in for what the perturbed cells would have looked like without the knockdown.

### 3.2 Stage 1: effect estimate and standard error

**Input**: the count matrix of one source dataset (cells by genes) and, for each cell, its guide label (target gene or `non-targeting`).

**Steps** for one (line, target g, response gene j):

```
1. Convert every cell's counts to CPM:           cpm[cell, gene] = counts / (cell total) * 1e6
2. Group means of gene j:                        m_perturbed = mean of cpm[., j] over the n_p cells carrying target g
                                                 m_control   = mean of cpm[., j] over the n_c non-targeting cells
3. Group variances of gene j across cells:       v_perturbed, v_control    (sample variances of cpm[., j])
4. Effect estimate:                              beta_hat = log2( m_perturbed / m_control )
5. Standard error (delta method):                s = sqrt( v_perturbed / (n_p * m_perturbed^2)
                                                         + v_control   / (n_c * m_control^2) ) / ln(2)
```

**How the standard error arises.**
- The mean of n cells has variance v / n.
- For a ratio, take logs. For a small relative error, the variance of log(m) is approximately (variance of m) / m^2 = v / (n m^2). This is the delta method.
- The two groups are independent, so the two variances add.
- Dividing by ln(2) converts from natural log to log2, so the result is in the same units as beta_hat.
- Interpretation: s is how much beta_hat would move if the experiment were repeated with different cells.

**Output row**: `dataset, line, target g, response gene j, beta_hat, s, n_p, n_c, m_control` (plus the knockdown of the target itself, as a quality flag).

The definition matches the scorer's effect size, `log2((mean_pert + 1e-9) / (mean_ctrl + 1e-9))` on cells normalised to a million counts (`SCORING.md`), so Stage 1 measures effects the way the scorer will.

#### Worked example (null experiment on our data)

To see real numbers with no true effect, 400 random control cells of context A (seed 0) were treated as the perturbed group and compared with the other 18,000 control cells. Any nonzero beta_hat here is noise.

Intermediate values for four genes:

| Gene | m_control | m_perturbed | v_control | v_perturbed | term_c = v_c/(n_c m_c^2) | term_p = v_p/(n_p m_p^2) |
|---|---|---|---|---|---|---|
| CGGBP1 | 397.13 | 400.25 | 31,149.1 | 32,828.7 | 0.00001 | 0.00051 |
| ADAT1 | 119.03 | 121.48 | 9,044.4 | 9,140.2 | 0.00004 | 0.00155 |
| IMMP1L | 43.11 | 46.00 | 2,911.3 | 3,555.9 | 0.00009 | 0.00420 |
| RASIP1 | 10.08 | 10.87 | 753.1 | 736.0 | 0.00041 | 0.01557 |

Results, with n_p = 400 and n_c = 18,000:

| Gene | Control CPM | Perturbed CPM | beta_hat | s |
|---|---|---|---|---|
| CGGBP1 | 397.1 | 400.2 | 0.011 | 0.033 |
| ADAT1 | 119.0 | 121.5 | 0.029 | 0.057 |
| IMMP1L | 43.1 | 46.0 | 0.093 | 0.094 |
| RASIP1 | 10.1 | 10.9 | 0.110 | 0.182 |

Hand calculation for RASIP1:

```
beta_hat = log2(10.87 / 10.08) = log2(1.0784) = 0.1095
term_p   = 736.0  / (400   * 10.87^2) = 736.0  / 47,264    = 0.01557
term_c   = 753.1  / (18000 * 10.08^2) = 753.1  / 1,828,800 = 0.00041
s        = sqrt(0.01557 + 0.00041) / ln(2) = 0.1264 / 0.6931 = 0.1824
```

Here beta_hat / s = 0.6, so the apparent 11% increase is well inside noise.

The same experiment repeated over 30 draws and all 9,929 testable genes of context A:

| Control mean CPM | Genes | Median s | Cells with zero count | Share of genes with abs(beta_hat / s) > 1.96 |
|---|---|---|---|---|
| 5 to 10 | 778 | 0.226 | 87% | 5.6% |
| 10 to 50 | 3,689 | 0.119 | 62% | 5.3% |
| 50 to 200 | 4,308 | 0.067 | 24% | 5.2% |
| 200 to 1000 | 1,120 | 0.041 | 4% | 5.5% |
| above 1000 | 34 | 0.030 | 0% | 5.2% |

What it shows:
- **s is calibrated in this null setting.** About 5% of null genes exceed 1.96 standard errors, which is what chance predicts, and the observed spread of beta_hat matches s.
- **Noise depends on abundance.** With 400 cells, a gene at 50 to 200 CPM needs an effect of about 0.13 (log2 units) to stand out; at 5 to 10 CPM about 0.44.
- **Sparse genes are fragile.** At 5 to 10 CPM, 87% of cells have a zero count, where the ratio of means is shaky.
- **Source datasets will have larger s.** s scales roughly with 1/sqrt(cells), and public datasets often have fewer cells per perturbation than 400.
- **Limits of the test.** It uses one context, exchangeable cells, and no guide or batch structure. It validates the formula, not the whole pipeline.

**decided (Stage 1 estimator).** Start with the ratio of group means, as specified above: it is simple, closed-form and matches the scorer's effect size. Then compare it with a negative-binomial regression log-ratio (counts regressed on perturbed versus control, with `beta_hat` and `s` taken from the fitted model), which should handle sparse genes better. Both give the same output row, so the comparison needs no change to Stage 2. Run it as an ablation: same source data, same Stage 2, same held-out benchmark (section 7). Switch the baseline to NB only if it wins there, most likely on low-count genes (below about 50 CPM).

**open (Stage 1).**
1. Ratio of means versus NB: decided above (start with ratio, compare NB). Still open: whether NB is worth its cost beyond the sparse genes.
2. Control group: all non-targeting cells in the dataset, or only those matched by batch or sequencing lane.
3. Genes with zero perturbed mean: pseudocount, or drop the row (treat as missing, never as zero).
4. Several guides per gene: pool the cells, or estimate separately.
5. Compositional shifts from CPM normalisation when a knockdown changes many genes at once.

### 3.3 Stage 2: pooling across lines, and what tau^2 is

**Model.** For one pair (target g, response gene j), source line c = 1..K:

$$\hat{\beta}[c]  =  \theta  +  \eta[c]  +  e[c]$$

```
θ   typical true effect of g on j across cell lines (unknown, to estimate)
η[c]  true deviation of line c from the typical effect, η[c] ~ Normal(0, τ^2)
e[c]  measurement error of Stage 1,                     e[c] ~ Normal(0, s[c]^2)   (s[c] known from Stage 1)
```

**$\tau^2$ (tau-squared)** is the variance of $\eta$: how much the **true** effect of the same knockdown on the same gene differs between cell lines, beyond what measurement noise explains.
- Units: squared $log_2$ fold change. $\tau = \sqrt{(\tau^2)}$ is the typical (one standard deviation) difference between lines in $log_2$ units.
- $\tau^2 = 0$ means all lines share one true effect, and differences between estimates are noise.
- A larger $\tau^2$ means the effect is context-dependent, so what was measured elsewhere is a weaker guide to a new line.
- PK analogy: $\tau^2$ plays the role of the between-subject variance ($\omega^2$) of a parameter, and $s^2$ the role of the residual variance.

**Estimator** (DerSimonian-Laird, closed form, no iterative fitting):

```
w[c]       = 1 / s[c]^2                       fixed-effect weights
theta_FE   = sum(w * beta_hat) / sum(w)
Q          = sum( w * (beta_hat - theta_FE)^2 )       heterogeneity statistic; about K-1 if no true variation
C          = sum(w) - sum(w^2) / sum(w)
tau^2      = max( 0, (Q - (K - 1)) / C )

w*[c]      = 1 / (s[c]^2 + tau^2)             random-effects weights
theta_hat  = sum(w* * beta_hat) / sum(w*)
var(theta_hat) = 1 / sum(w*)
```

**Prediction for a new line.** The new line has its own deviation, so the predictive uncertainty is larger than the uncertainty of theta_hat:

```
predictive sd for a new line  =  sqrt( var(theta_hat) + tau^2 )
```

#### Worked example (illustrative numbers, not real data)

Five source lines measured the same pair (g, j). Two scenarios with identical standard errors:

| Line | s | beta_hat, case A (lines disagree) | beta_hat, case B (lines agree) |
|---|---|---|---|
| 1 | 0.10 | -0.80 | -0.62 |
| 2 | 0.12 | -0.65 | -0.58 |
| 3 | 0.15 | -0.20 | -0.66 |
| 4 | 0.20 | -0.95 | -0.55 |
| 5 | 0.10 | -0.55 | -0.60 |

| Quantity | Case A | Case B |
|---|---|---|
| Fixed-effect theta_FE | -0.628 | -0.606 |
| Q (expected about 4 if no variation) | 14.33 (chi-square p about 0.006) | 0.28 |
| C | 257.97 | 257.97 |
| **tau^2** | **(14.33 - 4) / 257.97 = 0.0401** (tau = 0.20) | 0 (Q below K - 1, so truncated at 0) |
| Normalised random-effects weights, lines 1 to 5 | 0.23, 0.21, 0.18, 0.14, 0.23 | 0.30, 0.21, 0.13, 0.07, 0.30 |
| theta_hat | -0.622 | -0.606 |
| sd of theta_hat | 0.107 | 0.054 |
| **Predictive sd for a new line** | **0.227** | **0.054** |

Reading it:
- Both cases pool to nearly the same effect (about -0.6).
- In case A the lines disagree more than their noise allows, so tau^2 is positive. Weights flatten toward equal: the precise lines 1 and 5 drop from 0.30 to 0.23 each, while the imprecise line 4 rises from 0.07 to 0.14. The prediction for a new line is about four times less certain than in case B.
- Case B is the "effect is shared across lines" situation, where a new line can be predicted with confidence.
- Caution: with K = 3 to 8 lines, tau^2 is poorly determined. Two ways to stabilise it: give tau^2 a prior shared across genes (it should depend on expression level and effect size), or fix it at a pooled value for pairs with K <= 2.

### 3.4 Sparse prior (adaptive shrinkage)

**proposed.** Most pairs (g, j) have no true effect. The pooled estimate for a new line is treated as a noisy reading of the new line's true effect, with variance `var(theta_hat) + tau^2`, and a prior concentrated at zero is placed on that true effect.

```
prior on the new line's true effect beta_new:   pi0 * (point mass at 0)  +  sum_k pi_k * Normal(0, sigma_k^2)
posterior (per pair) gives:
   P(null)        probability the true effect is zero
   posterior mean the shrunken effect (shrinks to zero when the evidence is weak)
   lfsr           local false sign rate: probability the sign of the effect is wrong
                  (null mass counts as "sign wrong")
```

The mixture weights pi are estimated from all pairs together (as in the `ashr` method of Stephens), so the amount of shrinkage is learned from the data and not set by hand.

#### Worked example (illustrative: pi0 = 0.90, one slab with sd 0.5; in practice a fitted mixture of several slabs)

| | Case A | Case B |
|---|---|---|
| Pooled estimate and predictive sd | -0.622, 0.227 | -0.606, 0.054 |
| P(null) | 0.49 | about 0 |
| Posterior mean | -0.26 | -0.60 |
| lfsr | 0.50 | about 0 |

Two pairs with the same pooled effect lead to very different decisions: case B is called a confident negative responder, case A is shrunk heavily and ranks low or drops out. That is what tau^2 buys.

**Outputs mapped to the scoring:**
- **Responder set** (Jaccard, yield): keep pairs with lfsr below a threshold; tune the threshold so that the expected number of responders matches the number the real data typically has.
- **Direction and magnitude** (fidelity, fold-change error): sign and posterior mean.
- **Confidence ranking** (reach): order genes by lfsr, smallest first.

### 3.5 Coverage tiers and missing data

| Tier | Condition | Treatment |
|---|---|---|
| 1 | Target measured in 3 or more source lines | Full pooling, tau^2 estimated with a shared prior |
| 2 | 1 or 2 lines | Pool; take tau^2 from the shared trend |
| 3 | No source line | Fallback regression from gene features (ridge or low-rank), with a conservative lfsr |

Missingness types:
1. **Target not perturbed in a line.** The main one; its extent depends on which datasets cover the panel. Report coverage by tier for the 300 targets as soon as the data are downloaded.
2. **Response gene absent from a dataset's panel.** Treat the row as missing, never as zero.
3. **Response gene not expressed in a line.** About 4,000 genes are testable in some but not all of A, B and C. A gene at zero cannot show an effect, so it is missing for a structural reason and must not be pooled as a measured zero.

### 3.6 Predictor ladder

**proposed.** One change per rung, each scored on the leave-one-cell-line-out benchmark.

| Version | Change |
|---|---|
| v1 | Mean effect over all perturbations and lines |
| v2 | Per-target effect averaged across source lines (fixed-effect pooling) |
| v3 | Random-effects pooling with tau^2 (section 3.3) |
| v3b | Sparse prior and lfsr-based responder selection (section 3.4) |
| v4 | Weight lines by similarity to the context's controls (meta-regression on control-based covariates) |
| v5 | Ridge or low-rank fallback from gene features for tier 3 targets |
| v6 | Effect-size calibration |
| v7 | One foundation-model embedding as a feature |

**open.**
- How to represent a context for similarity: control pseudobulk correlation, a learned embedding, or both.
- Whether context weights are per gene or global.
- Whether tau^2 is estimated per pair, per expression bin, or by a hyperprior.

---

## 4. Other components

### 4.1 Effect atlas (source of effects)

**proposed.** Stage 1 output for all source datasets, stored as one long table. Harmonise genes to the 18,533 symbols. Only the 300 targets of the current panel are needed for transfer, which is about 3 million (target, gene) values per line.

Sources, in order: 2025 H1, Replogle K562 and RPE1 essential, Nadig HepG2 and Jurkat, then K562 genome-wide and others if target overlap justifies the download (`APPROACH.md` 5.4).

### 4.2 Responder selection

**proposed.** Primary option: lfsr threshold from section 3.4. Alternatives to compare:
- signal-to-noise threshold on the pooled estimate;
- top-k genes per perturbation, with k tuned or predicted;
- keep genes passing the scorer's own test applied to the predicted cells (self-consistency).

The right number of responders is probably perturbation-dependent: strong, essential-gene knockdowns have hundreds of DE genes, weak ones few. The 2025 H1 data averaged about 1,519 DE genes out of 10,780 per target, and the 2026 targets have far fewer (from the public benchmark's notes).

### 4.3 Cell generator

**proposed.**
1. Sample control cells from the target context with replacement. This preserves library size, sparsity and gene-gene correlation.
2. Apply the predicted fold change on the count scale. For decreases, thin counts binomially. For increases, add Poisson or negative-binomial counts.
3. Enforce integers, non-negativity, totals under 1,000,000, and the stored-entry cap.

**open.**
- Per-cell heterogeneity: real knockdown is incomplete and varies by cell. Test a mixture of responding and weakly responding cells.
- Noise model: plain control resampling versus a fitted negative-binomial model (scVI-style).
- How the 46 non-targeting guides (`ntc_id`) can be used to estimate the "nothing happened" variability.
- Sampling controls evenly across `ntc_id` versus at random.

---

## 5. Alternatives considered and why this baseline

| Alternative | Why not the baseline |
|---|---|
| Univariate mixed model per response gene, fitted iteratively | The same structure as the baseline, but 18,533 fits for no gain; the closed-form two-stage version gives the same estimates faster and is easier to debug |
| Multivariate model over all genes jointly | Gene sets differ between datasets, which makes a fixed-width input awkward. The long format (one row per target, gene, line) avoids this. A low-rank structure remains an option for tier 3 |
| Plain weighted average of line effects (fixed effect) | Kept as rung v2. It has no between-line variance, so it cannot express "this effect varies by context" |
| Deep models (scGPT, scFoundation, GEARS) | Hardware limits (8 GB GPU), and independent benchmarks report no consistent gain over simple baselines |
| STATE, flow matching, VAEs | Need a rented GPU and more time; deferred to after the deadline |
| One-stage full Bayesian model on counts | Chosen as the advanced comparison, not the baseline (section 6) |

Reasons for choosing the two-stage baseline, in order of weight:
1. **Fit to the scoring.** The sparse-prior layer yields responder sets, signs and confidence rankings directly, which are what four of the six metrics reward.
2. **Cost.** Closed-form and vectorised; no GPU.
3. **Uses known measurement uncertainty.** Each estimate carries its own s.
4. **Handles unbalanced data.** Pairs with different numbers of lines are treated naturally.
5. **A clean comparison.** The full Bayesian model differs only in joint estimation, so the ablation is well posed.

**Novelty (secondary motivation, unverified).** The combination was also chosen because it appears less explored than the usual deep-learning approaches. This is a hypothesis, not a result. A literature check on 2026-10-09 (two extended web searches) found the components separately and no source combining them:
- Random-effects meta-analysis with predictive distributions for a new study exists in the statistics literature (the `bayesmeta` package, study-specific prediction intervals).
- Adaptive shrinkage with lfsr exists (`ashr`, offered inside DESeq2's `lfcShrink`), and one Perturb-seq network-inference preprint uses adaptive shrinkage.
- Cross-context transfer of perturbation responses is an active area (for example PerturbMap and C3TL), mostly with learned models, and with reports that cross-context generalisation often falls to baseline levels.
- No source found that applies hierarchical random-effects pooling plus lfsr to knockdown effects for a new cell line. Absence in two searches is not proof; a deeper search is needed before claiming novelty in a write-up.

Full findings, links and follow-up searches are in `NOVELTY_CHECK.md`. Whether the approach is better is decided by the benchmark, not by novelty.

---

## 6. Advanced comparison: one-stage full Bayesian model

**proposed.** The one-stage counterpart of the baseline, as one-stage nonlinear mixed effects is to two-stage in population PK.

| | Two-stage baseline | One-stage full Bayesian |
|---|---|---|
| Likelihood | Gaussian on Stage 1 estimates | Negative binomial on raw counts |
| Stage 1 uncertainty | Treated as known | Propagated into the pooling |
| tau^2 | Estimated from a handful of lines, or shared prior | Estimated jointly with priors and hierarchies |
| Cell-level noise | Separate generator | Posterior predictive sampling can supply it |
| Cost | Minutes | Hours or days; needs low-rank structure or a subset of genes |
| Main risks | Overconfident intervals | Prior sensitivity; weak identifiability with few lines |

Question it answers: do joint modelling and uncertainty propagation add anything beyond the two-stage baseline?

Design: same features, same held-out protocol, same scorer. Compare on three things: the six-metric score, calibration (do predicted sign probabilities match observed frequencies on held-out lines), and runtime. Implementation: NumPyro or PyMC, on a subset of genes (for example the top responders per target by the baseline) and with a low-rank structure.

Expectation: no clear win, since the score rewards the point prediction and independent benchmarks find simple methods hard to beat. The comparison is valuable for what it teaches.

---

## 7. Evaluation design

**decided.** Primary benchmark is leave-one-cell-line-out: hold out one source line from the atlas, weights, scale factors and regression, predict it from its controls only, score with `cell-eval2 --preset vcc2026`, rotate over the lines.

Rules:
- Tune only across several held-out lines, never on one.
- Develop on 50 to 100 perturbations, confirm on the full set.
- Record config, code commit, `cell-eval2` version and all six metrics for every run.
- Trust the ordering of methods, not the absolute numbers.

**open.**
- Which sub-score each step (A, B, C) should be tuned on in isolation.
- Whether the H1 data can stand in as a held-out line given it is the only source with the same chemistry.

---

## 8. Open questions to resolve next

1. Stage 1 estimator: ratio of means or negative-binomial log-ratio.
2. How to estimate tau^2 stably with 3 to 8 lines, and how much a shared prior helps.
3. How to represent a context for similarity weighting.
4. What fraction of the unknown final 300 targets falls in tier 1, 2 and 3.
5. How the noise model is chosen: resampling, negative binomial, or a learned generator.
6. Is the sparse prior needed beyond random-effects pooling, as measured on held-out lines.

---

## 9. To expand

- Pseudocode for the Stage 1 builder, the pooling step and the cell generator.
- The mixture fit for the sparse prior (EM over all pairs).
- Meta-regression with control-based covariates (rung v4).
- Ablation plan with the expected cost of each rung.
- Failure analysis template: per-perturbation breakdown of where each metric loses points.

---

## Sources

- [bayesmeta: Bayesian random-effects meta-analysis, with predictive distributions](https://cran.r-project.org/web/packages/bayesmeta/vignettes/bayesmeta.html)
- [Study-specific prediction intervals for random-effects meta-analysis](https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1490)
- [ashr: adaptive shrinkage and lfsr](https://rdrr.io/cran/ashr/man/ash.html)
- [DESeq2 lfcShrink](https://rdrr.io/bioc/DESeq2/man/lfcShrink.html)
- [Causal gene regulatory network inference from Perturb-seq (uses adaptive shrinkage)](https://www.biorxiv.org/content/biorxiv/early/2026/02/19/2026.02.18.706642.full.pdf)
- [PerturbMap: cross-context transfer of single-cell perturbation responses](https://arxiv.org/html/2607.28090v1)
- [C3TL: causal cellular context transfer learning](https://arxiv.org/pdf/2603.13051)
- `docs/SCORING.md` for the scorer definitions used here
