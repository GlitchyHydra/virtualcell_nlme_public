# Scoring: how your submission is measured

Written 2026-10-08. Source: the `cell-eval2` metric specification (`docs/vcc2026_metrics/vcc2026-metrics.md` in the [cell-eval2 repo](https://github.com/ArcInstitute/cell-eval2)), read through an extraction, plus the organisers' dataset format description (not redistributed here). The extraction summarised the jackknife and anchor details, so the full spec (`main.tex` and the PDF in the same folder) remains the authority for any formula you implement yourself.

Items marked **[verify]** were not stated clearly in what I read.

---

## 1. What you submit

One `prediction.h5ad`, packaged by `vcc prep` into `prediction.vcc`.

| Requirement | Value |
|---|---|
| Rows | 360,000 = 300 perturbations x 400 cells x 3 contexts |
| Columns | 18,533 genes, in the order of `gene_names.csv` |
| `obs` | `target_gene` (gene symbol, e.g. `ADNP`) and `context` (`A`, `B` or `C`), reused exactly as in the control files |
| `X` | Raw counts: non-negative, integer, finite; no cell total above 1,000,000 |
| Storage | Sparse; at most 4,750,000,000 stored entries (about 13,200 per cell). Explicitly stored zeros count. |
| Forbidden | Control (`non-targeting`) rows; log-normalised values; construct ids such as `ADNP-1` |

`vcc prep prediction.h5ad -g gene_names.csv --perts pert_counts.csv -o prediction.vcc` validates locally. Add `--dry-run` to check without writing.

---

## 2. The scoring pipeline

Scoring is done per context, per perturbation, then averaged.

```
For each context and each perturbation p:

  REFERENCE (real data):   real perturbed cells  vs  control cells   -> reference effect and reference DE
  SUBMISSION (yours):      your predicted cells  vs  control cells   -> predicted effect and predicted DE

  Six metrics compare the submission result with the reference result.
  Each is rescaled:  s = (u - b) / (r - b)
      b = baseline score   (a prediction of the mean perturbation response: for every target, 400 control cells
                        drawn with replacement and scaled gene by gene to the real mean profile, "dispersed"
                        emission in cell_eval2/baseline.py; this is our generator fed the true mean effect)
      r = replicate score  (split-half of the real data, 5 splits, seed 0)
  s = 0  ->  as good as predicting the mean response
  s = 1  ->  as good as repeating the experiment

Context score = equal-weight mean of the six rescaled metrics
Final score   = mean over the three contexts
```

The reference control cells are the origin on both sides: the same control group defines "change" for the real data and for yours.

### 2.1 Preprocessing the scorer applies

- **Pseudobulk profile** (used by metrics 1 and 2): sum counts per perturbation, normalise to a total of 50,000, then log: `b = log(1 + 50000 * P / sum(P))`. The effect is this profile minus the control profile.
- **DE tests** (metrics 3 to 6): per-cell normalisation to 1,000,000 total, then a Wilcoxon rank-sum test of perturbed cells against controls, one gene at a time.

### 2.2 Differential-expression rules (metrics 3 to 6)

| Rule | Value |
|---|---|
| Test | Two-sided Wilcoxon rank-sum (Mann-Whitney U), normal approximation |
| Effect size | log2((mean_pert + 1e-9) / (mean_ctrl + 1e-9)), arithmetic means |
| Gene filter | Tested only if mean control expression exceeds 5 CPM. Same gene set for every perturbation, for your submission and for the reference. |
| Multiple testing | Benjamini-Hochberg within each perturbation, after the filter |
| Significant | Adjusted p < 0.05 |

### 2.3 Genes excluded

- The perturbed gene itself is excluded from all six metrics for that perturbation.
- For `pds_cosine`, all 300 panel target genes are excluded from the comparison space.

---

## 3. The six metrics

| # | Id | In plain words | Range before rescaling |
|---|---|---|---|
| 1 | `pds_cosine` | Is your predicted effect closer to its own real effect than to the other 299? | 0 to 1 (0.5 = no information) |
| 2 | `expr_mse_unbiased_capped_norm` | How large is the error in the pseudobulk effect? | Clamped to 0 to 1 after rescaling |
| 3 | `de_wilcoxon_direction_fidelity_yield_raw` | Of the genes the real data calls significant, how many do you call significant with the right sign, penalised for calling too many or too few? | 0 to 1 |
| 4 | `de_wilcoxon_direction_reach_raw` | How deep into the real significant set do your signs stay right? | 0 to 1 |
| 5 | `de_wilcoxon_sig_jaccard` | Overlap between your significant gene set and the real one | 0 to 1 |
| 6 | `de_wilcoxon_lfc_nmae` | Accuracy of fold-change size on the real significant genes | Floored at -6 after rescaling |

### 3.1 Details that drive modelling choices

**1. `pds_cosine`**
- Cosine distance between your predicted effect for p and each of the 300 real effects, in the space with all target genes removed.
- `PDS_p = 1 - k_p / (n - 1)` where `k_p` is the rank of the correct match (ties averaged). Mean over the 300 perturbations.
- It is a rank. Magnitude does not matter, only whether each perturbation's pattern is distinguishable. A model that gives every perturbation the same prediction scores 0.5, which becomes 0 after rescaling.

**2. `expr_mse_unbiased_capped_norm`**
- One ratio of panel-wide sums: `sum(N_p) / sum(D_p)`. There is no per-perturbation value.
- `N_p` is your squared error on the effect, minus a jackknife noise correction; `D_p` is the squared distance of the real effect from the control, similarly corrected.
- The correction you can claim is capped by your own spread across perturbations, so inflating variance does not game it.

**3. `de_wilcoxon_direction_fidelity_yield_raw`**
- `F_p = k / max(n_pred, n_real)`, where `n_real` is the real significant count, `n_pred` is your significant count (genes where the real log2FC is non-zero), and `k` is how many of YOUR significant genes have the same sign as the real log2FC, **whether or not the real data calls them significant**. Checked in the code on 2026-10-19 (`cell_eval2/metrics/direction.py`, `_components` and `_fidelity_family`); the earlier reading (k counted only real hits) was wrong.
- So it is sign precision over your calls, times `min(1, n_pred / n_real)`. Calling fewer genes than the real data is punished; calling more costs only through the sign precision of the extra calls, which falls toward 0.5 (a coin flip) for genes with no real effect.
- cell-eval2 notes that the generic baselines score about 0.5 here in every line.

**4. `de_wilcoxon_direction_reach_raw`**
- The real significant genes are ordered by your own confidence. `R_p = k* / n_real` where `k*` is the deepest prefix whose sign purity is at least 0.9.
- Your confidence ranking must put the most reliable genes first.

Panel regime (from cell-eval2 code comments about the three official val lines): about 12 to 30% of the 300 targets have 10 or fewer reference-significant genes and 46 to 61% fewer than 40; the smallest count in val A is 4. The val A baseline reads 0.042 on raw reach and the replicate 0.904.

**5. `de_wilcoxon_sig_jaccard`**
- `|real significant ∩ your significant| / |union|`. An empty union counts as 1.

**6. `de_wilcoxon_lfc_nmae`**
- Over the real significant genes with finite fold change: `sum|predicted lfc - real lfc| / sum|real lfc|`. Missing predicted values count as zero.
- Perturbations with fewer than 10 such genes are omitted (209 to 261 of 300 are kept).

### 3.2 Rescaling and clipping

| Metric | Clipping after rescaling |
|---|---|
| Expression error (2) | Clamped to [0, 1] |
| Fold-change error (6) | Floored at -6 |
| Others (1, 3, 4, 5) | Not clipped; values above 1 are reported |

---

## 4. What this means for a model

These follow from the definitions above. Treat the reasoning as mine, not the organisers'.

| Fact | Consequence |
|---|---|
| Both sides are scored against the same real control cells | Your predicted cells must be consistent with those controls. Resampling them is a safe start. |
| A prediction of the mean perturbation response scores 0 by construction | The score measures what you add beyond the average response, so average-response effects (generic stress, growth arrest) earn little. Perturbation-specific signal earns the rest. |
| Copying controls scores about -0.30 on the validation leaderboard | Below zero: no significant genes means almost no credit on 4 metrics, and the mean-response baseline gets some. |
| Metrics 3 and 5 reward the right number of significant genes | Calibrate how many genes respond. Both too many and too few lose points. |
| Only genes above 5 CPM in the controls are tested | Effects on lowly expressed genes never count. Spend no effort there. |
| The Wilcoxon test runs on your 400 cells | Cell-to-cell noise decides what is significant. Wrong noise gives false or missing significant genes. |
| Reach ranks by your own confidence | Output stronger, more consistent shifts for genes you trust more; weak or uncertain genes should shift less. |
| PDS is a rank over all 300 perturbations | Predictions must differ between perturbations. Uniform shrinkage towards the mean hurts it. |
| Raw counts are required; log-normalised data is rejected | Generate integer counts, not normalised values. |
| Per-context scores are averaged | A weak context costs as much as a strong one gains. |

A public benchmark showed the same method scoring 0.083 at full effect size and -0.183 at 0.13x. Shrinking effects is punished.

---

## 5. Leaderboard versus final

| | Validation phase | Final |
|---|---|---|
| Contexts | A, B, C | D, E, F (different cell lines) |
| Perturbations | 300 | A different 300 |
| Metrics | A subset from `cell-eval2` **[verify which]** | Aggregate across the six metrics above |
| Feedback | Live leaderboard | None before the deadline |
| Comparable | Not directly comparable to the final in absolute terms | |

Reference leaderboard values from public repositories (validation phase): control resampling -0.304; simple transfer at full effect 0.083; transfer with scattered sparse rises 0.098; a solo atlas-transfer entry reached 0.1546 (rank 82 on 2026-09-10).

---

## 6. Running the scorer locally

```bash
pip install cell-eval2
# optional GPU extras: pip install 'cell-eval2[gpudge]'
cell-eval2 run -ap pred.h5ad -ar real.h5ad --preset vcc2026 --pert-col target_gene -o out/
```

- Python 3.11 or newer. Windows support is not documented; use WSL2 if installation fails.
- CPU works. Differential expression runs on `pdex` (or `scanpy`) when no GPU engine is installed. On a 4-core CPU the Wilcoxon step is slow, so develop on a subset of perturbations.
- The changelog says scored values changed between releases. Pin the version and record it with every score.
- Packages: the `vcc` command-line tool packages the submission and is separate from `cell-eval2`.
- A local stand-in benchmark exists: [vcc2026-h1-benchmark](https://github.com/forrestsheldon/vcc2026-h1-benchmark) scores H1 data with the 2026 metrics.

---

## 7. Open items

- Exact subset of metrics on the live leaderboard.
- Whether the final round accepts several submissions and which one counts.
- Whether the jackknife and anchor construction in the full spec changes how a self-built benchmark should be set up.
- Numerical anchors (baseline and replicate values) per context; they are in the repo's metric folder.
