# EDA plan

Started 2026-10-13. Order of exploration and downloads, with a decision rule at each step. Source choices are in `DATASET_PREPARATION.md` (section 3), the model in `MODEL_DESIGN.md`. Code is in `eda/`, the notebook is `eda/EDA.ipynb`.

Status markers: **done** (run on 2026-10-13), **next**, **later**.

## 1. Their data first, or the checklist?

Their data first, but it takes 20 seconds, so it is not a real choice. The three control files are already local, they fix the gene axis, depth and testable genes that every later step depends on, and they tell you which source lines resemble A, B and C. The checklist items (downloads, Stage 1 builder) all assume those facts. So: run phase 0 today, then the checklist in the order below.

## 2. Phases

| Phase | What | Download | Time | Status |
|---|---|---|---|---|
| 0 | Control cells A, B, C: depth, sparsity, `ntc_id` balance, testable genes, panel-target expression, null calibration with the 46 non-targeting guides | none | 20 s | **done** |
| 1 | Small sources: K562 pseudobulk, X-Atlas guide calls, CD4+ T knockdown table; coverage and tiers | 0.7 GB | 1 min | **done** |
| 2 | First between-line check: K562 against CD4+ T effects for the panel targets | 0.15 GB, remote | 8 min | **done** |
| 3 | H1 2025 as the benchmark line (train, validation, test: 300 targets, 34.4 GB): downloaded and checked (notebook section 10); next its scorer run on control resampling | 34.4 GB | 1 h | **download done**, scorer **next** |
| 4 | Stage 1 for K562 and CD4+ T over all targets; Stage 2 on the two lines; first leave-one-line-out on H1 | none (CD4+ T rows remote) | half a day | **next** |
| 5 | X-Atlas/Orion HCT116 streamed once, all targets | 46.6 GB | 3 to 6 h download | **later** |
| 6 | X-Atlas/Orion HEK293T | 79.7 GB | 5 to 10 h | **later** |

Phase 0 to 2 scripts: `eda/01_context_summary.py`, `02_download_small.py`, `03_coverage.py`, `04_concordance_k562_cd4.py`. Run them with `py -3.9 eda/<script>` (this interpreter has h5py, numpy, pandas, scipy, matplotlib, jupyter; 3.11 and 3.13 have none of them).

## 3. What phases 0 to 2 showed

All numbers are in `eda/out/` and reproduced in the notebook.

| Finding | Number | Consequence |
|---|---|---|
| Contexts are balanced | 46 `ntc_id`, 400 cells each, about 20,000 UMI per cell, 5,500 to 6,000 stored entries per cell (cap 13,200) | A generator that resamples controls stays well under the cap |
| Testable genes | 9,929 / 9,626 / 10,124; 7,991 in all three | Same as `MODEL_DESIGN.md` section 2 |
| Panel targets expressed in controls | 1 (A), 5 (B), 7 (C) at or below 5 CPM | Knockdown is visible in expression for almost all targets |
| Contexts differ | Spearman of log CPM 0.45 to 0.61 between pairs | Three genuinely different lines; a transfer must cope with shifts |
| Standard error calibrated | sd of z 1.01 to 1.10; 5 to 7% beyond 1.96 under the null; 1.2 only above 1000 CPM (34 to 56 genes) | `MODEL_DESIGN.md` 3.2 holds with the organisers' guide structure |
| K562 noise scales as 1/sqrt(cells) | sd(`beta`) times sqrt(cells) = 1.57 in every cell-count bin (50 to 200 CPM genes) | A pseudobulk gives usable `s` |
| Coverage by four lines (K562, HCT116, HEK293T, CD4+ T) | 299 of 300 targets in 3 or more lines; 273 with at least 50 cells in 3 or more | Tier 1 is reachable |
| Effects transfer weakly over all genes | K562 against CD4+ T, median Spearman 0.01 (shuffled 0.00); only 9 of 259 targets above 0.1 | The effect comes from a few genes, not the whole profile |
| Where both lines call a gene, signs agree | 86 to 100% for the strongest quartile of targets | Transfer works for strong, shared effects |

| First Stage 2 proxy (2026-10-16): K562 + CD4+ T Rest, predicted against held-out H1 effects on 126 benchmark targets (`eda/09_eval_stage2.py`) | Rank correlation 0.02 to 0.04 for every rung; sign accuracy among H1 responders 0.52 to 0.54 (chance 0.50); unshrunk pooling (v2, v3) is much worse than predicting no change (skill -6 to -10); with the sparse prior (v3b) the prediction collapses to about no change (skill -0.03, null weight 0.99) | With two noisy lines the pooling adds nothing over the generic response (v1). Per target the picture varies: METTL3 and METTL14 agree with H1 (rho 0.37 to 0.39, sign 0.9), DHX36 and MED12 partly, most do not. H1 effects are tiny (median abs beta 0.04, s 0.03) against K562 and CD4+ T `s` of 0.14 to 0.18, so most H1 "responders" are below what the sources resolve |
| Re-cut of the same proxy (2026-10-17): large true effects (\|beta\| > 0.2) and targets split by how strongly the sources see them | Large effects, K562 + CD4+ T: sign accuracy 0.57 (v1 0.52), precision at k 0.05 to 0.06 against chance 0.018. By source strength (quartiles of genes with pooled \|z\| > 3): rank correlation for v2 to v3b rises from 0.01 (weakest) to 0.10 (strongest), v1 stays at 0.02 to 0.06; sign accuracy 0.52 to 0.575. Unshrunk v2 and v3 stay far below "no change" in skill (-2 to -20) in every quartile; v3b stays near 0 | Pooling has signal where the sources see the target strongly, and there it beats v1 by a modest margin; elsewhere it adds nothing. Shrinkage (v3b) is required to avoid harm. The panel has similar source strength to the benchmark: median 55 against 59 genes with \|z\| > 3, but only 11% of panel targets are in the strongest quartile (25% in the benchmark) |
| Four lines (2026-10-18/19): K562 + CD4+ T Rest + HCT116 + HEK293T against H1 (`eda/09_eval_stage2.py`, tags `_4src_fix`) | Best rung v3b: rank correlation 0.0365, sign accuracy 0.551; v1 (no pooling) 0.0196 and 0.509; K562 + CD4+ T only 0.030 and 0.540. Scoring only genes above 20 CPM in every source (`eda/12_eval_by_cpm.py`): rank correlation 0.065 (pooled) against 0.065 to 0.08 (v1), sign accuracy 0.57 against 0.56. Dropping low-CPM genes from training changes nothing (`eda/13_train_mask.py`) | Each added line helps a little; the sign information is the part pooling adds, the rank gain sits in low and middle expression. Absolute level stays near chance. `s` in all four Stage 1 tables is calibrated (robust sd of beta / s = 1.04 to 1.09). The earlier z_sd of 8.8 (HCT116) came from v1 averaging genes seen in one estimate; fixed in `stage2.generic` (minimum 30 estimates) |
| Add-a-line test (2026-10-19, `eda/14_add_line.py`): hold out HCT116 or HEK293T, fit the other lines without and with H1 | Median rank correlation 0.013 to 0.023 and sign accuracy 0.55 to 0.58 for pooled rungs, with or without H1; the differences are at most 0.005 and have no consistent sign, on the 149 targets H1 covers and on the 425 it does not. v1 sign accuracy is 0.44 to 0.49 (below chance) against 0.55 to 0.58 for pooled | H1 is neutral, so it stays in the final fit (rule in 5b). X-Atlas lines are predicted from the other lines at a level close to chance for ranking; the useful part is sign |
| First real-scorer result (2026-10-19, `eda/15_generate_h1.py` + `vcc-h1 score`): v3b effects from K562 + CD4+ T Rest + HCT116 + HEK293T (no H1), applied to resampled H1 control cells, 126 targets x 400 cells | avg_score -0.142 (unchanged controls -0.303; replicate ceiling 1, baseline 0). Per metric against unchanged controls: pds_cosine +0.359 (-0.031), direction fidelity -0.867 (-1.350), direction reach +0.055 (-0.007), sig_jaccard -0.391 (-0.424), lfc_nmae -0.005 (-0.004) | Generated effects help (+0.16 over unchanged controls), mostly through the profile cosine and the sign of called DE genes, but the score is still below the control-mean baseline. Run time of `vcc-h1 score` is about 25 min per file |
| Real scorer, three rungs (2026-10-19, same four source lines, no H1, one seed each) | avg_score: unchanged controls -0.303, v1 -0.180, **v3 +0.076**, v3b -0.142. v3 per metric: pds_cosine +0.572, direction fidelity -0.040, direction reach +0.093, sig_jaccard -0.121, lfc_nmae -0.050 | Pooling across lines is worth +0.26 over the generic response (v1), and it beats the control-mean baseline. The sparse prior (v3b) costs 0.22 under the real scorer although the proxy metrics ranked it first: shrinking effects toward zero removes the signal the DE-based metrics need. Next: apply v3 without the prior, and tune the responder cutoff and the effect size on H1 before the A/B/C file. One seed per rung, so differences below about 0.05 are not established |

Reading the last two together: for most panel targets the response is small (median 5 DE genes in K562) and a 170-cell pseudobulk cannot see it. This is the main risk to the whole approach (section 5).

## 4. What to explore next, in order

Each item says what would change if the answer is bad.

1. **Notebook section 1, questions to try** (30 min). Marker genes of A, B, C; whether `ntc_id` groups show batch structure. If an `ntc_id` is an outlier, the control group in Stage 1 should exclude it, or use matched guides (`MODEL_DESIGN.md` 3.2, open item 2).
2. **H1 2025 EDA** (phase 3). Same four things as phase 0 (depth, sparsity, `ntc_id` or guide balance, expressed genes), plus its 300 targets (train 150, validation 50, test 100; 298 with at least 50 cells) and the effect-size distribution. The structure check is done (notebook section 10): the three files share the same 38,176 control cells, depth is about 54,000 UMI per cell, 48 batches. Then score control resampling with `cell-eval2` and check it returns about -0.30 (`APPROACH.md` 5.3), so the benchmark is trustworthy before any model is scored on it.
3. **How much of the panel signal is learnable.** On H1, take the 271 targets that K562, HCT116, HEK293T and CD4+ T all measure (293 in at least three of the four) and run rungs v1 to v3 (`MODEL_DESIGN.md` 3.6). Report the score at each rung. If v3 does not beat v1 (mean effect), the pooling idea has no purchase on this regime and the plan changes to section 5.
4. **Stage 1 comparison on K562 single cells** (post-deadline unless time allows): pseudobulk ratio against mean of per-cell CPM, to check that the pseudobulk route of `DATASET_PREPARATION.md` 3.6 is unbiased.
5. **`tau^2` by expression level and effect size.** The 8,999 targets perturbed in K562, HCT116, HEK293T and CD4+ T give thousands of pairs. This is the trend `MODEL_DESIGN.md` 3.3 needs, and it is only available once HCT116 is streamed.
6. **Similarity of contexts to lines** (rung v4). With HCT116 and HEK293T single cells you can build per-line control profiles and test whether similarity predicts which line's effect transfers best to which.

## 5. Decision rules and the main risk

| After | If | Then |
|---|---|---|
| Item 3 | v3 beats v1 by a clear margin on at least 3 of the metrics | Keep the two-stage plan; spend the time on HCT116 |
| Item 3 | v3 is at or below v1 | Drop pooling as the main idea. Fall back to: per-target effect from the best single line (CD4+ T has most cells), shrunk with the sparse prior; then generator tuning |
| Item 2 | Control resampling does not score about -0.30 | Fix the plumbing before anything else. **Passed 2026-10-15:** -0.302589 on H1 (126 targets), equal to the published value |
| Phase 4 | Stage 2 `tau^2` is near zero for strong targets and large for weak ones | Expected; use a prior on `tau^2` by effect size (`MODEL_DESIGN.md` 3.3, caution) |
| 2026-10-22 | Test targets are mostly outside K562 genome-wide | Rely on HCT116, HEK293T, CD4+ T; K562 becomes a secondary line |

**Main risk.** The panel targets are weak, so (a) most of the score comes from a minority of strong targets, and (b) the noise in each source's `beta_hat` is comparable to the effect. Both are visible in phase 2. Do not conclude from phase 2 alone that transfer fails: K562 pseudobulk `s` is about 0.1 to 0.2 for typical genes at 170 cells, so only effects above about 0.4 log2 units are seen.

## 5b. Experiment order for lines and H1 (decided 2026-10-16)

H1 is a line like the others. Two uses, never mixed in one run:

| Step | Sources used in the fit | Scored on | What it answers |
|---|---|---|---|
| 1. Development | K562, CD4+ T, then HCT116, HEK293T. **All of H1 is held out** (leave one line out, no leakage) | H1 benchmark (`vcc-h1`, 126 targets) | Does pooling (v3, v3b) beat the mean effect (v1)? |
| 2. Add-a-line test | Hold out HCT116 (or HEK293T); fit with the other lines **without H1**, then **with H1** | Same held-out line, on the targets H1 covers (299 of 300 are in HCT116) | Does adding a line help? Keep H1 if it helps or is neutral, drop it if it hurts |
| 3. Later lines | Add each new line the same way: one more block in the Stage 1 table, rerun step 2 | Same | Is the gain from lines still growing? |
| 4. Final fit | All lines including H1 | A/B/C leaderboard as an external check only, then D/E/F | The submission |

Notes. (a) H1 covers only 25 of the 300 panel targets, so its effect on the A/B/C leaderboard is diluted; do not judge the add-a-line test there. (b) Not changing anything is a result: H1 stays in if it is neutral. (c) A fast proxy exists before generated cells: `eda/09_eval_stage2.py` compares predicted with held-out H1 effects. The real score needs the cell generator and `vcc-h1 score`.

## 6. Download order

1. Done: the 0.7 GB set (`02_download_small.py`).
2. Done (2026-10-14): H1 2025 from Arc's public bucket `gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/` (train 15.5, validation 6.9, test 12.0 GB), by `05_download_h1.py`, checksums verified. The Hugging Face mirror is not needed.
3. Then: X-Atlas/Orion HCT116 (46.6 GB), streamed, not stored: one pass that accumulates per-target sums and sums of squares of CPM for **all** targets. Needs roughly 6 GB of RAM for the accumulators (18,300 targets by 18,533 genes in float64, twice). Do not start HEK293T (79.7 GB) until HCT116 has been used.
4. Last: CD4+ T DE statistics for all targets (16.8 GB), only if the panel rows turn out not to be enough. K562 single cells (65.8 GB) and KOLF2.1J (189 GB) are post-deadline.

Queued 2026-10-17, unattended (`eda/10_download_xatlas.py`, `eda/11_download_more.py`): HCT116, then HEK293T with Stage 1 streaming after each, then four more files in this order:

| File | Size | Why |
|---|---|---|
| Nadig Jurkat single cells (GEO GSE264667) | 9.4 GB | Resembles context A (Spearman 0.71 of control profiles). A second held-out line, with real cells for `cell-eval2`. `MODEL_DESIGN.md` 7 says to tune across several held-out lines, never one, and H1 alone is one |
| Nadig HepG2 single cells | 5.6 GB | A third held-out line |
| Replogle RPE1 single cells | 8.7 GB | A fourth held-out line |
| Replogle K562 genome-wide single cells | 65.8 GB | Exact per-cell `s`, a check of the pseudobulk shortcut, the NB ablation |

Not queued: KOLF2.1J (189 GB, stored column-wise, would not fit beside the rest), K562 essential single cells (same line as K562), CD4+ T cell-level files (119 to 173 GB each), Tahoe-100M and scBaseCount (not CRISPRi). The three essential-gene lines have no panel targets (0 of 300), so they serve as benchmarks only, not as sources.

Disk: 359 GB free on D: after the H1 download (34.4 GB used). HCT116 stored is another 47 GB; stream HEK293T instead of storing it.

## 7. Open

- H1 gene axis is 18,080 (organisers' axis 18,533, 457 of whose genes are missing from H1). Predictions for those 457 cannot be scored on H1.
- Whether controls differ between the 48 Flex batches (matched-batch controls for H1 Stage 1).
- Two H1 training targets have fewer than 50 cells (ATP6V0C 33, RNF20 47): exclude or flag in the benchmark.
- Which `cell-eval2` version the organisers used (`APPROACH.md` 5.3).
- How to run the scorer on a held-out source line that has fewer than 400 cells per target.
