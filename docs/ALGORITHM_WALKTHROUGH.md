# How the model works, step by step

A reminder of the current algorithm (state of 2026-10-21, leaderboard entry L5 = 0.1705). Formulas first, then one worked
example, then every setting on top and why it is there. Experiment numbers (H.., E.., L..) point to `docs/EXPERIMENT_LOG.md`.

## 0. The task in one paragraph

A new context (a cell line we have never seen) comes with control cells only. For each of 300 knockdown targets we must
deliver 400 cells of raw counts. We never predict cells directly. We predict one number per (target t, gene g): the log2
fold change P(t, g) that knocking down t causes in g. Then we take real control cells of the context and change their
counts by that fold change. The scorer compares our cells with the real knockdown cells.

## 1. Words used

| Symbol | Name | Meaning |
|---|---|---|
| line l | source line | a cell line with a public knockdown screen (K562, HCT116, ...) |
| β_l | measured effect | log2 fold change of gene g after knocking down t, **measured in line l only** (Stage 1) |
| s_l | standard error | how noisy β_l is; depends on how many cells that screen had |
| θ | shared effect | the part of the effect common to all lines (unknown, estimated) |
| u_l | line deviation | how line l's true effect differs from θ (unknown) |
| e_l | sampling noise | measurement error of that screen (unknown) |
| τ² | between-line variance | how far a typical line's true effect is from θ; one value per expression bin of g |
| w_l | weight | 1 / (s_l² + τ²): how much line l counts in the average |
| θ̂ | pooled mean | weighted average of the lines' β_l: our estimate of θ |
| Var(θ̂) | uncertainty of the mean | 1 / Σ w_l; shrinks as more lines measure the pair |
| sd | predictive sd | sqrt(Var(θ̂) + τ²): how unsure we are about the effect in a NEW line |
| z | standardised effect | θ̂ / sd: how many sds the effect is away from zero |
| m | reference sd | the median sd over all measured (target, gene) pairs |
| P | emitted effect | the log2 fold change the cell generator applies |

The model behind the formulas: in every line, `β_l = θ + u_l + e_l`. This is an assumption about what a measurement is
made of. It is never computed; we only see β_l and s_l.

## 2. The pipeline at a glance

```
public screens ──Stage 1──> per line: β_l, s_l for 575 targets × 11,961 genes       (etl.stage1 -> out/stage1_<line>.npz)
               ──Stage 2──> per pair: τ², weights, θ̂, sd                             (model.pool.predict)
               ──z──────>   P = 1.5 × m × θ̂ / sd                                    (model.pool.z_effect)
               ──[H51]──>   remove α × the top 3 patterns shared by all targets      (tested, NOT used: failed/h51*)
               ──context──> × bin multiplier, promoter-neighbour cap, floor 0.02      (model.effects; `vcell train`)
               ──generator> 400 resampled control cells per target, counts changed by 2^P   (model.generate)
               ──vcc prep─> .vcc file to upload                                     (model.package)
```

## 3. Step by step, one (target, gene) pair

Example: target X, gene G, three lines measured it. The numbers are made up to show the arithmetic.

**Step 1. Read each line's measurement (Stage 1, done once per line).**

| Line | β_l (log2FC) | s_l |
|---|---|---|
| K562 | −0.40 | 0.10 |
| HCT116 | −0.20 | 0.15 |
| HEK293T | −0.60 | 0.20 |

Each row comes from that line's own screen: knocked-down cells against that line's own control cells.

**Step 2. Estimate τ², the between-line variance (`stage2.tau2_by_bin`).**
Not from this pair alone (three numbers are too few). All pairs whose gene G falls in the same expression bin (CPM edges
5 / 10 / 50 / 200 / 1000, by the gene's expression in the source lines) are used together: how much do the lines disagree
beyond what their s_l explains? The excess is τ² (DerSimonian–Laird method of moments):
`τ² = max(0, (ΣQ − Σ(K−1)) / ΣC)` over the pairs in the bin, where Q is the weighted squared disagreement of a pair and K
the number of lines that measured it. Say G's bin gives **τ² = 0.02** (τ ≈ 0.14 log2).

**Step 3. Weight each line: w_l = 1 / (s_l² + τ²).**

| Line | s_l² + τ² | w_l |
|---|---|---|
| K562 | 0.0100 + 0.02 = 0.0300 | 33.3 |
| HCT116 | 0.0225 + 0.02 = 0.0425 | 23.5 |
| HEK293T | 0.0400 + 0.02 = 0.0600 | 16.7 |

Σw = 73.5. Without τ² the weights would be 100 / 44 / 25 and K562 would dominate. τ² says "even a perfect screen of
one line tells you only about that line", so it evens the weights out.
Lines can also get a fixed extra factor λ_l (KOLF and Pan_H1 have λ = 0.25): the weight becomes λ_l · w_l.

**Step 4. Pooled mean θ̂ = Σ w_l β_l / Σ w_l.**
θ̂ = (33.3 × −0.40 + 23.5 × −0.20 + 16.7 × −0.60) / 73.5 = **−0.381**.
Our best guess of what knocking down X does to G in "a typical line".

**Step 5. Uncertainty of that mean: Var(θ̂) = 1 / Σ w_l = 1 / 73.5 = 0.0136.**

**Step 6. Predictive sd for the new line: sd = sqrt(Var(θ̂) + τ²) = sqrt(0.0136 + 0.02) = 0.183.**
Two unknowns add up: we do not know θ exactly (Var(θ̂)), and the new line has its own deviation u_new (τ²). We know
nothing about u_new, so the centre of the prediction stays θ̂: β_new ~ N(θ̂, sd²).

**Step 7. Standardise: z = θ̂ / sd = −0.381 / 0.183 = −2.08.**
The effect in the new line is predicted about 2 sds below zero: probability that it really goes down is Φ(2.08) ≈ 98%.

**Step 8. Emitted effect: P = 1.5 × m × z (`stage2.z_effect`).**
With m = 0.25: P = 1.5 × 0.25 × −2.08 = **−0.78 log2**.
In words: a pair with average uncertainty (sd = m) gets 1.5 × θ̂. A pair the lines agree on (sd < m) gets more; an
uncertain pair (sd > m) gets less. So P is a "confidence map", not the expected size of the effect. Pairs no line measured
get the largest sd, so almost no effect. The 1.5 was tuned (H11, H26).

**Step 9. Context adjustments (`16_generate_abc.py`).**
- Bin multiplier by G's CPM in the new context's control cells: <15 ×1.5, 15–40 ×1.25, 40–120 ×1, 120–400 ×1,
  >400 ×0.8 (H36). Say G has 60 CPM: ×1, still −0.78.
- Promoter neighbours: genes whose start lies within 5 kb of the target's are also silenced by CRISPRi, so their effect is
  capped at log2(0.15) = −2.7 at ≤ 500 bp, easing to 0 at 5 kb (H37).
- Floor: |P| < 0.02 is set to 0.

**Step 10. Make the cells.** Fold change 2^−0.78 = 0.58. Draw 400 control cells of the context at random (with
replacement). In each, every count of G is kept with probability 0.58 (binomial thinning). For an increase
(P > 0), extra Poisson counts with mean (2^P − 1) × mean count of G × the cell's size factor are added instead.

**Step 11. Package with `vcc prep`** (3 contexts × 300 targets × 400 cells) and upload.

When the lines disagree, the effect cancels instead: β = +0.3, −0.5, +0.1 with the same s gives θ̂ ≈ 0.00, so P ≈ 0.
Disagreement on one pair does not change that pair's sd (τ² is per bin), it changes θ̂.

## 4. The whole formula

```
P(t, g) = scale × bin(g) × m × θ̂(t, g) / sqrt( 1/Σ_l w_l + τ²(bin of g) )
with      θ̂ = Σ_l λ_l w_l β_l / Σ_l λ_l w_l,    w_l = 1 / (s_l² + τ²(bin of g))
then      neighbour cap, floor 0.02   (H51 shared-pattern removal: tested, lost on A/B/C, not used)
```
- From the data: β_l and s_l of every line.
- Estimated from all pairs: τ² per bin, m.
- Tuned on held-out lines: scale 1.5, the bin multipliers, the neighbour rule, λ (KOLF 0.25, Pan_H1 0.25).

## 5. The source lines (A/B/C build)

| Line | What it is | λ | Panel targets measured (of 300) |
|---|---|---|---|
| K562 | leukaemia line, Replogle genome-wide screen (10x 3') | 1 | 272 |
| CD4_Rest | primary CD4+ T cells, resting (Zhu et al., DESeq2 statistics) | 1 | 283 |
| HCT116 | colon cancer line, X-Atlas/Orion (10x Flex) | 1 | 300 |
| HEK293T | kidney line, X-Atlas/Orion (10x Flex) | 1 | 300 |
| H1 | embryonic stem cells, Arc's own 2025 data | 1 | 13 |
| KOLF | KOLF2.1J induced stem cells | 0.25 | 282 |
| Pan_H1 | H1 stem cells from another lab | 0.25 | 66 |

Every panel target is measured by at least 2 lines; 260 of 300 by 5 or more.
For D/E/F, `python -m vcell qc identity` (vcell.qc.identity) checks whether a context IS one of these cell types (markers, Y genes, genes that only
that line expresses). If so, that line gets λ = 4 (H44, H47: +0.06 to +0.07 on the official H1 scorer, the largest gain of
the project). A stem-like context gets Pan_H1 at 4.

## 6. Shared-pattern removal (H51), the newest step

**What it does.** Write the emitted effects as a table: 300 rows (targets) × ~12,000 columns (genes). Many rows contain the
same pattern: a group of genes that moves after almost any knockdown in the source lines. SVD finds the gene patterns that
best describe all rows at once; the top 3 are the patterns most targets share. Each row is then written as "its share of
these 3 patterns + its own rest", and α × the shared part is subtracted. α = 1 removes it all, α = 0.5 half. No gene is
removed: all genes stay, only the shared pattern is subtracted from every target. The 3 patterns hold 9% of the total
squared effect over the 300 A/B/C targets (18% over the 126 H1 targets); for a typical target they are 5% of its own
effect (quartiles 2% / 5% / 9%).

**What the 3 patterns are** (`experiments/analysis/h51c_shared_patterns.py`, log `out/shared_patterns.log`). Not ribosomes:

| Pattern | Energy | Genes | Carried mostly by |
|---|---|---|---|
| 1 | 4.9% | **long genes go down** (LRP1B, CACNA2D3, AUTS2, ERBB4, ROBO1 ...): its top 100 genes have median length 408 kb against 35 kb for all genes; loading vs gene length Spearman −0.70 over all genes | HEK293T, then Pan_H1, HCT116, KOLF; absent in Arc's H1 (mean projection −0.12) |
| 2 | 2.6% | interferon and p53 stress genes go up (TRIM22, XAF1, SLFN5, TP53INP1, TXNIP) | all lines in the same direction, strongest in H1 and CD4+ T |
| 3 | 1.7% | amino-acid stress genes (CHAC1, WARS, STC2) and others | the lines disagree: H1 has the opposite sign |

So pattern 1 is a gene-length trend (a global transcription or technical effect of the source screens, not target
biology), pattern 2 a real generic stress response, pattern 3 a response whose direction depends on the line. All three
are things a new line may not share, which is why removing them helps.

**Toy example** (4 genes; G1 and G2 go down for every target, G3 only for A, G4 only for B):

| Target | G1 | G2 | G3 | G4 | after removal (α = 1) |
|---|---|---|---|---|---|
| A | −1 | −1 | −0.6 | 0 | 0, 0, −0.6, 0 |
| B | −1 | −1 | 0 | −0.6 | 0, 0, 0, −0.6 |
| C | −1 | −1 | 0 | 0 | 0, 0, 0, 0 |

Before, A and B look 85% alike (cosine 0.85); after, they do not overlap at all (cosine 0).

**Why it can help.** The pds metric asks: is the prediction for target t closer to t's real profile than to the other
targets' real profiles? If all predictions share a big common part, they all look alike and this ranking is decided by
noise. Removing it makes each prediction distinct. Also, the shared part is learned from other cell lines; the generic
response of a new line is different, so predicting it everywhere is partly wrong.

**What it costs.** Target C loses everything: if the generic genes really change, those DE calls are gone (fid and jac fall
a little).

**What it gained** (overall score on the leaderboard scale: 0 = the organisers' mean-response baseline, 1 = a repeat of
the real experiment; for reference, our +0.0105 from L2 to L5 moved us from rank 463 to 438):

| Check | No removal | α = 0.5 | α = 1 |
|---|---|---|---|
| Official H1 scorer, H1 left out of the pool, seed 0 / 1 | 0.121 / 0.131 | 0.133 / 0.139 | 0.151 / 0.136 |
| same, mean | 0.126 | **0.136 (+0.010)** | **0.143 (+0.018)** |
| Proxy, candidate settings, 8 held-out lines | — | +0.007 (8 of 8 lines up) | +0.009 (6 of 8 up) |

Per metric on H1 (mean of the seeds, none → α 0.5 → α 1): pds 0.688 → 0.740 → 0.773, reach 0.092 → 0.100 → 0.118,
nmae −0.006 → 0.005 → 0.015, fid 0.042 → 0.038 → 0.026, jac −0.061 → −0.068 → −0.071, mse 0.
**Raw mse** (1 = predicting no change; the mse score is 0 above the baseline's ~0.98): 2.19 → 1.98 → 1.92 on H1. Our
leaderboard raw mse is 3.81; removing 9% of the effect energy should bring it to about 3.5 to 3.6, so the mse score stays 0.

**All settings tested** (proxy, candidate, 8 held-out lines; change of the estimated overall; k = number of patterns):

| k | α = 0.5 | α = 1 |
|---|---|---|
| 1 | +0.004 (8 of 8 lines up) | +0.004 (6) |
| 3 | +0.007 (8) | **+0.009 (6)** |
| 5 | +0.007 (8) | +0.008 (7) |
| 10 | +0.006 (7) | −0.001 (6) |
| 20 | +0.005 (6) | −0.008 (4) |
| 40 | +0.004 (7) | −0.017 (1) |

Subtracting the plain average over targets instead: +0.001 / +0.002. pds rises with k, fid and jac fall with k; past
about 5 patterns the removed part holds real target-specific signal and the DE metrics lose more than pds gains.

Second round (`experiments/failed/h51d_shared_pattern_variants.py`, H51d): **k 3 at α 0.75 is best, +0.010 with all 8 lines up**. Removing
each source line's own patterns before pooling (+0.005), damping genes that move for most targets (+0.009) and removing
the gene-length trend directly (+0.002) were not better. The second upload of 2026-10-21 (`cand_c3.vcc`) uses k 3, α 0.75.

**Expected on the leaderboard:** about +0.01 (likely +0.005 to +0.015). The leaderboard noise between generator seeds is
probably about ±0.004.

**Actual leaderboard result (L6, 2026-10-21): 0.1683 against 0.1705 without it.** Raw mse fell as predicted
(3.81 → 3.56) and fid, reach and jac lost a little as expected, but pds did not rise (raw 0.7727 → 0.7743, expected
about +0.04). In Arc's A/B/C cells the real profiles probably share these patterns too, unlike H1. **Not used in the
final.** Lesson: a change that works through the shared part of the response depends on the context; the H1 stand-in
and the held-out lines can be wrong about it.

## 7. What limits the score

- **Most of a line's response is its own.** Even an infinitely large pool of other cell types would correlate only about
  0.06 to 0.15 with a new line (H43). Same-cell-type data reaches 0.31. That is why the same-type rule is the big lever and
  why modelling changes move the score by thousandths.
- **mse versus the DE metrics.** With such low correlation, the effect size that minimises mse is about 10x smaller than
  what the DE metrics need to call genes significant in 400 cells. Honest emission cannot have both; we keep mse at 0 and
  win the others (H34, H39b, `out/fig_mse_attempts.png`).
- **Tried and rejected** (each in the log): sparse shrinkage (H2), confidence cut-offs (H8), similarity weighting of lines
  (H13, H35), related lineage (H48), gene programs (H24), learned models on top (H25, H27), the context's own control
  covariance (H50).

### Why the model generalises but stays near 0.17

In model terms the new line's effect is β_new = θ + u_new. We estimate θ well, but in the data u_new is far larger:

| What we feed the generator (H1 left out) | Correlation with the real H1 effects | Official H1 score |
|---|---|---|
| The true H1 effects | 1 | 0.97 (so making cells is not the problem, H42) |
| Our pool of 4 other cell types | 0.09 to 0.15 (explains 1 to 2%) | 0.126 |
| The real measured profile of one other cell type | 0.08 to 0.27 | 0.014 to 0.063 (H42b) |
| Infinitely many other cell types (extrapolated) | ceiling 0.06 to 0.15 | — (H43) |
| Same cell type from another lab (Pan_H1, covers 37% of targets) | 0.31 | 0.190 with the pool (H47) |

Biologically: a knockdown acts through the cell's own wiring (which transcription factors, pathways and chromatin are
active). The same target in a T cell and in an epithelial cell reaches different genes. What all cell types share (the
generic stress and gene-length patterns of section 6) is a small part, and part of it does not even transfer.

### How the other teams score (leaderboard rows seen 2026-10-20 / 21)

| Rank | Overall | mse score (raw) | pds | Most likely source of the score |
|---|---|---|---|---|
| 1 Illumina | 0.473 | 0.79 (0.24) | 0.87 | proprietary atlas of CRISPR screens in 200+ cell lines: same-type data, correlation about 0.87 |
| 2-4 | 0.33 | 0.46-0.51 (0.50-0.55) | 0.81-0.84 | unknown; correlation about 0.7 implies same-type data, likely plus the mse noise allowance |
| 33-34 | 0.304 | 0.47 (0.54) | 0.82 | same pattern as 2-4 |
| 306 | 0.197 | 0 (2.59) | 0.77 | honest-looking like ours; its lead is pds (0.767 vs our 0.603) |
| 331-335, 421-426 | 0.17-0.19 | 0.08-0.18 (0.82-0.91) | 0.65-0.78 | raw mse in the band that pinned emission reaches with no skill (0.64-0.81 + error); without it about our level |
| ours, L5 | 0.1705 | 0 (3.81) | 0.60 | public screens of other cell types, honest cells |

Ranks 61-70 and 91-100 (pasted 2026-10-21): overall 0.272-0.288; mse 0.35-0.45 (raw 0.56-0.66); pds 0.77-0.83
(raw 0.85-0.88); jac -0.01 to 0.03, nmae 0.16-0.24, fid 0.01-0.05, reach 0.21-0.28, i.e. the four significance metrics at
our level. Without mse they are at 0.205-0.219, +0.035 to +0.05 over us, almost all pds. Several rows are identical
(94, 95, 97; 64, 65): shared code. 50-114 submissions each.

**The mse loophole** (not "many submissions"). The scorer's mse is "unbiased": it subtracts an allowance for sampling
noise, estimated from how much the submitted 400 cells vary among themselves. With honest cells (independent draws) the
average of the 400 cells really carries that noise, so the allowance only cancels it. With *pinned* cells, built so their
average equals the intended effect exactly while the cells still vary, the average has no noise but the allowance is
still subtracted: the error is reported smaller than it is. By our tables, pinning with no skill at all reads raw 0.64 to
0.81. The organisers' own code calls this an exploit; Arc may change scoring for entries against the spirit of the rules.
Pinning probably also lifts pds (computed on the same averages, now noise-free) but not the four significance metrics
(per-cell Wilcoxon tests) - exactly the pattern of ranks 61-100. Not measured yet.

Above about 0.20, every team seen either has perturbation data in (nearly) the same cells or earns mse through the noise
allowance (pinned pseudobulks), which we ruled out as against the spirit of the rules. Within honest public-data
entries, the remaining gap is mostly pds, which is what the shared-pattern removal (section 6) improves.

## 8. Where the code is

| File | Role |
|---|---|
| `src/vcell/etl/stage1/` (`python -m vcell etl stage1 ...`) | Stage 1 tables per line |
| `src/vcell/model/pool.py` | pooling, τ² by bin, predictive sd, `z_effect` |
| `src/vcell/model/effects.py`, `model/generate.py`, `model/package.py` | the build for A/B/C or D/E/F (`python -m vcell build/package --config configs/<run>.toml`) |
| `configs/final_def.toml`, `configs/l5.toml` | every setting of the final entry and of L5 |
| `src/vcell/evaluation/h1_benchmark.py` | the same for the public H1 benchmark (offline official scorer) |
| `src/vcell/model/promoter.py` | neighbour cap |
| `src/vcell/qc/identity.py`, `model/sources.py` | is a new context one of our cell types? the same-line weights |
| `src/vcell/evaluation/panel.py`, `evaluation/harness.py` | offline score estimate, 8 held-out lines |
| `experiments/` (index `experiments/README.md`) | every tested idea: adopted, analysis, failed |
| `docs/FINAL_RUNBOOK.md` | the D/E/F steps |