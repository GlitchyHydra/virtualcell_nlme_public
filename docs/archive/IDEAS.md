# Ideas to check (2026-10-20)

Goal: the model that generalises best to unseen cell types (final contexts D, E, F), not the best number on A/B/C. Direct
measurement of the same target in other lines stays the strongest evidence, so every idea below either adds to the pooled
estimate (more lines, better weights) or acts as a prior around it. Nothing replaces it.

Proxy = `eda/proxy.py`, leave one line out, five lines (plus Jurkat, HepG2, RPE1 on the essential axis, `eda/30`).
"Cost" is my guess for building and testing it. Priority: A first, C last.

## Already tested (do not repeat)

| Idea | Result | Where |
|---|---|---|
| Shrink to zero (sparse prior v3b) | much worse on the real scorer (-0.22) | H2, E3 |
| Shrink towards the generic mean (v1) | worse than pooling | E1 |
| Standardised effect z, scale 1.5 | best; +0.045 on H1, +0.068 on the leaderboard | H11, L2 |
| Scale 1 to 3 on lines we did not tune on | 1.5 best or tied in Jurkat, HepG2, RPE1 | H21 |
| Hard thresholds, visibility switch (significant genes separate from magnitude) | worse | H8, H16 |
| Group weights by line similarity, low rank, per-target scale, stacked meta-predictor | no gain | H13, H14, H15, H17 |
| Per-gene transferability weights | pds up, DE metrics down, net worse | H19 |
| Target's own expression predicts effect strength | no | H18 |
| K562 from single cells | no gain over pseudobulk | H20 |
| Essential screens as extra sources | help in the proxy (+0.005 to +0.03), but they cover no validation target | H21 |

## To check

| # | Idea | What it is | Why it might help | Cost | Priority |
|---|---|---|---|---|---|
| 1 | KOLF2.1J as a sixth genome-wide line | Stream 75 GB, Stage 1 table, add to pool | Each added line helped steadily; licence-clean fallback for X-Atlas | running (about 3 h) | A |
| 2 | Leaderboard probes z x2 (file ready), z x1 | One upload each | Curiosity of the user; checks what A/B/C rewards | 1 upload each | A |
| 3 | Multivariate shrinkage (gene programs) | See below | Pooled estimates are noisy per gene; genes move together | 2 to 3 h | A |
| 4 | Target-feature prior | See below | Gives a principled "shrink towards what" instead of zero or the global mean | 4 to 8 h | A |
| 5 | Knockdown-aware pooling | Per (line, target): rescale the effect by its knockdown (the `kd` array in each table) towards the typical knockdown, or inflate s when knockdown is weak. HEK293T median knockdown is 52% | Weak knockdown in one line looks like a small effect, which inflates the between-line variance | 1 h in the proxy | A |
| 6 | Control-cell co-expression prior | In the new line's own control cells (A/B/C now, D/E/F later), correlate each target with every response gene across cells | The only information that is specific to the new cell line; regulatory structure shows up as co-variation | 3 to 4 h | B |
| 7 | Similarity weights from control profiles | Weight each source line by how close its control profile is to the new line's (continuous, per context), not by two groups | Context A resembles Jurkat and T cells; B and C resemble HCT116 and stem cells. The group version failed, but only with two hard groups | 1 h | B |
| 8 | Effect floor and noise realism | Proxy: MIN_EFFECT 0.02 and 0.1 instead of 0.05. Generator: added counts follow a negative binomial, not Poisson | Our added counts are less variable than real cells, which changes the rank test | 1 to 2 h | B |
| 9 | Essential screens as sources | If the D/E/F targets overlap the essential library, add Jurkat, HepG2, RPE1, K562 essential, and rerun the proxy | Four more lines for those targets | 2 h, only after 2026-10-22 | B (depends on the panel) |
| 10 | Per-bin scale | A separate z scale for each expression bin of the response gene | Low-expressed genes are noisier | 1 h | C |
| 11 | Foundation models as a prior | Pretrained single-cell or gene-embedding models (the FAQ allows any data) as a third predictor | A different information source: sequence and literature instead of screens | days; zero-shot results are usually weak | C |
| 12 | Jiang et al. DE tables | Six more cell lines, 9 panel targets | Between-line variance only | 3 h, R objects | C |
| 13 | Licence of X-Atlas | Ask the organisers whether CC BY-NC-SA data is allowed in a prize entry | Decides whether HCT116 and HEK293T stay | one message | A, no code |

## Candidate settings to evaluate later (2026-10-20)

From the tuning on all eight held-out lines (`eda/38_tune_all.py`, H29); every gain is within the noise of about +-0.003, so
these are candidates for a leaderboard check, not decisions.

| Setting | Scale c | kappa (tau^2 weight in the standardising sd) | Effect floor (log2) | KOLF weight | Proxy, all 8 lines | Main 5 | Essential 3 |
|---|---|---|---|---|---|---|---|
| Uploaded (L2, 0.160) | 1.5 | 1 | 0.05 | 0 | 0.1676 | 0.1457 | 0.2041 |
| **Best found** | **1.875** | **2** | **0.02** | **0.5** | **0.1698** | 0.1480 | 0.2062 |
| Best without KOLF | 1.75 | 2 | 0.02 | 0 | 0.1685 | 0.1465 | 0.2052 |
| Runner-up | 1.75 | 1.5 | 0.02 | 0.5 | 0.1698 | 0.1483 | 0.2056 |

| Per-bin scale (H36) | 1.5 x [1.5, 1.25, 1, 1, 0.8] by control CPM bin <15, 15-40, 40-120, 120-400, >400 | 1 | 0.05 | 0 | **0.1703** | | |
| **Best, differentiated context (H38 + H40)** | per-bin as above; neighbour rule f 0.15 (H37) | 1 | 0.02 | 0.25 (clean weighting), plus Pan_H1 0.25 | **0.1748** | | |
| Stem-like context (H40) | as above, Pan_H1 at weight 1 | 1 | 0.02 | 0.25 | H1 alone +0.037 vs uploaded | | |

The KOLF weight 0.5 rows used the old weighting by inflated s, which biases tau^2 low (H35); recheck with `proxy.rungs(lam=...)`
before building them. Similarity weights per context (H35) lose at every sharpness: the per-context selector should only add or
drop whole lines, never down-weight within the pool.

Leaderboard facts so far: scale 2 (L3, 0.158) traded pds +0.030 against nmae -0.062; KOLF at full weight (L4, 0.157) cost pds.
To build any row: `16_generate_abc.py --transform z --scale c --min-effect floor` plus kappa and line weights, which the
generator does not take yet (stage2.predict needs a kappa and a weights argument).

## Per-context source selector (2026-10-20)

In short: **one model, one formula; only the list of source datasets that are averaged changes per context.** No separate
training.

Today one pool of source lines predicts every context. The evidence says a source helps a held-out line only when it
resembles it: KOLF helps the other stem line (H1) and hurts the differentiated lines (H28; leaderboard -0.004, L4); the
essential screens help other essential screens (H21). Contexts differ in kind (A T-cell, B mesenchymal or epithelial, C
squamous; one of D/E/F is probably a stem line), so each context should get its own pool.

How it would work, per context:
1. Profile its control cells (mean CPM of the testable genes) and compare with each source line's control profile
   (Spearman over shared genes, after log), plus marker rules: stem (POU5F1, NANOG, LIN28A), T cell (CD3E, PTPRC), epithelial
   (EPCAM, KRT5/8), mesenchymal (VIM).
2. Turn similarity into precision weights for the pooling (proxy.rungs already accepts per-line weights): the core lines keep
   weight 1, while a conditional line such as KOLF, H1 or CD4+ T Stim48hr enters with full weight only for a context it matches
   (KOLF and H1 for a stem context; Stim48hr for a T-cell context) and weight 0 to 0.5 otherwise.
3. Build each context's predictions with its own weights (the generator loops over contexts already).

Validation context set (A, B, C): the selector should reproduce today's pool for B and C and at most add Stim48hr for A.
Test context set (D, E, F): decide per context on 2026-10-22, before the build.
How to test now: the proxy, where each held-out line plays the role of a context (H1 and Jurkat as the stem-like and T-like
cases), and one leaderboard upload with a T-cell-specific pool for A only.

## 3. Multivariate shrinkage (gene programs), in detail

Today each (target, gene) pair is estimated alone. The five lines give about 1 to 5 noisy values per pair, and shrinkage uses
only that pair's own spread. But effects across genes are correlated: knocking down a transcription factor moves a whole group
of genes in one direction, and the same group moves for related targets.

How it works:
1. For one line, put the effects in a matrix (targets x response genes). Estimate the gene-gene covariance of effects
   across targets, plus the measurement noise per entry (the `s` values we already have).
2. For a new target, treat its pooled profile as a noisy vector. The best estimate is `x_hat = Sigma (Sigma + N)^-1 x`: noise that
   is not shared with other genes is removed, and a gene measured badly is filled in from correlated genes. K562 misses 29%
   of the testable genes, so this also imputes them.
3. The covariance is learned from many targets. This is where the non-overlapping data matters: the essential screens add
   about 2,400 targets in three lines, and the genome-wide lines add many thousands more beyond the 575 we tabulated (the
   X-Atlas accumulators already hold all 18,000 targets for HCT116 and HEK293T). None of those targets needs to overlap the
   panel, because only the gene-gene covariance is learned from them.

How it differs from the low-rank test that failed: that test replaced every profile with its projection onto a hard-truncated
set of components, fitted on only 575 targets. This is a soft version, weighted by each gene's noise, with the covariance from
far more targets.

Main risk: essential-gene knockdowns share a strong common growth-arrest component, so the dominant programs may be the
generic response, which the scorer's baseline already gives for free. The fix is to remove the mean response across targets
(per line) before estimating the covariance, so only the target-specific programs remain.

Test: proxy, hold out one line, fit the covariance on the other lines (panel targets only, then with the essential screens
and the extra genome-wide targets), shrink the v3 profile, then standardise and scale as now.

## 4. Target-feature prior, in detail

The question you asked at the start was what to shrink towards. This answers it with the target's own properties instead
of zero or the global mean.

How it works: for every target we know things independent of any screen. Examples are its pathway and complex membership
(GO, Reactome, CORUM), its protein-interaction neighbours (STRING), its sequence or text embedding (ESM, GenePT), and the
effects of its nearest neighbours in the screens. The idea:
1. Learn a map from those features to the effect profile, using every target in every line, with no need for overlap with the panel.
   In its simplest form it is a nearest-neighbour average: the effect of a target is predicted by the average effect of the
   targets most similar to it (for example, other members of the same complex).
2. Use that prediction as the prior mean for the pooled estimate, in place of zero. With strong direct measurements, the
   pooled estimate dominates (inverse-variance weighting); where the lines disagree or are missing, the prior fills in.

So direct overlap still comes first, as you say; this only changes what the shrinkage pulls towards.

Why the essential screens matter here: they add thousands of labelled (target features, effect profile) examples, in
three cell types. Complex membership is a particularly informative feature, because members of one complex give similar profiles.

Cheap first test (no new data): for each panel target in a held-out line, find its nearest neighbours among all other targets
by profile similarity in the source lines, then average the neighbours' pooled profiles and blend with the target's own
profile. A gain here says the idea is worth building with external features (pathways, interactions).

Main risk: the panel targets are weak (median 5 DE genes in K562), so for most of them there is little to predict and the
prior mainly adds noise. The test above will tell quickly.

## What I would run next

1. KOLF2.1J finishes (running).
2. Idea 5 (knockdown-aware pooling): cheapest, uses tables we already have.
3. Idea 3 (multivariate shrinkage) with the nearest-neighbour test of idea 4 right after.
4. Leaderboard probes whenever you want an upload slot.
