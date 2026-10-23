# model: from effect tables to predicted cells

This stage is the prediction itself. It pools the Stage 1 tables of all source lines into one effect per (target, gene)
for a line that was never measured, adapts it to the new context, and writes 400 cells per target by changing the
context's own control cells. The "training" is a meta-analysis fit: there are no learned weights, only a few settings
chosen on held-out lines (`configs/*.toml`).

```
out/stage1_<line>.npz ──► pool.py (Stage 2) ──► effects.py ──► out/effects_<tag>.npz ──► generate.py ──► pred_<tag>.h5ad ──► package.py ──► <tag>.vcc
                          sources.py (same-line weights), promoter.py (neighbour cap)
```

`python -m vcell train | generate | build | package --config configs/final_def.toml`

## 1. Stage 2: random-effects pooling (`pool.py`)

Every line's measurement is modelled as `beta_l = theta + u_l + e_l`: the shared effect, the line's own deviation
(variance tau^2) and sampling noise (variance s_l^2). Per (target, gene):

```
w_l   = 1 / (s_l^2 + tau^2)                      weights
theta = sum w_l beta_l / sum w_l                 pooled effect
sd    = sqrt(1 / sum w_l + tau^2)                predictive sd for a NEW line
```

tau^2 is estimated by method of moments (DerSimonian-Laird), pooled over all pairs inside an expression bin of the
gene, because a few lines cannot give a stable variance per pair. Line weights (lam) act on the combination only, after
an unweighted tau^2: weighting by inflating s biases tau^2 low (found in H35). Pairs no line measured fall back to the
generic response (rung v1). The ladder v1 / v2 / v3 / v3b is kept: v3 is the model, v3b (a sparse prior) was rejected (H2).

## 2. The effect recipe (`effects.py`)

1. **Standardise** (`pool.z_effect`): `P = 1.5 x median(sd) x theta / sd`. Effects the lines agree on keep or gain size,
   inconsistent ones shrink. This beat raw theta in every held-out line (H11) and on the leaderboard (0.092 -> 0.160).
   The scale 1.5 is flat between 1.5 and 2 (H12, H26, H29).
2. **Expression bins** of the gene in the new context's controls: x1.5 below 15 CPM, x1.25 for 15-40, x1 up to 400, x0.8
   above (H36). Weakly expressed genes are shrunk most by the standardisation and need more scale to be called.
3. **Promoter neighbours** (`promoter.py`): CRISPRi represses the target's promoter, so genes whose TSS lies within 5 kb
   are capped at log2(0.15) at <= 500 bp, rising to no cap at 5 kb (H37).
4. **Floor**: effects below 0.02 log2 are left out.
5. **Per-context source weights** (`sources.py`, from `vcell.qc.identity`): a same-cell-type screen at weight 4 (H44,
   H47, H53); for the final entry, the U-87 MG table for context E.

Lines in the final pool: K562, CD4+ T, HCT116, HEK293T, H1 at 1, KOLF2.1J and Pan_H1 at 0.25 (H38, H40). The result is
an `Effects` object: per context, the emitted log2 effect of every panel target on the universe genes.

Rejected steps are not in this package; they live in `experiments/failed/` and plug in through the `post=` hook of
`effects.train` (e.g. removing the patterns all targets share, H51: better in the proxy, worse on the leaderboard).

## 3. Cell generation (`generate.py`)

For every context and target, 400 cells are drawn with replacement from that context's own control cells (keeping library
size, sparsity and gene-gene structure), then per gene with f = 2^effect: f < 1 thins each count binomially with p = f;
f > 1 adds Poisson counts with mean (f - 1) x the gene's mean x the cell's size factor. Counts stay integer, nothing is
renormalised, genes outside the universe are untouched. One random generator drives every draw in a fixed order, so a
build is reproducible bit for bit and a `--limit 3` run is a prefix of the full one. The 2-billion-value file is streamed
to disk target by target.

Fed the true effects, this generator scores 0.97 on the official H1 scorer against 1.07 for real cells (H42): emission
is not where the score is lost, so it stays simple and every cell is an honest i.i.d. draw.

## 4. Packaging (`package.py`)

A local check of the submission rules (`docs/SCORING.md` section 1: row count, gene order, integer non-negative counts,
cell totals, stored entries, no control rows), then `vcc prep` with `PYTHONIOENCODING=utf-8` and TEMP on D:. Submitting is
the user's step.

## Checks

`tests/regression/test_build_golden.py` and `test_release.py` require the build to reproduce the uploaded files
(`pred_final.h5ad`: all 360,000 cells); `tests/unit/` covers pooling (the worked example of the design doc), the step
order, the configs (no centring step may appear), emission statistics, determinism and the package checks.
