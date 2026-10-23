# evaluation: how a change is judged before it is uploaded

The final leaderboard is hidden and the validation leaderboard allows a few uploads a day, so almost every decision was
made offline. The rule: never judge a change on the data it was tuned on. Each source line takes its turn as the "new"
line: the model is fitted on the other lines and scored against the held-out line's real effects. Three tools, from
seconds to half an hour:

| Tool | What it scores | Cost |
|---|---|---|
| `panel.py` + `proxy.py` | five of the six metrics, leave one line out, from Stage 1 tables (no cells generated) | seconds |
| `harness.py` | the same on 8 held-out lines (5 main + Jurkat, HepG2, RPE1 on the essential axis) | ~1 min per setting |
| `h1_benchmark.py` | the official scorer (cell-eval2) on the public H1 benchmark, H1 held out | ~30 min |

`python -m vcell evaluate proxy | harness | h1 --config configs/<run>.toml`

## 1. The proxies (`panel.py`, `proxy.py`)

A `Panel` stacks the tables of its lines (B, S, NC: lines x targets x genes). For held-out line h:

- **pds**: cosine of each predicted effect with every real effect of h, weighted as the scorer's pseudobulk space;
  1 - rank of the correct target / (n - 1).
- **DE metrics**: the real side is h's estimate brought to 400 cells; ours is the predicted effect (after the floor) plus
  the noise of 400 cells; Benjamini-Hochberg on both. fid = our significant genes with the right sign (real-significant or
  not) / max(n_ours, n_real), as `cell_eval2/metrics/direction.py` does; Jaccard; reach (deepest prefix of the real
  significant genes, ordered by our confidence, with sign purity >= 0.9); nmae of the fold changes.
- **mse** (`mse_proxy.py`): noise-corrected pseudobulk error ratio; 1 = predicting no change.
- `proxy.est` puts pds and the DE part on the leaderboard scale and averages the DE part over an A/B/C-like mix of target
  strengths (the held-out lines have many more DE genes per target than the panel).

Validated on H1 against the scorer (raw fid / jac / reach / nmae for v3: 0.492 / 0.092 / 0.144 / 1.051 vs 0.484 / 0.091 /
0.122 / 1.036). It ranks variants the way the scorer does and understates the gains (+0.019 proxy, +0.045 H1, +0.068
leaderboard for the standardisation). Only differences between variants are meaningful, not levels.

## 2. The harness (`harness.py`)

`Harness().objective(c=1.5, mult=[1.5, 1.25, 1, 1, 0.8], neighbour=0.15, floor=0.02, w_kolf=0.25, w_pan=0.25)` returns
proxy.est per held-out line; this is the final model. Three essential-gene lines that were never used for tuning guard
against overfitting the five main ones. `sweep()` runs a list of settings against the first, `model_effect()` applies the
final recipe to any pool, `cases()` lets an experiment plug in its own effect function. The noise of the harness is
about +-0.003, so smaller gains are not trusted.

## 3. The official H1 scorer (`h1_benchmark.py`)

The public stand-in benchmark (126 H1 targets x 400 cells, `vcc-h1`, cell-eval2 0.16.0). The model is fitted without
H1, cells are generated from H1 controls, and `vcc-h1 score` gives all six metrics rescaled between the control baseline
and a real replicate. `run(cfg)` generates, scores and deletes the 3.5 GB file. `oracle_effects()` feeds a table's own
effects (the generator ceiling, H42). One draw has a seed-to-seed spread of about 0.01.

## What it taught

Every estimator that was best for squared error lost on the scorer (sparse prior, boosting, stacking), and the two changes
that won were chosen by the score itself (standardisation, same-line weight). The proxies made it cheap to find that out:
about 50 ideas were tested and most were rejected (`experiments/README.md`).

## Checks

`tests/regression/test_proxy_golden.py` (proxy values of F10, the harness objective, the mse validation) and
`test_h1_golden.py` (the E4 file bit for bit; with `-m release` its official score 0.130); `tests/unit/test_proxy_metrics.py`
checks the metrics on synthetic panels with known answers.
