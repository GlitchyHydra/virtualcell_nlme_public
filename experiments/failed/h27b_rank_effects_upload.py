"""H27b (ranking learner for the A/B/C generator, not uploaded): Per-context effects with the ranking learner (docs/EXPERIMENT_LOG.md H27), for the A/B/C generator.

1. Base effects exactly as uploaded: Stage 2 v3 on the source lines, standardised x scale (stage2.z_effect).
2. A LightGBM LambdaRank model learns which genes of a target respond in a new line: one query per (line, target), items =
   the target's testable genes, relevance = grade of |beta / s| (0: < 1, 1: 1-2, 2: 2-3, 3: 3-5, 4: > 5). Training is
   stacked over the source lines: each line j in turn supplies labels, features come from the other lines plus j's control
   expression (experiments/failed/gbm_features.py).
3. Per context (A, B, C; later D, E, F): features from all source lines plus that context's control expression; the
   learner's rank and the pool's |z| rank are mixed (weight w for the learner) over the context's testable genes (> 5 CPM),
   and each target's own effect sizes are handed out in that order; signs stay the pool's.
Output: OUT/experiments/effects_<tag>.npz with one (targets x genes) effect matrix per context, read by vcell.model --effects.

    python experiments/failed/h27b_rank_effects_upload.py --weight 0.15 --tag rank15

Verdict (EXPERIMENT_LOG H27): the per-context effects of the rank ensemble (weight 0.15) were packaged as rank15.vcc but
not uploaded: the light mixes only tied the pool in the proxy. Building cells from them used the old
`vcell.model --effects` option, which left the pipeline with this experiment (tag final-def-build has it).
Was eda/41_rank_effects.py. Snapshot abc (the A/B/C validation tables). About an hour.
"""
from __future__ import annotations

import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

from vcell import config, io
from vcell.evaluation.panel import Panel

from gbm_features import Features
from vcell.model import pool

config.use_snapshot("abc")
X = Panel.load()
GF = Features(X)
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--weight", type=float, default=0.15, help="share of the learner's rank in the mixed rank")
ap.add_argument("--scale", type=float, default=1.5)
ap.add_argument("--contexts", nargs="+", default=None, help="default: the manifest's contexts")
ap.add_argument("--genes-per-target", type=int, default=2000)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tag", default="rank15")
args = ap.parse_args()
rng = np.random.default_rng(0)
EDGES = [1, 2, 3, 5]
SRC = list(range(len(X.lines)))                                  # all five main source lines
contexts = args.contexts or config.manifest()["contexts"]
t0 = time.time()

# 1. base effects, exactly as the uploaded generator computes them
fit = pool.predict(X.B, X.S, X.max_cpm)
base = pool.z_effect(fit["v3"]["pred"], fit["v3"]["sd"], fit["v3"]["K"] >= 1, args.scale)

# 2. ranking learner, stacked over the source lines
Xs, Ys, groups = [], [], []
for j in SRC:
    f, _ = GF.features([i for i in SRC if i != j], j)
    rows_j = X.rows_for(j) if X.lines[j] != "H1" else np.flatnonzero(np.isfinite(X.B[j]).sum(1) > 100)
    cols_all = np.flatnonzero(X.ctrl_cpm(j) > 5)
    for t in rows_j:
        y = X.B[j, t, cols_all] / X.S[j, t, cols_all]
        ok = cols_all[np.isfinite(y)]
        if len(ok) < 100:
            continue
        cols = np.sort(rng.choice(ok, min(len(ok), args.genes_per_target), replace=False))
        M, names = GF.to_matrix(f, [t], cols)
        Xs.append(M)
        Ys.append(np.digitize(np.abs(X.B[j, t, cols] / X.S[j, t, cols]), EDGES))
        groups.append(len(cols))
    del f
    print(f"features for line {X.lines[j]} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
model = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                       subsample=0.5, subsample_freq=1, colsample_bytree=0.8, lambdarank_truncation_level=200,
                       label_gain=[0, 1, 3, 7, 15], n_jobs=args.threads, verbose=-1)
model.fit(np.concatenate(Xs), np.concatenate(Ys), group=groups)
del Xs, Ys
imp = model.booster_.feature_importance("gain")
print("trained; top features " + ", ".join(f"{a} {b:.2f}" for a, b in sorted(zip(names, imp / imp.sum()), key=lambda x: -x[1])[:6]),
      flush=True)


def context_cpm(ctx):
    """Mean CPM (full-library normalisation) of the universe genes in a context's control cells; NaN if absent."""
    X0, _, gn = io.load_context(ctx)
    lib = np.asarray(X0.sum(1)).ravel()
    cpm = np.asarray(X0.multiply(1e6 / np.maximum(lib, 1)[:, None]).mean(0)).ravel()
    pos = {g: i for i, g in enumerate(gn)}
    return np.array([cpm[pos[g]] if g in pos else np.nan for g in X.genes])


def reassign(ref, score):
    out = np.zeros_like(ref)
    for i in range(ref.shape[0]):
        vals = np.sort(np.abs(ref[i]))[::-1]
        a = np.empty_like(vals)
        a[np.argsort(-score[i])] = vals
        out[i] = np.sign(ref[i]) * a
    return out


# 3. per-context effects
panel = set(config.panel_targets())
rows = np.array([i for i, t in enumerate(X.targets) if t in panel])
out = {"targets": np.array(X.targets), "genes": np.array(X.genes), "contexts": np.array(contexts)}
for ctx in contexts:
    cp = context_cpm(ctx)
    f, _ = GF.features(SRC, 0, ctx_cpm=cp)                        # ctx index unused when ctx_cpm is given
    M, _ = GF.to_matrix(f, rows, np.arange(GF.G))
    s = model.predict(M).reshape(len(rows), GF.G)
    del M, f
    test = np.nan_to_num(cp) > 5
    mix = args.weight * rankdata(s, axis=1) + (1 - args.weight) * rankdata(np.abs(np.nan_to_num(base[rows])), axis=1)
    mix = np.where(test[None, :], mix, -np.inf)                    # untestable genes keep the smallest sizes
    eff = base.copy()
    eff[rows] = reassign(np.nan_to_num(base[rows]), mix)
    moved = np.mean(np.abs(eff[rows] - np.nan_to_num(base[rows])) > 1e-9)
    print(f"context {ctx}: {test.sum()} testable genes; share of (target, gene) effects that changed {moved:.3f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    out[ctx] = eff.astype(np.float32)
path = config.results(f"h27b_rank_effects_upload_effects_{args.tag}.npz")
np.savez_compressed(path, **out)
print(f"saved {path}")
