"""H27 (ranking learner, not adopted): Gradient boosting with a ranking loss (LambdaRank): can a non-linear learner rank WHICH genes of a target respond
better than the pool does?

Training (stacked inside the training lines, as failed/h25_gbm.py): one query per target, items = its testable genes (2,000 sampled
per target), relevance = grade of the real |beta / s| in the inner held-out line (0: < 1, 1: 1-2, 2: 2-3, 3: 3-5, 4: > 5).
Features: experiments/failed/gbm_features.py.
Use: the pool keeps its signs and each target's set of effect sizes; only which gene gets which size changes.
    rank reassign      genes ordered by the learner's score receive the target's |effect| values in descending order
    rank ensemble      the same, ordered by the mean of the learner's rank and the pool's |z| rank
Diagnostic: precision of the top 50 genes per target against the real |beta / s| > 3 genes, learner vs pool.

    python experiments/failed/h27_rank_gbm.py [--lines K562 HCT116 H1]

Verdict (EXPERIMENT_LOG H27): better gene identity (precision of the top 50 genes 0.184 vs 0.132 for the pool) but the
score falls when its order reassigns the pool's sizes (0.137 vs 0.159); a light rank ensemble only ties (0.159).
Was eda/36_rank_gbm.py. Snapshot abc (the A/B/C validation tables). About an hour per line.
"""
from __future__ import annotations

import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel

from gbm_features import Features

config.use_snapshot("abc")
X = Panel.load()
GF = Features(X)
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=["K562", "HCT116", "H1"])
ap.add_argument("--genes-per-target", type=int, default=2000)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
rng = np.random.default_rng(0)
EDGES = [1, 2, 3, 5]


def reassign(ref, score):
    """Per row: the row's |ref| values, sorted, given to genes in descending order of score; signs from ref."""
    out = np.zeros_like(ref)
    for i in range(ref.shape[0]):
        vals = np.sort(np.abs(ref[i]))[::-1]
        order = np.argsort(-score[i])
        a = np.empty_like(vals)
        a[order] = vals
        out[i] = np.sign(ref[i]) * a
    return out


def top_precision(score, truth_snr, k=50):
    p = []
    for i in range(score.shape[0]):
        ok = np.isfinite(truth_snr[i])
        if (np.abs(truth_snr[i][ok]) > 3).sum() < 5:
            continue
        top = np.argsort(-np.where(ok, score[i], -np.inf))[:k]
        p.append(float((np.abs(truth_snr[i][top]) > 3).mean()))
    return float(np.mean(p))


recs = []
for name in args.lines:
    t0 = time.time()
    hi = X.lines.index(name)
    tr = [i for i in range(len(X.lines)) if i != hi]
    Xs, Ys, Ysign, groups = [], [], [], []
    for j in tr:
        f, _ = GF.features([i for i in tr if i != j], j)
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
            Ysign.append((X.B[j, t, cols] > 0).astype(int))
            groups.append(len(cols))
        del f
    Xt, yt = np.concatenate(Xs), np.concatenate(Ys)
    del Xs
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                           subsample=0.5, subsample_freq=1, colsample_bytree=0.8, lambdarank_truncation_level=200,
                           label_gain=[0, 1, 3, 7, 15], n_jobs=args.threads, verbose=-1)
    model.fit(Xt, yt, group=groups)
    imp = model.booster_.feature_importance("gain")
    ys = np.concatenate(Ysign)
    conf = yt >= 2                                                              # |beta / s| > 2 in the inner held-out line
    signm = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=63,
                               min_child_samples=200, subsample=0.5, subsample_freq=1, colsample_bytree=0.8,
                               n_jobs=args.threads, verbose=-1).fit(Xt[conf], ys[conf])
    del Xt

    f, zs = GF.features(tr, hi)
    rows = X.rows_for(hi)
    M, _ = GF.to_matrix(f, rows, np.arange(GF.G))
    s = model.predict(M).reshape(len(rows), GF.G)
    psgn = signm.predict_proba(M)[:, 1].reshape(len(rows), GF.G)                # P(effect > 0) from the sign classifier
    del M, f
    ref_eff = 1.5 * zs
    test = X.ctrl_cpm(hi) > 5
    s_t = np.where(test[None, :], s, -np.inf)                                  # rank only testable genes first
    zr = np.where(test[None, :], np.abs(zs[rows]), -np.inf)
    ens = np.where(test[None, :], rankdata(s, axis=1) + rankdata(np.abs(zs[rows]), axis=1), -np.inf)
    variants = {"z x1.5 (reference)": ref_eff}
    learned_sign = np.where(psgn >= 0.5, 1.0, -1.0)
    sure = np.abs(2 * psgn - 1)
    for vn, sc in (("rank reassign", s_t), ("rank ensemble", ens)):
        e = ref_eff.copy()
        e[rows] = reassign(ref_eff[rows], sc)
        variants[vn] = e
        e2 = ref_eff.copy()                                                     # sign from the classifier
        e2[rows] = np.abs(e[rows]) * learned_sign
        variants[f"{vn}, learned sign"] = e2
        e3 = ref_eff.copy()                                                     # and size damped by its confidence
        e3[rows] = e2[rows] * sure
        e3[rows] *= np.nanmedian(np.abs(ref_eff[rows])) / max(np.nanmedian(np.abs(e3[rows])), 1e-12)
        variants[f"{vn}, learned sign x confidence"] = e3
    rs, rz = rankdata(s, axis=1), rankdata(np.abs(zs[rows]), axis=1)
    for w in (0.15, 0.3):                                                       # lighter rank ensembles
        sc = np.where(test[None, :], w * rs + (1 - w) * rz, -np.inf)
        e = ref_eff.copy()
        e[rows] = reassign(ref_eff[rows], sc)
        variants[f"rank ensemble w {w}"] = e
    q = (rs - 1) / (GF.G - 1)                                                   # learner percentile per target, 0 to 1
    for a in (0.5, 1.0):                                                        # keep pool effects, scale by learner rank
        e = ref_eff.copy()
        e[rows] = ref_eff[rows] * np.clip(1 + a * (2 * q - 1), 0, None)
        e[rows] *= np.nanmedian(np.abs(ref_eff[rows])) / max(np.nanmedian(np.abs(e[rows])), 1e-12)
        variants[f"boost a {a}"] = e
    e4 = ref_eff.copy()                                                         # pool sizes and ranks, learned sign only
    e4[rows] = np.abs(ref_eff[rows]) * learned_sign
    variants["pool sizes, learned sign"] = e4
    truth_snr = X.B[hi][rows] / X.S[hi][rows]
    truth_snr = np.where(test[None, :], truth_snr, np.nan)
    real = np.isfinite(truth_snr) & (np.abs(truth_snr) > 3)
    top_l = np.zeros_like(real)
    np.put_along_axis(top_l, np.argsort(-s_t, axis=1)[:, :50], True, axis=1)
    sel = real & top_l
    print(f"   sign accuracy on the ranker's top-50 genes that are really responding: pool "
          f"{(np.sign(ref_eff[rows][sel]) == np.sign(truth_snr[sel])).mean():.3f}, classifier "
          f"{(learned_sign[sel] == np.sign(truth_snr[sel])).mean():.3f}; on all real pairs: pool "
          f"{(np.sign(ref_eff[rows][real]) == np.sign(truth_snr[real])).mean():.3f}, classifier "
          f"{(learned_sign[real] == np.sign(truth_snr[real])).mean():.3f}", flush=True)
    prec = {"z x1.5 (reference)": top_precision(zr, truth_snr), "rank reassign": top_precision(s_t, truth_snr),
            "rank ensemble": top_precision(ens, truth_snr)}
    for vn, e in variants.items():
        d = X.de(e, hi, rows, per_target=True)
        p = X.pds(e, hi, rows)
        recs.append({"held_out": name, "variant": vn, "pds": p, "de_mix": proxy.mixed(d), "est": proxy.est(p, d),
                     "fid": d["fid"].mean(), "jac": d["jac"].mean(), "reach": np.nanmean(d["reach"]),
                     "nmae": np.nanmean(d["nmae"]), "prec_top50": prec.get(vn.split(",")[0], np.nan)})
    top = sorted(zip(names, imp / imp.sum()), key=lambda x: -x[1])[:6]
    print(f"{name} done in {(time.time() - t0) / 60:.1f} min; top features " + ", ".join(f"{a} {b:.2f}" for a, b in top), flush=True)
    pd.DataFrame(recs).to_csv(config.results("h27_rank_gbm.csv"), index=False)

df = pd.DataFrame(recs)
for col in ["est", "pds", "de_mix", "fid", "jac", "reach", "nmae", "prec_top50"]:
    pv = df.pivot_table(index="variant", columns="held_out", values=col, sort=False)
    pv["mean"] = pv.mean(1)
    print(f"\n{col}\n" + pv.to_string(float_format=lambda x: f"{x:.3f}"))
