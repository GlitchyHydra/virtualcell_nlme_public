"""H25 (not supported): A non-linear learner on top of the pool (gradient boosting, LightGBM), leave one line out.

Question: is there non-linear signal the linear random-effects pool misses? For a held-out line h, features for every
(target, gene) pair come from the training lines only, plus the new line's own control cells (as for D/E/F). The model is
trained by stacking inside the training lines: for each training line j, features from the other training lines, label = j's
real effect. Then it predicts h.

Features (per target t, gene g):
    pool      theta (v3), predictive sd, z = theta / sd, number of lines K, generic gene mean (v1)
    lines     mean, min, max of the per-line effects, sign agreement, largest |beta / s|
    context   log CPM of g in the new line's controls, its mean over the source lines, and the difference
    target    strength (count of |z| > 3), log CPM of t in the new line, mean knockdown over the source lines
    borrowing gene neighbours: mean z of the 20 genes whose effects correlate most with g across targets (same target);
              target neighbours: mean z of the 10 targets whose profiles correlate most with t (same gene)
Objectives (the scorer is not MSE, so three are compared):
    l2      regression on beta, weighted by 1 / (s^2 + 0.01)
    huber   regression on beta / s (signal to noise, clipped at +-10)
    sign    classification of sign(beta) on pairs with |beta / s| > 2, weight min(|beta / s|, 10); prediction 2p - 1
Every prediction is resized to the median |effect| of z x1.5 before scoring, so only its pattern is compared.

    python experiments/failed/h25_gbm.py [--lines K562 HCT116 ...]    # held-out lines (default all five)

Verdict (EXPERIMENT_LOG H25): the learner alone 0.052-0.072 vs 0.146, 50/50 blends 0.107-0.116; on plain accuracy it is as
good as the pool (0.048 vs 0.044), so the loss is the mismatch between a least-squares fit and the rank, sign and
significance metrics. Dropped. Features: experiments/failed/gbm_features.py.
Was eda/33_gbm.py. Snapshot abc (the A/B/C validation tables). Hours and ~20 GB (a run was stopped by the low-memory reaper); run one line at a time with --lines.
"""
from __future__ import annotations

import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel

from gbm_features import Features

config.use_snapshot("abc")
X = Panel.load()
GF = Features(X)
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--lines", nargs="+", default=X.lines)
ap.add_argument("--train-per-fold", type=int, default=800_000)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--drop", nargs="*", default=[], help="feature names to leave out")
ap.add_argument("--tag", default="", help="suffix of the output csv names")
ap.add_argument("--max-train-targets", type=int, default=None,
                help="learning curve: use only this many random targets per inner training fold")
args = ap.parse_args()
rng = np.random.default_rng(0)
G = GF.G


def features(L, ctx):
    return GF.features(L, ctx, args.drop)


FEATS = None


def to_matrix(f, rows, cols):
    global FEATS
    M, FEATS = GF.to_matrix(f, rows, cols)
    return M


def pds_per_target(pred, hi, rows):
    """X.pds without the mean: the rank score of each target."""
    sw = np.sqrt(X.wts(hi))[X.not_target]
    P = np.nan_to_num(pred[rows][:, X.not_target]) * sw
    Y = np.nan_to_num(X.B[hi][rows][:, X.not_target]) * sw
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    sim = P @ Y.T
    d = np.diag(sim)[:, None]
    rank = (sim > d).sum(1) + 0.5 * ((sim == d).sum(1) - 1)
    return 1 - rank / (len(rows) - 1)


def de_target_score(d):
    """Per-target DE part on the leaderboard scale, as in proxy.mixed but without the strength mix."""
    return ((d["fid"] - 0.5) / proxy.SPAN["fid"] + d["jac"] / proxy.SPAN["jac"] + np.nan_to_num(d["reach"]) / proxy.SPAN["reach"]
            + (np.nan_to_num(d["nmae"], nan=1.0) - 1.0) / proxy.SPAN["nmae"])


recs, imps, conc = [], [], []
for name in args.lines:
    t0 = time.time()
    hi = X.lines.index(name)
    tr = [i for i in range(len(X.lines)) if i != hi]
    Xs, Y, Sy = [], [], []
    for j in tr:                                                             # stacking inside the training lines
        inner = [i for i in tr if i != j]
        f, _ = features(inner, j)
        rows_j = X.rows_for(j) if X.lines[j] != "H1" else np.flatnonzero(np.isfinite(X.B[j]).sum(1) > 100)
        if args.max_train_targets and len(rows_j) > args.max_train_targets:
            rows_j = np.sort(rng.choice(rows_j, args.max_train_targets, replace=False))
        cols_j = np.flatnonzero(X.ctrl_cpm(j) > 5)
        M = to_matrix(f, rows_j, cols_j)
        y = X.B[j][np.ix_(rows_j, cols_j)].ravel()
        s = X.S[j][np.ix_(rows_j, cols_j)].ravel()
        ok = np.flatnonzero(np.isfinite(y) & np.isfinite(s) & np.isfinite(M[:, 0]))
        ok = rng.choice(ok, min(len(ok), args.train_per_fold), replace=False)
        Xs.append(M[ok]); Y.append(y[ok]); Sy.append(s[ok])
        del M
    Xt, y, s = np.concatenate(Xs), np.concatenate(Y), np.concatenate(Sy)
    del Xs
    snr = np.clip(y / s, -10, 10)
    base = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=500, subsample=0.5,
                subsample_freq=1, colsample_bytree=0.8, n_jobs=args.threads, verbose=-1)
    models = {"l2": lgb.LGBMRegressor(objective="l2", **base).fit(Xt, y, sample_weight=1 / (s ** 2 + 0.01)),
              "huber": lgb.LGBMRegressor(objective="huber", alpha=2.0, **base).fit(Xt, snr)}
    conf = np.abs(snr) > 2
    models["sign"] = lgb.LGBMClassifier(objective="binary", **base).fit(
        Xt[conf], (y[conf] > 0).astype(int), sample_weight=np.minimum(np.abs(snr[conf]), 10))
    for mn, m in models.items():
        imp = m.booster_.feature_importance("gain")
        imps.append({"held_out": name, "model": mn, **dict(zip(FEATS, imp / imp.sum()))})
    del Xt

    f, zs = features(tr, hi)
    rows = X.rows_for(hi)
    allc = np.arange(G)
    M = to_matrix(f, rows, allc)
    ref_eff = 1.5 * zs
    size = np.nanmedian(np.abs(ref_eff[rows]))
    preds = {"l2": models["l2"].predict(M), "huber": models["huber"].predict(M),
             "sign": 2 * models["sign"].predict_proba(M)[:, 1] - 1}
    test = X.ctrl_cpm(hi) > 5
    truth = X.B[hi][rows]
    tm = np.isfinite(truth) & test[None, :]

    def corr(e):
        a, b = e[rows][tm], truth[tm]
        return float(np.corrcoef(a, b)[0, 1])

    variants = {"z x1.5 (reference)": ref_eff}
    for pn, p in preds.items():
        e = np.zeros_like(ref_eff)
        e[rows] = p.reshape(len(rows), G)
        e *= size / max(np.nanmedian(np.abs(e[rows])), 1e-9)
        variants[f"gbm {pn}"] = e
        variants[f"0.5 z + 0.5 gbm {pn}"] = 0.5 * ref_eff + 0.5 * e
        for tau in (0.1, 0.2):                       # hybrid: the pool where it is clear, the learner where the pool is near 0
            variants[f"hybrid {pn} |pool| < {tau}"] = np.where(np.abs(ref_eff) >= tau, ref_eff, e)
    for vn, e in variants.items():
        d = X.de(e, hi, rows, per_target=True)
        p = X.pds(e, hi, rows)
        recs.append({"held_out": name, "variant": vn, "pds": p, "de_mix": proxy.mixed(d), "est": proxy.est(p, d),
                     "fid": d["fid"].mean(), "jac": d["jac"].mean(), "nmae": np.nanmean(d["nmae"]), "corr": corr(e)})

    # Concordance: are the two models right on different pairs and targets?
    zref = ref_eff[rows]
    tsnr = truth / X.S[hi][rows]
    conf_pairs = tm & (np.abs(tsnr) > 3)                                   # confidently non-zero real effects
    s_true = np.sign(truth[conf_pairs])
    a_ok = np.sign(zref[conf_pairs]) == s_true
    pds_ref = pds_per_target(ref_eff, hi, rows)
    dz = X.de(ref_eff, hi, rows, per_target=True)
    for pn in preds:
        e = variants[f"gbm {pn}"]
        b_ok = np.sign(e[rows][conf_pairs]) == s_true
        pds_g = pds_per_target(e, hi, rows)
        dg = X.de(e, hi, rows, per_target=True)
        sc_ref = de_target_score(dz) + (pds_ref - 0.5) / proxy.SPAN["pds"]
        sc_g = de_target_score(dg) + (pds_g - 0.5) / proxy.SPAN["pds"]
        absz = np.abs(zref[conf_pairs])
        zbin = pd.cut(pd.Series(absz), [0, 0.5, 1, 2, 4, np.inf])
        win = pd.Series(b_ok & ~a_ok).groupby(zbin, observed=False).mean()
        loss = pd.Series(a_ok & ~b_ok).groupby(zbin, observed=False).mean()
        conc.append({"held_out": name, "model": pn, "n_conf_pairs": int(conf_pairs.sum()),
                     "both_right": float((a_ok & b_ok).mean()), "only_pool": float((a_ok & ~b_ok).mean()),
                     "only_gbm": float((~a_ok & b_ok).mean()), "both_wrong": float((~a_ok & ~b_ok).mean()),
                     "pds_target_corr": float(np.corrcoef(pds_ref, pds_g)[0, 1]),
                     "targets_gbm_better": float((sc_g > sc_ref).mean()),
                     "oracle_target_gain": float((np.maximum(sc_g, sc_ref) - sc_ref).mean() / 6),
                     **{f"gbm_wins_|z|{k}": float(v) for k, v in win.items()},
                     **{f"pool_wins_|z|{k}": float(v) for k, v in loss.items()}})
    pd.DataFrame(conc).to_csv(config.results(f"h25_gbm_gbm_concordance{args.tag}.csv"), index=False)
    print(f"{name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    pd.DataFrame(recs).to_csv(config.results(f"h25_gbm_gbm{args.tag}.csv"), index=False)

df = pd.DataFrame(recs)
for col in ["est", "pds", "de_mix", "fid", "jac", "nmae", "corr"]:
    pv = df.pivot_table(index="variant", columns="held_out", values=col, sort=False)
    pv = pv[[c for c in X.lines if c in pv.columns]]
    pv["mean"] = pv.mean(1)
    print(f"\n{col}\n" + pv.to_string(float_format=lambda x: f"{x:.3f}"))
print("\nconcordance (sign on confident real pairs |beta/s| > 3; per-target score = pds + DE parts, leaderboard scale)")
print(pd.DataFrame(conc).set_index(["held_out", "model"]).T.to_string(float_format=lambda x: f"{x:.3f}"))
im = pd.DataFrame(imps).groupby("model")[FEATS].mean().T
im.to_csv(config.results(f"h25_gbm_gbm_importance{args.tag}.csv"))
print("\nfeature importance (share of gain, mean over held-out lines)\n" + im.to_string(float_format=lambda x: f"{x:.3f}"))
