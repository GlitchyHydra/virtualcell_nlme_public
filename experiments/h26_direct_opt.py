"""H26: direct score optimisation of a small family that generalises the standardised effect,

    eff = c * size_norm * theta / sd^b * h,   h = clip((cpm_new / median cpm_new)^gamma, 0.25, 4)

theta and sd: pooled effect and predictive sd from the training lines; cpm_new: the gene's control expression in the held-out
line; size_norm makes the median |effect| equal to that of z x1, so (b 1, c 1.5, gamma 0) is exactly the uploaded model and
b = 0 is raw theta. Objective: proxy.est, leave one line out, mean over the five main lines, searched on a grid in stages;
then checked on lines never used for the search (Jurkat, HepG2, RPE1, full essential axis).

Verdict (EXPERIMENT_LOG H26): the current model is already the optimum. Best b 1.25, c 1.5, gamma 0 at 0.1459 vs 0.1457;
the ridge is flat for b 0.75-1.25; raw theta loses 0.024; weighting by expression loses. On the essential lines the
reference beats the "best" (0.2017 vs 0.1996), so the standardisation and the 1.5 scale are robust, not tuned noise.
Was eda/35_direct_opt.py. Snapshot abc (main lines) + the essential axis. About 15 minutes, ~12 GB for the check.

    python experiments/h26_direct_opt.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.harness import essential_panel
from vcell.evaluation.panel import Panel

warnings.filterwarnings("ignore")
config.use_snapshot("abc")


def prepare(X, lines_idx):
    """Per held-out line: theta, sd, ref, cpm_new and rows (training lines = all the others)."""
    out = {}
    for hi in lines_idx:
        R = X.rungs([i for i in range(len(X.lines)) if i != hi])
        ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
        out[hi] = {"theta": R["v3"], "sd": R["sd"], "ref": ref, "cpm": X.ctrl_cpm(hi), "rows": X.rows_for(hi)}
    return out


def effect(P, b, c, gamma):
    th, sd, rows = P["theta"], P["sd"], P["rows"]
    base = th / sd ** b
    if gamma:
        cp = np.maximum(P["cpm"], 0.1)
        base = base * np.clip((cp / np.median(cp)) ** gamma, 0.25, 4.0)[None, :]
    z1 = P["ref"] * th / sd
    norm = np.nanmedian(np.abs(z1[rows])) / max(np.nanmedian(np.abs(base[rows])), 1e-12)
    return c * norm * base


def score(X, PRE, b, c, gamma):
    vals = {}
    for hi, P in PRE.items():
        e = effect(P, b, c, gamma)
        d = X.de(e, hi, P["rows"], per_target=True)
        vals[X.lines[hi]] = proxy.est(X.pds(e, hi, P["rows"]), d)
    return vals


if __name__ == "__main__":
    X = Panel.load()
    recs = []

    def run(PRE, b, c, gamma, stage):
        v = score(X, PRE, b, c, gamma)
        recs.append({"stage": stage, "b": b, "c": c, "gamma": gamma, **v, "mean": float(np.mean(list(v.values())))})
        print(f"{stage}: b {b:g} c {c:g} gamma {gamma:g} -> mean {recs[-1]['mean']:.4f}", flush=True)
        return recs[-1]["mean"]

    PRE = prepare(X, range(len(X.lines)))
    best = (1.0, 1.5, 0.0)
    best_s = run(PRE, *best, "reference")
    for b in (0.0, 0.5, 0.75, 1.25, 1.5, 2.0, 1.0):
        for c in (1.0, 1.5, 2.0):
            if (b, c) == (1.0, 1.5):
                continue
            s = run(PRE, b, c, 0.0, "1: b x c")
            if s > best_s:
                best, best_s = (b, c, 0.0), s
    for gamma in (-0.5, -0.25, 0.25, 0.5):
        s = run(PRE, best[0], best[1], gamma, "2: gamma")
        if s > best_s:
            best, best_s = (best[0], best[1], gamma), s
    bb, cc, gg = best
    for b in (bb - 0.125, bb + 0.125):
        for c in (cc - 0.25, cc + 0.25):
            s = run(PRE, b, c, gg, "3: refine")
            if s > best_s:
                best, best_s = (b, c, gg), s
    print(f"\nbest on the five lines: b {best[0]:g}, c {best[1]:g}, gamma {best[2]:g}: {best_s:.4f} "
          f"(reference {recs[0]['mean']:.4f})", flush=True)
    pd.DataFrame(recs).to_csv(config.results("h26_direct_opt.csv"), index=False)

    del PRE
    X = essential_panel(None)                                   # lines never used for the search
    PRE2 = prepare(X, [X.lines.index(n) for n in ("jurkat", "hepg2", "rpe1")])
    check = []
    for nm, par in (("reference", (1.0, 1.5, 0.0)), ("best", best), ("best b, c; gamma 0", (best[0], best[1], 0.0))):
        v = score(X, PRE2, *par)
        check.append({"setting": nm, "b": par[0], "c": par[1], "gamma": par[2], **v, "mean": float(np.mean(list(v.values())))})
    ck = pd.DataFrame(check)
    ck.to_csv(config.results("h26_direct_opt_check.csv"), index=False)
    print("\ncheck on held-out essential screens (never used for the search)\n" + ck.to_string(float_format=lambda x: f"{x:.4f}"))
