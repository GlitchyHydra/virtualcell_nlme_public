"""H29: tune the uploaded model on all 8 held-out lines (the harness: 5 main lines + Jurkat, HepG2, RPE1 on 600 random
essential targets): scale c, weight kappa of tau^2 in the standardising sd, effect floor, KOLF weight. Search in stages
(c x kappa, then floor, then KOLF weight, then a local refine); objective the mean of proxy.est over the 8 lines.

Verdict (EXPERIMENT_LOG H29): gains within noise. Best c 1.875, kappa 2, floor 0.02, KOLF 0.5 at 0.1698 vs 0.1676
(+0.002); the noise level is about +-0.003. Floor 0.02 was kept (in the final model); the larger scale was probed on the
leaderboard (L3).
Note: KOLF enters here by inflating its standard errors (Panel.rungs(weights=...)), which biases tau^2 low; H35 found
this and the harness now weights the combination only (lam). The KOLF weight was rechecked that way in H38.
Was eda/38_tune_all.py. Snapshot abc. About 15 minutes.

    python experiments/h29_tune_all.py
"""
import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation.harness import Harness, KI
from vcell.evaluation.panel import standardised

config.use_snapshot("abc")
H = Harness()
CACHE = {}


def pooled(axis, hi, kappa, w_kolf):
    """The standardised pool with KOLF weighted through its precision (the H29 method)."""
    key = (axis, hi, kappa, w_kolf)
    if key not in CACHE:
        if axis == "main":
            tr = [i for i in range(5) if i != hi] + ([KI] if w_kolf > 0 else [])
            w = [1.0] * (len(tr) - (1 if w_kolf > 0 else 0)) + ([w_kolf] if w_kolf > 0 else [])
        else:
            tr, w = [i for i in range(len(H.ess.lines)) if i != hi], None
        CACHE[key] = standardised(H.states[axis][0].rungs(tr, weights=w, kappa=kappa))
    return CACHE[key]


def objective(c, kappa, floor, w_kolf):
    return {P.lines[hi]: H.score(P, c * pooled(axis, hi, kappa, w_kolf), hi, rows, floor) for axis, P, hi, rows, _ in H.cases()}


recs = []


def run(c, kappa, floor, w_kolf, stage):
    v = objective(c, kappa, floor, w_kolf)
    main = float(np.mean([v[n] for n in H.main.lines[:5]]))
    ess = float(np.mean([v[n] for n in ("jurkat", "hepg2", "rpe1")]))
    m = float(np.mean(list(v.values())))
    recs.append({"stage": stage, "c": c, "kappa": kappa, "floor": floor, "w_kolf": w_kolf, **v,
                 "mean_main5": main, "mean_ess3": ess, "mean_all8": m})
    print(f"{stage}: c {c:g} kappa {kappa:g} floor {floor:g} w_kolf {w_kolf:g} -> all8 {m:.4f} "
          f"(main5 {main:.4f}, ess3 {ess:.4f})", flush=True)
    pd.DataFrame(recs).to_csv(config.results("h29_tune_all.csv"), index=False)
    return m


if __name__ == "__main__":
    best = (1.5, 1.0, 0.05, 0.0)
    best_s = run(*best, "reference")
    for kappa in (0.5, 1.0, 2.0):
        for c in (1.25, 1.5, 1.75):
            if (c, kappa) == (1.5, 1.0):
                continue
            s = run(c, kappa, 0.05, 0.0, "1: c x kappa")
            if s > best_s:
                best, best_s = (c, kappa, 0.05, 0.0), s
    for floor in (0.02, 0.1):
        s = run(best[0], best[1], floor, 0.0, "2: floor")
        if s > best_s:
            best, best_s = (best[0], best[1], floor, 0.0), s
    for w in (0.25, 0.5, 1.0):
        s = run(best[0], best[1], best[2], w, "3: KOLF weight")
        if s > best_s:
            best, best_s = (best[0], best[1], best[2], w), s
    c0, k0, f0, w0 = best
    for c, kappa in ((c0 - 0.125, k0), (c0 + 0.125, k0), (c0, k0 * 0.75), (c0, k0 * 1.5)):
        s = run(c, kappa, f0, w0, "4: refine")
        if s > best_s:
            best, best_s = (c, kappa, f0, w0), s
    ref = recs[0]
    print(f"\nbest: c {best[0]:g}, kappa {best[1]:g}, floor {best[2]:g}, KOLF weight {best[3]:g}: all8 {best_s:.4f} "
          f"vs reference {ref['mean_all8']:.4f} (main5 {ref['mean_main5']:.4f}, ess3 {ref['mean_ess3']:.4f})")
