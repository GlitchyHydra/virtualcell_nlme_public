"""H53: is the same-line weight 4 (H44, H47) too high for a NOISY same-line source such as the U87 table for context E?
The rule was fitted with Pan_H1 for H1 held out (median s 0.09); the U87 table is about 7.5x noisier (sci-Plex, ~80 cells
per target). Pan_H1 is degraded to U87's noise, s' = k s and beta' = beta + N(0, s sqrt(k^2 - 1)), and its weight swept
with H1 held out (the other four lines at 1, KOLF 0.25; L5 settings). k = 1 repeats H44 on the 12,816-gene tables.

Verdict (EXPERIMENT_LOG H53): weight 4 still best. est on the 46 covered targets, w 0.25 / 1 / 2 / 4 / 8 (no Pan 0.217):
k 1 0.270 / 0.318 / 0.342 / 0.357 / 0.347; k 7.55 (U87 noise, 3 seeds) 0.220 / 0.219 / 0.221 / 0.223 / 0.216. The gain
shrinks with noise but never turns negative at w <= 4; U87 stays at 4 for E (configs/final_def.toml).
Was eda/69_noisy_sameline.py. Snapshot def. About 5 minutes.

    python experiments/h53_noisy_same_line.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.etl.stage1.table import load
from vcell.evaluation import proxy
from vcell.evaluation.harness import model_effect
from vcell.evaluation.panel import Panel
from vcell.model.promoter import neighbour_pairs

warnings.filterwarnings("ignore")
config.use_snapshot("def")
P = Panel.load().add("KOLF").add("Pan_H1")                      # indices 5 (KOLF) and 6 (Pan_H1)
P.min_effect = 0.02
B0, S0 = P.B[6].copy(), P.S[6].copy()
HI = 4                                                          # H1 held out
rows = P.rows_for(HI)
cov = np.array([i for i in rows if np.isfinite(B0[i]).sum() > 100])
u87 = load("U87")
ru = [i for i, t in enumerate(u87["targets"]) if t in set(config.panel_targets()) and np.isfinite(u87["beta"][i]).sum() > 100]
k_u87 = float(np.nanmedian(u87["s"][ru]) / np.nanmedian(S0[cov]))
pairs = neighbour_pairs(P.targets, P.genes)


def run(w: float) -> dict:
    tr = [0, 1, 2, 3, 5] + ([6] if w > 0 else [])
    R = P.rungs(tr, lam=[1.0] * 4 + [0.25] + ([w] if w > 0 else []))
    eff = model_effect(P, R, HI, pairs)
    out = {}
    for lab, r in [("all", rows), ("cov", cov)]:
        p, d = P.pds(eff, HI, r), P.de(eff, HI, r, per_target=True)
        out[f"est {lab}"] = round(proxy.est(p, d), 4)
        out[f"pds {lab}"] = round(p, 4)
    return out


if __name__ == "__main__":
    print(f"H1 held out: {len(rows)} benchmark targets, Pan_H1 covers {len(cov)}; median s Pan {np.nanmedian(S0[cov]):.3f}, "
          f"U87 {np.nanmedian(u87['s'][ru]):.3f} -> k_u87 {k_u87:.2f}", flush=True)
    res = []
    for k, seeds in [(1.0, [0]), (k_u87 / 2, [0, 1]), (k_u87, [0, 1, 2])]:
        for sd in seeds:
            rng = np.random.default_rng(sd)
            P.S[6] = S0 * k                                     # this experiment's own copy of the Pan_H1 arrays
            P.B[6] = B0 + (rng.normal(0, 1, B0.shape) * S0 * np.sqrt(max(k * k - 1, 0)) if k > 1 else 0)
            for w in [0, 0.25, 1, 2, 4, 8]:
                if w == 0 and k > 1:
                    continue
                r = {"k": round(k, 2), "seed": sd, "w": w, **run(w)}
                res.append(r)
                print(r, flush=True)
    df = pd.DataFrame(res)
    df.to_csv(config.results("h53_noisy_same_line.csv"), index=False)
    print("\n" + df.to_string(index=False))
    print("\nmean over seeds:\n" + df.groupby(["k", "w"])[["est all", "est cov", "pds cov"]].mean().round(4).to_string())
