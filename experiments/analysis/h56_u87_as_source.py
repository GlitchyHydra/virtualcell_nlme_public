"""H56: are the U87 tables sound as ordinary training sources? Added at weight 1 (no same-cell-type boost) to the pool for
each held-out line (K562, CD4_Rest, HCT116, HEK293T, H1; final pool: the other four at 1, KOLF 0.25, Pan_H1 0.25, L5
settings): U87 = vehicle only (the uploaded entry F15), U87x = all drug arms with the drug effect removed (variant F18).
A check of the tables, not of the rule for E (that is H53 and H55).

Result (EXPERIMENT_LOG H56): neutral overall (mean est 0.1512 / 0.1509 / 0.1507 for none / U87 / U87x), slightly negative
on the 12-21 U87-covered targets (0.0864 / 0.0829 / 0.0764): a glioblastoma drug screen does not transfer to other cell
types, as H48 found for related lineages. U87x is still the variant for E.
Was eda/72_u87_as_source.py. Snapshot def. About 10 minutes (DE metrics over 3 noise seeds).

    python experiments/analysis/h56_u87_as_source.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.harness import model_effect
from vcell.evaluation.panel import Panel
from vcell.model.promoter import neighbour_pairs

warnings.filterwarnings("ignore")
config.use_snapshot("def")
P = Panel.load()
for name in ["KOLF", "Pan_H1", "U87", "U87x"]:                     # indices 5, 6, 7, 8
    P = P.add(name)
P.min_effect = 0.02
pairs = neighbour_pairs(P.targets, P.genes)
u87cov = np.isfinite(P.B[8]).sum(1) > 100
VAR = {"none": [], "U87 (vehicle)": [7], "U87x (all arms)": [8]}

if __name__ == "__main__":
    out = []
    for hi in range(5):
        rows_all = P.rows_for(hi)
        rows_cov = np.array([i for i in rows_all if u87cov[i]])
        for var, extra in VAR.items():
            tr = [i for i in range(5) if i != hi] + [5, 6] + extra
            R = P.rungs(tr, lam=[1.0] * 4 + [0.25, 0.25] + [1.0] * len(extra))
            eff = model_effect(P, R, hi, pairs)
            for lab, rows in [("all", rows_all), ("U87 targets", rows_cov)]:
                p = P.pds(eff, hi, rows)
                ds = [P.de(eff, hi, rows, seed=s, per_target=True) for s in range(3)]
                raw = {k: float(np.mean([np.nanmean(d[k]) for d in ds])) for k in ["fid", "jac", "reach", "nmae"]}
                r = {"held out": P.lines[hi], "targets": lab, "n": len(rows), "source": var, "pds": p, **raw,
                     "est": float(np.mean([proxy.est(p, d) for d in ds]))}
                out.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

    df = pd.DataFrame(out)
    df.to_csv(config.results("h56_u87_as_source.csv"), index=False)
    pd.set_option("display.width", 200)
    for lab in ["all", "U87 targets"]:
        d = df[df.targets == lab]
        print(f"\n=== {lab} targets: raw metrics per held-out line ===")
        print(d.drop(columns="targets").round(4).to_string(index=False))
        print(f"\n=== {lab} targets: mean over the 5 held-out lines; difference vs none ===")
        m = d.groupby("source", sort=False)[["pds", "fid", "jac", "reach", "nmae", "est"]].mean()
        print(m.round(4).to_string())
        print((m - m.loc["none"]).round(4).to_string())
